"""
Generate images from a pre-trained SiT model and compute FID against ImageNet validation set.

Usage:
    python -m cfg_norm_clamped_eval.eval_fid                          # use defaults
    python -m cfg_norm_clamped_eval.eval_fid --num-samples 5000 --batch-size 16
    python -m cfg_norm_clamped_eval.eval_fid --compute-fid-only        # skip generation, only compute FID
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .sit_sampling import add_generation_arguments, generate_images

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _image_files(image_dir):
    files = sorted(
        path
        for path in Path(image_dir).rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        raise ValueError(f"No images found in {image_dir}")
    return files


def compute_saturation(image_dir):
    """Mean HSV saturation, averaged over pixels and then images.

    This follows the color metric used by Sadat et al. (APG, ICLR 2025).
    RGB and HSV channels are represented in [0, 1], so the result is also
    in [0, 1]. Lower is not universally better; compare against real data.
    """
    per_image = []
    for path in tqdm(_image_files(image_dir), desc="Saturation"):
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        value = rgb.max(axis=-1)
        chroma = value - rgb.min(axis=-1)
        saturation = np.divide(chroma, value, out=np.zeros_like(chroma), where=value > 0)
        per_image.append(float(saturation.mean()))
    return float(np.mean(per_image))


@torch.no_grad()
def compute_diversity_recall(
    gen_dir, ref_dir, device, batch_size=64, num_workers=8, k=3, distance_chunk_size=1024
):
    """Improved-precision/recall Recall as a distribution diversity score.

    In Clean-FID Inception feature space, each generated feature gets a
    hypersphere whose radius reaches its k-th nearest generated neighbour.
    Recall is the fraction of real features covered by the union of those
    spheres. This is the diversity/coverage metric used in CFG evaluations.
    """
    if k < 1:
        raise ValueError("diversity k must be at least 1")
    if distance_chunk_size < 1:
        raise ValueError("diversity chunk size must be at least 1")

    from cleanfid import fid
    from cleanfid.features import build_feature_extractor

    feature_model = build_feature_extractor("clean", device, use_dataparallel=False)
    common = dict(
        model=feature_model,
        num_workers=num_workers,
        batch_size=batch_size,
        device=device,
        mode="clean",
        verbose=True,
    )
    gen = fid.get_folder_features(gen_dir, description="Diversity: generated features", **common)
    real = fid.get_folder_features(ref_dir, description="Diversity: reference features", **common)
    if len(gen) <= k:
        raise ValueError(f"Diversity requires more than k={k} generated images; found {len(gen)}")

    gen = torch.from_numpy(gen).float().to(device)
    real = torch.from_numpy(real).float().to(device)
    chunk = distance_chunk_size

    # topk includes the query point itself at zero, hence k + 1.
    radii = []
    for start in tqdm(range(0, len(gen), chunk), desc="Diversity: k-NN radii"):
        query = gen[start : start + chunk]
        nearest = torch.full((len(query), k + 1), float("inf"), device=device)
        for ref_start in range(0, len(gen), chunk):
            distances = torch.cdist(query, gen[ref_start : ref_start + chunk])
            nearest = torch.topk(
                torch.cat((nearest, distances), dim=1),
                k=k + 1,
                dim=1,
                largest=False,
            ).values
        radii.append(nearest[:, -1].cpu())
    radii = torch.cat(radii).to(device)

    covered_count = 0
    for start in tqdm(range(0, len(real), chunk), desc="Diversity: real coverage"):
        query = real[start : start + chunk]
        covered = torch.zeros(len(query), dtype=torch.bool, device=device)
        for gen_start in range(0, len(gen), chunk):
            distances = torch.cdist(query, gen[gen_start : gen_start + chunk])
            covered |= (distances <= radii[gen_start : gen_start + chunk]).any(dim=1)
            if covered.all():
                break
        covered_count += int(covered.sum())

    return covered_count / len(real)


def main(args):
    if not args.compute_fid_only:
        generate_images(args)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    # ---- compute FID ----
    print("\nComputing FID...")
    print(f"  Generated: {args.out_dir}")
    print(f"  Reference:  {args.ref_dir}")

    from cleanfid import fid

    score = fid.compute_fid(
        fdir1=args.out_dir,
        fdir2=args.ref_dir,
        mode="clean",
        num_workers=args.num_workers,
        batch_size=args.inception_batch_size,
        device=device,
        verbose=True,
    )
    print(f"\nFID = {score:.4f}")

    if not args.skip_saturation:
        saturation = compute_saturation(args.out_dir)
        print(f"Saturation = {saturation:.6f}")

    if not args.skip_diversity:
        diversity = compute_diversity_recall(
            args.out_dir,
            args.ref_dir,
            device=device,
            batch_size=args.inception_batch_size,
            num_workers=args.num_workers,
            k=args.diversity_k,
            distance_chunk_size=args.diversity_chunk_size,
        )
        print(f"Diversity (Recall@k={args.diversity_k}) = {diversity:.6f}")


def cli():
    """Parse command-line arguments and run ImageNet generation/evaluation."""
    parser = argparse.ArgumentParser(description="Compute FID for SiT generated images")

    add_generation_arguments(parser)

    # FID
    parser.add_argument("--ref-dir", type=str, default="data/imagenet_val")
    parser.add_argument(
        "--compute-fid-only",
        action="store_true",
        help="Skip generation, only compute FID from existing out_dir",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--inception-batch-size", type=int, default=64)
    parser.add_argument(
        "--skip-saturation", action="store_true", help="Skip mean HSV saturation computation"
    )
    parser.add_argument(
        "--skip-diversity",
        action="store_true",
        help="Skip improved-precision/recall diversity (Recall)",
    )
    parser.add_argument(
        "--diversity-k",
        type=int,
        default=3,
        help="k-NN manifold size for diversity Recall (standard: 3)",
    )
    parser.add_argument(
        "--diversity-chunk-size",
        type=int,
        default=1024,
        help="Pairwise-distance chunk size; lower it if GPU memory is insufficient",
    )

    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli()
