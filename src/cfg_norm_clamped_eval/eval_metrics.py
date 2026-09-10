"""
Comprehensive image quality evaluation for generative models.

Metrics:
  - FID  (Frechet Inception Distance) — distribution-level quality
  - sFID (spatial FID)                — FID on spatial Inception features (texture/structure)
  - KID  (Kernel Inception Distance)  — unbiased FID alternative
  - IS   (Inception Score)            — clarity + class diversity (ImageNet only)
  - CMMD (CLIP-MMD)                   — modern FID alternative in CLIP space
  - CLIP-Score (ViT-L/14)             — text-image semantic alignment
  - ImageReward                      — human-preference alignment (T2I)

Usage:
    # ImageNet class-conditional generation (SiT)
    python -m cfg_norm_clamped_eval.eval_metrics --gen-dir fid_outputs --ref-dir /path/to/imagenet_val

    # Text-to-image generation (SD3.5), single prompt
    python -m cfg_norm_clamped_eval.eval_metrics --gen-dir results_banana/01_constant/cfg4.0 \
        --prompt "A photograph of banana with white background"

    # Text-to-image with per-image prompt file (one prompt per line)
    python -m cfg_norm_clamped_eval.eval_metrics --gen-dir outputs --prompt-file prompts.txt

    # All metrics with custom batch settings
    python -m cfg_norm_clamped_eval.eval_metrics --gen-dir outputs --ref-dir ref --prompt "a cat" \
        --batch-size 32 --num-workers 8
"""

import os

# ————————————————————————————————————————————
# Single-GPU enforcement: must happen BEFORE any torch/cuda import.
# Set the visible devices *before* any library touches CUDA.
# ————————————————————————————————————————————
_SINGLE_GPU = os.environ.get("EVAL_GPU", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", _SINGLE_GPU)
# Disable DataParallel / DistributedDataParallel in downstream libs
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("RANK", "0")

import argparse
import hashlib
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# CLIP model registry — used by both CMMD and CLIP-Score
DEFAULT_CLIP_MODEL = "openai/clip-vit-large-patch14"

CLIP_MODELS = {
    "vitl14": "openai/clip-vit-large-patch14",
    "vitb32": "openai/clip-vit-base-patch32",
}


# ============================================================
# FID & KID via clean-fid
# ============================================================


def compute_fid(
    gen_dir: str,
    ref_dir: str,
    device: str = "cuda",
    batch_size: int = 64,
    num_workers: int = 8,
    mode: str = "clean",
) -> float:
    """FID between generated and reference image directories."""
    from cleanfid import fid

    return fid.compute_fid(
        fdir1=gen_dir,
        fdir2=ref_dir,
        mode=mode,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        verbose=True,
    )


def compute_kid(
    gen_dir: str, ref_dir: str, device: str = "cuda", batch_size: int = 64, num_workers: int = 8
) -> float:
    """KID (Kernel Inception Distance) — unbiased alternative to FID."""
    from cleanfid import fid

    return fid.compute_kid(
        fdir1=gen_dir,
        fdir2=ref_dir,
        mode="clean",
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        verbose=True,
    )


# ============================================================
# sFID (spatial FID)
# ============================================================


@torch.no_grad()
def compute_sfid(
    gen_dir: str, ref_dir: str, device: str = "cuda", batch_size: int = 64, num_workers: int = 8
) -> float:
    """
    sFID (spatial FID) — treats Inception v3 block-2 spatial features
    as independent samples.

    Uses InceptionV3 block 2 (Mixed_5b → Mixed_6e, 768-d, 17×17) — matches
    ``mixed_6/conv:0`` in the TF graph used by the ADM evaluator.

    Each spatial location is treated as an independent 768-d sample, giving
    N*289 total samples.  An online (Welford/Chan) algorithm accumulates the
    mean and covariance without materialising the full feature matrix.

    References:
        - "Are GANs Created Equal? A Large-Scale Study" (Lucic et al., 2018)
        - ADM (guided-diffusion) evaluator, lines 24, 586-600
    """
    from cleanfid.inception_pytorch import InceptionV3
    from scipy import linalg

    D = 768  # block-2 feature dim

    model = InceptionV3(output_blocks=[2], resize_input=False, normalize_input=True).to(device)
    model.eval()

    def _online_stats(image_dir: str) -> tuple[np.ndarray, np.ndarray]:
        from torchvision import transforms

        img_files = sorted(Path(image_dir).glob("*"))
        img_files = [
            f for f in img_files if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        ]
        assert len(img_files) > 0, f"No images found in {image_dir}"

        tf = transforms.Compose(
            [
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
            ]
        )

        n = 0
        mean = np.zeros(D, dtype=np.float64)
        M2 = np.zeros((D, D), dtype=np.float64)

        for i in tqdm(range(0, len(img_files), batch_size), desc=f"sFID: {Path(image_dir).name}"):
            batch_files = img_files[i : i + batch_size]
            batch = torch.stack([tf(Image.open(f).convert("RGB")) for f in batch_files]).to(device)
            feats = model(batch)[0]  # (B, 768, 17, 17)
            B, C, H, W = feats.shape
            feats_np = feats.permute(0, 2, 3, 1).reshape(B * H * W, C).cpu().numpy()  # (B*289, 768)

            n_batch = feats_np.shape[0]
            b_mean = feats_np.mean(axis=0, dtype=np.float64)
            b_M2 = (feats_np - b_mean).T @ (feats_np - b_mean)

            # Chan parallel update (outer product!)
            n_prev = n
            n += n_batch
            delta = b_mean - mean
            mean += delta * n_batch / n
            M2 += b_M2 + np.outer(delta, delta) * (n_prev * n_batch / n)

        return mean, M2 / (n - 1) if n > 1 else M2

    mu1, sigma1 = _online_stats(gen_dir)
    mu2, sigma2 = _online_stats(ref_dir)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        covmean = linalg.sqrtm((sigma1 + np.eye(D) * 1e-6).dot(sigma2 + np.eye(D) * 1e-6))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


# ============================================================
# CMMD (CLIP Maximum Mean Discrepancy)
# ============================================================


@torch.no_grad()
def _extract_clip_features(
    image_dir: str, device: str, batch_size: int, num_workers: int, clip_model_id: str
) -> np.ndarray:
    """Extract CLIP image embeddings for all images in a directory."""
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(clip_model_id).to(device).eval()
    processor = CLIPProcessor.from_pretrained(clip_model_id)

    img_files = sorted(Path(image_dir).glob("*"))
    img_files = [
        f for f in img_files if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    ]

    all_features = []
    for i in tqdm(range(0, len(img_files), batch_size), desc="CMMD: extracting CLIP features"):
        batch_files = img_files[i : i + batch_size]
        batch_imgs = [Image.open(f).convert("RGB") for f in batch_files]
        inputs = processor(images=batch_imgs, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model.get_image_features(**inputs)
        # Handle both raw tensor and BaseModelOutputWithPooling
        if hasattr(outputs, "pooler_output"):
            img_embeds = outputs.pooler_output
        elif hasattr(outputs, "image_embeds"):
            img_embeds = outputs.image_embeds
        else:
            img_embeds = outputs
        # Normalize
        img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
        all_features.append(img_embeds.detach().cpu().numpy())

    return np.concatenate(all_features, axis=0)


def compute_cmmd(
    gen_dir: str,
    ref_dir: str,
    device: str = "cuda",
    batch_size: int = 64,
    num_workers: int = 8,
    clip_model_id: str = DEFAULT_CLIP_MODEL,
) -> float:
    """CMMD — CLIP-space Maximum Mean Discrepancy.

    Modern alternative to FID, used in SD3/SD3.5 evaluation.
    Lower = better. Uses RBF kernel with median heuristic for bandwidth.

    Args:
        gen_dir: generated images folder
        ref_dir: reference images folder
        clip_model_id: CLIP model for feature extraction (default: ViT-L/14)
    """
    gen_feats = _extract_clip_features(gen_dir, device, batch_size, num_workers, clip_model_id)
    ref_feats = _extract_clip_features(ref_dir, device, batch_size, num_workers, clip_model_id)

    # RBF MMD with median-distance bandwidth
    def rbf_mmd(x: np.ndarray, y: np.ndarray) -> float:
        # Pairwise sq distances
        xx = np.dot(x, x.T)
        yy = np.dot(y, y.T)
        xy = np.dot(x, y.T)

        x_norm = np.diag(xx)
        y_norm = np.diag(yy)

        d_xx = x_norm[:, None] + x_norm[None, :] - 2 * xx
        d_yy = y_norm[:, None] + y_norm[None, :] - 2 * yy
        d_xy = x_norm[:, None] + y_norm[None, :] - 2 * xy

        # Median heuristic for sigma
        all_dists = np.concatenate(
            [
                d_xx[np.triu_indices_from(d_xx, k=1)],
                d_yy[np.triu_indices_from(d_yy, k=1)],
                d_xy.flatten(),
            ]
        )
        sigma2 = np.median(all_dists[all_dists > 0])
        if sigma2 == 0:
            sigma2 = 1.0
        gamma = 1.0 / (2.0 * sigma2)

        k_xx = np.exp(-gamma * d_xx)
        k_yy = np.exp(-gamma * d_yy)
        k_xy = np.exp(-gamma * d_xy)

        n = x.shape[0]
        m = y.shape[0]

        # Unbiased estimate
        mmd2 = (
            (np.sum(k_xx) - np.trace(k_xx)) / (n * (n - 1))
            + (np.sum(k_yy) - np.trace(k_yy)) / (m * (m - 1))
            - 2 * np.sum(k_xy) / (n * m)
        )
        return float(max(0, mmd2))

    return rbf_mmd(gen_feats, ref_feats)


# ============================================================
# Inception Score (IS)
# ============================================================


def _load_inception_for_is(device: str) -> torch.nn.Module:
    """Inception v3 with classifier head — for IS (needs 1000-class softmax)."""
    from torchvision.models import Inception_V3_Weights, inception_v3

    model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
    model.to(device)
    model.eval()
    return model


def _inception_transform(image: Image.Image) -> torch.Tensor:
    """Inception v3 preprocessing: resize to 299x299, normalize to [-1,1]."""
    from torchvision import transforms

    tf = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    return tf(image)


@torch.no_grad()
def compute_is(
    image_dir: str, device: str = "cuda", batch_size: int = 64, num_splits: int = 10
) -> dict:
    """
    Inception Score (IS).

    IS = exp(E_x[KL(p(y|x) || p(y))])

    Uses Inception v3's 1000-way classifier output as p(y|x).
    Higher IS = images are both clear (low per-sample entropy) and
    diverse (high marginal entropy across samples).

    Typical values: ImageNet real ~200-300, good generative ~10-50.

    Returns {"IS_mean": float, "IS_std": float}
    """
    model = _load_inception_for_is(device)

    img_files = sorted(Path(image_dir).glob("*"))
    img_files = [
        f for f in img_files if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    ]
    assert len(img_files) > 0, f"No images found in {image_dir}"

    all_probs = []
    for i in tqdm(range(0, len(img_files), batch_size), desc="IS: extracting features"):
        batch_files = img_files[i : i + batch_size]
        batch_imgs = [_inception_transform(Image.open(f).convert("RGB")) for f in batch_files]
        batch = torch.stack(batch_imgs).to(device)

        # Inception v3 returns (logits, aux_logits) in train mode, just logits in eval
        logits = model(batch)  # (B, 1000)
        probs = torch.softmax(logits, dim=1)
        all_probs.append(probs.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)  # (N, 1000)

    # Split-based IS for robustness (standard practice from OpenAI)
    N = len(all_probs)
    split_scores = []
    indices = np.arange(N)
    np.random.seed(42)
    for _ in range(num_splits):
        np.random.shuffle(indices)
        for k in range(num_splits):
            split_idx = indices[k * N // num_splits : (k + 1) * N // num_splits]
            p_yx = all_probs[split_idx]  # (N_s, 1000)
            p_y = p_yx.mean(axis=0, keepdims=True)  # (1, 1000) marginal
            kl = p_yx * (np.log(p_yx + 1e-10) - np.log(p_y + 1e-10))
            kl_mean = kl.sum(axis=1).mean()
            split_scores.append(np.exp(kl_mean))

    return {"IS_mean": float(np.mean(split_scores)), "IS_std": float(np.std(split_scores))}


# ============================================================
# CLIP-Score (CLIP-T): text-image semantic alignment
# ============================================================


def _load_clip(device: str, model_id: str = DEFAULT_CLIP_MODEL):
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(model_id)
    model.to(device)
    model.eval()
    processor = CLIPProcessor.from_pretrained(model_id)
    return model, processor


@torch.no_grad()
def compute_clip_score(
    image_dir: str,
    prompts: list[str],
    device: str = "cuda",
    batch_size: int = 32,
    clip_model: str = DEFAULT_CLIP_MODEL,
) -> dict:
    """
    CLIP-Score: cosine similarity between image and text CLIP embeddings.

    For T2I generation. Higher = better prompt-image alignment.
    Default: CLIP ViT-L/14 (~0.20–0.35 range).

    Args:
        image_dir: folder of generated images
        prompts: list of prompts, one per image (sorted by filename)
                 or a single prompt applied to all images
        clip_model: HuggingFace model ID (default: ViT-L/14)

    Returns {"CLIP-Score_mean": float, "CLIP-Score_std": float}
    """
    model, processor = _load_clip(device, clip_model)

    img_files = sorted(Path(image_dir).glob("*"))
    img_files = [
        f for f in img_files if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    ]
    assert len(img_files) > 0, f"No images found in {image_dir}"

    # Expand single prompt to all images
    if len(prompts) == 1:
        prompts = prompts * len(img_files)
    assert len(prompts) == len(img_files), (
        f"Prompt count ({len(prompts)}) != image count ({len(img_files)})"
    )

    all_scores = []
    for i in tqdm(range(0, len(img_files), batch_size), desc="CLIP-Score"):
        batch_files = img_files[i : i + batch_size]
        batch_prompts = prompts[i : i + batch_size]
        batch_imgs = [Image.open(f).convert("RGB") for f in batch_files]

        inputs = processor(
            text=batch_prompts,
            images=batch_imgs,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)
        img_embeds = outputs.image_embeds  # (B, D)
        text_embeds = outputs.text_embeds  # (B, D)

        # Normalize and compute cosine similarity
        img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        scores = (img_embeds * text_embeds).sum(dim=-1)  # cosine similarity
        all_scores.append(scores.cpu().numpy())

    all_scores = np.concatenate(all_scores)
    return {"CLIP-Score_mean": float(all_scores.mean()), "CLIP-Score_std": float(all_scores.std())}


# ============================================================
# ImageReward: human-preference alignment for T2I
# ============================================================


def compute_image_reward(
    image_dir: str, prompts: list[str], device: str = "cuda", batch_size: int = 32
) -> dict:
    """
    ImageReward: human-preference score for text-to-image generation.

    Higher = better alignment with human aesthetic/preference judgments.
    Typical range: -2 to 3 (~0 = average, >1 = good).

    Returns {"ImageReward_mean": float, "ImageReward_std": float}
    """
    from .eval_image_reward import compute_image_reward as _ir_fn

    return _ir_fn(image_dir, prompts, device=device, batch_size=batch_size)


# ============================================================
# Result saver
# ============================================================


def _image_count(image_dir: str | None) -> int | None:
    if not image_dir or not Path(image_dir).is_dir():
        return None
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    return sum(p.suffix.lower() in extensions for p in Path(image_dir).iterdir())


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _file_sha256(path: str | None) -> str | None:
    if not path or not Path(path).is_file():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _generation_config(gen_dir: str) -> dict | None:
    config_path = Path(gen_dir) / "run_config.json"
    if not config_path.is_file():
        return None
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def build_metadata(args) -> dict:
    """Describe the complete evaluation protocol needed to compare scores."""
    return {
        "protocol": args.protocol,
        "generated_dir": args.gen_dir,
        "generated_image_count": _image_count(args.gen_dir),
        "generation_config": _generation_config(args.gen_dir),
        "reference_dir": args.ref_dir,
        "reference_image_count": _image_count(args.ref_dir),
        "prompt_file": args.prompt_file,
        "prompt_file_sha256": _file_sha256(args.prompt_file),
        "fid": {
            "implementation": "clean-fid",
            "mode": args.fid_mode,
            "feature_extractor": "Inception-v3 pool3",
        },
        "clip": {
            "model": CLIP_MODELS[args.clip_model],
            "definition": "mean raw cosine similarity (no clipping or scaling)",
        },
        "image_reward": {"model": "ImageReward-v1.0"},
        "packages": {
            name: _package_version(name)
            for name in ("clean-fid", "torch", "torchvision", "transformers", "image-reward")
        },
    }


def save_results(results: dict, output_path: str, args):
    """Save a readable summary and a machine-readable JSON sidecar."""
    metadata = build_metadata(args)
    with open(output_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("Image Quality Evaluation Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Protocol: {args.protocol}\n")
        f.write(f"Generated: {args.gen_dir}\n")
        if args.ref_dir:
            f.write(f"Reference: {args.ref_dir}\n")
        if args.prompt_file:
            f.write(f"Prompt file: {args.prompt_file}\n")
            f.write(f"Prompt SHA256: {metadata['prompt_file_sha256']}\n")
        elif args.prompt:
            f.write(f"Prompt: {args.prompt[:80]}\n")
        f.write(f"Generated images: {metadata['generated_image_count']}\n")
        if args.ref_dir:
            f.write(f"Reference images: {metadata['reference_image_count']}\n")
        f.write(f"FID implementation: clean-fid ({args.fid_mode})\n")
        f.write(f"CLIP definition: {metadata['clip']['model']} raw cosine\n")
        f.write("\n")

        if not results:
            f.write("(no metrics computed)\n")
        else:
            f.write(f"{'Metric':<25s} {'Value':>12s}\n")
            f.write("-" * 38 + "\n")
            for k, v in results.items():
                if isinstance(v, float):
                    f.write(f"{k:<25s} {v:>12.6f}\n")
                else:
                    f.write(f"{k:<25s} {str(v):>12s}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("Generated by: python -m cfg_norm_clamped_eval.eval_metrics\n")
        f.write(
            f"Timestamp: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )

    json_path = str(Path(output_path).with_suffix(".json"))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": results, "metadata": metadata}, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"\n✓ Results saved to {output_path} and {json_path}")


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive image quality evaluation for generative models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--gen-dir", type=str, required=True, help="Directory of generated images")
    parser.add_argument(
        "--ref-dir",
        type=str,
        default=None,
        help="Directory of reference/real images (for FID, KID, CMMD)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single text prompt for all images (for CLIP-Score)",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="File with one prompt per line, aligned with sorted image filenames",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="Single GPU index to use (sets CUDA_VISIBLE_DEVICES + torch.set_device)",
    )
    parser.add_argument("--skip-fid", action="store_true", help="Skip FID computation")
    parser.add_argument(
        "--skip-sfid", action="store_true", help="Skip sFID (spatial FID) computation"
    )
    parser.add_argument("--skip-is", action="store_true", help="Skip IS computation")
    parser.add_argument("--skip-kid", action="store_true", help="Skip KID computation")
    parser.add_argument("--skip-cmmd", action="store_true", help="Skip CMMD computation")
    parser.add_argument("--skip-clip", action="store_true", help="Skip CLIP-Score computation")
    parser.add_argument(
        "--skip-image-reward", "--skip-ir", action="store_true", help="Skip ImageReward computation"
    )
    parser.add_argument(
        "--save-results",
        "--save",
        type=str,
        default=None,
        metavar="FILE",
        help="Save results to a txt file",
    )
    parser.add_argument(
        "--clip-model",
        type=str,
        default="vitl14",
        choices=["vitl14", "vitb32"],
        help="CLIP model used for explicitly labelled raw cosine similarity",
    )
    parser.add_argument(
        "--fid-mode",
        type=str,
        default="clean",
        choices=["clean", "legacy_tensorflow", "legacy_pytorch"],
        help="clean-fid protocol; use clean for the primary modern benchmark",
    )
    parser.add_argument(
        "--protocol",
        type=str,
        default="custom",
        help="Explicit benchmark protocol name stored with the results",
    )

    args = parser.parse_args()

    # Restrict to a single GPU
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)  # use the first (and only) visible device

    # Validate gen_dir
    if not os.path.isdir(args.gen_dir):
        print(f"Error: gen-dir not found: {args.gen_dir}")
        sys.exit(1)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Generated images: {args.gen_dir}")
    print(f"Reference images: {args.ref_dir or '(not provided)'}")

    results = {}

    # --- FID ---
    if not args.skip_fid:
        if args.ref_dir is None:
            print("\n⚠  Skipping FID: --ref-dir not provided")
        else:
            print("\n" + "=" * 60)
            print("FID (Frechet Inception Distance)")
            print("=" * 60)
            fid_val = compute_fid(
                args.gen_dir, args.ref_dir, device, args.batch_size, args.num_workers, args.fid_mode
            )
            fid_name = "Clean-FID" if args.fid_mode == "clean" else f"FID-{args.fid_mode}"
            results[fid_name] = fid_val
            print(f"{fid_name} = {fid_val:.4f}")

    # --- sFID ---
    if not args.skip_sfid:
        if args.ref_dir is None:
            print("\n⚠  Skipping sFID: --ref-dir not provided")
        else:
            print("\n" + "=" * 60)
            print("sFID (spatial FID)")
            print("=" * 60)
            sfid_val = compute_sfid(
                args.gen_dir, args.ref_dir, device, args.batch_size, args.num_workers
            )
            results["sFID"] = sfid_val
            print(f"sFID = {sfid_val:.4f}")

    # --- KID ---
    if not args.skip_kid:
        if args.ref_dir is None:
            print("\n⚠  Skipping KID: --ref-dir not provided")
        else:
            print("\n" + "=" * 60)
            print("KID (Kernel Inception Distance)")
            print("=" * 60)
            kid_val = compute_kid(
                args.gen_dir, args.ref_dir, device, args.batch_size, args.num_workers
            )
            results["KID"] = kid_val
            print(f"KID = {kid_val:.6f} (×1000 = {kid_val * 1000:.4f})")

    # --- CMMD ---
    if not args.skip_cmmd:
        if args.ref_dir is None:
            print("\n⚠  Skipping CMMD: --ref-dir not provided")
        else:
            print("\n" + "=" * 60)
            print("CMMD (CLIP Maximum Mean Discrepancy)")
            print("=" * 60)
            clip_model_id = CLIP_MODELS[args.clip_model]
            cmmd_val = compute_cmmd(
                args.gen_dir,
                args.ref_dir,
                device,
                args.batch_size,
                args.num_workers,
                clip_model_id=clip_model_id,
            )
            results["CMMD"] = cmmd_val
            print(f"CMMD = {cmmd_val:.4f}")

    # --- IS (for ImageNet class-conditional generation; reference-only for T2I) ---
    if not args.skip_is:
        print("\n" + "=" * 60)
        print("IS (Inception Score) — ImageNet class-conditional metric")
        if args.prompt or args.prompt_file:
            print("⚠  IS measures ImageNet class diversity — not text-alignment.")
            print("   For T2I evaluation, CLIP-Score + FID/CMDD are the primary metrics.")
        print("=" * 60)
        is_result = compute_is(args.gen_dir, device, args.batch_size)
        results["IS_mean"] = is_result["IS_mean"]
        results["IS_std"] = is_result["IS_std"]
        print(f"IS = {is_result['IS_mean']:.4f} ± {is_result['IS_std']:.4f}")

    # --- CLIP-Score ---
    if not args.skip_clip:
        if args.prompt is None and args.prompt_file is None:
            print("\n⚠  Skipping CLIP-Score: --prompt or --prompt-file required")
        else:
            if args.prompt_file:
                with open(args.prompt_file) as f:
                    prompts = [line.strip() for line in f if line.strip()]
            else:
                prompts = [args.prompt]

            clip_model_id = CLIP_MODELS[args.clip_model]
            print("\n" + "=" * 60)
            print(f"CLIP-Score (Text-Image Alignment) — {args.clip_model}")
            print("=" * 60)
            print(f"Model: {clip_model_id}")
            print(f"Prompts: {len(prompts)} unique prompt(s)")
            clip_result = compute_clip_score(
                args.gen_dir, prompts, device, args.batch_size, clip_model=clip_model_id
            )
            clip_prefix = f"CLIP-{args.clip_model}-raw-cosine"
            results[f"{clip_prefix}_mean"] = clip_result["CLIP-Score_mean"]
            results[f"{clip_prefix}_std"] = clip_result["CLIP-Score_std"]
            print(
                f"{clip_prefix} = {clip_result['CLIP-Score_mean']:.4f} ± {clip_result['CLIP-Score_std']:.4f}"
            )

    # --- ImageReward ---
    if not args.skip_image_reward:
        if args.prompt is None and args.prompt_file is None:
            print("\n⚠  Skipping ImageReward: --prompt or --prompt-file required")
        else:
            if args.prompt_file:
                with open(args.prompt_file) as f:
                    ir_prompts = [line.strip() for line in f if line.strip()]
            else:
                ir_prompts = [args.prompt]

            print("\n" + "=" * 60)
            print("ImageReward (Human-Preference Alignment)")
            print("=" * 60)
            ir_result = compute_image_reward(args.gen_dir, ir_prompts, device, args.batch_size)
            results["ImageReward-v1.0_mean"] = ir_result["ImageReward_mean"]
            results["ImageReward-v1.0_std"] = ir_result["ImageReward_std"]
            print(
                f"ImageReward = {ir_result['ImageReward_mean']:.4f} ± {ir_result['ImageReward_std']:.4f}"
            )

    # --- Save results ---
    if args.save_results:
        save_results(results, args.save_results, args)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k:<20s} = {v:.4f}")
        else:
            print(f"  {k:<20s} = {v}")
    print()


if __name__ == "__main__":
    main()
