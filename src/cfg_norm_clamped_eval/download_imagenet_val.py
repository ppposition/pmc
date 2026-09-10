"""
Download ImageNet-1K validation set from HuggingFace via parquet files.

Output: 50,000 JPEG images in a flat directory, named by original ImageNet filename.

Usage:
    uv run python -m cfg_norm_clamped_eval.download_imagenet_val                         # full 50K at original size
    uv run python -m cfg_norm_clamped_eval.download_imagenet_val --size 256              # resize to 256x256
    uv run python -m cfg_norm_clamped_eval.download_imagenet_val --max-images 5000       # test with 5K
"""

import argparse
import io
import os
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image
from tqdm import tqdm


def main():
    p = argparse.ArgumentParser(description="Download ImageNet-1K validation set")
    p.add_argument("--out-dir", default="data/imagenet_val", help="Output directory for images")
    p.add_argument(
        "--size",
        type=int,
        default=None,
        help="Resize to square (e.g. 256 for 256x256). Default: keep original.",
    )
    p.add_argument(
        "--max-images", type=int, default=None, help="Max images to download (for testing)"
    )
    p.add_argument(
        "--keep-parquet", action="store_true", help="Keep downloaded parquet files after extraction"
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(out_dir.glob("*.JPEG")))
    if args.max_images is None and existing >= 50000:
        print(f"✓ {out_dir} already has {existing} images, skipping")
        return

    # List validation parquet files
    repo_files = list_repo_files("imagenet-1k", repo_type="dataset")
    val_parquets = sorted(f for f in repo_files if f.startswith("data/validation-"))
    print(f"Found {len(val_parquets)} validation parquet files")

    total = 0
    for parquet_path in val_parquets:
        if args.max_images and total >= args.max_images:
            break

        print(f"\nProcessing {parquet_path.split('/')[-1]} ...")
        local_path = hf_hub_download(
            "imagenet-1k",
            repo_type="dataset",
            filename=parquet_path,
        )

        table = pq.read_table(local_path)
        n_rows = table.num_rows
        for i in tqdm(range(n_rows), desc="Saving images"):
            if args.max_images and total >= args.max_images:
                break

            row = table.slice(i, 1).to_pydict()
            img_dict = row["image"][0]

            # Decode image from bytes
            img_bytes = img_dict["bytes"]
            if img_bytes is None:
                continue

            # Determine filename
            fname = img_dict.get("path", None)
            if fname is None or fname == "":
                fname = f"ILSVRC2012_val_{total:08d}.JPEG"
            else:
                fname = Path(fname).name  # strip any directory prefix if present

            out_path = out_dir / fname
            if out_path.exists():
                total += 1
                continue

            img = Image.open(io.BytesIO(img_bytes))
            if img.mode != "RGB":
                img = img.convert("RGB")

            if args.size:
                img = img.resize((args.size, args.size), Image.BICUBIC)

            img.save(out_path, "JPEG", quality=95)
            total += 1

        if not args.keep_parquet:
            os.remove(local_path)

    count = len(list(out_dir.glob("*.JPEG")))
    print(f"\n✓ Done: {count} images saved to {out_dir}")


if __name__ == "__main__":
    main()
