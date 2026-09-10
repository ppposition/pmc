"""
Download COCO 2014 validation set and extract reference images for FID evaluation.

The COCO 2014 validation set is the standard reference for T2I FID benchmarks
(SD3, SDXL, DALL-E 2, Imagen all report FID against this set).

This script downloads:
  - val2014 images (~6.2 GB, 40,504 images)
  - Captions annotations (~240 MB)

Usage:
    python -m cfg_norm_clamped_eval.download_coco --data-dir /path/to/coco_data
"""

import argparse
import zipfile
from pathlib import Path


def download_file(url: str, dest: str, desc: str):
    """Download a file with progress bar."""
    try:
        import urllib.request

        from tqdm import tqdm

        class DownloadProgressBar:
            def __init__(self, total):
                self.pbar = tqdm(total=total, unit="B", unit_scale=True, desc=desc)

            def __call__(self, block_num, block_size, total_size):
                if self.pbar.total is None and total_size > 0:
                    self.pbar.total = total_size
                self.pbar.update(block_size)

        urllib.request.urlretrieve(url, dest, reporthook=DownloadProgressBar(0))
    except ImportError:
        import urllib.request

        print(f"Downloading {desc}...")
        urllib.request.urlretrieve(url, dest)
        print(f"Saved to {dest}")


def main():
    parser = argparse.ArgumentParser(description="Download COCO 2014 val for FID evaluation")
    parser.add_argument(
        "--data-dir", type=str, default="./coco_data", help="Directory to store COCO data"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # URLs
    val_images_url = "http://images.cocodataset.org/zips/val2014.zip"
    annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"

    # 1. Download val2014 images
    val_zip = data_dir / "val2014.zip"
    val_dir = data_dir / "val2014"

    if val_dir.exists() and len(list(val_dir.glob("*.jpg"))) > 40000:
        print(f"✓ val2014 already exists: {val_dir} ({len(list(val_dir.glob('*.jpg')))} images)")
    else:
        # Check if existing zip is valid (not a partial download)
        if val_zip.exists():
            try:
                with zipfile.ZipFile(val_zip, "r") as zf:
                    bad = zf.testzip()
                if bad is not None:
                    print(f"⚠  Corrupt zip detected ({bad}), re-downloading...")
                    val_zip.unlink()
            except (zipfile.BadZipFile, OSError):
                print("⚠  Invalid zip file, re-downloading...")
                val_zip.unlink()

        if not val_zip.exists():
            print("\nDownloading COCO val2014 images (~6.2 GB)...")
            print(f"  URL: {val_images_url}")
            download_file(val_images_url, str(val_zip), "val2014.zip")

        print(f"\nExtracting {val_zip}...")
        with zipfile.ZipFile(val_zip, "r") as zf:
            zf.extractall(data_dir)
        print(f"✓ Extracted to {val_dir}")

    # 2. Download annotations
    ann_zip = data_dir / "annotations_trainval2014.zip"
    ann_file = data_dir / "annotations" / "captions_val2014.json"

    if ann_file.exists():
        print(f"✓ Captions already exist: {ann_file}")
    else:
        # Check if existing zip is valid
        if ann_zip.exists():
            try:
                with zipfile.ZipFile(ann_zip, "r") as zf:
                    bad = zf.testzip()
                if bad is not None:
                    print(f"⚠  Corrupt annotations zip detected ({bad}), re-downloading...")
                    ann_zip.unlink()
            except (zipfile.BadZipFile, OSError):
                print("⚠  Invalid annotations zip, re-downloading...")
                ann_zip.unlink()

        if not ann_zip.exists():
            print("\nDownloading COCO annotations (~240 MB)...")
            print(f"  URL: {annotations_url}")
            download_file(annotations_url, str(ann_zip), "annotations.zip")

        print(f"\nExtracting {ann_zip}...")
        with zipfile.ZipFile(ann_zip, "r") as zf:
            zf.extractall(data_dir)
        print("✓ Extracted annotations")

    # 3. Print summary
    val_images = sorted(val_dir.glob("*.jpg"))
    print(f"\n{'=' * 50}")
    print("COCO FID-30K reference ready:")
    print(f"  Reference images: {val_dir} ({len(val_images)} images)")
    print(f"  Captions file:    {ann_file}")
    print("\nNext step:")
    print(f"  python -m cfg_norm_clamped_eval.generate_coco --data-dir {data_dir}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
