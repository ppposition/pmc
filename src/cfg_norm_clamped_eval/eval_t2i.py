"""
Evaluate text-to-image generation (SD3.5 etc.).

Metrics:
  - ImageReward   — human-preference alignment score
  - CLIP-Score    — text-image cosine similarity (ViT-L/14)
  - Saturation    — mean HSV saturation channel (higher = more colorful)
  - FID           — Fréchet Inception Distance (requires --ref-dir)

Usage:
    # Per-prompt evaluation
    python -m cfg_norm_clamped_eval.eval_t2i --gen-dir results_sd35/cat/01_constant/cfg1.0/images --prompt "a cat"

    # Batch mode: evaluate all subdirectories
    python -m cfg_norm_clamped_eval.eval_t2i --gen-dir results_sd35/cat/01_constant --batch --prompt "a cat"

    # COCO FID-30K with per-image prompts
    python -m cfg_norm_clamped_eval.eval_t2i --gen-dir coco_data/generated_euler \\
        --ref-dir coco_data/val2014 --prompt-file coco_data/selected_captions.txt

    # Batch over categories
    python -m cfg_norm_clamped_eval.eval_t2i --gen-dir results_sd35 --batch --prompt "a photo"
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ── ImageReward ─────────────────────────────────────────────────────────


_REWARD_MODEL = None


def _load_image_reward(device: str):
    global _REWARD_MODEL
    if _REWARD_MODEL is not None:
        return _REWARD_MODEL

    import logging

    import transformers
    import transformers.modeling_utils as mu

    # Monkey-patches for ImageReward compatibility with transformers 5.x
    os.environ["WANDB_MODE"] = "disabled"
    logging.disable(logging.WARNING)

    mu.apply_chunking_to_forward = transformers.apply_chunking_to_forward
    mu.prune_linear_layer = transformers.pytorch_utils.prune_linear_layer

    def _find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
        heads = set(heads) - already_pruned_heads
        result = {}
        for head in sorted(heads):
            result[head] = torch.arange(head_size) + head * head_size
        return result

    mu.find_pruneable_heads_and_indices = _find_pruneable_heads_and_indices

    _orig_tie = transformers.PreTrainedModel.tie_weights

    def _patched_tie(self, *args, **kwargs):
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = getattr(self, "_tied_weights_keys", None) or {}
        return _orig_tie(self, *args, **kwargs)

    transformers.PreTrainedModel.tie_weights = _patched_tie

    def _get_head_mask(self, head_mask, num_hidden_layers):
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * num_hidden_layers
        return head_mask

    transformers.PreTrainedModel.get_head_mask = _get_head_mask

    def _invert_attention_mask(self, encoder_attention_mask):
        if encoder_attention_mask.dim() == 3:
            ext = encoder_attention_mask[:, None, :, :]
        elif encoder_attention_mask.dim() == 2:
            ext = encoder_attention_mask[:, None, None, :]
        else:
            ext = encoder_attention_mask
        ext = ext.to(dtype=self.dtype)
        ext = (1.0 - ext) * -10000.0
        return ext

    transformers.PreTrainedModel.invert_attention_mask = _invert_attention_mask

    from transformers import BertTokenizer

    def _patched_init_tokenizer():
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
        tokenizer.enc_token_id = tokenizer.convert_tokens_to_ids("[ENC]")
        return tokenizer

    import ImageReward.models.BLIP.blip as _blip
    import ImageReward.models.BLIP.blip_pretrain as _bp

    _blip.init_tokenizer = _patched_init_tokenizer
    _bp.init_tokenizer = _patched_init_tokenizer

    from ImageReward import utils as ir_utils

    model = ir_utils.load("ImageReward-v1.0")
    model.to(device)
    model.eval()
    _REWARD_MODEL = model
    return model


def compute_image_reward(
    image_dir: str, prompts: list[str], device: str = "cuda", batch_size: int = 32
) -> dict:
    model = _load_image_reward(device)

    img_files = sorted(Path(image_dir).glob("*"))
    img_files = [
        f for f in img_files if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    ]
    assert img_files, f"No images found in {image_dir}"

    if len(prompts) == 1:
        prompts = prompts * len(img_files)
    assert len(prompts) == len(img_files), f"Need {len(img_files)} prompts, got {len(prompts)}"

    all_scores = []
    for i in tqdm(range(0, len(img_files), batch_size), desc="ImageReward"):
        batch_files = img_files[i : i + batch_size]
        batch_prompts = prompts[i : i + batch_size]
        for fp, prompt in zip(batch_files, batch_prompts):
            score = model.score(prompt, Image.open(fp).convert("RGB"))
            all_scores.append(score)

    scores = torch.tensor(all_scores).cpu().numpy()
    return {"ImageReward_mean": float(scores.mean()), "ImageReward_std": float(scores.std())}


# ── CLIP-Score ──────────────────────────────────────────────────────────


_CLIP_MODEL = None
_CLIP_PROCESSOR = None


def _load_clip(device: str):
    global _CLIP_MODEL, _CLIP_PROCESSOR
    if _CLIP_MODEL is not None:
        return _CLIP_MODEL, _CLIP_PROCESSOR
    from transformers import CLIPModel, CLIPProcessor

    _CLIP_MODEL = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
    _CLIP_PROCESSOR = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    return _CLIP_MODEL, _CLIP_PROCESSOR


@torch.no_grad()
def compute_clip_score(
    image_dir: str, prompts: list[str], device: str = "cuda", batch_size: int = 32
) -> dict:
    model, processor = _load_clip(device)

    img_files = sorted(Path(image_dir).glob("*"))
    img_files = [
        f for f in img_files if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    ]
    assert img_files, f"No images found in {image_dir}"

    if len(prompts) == 1:
        prompts = prompts * len(img_files)
    assert len(prompts) == len(img_files), f"Need {len(img_files)} prompts, got {len(prompts)}"

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

        img_emb = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        txt_emb = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
        scores = (img_emb * txt_emb).sum(dim=-1)
        all_scores.append(scores.cpu().numpy())

    all_scores = np.concatenate(all_scores)
    return {"CLIP-Score_mean": float(all_scores.mean()), "CLIP-Score_std": float(all_scores.std())}


# ── Saturation (HSV S-channel mean) ─────────────────────────────────────


def compute_saturation(image_dir: str) -> dict:
    """Mean HSV saturation across all images.

    Each pixel yields S ∈ [0, 1]; result is the global average.
    Higher = more colorful / saturated.
    """
    img_files = sorted(Path(image_dir).glob("*"))
    img_files = [
        f for f in img_files if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    ]
    assert img_files, f"No images found in {image_dir}"

    total_s = 0.0
    total_pixels = 0
    for fp in tqdm(img_files, desc="Saturation"):
        img = Image.open(fp).convert("HSV")
        s = np.array(img, dtype=np.float32)[:, :, 1] / 255.0
        total_s += s.sum()
        total_pixels += s.size

    mean_sat = total_s / total_pixels
    return {"Saturation_mean": float(mean_sat)}


# ── FID ─────────────────────────────────────────────────────────────────


def compute_fid(
    gen_dir: str, ref_dir: str, device: str = "cuda", batch_size: int = 64, num_workers: int = 8
) -> float:
    from cleanfid import fid as cleanfid

    return cleanfid.compute_fid(
        fdir1=str(gen_dir),
        fdir2=str(ref_dir),
        mode="clean",
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        verbose=True,
    )


# ── Batch helpers ───────────────────────────────────────────────────────


def find_image_dirs(base_dir: Path) -> list[Path]:
    """Find directories containing images (either directly or via `images/`)."""
    result = []
    for d in sorted(base_dir.iterdir()):
        if not d.is_dir():
            continue
        # Check direct children for images
        imgs = [
            f for f in d.iterdir() if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        ]
        if imgs:
            result.append(d)
            continue
        # Check for images/ subdirectory
        img_dir = d / "images"
        if img_dir.is_dir():
            imgs = list(img_dir.glob("*"))
            if any(f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"} for f in imgs):
                result.append(img_dir)
    return result


def save_csv(results: dict, path: str):
    import csv

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        all_keys = set()
        for v in results.values():
            all_keys.update(v.keys())
        headers = ["dir"] + sorted(all_keys)
        w.writerow(headers)
        for d, metrics in sorted(results.items()):
            row = [d] + [metrics.get(k, "") for k in headers[1:]]
            w.writerow(row)
    print(f"\n✓ Saved to {path}")


# ── Main ────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(
        description="T2I evaluation (ImageReward, CLIP-Score, Saturation, FID)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--gen-dir", required=True, help="Directory of generated images")
    p.add_argument(
        "--ref-dir", default=None, help="Reference/real images (for FID; e.g. COCO val2014)"
    )
    p.add_argument("--prompt", default=None, help="Text prompt for all images")
    p.add_argument(
        "--prompt-file", default=None, help="File with one prompt per line (matches sorted images)"
    )
    p.add_argument("--batch", action="store_true", help="Evaluate every subdirectory of gen-dir")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--skip-ir", action="store_true")
    p.add_argument("--skip-clip", action="store_true")
    p.add_argument("--skip-saturation", action="store_true")
    p.add_argument("--skip-fid", action="store_true")
    p.add_argument("--save", type=str, default=None, help="Save results to CSV")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Read prompts
    if args.prompt_file:
        with open(args.prompt_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
    elif args.prompt:
        prompts = [args.prompt]
    else:
        prompts = None

    # Collect directories
    gen_dirs: list[Path] = []
    if args.batch:
        gen_dirs = find_image_dirs(Path(args.gen_dir))
        if not gen_dirs:
            print(f"Error: no subdirectories with images found in {args.gen_dir}")
            sys.exit(1)
        print(f"Batch mode: {len(gen_dirs)} directories found")
    else:
        gen_dirs = [Path(args.gen_dir)]

    all_results = {}

    ref_base = Path(args.gen_dir)
    for gen_dir in gen_dirs:
        label = str(gen_dir.relative_to(ref_base)) if args.batch else gen_dir.name

        print(f"\n{'=' * 60}")
        print(f"Evaluating: {gen_dir}")

        # Image count
        img_files = [
            f
            for f in sorted(gen_dir.glob("*"))
            if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        ]
        print(f"  Images: {len(img_files)}")

        results = {}

        # ── ImageReward ──
        if not args.skip_ir:
            if prompts is not None:
                print("\n--- ImageReward ---")
                results.update(compute_image_reward(str(gen_dir), prompts, device, args.batch_size))
                print(
                    f"  ImageReward = {results['ImageReward_mean']:.4f} ± "
                    f"{results['ImageReward_std']:.4f}"
                )
            else:
                print("  ⚠  Skip ImageReward: no prompt provided")

        # ── CLIP-Score ──
        if not args.skip_clip:
            if prompts is not None:
                print("\n--- CLIP-Score ---")
                results.update(compute_clip_score(str(gen_dir), prompts, device, args.batch_size))
                print(
                    f"  CLIP-Score = {results['CLIP-Score_mean']:.4f} ± "
                    f"{results['CLIP-Score_std']:.4f}"
                )
            else:
                print("  ⚠  Skip CLIP-Score: no prompt provided")

        # ── Saturation ──
        if not args.skip_saturation:
            print("\n--- Saturation ---")
            results.update(compute_saturation(str(gen_dir)))
            print(f"  Saturation = {results['Saturation_mean']:.4f}")

        # ── FID ──
        if not args.skip_fid:
            if args.ref_dir is not None:
                print("\n--- FID ---")
                results["FID"] = compute_fid(
                    str(gen_dir), args.ref_dir, device, args.batch_size, args.num_workers
                )
                print(f"  FID = {results['FID']:.4f}")
            else:
                print("  ⚠  Skip FID: no --ref-dir provided")

        all_results[label] = results

        print(f"\n  Summary for {label}:")
        for k, v in results.items():
            if isinstance(v, float):
                print(f"    {k:<20s} = {v:.4f}")

    # ── CSV ──
    if args.save:
        save_csv(all_results, args.save)

    # ── Overall summary ──
    if len(all_results) > 1:
        print(f"\n{'=' * 60}")
        print("Overall Summary")
        print(f"{'=' * 60}")
        for label, results in all_results.items():
            parts = [f"{label}:"]
            for k, v in results.items():
                if isinstance(v, float):
                    parts.append(f"{k}={v:.4f}")
            print("  " + ", ".join(parts))


if __name__ == "__main__":
    main()
