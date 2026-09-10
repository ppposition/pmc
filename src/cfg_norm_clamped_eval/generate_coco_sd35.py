#!/usr/bin/env python3
"""Reproducible SD3.5 sampling on a fixed subset of COCO 2014 val captions.

The script creates one deterministic manifest shared by all CFG runs. Generated
images use sequential filenames, so line N in prompts.txt always corresponds to
image N (a requirement of eval_metrics.py).
"""

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from .cfg_schedules import sd35_norm_clamped

MODEL_ID = "stabilityai/stable-diffusion-3.5-medium"


def select_coco_prompts(annotation_file: Path, count: int, seed: int) -> list[dict]:
    with annotation_file.open(encoding="utf-8") as f:
        annotations = json.load(f)["annotations"]

    # One deterministic caption per real image.
    annotations.sort(key=lambda item: (item["image_id"], item["id"]))
    first_by_image = {}
    for item in annotations:
        first_by_image.setdefault(
            item["image_id"],
            {
                "coco_image_id": item["image_id"],
                "caption_id": item["id"],
                "prompt": item["caption"].strip(),
            },
        )

    items = list(first_by_image.values())
    if count > len(items):
        raise ValueError(
            f"Requested {count} unique prompts, but COCO val has only {len(items)} images"
        )
    random.Random(seed).shuffle(items)
    return items[:count]


def prepare_manifest(
    data_dir: Path, count: int, selection_seed: int
) -> tuple[Path, Path, list[dict]]:
    annotation_file = data_dir / "annotations" / "captions_val2014.json"
    if not annotation_file.is_file():
        raise FileNotFoundError(f"Missing COCO captions: {annotation_file}")

    manifest_dir = data_dir / "eval_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    stem = f"coco_val_first_caption_n{count}_seed{selection_seed}"
    manifest_file = manifest_dir / f"{stem}.jsonl"
    prompt_file = manifest_dir / f"{stem}.txt"

    selected = select_coco_prompts(annotation_file, count, selection_seed)
    records = []
    for index, item in enumerate(selected):
        records.append(
            {
                "index": index,
                "file": f"{index:06d}.png",
                **item,
            }
        )

    expected_jsonl = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records)
    expected_prompts = "".join(x["prompt"].replace("\n", " ") + "\n" for x in records)

    # Never silently change an existing benchmark manifest.
    if manifest_file.exists() and manifest_file.read_text(encoding="utf-8") != expected_jsonl:
        raise RuntimeError(f"Existing manifest does not match requested setup: {manifest_file}")
    if prompt_file.exists() and prompt_file.read_text(encoding="utf-8") != expected_prompts:
        raise RuntimeError(f"Existing prompt file does not match requested setup: {prompt_file}")

    manifest_file.write_text(expected_jsonl, encoding="utf-8")
    prompt_file.write_text(expected_prompts, encoding="utf-8")
    return manifest_file, prompt_file, records


def valid_image(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="coco_data")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--num-samples", type=int, default=30_000)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--noise-seed", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument("--cfg-scale", type=float, required=True)
    parser.add_argument(
        "--cfg-gamma",
        type=float,
        default=None,
        help="Relaxation gamma for norm-clamped (None = schedule default 1.1)",
    )
    parser.add_argument(
        "--cfg-schedule",
        choices=("constant", "norm-clamped"),
        default="constant",
        help="Change only the per-step CFG scale; norm-clamped uses gamma from --cfg-gamma (default 1.1).",
    )
    parser.add_argument("--scheduler", choices=("euler", "heun"), default="euler")
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    manifest_file, prompt_file, records = prepare_manifest(
        Path(args.data_dir), args.num_samples, args.selection_seed
    )
    generate_images(args, records, manifest_file, prompt_file)


def generate_images(args, records, manifest_file=None, prompt_file=None):
    """Run the shared SD3.5 sampling engine for explicit prompt records."""
    if args.num_samples <= 0 or args.batch_size <= 0 or args.num_steps <= 0:
        raise ValueError("Sample count, batch size, and step count must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("SD3.5 sampling requires a CUDA GPU")
    torch.cuda.set_device(args.gpu)
    device = f"cuda:{args.gpu}"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        **vars(args),
        "model_id": args.model_id,
        "manifest": str(manifest_file),
        "prompt_file": str(prompt_file),
        "device": device,
    }
    config_file = out_dir / "run_config.json"
    if config_file.exists():
        old_config = json.loads(config_file.read_text(encoding="utf-8"))
        keys = (
            "model_id",
            "num_samples",
            "selection_seed",
            "noise_seed",
            "height",
            "width",
            "num_steps",
            "cfg_scale",
            "cfg_gamma",
            "cfg_schedule",
            "scheduler",
            "max_sequence_length",
        )
        mismatches = [key for key in keys if old_config.get(key) != run_config.get(key)]
        if mismatches:
            raise RuntimeError(
                f"Refusing to mix incompatible samples in {out_dir}; changed: {mismatches}"
            )
    config_file.write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    pending = [record for record in records if not valid_image(out_dir / record["file"])]
    print(f"Manifest: {manifest_file}")
    print(f"Prompts:  {prompt_file}")
    print(f"Output:   {out_dir}")
    print(f"Complete: {len(records) - len(pending)}, pending: {len(pending)}")
    if not pending:
        return

    from diffusers import (
        FlowMatchEulerDiscreteScheduler,
        FlowMatchHeunDiscreteScheduler,
        StableDiffusion3Pipeline,
    )

    class DynamicGuidanceStableDiffusion3Pipeline(StableDiffusion3Pipeline):
        @property
        def do_classifier_free_guidance(self):
            # A norm-clamped step temporarily stores one guidance scale per
            # sample. Keep CFG's control-flow decision scalar while preserving
            # the tensor for the native guidance calculation.
            if torch.is_tensor(self._guidance_scale):
                return self._dynamic_cfg_enabled
            return self._guidance_scale > 1

    pipe = DynamicGuidanceStableDiffusion3Pipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
    )
    scheduler_cls = (
        FlowMatchEulerDiscreteScheduler
        if args.scheduler == "euler"
        else FlowMatchHeunDiscreteScheduler
    )
    pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    cfg_hook = None
    if args.cfg_schedule == "norm-clamped":
        base_cfg_scale = args.cfg_scale
        pipe._dynamic_cfg_enabled = base_cfg_scale > 1

        def set_norm_clamped_scale(module, hook_args, hook_kwargs, output):
            # The first half is unconditional and the second half conditional,
            # matching StableDiffusion3Pipeline's native CFG batching.
            if hook_kwargs.get("skip_layers") is not None:
                return
            model_output = output[0]
            if model_output.shape[0] % 2:
                return
            batch_size = model_output.shape[0] // 2
            v_uncond, v_cond = model_output.chunk(2)
            xt = hook_kwargs["hidden_states"][:batch_size]
            timestep = hook_kwargs["timestep"]
            sigma = float(timestep[0].float().item() / 1000.0)
            norm_kwargs = dict(
                gap=v_cond - v_uncond,
                v_uncond=v_uncond,
                v_cond=v_cond,
                cfg_scale=base_cfg_scale,
                xt=xt,
            )
            if args.cfg_gamma is not None:
                norm_kwargs["gamma"] = args.cfg_gamma
            weight = sd35_norm_clamped(sigma, **norm_kwargs)
            pipe._guidance_scale = weight.view(-1, 1, 1, 1).to(v_uncond.dtype)

        cfg_hook = pipe.transformer.register_forward_hook(set_norm_clamped_scale, with_kwargs=True)

        def restore_base_cfg_scale(pipe, step, timestep, callback_kwargs):
            # The next loop iteration needs a scalar for the pipeline's CFG test.
            pipe._guidance_scale = base_cfg_scale
            return callback_kwargs

    common_kwargs = {
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.num_steps,
        "guidance_scale": args.cfg_scale,
        "max_sequence_length": args.max_sequence_length,
        "output_type": "pil",
    }
    if args.cfg_schedule == "norm-clamped":
        common_kwargs["callback_on_step_end"] = restore_base_cfg_scale

    def run_pipe(**kwargs):
        try:
            return pipe(**kwargs)
        finally:
            # Also restore after errors (for example, a recoverable batch OOM).
            if args.cfg_schedule == "norm-clamped":
                pipe._guidance_scale = base_cfg_scale

    for start in tqdm(range(0, len(pending), args.batch_size), desc="SD3.5 sampling"):
        batch = pending[start : start + args.batch_size]
        prompts = [x["prompt"] for x in batch]
        generators = [
            torch.Generator(device=device).manual_seed(args.noise_seed + x["index"]) for x in batch
        ]
        try:
            images = run_pipe(prompt=prompts, generator=generators, **common_kwargs).images
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            images = []
            # Recreate generators because the failed batched call may have advanced them.
            for record, prompt in zip(batch, prompts):
                generator = torch.Generator(device=device).manual_seed(
                    args.noise_seed + record["index"]
                )
                image = run_pipe(
                    prompt=prompt,
                    generator=generator,
                    **common_kwargs,
                ).images[0]
                images.append(image)

        for record, image in zip(batch, images):
            image_path = out_dir / record["file"]
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(image_path)

    remaining = [x["file"] for x in records if not valid_image(out_dir / x["file"])]
    if remaining:
        print(f"Generation incomplete; {len(remaining)} invalid/missing images", file=sys.stderr)
        sys.exit(2)
    print(f"Finished {len(records)} images in {out_dir}")

    if cfg_hook is not None:
        cfg_hook.remove()


if __name__ == "__main__":
    main()
