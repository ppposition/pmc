"""Generate SD3.5 images from custom prompts, with a gallery and no evaluation."""

import argparse
import json
import math
from pathlib import Path

from .gallery import (
    add_gallery_arguments,
    prepare_output,
    save_metadata,
    validate_gallery_arguments,
    write_gallery,
)
from .generate_coco_sd35 import MODEL_ID, generate_images


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    prompts = parser.add_mutually_exclusive_group(required=True)
    prompts.add_argument("--prompt", help="One prompt used for every sample")
    prompts.add_argument(
        "--prompt-file", help="UTF-8 prompts, one per line; cycle to fill the count"
    )
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--out-dir", default="outputs/sd35_preview")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--seed", "--noise-seed", dest="noise_seed", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--cfg-gamma", type=float, default=1.1)
    parser.add_argument("--cfg-schedule", choices=("constant", "norm-clamped"), default="constant")
    parser.add_argument("--scheduler", choices=("euler", "heun"), default="euler")
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--gpu", type=int, default=0)
    add_gallery_arguments(parser)
    return parser


def prepare_records(args):
    if args.num_samples <= 0 or args.batch_size <= 0 or args.num_steps <= 0:
        raise ValueError("Sample count, batch size, and step count must be positive")
    if args.height <= 0 or args.width <= 0 or args.height % 16 or args.width % 16:
        raise ValueError("Height and width must be positive multiples of 16")
    if not 1 <= args.max_sequence_length <= 512:
        raise ValueError("--max-sequence-length must be between 1 and 512")
    if not math.isfinite(args.cfg_scale) or args.cfg_scale < 1:
        raise ValueError("--cfg-scale must be finite and >= 1")
    if not math.isfinite(args.cfg_gamma) or args.cfg_gamma < 1:
        raise ValueError("--cfg-gamma must be finite and >= 1")
    prompts = (
        Path(args.prompt_file).read_text(encoding="utf-8").splitlines()
        if args.prompt_file
        else [args.prompt]
    )
    prompts = [prompt.strip() for prompt in prompts]
    if not prompts or any(not prompt for prompt in prompts):
        raise ValueError("Provide nonempty prompts; blank lines in prompt files are not allowed")
    return [
        {
            "index": index,
            "file": f"images/{index:06d}.png",
            "seed": args.noise_seed + index,
            "prompt": prompts[index % len(prompts)],
        }
        for index in range(args.num_samples)
    ]


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_gallery_arguments(args)
        records = prepare_records(args)
        root = prepare_output(args.out_dir)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    # The shared COCO engine records these fields for its manifest-based runs.
    args.selection_seed = None
    prompt_file = root / "prompts.txt"
    prompt_file.write_text(
        "".join(record["prompt"].replace("\n", " ") + "\n" for record in records), encoding="utf-8"
    )
    # Write sample metadata before generation so failed runs remain traceable.
    config = {**vars(args), "backend": "sd35", "seed_strategy": "seed + index"}
    save_metadata(root, config, records)
    generate_images(args, records, root / "samples.jsonl", prompt_file)
    config.update(json.loads((root / "run_config.json").read_text(encoding="utf-8")))
    save_metadata(root, config, records)
    write_gallery(
        root,
        records,
        columns=args.grid_columns,
        page_size=args.grid_page_size,
        thumbnail_size=args.thumbnail_size,
    )


if __name__ == "__main__":
    main()
