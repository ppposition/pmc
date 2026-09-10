"""Generate a requested number of SiT images with a gallery, without evaluation."""

import argparse
from copy import copy

from .gallery import (
    add_gallery_arguments,
    prepare_output,
    save_metadata,
    validate_gallery_arguments,
    write_gallery,
)
from .sit_sampling import (
    add_generation_arguments,
    generate_images,
    validate_generation_arguments,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_generation_arguments(parser)
    add_gallery_arguments(parser)
    parser.set_defaults(
        num_samples=16, batch_size=4, out_dir="outputs/imagenet_preview", cfg_schedule="constant"
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_generation_arguments(args)
        validate_gallery_arguments(args)
        root = prepare_output(args.out_dir)
    except ValueError as exc:
        parser.error(str(exc))

    records = [
        {
            "index": index,
            "file": f"images/{index:06d}.png",
            "seed": args.seed + index,
            "class_id": args.class_id if args.class_id is not None else index % args.num_classes,
        }
        for index in range(args.num_samples)
    ]
    save_metadata(root, {**vars(args), "backend": "sit", "seed_strategy": "seed + index"}, records)
    sampling_args = copy(args)
    sampling_args.out_dir = str(root / "images")
    sampling_args.per_sample_seed = True
    generate_images(sampling_args)
    write_gallery(
        root,
        records,
        columns=args.grid_columns,
        page_size=args.grid_page_size,
        thumbnail_size=args.thumbnail_size,
    )


if __name__ == "__main__":
    main()
