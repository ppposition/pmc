"""Portable image galleries and bounded-size contact sheets for generated samples."""

import html
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


def add_gallery_arguments(parser):
    parser.add_argument("--grid-columns", type=int, default=4)
    parser.add_argument("--grid-page-size", type=int, default=64)
    parser.add_argument("--thumbnail-size", type=int, default=256)


def validate_gallery_arguments(args):
    if not 1 <= args.grid_columns <= 16:
        raise ValueError("--grid-columns must be between 1 and 16")
    if not 1 <= args.grid_page_size <= 256:
        raise ValueError("--grid-page-size must be between 1 and 256")
    if not 32 <= args.thumbnail_size <= 512:
        raise ValueError("--thumbnail-size must be between 32 and 512")


def prepare_output(out_dir):
    """Require an empty run directory to avoid mixing or overwriting samples."""
    root = Path(out_dir)
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError(f"Output must be a new or empty directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "images").mkdir()
    return root


def save_metadata(root, config, records):
    root = Path(root)
    (root / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (root / "samples.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def write_gallery(out_dir, records, *, columns=4, page_size=64, thumbnail_size=256):
    """Create paginated PNG sheets and a local HTML gallery; never modify originals."""
    if not records:
        raise ValueError("Cannot create a gallery without samples")
    root = Path(out_dir)
    preview = root / "preview"
    preview.mkdir(exist_ok=True)
    sheets = []
    label_height = 30
    for start in range(0, len(records), page_size):
        batch = records[start : start + page_size]
        width = min(columns, len(batch)) * thumbnail_size
        height = math.ceil(len(batch) / columns) * (thumbnail_size + label_height)
        sheet = Image.new("RGB", (width, height), "#f4f4f5")
        draw = ImageDraw.Draw(sheet)
        for offset, record in enumerate(batch):
            with Image.open(root / record["file"]) as source:
                thumb = ImageOps.contain(
                    source.convert("RGB"),
                    (thumbnail_size, thumbnail_size),
                    Image.Resampling.LANCZOS,
                )
            x = (offset % columns) * thumbnail_size
            y = (offset // columns) * (thumbnail_size + label_height)
            sheet.paste(thumb, (x + (thumbnail_size - thumb.width) // 2, y))
            label = f"#{record['index']:06d}"
            if "class_id" in record:
                label += f"  class {record['class_id']}"
            draw.text((x + 6, y + thumbnail_size + 7), label, fill="#18181b")
        filename = f"grid_{len(sheets):04d}.png"
        sheet.save(preview / filename)
        sheets.append(filename)

    cards = []
    for record in records:
        filename = html.escape(record["file"], quote=True)
        description = record.get("prompt", f"Class {record.get('class_id', '')}")
        caption = html.escape(f"#{record['index']:06d} · seed {record['seed']} · {description}")
        cards.append(
            f'<figure><a href="{filename}" target="_blank" rel="noopener">'
            f'<img loading="lazy" src="{filename}" alt="{caption}"></a>'
            f"<figcaption>{caption}</figcaption></figure>"
        )
    links = " · ".join(
        f'<a href="preview/{name}">Grid {i + 1}</a>' for i, name in enumerate(sheets)
    )
    document = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Generated samples</title>
<style>
body{font:16px system-ui,sans-serif;margin:32px;background:#fafafa;color:#18181b}
main{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:20px}
figure{margin:0;background:white;border:1px solid #ddd;border-radius:8px;overflow:hidden}
img{width:100%;aspect-ratio:1;object-fit:contain;background:#eee}
figcaption{padding:12px;overflow-wrap:anywhere} a{color:#1756a9} nav{margin:20px 0}
</style><h1>Generated samples</h1>
"""
    document += f"<p>{len(records)} images. Click an image to open the original PNG.</p>"
    document += f"<nav>{links}</nav><main>{''.join(cards)}</main></html>\n"
    (root / "index.html").write_text(document, encoding="utf-8")
    print(f"Gallery: {(root / 'index.html').resolve()}")
    print(f"Contact sheets: {preview.resolve()} ({len(sheets)} pages)")
