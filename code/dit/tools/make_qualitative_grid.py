#!/usr/bin/env python3
"""Render paired DiT samples into a compact qualitative-comparison grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DEFAULT_METHODS = ("cfg", "cfg_rescale", "apg", "fdg", "c2fg", "tfdg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--class-id", type=int, required=True)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_rows(path: Path, class_id: int, max_samples: int) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("samples", []) if isinstance(payload, dict) else payload
    selected = [row for row in rows if int(row["class_id"]) == class_id][:max_samples]
    if not selected:
        raise ValueError(f"No manifest rows for class {class_id}")
    return selected


def main() -> None:
    args = parse_args()
    if args.max_samples < 1:
        raise ValueError("--max-samples must be positive")
    rows = load_rows(args.manifest, args.class_id, args.max_samples)
    images: dict[tuple[str, str], Image.Image] = {}
    image_size = None
    for method in args.methods:
        for row in rows:
            sample_id = str(row["sample_id"])
            path = args.generated_root / method / f"w{args.scale:g}" / f"{sample_id}.png"
            if not path.is_file():
                raise FileNotFoundError(f"Missing paired sample: {path}")
            with Image.open(path) as image:
                rendered = image.convert("RGB")
            if image_size is None:
                image_size = rendered.size
            elif rendered.size != image_size:
                raise ValueError("All images must have the same size")
            images[method, sample_id] = rendered
    assert image_size is not None
    width, height = image_size
    label_width, header_height, gap = 180, 38, 4
    canvas = Image.new(
        "RGB",
        (label_width + len(rows) * width + (len(rows) - 1) * gap,
         header_height + len(args.methods) * height + (len(args.methods) - 1) * gap),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for column, row in enumerate(rows):
        x = label_width + column * (width + gap)
        caption = f"class {args.class_id:04d}, seed {int(row['seed'])}"
        draw.text((x + 4, 12), caption, fill="black", font=font)
    for index, method in enumerate(args.methods):
        y = header_height + index * (height + gap)
        label = "TFDG (Ours)" if method == "tfdg" else method.replace("_", "-").upper()
        draw.text((12, y + height // 2 - 5), label, fill="black", font=font)
        for column, row in enumerate(rows):
            x = label_width + column * (width + gap)
            canvas.paste(images[method, str(row["sample_id"])], (x, y))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"Saved qualitative grid: {args.output}")


if __name__ == "__main__":
    main()
