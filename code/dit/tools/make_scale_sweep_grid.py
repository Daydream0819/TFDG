#!/usr/bin/env python3
"""Render paired DiT CFG scales as columns for qualitative comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scales", nargs="+", type=float, required=True)
    parser.add_argument(
        "--run-root-template",
        required=True,
        help="Path template such as 'runs/qual_cfg_w{scale}'.",
    )
    parser.add_argument("--method", default="cfg")
    parser.add_argument("--class-ids", nargs="+", type=int, required=True)
    parser.add_argument(
        "--seeds-per-class", type=int, default=2,
        help="Number of earliest frozen seeds to display per class.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def scale_text(scale: float) -> str:
    return f"{scale:g}"


def load_rows(manifest: Path, class_ids: list[int], per_class: int) -> list[dict]:
    if per_class < 1:
        raise ValueError("--seeds-per-class must be positive")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    rows = payload.get("samples", []) if isinstance(payload, dict) else payload
    selected: list[dict] = []
    for class_id in class_ids:
        class_rows = [row for row in rows if int(row["class_id"]) == class_id]
        if len(class_rows) < per_class:
            raise ValueError(
                f"Manifest has only {len(class_rows)} rows for class {class_id}; "
                f"requested {per_class}."
            )
        selected.extend(class_rows[:per_class])
    return selected


def image_path(template: str, method: str, scale: float, sample_id: str) -> Path:
    root = Path(template.format(scale=scale_text(scale)))
    return root / method / f"w{scale_text(scale)}" / f"{sample_id}.png"


def main() -> None:
    args = parse_args()
    rows = load_rows(args.manifest, args.class_ids, args.seeds_per_class)
    rendered: dict[tuple[str, float], Image.Image] = {}
    image_size = None
    for row in rows:
        sample_id = str(row["sample_id"])
        for scale in args.scales:
            path = image_path(args.run_root_template, args.method, scale, sample_id)
            if not path.is_file():
                raise FileNotFoundError(f"Missing paired sample: {path}")
            with Image.open(path) as image:
                item = image.convert("RGB")
            if image_size is None:
                image_size = item.size
            elif item.size != image_size:
                raise ValueError("All displayed samples must have identical dimensions")
            rendered[sample_id, scale] = item
    assert image_size is not None

    width, height = image_size
    label_width, header_height, gap = 160, 42, 4
    canvas = Image.new(
        "RGB",
        (
            label_width + len(args.scales) * width + (len(args.scales) - 1) * gap,
            header_height + len(rows) * height + (len(rows) - 1) * gap,
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((10, 14), "Class / seed", fill="black", font=font)
    for column, scale in enumerate(args.scales):
        x = label_width + column * (width + gap)
        label = f"CFG (w={scale_text(scale)})"
        draw.text((x + 8, 14), label, fill="black", font=font)

    for row_index, row in enumerate(rows):
        y = header_height + row_index * (height + gap)
        label = f"class {int(row['class_id']):04d}\nseed {int(row['seed'])}"
        draw.multiline_text((10, y + height // 2 - 10), label, fill="black", font=font)
        sample_id = str(row["sample_id"])
        for column, scale in enumerate(args.scales):
            x = label_width + column * (width + gap)
            canvas.paste(rendered[sample_id, scale], (x, y))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"Saved CFG scale sweep grid: {args.output}")


if __name__ == "__main__":
    main()
