#!/usr/bin/env python3
"""Create deterministic DiT class--seed manifests for paired experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--class-ids", nargs="+", type=int, default=None)
    parser.add_argument("--seeds-per-class", type=int, default=None)
    parser.add_argument("--seed-offset", type=int, default=None)
    parser.add_argument(
        "--subset-of", type=Path, default=None,
        help="Use the first --max-samples rows of an existing frozen manifest.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def evenly_spaced_classes(count):
    if not 1 <= count <= 1000:
        raise ValueError("--num-classes must lie in [1, 1000]")
    if count == 1:
        return [0]
    return [round(index * 999 / (count - 1)) for index in range(count)]


def read_rows(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("samples", []) if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError("--subset-of must contain a non-empty JSON manifest")
    return [
        {
            "sample_id": str(row["sample_id"]),
            "class_id": int(row["class_id"]),
            "seed": int(row["seed"]),
        }
        for row in rows
    ]


def main():
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite manifest: {args.output}")
    if args.subset_of is not None:
        if args.class_ids is not None or args.num_classes is not None:
            raise ValueError("--subset-of cannot be combined with class-selection arguments")
        if args.max_samples is None or args.max_samples < 1:
            raise ValueError("--subset-of requires a positive --max-samples")
        rows = read_rows(args.subset_of)[: args.max_samples]
        if len(rows) != args.max_samples:
            raise ValueError("--max-samples exceeds the source manifest length")
    else:
        if args.seeds_per_class is None or args.seed_offset is None:
            raise ValueError("--seeds-per-class and --seed-offset are required without --subset-of")
        if args.seeds_per_class < 1:
            raise ValueError("--seeds-per-class must be positive")
        if (args.class_ids is None) == (args.num_classes is None):
            raise ValueError("Specify exactly one of --num-classes or --class-ids")
        classes = args.class_ids if args.class_ids is not None else evenly_spaced_classes(args.num_classes)
        if len(set(classes)) != len(classes) or any(not 0 <= class_id < 1000 for class_id in classes):
            raise ValueError("class ids must be unique values in [0, 999]")
        rows = []
        for class_index, class_id in enumerate(classes):
            for seed_index in range(args.seeds_per_class):
                seed = args.seed_offset + class_index * args.seeds_per_class + seed_index
                rows.append(
                    {
                        "sample_id": f"class{class_id:04d}_seed{seed:06d}",
                        "class_id": class_id,
                        "seed": seed,
                    }
                )
        if args.max_samples is not None:
            rows = rows[: args.max_samples]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(rows)} samples to {args.output}")


if __name__ == "__main__":
    main()
