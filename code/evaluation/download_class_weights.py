#!/usr/bin/env python3
"""Download the frozen weights used by DiT/EDM2 class-conditional evaluation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/root/autodl-tmp/metric_cache/torch"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = args.cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(cache_dir)

    from torchvision import models

    inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    del inception, resnet
    try:
        import pyiqa

        clipiqa = pyiqa.create_metric("clipiqa", device="cpu")
        del clipiqa
    except ImportError as error:
        raise RuntimeError("Install requirements_class.txt before downloading CLIP-IQA.") from error
    print(f"[OK] Inception-v3, ResNet-50, and CLIP-IQA weights are cached under {cache_dir}")


if __name__ == "__main__":
    main()
