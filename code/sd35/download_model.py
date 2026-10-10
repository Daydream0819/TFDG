#!/usr/bin/env python3
"""Download the SD3.5-Large Diffusers checkpoint into this model folder."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_REPO = "stabilityai/stable-diffusion-3.5-large"
MODEL_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--output",
        type=Path,
        default=MODEL_ROOT / "weights/sd35-large",
        help="Local Diffusers directory used by run.py.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token; HF_TOKEN is used when omitted.",
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        default=None,
        help="Optional Hugging Face allow-pattern; repeat for multiple patterns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.token:
        raise SystemExit(
            "SD3.5-Large is gated. Accept the model license and set HF_TOKEN "
            "before downloading."
        )
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
    from huggingface_hub import snapshot_download

    print(f"Downloading {args.repo}@{args.revision} -> {args.output}")
    snapshot_download(
        repo_id=args.repo,
        revision=args.revision,
        token=args.token,
        local_dir=str(args.output),
        allow_patterns=args.allow_pattern,
    )
    required = args.output / "model_index.json"
    if not required.is_file():
        raise RuntimeError(f"Download completed without {required}")
    print(f"[OK] SD3.5 checkpoint is ready at {args.output}")


if __name__ == "__main__":
    main()
