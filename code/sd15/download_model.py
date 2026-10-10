#!/usr/bin/env python3
"""Download Stable Diffusion v1.5 into a local Diffusers directory."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time


MODEL_ROOT = Path(__file__).resolve().parent
DEFAULT_REPO = "runwayml/stable-diffusion-v1-5"
DOWNLOAD_PROFILE = "fp16_safetensors_inference_only_v1"
ALLOW_PATTERNS = (
    "model_index.json",
    "scheduler/*",
    "tokenizer/*",
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",
    "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--output", type=Path, default=MODEL_ROOT / "weights/sd15")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    completion_marker = args.output / ".download_complete.json"
    if completion_marker.is_file():
        try:
            completed = json.loads(completion_marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            completed = {}
        if completed.get("profile") == DOWNLOAD_PROFILE:
            print(f"[OK] SD1.5 checkpoint already exists: {args.output}")
            return
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "600")
    from huggingface_hub import snapshot_download

    print(f"Downloading {args.repo}@{args.revision} -> {args.output}")
    workers = int(os.environ.get("SD15_DOWNLOAD_WORKERS", "4"))
    last_error = None
    for attempt in range(1, 6):
        try:
            snapshot_path = snapshot_download(
                repo_id=args.repo,
                revision=args.revision,
                token=args.token,
                local_dir=str(args.output),
                max_workers=workers,
                allow_patterns=list(ALLOW_PATTERNS),
            )
            break
        except Exception as error:
            last_error = error
            if attempt == 5:
                raise
            delay = min(5 * attempt, 20)
            print(
                f"Download attempt {attempt}/5 was interrupted; "
                f"resuming from the same cache in {delay}s: {error}"
            )
            time.sleep(delay)
    else:  # pragma: no cover - defensive; the loop either breaks or raises.
        raise RuntimeError("SD1.5 download failed") from last_error
    required = [
        args.output / "model_index.json",
        args.output / "scheduler/scheduler_config.json",
        args.output / "tokenizer/tokenizer_config.json",
        args.output / "text_encoder/config.json",
        args.output / "text_encoder/model.fp16.safetensors",
        args.output / "unet/config.json",
        args.output / "unet/diffusion_pytorch_model.fp16.safetensors",
        args.output / "vae/config.json",
        args.output / "vae/diffusion_pytorch_model.fp16.safetensors",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Download completed with missing model files: {missing}")
    completion_marker.write_text(
        json.dumps(
            {
                "repo": args.repo,
                "revision": args.revision,
                "profile": DOWNLOAD_PROFILE,
                "snapshot_path": str(snapshot_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[OK] SD1.5 checkpoint is ready: {args.output}")


if __name__ == "__main__":
    main()
