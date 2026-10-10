#!/usr/bin/env python3
"""Prepare the official DiT-XL/2 checkpoint, VAE, and source repository."""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path


MODEL_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = (
    "https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt"
)
DEFAULT_REPO = "https://github.com/facebookresearch/DiT.git"
DEFAULT_VAE = "stabilityai/sd-vae-ft-mse"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.environ.get("DIT_REPO_URL", DEFAULT_REPO))
    parser.add_argument("--repo-dir", type=Path, default=MODEL_ROOT / "vendor/DiT")
    parser.add_argument("--checkpoint-url", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=MODEL_ROOT / "weights/DiT-XL-2-256x256.pt",
    )
    parser.add_argument("--vae-repo", default=DEFAULT_VAE)
    parser.add_argument("--vae", type=Path, default=MODEL_ROOT / "weights/vae")
    parser.add_argument("--skip-repo", action="store_true")
    parser.add_argument("--skip-checkpoint", action="store_true")
    parser.add_argument("--skip-vae", action="store_true")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    return parser.parse_args()


def clone_repo(url: str, destination: Path) -> None:
    if destination.exists():
        if not (destination / "download.py").is_file():
            raise RuntimeError(f"Existing DiT directory is not the official repo: {destination}")
        print(f"[OK] DiT repository already exists: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 4):
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.clone-", dir=destination.parent)
        )
        temporary.rmdir()
        command = [
            "git", "-c", "http.version=HTTP/1.1", "clone", "--depth", "1",
            "--filter=blob:none", url, str(temporary),
        ]
        try:
            subprocess.run(command, check=True)
            temporary.replace(destination)
            break
        except subprocess.CalledProcessError:
            if attempt == 3:
                raise RuntimeError(
                    "Unable to clone DiT after three attempts. Set DIT_REPO_URL to "
                    "a reachable mirror of the official repository and rerun setup_server.sh."
                )
    print(f"[OK] DiT repository: {destination}")


def download_url(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 1024:
        print(f"[OK] DiT checkpoint already exists: {destination}")
        return
    partial = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {url} -> {destination}")
    subprocess.run(
        [
            "curl",
            "--fail",
            "--location",
            "--retry",
            "8",
            "--retry-all-errors",
            "-C",
            "-",
            "-o",
            str(partial),
            url,
        ],
        check=True,
    )
    partial.replace(destination)
    if destination.stat().st_size <= 1024:
        raise RuntimeError(f"Downloaded checkpoint is unexpectedly small: {destination}")
    print(f"[OK] DiT checkpoint: {destination}")


def download_vae(repo_id: str, destination: Path, token: str | None) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "config.json").is_file():
        print(f"[OK] DiT VAE already exists: {destination}")
        return
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo_id, local_dir=str(destination), token=token)
    if not (destination / "config.json").is_file():
        raise RuntimeError(f"VAE download completed without {destination / 'config.json'}")
    print(f"[OK] DiT VAE: {destination}")


def main() -> None:
    args = parse_args()
    if not args.skip_repo:
        clone_repo(args.repo, args.repo_dir)
    if not args.skip_checkpoint:
        download_url(args.checkpoint_url, args.checkpoint)
    if not args.skip_vae:
        download_vae(args.vae_repo, args.vae, args.token)
    print("DiT model assets are ready.")


if __name__ == "__main__":
    main()
