#!/usr/bin/env python3
"""Prepare the official EDM2-S guided preset and source repository."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


MODEL_ROOT = Path(__file__).resolve().parent
DEFAULT_REPO = "https://github.com/NVlabs/edm2.git"
DEFAULT_NET = (
    "https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/"
    "edm2-img512-s-2147483-0.085.pkl"
)
DEFAULT_GNET = (
    "https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/"
    "edm2-img512-xs-uncond-2147483-0.085.pkl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--repo-dir", type=Path, default=MODEL_ROOT / "vendor/edm2")
    parser.add_argument("--cache-dir", type=Path, default=MODEL_ROOT / "weights/cache")
    parser.add_argument("--net-url", default=DEFAULT_NET)
    parser.add_argument("--gnet-url", default=DEFAULT_GNET)
    parser.add_argument("--skip-repo", action="store_true")
    parser.add_argument("--skip-net", action="store_true")
    parser.add_argument("--skip-gnet", action="store_true")
    return parser.parse_args()


def clone_repo(url: str, destination: Path) -> None:
    if destination.exists():
        if not (destination / "generate_images.py").is_file():
            raise RuntimeError(f"Existing EDM2 directory is not the official repo: {destination}")
        print(f"[OK] EDM2 repository already exists: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", url, str(destination)], check=True)
    print(f"[OK] EDM2 repository: {destination}")


def download_url(url: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / url.rsplit("/", 1)[-1]
    if destination.is_file() and destination.stat().st_size > 1024:
        print(f"[OK] EDM2 checkpoint already exists: {destination}")
        return destination
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
    print(f"[OK] EDM2 checkpoint: {destination}")
    return destination


def main() -> None:
    args = parse_args()
    if not args.skip_repo:
        clone_repo(args.repo, args.repo_dir)
    if not args.skip_net:
        download_url(args.net_url, args.cache_dir)
    if not args.skip_gnet:
        download_url(args.gnet_url, args.cache_dir)
    print(f"Set DNNLIB_CACHE_DIR={args.cache_dir} before running edm2/run.py.")


if __name__ == "__main__":
    main()
