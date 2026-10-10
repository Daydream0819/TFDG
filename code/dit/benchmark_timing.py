#!/usr/bin/env python3
"""Measure paired DiT-XL/2 reverse-denoising latency for six methods.

Only the 250-step reverse denoising loop is timed. Checkpoint loading, VAE
decoding, image I/O, and other method-invariant setup are excluded so the
result directly measures the inference overhead of the guidance rule.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

MODEL_ROOT = Path(__file__).resolve().parent
CODE_ROOT = MODEL_ROOT.parent
RELEASE_ROOT = CODE_ROOT.parent
for path in (CODE_ROOT, RELEASE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.experiment import add_guidance_args, guidance_config
from run import DiTGuidanceModel, load_samples


BENCHMARK_METHODS = ("cfg", "cfg_rescale", "apg", "fdg", "c2fg", "tfdg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dit-root", type=Path, default=MODEL_ROOT / "vendor/DiT")
    parser.add_argument(
        "--checkpoint", type=Path,
        default=MODEL_ROOT / "weights/DiT-XL-2-256x256.pt",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--details-output", type=Path)
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--image-size", choices=[256, 512], type=int, default=256)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--guidance-channels", choices=[3, 4], type=int, default=3)
    parser.add_argument("--device", default="cuda")
    # load_samples() requires these execution fields even though timing uses
    # one unsharded, frozen manifest.
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    add_guidance_args(parser, default_scale=10.0, default_norm=5.0, default_momentum=-0.5)
    parser.set_defaults(
        methods=list(BENCHMARK_METHODS),
        scales=[10.0],
        cutoff=0.30,
        low_min_cfg=1.0,
        high_min_cfg=1.0,
        low_eta=8.0,
        high_eta=4.0,
    )
    args = parser.parse_args()
    if args.max_samples < 1:
        parser.error("--max-samples must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if len(args.scales) != 1:
        parser.error("timing requires exactly one value in --scales")
    unexpected = set(args.methods).difference(BENCHMARK_METHODS)
    if unexpected:
        parser.error(
            "timing is registered for CFG, CFG-Rescale, APG, FDG, C2FG, and "
            f"TFDG only; unsupported methods: {sorted(unexpected)}"
        )
    return args


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_dit(args: argparse.Namespace):
    sys.path.insert(0, str(args.dit_root.resolve()))
    from diffusion import create_diffusion
    from download import find_model
    from models import DiT_models

    device = torch.device(args.device)
    latent_size = args.image_size // 8
    model = DiT_models["DiT-XL/2"](
        input_size=latent_size, num_classes=1000, learn_sigma=True
    ).to(device)
    model.load_state_dict(find_model(str(args.checkpoint)), strict=True)
    model.eval()
    return model, create_diffusion(str(args.steps)), create_diffusion(""), device, latent_size


def initial_noise_and_rng_state(
    device: torch.device, latent_size: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(
        (1, 4, latent_size, latent_size), generator=generator, device=device
    )
    state = (
        torch.cuda.get_rng_state(device)
        if device.type == "cuda"
        else torch.random.get_rng_state()
    )
    return noise, state


def restore_rng_state(device: torch.device, state: torch.Tensor) -> None:
    if device.type == "cuda":
        torch.cuda.set_rng_state(state, device)
    else:
        torch.random.set_rng_state(state)


def summarize(method: str, values: list[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "method": method,
        "sample_count": int(array.size),
        "denoising_seconds_per_image_mean": float(array.mean()),
        "denoising_seconds_per_image_std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "denoising_seconds_per_image_median": float(np.median(array)),
        "denoising_seconds_per_image_p95": float(np.percentile(array, 95)),
        "images_per_second": float(1.0 / array.mean()),
    }


def run_once(
    *, model, diffusion, base_diffusion, device, noise, rng_state, class_id,
    config, guidance_channels: int, measure: bool,
) -> tuple[torch.Tensor, float | None]:
    restore_rng_state(device, rng_state)
    guided_model = DiTGuidanceModel(
        model, base_diffusion, class_id, config, guidance_channels
    )
    synchronize(device)
    start = time.perf_counter() if measure else None
    generated = diffusion.p_sample_loop(
        guided_model,
        noise.shape,
        noise=noise.clone(),
        clip_denoised=False,
        model_kwargs={},
        progress=False,
        device=device,
    )
    synchronize(device)
    elapsed = time.perf_counter() - start if start is not None else None
    del guided_model
    return generated, elapsed


def main() -> None:
    args = parse_args()
    samples = load_samples(args)
    if len(samples) != args.max_samples:
        raise RuntimeError(
            f"Expected {args.max_samples} class--seed rows, found {len(samples)}. "
            "Use a frozen DiT manifest with at least 50 entries."
        )

    model, diffusion, base_diffusion, device, latent_size = load_dit(args)
    scale = float(args.scales[0])
    configs = {method: guidance_config(args, method, scale) for method in args.methods}

    # Warm each method's kernels and FFT plans with the same first condition.
    warmup = samples[0]
    noise, rng_state = initial_noise_and_rng_state(device, latent_size, warmup["seed"])
    for method in args.methods:
        for _ in range(args.warmup):
            generated, _ = run_once(
                model=model, diffusion=diffusion, base_diffusion=base_diffusion,
                device=device, noise=noise, rng_state=rng_state,
                class_id=warmup["class_id"], config=configs[method],
                guidance_channels=args.guidance_channels, measure=False,
            )
            del generated
    del noise, rng_state
    synchronize(device)

    elapsed_by_method: dict[str, list[float]] = {method: [] for method in args.methods}
    details: list[dict[str, object]] = []
    with tqdm(
        total=len(samples) * len(args.methods),
        desc="DiT reverse-denoising timing", unit="image",
    ) as progress:
        for row in samples:
            noise, rng_state = initial_noise_and_rng_state(
                device, latent_size, row["seed"]
            )
            for method in args.methods:
                generated, elapsed = run_once(
                    model=model, diffusion=diffusion, base_diffusion=base_diffusion,
                    device=device, noise=noise, rng_state=rng_state,
                    class_id=row["class_id"], config=configs[method],
                    guidance_channels=args.guidance_channels, measure=True,
                )
                elapsed_by_method[method].append(float(elapsed))
                details.append(
                    {
                        "method": method,
                        "sample_id": row["sample_id"],
                        "class_id": row["class_id"],
                        "seed": row["seed"],
                        "denoising_seconds": float(elapsed),
                    }
                )
                del generated
                progress.update(1)
            del noise, rng_state

    summaries = [summarize(method, elapsed_by_method[method]) for method in args.methods]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    details_output = args.details_output or args.output.with_name(
        f"{args.output.stem}_per_image.csv"
    )
    with details_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)

    metadata = {
        "measurement": "CUDA-synchronized DiT reverse-denoising latency only",
        "excluded": ["checkpoint loading", "VAE decoding", "image I/O"],
        "methods": list(args.methods),
        "sample_count_per_method": len(samples),
        "warmup_per_method": args.warmup,
        "guidance_scale": scale,
        "steps": args.steps,
        "image_size": args.image_size,
        "guidance_channels": args.guidance_channels,
        "cutoff": args.cutoff,
        "low_min_cfg": args.low_min_cfg,
        "high_min_cfg": args.high_min_cfg,
        "low_eta": args.low_eta,
        "high_eta": args.high_eta,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
    }
    metadata_path = args.output.with_name(f"{args.output.stem}_protocol.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2))
    print(f"Saved summary: {args.output}")
    print(f"Saved per-image measurements: {details_output}")
    print(f"Saved protocol: {metadata_path}")


if __name__ == "__main__":
    main()
