#!/usr/bin/env python3
"""EDM2 clean-prediction experiments using the official EDM2 generator."""

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import pickle
from pathlib import Path
import sys

import torch
from tqdm import tqdm

CODE_ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = Path(__file__).resolve().parent
for path in (CODE_ROOT, RELEASE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import REPORTED_METHODS, GuidanceConfig, GuidanceContext, GuidanceState, apply_model_guidance
from common.experiment import (
    add_guidance_args,
    guidance_config,
    implementation_fingerprint,
    method_protocol,
    parse_int_ranges,
    write_manifest,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edm2-root", type=Path, default=MODEL_ROOT / "vendor/edm2")
    parser.add_argument("--preset", default="edm2-img512-s-guid-dino")
    parser.add_argument("--outdir", type=Path, default=RELEASE_ROOT / "runs/edm2")
    parser.add_argument("--class-id", type=int, default=207)
    parser.add_argument("--seeds", type=parse_int_ranges, default=parse_int_ranges("0"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--max-batch-size", type=int, default=8)
    add_guidance_args(parser, default_scale=4.0, default_norm=2.5, default_momentum=-0.75)
    parser.set_defaults(low_min_cfg=1.0, high_min_cfg=3.0, low_eta=1.0, high_eta=0.25, cutoff=0.30)
    # The official guided presets specify their own recommended guidance.
    # A user-supplied --scales value intentionally overrides that preset in a
    # controlled scale sweep, whereas an omitted value reproduces the preset.
    parser.set_defaults(scales=None)
    return parser.parse_args()


def open_checkpoint(url, dnnlib, cache_dir):
    """Prefer an already verified local EDM2 cache entry over the network."""
    if cache_dir:
        cache_root = Path(cache_dir)
        digest = hashlib.md5(url.encode("utf-8")).hexdigest()
        encoded = url.replace("://", "_").replace("/", "_")
        basename = url.rsplit("/", 1)[-1]
        candidates = [
            cache_root / basename,
            cache_root / f"{digest}_{encoded}",
        ]
        candidates.extend(cache_root.glob(f"*_{basename}"))
        for path in candidates:
            if path.is_file() and path.stat().st_size > 0:
                print(f"Loading cached EDM2 checkpoint: {path}")
                return path.open("rb")
    kwargs = {"verbose": True}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    return dnnlib.util.open_url(url, **kwargs)


def load_samples(args):
    if args.manifest is None:
        rows = [
            {"sample_id": f"class{args.class_id:04d}_seed{s:06d}", "class_id": args.class_id, "seed": s}
            for s in args.seeds
        ]
    else:
        text = args.manifest.read_text(encoding="utf-8")
        rows = json.loads(text) if args.manifest.suffix.lower() == ".json" else list(csv.DictReader(text.splitlines()))
        rows = [{"sample_id": str(r["sample_id"]), "class_id": int(r["class_id"]), "seed": int(r["seed"])} for r in rows]
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("--max-samples must be positive")
        rows = rows[: args.max_samples]
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard arguments")
    seeds = [row["seed"] for row in rows]
    if len(seeds) != len(set(seeds)):
        raise ValueError(
            "EDM2 filenames are seed-based, so every manifest seed must be unique"
        )
    return rows


@torch.no_grad()
def shared_guidance_sampler(
    net,
    noise,
    labels=None,
    gnet=None,
    num_steps=32,
    sigma_min=0.002,
    sigma_max=80,
    rho=7,
    guidance_config_dict=None,
    dtype=torch.float32,
    randn_like=torch.randn_like,
    **_unused,
):
    del randn_like
    config = GuidanceConfig(**guidance_config_dict)
    state = GuidanceState()

    def denoise(x, sigma, update_state):
        clean_cond = net(x, sigma, labels).to(dtype)
        clean_uncond = gnet(x, sigma, labels).to(dtype)
        sigma_value = float(sigma)
        noise_time = math.log(max(sigma_value, sigma_min) / sigma_min) / math.log(sigma_max / sigma_min)
        return apply_model_guidance(
            clean_cond,
            clean_uncond,
            config,
            to_clean=lambda prediction: prediction,
            from_clean=lambda clean: clean,
            state=state,
            update_state=update_state,
            context=GuidanceContext(alpha=1.0, sigma=sigma_value, noise_time=noise_time),
        )

    indices = torch.arange(num_steps, dtype=dtype, device=noise.device)
    sigma_steps = (
        sigma_max ** (1 / rho)
        + indices / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    sigma_steps = torch.cat([sigma_steps, torch.zeros_like(sigma_steps[:1])])
    x_next = noise.to(dtype) * sigma_steps[0]

    for index, (sigma_cur, sigma_next) in enumerate(
        zip(sigma_steps[:-1], sigma_steps[1:])
    ):
        x_cur = x_next
        clean_cur = denoise(x_cur, sigma_cur, update_state=True)
        derivative = (x_cur - clean_cur) / sigma_cur
        x_euler = x_cur + (sigma_next - sigma_cur) * derivative
        if index < num_steps - 1:
            clean_prime = denoise(x_euler, sigma_next, update_state=False)
            derivative_prime = (x_euler - clean_prime) / sigma_next
            x_next = x_cur + (sigma_next - sigma_cur) * 0.5 * (
                derivative + derivative_prime
            )
        else:
            x_next = x_euler
    return x_next


def main():
    args = parse_args()
    sys.path.insert(0, str(args.edm2_root.resolve()))
    from generate_images import config_presets, generate_images
    import dnnlib
    from torch_utils import distributed as dist

    if args.preset not in config_presets:
        raise ValueError(f"Unknown EDM2 preset: {args.preset}")
    preset = config_presets[args.preset]
    if "gnet" not in preset:
        raise ValueError("Choose an EDM2 guided preset containing gnet")
    if args.scales is None:
        args.scales = [float(getattr(preset, "guidance", 1.0))]

    samples = load_samples(args)
    write_manifest(
        args.outdir,
        args,
        {
            "native_prediction": "clean_x0",
            "note": "EDM2 official conditional net plus official XS unconditional guide net",
            "method_domains": method_protocol(REPORTED_METHODS),
            "implementation_fingerprint": implementation_fingerprint(__file__),
            "sample_count": len(samples),
            "samples": samples,
        },
    )

    dist.init()
    device = torch.device("cuda")
    # Load each official checkpoint once. Calling generate_images with URL
    # strings inside the class/method loops would repeatedly deserialize the
    # 512-pixel networks and make a production run impractical.
    cache_dir = os.environ.get("DNNLIB_CACHE_DIR", str(MODEL_ROOT / "weights/cache"))
    with open_checkpoint(preset.net, dnnlib, cache_dir) as source:
        net_data = pickle.load(source)
    with open_checkpoint(preset.gnet, dnnlib, cache_dir) as source:
        gnet_data = pickle.load(source)
    net = net_data["ema"].to(device).eval()
    gnet = gnet_data["ema"].to(device).eval()
    encoder = net_data.get("encoder")
    if encoder is None:
        raise RuntimeError("EDM2 checkpoint does not contain its encoder")
    assigned_samples = samples[args.shard_index :: args.num_shards]
    by_class = {}
    for row in assigned_samples:
        by_class.setdefault(row["class_id"], []).append(row)
    for scale in args.scales:
        for method in args.methods:
            config = guidance_config(args, method, scale)
            method_dir = args.outdir / method / f"w{scale:g}"
            print(f"Generating {method}, w={scale:g} -> {method_dir}")
            for class_id, class_rows in tqdm(
                sorted(by_class.items()), desc=f"{method} classes", unit="class"
            ):
                pending = [
                    row
                    for row in class_rows
                    if not (
                        args.resume
                        and (method_dir / f"{row['seed']:06d}.png").is_file()
                        and (method_dir / f"{row['seed']:06d}.png").stat().st_size
                        > 1024
                    )
                ]
                if not pending:
                    continue
                iterator = generate_images(
                    net=net,
                    gnet=gnet,
                    encoder=encoder,
                    outdir=str(method_dir),
                    seeds=[r["seed"] for r in pending],
                    class_idx=class_id,
                    max_batch_size=args.max_batch_size,
                    sampler_fn=shared_guidance_sampler,
                    num_steps=args.steps,
                    guidance_config_dict=config.__dict__,
                    device=device,
                )
                for _batch in iterator:
                    pass
                del iterator
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
