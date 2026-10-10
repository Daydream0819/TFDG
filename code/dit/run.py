#!/usr/bin/env python3
"""Paired DiT-XL/2 experiments with reference-domain guidance."""

import argparse
import csv
import gc
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

CODE_ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = Path(__file__).resolve().parent
for path in (CODE_ROOT, RELEASE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import (
    REPORTED_METHODS,
    GuidanceContext,
    GuidanceState,
    apply_model_guidance,
    epsilon_to_x0,
    x0_to_epsilon,
)
from common.experiment import (
    add_guidance_args,
    guidance_config,
    implementation_fingerprint,
    method_protocol,
    parse_int_ranges,
    write_manifest,
)
from common.spectral import frequency_masks, spectral_geometry


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dit-root", type=Path, default=MODEL_ROOT / "vendor/DiT")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=MODEL_ROOT / "weights/DiT-XL-2-256x256.pt",
    )
    parser.add_argument("--vae", default=str(MODEL_ROOT / "weights/vae"))
    parser.add_argument("--outdir", type=Path, default=RELEASE_ROOT / "runs/dit")
    parser.add_argument("--image-size", choices=[256, 512], type=int, default=256)
    parser.add_argument("--class-ids", nargs="+", type=int, default=[207])
    parser.add_argument("--seeds", type=parse_int_ranges, default=parse_int_ranges("0"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--guidance-channels", choices=[3, 4], type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--online-vae", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument(
        "--geometry-output", type=Path, default=None,
        help=(
            "Write the unweighted clean-anchor residual geometry of a CFG "
            "trajectory to this JSON file. Requires exactly --methods cfg "
            "and one --scales value."
        ),
    )
    parser.add_argument(
        "--geometry-cutoffs", nargs="+", type=float,
        default=[0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
        help="Paper radial-frequency candidates for early energy allocation.",
    )
    add_guidance_args(parser, default_scale=4.0, default_norm=5.0, default_momentum=-0.5)
    return parser.parse_args()


def load_samples(args):
    """Load deterministic ``sample_id,class_id,seed`` rows for paired runs."""
    if args.manifest is None:
        rows = [
            {"sample_id": f"class{c:04d}_seed{s:06d}", "class_id": c, "seed": s}
            for c in args.class_ids
            for s in args.seeds
        ]
    else:
        text = args.manifest.read_text(encoding="utf-8")
        if args.manifest.suffix.lower() == ".json":
            rows = json.loads(text)
        else:
            rows = list(csv.DictReader(text.splitlines()))
        rows = [
            {
                "sample_id": str(row["sample_id"]),
                "class_id": int(row["class_id"]),
                "seed": int(row["seed"]),
            }
            for row in rows
        ]
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("--max-samples must be positive")
        rows = rows[: args.max_samples]
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard arguments")
    return rows


def extract_alpha_sigma(base_diffusion, t, reference):
    index = t.long().cpu().numpy()
    alpha_bar = torch.as_tensor(
        base_diffusion.alphas_cumprod[index],
        device=reference.device,
        dtype=reference.dtype,
    )
    return alpha_bar.sqrt(), (1.0 - alpha_bar).sqrt()


def rfft_energy(value):
    """Return full-spectrum-equivalent squared energy for an rFFT tensor."""
    energy = value.abs().square()
    width = (value.shape[-1] - 1) * 2
    if value.shape[-1] > 1:
        energy = energy.clone()
        if width % 2 == 0:
            energy[..., 1:-1] *= 2.0
        else:
            energy[..., 1:] *= 2.0
    return energy.sum(dim=tuple(range(1, energy.ndim)))


class GeometryRecorder:
    """Aggregate native clean-anchor residual geometry without retaining tensors."""

    def __init__(self, output, cutoffs, total_steps):
        if any(not 0 < cutoff <= 1 for cutoff in cutoffs):
            raise ValueError("geometry cutoffs must lie in (0, 1]")
        self.output = Path(output)
        self.cutoffs = tuple(sorted(set(float(cutoff) for cutoff in cutoffs)))
        self.total_steps = total_steps
        self.steps = [
            {
                "count": 0,
                "alpha": 0.0,
                "sigma": 0.0,
                "noise_time": 0.0,
                "radial": 0.0,
                "tangential": 0.0,
                "cutoffs": {
                    f"{cutoff:g}": {"low_radial": 0.0, "high_radial": 0.0}
                    for cutoff in self.cutoffs
                },
            }
            for _ in range(total_steps)
        ]

    @torch.no_grad()
    def record(self, clean_cond, clean_uncond, step_index, alpha, sigma, noise_time):
        if not 0 <= step_index < self.total_steps:
            raise ValueError("geometry recorder received an invalid sampler step")
        _, radial, tangential, _, _, _ = spectral_geometry(
            clean_cond, clean_uncond, cutoff=self.cutoffs[0]
        )
        entry = self.steps[step_index]
        count = int(clean_cond.shape[0])
        entry["count"] += count
        entry["alpha"] = float(alpha)
        entry["sigma"] = float(sigma)
        entry["noise_time"] = float(noise_time)
        entry["radial"] += float(rfft_energy(radial).sum().item())
        entry["tangential"] += float(rfft_energy(tangential).sum().item())
        for paper_rho in self.cutoffs:
            low_mask, high_mask = frequency_masks(clean_cond, 2.0 * paper_rho)
            bucket = entry["cutoffs"][f"{paper_rho:g}"]
            bucket["low_radial"] += float(rfft_energy(low_mask * radial).sum().item())
            bucket["high_radial"] += float(rfft_energy(high_mask * radial).sum().item())

    def write(self, metadata):
        if self.output.exists():
            raise FileExistsError(f"Refusing to overwrite geometry log: {self.output}")
        rows = []
        early_steps = max(1, math.ceil(self.total_steps * 0.25))
        for index, entry in enumerate(self.steps):
            if entry["count"] == 0:
                raise RuntimeError(f"geometry log has no samples for step {index}")
            count = entry["count"]
            row = {
                "step_index": index,
                "count": count,
                "alpha": entry["alpha"],
                "sigma": entry["sigma"],
                "noise_time": entry["noise_time"],
                "radial": entry["radial"] / count,
                "tangential": entry["tangential"] / count,
                "cutoffs": {},
            }
            for cutoff, bucket in entry["cutoffs"].items():
                low, high = bucket["low_radial"] / count, bucket["high_radial"] / count
                row["cutoffs"][cutoff] = {
                    "low_radial": low,
                    "high_radial": high,
                    "low_radial_fraction": low / max(low + high, 1e-12),
                }
            rows.append(row)
        early_fraction = {}
        for cutoff in rows[0]["cutoffs"]:
            low = sum(row["cutoffs"][cutoff]["low_radial"] for row in rows[:early_steps])
            high = sum(row["cutoffs"][cutoff]["high_radial"] for row in rows[:early_steps])
            early_fraction[cutoff] = low / max(low + high, 1e-12)
        payload = {
            **metadata,
            "definition": "unweighted clean-anchor residual: conditional clean prediction minus unconditional clean prediction",
            "energy": "full-spectrum-equivalent squared l2 energy of orthogonal rFFT components",
            "early_window_steps": early_steps,
            "early_low_radial_fraction": early_fraction,
            "steps": rows,
        }
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Saved geometry log: {self.output}")


class DiTGuidanceModel:
    def __init__(
        self,
        model,
        base_diffusion,
        class_id,
        config,
        guidance_channels,
        geometry_recorder=None,
    ):
        self.model = model
        self.base_diffusion = base_diffusion
        self.class_id = class_id
        self.config = config
        self.guidance_channels = guidance_channels
        self.state = GuidanceState()
        self.geometry_recorder = geometry_recorder
        self.geometry_step = 0

    @torch.inference_mode()
    def __call__(self, x, t, **_kwargs):
        batch = x.shape[0]
        x_pair = torch.cat([x, x], dim=0)
        t_pair = torch.cat([t, t], dim=0)
        labels = torch.cat(
            [
                torch.full((batch,), self.class_id, device=x.device, dtype=torch.long),
                torch.full((batch,), 1000, device=x.device, dtype=torch.long),
            ]
        )
        cond_out, uncond_out = self.model(x_pair, t_pair, labels).chunk(2, dim=0)
        eps_cond, variance_cond = cond_out[:, :4], cond_out[:, 4:]
        eps_uncond = uncond_out[:, :4]

        channels = self.guidance_channels
        x_guided = x[:, :channels]
        alpha, sigma = extract_alpha_sigma(self.base_diffusion, t, x_guided)
        if self.geometry_recorder is not None:
            clean_cond = epsilon_to_x0(x_guided, eps_cond[:, :channels], alpha, sigma)
            clean_uncond = epsilon_to_x0(x_guided, eps_uncond[:, :channels], alpha, sigma)
            self.geometry_recorder.record(
                clean_cond, clean_uncond, self.geometry_step,
                float(alpha[0]), float(sigma[0]), float(t[0]) / 999.0,
            )
            self.geometry_step += 1
        guided_part = apply_model_guidance(
            eps_cond[:, :channels],
            eps_uncond[:, :channels],
            self.config,
            to_clean=lambda prediction: epsilon_to_x0(
                x_guided, prediction, alpha, sigma
            ),
            from_clean=lambda clean: x0_to_epsilon(
                x_guided, clean, alpha, sigma
            ),
            state=self.state,
            context=GuidanceContext(
                alpha=float(alpha[0]),
                sigma=float(sigma[0]),
                noise_time=float(t[0]) / 999.0,
            ),
        )
        if channels < eps_cond.shape[1]:
            eps_guided = torch.cat([guided_part, eps_cond[:, channels:]], dim=1)
        else:
            eps_guided = guided_part
        return torch.cat([eps_guided, variance_cond], dim=1)


@torch.inference_mode()
def decode_vae(vae, latent):
    decoded = vae.decode(latent / 0.18215).sample
    decoded = (decoded / 2 + 0.5).clamp(0, 1)
    array = decoded[0].cpu().permute(1, 2, 0).numpy()
    return Image.fromarray((array * 255).astype(np.uint8))


def main():
    args = parse_args()
    if args.geometry_output is not None:
        if args.methods != ["cfg"] or len(args.scales) != 1:
            raise ValueError("--geometry-output requires exactly --methods cfg and one --scales value")
        if args.num_shards != 1:
            raise ValueError("--geometry-output does not support manifest sharding")
        if args.resume:
            raise ValueError("--geometry-output requires a fresh run without --resume")
        if args.geometry_output.exists():
            raise FileExistsError(f"Refusing to overwrite geometry log: {args.geometry_output}")
    if any(not 0 <= class_id < 1000 for class_id in args.class_ids):
        raise ValueError("ImageNet class ids must be in [0, 999]")
    samples = load_samples(args)
    write_manifest(
        args.outdir,
        args,
        {
            "native_prediction": "epsilon_plus_learned_variance",
            "note": "Only epsilon is guided; learned variance remains conditional.",
            "method_domains": method_protocol(REPORTED_METHODS),
            "implementation_fingerprint": implementation_fingerprint(__file__),
            "sample_count": len(samples),
            "samples": samples,
        },
    )
    sys.path.insert(0, str(args.dit_root.resolve()))
    from diffusers.models import AutoencoderKL
    from diffusion import create_diffusion
    from download import find_model
    from models import DiT_models

    device = torch.device(args.device)
    latent_size = args.image_size // 8
    model = DiT_models["DiT-XL/2"](
        input_size=latent_size,
        num_classes=1000,
        learn_sigma=True,
    ).to(device)
    state_dict = find_model(str(args.checkpoint))
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    vae = AutoencoderKL.from_pretrained(
        args.vae,
        local_files_only=not args.online_vae,
    ).to(device)
    diffusion = create_diffusion(str(args.steps))
    # The spaced sampler maps its local index back to the original 1000-step
    # timestep before calling the model. The original schedule is therefore
    # used for the epsilon <-> x0 conversion.
    base_diffusion = create_diffusion("")

    geometry_recorder = None
    if args.geometry_output is not None:
        geometry_recorder = GeometryRecorder(
            args.geometry_output, args.geometry_cutoffs, args.steps
        )

    assigned_samples = samples[args.shard_index :: args.num_shards]
    total_jobs = len(assigned_samples) * len(args.scales) * len(args.methods)
    progress = tqdm(total=total_jobs, desc="DiT generation", unit="image")
    for row in assigned_samples:
        class_id, seed, sample_id = row["class_id"], row["seed"], row["sample_id"]
        # ``p_sample_loop`` draws the DDPM transition noise from PyTorch's
        # default CUDA generator. Reset it for every manifest row so paired
        # methods remain reproducible even when they are launched separately.
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        else:
            torch.manual_seed(seed)
        generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn((1, 4, latent_size, latent_size), generator=generator, device=device)
        # Freeze the DDPM transition-noise stream once per manifest row. Every
        # scale and every method therefore receives the same initial latent and
        # the same stochastic sampler trajectory.
        base_rng_state = (
            torch.cuda.get_rng_state(device)
            if device.type == "cuda"
            else torch.random.get_rng_state()
        )
        for scale in args.scales:
            for method in args.methods:
                if device.type == "cuda":
                    torch.cuda.set_rng_state(base_rng_state, device)
                else:
                    torch.random.set_rng_state(base_rng_state)
                config = guidance_config(args, method, scale)
                guided_model = DiTGuidanceModel(
                    model, base_diffusion, class_id, config, args.guidance_channels,
                    geometry_recorder=geometry_recorder,
                )
                folder = args.outdir / method / f"w{scale:g}"
                folder.mkdir(parents=True, exist_ok=True)
                output_path = folder / f"{sample_id}.png"
                if args.resume and output_path.exists() and output_path.stat().st_size > 1024:
                    progress.update(1)
                    continue
                generated = diffusion.p_sample_loop(
                    guided_model, noise.shape, noise=noise.clone(), clip_denoised=False,
                    model_kwargs={}, progress=args.progress, device=device,
                )
                image = decode_vae(vae, generated)
                tmp_path = output_path.with_suffix(".tmp.png")
                image.save(tmp_path)
                tmp_path.replace(output_path)
                del guided_model, generated, image
                torch.cuda.empty_cache()
                progress.update(1)
        gc.collect()
    progress.close()
    if geometry_recorder is not None:
        geometry_recorder.write(
            {
                "backbone": "DiT-XL/2",
                "native_prediction": "epsilon_plus_learned_variance",
                "guided_epsilon_channels": args.guidance_channels,
                "guidance_method": "cfg",
                "guidance_scale": args.scales[0],
                "num_steps": args.steps,
                "sample_count": len(assigned_samples),
                "geometry_cutoffs": list(geometry_recorder.cutoffs),
            }
        )


if __name__ == "__main__":
    main()
