#!/usr/bin/env python3
"""Paired SD1.5 experiments with epsilon-to-clean-anchor TFDG."""

from __future__ import annotations

import argparse
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
    parse_int_ranges,
    method_protocol,
    torch_dtype,
    write_manifest,
)
from common.spectral import frequency_masks, spectral_geometry


DEFAULT_PROMPTS = [
    "A red fox standing in a snowy forest, realistic wildlife photography.",
    "A gothic cathedral interior with stained glass, realistic photography.",
    "A bowl of fresh fruit in natural light, highly detailed photography.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(MODEL_ROOT / "weights/sd15"))
    parser.add_argument("--outdir", type=Path, default=RELEASE_ROOT / "runs/sd15")
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--seeds", type=parse_int_ranges, default=parse_int_ranges("0"))
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--image-format", choices=["png", "jpg"], default="png")
    parser.add_argument("--jpeg-quality", type=int, default=95)
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
    add_guidance_args(parser, default_scale=12.0, default_norm=10.0, default_momentum=-0.5)
    return parser.parse_args()


def load_samples(args: argparse.Namespace) -> list[dict]:
    if args.manifest and (args.prompt or args.prompt_file):
        raise ValueError("--manifest cannot be combined with --prompt or --prompt-file")
    if args.manifest:
        rows = json.loads(args.manifest.read_text(encoding="utf-8"))
        rows = rows.get("samples", []) if isinstance(rows, dict) else rows
        samples = []
        for index, row in enumerate(rows):
            prompt = str(row.get("prompt", row.get("caption", ""))).strip()
            if not prompt:
                raise ValueError(f"manifest row {index} has no prompt")
            sample_id = str(row.get("sample_id", f"sample_{index:05d}"))
            samples.append({"sample_id": sample_id, "prompt": prompt, "seed": int(row.get("seed", 42 + index))})
    else:
        prompts = list(args.prompt)
        if args.prompt_file:
            text = args.prompt_file.read_text(encoding="utf-8")
            if args.prompt_file.suffix.lower() == ".json":
                prompts.extend(item["prompt"] if isinstance(item, dict) else str(item) for item in json.loads(text))
            else:
                prompts.extend(line.strip() for line in text.splitlines() if line.strip())
        prompts = prompts or DEFAULT_PROMPTS
        samples = [
            {"sample_id": f"prompt{index:04d}_seed{seed:06d}", "prompt": prompt, "seed": seed}
            for index, prompt in enumerate(prompts) for seed in args.seeds
        ]
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("--max-samples must be positive")
        samples = samples[:args.max_samples]
    if not samples:
        raise ValueError("no samples requested")
    if len({row["sample_id"] for row in samples}) != len(samples):
        raise ValueError("sample_id values must be unique")
    return samples


class SD15Adapter:
    def __init__(self, model_path: str, device: str, dtype: torch.dtype, online: bool):
        from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

        self.device = torch.device(device)
        self.dtype = dtype
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_path, torch_dtype=dtype, local_files_only=not online,
            variant="fp16", use_safetensors=True,
            safety_checker=None, feature_extractor=None,
            requires_safety_checker=False,
        )
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True,
        )
        self.pipe.to(self.device)

    @torch.inference_mode()
    def encode(self, prompt: str, negative_prompt: str):
        encoded = self.pipe.encode_prompt(
            prompt=prompt, negative_prompt=negative_prompt,
            device=self.device, num_images_per_prompt=1,
            do_classifier_free_guidance=True,
        )
        prompt_embeds, negative_embeds = encoded[:2]
        return prompt_embeds.to(self.device, dtype=self.dtype), negative_embeds.to(self.device, dtype=self.dtype)

    @torch.inference_mode()
    def predict(self, latent: torch.Tensor, timestep: torch.Tensor, context: tuple[torch.Tensor, torch.Tensor]):
        prompt_embeds, negative_embeds = context
        latent_pair = torch.cat([latent, latent], dim=0)
        latent_pair = self.pipe.scheduler.scale_model_input(latent_pair, timestep)
        embeddings = torch.cat([negative_embeds, prompt_embeds], dim=0)
        epsilon = self.pipe.unet(latent_pair, timestep, encoder_hidden_states=embeddings).sample
        return epsilon.chunk(2, dim=0)

    @torch.inference_mode()
    def decode(self, latent: torch.Tensor) -> Image.Image:
        image = self.pipe.vae.decode(latent / self.pipe.vae.config.scaling_factor).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        pixels = image[0].float().cpu().permute(1, 2, 0).numpy()
        return Image.fromarray((pixels * 255).round().astype(np.uint8))


def alpha_sigma(scheduler, timestep: torch.Tensor, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    alpha_bar = scheduler.alphas_cumprod.to(reference.device, dtype=reference.dtype)[timestep.long()]
    return alpha_bar.sqrt(), (1.0 - alpha_bar).sqrt()


def rfft_energy(value: torch.Tensor) -> torch.Tensor:
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

    def __init__(self, output: Path, cutoffs: list[float], total_steps: int):
        if any(not 0 < cutoff <= 1 for cutoff in cutoffs):
            raise ValueError("geometry cutoffs must lie in (0, 1]")
        self.output = Path(output)
        self.cutoffs = tuple(sorted(set(float(cutoff) for cutoff in cutoffs)))
        self.total_steps = total_steps
        self.steps = [
            {
                "count": 0, "alpha": 0.0, "sigma": 0.0, "noise_time": 0.0,
                "radial": 0.0, "tangential": 0.0,
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
            # ``frequency_masks`` receives a Nyquist-normalized diameter and
            # internally thresholds radius <= 0.5 * cutoff.  Geometry
            # candidates are recorded in the paper's radial coordinate rho.
            low_mask, high_mask = frequency_masks(clean_cond, 2.0 * paper_rho)
            bucket = entry["cutoffs"][f"{paper_rho:g}"]
            bucket["low_radial"] += float(rfft_energy(low_mask * radial).sum().item())
            bucket["high_radial"] += float(rfft_energy(high_mask * radial).sum().item())

    def write(self, metadata: dict) -> None:
        if self.output.exists():
            raise FileExistsError(f"Refusing to overwrite geometry log: {self.output}")
        rows = []
        early_steps = max(1, math.ceil(self.total_steps * 0.25))
        for index, entry in enumerate(self.steps):
            if entry["count"] == 0:
                raise RuntimeError(f"geometry log has no samples for step {index}")
            count = entry["count"]
            row = {
                "step_index": index, "count": count,
                "alpha": entry["alpha"], "sigma": entry["sigma"],
                "noise_time": entry["noise_time"],
                "radial": entry["radial"] / count,
                "tangential": entry["tangential"] / count,
                "cutoffs": {},
            }
            for cutoff, bucket in entry["cutoffs"].items():
                low, high = bucket["low_radial"] / count, bucket["high_radial"] / count
                row["cutoffs"][cutoff] = {
                    "low_radial": low, "high_radial": high,
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


@torch.inference_mode()
def sample(adapter: SD15Adapter, noise: torch.Tensor, context, config, steps: int, geometry_recorder=None) -> torch.Tensor:
    scheduler = adapter.pipe.scheduler
    scheduler.set_timesteps(steps, device=adapter.device)
    state = GuidanceState()
    latent = noise.clone()
    for step_index, timestep in enumerate(scheduler.timesteps):
        epsilon_uncond, epsilon_cond = adapter.predict(latent, timestep, context)
        alpha, sigma = alpha_sigma(scheduler, timestep, latent)
        if geometry_recorder is not None:
            geometry_recorder.record(
                epsilon_to_x0(latent, epsilon_cond, alpha, sigma),
                epsilon_to_x0(latent, epsilon_uncond, alpha, sigma),
                step_index, float(alpha), float(sigma), float(timestep) / 999.0,
            )
        guided = apply_model_guidance(
            epsilon_cond, epsilon_uncond, config,
            to_clean=lambda value: epsilon_to_x0(latent, value, alpha, sigma),
            from_clean=lambda value: x0_to_epsilon(latent, value, alpha, sigma),
            state=state,
            context=GuidanceContext(alpha=float(alpha), sigma=float(sigma), noise_time=float(timestep) / 999.0),
        )
        latent = scheduler.step(guided, timestep, latent).prev_sample
    return latent


def main() -> None:
    args = parse_args()
    if args.height % 8 or args.width % 8:
        raise ValueError("--height and --width must be divisible by 8")
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.geometry_output is not None:
        if args.methods != ["cfg"] or len(args.scales) != 1:
            raise ValueError("--geometry-output requires exactly --methods cfg and one --scales value")
        if args.resume:
            raise ValueError("--geometry-output requires a fresh run without --resume")
        if args.geometry_output.exists():
            raise FileExistsError(f"Refusing to overwrite geometry log: {args.geometry_output}")
    samples = load_samples(args)
    write_manifest(args.outdir, args, {
        "samples": samples,
        "sample_count": len(samples),
        "native_prediction": "epsilon",
        "sampler": "DPMSolverMultistepScheduler (DPM-Solver++)",
        "method_domains": method_protocol(REPORTED_METHODS),
        "implementation_fingerprint": implementation_fingerprint(__file__),
    })
    dtype = torch_dtype(args.dtype)
    adapter = SD15Adapter(args.model, args.device, dtype, args.online)
    geometry_recorder = (
        GeometryRecorder(args.geometry_output, args.geometry_cutoffs, args.steps)
        if args.geometry_output is not None else None
    )
    latent_shape = (1, adapter.pipe.unet.config.in_channels, args.height // 8, args.width // 8)
    jobs = len(samples) * len(args.scales) * len(args.methods)
    with tqdm(total=jobs, desc="SD1.5 generation", unit="image") as progress:
        for row in samples:
            context = adapter.encode(row["prompt"], args.negative_prompt)
            noise = torch.randn(latent_shape, generator=torch.Generator(device=adapter.device).manual_seed(row["seed"]), device=adapter.device, dtype=dtype)
            for scale in args.scales:
                for method in args.methods:
                    folder = args.outdir / method / f"w{scale:g}"
                    folder.mkdir(parents=True, exist_ok=True)
                    output = folder / f"{row['sample_id']}.{args.image_format}"
                    if args.resume and output.is_file() and output.stat().st_size > 1024:
                        progress.update(1)
                        continue
                    latent = sample(
                        adapter, noise, context, guidance_config(args, method, scale),
                        args.steps, geometry_recorder=geometry_recorder,
                    )
                    image = adapter.decode(latent)
                    temporary = output.with_name(f"{output.stem}.tmp{output.suffix}")
                    image.save(temporary, format="JPEG" if args.image_format == "jpg" else "PNG", **({"quality": args.jpeg_quality} if args.image_format == "jpg" else {}))
                    temporary.replace(output)
                    del latent, image
                    torch.cuda.empty_cache()
                    progress.update(1)
            del context, noise
            gc.collect()
    if geometry_recorder is not None:
        geometry_recorder.write({
            "backbone": "Stable Diffusion v1.5",
            "native_prediction": "epsilon",
            "guidance_method": "cfg",
            "guidance_scale": args.scales[0],
            "num_steps": args.steps,
            "sample_count": len(samples),
            "geometry_cutoffs": list(geometry_recorder.cutoffs),
        })


if __name__ == "__main__":
    main()
