#!/usr/bin/env python3
"""Paired SD3.5 Large experiments with reference-domain guidance."""

import argparse
import gc
import json
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
    flow_velocity_to_x0,
    x0_to_flow_velocity,
)
from common.experiment import (
    add_guidance_args,
    guidance_config,
    implementation_fingerprint,
    method_protocol,
    parse_int_ranges,
    torch_dtype,
    write_manifest,
)


DEFAULT_PROMPTS = [
    "A red fox standing in a snowy forest, detailed fur, realistic wildlife photography",
    "A gothic cathedral interior with stained glass windows, realistic photography",
    "A bowl of fresh fruit in natural light, highly detailed photography",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(MODEL_ROOT / "weights/sd35-large"))
    parser.add_argument("--outdir", type=Path, default=RELEASE_ROOT / "runs/sd35")
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="JSON rows with sample_id, prompt, and seed for paired evaluation",
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--seeds", type=parse_int_ranges, default=parse_int_ranges("0"))
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--online", action="store_true", help="Allow Hub access instead of local-only loading")
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--image-format", choices=["png", "jpg"], default="png")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use sequential CPU offload when GPU memory is limited",
    )
    add_guidance_args(parser, default_scale=15.0, default_norm=15.0, default_momentum=-0.5)
    return parser.parse_args()


def load_samples(args):
    if args.manifest and (args.prompt or args.prompt_file):
        raise ValueError("--manifest cannot be combined with --prompt/--prompt-file")
    if args.manifest:
        rows = json.loads(args.manifest.read_text(encoding="utf-8"))
        if isinstance(rows, dict):
            rows = rows.get("samples", [])
        samples = []
        for index, row in enumerate(rows):
            generated_index = str(row.get("generated_index", index)).zfill(5)
            sample_id = str(row.get("sample_id", f"coco_{generated_index}"))
            prompt = str(row.get("prompt", row.get("caption", ""))).strip()
            if not prompt:
                raise ValueError(f"Sample {sample_id} has no prompt/caption")
            samples.append(
                {
                    "sample_id": sample_id,
                    "prompt": prompt,
                    "seed": int(row.get("seed", 42 + index)),
                }
            )
        if args.max_samples is not None:
            if args.max_samples < 1:
                raise ValueError("--max-samples must be positive")
            samples = samples[: args.max_samples]
        if not samples:
            raise ValueError(f"No samples found in {args.manifest}")
        return samples

    prompts = list(args.prompt)
    if args.prompt_file:
        text = args.prompt_file.read_text(encoding="utf-8")
        if args.prompt_file.suffix.lower() == ".json":
            data = json.loads(text)
            prompts.extend(item["prompt"] if isinstance(item, dict) else item for item in data)
        else:
            prompts.extend(line.strip() for line in text.splitlines() if line.strip())
    prompts = prompts or DEFAULT_PROMPTS
    samples = [
        {
            "sample_id": f"prompt{index:04d}_seed{seed:06d}",
            "prompt": prompt,
            "seed": seed,
        }
        for index, prompt in enumerate(prompts)
        for seed in args.seeds
    ]
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("--max-samples must be positive")
        samples = samples[: args.max_samples]
    return samples


class SD35Adapter:
    def __init__(self, model, device, dtype, online, cpu_offload=True):
        from diffusers import StableDiffusion3Pipeline

        self.device = torch.device(device)
        self.dtype = dtype
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            model,
            torch_dtype=dtype,
            local_files_only=not online,
        )
        # Keep VAE decoding in float32, matching the verified single-GPU
        # implementation and avoiding mixed input/bias dtypes under offload.
        self.pipe.vae.to(dtype=torch.float32)
        self.cpu_offload = bool(cpu_offload and self.device.type == "cuda")
        if self.cpu_offload:
            self.pipe.enable_sequential_cpu_offload(gpu_id=self.device.index or 0)
            self.transformer = self.pipe.transformer
            self.vae = self.pipe.vae
        else:
            # Match the previously validated single-GPU implementation.
            self.transformer = self.pipe.transformer.to(self.device, dtype=dtype)
            self.vae = self.pipe.vae.to(self.device, dtype=torch.float32)

    @torch.inference_mode()
    def encode(self, prompt, negative_prompt, max_length):
        if not self.cpu_offload:
            for name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
                encoder = getattr(self.pipe, name, None)
                if encoder is not None:
                    encoder.to(self.device)
        output = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_3=prompt,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt,
            negative_prompt_3=negative_prompt,
            device=self.device,
            do_classifier_free_guidance=True,
            max_sequence_length=max_length,
        )
        if not self.cpu_offload:
            for name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
                encoder = getattr(self.pipe, name, None)
                if encoder is not None:
                    encoder.to("cpu")
        torch.cuda.empty_cache()
        return tuple(tensor.to(self.device, dtype=self.dtype) for tensor in output)

    @torch.inference_mode()
    def predict(self, z_t, t, context):
        prompt, negative, pooled, negative_pooled = context
        z_pair = torch.cat([z_t, z_t], dim=0)
        t_pair = torch.full(
            (z_pair.shape[0],),
            float(t) * 1000.0,
            device=self.device,
            dtype=self.dtype,
        )
        hidden = torch.cat([negative, prompt], dim=0)
        pooled_pair = torch.cat([negative_pooled, pooled], dim=0)
        velocity = self.transformer(
            hidden_states=z_pair,
            timestep=t_pair,
            encoder_hidden_states=hidden,
            pooled_projections=pooled_pair,
            return_dict=False,
        )[0]
        return velocity.chunk(2, dim=0)  # uncond, cond

    @torch.inference_mode()
    def decode(self, latent):
        shift = self.vae.config.shift_factor
        scale = self.vae.config.scaling_factor
        decoded = self.vae.decode((latent / scale + shift).float()).sample
        decoded = (decoded / 2 + 0.5).clamp(0, 1)
        array = decoded[0].cpu().permute(1, 2, 0).numpy()
        return Image.fromarray((array * 255).astype(np.uint8))


def shifted_timesteps(steps, shift, device, dtype):
    sequence = torch.linspace(0, 1, steps + 1, device=device, dtype=dtype)
    sequence = (shift * sequence) / (1 + (shift - 1) * sequence)
    return torch.flip(sequence, dims=[0])


@torch.inference_mode()
def sample(adapter, noise, context, config, steps, shift):
    state = GuidanceState()
    z_t = noise.clone()
    timesteps = shifted_timesteps(steps, shift, adapter.device, adapter.dtype)
    for current, following in zip(timesteps[:-1], timesteps[1:]):
        v_uncond, v_cond = adapter.predict(z_t, current, context)
        guidance_context = GuidanceContext(
            alpha=float(1.0 - current),
            sigma=float(current),
            noise_time=float(current),
        )
        velocity = apply_model_guidance(
            v_cond,
            v_uncond,
            config,
            to_clean=lambda prediction: flow_velocity_to_x0(
                z_t, prediction, current
            ),
            from_clean=lambda clean: x0_to_flow_velocity(z_t, clean, current),
            state=state,
            context=guidance_context,
        )
        z_t = z_t + (following - current) * velocity
    return z_t


def main():
    args = parse_args()
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be in [1, 100]")
    samples = load_samples(args)
    write_manifest(
        args.outdir,
        args,
        {
            "samples": samples,
            "sample_count": len(samples),
            "native_prediction": "flow_velocity",
            "method_domains": method_protocol(REPORTED_METHODS),
            "implementation_fingerprint": implementation_fingerprint(__file__),
        },
    )
    dtype = torch_dtype(args.dtype)
    adapter = SD35Adapter(args.model, args.device, dtype, args.online, args.cpu_offload)
    latent_h, latent_w = args.height // 8, args.width // 8

    total_jobs = len(samples) * len(args.scales) * len(args.methods)
    with tqdm(total=total_jobs, desc="SD3.5 generation", unit="image") as progress:
        for sample_row in samples:
            prompt = sample_row["prompt"]
            seed = sample_row["seed"]
            sample_id = sample_row["sample_id"]
            context = adapter.encode(prompt, args.negative_prompt, args.max_sequence_length)
            generator = torch.Generator(device=adapter.device).manual_seed(seed)
            noise = torch.randn(
                (1, 16, latent_h, latent_w),
                generator=generator,
                device=adapter.device,
                dtype=dtype,
            )
            for scale in args.scales:
                for method in args.methods:
                    config = guidance_config(args, method, scale)
                    folder = args.outdir / method / f"w{scale:g}"
                    folder.mkdir(parents=True, exist_ok=True)
                    output_path = folder / f"{sample_id}.{args.image_format}"
                    if args.resume and output_path.is_file() and output_path.stat().st_size > 1024:
                        progress.update(1)
                        continue
                    latent = sample(adapter, noise, context, config, args.steps, args.shift)
                    image = adapter.decode(latent)
                    temporary = output_path.with_name(
                        f"{output_path.stem}.tmp{output_path.suffix}"
                    )
                    if args.image_format == "jpg":
                        image.save(temporary, format="JPEG", quality=args.jpeg_quality)
                    else:
                        image.save(temporary, format="PNG")
                    temporary.replace(output_path)
                    del latent, image
                    torch.cuda.empty_cache()
                    progress.update(1)
            del context, noise
            gc.collect()


if __name__ == "__main__":
    main()
