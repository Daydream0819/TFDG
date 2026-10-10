#!/usr/bin/env python3
"""Capture one reproducible SD3.5 TFDG trajectory for the method overview.

The script runs one prompt--seed pair through the published SD3.5 sampling
configuration.  It saves every latent state in the trajectory and dumps the
clean-anchor, spectral-decomposition, temporal-weight, and sampler quantities
for a small set of candidate denoising steps.  All saved quantities therefore
belong to one trajectory; no intermediate image is regenerated independently.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from common import (  # noqa: E402
    GuidanceConfig,
    GuidanceContext,
    apply_clean_guidance,
    decompose_residual,
    flow_velocity_to_x0,
    x0_to_flow_velocity,
)
from common.experiment import implementation_fingerprint, torch_dtype  # noqa: E402
from sd35.run import SD35Adapter, shifted_timesteps  # noqa: E402


DEFAULT_PROMPT = (
    "A scarlet macaw perched on a moss-covered branch in a tropical rainforest, "
    "vivid red, blue, and yellow feathers, soft morning mist, detailed feather "
    "texture, natural dappled sunlight, realistic wildlife photography."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(MODEL_ROOT / "weights/sd35-large"))
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        help="Optional JSON manifest containing exactly one sample")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=1048)
    parser.add_argument("--sample-id", default="fig1_macaw_seed001048")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--candidate-steps", nargs="+", type=int,
                        default=(5, 7, 9, 11))
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--scale", type=float, default=15.0)
    parser.add_argument("--cutoff", type=float, default=0.30,
                        help="Implementation cutoff; paper radius is cutoff / 2")
    parser.add_argument("--low-min-cfg", type=float, default=1.0)
    parser.add_argument("--low-eta", type=float, default=0.5)
    parser.add_argument("--high-min-cfg", type=float, default=3.0)
    parser.add_argument("--high-eta", type=float, default=1.0)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"),
                        default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--cpu-offload", action=argparse.BooleanOptionalAction,
                        default=False)
    args = parser.parse_args()

    if args.steps < 2:
        parser.error("--steps must be at least two")
    if args.height % 8 or args.width % 8:
        parser.error("--height and --width must be divisible by eight")
    if args.scale < 1:
        parser.error("--scale must be at least one")
    if not 0 < args.cutoff <= 1:
        parser.error("--cutoff must lie in (0, 1]")
    if not 1 <= args.low_min_cfg <= args.scale:
        parser.error("--low-min-cfg must lie in [1, --scale]")
    if not 1 <= args.high_min_cfg <= args.scale:
        parser.error("--high-min-cfg must lie in [1, --scale]")
    if args.low_eta < 0 or args.high_eta < 0:
        parser.error("--low-eta and --high-eta must be non-negative")
    candidate_steps = sorted(set(args.candidate_steps))
    if any(step < 0 or step >= args.steps for step in candidate_steps):
        parser.error("candidate steps must lie in [0, steps - 1]")
    args.candidate_steps = candidate_steps
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def load_sample(args: argparse.Namespace) -> tuple[dict, dict | None]:
    if args.manifest is None:
        return {
            "sample_id": args.sample_id,
            "prompt": args.prompt,
            "seed": args.seed,
        }, None

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = payload.get("samples", []) if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or len(rows) != 1:
        raise ValueError("--manifest must contain exactly one sample")
    row = rows[0]
    prompt = str(row.get("prompt", row.get("caption", ""))).strip()
    if not prompt:
        raise ValueError("The manifest sample has no prompt/caption")
    sample = {
        "sample_id": str(row.get("sample_id", args.sample_id)),
        "prompt": prompt,
        "seed": int(row.get("seed", args.seed)),
    }
    source = {
        "path": str(args.manifest.resolve()),
        "sha256": sha256_file(args.manifest),
    }
    return sample, source


def require_empty_outdir(outdir: Path) -> None:
    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(
            f"Refusing to reuse non-empty output directory: {outdir}. "
            "Choose a new directory so the visualization protocol remains immutable."
        )
    outdir.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def save_tensor(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.detach().cpu(), path)


def save_image(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    image.save(temporary, format="PNG")
    temporary.replace(path)


def scalar_map_to_image(array: np.ndarray) -> Image.Image:
    array = np.asarray(array, dtype=np.float32)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError("Cannot visualize an array without finite values")
    low, high = np.percentile(finite, (1.0, 99.5))
    if not high > low:
        low = float(finite.min())
        high = float(finite.max())
    normalized = np.zeros_like(array) if not high > low else (array - low) / (high - low)
    normalized = np.clip(normalized, 0.0, 1.0)
    return Image.fromarray(np.round(255.0 * normalized).astype(np.uint8), mode="L")


def spatial_magnitude(tensor: torch.Tensor) -> np.ndarray:
    magnitude = torch.sqrt(torch.sum(tensor.detach().float().square(), dim=1))[0]
    return magnitude.cpu().numpy()


def spectrum_magnitude(
    spectrum: torch.Tensor, spatial_shape: tuple[int, int]
) -> np.ndarray:
    spatial = torch.fft.irfft2(spectrum, s=spatial_shape, norm="ortho")
    full = torch.fft.fftshift(
        torch.fft.fft2(spatial.float(), norm="ortho"), dim=(-2, -1)
    )
    magnitude = torch.sqrt(torch.sum(torch.abs(full).square(), dim=1))[0]
    return torch.log1p(magnitude).cpu().numpy()


def save_map(path_without_suffix: Path, array: np.ndarray) -> None:
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    np.save(path_without_suffix.with_suffix(".npy"), np.asarray(array, dtype=np.float32))
    save_image(path_without_suffix.with_suffix(".png"), scalar_map_to_image(array))


def rfft_multiplicity(width: int, reference: torch.Tensor) -> torch.Tensor:
    coefficients = reference.shape[-1]
    weights = torch.full(
        (coefficients,), 2.0, dtype=torch.float64, device=reference.device
    )
    weights[0] = 1.0
    if width % 2 == 0:
        weights[-1] = 1.0
    return weights.view(*([1] * (reference.ndim - 1)), coefficients)


def rfft_energy(spectrum: torch.Tensor, spatial_width: int) -> float:
    weights = rfft_multiplicity(spatial_width, spectrum)
    # Take the complex magnitude before converting to float64. Calling
    # ``complex_tensor.double()`` would silently discard the imaginary part.
    energy = (torch.abs(spectrum).double().square() * weights).sum()
    return float(energy.item())


def relative_l2_error(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    # This helper is used for both real anchors and complex rFFT tensors.
    # Computing the magnitude first preserves both real and imaginary parts.
    numerator = torch.sqrt(torch.abs(reference - estimate).double().square().sum())
    denominator = torch.sqrt(torch.abs(reference).double().square().sum()).clamp_min(1e-30)
    return float((numerator / denominator).item())


def temporal_weight(
    minimum: float, eta: float, scale: float, alpha: float, sigma: float
) -> float:
    if eta == 0:
        recovery = 1.0
    else:
        denominator = math.sqrt(alpha * alpha + eta * sigma * sigma)
        recovery = abs(alpha) / max(denominator, 1e-12)
    return minimum + (scale - minimum) * recovery


def make_contact_sheet(
    image_paths: list[Path], labels: list[str], output: Path, columns: int = 7
) -> None:
    if not image_paths:
        return
    tile_size = 176
    label_height = 24
    rows = math.ceil(len(image_paths) / columns)
    canvas = Image.new("RGB", (columns * tile_size, rows * (tile_size + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (path, label) in enumerate(zip(image_paths, labels)):
        image = Image.open(path).convert("RGB")
        image.thumbnail((tile_size, tile_size), Image.Resampling.LANCZOS)
        x = (index % columns) * tile_size + (tile_size - image.width) // 2
        y0 = (index // columns) * (tile_size + label_height)
        y = y0 + (tile_size - image.height) // 2
        canvas.paste(image, (x, y))
        draw.text((x + 4, y0 + tile_size + 4), label, fill=(20, 42, 83))
    save_image(output, canvas)


def dump_candidate(
    step_dir: Path,
    *,
    z_t: torch.Tensor,
    z_next: torch.Tensor,
    v_cond: torch.Tensor,
    v_uncond: torch.Tensor,
    clean_cond: torch.Tensor,
    clean_uncond: torch.Tensor,
    clean_tfdg: torch.Tensor,
    velocity_tfdg: torch.Tensor,
    delta_x0: torch.Tensor,
    delta_spectrum: torch.Tensor,
    components: dict[str, torch.Tensor],
    tangential: torch.Tensor,
    weighted: dict[str, torch.Tensor],
    guided_spectrum: torch.Tensor,
    guided_update: torch.Tensor,
    adapter: SD35Adapter,
    metrics: dict,
) -> None:
    step_dir.mkdir(parents=True, exist_ok=False)
    # Keep the complete radial branch explicit in the Fig. 1 dump.  The
    # low/high split is applied only after this relative-phase projection.
    radial = components["low_radial"] + components["high_radial"]
    tensors = {
        "x_t": z_t,
        "x_t_minus_1": z_next,
        "v_cond": v_cond,
        "v_uncond": v_uncond,
        "x0_cond": clean_cond,
        "x0_uncond": clean_uncond,
        "delta_x0": delta_x0,
        "delta_X": delta_spectrum,
        "delta_X_r": radial,
        "delta_X_tau": tangential,
        "delta_X_r_low": components["low_radial"],
        "delta_X_r_high": components["high_radial"],
        "delta_X_tau_low": components["low_tangential"],
        "delta_X_tau_high": components["high_tangential"],
        "weighted_tau": weighted["tau"],
        "weighted_r_low": weighted["r_low"],
        "weighted_r_high": weighted["r_high"],
        "delta_X_tfdg": guided_spectrum,
        "delta_x0_tfdg": guided_update,
        "x0_tfdg": clean_tfdg,
        "v_tfdg": velocity_tfdg,
    }
    for name, tensor in tensors.items():
        save_tensor(step_dir / f"{name}.pt", tensor)

    decoded = {
        "x_t_decoded.png": adapter.decode(z_t),
        "x_t_minus_1_decoded.png": adapter.decode(z_next),
        "x0_cond_decoded.png": adapter.decode(clean_cond),
        "x0_uncond_decoded.png": adapter.decode(clean_uncond),
        "x0_tfdg_decoded.png": adapter.decode(clean_tfdg),
    }
    for name, image in decoded.items():
        save_image(step_dir / name, image)

    spatial_shape = tuple(delta_x0.shape[-2:])
    save_map(step_dir / "delta_x0_magnitude", spatial_magnitude(delta_x0))
    for name, spectrum in {
        "delta_X_magnitude": delta_spectrum,
        "delta_X_r_magnitude": radial,
        "delta_X_tau_magnitude": tangential,
        "delta_X_r_low_magnitude": components["low_radial"],
        "delta_X_r_high_magnitude": components["high_radial"],
        "weighted_tau_magnitude": weighted["tau"],
        "weighted_r_low_magnitude": weighted["r_low"],
        "weighted_r_high_magnitude": weighted["r_high"],
        "delta_X_tfdg_magnitude": guided_spectrum,
    }.items():
        save_map(step_dir / name, spectrum_magnitude(spectrum, spatial_shape))
    save_map(step_dir / "delta_x0_tfdg_magnitude", spatial_magnitude(guided_update))
    save_json(step_dir / "step_metadata.json", metrics)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    require_empty_outdir(args.outdir)
    sample, source_manifest = load_sample(args)

    config = GuidanceConfig(
        method="tfdg",
        scale=args.scale,
        cutoff=args.cutoff,
        low_min_cfg=args.low_min_cfg,
        low_eta=args.low_eta,
        high_min_cfg=args.high_min_cfg,
        high_eta=args.high_eta,
    )
    dtype = torch_dtype(args.dtype)
    adapter = SD35Adapter(
        args.model, args.device, dtype, args.online, args.cpu_offload
    )
    context = adapter.encode(
        sample["prompt"], args.negative_prompt, args.max_sequence_length
    )
    generator = torch.Generator(device=adapter.device).manual_seed(sample["seed"])
    z_t = torch.randn(
        (1, 16, args.height // 8, args.width // 8),
        generator=generator,
        device=adapter.device,
        dtype=dtype,
    )
    timesteps = shifted_timesteps(args.steps, args.shift, adapter.device, dtype)

    trajectory_dir = args.outdir / "trajectory"
    candidates_dir = args.outdir / "candidate_steps"
    trajectory_rows: list[dict] = []
    energy_rows: list[dict] = []
    weight_rows: list[dict] = []
    trajectory_images: list[Path] = []
    trajectory_labels: list[str] = []
    candidate_pair_paths: list[Path] = []
    candidate_pair_labels: list[str] = []

    def save_trajectory_state(state_index: int, time_value: float, latent: torch.Tensor) -> None:
        stem = f"state_{state_index:02d}_t{time_value:.6f}"
        latent_path = trajectory_dir / f"{stem}_latent.pt"
        image_path = trajectory_dir / f"{stem}_decoded.png"
        save_tensor(latent_path, latent)
        save_image(image_path, adapter.decode(latent))
        trajectory_rows.append({
            "state_index": state_index,
            "t_scheduler": time_value,
            "latent": str(latent_path.relative_to(args.outdir)),
            "decoded": str(image_path.relative_to(args.outdir)),
        })
        trajectory_images.append(image_path)
        trajectory_labels.append(f"state {state_index}, t={time_value:.3f}")

    save_trajectory_state(0, float(timesteps[0]), z_t)

    for step_index, (current, following) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        current_value = float(current)
        following_value = float(following)
        alpha = 1.0 - current_value
        sigma = current_value
        guidance_context = GuidanceContext(
            alpha=alpha,
            sigma=sigma,
            noise_time=current_value,
            step_index=step_index,
            total_steps=args.steps,
        )

        v_uncond, v_cond = adapter.predict(z_t, current, context)
        clean_cond = flow_velocity_to_x0(z_t, v_cond, current)
        clean_uncond = flow_velocity_to_x0(z_t, v_uncond, current)
        delta_x0 = clean_cond - clean_uncond
        delta_spectrum = torch.fft.rfft2(delta_x0.float(), norm="ortho")
        _, components = decompose_residual(
            clean_cond, clean_uncond, cutoff=args.cutoff
        )
        tangential = components["low_tangential"] + components["high_tangential"]
        reconstructed = (
            tangential + components["low_radial"] + components["high_radial"]
        )

        low_weight = temporal_weight(
            args.low_min_cfg, args.low_eta, args.scale, alpha, sigma
        )
        high_weight = temporal_weight(
            args.high_min_cfg, args.high_eta, args.scale, alpha, sigma
        )
        tangential_weight = args.scale
        weighted = {
            "tau": (tangential_weight - 1.0) * tangential,
            "r_low": (low_weight - 1.0) * components["low_radial"],
            "r_high": (high_weight - 1.0) * components["high_radial"],
        }
        guided_spectrum = weighted["tau"] + weighted["r_low"] + weighted["r_high"]
        guided_update = torch.fft.irfft2(
            guided_spectrum, s=clean_cond.shape[-2:], norm="ortho"
        )
        # Match ``common.guidance._tfdg`` exactly: add the float32 inverse-FFT
        # update first, then cast the completed anchor back to the model dtype.
        clean_tfdg_from_components = (clean_cond + guided_update).to(clean_cond.dtype)
        clean_tfdg = apply_clean_guidance(
            clean_cond,
            clean_uncond,
            config,
            context=guidance_context,
        )
        decomposition_error = relative_l2_error(delta_spectrum, reconstructed)
        guided_anchor_error = relative_l2_error(
            clean_tfdg, clean_tfdg_from_components
        )
        if decomposition_error > 1e-5:
            raise RuntimeError(
                f"Step {step_index}: decomposition reconstruction error "
                f"{decomposition_error:.3e} exceeds tolerance"
            )
        if guided_anchor_error > 1e-5:
            raise RuntimeError(
                f"Step {step_index}: dumped TFDG anchor differs from the live "
                f"implementation by {guided_anchor_error:.3e}"
            )

        velocity_tfdg = x0_to_flow_velocity(z_t, clean_tfdg, current)
        z_next = z_t + (following - current) * velocity_tfdg

        height, width = delta_x0.shape[-2:]
        energies = {
            "E_total": rfft_energy(delta_spectrum, width),
            "E_tangential": rfft_energy(tangential, width),
            "E_r_low": rfft_energy(components["low_radial"], width),
            "E_r_high": rfft_energy(components["high_radial"], width),
        }
        partition_sum = (
            energies["E_tangential"] + energies["E_r_low"] + energies["E_r_high"]
        )
        energy_row = {
            "step_index": step_index,
            "t_scheduler": current_value,
            **energies,
            "radial_fraction": (
                energies["E_r_low"] + energies["E_r_high"]
            ) / max(energies["E_total"], 1e-30),
            "low_radial_fraction_of_radial": energies["E_r_low"] / max(
                energies["E_r_low"] + energies["E_r_high"], 1e-30
            ),
            "energy_partition_relative_error": abs(
                energies["E_total"] - partition_sum
            ) / max(energies["E_total"], 1e-30),
            "spectral_reconstruction_relative_error": decomposition_error,
        }
        weight_row = {
            "step_index": step_index,
            "t_scheduler": current_value,
            "alpha": alpha,
            "sigma": sigma,
            "w_tangential": tangential_weight,
            "w_r_low": low_weight,
            "w_r_high": high_weight,
            "multiplier_tangential": tangential_weight - 1.0,
            "multiplier_r_low": low_weight - 1.0,
            "multiplier_r_high": high_weight - 1.0,
        }
        energy_rows.append(energy_row)
        weight_rows.append(weight_row)

        if step_index in args.candidate_steps:
            step_dir = candidates_dir / f"step_{step_index:02d}_t{current_value:.6f}"
            metrics = {
                "sample": sample,
                "step_index": step_index,
                "t_scheduler": current_value,
                "t_next": following_value,
                "weights": weight_row,
                "energies": energy_row,
                "checks": {
                    "spectral_reconstruction_relative_error": decomposition_error,
                    "guided_anchor_relative_error": guided_anchor_error,
                    "next_state_is_live_trajectory_state": True,
                },
            }
            dump_candidate(
                step_dir,
                z_t=z_t,
                z_next=z_next,
                v_cond=v_cond,
                v_uncond=v_uncond,
                clean_cond=clean_cond,
                clean_uncond=clean_uncond,
                clean_tfdg=clean_tfdg,
                velocity_tfdg=velocity_tfdg,
                delta_x0=delta_x0,
                delta_spectrum=delta_spectrum,
                components=components,
                tangential=tangential,
                weighted=weighted,
                guided_spectrum=guided_spectrum,
                guided_update=guided_update,
                adapter=adapter,
                metrics=metrics,
            )
            for name in ("x_t_decoded.png", "x_t_minus_1_decoded.png"):
                candidate_pair_paths.append(step_dir / name)
                candidate_pair_labels.append(f"step {step_index}: {name.removesuffix('_decoded.png')}")

        z_t = z_next
        save_trajectory_state(step_index + 1, following_value, z_t)
        print(
            f"[step {step_index:02d}] t={current_value:.6f} -> {following_value:.6f}; "
            f"wL={low_weight:.4f}, wH={high_weight:.4f}; "
            f"reconstruction={decomposition_error:.2e}"
        )

    for filename, rows in (
        ("trajectory_index.csv", trajectory_rows),
        ("component_energy.csv", energy_rows),
        ("guidance_weights.csv", weight_rows),
    ):
        with (args.outdir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    make_contact_sheet(
        trajectory_images,
        trajectory_labels,
        args.outdir / "trajectory_contact_sheet.png",
        columns=7,
    )
    make_contact_sheet(
        candidate_pair_paths,
        candidate_pair_labels,
        args.outdir / "candidate_step_pairs.png",
        columns=2,
    )

    metadata = {
        "purpose": "Figure 1 single-trajectory visualization material",
        "trajectory_method": "tfdg",
        "sample": sample,
        "source_manifest": source_manifest,
        "model": str(Path(args.model).resolve()),
        "native_prediction": "flow_matching_velocity",
        "sampler": "Euler",
        "steps": args.steps,
        "trajectory_state_count": args.steps + 1,
        "scheduler_shift": args.shift,
        "resolution": [args.height, args.width],
        "dtype": args.dtype,
        "seed_rule": "single fixed seed",
        "candidate_step_indices": args.candidate_steps,
        "tfdg": {
            "global_cfg_weight": args.scale,
            "implementation_cutoff": args.cutoff,
            "paper_radial_cutoff": args.cutoff / 2.0,
            "low_min_cfg": args.low_min_cfg,
            "low_eta": args.low_eta,
            "high_min_cfg": args.high_min_cfg,
            "high_eta": args.high_eta,
            "tangential_weight": args.scale,
            "residual_multiplier_convention": "effective CFG weight minus one",
        },
        "fft": {
            "transform": "torch.fft.rfft2",
            "normalization": "ortho",
            "energy": "Hermitian multiplicity corrected one-sided spectrum",
            "visualization": "full centered FFT log-magnitude reconstructed from saved rFFT tensor",
        },
        "implementation_fingerprint": implementation_fingerprint(__file__),
        "software": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(adapter.device),
            "gpu": (
                torch.cuda.get_device_name(adapter.device)
                if adapter.device.type == "cuda" else None
            ),
        },
        "outputs": {
            "trajectory": "trajectory/",
            "candidate_steps": "candidate_steps/",
            "trajectory_index": "trajectory_index.csv",
            "component_energy": "component_energy.csv",
            "guidance_weights": "guidance_weights.csv",
            "trajectory_preview": "trajectory_contact_sheet.png",
            "candidate_preview": "candidate_step_pairs.png",
        },
    }
    save_json(args.outdir / "experiment_metadata.json", metadata)
    print(f"[OK] Figure 1 materials written to {args.outdir}")


if __name__ == "__main__":
    main()
