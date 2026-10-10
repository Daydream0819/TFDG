import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

try:
    import open_clip
except ModuleNotFoundError:
    open_clip = None

try:
    import clip as openai_clip
except ModuleNotFoundError:
    openai_clip = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from dataset_io import load_samples, match_images, require_exact_images


DEFAULT_METHODS = [
    "cfg",
    "cfg_rescale",
    "apg",
    "fdg",
    "c2fg",
    "tfdg",
]


def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the SD3 COCO benchmark."
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--real_images_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--generated_root",
        type=str,
        required=True,
    )
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--scale", type=float, default=15.0)
    parser.add_argument("--clip_batch_size", type=int, default=16)
    parser.add_argument("--clip_model", type=str, default="ViT-H-14")
    parser.add_argument("--clip_pretrained", type=str, default="laion2b_s32b_b79k")
    parser.add_argument(
        "--clip_backend",
        type=str,
        choices=("openai", "openclip", "auto"),
        default="openclip",
        help="CLIP implementation used for CLIPScore.",
    )
    parser.add_argument(
        "--openai_clip_model",
        type=str,
        default="ViT-B/32",
        help="OpenAI CLIP architecture used when --clip_backend=openai.",
    )
    parser.add_argument("--skip_fid", action="store_true")
    parser.add_argument(
        "--skip_artifact_metrics",
        action="store_true",
        help=(
            "Skip reference-image artifact metrics. This permits CLIP-only "
            "evaluation when paired real images are unavailable."
        ),
    )
    parser.add_argument("--fid_batch_size", type=int, default=32)
    parser.add_argument("--fid_num_workers", type=int, default=0)
    parser.add_argument(
        "--fid_use_dataparallel",
        action="store_true",
        help="Enable clean-fid DataParallel. Disabled by default to avoid NCCL errors on multi-GPU servers.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


RESULT_FIELDS = [
    "method",
    "clip_backend",
    "clip_model",
    "clipscore_mean",
    "clipscore_std",
    "clipscore_count",
    "clip_cosine_mean",
    "saturation_mean",
    "saturation_drift_mean",
    "contrast_mean",
    "brightness_mean",
    "brightness_drift_mean",
    "color_shift_mean",
    "fid",
    "missing_images",
    "extra_images",
]


def save_results(results, generated_root):
    json_path = os.path.join(generated_root, "benchmark_results.json")
    csv_path = os.path.join(generated_root, "benchmark_results.csv")
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
    with open(csv_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(results)
    return json_path, csv_path


def compute_artifact_metrics(items, real_items):
    saturation_values = []
    saturation_drifts = []
    contrast_values = []
    brightness_values = []
    brightness_drifts = []
    color_shifts = []

    real_by_id = {item["sample_id"]: path for item, path in real_items}
    for item, generated_path in tqdm(items, desc="Artifact metrics"):
        generated_image = Image.open(generated_path).convert("RGB")
        real_path = real_by_id[item["sample_id"]]
        real_image = Image.open(real_path).convert("RGB")

        generated_rgb = np.asarray(generated_image, dtype=np.float32) / 255.0
        real_rgb = np.asarray(real_image, dtype=np.float32) / 255.0
        generated_gray = (
            np.asarray(generated_image.convert("L"), dtype=np.float32) / 255.0
        )
        real_gray = np.asarray(real_image.convert("L"), dtype=np.float32) / 255.0
        generated_saturation = (
            np.asarray(generated_image.convert("HSV"), dtype=np.float32)[:, :, 1]
            / 255.0
        )
        real_saturation = (
            np.asarray(real_image.convert("HSV"), dtype=np.float32)[:, :, 1] / 255.0
        )

        saturation_values.append(float(generated_saturation.mean()))
        saturation_drifts.append(
            float(abs(generated_saturation.mean() - real_saturation.mean()))
        )
        contrast_values.append(float(generated_gray.std()))
        brightness_values.append(float(generated_gray.mean()))
        brightness_drifts.append(float(abs(generated_gray.mean() - real_gray.mean())))
        color_shifts.append(
            float(
                np.linalg.norm(
                    generated_rgb.mean(axis=(0, 1)) - real_rgb.mean(axis=(0, 1))
                )
            )
        )

    return {
        "saturation_mean": float(np.mean(saturation_values)),
        "saturation_drift_mean": float(np.mean(saturation_drifts)),
        "contrast_mean": float(np.mean(contrast_values)),
        "brightness_mean": float(np.mean(brightness_values)),
        "brightness_drift_mean": float(np.mean(brightness_drifts)),
        "color_shift_mean": float(np.mean(color_shifts)),
    }


@torch.no_grad()
def compute_clip_score(model, preprocess, tokenizer, device, items, batch_size, description):
    scores = []
    cosine_scores = []
    for start in tqdm(range(0, len(items), batch_size), desc=description):
        batch = items[start : start + batch_size]
        images = torch.stack(
            [preprocess(Image.open(path).convert("RGB")) for _, path in batch]
        ).to(device)
        texts = tokenizer(
            [f"A photo depicts {item['caption']}" for item, _ in batch]
        ).to(device)

        image_features = model.encode_image(images)
        text_features = model.encode_text(texts)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        cosine = torch.sum(image_features * text_features, dim=-1)
        cosine_scores.extend(cosine.float().cpu().tolist())
        scores.extend((2.5 * cosine.clamp_min(0.0)).float().cpu().tolist())

    score_tensor = torch.tensor(scores, dtype=torch.float32)
    cosine_tensor = torch.tensor(cosine_scores, dtype=torch.float32)
    return {
        "clipscore_mean": score_tensor.mean().item(),
        "clipscore_std": score_tensor.std(unbiased=False).item(),
        "clipscore_count": len(scores),
        "clip_cosine_mean": cosine_tensor.mean().item(),
    }


def run_evaluation():
    args = parse_cli_args()
    metadata = load_samples(args.metadata_path)
    requires_real_images = not args.skip_artifact_metrics or not args.skip_fid
    real_items = None
    if requires_real_images:
        real_items = require_exact_images(
            args.real_images_dir,
            metadata,
            filename_mode="generated_index",
        )

    if args.clip_backend in ("openclip", "auto") and open_clip is not None:
        print(f"Loading OpenCLIP model: {args.clip_model}")
        model, _, preprocess = open_clip.create_model_and_transforms(
            args.clip_model,
            pretrained=args.clip_pretrained,
        )
        model = model.to(args.device).eval()
        tokenizer = open_clip.get_tokenizer(args.clip_model)
        clip_backend_name = "openclip"
        clip_model_name = f"{args.clip_model}/{args.clip_pretrained}"
    elif args.clip_backend in ("openai", "auto") and openai_clip is not None:
        print(f"Loading OpenAI CLIP model: {args.openai_clip_model}")
        model, preprocess = openai_clip.load(
            args.openai_clip_model,
            device=args.device,
            download_root=os.environ.get("CLIP_HOME"),
        )
        model = model.eval()
        tokenizer = openai_clip.tokenize
        clip_backend_name = "openai_clip"
        clip_model_name = args.openai_clip_model
    else:
        raise ImportError(
            f"The requested CLIP backend '{args.clip_backend}' is unavailable."
        )

    results = []
    for method_name in args.methods:
        method_dir = os.path.join(
            args.generated_root,
            method_name,
            f"w{args.scale:g}",
        )
        if not os.path.isdir(method_dir):
            print(f"Skipping missing directory: {method_dir}")
            continue

        items, missing, ambiguous, extra = match_images(
            method_dir,
            metadata,
            filename_mode="sample_id",
        )
        if missing or ambiguous or extra:
            raise RuntimeError(
                f"{method_name} does not match the frozen sample list: "
                f"{len(missing)} missing, {len(ambiguous)} ambiguous, "
                f"{len(extra)} extra images."
            )

        clip_result = compute_clip_score(
            model,
            preprocess,
            tokenizer,
            args.device,
            items,
            args.clip_batch_size,
            f"CLIP {method_name}",
        )
        if args.skip_artifact_metrics:
            artifact_result = {
                "saturation_mean": None,
                "saturation_drift_mean": None,
                "contrast_mean": None,
                "brightness_mean": None,
                "brightness_drift_mean": None,
                "color_shift_mean": None,
            }
        else:
            artifact_result = compute_artifact_metrics(items, real_items)

        fid_score = None
        if not args.skip_fid and not missing and not extra:
            print(f"Computing FID: {method_name}")
            try:
                from cleanfid import fid
            except ModuleNotFoundError as error:
                raise ImportError(
                    "FID evaluation requires clean-fid. Install it with "
                    "`pip install clean-fid`, or pass --skip_fid."
                ) from error
            fid_score = float(
                fid.compute_fid(
                    args.real_images_dir,
                    method_dir,
                    batch_size=args.fid_batch_size,
                    num_workers=args.fid_num_workers,
                    device=args.device,
                    use_dataparallel=args.fid_use_dataparallel,
                )
            )
        elif args.skip_fid:
            print(f"Skipping FID by request: {method_name}")

        result = {
            "method": method_name,
            "clip_backend": clip_backend_name,
            "clip_model": clip_model_name,
            **clip_result,
            **artifact_result,
            "fid": fid_score,
            "missing_images": len(missing),
            "extra_images": len(extra),
        }
        results.append(result)
        print(
            f"{method_name}: CLIPScore={result['clipscore_mean']:.4f}, "
            f"FID={fid_score if fid_score is not None else 'SKIPPED'}"
        )
        save_results(results, args.generated_root)

    json_path, csv_path = save_results(results, args.generated_root)
    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    run_evaluation()
