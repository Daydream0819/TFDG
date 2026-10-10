#!/usr/bin/env python3
"""Evaluate class-conditional guidance runs in one frozen feature space.

The evaluator treats high-guidance appearance drift as excursion from a matched
weak-CFG sample, never as distance to an ImageNet target. Distribution metrics
are intentionally disabled during small development screens.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms


METHODS = ("cfg", "cfg_rescale", "apg", "fdg", "c2fg", "tfdg")


def image_files(root):
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted(
        path for path in Path(root).rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def make_loader(paths, transform, batch_size):
    class Dataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, index):
            with Image.open(paths[index]) as image:
                return transform(image.convert("RGB")), str(paths[index])

    return torch.utils.data.DataLoader(
        Dataset(), batch_size=batch_size, shuffle=False, num_workers=4,
        pin_memory=True,
    )


@torch.inference_mode()
def extract_features(paths, model, transform, device, batch_size):
    features = []
    for images, _paths in make_loader(paths, transform, batch_size):
        features.append(model(images.to(device, non_blocking=True)).cpu())
    if not features:
        raise ValueError("no images found")
    return torch.cat(features).numpy().astype(np.float64)


def fid_score(real, fake):
    from scipy.linalg import sqrtm

    mu_r, mu_f = real.mean(0), fake.mean(0)
    cov_r, cov_f = np.cov(real, rowvar=False), np.cov(fake, rowvar=False)
    try:
        covmean, _ = sqrtm(cov_r @ cov_f, disp=False)
    except TypeError:
        covmean = sqrtm(cov_r @ cov_f)
    if not np.isfinite(covmean).all():
        eps = 1e-6
        product = (cov_r + np.eye(cov_r.shape[0]) * eps) @ (
            cov_f + np.eye(cov_f.shape[0]) * eps
        )
        try:
            covmean, _ = sqrtm(product, disp=False)
        except TypeError:
            covmean = sqrtm(product)
    covmean = covmean.real
    diff = mu_r - mu_f
    return float(diff @ diff + np.trace(cov_r + cov_f - 2 * covmean))


def kth_radius(features, device, k=3, chunk=512):
    radii = []
    values = torch.from_numpy(features).float().to(device)
    for start in range(0, len(values), chunk):
        distances = torch.cdist(values[start : start + chunk], values)
        local = torch.arange(len(distances), device=device)
        distances[local, start + local] = torch.inf
        radii.append(torch.topk(distances, k=k, largest=False).values[:, -1].cpu())
    return torch.cat(radii)


def precision_recall(real, fake, device, k=3, chunk=256):
    """Improved precision/recall using union-of-kNN-balls membership."""
    real_radius = kth_radius(real, device, k).to(device)
    fake_radius = kth_radius(fake, device, k).to(device)
    real_t, fake_t = torch.from_numpy(real).float().to(device), torch.from_numpy(fake).float().to(device)
    precision_count = recall_count = 0
    for start in range(0, len(fake_t), chunk):
        distance = torch.cdist(fake_t[start : start + chunk], real_t)
        precision_count += (distance <= real_radius.unsqueeze(0)).any(dim=1).sum().item()
    for start in range(0, len(real_t), chunk):
        distance = torch.cdist(real_t[start : start + chunk], fake_t)
        recall_count += (distance <= fake_radius.unsqueeze(0)).any(dim=1).sum().item()
    return precision_count / len(fake_t), recall_count / len(real_t)


def manifest_rows(manifest):
    payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
    rows = payload.get("samples", []) if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest must contain a non-empty sample list")
    return rows


def class_map(manifest):
    result = {}
    for row in manifest_rows(manifest):
        result[str(row["sample_id"])] = int(row["class_id"])
        result[f"{int(row['seed']):06d}"] = int(row["class_id"])
    return result


def manifest_sample_names(manifest, filename_mode):
    rows = manifest_rows(manifest)
    if filename_mode == "sample_id":
        return {str(row["sample_id"]) for row in rows}
    if filename_mode == "seed":
        return {f"{int(row['seed']):06d}" for row in rows}
    raise ValueError(f"Unsupported filename mode: {filename_mode}")


def generated_files(root, allowed_names, expected_count):
    paths = [path for path in image_files(root) if path.stem in allowed_names]
    if len(paths) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} manifest images under {root}, found {len(paths)}. "
            "Re-run generation with the same manifest or use a new run directory."
        )
    return paths


@torch.inference_mode()
def class_accuracy(paths, labels, model, transform, device, batch_size):
    correct1 = correct5 = total = 0
    for images, names in make_loader(paths, transform, batch_size):
        expected = [labels.get(Path(name).stem, -1) for name in names]
        keep = torch.tensor([label >= 0 for label in expected], dtype=torch.bool)
        if not keep.any():
            continue
        logits = model(images.to(device, non_blocking=True))[keep.to(device)]
        target = torch.tensor([label for label in expected if label >= 0], device=device)
        correct1 += (logits.argmax(1) == target).sum().item()
        correct5 += logits.topk(k=5, dim=1).indices.eq(target[:, None]).any(1).sum().item()
        total += len(target)
    if not total:
        return float("nan"), float("nan")
    return correct1 / total, correct5 / total


def appearance_by_sample(paths):
    values = {}
    for path in paths:
        image = Image.open(path).convert("RGB")
        rgb = np.asarray(image, dtype=np.float32) / 255.0
        saturation = np.asarray(image.convert("HSV"), dtype=np.float32)[:, :, 1] / 255.0
        luminance = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        values[path.stem] = {
            "saturation_hsv": float(saturation.mean()),
            "brightness": float(luminance.mean()),
            "contrast": float(luminance.std()),
            "mean_rgb": rgb.mean(axis=(0, 1)),
        }
    return values


def appearance_summary(paths, reference_stats=None):
    values = appearance_by_sample(paths)
    result = {
        "saturation_hsv": float(np.mean([item["saturation_hsv"] for item in values.values()])),
        "brightness": float(np.mean([item["brightness"] for item in values.values()])),
        "contrast": float(np.mean([item["contrast"] for item in values.values()])),
        "saturation_drift": float("nan"),
        "brightness_drift": float("nan"),
        "contrast_drift": float("nan"),
        "color_shift": float("nan"),
    }
    if reference_stats is None:
        return result
    matched = [sample_id for sample_id in values if sample_id in reference_stats]
    if not matched:
        return result
    result["saturation_drift"] = float(np.mean([
        abs(values[key]["saturation_hsv"] - reference_stats[key]["saturation_hsv"])
        for key in matched
    ]))
    result["brightness_drift"] = float(np.mean([
        abs(values[key]["brightness"] - reference_stats[key]["brightness"])
        for key in matched
    ]))
    result["contrast_drift"] = float(np.mean([
        abs(values[key]["contrast"] - reference_stats[key]["contrast"])
        for key in matched
    ]))
    result["color_shift"] = float(np.mean([
        np.linalg.norm(values[key]["mean_rgb"] - reference_stats[key]["mean_rgb"])
        for key in matched
    ]))
    return result


@torch.inference_mode()
def build_clipiqa(device):
    try:
        import pyiqa
    except ImportError as error:
        raise RuntimeError("CLIP-IQA requires pyiqa. Re-run setup_server.sh or pass --skip-clipiqa.") from error
    return pyiqa.create_metric("clipiqa", device=device)


@torch.inference_mode()
def clipiqa_score(paths, metric, device, batch_size):
    to_tensor = transforms.ToTensor()
    scores = []
    for start in range(0, len(paths), batch_size):
        images = torch.stack([
            to_tensor(Image.open(path).convert("RGB"))
            for path in paths[start : start + batch_size]
        ])
        scores.append(metric(images.to(device, non_blocking=True)).reshape(-1).cpu())
    return float(torch.cat(scores).mean().item())


@torch.inference_mode()
def inception_score(paths, model, transform, device, batch_size, splits=10):
    probabilities = []
    for images, _names in make_loader(paths, transform, batch_size):
        probabilities.append(torch.softmax(model(images.to(device, non_blocking=True)), dim=1).cpu())
    probs = torch.cat(probabilities).numpy().astype(np.float64)
    scores = []
    for chunk in np.array_split(probs, min(splits, len(probs))):
        marginal = chunk.mean(axis=0, keepdims=True)
        divergence = chunk * (np.log(chunk + 1e-12) - np.log(marginal + 1e-12))
        scores.append(float(np.exp(divergence.sum(axis=1).mean())))
    return float(np.mean(scores)), float(np.std(scores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-reference", type=int, default=50000)
    parser.add_argument("--cfg-root", type=Path, default=None)
    parser.add_argument("--cfg-scale", type=float, default=None)
    parser.add_argument("--skip-clipiqa", action="store_true")
    parser.add_argument("--skip-distribution", action="store_true")
    parser.add_argument("--filename-mode", choices=["sample_id", "seed"], required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    classifier = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).eval().to(device)
    classifier_transform = models.ResNet50_Weights.DEFAULT.transforms()
    inception = score_model = feature_transform = reference_features = None
    if not args.skip_distribution:
        if args.reference_root is None or not args.reference_root.is_dir():
            raise ValueError("--reference-root must be an ImageNet validation directory")
        weights = models.Inception_V3_Weights.DEFAULT
        inception = models.inception_v3(weights=weights)
        inception.aux_logits = False
        inception.AuxLogits = None
        inception.fc = torch.nn.Identity()
        inception.eval().to(device)
        score_model = models.inception_v3(weights=weights).eval().to(device)
        feature_transform = weights.transforms()
        reference_paths = image_files(args.reference_root)[: args.max_reference]
        if not reference_paths:
            raise ValueError(f"no reference images found under {args.reference_root}")
        reference_features = extract_features(reference_paths, inception, feature_transform, device, args.batch_size)

    expected_names = manifest_sample_names(args.manifest, args.filename_mode)
    expected_count = len(manifest_rows(args.manifest))
    cfg_root = args.cfg_root if args.cfg_root is not None else args.generated_root
    cfg_scale = args.scale if args.cfg_scale is None else args.cfg_scale
    cfg_folder = cfg_root / "cfg" / f"w{cfg_scale:g}"
    reference_appearance = None
    if cfg_folder.is_dir():
        reference_appearance = appearance_by_sample(
            generated_files(cfg_folder, expected_names, expected_count)
        )
    clipiqa_metric = None if args.skip_clipiqa else build_clipiqa(device)
    labels = class_map(args.manifest)
    rows = []
    for method in args.methods:
        paths = generated_files(args.generated_root / method / f"w{args.scale:g}", expected_names, expected_count)
        if args.skip_distribution:
            fid = precision = recall = is_mean = is_std = float("nan")
        else:
            fake_features = extract_features(paths, inception, feature_transform, device, args.batch_size)
            precision, recall = precision_recall(reference_features, fake_features, device)
            is_mean, is_std = inception_score(paths, score_model, feature_transform, device, args.batch_size)
            fid = fid_score(reference_features, fake_features)
        top1, top5 = class_accuracy(paths, labels, classifier, classifier_transform, device, args.batch_size)
        appearance = appearance_summary(paths, reference_appearance)
        row = {
            "method": method,
            "sample_count": len(paths),
            "fid": fid,
            "precision": precision,
            "recall": recall,
            "inception_score_mean": is_mean,
            "inception_score_std": is_std,
            "top1": top1,
            "top5": top5,
            "clipiqa": float("nan") if clipiqa_metric is None else clipiqa_score(paths, clipiqa_metric, device, args.batch_size),
            **appearance,
        }
        rows.append(row)
        print(row)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {args.output_csv}")


if __name__ == "__main__":
    main()
