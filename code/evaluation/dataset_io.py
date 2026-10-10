"""Strict image/manifest matching shared by the production evaluators."""

import json
from pathlib import Path


IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})


def image_files(directory):
    directory = Path(directory)
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def load_samples(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("samples", []) if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError("metadata/manifest must contain a non-empty sample list")
    normalized = []
    for index, source in enumerate(rows):
        row = dict(source)
        generated_index = str(row.get("generated_index", index))
        padded = generated_index.zfill(5) if generated_index.isdigit() else generated_index
        row["generated_index"] = generated_index
        row["sample_id"] = str(row.get("sample_id", f"coco_{padded}"))
        row["caption"] = str(row.get("caption", row.get("prompt", ""))).strip()
        if not row["caption"]:
            raise ValueError(f"sample {row['sample_id']} has no caption/prompt")
        normalized.append(row)
    return normalized


def candidate_stems(row, filename_mode="sample_id"):
    generated_index = str(row["generated_index"])
    padded = generated_index.zfill(5) if generated_index.isdigit() else generated_index
    if filename_mode == "seed":
        return [f"{int(row['seed']):06d}"]
    if filename_mode == "sample_id":
        return list(
            dict.fromkeys(
                [row["sample_id"], f"coco_{padded}", padded, generated_index]
            )
        )
    if filename_mode == "generated_index":
        return list(dict.fromkeys([generated_index, padded, f"coco_{padded}"]))
    raise ValueError(f"unknown filename mode: {filename_mode}")


def resolve_image(directory, stems):
    directory = Path(directory)
    stem_set = set(stems)
    matches = [path for path in image_files(directory) if path.stem in stem_set]
    if len(matches) != 1:
        return None, matches
    return matches[0], matches


def match_images(directory, rows, filename_mode="sample_id"):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    items = []
    missing = []
    ambiguous = []
    used = set()
    for row in rows:
        path, matches = resolve_image(directory, candidate_stems(row, filename_mode))
        if path is None:
            target = ambiguous if matches else missing
            target.append(row["sample_id"])
            continue
        resolved = path.resolve()
        if resolved in used:
            ambiguous.append(row["sample_id"])
            continue
        used.add(resolved)
        items.append((row, path))
    extras = [path for path in image_files(directory) if path.resolve() not in used]
    return items, missing, ambiguous, extras


def require_exact_images(directory, rows, filename_mode="sample_id"):
    items, missing, ambiguous, extras = match_images(directory, rows, filename_mode)
    if missing or ambiguous or extras:
        raise RuntimeError(
            f"{directory} does not match the frozen sample list: "
            f"{len(missing)} missing, {len(ambiguous)} ambiguous, "
            f"{len(extras)} extra images"
        )
    return items
