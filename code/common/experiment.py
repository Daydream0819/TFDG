"""Shared experiment configuration and immutable protocol manifests."""

import hashlib
import json
from pathlib import Path
import tempfile

import torch

from .guidance import GuidanceConfig, REPORTED_METHODS, guidance_domain


def parse_int_ranges(text):
    values = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = map(int, part.split("-", 1))
            values.extend(range(start, end + 1))
        else:
            values.append(int(part))
    return values


def torch_dtype(name):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def guidance_config(args, method, scale):
    return GuidanceConfig(
        method=method,
        scale=scale,
        momentum=args.momentum,
        norm_threshold=args.norm_threshold,
        parallel_weight=args.apg_eta,
        fdg_low_scale=args.fdg_low_scale,
        cutoff=args.cutoff,
        rescale_factor=args.rescale_factor,
        c2fg_lambda=args.c2fg_lambda,
        low_min_cfg=args.low_min_cfg,
        low_max_cfg=args.low_max_cfg,
        high_min_cfg=args.high_min_cfg,
        high_max_cfg=args.high_max_cfg,
        low_eta=args.low_eta,
        high_eta=args.high_eta,
    )


def method_protocol(methods):
    """Record the reference-defined execution domain of each method."""
    return {method: guidance_domain(method) for method in methods}


def implementation_fingerprint(model_file):
    """Hash the model adapter together with the shared mathematical core."""
    common_root = Path(__file__).resolve().parent
    paths = [
        common_root / "experiment.py",
        common_root / "guidance.py",
        common_root / "parameterization.py",
        common_root / "spectral.py",
        Path(model_file).resolve(),
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def add_guidance_args(parser, default_scale, default_norm, default_momentum):
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(REPORTED_METHODS),
        choices=[
            "cfg",
            "cfg_rescale",
            "c2fg",
            "apg",
            "fdg",
            "rptg",
            "rptg_high",
            "tfdg",
        ],
    )
    parser.add_argument("--scales", nargs="+", type=float, default=[default_scale])
    parser.add_argument("--momentum", type=float, default=default_momentum)
    parser.add_argument("--norm-threshold", type=float, default=default_norm)
    parser.add_argument("--apg-eta", type=float, default=0.0)
    parser.add_argument("--fdg-low-scale", type=float, default=1.0)
    parser.add_argument("--cutoff", type=float, default=0.30)
    parser.add_argument("--rescale-factor", type=float, default=0.7)
    parser.add_argument("--c2fg-lambda", type=float, default=1.0)
    parser.add_argument("--low-min-cfg", type=float, default=1.0)
    parser.add_argument(
        "--low-max-cfg", type=float, default=None,
        help="Optional low-radial effective-CFG ceiling; omit to recover to --scales.",
    )
    parser.add_argument("--high-min-cfg", type=float, default=3.0)
    parser.add_argument(
        "--high-max-cfg", type=float, default=None,
        help="Optional high-radial effective-CFG ceiling; omit to recover to --scales.",
    )
    parser.add_argument("--low-eta", type=float, default=0.5)
    parser.add_argument("--high-eta", type=float, default=1.0)
    return parser


def _jsonable_args(args):
    payload = vars(args).copy()
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


def _sample_key(row):
    return (
        str(row.get("sample_id", "")),
        int(row.get("seed", -1)),
        int(row.get("class_id", -1)),
        str(row.get("prompt", "")),
    )


def _is_prefix(old_rows, new_rows):
    if len(old_rows) > len(new_rows):
        return False
    return all(_sample_key(old) == _sample_key(new) for old, new in zip(old_rows, new_rows))


def _atomic_json_write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def write_manifest(outdir, args, extra=None):
    """Write one model-level protocol manifest.

    ``methods``, ``resume``, sharding fields, and ``max_samples`` are execution
    controls rather than scientific settings.  Ignoring them lets independent
    method workers share a run directory and lets a verified sample prefix grow
    without silently accepting a changed protocol.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    payload = _jsonable_args(args)
    for key in ("methods", "resume", "max_samples", "shard_index", "num_shards"):
        payload.pop(key, None)
    if extra:
        payload.update(extra)
    samples = list(payload.get("samples", []))
    payload["sample_count"] = len(samples)
    payload["protocol_hash"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in payload.items() if key not in {"samples", "sample_count", "protocol_hash"}},
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    path = outdir / "experiment_manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        old_protocol = {key: value for key, value in existing.items() if key not in {"samples", "sample_count"}}
        new_protocol = {key: value for key, value in payload.items() if key not in {"samples", "sample_count"}}
        if old_protocol != new_protocol:
            raise RuntimeError(
                f"Refusing to reuse {outdir}: scientific protocol differs. "
                "Choose a new run directory."
            )
        old_samples = list(existing.get("samples", []))
        if not (_is_prefix(old_samples, samples) or _is_prefix(samples, old_samples)):
            raise RuntimeError(
                f"Refusing to reuse {outdir}: sample lists are not prefix-compatible."
            )
        if len(samples) <= len(old_samples):
            return
    _atomic_json_write(path, payload)
