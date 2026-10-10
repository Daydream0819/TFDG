#!/usr/bin/env python3
"""Select the pre-registered SD1.5 radial cutoff from a geometry log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_CANDIDATES = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--candidates", nargs="+", type=float, default=DEFAULT_CANDIDATES)
    args = parser.parse_args()
    if not 0 < args.threshold <= 1:
        raise ValueError("threshold must lie in (0, 1]")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite selected cutoff: {args.output}")
    payload = json.loads(args.geometry_log.read_text(encoding="utf-8"))
    observed = payload.get("early_low_radial_fraction", {})
    selected, rows = None, []
    for paper_rho in sorted(set(float(value) for value in args.candidates)):
        key = f"{paper_rho:g}"
        if key not in observed:
            raise KeyError(f"Geometry log has no candidate rho={key}")
        fraction = float(observed[key])
        rows.append({"paper_rho": paper_rho, "early_low_radial_fraction": fraction})
        if selected is None and fraction >= args.threshold:
            selected = paper_rho
    if selected is None:
        raise RuntimeError("No pre-registered cutoff reached the required energy fraction")
    result = {
        "rule": "smallest pre-registered paper rho with Q(rho) >= threshold",
        "threshold": args.threshold,
        "candidates": rows,
        "selected_paper_rho": selected,
        "selected_code_cutoff": 2.0 * selected,
        "code_mapping": "frequency_masks uses radius <= 0.5 * --cutoff",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
