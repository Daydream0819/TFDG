#!/usr/bin/env python3
"""Merge SD3.5 distribution, alignment, artifact, and preference metrics."""

import argparse
import csv
from pathlib import Path


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--preference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = {row["method"]: dict(row) for row in read_rows(args.benchmark)}
    for preference in read_rows(args.preference):
        method = preference["method"]
        if method not in rows:
            raise ValueError(f"Preference result has unknown method: {method}")
        rows[method].update({key: value for key, value in preference.items() if key != "method"})
    ordered = list(rows.values())
    if not ordered:
        raise ValueError("No result rows were found")
    fields = []
    for row in ordered:
        for key in row:
            if key not in fields:
                fields.append(key)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ordered)
    print(f"Saved merged metrics: {args.output}")


if __name__ == "__main__":
    main()
