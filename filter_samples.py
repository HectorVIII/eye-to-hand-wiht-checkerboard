#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
filter_samples.py

Filter hand-eye calibration samples based on reprojection error.

Input:
    calib_dataset/samples.json

Output:
    calib_dataset/samples_filtered.json

Usage examples:
    python3 filter_samples.py
    python3 filter_samples.py --keep-best 20
    python3 filter_samples.py --drop-worst 5
    python3 filter_samples.py --max-reproj 0.8
"""

import json
import argparse
from pathlib import Path
from statistics import mean


# =========================================================
# Default paths
# =========================================================
DEFAULT_INPUT = Path("calib_dataset/samples.json")
DEFAULT_OUTPUT = Path("calib_dataset/samples_filtered.json")


# =========================================================
# Helpers
# =========================================================
def load_samples(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "samples" not in data:
        raise ValueError("Input JSON does not contain key 'samples'.")

    samples = data["samples"]
    if not isinstance(samples, list) or len(samples) == 0:
        raise ValueError("No samples found in input file.")

    return data, samples


def save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved filtered samples to: {path.resolve()}")


def get_reproj(sample: dict) -> float:
    """
    Read reprojection error from sample.
    """
    if "reproj_error_px" not in sample:
        raise ValueError(f"Sample id={sample.get('id', 'unknown')} missing 'reproj_error_px'")
    return float(sample["reproj_error_px"])


def summarize(samples, name="Samples"):
    errs = [get_reproj(s) for s in samples]
    print(f"{name}: {len(samples)}")
    print(f"  min reproj = {min(errs):.6f} px")
    print(f"  max reproj = {max(errs):.6f} px")
    print(f"  mean reproj = {mean(errs):.6f} px")


def print_ranked_table(samples, top_n=None):
    ranked = sorted(samples, key=get_reproj)
    if top_n is None:
        top_n = len(ranked)

    print("\nRanked samples by reprojection error (best -> worst):")
    print("-" * 72)
    print(f"{'rank':>4} | {'id':>4} | {'reproj_error_px':>16} | image_path")
    print("-" * 72)

    for rank, s in enumerate(ranked[:top_n], start=1):
        sid = s.get("id", -1)
        err = get_reproj(s)
        image_path = s.get("image_path", "")
        print(f"{rank:>4} | {sid:>4} | {err:>16.6f} | {image_path}")


def reindex_samples(samples):
    """
    Reassign ids from 0..N-1 after filtering.
    Keep original id in 'original_id' for traceability.
    """
    out = []
    for new_id, s in enumerate(samples):
        s2 = dict(s)
        s2["original_id"] = s.get("id", new_id)
        s2["id"] = new_id
        out.append(s2)
    return out


# =========================================================
# Main filter logic
# =========================================================
def filter_samples(samples, keep_best=None, drop_worst=None, max_reproj=None):
    ranked = sorted(samples, key=get_reproj)

    # 1) threshold filter first
    if max_reproj is not None:
        ranked = [s for s in ranked if get_reproj(s) <= max_reproj]

    # 2) then keep best N if requested
    if keep_best is not None:
        if keep_best <= 0:
            raise ValueError("--keep-best must be > 0")
        ranked = ranked[:keep_best]

    # 3) or drop worst K if requested
    elif drop_worst is not None:
        if drop_worst < 0:
            raise ValueError("--drop-worst must be >= 0")
        if drop_worst >= len(ranked):
            raise ValueError("drop_worst is too large; no samples would remain.")
        ranked = ranked[: len(ranked) - drop_worst]

    return ranked


# =========================================================
# CLI
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Filter calibration samples by reprojection error.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input samples.json path")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output filtered JSON path")

    parser.add_argument(
        "--keep-best",
        type=int,
        default=None,
        help="Keep only the best N samples after sorting by reproj_error_px"
    )
    parser.add_argument(
        "--drop-worst",
        type=int,
        default=5,
        help="Drop the worst K samples after sorting by reproj_error_px (default: 5)"
    )
    parser.add_argument(
        "--max-reproj",
        type=float,
        default=None,
        help="Keep only samples with reproj_error_px <= this threshold"
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Print the full ranked table instead of only top/bottom summary"
    )

    return parser.parse_args()


# =========================================================
# Main
# =========================================================
def main():
    args = parse_args()

    data, samples = load_samples(args.input)

    print("=" * 72)
    print("Filter Calibration Samples")
    print("=" * 72)
    print(f"Input : {args.input}")
    summarize(samples, "Original samples")

    ranked = sorted(samples, key=get_reproj)

    if args.show_all:
        print_ranked_table(ranked, top_n=len(ranked))
    else:
        print_ranked_table(ranked, top_n=min(10, len(ranked)))

        print("\nWorst 5 samples:")
        print("-" * 72)
        print(f"{'rank':>4} | {'id':>4} | {'reproj_error_px':>16} | image_path")
        print("-" * 72)
        worst = list(reversed(ranked[-5:]))
        total = len(ranked)
        for i, s in enumerate(worst):
            sid = s.get("id", -1)
            err = get_reproj(s)
            image_path = s.get("image_path", "")
            print(f"{total - i:>4} | {sid:>4} | {err:>16.6f} | {image_path}")

    filtered = filter_samples(
        samples=samples,
        keep_best=args.keep_best,
        drop_worst=args.drop_worst,
        max_reproj=args.max_reproj,
    )

    if len(filtered) < 3:
        raise RuntimeError("Too few samples remain after filtering. Need at least 3.")

    filtered = reindex_samples(filtered)

    summarize(filtered, "Filtered samples")

    output = {
        "meta": {
            "source_file": str(args.input),
            "num_original_samples": len(samples),
            "num_filtered_samples": len(filtered),
            "filter_rule": {
                "keep_best": args.keep_best,
                "drop_worst": args.drop_worst,
                "max_reproj": args.max_reproj,
            },
            "description": "Filtered calibration samples based on reprojection error",
        },
        "samples": filtered,
    }

    save_json(args.output, output)

    print("\nRecommended next step:")
    print(f"  python3 calibrate_handeye.py --input {args.output}")


if __name__ == "__main__":
    main()
