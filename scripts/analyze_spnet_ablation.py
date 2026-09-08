#!/usr/bin/env python3
"""Aggregate paired with/without-SPNet per-frame metrics."""

import argparse
import csv
import math
from pathlib import Path


METRICS = ("psnr", "ssim", "lpips", "blind_psnr", "blind_alpha_coverage")


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        name = row["image_name"]
        if name in result:
            raise ValueError(f"duplicate frame {name} in {path}")
        result[name] = row
    return result


def number(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.nan


def mean(values):
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else math.nan


def write_csv(path: Path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_single_row(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected one row in {path}, found {len(rows)}")
    return rows[0]


def peak_gpu_memory(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        values = [number(row, "memory_used_mb") for row in csv.DictReader(handle)]
    finite = [value for value in values if math.isfinite(value)]
    return max(finite) if finite else math.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True)
    args = parser.parse_args()
    root = Path(args.experiment_root).expanduser().resolve()
    analysis_dir = root / "analysis"
    paired = []
    summaries = []
    run_summaries = []

    sequence_dirs = sorted(path for path in root.iterdir() if path.is_dir() and path.name != "analysis")
    for sequence_dir in sequence_dirs:
        with_path = sequence_dir / "with_spnet" / "visual_quality" / "test" / "frame_metrics.csv"
        without_path = sequence_dir / "without_spnet" / "visual_quality" / "test" / "frame_metrics.csv"
        if not with_path.exists() or not without_path.exists():
            raise FileNotFoundError(f"missing paired metrics under {sequence_dir}")
        with_rows = read_rows(with_path)
        without_rows = read_rows(without_path)
        if set(with_rows) != set(without_rows):
            missing_with = sorted(set(without_rows) - set(with_rows))
            missing_without = sorted(set(with_rows) - set(without_rows))
            raise ValueError(
                f"frame mismatch for {sequence_dir.name}: missing_with={missing_with}, "
                f"missing_without={missing_without}"
            )

        sequence_pairs = []
        for image_name in sorted(with_rows):
            row = {"sequence": sequence_dir.name, "image_name": image_name}
            for metric in METRICS:
                with_value = number(with_rows[image_name], metric)
                without_value = number(without_rows[image_name], metric)
                row[f"with_{metric}"] = with_value
                row[f"without_{metric}"] = without_value
                row[f"delta_{metric}"] = with_value - without_value
            row["blind_ratio"] = number(with_rows[image_name], "blind_ratio")
            without_blind_ratio = number(without_rows[image_name], "blind_ratio")
            if math.isfinite(row["blind_ratio"]) and math.isfinite(without_blind_ratio):
                if abs(row["blind_ratio"] - without_blind_ratio) > 1e-8:
                    raise ValueError(f"blind mask mismatch: {sequence_dir.name}/{image_name}")
            sequence_pairs.append(row)
        paired.extend(sequence_pairs)

        summary = {"sequence": sequence_dir.name, "frame_count": len(sequence_pairs)}
        for metric in METRICS:
            summary[f"with_{metric}"] = mean([row[f"with_{metric}"] for row in sequence_pairs])
            summary[f"without_{metric}"] = mean([row[f"without_{metric}"] for row in sequence_pairs])
            summary[f"delta_{metric}"] = mean([row[f"delta_{metric}"] for row in sequence_pairs])
        summaries.append(summary)

        for condition in ("with_spnet", "without_spnet"):
            run_dir = sequence_dir / condition
            runtime = read_single_row(run_dir / "ablation" / "runtime_summary.csv")
            candidate_path = run_dir / "ablation" / "candidate_flow.csv"
            with candidate_path.open(newline="", encoding="utf-8") as handle:
                candidate_rows = list(csv.DictReader(handle))
            proposed = sum(int(row["spnet_proposed"]) for row in candidate_rows)
            inserted = sum(int(row["spnet_inserted"]) for row in candidate_rows)
            run_summaries.append({
                "sequence": sequence_dir.name,
                "condition": condition,
                "spnet_proposed": proposed,
                "spnet_inserted": inserted,
                "spnet_insert_rate": inserted / proposed if proposed else math.nan,
                "spnet_mean_ms": number(runtime, "spnet_mean_ms"),
                "total_mapping_seconds": number(runtime, "total_mapping_seconds"),
                "total_extending_seconds": number(runtime, "total_extending_seconds"),
                "final_gaussians": number(runtime, "final_gaussians"),
                "peak_gpu_memory_mb": peak_gpu_memory(run_dir / "gpu_memory.csv"),
            })

    paired_fields = ["sequence", "image_name", "blind_ratio"]
    for metric in METRICS:
        paired_fields.extend((f"with_{metric}", f"without_{metric}", f"delta_{metric}"))
    summary_fields = ["sequence", "frame_count"]
    for metric in METRICS:
        summary_fields.extend((f"with_{metric}", f"without_{metric}", f"delta_{metric}"))
    write_csv(analysis_dir / "paired_frame_metrics.csv", paired_fields, paired)
    write_csv(analysis_dir / "summary_by_sequence.csv", summary_fields, summaries)
    run_summary_fields = [
        "sequence", "condition", "spnet_proposed", "spnet_inserted", "spnet_insert_rate",
        "spnet_mean_ms", "total_mapping_seconds", "total_extending_seconds",
        "final_gaussians", "peak_gpu_memory_mb",
    ]
    write_csv(analysis_dir / "run_summary.csv", run_summary_fields, run_summaries)
    write_csv(analysis_dir / "worst_overall_frames.csv", paired_fields,
              sorted(paired, key=lambda row: min(row["with_psnr"], row["without_psnr"])))
    write_csv(analysis_dir / "worst_blind_frames.csv", paired_fields,
              sorted(paired, key=lambda row: min(row["with_blind_psnr"], row["without_blind_psnr"])
                     if math.isfinite(row["with_blind_psnr"]) and math.isfinite(row["without_blind_psnr"])
                     else math.inf))
    write_csv(analysis_dir / "largest_spnet_regressions.csv", paired_fields,
              sorted(paired, key=lambda row: row["delta_psnr"]))
    print(f"Wrote paired analysis to {analysis_dir}")


if __name__ == "__main__":
    main()
