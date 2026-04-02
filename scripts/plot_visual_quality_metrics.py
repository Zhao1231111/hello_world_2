#!/usr/bin/env python3

"""
根据 evaluateVisualQuality 导出的逐帧指标 CSV，分别绘制 train/test 指标曲线。

目录结构约定:
result_root/
  visual_quality/
    train/
      frame_metrics.csv
    test/
      frame_metrics.csv

输出:
result_root/
  visual_quality/
    train/
      metrics_curve.png
    test/
      metrics_curve.png
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="绘制 train/test 逐帧 PSNR、SSIM、LPIPS 曲线。")
    parser.add_argument(
        "--result_root",
        required=True,
        help="evaluateVisualQuality 的结果根目录。",
    )
    return parser.parse_args()


def load_records(csv_path: Path):
    records = []
    if not csv_path.exists():
        return records

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                records.append(
                    {
                        "image_name": row["image_name"],
                        "frame_id": int(row["frame_id"]),
                        "subset_index": int(row.get("subset_index", "-1")),
                        "psnr": float(row["psnr"]),
                        "ssim": float(row["ssim"]),
                        "lpips": float(row["lpips"]),
                    }
                )
            except (KeyError, ValueError):
                continue

    records.sort(key=lambda item: (item["frame_id"], item["subset_index"]))
    return records


def plot_split(split_name: str, records, output_path: Path):
    if not records:
        print(f"[Skip] {split_name}: 没有可绘制的数据。")
        return

    frame_ids = [item["frame_id"] for item in records]

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    metric_specs = [
        ("psnr", "PSNR", "#d95f02"),
        ("ssim", "SSIM", "#1b9e77"),
        ("lpips", "LPIPS", "#7570b3"),
    ]

    for ax, (key, ylabel, color) in zip(axes, metric_specs):
        values = [item[key] for item in records]
        ax.plot(frame_ids, values, marker="o", markersize=4, linewidth=1.5, color=color)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.4)

    axes[2].set_xlabel("Frame ID")
    fig.suptitle(f"{split_name} Visual Quality")
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"[OK] {output_path}")


def main():
    args = parse_args()
    result_root = Path(args.result_root).expanduser().resolve()
    visual_quality_root = result_root / "visual_quality"

    split_to_paths = {
        "train": (
            visual_quality_root / "train" / "frame_metrics.csv",
            visual_quality_root / "train" / "metrics_curve.png",
        ),
        "test": (
            visual_quality_root / "test" / "frame_metrics.csv",
            visual_quality_root / "test" / "metrics_curve.png",
        ),
    }

    if not result_root.exists():
        raise FileNotFoundError(f"目录不存在: {result_root}")

    for split_name, (csv_path, output_path) in split_to_paths.items():
        records = load_records(csv_path)
        if not records:
            print(f"[Skip] {split_name}: 未找到或无法解析 {csv_path}")
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plot_split(split_name, records, output_path)

    print("绘图完成。")


if __name__ == "__main__":
    main()
