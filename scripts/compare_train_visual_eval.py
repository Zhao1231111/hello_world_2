#!/usr/bin/env python3

"""
对比两个 train_visual_eval 目录下同名训练图像的指标变化曲线。

输入目录结构要求：
root/
  train_0005/
    metrics_history.csv
  train_0010/
    metrics_history.csv

输出：
output_dir/
  train_0005_compare.png
  train_0010_compare.png
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="将两个 train_visual_eval 的同名图像指标变化画到同一张图里。")
    parser.add_argument("--root_a", default="/root/autodl-tmp/dataset/gs_rg-10/baseline-hku_campus_seq_00-4-4-forcomparison/train_visual_eval/", help="第一个 train_visual_eval 根目录。")
    parser.add_argument("--root_b", default="/root/autodl-tmp/dataset/gs_rg-13-temp/evaluation-hku_campus_seq_00-4-5-compaison-2-moreRegress/train_visual_eval/", help="第二个 train_visual_eval 根目录。")
    parser.add_argument("--label_a", default="baseline", help="第一个数据集在图例中的名字。")
    parser.add_argument("--label_b", default="regressor", help="第二个数据集在图例中的名字。")
    parser.add_argument(
        "--output_dir",
        default="/root/autodl-tmp/dataset/gs_rg-10/comparison-4-5/train_visual_eval_compare-3",
        help="对比图输出目录，默认保存在当前目录下的 train_visual_eval_compare。",
    )
    parser.add_argument(
        "--only_common",
        action="store_true",
        help="只绘制两个目录都存在的 train_xxxx。",
    )
    return parser.parse_args()


def load_metrics_history(csv_path: Path):
    records = []
    if not csv_path.exists():
        return records

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                records.append(
                    {
                        "train_times": int(row["train_times"]),
                        "optimize_round": int(row.get("optimize_round", "-1")),
                        "psnr": float(row["psnr"]),
                        "ssim": float(row["ssim"]),
                        "lpips": float(row["lpips"]),
                    }
                )
            except (KeyError, ValueError):
                continue

    records.sort(key=lambda item: item["train_times"])
    return records


def collect_train_dirs(root: Path):
    train_dirs = {}
    for path in sorted(root.iterdir()):
        if path.is_dir() and path.name.startswith("train_"):
            history_path = path / "metrics_history.csv"
            if history_path.exists():
                train_dirs[path.name] = history_path
    return train_dirs


def plot_compare(train_name: str, records_a, records_b, label_a: str, label_b: str, output_path: Path):
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    if records_a:
        train_times_a = [item["train_times"] for item in records_a]
        axes[0].plot(train_times_a, [item["psnr"] for item in records_a], marker="o", label=label_a, color="#d95f02")
        axes[1].plot(train_times_a, [item["ssim"] for item in records_a], marker="o", label=label_a, color="#1b9e77")
        axes[2].plot(train_times_a, [item["lpips"] for item in records_a], marker="o", label=label_a, color="#7570b3")

    if records_b:
        train_times_b = [item["train_times"] for item in records_b]
        axes[0].plot(train_times_b, [item["psnr"] for item in records_b], marker="s", label=label_b, color="#e7298a")
        axes[1].plot(train_times_b, [item["ssim"] for item in records_b], marker="s", label=label_b, color="#66a61e")
        axes[2].plot(train_times_b, [item["lpips"] for item in records_b], marker="s", label=label_b, color="#1f78b4")

    axes[0].set_ylabel("PSNR")
    axes[1].set_ylabel("SSIM")
    axes[2].set_ylabel("LPIPS")
    axes[2].set_xlabel("Train Times")

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()

    fig.suptitle(train_name)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()

    root_a = Path(args.root_a).expanduser().resolve()
    root_b = Path(args.root_b).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not root_a.exists():
        raise FileNotFoundError(f"目录不存在: {root_a}")
    if not root_b.exists():
        raise FileNotFoundError(f"目录不存在: {root_b}")

    output_dir.mkdir(parents=True, exist_ok=True)

    train_dirs_a = collect_train_dirs(root_a)
    train_dirs_b = collect_train_dirs(root_b)

    names_a = set(train_dirs_a.keys())
    names_b = set(train_dirs_b.keys())
    if args.only_common:
        train_names = sorted(names_a & names_b)
    else:
        train_names = sorted(names_a | names_b)

    if not train_names:
        print("没有找到可对比的 train_xxxx/metrics_history.csv。")
        return

    for train_name in train_names:
        records_a = load_metrics_history(train_dirs_a[train_name]) if train_name in train_dirs_a else []
        records_b = load_metrics_history(train_dirs_b[train_name]) if train_name in train_dirs_b else []

        if not records_a and not records_b:
            continue

        output_path = output_dir / f"{train_name}_compare.png"
        plot_compare(train_name, records_a, records_b, args.label_a, args.label_b, output_path)
        print(f"[OK] {output_path}")

    print("全部对比图生成完成。")


if __name__ == "__main__":
    main()
