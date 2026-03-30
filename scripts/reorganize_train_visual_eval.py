#!/usr/bin/env python3

"""
将 train_visual_eval 的旧结构整理成更易查看的平铺结构，并绘制指标曲线。

旧结构:
train_0005/
  gt.png
  metrics_history.csv
  times_0001/
    render.png
    metrics.txt

新结构:
train_0005/
  gt.png
  metrics_history.csv
  metrics_curve.png
  times_0001.jpg
  times_0001_metrics.txt
"""

import argparse
import csv
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


def parse_args():
    parser = argparse.ArgumentParser(description="整理 train_visual_eval 目录并绘制指标曲线。")
    parser.add_argument("--root", required=True, help="train_visual_eval 根目录。")
    parser.add_argument(
        "--cleanup_old_times_dirs",
        action="store_true",
        help="整理完成后删除旧的 times_xxxx 子目录。",
    )
    return parser.parse_args()


def parse_metrics_txt(metrics_path: Path):
    record = {}
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            record[key.strip()] = value.strip()

    if "train_times" not in record:
        return None

    return {
        "train_times": int(record["train_times"]),
        "optimize_round": int(record.get("optimize_round", "-1")),
        "psnr": float(record.get("psnr", "nan")),
        "ssim": float(record.get("ssim", "nan")),
        "lpips": float(record.get("lpips", "nan")),
    }


def load_history_csv(csv_path: Path):
    records = {}
    if not csv_path.exists():
        return records

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                train_times = int(row["train_times"])
            except (KeyError, ValueError):
                continue
            records[train_times] = {
                "train_times": train_times,
                "optimize_round": int(row.get("optimize_round", "-1")),
                "psnr": float(row.get("psnr", "nan")),
                "ssim": float(row.get("ssim", "nan")),
                "lpips": float(row.get("lpips", "nan")),
            }
    return records


def save_history_csv(csv_path: Path, records):
    ordered_records = sorted(records.values(), key=lambda item: item["train_times"])
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["train_times", "optimize_round", "psnr", "ssim", "lpips"])
        for item in ordered_records:
            writer.writerow(
                [
                    item["train_times"],
                    item["optimize_round"],
                    item["psnr"],
                    item["ssim"],
                    item["lpips"],
                ]
            )


def convert_image_to_jpg(src_path: Path, dst_path: Path):
    if dst_path.exists():
        return

    if cv2 is not None:
        image = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"无法读取图片: {src_path}")
        cv2.imwrite(str(dst_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        return

    if Image is not None:
        image = Image.open(src_path).convert("RGB")
        image.save(dst_path, quality=95)
        return

    raise RuntimeError("既没有 cv2，也没有 PIL，无法把旧图片转换成 jpg。")


def collect_records_from_flat_files(train_dir: Path):
    records = {}
    for metrics_path in sorted(train_dir.glob("times_*_metrics.txt")):
        record = parse_metrics_txt(metrics_path)
        if record is None:
            continue
        records[record["train_times"]] = record
    return records


def migrate_old_times_dirs(train_dir: Path):
    migrated_records = {}
    old_dirs = sorted([path for path in train_dir.iterdir() if path.is_dir() and path.name.startswith("times_")])

    for old_dir in old_dirs:
        render_src = old_dir / "render.png"
        metrics_src = old_dir / "metrics.txt"
        render_dst = train_dir / f"{old_dir.name}.jpg"
        metrics_dst = train_dir / f"{old_dir.name}_metrics.txt"

        if render_src.exists():
            convert_image_to_jpg(render_src, render_dst)
        if metrics_src.exists() and not metrics_dst.exists():
            shutil.copy2(metrics_src, metrics_dst)

        if metrics_dst.exists():
            record = parse_metrics_txt(metrics_dst)
            if record is not None:
                migrated_records[record["train_times"]] = record

    return old_dirs, migrated_records


def plot_metrics_curve(train_dir: Path, records):
    ordered_records = sorted(records.values(), key=lambda item: item["train_times"])
    if not ordered_records:
        return

    train_times = [item["train_times"] for item in ordered_records]
    psnr_values = [item["psnr"] for item in ordered_records]
    ssim_values = [item["ssim"] for item in ordered_records]
    lpips_values = [item["lpips"] for item in ordered_records]

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    axes[0].plot(train_times, psnr_values, marker="o", color="#d95f02")
    axes[0].set_ylabel("PSNR")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].plot(train_times, ssim_values, marker="o", color="#1b9e77")
    axes[1].set_ylabel("SSIM")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    axes[2].plot(train_times, lpips_values, marker="o", color="#7570b3")
    axes[2].set_ylabel("LPIPS")
    axes[2].set_xlabel("Train Times")
    axes[2].grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(train_dir.name)
    fig.tight_layout()
    fig.savefig(train_dir / "metrics_curve.png", dpi=200)
    plt.close(fig)


def process_train_dir(train_dir: Path, cleanup_old_times_dirs: bool):
    history_csv_path = train_dir / "metrics_history.csv"

    records = load_history_csv(history_csv_path)

    old_dirs, migrated_records = migrate_old_times_dirs(train_dir)
    records.update(migrated_records)

    flat_records = collect_records_from_flat_files(train_dir)
    records.update(flat_records)

    if records:
        save_history_csv(history_csv_path, records)
        plot_metrics_curve(train_dir, records)

    if cleanup_old_times_dirs:
        for old_dir in old_dirs:
            shutil.rmtree(old_dir, ignore_errors=True)

    print(f"[OK] {train_dir}")


def main():
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"目录不存在: {root}")

    train_dirs = sorted([path for path in root.iterdir() if path.is_dir() and path.name.startswith("train_")])
    if not train_dirs:
        print(f"在 {root} 下没有找到 train_xxxx 目录。")
        return

    for train_dir in train_dirs:
        process_train_dir(train_dir, args.cleanup_old_times_dirs)

    print("全部整理完成。")


if __name__ == "__main__":
    main()
