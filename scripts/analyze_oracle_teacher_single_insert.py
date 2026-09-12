#!/usr/bin/env python3
"""配对 reference/replay 的逐关键帧 Oracle teacher 收敛曲线。"""

import argparse
import csv
import math
from pathlib import Path
import re

import matplotlib.pyplot as plt


METRICS = ("psnr", "ssim", "lpips")
KEY_COLUMNS = ("eval_frame_id", "G")


def load_metrics(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"指标文件不存在: {path}")

    rows = {}
    with path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = set(KEY_COLUMNS) | set(METRICS) | {
            "G_t", "relative_updates", "image_name", "eval_keyframe_index"
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} 缺少列: {sorted(missing)}")

        for row in reader:
            key = tuple(int(row[column]) for column in KEY_COLUMNS)
            if key in rows:
                raise ValueError(f"{path} 存在重复检查点: frame={key[0]}, G={key[1]}")
            for metric in METRICS:
                value = float(row[metric])
                if not math.isfinite(value):
                    raise ValueError(f"{path} 的 {metric} 非有限: frame={key[0]}, G={key[1]}")
                row[metric] = value
            row["G_t"] = int(row["G_t"])
            row["relative_updates"] = int(row["relative_updates"])
            row["eval_keyframe_index"] = int(row["eval_keyframe_index"])
            rows[key] = row
    if not rows:
        raise ValueError(f"指标文件为空: {path}")
    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="对比单次 Oracle teacher 实验中每个关键帧随绝对全局更新次数 G 的曲线。"
    )
    parser.add_argument("--reference-metrics", type=Path, required=True)
    # 兼容既有单一 treatment 命令；该模式的输出文件名保持不变。
    parser.add_argument("--replay-metrics", type=Path)
    # 多 treatment 叠加绘图使用“标签=CSV 路径”。标签会直接出现在图例中，
    # 便于把 full teacher 与各属性消融放在同一张检查关键帧曲线上比较。
    parser.add_argument(
        "--replay", action="append", default=[], metavar="LABEL=METRICS_CSV",
        help="可重复传入；不能与 --replay-metrics 同时使用",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.replay_metrics is not None and args.replay:
        parser.error("--replay-metrics 与 --replay 不能同时使用")
    if args.replay_metrics is None and not args.replay:
        parser.error("必须提供 --replay-metrics 或至少一个 --replay")
    return args


def parse_replay_specs(args):
    """返回 (图例标签, metrics 路径)；单 replay 保持旧版图例名称。"""
    if args.replay_metrics is not None:
        return [("teacher replay", args.replay_metrics)]

    specs = []
    used_labels = set()
    for raw_spec in args.replay:
        label, separator, path_text = raw_spec.partition("=")
        label = label.strip()
        path_text = path_text.strip()
        if not separator or not label or not path_text:
            raise ValueError("--replay 必须使用 LABEL=METRICS_CSV 格式")
        if label in used_labels:
            raise ValueError(f"--replay 标签重复: {label}")
        used_labels.add(label)
        specs.append((label, Path(path_text)))
    return specs


def label_stem(label: str):
    """把图例标签转为稳定、安全的配对 CSV 文件名片段。"""
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("._-")
    return stem or "replay"


def write_paired_metrics(path: Path, reference, replay, common_keys):
    """保存一个 treatment 与 reference 的逐检查点配对结果。"""
    with path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "eval_frame_id", "image_name", "eval_keyframe_index", "G_t", "G",
            "relative_updates", "reference_psnr", "replay_psnr", "delta_psnr",
            "reference_ssim", "replay_ssim", "delta_ssim",
            "reference_lpips", "replay_lpips", "delta_lpips",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for frame_id, global_update in common_keys:
            baseline = reference[(frame_id, global_update)]
            oracle = replay[(frame_id, global_update)]
            writer.writerow({
                "eval_frame_id": frame_id,
                "image_name": baseline["image_name"],
                "eval_keyframe_index": baseline["eval_keyframe_index"],
                "G_t": baseline["G_t"],
                "G": global_update,
                "relative_updates": baseline["relative_updates"],
                "reference_psnr": baseline["psnr"],
                "replay_psnr": oracle["psnr"],
                "delta_psnr": oracle["psnr"] - baseline["psnr"],
                "reference_ssim": baseline["ssim"],
                "replay_ssim": oracle["ssim"],
                "delta_ssim": oracle["ssim"] - baseline["ssim"],
                "reference_lpips": baseline["lpips"],
                "replay_lpips": oracle["lpips"],
                "delta_lpips": oracle["lpips"] - baseline["lpips"],
            })


def main():
    args = parse_args()
    reference = load_metrics(args.reference_metrics)
    replay_specs = parse_replay_specs(args)
    replays = [(label, path, load_metrics(path)) for label, path in replay_specs]

    reference_gt = {row["G_t"] for row in reference.values()}
    reference_keys = set(reference)
    if len(reference_gt) != 1:
        raise ValueError(f"reference 的 G_t 不唯一: {reference_gt}")
    if not reference_keys:
        raise ValueError("reference 没有检查点")

    # 每个 treatment 都必须与 reference 使用完全相同的检查点；否则一张叠加图会把
    # 不同 G 或不同关键帧的值错当成可比较曲线。
    for label, _, replay in replays:
        replay_gt = {row["G_t"] for row in replay.values()}
        if reference_gt != replay_gt:
            raise ValueError(
                f"{label} 的 G_t 不一致: reference={reference_gt}, replay={replay_gt}")
        replay_keys = set(replay)
        reference_only_keys = reference_keys.difference(replay_keys)
        replay_only_keys = replay_keys.difference(reference_keys)
        if reference_only_keys or replay_only_keys:
            raise ValueError(
                f"{label} 的 (eval_frame_id, G) 检查点集合不一致: "
                f"reference_only={len(reference_only_keys)}, replay_only={len(replay_only_keys)}")
        for key in reference_keys:
            baseline = reference[key]
            oracle = replay[key]
            if baseline["image_name"] != oracle["image_name"]:
                raise ValueError(f"{label}: frame={key[0]}, G={key[1]} 的 image_name 不一致")
            if baseline["eval_keyframe_index"] != oracle["eval_keyframe_index"]:
                raise ValueError(f"{label}: frame={key[0]}, G={key[1]} 的关键帧序号不一致")
            if baseline["relative_updates"] != oracle["relative_updates"]:
                raise ValueError(f"{label}: frame={key[0]}, G={key[1]} 的相对更新次数不一致")

    common_keys = sorted(reference_keys, key=lambda item: (item[0], item[1]))

    args.output_dir.mkdir(parents=True, exist_ok=False)
    for label, _, replay in replays:
        # 单 treatment 沿用历史文件名，避免打断已有使用方；多 treatment 时按标签拆分。
        paired_name = "paired_metrics.csv" if len(replays) == 1 else f"paired_metrics_{label_stem(label)}.csv"
        write_paired_metrics(args.output_dir / paired_name, reference, replay, common_keys)

    frame_ids = sorted({frame_id for frame_id, _ in common_keys})
    for frame_id in frame_ids:
        frame_keys = [key for key in common_keys if key[0] == frame_id]
        global_updates = [key[1] for key in frame_keys]
        image_name = reference[frame_keys[0]]["image_name"]
        figure, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
        for axis, metric in zip(axes, METRICS):
            axis.plot(
                global_updates,
                [reference[key][metric] for key in frame_keys],
                label="baseline reference",
                linewidth=1.4,
            )
            for label, _, replay in replays:
                axis.plot(
                    global_updates,
                    [replay[key][metric] for key in frame_keys],
                    label=label,
                    linewidth=1.25,
                )
            axis.set_ylabel(metric.upper())
            axis.grid(alpha=0.25)
        axes[0].legend()
        axes[-1].set_xlabel("Absolute global single-view update count G")
        figure.suptitle(f"{image_name}: baseline vs Oracle teacher replay variants")
        figure.tight_layout()
        figure.savefig(args.output_dir / f"frame_{frame_id:04d}_curves.png", dpi=160)
        plt.close(figure)

    with (args.output_dir / "pairing_report.txt").open("w", encoding="utf-8") as stream:
        stream.write(f"paired_rows: {len(common_keys)}\n")
        stream.write(f"paired_frames: {len(frame_ids)}\n")
        stream.write("reference_only_rows: 0\n")
        stream.write("replay_only_rows: 0\n")
        stream.write(f"G_t: {next(iter(reference_gt))}\n")
        stream.write("replay_variants:\n")
        for label, path, _ in replays:
            stream.write(f"  {label}: {path}\n")


if __name__ == "__main__":
    main()
