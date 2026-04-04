import csv
import argparse
import sys

def load_csv(file_path):
    """读取CSV文件并将数据存入字典，以 (image_name, frame_id, subset_index) 为主键"""
    data = {}
    try:
        with open(file_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row['image_name'], row['frame_id'], row['subset_index'])
                data[key] = {
                    'psnr': float(row['psnr']),
                    'ssim': float(row['ssim']),
                    'lpips': float(row['lpips'])
                }
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        sys.exit(1)
    return data

def main():
    parser = argparse.ArgumentParser(description="比较 Baseline 和 Proposed 的图像评价指标。")
    parser.add_argument("baseline", help="Baseline CSV 文件的路径")
    parser.add_argument("proposed", help="Proposed CSV 文件的路径")
    args = parser.parse_args()

    # 加载数据
    baseline_data = load_csv(args.baseline)
    proposed_data = load_csv(args.proposed)

    print(f"{'Image Name':<18} | {'Frame':<5} | {'Metric':<5} | {'Baseline':<12} | {'Proposed':<12} | {'Result'}")
    print("-" * 80)

    count_better_images = 0

    for key, base_metrics in baseline_data.items():
        if key not in proposed_data:
            continue  # 如果proposed里没有这张图，则跳过

        prop_metrics = proposed_data[key]
        image_name, frame_id, _ = key
        
        better_metrics = []

        # 1. PSNR: 越大越好 (Baseline > Proposed 则说明 Baseline 更好)
        if base_metrics['psnr'] > prop_metrics['psnr']:
            better_metrics.append(('PSNR', base_metrics['psnr'], prop_metrics['psnr']))

        # 2. SSIM: 越大越好 (Baseline > Proposed 则说明 Baseline 更好)
        if base_metrics['ssim'] > prop_metrics['ssim']:
            better_metrics.append(('SSIM', base_metrics['ssim'], prop_metrics['ssim']))

        # 3. LPIPS: 越小越好 (Baseline < Proposed 则说明 Baseline 更好)
        if base_metrics['lpips'] < prop_metrics['lpips']:
            better_metrics.append(('LPIPS', base_metrics['lpips'], prop_metrics['lpips']))

        # 如果存在任意一个指标 Baseline 更好，则打印输出
        if better_metrics:
            count_better_images += 1
            for metric, b_val, p_val in better_metrics:
                print(f"{image_name:<18} | {frame_id:<5} | {metric:<5} | {b_val:<12.8f} | {p_val:<12.8f} | Baseline Better")

    print("-" * 80)
    print(f"总计: 有 {count_better_images} 张图像/帧在至少一个指标上 Baseline 优于 Proposed。")

if __name__ == "__main__":
    main()