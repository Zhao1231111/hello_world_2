import sys
import os
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def compute_color_gradient(image_path):
    """
    复现 C++ 中 Phase 3.2 提取颜色梯度的逻辑
    """
    # 读取图像，OpenCV默认使用BGR
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Error: 无法读取图像 {image_path}")
        return None, None
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 转换为 float，范围限制在 [0, 1], shape: (C, H, W)
    # 这与 C++ 中 viewpoint_cam->original_image_ 的格式相同
    tensor_img = torch.from_numpy(img_rgb).float() / 255.0
    tensor_img = tensor_img.permute(2, 0, 1) 
    
    # 1. 转换为灰度图: 0.299*R + 0.587*G + 0.114*B
    gray_img = 0.299 * tensor_img[0] + 0.587 * tensor_img[1] + 0.114 * tensor_img[2]
    
    # 2. padding (mode='replicate')
    padded_gray = F.pad(gray_img.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='replicate').squeeze(0).squeeze(0)
    
    # 3. 相邻像素差分 (中点差分)
    # grad_x = (Right - Left) / 2.0
    grad_x = (padded_gray[1:-1, 2:] - padded_gray[1:-1, 0:-2]) / 2.0
    # grad_y = (Bottom - Top) / 2.0
    grad_y = (padded_gray[2:, 1:-1] - padded_gray[0:-2, 1:-1]) / 2.0
    
    # 4. 梯度幅度
    grad_mag = torch.sqrt(grad_x * grad_x + grad_y * grad_y)
    
    return img_rgb, grad_mag.numpy()

def main():
    parser = argparse.ArgumentParser(description="颜色梯度阈值辅助选择工具 (Color Gradient Threshold Helper)")
    parser.add_argument('image_path', type=str, help="输入图像路径 (用于计算梯度的样本图像)")
    parser.add_argument('--save', type=str, default=None, help="如果提供，将热力图结果保存至此路径而不是直接展示")
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"错误: 找不到文件 {args.image_path}")
        sys.exit(1)
        
    img_rgb, grad_mag = compute_color_gradient(args.image_path)
    if img_rgb is None:
        sys.exit(1)
        
    # 基于结果绘制
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 绘制原图
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].imshow(img_rgb)
    axes[0].axis('off')
    
    # 绘制梯度热力图
    axes[1].set_title("Color Gradient Magnitude", fontsize=14)
    # 使用 jet 热力图，通常蓝色低、红色高
    im = axes[1].imshow(grad_mag, cmap='jet')
    axes[1].axis('off')
    
    # 在热力图右侧添加 Colorbar
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('Gradient Magnitude', rotation=270, labelpad=20, fontsize=12)
    
    plt.tight_layout()
    
    # 展示或保存图像
    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches='tight')
        print(f"热力图已成功保存至: {args.save}")
    else:
        print("展示热力图 (请关闭窗口以结束程序)...")
        plt.show()
        
    # 输出一些统计信息供参考
    print("\n--- 梯度统计信息 ---")
    print(f"最小值 (Min)  : {np.min(grad_mag):.4f}")
    print(f"最大值 (Max)  : {np.max(grad_mag):.4f}")
    print(f"平均值 (Mean) : {np.mean(grad_mag):.4f}")
    print(f"中位数 (Median): {np.median(grad_mag):.4f}")
    print("建议：你可以选择处于你希望保留边缘的最小梯度幅度作为阈值 color_gradient_threshold")

if __name__ == '__main__':
    main()
