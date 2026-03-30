import argparse
import os
import sys
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# 动态将模型路径加入环境变量
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "..", "src")
sys.path.append(src_dir)

from model.regressor.gaussian_mlp import GaussianMLP, GaussianMLPCfg
from dataset_sharded import get_sharded_dataloader

def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"train_{time.strftime('%Y%m%d_%H%M%S')}.log")
    
    logger = logging.getLogger("GaussianMLP_Trainer")
    logger.setLevel(logging.INFO)
    
    # 格式化输出
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%main:%S')
    
    # 文件 Handler
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # 控制台 Handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger, log_file

# =============== 定义各项 Loss 函数 ===============

def compute_scale_loss(pred_pixel_scale_raw, gt_log_world_scale, inv_depth, focal):
    s_pixel = pred_pixel_scale_raw + 1e-6
    if torch.isnan(s_pixel).any(): print("NaN in scale_loss: s_pixel")
    
    log_s_pixel = torch.log(s_pixel)
    if torch.isnan(log_s_pixel).any(): print("NaN in scale_loss: log_s_pixel")
    
    log_inv_depth = torch.log(inv_depth.clamp(min=1e-6))
    if torch.isnan(log_inv_depth).any(): print("NaN in scale_loss: log_inv_depth")
    
    log_focal = torch.log(torch.tensor(focal).to(inv_depth.device))
    
    pred_log_world_scale = log_s_pixel - log_inv_depth - log_focal
    if torch.isnan(pred_log_world_scale).any(): print("NaN in scale_loss: pred_log_world_scale")
    
    loss = F.l1_loss(pred_log_world_scale, gt_log_world_scale)
    if torch.isnan(loss): print("NaN in scale_loss: final loss")
    return loss

def compute_rotation_loss(pred_q, gt_q):
    # 余弦相似度计算四元数距离: 1 - |<q1, q2>|
    dot = torch.sum(pred_q * gt_q, dim=-1)
    if torch.isnan(dot).any(): print("NaN in rotation_loss: dot")
    
    loss = 1.0 - torch.abs(dot).mean()
    if torch.isnan(loss): print("NaN in rotation_loss: final loss")
    return loss

def compute_color_dc_loss(pred_dc, gt_dc):
    loss_dc = F.mse_loss(pred_dc.view(-1, 3), gt_dc.view(-1, 3))
    if torch.isnan(loss_dc): print("NaN in color_loss: loss_dc")
    return loss_dc

def compute_color_rest_loss(pred_rest, gt_rest):
    loss_rest = F.mse_loss(pred_rest.view(-1, 45), gt_rest.view(-1, 45))
    if torch.isnan(loss_rest): print("NaN in color_loss: loss_rest")
    return loss_rest


def compute_opacity_loss(pred_alpha, gt_alpha):
    loss = F.l1_loss(pred_alpha, gt_alpha)
    if torch.isnan(loss): print("NaN in opacity_loss: final loss")
    return loss

def compute_position_loss(pred_delta_pixel, pred_delta_inv_depth, 
                          base_pixel, base_inv_depth, 
                          gt_xyz, cam_params, R_wc, t_wc):
    """
    Project GT_XYZ to camera frame and compute the delta in 2.5D space.
    P_cam = R_cw * P_world + t_cw
    We have R_wc (C2W rot) and t_wc (C2W trans).
    So: R_cw = R_wc^T, t_cw = -R_cw * t_wc
    """
    fx = cam_params[:, 0:1]
    fy = cam_params[:, 1:2]
    cx = cam_params[:, 2:3]
    cy = cam_params[:, 3:4]
    
    # 1. World to Cam
    R_cw = R_wc.transpose(1, 2) # (N, 3, 3)
    # gt_xyz: (N, 3), t_wc: (N, 3)
    p_world = gt_xyz.unsqueeze(-1) # (N, 3, 1)
    t_wc_vec = t_wc.unsqueeze(-1)  # (N, 3, 1)
    
    t_cw = -torch.bmm(R_cw, t_wc_vec) # (N, 3, 1)
    p_cam = torch.bmm(R_cw, p_world) + t_cw # (N, 3, 1)
    p_cam = p_cam.squeeze(-1) # (N, 3)
    
    # 2. Extract gt inv_depth and gt pixel
    z_cam = p_cam[:, 2:3].clamp(min=1e-6)
    gt_inv_depth = 1.0 / z_cam
    
    x_cam = p_cam[:, 0:1]
    y_cam = p_cam[:, 1:2]
    gt_u = (x_cam * fx) / z_cam + cx
    gt_v = (y_cam * fy) / z_cam + cy
    gt_pixel = torch.cat([gt_u, gt_v], dim=-1) # (N, 2)
    
    # 3. Compute GT deltas
    gt_delta_pixel = gt_pixel - base_pixel
    gt_delta_inv_depth = gt_inv_depth - base_inv_depth
    
    # 4. Losses
    # Use nan_to_num to avoid total loss explosion if an outlier exists
    gt_delta_pixel = torch.nan_to_num(gt_delta_pixel, posinf=0.0, neginf=0.0)
    gt_delta_inv_depth = torch.nan_to_num(gt_delta_inv_depth, posinf=0.0, neginf=0.0)
    
    loss_pixel = F.l1_loss(pred_delta_pixel, gt_delta_pixel)
    loss_inv_depth = F.l1_loss(pred_delta_inv_depth, gt_delta_inv_depth)
    
    if torch.isnan(loss_pixel) or torch.isnan(loss_inv_depth):
        return torch.tensor(0.0, device=gt_xyz.device), torch.tensor(0.0, device=gt_xyz.device)
    
    return loss_pixel, loss_inv_depth

# =============== 验证函数 ===============

@torch.no_grad()
def evaluate(mlp, cfg, val_loader, device, args, logger, writer, global_step):
    mlp.eval()
    val_loss = 0.0
    val_l_scale = 0.0
    val_l_rot = 0.0
    val_l_sh_dc = 0.0
    val_l_sh_rest = 0.0
    val_l_opacity = 0.0
    val_l_pixel = 0.0
    val_l_inv_depth = 0.0
    val_steps = 0
    
    logger.info(f"--- 🔍 Starting Evaluation at Step {global_step} ---")
    
    for X_batch, Y_batch in val_loader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)
        current_batch_size = X_batch.shape[0]
        
        # W * 260 matrix values start at column 256.
        # Constant inputs: f_curr(128) + render_f_curr(128) = 256
        # Post-matrix constant inputs: curr_dir(3) + inv_depth(1) + n_cam_in(3) + mask(1) + R_wc(9) + base_sh(3) + cam_params(6) + t_wc(3) + base_pixel(2) + dis(1) = 32
        # Total constants = 256 + 32 = 288
        W = (X_batch.shape[1] - 288) // 260
        
        f_curr         = X_batch[:, 0:128]
        render_f_curr  = X_batch[:, 128:256]
        hist_f         = X_batch[:, 256:256 + W*128].view(-1, W, 128)
        hist_render_f  = X_batch[:, 256 + W*128:256 + W*256].view(-1, W, 128)
        
        curr_dir       = X_batch[:, 256 + W*256 : 256 + W*256 + 3]
        hist_dir       = X_batch[:, 256 + W*256 + 3 : 256 + W*256 + 3 + W*3].view(-1, W, 3)
        hist_mask      = X_batch[:, 256 + W*256 + 3 + W*3 : 256 + W*256 + 3 + W*4].view(-1, W, 1)
        
        c = 256 + W*256 + 3 + W*4
        inv_depth      = X_batch[:, c : c+1]
        n_cam_in       = X_batch[:, c+1 : c+4]
        mask_t         = X_batch[:, c+4 : c+5]
        R_wc           = X_batch[:, c+5 : c+14].view(-1, 3, 3)
        
        base_sh        = X_batch[:, c+14 : c+17] if X_batch.shape[1] >= c+17 else torch.zeros((current_batch_size, 3), device=device)
        cam_params     = X_batch[:, c+17 : c+23] if X_batch.shape[1] >= c+23 else torch.zeros((current_batch_size, 6), device=device)
        t_wc           = X_batch[:, c+23 : c+26] if X_batch.shape[1] >= c+26 else torch.zeros((current_batch_size, 3), device=device)
        base_pixel     = X_batch[:, c+26 : c+28] if X_batch.shape[1] >= c+28 else torch.zeros((current_batch_size, 2), device=device)
        dis            = X_batch[:, c+28 : c+29] if X_batch.shape[1] >= c+29 else torch.zeros((current_batch_size, 1), device=device)
        
        pred_scale, pred_rot, pred_opacity, pred_dc_residual, pred_rest, pred_delta_pixel, pred_delta_inv_depth = mlp(
            f_curr=f_curr, render_f_curr=render_f_curr,
            hist_f=hist_f, hist_render_f=hist_render_f,
            curr_dir=curr_dir, hist_dir=hist_dir, hist_mask=hist_mask,
            inv_depth=inv_depth, n_cam_in=n_cam_in,
            mask=mask_t, dis=dis, R_wc=R_wc
        )
        
        pred_dc = pred_dc_residual + base_sh.unsqueeze(1)
        
        gt_f_dc    = Y_batch[:, 0:3]
        gt_f_rest  = Y_batch[:, 3:48]
        gt_opacity = Y_batch[:, 48:49]
        
        scale_dim = cfg.scale_dim
        gt_scale = Y_batch[:, 49 : 49+scale_dim]  
        gt_rot   = Y_batch[:, 49+scale_dim : 49+scale_dim+4]
        gt_xyz   = Y_batch[:, 49+scale_dim+4 : 49+scale_dim+4+3] if Y_batch.shape[1] >= 49+scale_dim+4+3 else torch.zeros((current_batch_size, 3), device=device)
        
        l_scale   = compute_scale_loss(pred_scale, gt_scale, inv_depth, args.focal_length)
        l_rot     = compute_rotation_loss(pred_rot, gt_rot)
        l_sh_dc   = compute_color_dc_loss(pred_dc, gt_f_dc)
        l_sh_rest = compute_color_rest_loss(pred_rest, gt_f_rest)
        l_opacity = compute_opacity_loss(pred_opacity, gt_opacity)
        
        # Position Loss
        if Y_batch.shape[1] >= 49+scale_dim+4+3 and X_batch.shape[1] >= 796:
            l_pixel, l_inv_depth = compute_position_loss(pred_delta_pixel, pred_delta_inv_depth, base_pixel, inv_depth, gt_xyz, cam_params, R_wc, t_wc)
        else:
            l_pixel, l_inv_depth = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
            
        total_loss = args.lambda_scale * l_scale + args.lambda_rot * l_rot + args.lambda_dc * l_sh_dc + args.lambda_rest * l_sh_rest + args.lambda_opac * l_opacity + args.lambda_pos * (l_pixel + l_inv_depth)
        
        val_loss += total_loss.item()
        val_l_scale += l_scale.item()
        val_l_rot += l_rot.item()
        val_l_sh_dc += l_sh_dc.item()
        val_l_sh_rest += l_sh_rest.item()
        val_l_opacity += l_opacity.item()
        val_l_pixel += l_pixel.item()
        val_l_inv_depth += l_inv_depth.item()
        val_steps += 1
        
        if val_steps >= args.val_batches:
            break
            
    if val_steps > 0:
        val_loss /= val_steps
        val_l_scale /= val_steps
        val_l_rot /= val_steps
        val_l_sh_dc /= val_steps
        val_l_sh_rest /= val_steps
        val_l_opacity /= val_steps
        val_l_pixel /= val_steps
        val_l_inv_depth /= val_steps
        
        logger.info(f"✔️ Validation Results - Avg Total Loss: {val_loss:.5f} | "
                    f"Sc: {val_l_scale:.4f}, Rot: {val_l_rot:.4f}, "
                    f"SH: {val_l_sh_dc:.4f}, Op: {val_l_opacity:.4f}, "
                    f"Px: {val_l_pixel:.4f}, IDep: {val_l_inv_depth:.4f}")
        
        # ⚠️ 明确区分 Train 与 Val 前缀，确保 Tensorboard 曲线独立
        writer.add_scalar("Loss/Total_Val", val_loss, global_step)
        writer.add_scalar("Loss/Scale_Val", val_l_scale, global_step)
        writer.add_scalar("Loss/Rotation_Val", val_l_rot, global_step)
        writer.add_scalar("Loss/SH_Color_Val", val_l_sh_dc, global_step)
        writer.add_scalar("Loss/SH_Rest_Val", val_l_sh_rest, global_step)
        writer.add_scalar("Loss/Opacity_Val", val_l_opacity, global_step)
        writer.add_scalar("Loss/Pixel_Val", val_l_pixel, global_step)
        writer.add_scalar("Loss/InvDepth_Val", val_l_inv_depth, global_step)
    
    mlp.train() 

# =============== 主训练函数 ===============

def train(args):
    # 1. 准备环境与验证
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    chkpt_dir = os.path.join(args.save_dir, "checkpoints")
    tb_dir = os.path.join(args.save_dir, "tensorboard")
    os.makedirs(chkpt_dir, exist_ok=True)
    
    logger, log_file_path = setup_logger(args.save_dir)
    writer = SummaryWriter(log_dir=tb_dir)
    
    logger.info("="*50)
    logger.info(f"🚀 Starting Gaussian Regression Training on {device}")
    logger.info(f"📂 Shards Directory: {args.shards_dir}")
    logger.info(f"💾 Checkpoints Output: {chkpt_dir}")
    logger.info(f"📊 TensorBoard Output: {tb_dir}")
    logger.info("="*50)

    # 2. 构建模型与优化器
    cfg = GaussianMLPCfg(d_in=128, d_hidden=512, n_layers=5, sh_degree=3, scale_dim=2)
    mlp = GaussianMLP(cfg).to(device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=args.lr)
    
    start_epoch = 0
    global_step = 0
    
    # 3. 恢复断点 (如启用)
    if args.resume:
        if os.path.exists(args.resume):
            logger.info(f"🔄 Resuming from Checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
            mlp.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            global_step = checkpoint['global_step']
            logger.info(f"✅ State restored. Resuming at Epoch {start_epoch}, Step {global_step}")
        else:
            logger.warning(f"⚠️ Checkpoint not found: {args.resume}. Starting from scratch.")

    # 4. 初始化 DataLoader 的 Generator
    train_loader = get_sharded_dataloader(
        args.shards_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        buffer_size_mb=args.buffer_size_mb,
        max_buffer_points=args.max_buffer_points,
    )
    
    val_loader = None
    if args.val_shards_dir:
        logger.info(f"📂 Validation Shards Directory (Found): {args.val_shards_dir}")
        from dataset_sharded import StreamingShardedDataset
        from torch.utils.data import DataLoader
        val_dataset = StreamingShardedDataset(
            args.val_shards_dir,
            buffer_size_mb=args.buffer_size_mb,
            max_buffer_points=args.max_buffer_points,
        )
        # 固定使用0 worker防止竞争
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0)

    # 5. 开启训练循环
    for epoch in range(start_epoch, args.epochs):
        mlp.train()
        logger.info(f"\n▶️ Starting Epoch {epoch + 1}/{args.epochs}")
        
        # NOTE: 我们的 dataset_sharded.py 虽然是一个无限流式 IterableDataset，
        # 但是只要硬盘里所有的 `.pt` 被读空一轮，它就会 raise StopIteration
        # 从而自然地通过 enumerate 结束当前 Epoch。然后下个 Epoch DataLoader 会重新实例化它。
        epoch_loss = 0.0
        steps_in_epoch = 0
        
        for batch_idx, (X_batch, Y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            current_batch_size = X_batch.shape[0]
            
            # --- 解析输入 X ---
            W = (X_batch.shape[1] - 288) // 260
            if X_batch.shape[1] < 288 + W*260:
                raise ValueError(f"Feature dimension too small: {X_batch.shape[1]}. 288 + {W}*260 is needed for Cross-Attention fusion.")

            # 仅在训练开始时打印一次维度和理论张量大小，便于排查 OOM 根因。
            if epoch == start_epoch and batch_idx == 0:
                bytes_per_sample = (X_batch.shape[1] + Y_batch.shape[1]) * 4
                approx_batch_mb = bytes_per_sample * current_batch_size / (1024 * 1024)
                logger.info(
                    f"[DataProfile] X_dim={X_batch.shape[1]}, Y_dim={Y_batch.shape[1]}, "
                    f"W={W}, batch_size={current_batch_size}, "
                    f"approx_tensor_mem={approx_batch_mb:.2f} MB (X+Y only)"
                )
            
            f_curr         = X_batch[:, 0:128]
            render_f_curr  = X_batch[:, 128:256]
            hist_f         = X_batch[:, 256:256 + W*128].view(-1, W, 128)
            hist_render_f  = X_batch[:, 256 + W*128:256 + W*256].view(-1, W, 128)
            
            curr_dir       = X_batch[:, 256 + W*256 : 256 + W*256 + 3]
            hist_dir       = X_batch[:, 256 + W*256 + 3 : 256 + W*256 + 3 + W*3].view(-1, W, 3)
            hist_mask      = X_batch[:, 256 + W*256 + 3 + W*3 : 256 + W*256 + 3 + W*4].view(-1, W, 1)
            
            c = 256 + W*256 + 3 + W*4
            inv_depth      = X_batch[:, c : c+1]
            n_cam_in       = X_batch[:, c+1 : c+4]
            mask_t         = X_batch[:, c+4 : c+5]
            R_wc           = X_batch[:, c+5 : c+14].view(-1, 3, 3)
            
            base_sh        = X_batch[:, c+14 : c+17] if X_batch.shape[1] >= c+17 else torch.zeros((current_batch_size, 3), device=device)
            cam_params     = X_batch[:, c+17 : c+23] if X_batch.shape[1] >= c+23 else torch.zeros((current_batch_size, 6), device=device)
            t_wc           = X_batch[:, c+23 : c+26] if X_batch.shape[1] >= c+26 else torch.zeros((current_batch_size, 3), device=device)
            base_pixel     = X_batch[:, c+26 : c+28] if X_batch.shape[1] >= c+28 else torch.zeros((current_batch_size, 2), device=device)
            dis            = X_batch[:, c+28 : c+29] if X_batch.shape[1] >= c+29 else torch.zeros((current_batch_size, 1), device=device)
            
            # --- 前向传播 ---
            pred_scale, pred_rot, pred_opacity, pred_dc_residual, pred_rest, pred_delta_pixel, pred_delta_inv_depth = mlp(
                f_curr=f_curr, render_f_curr=render_f_curr,
                hist_f=hist_f, hist_render_f=hist_render_f,
                curr_dir=curr_dir, hist_dir=hist_dir, hist_mask=hist_mask,
                inv_depth=inv_depth, n_cam_in=n_cam_in,
                mask=mask_t, dis=dis, R_wc=R_wc
            )
            
            pred_dc = pred_dc_residual + base_sh.unsqueeze(1)
            
            # --- 解析 GT 标签 Y ---
            # Y: f_dc(3) | f_rest(45) | opacity(1) | scale(2 或 3) | rot(4)
            gt_f_dc    = Y_batch[:, 0:3]
            gt_f_rest  = Y_batch[:, 3:48]
            gt_opacity = Y_batch[:, 48:49]
            
            scale_dim = cfg.scale_dim
            gt_scale = Y_batch[:, 49 : 49+scale_dim]  
            gt_rot   = Y_batch[:, 49+scale_dim : 49+scale_dim+4]
            gt_xyz   = Y_batch[:, 49+scale_dim+4 : 49+scale_dim+4+3] if Y_batch.shape[1] >= 49+scale_dim+4+3 else torch.zeros((current_batch_size, 3), device=device)
            
            # --- 计算 Loss ---
            l_scale   = compute_scale_loss(pred_scale, gt_scale, inv_depth, args.focal_length)
            l_rot     = compute_rotation_loss(pred_rot, gt_rot)
            l_sh_dc   = compute_color_dc_loss(pred_dc, gt_f_dc)
            l_sh_rest = compute_color_rest_loss(pred_rest, gt_f_rest)
            l_opacity = compute_opacity_loss(pred_opacity, gt_opacity)
            
            # Position Loss
            if Y_batch.shape[1] >= 49+scale_dim+4+3 and X_batch.shape[1] >= 412:
                l_pixel, l_inv_depth = compute_position_loss(pred_delta_pixel, pred_delta_inv_depth, base_pixel, inv_depth, gt_xyz, cam_params, R_wc, t_wc)
            else:
                l_pixel, l_inv_depth = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
                
            total_loss = args.lambda_scale * l_scale + args.lambda_rot * l_rot + args.lambda_dc * l_sh_dc + args.lambda_rest * l_sh_rest + args.lambda_opac * l_opacity + args.lambda_pos * (l_pixel + l_inv_depth)
            
            if torch.isnan(total_loss):
                nan_components = []
                if torch.isnan(l_scale): nan_components.append("Scale Loss (Sc)")
                if torch.isnan(l_rot): nan_components.append("Rotation Loss (Rot)")
                if torch.isnan(l_sh_dc) or torch.isnan(l_sh_rest): nan_components.append("SH Color Loss (SH)")
                if torch.isnan(l_opacity): nan_components.append("Opacity Loss (Op)")
                if torch.isnan(l_pixel): nan_components.append("Pixel Position Loss (Px)")
                if torch.isnan(l_inv_depth): nan_components.append("Inv. Depth Loss (IDep)")
                
                print(f"\n[WARNING] NaN Output Detected in Step {global_step}!")
                print("NaN generated by:", ", ".join(nan_components) if nan_components else "Network Internal (Likely exploding Gradients)")
                print(f"Loss Values -> Sc: {l_scale.item()}, Rot: {l_rot.item()}, SH_DC: {l_sh_dc.item()}, SH_Rest: {l_sh_rest.item()}, Op: {l_opacity.item()}, Px: {l_pixel.item()}, IDep: {l_inv_depth.item()}\n")
            else:
                # --- 反向传播 ---
                optimizer.zero_grad()
                total_loss.backward()
                
                # 实施梯度裁剪防止爆炸
                torch.nn.utils.clip_grad_norm_(mlp.parameters(), max_norm=1.0)
                
                optimizer.step()
            
            epoch_loss += total_loss.item()
            steps_in_epoch += 1
            global_step += 1
            
            # --- 录入与展示 ---
            if batch_idx % args.log_interval == 0:
                logger.info(
                    f"Epoch [{epoch+1}/{args.epochs}] Step [{batch_idx:04d}] "
                    f"Total Loss: {total_loss.item():.5f} | "
                    f"Sc: {l_scale.item():.4f}, Rot: {l_rot.item():.4f}, "
                    f"SH_DC: {l_sh_dc.item():.4f}, SH_Rest: {l_sh_rest.item():.4f}, Op: {l_opacity.item():.4f}, "
                    f"Px: {l_pixel.item():.4f}, IDep: {l_inv_depth.item():.4f}"
                )
                
                # 写入 TensorBoard，并使用明确区分的前缀
                writer.add_scalar("Loss/Total_Train", total_loss.item(), global_step)
                writer.add_scalar("Loss/Scale_Train", l_scale.item(), global_step)
                writer.add_scalar("Loss/Rotation_Train", l_rot.item(), global_step)
                writer.add_scalar("Loss/SH_Color_Train", l_sh_dc.item(), global_step)
                writer.add_scalar("Loss/SH_Rest_Train", l_sh_rest.item(), global_step)
                writer.add_scalar("Loss/Opacity_Train", l_opacity.item(), global_step)
                writer.add_scalar("Loss/Pixel_Train", l_pixel.item(), global_step)
                writer.add_scalar("Loss/InvDepth_Train", l_inv_depth.item(), global_step)
        
        # Epoch 结束后的统计
        avg_epoch_loss = epoch_loss / max(1, steps_in_epoch)
        logger.info(f"✔️ Epoch {epoch+1} Completed. Average Loss: {avg_epoch_loss:.5f}")
        writer.add_scalar("Loss/Epoch_Average_Train", avg_epoch_loss, epoch + 1)
        
        # 存取 Checkpoint 与执行评估
        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
            
            # --- 存盘前执行一次统一 Eval ---
            if val_loader is not None:
                evaluate(mlp, cfg, val_loader, device, args, logger, writer, global_step)
                
            chkpt_name = os.path.join(chkpt_dir, f"mlp_epoch_{epoch+1:03d}.pth")
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': mlp.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, chkpt_name)
            logger.info(f"💾 Checkpoint saved to: {chkpt_name}")

    writer.close()
    logger.info("🎉 All Epochs completed successfully. Training Terminated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Scenario Formal Regressor Training Script")
    parser.add_argument("--shards_dir", type=str, default="/home/lab404/dataset/gs_rg_r3live/shards", help="Path to precomputed `.pt` buckets for training")
    parser.add_argument("--val_shards_dir", type=str, default="", help="Path to validation buckets. If empty, validation is skipped.")
    parser.add_argument("--save_dir", type=str, default="/home/lab404/dataset/gs_rg_r3live/training_results", help="Directory to save checkpoints, logs and tb data")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint `.pth` to resume training from")
    
    # 核心训练参数
    parser.add_argument("--epochs", type=int, default=50, help="Total number of epochs to run")
    parser.add_argument("--batch_size", type=int, default=8192, help="Batch size per forwarding")
    parser.add_argument("--lr", type=float, default=1e-3, help="Optimizer Learning Rate")
    parser.add_argument("--num_workers", type=int, default=0, help="Dataloader workers (keep 0-2 for Sharded buffered loads)")
    parser.add_argument("--buffer_size_mb", type=int, default=512, help="CPU RAM budget for shard shuffle buffer (MB)")
    parser.add_argument("--max_buffer_points", type=int, default=300000, help="Hard cap for shuffle buffer points")
    
    # 相机标定先验
    parser.add_argument("--focal_length", type=float, default=431.7103, help="Camera focal length for scale un-projection")
    
    # 损失权重惩罚项
    parser.add_argument("--lambda_scale", type=float, default=1.0)
    parser.add_argument("--lambda_rot", type=float, default=0.1)
    parser.add_argument("--lambda_dc", type=float, default=1.0)
    parser.add_argument("--lambda_rest", type=float, default=0.1)
    parser.add_argument("--lambda_opac", type=float, default=0.5)
    parser.add_argument("--lambda_pos", type=float, default=1.0, help="Weight for 2.5D position loss")
    
    # 日志输出频次与测试
    parser.add_argument("--log_interval", type=int, default=10, help="Print log every X steps")
    parser.add_argument("--val_batches", type=int, default=50, help="Number of batches to evaluate during validation phase")
    parser.add_argument("--save_interval", type=int, default=5, help="Save .pth checkpoint every X epochs")
    
    args = parser.parse_args()
    
    train(args)
