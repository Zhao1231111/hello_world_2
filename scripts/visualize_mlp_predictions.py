import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm
import argparse

# Setup paths
current_file = os.path.abspath(__file__)
mvsplat_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
sys.path.append(mvsplat_root)

# Import local GaussianMLP
sys.path.append(os.path.join(os.path.dirname(current_file), "../src/model/regressor"))
from gaussian_mlp import GaussianMLP, GaussianMLPCfg

def save_ply(path, xyz, f_dc, f_rest, opacities, scales, rotations):
    """
    Save 3D (or 2.5D) Gaussian Splatting attributes into a standard binary .ply file.
    Args:
        xyz: (N, 3) numpy array
        f_dc: (N, 1, 3) numpy array
        f_rest: (N, K, 3) numpy array  (K is usually 15 for degree 3)
        opacities: (N, 1) numpy array
        scales: (N, 2) or (N, 3) numpy array
        rotations: (N, 4) numpy array
    """
    N = xyz.shape[0]

    # Reshape f_dc, f_rest
    f_dc = f_dc.reshape((N, 3))
    f_rest = f_rest.reshape((N, -1))
    
    # Check if scale is 2D or 3D
    if scales.shape[1] == 2:
        # pad to 3 scales with a tiny scale for Z
        scales_padded = np.zeros((N, 3), dtype=np.float32)
        scales_padded[:, :2] = scales
        scales_padded[:, 2] = -10.0 # heavily squashed if converted via exp(), but since scales are already exp'd in MLP, let's use tiny value or 0.0001
        # Actually our scale is directly the true positive scale! Not log scaler.
        scales_padded[:, 2] = 0.00001
        scales = scales_padded
    
    # Typically ply structure for GS:
    # x, y, z, nx, ny, nz, f_dc_0, f_dc_1, f_dc_2, f_rest_0...44, opacity, scale_0...2, rot_0...3
    # Note: the MLP outputs scale as actual value. Most viewers expect log-scale in PLY.
    # Therefore we apply np.log to scale.
    scales_log = np.log(scales + 1e-8)
    
    # We apply inverse sigmoid to opacity so viewers that apply sigmoid won't double sigmoid it
    # opacities are true [0,1]. Use inverse_sigmoid:
    opacities_inv = np.clip(opacities, 1e-6, 1.0 - 1e-6)
    opacities_inv = np.log(opacities_inv / (1 - opacities_inv))

    # Define dtype for structured array
    dtype_list = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), 
                  ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')]
    
    for i in range(3): dtype_list.append((f'f_dc_{i}', 'f4'))
    for i in range(f_rest.shape[1]): dtype_list.append((f'f_rest_{i}', 'f4'))
    
    dtype_list.append(('opacity', 'f4'))
    
    for i in range(3): dtype_list.append((f'scale_{i}', 'f4'))
    for i in range(4): dtype_list.append((f'rot_{i}', 'f4'))
    
    # Build data array
    elements = np.empty(N, dtype=dtype_list)
    elements['x'], elements['y'], elements['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    elements['nx'], elements['ny'], elements['nz'] = np.zeros(N), np.zeros(N), np.zeros(N)
    
    for i in range(3): elements[f'f_dc_{i}'] = f_dc[:, i]
    for i in range(f_rest.shape[1]): elements[f'f_rest_{i}'] = f_rest[:, i]
    
    elements['opacity'] = opacities_inv[:, 0]
    
    for i in range(3): elements[f'scale_{i}'] = scales_log[:, i]
    for i in range(4): elements[f'rot_{i}'] = rotations[:, i]
    
    # Write binary ply
    try:
        from plyfile import PlyData, PlyElement
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
    except ImportError:
        # Fallback raw binary write if plyfile not installed
        header = f"""ply
format binary_little_endian 1.0
element vertex {N}
property float x
property float y
property float z
property float nx
property float ny
property float nz
"""
        for i in range(3): header += f"property float f_dc_{i}\n"
        for i in range(f_rest.shape[1]): header += f"property float f_rest_{i}\n"
        header += "property float opacity\n"
        for i in range(3): header += f"property float scale_{i}\n"
        for i in range(4): header += f"property float rot_{i}\n"
        header += "end_header\n"
        
        with open(path, 'wb') as f:
            f.write(header.encode('ascii'))
            f.write(elements.tobytes())

def process_and_visualize(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure MLP matching the new architecture (Residual)
    cfg = GaussianMLPCfg(d_in=128, d_hidden=512, n_layers=5, sh_degree=3, scale_dim=2)
    mlp = GaussianMLP(cfg).to(device)
    
    if args.ckpt_path and os.path.exists(args.ckpt_path):
        print(f"Loading checkpoint from: {args.ckpt_path}")
        checkpoint = torch.load(args.ckpt_path, map_location=device, weights_only=True)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        mlp.load_state_dict(state_dict)
    else:
        print(f"[WARNING] Checkpoint {args.ckpt_path} not found! Using untrained model...")
        
    mlp.eval()
    
    # Load metadata (same as prepare_shards.py)
    meta_path = os.path.join(args.frames_dir, "../train_cameras.json")
    if not os.path.exists(meta_path):
        meta_path = os.path.join(args.frames_dir, "../test_cameras.json")
    
    meta_dict = {}
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta_data = json.load(f)
            for item in meta_data:
                meta_dict[item["seed_file"]] = item
    else:
        print(f"[WARNING] Meta dictionary {meta_path} missing. Geometric projection may be incorrect.")

    frame_files = [f for f in os.listdir(args.frames_dir) if f.endswith(".pt")]
    print(f"Found {len(frame_files)} frame shards to predict. Saving `.ply` to: {args.output_dir}")
    
    with torch.no_grad():
        for filename in tqdm(frame_files):
            pt_path = os.path.join(args.frames_dir, filename)
            
            try:
                script_module = torch.jit.load(pt_path, map_location=device)
                data_list = list(script_module.parameters())
            except Exception as e:
                print(f"Failed loading {filename}: {e}")
                continue
                
            if len(data_list) not in [11, 12]:
                continue
                
            if len(data_list) == 12:
                f_curr, ctx, render_f_curr, render_ctx, inv_depth, n_cam, mask_tensor, added_ids, R_wc, base_sh, base_pixel, dis = data_list
            else:
                f_curr, ctx, render_f_curr, render_ctx, inv_depth, n_cam, mask_tensor, added_ids, R_wc, base_sh, base_pixel = data_list
                dis = torch.zeros((added_ids.shape[0], 1), dtype=torch.float32, device=device)

            N = added_ids.shape[0]
            if N == 0: continue
            
            f_curr = f_curr.view(N, -1)
            ctx = ctx.view(N, -1)
            render_f_curr = render_f_curr.view(N, -1)
            render_ctx = render_ctx.view(N, -1)
            inv_depth = inv_depth.view(N, 1)
            n_cam = n_cam.view(N, 3)
            mask_tensor = mask_tensor.view(N, 1)
            R_wc = R_wc.view(N, 3, 3)
            base_sh = base_sh.view(N, 3)
            base_pixel = base_pixel.view(N, 2)
            dis = dis.view(N, 1)
            
            if filename in meta_dict:
                item = meta_dict[filename]
                c_params = [item["fx"], item["fy"], item["cx"], item["cy"], float(item["width"]), float(item["height"])]
                c_pos = item["position"]
                cam_params = torch.tensor(c_params, dtype=torch.float32, device=device).unsqueeze(0).expand(N, 6)
                t_wc = torch.tensor(c_pos, dtype=torch.float32, device=device).unsqueeze(0).expand(N, 3)
            else:
                cam_params = torch.zeros((N, 6), device=device)
                t_wc = torch.zeros((N, 3), device=device)

            pred_scale, pred_rot, pred_opacity, pred_dc_residual, pred_rest, pred_delta_pixel, pred_delta_inv_depth = mlp(
                f_curr=f_curr, ctx=ctx,
                render_f_curr=render_f_curr, render_ctx=render_ctx,
                inv_depth=inv_depth, n_cam_in=n_cam,
                mask=mask_tensor, dis=dis, R_wc=R_wc
            )
            
            # Form final DC
            pred_dc = pred_dc_residual + base_sh.unsqueeze(1)
            
            # ----------------------------------------------------
            #  2.5D To 3D Back-Projection
            # ----------------------------------------------------
            fx = cam_params[:, 0:1]
            fy = cam_params[:, 1:2]
            cx = cam_params[:, 2:3]
            cy = cam_params[:, 3:4]
            
            # Shift 2D Pixel and Depth
            u = base_pixel[:, 0:1] + pred_delta_pixel[:, 0:1]
            v = base_pixel[:, 1:2] + pred_delta_pixel[:, 1:2]
            
            final_inv_depth = inv_depth + pred_delta_inv_depth
            z_cam = 1.0 / final_inv_depth.clamp(min=1e-6)
            
            x_cam = (u - cx) * z_cam / fx
            y_cam = (v - cy) * z_cam / fy
            
            p_cam = torch.cat([x_cam, y_cam, z_cam], dim=-1).unsqueeze(-1) # (N, 3, 1)
            
            # P_world = R_wc * P_cam + t_wc
            p_world = torch.bmm(R_wc, p_cam) + t_wc.unsqueeze(-1) # (N, 3, 1)
            pred_xyz = p_world.squeeze(-1) # (N, 3)
            
            # Export to .ply
            out_name = filename.replace(".pt", ".ply")
            out_path = os.path.join(args.output_dir, out_name)
            
            save_ply(out_path, 
                     pred_xyz.cpu().numpy(), 
                     pred_dc.cpu().numpy(), 
                     pred_rest.cpu().numpy(), 
                     pred_opacity.cpu().numpy(), 
                     pred_scale.cpu().numpy(), 
                     pred_rot.cpu().numpy())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize and Extract Gaussian MLP outputs to PLY point clouds.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to trained .pth model")
    parser.add_argument("--frames_dir", type=str, required=True, help="Directory containing .pt frame seed files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for generated .ply files")
    
    args = parser.parse_args()
    process_and_visualize(args)
