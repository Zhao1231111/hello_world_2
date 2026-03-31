
import sys
import os
import torch
import torch.nn as nn
from typing import Dict

# Setup paths
current_file = os.path.abspath(__file__)
mvsplat_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
sys.path.append(mvsplat_root)

# Import local GaussianMLP
sys.path.append(os.path.dirname(current_file))
import yaml
import argparse
from gaussian_mlp import GaussianMLP, GaussianMLPCfg

def export_mlp(ckpt_path: str, output_path: str, yaml_path: str):
    # Parse YAML for slide_window_size
    window_size = 2 # default fallback
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            r3live_cfg = yaml.safe_load(f)
            if 'slide_window_size' in r3live_cfg:
                window_size = r3live_cfg['slide_window_size']
                print(f"Loaded slide_window_size from YAML: {window_size}")
    
    cfg = GaussianMLPCfg()
    model = GaussianMLP(cfg)
    
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading checkpoint from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        # Handle custom dict from train_regressor or direct model dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            print(f"Warning: Architecture mismatch when loading checkpoint. Error: {e}")
            print("Proceeding to export UNTRAINED model for testing purposes...")
    else:
        print(f"Warning: Checkpoint '{ckpt_path}' not found, exporting untrained model...")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Tracing model on device: {device}")
    
    # Create dummy inputs
    B, C = 1, cfg.d_in
    W = window_size
    f_curr = torch.randn(B, C).to(device)
    render_f_curr = torch.randn(B, C).to(device)
    hist_f = torch.randn(B, W, C).to(device)
    hist_render_f = torch.randn(B, W, C).to(device)
    curr_dir = torch.randn(B, 3).to(device)
    hist_dir = torch.randn(B, W, 3).to(device)
    hist_mask = torch.ones(B, W, 1).to(device)
    inv_depth = torch.randn(B, 1).to(device)
    n_cam = torch.randn(B, 3).to(device)
    n_cam = torch.nn.functional.normalize(n_cam, dim=-1)
    mask = torch.ones(B, 1).to(device)
    dis = torch.ones(B, 1).to(device)
    
    # expanded R_wc from (3, 3) to (B, 3, 3) if we calculate it per-batch.
    R_wc = torch.eye(3).unsqueeze(0).expand(B, 3, 3).to(device)
    
    # Trace
    traced_model = torch.jit.trace(model, (f_curr, render_f_curr, hist_f, hist_render_f, curr_dir, hist_dir, hist_mask, inv_depth, n_cam, mask, dis, R_wc))
    
    # Make sure output dir exists
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    traced_model.save(output_path)
    print(f"Exported MLP to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export trained Gaussian MLP to TorchScript (.pt) format")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the trained .pth checkpoint")
    parser.add_argument("--output_path", type=str, default="gaussian_mlp.pt", help="Path to save the traced .pt file")
    parser.add_argument("--yaml_path", type=str, default="/root/catkin_gaussian/src/Gaussian-LIC/config/r3live.yaml", help="Path to config yaml")
    
    args = parser.parse_args()
    
    yaml_path = args.yaml_path
    if not os.path.isabs(yaml_path):
        yaml_path = os.path.join(mvsplat_root, "src", "Gaussian-LIC", "config", "r3live.yaml")
        
    export_mlp(args.ckpt_path, args.output_path, yaml_path)
