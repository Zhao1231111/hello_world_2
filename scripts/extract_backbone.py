
import functools
from dataclasses import dataclass
from typing import Literal, Dict, TypeVar, Generic, Optional
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor

# ==========================================
# 1. 基础类定义 (与项目代码同步)
# ==========================================

T = TypeVar("T")

class Backbone(nn.Module, ABC, Generic[T]):
    cfg: T
    def __init__(self, cfg: T) -> None:
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(self, image: Tensor) -> Tensor:
        pass

    @property
    @abstractmethod
    def d_out(self) -> int:
        pass

# ==========================================
# 2. ResNet 定义
# ==========================================

@dataclass
class BackboneResnetCfg:
    name: Literal["resnet"]
    model: Literal["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "dino_resnet50"]
    num_layers: int
    use_first_pool: bool
    d_out: int

class BackboneResnet(Backbone[BackboneResnetCfg]):
    model: torchvision.models.ResNet

    def __init__(self, cfg: BackboneResnetCfg, d_in: int = 3) -> None:
        super().__init__(cfg)

        norm_layer = functools.partial(
            nn.InstanceNorm2d,
            affine=False,
            track_running_stats=False,
        )

        if cfg.model == "dino_resnet50":
            # Modified to match likely hub usage if local cache fails or differing versions
            try:
                self.model = torch.hub.load("facebookresearch/dino:main", "dino_resnet50")
            except:
                print("Warning: Failed to load dino_resnet50 from hub, falling back to standard resnet50")
                self.model = torchvision.models.resnet50(norm_layer=norm_layer)
        else:
            self.model = getattr(torchvision.models, cfg.model)(norm_layer=norm_layer)

        # Set up projections
        self.projections = nn.ModuleDict({})
        for index in range(1, cfg.num_layers):
            key = f"layer{index}"
            block = getattr(self.model, key)
            conv_index = 1
            try:
                while True:
                    d_layer_out = getattr(block[-1], f"conv{conv_index}").out_channels
                    conv_index += 1
            except AttributeError:
                pass
            self.projections[key] = nn.Conv2d(d_layer_out, cfg.d_out, 1)

        # Add a projection for the first layer.
        self.projections["layer0"] = nn.Conv2d(
            self.model.conv1.out_channels, cfg.d_out, 1
        )

    def forward(self, image: Float[Tensor, "batch view 3 height width"]) -> Float[Tensor, "batch view d_out height width"]:
        
        # Handle input shape flexibility: (B, V, C, H, W) vs (B, C, H, W)
        is_batched_view = image.ndim == 5
        if is_batched_view:
            b, v, _, h, w = image.shape
            x = rearrange(image, "b v c h w -> (b v) c h w")
        else:
            # Assume (B, C, H, W)
            x = image
            h, w = x.shape[-2:]

        # Run the images through the resnet.
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        features = [self.projections["layer0"](x)]

        # Propagate the input through the resnet's layers.
        for index in range(1, self.cfg.num_layers):
            key = f"layer{index}"
            if index == 0 and self.cfg.use_first_pool:
                x = self.model.maxpool(x)
            x = getattr(self.model, key)(x)
            features.append(self.projections[key](x))

        # Upscale the features sequentially and accumulate to save Peak Memory (important for 8GB GPUs)
        res = F.interpolate(features[0], (h, w), mode="bilinear", align_corners=True)
        for i in range(1, len(features)):
            res.add_(F.interpolate(features[i], (h, w), mode="bilinear", align_corners=True))
        features = res

        # Restore input dimensions if needed
        if is_batched_view:
            return rearrange(features, "(b v) c h w -> b v c h w", b=b, v=v)
        else:
            return features

    @property
    def d_out(self) -> int:
        return self.cfg.d_out

# ==========================================
# 3. DINO Backbone 定义
# ==========================================

@dataclass
class BackboneDinoCfg:
    name: Literal["dino"]
    model: Literal["dino_vits16", "dino_vits8", "dino_vitb16", "dino_vitb8"]
    d_out: int
    disable_dim_reduction: bool = False

class BackboneDino(Backbone[BackboneDinoCfg]):
    def __init__(self, cfg: BackboneDinoCfg, d_in: int) -> None:
        super().__init__(cfg)
        
        # Load DINO ViT
        self.dino = torch.hub.load("facebookresearch/dino:main", cfg.model)
        
        # Monkey Patch for Tracing
        import types
        self.dino.interpolate_pos_encoding = types.MethodType(interpolate_pos_encoding_trace_friendly, self.dino)
        
        # Load DINO ResNet50 (Hybrid)
        resnet_cfg = BackboneResnetCfg(
            name="resnet", 
            model="dino_resnet50", 
            num_layers=4, 
            use_first_pool=False, # Note: PixelSplat DINO mode defaults
            d_out=cfg.d_out
        )
        self.resnet_backbone = BackboneResnet(resnet_cfg, d_in=d_in)
        
        # Projectors for ViT tokens
        # Typically ViT-S=384, ViT-B=768
        vit_dim = 384 if "vits" in cfg.model else 768
        
        self.global_token_mlp = nn.Sequential(
            nn.Linear(vit_dim, vit_dim),
            nn.ReLU(),
            nn.Linear(vit_dim, cfg.d_out),
        )
        self.local_token_mlp = nn.Sequential(
            nn.Linear(vit_dim, vit_dim),
            nn.ReLU(),
            nn.Linear(vit_dim, cfg.d_out),
        )

    @property
    def patch_size(self) -> int:
        # Extract last digits: dino_vits8 -> 8
        import re
        match = re.search(r'\d+$', self.cfg.model)
        return int(match.group()) if match else 16

    def forward(self, image: Float[Tensor, "batch view 3 height width"]) -> Float[Tensor, "batch view d_out height width"]:
        # 1. ResNet Features
        resnet_features = self.resnet_backbone(image)

        # 2. ViT Features
        is_batched_view = image.ndim == 5
        if is_batched_view:
            b, v, _, h, w = image.shape
            tokens_in = rearrange(image, "b v c h w -> (b v) c h w")
        else:
            tokens_in = image
            b, c, h, w = image.shape
            v = 1 # Dummy view for logic consistency or ignore

        # Ensure input divisibility by patch size
        pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            tokens_in_padded = F.pad(tokens_in, (0, pad_w, 0, pad_h))
        else:
            tokens_in_padded = tokens_in

        # Pass through DINO ViT
        # get_intermediate_layers usually returns tuple of (last_layer_tokens)
        # Output shape: (B, N_patches+1, Dim)
        # Note: Tracing specific behavior for hub models can be tricky
        # but intermediate layers access is standard in DINO hub models.
        
        vit_out = self.dino.get_intermediate_layers(tokens_in_padded, n=1)[0]
        
        # vit_out: [Batch, N_tokens, Dim]
        # Token 0 is CLS (Global), 1: are patches
        
        global_token = self.global_token_mlp(vit_out[:, 0])
        local_tokens = self.local_token_mlp(vit_out[:, 1:])

        # Reshape Global Token
        # (BV, C) -> (BV, C, 1, 1) -> Broadcast to (BV, C, H, W)
        if is_batched_view:
             global_token = repeat(global_token, "(b v) c -> b v c h w", b=b, v=v, h=h, w=w)
        else:
             global_token = repeat(global_token, "b c -> b c h w", b=b, h=h, w=w)

        # Reshape Local Tokens
        # (BV, H_p*W_p, C) -> (BV, C, H_p, W_p) -> Upsample/Repeat to (BV, C, H, W)
        
        # Calculate patch grid dims
        h_p = (h + pad_h) // self.patch_size
        w_p = (w + pad_w) // self.patch_size
        
        local_tokens = rearrange(local_tokens, "b (h w) c -> b c h w", h=h_p, w=w_p)
        
        # Upsample by repeating
        local_tokens = repeat(
            local_tokens,
            "b c h w -> b c (h hps) (w wps)",
            hps=self.patch_size,
            wps=self.patch_size,
        )
        
        # Crop back to original size if padded
        if pad_h > 0 or pad_w > 0:
            local_tokens = local_tokens[..., :h, :w]
            
        # Accumulate sequentially to save Peak Memory
        resnet_features.add_(local_tokens)
        resnet_features.add_(global_token)
        
        # ==========================================================
        # 降维操作 (Dimensionality Reduction): 512 -> 128
        # User requested to sum/average channels to reduce memory footprint.
        # We group every (C // 128) channels and average them.
        # ==========================================================
        if not self.cfg.disable_dim_reduction:
            target_dim = 128
            b_res, c_res, h_res, w_res = resnet_features.shape
            if c_res > target_dim and c_res % target_dim == 0:
                group_size = c_res // target_dim
                # Reshape to (B, 128, GroupSize, H, W) and take the mean along the GroupSize dimension
                resnet_features = resnet_features.view(b_res, target_dim, group_size, h_res, w_res).mean(dim=2)

        return resnet_features

    @property
    def d_out(self) -> int:
        return self.cfg.d_out

def interpolate_pos_encoding_trace_friendly(self, x, w, h):
    # This is a modified version of DINO's interpolate_pos_encoding
    # that is friendly to torch.jit.trace by avoiding Tensor-to-float conversions
    # inside the scale_factor argument construction.
    
    npatch = x.shape[1] - 1
    N = self.pos_embed.shape[1] - 1
    
    if npatch == N and w == h:
        return self.pos_embed
        
    class_pos_embed = self.pos_embed[:, 0]
    patch_pos_embed = self.pos_embed[:, 1:]
    
    dim = x.shape[-1]
    
    # Compute dimensions
    # We use integer arithmetic where possible or ensuring scalars
    
    # Handle patch_size being int or tuple
    patch_size = self.patch_embed.patch_size
    if isinstance(patch_size, int):
        ps_h = patch_size
        ps_w = patch_size
    else:
        ps_h = patch_size[1]
        ps_w = patch_size[0]
        
    w0 = w // ps_w
    h0 = h // ps_h
    
    # We add a small epsilon to avoid floating point errors
    # or just use interpolate with explicit size instead of scale_factor
    # Using size is much safer for Tracing!
    
    N_sqrt = int(torch.sqrt(torch.tensor(N))) # constant during trace usually
    
    # reshape: (1, N, dim) -> (1, sqrt(N), sqrt(N), dim) -> (1, dim, sqrt(N), sqrt(N))
    patch_pos_embed = patch_pos_embed.reshape(1, N_sqrt, N_sqrt, dim).permute(0, 3, 1, 2)
    
    # Interpolate
    # Instead of scale_factor, we calculate the target size directly
    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed,
        size=(h0, w0), # Explicit size is TorchScript friendly
        mode="bicubic",
        align_corners=False,
    )
    
    # (1, dim, h0, w0) -> (1, dim, h0*w0) -> (1, h0*w0, dim)
    patch_pos_embed = patch_pos_embed.flatten(2).transpose(1, 2)
    
    return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)


# ==========================================
# 4. Main Logic
# ==========================================

def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Extract Backbone Weights for C++")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to PixelSplat pytorch_lightning checkpoint (.ckpt)")
    parser.add_argument("--output", type=str, default="backbone.pt", help="Output TorchScript file path")
    
    # Model Config
    parser.add_argument("--backbone_type", type=str, default="resnet18", help="resnet18 | dino_vits8")
    parser.add_argument("--d_out", type=int, default=128, help="Output feature dimension")
    parser.add_argument("--disable_dim_reduction", action="store_true", help="Disable the 512 -> 128 dimension reduction in DINO backbone")
    
    args = parser.parse_args()

    # Determine Model Type
    model = None
    if "resnet" in args.backbone_type and "dino_" not in args.backbone_type:
        print(f"Creating ResNet Backbone: {args.backbone_type}")
        cfg = BackboneResnetCfg(
            name="resnet",
            model=args.backbone_type,
            num_layers=4,
            use_first_pool=True,
            d_out=args.d_out
        )
        model = BackboneResnet(cfg)
    elif "dino" in args.backbone_type:
        print(f"Creating DINO Backbone: {args.backbone_type}")
        # Assuming dino_vits8 
        cfg = BackboneDinoCfg(
            name="dino",
            model=args.backbone_type, # e.g. dino_vits8
            d_out=args.d_out,
            disable_dim_reduction=args.disable_dim_reduction
        )
        model = BackboneDino(cfg, d_in=3)
    else:
        print(f"Unknown backbone type: {args.backbone_type}")
        return

    # Load Weights
    print(f"Loading checkpoint from {args.ckpt}...")
    if not os.path.exists(args.ckpt):
        print(f"Error: Checkpoint file not found.")
        return

    try:
        checkpoint = torch.load(args.ckpt, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # Filter keys
    new_state_dict = {}
    matched_keys = 0
    
    # Key Mapping Logistics for PixelSplat Checkpoints
    # PixelSplat usually nests the backbone under "encoder.backbone."
    # If the user is using DINO, it expects:
    #   - self.dino (ViT)
    #   - self.resnet_backbone (ResNet)
    #   - self.global_token_mlp
    #   - self.local_token_mlp
    
    # In checkpoint:
    #   encoder.backbone.dino.* -> self.dino.*
    #   encoder.backbone.resnet_backbone.* -> self.resnet_backbone.*
    #   encoder.backbone.global_token_mlp.* -> self.global_token_mlp.*
    
    prefix = "encoder.backbone."
    
    for k, v in state_dict.items():
        if k.startswith(prefix):
            target_key = k[len(prefix):] # Remove encoder.backbone.
            
            # Additional cleanup if needed (e.g. if script uses different names)
            # But based on code copy, names should match.
            
            if target_key in model.state_dict():
                new_state_dict[target_key] = v
                matched_keys += 1
            else:
                # Debug mismatch
                pass
                
    if matched_keys == 0:
        print("Error: No matching keys found! Trying looser matching...")
        # Fallback: try direct matching
        for k, v in state_dict.items():
             if k in model.state_dict():
                 new_state_dict[k] = v
                 matched_keys += 1

    print(f"Weights loaded. Matched: {matched_keys}")
    
    # Load it
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    
    if len(missing) > 0:
        print(f"Warning: Missing keys: {len(missing)}")
        # print(missing[:5])

    # Trace
    print("Tracing and exporting...")
    model.eval()
    
    # Dummy input
    # Need to be careful with DINO patch size constraints for dummy input
    # Assuming standard 256x256
    dummy_input = torch.randn(1, 3, 256, 256)
    
    try:
        traced = torch.jit.trace(model, dummy_input)
        traced.save(args.output)
        print(f"Successfully saved to {args.output}")
    except Exception as e:
        print(f"Tracing failed: {e}")

if __name__ == "__main__":
    main()
