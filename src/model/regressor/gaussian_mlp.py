import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

def get_embedder(v: torch.Tensor, bands: torch.Tensor) -> torch.Tensor:
    """
    Apply positional encoding to the input tensor v using precomputed bands.
    Formula: gamma(v) = [v, sin(2^0 pi v), cos(2^0 pi v), ..., sin(2^{L-1} pi v), cos(2^{L-1} pi v)]
    """
    # v: (..., D), bands: (L,) -> (..., D, L)
    v_expanded = v.unsqueeze(-1) * bands
    
    # sin, cos -> (..., D, L)
    sins = torch.sin(v_expanded)
    coss = torch.cos(v_expanded)
    
    # stack and reshape -> (..., D * 2L)
    encoded = torch.stack([sins, coss], dim=-1).view(*v.shape[:-1], -1)
    
    # concat [v, encoded] -> (..., D * (1 + 2L))
    return torch.cat([v, encoded], dim=-1)

class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dx = self.linear(x)
        dx = self.relu(dx)
        return x + dx

class CrossAttentionContext(nn.Module):
    def __init__(self, d_in: int, d_k: int = 64):
        super().__init__()
        self.d_k = d_k
        self.q_proj = nn.Linear(3 + 2 * d_in, d_k)
        self.k_proj = nn.Linear(3 + 2 * d_in, d_k)
        self.v_proj = nn.Linear(3 + 2 * d_in, 2 * d_in)

    def forward(self, 
                curr_dir: torch.Tensor, f_curr: torch.Tensor, render_f_curr: torch.Tensor,
                hist_dir: torch.Tensor, hist_f: torch.Tensor, hist_render_f: torch.Tensor,
                hist_mask: torch.Tensor) -> torch.Tensor:
        
        curr_input = torch.cat([curr_dir, f_curr, render_f_curr], dim=-1)
        q = self.q_proj(curr_input).unsqueeze(1)
        
        hist_input = torch.cat([hist_dir, hist_f, hist_render_f], dim=-1)
        k = self.k_proj(hist_input)
        v = self.v_proj(hist_input)
        
        attn = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.d_k)
        
        if hist_mask is not None:
            attn = attn.masked_fill(hist_mask.transpose(1, 2) == 0, -1e9)
            
        attn_weights = F.softmax(attn, dim=-1)
        out = torch.bmm(attn_weights, v).squeeze(1)
        return out

class PredictionHead(nn.Module):
    """
    通用的预测头，Linear -> Act -> Linear
    """
    # TODO: 需要加入残差块吗？
    def __init__(self, d_in: int, d_out: int, d_hidden = None):
        super().__init__()
        if d_hidden is None:
            d_hidden = d_in // 2

        self.head = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, d_out)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

@dataclass
class GaussianMLPCfg:
    d_in: int = 128
    d_hidden: int = 512
    n_layers: int = 5
    sh_degree: int = 3
    scale_dim: int = 2  # 2D Gaussian

class GaussianMLP(nn.Module):
    def __init__(self, cfg: GaussianMLPCfg):
        super().__init__()
        self.cfg = cfg
        # Input features:
        # f_curr (C) + ctx (C) 
        # + render_f_curr(C) + render_ctx(C)
        # + embed(inv_depth) (9) + embed(n_cam_in) (15) + mask (1) + embed(dis) (7)
        self.input_dim = 4 * cfg.d_in + 9 + 15 + 1 + 7
        
        # MLP Backbone with Residual Blocks
        layers = []
        layers.append(nn.Linear(self.input_dim, cfg.d_hidden))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(cfg.n_layers - 1):
            layers.append(ResidualBlock(cfg.d_hidden))
            
        self.encoder = nn.Sequential(*layers)
        
        self.cross_attention = CrossAttentionContext(cfg.d_in)

        # Precompute Positional Encoding Bands
        self.register_buffer('inv_depth_bands', (2.0 ** torch.arange(4)) * math.pi)
        self.register_buffer('n_cam_bands', (2.0 ** torch.arange(2)) * math.pi)
        self.register_buffer('dis_bands', (2.0 ** torch.arange(3)) * math.pi)
        
        # Heads
        self.scale_head = PredictionHead(cfg.d_hidden, cfg.scale_dim, 256)
        
        self.delta_z_head = PredictionHead(cfg.d_hidden, 3)
        self.aux_a_head = PredictionHead(cfg.d_hidden, 3)
        self.lambda_head = PredictionHead(cfg.d_hidden, 1)

        self.opacity_head = PredictionHead(cfg.d_hidden, 1)
        
        self.delta_pixel_head = nn.Linear(cfg.d_hidden, 2)
        self.delta_inv_depth_head = nn.Linear(cfg.d_hidden, 1)
        
        self.num_sh_base_coeffs = 1
        self.num_sh_rest_coeffs = (cfg.sh_degree + 1) ** 2 - 1
        self.color_base_head = PredictionHead(cfg.d_hidden, 3)
        self.color_rest_head = PredictionHead(cfg.d_hidden, self.num_sh_rest_coeffs * 3, 256)
        
        # 【初始化技巧】
        # 让 aux_a 初始倾向于非零，防止初始阶段 cross product 为 0
        # 让 lambda 初始很小，保证初始法向遵循几何输入
        nn.init.normal_(self.aux_a_head.head[-1].weight, std=0.01)
        nn.init.constant_(self.aux_a_head.head[-1].bias, 0.1) # Bias it slightly
        nn.init.constant_(self.lambda_head.head[-1].bias, -2.0) # Softplus(-2) is small
        
        # 初始微调增量全部置 0，保证起手阶段严格与 2.5D 先验对齐
        nn.init.zeros_(self.delta_pixel_head.weight)
        nn.init.zeros_(self.delta_pixel_head.bias)
        nn.init.zeros_(self.delta_inv_depth_head.weight)
        nn.init.zeros_(self.delta_inv_depth_head.bias)
        
        # 让基础颜色修正项初始化为0，完美贴合原有点云 base_sh
        nn.init.zeros_(self.color_base_head.head[-1].weight)
        nn.init.zeros_(self.color_base_head.head[-1].bias)

    def _matrix_to_quaternion(self, matrix: torch.Tensor) -> torch.Tensor:
        """
        更紧凑、GPU 友好的实现，无 Python 控制流，完全向量化。
        """
        m = matrix
        # 提取对角线
        m00, m11, m22 = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
        
        # 计算 trace
        trace = m00 + m11 + m22
        
        def safe_sqrt(x):
            return torch.sqrt(torch.clamp(x, min=1e-6)) # 这里的 epsilon 很重要

        # 构造 4 种可能的候选解 (Batch, 4, 4)
        # 这里的逻辑和你的一样，只是写法上利用 stack 避免 if/else
        
        # Case 0: Trace > 0
        s0 = safe_sqrt(trace + 1.0) * 2.0
        q0 = torch.stack([
            0.25 * s0,
            (m[..., 2, 1] - m[..., 1, 2]) / s0,
            (m[..., 0, 2] - m[..., 2, 0]) / s0,
            (m[..., 1, 0] - m[..., 0, 1]) / s0
        ], dim=-1)

        # Case 1: m00 max
        s1 = safe_sqrt(1.0 + m00 - m11 - m22) * 2.0
        q1 = torch.stack([
            (m[..., 2, 1] - m[..., 1, 2]) / s1,
            0.25 * s1,
            (m[..., 0, 1] + m[..., 1, 0]) / s1,
            (m[..., 0, 2] + m[..., 2, 0]) / s1
        ], dim=-1)

        # Case 2: m11 max
        s2 = safe_sqrt(1.0 + m11 - m00 - m22) * 2.0
        q2 = torch.stack([
            (m[..., 0, 2] - m[..., 2, 0]) / s2,
            (m[..., 0, 1] + m[..., 1, 0]) / s2,
            0.25 * s2,
            (m[..., 1, 2] + m[..., 2, 1]) / s2
        ], dim=-1)

        # Case 3: m22 max
        s3 = safe_sqrt(1.0 + m22 - m00 - m11) * 2.0
        q3 = torch.stack([
            (m[..., 1, 0] - m[..., 0, 1]) / s3,
            (m[..., 0, 2] + m[..., 2, 0]) / s3,
            (m[..., 1, 2] + m[..., 2, 1]) / s3,
            0.25 * s3
        ], dim=-1)

        # 选择最佳分支
        # 比较 trace, m00, m11, m22 谁产生最大的 sqrt 值
        # 注意：这里直接用 s0^2 等效于比较 trace 等，避免重复计算
        # scores = [trace, m00, m11, m22] 逻辑并不完全对齐，
        # 标准做法是比较四个 candidate 的分母 s
        stack_s = torch.stack([s0, s1, s2, s3], dim=-1) # (B, 4)
        best_idx = torch.argmax(stack_s, dim=-1) # (B,)
        
        # Gather best q
        stack_q = torch.stack([q0, q1, q2, q3], dim=1) # (B, 4, 4)
        
        # Advanced indexing to select the best Q for each batch element
        # (B, 4)
        batch_indices = torch.arange(best_idx.shape[0], device=matrix.device)
        q_best = stack_q[batch_indices, best_idx]
        
        return F.normalize(q_best, dim=-1)

    def forward(
        self,
        f_curr: torch.Tensor,       
        render_f_curr: torch.Tensor,
        hist_f: torch.Tensor,
        hist_render_f: torch.Tensor,
        curr_dir: torch.Tensor,
        hist_dir: torch.Tensor,
        hist_mask: torch.Tensor,
        inv_depth: torch.Tensor,    
        n_cam_in: torch.Tensor,     
        mask: torch.Tensor,         
        dis: torch.Tensor,
        R_wc: torch.Tensor,         
    ):
        # Apply Cross-Attention
        ctx_agg = self.cross_attention(curr_dir, f_curr, render_f_curr,
                                       hist_dir, hist_f, hist_render_f, hist_mask)
        
        # 1. Positional Encoding
        inv_depth_norm = torch.clamp(inv_depth, 0.0, 1.0)
        inv_depth_emb = get_embedder(inv_depth_norm, self.inv_depth_bands)
        
        n_cam_emb = get_embedder(n_cam_in, self.n_cam_bands)
        
        dis_log = torch.log(dis + 1e-6)
        dis_emb = get_embedder(dis_log, self.dis_bands)
        
        # Concatenate inputs
        x = torch.cat([
            f_curr, render_f_curr, ctx_agg, 
            inv_depth_emb, n_cam_emb, mask, dis_emb
        ], dim=-1)
        
        feat = self.encoder(x)
        
        # 1. Scale，取指数是为了保证 scale 是正数，它代表归一化到像素平面上的正数尺度
        # 用 (dis / 2.0) 作为尺度初始化范围缩放因子
        scale = (dis / 2.0) * torch.exp(self.scale_head(feat))
        
        # 新增微调位移（直接在 2.5D 空间输出残差）
        delta_pixel = self.delta_pixel_head(feat)
        delta_inv_depth = self.delta_inv_depth_head(feat)
        
        # 2. Rotation Construction
        delta_z = self.delta_z_head(feat)
        aux_a = self.aux_a_head(feat)
        lambda_val = F.softplus(self.lambda_head(feat))
        
        # 【修改 3】增加 eps 防止 NaN
        eps = 1e-6
        
        # Adjust Normal
        z_unnorm = n_cam_in + lambda_val * delta_z
        z_new = F.normalize(z_unnorm + eps, dim=-1)
        
        # Construct Basis
        # 【修改 4】防止 cross 结果为 0
        y_raw = torch.cross(z_new, aux_a, dim=-1)
        y_cam = F.normalize(y_raw + eps, dim=-1) # Add eps!
        
        # x 轴自然垂直，且模长为 1 (因为 y 和 z 归一化且垂直)
        x_cam = torch.cross(y_cam, z_new, dim=-1)
        
        # Stack R_cam
        R_cam = torch.stack([x_cam, y_cam, z_new], dim=-1)
                
        # Apply World Rotation
        # R_wc: (B, 3, 3), R_cam: (B, 3, 3) -> (B, 3, 3)
        R_world = torch.bmm(R_wc, R_cam)
        
        # Convert to Quaternion
        rotation = self._matrix_to_quaternion(R_world)
        
        # 3. Opacity，输出真实的 opacity 值（而不是logits，反sigmoid空间）
        opacity = torch.sigmoid(self.opacity_head(feat))
        
        # 4. Color (SH Coefficients)
        # 拆分为 DC (base) 和 高阶 (rest)
        color_dc_flat = self.color_base_head(feat)
        color_dc = color_dc_flat.view(-1, self.num_sh_base_coeffs, 3)
        
        color_rest_flat = self.color_rest_head(feat)
        color_rest = color_rest_flat.view(-1, self.num_sh_rest_coeffs, 3)
        
        return scale, rotation, opacity, color_dc, color_rest, delta_pixel, delta_inv_depth