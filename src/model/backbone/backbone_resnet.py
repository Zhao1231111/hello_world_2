import functools
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
import torchvision
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn
from torchvision.models import ResNet

from .backbone import Backbone


@dataclass
class BackboneResnetCfg:
    name: Literal["resnet"]
    model: Literal[
        "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "dino_resnet50"
    ]
    num_layers: int
    use_first_pool: bool
    d_out: int


class BackboneResnet(Backbone[BackboneResnetCfg]):
    model: ResNet

    def __init__(self, cfg: BackboneResnetCfg, d_in: int) -> None:
        super().__init__(cfg)

        assert d_in == 3

        norm_layer = functools.partial(
            nn.InstanceNorm2d,
            affine=False,
            track_running_stats=False,
        )

        if cfg.model == "dino_resnet50":
            self.model = torch.hub.load("facebookresearch/dino:main", "dino_resnet50")
        else:
            self.model = getattr(torchvision.models, cfg.model)(norm_layer=norm_layer)

        # Set up projections
        # pytorch字典，会向 PyTorch “注册”它包含的所有层。
        # 这里的projection的value是用于将每一层resnet输出“投影”到d_out维度的卷积层
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

    def forward(
        self,
        image: Float[Tensor, "batch view 3 height width"],
    ) -> Float[Tensor, "batch view d_out height width"]:
        
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

        # Upscale the features.
        # 上采样到原始图片大小
        features = [
            F.interpolate(f, (h, w), mode="bilinear", align_corners=True)
            for f in features
        ]
        features = torch.stack(features).sum(dim=0) # 特征相加

        # Restore input dimensions if needed
        if is_batched_view:
            return rearrange(features, "(b v) c h w -> b v c h w", b=b, v=v)
        else:
            return features

    @property
    def d_out(self) -> int:
        return self.cfg.d_out
