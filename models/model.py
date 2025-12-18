import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from utils.logger import setup_logger

logger = setup_logger(
    name="model",
    log_dir="logs/model",
)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.fc = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, dim)
        residual = x
        x = self.norm(x)
        x = self.act(x)
        x = self.fc(x)
        x = self.dropout(x)
        return x + residual


class Classifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256,
                 num_blocks: int = 2, dropout: float = 0.4):
        super().__init__()
        self.fc_in = nn.Linear(in_dim, hidden_dim)

        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout=dropout) for _ in range(num_blocks)]
        )

        self.norm_out = nn.LayerNorm(hidden_dim)
        self.act_out = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (B, in_dim)
        x = self.fc_in(x)          # (B, hidden_dim)
        x = self.blocks(x)         # residual MLP
        x = self.norm_out(x)
        x = self.act_out(x)
        x = self.dropout(x)
        x = self.fc_out(x)         # (B, 1)
        return x


# ===================== CBAM 3D =====================

class ChannelGate2D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg = self.avg_pool(x).view(b, c)
        maxv = self.max_pool(x).view(b, c)
        att = self.mlp(avg) + self.mlp(maxv)
        att = self.sigmoid(att).view(b, c, 1, 1)
        return x * att


class SpatialGate2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        maxv, _ = torch.max(x, dim=1, keepdim=True)
        s = torch.cat([avg, maxv], dim=1)          # (B,2,H,W)
        att = self.sigmoid(self.conv(s))           # (B,1,H,W)
        return x * att


class CBAM2D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_gate = ChannelGate2D(channels, reduction)
        self.spatial_gate = SpatialGate2D()

    def forward(self, x):
        x = self.channel_gate(x)
        x = self.spatial_gate(x)
        return x

class SEBlock2D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, C, H, W)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)     # (B, C)
        y = self.fc(y).view(b, c, 1, 1)     # (B, C, 1, 1)
        return x * y

class ConvNeXtTiny_2D_Meta_CBAM(nn.Module):
    """
    ConvNeXt-Tiny + CBAM2D + metadata + residual MLP classifier

    Input:
      - img:  (B, C, H, W)   (C = 3 hoặc 5 slices)
      - meta: (B, meta_dim)
    Output:
      - logits: (B, 1, 1)
    """

    def __init__(
        self,
        meta_dim: int = 2,
        in_channels: int = 5,
        pretrained: bool = True,
    ):
        super().__init__()

        # -------- Backbone --------
        weights = None
        backbone = convnext_tiny(weights=weights)

        # ---- adapt input channels (3 -> in_channels) ----
        if in_channels != 3:
            old_conv = backbone.features[0][0]  # Conv2d(3, 96, kernel=4, stride=4)
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
            with torch.no_grad():
                w = old_conv.weight
                if in_channels > 3:
                    new_conv.weight[:, :3] = w
                    for c in range(3, in_channels):
                        new_conv.weight[:, c:c+1] = w.mean(dim=1, keepdim=True)
                else:
                    new_conv.weight.copy_(w[:, :in_channels])
            backbone.features[0][0] = new_conv

        self.backbone = backbone.features     # output: (B, 768, H', W')
        self.backbone_out_dim = 768           # ConvNeXt-Tiny

        # -------- CBAM --------
        self.cbam2d = CBAM2D(self.backbone_out_dim, reduction=16)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.feat_norm = nn.LayerNorm(self.backbone_out_dim)

        # -------- Metadata MLP --------
        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )

        fused_dim = self.backbone_out_dim + 64
        self.fused_dropout = nn.Dropout(0.4)

        # -------- Classifier --------
        self.classifier = Classifier(
            in_dim=fused_dim,
            hidden_dim=256,
            num_blocks=2,
            dropout=0.4,
        )

    def forward(self, img, meta):
        """
        img:  (B, C, H, W)
        meta: (B, meta_dim)
        """
        x = self.backbone(img)               # (B, 768, H', W')
        x = self.cbam2d(x)                   # CBAM attention
        x = self.global_pool(x).flatten(1)   # (B, 768)
        x = self.feat_norm(x)

        meta_feat = self.meta_mlp(meta)      # (B, 64)

        fused = torch.cat([x, meta_feat], dim=1)
        fused = self.fused_dropout(fused)

        logits = self.classifier(fused)      # (B, 1)
        return logits.unsqueeze(1)           # (B, 1, 1)


def load_flexible_state_dict(model, ckpt_path, device="cpu", verbose=True):
    logger.info(f"Loading weights from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    model_state = model.state_dict()
    loaded_params = 0
    skipped = []

    for k, v in state.items():
        candidates = []

        # 1) original key
        candidates.append(k)

        # 2) strip 'module.' (MedicalNet typically has module.*)
        if k.startswith("module."):
            k2 = k[len("module."):]
            candidates.append(k2)
            candidates.append("backbone." + k2)

        # 3) add backbone prefix directly
        candidates.append("backbone." + k)

        used = False
        for cand in candidates:
            if cand in model_state and model_state[cand].shape == v.shape:
                model_state[cand] = v
                loaded_params += 1
                used = True
                break

        if not used:
            skipped.append((k, v.shape))

    model.load_state_dict(model_state)

    if verbose:
        logger.info(f"Loaded tensors:  {loaded_params}")
        logger.info(f"Skipped tensors: {len(skipped)}")
        for k, shape in skipped[:10]:
            logger.info(f"  [SKIP] {k}  shape={shape}")
        if len(skipped) > 10:
            logger.info(f"  ... and {len(skipped)-10} more")
    return model