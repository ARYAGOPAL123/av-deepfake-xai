import timm
import torch.nn as nn


class VisualEncoder(nn.Module):
    """Per-frame image backbone (EfficientNet-B0 / ViT-B/16 via timm) -> 512-d frame tokens."""

    def __init__(self, name="efficientnet_b0", d=512, pretrained=True, freeze_blocks=4):
        super().__init__()
        self.backbone = timm.create_model(name, pretrained=pretrained, num_classes=0)
        # MobileNetV3 & co. have a conv head after pooling: pooled width is head_hidden_size, not num_features
        feat = getattr(self.backbone, "head_hidden_size", None) or self.backbone.num_features
        self.proj = nn.Linear(feat, d)
        self.is_vit = "vit" in name
        self.frozen = freeze_blocks is not None and freeze_blocks < 0   # -1: freeze the whole backbone
        if self.frozen:
            for p in self.backbone.parameters():
                p.requires_grad = False
        elif freeze_blocks and hasattr(self.backbone, "blocks"):
            stem = [m for n, m in self.backbone.named_children() if n in ("conv_stem", "bn1", "patch_embed", "cls_token", "pos_embed")]
            for m in stem + list(self.backbone.blocks[:freeze_blocks]):
                for p in m.parameters():
                    p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:                       # keep BatchNorm statistics of a frozen backbone fixed
            self.backbone.eval()
        return self

    def target_layer(self):
        """Layer used by Grad-CAM."""
        if hasattr(self.backbone, "conv_head"):
            return self.backbone.conv_head
        return self.backbone.blocks[-1].norm1

    def forward(self, faces):                 # faces: (B,T,3,H,W)
        B, T = faces.shape[:2]
        f = self.backbone(faces.flatten(0, 1))  # (B*T, C)
        return self.proj(f).view(B, T, -1)      # (B,T,d)
