import torch.nn as nn

from .audio_encoder import build_audio_encoder
from .fusion import ConcatFusion, CrossAttentionFusion, Head
from .visual_encoder import VisualEncoder


class AVDeepfakeDetector(nn.Module):
    """mode: fusion | concat | video_only | audio_only"""

    def __init__(self, mode="fusion", visual="efficientnet_b0", visual_pretrained=True, freeze_visual_blocks=4,
                 audio="microsoft/wavlm-base-plus", freeze_audio_ssl=True, d=512, heads=8, p=0.3, evidential=False):
        super().__init__()
        self.mode, self.evidential = mode, evidential
        self.visual = VisualEncoder(visual, d, visual_pretrained, freeze_visual_blocks) if mode != "audio_only" else None
        self.audio = build_audio_encoder(audio, d, freeze_audio_ssl) if mode != "video_only" else None
        if mode == "fusion":
            self.fusion, d_in = CrossAttentionFusion(d, heads, p), 2 * d
        elif mode == "concat":
            self.fusion, d_in = ConcatFusion(d, p), 2 * d
        else:
            self.fusion, d_in = None, d
            self.drop = nn.Dropout(p)
        self.head = Head(d_in, p)

    # --- stages kept separate so MC Dropout re-samples only fusion + head ---
    def encode(self, faces, wav):
        V = self.visual(faces) if self.visual is not None else None
        A = self.audio(wav) if self.audio is not None else None
        return V, A

    def fuse(self, V, A):
        if self.mode == "video_only":
            return self.head(self.drop(V.mean(1))), None
        if self.mode == "audio_only":
            return self.head(self.drop(A.mean(1))), None
        h, attn = self.fusion(V, A)
        return self.head(h), attn

    def forward(self, faces, wav):
        V, A = self.encode(faces, wav)
        return self.fuse(V, A)


def build_model(cfg):
    m = cfg.model
    return AVDeepfakeDetector(mode=m.mode, visual=m.visual_backbone, visual_pretrained=m.visual_pretrained,
                              freeze_visual_blocks=m.freeze_visual_blocks, audio=m.audio_backbone,
                              freeze_audio_ssl=m.freeze_audio_ssl, d=m.d_model, heads=m.heads, p=m.dropout,
                              evidential=m.evidential)
