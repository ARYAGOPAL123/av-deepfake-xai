import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """Bidirectional cross-modal multi-head attention (Eq. 3.1-3.4 of the report)."""

    def __init__(self, d=512, heads=8, p=0.3):
        super().__init__()
        self.v2a = nn.MultiheadAttention(d, heads, dropout=p, batch_first=True)
        self.a2v = nn.MultiheadAttention(d, heads, dropout=p, batch_first=True)
        self.norm_v, self.norm_a = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Dropout(p))

    def forward(self, V, A):
        zv, w_va = self.v2a(V, A, A)          # visual queries attend to audio
        za, w_av = self.a2v(A, V, V)          # audio queries attend to video
        zv = self.norm_v(V + zv).mean(1)
        za = self.norm_a(A + za).mean(1)
        return self.ffn(torch.cat([zv, za], -1)), (w_va, w_av)


class ConcatFusion(nn.Module):
    """Early concatenation baseline (ablation A3): no attention."""

    def __init__(self, d=512, p=0.3):
        super().__init__()
        self.ffn = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Dropout(p))

    def forward(self, V, A):
        return self.ffn(torch.cat([V.mean(1), A.mean(1)], -1)), None


class Head(nn.Module):
    def __init__(self, d_in, p=0.3, n_cls=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 256), nn.GELU(), nn.Dropout(p), nn.Linear(256, n_cls))

    def forward(self, h):
        return self.net(h)
