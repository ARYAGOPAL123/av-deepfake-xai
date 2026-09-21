import torch
import torch.nn as nn
import torch.nn.functional as F


def enable_dropout(model):
    """Switch only Dropout layers (incl. attention dropout) to train mode; BatchNorm stays in eval."""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
        if isinstance(m, nn.MultiheadAttention):
            m.train()   # MHA applies its own attention-weight dropout only in training mode


def entropy(p):
    return -(p * p.clamp_min(1e-8).log()).sum(-1)


def evidential_probs(logits, K=2):
    alpha = F.softplus(logits.float()) + 1
    S = alpha.sum(-1, keepdim=True)
    return alpha / S, (K / S).squeeze(-1)


@torch.no_grad()
def mc_predict(model, V, A, T=30):
    """Encoders run once; only fusion + head are sampled T times."""
    model.eval()
    enable_dropout(model)
    probs = torch.stack([model.fuse(V, A)[0].float().softmax(-1) for _ in range(T)])  # (T,B,2)
    model.eval()
    mean = probs.mean(0)
    return mean, probs.var(0).sum(-1), entropy(mean)


@torch.no_grad()
def predict_with_uncertainty(model, faces, wav, method="mc_dropout", T=30):
    """Returns dict of tensors: prob (B,2), uncertainty (B,), variance (B,), attn."""
    model.eval()
    V, A = model.encode(faces, wav)
    logits, attn = model.fuse(V, A)
    if method == "mc_dropout":
        prob, var, unc = mc_predict(model, V, A, T)
    elif method == "evidential":
        prob, unc = evidential_probs(logits)
        var = torch.zeros_like(unc)
    else:
        prob = logits.float().softmax(-1)
        unc, var = entropy(prob), torch.zeros(prob.shape[0], device=prob.device)
    return {"prob": prob, "uncertainty": unc, "variance": var, "attn": attn}
