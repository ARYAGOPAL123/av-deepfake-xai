import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.register_buffer("alpha", alpha if alpha is not None else torch.ones(2))
        self.gamma = gamma

    def forward(self, logits, y):
        logp = F.log_softmax(logits.float(), -1).gather(1, y[:, None]).squeeze(1)
        p = logp.exp()
        return (-self.alpha[y] * (1 - p) ** self.gamma * logp).mean()


def edl_loss(logits, y, epoch, anneal_epochs=10, K=2):
    """Evidential loss (Sensoy et al., 2018): expected log-likelihood under Dirichlet + annealed KL."""
    alpha = F.softplus(logits.float()) + 1
    S = alpha.sum(-1, keepdim=True)
    yo = F.one_hot(y, K).float()
    nll = (yo * (torch.log(S) - torch.log(alpha))).sum(-1)
    alpha_t = yo + (1 - yo) * alpha                       # remove evidence of the true class
    S_t = alpha_t.sum(-1, keepdim=True)
    kl = (torch.lgamma(S_t).squeeze(-1) - torch.lgamma(torch.tensor(float(K), device=logits.device))
          - torch.lgamma(alpha_t).sum(-1)
          + ((alpha_t - 1) * (torch.digamma(alpha_t) - torch.digamma(S_t))).sum(-1))
    coef = min(1.0, epoch / max(anneal_epochs, 1))
    return (nll + coef * kl).mean()


def class_weights(labels):
    counts = torch.bincount(torch.as_tensor(labels), minlength=2).float()
    w = counts.sum() / (2 * counts.clamp(min=1))
    return w / w.mean()
