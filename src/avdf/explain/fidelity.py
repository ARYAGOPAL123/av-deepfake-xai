"""Deletion / insertion AUC for Grad-CAM (visual) explanations.

python -m avdf.explain.fidelity --config configs/default.yaml --n 100
"""
import argparse
import json

import numpy as np
import torch

from .. import FAKE
from .gradcam import gradcam_frames


@torch.no_grad()
def _prob(model, faces, wav):
    return float(model(faces, wav)[0].float().softmax(-1)[0, FAKE])


def deletion_insertion(model, faces, wav, cam, steps=10):
    """Progressively remove (deletion) / reveal (insertion) the most salient pixels across all frames."""
    order = np.argsort(-cam.ravel())
    base = torch.zeros_like(faces)                               # zero = mean image after normalisation
    flat_n = cam.size
    dels, ins = [], []
    for k in range(steps + 1):
        n = int(flat_n * k / steps)
        mask = np.zeros(flat_n, dtype=bool); mask[order[:n]] = True
        m = torch.as_tensor(mask.reshape(cam.shape), device=faces.device)[None, :, None]  # (1,T,1,H,W)
        dels.append(_prob(model, torch.where(m, base, faces), wav))
        ins.append(_prob(model, torch.where(m, faces, base), wav))
    x = np.linspace(0, 1, steps + 1)
    trap = getattr(np, "trapezoid", None) or np.trapz
    return float(trap(dels, x)), float(trap(ins, x))


def main():
    from torch.utils.data import DataLoader
    from ..datasets import AVDataset
    from ..evaluate import load_checkpoint
    from ..utils import ensure_dir, get_device, load_config
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    cfg = load_config(a.config); dev = get_device(cfg)
    model, c = load_checkpoint(cfg, None, dev)
    d = c.data
    ds = AVDataset(d.manifest, "test", d.cache_dir, d.n_frames, d.sample_rate, d.max_audio_sec)
    ds.df = ds.df[ds.df.label == "fake"].sample(min(a.n, (ds.df.label == "fake").sum()), random_state=0).reset_index(drop=True)
    D, I = [], []
    for faces, wav, *_ in DataLoader(ds, batch_size=1):
        faces, wav = faces.to(dev), wav.to(dev)
        cam, _ = gradcam_frames(model, faces, wav)
        dl, il = deletion_insertion(model, faces, wav, cam)
        D.append(dl); I.append(il)
    res = {"deletion_auc": float(np.mean(D)), "insertion_auc": float(np.mean(I)), "n": len(D)}
    out = ensure_dir(f"{cfg.paths.results_dir}/{c.experiment}")
    json.dump(res, open(out / "gradcam_fidelity.json", "w"), indent=1)
    print(res)


if __name__ == "__main__":
    main()
