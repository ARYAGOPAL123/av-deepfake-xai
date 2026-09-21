"""Evaluate a checkpoint on a split (or a cross-dataset manifest).

python -m avdf.evaluate --config configs/default.yaml --split test
python -m avdf.evaluate --config configs/default.yaml --manifest data/manifests/avdeepfake1m.csv \
       --cache data/cache/avdeepfake1m --split test --tag cross_avdf1m
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import AVDataset
from .metrics import classification_metrics, reliability, selective_metrics
from .models import build_model
from .uncertainty import predict_with_uncertainty, route
from .utils import Cfg, ensure_dir, get_device, load_config


def load_checkpoint(cfg, path=None, dev="cpu"):
    path = path or cfg.paths.checkpoint.replace("best.pt", f"{cfg.experiment}_best.pt")
    ck = torch.load(path, map_location=dev)
    ccfg = Cfg(ck["config"])
    model = build_model(ccfg)
    model.load_state_dict(ck["state_dict"])
    return model.to(dev).eval(), ccfg


def run_inference(model, dl, dev, method, T):
    rows = []
    for faces, wav, y, cat, cid in tqdm(dl, desc="evaluate"):
        out = predict_with_uncertainty(model, faces.to(dev), wav.to(dev), method, T)
        p, u = out["prob"].cpu(), out["uncertainty"].cpu()
        for i in range(len(y)):
            rows.append(dict(clip_id=cid[i], category=cat[i], label=int(y[i]), p_fake=float(p[i, 1]),
                             confidence=float(p[i].max()), uncertainty=float(u[i]), variance=float(out["variance"][i])))
    return pd.DataFrame(rows)


def plots(df, out, tag):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, roc_curve
    y, p = df.label.values, df.p_fake.values
    cm = confusion_matrix(y, (p >= 0.5).astype(int), labels=[0, 1])
    plt.figure(figsize=(4, 3.4)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["real", "fake"], yticklabels=["real", "fake"])
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout(); plt.savefig(out / f"{tag}_confusion_matrix.png", dpi=200); plt.close()
    if len(np.unique(y)) == 2:
        fpr, tpr, _ = roc_curve(y, p)
        plt.figure(figsize=(4, 4)); plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], "--", c="gray")
        plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.tight_layout(); plt.savefig(out / f"{tag}_roc.png", dpi=200); plt.close()
    xs, ys, _ = reliability(df.confidence.values, ((p >= 0.5).astype(int) == y).astype(float))
    plt.figure(figsize=(4, 4)); plt.plot([0, 1], [0, 1], "--", c="gray", label="perfect"); plt.plot(xs, ys, "o-", label="model")
    plt.xlabel("Confidence"); plt.ylabel("Accuracy"); plt.legend(); plt.tight_layout(); plt.savefig(out / f"{tag}_reliability.png", dpi=200); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--method", default=None, help="mc_dropout | evidential | softmax")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    dev = get_device(cfg)
    model, ccfg = load_checkpoint(cfg, a.checkpoint, dev)
    d = ccfg.data
    ds = AVDataset(a.manifest or d.manifest, a.split, a.cache or d.cache_dir, d.n_frames, d.sample_rate, d.max_audio_sec)
    dl = DataLoader(ds, batch_size=cfg.train.batch_size, num_workers=d.num_workers)
    method = a.method or cfg.uncertainty.method
    df = run_inference(model, dl, dev, method, cfg.uncertainty.mc_passes)
    pred, conf, review = route(torch.tensor(np.stack([1 - df.p_fake, df.p_fake], 1)), torch.tensor(df.uncertainty.values),
                               cfg.uncertainty.tau, cfg.uncertainty.u_max)
    df["pred"], df["review"] = pred.numpy(), review.numpy()
    m = classification_metrics(df.label, df.p_fake, df.confidence)
    m.update(selective_metrics(df.label, df.pred, df.review))
    m["per_category_accuracy"] = df.assign(ok=df.pred == df.label).groupby("category").ok.mean().to_dict()
    m["method"], m["n"] = method, len(df)
    tag = a.tag or f"{ccfg.experiment}_{a.split}_{method}"
    out = ensure_dir(f"{cfg.paths.results_dir}/{ccfg.experiment}")
    df.to_csv(out / f"{tag}_predictions.csv", index=False)
    json.dump(m, open(out / f"{tag}_metrics.json", "w"), indent=1)
    plots(df, out, tag)
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
