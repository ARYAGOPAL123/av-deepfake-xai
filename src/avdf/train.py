"""Train a detector.   python -m avdf.train --config configs/default.yaml"""
import argparse
import json
import time
from contextlib import nullcontext

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import AVDataset, balanced_sampler
from .losses import FocalLoss, class_weights, edl_loss
from .metrics import classification_metrics
from .models import build_model
from .uncertainty import evidential_probs
from .utils import ensure_dir, get_device, load_config, seed_everything

try:
    import mlflow
except ImportError:  # optional
    mlflow = None


def loaders(cfg):
    d = cfg.data
    mk = lambda s: AVDataset(d.manifest, s, d.cache_dir, d.n_frames, d.sample_rate, d.max_audio_sec)
    tr, va = mk("train"), mk("val")
    sampler = balanced_sampler(tr) if cfg.train.balanced_sampler else None
    dl_tr = DataLoader(tr, batch_size=cfg.train.batch_size, sampler=sampler, shuffle=sampler is None,
                       num_workers=d.num_workers, pin_memory=True, drop_last=len(tr) > cfg.train.batch_size)
    dl_va = DataLoader(va, batch_size=cfg.train.batch_size, shuffle=False, num_workers=d.num_workers)
    return tr, dl_tr, dl_va


@torch.no_grad()
def validate(model, dl, dev, evidential=False):
    model.eval()
    ys, ps = [], []
    for faces, wav, y, *_ in dl:
        logits, _ = model(faces.to(dev), wav.to(dev))
        p = evidential_probs(logits)[0] if evidential else logits.float().softmax(-1)
        ys.append(y.numpy()); ps.append(p[:, 1].cpu().numpy())
    return classification_metrics(np.concatenate(ys), np.concatenate(ps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    cfg = load_config(ap.parse_args().config)
    seed_everything(cfg.seed)
    dev = get_device(cfg)
    tr, dl_tr, dl_va = loaders(cfg)
    model = build_model(cfg).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in params)/1e6:.2f}M | train clips: {len(tr)}")
    opt = torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.train.epochs)
    crit = FocalLoss(class_weights(tr.labels()).to(dev), cfg.train.focal_gamma)
    use_amp = cfg.train.amp and dev.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    out_dir = ensure_dir(f"{cfg.paths.results_dir}/{cfg.experiment}")
    ensure_dir("checkpoints")
    ckpt = cfg.paths.checkpoint.replace("best.pt", f"{cfg.experiment}_best.pt")
    run = mlflow.start_run(run_name=cfg.experiment) if mlflow else nullcontext()
    best, history = -1.0, []
    with run:
        if mlflow:
            mlflow.log_params({"mode": cfg.model.mode, "visual": cfg.model.visual_backbone, "audio": cfg.model.audio_backbone,
                               "lr": cfg.train.lr, "batch": cfg.train.batch_size, "evidential": cfg.model.evidential})
        for epoch in range(cfg.train.epochs):
            model.train()
            t0, losses = time.time(), []
            for faces, wav, y, *_ in tqdm(dl_tr, desc=f"epoch {epoch+1}/{cfg.train.epochs}"):
                faces, wav, y = faces.to(dev, non_blocking=True), wav.to(dev, non_blocking=True), y.to(dev)
                with torch.autocast("cuda", enabled=use_amp):
                    logits, _ = model(faces, wav)
                loss = edl_loss(logits, y, epoch, cfg.train.kl_anneal_epochs) if cfg.model.evidential else crit(logits, y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
                scaler.step(opt); scaler.update()
                losses.append(loss.item())
            sched.step()
            val = validate(model, dl_va, dev, cfg.model.evidential)
            rec = {"epoch": epoch + 1, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val.items() if k != "confusion_matrix"}}
            history.append(rec)
            print(json.dumps(rec), f"({time.time()-t0:.0f}s)")
            if mlflow:
                mlflow.log_metrics({k: v for k, v in rec.items() if isinstance(v, float)}, step=epoch)
            if val["f1"] > best:
                best = val["f1"]
                torch.save({"state_dict": model.state_dict(), "config": dict(cfg), "epoch": epoch + 1, "val": val}, ckpt)
                print(f"  saved best checkpoint (val F1={best:.4f}) -> {ckpt}")
    json.dump(history, open(out_dir / "history.json", "w"), indent=1)
    _plot_history(history, out_dir)


def _plot_history(h, out_dir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    e = [r["epoch"] for r in h]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(e, [r["train_loss"] for r in h], label="train loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(e, [r["val_f1"] for r in h], label="val F1"); ax[1].plot(e, [r.get("val_auc", np.nan) for r in h], label="val AUC")
    ax[1].set_xlabel("epoch"); ax[1].legend()
    plt.tight_layout(); plt.savefig(out_dir / "training_curves.png", dpi=200); plt.close()


if __name__ == "__main__":
    main()
