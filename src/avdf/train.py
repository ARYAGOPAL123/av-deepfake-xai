"""Train a detector.   python -m avdf.train --config configs/default.yaml"""
import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

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


class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, manifest, split, cache_dir, n_frames, sample_rate, max_audio_sec, embedding_dir):
        self.base = AVDataset(manifest, split, cache_dir, n_frames, sample_rate, max_audio_sec)
        self.embedding_dir = Path(embedding_dir)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        _, _, y, cat, cid = self.base[index]
        item = torch.load(self.embedding_dir / f"{cid}.pt", map_location="cpu", weights_only=True)
        visual = item.get("visual", torch.empty(0))
        audio = item.get("audio", torch.empty(0))
        while visual.ndim > 2 and visual.shape[0] == 1:
            visual = visual.squeeze(0)
        while audio.ndim > 2 and audio.shape[0] == 1:
            audio = audio.squeeze(0)
        return visual.float(), audio.float(), y, cat, cid

    def labels(self):
        return self.base.labels()


@torch.no_grad()
def cache_embeddings(cfg, model, dev):
    """Encode each cached clip once; frozen encoder outputs are stored as float16."""
    d = cfg.data
    out = Path(d.cache_dir) / f"embeddings_d{cfg.model.d_model}"
    out.mkdir(parents=True, exist_ok=True)
    ds = AVDataset(d.manifest, None, d.cache_dir, d.n_frames, d.sample_rate, d.max_audio_sec)
    for faces, wav, _, _, cids in tqdm(DataLoader(ds, batch_size=1, shuffle=False), desc="cache embeddings"):
        target = out / f"{cids[0]}.pt"
        item = torch.load(target, map_location="cpu", weights_only=True) if target.exists() else {}
        need_visual = model.visual is not None and "visual" not in item
        need_audio = model.audio is not None and "audio" not in item
        if not need_visual and not need_audio:
            continue
        visual, audio = model.encode(faces.to(dev), wav.to(dev))
        if need_visual and visual is not None:
            item["visual"] = visual.cpu().half()
        if need_audio and audio is not None:
            item["audio"] = audio.cpu().half()
        torch.save(item, target)
    return out


def loaders(cfg, model, dev):
    d = cfg.data
    embedding_dir = cache_embeddings(cfg, model, dev)
    mk = lambda s: EmbeddingDataset(d.manifest, s, d.cache_dir, d.n_frames, d.sample_rate, d.max_audio_sec, embedding_dir)
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
    for visual, audio, y, *_ in dl:
        visual = visual.to(dev) if visual.numel() else None
        audio = audio.to(dev) if audio.numel() else None
        logits, _ = model.fuse(visual, audio)
        p = evidential_probs(logits)[0] if evidential else logits.float().softmax(-1)
        ys.append(y.numpy()); ps.append(p[:, 1].cpu().numpy())
    return classification_metrics(np.concatenate(ys), np.concatenate(ps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--cache-only", action="store_true", help="build frozen embeddings and exit")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    dev = get_device(cfg)
    model = build_model(cfg).to(dev)
    for encoder in (model.visual, model.audio):
        if encoder is not None:
            for parameter in encoder.parameters():
                parameter.requires_grad = False
    tr, dl_tr, dl_va = loaders(cfg, model, dev)
    if args.cache_only:
        print(f"cached embeddings -> {Path(cfg.data.cache_dir) / f'embeddings_d{cfg.model.d_model}'}")
        return
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
            for visual, audio, y, *_ in tqdm(dl_tr, desc=f"epoch {epoch+1}/{cfg.train.epochs}"):
                visual = visual.to(dev, non_blocking=True) if visual.numel() else None
                audio = audio.to(dev, non_blocking=True) if audio.numel() else None
                y = y.to(dev)
                with torch.autocast("cuda", enabled=use_amp):
                    logits, _ = model.fuse(visual, audio)
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
