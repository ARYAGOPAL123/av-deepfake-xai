"""Write RESULTS.md for the DFDC preliminary subset from the files the runs actually produced.

python scripts/dfdc_results.py      (after scripts/run_dfdc.ps1)

Every number comes from data/manifests/dfdc.csv or results/dfdc_*/*_metrics.json; nothing is typed in by hand.
"""
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/manifests/dfdc.csv"
RUNS = [("Video-only", "dfdc_video_only", "softmax"), ("Audio-only", "dfdc_audio_only", "softmax"),
        ("Fusion (cross-attention)", "dfdc_fusion", "softmax"), ("Fusion + MC Dropout", "dfdc_fusion", "mc_dropout")]


def fmt(v):
    return "n/a" if v is None or v != v else f"{v:.3f}"


def main():
    if not MANIFEST.exists():
        raise SystemExit("data/manifests/dfdc.csv not found - run scripts/run_dfdc.ps1 first")
    df = pd.read_csv(MANIFEST)
    found = [(name, exp, meth, ROOT / f"results/{exp}/{exp}_test_{meth}_metrics.json") for name, exp, meth in RUNS]
    found = [(n, e, m, json.loads(p.read_text())) for n, e, m, p in found if p.exists()]
    if not found:
        raise SystemExit("No DFDC metrics found under results/dfdc_* - nothing to report")

    counts = df.groupby(["split", "label"]).size().unstack(fill_value=0).reindex(["train", "val", "test"]).fillna(0).astype(int)
    cfg = yaml.safe_load(open(ROOT / "configs/dfdc_fusion.yaml"))
    L = [
        "# Results - DFDC small preliminary subset",
        "",
        "> **This is a small preliminary DFDC subset, not the full DFDC benchmark.** "
        f"Only {len(df)} videos ({(df.label == 'real').sum()} REAL, {(df.label == 'fake').sum()} FAKE) were used, "
        f"and the test split has {int(counts.loc['test'].sum())} clips. With so few clips a single prediction moves "
        "accuracy by several points, so these numbers show that the pipeline runs end-to-end on real DFDC data. "
        "They are **not** evidence of detector performance and cannot be compared with published DFDC results.",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} by `scripts/dfdc_results.py` from `data/manifests/dfdc.csv` "
        "and `results/dfdc_*/*_metrics.json`. All values are real outputs of the commands in `scripts/run_dfdc.ps1`.",
        "",
        "## Data",
        "",
        "Labels come from DFDC `metadata.json`. The split is leak-free: clips are grouped by source video "
        "(a FAKE's `original`, a REAL's own file), and each group sits entirely in one split. "
        "(DFDC metadata has no actor IDs, so the same actor can still appear in two different source videos.)",
        "",
        "| split | REAL | FAKE | total |",
        "|---|---:|---:|---:|",
    ]
    for s, r in counts.iterrows():
        L.append(f"| {s} | {r.get('real', 0)} | {r.get('fake', 0)} | {r.sum()} |")
    d, m, t = cfg["data"], cfg["model"], cfg["train"]
    L += ["", "## Setup (CPU, no GPU)", "",
          f"- {d['n_frames']} face crops per clip (MTCNN, {d['image_size']}x{d['image_size']}), {d['max_audio_sec']} s of 16 kHz audio",
          f"- Visual encoder: {m['visual_backbone']} (ImageNet), fully frozen; audio encoder: {m['audio_backbone']}, frozen",
          f"- Only the projection, fusion and classifier layers are trained: {t['epochs']} epochs, batch {t['batch_size']}, "
          f"AdamW lr {t['lr']}, focal loss, class-balanced sampler",
          "- The best epoch is picked by validation F1; all metrics below are on the held-out **test** split at threshold 0.5",
          "", "## Test results", "",
          "| model | n | accuracy | precision | recall | F1 | AUC | EER | ECE |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, _, _, r in found:
        L.append(f"| {name} | {r['n']} | {fmt(r['accuracy'])} | {fmt(r['precision'])} | {fmt(r['recall'])} | "
                 f"{fmt(r['f1'])} | {fmt(r.get('auc'))} | {fmt(r.get('eer'))} | {fmt(r['ece'])} |")
    L += ["", "AUC and EER are n/a when the test split holds only one class.", "",
          "Confusion matrices (rows = true real/fake, columns = predicted real/fake):", ""]
    for name, _, _, r in found:
        cm = r["confusion_matrix"]
        L.append(f"- {name}: `{cm}`")
    mc = next((r for n, e, meth, r in found if meth == "mc_dropout"), None)
    if mc:
        L += ["", "## Uncertainty routing (fusion, MC Dropout)", "",
              f"With tau = {cfg['uncertainty']['tau']} and u_max = {cfg['uncertainty']['u_max']} (defaults, not calibrated: "
              f"the validation split is too small to tune them), {fmt(mc.get('flagged_pct'))}% of test clips go to human review; "
              f"selective accuracy on the auto-accepted clips = {fmt(mc.get('selective_accuracy'))}, coverage = {fmt(mc.get('coverage'))}."]
    L += ["", "## Limitations", "",
          "- Very small sample: results have wide uncertainty and can change completely with a different split or seed.",
          "- Encoders are frozen and training is only a few CPU epochs, so the models are under-trained by design.",
          "- DFDC fakes are mostly visual face swaps, and some carry manipulated audio. metadata.json has only a clip-level label, so "
          "the audio-only model is trained to predict that label even when the audio is genuine.",
          "- These numbers must not be reported as DFDC benchmark results.", ""]
    (ROOT / "RESULTS.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
