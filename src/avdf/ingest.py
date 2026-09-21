"""Build dataset manifests (CSV) from the raw dataset folders.

Manifest columns: clip_id, path, label (real/fake), category, subject, dataset
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}
FAV_CATS = {
    "RealVideo-RealAudio": ("real", "RV-RA"),
    "RealVideo-FakeAudio": ("fake", "RV-FA"),
    "FakeVideo-RealAudio": ("fake", "FV-RA"),
    "FakeVideo-FakeAudio": ("fake", "FV-FA"),
}


def clip_id(path: Path) -> str:
    return hashlib.md5(str(path).encode()).hexdigest()[:16]


def build_fakeavceleb(root: Path) -> pd.DataFrame:
    """Walks FakeAVCeleb_v1.2/<Category>/<ethnicity>/<gender>/<idXXXXX>/*.mp4"""
    rows = []
    for vid in root.rglob("*"):
        if vid.suffix.lower() not in VIDEO_EXT:
            continue
        cat_dir = next((p for p in vid.parts if p in FAV_CATS), None)
        if cat_dir is None:
            continue
        label, cat = FAV_CATS[cat_dir]
        m = re.search(r"(id\d+)", str(vid))
        rows.append(dict(clip_id=clip_id(vid), path=str(vid), label=label, category=cat,
                         subject=m.group(1) if m else vid.parent.name, dataset="fakeavceleb"))
    return pd.DataFrame(rows)


def build_avdeepfake1m(root: Path, meta_json: str = None) -> pd.DataFrame:
    """AV-Deepfake1M: uses the official metadata JSON(s) (e.g. val_metadata.json).
    A clip is fake if it has any fake period or modified audio/video."""
    rows = []
    metas = [Path(meta_json)] if meta_json else sorted(root.glob("*metadata*.json"))
    for mj in metas:
        for it in json.load(open(mj)):
            f = root / it["file"]
            if not f.exists():
                f = root / mj.stem.split("_")[0] / it["file"]
            fake = bool(it.get("fake_periods")) or bool(it.get("n_fakes", 0)) or it.get("modify_video") or it.get("modify_audio")
            mv, ma = bool(it.get("modify_video")), bool(it.get("modify_audio"))
            cat = "RV-RA" if not fake else ("FV-FA" if mv and ma else "FV-RA" if mv else "RV-FA" if ma else "FAKE")
            rows.append(dict(clip_id=clip_id(f), path=str(f), label="fake" if fake else "real", category=cat,
                             subject=Path(it["file"]).parts[0], dataset="avdeepfake1m"))
    return pd.DataFrame(rows)


def build_dfd(root: Path) -> pd.DataFrame:
    """Google/Jigsaw DFD (FaceForensics++ layout):
    original_sequences/actors/.../*.mp4 (real), manipulated_sequences/DeepFakeDetection/.../*.mp4 (fake)"""
    rows = []
    for vid in root.rglob("*"):
        if vid.suffix.lower() not in VIDEO_EXT:
            continue
        s = str(vid).lower()
        if "original" in s or "actors" in s:
            label, cat = "real", "RV-RA"
        elif "manipulated" in s or "deepfakedetection" in s:
            label, cat = "fake", "FV-RA"
        else:
            continue
        subj = vid.stem.split("_")[0]
        rows.append(dict(clip_id=clip_id(vid), path=str(vid), label=label, category=cat, subject=subj, dataset="dfd"))
    return pd.DataFrame(rows)


def _dfdc_meta(root: Path, meta_json: str = None) -> dict:
    mj = Path(meta_json) if meta_json else root / "metadata.json"
    if not mj.exists():
        raise SystemExit(f"DFDC metadata not found: {mj}. Put metadata.json next to the videos (see data/README.md)")
    return json.load(open(mj))


def build_dfdc(root: Path, meta_json: str = None) -> pd.DataFrame:
    """DFDC (Kaggle layout): <root>/*.mp4 + metadata.json {file: {label: REAL|FAKE, split, original}}.
    Only files present on disk are used. subject = the source video (a fake's "original", a real's own name),
    so a real clip and every fake made from it always land in the same split."""
    rows = []
    for name, it in _dfdc_meta(root, meta_json).items():
        f = root / name
        if not f.exists():
            continue
        fake = it["label"].upper() == "FAKE"
        src = (it.get("original") or name) if fake else name
        rows.append(dict(clip_id=clip_id(f), path=str(f), label="fake" if fake else "real",
                         category="FAKE" if fake else "REAL", subject=Path(src).stem, dataset="dfdc"))
    return pd.DataFrame(rows)


def dfdc_status(root: Path, meta_json: str = None) -> dict:
    """Which metadata.json entries are on disk, and which REAL videos are still missing."""
    meta = _dfdc_meta(root, meta_json)
    on_disk = {p.name for p in root.iterdir() if p.suffix.lower() in VIDEO_EXT}
    have = {n: it["label"].upper() for n, it in meta.items() if n in on_disk}
    real_missing = sorted(n for n, it in meta.items() if it["label"].upper() == "REAL" and n not in on_disk)
    # REAL sources of the fakes you have: downloading these first gives matched real/fake pairs
    originals = {it.get("original") for n, it in meta.items() if n in on_disk and it["label"].upper() == "FAKE"}
    return dict(n_meta=len(meta), n_meta_real=sum(it["label"].upper() == "REAL" for it in meta.values()),
                have_real=sum(v == "REAL" for v in have.values()), have_fake=sum(v == "FAKE" for v in have.values()),
                not_in_meta=sorted(on_disk - set(meta)), real_missing=real_missing,
                real_missing_paired=[n for n in real_missing if n in originals])


def group_stratified_split(df, train=0.70, val=0.15, seed=42):
    """Split by subject (source video) so no source appears in two splits, while keeping the real/fake
    ratio of each split close to the overall one (greedy assignment, largest groups first)."""
    frac = {"train": train, "val": val, "test": 1 - train - val}
    y = (df.label == "fake").astype(int)
    total = y.groupby(y).size().reindex([0, 1], fill_value=0)
    groups = df.assign(y=y).groupby("subject").y.agg(n_real=lambda s: (s == 0).sum(), n_fake="sum")
    groups = groups.sample(frac=1.0, random_state=seed)
    groups = groups.loc[(groups.n_real + groups.n_fake).sort_values(ascending=False, kind="stable").index]
    have = {s: [0, 0] for s in frac}
    assign = {}
    for subj, g in groups.iterrows():
        need = {s: g.n_real * (frac[s] * total[0] - have[s][0]) + g.n_fake * (frac[s] * total[1] - have[s][1]) for s in frac}
        best = max(need, key=need.get)
        assign[subj] = best
        have[best][0] += g.n_real; have[best][1] += g.n_fake
    df = df.copy()
    df["split"] = df.subject.map(assign)
    return df


def subject_disjoint_split(df, train=0.70, val=0.15, seed=42):
    """Assign splits by subject so no identity appears in two splits; stratified by whether a subject has real clips."""
    rng = pd.Series(df.subject.unique()).sample(frac=1.0, random_state=seed).tolist()
    n = len(rng)
    tr, va = set(rng[: int(train * n)]), set(rng[int(train * n): int((train + val) * n)])
    df = df.copy()
    df["split"] = ["train" if s in tr else "val" if s in va else "test" for s in df.subject]
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["fakeavceleb", "avdeepfake1m", "dfd", "dfdc"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--meta", default=None, help="AV-Deepfake1M / DFDC metadata json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--split", action="store_true", help="create subject-disjoint train/val/test split")
    ap.add_argument("--status", action="store_true", help="DFDC: report REAL/FAKE on disk and missing REAL files, then exit")
    a = ap.parse_args()
    root = Path(a.root)
    if a.status:
        st = dfdc_status(root, a.meta)
        print(f"metadata.json: {st['n_meta']} entries ({st['n_meta_real']} REAL, {st['n_meta'] - st['n_meta_real']} FAKE)")
        print(f"on disk:       {st['have_real']} REAL, {st['have_fake']} FAKE")
        if st["not_in_meta"]:
            print(f"on disk but not in metadata.json (ignored): {', '.join(st['not_in_meta'])}")
        print(f"\nREAL videos in metadata.json not downloaded yet: {len(st['real_missing'])}")
        print(f"  of which are the originals of fakes you already have ({len(st['real_missing_paired'])}):")
        for n in st["real_missing_paired"]:
            print("   ", n)
        out = Path(a.out or "data/manifests/dfdc_missing_real.txt")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(st["real_missing"]) + "\n", encoding="utf-8")
        print(f"full list -> {out}")
        raise SystemExit(0)
    if not a.out:
        ap.error("--out is required")
    builders = {"fakeavceleb": build_fakeavceleb, "dfd": build_dfd,
                "avdeepfake1m": lambda r: build_avdeepfake1m(r, a.meta), "dfdc": lambda r: build_dfdc(r, a.meta)}
    df = builders[a.dataset](root)
    if df.empty:
        raise SystemExit(f"No videos found under {root}. Check the folder layout in data/README.md")
    df["split"] = "test"
    if a.split:
        df = group_stratified_split(df) if a.dataset == "dfdc" else subject_disjoint_split(df)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(df.groupby(["split", "category"]).size().unstack(fill_value=0))
    print(f"Saved {len(df)} clips -> {a.out}")
