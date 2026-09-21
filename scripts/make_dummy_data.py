"""Create a tiny SYNTHETIC dataset (random faces + tones) to smoke-test the whole pipeline without real data.
Numbers produced on this data are meaningless — it only checks that the code runs end-to-end.

python scripts/make_dummy_data.py --n 24
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=24)
ap.add_argument("--frames", type=int, default=8)
ap.add_argument("--cache", default="data/cache/dummy")
ap.add_argument("--manifest", default="data/manifests/dummy.csv")
a = ap.parse_args()
rng = np.random.default_rng(0)
cache = Path(a.cache); cache.mkdir(parents=True, exist_ok=True)
cats = ["RV-RA", "FV-RA", "RV-FA", "FV-FA"]
rows = []
for i in range(a.n):
    cat = cats[i % 4]
    cid = f"dummy{i:04d}"
    faces = torch.from_numpy(rng.integers(0, 255, (a.frames, 3, 224, 224), dtype=np.uint8))
    t = np.arange(16000 * 4) / 16000
    f0 = 180 if cat.endswith("RA") else 320
    wav = (0.3 * np.sin(2 * np.pi * f0 * t) + 0.05 * rng.standard_normal(len(t))).astype("float32")
    torch.save(faces, cache / f"{cid}_faces.pt"); sf.write(cache / f"{cid}.wav", wav, 16000)
    rows.append(dict(clip_id=cid, path=f"synthetic/{cid}.mp4", label="real" if cat == "RV-RA" else "fake",
                     category=cat, subject=f"id{i // 2:03d}", dataset="dummy",
                     split="train" if i < a.n * 0.6 else "val" if i < a.n * 0.8 else "test"))
Path(a.manifest).parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(a.manifest, index=False)
print(f"wrote {a.n} synthetic clips -> {a.manifest}")
