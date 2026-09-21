"""Collect all *_metrics.json into results/summary.csv (ready to paste into the report tables)."""
import json
from pathlib import Path

import pandas as pd

rows = []
for f in sorted(Path("results").rglob("*_metrics.json")):
    m = json.load(open(f))
    rows.append({"run": f.stem.replace("_metrics", ""), **{k: m.get(k) for k in
                ["accuracy", "precision", "recall", "f1", "auc", "eer", "ece", "selective_accuracy", "flagged_pct", "errors_caught", "n"]}})
df = pd.DataFrame(rows)
df.to_csv("results/summary.csv", index=False)
print(df.round(4).to_string(index=False))
