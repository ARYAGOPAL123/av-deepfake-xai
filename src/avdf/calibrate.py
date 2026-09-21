"""Tune routing thresholds (tau, u_max) on validation predictions.

python -m avdf.evaluate --config configs/default.yaml --split val --tag val
python -m avdf.calibrate --predictions results/fusion_full/val_predictions.csv --target_error 0.02
"""
import argparse
import json

import numpy as np
import pandas as pd


def sweep(df, taus, umaxs):
    y, pred = df.label.values, (df.p_fake.values >= 0.5).astype(int)
    rows = []
    for t in taus:
        for u in umaxs:
            acc = (df.confidence.values >= t) & (df.uncertainty.values <= u)
            cov = acc.mean()
            err = (pred[acc] != y[acc]).mean() if acc.any() else 0.0
            rows.append(dict(tau=t, u_max=u, coverage=cov, selective_error=err, flagged_pct=(1 - cov) * 100))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--target_error", type=float, default=0.02)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    df = pd.read_csv(a.predictions)
    umax_grid = np.quantile(df.uncertainty, np.linspace(0.5, 1.0, 11))
    s = sweep(df, np.round(np.arange(0.5, 1.0, 0.01), 2), umax_grid)
    ok = s[s.selective_error <= a.target_error]
    best = (ok if len(ok) else s).sort_values(["coverage", "selective_error"], ascending=[False, True]).iloc[0]
    res = {k: float(v) for k, v in best.items()}
    print(json.dumps(res, indent=1))
    out = a.out or a.predictions.replace("_predictions.csv", "_thresholds.json")
    json.dump(res, open(out, "w"), indent=1)
    s.to_csv(out.replace(".json", "_sweep.csv"), index=False)
    print(f"Copy tau / u_max into the config's `uncertainty` section. Saved -> {out}")


if __name__ == "__main__":
    main()
