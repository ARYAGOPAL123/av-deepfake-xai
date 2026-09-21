#!/usr/bin/env bash
# Trains and evaluates every configuration used in the report (Tables 5.1-5.5).
set -e
export PYTHONPATH=src
for c in video_only audio_only concat default evidential; do
  python -m avdf.train --config configs/$c.yaml
  python -m avdf.evaluate --config configs/$c.yaml --split test --method softmax
done
# Uncertainty variants of the full model (Table 5.1 rows 5-6, Table 5.3)
python -m avdf.evaluate --config configs/default.yaml --split test --method mc_dropout
python -m avdf.evaluate --config configs/evidential.yaml --split test --method evidential
# Threshold calibration on validation, then re-evaluate
python -m avdf.evaluate --config configs/default.yaml --split val --method mc_dropout --tag val
python -m avdf.calibrate --predictions results/fusion_full/val_predictions.csv --target_error 0.02
# Explanation fidelity (Table 5.4)
python -m avdf.explain.fidelity --config configs/default.yaml --n 100
# Cross-dataset (Table 5.5)
[ -f data/manifests/avdeepfake1m.csv ] && python -m avdf.evaluate --config configs/default.yaml --manifest data/manifests/avdeepfake1m.csv --cache data/cache/avdeepfake1m --split test --tag cross_avdeepfake1m
[ -f data/manifests/dfd.csv ] && python -m avdf.evaluate --config configs/default.yaml --manifest data/manifests/dfd.csv --cache data/cache/dfd --split test --tag cross_dfd
python scripts/collect_results.py
