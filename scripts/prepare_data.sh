#!/usr/bin/env bash
# Build manifests and preprocess (face crops + 16 kHz audio) for all datasets.
# Expects the raw datasets under data/raw/ (see data/README.md for how to obtain them).
set -e
export PYTHONPATH=src
# 1. FakeAVCeleb (train/val/test, subject-disjoint)
python -m avdf.ingest --dataset fakeavceleb --root data/raw/FakeAVCeleb_v1.2 --out data/manifests/fakeavceleb.csv --split
python -m avdf.preprocess --manifest data/manifests/fakeavceleb.csv --cache data/cache/fakeavceleb
# 2. AV-Deepfake1M (test only; use the validation subset metadata)
if [ -d data/raw/AV-Deepfake1M ]; then
  python -m avdf.ingest --dataset avdeepfake1m --root data/raw/AV-Deepfake1M --meta data/raw/AV-Deepfake1M/val_metadata.json --out data/manifests/avdeepfake1m.csv
  python -m avdf.preprocess --manifest data/manifests/avdeepfake1m.csv --cache data/cache/avdeepfake1m
fi
# 3. DFD (test only)
if [ -d data/raw/DFD ]; then
  python -m avdf.ingest --dataset dfd --root data/raw/DFD --out data/manifests/dfd.csv
  python -m avdf.preprocess --manifest data/manifests/dfd.csv --cache data/cache/dfd
fi
