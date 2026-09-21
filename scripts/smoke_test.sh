#!/usr/bin/env bash
# End-to-end check on synthetic data (CPU, no downloads needed except timm model definitions).
set -e
export PYTHONPATH=src
python scripts/make_dummy_data.py --n 24 --frames 8
python -m avdf.train --config configs/smoke_test.yaml
python -m avdf.evaluate --config configs/smoke_test.yaml --split test
pytest -q tests
echo "Smoke test passed."
