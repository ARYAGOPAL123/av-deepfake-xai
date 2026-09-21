# Phase 0 Status

Date: 2026-09-21

## Environment

- Python: 3.13.5
- PyTorch: 2.12.0+cpu
- CUDA: unavailable (`torch.cuda.is_available()` is `False`); no NVIDIA GPU detected
- Free disk at check: 17,974,841,344 bytes (about 16.7 GiB)
- `ffmpeg`: not found on PATH
- Dataset: unavailable in this workspace; `data/raw/` is empty and `data/manifests/` contains only `.gitkeep`
- User-provided dataset location, hardware, and training time: not supplied; the prompt still contains placeholders
- IEEE paper and LaTeX source: not present under `docs/`
- Isolated environment: `.venv` created and `requirements.txt` plus editable package installation completed successfully.

## Code/report mismatches

The repository has core scaffolding for the specified system, but it is not yet evidence-complete:

- `src/avdf/train.py` supports the principal model configurations, focal/evidential loss, validation checkpoints, and training curves.
- `src/avdf/evaluate.py` produces the requested core classification metrics, confusion matrix, ROC, reliability diagram, selective routing metrics, and per-category accuracy.
- `scripts/run_experiments.sh` does not define separate report ablations A5 (without explainability) and A6 (without uncertainty).
- `src/avdf/calibrate.py` reports calibration values for manual copying; it does not update a config automatically.
- `src/avdf/explain/fidelity.py` covers visual Grad-CAM deletion/insertion; the report also requires SHAP/group-deletion fidelity and attribution stability.
- The current tests cover model modes, uncertainty helpers, Grad-CAM, and basic metrics, but not the report's TC-01 through TC-12 integration tests.
- `api/main.py` and `dashboard/app.py` cover inference, review, and audit basics but do not cover every administrative/model-version and display requirement described in the report.
- Report and presentation result placeholders cannot be filled yet because there are no real-data runs.
- No IEEE paper source exists, so the requested LaTeX update/recompile cannot be performed until that source is supplied.

## Evidence state

No scientific result is available yet. The existing `results/presentation_demo/` output is deterministic UI fixture data and is explicitly marked non-scientific in its status JSON. It must not be copied into the report, slides, or paper.

## Required decisions before Phase 2/3

1. Supply the approved FakeAVCeleb path, or explicitly confirm that the dataset is unavailable.
2. Supply the actual hardware and training time budget. Current detection indicates CPU-only execution.
3. If using Colab, upload this repository and the licensed dataset to Google Drive; the existing `notebooks/colab_quickstart.ipynb` should be adapted after the hardware decision.
4. Confirm whether a licensed AV-Deepfake1M and/or DFD copy is available for test-only evaluation.

## Phase 0 conclusion

Phase 0 environment inspection is complete. The isolated software environment is ready. A Phase 1 smoke attempt was started: synthetic data generation succeeded, but CPU training did not produce a checkpoint within the run window, so Phase 1 is not marked passed. Training and real evaluation remain blocked by missing data, CPU-only hardware, and missing `ffmpeg`. No files under the raw data directory were modified.
