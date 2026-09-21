# Run Log

## Phase 0

| UTC time | Command / check | Outcome |
|---|---|---|
| 2026-09-21 | `python --version` | Python 3.13.5 |
| 2026-09-21 | `python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda, torch.cuda.device_count())"` | PyTorch 2.12.0+cpu; CUDA unavailable; 0 GPUs |
| 2026-09-21 | `Get-PSDrive -Name C` | 17,974,841,344 bytes free at check time |
| 2026-09-21 | `Get-ChildItem data\raw -Recurse -File` | No dataset files found |
| 2026-09-21 | `Get-ChildItem data\manifests -Recurse -File` | Only `data/manifests/.gitkeep` found |
| 2026-09-21 | `Get-Command nvidia-smi`, `Get-Command ffmpeg` | No executable path returned |
| 2026-09-21 | Document and source inspection | Report and presentation exist; no IEEE paper or LaTeX source; see `docs/PHASE0_STATUS.md` |
| 2026-09-21 | `python scripts/presentation_demo.py --port 8765` | Existing dependency-free presentation demo started successfully |
| 2026-09-21 | HTTP request to `/dashboard/presentation_demo.html` | HTTP 200; truthful demo status artifact served |
| 2026-09-21 | `python scripts/make_dummy_data.py --n 8 --frames 4` | Failed before generation: `ModuleNotFoundError: soundfile` |

## Rules

- Every preprocessing, training, evaluation, and document-generation command will be appended here before execution.
- No real-data metric is valid unless its command completes and the source file under `results/` is recorded.
- The presentation demo is explicitly non-scientific and must not supply report metrics.

## Planned environment setup

| 2026-09-21 | `python -m venv .venv; .\.venv\Scripts\python.exe -m pip install -r requirements.txt; .\.venv\Scripts\python.exe -m pip install -e .` | Estimated 10–30 minutes and 3–6 GB temporary/final disk usage on this CPU-only Python 3.13 machine; execution begins after this entry is recorded. |
| 2026-09-21 | Isolated environment verification | Completed. `.venv` contains PyTorch 2.14.0, FastAPI, Streamlit, soundfile, OpenCV, librosa, timm, transformers, SHAP, scikit-learn, and editable `avdf`. |
| 2026-09-21 | Planned Phase 1: `bash scripts/smoke_test.sh` using `.venv` | Expected 5–20 minutes on CPU and less than 1 GB of generated synthetic data/checkpoints; no raw data is touched. |
| 2026-09-21 | `bash scripts/smoke_test.sh` | Failed to launch: `bash` is not installed on this Windows host. Equivalent PowerShell steps are being run below. |
| 2026-09-21 | `\.venv\Scripts\python.exe scripts/make_dummy_data.py --n 24 --frames 8; ...` | In progress as the Windows-compatible equivalent of `scripts/smoke_test.sh`. |
| 2026-09-21 | Windows-compatible smoke steps | Synthetic manifest generated. Initial training reached the EfficientNet forward pass but exceeded the CPU run window; no checkpoint was written. |
| 2026-09-21 | `configs/smoke_test.yaml` CPU adjustment | Reduced frames/image/audio/model/batch size and selected audio-only mode for integration plumbing; real experiment configs were not changed. |
| 2026-09-21 | Audio-only smoke rerun | Started but produced no checkpoint or results artifact within the available run window; Phase 1 remains failed/blocked and requires a clean terminal rerun. |
| 2026-09-21 | `python -m py_compile api/main.py src/avdf/db.py` | Passed after removing the stale legacy API block. |
| 2026-09-21 | `python -m uvicorn api.main:app --host 127.0.0.1 --port 8000` | Started unified FastAPI app; `/`, `/api/status`, `/api/fixtures`, `/api/metrics`, and PDF endpoint returned successfully. |
| 2026-09-21 | Fixture POST to `/api/analyse?fixture_name=real` and GET `/api/jobs/{id}` | Passed; returned DEMO MODE result with explicit non-prediction badge. |
| 2026-09-21 | Browser smoke test on `http://127.0.0.1:8000/` | Passed; all five tabs loaded, REAL fixture completed, and no browser console errors were observed. |
| 2026-09-21 | `python -m pytest -q tests/test_api.py` | Passed: 7 tests. FastAPI/Starlette deprecation warnings remain. |

## DFDC support (2026-09-21)

| Local time | Command / check | Outcome |
|---|---|---|
| 2026-09-21 | Look for `data/raw/DFDC_sample/` and `metadata.json` | Not present. The 9 DFDC `.mp4` files are still in `dataset/`; no `metadata.json` on this machine. No DFDC labels, so no training or evaluation was run. |
| 2026-09-21 | `python -m pytest -q tests/` | Passed: 15 tests (incl. 3 new DFDC ingest/split/status tests on synthetic metadata). |
| 2026-09-21 | `preprocess_clip("dataset/aassnaulhq.mp4", ...)` with 8 and 16 frames | Passed: faces detected 8/8 and 16/16, 10.0 s audio, 3.3 s / 4.0 s per clip on CPU, ~3 MB cache per clip. |
| 2026-09-21 | Frozen EfficientNet-B0 video-only forward/backward + Grad-CAM (random init, no download) | Passed: 0.394M trainable params, backbone stays in eval mode, 1.4 s per batch of 4, Grad-CAM (8, 224, 224). |
| 2026-09-21 | Download of EfficientNet-B0 / WavLM weights from Hugging Face in the assistant's session | Failed: TLS to `us.aws.cdn.hf.co` ends in an untrusted root in that session. `truststore` added so Python uses the Windows certificate store. |
