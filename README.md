# Explainable and Uncertainty-Aware Audio-Visual Deepfake Detection using Cross-Modal Attention Fusion

M.Tech Dissertation — **Arya Gopal (M250801)**, M.Tech CSE (AI & ML), Rajagiri School of Engineering & Technology (Autonomous), Kochi
Guide: **Mrs. Megha V**, Assistant Professor, Department of AI & DS

The framework detects audio-visual deepfakes and, for every clip, returns

1. a **REAL / FAKE verdict** from a bidirectional cross-modal attention fusion model (EfficientNet-B0 / ViT faces + WavLM / XLS-R speech),
2. **visual evidence** — Grad-CAM heatmaps over the face, computed through the fused model,
3. **audio evidence** — SHAP attributions over Mel-frequency bands and time segments,
4. a **trust signal** — MC-Dropout or evidential uncertainty; low-confidence clips are routed to a human review queue and every decision is written to an audit log.

## Folder structure

```
av-deepfake-xai/
├── configs/            experiment configs (full model, ablations, evidential, smoke test)
├── data/
│   ├── raw/            ← put the downloaded datasets here (see data/README.md)
│   ├── cache/          face crops + 16 kHz audio produced by preprocessing
│   └── manifests/      CSV manifests with subject-disjoint splits
├── src/avdf/           Python package
│   ├── ingest.py         build manifests, subject-disjoint split
│   ├── preprocess.py     validation, MTCNN face crops, audio extraction + VAD
│   ├── datasets.py       AVDataset, balanced sampler
│   ├── models/           visual encoder, audio encoder (SLS), cross-attention fusion, detector
│   ├── losses.py         focal loss, evidential loss
│   ├── train.py          training (AdamW, cosine LR, AMP, best-F1 checkpoint, MLflow)
│   ├── evaluate.py       metrics, per-category accuracy, ROC, confusion matrix, reliability diagram
│   ├── calibrate.py      tune routing thresholds tau / u_max on validation
│   ├── explain/          Grad-CAM, SHAP (audio), deletion/insertion fidelity
│   ├── uncertainty/      MC Dropout, evidential probabilities, routing
│   ├── db.py             SQLite audit log + review queue
│   └── pipeline.py       one-video end-to-end analysis
├── api/main.py         FastAPI service
├── dashboard/app.py    Streamlit analyst dashboard
├── db/schema.sql       database schema (ER diagram in the report)
├── scripts/            data preparation, experiments, smoke test, results collection
├── tests/              unit / smoke tests
├── docs/               final report (DOCX + PDF) and final presentation (PPTX)
├── results/            metrics, plots and predictions written here
└── checkpoints/        trained model weights written here
```

## 1. Setup

```bash
# Python 3.10+, CUDA GPU recommended; ffmpeg must be installed (sudo apt install ffmpeg)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Google Colab: upload the folder to Drive, then `!pip install -r requirements.txt && pip install -e .` and run the same commands with `!`.

## 2. Quick smoke test (no dataset needed)

```bash
bash scripts/smoke_test.sh
```
Creates a tiny synthetic dataset, trains for 2 epochs on CPU, evaluates and runs the unit tests. The numbers are meaningless; it only proves the pipeline runs.

## 3. Data

Download the datasets as described in **data/README.md** and place them under `data/raw/`, then:

```bash
bash scripts/prepare_data.sh
```

## 4. Train and evaluate everything in the report

```bash
bash scripts/run_experiments.sh      # all ablations, uncertainty, calibration, fidelity, cross-dataset
cat results/summary.csv              # numbers for Tables 5.1–5.5
```
Individual steps:
```bash
export PYTHONPATH=src
python -m avdf.train     --config configs/default.yaml                 # full cross-attention model
python -m avdf.evaluate  --config configs/default.yaml --split test     # MC-Dropout by default
python -m avdf.evaluate  --config configs/default.yaml --split val --tag val
python -m avdf.calibrate --predictions results/fusion_full/val_predictions.csv --target_error 0.02
python -m avdf.explain.fidelity --config configs/default.yaml --n 100
```

| Report item | Config / command |
|---|---|
| A1 video-only | `configs/video_only.yaml` |
| A2 audio-only | `configs/audio_only.yaml` |
| A3 early concatenation | `configs/concat.yaml` |
| A4 full cross-attention fusion | `configs/default.yaml` |
| Evidential head | `configs/evidential.yaml` |
| A7 cross-dataset | `evaluate --manifest data/manifests/avdeepfake1m.csv --cache data/cache/avdeepfake1m` |

Plots (confusion matrix, ROC, reliability diagram, training curves) are saved in `results/<experiment>/` — insert them into Chapter 5 of the report.

## 5. Analyse a single video / live demo

### Run the forensic application on Windows

```powershell
.\scripts\run_app.ps1
```

The launcher creates `.venv` if needed, starts the FastAPI backend, opens
`http://127.0.0.1:8000/`, and serves the single-page forensic application.
The header always shows **LIVE MODE** when a configured checkpoint exists or
**DEMO MODE** when it does not. DEMO MODE fixtures are explicitly labelled and
uploaded videos never receive a fixture verdict.

```bash
export PYTHONPATH=src
python -m avdf.pipeline --config configs/default.yaml --video sample.mp4     # JSON + Grad-CAM/SHAP images
streamlit run dashboard/app.py -- --config configs/default.yaml              # analyst dashboard
uvicorn api.main:app --port 8000                                             # REST API (POST /analyse)
```

### Presentation rehearsal (works without model dependencies)

When the restricted datasets and a trained checkpoint are not available, use the
dependency-free rehearsal UI to demonstrate the product workflow without presenting
fixtures as scientific results:

```bash
python scripts/presentation_demo.py
# open http://127.0.0.1:8765/dashboard/presentation_demo.html
```

The page demonstrates verdicts, confidence, uncertainty routing, visual/audio evidence,
and an audit event. It is intentionally labelled **DEMO MODE**. The generated
`results/presentation_demo/status.json` records that no trained model, real dataset, or
real metrics are available; do not copy these values into report tables.

## 6. Docker

```bash
docker build -t avdf .
docker run --gpus all -p 8501:8501 -v $PWD/checkpoints:/app/checkpoints avdf
```

## Notes
* Train on FakeAVCeleb only; AV-Deepfake1M and DFD are used only for testing.
* Splits are subject-disjoint (no identity appears in two splits).
* Datasets are **not** included — they are licence-restricted and must be requested from their authors.
