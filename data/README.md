# Datasets

The datasets are **not redistributed** with this project. Each is released for research use under its own
licence/agreement and must be requested from the authors. Place them as shown below.

| Dataset | Use | How to obtain |
|---|---|---|
| FakeAVCeleb v1.2 | train / val / test | Fill in the request form linked at https://github.com/DASH-Lab/FakeAVCeleb — a download link is emailed after approval |
| AV-Deepfake1M | cross-dataset test | Accept the licence at https://github.com/ControlNet/AV-Deepfake1M (Hugging Face gated download) |
| DFD (Google/Jigsaw DeepFakeDetection) | cross-dataset test (visual) | Part of FaceForensics++ — request access via https://github.com/ondyari/FaceForensics and use its download script with `-d DeepFakeDetection` and `-d DeepFakeDetection_original` |

The full datasets are large (tens to hundreds of GB). For AV-Deepfake1M, using only the validation subset is enough for cross-dataset testing.

## Expected layout

```
data/raw/
├── FakeAVCeleb_v1.2/
│   ├── RealVideo-RealAudio/<ethnicity>/<gender>/idXXXXX/*.mp4
│   ├── RealVideo-FakeAudio/...
│   ├── FakeVideo-RealAudio/...
│   └── FakeVideo-FakeAudio/...
├── AV-Deepfake1M/
│   ├── val/ ...                       (video files)
│   └── val_metadata.json
└── DFD/
    ├── original_sequences/actors/c23/videos/*.mp4
    └── manipulated_sequences/DeepFakeDetection/c23/videos/*.mp4
```

Then run `bash scripts/prepare_data.sh`. It creates `data/manifests/*.csv` (clip_id, path, label, category,
subject, dataset, split) and caches 32 aligned 224×224 face crops (`*_faces.pt`) and 16 kHz mono audio (`*.wav`)
per clip in `data/cache/`. Clips that fail validation (corrupt, no audio, no video) are listed in `failed.csv`.

Note: DFD clips carry original (real) audio, so they test whether the model relies on visual evidence when the
audio is genuine.

## DFDC small preliminary subset (CPU)

A handful of DFDC videos (e.g. from the Kaggle `train_sample_videos` release) can be used for a small preliminary experiment:

```
data/raw/DFDC_sample/
├── metadata.json      {"abc.mp4": {"label": "REAL"|"FAKE", "split": ..., "original": "xyz.mp4"|null}, ...}
└── *.mp4
```

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_dfdc.ps1 -StatusOnly   # REAL/FAKE on disk + REAL files still missing
powershell -ExecutionPolicy Bypass -File scripts\run_dfdc.ps1              # ingest, preprocess, train/evaluate 3 models, RESULTS.md
powershell -ExecutionPolicy Bypass -File scripts\run_app.ps1 -Config configs/dfdc_fusion.yaml   # dashboard in LIVE MODE
```

Labels come from `metadata.json`; videos listed there but not on disk are skipped. The split is grouped by source video
(a FAKE's `original`, a REAL's own name), so a real clip and its fakes never land in different splits. The missing REAL
files are written to `data/manifests/dfdc_missing_real.txt`. CPU settings (8 frames, batch 4, 5 epochs, frozen
EfficientNet-B0 + WavLM) are in `configs/dfdc_*.yaml`.
