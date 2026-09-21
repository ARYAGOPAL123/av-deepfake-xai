# AVDF Five-Minute Demonstration

## Start

Windows PowerShell:

```powershell
.\scripts\run_app.ps1
```

The launcher prints and opens `http://127.0.0.1:8000/`.

## Talk track

1. Point to the header mode badge. In the current repository it reads **DEMO MODE** because no trained checkpoint exists in `checkpoints/`.
2. Open **Analyse** and click **REAL**. Explain that this is a labelled presentation fixture, not a model prediction. Show the verdict card, confidence threshold `tau`, uncertainty threshold `u_max`, decision, technical JSON, and PDF action.
3. Click **FAKE** and describe the same evidence-oriented workflow for a manipulated example.
4. Click **UNCERTAIN**. Explain how a low-confidence or high-uncertainty case is routed to **HUMAN REVIEW** instead of being silently accepted.
5. Upload a video only when demonstrating preprocessing. In DEMO MODE, the backend validates the media and attempts face/audio extraction, then explicitly shows **No trained model loaded — verdict unavailable**. It must never assign an uploaded file a fixture verdict.
6. Open **Results & Metrics**. The empty state is intentional until `results/summary.csv` or `*_metrics.json` files exist. No metric is hard-coded in the interface.
7. Open **Audit Log** and **Review Queue** to show the operational surfaces. They are empty until a trained inference produces routed/audited cases.
8. Open **About the System** and explain the video/audio preprocessing, cross-modal attention, uncertainty routing, Grad-CAM, SHAP, and dataset boundaries.

## Live mode

After a trained checkpoint is placed at the configured checkpoint path, restart the app. The header changes to **LIVE MODE**, fixtures are disabled, and uploaded videos are sent through `avdf.pipeline.Pipeline` for real inference and explanations. Never call a fixture result a live prediction.

## Important limitation

The current workspace has no licensed FakeAVCeleb data and no trained checkpoint. The demo is therefore suitable for demonstrating the product workflow, not for claiming research metrics.
