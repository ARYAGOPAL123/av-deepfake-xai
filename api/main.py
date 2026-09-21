"""Single backend for the AVDF forensic analysis application.

    uvicorn api.main:app --port 8000        (or scripts/run_app.ps1 / scripts/run_app.sh)
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.chdir(ROOT)  # configs use repo-relative paths (db, results, checkpoints)

from avdf import __version__  # noqa: E402
from avdf.db import AuditDB  # noqa: E402
from avdf.media import ffmpeg_available  # noqa: E402
from avdf.utils import get_device, load_config  # noqa: E402

CONFIG_PATH = os.environ.get("AVDF_CONFIG", "configs/default.yaml")
CONFIG_FILE = ROOT / CONFIG_PATH
RESULTS = ROOT / "results"
UPLOADS = RESULTS / "uploads"
PREVIEWS = RESULTS / "previews"
for _d in (UPLOADS, PREVIEWS):
    _d.mkdir(parents=True, exist_ok=True)
FRONTEND = ROOT / "dashboard" / "index.html"
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_UPLOAD = 500 * 1024 * 1024
STAGES = ["Queued", "Validating", "Extracting faces", "Extracting audio", "Inference", "Uncertainty", "Explanations", "Done"]
PROGRESS = {s: round(100 * i / (len(STAGES) - 1)) for i, s in enumerate(STAGES)}
SKIPPED_NO_MODEL = ["Inference", "Uncertainty", "Explanations"]  # not run in DEMO MODE

FIXTURES = {
    "real": {"name": "real", "file": "fixture-real.mp4", "verdict": "real", "confidence": .94, "uncertainty": .08, "p_fake": .06,
             "decision": "AUTO-ACCEPT", "title": "Authentic interview clip",
             "reason": "Fixture demonstrates a confident, low-uncertainty case that is auto-accepted."},
    "fake": {"name": "fake", "file": "fixture-fake.mp4", "verdict": "fake", "confidence": .91, "uncertainty": .12, "p_fake": .91,
             "decision": "AUTO-ACCEPT", "title": "Face-swap with cloned voice",
             "reason": "Fixture demonstrates a confident manipulated case that is auto-accepted."},
    "uncertain": {"name": "uncertain", "file": "fixture-uncertain.mp4", "verdict": "fake", "confidence": .62, "uncertainty": .57,
                  "p_fake": .62, "decision": "HUMAN REVIEW", "title": "Low-light, compressed clip",
                  "reason": "Fixture demonstrates routing when confidence is below tau and uncertainty is above u_max."},
}
JOBS: dict[str, dict[str, Any]] = {}
JOB_FILES: dict[str, dict[str, str]] = {}  # job_id -> {kind: filesystem path}, kept out of the public payload
JOB_LOCK = threading.Lock()
PIPELINE_LOCK = threading.Lock()
MAX_JOBS = 200


# ---------------------------------------------------------------- helpers
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def cfg():
    return load_config(CONFIG_FILE)


def checkpoint_path() -> Path:
    c = cfg()
    candidate = Path(str(c.paths.checkpoint).replace("best.pt", f"{c.experiment}_best.pt"))
    return candidate if candidate.is_absolute() else ROOT / candidate


def mode() -> str:
    return "LIVE MODE" if checkpoint_path().exists() else "DEMO MODE"


def status() -> dict[str, Any]:
    c = cfg()
    cp = checkpoint_path()
    try:
        expected = str(cp.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        expected = str(cp)
    return {"mode": mode(), "version": __version__, "checkpoint": cp.name if cp.exists() else None,
            "checkpoint_expected": expected,
            "model_config": {"experiment": c.experiment, "mode": c.model.mode, "visual_backbone": c.model.visual_backbone,
                             "audio_backbone": c.model.audio_backbone, "evidential": bool(c.model.evidential),
                             "uncertainty_method": c.uncertainty.method, "mc_passes": int(c.uncertainty.mc_passes),
                             "n_frames": int(c.data.n_frames), "sample_rate": int(c.data.sample_rate)},
            "tau": float(c.uncertainty.tau), "u_max": float(c.uncertainty.u_max),
            "device": str(get_device(c)), "config": CONFIG_PATH, "ffmpeg": ffmpeg_available(),
            "max_upload_mb": MAX_UPLOAD // (1024 * 1024)}


def db() -> AuditDB:
    return AuditDB(ROOT / cfg().paths.db)


def display_name(path: str | None) -> str:
    """Strip directory and the random upload prefix from a stored path."""
    return re.sub(r"^[0-9a-f]{32}_", "", Path(path or "").name)


def analyses_dir() -> Path:
    return ROOT / cfg().paths.results_dir / "analyses"


def fixture_result(name: str) -> dict[str, Any]:
    if name not in FIXTURES:
        raise HTTPException(404, "Unknown fixture. Use real, fake, or uncertain.")
    result = dict(FIXTURES[name])
    result.update({"mode": "DEMO MODE", "demo_fixture": True, "demo_badge": "Demo fixture — not a model prediction",
                   "result_id": f"fixture-{name}", "sha256": hashlib.sha256(name.encode()).hexdigest(),
                   "display_name": result["file"], "created_at": utc_now(), "method": "labelled presentation fixture",
                   "review": result["decision"] == "HUMAN REVIEW", "skipped_stages": SKIPPED_NO_MODEL})
    return result


def set_job(job_id: str, **values: Any) -> None:
    with JOB_LOCK:
        job = JOBS.setdefault(job_id, {"job_id": job_id, "created_at": utc_now()})
        if "stage" in values and values["stage"] in PROGRESS:
            values.setdefault("progress", PROGRESS[values["stage"]])
            done = job.setdefault("stages_done", [])
            if job.get("stage") and job["stage"] not in done and job["stage"] != values["stage"]:
                done.append(job["stage"])
        job.update(values)
        job["updated_at"] = utc_now()
        while len(JOBS) > MAX_JOBS:
            old = next(iter(JOBS)); JOBS.pop(old); JOB_FILES.pop(old, None)


def evidence_urls(result_id: Any, res: dict[str, Any]) -> dict[str, Any]:
    """Replace filesystem image paths with API URLs the browser can load."""
    out = dict(res)
    for kind in ("gradcam", "shap", "faces_preview"):
        out[f"{kind}_url"] = f"/api/cases/{result_id}/evidence/{kind}" if res.get(kind) else None
        out.pop(kind, None)
    return out


def save_audio_preview(wav, sr: int, path: Path) -> Path:
    """Waveform + log-Mel spectrogram of the extracted audio (what the audio branch sees)."""
    import librosa
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(2, 1, figsize=(10, 3.6), gridspec_kw={"height_ratios": [1, 2]}, sharex=True)
    t = np.arange(len(wav)) / sr
    ax[0].plot(t, wav, lw=0.4, color="#1f5f99"); ax[0].set_ylabel("amp"); ax[0].set_yticks([])
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=400, hop_length=160, n_mels=80), ref=np.max)
    ax[1].imshow(mel, aspect="auto", origin="lower", cmap="magma", extent=[0, len(wav) / sr, 0, 80])
    ax[1].set_ylabel("Mel bin"); ax[1].set_xlabel("time (s)")
    plt.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    return path


# ---------------------------------------------------------------- background work
def preprocess_only(job_id: str, path: Path) -> dict[str, Any]:
    """DEMO MODE: validate and preprocess, but never produce a verdict without a trained model."""
    from avdf.media import probe
    from avdf.pipeline import save_face_grid
    from avdf.preprocess import FaceExtractor, extract_audio, trim_silence
    c = cfg()
    info = probe(path)
    result: dict[str, Any] = {"mode": "DEMO MODE", "verdict": None, "file": path.name, "display_name": display_name(str(path)),
                              "message": "No trained model loaded — verdict unavailable. The clip was validated and "
                                         "preprocessed exactly as the live pipeline would.", "media": info,
                              "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "warnings": []}
    if not info["has_video"]:
        raise ValueError("No decodable video stream found in this file.")
    if not info["has_audio"]:
        result["warnings"].append("No audio stream found — the audio branch would receive silence.")
    files = JOB_FILES.setdefault(job_id, {})
    set_job(job_id, stage="Extracting faces")
    extractor = FaceExtractor(c.data.image_size, device=str(get_device(c)))
    faces = extractor.crop(path, c.data.n_frames, c.data.fps)
    result["frames_extracted"] = int(faces.shape[0])
    result["faces_detected"] = extractor.last_detected
    if extractor.last_detected == 0:
        result["warnings"].append("No face detected — centre crops were used instead.")
    files["faces_preview"] = str(save_face_grid(faces, PREVIEWS / f"{job_id}_faces.png"))
    set_job(job_id, stage="Extracting audio")
    sr = int(c.data.sample_rate)
    if not info["has_audio"]:
        result["audio_error"] = "The file has no audio stream (ffprobe found video only)."
    else:
        try:  # report the real failure instead of losing the whole job
            raw = extract_audio(path, sr)
            wav = trim_silence(raw, sr)
            result.update(audio_samples_raw=int(len(raw)), audio_samples=int(len(wav)), audio_sample_rate=sr,
                          audio_sec=round(len(wav) / sr, 2))
            files["audio_preview"] = str(save_audio_preview(wav, sr, PREVIEWS / f"{job_id}_audio.png"))
        except Exception as exc:  # noqa
            result["audio_error"] = f"{type(exc).__name__}: {exc}"
    if "audio_error" in result:
        result["warnings"].append(f"Audio extraction failed: {result['audio_error']}")
    result["skipped_stages"] = SKIPPED_NO_MODEL
    for kind in ("faces_preview", "audio_preview"):
        result[f"{kind}_url"] = f"/api/jobs/{job_id}/evidence/{kind}" if kind in files else None
    return result


def process_upload(job_id: str, path: Path, explain: bool) -> None:
    started = time.perf_counter()
    try:
        set_job(job_id, stage="Validating")
        if not ffmpeg_available():
            raise RuntimeError("ffmpeg is not available. Run `pip install imageio-ffmpeg` or install ffmpeg.")
        pipe = getattr(app.state, "pipeline", None)
        if pipe is None:
            result = preprocess_only(job_id, path)
        else:
            with PIPELINE_LOCK:  # the model and its explain flag are shared across requests
                pipe.explain = explain
                result = pipe.analyse(path, progress=lambda stage: set_job(job_id, stage=stage))
            from avdf.media import probe
            result = evidence_urls(result["result_id"], result)
            result.update(mode="LIVE MODE", display_name=display_name(str(path)), created_at=utc_now(), media=probe(path))
        result["elapsed_sec"] = round(time.perf_counter() - started, 2)
        set_job(job_id, stage="Done", result=result)
    except Exception as exc:  # surfaced to the UI
        set_job(job_id, stage="Error", progress=100, error=f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------- app
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = None
    if mode() == "LIVE MODE":
        from avdf.pipeline import Pipeline
        app.state.pipeline = Pipeline(CONFIG_FILE, checkpoint_path())
    yield


app = FastAPI(title="AVDF Forensic Analysis API", version=__version__, lifespan=lifespan,
              description="Explainable, uncertainty-aware audio-visual deepfake detection.")
app.mount("/results", StaticFiles(directory=str(RESULTS)), name="results")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def frontend() -> str:
    return FRONTEND.read_text(encoding="utf-8")


@app.get("/health", tags=["system"])
def health():
    return {"status": "ok", **status()}


@app.get("/api/status", tags=["system"])
def api_status() -> dict[str, Any]:
    return status()


@app.get("/api/overview", tags=["system"])
def overview() -> dict[str, Any]:
    database = db()
    stats = database.stats()
    recent = database.audit(8, 0, "")
    for r in recent:
        r["display_name"] = display_name(r["file_path"])
    return {"status": status(), "stats": stats, "recent": recent,
            "jobs": sorted(({k: v for k, v in j.items() if k != "result"} for j in JOBS.values()),
                           key=lambda j: j["created_at"], reverse=True)[:8]}


@app.get("/api/configs", tags=["system"])
def configs() -> list[dict[str, Any]]:
    out = []
    for p in sorted((ROOT / "configs").glob("*.yaml")):
        c = load_config(p)
        cp = Path(str(c.paths.checkpoint).replace("best.pt", f"{c.experiment}_best.pt"))
        out.append({"file": p.name, "experiment": c.experiment, "mode": c.model.mode, "visual": c.model.visual_backbone,
                    "audio": c.model.audio_backbone, "evidential": bool(c.model.evidential),
                    "uncertainty": c.uncertainty.method, "epochs": c.train.epochs,
                    "trained": (cp if cp.is_absolute() else ROOT / cp).exists(), "active": p.name == Path(CONFIG_PATH).name})
    return out


@app.get("/api/fixtures", tags=["analysis"])
def fixtures() -> list[dict[str, Any]]:
    return [{"name": k, "file": v["file"], "title": v["title"], "verdict": v["verdict"], "decision": v["decision"]}
            for k, v in FIXTURES.items()]


@app.get("/api/fixtures/{name}", tags=["analysis"])
def fixture(name: str) -> dict[str, Any]:
    return fixture_result(name)


@app.post("/api/analyse", tags=["analysis"])
async def analyse(background_tasks: BackgroundTasks, file: UploadFile | None = File(None), fixture_name: str | None = None,
                  explain: bool = True) -> dict[str, str]:
    if fixture_name:
        if mode() != "DEMO MODE":
            raise HTTPException(409, "Fixtures are available only in DEMO MODE.")
        result = fixture_result(fixture_name)
        job_id = f"fixture-{uuid.uuid4().hex[:10]}"
        set_job(job_id, kind="fixture", filename=result["file"], stage="Done", result=result)
        return {"job_id": job_id}
    if file is None or not file.filename or Path(file.filename).suffix.lower() not in VIDEO_EXT:
        raise HTTPException(400, "Upload an MP4, AVI, MOV, MKV or WEBM video file.")
    path = UPLOADS / f"{uuid.uuid4().hex}_{Path(file.filename).name}"
    size = 0
    with path.open("wb") as out:  # stream to disk instead of holding the whole video in memory
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD:
                out.close(); path.unlink(missing_ok=True)
                raise HTTPException(413, f"File exceeds the {MAX_UPLOAD // (1024 * 1024)} MB upload limit.")
            out.write(chunk)
    if size == 0:
        path.unlink(missing_ok=True)
        raise HTTPException(400, "The uploaded file is empty.")
    job_id = uuid.uuid4().hex
    set_job(job_id, kind="upload", filename=file.filename, size_bytes=size, stage="Queued", result=None, explain=explain)
    background_tasks.add_task(process_upload, job_id, path, explain)
    return {"job_id": job_id}


@app.get("/api/jobs", tags=["analysis"])
def jobs() -> list[dict[str, Any]]:
    return sorted(({k: v for k, v in j.items() if k != "result"} for j in JOBS.values()),
                  key=lambda j: j["created_at"], reverse=True)


@app.get("/api/jobs/{job_id}", tags=["analysis"])
def job(job_id: str) -> dict[str, Any]:
    if job_id not in JOBS:
        raise HTTPException(404, "Analysis job not found.")
    return JOBS[job_id]


@app.get("/api/jobs/{job_id}/evidence/{kind}", tags=["analysis"], response_class=FileResponse)
def job_evidence(job_id: str, kind: str):
    path = JOB_FILES.get(job_id, {}).get(kind)
    if not path or not Path(path).exists():
        raise HTTPException(404, "Evidence not found.")
    return FileResponse(path)


@app.get("/api/cases/{result_id}", tags=["cases"])
def case(result_id: int) -> dict[str, Any]:
    row = db().case(result_id)
    if row is None:
        raise HTTPException(404, "Case not found.")
    row["display_name"] = display_name(row["file_path"])
    row["faces_preview"] = _faces_path(row)
    row["decision"] = "HUMAN REVIEW" if row["review_id"] else "AUTO-ACCEPT"
    _with_sidecar(row)
    return evidence_urls(result_id, row)


def _artifact(row: dict[str, Any], suffix: str) -> Path:
    return analyses_dir() / f"{row['result_id']}_{(row.get('input_hash') or '')[:8]}_{suffix}"


def _faces_path(row: dict[str, Any]) -> str | None:
    p = _artifact(row, "faces.png")
    return str(p) if p.exists() else None


def _with_sidecar(row: dict[str, Any]) -> dict[str, Any]:
    """Merge the pipeline's result JSON (method, p_fake, timings, ...) into a DB row without overriding it."""
    p = _artifact(row, "result.json")
    if p.exists():
        extra = json.loads(p.read_text(encoding="utf-8"))
        for k in ("method", "p_fake", "variance", "reason", "model", "mc_passes", "frames_analysed", "audio_sec",
                  "shap_top_bands", "timings_sec", "explain_errors"):
            if k in extra:
                row.setdefault(k, extra[k])
    return row


@app.get("/api/cases/{result_id}/evidence/{kind}", tags=["cases"], response_class=FileResponse)
def case_evidence(result_id: int, kind: str):
    row = db().case(result_id)
    if row is None or kind not in {"gradcam", "shap", "faces_preview"}:
        raise HTTPException(404, "Evidence not found.")
    path = _faces_path(row) if kind == "faces_preview" else row.get(f"{kind}_path")
    if not path or not Path(path).exists():
        raise HTTPException(404, "Evidence file not found.")
    return FileResponse(path)


class ReviewUpdate(BaseModel):
    status: str
    reviewer: str
    note: str


@app.get("/api/review", tags=["review"])
def review() -> list[dict[str, Any]]:
    rows = db().pending()
    for r in rows:
        r["display_name"] = display_name(r["file_path"])
    return rows


@app.get("/api/review/history", tags=["review"])
def review_history(limit: int = Query(50, ge=1, le=500)) -> list[dict[str, Any]]:
    rows = db().reviewed(limit)
    for r in rows:
        r["display_name"] = display_name(r["file_path"])
    return rows


@app.post("/api/review/{review_id}", tags=["review"])
def update_review(review_id: int, update: ReviewUpdate) -> dict[str, Any]:
    if update.status not in {"approved", "rejected", "escalated"} or not update.reviewer.strip() or not update.note.strip():
        raise HTTPException(400, "Status, reviewer name, and note are required.")
    database = db()
    if review_id not in {r["review_id"] for r in database.pending()}:
        raise HTTPException(404, "No pending review with this id.")
    database.resolve(review_id, update.status, reviewer=update.reviewer.strip(), note=update.note.strip())
    return {"review_id": review_id, "status": update.status, "history": database.review_history(review_id)}


@app.get("/api/audit", tags=["audit"])
def audit(limit: int = Query(25, ge=1, le=500), offset: int = Query(0, ge=0), q: str = "",
          verdict: str = Query("", pattern="^(real|fake)?$"),
          review: str = Query("", pattern="^(auto|pending|approved|rejected|escalated)?$")) -> dict[str, Any]:
    database = db()
    items = database.audit(limit, offset, q, verdict, review)
    for r in items:
        r["display_name"] = display_name(r["file_path"])
    return {"items": items, "total": database.audit_count(q, verdict, review), "limit": limit, "offset": offset}


@app.get("/api/audit/export.csv", tags=["audit"])
def audit_export(q: str = "", verdict: str = "", review: str = "") -> Response:
    rows = db().audit(100000, 0, q, verdict, review)
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(rows[0]) if rows else ["result_id"])
    writer.writeheader(); writer.writerows(rows)
    return Response(out.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=avdf_audit_{datetime.now():%Y%m%d_%H%M}.csv"})


def _scalar_metrics(m: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in m.items() if isinstance(v, (int, float, str)) or v is None}


@app.get("/api/metrics", tags=["results"])
def metrics() -> dict[str, Any]:
    """Everything experiment-related found under results/. Nothing here is hard-coded."""
    items, runs, curves = [], [], []
    summary = RESULTS / "summary.csv"
    if summary.exists():
        text = summary.read_text(encoding="utf-8")
        items.append({"name": summary.name, "type": "table", "content": text})
    for path in sorted(RESULTS.rglob("*_metrics.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        items.append({"name": str(path.relative_to(ROOT)).replace("\\", "/"), "type": "metrics", "content": m})
        runs.append({"run": path.stem.removesuffix("_metrics"), "experiment": path.parent.name, **_scalar_metrics(m),
                     "confusion_matrix": m.get("confusion_matrix"), "per_category_accuracy": m.get("per_category_accuracy")})
    for path in sorted(RESULTS.rglob("history.json")):
        curves.append({"experiment": path.parent.name, "history": json.loads(path.read_text(encoding="utf-8"))})
    skip = {"uploads", "analyses", "previews", "presentation_demo"}
    figures = [{"url": "/" + str(p.relative_to(ROOT)).replace("\\", "/"), "experiment": p.parent.name, "name": p.stem}
               for p in sorted(RESULTS.rglob("*.png")) if not skip & set(p.relative_to(RESULTS).parts)]
    return {"items": items, "runs": runs, "curves": curves, "figures": figures}


def _report_case(result_id: str) -> dict[str, Any]:
    if result_id.startswith("fixture-"):
        return fixture_result(result_id.removeprefix("fixture-"))
    if not result_id.isdigit():
        raise HTTPException(404, "Unknown case id.")
    row = db().case(int(result_id))
    if row is None:
        raise HTTPException(404, "Case not found.")
    row.update(display_name=display_name(row["file_path"]), sha256=row["input_hash"], confidence=row["confidence_score"],
               uncertainty=row["uncertainty_score"], mode=mode(), faces_preview=_faces_path(row),
               gradcam=row.get("gradcam_path"), shap=row.get("shap_path"),
               decision="HUMAN REVIEW" if row["review_id"] else "AUTO-ACCEPT")
    _with_sidecar(row)
    if "reason" not in row:
        from avdf.pipeline import route_reason
        c = cfg()
        row["reason"] = route_reason(row["confidence"], row["uncertainty"], c.uncertainty.tau, c.uncertainty.u_max)
    return row


@app.get("/api/report/{result_id}.pdf", tags=["reports"])
def report(result_id: str) -> Response:
    from avdf.report import build_pdf
    c = cfg()
    pdf = build_pdf(_report_case(result_id), float(c.uncertainty.tau), float(c.uncertainty.u_max))
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f"attachment; filename=avdf_report_{result_id}.pdf"})


@app.get("/api/report/{result_id}.json", tags=["reports"])
def report_json(result_id: str) -> dict[str, Any]:
    case_ = _report_case(result_id)
    for k in ("faces_preview", "gradcam", "shap", "gradcam_path", "shap_path"):
        case_.pop(k, None)  # no server filesystem paths in exported reports
    return {"generated_at": utc_now(), "config": CONFIG_PATH, "case": case_}
