"""End-to-end analysis of one video: preprocess -> predict + uncertainty -> route -> explain -> audit log.

python -m avdf.pipeline --config configs/default.yaml --video path/to/clip.mp4
"""
import argparse
import hashlib
import json
import time
from pathlib import Path


from . import LABELS
from .datasets import fix_length, normalise_faces, normalise_wav
from .db import AuditDB
from .evaluate import load_checkpoint
from .explain import gradcam_frames, plot_shap, shap_audio
from .explain.gradcam import save_gradcam_grid
from .preprocess import FaceExtractor, preprocess_clip
from .uncertainty import predict_with_uncertainty, route
from .utils import ensure_dir, get_device, load_config


class Pipeline:
    def __init__(self, config="configs/default.yaml", checkpoint=None, explain=True):
        self.cfg = load_config(config)
        self.dev = get_device(self.cfg)
        self.model, self.mcfg = load_checkpoint(self.cfg, checkpoint, self.dev)
        self.extractor = FaceExtractor(self.mcfg.data.image_size, device=str(self.dev))
        self.db = AuditDB(self.cfg.paths.db)
        self.model_name = f"{self.mcfg.model.mode}:{self.mcfg.model.visual_backbone}+{self.mcfg.model.audio_backbone}"
        self.model_id = self.db.register_model(self.model_name, self.mcfg.data.get("dataset", "FakeAVCeleb"))
        self.explain = explain
        self.out = ensure_dir(Path(self.cfg.paths.results_dir) / "analyses")

    def analyse(self, video_path, progress=None):
        """progress: optional callback(stage: str) called as each stage starts."""
        step = progress or (lambda stage: None)
        d, u = self.mcfg.data, self.cfg.uncertainty
        timings, t = {}, time.perf_counter()

        def lap(name):
            nonlocal t
            now = time.perf_counter(); timings[name] = round(now - t, 3); t = now

        step("Validating")
        sha = hashlib.sha256(Path(video_path).read_bytes()).hexdigest()
        step("Extracting faces")
        faces_u8, wav = preprocess_clip(video_path, self.extractor, d.n_frames, d.fps, d.sample_rate)
        faces = normalise_faces(faces_u8)[None].to(self.dev)
        wav_t = normalise_wav(fix_length(wav, int(d.sample_rate * d.max_audio_sec)))[None].to(self.dev)
        lap("preprocess")
        step("Inference")
        method = "evidential" if self.mcfg.model.evidential else u.method
        o = predict_with_uncertainty(self.model, faces, wav_t, method, u.mc_passes)
        step("Uncertainty")
        pred, conf, review = route(o["prob"], o["uncertainty"], u.tau, u.u_max)
        lap("inference")
        res = dict(file=str(video_path), sha256=sha, verdict=LABELS[int(pred)], p_fake=float(o["prob"][0, 1]),
                   confidence=float(conf), uncertainty=float(o["uncertainty"][0]), variance=float(o["variance"][0]),
                   review=bool(review), decision="HUMAN REVIEW" if bool(review) else "AUTO-ACCEPT", method=method,
                   reason=route_reason(float(conf), float(o["uncertainty"][0]), u.tau, u.u_max),
                   model=self.model_name, mc_passes=int(u.mc_passes) if method == "mc_dropout" else None,
                   frames_analysed=int(faces_u8.shape[0]), audio_sec=round(len(wav) / d.sample_rate, 2))
        rid = self.db.log_result(str(video_path), sha, res["verdict"], res["confidence"], res["uncertainty"], self.model_id, len(wav) / d.sample_rate)
        res["result_id"] = rid
        stem = self.out / f"{rid}_{sha[:8]}"
        res["faces_preview"] = str(save_face_strip(faces_u8, f"{stem}_faces.png"))
        if self.explain:
            step("Explanations")
            gp = sp = None
            errors = []   # the verdict is already logged, so an explanation failure must not lose it
            if self.model.visual is not None:
                try:
                    cams, _ = gradcam_frames(self.model, faces, wav_t)
                    gp = str(save_gradcam_grid(faces.cpu(), cams, f"{stem}_gradcam.png"))
                except Exception as ex:  # noqa
                    errors.append(f"Grad-CAM failed: {ex}")
            if self.model.audio is not None:
                try:
                    e = self.cfg.explain
                    phi, base, hz = shap_audio(self.model, faces, wav_t, e.shap_bands, e.shap_segments, e.shap_nsamples)
                    sp = str(plot_shap(phi, hz, f"{stem}_shap.png"))
                    res["shap_top_bands"] = top_bands(phi, hz)
                except Exception as ex:  # noqa
                    errors.append(f"SHAP failed: {ex}")
            if errors:
                res["explain_errors"] = errors
            self.db.add_explanation(rid, gp, sp)
            res.update(gradcam=gp, shap=sp)
            lap("explain")
        if res["review"]:
            self.db.enqueue(rid)
        res["timings_sec"] = timings
        # full result next to the evidence images: fields the audit schema does not store (method, p_fake, ...)
        Path(f"{stem}_result.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        return res


def route_reason(conf, unc, tau, u_max):
    """Plain-language explanation of the routing rule for this case."""
    why = []
    if conf < tau:
        why.append(f"confidence {conf:.2f} is below tau = {tau}")
    if unc > u_max:
        why.append(f"uncertainty {unc:.3f} exceeds u_max = {u_max}")
    if why:
        return "Sent to human review: " + " and ".join(why) + "."
    return f"Auto-accepted: confidence {conf:.2f} >= tau = {tau} and uncertainty {unc:.3f} <= u_max = {u_max}."


def top_bands(phi, hz, k=3):
    """The k Mel bands with the largest absolute SHAP contribution (sign kept)."""
    band = phi.sum(1)
    order = sorted(range(len(band)), key=lambda i: -abs(band[i]))[:k]
    return [dict(band=f"{int(hz[i])}-{int(hz[i + 1])} Hz", shap=float(band[i])) for i in order]


def save_face_strip(faces_u8, path, k=6):
    """Save up to k evenly spaced face crops (uint8 T x 3 x H x W) side by side."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    idx = np.linspace(0, len(faces_u8) - 1, min(k, len(faces_u8))).astype(int)
    fig, axes = plt.subplots(1, len(idx), figsize=(2.2 * len(idx), 2.4))
    for ax, i in zip(np.atleast_1d(axes), idx):
        ax.imshow(faces_u8[i].permute(1, 2, 0).numpy()); ax.set_title(f"frame {i}", fontsize=9); ax.axis("off")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close(fig)
    return path


def save_face_grid(faces_u8, path, cols=8, size=112):
    """Save every face crop (uint8 T x 3 x H x W) as one tiled PNG, frames in order left-to-right."""
    import cv2
    import numpy as np
    tiles = [cv2.resize(f.permute(1, 2, 0).numpy(), (size, size), interpolation=cv2.INTER_AREA) for f in faces_u8]
    rows = -(-len(tiles) // cols)
    grid = np.full((rows * (size + 4) - 4, min(cols, len(tiles)) * (size + 4) - 4, 3), 255, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        grid[r * (size + 4):r * (size + 4) + size, c * (size + 4):c * (size + 4) + size] = t
    cv2.imwrite(str(path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    return path

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--video", required=True)
    ap.add_argument("--no_explain", action="store_true")
    a = ap.parse_args()
    print(json.dumps(Pipeline(a.config, a.checkpoint, not a.no_explain).analyse(a.video), indent=1))
