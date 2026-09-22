"""Preprocessing: validation, demuxing, MTCNN face crops (visual) and 16 kHz mono audio.

Cached per clip:  <cache>/<clip_id>_faces.pt  (uint8, T x 3 x 224 x 224)
                  <cache>/<clip_id>.wav       (16 kHz mono)
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm

from .media import ffmpeg_exe, probe


def probe_streams(path):
    """Return (has_video, has_audio)."""
    info = probe(path)
    return info["has_video"], info["has_audio"]


def extract_audio(path, sr=16000):
    """Demux audio to 16 kHz mono float32 numpy array using ffmpeg."""
    with tempfile.TemporaryDirectory() as tmp:   # a directory, so ffmpeg can reopen the file on Windows
        out = Path(tmp) / "audio.wav"
        proc = subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar", str(sr), "-vn", str(out)],
                              capture_output=True, text=True, errors="replace")
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(f"ffmpeg exited with code {proc.returncode}: {proc.stderr.strip() or 'no output written'}")
        wav, _ = sf.read(out, dtype="float32")
    if len(wav) == 0:
        raise RuntimeError("ffmpeg decoded 0 audio samples")
    return wav


def trim_silence(wav, sr=16000, top_db=30):
    """Voice-activity trimming of leading/trailing silence (librosa energy-based VAD)."""
    import librosa
    trimmed, _ = librosa.effects.trim(wav, top_db=top_db)
    return trimmed if len(trimmed) > sr * 0.5 else wav


def read_frames(path, fps=25):
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    step = max(src_fps / fps, 1.0)
    frames, i, nxt = [], 0, 0.0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i >= nxt:
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            nxt += step
        i += 1
    cap.release()
    return frames


def read_frames_sampled(path, n_frames):
    """Decode only n_frames evenly spaced frames (grab() skips the rest without converting them), so
    long 1080p clips such as DFDC never sit fully in memory. Returns [] if the frame count is unknown."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return []
    want = set(np.linspace(0, total - 1, n_frames).astype(int).tolist())
    frames, last = [], max(want)
    for i in range(last + 1):
        if not cap.grab():
            break
        if i in want:
            ok, fr = cap.retrieve()
            if ok:
                frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


class FaceExtractor:
    def __init__(self, image_size=224, device="cpu"):
        from facenet_pytorch import MTCNN
        self.size = image_size
        self.mtcnn = MTCNN(image_size=image_size, margin=20, post_process=False, select_largest=True, device=device)
        self.last_detected = 0       # frames with a detected face in the most recent crop() call

    def crop(self, video_path, n_frames=32, fps=25):
        frames = read_frames_sampled(video_path, n_frames)
        if len(frames) < n_frames:          # unknown/wrong frame count in the header: decode everything
            frames = read_frames(video_path, fps)
        if not frames:
            raise ValueError("no frames decoded")
        idx = np.linspace(0, len(frames) - 1, n_frames).astype(int)
        faces, last, self.last_detected = [], None, 0
        for i in idx:
            f = self.mtcnn(frames[i])
            if f is None:            # detection miss -> reuse previous face or centre crop
                f = last if last is not None else self._center(frames[i])
            else:
                self.last_detected += 1
            last = f
            faces.append(f.clamp(0, 255).to(torch.uint8))
        return torch.stack(faces)  # (T,3,H,W) uint8

    def _center(self, fr):
        h, w = fr.shape[:2]
        s = min(h, w)
        c = fr[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2]
        c = cv2.resize(c, (self.size, self.size))
        return torch.from_numpy(c).permute(2, 0, 1).float()


def preprocess_clip(path, extractor, n_frames=32, fps=25, sr=16000, progress=None):
    progress = progress or (lambda stage: None)
    has_v, has_a = probe_streams(path)
    if not (has_v and has_a):
        raise ValueError(f"missing stream (video={has_v}, audio={has_a})")
    progress("Extracting faces")
    faces = extractor.crop(path, n_frames, fps)
    progress("Extracting audio")
    wav = trim_silence(extract_audio(path, sr), sr)
    return faces, wav


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--n_frames", type=int, default=32)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.manifest)
    ext = FaceExtractor(a.image_size, device=a.device)
    bad = []
    for r in tqdm(df.itertuples(), total=len(df), desc="preprocess"):
        fp, wp = cache / f"{r.clip_id}_faces.pt", cache / f"{r.clip_id}.wav"
        if fp.exists() and wp.exists():
            continue
        try:
            faces, wav = preprocess_clip(r.path, ext, a.n_frames)
            torch.save(faces, fp)
            sf.write(wp, wav, 16000)
        except Exception as e:  # noqa
            bad.append((r.clip_id, r.path, str(e)))
    if bad:
        pd.DataFrame(bad, columns=["clip_id", "path", "error"]).to_csv(cache / "failed.csv", index=False)
        keep = ~df.clip_id.isin([b[0] for b in bad])
        df[keep].to_csv(a.manifest, index=False)
        print(f"{len(bad)} clips failed validation and were removed from the manifest (see failed.csv)")
    print("done")


if __name__ == "__main__":
    main()
