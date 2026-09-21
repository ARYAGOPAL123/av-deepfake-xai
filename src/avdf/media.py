"""Media helpers: locate ffmpeg (system or bundled via imageio-ffmpeg) and inspect video files without ffprobe."""
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import cv2


@lru_cache(maxsize=1)
def ffmpeg_exe():
    """Return an ffmpeg executable path: system ffmpeg first, then the imageio-ffmpeg bundled binary."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:  # noqa
        raise RuntimeError("ffmpeg not found. Install ffmpeg or `pip install imageio-ffmpeg`.") from e


def ffmpeg_available():
    try:
        ffmpeg_exe()
        return True
    except RuntimeError:
        return False


def _ffmpeg_info(path):
    """Parse the stream summary that `ffmpeg -i` prints to stderr."""
    err = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True,
                         errors="replace").stderr
    video = re.search(r"Stream #\S+.*?Video: (\w+)", err)
    audio = re.search(r"Stream #\S+.*?Audio: (\w+)[^\n]*?(\d+) Hz(?:, ([\w.() ]+?))?,", err)
    dur = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err)
    return {
        "video_codec": video.group(1) if video else None,
        "audio_codec": audio.group(1) if audio else None,
        "audio_sample_rate": int(audio.group(2)) if audio else None,
        "audio_channels": audio.group(3).strip() if audio and audio.group(3) else None,
        "duration_sec": int(dur.group(1)) * 3600 + int(dur.group(2)) * 60 + float(dur.group(3)) if dur else None,
    }


def probe(path):
    """Return stream/format metadata for a media file."""
    cap = cv2.VideoCapture(str(path))
    opened = cap.isOpened()
    fps = cap.get(cv2.CAP_PROP_FPS) if opened else 0.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0
    decodable = opened and cap.read()[0]
    cap.release()
    info = {"has_video": bool(decodable), "has_audio": False, "fps": round(fps, 3) if fps else None,
            "frame_count": frames or None, "width": width or None, "height": height or None,
            "size_bytes": Path(path).stat().st_size}
    if ffmpeg_available():
        ff = _ffmpeg_info(path)
        info.update(ff)
        info["has_audio"] = ff["audio_codec"] is not None
        info["has_video"] = info["has_video"] or ff["video_codec"] is not None
    if not info.get("duration_sec") and fps and frames:
        info["duration_sec"] = frames / fps
    return info
