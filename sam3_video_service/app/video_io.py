"""Video probe and frame extraction via ffmpeg."""

from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path


class VideoIOError(RuntimeError):
    pass


def probe_video(path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames,avg_frame_rate,width,height,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        path.as_posix(),
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    except FileNotFoundError as e:
        raise VideoIOError("ffprobe not found; install ffmpeg") from e
    except subprocess.CalledProcessError as e:
        raise VideoIOError(f"ffprobe failed: {e.output}") from e

    data = json.loads(out)
    streams = data.get("streams") or []
    if not streams:
        raise VideoIOError(f"No video stream in {path}")
    stream = streams[0]
    fmt = data.get("format") or {}
    duration = float(stream.get("duration") or fmt.get("duration") or 0)
    fps = _parse_fps(stream.get("avg_frame_rate") or "30/1")
    frame_count = stream.get("nb_frames")
    if frame_count is not None and str(frame_count) not in ("N/A", "0"):
        frame_count = int(frame_count)
    elif duration > 0 and fps > 0:
        frame_count = max(1, int(round(duration * fps)))
    else:
        frame_count = _count_frames_slow(path)
    return {
        "frame_count": frame_count,
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "duration": duration,
        "fps": fps,
    }


def _count_frames_slow(path: Path) -> int:
    """Last resort — reads every frame; can take minutes on long AVI files."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "json",
        path.as_posix(),
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as e:
        raise VideoIOError(f"ffprobe frame count failed: {e.output}") from e
    data = json.loads(out)
    streams = data.get("streams") or []
    if not streams:
        return 1
    frame_count = streams[0].get("nb_read_frames")
    if frame_count is None or str(frame_count) == "N/A":
        return 1
    return max(1, int(frame_count))


def _parse_fps(rate: str) -> float:
    if "/" in rate:
        num, den = rate.split("/", 1)
        den_f = float(den) or 1.0
        return float(num) / den_f
    return float(rate)


def extract_frames(
    video_path: Path,
    out_dir: Path,
    start_frame: int,
    end_frame: int,
    fps: float | None = None,
) -> list[Path]:
    """Extract inclusive frame range to out_dir/%06d.jpg (0-based global indices in names)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()

    # Use select filter for exact frame indices when fps known; fallback ss/to by time.
    if fps is None:
        info = probe_video(video_path)
        fps = float(info.get("fps") or 30.0)

    start_t = start_frame / fps
    end_t = (end_frame + 1) / fps

    pattern = (out_dir / "%06d.jpg").as_posix()
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_t:.6f}",
        "-to",
        f"{end_t:.6f}",
        "-i",
        video_path.as_posix(),
        "-vsync",
        "0",
        "-q:v",
        "2",
        pattern,
    ]
    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    except FileNotFoundError as e:
        raise VideoIOError("ffmpeg not found; install ffmpeg") from e
    except subprocess.CalledProcessError as e:
        raise VideoIOError(f"ffmpeg extract failed: {e.output}") from e

    produced = sorted(out_dir.glob("*.jpg"))
    # Rename to global frame indices.
    renamed: list[Path] = []
    for offset, src in enumerate(produced):
        global_idx = start_frame + offset
        if global_idx > end_frame:
            src.unlink(missing_ok=True)
            continue
        dst = out_dir / f"{global_idx:06d}.jpg"
        if dst != src:
            src.rename(dst)
        renamed.append(dst)
    return renamed


def pick_sample_frame(start: int, end: int, exclude: set[int] | None = None) -> int:
    exclude = exclude or set()
    candidates = [i for i in range(start, end + 1) if i not in exclude]
    if not candidates:
        return start
    mid = (start + end) // 2
    if mid in candidates:
        return mid
    return random.choice(candidates)
