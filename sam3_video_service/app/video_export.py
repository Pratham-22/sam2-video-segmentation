"""Build annotated MP4 from source video + per-frame mask PNGs."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from app import storage
from app.chunk_planner import plan_chunks
from app.config import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP
from app import video_io


class ExportError(RuntimeError):
    pass


def _mask_paths_for_upload(upload_id: str) -> dict[int, Path]:
    meta = storage.load_upload_meta(upload_id)
    plans = plan_chunks(
        meta["frame_count"],
        meta.get("chunk_size", DEFAULT_CHUNK_SIZE),
        meta.get("overlap", DEFAULT_OVERLAP),
    )
    out: dict[int, Path] = {}
    for plan in plans:
        masks_dir = storage.chunk_masks_dir(upload_id, plan.chunk_index)
        if not masks_dir.is_dir():
            continue
        for path in sorted(masks_dir.glob("*_obj*.png")):
            frame_idx = int(path.name.split("_", 1)[0])
            out[frame_idx] = path
        # Fallback: JSON-only masks from older runs (bbox rectangle).
        for path in sorted(masks_dir.glob("*_obj*.json")):
            frame_idx = int(path.name.split("_", 1)[0])
            if frame_idx not in out:
                out[frame_idx] = path
    return out


def export_annotated_video(upload_id: str) -> Path:
    """Overlay saved masks on source video; returns path to annotated.mp4."""
    meta = storage.load_upload_meta(upload_id)
    video_path = storage.source_video_path(upload_id)
    if not video_path.is_file():
        raise ExportError("source.mp4 missing")

    mask_map = _mask_paths_for_upload(upload_id)
    if not mask_map:
        raise ExportError("No masks found — track at least one chunk first")

    out_path = storage.export_video_path(upload_id)
    fps = float(meta.get("fps") or 30.0)
    frame_count = int(meta["frame_count"])
    overlay_color = (34, 197, 94)  # green RGB
    alpha = 0.45

    with tempfile.TemporaryDirectory(prefix="sam3_export_") as tmp:
        tmp_dir = Path(tmp)
        frames_dir = tmp_dir / "frames"
        annotated_dir = tmp_dir / "annotated"
        frames_dir.mkdir()
        annotated_dir.mkdir()

        pattern_in = (frames_dir / "%06d.jpg").as_posix()
        try:
            subprocess.check_output(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    video_path.as_posix(),
                    "-start_number",
                    "0",
                    "-q:v",
                    "2",
                    pattern_in,
                ],
                stderr=subprocess.STDOUT,
                text=True,
            )
        except FileNotFoundError as e:
            raise ExportError("ffmpeg not found; module load ffmpeg") from e
        except subprocess.CalledProcessError as e:
            raise ExportError(f"ffmpeg frame extract failed: {e.output}") from e

        for frame_idx in range(frame_count):
            src = frames_dir / f"{frame_idx:06d}.jpg"
            if not src.is_file():
                # ffmpeg may use 1-based names when start_number omitted on older builds
                src = frames_dir / f"{frame_idx + 1:06d}.jpg"
            if not src.is_file():
                raise ExportError(f"Missing extracted frame {frame_idx}")

            base = Image.open(src).convert("RGBA")
            mask_path = mask_map.get(frame_idx)
            if mask_path and mask_path.is_file():
                if mask_path.suffix == ".png":
                    mask = Image.open(mask_path).convert("L").resize(base.size, Image.NEAREST)
                    mask_arr = np.array(mask) > 127
                else:
                    mask_arr = _mask_from_json(mask_path, base.size)
                if mask_arr.any():
                    overlay = Image.new("RGBA", base.size, (*overlay_color, 0))
                    overlay_arr = np.array(overlay)
                    overlay_arr[mask_arr, 3] = int(255 * alpha)
                    base = Image.alpha_composite(base, Image.fromarray(overlay_arr, "RGBA"))

            dst = annotated_dir / f"{frame_idx:06d}.jpg"
            base.convert("RGB").save(dst, quality=92)

        pattern_out = (annotated_dir / "%06d.jpg").as_posix()
        encode_base = [
            "ffmpeg",
            "-y",
            "-framerate",
            f"{fps:.6f}",
            "-start_number",
            "0",
            "-i",
            pattern_out,
        ]
        encode_tail = ["-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path.as_posix()]
        last_err: subprocess.CalledProcessError | None = None
        for codec in ("libx264", "mpeg4", "mjpeg"):
            try:
                subprocess.check_output(
                    encode_base + ["-c:v", codec] + encode_tail,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                break
            except subprocess.CalledProcessError as e:
                last_err = e
        else:
            detail = last_err.output if last_err else "no encoder"
            raise ExportError(f"ffmpeg encode failed: {detail}") from last_err

    meta["export_path"] = out_path.name
    meta["export_status"] = "ready"
    storage.save_upload_meta(upload_id, meta)
    return out_path


def _mask_from_json(json_path: Path, size: tuple[int, int]) -> np.ndarray:
    import json

    data = json.loads(json_path.read_text(encoding="utf-8"))
    w, h = size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    bbox = data.get("bbox") or []
    if len(bbox) == 4:
        draw.rectangle(bbox, fill=255)
    return np.array(mask) > 127
