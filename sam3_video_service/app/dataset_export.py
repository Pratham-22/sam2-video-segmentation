"""Export labeled frames as COCO / YOLO JSON and bbox-overlay image ZIPs."""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from app import storage
from app.chunk_planner import plan_chunks
from app.config import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP


class DatasetExportError(RuntimeError):
    pass


DEFAULT_CATEGORY = "object"


@dataclass
class FrameObject:
    frame_idx: int
    obj_id: int
    bbox_xyxy: list[float]  # x1, y1, x2, y2
    width: int
    height: int
    segmentation: list[list[float]] | None = None  # [[x,y], ...]
    confidence: float | None = None
    mask_png: Path | None = None


def _xyxy_to_xywh(bbox: list[float]) -> list[float]:
    x1, y1, x2, y2 = bbox
    return [float(x1), float(y1), float(max(0.0, x2 - x1)), float(max(0.0, y2 - y1))]


def _xyxy_to_yolo(bbox: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = bbox
    bw = max(1e-6, x2 - x1)
    bh = max(1e-6, y2 - y1)
    cx = (x1 + x2) / 2.0 / width
    cy = (y1 + y2) / 2.0 / height
    return [
        float(np.clip(cx, 0.0, 1.0)),
        float(np.clip(cy, 0.0, 1.0)),
        float(np.clip(bw / width, 0.0, 1.0)),
        float(np.clip(bh / height, 0.0, 1.0)),
    ]


def _bbox_from_mask(mask_path: Path) -> tuple[list[float], int, int] | None:
    mask = Image.open(mask_path).convert("L")
    arr = np.array(mask)
    ys, xs = np.where(arr > 127)
    if len(xs) == 0:
        return None
    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())
    return [x1, y1, x2, y2], mask.width, mask.height


def _normalize_bbox(raw: list[Any], width: int, height: int) -> list[float] | None:
    if len(raw) != 4:
        return None
    vals = [float(v) for v in raw]
    x1, y1, x2, y2 = vals
    # Heuristic: COCO xywh if w/h look like sizes and x2/y2 < image dims awkwardly.
    # Existing SAM3 JSON uses xyxy (x2 > x1 and often much larger than a typical width field).
    if x2 > x1 and y2 > y1 and (x2 <= width + 1 and y2 <= height + 1):
        # Could still be xywh if x2,y2 are width/height. Prefer xyxy when x2 or y2
        # exceeds a plausible object size relative to origin — our stored format is xyxy.
        return [
            max(0.0, min(x1, width - 1)),
            max(0.0, min(y1, height - 1)),
            max(0.0, min(x2, width)),
            max(0.0, min(y2, height)),
        ]
    # Treat as xywh
    return [
        max(0.0, x1),
        max(0.0, y1),
        max(0.0, min(x1 + x2, width)),
        max(0.0, min(y1 + y2, height)),
    ]


def collect_frame_objects(upload_id: str) -> list[FrameObject]:
    meta = storage.load_upload_meta(upload_id)
    width = int(meta.get("width") or 0)
    height = int(meta.get("height") or 0)
    plans = plan_chunks(
        meta["frame_count"],
        meta.get("chunk_size", DEFAULT_CHUNK_SIZE),
        meta.get("overlap", DEFAULT_OVERLAP),
    )
    objects: list[FrameObject] = []

    for plan in plans:
        masks_dir = storage.chunk_masks_dir(upload_id, plan.chunk_index)
        if not masks_dir.is_dir():
            continue

        by_key: dict[tuple[int, int], FrameObject] = {}

        json_paths = {p.stem: p for p in masks_dir.glob("*_obj*.json")}
        png_paths = {p.stem: p for p in masks_dir.glob("*_obj*.png")}

        for stem, path in sorted(json_paths.items()):
            try:
                frame_part, obj_part = stem.split("_obj", 1)
                frame_idx = int(frame_part)
                obj_id = int(obj_part)
            except ValueError:
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            w = int(data.get("width") or width or 0)
            h = int(data.get("height") or height or 0)
            png = png_paths.get(stem)
            if (w <= 0 or h <= 0) and png is not None:
                with Image.open(png) as im:
                    w, h = im.size
            if w <= 0 or h <= 0:
                continue
            bbox = _normalize_bbox(list(data.get("bbox") or []), w, h)
            if bbox is None and png is not None:
                derived = _bbox_from_mask(png)
                if derived is not None:
                    bbox, w, h = derived
            if bbox is None:
                continue
            seg = data.get("segmentation")
            segmentation = None
            if isinstance(seg, list) and seg:
                if isinstance(seg[0], (int, float)):
                    pts = [[float(seg[i]), float(seg[i + 1])] for i in range(0, len(seg) - 1, 2)]
                    segmentation = pts
                elif isinstance(seg[0], list):
                    segmentation = [[float(p[0]), float(p[1])] for p in seg if len(p) >= 2]
            by_key[(frame_idx, obj_id)] = FrameObject(
                frame_idx=frame_idx,
                obj_id=obj_id,
                bbox_xyxy=bbox,
                width=w,
                height=h,
                segmentation=segmentation,
                confidence=float(data["confidence"]) if data.get("confidence") is not None else None,
                mask_png=png,
            )

        for stem, path in sorted(png_paths.items()):
            if stem in json_paths:
                continue
            try:
                frame_part, obj_part = stem.split("_obj", 1)
                frame_idx = int(frame_part)
                obj_id = int(obj_part)
            except ValueError:
                continue
            derived = _bbox_from_mask(path)
            if derived is None:
                continue
            bbox, mw, mh = derived
            by_key[(frame_idx, obj_id)] = FrameObject(
                frame_idx=frame_idx,
                obj_id=obj_id,
                bbox_xyxy=bbox,
                width=width or mw,
                height=height or mh,
                mask_png=path,
            )

        objects.extend(by_key.values())

    objects.sort(key=lambda o: (o.frame_idx, o.obj_id))
    if not objects:
        raise DatasetExportError("No masks found — track at least one chunk first")
    return objects


def _stem_name(upload_id: str) -> str:
    meta = storage.load_upload_meta(upload_id)
    return Path(meta.get("original_filename") or upload_id).stem


def export_dir(upload_id: str) -> Path:
    path = storage.upload_dir(upload_id) / "dataset_exports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_coco_json(upload_id: str, category_name: str = DEFAULT_CATEGORY) -> Path:
    objects = collect_frame_objects(upload_id)
    meta = storage.load_upload_meta(upload_id)
    stem = _stem_name(upload_id)

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    image_ids: dict[int, int] = {}

    for obj in objects:
        if obj.frame_idx not in image_ids:
            image_id = len(image_ids) + 1
            image_ids[obj.frame_idx] = image_id
            file_name = f"{stem}_{obj.frame_idx:06d}.jpg"
            images.append(
                {
                    "id": image_id,
                    "file_name": file_name,
                    "width": obj.width,
                    "height": obj.height,
                    "frame_idx": obj.frame_idx,
                }
            )
        image_id = image_ids[obj.frame_idx]
        xywh = _xyxy_to_xywh(obj.bbox_xyxy)
        ann: dict[str, Any] = {
            "id": len(annotations) + 1,
            "image_id": image_id,
            "category_id": 1,
            "bbox": [round(v, 2) for v in xywh],
            "area": round(xywh[2] * xywh[3], 2),
            "iscrowd": 0,
            "obj_id": obj.obj_id,
        }
        if obj.segmentation:
            flat = [coord for pt in obj.segmentation for coord in pt]
            if len(flat) >= 6:
                ann["segmentation"] = [flat]
        if obj.confidence is not None:
            ann["score"] = obj.confidence
        annotations.append(ann)

    coco = {
        "info": {
            "description": f"SAM3 video labels for {stem}",
            "version": "1.0",
            "upload_id": upload_id,
            "original_filename": meta.get("original_filename"),
            "fps": meta.get("fps"),
            "frame_count": meta.get("frame_count"),
        },
        "licenses": [],
        "categories": [
            {"id": 1, "name": category_name, "supercategory": "object"},
        ],
        "images": images,
        "annotations": annotations,
    }
    out = export_dir(upload_id) / f"{stem}_coco.json"
    out.write_text(json.dumps(coco, indent=2), encoding="utf-8")
    return out


def build_yolo_json(upload_id: str, category_name: str = DEFAULT_CATEGORY) -> Path:
    objects = collect_frame_objects(upload_id)
    meta = storage.load_upload_meta(upload_id)
    stem = _stem_name(upload_id)

    by_frame: dict[int, list[FrameObject]] = {}
    for obj in objects:
        by_frame.setdefault(obj.frame_idx, []).append(obj)

    annotations = []
    for frame_idx in sorted(by_frame):
        objs = by_frame[frame_idx]
        w, h = objs[0].width, objs[0].height
        annotations.append(
            {
                "file_name": f"{stem}_{frame_idx:06d}.jpg",
                "frame_idx": frame_idx,
                "width": w,
                "height": h,
                "objects": [
                    {
                        "class_id": 0,
                        "class_name": category_name,
                        "obj_id": o.obj_id,
                        "bbox_xyxy": [round(v, 2) for v in o.bbox_xyxy],
                        "bbox_xywhn": [round(v, 6) for v in _xyxy_to_yolo(o.bbox_xyxy, w, h)],
                        "confidence": o.confidence,
                    }
                    for o in objs
                ],
            }
        )

    payload = {
        "format": "yolo",
        "classes": [category_name],
        "upload_id": upload_id,
        "original_filename": meta.get("original_filename"),
        "note": (
            "bbox_xywhn is YOLO normalized [cx, cy, w, h]. "
            "Each objects[] entry can also be written as a line: "
            "`class_id cx cy w h` in labels/<stem>.txt."
        ),
        "annotations": annotations,
    }
    out = export_dir(upload_id) / f"{stem}_yolo.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def _find_frame_image(upload_id: str, frame_idx: int) -> Path | None:
    meta = storage.load_upload_meta(upload_id)
    plans = plan_chunks(
        meta["frame_count"],
        meta.get("chunk_size", DEFAULT_CHUNK_SIZE),
        meta.get("overlap", DEFAULT_OVERLAP),
    )
    for plan in plans:
        if plan.process_start <= frame_idx <= plan.process_end:
            path = storage.chunk_frames_dir(upload_id, plan.chunk_index) / f"{frame_idx:06d}.jpg"
            if path.is_file():
                return path
    # Any chunk may still hold the frame after overlap trimming.
    for plan in plans:
        path = storage.chunk_frames_dir(upload_id, plan.chunk_index) / f"{frame_idx:06d}.jpg"
        if path.is_file():
            return path
    return None


def _extract_frames_for_indices(upload_id: str, frame_indices: list[int], dest_dir: Path) -> dict[int, Path]:
    """Extract only requested frames from the source video into dest_dir."""
    import cv2

    video_path = storage.source_video_path(upload_id)
    if not video_path.is_file():
        raise DatasetExportError("source video missing")

    dest_dir.mkdir(parents=True, exist_ok=True)
    needed = set(frame_indices)
    found: dict[int, Path] = {}

    # Prefer already-extracted chunk frames.
    for idx in list(needed):
        existing = _find_frame_image(upload_id, idx)
        if existing is not None:
            out = dest_dir / f"{idx:06d}.jpg"
            if not out.is_file():
                Image.open(existing).convert("RGB").save(out, quality=92)
            found[idx] = out
            needed.discard(idx)

    if not needed:
        return found

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise DatasetExportError(f"Could not open video: {video_path}")

    # Sequential read is more reliable than random seek on some AVI files.
    frame_idx = 0
    max_needed = max(needed)
    while frame_idx <= max_needed:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx in needed:
            out = dest_dir / f"{frame_idx:06d}.jpg"
            if not cv2.imwrite(str(out), frame):
                cap.release()
                raise DatasetExportError(f"Failed to write frame {frame_idx}")
            found[frame_idx] = out
            needed.discard(frame_idx)
            if not needed:
                break
        frame_idx += 1
    cap.release()

    if needed:
        sample = sorted(needed)[:5]
        raise DatasetExportError(
            f"Could not extract {len(needed)} frame(s) from video (e.g. {sample})"
        )
    return found


def build_bbox_overlay_zip(upload_id: str, category_name: str = DEFAULT_CATEGORY) -> Path:
    objects = collect_frame_objects(upload_id)
    stem = _stem_name(upload_id)
    out_zip = export_dir(upload_id) / f"{stem}_bbox_images.zip"

    by_frame: dict[int, list[FrameObject]] = {}
    for obj in objects:
        by_frame.setdefault(obj.frame_idx, []).append(obj)

    import tempfile

    with tempfile.TemporaryDirectory(prefix="sam3_bbox_") as tmp:
        tmp_dir = Path(tmp)
        frames_dir = tmp_dir / "frames"
        overlays_dir = tmp_dir / "overlays"
        overlays_dir.mkdir()
        frame_paths = _extract_frames_for_indices(upload_id, sorted(by_frame), frames_dir)

        for frame_idx, objs in by_frame.items():
            src = frame_paths[frame_idx]
            img = Image.open(src).convert("RGB")
            draw = ImageDraw.Draw(img)
            for obj in objs:
                x1, y1, x2, y2 = obj.bbox_xyxy
                draw.rectangle([x1, y1, x2, y2], outline=(34, 197, 94), width=3)
                label = f"{category_name}#{obj.obj_id}"
                # Simple text background
                tx, ty = x1, max(0, y1 - 16)
                draw.rectangle([tx, ty, tx + 8 * len(label), ty + 14], fill=(34, 197, 94))
                draw.text((tx + 2, ty + 1), label, fill=(0, 0, 0))
            out_name = f"{stem}_{frame_idx:06d}.jpg"
            dest = overlays_dir / out_name
            img.save(dest, quality=92)

        with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(overlays_dir.glob("*.jpg")):
                zf.write(path, arcname=f"bbox_images/{path.name}")

    return out_zip


def coco_json_path(upload_id: str) -> Path:
    return export_dir(upload_id) / f"{_stem_name(upload_id)}_coco.json"


def yolo_json_path(upload_id: str) -> Path:
    return export_dir(upload_id) / f"{_stem_name(upload_id)}_yolo.json"


def bbox_zip_path(upload_id: str) -> Path:
    return export_dir(upload_id) / f"{_stem_name(upload_id)}_bbox_images.zip"
