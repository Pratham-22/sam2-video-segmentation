"""HTTP client for the SAM3 video GPU service."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import requests


class Sam3VideoClient:
    def __init__(self, base_url: str, timeout: int = 600) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict[str, Any]:
        r = requests.get(f"{self.base_url}/health", timeout=10)
        r.raise_for_status()
        return r.json()

    def finalize_upload(self, upload_id: str, original_filename: str) -> dict[str, Any]:
        r = requests.post(
            f"{self.base_url}/uploads/{upload_id}/finalize",
            json={"original_filename": original_filename},
            timeout=self.timeout,
        )
        if not r.ok:
            detail = r.text
            try:
                body = r.json()
                detail = body.get("detail", detail)
            except Exception:
                pass
            raise requests.HTTPError(
                f"Finalize upload failed ({r.status_code}): {detail}",
                response=r,
            )
        return r.json()

    def upload(self, video_path: Path) -> dict[str, Any]:
        with video_path.open("rb") as f:
            r = requests.post(
                f"{self.base_url}/uploads",
                files={"file": (video_path.name, f, "video/mp4")},
                timeout=self.timeout,
            )
        if not r.ok:
            detail = r.text
            try:
                body = r.json()
                detail = body.get("detail", detail)
            except Exception:
                pass
            raise requests.HTTPError(
                f"SAM3 upload failed ({r.status_code}): {detail}",
                response=r,
            )
        return r.json()

    def list_chunks(self, upload_id: str) -> dict[str, Any]:
        r = requests.get(f"{self.base_url}/uploads/{upload_id}/chunks", timeout=30)
        r.raise_for_status()
        return r.json()

    def upload_status(self, upload_id: str) -> dict[str, Any]:
        r = requests.get(f"{self.base_url}/uploads/{upload_id}/status", timeout=30)
        r.raise_for_status()
        return r.json()

    def export_video(self, upload_id: str) -> dict[str, Any]:
        r = requests.post(f"{self.base_url}/uploads/{upload_id}/export", timeout=3600)
        if not r.ok:
            detail = r.text
            try:
                detail = r.json().get("detail", detail)
            except Exception:
                pass
            raise requests.HTTPError(f"Export failed ({r.status_code}): {detail}", response=r)
        return r.json()

    def export_download_url(self, upload_id: str) -> str:
        return f"{self.base_url}/uploads/{upload_id}/export/download"

    def export_dataset(self, upload_id: str, kind: str) -> dict[str, Any]:
        """kind: coco | yolo | bbox-zip"""
        r = requests.post(
            f"{self.base_url}/uploads/{upload_id}/export/{kind}",
            timeout=3600,
        )
        if not r.ok:
            detail = r.text
            try:
                detail = r.json().get("detail", detail)
            except Exception:
                pass
            raise requests.HTTPError(
                f"Dataset export ({kind}) failed ({r.status_code}): {detail}",
                response=r,
            )
        return r.json()

    def dataset_download_url(self, upload_id: str, kind: str) -> str:
        return f"{self.base_url}/uploads/{upload_id}/export/{kind}/download"

    def prepare_chunk(self, upload_id: str, chunk_index: int) -> dict[str, Any]:
        r = requests.post(
            f"{self.base_url}/uploads/{upload_id}/chunks/prepare",
            json={"chunk_index": chunk_index},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()

    def frame_url(self, upload_id: str, chunk_index: int, frame_idx: int) -> str:
        return (
            f"{self.base_url}/uploads/{upload_id}/chunks/{chunk_index}/frames/{frame_idx:06d}.jpg"
        )

    def resample(self, upload_id: str, chunk_index: int, exclude: list[int] | None = None) -> int:
        r = requests.post(
            f"{self.base_url}/uploads/{upload_id}/chunks/{chunk_index}/sample-frame",
            json={"exclude": exclude or []},
            timeout=30,
        )
        r.raise_for_status()
        return int(r.json()["sample_frame"])

    def add_points(
        self,
        session_id: str,
        frame_idx: int,
        points: list[dict[str, int]],
        obj_id: int = 1,
        replace: bool = False,
    ) -> None:
        r = requests.post(
            f"{self.base_url}/sessions/{session_id}/prompt/points",
            json={
                "frame_idx": frame_idx,
                "obj_id": obj_id,
                "points": points,
                "replace": replace,
            },
            timeout=30,
        )
        r.raise_for_status()

    def refine(self, session_id: str, frame_idx: int, obj_id: int = 1) -> dict[str, Any]:
        r = requests.post(
            f"{self.base_url}/sessions/{session_id}/refine",
            json={"frame_idx": frame_idx, "obj_id": obj_id, "mode": "points"},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()

    def propagate(self, session_id: str) -> str:
        r = requests.post(
            f"{self.base_url}/sessions/{session_id}/propagate",
            json={"mode": "points", "direction": "both"},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["job_id"]

    def stream_job(self, job_id: str) -> Iterator[dict[str, Any]]:
        url = f"{self.base_url}/jobs/{job_id}/stream"
        with requests.get(url, stream=True, timeout=(30, self.timeout)) as r:
            r.raise_for_status()
            buf = ""
            for chunk in r.iter_content(decode_unicode=True):
                if not chunk:
                    continue
                buf += chunk
                while "\n\n" in buf:
                    block, buf = buf.split("\n\n", 1)
                    for line in block.splitlines():
                        if line.startswith("data: "):
                            yield json.loads(line[6:])
