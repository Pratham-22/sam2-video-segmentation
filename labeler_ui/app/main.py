"""Video labeler UI — upload + chunk labeling (calls sam3-video-service)."""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path

import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.sam3_client import Sam3VideoClient

logger = logging.getLogger(__name__)

SAM3_VIDEO_URL = os.environ.get("SAM3_VIDEO_URL", "http://127.0.0.1:2129").rstrip("/")
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/data")).expanduser().resolve()
STATIC_DIR = Path(__file__).resolve().parent / "static"
CHUNK_SIZE = 1024 * 1024

app = FastAPI(title="Video Labeler UI", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
client = Sam3VideoClient(SAM3_VIDEO_URL)


class PointsBody(BaseModel):
    session_id: str
    frame_idx: int
    points: list[dict[str, int]] = Field(..., min_length=1)
    obj_id: int = 1
    replace: bool = False


class RefineBody(BaseModel):
    session_id: str
    frame_idx: int
    obj_id: int = 1


class PropagateBody(BaseModel):
    session_id: str


@app.get("/health")
def health():
    try:
        sam3 = client.health()
        return {"status": "ok", "sam3_video": sam3}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


@app.get("/")
def index():
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "filename required")
    upload_id = str(uuid.uuid4())
    suffix = Path(file.filename).suffix.lower() or ".mp4"
    upload_dir = DATA_ROOT / "uploads" / upload_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / f"source{suffix}"
    logger.info("Receiving upload %s -> %s", file.filename, dest)
    try:
        with dest.open("wb") as out:
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                out.write(chunk)
        logger.info("Upload saved (%s bytes), finalizing…", dest.stat().st_size)
        meta = client.finalize_upload(upload_id, file.filename)
        return meta
    except requests.HTTPError as e:
        dest.unlink(missing_ok=True)
        status = e.response.status_code if e.response is not None else 502
        detail = str(e)
        if e.response is not None:
            try:
                detail = e.response.json().get("detail", detail)
            except Exception:
                detail = e.response.text or detail
        raise HTTPException(status, detail) from e
    except OSError as e:
        dest.unlink(missing_ok=True)
        if e.errno == 122:
            raise HTTPException(
                507,
                "Disk quota exceeded. Restart with DATA_ROOT on project storage: "
                "export DATA_ROOT=/fs/ess/PAS2699/$USER/sam3-video-labeler-data",
            ) from e
        raise HTTPException(500, f"Failed to save upload: {e}") from e
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(502, f"Upload failed: {e}") from e


@app.get("/api/uploads/{upload_id}/status")
def upload_status(upload_id: str):
    try:
        return client.upload_status(upload_id)
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else 502
        raise HTTPException(status, str(e)) from e
    except Exception as e:
        raise HTTPException(502, str(e)) from e


@app.post("/api/uploads/{upload_id}/export")
def export_video(upload_id: str):
    try:
        return client.export_video(upload_id)
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else 502
        raise HTTPException(status, str(e)) from e
    except Exception as e:
        raise HTTPException(502, str(e)) from e


@app.get("/api/uploads/{upload_id}/export/download")
def download_export(upload_id: str):
    import requests as req

    url = client.export_download_url(upload_id)
    try:
        r = req.get(url, stream=True, timeout=3600)
        r.raise_for_status()
    except Exception as e:
        raise HTTPException(502, str(e)) from e
    from fastapi.responses import StreamingResponse

    return StreamingResponse(
        r.iter_content(chunk_size=65536),
        media_type="video/mp4",
        headers={"Content-Disposition": f'attachment; filename="{upload_id}_annotated.mp4"'},
    )


@app.get("/api/uploads/{upload_id}/chunks")
def chunks(upload_id: str):
    try:
        return client.list_chunks(upload_id)
    except Exception as e:
        raise HTTPException(502, str(e)) from e


@app.post("/api/uploads/{upload_id}/chunks/{chunk_index}/prepare")
def prepare(upload_id: str, chunk_index: int):
    try:
        return client.prepare_chunk(upload_id, chunk_index)
    except Exception as e:
        raise HTTPException(502, str(e)) from e


@app.get("/api/frame/{upload_id}/{chunk_index}/{frame_idx}")
def proxy_frame(upload_id: str, chunk_index: int, frame_idx: int):
    import requests

    url = client.frame_url(upload_id, chunk_index, frame_idx)
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
    except Exception as e:
        raise HTTPException(502, str(e)) from e
    from fastapi.responses import Response

    return Response(content=r.content, media_type="image/jpeg")


@app.post("/api/points")
def points(body: PointsBody):
    try:
        client.add_points(
            body.session_id,
            body.frame_idx,
            body.points,
            obj_id=body.obj_id,
            replace=body.replace,
        )
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(502, str(e)) from e


@app.post("/api/refine")
def refine(body: RefineBody):
    try:
        return client.refine(body.session_id, body.frame_idx, body.obj_id)
    except Exception as e:
        raise HTTPException(502, str(e)) from e


@app.post("/api/propagate")
def propagate(body: PropagateBody):
    try:
        job_id = client.propagate(body.session_id)
        return {"job_id": job_id}
    except Exception as e:
        raise HTTPException(502, str(e)) from e


@app.get("/api/jobs/{job_id}/stream")
def stream_job(job_id: str):
    from fastapi.responses import StreamingResponse
    import json

    def generate():
        try:
            for ev in client.stream_job(job_id):
                yield f"data: {json.dumps(ev)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
