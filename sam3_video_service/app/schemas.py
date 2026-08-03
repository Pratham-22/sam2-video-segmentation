"""Request/response models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class PointPrompt(BaseModel):
    x: int
    y: int
    label: int = Field(..., description="1=positive, 0=negative")

    @field_validator("label")
    @classmethod
    def validate_label(cls, v: int) -> int:
        if v not in (0, 1):
            raise ValueError("label must be 0 (negative) or 1 (positive)")
        return v


class ChunkInfo(BaseModel):
    chunk_index: int
    process_start: int
    process_end: int
    save_start: int
    save_end: int


class PrepareChunkRequest(BaseModel):
    chunk_index: int = Field(..., ge=0)


class FinalizeUploadRequest(BaseModel):
    original_filename: str = Field(..., min_length=1)


class PointsRequest(BaseModel):
    frame_idx: int = Field(..., ge=0)
    obj_id: int = Field(1, ge=1)
    points: list[PointPrompt] = Field(..., min_length=1)
    replace: bool = False


class TextPromptRequest(BaseModel):
    text: str = Field(..., min_length=1)
    frame_idx: int = Field(0, ge=0)


class RefineRequest(BaseModel):
    frame_idx: int = Field(..., ge=0)
    obj_id: int = Field(1, ge=1)
    mode: Literal["points", "text"] = "points"


class PropagateRequest(BaseModel):
    mode: Literal["points", "text"] = "points"
    direction: Literal["forward", "backward", "both"] = "both"


class FrameMaskResult(BaseModel):
    frame_idx: int
    obj_id: int
    bbox: list[int]
    confidence: float
    segmentation: list[list[int]] = Field(default_factory=list)


class JobProgressEvent(BaseModel):
    type: str
    frame_idx: int | None = None
    total: int | None = None
    message: str | None = None
    result: FrameMaskResult | None = None
