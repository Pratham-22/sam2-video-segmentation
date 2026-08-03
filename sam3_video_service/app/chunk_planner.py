"""Compute chunk ranges with overlap (process vs save windows)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkPlan:
    chunk_index: int
    process_start: int  # inclusive, 0-based frame index
    process_end: int  # inclusive
    save_start: int  # inclusive — frames written to final export
    save_end: int  # inclusive


def plan_chunks(
    frame_count: int,
    chunk_size: int,
    overlap: int,
) -> list[ChunkPlan]:
    if frame_count <= 0:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")

    plans: list[ChunkPlan] = []
    chunk_index = 0
    save_start = 0

    while save_start < frame_count:
        save_end = min(save_start + chunk_size - 1, frame_count - 1)
        process_start = max(0, save_start - overlap) if chunk_index > 0 else 0
        process_end = min(frame_count - 1, save_end + overlap)

        plans.append(
            ChunkPlan(
                chunk_index=chunk_index,
                process_start=process_start,
                process_end=process_end,
                save_start=save_start,
                save_end=save_end,
            )
        )
        chunk_index += 1
        save_start = save_end + 1

    return plans
