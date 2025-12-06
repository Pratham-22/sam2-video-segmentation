import cv2
import numpy as np
import shutil

from .point_extraction import get_colored_points
from .frame_utils import create_temp_folder_with_numbered_frames
from .predictor_builder import build_predictor
from .mask_saver import save_masks


def run_segmentation(
    image_folder,
    reference_image_path,
    model_cfg,
    checkpoint,
    output_folder,
    device="cuda",
    cleanup_temp=False,
):
    predictor = build_predictor(model_cfg, checkpoint, device=device)

    ref_bgr = cv2.imread(reference_image_path)
    if ref_bgr is None:
        raise IOError("Reference image not found")

    ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
    points, labels = get_colored_points(ref_rgb)

    temp_folder, filename_map = create_temp_folder_with_numbered_frames(image_folder)
    state = predictor.init_state(video_path=temp_folder)

    frame_idx = 0
    obj_id = 1

    predictor.add_new_points_or_box(
        inference_state=state,
        frame_idx=frame_idx,
        obj_id=obj_id,
        points=points,
        labels=labels,
    )

    video_segments = {}
    for idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
        objects = {}
        for i, oid in enumerate(obj_ids):
            mask = (mask_logits[i] > 0.0).cpu().numpy()
            objects[oid] = mask
        video_segments[idx] = objects

    save_masks(video_segments, filename_map, output_folder)

    if cleanup_temp:
        shutil.rmtree(temp_folder, ignore_errors=True)
