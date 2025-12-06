import os
import cv2
import numpy as np


def save_masks(video_segments, filename_map, output_folder):
    """Save per-frame segmentation masks with original frame filenames."""
    os.makedirs(output_folder, exist_ok=True)

    for frame_idx, objects in video_segments.items():
        temp_name = f"{frame_idx}.jpg"
        original_name = filename_map.get(temp_name)
        if not original_name:
            continue

        base, _ = os.path.splitext(original_name)

        for obj_id, mask in objects.items():
            if mask is None:
                continue

            if mask.ndim == 3 and mask.shape[0] == 1:
                mask = mask.squeeze(0)

            mask_bin = (mask > 0.0).astype(np.uint8) * 255
            out_path = os.path.join(output_folder, f"{base}_obj{obj_id}_mask.png")

            cv2.imwrite(out_path, mask_bin)
