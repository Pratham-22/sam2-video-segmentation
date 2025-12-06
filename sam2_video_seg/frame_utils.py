import os
import re
import cv2


def create_temp_folder_with_numbered_frames(image_folder):
    """Copy frames into temp_frames/ as 0.jpg, 1.jpg, ... and return a filename map."""
    temp_folder = os.path.join(os.path.dirname(image_folder), "temp_frames")
    os.makedirs(temp_folder, exist_ok=True)

    frame_files = [
        f for f in os.listdir(image_folder)
        if os.path.splitext(f)[1].lower() in [".jpg", ".jpeg", ".png"]
    ]
    if not frame_files:
        raise ValueError(f"No frames found in {image_folder}")

    frame_numbers = {}
    for f in frame_files:
        m = re.search(r"frame(\d+)", f)
        frame_numbers[f] = int(m.group(1)) if m else 0

    sorted_frames = sorted(frame_files, key=lambda x: frame_numbers[x])

    filename_map = {}
    for i, orig_name in enumerate(sorted_frames):
        new_name = f"{i}.jpg"
        filename_map[new_name] = orig_name

        img = cv2.imread(os.path.join(image_folder, orig_name))
        cv2.imwrite(os.path.join(temp_folder, new_name), img)

    map_file = os.path.join(temp_folder, "filename_map.txt")
    with open(map_file, "w") as f:
        for new_name, orig in filename_map.items():
            f.write(f"{new_name},{orig}\n")

    return temp_folder, filename_map
