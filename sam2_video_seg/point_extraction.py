import cv2
import numpy as np


def get_colored_points(image_rgb):
    """Extract red (1) and green (0) points from an RGB reference image."""
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    lower_green = np.array([40, 50, 50])
    upper_green = np.array([90, 255, 255])

    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    def extract_points(mask):
        points = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            m = cv2.moments(contour)
            if m["m00"] == 0:
                continue
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            points.append([cx, cy])
        return points

    red_pts = extract_points(red_mask)
    green_pts = extract_points(green_mask)

    if not red_pts and not green_pts:
        raise ValueError("No red or green points detected.")

    points = np.array(red_pts + green_pts, dtype=np.float32)
    labels = np.array([1] * len(red_pts) + [0] * len(green_pts), dtype=np.int32)

    return points, labels
