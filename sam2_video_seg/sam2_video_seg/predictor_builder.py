import os
import torch
from sam2.build_sam import build_sam2_video_predictor


def build_predictor(model_cfg, checkpoint, device="cuda"):
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    if not os.path.exists(model_cfg):
        raise FileNotFoundError(model_cfg)

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(checkpoint)

    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device=device)
    return predictor
