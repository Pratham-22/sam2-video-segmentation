import argparse
import sys
from sam2_video_seg.segmentation_engine import run_segmentation


def parse_args():
    p = argparse.ArgumentParser(description="SAM2 Video Segmentation with Point Annotations")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--ref-image", required=True)
    p.add_argument("--model-cfg", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--cleanup-temp", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        run_segmentation(
            image_folder=args.input_dir,
            reference_image_path=args.ref_image,
            model_cfg=args.model_cfg,
            checkpoint=args.checkpoint,
            output_folder=args.output_dir,
            device=args.device,
            cleanup_temp=args.cleanup_temp,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
