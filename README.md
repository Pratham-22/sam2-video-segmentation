## SAM2 Video Segmentation with Point Annotations

This repository provides a modular pipeline for performing video object segmentation using the SAM2 model. The system extracts positive (red) and negative (green) point prompts from a reference image, initializes segmentation on the first frame, and propagates masks across a full sequence of frames.
Designed for research workflows, reproducible experimentation, and integration into applied computer vision pipelines.

⸻

### Features

	•	Extracts positive (red) and negative (green) clicks from a reference annotation image
	•	Converts arbitrary image sequences into SAM2-compatible numbered frames
	•	Initializes SAM2 segmentation with point prompts
	•	Propagates the mask across all video frames using the official SAM2 video predictor
	•	Saves per-frame segmentation masks using original filenames
	•	Modular code structure for easy extension and maintenance

⸻

### Project Structure
```
sam2-video-segmentation/
│
├── sam2_video_seg/
│   ├── __init__.py
│   ├── point_extraction.py        # Detects red/green points in reference image
│   ├── frame_utils.py             # Handles frame sorting and temp renaming
│   ├── predictor_builder.py       # Loads SAM2 model and predictor
│   ├── segmentation_engine.py     # Core segmentation pipeline
│   └── mask_saver.py              # Saves propagated segmentation masks
│
├── run_segmentation.py            # Entry point CLI script
├── requirements.txt
└── README.md
```
## Installation

### 1. Clone the repository:
```
git clone https://github.com/yourusername/sam2-video-segmentation.git
cd sam2-video-segmentation
```

### 2. Install Dependencies:
```
pip install -r requirements.txt
```
### 3. Ensure SAM2 is installed correctly and accessible:
```
build_sam2_video_predictor must be importable from sam2.build_sam.
```

### Input Requirements

1. Frame Folder
   ```
   A directory containig video frames in any order.
   Supported formats: .jpeg, .jpg, .png
   ```
2. Reference Image
   ```
   An RGB image containing:
   ```
```
•	Red dots → positive prompts
•	Green dots → negative prompts
```

### Usage

#### Run the segmentation pipeline via the CLI:
```
python run_segmentation.py \
  --input-dir /path/to/frames \
  --ref-image /path/to/reference.png \
  --model-cfg /path/to/sam2_config.yaml \
  --checkpoint /path/to/sam2_checkpoint.pt \
  --output-dir /path/to/output_masks \
  --device cuda \
  --cleanup-temp
```
```
| Argument       | Description                                   |
|----------------|-----------------------------------------------|
| `--input-dir`  | Folder containing raw video frames            |
| `--ref-image`  | Reference image with red/green annotation points |
| `--model-cfg`  | Path to SAM2 YAML configuration               |
| `--checkpoint` | Path to SAM2 model checkpoint                 |
| `--output-dir` | Directory where masks will be saved           |
| `--device`     | `cuda` or `cpu`                               |
| `--cleanup-temp` | Optional: remove temp numbered frame folder |
```
### Extending the Pipeline

The modular structure allows easy extension:

	•	Replace point extraction logic
	•	Add bounding box support
	•	Add multi-object segmentation
	•	Use alternative SAM2 predictors
	•	Integrate into training or annotation workflows

Each module handles one responsibility, making the system easy to maintain.

⸻

### Requirements

	•	Python 3.8+
	•	PyTorch (GPU recommended)
	•	OpenCV
	•	NumPy
	•	SAM2 dependencies

All Python dependencies are listed in requirements.txt.

### Citation
```
@misc{ravi2024sam2segmentimages,
      title={SAM 2: Segment Anything in Images and Videos}, 
      author={Nikhila Ravi and Valentin Gabeur and Yuan-Ting Hu and Ronghang Hu and Chaitanya Ryali and Tengyu Ma and Haitham Khedr and Roman Rädle and Chloe Rolland and Laura Gustafson and Eric Mintun and Junting Pan and Kalyan Vasudev Alwala and Nicolas Carion and Chao-Yuan Wu and Ross Girshick and Piotr Dollár and Christoph Feichtenhofer},
      year={2024},
      eprint={2408.00714},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.00714}, 
}
```
