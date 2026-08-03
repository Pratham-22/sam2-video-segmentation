# SAM3 Video Labeler

Browser-based welding / industrial video labeling:

**upload → chunk → click points → preview mask → track → export MP4**

Built around Meta **SAM3** video tracking.

## Layout

```text
sam3-video-labeler/
  sam3_video_service/   # GPU API (port 2129)
  labeler_ui/           # Wizard UI (port 8080)
  scripts/
    setup_env.sh        # create venv + install deps + clone Meta SAM3
    run_local.sh        # start API + UI
  data/                 # uploads + masks (gitignored)
  third_party/sam3/     # created by setup_env.sh (gitignored)
```

## Requirements

- Python **3.10+** (3.12 recommended)
- **ffmpeg** / `ffprobe` (video probe + export)
- NVIDIA GPU + CUDA for real SAM3 tracking (optional — mock mode works on CPU)
- Hugging Face access to [`facebook/sam3`](https://huggingface.co/facebook/sam3)

## One-time setup

```bash
git clone https://github.com/OSU-SAI-Lab/Welding_sam3.git
cd Welding_sam3
chmod +x scripts/*.sh
./scripts/setup_env.sh
```

Then request model access and log in:

```bash
source .venv/bin/activate
huggingface-cli login
# or: export HF_TOKEN=hf_...
```

## Run

```bash
# Optional: put large uploads on project storage
export DATA_ROOT=/path/to/large/disk/sam3-video-labeler-data
mkdir -p "$DATA_ROOT"

./scripts/run_local.sh
```

- **UI:** http://127.0.0.1:8080  
- **API health:** http://127.0.0.1:2129/health  

### SSH tunnel (cluster / remote GPU)

```bash
ssh -L 8080:localhost:8080 -L 2129:localhost:2129 user@gpu-host
```

Then open http://127.0.0.1:8080 on your laptop.

### OSC notes

```bash
# inside an interactive GPU allocation
module load python/3.12
module load ffmpeg/6.1.1
export DATA_ROOT=/fs/ess/PAS2699/$USER/sam3-video-labeler-data
mkdir -p "$DATA_ROOT"
./scripts/run_local.sh
```

## UI features

- Step-by-step wizard with time estimates
- Overall progress bar across chunks
- Drag-and-drop upload
- Positive / negative clicks, undo, delete points
- Preview mask on one frame before tracking
- Track chunk with elapsed + remaining time
- Export annotated MP4 when all chunks are done

## Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `DATA_ROOT` | `./data` | Uploads, masks, exports |
| `SAM3_MOCK` | auto (`0` on GPU, `1` otherwise) | `1` = fake masks for UI testing |
| `HF_TOKEN` | — | Hugging Face token for SAM3 weights |
| `SAM3_REPO` | `./third_party/sam3` | Path to Meta SAM3 checkout |
| `SAM3_BACKEND` | `transformers` | `transformers` or `meta` |
| `SAM3_VERSION` | `sam3` | `sam3` or `sam3.1` |
| `CHUNK_SIZE` | `1000` | Frames per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `VENV_DIR` | `./.venv` | Override venv location for `setup_env.sh` |

## API (`sam3_video_service`)

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/uploads` | Upload video |
| GET | `/uploads/{id}/status` | Chunk progress |
| POST | `/uploads/{id}/chunks/prepare` | Extract frames + session |
| POST | `/sessions/{id}/prompt/points` | Positive / negative clicks |
| POST | `/sessions/{id}/refine` | Preview one frame |
| POST | `/sessions/{id}/propagate` | Track chunk (SSE job) |
| POST | `/uploads/{id}/export` | Build annotated MP4 |
| GET | `/uploads/{id}/export/download` | Download MP4 |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Disk quota exceeded on upload | Set `DATA_ROOT` to a large disk |
| `ffprobe not found` | Install ffmpeg / `module load ffmpeg` |
| Stuck loading SAM3 | First track loads weights (~1–3 min) |
| Mock masks only | GPU available? Set `SAM3_MOCK=0` and log in to Hugging Face |
| `No venv found` | Run `./scripts/setup_env.sh` |

## License / third-party

- This UI/service code: use / share as needed for your project.
- Meta SAM3 (`facebookresearch/sam3`) and Hugging Face weights have their own licenses and access terms — request access before using real tracking.
