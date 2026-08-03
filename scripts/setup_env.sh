#!/usr/bin/env bash
# One-time setup: create a local venv with torch + transformers 5 + SAM3 + labeler deps.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV_DIR:-${ROOT}/.venv}"
SAM3_REPO="${SAM3_REPO:-${ROOT}/third_party/sam3}"

echo "=== SAM3 Video Labeler — environment setup ==="
echo "Root      : ${ROOT}"
echo "Venv      : ${VENV}"
echo "SAM3 repo : ${SAM3_REPO}"

if command -v module &>/dev/null; then
  module load python/3.12 2>/dev/null || module load python/3.10 2>/dev/null || true
fi

pick_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    echo "${PYTHON_BIN}"
    return
  fi
  for candidate in python3.12 python3.13 python3 python; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      version="$("${candidate}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
      major="${version%%.*}"
      minor="${version#*.}"
      if (( major > 3 || (major == 3 && minor >= 10) )); then
        echo "${candidate}"
        return
      fi
    fi
  done
  echo ""
}

PYTHON_BIN="$(pick_python)"
if [[ -z "${PYTHON_BIN}" ]]; then
  echo "Python 3.10+ not found."
  echo "On OSC: module load python/3.12"
  exit 1
fi
echo "Python    : ${PYTHON_BIN} ($(${PYTHON_BIN} --version))"

if [[ ! -x "${VENV}/bin/python" ]]; then
  echo "Creating venv …"
  "${PYTHON_BIN}" -m venv "${VENV}"
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
pip install --upgrade pip

# Prefer CUDA torch when available; CPU wheel still works for mock UI testing.
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
  echo "Installing PyTorch (CUDA) …"
  pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu128 \
    || pip install -q torch torchvision
else
  echo "Installing PyTorch (CPU / default) …"
  pip install -q torch torchvision
fi

mkdir -p "$(dirname "${SAM3_REPO}")"
if [[ ! -d "${SAM3_REPO}/.git" ]]; then
  echo "Cloning Meta SAM3 …"
  git clone --depth 1 https://github.com/facebookresearch/sam3.git "${SAM3_REPO}"
else
  echo "Updating Meta SAM3 …"
  git -C "${SAM3_REPO}" pull --ff-only || true
fi

echo "Installing Meta SAM3 (editable) …"
pip install -q -e "${SAM3_REPO}"
pip install -q pycocotools || true

pip install -q -U "transformers>=5.0.0" "accelerate>=1.0.0"
pip install -q -r "${ROOT}/sam3_video_service/requirements.txt"
pip install -q -r "${ROOT}/labeler_ui/requirements.txt"

python -c "
import torch
print('CUDA:', torch.cuda.is_available())
try:
    from sam3.model_builder import build_sam3_predictor
    print('OK Meta SAM3 import')
except Exception as exc:
    print('WARN Meta SAM3 import failed:', exc)
"

echo ""
echo "=== Hugging Face weights ==="
echo "1. Request access: https://huggingface.co/facebook/sam3"
echo "2. Create a token: https://huggingface.co/settings/tokens"
echo "3. Login:"
echo "   source ${VENV}/bin/activate"
echo "   huggingface-cli login"
echo "   # or: export HF_TOKEN=hf_..."
echo ""
echo "Setup complete. Run the labeler:"
echo "  cd ${ROOT} && ./scripts/run_local.sh"
