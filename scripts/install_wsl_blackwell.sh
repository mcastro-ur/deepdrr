#!/usr/bin/env bash
set -euo pipefail

echo "[1/8] Vérification GPU côté WSL"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERREUR: nvidia-smi introuvable dans WSL."
  echo "Installe/met à jour le driver NVIDIA sous Windows (WSL GPU support), puis réessaie."
  exit 1
fi
nvidia-smi || true

echo "[2/8] Dépendances système"
sudo apt update
sudo apt install -y \
  build-essential git cmake ninja-build pkg-config \
  python3.10 python3.10-venv python3-pip \
  libgl1 libegl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
  libx11-6 libxrandr2 libxinerama1 libxcursor1 libxi6

echo "[3/8] Variables CUDA arch (compat GPU récents)"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0+PTX}"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

echo "[4/8] Environnement virtuel Python"
if [ ! -d .venv ]; then
  python3.10 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

echo "[5/8] Installation PyTorch CUDA 12.4"
pip install --upgrade \
  torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu124

echo "[6/8] Installation deps projet"
if [ -f requirements.txt ]; then
  pip install --upgrade --upgrade-strategy only-if-needed -r requirements.txt || true
fi

echo "[7/8] Installation DeepDRR editable"
pip install -e .[cuda12x] || pip install -e .

echo "[8/8] Smoke test"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

echo "OK: installation terminée."
