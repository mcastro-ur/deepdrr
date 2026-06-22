# DeepDRR on WSL2 Ubuntu 22.04 + NVIDIA Blackwell (RTX Pro)

Ce guide donne une installation robuste pour exécuter DeepDRR sur WSL2 Ubuntu 22.04 avec un GPU NVIDIA récent (incluant Blackwell), tout en conservant un fallback CPU propre.

## 1) Pré-requis

- Windows avec driver NVIDIA compatible WSL GPU.
- WSL2 Ubuntu 22.04.
- Vérification GPU dans WSL:

```bash
nvidia-smi
```

Si `nvidia-smi` ne fonctionne pas dans WSL, corriger d'abord la partie driver Windows/WSL.

## 2) Installation rapide

Depuis la racine du repo:

```bash
chmod +x scripts/install_wsl_blackwell.sh
./scripts/install_wsl_blackwell.sh
```

Le script:
- installe les dépendances système,
- crée un venv Python 3.10,
- installe PyTorch CUDA 12.4,
- installe DeepDRR en editable,
- exécute un smoke test CUDA.

## 3) Compatibilité device dans le code

Deux utilitaires sont ajoutés:

- `deepdrr/utils/device.py`
  - `get_device()` retourne `cuda` si disponible, sinon `cpu`.
  - `to_device(x)` envoie un objet vers le device actif.

- `deepdrr/utils/cuda_fallback.py`
  - `run_with_cuda_fallback(cuda_fn, cpu_fn, ...)` exécute le chemin CPU si un kernel CUDA custom n'est pas compatible (`no kernel image`, `invalid device function`).

## 4) Bonnes pratiques de migration

Remplacer les usages en dur de `.cuda()` par `.to(device)`:

```python
from deepdrr.utils.device import get_device

device = get_device()
model = model.to(device)
x = x.to(device, non_blocking=True)
```

Pour forcer le CPU:

```bash
DEEPDRR_FORCE_CPU=1 python your_script.py
```

## 5) Validation

- Vérifier import:

```bash
python -c "import deepdrr; print('ok')"
```

- Vérifier CUDA:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

- Exécuter un rendu DRR minimal et comparer chemin GPU vs CPU.
