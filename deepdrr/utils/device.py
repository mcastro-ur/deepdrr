import os
import torch


def get_device() -> torch.device:
    force_cpu = os.environ.get("DEEPDRR_FORCE_CPU", "0") == "1"
    if (not force_cpu) and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_device(x, device: torch.device | None = None):
    if device is None:
        device = get_device()
    if hasattr(x, "to"):
        return x.to(device, non_blocking=True)
    return x
