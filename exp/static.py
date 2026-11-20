"""One-off facts about the model and the machine, logged at run start."""
import platform
import subprocess
import sys

import torch


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def env_info():
    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": _version("torchvision"),
        "detectron2": _version("detectron2"),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_version": torch.version.cuda,
        "hip_version": getattr(torch.version, "hip", None),
        "open_clip_file": _file("open_clip"),
    }
    if torch.cuda.is_available():
        try:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_total_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 2)
        except Exception:
            pass
    return info


def _version(module_name):
    try:
        return __import__(module_name).__version__
    except Exception:
        return None


def _file(module_name):
    """Where a module was imported from - CAT-Seg's open_clip copy must be this repo's."""
    try:
        return __import__(module_name).__file__
    except Exception:
        return None


def rocm_smi():
    try:
        out = subprocess.run(["rocm-smi", "--showproductname"],
                             capture_output=True, text=True, timeout=20)
        return out.stdout.strip()[:2000]
    except Exception:
        return None
