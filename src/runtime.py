"""GPU isolation and validation without importing PyTorch during bootstrap."""
import os
import shutil
import subprocess
import sys


def required_download_bytes(budgets: list[int], cached: list[int], *, purge: bool) -> int:
    """Conservative additional disk budget, crediting existing sequential caches."""
    needed = peak = reclaimed = 0
    for budget, size in zip(budgets, cached, strict=True):
        size = max(0, size)
        if purge:
            peak = max(peak, budget - size - reclaimed)
            reclaimed += size
        else:
            needed += max(0, budget - size)
    return max(0, peak if purge else needed)


def gpu_zero_info() -> dict | None:
    executable = shutil.which("nvidia-smi")
    if not executable:
        candidate = r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
        executable = candidate if os.path.isfile(candidate) else None
    if not executable:
        return None
    try:
        result = subprocess.run([executable, "--id=0", "--query-gpu=uuid,name,memory.total,memory.free",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                check=True, timeout=15)
        uuid, name, total, free = [part.strip() for part in result.stdout.strip().split(",")]
        return {"uuid": uuid, "name": name, "total_mib": float(total), "free_mib": float(free)}
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def isolate_gpu_zero() -> str | None:
    loaded = sys.modules.get("torch")
    if loaded is not None and loaded.cuda.is_initialized():
        raise RuntimeError("Import config before initializing CUDA. Restart this Python process.")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    info = gpu_zero_info()
    os.environ["CUDA_VISIBLE_DEVICES"] = info["uuid"] if info else "0"
    os.environ.setdefault("PYTHONUTF8", "1")
    return info["uuid"] if info else None


def validate_gpu(torch_module, expected_uuid: str | None = None) -> dict:
    cuda = torch_module.cuda
    if not cuda.is_available():
        raise RuntimeError("CUDA unavailable. Install CUDA-enabled PyTorch and check the NVIDIA driver.")
    if cuda.device_count() != 1:
        raise RuntimeError("Exactly one GPU must be visible; refusing to use other server GPUs.")
    properties = cuda.get_device_properties(0)
    if "H100" not in properties.name or properties.total_memory < 70 * 1024**3:
        raise RuntimeError(f"Expected an 80 GB H100, found {properties.name}.")
    if expected_uuid and os.environ.get("CUDA_VISIBLE_DEVICES") != expected_uuid:
        raise RuntimeError("GPU mask no longer matches physical GPU 0's UUID.")
    return {"name": properties.name, "total_gib": properties.total_memory / 1024**3,
            "cuda_runtime": torch_module.version.cuda, "visible_devices": 1}
