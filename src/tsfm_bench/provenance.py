"""Record how a result file was produced.

Results can differ across CUDA and pandas builds, so every driver writes a sidecar
manifest next to its outputs recording the environment that produced them.
"""
from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .paths import RESULTS

_TRACKED = ("torch", "numpy", "pandas", "scipy", "gluonts", "uni2ts",
            "chronos-forecasting", "timesfm", "statsforecast",
            "hierarchicalforecast", "datasetsforecast")


def snapshot() -> dict:
    """Environment fingerprint: versions, GPU, and the uv lock hash."""
    pkgs = {}
    for name in _TRACKED:
        try:
            pkgs[name] = version(name)
        except PackageNotFoundError:
            pkgs[name] = None

    gpu = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu = {"name": torch.cuda.get_device_name(0),
                   "capability": list(torch.cuda.get_device_capability()),
                   "torch_cuda": torch.version.cuda,
                   "arch_list": torch.cuda.get_arch_list()}
    except Exception:
        pass

    lock = RESULTS.parent / "uv.lock"
    lock_hash = None
    if lock.exists():
        try:
            lock_hash = subprocess.run(["sha256sum", str(lock)], capture_output=True,
                                       text=True, timeout=30).stdout.split()[0][:16]
        except Exception:
            pass

    return {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python": platform.python_version(), "host": platform.node(),
            "packages": pkgs, "gpu": gpu, "uv_lock_sha256_prefix": lock_hash}


def record(tag: str, outputs: list[str] | None = None, **params) -> Path:
    """Write results/_provenance/<tag>.json describing one experiment run."""
    d = RESULTS / "_provenance"
    d.mkdir(parents=True, exist_ok=True)
    payload = {"tag": tag, "params": params, "outputs": outputs or [],
               "environment": snapshot()}
    out = d / f"{tag}.json"
    out.write_text(json.dumps(payload, indent=2, default=str))
    return out
