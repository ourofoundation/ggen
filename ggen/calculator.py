import ctypes
import gc
import logging
import os
import resource
import sys
from pathlib import Path

import torch
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator

# Default models path - can be overridden via environment variable
MODELS_PATH = Path(os.environ.get("ORB_MODELS_PATH", Path.home() / ".orb_models"))

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

# Enable torch.compile by default for performance. Set GGEN_TORCH_COMPILE=0
# to force eager mode when diagnosing memory/perf regressions.
TORCH_COMPILE = os.environ.get("GGEN_TORCH_COMPILE", "1") != "0"

# Soft RSS ceiling (MiB).  When exceeded, attempt glibc malloc_trim to
# return freed pages to the OS.  Override with GGEN_RSS_LIMIT_MB.
RSS_LIMIT_MB = int(os.environ.get("GGEN_RSS_LIMIT_MB", 20_000))


def rss_mb() -> float:
    """Current committed memory (RSS + swap) in MiB.

    Reads from /proc/self/status for accuracy.  Falls back to
    ru_maxrss (peak RSS only) if /proc is unavailable.
    """
    try:
        with open("/proc/self/status") as f:
            rss = swap = 0
            for line in f:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1])  # kB
                elif line.startswith("VmSwap:"):
                    swap = int(line.split()[1])
            return (rss + swap) / 1024
    except (OSError, ValueError):
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def reset_dynamo_cache() -> None:
    """Clear torch.compile (dynamo + inductor) caches and reclaim memory."""
    try:
        torch._dynamo.reset()
    except AttributeError:
        pass
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _detect_device() -> str:
    """Detect the best available device (CUDA if available, otherwise CPU)."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def get_orb_calculator(
    device: str = None,
    weights_path: str = None,
):
    """Return an instance of the ORB calculator.

    Args:
        device: Device to run the calculator on ('cpu' or 'cuda').
                If None, automatically detects CUDA availability and uses it if available.
        weights_path: Optional path to model weights file.

    Returns:
        ORBCalculator instance configured for the specified device.
    """
    # Auto-detect device if not specified
    if device is None:
        device = _detect_device()

    try:
        compile_mode = "compile" if TORCH_COMPILE else "eager"
        logger.info("RSS before model load: %.0f MiB", rss_mb())
        if weights_path:
            logger.info("Loading ORB calculator from %s (%s)", weights_path, compile_mode)
            logger.info("Using device: %s", device)

            weights_file = Path(weights_path)
            if not weights_file.exists():
                raise FileNotFoundError(f"Model weights not found at {weights_path}")

            orbff = pretrained.orb_v3_conservative_inf_mpa(
                device=device,
                weights_path=weights_path,
                compile=TORCH_COMPILE,
                precision="float32-high",
            )
        else:
            logger.info("Loading ORB calculator (%s, device: %s)", compile_mode, device)
            orbff = pretrained.orb_v3_conservative_inf_mpa(
                device=device, compile=TORCH_COMPILE, precision="float32-high"
            )
        calculator = ORBCalculator(orbff, device=device)
        logger.info("RSS after model load: %.0f MiB", rss_mb())
        return calculator

    except Exception as e:
        logger.error("Error loading ORB calculator: %s", e)
        raise RuntimeError(f"Failed to load ORB calculator: {str(e)}") from e
