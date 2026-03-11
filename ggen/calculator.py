import ctypes
import gc
import logging
import os
import resource
import sys
from pathlib import Path

import torch
from orb_models.forcefield import pretrained
try:
    from orb_models.forcefield.inference.calculator import ORBCalculator
except ImportError:
    from orb_models.forcefield.calculator import ORBCalculator

# Default models path - can be overridden via environment variable
MODELS_PATH = Path(os.environ.get("ORB_MODELS_PATH", Path.home() / ".orb_models"))

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

# Enable torch.compile by default for performance. Set GGEN_TORCH_COMPILE=0
# to force eager mode when diagnosing memory/perf regressions.
TORCH_COMPILE = os.environ.get("GGEN_TORCH_COMPILE", "1") != "0"
ORB_MODEL = os.environ.get("GGEN_ORB_MODEL", "orb_v3_conservative_inf_mpa")
ORB_PRECISION = os.environ.get("GGEN_ORB_PRECISION", "float32-high")
ORB_EDGE_METHOD = os.environ.get("GGEN_ORB_EDGE_METHOD", "knn_alchemi")

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


def extract_orb_model(calculator):
    """Return the raw ORB model stored on an ASE calculator."""
    if hasattr(calculator, "orbff"):
        return calculator.orbff
    if hasattr(calculator, "model"):
        return calculator.model
    raise AttributeError(
        "Cannot extract ORB model from calculator. Expected 'orbff' or 'model'."
    )


def build_orb_torchsim_model(calculator):
    """Build a TorchSim-compatible ORB wrapper from the configured calculator."""
    raw_model = extract_orb_model(calculator)
    atoms_adapter = getattr(calculator, "atoms_adapter", None)
    edge_method = getattr(calculator, "edge_method", ORB_EDGE_METHOD)

    # Prefer ORB's native TorchSim wrapper when available because it accepts the
    # atoms adapter and accelerated edge construction options used by orb-models >= 0.6.
    if atoms_adapter is not None:
        try:
            from orb_models.forcefield.inference.orb_torchsim import OrbTorchSimModel

            return OrbTorchSimModel(
                raw_model,
                atoms_adapter,
                edge_method=edge_method,
            )
        except ImportError:
            logger.debug(
                "Falling back to legacy torch-sim ORB wrapper; native ORB TorchSim wrapper unavailable."
            )

    from torch_sim.models.orb import OrbModel as TorchSimOrbModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return TorchSimOrbModel(
        model=raw_model,
        compute_stress=True,
        compute_forces=True,
        device=device,
    )


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
        model_loader = getattr(pretrained, ORB_MODEL)
        if weights_path:
            logger.info(
                "Loading ORB calculator from %s (%s, model=%s, edge_method=%s)",
                weights_path,
                compile_mode,
                ORB_MODEL,
                ORB_EDGE_METHOD,
            )
            logger.info("Using device: %s", device)

            weights_file = Path(weights_path)
            if not weights_file.exists():
                raise FileNotFoundError(f"Model weights not found at {weights_path}")

            loaded = model_loader(
                device=device,
                weights_path=weights_path,
                compile=TORCH_COMPILE,
                precision=ORB_PRECISION,
            )
        else:
            logger.info(
                "Loading ORB calculator (%s, device=%s, model=%s, edge_method=%s)",
                compile_mode,
                device,
                ORB_MODEL,
                ORB_EDGE_METHOD,
            )
            loaded = model_loader(
                device=device,
                compile=TORCH_COMPILE,
                precision=ORB_PRECISION,
            )

        if isinstance(loaded, tuple):
            orbff, atoms_adapter = loaded
        else:
            orbff, atoms_adapter = loaded, None

        calculator_kwargs = {"device": device}
        if atoms_adapter is not None:
            calculator_kwargs["atoms_adapter"] = atoms_adapter
        if ORB_EDGE_METHOD:
            calculator_kwargs["edge_method"] = ORB_EDGE_METHOD

        calculator = ORBCalculator(orbff, **calculator_kwargs)
        calculator.orbff = orbff
        calculator.atoms_adapter = atoms_adapter
        calculator.edge_method = ORB_EDGE_METHOD
        calculator.orb_model_name = ORB_MODEL
        calculator.orb_precision = ORB_PRECISION
        logger.info("RSS after model load: %.0f MiB", rss_mb())
        return calculator

    except Exception as e:
        logger.error("Error loading ORB calculator: %s", e)
        raise RuntimeError(f"Failed to load ORB calculator: {str(e)}") from e
