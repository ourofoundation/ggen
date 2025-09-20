import os
import sys
from pathlib import Path

from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator

# Default models path - can be overridden via environment variable
MODELS_PATH = Path(os.environ.get("ORB_MODELS_PATH", Path.home() / ".orb_models"))


def get_orb_calculator(
    device: str = "cpu",
    weights_path: str = None,
):
    """Return an instance of the ORB calculator."""
    try:
        if weights_path:
            print(f"Loading ORB calculator from {weights_path}")
            print(f"Using device: {device}")

            # Check if weights file exists
            weights_file = Path(weights_path)
            if not weights_file.exists():
                raise FileNotFoundError(f"Model weights not found at {weights_path}")

            # Load the model
            orbff = pretrained.orb_v3_conservative_inf_mpa(
                device=device, weights_path=weights_path, compile=False
            )
        else:
            orbff = pretrained.orb_v3_conservative_inf_mpa(device=device, compile=False)
        calculator = ORBCalculator(orbff, device=device)
        return calculator

    except Exception as e:
        print(f"Error loading ORB calculator: {str(e)}")
        raise RuntimeError(f"Failed to load ORB calculator: {str(e)}") from e
