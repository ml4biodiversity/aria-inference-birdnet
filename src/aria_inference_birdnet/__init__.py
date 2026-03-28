"""ARIA BirdNET-only inference package.

Quick start with aviary config::

    from aria_inference_birdnet import run_detection
    from pathlib import Path

    run_detection(
        input_path=Path("aviary_1/"),
        output_csv=Path("predictions_aviary_1.csv"),
        model_dir=Path("models/"),
        aviary_config=Path("aviary_config.json"),
        aviary_id="aviary_1",
    )
"""

__version__ = "0.1.3"

from .detector import run_detection
from .model_store import ensure_birdnet_assets
from .species_filter import load_allowed_species, load_aviary_species

__all__ = [
    "run_detection",
    "ensure_birdnet_assets",
    "load_allowed_species",
    "load_aviary_species",
]
