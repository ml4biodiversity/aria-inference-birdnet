"""Command-line interface for ``aria-inference-birdnet``.

Usage examples::

    # Download model files
    aria-inference-birdnet download-models --dir ./models

    # Run detection with aviary config (recommended for BioDCASE)
    aria-inference-birdnet detect \\
        --input aviary_1/ \\
        --output predictions_aviary_1.csv \\
        --model-dir ./models \\
        --aviary-config aviary_config.json \\
        --aviary aviary_1

    # List available aviaries
    aria-inference-birdnet list-aviaries --aviary-config aviary_config.json

    # Run detection with flat species file (alternative)
    aria-inference-birdnet detect \\
        --input /path/to/audio \\
        --output predictions.csv \\
        --model-dir ./models \\
        --allowed-species-file allowed_species.txt
"""

from __future__ import annotations

from pathlib import Path

import click


@click.group()
@click.version_option(package_name="aria-inference-birdnet")
def main():
    """ARIA BirdNET-only inference."""


# ── download-models ──────────────────────────────────────────────────

@main.command("download-models")
@click.option(
    "--dir", "model_dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Local directory to store model files.",
)
@click.option(
    "--skip-optional", is_flag=True, default=False,
    help="Skip the evaluation CSV (disables per-class thresholds).",
)
def download_models(model_dir: Path, skip_optional: bool):
    """Download BirdNET model assets from GitHub Releases."""
    from .model_store import ensure_birdnet_assets

    ensure_birdnet_assets(
        model_dir,
        download=True,
        include_optional=not skip_optional,
    )
    click.echo(f"BirdNET assets ready in {model_dir}")


# ── list-aviaries ────────────────────────────────────────────────────

@main.command("list-aviaries")
@click.option(
    "--aviary-config",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to aviary_config.json.",
)
def list_aviaries_cmd(aviary_config: Path):
    """List available aviary IDs from the config."""
    from .species_filter import list_aviaries, load_aviary_species

    ids = list_aviaries(aviary_config)
    click.echo(f"Available aviaries ({len(ids)}):\n")
    for aviary_id in ids:
        species = load_aviary_species(aviary_config, aviary_id)
        # Subtract wild birds to show aviary-specific count
        import json
        with open(aviary_config, "r") as f:
            n_wild = len(json.load(f).get("wild_birds", []))
        n_aviary = len(species) - n_wild
        click.echo(f"  {aviary_id}  ({n_aviary} aviary species + {n_wild} wild birds)")


# ── detect ───────────────────────────────────────────────────────────

@main.command("detect")
@click.option(
    "--input", "input_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Audio file or directory of audio files.",
)
@click.option(
    "--output", "output_csv",
    type=click.Path(path_type=Path),
    required=True,
    help="Path for the final predictions CSV.",
)
@click.option(
    "--model-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory containing ZooCustom_v1.tflite and labels.",
)
@click.option(
    "--aviary-config",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to aviary_config.json for per-aviary species filtering.",
)
@click.option(
    "--aviary",
    "aviary_id",
    type=str,
    default=None,
    help="Aviary ID to process (e.g. aviary_1). Requires --aviary-config.",
)
@click.option(
    "--allowed-species-file",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Flat species whitelist file (alternative to --aviary-config).",
)
@click.option(
    "--min-confidence",
    type=float, default=0.05, show_default=True,
    help="Global fallback confidence threshold.",
)
@click.option(
    "--temperature",
    type=float, default=1.8, show_default=True,
    help="Temperature for confidence scaling (T>1 reduces saturation). "
         "Set to 1.0 to disable.",
)
@click.option(
    "--overlap",
    type=float, default=0.0, show_default=True,
    help="Overlap in seconds between BirdNET analysis windows.",
)
@click.option(
    "--threads",
    type=int, default=4, show_default=True,
    help="CPU threads for BirdNET.",
)
@click.option(
    "--topk",
    type=int, default=3, show_default=True,
    help="Keep at most K species per detection window (0=disable).",
)
@click.option(
    "--eval-csv",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to *_evaluation.csv for per-class thresholds. "
         "Auto-detected from model-dir when omitted.",
)
def detect(
    input_path: Path,
    output_csv: Path,
    model_dir: Path,
    aviary_config: Path | None,
    aviary_id: str | None,
    allowed_species_file: Path | None,
    min_confidence: float,
    temperature: float,
    overlap: float,
    threads: int,
    topk: int,
    eval_csv: Path | None,
):
    """Run BirdNET inference on audio files."""
    from .detector import run_detection

    run_detection(
        input_path=input_path,
        output_csv=output_csv,
        model_dir=model_dir,
        aviary_config=aviary_config,
        aviary_id=aviary_id,
        allowed_species_file=allowed_species_file,
        min_confidence=min_confidence,
        temperature=temperature if temperature != 1.0 else None,
        overlap=overlap,
        threads=threads,
        top_k=topk,
        eval_csv=eval_csv,
    )
