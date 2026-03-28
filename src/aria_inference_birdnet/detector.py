"""Detection pipeline orchestrator.

Species filtering is controlled by one of two mechanisms:

1. **Aviary config** (recommended for BioDCASE): ``--aviary-config``
   plus ``--aviary`` look up per-aviary species from an anonymized
   JSON.  Each aviary has its own species list and wild birds are
   merged automatically.

2. **Flat allowed-species file**: ``--allowed-species-file`` applies
   the same species list to all input audio.

Pipeline stages:

1. Run ``birdnet_analyzer.analyze`` via subprocess.
2. Filter per-file CSVs to allowed species.
3. Apply temperature scaling (T=1.8 by default).
4. Merge all per-file CSVs into one.
5. Apply per-class optimal thresholds or global min-confidence.
6. Keep only top-K species per detection window.
7. Write the final output CSV.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .birdnet_runtime import run_birdnet_analyze
from .io import merge_csvs
from .model_store import ensure_birdnet_assets
from .postprocess import (
    apply_threshold_filter,
    default_eval_csv_for,
    load_optimal_thresholds,
    topk_per_window,
)
from .species_filter import load_allowed_species, load_aviary_species
from .temperature import (
    compute_saturation_metrics,
    find_optimal_temperature,
    scale_individual_csvs,
)


def _resolve_allowed_species(
    aviary_config: Path | None,
    aviary_id: str | None,
    allowed_species_file: Path | None,
) -> set[str] | None:
    """Determine the allowed-species set from the provided arguments.

    Priority:
    1. ``aviary_config`` + ``aviary_id`` → per-aviary lookup.
    2. ``allowed_species_file`` → flat file.
    3. Neither → ``None`` (allow everything).
    """
    if aviary_config is not None and aviary_id is not None:
        return load_aviary_species(aviary_config, aviary_id)

    if aviary_config is not None and aviary_id is None:
        raise ValueError(
            "--aviary-config was provided but --aviary was not. "
            "Specify which aviary to process, e.g. --aviary aviary_1"
        )

    return load_allowed_species(allowed_species_file)


def run_detection(
    input_path: Path,
    output_csv: Path,
    model_dir: Path,
    *,
    aviary_config: Path | None = None,
    aviary_id: str | None = None,
    allowed_species_file: Path | None = None,
    min_confidence: float = 0.05,
    temperature: float | None = 1.8,
    target_saturation: float = 0.30,
    overlap: float = 0.0,
    threads: int = 4,
    top_k: int = 3,
    eval_csv: Path | None = None,
) -> Path:
    """Run the full BirdNET detection pipeline.

    Args:
        input_path: Audio file or directory of audio files.
        output_csv: Final output CSV path.
        model_dir: Directory containing ``ZooCustom_v1.tflite`` and
            ``ZooCustom_v1_Labels.txt``.
        aviary_config: Path to ``aviary_config.json`` for per-aviary
            species filtering.  Must be combined with *aviary_id*.
        aviary_id: Aviary identifier (e.g. ``"aviary_1"``).
        allowed_species_file: Flat species whitelist file (alternative
            to aviary config).
        min_confidence: Global fallback confidence threshold.
        temperature: Temperature for confidence scaling (``T > 1``
            reduces saturation).  Set to ``1.0`` to disable.
        target_saturation: Target fraction of detections at ≥ 0.9999
            when auto-searching temperature (used when *temperature*
            is ``None``).
        overlap: Overlap in seconds between BirdNET analysis windows.
        threads: CPU threads for BirdNET.
        top_k: Keep at most this many species per detection window.
            Set to ``0`` to disable.
        eval_csv: Path to ``*_evaluation.csv`` for per-class optimal
            thresholds.  Inferred from the model filename when ``None``.

    Returns:
        Path to the final output CSV.
    """
    # ── 0.  Verify model assets ──────────────────────────────────────
    model_path = model_dir / "ZooCustom_v1.tflite"
    ensure_birdnet_assets(model_dir, download=False, include_optional=False)

    # ── 1.  Resolve allowed species ──────────────────────────────────
    allowed = _resolve_allowed_species(
        aviary_config, aviary_id, allowed_species_file,
    )

    # ── 2.  Load per-class thresholds (optional) ─────────────────────
    if eval_csv is None:
        candidate = default_eval_csv_for(model_path)
        eval_csv = candidate if candidate.exists() else None
    per_class_thr = load_optimal_thresholds(eval_csv)
    if not per_class_thr:
        print(
            f"[ARIA] No per-class thresholds loaded; "
            f"using global min_confidence={min_confidence}"
        )

    # ── 3.  Run BirdNET ──────────────────────────────────────────────
    work_dir = output_csv.parent / f"_aria_work_{output_csv.stem}"
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ARIA] Running BirdNET on {input_path} ...")
    run_birdnet_analyze(
        input_path=input_path,
        out_dir=work_dir,
        model_path=model_path,
        min_conf=min_confidence,
        overlap=overlap,
        threads=threads,
        allowed_species=allowed,
    )

    # ── 4.  Temperature scaling (on by default, T=1.8) ───────────────
    apply_temp = temperature is None or (
        temperature is not None and temperature != 1.0
    )

    if apply_temp:
        if temperature is None:
            temp_merged = work_dir / "_temp_merged.csv"
            merge_csvs(
                work_dir,
                input_path if input_path.is_dir() else None,
                temp_merged,
            )
            temperature, _ = find_optimal_temperature(
                temp_merged, target_saturation,
            )
            print(f"[ARIA] Auto-selected temperature T={temperature:.2f}")
            temp_merged.unlink(missing_ok=True)

        print(f"[ARIA] Applying temperature scaling (T={temperature:.2f}) ...")
        n_scaled = scale_individual_csvs(work_dir, temperature)
        print(f"[ARIA] Scaled {n_scaled} per-file CSVs")

    # ── 5.  Merge all per-file CSVs ──────────────────────────────────
    merged_csv = work_dir / "all_results.csv"
    n_merged = merge_csvs(
        work_dir,
        input_path if input_path.is_dir() else None,
        merged_csv,
    )
    print(f"[ARIA] Merged {n_merged} detection rows")

    if apply_temp:
        metrics = compute_saturation_metrics(merged_csv)
        if metrics:
            print(
                f"[ARIA] Saturation ≥0.9999: "
                f"{metrics.get('sat_0.9999', 0):.1f}%  "
                f"| Mean conf: {metrics.get('mean', 0):.3f}"
            )

    # ── 6.  Per-class threshold filter ───────────────────────────────
    filtered_csv = work_dir / "all_results.filtered.csv"
    before, after = apply_threshold_filter(
        merged_csv, filtered_csv, per_class_thr, min_confidence,
    )
    print(f"[ARIA] Threshold filter: {after}/{before} rows kept")

    # ── 7.  Top-K per window ─────────────────────────────────────────
    final_work_csv = filtered_csv
    if top_k > 0 and filtered_csv.exists():
        topk_csv = work_dir / f"all_results.filtered.top{top_k}.csv"
        tb, ta = topk_per_window(filtered_csv, topk_csv, top_k)
        print(f"[ARIA] Top-{top_k} per window: {ta}/{tb} rows kept")
        final_work_csv = topk_csv

    # ── 8.  Copy final result to requested output path ───────────────
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(final_work_csv, output_csv)
    print(f"[ARIA] Final predictions → {output_csv}")

    return output_csv
