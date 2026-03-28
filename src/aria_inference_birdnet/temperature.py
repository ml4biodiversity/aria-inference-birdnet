"""Temperature scaling for BirdNET confidence calibration.

BirdNET's custom classifiers often produce over-saturated confidences
(many predictions at 0.9999).  Temperature scaling converts each
confidence back to a logit, divides by *T*, then re-applies sigmoid.
A temperature *T* > 1 spreads the distribution and reduces saturation.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Tuple

import numpy as np

from .utils import find_col_index


# ── core math ────────────────────────────────────────────────────────

def inverse_sigmoid(conf: float) -> float:
    """Convert a sigmoid probability back to its logit."""
    conf = np.clip(conf, 1e-7, 1 - 1e-7)
    return np.log(conf / (1 - conf))


def sigmoid(logit: float) -> float:
    """Standard sigmoid function."""
    return 1.0 / (1.0 + np.exp(-logit))


def apply_temperature_scaling(conf: float, temperature: float) -> float:
    """Scale a single confidence value.

    Args:
        conf: Original confidence in (0, 1).
        temperature: *T* > 1.0 reduces confidence; *T* < 1.0 increases it.

    Returns:
        Rescaled confidence.
    """
    if temperature == 1.0:
        return conf
    logit = inverse_sigmoid(conf)
    return sigmoid(logit / temperature)


# ── CSV-level helpers ────────────────────────────────────────────────

def scale_csv_confidences(
    csv_path: Path,
    temperature: float,
    output_path: Path | None = None,
) -> int:
    """Apply temperature scaling to every confidence in a BirdNET CSV.

    Args:
        csv_path: Input ``.BirdNET.results.csv``.
        temperature: Temperature value.
        output_path: Where to write the result.  Overwrites *csv_path*
            when ``None``.

    Returns:
        Number of data rows processed.
    """
    if not csv_path.exists():
        return 0

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if len(rows) <= 1:
        return 0

    header = rows[0]
    conf_idx = find_col_index(header, ["Confidence"])
    if conf_idx is None:
        return 0

    scaled_rows: list[list[str]] = []
    for r in rows[1:]:
        try:
            original = float(r[conf_idx])
            r[conf_idx] = f"{apply_temperature_scaling(original, temperature):.4f}"
        except (ValueError, IndexError):
            pass
        scaled_rows.append(r)

    dest = output_path or csv_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(scaled_rows)

    return len(scaled_rows)


def scale_individual_csvs(out_dir: Path, temperature: float) -> int:
    """Apply temperature scaling to every per-file BirdNET CSV in *out_dir*.

    Merged / aggregated CSVs (``all_results*.csv``) are skipped.

    Returns:
        Number of files successfully scaled.
    """
    skip_names = {
        "all_results.csv",
        "all_results.scaled.csv",
        "all_results.filtered.csv",
        "all_results.filtered.top3.csv",
    }
    csv_files = [
        p
        for p in out_dir.rglob("*.BirdNET.results.csv")
        if p.name not in skip_names
    ]

    scaled = 0
    for csv_file in csv_files:
        if scale_csv_confidences(csv_file, temperature, output_path=csv_file) > 0:
            scaled += 1
    return scaled


def compute_saturation_metrics(csv_path: Path) -> dict:
    """Return saturation statistics for the confidences in *csv_path*."""
    if not csv_path.exists():
        return {}

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if len(rows) <= 1:
        return {}

    conf_idx = find_col_index(rows[0], ["Confidence"])
    if conf_idx is None:
        return {}

    confs: list[float] = []
    for r in rows[1:]:
        try:
            confs.append(float(r[conf_idx]))
        except (ValueError, IndexError):
            continue
    if not confs:
        return {}

    total = len(confs)
    arr = np.array(confs)
    return {
        "total": total,
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "sat_0.9999": (arr >= 0.9999).sum() / total * 100,
        "sat_0.99": (arr >= 0.99).sum() / total * 100,
        "sat_0.95": (arr >= 0.95).sum() / total * 100,
        "range_low": (arr < 0.60).sum() / total * 100,
        "range_mid": ((arr >= 0.60) & (arr < 0.85)).sum() / total * 100,
        "range_high": (arr >= 0.85).sum() / total * 100,
    }


def find_optimal_temperature(
    csv_path: Path,
    target_saturation: float = 0.30,
) -> Tuple[float, dict]:
    """Search for the temperature that brings ≥0.9999 saturation closest to *target_saturation*.

    Args:
        csv_path: Merged CSV with original (unscaled) confidences.
        target_saturation: Desired fraction (0–1) of detections at ≥ 0.9999.

    Returns:
        ``(best_temperature, metrics_dict)``
    """
    if not csv_path.exists():
        return 1.0, {}

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if len(rows) <= 1:
        return 1.0, {}

    conf_idx = find_col_index(rows[0], ["Confidence"])
    if conf_idx is None:
        return 1.0, {}

    original_confs: list[float] = []
    for r in rows[1:]:
        try:
            original_confs.append(float(r[conf_idx]))
        except (ValueError, IndexError):
            continue
    if not original_confs:
        return 1.0, {}

    arr = np.array(original_confs)
    temperatures = np.arange(1.0, 3.0, 0.1)
    best_temp = 1.0
    best_diff = float("inf")
    best_metrics: dict = {}

    for T in temperatures:
        scaled = np.array([apply_temperature_scaling(c, float(T)) for c in arr])
        sat = float((scaled >= 0.9999).sum() / len(scaled))
        diff = abs(sat - target_saturation)
        if diff < best_diff:
            best_diff = diff
            best_temp = float(T)
            best_metrics = {
                "temperature": best_temp,
                "sat_0.9999": sat * 100,
                "sat_0.99": float((scaled >= 0.99).sum() / len(scaled) * 100),
                "mean": float(scaled.mean()),
                "median": float(np.median(scaled)),
            }

    return best_temp, best_metrics
