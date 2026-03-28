"""Post-processing filters applied after BirdNET inference.

Provides two stages that mirror the internal ARIA pipeline:

1. **Per-class threshold filtering** – each species can have its own
   optimal confidence cutoff loaded from the ``*_evaluation.csv`` that
   BirdNET produces during training.
2. **Top-K per window** – for every 3-second detection window, keep at
   most *K* species sorted by descending confidence.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

from .utils import class_key, find_col_index


# ── per-class optimal thresholds ─────────────────────────────────────

def load_optimal_thresholds(eval_csv: Path | None) -> dict[str, float]:
    """Load ``{Scientific_Common: threshold}`` from a BirdNET evaluation CSV.

    The CSV is expected to contain at least a ``Class`` column and an
    ``Optimal Threshold`` column (case-insensitive match).

    Args:
        eval_csv: Path to the ``*_evaluation.csv``.  Returns an empty
            dict when ``None`` or the file does not exist.
    """
    if eval_csv is None or not eval_csv.exists():
        return {}

    with open(eval_csv, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return {}

    header = rows[0]
    class_idx = find_col_index(header, ["Class"])
    thr_idx = find_col_index(
        header,
        ["Optimal Threshold", "Optimal threshold", "optimal_threshold"],
    )
    if class_idx is None or thr_idx is None:
        return {}

    out: dict[str, float] = {}
    for r in rows[1:]:
        try:
            label = r[class_idx].strip()
            thr = float(r[thr_idx]) if r[thr_idx] not in ("", None) else math.nan
            if label and not math.isnan(thr):
                out[label] = thr
        except Exception:
            continue
    return out


def default_eval_csv_for(model_path: Path) -> Path:
    """Infer the evaluation CSV path from a ``.tflite`` model path.

    ``models/ZooCustom_v1.tflite`` → ``models/ZooCustom_v1_evaluation.csv``
    """
    name = model_path.name
    if name.endswith(".tflite"):
        stem = name[: -len(".tflite")]
        return model_path.parent / f"{stem}_evaluation.csv"
    return model_path.with_name(name + "_evaluation.csv")


# ── confidence threshold filter ──────────────────────────────────────

def apply_threshold_filter(
    all_csv: Path,
    filtered_csv: Path,
    per_class_thr: dict[str, float],
    fallback_min_conf: float,
) -> tuple[int, int]:
    """Filter a merged results CSV using per-class or global thresholds.

    Each row is kept when its confidence ≥ the per-class threshold (if
    one exists) or ≥ *fallback_min_conf* otherwise.

    Args:
        all_csv: Merged input CSV.
        filtered_csv: Destination for the filtered CSV.
        per_class_thr: Mapping from ``Scientific_Common`` label to
            optimal threshold.
        fallback_min_conf: Global fallback confidence cutoff.

    Returns:
        ``(rows_before, rows_after)`` tuple.
    """
    if not all_csv.exists():
        return 0, 0

    with open(all_csv, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return 0, 0

    header = rows[0]
    sci_idx = find_col_index(header, ["Scientific name", "Scientific", "Scientific Name"])
    com_idx = find_col_index(header, ["Common name", "Common", "Common Name"])
    conf_idx = find_col_index(header, ["Confidence"])
    if None in (sci_idx, com_idx, conf_idx):
        return 0, 0

    kept: list[list[str]] = []
    for r in rows[1:]:
        try:
            conf = float(r[conf_idx])
        except Exception:
            continue
        key = class_key(r[sci_idx], r[com_idx])
        thr = per_class_thr.get(key, fallback_min_conf)
        if conf >= thr:
            kept.append(r)

    filtered_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(filtered_csv, "w", newline="", encoding="utf-8") as out_f:
        w = csv.writer(out_f)
        w.writerow(header)
        w.writerows(kept)

    return len(rows) - 1, len(kept)


# ── top-K per detection window ───────────────────────────────────────

def topk_per_window(
    filtered_csv: Path,
    out_csv: Path,
    k: int,
) -> tuple[int, int]:
    """Keep only the top-*k* species per ``(file, start, end)`` window.

    Args:
        filtered_csv: Input CSV (already threshold-filtered).
        out_csv: Output CSV path.
        k: Maximum species to retain per window.

    Returns:
        ``(rows_before, rows_after)`` tuple.
    """
    if k <= 0:
        return 0, 0

    with open(filtered_csv, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return 0, 0

    header = rows[0]
    start_idx = find_col_index(header, ["Start (s)", "Start"])
    end_idx = find_col_index(header, ["End (s)", "End"])
    conf_idx = find_col_index(header, ["Confidence"])
    file_idx = find_col_index(header, ["File name", "Filename", "File"])
    if None in (start_idx, end_idx, conf_idx, file_idx):
        return 0, 0

    buckets: dict[tuple, list[tuple[float, list[str]]]] = defaultdict(list)
    for r in rows[1:]:
        try:
            key = (r[file_idx], r[start_idx], r[end_idx])
            conf = float(r[conf_idx])
        except Exception:
            continue
        buckets[key].append((conf, r))

    kept: list[list[str]] = []
    for lst in buckets.values():
        lst.sort(key=lambda x: x[0], reverse=True)
        kept.extend(r for _, r in lst[:k])

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(kept)

    return len(rows) - 1, len(kept)
