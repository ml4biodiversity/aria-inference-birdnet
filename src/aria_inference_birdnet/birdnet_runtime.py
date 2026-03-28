"""BirdNET inference runtime.

Calls ``birdnet_analyzer.analyze`` as a subprocess, then parses the
resulting per-file ``.BirdNET.results.csv`` files into a flat list of
detection dicts.

``birdnet-analyzer`` is a declared pip dependency of this package and
is installed automatically via ``pip install aria-inference-birdnet``.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

from .species_filter import filter_csv_in_place
from .utils import find_col_index


# ── subprocess call ──────────────────────────────────────────────────

def run_birdnet_analyze(
    input_path: Path,
    out_dir: Path,
    model_path: Path,
    *,
    min_conf: float = 0.05,
    overlap: float = 0.0,
    threads: int = 4,
    allowed_species: set[str] | None = None,
) -> None:
    """Run ``birdnet_analyzer.analyze`` on *input_path*.

    Results are written as ``.BirdNET.results.csv`` files under
    *out_dir*.  If *allowed_species* is provided, every per-file CSV is
    immediately filtered to keep only those species.

    Args:
        input_path: Single audio file **or** directory of audio files.
        out_dir: Where BirdNET writes its result CSVs.
        model_path: Path to the custom ``.tflite`` classifier.
        min_conf: Minimum confidence passed to ``--min_conf``.
        overlap: Overlap in seconds between analysis windows.
        threads: CPU threads for BirdNET.
        allowed_species: Optional species whitelist (common names).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "birdnet_analyzer.analyze",
        str(input_path),
        "-o", str(out_dir),
        "--classifier", str(model_path),
        "--min_conf", str(min_conf),
        "--overlap", str(overlap),
        "--threads", str(threads),
        "--rtype", "csv",
        "--skip_existing_results",
    ]

    subprocess.run(cmd, check=True)

    # Post-filter each CSV to allowed species
    if allowed_species is not None:
        for csv_file in out_dir.rglob("*.BirdNET.results.csv"):
            filter_csv_in_place(csv_file, allowed_species)


# ── CSV → list[dict] parser ─────────────────────────────────────────

def parse_birdnet_results(results_dir: Path) -> list[dict]:
    """Parse all ``.BirdNET.results.csv`` files into a flat detection list.

    Each dict contains:
    - ``file`` – source audio filename
    - ``start`` – detection start time (seconds)
    - ``end`` – detection end time (seconds)
    - ``species`` – common name
    - ``scientific_name`` – scientific name
    - ``confidence`` – float confidence score

    Args:
        results_dir: Directory to search recursively.

    Returns:
        Flat list of detection dicts, sorted by file then start time.
    """
    detections: list[dict] = []

    for csv_path in sorted(results_dir.rglob("*.BirdNET.results.csv")):
        with open(csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.reader(f))

        if len(rows) <= 1:
            continue

        header = rows[0]
        file_idx = find_col_index(header, ["File name", "Filename", "File"])
        start_idx = find_col_index(header, ["Start (s)", "Start"])
        end_idx = find_col_index(header, ["End (s)", "End"])
        com_idx = find_col_index(header, ["Common name", "Common_name", "Common Name"])
        sci_idx = find_col_index(header, ["Scientific name", "Scientific", "Scientific Name"])
        conf_idx = find_col_index(header, ["Confidence"])

        if None in (file_idx, start_idx, end_idx, com_idx, conf_idx):
            continue

        for r in rows[1:]:
            try:
                detections.append(
                    {
                        "file": r[file_idx],
                        "start": float(r[start_idx]),
                        "end": float(r[end_idx]),
                        "species": r[com_idx].strip(),
                        "scientific_name": r[sci_idx].strip() if sci_idx is not None else "",
                        "confidence": float(r[conf_idx]),
                    }
                )
            except (ValueError, IndexError):
                continue

    detections.sort(key=lambda d: (d["file"], d["start"]))
    return detections
