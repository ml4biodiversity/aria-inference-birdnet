"""CSV I/O helpers: merging per-file BirdNET results and writing output."""

from __future__ import annotations

import csv
from pathlib import Path


def merge_csvs(
    search_root: Path,
    fallback_root: Path | None,
    merged_path: Path,
) -> int:
    """Merge all ``.BirdNET.results.csv`` files under *search_root*.

    If no CSVs are found under *search_root* but *fallback_root* is
    given, the search is retried there (BirdNET sometimes writes results
    next to the input audio instead of the requested output directory).

    Args:
        search_root: Primary directory to search (recursively).
        fallback_root: Secondary directory to try when *search_root*
            yields nothing.
        merged_path: Destination for the single merged CSV.

    Returns:
        Total number of data rows written.
    """
    csvs = sorted(search_root.rglob("*.BirdNET.results.csv"))

    if not csvs and fallback_root is not None:
        csvs = sorted(fallback_root.rglob("*.BirdNET.results.csv"))

    if not csvs:
        return 0

    merged_path.parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    writer = None

    with open(merged_path, "w", newline="", encoding="utf-8") as out_f:
        for csv_file in csvs:
            with open(csv_file, "r", encoding="utf-8") as in_f:
                rows = list(csv.reader(in_f))
            if not rows:
                continue
            header, data = rows[0], rows[1:]
            if writer is None:
                writer = csv.writer(out_f)
                writer.writerow(header)
            for r in data:
                writer.writerow(r)
            total_rows += len(data)

    return total_rows


def write_predictions_csv(rows: list[dict], output_csv: Path) -> None:
    """Write a list of prediction dicts to a simple 5-column CSV.

    Expected keys per row: ``file``, ``start``, ``end``, ``species``,
    ``confidence``.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        f.write("file,start,end,species,confidence\n")
        for r in rows:
            f.write(
                f"{r['file']},{r['start']},{r['end']},"
                f"{r['species']},{r['confidence']:.4f}\n"
            )
