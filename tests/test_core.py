"""Tests for CLI, I/O helpers, and utility functions."""

import csv
from pathlib import Path

from click.testing import CliRunner

from aria_inference_birdnet.cli import main
from aria_inference_birdnet.io import merge_csvs, write_predictions_csv
from aria_inference_birdnet.utils import class_key, find_col_index


# ── CLI ──────────────────────────────────────────────────────────────

def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "ARIA" in result.output


def test_cli_detect_help():
    runner = CliRunner()
    result = runner.invoke(main, ["detect", "--help"])
    assert result.exit_code == 0
    assert "--input" in result.output
    assert "--temperature" in result.output
    assert "--aviary-config" in result.output
    assert "--aviary" in result.output


def test_cli_download_help():
    runner = CliRunner()
    result = runner.invoke(main, ["download-models", "--help"])
    assert result.exit_code == 0


def test_cli_list_aviaries_help():
    runner = CliRunner()
    result = runner.invoke(main, ["list-aviaries", "--help"])
    assert result.exit_code == 0


# ── utils ────────────────────────────────────────────────────────────

def test_find_col_index_found():
    header = ["File name", "Start (s)", "Confidence"]
    assert find_col_index(header, ["Confidence"]) == 2


def test_find_col_index_case_insensitive():
    header = ["file name", "CONFIDENCE"]
    assert find_col_index(header, ["confidence"]) == 1


def test_find_col_index_not_found():
    header = ["File", "Start"]
    assert find_col_index(header, ["Confidence"]) is None


def test_find_col_index_priority():
    header = ["Start", "Start (s)"]
    # "Start (s)" should match first when listed first in candidates
    assert find_col_index(header, ["Start (s)", "Start"]) == 1


def test_class_key():
    assert class_key("Phoenicopterus roseus", "Greater Flamingo") == "Phoenicopterus roseus_Greater Flamingo"
    assert class_key("", "Flamingo") == "Flamingo"
    assert class_key("Genus", "") == "Genus"


# ── I/O ──────────────────────────────────────────────────────────────

def test_write_predictions_csv(tmp_path: Path):
    rows = [
        {"file": "a.wav", "start": 0.0, "end": 3.0, "species": "Flamingo", "confidence": 0.91},
        {"file": "a.wav", "start": 3.0, "end": 6.0, "species": "Ibis", "confidence": 0.85},
    ]
    out = tmp_path / "pred.csv"
    write_predictions_csv(rows, out)

    with open(out, "r", encoding="utf-8") as f:
        lines = f.readlines()

    assert lines[0].strip() == "file,start,end,species,confidence"
    assert len(lines) == 3


def test_merge_csvs(tmp_path: Path):
    header = ["File name", "Start (s)", "End (s)", "Common name", "Confidence"]

    # Create two per-file CSVs
    d = tmp_path / "results"
    d.mkdir()
    for name, species, conf in [("a.BirdNET.results.csv", "Flamingo", "0.9"),
                                 ("b.BirdNET.results.csv", "Ibis", "0.8")]:
        with open(d / name, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerow(["x.wav", "0", "3", species, conf])

    merged = tmp_path / "merged.csv"
    n = merge_csvs(d, None, merged)
    assert n == 2

    with open(merged, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 3  # header + 2 data rows
