"""Tests for postprocess module (thresholds, top-K)."""

import csv
from pathlib import Path

from aria_inference_birdnet.postprocess import (
    apply_threshold_filter,
    default_eval_csv_for,
    load_optimal_thresholds,
    topk_per_window,
)


def _write_results_csv(path: Path, rows: list[list[str]]) -> None:
    header = ["File name", "Start (s)", "End (s)", "Scientific name", "Common name", "Confidence"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def test_default_eval_csv_for():
    p = Path("/models/ZooCustom_v1.tflite")
    assert default_eval_csv_for(p) == Path("/models/ZooCustom_v1_evaluation.csv")


def test_load_optimal_thresholds(tmp_path: Path):
    eval_path = tmp_path / "eval.csv"
    with open(eval_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Class", "Precision", "Optimal Threshold"])
        w.writerow(["Phoenicopterus roseus_Greater Flamingo", "0.92", "0.45"])
        w.writerow(["Bostrychia hagedash_Hadada Ibis", "0.88", "0.30"])

    thresholds = load_optimal_thresholds(eval_path)
    assert len(thresholds) == 2
    assert thresholds["Phoenicopterus roseus_Greater Flamingo"] == 0.45


def test_load_optimal_thresholds_missing():
    assert load_optimal_thresholds(Path("/nonexistent.csv")) == {}
    assert load_optimal_thresholds(None) == {}


def test_apply_threshold_filter(tmp_path: Path):
    src = tmp_path / "all.csv"
    _write_results_csv(src, [
        ["a.wav", "0.0", "3.0", "Phoenicopterus roseus", "Greater Flamingo", "0.50"],
        ["a.wav", "0.0", "3.0", "Bostrychia hagedash", "Hadada Ibis", "0.10"],
        ["a.wav", "3.0", "6.0", "Passer domesticus", "House Sparrow", "0.80"],
    ])

    dst = tmp_path / "filtered.csv"
    per_class = {"Phoenicopterus roseus_Greater Flamingo": 0.45}
    before, after = apply_threshold_filter(src, dst, per_class, fallback_min_conf=0.30)

    assert before == 3
    assert after == 2  # Flamingo ≥0.45 ✓, Ibis 0.10<0.30 ✗, Sparrow 0.80≥0.30 ✓


def test_topk_per_window(tmp_path: Path):
    src = tmp_path / "filtered.csv"
    _write_results_csv(src, [
        ["a.wav", "0.0", "3.0", "Sp1", "Species A", "0.90"],
        ["a.wav", "0.0", "3.0", "Sp2", "Species B", "0.80"],
        ["a.wav", "0.0", "3.0", "Sp3", "Species C", "0.70"],
        ["a.wav", "0.0", "3.0", "Sp4", "Species D", "0.60"],
    ])

    dst = tmp_path / "topk.csv"
    before, after = topk_per_window(src, dst, k=2)

    assert before == 4
    assert after == 2

    with open(dst, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    confs = [float(r[5]) for r in rows[1:]]
    assert confs == [0.90, 0.80]
