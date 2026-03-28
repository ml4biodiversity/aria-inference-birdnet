"""Tests for temperature scaling module."""

import csv
from pathlib import Path

import numpy as np
import pytest

from aria_inference_birdnet.temperature import (
    apply_temperature_scaling,
    inverse_sigmoid,
    scale_csv_confidences,
    sigmoid,
)


def test_sigmoid_inverse_roundtrip():
    for val in [0.01, 0.25, 0.5, 0.75, 0.99]:
        assert abs(sigmoid(inverse_sigmoid(val)) - val) < 1e-6


def test_temperature_1_is_identity():
    assert apply_temperature_scaling(0.85, 1.0) == 0.85


def test_temperature_gt1_reduces_confidence():
    original = 0.95
    scaled = apply_temperature_scaling(original, 1.8)
    assert scaled < original


def test_temperature_lt1_increases_confidence():
    original = 0.60
    scaled = apply_temperature_scaling(original, 0.5)
    assert scaled > original


def test_scale_csv_confidences(tmp_path: Path):
    csv_path = tmp_path / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["File", "Species", "Confidence"])
        w.writerow(["a.wav", "Flamingo", "0.9999"])
        w.writerow(["a.wav", "Ibis", "0.5000"])

    n = scale_csv_confidences(csv_path, temperature=1.8)
    assert n == 2

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    # Original 0.9999 should be reduced
    scaled_conf = float(rows[1][2])
    assert scaled_conf < 0.9999


def test_scale_csv_nonexistent(tmp_path: Path):
    assert scale_csv_confidences(tmp_path / "nope.csv", 1.5) == 0
