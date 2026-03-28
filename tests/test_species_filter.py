"""Tests for species_filter module."""

import csv
import json
from pathlib import Path

import pytest

from aria_inference_birdnet.species_filter import (
    _parse_species_line,
    filter_csv_in_place,
    filter_predictions,
    list_aviaries,
    load_allowed_species,
    load_aviary_species,
)


# ── helpers ──────────────────────────────────────────────────────────

def _write_aviary_config(tmp_path: Path) -> Path:
    config = {
        "wild_birds": ["House Sparrow", "Common Blackbird"],
        "aviaries": {
            "aviary_1": {
                "species": ["Greater Flamingo", "Hadada Ibis"]
            },
            "aviary_2": {
                "species": ["Red-billed Quelea"]
            },
        },
    }
    p = tmp_path / "aviary_config.json"
    p.write_text(json.dumps(config), encoding="utf-8")
    return p


# ── _parse_species_line ──────────────────────────────────────────────

def test_parse_birdnet_label_format():
    common, scientific = _parse_species_line("Phoenicopterus roseus_Greater Flamingo")
    assert common == "Greater Flamingo"
    assert scientific == "Phoenicopterus roseus"


def test_parse_bare_common_name():
    common, scientific = _parse_species_line("Greater Flamingo")
    assert common == "Greater Flamingo"
    assert scientific == ""


def test_parse_subspecies_label():
    common, scientific = _parse_species_line(
        "Burhinus oedicnemus oedicnemus_Stone Curlew"
    )
    assert common == "Stone Curlew"
    assert scientific == "Burhinus oedicnemus oedicnemus"


# ── load_aviary_species ──────────────────────────────────────────────

def test_load_aviary_species(tmp_path: Path):
    config_path = _write_aviary_config(tmp_path)
    allowed = load_aviary_species(config_path, "aviary_1")
    # aviary species + wild birds
    assert allowed == {"Greater Flamingo", "Hadada Ibis", "House Sparrow", "Common Blackbird"}


def test_load_aviary_species_unknown_id(tmp_path: Path):
    config_path = _write_aviary_config(tmp_path)
    with pytest.raises(KeyError, match="aviary_99"):
        load_aviary_species(config_path, "aviary_99")


def test_list_aviaries(tmp_path: Path):
    config_path = _write_aviary_config(tmp_path)
    ids = list_aviaries(config_path)
    assert ids == ["aviary_1", "aviary_2"]


# ── load_allowed_species ─────────────────────────────────────────────

def test_load_allowed_species_birdnet_format(tmp_path: Path):
    p = tmp_path / "allowed.txt"
    p.write_text(
        "Phoenicopterus roseus_Greater Flamingo\n"
        "Bostrychia hagedash_Hadada Ibis\n\n",
        encoding="utf-8",
    )
    allowed = load_allowed_species(p)
    assert allowed == {"Greater Flamingo", "Hadada Ibis"}


def test_load_allowed_species_bare_names(tmp_path: Path):
    p = tmp_path / "allowed.txt"
    p.write_text("Greater Flamingo\nHadada Ibis\n", encoding="utf-8")
    allowed = load_allowed_species(p)
    assert allowed == {"Greater Flamingo", "Hadada Ibis"}


def test_load_allowed_species_mixed_formats(tmp_path: Path):
    p = tmp_path / "allowed.txt"
    p.write_text(
        "Phoenicopterus roseus_Greater Flamingo\n"
        "Hadada Ibis\n",
        encoding="utf-8",
    )
    allowed = load_allowed_species(p)
    assert allowed == {"Greater Flamingo", "Hadada Ibis"}


def test_load_allowed_species_none():
    assert load_allowed_species(None) is None


# ── filter_predictions ───────────────────────────────────────────────

def test_filter_predictions_with_allowed():
    rows = [
        {"species": "Greater Flamingo", "confidence": 0.9},
        {"species": "Hadada Ibis", "confidence": 0.8},
        {"species": "House Sparrow", "confidence": 0.7},
    ]
    allowed = {"Greater Flamingo", "Hadada Ibis"}
    out = filter_predictions(rows, allowed)
    assert len(out) == 2
    assert {r["species"] for r in out} == {"Greater Flamingo", "Hadada Ibis"}


def test_filter_predictions_none_allows_all():
    rows = [
        {"species": "Greater Flamingo", "confidence": 0.9},
        {"species": "House Sparrow", "confidence": 0.7},
    ]
    out = filter_predictions(rows, None)
    assert len(out) == 2


# ── filter_csv_in_place ──────────────────────────────────────────────

def test_filter_csv_in_place(tmp_path: Path):
    csv_path = tmp_path / "test.BirdNET.results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["File name", "Start (s)", "End (s)", "Scientific name", "Common name", "Confidence"])
        w.writerow(["a.wav", "0.0", "3.0", "Phoenicopterus roseus", "Greater Flamingo", "0.95"])
        w.writerow(["a.wav", "3.0", "6.0", "Passer domesticus", "House Sparrow", "0.80"])
        w.writerow(["a.wav", "6.0", "9.0", "Bostrychia hagedash", "Hadada Ibis", "0.70"])

    filter_csv_in_place(csv_path, {"Greater Flamingo", "Hadada Ibis"})

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    assert len(rows) == 3
    species_kept = {r[4] for r in rows[1:]}
    assert species_kept == {"Greater Flamingo", "Hadada Ibis"}


def test_filter_csv_in_place_none_is_noop(tmp_path: Path):
    csv_path = tmp_path / "test.BirdNET.results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Common name", "Confidence"])
        w.writerow(["House Sparrow", "0.80"])

    filter_csv_in_place(csv_path, None)

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert len(rows) == 2
