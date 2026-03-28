# aria-inference-birdnet

BirdNET-only inference package for [ARIA](https://github.com/ml4biodiversity/ARIA) (Acoustic Recognition for Inventories of Aviaries).

This is the baseline detection system for the [BioDCASE 2026 Challenge](https://biodcase.github.io/).

## Installation

Requires **Python 3.11 or 3.12**.

```bash
pip install aria-inference-birdnet
```

This automatically installs [BirdNET-Analyzer](https://github.com/birdnet-team/BirdNET-Analyzer) and all other dependencies.

## Quick start

### 1. Download model files

```bash
aria-inference-birdnet download-models --dir ./models
```

This fetches the custom BirdNET classifier and species labels from GitHub Releases.

### 2. Run detection on an aviary

Each aviary has a different set of species. The `aviary_config.json` (provided with the challenge data) maps each aviary ID to its species list, so that predictions are filtered to only the species that can actually be present.

```bash
aria-inference-birdnet detect \
  --input aviary_1/ \
  --output predictions_aviary_1.csv \
  --model-dir ./models \
  --aviary-config aviary_config.json \
  --aviary aviary_1
```

### 3. List available aviaries

```bash
aria-inference-birdnet list-aviaries --aviary-config aviary_config.json
```

### 4. Python API

```python
from pathlib import Path
from aria_inference_birdnet import run_detection

run_detection(
    input_path=Path("aviary_1/"),
    output_csv=Path("predictions_aviary_1.csv"),
    model_dir=Path("models/"),
    aviary_config=Path("aviary_config.json"),
    aviary_id="aviary_1",
)
```

## How aviary filtering works

The `aviary_config.json` contains two things:

- **Wild birds**: species common to all aviaries (e.g. House Sparrow, Common Blackbird). These are always included in predictions.
- **Per-aviary species**: species that live in a specific aviary.

When you specify `--aviary aviary_1`, the detector only keeps predictions for that aviary's species plus wild birds. This removes false positives from species that are not present in the aviary.

Alternatively, you can use a flat species file with `--allowed-species-file` instead of the aviary config. This applies the same species list to all input audio regardless of aviary.

## CLI reference

### `detect`

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | *(required)* | Audio file or directory |
| `--output` | *(required)* | Output CSV path |
| `--model-dir` | *(required)* | Directory with model files |
| `--aviary-config` | `None` | Path to `aviary_config.json` |
| `--aviary` | `None` | Aviary ID (e.g. `aviary_1`). Requires `--aviary-config` |
| `--allowed-species-file` | `None` | Flat species whitelist (alternative to aviary config) |
| `--min-confidence` | `0.05` | Global confidence cutoff |
| `--temperature` | `1.8` | Confidence scaling (T>1 reduces saturation) |
| `--overlap` | `0.0` | Window overlap in seconds |
| `--threads` | `4` | CPU threads |
| `--topk` | `3` | Max species per window (0=disable) |
| `--eval-csv` | *(auto)* | Per-class threshold CSV |

### `download-models`

| Flag | Default | Description |
|------|---------|-------------|
| `--dir` | *(required)* | Local directory for model files |
| `--skip-optional` | `False` | Skip the evaluation CSV |

### `list-aviaries`

| Flag | Default | Description |
|------|---------|-------------|
| `--aviary-config` | *(required)* | Path to `aviary_config.json` |

## Pipeline stages

1. **BirdNET analysis** – runs `birdnet_analyzer.analyze` on the input audio
2. **Species filtering** – keeps only species present in the aviary (from config) or in the allowed-species file
3. **Temperature scaling** – calibrates over-saturated confidences (T=1.8 by default)
4. **CSV merging** – combines per-file result CSVs into one
5. **Per-class thresholds** – applies optimal thresholds from training evaluation (if available)
6. **Top-K filtering** – keeps at most K species per 3-second window

## Output format

```
File name,Start (s),End (s),Scientific name,Common name,Confidence
recording_001.wav,0.0,3.0,Phoenicopterus roseus,Greater Flamingo,0.8734
recording_001.wav,3.0,6.0,Bostrychia hagedash,Hadada Ibis,0.7521
```

## Notes

- This package is **inference-only**. It does not include training, data fetching, or segmentation.
- Model binaries are hosted as GitHub Release assets to keep the pip package small.
- Python 3.10 is **not** supported because [birdnet-analyzer requires ≥3.11](https://pypi.org/project/birdnet-analyzer/).

## License

Apache-2.0
