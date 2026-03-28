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

This fetches the following from GitHub Releases:

- `ZooCustom_v1.tflite` – custom BirdNET classifier trained on zoo species
- `ZooCustom_v1_Labels.txt` – species label list

### 2. Run detection

```bash
aria-inference-birdnet detect \
  --input /path/to/audio_or_folder \
  --output predictions.csv \
  --model-dir ./models \
  --allowed-species-file allowed_species.txt
```

### 3. Python API

```python
from pathlib import Path
from aria_inference_birdnet import run_detection

run_detection(
    input_path=Path("recordings/"),
    output_csv=Path("predictions.csv"),
    model_dir=Path("models/"),
    allowed_species_file=Path("allowed_species.txt"),
    temperature=1.8,
    top_k=3,
)
```

## CLI reference

### `detect`

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | *(required)* | Audio file or directory |
| `--output` | *(required)* | Output CSV path |
| `--model-dir` | *(required)* | Directory with model files |
| `--allowed-species-file` | `None` | Species whitelist (one per line) |
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

## Pipeline stages

1. **BirdNET analysis** – runs `birdnet_analyzer.analyze` on the input audio
2. **Species filtering** – keeps only species listed in `allowed_species.txt`
3. **Temperature scaling** – calibrates over-saturated confidences (T=1.8 by default)
4. **CSV merging** – combines per-file result CSVs into one
5. **Per-class thresholds** – applies optimal thresholds from training evaluation (if available)
6. **Top-K filtering** – keeps at most K species per 3-second window

## Output format

The output CSV has BirdNET's standard columns:

```
File name,Start (s),End (s),Scientific name,Common name,Confidence
recording_001.wav,0.0,3.0,Phoenicopterus roseus,Greater Flamingo,0.8734
recording_001.wav,3.0,6.0,Bostrychia hagedash,Hadada Ibis,0.7521
```

## Notes

- This package is **inference-only**. It does not include training, data fetching, or segmentation.
- Species filtering is controlled **only** by the user-provided `allowed_species.txt`. There is no zoo-specific configuration.
- Model binaries are hosted as GitHub Release assets to keep the pip package small.
- Python 3.10 is **not** supported because [birdnet-analyzer requires ≥3.11](https://pypi.org/project/birdnet-analyzer/).

## License

Apache-2.0
