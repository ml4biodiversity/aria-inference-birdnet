"""Verify (and optionally download) BirdNET runtime model assets.

Model binaries are hosted as GitHub Release attachments to keep the pip
wheel small.  The ``download-models`` CLI command calls
:func:`ensure_birdnet_assets` which fetches any missing files.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

_RELEASE_BASE = (
    "https://github.com/ml4biodiversity/aria-inference-birdnet"
    "/releases/download"
)

# Update this when you cut a new release with updated model files.
_RELEASE_TAG = "v0.1.1"

# Mapping: local filename → download URL.
_ASSETS: dict[str, str] = {
    "ZooCustom_v1.tflite": f"{_RELEASE_BASE}/{_RELEASE_TAG}/ZooCustom_v1.tflite",
    "ZooCustom_v1_Labels.txt": f"{_RELEASE_BASE}/{_RELEASE_TAG}/ZooCustom_v1_Labels.txt",
}

# Optional asset — only needed when per-class thresholds are desired.
_OPTIONAL_ASSETS: dict[str, str] = {
    "ZooCustom_v1_evaluation.csv": (
        f"{_RELEASE_BASE}/{_RELEASE_TAG}/ZooCustom_v1_evaluation.csv"
    ),
}


def ensure_birdnet_assets(
    model_dir: Path,
    *,
    download: bool = True,
    include_optional: bool = True,
) -> None:
    """Ensure all required BirdNET model files exist in *model_dir*.

    When *download* is ``True`` (the default), missing files are fetched
    from the GitHub Release.  Otherwise a :class:`FileNotFoundError` is
    raised listing the missing files.

    Args:
        model_dir: Local directory that should contain the model files.
        download: Whether to auto-download missing files.
        include_optional: Also fetch optional assets like the evaluation
            CSV (used for per-class thresholds).
    """
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── Required assets (fail if missing and not downloadable) ────────
    missing: list[str] = []
    for filename, url in _ASSETS.items():
        dest = model_dir / filename
        if dest.exists():
            continue
        if download:
            print(f"Downloading {filename} ...")
            urllib.request.urlretrieve(url, dest)
            print(f"  → {dest}")
        else:
            missing.append(filename)

    if missing:
        raise FileNotFoundError(
            f"Missing BirdNET runtime assets in {model_dir}: "
            + ", ".join(missing)
        )

    # ── Optional assets  ────────────────────────
    if include_optional:
        for filename, url in _OPTIONAL_ASSETS.items():
            dest = model_dir / filename
            if dest.exists():
                continue
            if download:
                try:
                    print(f"Downloading {filename} ...")
                    urllib.request.urlretrieve(url, dest)
                    print(f"  → {dest}")
                except urllib.error.HTTPError:
                    print(f"  ⊘ {filename} not available (optional, skipping)")