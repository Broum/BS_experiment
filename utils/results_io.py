"""Dated-folder layout for experiment results.

Each experiment run lands in `Results/<experiment>_YYYY-MM-DD_HHMMSS/`. The
folder holds `results.json` plus any PNGs the notebook saved alongside it.

Back-compat: `load_run` also accepts a direct .json path so legacy root-level
result files (e.g. `results_pyramid_BS_8_layers_14.5.json`) still work.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Mapping


_DEFAULT_ROOT = Path("Results")


def make_run_dir(experiment: str,
                 root: str | Path = _DEFAULT_ROOT,
                 timestamp: datetime | None = None) -> Path:
    """Create and return `<root>/<experiment>_YYYY-MM-DD_HHMMSS/`.

    `timestamp` defaults to `datetime.now()`. The folder is created if missing;
    re-using the same timestamp is harmless (exist_ok=True).
    """
    ts = (timestamp or datetime.now()).strftime("%Y-%m-%d_%H%M%S")
    folder = Path(root) / f"{experiment}_{ts}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def save_run(folder: str | Path,
             results: Mapping,
             name: str = "results.json") -> Path:
    """Dump `results` as JSON into `<folder>/<name>` and return the file path.

    Integer keys at the top level are stringified (matches the convention used
    by the existing N2.1 persist cell).
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    payload = {str(k): v for k, v in results.items()}
    target = folder / name
    target.write_text(json.dumps(payload, indent=2))
    return target


def load_run(folder_or_path: str | Path,
             name: str = "results.json") -> dict[int, dict]:
    """Load a run's results dict back from disk.

    - If `folder_or_path` is a directory, reads `<folder>/<name>`.
    - If `folder_or_path` is a .json file, reads it directly (legacy support).

    Top-level keys are restored to integers if they look like integers (the
    common case is layer counts).
    """
    p = Path(folder_or_path)
    target = p / name if p.is_dir() else p
    raw = json.loads(Path(target).read_text())
    return {(int(k) if str(k).lstrip("-").isdigit() else k): v
            for k, v in raw.items()}


def latest_run(experiment_prefix: str | None = None,
               root: str | Path = _DEFAULT_ROOT) -> Path | None:
    """Return the most-recent run folder, optionally filtered by prefix.

    Folders are matched by name only (`<prefix>_*`), not by mtime; this is
    robust against file-system clock skew and accidental touches.
    Returns None if no matching folder exists.
    """
    root = Path(root)
    if not root.is_dir():
        return None
    candidates = [p for p in root.iterdir()
                  if p.is_dir() and (experiment_prefix is None or p.name.startswith(f"{experiment_prefix}_"))]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.name)[-1]
