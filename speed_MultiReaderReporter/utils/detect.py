# speed_MultiReaderReporter/utils/detect.py
from __future__ import annotations
from pathlib import Path
import zipfile
from dataclasses import dataclass
from typing import Literal

DetectedKind = Literal["mat", "csvzip", "csv", "unknown"]

@dataclass(frozen=True)
class DetectedItem:
    path: Path        # actual path on disk
    kind: DetectedKind

def _is_zip_with_csv(p: Path) -> bool:
    if not p.is_file():
        return False
    try:
        if not zipfile.is_zipfile(p):
            return False
        with zipfile.ZipFile(p, "r") as zf:
            for name in zf.namelist():
                if name.lower().endswith(".csv"):
                    return True
        return False
    except Exception:
        return False

def detect_kind(p: Path) -> DetectedKind:
    """
    Classify a single path.
    - .mat  -> 'mat'
    - .zip (with any .csv member) -> 'csvzip'
    - .csv  -> 'csv'
    else    -> 'unknown'
    """
    if p.suffix.lower() == ".mat":
        return "mat"
    if p.suffix.lower() == ".csv":
        return "csv"
    if p.suffix.lower() == ".zip" and _is_zip_with_csv(p):
        return "csvzip"
    if p.suffix.lower() == ".pkl":
        return "pkl"
    return "unknown"

def discover_inputs(root: Path, recurse: bool = True) -> list[DetectedItem]:
    """
    If 'root' is a file -> return that one item (if known).
    If 'root' is a folder -> walk (optionally recursively) and collect .mat/.csv/.zip(with csv).
    """
    items: list[DetectedItem] = []
    if root.is_file():
        kind = detect_kind(root)
        if kind != "unknown":
            items.append(DetectedItem(root.resolve(), kind))
        return items

    # folder
    if recurse:
        it = root.rglob("*")
    else:
        it = root.glob("*")

    for p in it:
        if not p.is_file():
            continue
        kind = detect_kind(p)
        if kind != "unknown":
            items.append(DetectedItem(p.resolve(), kind))

    # deterministic ordering
    items.sort(key=lambda x: (x.kind, str(x.path)))
    return items
