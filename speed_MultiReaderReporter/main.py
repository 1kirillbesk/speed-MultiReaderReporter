# speed_MultiReaderReporter/main.py
from __future__ import annotations
from pathlib import Path
import json
import sys
import yaml
import re

# Make direct script execution work by exposing the package parent before imports.
here = Path(__file__).resolve().parent
package_parent = here.parent
for path in (package_parent, here, here / "core", here / "loaders", here / "utils"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

try:
    from .loaders import csvzip_loader, mat_loader, pkl_loader
    from .utils.detect import DetectedItem, discover_inputs
    from .core.pipeline import run_pipeline
except ImportError:
    from loaders import csvzip_loader, mat_loader, pkl_loader
    from utils.detect import DetectedItem, discover_inputs
    from core.pipeline import run_pipeline

MANIFEST_NAME = ".processing_manifest.json"
CACHE_MODE_SUMMARY_ONLY = "summary_only"
CACHE_MODE_FULL_RAW_PLOTS = "full_raw_plots"

def load_config(cfg_path: Path) -> dict:
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _manifest_path(out_root: Path) -> Path:
    return out_root / MANIFEST_NAME

def _empty_manifest() -> dict:
    return {"version": 1, "cells": {}}

def load_manifest(out_root: Path, verbose: bool = True) -> dict:
    manifest_path = _manifest_path(out_root)
    if not manifest_path.exists():
        return _empty_manifest()
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        if verbose:
            print(f"[WARN] failed to read cache manifest {manifest_path.name}: {e}")
        return _empty_manifest()

    if not isinstance(data, dict):
        return _empty_manifest()
    data.setdefault("version", 1)
    cells = data.get("cells")
    if not isinstance(cells, dict):
        data["cells"] = {}
    return data

def save_manifest(out_root: Path, manifest: dict) -> None:
    manifest_path = _manifest_path(out_root)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

def _summary_path_for_cell(out_root: Path, cell_id: str) -> Path:
    return out_root / "cell_feature" / f"{cell_id}.csv"

def build_item_manifest(items: list[DetectedItem]) -> dict[str, dict]:
    item_manifest: dict[str, dict] = {}
    for item in items:
        stat = item.path.stat()
        item_manifest[str(item.path)] = {
            "kind": item.kind,
            "mtime_ns": stat.st_mtime_ns,
            "size": stat.st_size,
        }
    return item_manifest

def should_process_cell(cell_id: str,
                        items: list[DetectedItem],
                        out_root: Path,
                        manifest: dict,
                        cache_mode: str) -> tuple[bool, str, bool]:
    if cache_mode not in {CACHE_MODE_SUMMARY_ONLY, CACHE_MODE_FULL_RAW_PLOTS}:
        raise ValueError(
            f"Unsupported existing_cell_mode={cache_mode!r}. "
            f"Use {CACHE_MODE_SUMMARY_ONLY!r} or {CACHE_MODE_FULL_RAW_PLOTS!r}."
        )

    current_inputs = build_item_manifest(items)
    summary_path = _summary_path_for_cell(out_root, cell_id)

    if cache_mode == CACHE_MODE_FULL_RAW_PLOTS:
        return True, "existing_cell_mode forces full raw reload", False

    cached_inputs = manifest.get("cells", {}).get(cell_id, {}).get("inputs")
    if cached_inputs == current_inputs:
        if summary_path.exists():
            return False, "inputs unchanged; summary already exists", False
        return True, "inputs unchanged but summary is missing", False

    if cached_inputs is None and summary_path.exists():
        summary_mtime_ns = summary_path.stat().st_mtime_ns
        if current_inputs and all(meta["mtime_ns"] <= summary_mtime_ns for meta in current_inputs.values()):
            manifest.setdefault("cells", {})[cell_id] = {"inputs": current_inputs}
            return False, "summary newer than all inputs; bootstrapped cache manifest", True

    if cached_inputs is None:
        return True, "no cache record for this cell", False
    return True, "new, changed, or removed input detected", False

def update_manifest_for_cell(manifest: dict, cell_id: str, items: list[DetectedItem]) -> None:
    manifest.setdefault("cells", {})[cell_id] = {"inputs": build_item_manifest(items)}

def main():
    # ---------- config ----------
    here = Path(__file__).resolve().parent
    cfg = load_config(here / "pc602/config_lw.yaml")

    in_path = Path(cfg["input"]["path"]).resolve()
    recurse = bool(cfg["input"].get("recurse", True))
    out_root = Path(cfg["output"]["root"]).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    verbose = bool(cfg.get("logging", {}).get("verbose", True))
    cache_mode = str(cfg.get("cache", {}).get("existing_cell_mode", CACHE_MODE_SUMMARY_ONLY)).lower()
    manifest = load_manifest(out_root, verbose=verbose)
    manifest_dirty = False

    if verbose:
        print(f"[cfg] input={in_path} (recurse={recurse})")
        print(f"[cfg] output={out_root}")
        print(f"[cfg] cache mode={cache_mode}")

    # ---------- discover ----------
    detected = discover_inputs(in_path, recurse=recurse)
    name_contains = cfg.get("input", {}).get("name_contains", [])
    needles = [n.lower() for n in name_contains]
    if needles:
        detected = [
            d for d in detected
            if any(n in d.path.name.lower() for n in needles)
        ]

    cell_names = cfg["input"]["cell_name_contain"]
    if cell_names:
        cell_patterns = [
            re.compile(rf"{re.escape(cell_name)}(?:_|$)", re.IGNORECASE)
            for cell_name in cell_names
        ]
        detected = [
            d for d in detected
            if any(p.search(d.path.name) for p in cell_patterns)
        ]

    if not detected:
        print(f"[INFO] No MAT/CSV/ZIP(CSV) inputs found under: {in_path}")
        sys.exit(0)
    if verbose:
        kinds = {}
        for d in detected:
            kinds.setdefault(d.kind, 0)
            kinds[d.kind] += 1
        print(f"[detector] found {sum(kinds.values())} inputs → {kinds}")

    # ---------- loader registry ----------
    registry = {
        "csvzip": csvzip_loader.load,
        "csv":    csvzip_loader.load,
        "mat":    mat_loader.load,
        "pkl":    pkl_loader.load,
    }
    cell_resolver = {
        "csvzip": csvzip_loader.infer_cell_from_path,
        "csv":    csvzip_loader.infer_cell_from_path,
        "mat":    mat_loader.infer_cell_from_path,
        "pkl":    pkl_loader.infer_cell_from_path,
    }

    # group detected items per cell to bound memory and process one cell at a time
    items_by_cell: dict[str, list] = {}
    for item in detected:
        loader = registry.get(item.kind)
        resolver = cell_resolver.get(item.kind)
        if loader is None or resolver is None:
            if verbose:
                print(f"[skip] no loader for {item.kind}: {item.path.name}")
            continue
        try:
            cell_id = resolver(item.path)
        except Exception as e:
            if verbose:
                print(f"[WARN] failed to infer cell for {item.path.name}: {e}")
            continue
        items_by_cell.setdefault(cell_id, []).append(item)

    if not items_by_cell:
        if verbose:
            print("[INFO] No runs loaded; exiting without processing pipeline.")
        sys.exit(0)

    if verbose:
        print(
            f"[grouping] processing {len(detected)} detected item(s) "
            f"across {len(items_by_cell)} cell(s): {', '.join(sorted(items_by_cell))}"
        )

    processed_cells = 0
    skipped_cells = 0
    for cell_id, items in sorted(items_by_cell.items()):
        try:
            should_process, reason, manifest_updated = should_process_cell(
                cell_id, items, out_root, manifest, cache_mode
            )
        except ValueError as e:
            print(f"[ERROR] {e}")
            sys.exit(2)

        if not should_process:
            skipped_cells += 1
            manifest_dirty = manifest_dirty or manifest_updated
            if verbose:
                print(f"[cache] {cell_id}: {reason}")
            continue

        # Load all raw inputs for the same cell together so one pipeline run handles the aggregated data.
        cell_runs = []
        if verbose:
            print(f"[cell] {cell_id}: loading {len(items)} item(s) ({reason})")
        for item in items:
            loader = registry[item.kind]
            if verbose:
                print(f"  [load] {item.kind:6} {item.path.name}")
            try:
                runs = loader(item.path, cfg, out_root)
                if not runs:
                    continue
                cell_runs.extend(runs)
            except Exception as e:
                print(f"[WARN] loader failed for {item.path.name}: {e}")

        if not cell_runs:
            if verbose:
                print(f"[cell] {cell_id}: no runs loaded; skipping pipeline.")
            continue

        if verbose:
            print(
                f"[pipeline] {cell_id}: processing {len(cell_runs)} run(s) "
                f"from {len(items)} item(s)"
            )
        run_pipeline(cell_runs, cfg, out_root)
        update_manifest_for_cell(manifest, cell_id, items)
        manifest_dirty = True
        processed_cells += 1

        if verbose:
            print(f"[summary] finished cell {cell_id} with {len(cell_runs)} run(s)")

    if manifest_dirty:
        save_manifest(out_root, manifest)

    if verbose:
        print(f"[done] processed {processed_cells} cell(s); skipped {skipped_cells} unchanged cell(s)")

if __name__ == "__main__":
    main()
