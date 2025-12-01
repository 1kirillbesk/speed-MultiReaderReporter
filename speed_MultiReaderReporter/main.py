# speed_MultiReaderReporter/main.py
from __future__ import annotations
from pathlib import Path
import sys
import yaml

from loaders import csvzip_loader, mat_loader, pkl_loader
from utils.detect import discover_inputs
from core.pipeline import run_pipeline

# --- relative paths ---
here = Path(__file__).resolve().parent
sys.path.append(str(here))
sys.path.append(str(here / "core"))
sys.path.append(str(here / "loaders"))
sys.path.append(str(here / "utils"))

def load_config(cfg_path: Path) -> dict:
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    # ---------- config ----------
    here = Path(__file__).resolve().parent
    cfg = load_config(here / "config.yaml")

    in_path = Path(cfg["input"]["path"]).resolve()
    recurse = bool(cfg["input"].get("recurse", True))
    out_root = Path(cfg["output"]["root"]).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    verbose = bool(cfg.get("logging", {}).get("verbose", True))
    if verbose:
        print(f"[cfg] input={in_path} (recurse={recurse})")
        print(f"[cfg] output={out_root}")

    # ---------- discover ----------
    detected = discover_inputs(in_path, recurse=recurse)
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

    for cell_id, items in sorted(items_by_cell.items()):
        # Load all raw inputs for the same cell together so one pipeline run handles the aggregated data.
        cell_runs = []
        if verbose:
            print(f"[cell] {cell_id}: loading {len(items)} item(s)")
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

        if verbose:
            print(f"[summary] finished cell {cell_id} with {len(cell_runs)} run(s)")

if __name__ == "__main__":
    main()
