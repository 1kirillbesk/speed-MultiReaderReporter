# speed_MultiReaderReporter/main.py
from __future__ import annotations
from pathlib import Path
import sys
import yaml
import matplotlib.pyplot as plt

from loaders import csvzip_loader, mat_loader, pkl_loader
from utils.detect import discover_inputs
from utils.processing_manifest import ProcessingManifest
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
    cfg = load_config(here / "config.yaml") #config602

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
    name_contains = cfg.get("input", {}).get("name_contains", [])
    needles = [n.lower() for n in name_contains]
    if needles:
        detected = [
            d for d in detected
            if any(n in d.path.name.lower() for n in needles)
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

    processing_cfg = cfg.get("processing", {})
    processing_mode = str(processing_cfg.get("mode", "all")).strip().lower()
    if processing_mode not in {"all", "unprocessed"}:
        raise ValueError("processing.mode must be 'all' or 'unprocessed'")
    manifest_path = Path(processing_cfg.get("manifest", out_root / "processing_manifest.json"))
    if not manifest_path.is_absolute():
        manifest_path = out_root / manifest_path
    manifest = ProcessingManifest(manifest_path)

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
    pending_by_cell: dict[str, list] = {}
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
        if not manifest.is_processed(item.path, item.kind, cell_id):
            pending_by_cell.setdefault(cell_id, []).append(item)

    if processing_mode == "unprocessed":
        if not pending_by_cell:
            print(f"[INFO] No new or modified inputs; manifest is up to date: {manifest_path}")
            sys.exit(0)
        selected_cells = set(pending_by_cell)
        if verbose:
            print(
                f"[processing] mode=unprocessed; {sum(map(len, pending_by_cell.values()))} "
                f"new/modified input(s) across {len(selected_cells)} cell(s). "
                "Reprocessing each affected cell using all its discovered inputs."
            )
    else:
        selected_cells = set(items_by_cell)
        if verbose:
            print("[processing] mode=all; all discovered inputs and cells will be processed.")

    if not selected_cells:
        if verbose:
            print("[INFO] No runs loaded; exiting without processing pipeline.")
        sys.exit(0)

    if verbose:
        selected_items = sum(len(items_by_cell[cell]) for cell in selected_cells)
        print(
            f"[grouping] processing {selected_items} input(s) "
            f"across {len(selected_cells)} cell(s): {', '.join(sorted(selected_cells))}"
        )

    for cell_id in sorted(selected_cells):
        items = items_by_cell[cell_id]
        # Load all raw inputs for the same cell together so one pipeline run handles the aggregated data.
        cell_runs = []
        loaded_paths: set[str] = set()
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
                loaded_paths.add(str(item.path.resolve()).casefold())
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

        # Record source files only after this cell made it through pipeline execution.
        manifest.record(
            (item.path, item.kind, cell_id)
            for item in items
            if str(item.path.resolve()).casefold() in loaded_paths
        )
        manifest.save()

        if verbose:
            print(f"[summary] finished cell {cell_id} with {len(cell_runs)} run(s)")

if __name__ == "__main__":
    main()
