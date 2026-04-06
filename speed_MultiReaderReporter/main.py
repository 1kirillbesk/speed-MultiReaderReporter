# speed_MultiReaderReporter/main.py
from __future__ import annotations
from pathlib import Path
import sys
import yaml
import matplotlib.pyplot as plt
import re

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


def process_items_by_cell(
    items_by_cell: dict[str, list],
    *,
    cfg: dict,
    out_root: Path,
    registry: dict,
    verbose: bool,
    tag: str,
):
    out_root.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"\n[{tag}] output root: {out_root}")
        print(f"[{tag}] cells: {', '.join(sorted(items_by_cell)) if items_by_cell else '[none]'}")

    for cell_id, items in sorted(items_by_cell.items()):
        cell_runs = []
        if verbose:
            print(f"[{tag}][cell] {cell_id}: loading {len(items)} item(s)")
        for item in items:
            loader = registry[item.kind]
            if verbose:
                print(f"  [{tag}][load] {item.kind:6} {item.path.name}")
            try:
                runs = loader(item.path, cfg, out_root)
                if not runs:
                    continue
                cell_runs.extend(runs)
            except Exception as e:
                print(f"[WARN][{tag}] loader failed for {item.path.name}: {e}")

        if not cell_runs:
            if verbose:
                print(f"[{tag}][cell] {cell_id}: no runs loaded; skipping pipeline.")
            continue

        if verbose:
            print(
                f"[{tag}][pipeline] {cell_id}: processing {len(cell_runs)} run(s) "
                f"from {len(items)} item(s)"
            )
        run_pipeline(cell_runs, cfg, out_root)

        if verbose:
            print(f"[{tag}][summary] finished cell {cell_id} with {len(cell_runs)} run(s)")


def main():
    # ---------- config ----------
    here = Path(__file__).resolve().parent
    cfg = load_config(here / "private/config_lw.yaml")

    in_path = Path(cfg["input"]["path"]).resolve()
    recurse = bool(cfg["input"].get("recurse", True))

    # split output roots (prefer explicit config keys)
    output_cfg = cfg.get("output", {})
    root_lw_cfg = output_cfg.get("root_lw")
    root_lwhk_cfg = output_cfg.get("root_lwhk")
    if root_lw_cfg and root_lwhk_cfg:
        out_root_lw = Path(root_lw_cfg).resolve()
        out_root_lwhk = Path(root_lwhk_cfg).resolve()
    else:
        out_root_cfg = Path(output_cfg["root"]).resolve()
        out_parent = out_root_cfg.parent
        out_root_lw = out_parent / "out_lw_new"
        out_root_lwhk = out_parent / "out_lwhk_new"
    out_root_lw.mkdir(parents=True, exist_ok=True)
    out_root_lwhk.mkdir(parents=True, exist_ok=True)

    verbose = bool(cfg.get("logging", {}).get("verbose", True))
    if verbose:
        print(f"[cfg] input={in_path} (recurse={recurse})")
        print(f"[cfg] output LW={out_root_lw}")
        print(f"[cfg] output LWHK={out_root_lwhk}")

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
        print(f"[detector] found {sum(kinds.values())} inputs -> {kinds}")

    # ---------- loader registry ----------
    registry = {
        "csvzip": csvzip_loader.load,
        "csv": csvzip_loader.load,
        "mat": mat_loader.load,
        "pkl": pkl_loader.load,
    }
    cell_resolver = {
        "csvzip": csvzip_loader.infer_cell_from_path,
        "csv": csvzip_loader.infer_cell_from_path,
        "mat": mat_loader.infer_cell_from_path,
        "pkl": pkl_loader.infer_cell_from_path,
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

    # split cells
    items_lwhk = {}
    items_lw = {}
    for cell_id, items in items_by_cell.items():
        c = cell_id.lower()
        if "lwhk" in c:
            items_lwhk[cell_id] = items
        elif "lw" in c:
            items_lw[cell_id] = items
        elif verbose:
            print(f"[skip] neither LW nor LWHK: {cell_id}")

    if not items_lw and not items_lwhk:
        print("[INFO] No LW/LWHK cells found after grouping.")
        sys.exit(0)

    if items_lw:
        process_items_by_cell(
            items_lw,
            cfg=cfg,
            out_root=out_root_lw,
            registry=registry,
            verbose=verbose,
            tag="LW",
        )
    else:
        print("[INFO] No LW cells found.")

    if items_lwhk:
        process_items_by_cell(
            items_lwhk,
            cfg=cfg,
            out_root=out_root_lwhk,
            registry=registry,
            verbose=verbose,
            tag="LWHK",
        )
    else:
        print("[INFO] No LWHK cells found.")


if __name__ == "__main__":
    main()
