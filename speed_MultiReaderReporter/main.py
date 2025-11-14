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
        "pkl": pkl_loader.load,
    }

    processed_cells: set[str] = set()
    for item in detected:
        loader = registry.get(item.kind)
        if loader is None:
            if verbose:
                print(f"[skip] no loader for {item.kind}: {item.path.name}")
            continue
        if verbose:
            print(f"[load] {item.kind:6}  {item.path.name}")
        try:
            runs = loader(item.path, cfg, out_root)
            if not runs:
                continue
            if verbose:
                cells = sorted({r.cell for r in runs})
                print(
                    f"[pipeline] processing {len(runs)} run(s) "
                    f"across {len(cells)} cell(s): {', '.join(cells)}"
                )
            run_pipeline(runs, cfg, out_root)
            processed_cells.update(r.cell for r in runs)
            del runs
        except Exception as e:
            print(f"[WARN] loader failed for {item.path.name}: {e}")
    if verbose:
        print(
            f"[summary] processed cells: {len(processed_cells)}"
            + (f" → {sorted(processed_cells)}" if processed_cells else "")
        )

if __name__ == "__main__":
    main()
