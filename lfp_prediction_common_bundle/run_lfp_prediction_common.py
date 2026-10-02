from __future__ import annotations

import argparse
from pathlib import Path
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run lfp_prediction_common from the portable bundle.")
    parser.add_argument(
        "--interp-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "cell_feature",
        help="Directory containing the per-cell feature CSV files.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable matplotlib windows and only print a short result summary.",
    )
    return parser


def main() -> None:
    bundle_root = Path(__file__).resolve().parent
    module_root = bundle_root / "speed_MultiReaderReporter"
    if str(module_root) not in sys.path:
        sys.path.insert(0, str(module_root))

    from lfp_prediction_common import run_lfp_prediction_pipeline

    args = build_parser().parse_args()
    result = run_lfp_prediction_pipeline(
        interp_dir=args.interp_dir,
        show_plot=not args.no_plot,
    )
    metadata = result.get("metadata", {})
    print("lfp_prediction_common finished")
    print(f"interp_dir={Path(args.interp_dir).resolve()}")
    print(f"train_real_count={metadata.get('train_real_count')}")
    print(f"train_aug_count={metadata.get('train_aug_count')}")
    print(f"test_count={metadata.get('test_count')}")
    print(f"prediction_model={metadata.get('prediction_model')}")


if __name__ == "__main__":
    main()
