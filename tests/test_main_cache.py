import os
import tempfile
import unittest
from pathlib import Path

from speed_MultiReaderReporter.main import (
    CACHE_MODE_FULL_RAW_PLOTS,
    CACHE_MODE_SUMMARY_ONLY,
    _empty_manifest,
    build_item_manifest,
    should_process_cell,
)
from speed_MultiReaderReporter.utils.detect import DetectedItem


class MainCacheTests(unittest.TestCase):
    def _make_item(self, root: Path, name: str, kind: str = "csv", content: str = "x") -> DetectedItem:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return DetectedItem(path=path.resolve(), kind=kind)

    def _write_summary(self, out_root: Path, cell_id: str, content: str = "summary") -> Path:
        summary_path = out_root / "cell_feature" / f"{cell_id}.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(content, encoding="utf-8")
        return summary_path

    def test_bootstrap_skips_when_summary_is_newer_than_inputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            out_root = root / "out"
            item = self._make_item(root, "raw=CELL_A=x=x=x=prog=x.csv")
            summary_path = self._write_summary(out_root, "CELL_A")

            raw_stat = item.path.stat()
            os.utime(
                summary_path,
                ns=(raw_stat.st_mtime_ns + 1_000_000, raw_stat.st_mtime_ns + 1_000_000),
            )

            manifest = _empty_manifest()
            should_process, reason, manifest_updated = should_process_cell(
                "CELL_A",
                [item],
                out_root,
                manifest,
                CACHE_MODE_SUMMARY_ONLY,
            )

            self.assertFalse(should_process)
            self.assertTrue(manifest_updated)
            self.assertIn("bootstrapped", reason)
            self.assertEqual(build_item_manifest([item]), manifest["cells"]["CELL_A"]["inputs"])

    def test_manifest_match_skips_cell_without_reprocessing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            out_root = root / "out"
            item = self._make_item(root, "raw=CELL_B=x=x=x=prog=x.csv")
            self._write_summary(out_root, "CELL_B")

            manifest = _empty_manifest()
            manifest["cells"]["CELL_B"] = {"inputs": build_item_manifest([item])}

            should_process, reason, manifest_updated = should_process_cell(
                "CELL_B",
                [item],
                out_root,
                manifest,
                CACHE_MODE_SUMMARY_ONLY,
            )

            self.assertFalse(should_process)
            self.assertFalse(manifest_updated)
            self.assertIn("unchanged", reason)

    def test_new_input_for_existing_cell_forces_rebuild(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            out_root = root / "out"
            item_old = self._make_item(root, "raw=CELL_C=x=x=x=prog1=x.csv")
            item_new = self._make_item(root, "raw=CELL_C=x=x=x=prog2=x.csv")
            self._write_summary(out_root, "CELL_C")

            manifest = _empty_manifest()
            manifest["cells"]["CELL_C"] = {"inputs": build_item_manifest([item_old])}

            should_process, reason, manifest_updated = should_process_cell(
                "CELL_C",
                [item_old, item_new],
                out_root,
                manifest,
                CACHE_MODE_SUMMARY_ONLY,
            )

            self.assertTrue(should_process)
            self.assertFalse(manifest_updated)
            self.assertIn("new, changed, or removed", reason)

    def test_full_raw_plots_mode_disables_cache_skip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            out_root = root / "out"
            item = self._make_item(root, "raw=CELL_D=x=x=x=prog=x.csv")
            self._write_summary(out_root, "CELL_D")

            manifest = _empty_manifest()
            manifest["cells"]["CELL_D"] = {"inputs": build_item_manifest([item])}

            should_process, reason, manifest_updated = should_process_cell(
                "CELL_D",
                [item],
                out_root,
                manifest,
                CACHE_MODE_FULL_RAW_PLOTS,
            )

            self.assertTrue(should_process)
            self.assertFalse(manifest_updated)
            self.assertIn("forces full raw reload", reason)


if __name__ == "__main__":
    unittest.main()
