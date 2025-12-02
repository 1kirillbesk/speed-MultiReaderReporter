import pandas as pd
from pathlib import Path
import tempfile
import unittest

from speed_MultiReaderReporter.core.classify import configure_from_config, is_checkup_run
from speed_MultiReaderReporter.core.model import RunRecord
from speed_MultiReaderReporter.core.pipeline import run_pipeline
from speed_MultiReaderReporter.loaders.csvzip_loader import _segment_by_procedure


def _make_df(step_values, start="2024-01-01 00:00:00", freq="10min"):
    times = pd.date_range(start=start, periods=len(step_values), freq=freq)
    return pd.DataFrame(
        {
            "abs_time": times,
            "current_A": [0.1 * (i + 1) for i in range(len(step_values))],
            "voltage_V": [3.5] * len(step_values),
            "step_int": step_values,
        }
    )


class ClassificationTests(unittest.TestCase):
    def test_cycling_keyword_overrides_checkup_heuristics(self):
        configure_from_config({"classification": {"cycling_keywords": ["cyc"]}})
        df = _make_df([1, 19, 22], freq="15min")
        # Would qualify as checkup by step pattern + short duration, but name contains cycling keyword
        self.assertFalse(is_checkup_run("rul_inhomoSAM_cyc_1C_4OSOC100", df))

    def test_checkup_keyword_without_cycling_match(self):
        configure_from_config({})
        df = _make_df([1, 19, 22], freq="45min")
        self.assertTrue(is_checkup_run("reference_cu_profile", df))


class PipelineGroupingTests(unittest.TestCase):
    def test_pipeline_outputs_per_cell_with_multiple_runs(self):
        configure_from_config({})
        checkup_df = _make_df([1, 19, 22], freq="30min")
        cycling_df = _make_df([1, 2, 3], freq="30min")

        runs = [
            RunRecord(cell="CELL_A", program="demo_cu", df=checkup_df, source_path=Path("file1.csv"), loader="csvzip"),
            RunRecord(cell="CELL_A", program="demo_cyc", df=cycling_df, source_path=Path("file2.csv"), loader="csvzip"),
        ]

        cfg = {
            "reports": {"format": "csv", "mat_variable": "report"},
            "legend": {"ncol": 1},
            "classification": {},
            "checkup_grouping": {"mode": "off"},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            out_root = Path(tmpdir) / "out"
            run_pipeline(runs, cfg, out_root)

            cell_dir = out_root / "CELL_A"
            total_report = cell_dir / "total" / "report.csv"
            cycling_report = cell_dir / "cycling" / "report.csv"
            checkup_report = cell_dir / "checkup" / "report.csv"

            self.assertTrue(total_report.exists(), "total report missing")
            self.assertTrue(cycling_report.exists(), "cycling report missing")
            self.assertTrue(checkup_report.exists(), "checkup report missing")

            # Ensure both runs appear in the total report (plus TOTAL row)
            df_total = pd.read_csv(total_report)
            self.assertTrue({"demo_cu", "demo_cyc", "TOTAL"}.issubset(set(df_total["program"].tolist())))

            # Ensure cycling override preserved the cycling run
            df_cyc = pd.read_csv(cycling_report)
            self.assertIn("demo_cyc", df_cyc["program"].tolist())

            # Ensure checkup run grouped correctly
            df_chk = pd.read_csv(checkup_report)
            self.assertIn("demo_cu", df_chk["program"].tolist())


class ProcedureSegmentationTests(unittest.TestCase):
    def _df_with_procedure(self, procs):
        times = pd.date_range(start="2024-01-01 00:00:00", periods=len(procs), freq="1min")
        return pd.DataFrame({
            "abs_time": times,
            "current_A": [0.1] * len(procs),
            "procedure": procs,
        })

    def test_pulse_procedures_are_attached_to_active(self):
        procs = [
            "rul_CU_main",
            "rul_CU_main",
            "rul_Pulse_SAM",
            "rul_Pulse_SAM",
            "rul_CU_main",
        ]
        df = self._df_with_procedure(procs)

        segments = _segment_by_procedure(df, "CELL_X", "program_stub", Path("file.csv"), "csvzip", cfg={})

        self.assertEqual(1, len(segments))
        self.assertEqual("rul_CU_main", segments[0].program)
        self.assertEqual(len(procs), len(segments[0].df))

    def test_multiple_programs_with_pulses_split_cleanly(self):
        procs = ["checkup", "checkup", "helper_pulse", "cycling", "cycling"]
        df = self._df_with_procedure(procs)

        cfg = {"classification": {"procedure_pulse_keywords": ["pulse"]}}
        segments = _segment_by_procedure(df, "CELL_Y", "program_stub", Path("file.csv"), "csvzip", cfg)

        self.assertEqual(2, len(segments))
        self.assertEqual("checkup", segments[0].program)
        self.assertEqual(3, len(segments[0].df))
        self.assertEqual("cycling", segments[1].program)
        self.assertEqual(2, len(segments[1].df))
