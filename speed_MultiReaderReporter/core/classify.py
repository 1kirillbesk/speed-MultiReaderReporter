# speed_MultiReaderReporter/core/classify.py
from __future__ import annotations
import numpy as np
import pandas as pd

# Default knobs (kept local so the signature stays the same).
# If you later want these in config.yaml, we can wire that easily.
_CHECKUP_KEYWORDS = ("cu", "glu")
_DURATION_THRESHOLD_MIN = 60         # < 1h → checkup (your existing rule)
_STEP_MIN_REQUIRED = 20              # like MAT: dismiss if max(step) < 20
_REQUIRE_19_AND_22 = True            # like MAT: look for long-step pattern 19..22
_SKIP_GLU = False                    # set True iff you ever want to skip GLU checkups

def is_checkup_run(program_name: str, df: pd.DataFrame) -> bool:
    """
    Robust checkup vs cycling classifier.

    Order of checks:
      1) Program-name keywords ('cu' / 'glu' unless _SKIP_GLU): checkup
      2) Step-aware fallback (if df has 'step_int'):
         - ignore 9999 sentinels, require max(step) >= 20
         - require step 19 and step 22 (configurable)
      3) Duration fallback: if run duration < 60 min → checkup
      4) Otherwise: cycling
    """
    p = (program_name or "").lower()

    # 1) Program-name rule
    if any(k in p for k in _CHECKUP_KEYWORDS if not (_SKIP_GLU and k == "glu")):
        return True

    if df is not None and not df.empty:
        # 2) Step-aware fallback (mirrors your MAT logic, but non-blocking)
        if "step_int" in df.columns:
            steps_raw = pd.to_numeric(df["step_int"], errors="coerce").to_numpy()
            # treat 9999 as NaN sentinels
            steps = np.where(steps_raw == 9999, np.nan, steps_raw)
            finite = steps[np.isfinite(steps)]
            if finite.size > 0 and int(np.nanmax(finite)) >= _STEP_MIN_REQUIRED:
                has19 = np.any(steps == 19)
                has22 = np.any(steps == 22)
                if (_REQUIRE_19_AND_22 and has19 and has22) or (not _REQUIRE_19_AND_22 and (has19 or has22)):
                    return True

        # 3) Duration fallback (your previous rule)
        dt = df["abs_time"].max() - df["abs_time"].min()
        if dt.total_seconds() < _DURATION_THRESHOLD_MIN * 60:
            return True

    # 4) Default: cycling
    return False
