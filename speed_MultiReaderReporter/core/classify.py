# speed_MultiReaderReporter/core/classify.py
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from typing import Iterable

# ----- defaults (used if configure_from_config isn't called) -----
_CHECKUP_KEYWORDS: tuple[str, ...] = ("cu", "glu", "rpt")
_CYCLING_KEYWORDS: tuple[str, ...] = ("cyc",)
_DURATION_THRESHOLD_MIN: int = 60
_STEP_MIN_REQUIRED: int = 20
_REQUIRE_19_AND_22: bool = True
_SKIP_GLU: bool = False

_LOG = logging.getLogger(__name__)

def configure_from_config(cfg: dict) -> None:
    """
    Optional: call once at startup to override defaults from config.yaml.
    Keeps is_checkup_run signature unchanged.
    """
    global _CHECKUP_KEYWORDS, _CYCLING_KEYWORDS
    global _DURATION_THRESHOLD_MIN, _STEP_MIN_REQUIRED, _REQUIRE_19_AND_22, _SKIP_GLU

    # reset to defaults each call so repeated invocations do not accumulate
    _CHECKUP_KEYWORDS = ("cu", "glu", "rpt")
    _CYCLING_KEYWORDS = ("cyc",)
    _DURATION_THRESHOLD_MIN = 60
    _STEP_MIN_REQUIRED = 20
    _REQUIRE_19_AND_22 = True
    _SKIP_GLU = False
    cls = (cfg or {}).get("classification", {}) if cfg else {}
    # keywords
    checkup_kws = cls.get("checkup_keywords", None)
    if isinstance(checkup_kws, Iterable) and not isinstance(checkup_kws, (str, bytes)):
        _CHECKUP_KEYWORDS = tuple(str(k).lower() for k in checkup_kws)

    cycling_kws = cls.get("cycling_keywords", None)
    if isinstance(cycling_kws, Iterable) and not isinstance(cycling_kws, (str, bytes)):
        _CYCLING_KEYWORDS = tuple(str(k).lower() for k in cycling_kws)
    # thresholds / flags
    _DURATION_THRESHOLD_MIN = int(cls.get("duration_threshold_minutes", _DURATION_THRESHOLD_MIN))
    _STEP_MIN_REQUIRED      = int(cls.get("step_min_required", _STEP_MIN_REQUIRED))
    _REQUIRE_19_AND_22      = bool(cls.get("require_steps_19_22", _REQUIRE_19_AND_22))
    _SKIP_GLU               = bool(cls.get("skip_glu", _SKIP_GLU))

def is_checkup_run(program_name: str, df: pd.DataFrame) -> bool:
    """
    Robust checkup vs cycling classifier (config-driven).

    Order:
      0) Cycling keywords (from config) override everything else.
      1) Program keywords (from config; 'glu' can be skipped via skip_glu)
      2) Step-aware fallback (if 'step_int' exists): ignore 9999, require max(step) >= step_min_required,
         and (optionally) presence of steps 19 & 22.
      3) Duration fallback: if duration < duration_threshold_minutes → checkup
      4) Else: cycling
    """
    p = (program_name or "").lower()

    # 0) Explicit cycling override
    for k in _CYCLING_KEYWORDS:
        if k in p:
            _LOG.debug("forced cycling by program name keyword '%s'", k)
            return False

    # 1) Program-name rule (checkup)
    for k in _CHECKUP_KEYWORDS:
        if _SKIP_GLU and k == "glu":
            continue
        if k in p:
            _LOG.debug("checkup by program name keyword '%s'", k)
            return True

    if df is not None and not df.empty:
        # 2) Step-aware fallback
        if "step_int" in df.columns:
            steps_raw = pd.to_numeric(df["step_int"], errors="coerce").to_numpy()
            steps = np.where(steps_raw == 9999, np.nan, steps_raw)  # treat 9999 as sentinel
            finite = steps[np.isfinite(steps)]
            if finite.size > 0 and int(np.nanmax(finite)) >= _STEP_MIN_REQUIRED:
                has19 = np.any(steps == 19)
                has22 = np.any(steps == 22)
                if (_REQUIRE_19_AND_22 and has19 and has22) or (not _REQUIRE_19_AND_22 and (has19 or has22)):
                    _LOG.debug("checkup by step rule (19/22=%s/%s)", has19, has22)
                    return True

        # 3) Duration fallback
        dt = df["abs_time"].max() - df["abs_time"].min()
        if dt.total_seconds() < _DURATION_THRESHOLD_MIN * 60:
            _LOG.debug("checkup by duration under %d min", _DURATION_THRESHOLD_MIN)
            return True

    # 4) Default: cycling
    _LOG.debug("defaulting to cycling")
    return False
