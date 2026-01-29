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

def is_checkup_run(program_name: str, df: pd.DataFrame, cu_keywords) -> bool:
    p = (program_name or "").lower()

    # 0) Explicit cycling override
    if any(k in p for k in cu_keywords):
        return True

    # 4) Default: cycling
    _LOG.debug("defaulting to cycling")
    return False

def split_total_list(cell, total_list, cfg):
    """
    Simplest version:

    - While-loop through total_list (each item is (df, label))
    - If label contains ANY rpt_keywords -> checkup_list
      else -> cycling_list
    """
    checkup_list, cycling_list = [], []

    rpt_keywords = tuple(cfg["classification"]["rpt_keywords"])

    i = 0
    n = len(total_list)
    while i < n:
        df, label = total_list[i]
        label_lower = (label or "").lower()

        if any(k in label_lower for k in rpt_keywords):
            checkup_list.append((df, label))
        else:
            cycling_list.append((df, label))

        i += 1

    return checkup_list, cycling_list


# import matplotlib.pyplot as plt
# plt.plot(processed_total_list[0][0]['voltage_V'])
# plt.show()