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
    total_list: list of tuples -> (df, label)

    Pre-processing logic (runs before the main while-loop):
      - ONLY if "homocomp" in label (case-insensitive)
      - and df contains (step_int == 16) AND (state contains "SAVE")
      - then replace that single (df,label) with TWO elements:
            1) df where step_int > 16   (FIRST)
            2) df where step_int <= 16  (AFTER)
        (label stays identical for both)
    """
    checkup_list, cycling_list = [], []

    # keywords
    rpt_keywords = tuple(cfg["classification"]["rpt_keywords"])
    cu_keyword = cfg["classification"]["cu_keyword"]

    # ------------------------------------------------------------------
    # NEW: preprocess total_list (split homocomp items if condition holds)
    # ------------------------------------------------------------------
    processed_total_list = []
    for df, label in total_list:
        label_lower = (label or "").lower()

        # ONLY apply logic to labels containing "homocomp"
        if "homocomp" in label_lower:
            try:
                if "step_int" in df.columns and "state" in df.columns:
                    cond_step16 = (df["step_int"] == 16).any()
                    cond_save = df["state"].astype(str).str.contains("SAVE", na=False).any()

                    if cond_step16 and cond_save:
                        # SPLIT LOGIC (order matters!)
                        df_high = df[df["step_int"] > 16].copy()
                        df_low  = df[df["step_int"] <= 16].copy()

                        # append HIGH first, then LOW
                        if not df_high.empty:
                            processed_total_list.append((df_high, label))
                        if not df_low.empty:
                            processed_total_list.append((df_low, label))

                        continue  # done with this element
            except Exception as e:
                logging.exception(f"[{cell}] homocomp split failed for '{label}': {e}")
                processed_total_list.append((df, label))
                continue

        # default: keep as-is
        processed_total_list.append((df, label))

    total_list = processed_total_list
    n = len(total_list)

    # --- First pass: estimate how many "rpt" checkups exist (for mode decision) ---
    rpt_count = sum(
        1 for _, label in total_list
        if any(k in (label or "").lower() for k in rpt_keywords)
    )
    pairing_mode = rpt_count < (n / 2)

    i = 0
    while i < n:
        df, label = total_list[i]
        label_lower = (label or "").lower()

        # -------------------------
        # MODE A: Pair lw_rpt with immediately previous item
        # -------------------------
        if pairing_mode:
            if any(k in label_lower for k in rpt_keywords):
                if i == 0:
                    logging.warning(
                        f"[{cell}] '{label}' is SAM_rpt but has no previous item; putting into checkup as-is."
                    )
                    checkup_list.append((df, label))
                    i += 1
                    continue

                prev_df, prev_label = total_list[i - 1]

                try:
                    combined_df = pd.concat([prev_df, df], ignore_index=True, sort=False)
                    combined_label = f"{prev_label}__PLUS__{label}"
                    checkup_list.append((combined_df, combined_label))
                except Exception as e:
                    logging.exception(
                        f"[{cell}] concat failed for prev '{prev_label}' + rpt '{label}': {e}"
                    )
                    cycling_list.append((prev_df, prev_label))
                    checkup_list.append((df, label))

                if cycling_list and cycling_list[-1][1] == prev_label and cycling_list[-1][0] is prev_df:
                    cycling_list.pop()

                i += 1
                continue

            cycling_list.append((df, label))
            i += 1
            continue

        # -------------------------
        # MODE B: Normal logic
        # -------------------------
        if cu_keyword in label_lower:
            if (df["step_int"] == 34).any():
                checkup_list.append((df, label))
        else:
            cycling_list.append((df, label))

        i += 1

    # --- Cleanup for pairing mode ---
    if pairing_mode:
        indices_before_rpt = set()
        for idx in range(1, n):
            lbl = (total_list[idx][1] or "").lower()
            if any(k in lbl for k in rpt_keywords):
                indices_before_rpt.add(idx - 1)

        cycling_list = [
            (df, lbl)
            for idx, (df, lbl) in enumerate(total_list)
            if idx not in indices_before_rpt
            and not any(k in (lbl or "").lower() for k in rpt_keywords)
        ]

    return checkup_list, cycling_list


# import matplotlib.pyplot as plt
# plt.plot(processed_total_list[0][0]['voltage_V'])
# plt.show()