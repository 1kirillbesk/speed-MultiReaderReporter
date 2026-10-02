# speed_MultiReaderReporter/analyze_condition_influence_bald.py
"""Influence of the BALD / Samsung 35E conditions on the features, at a common SoH.

Same machinery as analyze_condition_influence_jgne.py - see that file for the
SOH definition and the interpolate-never-extrapolate rule - keyed on the BALD
condition table written by document_conditions_bald.py.

BALD carries BOTH capacities in the same checkup (rul_SAM_CU_varCapa), so SoH can
be defined either way and no separate extraction is needed:

    --soh-col cap_ocv_dis   step 19, C/10 OCV discharge   (default)
    --soh-col cap_dis       step  9, C/2 discharge

The two are different quantities - the C/2 one includes polarisation - so pick
one per run and keep the outputs apart with --prefix.

Usage:
    python speed_MultiReaderReporter/analyze_condition_influence_bald.py --report-only
    python speed_MultiReaderReporter/analyze_condition_influence_bald.py --target-soh 0.95
    python speed_MultiReaderReporter/analyze_condition_influence_bald.py \
        --soh-col cap_dis --prefix bald_c2 --target-soh 0.95
"""
from __future__ import annotations

from pathlib import Path

import analyze_condition_influence_jgne as core
from analyze_condition_influence_jgne import run

BALD_CONDITIONS = ["soc_start", "soc_end", "soc_mean", "soc_dod", "c_rate", "temp_C", "dyn", "sdod", "scur",
                   "pau", "short", "pulsecyc", "pause_h", "dod", "n_cyc"]

def main():
    # BALD lives under its own output root
    core.DEFAULT_OUT_DIR = Path("E:/download/BALD/out_bald")
    run(family="BALD",
        pattern="BALD_35E_*",
        cond_csv_name="condition_overview_BALD_35E.csv",
        conditions=BALD_CONDITIONS,
        prefer_col="c_rate",
        argv_desc=__doc__)

if __name__ == "__main__":
    main()
