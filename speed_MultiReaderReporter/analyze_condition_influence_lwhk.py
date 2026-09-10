# speed_MultiReaderReporter/analyze_condition_influence_lwhk.py
"""Influence of the LWHK chamber conditions (fec, rest_min) on the features.

Same machinery as analyze_condition_influence_jgne.py - see that file for the
SOH definition and the interpolate-never-extrapolate rule - but keyed on the
LWHK condition table (document_conditions_lwhk.py).

Read this before trusting the numbers: every LWHK cell currently has only TWO
checkups, and the deepest SoH any of them reaches is about 0.987. A target of
0.98 is therefore outside every cell's measured range and the script will
correctly refuse it. Around 0.995 roughly two thirds of the cells qualify, and
even there each cell contributes a straight line through two points.

Usage:
    python speed_MultiReaderReporter/analyze_condition_influence_lwhk.py --report-only
    python speed_MultiReaderReporter/analyze_condition_influence_lwhk.py --target-soh 0.995
"""
from __future__ import annotations

from analyze_condition_influence_jgne import run

LWHK_CONDITIONS = ["fec", "rest_min", "ref"]

def main():
    run(family="LWHK",
        pattern="SPEED_LWHK_*",
        cond_csv_name="condition_overview_SPEED_LWHK.csv",
        conditions=LWHK_CONDITIONS,
        prefer_col="fec",
        argv_desc=__doc__)

if __name__ == "__main__":
    main()
