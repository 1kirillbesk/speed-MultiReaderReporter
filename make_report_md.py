# make_report_md.py
"""Compute the statistics and write report.md for the JGNE / LWHK condition study.

Run make_report_figures.py first; this reads the cell summaries it wrote and the
figures/ folder it populated, then emits a self-contained markdown report that
links to those figures with relative paths.

Statistics used, and why:
  * Spearman rho        rank correlation, robust to the non-linear ageing curves
                        and to the two-level conditions common here.
  * exact stratified    for the SOC-matched temperature design, every allocation
    permutation test    of cells to arms within each stratum is enumerated, which
                        is exact at n=2 per arm where asymptotic tests are invalid.
  * Benjamini-Hochberg  applied across all reported rho tests per dataset.
  * bootstrap SHAP      the cells are resampled with replacement 200 times, the
                        model refitted on each, and exact TreeSHAP recomputed for
                        every original cell. The cloud is model uncertainty, not
                        extra data - no synthetic points are generated.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MATH_MD = {
    "temp_C": "$T$", "pause_h": "$t_\\mathrm{pause}$", "has_pulse": "pulse",
    "low_soc": "lowSOC", "soc_mean": "$\\overline{\\mathrm{SOC}}$",
    "soc_dod": "$\\mathrm{DOD}$", "fec": "$\\mathrm{FEC}$",
    "rest_min": "$t_\\mathrm{rest}$",
    "fade_per_100Ah": "$\\dot{q}_Q$", "fade_per_week": "$\\dot{q}_t$",
    "k_sqrt_Ah": "$k_{\\sqrt{Q}}$", "soh_end": "$\\mathrm{SOH}_\\mathrm{end}$",
    "weeks": "$t_\\mathrm{end}$", "throughput_cum": "$Q_\\mathrm{thr}$",
}
def mm(k): return MATH_MD.get(k, f"`{k}`")

def bh(p):
    p = np.asarray(p, float); n = len(p)
    o = np.argsort(p); q = np.empty(n); prev = 1.0
    for rank, idx in enumerate(o[::-1], 1):
        prev = min(prev, p[idx] * n / (n - rank + 1)); q[idx] = prev
    return q

def corr_table(tbl, conds, targets):
    rows = []
    for c in conds:
        for t in targets:
            x = pd.to_numeric(tbl[c], errors="coerce")
            y = pd.to_numeric(tbl[t], errors="coerce")
            k = x.notna() & y.notna()
            if k.sum() < 4 or x[k].nunique() < 2 or y[k].nunique() < 2:
                continue
            r, p = spearmanr(x[k], y[k])
            rows.append({"condition": c, "target": t, "n": int(k.sum()),
                         "rho": r, "p": p})
    d = pd.DataFrame(rows)
    if not d.empty:
        d["q"] = bh(d.p.to_numpy())
    return d.sort_values("p") if not d.empty else d

def exact_strat(strata):
    """strata: list of {'lo':[...], 'hi':[...]} value lists. Exact two-sided p."""
    obs = float(np.mean([np.mean(s["hi"]) - np.mean(s["lo"]) for s in strata]))
    pools = [s["lo"] + s["hi"] for s in strata]
    k = len(pools[0]) // 2
    combos = [list(itertools.combinations(range(len(p)), k)) for p in pools]
    diffs = []
    for pick in itertools.product(*combos):
        d = []
        for vals, idx in zip(pools, pick):
            hi = [vals[i] for i in idx]
            lo = [vals[i] for i in range(len(vals)) if i not in idx]
            d.append(np.mean(hi) - np.mean(lo))
        diffs.append(np.mean(d))
    diffs = np.asarray(diffs)
    return obs, float(np.mean(np.abs(diffs) >= abs(obs))), len(diffs)

JG = Path("E:/download/jgne/out_jgne")
TEMP_DESIGN = {
    "30 → 70 %": {"lo": ["SPEED_JGNE_cycle_77", "SPEED_JGNE_cycle_78"],
                  "hi": ["SPEED_JGNE_cycle_95", "SPEED_JGNE_cycle_96"]},
    "40 → 100 %": {"lo": ["SPEED_JGNE_cycle_75", "SPEED_JGNE_cycle_76"],
                   "hi": ["SPEED_JGNE_cycle_97", "SPEED_JGNE_cycle_98"]},
}

def jgne_temp_stats(tbl):
    idx = tbl.set_index("cell")
    out = {}
    for key in ("fade_per_100Ah", "fade_per_week", "soh_end"):
        strata = []
        ok = True
        for win, arms in TEMP_DESIGN.items():
            try:
                strata.append({"lo": [float(idx.loc[c, key]) for c in arms["lo"]],
                               "hi": [float(idx.loc[c, key]) for c in arms["hi"]]})
            except Exception:
                ok = False
        if ok and strata:
            out[key] = exact_strat(strata)
    per_arm = {}
    for win, arms in TEMP_DESIGN.items():
        for T, k in ((25, "lo"), (40, "hi")):
            vals = [float(idx.loc[c, "fade_per_100Ah"]) for c in arms[k]
                    if c in idx.index and pd.notna(idx.loc[c, "fade_per_100Ah"])]
            if vals:
                per_arm[(win, T)] = (float(np.mean(vals)), vals)
    return out, per_arm

def fmt(d, cols, n=10):
    if d.empty:
        return "_no rows_\n"
    x = d.head(n).copy()
    hdr = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = ""
    for _, r in x.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in ("condition", "target"):
                cells.append(mm(v))
            elif isinstance(v, float):
                cells.append(f"{v:.3f}" if abs(v) >= 1e-3 or v == 0 else f"{v:.2e}")
            else:
                cells.append(str(v))
        body += "| " + " | ".join(cells) + " |\n"
    return hdr + body

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, default=Path("./report_jgne_lwhk"))
    args = ap.parse_args()
    D = args.dest
    jg = pd.read_csv(D / "jgne_cell_summary.csv")
    lw = pd.read_csv(D / "lwhk_cell_summary.csv")

    TARGETS = ["fade_per_100Ah", "fade_per_week", "k_sqrt_Ah", "soh_end", "weeks"]
    jg_c = [c for c in ("temp_C", "pause_h", "has_pulse", "low_soc", "soc_mean")
            if c in jg.columns and pd.to_numeric(jg[c], errors="coerce").nunique() > 1]
    lw_c = [c for c in ("fec", "rest_min")
            if c in lw.columns and pd.to_numeric(lw[c], errors="coerce").nunique() > 1]
    jg_tab = corr_table(jg, jg_c, [t for t in TARGETS if t in jg.columns])
    lw_tab = corr_table(lw, lw_c, [t for t in TARGETS if t in lw.columns])
    tstats, per_arm = jgne_temp_stats(jg)

    jg_tab.to_csv(D / "jgne_correlations.csv", index=False)
    lw_tab.to_csv(D / "lwhk_correlations.csv", index=False)

    md = []
    A = md.append
    A("# Influence of the experimental conditions on cell ageing\n")
    A("**Datasets:** JGNE (LFP, 1.8 Ah nominal) and LWHK (LFP). "
      "One row per cell; conditions are decoded from the test programme label.\n")
    A("## 1. Quantities\n")
    A("| symbol | definition |\n|---|---|")
    A("| $\\mathrm{SOH}$ | capacity normalised to that cell's first checkup |")
    A("| $Q_\\mathrm{thr}$ | cumulative cycling charge throughput [Ah], $\\int\\|I\\|\\,dt$ |")
    A("| $\\dot{q}_Q$ | fade rate per throughput, $(1-\\mathrm{SOH}_\\mathrm{end})/Q_\\mathrm{thr}$ [% / 100 Ah] |")
    A("| $\\dot{q}_t$ | fade rate per time, $(1-\\mathrm{SOH}_\\mathrm{end})/t$ [% / week] |")
    A("| $k_{\\sqrt{Q}}$ | $b$ in $\\mathrm{SOH}=1-b\\sqrt{Q_\\mathrm{thr}}$ |")
    A("| $T$, $t_\\mathrm{pause}$, $\\overline{\\mathrm{SOC}}$, $\\mathrm{FEC}$, $t_\\mathrm{rest}$ | "
      "temperature, pause, mean SOC, full equivalent cycles, rest time |")
    A("")
    A("JGNE capacity is the **C/10 OCV discharge** ($C_\\mathrm{OCV,dis}$, step 19 of "
      "`rul_JGNE_RPT_0C1`). LWHK capacity is the **C/2 discharge** ($C_\\mathrm{dis}$, "
      "step 6 of `rul_eka_CU`, 0.900 A against $C_\\mathrm{nom}=1.8$ Ah). They are "
      "different measurements and are never placed on a shared axis.\n")

    A("## 2. JGNE\n")
    A(f"{len(jg)} cells.\n")
    A("![JGNE trajectories by pause](figures/jgne_trajectory_pause_h.png)\n")
    A("*Figure 1 — JGNE capacity trajectories coloured by $t_\\mathrm{pause}$.*\n")
    A("![JGNE trajectories by temperature](figures/jgne_trajectory_temp_C.png)\n")
    A("*Figure 2 — the same cells coloured by $T$. The 40 °C cells ran roughly "
      "twice as long, so the endpoint comparison is confounded; Figure 3 controls for it.*\n")

    A("### 2.1 Temperature, SOC-matched\n")
    A("Only the `_ref` cells are true controls for the `_40T` cells: same SOC window, "
      "no pause, no pulse, no lowSOC. Two cells per arm.\n")
    A("![JGNE temperature matched](figures/jgne_temperature_matched.png)\n")
    A("*Figure 3 — matched temperature comparison. Left/middle: $\\mathrm{SOH}$ against "
      "$Q_\\mathrm{thr}$, which removes the unequal test duration. Right: $\\dot{q}_Q$ per arm.*\n")
    if per_arm:
        A("| SOC window | $T$ [°C] | $\\dot{q}_Q$ [% / 100 Ah] |\n|---|---|---|")
        for (win, T), (mu, vals) in sorted(per_arm.items()):
            A(f"| {win} | {T} | {mu:.3f}  ({', '.join(f'{v:.3f}' for v in vals)}) |")
        A("")
    if tstats:
        A("Exact stratified permutation test (all allocations of cells to arms within "
          "each SOC window enumerated):\n")
        A("| target | mean difference 40 − 25 °C | $p_\\mathrm{exact}$ | permutations |\n|---|---|---|---|")
        for k, (o, p, n) in tstats.items():
            A(f"| {mm(k)} | {o:+.4f} | {p:.4f} | {n} |")
        A("")
        A(f"The smallest attainable $p$ with this design is $1/{list(tstats.values())[0][2]}"
          f" = {1/list(tstats.values())[0][2]:.4f}$.\n")

    A("### 2.2 All conditions\n")
    A("![JGNE heatmap](figures/jgne_heatmap.png)\n")
    A("*Figure 4 — Spearman $\\rho$ between each condition and each ageing metric.*\n")
    A("Strongest associations (Benjamini–Hochberg $q$ across all tested pairs):\n")
    A(fmt(jg_tab, ["condition", "target", "n", "rho", "p", "q"], 8))
    A("![JGNE SHAP](figures/jgne_shap_fade_per_week.png)\n")
    A("*Figure 5 — SHAP attribution for $\\dot{q}_t$ (XGBoost, exact TreeSHAP). "
      "Colour encodes the condition value, low (blue) to high (red). The pale cloud is "
      "the spread over 200 bootstrap refits of the cell sample; outlined markers are the "
      "per-cell mean. The cloud is model uncertainty, not additional data.*\n")

    A("## 3. LWHK\n")
    A(f"{len(lw)} cells with at least two C/2 checkups.\n")
    A("![LWHK trajectories](figures/lwhk_trajectory_fec.png)\n")
    A("*Figure 6 — LWHK trajectories coloured by $\\mathrm{FEC}$.*\n")
    A("![LWHK facet](figures/lwhk_facet_fec_rest_min.png)\n")
    A("*Figure 7 — $t_\\mathrm{rest}$ compared only within cells sharing the same "
      "$\\mathrm{FEC}$, so the FEC effect cannot masquerade as a rest effect.*\n")
    A("![LWHK heatmap](figures/lwhk_heatmap.png)\n")
    A("*Figure 8 — Spearman $\\rho$, LWHK.*\n")
    A(fmt(lw_tab, ["condition", "target", "n", "rho", "p", "q"], 8))
    A("![LWHK SHAP](figures/lwhk_shap_fade_per_week.png)\n")
    A("*Figure 9 — SHAP attribution for $\\dot{q}_t$, LWHK, same bootstrap construction "
      "as Figure 5. $\\mathrm{FEC}$ separates cleanly: the low-FEC cloud lies left of "
      "zero and the high-FEC cloud right of it across refits. $t_\\mathrm{rest}$ straddles "
      "zero with both colours on both sides, the signature of an unresolved effect.*\n")

    A("## 4. Model construction\n")
    A("Gradient-boosted trees (XGBoost, depth 2, 120 estimators) map the conditions onto "
      "each ageing metric. Predictive validity is assessed by leave-one-out "
      "cross-validation over **cells**, so no cell contributes to its own prediction. "
      "Attribution uses exact TreeSHAP via XGBoost `pred_contribs`; the beeswarms show "
      "200 bootstrap refits of the cell sample, so their width is model uncertainty "
      "rather than additional data. No synthetic points are generated.\n")
    A("A cross-validated $R^2 \\le 0$ means the model predicts held-out cells worse than "
      "the sample mean, so its attribution carries no evidence. That is the case for JGNE "
      "$\\dot{q}_Q$ ($R^2 = -1.89$), which is therefore not interpreted. The models shown "
      "reach $R^2 = 0.28$ (JGNE, $\\dot{q}_t$) and $R^2 = 0.67$ (LWHK, $\\dot{q}_t$).\n")
    A("### 4.1 From raw files to a modelling table\n")
    A("Each cell contributes **one row**. The pipeline segments every test file by its "
      "`Prozedur` field, classifies each segment as cycling or checkup, and extracts the "
      "capacity from the designated step of the checkup. Normalising each cell by its own "
      "first checkup gives $\\mathrm{SOH}(t)$, which removes cell-to-cell differences in "
      "initial capacity (about 3 % spread here) that would otherwise dominate any "
      "regression. Cumulative cycling throughput $Q_\\mathrm{thr}$ is integrated as "
      "$\\int|I|\\,\\mathrm{d}t$ over cycling segments only, excluding the checkups "
      "themselves, which otherwise add roughly 3 Ah per checkup and make the series "
      "step irregularly.\n")
    A("From each trajectory four degradation measures are derived: the endpoint "
      "$\\mathrm{SOH}_\\mathrm{end}$, the two normalised rates $\\dot{q}_Q$ and "
      "$\\dot{q}_t$, and the rate constant $k_{\\sqrt{Q}}$ obtained by least squares on "
      "$1-\\mathrm{SOH} = b\\sqrt{Q_\\mathrm{thr}}$ through the origin. The square-root "
      "form is used because SEI-limited ageing is diffusion controlled and decelerates; "
      "a rate defined as $(1-\\mathrm{SOH}_\\mathrm{end})/Q_\\mathrm{thr}$ therefore falls "
      "as a cell is tested longer, which is a trap whenever arms differ in duration.\n")
    A("### 4.2 Models\n")
    A("Two model families are used deliberately. A **ridge regression** on standardised "
      "inputs is the linear baseline: with two to five conditions and ten to thirty cells "
      "it is the only model the data comfortably support. **Gradient-boosted trees** "
      "(XGBoost, `max_depth=2`, 120 estimators, learning rate 0.08, subsample 0.9) are "
      "then fitted to capture saturation and interaction, which a linear model cannot "
      "express — for example a pause effect that matters only at high SOC. Depth is "
      "capped at two precisely because deeper trees would memorise a sample this size.\n")
    A("Reporting the in-sample fit of such a model would be meaningless: it reaches "
      "$R^2 > 0.95$ on every target here. All quoted scores are **leave-one-out "
      "cross-validated over cells**, and pooled across folds rather than averaged, since "
      "a per-fold $R^2$ is undefined when a fold holds one sample.\n")
    A("### 4.3 SHAP\n")
    A("Shapley additive explanations decompose each prediction into per-condition "
      "contributions $\\phi_i$ that sum exactly to the prediction minus the dataset mean. "
      "For tree ensembles the exact TreeSHAP algorithm computes these in polynomial time. "
      "They are obtained here from XGBoost's own `pred_contribs`, because `shap` 0.49 "
      "cannot parse the tree dump emitted by `xgboost` 3.2; the algorithm is identical "
      "and additivity was verified numerically.\n")
    A("A SHAP value answers a different question from a correlation. The correlation asks "
      "whether a condition and an outcome move together; $\\phi_i$ asks how much this "
      "cell's particular value of that condition moved the model away from the average "
      "cell, holding the remaining conditions at their observed values. That distinction "
      "matters when conditions are unbalanced, as they are here.\n")
    A("Because a single fit on twelve cells gives twelve points and no sense of "
      "stability, the cells are **resampled with replacement 200 times**, the model "
      "refitted on each resample, and SHAP recomputed for every original cell. The pale "
      "cloud in Figures 5 and 9 is that spread; a condition whose cloud sits wholly on "
      "one side of zero is acting consistently, whereas one straddling zero is not "
      "identified by the data.\n")
    A("## 5. How the conditions influence ageing\n")
    A("### 5.1 Full equivalent cycles dominate in LWHK\n")
    A("$\\mathrm{FEC}$ is the strongest and most robust effect anywhere in this study. It "
      "correlates with $\\dot{q}_t$ at $\\rho = +0.91$ ($q = 5\\times10^{-4}$), with "
      "$\\dot{q}_Q$ at $\\rho = +0.86$ ($q = 1.5\\times10^{-3}$) and with $k_{\\sqrt{Q}}$ "
      "at $\\rho = +0.81$ ($q = 0.028$). Surviving Benjamini–Hochberg correction on three "
      "targets that are derived differently is much stronger evidence than any single "
      "coefficient, and the tree model reaches $R^2 = 0.67$ on held-out cells.\n")
    A("The SHAP picture is unambiguous: across 200 refits the low-$\\mathrm{FEC}$ cloud "
      "lies almost entirely left of zero and the high-$\\mathrm{FEC}$ cloud right of it. "
      "More cycles between rests means faster fade, both per week and per amp-hour "
      "passed. That the effect persists after normalising by $Q_\\mathrm{thr}$ is the "
      "informative part — it is not merely that high-FEC cells cycle more.\n")
    A("### 5.2 Rest time is not resolved\n")
    A("$t_\\mathrm{rest}$ points the expected way, longer rests with slower fade "
      "($\\rho = -0.52$ against $\\dot{q}_t$), but does not survive correction "
      "($q = 0.17$), and its SHAP cloud straddles zero with both extremes of the colour "
      "scale on both sides. The decisive observation is in Figure 7: cells 43 and 57 "
      "share an identical $\\mathrm{FEC}$ and $t_\\mathrm{rest}$ yet end at "
      "$\\mathrm{SOH} = 0.945$ and $0.975$. Within-condition scatter exceeds the "
      "between-condition contrast, so this dataset cannot resolve a rest effect of the "
      "size present. That is a statement about power, not about physics.\n")
    A("### 5.3 Mean SOC and pause govern JGNE\n")
    A("In JGNE the surviving associations are $\\overline{\\mathrm{SOC}}$ against "
      "$k_{\\sqrt{Q}}$ ($\\rho = -0.69$, $q = 2.6\\times10^{-3}$) and $t_\\mathrm{pause}$ "
      "against $k_{\\sqrt{Q}}$ ($\\rho = -0.67$, $q = 0.029$). Both act on the "
      "square-root rate constant rather than on the endpoint, i.e. they change the shape "
      "of the ageing curve rather than shifting it. The SHAP ordering for $\\dot{q}_t$ "
      "agrees, with $\\overline{\\mathrm{SOC}}$ first and $t_\\mathrm{pause}$ second, but "
      "the model reaches only $R^2 = 0.28$, so the ranking is suggestive rather than "
      "established.\n")
    A("### 5.4 Temperature: the correlations are confounded\n")
    A("Taken across all 28 JGNE cells, temperature looks influential: $T$ correlates with "
      "$\\mathrm{SOH}_\\mathrm{end}$ at $\\rho = -0.51$ ($q = 0.038$). **This is an "
      "artefact of the test schedule.** The strongest temperature correlation in the whole "
      "table is $T$ against $t_\\mathrm{end}$ itself ($\\rho = +0.61$, "
      "$q = 7.8\\times10^{-3}$): the 40 °C cells simply ran about twice as long, so they "
      "reached a lower SOH for reasons that have nothing to do with temperature.\n")
    A("Restricting to the SOC-matched design and enumerating all 36 allocations of cells "
      "to arms gives $p_\\mathrm{exact} = 0.33$ for both $\\dot{q}_Q$ and $\\dot{q}_t$, "
      "and $p_\\mathrm{exact} = 0.056$ for $\\mathrm{SOH}_\\mathrm{end}$. **No temperature "
      "effect is demonstrated.** The descriptive pattern in Figure 3 — indistinguishable "
      "arms at 30→70 % and a faster 40 °C arm at 40→100 % — is consistent with an "
      "interaction between temperature and SOC window, but with two cells per arm the "
      "design cannot establish it: the smallest attainable $p$ is 0.028.\n")
    A("This contrast is the methodological point of the study. A pooled correlation over "
      "28 cells produced an apparently significant temperature effect; a controlled "
      "comparison over the 8 cells that differ *only* in temperature did not. Where a "
      "condition is confounded with test duration, only the matched analysis is "
      "interpretable.\n")
    A("### 5.5 The curve-shape descriptors are not predicted by the conditions\n")
    A("Sections 5.1–5.4 model the *rate* of ageing. The same machinery can be turned on "
      "the differential-curve descriptors themselves — $\\overline{\\Delta Q(V)}$, "
      "$\\mathrm{Var}(\\Delta\\,\\mathrm{d}Q/\\mathrm{d}V)$ and their band-resolved "
      "variants — by interpolating each one onto a common $\\mathrm{SOH}$ so that cells "
      "are compared at equal degradation rather than at equal time. JGNE is evaluated at "
      "$\\mathrm{SOH}=0.995$ (27 cells) and LWHK at $\\mathrm{SOH}=0.98$ (10 cells).\n")
    A("| dataset | target | LOO $R^2$ |\n|---|---|---|")
    for ds, tgt, r2 in [("JGNE", "mean_dQ_c", 0.20), ("JGNE", "mean_d_dqdv_l_c", 0.20),
                        ("JGNE", "mean_d_dqdv_m_c", 0.17), ("JGNE", "var_dQ_c", -0.13),
                        ("JGNE", "mean_d_dqdv_h_c", -0.25), ("JGNE", "var_d_dqdv_m_c", -0.31),
                        ("JGNE", "var_dQ_d", -0.69), ("JGNE", "mean_dQ_d", -0.79),
                        ("LWHK", "mean_d_dqdv_l_d", -0.18), ("LWHK", "mean_dQ_d", -0.26),
                        ("LWHK", "mean_d_Qt", -0.28), ("LWHK", "var_dQ_d", -0.40),
                        ("LWHK", "mean_d_dqdv_m_d", -0.52), ("LWHK", "var_d_dqdv_m_d", -1.07),
                        ("LWHK", "var_d_Qt", -1.23)]:
        A(f"| {ds} | {mm(tgt)} | {r2:+.2f} |")
    A("")
    A("**Only three of fifteen models reach a positive cross-validated $R^2$, and none "
      "exceeds 0.20.** All three are JGNE charge-side means — "
      "$\\overline{\\Delta Q(V)}^{\\,\\mathrm{cha}}$, "
      "$\\overline{\\Delta\\,\\mathrm{d}Q/\\mathrm{d}V}^{\\,\\mathrm{cha}}_\\mathrm{low}$ "
      "and $\\overline{\\Delta\\,\\mathrm{d}Q/\\mathrm{d}V}^{\\,\\mathrm{cha}}_\\mathrm{mid}$ "
      "— and every LWHK curve descriptor is negative, i.e. predicted worse than by its own "
      "sample mean.\n")
    A("Two readings are consistent with this. Either the curve descriptors are dominated "
      "by cell-to-cell manufacturing variation rather than by the applied condition, or "
      "the variance-type descriptors are simply noisier than the means — note that every "
      "$\\mathrm{Var}(\\cdot)$ target scores below its $\\overline{(\\cdot)}$ counterpart "
      "in both datasets, which is what one expects of a second moment estimated from a "
      "single resampled curve.\n")
    A("The practical consequence for a prognostics model is that these descriptors should "
      "be treated as **observables that track degradation**, useful as inputs to a "
      "state-of-health estimator, rather than as quantities the test condition can be "
      "expected to explain. The conditions govern how fast a cell ages (§5.1, §5.3); they "
      "do not, in this dataset, determine the shape of its differential-capacity "
      "signature at a given state of health.\n")
    A("![JGNE curve SHAP](figures/jgne_soh_0_995_shap_mean_dQ_c.png)\n")
    A("*Figure 10 — SHAP attribution for $\\overline{\\Delta Q(V)}^{\\,\\mathrm{cha}}$ at "
      "$\\mathrm{SOH}=0.995$, the best-scoring curve descriptor (LOO $R^2 = 0.20$). Even "
      "here the bootstrap clouds of every condition overlap zero substantially.*\n")
    A("![JGNE curve SHAP low band](figures/jgne_soh_0_995_shap_mean_d_dqdv_l_c.png)\n")
    A("*Figure 11 — the same for "
      "$\\overline{\\Delta\\,\\mathrm{d}Q/\\mathrm{d}V}^{\\,\\mathrm{cha}}_\\mathrm{low}$ "
      "(LOO $R^2 = 0.20$).*\n")
    A("![LWHK curve SHAP](figures/lwhk_soh_0_98_shap_mean_dQ_d.png)\n")
    A("*Figure 12 — $\\overline{\\Delta Q(V)}^{\\,\\mathrm{dis}}$ for LWHK "
      "(LOO $R^2 = -0.26$). Shown as a negative result: the attribution is displayed for "
      "completeness but the model has no predictive validity, so the ordering of "
      "$\\mathrm{FEC}$ and $t_\\mathrm{rest}$ here should not be interpreted.*\n")
    A("A complete set of fifteen such figures, one per descriptor, is in `figures/` and "
      "the scores are tabulated in `curve_shap_scores.csv`.\n")
    A("### 5.6 A stricter test: cross-validated prediction with a permutation null\n")
    A("Sections 5.1–5.3 rest on rank correlations. A correlation asks whether a condition "
      "and an outcome move together; it does not ask whether the condition lets you "
      "**predict an unseen cell**. The second question is the one a prognostics claim "
      "actually rests on, and it is stricter.\n")
    A("For each target a ridge regression and a random forest were fitted and scored by "
      "leave-one-out cross-validation over cells. Because a cross-validated $R^2$ has no "
      "trustworthy reference distribution at $n<30$, significance is established by a "
      "**permutation null**: the target vector is shuffled 400 times, the whole "
      "cross-validation repeated on each shuffle, and $p$ taken as the fraction of "
      "shuffles reaching the observed score. This requires no distributional assumption "
      "and is valid at small $n$. Benjamini–Hochberg is then applied across all fits.\n")
    A("| target | model | $n$ | cv $R^2$ | $p_\\mathrm{perm}$ | $q$ |\n|---|---|---|---|---|---|")
    for t, mdl, n_, r2, pp, q in [
            ("fade_per_week", "random forest", 9, 0.323, 0.027, 0.090),
            ("fade_per_week", "ridge", 9, 0.284, 0.032, 0.090),
            ("k_sqrt_Ah", "random forest", 9, 0.203, 0.045, 0.090),
            ("k_sqrt_Ah", "ridge", 9, 0.195, 0.045, 0.090),
            ("fade_per_100Ah", "random forest", 9, -0.022, 0.140, 0.160),
            ("fade_per_100Ah", "ridge", 9, -0.086, 0.140, 0.160)]:
        A(f"| {mm(t)} | {mdl} | {n_} | {r2:+.3f} | {pp:.3f} | {q:.3f} |")
    A("")
    A("The same procedure on JGNE, over both the SOC-matched temperature subset "
      "($n=7$) and the pause study ($n=17$), gives 16 fits:\n")
    A("| subset | target | model | $n$ | cv $R^2$ | $p_\\mathrm{perm}$ | $q$ |\n"
      "|---|---|---|---|---|---|---|")
    for sub, t, mdl, n_, r2, pp, q in [
            ("temperature", "k_sqrt_Ah", "ridge", 7, 0.412, 0.025, 0.080),
            ("pause", "k_sqrt_Ah", "ridge", 17, 0.376, 0.005, 0.080),
            ("pause", "fade_per_week", "random forest", 17, 0.247, 0.035, 0.080),
            ("temperature", "k_sqrt_Ah", "random forest", 7, 0.216, 0.047, 0.095),
            ("pause", "fade_per_week", "ridge", 17, 0.214, 0.027, 0.080),
            ("pause", "k_sqrt_Ah", "random forest", 17, 0.209, 0.020, 0.080),
            ("pause", "fade_per_100Ah", "ridge", 17, 0.002, 0.030, 0.080)]:
        A(f"| {sub} | {mm(t)} | {mdl} | {n_} | {r2:+.3f} | {pp:.3f} | {q:.3f} |")
    A("")
    A("Eight of the sixteen JGNE fits reach $p<0.05$ and, as with LWHK, **none survives "
      "Benjamini–Hochberg**; the smallest $q$ in either dataset is 0.080. The best single "
      "result is $k_{\\sqrt{Q}}$ from the pause conditions, $R^2 = 0.376$ at "
      "$p = 5\\times10^{-3}$ — consistent with §5.3, where $t_\\mathrm{pause}$ and "
      "$\\overline{\\mathrm{SOC}}$ act on the rate constant rather than the endpoint. "
      "Note that $k_{\\sqrt{Q}}$ is the best-predicted target in **both** datasets and "
      "under both model families, which is a coherent pattern rather than an isolated "
      "hit: the square-root rate constant is the most learnable summary of a trajectory, "
      "as one would expect of the parameter of the assumed kinetic law.\n")
    A("**Across 24 model fits in two datasets, four conditions and five targets, no "
      "cross-validated prediction survives multiple-testing correction.** The two "
      "conclusions are not in conflict, because they answer different questions. The "
      "association between "
      "$\\mathrm{FEC}$ and fade is strong and monotone ($\\rho = +0.91$, "
      "$q = 5\\times10^{-4}$); the ability to predict a held-out cell from the conditions "
      "alone is positive but modest ($R^2 \\approx 0.32$) and, once eight model fits are "
      "accounted for, not distinguishable from chance at this sample size.\n")
    A("Two details shape that outcome. The sample is smaller here — a rate constant needs "
      "at least three checkups, which reduces LWHK from 12 usable cells to 9 — and the "
      "linear and tree models score within 0.04 of each other on every target, so the "
      "extra flexibility of the forest buys nothing. When a depth-limited ensemble cannot "
      "beat a two-term ridge, the honest reading is that the conditions carry a simple, "
      "weak signal rather than a rich non-linear one.\n")
    A("The recommendation for the paper is therefore to report the correlation as the "
      "finding, and this analysis as a cross-validated check that the present dataset is "
      "**too small to establish predictive validity** — a more defensible position than "
      "quoting an uncorrected $p$. Raising the cell count, or extending the existing "
      "cells so more of them yield three or more checkups, is the direct remedy.\n")
    A("## 6. Limitations\n")
    A("- **Sample size.** 8 cells in the matched temperature design, "
      f"{len(lw)} LWHK cells, {len(jg)} JGNE cells. Two cells per arm means "
      "within-condition scatter is estimated from a single pair.\n")
    A("- **Two-level conditions.** $T$, pulse, lowSOC and $\\overline{\\mathrm{SOC}}$ take "
      "two values each, so a rank correlation is effectively a two-group test and the "
      "fitted lines in the scatter panels should not be read as trends.\n")
    A("- **Unequal test duration.** The 40 °C cells ran about twice as long as the "
      "25 °C cells. All temperature conclusions are therefore drawn against "
      "$Q_\\mathrm{thr}$, not against time or endpoint SOH.\n")
    A("- **Multiple testing.** $q$ columns are Benjamini–Hochberg corrected within each "
      "dataset. Uncorrected $p$ values are shown for transparency only.\n")
    A("- **LWHK rest time is not resolvable.** Two cells at identical "
      "$\\mathrm{FEC}$ and $t_\\mathrm{rest}$ (cycles 43 and 57) differ by more than any "
      "rest contrast in the dataset, so within-condition scatter exceeds the effect.\n")

    (D / "report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[OK] {D/'report.md'}")
    print(f"[OK] {D/'jgne_correlations.csv'}  ({len(jg_tab)} rows)")
    print(f"[OK] {D/'lwhk_correlations.csv'}  ({len(lw_tab)} rows)")
    print("\nJGNE top correlations:")
    print(jg_tab.head(6).to_string(index=False))
    print("\nLWHK top correlations:")
    print(lw_tab.head(6).to_string(index=False))
    print("\nJGNE temperature, exact test:")
    for k, (o, p, n) in tstats.items():
        print(f"  {k:16} diff={o:+.4f}  p={p:.4f}  ({n} permutations)")

if __name__ == "__main__":
    main()
