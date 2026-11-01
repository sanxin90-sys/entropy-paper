"""Fig. S1 (file f6_sensitivity_tornado) with bar labels equal to the row names of Table S4
of the manuscript (v10). Same style module as the pipeline figures (figures.py, unchanged).

Bars: change of the 2015-2016 sigma relative to the core configuration, under S0 for the
formulation, bias and sample rows and under the stated scenario for the closure rows.
A bias row of Table S4 (+ / -) is drawn as two bars on one line, each marked with its sign.
Not drawn: the core row, the radiation-only sample (sigma not defined) and the per-period S3 row,
whose 2015-2016 value equals that of the single-CR S3 row (the full-period CR is the same).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import figures as F            # pipeline style (rcParams, save(), colours)
import config as C

plt = F.plt

# pipeline label -> Table S4 row name (v10)
S4_NAMES = {
    "eps_s = 0.95": r"$\varepsilon_s$ = 0.95",
    "eps_s = 0.97": r"$\varepsilon_s$ = 0.97",
    "eps_s = 0.99": r"$\varepsilon_s$ = 0.99",
    "J_LWin with T_a": r"$J_{LW,in}$ with $T_a$ instead of $T_{sky}$",
    "J_LWin diluted X(delta_a), T_a": r"$J_{LW,in}$ diluted, $(4/3)(LW_{in}/T_a)\,X(\delta_a)$",
    "J_H, J_LE with T_a": r"$J_H$, $J_{LE}$ with $T_a$",
    "J_SWout variant (b) diluted": r"$J_{SW,out}$ diluted blackbody (variant b)",
    "J_SWin hemispheric-diluted upper bound": r"$J_{SW,in}$ fully diffuse hemispheric (bound)",
    "J_LWout greybody eq. 11": r"$J_{LW,out}$ greybody (emitted + reflected at sky entropy)",
    "J_LWout greybody + diluted reflection (upper bound)": r"$J_{LW,out}$ greybody + diluted reflection (bound)",
    "T_sun = 5778 K": r"$T_{sun}$ = 5778 K",
    "closure S1 (residual to LE)": r"Closure S1 (residual to $LE$)",
    "closure S2 (residual to H)": r"Closure S2 (residual to $H$)",
    "closure S3 (Bowen-preserving, period CR)": r"Closure S3 (Bowen-preserving, single period CR = 0.826)",
    "sample: dawn-relaxed (SW_IN_F_QC<=1 when SW_IN_POT<50)": "Sample: dawn-relaxed (n = 26,063)",
    "sample: gap-filled _F fluxes (not measurements)": "Sample: gap-filled _F fluxes (n = 31,534; model values)",
    "radiation convention F/T for all four radiation terms (no 4/3)": r"$F/T$ convention for all four radiation terms",
    "J_LWin with T_sky from eps_atm = 0.85": r"$J_{LW,in}$ with $T_{sky}$ from $\varepsilon_{atm}$ = 0.85",
}
BIAS_NAMES = {
    "SW_IN_F": r"Bias $SW_{in}$ +5 % / $-$5 %",
    "SW_OUT": r"Bias $SW_{out}$ +5 % / $-$5 %",
    "LW_IN_F": r"Bias $LW_{in}$ +10 / $-$10 W m$^{-2}$",
    "LW_OUT": r"Bias $LW_{out}$ +10 / $-$10 W m$^{-2}$",
}
NOT_DRAWN = ("core", "sample: radiation-only (J_rad terms; sigma_rad only)",
             "closure S3 (Bowen-preserving, per-period CR: wet and dry pooled each with own CR)")


def rows_for_figure(t4: pd.DataFrame) -> list[tuple[str, list[tuple[float, str]]]]:
    """[(Table S4 row name, [(delta sigma, sign mark), ...]), ...]; raises on an unmapped label."""
    out: dict[str, list] = {}
    order = []
    for _, r in t4.iterrows():
        lab = r["label"]
        if lab in NOT_DRAWN:
            continue
        d = float(r["d_sigma_vs_core"])
        if not np.isfinite(d):
            raise ValueError(f"non-finite delta for drawn row {lab}")
        if lab.startswith("bias "):
            var = lab.split()[1]
            name = BIAS_NAMES[var]
            mark = "+" if " +" in lab else "−"
        else:
            name = S4_NAMES[lab]
            mark = ""
        if name not in out:
            out[name] = []; order.append(name)
        out[name].append((d, mark))
    return [(n, out[n]) for n in order]


def f6_tornado_s4(t4: pd.DataFrame, stem="f6_sensitivity_tornado"):
    rows = rows_for_figure(t4)
    rows.sort(key=lambda r: max(abs(v) for v, _ in r[1]))
    fig, ax = plt.subplots(figsize=(7.4, 0.3 * len(rows) + 1.3))
    for i, (name, bars) in enumerate(rows):
        for v, mark in bars:
            ax.barh(i, v, color=C.COLORS["lw_out"] if v >= 0 else C.COLORS["dry"], height=0.6)
            if mark:
                ax.text(v + (0.002 if v >= 0 else -0.002), i, mark, va="center", ha="left" if v >= 0 else "right", fontsize=7)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([n for n, _ in rows])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel(r"$\Delta\langle\sigma\rangle$ relative to the core configuration (W m$^{-2}$ K$^{-1}$), 2015-2016")
    ax.grid(axis="y", visible=False)
    lo = min(v for _, b in rows for v, _ in b); hi = max(v for _, b in rows for v, _ in b)
    ax.set_xlim(lo - 0.015, hi + 0.015)
    F.save(fig, stem)
    return rows
