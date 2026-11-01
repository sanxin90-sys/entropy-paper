"""Energy-balance closure diagnostics (METHODS_SPEC.md, section 3)."""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C


def ols(x, y):
    """OLS y = a x + b with R^2; finite pairs only. Returns dict."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return {"slope": np.nan, "intercept": np.nan, "r2": np.nan, "n": int(len(x))}
    a, b = np.polyfit(x, y, 1)
    yhat = a * x + b
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return {"slope": float(a), "intercept": float(b), "r2": float(1 - ss_res / ss_tot), "n": int(len(x))}


def cr_sum(h, le, rn):
    h = np.asarray(h, dtype=float); le = np.asarray(le, dtype=float); rn = np.asarray(rn, dtype=float)
    ok = np.isfinite(h) & np.isfinite(le) & np.isfinite(rn)
    return float((h[ok] + le[ok]).sum() / rn[ok].sum())


def closure_stats(d: pd.DataFrame) -> dict:
    """CR_sum, OLS (H+LE vs Rn), mean residual and fraction R' < 0 on half-hourly rows of d."""
    y = d["H"] + d["LE"]
    fit = ols(d["Rn"], y)
    return {
        "n_hh": int(len(d)),
        "CR_sum_hh": cr_sum(d["H"], d["LE"], d["Rn"]),
        "ols_slope": fit["slope"], "ols_intercept_Wm2": fit["intercept"], "ols_r2": fit["r2"],
        "Rres_mean_Wm2": float(d["Rres"].mean()),
        "Rres_median_Wm2": float(d["Rres"].median()),
        "frac_Rres_negative": float((d["Rres"] < 0).mean()),
    }


def daily_closure(d: pd.DataFrame, min_hh=C.MIN_HH_PER_DAY) -> dict:
    """Closure from daily means over complete days (>= min_hh selected half-hours)."""
    g = d.groupby("date")
    n = g.size()
    daily = g[["Rn", "H", "LE"]].mean()
    daily = daily[n >= min_hh]
    if len(daily) < 3:
        return {"n_days": int(len(daily)), "CR_sum_daily": np.nan, "ols_slope_daily": np.nan,
                "ols_intercept_daily_Wm2": np.nan, "ols_r2_daily": np.nan}
    fit = ols(daily["Rn"], daily["H"] + daily["LE"])
    return {"n_days": int(len(daily)), "CR_sum_daily": cr_sum(daily["H"], daily["LE"], daily["Rn"]),
            "ols_slope_daily": fit["slope"], "ols_intercept_daily_Wm2": fit["intercept"], "ols_r2_daily": fit["r2"]}
