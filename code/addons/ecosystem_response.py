"""Ecosystem-response analyses of the BR-Npw surface entropy budget (addon of 2026-11-17).

Run from code/:  python3 -E -P addons/ecosystem_response.py   (not -I: matplotlib is in the user site-packages here)

This script does NOT modify the pipeline (code/*.py; code hash a5a3ec682769). It imports the
pipeline modules unchanged, rebuilds the core sample (25,662 half-hours, 2015-2016) and the core
budget (S0) plus the closed-budget variant S1 exactly as run_all.py does, and adds:

  E1  Flood-phase contrast. Phases from the published anaerobic (flooded-soil) windows of
      Dalmagro et al. (2019, Glob Change Biol 25:1967-1981, Table 1, p. 1973):
        23 Dec 2014 + 190 d -> 23 Dec 2014-30 Jun 2015;  4 Feb 2016 + 138 d -> 4 Feb-20 Jun 2016;
        18 Nov 2016 (219 d) -> 18 Nov 2016-24 Jun 2017.
      flooded = day inside an anaerobic window; wet_nf = Nov-Apr day outside it;
      dry = May-Oct day outside it. Ratio-of-means metrics per phase with 7-day moving-block
      bootstrap (1000 replicates, seed 20261030 + phase index), differences, regression
      sigma ~ absSW + phase on daily means of complete days, matched-absSW-bin differences and a
      multiplicative decomposition sigma = CF x SW_IN x (1 - alpha_eff).
  E2  Atmospheric demand and soil: correlations (Pearson, Spearman; total and partial on absSW)
      of daily sigma, conversion factor and evaporative fraction with daytime VPD, soil and air
      temperature and antecedent precipitation; strata all / wet / dry months / phases.
  E3  Dry season 2015 vs 2016 (calendar May-Oct, matched months, aerobic dry phase), P_F totals.
  E4  GPP (ONEFlux daytime partitioning, GPP_DT_VUT_REF; the night-time GPP_NT_* columns are
      empty for this site) and water use (GPP/ET) by phase, and their correlation with sigma.

Outputs: results/tables/e*_*.csv, results/figures/e1_*.{png,pdf}, e2_*.{png,pdf},
results/run_log_ecosystem_2026-11-17.txt (appended to results/run_log.txt by the caller).
"""
from __future__ import annotations

import hashlib
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))   # code/

import numpy as np
import pandas as pd
from scipy.stats import rankdata

import config as C
import load as L
import entropy as E
import stats as S

TAG = "2026-11-17"
BD = C.BOOT_BLOCK_DAYS_HEADLINE
NB = C.BOOT_N
SEED = C.BOOT_SEED
LOG = []
T0_ = time.time()

# Published anaerobic (flooded-soil) windows, Dalmagro et al. 2019 Table 1 p. 1973 (start date +
# length in days; inclusive end = start + length - 1). The aerobic 2015 start printed as
# "31 Jun 2015" is read as 1 Jul 2015, consistent with the 190-day anaerobic length.
ANAEROBIC = [("2014-12-23", 190), ("2016-02-04", 138), ("2016-11-18", 219)]
PHASES = ["flooded", "wet_nf", "dry"]
PHASE_LABEL = {"flooded": "flooded (anaerobic)", "wet_nf": "wet season, not flooded", "dry": "dry season, not flooded"}
EXTRA = ["VPD_F", "P_F", "TS_F_MDS_1", "TS_F_MDS_2", "GPP_DT_VUT_REF", "GPP_DT_VUT_25", "GPP_DT_VUT_75",
         "NEE_VUT_REF", "NEE_VUT_REF_QC", "P_F_QC", "USTAR", "LE_F_MDS", "H_F_MDS", "TA_F", "NIGHT"]


def log(msg=""):
    line = f"[{time.time() - T0_:7.1f}s] {msg}" if msg else ""
    print(line, flush=True)
    LOG.append(line)


def anaerobic_windows():
    return [(pd.Timestamp(s), pd.Timestamp(s) + pd.Timedelta(days=n - 1)) for s, n in ANAEROBIC]


def phase_of(dates: pd.Series) -> np.ndarray:
    d = pd.to_datetime(dates)
    flooded = np.zeros(len(d), bool)
    for a, b in anaerobic_windows():
        flooded |= ((d >= a) & (d <= b)).to_numpy()
    wetm = d.dt.month.isin(C.WET_MONTHS).to_numpy()
    return np.where(flooded, "flooded", np.where(wetm, "wet_nf", "dry"))


# ----------------------------------------------------------------------------
# Ratio-of-sums metrics with a daily moving-block bootstrap
# ----------------------------------------------------------------------------
# metric = sum(num)/sum(den) over the half-hours of the stratum (identical to the pipeline's
# ratio-of-means when num and den are defined on the same rows).
METRICS = {
    "sigma_S0": ("sigma0", "one", "W m-2 K-1"),
    "sigma_S1": ("sigma1", "one", "W m-2 K-1"),
    "absSW": ("absSW", "one", "W m-2"),
    "SW_IN": ("SW_IN", "one", "W m-2"),
    "CF_S0": ("sigma0", "absSW", "K-1"),
    "CF_S1": ("sigma1", "absSW", "K-1"),
    "resid_frac_Rres_over_absSW": ("Rres", "absSW", "-"),
    "J_LE_share_of_J_out": ("J_LE", "J_out", "-"),
    "J_LE_share_of_J_turb": ("J_LE", "J_turb", "-"),
    "J_LE": ("J_LE", "one", "W m-2 K-1"),
    "J_H": ("J_H", "one", "W m-2 K-1"),
    "Bowen_day": ("H_day", "LE_day", "-"),
    "EF_LE_over_Rn_day": ("LE_day", "Rn_day", "-"),
    "EF_LE_over_HplusLE_day": ("LE_day", "HLE_day", "-"),
    "closure_CR_day": ("HLE_day", "Rn_day", "-"),
    "albedo_day": ("SWOUT_day", "SWIN_day", "-"),
    "alpha_eff": ("SWOUT", "SW_IN", "-"),        # 1 - <absSW>/<SW_IN>, exact for the decomposition
    "Ts_minus_Ta_all": ("dT", "one", "K"),
    "Ts_minus_Ta_day": ("dT_day", "one_day", "K"),
    "Ts": ("Ts", "one", "K"),
    "Ts_absSW_weighted": ("absSW_x_Ts", "absSW", "K"),
    "inv_Ts_absSW_weighted": ("absSW_over_Ts", "absSW", "K-1"),
    "Ta": ("Ta", "one", "K"),
    "VPD_day_kPa": ("VPD_day", "one_day", "kPa"),
    "Rn": ("Rn", "one", "W m-2"),
    "LE": ("LE", "one", "W m-2"),
    "H": ("H", "one", "W m-2"),
    "Rres": ("Rres", "one", "W m-2"),
}
NEEDED = sorted({c for v in METRICS.values() for c in v[:2]})


def daily_sums(W: pd.DataFrame, mask) -> tuple[np.ndarray, np.ndarray, list]:
    d = W.loc[mask, NEEDED + ["date"]]
    g = d.groupby("date")[NEEDED].sum()
    return g.to_numpy(float), g.index.to_numpy(), NEEDED


def metric_values(tot: np.ndarray, cols: list) -> dict:
    """tot: (..., ncols) sums -> dict metric -> array(...)"""
    ix = {c: i for i, c in enumerate(cols)}
    out = {}
    for m, (num, den, _) in METRICS.items():
        with np.errstate(divide="ignore", invalid="ignore"):
            out[m] = tot[..., ix[num]] / tot[..., ix[den]]
    return out


def boot_stratum(W, mask, seed, n_boot=NB, block_days=BD):
    """Point values and (n_boot,) replicate arrays of every metric for one stratum."""
    sums, days, cols = daily_sums(W, mask)
    nd = len(days)
    point = metric_values(sums.sum(axis=0), cols)
    rng = np.random.default_rng(seed)
    idx = S._block_indices(nd, n_boot, block_days, rng)
    reps = metric_values(sums[idx].sum(axis=1), cols)
    return {"point": point, "reps": reps, "n_days": nd, "n_hh": int(np.asarray(mask).sum())}


def frac_opposite(diff, point):
    """Fraction of replicates on the other side of zero from the point estimate (descriptive)."""
    k = int(np.sum(diff <= 0)) if point > 0 else int(np.sum(diff >= 0))
    return k, (f"< {1 / len(diff):g} (0 of {len(diff)})" if k == 0 else f"{k / len(diff):.3f} ({k} of {len(diff)})")


def stratum_rows(name, res):
    rows = []
    for m, (_, _, unit) in METRICS.items():
        lo, hi = np.nanpercentile(res["reps"][m], [2.5, 97.5])
        rows.append({"stratum": name, "metric": m, "unit": unit, "value": float(res["point"][m]),
                     "lo95": float(lo), "hi95": float(hi), "n_hh": res["n_hh"], "n_days": res["n_days"]})
    return rows


def diff_rows(name_a, ra, name_b, rb):
    rows = []
    for m, (_, _, unit) in METRICS.items():
        d = ra["reps"][m] - rb["reps"][m]
        pt = float(ra["point"][m] - rb["point"][m])
        lo, hi = np.nanpercentile(d, [2.5, 97.5])
        k, s = frac_opposite(d, pt)
        with np.errstate(divide="ignore", invalid="ignore"):
            rl, rh = np.nanpercentile(100 * (ra["reps"][m] / rb["reps"][m] - 1), [2.5, 97.5])
        rows.append({"contrast": f"{name_a} - {name_b}", "metric": m, "unit": unit, "a": float(ra["point"][m]),
                     "b": float(rb["point"][m]), "diff": pt, "lo95": float(lo), "hi95": float(hi),
                     "rel_diff_pct": 100 * pt / float(rb["point"][m]) if rb["point"][m] else np.nan,
                     "rel_lo95_pct": float(rl), "rel_hi95_pct": float(rh),
                     "resolved_95": bool(lo > 0 or hi < 0), "frac_opposite_sign": s,
                     "n_days_a": ra["n_days"], "n_days_b": rb["n_days"]})
    return rows


# ----------------------------------------------------------------------------
# Daily tables
# ----------------------------------------------------------------------------
def build(df, core):
    J0 = E.budget(df[core], C.CORE)
    J1 = E.budget(df[core], C.Options(closure="S1"))
    d = df[core]
    is_day = d["is_day"].to_numpy()
    W = pd.DataFrame(index=J0.index)
    W["date"] = d["date"].to_numpy()
    W["month"] = d["month"].to_numpy()
    W["year"] = d["year"].to_numpy()
    W["one"] = 1.0
    W["one_day"] = is_day.astype(float)
    W["sigma0"] = J0["sigma"]
    W["sigma1"] = J1["sigma"]
    W["absSW"] = J0["absSW"]
    W["SW_IN"] = d["SW_IN_F"].to_numpy(float)
    W["SWOUT"] = d["SW_OUT"].to_numpy(float)
    W["Rres"] = J0["Rres"]
    W["Rn"] = J0["Rn"]
    W["H"] = J0["H_s"]
    W["LE"] = J0["LE_s"]
    W["J_LE"] = J0["J_LE"]
    W["J_H"] = J0["J_H"]
    W["J_out"] = J0["J_out"]
    W["J_turb"] = J0["J_H"] + J0["J_LE"]
    W["Ts"] = J0["Ts"]
    W["Ta"] = J0["Ta"]
    W["dT"] = J0["Ts"] - J0["Ta"]
    W["absSW_x_Ts"] = J0["absSW"] * J0["Ts"]
    W["absSW_over_Ts"] = J0["absSW"] / J0["Ts"]
    z = lambda s: np.where(is_day, s, 0.0)
    W["dT_day"] = z(W["dT"])
    W["H_day"] = z(W["H"]); W["LE_day"] = z(W["LE"]); W["Rn_day"] = z(W["Rn"])
    W["HLE_day"] = W["H_day"] + W["LE_day"]
    W["SWOUT_day"] = z(W["SWOUT"]); W["SWIN_day"] = z(W["SW_IN"])
    W["VPD_day"] = z(d["VPD_F"].to_numpy(float) / 10.0)    # hPa -> kPa
    W["phase"] = phase_of(W["date"])
    assert np.isfinite(W[NEEDED].to_numpy()).all()
    return W, J0, J1


def daily_table(W, full):
    """Complete core days (>= MIN_HH_PER_DAY): core-sample daily means + full-record daily drivers."""
    g = W.groupby("date")
    n = g.size()
    keep = n[n >= C.MIN_HH_PER_DAY].index
    agg = g[["sigma0", "sigma1", "absSW", "SW_IN", "Rres", "J_LE", "J_H", "J_out", "Ts", "Ta", "dT",
             "H_day", "LE_day", "Rn_day", "HLE_day", "SWOUT_day", "SWIN_day"]].sum()
    cnt = n.astype(float)
    D = pd.DataFrame(index=agg.index)
    D["n_hh"] = n
    for c in ["sigma0", "sigma1", "absSW", "SW_IN", "Rres", "J_LE", "J_H", "J_out", "Ts", "Ta", "dT"]:
        D[c] = agg[c] / cnt
    D["CF0"] = agg["sigma0"] / agg["absSW"]
    D["CF1"] = agg["sigma1"] / agg["absSW"]
    D["EF"] = agg["LE_day"] / agg["Rn_day"]
    D["EFt"] = agg["LE_day"] / agg["HLE_day"]
    D["Bowen"] = agg["H_day"] / agg["LE_day"]
    D["albedo"] = agg["SWOUT_day"] / agg["SWIN_day"]
    D["resid_frac"] = agg["Rres"] / agg["absSW"]
    D = D.loc[keep].copy()
    # full-record daily drivers (all 48 half-hours of the day)
    f = full.copy()
    f["date"] = f["time"].dt.normalize()
    f["VPD_day"] = np.where(f["NIGHT"] == 0, f["VPD_F"] / 10.0, np.nan)
    lam = (2.501 - 0.002361 * f["TA_F"]) * 1e6                     # J kg-1
    f["ET_mm"] = f["LE_F_MDS"] / lam * C.DT_HH_S                   # kg m-2 per half-hour
    f["GPP_g"] = f["GPP_DT_VUT_REF"] * 12.011e-6 * C.DT_HH_S        # g C m-2 per half-hour
    f["GPP25_g"] = f["GPP_DT_VUT_25"] * 12.011e-6 * C.DT_HH_S
    f["GPP75_g"] = f["GPP_DT_VUT_75"] * 12.011e-6 * C.DT_HH_S
    f["nee_meas"] = (f["NEE_VUT_REF_QC"] == 0).astype(float)
    gd = f.groupby("date")
    P = gd["P_F"].sum()
    Pfull = P.reindex(pd.date_range(P.index.min(), P.index.max()), fill_value=np.nan)
    drv = pd.DataFrame({
        "VPD_day_kPa": gd["VPD_day"].mean(), "Ta_C_full": gd["TA_F"].mean(),
        "TS1_C": gd["TS_F_MDS_1"].mean(), "TS2_C": gd["TS_F_MDS_2"].mean(),
        "P_day_mm": P, "P_prev7_mm": Pfull.shift(1).rolling(7, min_periods=7).sum(),
        "P_prev30_mm": Pfull.shift(1).rolling(30, min_periods=30).sum(),
        "GPP_gC": gd["GPP_g"].sum(min_count=48), "GPP_VUT25_gC": gd["GPP25_g"].sum(min_count=48), "GPP_VUT75_gC": gd["GPP75_g"].sum(min_count=48),
        "ET_mm": gd["ET_mm"].sum(min_count=48), "frac_NEE_measured": gd["nee_meas"].mean(),
    })
    D = D.join(drv, how="left")
    D["WUE_gC_per_kg"] = D["GPP_gC"] / D["ET_mm"]
    D["GPP_per_absSW_gC_MJ"] = D["GPP_gC"] / (D["absSW"] * 0.0864)
    D["phase"] = phase_of(pd.Series(D.index))
    D["season"] = np.where(pd.DatetimeIndex(D.index).month.isin(C.WET_MONTHS), "wet", "dry")
    D.index.name = "date"
    return D


# ----------------------------------------------------------------------------
# Daily-level bootstrap helpers (blocks of consecutive complete days, within stratum)
# ----------------------------------------------------------------------------
def block_idx(nd, seed, n_boot=NB, block_days=BD):
    return S._block_indices(nd, n_boot, block_days, np.random.default_rng(seed))


def _resid(a, z):
    """Residuals of each column of a (n, k) on [1, z]."""
    X = np.column_stack([np.ones(len(z)), z])
    beta, *_ = np.linalg.lstsq(X, a, rcond=None)
    return a - X @ beta


def _corr(Y, X):
    with np.errstate(divide="ignore", invalid="ignore"):
        return _corr_(Y, X)


def _corr_(Y, X):
    Y = (Y - Y.mean(0)) / Y.std(0)
    X = (X - X.mean(0)) / X.std(0)
    return Y.T @ X / len(Y)


def corr_set(Y, X, z=None, spearman=False):
    if spearman:
        Y = np.apply_along_axis(rankdata, 0, Y); X = np.apply_along_axis(rankdata, 0, X)
        if z is not None:
            z = rankdata(z)
    if z is not None:
        Y = _resid(Y, z); X = _resid(X, z)
    return _corr(Y, X)


def corr_table(D, ys, xs, strata, seed_base):
    rows = []
    for si, (sname, sm) in enumerate(strata):
        d = D[sm].dropna(subset=ys + xs + ["absSW"])
        nd = len(d)
        if nd < 15:
            continue
        Y = d[ys].to_numpy(float); X = d[xs].to_numpy(float); z = d["absSW"].to_numpy(float)
        idx = block_idx(nd, seed_base + si)
        for kind, zz, sp in (("pearson", None, False), ("pearson_partial_absSW", z, False),
                             ("spearman", None, True), ("spearman_partial_absSW", z, True)):
            pt = corr_set(Y, X, zz, sp)
            reps = np.array([corr_set(Y[i], X[i], None if zz is None else zz[i], sp) for i in idx])
            lo, hi = np.nanpercentile(reps, [2.5, 97.5], axis=0)
            for a, yv in enumerate(ys):
                for b, xv in enumerate(xs):
                    if yv == xv or (xv == "absSW" and kind.endswith("partial_absSW")):
                        continue
                    rows.append({"stratum": sname, "n_days": nd, "response": yv, "driver": xv, "kind": kind,
                                 "r": float(pt[a, b]), "lo95": float(lo[a, b]), "hi95": float(hi[a, b]),
                                 "resolved_95": bool(lo[a, b] > 0 or hi[a, b] < 0)})
    return pd.DataFrame(rows)


def ols(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def regression_phase(D, ycol, seed):
    """y ~ b0 + b1 absSW + b_flooded + b_wet_nf (reference dry), daily means of complete days;
    and the interaction model (separate slope per phase). Phase-stratified block bootstrap."""
    d = D.dropna(subset=[ycol, "absSW"])
    groups = {p: np.where(d["phase"].to_numpy() == p)[0] for p in PHASES}
    def design(dd, inter=False):
        a = dd["absSW"].to_numpy(float); ph = dd["phase"].to_numpy()
        fl = (ph == "flooded").astype(float); wn = (ph == "wet_nf").astype(float)
        if not inter:
            return np.column_stack([np.ones(len(a)), a, fl, wn])
        return np.column_stack([np.ones(len(a)), a, fl, wn, a * fl, a * wn])
    y = d[ycol].to_numpy(float)
    names_main = ["intercept", "absSW", "flooded_vs_dry", "wet_nf_vs_dry"]
    names_int = names_main + ["absSW_x_flooded", "absSW_x_wet_nf"]
    b_main = ols(y, design(d)); b_int = ols(y, design(d, True))
    yhat = design(d) @ b_main
    r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
    idxs = {p: block_idx(len(g), seed + k) for k, (p, g) in enumerate(groups.items())}
    rm, ri = [], []
    for r in range(NB):
        sel = np.concatenate([groups[p][idxs[p][r]] for p in PHASES])
        dd = d.iloc[sel]
        yy = y[sel]
        rm.append(ols(yy, design(dd))); ri.append(ols(yy, design(dd, True)))
    rm = np.array(rm); ri = np.array(ri)
    rows = []
    for model, b, reps, names in (("main", b_main, rm, names_main), ("interaction", b_int, ri, names_int)):
        lo, hi = np.percentile(reps, [2.5, 97.5], axis=0)
        for j, nm in enumerate(names):
            rows.append({"response": ycol, "model": model, "term": nm, "coef": float(b[j]), "lo95": float(lo[j]),
                         "hi95": float(hi[j]), "resolved_95": bool(lo[j] > 0 or hi[j] < 0),
                         "n_days": len(d), "R2_main": float(r2),
                         **{f"n_days_{p}": len(groups[p]) for p in PHASES}})
    return rows


def matched_bins(D, ycol, a, b, seed, width=25.0, min_n=3):
    """Difference a - b of mean daily y within absSW bins (both phases >= min_n days), weighted by
    n_a n_b/(n_a + n_b); phase-stratified block bootstrap with the bin set recomputed."""
    da = D[D["phase"] == a].dropna(subset=[ycol]); db = D[D["phase"] == b].dropna(subset=[ycol])
    ya, xa = da[ycol].to_numpy(float), np.floor(da["absSW"].to_numpy(float) / width)
    yb, xb = db[ycol].to_numpy(float), np.floor(db["absSW"].to_numpy(float) / width)
    def est(ya, xa, yb, xb):
        num = 0.0; wt = 0.0; nb_used = 0; na_d = 0; nb_d = 0
        for k in np.intersect1d(np.unique(xa), np.unique(xb)):
            ma = xa == k; mb = xb == k
            na, nb = ma.sum(), mb.sum()
            if na >= min_n and nb >= min_n:
                w = na * nb / (na + nb)
                num += w * (ya[ma].mean() - yb[mb].mean()); wt += w; nb_used += 1; na_d += na; nb_d += nb
        return (num / wt if wt else np.nan), nb_used, na_d, nb_d
    pt, nbins, na_d, nb_d = est(ya, xa, yb, xb)
    ia = block_idx(len(ya), seed); ib = block_idx(len(yb), seed + 1)
    reps = np.array([est(ya[ia[r]], xa[ia[r]], yb[ib[r]], xb[ib[r]])[0] for r in range(NB)])
    lo, hi = np.nanpercentile(reps, [2.5, 97.5])
    raw = float(ya.mean() - yb.mean())
    return {"response": ycol, "contrast": f"{a} - {b}", "bin_width_Wm2": width, "min_days_per_bin": min_n,
            "n_bins": nbins, "days_a_used": int(na_d), "days_a_total": len(ya), "days_b_used": int(nb_d),
            "days_b_total": len(yb), "raw_diff_daily_means": raw, "matched_diff": float(pt), "lo95": float(lo),
            "hi95": float(hi), "resolved_95": bool(lo > 0 or hi < 0)}


def decomposition(name_a, ra, name_b, rb):
    """sigma = CF x SW_IN x (1 - alpha_eff) exactly (ratio-of-sums definitions); log-differences."""
    rows = []
    for s, cf in (("S0", "CF_S0"), ("S1", "CF_S1")):
        def parts(r, which):
            v = r[which]
            return {"sigma": np.log(v[f"sigma_{s}"]), "CF": np.log(v[cf]), "SW_IN": np.log(v["SW_IN"]),
                    "one_minus_alpha": np.log(1 - v["alpha_eff"])}
        pa, pb = parts(ra, "point"), parts(rb, "point")
        qa, qb = parts(ra, "reps"), parts(rb, "reps")
        chk = (pa["CF"] + pa["SW_IN"] + pa["one_minus_alpha"]) - pa["sigma"]
        assert abs(chk) < 1e-12, chk
        for k in ("sigma", "CF", "SW_IN", "one_minus_alpha"):
            d = qa[k] - qb[k]
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            rows.append({"contrast": f"{name_a} - {name_b}", "closure": s, "factor": k,
                         "dlog_pct": 100 * float(pa[k] - pb[k]), "lo95_pct": 100 * float(lo), "hi95_pct": 100 * float(hi),
                         "resolved_95": bool(lo > 0 or hi < 0)})
    return rows


# ----------------------------------------------------------------------------
# Figures (Okabe-Ito; phase colours validated for CVD separation; marker shape as 2nd encoding)
# ----------------------------------------------------------------------------
PCOL = {"flooded": "#009E73", "wet_nf": "#0072B2", "dry": "#D55E00"}
PMRK = {"flooded": "o", "wet_nf": "s", "dry": "^"}
PSHORT = {"flooded": "flooded (anaerobic)", "wet_nf": "wet, not flooded", "dry": "dry, not flooded"}


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 7, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                         "ytick.major.width": 0.6, "pdf.fonttype": 42, "savefig.dpi": 300})
    return plt


def _panel_letter(ax, s):
    ax.text(-0.02, 1.02, s, transform=ax.transAxes, fontweight="bold", fontsize=10, ha="right", va="bottom")


def figures(D, pdiff, e4, ct, reg, fdir):
    plt = _style()
    # ---------------- e1: (a) daily sigma vs absSW by phase; (b) relative differences vs dry phase
    fig = plt.figure(figsize=(7.2, 3.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25], wspace=0.55)
    ax = fig.add_subplot(gs[0])
    xx = np.linspace(D["absSW"].min(), D["absSW"].max(), 50)
    for p in PHASES:
        d = D[D["phase"] == p]
        ax.scatter(d["absSW"], d["sigma0"], s=9, marker=PMRK[p], facecolor=PCOL[p], edgecolor="white",
                   linewidth=0.3, alpha=0.85, label=f"{PSHORT[p]} (n = {len(d)} d)", zorder=2)
        b = np.polyfit(d["absSW"], d["sigma0"], 1)
        xr = np.linspace(d["absSW"].min(), d["absSW"].max(), 20)
        ax.plot(xr, np.polyval(b, xr), color=PCOL[p], lw=1.4, zorder=3)
    g = reg[(reg["response"] == "sigma0") & (reg["model"] == "main")].set_index("term")
    ax.text(0.98, 0.03, "S0, daily means of complete days\n"
            f"common slope {g.loc['absSW','coef']*1e3:.2f} ({g.loc['absSW','lo95']*1e3:.2f}–{g.loc['absSW','hi95']*1e3:.2f}) "
            "×10$^{-3}$ K$^{-1}$\n"
            f"flooded offset {g.loc['flooded_vs_dry','coef']:+.3f} ({g.loc['flooded_vs_dry','lo95']:+.3f} to {g.loc['flooded_vs_dry','hi95']:+.3f})",
            transform=ax.transAxes, va="bottom", ha="right", fontsize=6.5, color="#333333")
    ax.set_xlabel("Absorbed shortwave, daily mean (W m$^{-2}$)")
    ax.set_ylabel("Entropy production $\\sigma_{S0}$ (W m$^{-2}$ K$^{-1}$)")
    ax.legend(loc="upper left", frameon=False, handletextpad=0.3, borderaxespad=0.2)
    _panel_letter(ax, "a")

    ax = fig.add_subplot(gs[1])
    items = [("sigma_S0", "$\\sigma$ (S0)", "e1"), ("sigma_S1", "$\\sigma$ (S1, closed)", "e1"),
             ("absSW", "absorbed SW", "e1"), ("CF_S0", "conversion factor (S0)", "e1"),
             ("CF_S1", "conversion factor (S1)", "e1"), ("albedo_day", "albedo (day)", "e1"),
             ("EF_LE_over_Rn_day", "LE/R$_n$ (day)", "e1"), ("EF_LE_over_HplusLE_day", "LE/(H+LE) (day)", "e1"),
             ("Bowen_day", "Bowen ratio (day)", "e1"), ("GPP_gC_m2_d", "GPP", "e4"), ("WUE_gC_per_kgH2O", "GPP/ET", "e4")]
    y = np.arange(len(items))[::-1].astype(float)
    for j, (p, off) in enumerate((("flooded", -0.17), ("wet_nf", 0.17))):
        for i, (m, lab, src) in enumerate(items):
            if src == "e1":
                r = pdiff[(pdiff["contrast"] == f"{p} - dry") & (pdiff["metric"] == m)].iloc[0]
                v, lo, hi = r["rel_diff_pct"], r["rel_lo95_pct"], r["rel_hi95_pct"]
            else:
                r = e4[(e4["stratum"] == f"{p} - dry") & (e4["metric"] == m)].iloc[0]
                v, lo, hi = r["rel_diff_pct"], r["rel_lo95_pct"], r["rel_hi95_pct"]
            ax.plot([lo, hi], [y[i] + off] * 2, color=PCOL[p], lw=1.4, solid_capstyle="round")
            ax.plot(v, y[i] + off, marker=PMRK[p], color=PCOL[p], ms=4.5, mec="white", mew=0.4, ls="none",
                    label=(f"{PSHORT[p]} − dry" if i == 0 else None))
    ax.axvline(0, color="#999999", lw=0.6, zorder=0)
    for yy in (y[4] - 0.5, y[8] - 0.5):
        ax.axhline(yy, color="#dddddd", lw=0.6, zorder=0)
    ax.set_yticks(y); ax.set_yticklabels([it[1] for it in items])
    ax.set_xscale("symlog", linthresh=10, linscale=1.0)
    ax.set_xticks([-40, -10, -5, 0, 5, 10, 30, 100]); ax.set_xticklabels(["−40", "−10", "−5", "0", "5", "10", "30", "100"])
    ax.set_xlabel("Difference relative to the dry, not-flooded phase (%; 95 % block-bootstrap interval)")
    ax.xaxis.label.set_size(7)
    ax.legend(loc="lower center", bbox_to_anchor=(0.56, 1.0), ncol=2, frameon=False, handletextpad=0.3, columnspacing=1.0)
    _panel_letter(ax, "b")
    fig.savefig(fdir / "e1_phase_response.png", bbox_inches="tight")
    fig.savefig(fdir / "e1_phase_response.pdf", bbox_inches="tight")
    plt.close(fig)

    # ---------------- e2: partial correlations (absSW held fixed) with daily drivers, by phase
    resp = [("sigma0", "$\\sigma$ (S0)"), ("sigma1", "$\\sigma$ (S1)"), ("CF1", "conv. factor (S1)"),
            ("EF", "LE/R$_n$ (day)")]
    drv = [("VPD_day_kPa", "daytime VPD"), ("Ta_C_full", "air temperature"), ("TS1_C", "soil temperature, 0.10 m"),
           ("P_prev30_mm", "rain, previous 30 d")]
    fig, axs = plt.subplots(1, len(drv), figsize=(7.2, 2.6), sharey=True)
    yb = np.arange(len(resp))[::-1].astype(float)
    for k, (dv, dl) in enumerate(drv):
        ax = axs[k]
        for j, (p, off) in enumerate((("flooded", 0.22), ("wet_nf", 0.0), ("dry", -0.22))):
            for i, (rv, rl) in enumerate(resp):
                r = ct[(ct["stratum"] == f"phase {p}") & (ct["response"] == rv) & (ct["driver"] == dv)
                       & (ct["kind"] == "pearson_partial_absSW")].iloc[0]
                ax.plot([r["lo95"], r["hi95"]], [yb[i] + off] * 2, color=PCOL[p], lw=1.2, solid_capstyle="round")
                ax.plot(r["r"], yb[i] + off, marker=PMRK[p], color=PCOL[p], ms=4, mec="white", mew=0.4, ls="none",
                        label=(PSHORT[p] if (i == 0 and k == 0) else None))
        ax.axvline(0, color="#999999", lw=0.6, zorder=0)
        ax.set_xlim(-1, 1); ax.set_xticks([-1, -0.5, 0, 0.5, 1]); ax.set_xticklabels(["−1", "−0.5", "0", "0.5", "1"])
        ax.set_title(dl)
        ax.set_yticks(yb); ax.set_yticklabels([r[1] for r in resp])
        _panel_letter(ax, "abcd"[k])
    fig.supxlabel("Partial correlation with the driver, absorbed shortwave held fixed (daily means; 95 % block-bootstrap interval)", fontsize=7, y=0.06)
    fig.legend(loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(fdir / "e2_driver_partial_correlations.png", bbox_inches="tight")
    fig.savefig(fdir / "e2_driver_partial_correlations.pdf", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
def main():
    tdir = C.TABLES_DIR; fdir = C.FIGURES_DIR
    log(f"ecosystem-response addon {TAG} started {datetime.now().isoformat(timespec='seconds')} (container time)")
    log(f"pipeline code hash {C.code_hash()} (code/*.py, unchanged) | config hash {C.config_hash()} | "
        f"addon sha256 {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]} | seed {SEED} | n_boot {NB} | block {BD} d")
    sha = C.file_sha256(C.HH_FILE)
    log(f"data sha256 {sha} ({'matches' if sha == C.HH_FILE_SHA256 else 'DIFFERS FROM'} config.HH_FILE_SHA256)")
    df = L.add_derived(L.read_hh(), C.EPS_CORE)
    core = L.mask_core(df)
    assert int(core.sum()) == C.CORE_N_REGRESSION
    extra = pd.read_csv(C.HH_FILE, usecols=["TIMESTAMP_START"] + [c for c in EXTRA if c not in ("TA_F", "NIGHT", "VPD_F", "P_F", "TS_F_MDS_1", "TS_F_MDS_2", "LE_F_MDS", "H_F_MDS")],
                        na_values=C.NA_VALUES)
    assert (extra["TIMESTAMP_START"].to_numpy() == df["TIMESTAMP_START"].to_numpy()).all()
    for c in extra.columns[1:]:
        df[c] = extra[c].to_numpy()
    log(f"core sample n = {int(core.sum())} (regression value reproduced)")
    log("variables: SWC_* absent from the product (no soil water content); GPP_NT_* and RECO_NT_* empty for every year; "
        "GPP_DT_VUT_REF present 100 % in 2015-2016 (GPP_DT_CUT_REF identical to it in 2015-2016, so the USTAR-threshold sensitivity uses GPP_DT_VUT_25/75); VPD_F_QC == 0 for all 2015-16 half-hours; "
        "TS_F_MDS_1/2 depths in BIFVARINFO: -0.10 / -0.20 m")
    for (a, b) in anaerobic_windows():
        log(f"anaerobic window (Dalmagro et al. 2019 Table 1 p. 1973): {a.date()} to {b.date()} ({(b - a).days + 1} d)")

    W, J0, J1 = build(df, core)
    log(f"check: sigma_S0 core mean {W['sigma0'].mean():.4f} (T2 0.4954), sigma_S1 {W['sigma1'].mean():.4f} (T3 0.5796)")
    full = df[df["year"].isin(C.ALL_YEARS)][["time", "date", "year", "month"] + [c for c in EXTRA if c in df.columns]]
    D = daily_table(W, full)
    D.reset_index().to_csv(tdir / "e0_daily_ecosystem.csv", index=False, float_format="%.6g")
    log(f"complete days: {len(D)} (pipeline: 485); by phase: {D['phase'].value_counts().to_dict()}")

    # --------------------------------------------------------------- E1 phase contrast
    dt = pd.to_datetime(W["date"])
    ph = W["phase"].to_numpy()
    yr = W["year"].to_numpy()
    strata = {
        "flooded": ph == "flooded", "wet_nf": ph == "wet_nf", "dry": ph == "dry",
        # single hydrological cycle 2015/16 (the only complete wet season)
        "wet_nf 2015/16 (1 Nov 2015-3 Feb 2016)": (ph == "wet_nf") & (dt >= "2015-11-01").to_numpy() & (dt <= "2016-02-03").to_numpy(),
        "flooded 2016 (4 Feb-20 Jun 2016)": (ph == "flooded") & (yr == 2016) & (dt <= "2016-06-20").to_numpy(),
        "dry 2016 (21 Jun-31 Oct 2016)": (ph == "dry") & (yr == 2016),
        "flooded 2015 (core: Mar-30 Jun 2015)": (ph == "flooded") & (yr == 2015),
        "dry 2015 (1 Jul-31 Oct 2015)": (ph == "dry") & (yr == 2015),
        "flooded, wet months (Nov-Apr)": (ph == "flooded") & np.isin(W["month"], C.WET_MONTHS),
        "flooded, dry months (May-Jun)": (ph == "flooded") & np.isin(W["month"], C.DRY_MONTHS),
    }
    rows_def = []
    for k, (nm, m) in enumerate(strata.items()):
        dd = W.loc[m, "date"]
        rows_def.append({"stratum": nm, "n_hh": int(m.sum()), "n_days_with_data": int(dd.nunique()),
                         "first_day": str(pd.Timestamp(dd.min()).date()), "last_day": str(pd.Timestamp(dd.max()).date()),
                         "months": " ".join(sorted({pd.Timestamp(x).strftime('%Y-%m') for x in dd.unique()})),
                         "seed": SEED + k,
                         "source": "anaerobic windows: Dalmagro et al. 2019 Table 1 p. 1973; seasons: site metadata (Nov-Apr wet)"})
    pd.DataFrame(rows_def).to_csv(tdir / "e1_phase_definition.csv", index=False)
    res = {nm: boot_stratum(W, m, SEED + k) for k, (nm, m) in enumerate(strata.items())}
    pm = pd.DataFrame([r for nm, rr in res.items() for r in stratum_rows(nm, rr)])
    pm.to_csv(tdir / "e1_phase_means.csv", index=False, float_format="%.6g")
    contrasts = [("flooded", "dry"), ("flooded", "wet_nf"), ("wet_nf", "dry"),
                 ("flooded 2016 (4 Feb-20 Jun 2016)", "dry 2016 (21 Jun-31 Oct 2016)"),
                 ("flooded 2016 (4 Feb-20 Jun 2016)", "wet_nf 2015/16 (1 Nov 2015-3 Feb 2016)"),
                 ("wet_nf 2015/16 (1 Nov 2015-3 Feb 2016)", "dry 2016 (21 Jun-31 Oct 2016)"),
                 ("flooded 2015 (core: Mar-30 Jun 2015)", "dry 2015 (1 Jul-31 Oct 2015)"),
                 ("flooded, dry months (May-Jun)", "dry")]
    pdiff = pd.DataFrame([r for a, b in contrasts for r in diff_rows(a, res[a], b, res[b])])
    pdiff.to_csv(tdir / "e1b_phase_differences.csv", index=False, float_format="%.6g")
    dec = pd.DataFrame([r for a, b in contrasts[:3] + contrasts[3:6] for r in decomposition(a, res[a], b, res[b])])
    dec.to_csv(tdir / "e1e_decomposition_CF_SWin_albedo.csv", index=False, float_format="%.6g")
    for nm in PHASES:
        p = res[nm]["point"]
        log(f"E1 {nm}: days {res[nm]['n_days']}, n {res[nm]['n_hh']}; sigma S0 {p['sigma_S0']:.4f} S1 {p['sigma_S1']:.4f}; "
            f"absSW {p['absSW']:.1f}; CF S0 {p['CF_S0']:.5f} S1 {p['CF_S1']:.5f}; R'/absSW {p['resid_frac_Rres_over_absSW']:.3f}; "
            f"EF {p['EF_LE_over_Rn_day']:.3f}; LE/(H+LE) {p['EF_LE_over_HplusLE_day']:.3f}; Bowen {p['Bowen_day']:.3f}; "
            f"albedo {p['albedo_day']:.4f}; Ts-Ta day {p['Ts_minus_Ta_day']:+.2f}")
    reg = pd.DataFrame([r for k, y in enumerate(["sigma0", "sigma1", "CF0", "CF1", "EF", "EFt"])
                        for r in regression_phase(D, y, SEED + 100 + 10 * k)])
    reg.to_csv(tdir / "e1c_regression_phase_absSW.csv", index=False, float_format="%.6g")
    mb = pd.DataFrame([matched_bins(D, y, a, b, SEED + 200 + 10 * k + j)
                       for k, y in enumerate(["sigma0", "sigma1", "CF0", "CF1", "EF", "EFt"])
                       for j, (a, b) in enumerate([("flooded", "dry"), ("flooded", "wet_nf"), ("wet_nf", "dry")])])
    mb.to_csv(tdir / "e1d_matched_absSW_bins.csv", index=False, float_format="%.6g")
    log("E1 tables: e1_phase_definition, e1_phase_means, e1b_phase_differences, e1c_regression_phase_absSW, "
        "e1d_matched_absSW_bins, e1e_decomposition_CF_SWin_albedo")

    # --------------------------------------------------------------- E2 drivers
    ys = ["sigma0", "sigma1", "CF0", "CF1", "EF", "EFt"]
    xs = ["VPD_day_kPa", "TS1_C", "TS2_C", "Ta_C_full", "P_prev7_mm", "P_prev30_mm", "absSW"]
    st2 = [("all complete days", np.ones(len(D), bool)), ("wet months (Nov-Apr)", (D["season"] == "wet").to_numpy()),
           ("dry months (May-Oct)", (D["season"] == "dry").to_numpy())] + \
          [(f"phase {p}", (D["phase"] == p).to_numpy()) for p in PHASES]
    ct = corr_table(D, ys, xs, st2, SEED + 300)
    ct.to_csv(tdir / "e2_driver_correlations.csv", index=False, float_format="%.4g")
    log(f"E2: {len(ct)} correlations (6 responses x 7 drivers x 4 kinds x {len(st2)} strata); multiple comparisons, descriptive only")

    # --------------------------------------------------------------- E3 dry 2015 vs 2016
    mo = W["month"].to_numpy()
    nmo = W.groupby(["year", "month"]).size().unstack(0)
    matched = [m for m in C.DRY_MONTHS if nmo.loc[m].min() >= C.MATCHED_MONTH_MIN_HH]
    s3 = {
        "dry season 2015 (May-Oct)": (yr == 2015) & np.isin(mo, C.DRY_MONTHS),
        "dry season 2016 (May-Oct)": (yr == 2016) & np.isin(mo, C.DRY_MONTHS),
        f"2015 matched dry months {matched}": (yr == 2015) & np.isin(mo, matched),
        f"2016 matched dry months {matched}": (yr == 2016) & np.isin(mo, matched),
        "dry phase 2015 (aerobic, 1 Jul-31 Oct)": (ph == "dry") & (yr == 2015),
        "dry phase 2016 (aerobic, 21 Jun-31 Oct)": (ph == "dry") & (yr == 2016),
        "2015 Jul-Oct": (yr == 2015) & np.isin(mo, [7, 8, 9, 10]),
        "2016 Jul-Oct": (yr == 2016) & np.isin(mo, [7, 8, 9, 10]),
    }
    r3 = {nm: boot_stratum(W, m, SEED + 400 + k) for k, (nm, m) in enumerate(s3.items())}
    names3 = list(s3)
    e3 = pd.DataFrame([r for i in range(0, len(names3), 2) for r in diff_rows(names3[i], r3[names3[i]], names3[i + 1], r3[names3[i + 1]])])
    e3.to_csv(tdir / "e3_dry2015_vs_dry2016.csv", index=False, float_format="%.6g")
    pd.DataFrame([r for nm, rr in r3.items() for r in stratum_rows(nm, rr)]).to_csv(tdir / "e3a_dry_season_means.csv", index=False, float_format="%.6g")
    # precipitation and climate from the full record (all half-hours, not only the core sample)
    f = df[df["year"].isin([2014, 2015, 2016])]
    def win(a, b):
        m = (f["date"] >= a) & (f["date"] <= b)
        g = f[m]
        return {"window": f"{a} to {b}", "P_F_mm": float(g["P_F"].sum()), "days_P_gt_1mm": int((g.groupby("date")["P_F"].sum() > 1).sum()),
                "TA_mean_C": float(g["TA_F"].mean()), "VPD_day_mean_kPa": float(g.loc[g["NIGHT"] == 0, "VPD_F"].mean() / 10),
                "VPD_day_max_daily_kPa": float(g.loc[g["NIGHT"] == 0].groupby("date")["VPD_F"].mean().max() / 10),
                "SW_IN_mean": float(g["SW_IN_F"].mean()), "n_hh": int(len(g)), "P_F_QC0_frac": float((g["P_F_QC"] == 0).mean()) if "P_F_QC" in g else np.nan}
    pw = [win("2015-05-01", "2015-10-31"), win("2016-05-01", "2016-10-31"), win("2015-06-01", "2015-09-30"),
          win("2016-06-01", "2016-09-30"), win("2014-11-01", "2015-04-30"), win("2015-11-01", "2016-04-30"),
          win("2015-01-01", "2015-12-31"), win("2016-01-01", "2016-12-31")]
    pd.DataFrame(pw).to_csv(tdir / "e3b_precip_climate_full_record.csv", index=False, float_format="%.5g")
    for r in pw:
        log(f"E3 full record {r['window']}: P_F {r['P_F_mm']:.0f} mm, TA {r['TA_mean_C']:.2f} C, VPD(day) {r['VPD_day_mean_kPa']:.3f} kPa")

    # --------------------------------------------------------------- E4 GPP, WUE
    st4 = [("all complete days", np.ones(len(D), bool))] + [(f"phase {p}", (D["phase"] == p).to_numpy()) for p in PHASES] + \
          [("wet months (Nov-Apr)", (D["season"] == "wet").to_numpy()), ("dry months (May-Oct)", (D["season"] == "dry").to_numpy())]
    ct4 = corr_table(D, ["sigma0", "sigma1", "J_LE", "CF0"], ["GPP_gC", "WUE_gC_per_kg", "ET_mm", "GPP_per_absSW_gC_MJ", "absSW"], st4, SEED + 500)
    ct4.to_csv(tdir / "e4b_gpp_correlations.csv", index=False, float_format="%.4g")
    rows4 = []; reps4 = {}; pts4 = {}
    def stats4(a):
        s = a.sum(axis=-2); m = a.mean(axis=-2)
        return {"GPP_gC_m2_d": m[..., 0], "GPP_VUT25_gC_m2_d": m[..., 1], "GPP_VUT75_gC_m2_d": m[..., 2],
                "ET_mm_d": m[..., 3], "WUE_gC_per_kgH2O": s[..., 0] / s[..., 3],
                "GPP_per_absSW_gC_MJ": s[..., 0] / (s[..., 6] * 0.0864),
                "sigma_S0_daily_mean": m[..., 4], "sigma_S1_daily_mean": m[..., 5],
                "CF_S0_daily_basis": s[..., 4] / s[..., 6], "CF_S1_daily_basis": s[..., 5] / s[..., 6],
                "frac_NEE_measured": m[..., 7]}
    for k, p in enumerate(PHASES + ["all"]):
        d = D if p == "all" else D[D["phase"] == p]
        d = d.dropna(subset=["GPP_gC", "ET_mm"])
        A = d[["GPP_gC", "GPP_VUT25_gC", "GPP_VUT75_gC", "ET_mm", "sigma0", "sigma1", "absSW", "frac_NEE_measured"]].to_numpy(float)
        idx = block_idx(len(d), SEED + 600 + k)
        pts4[p] = stats4(A); reps4[p] = stats4(A[idx])
        for kk in pts4[p]:
            lo, hi = np.nanpercentile(reps4[p][kk], [2.5, 97.5])
            rows4.append({"stratum": p, "metric": kk, "value": float(pts4[p][kk]), "lo95": float(lo), "hi95": float(hi),
                          "resolved_95": "", "n_days": len(d)})
    for a_, b_ in (("flooded", "dry"), ("flooded", "wet_nf"), ("wet_nf", "dry")):
        for kk in pts4[a_]:
            dd = reps4[a_][kk] - reps4[b_][kk]
            pt = float(pts4[a_][kk] - pts4[b_][kk])
            lo, hi = np.nanpercentile(dd, [2.5, 97.5])
            rl, rh = np.nanpercentile(100 * (reps4[a_][kk] / reps4[b_][kk] - 1), [2.5, 97.5])
            rows4.append({"rel_lo95_pct": float(rl), "rel_hi95_pct": float(rh), "stratum": f"{a_} - {b_}", "metric": kk, "value": pt, "lo95": float(lo), "hi95": float(hi),
                          "resolved_95": bool(lo > 0 or hi < 0), "n_days": f"{len(D[D['phase'] == a_])} vs {len(D[D['phase'] == b_])}",
                          "rel_diff_pct": 100 * pt / float(pts4[b_][kk])})
    e4 = pd.DataFrame(rows4)
    e4.to_csv(tdir / "e4_gpp_wue_by_phase.csv", index=False, float_format="%.6g")
    for p in PHASES:
        s = e4[e4["stratum"] == p].set_index("metric")
        log(f"E4 {p}: GPP {s.loc['GPP_gC_m2_d','value']:.2f} gC m-2 d-1, ET {s.loc['ET_mm_d','value']:.2f} mm d-1, "
            f"WUE {s.loc['WUE_gC_per_kgH2O','value']:.2f} gC kg-1, NEE measured {s.loc['frac_NEE_measured','value']:.2f}")

    figures(D, pdiff, e4, ct, reg, fdir)
    log("figures: e1_phase_response.{png,pdf}, e2_driver_partial_correlations.{png,pdf}")
    return W, D, res, pm, pdiff, reg, mb, dec, ct, e3, ct4, e4


if __name__ == "__main__":
    out = main()
    (C.RESULTS_DIR / f"run_log_ecosystem_{TAG}.txt").write_text("\n".join(LOG) + "\n")
