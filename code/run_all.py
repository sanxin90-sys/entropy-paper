"""End-to-end pipeline: surface entropy budget of BR-Npw (ONEFlux v1.3 r1).

Run from code/:  python3 -I run_all.py
Writes results/tables/*.csv (+ Markdown renderings, INDEX.md), results/figures/*.{pdf,png}
and results/run_log.txt (versions, data/code/config provenance, row counts, every filter,
data flags, selection diagnostics, headline numbers).
"""
from __future__ import annotations

import platform
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # -I does not add the script dir

import numpy as np
import pandas as pd

import config as C
import load as L
import entropy as E
import closure as CL
import storage as ST
import stats as S

LOG = []
T_START = time.time()
BD_H = C.BOOT_BLOCK_DAYS_HEADLINE
BD_O = C.BOOT_BLOCK_DAYS_OPTIMISTIC


def log(msg=""):
    line = f"[{time.time() - T_START:7.1f}s] {msg}" if msg else ""
    print(line, flush=True)
    LOG.append(line)


def write_csv(df: pd.DataFrame, name: str, index=False):
    path = C.TABLES_DIR / name
    df.to_csv(path, index=index, float_format="%.6g")
    return path


def df_to_md(df: pd.DataFrame, cols=None, ndig=4) -> str:
    d = df if cols is None else df[cols]
    lines = ["| " + " | ".join(str(c) for c in d.columns) + " |", "|" + "---|" * len(d.columns)]
    for _, r in d.iterrows():
        cells = []
        for v in r:
            if isinstance(v, (float, np.floating)):
                cells.append("" if not np.isfinite(v) else f"{v:.{ndig}g}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


T7_UNITS = {"sigma": "W m-2 K-1", "sigma_rad": "W m-2 K-1", "sigma_turb": "W m-2 K-1",
            "J_LWin": "W m-2 K-1", "J_LWout": "W m-2 K-1", "J_SWin": "W m-2 K-1", "J_SWout": "W m-2 K-1",
            "J_H": "W m-2 K-1", "J_LE": "W m-2 K-1", "sigma_soil_lo": "W m-2 K-1", "sigma_soil_hi": "W m-2 K-1",
            "Rn": "W m-2", "H": "W m-2", "LE": "W m-2", "Rres": "W m-2",
            "dQ_soil_min": "W m-2", "dQ_soil_max": "W m-2", "dQ_water_min": "W m-2", "dQ_water_max": "W m-2",
            "Ts_minus_Ta": "K", "Tsky_minus_Ta": "K"}
DAWN_SLOTS = (10, 11, 12, 13)   # 05:00-06:30 LST: conditional composite (ONEFlux dawn QC hole)


def main():
    for p in (C.TABLES_DIR, C.FIGURES_DIR, C.SCRATCH_DIR):
        p.mkdir(parents=True, exist_ok=True)
    import figures as F   # imported here so the matplotlib fallback is logged
    log(f"run started {datetime.now().isoformat(timespec='seconds')} (local container time)")
    log(f"python {platform.python_version()} | pandas {pd.__version__} | numpy {np.__version__} | matplotlib {F.MPL_VERSION}")
    log(f"config hash {C.config_hash()} | code hash {C.code_hash()} | seed {C.BOOT_SEED} | n_boot {C.BOOT_N} | block days headline {BD_H}, optimistic {BD_O}")
    log(f"matplotlib path: {sys.modules['matplotlib'].__file__}")

    # ------------------------------------------------------------------ data provenance
    if not C.HH_FILE.exists():
        raise FileNotFoundError(f"{C.HH_FILE} not found. Put the AmeriFlux FLUXNET file in {C.DATA_DIR} or set NPW_DATA_DIR (see docs/).")
    sha = C.file_sha256(C.HH_FILE)
    st = C.HH_FILE.stat()
    log(f"data file {C.HH_FILE} | size {st.st_size} bytes | mtime {datetime.fromtimestamp(st.st_mtime).isoformat(timespec='seconds')} | sha256 {sha}")
    if sha != C.HH_FILE_SHA256:
        log(f"  WARNING: sha256 differs from the reference value in config ({C.HH_FILE_SHA256}); results may not be comparable")
    else:
        log("  sha256 matches config.HH_FILE_SHA256 (same input file as the reference run)")

    # ------------------------------------------------------------------ load
    df = L.read_hh()
    log(f"read {C.HH_FILE.name}: {len(df)} rows x {df.shape[1]} cols (usecols={len(C.USECOLS)}), -9999 -> NaN")
    df = L.add_derived(df, C.EPS_CORE)
    core = L.mask_core(df)
    for step, n in L.filter_log(df):
        log(f"  filter {step}: {n} rows remain")
    flags = L.data_flags(df, core)
    for k, v in flags.items():
        log(f"  flag {k}: {v:.4g}" if isinstance(v, float) else f"  flag {k}: {v}")
    log(f"core sample 2015-2016: n = {int(core.sum())} (regression value {C.CORE_N_REGRESSION})")
    assert int(core.sum()) == C.CORE_N_REGRESSION, "core sample count changed"
    log("time labelling: TIMESTAMP_START (a composite value at 11:30 is the 11:30-12:00 interval); in-house solar noon at the site 11:35-11:53 LST")

    # X(delta) check (spec eq. 5 coefficients vs exact spectral integral)
    log("Dilution function check (approx eq. 5 vs exact spectral integral):")
    for dlt in (1.0, 0.98, 0.9, 0.1, 1e-3, 1e-6):
        log(f"  delta={dlt:g}: approx {float(E.x_dilution_approx(dlt)):.4f}  exact {float(E.x_dilution(dlt)):.4f}")
    log("  -> exact X(delta) (table interpolation) is used in every computation; the approximation is only logged.")
    pe = E.planck_entropy_flux(300.0); bb = 4 / 3 * C.SIGMA_SB * 300.0 ** 3
    log(f"Planck spectral entropy at 300 K: {pe:.6f} vs (4/3) sigma T^3 = {bb:.6f} W m-2 K-1")
    n_gt1, dmax = E.count_delta_gt1(df.loc[core, "LW_IN_F"], df.loc[core, "TA_F"] + C.T0)
    log(f"delta_a = LW_IN/(sigma T_a^4) > 1 in {n_gt1} of {int(core.sum())} core half-hours ({100 * n_gt1 / core.sum():.2f} %, max {dmax:.3f}); "
        "clipped to X = 1 in the 'diluted' LW_IN variant and in the reflected part of 'grey_refl'")

    # ------------------------------------------------------------------ periods
    yr, mo, se, sy, s3 = df["year"], df["month"], df["season"], df["season_year"], df["season3"]
    periods = {
        "2015-2016": core,
        "2015": core & (yr == 2015),
        "2016": core & (yr == 2016),
        "wet pooled (Nov-Apr)": core & (se == "wet"),
        "dry pooled (May-Oct)": core & (se == "dry"),
        "wet 2015 (Mar-Apr 2015, partial)": core & (se == "wet") & (sy == 2015),
        "dry 2015 (May-Oct 2015)": core & (se == "dry") & (yr == 2015),
        "wet 2016 (Nov 2015-Apr 2016)": core & (se == "wet") & (sy == 2016),
        "dry 2016 (May-Oct 2016)": core & (se == "dry") & (yr == 2016),
        "wet 2017 (Nov 2016, partial)": core & (se == "wet") & (sy == 2017),
        "3-season dry May-Sep (Preliminary)": core & (s3 == "dry"),
        "3-season transition Oct-Dec (Preliminary)": core & (s3 == "transition"),
        "3-season flooded Jan-Apr (Preliminary)": core & (s3 == "flooded"),
        "2014 (sensitivity, 55 % radiation coverage)": L.mask_core(df, (2014,)),
        "2017 (sensitivity, 24 % radiation coverage)": L.mask_core(df, (2017,)),
    }
    # matched months for the like-for-like 2016 vs 2015 comparison
    cnt = df[core].groupby(["year", "month"]).size().unstack(0).reindex(range(1, 13)).fillna(0)
    matched = [m for m in range(1, 13) if cnt.loc[m, 2015] >= C.MATCHED_MONTH_MIN_HH and cnt.loc[m, 2016] >= C.MATCHED_MONTH_MIN_HH]
    log(f"matched months (>= {C.MATCHED_MONTH_MIN_HH} core half-hours in both 2015 and 2016): {matched}")
    periods[f"2015 matched months {matched}"] = core & (yr == 2015) & mo.isin(matched)
    periods[f"2016 matched months {matched}"] = core & (yr == 2016) & mo.isin(matched)
    subsets = {"all": pd.Series(True, index=df.index), "day": df["is_day"], "night": ~df["is_day"]}
    for name, m in periods.items():
        log(f"  period {name}: n = {int(m.sum())}, days with data = {df.loc[m, 'date'].nunique()}")

    # ------------------------------------------------------------------ T0, T0b, T0c
    t0 = L.coverage_table(df)
    write_csv(t0, "t0_data_coverage.csv")
    t0b = L.selection_by_slot(df, core)
    write_csv(t0b, "t0b_selection_fraction_by_slot.csv")
    dawn = t0b[t0b["slot"].isin(DAWN_SLOTS[:3])]
    other = t0b[~t0b["slot"].isin(DAWN_SLOTS[:3])]
    log(f"selection fraction by slot (2015-16): 05:00-06:00 LST {dawn['frac_core'].min():.3f}-{dawn['frac_core'].max():.3f} "
        f"vs {other['frac_core'].min():.3f}-{other['frac_core'].max():.3f} elsewhere (ONEFlux dawn SW_IN QC hole: SW_IN_F_QC = 1 at low sun); "
        f"evening 18:00-20:30 turbulent-QC dip {t0b.loc[t0b['slot'].between(36, 41), 'frac_core'].mean():.3f}")
    log("T0 and T0b written")

    # ------------------------------------------------------------------ core budget
    J = E.budget(df, C.CORE)
    dates = df["date"].to_numpy()
    months = df["month"].to_numpy()
    slots = df["slot"].to_numpy()
    t0c = L.selection_bias_table(df, J["sigma"])
    write_csv(t0c, "t0c_selection_bias_core_vs_excluded.csv")
    for _, r in t0c.iterrows():
        log(f"  selection bias [{r['sample']}]: n {int(r['n'])}, P_F>0 {r['frac_P_F_gt0']:.3f}, SW_IN day {r['SW_IN_day_Wm2']:.0f}, clearness {r['clearness_day']:.3f}, sigma(_F fluxes) {r['sigma_with_F_fluxes']:.3f}")
    log("T0c written")

    # ------------------------------------------------------------------ T1 energy fluxes
    rows = []
    for pname, pm in periods.items():
        for sname, sm in subsets.items():
            m = pm & sm
            d = df[m]
            if len(d) == 0:
                continue
            r = {"period": pname, "subset": sname, "n": int(len(d)),
                 "SW_IN": d["SW_IN_F"].mean(), "SW_OUT": d["SW_OUT"].mean(), "LW_IN": d["LW_IN_F"].mean(), "LW_OUT": d["LW_OUT"].mean(),
                 "absSW": (d["SW_IN_F"] - d["SW_OUT"]).mean(),
                 "Rn": d["Rn"].mean(), "NETRAD": d["NETRAD"].mean(), "H": d["H"].mean(), "LE": d["LE"].mean(), "Rres": d["Rres"].mean(),
                 "albedo_ratio_of_sums_day": (d.loc[d["is_day"], "SW_OUT"].sum() / d.loc[d["is_day"], "SW_IN_F"].sum()) if d["is_day"].any() else np.nan,
                 "Bowen_ratio_of_sums": d["H"].sum() / d["LE"].sum() if d["LE"].sum() != 0 else np.nan,
                 "Ta_K": d["Ta"].mean(), "Ts_K": d["Ts"].mean(), "Tsky_K": d["Tsky"].mean(), "Ts_minus_Ta_K": (d["Ts"] - d["Ta"]).mean(),
                 "TS1_C": d["TS_F_MDS_1"].mean(), "P_F_mm_sum_selected_hh": d["P_F"].sum(), "VPD_hPa": d["VPD_F"].mean(), "WS_ms": d["WS_F"].mean()}
            rows.append(r)
    t1 = pd.DataFrame(rows)
    write_csv(t1, "t1_energy_fluxes.csv")
    log("T1 written")

    # ------------------------------------------------------------------ T2 entropy budget (core config)
    rows = []
    hours = df["hour"]
    for pname, pm in periods.items():
        for sname, sm in subsets.items():
            m = pm & sm
            if m.sum() == 0:
                continue
            r = {"period": pname, "subset": sname}
            r.update(S.period_stats(J, m))
            r.update(S.bootstrap_summary(J, dates, m, block_days=BD_H, suffix=f"_{BD_H}d"))
            r.update(S.bootstrap_summary(J, dates, m, block_days=BD_O, suffix=f"_{BD_O}d"))
            if sname == "all":
                r.update(S.stratified_mean(J, months, slots, m))
                bs, names = S.bootstrap_stratified(J, dates, months, slots, m, block_days=BD_H)
                for nm in ("sigma", "SR"):
                    lo, hi = S.ci(bs[:, names.index(nm)])
                    r[f"{nm}_strat_lo_{BD_H}d"] = float(lo); r[f"{nm}_strat_hi_{BD_H}d"] = float(hi)
                daily = S.daily_means(J, df["date"], m)
                r["n_complete_days"] = int(len(daily))
                r["SR_daily_Lin"] = S.sr_daily(daily)
                r["sigma_daily_mean"] = float(daily["sigma"].mean()) if len(daily) else np.nan
                r["SR_midday_11_13"], r["n_midday"] = S.sr_window(J, hours, m, C.MIDDAY_HOURS)
                r["SR_midnight_23_01"], r["n_midnight"] = S.sr_window(J, hours, m, C.MIDNIGHT_HOURS)
                me = m & (df["sun_elev"] > C.MIDDAY_ELEV_DEG)
                r["SR_elev_gt60"] = float(J.loc[me, "J_out"].mean() / J.loc[me, "J_in"].mean()) if me.sum() else np.nan
                r["n_elev_gt60"] = int(me.sum())
            rows.append(r)
    t2 = pd.DataFrame(rows)
    write_csv(t2, "t2_entropy_budget.csv")
    log("T2 written (raw and stratified means; bootstrap intervals with 7-day blocks (headline) and 1-day blocks (optimistic))")

    # analytic decomposition (spec 2.7) per period
    A = E.sigma_analytic_terms(J)
    rows = []
    for pname, pm in periods.items():
        for sname, sm in subsets.items():
            m = pm & sm
            if m.sum() == 0:
                continue
            a = A[m].mean()
            rows.append({"period": pname, "subset": sname, "n": int(m.sum()), "sigma_S0": float(J.loc[m, "sigma"].mean()),
                         "absSW_over_Ts": a["absSW_over_Ts"], "minus_J_SWnet": a["minus_J_SWnet"], "LWterm": a["LWterm"],
                         "LWterm_second_order_approx": a["LWterm_second_order"], "minus_Rres_over_Ts": a["minus_Rres_over_Ts"],
                         "sigma_reconstructed": a["sigma_reconstructed"],
                         "sigma_S1": float(J.loc[m, "sigma"].mean() - a["minus_Rres_over_Ts"]),
                         "sigma_S1_analytic_absSW_x_(1/Ts-4/3Tsun)": a["sigma_S1_analytic"],
                         "sigma_per_absSW_Kinv": float(J.loc[m, "sigma"].mean() / J.loc[m, "absSW"].mean()),
                         "sigma_S1_per_absSW_Kinv": float((J.loc[m, "sigma"].mean() - a["minus_Rres_over_Ts"]) / J.loc[m, "absSW"].mean()),
                         "absSW_Wm2": float(J.loc[m, "absSW"].mean()), "Ts_K": float(J.loc[m, "Ts"].mean())})
    t2c = pd.DataFrame(rows)
    write_csv(t2c, "t2c_analytic_decomposition.csv")
    a = t2c[(t2c["period"] == "2015-2016") & (t2c["subset"] == "all")].iloc[0]
    log(f"analytic decomposition 2015-16 all: sigma_S0 {a['sigma_S0']:.4f} = absSW/Ts {a['absSW_over_Ts']:.4f} - J_SWnet {a['minus_J_SWnet']:+.4f} + LWterm {a['LWterm']:+.4f} - R'/Ts {a['minus_Rres_over_Ts']:+.4f} "
        f"(reconstructed {a['sigma_reconstructed']:.4f}); sigma_S1 {a['sigma_S1']:.4f} vs absSW(1/Ts-4/3Tsun) {a['sigma_S1_analytic_absSW_x_(1/Ts-4/3Tsun)']:.4f}")
    for pn in ("wet pooled (Nov-Apr)", "dry pooled (May-Oct)", "3-season dry May-Sep (Preliminary)", "3-season transition Oct-Dec (Preliminary)", "3-season flooded Jan-Apr (Preliminary)"):
        a = t2c[(t2c["period"] == pn) & (t2c["subset"] == "all")].iloc[0]
        log(f"  {pn}: sigma {a['sigma_S0']:.4f}, absSW {a['absSW_Wm2']:.1f} W m-2, sigma/absSW {a['sigma_per_absSW_Kinv']:.5f} K-1 (S1: {a['sigma_S1_per_absSW_Kinv']:.5f}; reference 1/Ts-4/3Tsun {1 / a['Ts_K'] - 4 / (3 * C.T_SUN):.5f})")
    an = t2c[(t2c["period"] == "2015-2016") & (t2c["subset"] == "night")].iloc[0]
    log(f"  night 2015-16: J_out-J_in {an['sigma_S0']:.4f}; R'/Ts {-an['minus_Rres_over_Ts']:+.4f}; S1 (= storage interpretation with T_store ~ T_s) {an['sigma_S1']:.4f}; "
        f"expected second-order radiative production LWterm {an['LWterm']:+.4f} (2 sigma_SB Ts dT^2: {an['LWterm_second_order_approx']:.4f})")
    # year-month correlation sigma vs absorbed SW
    ym = pd.DataFrame({"sigma": J.loc[core, "sigma"].to_numpy(), "absSW": J.loc[core, "absSW"].to_numpy(),
                       "ym": (df.loc[core, "year"] * 100 + df.loc[core, "month"]).to_numpy()}).groupby("ym").mean()
    log(f"  year-month means (n = {len(ym)}): corr(sigma, absSW) = {np.corrcoef(ym['sigma'], ym['absSW'])[0, 1]:.3f}")
    log("T2c written")

    # ------------------------------------------------------------------ T2b bootstrap differences
    diffs = []
    pairs = [("wet pooled (Nov-Apr)", "dry pooled (May-Oct)"),
             ("wet 2016 (Nov 2015-Apr 2016)", "dry 2016 (May-Oct 2016)"),
             ("wet 2016 (Nov 2015-Apr 2016)", "dry 2015 (May-Oct 2015)"),
             ("2016", "2015"),
             (f"2016 matched months {matched}", f"2015 matched months {matched}"),
             ("3-season flooded Jan-Apr (Preliminary)", "3-season dry May-Sep (Preliminary)")]
    for bd in (BD_H, BD_O):
        for a_, b_ in pairs:
            r = {"a": a_, "b": b_, "block_days": bd, "headline": bd == BD_H}
            r.update(S.bootstrap_difference(J, dates, periods[a_], periods[b_], block_days=bd))
            diffs.append(r)
    # per-month 2016-2015 differences on matched months
    pm = pd.DataFrame({"sigma": J.loc[core, "sigma"].to_numpy(), "year": yr[core].to_numpy(), "month": mo[core].to_numpy()})
    pmt = pm.groupby(["month", "year"])["sigma"].mean().unstack(1).loc[matched]
    pmt["d_2016_minus_2015"] = pmt[2016] - pmt[2015]
    log(f"per-month 2016-2015 sigma differences (matched months): {pmt['d_2016_minus_2015'].round(4).to_dict()}; mean {pmt['d_2016_minus_2015'].mean():+.4f}")
    t2b = pd.DataFrame(diffs)
    write_csv(t2b, "t2b_bootstrap_differences.csv")
    write_csv(pmt.reset_index().rename(columns={2015: "sigma_2015", 2016: "sigma_2016"}), "t2d_matched_month_differences.csv")
    # block-length sensitivity of the headline interval (documented, not only checked)
    bl_rows = []
    for bd in (1, 3, 7, 14, 30):
        b, names = S.bootstrap_means(J, dates, core, block_days=bd)
        lo, hi = S.ci(b[:, names.index("sigma")])
        dd = S.bootstrap_difference(J, dates, periods["wet pooled (Nov-Apr)"], periods["dry pooled (May-Oct)"], block_days=bd)
        bl_rows.append({"block_days": bd, "sigma_lo": float(lo), "sigma_hi": float(hi), "half_width": float((hi - lo) / 2),
                        "d_wet_dry": dd["d_sigma"], "d_wet_dry_lo": dd["d_sigma_lo"], "d_wet_dry_hi": dd["d_sigma_hi"]})
    t2e = pd.DataFrame(bl_rows)
    write_csv(t2e, "t2e_block_length_sensitivity.csv")
    # autocorrelation of daily sigma anomalies (justification of the block length)
    daily_all = S.daily_means(J, df["date"], core)
    dsig = daily_all["sigma"]
    idx = pd.DatetimeIndex(dsig.index)
    anom = dsig - dsig.groupby(idx.month).transform("mean")
    full = pd.Series(np.nan, index=pd.date_range(idx.min(), idx.max(), freq="D"))
    full.loc[idx] = anom.to_numpy()
    acf = {k: float(full.autocorr(lag=k)) for k in (1, 2, 3, 5, 7, 14)}
    acf_raw = {k: float(pd.Series(np.nan, index=full.index).combine_first(pd.Series(dsig.to_numpy(), index=idx)).autocorr(lag=k)) for k in (1, 2, 3, 7)}
    log(f"daily-mean sigma autocorrelation (complete days, calendar lags): raw {acf_raw}; anomalies from month means {acf}")
    for _, r in t2e.iterrows():
        log(f"  block {int(r['block_days']):2d} d: sigma 95 % {r['sigma_lo']:.4f}-{r['sigma_hi']:.4f} (half-width {r['half_width']:.4f}); wet-dry {r['d_wet_dry']:+.4f} [{r['d_wet_dry_lo']:.4f}, {r['d_wet_dry_hi']:.4f}]")
    log("T2b, T2d, T2e written")

    # ------------------------------------------------------------------ T3 closure
    d_core = df[core]
    cr_core = CL.cr_sum(d_core["H"], d_core["LE"], d_core["Rn"])
    log(f"closure ratio (sum) core 2015-2016, half-hourly: {cr_core:.4f} (used for S3 everywhere)")
    J_sc = {sc: E.budget(df, C.Options(closure=sc), cr=cr_core) for sc in ("S0", "S1", "S2", "S3")}
    rows = []
    for pname, pm_ in periods.items():
        for sname, sm in subsets.items():
            m = pm_ & sm
            d = df[m]
            if len(d) == 0:
                continue
            r = {"period": pname, "subset": sname}
            r.update(CL.closure_stats(d))
            if sname == "all":
                r.update(CL.daily_closure(d))
            for sc, Jsc in J_sc.items():
                ps = S.period_stats(Jsc, m)
                r[f"sigma_{sc}"] = ps["sigma"]; r[f"SR_{sc}"] = ps["SR"]
            r["sigma_S1_minus_S0"] = r["sigma_S1"] - r["sigma_S0"]
            rows.append(r)
    t3 = pd.DataFrame(rows)
    write_csv(t3, "t3_closure_scenarios.csv")
    log("T3 written")

    # ------------------------------------------------------------------ T4 sensitivity (under S0 and, for every non-closure row, also under S1)
    runs = [("core", C.CORE, core)]
    for e in C.EPS_SENSITIVITY:
        runs.append((f"eps_s = {e}", C.Options(eps=e), core))
    runs += [("J_LWin with T_a", C.Options(lwin_temp="air"), core),
             ("J_LWin diluted X(delta_a), T_a", C.Options(lwin_temp="diluted"), core),
             ("J_H, J_LE with T_a", C.Options(turb_temp="ta"), core),
             ("J_SWout variant (b) diluted", C.Options(swout_variant="b"), core),
             ("J_SWin hemispheric-diluted upper bound", C.Options(swin_variant="diluted"), core),
             ("J_LWout greybody eq. 11", C.Options(lwout_variant="grey"), core),
             ("J_LWout greybody + diluted reflection (upper bound)", C.Options(lwout_variant="grey_refl"), core),
             ("T_sun = 5778 K", C.Options(t_sun=C.T_SUN_ALT), core),
             ("closure S1 (residual to LE)", C.Options(closure="S1"), core),
             ("closure S2 (residual to H)", C.Options(closure="S2"), core),
             ("closure S3 (Bowen-preserving, period CR)", C.Options(closure="S3"), core)]
    for var, (kind, val) in C.RAD_BIAS.items():
        for sgn in (+1, -1):
            lab = f"bias {var} {'+' if sgn > 0 else '-'}{val * 100:.0f} %" if kind == "rel" else f"bias {var} {'+' if sgn > 0 else '-'}{val:.0f} W m-2"
            runs.append((lab, C.Options(bias=((var, kind, sgn * val),)), core))
    runs.append(("sample: dawn-relaxed (SW_IN_F_QC<=1 when SW_IN_POT<50)", C.CORE, L.mask_core_dawn_relaxed(df)))
    runs.append(("sample: gap-filled _F fluxes (not measurements)", C.CORE, L.mask_gapfilled(df)))
    runs.append(("sample: radiation-only (J_rad terms; sigma_rad only)", C.CORE, L.mask_radiation_only(df)))
    rows = []
    for lab, opts, m in runs:
        Jr = E.budget(df, opts, cr=cr_core if opts.closure == "S3" else None)
        a = S.period_stats(Jr, m); w = S.period_stats(Jr, m & (se == "wet")); dr = S.period_stats(Jr, m & (se == "dry"))
        row = {"label": lab, "n": a["n"], "sigma": a["sigma"], "SR": a["SR"], "sigma_rad": a["sigma_rad"], "sigma_turb": a["sigma_turb"],
               "J_in": a["J_in"], "J_out": a["J_out"], "sigma_per_absSW_Kinv": a["sigma_per_absSW_Kinv"],
               "sigma_wet": w["sigma"], "sigma_dry": dr["sigma"], "d_wet_minus_dry": w["sigma"] - dr["sigma"],
               "SR_wet": w["SR"], "SR_dry": dr["SR"]}
        if opts.closure == "S0":
            J1 = E.budget(df, C.Options(**{**opts.to_dict(), "closure": "S1"}))
            a1 = S.period_stats(J1, m)
            row["sigma_S1"] = a1["sigma"]; row["SR_S1"] = a1["SR"]
        else:
            row["sigma_S1"] = np.nan; row["SR_S1"] = np.nan
        rows.append(row)
    t4 = pd.DataFrame(rows)
    core_row = t4.iloc[0]
    t4["d_sigma_vs_core"] = t4["sigma"] - core_row["sigma"]
    t4["d_sigma_S1_vs_core_S1"] = t4["sigma_S1"] - core_row["sigma_S1"]
    t4["d_SR_vs_core"] = t4["SR"] - core_row["SR"]
    t4["d_sigma_rad_vs_core"] = t4["sigma_rad"] - core_row["sigma_rad"]
    # radiation-only sample: H/LE not all measured -> sigma not meaningful; blank it
    ro = t4["label"].str.startswith("sample: radiation-only")
    t4.loc[ro, ["sigma", "SR", "sigma_turb", "J_out", "sigma_wet", "sigma_dry", "d_wet_minus_dry", "SR_wet", "SR_dry", "d_sigma_vs_core", "d_SR_vs_core",
                "sigma_S1", "SR_S1", "d_sigma_S1_vs_core_S1", "sigma_per_absSW_Kinv"]] = np.nan
    write_csv(t4, "t4_sensitivity.csv")
    log("T4 written (every non-closure row also evaluated under closure S1)")

    # ------------------------------------------------------------------ T5 monthly climatology
    t5 = S.monthly_table(J, df, core, by_year=False)
    t5y = S.monthly_table(J, df, core, by_year=True)
    write_csv(t5, "t5_monthly_climatology.csv")
    write_csv(t5y, "t5_monthly_climatology_by_year.csv")
    log("T5 written (P_F column restricted to the core years)")

    # ------------------------------------------------------------------ T6 literature (abstract-verified values only)
    h2 = t2[(t2["period"] == "2015-2016") & (t2["subset"] == "all")].iloc[0]
    t6 = pd.DataFrame([
        {"site": "BR-Npw (this study, core, raw mean)", "biome": "seasonally flooded woody savanna, Pantanal", "period": "2015-2016 measured half-hours",
         "formulation": "sigma = J_out - J_in; SW at T_sun (4/3); LW_IN at T_sky; LW_OUT at T_s (eps 0.98); H, LE at T_s; closure S0", "sigma_Wm2K": f"{core_row['sigma']:.4f}", "SR": f"{core_row['SR']:.4f}", "source": "this run", "status": "Reproduced (this run)"},
        {"site": "BR-Npw (this study, core, stratified month x slot mean)", "biome": "as above", "period": "as above",
         "formulation": "as above; climatological weighting", "sigma_Wm2K": f"{h2['sigma_strat']:.4f}", "SR": f"{h2['SR_strat']:.4f}", "source": "this run", "status": "Reproduced (this run)"},
        {"site": "BR-Npw (this study, closure S1-S3)", "biome": "as above", "period": "as above",
         "formulation": "residual R' attributed to turbulent fluxes", "sigma_Wm2K": f"{t4.loc[t4['label'].str.startswith('closure S1'), 'sigma'].iloc[0]:.4f}-{t4.loc[t4['label'].str.startswith('closure S3'), 'sigma'].iloc[0]:.4f}",
         "SR": f"{t4.loc[t4['label'].str.startswith('closure S1'), 'SR'].iloc[0]:.4f}-{t4.loc[t4['label'].str.startswith('closure S3'), 'SR'].iloc[0]:.4f}", "source": "this run", "status": "Reproduced (this run)"},
        {"site": "Amazon forest (Holdaway et al. 2010)", "biome": "tropical forest", "period": "", "formulation": "entropy production from EC-tower energy budget; details to confirm against full text", "sigma_Wm2K": "0.461", "SR": "", "source": "Holdaway, Sparrow & Coomes 2010, Phil Trans R Soc B 365:1437 (abstract)", "status": "Verified abstract"},
        {"site": "Amazon pasture (Holdaway et al. 2010)", "biome": "pasture", "period": "", "formulation": "as above", "sigma_Wm2K": "0.422", "SR": "", "source": "Holdaway et al. 2010 (abstract)", "status": "Verified abstract"},
        {"site": "7 AmeriFlux sites (Lin 2015), range", "biome": "various", "period": "", "formulation": "Delta S_e = J_in - J_out (negative); SR from daily-averaged fluxes", "sigma_Wm2K": "0.258-0.609 (|Delta S_e|)", "SR": "1.23-1.53 (mean 1.35 +/- 0.10)", "source": "Lin 2015, Ecol Model 298:75 (abstract)", "status": "Verified abstract"},
        {"site": "Lin 2015 midday / midnight SR", "biome": "", "period": "", "formulation": "", "sigma_Wm2K": "", "SR": "midday 1.46-2.23; midnight 1.09-1.22", "source": "Lin 2015 (abstract)", "status": "Verified abstract"},
    ])
    write_csv(t6, "t6_literature_comparison.csv")
    log("T6 written")

    # ------------------------------------------------------------------ diurnal composites + storage scenarios
    stor = ST.storage_scenarios(df, J["Ts"])
    soil_cols = [ST.soil_name(c, z) for c in C.C_SOIL_RANGE for z in C.Z_SOIL_RANGE]
    water_cols = [ST.water_name(h) for h in C.H_WATER_RANGE]
    soil_max_key = ST.soil_name(max(C.C_SOIL_RANGE), max(C.Z_SOIL_RANGE))
    stor["dQ_soil_min"] = stor[soil_cols].min(axis=1); stor["dQ_soil_max"] = stor[soil_cols].max(axis=1)
    stor["dQ_water_min"] = stor[water_cols].min(axis=1); stor["dQ_water_max"] = stor[water_cols].max(axis=1)
    dS_soil = stor[["dS_" + c for c in soil_cols]]
    sig_soil_lo = J["sigma"] + dS_soil.min(axis=1)
    sig_soil_hi = J["sigma"] + dS_soil.max(axis=1)
    comp = {}
    comp_rows = []
    slots_s = df["slot"]
    series = {"sigma": J["sigma"], "sigma_rad": J["sigma_rad"], "sigma_turb": J["sigma_turb"],
              "J_LWin": J["J_LWin"], "J_LWout": J["J_LWout"], "J_SWin": J["J_SWin"], "J_SWout": J["J_SWout"], "J_H": J["J_H"], "J_LE": J["J_LE"],
              "sigma_soil_lo": sig_soil_lo, "sigma_soil_hi": sig_soil_hi,
              "Rn": df["Rn"], "H": df["H"], "LE": df["LE"], "Rres": df["Rres"],
              "dQ_soil_min": stor["dQ_soil_min"], "dQ_soil_max": stor["dQ_soil_max"],
              "dQ_water_min": stor["dQ_water_min"], "dQ_water_max": stor["dQ_water_max"],
              "Ts_minus_Ta": J["Ts"] - J["Ta"], "Tsky_minus_Ta": J["Tsky"] - J["Ta"]}
    for season in ("wet", "dry"):
        m = core & (se == season)
        comp[season] = {}
        for name, v in series.items():
            c = S.diurnal_composite(v, slots_s, dates, m, n_boot=C.BOOT_N)   # every series bootstrapped (1000 day-resamples)
            comp[season][name] = c
            cc = c.copy(); cc.insert(0, "units", T7_UNITS.get(name, "")); cc.insert(0, "variable", name); cc.insert(0, "season", season)
            cc["note"] = np.where(cc["slot"].isin(DAWN_SLOTS), "conditional: dawn slots under-sampled by ONEFlux SW_IN QC (05:00-06:30 LST)", "")
            comp_rows.append(cc)
    t7 = pd.concat(comp_rows, ignore_index=True)
    write_csv(t7, "t7_diurnal_composites.csv")
    log("T7 (diurnal composites incl. storage scenarios) written; lo/hi = 2.5-97.5 % day-resampling percentiles, n_boot = 1000 for every variable")
    # storage diagnostics summary
    def _rng(c):
        v = c["mean"].to_numpy(dtype=float)
        return "n/a" if not np.isfinite(v).any() else f"{np.nanmin(v):.1f} .. {np.nanmax(v):.1f}"
    skin = stor.loc[core & df["month"].isin(C.FLOODED_MONTHS_SCENARIO), ST.water_name(1.0, "skin")]
    log(f"  water-column scenario with the skin-temperature proxy (1.0 m, Jan-Apr core half-hours): 1-99 % range {skin.quantile(0.01):.0f} .. {skin.quantile(0.99):.0f} W m-2 -> rejected as implausible; TS_1 proxy used instead")
    for season in ("wet", "dry"):
        m = core & (se == season)
        log(f"  storage scenario amplitude ({season}): soil dQ composite means {_rng(comp[season]['dQ_soil_min'])} / {_rng(comp[season]['dQ_soil_max'])} W m-2 (min/max scenario); "
            f"water 0.2 m {_rng(comp[season]['dQ_water_min'])}, 1.0 m {_rng(comp[season]['dQ_water_max'])} W m-2 (Jan-Apr only, TS_1 proxy)")
        log(f"  daily-mean storage check ({season}): mean dQ_soil(max) over selected half-hours = {stor.loc[m, 'dQ_soil_max'].mean():.3f} W m-2; mean dS_soil(max) = {dS_soil.loc[m].max(axis=1).mean():.5f} W m-2 K-1")
        sig_st = J["sigma"] + stor["dS_" + soil_max_key]
        log(f"  day/night exchange vs storage-corrected ({season}): exchange day {J.loc[m & df['is_day'], 'sigma'].mean():.3f}, night {J.loc[m & ~df['is_day'], 'sigma'].mean():.3f}; "
            f"with soil storage ({soil_max_key}): day {sig_st[m & df['is_day']].mean():.3f}, night {sig_st[m & ~df['is_day']].mean():.3f}")
    # nighttime balance: exchange vs storage release (flag 2)
    mn = core & ~df["is_day"]
    t_store_n = (df.loc[mn, "TS_F_MDS_1"] + C.T0)
    log(f"  night 2015-16: J_out-J_in {J.loc[mn, 'sigma'].mean():.4f}; R'/T_s {(df.loc[mn, 'Rres'] / J.loc[mn, 'Ts']).mean():+.4f}; R'/T_store(TS_1) {(df.loc[mn, 'Rres'] / t_store_n).mean():+.4f}; "
        f"S1 night {J_sc['S1'].loc[mn, 'sigma'].mean():+.4f}; exchange + R'/T_store {(J.loc[mn, 'sigma'] + df.loc[mn, 'Rres'] / t_store_n).mean():+.4f}; "
        f"dsigma/dLW_OUT from T4 {(t4.loc[t4['label'] == 'bias LW_OUT +10 W m-2', 'd_sigma_vs_core'].iloc[0] / 10):.4f} per W m-2")
    # T_s - T_a vs H diagnostic (flag 1)
    dcore = df[core].assign(Ts=J.loc[core, "Ts"], Ta=J.loc[core, "Ta"])
    h0, h1 = C.NOON_DIAG_HOURS
    mid_d = dcore[(dcore["hour"] >= h0) & (dcore["hour"] < h1)]
    x = (mid_d["Ts"] - mid_d["Ta"]).to_numpy(); yv = mid_d["H"].to_numpy()
    log(f"  H vs Ts-Ta, {h0:02d}:00-{h1:02d}:00 LST core: n {len(x)}, corr {np.corrcoef(x, yv)[0, 1]:.3f}, OLS slope {np.polyfit(x, yv, 1)[0]:.1f} W m-2 K-1, frac Ts<Ta {(x < 0).mean():.3f}, mean dT {x.mean():+.2f} K; "
        f"H>50: frac Ts<Ta {(x[yv > 50] < 0).mean():.3f}, mean dT {x[yv > 50].mean():+.2f} K")
    for season in ("wet", "dry"):
        c = comp[season]
        for h_ in (12.0, 15.0, 16.5):
            i = int(h_ * 2)
            log(f"  composite {season} {h_:04.1f} LST: H {c['H']['mean'][i]:+.1f}, LE {c['LE']['mean'][i]:+.1f} W m-2, Ts-Ta {c['Ts_minus_Ta']['mean'][i]:+.2f} K")
    for yv_ in (2014, 2017):
        mk = L.mask_core(df, (yv_,)); jj = E.budget(df, C.CORE)[mk]; dd = df[mk]
        log(f"  Ts-Ta {yv_}: day {(jj['Ts'] - jj['Ta'])[dd['is_day']].mean():+.2f} K, night {(jj['Ts'] - jj['Ta'])[~dd['is_day']].mean():+.2f} K")

    # T_s - T_a by eps for F5
    dT = {}
    for season in ("wet", "dry"):
        m = core & (se == season)
        dT[season] = {}
        for e in C.EPS_FIGURE:
            ts_e = E.t_surface(df["LW_OUT"], df["LW_IN_F"], e)
            dT[season][e] = S.diurnal_composite(ts_e - J["Ta"], slots_s, dates, m, n_boot=0)
    dTsky = {s_: comp[s_]["Tsky_minus_Ta"] for s_ in ("wet", "dry")}
    # midday albedo by month
    mh0, mh1 = C.MIDDAY_HOURS
    mid = core & (df["hour"] >= mh0) & (df["hour"] < mh1)
    arows = []
    for y in (2015, 2016, "pooled"):
        my = mid if y == "pooled" else mid & (yr == y)
        g = df[my].groupby("month")
        a_ = (g["SW_OUT"].sum() / g["SW_IN_F"].sum()).rename("albedo_midday").reset_index()
        a_["n"] = g.size().to_numpy(); a_["year"] = y
        arows.append(a_)
    t8 = pd.concat(arows, ignore_index=True)
    write_csv(t8, "t8_midday_albedo_by_month.csv")
    log("T8 (midday albedo) written")

    # daily table for F3/F4
    daily = S.daily_means(J, df["date"], core)
    dd = df[core].groupby("date")[["Rn", "H", "LE", "P_F"]].mean()
    daily = daily.join(dd[["H", "LE", "P_F"]])
    daily["season"] = np.where(pd.DatetimeIndex(daily.index).month.isin(C.WET_MONTHS), "wet", "dry")
    daily.index.name = "date"
    write_csv(daily.reset_index(), "t9_daily_means_complete_days.csv")
    log(f"T9 daily means written: {len(daily)} complete days (>= {C.MIN_HH_PER_DAY} selected half-hours)")

    # ------------------------------------------------------------------ figures
    F.f1_monthly(t5, t5y)
    F.f2_diurnal(comp)
    F.f3_sigma_vs_rn(daily.reset_index())
    F.f4_closure(d_core, daily.reset_index(), {s_: comp[s_]["Rres"] for s_ in ("wet", "dry")})
    F.f5_diagnostics(dT, dTsky, t8)
    F.f6_tornado(t4)
    F.f7_h_vs_dt(dcore)
    log("figures F1-F7 written (pdf + png)")

    # ------------------------------------------------------------------ Markdown renderings + INDEX
    prov = (f"Config hash {C.config_hash()}; code hash {C.code_hash()}; data sha256 {sha[:12]}...; seed {C.BOOT_SEED}; "
            f"core configuration: eps_s={C.EPS_CORE}, T_sun={C.T_SUN} K, J_LWin at T_sky, J_H/J_LE at T_s, J_SWout variant (a), closure S0; "
            f"bootstrap intervals: {BD_H}-day moving blocks (headline), {BD_O}-day (optimistic).")
    md = ["# Main tables (Markdown rendering; CSV files are the source)", "", prov, "",
          "## T2 entropy budget (W m-2 K-1), core configuration", "",
          df_to_md(t2, ["period", "subset", "n", "J_SWin", "J_SWout", "J_LWin", "J_LWout", "J_H", "J_LE", "J_in", "J_out", "sigma",
                        f"sigma_boot_lo_{BD_H}d", f"sigma_boot_hi_{BD_H}d", "sigma_strat", f"sigma_strat_lo_{BD_H}d", f"sigma_strat_hi_{BD_H}d",
                        "sigma_rad", "sigma_turb", "SR", f"SR_boot_lo_{BD_H}d", f"SR_boot_hi_{BD_H}d", "SR_strat", "SR_daily_Lin", "SR_midday_11_13", "SR_midnight_23_01",
                        "sigma_per_absSW_Kinv", "eta_sigma_Rnabs_Kinv", "u_sigma_rand_mean"]), "",
          "## T2b bootstrap differences", "", df_to_md(t2b), "",
          "## T2c analytic decomposition (W m-2 K-1; sigma/absSW in K-1)", "", df_to_md(t2c), "",
          "## T2e block-length sensitivity of the 2015-2016 interval", "", df_to_md(t2e), "",
          "## T3 closure scenarios", "",
          df_to_md(t3, ["period", "subset", "n_hh", "CR_sum_hh", "ols_slope", "ols_intercept_Wm2", "ols_r2", "Rres_mean_Wm2", "frac_Rres_negative", "n_days", "CR_sum_daily", "sigma_S0", "sigma_S1", "sigma_S2", "sigma_S3", "SR_S0", "SR_S1", "SR_S2", "SR_S3"]), "",
          "## T4 sensitivity (2015-2016 core sample unless stated; sigma under S0 and S1)", "", df_to_md(t4), ""]
    (C.TABLES_DIR / "MAIN_TABLES.md").write_text("\n".join(md), encoding="utf-8")
    index = ["# results/tables and results/figures index", "",
             f"Generated by code/run_all.py on {time.strftime('%Y-%m-%d %H:%M:%S')} (local container time).", "",
             f"Provenance: config hash {C.config_hash()}; code hash {C.code_hash()} (sha256 of code/*.py); data {C.HH_FILE.name} sha256 {sha}; seed {C.BOOT_SEED}; n_boot {C.BOOT_N}.", "",
             "Units: energy fluxes W m-2; entropy fluxes and sigma W m-2 K-1; temperatures K (TA_C, TS1_C in degC); sigma_per_absSW and eta_sigma K-1; SR dimensionless; P_F mm.",
             "Time labelling: TIMESTAMP_START (hour 11.5 = 11:30-12:00 LST, UTC-4). Columns *_lo_7d/_hi_7d are 2.5-97.5 % percentiles of a 7-day moving-block day bootstrap (headline); *_1d the 1-day (independent-days, optimistic) version.",
             "", "| file | content | units / notes | script |", "|---|---|---|---|"]
    items = [("t0_data_coverage.csv", "per year-month counts, complete days, P_F sum, mean TA", "counts; mm; degC", "load.coverage_table"),
             ("t0b_selection_fraction_by_slot.csv", "fraction of 2015-16 half-hours passing each filter, by 30-min slot (dawn hole)", "fractions", "load.selection_by_slot"),
             ("t0c_selection_bias_core_vs_excluded.csv", "rain, radiation, clearness: all vs core vs excluded-with-radiation", "fractions; W m-2; sigma W m-2 K-1 (_F fluxes)", "load.selection_bias_table"),
             ("t1_energy_fluxes.csv", "energy fluxes and temperatures per period x all/day/night", "W m-2, K", "run_all"),
             ("t2_entropy_budget.csv", "entropy terms, sigma raw and stratified (month x slot), SR, bootstrap 95 % (7-day and 1-day blocks), SR daily/midday/midnight, sigma per absorbed SW", "W m-2 K-1; K-1", "run_all + stats"),
             ("t2b_bootstrap_differences.csv", "bootstrap of wet-dry, 2016-2015 (raw and matched months), flooded-dry differences; 7-day (headline) and 1-day blocks", "W m-2 K-1; frac_le0 as text bound", "stats.bootstrap_difference"),
             ("t2c_analytic_decomposition.csv", "sigma = absSW/Ts - J_SWnet + LWterm - R'/Ts per period; sigma per absorbed SW", "W m-2 K-1; K-1", "entropy.sigma_analytic_terms"),
             ("t2d_matched_month_differences.csv", "per-month 2016 - 2015 sigma on the matched months", "W m-2 K-1", "run_all"),
             ("t2e_block_length_sensitivity.csv", "95 % interval of sigma 2015-16 and of wet-dry vs bootstrap block length", "W m-2 K-1", "stats.bootstrap_means"),
             ("t3_closure_scenarios.csv", "closure ratio, OLS, residual; sigma/SR under S0-S3", "W m-2; W m-2 K-1", "closure + entropy.closure_fluxes"),
             ("t4_sensitivity.csv", "one-at-a-time methodological, bias and sample sensitivity; sigma under S0 and under S1", "W m-2 K-1", "run_all"),
             ("t5_monthly_climatology.csv", "monthly means pooled 2015-16 (P_F mean of the two core years)", "W m-2 K-1; W m-2; K; mm", "stats.monthly_table"),
             ("t5_monthly_climatology_by_year.csv", "monthly means per year", "as above", "stats.monthly_table"),
             ("t6_literature_comparison.csv", "abstract-verified literature anchors", "W m-2 K-1", "run_all"),
             ("t7_diurnal_composites.csv", "30-min composites by season of all terms, residual and storage scenarios (Preliminary); units column; lo/hi 2.5-97.5 % of 1000 day-resamples; dawn slots flagged conditional", "see units column", "stats.diurnal_composite + storage"),
             ("t8_midday_albedo_by_month.csv", "midday albedo (ratio of sums) by month and year", "dimensionless", "run_all"),
             ("t9_daily_means_complete_days.csv", "daily means of all terms over complete days", "W m-2 K-1; W m-2; mm", "stats.daily_means"),
             ("MAIN_TABLES.md", "Markdown rendering of T2, T2b, T2c, T2e, T3, T4", "", "run_all"),
             ("../figures/f1_monthly_climatology.pdf/.png", "monthly climatology of J terms, sigma, SR", "", "figures.f1_monthly"),
             ("../figures/f2_diurnal_composite.pdf/.png", "wet/dry diurnal composites with storage scenario bands", "", "figures.f2_diurnal"),
             ("../figures/f3_sigma_vs_rn.pdf/.png", "daily sigma vs Rn and vs LE by season", "", "figures.f3_sigma_vs_rn"),
             ("../figures/f4_closure.pdf/.png", "energy-balance closure, half-hourly and daily, residual composite", "", "figures.f4_closure"),
             ("../figures/f5_ts_minus_ta.pdf/.png", "Ts-Ta (eps 0.95-0.99) and Tsky-Ta composites, midday albedo by month", "", "figures.f5_diagnostics"),
             ("../figures/f6_sensitivity_tornado.pdf/.png", "tornado of delta sigma per factor (S0)", "", "figures.f6_tornado"),
             ("../figures/f7_h_vs_ts_minus_ta.pdf/.png", "diagnostic: H vs Ts-Ta, 10:00-14:00 LST, by season", "", "figures.f7_h_vs_dt"),
             ("../run_log.txt", "run log: versions, provenance, filters, flags, diagnostics, timings", "", "run_all")]
    for f, c, u, s_ in items:
        index.append(f"| {f} | {c} | {u} | {s_} |")
    (C.TABLES_DIR / "INDEX.md").write_text("\n".join(index), encoding="utf-8")

    # ------------------------------------------------------------------ headline numbers to log
    h = h2
    log(f"HEADLINE sigma 2015-2016 raw = {h['sigma']:.4f} (95 % {BD_H}-day blocks {h[f'sigma_boot_lo_{BD_H}d']:.4f}-{h[f'sigma_boot_hi_{BD_H}d']:.4f}; 1-day {h[f'sigma_boot_lo_{BD_O}d']:.4f}-{h[f'sigma_boot_hi_{BD_O}d']:.4f}); "
        f"stratified month x slot = {h['sigma_strat']:.4f} ({h[f'sigma_strat_lo_{BD_H}d']:.4f}-{h[f'sigma_strat_hi_{BD_H}d']:.4f}); daily-mean over complete days {h['sigma_daily_mean']:.4f}; "
        f"SR raw = {h['SR']:.4f} ({h[f'SR_boot_lo_{BD_H}d']:.4f}-{h[f'SR_boot_hi_{BD_H}d']:.4f}), stratified {h['SR_strat']:.4f}; SR_daily(Lin) = {h['SR_daily_Lin']:.4f}; "
        f"sigma/absSW {h['sigma_per_absSW_Kinv']:.5f} K-1")
    for pn in ("wet pooled (Nov-Apr)", "dry pooled (May-Oct)", "wet 2016 (Nov 2015-Apr 2016)", "dry 2015 (May-Oct 2015)", "dry 2016 (May-Oct 2016)", "2015", "2016",
               "3-season dry May-Sep (Preliminary)", "3-season transition Oct-Dec (Preliminary)", "3-season flooded Jan-Apr (Preliminary)"):
        r = t2[(t2["period"] == pn) & (t2["subset"] == "all")].iloc[0]
        log(f"  {pn}: sigma raw {r['sigma']:.4f} ({r[f'sigma_boot_lo_{BD_H}d']:.4f}-{r[f'sigma_boot_hi_{BD_H}d']:.4f}), stratified {r['sigma_strat']:.4f} ({r[f'sigma_strat_lo_{BD_H}d']:.4f}-{r[f'sigma_strat_hi_{BD_H}d']:.4f}), "
            f"SR {r['SR']:.4f}, sigma/absSW {r['sigma_per_absSW_Kinv']:.5f}, n {int(r['n'])}")
    for _, r in t2b[t2b["headline"]].iterrows():
        log(f"  diff [{r['a']}] - [{r['b']}] ({int(r['block_days'])}-day blocks): d sigma {r['d_sigma']:+.4f} ({r['d_sigma_lo']:.4f} to {r['d_sigma_hi']:.4f}), frac<=0 {r['d_sigma_frac_le0']}; d SR {r['d_SR']:+.4f} ({r['d_SR_lo']:.4f} to {r['d_SR_hi']:.4f})")
    c3 = t3[(t3["period"] == "2015-2016") & (t3["subset"] == "all")].iloc[0]
    log(f"  closure: CR_hh {c3['CR_sum_hh']:.3f}, CR_daily {c3['CR_sum_daily']:.3f}, slope {c3['ols_slope']:.3f}, intercept {c3['ols_intercept_Wm2']:.1f}; sigma S0 {c3['sigma_S0']:.4f} S1 {c3['sigma_S1']:.4f} S2 {c3['sigma_S2']:.4f} S3 {c3['sigma_S3']:.4f}")
    ns = ~t4["label"].str.startswith("sample")
    log(f"  sensitivity range of sigma across T4 (excluding samples): {t4.loc[ns, 'sigma'].min():.4f} ({t4.loc[t4.loc[ns, 'sigma'].idxmin(), 'label']}) .. {t4.loc[ns, 'sigma'].max():.4f} ({t4.loc[t4.loc[ns, 'sigma'].idxmax(), 'label']})")
    for _, r in t4.iterrows():
        log(f"    T4 {r['label']}: sigma {r['sigma']:.4f} (d {r['d_sigma_vs_core']:+.4f}) | under S1 {r['sigma_S1']:.4f} (d {r['d_sigma_S1_vs_core_S1']:+.4f}) | SR {r['SR']:.4f}, wet-dry {r['d_wet_minus_dry']:.4f}, sigma_rad {r['sigma_rad']:.4f}")
    log("done")
    (C.RESULTS_DIR / "run_log.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
