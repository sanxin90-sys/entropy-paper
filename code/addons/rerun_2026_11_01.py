"""Additional variants requested for manuscript v10 (rerun of 2026-11-01).

Run from code/:  python3 -I addons/rerun_2026_11_01.py

This script does NOT modify the pipeline (code/*.py, whose code hash a5a3ec682769 identifies
every existing result). It imports the pipeline modules unchanged, rebuilds the core sample and
the core budget exactly as run_all.py does, and adds:

  1. F/T convention for all four radiation terms (computed directly from the fluxes, not by
     scaling), under S0 and S1; check of the analytic shift -sigma_rad/4.
  2. Closure scenario S3 with a per-period closure ratio (each period divided by its own
     half-hourly CR_sum; day and night rows use the CR of their period).
  3. Bootstrap intervals (7-day moving blocks, 1000 replicates, seed 20261030) of the
     complete-day mean estimator and SR_daily, and a recomputation of the stratified intervals,
     for 2015-2016, wet pooled and dry pooled; stratified wet - dry difference.
  4. Exact counts of the records removed by SW_IN_F_QC == 0 (several definitions).
  5. Night and midnight SR for every S0 row of Table S4 (incl. J_LWin at T_a).
  6. J_LWin with T_sky from an atmospheric emissivity of 0.85: T_sky = (LW_IN/(0.85 sigma_SB))^(1/4).
  7. RMA slope of the half-hourly closure and median |R'| (comparison with Nicolini et al. 2026).
  8. Fig. S1 tornado regenerated with the Table S4 row names (file f6_sensitivity_tornado.*).

Outputs (results/tables): t4_sensitivity.csv (3 rows appended, existing rows byte-identical),
t4b_sr_night_midnight.csv, t3b_closure_S3_per_period_cr.csv, t2f_bootstrap_estimators.csv,
t0d_swin_qc_counts.csv, t3c_closure_rma_and_abs_residual.csv, t4c_ft_convention_check.csv;
results/figures/f6_sensitivity_tornado.{pdf,png}; results/run_log_rerun_2026-11-01.txt.
"""
from __future__ import annotations

import hashlib
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))   # code/
sys.path.insert(1, str(HERE))          # addons/ (figures_v10)

import numpy as np
import pandas as pd

import config as C
import load as L
import entropy as E
import closure as CL
import stats as S

LOG = []
T0_ = time.time()
BD = C.BOOT_BLOCK_DAYS_HEADLINE
EPS_ATM = 0.85


def log(msg=""):
    line = f"[{time.time() - T0_:7.1f}s] {msg}" if msg else ""
    print(line, flush=True)
    LOG.append(line)


# ----------------------------------------------------------------------------
# Variant budgets (built from the unchanged core budget, radiation terms recomputed)
# ----------------------------------------------------------------------------
def _recombine(J):
    J = J.copy()
    J["J_in"] = J["J_SWin"] + J["J_LWin"]
    J["J_out"] = J["J_SWout"] + J["J_LWout"] + J["J_H"] + J["J_LE"]
    J["sigma"] = J["J_out"] - J["J_in"]
    J["sigma_rad"] = J["J_SWout"] + J["J_LWout"] - J["J_SWin"] - J["J_LWin"]
    J["sigma_turb"] = J["J_H"] + J["J_LE"]
    return J


def budget_ft(df, closure="S0"):
    """F/T convention: every radiation term = F/T (no 4/3), same temperatures as the core
    (T_sun, T_sky, T_s); turbulent terms unchanged. Computed from the fluxes directly."""
    J = E.budget(df, C.Options(closure=closure))
    sw_in = df["SW_IN_F"].to_numpy(float); sw_out = df["SW_OUT"].to_numpy(float)
    lw_in = df["LW_IN_F"].to_numpy(float); lw_out = df["LW_OUT"].to_numpy(float)
    is_day = (df["NIGHT"].to_numpy() == 0)
    valid = is_day & (sw_in > 0) & (sw_out > 0)
    ts = E.t_surface(lw_out, lw_in, C.EPS_CORE)
    J["J_SWin"] = sw_in / C.T_SUN
    J["J_SWout"] = np.where(valid, sw_out / C.T_SUN, 0.0)
    J["J_LWin"] = lw_in / E.t_sky(lw_in)
    J["J_LWout"] = lw_out / ts
    return _recombine(J)


def budget_eps_atm(df, closure="S0", eps_atm=EPS_ATM):
    """J_LWin = (4/3) LW_IN / T_sky,atm with T_sky,atm = (LW_IN/(eps_atm sigma_SB))^(1/4)
    (inverted Stefan-Boltzmann law with an atmospheric emissivity, as in Brunsell et al. 2011,
    Stoy et al. 2014, Wiesner et al. 2019); every other term as in the core."""
    J = E.budget(df, C.Options(closure=closure))
    lw_in = df["LW_IN_F"].to_numpy(float)
    t_sky_atm = (lw_in / (eps_atm * C.SIGMA_SB)) ** 0.25
    J["J_LWin"] = (4.0 / 3.0) * lw_in / t_sky_atm
    return _recombine(J)


def t4_row(label, Ja, m, se, J1=None, Jw=None, Jd=None):
    """Row in the t4 layout. Jw/Jd: budgets to use for the wet/dry means (per-period S3)."""
    Jw = Ja if Jw is None else Jw
    Jd = Ja if Jd is None else Jd
    a = S.period_stats(Ja, m); w = S.period_stats(Jw, m & (se == "wet")); dr = S.period_stats(Jd, m & (se == "dry"))
    row = {"label": label, "n": a["n"], "sigma": a["sigma"], "SR": a["SR"], "sigma_rad": a["sigma_rad"], "sigma_turb": a["sigma_turb"],
           "J_in": a["J_in"], "J_out": a["J_out"], "sigma_per_absSW_Kinv": a["sigma_per_absSW_Kinv"],
           "sigma_wet": w["sigma"], "sigma_dry": dr["sigma"], "d_wet_minus_dry": w["sigma"] - dr["sigma"],
           "SR_wet": w["SR"], "SR_dry": dr["SR"]}
    if J1 is not None:
        a1 = S.period_stats(J1, m)
        row["sigma_S1"] = a1["sigma"]; row["SR_S1"] = a1["SR"]
    else:
        row["sigma_S1"] = np.nan; row["SR_S1"] = np.nan
    return row


def boot_daily_mean(J, dates, mask, n_boot=C.BOOT_N, seed=C.BOOT_SEED, block_days=BD):
    """Moving-block bootstrap (same block construction and seed as stats.bootstrap_means) of the
    complete-day estimators: mean of daily-mean sigma over complete days, and SR_daily (ratio of
    means of daily means, Lin 2015). Unit = complete day, blocks of consecutive complete days."""
    daily = S.daily_means(J, pd.Series(dates, index=J.index), mask)
    sig = daily["sigma"].to_numpy(float); jo = daily["J_out"].to_numpy(float); ji = daily["J_in"].to_numpy(float)
    nd = len(sig)
    rng = np.random.default_rng(seed)
    idx = S._block_indices(nd, n_boot, block_days, rng)
    bs = sig[idx].mean(axis=1)
    bsr = jo[idx].mean(axis=1) / ji[idx].mean(axis=1)
    return nd, float(sig.mean()), S.ci(bs), float(jo.mean() / ji.mean()), S.ci(bsr)


def main():
    out_tables = C.TABLES_DIR
    log(f"rerun 2026-11-01 started {datetime.now().isoformat(timespec='seconds')} (local container time)")
    log(f"pipeline code hash {C.code_hash()} (code/*.py, unchanged) | config hash {C.config_hash()} | "
        f"addon sha256 {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]} | seed {C.BOOT_SEED} | n_boot {C.BOOT_N} | block {BD} d")
    sha = C.file_sha256(C.HH_FILE)
    log(f"data sha256 {sha} ({'matches' if sha == C.HH_FILE_SHA256 else 'DIFFERS FROM'} config.HH_FILE_SHA256)")
    df = L.add_derived(L.read_hh(), C.EPS_CORE)
    core = L.mask_core(df)
    assert int(core.sum()) == C.CORE_N_REGRESSION
    yr, mo, se, sy, s3 = df["year"], df["month"], df["season"], df["season_year"], df["season3"]
    dates = df["date"].to_numpy(); months = df["month"].to_numpy(); slots = df["slot"].to_numpy()
    hours = df["hour"]; night = ~df["is_day"]
    J = E.budget(df, C.CORE)
    J1 = E.budget(df, C.Options(closure="S1"))
    # sanity: the core values reproduce t4 row 0
    t4_old = pd.read_csv(out_tables / "t4_sensitivity.csv")
    assert abs(S.period_stats(J, core)["sigma"] - t4_old.loc[0, "sigma"]) < 1e-5
    periods = {
        "2015-2016": core, "2015": core & (yr == 2015), "2016": core & (yr == 2016),
        "wet pooled (Nov-Apr)": core & (se == "wet"), "dry pooled (May-Oct)": core & (se == "dry"),
        "wet 2015 (Mar-Apr 2015, partial)": core & (se == "wet") & (sy == 2015),
        "dry 2015 (May-Oct 2015)": core & (se == "dry") & (yr == 2015),
        "wet 2016 (Nov 2015-Apr 2016)": core & (se == "wet") & (sy == 2016),
        "dry 2016 (May-Oct 2016)": core & (se == "dry") & (yr == 2016),
        "wet 2017 (Nov 2016, partial)": core & (se == "wet") & (sy == 2017),
        "3-season dry May-Sep (Preliminary)": core & (s3 == "dry"),
        "3-season transition Oct-Dec (Preliminary)": core & (s3 == "transition"),
        "3-season flooded Jan-Apr (Preliminary)": core & (s3 == "flooded"),
    }
    subsets = {"all": pd.Series(True, index=df.index), "day": df["is_day"], "night": night}

    # ------------------------------------------------------------------ 4. SW_IN_F_QC counts
    y = df["year"].isin(C.CORE_YEARS)
    turb = y & L.mask_turb_measured(df)
    sw_bad = df["SW_IN_F_QC"] != 0
    lw_bad = df["LW_IN_F_QC"] != 0
    dawn = df["slot"].between(10, 13)          # 05:00-06:30 LST (slots starting 05:00, 05:30, 06:00, 06:30)
    dawn3 = df["slot"].between(10, 12)         # 05:00-06:00 starts (05:00-06:30 span)
    rows = []
    def cnt(name, m, note):
        rows.append({"definition": name, "n": int(m.sum()), "n_slots_05:00-06:30_start": int((m & dawn).sum()),
                     "n_slots_05:00-06:00_start": int((m & dawn3).sum()), "note": note})
    cnt("turbulent fluxes measured, 2015-2016", turb, "28,525 in the run log")
    cnt("... and fail the incoming-radiation condition (SW_IN_F_QC != 0 or LW_IN_F_QC != 0)", turb & (sw_bad | lw_bad), "2,863 = 28,525 - 25,662")
    cnt("... SW_IN_F_QC != 0 with LW_IN_F_QC == 0 (removed by the SW condition alone, beyond those lacking measured LW_IN)", turb & sw_bad & ~lw_bad, "definition of the SI v9 sentence (Sect. S2)")
    cnt("... LW_IN_F_QC != 0 (lacking measured LW_IN)", turb & lw_bad, "")
    cnt("... SW_IN_F_QC != 0 (any LW flag)", turb & sw_bad, "")
    cnt("... SW_IN_F_QC != 0 and LW_IN_F_QC != 0", turb & sw_bad & lw_bad, "")
    cnt("all 2015-2016 records with SW_IN_F_QC != 0 and LW_IN_F_QC == 0 (no turbulent condition)", y & sw_bad & ~lw_bad, "")
    cnt("all 2015-2016 records with SW_IN_F_QC != 0", y & sw_bad, "")
    t0d = pd.DataFrame(rows)
    t0d.to_csv(out_tables / "t0d_swin_qc_counts.csv", index=False)
    for _, r in t0d.iterrows():
        log(f"SW_IN_F_QC count [{r['definition']}]: n = {r['n']} (05:00-06:30 start slots {r['n_slots_05:00-06:30_start']}; 05:00-06:00 start slots {r['n_slots_05:00-06:00_start']})")
    m_def = turb & sw_bad & ~lw_bad
    hh = df.loc[m_def, "hour"].value_counts().sort_index()
    log(f"  SI-definition records by start hour (LST): {hh.to_dict()}")

    # ------------------------------------------------------------------ 1. F/T convention
    Jft0 = budget_ft(df, "S0"); Jft1 = budget_ft(df, "S1")
    chk = []
    for pn in ("2015-2016", "wet pooled (Nov-Apr)", "dry pooled (May-Oct)"):
        m = periods[pn]
        a = S.period_stats(J, m); f = S.period_stats(Jft0, m); a1 = S.period_stats(J1, m); f1 = S.period_stats(Jft1, m)
        chk.append({"period": pn, "sigma_core_S0": a["sigma"], "sigma_FT_S0": f["sigma"], "d_sigma_S0": f["sigma"] - a["sigma"],
                    "minus_sigma_rad_over_4": -a["sigma_rad"] / 4, "abs_diff_vs_analytic": abs(f["sigma"] - a["sigma"] + a["sigma_rad"] / 4),
                    "sigma_core_S1": a1["sigma"], "sigma_FT_S1": f1["sigma"], "d_sigma_S1": f1["sigma"] - a1["sigma"],
                    "SR_core": a["SR"], "SR_FT": f["SR"], "J_in_FT": f["J_in"], "J_out_FT": f["J_out"], "d_J_in": f["J_in"] - a["J_in"], "d_J_out": f["J_out"] - a["J_out"]})
    t4c = pd.DataFrame(chk)
    t4c.to_csv(out_tables / "t4c_ft_convention_check.csv", index=False, float_format="%.6g")
    for _, r in t4c.iterrows():
        log(f"F/T [{r['period']}]: sigma S0 {r['sigma_core_S0']:.6f} -> {r['sigma_FT_S0']:.6f} (d {r['d_sigma_S0']:+.6f}; -sigma_rad/4 = {r['minus_sigma_rad_over_4']:+.6f}; |diff| {r['abs_diff_vs_analytic']:.2e}); "
            f"S1 {r['sigma_core_S1']:.6f} -> {r['sigma_FT_S1']:.6f} (d {r['d_sigma_S1']:+.6f}); SR {r['SR_core']:.4f} -> {r['SR_FT']:.4f}; d J_in {r['d_J_in']:+.4f}, d J_out {r['d_J_out']:+.4f}")

    # ------------------------------------------------------------------ 6. eps_atm = 0.85
    Jea0 = budget_eps_atm(df, "S0"); Jea1 = budget_eps_atm(df, "S1")
    tsk = (df.loc[core, "LW_IN_F"] / (EPS_ATM * C.SIGMA_SB)) ** 0.25
    log(f"eps_atm = {EPS_ATM}: T_sky,atm / T_sky = {EPS_ATM ** -0.25:.5f}; mean T_sky,atm (core) {tsk.mean():.2f} K vs T_sky {J.loc[core, 'Tsky'].mean():.2f} K and T_a {J.loc[core, 'Ta'].mean():.2f} K")

    # ------------------------------------------------------------------ 2. S3 per-period CR
    crs = {}
    rows = []
    t3_old = pd.read_csv(out_tables / "t3_closure_scenarios.csv")
    for pn, pm in periods.items():
        d = df[pm]
        cr = CL.cr_sum(d["H"], d["LE"], d["Rn"])
        crs[pn] = cr
        J3 = E.budget(df, C.Options(closure="S3"), cr=cr)
        for sn, sm in subsets.items():
            m = pm & sm
            if m.sum() == 0:
                continue
            ps = S.period_stats(J3, m)
            old = t3_old[(t3_old["period"] == pn) & (t3_old["subset"] == sn)].iloc[0]
            rows.append({"period": pn, "subset": sn, "n_hh": int(m.sum()), "CR_used_per_period": cr,
                         "sigma_S3_per_period_CR": ps["sigma"], "SR_S3_per_period_CR": ps["SR"],
                         "sigma_S3_single_CR": old["sigma_S3"], "sigma_S1": old["sigma_S1"], "sigma_S0": old["sigma_S0"],
                         "d_S3_per_period_minus_S1": ps["sigma"] - old["sigma_S1"]})
    t3b = pd.DataFrame(rows)
    t3b.to_csv(out_tables / "t3b_closure_S3_per_period_cr.csv", index=False, float_format="%.6g")
    g = lambda pn, col="sigma_S3_per_period_CR": float(t3b[(t3b["period"] == pn) & (t3b["subset"] == "all")][col].iloc[0])
    for a_, b_ in (("wet pooled (Nov-Apr)", "dry pooled (May-Oct)"), ("wet 2016 (Nov 2015-Apr 2016)", "dry 2015 (May-Oct 2015)"),
                   ("wet 2016 (Nov 2015-Apr 2016)", "dry 2016 (May-Oct 2016)"), ("2016", "2015")):
        log(f"S3 per-period CR: [{a_}] - [{b_}] = {g(a_) - g(b_):+.4f} (single CR {g(a_, 'sigma_S3_single_CR') - g(b_, 'sigma_S3_single_CR'):+.4f}; S1 {g(a_, 'sigma_S1') - g(b_, 'sigma_S1'):+.4f})")
    for _, r in t3b[t3b["subset"] == "all"].iterrows():
        log(f"  S3 per-period [{r['period']}]: CR {r['CR_used_per_period']:.4f}, sigma {r['sigma_S3_per_period_CR']:.4f} (single CR {r['sigma_S3_single_CR']:.4f}; S1 {r['sigma_S1']:.4f}), SR {r['SR_S3_per_period_CR']:.4f}")
    Jw3 = E.budget(df, C.Options(closure="S3"), cr=crs["wet pooled (Nov-Apr)"])
    Jd3 = E.budget(df, C.Options(closure="S3"), cr=crs["dry pooled (May-Oct)"])
    Jc3 = E.budget(df, C.Options(closure="S3"), cr=crs["2015-2016"])

    # ------------------------------------------------------------------ t4: append rows (existing rows untouched)
    # deltas against the unrounded core values (as run_all does), not against the 6-digit CSV row
    _c0 = S.period_stats(J, core); _c1 = S.period_stats(J1, core)
    core_row = {"sigma": _c0["sigma"], "SR": _c0["SR"], "sigma_rad": _c0["sigma_rad"], "sigma_S1": _c1["sigma"]}
    new = [t4_row("radiation convention F/T for all four radiation terms (no 4/3)", Jft0, core, se, J1=Jft1),
           t4_row(f"J_LWin with T_sky from eps_atm = {EPS_ATM}", Jea0, core, se, J1=Jea1),
           t4_row("closure S3 (Bowen-preserving, per-period CR: wet and dry pooled each with own CR)", Jc3, core, se, Jw=Jw3, Jd=Jd3)]
    t4n = pd.DataFrame(new)
    t4n["d_sigma_vs_core"] = t4n["sigma"] - core_row["sigma"]
    t4n["d_sigma_S1_vs_core_S1"] = t4n["sigma_S1"] - core_row["sigma_S1"]
    t4n["d_SR_vs_core"] = t4n["SR"] - core_row["SR"]
    t4n["d_sigma_rad_vs_core"] = t4n["sigma_rad"] - core_row["sigma_rad"]
    t4n = t4n[list(t4_old.columns)]
    old_text = (out_tables / "t4_sensitivity.csv").read_text(encoding="utf-8")
    if "radiation convention F/T" in old_text:
        raise SystemExit("t4_sensitivity.csv already holds the appended rows; restore the original before rerunning")
    add = t4n.to_csv(index=False, header=False, float_format="%.6g")
    (out_tables / "t4_sensitivity.csv").write_text(old_text + add, encoding="utf-8")
    for _, r in t4n.iterrows():
        log(f"T4 appended [{r['label']}]: sigma {r['sigma']:.4f} (d {r['d_sigma_vs_core']:+.4f}) | S1 {r['sigma_S1']:.4f} (d {r['d_sigma_S1_vs_core_S1']:+.4f}) | SR {r['SR']:.4f} (d {r['d_SR_vs_core']:+.4f}) | "
            f"wet {r['sigma_wet']:.4f} dry {r['sigma_dry']:.4f} wet-dry {r['d_wet_minus_dry']:.4f} | sigma_rad {r['sigma_rad']:.4f}")

    # ------------------------------------------------------------------ 5. night / midnight SR for S0 variants
    variants = [("core", C.CORE), ("eps_s = 0.95", C.Options(eps=0.95)), ("eps_s = 0.97", C.Options(eps=0.97)), ("eps_s = 0.99", C.Options(eps=0.99)),
                ("J_LWin with T_a", C.Options(lwin_temp="air")), ("J_LWin diluted X(delta_a), T_a", C.Options(lwin_temp="diluted")),
                ("J_H, J_LE with T_a", C.Options(turb_temp="ta")), ("J_SWout variant (b) diluted", C.Options(swout_variant="b")),
                ("J_SWin hemispheric-diluted upper bound", C.Options(swin_variant="diluted")), ("J_LWout greybody eq. 11", C.Options(lwout_variant="grey")),
                ("J_LWout greybody + diluted reflection (upper bound)", C.Options(lwout_variant="grey_refl")), ("T_sun = 5778 K", C.Options(t_sun=C.T_SUN_ALT))]
    rows = []
    def sr_rows(lab, Jv):
        a = S.period_stats(Jv, core & night)
        mn, n_mn = S.sr_window(Jv, hours, core, C.MIDNIGHT_HOURS)
        md, n_md = S.sr_window(Jv, hours, core, C.MIDDAY_HOURS)
        daily = S.daily_means(Jv, df["date"], core)
        rows.append({"label": lab, "SR_night": a["SR"], "n_night": a["n"], "SR_midnight_23_01": mn, "n_midnight": n_mn,
                     "SR_midday_11_13": md, "n_midday": n_md, "SR_daily_Lin": S.sr_daily(daily), "sigma_night": a["sigma"]})
    for lab, opts in variants:
        sr_rows(lab, E.budget(df, opts))
    sr_rows("radiation convention F/T for all four radiation terms (no 4/3)", Jft0)
    sr_rows(f"J_LWin with T_sky from eps_atm = {EPS_ATM}", Jea0)
    t4b = pd.DataFrame(rows)
    t4b.to_csv(out_tables / "t4b_sr_night_midnight.csv", index=False, float_format="%.6g")
    for _, r in t4b.iterrows():
        log(f"SR night/midnight/midday/daily [{r['label']}]: {r['SR_night']:.4f} / {r['SR_midnight_23_01']:.4f} / {r['SR_midday_11_13']:.4f} / {r['SR_daily_Lin']:.4f} (n night {r['n_night']}, midnight {r['n_midnight']})")

    # ------------------------------------------------------------------ 3. bootstrap of the estimators
    t2_old = pd.read_csv(out_tables / "t2_entropy_budget.csv")
    rows = []
    for pn in ("2015-2016", "wet pooled (Nov-Apr)", "dry pooled (May-Oct)"):
        m = periods[pn]
        o = t2_old[(t2_old["period"] == pn) & (t2_old["subset"] == "all")].iloc[0]
        st = S.stratified_mean(J, months, slots, m)
        bs, names = S.bootstrap_stratified(J, dates, months, slots, m, block_days=BD)
        slo, shi = S.ci(bs[:, names.index("sigma")])
        nd, dm, (dlo, dhi), srd, (srlo, srhi) = boot_daily_mean(J, dates, m)
        rows.append({"period": pn, "sigma_raw": o["sigma"], "sigma_raw_lo_7d": o["sigma_boot_lo_7d"], "sigma_raw_hi_7d": o["sigma_boot_hi_7d"],
                     "sigma_strat": st["sigma_strat"], "sigma_strat_lo_7d": slo, "sigma_strat_hi_7d": shi,
                     "strat_matches_t2": bool(float(f"{slo:.6g}") == o["sigma_strat_lo_7d"] and float(f"{shi:.6g}") == o["sigma_strat_hi_7d"]),   # t2 CSV carries 6 significant digits
                     "n_complete_days": nd, "sigma_daily_mean": dm, "sigma_daily_mean_lo_7d": dlo, "sigma_daily_mean_hi_7d": dhi,
                     "SR_daily_Lin": srd, "SR_daily_lo_7d": srlo, "SR_daily_hi_7d": srhi})
    # stratified wet - dry (strata resampled independently: seed and seed + 1, as stats.bootstrap_difference)
    bw, nm = S.bootstrap_stratified(J, dates, months, slots, periods["wet pooled (Nov-Apr)"], block_days=BD, seed=C.BOOT_SEED)
    bd_, _ = S.bootstrap_stratified(J, dates, months, slots, periods["dry pooled (May-Oct)"], block_days=BD, seed=C.BOOT_SEED + 1)
    dif = bw[:, nm.index("sigma")] - bd_[:, nm.index("sigma")]
    dlo, dhi = S.ci(dif)
    st_w = S.stratified_mean(J, months, slots, periods["wet pooled (Nov-Apr)"])["sigma_strat"]
    st_d = S.stratified_mean(J, months, slots, periods["dry pooled (May-Oct)"])["sigma_strat"]
    rows.append({"period": "stratified wet pooled - dry pooled", "sigma_strat": st_w - st_d, "sigma_strat_lo_7d": dlo, "sigma_strat_hi_7d": dhi,
                 "strat_matches_t2": np.nan})
    t2f = pd.DataFrame(rows)
    t2f.to_csv(out_tables / "t2f_bootstrap_estimators.csv", index=False, float_format="%.6g")
    for _, r in t2f.iterrows():
        if r["period"].startswith("stratified wet"):
            log(f"stratified wet - dry: {r['sigma_strat']:+.4f} ({r['sigma_strat_lo_7d']:.4f} to {r['sigma_strat_hi_7d']:.4f}; frac <= 0: {int((dif <= 0).sum())} of {len(dif)})")
        else:
            log(f"estimators [{r['period']}]: raw {r['sigma_raw']:.4f} ({r['sigma_raw_lo_7d']:.4f}-{r['sigma_raw_hi_7d']:.4f}); stratified {r['sigma_strat']:.4f} ({r['sigma_strat_lo_7d']:.4f}-{r['sigma_strat_hi_7d']:.4f}; identical to t2: {r['strat_matches_t2']}); "
                f"complete-day mean {r['sigma_daily_mean']:.4f} ({r['sigma_daily_mean_lo_7d']:.4f}-{r['sigma_daily_mean_hi_7d']:.4f}; {int(r['n_complete_days'])} days); SR_daily {r['SR_daily_Lin']:.4f} ({r['SR_daily_lo_7d']:.4f}-{r['SR_daily_hi_7d']:.4f})")

    # ------------------------------------------------------------------ 7. RMA slope and |R'|
    rows = []
    for pn in ("2015-2016", "wet pooled (Nov-Apr)", "dry pooled (May-Oct)", "2015", "2016"):
        d = df[periods[pn]]
        x = d["Rn"].to_numpy(float); yv = (d["H"] + d["LE"]).to_numpy(float)
        r = np.corrcoef(x, yv)[0, 1]
        rma = np.sign(r) * yv.std(ddof=1) / x.std(ddof=1)
        ols = CL.ols(x, yv)
        absr = d["Rres"].abs()
        ann_med = d.assign(absR=absr).groupby("year")["absR"].median()
        rows.append({"period": pn, "n_hh": len(d), "ols_slope": ols["slope"], "r": r, "rma_slope": rma,
                     "rma_intercept_Wm2": yv.mean() - rma * x.mean(), "Rres_mean_Wm2": d["Rres"].mean(),
                     "abs_Rres_median_Wm2": absr.median(), "abs_Rres_median_by_year_mean_Wm2": ann_med.mean(),
                     "abs_Rres_median_by_year": "; ".join(f"{k}: {v:.1f}" for k, v in ann_med.items())})
    t3c = pd.DataFrame(rows)
    t3c.to_csv(out_tables / "t3c_closure_rma_and_abs_residual.csv", index=False, float_format="%.6g")
    for _, r in t3c.iterrows():
        log(f"closure [{r['period']}]: OLS slope {r['ols_slope']:.4f}, r {r['r']:.4f}, RMA slope {r['rma_slope']:.4f} (intercept {r['rma_intercept_Wm2']:.1f}); mean R' {r['Rres_mean_Wm2']:.1f}; median |R'| {r['abs_Rres_median_Wm2']:.1f} (annual medians {r['abs_Rres_median_by_year']})")

    # ------------------------------------------------------------------ 8. tornado with Table S4 row names
    import figures_v10 as FV
    t4_all = pd.read_csv(out_tables / "t4_sensitivity.csv")
    FV.f6_tornado_s4(t4_all)
    log("Fig. S1 regenerated: results/figures/f6_sensitivity_tornado.{pdf,png} with Table S4 row names (addons/figures_v10.py)")
    log("done")
    (C.RESULTS_DIR / "run_log_rerun_2026-11-01.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
