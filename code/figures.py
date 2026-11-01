"""Figures F1-F7 (METHODS_SPEC.md, section 6.2). matplotlib only; PDF + PNG 300 dpi.

Colours: Okabe-Ito (colour-blind safe), wet/dry fixed in config.COLORS. No dual axes.
matplotlib lives in the user site-packages on this machine, which `python3 -I` hides;
the fallback below adds that path explicitly (documented in the run log).
"""
from __future__ import annotations

import sys

try:
    import matplotlib
except ModuleNotFoundError:  # isolated mode (-I) hides the user site-packages
    import site
    sys.path.append(site.getusersitepackages())
    import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                 # noqa: E402
import pandas as pd                # noqa: E402

import config as C                 # noqa: E402

MPL_VERSION = matplotlib.__version__
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#dddddd", "grid.linewidth": 0.5, "lines.linewidth": 1.4,
    "figure.dpi": 100, "savefig.dpi": 300,
})
SIG_LABEL = r"$\sigma$ (W m$^{-2}$ K$^{-1}$)"
J_LABEL = r"Entropy flux (W m$^{-2}$ K$^{-1}$)"
E_LABEL = r"Energy flux (W m$^{-2}$)"
MONTHS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(C.FIGURES_DIR / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)


def shade_wet(ax):
    for m in C.WET_MONTHS:
        ax.axvspan(m - 0.5, m + 0.5, color=C.COLORS["wet"], alpha=0.07, lw=0)


# ----------------------------------------------------------------------------
def f1_monthly(t5: pd.DataFrame, t5y: pd.DataFrame):
    fig, axs = plt.subplots(4, 1, figsize=(6.5, 9), sharex=True)
    m = t5["month"].to_numpy()
    ax = axs[0]
    ax.plot(m, t5["J_LWout"], color=C.COLORS["lw_out"], marker="o", ms=4, label=r"$J_{LW,out}$")
    ax.plot(m, t5["J_LWin"], color=C.COLORS["lw_in"], marker="o", ms=4, ls="--", label=r"$J_{LW,in}$ (input)")
    ax.set_ylabel(J_LABEL); ax.legend(ncol=2, frameon=False, loc="lower left"); ax.set_title("Monthly means, core sample 2015-2016 (pooled)")
    ax = axs[1]
    ax.plot(m, t5["J_LE"], color=C.COLORS["le"], marker="o", ms=4, label=r"$J_{LE}$")
    ax.plot(m, t5["J_H"], color=C.COLORS["h"], marker="s", ms=4, label=r"$J_{H}$")
    ax.plot(m, t5["J_SWin"], color=C.COLORS["sw_in"], marker="^", ms=4, ls="--", label=r"$J_{SW,in}$ (input)")
    ax.plot(m, t5["J_SWout"], color=C.COLORS["sw_out"], marker="v", ms=4, label=r"$J_{SW,out}$")
    ax.axhline(0, color="k", lw=0.6); ax.set_ylabel(J_LABEL); ax.legend(ncol=4, frameon=False)
    ax = axs[2]
    for y in sorted(t5y["year"].unique()):
        s = t5y[t5y["year"] == y]
        ax.plot(s["month"], s["sigma"], ls="none", marker="o", mfc="none", ms=5, color=C.COLORS[str(y)], label=str(y))
    ax.plot(m, t5["sigma"], color=C.COLORS["sigma"], marker="o", ms=4, label="pooled")
    for mi, n in zip(m, t5["n"]):
        ax.annotate(f"n={int(n)}", (mi, 1.0), xycoords=("data", "axes fraction"), xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6, color="#555555", annotation_clip=False)
    ax.set_ylabel(SIG_LABEL); ax.legend(ncol=3, frameon=False, loc="lower right")
    ax = axs[3]
    for y in sorted(t5y["year"].unique()):
        s = t5y[t5y["year"] == y]
        ax.plot(s["month"], s["SR"], ls="none", marker="o", mfc="none", ms=5, color=C.COLORS[str(y)], label=str(y))
    ax.plot(m, t5["SR"], color=C.COLORS["sigma"], marker="o", ms=4, label="pooled")
    ax.set_ylabel(r"SR = $\langle J_{out}\rangle / \langle J_{in}\rangle$"); ax.set_xlabel("Month (shaded = wet season Nov-Apr)")
    ax.set_xticks(range(1, 13)); ax.set_xticklabels(MONTHS)
    for a in axs:
        shade_wet(a)
    save(fig, "f1_monthly_climatology")


# ----------------------------------------------------------------------------
def f2_diurnal(comp: dict):
    """comp[season][name] -> DataFrame(hour, mean, lo, hi). Names: sigma, sigma_rad, sigma_turb,
    sigma_soil_lo/hi (storage scenario ii bounds), Rn, H, LE, Rres, dQ_soil_min/max, dQ_water_min/max."""
    fig, axs = plt.subplots(2, 2, figsize=(9, 6.5), sharex=True)
    for j, season in enumerate(("wet", "dry")):
        c = comp[season]
        ax = axs[0, j]
        s = c["sigma"]
        ax.fill_between(s["hour"], s["lo"], s["hi"], color=C.COLORS[season], alpha=0.25, lw=0, label="bootstrap 95 %")
        ax.plot(s["hour"], s["mean"], color=C.COLORS[season], label=r"$J_{out}-J_{in}$ (exchange)")
        ax.plot(c["sigma_rad"]["hour"], c["sigma_rad"]["mean"], color=C.COLORS["lw_out"], ls="--", label=r"$J_{LW,out}+J_{SW,out}-J_{in}$")
        ax.plot(c["sigma_turb"]["hour"], c["sigma_turb"]["mean"], color=C.COLORS["h"], ls=":", label=r"$J_H+J_{LE}$")
        if "sigma_soil_lo" in c:
            ax.fill_between(c["sigma_soil_lo"]["hour"], c["sigma_soil_lo"]["mean"], c["sigma_soil_hi"]["mean"],
                            color="#999999", alpha=0.35, lw=0, label=r"+ $\Delta Q_{soil}/T$ (scenario ii)")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"{season} season")
        if j == 0:
            ax.set_ylabel(J_LABEL)
        ax = axs[1, j]
        for name, col, lab in (("Rn", C.COLORS["sw_in"], r"$R_n$"), ("H", C.COLORS["h"], "H"), ("LE", C.COLORS["le"], "LE"), ("Rres", "#000000", r"$R' = R_n - H - LE$")):
            ax.plot(c[name]["hour"], c[name]["mean"], color=col, label=lab)
        if "dQ_soil_min" in c:
            ax.fill_between(c["dQ_soil_min"]["hour"], c["dQ_soil_min"]["mean"], c["dQ_soil_max"]["mean"], color="#999999", alpha=0.4, lw=0, label=r"$\Delta Q_{soil}$ scenario (ii)")
        if "dQ_water_min" in c and np.isfinite(c["dQ_water_min"]["mean"]).any():
            ax.fill_between(c["dQ_water_min"]["hour"], c["dQ_water_min"]["mean"], c["dQ_water_max"]["mean"], color=C.COLORS["water"], alpha=0.3, lw=0, hatch="//", label=r"$\Delta Q_{water}$ 0.2-1.0 m (iii, Jan-Apr; TS$_1$ proxy)")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("Local standard time (h)")
        if j == 0:
            ax.set_ylabel(E_LABEL)
    axs[0, 0].legend(frameon=False, fontsize=6.5, loc="upper left")
    # shared legend of the bottom row: handles from both panels (the water band exists only in the wet panel)
    handles, labels = [], []
    for a in (axs[1, 0], axs[1, 1]):
        for h_, l_ in zip(*a.get_legend_handles_labels()):
            if l_ not in labels:
                handles.append(h_); labels.append(l_)
    axs[1, 0].legend(handles, labels, frameon=False, fontsize=6.5, loc="upper left")
    axs[0, 0].set_xlim(0, 24); axs[0, 0].set_xticks(range(0, 25, 4))
    fig.suptitle("Diurnal composites, core sample 2015-2016. Half-hourly $J_{out}-J_{in}$ is an exchange, not the production (storage inside the volume).", fontsize=8)
    save(fig, "f2_diurnal_composite")


# ----------------------------------------------------------------------------
def _ols_line(ax, x, y, color, label):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return
    a, b = np.polyfit(x[ok], y[ok], 1)
    r2 = 1 - ((y[ok] - (a * x[ok] + b)) ** 2).sum() / ((y[ok] - y[ok].mean()) ** 2).sum()
    xx = np.linspace(np.nanmin(x[ok]), np.nanmax(x[ok]), 20)
    ax.plot(xx, a * xx + b, color=color, lw=1, label=f"{label}: y = {a:.4f} x + {b:.3f}, R² = {r2:.2f}")


def f3_sigma_vs_rn(daily: pd.DataFrame):
    """daily: columns season, Rn, LE_s, sigma (daily means over complete days)."""
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    for season in ("wet", "dry"):
        d = daily[daily["season"] == season]
        for ax, xv in zip(axs, ("Rn", "LE_s")):
            ax.plot(d[xv], d["sigma"], ls="none", marker="o", ms=4, alpha=0.6, color=C.COLORS[season], mec="white", mew=0.3)
            _ols_line(ax, d[xv].to_numpy(), d["sigma"].to_numpy(), C.COLORS[season], season)
    axs[0].set_xlabel(r"Daily mean $R_n$ (W m$^{-2}$)"); axs[1].set_xlabel(r"Daily mean LE (W m$^{-2}$)")
    for ax in axs:
        ax.set_ylabel(SIG_LABEL); ax.legend(frameon=False, fontsize=7)
    fig.suptitle(f"Daily entropy production vs net radiation and latent heat, complete days (n = {len(daily)}), 2015-2016", fontsize=9)
    save(fig, "f3_sigma_vs_rn")


# ----------------------------------------------------------------------------
def f4_closure(d: pd.DataFrame, daily: pd.DataFrame, comp_res: dict):
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 4), layout="constrained")
    ax = axs[0]
    x = d["Rn"].to_numpy(); y = (d["H"] + d["LE"]).to_numpy()
    hb = ax.hexbin(x, y, gridsize=60, cmap="Blues", mincnt=1, bins="log", linewidths=0.1)
    lim = [np.nanmin(x) - 20, np.nanmax(x) + 20]
    ax.plot(lim, lim, color="k", lw=0.8, ls="--", label="1:1")
    _ols_line(ax, x, y, C.COLORS["dry"], "OLS half-hourly")
    ax.set_xlabel(r"$R_n$ (W m$^{-2}$)"); ax.set_ylabel(r"H + LE (W m$^{-2}$)"); ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.set_title(f"Half-hourly, n = {len(d)}"); fig.colorbar(hb, ax=ax, label="count (log)", pad=0.01)
    ax = axs[1]
    for season in ("wet", "dry"):
        dd = daily[daily["season"] == season]
        ax.plot(dd["Rn"], dd["H"] + dd["LE"], ls="none", marker="o", ms=4, alpha=0.7, color=C.COLORS[season], mec="white", mew=0.3, label=season)
    x = daily["Rn"].to_numpy(); y = (daily["H"] + daily["LE"]).to_numpy()
    lim = [np.nanmin(x) - 10, np.nanmax(x) + 10]
    ax.plot(lim, lim, color="k", lw=0.8, ls="--", label="1:1")
    _ols_line(ax, x, y, "k", "OLS daily")
    ax.set_xlabel(r"Daily mean $R_n$ (W m$^{-2}$)"); ax.set_ylabel(r"Daily mean H + LE (W m$^{-2}$)"); ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.set_title(f"Daily, complete days, n = {len(daily)}")
    ax = axs[2]
    for season in ("wet", "dry"):
        c = comp_res[season]
        ax.fill_between(c["hour"], c["lo"], c["hi"], color=C.COLORS[season], alpha=0.25, lw=0)
        ax.plot(c["hour"], c["mean"], color=C.COLORS[season], label=season)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("Local standard time (h)"); ax.set_ylabel(r"$R' = R_n - H - LE$ (W m$^{-2}$)"); ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 4))
    ax.legend(frameon=False, fontsize=7); ax.set_title("Diurnal composite of the residual (no G)")
    save(fig, "f4_closure")


# ----------------------------------------------------------------------------
def f5_diagnostics(dT: dict, dTsky: dict, albedo_month: pd.DataFrame):
    """dT[season][eps] -> composite of Ts - Ta; dTsky[season] -> composite of Tsky - Ta;
    albedo_month: columns month, year|'pooled', albedo_midday (ratio of sums), n."""
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
    ax = axs[0]
    ls_cycle = [":", "-.", "-", "--"]
    styles = {e: (ls_cycle[i % len(ls_cycle)] if e != C.EPS_CORE else "-") for i, e in enumerate(C.EPS_FIGURE)}
    for season in ("wet", "dry"):
        for eps, ls in styles.items():
            if eps not in dT[season]:
                continue
            c = dT[season][eps]
            ax.plot(c["hour"], c["mean"], color=C.COLORS[season], ls=ls, label=f"{season}, ε={eps}")
    ax.axhline(0, color="k", lw=0.6); ax.set_ylabel(r"$T_s - T_a$ (K)"); ax.set_xlabel("Local standard time (h)")
    ax.legend(frameon=False, fontsize=6, ncol=2); ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 4))
    ax.set_title("Radiometric surface minus air temperature")
    ax = axs[1]
    for season in ("wet", "dry"):
        c = dTsky[season]
        ax.fill_between(c["hour"], c["lo"], c["hi"], color=C.COLORS[season], alpha=0.25, lw=0)
        ax.plot(c["hour"], c["mean"], color=C.COLORS[season], label=season)
    ax.set_ylabel(r"$T_{sky} - T_a$ (K)"); ax.set_xlabel("Local standard time (h)"); ax.legend(frameon=False, fontsize=7)
    ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 4)); ax.set_title("Effective sky minus air temperature")
    ax = axs[2]
    pooled = albedo_month[albedo_month["year"] == "pooled"]
    for y in sorted(set(albedo_month["year"]) - {"pooled"}):
        s = albedo_month[albedo_month["year"] == y]
        ax.plot(s["month"], s["albedo_midday"], ls="none", marker="o", mfc="none", ms=5, color=C.COLORS[str(y)], label=str(y))
    ax.plot(pooled["month"], pooled["albedo_midday"], color="k", marker="o", ms=4, label="pooled")
    shade_wet(ax); ax.set_xticks(range(1, 13)); ax.set_xticklabels(MONTHS)
    ax.set_ylabel("Midday albedo (11-13 h, ratio of sums)"); ax.set_xlabel("Month"); ax.legend(frameon=False, fontsize=7)
    ax.set_title("Midday albedo")
    save(fig, "f5_ts_minus_ta")


# ----------------------------------------------------------------------------
def f6_tornado(t4: pd.DataFrame):
    t = t4[t4["label"] != "core"].copy()
    t = t[np.isfinite(t["d_sigma_vs_core"])]
    t = t.sort_values("d_sigma_vs_core", key=np.abs)
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(t) + 1.2))
    colors = np.where(t["d_sigma_vs_core"] >= 0, C.COLORS["lw_out"], C.COLORS["dry"])
    ax.barh(t["label"], t["d_sigma_vs_core"], color=colors, height=0.6)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel(r"$\Delta\langle\sigma\rangle$ relative to core configuration (W m$^{-2}$ K$^{-1}$), 2015-2016")
    ax.grid(axis="y", visible=False)
    save(fig, "f6_sensitivity_tornado")


# ----------------------------------------------------------------------------
def f7_h_vs_dt(d: pd.DataFrame, hours=C.NOON_DIAG_HOURS):
    """Diagnostic: half-hourly H against radiometric T_s - T_a in the midday window, by season
    (d: core rows with columns season, hour, H, Ts, Ta). The relevant test of the T_s < T_a flag."""
    h0, h1 = hours
    dd = d[(d["hour"] >= h0) & (d["hour"] < h1)]
    fig, axs = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, season in zip(axs, ("wet", "dry")):
        s = dd[dd["season"] == season]
        x = (s["Ts"] - s["Ta"]).to_numpy(); y = s["H"].to_numpy()
        ax.plot(x, y, ls="none", marker="o", ms=2.5, alpha=0.35, color=C.COLORS[season], mec="none")
        _ols_line(ax, x, y, "k", f"OLS {season}")
        ax.axvline(0, color="k", lw=0.6); ax.axhline(0, color="k", lw=0.6)
        ok = np.isfinite(x) & np.isfinite(y)
        r = np.corrcoef(x[ok], y[ok])[0, 1] if ok.sum() > 2 else np.nan
        ax.set_title(f"{season}: {h0:02d}:00-{h1:02d}:00 LST, n = {ok.sum()}, r = {r:.2f}, "
                     f"frac $T_s<T_a$ = {(x[ok] < 0).mean():.2f}", fontsize=8)
        ax.set_xlabel(r"$T_s - T_a$ (K), $\varepsilon_s$ = " + f"{C.EPS_CORE}")
        ax.legend(frameon=False, fontsize=7, loc="upper left")
    axs[0].set_ylabel(r"H (W m$^{-2}$)")
    fig.suptitle("Diagnostic: sensible heat flux vs radiometric surface-air temperature difference (core 2015-2016)", fontsize=9)
    save(fig, "f7_h_vs_ts_minus_ta")
