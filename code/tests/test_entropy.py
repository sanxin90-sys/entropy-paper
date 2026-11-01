"""Unit tests for the entropy budget. Run with `python3 -I -m pytest tests/` from code/
or, without pytest, `python3 -I tests/test_entropy.py`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

import config as C
import entropy as E
import stats as S
import load as L
import closure as CL
import storage as ST


def test_blackbody_identity():
    """Spectral integration of the Planck entropy equals (4/3) sigma T^3."""
    for t in (250.0, 300.0, 5772.0):
        assert abs(E.planck_entropy_flux(t) / (4 / 3 * C.SIGMA_SB * t ** 3) - 1) < 1e-6


def test_dilution_function():
    """Exact X(1) = 1; approximate eq. 5 within 2 % at delta = 1 and within 0.1 % at delta <= 0.1."""
    assert abs(float(E.x_dilution(1.0)) - 1.0) < 1e-6
    assert abs(E.x_dilution_exact(1.0) - 1.0) < 1e-6
    assert abs(float(E.x_dilution_approx(1.0)) - 1.0) < 0.02
    for d in (0.1, 1e-3, 1e-6):
        assert abs(float(E.x_dilution(d)) / float(E.x_dilution_approx(d)) - 1) < 1e-3
    assert np.isnan(E.x_dilution(0.0))
    # X increases as the radiation is diluted
    xs = E.x_dilution([1.0, 0.5, 1e-2, 1e-6])
    assert np.all(np.diff(xs) > 0)


def test_ts_round_trip():
    ts = np.array([280.0, 300.0, 315.0])
    lw_in = np.array([350.0, 420.0, 450.0])
    for eps in (0.95, 0.98, 0.99):
        lw_out = E.lw_out_from_ts(ts, lw_in, eps)
        back = E.t_surface(lw_out, lw_in, eps)
        assert np.allclose(back, ts, atol=1e-9)


def test_signs_negative_fluxes_kept():
    j = E.j_heat(np.array([-20.0, 30.0]), np.array([290.0, 300.0]))
    assert j[0] < 0 and j[1] > 0
    assert np.isclose(j[1], 0.1)


def test_nighttime_j_swout_zero_and_variants():
    sw_out = np.array([2.0, 40.0, 40.0, -1.0])
    sw_in = np.array([0.0, 200.0, 200.0, 0.0])
    is_day = np.array([False, True, True, False])
    a = E.j_sw_out(sw_out, sw_in, is_day, variant="a")
    b = E.j_sw_out(sw_out, sw_in, is_day, variant="b")
    assert a[0] == 0 and a[3] == 0 and b[0] == 0 and b[3] == 0
    assert np.isclose(a[1], 4 / 3 * 40 / C.T_SUN)
    assert 4 < b[1] / a[1] < 6          # diluted variant ~5x larger at delta ~ 6e-7


def test_lw_in_variants():
    lw = np.array([400.0]); ta = np.array([298.0])
    sky = E.j_lw_in(lw, ta, "sky"); air = E.j_lw_in(lw, ta, "air"); dil = E.j_lw_in(lw, ta, "diluted")
    assert np.isclose(sky[0], 4 / 3 * C.SIGMA_SB ** 0.25 * 400 ** 0.75)
    assert air[0] < sky[0] < dil[0] * 1.05      # T_sky < T_a -> sky variant larger than air variant


def _synthetic(n=48):
    """One synthetic day: SW_IN half-sine, radiative/turbulent fluxes of realistic size."""
    hour = np.arange(n) / 2.0
    day = (hour >= 6) & (hour <= 18)
    sw_in = np.where(day, 900 * np.sin(np.pi * (hour - 6) / 12), 0.0)
    df = pd.DataFrame({
        "TA_F": 25 + 5 * np.sin(np.pi * (hour - 9) / 12), "SW_IN_F": sw_in, "SW_OUT": 0.16 * sw_in + 1.5,
        "LW_IN_F": 410.0, "LW_OUT": 450 + 30 * np.sin(np.pi * (hour - 8) / 12), "NIGHT": (~day).astype(int),
        "H_F_MDS": np.where(day, 0.15 * sw_in, -15.0), "LE_F_MDS": np.where(day, 0.45 * sw_in, 5.0),
        "H_RANDUNC": 10.0, "LE_RANDUNC": 25.0,
    })
    return df


def test_budget_units_and_magnitude():
    J = E.budget(_synthetic(), C.CORE)
    s = J["sigma"].mean()
    assert 0.1 < s < 1.5, s                   # W m-2 K-1, order 0.1-1
    assert 1.5 < J["J_LWin"].mean() < 2.2 and 1.7 < J["J_LWout"].mean() < 2.5
    assert np.isclose(J["J_in"], J["J_SWin"] + J["J_LWin"]).all()
    assert np.isclose(J["sigma"], J["sigma_rad"] + J["sigma_turb"]).all()


def test_closure_scenarios_identity():
    """With both turbulent fluxes at the same temperature, S1 and S2 give the same sigma,
    namely sigma_S0 + R'/T; S3 with the period CR gives the same period mean when T is constant."""
    df = _synthetic()
    df["LW_OUT"] = 450.0                      # constant T_s so that the S3 identity is exact
    J0 = E.budget(df, C.Options(closure="S0"))
    J1 = E.budget(df, C.Options(closure="S1"))
    J2 = E.budget(df, C.Options(closure="S2"))
    J3 = E.budget(df, C.Options(closure="S3"))
    assert np.allclose(J1["sigma"], J2["sigma"])
    assert np.allclose(J1["sigma"], J0["sigma"] + J0["Rres"] / J0["Ts"])
    assert np.isclose(J3["sigma"].mean(), J1["sigma"].mean())
    assert np.allclose(J1["J_SWin"], J0["J_SWin"]) and np.allclose(J1["J_LWout"], J0["J_LWout"])


def test_sr_is_ratio_of_means():
    J = E.budget(_synthetic(), C.CORE)
    st = S.period_stats(J)
    assert np.isclose(st["SR"], J["J_out"].mean() / J["J_in"].mean())
    # the mean of half-hourly ratios is a different (unstable) statistic
    assert abs(st["SR"] - (J["J_out"] / J["J_in"]).mean()) > 1e-4


def test_bias_shifts_ts():
    df = _synthetic()
    J0 = E.budget(df, C.CORE)
    Jb = E.budget(df, C.Options(bias=(("LW_OUT", "abs", 10.0),)))
    assert (Jb["Ts"] > J0["Ts"]).all()
    assert Jb["J_LWout"].mean() > J0["J_LWout"].mean()


def test_bootstrap_shapes_and_coverage():
    df = pd.concat([_synthetic()] * 30, ignore_index=True)
    rng = np.random.default_rng(1)
    df["H_F_MDS"] = df["H_F_MDS"] + rng.normal(0, 20, len(df))
    J = E.budget(df, C.CORE)
    dates = np.repeat(np.arange(30), 48)
    mask = np.ones(len(df), dtype=bool)
    b, names = S.bootstrap_means(J, dates, mask, n_boot=200, seed=3)
    assert b.shape == (200, 6) and names[-1] == "SR"
    lo, hi = S.ci(b[:, names.index("sigma")])
    assert lo <= J["sigma"].mean() <= hi


def test_solar_elevation_equinox_noon():
    """Solar noon on the equinoxes at lat -16.5: elevation ~ 90 - 16.5 = 73.5 deg (within 1 deg,
    low-precision algorithm, sun declination not exactly zero at local noon); June solstice ~50 deg."""
    t = pd.Series(pd.to_datetime(["2015-03-20 11:45", "2015-09-23 11:45", "2015-06-21 11:45", "2015-12-21 11:45"]))
    el = L.solar_elevation(t)
    assert abs(el[0] - 73.5) < 1.0 and abs(el[1] - 73.5) < 1.0, el
    assert abs(el[2] - (90 - 16.5 - 23.44)) < 1.0, el
    assert abs(el[3] - (90 - abs(-16.5 + 23.44))) < 1.0, el
    # midnight: below the horizon
    assert L.solar_elevation(pd.Series(pd.to_datetime(["2015-03-20 00:00"])))[0] < 0


def test_ols_exact_line_and_cr():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0]); y = 0.7 * x + 18.0
    fit = CL.ols(x, y)
    assert np.isclose(fit["slope"], 0.7) and np.isclose(fit["intercept"], 18.0) and np.isclose(fit["r2"], 1.0)
    assert np.isclose(CL.cr_sum(np.array([10.0, 20.0]), np.array([30.0, 40.0]), np.array([100.0, 100.0])), 0.5)


def test_centred_rate_gap():
    t = pd.Series(pd.to_datetime(["2015-01-01 00:00", "2015-01-01 00:30", "2015-01-01 01:00", "2015-01-01 02:00", "2015-01-01 02:30"]))
    s = pd.Series([10.0, 11.0, 12.0, 14.0, 15.0])
    r = ST.centred_rate(s, t)
    assert np.isclose(r[1], 2.0 / 3600.0)              # regular neighbours: (12-10)/(2*1800 s)
    assert np.isnan(r[2]) and np.isnan(r[3])           # neighbours not exactly +-30 min
    assert np.isnan(r[0]) and np.isnan(r[4])


def test_diurnal_composite_toy():
    """Two days, slot 0 has data on both, slot 1 on one day only (masked by min_n=2); n_boot < 100 -> NaN band."""
    values = pd.Series([1.0, 3.0, 5.0]); slots = pd.Series([0, 0, 1]); dates = pd.Series(["d1", "d2", "d1"])
    c = S.diurnal_composite(values, slots, dates, np.ones(3, dtype=bool), min_n=2, n_boot=0)
    assert np.isclose(c.loc[0, "mean"], 2.0) and np.isnan(c.loc[1, "mean"]) and c.loc[0, "n"] == 2 and c.loc[1, "n"] == 1
    assert np.isnan(c.loc[0, "lo"]) and c.loc[0, "n_boot"] == 0
    c2 = S.diurnal_composite(values, slots, dates, np.ones(3, dtype=bool), min_n=2, n_boot=200)
    assert c2.loc[0, "n_boot"] == 200 and c2.loc[0, "lo"] <= 2.0 <= c2.loc[0, "hi"]


def test_bootstrap_difference_sign_and_stratified():
    df = pd.concat([_synthetic()] * 20, ignore_index=True)
    J = E.budget(df, C.CORE)
    dates = np.repeat(np.arange(20), 48)
    months = np.where(dates < 10, 1, 7); slots = np.tile(np.arange(48), 20)
    Jb = J.copy(); Jb["sigma"] = Jb["sigma"] + 0.2   # stratum b shifted up by 0.2
    Jab = pd.concat([J, Jb], ignore_index=True)
    d2 = np.concatenate([dates, dates + 100]); m_a = np.r_[np.ones(len(J), bool), np.zeros(len(J), bool)]
    r = S.bootstrap_difference(Jab, d2, ~m_a, m_a, n_boot=100, block_days=1)
    assert np.isclose(r["d_sigma"], 0.2, atol=1e-6) and r["d_sigma_lo"] > 0.19 and r["d_sigma_n_le0"] == 0
    assert r["d_sigma_frac_le0"].startswith("<")
    # stratified mean of a uniform sample equals the raw mean; bootstrap of it covers it
    st = S.stratified_mean(J, months, slots, np.ones(len(J), bool), min_n=1)
    assert np.isclose(st["sigma_strat"], J["sigma"].mean(), atol=1e-9)
    b, names = S.bootstrap_stratified(J, dates, months, slots, np.ones(len(J), bool), n_boot=50, block_days=3, min_n=1)
    assert b.shape == (50, 4) and np.all(np.isfinite(b))
    lo, hi = S.ci(b[:, names.index("sigma")]); assert lo <= J["sigma"].mean() + 1e-9 and hi >= J["sigma"].mean() - 1e-9


def test_analytic_decomposition_identity():
    J = E.budget(_synthetic(), C.CORE)
    A = E.sigma_analytic_terms(J)
    assert np.allclose(A["sigma_reconstructed"], J["sigma"], atol=1e-10)


def test_greybody_variants_order():
    lw_out = np.array([450.0]); lw_in = np.array([400.0]); ta = np.array([298.0])
    ts = E.t_surface(lw_out, lw_in, 0.98)
    jin = E.j_lw_in(lw_in, ta, "sky")
    core = E.j_lw_out(lw_out, ts); grey = E.j_lw_out(lw_out, ts, jin, 0.98, "grey"); refl = E.j_lw_out(lw_out, ts, jin, 0.98, "grey_refl", lw_in=lw_in, ta=ta)
    assert core[0] < grey[0] < refl[0]
    assert (refl[0] - core[0]) / core[0] < 0.05


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1; print(f"FAIL {name}: {exc!r}")
    sys.exit(1 if failed else 0)
