"""Heat-storage scenarios for the diurnal diagnostics (METHODS_SPEC.md, section 2.8).

G_F_MDS is absent. Storage is internal to the control volume: at daily and longer
scales its mean is ~0 and does not enter sigma; at half-hourly scale
sigma_HH = J_out - J_in + dQ/T_store. Everything here is labelled Preliminary/Scenario.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C


def soil_name(c_soil: float, z_eff: float) -> str:
    """Column name of the soil-storage scenario (single place where the key is built)."""
    return f"dQ_soil_C{c_soil / 1e6:.1f}_z{z_eff:.2f}"


def water_name(h: float, proxy: str = "ts1") -> str:
    """Column name of the water-column scenario; proxy 'ts1' (default) or 'skin'."""
    return f"dQ_water_h{h:.1f}" + ("" if proxy == "ts1" else "_skinproxy")


def centred_rate(series: pd.Series, time: pd.Series, dt=C.DT_HH_S) -> pd.Series:
    """Centred time derivative (K s-1) on a regular half-hourly series; NaN at gaps or
    where the neighbours are not exactly +-30 min apart."""
    s = series.to_numpy(dtype=float)
    t = time.to_numpy()
    rate = np.full(len(s), np.nan)
    if len(s) < 3:
        return pd.Series(rate, index=series.index)
    ok = (t[2:] - t[:-2]) == np.timedelta64(int(2 * dt), "s")
    rate[1:-1] = np.where(ok, (s[2:] - s[:-2]) / (2 * dt), np.nan)
    return pd.Series(rate, index=series.index)


def dq_soil(df: pd.DataFrame, c_soil: float, z_eff: float) -> pd.Series:
    """Scenario (ii): dQ_soil = C_s z_eff dTS_1/dt  (W m-2). Depth of TS_1 unknown."""
    return c_soil * z_eff * centred_rate(df["TS_F_MDS_1"], df["time"])


def dq_water(df: pd.DataFrame, h: float, ts: pd.Series, months=C.FLOODED_MONTHS_SCENARIO, proxy="ts1") -> pd.Series:
    """Scenario (iii): dQ_w = rho_w c_w h dT_w/dt (W m-2), applied only in months flagged as
    flooded; NaN elsewhere. proxy 'ts1': dT_w/dt from TS_F_MDS_1 (bulk-like, damped);
    proxy 'skin': from the radiometric T_s (rejected as the default: the skin temperature
    changes by ~1 K per half-hour at sunrise, which for a 1 m column implies > 1000 W m-2,
    physically implausible for the bulk water; kept only for the run-log comparison)."""
    temp = (df["TS_F_MDS_1"] + C.T0) if proxy == "ts1" else ts
    dq = C.RHO_C_WATER * h * centred_rate(temp, df["time"])
    return dq.where(df["month"].isin(months))


def storage_scenarios(df: pd.DataFrame, ts: pd.Series) -> pd.DataFrame:
    """All storage scenarios as columns (W m-2) plus the entropy rate dS/dt = dQ/T_store
    (W m-2 K-1) with T_store = TS_1 + 273.15 for both soil and water (the TS_1 sensor is the
    proxy of the bulk store in both scenarios; the radiometric T_s is used only for the
    rejected skin-proxy column). `ts` is only needed for that skin-proxy column."""
    out = pd.DataFrame(index=df.index)
    t_store = df["TS_F_MDS_1"] + C.T0
    for c_s in C.C_SOIL_RANGE:
        for z in C.Z_SOIL_RANGE:
            name = soil_name(c_s, z)
            out[name] = dq_soil(df, c_s, z)
            out["dS_" + name] = out[name] / t_store
    for h in C.H_WATER_RANGE:
        name = water_name(h)
        out[name] = dq_water(df, h, ts, proxy="ts1")
        out["dS_" + name] = out[name] / t_store
    out[water_name(1.0, "skin")] = dq_water(df, 1.0, ts, proxy="skin")
    return out
