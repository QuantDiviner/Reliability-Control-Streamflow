"""HBV-Lite: minimal lumped HBV-style hydrological model.

References
----------
- Bergström S. (1976) Development and application of a conceptual runoff model
  for Scandinavian catchments. SMHI Reports RHO 7.
- Lindström G., Johansson B., Persson M., Gardelin M., Bergström S. (1997)
  Development and test of the distributed HBV-96 hydrological model. J. Hydrol.
  201:272-288.
- Seibert J., Vis M. J. P. (2012) Teaching hydrological modeling with a
  user-friendly catchment-runoff-model software package. HESS 16:3315-3325.

Implementation notes
--------------------
- Lumped (single-cell) HBV with snow + soil moisture + 2 groundwater buckets.
- State variables exposed: ``swe`` (snow water equivalent, mm) and
  ``sm`` (unsaturated zone soil moisture, mm). These are returned as
  per-day arrays alongside discharge ``q``.
- This implementation is intentionally compact (~150 lines of model code)
  and follows the parameter ranges in Seibert & Vis 2012 Table 1.
- No external dependencies beyond NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np


@dataclass
class HBVParameters:
    """HBV-Lite parameter set.

    Default values follow the central tendency of Seibert & Vis 2012 Table 1
    calibrated ranges across diverse catchments. For per-basin instantiation,
    derive parameters from CAMELS-US attributes (see ``params_from_camels``).
    """
    # Snow routine
    tt: float = 0.0           # threshold temperature (degC)
    cfmax: float = 3.0        # snowmelt factor (mm / degC / day)
    sfcf: float = 1.0         # snowfall correction (-)
    cwh: float = 0.10         # water-holding capacity in snowpack (-)
    cfr: float = 0.05         # refreezing factor (-)

    # Soil routine
    fc: float = 250.0         # field capacity (mm)
    lp: float = 0.7           # PET threshold fraction of FC (-)
    beta: float = 1.5         # shape parameter for runoff coefficient (-)

    # Response routine (2 buckets: UZ, LZ)
    perc: float = 1.5         # max percolation UZ -> LZ (mm/day)
    uzl: float = 50.0         # threshold for K0 outflow (mm)
    k0: float = 0.30          # near-surface recession (1/day)
    k1: float = 0.10          # interflow recession (1/day)
    k2: float = 0.05          # baseflow recession (1/day)

    # Routing
    maxbas: int = 2           # triangular routing window (days)


def params_from_camels(attrs: Dict[str, float]) -> HBVParameters:
    """Derive a plausible HBVParameters from CAMELS-US per-basin attributes.

    This is a *physically-motivated heuristic*, not a calibrated estimate.
    For exp005 Lean MVP we just need parameter spreads consistent with
    real basin diversity; the absolute discharge magnitudes only need to
    be physically plausible (since LSTM is trained on the synthetic Q
    we generate, not real Q).

    Required keys in ``attrs``:
        soil_porosity, soil_depth_pelletier, frac_snow, aridity, p_mean

    Returns
    -------
    HBVParameters
    """
    p = HBVParameters()

    # FC scales with soil storage capacity (porosity x depth, in mm).
    porosity = float(attrs.get("soil_porosity", 0.45))
    depth_m = float(attrs.get("soil_depth_pelletier", 1.0))
    p.fc = float(np.clip(porosity * depth_m * 1000.0 * 0.4, 50.0, 600.0))

    # CFMAX higher in snow-dominated basins (more vigorous melt).
    fs = float(attrs.get("frac_snow", 0.0))
    p.cfmax = float(np.clip(2.0 + 4.0 * fs, 1.5, 6.0))

    # Beta higher in arid basins (sharper threshold for runoff).
    aridity = float(attrs.get("aridity", 1.0))
    p.beta = float(np.clip(1.0 + 0.5 * np.log1p(max(aridity - 1.0, 0.0)), 1.0, 3.0))

    # K2 (baseflow recession) slower in humid basins with deep groundwater.
    if aridity < 1.0:
        p.k2 = 0.03
    elif aridity > 1.5:
        p.k2 = 0.08
    # else: leave default 0.05

    return p


def simulate(
    precip: np.ndarray,
    temp: np.ndarray,
    pet: np.ndarray,
    params: HBVParameters,
    initial_state: Dict[str, float] | None = None,
    spinup_days: int = 365,
) -> Dict[str, np.ndarray]:
    """Run HBV-Lite over a forcing series.

    Parameters
    ----------
    precip : array, mm/day
    temp : array, degC
    pet : array, mm/day
    params : HBVParameters
    initial_state : dict, optional
        Override default initial state. Keys: swe, sm, uz, lz.
    spinup_days : int
        Days to discard at the start to allow state convergence.
        Outputs are still full-length; the caller can mask first
        ``spinup_days`` if desired.

    Returns
    -------
    dict with arrays (full input length):
        q       — discharge (mm/day)
        swe     — snow water equivalent end-of-day (mm)
        sm      — unsaturated soil moisture end-of-day (mm)
        uz      — upper zone storage (mm)
        lz      — lower zone storage (mm)
        recharge — UZ recharge (mm/day; the "P_eff" component)
        actual_et — actual ET (mm/day)
    """
    n = len(precip)
    assert len(temp) == n and len(pet) == n, "precip/temp/pet must align"

    # Initial state
    state = {"swe": 0.0, "sm": params.fc * 0.5, "uz": 0.0, "lz": 0.0, "snow_liquid": 0.0}
    if initial_state is not None:
        state.update(initial_state)

    swe_arr = np.zeros(n)
    sm_arr = np.zeros(n)
    uz_arr = np.zeros(n)
    lz_arr = np.zeros(n)
    q_raw = np.zeros(n)
    rec_arr = np.zeros(n)
    aet_arr = np.zeros(n)

    for i in range(n):
        p_i = max(precip[i], 0.0)
        t_i = temp[i]
        pet_i = max(pet[i], 0.0)

        # --- Snow routine (degree-day) ---
        if t_i < params.tt:
            snowfall = p_i * params.sfcf
            rainfall = 0.0
        else:
            snowfall = 0.0
            rainfall = p_i

        state["swe"] += snowfall

        if t_i > params.tt:
            potential_melt = params.cfmax * (t_i - params.tt)
            melt = min(state["swe"], potential_melt)
            state["swe"] -= melt
        else:
            potential_refreeze = params.cfr * params.cfmax * (params.tt - t_i)
            refreeze = min(state["snow_liquid"], potential_refreeze)
            state["snow_liquid"] -= refreeze
            state["swe"] += refreeze
            melt = 0.0

        # Liquid water in snowpack: rain + melt accumulates until capacity exceeded
        state["snow_liquid"] += rainfall + melt
        capacity = params.cwh * state["swe"]
        if state["snow_liquid"] > capacity:
            water_input = state["snow_liquid"] - capacity
            state["snow_liquid"] = capacity
        else:
            water_input = 0.0

        # --- Soil moisture routine ---
        # Effective recharge fraction: (sm/fc)^beta
        if state["sm"] > 0:
            ratio = min(state["sm"] / params.fc, 1.0)
            recharge_fraction = ratio ** params.beta
        else:
            recharge_fraction = 0.0
        recharge = water_input * recharge_fraction
        infiltration = water_input - recharge

        state["sm"] += infiltration

        # Actual ET
        if state["sm"] > params.lp * params.fc:
            aet = pet_i
        else:
            aet = pet_i * (state["sm"] / max(params.lp * params.fc, 1e-9))
        aet = min(aet, state["sm"])
        state["sm"] -= aet

        # Cap soil at FC; excess becomes additional recharge
        if state["sm"] > params.fc:
            recharge += state["sm"] - params.fc
            state["sm"] = params.fc

        # --- Response routine (2 buckets) ---
        state["uz"] += recharge

        # Percolation UZ -> LZ
        perc = min(state["uz"], params.perc)
        state["uz"] -= perc
        state["lz"] += perc

        # K0 (near-surface, only above threshold)
        q0 = max(state["uz"] - params.uzl, 0.0) * params.k0
        state["uz"] -= q0
        # K1 (interflow)
        q1 = state["uz"] * params.k1
        state["uz"] -= q1
        # K2 (baseflow)
        q2 = state["lz"] * params.k2
        state["lz"] -= q2

        q_today = q0 + q1 + q2

        swe_arr[i] = state["swe"] + state["snow_liquid"]
        sm_arr[i] = state["sm"]
        uz_arr[i] = state["uz"]
        lz_arr[i] = state["lz"]
        q_raw[i] = q_today
        rec_arr[i] = recharge
        aet_arr[i] = aet

    # --- Triangular routing ---
    if params.maxbas <= 1:
        q_routed = q_raw.copy()
    else:
        weights = _triangular_weights(params.maxbas)
        q_routed = np.convolve(q_raw, weights, mode="full")[: len(q_raw)]

    return {
        "q": q_routed,
        "swe": swe_arr,
        "sm": sm_arr,
        "uz": uz_arr,
        "lz": lz_arr,
        "recharge": rec_arr,
        "actual_et": aet_arr,
    }


def _triangular_weights(maxbas: int) -> np.ndarray:
    """Symmetric triangular weights summing to 1 over `maxbas` days."""
    half = maxbas / 2.0
    centers = np.arange(maxbas) + 0.5
    w = np.maximum(0.0, half - np.abs(centers - half))
    w = w / w.sum()
    return w


def estimate_pet_hamon(temp_c: np.ndarray, day_of_year: np.ndarray, latitude_deg: float) -> np.ndarray:
    """Hamon (1961) potential evapotranspiration (mm/day).

    Used when CAMELS does not directly provide PET.
    """
    lat_rad = np.deg2rad(latitude_deg)
    # Solar declination (rad)
    decl = 0.4093 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
    # Day length (hours)
    cos_omega = -np.tan(lat_rad) * np.tan(decl)
    cos_omega = np.clip(cos_omega, -1.0, 1.0)
    day_length = 24.0 * np.arccos(cos_omega) / np.pi
    # Saturation vapor pressure: Tetens gives kPa; Hamon original formula uses mb (= hPa = 10*kPa).
    es_kpa = 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))
    es_mb = es_kpa * 10.0
    pet = 0.1651 * (day_length / 12.0) * (216.7 * es_mb / (temp_c + 273.3))
    return np.maximum(pet, 0.0)


if __name__ == "__main__":
    # Self-test: synthetic forcing, check water balance
    rng = np.random.default_rng(42)
    n = 1000
    p = np.maximum(0, rng.exponential(3.0, n) * (rng.random(n) < 0.4))
    t = 15 + 10 * np.sin(2 * np.pi * np.arange(n) / 365.25) + rng.normal(0, 3, n)
    pet = np.maximum(0, 4 * (t > 0) + rng.normal(0, 0.5, n))
    params = HBVParameters()
    out = simulate(p, t, pet, params)
    p_total = p.sum()
    q_total = out["q"].sum()
    aet_total = out["actual_et"].sum()
    storage_change = (out["swe"][-1] + out["sm"][-1] + out["uz"][-1] + out["lz"][-1]) - (params.fc * 0.5)
    print(f"Water balance over {n} days:")
    print(f"  P  = {p_total:.1f}")
    print(f"  Q  = {q_total:.1f}")
    print(f"  ET = {aet_total:.1f}")
    print(f"  ΔS = {storage_change:.1f}")
    print(f"  Residual (P - Q - ET - ΔS) = {p_total - q_total - aet_total - storage_change:.2f} mm")
    print(f"  Relative imbalance: {(p_total - q_total - aet_total - storage_change) / p_total * 100:.2f}%")
