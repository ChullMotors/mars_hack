"""
Site-dependent energy capacity built ON the team's household_sizing.py model.

The team's simulator uses a single solar-yield constant and a hand-set +/-15%
seasonal cosine. Here the per-sol generation is replaced by
    area_m2 * PV_EFF * insolation(lat, elev)[sol] * storm_opacity[sol]
and the storm Monte Carlo is reused unchanged. capacity() is the EXPENSIVE
function the GP surrogates: a bisection on daily demand, each step a
Monte Carlo of N storm-years through the sol-by-sol balance.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import household_sizing as hs  # noqa: E402  (teammate's model)
from .solar_site import daily_insolation_profile  # noqa: E402
from .thermal_site import heating_load_kwh_per_sol  # noqa: E402  (1D regolith thermal model)

hs.SEASONAL_AMPLITUDE = 0.0   # season now comes from the physics-based profile, not a cosine

PV_EFF = 0.22                 # CITED in household_sizing: 22%-efficient cells
# Fixed hardware per spoke (mid-range of the team's Optuna search space):
SPOKE_PV_AREA_M2 = 3000.0     # sized so ~50N is borderline for 400 kWh/sol; report required area per site
SPOKE_BATTERY_KWH = 1000.0
SPOKE_H2_KWH = 4000.0
N_STORM_YEARS = 40
RELIABILITY_TARGET = hs.RELIABILITY_TARGET


def storm_profiles(n_years, seed):
    rng = np.random.default_rng(seed)
    return np.stack([hs.generate_storm_year(rng) for _ in range(n_years)])


def reliability_at_demand(generation_profiles, demand_kwh):
    old = hs.DAILY_DEMAND_KWH
    hs.DAILY_DEMAND_KWH = float(demand_kwh)
    try:
        rel = [hs.simulate_year(1.0, SPOKE_BATTERY_KWH, SPOKE_H2_KWH, g) for g in generation_profiles]
    finally:
        hs.DAILY_DEMAND_KWH = old
    return float(np.mean(rel))


INCLUDE_HEATING = True        # subtract habitat heating load (thermal_site.py) sol by sol
TAU_RANGE = (0.3, 1.0)        # background (non-storm) dust optical depth, drawn once per
                              # Monte Carlo year. CITED range: Viking/MER clear-season tau
                              # ~0.3-1.0. Storms are NOT in tau -- household_sizing's
                              # storm multipliers handle them (no double counting).


def capacity_kwh_per_sol(lat_deg, lon_deg, elev_m, pv_area_m2=SPOKE_PV_AREA_M2,
                         n_years=N_STORM_YEARS, seed=0, tau=None, tol=2.0,
                         include_heating=None):
    """Max daily NON-heating demand (kWh/sol) the spoke meets at >= 95% reliability.
    Generation per sol = PV output under storms minus the habitat heating load from the
    1D regolith thermal model, so cold high-latitude sites are penalised twice (less sun,
    more heating). Background dust tau is uncertain: drawn ~U(TAU_RANGE) per Monte Carlo
    year (pass a float `tau` to fix it)."""
    if include_heating is None:
        include_heating = INCLUDE_HEATING
    rng = np.random.default_rng(seed + 12345)
    taus = np.full(n_years, float(tau)) if tau is not None else rng.uniform(*TAU_RANGE, size=n_years)
    insol = np.stack([daily_insolation_profile(lat_deg, elev_m, tau=t) for t in taus])   # (years, sols)
    clear = pv_area_m2 * PV_EFF * insol                                  # kWh/sol clear-sky
    gens = clear * storm_profiles(n_years, seed)                         # common random numbers
    if include_heating:
        # thermal model is ~0.4 s/call, so evaluate once at the mean tau (approximation)
        heat = heating_load_kwh_per_sol(lat_deg, elev_m, tau=float(np.mean(taus)))["profile"]
        gens = gens - heat[None, :]
    lo, hi = 0.0, float(clear.max())
    if reliability_at_demand(gens, hi) >= RELIABILITY_TARGET:
        return hi
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if reliability_at_demand(gens, mid) >= RELIABILITY_TARGET:
            lo = mid
        else:
            hi = mid
    return lo


if __name__ == "__main__":
    import time
    for lat in (0, 30, 45, 55, -35, -48):
        t = time.time()
        c = capacity_kwh_per_sol(lat, 0.0, 0.0)
        print(f"lat {lat:4d}: capacity {c:7.1f} kWh/sol   ({time.time()-t:.2f}s)")
