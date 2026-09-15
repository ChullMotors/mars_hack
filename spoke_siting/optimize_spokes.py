"""Greedy constrained spoke selection on the GP posterior."""
import numpy as np
from .geo import haversine_km
from .gp_surrogate import ELEV_MIN_M, ELEV_MAX_M

DEMAND_KWH = 400.0        # team's household design point
Z_LCB = 1.64              # 95% one-sided lower confidence bound
BASE_PV_AREA_M2 = 3000.0  # reference array; each spoke's array is SIZED to its site
MAX_PV_AREA_M2 = 7500.0   # largest array we will ship to one spoke (2.5x reference)

# Score is in "kWh/sol-equivalent" units; the three tradeoff constants are the
# user-defined knobs of the p-median framing (pipeline cost-per-km, panel cost, coverage).
LAMBDA_AREA = 200.0       # cost of one extra BASE array's worth of panels (doubling costs 200)
LAMBDA_DIST = 0.05        # cost per km of pipeline (2500 km costs 125)
LAMBDA_CROWD = 300.0      # coverage: penalty for sitting on top of an existing spoke
CROWD_SCALE_KM = 700.0    # planetary dispersion scale (~ half the spacing of 80 uniform sites)
MIN_SPACING_KM = 150.0    # hard minimum spacing between any two spokes


def select_spokes(cand, mean, std, n_spokes, range_km):
    """Joint greedy selection across ALL hubs (candidates from every hub disk concatenated,
    dist_km is to the candidate's own hub). Round-robin over hubs, one spoke per hub per
    pass, crowding applied against every spoke chosen so far.

    Each candidate's PV array is sized so its 95%-LCB capacity meets 400 kWh/sol:
        required_area = BASE * DEMAND / LCB(capacity at BASE area)
    (capacity is ~linear in array area). Sites needing more than MAX_PV_AREA_M2 are
    infeasible. Score = -panel cost - pipeline cost - crowding, so the optimiser trades
    a few poleward, bigger-array spokes for planetary coverage instead of piling
    everything onto the sunniest latitude."""
    lat, lon, dist, hub_id = cand["lat"], cand["lon"], cand["dist_km"], cand["hub_id"]
    lcb = mean - Z_LCB * std
    with np.errstate(divide="ignore"):
        req_area = np.where(lcb > 1.0, BASE_PV_AREA_M2 * DEMAND_KWH / np.maximum(lcb, 1.0), np.inf)
    req_area = np.maximum(req_area, BASE_PV_AREA_M2)
    elev_ok = (cand["elev"] >= ELEV_MIN_M) & (cand["elev"] <= ELEV_MAX_M)
    feasible = (req_area <= MAX_PV_AREA_M2) & elev_ok
    base = -LAMBDA_AREA * (req_area - BASE_PV_AREA_M2) / BASE_PV_AREA_M2 - LAMBDA_DIST * dist
    chosen, avail = [], elev_ok.copy()
    crowd = np.zeros_like(base)
    hubs = list(dict.fromkeys(hub_id.tolist()))
    for _ in range(n_spokes):
        for h in hubs:
            mine = hub_id == h
            pool = avail & feasible & mine
            if not pool.any():             # nothing feasible left for this hub: least-bad site, flagged
                pool = avail & mine
                if not pool.any():
                    continue
            score = np.where(np.isfinite(base), base, -1e9) - crowd
            j = int(np.argmax(np.where(pool, score, -np.inf)))
            chosen.append(j)
            d = haversine_km(lat, lon, lat[j], lon[j])
            crowd += LAMBDA_CROWD * np.exp(-(d / CROWD_SCALE_KM) ** 2)
            avail &= d >= MIN_SPACING_KM
    rows = []
    for j in chosen:
        rows.append(dict(hub_id=int(cand["hub_id"][j]), lat=lat[j], lon=lon[j], elev_m=float(cand["elev"][j]),
                         dist_km=float(dist[j]), cap_mean=float(mean[j]), cap_std=float(std[j]),
                         cap_lcb_at_base_area=float(lcb[j]), feasible=bool(feasible[j]),
                         pv_area_m2=float(min(req_area[j], MAX_PV_AREA_M2)),
                         required_pv_area_m2=float(req_area[j])))
    return rows
