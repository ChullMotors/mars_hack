"""Greedy constrained spoke selection on the GP posterior."""
import numpy as np
from .geo import haversine_km
from .gp_surrogate import ELEV_MIN_M, ELEV_MAX_M

DEMAND_KWH = 400.0        # team's household design point
Z_LCB = 1.64              # 95% one-sided lower confidence bound
LAMBDA_DIST = 0.20        # kWh/sol per km of pipeline  (100 km ~ 20 kWh/sol)
LAMBDA_CROWD = 150.0      # kWh/sol-equivalent penalty for sitting on top of another spoke
CROWD_SCALE_KM = 60.0     # repulsion length scale
MIN_SPACING_KM = 30.0     # hard minimum spacing between spokes


def select_spokes(cand, mean, std, n_spokes, pv_area_m2):
    """cand: dict from candidate_grid; mean/std: GP posterior (kWh/sol)."""
    lat, lon, dist = cand["lat"], cand["lon"], cand["dist_km"]
    lcb = mean - Z_LCB * std
    elev_ok = (cand["elev"] >= ELEV_MIN_M) & (cand["elev"] <= ELEV_MAX_M)
    feasible = (lcb >= DEMAND_KWH) & elev_ok
    base = mean - LAMBDA_DIST * dist
    chosen, avail = [], elev_ok.copy()
    for _ in range(n_spokes):
        crowd = np.zeros_like(base)
        for j in chosen:
            d = haversine_km(lat, lon, lat[j], lon[j])
            crowd += LAMBDA_CROWD * np.exp(-(d / CROWD_SCALE_KM) ** 2)
            avail &= d >= MIN_SPACING_KM
        score = base - crowd
        pool = avail & feasible
        if not pool.any():                 # no feasible site left: take best LCB, flag it
            pool = avail
            if not pool.any():
                break
            score = lcb - LAMBDA_DIST * dist - crowd
        j = int(np.argmax(np.where(pool, score, -np.inf)))
        chosen.append(j)
        avail[j] = False
    rows = []
    for j in chosen:
        rows.append(dict(hub_id=int(cand["hub_id"][j]), lat=lat[j], lon=lon[j], elev_m=float(cand["elev"][j]),
                         dist_km=float(dist[j]), cap_mean=float(mean[j]), cap_std=float(std[j]),
                         cap_lcb=float(lcb[j]), feasible=bool(feasible[j]),
                         pv_area_m2=pv_area_m2, required_pv_area_m2=float(pv_area_m2 * DEMAND_KWH / max(lcb[j], 1.0))))
    return rows
