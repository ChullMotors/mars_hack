"""Greedy constrained spoke selection on the GP posterior."""
import numpy as np
from .geo import haversine_km
from .gp_surrogate import ELEV_MIN_M, ELEV_MAX_M

DEMAND_KWH = 400.0        # team's household design point
Z_LCB = 1.64              # 95% one-sided lower confidence bound
LAMBDA_DIST = 0.20        # kWh/sol per km of pipeline  (100 km ~ 20 kWh/sol)
LAMBDA_CROWD = 150.0      # kWh/sol-equivalent penalty for sitting on top of another spoke
# repulsion length scale and hard minimum spacing scale with the hub range
CROWD_SCALE_FRAC = 0.2    # 1000 km range -> 200 km repulsion scale
MIN_SPACING_FRAC = 0.05   # 1000 km range ->  50 km hard minimum spacing


def select_spokes(cand, mean, std, n_spokes, pv_area_m2, range_km):
    """Joint greedy selection across ALL hubs (candidates from every hub disk concatenated,
    dist_km is to the candidate's own hub). Round-robin over hubs, one spoke per hub per
    pass, with crowding applied against every spoke chosen so far -- so overlapping hub
    disks don't stack spokes on top of each other."""
    lat, lon, dist, hub_id = cand["lat"], cand["lon"], cand["dist_km"], cand["hub_id"]
    crowd_scale, min_spacing = CROWD_SCALE_FRAC * range_km, MIN_SPACING_FRAC * range_km
    lcb = mean - Z_LCB * std
    elev_ok = (cand["elev"] >= ELEV_MIN_M) & (cand["elev"] <= ELEV_MAX_M)
    feasible = (lcb >= DEMAND_KWH) & elev_ok
    base = mean - LAMBDA_DIST * dist
    fallback = lcb - LAMBDA_DIST * dist
    chosen, avail = [], elev_ok.copy()
    crowd = np.zeros_like(base)
    hubs = list(dict.fromkeys(hub_id.tolist()))
    for _ in range(n_spokes):
        for h in hubs:
            mine = hub_id == h
            pool = avail & feasible & mine
            score = base - crowd
            if not pool.any():             # no feasible site left for this hub: best LCB, flagged
                pool = avail & mine
                if not pool.any():
                    continue
                score = fallback - crowd
            j = int(np.argmax(np.where(pool, score, -np.inf)))
            chosen.append(j)
            d = haversine_km(lat, lon, lat[j], lon[j])
            crowd += LAMBDA_CROWD * np.exp(-(d / crowd_scale) ** 2)
            avail &= d >= min_spacing
    rows = []
    for j in chosen:
        rows.append(dict(hub_id=int(cand["hub_id"][j]), lat=lat[j], lon=lon[j], elev_m=float(cand["elev"][j]),
                         dist_km=float(dist[j]), cap_mean=float(mean[j]), cap_std=float(std[j]),
                         cap_lcb=float(lcb[j]), feasible=bool(feasible[j]),
                         pv_area_m2=pv_area_m2, required_pv_area_m2=float(pv_area_m2 * DEMAND_KWH / max(lcb[j], 1.0))))
    return rows
