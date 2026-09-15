# PLAN — ML-assisted spoke siting (GP surrogate + optimisation)

Written 19:25 BST, code freeze 20:35. Core must run end to end by ~20:05.

## Decision the ML drives
Where to place 10 spoke colonies around each of the 8 hubs (80 spokes), such that each
spoke can deliver >= 400 kWh/sol at 95% reliability (teammate's energy model), while
minimising pipeline distance to its hub and spreading spokes out.

## Why a GP
The per-site energy capacity is an *expensive* function: it runs the team's sol-by-sol
Monte Carlo (60 storm-years) inside a bisection on demand — seconds per site, hours for a
planet-wide grid. We evaluate it at ~50 sampled sites, fit a Gaussian process over
(lat, lon, elevation), and optimise siting on the posterior. The posterior variance tells
the settlement where the next expensive evaluation/survey should go.

## Components (core — main agent)
1. `spoke_siting/solar_site.py` — stand-in for the team's solar model. Per-site 668-sol
   daily insolation profile (kWh/m²/sol): Mars orbit (e=0.0934, obliquity 25.19°, Ls),
   daily-integrated cos(zenith), Beer-Lambert with tau=0.5 scaled by surface pressure
   ratio exp(-elev/11.1 km). Swap for the team's model when available.
2. `spoke_siting/mola.py` — MOLA MEGDR 4 ppd elevation (2 MB download, np.fromfile).
   Fallback: elev=0 if offline. Provides elevation-safety cutoff (reject > +5 km, < -5 km).
3. `spoke_siting/site_energy.py` — imports `household_sizing` (teammate's model) and
   replaces its constant yield + 15% cosine with the site's insolation profile.
   `capacity_kwh_per_sol(lat, lon, elev, pv_area_m2)` = max daily demand meeting 95%
   reliability (bisection over demand, Monte Carlo storms). This is the expensive f(x).
4. `spoke_siting/gp_surrogate.py` — sample sites in 300 km disks around hubs, evaluate f,
   fit sklearn GP (Matern ARD + WhiteKernel), predict mean/std on candidate grid.
5. `spoke_siting/optimize_spokes.py` — greedy sequential selection, 10 per hub:
   objective = capacity_mean - λ_d·dist_km - λ_c·crowding(min dist to already-chosen)
   subject to LCB(capacity) = mean - 1.64·std >= 400 kWh/sol, dist <= 300 km (comms/
   pipeline range — one simple intuitive constant), elevation cutoff.
6. `spoke_siting/run.py` — runs everything, writes `outputs/spokes.csv`,
   `outputs/spoke_map.png` (hubs, posterior mean, uncertainty, chosen spokes).
7. README section: what, which track, how ML is used, how to run.

## Parallel — sub-agent (option 3)
`spoke_siting/thermal_site.py` — 1D transient regolith/habitat thermal model (explicit
finite difference, one Mars year, diurnal forcing). Exposes
`heating_load_kwh_per_sol(lat_deg, elev_m, thermal_inertia=None, albedo=None) -> dict`
(annual mean and worst-sol heating load for a 4-person habitat). If it lands in time,
net capacity = capacity - heating_load becomes the GP target / constraint.

## Stretch
Active-survey loop: pick next evaluation site by UCB on the siting objective.

## Defaults
10 spokes/hub · 300 km range · PV area fixed per spoke (from teammate's Optuna design,
~ pv_kwh_day/0.132 m²) · tau=0.5 · λ_d, λ_c set so 100 km ≈ 20 kWh/sol.
