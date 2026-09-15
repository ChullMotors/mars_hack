# Spoke siting with a Gaussian-process surrogate

**Track:** Life Support & Resource Systems. **Decision the ML drives:** where to put the
10 solar "spoke" colonies around each of the 8 ice "hub" sites (80 spokes), such that every
spoke can supply the household design point of **>= 400 kWh/sol at 95 % reliability**, with
the shortest pipelines to its hub and an even spread.

## Why a Gaussian process
Whether a site can hit 400 kWh/sol is *expensive* to evaluate. Per site we run:

1. a physics solar model (orbit -> zenith angle -> Beer-Lambert, background dust tau ~ U(0.3, 1.0) drawn per storm-year, pressure-scaled by
   MOLA elevation) giving a 668-sol insolation profile,
2. a 1D transient regolith + habitat thermal model (finite difference, one Mars year of
   diurnal forcing, CO2 frost clamp) giving the sol-by-sol **heating load**,
3. the team's `household_sizing.py` sol-by-sol energy balance, driven by that insolation
   minus heating, through **40 Monte Carlo dust-storm years**, inside a bisection on demand.

That is ~1 s per site, so a 15 000-site candidate grid would take hours and a planet at MOLA
resolution days. Instead we evaluate ~64 sites, fit a GP over (lat, lon, elevation) with an
ARD Matern kernel, and optimise on the posterior. The posterior *uncertainty* does two jobs:
the feasibility constraint is applied to the 95 % lower confidence bound (risk-aware siting),
and an active-learning loop spends extra expensive evaluations where uncertainty is high
*near the 400 kWh/sol decision boundary*.

## Two locked scenarios (one shared GP)
| | Scenario 1: self-sufficient | Scenario 2: hub-supported |
|---|---|---|
| array cap per spoke | 7500 m2 | 15000 m2 |
| energy piped from hub | none | 200 kWh/sol (hub H2) |
| outputs | `outputs/scenario1_self_sufficient/` | `outputs/scenario2_hub_supported/` |

Scenario 1 shows that solar-only spokes cannot survive a polar winter at 95 % reliability:
feasibility ends near 50 deg latitude, right at the hubs' own latitudes (which is why hubs
make H2). Scenario 2 lets the hub pipe part of the demand back, and spokes reach past 70 deg.
A constant per-sol import is exactly a demand reduction, so both scenarios reuse one GP.

## Optimisation
Each spoke's PV array is **sized to its site**. PV output scales with area but the heating
load does not, so with a second GP predicting mean heating load `h`:

    required_area = 3000 m2 * (400 - import + h) / (LCB95(capacity at 3000 m2) + h)

Sites needing more than the scenario's cap are infeasible. Greedy joint selection over all hubs (round-robin, one
spoke per hub per pass) on a 0.5 deg candidate grid within 2500 km of each hub:

    score = - 200 * (extra panel area / 3000 m2)   # panel cost
            - 0.05 * pipeline km                   # pipeline cost
            - 300 * sum_j exp(-(d_j / 700 km)^2)   # coverage: planetary-scale dispersion
    subject to  required area <= cap,  |elevation| <= 5 km,  spacing >= 150 km,
                >= 400 km from every hub

The three constants are the user-defined knobs of the p-median framing. The coverage term
spreads spokes over the reachable planet and buys a few poleward, bigger-array spokes
instead of piling everything on the sunniest latitude. If a hub cannot host 10 feasible
spokes, the least-bad sites are still reported and flagged.

## Run
    uv venv .venv && uv pip install -p .venv/bin/python numpy scipy matplotlib scikit-learn optuna
    .venv/bin/python -m spoke_siting.run            # ~2 min;  --fast for a 20 s smoke test
Outputs per scenario: `spokes.csv`, `spoke_map.png` (colour = parent hub, dashed = pipeline),
`hub_panels.png`; shared `outputs/gp_training_sites.csv` and `outputs/scenario_overlay.png` (both layouts). Individual models have self-tests:
`python -m spoke_siting.site_energy`, `python spoke_siting/thermal_site.py`.

## Files
| file | role |
|---|---|
| `solar_site.py` | per-site insolation profile (stand-in for the team's solar model, same physics) |
| `thermal_site.py` | 1D regolith/habitat thermal model -> heating load per sol |
| `site_energy.py` | wraps `household_sizing.py`: site-dependent capacity at 95 % reliability |
| `mola.py` | MOLA 4 px/deg elevation (`data/megt90n000cb.img`, PDS) |
| `gp_surrogate.py` | site sampling, GP fit, candidate grids |
| `optimize_spokes.py` | greedy constrained spoke selection on the posterior |
| `run.py` | end-to-end driver, active learning, figures |

## Honest caveats
- Solar and thermal models are ours, with labelled assumptions
  (diffuse fraction, habitat U-value, 20 W/m2 downwelling IR). Dust storms enter only via the
  team's storm Monte Carlo, not the thermal model.
- In this stand-in the capacity depends on latitude and elevation only, so the GP field is
  smooth; longitude is an input so a dust-climatology or thermal-inertia-aware model slots in
  with no code change.
- Spoke storage is fixed at 1000 kWh battery + 4000 kWh H2 (mid-range of the team's Optuna
  search space); only the PV array is sized per site.
