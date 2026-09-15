# mars_hack

Mars hackathon project.

**Track:** Life Support & Resource Systems — a hub-and-spoke Martian settlement.
Hubs sit on subsurface ice (SWIM) and make water, H2 and O2; spokes are sited for solar power
and feed energy back over pipelines.

- `household_sizing.py` — sol-by-sol energy balance with Monte Carlo dust storms; Optuna sizes
  the lightest PV / battery / H2 design for 400 kWh/sol at 95 % reliability.
- `spoke_siting/` — **ML spoke siting**: a Gaussian-process surrogate of the (expensive) per-site
  capacity model, with active learning and a constrained optimiser placing 10 spokes per hub.
  See `spoke_siting/README.md`. Run: `.venv/bin/python -m spoke_siting.run`.
- `Hub Locations` — the 8 hub sites. `outputs/` — spoke table and maps.
