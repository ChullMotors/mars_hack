"""
End-to-end: sample sites -> expensive capacity evals -> GP -> (active learning) ->
greedy constrained spoke selection -> CSV + maps.
Run:  .venv/bin/python -m spoke_siting.run  [--spokes 10] [--range-km 300] [--fast]
"""
import argparse
import csv
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .geo import load_hubs
from .mola import _load as load_mola
from .gp_surrogate import sample_disk, evaluate_sites, fit_gp, features, candidate_grid
from .optimize_spokes import select_spokes, DEMAND_KWH, MAX_PV_AREA_M2
from .site_energy import SPOKE_PV_AREA_M2

ROOT = os.path.join(os.path.dirname(__file__), "..")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spokes", type=int, default=10)
    ap.add_argument("--range-km", type=float, default=2500.0, help="max pipeline / comms relay range from hub")
    ap.add_argument("--init-per-hub", type=int, default=8)
    ap.add_argument("--active-rounds", type=int, default=2)
    ap.add_argument("--active-per-round", type=int, default=8)
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    if a.fast:
        a.init_per_hub, a.active_rounds = 3, 0

    rng = np.random.default_rng(0)
    hubs = load_hubs(os.path.join(ROOT, "Hub Locations"))
    t0 = time.time()

    # 1. initial design: a few expensive evaluations per hub disk
    X_sites = np.vstack([sample_disk(h, a.init_per_hub, a.range_km, rng) for h in hubs])
    print(f"Evaluating {len(X_sites)} initial sites with the team's Monte Carlo energy model ...")
    y = evaluate_sites(X_sites)
    gp = fit_gp(features(*X_sites.T), y)

    # candidate grids (cheap) for every hub
    grids = {h["id"]: candidate_grid(h, a.range_km) for h in hubs}
    allc = {k: np.concatenate([g[k] for g in grids.values()]) for k in ("lat", "lon", "elev", "dist_km", "hub_id")}

    # 2. active learning: spend more expensive evals where the surrogate is least sure
    for r in range(a.active_rounds):
        mean, std = gp.predict(features(allc["lat"], allc["lon"], allc["elev"]), return_std=True)
        # prioritise uncertainty near the 400 kWh/sol decision boundary
        acq = std * np.exp(-((mean - DEMAND_KWH) / 150.0) ** 2) + 0.2 * std
        idx = []
        for _ in range(a.active_per_round):
            j = int(np.argmax(acq)); idx.append(j)
            from .geo import haversine_km
            acq *= 1 - np.exp(-(haversine_km(allc["lat"], allc["lon"], allc["lat"][j], allc["lon"][j]) / (0.15 * a.range_km)) ** 2)
        new = np.column_stack([allc["lat"][idx], allc["lon"][idx], allc["elev"][idx]])
        print(f"Active round {r+1}: evaluating {len(new)} most-informative sites ...")
        X_sites = np.vstack([X_sites, new]); y = np.concatenate([y, evaluate_sites(new, verbose=False)])
        gp = fit_gp(features(*X_sites.T), y)
    print("GP kernel:", gp.kernel_)

    # 3. optimise spoke placement per hub on the posterior
    post = {}
    mean_all, std_all = gp.predict(features(allc["lat"], allc["lon"], allc["elev"]), return_std=True)
    rows = select_spokes(allc, mean_all, std_all, a.spokes, a.range_km)
    rows.sort(key=lambda r: r["hub_id"])
    for h in hubs:
        g = grids[h["id"]]
        post[h["id"]] = gp.predict(features(g["lat"], g["lon"], g["elev"]), return_std=True)

    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    with open(os.path.join(ROOT, "outputs", "spokes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(os.path.join(ROOT, "outputs", "gp_training_sites.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["lat", "lon", "elev_m", "capacity_kwh_per_sol"])
        w.writerows(np.column_stack([X_sites, y]).tolist())

    # 4. report
    print(f"\n{len(X_sites)} expensive evaluations, {sum(len(g['lat']) for g in grids.values())} candidate sites scored by GP, {time.time()-t0:.0f}s total")
    print(f"{'hub':>3} {'feasible':>8} {'lat range':>12} {'mean dist km':>12} {'PV area m2 (min-max)':>22}")
    for h in hubs:
        hr = [r for r in rows if r["hub_id"] == h["id"]]
        nf = sum(r["feasible"] for r in hr)
        lats = [r["lat"] for r in hr]; areas = [r["pv_area_m2"] for r in hr]
        print(f"{h['id']:>3} {nf:>5}/{len(hr):<2} {min(lats):5.0f}..{max(lats):4.0f} {np.mean([r['dist_km'] for r in hr]):12.0f} "
              f"{min(areas):10.0f}-{max(areas):.0f}")

    plot(hubs, grids, post, rows, X_sites, y, a)


def plot(hubs, grids, post, rows, X_sites, y, a):
    mola = load_mola()
    # --- global map ---
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(mola / 1000, extent=[0, 360, -90, 90], cmap="gray", origin="upper", aspect="auto", alpha=0.85)
    sc = ax.scatter([r["lon"] for r in rows], [r["lat"] for r in rows], c=[r["cap_mean"] for r in rows],
                    cmap="viridis", s=[12 + 40 * (r["pv_area_m2"] / MAX_PV_AREA_M2) for r in rows], vmin=100, vmax=900, edgecolors="none", zorder=3)
    ax.scatter([r["lon"] for r in rows if not r["feasible"]], [r["lat"] for r in rows if not r["feasible"]],
               facecolors="none", edgecolors="red", s=60, linewidths=0.8, zorder=4, label=f"infeasible (needs > {MAX_PV_AREA_M2:.0f} m2 array)")
    ax.scatter([h["lon"] for h in hubs], [h["lat"] for h in hubs], marker="*", s=220, c="orange", edgecolors="k", zorder=5, label="hub (ice)")
    for h in hubs:
        ax.annotate(f"H{h['id']}", (h["lon"], h["lat"]), xytext=(6, 6), textcoords="offset points", color="w", fontsize=9)
    plt.colorbar(sc, ax=ax, label="GP posterior mean capacity at 95% reliability (kWh/sol)", shrink=0.8)
    ax.set_xlabel("longitude (°E)"); ax.set_ylabel("latitude"); ax.set_xlim(0, 360); ax.set_ylim(-90, 90)
    ax.set_title(f"{a.spokes} spokes/hub within {a.range_km:.0f} km, arrays sized per site (marker size)\n"
                 f"GP surrogate of the storm Monte Carlo energy model, {len(y)} expensive evaluations", fontsize=11)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(ROOT, "outputs", "spoke_map.png"), dpi=130); plt.close(fig)

    # --- per-hub panels: posterior mean, uncertainty, chosen spokes ---
    fig, axes = plt.subplots(2, 8, figsize=(26, 7))
    for k, h in enumerate(hubs):
        g = grids[h["id"]]; mean, std = post[h["id"]]
        hr = [r for r in rows if r["hub_id"] == h["id"]]
        for row_i, (vals, cmap, lab, vmin, vmax) in enumerate([(mean, "viridis", "mean kWh/sol", 100, 900), (std, "magma", "std kWh/sol", 0, None)]):
            ax = axes[row_i, k]
            s = ax.scatter(g["lon"], g["lat"], c=vals, s=6, cmap=cmap, vmin=vmin, vmax=vmax, marker="s")
            if row_i == 0:
                ax.tricontour(g["lon"], g["lat"], mean - 1.64 * std, levels=[DEMAND_KWH], colors="w", linewidths=1)
            ax.scatter([r["lon"] for r in hr], [r["lat"] for r in hr], c="none", edgecolors=["k" if r["feasible"] else "red" for r in hr], s=40, linewidths=1.2)
            ax.scatter(h["lon"], h["lat"], marker="*", s=180, c="orange", edgecolors="k")
            m = (np.abs(X_sites[:, 0] - h["lat"]) < 6) & (np.abs(((X_sites[:, 1] - h["lon"] + 180) % 360) - 180) < 9)
            ax.scatter(X_sites[m, 1], X_sites[m, 0], marker="x", c="cyan" if row_i else "w", s=25, linewidths=0.8)
            ax.set_title(f"Hub {h['id']} ({h['lat']:.0f}°, {h['lon']:.0f}°E) {lab}" if row_i == 0 else "GP uncertainty", fontsize=9)
            ax.set_aspect(1 / np.cos(np.deg2rad(h["lat"]))); ax.tick_params(labelsize=7)
            plt.colorbar(s, ax=ax, shrink=0.8)
    fig.suptitle("Per-hub GP posterior (x = expensive evaluations, white line = LCB 400 kWh/sol boundary, circles = chosen spokes, red = infeasible)", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(ROOT, "outputs", "hub_panels.png"), dpi=110); plt.close(fig)
    print("wrote outputs/spokes.csv, outputs/spoke_map.png, outputs/hub_panels.png")


if __name__ == "__main__":
    main()
