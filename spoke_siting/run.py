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
    # Two locked scenarios, sharing one GP:
    SCENARIOS = [
        dict(name="scenario1_self_sufficient", max_area=MAX_PV_AREA_M2, hub_import=0.0,
             label="Scenario 1: self-sufficient spokes, array cap 7500 m2, no hub import"),
        dict(name="scenario2_hub_supported", max_area=15000.0, hub_import=200.0,
             label="Scenario 2: hub-supported spokes, array cap 15000 m2, 200 kWh/sol piped from hub"),
    ]
    if a.fast:
        a.init_per_hub, a.active_rounds = 3, 0

    rng = np.random.default_rng(0)
    hubs = load_hubs(os.path.join(ROOT, "Hub Locations"))
    t0 = time.time()

    # 1. initial design: a few expensive evaluations per hub disk
    X_sites = np.vstack([sample_disk(h, a.init_per_hub, a.range_km, rng) for h in hubs])
    print(f"Evaluating {len(X_sites)} initial sites with the team's Monte Carlo energy model ...")
    y, heat = evaluate_sites(X_sites)
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
        y_new, h_new = evaluate_sites(new, verbose=False)
        X_sites = np.vstack([X_sites, new]); y = np.concatenate([y, y_new]); heat = np.concatenate([heat, h_new])
        gp = fit_gp(features(*X_sites.T), y)
    print("GP kernel:", gp.kernel_)

    # 3. optimise spoke placement on the posterior -- once per scenario
    mean_all, std_all = gp.predict(features(allc["lat"], allc["lon"], allc["elev"]), return_std=True)
    gp_heat = fit_gp(features(*X_sites.T), heat)          # second surrogate: mean heating load
    heat_all = np.maximum(gp_heat.predict(features(allc["lat"], allc["lon"], allc["elev"])), 0.0)
    print("heating GP kernel:", gp_heat.kernel_)
    post = {}
    for h in hubs:
        g = grids[h["id"]]
        post[h["id"]] = gp.predict(features(g["lat"], g["lon"], g["elev"]), return_std=True)
    with open(os.path.join(ROOT, "outputs", "gp_training_sites.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["lat", "lon", "elev_m", "capacity_kwh_per_sol", "heating_kwh_per_sol"])
        w.writerows(np.column_stack([X_sites, y, heat]).tolist())
    print(f"\n{len(X_sites)} expensive evaluations, {len(allc['lat'])} candidate sites scored by GP, {time.time()-t0:.0f}s")

    for sc in SCENARIOS:
        rows = select_spokes(allc, mean_all, std_all, heat_all, a.spokes, a.range_km, sc["max_area"], sc["hub_import"], hubs=hubs)
        rows.sort(key=lambda r: r["hub_id"])
        out = os.path.join(ROOT, "outputs", sc["name"]); os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "spokes.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"\n=== {sc['label']} ===")
        print(f"{'hub':>3} {'feasible':>8} {'lat range':>12} {'mean dist km':>12} {'PV area m2 (min-max)':>22}")
        for h in hubs:
            hr = [r for r in rows if r["hub_id"] == h["id"]]
            nf = sum(r["feasible"] for r in hr)
            lats = [r["lat"] for r in hr]; areas = [r["pv_area_m2"] for r in hr]
            print(f"{h['id']:>3} {nf:>5}/{len(hr):<2} {min(lats):5.0f}..{max(lats):4.0f} {np.mean([r['dist_km'] for r in hr]):12.0f} "
                  f"{min(areas):10.0f}-{max(areas):.0f}")
        print(f"  poleward of own hub: {sum(abs(r['lat']) > abs(next(h['lat'] for h in hubs if h['id']==r['hub_id'])) for r in rows)}/{len(rows)} spokes;"
              f"  highest |lat| {max(abs(r['lat']) for r in rows):.0f} deg")
        plot(hubs, grids, post, rows, X_sites, y, a, sc, out)
        sc["rows"] = rows
    plot_overlay(hubs, SCENARIOS, a)


HUB_COLORS = plt.cm.tab10.colors


def draw_network(ax, hubs, rows, max_area, marker="o", alpha=1.0, label_prefix="", lines=True):
    for k, h in enumerate(hubs):
        c = HUB_COLORS[k % 10]
        hr = [r for r in rows if r["hub_id"] == h["id"]]
        if lines:
            for r in hr:
                lon = r["lon"] + (360 if r["lon"] - h["lon"] < -180 else -360 if r["lon"] - h["lon"] > 180 else 0)
                for shift in (0, 360, -360):   # draw wrapped copies so meridian-crossing pipelines stay visible
                    ax.plot([h["lon"] + shift, lon + shift], [h["lat"], r["lat"]], ls="--", lw=0.7, color=c, alpha=0.6 * alpha, zorder=2)
        ax.scatter([r["lon"] for r in hr], [r["lat"] for r in hr], marker=marker, color=c, alpha=alpha,
                   s=[20 + 60 * (r["pv_area_m2"] / max_area) for r in hr], edgecolors="k", linewidths=0.4, zorder=3,
                   label="spoke (colour = parent hub, size = array area)" if (k == 0 and lines) else None)
        bad = [r for r in hr if not r["feasible"]]
        if bad:
            ax.scatter([r["lon"] for r in bad], [r["lat"] for r in bad], facecolors="none", edgecolors="red", s=90, linewidths=1.0, zorder=4)
    ax.scatter([h["lon"] for h in hubs], [h["lat"] for h in hubs], marker="*", s=260,
               c=[HUB_COLORS[k % 10] for k in range(len(hubs))], edgecolors="k", zorder=5, label="hub (ice / H2)" if lines else None)
    for k, h in enumerate(hubs):
        ax.annotate(f"H{h['id']}", (h["lon"], h["lat"]), xytext=(6, 6), textcoords="offset points", color="w", fontsize=9)


def plot_overlay(hubs, scenarios, a):
    mola = load_mola()
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(mola / 1000, extent=[0, 360, -90, 90], cmap="gray", origin="upper", aspect="auto", alpha=0.85)
    s1, s2 = scenarios[0], scenarios[1]
    draw_network(ax, hubs, s1["rows"], s1["max_area"], marker="o", alpha=0.9, lines=False)
    draw_network(ax, hubs, s2["rows"], s2["max_area"], marker="s", alpha=0.55, lines=True)
    ax.scatter([], [], marker="o", c="w", edgecolors="k", label="Scenario 1 (circles): self-sufficient")
    ax.scatter([], [], marker="s", c="w", edgecolors="k", alpha=0.6, label="Scenario 2 (squares, dashed pipelines): hub-supported")
    h, l = ax.get_legend_handles_labels()
    keep = [i for i, t in enumerate(l) if t.startswith("Scenario") or t.startswith("hub")]
    ax.legend([h[i] for i in keep], [l[i] for i in keep], loc="lower left", fontsize=8)
    ax.text(0.01, 0.99, "spoke colour = parent hub, marker size = PV array area, red ring = infeasible", transform=ax.transAxes, va="top", fontsize=8, color="w")
    ax.set_xlabel("longitude (°E)"); ax.set_ylabel("latitude"); ax.set_xlim(0, 360); ax.set_ylim(-90, 90)
    ax.set_title("Spoke layouts, both scenarios (colour = parent hub, size = array area)", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(ROOT, "outputs", "scenario_overlay.png"), dpi=130); plt.close(fig)
    print("  wrote outputs/scenario_overlay.png")


def plot(hubs, grids, post, rows, X_sites, y, a, sc, out):
    mola = load_mola()
    # --- global map: colour = parent hub, star = hub, circle = spoke, dashed = pipeline ---
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(mola / 1000, extent=[0, 360, -90, 90], cmap="gray", origin="upper", aspect="auto", alpha=0.85)
    draw_network(ax, hubs, rows, sc["max_area"])
    ax.set_xlabel("longitude (°E)"); ax.set_ylabel("latitude"); ax.set_xlim(0, 360); ax.set_ylim(-90, 90)
    ax.set_title(f"{sc['label']}\n{a.spokes} spokes/hub within {a.range_km:.0f} km, arrays sized per site (marker size); "
                 f"GP surrogate of the storm Monte Carlo energy model, {len(y)} expensive evaluations", fontsize=11)
    ax.legend(loc="lower left", fontsize=8, ncol=3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "spoke_map.png"), dpi=130); plt.close(fig)

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
    fig.suptitle(sc["label"] + " -- per-hub GP posterior (x = expensive evaluations, white line = LCB 400 kWh/sol boundary, circles = chosen spokes, red = infeasible)", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(out, "hub_panels.png"), dpi=110); plt.close(fig)
    print(f"  wrote {os.path.relpath(out, ROOT)}/spokes.csv, spoke_map.png, hub_panels.png")


if __name__ == "__main__":
    main()
