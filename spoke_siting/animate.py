"""Animate the greedy optimiser placing spokes (round-robin over hubs) from a scenario CSV.
Run: .venv/bin/python -m spoke_siting.animate [scenario_dir]  -> outputs/<scenario>/spoke_selection.gif"""
import csv
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from .geo import load_hubs
from .mola import _load as load_mola
from .run import HUB_COLORS, ROOT


def selection_order(rows, hubs):
    """CSV is sorted by hub (stable), so within a hub rows are in pick order. The optimiser
    picked round-robin: pass 0 -> one spoke per hub, pass 1 -> next one, ..."""
    by_hub = {h["id"]: [r for r in rows if r["hub_id"] == h["id"]] for h in hubs}
    order, n = [], max(len(v) for v in by_hub.values())
    for k in range(n):
        for h in hubs:
            if k < len(by_hub[h["id"]]):
                order.append(by_hub[h["id"]][k])
    return order


def main(scen_dir):
    hubs = load_hubs(os.path.join(ROOT, "Hub Locations"))
    with open(os.path.join(scen_dir, "spokes.csv")) as f:
        rows = [{k: (float(v) if k not in ("feasible",) else v == "True") for k, v in r.items()} for r in csv.DictReader(f)]
    for r in rows:
        r["hub_id"] = int(r["hub_id"])
    order = selection_order(rows, hubs)
    max_area = max(r["pv_area_m2"] for r in rows)
    col = {h["id"]: HUB_COLORS[k % 10] for k, h in enumerate(hubs)}

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(load_mola() / 1000, extent=[0, 360, -90, 90], cmap="gray", origin="upper", aspect="auto", alpha=0.85)
    ax.scatter([h["lon"] for h in hubs], [h["lat"] for h in hubs], marker="*", s=220, c=[col[h["id"]] for h in hubs], edgecolors="k", zorder=5)
    for h in hubs:
        ax.annotate(f"H{h['id']}", (h["lon"], h["lat"]), xytext=(5, 5), textcoords="offset points", color="w", fontsize=8)
    ax.set_xlim(0, 360); ax.set_ylim(-90, 90); ax.set_xlabel("longitude (°E)"); ax.set_ylabel("latitude")
    title = ax.set_title("")
    fig.tight_layout()

    def frame(i):
        if i >= len(order):
            return
        r = order[i]; h = next(h for h in hubs if h["id"] == r["hub_id"]); c = col[h["id"]]
        lon = r["lon"] + (360 if r["lon"] - h["lon"] < -180 else -360 if r["lon"] - h["lon"] > 180 else 0)
        for sh in (0, 360, -360):
            ax.plot([h["lon"] + sh, lon + sh], [h["lat"], r["lat"]], ls="--", lw=0.7, color=c, alpha=0.6, zorder=2)
        ax.scatter(r["lon"], r["lat"], marker="o", color=c, s=20 + 60 * r["pv_area_m2"] / max_area, edgecolors="k", linewidths=0.4, zorder=3)
        title.set_text(f"Greedy siting on the GP posterior — pass {i // len(hubs) + 1}, spoke {i + 1}/{len(order)}: "
                       f"H{h['id']} → ({r['lat']:.0f}°, {r['lon']:.0f}°E), {r['dist_km']:.0f} km pipeline, {r['pv_area_m2']:.0f} m² PV")

    anim = FuncAnimation(fig, frame, frames=len(order) + 12, interval=120)
    out = os.path.join(scen_dir, "spoke_selection.gif")
    anim.save(out, writer=PillowWriter(fps=8), dpi=72)
    print("wrote", os.path.relpath(out, ROOT), f"{os.path.getsize(out)/1e6:.1f} MB")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "outputs", "scenario2_hub_supported"))
