import re
import numpy as np

MARS_R_KM = 3389.5  # CITED: Mars mean radius


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dphi = p2 - p1
    dl = np.deg2rad(lon2) - np.deg2rad(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * MARS_R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def load_hubs(path):
    hubs = []
    pat = re.compile(r"Hub (\d+) \[(\w)\]: lat=\s*(-?[\d.]+)\s+lon=\s*(-?[\d.]+)")
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                hubs.append(dict(id=int(m[1]), hemi=m[2], lat=float(m[3]), lon=float(m[4])))
    return hubs
