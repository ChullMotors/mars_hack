"""MOLA MEGDR 4 pixel/degree global elevation (data/megt90n000cb.img, 2 MB, PDS)."""
import os
import numpy as np

_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "megt90n000cb.img")
_GRID = None


def _load():
    global _GRID
    if _GRID is None:
        if os.path.exists(_PATH):
            _GRID = np.fromfile(_PATH, dtype=">i2").reshape(720, 1440).astype(float)
        else:
            print("[mola] grid missing -> elevation = 0 everywhere")
            _GRID = np.zeros((720, 1440))
    return _GRID


def elevation_m(lat_deg, lon_deg_east):
    """Elevation (m, relative to areoid) at lat/lon (lon in 0-360 E). Vectorised."""
    g = _load()
    lat = np.asarray(lat_deg, float)
    lon = np.mod(np.asarray(lon_deg_east, float), 360.0)
    row = np.clip(((90.0 - lat) * 4.0).astype(int), 0, 719)
    col = np.clip((lon * 4.0).astype(int), 0, 1439)
    return g[row, col]


if __name__ == "__main__":
    print("Olympus Mons (18.65N, 226.2E):", elevation_m(18.65, 226.2), "m")
    print("Hellas (-42, 70E):", elevation_m(-42, 70), "m")
