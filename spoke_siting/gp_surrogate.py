"""Gaussian-process surrogate of the expensive per-site capacity function."""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

from .geo import haversine_km
from .mola import elevation_m
from .site_energy import capacity_kwh_per_sol

ELEV_MIN_M, ELEV_MAX_M = -5000.0, 5000.0   # elevation safety cutoff (MOLA)


def features(lat, lon, elev):
    """GP inputs: lat (deg), lon (deg), elevation (km). lon is kept so a dust/
    thermal-inertia-aware model can vary along it; ARD lengthscales learn its relevance."""
    return np.column_stack([np.asarray(lat, float), np.asarray(lon, float), np.asarray(elev, float) / 1000.0])


def sample_disk(hub, n, radius_km, rng):
    """n points ~uniform in the disk of radius_km around hub (rejection on elevation)."""
    out = []
    while len(out) < n:
        r = radius_km * np.sqrt(rng.random())
        th = rng.random() * 2 * np.pi
        dlat = (r * np.cos(th)) / 59.15                      # km per degree on Mars
        dlon = (r * np.sin(th)) / (59.15 * np.cos(np.deg2rad(hub["lat"])))
        lat, lon = hub["lat"] + dlat, np.mod(hub["lon"] + dlon, 360)
        e = float(elevation_m(lat, lon))
        if ELEV_MIN_M <= e <= ELEV_MAX_M:
            out.append((lat, lon, e))
    return np.array(out)


def evaluate_sites(sites, seed=0, verbose=True):
    y = np.empty(len(sites))
    for i, (lat, lon, e) in enumerate(sites):
        y[i] = capacity_kwh_per_sol(lat, lon, e, seed=seed)
        if verbose and i % 10 == 0:
            print(f"  [expensive eval] {i+1}/{len(sites)}  lat={lat:6.2f} lon={lon:7.2f} elev={e:7.0f} m -> {y[i]:6.1f} kWh/sol")
    return y


def fit_gp(X, y):
    # y is normalised inside the GP, so amplitude/noise are in unit-variance terms
    kernel = (ConstantKernel(1.0, (1e-2, 1e2))
              * Matern(length_scale=[10.0, 60.0, 3.0], length_scale_bounds=(0.5, 1e4), nu=2.5)
              + WhiteKernel(1e-2, (1e-6, 1.0)))
    gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=3, random_state=0)
    gp.fit(X, y)
    return gp


def candidate_grid(hub, radius_km, step_deg=0.5):
    dl = radius_km / 59.15 + step_deg
    lats = np.arange(hub["lat"] - dl, hub["lat"] + dl, step_deg)
    lons = np.arange(hub["lon"] - dl / np.cos(np.deg2rad(hub["lat"])), hub["lon"] + dl / np.cos(np.deg2rad(hub["lat"])), step_deg)
    LAT, LON = np.meshgrid(lats, lons, indexing="ij")
    lat, lon = LAT.ravel(), np.mod(LON.ravel(), 360)
    d = haversine_km(lat, lon, hub["lat"], hub["lon"])
    keep = d <= radius_km
    lat, lon, d = lat[keep], lon[keep], d[keep]
    elev = elevation_m(lat, lon)
    return dict(lat=lat, lon=lon, elev=elev, dist_km=d, hub_id=np.full(lat.shape, hub["id"]))
