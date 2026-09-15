"""
Per-site Martian insolation profile -- STAND-IN for the team's solar model.
Same physics family (zenith geometry -> Beer-Lambert -> integrate over Ls),
fixed dust opacity tau = 0.5 (stated assumption). Swap `daily_insolation_profile`
for the team's function when it lands; everything downstream only needs a
(668,) array of kWh/m^2/sol on a horizontal surface.
"""
import numpy as np

SOLS_PER_YEAR = 668
YEAR_SOLS = 668.6                # CITED: Mars tropical year in sols
SOL_HOURS = 24.6597              # CITED
OBLIQUITY_DEG = 25.19            # CITED
ECCENTRICITY = 0.0934            # CITED
LS_PERIHELION_DEG = 251.0        # CITED: perihelion near Ls = 251 (southern summer)
SEMI_MAJOR_AU = 1.5237           # CITED
SOLAR_CONST_1AU = 1361.0         # CITED (W/m^2)
SCALE_HEIGHT_M = 11100.0         # CITED: Mars atmospheric scale height ~11.1 km
DIFFUSE_FRACTION = 0.5           # ASSUMPTION: half the scattered beam reaches the ground
SUB_STEPS = 48


def _mean_anomaly_at_ls0():
    """Mean anomaly at Ls = 0 (northern spring equinox), so sol 0 = Ls 0 -- same
    convention as thermal_site.py."""
    e = ECCENTRICITY
    nu = np.deg2rad(-LS_PERIHELION_DEG)             # true anomaly at Ls = 0
    E = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(nu / 2))
    return E - e * np.sin(E)


M_LS0 = _mean_anomaly_at_ls0()


def orbit_from_sol(sol):
    """Return (Ls_rad, r_AU) for sol index (0 = Ls 0, northern spring equinox)."""
    M = M_LS0 + 2 * np.pi * (np.asarray(sol, float) / YEAR_SOLS)
    e = ECCENTRICITY
    nu = M + 2 * e * np.sin(M) + 1.25 * e**2 * np.sin(2 * M)  # equation of centre
    Ls = nu + np.deg2rad(LS_PERIHELION_DEG)
    r = SEMI_MAJOR_AU * (1 - e**2) / (1 + e * np.cos(nu))
    return Ls, r


def daily_insolation_profile(lat_deg, elev_m=0.0, tau=0.5, n_sols=SOLS_PER_YEAR):
    """(n_sols,) array of ground-level insolation on a horizontal surface, kWh/m^2/sol."""
    sols = np.arange(n_sols)
    Ls, r = orbit_from_sol(sols)
    dec = np.arcsin(np.sin(np.deg2rad(OBLIQUITY_DEG)) * np.sin(Ls))       # (n,)
    S = SOLAR_CONST_1AU / r**2                                             # (n,)
    phi = np.deg2rad(lat_deg)
    h = np.linspace(-np.pi, np.pi, SUB_STEPS, endpoint=False) + np.pi / SUB_STEPS
    cosz = np.sin(phi) * np.sin(dec)[:, None] + np.cos(phi) * np.cos(dec)[:, None] * np.cos(h)[None, :]
    cosz = np.clip(cosz, 0.0, None)
    tau_eff = tau * np.exp(-elev_m / SCALE_HEIGHT_M)  # thinner air at altitude -> less extinction
    with np.errstate(divide="ignore", invalid="ignore"):
        trans = np.where(cosz > 1e-3, np.exp(-tau_eff / np.maximum(cosz, 1e-3)), 0.0)
    direct = S[:, None] * cosz * trans
    diffuse = DIFFUSE_FRACTION * S[:, None] * cosz * (1 - trans)
    flux = direct + diffuse                                                # W/m^2
    dt_h = SOL_HOURS / SUB_STEPS
    return flux.sum(axis=1) * dt_h / 1000.0                                # kWh/m^2/sol


if __name__ == "__main__":
    for lat in (0, 30, 45, 55, -35, -48):
        p = daily_insolation_profile(lat)
        print(f"lat {lat:4d}: mean {p.mean():.2f}  min {p.min():.2f}  max {p.max():.2f} kWh/m2/sol")
