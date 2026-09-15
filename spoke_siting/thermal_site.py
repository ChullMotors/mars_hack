"""
thermal_site.py -- 1D transient regolith + habitat thermal model for Mars sites.

Purpose
-------
Returns the heating energy (kWh/sol) a 4-person pressurised habitat needs at a
given Mars site, so the siting optimiser can subtract it from the site's solar
capacity.  This is deliberately the "expensive simulation" that a Gaussian
process surrogate will later stand in for: physically motivated, but simple.

Model
-----
* Mars orbit -> solar declination and Mars-Sun distance vs areocentric
  longitude Ls; instantaneous cos(zenith) through the sol for the latitude.
* Beer-Lambert direct beam through dust of optical depth tau (scaled by the
  surface-pressure ratio for elevation) plus a crude diffuse term.
* Surface energy balance solved by Newton iteration each sub-step:
      (1-A) S_down + IR_down = eps sigma Ts^4 + G_into_regolith
* Regolith column ~3 m deep, geometric grid (1 cm first cell), zero-flux
  bottom.  Conduction is integrated with backward Euler using a precomputed
  dense inverse of the (constant) tridiagonal operator -- one 40x40 matvec
  per step, so a full Mars year at 48 steps/sol runs in well under a second.
* Habitat: U*A*(T_in - T) losses to ambient (walls) and to regolith at 0.5 m
  (floor), minus internal gains; positive demand integrated per sol.

Only numpy (and matplotlib under __main__) are required.

Assumption labels follow household_sizing.py:
  CITED      -- published number
  ASSUMPTION -- design choice / simplification made here
"""
from __future__ import annotations

import math
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Orbit / rotation.  CITED: Allison & McEwen 2000 (Planet. Space Sci. 48),
# NASA Mars fact sheet.
OBLIQUITY_DEG = 25.19
ECCENTRICITY = 0.0934
LS_PERIHELION_DEG = 251.0
SOLS_PER_YEAR = 668.6
SOL_S = 88775.0
SOLAR_CONST_MARS = 590.0   # W/m^2 at 1.52 AU (mean distance). CITED: Appelbaum & Flood 1990
A_MEAN_AU = 1.523679       # semi-major axis, AU. CITED

# Atmosphere
SCALE_HEIGHT_M = 11100.0   # CITED: Mars atmospheric scale height ~11.1 km
DIFFUSE_FRACTION = 0.5     # ASSUMPTION: half of the extinguished beam reaches the
                           # ground diffusely (crude single-scattering proxy for
                           # forward-scattering dust; Pollack et al. 1990 -> ~0.4-0.6)
IR_DOWN_W_M2 = 20.0        # ASSUMPTION: fixed downwelling atmospheric IR at datum
                           # pressure (thin CO2 atmosphere; Savijarvi 1995 gives
                           # ~15-30 W/m^2). Scaled with pressure ratio below.

# Surface / regolith
SIGMA = 5.670374e-8
EMISSIVITY = 0.95          # CITED: TES-derived Mars surface emissivity ~0.9-0.97
ALBEDO_DEFAULT = 0.25      # CITED: typical mid-latitude plains (TES albedo 0.2-0.3)
THERMAL_INERTIA_DEFAULT = 250.0  # J m^-2 K^-1 s^-1/2. CITED: Putzig & Mellon 2007,
                                 # TES-derived, typical mid-latitude plains
RHO_REGOLITH = 1500.0      # kg/m^3. ASSUMPTION (Mellon et al. 2000 use 1000-1700)
CP_REGOLITH = 800.0        # J/kg/K.  ASSUMPTION (typical basaltic regolith ~ 600-900)
T_CO2_FROST = 148.0        # K. CITED: CO2 condensation temperature at ~6 mbar.
                           # Surface T is clamped here: latent heat of frost
                           # deposition prevents further cooling.

# Column grid
N_CELLS = 40
DZ_FIRST_M = 0.01          # 1 cm first cell -- diurnal skin depth is ~3-4 cm
DEPTH_M = 3.0              # ~3 x annual skin depth for I=250

# Habitat.  ASSUMPTIONS (design choices, tied to the team's household model).
N_PEOPLE = 4
FLOOR_AREA_M2 = 150.0      # from the team's household sizing (~37 m^2/person)
WALL_AREA_M2 = 300.0       # envelope exposed to ambient (walls + roof)
U_WALL = 0.15              # W/m^2/K. Well-insulated pressurised shell (MLI /
U_FLOOR = 0.15             #   aerogel class; ~ Passivhaus wall U-value)
T_SETPOINT_K = 293.0       # 20 C interior
GAIN_PER_PERSON_W = 100.0  # CITED: metabolic heat ~100 W/person (ASHRAE, seated)
EQUIPMENT_GAIN_W = 2000.0  # ASSUMPTION: lighting, ECLSS, electronics dissipation
FLOOR_COUPLING_DEPTH_M = 0.5


# ---------------------------------------------------------------------------
# Orbit and insolation
# ---------------------------------------------------------------------------
def _kepler_E(M: np.ndarray, e: float) -> np.ndarray:
    """Solve Kepler's equation E - e sin E = M (vectorised Newton)."""
    E = M.copy()
    for _ in range(8):
        E = E - (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
    return E


def orbit_vs_time(t_sol: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (Ls in radians, heliocentric distance in AU) at times t (sols),
    with t=0 at Ls=0 (northern spring equinox)."""
    e = ECCENTRICITY
    # true anomaly at Ls=0:  Ls = nu + Ls_peri  ->  nu0 = -Ls_peri
    nu0 = math.radians(-LS_PERIHELION_DEG)
    E0 = 2.0 * math.atan(math.sqrt((1 - e) / (1 + e)) * math.tan(nu0 / 2.0))
    M0 = E0 - e * math.sin(E0)
    M = M0 + 2.0 * np.pi * t_sol / SOLS_PER_YEAR
    E = _kepler_E(np.mod(M, 2 * np.pi), e)
    nu = 2.0 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
    r_au = A_MEAN_AU * (1.0 - e * np.cos(E))
    Ls = np.mod(nu + math.radians(LS_PERIHELION_DEG), 2 * np.pi)
    return Ls, r_au


def surface_insolation(lat_deg: float, t_sol: np.ndarray, tau: float,
                       elev_m: float) -> np.ndarray:
    """Global horizontal solar flux at the surface, W/m^2, at each time (sols)."""
    Ls, r_au = orbit_vs_time(t_sol)
    S_toa = SOLAR_CONST_MARS * (A_MEAN_AU / r_au) ** 2
    decl = np.arcsin(math.sin(math.radians(OBLIQUITY_DEG)) * np.sin(Ls))
    phi = math.radians(lat_deg)
    hour_angle = 2.0 * np.pi * (np.mod(t_sol, 1.0) - 0.5)  # 0 at local noon
    cosz = math.sin(phi) * np.sin(decl) + math.cos(phi) * np.cos(decl) * np.cos(hour_angle)
    cosz = np.clip(cosz, 0.0, None)
    # elevation-scaled dust optical depth (column mass ~ surface pressure)
    tau_eff = tau * math.exp(-elev_m / SCALE_HEIGHT_M)
    with np.errstate(divide="ignore", over="ignore"):
        trans = np.where(cosz > 1e-3, np.exp(-tau_eff / np.maximum(cosz, 1e-3)), 0.0)
    direct = S_toa * cosz * trans
    diffuse = S_toa * cosz * (1.0 - trans) * DIFFUSE_FRACTION     # ASSUMPTION
    return np.where(cosz > 0, direct + diffuse, 0.0)


# ---------------------------------------------------------------------------
# Regolith column
# ---------------------------------------------------------------------------
def _build_grid(n: int = N_CELLS, dz0: float = DZ_FIRST_M, depth: float = DEPTH_M):
    """Geometric grid: cell widths dz0 * g**i summing to `depth`."""
    # solve dz0 (g^n - 1)/(g - 1) = depth for g by bisection
    lo, hi = 1.0001, 2.0
    for _ in range(60):
        g = 0.5 * (lo + hi)
        if dz0 * (g ** n - 1) / (g - 1) > depth:
            hi = g
        else:
            lo = g
    dz = dz0 * g ** np.arange(n)
    z_centre = np.cumsum(dz) - 0.5 * dz
    return dz, z_centre


def _build_implicit_operator(dz: np.ndarray, k: float, rho_cp: float, dt: float):
    """Dense inverse of (I - dt*L) for backward-Euler conduction with zero-flux
    boundaries at both ends; the surface flux is applied as an explicit source
    into cell 0.  Matrix is constant, so invert once per site."""
    n = len(dz)
    # conductance between cell i and i+1 per unit area: k / (distance between centres)
    d_centres = 0.5 * (dz[:-1] + dz[1:])
    cond = k / d_centres
    A = np.zeros((n, n))
    for i in range(n):
        c = dt / (rho_cp * dz[i])
        if i > 0:
            A[i, i - 1] -= c * cond[i - 1]
            A[i, i] += c * cond[i - 1]
        if i < n - 1:
            A[i, i + 1] -= c * cond[i]
            A[i, i] += c * cond[i]
    return np.linalg.inv(np.eye(n) + A)


def _surface_temperature(T1: float, S_abs: float, k_over_half_dz: float,
                         ir_down: float, Ts_guess: float) -> float:
    """Newton solve for the zero-heat-capacity skin temperature:
       S_abs + ir_down - eps*sigma*Ts^4 - k/(dz/2) (Ts - T1) = 0."""
    Ts = Ts_guess
    es = EMISSIVITY * SIGMA
    for _ in range(6):
        f = S_abs + ir_down - es * Ts ** 4 - k_over_half_dz * (Ts - T1)
        df = -4.0 * es * Ts ** 3 - k_over_half_dz
        step = f / df
        Ts -= step
        if abs(step) < 1e-3:
            break
    # CO2 frost clamp: latent heat of condensation holds the surface at ~148 K
    return Ts if Ts > T_CO2_FROST else T_CO2_FROST


def simulate_column(lat_deg: float, elev_m: float = 0.0,
                    thermal_inertia: float | None = None,
                    albedo: float | None = None, tau: float = 0.5,
                    n_sols: int = 668, sub_steps_per_sol: int = 48,
                    spinup_sols: int | None = None) -> dict:
    """Run the regolith column.  Returns per-step surface T and per-step
    temperature at the floor coupling depth, for the final `n_sols` sols."""
    I = THERMAL_INERTIA_DEFAULT if thermal_inertia is None else float(thermal_inertia)
    A = ALBEDO_DEFAULT if albedo is None else float(albedo)
    rho_cp = RHO_REGOLITH * CP_REGOLITH
    k = I ** 2 / rho_cp                       # I = sqrt(k rho cp)
    dt = SOL_S / sub_steps_per_sol
    p_ratio = math.exp(-elev_m / SCALE_HEIGHT_M)
    ir_down = IR_DOWN_W_M2 * p_ratio          # ASSUMPTION: IR scales with column mass

    dz, z_c = _build_grid()
    Minv = _build_implicit_operator(dz, k, rho_cp, dt)
    k_half = k / (0.5 * dz[0])
    src_scale = dt / (rho_cp * dz[0])
    i_floor = int(np.argmin(np.abs(z_c - FLOOR_COUPLING_DEPTH_M)))

    # spin-up: one full year by default (annual skin depth ~1 m; 3 m column
    # deep layers relax slowly but matter little for the surface/0.5 m signal)
    if spinup_sols is None:
        spinup_sols = int(round(SOLS_PER_YEAR))
    total_sols = spinup_sols + n_sols
    n_steps = total_sols * sub_steps_per_sol
    t = (np.arange(n_steps) + 0.5) * (1.0 / sub_steps_per_sol)
    S_abs = (1.0 - A) * surface_insolation(lat_deg, t, tau, elev_m)

    # initial column: radiative-equilibrium estimate for the annual mean flux
    S_mean = float(S_abs.mean()) + ir_down
    T0 = max((S_mean / (EMISSIVITY * SIGMA)) ** 0.25, T_CO2_FROST)
    T = np.full(N_CELLS, T0)
    Ts = T0

    n_rec = n_sols * sub_steps_per_sol
    Ts_rec = np.empty(n_rec)
    Tf_rec = np.empty(n_rec)
    rec0 = n_steps - n_rec
    src = np.zeros(N_CELLS)
    S_list = S_abs.tolist()                   # scalar access is faster than np indexing
    for step in range(n_steps):
        T1 = T[0]
        Ts = _surface_temperature(T1, S_list[step], k_half, ir_down, Ts)
        G = k_half * (Ts - T1)                # W/m^2 into regolith
        src[0] = G * src_scale
        T = Minv @ (T + src)
        if step >= rec0:
            j = step - rec0
            Ts_rec[j] = Ts
            Tf_rec[j] = T[i_floor]
    return {"Ts": Ts_rec, "T_floor": Tf_rec, "dt_s": dt,
            "sub_steps_per_sol": sub_steps_per_sol, "n_sols": n_sols,
            "k_W_mK": k, "z_floor_m": float(z_c[i_floor])}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def heating_load_kwh_per_sol(lat_deg: float, elev_m: float = 0.0,
                             thermal_inertia: float | None = None,
                             albedo: float | None = None,
                             tau: float = 0.5,
                             n_sols: int = 668,
                             sub_steps_per_sol: int = 48) -> dict:
    """Heating energy needed by a 4-person habitat at a Mars site.

    Returns dict with keys:
       'mean_kwh_per_sol'  : annual-mean heating load
       'worst_kwh_per_sol' : worst single sol
       'profile'           : np.ndarray shape (n_sols,) heating load per sol (kWh)
       'surface_T_mean'    : annual-mean surface temperature (K)
       'surface_T_min'     : minimum surface temperature (K)
    """
    col = simulate_column(lat_deg, elev_m, thermal_inertia, albedo, tau,
                          n_sols, sub_steps_per_sol)
    Ts, Tf, dt = col["Ts"], col["T_floor"], col["dt_s"]
    # ASSUMPTION: near-surface air tracks the ground, so ambient = surface T.
    # Losses through the envelope and through the floor into the regolith at
    # ~0.5 m; internal gains from crew metabolism and equipment.
    loss_W = (U_WALL * WALL_AREA_M2 * (T_SETPOINT_K - Ts)
              + U_FLOOR * FLOOR_AREA_M2 * (T_SETPOINT_K - Tf))
    gains_W = N_PEOPLE * GAIN_PER_PERSON_W + EQUIPMENT_GAIN_W
    demand_W = np.clip(loss_W - gains_W, 0.0, None)     # cooling ignored
    per_sol_kwh = demand_W.reshape(n_sols, sub_steps_per_sol).sum(axis=1) * dt / 3.6e6
    return {
        "mean_kwh_per_sol": float(per_sol_kwh.mean()),
        "worst_kwh_per_sol": float(per_sol_kwh.max()),
        "profile": per_sol_kwh,
        "surface_T_mean": float(Ts.mean()),
        "surface_T_min": float(Ts.min()),
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import time

    def _sol_stats(col):
        n = col["sub_steps_per_sol"]
        Ts = col["Ts"].reshape(-1, n)
        return Ts.mean(axis=1), Ts.min(axis=1), Ts.max(axis=1)

    print("=== Surface temperature checks (default params, tau=0.5, elev=0) ===")
    cols = {}
    for lat in (0.0, 45.0):
        t0 = time.perf_counter()
        col = simulate_column(lat)
        dt_run = time.perf_counter() - t0
        cols[lat] = col
        smean, smin, smax = _sol_stats(col)
        print(f"lat {lat:5.1f}: annual-mean Ts {col['Ts'].mean():6.1f} K, "
              f"min {col['Ts'].min():6.1f} K, max {col['Ts'].max():6.1f} K, "
              f"mean diurnal range {np.mean(smax - smin):5.1f} K, "
              f"sol-mean range {smean.min():5.1f}-{smean.max():5.1f} K  "
              f"[{dt_run:.2f} s]")
    print(f"(k = {cols[0.0]['k_W_mK']:.4f} W/m/K, floor coupling depth "
          f"{cols[0.0]['z_floor_m']:.2f} m)")

    print("\n=== Heating load, 4-person habitat ===")
    results = {}
    for lat in (0, 30, 45, 55, -35, -48):
        t0 = time.perf_counter()
        r = heating_load_kwh_per_sol(lat)
        dt_run = time.perf_counter() - t0
        results[lat] = r
        print(f"lat {lat:4d}: mean {r['mean_kwh_per_sol']:6.1f} kWh/sol, "
              f"worst {r['worst_kwh_per_sol']:6.1f} kWh/sol, "
              f"Ts mean {r['surface_T_mean']:6.1f} K, Ts min {r['surface_T_min']:6.1f} K  "
              f"[{dt_run:.2f} s/call]")
    r_hi = heating_load_kwh_per_sol(45, elev_m=3000.0)
    print(f"lat 45 @ +3 km: mean {r_hi['mean_kwh_per_sol']:6.1f}, "
          f"worst {r_hi['worst_kwh_per_sol']:6.1f} kWh/sol")

    # --- plot ---------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C = {0.0: "#2a78d6", 45.0: "#eb6834"}        # categorical slots 1, 2
    fig, axes = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True)
    fig.patch.set_facecolor("#fcfcfb")
    sols = np.arange(668)
    ax = axes[0]
    for lat, col in cols.items():
        smean, smin, smax = _sol_stats(col)
        ax.fill_between(sols, smin, smax, color=C[lat], alpha=0.18, linewidth=0)
        ax.plot(sols, smean, color=C[lat], lw=2, label=f"lat {lat:.0f}° (sol mean; band = diurnal min–max)")
    ax.axhline(T_CO2_FROST, color="#52514e", lw=1, ls="--")
    ax.text(5, T_CO2_FROST + 2, "CO$_2$ frost point 148 K", color="#52514e", fontsize=9)
    ax.set_ylabel("Surface temperature (K)")
    ax.set_title("Regolith surface temperature over one Mars year (I=250, A=0.25, τ=0.5)",
                 loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax = axes[1]
    for lat in (0.0, 45.0):
        ax.plot(sols, results[int(lat)]["profile"], color=C[lat], lw=2, label=f"lat {lat:.0f}°")
    ax.set_ylabel("Habitat heating load (kWh/sol)")
    ax.set_xlabel("Sol from northern spring equinox (Ls = 0°)")
    ax.set_title("4-person habitat heating demand", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    for ax in axes:
        ax.set_facecolor("#fcfcfb")
        ax.grid(True, color="#e6e5e0", lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xlim(0, 668)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thermal_validation.png")
    fig.savefig(out, dpi=130)
    print(f"\nSaved {out}")
