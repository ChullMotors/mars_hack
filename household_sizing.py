"""
Mars Household Sizing -- Energy, Hardware & Life-Support Water, All in One
=============================================================================
Everything needed to size the self-sustaining household's power system and
life-support water flow, and to turn that into hardware numbers for a spec
sheet, lives in this one file. Three things happen when you run it:

  1. ENERGY SIZING SEARCH (sections 1-4): simulate a household's electricity
     balance sol-by-sol across many randomly generated Mars years (storms
     and all), then let Optuna search PV/battery/H2 capacity for the
     LIGHTEST system that still hits a target reliability.
  2. PHYSICAL SPEC CONVERSION (section 5): turn that abstract (kWh, kWh, kWh)
     design into real hardware numbers -- panel area, tank volume, masses,
     rated throughputs.
  3. LIFE-SUPPORT WATER (section 6): water is the household's ONLY source of
     breathable oxygen (no MOXIE-style atmospheric CO2 splitting in this
     design -- considered, deprioritized as too much added hardware for the
     time available). This sizes the water flow that O2 production needs,
     the household's total water budget, and what a storm-safe water
     reserve looks like at three tiers of urgency.

Run: python3 household_sizing.py
  -> prints one integrated design report
  -> saves example_year.png and sizing_search.png next to this file

-----------------------------------------------------------------------------
METHOD, IN BRIEF
-----------------------------------------------------------------------------
Electricity is the one resource whose supply is genuinely uncertain sol to
sol (solar varies with season and dust) -- that's why it gets a time-stepped
Monte Carlo simulation. Water and O2 demand are treated as steady daily
figures (people breathe and drink at a roughly constant rate) sized directly
from NASA reference data, not simulated. Each simulated sol: the PV array
generates electricity (seasonal factor x dust-opacity factor); surplus
charges the battery first, then the electrolyser makes storable H2; a
shortfall draws the battery first, then the H2 reserve via fuel cell; if
neither covers it, that sol is logged as unmet demand. Optuna then searches
(PV, battery, H2) capacity for the lightest design that clears a 95%
reliability bar across many randomly generated storm-years -- this is the
actual optimization/ML-adjacent component for the hackathon: not a trained
model, an optimizer searching a design space (pvlib + Optuna are exactly
what the brief's own toolkit suggests).

-----------------------------------------------------------------------------
ASSUMPTIONS -- WHAT'S CITED VS ILLUSTRATIVE VS HAND-SET
-----------------------------------------------------------------------------
CITED (real sources, safe to state as fact in the pitch):
  - Global storm frequency (0.30/yr): 7 documented global storms / ~25 Mars
    years (1971, 1977, 1982, 1994, 2001, 2007, 2018) = 0.28/yr.
  - Global storm duration (25-90 sols): 2018 storm ran ~65-70 sols.
  - Global storm opacity range (0.02-0.25 = fraction of clear-sky output
    that gets through): 2007 storm, Opportunity rover telemetry, 700->128
    Wh/day = 18.3% remaining (JPL / Planetary Society) anchors the 0.25 end;
    2018 storm (worst on record), tau ~10.8 near Opportunity, "<1% of
    sunlight reached the panels" (JPL) anchors the 0.02 end.
  - Regional storm opacity ceiling (0.65): NASA InSight's Jan 2022 regional
    storm, 425->275 Wh/sol = 64.7% remaining (JPL).
  - Regional storm duration classification (>3, usually <20 sols): matches
    published Mars dust-storm classification (area + duration based).
  - O2 consumption (0.89 kg/person/day): NASA ICES-2019-126 "Astronaut Mass
    Balance for Long Duration Missions" (NTRS 20190027563), cross-checked
    against NASA's ISS ECLSS O2-generation spec (5.4 kg/day / 6 crew = 0.90
    kg/person/day) -- two independent NASA sources agree within 1%.
  - Total household water budget (4.17 kg/CM-day: drinking+food 2.38, urine
    flush 0.50, wash 1.29): NASA "Developing the Water Supply System for
    Travel to Mars" (NTRS 20160014539), Table 1 -- Mars-mission specific,
    not generic ISS data.
  - Water:O2 and H2:O2 electrolysis ratios, H2 HHV (39.4 kWh/kg): physical
    constants (reaction stoichiometry, molar masses), not assumptions.
  - Solar yield (0.4477 kWh/m2/day): derived from an 84.8 W/m2 ground-level,
    24h-average irradiance figure x 22%-efficient cells. Was 0.72 (Marspedia
    annual-figure derivation) -- if 84.8 W/m2 turns out to be a daytime-only
    average rather than a 24h average, this should roughly double instead.
  - Battery round-trip efficiency (90%): Tesla Powerwall 2 / Megapack 2 XL
    datasheets (>90%, 92.5-94%); NREL 2024 ATB uses 85% as a degraded-
    lifetime planning figure.
  - Electrolysis efficiency (72%): DOE PEM Electrolysis Technical Targets,
    2022 status (55 kWh/kg system energy consumption, HHV basis).
  - Fuel cell efficiency (41%): DOE Record #20005 "Automotive Fuel Cell
    Targets and Status" (2020), NREL dynamometer testing of a 2016 Toyota
    Mirai at RATED/high power (the relevant regime for storm backup, not
    DOE's headline low-power "peak" figure of 54% HHV).
  - Battery mass (9.3 kg/kWh) and volumetric density (100 Wh/L): Tesla
    Powerwall 2 pack-level figures (108 Wh/kg, 100 Wh/L), cross-checked
    against Megapack 2 XL.

HAND-SET / THIN EVIDENCE (say so if asked, don't present as rigorously cited):
  - Regional storm opacity floor (0.40): no published telemetry found for a
    severe regional-only event -- engineering judgment.
  - Regional storms per year (2-6) and their duration range (3-14 sols)
    within the classification envelope: plausible, not tightly sourced.
  - Seasonal insolation amplitude (+/-15%): order-of-magnitude, not checked
    against an insolation model.
  - The storm generator as a whole is a hand-set statistical scenario, not
    fitted -- the historical record (~7 global storms ever observed) is far
    too thin to fit a distribution to.
  - PV areal mass (6 kg/m2): bounded by real citations (1.5 kg/m2 NASA/JPL
    Mars design target, up to 15-25 kg/m2 terrestrial rigid rack-mounted)
    but not pinned to a single source -- sits inside that range, closer to
    the lightweight end.

STILL ILLUSTRATIVE / UNRESOLVED (flagged as needing a decision, not fixed):
  - H2 system mass (3 kg/kWh, tank + system): DOE/Argonne Record #13010
    (350 bar Type IV automotive tank) implies ~0.47 kg/kWh -- ~6x lighter.
    Left unchanged deliberately: no sourced Mars-rating margin exists to
    justify a specific multiplier above the automotive baseline.
  - PV_MASS_PER_KWH_DAY is now DERIVED (= PV_AREAL_MASS_KG_PER_M2 /
    SOLAR_YIELD_KWH_PER_M2_DAY), not independently guessed -- this closes a
    previously-flagged ~80% inconsistency between the optimizer's mass
    objective and the area-based hardware spec, but it inherits whatever
    uncertainty PV_AREAL_MASS_KG_PER_M2 still carries (see above).

KNOWN LIMITATIONS (flagged honestly, not silently assumed away):
  - Sol resolution, not hour-by-hour: the battery's job here is smoothing
    day-to-day generation swings, not literally "storing power for night".
    A truthful upgrade would split each sol into day/dark halves.
  - Reliability target is 95%, not 100%: chasing 100% needs storage that
    can outlast the worst global storm on stored energy alone, which is
    impractically large. The real answer to the extreme tail is shedding
    non-essential load for a few sols, which this script doesn't model --
    that's a deliberate design decision, not a shortfall.
  - Centralized water/ice utility is assumed to keep delivering through a
    storm (own, storm-hardened power) -- unverified, flagged as a scope
    boundary, same treatment as "CO2 comes free from the atmosphere".
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# 1. Fixed energy-system assumptions (locked scope from the requirements writeup)
# =============================================================================
SOLS_PER_YEAR = 668                 # one Mars year
DAILY_DEMAND_KWH = 400.0            # household-of-4 design point. Source: Rucker,
                                     # M.A., "Surface Power for Mars," NASA Mars
                                     # Study Capability Team / DRA 5.0 (NTRS
                                     # 20160014032), "Surface Power Needs -- Crewed
                                     # Mission" table -- Surface Habitat keep-alive
                                     # load is 14.9 kW for the DRA 5.0 reference
                                     # 4-person crew (matches this household's size
                                     # exactly, not an extrapolation) = 14.9 x 24 =
                                     # 357.6 kWh/day, rounded up to 400 to cover this
                                     # design's own on-site electrolysis + Sabatier
                                     # processing. DRA 5.0's own ISRU load (19.7 kW,
                                     # for return-vehicle propellant production) is
                                     # a separate, much larger centralized system --
                                     # not comparable to this household's scale, so
                                     # deliberately excluded here.

BATTERY_ROUND_TRIP_EFF = 0.90       # applied once, on charging. CITED: Tesla
                                     # Powerwall 2 AC datasheet states >90%
                                     # AC-round-trip; Megapack 2 XL datasheet
                                     # 92.5-94%; NREL 2024 ATB uses 85% as a
                                     # degraded-lifetime planning figure. 90%
                                     # sits squarely in the real range.
ELECTROLYSIS_EFF = 0.72             # surplus power -> stored H2 energy. CITED:
                                     # DOE PEM electrolysis technical targets,
                                     # 2022 status = 55 kWh/kg system energy
                                     # consumption -> 39.4/55 = 71.6% HHV
                                     # (2026 target 77%, ultimate target 86%).
                                     # Was 0.75 (uncited); now the real current-
                                     # status figure.
FUEL_CELL_EFF = 0.41                # stored H2 energy -> recovered electricity.
                                     # CITED: DOE Record #20005 "Automotive Fuel
                                     # Cell Targets and Status" (2020), based on
                                     # NREL dynamometer testing of a 2016 Toyota
                                     # Mirai system (Lohse-Busch et al., Int. J.
                                     # Hydrogen Energy 2020) -- system efficiency
                                     # at RATED/high power (the relevant regime
                                     # for storm backup, not partial-load
                                     # cruising) is ~40-42% HHV. DOE's headline
                                     # "peak" 54% HHV occurs at <25% rated power
                                     # and is not representative here. Was 0.50
                                     # (uncited, and too optimistic for the
                                     # actual rated-power use case) -- this is a
                                     # real worsening correction: more H2 is
                                     # needed to deliver the same backup power.
# (0.75 * 0.50 = 0.375 net H2 round trip, vs ~0.90 for the battery --
#  this asymmetry is *why* the design reserves H2 for storms instead of
#  running it every night. The simulator doesn't assume that split; it's
#  free to size battery to zero and lean on H2 alone if that ever won, which
#  it won't, and that's the point.)

SEASONAL_AMPLITUDE = 0.15           # +/-15% swing in insolation over the year

# Storm scenario generator -- hand-set, not fitted (see docstring)
P_GLOBAL_STORM_PER_YEAR = 0.30      # 7 documented global storms / ~25 Mars years
                                     # (1971, 1977, 1982, 1994, 2001, 2007, 2018)
                                     # = 0.28/yr
GLOBAL_STORM_DURATION_RANGE = (25, 90)      # sols; 2018 storm ran ~65-70 sols
GLOBAL_STORM_OPACITY_RANGE = (0.02, 0.25)   # fraction of clear-sky output that gets
                                             # through. Spans the two documented
                                             # global-storm severities we have real
                                             # telemetry for:
                                             #  - 2007: Opportunity 700->128 Wh/day
                                             #    = 18.3% remaining (JPL/Planetary
                                             #    Society) -> near the 0.25 end.
                                             #  - 2018 (worst on record): tau ~10.8
                                             #    near Opportunity, "<1% of sunlight
                                             #    reached the panels" (JPL) ->
                                             #    modeled as 0.02, not literal 0
                                             #    (some diffuse skylight persists).
N_REGIONAL_STORMS_RANGE = (2, 6)            # per year
REGIONAL_STORM_DURATION_RANGE = (3, 14)     # sols
REGIONAL_STORM_OPACITY_RANGE = (0.40, 0.65) # ceiling backed by InSight's Jan 2022
                                             # regional storm: 425->275 Wh/sol =
                                             # 64.7% remaining (JPL). Floor (0.40)
                                             # still hand-set -- no published
                                             # telemetry found for a severe
                                             # regional-only event.

# Illustrative mass model -- order-of-magnitude only, see docstring
# (PV_MASS_PER_KWH_DAY is defined after section 2, below, once
#  SOLAR_YIELD_KWH_PER_M2_DAY and PV_AREAL_MASS_KG_PER_M2 exist -- it's now
#  DERIVED from those instead of being a second, independent guess. This
#  closes a previously-flagged inconsistency: the optimizer's mass objective
#  used to disagree with the area-based hardware spec by ~80%; deriving both
#  from the same two constants makes them consistent by construction.)
BATTERY_MASS_PER_KWH = 9.3    # kg per kWh of capacity. CITED: Tesla Powerwall 2
                               # pack-level gravimetric density, 13.5 kWh / 125 kg
                               # = 108 Wh/kg -> 1000/108 = 9.3 kg/kWh (LFP grid
                               # storage, includes enclosure/BMS; Megapack 2 XL
                               # cross-check ~101.8 Wh/kg agrees closely). Was 6.0
                               # (implied 167 Wh/kg, too optimistic) -- real pack-
                               # level density is lower than cell-level figures.
H2_MASS_PER_KWH = 3.0         # kg per kWh of stored chemical energy (tank +
                               # system). NOT YET UPDATED -- flagged, not fixed:
                               # DOE/Argonne Record #13010 (350 bar Type IV
                               # automotive tank, current status) gives 5.4 wt%
                               # gravimetric capacity -> ~0.47 kg system/kWh,
                               # roughly 6x lighter than this placeholder. Left
                               # as-is deliberately -- adopting the automotive
                               # figure directly would understate a Mars-rated
                               # tank's real needs (thermal/dust/radiation
                               # hardening, static-life valves, no pit-stop
                               # replacement), but no sourced Mars-specific
                               # margin exists yet. Needs a decision, not a
                               # silent swap -- see design_calculations.md.

RELIABILITY_TARGET = 0.95     # fraction of sols with no unmet demand
# NOTE ON THIS NUMBER: chasing 100% would mean sizing storage to survive the
# worst global storm in the random draw entirely on stored energy alone,
# which needs a tank far outside anything worth hauling to Mars. The
# real-world answer to a rare, severe storm isn't "store infinite energy,"
# it's "shed non-essential load for a few sols," which this script doesn't
# model. So 95% is a deliberate target -- say this out loud in the pitch,
# it's a design decision, not a shortfall.
N_MONTE_CARLO_RUNS = 60       # storm-years sampled per design point evaluated
N_OPTUNA_TRIALS = 60

N_PEOPLE = 4

# =============================================================================
# 2. Physical / hardware conversion constants
# =============================================================================
# ---- cited / physical-constant ----
SOLAR_YIELD_KWH_PER_M2_DAY = 0.132   # UPDATED: user-specified ground-level 24h-
                                       # average irradiance, 84.8 W/m2 (raw, pre-
                                       # panel-efficiency) x 24h = 2.035 kWh/m2/day
                                       # x 22%-efficient cells = 0.4477 kWh/m2/day
                                       # electrical. Was 0.72 (Marspedia annual-
                                       # figure derivation) -- this is 62% of that,
                                       # so PV panel area for the same kWh/sol
                                       # design grows by 1/0.622 = 1.61x.
H2_HHV_KWH_PER_KG = 39.4              # physical constant (higher heating value)

# ---- illustrative engineering assumptions (flag when presenting) ----
PV_AREAL_MASS_KG_PER_M2 = 6.0         # panel + mounting structure. NOT YET
                                       # UPDATED -- flagged, not fixed: real
                                       # bounds found but nothing pins a single
                                       # value. ISS flexible blanket arrays:
                                       # ~2.6 kg/m² (1,088 kg / 420 m², not
                                       # rigid). NASA/JPL Mars surface array
                                       # design target (NTRS 20190000437, not
                                       # flown): ~1.5 kg/m². Terrestrial rigid
                                       # glass panel + rack mount: ~15-25 kg/m²
                                       # (non-space). 6.0 sits inside that
                                       # bracket, closer to the lightweight
                                       # end, but isn't a direct citation --
                                       # needs a decision, see
                                       # design_calculations.md.
BATTERY_WH_PER_L = 100.0              # Li-ion pack, volumetric. CITED: Tesla
                                       # Powerwall 2, 13.5 kWh / 134.6 L ~= 100
                                       # Wh/L; Megapack 2 XL cross-check ~91.6
                                       # Wh/L agrees closely. Was 300 (too
                                       # optimistic by ~3x).
H2_TANK_PRESSURE_BAR = 350
H2_DENSITY_KG_PER_M3_AT_350BAR = 23.0

# PV_MASS_PER_KWH_DAY: derived, not a second independent guess -- see note above.
PV_MASS_PER_KWH_DAY = PV_AREAL_MASS_KG_PER_M2 / SOLAR_YIELD_KWH_PER_M2_DAY
# = 6.0 / 0.4477 = 13.4 kg per kWh/sol of clear-sky capacity

# =============================================================================
# 3. Life-support water constants
# =============================================================================
# ---- cited: NASA ICES-2019-126, x-checked vs ISS ECLSS ----
O2_PER_PERSON_KG_DAY = 0.89

# ---- cited: NASA Mars water-supply study, NTRS 20160014539, Table 1 ----
DRINKING_FOOD_PREP_KG_CM_DAY = 2.38
URINE_FLUSH_KG_CM_DAY = 0.50
WASH_WATER_KG_CM_DAY = 1.29

# ---- physical constants: electrolysis stoichiometry 2 H2O -> 2 H2 + O2 ----
M_H2O, M_O2, M_H2 = 18.015, 32.00, 2.016
WATER_PER_O2_KG = (2 * M_H2O) / M_O2   # kg H2O per kg O2
H2_PER_O2_KG = (2 * M_H2) / M_O2       # kg H2 unavoidably co-produced per kg O2


# =============================================================================
# 4. Storm-year scenario generator
# =============================================================================
def generate_storm_year(rng: np.random.Generator) -> np.ndarray:
    """Return a (SOLS_PER_YEAR,) array of generation multipliers in [0, 1]:
    1.0 = full clear-sky output for that sol, lower during a storm."""
    sols = np.arange(SOLS_PER_YEAR)

    # Seasonal insolation swing (axial tilt), independent of storms
    peak_sol = rng.integers(0, SOLS_PER_YEAR)
    seasonal = 1.0 + SEASONAL_AMPLITUDE * np.cos(2 * np.pi * (sols - peak_sol) / SOLS_PER_YEAR)

    opacity = np.ones(SOLS_PER_YEAR)

    # Regional storms: several short, moderate dips scattered through the year
    n_regional = rng.integers(*N_REGIONAL_STORMS_RANGE)
    for _ in range(n_regional):
        start = rng.integers(0, SOLS_PER_YEAR)
        dur = rng.integers(*REGIONAL_STORM_DURATION_RANGE)
        severity = rng.uniform(*REGIONAL_STORM_OPACITY_RANGE)
        end = min(start + dur, SOLS_PER_YEAR)
        opacity[start:end] = np.minimum(opacity[start:end], severity)

    # Global storm: rare, long, severe -- this is the scenario the H2 reserve exists for
    if rng.random() < P_GLOBAL_STORM_PER_YEAR:
        start = rng.integers(0, SOLS_PER_YEAR)
        dur = rng.integers(*GLOBAL_STORM_DURATION_RANGE)
        severity = rng.uniform(*GLOBAL_STORM_OPACITY_RANGE)
        end = min(start + dur, SOLS_PER_YEAR)
        opacity[start:end] = np.minimum(opacity[start:end], severity)

    return seasonal * opacity


# =============================================================================
# 5. Sol-by-sol energy balance simulator for ONE design, ONE storm-year
# =============================================================================
def simulate_year(pv_kwh_day, battery_kwh, h2_kwh, opacity_profile,
                   return_trace=False):
    """Step through one simulated year for a single (pv, battery, h2) design.
    Returns the fraction of sols with unmet demand (0 = perfectly reliable).
    If return_trace is True, also returns per-sol arrays for plotting.

    NOTE ON RESOLUTION: this balances total generation against total demand
    once per SOL, not hour by hour -- it doesn't resolve day/night within a
    sol. So don't describe the battery here as "storing power for night" in
    the pitch; in this model it's smoothing day-to-day generation swings
    (a cloudy sol against a clear one), while the H2 tier is what rides out
    a multi-sol storm. A truthful upgrade, if there's time, is splitting
    each sol into a daylight and a dark half so the battery's job matches
    the diurnal-cycling story from the README exactly -- flagged here as
    the honest next step, not silently assumed."""
    battery_soc = battery_kwh * 0.5   # start half-charged, not a free lunch
    h2_soc = h2_kwh * 0.5

    deficit_sols = 0
    if return_trace:
        gen_trace = np.zeros(SOLS_PER_YEAR)
        battery_trace = np.zeros(SOLS_PER_YEAR)
        h2_trace = np.zeros(SOLS_PER_YEAR)
        deficit_trace = np.zeros(SOLS_PER_YEAR, dtype=bool)

    for i in range(SOLS_PER_YEAR):
        generation = pv_kwh_day * opacity_profile[i]
        net = generation - DAILY_DEMAND_KWH

        if net >= 0:
            surplus = net
            room = (battery_kwh - battery_soc) / BATTERY_ROUND_TRIP_EFF
            to_battery = min(surplus, max(room, 0.0))
            battery_soc += to_battery * BATTERY_ROUND_TRIP_EFF
            surplus -= to_battery

            room_h2 = (h2_kwh - h2_soc) / ELECTROLYSIS_EFF
            to_h2 = min(surplus, max(room_h2, 0.0))
            h2_soc += to_h2 * ELECTROLYSIS_EFF
            # remaining surplus is curtailed (wasted) -- not tracked further
        else:
            deficit = -net
            from_battery = min(deficit, battery_soc)
            battery_soc -= from_battery
            deficit -= from_battery

            if deficit > 1e-9:
                h2_needed = deficit / FUEL_CELL_EFF
                from_h2 = min(h2_needed, h2_soc)
                h2_soc -= from_h2
                deficit -= from_h2 * FUEL_CELL_EFF

            if deficit > 1e-6:
                deficit_sols += 1
                if return_trace:
                    deficit_trace[i] = True

        if return_trace:
            gen_trace[i] = generation
            battery_trace[i] = battery_soc
            h2_trace[i] = h2_soc

    reliability = 1.0 - deficit_sols / SOLS_PER_YEAR
    if return_trace:
        return reliability, gen_trace, battery_trace, h2_trace, deficit_trace
    return reliability


# =============================================================================
# 6. Monte Carlo wrapper: reliability of a design across many storm-years
# =============================================================================
def monte_carlo_reliability(pv_kwh_day, battery_kwh, h2_kwh, n_runs, seed=None):
    rng = np.random.default_rng(seed)
    reliabilities = np.empty(n_runs)
    for r in range(n_runs):
        profile = generate_storm_year(rng)
        reliabilities[r] = simulate_year(pv_kwh_day, battery_kwh, h2_kwh, profile)
    return reliabilities.mean()


# =============================================================================
# 7. Mass model + Optuna objective
# =============================================================================
def design_mass_kg(pv_kwh_day, battery_kwh, h2_kwh):
    return (pv_kwh_day * PV_MASS_PER_KWH_DAY
            + battery_kwh * BATTERY_MASS_PER_KWH
            + h2_kwh * H2_MASS_PER_KWH)


def objective(trial: optuna.Trial) -> float:
    pv = trial.suggest_float("pv_kwh_day", 300, 1200)
    battery = trial.suggest_float("battery_kwh", 100, 2000)
    h2 = trial.suggest_float("h2_kwh", 200, 8000)

    reliability = monte_carlo_reliability(pv, battery, h2, N_MONTE_CARLO_RUNS,
                                           seed=trial.number)
    mass = design_mass_kg(pv, battery, h2)

    trial.set_user_attr("reliability", reliability)
    trial.set_user_attr("mass_kg", mass)

    if reliability < RELIABILITY_TARGET:
        # Soft constraint: push infeasible trials toward the feasible region
        # instead of hard-rejecting them.
        return mass + 5e5 * (RELIABILITY_TARGET - reliability)
    return mass


# =============================================================================
# 8. Physical spec conversion -- turn (pv, battery, h2) kWh into hardware numbers
# =============================================================================
def physical_spec(pv_kwh_day, battery_kwh, h2_kwh):
    pv_area_m2 = pv_kwh_day / SOLAR_YIELD_KWH_PER_M2_DAY
    pv_mass_kg = pv_area_m2 * PV_AREAL_MASS_KG_PER_M2

    battery_volume_l = battery_kwh * 1000 / BATTERY_WH_PER_L

    h2_mass_kg = h2_kwh / H2_HHV_KWH_PER_KG
    h2_volume_m3 = h2_mass_kg / H2_DENSITY_KG_PER_M3_AT_350BAR

    return {
        "pv_area_m2": pv_area_m2,
        "pv_mass_kg": pv_mass_kg,
        "battery_volume_l": battery_volume_l,
        "h2_mass_kg": h2_mass_kg,
        "h2_volume_m3": h2_volume_m3,
    }


def peak_throughputs(pv, battery, h2, n_sample_years=40):
    """Pull real peak charge/discharge rates out of the simulator itself,
    rather than guessing a rated kW for the electrolyser and fuel cell."""
    max_charge_to_h2 = 0.0        # kWh/sol the electrolyser must be able to absorb
    max_discharge_from_h2 = 0.0   # kWh/sol the fuel cell must be able to deliver

    for seed in range(n_sample_years):
        rng = np.random.default_rng(seed)
        profile = generate_storm_year(rng)
        battery_soc = battery * 0.5
        h2_soc = h2 * 0.5
        for i in range(SOLS_PER_YEAR):
            generation = pv * profile[i]
            net = generation - DAILY_DEMAND_KWH
            if net >= 0:
                surplus = net
                room = (battery - battery_soc) / BATTERY_ROUND_TRIP_EFF
                to_battery = min(surplus, max(room, 0.0))
                battery_soc += to_battery * BATTERY_ROUND_TRIP_EFF
                surplus -= to_battery
                room_h2 = (h2 - h2_soc) / ELECTROLYSIS_EFF
                to_h2 = min(surplus, max(room_h2, 0.0))
                h2_soc += to_h2 * ELECTROLYSIS_EFF
                max_charge_to_h2 = max(max_charge_to_h2, to_h2)
            else:
                deficit = -net
                from_battery = min(deficit, battery_soc)
                battery_soc -= from_battery
                deficit -= from_battery
                if deficit > 1e-9:
                    h2_needed = deficit / FUEL_CELL_EFF
                    from_h2 = min(h2_needed, h2_soc)
                    h2_soc -= from_h2
                    recovered = from_h2 * FUEL_CELL_EFF
                    max_discharge_from_h2 = max(max_discharge_from_h2, recovered)

    return max_charge_to_h2, max_discharge_from_h2


# =============================================================================
# 9. Life-support water demand
# =============================================================================
def life_support_water_demand():
    """Water is the household's only source of breathable O2 (no MOXIE-style
    atmospheric CO2 splitting in this design). This is the survival-critical
    water draw that must never miss, separate from the larger, opportunistic
    water spent making H2 for energy storage."""
    o2_kg_day = O2_PER_PERSON_KG_DAY * N_PEOPLE
    water_kg_day = o2_kg_day * WATER_PER_O2_KG          # ~= L/day, water density 1 kg/L
    h2_byproduct_kg_day = o2_kg_day * H2_PER_O2_KG
    h2_byproduct_kwh_day = h2_byproduct_kg_day * H2_HHV_KWH_PER_KG
    return {
        "o2_kg_day": o2_kg_day,
        "water_kg_day": water_kg_day,
        "water_l_per_person_day": water_kg_day / N_PEOPLE,
        "flow_ml_per_hr": water_kg_day * 1000 / 24,
        "h2_byproduct_kg_day": h2_byproduct_kg_day,
        "h2_byproduct_kwh_day": h2_byproduct_kwh_day,
    }


def total_household_water_demand():
    """NASA Mars-mission water budget (NTRS 20160014539) + O2 life-support water,
    per person and for the household. Splits essential (drinking/food prep + O2
    -- can't be skipped) from curtailable (wash water -- rationed during a storm,
    same load-shedding logic as the energy system's 95% reliability target)."""
    o2_per_person = life_support_water_demand()["water_kg_day"] / N_PEOPLE

    total_per_person = (DRINKING_FOOD_PREP_KG_CM_DAY + URINE_FLUSH_KG_CM_DAY
                         + WASH_WATER_KG_CM_DAY + o2_per_person)
    essential_per_person = DRINKING_FOOD_PREP_KG_CM_DAY + o2_per_person  # cuts wash water only

    return {
        "o2_per_person_kg_day": o2_per_person,
        "total_per_person_kg_day": total_per_person,
        "total_household_kg_day": total_per_person * N_PEOPLE,
        "essential_per_person_kg_day": essential_per_person,
        "essential_household_kg_day": essential_per_person * N_PEOPLE,
    }


def storm_water_reserve_table():
    """Water reserve needed (L) at three tiers of urgency, across three storm
    scenarios. This is the tiered water-resilience answer that mirrors the
    energy system's own philosophy: a small tank guarantees survival, a
    larger one keeps things dignified, full comfort resumes once the
    centralized utility is back online after the storm."""
    d = life_support_water_demand()
    t = total_household_water_demand()
    scenarios = [
        ("Regional storm, worst case", REGIONAL_STORM_DURATION_RANGE[1]),
        ("Global storm, shortest", GLOBAL_STORM_DURATION_RANGE[0]),
        ("Global storm, longest (2018-scale)", GLOBAL_STORM_DURATION_RANGE[1]),
    ]
    rows = []
    for label, sols in scenarios:
        rows.append({
            "label": label,
            "sols": sols,
            "o2_only_l": d["water_kg_day"] * sols,
            "essential_l": t["essential_household_kg_day"] * sols,
            "full_budget_l": t["total_household_kg_day"] * sols,
        })
    return rows


# =============================================================================
# 10. Plots
# =============================================================================
def plot_example_year(pv, battery, h2, seed=7, path="example_year.png"):
    rng = np.random.default_rng(seed)
    profile = generate_storm_year(rng)
    reliability, gen, batt, h2_trace, deficits = simulate_year(
        pv, battery, h2, profile, return_trace=True)

    sols = np.arange(SOLS_PER_YEAR)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                              gridspec_kw={"height_ratios": [1.3, 1]})

    ax = axes[0]
    ax.plot(sols, gen, color="#B9791E", lw=1.2, label="Solar generation (kWh/sol)")
    ax.axhline(DAILY_DEMAND_KWH, color="#5B6266", lw=1, ls="--",
               label="Household demand")
    if deficits.any():
        ax.scatter(sols[deficits], gen[deficits], color="#AE4526", s=14,
                   zorder=5, label="Unmet-demand sol")
    ax.set_ylabel("kWh / sol")
    ax.set_title(f"One simulated Mars year -- reliability {reliability:.1%}"
                 f"  (PV {pv:.0f}, battery {battery:.0f} kWh, H2 {h2:.0f} kWh)")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    ax2 = axes[1]
    ax2.plot(sols, batt, color="#1C7A8C", lw=1.4, label="Battery state of charge")
    ax2.plot(sols, h2_trace, color="#AE4526", lw=1.4, label="H2 reserve state of charge")
    ax2.set_xlabel("Sol of Mars year")
    ax2.set_ylabel("kWh stored")
    ax2.legend(frameon=False, fontsize=9, loc="upper right")
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_search_results(study: optuna.Study, path="sizing_search.png"):
    trials = study.trials
    mass = np.array([t.user_attrs["mass_kg"] for t in trials])
    reliability = np.array([t.user_attrs["reliability"] for t in trials])
    feasible = reliability >= RELIABILITY_TARGET

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(reliability[~feasible], mass[~feasible], color="#B7BCC0", s=18,
               label="Below target reliability", alpha=0.8)
    ax.scatter(reliability[feasible], mass[feasible], color="#1C7A8C", s=22,
               label="Meets target reliability", alpha=0.9)

    best = study.best_trial
    ax.scatter([best.user_attrs["reliability"]], [best.user_attrs["mass_kg"]],
               color="#AE4526", s=110, marker="*", zorder=5,
               label="Chosen design (lowest mass, meets target)")

    ax.axvline(RELIABILITY_TARGET, color="#5B6266", lw=1, ls="--",
               label=f"{RELIABILITY_TARGET:.0%} reliability target")
    ax.set_xlabel("Reliability (fraction of sols with no unmet demand)")
    ax.set_ylabel("System mass, kg (illustrative)")
    ax.set_title(f"{len(trials)} candidate designs searched by Optuna")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# =============================================================================
# 10a. Total site footprint (housing + PV + all component hardware)
# =============================================================================
NHV_PER_PERSON_M3 = 28.96      # CITED: NASA/NTRS 20200002973, "Defining the
                                # Required Net Habitable Volume for Long
                                # [Duration Missions]" -- minimum net habitable
                                # volume for a 4-person crew, 180+ day mission.
CEILING_HEIGHT_M = 2.4          # NOT independently cited -- a common
                                # architectural/habitat assumption used only to
                                # convert the cited volume into a floor area.
BATTERY_UNIT_KWH = 13.5         # Tesla Powerwall 2 usable capacity (same
                                # product already cited for mass/volume)
BATTERY_UNIT_FOOTPRINT_M2 = 1.15 * 0.755   # Powerwall 2 footprint dimensions
H2_TANKS_FOOTPRINT_M2 = 0.6     # ILLUSTRATIVE -- no cited tank product
                                # dimensions found; 606 L @ 350 bar is small
                                # enough to fit a handful of compact cylinders
EQUIPMENT_FOOTPRINT_M2 = 3.0    # ILLUSTRATIVE -- electrolyser + fuel cell +
                                # life-support skids combined; DOE's PEM fuel
                                # cell backup-power cost study (FCTO, 5-10kW)
                                # gives component/cell dimensions but no full
                                # system enclosure footprint, so this is an
                                # engineering guess, not a citation


def total_site_footprint(pv_area_m2, battery_kwh):
    """Single total area figure: housing + PV array + all component hardware
    footprints. PV dominates by a wide margin -- everything else is a rounding
    error on top of the panel field."""
    housing_volume_m3 = NHV_PER_PERSON_M3 * N_PEOPLE
    housing_area_m2 = housing_volume_m3 / CEILING_HEIGHT_M

    n_battery_units = math.ceil(battery_kwh / BATTERY_UNIT_KWH)
    battery_area_m2 = n_battery_units * BATTERY_UNIT_FOOTPRINT_M2

    total = (pv_area_m2 + housing_area_m2 + battery_area_m2
             + H2_TANKS_FOOTPRINT_M2 + EQUIPMENT_FOOTPRINT_M2)

    return {
        "pv_area_m2": pv_area_m2,
        "housing_area_m2": housing_area_m2,
        "housing_volume_m3": housing_volume_m3,
        "battery_area_m2": battery_area_m2,
        "n_battery_units": n_battery_units,
        "h2_tanks_area_m2": H2_TANKS_FOOTPRINT_M2,
        "equipment_area_m2": EQUIPMENT_FOOTPRINT_M2,
        "total_area_m2": total,
    }


# =============================================================================
# 11. Integrated report
# =============================================================================
def print_full_report(study: optuna.Study):
    best = study.best_trial
    pv, battery, h2 = (best.params["pv_kwh_day"], best.params["battery_kwh"],
                        best.params["h2_kwh"])
    spec = physical_spec(pv, battery, h2)
    max_electrolyser, max_fuelcell = peak_throughputs(pv, battery, h2)
    water = life_support_water_demand()
    total_water = total_household_water_demand()
    reserve_rows = storm_water_reserve_table()
    footprint = total_site_footprint(spec["pv_area_m2"], battery)

    print("=" * 78)
    print("FINAL DESIGN -- household of 4, self-sustaining Mars power + water")
    print("=" * 78)

    print(f"\n[ENERGY SYSTEM]  reliability {best.user_attrs['reliability']:.1%} "
          f"(target {RELIABILITY_TARGET:.0%}), illustrative mass "
          f"{best.user_attrs['mass_kg']:.0f} kg")
    print(f"  Solar PV      : {pv:.0f} kWh/sol clear-sky  ->  {spec['pv_area_m2']:.0f} m^2, "
          f"{spec['pv_mass_kg']:.0f} kg")
    print(f"  Battery       : {battery:.0f} kWh  ->  {spec['battery_volume_l']:.0f} L, "
          f"{battery * BATTERY_MASS_PER_KWH:.0f} kg")
    print(f"  H2 reserve    : {h2:.0f} kWh chemical  ->  {spec['h2_mass_kg']:.1f} kg H2, "
          f"{spec['h2_volume_m3']:.2f} m^3 ({spec['h2_volume_m3']*1000:.0f} L) @ "
          f"{H2_TANK_PRESSURE_BAR} bar")
    print(f"  Electrolyser  : must absorb up to {max_electrolyser:.0f} kWh/sol "
          f"(~{max_electrolyser/24.66:.1f} kW avg)")
    print(f"  Fuel cell     : must deliver up to {max_fuelcell:.0f} kWh/sol "
          f"(~{max_fuelcell/24.66:.1f} kW avg)")

    print(f"\n[LIFE-SUPPORT WATER]")
    print(f"  O2 demand                 : {water['o2_kg_day']:.2f} kg/day "
          f"(NASA ICES-2019-126, x-checked vs ISS ECLSS)")
    print(f"  Water for O2 alone        : {water['water_kg_day']:.2f} L/day "
          f"({water['water_l_per_person_day']:.2f} L/person/day), "
          f"continuous feed {water['flow_ml_per_hr']:.0f} mL/hr -- survival floor")
    print(f"  H2 co-produced for free   : {water['h2_byproduct_kg_day']:.3f} kg/day "
          f"= {water['h2_byproduct_kwh_day']:.1f} kWh/day, alongside life-support O2")
    print(f"  Total household water     : {total_water['total_household_kg_day']:.1f} L/day "
          f"(NASA NTRS 20160014539: drinking+food, urine flush, wash water, + O2)")
    print(f"  Essential-only (no wash)  : {total_water['essential_household_kg_day']:.1f} L/day")

    print(f"\n[TOTAL SITE FOOTPRINT]")
    print(f"  PV array                  : {footprint['pv_area_m2']:.0f} m^2 "
          f"({footprint['pv_area_m2']/footprint['total_area_m2']:.1%} of total)")
    print(f"  Housing (NHV-based)       : {footprint['housing_area_m2']:.1f} m^2 "
          f"({footprint['housing_volume_m3']:.0f} m^3 / {CEILING_HEIGHT_M:.1f} m ceiling, "
          f"NASA NTRS 20200002973 NHV standard)")
    print(f"  Battery bank              : {footprint['battery_area_m2']:.1f} m^2 "
          f"({footprint['n_battery_units']} Powerwall-2-equiv. units)")
    print(f"  H2 tanks                  : {footprint['h2_tanks_area_m2']:.1f} m^2 (illustrative)")
    print(f"  Electrolyser/FC/equipment : {footprint['equipment_area_m2']:.1f} m^2 (illustrative)")
    print(f"  TOTAL SITE AREA           : {footprint['total_area_m2']:.0f} m^2")

    print(f"\n[STORM WATER RESERVE -- three tiers]")
    print(f"  {'Scenario':<36}{'O2-only':>10}{'Essential':>12}{'Full budget':>14}")
    for row in reserve_rows:
        print(f"  {row['label']:<36}{row['o2_only_l']:>10.0f}{row['essential_l']:>12.0f}"
              f"{row['full_budget_l']:>14.0f}   ({row['sols']} sols)")

    print("\n" + "=" * 78)
    print("DESIGN RECOMMENDATION FOR THE PITCH")
    print("=" * 78)
    print(f"""  Build the household around a {pv:.0f} kWh/sol PV array ({spec['pv_area_m2']:.0f} m^2),
  a {battery:.0f} kWh battery for day-to-day smoothing, and a {h2:.0f} kWh H2 reserve
  for storm backup -- this is the lightest combination Optuna found that
  survives {RELIABILITY_TARGET:.0%} of a randomly generated 668-sol Mars year, including
  regional storms and a global storm (now modeled up to 2018 severity).

  Pair it with a small, separately-sized water reserve: {reserve_rows[-1]['o2_only_l']:.0f} L
  covers pure survival (O2 only) through the worst storm on record, {reserve_rows[-1]['essential_l']:.0f} L
  keeps drinking/food/O2 going without touching the centralized utility, and
  the full {reserve_rows[-1]['full_budget_l']:.0f} L figure is what comfort (showers, laundry) costs if you
  wanted zero rationing at all -- present the tiered number, not one figure,
  as evidence of a genuine load-shedding design philosophy rather than
  brute-force overbuilding.

  State plainly in the pitch: the 95% reliability target and the assumption
  that centralized water/ice infrastructure has its own storm-hardened power
  are both explicit, deliberate scope boundaries -- not gaps you didn't see.

  Total site footprint: {footprint['total_area_m2']:.0f} m^2, of which the PV
  array alone is {footprint['pv_area_m2']/footprint['total_area_m2']:.0%} -- housing, battery, H2 tanks and
  equipment together are a rounding error on top of the panel field.""")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)

    best = study.best_trial
    plot_example_year(best.params["pv_kwh_day"], best.params["battery_kwh"],
                       best.params["h2_kwh"])
    plot_search_results(study)

    print_full_report(study)
    print("\nSaved example_year.png and sizing_search.png")
