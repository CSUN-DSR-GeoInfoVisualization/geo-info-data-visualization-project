"""Dead fuel moisture (100-hour, 1000-hour), NFDRS 1978 (Cohen & Deeming 1985),
computed from our own daily weather: the second source beside gridMET, so the
fuel-moisture inputs keep the two-source rule (Weather App Plan, 2026-11-26).

Daily inputs: max/min temperature (C), max/min relative humidity (%),
precipitation duration (hours), day length (hours). 1000-hour needs a long
spin-up, so callers pass >= 60 days and read the last day.
"""
import math


def emc(t_c, rh):
    """Equilibrium moisture content (%), Simard 1968, as used by NFDRS."""
    t = t_c * 1.8 + 32
    if rh < 10:
        return 0.03229 + 0.281073 * rh - 0.000578 * rh * t
    if rh < 50:
        return 2.22749 + 0.160107 * rh - 0.01478 * t
    return 21.0606 + 0.005565 * rh ** 2 - 0.00035 * rh * t - 0.483199 * rh


def day_length(lat, doy):
    """Hours of daylight (NFDRS uses it to weight day vs night EMC)."""
    dec = 0.41008 * math.sin(math.radians((doy - 82) * 360 / 365))
    x = max(-1.0, min(1.0, -math.tan(math.radians(lat)) * math.tan(dec)))
    return 24 * (1 - math.acos(x) / math.pi)


def dead_fuel_moisture(days, lat):
    """days: chronological [(doy, tmax, tmin, rhmax, rhmin, precip_hours)].
    Returns (mc100, mc1000) for the last day, percent."""
    mc100, mc1000, bnd = 20.0, 30.0, []
    for doy, tmax, tmin, rhmax, rhmin, pdur in days:
        dl = day_length(lat, doy)
        emc_min, emc_max = emc(tmax, rhmin), emc(tmin, rhmax)  # afternoon low, night high
        emcbar = (dl * emc_min + (24 - dl) * emc_max) / 24
        pdur = max(0.0, min(24.0, pdur))
        b = ((24 - pdur) * emcbar + pdur * (0.5 * pdur + 41)) / 24
        mc100 += (b - mc100) * (1 - 0.87 * math.exp(-0.24))
        bnd.append(b)
        bdybar = sum(bnd[-7:]) / len(bnd[-7:])
        mc1000 += (bdybar - mc1000) * (1 - 0.82 * math.exp(-0.168))
    return round(mc100, 2), round(mc1000, 2)
