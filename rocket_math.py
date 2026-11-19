"""Lightweight, vectorized launch-vehicle performance and trajectory models.

The integrator is an educational engineering approximation, not a certified
flight-dynamics or vehicle-design tool.
"""

from __future__ import annotations

from typing import Any

import numpy as np

G0 = 9.80665
R_E = 6_371_000.0
MU_E = 3.986004418e14
OMEGA_E = 7.2921159e-5
AIR_GAS_CONSTANT = 287.05287

PROPELLANTS_DB: dict[str, dict[str, float]] = {
    "RP-1 / LOX": {"isp_sl": 282.0, "isp_vac": 348.0, "density": 1030.0, "cost": 0.34},
    "CH4 / LOX": {"isp_sl": 330.0, "isp_vac": 380.0, "density": 820.0, "cost": 0.48},
    "LH2 / LOX": {"isp_sl": 366.0, "isp_vac": 452.0, "density": 360.0, "cost": 1.35},
    "HTPB Solid Propellant": {"isp_sl": 245.0, "isp_vac": 268.0, "density": 1750.0, "cost": 0.22},
}

MATERIALS_DB: dict[str, dict[str, float]] = {
    "Al-Li 2195 Aerospace Alloy": {"dry_fraction": 0.085, "max_q_kpa": 75.0, "heat_limit_w_cm2": 18.0},
    "304L Cryogenic Stainless Steel": {"dry_fraction": 0.105, "max_q_kpa": 105.0, "heat_limit_w_cm2": 24.0},
    "Carbon-Epoxy Filament Composite": {"dry_fraction": 0.065, "max_q_kpa": 58.0, "heat_limit_w_cm2": 12.0},
    "PICA-X Ablative Thermal Shield": {"dry_fraction": 0.14, "max_q_kpa": 125.0, "heat_limit_w_cm2": 120.0},
}

NOSECONES_DB: dict[str, dict[str, Any]] = {
    "Tangent Ogive (Low Drag)": {"wave_drag": 0.82, "fineness_ratio": 3.5, "shape": "ogive"},
    "Parabolic Sears-Haack": {"wave_drag": 0.72, "fineness_ratio": 4.0, "shape": "sears-haack"},
    "Conical 15-Deg": {"wave_drag": 1.18, "fineness_ratio": 3.0, "shape": "cone"},
    "Blunt Reentry Capsule": {"wave_drag": 1.48, "fineness_ratio": 1.6, "shape": "capsule"},
}

DESTINATIONS_DB: dict[str, dict[str, float]] = {
    "LEO (400 km ISS)": {"altitude_km": 400.0, "required_dv": 9400.0, "orbit_speed": 7670.0},
    "Sun-Synchronous (700 km)": {"altitude_km": 700.0, "required_dv": 9700.0, "orbit_speed": 7500.0},
    "GTO (35,786 km)": {"altitude_km": 35786.0, "required_dv": 11800.0, "orbit_speed": 10200.0},
    "Lunar Transfer (Moon)": {"altitude_km": 384400.0, "required_dv": 12500.0, "orbit_speed": 10900.0},
    "Mars Transfer": {"altitude_km": 225000000.0, "required_dv": 13500.0, "orbit_speed": 11000.0},
}

ATMOSPHERE_LAYERS = np.array([0.0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0, 86000.0, 120000.0])
ATMOSPHERE_LAPSE = np.array([-0.0065, 0.0, 0.0010, 0.0028, 0.0, -0.0028, -0.0020, 0.0])


def isa_atmosphere_extended(altitude_m: Any, velocity_m_s: Any = 0.0) -> dict[str, np.ndarray]:
    """Return ISA-like density, pressure, temperature, sound speed and Mach to 120 km."""
    h = np.atleast_1d(np.clip(np.asarray(altitude_m, dtype=float), 0.0, 120000.0))
    v = np.broadcast_to(np.asarray(velocity_m_s, dtype=float), h.shape)
    base_t = np.empty(len(ATMOSPHERE_LAPSE) + 1)
    base_p = np.empty_like(base_t)
    base_t[0], base_p[0] = 288.15, 101325.0
    for i, lapse in enumerate(ATMOSPHERE_LAPSE):
        dh = ATMOSPHERE_LAYERS[i + 1] - ATMOSPHERE_LAYERS[i]
        base_t[i + 1] = base_t[i] + lapse * dh
        if abs(lapse) < 1e-12:
            base_p[i + 1] = base_p[i] * np.exp(-G0 * dh / (AIR_GAS_CONSTANT * base_t[i]))
        else:
            base_p[i + 1] = base_p[i] * (base_t[i + 1] / base_t[i]) ** (-G0 / (AIR_GAS_CONSTANT * lapse))

    layer = np.clip(np.searchsorted(ATMOSPHERE_LAYERS, h, side="right") - 1, 0, len(ATMOSPHERE_LAPSE) - 1)
    lapse = ATMOSPHERE_LAPSE[layer]
    dh = h - ATMOSPHERE_LAYERS[layer]
    temperature = base_t[layer] + lapse * dh
    pressure = np.empty_like(h)
    isothermal = np.abs(lapse) < 1e-12
    pressure[isothermal] = base_p[layer[isothermal]] * np.exp(
        -G0 * dh[isothermal] / (AIR_GAS_CONSTANT * base_t[layer[isothermal]])
    )
    gradient = ~isothermal
    pressure[gradient] = base_p[layer[gradient]] * (
        temperature[gradient] / base_t[layer[gradient]]
    ) ** (-G0 / (AIR_GAS_CONSTANT * lapse[gradient]))
    density = pressure / (AIR_GAS_CONSTANT * temperature)
    sound_speed = np.sqrt(1.4 * AIR_GAS_CONSTANT * temperature)
    mach = np.abs(v) / sound_speed
    return {
        "density": density,
        "pressure": pressure,
        "temperature": temperature,
        "speed_of_sound": sound_speed,
        "mach": mach,
    }


def aerodynamic_coefficients(
    mach: Any,
    nosecone: str = "Tangent Ogive (Low Drag)",
    wave_drag_multiplier: float | None = None,
) -> np.ndarray:
    """Approximate drag coefficient, including a transonic wave-drag rise."""
    m = np.asarray(mach, dtype=float)
    shape_factor = float(wave_drag_multiplier) if wave_drag_multiplier is not None else NOSECONES_DB.get(nosecone, NOSECONES_DB["Tangent Ogive (Low Drag)"])["wave_drag"]
    transonic_rise = 0.34 * shape_factor * np.exp(-0.5 * ((m - 1.05) / 0.19) ** 2)
    supersonic_rise = 0.08 * shape_factor * np.clip(m - 1.2, 0.0, 3.0) / 3.0
    return np.clip(0.22 + transonic_rise + supersonic_rise, 0.12, 1.2)


def _orbit_elements(position: np.ndarray, velocity: np.ndarray) -> tuple[float, float, float]:
    radius = float(np.linalg.norm(position))
    speed2 = float(np.dot(velocity, velocity))
    energy = speed2 / 2.0 - MU_E / radius
    angular_momentum = float(position[0] * velocity[1] - position[1] * velocity[0])
    eccentricity2 = max(0.0, 1.0 + 2.0 * energy * angular_momentum**2 / MU_E**2)
    eccentricity = np.sqrt(eccentricity2)
    semi_major = -MU_E / (2.0 * energy) if abs(energy) > 1e-9 else np.inf
    if energy >= 0.0:
        return max(0.0, radius - R_E), float("inf"), energy
    perigee = semi_major * (1.0 - eccentricity) - R_E
    apogee = semi_major * (1.0 + eccentricity) - R_E
    return max(-R_E, perigee), max(-R_E, apogee), energy


def simulate_launch(
    rocket: dict[str, Any] | None = None,
    *,
    latitude_deg: float = 28.5,
    destination: str = "LEO (400 km ISS)",
    dt: float = 0.5,
    max_time_s: float = 1600.0,
    **overrides: Any,
) -> dict[str, Any]:
    """Integrate a simplified two-stage ascent in a rotating, spherical-Earth frame."""
    cfg: dict[str, Any] = {
        "payload_kg": 10000.0, "stage1_propellant_t": 400.0, "stage1_thrust_kn": 7600.0,
        "stage2_propellant_t": 100.0, "fairing_diameter_m": 5.2, "gravity_turn_deg": 8.0,
        "propellant": "RP-1 / LOX", "material": "Al-Li 2195 Aerospace Alloy",
        "nosecone": "Tangent Ogive (Low Drag)", "recovery": True,
    }
    if rocket:
        cfg.update(rocket)
    cfg.update(overrides)
    prop = PROPELLANTS_DB.get(cfg["propellant"], PROPELLANTS_DB["RP-1 / LOX"])
    material = MATERIALS_DB.get(cfg["material"], MATERIALS_DB["Al-Li 2195 Aerospace Alloy"])
    prop1 = max(0.1, float(cfg["stage1_propellant_t"]) * 1000.0)
    prop2 = max(0.1, float(cfg["stage2_propellant_t"]) * 1000.0)
    reserve = 0.13 if cfg["recovery"] else 0.0
    usable1 = prop1 * (1.0 - reserve)
    payload = max(1.0, float(cfg["payload_kg"]))
    diameter = max(0.5, float(cfg["fairing_diameter_m"]))
    area = np.pi * diameter**2 / 4.0
    nose = str(cfg["nosecone"])
    wave_drag_multiplier = cfg.get("wave_drag_multiplier")
    dry1 = prop1 * (material["dry_fraction"] * 0.65 + 0.025)
    dry2 = prop2 * (material["dry_fraction"] * 0.45 + 0.02)
    fairing_mass = 180.0 + 85.0 * diameter
    mass = prop1 + prop2 + dry1 + dry2 + payload + fairing_mass
    stage1_dry = dry1
    stage2_dry = dry2
    thrust1 = max(100.0, float(cfg["stage1_thrust_kn"]) * 1000.0)
    thrust2 = max(50.0, thrust1 * 0.13)
    flow1 = thrust1 / (prop["isp_sl"] * G0)
    flow2 = thrust2 / (prop["isp_vac"] * G0)
    lat = np.deg2rad(float(latitude_deg))
    rotation_speed = OMEGA_E * R_E * np.cos(lat)
    position = np.array([0.0, R_E])
    velocity = np.array([rotation_speed, 0.0])
    time_s = 0.0
    prop1_used = 0.0
    prop2_used = 0.0
    stage = 1
    stage1_burnout_time = np.nan
    fairing_jettison_time = np.nan
    separated = False
    fairing_jettisoned = False
    pitch_kick = np.deg2rad(float(cfg["gravity_turn_deg"]))
    stage1_terminal_pitch = float(cfg.get("stage1_terminal_pitch_deg", 35.0))
    stage1_turn_duration = max(1.0, float(cfg.get("stage1_turn_duration_s", 124.0)))
    stage2_start_pitch = float(cfg.get("stage2_start_pitch_deg", 60.0))
    stage2_end_pitch = float(cfg.get("stage2_end_pitch_deg", 5.0))
    stage2_pitch_decay = max(1.0, float(cfg.get("stage2_pitch_decay_s", 65.0)))
    history: dict[str, list[float]] = {k: [] for k in ("time", "altitude", "velocity", "mach", "q", "g_load", "heat", "mass", "stage", "x", "y")}
    max_steps = int(max_time_s / dt) + 1
    max_q = peak_g = peak_heat = 0.0
    peak_q_time = peak_q_alt = 0.0
    max_q_limit_exceeded = False
    previous_altitude: float | None = None
    ascent_ended = False

    for step in range(max_steps):
        altitude = max(0.0, float(np.linalg.norm(position) - R_E))
        if previous_altitude is not None and altitude < previous_altitude - 1.0:
            ascent_ended = True
        previous_altitude = altitude
        speed = float(np.linalg.norm(velocity))
        atm = isa_atmosphere_extended(np.array([altitude]), np.array([speed]))
        density = float(atm["density"][0])
        relative_velocity = velocity - np.array([rotation_speed * max(0.0, 1.0 - altitude / 120000.0), 0.0])
        relative_speed = float(np.linalg.norm(relative_velocity))
        mach = relative_speed / float(atm["speed_of_sound"][0])
        q_pa = 0.5 * density * relative_speed**2
        cd = float(aerodynamic_coefficients(np.array([mach]), nose, wave_drag_multiplier)[0])
        drag = 0.5 * density * relative_speed**2 * cd * area
        heat_w_cm2 = 1.83e-8 * np.sqrt(max(density, 0.0) / max(diameter * 0.08, 0.15)) * relative_speed**3
        acceleration_load = thrust1 / max(mass, 1.0) / G0 if stage == 1 else thrust2 / max(mass, 1.0) / G0
        axial_g = max(0.0, acceleration_load - drag / max(mass * G0, 1.0))
        for key, value in (
            ("time", time_s), ("altitude", altitude), ("velocity", speed), ("mach", mach),
            ("q", q_pa / 1000.0), ("g_load", axial_g), ("heat", heat_w_cm2),
            ("mass", mass), ("stage", float(stage)), ("x", position[0]), ("y", position[1]),
        ):
            history[key].append(float(value))
        if not ascent_ended:
            peak_g = max(peak_g, axial_g)
        peak_heat = max(peak_heat, heat_w_cm2)
        if not ascent_ended and q_pa / 1000.0 > max_q:
            max_q, peak_q_time, peak_q_alt = q_pa / 1000.0, time_s, altitude
        if not ascent_ended and q_pa / 1000.0 > material["max_q_kpa"]:
            max_q_limit_exceeded = True

        if stage == 1 and prop1_used >= usable1:
            if not separated:
                stage1_burnout_time = time_s
                mass -= stage1_dry + max(0.0, prop1 - prop1_used)
                stage = 2
                separated = True
        elif stage == 2 and altitude >= 110000.0 and not fairing_jettisoned:
            mass -= fairing_mass
            fairing_jettisoned = True
            fairing_jettison_time = time_s

        if stage == 1:
            thrust, isp = thrust1, prop["isp_sl"] + (prop["isp_vac"] - prop["isp_sl"]) * min(altitude / 100000.0, 1.0)
            mdot = thrust1 / (isp * G0)
            # Smooth gravity turn after the pitch-over kick.
            turn_fraction = np.clip((time_s - 4.0) / stage1_turn_duration, 0.0, 1.0)
            initial_angle = np.pi / 2.0 - pitch_kick
            terminal_angle = np.deg2rad(stage1_terminal_pitch)
            thrust_angle = initial_angle - (initial_angle - terminal_angle) * turn_fraction
            thrust = thrust1
        elif prop2_used < prop2:
            thrust, isp = thrust2, prop["isp_vac"]
            mdot = flow2
            elapsed_stage2 = max(0.0, time_s - stage1_burnout_time)
            thrust_angle = np.deg2rad(
                stage2_end_pitch + (stage2_start_pitch - stage2_end_pitch) * np.exp(-elapsed_stage2 / stage2_pitch_decay)
            )
        else:
            break
        if stage == 1 and prop1_used >= usable1:
            continue
        local_tangent_angle = np.arctan2(position[1], position[0]) - np.pi / 2.0
        thrust_angle += local_tangent_angle
        gravity = -MU_E * position / max(float(np.linalg.norm(position)) ** 3, 1.0)
        drag_accel = -drag * relative_velocity / max(relative_speed * mass, 1.0)
        thrust_accel = thrust / max(mass, 1.0) * np.array([np.cos(thrust_angle), np.sin(thrust_angle)])
        velocity += (gravity + drag_accel + thrust_accel) * dt
        position += velocity * dt
        if stage == 1:
            burned = min(usable1 - prop1_used, mdot * dt)
            prop1_used += burned
            mass -= burned
        else:
            burned = min(prop2 - prop2_used, mdot * dt)
            prop2_used += burned
            mass -= burned
        time_s += dt
        if altitude <= 0.0 and step > 2:
            break

    for key in history:
        history[key] = np.asarray(history[key], dtype=float)
    final_position = np.array([history["x"][-1], history["y"][-1]]) if len(history["x"]) else position
    final_velocity = np.array([velocity[0], velocity[1]])
    perigee_m, apogee_m, energy = _orbit_elements(final_position, final_velocity)
    destination_data = DESTINATIONS_DB.get(destination, DESTINATIONS_DB["LEO (400 km ISS)"])
    remaining_dv = float(np.linalg.norm(final_velocity) - destination_data["required_dv"])
    achieved_orbit = energy < 0 and perigee_m >= 120000.0 and apogee_m >= destination_data["altitude_km"] * 1000.0
    return {
        **history,
        "apogee_km": apogee_m / 1000.0,
        "perigee_km": perigee_m / 1000.0,
        "specific_energy": energy,
        "remaining_dv_m_s": remaining_dv,
        "max_q_kpa": max_q,
        "max_q_time_s": peak_q_time,
        "max_q_altitude_km": peak_q_alt / 1000.0,
        "peak_g_load": peak_g,
        "peak_heat_flux_w_cm2": peak_heat,
        "liftoff_mass_t": (prop1 + prop2 + dry1 + dry2 + payload + fairing_mass) / 1000.0,
        "rotation_assist_m_s": rotation_speed,
        "stage1_burnout_time_s": float(stage1_burnout_time if np.isfinite(stage1_burnout_time) else 0.0),
        "fairing_jettison_time_s": float(fairing_jettison_time if np.isfinite(fairing_jettison_time) else 0.0),
        "stage1_propellant_used_kg": prop1_used,
        "stage2_propellant_used_kg": prop2_used,
        "recovery_reserve_kg": prop1 * reserve,
        "max_q_limit_exceeded": max_q_limit_exceeded,
        "orbit_achieved": bool(achieved_orbit),
        "destination": destination,
    }


def calculate_mission_economics(
    rocket: dict[str, Any],
    result: dict[str, Any],
    *,
    payload_kg: float | None = None,
) -> dict[str, float]:
    """Estimate propellant, hardware/refurbishment and payload-normalized launch costs."""
    prop = PROPELLANTS_DB.get(rocket.get("propellant", ""), PROPELLANTS_DB["RP-1 / LOX"])
    p1 = float(rocket.get("stage1_propellant_t", 0.0)) * 1000.0
    p2 = float(rocket.get("stage2_propellant_t", 0.0)) * 1000.0
    propellant_cost = (p1 + p2) * prop["cost"]
    hardware_m = 48.0 + 0.065 * (p1 + p2) / 1000.0
    if rocket.get("recovery", False):
        hardware_m *= 0.34
        refurbishment_m = 4.2 + 0.006 * p1 / 1000.0
    else:
        refurbishment_m = 0.0
    mission_m = hardware_m + refurbishment_m + propellant_cost / 1_000_000.0
    payload = max(1.0, float(payload_kg if payload_kg is not None else rocket.get("payload_kg", 1.0)))
    cost_per_kg = mission_m * 1_000_000.0 / payload
    return {
        "propellant_cost_usd": propellant_cost,
        "hardware_cost_musd": hardware_m,
        "refurbishment_cost_musd": refurbishment_m,
        "total_mission_cost_musd": mission_m,
        "cost_per_kg_usd": cost_per_kg,
    }


def compare_against_baseline(current: dict[str, float], baseline: dict[str, float]) -> dict[str, dict[str, float]]:
    """Return signed absolute and percentage deltas for matching telemetry values."""
    keys = ("apogee_km", "perigee_km", "remaining_dv_m_s", "max_q_kpa", "peak_g_load", "peak_heat_flux_w_cm2", "cost_per_kg_usd")
    deltas: dict[str, dict[str, float]] = {}
    for key in keys:
        value = float(current.get(key, 0.0))
        reference = float(baseline.get(key, 0.0))
        delta = value - reference
        deltas[key] = {"absolute": delta, "percent": 100.0 * delta / abs(reference) if reference else 0.0}
    return deltas


def optimize_100_trajectories(
    rocket: dict[str, Any],
    destination: str = "LEO (400 km ISS)",
    latitude_deg: float = 28.5,
) -> list[dict[str, float]]:
    """Vectorized 100-point guidance/staging sweep with an inexpensive performance surrogate."""
    angles = np.linspace(3.0, 15.0, 10)
    throttles = np.linspace(0.88, 1.0, 10)
    angle_grid, throttle_grid = np.meshgrid(angles, throttles, indexing="ij")
    angle_flat, throttle_flat = angle_grid.ravel(), throttle_grid.ravel()
    prop = PROPELLANTS_DB.get(rocket.get("propellant", ""), PROPELLANTS_DB["RP-1 / LOX"])
    p1 = float(rocket["stage1_propellant_t"]) * 1000.0
    p2 = float(rocket["stage2_propellant_t"]) * 1000.0
    reserve = 0.13 if rocket.get("recovery") else 0.0
    payload = max(1.0, float(rocket["payload_kg"]))
    dry1 = p1 * (MATERIALS_DB[rocket["material"]]["dry_fraction"] * 0.65 + 0.025)
    dry2 = p2 * (MATERIALS_DB[rocket["material"]]["dry_fraction"] * 0.45 + 0.02)
    mass_ratio = (p1 * (1.0 - reserve) + p2) / max(payload + dry1 + dry2, 1.0)
    ideal_dv = prop["isp_vac"] * G0 * np.log1p(max(mass_ratio, 0.0))
    destination_dv = DESTINATIONS_DB[destination]["required_dv"]
    rotation = OMEGA_E * R_E * np.cos(np.deg2rad(latitude_deg))
    angle_penalty = 15.0 * (angle_flat - 8.0) ** 2
    throttle_gain = (throttle_flat - 0.94) * 950.0
    margin = ideal_dv - 1850.0 - destination_dv + rotation - angle_penalty + throttle_gain
    max_q = 28.0 + 12.0 * throttle_flat + 0.17 * (8.0 - angle_flat) ** 2
    cost = calculate_mission_economics(rocket, {}, payload_kg=payload)["cost_per_kg_usd"] * (
        1.0 + (1.0 - throttle_flat) * 0.12 + np.abs(angle_flat - 8.0) * 0.003
    )
    # Rank by the requested delta-v/cost trade-off while penalizing excessive Max-Q.
    score = margin / 1000.0 - np.log1p(cost) / 8.0 - np.maximum(max_q - MATERIALS_DB[rocket["material"]]["max_q_kpa"], 0.0) / 30.0
    order = np.argsort(score)[::-1]
    return [
        {
            "rank": float(rank + 1), "pitch_kick_deg": float(angle_flat[i]),
            "throttle": float(throttle_flat[i]), "remaining_dv_m_s": float(margin[i]),
            "max_q_kpa": float(max_q[i]), "cost_per_kg_usd": float(cost[i]),
            "score": float(score[i]),
        }
        for rank, i in enumerate(order)
    ]
