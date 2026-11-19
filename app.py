"""Dark aerospace mission-control dashboard for the rocket trajectory model."""

from __future__ import annotations

import csv
import io
from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from rocket_math import (
    DESTINATIONS_DB,
    MATERIALS_DB,
    NOSECONES_DB,
    PROPELLANTS_DB,
    R_E,
    calculate_mission_economics,
    compare_against_baseline,
    isa_atmosphere_extended,
    optimize_100_trajectories,
    simulate_launch,
)

SPACE = "#0B0F19"
PANEL = "#111827"
CYAN = "#38BDF8"
AMBER = "#FBBF24"
MUTED = "#94A3B8"

SITES = {
    "Tanegashima Space Center (30.4° N, Japan)": 30.4,
    "Cape Canaveral LC-39A (28.5° N)": 28.5,
    "Kourou Spaceport (5.2° N)": 5.2,
    "Baikonur Cosmodrome (45.9° N)": 45.9,
}

PRESETS: dict[str, dict[str, Any]] = {
    "Custom Vehicle": {},
    "Falcon 9 Block 5 (Reusable LEO)": {
        "payload_kg": 15000, "stage1_propellant_t": 395.0, "stage1_thrust_kn": 7600,
        "stage2_propellant_t": 112.0, "fairing_diameter_m": 5.2, "gravity_turn_deg": 8.0,
        "propellant": "RP-1 / LOX", "material": "Al-Li 2195 Aerospace Alloy",
        "nosecone": "Tangent Ogive (Low Drag)", "recovery": True,
    },
    "JAXA H3-22S (Tanegashima)": {
        "payload_kg": 6000, "stage1_propellant_t": 225.0, "stage1_thrust_kn": 4300,
        "stage2_propellant_t": 30.0, "fairing_diameter_m": 4.0, "gravity_turn_deg": 9.0,
        "propellant": "LH2 / LOX", "material": "Al-Li 2195 Aerospace Alloy",
        "nosecone": "Parabolic Sears-Haack", "recovery": False,
    },
    "Electron (SmallSat)": {
        "payload_kg": 250, "stage1_propellant_t": 9.5, "stage1_thrust_kn": 190,
        "stage2_propellant_t": 2.2, "fairing_diameter_m": 1.2, "gravity_turn_deg": 11.0,
        "propellant": "HTPB Solid Propellant", "material": "Carbon-Epoxy Filament Composite",
        "nosecone": "Conical 15-Deg", "recovery": False,
    },
    "Starship-Class Heavy": {
        "payload_kg": 100000, "stage1_propellant_t": 3400.0, "stage1_thrust_kn": 74000,
        "stage2_propellant_t": 1200.0, "fairing_diameter_m": 9.0, "gravity_turn_deg": 7.0,
        "propellant": "CH4 / LOX", "material": "304L Cryogenic Stainless Steel",
        "nosecone": "Blunt Reentry Capsule", "recovery": True,
    },
}

DEFAULTS = {
    "payload_kg": 10000, "stage1_propellant_t": 400.0, "stage1_thrust_kn": 7600,
    "stage2_propellant_t": 100.0, "fairing_diameter_m": 5.2, "gravity_turn_deg": 8.0,
    "propellant": "RP-1 / LOX", "material": "Al-Li 2195 Aerospace Alloy",
    "nosecone": "Tangent Ogive (Low Drag)", "recovery": True,
}


def _apply_preset() -> None:
    preset = PRESETS.get(st.session_state.get("mission_preset", "Custom Vehicle"), {})
    for key, value in preset.items():
        st.session_state[key] = value


def _metric_value(result: dict[str, Any], economics: dict[str, float]) -> dict[str, float]:
    return {
        "apogee_km": float(result["apogee_km"]),
        "perigee_km": float(result["perigee_km"]),
        "remaining_dv_m_s": float(result["remaining_dv_m_s"]),
        "max_q_kpa": float(result["max_q_kpa"]),
        "peak_g_load": float(result["peak_g_load"]),
        "peak_heat_flux_w_cm2": float(result["peak_heat_flux_w_cm2"]),
        "cost_per_kg_usd": float(economics["cost_per_kg_usd"]),
    }


def _fmt_delta(value: float | None) -> str:
    if value is None:
        return "BASELINE"
    return f"{value:+.1f}% vs baseline"


def _make_figure() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=SPACE, plot_bgcolor=SPACE,
        font={"color": "#F8FAFC", "family": "Inter, sans-serif"},
        margin={"l": 12, "r": 12, "t": 44, "b": 12},
        xaxis={"gridcolor": "#253247", "zerolinecolor": "#334155"},
        yaxis={"gridcolor": "#253247", "zerolinecolor": "#334155"},
    )
    return fig


def _read_profile(uploaded: Any) -> tuple[np.ndarray, float] | None:
    if uploaded is None:
        return None
    try:
        text = uploaded.getvalue().decode("utf-8-sig")
    except UnicodeDecodeError:
        st.sidebar.error("Profile file must be UTF-8 encoded.")
        return None
    points: list[tuple[float, float]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "//")):
            continue
        fields = stripped.replace(",", " ").replace(";", " ").split()
        if len(fields) < 2:
            continue
        try:
            x, y = float(fields[0]), float(fields[1])
        except ValueError:
            continue
        if np.isfinite(x) and np.isfinite(y):
            points.append((x, y))
    if len(points) < 3:
        st.sidebar.error("Provide at least 3 numeric coordinate pairs (x, radius).")
        return None
    profile = np.asarray(points, dtype=float)
    profile = profile[np.argsort(profile[:, 0])]
    length = float(np.ptp(profile[:, 0]))
    radius = float(np.max(np.abs(profile[:, 1])))
    if length <= 0.0 or radius <= 0.0:
        st.sidebar.error("Profile coordinates must have non-zero length and radius.")
        return None
    fineness_ratio = length / (2.0 * radius)
    profile[:, 0] = (profile[:, 0] - profile[0, 0]) / length
    profile[:, 1] = profile[:, 1] / radius
    return profile, fineness_ratio


st.set_page_config(page_title="Astra | Mission Control", page_icon="🚀", layout="wide")

if "pending_optimizer_values" in st.session_state:
    for key, value in st.session_state.pop("pending_optimizer_values").items():
        st.session_state[key] = value
for _key, _value in DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _value
if "mission_preset" not in st.session_state:
    st.session_state["mission_preset"] = "Custom Vehicle"

st.markdown(
    f"""
    <style>
    .stApp {{ background: {SPACE}; }}
    [data-testid="stSidebar"] {{ background: {PANEL}; border-right: 1px solid #253247; }}
    div[data-testid="stMetric"] {{ background: {PANEL}; border: 1px solid #253247;
       padding: 14px 16px; border-radius: 12px; min-height: 108px; }}
    div[data-testid="stMetricLabel"] p {{ color: {MUTED}; font-size: .78rem; }}
    div[data-testid="stMetricValue"] {{ color: #F8FAFC; }}
    .status-chip {{ display:inline-block; padding:7px 12px; border-radius:999px;
       border:1px solid {CYAN}; color:{CYAN}; font-weight:700; letter-spacing:.08em;
       background:#0c2431; }}
    .subtle {{ color:{MUTED}; font-size:.88rem; }}
    .section-label {{ color:{CYAN}; letter-spacing:.13em; font-size:.76rem; font-weight:700; }}
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 🚀 ASTRA")
    st.caption("MISSION CONTROL · FLIGHT DYNAMICS")
    st.selectbox("Mission preset", list(PRESETS), key="mission_preset", on_change=_apply_preset)
    site = st.selectbox("Launch site", list(SITES))
    destination = st.selectbox("Target destination", list(DESTINATIONS_DB))
    st.divider()
    st.markdown("##### VEHICLE CONFIGURATION")
    propellant = st.selectbox("Propellant", list(PROPELLANTS_DB), key="propellant")
    material = st.selectbox("Airframe / heat-shield material", list(MATERIALS_DB), key="material")
    nosecone = st.selectbox("Nosecone profile", list(NOSECONES_DB), key="nosecone")
    profile_file = st.file_uploader("Optional .dat / .csv profile (x, radius)", type=["dat", "csv"])
    recovery = st.toggle("Propulsive booster recovery", key="recovery")
    st.divider()
    st.markdown("##### FLIGHT PARAMETERS")
    payload_kg = st.slider("Payload mass (kg)", 100, 150000, key="payload_kg", step=100)
    stage1_propellant_t = st.slider("Stage 1 propellant (t)", 2.0, 3500.0, key="stage1_propellant_t", step=1.0)
    stage1_thrust_kn = st.slider("Stage 1 thrust (kN)", 100, 80000, key="stage1_thrust_kn", step=100)
    stage2_propellant_t = st.slider("Stage 2 propellant (t)", 1.0, 1300.0, key="stage2_propellant_t", step=1.0)
    fairing_diameter_m = st.slider("Fairing diameter (m)", 1.0, 12.0, key="fairing_diameter_m", step=0.1)
    gravity_turn_deg = st.slider("Gravity-turn pitch kick (deg)", 2.0, 18.0, key="gravity_turn_deg", step=0.5)

rocket = {
    "payload_kg": payload_kg, "stage1_propellant_t": stage1_propellant_t,
    "stage1_thrust_kn": stage1_thrust_kn, "stage2_propellant_t": stage2_propellant_t,
    "fairing_diameter_m": fairing_diameter_m, "gravity_turn_deg": gravity_turn_deg,
    "propellant": propellant, "material": material, "nosecone": nosecone, "recovery": recovery,
}
profile_data = _read_profile(profile_file)
custom_profile = profile_data[0] if profile_data is not None else None
custom_fineness_ratio = profile_data[1] if profile_data is not None else None
if custom_fineness_ratio is not None:
    rocket["wave_drag_multiplier"] = float(
        NOSECONES_DB[nosecone]["wave_drag"] * np.clip(3.5 / custom_fineness_ratio, 0.55, 1.8)
    )

result = simulate_launch(rocket, latitude_deg=SITES[site], destination=destination)
economics = calculate_mission_economics(rocket, result)
metrics = _metric_value(result, economics)
baseline_result = None
if "baseline_rocket" in st.session_state:
    baseline_result = st.session_state["baseline_result"]
else:
    baseline_rocket = dict(DEFAULTS)
    baseline_result = simulate_launch(baseline_rocket, latitude_deg=SITES[site], destination=destination)
    baseline_economics = calculate_mission_economics(baseline_rocket, baseline_result)
    baseline_result = _metric_value(baseline_result, baseline_economics)

if st.sidebar.button("🔒 Lock Current Rocket as Baseline", width="stretch"):
    st.session_state["baseline_rocket"] = dict(rocket)
    st.session_state["baseline_result"] = dict(metrics)
    st.toast("Current vehicle locked as the comparison baseline.")
    st.rerun()

baseline_deltas = compare_against_baseline(metrics, baseline_result) if baseline_result else {}
if result["orbit_achieved"]:
    status_text, status_color = "ORBIT ACHIEVED", CYAN
elif result["specific_energy"] >= 0:
    status_text, status_color = "ESCAPE TRAJECTORY", AMBER
elif result["max_q_limit_exceeded"] or result["peak_heat_flux_w_cm2"] > MATERIALS_DB[material]["heat_limit_w_cm2"]:
    status_text, status_color = "MAX-Q / THERMAL WARNING", AMBER
else:
    status_text, status_color = "SUBORBITAL", AMBER

st.markdown("# Launch Control Center")
st.markdown(
    f'<span class="status-chip" style="border-color:{status_color};color:{status_color}">{status_text}</span>'
    f'&nbsp;&nbsp; <span class="subtle">ROTATION ASSIST&nbsp; {result["rotation_assist_m_s"]:.0f} m/s'
    f'&nbsp;&nbsp; · &nbsp; EST. Δv&nbsp; {result["remaining_dv_m_s"] + DESTINATIONS_DB[destination]["required_dv"]:.0f} m/s'
    f'&nbsp;&nbsp; · &nbsp; PROPELLANT MASS FRACTION&nbsp; '
    f'{(stage1_propellant_t + stage2_propellant_t) * 1000 / (result["liftoff_mass_t"] * 1000):.1%}</span>',
    unsafe_allow_html=True,
)
st.markdown("")
tab_trajectory, tab_optimizer, tab_lab = st.tabs(
    ["01  ·  FLIGHT TRAJECTORY", "02  ·  BATCH OPTIMIZER", "03  ·  NOSECONE & REENTRY LAB"]
)

with tab_trajectory:
    labels = [
        ("APOGEE", f'{result["apogee_km"]:,.0f} km', "apogee_km"),
        ("PERIGEE", f'{result["perigee_km"]:,.0f} km', "perigee_km"),
        ("REMAINING Δv", f'{result["remaining_dv_m_s"]:+,.0f} m/s', "remaining_dv_m_s"),
        ("MAX-Q", f'{result["max_q_kpa"]:,.1f} kPa', "max_q_kpa"),
        ("PEAK G-LOAD", f'{result["peak_g_load"]:,.2f} G', "peak_g_load"),
        ("PEAK HEAT FLUX", f'{result["peak_heat_flux_w_cm2"]:,.1f} W/cm²', "peak_heat_flux_w_cm2"),
        ("LIFTOFF MASS", f'{result["liftoff_mass_t"]:,.1f} t', None),
        ("LAUNCH COST", f'${economics["cost_per_kg_usd"]:,.0f}/kg', "cost_per_kg_usd"),
    ]
    for start in range(0, 8, 4):
        cols = st.columns(4)
        for col, (label, value, metric_key) in zip(cols, labels[start:start + 4]):
            delta = baseline_deltas.get(metric_key, {}).get("percent") if metric_key else None
            col.metric(label, value, _fmt_delta(delta) if delta is not None else "MODEL ESTIMATE", delta_color="off")

    st.markdown("#### Mission playback")
    if len(result["time"]) > 1:
        max_time = float(result["time"][-1])
        scrub_time = st.slider("FLIGHT TIMELINE  ·  T+ seconds", 0.0, max_time, min(max_time, max_time * 0.62), step=1.0, key="timeline_scrubber")
        current_index = int(np.argmin(np.abs(result["time"] - scrub_time)))
        st.caption(
            f'T+ {result["time"][current_index]:.0f} s  ·  ALT {result["altitude"][current_index] / 1000:.1f} km'
            f'  ·  V {result["velocity"][current_index]:,.0f} m/s  ·  MACH {result["mach"][current_index]:.2f}'
            f'  ·  Q {result["q"][current_index]:.1f} kPa'
        )

        fig3d = _make_figure()
        sphere_u = np.linspace(0, 2 * np.pi, 48)
        sphere_v = np.linspace(-np.pi / 2, np.pi / 2, 24)
        sx = (R_E / 1000.0) * np.outer(np.cos(sphere_v), np.cos(sphere_u))
        sy = (R_E / 1000.0) * np.outer(np.cos(sphere_v), np.sin(sphere_u))
        sz = (R_E / 1000.0) * np.outer(np.sin(sphere_v), np.ones_like(sphere_u))
        fig3d.add_trace(go.Surface(x=sx, y=sy, z=sz, colorscale=[[0, "#10273A"], [1, "#17354A"]], opacity=0.72, showscale=False, hoverinfo="skip", name="Earth"))
        for latitude in np.arange(-60, 61, 30):
            longitude = np.linspace(-np.pi, np.pi, 100)
            lat = np.deg2rad(latitude)
            radius = R_E / 1000.0 * 1.002
            fig3d.add_trace(go.Scatter3d(
                x=radius * np.cos(lat) * np.cos(longitude), y=radius * np.cos(lat) * np.sin(longitude),
                z=radius * np.sin(lat) * np.ones_like(longitude), mode="lines",
                line={"color": "#33506A", "width": 1}, showlegend=False, hoverinfo="skip",
            ))
        for longitude_deg in range(0, 360, 30):
            latitudes = np.linspace(-np.pi / 2, np.pi / 2, 80)
            lon = np.deg2rad(longitude_deg)
            radius = R_E / 1000.0 * 1.002
            fig3d.add_trace(go.Scatter3d(
                x=radius * np.cos(latitudes) * np.cos(lon), y=radius * np.cos(latitudes) * np.sin(lon),
                z=radius * np.sin(latitudes), mode="lines", line={"color": "#33506A", "width": 1},
                showlegend=False, hoverinfo="skip",
            ))
        site_lat = np.deg2rad(SITES[site])
        site_lon = np.deg2rad(140.0 if "Tanegashima" in site else (-80.6 if "Cape" in site else (-52.8 if "Kourou" in site else 63.3)))

        def world_track(data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            radius = np.hypot(data["x"], data["y"]) / 1000.0
            heading = np.arctan2(data["x"], np.maximum(data["y"], 1.0)) / max(np.cos(site_lat), 0.1) + site_lon
            latitudes = site_lat + data["altitude"] * 0.0
            return radius * np.cos(latitudes) * np.cos(heading), radius * np.cos(latitudes) * np.sin(heading), radius * np.sin(latitudes)

        if "baseline_rocket" in st.session_state:
            ghost = simulate_launch(st.session_state["baseline_rocket"], latitude_deg=28.5)
        else:
            ghost = simulate_launch(DEFAULTS, latitude_deg=28.5)
        gx, gy, gz = world_track(ghost)
        fig3d.add_trace(go.Scatter3d(x=gx, y=gy, z=gz, mode="lines", name="Baseline", line={"color": "#64748B", "width": 3, "dash": "dash"}, opacity=0.55))
        tx, ty, tz = world_track(result)
        fig3d.add_trace(go.Scatter3d(
            x=tx, y=ty, z=tz, mode="lines+markers", name="Active trajectory",
            line={"color": CYAN, "width": 5},
            marker={"size": 2, "color": result["stage"], "colorscale": [[0, AMBER], [1, CYAN]], "showscale": False},
            customdata=np.column_stack((result["time"], result["altitude"] / 1000, result["velocity"])),
            hovertemplate="T+ %{customdata[0]:.0f}s<br>Altitude %{customdata[1]:.1f} km<br>Speed %{customdata[2]:.0f} m/s<extra></extra>",
        ))
        live_x, live_y, live_z = tx[current_index], ty[current_index], tz[current_index]
        fig3d.add_trace(go.Scatter3d(x=[live_x], y=[live_y], z=[live_z], mode="markers", name="Live vehicle",
                                     marker={"size": 8, "color": "#FFFFFF", "line": {"color": CYAN, "width": 4}}))
        fig3d.add_trace(go.Scatter3d(x=[R_E / 1000 * np.cos(site_lat) * np.cos(site_lon)],
                                     y=[R_E / 1000 * np.cos(site_lat) * np.sin(site_lon)],
                                     z=[R_E / 1000 * np.sin(site_lat)], mode="markers", name="Launch pad",
                                     marker={"size": 5, "color": AMBER, "symbol": "diamond"}))
        for marker_time, marker_name, marker_color in (
            (result["stage1_burnout_time_s"], "MECO / staging", AMBER),
            (result["fairing_jettison_time_s"], "Fairing separation", "#A78BFA"),
        ):
            if marker_time > 0 and len(result["time"]):
                ix = int(np.argmin(np.abs(result["time"] - marker_time)))
                fig3d.add_trace(go.Scatter3d(x=[tx[ix]], y=[ty[ix]], z=[tz[ix]], mode="markers", name=marker_name,
                                             marker={"size": 5, "color": marker_color, "symbol": "diamond"}))
        fig3d.update_layout(
            title="EARTH-CENTERED ASCENT · LIVE VEHICLE TRACK", scene={
                "xaxis": {"title": "X · km", "gridcolor": "#253247", "backgroundcolor": SPACE},
                "yaxis": {"title": "Y · km", "gridcolor": "#253247", "backgroundcolor": SPACE},
                "zaxis": {"title": "Z · km", "gridcolor": "#253247", "backgroundcolor": SPACE},
                "aspectmode": "data", "bgcolor": SPACE,
            }, height=600, legend={"orientation": "h", "y": -0.08},
        )
        st.plotly_chart(fig3d, width="stretch")

        left, right = st.columns(2)
        altitude_velocity = make_subplots(specs=[[{"secondary_y": True}]])
        altitude_velocity.add_trace(go.Scatter(x=result["time"], y=result["altitude"] / 1000, name="Altitude · km", line={"color": CYAN, "width": 2}), secondary_y=False)
        altitude_velocity.add_trace(go.Scatter(x=result["time"], y=result["velocity"], name="Velocity · m/s", line={"color": AMBER, "width": 2}), secondary_y=True)
        altitude_velocity.update_layout(template="plotly_dark", paper_bgcolor=SPACE, plot_bgcolor=SPACE, title="ALTITUDE & VELOCITY", margin={"l": 8, "r": 8, "t": 45, "b": 20}, legend={"orientation": "h"})
        altitude_velocity.update_xaxes(title_text="Mission time · s", gridcolor="#253247")
        altitude_velocity.update_yaxes(title_text="Altitude · km", secondary_y=False, gridcolor="#253247")
        altitude_velocity.update_yaxes(title_text="Velocity · m/s", secondary_y=True)
        q_g = make_subplots(specs=[[{"secondary_y": True}]])
        q_g.add_trace(go.Scatter(x=result["time"], y=result["q"], name="Dynamic pressure · kPa", line={"color": CYAN, "width": 2}), secondary_y=False)
        q_g.add_trace(go.Scatter(x=result["time"], y=result["g_load"], name="Axial load · G", line={"color": AMBER, "width": 2}), secondary_y=True)
        q_g.add_vline(x=result["max_q_time_s"], line_dash="dash", line_color="#F87171")
        q_g.update_layout(template="plotly_dark", paper_bgcolor=SPACE, plot_bgcolor=SPACE, title="MAX-Q & AXIAL LOAD", margin={"l": 8, "r": 8, "t": 45, "b": 20}, legend={"orientation": "h"})
        q_g.update_xaxes(title_text="Mission time · s", gridcolor="#253247")
        q_g.update_yaxes(title_text="Dynamic pressure · kPa", secondary_y=False, gridcolor="#253247")
        q_g.update_yaxes(title_text="Axial load · G", secondary_y=True)
        left.plotly_chart(altitude_velocity, width="stretch")
        right.plotly_chart(q_g, width="stretch")
        st.caption(
            f'Max-Q {result["max_q_kpa"]:.1f} kPa at T+{result["max_q_time_s"]:.0f} s / '
            f'{result["max_q_altitude_km"]:.1f} km  ·  Estimated mission cost '
            f'${economics["total_mission_cost_musd"]:.1f}M'
        )
    else:
        st.warning("No flight samples were generated. Increase available vehicle propellant or thrust.")

with tab_optimizer:
    st.markdown("#### 100-point guidance sweep")
    st.caption("Vectorized pitch-over and throttle sweep · ranked by Δv margin, cost, and structural Max-Q limit.")
    candidates = optimize_100_trajectories(rocket, destination, SITES[site])
    best = candidates[0]
    pareto = _make_figure()
    pareto.add_trace(go.Scatter(
        x=[item["remaining_dv_m_s"] for item in candidates],
        y=[item["max_q_kpa"] for item in candidates], mode="markers",
        marker={"size": 10, "color": [item["cost_per_kg_usd"] for item in candidates],
                "colorscale": "Turbo", "showscale": True, "colorbar": {"title": "Cost $/kg"}},
        customdata=np.array([[item["rank"], item["pitch_kick_deg"], item["throttle"]] for item in candidates]),
        hovertemplate="Rank #%{customdata[0]:.0f}<br>Margin %{x:.0f} m/s<br>Max-Q %{y:.1f} kPa<br>Pitch %{customdata[1]:.1f}° · Throttle %{customdata[2]:.2f}<extra></extra>",
        name="Guidance candidates",
    ))
    pareto.add_trace(go.Scatter(x=[result["remaining_dv_m_s"]], y=[result["max_q_kpa"]], mode="markers", name="Active rocket",
                                marker={"size": 16, "color": "#FFFFFF", "symbol": "star", "line": {"color": CYAN, "width": 2}}))
    pareto.add_trace(go.Scatter(x=[best["remaining_dv_m_s"]], y=[best["max_q_kpa"]], mode="markers", name="#1 optimum",
                                marker={"size": 15, "color": AMBER, "symbol": "diamond"}))
    pareto.update_layout(title="TRAJECTORY PARETO FRONTIER", xaxis_title="Remaining orbital Δv margin · m/s", yaxis_title="Peak dynamic pressure · kPa")
    st.plotly_chart(pareto, width="stretch")
    if st.button("🚀 Load #1 Optimal Trajectory into Mission Control", type="primary"):
        st.session_state["pending_optimizer_values"] = {
            "gravity_turn_deg": best["pitch_kick_deg"],
            "stage1_thrust_kn": int(np.clip(round(stage1_thrust_kn * best["throttle"] / 100) * 100, 100, 80000)),
        }
        st.rerun()
    out = io.StringIO()
    fields = list(candidates[0].keys())
    writer = csv.DictWriter(out, fieldnames=fields)
    writer.writeheader()
    writer.writerows(candidates)
    st.download_button("Download optimizer leaderboard · CSV", out.getvalue(), "astra_trajectory_optimizer.csv", "text/csv")
    st.dataframe(
        [{"Rank": int(item["rank"]), "Pitch kick (deg)": item["pitch_kick_deg"], "Throttle": item["throttle"],
          "Remaining Δv (m/s)": item["remaining_dv_m_s"], "Max-Q (kPa)": item["max_q_kpa"],
          "Cost ($/kg)": item["cost_per_kg_usd"]} for item in candidates],
        width="stretch", hide_index=True,
    )

with tab_lab:
    st.markdown("#### Geometry & thermal survival")
    nose_col, thermal_col = st.columns(2)
    with nose_col:
        active_profile = custom_profile
        if active_profile is None:
            x = np.linspace(0.0, 1.0, 100)
            shape = NOSECONES_DB[nosecone]["shape"]
            if shape == "cone":
                radius = x
            elif shape == "capsule":
                radius = np.sqrt(np.clip(1.0 - (1.0 - x) ** 2, 0.0, 1.0))
            elif shape == "sears-haack":
                radius = np.sqrt(np.maximum(0.0, 16.0 / (3.0 * np.pi) * (np.arccos(1.0 - 2.0 * x) - (1.0 - 2.0 * x) * np.sqrt(np.maximum(0.0, 1.0 - (1.0 - 2.0 * x) ** 2))))) / 1.65
            else:
                radius = np.sqrt(np.clip(1.0 - (1.0 - x) ** 2, 0.0, 1.0))
            active_profile = np.column_stack((x, radius))
        xbase = np.linspace(0, 1, 100)
        base_radius = np.sqrt(np.clip(1.0 - (1.0 - xbase) ** 2, 0.0, 1.0))
        geometry = _make_figure()
        geometry.add_trace(go.Scatter(x=active_profile[:, 0], y=active_profile[:, 1], mode="lines", name="Active profile", line={"color": CYAN, "width": 3}, fill="tozeroy", fillcolor="rgba(56,189,248,.12)"))
        geometry.add_trace(go.Scatter(x=active_profile[:, 0], y=-active_profile[:, 1], mode="lines", name="Lower surface", line={"color": CYAN, "width": 2}, showlegend=False))
        geometry.add_trace(go.Scatter(x=xbase, y=base_radius, mode="lines", name="Blunt baseline", line={"color": "#64748B", "dash": "dash"}))
        geometry.add_trace(go.Scatter(x=xbase, y=-base_radius, mode="lines", name="Baseline lower", line={"color": "#64748B", "dash": "dash"}, showlegend=False))
        geometry.update_layout(title="NOSECONE / FAIRING PROFILE", xaxis_title="Normalized axial station", yaxis_title="Normalized radius", yaxis={"scaleanchor": "x", "scaleratio": 1})
        st.plotly_chart(geometry, width="stretch")
        nose_factor = NOSECONES_DB[nosecone]["wave_drag"]
        if custom_profile is not None:
            st.metric("UPLOADED FINENESS RATIO", f"{custom_fineness_ratio:.2f}", f"{len(custom_profile)} coordinate points")
        else:
            st.metric("WAVE-DRAG MULTIPLIER", f"{nose_factor:.2f}×", f'{NOSECONES_DB[nosecone]["fineness_ratio"]:.1f} fineness ratio')
    with thermal_col:
        reentry_altitude = np.linspace(80000.0, 0.0, 180)
        reentry_velocity = 7800.0 * np.sqrt(np.clip(1.0 - reentry_altitude / 150000.0, 0.0, 1.0))
        reentry_atm = isa_atmosphere_extended(reentry_altitude, reentry_velocity)
        heat_curve = 1.83e-8 * np.sqrt(reentry_atm["density"] / max(fairing_diameter_m * 0.08, 0.15)) * reentry_velocity**3
        heat_curve = np.nan_to_num(heat_curve, nan=0.0, posinf=0.0)
        heating = _make_figure()
        heating.add_trace(go.Scatter(x=reentry_altitude / 1000, y=heat_curve, mode="lines", name="Sutton–Graves estimate", line={"color": AMBER, "width": 3}, fill="tozeroy", fillcolor="rgba(251,191,36,.12)"))
        heating.add_hline(y=MATERIALS_DB[material]["heat_limit_w_cm2"], line_dash="dash", line_color="#F87171", annotation_text="Selected material limit")
        heating.update_layout(title="REENTRY STAGNATION HEATING", xaxis_title="Altitude · km", yaxis_title="Heat flux · W/cm²")
        heating.update_xaxes(autorange="reversed")
        st.plotly_chart(heating, width="stretch")
        peak_reentry = float(np.max(heat_curve))
        st.metric("PEAK REENTRY HEAT ESTIMATE", f"{peak_reentry:.1f} W/cm²", f'limit {MATERIALS_DB[material]["heat_limit_w_cm2"]:.0f} W/cm²', delta_color="inverse")
    st.info("Thermal curve is a representative Earth-return profile, not a propagated reentry trajectory. Uploaded profile files accept whitespace-, comma-, or semicolon-separated x/radius pairs.")

st.markdown("---")
st.markdown(
    '<span class="subtle">ASTRA FLIGHT DYNAMICS · Simplified point-mass model · Estimates are illustrative and not for flight certification.</span>',
    unsafe_allow_html=True,
)
