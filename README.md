# ASTRA Mission Control

A dark-mode Streamlit dashboard for exploring a simplified two-stage launch vehicle, ascent trajectory, orbital energy, booster recovery trade-offs, aerodynamic loads, reentry heating, and launch economics.

## Run locally

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The project intentionally depends only on NumPy, Plotly, and Streamlit. The `.streamlit/config.toml` file supplies the dark mission-control theme. Streamlit opens the local app in the system's default browser; set Chrome as the Windows default browser if it should open there.

## Dashboard

- **Flight Trajectory:** vehicle and destination controls, eight telemetry metrics, baseline comparison, timeline scrubber, Earth-centered 3D ascent, and synchronized altitude/velocity and Max-Q/load charts.
- **Batch Optimizer:** a 100-point vectorized guidance/throttle screening sweep, Pareto chart, ranked results, and standard-library CSV export.
- **Nosecone & Reentry Lab:** preset or uploaded coordinate profiles and a representative stagnation-heating curve against material limits.

The optional `.dat` / `.csv` profile format is two numeric columns per row: axial coordinate and radius. Commas, semicolons, or whitespace are accepted; blank lines and lines beginning with `#` or `//` are ignored.

## Model scope

`rocket_math.py` contains the propellant, material, nosecone, and destination catalogs; an extended ISA-like atmosphere; a transonic drag approximation; a point-mass two-dimensional ascent integrator with central gravity and launch-site rotation; Sutton–Graves-style heating; economics estimates; baseline deltas; and a vectorized optimizer surrogate.

This is an interactive educational model, **not** a flight-qualified simulation. The integrator uses simplified stage geometry, fixed representative engine performance, idealized guidance, no engine transients, and a point-mass vehicle. Booster recovery reserves propellant and estimates refurbishment costs; it does not propagate a separate boostback, entry, or landing trajectory. The batch optimizer is a screening surrogate, not 100 independent high-fidelity trajectory integrations. The reentry plot is a representative return profile rather than a reentry propagated from the ascent. Results should not be used for vehicle design, operational decisions, or safety certification.
