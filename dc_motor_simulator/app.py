"""Streamlit application entry point for the DC shunt motor simulator."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Streamlit may run this file with the package directory as sys.path[0].
# Add the repository root so package imports work from either launch location.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from dc_motor_simulator.data.recorder import SimulationRecorder
from dc_motor_simulator.data.report_exporter import build_report_archive
from dc_motor_simulator.dashboard.plots import (
    combined_current_plot,
    combined_temperature_plot,
    line_plot,
)
from dc_motor_simulator.parameters.defaults import (
    DEFAULT_INITIAL_STATE,
    DEFAULT_INPUTS,
    DEFAULT_SIMULATION_SETTINGS,
    PARAMETER_METADATA,
    get_default_parameter_values,
)
from dc_motor_simulator.parameters.parameter_manager import ParameterManager
from dc_motor_simulator.simulation.energy_balance import compute_energy_residual
from dc_motor_simulator.simulation.equations import state_derivative
from dc_motor_simulator.simulation.solver import (
    SimulationResult,
    simulate_motor_live,
    vector_to_state,
)
from dc_motor_simulator.simulation.validation import validate_parameter_set


PARAMETER_TABS = {
    "Electrical": "Electrical",
    "Mechanical": "Mechanical",
    "Thermal": "Thermal",
    "Constants": "Physical and numerical",
    "Derived": "Derived",
}

INITIAL_LABELS = {
    "if_": ("Field current", "A"),
    "ia": ("Armature current", "A"),
    "omega": ("Angular velocity", "rad/s"),
    "theta_a": ("Armature temperature", "°C"),
    "theta_b": ("Brush temperature", "°C"),
    "theta_f": ("Field temperature", "°C"),
    "theta_fr": ("Frame temperature", "°C"),
}


def _set_state_defaults() -> None:
    if "pm" not in st.session_state:
        st.session_state.pm = ParameterManager()
    if "inputs" not in st.session_state:
        st.session_state.inputs = DEFAULT_INPUTS.copy()
    if "initial_state" not in st.session_state:
        st.session_state.initial_state = DEFAULT_INITIAL_STATE.copy()
    if "sim_result" not in st.session_state:
        st.session_state.sim_result = None
    if "recorder" not in st.session_state:
        st.session_state.recorder = SimulationRecorder()
    if "recording" not in st.session_state:
        st.session_state.recording = False
    if "selected_time" not in st.session_state:
        st.session_state.selected_time = 0.0
    if "duration" not in st.session_state:
        st.session_state.duration = float(DEFAULT_SIMULATION_SETTINGS["duration"])
    if "sample_interval" not in st.session_state:
        st.session_state.sample_interval = float(DEFAULT_SIMULATION_SETTINGS["dt"])
    if "solver_method" not in st.session_state:
        st.session_state.solver_method = DEFAULT_SIMULATION_SETTINGS["solver_method"]
    if "playback_speed" not in st.session_state:
        st.session_state.playback_speed = float(DEFAULT_SIMULATION_SETTINGS["playback_speed"])
    if "run_conditions" not in st.session_state:
        st.session_state.run_conditions = None
    if "run_report_archive" not in st.session_state:
        st.session_state.run_report_archive = None


def _apply_parameter_values(values: dict[str, float], auto_phi: bool) -> None:
    manager = ParameterManager(values, auto_phi=auto_phi)
    st.session_state.pm = manager
    for name, value in manager.as_dict().items():
        st.session_state[f"parameter_{name}"] = float(value)


def _render_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1500px;}
        [data-testid="stMetric"] {
            background: #f7f9fc; border: 1px solid #e6eaf0; padding: 0.75rem 1rem;
            border-radius: 0.65rem;
        }
        [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] *,
        [data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
            color: #172b4d !important; -webkit-text-fill-color: #172b4d !important;
        }
        [data-testid="stMetricLabel"] {font-size: 0.82rem;}
        [data-testid="stMetricValue"] {font-size: 1.4rem;}
        div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: #e6eaf0; border-radius: 0.7rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_input_column() -> tuple[float, float, str, float]:
    st.subheader("Operating inputs")
    inputs = st.session_state.inputs
    inputs["V"] = st.number_input("Supply voltage · V", value=float(inputs["V"]), step=1.0, key="input_voltage")
    inputs["TL"] = st.number_input("Load torque · N·m", value=float(inputs["TL"]), step=0.1, key="input_load")
    inputs["Tamb"] = st.number_input("Ambient temperature · °C", value=float(inputs["Tamb"]), step=1.0, key="input_ambient")

    with st.expander("Initial conditions", expanded=False):
        for key, (label, unit) in INITIAL_LABELS.items():
            st.session_state.initial_state[key] = st.number_input(
                f"{label} · {unit}",
                value=float(st.session_state.initial_state[key]),
                step=0.1,
                key=f"initial_{key}",
            )

    st.divider()
    st.subheader("Run configuration")
    duration = st.number_input(
        "Physical duration · s",
        min_value=0.01,
        value=float(st.session_state.duration),
        step=1.0,
        key="duration_input",
    )
    dt = st.number_input(
        "Sample interval · s",
        min_value=0.001,
        value=float(st.session_state.sample_interval),
        step=0.05,
        key="dt_input",
    )
    method = st.selectbox(
        "Numerical solver",
        options=["BDF", "Radau"],
        index=["BDF", "Radau"].index(st.session_state.solver_method),
        key="solver_input",
    )
    playback = st.select_slider(
        "Display playback speed",
        options=[0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
        value=float(st.session_state.playback_speed),
        key="playback_input",
    )
    st.caption("Playback changes only the display pace, not the physical model.")
    st.session_state.duration = float(duration)
    st.session_state.sample_interval = float(dt)
    st.session_state.solver_method = method
    st.session_state.playback_speed = float(playback)
    return duration, dt, method, playback


def _render_parameter_tabs() -> None:
    st.subheader("Motor parameters")
    st.caption("Edit values by engineering category. Parameter source and status are shown in each field's help text.")
    tabs = st.tabs(list(PARAMETER_TABS))
    manager = st.session_state.pm
    for tab, category in zip(tabs, PARAMETER_TABS.values()):
        with tab:
            if category == "Thermal":
                st.checkbox(
                    "Calculate φa, φb, φp automatically from resistance fractions",
                    value=manager.auto_phi,
                    key="auto_phi",
                )
                manager.auto_phi = st.session_state.auto_phi
                if manager.auto_phi:
                    calculated = manager.calculate_phi()
                    for name, value in calculated.items():
                        manager.set(name, value)
                        st.session_state[f"parameter_{name}"] = float(value)

            names = [
                name
                for name, meta in PARAMETER_METADATA.items()
                if meta["category"] == category
            ]
            columns = st.columns(2)
            for index, name in enumerate(names):
                meta = manager.metadata_for(name)
                key = f"parameter_{name}"
                if key not in st.session_state:
                    st.session_state[key] = float(meta["value"])
                with columns[index % 2]:
                    if name == "R0":
                        st.number_input(
                            f"{meta['name']} · {meta['units']}",
                            value=float(manager.get(name)),
                            disabled=True,
                            help="Calculated automatically as Ra0 + Rip + Rbr + Rsl.",
                            key="derived_R0_display",
                        )
                    elif category == "Thermal" and name in {"phi_a", "phi_b", "phi_p"} and manager.auto_phi:
                        st.number_input(
                            f"{meta['name']} · {meta['units']}",
                            value=float(manager.get(name)),
                            disabled=True,
                            help="Calculated from the armature-path resistance fractions.",
                            key=f"calculated_{name}",
                        )
                    else:
                        st.number_input(
                            f"{meta['name']} · {meta['units']}",
                            key=key,
                            step=0.001,
                            format="%.6g",
                            help=f"Source: {meta['source']} | Status: {meta['status']}",
                        )
                        manager.set(name, float(st.session_state[key]))

            if category == "Thermal":
                if not manager.auto_phi:
                    st.caption("With automatic calculation off, edit the three fractions directly; they must sum to 1.")

    with st.expander("Parameter presets and JSON", expanded=False):
        preset_name = st.text_input("Preset name", value="motor-parameters", key="preset_name")
        save_col, reset_col = st.columns(2)
        with save_col:
            st.download_button(
                "Download parameter JSON",
                data=json.dumps(
                    {"auto_phi": manager.auto_phi, "parameters": manager.as_dict()},
                    indent=2,
                ),
                file_name=f"{preset_name}.json",
                mime="application/json",
            )
        with reset_col:
            if st.button("Reset parameters", key="reset_parameters"):
                _apply_parameter_values(get_default_parameter_values(), auto_phi=True)
                st.rerun()

        uploaded = st.file_uploader("Load parameter JSON", type=["json"], key="parameter_upload")
        if uploaded is not None and st.button("Apply imported parameters", key="apply_import"):
            try:
                payload = json.loads(uploaded.getvalue().decode("utf-8"))
                values = payload.get("parameters", payload)
                if not isinstance(values, dict):
                    raise ValueError("The JSON file must contain a parameter object.")
                _apply_parameter_values(values, bool(payload.get("auto_phi", True)))
                st.success("Parameters loaded.")
                st.rerun()
            except (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError) as error:
                st.error(f"Could not load parameter file: {error}")


def _state_metrics(state: dict[str, float], torque: float, load_torque: float) -> None:
    columns = st.columns(4)
    metrics = [
        ("Armature current", state["ia"], "A"),
        ("Field current", state["if_"], "A"),
        ("Speed", state["omega"] * 60 / (2 * np.pi), "RPM"),
        ("Speed", state["omega"], "rad/s"),
        ("Electromagnetic torque", torque, "N·m"),
        ("Load torque", load_torque, "N·m"),
    ]
    for index, (label, value, unit) in enumerate(metrics):
        columns[index % 4].metric(label, f"{value:,.3f} {unit}")
    temperature_columns = st.columns(4)
    for column, (key, label) in zip(
        temperature_columns,
        [
            ("theta_a", "Armature"),
            ("theta_b", "Brush"),
            ("theta_f", "Field"),
            ("theta_fr", "Frame"),
        ],
    ):
        column.metric(f"{label} temperature", f"{state[key]:,.2f} °C")


def _live_figures(
    result: SimulationResult,
    params: dict[str, float],
    inputs: dict[str, float],
) -> list[go.Figure]:
    derived = _derived_data(result, params, inputs)
    sample_count = len(result.time)
    stride = max(1, int(np.ceil(sample_count / 1500)))
    selected = np.arange(0, sample_count, stride)
    if selected[-1] != sample_count - 1:
        selected = np.append(selected, sample_count - 1)

    sample_time = result.time[selected]
    states = result.states[:, selected]
    figures = [
        combined_current_plot(sample_time, states[1], states[0]),
        line_plot(sample_time, states[2], title="Motor speed", yaxis_title="Speed (rad/s)", line_name="ω"),
        combined_temperature_plot(sample_time, states[3], states[4], states[5], states[6]),
        line_plot(sample_time, derived["torque"][selected], title="Electromagnetic torque", yaxis_title="Torque (N·m)", line_name="Torque"),
    ]

    power_figure = go.Figure()
    power_figure.add_trace(
        go.Scatter(
            x=sample_time,
            y=derived["electrical_power"][selected],
            name="Electrical input",
            mode="lines",
            line=dict(color="#2563eb", width=2.4),
        )
    )
    power_figure.add_trace(
        go.Scatter(
            x=sample_time,
            y=derived["mechanical_power"][selected],
            name="Electromagnetic mechanical",
            mode="lines",
            line=dict(color="#0f766e", width=2.4),
        )
    )
    power_figure.update_layout(
        template="plotly_white",
        title="Electrical and mechanical power",
        height=360,
        margin=dict(l=20, r=20, t=62, b=24),
        hovermode="x unified",
        xaxis_title="Simulation time (s)",
        yaxis_title="Power (W)",
        legend=dict(orientation="h", y=1.02, x=0),
    )
    figures.append(power_figure)
    figures.append(
        line_plot(
            sample_time,
            derived["residual"][selected],
            title="Specified energy residual",
            yaxis_title="Residual (W)",
            line_name="Residual",
        )
    )
    return figures


def _derived_data(result: SimulationResult, params: dict[str, float], inputs: dict[str, float]) -> dict[str, np.ndarray]:
    torques = []
    powers = []
    residuals = []
    electrical_power = []
    for index, time_value in enumerate(result.time):
        state = vector_to_state(result.states[:, index])
        derivative = state_derivative(time_value, result.states[:, index].tolist(), params, inputs)
        derivative_map = {
            "if_": derivative[0],
            "ia": derivative[1],
            "omega": derivative[2],
            "theta_a": derivative[3],
            "theta_b": derivative[4],
            "theta_f": derivative[5],
            "theta_fr": derivative[6],
        }
        torques.append(params["Laf"] * state["if_"] * state["ia"])
        powers.append(state["omega"] * params["Laf"] * state["if_"] * state["ia"])
        residuals.append(compute_energy_residual(params, state, inputs, derivative_map))
        electrical_power.append(inputs["V"] * (state["ia"] + state["if_"]))
    return {
        "torque": np.asarray(torques),
        "mechanical_power": np.asarray(powers),
        "residual": np.asarray(residuals),
        "electrical_power": np.asarray(electrical_power),
    }


def _record_result(result: SimulationResult) -> None:
    recorder = st.session_state.recorder
    recorder.clear()
    params = st.session_state.pm.as_dict()
    derived = _derived_data(result, params, st.session_state.inputs)
    for index, time_value in enumerate(result.time):
        state = vector_to_state(result.states[:, index])
        torque = params["Laf"] * state["if_"] * state["ia"]
        power = st.session_state.inputs["V"] * (state["ia"] + state["if_"])
        recorder.record(
            time_value,
            state,
            st.session_state.inputs,
            torque,
            power,
            mechanical_power=float(derived["mechanical_power"][index]),
            energy_residual=float(derived["residual"][index]),
        )


def _run_live_simulation(duration: float, dt: float, method: str, playback: float, placeholders: dict) -> None:
    manager = st.session_state.pm
    params = manager.as_dict()
    inputs = st.session_state.inputs.copy()
    progress = placeholders["progress"]
    status = placeholders["status"]
    dashboard = placeholders["dashboard"]
    live_plots = placeholders["plots"]
    speed = max(playback, 0.01)
    current_result = None
    previous_time = 0.0
    report_error = None

    try:
        for current_result in simulate_motor_live(
            params,
            inputs,
            st.session_state.initial_state.copy(),
            duration,
            dt,
            method,
        ):
            latest = float(current_result.time[-1])
            state = vector_to_state(current_result.states[:, -1])
            torque = params["Laf"] * state["if_"] * state["ia"]
            with dashboard:
                _state_metrics(state, torque, inputs["TL"])
            for index, (plot_slot, figure) in enumerate(
                zip(live_plots, _live_figures(current_result, params, inputs))
            ):
                plot_slot.plotly_chart(
                    figure,
                    use_container_width=True,
                    key=f"live_output_plot_{index}_{len(current_result.time)}",
                )
            progress.progress(min(latest / duration, 1.0), text=f"Simulation time {latest:.3f} / {duration:.3f} s")
            status.caption(f"Running · {method} · {latest:.3f} s")
            elapsed = max(0.0, latest - previous_time)
            previous_time = latest
            if latest < duration and elapsed > 0:
                time.sleep(min(elapsed / speed, 0.5))

        if current_result is None:
            raise RuntimeError("The solver returned no simulation samples.")
        st.session_state.sim_result = current_result
        st.session_state.selected_time = float(current_result.time[-1])
        _record_result(current_result)
        try:
            st.session_state.run_report_archive = build_report_archive(
                st.session_state.recorder.to_dataframe(),
                st.session_state.run_conditions,
                params,
            )
        except (ValueError, TypeError, OSError) as error:
            st.session_state.run_report_archive = None
            report_error = error
        status.success(f"Complete · {len(current_result.time):,} samples · {duration:g} simulated seconds")
        if report_error is not None:
            status.warning(f"Simulation completed, but report generation failed: {report_error}")
        progress.progress(1.0, text="Simulation complete")
    except (ValueError, RuntimeError, FloatingPointError, OverflowError) as error:
        st.session_state.sim_result = current_result
        status.error(f"Simulation failed: {error}")


def _render_results() -> None:
    result = st.session_state.sim_result
    if result is None or len(result.time) == 0:
        st.info("Set operating conditions in the input column and start a run to view live results.")
        return

    selected_time = st.slider(
        "Inspect simulation time · s",
        min_value=float(result.time[0]),
        max_value=float(result.time[-1]),
        value=float(np.clip(st.session_state.selected_time, result.time[0], result.time[-1])),
        step=max(float(result.time[-1] - result.time[0]) / 500, 1e-6),
        key="inspect_time",
    )
    st.session_state.selected_time = selected_time
    state_index = int(np.argmin(np.abs(result.time - selected_time)))
    state = vector_to_state(result.states[:, state_index])
    torque = st.session_state.pm.get("Laf") * state["if_"] * state["ia"]
    _state_metrics(state, torque, st.session_state.inputs["TL"])
    derived = _derived_data(result, st.session_state.pm.as_dict(), st.session_state.inputs)

    overview, electrical, thermal, mechanical, energy = st.tabs(
        ["Overview", "Electrical", "Thermal", "Mechanical", "Energy"]
    )
    with overview:
        st.plotly_chart(
            combined_current_plot(result.time, result.states[1], result.states[0]),
            use_container_width=True,
            key="overview_response",
        )
        st.plotly_chart(
            combined_temperature_plot(
                result.time,
                result.states[3],
                result.states[4],
                result.states[5],
                result.states[6],
            ),
            use_container_width=True,
            key="overview_temperature",
        )
    with electrical:
        columns = st.columns(2)
        columns[0].plotly_chart(line_plot(result.time, result.states[1], title="Armature current", yaxis_title="Current (A)", line_name="ia"), use_container_width=True)
        columns[1].plotly_chart(line_plot(result.time, result.states[0], title="Field current", yaxis_title="Current (A)", line_name="if"), use_container_width=True)
        st.plotly_chart(combined_current_plot(result.time, result.states[1], result.states[0]), use_container_width=True)
    with thermal:
        for row in range(2):
            columns = st.columns(2)
            for column_index in range(2):
                state_index_plot = 3 + row * 2 + column_index
                label = ["Armature", "Brush", "Field", "Frame"][row * 2 + column_index]
                columns[column_index].plotly_chart(
                    line_plot(result.time, result.states[state_index_plot], title=f"{label} temperature", yaxis_title="Temperature (°C)", line_name=label),
                    use_container_width=True,
                )
    with mechanical:
        columns = st.columns(2)
        columns[0].plotly_chart(line_plot(result.time, result.states[2], title="Angular speed", yaxis_title="Speed (rad/s)", line_name="omega"), use_container_width=True)
        columns[1].plotly_chart(line_plot(result.time, derived["torque"], title="Electromagnetic torque", yaxis_title="Torque (N·m)", line_name="Te"), use_container_width=True)
        power_figure = go.Figure()
        power_figure.add_trace(go.Scatter(x=result.time, y=derived["electrical_power"], name="Electrical input", mode="lines"))
        power_figure.add_trace(go.Scatter(x=result.time, y=derived["mechanical_power"], name="Electromagnetic mechanical", mode="lines"))
        power_figure.update_layout(template="plotly_white", title="Electrical and mechanical power", xaxis_title="Time (s)", yaxis_title="Power (W)", hovermode="x unified", height=370)
        st.plotly_chart(power_figure, use_container_width=True)
    with energy:
        st.plotly_chart(line_plot(result.time, derived["residual"], title="Specified energy residual", yaxis_title="Residual (W)", line_name="Residual"), use_container_width=True)
        st.caption("Residual follows the supplied expression exactly; a non-zero residual is shown rather than corrected.")


def _render_recording() -> None:
    st.subheader("Recorded data")
    result = st.session_state.sim_result
    if result is None or len(result.time) == 0:
        st.caption("Recording begins with the next simulation run.")
        return
    frame = st.session_state.recorder.to_dataframe()
    if frame.empty:
        _record_result(result)
        frame = st.session_state.recorder.to_dataframe()
    lower, upper = st.columns(2)
    start = lower.number_input("Export from · s", min_value=float(frame["time_s"].min()), max_value=float(frame["time_s"].max()), value=float(frame["time_s"].min()), key="csv_start")
    end = upper.number_input("Export to · s", min_value=float(frame["time_s"].min()), max_value=float(frame["time_s"].max()), value=float(frame["time_s"].max()), key="csv_end")
    selected = frame[(frame["time_s"] >= start) & (frame["time_s"] <= end)]
    st.download_button(
        "Download selected CSV",
        data=selected.to_csv(index=False).encode("utf-8"),
        file_name="dc_motor_simulation.csv",
        mime="text/csv",
    )
    report_archive = st.session_state.get("run_report_archive")
    if report_archive is not None:
        st.download_button(
            "Download complete test report (ZIP)",
            data=report_archive,
            file_name="dc_motor_simulation_report.zip",
            mime="application/zip",
            help="Includes a self-contained HTML report with plots, all configured parameters, run conditions, and the complete CSV data.",
            type="primary",
            key="download_complete_report",
        )


def main() -> None:
    st.set_page_config(
        page_title="DC Motor Simulator",
        page_icon="⚙️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _set_state_defaults()
    _render_styles()
    st.title("DC Shunt Motor Simulator")
    st.caption("Coupled electrical, mechanical, and thermal model · SI units · BDF / Radau integration")

    inputs_column, workspace_column = st.columns([0.85, 2.2], gap="large")
    with inputs_column:
        duration, dt, method, playback = _render_input_column()
        start = st.button("▶  Start live simulation", type="primary", use_container_width=True, key="start_simulation")
        stop = st.button("■  Stop / reset results", use_container_width=True, key="stop_simulation")
        if stop:
            st.session_state.sim_result = None
            st.session_state.recorder.clear()
            st.session_state.selected_time = 0.0
            st.session_state.run_conditions = None
            st.session_state.run_report_archive = None
        if st.session_state.sim_result is not None:
            st.caption("Completed run available for inspection below.")

    with workspace_column:
        live_status = st.empty()
        live_progress = st.empty()
        live_dashboard = st.empty()
        live_chart_slots = []
        for _ in range(3):
            chart_columns = st.columns(2)
            live_chart_slots.extend(column.empty() for column in chart_columns)
        if start:
            warnings = validate_parameter_set(st.session_state.pm.as_dict())
            if warnings:
                live_status.error("Check parameters before running: " + "; ".join(warnings))
            else:
                st.session_state.run_report_archive = None
                st.session_state.run_conditions = {
                    "run_started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                    "inputs": st.session_state.inputs.copy(),
                    "initial_state": st.session_state.initial_state.copy(),
                    "simulation_configuration": {
                        "physical_duration_s": float(duration),
                        "sample_interval_s": float(dt),
                        "solver": method,
                        "playback_speed": float(playback),
                        "relative_tolerance": 1e-6,
                        "absolute_tolerance": 1e-9,
                    },
                }
                _run_live_simulation(
                    duration,
                    dt,
                    method,
                    playback,
                    {
                        "status": live_status,
                        "progress": live_progress,
                        "dashboard": live_dashboard,
                        "plots": live_chart_slots,
                    },
                )
        _render_results()

    st.divider()
    _render_parameter_tabs()
    _render_recording()
    warnings = validate_parameter_set(st.session_state.pm.as_dict())
    with st.expander("Engineering validation", expanded=bool(warnings)):
        if warnings:
            for warning in warnings:
                st.warning(warning)
        else:
            st.success("Parameter checks passed. This does not constitute experimental motor validation.")


if __name__ == "__main__":
    main()
