"""Numerical solver for the DC motor simulation."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterator

import numpy as np
from scipy.integrate import solve_ivp

from dc_motor_simulator.parameters.defaults import DEFAULT_INITIAL_STATE
from dc_motor_simulator.simulation.equations import state_derivative
from dc_motor_simulator.simulation.validation import validate_parameter_set, validate_state_vector


@dataclass
class SimulationResult:
    time: np.ndarray
    states: np.ndarray
    success: bool
    message: str
    solver_method: str


def state_to_vector(state: dict[str, float]) -> np.ndarray:
    return np.array([
        float(state["if_"]),
        float(state["ia"]),
        float(state["omega"]),
        float(state["theta_a"]),
        float(state["theta_b"]),
        float(state["theta_f"]),
        float(state["theta_fr"]),
    ], dtype=float)


def vector_to_state(values: np.ndarray) -> dict[str, float]:
    return {
        "if_": float(values[0]),
        "ia": float(values[1]),
        "omega": float(values[2]),
        "theta_a": float(values[3]),
        "theta_b": float(values[4]),
        "theta_f": float(values[5]),
        "theta_fr": float(values[6]),
    }


def simulate_motor(params: dict[str, float], inputs: dict[str, float], initial_state: dict[str, float] | None = None, duration: float = 10.0, dt: float = 0.1, solver_method: str = "BDF") -> SimulationResult:
    """Solve the coupled motor equations over a fixed time horizon."""
    warnings = validate_parameter_set(params)
    if warnings:
        return SimulationResult(np.array([]), np.empty((7, 0)), False, "; ".join(warnings), solver_method)

    if initial_state is None:
        initial_state = DEFAULT_INITIAL_STATE

    state_values = state_to_vector(initial_state)
    state_warnings = validate_state_vector(state_values)
    if state_warnings:
        return SimulationResult(np.array([]), np.empty((7, 0)), False, "; ".join(state_warnings), solver_method)

    if duration <= 0 or dt <= 0:
        return SimulationResult(np.array([]), np.empty((7, 0)), False, "Simulation duration and sampling interval must be positive.", solver_method)

    t_eval = np.arange(0.0, duration + dt, dt, dtype=float)

    def rhs(t: float, y: np.ndarray):
        return np.asarray(state_derivative(t, y.tolist(), params, inputs), dtype=float)

    result = solve_ivp(
        rhs,
        (0.0, duration),
        state_values,
        method=solver_method,
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-9,
        max_step=max(dt, 0.05),
    )

    if not result.success:
        return SimulationResult(result.t, result.y, False, result.message, solver_method)

    return SimulationResult(result.t, result.y, True, "Simulation finished successfully.", solver_method)


def simulate_motor_live(
    params: dict[str, float],
    inputs: dict[str, float],
    initial_state: dict[str, float] | None = None,
    duration: float = 10.0,
    dt: float = 0.1,
    solver_method: str = "BDF",
    points_per_update: int = 10,
) -> Iterator[SimulationResult]:
    """Yield progressively integrated output using solve_ivp for every segment."""
    warnings = validate_parameter_set(params)
    if warnings:
        raise ValueError("; ".join(warnings))
    if duration <= 0 or dt <= 0:
        raise ValueError("Simulation duration and sampling interval must be positive.")
    if points_per_update < 1:
        raise ValueError("points_per_update must be at least one.")

    state = DEFAULT_INITIAL_STATE if initial_state is None else initial_state
    state_values = state_to_vector(state)
    state_warnings = validate_state_vector(state_values)
    if state_warnings:
        raise ValueError("; ".join(state_warnings))

    sample_times = np.arange(0.0, duration, dt, dtype=float)
    if sample_times.size == 0 or not np.isclose(sample_times[-1], duration):
        sample_times = np.append(sample_times, duration)
    else:
        sample_times[-1] = duration

    times = [float(sample_times[0])]
    states = [state_values.copy()]

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        return np.asarray(state_derivative(t, y.tolist(), params, inputs), dtype=float)

    yield SimulationResult(
        np.asarray(times),
        np.asarray(states).T,
        True,
        "Simulation initialized.",
        solver_method,
    )

    for start_index in range(0, len(sample_times) - 1, points_per_update):
        segment_times = sample_times[start_index + 1 : start_index + points_per_update + 1]
        segment_end = float(segment_times[-1])
        segment_start = float(times[-1])
        result = solve_ivp(
            rhs,
            (segment_start, segment_end),
            state_values,
            method=solver_method,
            t_eval=segment_times,
            rtol=1e-6,
            atol=1e-9,
            max_step=max(dt, 0.05),
        )
        if not result.success:
            raise RuntimeError(result.message)
        if not np.all(np.isfinite(result.y)):
            raise FloatingPointError("Solver produced non-finite state values.")

        times.extend(result.t.tolist())
        states.extend(result.y.T.copy())
        state_values = result.y[:, -1].copy()
        yield SimulationResult(
            np.asarray(times),
            np.asarray(states).T,
            True,
            f"Integrated to {result.t[-1]:.6g} s of {duration:.6g} s.",
            solver_method,
        )
