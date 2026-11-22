"""Tests for the energy-residual implementation."""

import pytest

from dc_motor_simulator.parameters.defaults import DEFAULT_INITIAL_STATE, DEFAULT_INPUTS, get_default_parameter_values
from dc_motor_simulator.simulation.energy_balance import compute_energy_residual, compute_thermal_balance_residual
from dc_motor_simulator.simulation.equations import state_derivative
from dc_motor_simulator.simulation.solver import simulate_motor


@pytest.mark.parametrize(
    "state_updates",
    [
        {},
        {"ia": 0.0, "if_": 0.0, "omega": 0.0},
        {"ia": -3.2, "if_": 0.9, "omega": 75.0},
        {"ia": 12.0, "if_": 1.1, "omega": 0.0},
    ],
)
def test_energy_residual_closes_for_model_equations(state_updates):
    params = get_default_parameter_values()
    state = DEFAULT_INITIAL_STATE.copy()
    state.update(state_updates)
    deriv = state_derivative(0.0, [
        state["if_"],
        state["ia"],
        state["omega"],
        state["theta_a"],
        state["theta_b"],
        state["theta_f"],
        state["theta_fr"],
    ], params, DEFAULT_INPUTS)
    residual = compute_energy_residual(params, state, DEFAULT_INPUTS, {
        "if_": deriv[0],
        "ia": deriv[1],
        "omega": deriv[2],
        "theta_a": deriv[3],
        "theta_b": deriv[4],
        "theta_f": deriv[5],
        "theta_fr": deriv[6],
    })
    assert residual == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize(
    "state_updates",
    [
        {},
        {"ia": 0.0, "if_": 0.0, "omega": 0.0},
        {"ia": 8.0, "if_": 1.2, "omega": 140.0},
    ],
)
def test_thermal_balance_closes_for_model_equations(state_updates):
    params = get_default_parameter_values()
    state = DEFAULT_INITIAL_STATE.copy()
    state.update(state_updates)
    deriv = state_derivative(0.0, [
        state["if_"],
        state["ia"],
        state["omega"],
        state["theta_a"],
        state["theta_b"],
        state["theta_f"],
        state["theta_fr"],
    ], params, DEFAULT_INPUTS)
    derivative_map = dict(zip(
        ["if_", "ia", "omega", "theta_a", "theta_b", "theta_f", "theta_fr"],
        deriv,
    ))
    residual = compute_thermal_balance_residual(
        params, state, DEFAULT_INPUTS, derivative_map
    )
    assert residual == pytest.approx(0.0, abs=1e-10)


def test_both_balance_checks_stay_below_tolerance_over_solver_run():
    params = get_default_parameter_values()
    result = simulate_motor(
        params,
        DEFAULT_INPUTS,
        DEFAULT_INITIAL_STATE,
        duration=0.5,
        dt=0.05,
        solver_method="BDF",
    )
    assert result.success

    thermal_residuals = []
    whole_motor_residuals = []
    for index, time_value in enumerate(result.time):
        state_vector = result.states[:, index]
        state = {
            "if_": state_vector[0],
            "ia": state_vector[1],
            "omega": state_vector[2],
            "theta_a": state_vector[3],
            "theta_b": state_vector[4],
            "theta_f": state_vector[5],
            "theta_fr": state_vector[6],
        }
        derivative = state_derivative(
            time_value, state_vector.tolist(), params, DEFAULT_INPUTS
        )
        derivative_map = dict(zip(
            ["if_", "ia", "omega", "theta_a", "theta_b", "theta_f", "theta_fr"],
            derivative,
        ))
        thermal_residuals.append(
            compute_thermal_balance_residual(
                params, state, DEFAULT_INPUTS, derivative_map
            )
        )
        whole_motor_residuals.append(
            compute_energy_residual(params, state, DEFAULT_INPUTS, derivative_map)
        )

    assert max(map(abs, thermal_residuals)) <= 1e-9
    assert max(map(abs, whole_motor_residuals)) <= 1e-9
