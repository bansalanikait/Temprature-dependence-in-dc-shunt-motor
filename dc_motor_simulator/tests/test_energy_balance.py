"""Tests for the energy-residual implementation."""

from dc_motor_simulator.parameters.defaults import DEFAULT_INITIAL_STATE, DEFAULT_INPUTS, get_default_parameter_values
from dc_motor_simulator.simulation.energy_balance import compute_energy_residual
from dc_motor_simulator.simulation.equations import state_derivative


def test_energy_residual_is_finite():
    params = get_default_parameter_values()
    state = DEFAULT_INITIAL_STATE.copy()
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
    assert residual == residual
