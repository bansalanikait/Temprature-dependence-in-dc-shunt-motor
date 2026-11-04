"""Tests for equation evaluation."""

from dc_motor_simulator.parameters.defaults import DEFAULT_INITIAL_STATE, DEFAULT_INPUTS, get_default_parameter_values
from dc_motor_simulator.simulation.equations import state_derivative


def test_state_derivative_is_finite():
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
    assert len(deriv) == 7
    for value in deriv:
        assert value == value
