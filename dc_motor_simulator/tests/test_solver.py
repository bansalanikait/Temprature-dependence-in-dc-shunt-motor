"""Tests for the numeric solver."""

from dc_motor_simulator.parameters.defaults import DEFAULT_INITIAL_STATE, DEFAULT_INPUTS, get_default_parameter_values
from dc_motor_simulator.simulation.solver import simulate_motor


def test_simulate_motor_runs_successfully():
    params = get_default_parameter_values()
    result = simulate_motor(params, DEFAULT_INPUTS, DEFAULT_INITIAL_STATE, duration=0.5, dt=0.05, solver_method="BDF")
    assert result.success is True
    assert len(result.time) > 1
    assert result.states.shape == (7, len(result.time))
