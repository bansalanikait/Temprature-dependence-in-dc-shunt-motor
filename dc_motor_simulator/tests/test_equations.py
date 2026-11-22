"""Tests for equation evaluation."""

import math

import pytest

from dc_motor_simulator.parameters.defaults import DEFAULT_INITIAL_STATE, DEFAULT_INPUTS, get_default_parameter_values
from dc_motor_simulator.simulation.equations import compute_helper_equations, state_derivative


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


def test_state_derivative_matches_corrected_model():
    params = get_default_parameter_values()
    state = {
        "if_": 0.9,
        "ia": 5.5,
        "omega": 80.0,
        "theta_a": 35.0,
        "theta_b": 31.0,
        "theta_f": 40.0,
        "theta_fr": 30.0,
    }
    inputs = {"V": 220.0, "TL": 4.0, "Tamb": 25.0}
    r = params["R0"] * (1 + params["alpha_e"] * (state["theta_a"] - params["theta0"]))
    rf = params["Rf0"] * (1 + params["alpha"] * (state["theta_f"] - params["theta0"]))
    si = math.tanh(state["ia"] / params["eps_i"])
    ga = params["Ga0"] * (1 + params["ka"] * state["omega"])
    gb = params["Gb0"] * (1 + params["kb"] * state["omega"])
    dth = state["theta_fr"] - inputs["Tamb"]
    dts = math.sqrt(dth ** 2 + params["delta"] ** 2)
    qnat = params["hn0"] * params["As"] * (dts / params["DTref"]) ** 0.25 * dth
    qfor = params["kF"] * state["omega"] * params["As"] * dth
    tfrk = state["theta_fr"] + 273.15
    tairk = inputs["Tamb"] + 273.15
    qrad = params["eps_s"] * params["sigma"] * params["As"] * (tfrk ** 4 - tairk ** 4)
    qmount = params["Gm"] * dth
    pcu_a = params["phi_a"] * r * state["ia"] ** 2
    pcu_b = params["phi_b"] * r * state["ia"] ** 2
    pcu_p = params["phi_p"] * r * state["ia"] ** 2
    piron = (params["ch"] * state["omega"] + params["ce"] * state["omega"] ** 2) * state["if_"] ** 2
    pbrdrop = params["Vb0"] * state["ia"] * si
    pbrfric = params["Tcb"] * state["omega"]
    pfield = rf * state["if_"] ** 2
    pbear = params["B"] * state["omega"] ** 2
    pwind = params["phi_w"] * params["cw"] * state["omega"] ** 3
    qaf = ga * (state["theta_a"] - state["theta_fr"])
    qbf = gb * (state["theta_b"] - state["theta_fr"])
    qab = params["Gab"] * (state["theta_a"] - state["theta_b"])
    qff = params["Gf"] * (state["theta_f"] - state["theta_fr"])
    qroom = qnat + qfor + qrad + qmount
    expected = [
        (inputs["V"] - (rf + params["Rx"]) * state["if_"]) / params["Lf"],
        (inputs["V"] - r * state["ia"] - params["Vb0"] * si - params["Laf"] * state["if_"] * state["omega"]) / params["La"],
        (
            params["Laf"] * state["if_"] * state["ia"]
            - inputs["TL"]
            - params["B"] * state["omega"]
            - params["Tcb"]
            - params["cw"] * state["omega"] ** 2
            - (params["ch"] + params["ce"] * state["omega"]) * state["if_"] ** 2
        ) / params["J"],
        (pcu_a + piron - qaf - qab) / params["Ca"],
        (pcu_b + pbrdrop + pbrfric + qab - qbf) / params["Cb"],
        (pfield - qff) / params["Cf"],
        (pcu_p + pbear + pwind + qaf + qbf + qff - qroom) / params["Cfr"],
    ]

    actual = state_derivative(
        0.0,
        [state[key] for key in ["if_", "ia", "omega", "theta_a", "theta_b", "theta_f", "theta_fr"]],
        params,
        inputs,
    )
    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)
    helper = compute_helper_equations(params, state, inputs)
    assert "Gs" not in helper
    assert "Q_room" in helper
