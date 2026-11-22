"""Power-balance checks for the DC motor model."""

from __future__ import annotations

from typing import Mapping

from dc_motor_simulator.simulation.equations import compute_helper_equations


def compute_thermal_balance_residual(
    params: Mapping[str, float],
    state: Mapping[str, float],
    inputs: Mapping[str, float],
    derivatives: Mapping[str, float],
) -> float:
    """Return the thermal balance residual in watts."""
    helper = compute_helper_equations(params, state, inputs)
    stored_heat_rate = (
        float(params["Ca"]) * derivatives["theta_a"]
        + float(params["Cb"]) * derivatives["theta_b"]
        + float(params["Cf"]) * derivatives["theta_f"]
        + float(params["Cfr"]) * derivatives["theta_fr"]
    )
    generated_heat = sum(
        helper[name]
        for name in (
            "P_cu_a",
            "P_cu_b",
            "P_cu_p",
            "P_iron",
            "P_brdrop",
            "P_brfric",
            "P_field",
            "P_bear",
            "P_wind",
        )
    )
    return stored_heat_rate - generated_heat + helper["Q_room"]


def compute_energy_residual(
    params: Mapping[str, float],
    state: Mapping[str, float],
    inputs: Mapping[str, float],
    derivatives: Mapping[str, float],
) -> float:
    """Return the whole-motor energy residual in watts."""
    helper = compute_helper_equations(params, state, inputs)
    ia = float(state["ia"])
    if_ = float(state["if_"])
    omega = float(state["omega"])
    phi_w = float(params["phi_w"])
    cw = float(params["cw"])
    losses_and_storage = (
        float(params["Rx"]) * if_ ** 2
        + (1.0 - phi_w) * cw * omega ** 3
        + float(inputs["TL"]) * omega
        + float(params["La"]) * ia * derivatives["ia"]
        + float(params["Lf"]) * if_ * derivatives["if_"]
        + float(params["J"]) * omega * derivatives["omega"]
        + sum(
            helper[name]
            for name in (
                "P_cu_a",
                "P_cu_b",
                "P_cu_p",
                "P_iron",
                "P_brdrop",
                "P_brfric",
                "P_field",
                "P_bear",
                "P_wind",
            )
        )
    )
    electrical_input_power = float(inputs["V"]) * (ia + if_)
    return electrical_input_power - losses_and_storage
