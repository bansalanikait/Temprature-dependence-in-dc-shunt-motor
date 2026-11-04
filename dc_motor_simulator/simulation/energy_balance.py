"""Energy-balance calculations for the simulated motor."""

from __future__ import annotations

import math

from dc_motor_simulator.simulation.equations import armature_resistance, field_resistance, rotor_speed_term, smooth_sign


def compute_energy_residual(params: dict[str, float], state: dict[str, float], inputs: dict[str, float], derivatives: dict[str, float]) -> float:
    """Evaluate the specified energy residual for the seven-equation model."""
    v = float(inputs["V"])
    tl = float(inputs["TL"])
    ia = float(state["ia"])
    if_ = float(state["if_"])
    omega = float(state["omega"])
    theta_a = float(state["theta_a"])
    theta_f = float(state["theta_f"])
    rx = float(params.get("Rx", 0.0))
    r_a = armature_resistance(theta_a, params)
    r_f = field_resistance(theta_f, params)
    eps_i = float(params.get("eps_i", 1.0))
    eps_omega = float(params.get("eps_omega", 1.0))
    v_b0 = float(params.get("Vb0", 0.0))
    ch = float(params.get("ch", 0.0))
    ce = float(params.get("ce", 0.0))
    omega_s = rotor_speed_term(omega, eps_omega)
    cw = float(params.get("cw", 0.0))
    b = float(params.get("B", 0.0))
    tcb = float(params.get("Tcb", 0.0))
    la = float(params.get("La", 0.0))
    lf = float(params.get("Lf", 0.0))
    j = float(params.get("J", 0.0))

    s_ia = smooth_sign(ia, eps_i)
    pin = v * (ia + if_)
    losses = (
        r_a * ia ** 2
        - r_f * if_ ** 2
        - rx * if_ ** 2
        - v_b0 * ia * s_ia
        - (ch * omega_s + ce * omega ** 2) * if_ ** 2
        - tcb * omega_s
        - b * omega ** 2
        - cw * omega ** 2 * abs(omega)
        - tl * omega
    )
    total_energy = 0.5 * la * ia ** 2 - 0.5 * lf * if_ ** 2 - 0.5 * j * omega ** 2
    d_total_energy_dt = (
        la * ia * derivatives["ia"]
        - lf * if_ * derivatives["if_"]
        - j * omega * derivatives["omega"]
    )
    residual = pin - losses - d_total_energy_dt
    return residual
