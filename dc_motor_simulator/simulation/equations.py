"""Seven-equation DC shunt motor model."""

from __future__ import annotations

import math
from typing import Mapping


def smooth_sign(value: float, width: float) -> float:
    return math.tanh(value / width) if width > 0 else 0.0


def armature_resistance(theta_a: float, params: Mapping[str, float]) -> float:
    r0 = float(params.get("R0", 0.0))
    alpha_e = float(params.get("alpha_e", 0.0))
    theta0 = float(params.get("theta0", 0.0))
    return r0 * (1.0 + alpha_e * (theta_a - theta0))


def field_resistance(theta_f: float, params: Mapping[str, float]) -> float:
    rf0 = float(params.get("Rf0", 0.0))
    alpha = float(params.get("alpha", 0.0))
    theta0 = float(params.get("theta0", 0.0))
    return rf0 * (1.0 + alpha * (theta_f - theta0))


def armature_frame_conductance(omega: float, params: Mapping[str, float]) -> float:
    ga0 = float(params.get("Ga0", 0.0))
    ka = float(params.get("ka", 0.0))
    return ga0 * (1.0 + ka * omega)


def brush_frame_conductance(omega: float, params: Mapping[str, float]) -> float:
    gb0 = float(params.get("Gb0", 0.0))
    kb = float(params.get("kb", 0.0))
    return gb0 * (1.0 + kb * omega)


def compute_helper_equations(params: Mapping[str, float], state: Mapping[str, float], inputs: Mapping[str, float]) -> dict[str, float]:
    """Compute all helper quantities used in the coupled model."""
    if_ = float(state["if_"])
    ia = float(state["ia"])
    omega = float(state["omega"])
    theta_a = float(state["theta_a"])
    theta_b = float(state["theta_b"])
    theta_f = float(state["theta_f"])
    theta_fr = float(state["theta_fr"])
    tamb = float(inputs["Tamb"])
    eps_i = float(params.get("eps_i", 1e-9))
    delta = float(params.get("delta", 0.1))
    dtref = float(params.get("DTref", 1.0))
    as_surface = float(params.get("As", 0.0))
    hn0 = float(params.get("hn0", 0.0))
    kf = float(params.get("kF", 0.0))
    g_m = float(params.get("Gm", 0.0))
    sigma = float(params.get("sigma", 5.670374419e-8))
    eps_s = float(params.get("eps_s", 0.0))

    s_ia = smooth_sign(ia, eps_i)

    r_a = armature_resistance(theta_a, params)
    r_f = field_resistance(theta_f, params)
    ga = armature_frame_conductance(omega, params)
    gb = brush_frame_conductance(omega, params)

    phi_a = float(params.get("phi_a", 0.0))
    phi_b = float(params.get("phi_b", 0.0))
    phi_p = float(params.get("phi_p", 0.0))
    phi_w = float(params.get("phi_w", 0.0))
    v_b0 = float(params.get("Vb0", 0.0))
    tcb = float(params.get("Tcb", 0.0))
    b = float(params.get("B", 0.0))
    cw = float(params.get("cw", 0.0))
    ch = float(params.get("ch", 0.0))
    ce = float(params.get("ce", 0.0))
    gab = float(params.get("Gab", 0.0))
    gf = float(params.get("Gf", 0.0))

    dth = theta_fr - tamb
    d_t_s = math.sqrt(dth ** 2 + delta ** 2)
    q_nat = hn0 * as_surface * (d_t_s / dtref) ** 0.25 * dth
    q_for = kf * omega * as_surface * dth
    t_fr_k = theta_fr + 273.15
    t_air_k = tamb + 273.15
    q_rad = eps_s * sigma * as_surface * (t_fr_k ** 4 - t_air_k ** 4)
    q_mount = g_m * dth
    q_room = q_nat + q_for + q_rad + q_mount

    p_cu_a = phi_a * r_a * ia ** 2
    p_cu_b = phi_b * r_a * ia ** 2
    p_cu_p = phi_p * r_a * ia ** 2
    p_iron = (ch * omega + ce * omega ** 2) * if_ ** 2
    p_brdrop = v_b0 * ia * s_ia
    p_brfric = tcb * omega
    p_field = r_f * if_ ** 2
    p_bear = b * omega ** 2
    p_wind = phi_w * cw * omega ** 3

    q_af = ga * (theta_a - theta_fr)
    q_bf = gb * (theta_b - theta_fr)
    q_ab = gab * (theta_a - theta_b)
    q_ff = gf * (theta_f - theta_fr)

    return {
        "s_ia": s_ia,
        "R_a": r_a,
        "R_f": r_f,
        "Ga": ga,
        "Gb": gb,
        "P_cu_a": p_cu_a,
        "P_cu_b": p_cu_b,
        "P_cu_p": p_cu_p,
        "P_iron": p_iron,
        "P_brdrop": p_brdrop,
        "P_brfric": p_brfric,
        "P_field": p_field,
        "P_bear": p_bear,
        "P_wind": p_wind,
        "Q_af": q_af,
        "Q_bf": q_bf,
        "Q_ab": q_ab,
        "Q_ff": q_ff,
        "dth": dth,
        "dT_s": d_t_s,
        "Q_nat": q_nat,
        "Q_for": q_for,
        "Q_rad": q_rad,
        "Q_mount": q_mount,
        "Q_room": q_room,
    }


def state_derivative(t: float, y: list[float], params: Mapping[str, float], inputs: Mapping[str, float]) -> list[float]:
    """Evaluate the seven coupled differential equations."""
    state = {
        "if_": y[0],
        "ia": y[1],
        "omega": y[2],
        "theta_a": y[3],
        "theta_b": y[4],
        "theta_f": y[5],
        "theta_fr": y[6],
    }
    v = float(inputs["V"])
    tl = float(inputs["TL"])
    if_ = state["if_"]
    ia = state["ia"]
    omega = state["omega"]

    helper = compute_helper_equations(params, state, inputs)
    s_ia = helper["s_ia"]
    r_a = helper["R_a"]
    r_f = helper["R_f"]
    v_b0 = float(params.get("Vb0", 0.0))
    laf = float(params.get("Laf", 0.0))
    j = float(params.get("J", 0.0))
    cw = float(params.get("cw", 0.0))
    ch = float(params.get("ch", 0.0))
    ce = float(params.get("ce", 0.0))
    rx = float(params.get("Rx", 0.0))
    l_f = float(params.get("Lf", 0.0))
    l_a = float(params.get("La", 0.0))
    ca = float(params.get("Ca", 0.0))
    cb = float(params.get("Cb", 0.0))
    cf = float(params.get("Cf", 0.0))
    cfr = float(params.get("Cfr", 0.0))

    # Equation 1: field circuit
    dif_dt = (v - (r_f + rx) * if_) / l_f

    # Equation 2: armature circuit
    dia_dt = (v - r_a * ia - v_b0 * s_ia - laf * if_ * omega) / l_a

    # Equation 3: mechanical equation
    d_omega_dt = (
        laf * if_ * ia
        - tl
        - float(params.get("B", 0.0)) * omega
        - float(params.get("Tcb", 0.0))
        - cw * omega ** 2
        - (ch + ce * omega) * if_ ** 2
    ) / j

    # Equation 4: armature thermal equation
    dtheta_a_dt = (
        helper["P_cu_a"]
        + helper["P_iron"]
        - helper["Q_af"]
        - helper["Q_ab"]
    ) / ca

    # Equation 5: brush thermal equation
    dtheta_b_dt = (
        helper["P_cu_b"]
        + helper["P_brdrop"]
        + helper["P_brfric"]
        + helper["Q_ab"]
        - helper["Q_bf"]
    ) / cb

    # Equation 6: field thermal equation
    dtheta_f_dt = (helper["P_field"] - helper["Q_ff"]) / cf

    # Equation 7: frame thermal equation
    dtheta_fr_dt = (
        helper["P_cu_p"]
        + helper["P_bear"]
        + helper["P_wind"]
        + helper["Q_af"]
        + helper["Q_bf"]
        + helper["Q_ff"]
        - helper["Q_room"]
    ) / cfr

    return [dif_dt, dia_dt, d_omega_dt, dtheta_a_dt, dtheta_b_dt, dtheta_f_dt, dtheta_fr_dt]
