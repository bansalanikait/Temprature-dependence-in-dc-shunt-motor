"""Validation utilities for motor parameters and numerical output."""

from __future__ import annotations

import math
from typing import Iterable


def _finite(value: float, name: str) -> list[str]:
    if not math.isfinite(float(value)):
        return [f"{name} is not finite."]
    return []


def validate_parameter_set(params: dict[str, float]) -> list[str]:
    """Return warnings for questionable or invalid parameters."""
    warnings: list[str] = []

    positive_checks = {
        "Ca": "Armature heat capacity",
        "Cb": "Brush heat capacity",
        "Cf": "Field heat capacity",
        "Cfr": "Frame heat capacity",
        "Lf": "Field inductance",
        "La": "Armature inductance",
        "J": "Moment of inertia",
        "eps_i": "Current smoothing width",
        "delta": "Temperature smoothing offset",
        "DTref": "Reference temperature rise",
    }
    for name, label in positive_checks.items():
        value = float(params.get(name, 0.0))
        if value <= 0:
            warnings.append(f"{label} must be positive.")
        warnings.extend(_finite(value, label))

    for name in ["Ga0", "Gb0", "Gab", "Gf", "As", "hn0", "kF", "Gm"]:
        value = float(params.get(name, 0.0))
        if value < 0:
            warnings.append(f"{name} must be non-negative.")
        warnings.extend(_finite(value, name))

    for name in ["Rf0", "Rx", "Ra0", "Rip", "Rbr", "Rsl", "R0"]:
        value = float(params.get(name, 0.0))
        if value < 0:
            warnings.append(f"{name} must be physically valid and non-negative.")
        warnings.extend(_finite(value, name))

    if not (0.0 <= float(params.get("eps_s", 0.0)) <= 1.0):
        warnings.append("Surface emissivity must be between 0 and 1.")

    phi_a = float(params.get("phi_a", 0.0))
    phi_b = float(params.get("phi_b", 0.0))
    phi_p = float(params.get("phi_p", 0.0))
    if phi_a < 0 or phi_b < 0 or phi_p < 0:
        warnings.append("Heat-loss fractions must be non-negative.")
    if abs((phi_a + phi_b + phi_p) - 1.0) > 1e-6:
        warnings.append("Heat-loss fractions do not sum to 1, which is inconsistent with the specified energy partition.")

    for key in ["sigma", "alpha", "alpha_e"]:
        value = float(params.get(key, 0.0))
        if value < 0:
            warnings.append(f"{key} should not be negative in a physically realistic DC motor model.")
        warnings.extend(_finite(value, key))

    if float(params.get("Vb0", 0.0)) < 0:
        warnings.append("Brush voltage drop should be positive in the forward-rotation model.")

    return warnings


def validate_state_vector(y: Iterable[float]) -> list[str]:
    warnings: list[str] = []
    values = list(y)
    names = ["if_", "ia", "omega", "theta_a", "theta_b", "theta_f", "theta_fr"]
    for idx, name in enumerate(names):
        value = float(values[idx])
        if not math.isfinite(value):
            warnings.append(f"{name} is not finite.")
        if name in {"theta_a", "theta_b", "theta_f", "theta_fr"} and abs(value) > 1e6:
            warnings.append(f"{name} is outside a numerically representable range.")
    return warnings
