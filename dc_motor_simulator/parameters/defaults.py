"""Default parameter values and state inputs for the DC shunt motor model."""

from __future__ import annotations

from copy import deepcopy

PARAMETER_METADATA = {
    # Electrical parameters
    "Rf0": {"name": "Field winding resistance", "category": "Electrical", "units": "Ohm", "source": "literature-derived", "status": "estimated", "value": 28.0},
    "Rx": {"name": "Field rheostat resistance", "category": "Electrical", "units": "Ohm", "source": "manufacturer-provided", "status": "estimated", "value": 0.5},
    "Lf": {"name": "Field inductance", "category": "Electrical", "units": "H", "source": "literature-derived", "status": "estimated", "value": 0.45},
    "alpha": {"name": "Field resistance temperature coefficient", "category": "Electrical", "units": "1/K", "source": "literature-derived", "status": "estimated", "value": 0.0039},
    "Ra0": {"name": "Armature winding resistance", "category": "Electrical", "units": "Ohm", "source": "manufacturer-provided", "status": "estimated", "value": 0.92},
    "Rip": {"name": "Interpole resistance", "category": "Electrical", "units": "Ohm", "source": "manufacturer-provided", "status": "estimated", "value": 0.08},
    "Rbr": {"name": "Brush contact resistance", "category": "Electrical", "units": "Ohm", "source": "literature-derived", "status": "estimated", "value": 0.05},
    "Rsl": {"name": "Stray-load resistance", "category": "Electrical", "units": "Ohm", "source": "literature-derived", "status": "estimated", "value": 0.03},
    "alpha_e": {"name": "Effective armature resistance temperature coefficient", "category": "Electrical", "units": "1/K", "source": "literature-derived", "status": "estimated", "value": 0.0035},
    "Vb0": {"name": "Brush voltage drop", "category": "Electrical", "units": "V", "source": "literature-derived", "status": "estimated", "value": 1.1},
    "La": {"name": "Armature inductance", "category": "Electrical", "units": "H", "source": "literature-derived", "status": "estimated", "value": 0.12},
    "theta0": {"name": "Reference temperature", "category": "Electrical", "units": "°C", "source": "literature-derived", "status": "estimated", "value": 25.0},

    # Mechanical parameters
    "Laf": {"name": "Field-armature mutual inductance", "category": "Mechanical", "units": "H", "source": "literature-derived", "status": "estimated", "value": 0.19},
    "J": {"name": "Total moment of inertia", "category": "Mechanical", "units": "kg.m2", "source": "manufacturer-provided", "status": "estimated", "value": 0.85},
    "B": {"name": "Viscous friction coefficient", "category": "Mechanical", "units": "N.m.s/rad", "source": "literature-derived", "status": "estimated", "value": 0.015},
    "Tcb": {"name": "Coulomb brush friction torque", "category": "Mechanical", "units": "N.m", "source": "literature-derived", "status": "estimated", "value": 0.18},
    "cw": {"name": "Windage coefficient", "category": "Mechanical", "units": "N.m.s2/rad2", "source": "literature-derived", "status": "estimated", "value": 0.0009},
    "ch": {"name": "Hysteresis torque coefficient", "category": "Mechanical", "units": "N.m", "source": "literature-derived", "status": "estimated", "value": 0.09},
    "ce": {"name": "Eddy-current torque coefficient", "category": "Mechanical", "units": "N.m.s/rad", "source": "literature-derived", "status": "estimated", "value": 0.003},

    # Thermal parameters
    "Ca": {"name": "Armature heat capacity", "category": "Thermal", "units": "J/K", "source": "literature-derived", "status": "estimated", "value": 5200.0},
    "Cb": {"name": "Brush heat capacity", "category": "Thermal", "units": "J/K", "source": "literature-derived", "status": "estimated", "value": 1500.0},
    "Cf": {"name": "Field heat capacity", "category": "Thermal", "units": "J/K", "source": "literature-derived", "status": "estimated", "value": 12500.0},
    "Cfr": {"name": "Frame heat capacity", "category": "Thermal", "units": "J/K", "source": "literature-derived", "status": "estimated", "value": 38000.0},
    "Ga0": {"name": "Armature-frame conductance", "category": "Thermal", "units": "W/K", "source": "literature-derived", "status": "estimated", "value": 10.0},
    "Gb0": {"name": "Brush-frame conductance", "category": "Thermal", "units": "W/K", "source": "literature-derived", "status": "estimated", "value": 8.0},
    "ka": {"name": "Armature cooling speed gain", "category": "Thermal", "units": "1/(rad/s)", "source": "literature-derived", "status": "estimated", "value": 0.05},
    "kb": {"name": "Brush cooling speed gain", "category": "Thermal", "units": "1/(rad/s)", "source": "literature-derived", "status": "estimated", "value": 0.03},
    "Gab": {"name": "Armature-brush conductance", "category": "Thermal", "units": "W/K", "source": "literature-derived", "status": "estimated", "value": 4.5},
    "Gf": {"name": "Field-frame conductance", "category": "Thermal", "units": "W/K", "source": "literature-derived", "status": "estimated", "value": 7.5},
    "As": {"name": "External frame surface area", "category": "Thermal", "units": "m2", "source": "literature-derived", "status": "estimated", "value": 0.35},
    "eps_s": {"name": "Surface emissivity", "category": "Thermal", "units": "-", "source": "literature-derived", "status": "estimated", "value": 0.82},
    "hn0": {"name": "Natural convection coefficient", "category": "Thermal", "units": "W/m2/K", "source": "literature-derived", "status": "estimated", "value": 5.2},
    "kF": {"name": "Forced convection gain", "category": "Thermal", "units": "W.s/m3/K", "source": "literature-derived", "status": "estimated", "value": 0.08},
    "Gm": {"name": "Mounting conduction", "category": "Thermal", "units": "W/K", "source": "literature-derived", "status": "estimated", "value": 1.5},
    "phi_a": {"name": "Armature loss fraction", "category": "Thermal", "units": "-", "source": "calculated", "status": "calculated", "value": 0.74},
    "phi_b": {"name": "Brush loss fraction", "category": "Thermal", "units": "-", "source": "calculated", "status": "calculated", "value": 0.08},
    "phi_p": {"name": "Interpole loss fraction", "category": "Thermal", "units": "-", "source": "calculated", "status": "calculated", "value": 0.18},
    "phi_w": {"name": "Windage heat fraction", "category": "Thermal", "units": "-", "source": "literature-derived", "status": "estimated", "value": 0.12},

    # Physical and numerical constants
    "sigma": {"name": "Stefan-Boltzmann constant", "category": "Physical and numerical", "units": "W/m2.K4", "source": "measured", "status": "fixed", "value": 5.670374419e-8},
    "eps_i": {"name": "Current smoothing width", "category": "Physical and numerical", "units": "A", "source": "literature-derived", "status": "estimated", "value": 0.2},
    "eps_omega": {"name": "Speed smoothing width", "category": "Physical and numerical", "units": "rad/s", "source": "literature-derived", "status": "estimated", "value": 3.0},
    "delta": {"name": "Temperature smoothing offset", "category": "Physical and numerical", "units": "K", "source": "literature-derived", "status": "estimated", "value": 0.1},
    "DTref": {"name": "Reference temperature rise", "category": "Physical and numerical", "units": "K", "source": "literature-derived", "status": "estimated", "value": 12.0},

    # Derived parameter
    "R0": {"name": "Total armature-path resistance", "category": "Derived", "units": "Ohm", "source": "calculated", "status": "calculated", "value": 1.08},
}

DEFAULT_INPUTS = {
    "V": 220.0,
    "TL": 4.0,
    "Tamb": 25.0,
}

DEFAULT_INITIAL_STATE = {
    "if_": 0.85,
    "ia": 2.5,
    "omega": 100.0,
    "theta_a": 25.0,
    "theta_b": 25.0,
    "theta_f": 25.0,
    "theta_fr": 25.0,
}

DEFAULT_SIMULATION_SETTINGS = {
    "duration": 10.0,
    "dt": 0.1,
    "solver_method": "BDF",
    "playback_speed": 1.0,
}


def get_default_parameter_values():
    """Return a deep copy of the default parameter dictionary."""
    data = {name: meta["value"] for name, meta in PARAMETER_METADATA.items()}
    data["R0"] = data["Ra0"] + data["Rip"] + data["Rbr"] + data["Rsl"]
    return data
