"""Tests for complete simulation report exports."""

import io
import json
import zipfile

import pandas as pd

from dc_motor_simulator.data.report_exporter import build_report_archive
from dc_motor_simulator.parameters.defaults import get_default_parameter_values


def test_report_archive_contains_report_data_and_exact_conditions():
    frame = pd.DataFrame(
        {
            "time_s": [0.0, 0.1],
            "if_A": [0.8, 0.9],
            "ia_A": [2.0, 2.5],
            "omega_rad_s": [10.0, 12.0],
            "theta_a_C": [25.0, 25.1],
            "theta_b_C": [25.0, 25.1],
            "theta_f_C": [25.0, 25.1],
            "theta_fr_C": [25.0, 25.1],
            "V_V": [220.0, 220.0],
            "TL_Nm": [4.0, 4.0],
            "Tamb_C": [25.0, 25.0],
            "torque_Nm": [0.3, 0.4],
            "power_W": [616.0, 748.0],
            "electrical_input_power_W": [616.0, 748.0],
            "mechanical_power_W": [3.0, 4.8],
            "energy_residual_W": [2.0, 1.0],
            "thermal_balance_residual_W": [0.5, -0.25],
        }
    )
    conditions = {
        "inputs": {"V": 220.0, "TL": 4.0, "Tamb": 25.0},
        "simulation_configuration": {"solver": "BDF"},
    }

    archive_data = build_report_archive(frame, conditions, get_default_parameter_values())

    with zipfile.ZipFile(io.BytesIO(archive_data)) as archive:
        assert set(archive.namelist()) == {
            "simulation_report.html",
            "simulation_results.csv",
            "test_conditions.json",
        }
        report = archive.read("simulation_report.html").decode("utf-8")
        dataset = archive.read("simulation_results.csv").decode("utf-8")
        config = json.loads(archive.read("test_conditions.json"))

    assert "DC Shunt Motor Simulation Report" in report
    assert "Power" in report
    assert "Balance checks" in report
    assert "Worst absolute residual" in report
    assert "<i>R</i><sub>f0</sub>" in report
    assert "<i>R</i><sub>x</sub>" in report
    assert "α<sub>e</sub>" in report
    assert "energy_residual_W" in dataset
    assert config["conditions"]["inputs"]["V"] == 220.0
    assert config["parameters"]["R0"] == get_default_parameter_values()["R0"]


def test_report_export_requires_samples_and_conditions():
    try:
        build_report_archive(pd.DataFrame(), {}, {})
    except ValueError as error:
        assert "without simulation samples" in str(error)
    else:
        raise AssertionError("Expected report export to reject an empty dataset.")
