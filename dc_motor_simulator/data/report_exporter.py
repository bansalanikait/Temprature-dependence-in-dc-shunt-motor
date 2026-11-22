"""Build a downloadable ZIP report containing simulation data and conditions."""

from __future__ import annotations

import html
import io
import json
import zipfile
from datetime import datetime
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from dc_motor_simulator.parameters.defaults import PARAMETER_METADATA


PARAMETER_SYMBOLS_HTML = {
    "Rf0": "<i>R</i><sub>f0</sub>",
    "Rx": "<i>R</i><sub>x</sub>",
    "Lf": "<i>L</i><sub>f</sub>",
    "alpha": "α",
    "Ra0": "<i>R</i><sub>a0</sub>",
    "Rip": "<i>R</i><sub>ip</sub>",
    "Rbr": "<i>R</i><sub>br</sub>",
    "Rsl": "<i>R</i><sub>sl</sub>",
    "alpha_e": "α<sub>e</sub>",
    "Vb0": "<i>V</i><sub>b0</sub>",
    "La": "<i>L</i><sub>a</sub>",
    "theta0": "θ<sub>0</sub>",
    "Laf": "<i>L</i><sub>af</sub>",
    "J": "<i>J</i>",
    "B": "<i>B</i>",
    "Tcb": "<i>T</i><sub>cb</sub>",
    "cw": "<i>c</i><sub>w</sub>",
    "ch": "<i>c</i><sub>h</sub>",
    "ce": "<i>c</i><sub>e</sub>",
    "Ca": "<i>C</i><sub>a</sub>",
    "Cb": "<i>C</i><sub>b</sub>",
    "Cf": "<i>C</i><sub>f</sub>",
    "Cfr": "<i>C</i><sub>fr</sub>",
    "Ga0": "<i>G</i><sub>a0</sub>",
    "Gb0": "<i>G</i><sub>b0</sub>",
    "ka": "<i>k</i><sub>a</sub>",
    "kb": "<i>k</i><sub>b</sub>",
    "Gab": "<i>G</i><sub>ab</sub>",
    "Gf": "<i>G</i><sub>f</sub>",
    "As": "<i>A</i><sub>s</sub>",
    "eps_s": "ε<sub>s</sub>",
    "hn0": "<i>h</i><sub>n0</sub>",
    "kF": "<i>k</i><sub>F</sub>",
    "Gm": "<i>G</i><sub>m</sub>",
    "phi_a": "φ<sub>a</sub>",
    "phi_b": "φ<sub>b</sub>",
    "phi_p": "φ<sub>p</sub>",
    "phi_w": "φ<sub>w</sub>",
    "sigma": "σ",
    "eps_i": "ε<sub>i</sub>",
    "delta": "δ",
    "DTref": "Δ<i>T</i><sub>ref</sub>",
    "R0": "<i>R</i><sub>0</sub>",
}


def _condition_rows(conditions: dict[str, Any]) -> str:
    rows = []
    for group, values in conditions.items():
        if isinstance(values, dict):
            for name, value in values.items():
                rows.append(
                    "<tr>"
                    f"<th>{html.escape(str(group))} · {html.escape(str(name))}</th>"
                    f"<td>{html.escape(str(value))}</td>"
                    "</tr>"
                )
        else:
            rows.append(
                f"<tr><th>{html.escape(str(group))}</th>"
                f"<td>{html.escape(str(values))}</td></tr>"
            )
    return "\n".join(rows)


def _parameter_rows(parameters: dict[str, float]) -> str:
    rows = []
    for name, value in parameters.items():
        metadata = PARAMETER_METADATA.get(name, {})
        symbol = PARAMETER_SYMBOLS_HTML.get(name, html.escape(str(name)))
        rows.append(
            "<tr>"
            f"<td class=\"symbol\">{symbol}</td>"
            f"<td>{html.escape(str(metadata.get('name', name)))}</td>"
            f"<td>{html.escape(str(value))}</td>"
            f"<td>{html.escape(str(metadata.get('units', '')))}</td>"
            f"<td>{html.escape(str(metadata.get('source', 'user-configured')))}</td>"
            f"<td>{html.escape(str(metadata.get('status', 'configured')))}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _summary_rows(frame: pd.DataFrame) -> str:
    result_columns = [
        column
        for column in frame.columns
        if column not in {"time_s", "V_V", "TL_Nm", "Tamb_C"}
    ]
    rows = []
    for column in result_columns:
        numeric = pd.to_numeric(frame[column], errors="coerce").dropna()
        if numeric.empty:
            continue
        rows.append(
            "<tr>"
            f"<td>{html.escape(column)}</td>"
            f"<td>{numeric.iloc[0]:.6g}</td>"
            f"<td>{numeric.min():.6g}</td>"
            f"<td>{numeric.max():.6g}</td>"
            f"<td>{numeric.iloc[-1]:.6g}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _balance_check_rows(frame: pd.DataFrame) -> str:
    checks = [
        ("Thermal balance residual", "thermal_balance_residual_W"),
        ("Whole-motor balance residual", "energy_residual_W"),
    ]
    rows = []
    for label, column in checks:
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            continue
        worst_index = values.abs().idxmax()
        rows.append(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{abs(values.loc[worst_index]):.6e} W</td>"
            f"<td>{float(frame.loc[worst_index, 'time_s']):.6g} s</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _charts_html(frame: pd.DataFrame) -> str:
    figures: list[go.Figure] = []

    def add_line_chart(title: str, y_title: str, traces: list[tuple[str, str, str]]):
        figure = go.Figure()
        for label, column, color in traces:
            figure.add_trace(
                go.Scatter(
                    x=frame["time_s"],
                    y=frame[column],
                    name=label,
                    mode="lines",
                    line=dict(width=2, color=color),
                )
            )
        figure.update_layout(
            template="plotly_white",
            title=title,
            height=390,
            hovermode="x unified",
            xaxis_title="Simulation time (s)",
            yaxis_title=y_title,
            margin=dict(l=45, r=30, t=55, b=45),
            legend=dict(orientation="h", y=1.12),
        )
        figures.append(figure)

    add_line_chart(
        "Current response",
        "Current (A)",
        [
            ("Armature current", "ia_A", "#2563eb"),
            ("Field current", "if_A", "#0f766e"),
        ],
    )
    add_line_chart(
        "Motor speed",
        "Angular speed (rad/s)",
        [("Speed", "omega_rad_s", "#7c3aed")],
    )
    add_line_chart(
        "Temperature response",
        "Temperature (°C)",
        [
            ("Armature", "theta_a_C", "#2563eb"),
            ("Brush", "theta_b_C", "#0f766e"),
            ("Field", "theta_f_C", "#d97706"),
            ("Frame", "theta_fr_C", "#7c3aed"),
        ],
    )
    add_line_chart(
        "Electromagnetic torque",
        "Torque (N·m)",
        [("Torque", "torque_Nm", "#2563eb")],
    )
    add_line_chart(
        "Power",
        "Power (W)",
        [
            ("Electrical input", "electrical_input_power_W", "#2563eb"),
            ("Mechanical output", "mechanical_power_W", "#0f766e"),
        ],
    )
    add_line_chart(
        "Whole-motor balance residual",
        "Residual (W)",
        [("Residual", "energy_residual_W", "#d97706")],
    )
    if "thermal_balance_residual_W" in frame:
        add_line_chart(
            "Thermal balance residual",
            "Residual (W)",
            [("Residual", "thermal_balance_residual_W", "#0f766e")],
        )

    fragments = []
    for index, figure in enumerate(figures):
        fragments.append(
            pio.to_html(
                figure,
                full_html=False,
                include_plotlyjs=True if index == 0 else False,
                config={"responsive": True, "displaylogo": False},
            )
        )
    return "\n".join(fragments)


def build_report_archive(
    frame: pd.DataFrame,
    conditions: dict[str, Any],
    parameters: dict[str, float],
) -> bytes:
    """Create a ZIP containing a self-contained HTML report, CSV, and JSON."""
    if frame.empty:
        raise ValueError("Cannot export a report without simulation samples.")
    if not conditions or not parameters:
        raise ValueError("Run conditions and parameter values are required for the report.")

    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    report_id = generated_at.replace(":", "").replace("-", "")
    conditions = {**conditions, "report_generated_at": generated_at, "samples_recorded": len(frame)}
    params_json = {
        "report_id": report_id,
        "conditions": conditions,
        "parameters": parameters,
    }

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DC Motor Simulation Report</title>
<style>
body {{ font: 15px/1.55 Arial, sans-serif; color: #243247; background: #f3f6fa; margin: 0; }}
main {{ max-width: 1150px; margin: 28px auto; padding: 34px 42px; background: #fff; }}
header {{ border-bottom: 3px solid #1d4ed8; margin-bottom: 28px; padding-bottom: 18px; }}
h1 {{ margin: 0; color: #12305a; }} h2 {{ color: #12305a; margin-top: 34px; }}
.subtle {{ color: #596a7d; }} .note {{ border-left: 4px solid #d97706; padding: 10px 14px; background: #fff8e8; }}
table {{ width: 100%; border-collapse: collapse; margin: 14px 0 24px; }}
th, td {{ text-align: left; border: 1px solid #dce3ec; padding: 8px 10px; }}
th {{ background: #eef3f9; }} .chart {{ margin: 22px 0; }}
.symbol {{ font: 1.08rem Georgia, "Times New Roman", serif; white-space: nowrap; }}
.symbol sub {{ font-size: .75em; }}
@media print {{ body {{ background: #fff; }} main {{ margin: 0; max-width: none; padding: 0; }} }}
</style>
</head>
<body><main>
<header>
<h1>DC Shunt Motor Simulation Report</h1>
<p class="subtle">Electro-thermal seven-state model · Report ID {html.escape(report_id)}</p>
<p class="subtle">Generated {html.escape(generated_at)}</p>
</header>
<h2>Run summary</h2>
<table><tbody>{_condition_rows(conditions)}</tbody></table>
<div class="note">Numerical simulation output is not experimental validation. Thermal and whole-motor balance checks use the configured equations; their worst absolute residuals are listed below.</div>
<h2>Result statistics</h2>
<p class="subtle">Initial, minimum, maximum, and final sample values for each recorded state/output.</p>
<table><thead><tr><th>Quantity</th><th>Initial</th><th>Minimum</th><th>Maximum</th><th>Final</th></tr></thead>
<tbody>{_summary_rows(frame)}</tbody></table>
<h2>Balance checks</h2>
<p class="subtle">Worst absolute residual over all recorded samples; expected to be about 1e-9 W or smaller.</p>
<table><thead><tr><th>Check</th><th>Worst absolute residual</th><th>At time</th></tr></thead>
<tbody>{_balance_check_rows(frame)}</tbody></table>
<h2>Response plots</h2>
{_charts_html(frame)}
<h2>Configured motor parameters</h2>
<table><thead><tr><th>Symbol</th><th>Description</th><th>Value</th><th>Unit</th><th>Source</th><th>Status</th></tr></thead>
<tbody>{_parameter_rows(parameters)}</tbody></table>
<h2>Included files</h2>
<ul><li>simulation_results.csv: complete sampled time-series output</li><li>test_conditions.json: exact inputs, initial state, solver settings, and parameters</li></ul>
</main></body></html>"""

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("simulation_report.html", document)
        archive.writestr("simulation_results.csv", frame.to_csv(index=False))
        archive.writestr(
            "test_conditions.json",
            json.dumps(params_json, indent=2, ensure_ascii=False),
        )
    return buffer.getvalue()
