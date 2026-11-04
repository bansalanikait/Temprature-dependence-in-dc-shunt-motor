"""Simulation measurement recorder and exporter."""

from __future__ import annotations

import pandas as pd


class SimulationRecorder:
    def __init__(self):
        self.rows: list[dict[str, float]] = []

    def record(
        self,
        time_value: float,
        state: dict[str, float],
        inputs: dict[str, float],
        torque: float,
        power: float,
        mechanical_power: float = 0.0,
        energy_residual: float = 0.0,
    ) -> None:
        row = {
            "time_s": time_value,
            "if_A": state["if_"],
            "ia_A": state["ia"],
            "omega_rad_s": state["omega"],
            "theta_a_C": state["theta_a"],
            "theta_b_C": state["theta_b"],
            "theta_f_C": state["theta_f"],
            "theta_fr_C": state["theta_fr"],
            "V_V": inputs["V"],
            "TL_Nm": inputs["TL"],
            "Tamb_C": inputs["Tamb"],
            "torque_Nm": torque,
            "power_W": power,
            "electrical_input_power_W": power,
            "mechanical_power_W": mechanical_power,
            "energy_residual_W": energy_residual,
        }
        self.rows.append(row)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def export_csv(self, path: str, *, start_time: float | None = None, end_time: float | None = None) -> None:
        frame = self.to_dataframe()
        if start_time is not None:
            frame = frame[frame["time_s"] >= start_time]
        if end_time is not None:
            frame = frame[frame["time_s"] <= end_time]
        frame.to_csv(path, index=False)

    def clear(self) -> None:
        self.rows.clear()
