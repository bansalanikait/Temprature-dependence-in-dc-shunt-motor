"""Central parameter management and persistence for the simulator."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from dc_motor_simulator.parameters.defaults import PARAMETER_METADATA, get_default_parameter_values


class ParameterManager:
    """Holds all configurable parameters and derived values."""

    def __init__(self, values: dict[str, float] | None = None, *, auto_phi: bool = True):
        self.metadata = deepcopy(PARAMETER_METADATA)
        self.values = deepcopy(get_default_parameter_values()) if values is None else dict(values)
        self.auto_phi = auto_phi
        self._sync_derived_parameters()

    def _sync_derived_parameters(self):
        self.values["R0"] = self.values.get("Ra0", 0.0) + self.values.get("Rip", 0.0) + self.values.get("Rbr", 0.0) + self.values.get("Rsl", 0.0)
        if self.auto_phi:
            total = self.values["R0"]
            if total > 0:
                self.values["phi_a"] = (self.values.get("Ra0", 0.0) + self.values.get("Rsl", 0.0)) / total
                self.values["phi_b"] = self.values.get("Rbr", 0.0) / total
                self.values["phi_p"] = self.values.get("Rip", 0.0) / total

    def get(self, name: str) -> float:
        return float(self.values.get(name, 0.0))

    def set(self, name: str, value: float) -> None:
        self.values[name] = float(value)
        if name in {"Ra0", "Rip", "Rbr", "Rsl"}:
            self._sync_derived_parameters()
        if self.auto_phi and name in {"phi_a", "phi_b", "phi_p"}:
            self._sync_derived_parameters()

    def as_dict(self) -> dict[str, float]:
        return deepcopy(self.values)

    def export_to_json(self, path: str | Path) -> None:
        payload = {"auto_phi": self.auto_phi, "parameters": self.values}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    def import_from_json(self, path: str | Path) -> None:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if isinstance(payload, dict) and "parameters" in payload:
            self.values = dict(payload["parameters"])
            self.auto_phi = bool(payload.get("auto_phi", self.auto_phi))
        else:
            self.values = dict(payload)
        self._sync_derived_parameters()

    def metadata_for(self, name: str) -> dict[str, Any]:
        meta = deepcopy(self.metadata.get(name, {}))
        meta["value"] = self.values.get(name, meta.get("value"))
        return meta

    def reset(self) -> None:
        self.values = deepcopy(get_default_parameter_values())
        self._sync_derived_parameters()

    def calculate_phi(self) -> dict[str, float]:
        total = self.values.get("R0", 0.0)
        if total <= 0:
            raise ValueError("Total armature-path resistance R0 must be positive to calculate heat-loss fractions.")
        phi_a = (self.values.get("Ra0", 0.0) + self.values.get("Rsl", 0.0)) / total
        phi_b = self.values.get("Rbr", 0.0) / total
        phi_p = self.values.get("Rip", 0.0) / total
        return {"phi_a": phi_a, "phi_b": phi_b, "phi_p": phi_p}

    def validate_loss_fractions(self) -> list[str]:
        warnings: list[str] = []
        phi_a = self.values.get("phi_a", 0.0)
        phi_b = self.values.get("phi_b", 0.0)
        phi_p = self.values.get("phi_p", 0.0)
        total = phi_a + phi_b + phi_p
        if not (abs(total - 1.0) < 1e-6):
            warnings.append("Heat-loss fractions do not sum to 1.0; the physical energy partition is inconsistent.")
        if phi_a < 0 or phi_b < 0 or phi_p < 0:
            warnings.append("Heat-loss fractions must be non-negative.")
        return warnings
