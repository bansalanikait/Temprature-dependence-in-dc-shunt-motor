"""Metric cards for the engineering dashboard."""

from __future__ import annotations

import streamlit as st


def metric_card(label: str, value: float, unit: str = "") -> None:
    formatted = f"{value:,.3f} {unit}".strip()
    st.metric(label, formatted)
