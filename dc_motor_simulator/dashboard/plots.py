"""Plotly visualization helpers for simulation output."""

from __future__ import annotations

import plotly.graph_objects as go


PLOT_COLORS = ["#2563eb", "#0f766e", "#d97706", "#7c3aed"]


def _apply_engineering_style(fig: go.Figure, title: str, yaxis_title: str) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=17, color="#172b4d")),
        height=360,
        margin=dict(l=20, r=20, t=62, b=24),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(family="Arial, sans-serif", size=12, color="#344054"),
        xaxis=dict(
            title="Simulation time (s)",
            showgrid=True,
            gridcolor="#edf0f4",
            zeroline=False,
            showline=True,
            linecolor="#d0d5dd",
            ticks="outside",
        ),
        yaxis=dict(
            title=yaxis_title,
            showgrid=True,
            gridcolor="#edf0f4",
            zeroline=True,
            zerolinecolor="#d0d5dd",
            showline=True,
            linecolor="#d0d5dd",
            ticks="outside",
        ),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="#d0d5dd", font_size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(255,255,255,0.75)",
        ),
    )
    fig.update_xaxes(rangemode="tozero")
    return fig


def line_plot(x_values, y_values, *, title: str, yaxis_title: str, line_name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="lines",
            name=line_name,
            line=dict(width=2.4, color=PLOT_COLORS[0]),
            hovertemplate="%{y:.5g}<extra>%{fullData.name}</extra>",
        )
    )
    return _apply_engineering_style(fig, title, yaxis_title)


def combined_temperature_plot(time, armature, brush, field, frame):
    fig = go.Figure()
    for index, (name, series) in enumerate({
        "Armature": armature,
        "Brush": brush,
        "Field": field,
        "Frame": frame,
    }.items()):
        fig.add_trace(
            go.Scatter(
                x=time,
                y=series,
                mode="lines",
                name=name,
                line=dict(width=2.4, color=PLOT_COLORS[index]),
                hovertemplate="%{y:.3f} °C<extra>%{fullData.name}</extra>",
            )
        )
    return _apply_engineering_style(fig, "Temperature comparison", "Temperature (°C)")


def combined_current_plot(time, armature_current, field_current):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time,
            y=armature_current,
            mode="lines",
            name="Armature current",
            line=dict(width=2.4, color=PLOT_COLORS[0]),
            hovertemplate="%{y:.5g} A<extra>%{fullData.name}</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=field_current,
            mode="lines",
            name="Field current",
            line=dict(width=2.4, color=PLOT_COLORS[1]),
            hovertemplate="%{y:.5g} A<extra>%{fullData.name}</extra>",
        )
    )
    return _apply_engineering_style(fig, "Current comparison", "Current (A)")


def mechanical_power_plot(time, torque, power):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=torque, mode="lines", name="Electromagnetic torque", line=dict(width=2.4, color=PLOT_COLORS[0])))
    fig.add_trace(go.Scatter(x=time, y=power, mode="lines", name="Power", line=dict(width=2.4, color=PLOT_COLORS[1]), yaxis="y2"))
    _apply_engineering_style(fig, "Mechanical torque and power", "Torque (N·m)")
    fig.update_layout(yaxis2=dict(title="Power (W)", overlaying="y", side="right", showgrid=False))
    return fig
