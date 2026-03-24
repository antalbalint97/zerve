from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go


SEGMENT_COLORS = {
    "Agent Builder": "#00b4d8",
    "Agent Runner": "#48cae4",
    "Manual Coder": "#90e0ef",
    "Viewer": "#555566",
    "Ghost": "#2d2d3a",
}
SEG_ORDER = ["Agent Builder", "Agent Runner", "Manual Coder", "Viewer", "Ghost"]

ADOPTION_COLORS = {
    "Korai adoptalo (<1h)": "#00b4d8",
    "Kesei adoptalo (1h+)": "#90e0ef",
    "Soha nem adoptalta": "#444455",
}
ADOPTION_ORDER = ["Korai adoptalo (<1h)", "Kesei adoptalo (1h+)", "Soha nem adoptalta"]

COHORT_COLORS = {
    "2025-09": "#00b4d8",
    "2025-10": "#48cae4",
    "2025-11": "#90e0ef",
    "2025-12": "#caf0f8",
}


def apply_dark_layout(fig: go.Figure, **layout_kwargs) -> go.Figure:
    fig.update_layout(template="plotly_dark", **layout_kwargs)
    return fig


def write_html(fig: go.Figure, path: str) -> None:
    Path(path).parent.mkdir(exist_ok=True)
    fig.write_html(path)
