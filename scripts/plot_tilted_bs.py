"""Plotly figures of a |1>|1> run in the style of Tomev/gbs-compiler experiments/tilted_bs (Apache-2.0).

Usage (from the project root):
    .venv/bin/python scripts/plot_tilted_bs.py Results/<run> [--backends FakeVLQ VLQ]

Writes into the run folder:
    tilted_bs_plot.*               raw P(2,0), P(1,1), P(0,2) vs n (theta = n pi / 32)
    tilted_bs_plot_postselected.*  the same after post-selecting n1 + n2 = 2
    tilted_bs_fidelity_plot.*      SBC with the exact distribution, raw and post-selected
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS = ["#2A78D6", "#EB6834", "#1BAF7A", "#EDA100", "#E87BA4", "#008300", "#6250D6", "#E34948"]
DASH_SIZES = [20, 16, 12, 10, 8, 8, 8, 8]  # narrowing dashes keep overlapping series visible
STATES = [("(2, 0)", "1000"), ("(1, 1)", "0101"), ("(0, 2)", "0010")]  # Fock key, q0q1q2q3 bits
PANEL_HEIGHT, PANEL_GAP, MARGIN_TOP, MARGIN_BOTTOM, LEGEND_OFFSET, TITLE_OFFSET = 182, 52, 140, 65, 30, 18

# Series: name -> {"n": [...], "y": [...]}
Series = dict[str, dict[str, list[float]]]


def theory(n: np.ndarray) -> dict[str, list[float]]:
    """Exact |1>|1> probabilities: P(2,0) = P(0,2) = sin^2(2 theta) / 2, P(1,1) = cos^2(2 theta)."""
    t = n * np.pi / 32
    s = 0.5 * np.sin(2 * t) ** 2
    return {"(2, 0)": list(s), "(1, 1)": list(np.cos(2 * t) ** 2), "(0, 2)": list(s)}


def load(run: Path, backends: list[str] | None) -> tuple[dict, dict[str, list[dict]], dict[str, list[int]]]:
    """Config, metrics rows and n values per backend."""
    r = json.loads((run / "results.json").read_text())
    cfg, metrics = r["config"], r["metrics"]
    names = backends or list(metrics)
    per = cfg.get("thetas_per_backend", {})
    ns = {b: [round(t * 32 / np.pi) for t in per.get(b, cfg["thetas"])] for b in names}
    return cfg, {b: metrics[b] for b in names}, ns


def probability_series(metrics: dict[str, list[dict]], ns: dict[str, list[int]], kind: str, state: str) -> Series:
    all_n = sorted({n for v in ns.values() for n in v})
    out: Series = {"Theoretical": {"n": all_n, "y": theory(np.array(all_n, float))[state]}}
    for b, rows in metrics.items():
        out[b] = {"n": ns[b], "y": [row[kind].get(state, 0.0) for row in rows]}
    return out


def fidelity_series(metrics: dict[str, list[dict]], ns: dict[str, list[int]], kind: str) -> Series:
    return {b: {"n": ns[b], "y": [row[f"SBC_{kind}"] for row in rows]} for b, rows in metrics.items()}


def add_panel(fig: go.Figure, series: Series, colors: dict[str, str], row: int, y_title: str) -> None:
    for (name, s), size in zip(series.items(), DASH_SIZES):
        fig.add_trace(go.Scatter(
            x=s["n"], y=s["y"], name=name, legendgroup=name, showlegend=False, mode="markers",
            marker=dict(symbol="line-ew", size=size, line=dict(color=colors[name], width=2)),
            hovertemplate="%{y:.3f}",
        ), row=row, col=1)
    fig.update_yaxes(title_text=y_title, range=[-0.045, 1.045], row=row, col=1)


def figure(panels: list[tuple[str, Series]], colors: dict[str, str], title: str, y_title: str) -> go.Figure:
    rows = len(panels)
    plot_height = rows * PANEL_HEIGHT + (rows - 1) * PANEL_GAP
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=PANEL_GAP / plot_height,
                        subplot_titles=[p[0] for p in panels])
    for row, (_, series) in enumerate(panels, start=1):
        add_panel(fig, series, colors, row, y_title)
    for name in dict.fromkeys(k for _, s in panels for k in s):  # one legend entry per series
        fig.add_trace(go.Scatter(x=[None], y=[None], name=name, legendgroup=name, mode="markers",
                                 marker=dict(symbol="line-ew", size=14, line=dict(color=colors[name], width=2))),
                      row=1, col=1)
    fig.update_xaxes(title_text="n (θ = n π / 32)", dtick=2, row=rows, col=1)
    height = MARGIN_TOP + plot_height + MARGIN_BOTTOM
    fig.update_layout(
        title=dict(text=title, y=1 - TITLE_OFFSET / height, yanchor="top"),
        template="plotly_white", height=height, width=980, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1 + LEGEND_OFFSET / plot_height, xanchor="right", x=1),
        margin=dict(t=MARGIN_TOP, b=MARGIN_BOTTOM, l=75, r=35),
    )
    return fig


def save(fig: go.Figure, run: Path, stem: str) -> None:
    fig.write_html(run / f"{stem}.html", include_plotlyjs=True)
    fig.write_image(run / f"{stem}.pdf")
    fig.write_image(run / f"{stem}.png", scale=3)
    print("saved:", run / f"{stem}.{{html,pdf,png}}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", type=Path, help="Results/<run> folder with results.json")
    ap.add_argument("--backends", nargs="+", help="subset and order of backends (default: all in the file)")
    args = ap.parse_args()

    cfg, metrics, ns = load(args.run, args.backends)
    if cfg["input"] != [1, 1]:
        raise SystemExit("only input |1>|1> is supported")
    colors = dict(zip(["Theoretical", *metrics], COLORS))
    layouts = cfg.get("layouts") or {}
    used = {b: tuple(layouts[b]) for b in metrics if b in layouts}
    if len(set(used.values())) == 1:
        qubits = f"{' and '.join(used)} on {', '.join(next(iter(used.values())))}"
    else:
        qubits = "; ".join(f"{b} on {', '.join(q)}" for b, q in used.items())
    sub = f"<br><sup>{cfg['circuit']} circuit, {cfg['shots']} shots" + (f"; {qubits}" if qubits else "") + "</sup>"

    for kind, stem, label in [("raw", "tilted_bs_plot", "state probabilities"),
                              ("post", "tilted_bs_plot_postselected", "post-selected state probabilities")]:
        panels = [(f"State |{bits}⟩ = |{fock[1:-1].replace(' ', '')}⟩<sub>Fock</sub>",
                   probability_series(metrics, ns, kind, fock)) for fock, bits in STATES]
        save(figure(panels, colors, f"Tilted beam splitter: {label}{sub}", "Probability"), args.run, stem)

    panels = [("All 16 outcomes", fidelity_series(metrics, ns, "raw")),
              ("Post-selected n₁ + n₂ = 2", fidelity_series(metrics, ns, "post"))]
    save(figure(panels, colors, f"Tilted beam splitter: fidelity with the theoretical state{sub}",
                "Fidelity"), args.run, "tilted_bs_fidelity_plot")


if __name__ == "__main__":
    main()
