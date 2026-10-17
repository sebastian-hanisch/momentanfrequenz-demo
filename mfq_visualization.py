"""Plotly-Figuren. Alle Achsen fest (fixedrange), damit Touch-Scrollen nicht am Diagramm hängen bleibt."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

import mfq_constants as C
from mfq_evaluation import Analysis


def _fixed(fig: go.Figure, height: int = 340, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _edges(fig: go.Figure, n: int) -> None:
    """Randbereiche (je EDGE Werte), die für die Fehlermaße ausgespart sind, schattieren."""
    fig.add_vrect(x0=0, x1=C.EDGE, fillcolor=C.COLOR_EDGE, line_width=0)
    fig.add_vrect(x0=n - 1 - C.EDGE, x1=n - 1, fillcolor=C.COLOR_EDGE, line_width=0)


def build_signal(a: Analysis) -> go.Figure:
    """Messsignal (mit Rauschen) und, falls Rauschen an ist, das saubere Signal."""
    t = np.arange(len(a.x))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=a.x, mode="lines", line=dict(color=C.COLOR_SIGNAL, width=1), name="Messsignal", hovertemplate="t = %{x}: %{y:.2f}<extra></extra>"))
    if a.settings.has_noise:
        fig.add_trace(go.Scatter(x=t, y=a.clean.x, mode="lines", line=dict(color=C.COLOR_TRUE, width=1, dash="dot"), name="sauberes Signal", hovertemplate="t = %{x}: %{y:.2f}<extra></extra>"))
    fig.update_xaxes(title_text="Zeit (Abtastwerte)")
    fig.update_yaxes(title_text="Signal")
    return _fixed(fig, height=260)


def build_envelope(a: Analysis) -> go.Figure:
    """Hüllkurve |z| des analytischen Signals gegen die wahre Hüllkurve (wo definiert)."""
    t = np.arange(len(a.env_est))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=a.env_est, mode="lines", line=dict(color=C.COLOR_EST, width=1.6), name="geschätzt |z|", hovertemplate="t = %{x}: %{y:.3f}<extra></extra>"))
    if a.clean.env_true is not None:
        fig.add_trace(go.Scatter(x=t, y=a.clean.env_true, mode="lines", line=dict(color=C.COLOR_TRUE, width=2, dash="dash"), name="wahr", hovertemplate="t = %{x}: %{y:.3f}<extra></extra>"))
    _edges(fig, len(t))
    fig.update_xaxes(title_text="Zeit (Abtastwerte)")
    fig.update_yaxes(title_text="Hüllkurve", rangemode="tozero")
    return _fixed(fig, height=280)


def build_frequency(a: Analysis) -> go.Figure:
    """Momentanfrequenz aus dem Phasenschritt gegen die wahre (wo definiert); bei zwei Komponenten die beiden Komponentenfrequenzen und das Mittel."""
    t = np.arange(len(a.f_est)) + 0.5
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=a.f_est, mode="lines", line=dict(color=C.COLOR_EST, width=1.4), name="geschätzt (Phasenschritt)", hovertemplate="t = %{x}: %{y:.3f}<extra></extra>"))
    if a.f_true is not None:
        fig.add_trace(go.Scatter(x=t, y=a.f_true, mode="lines", line=dict(color=C.COLOR_TRUE, width=2, dash="dash"), name="wahr", hovertemplate="t = %{x}: %{y:.3f}<extra></extra>"))
    else:
        for i, f in enumerate(a.clean.components):
            fig.add_hline(y=f, line=dict(color=C.COLOR_TRUE, width=1.2, dash="dash"), annotation_text=f"Komponente {i + 1}", annotation_position="top left")
        fig.add_hline(y=a.mean_f, line=dict(color=C.COLOR_AUX, width=1.2, dash="dot"), annotation_text="Mittel der Momentanfrequenz", annotation_position="bottom right")
    _edges(fig, len(a.x))
    fig.update_xaxes(title_text="Zeit (Abtastwerte)")
    fig.update_yaxes(title_text="Frequenz (Zyklen je Abtastwert)", range=[-0.5, 0.5])
    return _fixed(fig, height=340)


def build_chirp(rows) -> go.Figure:
    fig = go.Figure()
    for key, name in (("inner", "innen"), ("edge", "Rand (je 16 Werte)")):
        fig.add_trace(go.Scatter(x=[r["snr"] for r in rows], y=[r[key] for r in rows], mode="lines+markers", name=name, hovertemplate=name + ", %{x} dB: %{y:.4f}<extra></extra>"))
    fig.update_xaxes(title_text="Rauschabstand (dB)", autorange="reversed")
    fig.update_yaxes(title_text="RMSE der Momentanfrequenz", type="log")
    return _fixed(fig, height=340)


def build_bedrosian(rows) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[r["ratio"] for r in rows], y=[max(r["env_max"], 1e-5) for r in rows], mode="lines+markers", name="größter Hüllkurvenfehler", hovertemplate="fm/fc = %{x:.2f}: %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[r["ratio"] for r in rows], y=[max(r["f_rmse"], 1e-5) for r in rows], mode="lines+markers", name="RMSE der Momentanfrequenz", hovertemplate="fm/fc = %{x:.2f}: %{y:.4f}<extra></extra>"))
    fig.add_vline(x=1.0, line=dict(color="black", width=1, dash="dot"), annotation_text="fm = fc", annotation_position="top right")
    fig.update_xaxes(title_text="Modulationsfrequenz / Träger")
    fig.update_yaxes(title_text="Fehler (bis 1e-5 abgeschnitten)", type="log")
    return _fixed(fig, height=340)


def build_tones(rows) -> go.Figure:
    r_values = [r["r"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r_values, y=[r["f_max"] for r in rows], mode="lines+markers", name="größte Momentanfrequenz", hovertemplate="r = %{x}: %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=r_values, y=[r["f_min"] for r in rows], mode="lines+markers", name="kleinste Momentanfrequenz", hovertemplate="r = %{x}: %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=r_values, y=[r["mean"] for r in rows], mode="lines+markers", name="zeitliches Mittel", line=dict(dash="dot"), hovertemplate="r = %{x}: %{y:.4f}<extra></extra>"))
    fig.add_hline(y=0.0, line=dict(color="black", width=1))
    fig.update_xaxes(title_text="Amplitude r der zweiten Komponente")
    fig.update_yaxes(title_text="Frequenz (Zyklen je Abtastwert)")
    return _fixed(fig, height=340)


def build_mean(res) -> go.Figure:
    rows = res["mean"]
    labels = [f"c = {r['c']:g}".replace(".", ",") for r in rows]
    fig = go.Figure(go.Bar(x=labels, y=[r["mean"] for r in rows], name="mittlere Momentanfrequenz", marker_color=C.COLOR_EST, hovertemplate="%{x}: %{y:.4f}<extra></extra>"))
    fig.add_hline(y=res["f"], line=dict(color=C.COLOR_TRUE, width=1.6, dash="dash"), annotation_text="wahr", annotation_position="top right")
    fig.update_yaxes(title_text="mittlere Momentanfrequenz")
    fig.update_layout(showlegend=False)
    return _fixed(fig, height=320)


def build_smooth(rows) -> go.Figure:
    fig = go.Figure()
    names = {"plain": "einfach", "l5": "Mittelwert L = 5", "l11": "L = 11", "l21": "L = 21", "bandpass": "Bandpass", "bandpass_l11": "Bandpass + L = 11"}
    for key, name in names.items():
        fig.add_trace(go.Scatter(x=[r["snr"] for r in rows], y=[r[key] for r in rows], mode="lines+markers", name=name, hovertemplate=name + ", %{x} dB: %{y:.4f}<extra></extra>"))
    fig.update_xaxes(title_text="Rauschabstand (dB)", autorange="reversed")
    fig.update_yaxes(title_text="RMSE der Momentanfrequenz (innen)", type="log")
    return _fixed(fig, height=360)


def build_envelope_spectrum(rows) -> go.Figure:
    fig = go.Figure()
    for key, name in (("envelope", "Hüllkurvenspektrum"), ("direct", "direktes Spektrum (Seitenbänder)")):
        fig.add_trace(go.Scatter(x=[r["snr"] for r in rows], y=[100 * r[key] for r in rows], mode="lines+markers", name=name, hovertemplate=name + ", %{x} dB: %{y:.0f} %<extra></extra>"))
    fig.update_xaxes(title_text="Rauschabstand (dB)")
    fig.update_yaxes(title_text="Modulationsfrequenz gefunden (%)", range=[-3, 103])
    return _fixed(fig, height=320)
