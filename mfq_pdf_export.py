"""PDF-Export der aktuellen Einstellung (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und griechische Buchstaben werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

import mfq_constants as C
from mfq_evaluation import Analysis, de

_REPLACE = {"–": "-", "—": "-", "−": "-", "²": "2", "€": "EUR", "→": "->", "σ": "sigma", "Σ": "Summe", "±": "+-", "√": "Wurzel aus ", "×": "x", "é": "e", "„": '"', "“": '"', "β": "beta", "π": "pi"}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s = a.settings
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Momentanfrequenz: Einstellung und Kennzahlen"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    params = ", ".join(f"{label} {de(v, 3)}" for (label, *_rest), v in zip(C.PARAMS[s.kind], (s.p1, s.p2, s.p3)))
    noise = f"Rauschen {s.snr} dB gegen eine Linie der Amplitude 1" if s.has_noise else "ohne Rauschen"
    pdf.multi_cell(0, 5, _s(f"{s.kind} ({params}), n = {s.n}, {noise}, Glättung L = {s.smooth}, Bandpass {s.bandpass}, Seed {s.seed}. Alles simuliert."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    out = [f"Momentanfrequenz: Spanne {de(a.f_min, 3)} bis {de(a.f_max, 3)}, Mittel {de(a.mean_f, 4)}, Anteil negativ {de(100 * a.negative_share, 1)} %."]
    if a.rmse_inner is not None:
        out.append(f"Fehler gegen die wahre Momentanfrequenz: innen {de(a.rmse_inner, 4)}, an den Rändern {de(a.rmse_edge, 4)}.")
    if a.env_rmse_inner is not None:
        out.append(f"Fehler der Hüllkurve innen: {de(a.env_rmse_inner, 4)}.")
    if a.band is not None:
        out.append(f"Bandpass: {de(a.band[0], 3)} bis {de(a.band[1], 3)}.")
    for line in out:
        pdf.multi_cell(0, 5, _s(line), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Quelle: momentanfrequenz-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
