"""Hüllkurvenspektrum: Spektrum der mittelwertfreien Hüllkurve des analytischen Signals. Reine Funktionen, nur numpy."""

import numpy as np

from mfq_analytic import analytic_signal, envelope


def envelope_spectrum(x, lo: float, hi: float):
    """Periodogramm |FFT(Hüllkurve - Mittelwert)|^2 / n der Zeilen von x im Frequenzband lo <= f <= hi: (Frequenzen, Spektrum)."""
    env = envelope(analytic_signal(x))
    env = env - env.mean(axis=-1, keepdims=True)
    spec = np.abs(np.fft.rfft(env, axis=-1)) ** 2 / env.shape[-1]
    f = np.fft.rfftfreq(env.shape[-1])
    sel = (f >= lo) & (f <= hi)
    return f[sel], spec[..., sel]


def direct_spectrum(x, lo: float, hi: float):
    """Periodogramm |FFT(x)|^2 / n der Zeilen von x im Frequenzband lo <= f <= hi: (Frequenzen, Spektrum)."""
    x = np.asarray(x, dtype=float)
    spec = np.abs(np.fft.rfft(x, axis=-1)) ** 2 / x.shape[-1]
    f = np.fft.rfftfreq(x.shape[-1])
    sel = (f >= lo) & (f <= hi)
    return f[sel], spec[..., sel]
