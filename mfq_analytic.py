"""Kern: analytisches Signal, Hüllkurve, Phase, Momentanfrequenz, Glättung und Bandpass.

Reine Funktionen, kein Zufall, kein Streamlit; nur numpy. Abtastrate 1, Frequenzen in Zyklen je Abtastwert (Nyquist-Frequenz 0,5). Das analytische Signal entsteht per FFT: Gleichanteil und
Nyquist-Anteil einfach, positive Frequenzen doppelt, negative null (die FFT nimmt periodische Fortsetzung an: das erzeugt Randeffekte).
"""

import math

import numpy as np

TWO_PI = 2.0 * math.pi


def analytic_signal(x) -> np.ndarray:
    """Analytisches Signal z = x + i H{x} entlang der letzten Achse (FFT-Fassung)."""
    x = np.asarray(x, dtype=float)
    n = x.shape[-1]
    if n < 2:
        raise ValueError("mindestens zwei Werte nötig")
    h = np.zeros(n)
    h[0] = 1.0
    if n % 2 == 0:
        h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[1:(n + 1) // 2] = 2.0
    return np.fft.ifft(np.fft.fft(x, axis=-1) * h, axis=-1)


def envelope(z) -> np.ndarray:
    """Hüllkurve |z|."""
    return np.abs(z)


def phase(z) -> np.ndarray:
    """Phase arg z (auf (-pi, pi] gefaltet)."""
    return np.angle(z)


def inst_frequency(z, periodic: bool = False) -> np.ndarray:
    """Momentanfrequenz aus dem Phasenschritt arg(z[t+1] conj(z[t])) / (2 pi) in Zyklen je Abtastwert, gültig an den halben Zeitpunkten t + 1/2 (n - 1 Werte; periodic: n Werte,
    der letzte Schritt läuft von z[n-1] zu z[0]). Werte liegen in (-0,5, 0,5]."""
    z = np.asarray(z)
    if periodic:
        z = np.concatenate([z, z[..., :1]], axis=-1)
    return np.angle(z[..., 1:] * np.conj(z[..., :-1])) / TWO_PI


def fold(f) -> np.ndarray:
    """Frequenz auf das Intervall (-0,5, 0,5] falten (das Bild der Abtastung): ((f + 0,5) mod 1) - 0,5 mit der Obergrenze 0,5 eingeschlossen."""
    f = np.asarray(f, dtype=float)
    out = (f + 0.5) % 1.0 - 0.5
    return np.where(out == -0.5, 0.5, out)


def smooth(v, length: int) -> np.ndarray:
    """Gleitender Mittelwert der Länge `length` (ungerade sinnvoll) entlang der letzten Achse, an den Rändern mit der Zahl der vorhandenen Werte normiert (kein Abfall zum Rand)."""
    v = np.asarray(v, dtype=float)
    if length <= 1:
        return v
    k = np.ones(length)
    ones = np.ones(v.shape[-1])
    norm = np.convolve(ones, k, mode="same")
    return np.apply_along_axis(lambda row: np.convolve(row, k, mode="same"), -1, v) / norm


def bandpass(x, lo: float, hi: float) -> np.ndarray:
    """Rechteck-Bandpass per FFT: behält die Frequenzen mit lo <= |f| <= hi (lo > 0 entfernt auch den Mittelwert); Rückgabe reell."""
    x = np.asarray(x, dtype=float)
    n = x.shape[-1]
    mask = ((np.abs(np.fft.fftfreq(n)) >= lo) & (np.abs(np.fft.fftfreq(n)) <= hi)).astype(float)
    return np.real(np.fft.ifft(np.fft.fft(x, axis=-1) * mask, axis=-1))
