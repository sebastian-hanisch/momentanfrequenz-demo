"""Testsignale mit exakter Wahrheit: Chirp, Amplitudenmodulation, Frequenzmodulation, zwei Kosinus, Kosinus mit Mittelwert.

Reine Funktionen, kein Zufall. Zeit t = 0 .. n - 1, Abtastrate 1. `f_true` ist die wahre Momentanfrequenz an den halben Zeitpunkten t + 1/2 (n - 1 Werte, dort liefert der Phasenschritt
sie exakt), `env_true` die wahre Hüllkurve an den Zeitpunkten (n Werte); beide None, wo sie nicht definiert sind. `components` nennt die Frequenzen der Komponenten.
"""

import math
from dataclasses import dataclass

import numpy as np

TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class TestSignal:
    x: np.ndarray
    f_true: np.ndarray | None
    env_true: np.ndarray | None
    components: tuple


def chirp(n: int, f0: float, f1: float, phase: float = 0.0) -> TestSignal:
    """Linearer Chirp cos(2 pi (f0 t + (f1 - f0) t^2 / (2 (n - 1))) + phase): Momentanfrequenz f0 + (f1 - f0) t / (n - 1), Hüllkurve 1."""
    t = np.arange(n)
    ph = TWO_PI * (f0 * t + (f1 - f0) * t * t / (2.0 * (n - 1))) + phase
    f_true = f0 + (f1 - f0) * (t[:-1] + 0.5) / (n - 1)
    return TestSignal(np.cos(ph), f_true, np.ones(n), (f0, f1))


def am_signal(n: int, fc: float, fm: float, depth: float, phase: float = 0.0) -> TestSignal:
    """Amplitudenmodulation (1 + depth cos(2 pi fm t)) cos(2 pi fc t + phase): Hüllkurve 1 + depth cos, Momentanfrequenz des Trägers fc (Tiefe unter 1)."""
    t = np.arange(n)
    env = 1.0 + depth * np.cos(TWO_PI * fm * t)
    return TestSignal(env * np.cos(TWO_PI * fc * t + phase), np.full(n - 1, fc), env, (fc - fm, fc, fc + fm))


def fm_signal(n: int, fc: float, fm: float, beta: float, phase: float = 0.0) -> TestSignal:
    """Frequenzmodulation cos(2 pi fc t + beta sin(2 pi fm t) + phase): Hüllkurve 1; der Phasenschritt zwischen t und t + 1 ist 2 pi fc + 2 beta sin(pi fm) cos(2 pi fm (t + 1/2)),
    die Momentanfrequenz an den halben Zeitpunkten also fc + (beta / pi) sin(pi fm) cos(2 pi fm (t + 1/2))."""
    t = np.arange(n)
    x = np.cos(TWO_PI * fc * t + beta * np.sin(TWO_PI * fm * t) + phase)
    f_true = fc + (beta / math.pi) * math.sin(math.pi * fm) * np.cos(TWO_PI * fm * (t[:-1] + 0.5))
    return TestSignal(x, f_true, np.ones(n), (fc,))


def two_tones(n: int, f1: float, f2: float, r: float, phase: float = 0.0) -> TestSignal:
    """Summe cos(2 pi f1 t) + r cos(2 pi f2 t + phase): keine wahre Momentanfrequenz, die Komponentenfrequenzen sind f1 und f2."""
    t = np.arange(n)
    return TestSignal(np.cos(TWO_PI * f1 * t) + r * np.cos(TWO_PI * f2 * t + phase), None, None, (f1, f2))


def mean_tone(n: int, f: float, c: float, phase: float = 0.0) -> TestSignal:
    """Konstante plus Kosinus c + cos(2 pi f t + phase): Momentanfrequenz f (Mittelwert unter 1 gegen die Amplitude), Hüllkurve des Kosinus 1."""
    t = np.arange(n)
    return TestSignal(c + np.cos(TWO_PI * f * t + phase), np.full(n - 1, f), np.ones(n), (f,))


def two_tone_frequency(f1: float, f2: float, r: float, delta) -> np.ndarray:
    """Momentanfrequenz des Summenzeigers e^(i 2 pi f1 t) + r e^(i (2 pi f2 t + phase)) bei der Phasendifferenz delta der Komponenten: (f1 + r^2 f2 + r (f1 + f2) cos delta) / (1 + r^2 + 2 r cos delta)."""
    delta = np.asarray(delta, dtype=float)
    return (f1 + r * r * f2 + r * (f1 + f2) * np.cos(delta)) / (1.0 + r * r + 2.0 * r * np.cos(delta))
