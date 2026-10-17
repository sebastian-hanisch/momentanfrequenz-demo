"""Testsignale: Definitionen von Hand, exakte Wahrheit gegen die Differenz der Phase, Zwei-Komponenten-Formel, Hüllkurvenspektrum."""

import math

import numpy as np
import pytest

import mfq_analytic as A
import mfq_envelope as E
import mfq_signals as S

TWO_PI = 2 * math.pi


def test_chirp_by_hand_and_the_phase_difference_is_the_true_frequency():
    s = S.chirp(3, 0.1, 0.3)                                                                      # t = 0, 1, 2: Phase 2 pi (0,1 t + 0,05 t^2 / 2 ... ) mit (f1 - f0) / (2 (n - 1)) = 0,05
    assert s.x == pytest.approx(np.cos(TWO_PI * (0.1 * np.arange(3) + 0.05 * np.arange(3) ** 2)), abs=1e-12)
    assert s.f_true == pytest.approx([0.1 + 0.2 * 0.5 / 2, 0.1 + 0.2 * 1.5 / 2], abs=1e-12) and s.env_true.tolist() == [1.0, 1.0, 1.0] and s.components == (0.1, 0.3)
    big = S.chirp(300, 0.1, 0.4, 0.7)
    t = np.arange(300)
    ph = TWO_PI * (0.1 * t + 0.3 * t * t / (2 * 299)) + 0.7
    assert np.diff(ph) / TWO_PI == pytest.approx(big.f_true, abs=1e-12) and big.x == pytest.approx(np.cos(ph), abs=1e-12)


def test_am_signal_definition_and_envelope():
    s = S.am_signal(8, 0.2, 0.05, 0.5, 0.4)
    t = np.arange(8)
    assert s.env_true == pytest.approx(1 + 0.5 * np.cos(TWO_PI * 0.05 * t)) and s.x == pytest.approx(s.env_true * np.cos(TWO_PI * 0.2 * t + 0.4), abs=1e-12)
    assert s.f_true.tolist() == [0.2] * 7 and s.components == pytest.approx((0.15, 0.2, 0.25))


def test_fm_signal_phase_difference_equals_the_closed_form():
    s = S.fm_signal(400, 0.2, 0.013, 3.0, 0.5)
    t = np.arange(400)
    ph = TWO_PI * 0.2 * t + 3.0 * np.sin(TWO_PI * 0.013 * t) + 0.5
    assert np.diff(ph) / TWO_PI == pytest.approx(s.f_true, abs=1e-12) and s.x == pytest.approx(np.cos(ph), abs=1e-12) and s.env_true.tolist() == [1.0] * 400 and s.components == (0.2,)


def test_two_tones_and_mean_tone_definitions():
    t = np.arange(10)
    s = S.two_tones(10, 0.1, 0.2, 0.5, 0.3)
    assert s.x == pytest.approx(np.cos(TWO_PI * 0.1 * t) + 0.5 * np.cos(TWO_PI * 0.2 * t + 0.3), abs=1e-12) and s.f_true is None and s.env_true is None and s.components == (0.1, 0.2)
    m = S.mean_tone(10, 0.1, 1.5, 0.3)
    assert m.x == pytest.approx(1.5 + np.cos(TWO_PI * 0.1 * t + 0.3), abs=1e-12) and m.f_true.tolist() == [0.1] * 9 and m.env_true.tolist() == [1.0] * 10


def test_the_hilbert_frequency_of_a_chirp_and_an_fm_signal_matches_the_truth_inside():
    for s in (S.chirp(512, 0.1, 0.3), S.fm_signal(512, 0.2, 0.02, 3.0)):
        f = A.inst_frequency(A.analytic_signal(s.x))
        assert np.sqrt(np.mean((f - s.f_true)[16:-16] ** 2)) < 2e-3
        assert np.sqrt(np.mean(np.r_[(f - s.f_true)[:16], (f - s.f_true)[-16:]] ** 2)) > 10 * np.sqrt(np.mean((f - s.f_true)[16:-16] ** 2))


def test_two_tone_frequency_formula_equals_the_phasor_sum():
    f1, f2, r, phi = 0.002, 0.004, 0.6, 0.9
    t = np.arange(400)
    z = np.exp(1j * TWO_PI * f1 * t) + r * np.exp(1j * (TWO_PI * f2 * t + phi))
    f_num = A.inst_frequency(z)
    delta = TWO_PI * (f2 - f1) * (t[:-1] + 0.5) + phi
    assert S.two_tone_frequency(f1, f2, r, delta) == pytest.approx(f_num, abs=2e-5)
    assert S.two_tone_frequency(0.2, 0.25, 0.0, 1.0) == pytest.approx(0.2) and S.two_tone_frequency(0.2, 0.25, 0.5, 0.0) == pytest.approx((0.2 + 0.25 * 0.25 + 0.5 * 0.45) / 2.25)


def test_the_mean_of_the_frequency_is_the_frequency_of_the_stronger_component():
    n = 512
    f1, f2 = 102 / n, 128 / n
    weak = A.inst_frequency(A.analytic_signal(S.two_tones(n, f1, f2, 0.5, 0.4).x), periodic=True)
    strong = A.inst_frequency(A.analytic_signal(S.two_tones(n, f1, f2, 2.0, 0.4).x), periodic=True)
    assert weak.mean() == pytest.approx(f1, abs=1e-9) and strong.mean() == pytest.approx(f2, abs=1e-9)
    equal = A.inst_frequency(A.analytic_signal(S.two_tones(n, f1, f2, 1.0, 0.4).x), periodic=True)
    assert equal.min() < 0 < equal.max()


def test_a_mean_above_the_amplitude_stops_the_winding():
    n = 512
    f = 102 / n
    below = A.inst_frequency(A.analytic_signal(S.mean_tone(n, f, 0.5, 0.2).x), periodic=True)
    above = A.inst_frequency(A.analytic_signal(S.mean_tone(n, f, 1.5, 0.2).x), periodic=True)
    assert below.mean() == pytest.approx(f, abs=1e-9) and abs(above.mean()) < 1e-9


def test_bedrosian_exact_below_and_broken_at_the_carrier():
    n = 512
    fc = 102 / n
    for k, exact in ((26, True), (80, True), (102, False)):
        s = S.am_signal(n, fc, k / n, 0.5, 0.7)
        z = A.analytic_signal(s.x)
        err = np.abs(A.envelope(z) - s.env_true).max()
        assert (err < 1e-9) == exact


def test_envelope_spectrum_of_a_pure_modulation_has_the_hand_calculated_peak():
    n = 2048
    fm = 32 / n
    x = S.am_signal(n, 640 / n, fm, 0.3).x
    f, spec = E.envelope_spectrum(x, 0.002, 0.1)
    assert f[np.argmax(spec)] == pytest.approx(fm) and spec.max() == pytest.approx(n * 0.3 ** 2 / 4, rel=1e-9)
    assert np.sort(spec)[-2] < 1e-9 * spec.max()
    fd, sd = E.direct_spectrum(x, 0.0, 0.5)
    assert sd[np.argmin(np.abs(fd - 640 / n))] == pytest.approx(n / 4, rel=1e-9) and fd[0] == 0.0 and fd[-1] == 0.5
    assert E.envelope_spectrum(np.stack([x, x]), 0.002, 0.1)[1].shape[0] == 2


def test_envelope_spectrum_band_edges_are_inclusive():
    n = 2048
    fm = 32 / n
    x = S.am_signal(n, 640 / n, fm, 0.3).x
    f, spec = E.envelope_spectrum(x, fm, fm)
    assert len(f) == 1 and f[0] == pytest.approx(fm) and spec[0] == pytest.approx(n * 0.3 ** 2 / 4, rel=1e-9)
    f2, _ = E.envelope_spectrum(x, fm - 1.5 / n, fm + 1.5 / n)
    assert len(f2) == 3
    fd, _ = E.direct_spectrum(x, 0.0, 1 / n)
    assert fd.tolist() == pytest.approx([0.0, 1 / n])
