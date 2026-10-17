"""Kern des analytischen Signals: Mini-Instanz von Hand, scipy-Orakel, exakte Fälle, Faltung der Frequenz, Glättung, Bandpass."""

import math

import numpy as np
import pytest
from scipy.signal import hilbert

import mfq_analytic as A


def test_the_mini_instance_by_hand(mini_x):
    z = A.analytic_signal(mini_x)
    assert z == pytest.approx([1.0, 1.0j, -1.0, -1.0j], abs=1e-12) and A.envelope(z) == pytest.approx([1.0] * 4)
    assert A.inst_frequency(z, periodic=True) == pytest.approx([0.25] * 4, abs=1e-12) and A.inst_frequency(z).shape == (3,)
    assert A.phase(z) == pytest.approx([0.0, math.pi / 2, math.pi, -math.pi / 2], abs=1e-12) or A.phase(z)[2] == pytest.approx(math.pi)


@pytest.mark.parametrize("n", [2, 3, 8, 9, 64, 65])
def test_analytic_signal_equals_scipy_for_even_and_odd_lengths(n):
    y = np.random.default_rng(n).normal(size=n)
    z = A.analytic_signal(y)
    assert z == pytest.approx(hilbert(y), abs=1e-12) and z.real == pytest.approx(y, abs=1e-12)


def test_analytic_signal_of_several_rows_and_validation():
    y = np.random.default_rng(1).normal(size=(3, 20))
    assert A.analytic_signal(y) == pytest.approx(hilbert(y, axis=-1), abs=1e-12)
    with pytest.raises(ValueError):
        A.analytic_signal(np.array([1.0]))


def test_the_imaginary_part_of_a_cosine_is_the_matching_sine():
    n = 128
    t = np.arange(n)
    z = A.analytic_signal(np.cos(2 * np.pi * 9 * t / n + 0.4))
    assert z.imag == pytest.approx(np.sin(2 * np.pi * 9 * t / n + 0.4), abs=1e-12)
    assert A.analytic_signal(np.ones(16)) == pytest.approx(np.ones(16), abs=1e-12)                       # Gleichanteil bleibt einfach


def test_a_bin_exact_cosine_has_constant_envelope_and_exact_frequency():
    n = 256
    f = 51 / n
    z = A.analytic_signal(np.cos(2 * np.pi * f * np.arange(n) + 0.3))
    assert A.envelope(z) == pytest.approx([1.0] * n, abs=1e-12) and A.inst_frequency(z, periodic=True) == pytest.approx([f] * n, abs=1e-12)


def test_the_frequency_step_is_valid_between_samples_and_has_n_minus_one_values():
    z = np.exp(2j * np.pi * 0.1 * np.arange(10))
    f = A.inst_frequency(z)
    assert f.shape == (9,) and f == pytest.approx([0.1] * 9, abs=1e-12)
    assert A.inst_frequency(np.stack([z, np.conj(z)])).shape == (2, 9) and A.inst_frequency(np.conj(z)) == pytest.approx([-0.1] * 9, abs=1e-12)
    assert A.inst_frequency(z, periodic=True).shape == (10,)


def test_fold_maps_frequencies_into_the_half_open_unit_interval():
    assert A.fold(np.array([0.7, -0.7, 0.5, -0.5, 0.2, 1.2, -1.3, 0.0])) == pytest.approx([-0.3, 0.3, 0.5, 0.5, 0.2, 0.2, -0.3, 0.0], abs=1e-12)
    assert A.fold(0.5) == 0.5 and A.fold(np.array([1.5]))[0] == 0.5


def test_smooth_by_hand_and_with_normalised_edges():
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert A.smooth(v, 3).tolist() == pytest.approx([1.5, 2.0, 3.0, 4.0, 4.5])
    assert A.smooth(v, 1) is v or A.smooth(v, 1).tolist() == v.tolist()
    assert A.smooth(v, 5).tolist() == pytest.approx([2.0, 2.5, 3.0, 3.5, 4.0])
    assert A.smooth(np.ones(9), 5).tolist() == pytest.approx([1.0] * 9)                           # konstant bleibt konstant bis zum Rand
    rows = np.stack([v, 2 * v])
    assert A.smooth(rows, 3)[1].tolist() == pytest.approx([3.0, 4.0, 6.0, 8.0, 9.0])
    assert A.smooth(v, 0).tolist() == v.tolist()


def test_bandpass_keeps_the_band_and_removes_the_mean():
    n = 64
    t = np.arange(n)
    x = 3.0 + np.cos(2 * np.pi * 8 * t / n) + np.cos(2 * np.pi * 20 * t / n)
    y = A.bandpass(x, 0.05, 0.2)
    assert y == pytest.approx(np.cos(2 * np.pi * 8 * t / n), abs=1e-12)
    assert A.bandpass(x, 0.0, 0.5) == pytest.approx(x, abs=1e-12) and np.isrealobj(y)
    assert A.bandpass(x, 0.35, 0.4) == pytest.approx(np.zeros(n), abs=1e-12)
    assert A.bandpass(np.stack([x, x]), 0.05, 0.2).shape == (2, n)


def test_bandpass_edges_are_inclusive_and_smooth_of_length_two_averages():
    n = 64
    t = np.arange(n)
    x = np.cos(2 * np.pi * 8 * t / n) + np.cos(2 * np.pi * 20 * t / n)                           # Frequenzen 0,125 und 0,3125
    assert A.bandpass(x, 0.125, 0.3125) == pytest.approx(x, abs=1e-12)                          # beide Ränder gehören zum Band
    assert A.bandpass(x, 0.126, 0.3125) == pytest.approx(np.cos(2 * np.pi * 20 * t / n), abs=1e-12)
    assert A.bandpass(x, 0.125, 0.3124) == pytest.approx(np.cos(2 * np.pi * 8 * t / n), abs=1e-12)
    v = np.array([1.0, 3.0, 5.0, 7.0])
    assert A.smooth(v, 2).tolist() != v.tolist() and A.smooth(v, 2).tolist() == pytest.approx([1.0, 2.0, 4.0, 6.0])
