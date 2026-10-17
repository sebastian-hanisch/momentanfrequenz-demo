"""Einstellungen, Messung, Analyse und Kennzahlen: Datenstrom, Signaltypen, Bandpass, Glättung, Fehlermaße, Experimente mit kleinen Parametern."""

import math

import numpy as np
import pytest

import mfq_analytic as A
import mfq_constants as C
import mfq_evaluation as ev
import mfq_signals as S
import signal_generator as G


def settings(kind, *params, **kw):
    p = list(params) + [0.0] * (3 - len(params))
    return ev.Settings(kind=kind, p1=p[0], p2=p[1], p3=p[2], **kw)


def test_noise_sigma_is_the_documented_scale():
    assert ev.noise_sigma(0.0) == pytest.approx(math.sqrt(0.5)) and ev.noise_sigma(10.0) == pytest.approx(math.sqrt(0.05)) and ev.noise_sigma(20.0) == pytest.approx(math.sqrt(0.005))
    assert ev.Settings(noise="an").has_noise and not ev.Settings().has_noise


@pytest.mark.parametrize("kind", list(C.SIGNAL_TYPES))
def test_make_signal_matches_the_signal_module(kind):
    defaults = [d for (_l, _lo, _hi, _st, d, _h) in C.PARAMS[kind]]
    s = settings(kind, *defaults, n=256)
    sig = ev.make_signal(s)
    ref = {"Chirp": lambda: S.chirp(256, defaults[0], defaults[1], C.PHASE), "Amplitudenmodulation": lambda: S.am_signal(256, *defaults, C.PHASE), "Frequenzmodulation": lambda: S.fm_signal(256, *defaults, C.PHASE),
           "Zwei Kosinus": lambda: S.two_tones(256, *defaults, C.PHASE), "Kosinus mit Mittelwert": lambda: S.mean_tone(256, defaults[0], defaults[1], C.PHASE)}[kind]()
    assert sig.x == pytest.approx(ref.x) and sig.components == ref.components
    with pytest.raises(ValueError):
        ev.make_signal(ev.Settings(kind="Rauschen"))


def test_measurement_adds_noise_from_the_seed_only_when_on():
    s = settings("Chirp", 0.1, 0.3, n=128)
    clean_sig, x = ev.build_measurement(s)
    assert np.array_equal(x, clean_sig.x)
    noisy_sig, y = ev.build_measurement(settings("Chirp", 0.1, 0.3, n=128, noise="an", snr=10, seed=5))
    assert y == pytest.approx(noisy_sig.x + ev.noise_sigma(10) * G.SplitMix64(5).normal(128), abs=1e-12)
    assert not np.array_equal(y, ev.build_measurement(settings("Chirp", 0.1, 0.3, n=128, noise="an", snr=10, seed=6))[1])


def test_signal_band_per_type():
    chirp = ev.make_signal(settings("Chirp", 0.1, 0.3))
    assert ev.signal_band(settings("Chirp", 0.1, 0.3), chirp) == pytest.approx((0.05, 0.35))
    mean = settings("Kosinus mit Mittelwert", 0.2, 1.0)
    assert ev.signal_band(mean, ev.make_signal(mean)) == pytest.approx((0.15, 0.25))
    fm = settings("Frequenzmodulation", 0.2, 0.02, 3.0)
    assert ev.signal_band(fm, ev.make_signal(fm)) == pytest.approx((0.2 - 4 * 0.02 - 0.05, 0.2 + 4 * 0.02 + 0.05))
    low = settings("Chirp", 0.02, 0.48)
    lo, hi = ev.signal_band(low, ev.make_signal(low))
    assert lo == pytest.approx(0.005) and hi == pytest.approx(0.5)
    am = settings("Amplitudenmodulation", 0.2, 0.25, 0.5)
    assert ev.signal_band(am, ev.make_signal(am)) == pytest.approx((0.15, 0.5))                  # unteres Seitenband negativ: nur die positiven Komponenten zählen


def test_a_clean_chirp_is_measured_exactly_inside_and_worse_at_the_edges():
    a = ev.analyse(settings("Chirp", 0.1, 0.3))
    assert a.rmse_inner < 2e-3 and a.rmse_edge > 10 * a.rmse_inner and a.negative_share == 0.0 and a.env_rmse_inner < 0.01
    assert len(a.f_est) == 511 and len(a.env_est) == 512 and a.f_true is not None and a.band is None
    assert a.mean_f == pytest.approx(float(a.f_est.mean())) and a.f_min == pytest.approx(float(a.f_est.min())) and a.f_max == pytest.approx(float(a.f_est.max()))


def test_error_definitions_by_hand_for_a_known_estimate():
    a = ev.analyse(settings("Chirp", 0.1, 0.3, n=256))
    err = a.f_est - a.f_true
    e = C.EDGE
    assert a.rmse_inner == pytest.approx(math.sqrt(np.mean(err[e:-e] ** 2))) and a.rmse_edge == pytest.approx(math.sqrt(np.mean(np.r_[err[:e], err[-e:]] ** 2)))
    assert a.env_rmse_inner == pytest.approx(math.sqrt(np.mean((a.env_est - a.clean.env_true)[e:-e] ** 2)))


def test_two_tones_have_no_true_frequency_and_no_error_measures():
    a = ev.analyse(settings("Zwei Kosinus", 0.2, 0.25, 1.0))
    assert a.f_true is None and a.rmse_inner is None and a.rmse_edge is None and a.env_rmse_inner is None and a.negative_share > 0.0 and a.f_min < 0.0


def test_noise_and_smoothing_move_the_error_the_right_way():
    base = settings("Chirp", 0.1, 0.3, noise="an", snr=10)
    plain = ev.analyse(base)
    import dataclasses
    smooth = ev.analyse(dataclasses.replace(base, smooth=21))
    band = ev.analyse(dataclasses.replace(base, bandpass="an"))
    assert ev.analyse(settings("Chirp", 0.1, 0.3)).rmse_inner < plain.rmse_inner and smooth.rmse_inner < 0.3 * plain.rmse_inner and band.rmse_inner < plain.rmse_inner and band.band == pytest.approx((0.05, 0.35))
    assert smooth.f_est == pytest.approx(A.smooth(plain.f_est, 21))


def test_the_bandpass_repairs_the_mean():
    bad = ev.analyse(settings("Kosinus mit Mittelwert", 0.2, 1.5))
    fixed = ev.analyse(settings("Kosinus mit Mittelwert", 0.2, 1.5, bandpass="an"))
    assert abs(bad.mean_f) < 0.01 and fixed.mean_f == pytest.approx(0.2, abs=0.002) and fixed.rmse_inner < bad.rmse_inner


def test_frequency_modulation_truth_is_folded_into_the_valid_range():
    a = ev.analyse(settings("Frequenzmodulation", 0.45, 0.2, 10.0))
    assert np.all(a.f_true > -0.5 - 1e-12) and np.all(a.f_true <= 0.5 + 1e-12)
    assert len(a.f_true) == 511


def test_the_analysis_is_deterministic():
    s = settings("Chirp", 0.1, 0.3, noise="an", snr=5, seed=3)
    a, b = ev.analyse(s), ev.analyse(s)
    assert np.array_equal(a.f_est, b.f_est) and a.rmse_inner == b.rmse_inner


def test_sweeps_accept_small_parameters():
    assert len(ev.sweep_chirp(snrs=(20,), runs=5, n=128)) == 1
    assert len(ev.sweep_bedrosian(ks=(4, 102), n=512)) == 2
    assert len(ev.sweep_tones(rs=(0.5, 1.0), n=256)) == 2
    m = ev.sweep_mean(cs=(0.0, 2.0), n=512)
    assert len(m["mean"]) == 2 and len(m["edge"]) == len(C.EXP_EDGE_SEGMENTS)
    assert len(ev.sweep_smooth(snrs=(10,), runs=5, n=128)) == 1
    assert len(ev.sweep_envelope(snrs=(0,), runs=5, n=2048)) == 1


def test_signal_band_ignores_a_component_at_zero():
    am = settings("Amplitudenmodulation", 0.2, 0.2, 0.5)                                          # untere Seitenbandfrequenz genau 0
    sig = ev.make_signal(am)
    assert sig.components[0] == pytest.approx(0.0, abs=1e-12)
    assert ev.signal_band(am, sig) == pytest.approx((0.15, 0.45))
