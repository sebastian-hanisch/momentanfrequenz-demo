"""Randfälle: alle Signaltypen und Längen, Parametergrenzen, Rauschen, Glättung, Bandpass, Abbildungen und PDF."""

import dataclasses

import numpy as np
import pytest

import mfq_constants as C
import mfq_evaluation as ev
import mfq_visualization as V
from mfq_pdf_export import build_pdf


def settings(kind, params, **kw):
    p = list(params) + [0.0] * (3 - len(params))
    return ev.Settings(kind=kind, p1=p[0], p2=p[1], p3=p[2], **kw)


@pytest.mark.parametrize("kind", list(C.SIGNAL_TYPES))
def test_every_type_runs_for_the_shortest_and_the_longest_record(kind):
    defaults = [spec[4] for spec in C.PARAMS[kind]]
    for n in (min(C.N_CHOICES), max(C.N_CHOICES)):
        a = ev.analyse(settings(kind, defaults, n=n, noise="an", snr=5, smooth=21, bandpass="an"))
        assert np.isfinite(a.f_est).all() and np.isfinite(a.env_est).all() and len(a.f_est) == n - 1


@pytest.mark.parametrize("kind", list(C.SIGNAL_TYPES))
def test_every_parameter_at_its_limits_runs(kind):
    specs = C.PARAMS[kind]
    for pick in ("lo", "hi"):
        params = [spec[1] if pick == "lo" else spec[2] for spec in specs]
        for bandpass in C.BANDPASS_MODES:
            a = ev.analyse(settings(kind, params, n=256, bandpass=bandpass))
            assert np.isfinite(a.f_est).all() and np.isfinite(a.mean_f)


def test_extreme_noise_levels_and_seed_limits_run():
    for snr in (C.SNR_MIN, C.SNR_MAX):
        for seed in (0, C.SEED_MAX):
            a = ev.analyse(settings("Chirp", (0.1, 0.3), noise="an", snr=snr, seed=seed))
            assert np.isfinite(a.f_est).all() and np.isfinite(a.rmse_inner)


def test_the_error_ordering_over_all_smoothing_lengths():
    base = settings("Chirp", (0.1, 0.3), noise="an", snr=5)
    errors = [ev.analyse(dataclasses.replace(base, smooth=L)).rmse_inner for L in C.SMOOTH_CHOICES]
    assert errors == sorted(errors, reverse=True)


def test_pdf_builds_for_every_type():
    for kind in C.SIGNAL_TYPES:
        defaults = [spec[4] for spec in C.PARAMS[kind]]
        pdf = build_pdf(ev.analyse(settings(kind, defaults, noise="an", bandpass="an")))
        assert pdf[:4] == b"%PDF" and len(pdf) > 1000
    assert build_pdf(ev.analyse(settings("Zwei Kosinus", (0.2, 0.25, 1.0))))[:4] == b"%PDF"


def test_every_figure_builds():
    chirp = ev.analyse(settings("Chirp", (0.1, 0.3), noise="an"))
    tones = ev.analyse(settings("Zwei Kosinus", (0.2, 0.25, 1.0)))
    assert len(V.build_signal(chirp).data) == 2 and len(V.build_signal(ev.analyse(settings("Chirp", (0.1, 0.3)))).data) == 1
    assert len(V.build_envelope(chirp).data) == 2 and len(V.build_envelope(tones).data) == 1 and len(V.build_envelope(chirp).layout.shapes) == 2
    assert len(V.build_frequency(chirp).data) == 2 and len(V.build_frequency(chirp).layout.shapes) == 2
    f = V.build_frequency(tones)
    assert len(f.data) == 1 and len(f.layout.shapes) == 2 + 3
    chirp_rows = [{"snr": s, "inner": 0.01, "edge": 0.05, "over": 0.0, "runs": 1} for s in (40, 10)]
    assert len(V.build_chirp(chirp_rows).data) == 2
    bed = [{"ratio": r, "fm": 0.1, "env_max": e, "env_rmse": e, "f_rmse": e, "f_min": 0.2, "f_max": 0.2, "lowest_band": 0.0} for r, e in ((0.5, 0.0), (1.0, 0.1))]
    assert len(V.build_bedrosian(bed).data) == 2
    tone_rows = [{"r": r, "f_min": 0.1, "f_max": 0.2, "outside": 0.1, "negative": 0.0, "mean": 0.1992, "weighted": 0.2, "f1": 0.2, "f2": 0.25} for r in (0.5, 1.0)]
    assert len(V.build_tones(tone_rows).data) == 3
    mean = {"mean": [{"c": c, "mean": 0.2, "f_min": 0.0, "f_max": 0.0, "f_true": 0.2} for c in (0.0, 1.1)], "edge": [], "f": 0.2}
    assert len(V.build_mean(mean).data) == 1
    smooth = [{"snr": 10, "runs": 1, "plain": 0.03, "l5": 0.01, "l11": 0.005, "l21": 0.0025, "bandpass": 0.017, "bandpass_l11": 0.0036}]
    assert len(V.build_smooth(smooth).data) == 6
    assert len(V.build_envelope_spectrum([{"snr": -6, "envelope": 0.4, "direct": 0.7, "runs": 1}]).data) == 2
