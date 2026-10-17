"""Festwerte der Experimente (feste Zufallsströme): Das Ergebnis ist bei jedem Lauf dasselbe; Abweichungen bedeuten eine geänderte Rechnung."""

import pytest

from sweeps_cache import get

CHIRP = {40: (0.001468, 0.049404, 0.0), 30: (0.003448, 0.049542, 0.0), 20: (0.010499, 0.05046, 4.9e-05), 10: (0.034563, 0.064072, 0.135936), 5: (0.078879, 0.112485, 0.391969), 0: (0.157212, 0.183398, 0.606917)}

BEDROSIAN = [(0.039216, 0.0, 0.0, 0.0, 0.199219, 0.199219, 0.191406), (0.098039, 0.0, 0.0, 0.0, 0.199219, 0.199219, 0.179688), (0.254902, 0.0, 0.0, 0.0, 0.199219, 0.199219, 0.148438), (0.5, 0.0, 0.0, 0.0, 0.199219, 0.199219, 0.099609), (0.784314, 0.0, 0.0, 0.0, 0.199219, 0.199219, 0.042969), (1.0, 0.161045, 0.114748, 0.025268, 0.156508, 0.23412, 0.0), (1.27451, 0.495311, 0.254436, 0.048457, 0.114037, 0.292529, -0.054688)]

TONES = {0.05: (0.196559, 0.201628, 0.484375, 0.0, 0.199219, 0.199345), 0.1: (0.193609, 0.203821, 0.46875, 0.0, 0.199219, 0.199722), 0.3: (0.177698, 0.210917, 0.402344, 0.0, 0.199219, 0.203412), 0.5: (0.149694, 0.21613, 0.335938, 0.0, 0.199219, 0.209375), 0.8: (0.026146, 0.221782, 0.210938, 0.0, 0.199219, 0.219036), 1.0: (-0.275391, 0.224609, 0.050781, 0.050781, 0.199219, 0.224609)}

MEAN = {0.0: (0.199219, 0.199219, 0.199219), 0.25: (0.199219, 0.160647, 0.257032), 0.5: (0.199219, 0.133809, 0.344862), 0.9: (0.058594, -0.494029, 0.499789), 1.1: (0.0, -0.35387, 0.094705), 2.0: (-0.0, -0.145643, 0.06541)}

EDGE = [(0, 16, 0.010434), (16, 64, 0.001996), (64, 448, 0.000438), (448, 496, 0.002238), (496, 511, 0.050695)]

SMOOTH = {20: (0.010408, 0.003187, 0.001452, 0.000758, 0.005427, 0.001131), 10: (0.03432, 0.010341, 0.004719, 0.002463, 0.017324, 0.00362), 5: (0.077719, 0.027799, 0.016644, 0.011479, 0.036671, 0.008938), 0: (0.157189, 0.068534, 0.048659, 0.038818, 0.089584, 0.033979)}

ENVELOPE = {-6: (0.405, 0.6925), -3: (0.9375, 0.9825), 0: (1.0, 1.0), 3: (1.0, 1.0), 6: (1.0, 1.0), 10: (1.0, 1.0)}



def test_chirp_golden():
    rows = get("chirp")
    assert [r["snr"] for r in rows] == list(CHIRP)
    for r in rows:
        assert (r["inner"], r["edge"], r["over"]) == pytest.approx(CHIRP[r["snr"]], abs=1e-6) and r["runs"] == 300


def test_bedrosian_golden():
    rows = get("bedrosian")
    assert len(rows) == len(BEDROSIAN) == 7
    for r, ref in zip(rows, BEDROSIAN):
        assert (r["ratio"], r["env_max"], r["env_rmse"], r["f_rmse"], r["f_min"], r["f_max"], r["lowest_band"]) == pytest.approx(ref, abs=1e-5)


def test_tones_golden():
    rows = get("tones")
    assert [r["r"] for r in rows] == list(TONES)
    for r in rows:
        assert (r["f_min"], r["f_max"], r["outside"], r["negative"], r["mean"], r["weighted"]) == pytest.approx(TONES[r["r"]], abs=1e-5)


def test_mean_golden():
    res = get("mean")
    assert [r["c"] for r in res["mean"]] == list(MEAN)
    for r in res["mean"]:
        assert (r["mean"], r["f_min"], r["f_max"]) == pytest.approx(MEAN[r["c"]], abs=1e-5)
    assert [(r["lo"], r["hi"]) for r in res["edge"]] == [(lo, hi) for lo, hi, _ in EDGE]
    for r, (lo, hi, rmse) in zip(res["edge"], EDGE):
        assert r["rmse"] == pytest.approx(rmse, abs=1e-5)


def test_smooth_golden():
    rows = get("smooth")
    assert [r["snr"] for r in rows] == list(SMOOTH)
    for r in rows:
        assert tuple(r[k] for k in ("plain", "l5", "l11", "l21", "bandpass", "bandpass_l11")) == pytest.approx(SMOOTH[r["snr"]], abs=1e-5) and r["runs"] == 300


def test_envelope_golden():
    rows = get("envelope")
    assert [r["snr"] for r in rows] == list(ENVELOPE)
    for r in rows:
        assert (r["envelope"], r["direct"]) == pytest.approx(ENVELOPE[r["snr"]], abs=1e-6) and r["runs"] == 400
