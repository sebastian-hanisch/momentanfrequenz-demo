"""Jede Zahl und Aussage in den Texten der App (Presets, Einleitung, Experiment-Überschriften) und in der README ist hier an Rechnung oder Messung gebunden."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

import mfq_constants as C
import mfq_evaluation as ev
from sweeps_cache import get

README = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")


def preset(name, **over):
    p = C.PRESETS[name]["params"]
    base = dict(n=p and C.PRESETS[name]["n"], kind=C.PRESETS[name]["kind"], p1=p[0], p2=p[1], p3=p[2], noise=C.PRESETS[name]["noise"], snr=C.PRESETS[name]["snr"], smooth=C.PRESETS[name]["smooth"],
                bandpass=C.PRESETS[name]["bandpass"], seed=C.PRESETS[name]["seed"])
    return ev.Settings(**{**base, **over})


def A(name, **over):
    return ev.analyse(preset(name, **over))


def de(x, k=4):
    return f"{x:.{k}f}".replace(".", ",")


# --- Presets ----------------------------------------------------------------------------------------------------------------------------------


def test_preset_clean_chirp_is_exact_inside():
    a = A("Chirp: sauber gemessen")
    assert a.rmse_inner < 0.002 and a.rmse_edge > 0.02 and not a.settings.has_noise


def test_preset_noisy_chirp_has_an_error_near_three_hundredths_and_smoothing_or_bandpass_helps():
    name = "Chirp mit Rauschen: die Ableitung verstärkt es"
    a = A(name)
    assert 0.025 < a.rmse_inner < 0.04 and a.settings.snr == 10
    assert A(name, smooth=21).rmse_inner < 0.2 * a.rmse_inner and A(name, bandpass="an").rmse_inner < a.rmse_inner


def test_preset_modulation_breaks_above_the_carrier_and_is_exact_far_below():
    name = "Modulation: Bedrosian bricht, wenn fm den Träger erreicht"
    a = A(name)
    assert a.settings.p2 > a.settings.p1 and a.rmse_inner > 0.04 and a.env_rmse_inner > 0.2
    low = A(name, p2=0.05)
    assert low.rmse_inner < 0.005 and low.env_rmse_inner < 0.005
    assert "0,005" in C.PRESET_HELP[name] and "−0,05" in C.PRESET_HELP[name] and a.settings.p1 - a.settings.p2 == pytest.approx(-0.05)


def test_preset_two_equal_tones_have_negative_frequencies():
    a = A("Zwei gleich starke Töne: negative Frequenzen")
    assert a.settings.p3 == 1.0 and a.negative_share > 0.03 and a.f_min < -0.2
    assert A("Zwei gleich starke Töne: negative Frequenzen", p3=0.1).negative_share < 0.01


def test_preset_mean_collapses_the_mean_frequency():
    name = "Mittelwert: die Phase windet nicht mehr"
    a = A(name)
    assert a.settings.p2 == 1.5 and abs(a.mean_f) < 0.01 and A(name, p2=0.5).mean_f == pytest.approx(0.2, abs=0.005)


# --- README ------------------------------------------------------------------------------------------------------------------------------------


def test_readme_chirp_numbers():
    rows = {r["snr"]: r for r in get("chirp")}
    assert " / ".join(de(rows[s]["inner"]) for s in (40, 30, 20, 10, 5, 0)) == "0,0015 / 0,0034 / 0,0105 / 0,0346 / 0,0789 / 0,1572"
    assert " / ".join(de(rows[s]["edge"], 3) for s in (40, 30, 20, 10, 5, 0)) == "0,049 / 0,050 / 0,050 / 0,064 / 0,112 / 0,183"
    assert de(100 * rows[10]["over"], 1) == "13,6" and de(100 * rows[0]["over"], 1) == "60,7"
    for s in ("0,0015 / 0,0034 / 0,0105 / 0,0346 / 0,0789 / 0,1572", "0,049 / 0,050 / 0,050 / 0,064 / 0,112 / 0,183", "13,6 %", "60,7 %"):
        assert s in README
    assert all(abs(rows[s]["inner"] / 10 ** (-s / 20) - 0.15) < 0.1 for s in (30, 20, 10, 5))        # etwa 10^(-SNR/20)


def test_readme_bedrosian_numbers():
    rows = get("bedrosian")
    below = [r for r in rows if r["ratio"] < 1.0]
    assert len(below) == 5 and all(r["env_max"] < 1e-9 and r["f_rmse"] < 1e-9 for r in below) and round(max(r["ratio"] for r in below), 2) == 0.78
    at = [r for r in rows if abs(r["ratio"] - 1.0) < 1e-9][0]
    assert (de(at["env_max"], 3), de(at["f_min"], 3), de(at["f_max"], 3), de(at["f_rmse"], 4)) == ("0,161", "0,157", "0,234", "0,0253")
    above = rows[-1]
    assert (de(above["ratio"], 2), de(above["env_max"], 3), de(above["f_min"], 3), de(above["f_max"], 3)) == ("1,27", "0,495", "0,114", "0,293")
    for s in ("0,04 bis 0,78", "0,161", "0,157 bis 0,234", "0,0253", "1,27", "0,495", "0,114 bis 0,293"):
        assert s in README


def test_readme_two_tone_numbers():
    rows = {r["r"]: r for r in get("tones")}
    assert [(de(rows[r]["f_min"], 3), de(rows[r]["f_max"], 3)) for r in (0.3, 0.5, 0.8)] == [("0,178", "0,211"), ("0,150", "0,216"), ("0,026", "0,222")]
    assert (de(rows[1.0]["f_min"], 3), de(rows[1.0]["f_max"], 3), de(100 * rows[1.0]["negative"], 1)) == ("-0,275", "0,225", "5,1")
    assert {de(r["mean"], 4) for r in rows.values()} == {"0,1992"} and de(rows[1.0]["weighted"], 4) == "0,2246"
    for s in ("0,178 bis 0,211", "0,150 bis 0,216", "0,026 bis 0,222", "−0,275 bis 0,225", "5,1 % der Zeit negativ", "0,1992 = f1", "0,2246 bei r = 1"):
        assert s in README
    folded = ev.analyse(ev.Settings(kind="Zwei Kosinus", p1=0.2, p2=0.35, p3=1.5, n=512))        # r > 1 mit größerem Abstand: der Phasenschritt faltet
    assert de(folded.mean_f, 4) == "0,2462" and "0,2462 statt 0,35" in README and len(rows) == 6


def test_readme_mean_and_edge_numbers():
    res = get("mean")
    by = {r["c"]: r for r in res["mean"]}
    assert (de(by[0.5]["mean"], 4), de(by[0.5]["f_min"], 3), de(by[0.5]["f_max"], 3), de(by[0.9]["mean"], 4), de(by[1.1]["mean"], 4)) == ("0,1992", "0,134", "0,345", "0,0586", "0,0000")
    assert [de(r["rmse"], 4) for r in res["edge"]] == ["0,0104", "0,0020", "0,0004", "0,0022", "0,0507"] and res["edge"][-1]["hi"] - res["edge"][-1]["lo"] == 15
    for s in ("0,1992 (Spanne 0,134 bis 0,345)", "0,0586", "0,0000", "0,0104", "0,0004", "0,0507", "letzten 15 Werten"):
        assert s in README


def test_readme_smoothing_numbers():
    rows = {r["snr"]: r for r in get("smooth")}
    r10, r0 = rows[10], rows[0]
    assert [de(r10[k]) for k in ("plain", "l5", "l11", "l21", "bandpass", "bandpass_l11")] == ["0,0343", "0,0103", "0,0047", "0,0025", "0,0173", "0,0036"]
    assert de(r0["plain"]) == "0,1572" and de(r0["l21"]) == "0,0388"
    for s in ("0,0343", "0,0103 / 0,0047 / 0,0025", "0,0173", "0,0036", "0,1572 gegen 0,0388"):
        assert s in README


def test_readme_envelope_spectrum_numbers():
    rows = {r["snr"]: r for r in get("envelope")}
    assert (de(100 * rows[-6]["envelope"], 1), de(100 * rows[-6]["direct"], 1), de(100 * rows[-3]["envelope"], 1), de(100 * rows[-3]["direct"], 1)) == ("40,5", "69,2", "93,8", "98,2")
    assert all(rows[s]["envelope"] == 1.0 and rows[s]["direct"] == 1.0 for s in (0, 3, 6, 10))
    for s in ("40,5 %", "69,2 %", "93,8 % gegen 98,2 %"):
        assert s in README


def test_readme_method_remarks():
    assert "Seeds ab 910000" in README and C.EXP_CHIRP_SEED == 910000 and "höchstens 2,5 Prozentpunkte" in README and np.sqrt(0.25 / 400) * 100 <= 2.5
    assert C.EXP_CHIRP_RUNS == 300 and C.EXP_ENV_RUNS == 400


def test_the_app_text_the_hilbert_frequency_is_exact_for_chirp_and_fm_at_half_times():
    a = ev.analyse(ev.Settings(kind="Frequenzmodulation", p1=0.2, p2=0.02, p3=3.0))
    assert a.rmse_inner < 2e-3
    b = ev.analyse(dataclasses.replace(ev.Settings(), n=1024))
    assert b.rmse_inner < 2e-3
