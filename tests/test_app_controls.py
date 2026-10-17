"""AppTest: Auswahlfelder, Permalink, bedingte Regler, PDF (die Rauchtests in test_app.py decken nur Knöpfe und Slider ab)."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import mfq_constants as C
from mfq_presets import param_key

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 300
SNR = "Rauschabstand (dB)"


def _run(params=None):
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    if params:
        at.query_params.update(params)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _labels(at):
    return [x.label for x in at.slider]


@pytest.mark.parametrize("kind", list(C.SIGNAL_TYPES))
def test_every_signal_type_runs_and_shows_its_parameter_sliders(kind):
    at = _run()
    at.selectbox(key="kind_select").set_value(kind).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    labels = _labels(at)
    assert all(spec[0] in labels for spec in C.PARAMS[kind])
    others = {spec[0] for k, specs in C.PARAMS.items() if k != kind for spec in specs} - {spec[0] for spec in C.PARAMS[kind]}
    assert not (others & set(labels))


def test_the_noise_slider_exists_only_with_noise_and_keeps_its_value():
    at = _run()
    assert SNR not in _labels(at)
    at.radio(key="noise_select").set_value("an").run(timeout=TIMEOUT)
    assert not at.exception and SNR in _labels(at)
    at2 = _run({"r": "an", "snr": "7"})
    at2.radio(key="noise_select").set_value("aus").run(timeout=TIMEOUT)
    at2.radio(key="noise_select").set_value("an").run(timeout=TIMEOUT)
    assert {x.label: x for x in at2.slider}[SNR].value == 7


def test_parameters_are_kept_per_type_when_switching():
    at = _run({"q01": "0.15", "q02": "0.35"})
    by = {x.label: x for x in at.slider}
    assert by["Startfrequenz"].value == 0.15 and by["Endfrequenz"].value == 0.35
    at.selectbox(key="kind_select").set_value("Zwei Kosinus").run(timeout=TIMEOUT)
    at.selectbox(key="kind_select").set_value("Chirp").run(timeout=TIMEOUT)
    by = {x.label: x for x in at.slider}
    assert by["Startfrequenz"].value == 0.15 and by["Endfrequenz"].value == 0.35


def test_permalink_is_applied_and_invalid_values_are_ignored():
    at = _run({"n": "1024", "k": "Frequenzmodulation", "q21": "0.3", "q22": "9", "q23": "99", "r": "an", "snr": "-99", "g": "11", "b": "an", "seed": "42"})
    assert at.select_slider(key="n_select").value == 1024 and at.selectbox(key="kind_select").value == "Frequenzmodulation" and at.radio(key="noise_select").value == "an"
    assert at.select_slider(key="smooth_select").value == 11 and at.radio(key="bandpass_select").value == "an" and at.number_input(key="seed_input").value == 42
    by = {x.label: x for x in at.slider}
    assert by["Trägerfrequenz fc"].value == 0.3 and by["Modulationsfrequenz fm"].value == 0.2 and by["Phasenhub β"].value == 10.0 and by[SNR].value == C.SNR_MIN
    at2 = _run({"n": "3000", "k": "unsinn", "r": "vielleicht", "g": "7", "b": "x"})
    assert at2.select_slider(key="n_select").value == C.DEFAULT_N and at2.selectbox(key="kind_select").value == C.DEFAULT_TYPE and at2.radio(key="noise_select").value == C.DEFAULT_NOISE
    assert at2.select_slider(key="smooth_select").value == C.DEFAULT_SMOOTH and at2.radio(key="bandpass_select").value == C.DEFAULT_BANDPASS


def test_preset_button_sets_the_widgets():
    at = _run()
    at.button(key="preset_Zwei gleich starke Töne: negative Frequenzen").click().run(timeout=TIMEOUT)
    assert not at.exception
    by = {x.label: x for x in at.slider}
    assert at.selectbox(key="kind_select").value == "Zwei Kosinus" and by["Frequenz 2"].value == 0.25 and by["Amplitude r der zweiten"].value == 1.0
    at.button(key="preset_Chirp mit Rauschen: die Ableitung verstärkt es").click().run(timeout=TIMEOUT)
    assert at.selectbox(key="kind_select").value == "Chirp" and at.radio(key="noise_select").value == "an" and {x.label: x for x in at.slider}[SNR].value == 10


def test_pdf_button_builds_a_download():
    at = _run()
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert any(d.proto.label.startswith("⬇️") for d in at.get("download_button"))


def test_the_dice_button_changes_the_seed():
    at = _run()
    before = at.number_input(key="seed_input").value
    [b for b in at.button if "würfeln" in b.label][0].click().run(timeout=TIMEOUT)
    assert not at.exception and at.number_input(key="seed_input").value != before
    assert param_key(0, 1) == "p01_kept"
