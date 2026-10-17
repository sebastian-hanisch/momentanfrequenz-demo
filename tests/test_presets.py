"""Presets: Vollständigkeit, gültige Werte, Standardwerte, Parameterschlüssel."""

import mfq_constants as C
import mfq_evaluation as ev
import mfq_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS) | {"params"}, name
        assert C.PRESET_HELP[name] and len(preset["params"]) == 3


def test_preset_values_are_inside_the_choices_and_bounds():
    for name, p in C.PRESETS.items():
        assert p["n"] in C.N_CHOICES and p["kind"] in C.SIGNAL_TYPES and p["noise"] in C.NOISE_MODES and p["smooth"] in C.SMOOTH_CHOICES and p["bandpass"] in C.BANDPASS_MODES, name
        assert C.SNR_MIN <= p["snr"] <= C.SNR_MAX and 0 <= p["seed"] <= C.SEED_MAX
        for (label, lo, hi, step, default, help_text), v in zip(C.PARAMS[p["kind"]], p["params"]):
            assert lo <= v <= hi, (name, label)


def test_default_settings_are_valid_choices():
    s = ev.Settings()
    assert s.n in C.N_CHOICES and s.kind in C.SIGNAL_TYPES and s.noise in C.NOISE_MODES and s.smooth in C.SMOOTH_CHOICES and s.bandpass in C.BANDPASS_MODES
    for (label, lo, hi, step, default, help_text), v in zip(C.PARAMS[s.kind], (s.p1, s.p2)):
        assert lo <= v <= hi and v == default


def test_every_signal_type_has_two_or_three_parameters_with_valid_ranges():
    assert set(C.PARAMS) == set(C.SIGNAL_TYPES)
    for kind, specs in C.PARAMS.items():
        assert 2 <= len(specs) <= 3
        for label, lo, hi, step, default, help_text in specs:
            assert lo < hi and lo <= default <= hi and step > 0 and label and help_text


def test_setting_specs_cover_every_preset_key_and_every_parameter():
    assert set(P.PRESET_KEYS.values()) <= set(P.SETTING_SPECS)
    expected = {P.param_key(ti, slot) for ti, kind in enumerate(C.SIGNAL_TYPES) for slot in range(1, len(C.PARAMS[kind]) + 1)}
    assert expected <= set(P.SETTING_SPECS)
    s = ev.Settings()
    defaults = {"n": s.n, "kind": s.kind, "noise": s.noise, "snr": s.snr, "smooth": s.smooth, "bandpass": s.bandpass, "seed": s.seed}
    for key, state_key in P.PRESET_KEYS.items():
        assert P.SETTING_SPECS[state_key].default == defaults[key], key


def test_url_params_are_unique_and_preset_names_fit_the_buttons():
    params = [spec.url_param for spec in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params)) and max(len(n) for n in C.PRESETS) <= 60
    assert P.param_key(2, 3) == "p23_kept"
