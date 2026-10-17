"""Verdikte: reine Funktionen an handgesetzten Zeilen; Art, Schwellen und jede Zahl im Text."""

import numpy as np

import mfq_evaluation as ev
import mfq_signals as S


def make_analysis(kind="Chirp", p=(0.1, 0.3, 0.0), rmse_inner=0.001, rmse_edge=0.053, env=0.0042, neg=0.0, mean_f=0.1996, f_min=0.02, f_max=0.32):
    z = np.zeros(1)
    s = ev.Settings(kind=kind, p1=p[0], p2=p[1], p3=p[2])
    return ev.Analysis(s, S.TestSignal(z, None, None, ()), z, z, z, z, None, None, rmse_inner, rmse_edge, env, neg, mean_f, f_min, f_max)


def test_main_verdict_single_component_good_and_bad():
    kind, text = ev.verdict_main(make_analysis())
    assert kind == "success" and text == "Die Phasenableitung trifft die wahre Momentanfrequenz. Fehler innen 0,0010, an den Rändern 0,0530; Hüllkurve innen 0,0042."
    kind, text = ev.verdict_main(make_analysis(rmse_inner=0.0329, rmse_edge=0.1329, env=0.2307))
    assert kind == "warning" and text == "Die Phasenableitung weicht von der wahren Momentanfrequenz ab. Fehler innen 0,0329, an den Rändern 0,1329; Hüllkurve innen 0,2307."
    assert ev.verdict_main(make_analysis(rmse_inner=0.0099))[0] == "success" and ev.verdict_main(make_analysis(rmse_inner=0.01))[0] == "warning"


def test_main_verdict_two_tones_names_the_stronger_component():
    a = make_analysis("Zwei Kosinus", (0.2, 0.25, 1.0), None, None, None, 0.051, 0.1992, -0.443, 0.39)
    kind, text = ev.verdict_main(a)
    assert kind == "warning"
    assert text == ("Zwei Komponenten (bei 0,200 und 0,250): Die Momentanfrequenz schwankt zwischen -0,443 und 0,390, 5,1 % der Zeit negativ. "
                    "Ihr Mittel (0,199) liegt bei der stärkeren Komponente (0,200), nicht dazwischen.")
    b = make_analysis("Zwei Kosinus", (0.2, 0.25, 1.5), None, None, None, 0.0, 0.25, 0.2, 0.3)
    assert "bei der stärkeren Komponente (0,250)" in ev.verdict_main(b)[1]


def test_main_verdict_mean():
    ok = make_analysis("Kosinus mit Mittelwert", (0.2, 0.5, 0.0), 0.001, 0.05, 0.004, 0.0, 0.1995)
    kind, text = ev.verdict_main(ok)
    assert kind == "success" and text.startswith("Mittlere Momentanfrequenz 0,200 bei wahr 0,200 (Mittelwert 0,5 unter der Amplitude 1).") and text.endswith("Hüllkurve innen 0,0040.")
    bad = make_analysis("Kosinus mit Mittelwert", (0.2, 1.5, 0.0), 0.2259, 0.222, 0.95, 0.4, -0.0)
    kind, text = ev.verdict_main(bad)
    assert kind == "warning" and text.startswith("Mittlere Momentanfrequenz -0,000 statt 0,200: Der Mittelwert 1,5 verschiebt den Zeiger, die Phase windet nicht mehr um den Ursprung.")
    assert ev.verdict_main(make_analysis("Kosinus mit Mittelwert", (0.2, 0.5, 0.0), mean_f=0.2105))[0] == "warning"


def test_verdict_chirp_text():
    rows = [{"snr": s, "inner": i, "edge": e, "over": 0.0, "runs": 5} for s, i, e in ((40, 0.0015, 0.0494), (10, 0.0346, 0.0641), (0, 0.1572, 0.1834))]
    kind, text = ev.verdict_chirp(rows)
    assert kind == "info"
    assert text == ("Bei 40 dB liegt der Fehler der Momentanfrequenz im Inneren bei 0,0015, an den Rändern bei 0,049; bei 10 dB bei 0,035 und bei 0 dB bei 0,16. "
                    "**Das Ableiten der Phase verstärkt das Rauschen, und die Ränder tragen immer einen Fehler.**")


def test_verdict_bedrosian_text():
    rows = [{"ratio": r, "fm": 0.1, "env_max": e, "env_rmse": 0.0, "f_rmse": 0.0, "f_min": lo, "f_max": hi, "lowest_band": 0.0} for r, e, lo, hi in ((0.5, 0.0, 0.199, 0.199), (0.7843, 0.0, 0.199, 0.199), (1.0, 0.161, 0.1565, 0.2341), (1.2745, 0.4953, 0.114, 0.2925))]
    kind, text = ev.verdict_bedrosian(rows)
    assert kind == "warning"
    assert text == ("Bis fm/fc = 0,78 sind Hüllkurve und Momentanfrequenz exakt (Fehler 0,0000); bei fm = fc liegt der größte Hüllkurvenfehler bei 0,16 und die Momentanfrequenz zwischen 0,157 und 0,234. "
                    "**Das untere Seitenband darf den Nullpunkt nicht erreichen.**")


def test_verdict_tones_text():
    rows = [{"r": r, "f_min": lo, "f_max": hi, "outside": 0.1, "negative": neg, "mean": 0.1992, "weighted": w, "f1": 0.1992, "f2": 0.25} for r, lo, hi, neg, w in ((0.5, 0.1497, 0.2161, 0.0, 0.2094), (0.8, 0.0261, 0.2218, 0.0, 0.219),
                                                                                                                        (1.0, -0.2754, 0.2246, 0.0508, 0.2246))]
    kind, text = ev.verdict_tones(rows)
    assert kind == "warning"
    assert text == ("Bei gleich starken Komponenten schwankt die Momentanfrequenz zwischen -0,275 und 0,225 und ist in 5,1 % der Zeit negativ; bei r = 0,8 reicht sie von 0,026 bis 0,222. "
                    "Ihr Mittel ist in allen Fällen 0,1992, die Frequenz der stärkeren (ersten) Komponente, nicht die leistungsgewichtete (0,2246 bei r = 1). **Mit zwei Komponenten ist sie keine Frequenz mehr.**")


def test_verdict_mean_text():
    res = {"mean": [{"c": c, "mean": m, "f_min": 0.0, "f_max": 0.0, "f_true": 0.1992} for c, m in ((0.5, 0.1992), (1.1, 0.0))], "edge": [{"lo": 0, "hi": 16, "rmse": 0.0104}, {"lo": 16, "hi": 64, "rmse": 0.002},
           {"lo": 64, "hi": 448, "rmse": 0.0004}, {"lo": 448, "hi": 496, "rmse": 0.0022}, {"lo": 496, "hi": 511, "rmse": 0.0507}], "f": 0.1992}
    kind, text = ev.verdict_mean(res)
    assert kind == "warning"
    assert text == ("Bei Mittelwert 0,5 (gegen Amplitude 1) liegt die mittlere Momentanfrequenz noch bei 0,1992, bei 1,1 bei 0,0000 (wahr 0,1992). Am Ende der Reihe ist der Fehler 0,051, im Inneren 0,0004. "
                    "**Mittelwert abziehen und die Ränder verwerfen.**")


def test_verdict_smooth_text():
    rows = [{"snr": 10, "runs": 5, "plain": 0.0343, "l5": 0.0103, "l11": 0.0047, "l21": 0.0025, "bandpass": 0.0173, "bandpass_l11": 0.0036}]
    kind, text = ev.verdict_smooth(rows)
    assert kind == "info"
    assert text == ("Bei 10 dB: einfach 0,0343, gleitender Mittelwert über 21 Werte 0,0025, Bandpass 0,0173, Bandpass und Mittelwert über 11 Werte zusammen 0,0036. "
                    "**Glättung hilft stark, ist aber nur bei linearem Frequenzverlauf unverzerrt.**")


def test_verdict_envelope_text():
    rows = [{"snr": s, "envelope": e, "direct": d, "runs": 5} for s, e, d in ((-6, 0.405, 0.6925), (0, 1.0, 1.0))]
    kind, text = ev.verdict_envelope(rows)
    assert kind == "info"
    assert text == ("Bei -6 dB findet das Hüllkurvenspektrum die Modulationsfrequenz in 40 % der Läufe, das direkte Spektrum (Seitenbänder) in 69 %. "
                    "**Bei einfacher Modulation bringt das Hüllkurvenspektrum keinen Gewinn; es zeigt fm nur direkt statt als Abstand der Seitenbänder.**")


def test_decimal_comma_and_noise_scale():
    assert ev.de(1.26, 1) == "1,3" and ev.de(3.0, 0) == "3" and ev.de(0.04, 3) == "0,040"
    assert ev.noise_sigma(0.0) == np.sqrt(0.5)
