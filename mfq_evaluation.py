"""Analyse, Kennzahlen, Experimente (feste Ströme) und Verdikte der Demo. Reine Funktionen; Zufall nur über SplitMix64."""

import math
from dataclasses import dataclass

import numpy as np

import mfq_analytic as A
import mfq_constants as C
import mfq_envelope as E
import mfq_signals as S
import signal_generator as G


def de(x: float, digits: int = 1) -> str:
    """Zahl mit Dezimalkomma."""
    return f"{x:.{digits}f}".replace(".", ",")


def noise_sigma(snr_db: float) -> float:
    """Standardabweichung des Rauschens gegen eine Linie der Amplitude 1 (Leistung 1/2): sqrt(0,5 / 10^(SNR/10))."""
    return math.sqrt(0.5 / 10.0 ** (snr_db / 10.0))


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    kind: str = C.DEFAULT_TYPE
    p1: float = C.PARAMS[C.DEFAULT_TYPE][0][4]
    p2: float = C.PARAMS[C.DEFAULT_TYPE][1][4]
    p3: float = 0.0
    noise: str = C.DEFAULT_NOISE
    snr: int = C.DEFAULT_SNR
    smooth: int = C.DEFAULT_SMOOTH
    bandpass: str = C.DEFAULT_BANDPASS
    seed: int = C.DEFAULT_SEED

    @property
    def has_noise(self) -> bool:
        return self.noise == "an"


def make_signal(s: Settings) -> S.TestSignal:
    """Das saubere Signal des gewählten Typs (Anfangsphase PHASE)."""
    if s.kind == "Chirp":
        return S.chirp(s.n, s.p1, s.p2, C.PHASE)
    if s.kind == "Amplitudenmodulation":
        return S.am_signal(s.n, s.p1, s.p2, s.p3, C.PHASE)
    if s.kind == "Frequenzmodulation":
        return S.fm_signal(s.n, s.p1, s.p2, s.p3, C.PHASE)
    if s.kind == "Zwei Kosinus":
        return S.two_tones(s.n, s.p1, s.p2, s.p3, C.PHASE)
    if s.kind == "Kosinus mit Mittelwert":
        return S.mean_tone(s.n, s.p1, s.p2, C.PHASE)
    raise ValueError(f"unbekannter Signaltyp: {s.kind}")


def signal_band(s: Settings, sig: S.TestSignal) -> tuple[float, float]:
    """Durchlassbereich des Bandpasses: um die Komponentenfrequenzen mit dem Rand BAND_MARGIN (bei Frequenzmodulation um den Frequenzhub erweitert), auf (0, 0,5) begrenzt."""
    comps = [c for c in sig.components if c > 0]
    lo, hi = min(comps), max(comps)
    if s.kind == "Frequenzmodulation":
        lo, hi = lo - (s.p3 + 1.0) * s.p2, hi + (s.p3 + 1.0) * s.p2
    return max(lo - C.BAND_MARGIN, 0.005), min(hi + C.BAND_MARGIN, 0.5)


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    clean: S.TestSignal
    x: np.ndarray                   # gemessenes Signal (mit Rauschen)
    z: np.ndarray                   # analytisches Signal (nach dem Bandpass, falls an)
    env_est: np.ndarray
    f_est: np.ndarray               # Momentanfrequenz an den halben Zeitpunkten, nach der Glättung
    f_true: np.ndarray | None       # wahre Momentanfrequenz gefaltet auf (-0,5, 0,5]
    band: tuple | None
    rmse_inner: float | None
    rmse_edge: float | None
    env_rmse_inner: float | None
    negative_share: float
    mean_f: float
    f_min: float
    f_max: float


def build_measurement(s: Settings) -> tuple[S.TestSignal, np.ndarray]:
    sig = make_signal(s)
    x = sig.x.copy()
    if s.has_noise:
        x = x + noise_sigma(s.snr) * G.SplitMix64(s.seed).normal(s.n)
    return sig, x


def _rmse(v) -> float:
    return float(math.sqrt(np.mean(np.square(v))))


def analyse(s: Settings) -> Analysis:
    sig, x = build_measurement(s)
    band = signal_band(s, sig) if s.bandpass == "an" else None
    x_used = A.bandpass(x, *band) if band is not None else x
    z = A.analytic_signal(x_used)
    f_est = A.smooth(A.inst_frequency(z), s.smooth)
    env_est = A.envelope(z)
    f_true = A.fold(sig.f_true) if sig.f_true is not None else None
    e = C.EDGE
    rmse_inner = rmse_edge = env_rmse = None
    if f_true is not None:
        err = f_est - f_true
        rmse_inner = _rmse(err[e:-e])
        rmse_edge = _rmse(np.concatenate([err[:e], err[-e:]]))
    if sig.env_true is not None:
        env_rmse = _rmse((env_est - sig.env_true)[e:-e])
    return Analysis(s, sig, x, z, env_est, f_est, f_true, band, rmse_inner, rmse_edge, env_rmse, float(np.mean(f_est < 0)), float(f_est.mean()), float(f_est.min()), float(f_est.max()))


# --- Experimente ---------------------------------------------------------------------------------------------------------------------------


def sweep_chirp(snrs=C.EXP_CHIRP_SNR, runs: int = C.EXP_CHIRP_RUNS, n: int = C.EXP_CHIRP_N) -> list[dict]:
    """Chirp in weißem Rauschen: RMSE der Momentanfrequenz im Inneren und am Rand (je EDGE Werte) und Anteil der inneren Werte mit Fehler über EXP_CHIRP_OVER."""
    sig = S.chirp(n, C.EXP_CHIRP_F0, C.EXP_CHIRP_F1)
    e = C.EDGE
    rows = []
    for snr in snrs:
        rng = G.SplitMix64(C.EXP_CHIRP_SEED + snr)
        x = sig.x[None, :] + noise_sigma(snr) * rng.normal(runs * n).reshape(runs, n)
        err = A.inst_frequency(A.analytic_signal(x)) - sig.f_true[None, :]
        rows.append({"snr": snr, "inner": _rmse(err[:, e:-e]), "edge": _rmse(np.concatenate([err[:, :e], err[:, -e:]], axis=1)), "over": float(np.mean(np.abs(err[:, e:-e]) > C.EXP_CHIRP_OVER)),
                     "runs": runs})
    return rows


def sweep_bedrosian(ks=C.EXP_BED_KS, n: int = C.EXP_BED_N) -> list[dict]:
    """Amplitudenmodulation mit bingenauem Träger: Fehler von Hüllkurve und Momentanfrequenz gegen fm / fc (periodische Fassung, keine Randeffekte)."""
    fc = C.EXP_BED_FC_BIN / n
    rows = []
    for k in ks:
        fm = k / n
        sig = S.am_signal(n, fc, fm, C.EXP_BED_DEPTH, C.EXP_BED_PHASE)
        z = A.analytic_signal(sig.x)
        f = A.inst_frequency(z, periodic=True)
        env_err = np.abs(A.envelope(z) - sig.env_true)
        rows.append({"ratio": fm / fc, "fm": fm, "env_max": float(env_err.max()), "env_rmse": _rmse(env_err), "f_rmse": _rmse(f - fc), "f_min": float(f.min()), "f_max": float(f.max()),
                     "lowest_band": fc - fm})
    return rows


def sweep_tones(rs=C.EXP_TONES_RS, n: int = C.EXP_TONES_N) -> list[dict]:
    """Zwei Kosinus (bingenau, Amplituden 1 und r): Spanne der Momentanfrequenz, Anteil außerhalb von [f1, f2], Anteil negativ und zeitliches Mittel gegen f1 und die leistungsgewichtete Frequenz."""
    f1, f2 = C.EXP_TONES_F1_BIN / n, C.EXP_TONES_F2_BIN / n
    rows = []
    for r in rs:
        f = A.inst_frequency(A.analytic_signal(S.two_tones(n, f1, f2, r, C.EXP_TONES_PHASE).x), periodic=True)
        rows.append({"r": r, "f_min": float(f.min()), "f_max": float(f.max()), "outside": float(np.mean((f < f1 - 1e-9) | (f > f2 + 1e-9))), "negative": float(np.mean(f < 0)), "mean": float(f.mean()),
                     "weighted": (f1 + r * r * f2) / (1.0 + r * r), "f1": f1, "f2": f2})
    return rows


def sweep_mean(cs=C.EXP_MEAN_CS, n: int = C.EXP_MEAN_N) -> dict:
    """Mittelwert c plus Kosinus (bingenau): mittlere Momentanfrequenz und Spanne gegen c; dazu der Randeffekt bei einem nicht bingenauen Kosinus (RMSE je Abschnitt der Reihe)."""
    f0 = C.EXP_MEAN_F_BIN / n
    mean_rows = []
    for c in cs:
        f = A.inst_frequency(A.analytic_signal(S.mean_tone(n, f0, c, C.EXP_MEAN_PHASE).x), periodic=True)
        mean_rows.append({"c": c, "mean": float(f.mean()), "f_min": float(f.min()), "f_max": float(f.max()), "f_true": f0})
    f_off = C.EXP_EDGE_F_BIN / n
    f = A.inst_frequency(A.analytic_signal(S.mean_tone(n, f_off, 0.0, C.EXP_MEAN_PHASE).x))
    edge_rows = [{"lo": lo, "hi": hi, "rmse": _rmse(f[lo:hi] - f_off)} for lo, hi in C.EXP_EDGE_SEGMENTS]
    return {"mean": mean_rows, "edge": edge_rows, "f": f0}


def sweep_smooth(snrs=C.EXP_SMOOTH_SNR, runs: int = C.EXP_SMOOTH_RUNS, n: int = C.EXP_SMOOTH_N) -> list[dict]:
    """Chirp in Rauschen: RMSE der Momentanfrequenz im Inneren (ohne je EXP_SMOOTH_EDGE Randwerte) ohne Maßnahme, mit gleitendem Mittelwert (L = 5, 11, 21), mit Bandpass und mit beidem."""
    sig = S.chirp(n, C.EXP_CHIRP_F0, C.EXP_CHIRP_F1)
    edge = C.EXP_SMOOTH_EDGE
    rows = []
    for snr in snrs:
        rng = G.SplitMix64(C.EXP_SMOOTH_SEED + snr)
        x = sig.x[None, :] + noise_sigma(snr) * rng.normal(runs * n).reshape(runs, n)
        xb = A.bandpass(x, *C.EXP_SMOOTH_BAND)
        out = {"snr": snr, "runs": runs}
        for key, data, length in (("plain", x, 1), ("l5", x, 5), ("l11", x, 11), ("l21", x, 21), ("bandpass", xb, 1), ("bandpass_l11", xb, 11)):
            err = A.smooth(A.inst_frequency(A.analytic_signal(data)), length) - sig.f_true[None, :]
            out[key] = _rmse(err[:, edge:-edge])
        rows.append(out)
    return rows


def sweep_envelope(snrs=C.EXP_ENV_SNR, runs: int = C.EXP_ENV_RUNS, n: int = C.EXP_ENV_N) -> list[dict]:
    """Einfache Amplitudenmodulation in Rauschen: Anteil der Läufe, in denen das Hüllkurvenspektrum bzw. das direkte Spektrum (Seitenbänder fc ± fm) die Modulationsfrequenz am höchsten zeigt."""
    fc, fm = C.EXP_ENV_FC_BIN / n, C.EXP_ENV_FM_BIN / n
    clean = S.am_signal(n, fc, fm, C.EXP_ENV_DEPTH).x
    tol = C.EXP_ENV_TOL_BINS / n
    rows = []
    for snr in snrs:
        rng = G.SplitMix64(C.EXP_ENV_SEED + snr)
        x = clean[None, :] + noise_sigma(snr) * rng.normal(runs * n).reshape(runs, n)
        f_env, spec_env = E.envelope_spectrum(x, *C.EXP_ENV_BAND)
        f_dir, spec_dir = E.direct_spectrum(x, 0.0, 0.5)
        band = (np.abs(f_dir - fc) <= C.EXP_ENV_BAND[1]) & (np.abs(f_dir - fc) >= C.EXP_ENV_CARRIER_GAP)
        peak_dir = f_dir[band][np.argmax(spec_dir[:, band], axis=1)]
        rows.append({"snr": snr, "envelope": float(np.mean(np.abs(f_env[np.argmax(spec_env, axis=1)] - fm) < tol)), "direct": float(np.mean(np.abs(np.abs(peak_dir - fc) - fm) < tol)), "runs": runs})
    return rows


# --- Verdikte (reine Funktionen: (Art, Text)) ----------------------------------------------------------------------------------------------


def verdict_main(a: Analysis) -> tuple[str, str]:
    s = a.settings
    if s.kind == "Zwei Kosinus":
        f1, f2 = s.p1, s.p2
        strong = f1 if s.p3 <= 1.0 else f2
        head = (f"Zwei Komponenten (bei {de(f1, 3)} und {de(f2, 3)}): Die Momentanfrequenz schwankt zwischen {de(a.f_min, 3)} und {de(a.f_max, 3)}, {de(100 * a.negative_share, 1)} % der Zeit negativ. ")
        if abs(a.mean_f - strong) <= C.OK_RMSE:
            return "warning", head + f"Ihr Mittel ({de(a.mean_f, 3)}) liegt bei der stärkeren Komponente ({de(strong, 3)}), nicht dazwischen."
        return "warning", head + (f"Ihr Mittel ({de(a.mean_f, 3)}) liegt nicht bei der stärkeren Komponente ({de(strong, 3)}): Überschreitet die wahre Frequenz des Summenzeigers die Nyquist-Frequenz, "
                                  "faltet der Phasenschritt, und das Mittel verfehlt beide Komponenten.")
    err = a.rmse_inner
    tail = f" Fehler innen {de(err, 4)}, an den Rändern {de(a.rmse_edge, 4)}; Hüllkurve innen {de(a.env_rmse_inner, 4)}."
    if s.kind == "Kosinus mit Mittelwert":
        if abs(a.mean_f - s.p1) < C.OK_RMSE:
            return "success", f"Mittlere Momentanfrequenz {de(a.mean_f, 3)} bei wahr {de(s.p1, 3)} (Mittelwert {de(s.p2, 1)} unter der Amplitude 1)." + tail
        return "warning", f"Mittlere Momentanfrequenz {de(a.mean_f, 3)} statt {de(s.p1, 3)}: Der Mittelwert {de(s.p2, 1)} verschiebt den Zeiger, die Phase windet nicht mehr um den Ursprung." + tail
    if err < C.OK_RMSE:
        return "success", "Die Phasenableitung trifft die wahre Momentanfrequenz." + tail
    return "warning", "Die Phasenableitung weicht von der wahren Momentanfrequenz ab." + tail


def verdict_chirp(rows: list[dict]) -> tuple[str, str]:
    by = {r["snr"]: r for r in rows}
    hi, mid, lo = max(by), 10, min(by)
    return "info", (f"Bei {hi} dB liegt der Fehler der Momentanfrequenz im Inneren bei {de(by[hi]['inner'], 4)}, an den Rändern bei {de(by[hi]['edge'], 3)}; bei {mid} dB bei {de(by[mid]['inner'], 3)} und bei {lo} dB bei "
                    f"{de(by[lo]['inner'], 2)}. **Das Ableiten der Phase verstärkt das Rauschen, und die Ränder tragen bei diesem Chirp (nicht bingenau) einen Fehler von etwa 0,05 und mehr.**")


def verdict_bedrosian(rows: list[dict]) -> tuple[str, str]:
    ok = [r for r in rows if r["ratio"] < 1.0]
    at = [r for r in rows if abs(r["ratio"] - 1.0) < 1e-9][0]
    return "warning", (f"Bis fm/fc = {de(max(r['ratio'] for r in ok), 2)} sind Hüllkurve und Momentanfrequenz exakt (Fehler {de(max(r['env_max'] for r in ok), 4)}); bei fm = fc liegt der größte Hüllkurvenfehler bei "
                       f"{de(at['env_max'], 2)} und die Momentanfrequenz zwischen {de(at['f_min'], 3)} und {de(at['f_max'], 3)}. **In dieser Anordnung darf das untere Seitenband den Nullpunkt nicht erreichen; zusätzlich müssen das obere Seitenband unter der Nyquist-Frequenz und die Tiefe unter 1 bleiben.**")


def verdict_tones(rows: list[dict]) -> tuple[str, str]:
    by = {r["r"]: r for r in rows}
    one = by[1.0]
    return "warning", (f"Bei gleich starken Komponenten schwankt die Momentanfrequenz zwischen {de(one['f_min'], 3)} und {de(one['f_max'], 3)} und ist in {de(100 * one['negative'], 1)} % der Zeit negativ; "
                       f"bei r = {de(max(r for r in by if r < 1.0), 1)} reicht sie von {de(by[max(r for r in by if r < 1.0)]['f_min'], 3)} bis {de(by[max(r for r in by if r < 1.0)]['f_max'], 3)}. Ihr Mittel ist in allen gemessenen Fällen "
                       f"{de(one['mean'], 4)}, die Frequenz der ersten Komponente (bei r < 1 die stärkere), nicht die leistungsgewichtete ({de(one['weighted'], 4)} bei r = 1). **Mit zwei vergleichbar starken Komponenten ist sie keine Frequenz mehr.**")


def verdict_mean(res: dict) -> tuple[str, str]:
    by = {r["c"]: r for r in res["mean"]}
    edge = res["edge"]
    return "warning", (f"Bei Mittelwert 0,5 (gegen Amplitude 1) liegt die mittlere Momentanfrequenz noch bei {de(by[0.5]['mean'], 4)}, bei 1,1 bei {de(by[1.1]['mean'], 4)} (wahr {de(res['f'], 4)}). "
                       f"Am Ende der Reihe ist der Fehler {de(edge[-1]['rmse'], 3)}, im Inneren {de(edge[2]['rmse'], 4)}. **Mittelwert abziehen und die Ränder verwerfen.**")


def verdict_smooth(rows: list[dict]) -> tuple[str, str]:
    by = {r["snr"]: r for r in rows}
    return "info", (f"Bei 10 dB: einfach {de(by[10]['plain'], 4)}, gleitender Mittelwert über 21 Werte {de(by[10]['l21'], 4)}, Bandpass {de(by[10]['bandpass'], 4)}, Bandpass und Mittelwert über 11 Werte zusammen {de(by[10]['bandpass_l11'], 4)}. "
                    "**Glättung hilft stark, ist aber nur bei linearem Frequenzverlauf unverzerrt.**")


def verdict_envelope(rows: list[dict]) -> tuple[str, str]:
    by = {r["snr"]: r for r in rows}
    low = min(by)
    return "info", (f"Bei {low} dB findet das Hüllkurvenspektrum die Modulationsfrequenz in {de(100 * by[low]['envelope'], 0)} % der Läufe, das direkte Spektrum (Seitenbänder) in {de(100 * by[low]['direct'], 0)} %. "
                    "**Bei einfacher Modulation (Tiefe 0,3) bringt das Hüllkurvenspektrum in dieser Messung keinen Gewinn; es zeigt fm nur direkt statt als Abstand der Seitenbänder.**")
