"""Konstanten: Regler-Grenzen, Signaltypen, feste Annahmen, Experimente, Farben und Presets (Kern: mfq_analytic.py, mfq_signals.py, mfq_envelope.py)."""

N_CHOICES = (128, 256, 512, 1024, 2048)
DEFAULT_N = 512
SIGNAL_TYPES = ("Chirp", "Amplitudenmodulation", "Frequenzmodulation", "Zwei Kosinus", "Kosinus mit Mittelwert")
DEFAULT_TYPE = "Chirp"
# Je Signaltyp bis zu drei Parameter: (Beschriftung, Minimum, Maximum, Schritt, Standardwert, Hilfetext)
PARAMS = {
    "Chirp": (("Startfrequenz", 0.02, 0.48, 0.005, 0.1, "Frequenz am Anfang (Zyklen je Abtastwert)."), ("Endfrequenz", 0.02, 0.48, 0.005, 0.3, "Frequenz am Ende; dazwischen steigt sie linear.")),
    "Amplitudenmodulation": (("Trägerfrequenz fc", 0.05, 0.45, 0.005, 0.2, "Frequenz des Trägers."), ("Modulationsfrequenz fm", 0.005, 0.45, 0.005, 0.05, "Frequenz der Hüllkurve; das untere Seitenband liegt bei fc − fm."),
                             ("Modulationstiefe", 0.0, 1.0, 0.05, 0.5, "Hüllkurve 1 + Tiefe · cos(2π fm t).")),
    "Frequenzmodulation": (("Trägerfrequenz fc", 0.05, 0.45, 0.005, 0.2, "Mittlere Frequenz."), ("Modulationsfrequenz fm", 0.005, 0.2, 0.005, 0.02, "Frequenz, mit der die Frequenz schwankt."),
                           ("Phasenhub β", 0.5, 10.0, 0.5, 3.0, "Hub der Phase in Radiant; der Frequenzhub ist etwa β · fm.")),
    "Zwei Kosinus": (("Frequenz 1", 0.02, 0.45, 0.005, 0.2, "Frequenz der Komponente mit Amplitude 1."), ("Frequenz 2", 0.02, 0.48, 0.005, 0.25, "Frequenz der zweiten Komponente."),
                     ("Amplitude r der zweiten", 0.05, 1.5, 0.05, 0.5, "Amplitude der zweiten Komponente gegen die erste (1).")),
    "Kosinus mit Mittelwert": (("Frequenz", 0.02, 0.48, 0.005, 0.2, "Frequenz des Kosinus (Amplitude 1)."), ("Mittelwert c", 0.0, 3.0, 0.1, 0.5, "Konstanter Anteil gegen die Amplitude 1.")),
}
NOISE_MODES = ("aus", "an")
DEFAULT_NOISE = "aus"
SNR_MIN, SNR_MAX, DEFAULT_SNR = -5, 40, 20                                    # Linie der Amplitude 1 (Leistung 1/2) gegen die Rauschleistung, in dB
SMOOTH_CHOICES = (1, 5, 11, 21)
DEFAULT_SMOOTH = 1
BANDPASS_MODES = ("aus", "an")
DEFAULT_BANDPASS = "aus"
SEED_MAX, DEFAULT_SEED = 999999, 39
PHASE = 0.3                                                                   # feste Anfangsphase der Signale der App
EDGE = 16                                                                     # Randwerte je Seite, die für „innen“ ausgespart werden
BAND_MARGIN = 0.05                                                            # Bandpass: Rand um die Komponentenfrequenzen
OK_RMSE = 0.01                                                                # Fehler der Momentanfrequenz, unter dem die Schätzung „genau“ heißt

# Experimente (feste Ströme; Rauschvarianz 0,5 / 10^(SNR/10) gegen eine Linie der Amplitude 1)
EXP_CHIRP_N, EXP_CHIRP_F0, EXP_CHIRP_F1, EXP_CHIRP_RUNS, EXP_CHIRP_SEED = 512, 0.1, 0.3, 300, 910000
EXP_CHIRP_SNR = (40, 30, 20, 10, 5, 0)
EXP_CHIRP_OVER = 0.05                                                         # Fehler, über dem ein Wert „daneben“ heißt
EXP_BED_N, EXP_BED_FC_BIN, EXP_BED_DEPTH, EXP_BED_PHASE = 512, 102, 0.5, 0.7
EXP_BED_KS = (4, 10, 26, 51, 80, 102, 130)
EXP_TONES_N, EXP_TONES_F1_BIN, EXP_TONES_F2_BIN, EXP_TONES_PHASE = 512, 102, 128, 0.4
EXP_TONES_RS = (0.05, 0.1, 0.3, 0.5, 0.8, 1.0)
EXP_MEAN_N, EXP_MEAN_F_BIN, EXP_MEAN_PHASE = 512, 102, 0.2
EXP_MEAN_CS = (0.0, 0.25, 0.5, 0.9, 1.1, 2.0)
EXP_EDGE_F_BIN = 102.4                                                        # nicht bingenau: Randeffekt
EXP_EDGE_SEGMENTS = ((0, 16), (16, 64), (64, 448), (448, 496), (496, 511))
EXP_SMOOTH_N, EXP_SMOOTH_RUNS, EXP_SMOOTH_SEED, EXP_SMOOTH_EDGE = 512, 300, 920000, 32
EXP_SMOOTH_SNR = (20, 10, 5, 0)
EXP_SMOOTH_BAND = (0.05, 0.35)
EXP_ENV_N, EXP_ENV_FC_BIN, EXP_ENV_FM_BIN, EXP_ENV_DEPTH, EXP_ENV_RUNS, EXP_ENV_SEED = 2048, 640, 32, 0.3, 400, 930000
EXP_ENV_SNR = (-6, -3, 0, 3, 6, 10)
EXP_ENV_BAND, EXP_ENV_TOL_BINS, EXP_ENV_CARRIER_GAP = (0.002, 0.1), 1.5, 0.002

COLOR_EST = "rgba(76,120,168,0.95)"
COLOR_TRUE = "#B03A2E"
COLOR_SIGNAL = "rgba(120,120,120,0.9)"
COLOR_EDGE = "rgba(232,131,58,0.15)"
COLOR_AUX = "#4C9F70"

PRESETS = {
    "Chirp: sauber gemessen": {"n": 512, "kind": "Chirp", "params": (0.1, 0.3, 0.0), "noise": "aus", "snr": 20, "smooth": 1, "bandpass": "aus", "seed": DEFAULT_SEED},
    "Chirp mit Rauschen: die Ableitung verstärkt es": {"n": 512, "kind": "Chirp", "params": (0.1, 0.3, 0.0), "noise": "an", "snr": 10, "smooth": 1, "bandpass": "aus", "seed": DEFAULT_SEED},
    "Modulation: Bedrosian bricht, wenn fm den Träger erreicht": {"n": 512, "kind": "Amplitudenmodulation", "params": (0.2, 0.25, 0.5), "noise": "aus", "snr": 20, "smooth": 1, "bandpass": "aus", "seed": DEFAULT_SEED},
    "Zwei gleich starke Töne: negative Frequenzen": {"n": 512, "kind": "Zwei Kosinus", "params": (0.2, 0.25, 1.0), "noise": "aus", "snr": 20, "smooth": 1, "bandpass": "aus", "seed": DEFAULT_SEED},
    "Mittelwert: die Phase windet nicht mehr": {"n": 512, "kind": "Kosinus mit Mittelwert", "params": (0.2, 1.5, 0.0), "noise": "aus", "snr": 20, "smooth": 1, "bandpass": "aus", "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Chirp: sauber gemessen": "Linearer Chirp von 0,1 bis 0,3 ohne Rauschen: Die Phasenableitung trifft die wahre Frequenz im Inneren auf besser als 0,002, nur an den Rändern weicht sie ab.",
    "Chirp mit Rauschen: die Ableitung verstärkt es": "Derselbe Chirp bei 10 dB Rauschabstand: Das Ableiten der Phase verstärkt das Rauschen, der Fehler der Momentanfrequenz liegt bei etwa 0,03; Glättung und Bandpass helfen.",
    "Modulation: Bedrosian bricht, wenn fm den Träger erreicht": "Amplitudenmodulation mit fm = 0,25 über dem Träger 0,2: Das untere Seitenband liegt bei −0,05, das Signal ist nicht mehr schmalbandig; Hüllkurve und Momentanfrequenz weichen deutlich ab (bei fm = 0,05 sind sie auf 0,005 genau).",
    "Zwei gleich starke Töne: negative Frequenzen": "Zwei gleich starke Kosinus: Die Momentanfrequenz ist keine Frequenz mehr, sie springt und wird zeitweise sogar negativ.",
    "Mittelwert: die Phase windet nicht mehr": "Ein Kosinus auf einem Mittelwert 1,5 mal der Amplitude: Der Zeiger umkreist den Ursprung nicht mehr, die mittlere Momentanfrequenz bricht auf null zusammen.",
}
