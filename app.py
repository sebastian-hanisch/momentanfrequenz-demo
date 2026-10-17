"""Momentanfrequenz und analytisches Signal: Die Phasenableitung ist nur für ein Signal mit einer dominanten Komponente eine Frequenz - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 9 der Reihe „Spektral- und Wavelet-Analyse“. Fünf Testsignale mit exakt bekannter Wahrheit (Chirp, Amplitudenmodulation, Frequenzmodulation, zwei Kosinus, Kosinus mit Mittelwert) werden
über das analytische Signal (Hilbert-Transformation) in Hüllkurve und Momentanfrequenz zerlegt; gezeigt werden Rauschen, Randeffekte, die Bedrosian-Bedingung, die Mehrkomponenten-Grenze,
der Mittelwert, Glättung und das Hüllkurvenspektrum. Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import mfq_constants as C
import mfq_evaluation as ev
from mfq_evaluation import Settings, analyse, de
from mfq_pdf_export import build_pdf
from mfq_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    param_key,
    randomize_seed,
    sync_query_params,
)
from mfq_visualization import (
    build_bedrosian,
    build_chirp,
    build_envelope,
    build_envelope_spectrum,
    build_frequency,
    build_mean,
    build_signal,
    build_smooth,
    build_tones,
)

st.set_page_config(page_title="Momentanfrequenz – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner="Rechne die Experimente …")
def _experiment(name):
    return getattr(ev, "sweep_" + name)()


st.title("🌀 Momentanfrequenz: Wann ist die Phasenableitung eine Frequenz")
st.markdown(
    """
Das Periodogramm sagt, welche Frequenzen in der **ganzen** Aufzeichnung stecken. Ändert sich die Frequenz mit der Zeit (ein Chirp, ein Ton mit Vibrato), fragt man nach der **Momentanfrequenz**: der Frequenz zu einem Zeitpunkt.
Dafür baut man das **analytische Signal** $z = x + i\\,\\mathcal H\\{x\\}$ (die Hilbert-Transformierte ergänzt zu jedem Kosinus den zugehörigen Sinus), nimmt als **Hüllkurve** den Betrag $|z|$ und als Frequenz die
**Ableitung der Phase** $\\arg z$. Das ist eine gute Definition, aber nur für Signale mit **einer** dominanten, schmalbandigen Komponente. Die Demo zeigt, wo sie trägt und wo nicht.

Dies ist die neunte Demo der Reihe „Spektral- und Wavelet-Analyse“, auf das **Periodogramm** aufbauend; sie liefert das Orakel „wahre Momentanfrequenz“ für das Spektrogramm und die Wavelets danach. Alles ist **simuliert**:
Chirp, Modulation und Komponenten sind bekannt. Nicht zu verwechseln mit der „Envelope“ der Anomalieerkennung (robuste Kovarianz): hier ist die Hüllkurve die Einhüllende eines Signals.
Weiter unten: **🔬 Experimente**, **🚧 Wo die Annahmen enden** und **📐 Mathematische Formulierung**.
"""
)

with st.expander("So entsteht die Momentanfrequenz", expanded=True):
    st.markdown(
        """
1. **Analytisches Signal:** Die FFT des Signals wird bei negativen Frequenzen auf null gesetzt, bei positiven verdoppelt; zurück in die Zeit ergibt $z$ mit $\\operatorname{Re} z = x$.
2. **Hüllkurve und Phase:** $|z|$ ist die momentane Amplitude, $\\arg z$ die momentane Phase.
3. **Momentanfrequenz:** Die Phasenänderung von einem Wert zum nächsten, $\\arg(z_{t+1}\\bar z_t)/2\\pi$, gilt an den halben Zeitpunkten $t + \\tfrac12$.
4. **Bedingung (Bedrosian):** Hüllkurve und Phase lassen sich nur trennen, wenn das Spektrum der Hüllkurve unterhalb des Spektrums des Trägers liegt. Mit zwei Komponenten oder einem Mittelwert gilt das nicht mehr.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Signal**")
    n = st.select_slider("Länge n der Aufzeichnung", options=list(C.N_CHOICES), key="n_select", help="Zahl der Abtastwerte (Abtastrate 1, Frequenzen in Zyklen je Abtastwert, Nyquist-Frequenz 0,5).")
    kind = st.selectbox("Signaltyp", list(C.SIGNAL_TYPES), key="kind_select",
                        help="Chirp: Frequenz steigt linear. Amplitudenmodulation: Träger mit schwankender Amplitude. Frequenzmodulation: Träger mit schwankender Frequenz. Zwei Kosinus: zwei Komponenten. Kosinus mit Mittelwert: Konstante plus Kosinus.")
    ti = C.SIGNAL_TYPES.index(kind)
    values = []
    for slot, (label, lo, hi, step, default, help_text) in enumerate(C.PARAMS[kind], start=1):
        v = st.slider(label, lo, hi, float(st.session_state[param_key(ti, slot)]), step=step, format="%.3f" if step < 0.01 else "%.2f", help=help_text)
        st.session_state[param_key(ti, slot)] = float(v)
        values.append(float(v))
    while len(values) < 3:
        values.append(0.0)
    st.markdown("**Messung und Auswertung**")
    noise = st.radio("Rauschen", list(C.NOISE_MODES), key="noise_select", horizontal=True, help="Weißes Gaußrauschen zum Signal addiert.")
    if noise == "an":
        snr = st.slider("Rauschabstand (dB)", C.SNR_MIN, C.SNR_MAX, int(st.session_state["snr_kept"]), help="Leistung einer Linie der Amplitude 1 (also 1/2) gegen die Rauschleistung.")
        st.session_state["snr_kept"] = int(snr)
    else:
        snr = int(st.session_state["snr_kept"])
    smooth = st.select_slider("Glättung der Momentanfrequenz (gleitender Mittelwert)", options=list(C.SMOOTH_CHOICES), key="smooth_select", help="Länge L des gleitenden Mittelwerts; 1: keine Glättung.")
    bandpass = st.radio("Bandpass vor der Auswertung", list(C.BANDPASS_MODES), key="bandpass_select", horizontal=True,
                        help="Rechteck-Bandpass um die bekannten Komponentenfrequenzen (Rand 0,05); entfernt auch einen Mittelwert.")
    st.markdown("**Zufall**")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neu würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"n_select": int(n), "kind_select": kind, "noise_select": noise, "snr_kept": int(snr), "smooth_select": int(smooth), "bandpass_select": bandpass, "seed_input": int(seed),
                   **{param_key(ti, slot): v for slot, v in enumerate(values[:len(C.PARAMS[kind])], start=1)}})

settings = Settings(int(n), kind, values[0], values[1], values[2], noise, int(snr), int(smooth), bandpass, int(seed))
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
m1, m2, m3, m4 = st.columns(4)
if a.rmse_inner is not None:
    m1.metric("Fehler der Frequenz (innen)", de(a.rmse_inner, 4), delta=f"Rand {de(a.rmse_edge, 4)}", delta_color="off", help=f"RMSE gegen die wahre Momentanfrequenz ohne je {C.EDGE} Randwerte; Rand: nur diese Werte.")
else:
    m1.metric("Fehler der Frequenz (innen)", "–", delta="keine wahre Momentanfrequenz", delta_color="off", help="Bei zwei Komponenten gibt es keine einzelne wahre Momentanfrequenz.")
m2.metric("Mittel der Momentanfrequenz", de(a.mean_f, 4), delta=f"Spanne {de(a.f_min, 2)} bis {de(a.f_max, 2)}", delta_color="off")
m3.metric("Anteil negativer Frequenzen", f"{de(100 * a.negative_share, 1)} %", delta="ein Symptom, kein Messwert", delta_color="off", help="Bei einem echten Signal mit positiven Frequenzen darf die Momentanfrequenz nicht negativ sein.")
m4.metric("Hüllkurvenfehler (innen)", de(a.env_rmse_inner, 4) if a.env_rmse_inner is not None else "–", delta="RMSE gegen die wahre Hüllkurve" if a.env_rmse_inner is not None else "keine wahre Hüllkurve", delta_color="off")
st.plotly_chart(build_signal(a), width="stretch", key="signal")
st.plotly_chart(build_envelope(a), width="stretch", key="envelope")
st.caption("Hüllkurve des analytischen Signals (blau) gegen die wahre Hüllkurve (rot gestrichelt). Orange: die ersten und letzten 16 Werte, die für die Fehlermaße ausgespart sind (Randeffekte der FFT).")
st.plotly_chart(build_frequency(a), width="stretch", key="frequency")
st.caption("Momentanfrequenz aus dem Phasenschritt (blau) gegen die wahre (rot gestrichelt). Bei zwei Komponenten gibt es keine wahre Momentanfrequenz: gezeigt werden die beiden Komponentenfrequenzen und das Mittel der Schätzung.")

if st.button("📄 PDF erstellen", key="pdf_make"):
    st.session_state["pdf_bytes"] = build_pdf(a)
    st.session_state["pdf_for"] = settings
if st.session_state.get("pdf_for") == settings and "pdf_bytes" in st.session_state:
    st.download_button("⬇️ PDF herunterladen", st.session_state["pdf_bytes"], file_name="momentanfrequenz.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Wann ist die Ableitung der Phase eine Frequenz?")
kind_v, text = ev.verdict_main(a)
getattr(st, kind_v)(text)

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experimente")
st.caption("Feste Zufallsströme, unabhängig von den Reglern. Abtastrate 1, Frequenzen in Zyklen je Abtastwert. Rauschen mit Varianz 0,5/10^(SNR/10), also gegen eine Linie der Amplitude 1. „Innen“ lässt je 16 Randwerte aus "
           "(bei der Glättung je 32). „Bingenau“ heißt: Frequenzen sind ganze Vielfache von 1/n, dann gibt es keine Randeffekte.")

st.markdown("#### Chirp mit Rauschen und Rand")
chirp = _experiment("chirp")
st.caption(f"Linearer Chirp von {C.EXP_CHIRP_F0:g} bis {C.EXP_CHIRP_F1:g}, n = {C.EXP_CHIRP_N}, {C.EXP_CHIRP_RUNS} Läufe je Zeile; RMSE der Momentanfrequenz und Anteil der inneren Werte mit Fehler über {C.EXP_CHIRP_OVER:g}.")
st.dataframe(pd.DataFrame([{"Rauschabstand (dB)": r["snr"], "innen": de(r["inner"], 4), "Rand": de(r["edge"], 4), f"Fehler über {C.EXP_CHIRP_OVER:g}": f"{de(100 * r['over'], 1)} %"} for r in chirp]), hide_index=True, width="stretch")
st.plotly_chart(build_chirp(chirp), width="stretch", key="exp_chirp")
kind_v, text = ev.verdict_chirp(chirp)
getattr(st, kind_v)(text)

st.markdown("#### Bedrosian: wann die Hüllkurve bricht")
bed = _experiment("bedrosian")
st.caption(f"Amplitudenmodulation (1 + {C.EXP_BED_DEPTH:g} cos 2π fm t) cos 2π fc t mit fc = {C.EXP_BED_FC_BIN}/{C.EXP_BED_N}, bingenau (keine Randeffekte). Wahr: Hüllkurve 1 + {C.EXP_BED_DEPTH:g} cos 2π fm t und Momentanfrequenz fc.")
st.dataframe(pd.DataFrame([{"fm / fc": de(r["ratio"], 2), "unteres Seitenband": de(r["lowest_band"], 3), "Hüllkurve: größter Fehler": de(r["env_max"], 4), "Momentanfrequenz: RMSE": de(r["f_rmse"], 4),
                            "Momentanfrequenz: Spanne": f"{de(r['f_min'], 3)} bis {de(r['f_max'], 3)}"} for r in bed]), hide_index=True, width="stretch")
st.plotly_chart(build_bedrosian(bed), width="stretch", key="exp_bedrosian")
kind_v, text = ev.verdict_bedrosian(bed)
getattr(st, kind_v)(text)

st.markdown("#### Zwei Komponenten")
tones = _experiment("tones")
st.caption(f"cos 2π f1 t + r cos 2π f2 t mit f1 = {C.EXP_TONES_F1_BIN}/{C.EXP_TONES_N} und f2 = {C.EXP_TONES_F2_BIN}/{C.EXP_TONES_N}, bingenau. Leistungsgewichtet: (f1 + r² f2)/(1 + r²).")
st.dataframe(pd.DataFrame([{"r": de(r["r"], 2), "kleinste": de(r["f_min"], 3), "größte": de(r["f_max"], 3), "außerhalb von [f1, f2]": f"{de(100 * r['outside'], 1)} %", "negativ": f"{de(100 * r['negative'], 1)} %",
                            "Mittel": de(r["mean"], 4), "leistungsgewichtet": de(r["weighted"], 4)} for r in tones]), hide_index=True, width="stretch")
st.plotly_chart(build_tones(tones), width="stretch", key="exp_tones")
kind_v, text = ev.verdict_tones(tones)
getattr(st, kind_v)(text)

st.markdown("#### Mittelwert und Randeffekt")
mean = _experiment("mean")
st.caption(f"c + cos 2π f t mit f = {C.EXP_MEAN_F_BIN}/{C.EXP_MEAN_N} (bingenau): mittlere Momentanfrequenz gegen den Mittelwert c (Amplitude 1). Darunter der Randeffekt bei einem nicht bingenauen Kosinus (f = {C.EXP_EDGE_F_BIN:g}/{C.EXP_MEAN_N}).")
st.dataframe(pd.DataFrame([{"c": de(r["c"], 2), "Mittel der Momentanfrequenz": de(r["mean"], 4), "kleinste": de(r["f_min"], 3), "größte": de(r["f_max"], 3)} for r in mean["mean"]]), hide_index=True, width="stretch")
st.plotly_chart(build_mean(mean), width="stretch", key="exp_mean")
st.dataframe(pd.DataFrame([{"Werte": f"{r['lo']} bis {r['hi']}", "RMSE der Momentanfrequenz": de(r["rmse"], 4)} for r in mean["edge"]]), hide_index=True, width="stretch")
kind_v, text = ev.verdict_mean(mean)
getattr(st, kind_v)(text)

st.markdown("#### Glättung und Bandpass")
smooth_rows = _experiment("smooth")
st.caption(f"Chirp {C.EXP_CHIRP_F0:g} bis {C.EXP_CHIRP_F1:g}, n = {C.EXP_SMOOTH_N}, {C.EXP_SMOOTH_RUNS} Läufe; RMSE der Momentanfrequenz innen. Bandpass {C.EXP_SMOOTH_BAND[0]:g} bis {C.EXP_SMOOTH_BAND[1]:g}; gleitender Mittelwert der Länge L.")
st.dataframe(pd.DataFrame([{"Rauschabstand (dB)": r["snr"], "einfach": de(r["plain"], 4), "L = 5": de(r["l5"], 4), "L = 11": de(r["l11"], 4), "L = 21": de(r["l21"], 4), "Bandpass": de(r["bandpass"], 4),
                            "Bandpass + L = 11": de(r["bandpass_l11"], 4)} for r in smooth_rows]), hide_index=True, width="stretch")
st.plotly_chart(build_smooth(smooth_rows), width="stretch", key="exp_smooth")
kind_v, text = ev.verdict_smooth(smooth_rows)
getattr(st, kind_v)(text)

st.markdown("#### Hüllkurvenspektrum")
env = _experiment("envelope")
st.caption(f"Einfache Amplitudenmodulation (Tiefe {C.EXP_ENV_DEPTH:g}, fc = {C.EXP_ENV_FC_BIN}/{C.EXP_ENV_N}, fm = {C.EXP_ENV_FM_BIN}/{C.EXP_ENV_N}) in Rauschen, {C.EXP_ENV_RUNS} Läufe; Anteil, in dem der höchste Gipfel bei fm liegt "
           "(Hüllkurvenspektrum im Band 0,002 bis 0,1) bzw. bei den Seitenbändern fc ± fm (direktes Spektrum). Nur einfache Modulation; Stoßfolgen (Wälzlagerdiagnose) sind nicht gemessen.")
st.dataframe(pd.DataFrame([{"Rauschabstand (dB)": r["snr"], "Hüllkurvenspektrum": f"{de(100 * r['envelope'], 1)} %", "direktes Spektrum": f"{de(100 * r['direct'], 1)} %"} for r in env]), hide_index=True, width="stretch")
st.plotly_chart(build_envelope_spectrum(env), width="stretch", key="exp_envspec")
kind_v, text = ev.verdict_envelope(env)
getattr(st, kind_v)(text)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Eine dominante Komponente** | Bei zwei Komponenten schwankt die Momentanfrequenz um die Frequenz der stärkeren herum und kann negativ werden; sie ist keine Frequenz mehr. | Spektrogramm, Wavelets |
| **Schmalbandig (Bedrosian)** | Erreicht das untere Seitenband der Modulation den Nullpunkt, brechen Hüllkurve und Frequenz ein. | – |
| **Kein Mittelwert** | Ein Mittelwert von der Größe der Amplitude verschiebt den Zeiger; die Phase windet nicht mehr um den Ursprung. | Mittelwert abziehen, Bandpass |
| **Kein Rauschen** | Das Ableiten der Phase verstärkt das Rauschen; ohne Glättung ist die Momentanfrequenz bei 0 dB kaum zu gebrauchen. | Glättung, Bandpass |
| **Periodische Fortsetzung** | Die FFT-Hilbert-Transformation nimmt sie an: An den Rändern entstehen Fehler (bei Chirp und nicht bingenauen Frequenzen). | Randwerte verwerfen |
| **Stationäre Modulation** | Das Hüllkurvenspektrum zeigt die Modulationsfrequenz, ist bei einfacher Modulation aber nicht empfindlicher als das direkte Spektrum. | – |
""",
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Analytisches Signal.** Für ein reelles Signal mit DFT $X_k$ ist $Z_k = X_k\,h_k$ mit $h_0 = 1$, $h_k = 2$ für $0 < k < n/2$, $h_{n/2} = 1$ (gerade $n$) und $h_k = 0$ für $k > n/2$; $z = \mathrm{IDFT}(Z)$,
$\operatorname{Re} z = x$, $\operatorname{Im} z = \mathcal H\{x\}$.

**Hüllkurve und Phase.** $a_t = |z_t|$, $\varphi_t = \arg z_t$. Für $x = a(t)\cos\varphi(t)$ mit schmalbandigem $a$ ist $z \approx a\,e^{i\varphi}$ (Bedrosian: $\mathcal H\{a\cos\varphi\} = a\,\mathcal H\{\cos\varphi\}$, wenn die Spektren von $a$ und $\cos\varphi$ getrennt liegen).

**Momentanfrequenz.** $f_{t+1/2} = \arg(z_{t+1}\bar z_t)/(2\pi)$; für einen linearen Chirp ist dieser Schritt die exakte Ableitung der quadratischen Phase.

**Zwei Komponenten.** Für $e^{i2\pi f_1 t} + r\,e^{i(2\pi f_2 t + \phi)}$ mit der Phasendifferenz $\delta(t)$ ist die Momentanfrequenz
$\dfrac{f_1 + r^2 f_2 + r(f_1+f_2)\cos\delta}{1 + r^2 + 2r\cos\delta}$; sie wird für $r \to 1$ unbeschränkt (Nullstelle der Hüllkurve). Das zeitliche Mittel ist die Frequenz der stärkeren Komponente (Windungszahl des Zeigers um den Ursprung).

**Mittelwert.** Für $c + e^{i2\pi f t}$ mit $c < 1$ windet der Zeiger um den Ursprung (mittlere Frequenz $f$), mit $c > 1$ nicht (mittlere Frequenz 0).

**Rauschen.** Die Phase wird mit der Rauschamplitude gestört; die Ableitung vergrößert die Streuung etwa mit $10^{-\mathrm{SNR}/20}$.

**Quellen.** Lehrbuchwissen und Aufsätze (Gabor 1946; Bedrosian 1963; Boashash 1992). Formeln hier per Simulation, Handrechnung und gegen scipy geprüft; die Quellen selbst wurden nicht gelesen.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
