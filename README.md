# Momentanfrequenz und analytisches Signal: Die Phasenableitung ist nur für ein Signal mit einer dominanten Komponente eine Frequenz – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-momentanfrequenz-demo.streamlit.app/)** -->

Neunte Demo der Reihe **Spektral- und Wavelet-Analyse** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, aufbauend auf der Demo zum **Periodogramm**. Fünf
Testsignale mit exakt bekannter Wahrheit (Chirp, Amplitudenmodulation, Frequenzmodulation, zwei Kosinus, Kosinus mit Mittelwert) werden über das **analytische Signal** (Hilbert-Transformation) in Hüllkurve und
Momentanfrequenz zerlegt; Rauschen, Randeffekte, die Bedrosian-Bedingung, die Grenze bei zwei Komponenten, der Mittelwert, Glättung und das Hüllkurvenspektrum werden gegen die Wahrheit gemessen. Die Demo liefert das Orakel
„wahre Momentanfrequenz“ für das Spektrogramm und die Wavelets danach.

**Namensfalle:** Die „Envelope“ der Anomalieerkennung (`elliptic-envelope-demo`, robuste Kovarianz) hat mit der Hüllkurve eines Signals nichts zu tun.

## Warum dieses Problem

Das Periodogramm sagt, welche Frequenzen in der ganzen Aufzeichnung stecken. Ändert sich die Frequenz mit der Zeit, fragt man nach der Momentanfrequenz. Die Phasenableitung des analytischen Signals ist die übliche
Definition, aber sie ist nur für (nahezu) einkomponentige, schmalbandige Signale eine Frequenz. Die Demo zeigt, wo sie trägt, wo sie wegen Rauschen und Rändern ungenau wird, und wo sie versagt.

## Modell

Testsignale auf den Zeitpunkten t = 0 … n − 1 (Abtastrate 1, Frequenzen in Zyklen je Abtastwert, Nyquist-Frequenz 0,5): linearer Chirp, Amplitudenmodulation (1 + m cos 2π fm t) cos 2π fc t, Frequenzmodulation
cos(2π fc t + β sin 2π fm t), Summe zweier Kosinus, Konstante plus Kosinus. Das analytische Signal entsteht per FFT (positive Frequenzen doppelt, negative null), die Momentanfrequenz aus dem Phasenschritt
arg(z[t+1] z[t]*)/2π an den halben Zeitpunkten; dort ist sie für Chirp und Frequenzmodulation exakt die wahre Frequenz. Glättung (gleitender Mittelwert) und Rechteck-Bandpass sind wählbar. Alles selbst in numpy;
scipy dient nur als Orakel in den Tests.

## Methodik

n = 128 bis 2048. Das Rauschen hat die Varianz 0,5/10^(SNR/10), also gegen eine Linie der Amplitude 1. „Innen“ lässt je 16 Randwerte aus (bei der Glättung je 32), „bingenau“ heißt: Frequenzen sind ganze Vielfache von
1/n, dann gibt es keine Randeffekte. Experimente über feste Zufallsströme (Seeds ab 910000), die Rauschversuche mit 300 bis 400 Läufen (Standardfehler eines Anteils höchstens 2,5 Prozentpunkte), die übrigen ohne Zufall.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py` und `tests/test_experiments_golden.py`.

| Frage | Befund |
|---|---|
| Wie genau ist die Momentanfrequenz eines linearen Chirps von 0,1 bis 0,3 (n = 512) mit Rauschen? | RMSE im Inneren bei 40 / 30 / 20 / 10 / 5 / 0 dB: 0,0015 / 0,0034 / 0,0105 / 0,0346 / 0,0789 / 0,1572, also etwa 10^(−SNR/20); an den Rändern (je 16 Werte) 0,049 / 0,050 / 0,050 / 0,064 / 0,112 / 0,183. Bei 10 dB liegen 13,6 % der inneren Werte um mehr als 0,05 daneben (60,7 % bei 0 dB). |
| Wann bricht die Hüllkurve einer Amplitudenmodulation (Bedrosian, bingenau, Tiefe 0,5)? | Für fm/fc = 0,04 bis 0,78 sind Hüllkurve und Momentanfrequenz exakt (Fehler 0,0000); bei fm = fc: größter Hüllkurvenfehler 0,161, Momentanfrequenz 0,157 bis 0,234 (RMSE 0,0253); bei fm/fc = 1,27: Hüllkurvenfehler 0,495, Frequenz 0,114 bis 0,293. Die Schwelle ist scharf: Das untere Seitenband erreicht 0. |
| Was ist die Momentanfrequenz bei zwei Komponenten (f1 = 0,2 mit Amplitude 1, f2 = 0,25 mit Amplitude r)? | Spanne bei r = 0,3 / 0,5 / 0,8: 0,178 bis 0,211, 0,150 bis 0,216, 0,026 bis 0,222; **bei r = 1: −0,275 bis 0,225, 5,1 % der Zeit negativ.** Das zeitliche Mittel ist immer 0,1992 = f1, die Frequenz der stärkeren Komponente, nicht die leistungsgewichtete (0,2246 bei r = 1). |
| Was macht ein Mittelwert c (Amplitude 1)? | Mittlere Momentanfrequenz bei c = 0,5: 0,1992 (Spanne 0,134 bis 0,345); bei c = 0,9: 0,0586; ab c = 1,1: 0,0000, weil der Zeiger den Ursprung nicht mehr umkreist. |
| Wie groß ist der Randeffekt (Kosinus nicht bingenau, n = 512)? | RMSE 0,0104 bei den ersten 16 Werten, 0,0004 im Inneren, 0,0507 bei den letzten 15 Werten. |
| Was hilft gegen Rauschen (Chirp, n = 512, innen)? | Bei 10 dB: einfach 0,0343; gleitender Mittelwert über 5 / 11 / 21 Werte 0,0103 / 0,0047 / 0,0025; Bandpass 0,0173; Bandpass und Mittelwert über 11 Werte zusammen 0,0036. Bei 0 dB: 0,1572 gegen 0,0388 (L = 21). |
| Zeigt das Hüllkurvenspektrum die Modulationsfrequenz besser als das direkte Spektrum (einfache Modulation, Tiefe 0,3, n = 2048)? | Nein: bei −6 dB findet das Hüllkurvenspektrum sie in 40,5 % der Läufe, das direkte Spektrum (Seitenbänder fc ± fm) in 69,2 %; bei −3 dB 93,8 % gegen 98,2 %; ab 0 dB beide 100 %. |

## Ehrliche Grenzen

- **Heimvorteil:** Alle Testsignale sind exakt bekannte Modelle; echte Signale haben unbekannte, oft mehrere Komponenten.
- **FFT-Hilbert-Transformation:** Randeffekt und Periodizitätsannahme; ein FIR-Hilbert-Filter hat andere Ränder (nicht gebaut).
- **Das Hüllkurvenspektrum** ist nur für einfache Amplitudenmodulation gemessen; sein Nutzen bei Stoßfolgen (Wälzlagerdiagnose) ist nicht gemessen und wird nicht behauptet.
- **Glättung** ist nur bei linearem Frequenzverlauf unverzerrt.
- **Nicht gebaut:** Empirical Mode Decomposition (Hilbert-Huang), Teager-Kaiser-Operator, Wigner-Ville-Verteilung (Letztere in der Demo zum Reassignment).
- **Quellen** (Gabor 1946, Bedrosian 1963, Boashash 1992): nicht gelesen; Formeln per Simulation, Handrechnung und gegen scipy geprüft.

## Verifikation

- **Kern gegen Handrechnung:** n = 4, x = (1, 0, −1, 0): z = (1, i, −1, −i), Phasenschritt π/2, Momentanfrequenz 1/4 an allen Stellen, Hüllkurve 1.
- **Orakel (`tests/test_oracle_analytic.py`):** analytisches Signal gegen `scipy.signal.hilbert` (gerade und ungerade n, mehrere Zeilen), Realteil gleich dem Signal, bingenauer Kosinus exakt, Phasenschritt gegen die exakte Ableitung für
  Chirp und Frequenzmodulation, Zwei-Komponenten-Formel gegen den Summenzeiger, Hüllkurvenspektrum der reinen Modulation gegen die Handrechnung n·Tiefe²/4.
- **Generator** (`tests/test_generator.py`): wortgleich aus der ersten Demo.
- **Zahlen:** jede Zahl dieser README in `tests/test_claims.py`; Festwerte der Experimente in `tests/test_experiments_golden.py`; Verdikte in `tests/test_verdicts.py`.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler an den Grenzen), Auswahlfelder, Permalink, bedingte Regler, Footer, Laufzeit-Importe.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `signal_generator.py` | Gemeinsamer Generator der Reihe (wortgleich; hier nur der Zufallsstrom SplitMix64) |
| `mfq_analytic.py` | analytisches Signal, Hüllkurve, Phase, Momentanfrequenz, Glättung, Bandpass |
| `mfq_signals.py` | die fünf Testsignale mit exakter Wahrheit |
| `mfq_envelope.py` | Hüllkurvenspektrum und direktes Spektrum |
| `mfq_constants.py` | Regler-Grenzen, Signaltypen, feste Annahmen, Experimente, Presets |
| `mfq_evaluation.py` | Analyse, Experimente, Verdikte |
| `mfq_presets.py` | Permalink (Begrenzen), Presets, Zufalls-Seed-Button |
| `mfq_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `mfq_pdf_export.py` | PDF der aktuellen Einstellung (erst nach Klick) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy (nur Gegenproben) |
| `tests/` | Kern, Orakel, Signale, Zahlen, Verdikte, Presets, Randfälle, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Zeit-Frequenz-Auflösung bei mehreren Komponenten:** Thema der Demos zum Spektrogramm und zu den Wavelets.
- **FIR-Hilbert-Filter, Hilbert-Huang, Teager-Kaiser, Wigner-Ville:** nicht enthalten.
- **Gemessene Daten:** alles ist simuliert, damit die Wahrheit bekannt ist.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Python 3.12 oder neuer. Die Experimente brauchen beim ersten Aufruf wenige Sekunden (danach zwischengespeichert).

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, pandas und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
