# HVAC Equation Solver

Equation solver for teaching and rapid calculation of thermodynamic state changes in HVAC systems. Integrates CoolProp for thermodynamic properties and humid air calculations. Also used for developing benchmark tests for AI agents in building services engineering.

## Installation

### Erforderliche Bibliotheken

```bash
pip install numpy scipy CoolProp matplotlib pint customtkinter
```

| Bibliothek | Version | Zweck |
|------------|---------|-------|
| numpy | >= 1.20 | Array-Operationen, mathematische Funktionen |
| scipy | >= 1.7 | Numerische Solver (fsolve, brentq) |
| CoolProp | >= 6.4 | Thermodynamische Stoffdaten |
| matplotlib | >= 3.5 | Diagramme und Plots (optional) |
| pint | >= 0.20 | Einheitenhandling und Dimensionsanalyse |
| customtkinter | >= 5.0 | Moderne GUI |
| tkinter | - | GUI (in Python Standard-Library enthalten) |

### Programm starten

```bash
python3 main.py
```

### Installation für Studierende (offline nutzbar)

Anleitung: `INSTALLATION.md`. Die Skripte `installieren_windows.bat`,
`installieren_mac.command` (→ `install.sh`) und `install.sh` legen im Programmordner
eine eigene Umgebung `.venv` an und installieren `requirements.txt`; gestartet wird
mit `starten_windows.bat` / `starten_mac.command` / `start.sh`. Internet nur für
Installation und Update (neues Release bzw. `git pull`, dann Installation erneut).
Programmversion: `version.py` (Titel- und Statusleiste). `.bat`-Dateien müssen CRLF und
ASCII bleiben (siehe `.gitattributes`).

## Architektur

```
equation_solver/
├── main.py              # Tkinter GUI (Hauptanwendung)
├── parser.py            # Equation syntax → Python conversion
├── solver.py            # Block-Dekomposition + Bracket-Suche Solver
├── optimizer.py         # MINIMIZE/MAXIMIZE ... VARY ... (Raster + Brent/Powell, je Punkt)
├── thermodynamics.py    # CoolProp Wrapper mit Einheitenumrechnung
├── humid_air.py         # CoolProp HumidAirProp Wrapper für feuchte Luft
├── radiation.py         # Schwarzkörper-Strahlungsfunktionen (Planck, vektorisiert)
├── units.py             # Einheitenhandling und Konvertierung (v3.0)
├── unit_constraints.py  # Einheiten-Propagation und Konsistenzprüfung (v3.0)
├── diagnostics.py       # Generische Fehleranalyse (Struktur, Numerik, Namens-Hinweise)
├── version.py           # Programmversion (Titel- und Statusleiste)
├── requirements.txt     # Benötigte Pakete (mit getesteten Versionsbereichen)
├── install.sh, start.sh, installieren_*/starten_*  # Installation/Start je OS
├── INSTALLATION.md      # Installations- und Update-Anleitung (Studierende)
├── test_regressions.py  # Regressionstests Parser/Solver/Einheiten
├── test_unit_constraints.py  # Tests Einheiten-Propagation/Dimensionsprüfung
├── test_berechnungen.py # Berechnungsaufgaben Thermodynamik/Wärmeübertragung
├── test_optimierung.py  # Optimierung gegen analytische/scipy-Referenzen
├── test_gui.py          # GUI-Tests (headless)
```

### Tests

```bash
python3 test_regressions.py       # Parser, Solver, Einheiten (~15 s)
python3 test_unit_constraints.py  # Einheiten-Propagation, Dimensionsprüfung
python3 test_berechnungen.py      # 63 Aufgaben mit/ohne Einheiten gegen Referenzwerte
python3 test_optimierung.py       # Optimierung (MINIMIZE/MAXIMIZE), unabhängige Referenzen
python3 test_gui.py               # GUI headless (holt kurz den Fokus - nicht tippen)
```

`test_regressions.py` deckt Parser (Direktzuweisungs-Erkennung, Vektoren, Einheiten-Sweeps,
Schlüsselwörter, Kommentare), Solver (Wurzelwahl, Widerspruchserkennung, Parameterstudien,
Tearing, Determinismus) und Einheiten-System (SI-Umrechnung, Startwerte, delta_K,
Offset-Konvertierungen, Strahlung) ab. `test_berechnungen.py` rechnet Aufgaben aus
Thermodynamik und Wärmeübertragung jeweils MIT und OHNE Einheiten und vergleicht mit
unabhängig berechneten Referenzwerten (CoolProp direkt, analytisch).
Zusätzlich hat jedes Modul einen Selbsttest: `python3 <modul>.py`.

### Entwicklung: Arbeitsweise und Prüfablauf

- **Alle Regeln generisch**: aus der Struktur des Gleichungssystems, nie für bestimmte
  Gleichungsformen und nie abhängig von Variablennamen (bei Gleichstand: Reihenfolge im Blatt)
- **Intern immer SI, Temperaturen immer Kelvin**: keine Regel, die Zahlen oder Gleichungen nach
  dem Zusammenhang anders deutet (keine andere Temperaturskala, keine Einheit nach der Größe einer
  Zahl) - vermutete Fehleingaben nur als Hinweis "ⓘ", die Rechnung bleibt unverändert
- Neue Funktion: Tests in den Suiten, Hilfe (`FUNCTION_HELP_TEXT`, Zeilen ≤ 70 Zeichen - getestet),
  README.md, CLAUDE.md; das Beispielblatt (`main._insert_example`, Werte in `test_gui.py` geprüft)
  soll die Funktionen zeigen
- **Lokal, nicht im Repo** (`Testbeispiele/` in `.gitignore`, nie committen): 123 Lehrbeispiele mit
  Lösungsblättern und Testbericht, `Testbeispiele/PROJEKTSTAND.md` (Stand, Festlegungen, offene
  Punkte) und `Testbeispiele/Pruefskripte/alle_pruefungen.sh` - vor jedem Push: alle Suiten und
  alle Lehrbeispiele mit Einheiten, ohne Einheiten (alles SI), mit neutral umbenannten Variablen
  sowie Vergleich der Anzeige-Einheiten (ca. 40 min)
- Git: Arbeitsbranch (derzeit `fix/review-2026-10`), `main` per Fast-Forward; Commit, Push und
  Aktualisieren von `main` nur auf ausdrücklichen Wunsch

## Kernfunktionen

### Parser (parser.py)
- Converts equation syntax to Python: `^` → `**`, `ln` → `log`
- Comments: `"..."` and `{...}` (auch verschachtelt und mehrzeilig)
- Dezimalkomma wird als Fehler gemeldet (Punkt verwenden); `·` als Malzeichen; `lg` = `log10`
- **Fallunterscheidung** `IF(a, b, x, y, z)` wie EES: x für a < b, y für a = b, z für a > b
  (`parser.if_function`, elementweise für Arrays; NaN/komplexes a oder b -> NaN). Jede
  Schreibweise `if(`/`If(` wird in `mangle_keywords` zu `IF(` (sonst Python-Schlüsselwort);
  `if` ohne Klammer bleibt Variablenname. Alle Argumente werden ausgewertet (auch der nicht
  gewählte Zweig - Division durch null dort ist ein Auswertungsfehler). Einheiten: a, b gleiche
  Dimension, Ergebnis = Dimension der Zweige (`unit_constraints._same_dimension_result`)
- **Startwerte im Blatt**: Kommentarblock `{$Startwerte ... $}` (`parser.parse_start_values`,
  `start_values_edit`). Der Text ist die einzige Quelle: Solve liest den Block, der Dialog
  Solve > Initial Values schreibt ihn (ein Undo-Schritt; mit Einheit eingegebene Werte bleiben
  wie geschrieben, reine Zahlen sind SI). Fehler im Block mit Zeilennummer
- Einheiten-Kehrwerte nach Leerzeichen: `n = 0.3 1/h`, `1/K`, `h^-1`
- `%` und `‰` sind Einheiten (pint): `eta = 89.2 %` -> 0.892, auch in Sweeps, Wertelisten,
  Funktionsargumenten (`rh=50 %`) und zusammengesetzt (`%/K`); `a % b` zwischen Variablen
  bleibt der Modulo-Operator
- Konstanten OHNE Einheit gelten in Blättern mit Einheiten als dimensionslos - außer dem
  Wert 0: null ist in jeder Einheit null (wie die Zahl 0 in einer Summe), ihre Einheit folgt
  aus den Gleichungen (`Q_12 = 0` neben `Q_12 + W_12 = m*c_v*(T_2 - T_1)` -> kJ, keine
  Warnung; `main._dimensionless_constants`)
- Signaturprüfung beim Einlesen: Fluidname, Eigenschafts-/Parameternamen und
  Argumentanzahl aller Funktionen (enthalpy, HumidAir, Eb, sqrt, max, ...)
- **Namen nur aus a-z, A-Z, 0-9 und _** (`parser._check_ascii_names`): griechische Buchstaben und
  Umlaute (Φ, η, Q_wärme) werden beim Einlesen mit Zeile gemeldet (sonst fielen sie still aus den
  Gleichungen); Einheiten wie µm sind erlaubt
- `ceil`, `floor`, `round` (z.B. Anzahl Geräte); Einheit wie das Argument
- **Fortsetzungszeilen**: eine Gleichung geht in einer offenen Klammer weiter, wenn die Zeile mit
  `(`, `,` oder einem Operator endet bzw. die nächste damit beginnt (`parser._join_bracket_lines`,
  Original für Meldungen ebenso zusammengefügt); eine vergessene `)` ohne Fortsetzung bleibt ein
  Fehler ihrer Zeile
- **Bezugszustand** `REFERENCE R717 IIR` (eigene Zeile, wie EES; `parser.parse_reference_states`,
  `thermodynamics.set_reference_states` je Lösungslauf): IIR (h = 200 kJ/kg, s = 1 kJ/(kg K) für
  siedende Flüssigkeit bei 0 °C), ASHRAE (0 bei -40 °C), NBP (0 beim Normalsiedepunkt), DEFAULT
  (CoolProp). Verschiebt h, u, s um Konstanten (Ein- und Ausgaben), gilt für alle Namen des Fluids
  (`canonical_fluid`: R717 = ammonia); ohne Zeile CoolProp-Standard (für die meisten Kältemittel
  schon IIR, nicht Ammoniak/Wasser). Fehler mit Zeile (Fluid unbekannt, Bezugspunkt nicht im
  Nassdampfgebiet, Fluid doppelt)
- **Python-Schlüsselwörter als Variablennamen** (`lambda` für λ, `in`, `is`, ...) sind
  erlaubt: intern umbenannt (`lambda` → `_kw_lambda`), angezeigt wieder als `lambda`
  (`parser.display_name()` / `parser.unmangle()`)
- Thermodynamic function calls: `enthalpy(water, T=100, p=1)` → `enthalpy('water', T=100, p=1)`
- **Syntaxprüfung beim Einlesen** (generisch über den Python-Syntaxbaum): nicht parsebare
  Zeilen und Aufrufe von etwas, das keine Funktion ist (Variable/Zahl/Klammer vor `(`),
  werden mit Zeilennummer und Fundstelle `▶` gemeldet, z.B.
  `Zeile 14: 'r_1' ist keine Funktion - ... bei: 1 /( r_1  ▶( (1/r_1/h_i) + ln(…`
- Extracts variables from equations (filters function names and parameter keys)
- Vector syntax: `T = 0:10:100` (start:step:end) or `T = 0:100` (start:end, step=1)
  - MATLAB-Semantik: Die Schrittweite wird nie verfälscht; der Endwert ist nur
    enthalten, wenn er exakt auf dem Raster liegt (`0:0.3:1` → 0, 0.3, 0.6, 0.9)
  - Sweeps mit Offset-Einheiten (`T = 20:10:50 °C`) werden elementweise korrekt
    konvertiert (Offset, kein Faktor); °C ist immer absolut, Differenzen in K
- **Wertelisten** (Messdaten): `T_a = [-5.2 -4.8 -3.9] °C`, Trennzeichen Leerzeichen,
  Tabulator, Zeilenumbruch, `;` oder `,`; Dezimalpunkt (Dezimalkomma wird gemeldet).
  Eine Liste darf über mehrere Zeilen gehen (Spalte aus Excel/CSV/TXT zwischen `[`
  und `]` einfügen); Zeilennummern in Fehlermeldungen bleiben erhalten. Mehrere
  Listen/Sweeps werden punktweise kombiniert (gleiche Länge). `1 [kg/s]`
  (EES-Einheitenschreibweise) wird mit Hinweis abgelehnt
- **Direct assignments** like `T_1 = 450`, `m = 10000/3600` or `m_dot = 10000/3600 kg/s`
  (numerischer Ausdruck + Einheit) are treated as constants
- **Wichtig:** Eine Zeile ist nur dann eine Direktzuweisung, wenn links ein REINER
  Variablenname steht. `x + 5 = 2`, `sin(alpha) = 0.5` oder `x^2 = 9` sind
  Gleichungen und werden iterativ gelöst

### Solver (solver.py)

#### Lösungsstrategie (Block-Dekomposition)
1. **Konstanten zuweisen**: Explizite Definitionen wie `T_1 = 450`
2. **Direkte Auswertung**: Gleichungen der Form `var = ausdruck` werden sequentiell berechnet
3. **Einzelne Unbekannte**: Gleichungen mit nur einer Unbekannten werden mit Bracket-Suche + Brent's Methode gelöst
4. **Blockweise Lösung**: Zusammenhängende Gleichungsblöcke (Reihenfolge = Eingabereihenfolge,
   unabhängig vom Hash-Seed) werden gelöst:
   - **Tearing** zuerst: Lässt sich der Block mit EINER geschätzten Variable der Reihe nach
     direkt auswerten (z.B. Filmtemperatur-Iteration: T_s → T_f → Stoffwerte → Ra → Nu →
     Wärmestrom), wird nur über diese Variable mit Bracket-Suche iteriert - robust auch
     bei schlechten Startwerten. Zuerst nur explizite Zuweisungen; danach Ketten, in denen
     eine Gleichung mit genau einer offenen Größe, die strukturell nur LINEAR vorkommt
     (`eta = (h_1 - h_2)/(h_1 - h_2s)` nach h_2, Massenbilanz als Summe), exakt aufgelöst
     wird (`_LinearStep`: zwei Auswertungen, `_is_linear_in` am Syntaxbaum) - die natürliche
     implizite Schreibweise ist damit so schnell wie die explizite (Mehrfach-Tearing nur explizit)
   - **Mehrfach-Tearing**: sonst wenige Tearing-Variablen (z.B. Radiositäten + Oberflächen-
     temperaturen), alle übrigen der Reihe nach direkt; `least_squares` über k Variablen,
     Residuen fest gewichtet mit der Termgröße am Startpunkt (mitlaufende Normierung wäre
     nicht glatt, wenn Terme gegen null gehen, z.B. adiabate Wand)
   - sonst simultan mit `least_squares` (Levenberg-Marquardt) / `fsolve`, Zeitbudget ~20 s
   - gescheiterte Blöcke werden nicht erneut versucht; unabhängig gelöste Teilblöcke bleiben
     in der (Teil-)Lösung
   - **Zerlegung vor dem Lösen** (auch Blöcke ab 3 Größen): minimaler gekoppelter Kern zuerst,
     nachgelagerte Gleichungen danach. Scheitert ein Kern, werden seine Größen gesperrt und der
     nächste Kern unter den übrigen Gleichungen gesucht (`_solve_block_iteratively`) - eine nicht
     auswertbare Folgegleichung (`z = ln(-y)`, übersättigte Mischung) reißt den lösbaren Kern nicht
     mit; die Meldung nennt die Ursache
   - Zeitbudget eines simultanen Versuchs (~20 s) gilt auch INNERHALB eines least_squares-Laufs
     (Residuenfunktion bricht ab) - ein unlösbarer Punkt einer Parameterstudie verliert so nicht
     alle Werte durch das Gesamtlimit
   - Startvarianten: zusätzlich kleine, je Größe gestaffelte relative Verschiebungen (±2-6 %) -
     ein Start genau auf einem gegebenen Wert (0/0 in `(T_2 - T_3)/(T_2 - T_6)`) wird verlassen
   - **Überbestimmter Block** (mehr Gleichungen als Unbekannte, z.B. eine redundante Bilanz):
     nur der überbestimmte Kern nach Dulmage-Mendelsohn (`_overdetermined_part`: von überzähligen
     Gleichungen über alternierende Pfade erreichbar; Folgegleichungen danach) - darin ein
     strukturell lösbares quadratisches Teilsystem lösen (`_square_subsets`: perfekte
     Zuordnung, Blatt-Reihenfolge; bei singulärer Auswahl Varianten), die übrigen Gleichungen
     bleiben als Constraints offen und werden geprüft -> Lösung oder "Widersprüchliches System"
     mit der verletzten Gleichung. Unabhängig von der Schreibweise (`a = x + y` / `x + y = a`);
     `parser.validate_system` lehnt mehr Gleichungen als Unbekannte nicht mehr ab; ein Blatt nur
     aus Vorgaben und Kontrollgleichungen (keine Unbekannte) wird ebenso geprüft
   - **Eindeutigkeit** (`_non_unique_solution`): nach jedem gelösten Block mit >= 2 Größen
     Jacobi-Matrix am Lösungspunkt (Zeilen/Spalten normiert); ist sie singulär UND lässt sich
     die Lösung entlang der Nullraum-Richtung fortsetzen (Größe verschoben, Rest neu gelöst),
     ist die Lösung nicht eindeutig -> Meldung "Lösung nicht eindeutig: Gleichungen ... sind
     voneinander abhängig" statt eines beliebigen Punkts (`x + y = 1`, `2*x + 2*y = 2`;
     vertauschte Bilanz). Doppelwurzeln (isoliert) bestehen den Fortsetzungstest nicht
5. **Iteration**: Schritte 2-4 werden wiederholt bis alle Gleichungen gelöst sind

#### Robuste Wurzelfindung für einzelne Gleichungen
- **Bracket-Suche**: ~4000 Testpunkte (auch negative) über Größenordnungen von 1e-12 bis ±5e9
- **Adaptive Verfeinerung**: Bei großen relativen Funktionsänderungen wird das Intervall verfeinert
- **Brent's Methode**: Robuste Wurzelfindung bei Vorzeichenwechsel (Polstellen werden
  über einen Plausibilitätscheck verworfen, ebenso Underflow-Plateaus abklingender Funktionen)
- **Wurzelauswahl**: Bei mehreren Wurzeln wird die dem Startwert nächstgelegene gewählt
  (Tie-Break: positive Wurzel); `sin(alpha) = 0.5` liefert 30, nicht 150 oder −210
- **Standard-Startwert**: 1.0 für alle Variablen bzw. einheitenbasiert
  (`units.initial_values_from_units`): typischer Wert je Dimension; unbekannte ABSOLUTE
  Temperaturen starten beim Mittel der vorgegebenen Temperaturen (statt pauschal 350 K,
  sonst falsches Vorzeichen von Differenzen wie T_Raum - T_Scheibe -> Ra^(1/6) nicht reell).
  Mehrfach-Tearing startet gleichartige Größen mit gleichem Startwert zusätzlich gestaffelt
  (in der Reihenfolge im Blatt, nicht nach Namen).
  Ohne Einheit (auch in Blättern ganz ohne Einheiten): Startwert aus der STRUKTUR - Variablen,
  die als eigene Summanden in derselben Summe/Differenz stehen, bilden ein Netz (gleiche
  Dimension); Unbekannte eines gekoppelten Blocks starten beim Mittel ihrer Nachbarn im Netz,
  bekannte Werte fest (Interpolation: `T_R - T_G1`, `T_G1 - T_G2`, `T_G2 - T_G3`, `T_G3 - T_a`
  -> 285.65 / 278.15 / 270.65 K; nie genau auf einem Randwert, Differenz null wäre singulär).
  Bekannte zusammengesetzte Summanden zählen als feste Nachbarn (`sigma*T_D^4 - J_2`),
  Zahlenliterale nicht (`solver._structural_start_hints`). Nur Iterationsstart, nicht Anker
  der Wurzelauswahl.
- **Keine Abhängigkeit von Formelzeichen**: Startwerte, Wurzelauswahl, Tearing-Auswahl und
  Blockbildung hängen nicht von Variablennamen ab (bei Gleichstand entscheidet die Reihenfolge
  im Blatt, `solver._appearance_order`; keine Startwerte aus Namensähnlichkeit). Geprüft:
  alle Lehrbeispiele mit neutral umbenannten Variablen liefern dieselben Ergebnisse.
  Auch Temperaturdifferenzen: keine Namenskonvention (Differenzen in K, Charakter aus der Struktur)
- **Residuen-Bewertung**: relativ zur Größenordnung der Gleichungsterme - Divergenz
  zur Asymptote (z.B. `1/(x-2) = 0`) wird NICHT als Lösung akzeptiert
- **Meldung bei erfolgloser 1-D-Suche**: die Fehlermeldung des ungültigen Testpunkts, der dem besten
  gültigen am nächsten liegt (Rand des Definitionsbereichs), wird als Auswertungsfehler gemeldet -
  implizit `h = HumidAir(h, T=T_P, w=…)` im Nebelgebiet -> "Zustand übersättigt" (`_LAST_SEARCH_ERROR`)
- Statuszeile: "Lösung gefunden (14 direkt, 4 iterativ, 2 Blöcke (3+5 Größen))" - intern zerlegte
  Komponenten zählen mit ihren echten gekoppelten Kernen
- **Zeitbudget**: max. ~10 s pro Einzelgleichung und `solver.SOLVE_TIME_LIMIT` = 60 s pro
  Lösungslauf (bzw. pro Punkt einer Parameterstudie). Bei Überschreitung: Teillösung +
  Meldung "Abbruch nach Zeitlimit" mit den offenen Unbekannten (`SolveTimeout` ist
  BaseException, damit die `except Exception`-Fallbacks das Limit nicht verschlucken)
- **Komplexe Zwischenwerte** (z.B. `Ra^(1/6)` mit negativem `Ra`) gelten als ungültige
  Auswertung, nicht als Absturz
- **Konsistenzprüfung**: Constraint-Gleichungen (0 Unbekannte) mit großem Residuum
  führen zu "Widersprüchliches System" statt stillschweigendem Erfolg
- **Parameterstudien**: Warm-Start - die Lösung des Vorpunkts ist Startwert des nächsten
  Punkts (verhindert Sprünge zwischen Lösungsästen)
- **Auswertungsfehler** (0/0, CoolProp-/HumidAir-Fehlermeldungen) werden in der Meldung
  mit Originalzeile genannt statt "keine Lösung" - auch in der GUI vor der generischen
  Diagnose (`analysis.evaluation_errors`)
- Parameterstudien: gescheiterte Punkte werden gemeldet (Nummer + Grund), gelöste Größen
  dieser Punkte bleiben erhalten; sweep-unabhängige Größen erscheinen als Einzelwert.
  Scheitern zwei Punkte hintereinander am Zeitlimit, wird die Studie abgebrochen
  (ohne Warm-Start scheitern sonst meist alle weiteren: n Punkte x 60 s)

#### Parameterstudien
- Sweep-Variablen werden als Konstanten für jeden Punkt behandelt
- Für jeden Sweep-Punkt wird `solve_system` mit Block-Dekomposition aufgerufen
- Vektorisierte Auswertung für direkte Funktionen ohne Iteration

### Optimierung (optimizer.py)

Anweisung im Blatt (eigene Zeile, nach ',' Fortsetzung in der nächsten Zeile):
`MINIMIZE ziel VARY x = a .. b Einheit[, y = c .. d Einheit]` bzw. `MAXIMIZE ...`
(`parser.parse_optimization` -> `OptimizationGoal`, Grenzen in SI; die Zeilen sind für
`parse_equations` Leerzeilen). Einheit am Ende gilt für beide Grenzen (wie bei Sweeps).
- Die variierten Größen sind keine Unbekannten (das Blatt hat je Größe eine Gleichung
  weniger); Struktur-/Einheitenprüfung mit der Bereichsmitte als Platzhalter
  (`main._check_optimization`: kein fester Wert, keine Sweep-Variable, muss vorkommen)
- Generisch: für jeden Kandidaten löst `solve_system` das übrige System (Warm-Start);
  Zielfunktion = Wert der Zielgröße. Raster über den ganzen Bereich (21 Punkte bzw. ~100
  bei mehreren Größen, Schlangenlinie), dann lokal: Brent beschränkt (1 Größe) bzw. Powell
  beschränkt (mehrere) in normierten Koordinaten [0, 1]; log. Raster bei > 2 Dekaden.
  Kandidaten ohne Lösung gelten als schlecht (Strafwert), nicht als Abbruch
- Meldungen: Minimum/Maximum, "x an der Unter-/Obergrenze", Zahl der Lösungen und der
  Kandidaten ohne Lösung; Zielgröße konstant -> "ändert sich nicht" (Teillösung)
- Mehrere Anweisungen: der Reihe nach; Prüfung, ob jede Zielgröße bei den Endwerten ihr
  Optimum behält (sonst lokale Runden, max. 5), und Kopplungstest durch Verschieben der
  Größen der anderen Anweisungen -> Hinweis "beeinflussen sich gegenseitig"
- Mit Parameterstudie/Wertelisten: Optimum je Punkt (`optimize_parametric` ->
  `solve_parametric(point_solver=..., extra_result_vars=...)`), Optimum des Vorpunkts als
  zusätzlicher Kandidat
- Zeitlimit `OPTIMIZE_TIME_LIMIT` = 120 s je Optimierung (je Punkt); danach bestes Ergebnis
  mit Meldung "Zeitlimit der Optimierung ..."
- `solver._inferred_units` speichert die Einheiten-Propagation je Gleichungssatz zwischen
  (sonst ~80 % der Rechenzeit bei vielen Lösungen desselben Systems)

### Fehleranalyse (diagnostics.py)

**Grundsatz:** Fehleranalysen sind immer GENERISCH - sie arbeiten auf der Struktur des
Gleichungssystems bzw. auf Namen, nie auf der Form einzelner Gleichungen. Keine
Sonderbehandlung für bestimmte Gleichungstypen.

1. **Syntax** (parser.py, je Zeile): Python-Syntaxbaum, Zeile + Fundstelle
2. **Struktur** (`analyze_structure`): Dulmage-Mendelsohn-Zerlegung des bipartiten
   Graphen Gleichungen ↔ Unbekannte (maximales Matching). Liefert den unterbestimmten
   Teil (welche Unbekannten in welchen Zeilen, wie viele Gleichungen fehlen) und den
   überbestimmten Teil - auch wenn die Gesamtzahl von Gleichungen und Unbekannten stimmt
3. **Numerik** (`describe_unsolved`): Unbekannte, die trotz korrekter Struktur nicht
   gelöst wurden, mit Zeilen und möglichen Ursachen
4. **Namens-Hinweise** (`name_hints`): Unbekannte, die nur in EINER Gleichung vorkommen,
   daraus implizit berechnet werden und deren Name einem vorhandenen Namen bis auf
   Schreibweise gleicht oder aus zwei vorhandenen Namen besteht (`r_1h_i` = `r_1`+`h_i`).
   Nur Hinweis (GUI: "ⓘ HINWEISE (n)", klickbar) - ein formal lösbares System mit
   Tippfehler kann strukturell nicht als Fehler erkannt werden

Überbestimmte, aber widerspruchsfreie Teile sind erlaubt (der Solver prüft die
zusätzlichen Gleichungen); bei Widerspruch nennt die Meldung beide Seiten.

### Thermodynamik (thermodynamics.py)
- CoolProp wrapper with intuitive syntax
- Functions: `enthalpy`, `entropy`, `density`, `volume`, `intenergy`, `quality`, `temperature`, `pressure`, `viscosity`, `conductivity`, `prandtl`, `cp`, `cv`, `soundspeed`
- Input parameters: `T`, `p`, `h`, `s`, `x`, `rho`, `d`, `u`, `v`

### Humid Air (humid_air.py)
- CoolProp HumidAirProp wrapper for psychrometric calculations
- Syntax: `HumidAir(property, T=..., rh=..., p_tot=...)`
- **Internal units: SI (K, Pa, J/kg)**
- **Output properties** (first argument):
  - `T` - Dry bulb temperature [K]
  - `h` - Specific enthalpy [J/kg_dry_air]
  - `rh` - Relative humidity [-] (0-1)
  - `w` - Humidity ratio [kg_water/kg_dry_air]
  - `p_w` - Partial pressure of water vapor [Pa]
  - `rho_tot` - Density of humid air [kg/m³]
  - `rho_a` - Density of dry air [kg/m³]
  - `rho_w` - Density of water vapor [kg/m³]
  - `T_dp` - Dew point temperature [K]
  - `T_wb` - Wet bulb temperature [K]
- **Input parameters** (exactly 3 required):
  - `T` - Temperature [K] (use `25 °C` or `298.15 K`)
  - `p_tot` - Total pressure [Pa] (use `1 bar` or `100000 Pa`)
  - `rh` - Relative humidity [-]
  - `w` - Humidity ratio [kg/kg]
  - `p_w` - Partial pressure water vapor [Pa]
  - `h` - Enthalpy [J/kg]
  - `T_dp` - Dew point [K] (EES: D=), `T_wb` - Wet bulb temperature [K] (EES: B=)
- Case-insensitive: `HumidAir` = `humidair`; Meldungen nennen die Namen wie dokumentiert
  (`humid_air.display_names`: T, rF, T_dp)
- Deutsche Notation: `x` = w (Wassergehalt; in HumidAir eindeutig), `phi` = rh, `p` = p_tot, Ausgabe
  `v` (m³/kg trockene Luft); `AirH2O` (EES) in Stoffwertfunktionen -> Verweis auf HumidAir; w/x tragen
  das Label kg/kg (Startwert 0.01, Anzeige umschaltbar auf g/kg); rh außerhalb 0…1 -> Meldung
  ("als Anteil (0.5) oder mit Einheit (50 %)")
- **Sättigung** (`humid_air._saturation`, für jede Eingabekombination): w > w_s(T, p)·(1 + 1e-6)
  -> Meldung "Zustand übersättigt: w = … > w_s = … bei T, p" (Nebel/Kondensat in der Bilanz
  vergessen) statt stiller Werte; gesättigte Luft ergibt rh = 1 (CoolProp lehnt rh = 1 + ε ab)

**Examples:**
```
{Calculate enthalpy - result in J/kg}
T_air = 25 °C
p = 1 bar
h = HumidAir(h, T=T_air, rh=0.5, p_tot=p)

{Calculate temperature from enthalpy}
h_in = 50000 J/kg
T = HumidAir(T, h=h_in, rh=0.5, p_tot=p)

{Dew point temperature}
T_dp = HumidAir(T_dp, T=30°C, w=0.012, p_tot=1bar)

{Different parameter combinations}
w = HumidAir(w, T=25°C, rh=0.6, p_tot=1bar)
rh = HumidAir(rh, T=298.15K, w=0.01, p_tot=100000Pa)
rho = HumidAir(rho_tot, T=25°C, rh=0.5, p_tot=1bar)
```

### Strahlung (radiation.py)
- Schwarzkörper-Funktionen basierend auf dem Planck'schen Strahlungsgesetz
- **Alle Funktionen vektorisiert** (unterstützen numpy-Arrays)
- Funktionen:
  - `Eb(T, lambda)` - Spektrale Emissionsleistung [W/m³] (Anzeige: W/(m²·µm))
  - `Blackbody(T, lambda1, lambda2)` - Anteil der Energie im Wellenlängenbereich [-]
  - `Blackbody_cumulative(T, lambda)` - Kumulativer Anteil von 0 bis λ [-]
  - `Wien(T)` - Wellenlänge maximaler Emission [m] (Anzeige: µm)
  - `Stefan_Boltzmann(T)` - Gesamtemission [W/m²]
- Einheiten intern SI wie überall: T in K, λ in m (`L = 5 µm` → 5e-6 m)
- Zahlen ohne Einheit sind wie überall SI-Werte, keine Deutung nach der Größe: `Eb(1000, 5)`
  sind 5 m, `L = 5` sind 5 m (wie `T = 20` → 20 K). Eine Zahl als Wellenlänge im Aufruf
  ergibt einen Hinweis "ⓘ" mit Wert in m und µm (`parser.wavelength_literals`)
- Einheiten auch direkt in den Argumenten: `Eb(500 °C, 5 µm)`, `Wien(500 °C)`
- Eingabe: `T = 500 °C` oder `T = 773.15 K`
- Groß-/Kleinschreibung egal: `Eb` = `eb`, `Blackbody` = `blackbody`

## Einheiten

**Wichtig:** Alle Berechnungen erfolgen intern in **SI-Basiseinheiten** (K, Pa, J, W).
Eingaben mit anderen Einheiten (°C, bar, kJ) werden automatisch konvertiert.

| Größe | Interne Einheit (SI) | Eingabe-Beispiele |
|-------|----------------------|-------------------|
| Temperatur T | K | `25 °C`, `298.15 K` |
| Druck p | Pa | `1 bar`, `100000 Pa` |
| Enthalpie h | J/kg | `100 kJ/kg`, `100000 J/kg` |
| Entropie s | J/(kg·K) | `1 kJ/(kg*K)`, `1000 J/(kg*K)` |
| Innere Energie u | J/kg | |
| Spez. Wärme cp, cv | J/(kg·K) | `1000 J/(kg*K)`, `1 kJ/(kg*K)` |
| Gaskonstante R | J/(kg·K) | `287 J/(kg*K)` |
| Dichte rho | kg/m³ | |
| Spez. Volumen v | m³/kg | |
| Energie | J | `1 kJ`, `1000 J` |
| Leistung | W | `1 kW`, `1000 W` |
| Dampfqualität x | - (0-1) | |
| Länge, Wellenlänge λ | m | `5 µm`, `20 cm`, `5e-6 m` |
| Fläche, Volumen | m², m³ | `50 cm^2`, `200 L` |
| Spektrale Emission Eb | W/m³ (Anzeige W/(m²·µm)) | |
| Gesamtemission E | W/m² | |
| **Winkel (Trigonometrie)** | **Grad (°)** | `30 deg`, `0.5236 rad` (-> 30) |

### Humid Air Units

| Property | Internal Unit (SI) |
|----------|-------------------|
| Enthalpy h | J/kg_dry_air |
| Humidity ratio w | kg_water/kg_dry_air |
| Relative humidity rh | - (0-1) |
| Partial pressure p_w | Pa |
| Total pressure p_tot | Pa |
| Densities rho_tot, rho_a, rho_w | kg/m³ |
| Dew point temperature T_dp | K (Anzeige: °C wählbar) |
| Wet bulb temperature T_wb | K (Anzeige: °C wählbar) |

### Trigonometrische Funktionen

Alle trigonometrischen Funktionen verwenden **Grad**:

```
cos(60) = 0.5       {60°}
sin(30) = 0.5       {30°}
tan(45) = 1.0       {45°}
acos(0.5) = 60      {Ergebnis in Grad}
asin(0.5) = 30      {Ergebnis in Grad}
atan(1) = 45        {Ergebnis in Grad}
```

Hyperbolische Funktionen (`sinh`, `cosh`, `tanh`) verwenden Radiant.

Winkel mit Einheit werden nach Grad umgerechnet (interne Einheit wie die Winkelfunktionen;
`units._convert_to_standard`, Anzeige `from_si_base`): `a = 0.5236 rad` -> 30, `sin(a)` = 0.5.
Früher wurden sie nach Radiant (pint-Basis) umgerechnet und dann als Grad gelesen
(sin(30 deg) = 0.009, ohne Meldung).

## Einheiten-System (v3.0)

### Units Module (units.py)
- Einheiten-Parsing und Konvertierung basierend auf `pint`
- `UnitValue`-Klasse speichert SI-Wert und Original-Einheit
- Automatische Konvertierung zu SI für Berechnungen - für JEDE Einheit (auch cm², L,
  kW/m², kW/(m²K), mPa·s, mm²/s, µm); die Einheiten-Strings sind nur Anzeige-Labels
- Startwerte aus der Einheit über die Dimension (`get_initial_from_unit`), z.B.
  K → 350, delta_K → 10, W/(m²K) → 10, 1/K → 0.0034
- Unterstützte Einheiten: °C, K, bar, Pa, kJ, W, kg/s, m²/s, W/m²K, µm, etc.

### Unit Constraints Module (unit_constraints.py)
- **Zweistufige Einheiten-Ableitung** (wie Olsson 2025, "Improved Unit Inference and Checking in
  Modelica", Dymola): Stufe 1 = lokale Propagation (`_infer_dimensions`, hält lesbare Labels wie
  kW, kJ/kg); Stufe 2 = vollständige Hindley-Milner-Ableitung nach Kennedy (`_complete_inference`):
  jede Gleichung/jeder Teilausdruck liefert eine Einheiten-Gleichung (Summe: gleich, Produkt:
  Exponenten addieren, x^n: n-fach, sqrt: halb, sin/exp/ln: Argument dimensionslos; Zahlen in
  Produkten dimensionslos, allein in Summen beliebig) - ein lineares System in den Exponenten der
  SI-Basiseinheiten, gemeinsam exakt (Brüche) eliminiert. Findet gekoppelte Einheiten
  (a*b = X, a/b = Y; v*x = w, v*v = k). Widersprüchliche Gleichungen werden ausgelassen.
  Variable Exponenten (y^n): nur Exponent dimensionslos, keine Willkür-Inferenz der Basis
- **Fehlende Einheitenangaben** (`missing_unit_annotations`): Größen, deren Einheit aus den
  Gleichungen nicht folgt, und je Gruppe die freien Größen der Elimination (Rang-Defekt) -
  gibt man deren Einheit an, folgen alle übrigen. GUI-Hinweis "Einheit nicht bestimmbar ...
  Einheit von q angeben" (nur in Blättern mit Einheiten). Angabe als Startwert mit Einheit
  (`parser.start_value_units`, wie "Variable Info" in EES)
- **Einheiten-Propagation**: Leitet Einheiten für berechnete Variablen ab
- **Dimensionsanalyse**: Verwendet AST-Parsing für algebraische Ausdrücke
- **Rückwärts-Propagation**: Bei `q = h*dT` wird `h = q/dT` abgeleitet
- **Konsistenzprüfung**: Warnt bei inkonsistenten Einheiten; bei einer Summe mit verschiedenen
  Einheiten nennt die Warnung Terme und Einheiten und als Größe den Verdächtigen
  (`_incompatible_sum`: "h_9 - h_11s: h_9 in J/kg, h_11s dimensionslos (ohne Einheit eingegeben?)");
  ist die dimensionslose Größe berechnet, nennt sie die ohne Einheit eingegebene Größe, von der sie
  die Dimension über Summen erbt (`_dimensionless_sources`: T_B - T_s mit T_s = -10 -> T_s)
- **Additive Ketten**: Bei `A + B - C = 0` erhalten alle Terme die gleiche Dimension

**Wichtige Funktionen:**
- `propagate_all_units()` / `propagate_all_units_complete()`: Hauptfunktionen für Einheiten-Propagation
- `analyze_equation()`: Analysiert eine Gleichung und leitet Einheiten ab
- `_infer_from_additive_chain()`: Propagiert Dimensionen in Addition/Subtraktion-Ketten
- `_collect_additive_terms()`: Sammelt alle Terme aus +/- Ketten (ignoriert numerische Konstanten)
- `_infer_from_mult_div()`: Rückwärts-Inferenz für Multiplikation/Division
- `temperature_sum_conflicts()`: Summen von Temperaturen, die mit keinem Charakter (absolut 1 /
  Differenz 0) aufgehen - typisch eine Differenz in °C angegeben (Hinweis in der GUI)

### Einheiten-Syntax

```
T_s = 90 °C              {Temperatur}
p = 1 bar                {Druck}
sigma = 5.67e-8 W/m^2K^4 {Stefan-Boltzmann}
L = 4 µm                 {Wellenlänge}
h = 25 W/m^2K            {Wärmeübergangskoeffizient}
A = 20 cm2               {Exponent auch ohne ^: m2, m3/h, kg/m3, W/m2K}
U = 0.3 W/(m²·K)         {Malpunkt erlaubt}
```

- Unbekannte Einheiten sind ein **Fehler** ("Unbekannte Einheit 'qcm'") - früher wurde
  der Zahlenwert stillschweigend unumgerechnet übernommen.
- Einheiten in Funktionsargumenten werden auf die Dimension geprüft: `T=` Temperatur,
  `p=`/`p_tot=` Druck, `h=` J/kg, `s=` J/(kg·K), `x=`/`rh=`/`w=` dimensionslos, Strahlung
  `(T, λ, ...)` - `enthalpy(water, T=20 Grad, p=1 bar)` ergibt eine Fehlermeldung.

### Konsistente SI-Berechnung

Mit SI-Basiseinheiten funktioniert die Einheiten-Arithmetik korrekt:
```
{Ideale Gasgleichung: p*v = R*T}
T = 20 °C               {wird zu 293.15 K}
R = 287 J/(kg*K)        {bleibt 287 J/(kg·K)}
p = 1 bar               {wird zu 100000 Pa}
p*v = R*T               {v = 287*293.15/100000 = 0.84134 m³/kg}
```

### Automatische Einheiten-Ableitung

Bei Gleichungen wie:
```
q_dot = h*(T_s - T_inf)
```
wird automatisch erkannt, dass `q_dot` die Einheit `W/m²` hat.

### Einheiten-Propagation bei Addition/Subtraktion

Bei Gleichungen mit Addition/Subtraktion müssen alle Terme die gleiche Dimension haben.
Die Einheiten-Propagation erkennt dies automatisch:

```
{Massenbilanz - m_dot_3 wird automatisch als kg/s erkannt}
m_dot_1 = 1 kg/s
m_dot_2 = 2 kg/s
m_dot_1 + m_dot_2 - m_dot_3 = 0

{Energiebilanz - h_3 wird automatisch als kJ/kg erkannt}
h_1 = 100 kJ/kg
h_2 = 200 kJ/kg
m_dot_1*h_1 + m_dot_2*h_2 - m_dot_3*h_3 = 0
```

**Funktioniert für:**
- Beliebige Anzahl von Termen: `A + B + C + D + E - F = 0`
- Komplexe Ausdrücke: `dT/ln(T2/T1) + dT2*sin(x) + dT5*22 = 0`
- Numerische Konstanten (0, 1, etc.) werden ignoriert - sie sind dimensional neutral

### Dimensionslose Größen

Dimensionslose Zahlen werden automatisch erkannt:
- Nusselt-Zahl: `Nu = h*L/k`
- Grashof-Zahl: `Gr = g*beta*L^3*dT/nu^2`
- Prandtl-Zahl: `Pr = nu/alpha`
- Strahlungsanteile: `F = Blackbody(T, lambda1, lambda2)`

### Unterstützte Einheiten-Typen

| Kategorie | Einheiten |
|-----------|-----------|
| Temperatur | °C, K, °F |
| Temperaturdifferenz | K (Eingabe und Anzeige; intern delta_K) - aus der Struktur, nicht aus dem Namen |
| Druck | bar, Pa, kPa, MPa, atm, psi |
| Energie | kJ, J, kWh |
| Leistung | kW, W |
| Kraft | N, kN |
| Beschleunigung | m/s², m/s^2 |
| Masse | kg, g |
| Massenstrom | kg/s, kg/h |
| Wärmestromdichte | W/m² |
| Wärmeübergangskoeff. | W/m²K |
| Wärmedurchlasswiderstand | m²K/W |
| Wellenlänge | µm, nm, m |
| Stefan-Boltzmann | W/m²K⁴ |
| Kinematische Viskosität | m²/s |
| Wärmeleitfähigkeit | W/mK |

### Temperaturdifferenzen (immer in K)

**Festlegung:** `°C`/`°F` sind IMMER absolute Temperaturen (Offset); Temperaturdifferenzen
werden in `K` eingegeben (K hat keinen Offset - der SI-Wert ist für Temperatur und
Differenz derselbe). Damit ist jeder Zahlenwert ohne Blick auf den Namen richtig; es gibt
KEINE Namenskonvention mehr (früher: `dT...`/`delta...`).

Ob eine Größe eine absolute Temperatur (Gewicht 1) oder eine Differenz (0) ist, folgt aus
der Struktur (`unit_constraints._resolve_temperature_weights`, ohne Namen):
- in °C/°F eingegeben: absolut; in K eingegeben: offen (aus den Gleichungen bestimmt,
  `open_temperatures`)
- `T_1 - T_2` Differenz; `T_abs ± Differenz` absolut; `(T_1 + T_2)/2` absolut;
  `T=`-Argumente von Stoffwert-/HumidAir-/Strahlungsfunktionen absolut
- jede Temperaturgröße ist absolut ODER Differenz: in `T_2 = T_1 + x` (T_1 absolut) ist
  daher x die Differenz und T_2 absolut (`_integral_weight_candidates`)
- Anzeige zusätzlich (niedrigste Priorität, `differences_in_products=True`): eine Temperatur
  in einem Produkt/Quotienten, dessen Dimension KEINE Temperatur ist, ist eine Differenz
  (Konvention wie COMSOL): `Q = m*c*(T1 - T2)` -> (T1 - T2) Differenz, mit T1 absolut also
  T2 absolut (°C); `Q = m*c*theta` -> theta Differenz (K). Nur Kandidaten mit Ergebnis 0/1
  (Mittelwert (T_1 + T_2)/2 im Produkt bleibt absolut). Mathematischer Hintergrund: absolute
  Temperaturen sind Punkte (Torsor), Differenzen Vektoren (Punkt - Punkt = Vektor,
  Punkt ± Vektor = Punkt, Punkt + Punkt undefiniert)
- Anzeige zusätzlich: eine Temperatur-VARIABLE mit ganzzahligem Exponenten >= 2 (`sigma*T^4`)
  ist absolut (Kelvin-Verhältnisskala); gebrochene Exponenten/Summen als Basis bleiben offen
  (`(T_s - T_inf)^(1/3)`). Damit: `dT_solar = T_ms - T_s` mit T_ms, T_s aus T^4 -> Differenz
- Anzeige, vor der Produktregel: **Nullpunkt-Test** mit der Lösung (`_zero_point_candidates`,
  `main._zero_point_evaluator`): eine Gleichung bleibt gültig, wenn der Nullpunkt der
  Temperaturskala verschoben wird - absolute Temperaturen verschieben sich mit, Differenzen
  nicht. Je freier Temperaturgröße (übrige mit Charakter 0/1) numerisch: `dT = Q/(m*c)`,
  `x = q*R_si`, `y = (dT/dt)*t` -> Differenz (K); Mischung `m_3*T_3 = m_1*T_1 + m_2*T_2` mit
  m_3 = m_1 + m_2 -> absolut (°C). Nur für Gleichungen, die in allen Temperaturen linear sind
  (nicht sigma*T^4, Korrelationen, Stoffwertfunktionen), nur Ergebnis genau 0 oder 1;
  physikalische Gesetze in Kelvin (`p*v = R*T`, `T = p*v/R`) ergeben wie die Produktregel K.
  Umgestellte Formen derselben Gleichung (`Q = m*c*dT` / `dT = Q/(m*c)`) erhalten denselben
  Charakter
- nicht bestimmbar: berechnete Größen gelten als absolut (°C nach Settings; Startwerte wie
  absolut), in K eingegebene Größen werden wie eingegeben in K angezeigt
  (`main._kelvin_display`) - eine in K eingegebene Differenz erscheint nie als -263 °C
- Umrechnung im EES-/Excel-Stil `T + 273.15` (`scale_offset_literals`): eine Zahl gleich dem
  Nullpunkt einer Temperaturskala (273.15, 459.67 - aus pint) in einer Summe mit einer
  Temperatur -> Hinweis "ⓘ": T ist bereits in K, die Zahl verschiebt den Nullpunkt erneut
- Zahl ohne Einheit in einer Summe mit einer in einer Nicht-SI-Einheit eingegebenen Größe
  (`24/(24 - t_S)` mit t_S = 2 h -> 24 s; `si_number_literals`, ohne Temperaturen, dimensionslose
  Summen und 0) -> Hinweis "ⓘ" (die Zahl ist ein SI-Wert)
- Widerspruch (`temperature_sum_conflicts`): Werte in °C so addiert, dass weder Temperatur
  noch Differenz herauskommt (`T_2 = T_1 + x`, `x = 10 °C`) -> Hinweis "ⓘ HINWEISE"
- **Gerechnet wird IMMER in Kelvin** - keine Regel rechnet eine Gleichung auf einer anderen
  Skala oder deutet Zahlen nach dem Zusammenhang um (allgemeingültig). Auch eine Summe absoluter
  Temperaturen (T_3 = T_1 + T_2: 20 °C + 40 °C = 606.3 K = 333.15 °C) wird in Kelvin gerechnet,
  dazu der Hinweis aus `temperature_sum_conflicts`. Kein Hinweis für "Differenz mal Größe"
  (`EER_C*(T_c - T_0) = T_0`, aus EER_C = T_0/(T_c - T_0) ausmultipliziert; `_kelvin_law_term`)
- Anzeige: Differenzen in K (`pretty_unit('delta_K')` = 'K', DIN 1345 / ISO 80000-5; die
  Einheiten-Auswahl rechnet sie ohne Offset in °C/°F um), absolute Temperaturen nach
  Settings (°C/K)

## Bekannte Einschränkungen / Design-Entscheidungen

0. **Temperaturskala / Zahlenwertgleichungen**: Temperaturen werden intern in Kelvin
   gerechnet. Physikalische Gesetze (p·v = R·T, σT⁴, Isentrope) funktionieren direkt;
   Formeln, die nur für Zahlenwerte in bestimmten Einheiten gelten (Heizkurve in °C,
   Magnus-Formel, cp(ϑ)-Polynome, h = 5.7 + 3.8·v), werden mit `value(x, Einheit)` und
   `quantity(z, Einheit)` geschrieben (units.unit_number/unit_quantity, Nullpunkt und Faktor
   aus pint, Winkel bezogen auf Grad). Ohne diese Funktionen: falsches Ergebnis ohne Meldung -
   eine Zahlenwertgleichung ist strukturell nicht von einem Gesetz in K unterscheidbar.
   Die Einheiten-Ableitung kennt beide Funktionen (Argument von value hat die Dimension der
   Einheit, quantity liefert sie; °C/°F -> absolute Temperatur; falsche Einheit -> Warnung).

1. **T und p auf der Sättigungslinie** (Sattdampf mit T und p angegeben) -> Meldung "Dampfgehalt
   angeben (x=1 ...)" statt der englischen CoolProp-Meldung.
   **Nassdampf unterhalb des Tripelpunkts** (T < T_triple bzw. p < p_triple mit x gegeben) wird gemeldet
   (Wasser: Eis) - Grenzen aus CoolProp je Fluid.
   **Dampfgehalt x**: nur Rundungsfehler (±1e-6) werden auf [0, 1] begrenzt; x deutlich
   außerhalb (x = 2, x = quality(...) = -1 eines einphasigen Zustands) ist eine Meldung
   "Dampfgehalt x = … liegt außerhalb von 0 ... 1" (thermodynamics.py) - für den Solver ein
   ungültiger Iterationspunkt. `quality()` liefert außerhalb des Nassdampfgebiets -1
   (CoolProp; unterkühlt, überhitzt, überkritisch) - dann Hinweis "ⓘ" (`main._single_phase_quality_hints`:
   jeder quality-Aufruf wird mit der Lösung ausgewertet).

2. **Volumen als Input**: `v` wird intern zu Dichte umgerechnet (`rho = 1/v`), da CoolProp mit Dichte arbeitet.

3. **Indirekte Berechnungen**: Manche Kombinationen (z.B. `quality(water, h=2000, T=0)`) werden von CoolProp nicht direkt unterstützt. Workaround: Als iteratives Problem formulieren:
   ```
   h_ziel = 2000
   T = 0
   h_berechnet = enthalpy(water, T=T, x=x)
   h_berechnet = h_ziel
   ```

4. **Manuelle Startwerte**: Bei sehr speziellen Gleichungssystemen kann der Dialog "Solve → Initial Values..." zur manuellen Anpassung verwendet werden. Die Werte stehen danach als Block `{$Startwerte ... $}` im Blatt und werden mit der Datei gespeichert.

5. **Mehrdeutige Wurzeln**: Bei mehreren Lösungen wird die dem Startwert nächstgelegene
   gewählt (Startwert manuell bzw. aus der Einheit, sonst 1) - bei Bedarf Startwert im
   Dialog setzen (wird als Block `{$Startwerte ... $}` im Blatt gespeichert).

## GUI-Features

- File: New, Open, Save, Save As (.hes, .txt)
- Beispielblatt ("☰ Examples" / Edit > Insert Example): Wasser/Dampf, feuchte Luft, Strahlung,
  Heizkurve (value/quantity, Spreizung in K), Rohrströmung (IF), wirtschaftliche Dämmdicke
  (MINIMIZE); `test_gui.py` prüft die Werte (Dämmdicke gegen die analytische Lösung)
- View: Schriftgröße 6-36pt (Standard: 16pt)
- Solve: F5 oder Button, Initial Values Dialog (Werte auch mit Einheit; schreibt den Block `{$Startwerte ... $}` ins Blatt)
- Zwischenablage: Kopieren/Ausschneiden im Editor schreibt zusätzlich fest in die
  System-Zwischenablage (macOS `pbcopy`, Linux `wl-copy`/`xclip`/`xsel`) - Tk stellt
  Inhalte sonst nur bereit, solange das Programm läuft (nach Neustart wäre er weg)
- Einheiten-Warnungen: Klick auf "⚠ UNIT WARNINGS (n)" im Results-Tab springt zu den
  Warnungen im Residuals-Tab (Variable, Gleichung, links/rechts in SI-Einheiten)
- Widersprüchliches (überbestimmtes) System: Meldung nennt vorgegebenen und berechneten
  Wert, z.B. "q_dot ist vorgegeben (50), aus 'q_dot=...' folgt 29.06"
- Anzeige-Einheiten berechneter Größen: Vielfache bzw. Summen gleichartiger Eingaben behalten deren
  Einheit (`V_dot = n*V_dot_P` mit m3/h -> m3/h, `d_i = d_a - 2*s` in mm, `x_3 = x_1 + 0.001` in g/kg,
  `eta*0.5` in %; `unit_constraints._scaled_label`, `main._assign_result_units` übergibt die eingegebenen
  Einheiten als Labels) - außer wo die Settings bestimmen (Leistung, Energie, Druck, J/kg, J/(kg·K),
  Temperatur: `x = 2*P_el` mit P_el in MW erscheint in kW); dimensionslose Ergebnisse umschaltbar
  (-, %, ‰, g/kg), Feuchtebeladung mit Label kg/kg, 1/s auch in 1/h. Produkte/Quotienten zweier
  Größen mit Einheit (`_composed_label`): kürzt sich die Einheit auf eine (V_dot/V mit m3/h -> 1/h,
  V/V_dot -> h, A/L in mm), bleibt sie; Leistung mal Zeit je Länge/Volumen -> kWh/m, kWh/m³ (statt
  N/bar). IF/max/min behalten eine gemeinsame Eingabe-Einheit. Ein Startwert mit Einheit legt die
  Anzeige fest (wie Variable Info in EES): `q_V = 600 kJ/m^3` -> kJ/m³ statt bar (gleiche Dimension
  wie Druck - die Größenart folgt nicht aus der Dimension, p = R*T/v bleibt bar). Rundungsrest
  nach Offset-Umrechnung (-1.7e-13 °C) wird als 0 angezeigt (`main._si_to_unit`)
- Ergebnisanzeige: Wert und Einheit stammen immer aus derselben Einheit. Die Settings
  (°C/K, bar/Pa, kJ/J, kW/W) gelten für Temperaturen, Drücke, J/kJ, J/kg, J/(kg·K), W/kW;
  andere Einheiten (MW, kWh, W/(m²K), W/K, ...) bleiben wie eingegeben bzw. abgeleitet.
  Parameterstudien und Plots werden ebenfalls in diesen Einheiten angezeigt.
- Plot: Diagramme für Parameterstudien (erfordert matplotlib)
  - New Plot Window: Mehrere Y-Variablen, Labels, Titel, Optionen
  - Quick Plot X-Y: Schneller einfacher Plot
  - **Interaktive Toolbar**: Zoom, Pan, Home, Save (oben im Plot-Fenster)
- Help: Function Reference (Syntax, Einheiten, Funktionen inkl. IF, Startwerte, Meldungen), Fluid List
  (aus `thermodynamics.FLUID_ALIASES` und der CoolProp-Fluidliste erzeugt - alle 124 Fluide)
- Settings: Anzeige-Einheiten (°C/K, bar/Pa, kJ/J, kW/W) wirken sofort auf die Ergebnisse
- Einheiten werden als °C, °F, µm angezeigt (intern pint-Namen degC, degF, um)

## Parameterstudien (Sweep)

Vektor-Syntax für Parameter-Sweeps:
```
p_1 = 25:5:50       {start:step:end -> 25, 30, 35, 40, 45, 50}
T = 0:100           {start:end mit step=1}
h = enthalpy(water, T=T, x=0)
```

Direkte Funktionsauswertung (ohne Iteration):
```
L = 0.5:0.5:20 µm
E = Eb(300 °C, L)   {Spektrale Emission bei 300 °C über der Wellenlänge}
```

Nach dem Lösen: Plot → New Plot Window oder Quick Plot X-Y

## Beispiel: Dampfkraftprozess

```
{Frischdampf}
m_dot_1 = 10000/3600 kg/s
T_1 = 450 °C
p_1 = 30 bar
h_1 = enthalpy(water, p=p_1, T=T_1)
s_1 = entropy(water, p=p_1, T=T_1)

{Hochdruck-Turbine}
p_2 = 2.5 bar
eta_s_i_T = 0.8
h_2s = enthalpy(water, p=p_2, s=s_1)
eta_s_i_T = (h_2-h_1)/(h_2s-h_1)

{Kondensator}
x_4 = 0
T_4 = 50 °C
p_4 = pressure(water, x=x_4, T=T_4)
h_4 = enthalpy(water, x=x_4, T=T_4)

{Wirkungsgrad}
W_dot_T = m_dot_1*(h_1-h_2)
Q_dot = m_dot_1*(h_1-h_4)
eta_th = W_dot_T/Q_dot
```

## Example: Humid Air - Air Conditioning

```
{Outdoor air (State 1)}
T_1 = 35 °C
rh_1 = 0.6
p = 1 bar

{Calculate state properties}
h_1 = HumidAir(h, T=T_1, rh=rh_1, p_tot=p)
w_1 = HumidAir(w, T=T_1, rh=rh_1, p_tot=p)
T_dp_1 = HumidAir(T_dp, T=T_1, rh=rh_1, p_tot=p)
T_wb_1 = HumidAir(T_wb, T=T_1, rh=rh_1, p_tot=p)

{Conditioned air (State 2)}
T_2 = 22 °C
rh_2 = 0.5
h_2 = HumidAir(h, T=T_2, rh=rh_2, p_tot=p)
w_2 = HumidAir(w, T=T_2, rh=rh_2, p_tot=p)

{Cooling load}
m_dot_a = 1000/3600 kg/s
Q_dot_cool = m_dot_a*(h_1-h_2)
m_dot_condensate = m_dot_a*(w_1-w_2)
```

## Example: Mechanics - Force Calculation

```
{Force calculation: F = m * g}
m = 100 kg
g = 9.81 m/s^2
F = m * g
{Result: F = 981 N}
```

## Example: Thermal Radiation

```
{Surface temperature and properties}
T_surface = 500 °C
epsilon = 0.85
A = 2 m^2
sigma = 5.67E-8 W/(m^2*K^4)

{Stefan-Boltzmann radiation}
Q_rad = epsilon * sigma * A * T_surface^4

{Peak wavelength (Wien's law)}
lambda_max = Wien(T_surface)

{Spectral emissive power}
L = 5 µm
E_spectral = Eb(T_surface, L)

{Fraction of energy in visible range}
f_visible = Blackbody(T_surface, 0.38 µm, 0.75 µm)
```
