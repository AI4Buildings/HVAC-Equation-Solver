# HVAC Equation Solver

Ein EES-ähnlicher (Engineering Equation Solver) Gleichungslöser mit CoolProp-Integration für thermodynamische Stoffdaten und Feuchte-Luft-Berechnungen.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Features

- **EES-kompatible Syntax**: Gleichungen in natürlicher Form (`h = enthalpy(water, T=100, p=1)`)
- **Thermodynamische Stoffdaten**: Über 100 Fluide via CoolProp
- **Feuchte Luft**: Psychrometrische Berechnungen (`h = HumidAir(h, T=25, rh=0.5, p_tot=1)`)
- **Robuster Solver**: Block-Dekomposition mit Bracket-Suche und Brent's Methode
- **Parameterstudien**: Einfache Sweep-Syntax (`p = 25:5:50`)
- **Schwarzkörper-Strahlung**: Planck'sche Strahlungsfunktionen
- **GUI**: Tkinter-basierte Benutzeroberfläche mit Plot-Funktionen

## Installation

### Erforderliche Bibliotheken

```bash
pip install numpy scipy CoolProp matplotlib
```

| Bibliothek | Version | Zweck |
|------------|---------|-------|
| numpy | >= 1.20 | Array-Operationen, mathematische Funktionen |
| scipy | >= 1.7 | Numerische Solver (fsolve, brentq) |
| CoolProp | >= 6.4 | Thermodynamische Stoffdaten |
| matplotlib | >= 3.5 | Diagramme und Plots (optional) |
| tkinter | - | GUI (in Python Standard-Library enthalten) |

### Programm starten

```bash
python3 main.py
```

## Screenshot

Die Anwendung bietet eine intuitive Oberfläche zur Eingabe von Gleichungen und Anzeige der Ergebnisse:

- **Linkes Fenster**: Gleichungseingabe mit EES-Syntax
- **Rechtes Fenster**: Lösungsergebnisse
- **Plot-Funktion**: Für Parameterstudien

## Architektur

```
HVAC-Equation-Solver/
├── main.py           # Tkinter GUI (Hauptanwendung)
├── parser.py         # EES-Syntax → Python Konvertierung
├── solver.py         # Block-Dekomposition + Bracket-Suche Solver
├── thermodynamics.py # CoolProp Wrapper mit Einheitenumrechnung
├── humid_air.py      # CoolProp HumidAirProp Wrapper
├── radiation.py      # Schwarzkörper-Strahlungsfunktionen
├── CLAUDE.md         # Technische Dokumentation
└── README.md         # Diese Datei
```

## Schnellstart

### Einfaches Beispiel

```
{Wasserdampf bei 100°C und 1 bar}
T = 100
p = 1
h = enthalpy(water, T=T, p=p)
s = entropy(water, T=T, p=p)
```

### Feuchte Luft Beispiel

```
{Feuchte Luft bei 25°C und 50% rel. Feuchte}
T = 25
rh = 0.5
p_tot = 1
h = HumidAir(h, T=T, rh=rh, p_tot=p_tot)
w = HumidAir(w, T=T, rh=rh, p_tot=p_tot)
T_dp = HumidAir(T_dp, T=T, rh=rh, p_tot=p_tot)
```

### Gleichungssystem

```
x + y = 10
x - y = 2
```

### Parameterstudie

```
T = 0:10:100
p = 1
h = enthalpy(water, T=T, p=p)
```

## Dampfkraftprozess Beispiel

```
{Frischdampf}
m_dot = 10000/3600
T_1 = 450
p_1 = 30
h_1 = enthalpy(water, p=p_1, T=T_1)
s_1 = entropy(water, p=p_1, T=T_1)

{Turbine mit isentropem Wirkungsgrad}
p_2 = 2.5
eta_s = 0.8
h_2s = enthalpy(water, p=p_2, s=s_1)
eta_s = (h_2-h_1)/(h_2s-h_1)

{Turbinenleistung}
W_dot = m_dot*(h_1-h_2)
```

## Klimaanlage Beispiel

```
{Outdoor air}
T_1 = 35
rh_1 = 0.6
p = 1

h_1 = HumidAir(h, T=T_1, rh=rh_1, p_tot=p)
w_1 = HumidAir(w, T=T_1, rh=rh_1, p_tot=p)

{Conditioned air}
T_2 = 22
rh_2 = 0.5
h_2 = HumidAir(h, T=T_2, rh=rh_2, p_tot=p)
w_2 = HumidAir(w, T=T_2, rh=rh_2, p_tot=p)

{Cooling load}
m_dot_a = 1000/3600
Q_dot_cool = m_dot_a*(h_1-h_2)
```

## Solver-Strategie

Der Solver verwendet eine robuste Block-Dekomposition:

1. **Konstanten zuweisen**: Explizite Definitionen wie `T_1 = 450`
2. **Direkte Auswertung**: Gleichungen der Form `var = ausdruck` werden sequentiell berechnet
3. **Einzelne Unbekannte**: Bracket-Suche + Brent's Methode
4. **Blockweise Lösung**: Zusammenhängende Gleichungsblöcke mit `scipy.fsolve`
5. **Iteration**: Schritte 2-4 werden wiederholt bis alle Gleichungen gelöst sind

### Robuste Wurzelfindung

- ~1200 Testpunkte über Größenordnungen von 0.01 bis 10,000,000
- Adaptive Verfeinerung bei Singularitäten
- Standard-Startwert 1.0 für alle Variablen (wie EES)

## Einheiten

| Größe | Einheit |
|-------|---------|
| Temperatur T | °C |
| Druck p | bar |
| Enthalpie h | kJ/kg |
| Winkel (sin, cos, tan) | Grad (°) |
| Entropie s | kJ/(kg·K) |
| Dichte rho | kg/m³ |
| Dampfqualität x | - (0-1) |

### Feuchte Luft Einheiten

| Größe | Einheit |
|-------|---------|
| Enthalpie h | kJ/kg_dry_air |
| Humidity ratio w | kg_water/kg_dry_air |
| Relative humidity rh | - (0-1) |
| Dew point T_dp | °C |
| Wet bulb T_wb | °C |

## Verfügbare Funktionen

### Thermodynamik
`enthalpy`, `entropy`, `density`, `volume`, `intenergy`, `quality`, `temperature`, `pressure`, `viscosity`, `conductivity`, `prandtl`, `cp`, `cv`, `soundspeed`

### Feuchte Luft (HumidAir)
Output: `T`, `h`, `rh`, `w`, `p_w`, `rho_tot`, `rho_a`, `rho_w`, `T_dp`, `T_wb`
Input: `T`, `p_tot`, `rh`, `w`, `p_w`, `h`

### Mathematik
`sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, `exp`, `ln`, `log10`, `sqrt`, `abs`, `pi`

### Strahlung
`Eb`, `Blackbody`, `Blackbody_cumulative`, `Wien`, `Stefan_Boltzmann`

## Lizenz

MIT License

## Beiträge

Beiträge sind willkommen! Bitte erstellen Sie einen Pull Request oder öffnen Sie ein Issue.
