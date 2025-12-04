# HVAC Equation Solver

EES-ähnlicher Gleichungslöser mit CoolProp-Integration für thermodynamische Stoffdaten und Feuchte-Luft-Berechnungen.

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

## Architektur

```
equation_solver/
├── main.py           # Tkinter GUI (Hauptanwendung)
├── parser.py         # EES-Syntax → Python Konvertierung
├── solver.py         # Block-Dekomposition + Bracket-Suche Solver
├── thermodynamics.py # CoolProp Wrapper mit Einheitenumrechnung
├── humid_air.py      # CoolProp HumidAirProp Wrapper für feuchte Luft
├── radiation.py      # Schwarzkörper-Strahlungsfunktionen (Planck, vektorisiert)
```

## Kernfunktionen

### Parser (parser.py)
- Konvertiert EES-Syntax zu Python: `^` → `**`, `ln` → `log`
- Kommentare: `"..."` und `{...}`
- Thermodynamik-Funktionsaufrufe: `enthalpy(water, T=100, p=1)` → `enthalpy('water', T=100, p=1)`
- Extrahiert Variablen aus Gleichungen (filtert Funktionsnamen und Parameter-Keys)
- Vektor-Syntax: `T = 0:10:100` (start:step:end) oder `T = 0:100` (start:end, step=1)
- **Direkte Zuweisungen** wie `T_1 = 450` oder `m = 10000/3600` werden als Konstanten behandelt

### Solver (solver.py)

#### Lösungsstrategie (Block-Dekomposition)
1. **Konstanten zuweisen**: Explizite Definitionen wie `T_1 = 450`
2. **Direkte Auswertung**: Gleichungen der Form `var = ausdruck` werden sequentiell berechnet
3. **Einzelne Unbekannte**: Gleichungen mit nur einer Unbekannten werden mit Bracket-Suche + Brent's Methode gelöst
4. **Blockweise Lösung**: Zusammenhängende Gleichungsblöcke werden mit `scipy.fsolve` gelöst
5. **Iteration**: Schritte 2-4 werden wiederholt bis alle Gleichungen gelöst sind

#### Robuste Wurzelfindung für einzelne Gleichungen
- **Bracket-Suche**: ~1200 Testpunkte über Größenordnungen von 0.01 bis 10,000,000
- **Adaptive Verfeinerung**: Bei großen Funktionsänderungen wird das Intervall verfeinert
- **Brent's Methode**: Robuste Wurzelfindung bei Vorzeichenwechsel (funktioniert auch bei Singularitäten)
- **Standard-Startwert**: 1.0 für alle Variablen (wie EES)

#### Parameterstudien
- Sweep-Variablen werden als Konstanten für jeden Punkt behandelt
- Für jeden Sweep-Punkt wird `solve_system` mit Block-Dekomposition aufgerufen
- Vektorisierte Auswertung für direkte Funktionen ohne Iteration

### Thermodynamik (thermodynamics.py)
- CoolProp-Wrapper mit EES-kompatibler Syntax
- Funktionen: `enthalpy`, `entropy`, `density`, `volume`, `intenergy`, `quality`, `temperature`, `pressure`, `viscosity`, `conductivity`, `prandtl`, `cp`, `cv`, `soundspeed`
- Input-Parameter: `T`, `p`, `h`, `s`, `x`, `rho`, `d`, `u`, `v`

### Humid Air (humid_air.py)
- CoolProp HumidAirProp wrapper for psychrometric calculations
- Syntax: `HumidAir(property, T=..., rh=..., p_tot=...)`
- **Output properties** (first argument):
  - `T` - Dry bulb temperature [°C]
  - `h` - Specific enthalpy [kJ/kg_dry_air]
  - `rh` - Relative humidity [-] (0-1)
  - `w` - Humidity ratio [kg_water/kg_dry_air]
  - `p_w` - Partial pressure of water vapor [bar]
  - `rho_tot` - Density of humid air [kg/m³]
  - `rho_a` - Density of dry air [kg/m³]
  - `rho_w` - Density of water vapor [kg/m³]
  - `T_dp` - Dew point temperature [°C]
  - `T_wb` - Wet bulb temperature [°C]
- **Input parameters** (exactly 3 required):
  - `T` - Temperature [°C]
  - `p_tot` - Total pressure [bar]
  - `rh` - Relative humidity [-]
  - `w` - Humidity ratio [kg/kg]
  - `p_w` - Partial pressure water vapor [bar]
  - `h` - Enthalpy [kJ/kg]
- Case-insensitive: `HumidAir` = `humidair`

**Examples:**
```
{Calculate enthalpy}
h = HumidAir(h, T=25, rh=0.5, p_tot=1)

{Calculate temperature from enthalpy}
T = HumidAir(T, h=50, rh=0.5, p_tot=1)

{Dew point temperature}
T_dp = HumidAir(T_dp, T=30, w=0.012, p_tot=1)

{Different parameter combinations}
w = HumidAir(w, T=25, rh=0.6, p_tot=1)
rh = HumidAir(rh, T=25, w=0.01, p_tot=1)
rho = HumidAir(rho_tot, T=25, rh=0.5, p_tot=1)
```

### Strahlung (radiation.py)
- Schwarzkörper-Funktionen basierend auf dem Planck'schen Strahlungsgesetz
- **Alle Funktionen vektorisiert** (unterstützen numpy-Arrays)
- Funktionen:
  - `Eb(T, lambda)` - Spektrale Emissionsleistung [W/(m²·µm)]
  - `Blackbody(T, lambda1, lambda2)` - Anteil der Energie im Wellenlängenbereich [-]
  - `Blackbody_cumulative(T, lambda)` - Kumulativer Anteil von 0 bis λ [-]
  - `Wien(T)` - Wellenlänge maximaler Emission [µm]
  - `Stefan_Boltzmann(T)` - Gesamtemission [W/m²]
- Einheiten: T in °C, λ in µm
- Groß-/Kleinschreibung egal: `Eb` = `eb`, `Blackbody` = `blackbody`

## Einheiten

| Größe | Einheit |
|-------|---------|
| Temperatur T | °C |
| Druck p | bar |
| Enthalpie h | kJ/kg |
| Entropie s | kJ/(kg·K) |
| Innere Energie u | kJ/kg |
| Dichte rho | kg/m³ |
| Spez. Volumen v | m³/kg |
| Dampfqualität x | - (0-1) |
| Wellenlänge λ | µm |
| Spektrale Emission Eb | W/(m²·µm) |
| Gesamtemission E | W/m² |
| **Winkel (Trigonometrie)** | **Grad (°)** |

### Humid Air Units

| Property | Unit |
|----------|------|
| Enthalpy h | kJ/kg_dry_air |
| Humidity ratio w | kg_water/kg_dry_air |
| Relative humidity rh | - (0-1) |
| Partial pressure p_w | bar |
| Total pressure p_tot | bar |
| Densities rho_tot, rho_a, rho_w | kg/m³ |
| Dew point temperature T_dp | °C |
| Wet bulb temperature T_wb | °C |

### Trigonometrische Funktionen

Alle trigonometrischen Funktionen verwenden **Grad** (wie EES):

```
cos(60) = 0.5       {60°}
sin(30) = 0.5       {30°}
tan(45) = 1.0       {45°}
acos(0.5) = 60      {Ergebnis in Grad}
asin(0.5) = 30      {Ergebnis in Grad}
atan(1) = 45        {Ergebnis in Grad}
```

Hyperbolische Funktionen (`sinh`, `cosh`, `tanh`) verwenden Radiant.

## Bekannte Einschränkungen / Design-Entscheidungen

1. **Quality-Clamping**: Dampfqualität x wird auf [0, 1] begrenzt (thermodynamics.py), damit der iterative Solver nicht mit ungültigen Werten abstürzt.

2. **Volumen als Input**: `v` wird intern zu Dichte umgerechnet (`rho = 1/v`), da CoolProp mit Dichte arbeitet.

3. **Indirekte Berechnungen**: Manche Kombinationen (z.B. `quality(water, h=2000, T=0)`) werden von CoolProp nicht direkt unterstützt. Workaround: Als iteratives Problem formulieren:
   ```
   h_ziel = 2000
   T = 0
   h_berechnet = enthalpy(water, T=T, x=x)
   h_berechnet = h_ziel
   ```

4. **Manuelle Startwerte**: Bei sehr speziellen Gleichungssystemen kann der Dialog "Solve → Initial Values..." zur manuellen Anpassung verwendet werden.

## GUI-Features

- File: New, Open, Save, Save As (.hes, .txt)
- View: Schriftgröße 6-36pt (Standard: 16pt)
- Solve: F5 oder Button, Initial Values Dialog
- Plot: Diagramme für Parameterstudien (erfordert matplotlib)
  - New Plot Window: Mehrere Y-Variablen, Labels, Titel, Optionen
  - Quick Plot X-Y: Schneller einfacher Plot
  - **Interaktive Toolbar**: Zoom, Pan, Home, Save (oben im Plot-Fenster)
- Help: Function Reference, Fluid List

## Parameterstudien (Sweep)

Vektor-Syntax für Parameter-Sweeps:
```
p_1 = 25:5:50       {start:step:end -> 25, 30, 35, 40, 45, 50}
T = 0:100           {start:end mit step=1}
h = enthalpy(water, T=T, x=0)
```

Direkte Funktionsauswertung (ohne Iteration):
```
L = 0:0.1:10
E = Eb(300, L)      {Spektrale Emission bei 300°C über Wellenlänge}
```

Nach dem Lösen: Plot → New Plot Window oder Quick Plot X-Y

## Beispiel: Dampfkraftprozess

```
{Frischdampf}
m_dot_1 = 10000/3600
T_1 = 450
p_1 = 30
h_1 = enthalpy(water, p=p_1, T=T_1)
s_1 = entropy(water, p=p_1, T=T_1)

{Hochdruck-Turbine}
p_2 = 2.5
eta_s_i_T = 0.8
h_2s = enthalpy(water, p=p_2, s=s_1)
eta_s_i_T = (h_2-h_1)/(h_2s-h_1)

{Kondensator}
x_4 = 0
T_4 = 50
p_4 = pressure(water, x=x_4, T=T_4)
h_4 = enthalpy(water, x=x_4, T=T_4)

{Speisewasserpumpe}
eta_s_i_P = 0.92
p_5 = p_11
h_5s = enthalpy(water, p=p_5, s=s_4)
eta_s_i_P = (h_5s-h_4)/(h_5-h_4)

{Wirkungsgrad}
W_dot_T = m_dot_1*(h_1-h_2)
Q_dot = m_dot_1*(h_1-h_4)
eta_th = W_dot_T/Q_dot
```

## Example: Humid Air - Air Conditioning

```
{Outdoor air (State 1)}
T_1 = 35
rh_1 = 0.6
p = 1

{Calculate state properties}
h_1 = HumidAir(h, T=T_1, rh=rh_1, p_tot=p)
w_1 = HumidAir(w, T=T_1, rh=rh_1, p_tot=p)
T_dp_1 = HumidAir(T_dp, T=T_1, rh=rh_1, p_tot=p)
T_wb_1 = HumidAir(T_wb, T=T_1, rh=rh_1, p_tot=p)

{Densities}
rho_tot_1 = HumidAir(rho_tot, T=T_1, rh=rh_1, p_tot=p)
rho_a_1 = HumidAir(rho_a, T=T_1, rh=rh_1, p_tot=p)
rho_w_1 = HumidAir(rho_w, T=T_1, rh=rh_1, p_tot=p)

{Partial pressure}
p_w_1 = HumidAir(p_w, T=T_1, rh=rh_1, p_tot=p)

{Conditioned air (State 2)}
T_2 = 22
rh_2 = 0.5
h_2 = HumidAir(h, T=T_2, rh=rh_2, p_tot=p)
w_2 = HumidAir(w, T=T_2, rh=rh_2, p_tot=p)

{Cooling load}
m_dot_a = 1000/3600           {1000 kg/h dry air}
Q_dot_cool = m_dot_a*(h_1-h_2)
m_dot_condensate = m_dot_a*(w_1-w_2)
```
