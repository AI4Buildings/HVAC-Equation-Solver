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

## Architektur

```
equation_solver/
├── main.py              # Tkinter GUI (Hauptanwendung)
├── parser.py            # Equation syntax → Python conversion
├── solver.py            # Block-Dekomposition + Bracket-Suche Solver
├── thermodynamics.py    # CoolProp Wrapper mit Einheitenumrechnung
├── humid_air.py         # CoolProp HumidAirProp Wrapper für feuchte Luft
├── radiation.py         # Schwarzkörper-Strahlungsfunktionen (Planck, vektorisiert)
├── units.py             # Einheitenhandling und Konvertierung (v3.0)
├── unit_constraints.py  # Einheiten-Propagation und Konsistenzprüfung (v3.0)
```

## Kernfunktionen

### Parser (parser.py)
- Converts equation syntax to Python: `^` → `**`, `ln` → `log`
- Comments: `"..."` and `{...}`
- Thermodynamic function calls: `enthalpy(water, T=100, p=1)` → `enthalpy('water', T=100, p=1)`
- Extracts variables from equations (filters function names and parameter keys)
- Vector syntax: `T = 0:10:100` (start:step:end) or `T = 0:100` (start:end, step=1)
- **Direct assignments** like `T_1 = 450` or `m = 10000/3600` are treated as constants

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
- **Standard-Startwert**: 1.0 für alle Variablen

#### Parameterstudien
- Sweep-Variablen werden als Konstanten für jeden Punkt behandelt
- Für jeden Sweep-Punkt wird `solve_system` mit Block-Dekomposition aufgerufen
- Vektorisierte Auswertung für direkte Funktionen ohne Iteration

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
- Case-insensitive: `HumidAir` = `humidair`

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
  - `Eb(T, lambda)` - Spektrale Emissionsleistung [W/(m²·µm)]
  - `Blackbody(T, lambda1, lambda2)` - Anteil der Energie im Wellenlängenbereich [-]
  - `Blackbody_cumulative(T, lambda)` - Kumulativer Anteil von 0 bis λ [-]
  - `Wien(T)` - Wellenlänge maximaler Emission [µm]
  - `Stefan_Boltzmann(T)` - Gesamtemission [W/m²]
- Einheiten: T in K (intern), λ in µm
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
| Wellenlänge λ | µm | `5 µm`, `5e-6 m` |
| Spektrale Emission Eb | W/(m²·µm) | |
| Gesamtemission E | W/m² | |
| **Winkel (Trigonometrie)** | **Grad (°)** | |

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

## Einheiten-System (v3.0)

### Units Module (units.py)
- Einheiten-Parsing und Konvertierung basierend auf `pint`
- `UnitValue`-Klasse speichert SI-Wert und Original-Einheit
- Automatische Konvertierung zu Standard-Einheiten für Berechnungen
- Unterstützte Einheiten: °C, K, bar, Pa, kJ, W, kg/s, m²/s, W/m²K, µm, etc.

### Unit Constraints Module (unit_constraints.py)
- **Einheiten-Propagation**: Leitet Einheiten für berechnete Variablen ab
- **Dimensionsanalyse**: Verwendet AST-Parsing für algebraische Ausdrücke
- **Rückwärts-Propagation**: Bei `q = h*dT` wird `h = q/dT` abgeleitet
- **Konsistenzprüfung**: Warnt bei inkonsistenten Einheiten
- **Additive Ketten**: Bei `A + B - C = 0` erhalten alle Terme die gleiche Dimension

**Wichtige Funktionen:**
- `propagate_all_units()` / `propagate_all_units_complete()`: Hauptfunktionen für Einheiten-Propagation
- `analyze_equation()`: Analysiert eine Gleichung und leitet Einheiten ab
- `_infer_from_additive_chain()`: Propagiert Dimensionen in Addition/Subtraktion-Ketten
- `_collect_additive_terms()`: Sammelt alle Terme aus +/- Ketten (ignoriert numerische Konstanten)
- `_infer_from_mult_div()`: Rückwärts-Inferenz für Multiplikation/Division
- `is_temperature_difference_variable()`: Erkennt Temperaturdifferenz-Variablen (dT..., delta...)
- `adjust_unit_for_variable()`: Passt Einheit basierend auf Variablenname an (K → delta_K)

### Einheiten-Syntax

```
T_s = 90 °C              {Temperatur}
p = 1 bar                {Druck}
sigma = 5.67e-8 W/m^2K^4 {Stefan-Boltzmann}
L = 4 µm                 {Wellenlänge}
h = 25 W/m^2K            {Wärmeübergangskoeffizient}
```

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
| Temperaturdifferenz | delta_K (für Variablen wie dT..., delta...) |
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

### Temperaturdifferenzen

Variablen deren Name mit `dT` oder `delta` beginnt werden automatisch als
Temperaturdifferenzen erkannt und erhalten die Einheit `delta_K`.

```
dT_N = 49.83K            {→ Erkannt als delta_K}
delta_T = 10K            {→ Erkannt als delta_K}
dT_log = (T1-T2)/ln(...) {→ Abgeleitet als delta_K}
```

Dies vermeidet falsche Offset-Konvertierungen (K → °C würde -273.15 subtrahieren).

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
f_visible = Blackbody(T_surface, 0.38, 0.75)
```
