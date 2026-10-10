# HVAC Equation Solver

Equation solver for teaching and rapid calculation of thermodynamic state changes in HVAC systems. Also used for developing benchmark tests for AI agents in building services engineering.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Version](https://img.shields.io/badge/Version-4.0.0-orange.svg)

## Features

- **Intuitive Syntax**: Equations in natural form (`h = enthalpy(water, T=100°C, p=1bar)`)
- **Thermodynamic Properties**: Over 100 fluids via CoolProp
- **Humid Air**: Psychrometric calculations (`h = HumidAir(h, T=25°C, rh=0.5, p_tot=1bar)`)
- **Blackbody Radiation**: Planck's radiation functions (`Eb`, `Blackbody`, `Wien`, `Stefan_Boltzmann`)
- **Robust Solver**: Block decomposition with bracket search and Brent's method
- **Parameter Studies**: Simple sweep syntax (`p = 25:5:50 bar`) and value lists for measured data (`T = [20 25 31] °C`)
- **Case Distinction**: `IF(a, b, x, y, z)` as in EES (`Nu = IF(Re, 2300, Nu_lam, Nu_turb, Nu_turb)`)
- **Initial Values in the Sheet**: saved with the file as block `{$Startwerte ... $}`
- **Optimization**: `MAXIMIZE eps_tot VARY m_dot_gly = 0.1 .. 4 kg/s` - also several quantities and per point of a parameter study (measured data)
- **Unit System**: Automatic unit parsing, propagation and consistency checking
- **GUI**: Modern CustomTkinter interface with plotting capabilities
- **Temperature Display**: Configurable display in °C or K (Settings)

## Installation

The solver runs **locally and offline**; internet is only needed once for installation
and for updates. **Step-by-step instructions for students (German): [INSTALLATION.md](INSTALLATION.md)**

Short version:
1. Install Python 3.12 from <https://www.python.org/downloads/> (Windows: tick "Add python.exe to PATH").
2. Download the latest release (GitHub → Releases → *Source code (zip)*) and unzip it.
3. Install once: double-click `installieren_windows.bat` (Windows) / `installieren_mac.command`
   (macOS), or run `./install.sh` (Linux). This creates a local environment `.venv`
   in the program folder with the packages from `requirements.txt`.
4. Start: `starten_windows.bat` / `starten_mac.command` / `./start.sh`.

Update: replace the program folder with the new release (or `git pull`) and run the
install script again. The current version is shown in the title bar and status bar.

### Required Libraries (for developers)

```bash
pip install -r requirements.txt
```

| Library | Version | Purpose |
|---------|---------|---------|
| numpy | >= 1.24, < 3 | Array operations, mathematical functions |
| scipy | >= 1.10, < 2 | Numerical solvers (least_squares, fsolve, brentq) |
| CoolProp | >= 6.6, < 8 | Thermodynamic property data |
| matplotlib | >= 3.7, < 4 | Diagrams and plots |
| pint | >= 0.23, < 1 | Unit handling and dimensional analysis |
| customtkinter | >= 5.2, < 6 | Modern GUI |
| tkinter | - | GUI (included in Python from python.org) |

### Start the Program (developers)

```bash
python3 main.py
```

## Screenshot

The application provides an intuitive interface for entering equations and displaying results:

- **Left Panel**: Equation input
- **Right Panel**: Solution results
- **Plot Function**: For parameter studies

## Architecture

```
HVAC-Equation-Solver/
├── main.py              # Tkinter GUI (main application)
├── parser.py            # Equation syntax → Python conversion
├── solver.py            # Block decomposition + bracket search solver
├── optimizer.py         # MINIMIZE/MAXIMIZE (grid + bounded Brent/Powell, per point)
├── thermodynamics.py    # CoolProp wrapper with unit conversion
├── humid_air.py         # CoolProp HumidAirProp wrapper
├── radiation.py         # Blackbody radiation functions
├── units.py             # Unit handling and conversion (v3.0)
├── unit_constraints.py  # Unit propagation and consistency checking (v3.0)
├── diagnostics.py       # Generic error analysis (structure, numerics, name hints)
├── version.py           # Program version (shown in title and status bar)
├── requirements.txt     # Required packages
├── install.sh / start.sh                          # Install/start (macOS, Linux)
├── installieren_mac.command / starten_mac.command  # Double-click (macOS)
├── installieren_windows.bat / starten_windows.bat  # Double-click (Windows)
├── INSTALLATION.md      # Installation and update guide for students (German)
├── test_regressions.py  # Regression tests: parser, solver, units
├── test_unit_constraints.py  # Unit propagation / dimension checks
├── test_berechnungen.py # Thermodynamics & heat transfer problems vs. reference values
├── test_optimierung.py  # Optimization vs. analytic/scipy references
├── test_gui.py          # Headless GUI tests
├── CLAUDE.md            # Technical documentation
└── README.md            # This file
```

## Quick Start

**☰ Examples** (or Edit → Insert Example) loads a sheet that shows the main features: water/steam,
humid air, radiation, a heating curve in °C (`value`/`quantity`, temperature spread in K), pipe flow
with `IF` and an economic insulation thickness with `MINIMIZE`. Press F5 to solve.

### Simple Example

```
{Steam at 100°C and 1 bar}
T = 100 °C
p = 1 bar
h = enthalpy(water, T=T, p=p)
s = entropy(water, T=T, p=p)
```

### Mechanics Example

```
{Force calculation: F = m * g}
m = 100 kg
g = 9.81 m/s^2
F = m * g
```

### Humid Air Example

```
{Humid air at 25°C and 50% relative humidity}
T = 25 °C
rh = 0.5
p_tot = 1 bar
h = HumidAir(h, T=T, rh=rh, p_tot=p_tot)
w = HumidAir(w, T=T, rh=rh, p_tot=p_tot)
T_dp = HumidAir(T_dp, T=T, rh=rh, p_tot=p_tot)
```

### Thermal Radiation Example

```
{Stefan-Boltzmann radiation}
T_surface = 500 °C
epsilon = 0.85
A = 2 m^2
sigma = 5.67E-8 W/(m^2*K^4)
Q_rad = epsilon * sigma * A * T_surface^4

{Wien's displacement law}
lambda_max = Wien(T_surface)
```

### System of Equations

```
x + y = 10
x - y = 2
```

A long equation may continue on the next line inside an open bracket when the line
ends with `(`, `,` or an operator:

```
Nu = IF(Re, 2300, 3.66, 3.66,
        0.023*Re^0.8*Pr^0.4)
```

### Parameter Study

```
T = 0:10:100 °C
p = 1 bar
h = enthalpy(water, T=T, p=p)
```

### Measured Data (Value Lists)

```
T_a = [-5.2 -4.8 -3.9 -2.7] °C
m_dot = [0.95
         1.02
         1.00
         0.98] kg/s
T_i = 20 °C
c_p = 1006 J/(kg*K)
Q_dot = m_dot*c_p*(T_i - T_a)
```

Values are separated by spaces, tabs, line breaks, `;` or `,` (decimal point).
A column copied from Excel/CSV/TXT can be pasted between `[` and `]`.
Several lists are combined point by point.

### Case Distinction (IF)

```
Re_krit = 2300
Nu = IF(Re, Re_krit, Nu_lam, Nu_turb, Nu_turb)   {laminar below Re_krit}
```

`IF(a, b, x, y, z)` returns `x` for `a < b`, `y` for `a = b` and `z` for `a > b`
(as in EES; also `if(...)`). It can be nested and used in parameter studies. `a`, `b`
and `x`, `y`, `z` must have the same unit. All arguments are evaluated - the branch
not chosen must be computable too (no division by zero).

### Optimization (MINIMIZE / MAXIMIZE)

```
MAXIMIZE epsilon_tot VARY m_dot_gly = 0.1 .. 4 kg/s
MAXIMIZE e_tot VARY m_dot_w_2 = 0.5 .. 2 kg/s, m_dot_w_4 = 0.5 .. 2 kg/s
MINIMIZE q_L VARY s_L = 11 .. 30 mm
```

The varied quantities are chosen within their bounds so that the goal (a variable that
follows from the equations) becomes minimal or maximal; they are not unknowns, so the
sheet has one equation less per varied quantity. The unit at the end applies to both
bounds. Method: a grid over the whole range (finds the best of several local optima),
then an exact local search (Brent for one quantity, Powell for several, both bounded);
ranges over more than two decades are scanned logarithmically. For every candidate the
normal solver solves the remaining system (warm start). Optima at a bound are reported.
Together with a parameter study or value lists the optimum is found for every point
(like the Min/Max table in EES) - e.g. the optimal glycol mass flow for each operating
hour of a year of measured data.

### Initial Values in the Sheet

For equations with several solutions the solver takes the one closest to the
initial value. *Solve → Initial Values* writes your own values (SI or with unit)
into the sheet as a comment block, which is saved with the file:

```
x^2 = 9
{$Startwerte
x = -3
$}
```

The block can also be edited by hand; deleting it restores the automatic values.
A start value with unit also sets the display unit of a computed quantity (like Variable Info
in EES), e.g. `q_V = 600 kJ/m^3` for an energy per volume that would otherwise be shown as a
pressure (same dimension).

## Steam Power Cycle Example

```
{Live steam}
m_dot = 10000/3600 kg/s
T_1 = 450 °C
p_1 = 30 bar
h_1 = enthalpy(water, p=p_1, T=T_1)
s_1 = entropy(water, p=p_1, T=T_1)

{Turbine with isentropic efficiency}
p_2 = 2.5 bar
eta_s = 0.8
h_2s = enthalpy(water, p=p_2, s=s_1)
eta_s = (h_2-h_1)/(h_2s-h_1)

{Turbine power}
W_dot = m_dot*(h_1-h_2)
```

## Air Conditioning Example

```
{Outdoor air}
T_1 = 35 °C
rh_1 = 0.6
p = 1 bar

h_1 = HumidAir(h, T=T_1, rh=rh_1, p_tot=p)
w_1 = HumidAir(w, T=T_1, rh=rh_1, p_tot=p)

{Conditioned air}
T_2 = 22 °C
rh_2 = 0.5
h_2 = HumidAir(h, T=T_2, rh=rh_2, p_tot=p)
w_2 = HumidAir(w, T=T_2, rh=rh_2, p_tot=p)

{Cooling load}
m_dot_a = 1000/3600 kg/s
Q_dot_cool = m_dot_a*(h_1-h_2)
```

## Solver Strategy

The solver uses robust block decomposition:

1. **Assign Constants**: Explicit definitions like `T_1 = 450`
2. **Direct Evaluation**: Equations of the form `var = expression` are calculated sequentially
3. **Single Unknowns**: Bracket search + Brent's method
4. **Block-wise Solution**: Connected equation blocks with `scipy.fsolve`
5. **Iteration**: Steps 2-4 are repeated until all equations are solved

### Robust Root Finding

- ~4000 test points (including negative values) across magnitudes up to ±5e9
- Adaptive refinement at singularities; poles and underflow plateaus are rejected
- Residuals are evaluated relative to the magnitude of the equation terms
  (divergence to an asymptote is never accepted as a solution)
- With multiple roots, the one closest to the initial value is chosen
  (`sin(alpha) = 0.5` yields 30, not 150)
- Contradictory systems (e.g. `x+1=3` and `x+1=4`) are reported as errors; redundant but
  consistent equations (an extra balance) are allowed and checked, whichever way they are written
- Linearly dependent equations (`x + y = 1`, `2*x + 2*y = 2`, a balance written twice) are
  reported as "Lösung nicht eindeutig" instead of returning an arbitrary point
- Errors of property functions (outside the valid range, supersaturated humid air) and 0/0
  are reported with the original line
- Parameter studies use warm starts (previous point's solution as initial value)
- Initial values from the unit (unknown temperatures at the mean of the given ones) or,
  without units, from the structure: an unknown that stands as a summand next to known
  values (`T_R - T_G1`) starts near them - sheets work with and without units
- Time budget of ~10 s per single equation (unsolvable equations do not freeze the GUI)

**Note:** A line is only treated as a constant assignment if the left-hand side
is a bare variable name. `x + 5 = 2`, `sin(alpha) = 0.5` or `x^2 = 9` are
equations and are solved iteratively. Expression constants with units like
`m_dot = 10000/3600 kg/s` are supported.

## Units

**Important:** All calculations are performed internally in **SI base units**
(K, Pa, J, W). Inputs with other units (`°C`, `bar`, `kJ`) are automatically converted.

| Property | Internal Unit (SI) | Input Examples |
|----------|--------------------|----------------|
| Temperature T | K | `25 °C`, `298.15 K` |
| Temperature difference | delta_K | `dT = 7 K` (always in K; °C is an absolute temperature) |
| Pressure p | Pa | `1 bar`, `101325 Pa` |
| Enthalpy h | J/kg | `100 kJ/kg` |
| Angles (sin, cos, tan) | Degrees (°) | `30 deg`, `0.5236 rad` (→ 30) |
| Entropy s | J/(kg·K) | |
| Density rho | kg/m³ | |
| Vapor quality x | - (0-1) | |

`°C`/`°F` are always absolute temperatures; temperature differences are entered in `K`.
Whether a quantity is a temperature or a difference follows from the equations, never from
its name: `theta = T_1 - T_2` is a difference, `T_2 = T_1 + dT` makes `dT` a difference and
`T_2` absolute, `(T_1 + T_2)/2` is absolute, and a temperature inside a product whose
dimension is not a temperature is a difference (as in COMSOL): in `Q = m*c*(T_1 - T_2)` with
`T_1` absolute, `T_2` is absolute. For the display, the solution is also checked against a
shift of the temperature zero point (absolute temperatures shift, differences do not):
`dT = Q/(m*c)` or `dT = q*R` is a difference, a mixing temperature
`m_3*T_3 = m_1*T_1 + m_2*T_2` with `m_3 = m_1 + m_2` is absolute. Absolute temperatures are
shown in °C (Settings), differences in K. Temperatures are always calculated in Kelvin, without
exception: a sum of absolute temperatures (`T_1 = 20 °C`, `T_2 = 40 °C`, `T_3 = T_1 + T_2`) gives
606.3 K = 333.15 °C and a hint (ⓘ) - neither a temperature nor a difference.

Numeric-value equations (formulas that only hold for numbers in certain units, such as a
heating curve in °C) use `value(x, unit)` and `quantity(z, unit)`:
`T_VL = quantity(20 + 1.5*(20 - value(T_a, °C)), °C)`.

Units of computed variables are inferred in two stages (as described by Olsson 2025 for
Modelica): local propagation (keeps readable units like kW, kJ/kg), then complete
Hindley-Milner inference after Kennedy - all unit equations form one linear system in the
exponents of the SI base units, solved exactly; this also finds coupled units
(`a*b = X`, `a/b = Y`). If units cannot be determined, a hint names the quantities whose
unit has to be given (as a start value with unit) so that all others follow.

### Humid Air Units

| Property | Internal Unit (SI) |
|----------|--------------------|
| Enthalpy h | J/kg_dry_air |
| Humidity ratio w | kg_water/kg_dry_air |
| Relative humidity rh | - (0-1) |
| Dew point T_dp | K (display: °C selectable) |
| Wet bulb T_wb | K (display: °C selectable) |

## Available Functions

### Thermodynamics
`enthalpy`, `entropy`, `density`, `volume`, `intenergy`, `quality`, `temperature`, `pressure`, `viscosity`, `conductivity`, `prandtl`, `cp`, `cv`, `soundspeed`

Reference state of h, u, s per fluid as in EES, as an own line in the sheet:
`REFERENCE R717 IIR` (h = 200 kJ/kg, s = 1 kJ/(kg·K) for saturated liquid at 0 °C),
`ASHRAE` (0 at −40 °C), `NBP` (0 at the normal boiling point) or `DEFAULT` (CoolProp). It applies
to all names of the fluid (R717 = ammonia); differences, T, p, x, ρ are unchanged. CoolProp's
standard is already IIR for R134a, R32, R410A, CO2, propane and most refrigerants, but not for
ammonia (h' at 0 °C = 345.7 kJ/kg) and water (IAPWS).

### Humid Air (HumidAir)
Output: `T`, `h`, `rh`, `w`, `p_w`, `rho_tot`, `rho_a`, `rho_w`, `T_dp`, `T_wb`
Input: `T`, `p_tot`, `rh`, `w`, `p_w`, `h`, `T_dp` (dew point), `T_wb` (wet bulb)

German notation works as well: `x` for the humidity ratio, `phi` for the relative humidity, `p` for
the total pressure, and `v` (specific volume per kg dry air) as output.
A state with more water vapour than saturated air can hold (w > w_s at T, p) is reported
("Zustand übersättigt") instead of returning values - the condensate belongs into the balance.

### Mathematics
`sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, `exp`, `ln`, `log10`, `lg`, `sqrt`, `abs`, `ceil`, `floor`, `round`, `max`, `min`, `IF`, `value`, `quantity`, `pi`

### Radiation
`Eb`, `Blackbody`, `Blackbody_cumulative`, `Wien`, `Stefan_Boltzmann`

Like everything else, radiation quantities are SI internally: wavelengths in m
(`L = 5 µm` → 5e-6 m), `Eb` in W/m³ (displayed as W/(m²·µm)), `Wien` in m
(displayed in µm). Plain numbers are SI like everywhere: `Eb(1000, 5)` and `L = 5` mean
5 m - write `5 µm` (a number as wavelength in a call gives a hint ⓘ).
Units may also be written directly in the arguments: `Eb(500 °C, 5 µm)`.

## Error Analysis

Errors are analysed generically, never by recognising particular equations:
syntax errors are reported per line with their position (`▶`); the structure of the
system is analysed with a Dulmage-Mendelsohn decomposition (which unknowns are
under-determined, which equations over-determine something - even when the total
counts match); unknowns that could not be solved numerically are listed with their
lines; and unknowns that occur in only one equation and look like a typo of existing
names (e.g. `r_1h_i` = `r_1` + `h_i`) are shown as hints.

## Unit System

The solver supports automatic unit handling and propagation:

### Specifying Units

```
T_s = 90 °C
p = 1 bar
sigma = 5.67e-8 W/m^2K^4
L = 4 µm
A = 50 cm^2
V = 200 L
```

Every unit is converted to SI for the calculation (also cm², L, kW/m², kW/(m²K),
mPa·s, mm²/s, µm). Exponents may be written without `^` (`20 cm2`, `500 m3/h`,
`10 W/m2K`), and `·` is accepted as multiplication (`W/(m²·K)`). `%` and `‰` are units
(`eta = 89.2 %` is 0.892, also `rh=50 %` in function arguments). Unknown units are
reported as an error instead of being silently ignored, and units in function
arguments are checked against the expected dimension (`T=` temperature, `p=`
pressure, ...). Numbers without unit are SI values, also in equations: in `24/(24 - t_S)`
with `t_S = 2 h` the 24 is 24 s - a hint (ⓘ) points this out (define `t_d = 24 h`). Python keywords may be used as variable names, e.g.
`lambda = 0.04 W/mK` for a thermal conductivity.

### Automatic Unit Propagation

Units are automatically derived for calculated variables:

```
h = 25 W/m^2K           {heat transfer coefficient}
T_s = 90 °C
T_inf = 20 °C
q_dot = h*(T_s - T_inf)  {automatically gets W/m^2}
```

### Supported Unit Types

| Category | Examples |
|----------|----------|
| Temperature | °C, K, °F |
| Pressure | bar, Pa, kPa, MPa, atm, psi |
| Energy | kJ, J, kWh |
| Power | kW, W |
| Force | N, kN |
| Acceleration | m/s^2 |
| Mass | kg, g |
| Mass flow | kg/s, kg/h |
| Heat flux | W/m^2 |
| Heat transfer coeff. | W/m^2K |
| Thermal resistance | m^2K/W |
| Wavelength | µm, nm, m |
| Stefan-Boltzmann | W/m^2K^4 |

### Dimensional Analysis

The solver automatically:
- Propagates units through algebraic expressions
- Handles exponents (e.g., `T^4` with temperature units)
- Recognizes dimensionless quantities (Nusselt, Grashof, Prandtl numbers)
- Validates unit consistency in equations

### Settings

In the Settings dialog you can configure:
- **Font Size**: Adjust the editor font size
- **Temperature Display**: Choose between Kelvin (K) or Celsius (°C) for result display

## Tests

```bash
python3 test_regressions.py       # parser, solver, unit system
python3 test_unit_constraints.py  # unit propagation and dimension checks
python3 test_berechnungen.py      # 59 problems, with and without units
python3 test_optimierung.py       # MINIMIZE/MAXIMIZE against analytic/scipy references
python3 test_gui.py               # headless GUI (briefly takes focus - don't type)
```

`test_regressions.py` covers the parser (assignment detection, vectors, unit sweeps,
keywords, comments), the solver (root selection, contradiction detection, parameter
studies, tearing, determinism) and the unit system (SI conversion, start values,
delta_K, offset conversions, radiation). `test_berechnungen.py` solves thermodynamics
and heat-transfer problems both with and without units and compares them with
independently computed reference values (CoolProp, analytical solutions).
Each module also has a self-test: `python3 <module>.py`.

## License

MIT License

## Contributions

Contributions are welcome! Please create a pull request or open an issue.
