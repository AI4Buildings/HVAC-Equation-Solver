# HVAC Equation Solver

Equation solver for teaching and rapid calculation of thermodynamic state changes in HVAC systems. Also used for developing benchmark tests for AI agents in building services engineering.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Version](https://img.shields.io/badge/Version-4.1.1-orange.svg)

## Features

- **Intuitive Syntax**: Equations in natural form (`h = enthalpy(water, T=100°C, p=1bar)`)
- **Thermodynamic Properties**: 124 fluids via CoolProp, selectable reference states
  (`REFERENCE R717 IIR`)
- **Chemical Equilibrium**: statement `EQUILIBRIUM` - Gibbs minimisation as in NASA CEA, no
  reactions to choose, amounts always >= 0, soot (C(gr)) and condensate (H2O(L)) appear when stable;
  works with unknown temperature (adiabatic reformer) without start values
- **Ideal Gases and Condensed Species**: about 2000 species from the NASA database (thermo.inp)
  with prefix `IG:` for combustion and fuel cell calculations (`enthalpy(IG:N2, T=20 °C)`,
  `gibbs(IG:H2O(L), T=25 °C, p=1 bar)`, `molarmass(IG:CH4)`)
- **Humid Air**: Psychrometric calculations (`h = HumidAir(h, T=25°C, rh=0.5, p_tot=1bar)`)
- **Blackbody Radiation**: Planck's radiation functions (`Eb`, `Blackbody`, `Wien`, `Stefan_Boltzmann`)
- **Robust Solver**: Block decomposition, tearing, bracket search with Brent's method and
  Levenberg-Marquardt for coupled blocks; contradictory and non-unique systems are reported
- **Parameter Studies**: Simple sweep syntax (`p = 25:5:50 bar`) and value lists for measured data (`T = [20 25 31] °C`)
- **Case Distinction**: `IF(a, b, x, y, z)` (`Nu = IF(Re, 2300, Nu_lam, Nu_turb, Nu_turb)`)
- **Initial Values in the Sheet**: saved with the file as block `{$Startwerte ... $}`
- **Optimization**: `MAXIMIZE eps_tot VARY m_dot_gly = 0.1 .. 4 kg/s` - also several quantities and per point of a parameter study (measured data)
- **Unit System**: Automatic unit parsing, propagation and consistency checking; calculations
  always in SI and Kelvin
- **GUI**: Modern CustomTkinter interface with plotting capabilities; display units selectable
  per quantity and in the Settings (°C/K, bar/Pa, kJ/J, kW/W)
- **Language**: menus in English, messages and hints in German (teaching use at a German-speaking
  university)

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

## User Interface

- **Left panel**: equation sheet (syntax as below, comments in `{...}` or `"..."`)
- **Right panel**, tab *Results*: values with a selectable display unit per quantity, status line
  and messages; tab *Residuals*: equations, blocks and residuals, unit warnings (⚠) and hints (ⓘ)
- **Menus**: File (`.hes`, `.txt`), Edit (Insert Example), View (font size 6-36 pt), Solve (F5,
  Initial Values), Plot (plot windows for parameter studies, with zoom/pan/save), Settings
  (display units), Help (function reference, fluid list with all CoolProp and `IG:` species)

## Architecture

```
HVAC-Equation-Solver/
├── main.py              # Tkinter GUI (main application)
├── parser.py            # Equation syntax → Python conversion
├── solver.py            # Block decomposition + bracket search solver
├── optimizer.py         # MINIMIZE/MAXIMIZE (grid + bounded Brent/Powell, per point)
├── thermodynamics.py    # CoolProp wrapper with unit conversion
├── ideal_gas.py         # Ideal gases / condensed species from NASA thermo.inp (prefix IG:)
├── equilibrium.py       # Chemical equilibrium by Gibbs minimisation (statement EQUILIBRIUM)
├── data/nasa/           # thermo.inp, LICENSE.txt, NOTICE.txt (NASA CEA, Apache 2.0)
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
├── README.md            # This file
└── LICENSE              # MIT (data/nasa/* excluded: Apache 2.0)
```

## Quick Start

**☰ Examples** (or Edit → Insert Example) loads a sheet that shows the main features: water/steam,
humid air, radiation, a heating curve in °C (`value`/`quantity`, temperature spread in K), pipe flow
with `IF`, an economic insulation thickness with `MINIMIZE`, ammonia with `REFERENCE R717 IIR` and
ideal gases (heating value of methane, fuel cell voltage, steam reformer with `EQUILIBRIUM`).
Press F5 to solve.

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
(also written `if(...)`). It can be nested and used in parameter studies. `a`, `b`
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
Together with a parameter study or value lists the optimum is found for every point,
e.g. the optimal glycol mass flow for each operating hour of a year of measured data.

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
A start value with unit also sets the display unit of a computed quantity, e.g.
`q_V = 600 kJ/m^3` for an energy per volume that would otherwise be shown as a
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

## Combustion and Fuel Cell Example (ideal gases, `IG:`)

```
{Heating value of methane: CH4 + 2 O2 -> CO2 + 2 H2O}
T_0 = 25 °C
M_CH4 = molarmass(IG:CH4)                          {16.04 kg/kmol}
h_CH4 = enthalpy(IG:CH4, T=T_0)*M_CH4              {molar: -74.6 kJ/mol}
h_O2 = enthalpy(IG:O2, T=T_0)*molarmass(IG:O2)
h_CO2 = enthalpy(IG:CO2, T=T_0)*molarmass(IG:CO2)
h_H2O = enthalpy(IG:H2O, T=T_0)*molarmass(IG:H2O)
H_u = -(h_CO2 + 2*h_H2O - h_CH4 - 2*h_O2)/M_CH4    {50.03 MJ/kg}

{Adiabatic flame temperature with excess air ratio lambda (implicit)}
lambda = 1.2
n_O2 = 2*lambda
n_N2 = 3.76*n_O2
H_R = h_CH4 + n_O2*h_O2 + n_N2*enthalpy(IG:N2, T=T_0)*molarmass(IG:N2)
H_R = (enthalpy(IG:CO2, T=T_ad)*molarmass(IG:CO2) + 2*enthalpy(IG:H2O, T=T_ad)*molarmass(IG:H2O)
      + (n_O2 - 2)*enthalpy(IG:O2, T=T_ad)*molarmass(IG:O2) + n_N2*enthalpy(IG:N2, T=T_ad)*molarmass(IG:N2))

{Fuel cell H2 + 1/2 O2 -> H2O(L): reversible cell voltage}
p_0 = 1 bar
F = 96485.33212 C/mol
g_H2O = gibbs(IG:H2O(L), T=T_0, p=p_0)*molarmass(IG:H2O(L))
g_H2 = gibbs(IG:H2, T=T_0, p=p_0)*molarmass(IG:H2)
g_O2 = gibbs(IG:O2, T=T_0, p=p_0)*molarmass(IG:O2)
E_0 = -(g_H2O - g_H2 - 0.5*g_O2)/(2*F)             {1.229 V}
```

Results: `H_u` = 50.03 MJ/kg, `T_ad` = 1796 °C, `E_0` = 1.229 V (gaseous product water at
25 °C: 1.185 V, at 800 °C: 0.977 V). With the statement `EQUILIBRIUM` (see below) the equilibrium
composition with dissociation (OH, H, O, NO, ...) follows without choosing reactions.

## Simple Examples: Combustion, Chemical Equilibrium, Fuel Cells

All examples use the NASA data (prefix `IG:`, molar values via `molarmass`) and solve without start
values. `EQUILIBRIUM` minimises the Gibbs energy of the species listed under `OUT` - see
[Chemical Equilibrium](#chemical-equilibrium-equilibrium).

### Combustion: Adiabatic Flame Temperature with and without Dissociation

```
{Methane with air (lambda = 1.2), adiabatic: complete combustion vs. chemical equilibrium}
T_1 = 25 °C
p = 1 bar
lambda = 1.2
n_CH4 = 1 mol
n_O2 = 2*lambda*n_CH4
n_N2 = 3.76*n_O2
H_1 = (n_CH4*enthalpy(IG:CH4, T=T_1)*molarmass(IG:CH4) + n_O2*enthalpy(IG:O2, T=T_1)*molarmass(IG:O2)
       + n_N2*enthalpy(IG:N2, T=T_1)*molarmass(IG:N2))

{a) complete combustion CH4 + 2 O2 -> CO2 + 2 H2O: adiabatic flame temperature T_a}
H_1 = (n_CH4*enthalpy(IG:CO2, T=T_a)*molarmass(IG:CO2) + 2*n_CH4*enthalpy(IG:H2O, T=T_a)*molarmass(IG:H2O)
       + (n_O2 - 2*n_CH4)*enthalpy(IG:O2, T=T_a)*molarmass(IG:O2) + n_N2*enthalpy(IG:N2, T=T_a)*molarmass(IG:N2))

{b) chemical equilibrium with dissociation (CO, H2, OH, H, O, NO) - no reactions to choose}
EQUILIBRIUM T=T_b, p=p
  IN  IG:CH4 = n_CH4, IG:O2 = n_O2, IG:N2 = n_N2
  OUT IG:CO2 = b_CO2, IG:H2O = b_H2O, IG:O2 = b_O2, IG:N2 = b_N2, IG:CO = b_CO,
      IG:H2 = b_H2, IG:OH = b_OH, IG:H = b_H, IG:O = b_O, IG:NO = b_NO
H_1 = (b_CO2*enthalpy(IG:CO2, T=T_b)*molarmass(IG:CO2) + b_H2O*enthalpy(IG:H2O, T=T_b)*molarmass(IG:H2O)
       + b_O2*enthalpy(IG:O2, T=T_b)*molarmass(IG:O2) + b_N2*enthalpy(IG:N2, T=T_b)*molarmass(IG:N2)
       + b_CO*enthalpy(IG:CO, T=T_b)*molarmass(IG:CO) + b_H2*enthalpy(IG:H2, T=T_b)*molarmass(IG:H2)
       + b_OH*enthalpy(IG:OH, T=T_b)*molarmass(IG:OH) + b_H*enthalpy(IG:H, T=T_b)*molarmass(IG:H)
       + b_O*enthalpy(IG:O, T=T_b)*molarmass(IG:O) + b_NO*enthalpy(IG:NO, T=T_b)*molarmass(IG:NO))
x_NO = b_NO/(b_CO2 + b_H2O + b_O2 + b_N2 + b_CO + b_H2 + b_OH + b_H + b_O + b_NO)
```

Results: `T_a` = 1795.8 °C (complete combustion), `T_b` = 1770.4 °C (equilibrium: dissociation takes
up energy), `b_CO` = 0.0107 mol, `b_OH` = 0.0276 mol, NO in equilibrium `x_NO` = 0.31 % (3150 ppm - the
thermal NO limit; real flue gas stays far below because the reaction is slow).

### Gibbs Minimisation with Solid Carbon (Soot)

```
{Steam reforming of methane at 800 °C: when does solid carbon form? Study over steam/carbon S/C}
T = 800 °C
p = 1 bar
n_CH4 = 1 mol
SC = [0.25 0.5 0.75 1 1.25 1.5]
n_H2O = SC*n_CH4
EQUILIBRIUM T=T, p=p
  IN  IG:CH4 = n_CH4, IG:H2O = n_H2O
  OUT IG:CH4 = a_CH4, IG:H2O = a_H2O, IG:CO = a_CO, IG:CO2 = a_CO2, IG:H2 = a_H2, IG:C(gr) = a_C
```

| S/C | 0.25 | 0.5 | 0.75 | 1.0 | 1.25 | 1.5 |
|---|---|---|---|---|---|---|
| `a_C` (graphite) in mol | 0.695 | 0.470 | 0.244 | 0.018 | 0 | 0 |
| `a_H2` in mol | 2.061 | 2.286 | 2.512 | 2.737 | 2.960 | 3.065 |

Graphite appears only where it is stable (active set): above S/C ≈ 1 the soot vanishes. A solid or
liquid species is simply listed under `OUT` - no reaction such as 2 CO -> C + CO2 has to be written.

### Condensate: Flue Gas of a Condensing Boiler

```
{Flue gas of methane (lambda = 1.2) cooled to 40 °C - how much water condenses?}
p = 1 bar
T_FG = 40 °C
lambda = 1.2
n_CH4 = 1 mol
EQUILIBRIUM T=T_FG, p=p
  IN  IG:CH4 = n_CH4, IG:O2 = 2*lambda*n_CH4, IG:N2 = 3.76*2*lambda*n_CH4
  OUT IG:CO2 = n_CO2, IG:H2O = n_vap, IG:H2O(L) = n_cond, IG:O2 = n_O2, IG:N2 = n_N2
share_cond = n_cond/(n_vap + n_cond)              {condensed share of the water}
x_H2O = n_vap/(n_CO2 + n_vap + n_O2 + n_N2)       {vapour fraction = p_s(40 °C)/p}
```

Results: `n_cond` = 1.171 mol of 2 mol water condense (`share_cond` = 58.5 %), the remaining vapour
fraction `x_H2O` = 0.0737 corresponds to the saturation pressure 7.37 kPa at 40 °C (IAPWS: 7.38 kPa).
The liquid phase appears only below the dew point; it is pressure dependent because it follows from
the Gibbs minimisation (vapour and liquid have the same chemical potential at saturation).

### PEM Fuel Cell: Nernst Voltage

```
{PEM fuel cell at 80 °C: hydrogen and air at 1.5 bar, liquid product water}
T = 80 °C
F = 96485.33212 C/mol
p_H2 = 1.5 bar                 {anode, partial pressure of H2}
p_O2 = 0.21*1.5 bar            {cathode air at 1.5 bar}
g_H2 = gibbs(IG:H2, T=T, p=p_H2)*molarmass(IG:H2)
g_O2 = gibbs(IG:O2, T=T, p=p_O2)*molarmass(IG:O2)
g_H2O = gibbs(IG:H2O(L), T=T)*molarmass(IG:H2O(L))
U_N = -(g_H2O - g_H2 - 0.5*g_O2)/(2*F)
```

Result: `U_N` = 1.1805 V (molar Gibbs energies `g_H2` = -45.10, `g_O2` = -75.98, `g_H2O` =
-310.89 kJ/mol; gases with their partial pressure, the liquid without pressure effect).

### SOFC: Nernst Voltage - Two Ways

```
{SOFC at 800 °C: Nernst voltage (a) from the reaction H2 + 1/2 O2 -> H2O, (b) from the oxygen
 partial pressures of cathode and anode - the anode p_O2 follows from the Gibbs minimisation}
T = 800 °C
p = 1 bar
F = 96485.33212 C/mol
R = 8.314462618 J/(mol*K)
n_H2 = 0.9 mol                 {anode gas: 90 % H2, 10 % H2O}
n_H2O = 0.1 mol
x_O2_c = 0.21                  {cathode air}
x_H2 = n_H2/(n_H2 + n_H2O)
g_H2 = gibbs(IG:H2, T=T, p=x_H2*p)*molarmass(IG:H2)
g_H2O = gibbs(IG:H2O, T=T, p=(1 - x_H2)*p)*molarmass(IG:H2O)
g_O2 = gibbs(IG:O2, T=T, p=x_O2_c*p)*molarmass(IG:O2)
U_a = -(g_H2O - g_H2 - 0.5*g_O2)/(2*F)
EQUILIBRIUM T=T, p=p
  IN  IG:H2 = n_H2, IG:H2O = n_H2O
  OUT IG:H2 = a_H2, IG:H2O = a_H2O, IG:O2 = a_O2
p_O2_a = a_O2/(a_H2 + a_H2O + a_O2)*p
U_b = R*T/(4*F)*ln(x_O2_c*p/p_O2_a)
```

Results: `U_a` = `U_b` = 1.0425 V. The oxygen partial pressure in the anode gas is only
`p_O2_a` = 5.5e-21 bar - the Gibbs minimisation resolves such traces, and the oxygen concentration
cell (b) gives exactly the voltage of the reaction (a).

## Solver Strategy

The solver uses robust block decomposition:

1. **Assign Constants**: Explicit definitions like `T_1 = 450`
2. **Direct Evaluation**: Equations of the form `var = expression` are calculated sequentially
3. **Single Unknowns**: Bracket search + Brent's method
4. **Block-wise Solution**: coupled equations are split into their smallest independent cores
   (block triangular form) and solved core by core:
   - **tearing**: one estimated variable, all others follow one after another (e.g. a film
     temperature iteration) - solved by the 1-D bracket search; also several tearing variables
     (radiation networks) with `least_squares`
   - otherwise simultaneously with `least_squares` (Levenberg-Marquardt), `fsolve` as fallback
   - **over-determined blocks** (a redundant balance): a square subsystem is solved and the
     remaining equations are checked - consistent, or "Widersprüchliches System" with the violated
     equation
5. **Iteration**: Steps 2-4 are repeated until all equations are solved

### Robust Root Finding

- Test points (including negative values) across magnitudes from 1e-30 to ±1e30
- Adaptive refinement at singularities; poles and underflow plateaus are rejected
- Scale-independent: residuals are evaluated relative to the magnitude of the equation terms
  (divergence to an asymptote is never accepted as a solution), steps and tolerances relative
  to the value - small SI values such as a fibre cross-section of 3.14e-12 m² or a trace
  fraction of 1e-16 are solved as accurately as large ones
- An equation that is only approximately satisfied counts only if a small relative change of
  the variables could satisfy it - a flat region of min/max is never reported as a solution
- Coupled blocks are split exactly into their smallest independent cores (block triangular
  form); solved cores stay in the partial solution if the time limit is reached
- With multiple roots, the one closest to the initial value is chosen
  (`sin(alpha) = 0.5` yields 30, not 150); for a coupled block solved by tearing, the distance of
  all block variables to their initial values counts, so the solution does not depend on how the
  equations are written (`u = v^2` with `v = u - 6` or `u = v + 6` both give (4; -2))
- Contradictory systems (e.g. `x+1=3` and `x+1=4`) are reported as errors; redundant but
  consistent equations (an extra balance) are allowed and checked, whichever way they are written
  and wherever they stand in the sheet
- Linearly dependent equations (`x + y = 1`, `2*x + 2*y = 2`, a balance written twice) and
  equations that hold on a whole range (a flat `max`/`min`, a condensate amount 0 above the dew
  point) are reported as "Lösung nicht eindeutig" instead of returning an arbitrary point; in an
  over-determined block such a dependent loop becomes the check equation
- A sign change of a continuous function, pinned down to machine precision, is a root even if
  rounding in the function (e.g. a small condensate amount as the difference of large amounts)
  keeps the residual above the tolerance; poles and jumps (`IF`, `floor`) are not
- Errors of property functions (outside the valid range, supersaturated humid air) and 0/0
  are reported with the original line
- Parameter studies use warm starts (previous point's solution as initial value)
- Initial values from the unit (unknown temperatures at the mean of the given ones) or,
  without units, from the structure: an unknown that stands as a summand next to known
  values (`T_R - T_G1`) starts near them - sheets work with and without units
- Time budget of ~10 s per single equation and 60 s per solving run (per point of a parameter
  study); after the limit the partial solution is shown with the open unknowns

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
| Angle | rad (as SI) | `30 deg` (→ 0.5236), `10 rad/s`, `15 deg/h` |
| Entropy s | J/(kg·K) | |
| Density rho | kg/m³ | |
| Vapor quality x | - (0-1) | |
| Amount of substance | mol, mol/s | `n = 0.5 mol`, `n_dot = 3.6 kmol/h` |
| Molar mass M | kg/mol (display kg/kmol) | `28.97 kg/kmol`, `molarmass(IG:CH4)` |
| Molar enthalpy, Gibbs energy | J/mol (display kJ/mol) | `-241.8 kJ/mol` |
| Molar entropy | J/(mol·K) | |
| Cell voltage, current, resistance | V, A, Ω | `F = 96485.33212 C/mol`, `ASR = 0.5 Ω*cm^2` |
| Ratios | - | `eta = 89.2 %` (→ 0.892), `5 ‰`, `10 ppm` |

**Angles:** a number without unit in `sin`/`cos`/`tan` is in degrees (`sin(30) = 0.5`), and
`asin`/`acos`/`atan` return degrees. Angles with a unit are SI internally (rad), like
every quantity: with `alpha = 30 deg` and `omega = 10 rad/s`, `sin(alpha)`, `sin(omega*t)`,
`s = alpha*r` and `u = omega*r` are correct; `asin`/`acos`/`atan` return the angle when the
result is used as an angle with a unit (`k = asin(x) + alpha`). Which expressions are angles follows
from the units, never from names. A number in a sum with an angle (`90 - alpha`) is a value in rad
(hint ⓘ) - write `c = 90 deg`. Computed angles are shown in rad (switchable to deg).

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
exception. A sum of absolute temperatures (`T_1 = 20 °C`, `T_2 = 40 °C`, `T_3 = T_1 + T_2`) is
neither a temperature nor a difference - its value would depend on the zero point of the scale
(60 °C as Celsius numbers, 333.15 °C as Kelvin values) - and is reported as an error together
with the unambiguous forms `value(T_3, °C) = value(T_1, °C) + value(T_2, °C)` and
`value(T_3, K) = value(T_1, K) + value(T_2, K)`. A temperature with a factor
(`T_2 = T_1*(p_2/p_1)^0.286`, `2*T_1`) is a law in Kelvin.

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
| Humidity ratio w (or x) | kg_water/kg_dry_air (display kg/kg or g/kg) |
| Relative humidity rh (or phi) | - (0-1), or with unit `50 %` |
| Partial pressure p_w, total pressure p_tot | Pa |
| Densities rho_tot, rho_a, rho_w | kg/m³ |
| Specific volume v | m³/kg_dry_air |
| Dew point T_dp, wet bulb T_wb | K (display: °C selectable) |

## Available Functions

### Thermodynamics
`enthalpy`, `entropy`, `density`, `volume`, `intenergy`, `quality`, `temperature`, `pressure`, `viscosity`, `conductivity`, `prandtl`, `cp`, `cv`, `soundspeed`, `gibbs` (g = h − T·s), `molarmass` (kg/mol, shown in kg/kmol; `molarmass(water)` also for CoolProp fluids)

Reference state of h, u, s per fluid, as an own line in the sheet:
`REFERENCE R717 IIR` (h = 200 kJ/kg, s = 1 kJ/(kg·K) for saturated liquid at 0 °C),
`ASHRAE` (0 at −40 °C), `NBP` (0 at the normal boiling point) or `DEFAULT` (CoolProp). It applies
to all names of the fluid (R717 = ammonia); differences, T, p, x, ρ are unchanged. CoolProp's
standard is already IIR for R134a, R32, R410A, CO2, propane and most refrigerants, but not for
ammonia (h' at 0 °C = 345.7 kJ/kg) and water (IAPWS).

### Ideal Gases and Condensed Species (`IG:`)
With the prefix `IG:` (also `IG::`, not case-sensitive) the property functions use the NASA
database thermo.inp instead of CoolProp; without the prefix nothing changes. About 2000 species
(1276 gases, 769 condensed): `IG:N2`, `IG:O2`, `IG:CO2`, `IG:H2O`, `IG:CH4`, `IG:H2`, `IG:CO`,
`IG:NO`, `IG:Air`, `IG:H2O(L)` (liquid water), `IG:H2O(cr)` (ice), `IG:C(gr)` (graphite),
`IG:C8H18,isooctane`, ... (Help > Fluid List, with the data range of every species).

- Values per kg in SI like CoolProp (J/kg, J/(kg·K), m³/kg). `h` includes the enthalpy of
  formation (h = Δh_f at 25 °C, elements in their reference state 0), `s` is the absolute
  entropy - reaction, combustion and fuel cell balances need no reference point.
  `REFERENCE` does not apply to `IG:` species.
- Gases (ideal): `enthalpy`, `intenergy`, `cp`, `cv`, `soundspeed` need only `T` (`p` allowed,
  without effect); `entropy`, `gibbs`, `volume`, `density` need `T` and `p` - there is no silent
  1 bar: s = s°(T) − R·ln(p/p°), p° = 1 bar, v = R·T/p. `temperature` from `h`, `u` or `s` and
  `p`; `pressure` from `T` and `s` or `rho`.
- Condensed species (`(L)`, `(cr)`, ...): `enthalpy`, `entropy`, `gibbs`, `cp` from `T`, without
  pressure effect; `volume`, `density`, `cv` are reported (thermo.inp has no density).
- Molar values: multiply by `molarmass` (`h*M` in J/mol, shown in kJ/mol).
- A temperature outside the data range of a species is an error with the range (most gases
  200 or 300 ... 6000 K or 20000 K, `IG:Air` from 300 K). Names: exact, otherwise not
  case-sensitive if unique; `IG:co` is ambiguous (CO or Co) and reported with suggestions.

```
h = enthalpy(IG:N2, T=20 °C)               {-5.20 kJ/kg (with enthalpy of formation 0)}
s = entropy(IG:N2, T=300 °C, p=10 bar)     {6.842 kJ/(kg K), absolute}
T_2 = temperature(IG:N2, s=s, p=1 bar)     {isentropic expansion: 25.5 °C}
g = gibbs(IG:H2O(L), T=25 °C, p=1 bar)     {-17.02 MJ/kg}
```

### Chemical Equilibrium (`EQUILIBRIUM`)
An own statement in the sheet (like `MINIMIZE`); `IN` and `OUT` may stand in their own lines, a
list continues after `,` in the next line (comments and blank lines in between are allowed):

```
EQUILIBRIUM T=T_2, p=p
  IN  IG:CH4 = n_CH4_1, IG:O2 = n_O2_1, IG:N2 = n_N2_1
  OUT IG:CH4 = n_CH4, IG:H2O = n_H2O, IG:CO = n_CO, IG:CO2 = n_CO2, IG:H2 = n_H2, IG:N2 = n_N2
```

- `OUT` receives the composition with minimum Gibbs energy at `T` and `p` under the element
  balances of `IN` (NASA data, ideal gas mixture, p° = 1 bar): no reactions to choose, the minimum
  is unique (convex problem), all amounts are >= 0 and traces far below 1e-20 are resolved (for an
  exactly stoichiometric mixture they are limited by the resolution of the element balance,
  ~1e-16 of the main amounts).
- Method as NASA CEA (Gordon/McBride, RP-1311): element potentials (Lagrange multipliers) with
  logarithmic amounts, globally convergent; condensed species (`IG:C(gr)` soot, `IG:H2O(L)`
  condensate) by an active set - they appear only when stable and only inside their data range
  (`IG:H2O(cr)` up to 0 °C, `IG:H2O(L)` 0 ... 327 °C), as in NASA CEA. Checked against Cantera with
  the same data (800 random cases) and by the optimality conditions (|mu_i/RT - a_i*lambda| < 1e-11).
- Amounts in mol or mol/s (`OUT` in the unit of `IN`), also as numbers with unit
  (`IG:N2 = 3.76 mol`); `T` and `p` as variables, expressions or numbers with unit. The same species
  twice under `IN` (two inlet streams) is summed.
- The gas phase may vanish (stoichiometric H2/O2 below the boiling temperature at the given pressure
  gives only `IG:H2O(L)` - the result of the Gibbs minimisation);
- Species outside their data range are left out (amount 0): condensed ones as in NASA CEA with a hint
  ⓘ suggesting the phase of the same composition (`IG:H2O(L)` at -10 °C -> `IG:H2O(cr)`), gases (e.g.
  `IG:NO2`, data from 300 K) with a warning - harmless for a trace, wrong for a main product;
  an element balance that the `OUT` species cannot close is reported at once ("Überschuss an O ...
  e.g. IG:O2").
- Each `OUT` amount is an ordinary quantity of the sheet: `T` may be unknown, e.g. an adiabatic
  reformer with an energy balance `H_in = sum n_i*h_i(T_2)` - solved without start values; parameter
  studies over `T`, `p` or inlet amounts work as usual.
- Elements of `IN` that occur in no `OUT` species, amounts in kg, a missing `p`, a decimal comma,
  `;` instead of `,` etc. are reported with the line; a variable in kg/s as amount gives a unit
  warning ("Stoffmenge erwartet"). Messages name the statement: `EQUILIBRIUM (Zeile 5): IG:H2O = n_D`.
- Dew point: a condensate amount `n_L = 0` holds for every temperature above the dew point - the
  solver reports "Lösung nicht eindeutig" instead of an arbitrary T; prescribe a small amount
  (`n_L = 1e-6*n_W`) to find the dew point.

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

Errors are analysed generically, never by recognising particular equations (messages are in
German):

- syntax errors per line with their position (`▶`), e.g. a decimal comma, a non-ASCII name
  (`η`), a continuation line outside an open bracket
- the structure of the system with a Dulmage-Mendelsohn decomposition: which unknowns are
  under-determined, which equations over-determine something - even when the total counts match
- unknowns that could not be solved numerically, with their lines and the evaluation error of a
  property function (outside the data range, supersaturated humid air, 0/0); the same cause is
  named once
- contradictory systems ("Widersprüchliches System" with the given and the computed value) and
  non-unique solutions ("Lösung nicht eindeutig")
- hints (ⓘ, the calculation is unchanged): names that look like a typo of existing names
  (`r_1h_i` = `r_1` + `h_i`), a number without unit next to a non-SI input (`24 - t_S` with
  `t_S` in h), `T + 273.15` (T is already in K), a wavelength without unit, `quality = -1`
  outside the two-phase region, a species left out of `EQUILIBRIUM` (outside its data range)

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
`10 W/m2K`, `5.67e-8 W/m2K4`) or as superscripts (`W/(m²·K⁴)`, `0.3 h⁻¹`), and `·` is accepted as
multiplication (`W/(m²·K)`). `%` and `‰` are units
(`eta = 89.2 %` is 0.892, also `rh=50 %` in function arguments). Prefixed units in a
denominator stay prefixes (`kg/kWh`, `1/kPa`, `kg/MJ`), °C in a compound unit is a kelvin-sized
difference (`2 °C/h`, `1/°C`, `W/(m²·°C)`), and `rpm` / `U/min` mean revolutions per minute
(1/min, as in u = π·D·n). Resistances may be written `ohm`, `Ω`, `Ohm`, `kΩ` (e.g. an area
specific resistance `ASR = 0.5 Ω*cm²`). Assigning the same quantity twice (`x = 5`, later `x = 3`) is reported
as a contradiction instead of silently using the last value. Unknown units are
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

Any unit known to [pint](https://pint.readthedocs.io) can be used and is converted to SI; the
table lists the usual ones (all checked). Exponents may be written as `^4`, `4` or `⁴` (any
superscript digits, also negative: `s⁻¹`, `kg·m⁻³`), products
as `*` or `·`; a denominator without brackets is read as a whole (`W/m^2K` = W/(m²·K),
`kJ/kgK` = kJ/(kg·K)).

| Category | Examples |
|----------|----------|
| Temperature (absolute) | °C, K, °F |
| Temperature difference | K (°C/°F are always absolute; in compound units °C is a difference: `2 °C/h`, `W/(m²·°C)`) |
| Pressure | Pa, hPa, kPa, MPa, bar, mbar, atm, psi |
| Energy | J, kJ, MJ, kWh, MWh |
| Power, heat flow | W, kW, MW |
| Force | N, kN |
| Length, wavelength | m, km, cm, mm, µm, nm |
| Area, volume | m², cm², mm², m³, dm³, L |
| Mass | kg, g, t |
| Mass flow | kg/s, kg/h, t/h |
| Volume flow | m³/s, m³/h, L/s, L/min, L/h |
| Velocity, acceleration | m/s, km/h, m/s² |
| Time | s, min, h, d, a |
| Frequency, rotational speed | Hz, 1/s, 1/h, rpm, U/min, rps (revolutions, u = π·D·n) |
| Angle, angular velocity | deg, rad, rad/s, deg/h (SI rad internally) |
| Heat flux, power density | W/m², W/m³ |
| Heat transfer coefficient, U-value / UA-value | W/(m²·K), W/m²K / W/K |
| Thermal conductivity | W/(m·K), W/mK |
| Thermal resistance | m²K/W |
| Specific enthalpy, heat capacity, entropy | J/kg, kJ/kg, J/(kg·K), kJ/(kg·K), kJ/kgK |
| Density, specific volume | kg/m³, m³/kg |
| Dynamic / kinematic viscosity | Pa·s, mPa·s / m²/s, mm²/s |
| Amount of substance, molar flow | mol, kmol, mol/s, kmol/h |
| Molar mass, molar quantities | kg/kmol, g/mol, kJ/mol, J/(mol·K) |
| Electrochemistry | V, mV, A, A/m², A/cm², C/mol, Ω (ohm), kΩ, Ω·cm² |
| Energy per area / volume, annual energy | kWh/m², kJ/m³, kWh/a |
| Radiation | W/(m²·K⁴), W/m2K4, W/m^2K^4, W/(m²·µm) |
| Expansion coefficient, specific consumption | 1/K, kg/kWh |
| Ratios | %, ‰, ppm (dimensionless) |

### Dimensional Analysis

The solver automatically:
- Propagates units through algebraic expressions
- Handles exponents (e.g., `T^4` with temperature units)
- Recognizes dimensionless quantities (Nusselt, Grashof, Prandtl numbers)
- Validates unit consistency in equations

### Settings

The Settings dialog sets the display units of temperatures (°C/K), pressures (bar/Pa), energies
and specific values (kJ/J) and powers (kW/W); they act immediately on results, parameter studies
and plots. Other quantities keep the unit they were entered in or derived with (MW, kWh,
W/(m²K), ...), and every result can be switched to a compatible unit in the Results tab. The
font size is set in the View menu.

## Tests

```bash
python3 test_regressions.py       # parser, solver, unit system
python3 test_unit_constraints.py  # unit propagation and dimension checks
python3 test_berechnungen.py      # 82 problems, with and without units
python3 test_optimierung.py       # MINIMIZE/MAXIMIZE against analytic/scipy references
python3 test_gui.py               # headless GUI (briefly takes focus - don't type)
```

`test_regressions.py` covers the parser (assignment detection, vectors, unit sweeps,
keywords, comments), the solver (root selection, contradiction detection, parameter
studies, tearing, determinism) and the unit system (SI conversion, start values,
delta_K, offset conversions, radiation). `test_berechnungen.py` solves thermodynamics
and heat-transfer problems both with and without units and compares them with
independently computed reference values (CoolProp, analytical solutions, literature values
for heating values and fuel cell voltages).
Each module also has a self-test: `python3 <module>.py`.

## License

MIT License (see [LICENSE](LICENSE)) - except the files in `data/nasa/`, see below.

### Data source: NASA thermodynamic database (`data/nasa/`)

The property data of ideal gases and condensed species (`IG:`) are taken from the file
`thermo.inp` of NASA CEA (<https://github.com/nasa/cea>, commit `2f79a64`, copied on
2026-10-12, **unchanged**). They are licensed under the Apache License 2.0, see
`data/nasa/LICENSE.txt` and `data/nasa/NOTICE.txt` (Copyright 2025, United States Government
as represented by the Administrator of the National Aeronautics and Space Administration).
Source: B. J. McBride, M. J. Zehe, S. Gordon, *NASA Glenn Coefficients for Calculating
Thermodynamic Properties of Individual Species*, NASA TP-2002-211556 (2002).

- `data/nasa/*` is excluded from the MIT license of this project; the rest of the code,
  including the reader `ideal_gas.py`, is MIT.
- Keep `LICENSE.txt` and `NOTICE.txt` together with `thermo.inp` - also in packages and
  installers built from this repository.
- The data are provided by NASA without warranty; this project checks them with tests against
  reference values. Naming the source does not imply endorsement by NASA.
- `ideal_gas.py` reads the file as it is (it skips species without temperature range and empty
  ranges with Tmax ≤ Tmin) - the file itself is not modified.

## Contributions

Contributions are welcome! Please create a pull request or open an issue.
