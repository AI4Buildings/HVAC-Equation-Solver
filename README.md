# HVAC Equation Solver

Equation solver for teaching and rapid calculation of thermodynamic state changes in HVAC systems. Also used for developing benchmark tests for AI agents in building services engineering.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Version](https://img.shields.io/badge/Version-3.2.0-orange.svg)

## Features

- **Intuitive Syntax**: Equations in natural form (`h = enthalpy(water, T=100°C, p=1bar)`)
- **Thermodynamic Properties**: Over 100 fluids via CoolProp
- **Humid Air**: Psychrometric calculations (`h = HumidAir(h, T=25°C, rh=0.5, p_tot=1bar)`)
- **Blackbody Radiation**: Planck's radiation functions (`Eb`, `Blackbody`, `Wien`, `Stefan_Boltzmann`)
- **Robust Solver**: Block decomposition with bracket search and Brent's method
- **Parameter Studies**: Simple sweep syntax (`p = 25:5:50 bar`)
- **Unit System**: Automatic unit parsing, propagation and consistency checking
- **GUI**: Modern CustomTkinter interface with plotting capabilities
- **Temperature Display**: Configurable display in °C or K (Settings)

## Installation

### Required Libraries

```bash
pip install numpy scipy CoolProp matplotlib pint customtkinter
```

| Library | Version | Purpose |
|---------|---------|---------|
| numpy | >= 1.20 | Array operations, mathematical functions |
| scipy | >= 1.7 | Numerical solvers (fsolve, brentq) |
| CoolProp | >= 6.4 | Thermodynamic property data |
| matplotlib | >= 3.5 | Diagrams and plots (optional) |
| pint | >= 0.20 | Unit handling and dimensional analysis |
| customtkinter | >= 5.0 | Modern GUI |
| tkinter | - | GUI (included in Python standard library) |

### Start the Program

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
├── thermodynamics.py    # CoolProp wrapper with unit conversion
├── humid_air.py         # CoolProp HumidAirProp wrapper
├── radiation.py         # Blackbody radiation functions
├── units.py             # Unit handling and conversion (v3.0)
├── unit_constraints.py  # Unit propagation and consistency checking (v3.0)
├── test_regressions.py  # Regression test suite (python3 test_regressions.py)
├── CLAUDE.md            # Technical documentation
└── README.md            # This file
```

## Quick Start

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

### Parameter Study

```
T = 0:10:100 °C
p = 1 bar
h = enthalpy(water, T=T, p=p)
```

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
- Contradictory systems (e.g. `x+1=3` and `x+1=4`) are reported as errors
- Parameter studies use warm starts (previous point's solution as initial value)
- Default initial value 1.0 for all variables (or unit-based if units are known)
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
| Temperature difference | delta_K | `dT = 7 K`, `dT = 7 °C` (no offset) |
| Pressure p | Pa | `1 bar`, `101325 Pa` |
| Enthalpy h | J/kg | `100 kJ/kg` |
| Angles (sin, cos, tan) | Degrees (°) | |
| Entropy s | J/(kg·K) | |
| Density rho | kg/m³ | |
| Vapor quality x | - (0-1) | |

Variables starting with `dT` or `delta` are treated as temperature differences
(`delta_K`, no K↔°C offset). Computed differences like `theta = T_1 - T_2` are
recognized as `delta_K` automatically.

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

### Humid Air (HumidAir)
Output: `T`, `h`, `rh`, `w`, `p_w`, `rho_tot`, `rho_a`, `rho_w`, `T_dp`, `T_wb`
Input: `T`, `p_tot`, `rh`, `w`, `p_w`, `h`

### Mathematics
`sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, `exp`, `ln`, `log10`, `sqrt`, `abs`, `pi`

### Radiation
`Eb`, `Blackbody`, `Blackbody_cumulative`, `Wien`, `Stefan_Boltzmann`

## Unit System

The solver supports automatic unit handling and propagation:

### Specifying Units

```
T_s = 90 °C
p = 1 bar
sigma = 5.67e-8 W/m^2K^4
L = 4 µm
```

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
python3 test_regressions.py
```

Runs 46 regression tests covering the parser (assignment detection, vectors,
unit sweeps), solver (root selection, contradiction detection, parameter
studies) and the unit system (propagation, delta_K, offset conversions).
Each module also has a self-test: `python3 <module>.py`.

## License

MIT License

## Contributions

Contributions are welcome! Please create a pull request or open an issue.
