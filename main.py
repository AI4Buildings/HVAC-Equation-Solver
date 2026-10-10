#!/usr/bin/env python3
"""
HVAC Equation Solver - Ein EES-ähnlicher Gleichungslöser

Hauptanwendung mit CustomTkinter GUI.
"""

import math
import os
import sys

# Unterdrücke macOS-spezifische Warnungen
if sys.platform == 'darwin':
    os.environ['TK_SILENCE_DEPRECATION'] = '1'

import tkinter as tk
from tkinter import messagebox, filedialog
from typing import List, Optional

import customtkinter as ctk

from parser import (parse_equations, validate_system, display_name, unmangle,
                    parse_start_values, parse_start_value, start_value_entries, start_values_edit,
                    parse_optimization, start_value_units, parse_reference_states, remove_comments,
                    wavelength_literals)
from version import __version__
from solver import solve_system, solve_parametric, format_solution, SolveAnalysis
import solver as solver_module
import numpy as np

# Versuche Units-Modul zu laden
try:
    from units import (get_compatible_units, UnitValue, detect_unit_from_equation, get_initial_from_unit,
                       initial_values_from_units)
    UNITS_AVAILABLE = True
except ImportError:
    UNITS_AVAILABLE = False
    def get_initial_from_unit(unit_str):
        return 1.0

    def initial_values_from_units(variables, all_units, known_values):
        return {var: 1.0 for var in variables if all_units.get(var) is not None}

# Versuche Constraint-Propagation zu laden
try:
    from unit_constraints import (propagate_all_units, check_all_unit_consistency, propagate_all_units_complete,
                                  temperature_sum_conflicts, scale_origin,
                                  missing_unit_annotations, scale_offset_literals, si_number_literals)
    CONSTRAINT_PROPAGATION_AVAILABLE = True
except ImportError:
    CONSTRAINT_PROPAGATION_AVAILABLE = False

# Optimierung (MINIMIZE/MAXIMIZE ... VARY ...)
try:
    from optimizer import optimize_system, optimize_parametric
    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False

# Generische Fehleranalyse (Struktur, Numerik, Namens-Hinweise)
try:
    from diagnostics import analyze_structure, describe_structure, name_hints, diagnose
    DIAGNOSTICS_AVAILABLE = True
except ImportError:
    DIAGNOSTICS_AVAILABLE = False

# CustomTkinter Einstellungen
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Versuche Thermodynamik-Modul zu laden
try:
    from thermodynamics import get_fluid_info, THERMO_FUNCTIONS, set_reference_states
    THERMO_AVAILABLE = True
except ImportError:
    THERMO_AVAILABLE = False
    THERMO_FUNCTIONS = {}

# Versuche matplotlib zu laden
try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Versuche CoolProp Version zu ermitteln
try:
    import CoolProp
    COOLPROP_VERSION = CoolProp.__version__
except:
    COOLPROP_VERSION = None


# Farbschema
COLORS = {
    "bg_dark": "#1a1a2e",
    "bg_frame": "#16213e",
    "bg_input": "#0f0f1a",
    "accent": "#e94560",
    "accent_hover": "#ff6b6b",
    "text": "#eaeaea",
    "text_dim": "#8892a0",
    "success": "#4ade80",
    "error": "#f87171",
    "warning": "#fbbf24",
    "info": "#60a5fa",
    "value": "#fbbf24",
    "border": "#2d3748",
}

# Schriftgröße des Gleichungs-Editors (pt)
FONT_SIZE_MIN = 6
FONT_SIZE_MAX = 36
FONT_SIZE_DEFAULT = 16

# Text der Funktionsreferenz (Help -> Function Reference).
# Alle Werte intern in SI - Zahlen OHNE Einheit werden als SI interpretiert.
FUNCTION_HELP_TEXT = """=== HVAC EQUATION SOLVER - FUNCTION REFERENCE ===

SYNTAX:
-------
One equation per line, in any form and any order:
  x + y = 10            T_2 = T_1 + dT        Q = m*cp*(T_2 - T_1)
  ^ = power (x^2), * must always be written (2*x, r_1*(a+b));
  · can be used instead of * (0.475·10^-6)
Decimal POINT, not comma: 0.71 (a comma is reported as error)
Comments: "text" or {text} (also over several lines)
A long equation may continue on the next line inside an open
bracket, if the line ends with '(', ',' or an operator:
  Nu = IF(Re, 2300, 3.66, 3.66,
          0.023*Re^0.8*Pr^0.4)
Values with units only in assignments 'name = number unit':
  T_1 = 20 °C    p = 1 bar    A = 20 cm2    V_dot = 500 m3/h
  U = 0.3 W/(m²·K)   (m2 = m^2 = m², · or * between units)
  n = 0.3 1/h        (reciprocal units after a space; also h^-1)
  eta = 89.2 %       (% and ‰ are units: eta = 0.892; rh=50 %)
A constant WITHOUT unit is dimensionless (eta = 0.8) - except 0:
zero is zero in every unit, its unit follows from the equations
(Q_12 = 0 next to Q_12 + W_12 = m*c_v*(T_2 - T_1) is in kJ).
Not allowed: units inside equations (T_2 - 20 °C) - define a
variable instead (T_0 = 20 °C, then T_2 - T_0).

INTERNAL UNITS (SI):
--------------------
All calculations use SI base units internally:
  Temperature  K           Pressure       Pa
  Enthalpy     J/kg        Entropy, cp    J/(kg K)
  Energy       J           Power          W
  Length       m           (also µm, cm, mm -> m)
Inputs with units are converted automatically:
  T = 25 °C    p = 1 bar    h = 100 kJ/kg    Q = 5 kW
Plain numbers WITHOUT unit are SI values (p=1 means 1 Pa!).
Also in equations: in 24/(24 - t_S) with t_S = 2 h the 24 is
24 s (hint ⓘ) - define t_d = 24 h and write t_d - t_S.
Results are displayed in °C, bar, kJ/kg, kW (see Settings).

MATHEMATICAL FUNCTIONS:
-----------------------
sin(x), cos(x), tan(x)     Trigonometric (x in degrees)
asin(x), acos(x), atan(x)  Inverse trig functions (result in degrees)
sinh(x), cosh(x), tanh(x)  Hyperbolic functions (radians)
Angles with a unit are converted to degrees: a = 0.5236 rad -> 30.
exp(x)                      e^x
ln(x)                       Natural logarithm
log10(x), lg(x)             Base 10 logarithm
sqrt(x)                     Square root
abs(x)                      Absolute value
ceil(x), floor(x), round(x) Round up / down / to nearest integer
max(a, b), min(a, b)        Maximum / minimum
IF(a, b, x, y, z)           Case distinction: x if a < b,
                            y if a = b, z if a > b (see below)
value(x, unit)              Number of x in a unit (see below)
quantity(z, unit)           Quantity from a number in a unit
pi                          Pi constant
Angles are in degrees. Formulas from the literature that use radians
(e.g. view factors): atan(x)*pi/180 gives the angle in radians.
Not available (unlike EES): own functions.

CASE DISTINCTION - IF:
----------------------
IF(a, b, x, y, z) compares a with b (as in EES):
  a < b  ->  x          a = b  ->  y          a > b  ->  z
Usable in every equation, nested and in parametric studies.
Also written if(...) or If(...).
Examples:
  Re = 5000
  Re_krit = 2300
  Nu = IF(Re, Re_krit, 3.66, 3.66, 0.023*Re^0.8*0.7^0.4)
                         {laminar below Re_krit, else turbulent}
  T_1 = 20 °C
  T_2 = 30 °C
  T_max = IF(T_1, T_2, T_2, T_2, T_1)       {same as max(T_1, T_2)}
  eps = 1.1
  s_1 = IF(eps, 1.065, 0, 1, 1)             {0 below 1.065, else 1}
  k = 3
  F = IF(k, 2, 0.13, 0.33, IF(k, 4, 0.56, 0.87, 1.13))
                         {value from a table: k = 1, 2, 3, 4, >4}
a and b must have the same unit; x, y and z as well.
All five arguments are evaluated, also the branch not chosen:
it must be computable (no division by zero, no CoolProp call
outside the valid range); sqrt or ^ of a negative number does
no harm there.
If a or b is itself unknown (iterated), the equation jumps at
a = b; if the solver fails, set an initial value (see below).
Comparisons (<, >, ==) and if/else are not available otherwise.

OPTIMISATION - MINIMIZE / MAXIMIZE:
-----------------------------------
One line in the sheet:
  MINIMIZE goal VARY x = a .. b unit
  MAXIMIZE goal VARY x = a .. b unit, y = c .. d unit
x (and y) are chosen within the bounds so that the goal - a
variable that follows from the equations - becomes minimal or
maximal. Varied quantities are not unknowns: the sheet has one
equation less per varied quantity, and they must not have a
fixed value. The unit at the end applies to both bounds (as in
parametric studies); a bound may have its own unit. After a
comma the line may continue on the next line.
Examples:
  y = (x - 2)^2 + 1
  MINIMIZE y VARY x = 0 .. 5                   {x = 2, y = 1}
  MAXIMIZE eps_tot VARY m_dot_gly = 0.1 .. 4 kg/s
  MINIMIZE q VARY s = 11 .. 30 mm
  MAXIMIZE e_tot VARY m_dot_2 = 0.5 .. 2 kg/s,
                      m_dot_4 = 0.5 .. 2 kg/s
Result: the varied quantities appear in the results in the unit
of the bounds; the message says Minimum/Maximum and whether a
quantity is at a bound (optimum at the edge of the range).
Method: grid over the whole range (finds the best of several
local optima), then exact local search. Ranges over more than
two decades (lower bound > 0) are scanned logarithmically.
Candidates without a solution count as bad, not as an error.
Accuracy: the goal to about 1e-8 (relative); at a very flat
optimum the varied quantity to about 4 significant digits.
With a parametric study or value lists, the optimum is found
for every point (like the Min/Max table in EES).
Several statements are allowed, each with its own quantities.
If they influence each other, the result is an equilibrium
(no goal gets better alone) - for a joint optimum form ONE goal.
Diagram goal over x: replace the MINIMIZE line by a parametric
study x = a:step:b unit.
Time limit: 120 s per optimisation (per point of a study).

INITIAL VALUES (Solve > Initial Values):
----------------------------------------
Iterated unknowns start from automatic values: by unit (unknown
temperatures at the mean of the given ones), without units from
known neighbours in sums (T_R - T_G1: T_G1 starts near T_R).
Sheets therefore also work without any units (all values SI).
For equations with
several solutions (x^2 = 9: +3 or -3) the solver takes the one
closest to the initial value. Set your own initial values in
the dialog: in SI (288.15) or with unit (15 °C, 2 bar).
OK writes them into the sheet as a comment block:
  {$Startwerte
  x = -3
  T_2 = 15 °C
  $}
The block is saved with the file and used on every Solve, also
after opening the file on another computer. It can be edited by
hand (one value per line or separated by ';'); delete the block
to go back to the automatic values. Names that are not unknowns
of the sheet are ignored.
A start value WITH unit also sets the unit of a computed quantity
whose unit does not follow from the equations (reported under
ⓘ HINWEISE: "Einheit nicht bestimmbar ... Einheit von q angeben"),
and its display unit (as Variable Info in EES), e.g. an energy
per volume that would otherwise be shown as pressure (same
dimension): q_V = 600 kJ/m^3 in the block -> q_V in kJ/m³.

THERMODYNAMIC FUNCTIONS (CoolProp):
-----------------------------------
Syntax: function(fluid, param1=value1, param2=value2)
Fluid names: see Help > Fluid List (not case-sensitive).

Properties (results in SI):
  enthalpy(...)      Specific enthalpy [J/kg]
  entropy(...)       Specific entropy [J/(kg K)]
  intenergy(...)     Specific internal energy [J/kg]
  density(...)       Density [kg/m3]
  volume(...)        Specific volume [m3/kg]
  temperature(...)   Temperature [K]
  pressure(...)      Pressure [Pa]
  quality(...)       Vapor quality [-] (-1 outside the two-phase
                     region: subcooled, superheated, supercritical)
  cp(...), cv(...)   Specific heat capacity [J/(kg K)]
  viscosity(...)     Dynamic viscosity [Pa s]
  conductivity(...)  Thermal conductivity [W/(m K)]
  prandtl(...)       Prandtl number [-]
  soundspeed(...)    Speed of sound [m/s]

State properties (2 required; SI or with unit):
  T = Temperature [K]        e.g. T=373.15 K or T=100 °C
  p = Pressure [Pa]          e.g. p=1 bar or p=100000
  h = Enthalpy [J/kg]        e.g. h=2500 kJ/kg
  s = Entropy [J/(kg K)]     e.g. s=7 kJ/(kg*K)
  x = Vapor quality [-]      0 ... 1 (x = 2 is reported)
  rho (or d) [kg/m3], u [J/kg], v [m3/kg]
A unit that does not fit the property (e.g. p=1 kg) is reported.

Examples:
  h = enthalpy(water, T=373.15 K, p=1 bar)   {100°C, 1 bar}
  h = enthalpy(water, T=100 °C, p=1 bar)     {also valid}
  rho = density(R134a, T=298.15 K, x=1)      {25°C, sat. vapor}

Reference state (zero point of h, u, s) - own line, as in EES:
  REFERENCE R717 IIR    h = 200 kJ/kg, s = 1 kJ/(kg K) for
                        saturated liquid at 0 °C (refrigeration)
  REFERENCE R134a ASHRAE   h = 0, s = 0 sat. liquid at -40 °C
  REFERENCE water NBP   h = 0, s = 0 at the normal boiling point
  REFERENCE R717 DEFAULT   CoolProp standard (also without line)
Applies to the whole sheet and all names of the fluid (R717 =
ammonia). Differences (q_0, w_t, EER), T, p, x, rho stay the same.
Standard is already IIR for R134a, R32, R410A, CO2, propane, ...;
not for ammonia (h' at 0 °C = 345.7 kJ/kg) and water (IAPWS).

HUMID AIR FUNCTIONS:
--------------------
Syntax: HumidAir(property, T=..., rh=..., p_tot=...)  (3 inputs)
Outputs: T, T_dp, T_wb [K], h [J/kg dry air], w (or x) [kg/kg],
         rh (or phi) [-], p_w [Pa], rho_tot, rho_a, rho_w [kg/m3],
         v [m3/kg dry air], cp [J/(kg K), per kg dry air],
         cp_ha [per kg humid air]
Inputs:  T [K], p_tot (or p) [Pa], rh (or rF, phi) [-],
         w (or x) [kg/kg], p_w [Pa], h [J/kg],
         T_dp [K] (dew point), T_wb [K] (wet bulb)

  h = HumidAir(h, T=298.15 K, rh=0.5, p_tot=1 bar)   {25°C}
  h = HumidAir(h, T=25 °C, rh=0.5, p_tot=1 bar)      {also valid}
  w = HumidAir(w, T=30 °C, rh=0.6, p_tot=1 bar)
  T_dp = HumidAir(T_dp, T=25 °C, w=0.01, p_tot=1 bar)
  w = HumidAir(w, T=12 °C, T_dp=8 °C, p_tot=1 bar)
A state with more water than saturated air can hold (w > w_s at
T, p - fog) is reported as "Zustand übersättigt": the condensate
has to be part of the balance. Saturated air gives rh = 1.

RADIATION FUNCTIONS (Blackbody):
--------------------------------
Temperature T in K, wavelengths in m (SI). Units can be used for
variables and inside the call (500 °C, 5 µm). Plain numbers are
SI like everywhere: Eb(1000, 5) means 5 m - write 5 µm.

  Eb(T, lambda)              Spectral emissive power
                             [W/m3 internally, shown as W/(m2 µm)]
  Blackbody(T, l1, l2)       Fraction of energy in range l1..l2 [-]
  Blackbody_cumulative(T, l) Cumulative fraction from 0 to l [-]
  Wien(T)                    Wavelength of maximum emission
                             [m internally, shown in µm]
  Stefan_Boltzmann(T)        Total emissive power [W/m2]

Examples:
  T_s = 500 °C
  L = 5 µm
  E = Eb(T_s, L)                       {spectral power at 5 µm}
  lambda_max = Wien(T_s)               {peak wavelength}
  E_1 = Eb(300 °C, 5 µm)               {units inside the call}
  E_2 = Eb(573.15, 5e-6)               {numbers are SI: K and m}
  f = Blackbody(1273.15, 0.4 µm, 0.7 µm)   {visible, 1000°C}
  E_total = Stefan_Boltzmann(373.15)   {total emission at 100°C}

VARIABLE NAMES:
---------------
Letters, digits and _ (not starting with a digit), case-sensitive.
No Greek letters or umlauts (reported): Phi, eta, lambda, Q_waerme.
Python keywords can be used: lambda = 0.04 W/mK works.
Not usable: and, or, not, True, False, None.
e is a normal variable; Euler's number: exp(1).
Avoid naming a variable like a function you also call
(sin, exp, ln, sqrt, max, pi, enthalpy, cp, ...).

TEMPERATURE DIFFERENCES - ALWAYS IN K:
--------------------------------------
°C and °F are ALWAYS absolute temperatures (10 °C = 283.15 K).
Enter temperature DIFFERENCES in K:
  dT_1 = 10 K           {difference of 10 K}
  dT_1 = 10 °C          {WRONG: 283.15 K, an absolute temperature}
The name does not matter (dT, delta, theta, x, ...). Whether a
quantity is a temperature or a difference follows from the
equations (absolute temperatures are points, differences are
steps between them):
  theta = T_1 - T_2     {difference}
  T_1 = T_2 + x         {x difference, T_1 and T_2 absolute}
  T_m = (T_1 + T_2)/2   {absolute}
  enthalpy(water, T=T_s, ...), Eb(T, ...)  {T_s, T absolute}
  Q = m*c*(T_1 - T_2)   {a temperature in a product that is not
                         a temperature is a difference: T_1 - T_2;
                         with T_1 absolute, T_2 is absolute}
  Q = m*c*theta         {theta difference}
  theta = Q/(m*c)       {the same equation: theta difference}
  m_3*T_3 = m_1*T_1 + m_2*T_2   {mixing, m_3 = m_1 + m_2:
                         T_3 absolute like T_1, T_2}
  sigma*T^4             {T absolute (radiation needs Kelvin)}
For the display, each equation is also checked with the solution:
it must stay valid when the zero point of the temperature scale
is shifted (absolute temperatures shift, differences do not).
Absolute temperatures are shown in °C (Settings), differences
in K (the unit selector converts them without offset).
Values entered in K whose meaning cannot be decided are shown
as entered (K). T only in p*v = R*T (product): shown in K -
the value in K is right either way.
Sums of absolute temperatures (T_3 = T_1 + T_2) are neither a
temperature nor a difference. They are calculated in Kelvin like
everything else (20 °C + 40 °C = 606.3 K = 333.15 °C) and reported
under ⓘ HINWEISE - usually a difference was entered in °C.

TEMPERATURE SCALE - IMPORTANT:
------------------------------
Temperatures are ALWAYS calculated in KELVIN, without exception.
Physical laws work directly (p*v = R*T, sigma*T^4,
T_2 = T_1*(p_2/p_1)^...).
Formulas DEFINED IN °C (heating curve, Magnus formula,
cp(theta) polynomials) are numeric-value equations - write them
with value() and quantity() (see next section), otherwise the
result is wrong without any error message.

NUMERIC-VALUE EQUATIONS - value() / quantity():
-----------------------------------------------
Empirical formulas often hold only for NUMBERS in certain units
(heating curve in °C, h = 5.7 + 3.8*v with v in m/s):
  value(x, unit)     number of the quantity x in that unit
                     (dimensionless), e.g. value(T_a, °C)
  quantity(z, unit)  quantity from a number z in that unit
Any unit works (°C, °F, bar, kW, m3/h, %, ...).
Examples:
  T_a = -5 °C
  T_VL = quantity(20 + 1.5*(20 - value(T_a, °C)), °C)
                         {heating curve in °C: 57.5 °C}
  v = 2 m/s
  h = quantity(5.7 + 3.8*value(v, m/s), W/(m^2*K))
  p = 3 bar
  n = value(p, bar)      {3}
The unit check knows these functions: value(T_a, °C) requires a
temperature, the result of quantity(..., W/(m^2*K)) has W/(m²K).

PARAMETRIC STUDIES (Sweeps):
----------------------------
Syntax: variable = start:step:end [unit]

Examples:
  x = 0:0.25:1          {0, 0.25, 0.5, 0.75, 1}
  T = 20:5:40 °C        {20, 25, 30, 35, 40 °C}
  p = 1:0.5:3 bar       {1, 1.5, 2, 2.5, 3 bar}
Without unit, values are SI (T = 20:5:40 would be 20..40 K).

Value lists (e.g. measured data):
  T_a = [-5.2 -4.8 -3.9 -2.7] °C
  m_dot = [0.95; 1.02; 1.00; 0.98] kg/s
Separators: space, tab, line break, ';' or ','. Decimal POINT.
A list may span several lines - paste a column from Excel/CSV/
TXT between [ and ] (via clipboard). Unit after the ].

Several sweep variables / lists are combined point by point
(same number of values). Points without solution are reported.

After solving: Plot > New Plot Window (several curves) or
Plot > Quick Plot X-Y.

MESSAGES:
---------
Errors name the line and mark the position with ▶, e.g.
  Zeile 14: 'r_1' ist keine Funktion ... bei: r_1 ▶( ...
Unterbestimmt  = equations missing (lists the unknowns)
Überbestimmt / Widersprüchlich = too many or conflicting values
Abbruch nach Zeitlimit = no solution within 60 s: set initial
  values (Solve > Initial Values) or simplify the system
Startwert ... / Startwerte-Block = error in {$Startwerte ... $}
  (line number as for equations)
Einheit nicht bestimmbar = the unit of these quantities does not
  follow from the equations; give the named one a unit (start
  value with unit), the others follow
Optimierung: Minimum/Maximum von ... = optimisation result;
  "an der Untergrenze/Obergrenze" = optimum at the edge of the range
⚠ UNIT WARNINGS (n) and ⓘ HINWEISE (n) above the results are
clickable and open the details in the Residuals tab.
"""


def pretty_unit(unit: str) -> str:
    """
    Anzeigeform einer Einheit: degC -> °C, degF -> °F, um -> µm (pint versteht beide),
    delta_K -> K (Temperaturdifferenzen werden in K angezeigt, DIN 1345 / ISO 80000-5).
    """
    import re
    if unit == 'delta_K':
        return 'K'
    unit = unit.replace('degC', '°C').replace('degF', '°F')
    return re.sub(r'(?<![A-Za-z])um(?![A-Za-z])', 'µm', unit)


def fluid_help_text() -> str:
    """
    Text der Fluidliste - aus den Daten erzeugt (Kurznamen in thermodynamics.py,
    vollständige Liste aus CoolProp), damit die Hilfe nie vom Programm abweicht.
    """
    from thermodynamics import FLUID_ALIASES, get_available_fluids
    lines = ["=== AVAILABLE FLUIDS (CoolProp) ===", "",
             "Usage: enthalpy(water, T=20 °C, p=1 bar)",
             "Fluid names are not case-sensitive. Humid air: use HumidAir(...).", "",
             "SHORT NAMES:"]
    by_fluid = {}
    for alias, fluid in FLUID_ALIASES.items():
        by_fluid.setdefault(fluid, []).append(alias)
    for fluid in sorted(by_fluid, key=str.lower):
        lines.append(f"  {fluid:<12} {', '.join(sorted(by_fluid[fluid]))}")
    fluids = get_available_fluids()
    lines += ["", f"ALL COOLPROP FLUIDS ({len(fluids)}) - usable with these names:"]
    width = max((len(f) for f in fluids), default=10) + 2
    columns = 3
    for i in range(0, len(fluids), columns):
        lines.append("  " + "".join(f"{f:<{width}}" for f in fluids[i:i + columns]).rstrip())
    return "\n".join(lines) + "\n"


def export_to_system_clipboard(text: str) -> bool:
    """
    Übergibt Text fest an die Zwischenablage des Betriebssystems.

    Tk stellt kopierten Text auf macOS (und unter X11) nur "auf Anfrage" bereit:
    Solange das Programm läuft, können andere Programme einfügen - nach dem
    Beenden ist der Inhalt aber verloren. Daher wird beim Kopieren zusätzlich
    direkt in die System-Zwischenablage geschrieben.
    Windows: Tk übergibt die Daten beim Beenden selbst (WM_RENDERALLFORMATS).
    """
    import shutil
    import subprocess
    if sys.platform == 'darwin':
        command = ['pbcopy']
    elif sys.platform.startswith('linux'):
        for candidate in (['wl-copy'], ['xclip', '-selection', 'clipboard'],
                          ['xsel', '--clipboard', '--input']):
            if shutil.which(candidate[0]):
                command = candidate
                break
        else:
            return False
    else:
        return False
    try:
        # UTF-8 erzwingen, sonst werden °, ², µ je nach Locale verfälscht
        env = dict(os.environ, LANG='en_US.UTF-8', LC_ALL='en_US.UTF-8')
        subprocess.run(command, input=text.encode('utf-8'), check=True, timeout=5, env=env)
        return True
    except Exception:
        return False


def _suppress_macos_warning(func):
    """Wrapper um macOS Cocoa-Warnungen bei Dateidialogen zu unterdrücken."""
    if sys.platform != 'darwin':
        return func()
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        return func()
    finally:
        os.dup2(saved, 2)
        os.close(saved)


class EquationSolverApp(ctk.CTk):
    """Hauptanwendung für den Gleichungslöser."""

    def __init__(self):
        super().__init__()

        # Fenster-Konfiguration
        self.title(f"HVAC Equation Solver {__version__}")
        self.geometry("1200x800")
        self.minsize(800, 600)

        # Setze Hintergrundfarbe
        self.configure(fg_color=COLORS["bg_dark"])

        # Aktueller Dateipfad
        self.current_file = None

        # Schriftgröße (Standard: 16, Bereich FONT_SIZE_MIN..FONT_SIZE_MAX)
        self.font_size = FONT_SIZE_DEFAULT

        # Re-Entrancy-Schutz: solve() ruft self.update() auf (Fortschritt),
        # dabei dürfen F5/Solve/New/Open/Clear keinen zweiten Lauf starten
        self._solving = False

        # Gespeicherte Variablen und manuelle Startwerte
        self.known_variables = set()
        self.manual_initial_values = {}
        self.inferred_units = {}

        # Letzte Lösung (für Plots und Analysis)
        self.last_solution = None
        self.last_sweep_vars = {}
        self.last_solve_stats = {}
        self.last_analysis = None
        self.current_unit_values = {}  # Einheiten-Informationen für Variablen
        self.value_labels = {}  # Referenzen auf Value-Labels (für Unit-Änderung)
        self.unit_dropdowns = {}  # Referenzen auf Unit-Dropdowns
        self.temp_display_unit = ctk.StringVar(value="degC")  # Standard-Anzeigeeinheit für Temperaturen (°C)
        self.pressure_display_unit = ctk.StringVar(value="bar")  # Standard: bar (statt Pa)
        self.energy_display_unit = ctk.StringVar(value="kJ")  # Standard: kJ (statt J)
        self.power_display_unit = ctk.StringVar(value="kW")  # Standard: kW (statt W)
        # Änderung einer Anzeige-Einstellung -> Ergebnisse sofort neu anzeigen
        self._last_results_args = None
        for display_var in (self.temp_display_unit, self.pressure_display_unit,
                            self.energy_display_unit, self.power_display_unit):
            display_var.trace_add("write", lambda *args: self._refresh_results())

        # Grid-Konfiguration
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Erstelle die GUI-Elemente
        self._create_header()
        self._create_main_content()
        self._create_statusbar()
        self._create_menu()
        self._setup_bindings()

    def _create_header(self):
        """Erstellt den Header mit Logo, Titel und Buttons."""
        header = ctk.CTkFrame(self, fg_color=COLORS["bg_frame"], corner_radius=0, height=60)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(1, weight=1)

        # Logo/Icon Frame
        logo_frame = ctk.CTkFrame(header, fg_color=COLORS["accent"], width=50, height=50, corner_radius=8)
        logo_frame.grid(row=0, column=0, padx=15, pady=8)
        logo_frame.grid_propagate(False)

        logo_label = ctk.CTkLabel(logo_frame, text="Σ", font=ctk.CTkFont(size=28, weight="bold"),
                                   text_color="white")
        logo_label.place(relx=0.5, rely=0.5, anchor="center")

        # Titel
        title_frame = ctk.CTkFrame(header, fg_color="transparent")
        title_frame.grid(row=0, column=1, sticky="w", padx=10)

        title_label = ctk.CTkLabel(title_frame, text="HVAC Equation Solver",
                                    font=ctk.CTkFont(size=20, weight="bold"),
                                    text_color=COLORS["text"])
        title_label.pack(anchor="w")

        subtitle_label = ctk.CTkLabel(title_frame, text="Thermodynamic System Analysis",
                                       font=ctk.CTkFont(size=12),
                                       text_color=COLORS["text_dim"])
        subtitle_label.pack(anchor="w")

        # Buttons Frame
        buttons_frame = ctk.CTkFrame(header, fg_color="transparent")
        buttons_frame.grid(row=0, column=2, padx=15, pady=8)

        # Solve Button (prominent)
        self.solve_btn = ctk.CTkButton(
            buttons_frame,
            text="▷ Solve",
            command=self.solve,
            width=100,
            height=36,
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.solve_btn.pack(side="left", padx=(0, 5))

        # F5 Badge
        f5_label = ctk.CTkLabel(buttons_frame, text="F5", font=ctk.CTkFont(size=10),
                                 text_color=COLORS["text_dim"],
                                 fg_color=COLORS["bg_dark"], corner_radius=4,
                                 width=24, height=18)
        f5_label.pack(side="left", padx=(0, 10))

        # Clear Button
        self.clear_btn = ctk.CTkButton(
            buttons_frame,
            text="⊘ Clear",
            command=self.clear_all,
            width=80,
            height=36,
            fg_color=COLORS["bg_dark"],
            hover_color=COLORS["border"],
            border_width=1,
            border_color=COLORS["border"],
            font=ctk.CTkFont(size=13)
        )
        self.clear_btn.pack(side="left", padx=(0, 5))

        # Examples Button
        self.example_btn = ctk.CTkButton(
            buttons_frame,
            text="☰ Examples",
            command=self._insert_example,
            width=100,
            height=36,
            fg_color=COLORS["bg_dark"],
            hover_color=COLORS["border"],
            border_width=1,
            border_color=COLORS["border"],
            font=ctk.CTkFont(size=13)
        )
        self.example_btn.pack(side="left", padx=(0, 5))

        # Settings Button
        self.settings_btn = ctk.CTkButton(
            buttons_frame,
            text="⚙",
            command=self.show_settings,
            width=36,
            height=36,
            fg_color=COLORS["bg_dark"],
            hover_color=COLORS["border"],
            border_width=1,
            border_color=COLORS["border"],
            font=ctk.CTkFont(size=18)
        )
        self.settings_btn.pack(side="left")

    def _create_main_content(self):
        """Erstellt den Hauptinhalt mit Equations und Solution Panels."""
        # Container Frame
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        content.grid_columnconfigure(0, weight=1)
        content.grid_rowconfigure(0, weight=1)

        # PanedWindow für verschiebbare Trennung (verwende tkinter.ttk)
        from tkinter import ttk

        # Style für PanedWindow im Dark Mode
        style = ttk.Style()
        style.configure("Dark.TPanedwindow", background=COLORS["bg_dark"])

        self.paned = ttk.PanedWindow(content, orient=tk.HORIZONTAL)
        self.paned.grid(row=0, column=0, sticky="nsew")

        # Linkes Panel: Equations (in eigenem Frame für PanedWindow)
        eq_container = ctk.CTkFrame(self.paned, fg_color="transparent")
        self._create_equations_panel(eq_container)
        self.paned.add(eq_container, weight=2)

        # Rechtes Panel: Solution (in eigenem Frame für PanedWindow)
        sol_container = ctk.CTkFrame(self.paned, fg_color="transparent")
        self._create_solution_panel(sol_container)
        self.paned.add(sol_container, weight=1)

    def _create_equations_panel(self, parent):
        """Erstellt das Equations Panel."""
        # Parent konfigurieren für Ausdehnung
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)

        eq_frame = ctk.CTkFrame(parent, fg_color=COLORS["bg_frame"], corner_radius=10)
        eq_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=0)
        eq_frame.grid_columnconfigure(0, weight=1)
        eq_frame.grid_rowconfigure(1, weight=1)

        # Header
        header = ctk.CTkFrame(eq_frame, fg_color="transparent", height=40)
        header.grid(row=0, column=0, sticky="ew", padx=15, pady=(10, 5))

        # Icon und Titel
        ctk.CTkLabel(header, text="T", font=ctk.CTkFont(size=16, weight="bold"),
                      text_color=COLORS["info"]).pack(side="left")
        ctk.CTkLabel(header, text="  EQUATIONS", font=ctk.CTkFont(size=13, weight="bold"),
                      text_color=COLORS["text"]).pack(side="left")

        # Syntax-Hinweis
        syntax_hint = ctk.CTkLabel(
            header,
            text='x + y = 10   ·  Kommentare: "..." oder {...}',
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_dim"]
        )
        syntax_hint.pack(side="right")

        # Text Editor
        self.equations_text = ctk.CTkTextbox(
            eq_frame,
            font=ctk.CTkFont(family="Courier", size=self.font_size),
            fg_color=COLORS["bg_input"],
            text_color=COLORS["text"],
            corner_radius=8,
            border_width=1,
            border_color=COLORS["border"],
            wrap="none"
        )
        self.equations_text.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

        # Enable undo/redo on the internal text widget
        self.equations_text._textbox.configure(undo=True, maxundo=-1)

    def _create_solution_panel(self, parent):
        """Erstellt das Solution Panel mit TabView."""
        # Parent konfigurieren für Ausdehnung
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)

        sol_frame = ctk.CTkFrame(parent, fg_color=COLORS["bg_frame"], corner_radius=10)
        sol_frame.grid(row=0, column=0, sticky="nsew", padx=(5, 0), pady=0)
        sol_frame.grid_columnconfigure(0, weight=1)
        sol_frame.grid_rowconfigure(1, weight=1)

        # Header
        header = ctk.CTkFrame(sol_frame, fg_color="transparent", height=40)
        header.grid(row=0, column=0, sticky="ew", padx=15, pady=(10, 5))

        # Icon und Titel
        ctk.CTkLabel(header, text="☑", font=ctk.CTkFont(size=16),
                      text_color=COLORS["success"]).pack(side="left")
        ctk.CTkLabel(header, text="  SOLUTION", font=ctk.CTkFont(size=13, weight="bold"),
                      text_color=COLORS["text"]).pack(side="left")

        # TabView für Results und Residuals
        self.tab_view = ctk.CTkTabview(
            sol_frame,
            fg_color=COLORS["bg_input"],
            segmented_button_fg_color=COLORS["bg_dark"],
            segmented_button_selected_color=COLORS["accent"],
            segmented_button_unselected_color=COLORS["bg_frame"],
            corner_radius=8
        )
        self.tab_view.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

        # Results Tab
        self.results_tab = self.tab_view.add("Results")
        self._create_results_tab()

        # Residuals Tab
        self.residuals_tab = self.tab_view.add("Residuals")
        self._create_residuals_tab()

    def _create_results_tab(self):
        """Erstellt den Inhalt des Results Tabs."""
        self.results_tab.grid_columnconfigure(0, weight=1)
        self.results_tab.grid_rowconfigure(2, weight=1)

        # Status Frame
        self.status_frame = ctk.CTkFrame(self.results_tab, fg_color="transparent", height=50)
        self.status_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        # Status Label (SOLUTION FOUND / ERROR)
        self.result_status_label = ctk.CTkLabel(
            self.status_frame,
            text="",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLORS["text_dim"]
        )
        self.result_status_label.pack(side="left")

        # Unit Warning Label (⚠ UNIT WARNINGS)
        # Klickbar: springt zu den Warnungen im Residuals Tab
        self._unit_warning_font = ctk.CTkFont(size=12, weight="bold")
        self.unit_warning_label = ctk.CTkLabel(
            self.status_frame,
            text="",
            font=self._unit_warning_font,
            text_color=COLORS["warning"],
            cursor="hand2"
        )
        self.unit_warning_label.pack(side="left", padx=(20, 0))
        self.unit_warning_label.bind("<Button-1>", lambda e: self.show_unit_warnings())
        self.unit_warning_label.bind("<Enter>", lambda e: self._unit_warning_font.configure(underline=True))
        self.unit_warning_label.bind("<Leave>", lambda e: self._unit_warning_font.configure(underline=False))
        self.unit_warnings_content = None  # Inhalt der Warnungs-Sektion im Residuals Tab

        # Hinweise (z.B. mögliche Tippfehler) - ebenfalls klickbar
        self._hints_font = ctk.CTkFont(size=12, weight="bold")
        self.hints_label = ctk.CTkLabel(
            self.status_frame,
            text="",
            font=self._hints_font,
            text_color=COLORS["info"],
            cursor="hand2"
        )
        self.hints_label.pack(side="left", padx=(20, 0))
        self.hints_label.bind("<Button-1>", lambda e: self.show_hints())
        self.hints_label.bind("<Enter>", lambda e: self._hints_font.configure(underline=True))
        self.hints_label.bind("<Leave>", lambda e: self._hints_font.configure(underline=False))
        self.hints_content = None

        # Stats Label (15 direct, 1 iterative)
        self.result_stats_label = ctk.CTkLabel(
            self.status_frame,
            text="",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_dim"]
        )
        self.result_stats_label.pack(side="right")

        # Info Frame (Equations: X, Unknowns: Y, Status: OK)
        self.info_frame = ctk.CTkFrame(self.results_tab, fg_color=COLORS["bg_dark"],
                                        corner_radius=6, height=35)
        self.info_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=(0, 10))

        self.info_label = ctk.CTkLabel(
            self.info_frame,
            text="",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_dim"]
        )
        self.info_label.pack(padx=10, pady=8)

        # Scrollable Frame für Variablen-Tabelle
        self.results_scroll = ctk.CTkScrollableFrame(
            self.results_tab,
            fg_color="transparent",
            corner_radius=0
        )
        self.results_scroll.grid(row=2, column=0, sticky="nsew", padx=5)
        self.results_scroll.grid_columnconfigure(0, weight=1)
        self.results_scroll.grid_columnconfigure(1, weight=0)

        # Header der Tabelle
        self.table_header = ctk.CTkFrame(self.results_scroll, fg_color="transparent")
        self.table_header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 5))

        ctk.CTkLabel(self.table_header, text="VARIABLE", font=ctk.CTkFont(size=11, weight="bold"),
                      text_color=COLORS["text_dim"], width=150, anchor="w").pack(side="left", padx=5)
        ctk.CTkLabel(self.table_header, text="VALUE", font=ctk.CTkFont(size=11, weight="bold"),
                      text_color=COLORS["text_dim"], anchor="e").pack(side="right", padx=5)

        # Container für Variablen-Zeilen
        self.var_rows_container = ctk.CTkFrame(self.results_scroll, fg_color="transparent")
        self.var_rows_container.grid(row=1, column=0, columnspan=2, sticky="nsew")

    def _create_residuals_tab(self):
        """Erstellt den Inhalt des Residuals Tabs."""
        self.residuals_tab.grid_columnconfigure(0, weight=1)
        self.residuals_tab.grid_rowconfigure(0, weight=1)

        # Scrollable Frame für Residuals
        self.residuals_scroll = ctk.CTkScrollableFrame(
            self.residuals_tab,
            fg_color="transparent",
            corner_radius=0
        )
        self.residuals_scroll.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.residuals_scroll.grid_columnconfigure(0, weight=1)

        # Container für Residuals-Sektionen
        self.residuals_content = ctk.CTkFrame(self.residuals_scroll, fg_color="transparent")
        self.residuals_content.grid(row=0, column=0, sticky="nsew")
        self.residuals_content.grid_columnconfigure(0, weight=1)

        # Placeholder Text (wird versteckt wenn Daten vorhanden)
        self.residuals_placeholder = ctk.CTkLabel(
            self.residuals_content,
            text="Run Solve to see residuals.\n\n"
                 "This tab will show:\n"
                 "• Constants\n"
                 "• Direct Evaluations\n"
                 "• Block decomposition with residuals",
            font=ctk.CTkFont(size=12),
            text_color=COLORS["text_dim"],
            justify="center"
        )
        self.residuals_placeholder.grid(row=0, column=0, pady=50)

        # Sections storage
        self.residuals_sections = []

    def _create_collapsible_section(self, parent, title: str, count: int, row: int,
                                      header_color: Optional[str] = None) -> ctk.CTkFrame:
        """Erstellt eine aufklappbare Sektion für die Residuals."""
        # Main container
        section_frame = ctk.CTkFrame(parent, fg_color=COLORS["bg_frame"], corner_radius=6)
        section_frame.grid(row=row, column=0, sticky="ew", pady=3)
        section_frame.grid_columnconfigure(0, weight=1)

        # Header Frame (klickbar)
        header = ctk.CTkFrame(section_frame, fg_color="transparent", height=30)
        header.grid(row=0, column=0, sticky="ew", padx=5, pady=3)
        header.grid_columnconfigure(1, weight=1)

        # Expand/Collapse Button
        expand_var = tk.BooleanVar(value=True)
        expand_btn = ctk.CTkLabel(
            header, text="▼", width=20,
            font=ctk.CTkFont(size=10),
            text_color=header_color or COLORS["text_dim"]
        )
        expand_btn.grid(row=0, column=0, padx=(5, 0))

        # Title
        title_label = ctk.CTkLabel(
            header, text=f"{title} ({count})",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=header_color or COLORS["text"],
            anchor="w"
        )
        title_label.grid(row=0, column=1, sticky="w", padx=5)

        # Content Frame
        content_frame = ctk.CTkFrame(section_frame, fg_color="transparent")
        content_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 5))
        content_frame.grid_columnconfigure(0, weight=1)

        # Toggle function
        def toggle():
            if expand_var.get():
                content_frame.grid_remove()
                expand_btn.configure(text="▶")
                expand_var.set(False)
            else:
                content_frame.grid()
                expand_btn.configure(text="▼")
                expand_var.set(True)

        def expand():
            content_frame.grid()
            expand_btn.configure(text="▼")
            expand_var.set(True)

        # Bind click to toggle
        expand_btn.bind("<Button-1>", lambda e: toggle())
        title_label.bind("<Button-1>", lambda e: toggle())
        header.bind("<Button-1>", lambda e: toggle())

        content_frame.expand = expand  # z.B. für show_unit_warnings()
        content_frame.section_frame = section_frame
        return content_frame

    def _show_residuals_section(self, content):
        """Wechselt zum Residuals Tab und zeigt eine Sektion aufgeklappt."""
        if content is None:
            return
        self.tab_view.set("Residuals")
        content.expand()
        try:
            # Warnungen/Hinweise stehen oben im Residuals Tab
            self.update_idletasks()
            total = max(1, self.residuals_content.winfo_height())
            self.residuals_scroll._parent_canvas.yview_moveto(content.section_frame.winfo_y() / total)
        except Exception:
            pass

    def show_unit_warnings(self):
        """Wechselt zum Residuals Tab und zeigt die Einheiten-Warnungen (aufgeklappt)."""
        if self.unit_warning_label.cget("text"):
            self._show_residuals_section(self.unit_warnings_content)

    def show_hints(self):
        """Wechselt zum Residuals Tab und zeigt die Hinweise (aufgeklappt)."""
        if self.hints_label.cget("text"):
            self._show_residuals_section(self.hints_content)

    def _update_hints_label(self, analysis):
        """Hinweise-Label im Results Tab ("ⓘ HINWEISE (n)")."""
        hints = getattr(analysis, "hints", []) if analysis else []
        self.hints_label.configure(text=f"ⓘ HINWEISE ({len(hints)})" if hints else "")

    @staticmethod
    def _check_optimization(goals, variables, constants, sweep_vars) -> dict:
        """
        Prüft die variierten Größen der Optimierungsanweisungen und liefert
        {Name: Bereichsmitte} (Platzhalter für Struktur- und Einheitenprüfung).
        """
        optimized = {}
        for goal in goals:
            for name, lower, upper in zip(goal.names, goal.lower, goal.upper):
                shown = display_name(name)
                if name in constants:
                    raise ValueError(f"Zeile {goal.line}: {shown} hat im Blatt einen festen Wert - "
                                     f"für die Optimierung die Zuweisung entfernen")
                if name in sweep_vars:
                    raise ValueError(f"Zeile {goal.line}: {shown} ist eine Parameterstudie/Werteliste "
                                     f"und kann nicht zugleich optimiert werden")
                if name not in variables:
                    raise ValueError(f"Zeile {goal.line}: {shown} kommt in keiner Gleichung vor")
                optimized[name] = (lower + upper) / 2
        return optimized

    def _structure_message(self, equations, variables, constants, original_equations, source_text) -> str:
        """Strukturdiagnose (unter-/überbestimmte Teile + Namens-Hinweise) als Meldungstext."""
        if not DIAGNOSTICS_AVAILABLE or not equations:
            return ""
        try:
            report = analyze_structure(equations, variables)
            messages = describe_structure(report, original_equations, source_text, include_over=True)
            hints = name_hints(equations, variables, set(constants), original_equations, source_text)
        except Exception:
            return ""
        text = " ".join(messages)
        if hints:
            text += " Hinweis: " + " ".join(hints)
        return text.strip()

    def _add_equation_row(self, parent, equation: str, var: str, value: float, residual: float, row: int):
        """Fügt eine Zeile für eine Gleichung zu den Residuals hinzu."""
        row_frame = ctk.CTkFrame(parent, fg_color="transparent", height=22)
        row_frame.grid(row=row, column=0, sticky="ew", pady=1)
        row_frame.grid_columnconfigure(0, weight=1)

        # Residual color based on magnitude
        if abs(residual) < 1e-8:
            res_color = COLORS["success"]
        elif abs(residual) < 1e-4:
            res_color = COLORS["warning"]
        else:
            res_color = COLORS["error"]

        # Equation (truncated if too long; _kw_lambda -> lambda)
        equation = unmangle(equation)
        eq_display = equation if len(equation) < 50 else equation[:47] + "..."
        eq_label = ctk.CTkLabel(
            row_frame, text=eq_display,
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text"],
            anchor="w"
        )
        eq_label.grid(row=0, column=0, sticky="w")

        # Residual
        res_text = f"Res: {residual:.2E}"
        res_label = ctk.CTkLabel(
            row_frame, text=res_text,
            font=ctk.CTkFont(size=10),
            text_color=res_color,
            anchor="e"
        )
        res_label.grid(row=0, column=1, sticky="e", padx=5)

    def _add_unit_warning_row(self, parent, warning, row: int):
        """Fügt eine Zeile für eine Einheiten-Warnung hinzu."""
        import re

        # Main container für diese Warnung
        warning_frame = ctk.CTkFrame(parent, fg_color=COLORS["bg_dark"], corner_radius=4)
        warning_frame.grid(row=row, column=0, sticky="ew", pady=3)
        warning_frame.grid_columnconfigure(0, weight=1)

        # Kopfzeile: betroffene Variable (bzw. Art der Warnung) + Art
        var_frame = ctk.CTkFrame(warning_frame, fg_color="transparent")
        var_frame.pack(fill="x", padx=8, pady=(5, 2))

        name = warning.variable.lstrip("⚠ ").strip()
        is_variable = bool(re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', name))
        kind = "Dimensionsfehler" if "Dimensionsfehler" in warning.explanation else ""
        ctk.CTkLabel(
            var_frame, text=f"Variable: {display_name(name)}" if is_variable else name,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLORS["warning"],
            anchor="w"
        ).pack(side="left")
        if is_variable and kind:
            ctk.CTkLabel(
                var_frame, text=kind,
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color=COLORS["error"],
                anchor="e"
            ).pack(side="right")

        # Umrechnungsfaktor nur, wenn er etwas aussagt (z.B. 100 für bar·m³ vs kJ)
        if warning.conversion_factor not in (0, 1):
            ctk.CTkLabel(
                var_frame, text=f"Faktor {warning.conversion_factor:g}×",
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color=COLORS["error"],
                anchor="e"
            ).pack(side="right", padx=(0, 10))

        # Gleichungen mit Einheiten-Info (je Gleichung zwei Zeilen, umbrechend)
        for eq, unit in warning.units.items():
            ctk.CTkLabel(
                warning_frame, text=f"• {unmangle(eq)}",
                font=ctk.CTkFont(size=11),
                text_color=COLORS["text"],
                anchor="w", justify="left", wraplength=520
            ).pack(fill="x", padx=15, pady=(1, 0))
            ctk.CTkLabel(
                warning_frame, text=f"   {unmangle(unit) if unit else '(dimensionslos)'}",
                font=ctk.CTkFont(size=11),
                text_color=COLORS["accent"],
                anchor="w", justify="left", wraplength=520
            ).pack(fill="x", padx=15, pady=(0, 1))

        # Hinweis
        if is_variable and kind:
            hint = f"⚠ Einheit von {display_name(name)} oder die Gleichung prüfen!"
        else:
            hint = "⚠ Prüfen Sie die Einheiten in Ihren Gleichungen!"
        ctk.CTkLabel(
            warning_frame, text=hint,
            font=ctk.CTkFont(size=10),
            text_color=COLORS["warning"],
            anchor="w"
        ).pack(fill="x", padx=8, pady=(3, 5))

    def _update_residuals_tab(self, analysis: SolveAnalysis):
        """Aktualisiert den Residuals Tab mit den Lösungsdaten."""
        # Placeholder verstecken
        self.residuals_placeholder.grid_remove()

        # Alte Sektionen löschen
        for section in self.residuals_sections:
            section.destroy()
        self.residuals_sections = []

        # Residuals direkt anzeigen
        current_row = 0

        # === Unit Warnings Section ===
        self.unit_warnings_content = None
        if analysis.unit_warnings:
            content = self._create_collapsible_section(
                self.residuals_content, "⚠ UNIT WARNINGS", len(analysis.unit_warnings), current_row,
                header_color=COLORS["warning"]
            )
            self.unit_warnings_content = content
            self.residuals_sections.append(content.master)

            for i, warning in enumerate(analysis.unit_warnings):
                self._add_unit_warning_row(content, warning, i)

            current_row += 1

        # === Hinweise Section (generische Diagnose, z.B. mögliche Tippfehler) ===
        self.hints_content = None
        if getattr(analysis, "hints", None):
            content = self._create_collapsible_section(
                self.residuals_content, "ⓘ HINWEISE", len(analysis.hints), current_row,
                header_color=COLORS["info"]
            )
            self.hints_content = content
            self.residuals_sections.append(content.master)
            for i, hint in enumerate(analysis.hints):
                ctk.CTkLabel(
                    content, text=f"• {hint}",
                    font=ctk.CTkFont(size=11),
                    text_color=COLORS["text"],
                    anchor="w", justify="left", wraplength=560
                ).grid(row=i, column=0, sticky="ew", pady=2)
            current_row += 1

        # === Constants Section ===
        if analysis.constants:
            content = self._create_collapsible_section(
                self.residuals_content, "Constants", len(analysis.constants), current_row
            )
            self.residuals_sections.append(content.master)

            for i, eq_info in enumerate(analysis.constants):
                self._add_equation_row(
                    content, eq_info.original, eq_info.variable,
                    eq_info.value, eq_info.residual, i
                )
            current_row += 1

        # === Direct Evaluations Section ===
        if analysis.direct_evals:
            content = self._create_collapsible_section(
                self.residuals_content, "Direct Evaluations", len(analysis.direct_evals), current_row
            )
            self.residuals_sections.append(content.master)

            for i, eq_info in enumerate(analysis.direct_evals):
                self._add_equation_row(
                    content, eq_info.original, eq_info.variable,
                    eq_info.value, eq_info.residual, i
                )
            current_row += 1

        # === Single Unknowns Section ===
        if analysis.single_unknowns:
            content = self._create_collapsible_section(
                self.residuals_content, "Single Unknown (Iterative)", len(analysis.single_unknowns), current_row
            )
            self.residuals_sections.append(content.master)

            for i, eq_info in enumerate(analysis.single_unknowns):
                self._add_equation_row(
                    content, eq_info.original, eq_info.variable,
                    eq_info.value, eq_info.residual, i
                )
            current_row += 1

        # === Blocks Section ===
        for block in analysis.blocks:
            title = f"Block {block.block_number}"
            content = self._create_collapsible_section(
                self.residuals_content, title, len(block.equations), current_row
            )
            self.residuals_sections.append(content.master)

            # Block header mit Max-Residuum
            max_res_frame = ctk.CTkFrame(content, fg_color="transparent")
            max_res_frame.grid(row=0, column=0, sticky="ew", pady=(0, 3))

            vars_text = ", ".join(display_name(v) for v in block.variables)
            if len(vars_text) > 40:
                vars_text = vars_text[:37] + "..."

            ctk.CTkLabel(
                max_res_frame, text=f"Variables: {vars_text}",
                font=ctk.CTkFont(size=10),
                text_color=COLORS["text_dim"]
            ).pack(side="left")

            max_res_color = COLORS["success"] if block.max_residual < 1e-8 else (
                COLORS["warning"] if block.max_residual < 1e-4 else COLORS["error"]
            )
            ctk.CTkLabel(
                max_res_frame, text=f"Max Res: {block.max_residual:.2E}",
                font=ctk.CTkFont(size=10, weight="bold"),
                text_color=max_res_color
            ).pack(side="right")

            # Gleichungen im Block
            for i, (eq, res) in enumerate(zip(block.equations, block.residuals)):
                self._add_equation_row(content, eq, "", 0, res, i + 1)

            current_row += 1

        # Falls keine Daten vorhanden
        if current_row == 0:
            self.residuals_placeholder.grid()

    def _create_statusbar(self):
        """Erstellt die Statusbar am unteren Rand."""
        statusbar = ctk.CTkFrame(self, fg_color=COLORS["bg_frame"], corner_radius=0, height=30)
        statusbar.grid(row=2, column=0, sticky="ew")

        # Linke Seite: Dateiname
        self.file_label = ctk.CTkLabel(
            statusbar,
            text="Unsaved",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_dim"]
        )
        self.file_label.pack(side="left", padx=15, pady=5)

        # Rechte Seite: Versionsinfo
        version_text = f"Python {sys.version_info.major}.{sys.version_info.minor}"
        if COOLPROP_VERSION:
            version_text = f"CoolProp v{COOLPROP_VERSION}  •  {version_text}"
        version_text = f"Version {__version__}  •  {version_text}"

        version_label = ctk.CTkLabel(
            statusbar,
            text=version_text,
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_dim"]
        )
        version_label.pack(side="right", padx=15, pady=5)

        # Mitte: Status
        self.status_label = ctk.CTkLabel(
            statusbar,
            text="Ready",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text"]
        )
        self.status_label.pack(pady=5)

    def _create_menu(self):
        """Erstellt die Menüleiste."""
        menubar = tk.Menu(self)
        self.configure(menu=menubar)

        # File Menü
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New", command=self.new_file, accelerator="Ctrl+N")
        file_menu.add_command(label="Open...", command=self.open_file, accelerator="Ctrl+O")
        file_menu.add_command(label="Save", command=self.save_file, accelerator="Ctrl+S")
        file_menu.add_command(label="Save As...", command=self.save_file_as)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)

        # Edit Menü
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Clear All", command=self.clear_all)
        edit_menu.add_command(label="Insert Example", command=self._insert_example)

        # View Menü
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Increase Font Size", command=self.increase_font_size, accelerator="Ctrl++")
        view_menu.add_command(label="Decrease Font Size", command=self.decrease_font_size, accelerator="Ctrl+-")

        # Solve Menü
        solve_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Solve", menu=solve_menu)
        solve_menu.add_command(label="Solve", command=self.solve, accelerator="F5")
        solve_menu.add_separator()
        solve_menu.add_command(label="Initial Values...", command=self.show_initial_values_dialog)

        # Plot Menü
        if MATPLOTLIB_AVAILABLE:
            plot_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="Plot", menu=plot_menu)
            plot_menu.add_command(label="New Plot Window...", command=self.show_plot_dialog)
            plot_menu.add_command(label="Quick Plot X-Y...", command=self.show_quick_plot_dialog)

        # Help Menü
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Function Reference", command=self.show_function_help)
        if THERMO_AVAILABLE:
            help_menu.add_command(label="Fluid List (CoolProp)", command=self.show_fluid_help)

    def _setup_bindings(self):
        """Richtet Keyboard-Shortcuts ein."""
        shortcuts = {
            "<Control-n>": self.new_file,
            "<Control-o>": self.open_file,
            "<Control-s>": self.save_file,
            "<F5>": self.solve,
            "<Control-plus>": self.increase_font_size,
            "<Control-minus>": self.decrease_font_size,
            "<Control-equal>": self.increase_font_size,
        }

        def make_handler(func):
            def handler(event=None):
                func()
                # "break" verhindert, dass weitere Bindings (Text-Klasse,
                # Toplevel) dasselbe Event noch einmal verarbeiten
                return "break"
            return handler

        for sequence, func in shortcuts.items():
            handler = make_handler(func)
            # Fenster-weit (Fokus außerhalb des Editors)
            self.bind(sequence, handler)
            # Direkt am Editor: Instanz-Bindings laufen VOR den Tk-Text-
            # Klassenbindings. Ohne "break" hier würde z.B. <Control-o> der
            # Text-Klasse (Emacs "open line") einen Zeilenumbruch an der
            # Cursorposition einfügen und so eine Gleichung zerteilen.
            self.equations_text.bind(sequence, handler)

        # Kopieren/Ausschneiden: zusätzlich fest an die System-Zwischenablage
        # übergeben (siehe export_to_system_clipboard). Instanz-Binding läuft VOR
        # der Text-Klassenbindung, daher erst nach deren Kopieren exportieren.
        for sequence in ("<<Copy>>", "<<Cut>>"):
            self.equations_text._textbox.bind(
                sequence, lambda event: self.after_idle(self._export_clipboard), add="+")

        # Undo/Redo bindings (works on both Windows/Linux and macOS)
        self.equations_text.bind("<Control-z>", self._undo)
        self.equations_text.bind("<Command-z>", self._undo)
        self.equations_text.bind("<Control-y>", self._redo)  # Windows/Linux
        self.equations_text.bind("<Command-y>", self._redo)  # macOS

    def _export_clipboard(self):
        """Aktuellen Tk-Zwischenablageinhalt an das Betriebssystem übergeben."""
        try:
            text = self.clipboard_get()
        except tk.TclError:
            return
        export_to_system_clipboard(text)

    # === Schriftgröße ===

    def set_font_size(self, size: int):
        """Setzt die Schriftgröße."""
        self.font_size = max(FONT_SIZE_MIN, min(FONT_SIZE_MAX, int(size)))
        self.equations_text.configure(font=ctk.CTkFont(family="Courier", size=self.font_size))
        self.status_label.configure(text=f"Font size: {self.font_size}pt")

    def increase_font_size(self):
        self.set_font_size(self.font_size + 2)

    def decrease_font_size(self):
        self.set_font_size(self.font_size - 2)

    # === Undo/Redo ===

    def _undo(self, event=None):
        """Undo the last edit."""
        try:
            self.equations_text._textbox.edit_undo()
        except Exception:
            pass  # Nothing to undo
        return "break"

    def _redo(self, event=None):
        """Redo the last undone edit."""
        try:
            self.equations_text._textbox.edit_redo()
        except Exception:
            pass  # Nothing to redo
        return "break"

    # === Datei-Operationen ===

    def _reset_document_state(self):
        """Setzt alle dokumentbezogenen Zustände zurück (New/Open).

        Ohne Reset würden Undo-Stack, manuelle Startwerte und die letzte
        Lösung des VORHERIGEN Dokuments ins neue Dokument übernommen
        (z.B. Ctrl+Z nach Open -> alter Inhalt -> Ctrl+S überschreibt Datei).
        """
        self.equations_text._textbox.edit_reset()
        self.equations_text._textbox.edit_modified(False)
        self.manual_initial_values = {}
        self.known_variables = set()
        self.inferred_units = {}
        self.last_solution = None
        self.last_sweep_vars = {}
        self.last_analysis = None

    def new_file(self):
        """Erstellt eine neue leere Datei."""
        if self._solving:
            return
        self.equations_text.delete("1.0", "end")
        self.clear_results()
        self._reset_document_state()
        self.current_file = None
        self._update_file_label()
        self.status_label.configure(text="New file")

    @staticmethod
    def _read_text_file(filepath: str) -> str:
        """Liest eine Textdatei: UTF-8 (mit/ohne BOM), Fallback Latin-1."""
        try:
            with open(filepath, 'r', encoding='utf-8-sig') as f:
                return f.read()
        except UnicodeDecodeError:
            # Ältere Dateien (z.B. Windows-Editor) mit '°' als Byte 0xB0
            with open(filepath, 'r', encoding='latin-1') as f:
                return f.read()

    def open_file(self):
        """Öffnet eine Datei."""
        if self._solving:
            return
        filetypes = [("HES Files", "*.hes"), ("Text Files", "*.txt"), ("All Files", "*.*")]
        filepath = _suppress_macos_warning(lambda: filedialog.askopenfilename(
            title="Open File", filetypes=filetypes, defaultextension=".hes"
        ))
        if filepath:
            try:
                content = self._read_text_file(filepath)
            except Exception as e:
                messagebox.showerror("Error", f"Could not open file:\n{e}")
                return
            self.equations_text.delete("1.0", "end")
            self.equations_text.insert("1.0", content)
            self.clear_results()
            self._reset_document_state()
            self.current_file = filepath
            self._update_file_label()
            self.status_label.configure(text=f"Opened: {os.path.basename(filepath)}")

    def save_file(self):
        """Speichert die aktuelle Datei."""
        if self.current_file:
            self._save_to_file(self.current_file)
        else:
            self.save_file_as()

    def save_file_as(self):
        """Speichert die Datei unter neuem Namen."""
        filetypes = [("HES Files", "*.hes"), ("Text Files", "*.txt"), ("All Files", "*.*")]
        filepath = _suppress_macos_warning(lambda: filedialog.asksaveasfilename(
            title="Save File", filetypes=filetypes, defaultextension=".hes"
        ))
        # Nur bei erfolgreichem Speichern den neuen Pfad übernehmen -
        # sonst zeigt die Statusleiste "✓ Saved" für eine nie geschriebene Datei
        if filepath and self._save_to_file(filepath):
            self.current_file = filepath
            self._update_file_label()

    def _save_to_file(self, filepath: str) -> bool:
        """Speichert den Inhalt in eine Datei. Liefert True bei Erfolg."""
        try:
            content = self.equations_text.get("1.0", "end-1c")
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            self.status_label.configure(text=f"Saved: {os.path.basename(filepath)}")
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Could not save file:\n{e}")
            self.status_label.configure(text="Save failed")
            return False

    def _update_file_label(self):
        """Aktualisiert das Datei-Label in der Statusbar."""
        if self.current_file:
            self.file_label.configure(text=f"✓ Saved: {os.path.basename(self.current_file)}")
        else:
            self.file_label.configure(text="Unsaved")

    # === Lösen ===

    def solve(self):
        """Löst das Gleichungssystem (mit Schutz gegen Re-Entrancy).

        _solve_impl() ruft self.update() auf (Statusanzeige, Fortschritt bei
        Parameterstudien). Dabei verarbeitete Events (F5, Solve-Button) würden
        sonst einen zweiten, verschachtelten Lauf starten -> doppelte Zeilen
        bzw. vermischte Ergebnisse. Solche Aufrufe werden ignoriert.
        """
        if self._solving:
            return
        self._solving = True
        try:
            self.solve_btn.configure(state="disabled")
        except tk.TclError:
            pass
        try:
            self._solve_impl()
        finally:
            self._solving = False
            try:
                self.solve_btn.configure(state="normal")
            except tk.TclError:
                pass  # Fenster wurde während des Lösens geschlossen

    def _solve_impl(self):
        """Löst das Gleichungssystem."""
        self.clear_results()
        self.status_label.configure(text="Solving...")
        self.update()

        equations_text = self.equations_text.get("1.0", "end-1c")

        # Alte Lösung verwerfen, BEVOR geparst wird - sonst bliebe sie bei einem
        # Parse-Fehler (z.B. unbekannte Einheit) stehen und Plot zeigte alte Daten
        self.last_solution = None

        try:
            # Bezugszustände der Fluide (REFERENCE R717 IIR) gelten für diesen Lauf
            if THERMO_AVAILABLE:
                set_reference_states({})
                set_reference_states(parse_reference_states(equations_text))
            # Parse Gleichungen mit Einheiten
            equations, variables, initial_values, sweep_vars, original_equations, unit_values = parse_equations(equations_text, parse_units=True)

            # Startwerte stehen im Blatt (Block {$Startwerte ... $}, von
            # Solve > Initial Values geschrieben) - der Text ist die einzige Quelle
            self.manual_initial_values = parse_start_values(equations_text)

            # Optimierung: die variierten Größen sind keine Unbekannten
            goals = parse_optimization(equations_text)
            if goals and not OPTIMIZER_AVAILABLE:
                raise RuntimeError("Optimierung nicht verfügbar (optimizer.py fehlt)")
            optimized = self._check_optimization(goals, variables, initial_values, sweep_vars)
            for goal in goals:
                unit_values.update(goal.unit_values)   # Anzeige in der Einheit der Grenzen

            self.known_variables = variables.copy()
            self.current_unit_values = unit_values
            self.known_variables.update(sweep_vars.keys())
            variables = variables - set(optimized)

            self.last_solution = None
            self.last_sweep_vars = sweep_vars
            self.last_analysis = None  # Residuals-Daten

            # Validiere System (mit Konstanten für Constraint-Zählung; variierte
            # Größen zählen wie gegebene Werte)
            valid, msg = validate_system(equations, variables, {**initial_values, **optimized})

            n_equations = len(equations)
            n_variables = len(variables)

            # Spezialfall: Keine Gleichungen, nur Konstanten und/oder Sweep-Variablen
            # (beide zusammen anzeigen - der Sweep darf nicht verloren gehen)
            if not valid and n_equations == 0 and (initial_values or sweep_vars):
                result = dict(initial_values)
                result.update(sweep_vars)
                if sweep_vars:
                    n_points = len(next(iter(sweep_vars.values())))
                    self._show_results(result, f"Parametric: {n_points} points", 0, 0, "OK")
                    self.status_label.configure(text=f"Parametric study: {n_points} points")
                else:
                    self._show_results(result, "Constants only", 0, 0, "OK")
                    self.status_label.configure(text="Constants calculated")
                self.last_solution = result
                return

            if not valid:
                self._show_error(self._structure_message(
                    equations, variables, {**initial_values, **optimized}, original_equations,
                    equations_text) or msg)
                self.status_label.configure(text="Error: System not solvable")
                return

            constants = initial_values.copy()

            # Manuelle Startwerte
            solver_initial = {}
            for var, val in self.manual_initial_values.items():
                if var in variables or var in optimized:
                    solver_initial[var] = val

            # NEU: Einheiten-Propagation VOR dem Lösen für bessere Startwerte
            # Dies ist der generische Ansatz, der NICHT von Variablennamen abhängt
            self.inferred_units = {}  # Speichere abgeleitete Einheiten für Dialog
            temperature_hints = []
            if CONSTRAINT_PROPAGATION_AVAILABLE and UNITS_AVAILABLE:
                # Sammle bekannte Einheiten aus unit_values
                known_units = {}
                for var, uv in unit_values.items():
                    if uv.calc_unit:
                        known_units[var] = uv.calc_unit
                # In K eingegebene Größen: absolut oder Differenz folgt aus den Gleichungen
                open_k, celsius = self._temperature_inputs(unit_values)
                # Startwerte mit Einheit legen die Einheit berechneter Größen fest
                self._start_units = {}
                for name, uv in start_value_units(equations_text).items():
                    if name not in known_units and uv.calc_unit:
                        known_units[name] = uv.calc_unit
                        self._start_units[name] = uv
                        if uv.calc_unit == 'K' and uv.original_unit.strip() in ('K', 'kelvin'):
                            open_k.add(name)

                # Führe vollständige Einheiten-Propagation durch (nicht bestimmbarer
                # Temperatur-Charakter zählt für die Startwerte als absolut)
                all_units = propagate_all_units_complete(original_equations, known_units,
                                                         open_temperatures=open_k)
                self.inferred_units = all_units
                # Mittel der gegebenen Temperaturen nur über sicher absolute Werte
                definite = propagate_all_units_complete(original_equations, known_units,
                                                        open_temperatures=open_k,
                                                        undetermined_temperature='delta_K')
                start_units = {**all_units, **{v: definite.get(v) for v in open_k}}
                if any(uv.original_unit for uv in unit_values.values()):
                    temperature_hints += self._missing_unit_hints(
                        original_equations, {**known_units, **{c: '' for c in self._dimensionless_constants(
                            constants, sweep_vars) if c not in known_units}}, open_k)
                for source, number, scale in scale_offset_literals(original_equations, all_units):
                    temperature_hints.append(
                        f"'{unmangle(source)}': {number:g} ist der Nullpunkt der {scale}-Skala. "
                        f"Temperaturen werden hier in Kelvin gerechnet - die Umrechnung "
                        f"verschiebt den Nullpunkt ein zweites Mal. Formeln, die für Zahlenwerte "
                        f"in {scale} gelten, mit value()/quantity() schreiben (siehe Hilfe)")
                for source, number, name, unit, si in si_number_literals(
                        original_equations, {v: uv.original_unit for v, uv in unit_values.items()
                                             if uv.original_unit}):
                    temperature_hints.append(
                        f"'{unmangle(remove_comments(source)).strip()}': Die Zahl {number:g} steht in einer Summe mit "
                        f"{display_name(name)} (in {unit} eingegeben) und gilt als {number:g} {si} - "
                        f"Zahlen ohne Einheit sind SI-Werte. Ist {number:g} {unit} gemeint, als Größe "
                        f"mit Einheit angeben (z.B. c = {number:g} {unit})")
                for source, number in wavelength_literals(original_equations):
                    temperature_hints.append(
                        f"'{unmangle(remove_comments(source)).strip()}': Die Wellenlänge {number:g} ohne "
                        f"Einheit gilt als {number:g} m = {number * 1e6:g} µm - Zahlen ohne Einheit sind "
                        f"SI-Werte. Wellenlängen in µm mit Einheit angeben (z.B. {number:g} µm)")
                temperature_hints += self._celsius_difference_hints(
                    original_equations, known_units, open_k, celsius, unit_values)

                # Leite Startwerte aus Einheiten ab (nur für Variablen ohne manuellen Startwert)
                # Vorgaben inkl. Sweep-/Listenwerte (erster Punkt) für das Temperatur-Mittel
                sweep_first = {name: float(np.asarray(values).ravel()[0]) for name, values in sweep_vars.items()
                               if np.asarray(values).size}
                unit_initial = initial_values_from_units(variables, start_units,
                                                         {**sweep_first, **constants, **optimized})
                self.auto_initial_values = unit_initial  # Anzeige im Initial-Values-Dialog
                for var, value in unit_initial.items():
                    if var not in solver_initial:
                        solver_initial[var] = value

            # Löse System
            def progress_callback(current, total):
                self.status_label.configure(text=f"Solving... {current}/{total}")
                self.update()

            if goals and sweep_vars:
                # Optimum je Punkt (wie die Min/Max-Tabelle in EES)
                success, solution, solve_msg = optimize_parametric(
                    equations, variables, sweep_vars, goals, solver_initial,
                    progress_callback=progress_callback, constants=constants,
                    original_equations=original_equations
                )
                analysis = None
            elif goals:
                self.status_label.configure(text="Optimizing...")
                self.update()
                success, solution, solve_msg, analysis = optimize_system(
                    equations, variables, goals, solver_initial, constants,
                    original_equations, return_analysis=True
                )
                self.last_analysis = analysis
            elif sweep_vars:
                success, solution, solve_msg = solve_parametric(
                    equations, variables, sweep_vars, solver_initial,
                    progress_callback=progress_callback, constants=constants,
                    original_equations=original_equations
                )
                # Keine Residuals für Parameterstudien
                analysis = None
            else:
                result = solve_system(
                    equations, variables, solver_initial, constants=constants,
                    original_equations=original_equations, return_analysis=True
                )
                success, solution, solve_msg, analysis = result
                self.last_analysis = analysis

            # Einheiten der berechneten Variablen (auch bei Teillösung, damit
            # keine rohen SI-Zahlen ohne Einheit erscheinen)
            if UNITS_AVAILABLE and solution:
                self._assign_result_units(solution, original_equations, unit_values, constants)

                # Einheiten-Konsistenzprüfung
                if CONSTRAINT_PROPAGATION_AVAILABLE and analysis:
                    # WICHTIG: Auch leere Einheiten ('') bedeuten "dimensionslos" und müssen enthalten sein!
                    known_units = {var: uv.calc_unit for var, uv in self.current_unit_values.items()
                                   if uv.calc_unit is not None}
                    # Füge Konstanten ohne Einheit als dimensionslos hinzu
                    for var in self._dimensionless_constants(constants):
                        if var not in known_units:
                            known_units[var] = ''
                    unit_warnings = check_all_unit_consistency(solution, original_equations, known_units)
                    if unit_warnings:
                        analysis.unit_warnings = unit_warnings

            # Generische Diagnose: Struktur/Numerik (bei Fehler) + Namens-Hinweise
            diagnosis_errors = []
            if DIAGNOSTICS_AVAILABLE and not sweep_vars:
                try:
                    diagnosis_errors, hints = diagnose(
                        equations, variables, {**constants, **optimized}, original_equations,
                        equations_text, solution if not success else None)
                    if analysis is not None:
                        analysis.hints = (temperature_hints + hints
                                          + self._single_phase_quality_hints(original_equations, solution))
                except Exception:
                    diagnosis_errors = []

            if success:
                self._show_results(solution, solve_msg, n_equations,
                                   f"{n_variables} + {len(optimized)} optimized"
                                   if optimized else n_variables, "OK")
                self.last_solution = solution
                if sweep_vars and solve_msg.startswith("Parameterstudie teilweise"):
                    # Einzelne Punkte ohne Lösung: nicht als voller Erfolg anzeigen
                    self._show_error(solve_msg, status_text="● PARTIAL SOLUTION",
                                     status_color=COLORS["warning"])
                    self.status_label.configure(text="Parametric study: some points failed")
                elif sweep_vars:
                    self.status_label.configure(text=f"Parametric study: {len(list(sweep_vars.values())[0])} points")
                else:
                    self.status_label.configure(text="Optimum found" if goals else "Solution found")
                    if goals:
                        # Ergebnis der Optimierung (Minimum/Maximum, Grenzen) statt der Zählung
                        self.info_label.configure(text=solve_msg, text_color=COLORS["text"])
                    # Residuals Tab aktualisieren
                    if analysis:
                        self._update_residuals_tab(analysis)
                        # Unit Warning Label im Results Tab aktualisieren
                        if analysis.unit_warnings:
                            n_warnings = len(analysis.unit_warnings)
                            self.unit_warning_label.configure(
                                text=f"⚠ UNIT WARNINGS ({n_warnings})"
                            )
                        else:
                            self.unit_warning_label.configure(text="")
                        self._update_hints_label(analysis)
            else:
                # Erst die Teillösung, DANACH die Meldung des Solvers anzeigen -
                # _show_results überschreibt Status- und Info-Zeile, sonst wäre
                # z.B. "Widersprüchliches System: ..." nie sichtbar
                is_contradiction = "widersprüch" in (solve_msg or "").lower()
                # Nicht-Widerspruch: generische Diagnose (unterbestimmter Teil bzw.
                # numerisch ungelöste Unbekannte) statt "Unvollständig: n Gleichungen ..."
                if not is_contradiction and diagnosis_errors and not goals:
                    timed_out = (solve_msg or "").startswith("Zeitlimit")
                    solve_msg = " ".join(diagnosis_errors)
                    # Die konkrete Ursache zuerst: Fehlermeldung der Auswertung (Stoffwert-
                    # funktion außerhalb des Gültigkeitsbereichs, 0/0) mit Originalzeile
                    evaluation_errors = getattr(analysis, 'evaluation_errors', None) or []
                    if evaluation_errors:
                        solve_msg = ("Auswertungsfehler: " + "; ".join(evaluation_errors[:3])
                                     + ". " + solve_msg)
                    if timed_out:
                        solve_msg = (f"Abbruch nach Zeitlimit ({solver_module.SOLVE_TIME_LIMIT:.0f} s). "
                                     + solve_msg)
                if solution:
                    self._show_results(solution, "Partial solution", n_equations, n_variables, "FAIL")
                    self._show_error(solve_msg, status_text="● PARTIAL SOLUTION",
                                     status_color=COLORS["warning"])
                else:
                    self._show_error(solve_msg)
                self.status_label.configure(
                    text="Contradictory system" if is_contradiction else "Convergence problem")
                # Auch bei Fehler Residuals anzeigen
                if analysis:
                    self._update_residuals_tab(analysis)
                    # Unit Warning Label auch bei Fehler anzeigen
                    if analysis.unit_warnings:
                        n_warnings = len(analysis.unit_warnings)
                        self.unit_warning_label.configure(
                            text=f"⚠ UNIT WARNINGS ({n_warnings})"
                        )
                    else:
                        self.unit_warning_label.configure(text="")
                    self._update_hints_label(analysis)

        except Exception as e:
            self._show_error(str(e))
            self.status_label.configure(text=f"Error: {e}")

    @staticmethod
    def _temperature_inputs(unit_values: dict):
        """
        (in K eingegebene Größen, in °C/°F eingegebene Größen). °C/°F sind immer
        absolute Temperaturen; eine Angabe in K kann eine Temperatur oder eine
        Temperaturdifferenz sein - das folgt aus den Gleichungen, nicht aus dem Namen.
        """
        open_k, celsius = set(), set()
        for var, uv in unit_values.items():
            if uv.calc_unit != 'K' or not uv.original_unit:
                continue
            if uv.original_unit.strip() in ('K', 'kelvin'):
                open_k.add(var)
            else:
                celsius.add(var)
        return open_k, celsius

    @staticmethod
    def _single_phase_quality_hints(original_equations, solution) -> List[str]:
        """
        quality(...) liefert außerhalb des Nassdampfgebiets -1 (CoolProp: unterkühlt, überhitzt,
        überkritisch) - jeder Aufruf wird mit der Lösung ausgewertet; bei -1 ein Hinweis.
        """
        if not solution:
            return []
        try:
            from parser import _iter_call_spans
            from solver import _get_eval_context
            context = _get_eval_context()
        except Exception:
            return []
        values = {}
        for name, value in solution.items():
            try:
                values[name] = float(value[0]) if isinstance(value, np.ndarray) else float(value)
            except (TypeError, ValueError, IndexError):
                continue
        hints = []
        for parsed, original in original_equations.items():
            for start, _, close, _ in _iter_call_spans(parsed, {'quality'}):
                try:
                    x = float(eval(parsed[start:close + 1], {"__builtins__": {}}, {**context, **values}))
                except Exception:
                    continue
                if x == -1.0:
                    hints.append(f"'{unmangle(original or parsed)}': quality(...) = -1 - der Zustand liegt "
                                 f"nicht im Nassdampfgebiet (unterkühlt, überhitzt oder überkritisch)")
                    break
        return hints

    @staticmethod
    def _zero_point_evaluator(solution: dict):
        """
        (Werte, residual) für den Nullpunkt-Test des Temperatur-Charakters: Werte der Lösung
        in SI (Parameterstudie: erster Punkt) und die Auswertung einer Gleichung wie im Solver.
        """
        try:
            from solver import _calculate_residual, _get_eval_context
            context = _get_eval_context()
        except Exception:
            return None
        values = {}
        for name, value in solution.items():
            try:
                values[name] = float(value[0]) if isinstance(value, np.ndarray) else float(value)
            except (TypeError, ValueError, IndexError):
                continue
        return values, lambda equation, trial: _calculate_residual(equation, trial, context)

    @staticmethod
    def _dimensionless_constants(constants, sweep_vars=()) -> List[str]:
        """
        Konstanten ohne Einheit, die in einem Blatt mit Einheiten als dimensionslos gelten.
        Ausgenommen ist der Wert 0: null ist in jeder Einheit null (wie die Zahl 0 in einer
        Summe), seine Einheit folgt aus den Gleichungen (Q_12 = 0 neben Q_12 + W_12 = dU).
        """
        names = [name for name, value in constants.items()
                 if not (np.ndim(value) == 0 and float(value) == 0.0)]
        return names + list(sweep_vars)

    @staticmethod
    def _missing_unit_hints(original_equations, known_units, open_k) -> List[str]:
        """
        Hinweise für Größen, deren Einheit aus den Gleichungen nicht folgt, mit den
        Größen, deren Einheit man angeben muss (unit_constraints.missing_unit_annotations).
        """
        try:
            groups = missing_unit_annotations(original_equations, known_units, open_k)
        except Exception:
            return []
        hints = []
        for members, suggest in groups:
            names = [display_name(n) for n in members]
            given = [display_name(n) for n in suggest]
            others = [n for n in names if n not in given]
            alternative = f" (oder von {', '.join(others)})" if len(given) == 1 and others else ""
            hints.append(
                f"Einheit nicht bestimmbar: {', '.join(names)} - aus den Gleichungen folgt nur ihr "
                f"Zusammenhang. Einheit von {', '.join(given)}{alternative} angeben: als Startwert mit "
                f"Einheit (Solve > Initial Values, z.B. {given[0]} = 1 <Einheit>); die übrigen folgen daraus")
        return hints

    @staticmethod
    def _celsius_difference_hints(original_equations, known_units, open_k, celsius, unit_values):
        """
        Hinweise für Gleichungen, in denen in °C/°F angegebene Werte (absolute
        Temperaturen) so addiert werden, dass weder Temperatur noch Differenz
        herauskommt - meist eine Temperaturdifferenz in °C (T_2 = T_1 + x, x = 10 °C).
        """
        if not celsius:
            return []
        try:
            conflicts = temperature_sum_conflicts(original_equations, known_units, open_k)
        except Exception:
            return []
        hints = []
        for equation, names in conflicts:
            given = [display_name(n) for n in names if n in celsius]
            if not given:
                continue
            units = sorted({pretty_unit(unit_values[n].original_unit) for n in names if n in celsius})
            hints.append(
                f"'{unmangle(equation)}': {', '.join(given)} in {'/'.join(units)} angegeben, also "
                f"absolute Temperatur(en) - so kombiniert ergibt sich weder eine Temperatur noch "
                f"eine Temperaturdifferenz (gerechnet wird in Kelvin). Ist ein Wert eine "
                f"Temperaturdifferenz? Temperaturdifferenzen in K angeben (z.B. 10 K statt 10 °C); "
                f"Formeln für Zahlenwerte in °C mit value()/quantity() schreiben")
        return hints

    def _assign_result_units(self, solution: dict, original_equations: dict, unit_values: dict,
                             constants: dict):
        """
        Bestimmt die Anzeige-Einheiten der berechneten Variablen.

        Einzige Quelle ist die dimensionale Analyse (propagate_all_units_complete):
        sie kennt die Ausgabe-Einheiten der Stoffwert-/Strahlungsfunktionen und
        unterscheidet Temperaturdifferenzen (delta_K) von absoluten Temperaturen (K).
        Interne Werte sind immer SI; die Einheit ist nur das Anzeige-Label.

        Konstanten OHNE Einheit gelten als dimensionslos (epsilon, eta, kappa),
        aber nur wenn das Blatt überhaupt Einheiten verwendet. In einem reinen
        Zahlen-Blatt kann "m_dot = 2.78" genauso gut kg/s bedeuten - dort würde
        "dimensionslos" falsche Labels erzeugen (W = m_dot*(h_1-h_2) als kJ/kg).
        """
        if not CONSTRAINT_PROPAGATION_AVAILABLE:
            return
        known_units = {var: uv.calc_unit for var, uv in unit_values.items() if uv.calc_unit}
        if any(uv.original_unit for uv in unit_values.values()):
            for var in self._dimensionless_constants(constants):
                known_units.setdefault(var, '')
        open_k, _ = self._temperature_inputs(unit_values)
        for name, uv in getattr(self, '_start_units', {}).items():
            if name not in known_units:
                # Startwert mit Einheit legt auch die Anzeige fest (wie Variable Info in EES):
                # q_V = 600 kJ/m^3 -> kJ/m^3 statt bar (gleiche Dimension, andere Größenart)
                known_units[name] = (uv.original_unit.strip() if uv.calc_unit not in ('K', 'delta_K')
                                     and not scale_origin(uv.original_unit) else uv.calc_unit)
                if uv.calc_unit == 'K' and uv.original_unit.strip() in ('K', 'kelvin'):
                    open_k.add(name)
        self._kelvin_display = set()
        try:
            # Temperatur-Charakter aus der Struktur (Summen, Funktionsargumente; für die
            # Anzeige zusätzlich: Nullpunkt-Test mit der Lösung und Temperatur im Produkt
            # ohne Temperatur-Dimension = Differenz). Zwei Durchläufe unterscheiden
            # "bestimmt" von "nicht bestimmbar".
            zero_point = self._zero_point_evaluator(solution)
            # Anzeige-Labels: eingegebene Einheiten statt SI (V_dot = n*V_dot_P mit m^3/h
            # bleibt m^3/h); Temperaturen behalten K/delta_K für den Temperatur-Charakter, und wo
            # die Settings die Einheit bestimmen (Leistung, Energie, Druck, J/kg, J/(kg K)), gelten sie
            settings_controlled = (self._POWER_UNITS | self._ENERGY_UNITS | self._PRESSURE_UNITS
                                   | self._SPECIFIC_ENERGY_UNITS | self._SPECIFIC_HEAT_UNITS)
            for var, uv in unit_values.items():
                if (uv.original_unit and var in known_units and uv.calc_unit not in ('K', 'delta_K')
                        and uv.calc_unit not in settings_controlled and not scale_origin(uv.original_unit)):
                    known_units[var] = uv.original_unit.strip()
            units = propagate_all_units_complete(original_equations, known_units,
                                                 open_temperatures=open_k,
                                                 undetermined_temperature='K',
                                                 differences_in_products=True,
                                                 zero_point=zero_point)
            safe = propagate_all_units_complete(original_equations, known_units,
                                                open_temperatures=open_k,
                                                undetermined_temperature='delta_K',
                                                differences_in_products=True,
                                                zero_point=zero_point)
        except Exception:
            return
        for var, val in solution.items():
            existing = self.current_unit_values.get(var)
            if var in open_k:
                if units.get(var) == 'delta_K':
                    # In K eingegeben, aus den Gleichungen eine Differenz: in K anzeigen
                    try:
                        first = float(val[0]) if isinstance(val, np.ndarray) else float(val)
                        self.current_unit_values[var] = UnitValue.from_si_base(first, 'delta_K')
                    except Exception:
                        pass
                elif safe.get(var) == 'delta_K':
                    # In K eingegeben, Charakter nicht bestimmbar: wie eingegeben (K)
                    self._kelvin_display.add(var)
                continue
            if existing is not None and existing.original_unit:
                continue  # Vom Benutzer angegebene Einheit hat Vorrang
            unit = units.get(var)
            if not unit or unit == 'dimensionless':
                continue
            try:
                first = float(val[0]) if isinstance(val, np.ndarray) else float(val)
                self.current_unit_values[var] = UnitValue.from_si_base(first, unit)
            except Exception:
                pass

    # Einheiten, für die die Anzeige-Einstellungen (Settings) gelten
    _TEMPERATURE_UNITS = {'K', 'degC', 'degF', 'kelvin', 'celsius', 'fahrenheit', '°C', '°F'}  # noqa
    _PRESSURE_UNITS = {'Pa', 'bar', 'kPa', 'MPa', 'mbar', 'atm', 'psi'}
    _ENERGY_UNITS = {'J', 'kJ'}
    _POWER_UNITS = {'W', 'kW'}
    _SPECIFIC_ENERGY_UNITS = {'J/kg', 'kJ/kg'}
    _SPECIFIC_HEAT_UNITS = {'J/(kg*K)', 'kJ/(kg*K)', 'J/kgK', 'kJ/kgK', 'J/kg/K', 'kJ/kg/K',
                            'J/(kg·K)', 'kJ/(kg·K)', 'kJ/kgC'}

    def _display_unit_for(self, unit_value) -> str:
        """
        Anzeige-Einheit einer Variable. Wert UND Label werden immer aus dieser
        einen Einheit gebildet (früher: Wert in kW umgerechnet, Label "MW").

        Die Settings (°C/K, bar/Pa, kJ/J, kW/W) gelten nur für die jeweilige
        Standard-Familie; andere Einheiten (MW, kWh, W/(m^2*K), ...) bleiben wie
        angegeben bzw. abgeleitet.
        """
        unit = unit_value.original_unit
        if unit in self._TEMPERATURE_UNITS:
            return pretty_unit(self.temp_display_unit.get())
        if unit in self._PRESSURE_UNITS:
            return self.pressure_display_unit.get()
        energy = self.energy_display_unit.get()
        if unit in self._ENERGY_UNITS:
            return energy
        if unit in self._SPECIFIC_ENERGY_UNITS:
            return f"{energy}/kg"
        if unit in self._SPECIFIC_HEAT_UNITS:
            return f"{energy}/(kg*K)"
        if unit in self._POWER_UNITS:
            return self.power_display_unit.get()
        return pretty_unit(unit)

    @staticmethod
    def _si_to_unit(value_si: float, unit: str) -> float:
        """
        Rechnet einen SI-Wert in die Anzeige-Einheit um (inkl. Offset °C/°F). Bei einer
        Einheit mit Nullpunkt bleibt vom Abziehen ein Rundungsrest (273.15 K -> -1.7e-13 °C):
        Werte unter 1e-9 des SI-Werts sind 0.
        """
        try:
            value = UnitValue.from_si_base(float(value_si), unit).original_value
        except Exception:
            return float(value_si)
        try:
            if value and abs(value) < 1e-9 * abs(float(value_si)) and scale_origin(unit):
                return 0.0
        except Exception:
            pass
        return value

    def _display_values(self, var: str, val):
        """
        (Anzeige-Werte, Einheit) einer Variable - Skalar oder Array (Sweep).
        Ohne bekannte Einheit: SI-Wert und Einheit ''.
        """
        unit_value = self.current_unit_values.get(var) if UNITS_AVAILABLE else None
        if not (unit_value and unit_value.original_unit):
            return val, ''
        unit = self._display_unit_for(unit_value)
        if var in getattr(self, '_kelvin_display', ()):
            unit = 'K'   # in K eingegeben, Charakter nicht bestimmbar: wie eingegeben
        if isinstance(val, np.ndarray):
            return np.array([self._si_to_unit(v, unit) if np.isfinite(v) else np.nan
                             for v in val]), unit
        return self._si_to_unit(val, unit), unit

    def clear_results(self):
        """Löscht die Ergebnisanzeige."""
        # Status zurücksetzen
        self.result_status_label.configure(text="", text_color=COLORS["text_dim"])
        self.unit_warning_label.configure(text="")  # Unit Warning zurücksetzen
        self.unit_warnings_content = None
        self.hints_label.configure(text="")
        self.hints_content = None
        self._last_results_args = None
        self.result_stats_label.configure(text="")
        self.info_label.configure(text="", text_color=COLORS["text_dim"])

        # Variablen-Zeilen löschen
        for widget in self.var_rows_container.winfo_children():
            widget.destroy()

        # Unit-Referenzen zurücksetzen
        self.value_labels = {}
        self.unit_dropdowns = {}
        self.current_unit_values = {}

        # Residuals Tab zurücksetzen
        for section in self.residuals_sections:
            section.destroy()
        self.residuals_sections = []
        self.residuals_placeholder.grid()

    def _refresh_results(self):
        """Zeigt die letzten Ergebnisse erneut an (z.B. nach Wechsel der Anzeige-Einheit)."""
        if not self._last_results_args or not self.value_labels:
            return
        # Status- und Info-Zeile (evtl. Fehlermeldung) unverändert lassen
        saved = [(label, label.cget("text"), label.cget("text_color"))
                 for label in (self.result_status_label, self.info_label)]
        for widget in self.var_rows_container.winfo_children():
            widget.destroy()
        self._show_results(*self._last_results_args)
        for label, text, color in saved:
            label.configure(text=text, text_color=color)

    def _show_results(self, solution: dict, solve_msg: str, n_eq: int, n_var: int, status: str):
        """Zeigt die Ergebnisse im Results Tab an."""
        self._last_results_args = (solution, solve_msg, n_eq, n_var, status)
        # Status
        if status == "OK":
            self.result_status_label.configure(text="● SOLUTION FOUND", text_color=COLORS["success"])
        else:
            self.result_status_label.configure(text="● PARTIAL SOLUTION", text_color=COLORS["warning"])

        # Stats aus solve_msg extrahieren (falls vorhanden)
        self.result_stats_label.configure(text=solve_msg if len(solve_msg) < 80 else "")

        # Info-Zeile
        status_color = COLORS["success"] if status == "OK" else COLORS["error"]
        # Prüfe ob Parameterstudie (Arrays in Lösung)
        has_arrays = any(isinstance(v, np.ndarray) for v in solution.values())
        if has_arrays:
            n_points = max(len(v) for v in solution.values() if isinstance(v, np.ndarray))
            info_text = f"Parametric Study: {n_points} points          Use Plot menu for visualization"
        else:
            info_text = f"Equations: {n_eq}          Unknowns: {n_var}          Status: {status}"
        self.info_label.configure(text=info_text)

        # Referenzen zurücksetzen
        self.value_labels = {}
        self.unit_dropdowns = {}

        # Variablen-Tabelle
        row_idx = 0
        for var in sorted(solution.keys()):
            val = solution[var]

            # Zeile erstellen
            row = ctk.CTkFrame(self.var_rows_container, fg_color="transparent", height=28)
            row.pack(fill="x", pady=1)

            # Variable Name (intern umbenannte Schlüsselwörter: _kw_lambda -> lambda)
            var_label = ctk.CTkLabel(
                row, text=display_name(var),
                font=ctk.CTkFont(size=12),
                text_color=COLORS["text"],
                anchor="w", width=150
            )
            var_label.pack(side="left", padx=5)

            # Wert und Einheit für die Anzeige - beide aus DERSELBEN Einheit
            display_val, display_unit = self._display_values(var, val)
            has_unit = bool(display_unit)

            if isinstance(val, np.ndarray):
                # Für Arrays: Zeige Bereich (min → max) in der Anzeige-Einheit
                if np.all(np.isnan(display_val)):
                    val_text = f"[{len(val)}× nan]"
                else:
                    # Erster -> letzter Punkt (Reihenfolge der Parameterstudie), nicht min -> max
                    finite = display_val[np.isfinite(display_val)]
                    first_val, last_val = finite[0], finite[-1]
                    if np.all(finite == first_val):
                        val_text = f"[{len(val)}× {first_val:.4g}]"
                    else:
                        val_text = f"[{len(val)}× {first_val:.4g}→{last_val:.4g}]"
                    if finite.size < len(val):
                        val_text += f" ({len(val) - finite.size}× nan)"
            else:
                if abs(display_val) >= 1e6 or (abs(display_val) < 1e-4 and display_val != 0):
                    val_text = f"{display_val:.6e}"
                else:
                    val_text = f"{display_val:.6g}"

            # Unit Dropdown oder Platzhalter (rechts außen, vor Value)
            if has_unit and not isinstance(val, np.ndarray):
                stored = self.current_unit_values.get(var)
                is_difference = stored is not None and stored.original_unit == 'delta_K'
                compatible_units = list(dict.fromkeys(
                    pretty_unit(u) for u in get_compatible_units('delta_K' if is_difference else display_unit)))
                if display_unit not in compatible_units:
                    compatible_units = [display_unit] + list(compatible_units)

                unit_dropdown = ctk.CTkOptionMenu(
                    row,
                    values=compatible_units,
                    width=80,
                    height=24,
                    font=ctk.CTkFont(size=11),
                    fg_color=COLORS["bg_input"],
                    button_color=COLORS["bg_frame"],
                    button_hover_color=COLORS["accent"],
                    dropdown_fg_color=COLORS["bg_frame"],
                    dropdown_hover_color=COLORS["accent"],
                    command=lambda u, v=var: self._on_unit_changed(v, u)
                )
                unit_dropdown.set(display_unit)
                unit_dropdown.pack(side="right", padx=2)
                self.unit_dropdowns[var] = unit_dropdown
            elif isinstance(val, np.ndarray) and UNITS_AVAILABLE:
                # Für Arrays: Zeige Einheit als Label (wenn bekannt), sonst "array"
                unit_label = ctk.CTkLabel(
                    row, text=display_unit if has_unit else "array",
                    font=ctk.CTkFont(size=11),
                    text_color=COLORS["accent"] if has_unit else COLORS["text_dim"],
                    width=80, anchor="center"
                )
                unit_label.pack(side="right", padx=2)
            elif UNITS_AVAILABLE and not isinstance(val, np.ndarray):
                # Dimensionslos: Zahl, %, ‰ oder g/kg (relative Feuchte, Wassergehalt, Wirkungsgrad)
                self.current_unit_values[var] = UnitValue.from_si_base(float(display_val), '')
                unit_dropdown = ctk.CTkOptionMenu(
                    row,
                    values=get_compatible_units(''),
                    width=80,
                    height=24,
                    font=ctk.CTkFont(size=11),
                    fg_color=COLORS["bg_input"],
                    button_color=COLORS["bg_frame"],
                    button_hover_color=COLORS["accent"],
                    dropdown_fg_color=COLORS["bg_frame"],
                    dropdown_hover_color=COLORS["accent"],
                    command=lambda u, v=var: self._on_unit_changed(v, u)
                )
                unit_dropdown.set('-')
                unit_dropdown.pack(side="right", padx=2)
                self.unit_dropdowns[var] = unit_dropdown

            # Value Label
            val_label = ctk.CTkLabel(
                row, text=val_text,
                font=ctk.CTkFont(size=12),
                text_color=COLORS["value"],
                anchor="e",
                width=120
            )
            val_label.pack(side="right", padx=5)
            self.value_labels[var] = val_label

            row_idx += 1

    def _on_unit_changed(self, var: str, new_unit: str):
        """Aktualisiert den angezeigten Wert bei Einheitenänderung."""
        if var not in self.current_unit_values or var not in self.value_labels:
            return

        unit_value = self.current_unit_values[var]
        try:
            # Konvertiere zum neuen Unit ('-' = dimensionslose Zahl)
            if new_unit == '-':
                new_val = unit_value.to('dimensionless')
            else:
                new_val = unit_value.to(new_unit)

            # Formatiere Wert
            if abs(new_val) >= 1e6 or (abs(new_val) < 1e-4 and new_val != 0):
                val_text = f"{new_val:.6e}"
            else:
                val_text = f"{new_val:.6g}"

            # Update Label
            self.value_labels[var].configure(text=val_text)
        except Exception as e:
            print(f"Unit conversion error for {var}: {e}")

    def _show_error(self, message: str, status_text: str = "● ERROR",
                    status_color: Optional[str] = None):
        """Zeigt eine Fehlermeldung im Results Tab an (Info-Zeile, umbrechend)."""
        message = unmangle(message or "")  # _kw_lambda -> lambda
        self.result_status_label.configure(text=status_text,
                                           text_color=status_color or COLORS["error"])
        if len(message) > 300:
            message = message[:297] + "..."
        self.info_label.configure(text=message, text_color=COLORS["error"], wraplength=420)

    def _clear_last_solution(self):
        """Verwirft die letzte Lösung (Plot darf keine alten Daten zeigen)."""
        self.last_solution = None
        self.last_sweep_vars = {}
        self.last_analysis = None

    def clear_all(self):
        """Löscht alle Eingaben und Ausgaben (rückgängig machbar per Undo)."""
        if self._solving:
            return
        self.equations_text.delete("1.0", "end")
        self.clear_results()
        self._clear_last_solution()
        self.status_label.configure(text="Ready")

    # === Dialoge ===

    def show_settings(self):
        """Zeigt den Settings Dialog."""
        dialog = ctk.CTkToplevel(self)
        dialog.title("Settings")
        dialog.geometry("360x520")  # Größer für alle Einheiten-Optionen
        dialog.transient(self)
        dialog.grab_set()

        # Font Size
        ctk.CTkLabel(dialog, text="Font Size:", font=ctk.CTkFont(size=13)).pack(pady=(15, 5))

        font_slider = ctk.CTkSlider(dialog, from_=FONT_SIZE_MIN, to=FONT_SIZE_MAX,
                                     number_of_steps=(FONT_SIZE_MAX - FONT_SIZE_MIN) // 2,
                                     command=lambda v: self.set_font_size(int(v)))
        font_slider.set(self.font_size)
        font_slider.pack(pady=5, padx=20, fill="x")

        # Separator
        separator = ctk.CTkFrame(dialog, height=2, fg_color=COLORS["bg_frame"])
        separator.pack(fill="x", padx=20, pady=15)

        # Temperature Display Unit
        ctk.CTkLabel(dialog, text="Temperature Display:", font=ctk.CTkFont(size=13)).pack(pady=(5, 10))

        temp_unit_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        temp_unit_frame.pack(padx=20, anchor="w")

        temp_radio_k = ctk.CTkRadioButton(
            temp_unit_frame,
            text="Kelvin (K)",
            variable=self.temp_display_unit,
            value="K",
            font=ctk.CTkFont(size=12),
            fg_color=COLORS["accent"]
        )
        temp_radio_k.pack(side="left", padx=(0, 20))

        temp_radio_c = ctk.CTkRadioButton(
            temp_unit_frame,
            text="Celsius (°C)",
            variable=self.temp_display_unit,
            value="degC",
            font=ctk.CTkFont(size=12),
            fg_color=COLORS["accent"]
        )
        temp_radio_c.pack(side="left")

        ctk.CTkLabel(
            dialog,
            text="Default unit for temperature results\n(can still be changed per variable)",
            font=ctk.CTkFont(size=10),
            text_color=COLORS["text_dim"],
            justify="left"
        ).pack(padx=40, anchor="w", pady=(2, 0))

        # Separator
        separator2 = ctk.CTkFrame(dialog, height=2, fg_color=COLORS["bg_frame"])
        separator2.pack(fill="x", padx=20, pady=15)

        # Pressure Display Unit
        ctk.CTkLabel(dialog, text="Pressure Display:", font=ctk.CTkFont(size=13)).pack(pady=(5, 10))

        pressure_unit_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        pressure_unit_frame.pack(padx=20, anchor="w")

        ctk.CTkRadioButton(
            pressure_unit_frame, text="Pa", variable=self.pressure_display_unit,
            value="Pa", font=ctk.CTkFont(size=12), fg_color=COLORS["accent"]
        ).pack(side="left", padx=(0, 20))

        ctk.CTkRadioButton(
            pressure_unit_frame, text="bar", variable=self.pressure_display_unit,
            value="bar", font=ctk.CTkFont(size=12), fg_color=COLORS["accent"]
        ).pack(side="left")

        # Energy Display Unit
        ctk.CTkLabel(dialog, text="Energy Display:", font=ctk.CTkFont(size=13)).pack(pady=(15, 10))

        energy_unit_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        energy_unit_frame.pack(padx=20, anchor="w")

        ctk.CTkRadioButton(
            energy_unit_frame, text="J (J/kg, J/(kg·K))", variable=self.energy_display_unit,
            value="J", font=ctk.CTkFont(size=12), fg_color=COLORS["accent"]
        ).pack(side="left", padx=(0, 20))

        ctk.CTkRadioButton(
            energy_unit_frame, text="kJ (kJ/kg, kJ/(kg·K))", variable=self.energy_display_unit,
            value="kJ", font=ctk.CTkFont(size=12), fg_color=COLORS["accent"]
        ).pack(side="left")

        # Power Display Unit
        ctk.CTkLabel(dialog, text="Power Display:", font=ctk.CTkFont(size=13)).pack(pady=(15, 10))

        power_unit_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        power_unit_frame.pack(padx=20, anchor="w")

        ctk.CTkRadioButton(
            power_unit_frame, text="W", variable=self.power_display_unit,
            value="W", font=ctk.CTkFont(size=12), fg_color=COLORS["accent"]
        ).pack(side="left", padx=(0, 20))

        ctk.CTkRadioButton(
            power_unit_frame, text="kW", variable=self.power_display_unit,
            value="kW", font=ctk.CTkFont(size=12), fg_color=COLORS["accent"]
        ).pack(side="left")

        # Close Button
        ctk.CTkButton(dialog, text="Close", command=dialog.destroy).pack(pady=20)

    def show_initial_values_dialog(self):
        """Zeigt Dialog für manuelle Startwerte mit Einheiten-Anzeige."""
        if not self.known_variables:
            messagebox.showinfo("Initial Values", "Please run Solve first to detect variables.")
            return
        # Aktuellen Stand des Blocks {$Startwerte ... $} lesen (auch von Hand geändert)
        try:
            block_entries = start_value_entries(self.equations_text.get("1.0", "end-1c"))
        except Exception as exc:
            messagebox.showerror("Initial Values", str(exc))
            return
        self.manual_initial_values = {name: value for name, (value, _) in block_entries.items()}

        dialog = ctk.CTkToplevel(self)
        dialog.title("Initial Values")
        dialog.geometry("600x550")
        dialog.transient(self)
        dialog.grab_set()

        # Info
        ctk.CTkLabel(
            dialog,
            text="Initial values in SI units (K, Pa, J/kg, ...) or with unit (15 °C, 2 bar).\n"
                 "Grey values are automatic and are not stored; type a value to override.\n"
                 "OK writes the values into the sheet as block {$Startwerte ... $} (saved with the file).",
            font=ctk.CTkFont(size=12),
            text_color=COLORS["text_dim"]
        ).pack(pady=10)

        # Header Row
        header_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        header_frame.pack(fill="x", padx=10, pady=(0, 5))
        ctk.CTkLabel(header_frame, text="Variable", width=120, anchor="w",
                    font=ctk.CTkFont(weight="bold")).pack(side="left")
        ctk.CTkLabel(header_frame, text="Initial Value", width=110, anchor="w",
                    font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        ctk.CTkLabel(header_frame, text="Unit", width=100, anchor="w",
                    font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        ctk.CTkLabel(header_frame, text="Status", width=80, anchor="w",
                    font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)

        # Scrollable Frame für Variablen
        scroll_frame = ctk.CTkScrollableFrame(dialog, fg_color="transparent")
        scroll_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Sammle alle Variablen (bekannte + ungelöste)
        all_vars = set(self.known_variables)

        # Hole abgeleitete Einheiten (falls vorhanden)
        inferred_units = getattr(self, 'inferred_units', {})

        # Hole gelöste Variablen
        solved_vars = set(self.last_solution.keys()) if self.last_solution else set()

        entries = {}
        units_of = {}     # Einheit je Variable (None = unbekannt)
        auto_texts = {}   # automatisch (grau) eingetragener Text je Variable
        shown_texts = {}  # angezeigter Text gespeicherter Startwerte (unverändert -> Blockzeile bleibt)

        for var in sorted(all_vars):
            row = ctk.CTkFrame(scroll_frame, fg_color="transparent")
            row.pack(fill="x", pady=2)

            # Variable name
            ctk.CTkLabel(row, text=f"{display_name(var)}:", width=120, anchor="w").pack(side="left")

            # Initial value entry
            entry = ctk.CTkEntry(row, width=100)
            entry.pack(side="left", padx=5)

            # Einheit (SI-Recheneinheit, nur Anzeige)
            unit = inferred_units.get(var)
            if unit is None and var in self.current_unit_values:
                unit = self.current_unit_values[var].calc_unit
            units_of[var] = unit

            # Bestimme Standardwert
            if var in self.manual_initial_values:
                shown_texts[var] = f"{self.manual_initial_values[var]:.10g}"
                entry.insert(0, shown_texts[var])
            elif unit is not None:
                # Zeige automatischen Startwert basierend auf Einheit (grau).
                # Wird beim OK NICHT als manueller Wert übernommen.
                auto_value = getattr(self, 'auto_initial_values', {}).get(var, get_initial_from_unit(unit))
                auto_texts[var] = f"{auto_value:.6g}"
                entry.insert(0, auto_texts[var])
                entry.configure(text_color=COLORS["text_dim"])

            # Sobald der Benutzer tippt, normale Textfarbe
            entry.bind("<Key>", lambda e, ent=entry: ent.configure(text_color=COLORS["text"]))
            entries[var] = entry

            if unit is None:
                unit_text = "???"
            else:
                unit_text = pretty_unit(unit) if unit else "-"
            ctk.CTkLabel(row, text=unit_text, width=100, anchor="w",
                         text_color=COLORS["accent"]).pack(side="left", padx=5)

            # Status indicator
            if var in solved_vars:
                status_text = "solved"
                status_color = COLORS["success"]
            else:
                status_text = "unsolved"
                status_color = COLORS["warning"]

            ctk.CTkLabel(row, text=status_text, width=80, anchor="w",
                        text_color=status_color).pack(side="left", padx=5)

        # Buttons
        btn_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=10)

        def apply_values():
            # Erst vollständig validieren, dann übernehmen (bei Fehler bleiben
            # die bisherigen Startwerte erhalten)
            new_values = {}
            lines = []
            for var, entry in entries.items():
                val_str = entry.get().strip()
                # Leere Felder und unveränderte automatische (graue) Werte sind
                # KEINE manuellen Startwerte - sonst würden sie eingefroren
                if not val_str or val_str == auto_texts.get(var):
                    continue
                if val_str == shown_texts.get(var) and var in block_entries:
                    # Unverändert: Angabe im Blatt bleibt wie geschrieben (15 °C)
                    new_values[var], line = block_entries[var]
                    lines.append(line)
                    continue
                try:
                    _, new_values[var] = parse_start_value(f"{display_name(var)} = {val_str}")
                except Exception:
                    messagebox.showerror("Error", f"Invalid value for {display_name(var)}: '{val_str}'")
                    return
                lines.append(self._start_value_line(var, val_str, new_values[var], units_of.get(var)))
            # Startwerte im Block für Namen, die gerade nicht im Dialog stehen, bleiben
            for var, (value, line) in block_entries.items():
                if var not in entries:
                    new_values[var] = value
                    lines.append(line)

            self.manual_initial_values = new_values
            self._write_start_values_block(lines)
            dialog.destroy()
            self.status_label.configure(text=f"{len(self.manual_initial_values)} initial values set")

        def clear_all():
            auto_texts.clear()
            for e in entries.values():
                e.delete(0, "end")
                e.configure(text_color=COLORS["text"])

        def auto_fill():
            """Füllt alle leeren Felder mit automatischen Werten basierend auf Einheiten."""
            for var, entry in entries.items():
                if not entry.get().strip() and units_of.get(var) is not None:
                    auto_value = getattr(self, 'auto_initial_values', {}).get(
                        var, get_initial_from_unit(units_of[var]))
                    auto_texts[var] = f"{auto_value:.6g}"
                    entry.insert(0, auto_texts[var])
                    entry.configure(text_color=COLORS["text_dim"])

        ctk.CTkButton(btn_frame, text="Clear All", command=clear_all).pack(side="left")
        ctk.CTkButton(btn_frame, text="Auto-Fill", command=auto_fill,
                     fg_color=COLORS["accent"]).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="Cancel", command=dialog.destroy).pack(side="right", padx=5)
        ctk.CTkButton(btn_frame, text="OK", command=apply_values).pack(side="right")

    @staticmethod
    def _start_value_line(var: str, val_str: str, value: float, unit: Optional[str]) -> str:
        """
        Zeile im Startwerte-Block: mit Einheit eingetippte Werte bleiben wie
        eingegeben (15 °C), reine Zahlen sind SI und erhalten die SI-Einheit
        zur Lesbarkeit - nur wenn sie beim Wiedereinlesen denselben Wert ergibt.
        """
        name = display_name(var)
        try:
            float(val_str)
        except ValueError:
            return f"{name} = {val_str}"
        if unit:
            line = f"{name} = {value:.10g} {pretty_unit(unit.replace('delta_K', 'K'))}"
            try:
                if math.isclose(parse_start_value(line)[1], value, rel_tol=1e-9, abs_tol=1e-300):
                    return line
            except Exception:
                pass
        return f"{name} = {value:.10g}"

    def _write_start_values_block(self, lines: List[str]) -> None:
        """
        Schreibt den Block {$Startwerte ... $} ins Blatt (ersetzt, ergänzt am Ende
        oder entfernt ihn) - als EIN Undo-Schritt, der Rest des Texts bleibt unberührt.
        """
        text = self.equations_text.get("1.0", "end-1c")
        start, end, new = start_values_edit(text, lines)
        if text[start:end] == new:
            return

        def index(pos: int) -> str:
            line = text.count('\n', 0, pos) + 1
            return f"{line}.{pos - (text.rfind(chr(10), 0, pos) + 1)}"

        textbox = self.equations_text._textbox
        textbox.configure(autoseparators=False)
        try:
            textbox.edit_separator()
            start_index, end_index = index(start), index(end)
            if end > start:
                textbox.delete(start_index, end_index)
            if new:
                textbox.insert(start_index, new)
            textbox.edit_separator()
        finally:
            textbox.configure(autoseparators=True)

    def show_plot_dialog(self, multi: bool = True):
        """
        Plot-Dialog für Parameterstudien.

        multi=True  (New Plot Window): mehrere Y-Variablen, Titel, Gitter/Marker/Legende
        multi=False (Quick Plot X-Y):  eine X- und eine Y-Variable
        Daten und Achsen in den Anzeige-Einheiten (z.B. °C, bar).
        """
        if not MATPLOTLIB_AVAILABLE:
            messagebox.showerror("Error", "matplotlib not available")
            return

        if self.last_solution is None:
            messagebox.showinfo("Plot", "Please run Solve first.")
            return

        has_arrays = any(isinstance(v, np.ndarray) for v in self.last_solution.values())
        if not has_arrays:
            messagebox.showinfo("Plot", "Plot requires parametric study with vector data.")
            return

        array_vars = sorted([k for k, v in self.last_solution.items() if isinstance(v, np.ndarray)])
        shown_names = {display_name(v): v for v in array_vars}  # Anzeige -> intern

        dialog = ctk.CTkToplevel(self)
        dialog.title("New Plot" if multi else "Quick Plot X-Y")
        dialog.geometry("420x620" if multi else "400x400")
        dialog.transient(self)
        dialog.grab_set()

        # X-Achse
        ctk.CTkLabel(dialog, text="X-Axis:", font=ctk.CTkFont(size=13)).pack(pady=(20, 5))
        x_var = ctk.StringVar(value=display_name(array_vars[0]))
        x_combo = ctk.CTkComboBox(dialog, variable=x_var, values=list(shown_names), width=220)
        x_combo.pack()

        y_checks = {}
        y_var = ctk.StringVar(value=display_name(array_vars[1] if len(array_vars) > 1 else array_vars[0]))
        if multi:
            # Mehrere Y-Variablen per Checkbox
            ctk.CTkLabel(dialog, text="Y-Axis (one or more):", font=ctk.CTkFont(size=13)).pack(pady=(15, 5))
            y_frame = ctk.CTkScrollableFrame(dialog, width=260, height=180)
            y_frame.pack()
            for index, name in enumerate(shown_names):
                var = ctk.BooleanVar(value=(index == 1 or len(shown_names) == 1))
                ctk.CTkCheckBox(y_frame, text=name, variable=var).pack(anchor="w", pady=2)
                y_checks[name] = var
            ctk.CTkLabel(dialog, text="Title (optional):", font=ctk.CTkFont(size=13)).pack(pady=(15, 5))
            title_entry = ctk.CTkEntry(dialog, width=260)
            title_entry.pack()
        else:
            ctk.CTkLabel(dialog, text="Y-Axis:", font=ctk.CTkFont(size=13)).pack(pady=(20, 5))
            ctk.CTkComboBox(dialog, variable=y_var, values=list(shown_names), width=220).pack()
            title_entry = None

        # Optionen
        options = ctk.CTkFrame(dialog, fg_color="transparent")
        options.pack(pady=15)
        grid_var = ctk.BooleanVar(value=True)
        marker_var = ctk.BooleanVar(value=False)
        legend_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(options, text="Grid", variable=grid_var).pack(side="left", padx=5)
        if multi:
            ctk.CTkCheckBox(options, text="Markers", variable=marker_var).pack(side="left", padx=5)
            ctk.CTkCheckBox(options, text="Legend", variable=legend_var).pack(side="left", padx=5)

        def create_plot():
            x_name = shown_names.get(x_var.get())
            if multi:
                y_names = [shown_names[n] for n, v in y_checks.items() if v.get()]
            else:
                y_names = [shown_names[y_var.get()]] if y_var.get() in shown_names else []
            if not x_name or not y_names:
                return

            # Daten in Anzeige-Einheiten (z.B. °C statt K), Achsen mit Einheit
            x_data, x_unit = self._display_values(x_name, self.last_solution[x_name])
            series = []
            units = set()
            for name in y_names:
                y_data, y_unit = self._display_values(name, self.last_solution[name])
                units.add(y_unit)
                label = display_name(name) + (f" [{y_unit}]" if y_unit else "")
                series.append((label, y_data))
            x_label = display_name(x_name) + (f" [{x_unit}]" if x_unit else "")
            if len(series) == 1:
                y_label = series[0][0]
            elif len(units) == 1 and next(iter(units)):
                y_label = f"[{next(iter(units))}]"
            else:
                y_label = ""
            title = title_entry.get().strip() if title_entry is not None else ""
            if not title:
                title = f"{', '.join(display_name(n) for n in y_names)} vs {display_name(x_name)}"

            self._create_plot_window(x_data, series, x_label, y_label, title, grid_var.get(),
                                     legend_var.get() and len(series) > 1, marker_var.get())
            dialog.destroy()

        ctk.CTkButton(dialog, text="Plot", command=create_plot).pack(pady=10)

    def show_quick_plot_dialog(self):
        """Vereinfachter Plot-Dialog (eine X- und eine Y-Variable)."""
        self.show_plot_dialog(multi=False)

    def _create_plot_window(self, x_data, y_data_list, x_label="", y_label="", title="",
                            show_grid=True, show_legend=True, show_markers=False):
        """Erstellt ein Plot-Fenster."""
        if not MATPLOTLIB_AVAILABLE:
            return

        plot_window = ctk.CTkToplevel(self)
        plot_window.title(f"Plot: {title}" if title else "Plot")
        plot_window.geometry("800x600")

        fig = Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        marker = 'o' if show_markers else None

        for i, (name, y_data) in enumerate(y_data_list):
            ax.plot(x_data, y_data, label=name, color=colors[i % len(colors)], marker=marker, markersize=4)

        if x_label:
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_ylabel(y_label)
        if title:
            ax.set_title(title)
        if show_grid:
            ax.grid(True, linestyle='--', alpha=0.7)
        if show_legend and len(y_data_list) > 1:
            ax.legend()

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()

        # Toolbar
        toolbar_frame = ctk.CTkFrame(plot_window, fg_color="transparent")
        toolbar_frame.pack(side="top", fill="x")
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()

        canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    def show_function_help(self):
        """Zeigt Funktions-Hilfe."""
        dialog = ctk.CTkToplevel(self)
        dialog.title("Function Reference")
        dialog.geometry("700x750")

        text = ctk.CTkTextbox(dialog, font=ctk.CTkFont(family="Courier", size=11))
        text.pack(fill="both", expand=True, padx=10, pady=10)

        text.insert("1.0", FUNCTION_HELP_TEXT)
        text.configure(state="disabled")

    def show_fluid_help(self):
        """Zeigt Fluid-Liste."""
        dialog = ctk.CTkToplevel(self)
        dialog.title("Available Fluids")
        dialog.geometry("620x600")

        text = ctk.CTkTextbox(dialog, font=ctk.CTkFont(family="Courier", size=11))
        text.pack(fill="both", expand=True, padx=10, pady=10)

        help_text = fluid_help_text()
        text.insert("1.0", help_text)
        text.configure(state="disabled")

    def _insert_example(self):
        """Fügt ein Beispiel ein."""
        if self._solving:
            return
        if THERMO_AVAILABLE:
            example = '''"HVAC Equation Solver - Example"
"Internal units (SI): T[K], p[Pa], h[J/kg], s[J/(kg*K)]"

{--- Example 1: Water/Steam ---}
T_1 = 150 °C
p_1 = 5 bar
h_1 = enthalpy(water, T=T_1, p=p_1)
s_1 = entropy(water, T=T_1, p=p_1)

{Saturated steam at same pressure}
x_sat = 1
T_sat = temperature(water, p=p_1, x=x_sat)
h_sat = enthalpy(water, p=p_1, x=x_sat)

{--- Example 2: Humid Air ---}
T_air = 25 °C
rh = 0.6
p_tot = 1 bar

h_air = HumidAir(h, T=T_air, rh=rh, p_tot=p_tot)
w = HumidAir(w, T=T_air, rh=rh, p_tot=p_tot)
T_dp = HumidAir(T_dp, T=T_air, rh=rh, p_tot=p_tot)
T_wb = HumidAir(T_wb, T=T_air, rh=rh, p_tot=p_tot)

{--- Example 3: Thermal Radiation ---}
T_surface = 500 °C
epsilon = 0.85
A = 2 m^2
sigma = 5.67E-8 W/(m^2*K^4)

{Stefan-Boltzmann radiation}
Q_rad = epsilon * sigma * A * T_surface^4

{Peak wavelength (Wien's law)}
lambda_max = Wien(T_surface)

{--- Example 4: Heating curve and heat output ---}
T_a = -10 °C
{The heating curve holds for numbers in °C: value() and quantity()}
T_VL = quantity(20 + 1.5*(20 - value(T_a, °C)), °C)
sigma_w = 20 K              {temperature spread: differences always in K}
T_RL = T_VL - sigma_w
m_dot_w = 0.5 kg/s
c_w = 4.19 kJ/(kg*K)
Q_dot_H = m_dot_w*c_w*(T_VL - T_RL)

{--- Example 5: Pipe flow, laminar or turbulent (IF) ---}
v_m = 1.5 m/s
d_i = 20 mm
nu_w = 1.0E-6 m^2/s
lambda_w = 0.6 W/(m*K)
Pr_w = 7
Re = v_m*d_i/nu_w
Nu = IF(Re, 2300, 3.66, 3.66, 0.023*Re^0.8*Pr_w^0.4)   {laminar below 2300}
alpha_i = Nu*lambda_w/d_i

{--- Example 6: Economic insulation thickness (optimisation) ---}
MINIMIZE K_tot VARY s_ins = 0.01 .. 0.4 m
lambda_ins = 0.035 W/(m*K)
R_wall = 0.5 m^2*K/W        {wall without insulation}
dT_m = 15 K                 {mean temperature difference, heating season}
t_H = 5000 h                {heating hours per year}
k_E = 0.10                  {energy price per kWh}
k_ins = 120                 {insulation price per m3}
a_n = 0.08                  {annuity factor per year}
U = 1/(R_wall + s_ins/lambda_ins)
Q_a = U*dT_m*t_H            {heat loss per m2 and year}
K_E = k_E*value(Q_a, kWh/m^2)
K_ins = a_n*k_ins*value(s_ins, m)
K_tot = K_E + K_ins         {annual cost per m2}

{--- Example 7: Refrigeration cycle NH3, reference state IIR ---}
REFERENCE R717 IIR          {h = 200 kJ/kg, s = 1 kJ/(kg*K): sat. liquid 0 °C}
T_0 = -10 °C                {evaporation}
T_c = 35 °C                 {condensation}
h_r1 = enthalpy(R717, T=T_0, x=1)
s_r1 = entropy(R717, T=T_0, x=1)
p_c = pressure(R717, T=T_c, x=0)
h_r2s = enthalpy(R717, p=p_c, s=s_r1)
h_r3 = enthalpy(R717, T=T_c, x=0)
EER = (h_r1 - h_r3)/(h_r2s - h_r1)
EER_C*(T_c - T_0) = T_0     {Carnot in kelvin, also implicit}

"Press F5 to solve. Help > Function Reference describes all functions."
'''
        else:
            example = '''"Example: Nonlinear equation system"
"Right triangle calculation"

{Given values}
a = 3
b = 4

{Pythagorean theorem}
c^2 = a^2 + b^2

{Calculate angles}
tan(alpha) = a / b
alpha + beta = 90

"Press F5 to solve"
'''
        self.equations_text.delete("1.0", "end")
        self.equations_text.insert("1.0", example)
        self.clear_results()
        self._clear_last_solution()


def main():
    """Hauptfunktion."""
    app = EquationSolverApp()
    app.mainloop()


if __name__ == "__main__":
    main()
