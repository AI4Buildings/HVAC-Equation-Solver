"""
Regressionstests für den HVAC Equation Solver.

Deckt die im Code-Review (Juli 2026) gefundenen und behobenen Fehler ab.
Ausführen mit:  python3 test_regressions.py
"""
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from parser import parse_equations, parse_vector, remove_comments, tokenize_equation, extract_variables
from solver import solve_system, solve_parametric, _get_equation_unknowns
from unit_constraints import analyze_equation, check_equation_dimensions, check_all_unit_consistency
from units import UnitValue

PASSED = []
FAILED = []


def check(name, cond, extra=""):
    (PASSED if cond else FAILED).append(name)
    print(("OK  " if cond else "FAIL"), name, extra)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
print("=== Parser ===")

# Umgestellte Gleichungen dürfen NICHT als Konstante interpretiert werden
for text in ("x + 5 = 2", "sin(alpha) = 0.5", "2*pi*r = 10", "x^2 = 9",
             "m_dot*3600 = 500", "ln(x) = 2", "1/R = 0.25"):
    eqs, _, consts, _, _, _ = parse_equations(text)
    check(f"'{text}' ist Gleichung", len(eqs) == 1 and not consts)

# Echte Direktzuweisungen bleiben Konstanten
for text, expected in (("T1 = 300", 300.0), ("m = 10000/3600", 10000 / 3600),
                       ("alpha = asin(0.5)", 30.0)):
    eqs, _, consts, _, _, _ = parse_equations(text)
    val = list(consts.values())[0] if consts else None
    check(f"'{text}' ist Konstante", not eqs and val is not None and abs(val - expected) < 1e-9)

# Ausdruck + Einheit (CLAUDE.md-Beispiel)
eqs, _, consts, _, _, uv = parse_equations("m_dot_1 = 10000/3600 kg/s")
check("Ausdruck+Einheit 'kg/s'", not eqs and abs(consts.get('m_dot_1', 0) - 2.7778) < 1e-3
      and uv.get('m_dot_1') is not None)

# Vektor-Semantik (MATLAB start:step:end, Schrittweite nie verfälscht)
check("0:0.3:1 -> [0,0.3,0.6,0.9]", np.allclose(parse_vector("0:0.3:1"), [0, 0.3, 0.6, 0.9]))
check("25:5:50 inkl. Endwert", np.allclose(parse_vector("25:5:50"), [25, 30, 35, 40, 45, 50]))
check("0:10:95 ohne 95", np.allclose(parse_vector("0:10:95"), np.arange(0, 91, 10)))

# Sweep mit °C: Offset-Konvertierung elementweise
_, _, _, sweeps, _, _ = parse_equations("T = 20:10:50 °C")
check("Sweep 20:10:50 °C -> K", np.allclose(sweeps['T'], [293.15, 303.15, 313.15, 323.15]))

# Temperaturdifferenz in °C ohne Offset
_, _, consts, _, _, uv = parse_equations("dT_1 = 10 °C")
check("dT_1 = 10 °C -> 10 delta_K", abs(consts.get('dT_1', 0) - 10) < 1e-9
      and uv['dT_1'].original_unit == 'delta_K')

# Verschachtelte Kommentare
check("Kommentare verschachtelt", remove_comments("{a {b} c} x = 5").strip() == "x = 5")

# Verschachtelte Thermo-Aufrufe
eq = tokenize_equation("h_2 = enthalpy(water, T=temperature(water, p=p1, s=s1), p=p2)")
check("Verschachtelte Calls tokenisiert", "'water'" in eq and eq.count("'water'") == 2)
check("Verschachtelte Calls Variablen", extract_variables(eq) == {'h_2', 'p1', 's1', 'p2'})

# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
print("=== Solver ===")

s, sol, msg = solve_system(["(1/(x-2)) - (0)"], {'x'})
check("Divergenz zur Asymptote abgelehnt", not s)

s, sol, msg = solve_system(["(x + 1) - (3)", "(x + 1) - (4)"], {'x'})
check("Widersprüchliches System erkannt", not s and "Widerspr" in msg)

s, sol, msg = solve_system(["(a*b) - (6)", "(x+y+z)-(6)", "(x-y+2*z)-(5)", "(2*x+y-z)-(1)"],
                           {'a', 'b', 'x', 'y', 'z'})
check("Lösbarer Block trotz unterbestimmtem Block", abs(sol.get('x', 0) - 1) < 1e-6
      and abs(sol.get('z', 0) - 3) < 1e-6)

s, sol, msg = solve_parametric(["(y) - (s * z)", "(z) - (2)"], {'y', 'z'},
                               {'s': np.array([1., 2., 3.])}, initial_values={'z': 7.0})
check("Sweep: Startwerte fließen nicht ins Ergebnis", np.allclose(sol['y'], [2, 4, 6]))

s, sol, msg = solve_system(["(x**2 - 2*x) - (5)"], {'x'}, initial_values={'x': 1.0})
check("Wurzel nahe Startwert gewählt", abs(float(sol['x']) - 3.44949) < 1e-3)

s, sol, msg = solve_parametric(["(x**2 - 2*x) - (a)"], {'x'}, {'a': np.array([1., 2., 3., 4., 5.])})
check("Sweep: kein Lösungsast-Sprung (Warm-Start)", np.all(sol['x'] > 0))

s, sol, msg = solve_system(["((x-3)*exp(-(x-3)**2)) - (0)"], {'x'}, initial_values={'x': 50.0})
check("Underflow-Plateau nicht als Wurzel", s and abs(float(sol['x']) - 3) < 1e-6)

t0 = time.monotonic()
s, sol, msg = solve_system(["(exp(x) + 1) - (0)"], {'x'})
check("Unlösbare Gleichung -> schnell False", (not s) and time.monotonic() - t0 < 15)

u = _get_equation_unknowns("(h_2) - (HumidAir('h', T=T_1, rh=0.5, p_tot=p))",
                           set(), {'h_2', 'T', 'rh', 'p', 'T_1'})
check("kwargs/Strings zählen nicht als Unbekannte", u == {'h_2', 'T_1', 'p'})

# Basisfälle
s, sol, _ = solve_system(["(x + y) - (10)", "(x - y) - (2)"], {'x', 'y'})
check("Lineares 2x2", s and abs(sol['x'] - 6) < 1e-9)
s, sol, _ = solve_system(["(x + 5) - (2)"], {'x'})
check("Negative Lösung", s and abs(sol['x'] + 3) < 1e-9)
s, sol, _ = solve_system(["(x**2) - (9)"], {'x'})
check("x^2 = 9 -> 3", s and abs(abs(float(sol['x'])) - 3) < 1e-6)
s, sol, _ = solve_system(["(x + y + z) - (6)", "(x*y + z) - (5)", "(x*y*z) - (6)"], {'x', 'y', 'z'})
check("Nichtlinearer 3er-Block", s)

# ---------------------------------------------------------------------------
# Einheiten
# ---------------------------------------------------------------------------
print("=== Einheiten ===")

r = analyze_equation('x^2 = 4', {})
check("Potenz: dimensionslos inferiert", r.get('x') == '')
r = analyze_equation('x*ln(x) = r', {'r': ''})
check("ln bekannt im Dimension-Inferrer", r.get('x') == '')

e = check_equation_dimensions("F = m * g", {'F': 'W', 'm': 'kg', 'g': 'm/s^2'})
check("F=m*g mit F:W -> Dimensionsfehler", e is not None)
e = check_equation_dimensions("F = m * g", {'F': 'N', 'm': 'kg', 'g': 'm/s^2'})
check("F=m*g mit F:N -> konsistent", e is None)
e = check_equation_dimensions("E_ges = e*A", {'E_ges': 'W', 'e': 'W/m^2', 'A': 'm^2'})
check("Variable 'e' kein Falsch-Positiv", e is None)

w = check_all_unit_consistency({'x': 2.86, 'r': 3.0}, {'(x*log(x)) - (r)': 'x*ln(x) = r'}, {'r': ''})
check("Einheitenloses System -> keine Unit-Warnungen", w == [])

r = analyze_equation('theta = T_1 - T_2', {'T_1': 'K', 'T_2': 'K'})
check("theta = T1-T2 -> delta_K", r.get('theta') == 'delta_K')
r = analyze_equation('T_3 = (T_1 + T_2)/2', {'T_1': 'K', 'T_2': 'K'})
check("Mitteltemperatur bleibt K", r.get('T_3') in ('K', 'kelvin'))

uv = UnitValue.from_input(10, 'delta_K')
check("delta_K -> °C ohne Offset", abs(uv.to('°C') - 10) < 0.01)
uv = UnitValue.from_si_base(350, 'degC')
check("from_si_base(350,degC).to(°C) = 76.85", abs(uv.to('°C') - 76.85) < 0.01)
uv = UnitValue.from_si_base(300, '°F')
check("from_si_base(300,°F) = 80.33 °F", abs(uv.original_value - 80.33) < 0.01)
uv = UnitValue.from_input(90, '°C')
check("90 °C -> 363.15 K", abs(uv.calc_value - 363.15) < 0.01)

# ---------------------------------------------------------------------------
# End-to-End (Parser + Solver)
# ---------------------------------------------------------------------------
print("=== End-to-End ===")

text = """
sin(alpha) = 0.5
F = 100
Fx = F*cos(alpha)
"""
eqs, variables, consts, _, orig, _ = parse_equations(text)
s, sol, msg = solve_system(eqs, variables, constants=consts, original_equations=orig)
check("sin(alpha)=0.5 -> alpha=30, Fx=86.6",
      s and abs(sol.get('alpha', 0) - 30) < 1e-6 and abs(sol.get('Fx', 0) - 86.6025) < 1e-3)

text = """
T_h_in = 363.15
T_c_in = 293.15
m_h = 2
m_c = 3
c_p = 4186
U = 800
A = 15
Q = m_h*c_p*(T_h_in - T_h_out)
Q = m_c*c_p*(T_c_out - T_c_in)
Q = U*A*((T_h_in - T_c_out) - (T_h_out - T_c_in))/ln((T_h_in - T_c_out)/(T_h_out - T_c_in))
"""
eqs, variables, consts, _, orig, _ = parse_equations(text)
s, sol, msg = solve_system(eqs, variables, constants=consts, original_equations=orig)
check("LMTD-Wärmeübertrager (einheitenlos)",
      s and abs(sol.get('Q', 0) - 379505) < 10)

print()
print(f"{len(PASSED)}/{len(PASSED) + len(FAILED)} Tests bestanden")
if FAILED:
    print("FEHLGESCHLAGEN:", *FAILED, sep="\n  - ")
    sys.exit(1)
print("ALLE TESTS OK")
