"""
Regressionstests für den HVAC Equation Solver.

Deckt die in den Code-Reviews (Juli und Oktober 2026) gefundenen und
behobenen Fehler ab. Weitere Testdateien:
  test_unit_constraints.py  Einheiten-Propagation / Dimensionsprüfung
  test_berechnungen.py      Berechnungsaufgaben Thermodynamik/Wärmeübertragung
  test_gui.py               GUI (headless)
Ausführen mit:  python3 test_regressions.py
"""
import os
import subprocess
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from parser import (parse_equations, parse_vector, remove_comments, tokenize_equation, extract_variables,
                    display_name)
from solver import solve_system, solve_parametric, _get_equation_unknowns, _find_tear_candidates
from unit_constraints import analyze_equation, check_equation_dimensions, check_all_unit_consistency
from units import UnitValue, get_initial_from_unit, detect_unit_from_equation
from radiation import Eb, Wien_displacement, Blackbody

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
# Oktober 2026: SI-Umrechnung, Startwerte, Schlüsselwörter, Strahlung, Solver
# ---------------------------------------------------------------------------
print("=== Oktober 2026 ===")

# Jede Einheit wird nach SI umgerechnet (früher: cm², L, kW/m², ... unverändert)
for value, unit, si in ((50, 'cm^2', 0.005), (2, 'L', 0.002), (1, 'kW/m^2', 1000.0),
                        (1, 'mPa*s', 1e-3), (0.025, 'kW/(m^2*K)', 25.0), (15, 'mm^2/s', 1.5e-5),
                        (50, 'µm', 5e-5), (500, 'nm', 5e-7), (8, 'g/kg', 0.008), (2, 'MW', 2e6)):
    uv = UnitValue.from_input(value, unit)
    check(f"{value} {unit} -> {si} SI", abs(uv.calc_value - si) <= 1e-9 * abs(si), str(uv.calc_value))
check("dT in delta_K behält calc_unit delta_K", UnitValue.from_input(10, 'delta_K').calc_unit == 'delta_K')
_, _, consts, _, _, _ = parse_equations("A = 50 cm^2\nV = 200 L")
check("Parser: A = 50 cm^2 -> 0.005 m²", abs(consts['A'] - 0.005) < 1e-12)

# Startwerte über die Dimension (früher Teilstring: alles mit 'k' -> 350)
for unit, expected in (('1 / kelvin', 3.4e-3), ('kilogram / second', 1.0), ('W/m^2K', 10.0),
                       ('delta_K', 10.0), ('kW', 1e5), ('kg/kg', 0.01), ('um', 5e-6), ('K', 350.0)):
    check(f"Startwert für '{unit}' = {expected}", get_initial_from_unit(unit) == expected,
          str(get_initial_from_unit(unit)))

# Funktionseinheit nur bei reinem Funktionsaufruf auf der rechten Seite
check("enthalpy() im Ausdruck -> keine J/kg-Einheit",
      detect_unit_from_equation("Q = m*(enthalpy(water, T=T, p=p) - h_1)") == '')
check("temperature() - 273.15 -> keine K-Übernahme",
      detect_unit_from_equation("T_C = temperature(water, p=p, x=1) - 273.15") == '')
check("h = enthalpy(...) -> J/kg", detect_unit_from_equation("h = enthalpy(water, T=T, p=p)") == 'J/kg')

# Python-Schlüsselwörter als Variablennamen (lambda = Wärmeleitfähigkeit)
eqs, variables, consts, _, orig, _ = parse_equations("lambda = 0.6\nd = 0.1\nalpha = 2*lambda/d")
s_, sol, msg = solve_system(eqs, variables, constants=consts, original_equations=orig)
check("lambda als Variable: alpha = 12", s_ and abs(sol.get('alpha', 0) - 12) < 1e-9, msg)
check("display_name('_kw_lambda') == 'lambda'", display_name('_kw_lambda') == 'lambda')
_, _, consts, _, _, _ = parse_equations("d = 2 in")
check("Einheit 'in' (Zoll) wird nicht umbenannt", abs(consts.get('d', 0) - 0.0508) < 1e-12)

# Mehrzeilige Kommentare verschieben die Zuordnung zur Originalzeile nicht
_, _, _, _, orig, _ = parse_equations("{Kommentar\nüber zwei Zeilen}\np = 1e5\nh = enthalpy(water, T=300, p=p)")
check("Mehrzeiliger Kommentar: Originalzeile korrekt",
      list(orig.values()) == ["h = enthalpy(water, T=300, p=p)"], str(list(orig.values())))
check("Kommentare: Zeilenumbrüche bleiben erhalten", remove_comments('{a\nb} x = 1') == '\n x = 1')

# Strahlung: intern SI (Wellenlänge m, Eb W/m³, Wien m); µm-Zahlen weiter erkannt
check("Wien(1000 K) = 2.898e-6 m", abs(Wien_displacement(1000) - 2.897771955e-6) < 1e-15)
check("Eb(1000, 5 µm) in W/m³", abs(Eb(1000, 5e-6) / 7.13962e9 - 1) < 1e-5, str(Eb(1000, 5e-6)))
check("Eb: µm-Zahl == Meter-Wert", Eb(1000, 5) == Eb(1000, 5e-6))
check("Blackbody bis 1000 µm in Metern", abs(Blackbody(5800, 0.75e-6, 1e-3) - Blackbody(5800, 0.75, 1000)) < 1e-12)
check("Einheiten in Strahlungs-Argumenten", tokenize_equation("E = Eb(500°C, 5µm)") == "E = Eb(773.15, 4.9999999999999996e-06)",
      tokenize_equation("E = Eb(500°C, 5µm)"))

# Solver: komplexe Zwischenwerte ((-8)**(1/3)) dürfen nicht abstürzen
try:
    s_, sol, msg = solve_system(["(x**(1/3)) - (-2)"], {'x'})
    check("Komplexe Zwischenwerte: kein Absturz", not s_)
except Exception as exc:
    check("Komplexe Zwischenwerte: kein Absturz", False, repr(exc))

# Tearing: Block aus direkten Zuweisungen + einer Residuengleichung
cands = _find_tear_candidates(["(y) - (exp(x))", "(10) - (y**2 + x)"], {'x', 'y'}, set())
check("Tearing-Kandidat gefunden", bool(cands) and cands[0][2] == "(10) - (y**2 + x)", str(cands))
FILMTEMP = """T_inf = 293.15
H = 0.5
q = 100
p = 100000
T_f = (T_s + T_inf)/2
beta = 1/T_f
nu = viscosity(air, T=T_f, p=p)/density(air, T=T_f, p=p)
k = conductivity(air, T=T_f, p=p)
Pr = prandtl(air, T=T_f, p=p)
Ra = 9.81*beta*(T_s - T_inf)*H^3/nu^2*Pr
Nu = (0.825 + 0.387*Ra^(1/6)/(1 + (0.492/Pr)^(9/16))^(8/27))^2
q = Nu*k/H*(T_s - T_inf)"""
eqs, variables, consts, _, orig, _ = parse_equations(FILMTEMP)
t0 = time.monotonic()
s_, sol, msg = solve_system(eqs, variables, constants=consts, original_equations=orig)
check("Filmtemperatur-Iteration per Tearing (T_s = 316.52 K, < 10 s)",
      s_ and abs(sol.get('T_s', 0) - 316.5195) < 1e-3 and time.monotonic() - t0 < 10, msg)

# Deterministisch: Ergebnis darf nicht vom Hash-Seed abhängen
SEED_SCRIPT = (
    "import sys, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, '.');"
    "from parser import parse_equations; from solver import solve_system;"
    "t = 'p = 1e5\\nT_amb = 293.15\\nm = 0.5\\nUA = 100\\nh_in = 4e5\\nT_out = T_s - 2\\n'"
    "    'h_out = enthalpy(water, T=T_out, p=p)\\nm*(h_in - h_out) = UA*(T_s - T_amb)';"
    "e, v, c, s, o, u = parse_equations(t);"
    "ok, sol, msg = solve_system(e, v, {'T_s': 0.5, 'T_out': 0.5}, constants=c, original_equations=o);"
    "print(ok, round(sol.get('T_s', 0), 6))")
project_dir = os.path.dirname(os.path.abspath(__file__))
outputs = set()
for seed in ('0', '1', '2', '5'):
    res = subprocess.run([sys.executable, '-c', SEED_SCRIPT], capture_output=True, text=True,
                         cwd=project_dir, env={**os.environ, 'PYTHONHASHSEED': seed})
    outputs.add(res.stdout.strip() or res.stderr.strip()[-80:])
check("Gleiches Ergebnis für alle Hash-Seeds (T_s = 367.09 K)",
      outputs == {"True 367.087649"}, str(outputs))

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
