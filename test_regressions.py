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
import re
import subprocess
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from parser import (parse_equations, parse_vector, remove_comments, tokenize_equation, extract_variables,
                    display_name, if_function, parse_start_values, start_values_edit)
from solver import solve_system, solve_parametric, _get_equation_unknowns, _find_tear_candidates
from unit_constraints import (analyze_equation, check_equation_dimensions, check_all_unit_consistency,
                              propagate_all_units_complete)
from units import UnitValue, get_initial_from_unit
from radiation import Eb, Wien_displacement, Blackbody
from diagnostics import analyze_structure, describe_structure, name_hints, diagnose

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
for text, expected in (("T1 = 300", 300.0), ("m = 10000/3600", 10000 / 3600)):
    eqs, _, consts, _, _, _ = parse_equations(text)
    val = list(consts.values())[0] if consts else None
    check(f"'{text}' ist Konstante", not eqs and val is not None and abs(val - expected) < 1e-9)
# asin/acos/atan: Zahl in Grad oder Winkelgröße je nach Verwendung -> Gleichung (wie asin(x))
eqs, variables, consts, _, orig, _ = parse_equations("alpha = asin(0.5)")
check("'alpha = asin(0.5)' ist Gleichung (Ergebnis 30)", len(eqs) == 1 and not consts
      and abs(solve_system(eqs, variables, constants=consts, original_equations=orig)[1]['alpha'] - 30) < 1e-9)

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

# °C ist immer eine absolute Temperatur, unabhängig vom Namen; Differenzen in K
_, _, consts, _, _, uv = parse_equations("dT_1 = 10 °C\ndT_2 = 10 K\ntheta = 10 °C")
check("°C immer absolut (auch dT_1), K ohne Offset",
      abs(consts['dT_1'] - 283.15) < 1e-9 and abs(consts['theta'] - 283.15) < 1e-9
      and abs(consts['dT_2'] - 10) < 1e-12, str(consts))

# Temperatur-Charakter aus der Struktur, nicht aus dem Namen (gleiches Ergebnis für jeden Namen)
from unit_constraints import temperature_sum_conflicts
labels = []
for x, y, z in (("dT", "T_2", "theta"), ("a", "b", "c")):
    text = f"T_1 = 20 °C\n{x} = 10 K\n{y} = T_1 + {x}\nQ = 4190*{z}\n{z} = {y} - T_1"
    eqs, variables, consts, _, orig, uv = parse_equations(text)
    known = {v: u.calc_unit for v, u in uv.items()}
    units = propagate_all_units_complete(orig, known, open_temperatures={x}, undetermined_temperature='delta_K')
    labels.append((units.get(x), units.get(y), units.get(z)))
check("Charakter aus der Struktur: Differenz in K, Summe absolut, Differenz zweier Temperaturen",
      labels[0] == labels[1] == ('delta_K', 'K', 'delta_K'), str(labels))
eqs, variables, consts, _, orig, uv = parse_equations("Q = 41900 W\nm = 1 kg/s\nc = 4190 J/(kg*K)\nQ = m*c*theta")
units = propagate_all_units_complete(orig, {v: u.calc_unit for v, u in uv.items()},
                                     undetermined_temperature='delta_K')
check("Nur in Produkten: Charakter offen -> Anzeige in K", units.get('theta') == 'delta_K', str(units))
# Anzeige: Temperatur im Produkt ohne Temperatur-Dimension ist eine Differenz (wie COMSOL)
for a, b, q in (("T1", "T2", "Q_dot"), ("u", "w", "zz")):
    text = f"{a} = 80 °C\n{q} = 20 kW\nm_dot = 1 kg/s\nc = 4 kJ/kgK\n{q} = m_dot*c*({a} - {b})"
    eqs, variables, consts, _, orig, uv = parse_equations(text)
    units = propagate_all_units_complete(orig, {v: u.calc_unit for v, u in uv.items()},
                                         undetermined_temperature='delta_K', differences_in_products=True)
    check(f"Q = m*c*({a} - {b}) mit {a} absolut: {b} absolut", units.get(b) == 'K', str(units))
text = "T_R = 20 °C\nh = 7.7 W/(m^2*K)\nq = 30 W/m^2\nRa = 9.81/((T_R + T_G)/2)*(T_R - T_G)\nq = h*(T_R - T_G)"
eqs, variables, consts, _, orig, uv = parse_equations(text)
units = propagate_all_units_complete(orig, {v: u.calc_unit for v, u in uv.items()},
                                     undetermined_temperature='delta_K', differences_in_products=True)
check("Mittelwert (T_R + T_G)/2 im Produkt: T_G bleibt absolut", units.get('T_G') == 'K', str(units))
# sigma*T^4: Temperatur mit ganzzahligem Exponent >= 2 ist absolut -> Differenz zweier solcher ist Differenz
text = ("sigma = 5.67e-8 W/(m^2*K^4)\nT_a = 20 °C\nG = 300 W/m^2\nsigma*T_s^4 = sigma*T_a^4 + 50 W/m^2\n"
        "T_ms^4 = T_s^4 + G/sigma\ndT_solar = T_ms - T_s")
text = text.replace("+ 50 W/m^2", "+ G/6")
eqs, variables, consts, _, orig, uv = parse_equations(text)
units = propagate_all_units_complete(orig, {v: u.calc_unit for v, u in uv.items()},
                                     undetermined_temperature='delta_K', differences_in_products=True)
check("sigma*T^4: T_s, T_ms absolut, dT_solar = T_ms - T_s Differenz",
      (units.get('T_s'), units.get('T_ms'), units.get('dT_solar')) == ('K', 'K', 'delta_K'), str(units))
eqs, variables, consts, _, orig, uv = parse_equations("dT = 10 K\nh = 1.31*dT^(1/3)")
units = propagate_all_units_complete(orig, {v: u.calc_unit for v, u in uv.items()}, open_temperatures={'dT'},
                                     undetermined_temperature='delta_K', differences_in_products=True)
check("Gebrochener Exponent (dT^(1/3)): Charakter bleibt offen", units.get('dT') == 'delta_K', str(units))

# Vollständige Einheiten-Ableitung (Stufe 2, Kennedy/Olsson): gekoppelte Einheiten aus dem
# gemeinsamen linearen System der Exponenten - die lokale Propagation findet sie nicht
for label, text, expected in (
        ("a*b = X, a/b = Y", "X = 6 m^2\nY = 1.5\nX = a*b\nY = a/b", {'a': 'm', 'b': 'm'}),
        ("v*x = w, v*v = k (Olsson)", "w = 3 m\nk = 1\nv*x = w\nv*v = k", {'v': '', 'x': 'm'}),
        ("Olsson SecondOrder", "k = 2\nu = 1\ny = 3\nyd = 4 1/s\nydd = 2 1/s^2\nydd = w*(w*(k*u - y) - 2*D*yd)",
         {'w': '1/s', 'D': ''}),
        ("x = y^n (n Variable): keine Willkür", "y = 5 m\nn = 0.4\nx = y^n", {'x': None})):
    eqs, variables, consts, _, orig, uv = parse_equations(text)
    known = {v: u.calc_unit for v, u in uv.items() if u.calc_unit} | {v: '' for v in consts if v not in uv}
    units = propagate_all_units_complete(orig, known)
    check(f"Einheiten-Ableitung gekoppelt: {label}", {k: units.get(k) for k in expected} == expected,
          str({k: units.get(k) for k in expected}))

# Temperaturen werden IMMER in Kelvin gerechnet - keine Regel rechnet eine Gleichung auf
# einer anderen Skala; Summen absoluter Temperaturen ergeben nur einen Hinweis
import unit_constraints as _uc
check("Keine Skalen-Regel (Rechnung immer in Kelvin)", not hasattr(_uc, 'scale_dependent_sums'))
for text, expected in (("T_1 = 20 °C\nx = 10 °C\nT_2 = T_1 + x", 1), ("T_1 = 20 °C\nx = 10 K\nT_2 = T_1 + x", 0),
                       ("T_1 = 80 °C\nT_2 = 20 °C\nT_m = (T_1 + T_2)/2\ntheta = T_1 - T_2", 0),
                       ("T_1 = 20 °C\nT_2 = T_1 + 10", 0),
                       ("T_0 = -40 °C\nT_c = 25 °C\nEER_C*(T_c - T_0) = T_0", 0),
                       # Summe absoluter Temperaturen: Hinweis (gerechnet wird in Kelvin)
                       ("T_1 = 20 °C\nT_2 = 40 °C\nT_3 = T_2 + T_1", 1),
                       ("T_1 = 68 °F\nT_2 = 104 °F\nT_3 = T_2 + T_1", 1),
                       ("T_1 = 20 °C\nT_2 = 40 °C\nT_3 = T_2 + T_1 + 2*abs(T_2 - T_1)", 1),
                       # Summand skaliert eine Temperatur: Gesetz in Kelvin (Isentrope), keine Prüfung
                       ("T_1 = 20 °C\nT_2 = T_1*5^0.286", 0), ("T_1 = 20 °C\nT_3 = 2*T_1", 0),
                       ("T_1 = 20 °C\nT_2 = 40 °C\nT_m = (T_1 + T_2)/2", 0),
                       ("T_1 = 20 °C\nT_2 = 40 °C\nT_e = 2*T_2 - T_1", 0),
                       ("T_1 = 20 °C\nT_2 = 40 °C\nT_m = (T_1 + T_2)/2\ntheta = T_2 - T_1", 0),
                       ("T_1 = 20 °C\nr = 1.2\nT_2 = T_1*r", 0),
                       ("T_1 = 300 K\nT_2 = 400 K\nT_3 = T_1 + T_2", 0),
                       # Differenz mal Größe: ausmultipliziertes Gesetz in Kelvin (Carnot), kein Hinweis
                       ("T_0 = -40 °C\nT_c = 25 °C\nT_0 = EER_C*(T_c - T_0)", 0),
                       ("T_0 = -40 °C\nT_c = 25 °C\ndT = T_c - T_0\nEER_C*dT = T_0", 0),
                       ("T_0 = -40 °C\nT_c = 25 °C\nCOP*(T_c - T_0) = T_c", 0)):
    eqs, variables, consts, _, orig, uv = parse_equations(text)
    known = {v: u.calc_unit for v, u in uv.items()}
    open_k = {v for v, u in uv.items() if u.original_unit == 'K'}
    found = temperature_sum_conflicts(orig, known, open_k)
    check(f"Temperatur-Summe ohne Charakter erkannt: {text.splitlines()[-1]} -> {expected}",
          len(found) == expected, str(found))

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

# Exponent ohne '^' (m2, m3/h, W/m2K) und Malpunkt - früher stillschweigend NICHT umgerechnet
for value, unit, si in ((20, 'cm2', 0.002), (500, 'm3/h', 500 / 3600), (1.2, 'kg/m3', 1.2),
                        (10, 'W/m2K', 10.0), (5.67e-8, 'W/m2K4', 5.67e-8), (0.13, 'm2K/W', 0.13),
                        (0.04, 'W/(m·K)', 0.04), (4.19, 'kJ/(kg·K)', 4190.0)):
    uv = UnitValue.from_input(value, unit)
    check(f"{value} {unit} erkannt -> {si:.6g} SI", uv.quantity is not None and abs(uv.calc_value - si) <= 1e-9 * abs(si),
          f"{uv.calc_value} {uv.calc_unit}")
eqs, variables, consts, _, orig, _ = parse_equations(
    "alpha=10 W/m2K\nlambda=0.04 W/mK\nA=20cm2\nT_1=20°C\nT_2=15°C\nQ_dot=A*alpha*(T_1-T_2)")
s_, sol, _ = solve_system(eqs, variables, constants=consts, original_equations=orig)
check("Beispiel A=20cm2, alpha=10 W/m2K -> Q_dot = 0.1 W", s_ and abs(sol.get('Q_dot', 0) - 0.1) < 1e-12)

# Unbekannte / unpassende Einheiten sind ein Fehler (früher: still dimensionslos)
for text, fragment in (("A = 20 qcm", "Unbekannte Einheit 'qcm'"),
                       ("T = 20:10:40 xyz", "Unbekannte Einheit 'xyz'"),
                       ("h = enthalpy(water, T=20 Grad, p=1 bar)", "Einheit 'Grad' passt nicht"),
                       ("h = enthalpy(water, T=20 °C, p=1 kg)", "Einheit 'kg' passt nicht"),
                       ("E = Eb(500 °C, 5 bar)", "Einheit 'bar' passt nicht")):
    try:
        parse_equations(text)
        check(f"Fehler für '{text}'", False, "keine Exception")
    except Exception as exc:
        check(f"Fehler für '{text}'", fragment in str(exc), str(exc))

# Lesbare Dimensionsfehler-Warnung (Variable + SI-Labels statt [mass] / [time] ** 3)
w = check_all_unit_consistency({}, {'(q) - (U*A*dT)': 'q = U*A*dT'},
                               {'q': 'W/m^2', 'U': 'W/(m^2*K)', 'A': 'm^2', 'dT': 'delta_K'})
check("Dimensionsfehler-Warnung: Variable q, W/m^2 vs. W",
      len(w) == 1 and w[0].variable == 'q' and 'links: W/m^2 ≠ rechts: W' in list(w[0].units.values())[0],
      str(w))

# Widerspruchsmeldung mit beiden Seiten statt nur Residuum
eqs, variables, consts, _, orig, _ = parse_equations("q = 50\nA = 2\nU = 10\nq = U*A")
s_, sol, msg = solve_system(eqs, variables, constants=consts, original_equations=orig)
check("Widerspruch: 'q ist vorgegeben (50), aus ... folgt 20'",
      not s_ and "q ist vorgegeben (50)" in msg and "folgt 20" in msg and "überbestimmt" in msg, msg)
eqs, variables, consts, _, orig, _ = parse_equations("x + 1 = 3\nx + 1 = 4")
s_, sol, msg = solve_system(eqs, variables, constants=consts, original_equations=orig)
check("Widerspruch ohne vorgegebene Variable: links/rechts", not s_ and "links = " in msg, msg)

# Syntaxfehler werden beim Einlesen generisch gemeldet (Syntaxbaum): Zeile, Art, Fundstelle ▶
for text, fragments in (("a = 1\nU = 1/(r_1 ((1/r_1) + 2))", ("Zeile 2:", "'r_1' ist keine Funktion", "r_1 ▶((1/r_1)")),
                        ("y = 2(x+1)", ("Zeile 1:", "'2' ist keine Funktion", "2▶(x+1)")),
                        ("y = (a+b)(c+d)", ("'a+b' ist keine Funktion", "(a+b)▶(c+d)")),
                        ("y = sqr(x)", ("'sqr' ist keine Funktion", "sqr▶(x)")),
                        ("Q = m*cp*(T_2 - 20 °C)", ("Syntaxfehler", "20 ▶°C")),
                        ("y = 2x + 1", ("Syntaxfehler", "▶2x")),
                        ("{a\nb}\nx = 1\ny = x^^2", ("Zeile 4:", "Syntaxfehler", "x^▶^2"))):
    try:
        parse_equations(text)
        check(f"Syntaxfehler erkannt: {text!r}", False, "keine Exception")
    except Exception as exc:
        check(f"Syntaxfehler erkannt: {text!r}", all(f in str(exc) for f in fragments), str(exc))
for text in ("h = enthalpy(water, T=T_1, p=p)", "E = Eb(T, L) + eb(T, L)", "w = HumidAir(w, T=T, rh=0.5, p_tot=p)",
             "y = sqrt(x) + ln(x) + log10(x) + exp(-x) + abs(x) + max(x, 1) + sin(x) + tanh(x)", "cp = cv + R"):
    try:
        parse_equations(text)
        check(f"Gültige Gleichung ohne Fehlalarm: '{text}'", True)
    except Exception as exc:
        check(f"Gültige Gleichung ohne Fehlalarm: '{text}'", False, str(exc))

# Exponent einer Zahl ist keine Einheit: "x = 1e5*y" war früher stillschweigend x = 1
eqs, variables, consts, _, orig, _ = parse_equations("y = 2\nx = 1e5*y\nz = 2.5e6*y")
s_, sol, _ = solve_system(eqs, variables, constants=consts, original_equations=orig)
check("x = 1e5*y -> 2e5 (nicht 1)", s_ and sol.get('x') == 2e5 and sol.get('z') == 5e6, str(sol))

# Generische Diagnose (diagnostics.py): Struktur (Dulmage-Mendelsohn), Numerik, Namens-Hinweise
ROHR = """h_e=11
h_i=2000
r_1=0.1
r_2=0.104
q_dot=50
r_3=r_2+s
{S}
U = 1/(r_1*((1/r_1h_i) + ln(r_3/r_2)/0.04 + 1/(r_3*h_e)))
q_dot = U*2*r_1*pi*80"""
text = ROHR.replace("{S}", "")
e, v, c, _, o, _ = parse_equations(text)
rep = analyze_structure(e, v)
check("Struktur: unterbestimmter Teil {r_1h_i, r_3, s} mit 2 Gleichungen",
      rep.under_variables == ['r_1h_i', 'r_3', 's'] and len(rep.under_equations) == 2, str(rep))
msg = " ".join(describe_structure(rep, o, text))
check("Meldung nennt Zeilen 6, 8 und 1 fehlende Gleichung",
      "Zeilen 6, 8" in msg and "es fehlt 1 Gleichung" in msg, msg)
hints = name_hints(e, v, set(c), o, text)
check("Hinweis: r_1h_i = r_1 + h_i (Tippfehler?)",
      len(hints) == 1 and "r_1h_i" in hints[0] and "r_1 und h_i" in hints[0], str(hints))

text = ROHR.replace("{S}", "s=0.1")      # formal quadratisch: Tippfehler wird "gelöst"
e, v, c, _, o, _ = parse_equations(text)
s_, sol, _ = solve_system(e, v, constants=c, original_equations=o)
errors, hints = diagnose(e, v, c, o, text, None)
check("Formal lösbar + Tippfehler: kein Fehler, aber Hinweis", s_ and not errors and len(hints) == 1, str(hints))

e, v, c, _, o, _ = parse_equations("x + y = 1\nz^2 = 4\nz + 1 = 3")
rep = analyze_structure(e, v)
check("Zählung stimmt, Struktur nicht: unter {x,y} + über {z}",
      rep.under_variables == ['x', 'y'] and rep.over_variables == ['z'] and rep.surplus_equations == 1, str(rep))

e, v, c, _, o, _ = parse_equations("T_e = 10\nq = 100\nh = 5\nq = h*(T_s - T_E)\nT_s = 30")
hints = name_hints(e, v, set(c), o)
check("Hinweis: T_E ähnelt T_e (Schreibweise)", len(hints) == 1 and "ähnelt T_e" in hints[0], str(hints))

text = "a = 2\nexp(z) = -a\nb = a*3"
e, v, c, _, o, _ = parse_equations(text)
s_, sol, _ = solve_system(e, v, constants=c, original_equations=o)
errors, _ = diagnose(e, v, c, o, text, sol)
check("Numerisch unlösbar: Meldung nennt z und Zeile 2",
      not s_ and errors and "Keine numerische Lösung für z (Zeile 2)" in errors[0], str(errors))

# Keine Fehlalarme bei korrekten Systemen (Ergebnisgrößen, inverse Aufgaben, indizierte Namen)
for text in ("T_1 = 300\nT_2 = 350\nT_3 = (T_1 + T_2)/2",
             "R_si = 0.13\nd_1 = 0.2\nl_1 = 2.3\nU = 0.2\nU = 1/(R_si + d_1/l_1 + d_2/0.035 + 0.04)",
             "q = 100\nA = 2\ndT = 10\nq = h*A*dT"):
    e, v, c, _, o, _ = parse_equations(text)
    errors, hints = diagnose(e, v, c, o, text, None)
    check(f"Kein Fehlalarm: {text.splitlines()[-1]!r}", not errors and not hints, str(errors + hints))

# Startwerte über die Dimension (früher Teilstring: alles mit 'k' -> 350)
for unit, expected in (('1 / kelvin', 3.4e-3), ('kilogram / second', 1.0), ('W/m^2K', 10.0),
                       ('delta_K', 10.0), ('kW', 1e5), ('kg/kg', 0.01), ('um', 5e-6), ('K', 350.0)):
    check(f"Startwert für '{unit}' = {expected}", get_initial_from_unit(unit) == expected,
          str(get_initial_from_unit(unit)))


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

# Strahlung: intern SI (Wellenlänge m, Eb W/m³, Wien m); Zahlen sind Meter, keine µm-Deutung
check("Wien(1000 K) = 2.898e-6 m", abs(Wien_displacement(1000) - 2.897771955e-6) < 1e-15)
check("Eb(1000, 5 µm) in W/m³", abs(Eb(1000, 5e-6) / 7.13962e9 - 1) < 1e-5, str(Eb(1000, 5e-6)))
check("Eb: Zahl 5 ist 5 m (keine µm-Deutung)", Eb(1000, 5) < 1e-6 * Eb(1000, 5e-6), str(Eb(1000, 5)))
check("Blackbody: Zahlen sind Meter", Blackbody(5800, 0.75, 1000) < 1e-12 < Blackbody(5800, 0.75e-6, 1e-3))
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
# Tearing mit linear impliziten Gleichungen (natürliche Schreibweise), exakt aufgelöst
from solver import _is_linear_in, _LinearStep
check("Linear in h_2: eta = (h_1 - h_2)/(h_1 - h_2s)", _is_linear_in("(eta) - ((h_1 - h_2)/(h_1 - h_2s))", "h_2")
      and not _is_linear_in("(eta) - ((h_1 - h_2)/(h_1 - h_2s))", "h_2s")
      and not _is_linear_in("(y) - (exp(x))", "x") and _is_linear_in("(Q) - (m*(h_2 - h_10))", "m"))
cands = _find_tear_candidates(["(eta) - ((h_1 - h_2)/(h_1 - h_2s))", "(h_2s) - (h_1 - 0.5*p)",
                               "(m) - (Q/(h_2 - 1))", "(p*m) - (3)"], {'h_2', 'h_2s', 'm', 'p'},
                              {'eta', 'h_1', 'Q'}, allow_linear=True)
check("Tearing-Kette mit linearem Schritt", any(isinstance(step, _LinearStep) for c in cands for _, step in c[1]),
      str(cands))
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
# Befunde aus den Lehrbeispielen (Testbeispiele/Wärmetechnik, Oktober 2026)
# ---------------------------------------------------------------------------
print("=== Befunde Lehrbeispiele ===")
from parser import EquationSyntaxError


def parse_error(text):
    try:
        parse_equations(text)
        return ""
    except Exception as exc:
        return str(exc)


# Gleichung mit (numerisch) nur Null-Termen darf den Block nicht verwerfen
eqs, variables, consts, _, orig, _ = parse_equations("eps = 0\nq_x = eps*y\nx^2 + y = 10\nx*y + q_x = 6")
s_, sol, msg = solve_system(eqs, variables, constants=consts, original_equations=orig)
check("Null-Term (eps = 0) im Block: gelöst", s_ and abs(sol['x'] ** 2 + sol['y'] - 10) < 1e-7
      and abs(sol['x'] * sol['y'] - 6) < 1e-7, msg)

# Fehlermeldungen aus Funktionen / undefinierte Ausdrücke werden genannt
eqs, variables, consts, _, orig, _ = parse_equations(
    "C_r = 1\nNTU = 2\neps = (1-exp(-NTU*(1-C_r)))/(1-C_r*exp(-NTU*(1-C_r)))")
s_, sol, msg = solve_system(eqs, variables, constants=consts, original_equations=orig)
check("0/0 wird als 'nicht definiert' gemeldet", not s_ and "eps ist nicht definiert" in msg, msg)

# HumidAir: cp-Ausgabe, unbekannte Eigenschaft beim Einlesen gemeldet
eqs, variables, consts, _, orig, _ = parse_equations("c = HumidAir(cp, T=25 °C, rh=0.5, p_tot=1 bar)")
s_, sol, msg = solve_system(eqs, variables, constants=consts, original_equations=orig)
check("HumidAir(cp, ...) ~ 1025 J/(kg K)", s_ and abs(sol['c'] - 1025.2) < 0.5, str(sol))
check("HumidAir: unbekannte Eigenschaft gemeldet",
      "unbekannte Eigenschaft 'cpx'" in parse_error("c = HumidAir(cpx, T=25 °C, rh=0.5, p_tot=1 bar)"))

# Signaturprüfung (generisch für alle Funktionen)
for text, fragment in (("h = enthalpy(watr, T=300 K, p=1 bar)", "Unbekanntes Fluid 'watr'"),
                       ("h = enthalpy(water, T=300 K)", "braucht genau 2 Zustandsgrößen"),
                       ("h = enthalpy(water, T=300 K, q=1 bar)", "unbekannter Parameter 'q'"),
                       ("E = Eb(500 °C)", "Eb() erwartet 2 Argument"),
                       ("y = sqrt(4, 2)", "sqrt() erwartet genau 1 Argument"),
                       ("z = max(3)", "max() erwartet mindestens 2 Argumente")):
    message = parse_error(text)
    check(f"Signatur: {text!r}", fragment in message and message.startswith("Zeile 1:"), message)
check("Fluid R1234ze(E) und INCOMP::MEG-30% gültig",
      not parse_error("a = enthalpy(R1234ze(E), T=300 K, p=1 bar)\nb = enthalpy(INCOMP::MEG-30%, T=300 K, p=1 bar)"))

# Dezimalkomma, lg, Malpunkt, Kehrwert-Einheiten
check("Dezimalkomma gemeldet", "Dezimalkomma" in parse_error("Pr = 0,71\nx = 2*Pr"))
check("Dezimalkomma mit Einheit gemeldet", "Dezimalkomma" in parse_error("lambda = 0,6544 W/mK\nq = lambda*2"))
eqs, variables, consts, _, orig, _ = parse_equations(
    "Re = 1e5\nxi = (1.8*lg(Re) - 1.5)^(-2)\nnu = 0.475·10^-6 m^2/s\nb = 3·nu\nn_L = 0.3 1/h\nk = 2 h^-1")
s_, sol, msg = solve_system(eqs, variables, constants=consts, original_equations=orig)
check("lg = log10", s_ and abs(sol['xi'] - (1.8 * 5 - 1.5) ** -2) < 1e-12, msg)
check("Malpunkt als Multiplikation", abs(consts['nu'] - 4.75e-7) < 1e-15 and abs(sol['b'] - 1.425e-6) < 1e-15)
check("Kehrwert-Einheiten 1/h und h^-1", abs(consts['n_L'] - 0.3 / 3600) < 1e-15 and abs(consts['k'] - 2 / 3600) < 1e-15)

# Parameterstudie: gescheiterter Punkt gemeldet, gelöste Größen bleiben, Unabhängige als Einzelwert
eqs, variables, consts, sweeps, orig, _ = parse_equations(
    "C_g = 5000\nc = 5000\nm = 0.5:0.5:1.5\nC = m*c\nC_r = min(C_g, C)/max(C_g, C)\nd_i = 0.05\nr_i = d_i/2\n"
    "eps = (1-exp(-2*(1-C_r)))/(1-C_r*exp(-2*(1-C_r)))")
s_, sol, msg = solve_parametric(eqs, variables, sweeps, {}, constants=consts, original_equations=orig)
check("Sweep: Punkt ohne Lösung mit Nummer und Grund gemeldet",
      s_ and "ohne Lösung: Punkt 2" in msg and "nicht definiert" in msg, msg)
check("Sweep: wohldefinierte Größe im gescheiterten Punkt erhalten", np.allclose(sol['C'], [2500, 5000, 7500]))
check("Sweep: sweep-unabhängige Größe als Einzelwert", sol['r_i'] == 0.025, str(sol['r_i']))
eqs, variables, consts, sweeps, orig, _ = parse_equations("h = 10:10:1000\ne = 0:0.01:1\nq = h*e")
s_, sol, msg = solve_parametric(eqs, variables, sweeps, {}, constants=consts)
check("Sweep-Längen: Meldung nennt Variablen", "h (100 Werte)" in msg and "e (101 Werte)" in msg, msg)

# Potenzgesetz-Fit aus zwei Messpunkten ohne Startwerte (Heuristik-Startwert ~2e4 lief über)
eqs, variables, consts, _, orig, _ = parse_equations(
    "N_1 = 769.2\nN_2 = 961.5\nR_1 = 5e5\nR_2 = 666667\nN_1 = C*R_1^m\nN_2 = C*R_2^m")
s_, sol, msg = solve_system(eqs, variables, constants=consts, original_equations=orig)
check("Potenzgesetz-Fit N = C*Re^m ohne Startwerte", s_ and abs(sol['m'] - 0.77566) < 1e-4, msg)

# Adiabate Fläche als "Q_3 = 0" plus Bilanzgleichung (Null-Terme im Block)
eqs, variables, consts, _, orig, _ = parse_equations("""sigma = 5.67e-8
T_1 = 1073.15
T_2 = 300
eps_1 = 0.7
eps_2 = 0.5
Q_1 = (sigma*T_1^4 - J_1)/((1 - eps_1)/eps_1)
Q_1 = 0.2*(J_1 - J_2) + 0.8*(J_1 - J_3)
(sigma*T_2^4 - J_2)/((1 - eps_2)/eps_2) = 0.2*(J_2 - J_1) + 0.8*(J_2 - J_3)
Q_3 = 0
Q_3 = 2*(0.4*(J_3 - J_1) + 0.4*(J_3 - J_2))
J_3 = sigma*T_3^4""")
s_, sol, msg = solve_system(eqs, variables, constants=consts, original_equations=orig)
check("Adiabate Fläche (Q_3 = 0) im Radiositätsnetz", s_ and abs(sol['T_3'] - 942.4379) < 1e-3, msg)

# Positivliste der Ausdrucksformen: Listen, Indizes, Vergleiche, Attributzugriff gemeldet
for text, fragment in (("y = 2*[1, 2, 4]", "Wertelisten"), ("x = 2\ny = x[1]", "Indizes"),
                       ("y = ().__class__", "Punkt-Zugriff"), ("y = 3 > 2", "Vergleiche"),
                       ("a = 2\nb = a and 3", "Logische Verknüpfungen")):
    check(f"Nicht unterstützter Ausdruck gemeldet: {text!r}", fragment in parse_error(text), parse_error(text))

# Mehrfach-Tearing: Radiositätsnetz + erzwungene Konvektion mit Stoffwerten bei Filmtemperatur
RAD_KONV = """l = 0.2 m
T_1 = 700 °C
eps_1 = 0.94
eps_3 = 0.8
u_inf = 4 m/s
T_inf = 25 °C
T_Umg = 35 °C
p = 1 bar
sigma = 5.67e-8 W/m^2K^4
J_1 = eps_1*sigma*T_1^4 + (1 - eps_1)*(0.8*J_2 + 0.2*J_3)
J_2 = 0.2*J_1 + 0.6*J_2 + 0.2*J_3
J_3 = eps_3*sigma*T_3^4 + (1 - eps_3)*(0.2*J_1 + 0.8*J_2)
q_3_in = eps_3/(1 - eps_3)*(J_3 - sigma*T_3^4)
T_f = (T_3 + T_inf)/2
nu_L = viscosity(air, T=T_f, p=p)/density(air, T=T_f, p=p)
lambda_L = conductivity(air, T=T_f, p=p)
Pr_L = prandtl(air, T=T_f, p=p)
Re = u_inf*l/nu_L
Nu = 0.664*sqrt(Re)*Pr_L^(1/3)
h_inf = Nu*lambda_L/l
q_3_in = h_inf*(T_3 - T_inf) + eps_3*sigma*(T_3^4 - T_Umg^4)"""
eqs, variables, consts, _, orig, _ = parse_equations(RAD_KONV)
t0 = time.monotonic()
s_, sol, msg = solve_system(eqs, variables, constants=consts, original_equations=orig)
check("Mehrfach-Tearing: Strahlung + Konvektion bei Filmtemperatur (T_3 = 439.6 °C, < 10 s)",
      s_ and abs(sol['T_3'] - 712.739) < 0.01 and time.monotonic() - t0 < 10, msg)

# Mehrfach-Tearing mit fester Gewichtung: Raum-Strahlungsnetz mit adiabaten Wänden
# (alle Terme am Start null) + Fensterbilanz mit abs() in den Rayleigh-Zahlen
FENSTER_RAUM = """T_R = 20 °C
T_D = 25 °C
T_a = -10 °C
H = 3 m
L = 5 m
s = 20 mm
eps_G = 0.8
eps_W = 0.9
sigma = 5.67e-8 W/m^2K^4
g = 9.81 m/s^2
k_L = 0.02476 W/mK
nu_L = 1.49e-5 m^2/s
a_L = 2.04e-5 m^2/s
Pr_L = nu_L/a_L
D_g = sqrt(H^2 + L^2)
F_12 = (H + L - D_g)/(2*H)
F_13 = (D_g - L)/H
F_14 = F_12
F_21 = H/L*F_12
F_23 = (H + L - D_g)/(2*L)
F_24 = (D_g - H)/L
F_31 = F_13
F_32 = L/H*F_23
F_34 = F_12
F_41 = F_21
F_42 = F_24
F_43 = F_23
Q_1/H = (sigma*T_G1^4 - J_1)/((1 - eps_G)/eps_G)
Q_1 = H*(F_12*(J_1 - J_2) + F_13*(J_1 - J_3) + F_14*(J_1 - J_4))
Q_2/L = (sigma*T_D^4 - J_2)/((1 - eps_W)/eps_W)
Q_2 = L*(F_21*(J_2 - J_1) + F_23*(J_2 - J_3) + F_24*(J_2 - J_4))
0 = F_31*(J_3 - J_1) + F_32*(J_3 - J_2) + F_34*(J_3 - J_4)
0 = F_41*(J_4 - J_1) + F_42*(J_4 - J_2) + F_43*(J_4 - J_3)
Ra_i = g/((T_R + T_G1)/2)*abs(T_R - T_G1)*H^3/(nu_L*a_L)
Nu_i = (0.825 + 0.387*Ra_i^(1/6)/(1 + (0.492/Pr_L)^(9/16))^(8/27))^2
h_i = Nu_i*k_L/H
q_G = h_i*(T_R - T_G1) - Q_1/H
Ra_A = g/((T_G1 + T_G2)/2)*abs(T_G1 - T_G2)*s^3/(nu_L*a_L)
Nu_A = (1 + (0.0665*Ra_A^(1/3)*Ra_A^1.4/(Ra_A^1.4 + 9000^1.4))^2)^(1/2)
h_A = Nu_A*k_L/s
q_G = h_A*(T_G1 - T_G2) + sigma*(T_G1^4 - T_G2^4)/(2/eps_G - 1)
Ra_a = g/((T_G2 + T_a)/2)*abs(T_G2 - T_a)*H^3/(nu_L*a_L)
Nu_a = (0.825 + 0.387*Ra_a^(1/6)/(1 + (0.492/Pr_L)^(9/16))^(8/27))^2
h_a = Nu_a*k_L/H
q_G = h_a*(T_G2 - T_a) + eps_G*sigma*(T_G2^4 - T_a^4)"""
eqs, variables, consts, _, orig, _ = parse_equations(FENSTER_RAUM)
t0 = time.monotonic()
s_, sol, msg = solve_system(eqs, variables, constants=consts, original_equations=orig)
check("Raumstrahlung + Fensterbilanz (adiabate Wände, abs()) in < 10 s",
      s_ and abs(sol['T_G1'] - 285.7761) < 1e-3 and time.monotonic() - t0 < 10, msg)

# Startwerte: unbekannte Temperaturen beim Mittel der vorgegebenen Temperaturen
# (nicht pauschal 350 K) + gestaffelte Tearing-Starts. Natürliche Formulierung
# ohne abs(): bei 350 K wäre T_R - T_G1 < 0 -> Ra^(1/6) nicht reell
from units import initial_values_from_units, is_absolute_temperature_unit
from unit_constraints import propagate_all_units_complete
check("Absolute Temperatureinheit erkannt (K, °C), delta_K nicht",
      is_absolute_temperature_unit('K') and is_absolute_temperature_unit('degC')
      and not is_absolute_temperature_unit('delta_K') and not is_absolute_temperature_unit('W/m^2'))
start = initial_values_from_units(['T_x', 'q'], {'T_1': 'K', 'T_2': 'K', 'dT': 'delta_K', 'T_x': 'K', 'q': 'W/m^2'},
                                  {'T_1': 263.15, 'T_2': 293.15, 'dT': 5.0})
check("Startwert unbekannter Temperatur = Mittel der gegebenen", start == {'T_x': 278.15, 'q': start['q']}, str(start))
FENSTER_NATUERLICH = """T_R = 20 °C
T_D = 25 °C
T_a = -10 °C
H = 3 m
L = 5 m
s = 20 mm
eps_G = 0.8
eps_W = 0.9
sigma = 5.67e-8 W/m^2K^4
g = 9.81 m/s^2
k_L = 0.02476 W/mK
nu_L = 1.49e-5 m^2/s
a_L = 2.04e-5 m^2/s
Pr_L = nu_L/a_L
D_g = sqrt(H^2 + L^2)
F_12 = (H + L - D_g)/(2*H)
F_13 = (D_g - L)/H
F_14 = F_12
F_21 = H/L*F_12
F_23 = (H + L - D_g)/(2*L)
F_24 = (D_g - H)/L
F_31 = F_13
F_32 = L/H*F_23
F_34 = F_12
F_41 = F_21
F_42 = F_24
F_43 = F_23
Q_1/H = (sigma*T_G1^4 - J_1)/((1 - eps_G)/eps_G)
Q_1 = H*(F_12*(J_1 - J_2) + F_13*(J_1 - J_3) + F_14*(J_1 - J_4))
Q_2/L = (sigma*T_D^4 - J_2)/((1 - eps_W)/eps_W)
Q_2 = L*(F_21*(J_2 - J_1) + F_23*(J_2 - J_3) + F_24*(J_2 - J_4))
0 = F_31*(J_3 - J_1) + F_32*(J_3 - J_2) + F_34*(J_3 - J_4)
0 = F_41*(J_4 - J_1) + F_42*(J_4 - J_2) + F_43*(J_4 - J_3)
Ra_i = g/((T_R + T_G1)/2)*(T_R - T_G1)*H^3/(nu_L*a_L)
Nu_i = (0.825 + 0.387*Ra_i^(1/6)/(1 + (0.492/Pr_L)^(9/16))^(8/27))^2
h_i = Nu_i*k_L/H
q_G = h_i*(T_R - T_G1) - Q_1/H
Ra_A = g/((T_G1 + T_G2)/2)*(T_G1 - T_G2)*s^3/(nu_L*a_L)
Nu_A = (1 + (0.0665*Ra_A^(1/3)/(1 + (9000/Ra_A)^1.4))^2)^(1/2)
h_A = Nu_A*k_L/s
q_G = h_A*(T_G1 - T_G2) + sigma*(T_G1^4 - T_G2^4)/(2/eps_G - 1)
Ra_a = g/((T_G2 + T_a)/2)*(T_G2 - T_a)*H^3/(nu_L*a_L)
Nu_a = (0.825 + 0.387*Ra_a^(1/6)/(1 + (0.492/Pr_L)^(9/16))^(8/27))^2
h_a = Nu_a*k_L/H
q_G = h_a*(T_G2 - T_a) + eps_G*sigma*(T_G2^4 - T_a^4)"""
eqs, variables, consts, _, orig, uvals = parse_equations(FENSTER_NATUERLICH)
all_units = propagate_all_units_complete(orig, {v: uv.calc_unit for v, uv in uvals.items() if uv.calc_unit})
start = initial_values_from_units(variables, all_units, consts)
t0 = time.monotonic()
s_, sol, msg = solve_system(eqs, variables, start, constants=consts, original_equations=orig)
check("Fensterbilanz natürlich formuliert (ohne abs) mit Einheiten-Startwerten in < 10 s",
      s_ and abs(sol['T_G1'] - 285.7761) < 1e-3 and time.monotonic() - t0 < 10, msg)

# Dasselbe OHNE Einheiten (alle Werte in SI, keine Startwerte aus Einheiten): Startwerte aus
# der Struktur - T_G1, T_G2 stehen als Summanden neben T_R bzw. T_a (T_R - T_G1, T_G2 - T_a)
SI_VALUES = {"20 °C": "293.15", "25 °C": "298.15", "-10 °C": "263.15", "20 mm": "0.02"}
fenster_si = FENSTER_NATUERLICH
for with_unit, si in SI_VALUES.items():
    fenster_si = fenster_si.replace(f"= {with_unit}\n", f"= {si}\n")
fenster_si = re.sub(r"(= [-\d.e]+) [^\n]*?(?=\n)", r"\1", fenster_si)
eqs, variables, consts, _, orig, uvals = parse_equations(fenster_si)
t0 = time.monotonic()
s_, sol, msg = solve_system(eqs, variables, {}, constants=consts, original_equations=orig)
check("Fensterbilanz OHNE Einheiten (keine Einheit im Blatt, keine Startwerte) gleich",
      not uvals and s_ and abs(sol['T_G1'] - 285.7761) < 1e-3 and time.monotonic() - t0 < 10,
      f"{msg} {sorted(uvals)[:3]}")

# Formelzeichen dürfen das Ergebnis nicht beeinflussen (Regeln nur aus der Struktur):
# alle Variablen der einheitenfreien Fensterbilanz neutral umbenannt (Reihenfolge gemischt)
names = sorted(variables | set(consts), key=len, reverse=True)
mapping = {name: f"zq{(i * 7) % len(names)}" for i, name in enumerate(sorted(names))}
renamed = re.sub(r"(?<![\w.])[A-Za-z_]\w*(?!\w)", lambda m: mapping.get(m.group(0), m.group(0)), fenster_si)
eqs, variables, consts, _, orig, _ = parse_equations(renamed)
s_, sol_renamed, msg = solve_system(eqs, variables, {}, constants=consts, original_equations=orig)
check("Fensterbilanz mit neutralen Namen: gleiches Ergebnis",
      s_ and all(abs(sol_renamed[mapping[k]] - v) <= 1e-7 * max(1.0, abs(v)) for k, v in sol.items()), msg)
roots = []
for known, unknown in (("r_1", "r_2"), ("a", "b")):
    eqs, variables, consts, _, orig, _ = parse_equations(f"{known} = -5\n{unknown}^2 = 4")
    s_, sol_root, _ = solve_system(eqs, variables, constants=consts, original_equations=orig)
    roots.append(sol_root[unknown])
check("Wurzelwahl hängt nicht von ähnlichen Namen ab (r_2 wie b)",
      roots[0] == roots[1] and abs(roots[0] - 2) < 1e-9, str(roots))

# Parameterstudie: nach zwei Zeitüberschreitungen hintereinander abbrechen
import solver as _solver_module
_old_limit = _solver_module.SOLVE_TIME_LIMIT
_solver_module.SOLVE_TIME_LIMIT = 1.0
eqs, variables, consts, sweeps, orig, _ = parse_equations(
    "z = 1:1:10\np = 1 bar\na = exp(b) + c^2 + z + enthalpy(water, T=T_1, p=p)/1e6\nb = exp(c) + a^2 + 1\n"
    "c = exp(a) + b^2 + 1\nT_1 = 300 + d\nd^2 + e^2 = -1 - a^2\ne = d + f\nf*g = 1 + a\n"
    "g = sin(f) + 1\ny = 2*z")
t0 = time.monotonic()
s_, sol, msg = solve_parametric(eqs, variables, sweeps, {}, constants=consts, original_equations=orig)
elapsed = time.monotonic() - t0
_solver_module.SOLVE_TIME_LIMIT = _old_limit
check("Parameterstudie: Abbruch nach 2 Zeitüberschreitungen (nicht 10 x Limit)",
      not s_ and "Abgebrochen nach Punkt 2" in msg and elapsed < 4, f"{elapsed:.1f} s: {msg}")

# Reine Ausgabegleichungen (nirgends weiterverwendet) dürfen den Kernblock nicht stören
text = FENSTER_NATUERLICH + "\nRa_A_chk = abs(T_G1 - T_G2)\nRa_i_chk = (T_R - T_G1)/((T_R + T_G1)/2)"
eqs, variables, consts, _, orig, uvals = parse_equations(text)
all_units = propagate_all_units_complete(orig, {v: uv.calc_unit for v, uv in uvals.items() if uv.calc_unit})
t0 = time.monotonic()
s_, sol, msg = solve_system(eqs, variables, initial_values_from_units(variables, all_units, consts),
                            constants=consts, original_equations=orig)
check("Ausgabegleichungen am Block: Kern trotzdem gelöst (< 10 s)",
      s_ and abs(sol['T_G1'] - 285.7761) < 1e-3 and time.monotonic() - t0 < 10, msg)

# Gesamtzeitlimit: unlösbares gekoppeltes System bricht ab, Teillösung bleibt
import solver as _solver_module
_old_limit = _solver_module.SOLVE_TIME_LIMIT
_solver_module.SOLVE_TIME_LIMIT = 3.0
eqs, variables, consts, _, orig, _ = parse_equations(
    "p = 1 bar\na = exp(b) + c^2 + 1 + enthalpy(water, T=T_1, p=p)/1e6\nb = exp(c) + a^2 + 1\n"
    "c = exp(a) + b^2 + 1\nT_1 = 300 + d\nd^2 + e^2 = -1 - a^2\ne = d + f\nf*g = 1 + a\n"
    "g = sin(f) + 1\ny = 2*k\nk = 5")
t0 = time.monotonic()
s_, sol, msg = solve_system(eqs, variables, constants=consts, original_equations=orig)
elapsed = time.monotonic() - t0
_solver_module.SOLVE_TIME_LIMIT = _old_limit
check("Zeitlimit: Abbruch mit Meldung und Teillösung",
      not s_ and "Zeitlimit" in msg and elapsed < 6 and sol.get('y') == 10 and _solver_module._deadline is None,
      f"{elapsed:.1f} s: {msg}")

# Exponent ist dimensionslos (keine Warnung 'Einheit unbekannt: n')
w = check_all_unit_consistency({}, {'a': 'UA_1 = UA_0*(m_1/m_0)^n'},
                               {'UA_0': 'W/K', 'm_1': 'kg/s', 'm_0': 'kg/s', 'UA_1': 'W/K'})
check("Exponent n ohne Einheiten-Warnung", w == [], str(w))

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

# ---------------------------------------------------------------------------
# Wertelisten x = [ ... ] (Messdaten per Zwischenablage)
# ---------------------------------------------------------------------------
for text, expected in [("T = [20 25 30] °C", [293.15, 298.15, 303.15]),
                       ("T = [20; 25; 30] °C", [293.15, 298.15, 303.15]),
                       ("T = [20, 25, 30] °C", [293.15, 298.15, 303.15]),
                       ("T = [20\t25\t30] °C", [293.15, 298.15, 303.15]),
                       ("T = [\r\n20\r\n25\r\n30\r\n] °C", [293.15, 298.15, 303.15]),
                       ("x = [1e-3 2.5E2 -3]", [0.001, 250, -3]),
                       ("m = [1 2] kg/h", [1 / 3600, 2 / 3600]),
                       ("dT = [0 10 20] K", [0, 10, 20]),
                       ("dT = 0:5:20 K", [0, 5, 10, 15, 20]),
                       ("dT = [0 10] °C", [273.15, 283.15])]:
    _, _, _, sweeps, _, _ = parse_equations(text)
    name = text.split("=")[0].strip()
    check(f"Werteliste: {text!r}", np.allclose(sweeps.get(name, []), expected), str(sweeps))
text = "T = [20\n25\n30] °C\nm = [1 2 3] kg/s\nQ = m*4190*(T - 293.15)"
eqs, variables, consts, sweeps, orig, _ = parse_equations(text)
s_, sol, msg = solve_parametric(eqs, variables, sweeps, {}, constants=consts, original_equations=orig)
check("Werteliste mehrzeilig, punktweise mit zweiter Liste kombiniert",
      s_ and np.allclose(sol['Q'], [0, 41900, 125700]), msg)
for text, needle in [("x = [1,5 2,5 3,5]", "Dezimalkomma"),
                     ("x = [1 a 3]", "'a' ist keine Zahl"),
                     ("x = [1\n2\n3] kg\n\ny = x*", "Zeile 5"),
                     ("m = 1 [kg/s]", "Einheiten ohne Klammern"),
                     ("y = a[1]", "Indizes")]:
    try:
        parse_equations(text)
        check(f"Werteliste-Fehler {text!r}", False, "kein Fehler")
    except Exception as exc:
        check(f"Werteliste-Fehler {text!r} -> {needle}", needle in str(exc), str(exc))
_, _, _, sweeps, _, _ = parse_equations('"Liste [offen"\n{Klammer [1 2}\nx = [1 2]')
check("Eckige Klammern in Kommentaren stören nicht", np.allclose(sweeps['x'], [1, 2]))

# ---------------------------------------------------------------------------
# Wellenlängen: Zahlen und Variablen ohne Einheit immer SI (m) - keine Deutung nach der Größe
# ---------------------------------------------------------------------------
results = {}
for text in ("E = Eb(1000, 5e-6)", "E = Eb(1000, 5 µm)", "L = 5 µm\nE = Eb(1000, L)"):
    eqs, variables, consts, _, orig, _ = parse_equations(text)
    results[text] = solve_system(eqs, variables, constants=consts, original_equations=orig)[1]['E']
check("Wellenlänge: Literal 5e-6, 5 µm und L = 5 µm gleich",
      max(results.values()) - min(results.values()) < 1e-6 * max(results.values()), str(results))
eqs, variables, consts, _, orig, _ = parse_equations("E = Eb(1000, 5)")
sol = solve_system(eqs, variables, constants=consts, original_equations=orig)[1]
check("Wellenlänge als Zahl ohne Einheit = m (Eb(1000, 5) sind 5 m)", sol['E'] < 1e-10, str(sol))
from unit_constraints import numeric_value_form
for text, names, unit, expected in (
        ("T3=T1+T2", ['T1', 'T2', 'T3'], '°C', "value(T3, °C) = value(T1, °C) + value(T2, °C)"),
        ("T1 + T2 - T3 = 0 {Kommentar}", ['T1', 'T2', 'T3'], 'K', "value(T1, K) + value(T2, K) - value(T3, K) = 0"),
        ("y = T1 + enthalpy(water, T=T1, p=p)", ['T1', 'y'], '°F',
         "value(y, °F) = value(T1, °F) + enthalpy(water, T=T1, p=p)")):
    check(f"Zahlenwert-Schreibweise: {text} -> {expected}", numeric_value_form(text, names, unit) == expected,
          str(numeric_value_form(text, names, unit)))
from parser import wavelength_literals
for text, expected in (("E = Eb(1000, 5)", [5.0]), ("E = Eb(1000, 5 µm)", []), ("E = Eb(1000, L)", []),
                       ("f = Blackbody(T, 0.38, 0.75 µm) {Kommentar 3}", [0.38]),
                       ("F = blackbody_cumulative(5800, 1e-6) + Eb(T, 2)", [1e-6, 2.0]),
                       ("E = Wien(1000)", [])):
    found = [number for _, number in wavelength_literals({text: text})]
    check(f"Hinweis Wellenlänge ohne Einheit: {text} -> {expected}", found == expected, str(found))
eqs, variables, consts, _, orig, _ = parse_equations("L = 5\nE = Eb(1000, L)")
sol = solve_system(eqs, variables, constants=consts, original_equations=orig)[1]
check("Wellenlänge als Variable ohne Einheit = m (SI, wie T ohne Einheit = K)", sol['E'] < 1e-10, str(sol))
for start in (None, {'x': 1.0}, {'x': 5e-6}):
    eqs, variables, consts, _, orig, _ = parse_equations("F = Blackbody_cumulative(5800, x)\nF = 0.5")
    sol = solve_system(eqs, variables, start, constants=consts, original_equations=orig)[1]
    check(f"Iterierte Wellenlänge eindeutig in m (Start {start})",
          abs(sol.get('x', 0) - 7.0815e-7) < 1e-10, str(sol))

# Asymptotische Scheinlösung: Optimalitätsbedingung mit Eb(T, lambda) auf beiden Seiten
# geht für lambda -> unendlich gegen 0 = 0; darf nicht als Lösung gelten
SELEKTIV = """T_inf = 30 °C
h = 12 W/m^2K
q_solar = 800 W/m^2
T_sky = 20 °C
q_ab = 200 W/m^2
T_sun = 5800 K
sigma = 5.67e-8 W/m^2K^4
eps_1 = 0.9
eps_2 = 0.2
F_sun = Blackbody_cumulative(T_sun, lambda_c)
F_p = Blackbody_cumulative(T_p, lambda_c)
alpha_S = eps_1*F_sun + eps_2*(1 - F_sun)
eps_p = eps_1*F_p + eps_2*(1 - F_p)
alpha_S*q_solar = eps_p*sigma*(T_p^4 - T_sky^4) + h*(T_p - T_inf) + q_ab
q_solar*Eb(T_sun, lambda_c)/(sigma*T_sun^4) = sigma*(T_p^4 - T_sky^4)*Eb(T_p, lambda_c)/(sigma*T_p^4)"""
eqs, variables, consts, _, orig, uvals = parse_equations(SELEKTIV)
all_units = propagate_all_units_complete(orig, {v: uv.calc_unit for v, uv in uvals.items() if uv.calc_unit})
s_, sol, msg = solve_system(eqs, variables, initial_values_from_units(variables, all_units, consts),
                            constants=consts, original_equations=orig)
check("Keine asymptotische Scheinlösung (lambda_c = 4.10 µm, nicht 1e2 m)",
      s_ and abs(sol['lambda_c'] - 4.1035e-6) < 1e-9 and abs(sol['T_p'] - 340.229) < 0.01, f"{msg} {sol}")
eqs, variables, consts, _, orig, _ = parse_equations("Q_3 = 0\nJ_1 = 400\nJ_2 = 300\nQ_3 = 2*(0.4*(J_3 - J_1) + 0.4*(J_3 - J_2))")
s_, sol, msg = solve_system(eqs, variables, constants=consts, original_equations=orig)
check("Null-Gleichung mit großen inneren Termen wird akzeptiert", s_ and abs(sol['J_3'] - 350) < 1e-9, msg)

# ---------------------------------------------------------------------------
print("\n=== Fallunterscheidung IF(a, b, x, y, z) wie EES ===")
# ---------------------------------------------------------------------------
check("IF: a < b -> x, a = b -> y, a > b -> z",
      if_function(1, 2, 10, 20, 30) == 10 and if_function(2, 2, 10, 20, 30) == 20
      and if_function(3, 2, 10, 20, 30) == 30)
check("IF: ungültiger Vergleichswert (NaN, komplex) -> ungültiges Ergebnis",
      np.isnan(if_function(float('nan'), 2, 10, 20, 30)) and np.isnan(if_function(1 + 2j, 2, 10, 20, 30)))
check("IF: elementweise für Arrays",
      list(if_function(np.array([1.0, 2.0, 3.0]), 2, 10, 20, 30)) == [10, 20, 30])
eqs, variables, consts, _, orig, _ = parse_equations(
    "Re = 5000\nRe_krit = 2300\nNu_lam = 3.66\nNu = IF(Re, Re_krit, Nu_lam, Nu_lam, 0.023*Re^0.8*0.7^0.4)")
s_, sol, msg = solve_system(eqs, variables, constants=consts, original_equations=orig)
check("IF in Gleichung (turbulenter Zweig)", s_ and abs(sol['Nu'] - 0.023 * 5000**0.8 * 0.7**0.4) < 1e-9, msg)
for spelling in ("if", "If", "IF"):
    eqs, variables, consts, _, _, _ = parse_equations(f"a = 1\nb = {spelling}(a, 2, 3, 4, 5)")
    s_, sol, _ = solve_system(eqs, variables, constants=consts)
    check(f"Schreibweise {spelling}( wird erkannt", s_ and sol['b'] == 3)
eqs, variables, consts, _, _, _ = parse_equations("if = 3\ny = 2*if")
s_, sol, _ = solve_system(eqs, variables, constants=consts)
check("'if' ohne Klammer bleibt ein Variablenname", s_ and sol['y'] == 6)
_, _, consts, _, _, _ = parse_equations("x = IF(2, 3, 10, 20, 30)")
check("IF in reinem Zahlenausdruck ist eine Konstante", consts.get('x') == 10)
eqs, variables, consts, sweeps, orig, _ = parse_equations(
    "Re = 1000:1000:4000\nNu = IF(Re, 2300, 3.66, 3.66, 0.023*Re^0.8)")
s_, sol, msg = solve_parametric(eqs, variables, sweeps, {}, constants=consts, original_equations=orig)
check("IF in Parameterstudie (Umschaltung zwischen den Punkten)",
      s_ and list(np.round(sol['Nu'], 4)) == [3.66, 3.66, round(0.023 * 3000**0.8, 4), round(0.023 * 4000**0.8, 4)],
      str(sol.get('Nu')))
eqs, variables, consts, _, orig, _ = parse_equations("y = IF(x, 1, 2*x, 2, x + 1)\ny = 5")
s_, sol, msg = solve_system(eqs, variables, constants=consts, original_equations=orig)
check("IF mit iterierter Vergleichsgröße: nur der gültige Zweig ist Lösung",
      s_ and abs(sol['x'] - 4) < 1e-9, f"{msg} {sol}")
check("IF mit falscher Argumentzahl -> Meldung",
      "IF(a, b, x, y, z) erwartet 5 Argumente" in parse_error("a = 1\nb = IF(a, 2, 3, 4)"))
check("Vergleich ohne IF -> Hinweis auf IF",
      "IF(a, b, x, y, z)" in parse_error("a = 1\nb = (a < 2)*3"))
units = propagate_all_units_complete(
    {'(q_2) - (IF(T_1, T_2, q, 0, 2*q))': 'q_2 = IF(T_1, T_2, q, 0, 2*q)',
     '(T_m) - (IF(T_1, T_2, T_2, T_2, T_1))': 'T_m = IF(T_1, T_2, T_2, T_2, T_1)'},
    {'T_1': 'K', 'T_2': 'K', 'q': 'W/m^2'})
check("Einheit von IF = Einheit der Zweige", units.get('q_2') == 'W/m^2' and units.get('T_m') == 'K', str(units))
w = check_all_unit_consistency({}, {'(y) - (IF(T_1, p, 1, 2, 3))': 'y = IF(T_1, p, 1, 2, 3)'},
                               {'T_1': 'K', 'p': 'Pa', 'y': ''})
check("IF: Vergleich verschiedener Dimensionen -> Einheitenwarnung", len(w) == 1, str(w))
w = check_all_unit_consistency({}, {'(y) - (IF(T_1, T_2, q, 0, T_1))': 'y = IF(T_1, T_2, q, 0, T_1)'},
                               {'T_1': 'K', 'T_2': 'K', 'q': 'W/m^2', 'y': 'W/m^2'})
check("IF: Zweige verschiedener Dimension -> Einheitenwarnung", len(w) == 1, str(w))

# ---------------------------------------------------------------------------
print("\n=== Startwerte im Blatt {$Startwerte ... $} ===")
# ---------------------------------------------------------------------------
SHEET = "x^2 = 9\n{$Startwerte\nx = -3\nT_2 = 15 °C; lambda = 4 µm  {Kommentar}\ndT = 5 K\n$}\n"
check("Block wird gelesen (SI, Schlüsselwort-Namen, Temperaturdifferenz)",
      parse_start_values(SHEET) == {'x': -3.0, 'T_2': 288.15, '_kw_lambda': 4e-06, 'dT': 5.0},
      str(parse_start_values(SHEET)))
eqs, variables, _, _, _, _ = parse_equations(SHEET)
check("Block ist für das Gleichungssystem ein Kommentar", eqs == ['(x**2) - (9)'] and variables == {'x'})
check("Kein Block -> keine Startwerte", parse_start_values("x^2 = 9\n{Startwerte x = 3}") == {})
check("Block in einer Zeile", parse_start_values("{$startwerte x = 3 $}") == {'x': 3.0})
for bad, expected in (("a = 1\n{$Startwerte\nx = 2*y\n$}", "Zeile 3: Startwert als 'Name = Wert Einheit'"),
                      ("a = 1\n{$Startwerte\nx = 3 qcm\n$}", "Zeile 3:"),
                      ("a = 1\n{$Startwerte x = 3", "Zeile 2: Startwerte-Block nicht geschlossen")):
    try:
        parse_start_values(bad)
        message = ""
    except Exception as exc:
        message = str(exc)
    check(f"Fehler im Block mit Zeilennummer: {expected[:30]}", message.startswith(expected), message)


def apply_edit(text, lines):
    start, end, new = start_values_edit(text, lines)
    return text[:start] + new + text[end:]


check("Block anhängen (eine Leerzeile davor)",
      apply_edit("x^2 = 9", ["x = -3"]) == "x^2 = 9\n\n{$Startwerte\nx = -3\n$}\n"
      and apply_edit("x^2 = 9\n\n", ["x = -3"]) == "x^2 = 9\n\n{$Startwerte\nx = -3\n$}\n")
check("Block ersetzen (Rest unverändert)",
      apply_edit("a = 1\n{$Startwerte\nx = 1\n$}\nb = 2", ["x = 5"]) == "a = 1\n{$Startwerte\nx = 5\n$}\nb = 2")
check("Block entfernen", apply_edit("x^2 = 9\n\n{$Startwerte\nx = -3\n$}\n", []) == "x^2 = 9\n")

# Winkel: intern SI (rad) wie alles; sin/cos/tan lesen eine Winkelgröße in ihrer Einheit, eine
# Zahl ohne Einheit in Grad (früher: Winkel intern Grad -> sin(omega*t) mit rad/s, s = alpha*r falsch)
from units import UnitValue
for text, name, expected in (("a = 30 deg\ny = sin(a)", 'y', 0.5),
                             ("a = 0.5235987756 rad\ny = sin(a)", 'y', 0.5),
                             ("a = 0.25 turn\ny = cos(a)", 'y', 0.0),
                             ("a = 60\ny = cos(a)", 'y', 0.5),
                             ("a = 30 deg\nr = 2 m\ns = a*r", 's', np.pi / 3),
                             ("omega = 10 rad/s\nt = 2 s\ny = sin(omega*t)", 'y', np.sin(20)),
                             ("omega = 10 rad/s\nt = 2 s\nphi = omega*t\ny = sin(phi)", 'y', np.sin(20)),
                             ("w = 15 deg/h\nt = 2 h\nH = w*t\nc = cos(H)", 'c', np.cos(np.pi / 6)),
                             ("omega = 10 rad/s\nr = 0.5 m\nu = omega*r", 'u', 5.0),
                             ("a = 30 deg\ng = asin(0.5)\nk = g + a", 'k', np.pi / 3),
                             ("a = 30 deg\nsin(b) = 0.5\nc = b + a", 'c', np.pi / 3),
                             ("g = asin(0.5)\nk = 2*g", 'k', 60.0),
                             ("x = 0.5\ng = acos(x)\nc = cos(g)", 'c', 0.5),
                             ("a = 30 deg\ny = sin(value(a, deg))", 'y', 0.5),
                             ("y = sin(quantity(30, deg))", 'y', 0.5)):
    eqs, variables, consts, _, orig, _ = parse_equations(text)
    sol = solve_system(eqs, variables, constants=consts, original_equations=orig)[1]
    check(f"Winkel: {text.splitlines()[-1]} -> {name} = {expected:.6g}",
          abs(sol.get(name, np.nan) - expected) < 1e-9, str(sol))
_, _, _, sweeps, _, _ = parse_equations("b = 0:30:90 deg")
check("Winkel-Sweep in deg intern rad", np.allclose(sweeps['b'], np.radians([0, 30, 60, 90])), str(sweeps['b']))
check("Winkel-Anzeige: interner Wert rad -> deg/rad",
      abs(UnitValue.from_si_base(np.pi / 6, 'deg').original_value - 30) < 1e-12
      and UnitValue.from_si_base(np.pi / 6, 'rad').original_value == np.pi / 6)
eqs, variables, consts, sweeps, orig, _ = parse_equations("b = 0:30:90 deg\ny = cos(b)")
ok, results, _ = solve_parametric(eqs, variables, sweeps, constants=consts, original_equations=orig)[:3]
check("Winkel-Sweep: cos(b) mit b in deg", ok and np.allclose(results['y'], [1, np.sqrt(3) / 2, 0.5, 0]),
      str(results.get('y')))
eqs, variables, consts, _, orig, _ = parse_equations("g = asin(0.5)\n{$Startwerte\ng = 10 deg\n$}")
check("Startwert mit Winkeleinheit: asin liefert die Winkelgröße",
      abs(solve_system(eqs, variables, constants=consts, original_equations=orig)[1]['g'] - np.pi / 6) < 1e-9)

# Zahlenwertgleichungen: value(x, Einheit), quantity(z, Einheit) - generisch über pint
from units import unit_number, unit_quantity
check("value/quantity: °C, °F, bar, %, Winkel",
      abs(unit_number(293.15, '°C') - 20) < 1e-12 and abs(unit_quantity(68, '°F') - 293.15) < 1e-5
      and unit_number(3e5, 'bar') == 3 and abs(unit_number(0.5, '%') - 50) < 1e-12
      and abs(unit_number(np.pi, 'deg') - 180) < 1e-12 and unit_number(np.pi, 'rad') == np.pi
      and abs(unit_quantity(90, 'deg') - np.pi / 2) < 1e-12)
for text, name, expected in (
        ("T_a = -5 °C\nT_VL = quantity(20 + 1.5*(20 - value(T_a, °C)), °C)", 'T_VL', 330.65),
        ("v = 2 m/s\nh = quantity(5.7 + 3.8*value(v, m/s), W/(m^2*K))", 'h', 13.3),
        ("x = quantity(3, bar)", 'x', 3e5)):
    eqs, variables, consts, _, orig, _ = parse_equations(text)
    s_, sol, msg = solve_system(eqs, variables, constants=consts, original_equations=orig)
    check(f"Zahlenwertgleichung: {text.splitlines()[-1][:45]}", s_ and abs(sol[name] - expected) < 1e-9,
          str(sol.get(name)))
check("value(): unbekannte Einheit mit Zeile", "Zeile 2: Unbekannte Einheit 'qcm'"
      in parse_error("T_a = 0 °C\ny = value(T_a, qcm)"))
check("value(): falsche Argumente", "value(x, Einheit) erwartet" in parse_error("T_a = 0 °C\ny = value(T_a)"))
text = "v = 2 m/s\nh = quantity(5.7 + 3.8*value(v, m/s), W/(m^2*K))\nq = h*dT\ndT = 10 K"
eqs, variables, consts, _, orig, uv = parse_equations(text)
units = propagate_all_units_complete(orig, {v: u.calc_unit for v, u in uv.items()})
check("Einheiten durch quantity()/value() abgeleitet", units.get('h') == 'W/(m^2*K)' and units.get('q') == 'W/m^2',
      str(units))
w = check_all_unit_consistency({}, {"(n2) - (value(p, 'kg'))": "n2 = value(p, kg)"}, {'p': 'bar'})
check("value(p, kg) mit p in bar -> Einheitenwarnung", len(w) == 1, str(w))

# Fehlende Einheitenangaben benennen (Rang-Defekt der Einheiten-Gleichungen)
from unit_constraints import missing_unit_annotations
from parser import start_value_units
for text, expected in (("X = 6 m^2\nX = a*b", [(['a', 'b'], ['b'])]),
                       ("X = 6 m^2\nY = 1.5\nX = a*b\nY = a/b", []),
                       ("P = 2 kW\nx*y = P\nx^2*y^2 = P^2\nz = x + w", [(['x', 'y', 'z', 'w'], ['w'])]),
                       ("X = 6 m^2\nX = a*b\n{$Startwerte a = 2 m $}", [])):
    eqs, variables, consts, _, orig, uv = parse_equations(text)
    known = {v: u.calc_unit for v, u in uv.items()} | {v: '' for v in consts if v not in uv}
    known |= {v: u.calc_unit for v, u in start_value_units(text).items() if v not in known}
    found = missing_unit_annotations(orig, known)
    check(f"Fehlende Einheiten: {text.splitlines()[-1][:30]} -> {expected}", found == expected, str(found))


# Prozent und Promille als Einheit (Wirkungsgrade, relative Feuchte) - über pint wie jede Einheit
for text, name, expected, unit in (("eta = 89.2 %", 'eta', 0.892, '%'), ("eta = 89.2%", 'eta', 0.892, '%'),
                                   ("eta = 100*0.892 %", 'eta', 0.892, '%'), ("w = 1.5 ‰", 'w', 0.0015, '‰'),
                                   ("beta = 0.4 %/K", 'beta', 0.004, '%/K')):
    _, _, consts, _, _, uv = parse_equations(text)
    check(f"Einheit in Zuweisung: {text}", abs(consts.get(name, -1) - expected) < 1e-12
          and uv[name].original_unit == unit, f"{consts} {uv}")
_, _, _, sweeps, _, _ = parse_equations("rh = 40:20:80 %\nx = [10 20 30] %")
check("Prozent in Sweep und Werteliste", list(sweeps['rh']) == [0.4, 0.6, 0.8]
      and np.allclose(sweeps['x'], [0.1, 0.2, 0.3]), str(sweeps))
eqs, _, _, _, _, _ = parse_equations("a = 5\nb = a % 3")
check("% zwischen Variablen bleibt der Modulo-Operator", eqs == ['(b) - (a % 3)'], str(eqs))
eqs, _, _, _, _, _ = parse_equations("w = HumidAir(w, T=25 °C, rh=50 %, p_tot=1 bar)")
check("Prozent in Funktionsargumenten (rh=50 %)", "rh=0.5" in eqs[0], str(eqs))

# Überbestimmte, widerspruchsfreie Blöcke: quadratisches Teilsystem lösen, Rest prüfen -
# unabhängig von der Schreibweise (a = x + y bzw. x + y = a)
from parser import validate_system
for text in ("a = 2\na = x + y\nx + 3*y = 4\nx + 2*y = 3", "a = 2\nx + y = a\nx + 3*y = 4\nx + 2*y = 3",
             "x + y = 2\nx + 3*y = 4\nx + 2*y = 3", "x + y = 2\n2*x + 2*y = 4\nx - y = 0",
             "x - y = 1\nx^2 + y^2 = 5\nx*y = 2"):
    eqs, variables, consts, _, orig, _ = parse_equations(text)
    valid, _ = validate_system(eqs, variables, consts)
    s_, sol, msg = solve_system(eqs, variables, constants=consts, original_equations=orig)
    expected = (2, 1) if text.startswith("x - y") else (1, 1)
    check(f"Überbestimmt, widerspruchsfrei: {text.splitlines()[1]} ...", valid and s_
          and abs(sol['x'] - expected[0]) < 1e-6 and abs(sol['y'] - expected[1]) < 1e-6, msg)
eqs, variables, consts, _, orig, _ = parse_equations("a = 2\na = x + y\nx + 3*y = 4\nx + 2*y = 5")
s_, sol, msg = solve_system(eqs, variables, constants=consts, original_equations=orig)
check("Überbestimmt, widersprüchlich -> Meldung mit verletzter Gleichung",
      not s_ and "Widerspr" in msg and "x + 2*y = 5" in msg, msg)

# Feuchte Luft: Sättigung (rh = 1 trotz Rundungsfehler), Übersättigung mit klarer Meldung
from humid_air import HumidAir
fails = 0
for P in (0.8e5, 1e5, 2e5, 5e5, 10e5):
    for Tc in (0.5, 10, 20, 35, 60):
        w_s = HumidAir('w', T=Tc + 273.15, rh=1, p_tot=P)
        fails += abs(HumidAir('rh', T=Tc + 273.15, w=w_s, p_tot=P) - 1) > 1e-9
check("HumidAir(rh) bei gesättigter Luft = 1 (25 Zustände)", fails == 0, str(fails))
for out in ('h', 'T_dp', 'rh', 'rho_tot'):
    try:
        HumidAir(out, T=293.15, w=0.00889, p_tot=1e6)
        check(f"HumidAir({out}) übersättigt -> Meldung", False)
    except ValueError as exc:
        check(f"HumidAir({out}) übersättigt -> Meldung", "übersättigt" in str(exc) and "w_s" in str(exc), str(exc))
check("HumidAir mit Taupunkt als Eingabe (T_dp=)",
      abs(HumidAir('w', T=285.15, T_dp=281.15, p_tot=1e5) - 0.0067733) < 1e-6)
check("HumidAir mit Feuchtkugeltemperatur als Eingabe (T_wb=)",
      abs(HumidAir('w', T=303.15, T_wb=293.15, p_tot=1e5) - 0.0107726) < 1e-6)
check("HumidAir: Meldung nennt T, rF, T_dp wie dokumentiert",
      "T, p_tot, w, rh, rF" in parse_error("x = HumidAir(w, T=12 °C, feuchte=0.5, p_tot=1 bar)"))
# Deutsche Notation in HumidAir: x (Wassergehalt), phi (rel. Feuchte), p (Gesamtdruck), v (spez. Volumen)
check("HumidAir-Aliase x, phi, p, v",
      abs(HumidAir('x', T=299.15, phi=0.4, p=95000) - HumidAir('w', T=299.15, rh=0.4, p_tot=95000)) < 1e-15
      and abs(HumidAir('v', T=299.15, x=0.009, p=95000) - 1 / HumidAir('rho_a', T=299.15, w=0.009, p_tot=95000)) < 1e-12)
check("AirH2O -> Verweis auf HumidAir", "HumidAir(" in parse_error("h = enthalpy(AirH2O, T=300 K, p=1 bar)"))
# Namen nur aus a-z, A-Z, 0-9, _ - Griechisch/Umlaute klar gemeldet (Einheiten wie µm erlaubt)
for text in ("Φ = 0.6\nx = 2*Φ", "eta = 0.8\nQ = 5 kW\nQ_zu = Q/η", "Q_wärme = 5 kW"):
    check(f"Nicht-ASCII-Name gemeldet: {text.splitlines()[-1]}", "außerhalb von a-z" in parse_error(text),
          parse_error(text))
check("Einheit µm und Umlaut im Kommentar erlaubt", parse_error("L = 5 µm {Länge}\ny = 2*L") == "")

# Dampfgehalt außerhalb 0 ... 1: Meldung statt stillschweigender Begrenzung
from thermodynamics import enthalpy
try:
    enthalpy('CO2', T=293.15, x=2)
    check("enthalpy(x=2) -> Meldung", False)
except ValueError as exc:
    check("enthalpy(x=2) -> Meldung", "außerhalb von 0 ... 1" in str(exc), str(exc))
check("enthalpy(x=1+1e-12) (Rundungsfehler) wird begrenzt",
      abs(enthalpy('water', T=373.15, x=1 + 1e-12) - enthalpy('water', T=373.15, x=1)) < 1e-6)

# Blatt nur aus Vorgaben und Kontrollgleichungen: wird geprüft statt "Keine Variablen"
for text, ok_expected in (("x = 2\ny = x + 1\ny = 3", True), ("x = 2\ny = x + 1\ny = 4", False)):
    eqs, variables, consts, _, orig, _ = parse_equations(text)
    valid, _ = validate_system(eqs, variables, consts)
    s_, sol, msg = solve_system(eqs, variables, constants=consts, original_equations=orig)
    check(f"Nur Prüfgleichungen: {text.splitlines()[-1]} -> {'erfüllt' if ok_expected else 'Widerspruch'}",
          valid and s_ == ok_expected and (ok_expected or "Widerspr" in msg), msg)

# Blockzerlegung: der lösbare Kern bleibt erhalten, wenn eine Folgegleichung nicht auswertbar ist
# (auch bei kleinen und überbestimmten Blöcken - kein Scheinwiderspruch)
for text in ("x + y = 5\nx - y = 1\nz = ln(-y)", "x + y = 5\nx*y = 6\nz = ln(-y)",
             "x + y = 5\nx - y = 1\nx + 2*y = 7\nz = ln(-y)"):
    eqs, variables, consts, _, orig, _ = parse_equations(text)
    s_, sol, msg = solve_system(eqs, variables, constants=consts, original_equations=orig)
    check(f"Kern gelöst, Folgegleichung gemeldet: {text.splitlines()[-1]}",
          not s_ and 'x' in sol and 'y' in sol and abs(sol['x'] + sol['y'] - 5) < 1e-9
          and "ln(-y)" in msg and "Widerspr" not in msg, msg)
eqs, variables, consts, _, orig, _ = parse_equations("x + y = 5\nx - y = 1\nx + 2*y = 8\nz = ln(y)")
s_, sol, msg = solve_system(eqs, variables, constants=consts, original_equations=orig)
check("Überbestimmt mit Folgegleichung, echter Widerspruch bleibt erkannt", not s_ and "x + 2*y = 8" in msg, msg)

# Nassdampf unterhalb des Tripelpunkts (Wasser unter 0 °C: Eis) wird gemeldet
try:
    enthalpy('water', T=269.15, x=0)
    check("enthalpy(water, T=-4 °C, x=0) -> Meldung Tripelpunkt", False)
except ValueError as exc:
    check("enthalpy(water, T=-4 °C, x=0) -> Meldung Tripelpunkt", "Tripelpunkt" in str(exc), str(exc))
check("Nassdampf knapp über dem Tripelpunkt rechnet", abs(enthalpy('water', T=273.16, x=0)) < 10)

# Gescheiterter Kern sperrt nur seine Größen - unabhängige Kerne werden weiter gelöst
eqs, variables, consts, _, orig, _ = parse_equations("x + y = 5\nx - y = 1\na = 2\nb = 2\nz = 1/(a - b)\nq = x*z")
s_, sol, msg = solve_system(eqs, variables, constants=consts, original_equations=orig)
check("Unabhängiger Kern trotz gescheiterter Gleichung gelöst", not s_ and abs(sol.get('x', 0) - 3) < 1e-9
      and abs(sol.get('y', 0) - 2) < 1e-9 and 'z' not in sol, msg)
# Implizite Gleichung ohne Lösung: Meldung am Rand des Definitionsbereichs
eqs, variables, consts, _, orig, _ = parse_equations(
    "p = 950 mbar\nh_P = 34.15 kJ/kg\nw_P = 9.3 g/kg\nh_P = HumidAir(h, T=T_P, w=w_P, p_tot=p)")
s_, sol, msg = solve_system(eqs, variables, constants=consts, original_equations=orig)
check("Implizit im Nebelgebiet -> 'Zustand übersättigt'", not s_ and "übersättigt" in msg, msg)
# Gekoppelte HumidAir-Gleichungen mit automatischen Startwerten (w kg/kg, Start nicht genau auf T_6)
from unit_constraints import propagate_all_units_complete
from units import initial_values_from_units
text = ("p = 950 mbar\nT_6 = 19.2 °C\nh_1 = 64.9 kJ/kg\nh_4 = 46.4 kJ/kg\neta = 0.6\n"
        "h_1 = HumidAir(h, T=T_2, w=w_2, p_tot=p)\nh_4 = HumidAir(h, T=T_3, w=w_2, p_tot=p)\n"
        "eta = (T_2 - T_3)/(T_2 - T_6)")
eqs, variables, consts, _, orig, uv = parse_equations(text)
units_ = propagate_all_units_complete(orig, {v: u.calc_unit for v, u in uv.items()})
init = initial_values_from_units(variables, units_, consts)
s_, sol, msg = solve_system(eqs, variables, init, constants=consts, original_equations=orig)
check("Gekoppelte HumidAir-Gleichungen mit Einheiten-Startwerten", s_ and abs(sol['T_2'] - 322.645) < 0.01
      and init.get('w_2') == 0.01, f"{msg} {init}")
# Rundungsfunktionen
eqs, variables, consts, _, orig, _ = parse_equations("Q = 61.5\nn_1 = ceil(Q/5)\nn_2 = floor(Q/5)\nn_3 = round(Q/5)")
s_, sol, msg = solve_system(eqs, variables, constants=consts, original_equations=orig)
check("ceil/floor/round", s_ and (sol['n_1'], sol['n_2'], sol['n_3']) == (13, 12, 12), str(sol))
try:
    HumidAir('w', T=293.15, rh=50, p_tot=1e5)
    check("rh = 50 -> Meldung 0 ... 1", False)
except ValueError as exc:
    check("rh = 50 -> Meldung 0 ... 1", "außerhalb von 0 ... 1" in str(exc) and "50 %" in str(exc), str(exc))

# Bezugszustand je Fluid (REFERENCE R717 IIR): h, u, s verschoben, Umkehrfunktionen konsistent
import thermodynamics
from parser import parse_reference_states
from thermodynamics import set_reference_states, entropy, temperature, intenergy, density
h_default = enthalpy('ammonia', T=273.15, x=0)
set_reference_states(parse_reference_states("REFERENCE R717 IIR"))
check("REFERENCE R717 IIR: h' = 200 kJ/kg, s' = 1 kJ/(kg K) bei 0 °C (auch Name ammonia)",
      abs(enthalpy('ammonia', T=273.15, x=0) - 200e3) < 1e-3 and abs(entropy('nh3', T=273.15, x=0) - 1e3) < 1e-6)
h_iir = enthalpy('ammonia', T=263.15, p=3e5)
check("IIR: Umkehrfunktionen mit h und s", abs(temperature('ammonia', p=3e5, h=h_iir) - 263.15) < 1e-6
      and abs(temperature('ammonia', p=3e5, s=entropy('ammonia', T=263.15, p=3e5)) - 263.15) < 1e-6)
check("IIR: u um dieselbe Konstante wie h verschoben",
      abs((enthalpy('ammonia', T=300, p=2e5) - intenergy('ammonia', T=300, p=2e5))
          - 2e5/density('ammonia', T=300, p=2e5)) < 1e-6)
check("IIR: andere Fluide unverändert", abs(enthalpy('water', T=373.15, x=0) - 419.1e3) < 200)
set_reference_states({'R134a': 'ASHRAE', 'water': 'NBP'})
check("ASHRAE (-40 °C) und NBP (1 atm): h = 0; Ammoniak wieder Standard",
      abs(enthalpy('R134a', T=233.15, x=0)) < 1e-6 and abs(enthalpy('water', p=101325, x=0)) < 1e-6
      and abs(enthalpy('ammonia', T=273.15, x=0) - h_default) < 1e-6)
set_reference_states({})
for text, part in (("REFERENCE R717", "so angeben"), ("REFERENCE foo IIR", "Unbekanntes Fluid"),
                   ("REFERENCE water IIR", "nicht im Nassdampfgebiet"), ("REFERENCE ammonia XYZ", "möglich: IIR"),
                   ("REFERENCE ammonia IIR\nREFERENCE R717 NBP", "Zeile 2: Bezugszustand von R717 steht schon")):
    try:
        parse_reference_states(text)
        check(f"REFERENCE-Fehler: {text!r}", False)
    except EquationSyntaxError as exc:
        check(f"REFERENCE-Fehler: {text!r}", part in str(exc), str(exc))
eqs, variables, consts, _, orig, _ = parse_equations("REFERENCE R717 IIR\nreference = 2\ny = reference + 1")
check("REFERENCE-Zeile ist keine Gleichung, Variable reference bleibt möglich",
      variables == {'y'} and consts.get('reference') == 2, f"{variables} {consts}")

# Gleichung über mehrere Zeilen: Fortsetzung in offener Klammer nach '(' ',' oder Operator
eqs, variables, consts, _, orig, _ = parse_equations(
    "a = 2\ny = IF(a, 1, 10,\n      20, 30)   {Kommentar}\nNu = IF(a, 1, 3.66, 3.66,\n   0.023*a^0.8\n   *0.7^0.4)\nz = y + 1")
s_, sol, msg = solve_system(eqs, variables, constants=consts, original_equations=orig)
check("Mehrzeilige Gleichung (offene Klammer)", s_ and sol['y'] == 30 and abs(sol['Nu'] - 0.023*2**0.8*0.7**0.4) < 1e-12
      and 'y = IF(a, 1, 10, 20, 30)   {Kommentar}' in orig.values(), f"{sol} {list(orig.values())}")
try:
    parse_equations("a = 2\ny = (a + 1\nz = 3")
    check("Vergessene Klammer ohne Fortsetzung bleibt Fehler der Zeile", False)
except EquationSyntaxError as exc:
    check("Vergessene Klammer ohne Fortsetzung bleibt Fehler der Zeile", "Zeile 2" in str(exc), str(exc))
# Sattzustand mit T und p: verständliche Meldung statt CoolProp-Text
import CoolProp.CoolProp as CP
try:
    enthalpy('ammonia', T=263.15, p=CP.PropsSI('P', 'T', 263.15, 'Q', 1, 'Ammonia'))
    check("T und p auf der Sättigungslinie -> Meldung x angeben", False)
except ValueError as exc:
    check("T und p auf der Sättigungslinie -> Meldung x angeben", "Sättigungslinie" in str(exc) and "x=1" in str(exc), str(exc))

# Zeile ohne '=' (Tippfehler im Schlüsselwort, Ausdruck) ist ein Fehler statt still zu fehlen
for text, line_no in (("REFERENZ R717 IIR\nh_1 = 2", 1), ("a = 1\nh_1 - 100 kJ/kg", 2)):
    try:
        parse_equations(text)
        check(f"Zeile ohne '=' gemeldet: {text.splitlines()[line_no - 1]}", False)
    except EquationSyntaxError as exc:
        check(f"Zeile ohne '=' gemeldet: {text.splitlines()[line_no - 1]}",
              str(exc).startswith(f"Zeile {line_no}: ") and "keine Gleichung" in str(exc), str(exc))

# ---------------------------------------------------------------------------
# Prüfung auf nicht allgemeingültige Regeln (Oktober 2026): Einheiten, Parser, Solver
# ---------------------------------------------------------------------------
print("=== Allgemeingültigkeit ===")
# Einheiten-Umrechnung: Präfixe im Nenner bleiben Präfixe (früher k = Boltzmann, M = molar),
# °C in zusammengesetzten Einheiten ist eine Kelvin-Differenz, Drehzahl = Umdrehungen je Zeit
for value, unit, si in ((0.25, 'kg/kWh', 0.25 / 3.6e6), (2, 'K/kW', 2e-3), (1, '1/kPa', 1e-3),
                        (1, 'kg/MJ', 1e-6), (1, 'J/MPa', 1e-6), (1, 'm3/kWh', 1 / 3.6e6), (2, '°C/h', 2 / 3600),
                        (6.5, '°C/km', 6.5e-3), (2.1e-4, '1/°C', 2.1e-4), (0.5, 'W/m²°C', 0.5),
                        (1, 'kJ/(kg*°C)', 1000), (1, 'W/mK', 1), (1, 'W/cmK', 100), (3, 'kJ/kgK', 3000),
                        (1450, 'rpm', 1450 / 60), (1450, 'U/min', 1450 / 60), (68, '°F', 293.15)):
    calc = UnitValue.from_input(value, unit).calc_value
    check(f"Einheit {value} {unit} -> {si:g} SI", abs(calc - si) <= 1e-9 * abs(si), str(calc))
# e ist überall eine Variable (früher in "Ausdruck + Einheit" die Eulersche Zahl)
try:
    parse_equations("e = 2 m\nL = 2*e m")
    check("e in 'Ausdruck + Einheit' ist eine Variable (Einheit nur bei Zahlen)", False)
except EquationSyntaxError as exc:
    check("e in 'Ausdruck + Einheit' ist eine Variable (Einheit nur bei Zahlen)", "Zeile 2" in str(exc), str(exc))
# Doppelte Vorgabe: zweite Zeile als Prüfgleichung (Widerspruch statt still überschreiben)
for text, consistent in (("x = 5\nx = 3\ny = 2*x", False), ("x = 5\nx = 5.0\ny = 2*x", True),
                         ("p = 2 bar\np = 200 kPa\ny = 2*p", True)):
    eqs, variables, consts, _, orig, _ = parse_equations(text)
    ok, sol, msg = solve_system(eqs, variables, constants=consts, original_equations=orig)
    check(f"Doppelte Vorgabe {text.splitlines()[:2]} -> {'zulässig' if consistent else 'Widerspruch'}",
          ok == consistent and (consistent or "Widersprüchlich" in msg), msg)
# Funktionsaufruf mit Schlüsselwort-Argumenten links vom '='
eqs, variables, consts, _, orig, _ = parse_equations("p_1 = 1 bar\nT_1 = 20 °C\nenthalpy(water, T=T_1, p=p_1) = h_1")
ok, sol, _ = solve_system(eqs, variables, constants=consts, original_equations=orig)
check("enthalpy(water, T=T_1, p=p_1) = h_1 (T= links)", ok and abs(sol['h_1'] - 84006.1) < 1, str(sol))
# Variable namens minimize ist keine Optimierungs-Anweisung
eqs, variables, consts, _, _, _ = parse_equations("minimize = 3\ny = 2*minimize")
check("minimize = 3 ist eine Variable", consts.get('minimize') == 3.0, str(consts))
# Startwert: keine unbeteiligte Konstante als Anker (früher geometrisches Mittel aller Werte)
for extra in ("", "z = 1e6\n"):
    eqs, variables, consts, _, orig, _ = parse_equations(f"a = 101\nb = 96\nc = 4\n{extra}x^2 + y^2 = a*x - b\nx^2*y^2 = c*x^2")
    ok, sol, _ = solve_system(eqs, variables, constants=consts, original_equations=orig)
    check(f"Unbeteiligte Konstante ändert die Lösung nicht ({extra.strip() or 'ohne'})",
          ok and abs(sol['x'] - 1) < 1e-6 and abs(sol['y'] - 2) < 1e-6, str(sol))
# Reihenfolge im Blatt: Schlüsselwort-NAMEN (p=) sind kein Vorkommen einer Variablen p
results = []
for name in ("v", "p", "q"):
    eqs, variables, consts, _, orig, _ = parse_equations(
        f"0 = 0*prandtl(water, T=300, p=100000) + u - {name}^2\n{name} + 6 = u")
    ok, sol, _ = solve_system(eqs, variables, constants=consts, original_equations=orig)
    results.append((round(sol['u'], 6), round(sol[name], 6)))
check("Lösung unabhängig vom Namen (p= im Aufruf zählt nicht)", len(set(results)) == 1, str(results))
# 1-D: Newton springt nicht über die nähere Wurzel; Gleichstand -> positive Wurzel
s, sol, _ = solve_system(["(x**3 - 3.3*x + 1) - (0)"], {'x'}, initial_values={'x': 1.0})
check("Newton: nächstgelegene Wurzel (1.640, nicht -1.953)", abs(sol['x'] - 1.64022) < 1e-4, str(sol))
s, sol, _ = solve_system(["(x**2 - 2*x) - (5)"], {'x'}, initial_values={'x': 1.0})
check("Gleich weit entfernte Wurzeln: positive (1 + sqrt(6))", abs(sol['x'] - (1 + 6 ** 0.5)) < 1e-6, str(sol))
# Block mit mehreren Lösungen: Wahl nach dem Abstand ALLER Blockgrößen zu ihren Startwerten - nicht
# nach der Tearing-Größe (früher u = v^2 mit v = u - 6 -> (4; -2), mit u = v + 6 -> (9; 3))
results = set()
for text in ("u = v^2\nv = u - 6", "u = v^2\nu = v + 6", "v^2 = u\nu - v = 6", "u - v = 6\nu = v^2"):
    eqs, variables, consts, _, orig, _ = parse_equations(text)
    ok, sol, _ = solve_system(eqs, variables, constants=consts, original_equations=orig)
    results.add((round(sol['u'], 9), round(sol['v'], 9)))
check("Block: Lösung unabhängig von der Schreibweise (4; -2)", results == {(4.0, -2.0)}, str(results))
eqs, variables, consts, _, orig, _ = parse_equations("u = v^2\nv = u - 6")
ok, sol, _ = solve_system(eqs, variables, constants=consts, original_equations=orig, initial_values={'u': 10.0})
check("Block: Startwert wählt die Lösung (u = 10 -> (9; 3))", abs(sol['u'] - 9) < 1e-9 and abs(sol['v'] - 3) < 1e-9,
      str(sol))
# Widerspruchsprüfung: Rauschgrenze nur aus gleichartigen Größen (nicht aus 1e5 Pa)
eqs, variables, consts, _, orig, _ = parse_equations(
    "p = 1 bar\nV_dot = 1.05e-7 m^3/s\nk = 1e-12 m^3/(s*Pa)\nV_dot = k*p")
ok, _, msg = solve_system(eqs, variables, constants=consts, original_equations=orig)
check("Widerspruch kleiner SI-Werte erkannt (V_dot = k*p, 5 %)", not ok and "Widersprüchlich" in msg, msg)
eqs, variables, consts, _, orig, _ = parse_equations(
    "q_1 = 500.0000000000001\nq_2 = 500\nq_x = q_1 - q_2\neps = 0\ny = 20\nq_x = eps*y")
ok, _, msg = solve_system(eqs, variables, constants=consts, original_equations=orig)
check("Rundungsrest gegenüber gleichartigen Größen ist null (kein Widerspruch)", ok, msg)
# Wurzelsuche skalenunabhängig: Akzeptanz nur relativ zur Termgröße, Umgebung/Ableitung relativ
# zu x, Testpunkte 1e-30 ... 1e30 (früher |f| < 1e-10 absolut: y^2 = 4e-12 -> 7.8e-6 statt 2e-6)
_scale_cases = (("A = 3.14159e-12 m^2\nA = pi*d^2/4", 'd', (4 * 3.14159e-12 / np.pi) ** 0.5),
                ("y^2 = 4e-12", 'y', 2e-6), ("y^2 = 1e-30", 'y', 1e-15),
                ("V = 4.18879e-15 m^3\nV = pi/6*d^3", 'd', (6 * 4.18879e-15 / np.pi) ** (1 / 3)),
                ("D = 1e-9 m^2/s\nt = 1 s\nx^2 = D*t", 'x', 1e-9 ** 0.5),
                ("n = 4e-19 mol\nV = 1 m^3\nT = 300 K\nR = 8.314 J/(mol*K)\np*V = n*R*T", 'p', 4e-19 * 8.314 * 300),
                ("x^3 = 0", 'x', 0.0), ("tanh(x) = 1e-14", 'x', 1e-14), ("2*x = 1e21", 'x', 5e20),
                ("y^2 = 4e12", 'y', 2e6), ("K = 1e30\nK = 1/y", 'y', 1e-30))
_bad = []
for _text, _var, _ref in _scale_cases:
    eqs, variables, consts, _, orig, _ = parse_equations(_text)
    ok, sol, msg = solve_system(eqs, variables, constants=consts, original_equations=orig)
    if not ok or abs(sol.get(_var, np.nan) - _ref) > 1e-9 * abs(_ref) + (1e-300 if _ref else 0):
        _bad.append((_text.splitlines()[-1], sol.get(_var), _ref))
check("Wurzelsuche über alle Größenordnungen (1e-30 ... 5e20, Wurzel 0)", not _bad, str(_bad))
# Spurenanteil im Gleichgewicht (Knallgas, K = 1e8 -> n_O2 = 1e-16; früher keine Lösung unter 1e-12)
eqs, variables, consts, _, orig, _ = parse_equations(
    "K = 1e8\nn_H2_1 = 1\nn_O2_1 = 0.25\n2*n_H2_1 = 2*n_H2 + 2*n_H2O\n2*n_O2_1 = n_H2O + 2*n_O2\n"
    "n = n_H2 + n_H2O + n_O2\nK = (n_H2O/n)/((n_H2/n)*(n_O2/n)^0.5)")
ok, sol, msg = solve_system(eqs, variables, constants=consts, original_equations=orig)
check("Gleichgewicht mit Spurenanteil n_O2 = 1e-16", ok and abs(sol.get('n_O2', 0) - 1e-16) < 1e-20, f"{msg} {sol}")
# Ohm in allen Schreibweisen: ohm, Ω (griechisch), Ω (U+2126), Ohm (deutsch)
_, _, consts, _, _, _ = parse_equations("R_1 = 2 ohm\nR_2 = 3 Ω\nR_3 = 4 \u2126\nR_4 = 5 Ohm\nASR = 0.55 Ω*cm^2")
check("Ohm: ohm, Ω, U+2126, Ohm; ASR in Ω*cm^2 -> SI",
      [consts.get(k) for k in ('R_1', 'R_2', 'R_3', 'R_4')] == [2.0, 3.0, 4.0, 5.0]
      and abs(consts.get('ASR', 0) - 0.55e-4) < 1e-18, str(consts))
# Plateau von min/max: Terme wachsen mit der Lösung und heben sich auf -> keine Scheinlösung
# (früher "Lösung gefunden" mit n_w = 1049, letzte Gleichung um 100 % verletzt)
eqs, variables, consts, _, orig, _ = parse_equations(
    "W = 0.0017\n0.1 = n_w/(n_d + n_w)\n0.2 = n_s/(n_d + n_s)\nW = n_w + 0.0126 - min(n_w + 0.0126, n_s)")
ok, sol, msg = solve_system(eqs, variables, constants=consts, original_equations=orig,
                            initial_values={'n_d': 0.5, 'n_w': 0.5, 'n_s': 0.5})
eqs2, variables2, consts2, _, orig2, _ = parse_equations(
    "W = 0.0017\n0.1 = n_w/(n_d + n_w)\n0.2 = n_s/(n_d + n_s)\nn_a = n_w + 0.0126\nn_D = min(n_a, n_s)\n"
    "W = n_a - n_D")
ok2, sol2, msg2 = solve_system(eqs2, variables2, constants=consts2, original_equations=orig2,
                               initial_values={v: 1.0 for v in ('n_w', 'n_d', 'n_s', 'n_a', 'n_D')})
check("min()-Plateau, Widerspruch auf zwei Gleichungen verteilt: keine Scheinlösung",
      not ok2 or abs(sol2['n_a'] - min(sol2['n_a'], sol2['n_s']) - 0.0017) < 1e-9, f"{ok2} {sol2}")
# Simultan gelöster Block mit mehreren Lösungen (Gleichgewicht): die den Startwerten nächste, unabhängig
# von der Reihenfolge der Gleichungen (früher erste konvergierte Variante: n_CH4 = -0.46 mol)
_eq_lines = ["n_CH4_1 = n_CH4 + n_CO + n_CO2", "4*n_CH4_1 = 4*n_CH4 + 2*n_H2O + 2*n_H2",
             "2*n_O2_1 = n_H2O + n_CO + 2*n_CO2", "n_ges = n_CH4 + n_H2O + n_CO + n_CO2 + n_H2",
             "K_SR = (n_CO/n_ges)*(n_H2/n_ges)^3/((n_CH4/n_ges)*(n_H2O/n_ges))",
             "K_WGS = n_CO2*n_H2/(n_CO*n_H2O)"]
_results = set()
for _order in (_eq_lines, _eq_lines[::-1], _eq_lines[4:] + _eq_lines[:4]):
    _eqs, _vars, _consts, _, _orig, _ = parse_equations(
        "n_CH4_1 = 1 mol\nn_O2_1 = 0.68*n_CH4_1\nK_SR = 90.03\nK_WGS = 1.1876\n" + "\n".join(_order))
    _ok, _sol, _msg = solve_system(_eqs, _vars, constants=_consts, original_equations=_orig)
    _results.add(tuple(round(_sol.get(v, float('nan')), 6) for v in ('n_CH4', 'n_CO', 'n_CO2', 'n_H2', 'n_H2O')))
check("Gleichgewichtsblock: nächstgelegene (physikalische) Lösung, unabhängig von der Reihenfolge",
      len(_results) == 1 and all(x > 0 for x in next(iter(_results))), str(_results))
check("min()-Plateau: keine Scheinlösung (verletzte Gleichung ohne Empfindlichkeit)",
      not ok or abs(sol['n_w'] + 0.0126 - min(sol['n_w'] + 0.0126, sol['n_s']) - 0.0017) < 1e-9, f"{ok} {sol}")

# HumidAir: Reihenfolge der Feuchte-Eingaben aus dem Aufruf (nicht aus einer Menge)
import subprocess as _sp
code = "from humid_air import HumidAir; print(repr(HumidAir('h', rh=0.5, w=0.01, p_tot=1e5)))"
outputs = {_sp.run([sys.executable, '-c', code], cwd=os.path.dirname(os.path.abspath(__file__)), capture_output=True, text=True,
                   env={**os.environ, 'PYTHONHASHSEED': str(seed)}).stdout for seed in (0, 1, 3)}
check("HumidAir unabhängig vom Hash-Seed", len(outputs) == 1, str(outputs))

# ---------------------------------------------------------------------------
print("=== Ideale Gase und kondensierte Stoffe (NASA thermo.inp, IG:) ===")
import CoolProp.CoolProp as _CP
import ideal_gas
import thermodynamics as _th
from parser import parse_reference_states as _parse_reference

_data = ideal_gas.database()
check("thermo.inp eingelesen (> 2000 Spezies, Gase und kondensierte)",
      len(_data) > 2000 and not _data['N2'].condensed and _data['H2O(L)'].condensed, str(len(_data)))
check("Spezies ohne T-Bereich übersprungen (JP-4, CH4(L))", 'JP-4' not in _data and 'CH4(L)' not in _data)
check("Leerer Bereich (Tmax <= Tmin) übersprungen, gleicher Name zusammengefasst (Fe(a))",
      bool(np.all(_data['Ca(a)'].t_max > _data['Ca(a)'].t_min))
      and _data['Fe(a)'].range_text() == "300 ... 1184 K", _data['Fe(a)'].range_text())
_h2o, _h2ol = _data['H2O'], _data['H2O(L)']
check("h(IG:H2O, 298.15 K) = -241.826 kJ/mol, s° = 188.83 J/(mol K)",
      abs(_th.enthalpy('IG:H2O', T=298.15) * _h2o.molar_mass / 1e3 + 241.826) < 0.005
      and abs(_th.entropy('IG:H2O', T=298.15, p=1e5) * _h2o.molar_mass - 188.83) < 0.01)
check("h(IG:H2O(L), 298.15 K) = -285.83 kJ/mol, s° = 69.95 J/(mol K)",
      abs(_th.enthalpy('IG:H2O(L)', T=298.15) * _h2ol.molar_mass / 1e3 + 285.83) < 0.005
      and abs(_th.entropy('IG:H2O(L)', T=298.15) * _h2ol.molar_mass - 69.95) < 0.01)
check("JANAF N2: cp(300 K) = 29.125 J/(mol K), H(1000) - H(298.15) = 21.460 kJ/mol",
      abs(_th.cp('IG:N2', T=300) * 0.0280134 - 29.125) < 0.01
      and abs((_th.enthalpy('IG:N2', T=1000) - _th.enthalpy('IG:N2', T=298.15)) * 0.0280134 / 1e3 - 21.460) < 0.01)
_R = ideal_gas.R_UNIVERSAL / 0.0280134
check("Δs N2 bei 300 K, 1 -> 10 bar = -R·ln(10)",
      abs(_th.entropy('IG:N2', T=300, p=10e5) - _th.entropy('IG:N2', T=300, p=1e5) + _R * np.log(10)) < 1e-9)
check("Ideales Gas: v = R*T/p, rho = 1/v, u = h - R*T, cv = cp - R, g = h - T*s",
      abs(_th.volume('IG:N2', T=500, p=2e5) - _R * 500 / 2e5) < 1e-12
      and abs(_th.density('IG:N2', T=500, p=2e5) * _th.volume('IG:N2', T=500, p=2e5) - 1) < 1e-12
      and abs(_th.intenergy('IG:N2', T=500) - _th.enthalpy('IG:N2', T=500) + _R * 500) < 1e-6
      and abs(_th.cv('IG:N2', T=500) - _th.cp('IG:N2', T=500) + _R) < 1e-9
      and abs(_th.gibbs('IG:N2', T=500, p=2e5) - _th.enthalpy('IG:N2', T=500)
              + 500 * _th.entropy('IG:N2', T=500, p=2e5)) < 1e-6)
check("enthalpy/cp: p erlaubt, ohne Wirkung; kondensiert: s und g ohne Druckeinfluss",
      _th.enthalpy('IG:N2', T=500, p=7e5) == _th.enthalpy('IG:N2', T=500)
      and _th.entropy('IG:H2O(L)', T=300, p=50e5) == _th.entropy('IG:H2O(L)', T=300))
_h, _s = _th.enthalpy('IG:CO2', T=812.3), _th.entropy('IG:CO2', T=812.3, p=3e5)
check("Umkehrfunktionen: temperature aus h, aus s und p; pressure aus T und s, aus T und rho",
      abs(_th.temperature('IG:CO2', h=_h) - 812.3) < 1e-7
      and abs(_th.temperature('IG:CO2', s=_s, p=3e5) - 812.3) < 1e-7
      and abs(_th.pressure('IG:CO2', T=812.3, s=_s) - 3e5) < 1e-4
      and abs(_th.pressure('IG:CO2', T=812.3, rho=_th.density('IG:CO2', T=812.3, p=3e5)) - 3e5) < 1e-6)
_T = np.array([300.0, 1500.0, 7000.0])
check("Vektorisiert über alle T-Bereiche (Parameterstudien)",
      np.allclose(_th.enthalpy('IG:N2', T=_T), [_th.enthalpy('IG:N2', T=t) for t in _T]))
check("molarmass: IG in kg/mol (SI), CoolProp-Fluide ebenso",
      abs(_th.molarmass('IG:CH4') - 0.01604246) < 1e-10
      and abs(_th.molarmass('water') - _CP.PropsSI('M', 'Water')) < 1e-15)
check("Namen: exakt, sonst ohne Groß-/Kleinschreibung (ig:n2, IG::N2, IG:h2o(l))",
      _th.enthalpy('ig:n2', T=400) == _th.enthalpy('IG::N2', T=400) == _th.enthalpy('IG:N2', T=400)
      and _th.enthalpy('IG:h2o(l)', T=300) == _th.enthalpy('IG:H2O(L)', T=300))
check("CoolProp unverändert: enthalpy(water, ...) ohne Präfix",
      _th.enthalpy('water', T=300, p=1e5) == _CP.PropsSI('H', 'T', 300, 'P', 1e5, 'Water'))
for _text, _part in (("h = enthalpy(IG:co, T=300 K)", "IG:co ist nicht eindeutig: IG:CO, IG:Co"),
                     ("h = enthalpy(IG:Cs2, T=300 K)\nh_2 = enthalpy(IG:cs2, T=300 K)", "IG:CS2, IG:Cs2"),
                     ("s = entropy(IG:N2, T=300 K)", "Druck p angeben"),
                     ("g = gibbs(IG:CO2, T=300 K)", "Druck p angeben"),
                     ("v = volume(IG:H2O(L), T=300 K, p=1 bar)", "für kondensierte Stoffe nicht verfügbar"),
                     ("c = cv(IG:H2O(L), T=300 K)", "für kondensierte Stoffe nicht verfügbar"),
                     ("h = enthalpy(IG:N2, T=300 K, x=1)", "Dampfgehalt gibt es bei IG:-Stoffen nicht"),
                     ("h = enthalpy(IG:N2, T=300 K, h=1)", "hängen bei idealen Gasen beide nur von T ab"),
                     ("mu = viscosity(IG:N2, T=300 K, p=1 bar)", "für IG:-Stoffe nicht verfügbar"),
                     ("h = enthalpy(IG:isooctane, T=300 K)", "ähnlich: IG:C8H18,isooctane"),
                     ("M = molarmass(IG:N2, T=300 K)", "keine Zustandsgrößen")):
    check(f"IG-Meldung beim Einlesen: {_text.splitlines()[-1]!r}", _part in parse_error(_text), parse_error(_text))
try:
    _th.enthalpy('IG:N2', T=150.0)
    check("T außerhalb des Datenbereichs -> Meldung", False)
except ValueError as _exc:
    check("T außerhalb des Datenbereichs -> Meldung mit Bereich", "200 ... 20000 K" in str(_exc), str(_exc))
try:
    _parse_reference("REFERENCE IG:N2 IIR")
    check("REFERENCE IG:... -> Fehler", False)
except EquationSyntaxError as _exc:
    check("REFERENCE IG:... -> Fehler (Bildungsenthalpie, absolute Entropie)", "gilt nicht für IG:" in str(_exc))
set_reference_states(parse_reference_states("REFERENCE water NBP"))
_h_ig = _th.enthalpy('IG:H2O', T=400)
_g_ref = _th.gibbs('water', T=300, p=1e5)
_g_calc = _th.enthalpy('water', T=300, p=1e5) - 300 * _th.entropy('water', T=300, p=1e5)
set_reference_states({})
check("REFERENCE water verschiebt IG:H2O nicht", _h_ig == _th.enthalpy('IG:H2O', T=400))
check("gibbs(water) mit REFERENCE = h - T*s im Bezugszustand", abs(_g_ref - _g_calc) < 1e-6, f"{_g_ref} {_g_calc}")
# Stoffnamen mit Klammern und Komma bleiben ein Argument, keine Variablen
eqs, variables, consts, _, orig, uvals = parse_equations(
    "h = enthalpy(IG:C8H18,isooctane, T=500 K)\nh_l = enthalpy(IG:H2O(L), T=T_w)\nT_w = 300 K\n"
    "h_a = enthalpy(IG:ZnSO4(a'), T=T_z)\nT_z = 1000 K")
ok, sol, msg = solve_system(eqs, variables, constants=consts, original_equations=orig)
_units = propagate_all_units_complete(orig, {k: u.calc_unit for k, u in uvals.items() if u.calc_unit})
check("IG:C8H18,isooctane, IG:H2O(L), IG:ZnSO4(a'): ein Argument, keine Variablen daraus, Einheit J/kg",
      variables == {'h', 'h_l', 'h_a'} and ok
      and abs(sol['h'] - ideal_gas.calculate_property('enthalpy', 'C8H18,isooctane', T=500)) < 1e-9
      and abs(sol['h_a'] - ideal_gas.calculate_property('enthalpy', "ZnSO4(a')", T=1000)) < 1e-9
      and _units.get('h_a') == 'J/kg',
      f"{variables} {msg} {_units}")
check("Zeile mit '=' nur im Aufruf (Fortsetzung ohne Klammer) wird gemeldet, nicht still verworfen",
      "geht nur in einer offenen Klammer" in parse_error("a = 1\nb = a + 2\n  + enthalpy(water, T=a, p=1e5)"),
      parse_error("a = 1\nb = a + 2\n  + enthalpy(water, T=a, p=1e5)"))
eqs, variables, consts, _, orig, _ = parse_equations(
    "T_1 = 300 K\np_1 = 1 bar\np_2 = 10 bar\ns_1 = entropy(IG:N2, T=T_1, p=p_1)\n"
    "s_1 = entropy(IG:N2, T=T_2, p=p_2)")
ok, sol, msg = solve_system(eqs, variables, constants=consts, original_equations=orig)
check("Implizit: T_2 der Isentrope aus entropy(IG:N2, T=T_2, p=p_2) = s_1",
      ok and abs(sol['T_2'] - _th.temperature('IG:N2', s=_th.entropy('IG:N2', T=300, p=1e5), p=10e5)) < 1e-6,
      msg)

# ---------------------------------------------------------------------------
print("=== Chemisches Gleichgewicht (EQUILIBRIUM, Gibbs-Minimierung) ===")
import equilibrium as _eqm
from parser import parse_equilibria
from scipy.optimize import fsolve as _fsolve

_GG_HEAD = "n_CH4_1 = 1 mol\nn_O2_1 = 0.66 mol\nn_N2_1 = n_O2_1*0.79/0.21\nT = 745 °C\np = 1 bar\n"
_GG_OUT = "OUT IG:CH4 = a, IG:H2O = b, IG:CO = c, IG:CO2 = d, IG:H2 = e, IG:N2 = f"
for _form in ("EQUILIBRIUM T=T, p=p\n  IN IG:CH4 = n_CH4_1, IG:O2 = n_O2_1, IG:N2 = n_N2_1\n  " + _GG_OUT,
              "EQUILIBRIUM T=T, p=p IN IG:CH4 = n_CH4_1, IG:O2 = n_O2_1, IG:N2 = n_N2_1 " + _GG_OUT,
              "equilibrium t=T, p=p\n  in IG:CH4 = n_CH4_1, IG:O2 = n_O2_1,\n     IG:N2 = n_N2_1\n  out IG:CH4 = a,\n"
              "      IG:H2O = b, IG:CO = c, IG:CO2 = d, IG:H2 = e, IG:N2 = f"):
    _st = parse_equilibria(_GG_HEAD + _form)
    check(f"EQUILIBRIUM einlesen: {_form.splitlines()[0][:40]!r}",
          len(_st) == 1 and _st[0].line == 6 and [n for n, _ in _st[0].outputs][0] == 'IG:CH4'
          and len(_st[0].equations()) == 6, str(_st))
# Gibbs-Minimum = Gleichgewichtskonstanten (unabhängig: zwei K-Gleichungen mit fsolve, g° aus ideal_gas)
eqs, variables, consts, _, orig, uvals = parse_equations(
    _GG_HEAD + "EQUILIBRIUM T=T, p=p\n  IN IG:CH4 = n_CH4_1, IG:O2 = n_O2_1, IG:N2 = n_N2_1\n  " + _GG_OUT)
ok, sol, msg = solve_system(eqs, variables, constants=consts, original_equations=orig)
_g = {n: ideal_gas.standard_gibbs_RT(ideal_gas.find_species(n), 1018.15) for n in ('CH4', 'H2O', 'CO', 'CO2', 'H2')}
_K_SR, _K_WGS = np.exp(-(_g['CO'] + 3 * _g['H2'] - _g['CH4'] - _g['H2O'])), np.exp(-(_g['CO2'] + _g['H2'] - _g['CO'] - _g['H2O']))
def _kform(v):
    ch4, co, co2 = v
    h2o = 2 * 0.66 - co - 2 * co2
    h2 = (4 - 4 * ch4 - 2 * h2o) / 2
    n = ch4 + h2o + co + co2 + h2 + 0.66 * 0.79 / 0.21
    return [ch4 + co + co2 - 1, co * h2 ** 3 / (ch4 * h2o * n ** 2) / _K_SR - 1, co2 * h2 / (co * h2o) / _K_WGS - 1]
_ref = _fsolve(_kform, [0.02, 0.85, 0.13], xtol=1e-14)
check("EQUILIBRIUM = Gleichgewichtskonstanten (Dampfreformierung, Shift) bei 745 °C",
      ok and np.allclose([sol['a'], sol['c'], sol['d']], _ref, rtol=1e-8), f"{msg} {[sol.get(k) for k in 'acd']} {_ref}")
_units = propagate_all_units_complete(orig, {k: u.calc_unit for k, u in uvals.items() if u.calc_unit})
check("EQUILIBRIUM: Austrittsmengen in der Einheit der Eintrittsmengen (mol), T absolut",
      all(_units.get(v) in ('mol', 'mole') for v in 'abcdef') and _units.get('T') == 'K', str(_units))
# Spurenanteil und Positivität (Knallgas mit H2-Überschuss: O2 ~ 1e-20)
_m = _eqm.equilibrium_amounts(['IG:H2', 'IG:O2', 'IG:H2O'], ['IG:H2', 'IG:O2'], 1000.0, 1e5, [1.0, 0.25])
_Kp = np.exp(-(ideal_gas.standard_gibbs_RT(ideal_gas.find_species('H2O'), 1000.0)
               - ideal_gas.standard_gibbs_RT(ideal_gas.find_species('H2'), 1000.0)
               - 0.5 * ideal_gas.standard_gibbs_RT(ideal_gas.find_species('O2'), 1000.0)))
_x = {k: v / sum(_m.values()) for k, v in _m.items()}
check("Spurenanteil O2 ~ 1e-20 positiv und genau (K der Knallgasreaktion)",
      _m['IG:O2'] > 0 and abs(_x['IG:H2O'] / (_x['IG:H2'] * _x['IG:O2'] ** 0.5) / _Kp - 1) < 1e-9, str(_m))
# Ruß: Methanspaltung CH4 = C(gr) + 2 H2, K = x_H2^2/x_CH4 bei 1 bar (geschlossen)
_K = np.exp(-(ideal_gas.standard_gibbs_RT(ideal_gas.find_species('C(gr)'), 1000.0)
              + 2 * ideal_gas.standard_gibbs_RT(ideal_gas.find_species('H2'), 1000.0)
              - ideal_gas.standard_gibbs_RT(ideal_gas.find_species('CH4'), 1000.0)))
_y = (-_K + np.sqrt(_K ** 2 + 4 * _K)) / 2          # x_H2: x^2/(1 - x) = K
_c = _eqm.equilibrium_amounts(['IG:CH4', 'IG:H2', 'IG:C(gr)'], ['IG:CH4'], 1000.0, 1e5, [1.0])
check("Ruß: Methanspaltung bei 1000 K (C(gr) erscheint, x_H2 wie K)",
      _c['IG:C(gr)'] > 0.5 and abs(_c['IG:H2'] / (_c['IG:H2'] + _c['IG:CH4']) - _y) < 1e-10, f"{_c} {_y}")
_c2 = _eqm.equilibrium_amounts(['IG:CO', 'IG:CO2', 'IG:O2', 'IG:C(gr)'], ['IG:CO2'], 1500.0, 1e5, [1.0])
check("Ruß nur, wenn stabil: CO2 bei 1500 K bildet keinen C(gr)", _c2['IG:C(gr)'] == 0.0, str(_c2))
# Kondensat: H2O(L) - Partialdruck = Sättigungsdruck aus g°(Flüssigkeit) = g°(Gas) + ln(p_s/p°)
_w = _eqm.equilibrium_amounts(['IG:N2', 'IG:H2O', 'IG:H2O(L)'], ['IG:N2', 'IG:H2O'], 330.0, 1e5, [1.0, 0.3])
_ps = 1e5 * np.exp(ideal_gas.standard_gibbs_RT(ideal_gas.find_species('H2O(L)'), 330.0)
                   - ideal_gas.standard_gibbs_RT(ideal_gas.find_species('H2O'), 330.0))
check("Kondensat: H2O(L) erscheint, Partialdruck = Sättigungsdruck der Daten",
      _w['IG:H2O(L)'] > 0 and abs(_w['IG:H2O'] / (_w['IG:N2'] + _w['IG:H2O']) * 1e5 / _ps - 1) < 1e-9, f"{_w} {_ps}")
_w2 = _eqm.equilibrium_amounts(['IG:N2', 'IG:H2O', 'IG:H2O(L)'], ['IG:N2', 'IG:H2O'], 330.0, 1e5, [1.0, 0.05])
check("Kein Kondensat unter dem Sättigungsdruck", _w2['IG:H2O(L)'] == 0.0 and abs(_w2['IG:H2O'] - 0.05) < 1e-15, str(_w2))
# Adiabater CPOX-Reformer (Beispiel_02): T_2 unbekannt, OHNE Startwerte (Energiebilanz außen, Gleichgewicht innen)
_cpox = ("n_CH4_1 = 0.003717 mol/s\nlambda = 0.34\nn_O2_1 = 2*lambda*n_CH4_1\nn_N2_1 = n_O2_1*0.79/0.21\nT_1 = 20 °C\n"
         "p = 1 bar\nEQUILIBRIUM T=T_2, p=p\n  IN IG:CH4 = n_CH4_1, IG:O2 = n_O2_1, IG:N2 = n_N2_1\n"
         "  OUT IG:CH4 = n_CH4, IG:H2O = n_H2O, IG:CO = n_CO, IG:CO2 = n_CO2, IG:H2 = n_H2, IG:N2 = n_N2\n"
         "H_1 = n_CH4_1*enthalpy(IG:CH4, T=T_1)*molarmass(IG:CH4) + n_O2_1*enthalpy(IG:O2, T=T_1)*molarmass(IG:O2)"
         " + n_N2_1*enthalpy(IG:N2, T=T_1)*molarmass(IG:N2)\n"
         "H_1 = (n_CH4*enthalpy(IG:CH4, T=T_2)*molarmass(IG:CH4) + n_H2O*enthalpy(IG:H2O, T=T_2)*molarmass(IG:H2O)\n"
         "  + n_CO*enthalpy(IG:CO, T=T_2)*molarmass(IG:CO) + n_CO2*enthalpy(IG:CO2, T=T_2)*molarmass(IG:CO2)\n"
         "  + n_H2*enthalpy(IG:H2, T=T_2)*molarmass(IG:H2) + n_N2*enthalpy(IG:N2, T=T_2)*molarmass(IG:N2))")
_t0 = time.time()
eqs, variables, consts, _, orig, _ = parse_equations(_cpox)
ok, sol, msg = solve_system(eqs, variables, constants=consts, original_equations=orig)
_elapsed = time.time() - _t0
_H_out = sum(sol[v] * _th.enthalpy(f'IG:{s}', T=sol.get('T_2', 300.0)) * _th.molarmass(f'IG:{s}')
             for v, s in (('n_CH4', 'CH4'), ('n_H2O', 'H2O'), ('n_CO', 'CO'), ('n_CO2', 'CO2'), ('n_H2', 'H2'), ('n_N2', 'N2'))) if ok else 0
check("Adiabater CPOX ohne Startwerte: T_2 = 1047.87 K, Energiebilanz erfüllt",
      ok and abs(sol['T_2'] - 1047.8677) < 1e-3 and abs(_H_out / sol['H_1'] - 1) < 1e-9 and _elapsed < 20,
      f"{msg} T_2={sol.get('T_2')} {_elapsed:.1f} s")
for _text, _part in (("T = 1000 K\np = 1 bar\nn = 1 mol\nEQUILIBRIUM T=T, p=p\n  OUT IG:CH4 = a", "genau einen Teil IN"),
                     ("T = 1000 K\nn = 1 mol\nEQUILIBRIUM T=T\n  IN IG:CH4 = n\n  OUT IG:CH4 = a", "p=... angeben"),
                     ("T = 1000 K\np = 1 bar\nn = 1 mol\nEQUILIBRIUM T=T, p=p\n  IN IG:CH4 = n\n  OUT IG:H2 = b",
                      "Element C der Eintrittsstoffe"),
                     ("T = 1000 K\np = 1 bar\nEQUILIBRIUM T=T, p=p\n  IN IG:CH4 = 2 kg\n  OUT IG:CH4 = a, IG:H2 = b, IG:C(gr) = c",
                      "keine Stoffmenge"),
                     ("p = 1 bar\nn = 1 mol\nEQUILIBRIUM T=5 bar, p=p\n  IN IG:CH4 = n\n  OUT IG:CH4 = a, IG:H2 = b, IG:C(gr) = c",
                      "erwartet: K"),
                     ("T = 1000 K\np = 1 bar\nn = 1 mol\nEQUILIBRIUM T=T, p=p\n  IN IG:Methan = n\n  OUT IG:CH4 = a",
                      "Zeile 4: EQUILIBRIUM IN: Unbekannter Stoff IG:Methan"),
                     ("T = 1000 K\np = 1 bar\nn = 1 mol\nEQUILIBRIUM T=T, p=p\n  IN IG:CH4 = n\n  OUT IG:C(gr) = c",
                      "mindestens einen gasförmigen Stoff"),
                     ("T = 1000 K\np = 1 bar\nn = 1 mol\nEQUILIBRIUM T=T, p=p\n  IN IG:CH4 = n\n  OUT IG:CH4 = 2*a, IG:H2 = b",
                      "Name der Variable")):
    check(f"EQUILIBRIUM-Meldung: {_part}", _part in parse_error(_text), parse_error(_text))
# Name mit Komma unter OUT, Ein-Zeilen-Form, Parameterstudie über T
eqs, variables, consts, sweeps, orig, _ = parse_equations(
    "n = 1 mol\nT = 800:200:1200 K\np = 1 bar\nEQUILIBRIUM T=T, p=p IN IG:CH4 = n "
    "OUT IG:CH4 = a, IG:H2 = b, IG:C(gr) = c, IG:C2H2,acetylene = d")
from solver import solve_parametric as _solve_parametric
ok, sol, msg = _solve_parametric(eqs, variables, sweeps, {}, constants=consts, original_equations=orig)
check("EQUILIBRIUM: Parameterstudie über T, Name mit Komma (IG:C2H2,acetylene)",
      ok and np.all(np.diff(sol['c']) > 0) and np.all(np.asarray(sol['d']) > 0)
      and np.allclose(np.asarray(sol['a']) + np.asarray(sol['c']) + 2 * np.asarray(sol['d']), 1.0, rtol=1e-12),
      f"{msg} {sol.get('c')}")

# Befunde der Validierung (fc_b): Datenbereich kondensierter Stoffe, Einheiten, Schreibweisen, Meldungen
_c3 = _eqm.equilibrium_amounts(['IG:N2', 'IG:H2O', 'IG:H2O(cr)', 'IG:H2O(L)'], ['IG:N2', 'IG:H2O'], 333.15, 1e5, [1.0, 0.5])
check("Kondensierter Stoff außerhalb seines Datenbereichs wird ausgelassen (H2O(cr) bei 60 °C)",
      _c3['IG:H2O(cr)'] == 0.0 and _c3['IG:H2O(L)'] > 0, str(_c3))
eqs, variables, consts, _, orig, uvals = parse_equations(
    "T = 60 °C\nEQUILIBRIUM T=T, p=1 bar\n  {Eintritt}\n  IN IG:N2 = 1 mol,\n\n     IG:H2O = 0.5 mol\n"
    "  \"Austritt\"\n  OUT IG:N2 = a, IG:H2O = b, IG:H2O(L) = c\nin = 3\nout = b/(a + b)")
ok, sol, msg = solve_system(eqs, variables, constants=consts, original_equations=orig)
_units = propagate_all_units_complete(orig, {k: u.calc_unit for k, u in uvals.items() if u.calc_unit})
check("EQUILIBRIUM: Kommentar-/Leerzeilen in der Anweisung, Variablen in/out, Zahl mit Einheit behält mol",
      ok and _units.get('a') in ('mol', 'mole') and abs(sol['out'] * 1e5 - _ps_60) < 1 if (
          _ps_60 := 1e5 * np.exp(ideal_gas.standard_gibbs_RT(ideal_gas.find_species('H2O(L)'), 333.15)
                                 - ideal_gas.standard_gibbs_RT(ideal_gas.find_species('H2O'), 333.15))) else False,
      f"{msg} {_units.get('a')} {sol.get('out')}")
from parser import readable as _readable
check("EQUILIBRIUM: Meldungen zeigen die Anweisung (EQUILIBRIUM (Zeile n): Stoff = Variable), intern Gleichung",
      any(_readable(f"Fehler in '{o}': x") == "Fehler in 'EQUILIBRIUM (Zeile 2): IG:N2 = a': x" for o in orig.values())
      and all('equilibrium(' in o for o in orig.values() if 'IG:' in o), str(list(orig.values())))
for _text, _part in (("n = 1 mol\nEQUILIBRIUM T=300 K, p=1 bar\n  IN IG:N2 = 0,07 mol/s, IG:O2 = n\n  OUT IG:N2 = a, IG:O2 = b",
                      "Dezimalkomma"),
                     ("n = 1 mol\nEQUILIBRIUM T=20:10:80 °C, p=1 bar\n  IN IG:N2 = n\n  OUT IG:N2 = a", "Parameterstudie"),
                     ("n = 1 mol\nEQUILIBRIUM T=900 K, p=1 bar\n  IN IG:CH4 = n\n  OUT IG:CH4 = a, IG:H2 = a, IG:C(gr) = c",
                      "a steht mehrfach"),
                     ("n = 1 mol\nEQUILIBRIUM T=900 K, p=1 bar\n  IN IG:N2 n\n  OUT IG:N2 = a", "erwartet Stoff = Menge")):
    check(f"EQUILIBRIUM-Meldung: {_part}", _part in parse_error(_text), parse_error(_text))
eqs, variables, consts, _, orig, _ = parse_equations("equilibrium = 3\nx = equilibrium + 1")
check("Variable mit dem Namen equilibrium bleibt möglich", consts.get('equilibrium') == 3.0 and 'x' in variables,
      str((consts, variables)))
# Argument mit falscher Dimension (allgemein): T = 1 bar in enthalpy(), Masse als EQUILIBRIUM-Menge
from unit_constraints import check_all_unit_consistency as _check_units
_w = _check_units({'T': 1e5, 'p': 1e5, 'h': 1.0}, {"(h) - (enthalpy('water', T=T, p=p))": "h = enthalpy(water, T=T, p=p)"},
                  {'T': 'Pa', 'p': 'Pa'})
check("Einheitenwarnung: Druck als T= in enthalpy()", any('T= T hat die Einheit Pa' in str(w.units) for w in _w), str(_w))
_eq_m = "(a) - (equilibrium('IG:N2', 'IG:N2', 'IG:N2', T, p, m))"
_w = _check_units({'a': 1, 'm': 0.002, 'T': 300, 'p': 1e5}, {_eq_m: "EQUILIBRIUM (Zeile 3): IG:N2 = a"},
                  {'m': 'kg/s', 'T': 'K', 'p': 'Pa'})
check("Einheitenwarnung: Masse als Menge in EQUILIBRIUM (Stoffmenge erwartet)",
      any('keine' in str(w.units) or 'Stoffmenge' in str(w.units) for w in _w), str(_w))
# Plateau: Gleichung für einen ganzen Bereich erfüllt -> nicht eindeutig (keine willkürliche Lösung)
for _text, _var in (("y = 3\ny = max(0, x - 5) + 3", 'x'),
                    ("n_N2 = 0.01 mol/s\nn_W = 0.004 mol/s\np = 1 bar\nEQUILIBRIUM T=T, p=p\n  IN IG:N2 = n_N2, IG:H2O = n_W\n"
                     "  OUT IG:N2 = a, IG:H2O = b, IG:H2O(L) = n_L\nn_L = 0 mol/s", 'T')):
    eqs, variables, consts, _, orig, _ = parse_equations(_text)
    ok, sol, msg = solve_system(eqs, variables, constants=consts, original_equations=orig)
    check(f"Plateau -> 'Lösung nicht eindeutig' ({_text.splitlines()[-1]})",
          not ok and 'nicht eindeutig' in msg and f'Bereich von {_var}' in msg, msg)
eqs, variables, consts, _, orig, _ = parse_equations(
    "n_N2 = 0.01 mol/s\nn_W = 0.004 mol/s\np = 1 bar\nEQUILIBRIUM T=T, p=p\n  IN IG:N2 = n_N2, IG:H2O = n_W\n"
    "  OUT IG:N2 = a, IG:H2O = b, IG:H2O(L) = n_L\nn_L = 0.5*n_W")
ok, sol, msg = solve_system(eqs, variables, constants=consts, original_equations=orig)
check("Umkehraufgabe: Temperatur für gegebene Kondensatmenge (eindeutig)",
      ok and abs(sol['n_L'] - 0.002) < 1e-15 and 320 < sol['T'] < 335, f"{msg} {sol.get('T')}")

# Kleine Zielmenge knapp über dem Rauschen der Funktion (Taupunkt über n_L = 1e-9*n_W): Vorzeichenwechsel
# einer stetigen Funktion auf Maschinengenauigkeit ist eine Wurzel (Pol und Sprung bleiben keine)
eqs, variables, consts, _, orig, _ = parse_equations(
    "n_N2 = 0.07 mol/s\nn_O2 = 0.0126 mol/s\nn_W = 0.0226 mol/s\np = 1 bar\nEQUILIBRIUM T=T_tau, p=p\n"
    "  IN IG:N2 = n_N2, IG:O2 = n_O2, IG:H2O = n_W\n  OUT IG:N2 = a, IG:O2 = b, IG:H2O = n_D, IG:H2O(L) = n_L\n"
    "n_L = 1e-9*n_W")
ok, sol, msg = solve_system(eqs, variables, constants=consts, original_equations=orig)
_ps = 1e5 * 0.0226 / (0.07 + 0.0126 + 0.0226)       # Taupunkt: Dampfdruck = Partialdruck des Wassers
_g, _l = ideal_gas.find_species('H2O'), ideal_gas.find_species('H2O(L)')
from scipy.optimize import brentq as _brentq
_T_dew = _brentq(lambda T: 1e5 * np.exp(ideal_gas.standard_gibbs_RT(_l, T) - ideal_gas.standard_gibbs_RT(_g, T)) - _ps,
                300, 360)
check("Taupunkt als Umkehraufgabe mit n_L = 1e-9*n_W (Rauschen der Stoffwerte ~1e-10)",
      ok and abs(sol['T_tau'] - _T_dew) < 1e-6, f"{msg} {sol.get('T_tau')} {_T_dew}")
for _text in ("1/(x-2) = 0", "floor(x) - 2.5 = 0", "y = IF(x, 3, -1, 0, 1)\ny = 0.5"):
    eqs, variables, consts, _, orig, _ = parse_equations(_text)
    ok, sol, msg = solve_system(eqs, variables, constants=consts, original_equations=orig)
    check(f"Pol/Sprung bleibt ohne Lösung: {_text.splitlines()[0]}", not ok, f"{msg} {sol}")
for _text, _part in (("n = 1 mol\nEQUILIBRIUM T=300 K; p=1 bar\n  IN IG:N2 = n\n  OUT IG:N2 = a", "nicht ';'"),
                     ("n = 1 mol\nEQUILIBRIUM T=300 K, p=1 bar\n  IN N2 = n\n  OUT IG:N2 = a", "Präfix IG:"),
                     ("n = 1 mol\nEQUILIBRIUM\n  IN IG:N2 = n\n  OUT IG:N2 = a", "T=... und p=... angeben"),
                     ("n = 1 mol\nEQUILIBRIUM T=300 K, p=1 bar\n  IN IG:N2 = n\n  OUT IG:N2 = a,\ny = 2*a",
                      "Zeile 5 gehört wegen ','"),
                     ("n = 1 mol\nEQUILIBRIUM T=300 K, p=1 bar\n  IN IG:N2 = 0.5*n + 0.2 mol\n  OUT IG:N2 = a",
                      "Menge von IG:N2 = 0.5*n + 0.2 mol: Syntaxfehler")):
    check(f"EQUILIBRIUM-Meldung: {_part}", _part in parse_error(_text), parse_error(_text))
# Masse statt Stoffmenge: nur die Warnung mit der Ursache, keine Folgewarnung "Inkompatible Terme"
_eqs, _v, _c, _, _orig, _uv = parse_equations(
    "m = 2 g/s\nn_W = 0.004 mol/s\nEQUILIBRIUM T=300 K, p=1 bar\n  IN IG:N2 = m, IG:H2O = n_W\n"
    "  OUT IG:N2 = a, IG:H2O = b, IG:H2O(L) = c")
_ok, _sol, _ = solve_system(_eqs, _v, constants=_c, original_equations=_orig)
_w = _check_units({**_c, **_sol}, _orig, {k: u.calc_unit for k, u in _uv.items() if u.calc_unit})
check("Masse als Menge: genau eine Warnung (Stoffmenge erwartet)",
      len(_w) == 1 and 'Stoffmenge' in _w[0].explanation, str([w.explanation for w in _w]))

# Validierung fc_a/fc_c/fc_d: Gasphase verschwindet, unerfüllbare Bilanz, Ströme summieren, Meldungen
_out = ['IG:H2', 'IG:O2', 'IG:H2O', 'IG:H2O(L)']
_n350 = _eqm.equilibrium_amounts(_out, ['IG:H2', 'IG:O2'], 350.0, 1e5, [1.0, 0.5])
_n400 = _eqm.equilibrium_amounts(_out, ['IG:H2', 'IG:O2'], 400.0, 1e5, [1.0, 0.5])
check("Gasphase verschwindet (H2 + O2 stöchiometrisch, 350 K): nur H2O(L), keine Restgase; 400 K nur Dampf",
      _n350['IG:H2O(L)'] == 1.0 and _n350['IG:H2'] == 0 and _n350['IG:H2O'] == 0
      and _n400['IG:H2O(L)'] == 0 and abs(_n400['IG:H2O'] - 1) < 1e-12, f"{_n350} {_n400}")
import time as _time_mod
_t0 = _time_mod.time()
try:
    _eqm.equilibrium_amounts(['IG:H2', 'IG:H2O'], ['IG:H2', 'IG:O2'], 1000.0, 1e5, [1.0, 1.0])
    _msg_inf = ''
except Exception as _exc:
    _msg_inf = str(_exc)
check("Unerfüllbare Elementbilanz: sofort 'Überschuss an O' (statt 'keine Konvergenz' nach 30 s)",
      'Überschuss an O' in _msg_inf and 'IG:O2' in _msg_inf and _time_mod.time() - _t0 < 2, _msg_inf)
eqs, variables, consts, _, orig, _ = parse_equations(
    "n_a = 0.5 mol\nEQUILIBRIUM T=1000 K, p=1 bar\n  IN IG:H2 = 1 mol, IG:H2O = n_a, IG:H2O = 0.3 mol\n"
    "  OUT IG:H2 = a, IG:H2O = b, IG:O2 = c")
ok, sol, msg = solve_system(eqs, variables, constants=consts, original_equations=orig)
check("Derselbe Stoff zweimal unter IN: Mengen addiert (zwei Ströme)", ok and abs(sol['b'] - 0.8) < 1e-9, f"{msg} {sol}")
eqs, variables, consts, _, orig, _ = parse_equations(
    "in = 2 mol\nEQUILIBRIUM T=300 K, p=1 bar IN IG:N2 = in OUT IG:N2 = out\ny = out")
ok, sol, msg = solve_system(eqs, variables, constants=consts, original_equations=orig)
check("IN/OUT nur vor einem Eintrag Stoff = ...: Variable in direkt vor OUT", ok and sol.get('y') == 2.0, f"{msg} {sol}")
eqs, variables, consts, _, orig, _ = parse_equations(
    "n_1 = 1 mol\nn_2 = 0.1 mol\nEQUILIBRIUM T=100 K, p=1 bar\n  IN IG:N2 = n_1, IG:H2O = n_2\n  OUT IG:N2 = a, IG:H2O = b")
ok, sol, msg = solve_system(eqs, variables, constants=consts, original_equations=orig)
check("Gleiche Ursache in mehreren Gleichungen einmal gemeldet", msg.count('außerhalb ihres Datenbereichs') == 1, msg)
check("Anzeige fasst die Gleichungen einer Anweisung zusammen",
      "'EQUILIBRIUM (Zeile 3): IG:N2 = a, IG:H2O = b'" in _readable(msg), _readable(msg))
_w = _check_units({'a': 1, 'n_1': 1, 'n_2': 0.25}, {"(a) - (equilibrium('IG:H2', 'IG:H2|IG:O2', 'IG:H2|IG:H2O', T, p, n_1, n_2))":
                   "EQUILIBRIUM (Zeile 3): IG:H2 = a"}, {'n_1': 'mole', 'n_2': 'mole / second', 'T': 'K', 'p': 'Pa'})
check("IN-Mengen in verschiedenen Einheiten (mol, mol/s): Warnung mit beiden Einheiten",
      any('verschiedenen Einheiten' in w.explanation and 'mol/s' in w.explanation for w in _w), str([w.explanation for w in _w]))
# Meldung der 1-D-Suche nennt die richtige Grenze (gemerkte Werte kopiert: fsolve verändert sein Array)
eqs, variables, consts, _, orig, _ = parse_equations("h_t = -150 kJ/kg\nenthalpy(IG:O2, T=T_x) = h_t")
ok, sol, msg = solve_system(eqs, variables, constants=consts, original_equations=orig, initial_values={'T_x': 350.0})
check("Datenbereich: Meldung nennt die untere Grenze (T = 199 K), nicht 20000 K", 'T = 199 K' in msg, msg)
# Gescheiterter Block nennt den Auswertungsfehler (Stoffwert außerhalb des Datenbereichs)
eqs, variables, consts, _, orig, _ = parse_equations(
    "T_1 = 20 °C\nn_1 = 1 mol\nEQUILIBRIUM T=T_2, p=1 bar\n  IN IG:CH4 = n_1\n  OUT IG:CH4 = n_CH4, IG:H2 = n_H2, IG:C(gr) = n_C\n"
    "n_1*enthalpy(IG:CH4, T=T_1)*molarmass(IG:CH4) = n_CH4*enthalpy(IG:CH4, T=T_2)*molarmass(IG:CH4) + "
    "n_H2*enthalpy(IG:H2, T=T_2)*molarmass(IG:H2) + n_C*enthalpy(IG:C(gr), T=T_2)*molarmass(IG:C(gr))")
ok, sol, msg = solve_system(eqs, variables, constants=consts, original_equations=orig)
check("Block scheitert am Datenbereich von C(gr): Meldung nennt die Ursache", not ok and 'IG:C(gr)' in msg
      and 'außerhalb des Datenbereichs' in msg, msg)
# Plateau skalenfrei, Underflow kein Plateau
for _text, _expect in (("y = 3e-9\ny = max(0, x - 5e-6) + 3e-9", True), ("exp(-x^2) = 0", False)):
    eqs, variables, consts, _, orig, _ = parse_equations(_text)
    ok, sol, msg = solve_system(eqs, variables, constants=consts, original_equations=orig)
    check(f"Plateau-Erkennung skalenfrei: {_text.splitlines()[-1]} -> nicht eindeutig: {_expect}",
          ('nicht eindeutig' in msg) == _expect and not ok, msg)
# Überbestimmter Block mit abhängiger Schleife: eine ihrer Gleichungen wird Prüfgleichung
eqs, variables, consts, _, orig, _ = parse_equations("x + y = 10\n2*x + 2*y = 20\nw = ln(x - 7)\nw = 0.5")
ok, sol, msg = solve_system(eqs, variables, constants=consts, original_equations=orig)
check("Abhängige Schleife im überbestimmten Block: gelöst über die übrige Information",
      ok and abs(sol['x'] - (7 + np.exp(0.5))) < 1e-9 and abs(sol['y'] - (3 - np.exp(0.5))) < 1e-9, f"{msg} {sol}")
_n_gas = _eqm.equilibrium_amounts(['IG:N2', 'IG:O2', 'IG:NO2'], ['IG:N2', 'IG:O2'], 298.15, 1e5, [1.0, 1.0])
check("Gas außerhalb seines Datenbereichs (NO2 ab 300 K) ausgelassen: Menge 0, übrige wie ohne NO2",
      _n_gas['IG:NO2'] == 0 and _n_gas['IG:N2'] == 1.0 and _n_gas['IG:O2'] == 1.0, str(_n_gas))
try:
    _eqm.equilibrium_amounts(['IG:N2'], ['IG:N2'], 100.0, 1e5, [1.0])
    _msg_gas = ''
except Exception as _exc:
    _msg_gas = str(_exc)
check("Alle Stoffe eines Elements außerhalb ihres Datenbereichs: Meldung mit Bereich",
      'fallen weg' in _msg_gas and '200 ... 20000 K' in _msg_gas, _msg_gas)
try:
    _eqm.equilibrium_amounts(['IG:CO', 'IG:CO2', 'IG:H2O'], ['IG:CH4', 'IG:H2O', 'IG:CO', 'IG:CO2', 'IG:H2'], 1000.0, 1e5,
                             [1.0, 2.0, 2.0, 2.0, 2.0])
    _msg_sug = ''
except Exception as _exc:
    _msg_sug = str(_exc)
check("Vorschlag bei Überschuss: nur Eintrittsstoffe, die noch nicht unter OUT stehen",
      '(z.B. IG:CH4, IG:H2)' in _msg_sug, _msg_sug)
_eqs, _v, _c, _, _orig, _uv = parse_equations(
    "n_1 = 1 mol\nn_2 = 1 mol\nn_3 = 1 mol/s\nEQUILIBRIUM T=300 °C, p=1 bar\n  IN IG:CH4 = n_1, IG:H2O = n_2, IG:H2O = n_3\n"
    "  OUT IG:CH4 = a, IG:CO = b, IG:CO2 = c, IG:H2 = d, IG:H2O = e")
_ok, _sol, _ = solve_system(_eqs, _v, constants=_c, original_equations=_orig)
_w = _check_units({**_c, **_sol}, _orig, {k: u.calc_unit for k, u in _uv.items() if u.calc_unit})
check("Gleiche Einheitenwarnung in mehreren Gleichungen einmal (mit allen Gleichungen)",
      len(_w) == 1 and len(_w[0].equations) == 5, str([(w.explanation[:60], len(w.equations)) for w in _w]))
eqs, variables, consts, _, orig, _ = parse_equations("x = ln(-y)\ny = 2")
ok, sol, msg = solve_system(eqs, variables, constants=consts, original_equations=orig)
from diagnostics import describe_unsolved as _describe
_d = _describe(eqs, set(variables), sol, orig)
check("Direkte Auswertung aus festen Werten: kein Hinweis auf Startwerte", 'Eingaben prüfen' in _d
      and 'Startwerte' not in _d, _d)
check("Fortsetzung ohne Klammer: Hinweis auf die Klammerregel",
      'offenen Klammer' in parse_error("a = 1\nb = 2\ny = a + b +\n    a*b"), parse_error("a = 1\nb = 2\ny = a + b +\n    a*b"))

print()
print(f"{len(PASSED)}/{len(PASSED) + len(FAILED)} Tests bestanden")
if FAILED:
    print("FEHLGESCHLAGEN:", *FAILED, sep="\n  - ")
    sys.exit(1)
print("ALLE TESTS OK")
