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
check("Null-Term (eps = 0) im Block: gelöst", s_ and abs(sol['x'] ** 2 + sol['y'] - 10) < 1e-9, msg)

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
                       ("dT = [0 10 20] °C", [0, 10, 20]),
                       ("dT = 0:5:20 °C", [0, 5, 10, 15, 20])]:
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
# Wellenlängen: Zahlenliterale nach µm-Regel (Parser), Variablen immer SI (m)
# ---------------------------------------------------------------------------
results = {}
for text in ("E = Eb(1000, 5)", "E = Eb(1000, 5e-6)", "E = Eb(1000, 5 µm)", "L = 5 µm\nE = Eb(1000, L)"):
    eqs, variables, consts, _, orig, _ = parse_equations(text)
    results[text] = solve_system(eqs, variables, constants=consts, original_equations=orig)[1]['E']
check("Wellenlänge: Literal 5, 5e-6, 5 µm und L = 5 µm gleich",
      max(results.values()) - min(results.values()) < 1e-6 * max(results.values()), str(results))
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

print()
print(f"{len(PASSED)}/{len(PASSED) + len(FAILED)} Tests bestanden")
if FAILED:
    print("FEHLGESCHLAGEN:", *FAILED, sep="\n  - ")
    sys.exit(1)
print("ALLE TESTS OK")
