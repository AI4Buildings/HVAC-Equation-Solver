"""
Tests der Optimierung (MINIMIZE/MAXIMIZE ziel VARY x = a .. b Einheit).

Referenzwerte unabhängig vom Solver: analytisch oder mit scipy direkt an der
geschlossenen Funktion. Ausführen mit:  python3 test_optimierung.py
"""
import math
import sys
import time
import warnings

import numpy as np
from scipy.optimize import brentq, minimize_scalar

warnings.filterwarnings("ignore")

import optimizer
from optimizer import optimize_parametric, optimize_system
from parser import parse_equations, parse_optimization
from unit_constraints import propagate_all_units_complete
from units import initial_values_from_units

PASSED = []
FAILED = []


def check(name, cond, extra=""):
    (PASSED if cond else FAILED).append(name)
    print(("OK  " if cond else "FAIL"), name, extra)


def opt(text, start=None):
    """Löst ein Blatt mit Optimierung wie die GUI (ohne Fenster)."""
    eqs, variables, consts, sweeps, orig, uv = parse_equations(text)
    goals = parse_optimization(text)
    names = {n for g in goals for n in g.names}
    mid = {n: (lo + hi) / 2 for g in goals for n, lo, hi in zip(g.names, g.lower, g.upper)}
    for g in goals:
        uv.update(g.unit_values)
    units = propagate_all_units_complete(orig, {v: u.calc_unit for v, u in uv.items() if u.calc_unit})
    unknowns = variables - names
    init = initial_values_from_units(unknowns, units, {**consts, **mid})
    init.update(start or {})
    if sweeps:
        return optimize_parametric(eqs, unknowns, sweeps, goals, init, constants=consts, original_equations=orig)
    return optimize_system(eqs, unknowns, goals, init, consts, orig)


def error_of(func, *args):
    try:
        func(*args)
        return ""
    except Exception as exc:
        return str(exc)


# ---------------------------------------------------------------------------
print("\n=== Einlesen der Anweisung ===")
# ---------------------------------------------------------------------------
goals = parse_optimization("a = 1\nmaximize e VARY m = 0.5 .. 2 kg/s,\n   n = 500 .. 2000 g/s\nminimize q vary s = 11 .. 30 mm")
check("Groß-/Kleinschreibung, Fortsetzungszeile, zwei Anweisungen",
      [(g.sense, g.objective, g.names, g.line) for g in goals]
      == [('max', 'e', ['m', 'n'], 2), ('min', 'q', ['s'], 4)])
check("Grenzen in SI (kg/s, g/s, mm)",
      goals[0].lower == [0.5, 0.5] and goals[0].upper == [2.0, 2.0]
      and np.allclose(goals[1].lower + goals[1].upper, [0.011, 0.03]))
g = parse_optimization("MINIMIZE f VARY T = 0 .. 50 °C, dT = 5 .. 10 K, lambda = 1 .. 5 µm")[0]
check("°C mit Offset, Differenz in K, Schlüsselwort-Name, µm",
      np.allclose(g.lower, [273.15, 5.0, 1e-6]) and np.allclose(g.upper, [323.15, 10.0, 5e-6])
      and g.names == ['T', 'dT', '_kw_lambda'])
check("Einheit nur an der unteren Grenze gilt für beide",
      parse_optimization("MINIMIZE f VARY s = 11 mm .. 30")[0].upper == [0.03])
check("Anweisung im Kommentar wird ignoriert", parse_optimization("{MAXIMIZE e VARY m = 0 .. 1}\nx = 1") == [])
eqs, variables, consts, _, _, _ = parse_equations("x = 1\nMAXIMIZE y VARY z = 0 .. 1\ny = z*(1 - z) + x")
check("Anweisung ist keine Gleichung (Rest unverändert)", len(eqs) == 1 and variables == {'y', 'z'})
check("Zeilennummern danach bleiben richtig",
      error_of(parse_equations, "MAXIMIZE y VARY z = 0 .. 1,\n  w = 0 .. 1\ny = z*(1 - z\nx = 2").startswith("Zeile 3:"))
for text, expected in (
        ("MAXIMIZE e", "Zeile 1: Optimierung so schreiben"),
        ("MAXIMIZE e VARY m = 2 .. 1", "Untere Grenze von m muss kleiner"),
        ("MAXIMIZE e VARY m = 0,5 .. 2 kg/s", "Dezimalkomma"),
        ("MAXIMIZE e VARY m = a .. 2", "keine Zahlen"),
        ("MAXIMIZE e VARY m 1 .. 2", "Bereich als x = a .. b"),
        ("MAXIMIZE e VARY m = 1 .. 2 qcm", "Unbekannte Einheit 'qcm'"),
        ("MAXIMIZE e VARY e = 1 .. 2", "Zielgröße e kann nicht selbst variiert"),
        ("MAXIMIZE e VARY m = 1 .. 2\nMINIMIZE f VARY m = 1 .. 3", "Zeile 2: m wird schon in Zeile 1 variiert")):
    message = error_of(parse_optimization, text)
    check(f"Fehler: {expected}", expected in message, message)

# ---------------------------------------------------------------------------
print("\n=== Eine Größe: analytische Referenzen ===")
# ---------------------------------------------------------------------------
ok, sol, msg = opt("y = (x - 2)^2 + 1\nMINIMIZE y VARY x = 0 .. 5")
check("Inneres Minimum (x = 2, y = 1)", ok and abs(sol['x'] - 2) < 1e-5 and abs(sol['y'] - 1) < 1e-10, msg)
check("Meldung nennt Minimum und Anzahl der Lösungen", "Minimum von y" in msg and "Lösungen" in msg, msg)
ok, sol, msg = opt("y = x\nMINIMIZE y VARY x = 1 .. 3")
check("Minimum an der Untergrenze (exakt 1, gemeldet)", ok and sol['x'] == 1.0 and "x an der Untergrenze" in msg, msg)
ok, sol, msg = opt("y = x\nMAXIMIZE y VARY x = 1 .. 3")
check("Maximum an der Obergrenze (exakt 3, gemeldet)", ok and sol['x'] == 3.0 and "x an der Obergrenze" in msg, msg)

# Zwei lokale Minima: global bei x < 0 (Referenz: scipy auf dem Teilintervall)
ref = minimize_scalar(lambda x: (x**2 - 4)**2 + x, bounds=(-3, 0), method='bounded', options={'xatol': 1e-12})
ok, sol, msg = opt("y = (x^2 - 4)^2 + x\nMINIMIZE y VARY x = -3 .. 3")
check("Zwei lokale Minima: das globale wird gefunden", ok and abs(sol['x'] - ref.x) < 1e-5, f"{sol.get('x')} {ref.x}")
ok, sol, msg = opt("y = (x^2 - 4)^2 + x\nMINIMIZE y VARY x = -3 .. 3", start={'x': 2.0})
check("... auch mit Startwert beim schlechteren lokalen Minimum", ok and abs(sol['x'] - ref.x) < 1e-5, str(sol.get('x')))

# Zielgröße nur implizit bestimmt (Iteration je Kandidat): y^3 + y = x
def implicit(x):
    y = brentq(lambda y: y**3 + y - x, -10, 10, xtol=1e-14)
    return (y - 1)**2 + 0.1 * x
ref = minimize_scalar(implicit, bounds=(0, 5), method='bounded', options={'xatol': 1e-12})
ok, sol, msg = opt("y^3 + y = x\nf = (y - 1)^2 + 0.1*x\nMINIMIZE f VARY x = 0 .. 5")
check("Zielgröße aus impliziter Gleichung", ok and abs(sol['x'] - ref.x) < 1e-5 and abs(sol['f'] - ref.fun) < 1e-9,
      f"{sol.get('x')} {ref.x}")

# Logarithmisches Raster: Bereich über 6 Zehnerpotenzen, Optimum bei 1e-4
ok, sol, msg = opt("f = (ln(x) - ln(1e-4))^2\nMINIMIZE f VARY x = 1e-6 .. 1")
check("Bereich über viele Zehnerpotenzen (log. Raster)", ok and abs(sol['x'] / 1e-4 - 1) < 1e-4, str(sol.get('x')))

# Kandidaten ohne Lösung (sqrt negativ -> keine reelle Lösung) zählen als schlecht
ok, sol, msg = opt("y^2 = x - 1\nf = (x - 2)^2\nMINIMIZE f VARY x = 0 .. 3")
check("Teilbereich ohne Lösung: Optimum trotzdem, Meldung zählt sie",
      ok and abs(sol['x'] - 2) < 1e-5 and "ohne Lösung" in msg, msg)

# Nicht glatte Zielfunktion (abs, IF)
ok, sol, msg = opt("f = abs(x - 1.3) + IF(x, 2, 0, 0, 5)\nMINIMIZE f VARY x = 0 .. 4")
check("Nicht glatte Zielfunktion (abs, IF)", ok and abs(sol['x'] - 1.3) < 1e-5, str(sol.get('x')))

# ---------------------------------------------------------------------------
print("\n=== Einheiten ===")
# ---------------------------------------------------------------------------
ok, sol, msg = opt("T_0 = 25 °C\nf = (T - T_0)^2\nMINIMIZE f VARY T = 0 .. 50 °C")
check("Grenzen in °C: Optimum 298.15 K", ok and abs(sol['T'] - 298.15) < 1e-4, str(sol.get('T')))
ok, sol, msg = opt("f = (s - 0.0207)^2\nMINIMIZE f VARY s = 11 .. 30 mm")
check("Grenzen in mm: Optimum 0.0207 m", ok and abs(sol['s'] - 0.0207) < 1e-8, str(sol.get('s')))
ok, sol, msg = opt("dT_0 = 7 K\nf = (dT - dT_0)^2\nMINIMIZE f VARY dT = 5 .. 10 K")
check("Temperaturdifferenz in K", ok and abs(sol['dT'] - 7) < 1e-5, str(sol.get('dT')))

# ---------------------------------------------------------------------------
print("\n=== Mehrere Größen ===")
# ---------------------------------------------------------------------------
ok, sol, msg = opt("f = (a - 1)^2 + (b - 2)^2 + a*b\nMINIMIZE f VARY a = -3 .. 3, b = -3 .. 3")
check("Zwei Größen, gekoppelt (a = 0, b = 2, f = 1)",
      ok and abs(sol['a']) < 1e-4 and abs(sol['b'] - 2) < 1e-4 and abs(sol['f'] - 1) < 1e-8, f"{sol.get('a')} {sol.get('b')}")
ok, sol, msg = opt("f = (a - 5)^2 + (b + 1)^2\nMINIMIZE f VARY a = 0 .. 2, b = 0 .. 2")
check("Zwei Größen am Rand (a oben, b unten)",
      ok and sol['a'] == 2.0 and sol['b'] == 0.0 and "a an der Obergrenze" in msg and "b an der Untergrenze" in msg, msg)
ok, sol, msg = opt("f = (a - 1)^2 + (b - 2)^2 + (c - 3)^2 + a*b/10\nMINIMIZE f VARY a = 0 .. 5, b = 0 .. 5, c = 0 .. 5")
# Referenz: lineares Gleichungssystem des Gradienten
A = np.array([[2, 0.1, 0], [0.1, 2, 0], [0, 0, 2]]); rhs = np.array([2, 4, 6]); x_ref = np.linalg.solve(A, rhs)
check("Drei Größen", ok and np.allclose([sol['a'], sol['b'], sol['c']], x_ref, atol=1e-4), str([sol.get(k) for k in 'abc']))

# ---------------------------------------------------------------------------
print("\n=== Mehrere Anweisungen ===")
# ---------------------------------------------------------------------------
ok, sol, msg = opt("f = (a - 1)^2\ng = (b - 2)^2\nh = f + g\nMINIMIZE f VARY a = 0 .. 3\nMINIMIZE g VARY b = 0 .. 3")
check("Zwei unabhängige Anweisungen (eine Runde, keine Kopplungsmeldung)",
      ok and abs(sol['a'] - 1) < 1e-5 and abs(sol['b'] - 2) < 1e-5 and "beeinflussen" not in msg, msg)
ok, sol, msg = opt("f = (a - b)^2 + a\ng = (b - a)^2 - b\nMINIMIZE f VARY a = 0 .. 3\nMINIMIZE g VARY b = 0 .. 3")
# Gleichgewicht: a optimal zu b und b optimal zu a (a = 1, b = 1.5), Hinweis auf die Kopplung
check("Gekoppelte Anweisungen -> Gleichgewicht + Hinweis",
      ok and abs(sol['a'] - 1) < 1e-4 and abs(sol['b'] - 1.5) < 1e-4 and "beeinflussen sich gegenseitig" in msg, msg)
ok, sol, msg = opt("f = (a - b)^2 + a\ng = (b - 2*a)^2 + (b - 3)^2\nMINIMIZE f VARY a = 0 .. 3\nMINIMIZE g VARY b = 0 .. 3")
# Gleichgewicht: a = b - 0.5, b = (2a + 3)/2 -> b = (2b - 1 + 3)/2 = b + 1: keins im Inneren -> Rand
check("Gekoppelte Anweisungen ohne inneres Gleichgewicht -> Hinweis",
      "beeinflussen sich gegenseitig" in msg, msg)

# ---------------------------------------------------------------------------
print("\n=== Fehlerfälle ===")
# ---------------------------------------------------------------------------
ok, sol, msg = opt("y = 3\nx = 2*z\nMINIMIZE y VARY z = 0 .. 1")
check("Zielgröße mit festem Wert -> Meldung", not ok and "hat einen festen Wert" in msg, msg)
ok, sol, msg = opt("x = 2*z\nMINIMIZE q VARY z = 0 .. 1")
check("Zielgröße in keiner Gleichung -> Meldung", not ok and "kommt in keiner Gleichung vor" in msg, msg)
ok, sol, msg = opt("f = 5 + 0*x\nMINIMIZE f VARY x = 0 .. 1")
check("Zielgröße unabhängig von der Größe -> Meldung", not ok and "ändert sich nicht" in msg, msg)
ok, sol, msg = opt("y^2 = -1 - x^2\nMINIMIZE y VARY x = 0 .. 1")
check("Nirgends lösbar -> Meldung", not ok and "für keinen Wert von x im Bereich lösbar" in msg, msg)
goals = parse_optimization("MINIMIZE f VARY a = 0 .. 3\nMINIMIZE a VARY b = 0 .. 3")
ok, sol, msg = optimize_system(["(f) - ((a - 1)**2 + b)"], {'f'}, goals, {}, {}, {})
check("Zielgröße = variierte Größe einer anderen Anweisung -> Meldung",
      not ok and "wird in einer anderen Anweisung variiert" in msg, msg)

old_limit = optimizer.OPTIMIZE_TIME_LIMIT
optimizer.OPTIMIZE_TIME_LIMIT = 0.05
t0 = time.time()
ok, sol, msg = opt("y = exp(b) + enthalpy(water, T=T_1, p=p)/1e6 + c^2\nb^3 + b = x + y/100\nc = sin(b) + x\n"
                   "p = 1 bar\nT_1 = 300 + x\nMINIMIZE y VARY x = 0 .. 50")
optimizer.OPTIMIZE_TIME_LIMIT = old_limit
check("Zeitlimit: Abbruch mit bestem Ergebnis und Meldung",
      not ok and msg.startswith("Zeitlimit der Optimierung") and 'x' in sol and time.time() - t0 < 30, msg[:120])

# ---------------------------------------------------------------------------
print("\n=== Optimierung je Punkt (Parameterstudie / Messdaten) ===")
# ---------------------------------------------------------------------------
ok, sol, msg = opt("a = [1 2 3 2.5]\ny = 2*x\nf = (x - a)^2 + 1\nMINIMIZE f VARY x = 0 .. 5")
check("Optimum je Punkt (x = a)", ok and np.allclose(sol['x'], [1, 2, 3, 2.5], atol=1e-5), str(sol.get('x')))
check("Von der variierten Größe abhängige Größe bleibt Array",
      isinstance(sol['y'], np.ndarray) and np.allclose(sol['y'], 2 * np.array([1, 2, 3, 2.5]), atol=1e-4))
ok, sol, msg = opt("a = 0:1:3\nf = (x - a)^2 + c\nc = 4\nMINIMIZE f VARY x = 0 .. 2")
check("Je Punkt am Rand bzw. innen", ok and np.allclose(sol['x'], [0, 1, 2, 2], atol=1e-5), str(sol.get('x')))
ok, sol, msg = opt("a = [1 5 1.5]\ny^2 = x - a\nf = (y - 1)^2\nMINIMIZE f VARY x = 0 .. 3")
check("Punkt ohne lösbaren Kandidaten wird gemeldet, die anderen gelöst",
      ok and "teilweise" in msg and "Punkt 2" in msg and np.isnan(sol['x'][1])
      and abs(sol['x'][0] - 2) < 1e-5 and abs(sol['x'][2] - 2.5) < 1e-5, msg)

# ---------------------------------------------------------------------------
print("\n=== Mit und ohne Einheiten: gleiches Ergebnis ===")
# ---------------------------------------------------------------------------
# Jedes Blatt einmal mit Einheiten und einmal mit reinen SI-Zahlen (keine Einheit im Blatt).
# Ohne Einheiten fehlen die Startwerte aus den Einheiten - die Berechnung muss trotzdem gehen.

def same(sol_a, sol_b, names, rtol):
    return all(np.allclose(np.asarray(sol_a[n], float), np.asarray(sol_b[n], float), rtol=rtol, atol=0)
               for n in names)


WRG03_MODEL = """
UA = UA_ref*(m_dot_gly/m_dot_gly_ref)^n
C_dot_g = m_dot_g*c_g
C_dot_gly = m_dot_gly*c_gly
C_dot_min = min(C_dot_g, C_dot_gly)
C_r = C_dot_min/max(C_dot_g, C_dot_gly)
NTU = UA/C_dot_min
epsilon = IF(C_r, 1, (1 - exp(-NTU*(1 - C_r)))/(1 - C_r*exp(-NTU*(1 - C_r))), NTU/(1 + NTU), 0)
Q_dot = epsilon*C_dot_min*(T_2 - T_aul)
Q_dot = epsilon*C_dot_min*(T_abl - T_1)
Q_dot = C_dot_gly*(T_2 - T_1)
Q_dot = epsilon_tot*C_dot_g*(T_abl - T_aul)
"""
WRG03_UNITS = ("m_dot_g = 5 kg/s\nc_g = 1007 J/(kg*K)\nc_gly = 3.58 kJ/(kg*K)\nT_aul = -10 °C\nT_abl = 25 °C\n"
               "UA_ref = 10 kW/K\nm_dot_gly_ref = 1 kg/s\nn = 0.4\n" + WRG03_MODEL
               + "MAXIMIZE epsilon_tot VARY m_dot_gly = 0.1 .. 4 kg/s")
WRG03_SI = ("m_dot_g = 5\nc_g = 1007\nc_gly = 3580\nT_aul = 263.15\nT_abl = 298.15\n"
            "UA_ref = 10000\nm_dot_gly_ref = 1\nn = 0.4\n" + WRG03_MODEL
            + "MAXIMIZE epsilon_tot VARY m_dot_gly = 0.1 .. 4")


def wrg03_eps_tot(m):
    """Geschlossene Lösung: zwei gleiche Gegenstrom-WT im Glykolkreislauf."""
    C_g, C_gly = 5 * 1007, m * 3580
    C_min, C_max = min(C_g, C_gly), max(C_g, C_gly)
    NTU, C_r = 10000 * m**0.4 / C_min, C_min / C_max
    eps = (1 - math.exp(-NTU * (1 - C_r))) / (1 - C_r * math.exp(-NTU * (1 - C_r)))
    return C_gly / (C_g * (2 * C_gly / (eps * C_min) - 1))


ref = minimize_scalar(lambda m: -wrg03_eps_tot(m), bounds=(0.1, 4), method='bounded', options={'xatol': 1e-10})
ok1, sol1, msg1 = opt(WRG03_UNITS)
ok2, sol2, msg2 = opt(WRG03_SI)
# Sehr flaches Maximum: die Zielgröße ist auf ~1e-8 relativ genau (Block-Toleranz des Solvers),
# die variierte Größe daher auf ~sqrt(1e-8) = 1e-4 relativ
check("WRG Bsp_03 d) mit Einheiten = geschlossene Lösung",
      ok1 and abs(sol1['m_dot_gly'] / ref.x - 1) < 1e-3 and abs(sol1['epsilon_tot'] + ref.fun) < 1e-9,
      f"{sol1.get('m_dot_gly')} {ref.x}")
check("WRG Bsp_03 d) ohne Einheiten: gleiches Ergebnis",
      ok2 and same(sol1, sol2, ['m_dot_gly', 'epsilon_tot', 'T_1', 'T_2', 'Q_dot'], 1e-8), msg2)

STR04_MODEL = """
F_sun = Blackbody_cumulative(T_sun, lambda_c)
F_p = Blackbody_cumulative(T_p, lambda_c)
alpha_S = eps_1*F_sun + eps_2*(1 - F_sun)
eps_p = eps_1*F_p + eps_2*(1 - F_p)
alpha_S*q_solar = eps_p*sigma*(T_p^4 - T_sky^4) + h*(T_p - T_inf) + q_ab
"""
STR04_UNITS = ("T_inf = 30 °C\nh = 12 W/m^2K\nq_solar = 800 W/m^2\nT_sky = 20 °C\nq_ab = 200 W/m^2\n"
               "T_sun = 5800 K\nsigma = 5.67e-8 W/m^2K^4\neps_1 = 0.9\neps_2 = 0.2\n" + STR04_MODEL
               + "MAXIMIZE T_p VARY lambda_c = 0.5 .. 20 µm")
STR04_SI = ("T_inf = 303.15\nh = 12\nq_solar = 800\nT_sky = 293.15\nq_ab = 200\n"
            "T_sun = 5800\nsigma = 5.67e-8\neps_1 = 0.9\neps_2 = 0.2\n" + STR04_MODEL
            + "MAXIMIZE T_p VARY lambda_c = 5e-7 .. 2e-5")
ok1, sol1, msg1 = opt(STR04_UNITS)
ok2, sol2, msg2 = opt(STR04_SI)
# Referenz: Optimalitätsbedingung dT_p/dlambda_c = 0 (anders formuliert, siehe test_regressions)
check("Strahlung Bsp_04 mit Einheiten (lambda_c = 4.10 µm, T_p = 340.229 K)",
      ok1 and abs(sol1['lambda_c'] / 4.1035e-6 - 1) < 2e-4 and abs(sol1['T_p'] - 340.2292) < 1e-3,
      f"{sol1.get('lambda_c')} {sol1.get('T_p')}")
check("Strahlung Bsp_04 ohne Einheiten: gleiches Ergebnis",
      ok2 and same(sol1, sol2, ['lambda_c', 'T_p', 'eps_p', 'alpha_S'], 1e-8), msg2)

# Kreislaufverbundsystem mit Messdaten (Optimum je Punkt), unabhängiges Python-Modell
KVS_MODEL = """
m_dot_g_AZ = m_dot_air/(1 + x_AUL)
m_dot_g_AF = m_dot_air/(1 + x_ABL)
C_g_AZ = m_dot_g_AZ*(c_g + x_AUL*c_D)
C_g_AF = m_dot_g_AF*(c_g + x_ABL*c_D)
V_dot_gly = m_dot_gly/rho_gly
C_gly = m_dot_gly*c_gly
UA_AZ = 1/(1/(UA_ref*(m_dot_g_AZ/m_dot_g_ref)^n) + 1/(UA_ref*(m_dot_gly/m_dot_gly_ref)^n))
UA_AF = 1/(1/(UA_ref*(m_dot_g_AF/m_dot_g_ref)^n) + 1/(UA_ref*(m_dot_gly/m_dot_gly_ref)^n))
C_min_AZ = min(C_g_AZ, C_gly)
C_r_AZ = C_min_AZ/max(C_g_AZ, C_gly)
NTU_AZ = UA_AZ/C_min_AZ
eps_AZ = (1 - exp(-NTU_AZ*(1 - C_r_AZ)))/(1 - C_r_AZ*exp(-NTU_AZ*(1 - C_r_AZ)))
C_min_AF = min(C_g_AF, C_gly)
C_r_AF = C_min_AF/max(C_g_AF, C_gly)
NTU_AF = UA_AF/C_min_AF
eps_AF = (1 - exp(-NTU_AF*(1 - C_r_AF)))/(1 - C_r_AF*exp(-NTU_AF*(1 - C_r_AF)))
Q_dot_AZ = eps_AZ*C_min_AZ*(T_AUL - T_gly_AZ)
Q_dot_AF = eps_AF*C_min_AF*(T_ABL - T_gly_AF)
Q_dot_AZ = C_gly*(T_gly_AF - T_gly_AZ)
Q_dot_AF = C_gly*(T_gly_AZ - T_gly_AF)
W_dot_pump = 0.6*dp_ref*(V_dot_gly/V_dot_gly_ref)^2*V_dot_gly
V_dot_gly_ref = m_dot_gly_ref/rho_gly
eta = -(abs(Q_dot_AF) - 4*W_dot_pump)
"""
KVS_DATA = [(0.711, 0.00351557, 19.48, 0.00424032, 1.13649), (-8.122, 0.000995369, 20.905, 0.00495422, 1.16097),
            (12.4, 0.0062, 22.1, 0.0071, 0.95)]
KVS_UNITS = ("UA_ref = 5263.5 W/K\nn = 0.7529\nm_dot_g_ref = 1 kg/s\nm_dot_gly_ref = 0.842 kg/s\n"
             "c_gly = 3658 J/(kg*K)\nrho_gly = 1045 kg/m^3\nc_g = 1006 J/(kg*K)\nc_D = 1850 J/(kg*K)\n"
             "dp_ref = 75 kPa\n"
             f"T_AUL = [{' '.join(str(r[0]) for r in KVS_DATA)}] °C\n"
             f"x_AUL = [{' '.join(str(r[1]) for r in KVS_DATA)}]\n"
             f"T_ABL = [{' '.join(str(r[2]) for r in KVS_DATA)}] °C\n"
             f"x_ABL = [{' '.join(str(r[3]) for r in KVS_DATA)}]\n"
             f"m_dot_air = [{' '.join(str(r[4]) for r in KVS_DATA)}] kg/s\n" + KVS_MODEL
             + "MINIMIZE eta VARY m_dot_gly = 10 .. 10000 g/s")
KVS_SI = ("UA_ref = 5263.5\nn = 0.7529\nm_dot_g_ref = 1\nm_dot_gly_ref = 0.842\n"
          "c_gly = 3658\nrho_gly = 1045\nc_g = 1006\nc_D = 1850\ndp_ref = 75000\n"
          f"T_AUL = [{' '.join(repr(r[0] + 273.15) for r in KVS_DATA)}]\n"
          f"x_AUL = [{' '.join(str(r[1]) for r in KVS_DATA)}]\n"
          f"T_ABL = [{' '.join(repr(r[2] + 273.15) for r in KVS_DATA)}]\n"
          f"x_ABL = [{' '.join(str(r[3]) for r in KVS_DATA)}]\n"
          f"m_dot_air = [{' '.join(str(r[4]) for r in KVS_DATA)}]\n" + KVS_MODEL
          + "MINIMIZE eta VARY m_dot_gly = 0.01 .. 10")


def kvs_eta(m, T_AUL, x_AUL, T_ABL, x_ABL, m_air):
    """Unabhängig: bei festem m sind die Glykoltemperaturen linear -> 2x2-System exakt."""
    k = []
    for x in (x_AUL, x_ABL):
        m_g = m_air / (1 + x)
        C_g, C = m_g * (1006 + x * 1850), m * 3658
        UA = 1 / (1 / (5263.5 * m_g**0.7529) + 1 / (5263.5 * (m / 0.842)**0.7529))
        C_min, C_r = min(C_g, C), min(C_g, C) / max(C_g, C)
        NTU = UA / C_min
        k.append((1 - math.exp(-NTU * (1 - C_r))) / (1 - C_r * math.exp(-NTU * (1 - C_r))) * C_min)
    C = m * 3658
    T_a, T_b = np.linalg.solve([[C - k[0], -C], [-C, C - k[1]]], [-k[0] * T_AUL, -k[1] * T_ABL])
    Q_AF = k[1] * (T_ABL - T_b)
    V = m / 1045
    return -(abs(Q_AF) - 4 * 0.6 * 75000 * (V / (0.842 / 1045))**2 * V)


refs = []
for T_a, x_a, T_b, x_b, m_air in KVS_DATA:
    f = lambda m: kvs_eta(m, T_a + 273.15, x_a, T_b + 273.15, x_b, m_air)
    grid = np.geomspace(0.01, 10, 400)
    i = int(np.argmin([f(m) for m in grid]))
    refs.append(minimize_scalar(f, bounds=(grid[max(i - 1, 0)], grid[i + 1]), method='bounded',
                                options={'xatol': 1e-10}).x)
ok1, sol1, msg1 = opt(KVS_UNITS)
ok2, sol2, msg2 = opt(KVS_SI)
check("KVS mit Messdaten (Optimum je Punkt) mit Einheiten = unabhängiges Modell",
      ok1 and np.allclose(sol1['m_dot_gly'], refs, rtol=1e-5), f"{sol1.get('m_dot_gly')} {refs}")
check("KVS mit Messdaten ohne Einheiten: gleiches Ergebnis",
      ok2 and same(sol1, sol2, ['m_dot_gly', 'Q_dot_AF', 'eta', 'T_gly_AZ'], 1e-8), msg2)

# IF und Startwerte-Block mit und ohne Einheiten
ok1, sol1, msg1 = opt("T_1 = 20 °C\nT_2 = 30 °C\nq = 5 W/m^2\nq_2 = IF(T_1, T_2, q, 0, 2*q)\n"
                      "y = (T - T_2)^2 + q_2\nMINIMIZE y VARY T = 0 .. 50 °C")
ok2, sol2, msg2 = opt("T_1 = 293.15\nT_2 = 303.15\nq = 5\nq_2 = IF(T_1, T_2, q, 0, 2*q)\n"
                      "y = (T - T_2)^2 + q_2\nMINIMIZE y VARY T = 273.15 .. 323.15")
check("IF + Optimierung mit/ohne Einheiten gleich (T = 303.15 K, q_2 = q)",
      ok1 and ok2 and abs(sol1['T'] - 303.15) < 1e-4 and same(sol1, sol2, ['T', 'q_2', 'y'], 1e-9))
from parser import parse_equations as _pe, parse_start_values as _psv
from solver import solve_system as _ss
for label, text in (("mit", "T_0 = 300 K\ndT_2 = 100 K^2\n(T - T_0)^2 = dT_2\n{$Startwerte T = 250 K $}"),
                    ("ohne", "T_0 = 300\ndT_2 = 100\n(T - T_0)^2 = dT_2\n{$Startwerte T = 250 $}")):
    eqs, v, c, _, orig, _ = _pe(text)
    ok, sol, msg = _ss(eqs, v, _psv(text), constants=c, original_equations=orig)
    check(f"Startwerte-Block {label} Einheiten wählt die Wurzel T = 290 K", ok and abs(sol['T'] - 290) < 1e-8,
          str(sol.get('T')))

# ---------------------------------------------------------------------------
print("\n=== Determinismus ===")
# ---------------------------------------------------------------------------
results = [opt("y = (x^2 - 4)^2 + x + sin(40*x)/10\nMINIMIZE y VARY x = -3 .. 3")[1]['x'] for _ in range(2)]
check("Gleiches Ergebnis bei Wiederholung", results[0] == results[1], str(results))

print()
print(f"{len(PASSED)}/{len(PASSED) + len(FAILED)} Tests bestanden")
if FAILED:
    print("FEHLGESCHLAGEN:", *FAILED, sep="\n  - ")
    sys.exit(1)
print("ALLE TESTS OK")
