"""
Regressionstests für unit_constraints.py (Einheiten-Propagation und Dimensionsprüfung).

Ausführen mit:  python3 test_unit_constraints.py
"""
import itertools
import json
import os
import subprocess
import sys
import warnings

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
warnings.filterwarnings("ignore")

from unit_constraints import (analyze_equation, check_equation_dimensions, check_all_unit_consistency,
                              compute_expression_dimension, propagate_all_units_complete,
                              infer_units_from_function_arguments, unit_from_dimensionality)
from units import ureg, normalize_unit

PASSED = []
FAILED = []


def check(name, cond, extra=""):
    (PASSED if cond else FAILED).append(name)
    print(("OK  " if cond else "FAIL"), name, extra)


def same_dim(label, expected):
    """Gleiche physikalische Dimension (Labels sind nur Anzeige-Einheiten)."""
    if label is None:
        return False
    if label == '' or expected == '':
        return label == expected
    try:
        a = ureg.Quantity(1.0, normalize_unit(label)).dimensionality
        b = ureg.Quantity(1.0, normalize_unit(expected)).dimensionality
        return a == b
    except Exception:
        return False


def eqs(*lines):
    """{parsed_key: original} - Schlüssel sind für die Propagation irrelevant."""
    return {f"eq{i}": line for i, line in enumerate(lines)}


def propagate(lines, known):
    return propagate_all_units_complete(eqs(*lines), dict(known))


def warnings_for(lines, known):
    return check_all_unit_consistency({}, eqs(*lines), dict(known))


# ---------------------------------------------------------------------------
print("=== #1 Zahlenliterale in Summen sind dimensionsneutral ===")

system = ["T_out = T_s - 2",
          "h_out = enthalpy(water, T=T_out, p=p)",
          "m*(h_in - h_out) = UA*(T_s - T_amb)"]
known = {'p': 'Pa', 'm': 'kg/s', 'T_amb': 'K', 'UA': 'W/K', 'h_in': 'J/kg'}
for order in (system, list(reversed(system))):
    r = propagate(order, known)
    check(f"T_out = T_s - 2 -> T_s, T_out in K (Reihenfolge {order[0][:8]}...)",
          r.get('T_s') == 'K' and r.get('T_out') == 'K', str(r))
r = analyze_equation('T_2 = 10 + T_1', {'T_1': 'K'})
check("T_2 = 10 + T_1 -> K", r.get('T_2') == 'K', str(r))
r = analyze_equation('T_2 = T_1 + 10', {'T_1': 'K'})
check("T_2 = T_1 + 10 -> K", r.get('T_2') == 'K', str(r))
r = analyze_equation('p_2 = 0.2 + p_1', {'p_1': 'Pa'})
check("p_2 = 0.2 + p_1 -> Druck", same_dim(r.get('p_2'), 'Pa'), str(r))
r = analyze_equation('eta = 1 - T_c/T_h', {'T_c': 'K'})
check("eta = 1 - T_c/T_h -> eta '', T_h K", r.get('eta') == '' and r.get('T_h') == 'K', str(r))

# Ende-zu-Ende: Startwerte aus der Propagation -> Lösung unabhängig von der Reihenfolge
try:
    from parser import parse_equations
    from solver import solve_system
    from units import get_initial_from_unit
    base = "p = 1 bar\nm = 0.5 kg/s\nT_amb = 20 °C\nUA = 100 W/K\nh_in = 400 kJ/kg\n"
    for body in ("T_out = T_s - 2\nh_out = enthalpy(water, T=T_out, p=p)\nm*(h_in - h_out) = UA*(T_s - T_amb)\n",
                 "h_out = enthalpy(water, T=T_out, p=p)\nm*(h_in - h_out) = UA*(T_s - T_amb)\nT_out = T_s - 2\n"):
        e, v, consts, sw, orig, uv = parse_equations(base + body)
        ku = {k: x.calc_unit for k, x in uv.items() if x.calc_unit}
        units = propagate_all_units_complete(orig, ku)
        init = {var: get_initial_from_unit(units[var]) for var in v if var in units}
        ok, sol, msg = solve_system(e, v, init, constants=consts, original_equations=orig)
        check("Solve mit T_out = T_s - 2 (beide Reihenfolgen)",
              ok and abs(sol.get('T_s', 0) - sol.get('T_out', 0) - 2) < 1e-6 and sol.get('T_s', 0) > 300,
              msg)
except Exception as ex:  # pragma: no cover
    check("Solve mit T_out = T_s - 2 (beide Reihenfolgen)", False, repr(ex))

# ---------------------------------------------------------------------------
print("=== #4 'e' ist eine normale Variable ===")

r = analyze_equation('R*k*A = e', {'e': 'm', 'k': 'W/(m*K)', 'A': 'm^2'})
check("R*k*A = e -> R in K/W", same_dim(r.get('R'), 'K/W'), str(r))
r = analyze_equation('R = e/(k*A)', {'e': 'm', 'k': 'W/(m*K)', 'A': 'm^2'})
check("R = e/(k*A) -> K/W", same_dim(r.get('R'), 'K/W'), str(r))
w = warnings_for(["R*k*A = e"], {'e': 'm', 'k': 'W/(m*K)', 'A': 'm^2'})
check("R*k*A = e ohne Dimensionsfehler", w == [], str([x.explanation for x in w]))

# ---------------------------------------------------------------------------
print("=== #5 sqrt, symbolische Exponenten, unbekannte Dimensionen ===")

fin = ["m = sqrt(h*P/(k*A_c))", "eta_f = tanh(m*L)/(m*L)"]
fin_known = {'h': 'W/(m^2*K)', 'P': 'm', 'k': 'W/(m*K)', 'A_c': 'm^2', 'L': 'm'}
r = propagate(fin, fin_known)
check("Rippe: m = sqrt(h*P/(k*A_c)) -> 1/m", same_dim(r.get('m'), '1/m'), str(r.get('m')))
check("Rippe: eta_f dimensionslos", r.get('eta_f') == '', str(r.get('eta_f')))
w = warnings_for(fin, fin_known)
check("Rippe: keine Unit-Warnung", w == [], str([x.explanation for x in w]))
fin2 = ["q = h*dT_s", "m = sqrt(h*P/(k*A_c))", "eta_f = tanh(m*L)/(m*L)"]
r = propagate(fin2, {'q': 'W/m^2', 'dT_s': 'delta_K', 'P': 'm', 'k': 'W/(m*K)', 'A_c': 'm^2', 'L': 'm'})
check("Rippe mit implizitem h: m -> 1/m, eta_f ''",
      same_dim(r.get('m'), '1/m') and r.get('eta_f') == '', str(r))
r = analyze_equation('k_f = (h*P/(k*A_c))^(1/2)', fin_known)
check("(…)^(1/2) -> 1/m", same_dim(r.get('k_f'), '1/m'), str(r))
r = analyze_equation('mu = mu_0*(T/T_0)^n', {'mu_0': 'Pa*s', 'T': 'K', 'T_0': 'K', 'n': ''})
check("mu = mu_0*(T/T_0)^n -> Pa*s", same_dim(r.get('mu'), 'Pa*s'), str(r))
r = analyze_equation('x = y^n', {'y': 'm', 'n': ''})
check("x = y^n (y dimensionsbehaftet) -> keine Willkür-Inferenz", 'x' not in r, str(r))
e = check_equation_dimensions('x = y^n', {'x': 'm', 'y': 'm', 'n': ''})
check("x = y^n -> keine Warnung", e is None, str(e))
r = analyze_equation('x = y^-1', {'y': 'm'})
check("x = y^-1 -> 1/m", same_dim(r.get('x'), '1/m'), str(r))
r = analyze_equation('x^0.5 = y', {'y': 'm'})
check("x^0.5 = y -> m^2", same_dim(r.get('x'), 'm^2'), str(r))
r = analyze_equation('x = exp(-a*t)', {'t': 's'})
check("x = exp(-a*t) -> a in 1/s", same_dim(r.get('a'), '1/s') and r.get('x') == '', str(r))

# ---------------------------------------------------------------------------
print("=== #6 Absolute Temperatur vs. Temperaturdifferenz ===")

absolute_cases = [
    ('T_2 = T_1 - dT', {'T_1': 'K', 'dT': 'delta_K'}, 'T_2'),
    ('T_2 = T_1 + dT', {'T_1': 'K', 'dT': 'delta_K'}, 'T_2'),
    ('T_s = T_inf - q/h', {'T_inf': 'K', 'q': 'W/m^2', 'h': 'W/(m^2*K)'}, 'T_s'),
    ('T_si = T_i - q*R_si', {'T_i': 'K', 'q': 'W/m^2', 'R_si': 'm^2*K/W'}, 'T_si'),
    ('T_2 = T_1 - eta*(T_1 - T_2s)', {'T_1': 'K', 'T_2s': 'K', 'eta': ''}, 'T_2'),
    ('T_2 = T_1 - Q/(m*c_p)', {'T_1': 'K', 'Q': 'W', 'm': 'kg/s', 'c_p': 'J/(kg*K)'}, 'T_2'),
    ('T_m = T_1 - (T_1 - T_2)/2', {'T_1': 'K', 'T_2': 'K'}, 'T_m'),
    ('T_3 = (T_1 + T_2)/2', {'T_1': 'K', 'T_2': 'K'}, 'T_3'),
    ('T_m = 0.5*(T_1 + T_2)', {'T_1': 'K', 'T_2': 'K'}, 'T_m'),
    ('T_sat_C = temperature(water, p=p, x=1) - 273.15', {'p': 'Pa'}, 'T_sat_C'),
]
for eq, k, var in absolute_cases:
    r = analyze_equation(eq, k)
    check(f"{eq} -> {var} absolut (K)", r.get(var) == 'K', str(r))

difference_cases = [
    ('theta = T_1 - T_2', {'T_1': 'K', 'T_2': 'K'}, 'theta'),
    ('theta = abs(T_1 - T_2)', {'T_1': 'K', 'T_2': 'K'}, 'theta'),
    ('theta = dT_a + dT_b', {'dT_a': 'delta_K', 'dT_b': 'delta_K'}, 'theta'),
    ('theta = eta*(T_1 - T_2)', {'T_1': 'K', 'T_2': 'K', 'eta': ''}, 'theta'),
    ('LMTD = (th_1 - th_2)/ln(th_1/th_2)', {'th_1': 'delta_K', 'th_2': 'delta_K'}, 'LMTD'),
    ('theta = max(T_1 - T_2, T_1 - T_3)', {'T_1': 'K', 'T_2': 'K', 'T_3': 'K'}, 'theta'),
    ('dT_1 = T_h_in - T_c_out', {'T_h_in': 'K', 'T_c_out': 'K'}, 'dT_1'),
]
for eq, k, var in difference_cases:
    r = analyze_equation(eq, k)
    check(f"{eq} -> {var} Differenz (delta_K)", r.get(var) == 'delta_K', str(r))

r = propagate(["T_1 = T_x - 2", "x = (T_1 - T_0)/2"], {'T_x': 'K', 'T_0': 'K'})
check("Ketten: T_1 = T_x - 2 absolut, x = (T_1-T_0)/2 Differenz",
      r.get('T_1') == 'K' and r.get('x') == 'delta_K', str(r))
r = propagate(["T_m = (T_1 + T_2)/2"], {'T_m': 'K', 'T_1': 'K'})
check("Mittelwert rückwärts: T_2 absolut", r.get('T_2') == 'K', str(r))
r = analyze_equation('p*v = R*T', {'p': 'Pa', 'v': 'm^3/kg', 'R': 'J/(kg*K)'})
check("p*v = R*T -> T absolut (Produkt allein bleibt K)", r.get('T') == 'K', str(r))
r = propagate(["T_w = T_m - Q/(alpha*A)", "theta_w = T_w - T_c"],
              {'T_m': 'K', 'T_c': 'K', 'Q': 'W', 'alpha': 'W/(m^2*K)', 'A': 'm^2'})
check("T_w = T_m - Q/(alpha*A) -> K, theta_w = T_w - T_c -> delta_K",
      r.get('T_w') == 'K' and r.get('theta_w') == 'delta_K', str(r))
r = propagate(["T_2 = T_1 - x"], {'T_1': 'K', 'T_2': 'K'})
check("T_2 = T_1 - x (T_1, T_2 absolut) -> x Differenz", r.get('x') == 'delta_K', str(r))

# ---------------------------------------------------------------------------
print("=== #7 max/min ===")

r = analyze_equation('Q_h = max(0, Q_d)', {'Q_d': 'W'})
check("Q_h = max(0, Q_d) -> Leistung", same_dim(r.get('Q_h'), 'W'), str(r))
r = analyze_equation('Q_h = min(Q_d, 0)', {'Q_d': 'W'})
check("Q_h = min(Q_d, 0) -> Leistung", same_dim(r.get('Q_h'), 'W'), str(r))
u = {'Q': 'W', 'm': 'kg/s', 'c_p': 'J/(kg*K)', 'T_1': 'K', 'T_2': 'K', 'T_3': 'K', 'Q_d': 'W', 'Q_h': 'W'}
e = check_equation_dimensions('Q = m*c_p*max(T_1 - T_2, T_1 - T_3)', u)
check("Q = m*c_p*max(dT...) -> keine Warnung", e is None, str(e))
e = check_equation_dimensions('Q_h = max(0, Q_d)', u)
check("Q_h = max(0, Q_d) -> keine Warnung", e is None, str(e))
e = check_equation_dimensions('Q = m*c_p*max(T_1, Q)', u)
check("max(T_1, Q) -> Dimensionsfehler erkannt", e is not None and e['type'] == 'dimension_mismatch', str(e))

# ---------------------------------------------------------------------------
print("=== #8 Einheiten werden normalisiert (W/m^2K = W/(m^2*K)) ===")

e = check_equation_dimensions('q = h*(T_s - T_inf)', {'q': 'W/m^2', 'h': 'W/m^2K', 'T_s': 'K', 'T_inf': 'K'})
check("W/m^2K wird als W/(m^2*K) gelesen", e is None, str(e))
e = check_equation_dimensions('Q = m*c_p*dT', {'Q': 'W', 'm': 'kg/s', 'c_p': 'J/kgK', 'dT': 'delta_K'})
check("J/kgK wird gelesen", e is None, str(e))
e = check_equation_dimensions('R = L/(k*A)', {'R': 'm^2K/W', 'L': 'm', 'k': 'W/mK', 'A': 'm^2'})
check("m^2K/W vs. L/(k*A) -> Dimensionsfehler erkannt (R ist K/W)", e is not None, str(e))
w = warnings_for(["q = h*(T_s - T_inf)"], {'q': 'W/m^2', 'T_s': 'K', 'T_inf': 'K'})
check("Propagiertes h prüft die eigene Gleichung ohne Warnung", w == [], str([x.explanation for x in w]))

# ---------------------------------------------------------------------------
print("=== #9 Literale in der Dimensionsprüfung ===")

u = {'T_s': 'K', 'T_out': 'K', 'T_1': 'K', 'T_2': 'K', 'm_1': 'kg/s', 'm_2': 'kg/s', 'm_3': 'kg/s',
     'm': 'kg/s', 'Q': 'W', 'F': 'W', 'g': 'm/s^2', 'mass': 'kg', 'c_p': 'J/(kg*K)'}
for eq in ('T_out = T_s - 2', 'T_2 = 10 + T_1', 'm_1 + m_2 - m_3 = 0', 'm_1 + m_2 - m_3 = 0.0',
           '0 = m_1 + m_2 - m_3', 'Q = m*c_p*(T_1 - T_2) + 5'):
    e = check_equation_dimensions(eq, u)
    check(f"'{eq}' -> keine Warnung", e is None, str(e))
e = check_equation_dimensions('Q = m + T_1', u)
check("'Q = m + T_1' -> Inkompatible Terme", e is not None and e['type'] == 'dimension_mismatch', str(e))
e = check_equation_dimensions('F = mass*g', u)
check("'F = mass*g' mit F: W -> Dimensionsfehler", e is not None, str(e))
d, missing = compute_expression_dimension('T_s - 2', u)
check("compute_expression_dimension('T_s - 2') = [temperature]",
      d == ureg.kelvin.dimensionality and missing == [], str(d))
w = warnings_for(["T_out = T_s - 2", "h_out = enthalpy(water, T=T_out, p=p)"], {'p': 'Pa'})
check("System mit T_out = T_s - 2 -> keine Unit-Warnung", w == [], str([x.explanation for x in w]))

# ---------------------------------------------------------------------------
print("=== #10 Geklammerte linke Seite ist kein Solver-Format ===")

r = analyze_equation('(m*h_1) - (m*h_2) = Q', {'m': 'kg/s', 'h_1': 'J/kg', 'h_2': 'J/kg'})
check("(m*h_1) - (m*h_2) = Q -> Q Leistung", same_dim(r.get('Q'), 'W'), str(r))
r = analyze_equation('(h_1 - h_2) - (h_3 - h_4) = 0', {'h_1': 'J/kg', 'h_2': 'J/kg', 'h_3': 'J/kg'})
check("(h_1 - h_2) - (h_3 - h_4) = 0 -> h_4", same_dim(r.get('h_4'), 'J/kg'), str(r))
r = analyze_equation('(h_2) - (h_1 + dh)', {'h_1': 'kJ/kg', 'dh': 'kJ/kg'})
check("Solver-Format (h_2) - (h_1 + dh) weiterhin unterstützt", same_dim(r.get('h_2'), 'J/kg'), str(r))
w = warnings_for(["(m*h_1) - (m*h_2) = Q"], {'m': 'kg/s', 'h_1': 'J/kg', 'h_2': 'J/kg'})
check("(m*h_1) - (m*h_2) = Q -> keine 'Einheit unbekannt'", w == [], str([x.explanation for x in w]))

# ---------------------------------------------------------------------------
print("=== #11 Funktionsargumente bei beliebiger linker Seite ===")

r = infer_units_from_function_arguments('Q/m = enthalpy(water, T=T_2, p=p) - h_1', {})
check("Q/m = enthalpy(..., T=T_2, p=p) -> T_2: K, p: Pa", r.get('T_2') == 'K' and r.get('p') == 'Pa', str(r))
r = propagate(["Q/m = enthalpy(water, T=T_2, p=p) - h_1"], {'Q': 'W', 'm': 'kg/s', 'h_1': 'J/kg'})
check("Propagation: T_2 aus Funktionsargument", r.get('T_2') == 'K', str(r))
r = infer_units_from_function_arguments('E = Eb(T_s, L)', {})
check("Eb(T_s, L) -> T_s: K, L: Wellenlänge", r.get('T_s') == 'K' and same_dim(r.get('L'), 'm'), str(r))
r = propagate(["h_m = enthalpy(water, T=(T_1 + T_2)/2, p=p)"], {'T_1': 'K', 'p': 'Pa'})
check("T=(T_1 + T_2)/2 -> T_2 absolut", r.get('T_2') == 'K', str(r))

# ---------------------------------------------------------------------------
print("=== #12 Reihenfolge- und Hash-Unabhängigkeit ===")

r1 = propagate(["theta = T_1 - T_2", "T_1 = temperature(water, p=p, x=1)"], {'T_2': 'K', 'p': 'Pa'})
r2 = propagate(["T_1 = temperature(water, p=p, x=1)", "theta = T_1 - T_2"], {'T_2': 'K', 'p': 'Pa'})
check("theta = T_1 - T_2 vor/nach Definition von T_1 -> delta_K",
      r1.get('theta') == 'delta_K' and r2.get('theta') == 'delta_K', f"{r1.get('theta')} / {r2.get('theta')}")

hx = ["dT_1 = T_h_in - T_c_out", "dT_2 = T_h_out - T_c_in", "dT_lm = (dT_1 - dT_2)/ln(dT_1/dT_2)",
      "Q = U*A*dT_lm", "Q = m_h*c_p*(T_h_in - T_h_out)", "T_m = (T_h_in + T_h_out)/2",
      "T_w = T_m - Q/(alpha*A)", "theta_w = T_w - T_c_in"]
hx_known = {'T_h_in': 'K', 'T_h_out': 'K', 'T_c_in': 'K', 'T_c_out': 'K', 'U': 'W/(m^2*K)',
            'Q': 'W', 'c_p': 'J/(kg*K)', 'alpha': 'W/(m^2*K)'}
results = set()
for perm in itertools.islice(itertools.permutations(hx), 0, 5040, 97):
    results.add(json.dumps(propagate(list(perm), hx_known), sort_keys=True))
check("Wärmeübertrager: alle Permutationen identisch", len(results) == 1, f"{len(results)} Varianten")
ref = json.loads(next(iter(results)))
check("Wärmeübertrager: Einheiten korrekt",
      ref.get('dT_1') == 'delta_K' and ref.get('dT_lm') == 'delta_K' and same_dim(ref.get('A'), 'm^2')
      and same_dim(ref.get('m_h'), 'kg/s') and ref.get('T_m') == 'K' and ref.get('T_w') == 'K'
      and ref.get('theta_w') == 'delta_K', str(ref))

script = ("import json,sys; sys.path.insert(0,'.');"
          "from unit_constraints import propagate_all_units_complete as p;"
          f"print(json.dumps(p({{f'e{{i}}': e for i, e in enumerate({hx!r})}}, {hx_known!r}), sort_keys=True))")
outs = set()
for seed in ('0', '1', '42', '12345'):
    env = dict(os.environ, PYTHONHASHSEED=seed)
    res = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, env=env, cwd=PROJECT_DIR)
    outs.add(res.stdout.strip())
check("Unabhängig von PYTHONHASHSEED", len(outs) == 1 and next(iter(outs)) != '', str(len(outs)))

# ---------------------------------------------------------------------------
print("=== Labels für bisher nicht abgebildete Dimensionen ===")

label_cases = [('R = L/(k*A)', {'L': 'm', 'k': 'W/(m*K)', 'A': 'm^2'}, 'R', 'K/W'),
               ('UA = Q/dT', {'Q': 'W', 'dT': 'delta_K'}, 'UA', 'W/K'),
               ('C = m*c_p', {'m': 'kg', 'c_p': 'J/(kg*K)'}, 'C', 'J/K'),
               ('beta = 1/T', {'T': 'K'}, 'beta', '1/K'),
               ('nu = mu/rho', {'mu': 'Pa*s', 'rho': 'kg/m^3'}, 'nu', 'm^2/s'),
               ('mu = nu*rho', {'nu': 'm^2/s', 'rho': 'kg/m^3'}, 'mu', 'Pa*s'),
               ('q_v = Q/V', {'Q': 'W', 'V': 'm^3'}, 'q_v', 'W/m^3')]
for eq, k, var, expected in label_cases:
    r = analyze_equation(eq, k)
    check(f"{eq} -> Label '{expected}'", r.get(var) == expected, str(r))
lbl = unit_from_dimensionality(ureg.Quantity(1.0, 'kg*m/(s^3*K)').dimensionality)
check("Generisches SI-Label lesbar und pint-parsebar",
      'kelvin' not in lbl and ureg.Quantity(1.0, lbl).dimensionality == ureg.Quantity(1.0, 'kg*m/(s^3*K)').dimensionality,
      lbl)

# ---------------------------------------------------------------------------
print("=== ln(T_2) - ln(T_1) ===")

r = analyze_equation('ds = cp*(ln(T_2) - ln(T_1))', {'cp': 'J/(kg*K)', 'T_1': 'K', 'ds': 'J/(kg*K)'})
check("ds = cp*(ln(T_2) - ln(T_1)) -> T_2 K", r.get('T_2') == 'K', str(r))
r = analyze_equation('ds = cp*ln(T_2) - cp*ln(T_1)', {'cp': 'J/(kg*K)', 'T_1': 'K'})
check("ds = cp*ln(T_2) - cp*ln(T_1) -> T_2 K", r.get('T_2') == 'K', str(r))
r = analyze_equation('x*ln(x) = r', {'r': ''})
check("x*ln(x) = r -> x dimensionslos", r.get('x') == '', str(r))

# ---------------------------------------------------------------------------
print("=== Funktions-Ausgaben (var = func(...)) und Ausdrücke mit Funktionen ===")

func_cases = [
    ('h = enthalpy(water, T=T_1, p=p)', 'h', 'J/kg'),
    ('s = entropy(water, T=T_1, p=p)', 's', 'J/(kg*K)'),
    ('T = temperature(water, p=p, h=h_1)', 'T', 'K'),
    ('p_s = pressure(water, T=T_1, x=0)', 'p_s', 'Pa'),
    ('rho = density(water, T=T_1, p=p)', 'rho', 'kg/m^3'),
    ('v = volume(water, T=T_1, p=p)', 'v', 'm^3/kg'),
    ('x_q = quality(water, h=h_1, p=p)', 'x_q', ''),
    ('mu = viscosity(water, T=T_1, p=p)', 'mu', 'Pa*s'),
    ('k = conductivity(water, T=T_1, p=p)', 'k', 'W/(m*K)'),
    ('Pr = prandtl(water, T=T_1, p=p)', 'Pr', ''),
    ('c = soundspeed(air, T=T_1, p=p)', 'c', 'm/s'),
    ('h_a = HumidAir(h, T=T_1, rh=rh_1, p_tot=p)', 'h_a', 'J/kg'),
    ('w_a = HumidAir(w, T=T_1, rh=rh_1, p_tot=p)', 'w_a', ''),
    ('T_dp = HumidAir(T_dp, T=T_1, rh=rh_1, p_tot=p)', 'T_dp', 'K'),
    ('T_wb = HumidAir(T_wb, T=T_1, rh=rh_1, p_tot=p)', 'T_wb', 'K'),
    ('rho_a = HumidAir(rho_tot, T=T_1, rh=rh_1, p_tot=p)', 'rho_a', 'kg/m^3'),
    ('p_w = HumidAir(p_w, T=T_1, rh=rh_1, p_tot=p)', 'p_w', 'Pa'),
    ('E = Eb(T_1, L)', 'E', 'W/m^3'),
    ('lam = Wien(T_1)', 'lam', 'm'),
    ('E_tot = Stefan_Boltzmann(T_1)', 'E_tot', 'W/m^2'),
    ('f = Blackbody(T_1, 0.38, 0.75)', 'f', ''),
    ('f_c = Blackbody_cumulative(T_1, L)', 'f_c', ''),
    ('Q_dot = m_dot*(enthalpy(water, T=T_1, p=p) - h_1)', 'Q_dot', 'W'),
    ('T_sat_C = temperature(water, p=p, x=1) - 273.15', 'T_sat_C', 'K'),
    ('v_2 = 1/density(water, T=T_1, p=p)', 'v_2', 'm^3/kg'),
]
fk = {'T_1': 'K', 'p': 'Pa', 'h_1': 'J/kg', 'rh_1': '', 'L': 'm', 'm_dot': 'kg/s'}
for eq, var, expected in func_cases:
    r = analyze_equation(eq, fk)
    check(f"{eq} -> {var}: {expected or 'dimensionslos'}", same_dim(r.get(var), expected), str(r.get(var)))
r = analyze_equation('E = Eb(T_1, L)', fk)
check("Eb-Label 'W/(m^2*um)'", r.get('E') == 'W/(m^2*um)', str(r))
r = analyze_equation('lam = Wien(T_1)', fk)
check("Wien-Label 'um'", r.get('lam') == 'um', str(r))
r = analyze_equation('T_dp = HumidAir(T_dp, T=T_1, rh=rh_1, p_tot=p)', fk)
check("HumidAir(T_dp) -> 'K' (absolut)", r.get('T_dp') == 'K', str(r))

# ---------------------------------------------------------------------------
print("=== Beispiele aus der Dokumentation: keine Fehlalarme ===")

steam = ["h_1 = enthalpy(water, p=p_1, T=T_1)", "s_1 = entropy(water, p=p_1, T=T_1)",
         "h_2s = enthalpy(water, p=p_2, s=s_1)", "eta_s_i_T = (h_2-h_1)/(h_2s-h_1)",
         "p_4 = pressure(water, x=x_4, T=T_4)", "h_4 = enthalpy(water, x=x_4, T=T_4)",
         "W_dot_T = m_dot_1*(h_1-h_2)", "Q_dot = m_dot_1*(h_1-h_4)", "eta_th = W_dot_T/Q_dot"]
steam_known = {'m_dot_1': 'kg/s', 'T_1': 'K', 'p_1': 'Pa', 'p_2': 'Pa', 'eta_s_i_T': '', 'x_4': '', 'T_4': 'K'}
r = propagate(steam, steam_known)
check("Dampfkraftprozess: Einheiten",
      same_dim(r.get('h_2'), 'J/kg') and same_dim(r.get('W_dot_T'), 'W') and r.get('eta_th') == ''
      and same_dim(r.get('p_4'), 'Pa'), str(r))
w = warnings_for(steam, steam_known)
check("Dampfkraftprozess: keine Unit-Warnungen", w == [], str([x.explanation for x in w]))

humid = ["h_1 = HumidAir(h, T=T_1, rh=rh_1, p_tot=p)", "w_1 = HumidAir(w, T=T_1, rh=rh_1, p_tot=p)",
         "T_dp_1 = HumidAir(T_dp, T=T_1, rh=rh_1, p_tot=p)", "h_2 = HumidAir(h, T=T_2, rh=rh_2, p_tot=p)",
         "w_2 = HumidAir(w, T=T_2, rh=rh_2, p_tot=p)", "Q_dot_cool = m_dot_a*(h_1-h_2)",
         "m_dot_condensate = m_dot_a*(w_1-w_2)"]
humid_known = {'T_1': 'K', 'rh_1': '', 'p': 'Pa', 'T_2': 'K', 'rh_2': '', 'm_dot_a': 'kg/s'}
r = propagate(humid, humid_known)
check("Feuchte Luft: Einheiten",
      same_dim(r.get('Q_dot_cool'), 'W') and same_dim(r.get('m_dot_condensate'), 'kg/s')
      and r.get('T_dp_1') == 'K' and r.get('w_1') == '', str(r))
w = warnings_for(humid, humid_known)
check("Feuchte Luft: keine Unit-Warnungen", w == [], str([x.explanation for x in w]))

rad = ["Q_rad = epsilon * sigma * A * T_surface^4", "lambda_max = Wien(T_surface)",
       "E_spectral = Eb(T_surface, L)", "f_visible = Blackbody(T_surface, 0.38e-6, 0.75e-6)"]
rad_known = {'T_surface': 'K', 'epsilon': '', 'A': 'm^2', 'sigma': 'W/(m^2*K^4)', 'L': 'm'}
r = propagate(rad, rad_known)
check("Strahlung: Einheiten",
      same_dim(r.get('Q_rad'), 'W') and same_dim(r.get('lambda_max'), 'm')
      and same_dim(r.get('E_spectral'), 'W/m^3') and r.get('f_visible') == '', str(r))
w = warnings_for(rad, rad_known)
check("Strahlung: keine Unit-Warnungen", w == [], str([x.explanation for x in w]))

w = check_all_unit_consistency({'x': 2.86, 'r': 3.0}, {'(x*log(x)) - (r)': 'x*ln(x) = r'}, {'r': ''})
check("Einheitenloses System -> keine Unit-Warnungen", w == [])

# ---------------------------------------------------------------------------
print()
print(f"{len(PASSED)}/{len(PASSED) + len(FAILED)} Tests bestanden")
if FAILED:
    print("FEHLGESCHLAGEN:")
    for f in FAILED:
        print("  -", f)
    sys.exit(1)
print("ALLE TESTS OK")
