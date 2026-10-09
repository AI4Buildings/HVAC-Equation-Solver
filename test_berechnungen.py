"""
Berechnungstests Thermodynamik / Wärmeübertragung (Oktober 2026).

Jede Aufgabe wird MIT Einheiten (°C, bar, kJ/kg, mm, kW, ...) und OHNE
Einheiten (reine SI-Zahlen) gerechnet und gegen UNABHÄNGIG berechnete
Referenzwerte geprüft (CoolProp direkt, analytische Formeln, numpy/scipy).
Der Lösungsweg entspricht EquationSolverApp.solve() (Parser -> Einheiten-
Startwerte -> solve_system/solve_parametric), nur ohne GUI.

Ausführen mit:  python3 test_berechnungen.py
"""
import math
import sys
import warnings

import numpy as np
import CoolProp.CoolProp as CP
from scipy.optimize import brentq

warnings.filterwarnings("ignore")

from parser import parse_equations, validate_system
from solver import solve_system, solve_parametric
from units import get_initial_from_unit
from unit_constraints import propagate_all_units_complete

K0 = 273.15
SIG = 5.67e-8


def pipeline(text):
    """Wie main.solve(): parse -> validate -> Einheiten-Startwerte -> solve."""
    eqs, variables, consts, sweeps, orig, uvals = parse_equations(text, parse_units=True)
    valid, vmsg = validate_system(eqs, variables, consts)
    if not valid:
        if not eqs and consts:
            return True, dict(consts), "Constants only"
        return False, {}, vmsg
    known_units = {v: uv.calc_unit for v, uv in uvals.items() if uv.calc_unit}
    all_units = propagate_all_units_complete(orig, known_units)
    init = {}
    for v in variables:
        u = all_units.get(v)
        if u is not None:
            init[v] = get_initial_from_unit(u)
    if sweeps:
        return solve_parametric(eqs, variables, sweeps, init, constants=dict(consts))
    return solve_system(eqs, variables, init, constants=dict(consts),
                        original_equations=orig)


RESULTS = []


def check(case, text, expected, rtol=1e-4, atol=1e-9):
    """expected: {var: Referenzwert in SI}"""
    try:
        ok, sol, msg = pipeline(text)
    except Exception as e:
        RESULTS.append((case, False, f"EXCEPTION {type(e).__name__}: {e}"))
        print(f"FAIL {case}: Exception {e}")
        return None
    bad = []
    for var, ref in expected.items():
        val = sol.get(var)
        if val is None:
            bad.append(f"{var}: fehlt")
            continue
        try:
            good = np.allclose(np.asarray(val, float), np.asarray(ref, float), rtol=rtol, atol=atol)
        except Exception:
            good = False
        if not good:
            bad.append(f"{var}={val!r} (Ref {ref!r})")
    passed = ok and not bad
    RESULTS.append((case, passed, msg if passed else f"{msg} | " + "; ".join(bad)))
    print(("OK  " if passed else "FAIL"), case, "" if passed else f"-> {msg} | " + "; ".join(bad))
    return sol


def summary():
    n_ok = sum(1 for _, p, _ in RESULTS if p)
    print(f"\n{n_ok}/{len(RESULTS)} Tests bestanden")
    failed = [(c, m) for c, p, m in RESULTS if not p]
    if failed:
        print("FEHLGESCHLAGEN:")
        for c, m in failed:
            print("  -", c, "::", m)
        sys.exit(1)
    print("ALLE TESTS OK")


# ===========================================================================
# A1  Luftverdichter, ideales Gas, isentroper Wirkungsgrad (implizit für T_2)
# ===========================================================================
T1, p1, p2, kap, cp, eta, m = 20 + K0, 1e5, 8e5, 1.4, 1005.0, 0.85, 0.5
T2s = T1 * (p2 / p1) ** ((kap - 1) / kap)
T2 = T1 + (T2s - T1) / eta
P = m * cp * (T2 - T1)
ref = {'T_2s': T2s, 'T_2': T2, 'P': P}
check("A1 Verdichter MIT Einheiten", """
p_1 = 1 bar
T_1 = 20 °C
p_2 = 8 bar
kappa = 1.4
c_p = 1.005 kJ/(kg*K)
eta_s = 0.85
m_dot = 1800 kg/h
T_2s = T_1*(p_2/p_1)^((kappa-1)/kappa)
eta_s = (T_2s - T_1)/(T_2 - T_1)
P = m_dot*c_p*(T_2 - T_1)
""", ref)
check("A1 Verdichter OHNE Einheiten", """
p_1 = 100000
T_1 = 293.15
p_2 = 800000
kappa = 1.4
c_p = 1005
eta_s = 0.85
m_dot = 0.5
T_2s = T_1*(p_2/p_1)^((kappa-1)/kappa)
eta_s = (T_2s - T_1)/(T_2 - T_1)
P = m_dot*c_p*(T_2 - T_1)
""", ref)

# ===========================================================================
# A2  Ideales Gas: Masse in einem Behälter (implizite Form p*v = R*T)
# ===========================================================================
R, T, p, V = 287.0, 50 + K0, 6e5, 0.2
v = R * T / p
ref = {'v': v, 'm': V / v}
check("A2 Ideales Gas MIT Einheiten", """
R = 287 J/(kg*K)
T = 50 °C
p = 6 bar
V = 200 L
p*v = R*T
m = V/v
""", ref)
check("A2 Ideales Gas OHNE Einheiten", """
R = 287
T = 323.15
p = 600000
V = 0.2
p*v = R*T
m = V/v
""", ref)

# ===========================================================================
# A3  Clausius-Rankine-Prozess (Wasser), Turbine + Pumpe mit Wirkungsgrad
# ===========================================================================
f = 'Water'
T1, p1, p2, eT, eP, m = 450 + K0, 30e5, 0.1e5, 0.85, 0.75, 10000 / 3600
h1 = CP.PropsSI('H', 'T', T1, 'P', p1, f); s1 = CP.PropsSI('S', 'T', T1, 'P', p1, f)
h2s = CP.PropsSI('H', 'P', p2, 'S', s1, f); h2 = h1 - eT * (h1 - h2s)
x2 = CP.PropsSI('Q', 'P', p2, 'H', h2, f)
h3 = CP.PropsSI('H', 'P', p2, 'Q', 0, f); s3 = CP.PropsSI('S', 'P', p2, 'Q', 0, f)
h4s = CP.PropsSI('H', 'P', p1, 'S', s3, f); h4 = h3 + (h4s - h3) / eP
WT, WP, Q = m * (h1 - h2), m * (h4 - h3), m * (h1 - h4)
ref = {'h_1': h1, 'h_2': h2, 'x_2': x2, 'h_4': h4, 'W_dot_T': WT, 'W_dot_P': WP,
       'Q_dot_zu': Q, 'eta_th': (WT - WP) / Q}
rankine = """
m_dot = {m}
T_1 = {T1}
p_1 = {p1}
p_2 = {p2}
eta_T = 0.85
eta_P = 0.75
h_1 = enthalpy(water, T=T_1, p=p_1)
s_1 = entropy(water, T=T_1, p=p_1)
h_2s = enthalpy(water, p=p_2, s=s_1)
eta_T = (h_1 - h_2)/(h_1 - h_2s)
x_2 = quality(water, p=p_2, h=h_2)
h_3 = enthalpy(water, p=p_2, x=0)
s_3 = entropy(water, p=p_2, x=0)
h_4s = enthalpy(water, p=p_1, s=s_3)
eta_P = (h_4s - h_3)/(h_4 - h_3)
W_dot_T = m_dot*(h_1 - h_2)
W_dot_P = m_dot*(h_4 - h_3)
Q_dot_zu = m_dot*(h_1 - h_4)
eta_th = (W_dot_T - W_dot_P)/Q_dot_zu
"""
check("A3 Rankine MIT Einheiten", rankine.format(m="10000/3600 kg/s", T1="450 °C", p1="30 bar", p2="0.1 bar"), ref)
check("A3 Rankine OHNE Einheiten", rankine.format(m="10000/3600", T1="723.15", p1="3000000", p2="10000"), ref)

# ===========================================================================
# A4  Kaltdampf-Kompressionskälteprozess R134a (Massenstrom implizit aus Q_0)
# ===========================================================================
f = 'R134a'
T0, Tc, dTsh, eis, Q0 = -10 + K0, 40 + K0, 5.0, 0.7, 10000.0
p0 = CP.PropsSI('P', 'T', T0, 'Q', 1, f); pc = CP.PropsSI('P', 'T', Tc, 'Q', 0, f)
T1 = T0 + dTsh
h1 = CP.PropsSI('H', 'T', T1, 'P', p0, f); s1 = CP.PropsSI('S', 'T', T1, 'P', p0, f)
h2s = CP.PropsSI('H', 'P', pc, 'S', s1, f); h2 = h1 + (h2s - h1) / eis
T2 = CP.PropsSI('T', 'P', pc, 'H', h2, f)
h3 = CP.PropsSI('H', 'T', Tc, 'Q', 0, f)
mR = Q0 / (h1 - h3); Pel = mR * (h2 - h1)
ref = {'p_0': p0, 'p_c': pc, 'T_2': T2, 'm_dot': mR, 'P': Pel, 'COP': Q0 / Pel}
kaelte = """
T_0 = {T0}
T_c = {Tc}
dT_sh = {dT}
eta_is = 0.7
Q_dot_0 = {Q0}
p_0 = pressure(R134a, T=T_0, x=1)
p_c = pressure(R134a, T=T_c, x=0)
T_1 = T_0 + dT_sh
h_1 = enthalpy(R134a, T=T_1, p=p_0)
s_1 = entropy(R134a, T=T_1, p=p_0)
h_2s = enthalpy(R134a, p=p_c, s=s_1)
eta_is = (h_2s - h_1)/(h_2 - h_1)
T_2 = temperature(R134a, p=p_c, h=h_2)
h_3 = enthalpy(R134a, T=T_c, x=0)
h_4 = h_3
Q_dot_0 = m_dot*(h_1 - h_4)
P = m_dot*(h_2 - h_1)
COP = Q_dot_0/P
"""
check("A4 Kälteprozess MIT Einheiten", kaelte.format(T0="-10 °C", Tc="40 °C", dT="5 K", Q0="10 kW"), ref)
check("A4 Kälteprozess OHNE Einheiten", kaelte.format(T0="263.15", Tc="313.15", dT="5", Q0="10000"), ref)

# ===========================================================================
# B1  Mehrschichtige Außenwand: U-Wert, Wärmestrom, Oberflächentemperatur
# ===========================================================================
Rsi, Rse, d1, l1, d2, l2, d3, l3 = 0.13, 0.04, 0.20, 2.3, 0.16, 0.035, 0.015, 0.7
Ti, Te, A = 20 + K0, -10 + K0, 25.0
U = 1 / (Rsi + d1 / l1 + d2 / l2 + d3 / l3 + Rse)
q = U * (Ti - Te)
ref = {'U': U, 'q': q, 'Q_dot': q * A, 'T_si': Ti - q * Rsi}
wand = """
R_si = {Rsi}
R_se = {Rse}
d_1 = {d1}
lambda_1 = {l1}
d_2 = {d2}
lambda_2 = {l2}
d_3 = {d3}
lambda_3 = {l3}
T_i = {Ti}
T_e = {Te}
A = {A}
U = 1/(R_si + d_1/lambda_1 + d_2/lambda_2 + d_3/lambda_3 + R_se)
q = U*(T_i - T_e)
Q_dot = q*A
T_si = T_i - q*R_si
"""
check("B1 Wand MIT Einheiten (cm, W/mK, m^2*K/W)", wand.format(
    Rsi="0.13 m^2*K/W", Rse="0.04 m^2*K/W", d1="20 cm", l1="2.3 W/mK", d2="16 cm",
    l2="0.035 W/mK", d3="15 mm", l3="0.7 W/mK", Ti="20 °C", Te="-10 °C", A="25 m^2"), ref)
check("B1 Wand MIT Einheiten (m²K/W wie in CLAUDE.md)", wand.format(
    Rsi="0.13 m²K/W", Rse="0.04 m²K/W", d1="20 cm", l1="2.3 W/mK", d2="16 cm",
    l2="0.035 W/mK", d3="15 mm", l3="0.7 W/mK", Ti="20 °C", Te="-10 °C", A="25 m^2"), ref)
check("B1 Wand OHNE Einheiten", wand.format(
    Rsi="0.13", Rse="0.04", d1="0.20", l1="2.3", d2="0.16", l2="0.035", d3="0.015",
    l3="0.7", Ti="293.15", Te="263.15", A="25"), ref)

# B1b  implizit: welche Dämmdicke für U = 0.2 W/m²K?
d2_req = (1 / 0.2 - Rsi - d1 / l1 - d3 / l3 - Rse) * l2
ref = {'d_2': d2_req}
wand_impl = """
R_si = {Rsi}
R_se = {Rse}
d_1 = {d1}
lambda_1 = 2.3
lambda_2 = 0.035
d_3 = {d3}
lambda_3 = 0.7
U = {U}
U = 1/(R_si + d_1/lambda_1 + d_2/lambda_2 + d_3/lambda_3 + R_se)
"""
check("B1b Dämmdicke implizit MIT Einheiten", wand_impl.format(
    Rsi="0.13 m^2*K/W", Rse="0.04 m^2*K/W", d1="20 cm", d3="15 mm", U="0.2 W/m^2K"), ref)
check("B1b Dämmdicke implizit OHNE Einheiten", wand_impl.format(
    Rsi="0.13", Rse="0.04", d1="0.2", d3="0.015", U="0.2"), ref)

# ===========================================================================
# B2  Gedämmtes Rohr (Zylinderwand, Reihenschaltung von Widerständen)
# ===========================================================================
r1, r2, s_iso, lS, lI, hi, ha, Ti, Ta, L = 0.025, 0.030, 0.040, 50.0, 0.04, 1000.0, 10.0, 90 + K0, 20 + K0, 10.0
r3 = r2 + s_iso
Rp = 1 / (hi * 2 * math.pi * r1) + math.log(r2 / r1) / (2 * math.pi * lS) \
    + math.log(r3 / r2) / (2 * math.pi * lI) + 1 / (ha * 2 * math.pi * r3)
Qr = L * (Ti - Ta) / Rp
ref = {'r_3': r3, 'Q_dot': Qr, 'T_o': Ta + Qr / (ha * 2 * math.pi * r3 * L)}
rohr = """
r_1 = {r1}
r_2 = {r2}
s_iso = {s}
lambda_St = {lS}
lambda_iso = {lI}
h_i = {hi}
h_a = {ha}
T_i = {Ti}
T_a = {Ta}
L = {L}
r_3 = r_2 + s_iso
R_L = 1/(h_i*2*pi*r_1) + ln(r_2/r_1)/(2*pi*lambda_St) + ln(r_3/r_2)/(2*pi*lambda_iso) + 1/(h_a*2*pi*r_3)
Q_dot = L*(T_i - T_a)/R_L
T_o = T_a + Q_dot/(h_a*2*pi*r_3*L)
"""
check("B2 Rohrdämmung MIT Einheiten", rohr.format(r1="25 mm", r2="30 mm", s="40 mm", lS="50 W/mK",
      lI="0.04 W/mK", hi="1000 W/m^2K", ha="10 W/m^2K", Ti="90 °C", Ta="20 °C", L="10 m"), ref)
check("B2 Rohrdämmung OHNE Einheiten", rohr.format(r1="0.025", r2="0.030", s="0.040", lS="50",
      lI="0.04", hi="1000", ha="10", Ti="363.15", Ta="293.15", L="10"), ref)

# ===========================================================================
# B3  Gegenstrom-Wärmeübertrager, LMTD-Methode (gekoppelter Block)
#     Referenz: epsilon-NTU (analytisch)
# ===========================================================================
mh, cph, Thi, mc, cpc, Tci, Uw, Aw = 2.0, 4190.0, 90 + K0, 3.0, 4180.0, 15 + K0, 1200.0, 10.0
Ch, Cc = mh * cph, mc * cpc
Cmin, Cmax = min(Ch, Cc), max(Ch, Cc)
Cr, NTU = Cmin / Cmax, Uw * Aw / Cmin
epsi = (1 - math.exp(-NTU * (1 - Cr))) / (1 - Cr * math.exp(-NTU * (1 - Cr)))
Qw = epsi * Cmin * (Thi - Tci)
ref = {'Q_dot': Qw, 'T_h_out': Thi - Qw / Ch, 'T_c_out': Tci + Qw / Cc}
wt = """
m_dot_h = {mh}
c_p_h = {cph}
T_h_in = {Thi}
m_dot_c = {mc}
c_p_c = {cpc}
T_c_in = {Tci}
U = {U}
A = {A}
Q_dot = m_dot_h*c_p_h*(T_h_in - T_h_out)
Q_dot = m_dot_c*c_p_c*(T_c_out - T_c_in)
dT_1 = T_h_in - T_c_out
dT_2 = T_h_out - T_c_in
dT_m = (dT_1 - dT_2)/ln(dT_1/dT_2)
Q_dot = U*A*dT_m
"""
check("B3 Gegenstrom-WT MIT Einheiten", wt.format(mh="2 kg/s", cph="4.19 kJ/(kg*K)", Thi="90 °C",
      mc="10800 kg/h", cpc="4.18 kJ/kgK", Tci="15 °C", U="1.2 kW/(m^2*K)", A="10 m^2"), ref)
check("B3 Gegenstrom-WT MIT Einheiten (U in W/m^2K)", wt.format(mh="2 kg/s", cph="4.19 kJ/(kg*K)",
      Thi="90 °C", mc="10800 kg/h", cpc="4.18 kJ/kgK", Tci="15 °C", U="1200 W/m^2K", A="10 m^2"), ref)
check("B3 Gegenstrom-WT OHNE Einheiten", wt.format(mh="2", cph="4190", Thi="363.15",
      mc="3", cpc="4180", Tci="288.15", U="1200", A="10"), ref)

# ===========================================================================
# B4  Strahlung: (a) Austausch zwischen parallelen Platten,
#                (b) Energiebilanz Dachoberfläche (Solar + Konvektion + Strahlung, T^4 implizit)
# ===========================================================================
T1, T2, e1, e2 = 500 + K0, 100 + K0, 0.8, 0.6
q12 = SIG * (T1 ** 4 - T2 ** 4) / (1 / e1 + 1 / e2 - 1)
aS, G, eps, h, Tinf, Tsky = 0.6, 800.0, 0.9, 15.0, 25 + K0, 5 + K0
Ts = brentq(lambda x: aS * G - h * (x - Tinf) - eps * SIG * (x ** 4 - Tsky ** 4), 200, 500)
ref = {'q_12': q12, 'T_s': Ts}
strahlung = """
sigma = {sig}
T_1 = {T1}
T_2 = {T2}
eps_1 = 0.8
eps_2 = 0.6
q_12 = sigma*(T_1^4 - T_2^4)/(1/eps_1 + 1/eps_2 - 1)
alpha_s = 0.6
G_s = {G}
eps = 0.9
h = {h}
T_inf = {Tinf}
T_sky = {Tsky}
alpha_s*G_s = h*(T_s - T_inf) + eps*sigma*(T_s^4 - T_sky^4)
"""
check("B4 Strahlung MIT Einheiten", strahlung.format(sig="5.67e-8 W/m^2K^4", T1="500 °C", T2="100 °C",
      G="800 W/m^2", h="15 W/m^2K", Tinf="25 °C", Tsky="5 °C"), ref)
check("B4 Strahlung OHNE Einheiten", strahlung.format(sig="5.67e-8", T1="773.15", T2="373.15",
      G="800", h="15", Tinf="298.15", Tsky="278.15"), ref)

# ===========================================================================
# B5  Stabrippe (Pin-Fin) mit korrigierter Länge
# ===========================================================================
D, L, k, h, Tb, Ti = 0.005, 0.05, 200.0, 25.0, 80 + K0, 20 + K0
P, Ac = math.pi * D, math.pi * D ** 2 / 4
mf = math.sqrt(h * P / (k * Ac)); Lc = L + D / 4
etaf = math.tanh(mf * Lc) / (mf * Lc)
Qf = etaf * h * P * Lc * (Tb - Ti)
ref = {'m_f': mf, 'eta_f': etaf, 'Q_dot_f': Qf}
rippe = """
D = {D}
L = {L}
k = {k}
h = {h}
T_b = {Tb}
T_inf = {Ti}
P = pi*D
A_c = pi*D^2/4
m_f = sqrt(h*P/(k*A_c))
L_c = L + D/4
eta_f = tanh(m_f*L_c)/(m_f*L_c)
A_f = P*L_c
Q_dot_f = eta_f*h*A_f*(T_b - T_inf)
"""
check("B5 Rippe MIT Einheiten", rippe.format(D="5 mm", L="50 mm", k="200 W/mK", h="25 W/m^2K",
      Tb="80 °C", Ti="20 °C"), ref)
check("B5 Rippe OHNE Einheiten", rippe.format(D="0.005", L="0.05", k="200", h="25",
      Tb="353.15", Ti="293.15"), ref)

# ===========================================================================
# B6  Erzwungene Konvektion im Rohr (Dittus-Boelter) mit CoolProp-Stoffwerten
# ===========================================================================
Tm, p, d, u = 60 + K0, 3e5, 0.025, 1.5
rho = CP.PropsSI('D', 'T', Tm, 'P', p, 'Water'); mu = CP.PropsSI('V', 'T', Tm, 'P', p, 'Water')
lam = CP.PropsSI('L', 'T', Tm, 'P', p, 'Water'); Pr = CP.PropsSI('Prandtl', 'T', Tm, 'P', p, 'Water')
Re = rho * u * d / mu; Nu = 0.023 * Re ** 0.8 * Pr ** 0.4
ref = {'Re': Re, 'Nu': Nu, 'alpha': Nu * lam / d}
konv = """
T_m = {Tm}
p = {p}
d = {d}
u = {u}
rho = density(water, T=T_m, p=p)
mu = viscosity(water, T=T_m, p=p)
lambda = conductivity(water, T=T_m, p=p)
Pr = prandtl(water, T=T_m, p=p)
Re = rho*u*d/mu
Nu = 0.023*Re^0.8*Pr^0.4
alpha = Nu*lambda/d
"""
check("B6 Rohrströmung MIT Einheiten", konv.format(Tm="60 °C", p="3 bar", d="25 mm", u="1.5 m/s"), ref)
check("B6 Rohrströmung OHNE Einheiten", konv.format(Tm="333.15", p="300000", d="0.025", u="1.5"), ref)

# ===========================================================================
# B7  Instationäre Abkühlung (Blockkapazität), Zeit implizit
# ===========================================================================
D, rho, c, k, h, Ti, Tinf, Te = 0.02, 7800.0, 460.0, 45.0, 100.0, 300 + K0, 20 + K0, 50 + K0
V, A = math.pi * D ** 3 / 6, math.pi * D ** 2
t = -rho * V * c / (h * A) * math.log((Te - Tinf) / (Ti - Tinf))
ref = {'Bi': h * (V / A) / k, 't': t}
inst = """
D = {D}
rho = 7800
c = {c}
k = 45
h = 100
T_i = {Ti}
T_inf = {Tinf}
T_end = {Te}
V = pi*D^3/6
A = pi*D^2
Bi = h*(V/A)/k
(T_end - T_inf)/(T_i - T_inf) = exp(-h*A*t/(rho*V*c))
"""
check("B7 Abkühlung MIT Einheiten", inst.format(D="20 mm", c="0.46 kJ/(kg*K)", Ti="300 °C",
      Tinf="20 °C", Te="50 °C"), ref)
check("B7 Abkühlung OHNE Einheiten", inst.format(D="0.02", c="460", Ti="573.15",
      Tinf="293.15", Te="323.15"), ref)

# ===========================================================================
# C1  Feuchte Luft: Mischung Außenluft/Umluft (HumidAir, h-x)
# ===========================================================================
p = 101325.0
T1, r1, m1, T2, r2, m2 = -5 + K0, 0.8, 1.0, 22 + K0, 0.4, 2.0
H = lambda o, *a: CP.HAPropsSI(o, *a)
h1 = H('Hda', 'T', T1, 'R', r1, 'P', p); w1 = H('W', 'T', T1, 'R', r1, 'P', p)
h2 = H('Hda', 'T', T2, 'R', r2, 'P', p); w2 = H('W', 'T', T2, 'R', r2, 'P', p)
h3 = (m1 * h1 + m2 * h2) / (m1 + m2); w3 = (m1 * w1 + m2 * w2) / (m1 + m2)
T3 = H('T', 'Hda', h3, 'W', w3, 'P', p); r3 = H('R', 'Hda', h3, 'W', w3, 'P', p)
ref = {'h_3': h3, 'w_3': w3, 'T_3': T3, 'rh_3': r3}
mix = """
p = {p}
T_1 = {T1}
rh_1 = 0.8
m_dot_1 = {m1}
T_2 = {T2}
rh_2 = 0.4
m_dot_2 = {m2}
h_1 = HumidAir(h, T=T_1, rh=rh_1, p_tot=p)
w_1 = HumidAir(w, T=T_1, rh=rh_1, p_tot=p)
h_2 = HumidAir(h, T=T_2, rh=rh_2, p_tot=p)
w_2 = HumidAir(w, T=T_2, rh=rh_2, p_tot=p)
m_dot_3 = m_dot_1 + m_dot_2
m_dot_3*h_3 = m_dot_1*h_1 + m_dot_2*h_2
m_dot_3*w_3 = m_dot_1*w_1 + m_dot_2*w_2
T_3 = HumidAir(T, h=h_3, w=w_3, p_tot=p)
rh_3 = HumidAir(rh, h=h_3, w=w_3, p_tot=p)
"""
check("C1 Luftmischung MIT Einheiten", mix.format(p="1.01325 bar", T1="-5 °C", m1="3600 kg/h",
      T2="22 °C", m2="2 kg/s"), ref)
check("C1 Luftmischung OHNE Einheiten", mix.format(p="101325", T1="268.15", m1="1",
      T2="295.15", m2="2"), ref)

# ===========================================================================
# C2  Kühlregister mit Entfeuchtung + implizite rel. Feuchte aus h
# ===========================================================================
p, T1, r1, T2, r2, ma = 1e5, 30 + K0, 0.5, 14 + K0, 0.95, 1.2
h1 = H('Hda', 'T', T1, 'R', r1, 'P', p); w1 = H('W', 'T', T1, 'R', r1, 'P', p)
h2 = H('Hda', 'T', T2, 'R', r2, 'P', p); w2 = H('W', 'T', T2, 'R', r2, 'P', p)
Tdp1 = H('Tdp', 'T', T1, 'R', r1, 'P', p); Twb1 = H('Twb', 'T', T1, 'R', r1, 'P', p)
rx = H('R', 'T', 25 + K0, 'Hda', 50000.0, 'P', p)
ref = {'Q_dot_K': ma * (h1 - h2), 'm_dot_w': ma * (w1 - w2), 'T_dp_1': Tdp1, 'T_wb_1': Twb1, 'rh_x': rx}
kuehl = """
p = {p}
T_1 = {T1}
rh_1 = 0.5
T_2 = {T2}
rh_2 = 0.95
m_dot_a = {ma}
h_1 = HumidAir(h, T=T_1, rh=rh_1, p_tot=p)
w_1 = HumidAir(w, T=T_1, rh=rh_1, p_tot=p)
T_dp_1 = HumidAir(T_dp, T=T_1, rh=rh_1, p_tot=p)
T_wb_1 = HumidAir(T_wb, T=T_1, rh=rh_1, p_tot=p)
h_2 = HumidAir(h, T=T_2, rh=rh_2, p_tot=p)
w_2 = HumidAir(w, T=T_2, rh=rh_2, p_tot=p)
Q_dot_K = m_dot_a*(h_1 - h_2)
m_dot_w = m_dot_a*(w_1 - w_2)
T_x = {Tx}
h_x = {hx}
h_x = HumidAir(h, T=T_x, rh=rh_x, p_tot=p)
"""
check("C2 Kühlregister MIT Einheiten", kuehl.format(p="1 bar", T1="30 °C", T2="14 °C",
      ma="1.2 kg/s", Tx="25 °C", hx="50 kJ/kg"), ref)
check("C2 Kühlregister OHNE Einheiten", kuehl.format(p="100000", T1="303.15", T2="287.15",
      ma="1.2", Tx="298.15", hx="50000"), ref)

# ===========================================================================
# D   Parameterstudien
# ===========================================================================
Ts_ = np.arange(20, 101, 20) + K0
ref = {'p_s': np.array([CP.PropsSI('P', 'T', x, 'Q', 0, 'Water') for x in Ts_])}
check("D1 Sweep Sättigungsdruck MIT Einheiten", "T = 20:20:100 °C\np_s = pressure(water, T=T, x=0)", ref)
check("D1 Sweep Sättigungsdruck OHNE Einheiten", "T = 293.15:20:373.15\np_s = pressure(water, T=T, x=0)", ref)

Gs = np.arange(0, 1001, 250.0)
ref = {'T_s': np.array([brentq(lambda x: 0.6 * g - 15 * (x - (25 + K0)) - 0.9 * SIG * (x ** 4 - (5 + K0) ** 4), 200, 500)
                        for g in Gs])}
sweep_rad = """
G_s = {G}
sigma = 5.67e-8
h = {h}
T_inf = {Ti}
T_sky = {Ts}
0.6*G_s = h*(T_s - T_inf) + 0.9*sigma*(T_s^4 - T_sky^4)
"""
check("D2 Sweep Strahlungsbilanz implizit MIT Einheiten", sweep_rad.format(G="0:250:1000 W/m^2", h="15 W/m^2K",
      Ti="25 °C", Ts="5 °C"), ref)
check("D2 Sweep Strahlungsbilanz implizit OHNE Einheiten", sweep_rad.format(G="0:250:1000", h="15",
      Ti="298.15", Ts="278.15"), ref)

# ===========================================================================
# E   Einheiten-Einzeltests (Umrechnung nach SI)
# ===========================================================================
check("E1 Fläche in cm^2", "A = 50 cm^2\nh = 10 W/m^2K\ndT = 20 K\nQ = h*A*dT", {'Q': 1.0})
check("E2 Wärmestromdichte in kW/m^2", "q = 2 kW/m^2\nA = 3 m^2\nQ = q*A", {'Q': 6000.0})
check("E3 Schichtdicke in µm (Beschichtung)", "s = 50 µm\nk = 0.2 W/mK\nR = s/k", {'R': 2.5e-4})
check("E4 Volumen in L", "V = 200 L\nrho = 1000 kg/m^3\nm = rho*V", {'m': 200.0})
check("E5 Viskosität in mPa*s", "mu = 1 mPa*s\nrho = 1000 kg/m^3\nnu = mu/rho", {'nu': 1e-6})
check("E6 Temperaturleitfähigkeit mm^2/s", "a = 0.15 mm^2/s\nL = 0.1 m\nt = L^2/a", {'t': 0.01 / 0.15e-6})
check("E7 U-Wert in kW/(m^2*K)", "U = 0.025 kW/(m^2*K)\nA = 10 m^2\ndT = 10 K\nQ = U*A*dT", {'Q': 2500.0})
check("E8 Energie kWh / Zeit h", "E = 5 kWh\nt = 2 h\nP = E/t", {'P': 2500.0})
check("E9 Fahrenheit-Temperatur", "T = 68 °F\nx = T", {'x': 293.15}, rtol=1e-6)
check("E10 Druck in atm und kPa", "p_1 = 1 atm\np_2 = 250 kPa\ndp = p_2 - p_1", {'dp': 250000 - 101325})
check("E11 Volumenstrom m^3/h", "V_dot = 500 m^3/h\nrho = 1.2 kg/m^3\nm_dot = rho*V_dot", {'m_dot': 500 / 3600 * 1.2})
check("E12 dT in °C (Differenz)", "dT_1 = 10 °C\nm = 1 kg/s\nc = 4.19 kJ/(kg*K)\nQ = m*c*dT_1", {'Q': 41900.0})
check("E13 Mitteltemperatur aus °C", "T_1 = 10 °C\nT_2 = 30 °C\nT_m = (T_1 + T_2)/2", {'T_m': 293.15})


# ---------------------------------------------------------------------------
# F1  2D-stationäre Wärmeleitung, FD-Gitter 3x3 Innenknoten (linearer 9er-Block)
#     Rand: oben 100 °C, sonst 0 °C... hier: oben T_o, links/rechts/unten T_u
# ---------------------------------------------------------------------------
To, Tu = 100 + K0, 20 + K0
N = 3
idx = lambda i, j: i * N + j
Am = np.zeros((9, 9)); b = np.zeros(9)
for i in range(N):
    for j in range(N):
        r = idx(i, j); Am[r, r] = -4
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ii, jj = i + di, j + dj
            if 0 <= ii < N and 0 <= jj < N:
                Am[r, idx(ii, jj)] = 1
            else:
                b[r] -= To if ii < 0 else Tu
Tsol = np.linalg.solve(Am, b)
ref = {f"T_{i+1}{j+1}": Tsol[idx(i, j)] for i in range(N) for j in range(N)}
lines = []
for i in range(N):
    for j in range(N):
        nb = []
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ii, jj = i + di, j + dj
            if 0 <= ii < N and 0 <= jj < N:
                nb.append(f"T_{ii+1}{jj+1}")
            else:
                nb.append("T_o" if ii < 0 else "T_u")
        lines.append(f"4*T_{i+1}{j+1} = " + " + ".join(nb))
fd = "\n".join(lines)
check("F1 FD-Wärmeleitung 9er-Block MIT Einheiten", f"T_o = 100 °C\nT_u = 20 °C\n{fd}", ref)
check("F1 FD-Wärmeleitung 9er-Block OHNE Einheiten", f"T_o = 373.15\nT_u = 293.15\n{fd}", ref)

# ---------------------------------------------------------------------------
# F2  Strahlungsaustausch Dreiflächen-Hohlraum (Radiositäten, gekoppelt mit T^4)
#     Flächen 1, 2 grau mit vorgegebener Temperatur, Fläche 3 rückstrahlend (q_3 = 0)
# ---------------------------------------------------------------------------
A1, A2, A3 = 1.0, 1.0, 2.0
F12, F13, F21, F23 = 0.2, 0.8, 0.2, 0.8
F31, F32 = A1 * F13 / A3, A2 * F23 / A3
T1, T2, e1, e2 = 800 + K0, 300.0, 0.7, 0.5
Eb1, Eb2 = SIG * T1 ** 4, SIG * T2 ** 4
R1, R2 = (1 - e1) / (e1 * A1), (1 - e2) / (e2 * A2)
M = np.array([[1 / R1 + A1 * F12 + A1 * F13, -A1 * F12, -A1 * F13],
              [-A2 * F21, 1 / R2 + A2 * F21 + A2 * F23, -A2 * F23],
              [-A3 * F31, -A3 * F32, A3 * F31 + A3 * F32]])
J = np.linalg.solve(M, [Eb1 / R1, Eb2 / R2, 0.0])  # lineares System in J1..J3
Q1 = (Eb1 - J[0]) / R1
T3 = (J[2] / SIG) ** 0.25
ref = {'Q_1': Q1, 'T_3': T3}
hohl = """
sigma = 5.67e-8
A_1 = 1
A_2 = 1
A_3 = 2
F_12 = 0.2
F_13 = 0.8
F_21 = 0.2
F_23 = 0.8
F_31 = A_1*F_13/A_3
F_32 = A_2*F_23/A_3
T_1 = {T1}
T_2 = {T2}
eps_1 = 0.7
eps_2 = 0.5
Q_1 = (sigma*T_1^4 - J_1)/((1 - eps_1)/(eps_1*A_1))
Q_1 = A_1*F_12*(J_1 - J_2) + A_1*F_13*(J_1 - J_3)
(sigma*T_2^4 - J_2)/((1 - eps_2)/(eps_2*A_2)) = A_2*F_21*(J_2 - J_1) + A_2*F_23*(J_2 - J_3)
0 = A_3*F_31*(J_3 - J_1) + A_3*F_32*(J_3 - J_2)
J_3 = sigma*T_3^4
"""
check("F2 Strahlungs-Hohlraum MIT Einheiten", hohl.format(T1="800 °C", T2="300 K"), ref, rtol=1e-4)
check("F2 Strahlungs-Hohlraum OHNE Einheiten", hohl.format(T1="1073.15", T2="300"), ref, rtol=1e-4)

# ---------------------------------------------------------------------------
# F3  Implizite Parameterstudie mit CoolProp: T aus h und p (Sweep über p)
# ---------------------------------------------------------------------------
check("F3 Sweep implizit CoolProp MIT Einheiten",
      "p = 1:1:8 bar\nh = 2800 kJ/kg\nh = enthalpy(water, T=T, p=p)",
      {'T': np.array([CP.PropsSI('T', 'P', p * 1e5, 'H', 2.8e6, 'Water') for p in range(1, 9)])})
check("F3 Sweep implizit CoolProp OHNE Einheiten",
      "p = 100000:100000:800000\nh = 2800000\nh = enthalpy(water, T=T, p=p)",
      {'T': np.array([CP.PropsSI('T', 'P', p * 1e5, 'H', 2.8e6, 'Water') for p in range(1, 9)])})

# ---------------------------------------------------------------------------
# F4  Abkühlkurve als Sweep über die Zeit (vektorisiert)
# ---------------------------------------------------------------------------
t = np.arange(0, 601, 120.0)
tau = 7800 * 460 * 0.02 / 6 / 100
ref = {'T': 20 + K0 + 280 * np.exp(-t / tau)}
check("F4 Sweep Abkühlkurve MIT Einheiten", """t = 0:120:600 s
tau = rho*c*D/(6*h)
rho = 7800 kg/m^3
c = 460 J/(kg*K)
D = 20 mm
h = 100 W/m^2K
T_i = 300 °C
T_inf = 20 °C
T = T_inf + (T_i - T_inf)*exp(-t/tau)""", ref)
check("F4 Sweep Abkühlkurve OHNE Einheiten", """t = 0:120:600
tau = rho*c*D/(6*h)
rho = 7800
c = 460
D = 0.02
h = 100
T_i = 573.15
T_inf = 293.15
T = T_inf + (T_i - T_inf)*exp(-t/tau)""", ref)

# ---------------------------------------------------------------------------
# F5  Wärmepumpe Propan (R290): Verdampfer/Verflüssiger aus Quellen-/Senkentemperatur
#     mit Grädigkeiten, implizite Verdichterendtemperatur
# ---------------------------------------------------------------------------
f = 'R290'
T0, Tc = 0 + K0 - 5, 35 + K0 + 5
p0 = CP.PropsSI('P', 'T', T0, 'Q', 1, f); pc = CP.PropsSI('P', 'T', Tc, 'Q', 1, f)
h1 = CP.PropsSI('H', 'T', T0, 'Q', 1, f); s1 = CP.PropsSI('S', 'T', T0, 'Q', 1, f)
h2s = CP.PropsSI('H', 'P', pc, 'S', s1, f); h2 = h1 + (h2s - h1) / 0.65
h3 = CP.PropsSI('H', 'P', pc, 'Q', 0, f)
COP = (h2 - h3) / (h2 - h1)
ref = {'COP_H': COP, 'T_2': CP.PropsSI('T', 'P', pc, 'H', h2, f)}
wp = """
T_Q = {TQ}
T_S = {TS}
dT_V = {dTV}
dT_K = {dTK}
T_0 = T_Q - dT_V
T_c = T_S + dT_K
p_0 = pressure(propane, T=T_0, x=1)
p_c = pressure(propane, T=T_c, x=1)
h_1 = enthalpy(propane, T=T_0, x=1)
s_1 = entropy(propane, T=T_0, x=1)
h_2s = enthalpy(propane, p=p_c, s=s_1)
0.65 = (h_2s - h_1)/(h_2 - h_1)
T_2 = temperature(propane, p=p_c, h=h_2)
h_3 = enthalpy(propane, p=p_c, x=0)
COP_H = (h_2 - h_3)/(h_2 - h_1)
"""
check("F5 Wärmepumpe R290 MIT Einheiten", wp.format(TQ="0 °C", TS="35 °C", dTV="5 K", dTK="5 K"), ref)
check("F5 Wärmepumpe R290 OHNE Einheiten", wp.format(TQ="273.15", TS="308.15", dTV="5", dTK="5"), ref)

# ---------------------------------------------------------------------------
# F6  Natürliche Konvektion senkrechte Platte (Churchill-Chu), Stoffwerte Luft bei T_f
#     mit implizit bestimmter Oberflächentemperatur (Wärmestrom vorgegeben)
# ---------------------------------------------------------------------------
Tinf, H, q, p = 20 + K0, 0.5, 100.0, 1e5
def ts_res(Ts):
    Tf = 0.5 * (Ts + Tinf)
    nu = CP.PropsSI('V', 'T', Tf, 'P', p, 'Air') / CP.PropsSI('D', 'T', Tf, 'P', p, 'Air')
    k = CP.PropsSI('L', 'T', Tf, 'P', p, 'Air'); Pr = CP.PropsSI('Prandtl', 'T', Tf, 'P', p, 'Air')
    Ra = 9.81 * (1 / Tf) * (Ts - Tinf) * H ** 3 / nu ** 2 * Pr
    Nu = (0.825 + 0.387 * Ra ** (1 / 6) / (1 + (0.492 / Pr) ** (9 / 16)) ** (8 / 27)) ** 2
    return Nu * k / H * (Ts - Tinf) - q
Ts = brentq(ts_res, Tinf + 0.1, Tinf + 200)
ref = {'T_s': Ts}
natk = """
T_inf = {Ti}
H = {H}
q = {q}
p = {p}
g = 9.81
T_f = (T_s + T_inf)/2
beta = 1/T_f
nu = viscosity(air, T=T_f, p=p)/density(air, T=T_f, p=p)
k = conductivity(air, T=T_f, p=p)
Pr = prandtl(air, T=T_f, p=p)
Ra = g*beta*(T_s - T_inf)*H^3/nu^2*Pr
Nu = (0.825 + 0.387*Ra^(1/6)/(1 + (0.492/Pr)^(9/16))^(8/27))^2
q = Nu*k/H*(T_s - T_inf)
"""
check("F6 Freie Konvektion (Filmtemperatur-Iteration) MIT Einheiten", natk.format(Ti="20 °C", H="50 cm", q="100 W/m^2", p="1 bar"), ref)
check("F6 Freie Konvektion (Filmtemperatur-Iteration) OHNE Einheiten", natk.format(Ti="293.15", H="0.5", q="100", p="100000"), ref)


if __name__ == '__main__':
    summary()
