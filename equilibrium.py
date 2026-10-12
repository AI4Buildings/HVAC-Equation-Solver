"""
Chemisches Gleichgewicht durch Minimierung der freien Enthalpie (Gibbs-Minimierung)

Für die Anweisung EQUILIBRIUM im Blatt (parser.parse_equilibria):

    EQUILIBRIUM T=T_3, p=p_3
      IN   IG:CH4 = n_CH4_2, IG:O2 = n_O2_2, IG:N2 = n_N2_2
      OUT  IG:CH4 = n_CH4_3, IG:H2O = n_H2O_3, IG:CO = n_CO_3, IG:CO2 = n_CO2_3, IG:H2 = n_H2_3, IG:N2 = n_N2_3

Gesucht ist die Zusammensetzung n der Austrittsstoffe mit minimaler freier Enthalpie bei T, p

    min G(n) = sum_i n_i*mu_i,   mu_i = g°_i(T) + R*T*ln(n_i/n_Gas * p/p°)   (Gase, ideales Gemisch)
                                 mu_c = g°_c(T)                                (reine kondensierte Phase)
    unter den Elementbilanzen A*n = b (b aus den Eintrittsstoffen) und n >= 0.

Für ideale Gasgemische ist G bei festem T und p konvex - das Minimum ist eindeutig, es gibt keine
Auswahl unter mehreren Lösungen und keine negativen Stoffmengen. Gerechnet wird wie in NASA CEA
(Gordon/McBride, NASA RP-1311) über die Elementpotentiale lambda_j (Lagrange-Multiplikatoren der
Elementbilanzen): im Minimum gilt mu_i/(R*T) = sum_j a_ji*lambda_j, also für Gase

    n_i = n_Gas * exp(sum_j a_ji*lambda_j - g°_i/(R*T) - ln(p/p°))

- jede Gasspezies ist automatisch positiv, auch Spuren von 1e-30 werden aufgelöst. Newton-Verfahren
auf (lambda, ln n_Gas) mit Dämpfung. Kondensierte Stoffe (C(gr), H2O(L)) über aktive Mengen: eine
Phase kommt hinzu, wenn sie stabil wäre (g°_c/(R*T) < a_c*lambda), und fällt weg, wenn ihre Menge
negativ würde. Rückfall: direkte Minimierung von G mit Grenzen (scipy trust-constr, wie fmincon).

Stoffwerte aus ideal_gas.py (NASA thermo.inp). Stoffmengen in beliebiger, aber gleicher Einheit
(mol oder mol/s): das Ergebnis skaliert mit der Eintrittsmenge.
"""

from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

import numpy as np

import ideal_gas

SEPARATOR = '|'           # trennt die Stoffnamen im erzeugten Aufruf (kommt in keinem Namen vor)
_TOL = 1e-11              # relative Genauigkeit der Elementbilanzen und der Gleichgewichtsbedingungen


class EquilibriumError(ValueError):
    """Gleichgewicht nicht bestimmbar (Elemente fehlen, keine Gasphase, keine Konvergenz)."""


def _species(names: Sequence[str]) -> List[ideal_gas.Species]:
    result = []
    for name in names:
        part = ideal_gas.species_part(name)
        if part is None:
            raise EquilibriumError(f"EQUILIBRIUM: '{name}' ist kein IG:-Stoff (Präfix IG: angeben)")
        result.append(ideal_gas.find_species(part))
    return result


def _element_matrix(species: Sequence[ideal_gas.Species], elements: Sequence[str]) -> np.ndarray:
    matrix = np.zeros((len(elements), len(species)))
    index = {e: j for j, e in enumerate(elements)}
    for i, sp in enumerate(species):
        for element, count in sp.elements:
            if element in index:
                matrix[index[element], i] = count
    return matrix


def element_amounts(in_species: Sequence[str], in_amounts: Sequence[float]) -> Dict[str, float]:
    """Elementmengen b_j der Eintrittsstoffe (in der Einheit der Stoffmengen)."""
    amounts: Dict[str, float] = {}
    for sp, n in zip(_species(in_species), in_amounts):
        for element, count in sp.elements:
            amounts[element] = amounts.get(element, 0.0) + count * float(n)
    return amounts


def _newton(A_g, A_c, c_g, c_c, b, lam, ln_n, n_c, max_iter=200):
    """
    Newton-Verfahren auf die Gleichgewichtsbedingungen (normierte Elementmengen b, Summe 1):
      Elementbilanzen   N*A_g*x + A_c*n_c = b,   x_i = exp(a_i*lambda - c_i)
      Molenbrüche       sum x_i = 1
      Kondensat (aktiv) a_c*lambda = c_c
    Unbekannte: lambda (Elementpotentiale), ln N (Gasmenge), n_c (aktive kondensierte Stoffe).
    """
    m, k_c = len(b), A_c.shape[1]
    b_scale = np.maximum(b, 1e-300)

    def state(lam, ln_n, n_c):
        z = np.clip(A_g.T @ lam - c_g, -745.0, 700.0)
        x = np.exp(z)
        n_gas = np.exp(ln_n)
        r_e = n_gas * (A_g @ x) + (A_c @ n_c if k_c else 0.0) - b
        r_s = np.sum(x) - 1.0
        r_c = A_c.T @ lam - c_c if k_c else np.zeros(0)
        return x, n_gas, r_e, r_s, r_c

    def merit(r_e, r_s, r_c):
        return float(np.sum((r_e / b_scale) ** 2) + r_s ** 2 + np.sum(r_c ** 2))

    x, n_gas, r_e, r_s, r_c = state(lam, ln_n, n_c)
    current = merit(r_e, r_s, r_c)
    polish = 0
    for _ in range(max_iter):
        if (np.max(np.abs(r_e) / b_scale) < _TOL and abs(r_s) < _TOL
                and (not k_c or np.max(np.abs(r_c)) < _TOL)):
            # konvergiert: bis zu zwei weitere Schritte, solange der Fehler sinkt (Maschinengenauigkeit -
            # eine kleine Kondensatmenge ist die Differenz großer Mengen, n_L = n_W - n_Dampf)
            polish += 1
            if polish > 2:
                return lam, ln_n, n_c, True
        # Jacobi-Matrix: Zeilen (Elemente, Summe, Kondensat), Spalten (lambda, ln N, n_c)
        size = m + 1 + k_c
        J = np.zeros((size, size))
        weighted = A_g * x                        # a_ji * x_i
        J[:m, :m] = n_gas * (weighted @ A_g.T)
        J[:m, m] = n_gas * (A_g @ x)
        if k_c:
            J[:m, m + 1:] = A_c
            J[m + 1:, :m] = A_c.T
        J[m, :m] = A_g @ x
        residual = np.concatenate([r_e, [r_s], r_c])
        # Zeilen der Elementbilanzen relativ zur Elementmenge (Elemente mit 1e-4 neben 1)
        row_scale = np.concatenate([b_scale, [1.0], np.ones(k_c)])
        step = np.linalg.lstsq(J / row_scale[:, None], -residual / row_scale, rcond=None)[0]
        if not np.all(np.isfinite(step)):
            return lam, ln_n, n_c, False
        # Dämpfung: Änderung der Elementpotentiale bzw. ln N höchstens 2 je Schritt (wie CEA)
        largest = max(np.max(np.abs(step[:m + 1])), 1e-300)
        factor = min(1.0, 2.0 / largest)
        accepted = False
        for _ in range(40 if not polish else 1):
            trial_lam = lam + factor * step[:m]
            trial_ln = ln_n + factor * step[m]
            trial_nc = n_c + factor * step[m + 1:] if k_c else n_c
            t = state(trial_lam, trial_ln, trial_nc)
            value = merit(t[2], t[3], t[4])
            if np.isfinite(value) and value < current:
                lam, ln_n, n_c = trial_lam, trial_ln, trial_nc
                x, n_gas, r_e, r_s, r_c = t
                current = value
                accepted = True
                break
            factor *= 0.5
        if not accepted:
            if polish:
                return lam, ln_n, n_c, True
            break
    converged = (np.max(np.abs(r_e) / b_scale) < 1e-8 and abs(r_s) < 1e-8
                 and (not k_c or np.max(np.abs(r_c)) < 1e-8))
    return lam, ln_n, n_c, converged


def _initial_potentials(A_g, c_g, b):
    """Startwerte: Molenbrüche gleich verteilt, Elementpotentiale nach kleinsten Quadraten."""
    k_g = A_g.shape[1]
    lam = np.linalg.lstsq(A_g.T, np.full(k_g, -np.log(k_g)) + c_g, rcond=None)[0]
    x = np.exp(np.clip(A_g.T @ lam - c_g, -745.0, 700.0))
    ax = A_g @ x
    n_gas = float(ax @ b / max(ax @ ax, 1e-300))
    return lam, np.log(n_gas) if n_gas > 0 else 0.0


def _inner(A_g, c_g, ln_n, b, A_c, c_c, lam):
    """
    Bei fester Gasmenge N: Elementpotentiale lambda minimieren die streng konvexe Funktion
        Phi(lambda) = sum_i N*exp(a_i*lambda - c_i) - b*lambda   unter  A_c^T*lambda = c_c
    (aktive kondensierte Stoffe; ihre Mengen sind die Lagrange-Multiplikatoren). Gradient
    A_g*n - b (Elementbilanz), Hesse-Matrix A_g*diag(n)*A_g^T - gedämpftes Newton-Verfahren mit
    Armijo-Schrittweite, global konvergent. Rückgabe (lambda, n_Gas, n_c, konvergiert).
    """
    m, k_c = len(b), A_c.shape[1]
    b_scale = np.maximum(b, 1e-300)
    if k_c:   # Start auf die Ebene der aktiven Kondensate projizieren
        lam = lam + np.linalg.lstsq(A_c.T, c_c - A_c.T @ lam, rcond=None)[0]

    def phi(l):
        z = A_g.T @ l - c_g + ln_n
        if np.max(z) > 700:
            return np.inf, None
        n = np.exp(np.maximum(z, -745.0))
        return float(np.sum(n) - b @ l), n

    value, n = phi(lam)
    n_c = np.zeros(k_c)
    for _ in range(300):
        if n is None:
            return lam, None, n_c, False
        gradient = A_g @ n - b
        H = (A_g * n) @ A_g.T
        # minimale Regularisierung: Mengen von 1e-42 bis 1 machen H fast singulär (Kondition 1e17);
        # lstsq ließe die Richtung mit verschwindender Krümmung weg (O2 bleibt bei 1e-42 hängen) -
        # so wird der Schritt dort groß und von der Begrenzung unten auf 2 gekürzt
        H = H + np.eye(m) * (1e-13 * max(float(np.max(np.diag(H))), 1e-300))
        if k_c:
            K = np.zeros((m + k_c, m + k_c))
            K[:m, :m], K[:m, m:], K[m:, :m] = H, A_c, A_c.T
            rhs = np.concatenate([-gradient, np.zeros(k_c)])
            sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
            step, n_c = sol[:m], sol[m:]
        else:
            step = np.linalg.lstsq(H, -gradient, rcond=None)[0]
        balance = A_g @ n + (A_c @ n_c if k_c else 0.0) - b
        if np.max(np.abs(balance) / b_scale) < _TOL:
            return lam, n, n_c, True
        # Schritt der Elementpotentiale höchstens 2 (Mengen höchstens Faktor e^2 je Schritt, wie CEA):
        # ist ein Element fast nur in verschwindend kleinen Spezies, wäre der Newton-Schritt riesig
        largest = float(np.max(np.abs(step), initial=0.0))
        if largest > 2.0:
            step = step * (2.0 / largest)
        decrease = gradient @ step
        current_error = np.max(np.abs(balance) / b_scale)
        factor = 1.0
        for _ in range(60):
            trial = lam + factor * step
            trial_value, trial_n = phi(trial)
            if trial_n is not None:
                # Armijo-Abstieg; nahe am Minimum ändert sich Phi unterhalb der Rundung von Phi -
                # dann zählt, dass der Fehler der Elementbilanz sinkt (quadratische Konvergenz)
                if trial_value <= value + 1e-4 * factor * decrease:
                    break
                trial_balance = A_g @ trial_n - b
                if k_c:
                    trial_balance = trial_balance + A_c @ n_c
                if np.max(np.abs(trial_balance) / b_scale) < 0.5 * current_error:
                    break
            factor *= 0.5
        else:
            return lam, n, n_c, False
        lam, value, n = trial, trial_value, trial_n
    return lam, n, n_c, False


def _robust(A_g, c_g, A_c, c_c, b, lam):
    """
    Global konvergent: äußere Nullstellensuche (Brent) für ln N mit sum n(N) = N, innen _inner.
    Klammer: N liegt zwischen sum b/(größte Atomzahl einer Gasspezies) und sum b/(kleinste).
    """
    from scipy.optimize import brentq
    atoms = A_g.sum(axis=0)
    total = float(np.sum(b))
    if np.all(atoms > 0):
        lo, hi = np.log(total / atoms.max()) - 1e-6, np.log(total / atoms.min()) + 1e-6
    else:
        lo, hi = np.log(total) - 30.0, np.log(total) + 30.0
    if A_c.shape[1]:
        lo -= 60.0    # kondensierte Phasen nehmen Elemente auf: die Gasmenge kann viel kleiner sein
    # frische Startwerte (die des gescheiterten Newton-Verfahrens können extrem sein: exp-Überlauf)
    memory = {'lam': _initial_potentials(A_g, c_g, b)[0]}

    def h(ln_n):
        result = _inner(A_g, c_g, ln_n, b, A_c, c_c, memory['lam'])
        if not result[3]:
            raise EquilibriumError("innere Minimierung nicht konvergiert")
        memory['lam'] = result[0]
        return float(np.log(np.sum(result[1])) - ln_n)

    h_lo, h_hi = h(lo), h(hi)
    for _ in range(20):
        if h_lo >= 0:
            break
        lo -= 10.0
        h_lo = h(lo)
    for _ in range(20):
        if h_hi <= 0:
            break
        hi += 10.0
        h_hi = h(hi)
    if h_lo < 0 or h_hi > 0:
        raise EquilibriumError("keine Klammer für die Gasmenge")
    ln_n = brentq(h, lo, hi, xtol=1e-14, rtol=1e-15, maxiter=200)
    lam, n, n_c, converged = _inner(A_g, c_g, ln_n, b, A_c, c_c, memory['lam'])
    return lam, ln_n, n_c, converged


def _minimize_direct(A, c, gas, b):
    """Rückfall wie fmincon: min G(n) unter A*n = b, n >= 0 (scipy trust-constr)."""
    from scipy.optimize import minimize, LinearConstraint, Bounds
    k = A.shape[1]
    tiny = 1e-300

    def potentials(n):
        n = np.maximum(n, tiny)
        n_gas = max(np.sum(n[gas]), tiny)
        mu = c.copy()
        mu[gas] = c[gas] + np.log(n[gas] / n_gas)
        return n, mu

    def gibbs(n):
        n, mu = potentials(n)
        return float(n @ mu)

    def gradient(n):
        return potentials(n)[1]

    start = np.full(k, float(np.sum(b)) / k)
    result = minimize(gibbs, start, jac=gradient, method='trust-constr',
                      constraints=[LinearConstraint(A, b, b)],
                      bounds=Bounds(np.zeros(k), np.full(k, np.inf)),
                      options={'maxiter': 3000, 'gtol': 1e-12, 'xtol': 1e-14})
    return np.maximum(result.x, 0.0)


def _feasibility_error(A, b, present, in_species=(), out_species=()) -> str:
    """
    Ist die Elementbilanz A*n = b mit n >= 0 erfüllbar (NNLS)? Sonst Meldung mit dem Element, das
    übrig bleibt (H2 + O2 mit nur H2, H2O unter OUT: Überschuss O) - ohne diese Prüfung lief jeder
    Aufruf in den langsamen Rückfall und meldete "keine Konvergenz".
    """
    from scipy.optimize import nnls
    n, _ = nnls(A, b, maxiter=50 * A.shape[1])
    rest = b - A @ n
    if np.max(np.abs(rest)) <= 1e-9 * max(np.max(np.abs(b)), 1e-300):
        return ''
    surplus = [e for e, r in zip(present, rest) if r > 1e-9 * np.max(np.abs(b))]
    names = ', '.join(surplus) or ', '.join(present)
    # Vorschlag: Eintrittsstoffe, die das übrige Element enthalten und noch nicht unter OUT stehen
    listed = {sp.name for sp in out_species}
    examples = [f"IG:{sp.name}" for sp in in_species
                if sp.name not in listed and any(e in surplus for e, _ in sp.elements)]
    hint = f" (z.B. {', '.join(examples)})" if examples else ""
    return (f"EQUILIBRIUM: die Elementbilanz ist mit den Stoffen unter OUT nicht erfüllbar - Überschuss an "
            f"{names}: einen Stoff mit {names} unter OUT angeben{hint}")


def _condensed_only(A, c, gas, b):
    """
    Verschwindet die Gasphase (alles kondensiert, z.B. H2 + O2 stöchiometrisch bei 350 K -> nur
    H2O(L))? Kondensierte Phasen allein: lineares Programm min c_c*n_c unter A_c*n_c = b, n_c >= 0.
    Die Gasphase fehlt im Minimum, wenn es Elementpotentiale lambda gibt mit a_c*lambda = c_c für die
    vorhandenen, <= c_c für die übrigen kondensierten Stoffe und sum_g exp(a_g*lambda - c_g) <= 1
    (eine kleine Gasphase senkt G dann nicht - Optimalitätsbedingung bei n_Gas = 0). Rückgabe: Mengen
    oder None (Gasphase vorhanden -> übliches Verfahren, das ln n_Gas rechnet und 0 nie erreicht).
    """
    from scipy.optimize import linprog, minimize
    from scipy.special import logsumexp
    cond = ~gas
    if not cond.any():
        return None
    A_c, c_c = A[:, cond], c[cond]
    lp = linprog(c_c, A_eq=A_c, b_eq=b, bounds=[(0, None)] * A_c.shape[1], method='highs')
    if lp.status != 0 or np.max(np.abs(A_c @ lp.x - b)) > 1e-10 * np.max(np.abs(b)):
        return None
    n_c = np.maximum(lp.x, 0.0)
    present = n_c > 1e-14 * np.max(n_c)
    A_g, c_g = A[:, gas], c[gas]
    constraints = []
    if present.any():
        constraints.append({'type': 'eq', 'fun': lambda lam: A_c[:, present].T @ lam - c_c[present],
                            'jac': lambda lam: A_c[:, present].T})
    if (~present).any():
        constraints.append({'type': 'ineq', 'fun': lambda lam: c_c[~present] - A_c[:, ~present].T @ lam,
                            'jac': lambda lam: -A_c[:, ~present].T})

    def lse(lam):
        z = A_g.T @ lam - c_g
        value = logsumexp(z)
        return value, A_g @ np.exp(z - value)

    start = np.linalg.lstsq(A_c[:, present].T, c_c[present], rcond=None)[0] if present.any() \
        else np.zeros(A.shape[0])
    result = minimize(lse, start, jac=True, method='SLSQP', constraints=constraints,
                      options={'maxiter': 500, 'ftol': 1e-14})
    feasible = all(np.all(np.abs(con['fun'](result.x)) <= 1e-8) if con['type'] == 'eq'
                   else np.all(con['fun'](result.x) >= -1e-8) for con in constraints)
    if not (feasible and lse(result.x)[0] <= 0.0):
        return None
    n = np.zeros(A.shape[1])
    n[np.where(cond)[0]] = n_c
    return n


def _solve(out_species: Sequence[ideal_gas.Species], b_full: Dict[str, float], T: float, p: float,
           labels: Sequence[str], in_species: Sequence[ideal_gas.Species] = ()) -> np.ndarray:
    """Gleichgewichtsmengen der Austrittsstoffe (in der Einheit der Elementmengen)."""
    for element, amount in b_full.items():
        if amount < -1e-12 * max(abs(v) for v in b_full.values()):
            raise EquilibriumError(f"EQUILIBRIUM: negative Elementmenge {element} = {amount:.6g} "
                                   f"(Eintrittsmengen prüfen)")
    total = sum(max(v, 0.0) for v in b_full.values())
    if not total > 0:
        raise EquilibriumError("EQUILIBRIUM: die Eintrittsstoffe enthalten keine Elemente (Mengen null)")
    # Normierung mit einer Zweierpotenz: exakt, stöchiometrische Verhältnisse (H:O = 2:1) bleiben
    # erhalten - sonst bestimmte der Rundungsrest von 2/3, 1/3 die Spuren (O2 5e-16 statt 1e-25)
    total = 2.0 ** np.round(np.log2(total))
    present = sorted(e for e, v in b_full.items() if v > 1e-14 * total)
    # Austrittsstoffe mit einem Element, das am Eintritt fehlt, haben die Menge 0; ebenso Stoffe außerhalb
    # ihres Datenbereichs (Festlegung des Nutzers 12.10.2026: weglassen mit Hinweis, keine Extrapolation;
    # kondensierte wie NASA CEA: H2O(cr) nur bis 273.15 K, H2O(L) ab 273.15 K; Gase z.B. NO2 erst ab
    # 300 K - main._equilibrium_range_hints meldet jeden ausgelassenen Stoff)
    def in_range(sp):
        return any(lo - 1e-9 <= T <= hi + 1e-9 for lo, hi in zip(sp.t_min, sp.t_max))
    relevant = [i for i, sp in enumerate(out_species) if all(e in present for e, _ in sp.elements)]
    usable = [i for i in relevant if in_range(out_species[i])]
    outside = [out_species[i] for i in relevant if not in_range(out_species[i])]

    def out_of_range(candidates):
        ranges = ', '.join(f"{sp.label}: {min(sp.t_min):.6g} ... {max(sp.t_max):.6g} K" for sp in candidates)
        return (f"bei T = {T:.6g} K liegen sie außerhalb ihres Datenbereichs ({ranges}) - T im Datenbereich "
                f"wählen oder einen Stoff mit Daten für diese Temperatur angeben")
    missing = [e for e in present if not any(e == el for i in usable for el, _ in out_species[i].elements)]
    if missing:
        carriers = [sp for sp in outside if any(e in missing for e, _ in sp.elements)]
        if carriers:
            raise EquilibriumError(f"EQUILIBRIUM: die Stoffe mit {', '.join(missing)} unter OUT fallen weg - "
                                   + out_of_range(carriers))
        raise EquilibriumError(
            f"EQUILIBRIUM: Element {', '.join(missing)} aus den Eintrittsstoffen kommt in keinem "
            f"Austrittsstoff vor - einen Stoff mit {', '.join(missing)} unter OUT angeben")
    species = [out_species[i] for i in usable]
    gas = np.array([not sp.condensed for sp in species])
    if not gas.any():
        gases = [sp for sp in outside if not sp.condensed]
        if gases:
            raise EquilibriumError("EQUILIBRIUM: alle Gase unter OUT fallen weg - " + out_of_range(gases))
        raise EquilibriumError("EQUILIBRIUM: unter OUT mindestens einen gasförmigen Stoff angeben")
    A = _element_matrix(species, present)
    b = np.array([b_full[e] for e in present]) / total
    g_rt = np.array([ideal_gas.standard_gibbs_RT(sp, T) for sp in species])
    c = g_rt + np.where(gas, np.log(p / ideal_gas.P_STANDARD), 0.0)
    A_g, c_g = A[:, gas], c[gas]
    cond_index = [i for i in range(len(species)) if not gas[i]]
    error = _feasibility_error(A, b, present, in_species, out_species)
    if error:
        raise EquilibriumError(error)
    only_condensed = _condensed_only(A, c, gas, b) if cond_index else None
    if only_condensed is not None:
        result = np.zeros(len(out_species))
        for value, i in zip(only_condensed, usable):
            result[i] = value * total
        return result

    lam, ln_n = _initial_potentials(A_g, c_g, b)
    active: List[int] = []
    n_c = np.zeros(0)
    converged = False
    seen = set()
    for _ in range(2 * len(cond_index) + 3):
        A_c = A[:, active] if active else np.zeros((len(present), 0))
        c_c = c[active] if active else np.zeros(0)
        lam, ln_n, n_c, converged = _newton(A_g, A_c, c_g, c_c, b, lam, ln_n, n_c)
        if not converged:
            try:
                lam, ln_n, n_c, converged = _robust(A_g, c_g, A_c, c_c, b, lam)
            except (EquilibriumError, ValueError, FloatingPointError):
                converged = False
        if not converged:
            break
        # Kondensierte Stoffe: aktive mit negativer Menge entfernen, stabile inaktive hinzufügen
        if active and np.min(n_c) < 0:
            drop = int(np.argmin(n_c))
            active.pop(drop)
            n_c = np.delete(n_c, drop)
        else:
            inactive = [i for i in cond_index if i not in active]
            driving = [(c[i] - A[:, i] @ lam, i) for i in inactive]
            candidates = [(d, i) for d, i in driving if d < -1e-9]
            if not candidates:
                break
            add = min(candidates)[1]
            active.append(add)
            n_c = np.append(n_c, 0.0)
        key = tuple(sorted(active))
        if key in seen:
            break
        seen.add(key)

    n = np.zeros(len(species))
    if converged and (not active or np.min(n_c) >= -1e-14):
        z = np.clip(A_g.T @ lam - c_g, -745.0, 700.0)
        n[gas] = np.exp(ln_n) * np.exp(z)
        for value, i in zip(n_c, active):
            n[i] = max(value, 0.0)
    else:
        # Rückfall: direkte Minimierung (wie fmincon) - Ergebnis prüfen
        n = _minimize_direct(A, c, gas, b)
        if np.max(np.abs(A @ n - b) / np.maximum(b, 1e-300)) > 1e-6:
            raise EquilibriumError(
                f"EQUILIBRIUM bei T = {T:.6g} K, p = {p:.6g} Pa: keine Konvergenz der Gibbs-Minimierung")
    result = np.zeros(len(out_species))
    for value, i in zip(n, usable):
        result[i] = value * total
    return result


def excluded_species(out_names: Sequence[str], temperatures, in_names: Sequence[str] = ()
                     ) -> List[Tuple[str, bool, float, float, float, float, List[str]]]:
    """
    Austrittsstoffe, die bei den Temperaturen (Zahl oder Array) außerhalb ihres Datenbereichs liegen
    und deshalb ausgelassen werden (Menge 0) - für einen Hinweis: (Stoff, kondensiert?, Bereich von,
    bis, betroffene T von, bis, kondensierte Stoffe derselben Zusammensetzung, deren Bereich diese T
    abdeckt - nur für kondensierte Stoffe, z.B. H2O(L) -> H2O(cr)).
    """
    values = np.atleast_1d(np.asarray(temperatures, dtype=float)).ravel()
    values = values[np.isfinite(values)]
    # nur Stoffe, deren Elemente am Eintritt vorkommen (sonst ist die Menge ohnehin 0)
    present = None
    if in_names:
        present = {e for sp in _species(in_names) for e, _ in sp.elements}
    found = []
    for name in out_names:
        try:
            sp = ideal_gas.find_species(name[3:] if name.upper().startswith('IG:') else name)
        except Exception:
            continue
        if present is not None and not all(e in present for e, _ in sp.elements):
            continue
        outside = [T for T in values
                   if not any(lo - 1e-9 <= T <= hi + 1e-9 for lo, hi in zip(sp.t_min, sp.t_max))]
        if not outside:
            continue
        composition = sorted(sp.elements)
        alternatives = [] if not sp.condensed else [
            f"IG:{other.name}" for other in ideal_gas.database().values()
            if other.condensed and other.name != sp.name and sorted(other.elements) == composition
            and all(any(lo - 1e-9 <= T <= hi + 1e-9 for lo, hi in zip(other.t_min, other.t_max)) for T in outside)]
        found.append((f"IG:{sp.name}", sp.condensed, min(sp.t_min), max(sp.t_max), min(outside), max(outside),
                      alternatives))
    return found


def _equilibrium_cached(out_names: Tuple[str, ...], in_names: Tuple[str, ...], T: float, p: float,
                        in_amounts: Tuple[float, ...]) -> Tuple[float, ...]:
    """Gleichgewicht je (Stoffe, T, p, Mengen) zwischengespeichert - auch Fehler (die 1-D-Suche ruft
    einen unzulässigen Punkt sonst hundertfach neu auf)."""
    values, error = _equilibrium_or_error(out_names, in_names, T, p, in_amounts)
    if error is not None:
        raise error[0](error[1])
    return values


@lru_cache(maxsize=1024)
def _equilibrium_or_error(out_names, in_names, T, p, in_amounts):
    try:
        return _equilibrium_uncached(out_names, in_names, T, p, in_amounts), None
    except (EquilibriumError, ValueError) as exc:
        return None, (type(exc), str(exc))


def _equilibrium_uncached(out_names: Tuple[str, ...], in_names: Tuple[str, ...], T: float, p: float,
                          in_amounts: Tuple[float, ...]) -> Tuple[float, ...]:
    if not (np.isfinite(T) and T > 0):
        raise EquilibriumError(f"EQUILIBRIUM: ungültige Temperatur T = {T}")
    if not (np.isfinite(p) and p > 0):
        raise EquilibriumError(f"EQUILIBRIUM: Druck p = {p} muss positiv sein")
    if not all(np.isfinite(a) for a in in_amounts):
        raise EquilibriumError("EQUILIBRIUM: ungültige Eintrittsmenge")
    out_species = _species(out_names)
    b = element_amounts(in_names, in_amounts)
    return tuple(float(v) for v in _solve(out_species, b, T, p, out_names, _species(in_names)))


def equilibrium_amounts(out_names: Sequence[str], in_names: Sequence[str], T: float, p: float,
                        in_amounts: Sequence[float]) -> Dict[str, float]:
    """Gleichgewichtsmengen {Stoff: Menge} der Austrittsstoffe (Einheit wie die Eintrittsmengen)."""
    values = _equilibrium_cached(tuple(out_names), tuple(in_names), float(T), float(p),
                                 tuple(float(a) for a in in_amounts))
    return dict(zip(out_names, values))


def equilibrium(name: str, in_list: str, out_list: str, T, p, *in_amounts):
    """
    Menge des Austrittsstoffs name im Gleichgewicht - der Parser erzeugt aus jeder
    EQUILIBRIUM-Anweisung je Austrittsstoff einen solchen Aufruf. Elementweise für Arrays
    (Parameterstudien).
    """
    in_names = tuple(in_list.split(SEPARATOR))
    out_names = tuple(out_list.split(SEPARATOR))
    if name not in out_names:
        raise EquilibriumError(f"EQUILIBRIUM: {name} steht nicht unter OUT")
    position = out_names.index(name)
    arrays = [np.asarray(v, dtype=float) for v in (T, p) + tuple(in_amounts)]
    if all(a.ndim == 0 for a in arrays):
        return _equilibrium_cached(out_names, in_names, float(arrays[0]), float(arrays[1]),
                                   tuple(float(a) for a in arrays[2:]))[position]
    shape = np.broadcast_shapes(*(a.shape for a in arrays))
    flat = [np.broadcast_to(a, shape).ravel() for a in arrays]
    result = np.array([
        _equilibrium_cached(out_names, in_names, float(flat[0][k]), float(flat[1][k]),
                            tuple(float(f[k]) for f in flat[2:]))[position]
        for k in range(flat[0].size)])
    return result.reshape(shape)


if __name__ == "__main__":
    print("=== Gibbs-Minimierung (equilibrium.py) - Tests ===\n")

    def check(text, value, reference, tolerance):
        ok = abs(value - reference) <= tolerance
        print(f"  {'ok ' if ok else 'FEHLER'} {text}: {value:.6g} (Referenz {reference:.6g})")
        if not ok:
            raise SystemExit(1)

    # Beispiel: CH4 + Luft (lambda = 0.33) bei 745 °C, 1 bar - wie die K-Gleichungen (Dampfreformierung
    # und Shift) im Lehrbeispiel FuelCell/Beispiel_01, aber ohne Reaktionswahl
    out = ['IG:CH4', 'IG:H2O', 'IG:CO', 'IG:CO2', 'IG:H2', 'IG:N2']
    n_o2 = 2 * 0.33
    n = equilibrium_amounts(out, ['IG:CH4', 'IG:O2', 'IG:N2'], 1018.15, 1e5, [1.0, n_o2, n_o2 * 0.79 / 0.21])
    total = sum(n.values())
    check("x_CH4 in % (K-Form 0.3263 %)", 100 * n['IG:CH4'] / total, 0.3263, 5e-4)
    check("x_H2 in % (K-Form 32.2894 %)", 100 * n['IG:H2'] / total, 32.2894, 1e-3)
    # Elementbilanz und Positivität
    check("C-Bilanz", n['IG:CH4'] + n['IG:CO'] + n['IG:CO2'], 1.0, 1e-10)
    # Spurenanteil: O2 im Gleichgewicht mit Wasserstoffüberschuss (Knallgas), Kp aus g°
    m = equilibrium_amounts(['IG:H2', 'IG:O2', 'IG:H2O'], ['IG:H2', 'IG:O2'], 1000.0, 1e5, [1.0, 0.25])
    print(f"  ok  Spur O2 bei 1000 K: {m['IG:O2']:.3e} mol (positiv, aufgelöst)")
    # Kondensat: Wasser bei 300 K, 1 bar - Partialdruck im Gleichgewicht = Sättigungsdruck
    w = equilibrium_amounts(['IG:N2', 'IG:H2O', 'IG:H2O(L)'], ['IG:N2', 'IG:H2O'], 300.0, 1e5, [1.0, 0.2])
    p_w = w['IG:H2O'] / (w['IG:N2'] + w['IG:H2O']) * 1e5
    check("Sättigungsdruck Wasser 300 K aus dem Gleichgewicht in Pa (CoolProp 3536.8)", p_w, 3536.8, 40)
    print("\nAlle Tests bestanden")
