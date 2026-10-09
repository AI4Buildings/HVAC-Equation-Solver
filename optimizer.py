"""
Optimierung für den HVAC Equation Solver:

    MAXIMIZE ziel VARY x = a .. b Einheit[, y = c .. d Einheit]
    MINIMIZE ziel VARY ...

Generischer Ansatz (keine Annahmen über die Form der Gleichungen):
- Die variierten Größen werden wie Konstanten behandelt; für jeden Kandidaten
  löst solve_system das übrige Gleichungssystem (Warm-Start mit der zuletzt
  gefundenen Lösung). Zielfunktion = Wert der Zielgröße in dieser Lösung.
- Globale Phase: Raster über den ganzen Bereich. Es findet den Bereich des
  besten von mehreren lokalen Optima; Kandidaten ohne Lösung gelten als schlecht
  (kein Abbruch). Bereiche über mehr als zwei Zehnerpotenzen (untere Grenze > 0)
  werden logarithmisch gerastert.
- Lokale Phase: eine Größe -> Brent (beschränkt) zwischen den Nachbarn des
  besten Rasterpunkts; mehrere Größen -> Powell (beschränkt) ab dem besten Punkt.
- Mehrere Anweisungen werden der Reihe nach optimiert. Hängt eine Zielgröße von
  den Größen einer anderen Anweisung ab, wird lokal wiederholt, bis sich nichts
  mehr ändert (sonst Hinweis: zu einer Zielgröße zusammenfassen).
- Zeitlimit OPTIMIZE_TIME_LIMIT je Optimierung (bzw. je Punkt einer
  Parameterstudie): danach das beste bisher gefundene Ergebnis.
"""

import math
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.optimize import minimize, minimize_scalar

import solver as _solver
from solver import SolveTimeout, solve_parametric, solve_system

OPTIMIZE_TIME_LIMIT = 120.0   # s je Optimierung (je Punkt einer Parameterstudie)
GRID_POINTS_1D = 21           # Rasterpunkte bei einer variierten Größe
GRID_BUDGET = 100             # Rasterpunkte gesamt bei mehreren Größen
MAX_ROUNDS = 5                # Runden bei mehreren, voneinander abhängigen Anweisungen
_PENALTY = 1e30               # Zielwert eines Kandidaten ohne Lösung (für die lokale Suche)
_AT_BOUND = 1e-4              # Abstand zur Grenze (Anteil am Bereich) für "am Rand"


def _display(name: str) -> str:
    return name.replace('_kw_', '')


def _is_log(lower: float, upper: float) -> bool:
    """Bereich über mehr als zwei Zehnerpotenzen (positiv) -> logarithmisch rastern."""
    return lower > 0 and upper / lower > 100


def _to_x(u: float, lower: float, upper: float) -> float:
    """Position u in [0, 1] -> Wert im Bereich (linear bzw. logarithmisch)."""
    u = min(max(float(u), 0.0), 1.0)
    if u == 0.0:
        return lower
    if u == 1.0:
        return upper
    if _is_log(lower, upper):
        return lower * (upper / lower) ** u
    return lower + u * (upper - lower)


def _to_u(x: float, lower: float, upper: float) -> float:
    if _is_log(lower, upper):
        return math.log(x / lower) / math.log(upper / lower)
    return (x - lower) / (upper - lower)


def _snake(axis: np.ndarray, k: int) -> List[Tuple[float, ...]]:
    """Rasterpunkte in Schlangenlinie: Nachbarn nacheinander (guter Warm-Start)."""
    if k == 1:
        return [(float(a),) for a in axis]
    inner = _snake(axis, k - 1)
    points = []
    for i, a in enumerate(axis):
        points.extend((float(a),) + p for p in (inner if i % 2 == 0 else inner[::-1]))
    return points


class _Evaluator:
    """Zielfunktion einer Anweisung: Position u (je Größe in [0, 1]) -> Zielwert (zu minimieren)."""

    def __init__(self, equations, variables, constants, initial, original_equations, goal):
        self.equations = equations
        self.variables = variables
        self.constants = constants
        self.initial = dict(initial)
        self.warm = dict(initial)
        self.original_equations = original_equations
        self.goal = goal
        self.sign = 1.0 if goal.sense == 'min' else -1.0
        self.cache: Dict[Tuple[float, ...], float] = {}
        self.best: Optional[Tuple[float, Tuple[float, ...]]] = None   # (Zielwert, u)
        self.n_solves = 0
        self.n_failed = 0
        self.failure: Optional[str] = None

    def values(self, u) -> Dict[str, float]:
        return {name: _to_x(ui, lo, hi)
                for name, ui, lo, hi in zip(self.goal.names, u, self.goal.lower, self.goal.upper)}

    def __call__(self, u) -> float:
        key = tuple(min(max(float(v), 0.0), 1.0) for v in u)
        if key in self.cache:
            return self.cache[key]
        _solver._check_deadline()   # auch wenn das System ohne Iteration lösbar ist
        constants = dict(self.constants)
        constants.update(self.values(key))
        try:
            ok, solution, msg = solve_system(self.equations, self.variables, self.warm,
                                             constants=constants,
                                             original_equations=self.original_equations)
        except SolveTimeout:
            raise
        except Exception as exc:
            ok, solution, msg = False, {}, str(exc)
        self.n_solves += 1
        value = solution.get(self.goal.objective) if ok else None
        if value is None or not np.isfinite(value):
            f = float('nan')
            self.n_failed += 1
            if self.failure is None:
                self.failure = msg
        else:
            f = self.sign * float(value)
            self.warm = {**self.initial, **{v: float(solution[v]) for v in self.variables
                                            if v in solution and np.isfinite(solution[v])}}
            if self.best is None or f < self.best[0]:
                self.best = (f, key)
        self.cache[key] = f
        return f

    def penalized(self, u) -> float:
        f = self(u)
        return f if np.isfinite(f) else _PENALTY


def _grid_search(ev: _Evaluator, hint_u: Optional[Tuple[float, ...]]) -> List[float]:
    """Raster über den ganzen Bereich (+ Hinweis-Punkt); liefert die Zielwerte des Rasters."""
    k = len(ev.goal.names)
    if hint_u is not None:
        ev(hint_u)   # zuerst: guter Warm-Start, und ein schmales Optimum geht nicht verloren
    if k == 1:
        points = [(float(u),) for u in np.linspace(0.0, 1.0, GRID_POINTS_1D)]
    else:
        m = max(3, int(round(GRID_BUDGET ** (1.0 / k))))
        if m ** k > 3 * GRID_BUDGET:
            m = 2
        points = _snake(np.linspace(0.0, 1.0, m), k)
    return [ev(p) for p in points]


def _local_search(ev: _Evaluator, start_u: Tuple[float, ...], width: float) -> None:
    """Genaue Suche ab start_u: Brent im Intervall start ± width bzw. Powell (beschränkt)."""
    if len(start_u) == 1:
        a, b = max(0.0, start_u[0] - width), min(1.0, start_u[0] + width)
        if b > a:
            minimize_scalar(lambda u: ev.penalized((u,)), bounds=(a, b), method='bounded',
                            options={'xatol': 1e-7, 'maxiter': 200})
        return
    minimize(lambda u: ev.penalized(tuple(u)), np.array(start_u), method='Powell',
             bounds=[(0.0, 1.0)] * len(start_u),
             options={'xtol': 1e-6, 'ftol': 1e-12, 'maxfev': 300 * len(start_u)})


def _optimize_goal(ev: _Evaluator, start_u: Tuple[float, ...], full: bool,
                   hint_u: Optional[Tuple[float, ...]]) -> Dict[str, object]:
    """Optimiert eine Anweisung (full: Raster + lokal, sonst nur lokal ab start_u)."""
    info: Dict[str, object] = {'flat': False}
    if full:
        grid = _grid_search(ev, hint_u)
        finite = [f for f in grid if np.isfinite(f)]
        if len(finite) >= 2 and max(finite) - min(finite) <= 1e-12 * max(1.0, abs(min(finite))):
            info['flat'] = True
            return info
        if ev.best is None:
            return info
        step = 1.0 / (GRID_POINTS_1D - 1)
        _local_search(ev, ev.best[1], step)
    else:
        ev(start_u)
        if ev.best is None:
            return info
        _local_search(ev, ev.best[1], 2.0 / (GRID_POINTS_1D - 1))
    return info


def optimize_system(
    equations: List[str],
    variables: Set[str],
    goals: list,
    initial_values: Optional[Dict[str, float]] = None,
    constants: Optional[Dict[str, float]] = None,
    original_equations: Optional[Dict[str, str]] = None,
    return_analysis: bool = False,
    hints: Optional[Dict[str, float]] = None,
):
    """
    Löst das Gleichungssystem so, dass die Zielgrößen der Anweisungen minimal bzw.
    maximal werden (parser.OptimizationGoal, Grenzen in SI).

    Args:
        variables: Unbekannte OHNE die variierten Größen
        initial_values: Startwerte (auch für variierte Größen: zusätzlicher Kandidat)
        hints: Optimum des vorigen Punkts einer Parameterstudie (zusätzlicher Kandidat)

    Returns:
        success, solution (inkl. variierter Größen), message[, analysis] wie solve_system
    """
    initial_values = dict(initial_values or {})
    constants = dict(constants or {})
    hints = dict(hints or {})
    names = [n for g in goals for n in g.names]

    for goal in goals:
        if goal.objective in names:
            return _result(False, {}, (
                f"Zeile {goal.line}: Die Zielgröße {_display(goal.objective)} wird in einer anderen "
                f"Anweisung variiert - sie muss aus den Gleichungen folgen"), return_analysis)
        if goal.objective in constants:
            return _result(False, dict(constants), (
                f"Zeile {goal.line}: Die Zielgröße {_display(goal.objective)} hat einen festen Wert - "
                f"sie muss aus den Gleichungen folgen"), return_analysis)
        if goal.objective not in variables:
            return _result(False, {}, (
                f"Zeile {goal.line}: Die Zielgröße {_display(goal.objective)} kommt in keiner "
                f"Gleichung vor"), return_analysis)

    def start_u(goal) -> Tuple[Tuple[float, ...], Optional[Tuple[float, ...]]]:
        """Start (Hinweis bzw. Startwert, sonst Bereichsmitte) und ggf. Hinweis-Punkt."""
        hint = []
        for name, lo, hi in zip(goal.names, goal.lower, goal.upper):
            value = hints.get(name, initial_values.get(name))
            hint.append(_to_u(value, lo, hi) if value is not None and lo <= value <= hi else None)
        if all(h is not None for h in hint):
            return tuple(hint), tuple(hint)
        return tuple(0.5 if h is None else h for h in hint), None

    current: Dict[str, float] = {}
    for goal in goals:
        current.update(dict(zip(goal.names, (_to_x(u, lo, hi) for u, lo, hi in
                                             zip(start_u(goal)[0], goal.lower, goal.upper)))))

    own_deadline = _solver._deadline is None
    if own_deadline:
        _solver._deadline = time.monotonic() + OPTIMIZE_TIME_LIMIT
    warm = {k: v for k, v in initial_values.items() if k not in names}
    best_values: Dict[int, float] = {}     # bester Zielwert je Anweisung (Vorzeichen: min)
    infos: Dict[int, Dict[str, object]] = {}
    n_solves = n_failed = 0
    timed_out = False
    coupled = False
    failure = None
    try:
        for round_no in range(MAX_ROUNDS if len(goals) > 1 else 1):
            for index, goal in enumerate(goals):
                fixed = {n: v for n, v in current.items() if n not in goal.names}
                ev = _Evaluator(equations, variables, {**constants, **fixed}, warm,
                                original_equations, goal)
                start, hint_u = start_u(goal)
                if round_no > 0:
                    start = tuple(_to_u(current[n], lo, hi)
                                  for n, lo, hi in zip(goal.names, goal.lower, goal.upper))
                try:
                    infos[index] = _optimize_goal(ev, start, round_no == 0, hint_u)
                finally:
                    n_solves += ev.n_solves
                    n_failed += ev.n_failed
                if ev.best is None:
                    failure = (goal, ev.failure)
                    break
                current.update(ev.values(ev.best[1]))
                best_values[index] = ev.best[0]
                warm = ev.warm
            if failure or len(goals) == 1:
                break
            # Stimmig? Jede Zielgröße muss bei den endgültigen Werten ihr Optimum behalten
            ok, solution, _ = solve_system(equations, variables, warm, constants={**constants, **current},
                                           original_equations=original_equations)
            n_solves += 1
            same = ok and all(
                abs((1 if g.sense == 'min' else -1) * solution.get(g.objective, np.nan) - best_values[i])
                <= 1e-9 * max(1.0, abs(best_values[i])) for i, g in enumerate(goals))
            if same:
                coupled = _coupled(equations, variables, goals, constants, current, warm,
                                   original_equations, solution)
                n_solves += len(goals)
                break
        else:
            coupled = True
    except SolveTimeout:
        timed_out = True
    finally:
        if own_deadline:
            _solver._deadline = None

    if failure is not None:
        goal, reason = failure
        span = ', '.join(_display(n) for n in goal.names)
        return _result(False, {}, (f"Optimierung (Zeile {goal.line}): für keinen Wert von {span} im "
                                   f"Bereich lösbar. {reason or ''}").strip(), return_analysis)

    # Endgültige Lösung beim Optimum (mit Residuen/Analyse wie ein normaler Lauf)
    result = solve_system(equations, variables, warm, constants={**constants, **current},
                          original_equations=original_equations, return_analysis=return_analysis)
    success, solution, msg = result[:3]
    solution = dict(solution)
    solution.update(current)

    parts = []
    flat = []
    for index, goal in enumerate(goals):
        what = 'Minimum' if goal.sense == 'min' else 'Maximum'
        info = infos.get(index, {})
        if info.get('flat'):
            flat.append(f"Zeile {goal.line}: {_display(goal.objective)} ändert sich nicht, wenn "
                        f"{', '.join(_display(n) for n in goal.names)} variiert wird - hängt die "
                        f"Zielgröße davon ab?")
            continue
        edges = []
        for name, lo, hi in zip(goal.names, goal.lower, goal.upper):
            u = _to_u(current[name], lo, hi)
            if u <= _AT_BOUND:
                edges.append(f"{_display(name)} an der Untergrenze")
            elif u >= 1 - _AT_BOUND:
                edges.append(f"{_display(name)} an der Obergrenze")
        parts.append(f"{what} von {_display(goal.objective)}" + (f" ({', '.join(edges)})" if edges else ""))
    message = "Optimierung: " + "; ".join(parts) + f" - {n_solves} Lösungen des Gleichungssystems"
    if n_failed:
        message += f", davon {n_failed} Kandidaten ohne Lösung"
    if coupled:
        message += (". Die Anweisungen beeinflussen sich gegenseitig - Ergebnis, bei dem keine "
                    "Zielgröße allein besser wird; für ein gemeinsames Optimum EINE Zielgröße bilden")
    if flat:
        success = False
        message = "Optimierung nicht möglich. " + " ".join(flat)
    if timed_out:
        success = False
        message = (f"Zeitlimit der Optimierung ({OPTIMIZE_TIME_LIMIT:g} s) überschritten - bestes "
                   f"bisher gefundenes Ergebnis. {message}")
    if not result[0]:
        success = False
        message = f"{message}. Beim Optimum: {msg}"
    if return_analysis:
        return success, solution, message, result[3]
    return success, solution, message


def _coupled(equations, variables, goals, constants, current, warm, original_equations, solution) -> bool:
    """
    Hängt eine Zielgröße von den Größen einer ANDEREN Anweisung ab? Generisch:
    deren Größen leicht verschieben (1e-3 des Bereichs) und die Zielgröße vergleichen.
    Dann ist das Ergebnis nur ein Gleichgewicht (keine Zielgröße allein besser),
    kein gemeinsames Optimum.
    """
    for goal in goals:
        others = [g for g in goals if g is not goal]
        shifted = dict(current)
        for other in others:
            for name, lo, hi in zip(other.names, other.lower, other.upper):
                u = _to_u(current[name], lo, hi)
                shifted[name] = _to_x(u - 1e-3 if u > 0.5 else u + 1e-3, lo, hi)
        ok, moved, _ = solve_system(equations, variables, warm, constants={**constants, **shifted},
                                    original_equations=original_equations)
        before, after = solution.get(goal.objective), moved.get(goal.objective) if ok else None
        if before is None or after is None:
            continue
        if abs(after - before) > 1e-9 * max(1.0, abs(before)):
            return True
    return False


def _result(success, solution, message, return_analysis):
    if return_analysis:
        return success, solution, message, None
    return success, solution, message


def optimize_parametric(
    equations: List[str],
    variables: Set[str],
    sweep_vars: Dict[str, np.ndarray],
    goals: list,
    initial_values: Optional[Dict[str, float]] = None,
    progress_callback=None,
    constants: Optional[Dict[str, float]] = None,
    original_equations: Optional[Dict[str, str]] = None,
):
    """
    Optimierung für jeden Punkt einer Parameterstudie (wie die Min/Max-Tabelle in
    EES). Das Optimum des vorigen Punkts ist zusätzlicher Kandidat des nächsten.
    """
    names = {n for g in goals for n in g.names}
    previous: Dict[str, float] = {}

    def point_solver(eqs, unknowns, point_initial, constants=None, original_equations=None):
        success, solution, message = optimize_system(eqs, unknowns, goals, point_initial, constants,
                                                     original_equations, hints=previous)
        if success:
            previous.clear()
            previous.update({n: solution[n] for n in names if n in solution})
        return success, solution, message

    return solve_parametric(equations, variables, sweep_vars, initial_values,
                            progress_callback=progress_callback, constants=constants,
                            original_equations=original_equations, point_solver=point_solver,
                            extra_result_vars=names)


if __name__ == "__main__":
    from parser import parse_equations, parse_optimization

    text = "y = (x - 2)^2 + 1\nMINIMIZE y VARY x = 0 .. 5"
    eqs, variables, consts, _, orig, _ = parse_equations(text)
    goals = parse_optimization(text)
    ok, sol, msg = optimize_system(eqs, variables - {'x'}, goals, {}, consts, orig)
    print(msg)
    print(f"x = {sol['x']:.6f}, y = {sol['y']:.6f}  (erwartet 2, 1)")
    assert ok and abs(sol['x'] - 2) < 1e-5
