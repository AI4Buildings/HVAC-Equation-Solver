"""
Gleichungslöser für gekoppelte lineare und nichtlineare Gleichungssysteme.

Verwendet scipy.optimize.fsolve (Newton-Raphson / Levenberg-Marquardt).
Unterstützt Parameterstudien (Sweeps) mit Vektor-Variablen.
"""

import numpy as np
from scipy.optimize import fsolve
from typing import List, Set, Dict, Tuple, Optional, Any, Union
from numpy import exp, log, log10, sqrt, pi
from numpy import sinh, cosh, tanh


# Trigonometrische Funktionen in GRAD (wie EES)
def sin(x):
    """Sinus mit Argument in Grad."""
    return np.sin(np.radians(x))

def cos(x):
    """Cosinus mit Argument in Grad."""
    return np.cos(np.radians(x))

def tan(x):
    """Tangens mit Argument in Grad."""
    return np.tan(np.radians(x))

def asin(x):
    """Arcussinus, Ergebnis in Grad."""
    return np.degrees(np.arcsin(x))

def acos(x):
    """Arcuscosinus, Ergebnis in Grad."""
    return np.degrees(np.arccos(x))

def atan(x):
    """Arcustangens, Ergebnis in Grad."""
    return np.degrees(np.arctan(x))

# Importiere Thermodynamik-Funktionen
try:
    from thermodynamics import THERMO_FUNCTIONS
    THERMO_AVAILABLE = True
except ImportError:
    THERMO_FUNCTIONS = {}
    THERMO_AVAILABLE = False

# Importiere Strahlungs-Funktionen
try:
    from radiation import RADIATION_FUNCTIONS
    RADIATION_AVAILABLE = True
except ImportError:
    RADIATION_FUNCTIONS = {}
    RADIATION_AVAILABLE = False

# Importiere Feuchte-Luft-Funktionen
try:
    from humid_air import HUMID_AIR_FUNCTIONS
    HUMID_AIR_AVAILABLE = True
except ImportError:
    HUMID_AIR_FUNCTIONS = {}
    HUMID_AIR_AVAILABLE = False


def create_equation_function(equations: List[str], variables: List[str],
                             constants: Dict[str, float] = None):
    """
    Erstellt eine Funktion f(x) die das Gleichungssystem darstellt.

    Args:
        equations: Liste von Gleichungen in Python-Syntax (als f(x) = 0)
        variables: Liste der Variablennamen (geordnet)
        constants: Dictionary mit konstanten Werten (direkte Zuweisungen)

    Returns:
        Eine Funktion die einen Vektor x nimmt und einen Vektor f(x) zurückgibt
    """
    if constants is None:
        constants = {}

    def equation_system(x):
        # Erstelle ein Dictionary mit Variablenwerten
        var_dict = {var: val for var, val in zip(variables, x)}

        # Füge Konstanten hinzu (überschreiben keine Unbekannten)
        var_dict.update(constants)

        # Füge mathematische Funktionen hinzu
        var_dict.update({
            'sin': sin, 'cos': cos, 'tan': tan,
            'asin': asin, 'acos': acos, 'atan': atan,
            'sinh': sinh, 'cosh': cosh, 'tanh': tanh,
            'exp': exp, 'log': log, 'log10': log10,
            'sqrt': sqrt, 'abs': np.abs, 'pi': pi
        })

        # Füge Thermodynamik-Funktionen hinzu
        if THERMO_AVAILABLE:
            var_dict.update(THERMO_FUNCTIONS)

        # Füge Strahlungs-Funktionen hinzu
        if RADIATION_AVAILABLE:
            var_dict.update(RADIATION_FUNCTIONS)

        # Füge Feuchte-Luft-Funktionen hinzu
        if HUMID_AIR_AVAILABLE:
            var_dict.update(HUMID_AIR_FUNCTIONS)

        # Evaluiere jede Gleichung
        results = []
        for eq in equations:
            try:
                result = eval(eq, {"__builtins__": {}}, var_dict)
                results.append(result)
            except Exception as e:
                raise ValueError(f"Fehler beim Auswerten von '{eq}': {e}")

        return np.array(results)

    return equation_system


def solve_system(
    equations: List[str],
    variables: Set[str],
    initial_values: Dict[str, float] = None,
    initial_guess: float = 1.0,
    constants: Dict[str, float] = None
) -> Tuple[bool, Dict[str, float], str]:
    """
    Löst das Gleichungssystem mit blockweiser Dekomposition.

    Strategie:
    1. Konstanten zuweisen (explizite Definitionen)
    2. Sequentielle Auswertung (direkte Zuweisungen, Einzelgleichungen)
    3. Blockweise Lösung (zusammenhängende Gleichungsblöcke)
    4. Nach jedem Block: Wiederhole Schritt 2-3

    Args:
        equations: Liste von Gleichungen in Python-Syntax
        variables: Set der Variablennamen
        initial_values: Dictionary mit Startwerten für den Solver
        initial_guess: Standardstartwert für unbekannte Variablen
        constants: Dictionary mit festen Werten (direkte Zuweisungen)

    Returns:
        success: True wenn Lösung gefunden
        solution: Dictionary mit Variablen und ihren Werten
        message: Status- oder Fehlermeldung
    """
    if initial_values is None:
        initial_values = {}
    if constants is None:
        constants = {}

    context = _get_eval_context()

    # Phase 1: Starte mit Konstanten
    known_values = constants.copy()
    remaining_equations = list(equations)
    remaining_vars = set(variables)

    # Sammle Statistiken für Meldung
    stats = {
        'direct': 0,      # Direkt ausgewertete Gleichungen
        'single': 0,      # Einzelne Unbekannte iterativ gelöst
        'blocks': [],     # Gelöste Blockgrößen
    }

    max_iterations = len(equations) * 3 + 1
    iteration = 0

    while remaining_equations and iteration < max_iterations:
        iteration += 1
        made_progress = False

        # Phase 2: Sequentielle Auswertung
        # 2a: Direkte Auswertung für Gleichungen der Form "(var) - (expr)"
        for eq in remaining_equations[:]:
            for var in list(remaining_vars):
                if eq.startswith(f"({var}) - ("):
                    expr = eq[len(f"({var}) - "):]
                    if expr.startswith("(") and expr.endswith(")"):
                        expr = expr[1:-1]

                    try:
                        local_context = context.copy()
                        local_context.update(known_values)
                        result = eval(expr, {"__builtins__": {}}, local_context)

                        if np.isfinite(result):
                            known_values[var] = float(result)
                            remaining_vars.discard(var)
                            remaining_equations.remove(eq)
                            stats['direct'] += 1
                            made_progress = True
                            break
                    except Exception:
                        pass

        # 2b: Gleichungen mit einer Unbekannten iterativ lösen
        if not made_progress:
            for eq in remaining_equations[:]:
                unknowns = _get_equation_unknowns(eq, set(known_values.keys()), remaining_vars)
                if len(unknowns) == 1:
                    unknown = list(unknowns)[0]
                    success, value = _solve_single_unknown(
                        eq, unknown, known_values, context, initial_values
                    )
                    if success:
                        known_values[unknown] = value
                        remaining_vars.discard(unknown)
                        remaining_equations.remove(eq)
                        stats['single'] += 1
                        made_progress = True
                        break

        # Phase 3: Blockweise Lösung
        if not made_progress and remaining_equations:
            # Finde zusammenhängende Blöcke
            blocks = _find_equation_blocks(remaining_equations, remaining_vars, set(known_values.keys()))

            if blocks:
                # Löse den kleinsten Block zuerst
                block_eqs, block_vars = blocks[0]

                # Prüfe ob Block quadratisch ist
                if len(block_eqs) == len(block_vars):
                    success, block_solution, block_msg = _solve_equation_block(
                        block_eqs, block_vars, known_values, context, initial_values
                    )

                    if success:
                        # Aktualisiere bekannte Werte
                        known_values.update(block_solution)
                        remaining_vars -= block_vars

                        # Entferne gelöste Gleichungen
                        for eq in block_eqs:
                            if eq in remaining_equations:
                                remaining_equations.remove(eq)

                        stats['blocks'].append(len(block_vars))
                        made_progress = True

        if not made_progress:
            # Keine weitere Fortschritte möglich
            break

    # Ergebnis zusammenstellen
    result = known_values.copy()

    # Erstelle Statusmeldung
    if not remaining_equations:
        parts = []
        if stats['direct'] > 0:
            parts.append(f"{stats['direct']} direkt")
        if stats['single'] > 0:
            parts.append(f"{stats['single']} iterativ")
        if stats['blocks']:
            block_info = '+'.join(str(b) for b in stats['blocks'])
            parts.append(f"Blöcke: {block_info}")

        msg = "Lösung gefunden"
        if parts:
            msg += f" ({', '.join(parts)})"
        return True, result, msg
    else:
        # Nicht alle Gleichungen gelöst
        msg = f"Unvollständig: {len(remaining_equations)} Gleichungen, {len(remaining_vars)} Unbekannte verbleibend"
        return False, result, msg


def format_solution(solution: Dict[str, Any], precision: int = 6) -> str:
    """Formatiert die Lösung für die Anzeige."""
    if not solution:
        return "Keine Lösung"

    lines = []
    for var in sorted(solution.keys()):
        val = solution[var]

        # Prüfe ob es ein Array ist
        if isinstance(val, np.ndarray):
            if len(val) <= 5:
                arr_str = ', '.join(f'{v:.{precision}g}' for v in val)
                lines.append(f"{var} = [{arr_str}]")
            else:
                first = f'{val[0]:.{precision}g}'
                last = f'{val[-1]:.{precision}g}'
                lines.append(f"{var} = [{first}, ..., {last}] ({len(val)} Werte)")
        else:
            # Skalarer Wert
            if abs(val) < 1e-10:
                val = 0.0
            if abs(val) >= 1e6 or (abs(val) < 1e-4 and val != 0):
                lines.append(f"{var} = {val:.{precision}e}")
            else:
                lines.append(f"{var} = {val:.{precision}g}")

    return "\n".join(lines)


def create_equation_function_with_sweep(
    equations: List[str],
    variables: List[str],
    sweep_values: Dict[str, float],
    constants: Dict[str, float] = None
):
    """
    Erstellt eine Gleichungsfunktion mit fest eingesetzten Sweep-Werten.

    Args:
        equations: Liste von Gleichungen
        variables: Liste der zu lösenden Variablen
        sweep_values: Dict mit aktuellen Sweep-Werten {name: value}
        constants: Dict mit Konstanten (direkte Zuweisungen)
    """
    if constants is None:
        constants = {}

    def equation_system(x):
        # Erstelle ein Dictionary mit Variablenwerten
        var_dict = {var: val for var, val in zip(variables, x)}

        # Füge Konstanten hinzu
        var_dict.update(constants)

        # Füge Sweep-Werte hinzu
        var_dict.update(sweep_values)

        # Füge mathematische Funktionen hinzu
        var_dict.update({
            'sin': sin, 'cos': cos, 'tan': tan,
            'asin': asin, 'acos': acos, 'atan': atan,
            'sinh': sinh, 'cosh': cosh, 'tanh': tanh,
            'exp': exp, 'log': log, 'log10': log10,
            'sqrt': sqrt, 'abs': np.abs, 'pi': pi
        })

        # Füge Thermodynamik-Funktionen hinzu
        if THERMO_AVAILABLE:
            var_dict.update(THERMO_FUNCTIONS)

        # Füge Strahlungs-Funktionen hinzu
        if RADIATION_AVAILABLE:
            var_dict.update(RADIATION_FUNCTIONS)

        # Füge Feuchte-Luft-Funktionen hinzu
        if HUMID_AIR_AVAILABLE:
            var_dict.update(HUMID_AIR_FUNCTIONS)

        # Evaluiere jede Gleichung
        results = []
        for eq in equations:
            try:
                result = eval(eq, {"__builtins__": {}}, var_dict)
                results.append(result)
            except Exception as e:
                raise ValueError(f"Fehler beim Auswerten von '{eq}': {e}")

        return np.array(results)

    return equation_system


def _get_eval_context():
    """Erstellt den Kontext für eval mit allen verfügbaren Funktionen."""
    context = {
        'sin': sin, 'cos': cos, 'tan': tan,
        'asin': asin, 'acos': acos, 'atan': atan,
        'sinh': sinh, 'cosh': cosh, 'tanh': tanh,
        'exp': exp, 'log': log, 'log10': log10,
        'sqrt': sqrt, 'abs': np.abs, 'pi': pi
    }
    if THERMO_AVAILABLE:
        context.update(THERMO_FUNCTIONS)
    if RADIATION_AVAILABLE:
        context.update(RADIATION_FUNCTIONS)
    if HUMID_AIR_AVAILABLE:
        context.update(HUMID_AIR_FUNCTIONS)
    return context


def _get_equation_unknowns(equation: str, known_vars: Set[str], all_vars: Set[str]) -> Set[str]:
    """Findet die Unbekannten in einer Gleichung."""
    import re
    found_vars = set(re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', equation))
    # Filtere auf die tatsächlichen Variablen
    return (found_vars & all_vars) - known_vars


def _find_minimal_coupled_core(
    equations: List[str],
    variables: Set[str],
    known_vars: Set[str]
) -> Tuple[List[str], Set[str]]:
    """
    Findet den minimalen gekoppelten Kern eines Gleichungsblocks.

    Der Kern ist die kleinste Menge von Gleichungen und Variablen die:
    1. Quadratisch sind (gleich viele Gleichungen wie Variablen)
    2. Unabhängig lösbar sind (alle externen Abhängigkeiten sind bekannt)

    Verwendet einen Bottom-Up Ansatz: Starte mit kleinen Gruppen und
    expandiere nur wenn nötig.

    Returns:
        (equations, variables) für den minimalen unabhängig lösbaren Kern
    """
    # Mapping: Gleichung -> ihre Unbekannten
    eq_to_unknowns = {}
    for eq in equations:
        unknowns = _get_equation_unknowns(eq, known_vars, variables)
        eq_to_unknowns[eq] = unknowns

    # Mapping: Variable -> Gleichungen in denen sie vorkommt
    var_to_eqs = {}
    for eq, unknowns in eq_to_unknowns.items():
        for var in unknowns:
            if var not in var_to_eqs:
                var_to_eqs[var] = []
            var_to_eqs[var].append(eq)

    def is_independently_solvable(block_eqs: List[str], block_vars: Set[str]) -> bool:
        """Prüft ob ein Block unabhängig lösbar ist."""
        if len(block_eqs) != len(block_vars):
            return False
        # Alle Variablen in den Gleichungen müssen entweder im Block oder bekannt sein
        for eq in block_eqs:
            for var in eq_to_unknowns[eq]:
                if var not in block_vars and var not in known_vars:
                    return False
        return True

    def expand_to_closed_block(start_eqs: List[str]) -> Tuple[List[str], Set[str]]:
        """
        Expandiert eine Startmenge von Gleichungen zu einem geschlossenen Block.

        Ein geschlossener Block hat gleich viele Gleichungen wie Unbekannte
        (quadratischer Block).

        Strategie: Bei der Wahl neuer Gleichungen werden solche bevorzugt,
        die möglichst WENIGE neue Variablen einführen. Dies führt zum
        minimalen gekoppelten Kern.
        """
        block_eqs = list(start_eqs)
        block_vars = set()
        for eq in block_eqs:
            block_vars.update(eq_to_unknowns[eq])

        # Iterativ expandieren bis Block quadratisch ist
        max_iter = len(equations) * 2
        for _ in range(max_iter):
            # Prüfe ob Block quadratisch ist
            if len(block_eqs) == len(block_vars):
                break

            # Mehr Variablen als Gleichungen - füge Gleichungen hinzu
            if len(block_vars) > len(block_eqs):
                # Finde die beste Gleichung zum Hinzufügen:
                # Bevorzuge Gleichungen die KEINE oder WENIGE neue Variablen einführen
                best_eq = None
                best_new_vars = float('inf')

                for var in block_vars:
                    if var in known_vars:
                        continue
                    for eq in var_to_eqs.get(var, []):
                        if eq not in block_eqs:
                            # Zähle wie viele NEUE Variablen diese Gleichung einführt
                            eq_vars = eq_to_unknowns[eq]
                            new_vars = len(eq_vars - block_vars)
                            if new_vars < best_new_vars:
                                best_new_vars = new_vars
                                best_eq = eq
                                # Wenn 0 neue Variablen, sofort nehmen
                                if new_vars == 0:
                                    break
                    if best_new_vars == 0:
                        break

                if best_eq is not None:
                    block_eqs.append(best_eq)
                    block_vars.update(eq_to_unknowns[best_eq])
                else:
                    # Keine weiteren Gleichungen verfügbar
                    break
            else:
                # Mehr Gleichungen als Variablen - sollte nicht passieren
                break

        return block_eqs, block_vars

    # Strategie: Suche nach dem kleinsten unabhängig lösbaren Block
    # Sortiere Gleichungen nach Anzahl der Unbekannten (weniger = einfacher)
    sorted_eqs = sorted(equations, key=lambda eq: len(eq_to_unknowns[eq]))

    best_block = None
    best_size = float('inf')

    # Versuche verschiedene Startpunkte
    for start_eq in sorted_eqs:
        block_eqs, block_vars = expand_to_closed_block([start_eq])

        # Prüfe ob Block quadratisch ist
        if len(block_eqs) != len(block_vars):
            continue

        # Prüfe ob Block unabhängig lösbar ist
        if not is_independently_solvable(block_eqs, block_vars):
            continue

        # Ist dieser Block kleiner als der beste bisherige?
        if len(block_vars) < best_size:
            best_block = (block_eqs, block_vars)
            best_size = len(block_vars)

            # Wenn wir einen kleinen Block gefunden haben, nutze ihn
            if best_size <= 3:
                break

    if best_block:
        return best_block

    # Fallback: Versuche alle Gleichungen als einen Block
    all_vars = set()
    for eq in equations:
        all_vars.update(eq_to_unknowns[eq])

    if len(equations) == len(all_vars):
        return equations, all_vars

    # Wenn nichts funktioniert, gib den ursprünglichen Block zurück
    return equations, variables


def _find_equation_blocks(
    equations: List[str],
    variables: Set[str],
    known_vars: Set[str]
) -> List[Tuple[List[str], Set[str]]]:
    """
    Findet zusammenhängende Blöcke von Gleichungen.

    Ein Block ist eine Menge von Gleichungen, die gemeinsame Unbekannte teilen
    und daher zusammen gelöst werden müssen.

    Returns:
        Liste von (equations, variables) Tupeln für jeden Block,
        sortiert nach Blockgröße (kleinste zuerst)
    """
    if not equations:
        return []

    # Erstelle Mapping: Gleichung -> Unbekannte
    eq_to_vars = {}
    for eq in equations:
        unknowns = _get_equation_unknowns(eq, known_vars, variables)
        if unknowns:  # Nur Gleichungen mit Unbekannten
            eq_to_vars[eq] = unknowns

    if not eq_to_vars:
        return []

    # Union-Find Struktur für Gruppierung
    # Gruppiere Gleichungen die gemeinsame Variablen haben
    remaining_eqs = set(eq_to_vars.keys())
    blocks = []

    while remaining_eqs:
        # Starte mit einer beliebigen Gleichung
        current_block_eqs = set()
        current_block_vars = set()

        # Nimm erste verfügbare Gleichung
        start_eq = next(iter(remaining_eqs))
        to_process = [start_eq]

        while to_process:
            eq = to_process.pop()
            if eq in current_block_eqs:
                continue

            current_block_eqs.add(eq)
            eq_vars = eq_to_vars.get(eq, set())
            new_vars = eq_vars - current_block_vars
            current_block_vars.update(eq_vars)

            # Finde alle anderen Gleichungen die diese Variablen verwenden
            if new_vars:
                for other_eq in remaining_eqs - current_block_eqs:
                    other_vars = eq_to_vars.get(other_eq, set())
                    if other_vars & new_vars:  # Gemeinsame Variablen
                        to_process.append(other_eq)

        # Block gefunden
        remaining_eqs -= current_block_eqs
        blocks.append((list(current_block_eqs), current_block_vars))

    # Sortiere nach Blockgröße (kleinste zuerst für bessere Konvergenz)
    blocks.sort(key=lambda b: len(b[1]))

    return blocks


def _solve_equation_block(
    equations: List[str],
    variables: Set[str],
    known_values: Dict[str, float],
    context: dict,
    manual_initial: Dict[str, float] = None
) -> Tuple[bool, Dict[str, float], str]:
    """
    Löst einen Block von Gleichungen mit gemeinsamen Unbekannten.

    Bei größeren Blöcken wird zuerst versucht, den Block iterativ zu zerlegen
    und kleinere Sub-Blöcke sequentiell zu lösen.

    Args:
        manual_initial: Manuelle Startwerte (haben Priorität)

    Returns:
        (success, solution_dict, message)
    """
    if not equations or not variables:
        return True, {}, "Leerer Block"

    n_vars = len(variables)
    n_eqs = len(equations)

    if n_eqs != n_vars:
        return False, {}, f"Block nicht quadratisch: {n_eqs} Gleichungen, {n_vars} Unbekannte"

    # Bei größeren Blöcken: Versuche iterative Zerlegung
    if n_vars > 3:
        success, solution, msg = _solve_block_iteratively(
            equations, variables, known_values, context, manual_initial
        )
        if success:
            return success, solution, msg

    # Fallback: Löse den gesamten Block simultan
    return _solve_block_simultaneously(
        equations, variables, known_values, context, manual_initial
    )


def _solve_block_iteratively(
    equations: List[str],
    variables: Set[str],
    known_values: Dict[str, float],
    context: dict,
    manual_initial: Dict[str, float] = None
) -> Tuple[bool, Dict[str, float], str]:
    """
    Versucht einen Block iterativ zu lösen, indem nach jeder gelösten
    Gleichung geprüft wird, ob weitere Gleichungen direkt oder mit
    nur einer Unbekannten lösbar sind.
    """
    remaining_eqs = list(equations)
    remaining_vars = set(variables)
    local_known = known_values.copy()
    solved_values = {}
    stats = {'direct': 0, 'single': 0, 'subblocks': []}

    max_iterations = len(equations) * 2 + 1
    iteration = 0

    while remaining_eqs and iteration < max_iterations:
        iteration += 1
        made_progress = False

        # Phase 1: Direkte Auswertung für Gleichungen der Form "(var) - (expr)"
        for eq in remaining_eqs[:]:
            for var in list(remaining_vars):
                if eq.startswith(f"({var}) - ("):
                    expr = eq[len(f"({var}) - "):]
                    if expr.startswith("(") and expr.endswith(")"):
                        expr = expr[1:-1]

                    try:
                        local_context = context.copy()
                        local_context.update(local_known)
                        result = eval(expr, {"__builtins__": {}}, local_context)

                        if np.isfinite(result):
                            solved_values[var] = float(result)
                            local_known[var] = float(result)
                            remaining_vars.discard(var)
                            remaining_eqs.remove(eq)
                            stats['direct'] += 1
                            made_progress = True
                            break
                    except Exception:
                        pass

        if made_progress:
            continue

        # Phase 2: Gleichungen mit einer Unbekannten iterativ lösen
        for eq in remaining_eqs[:]:
            unknowns = _get_equation_unknowns(eq, set(local_known.keys()), remaining_vars)
            if len(unknowns) == 1:
                unknown = list(unknowns)[0]
                success, value = _solve_single_unknown(
                    eq, unknown, local_known, context, manual_initial
                )
                if success:
                    solved_values[unknown] = value
                    local_known[unknown] = value
                    remaining_vars.discard(unknown)
                    remaining_eqs.remove(eq)
                    stats['single'] += 1
                    made_progress = True
                    break

        if made_progress:
            continue

        # Phase 3: Finde und löse den minimalen gekoppelten Kern
        if remaining_eqs:
            core_eqs, core_vars = _find_minimal_coupled_core(
                remaining_eqs, remaining_vars, set(local_known.keys())
            )

            if core_vars and len(core_eqs) == len(core_vars):
                success, sub_solution, _ = _solve_block_simultaneously(
                    core_eqs, core_vars, local_known, context, manual_initial
                )

                if success:
                    solved_values.update(sub_solution)
                    local_known.update(sub_solution)
                    remaining_vars -= core_vars
                    for eq in core_eqs:
                        if eq in remaining_eqs:
                            remaining_eqs.remove(eq)
                    stats['subblocks'].append(len(core_vars))
                    made_progress = True

        if not made_progress:
            break

    if not remaining_eqs:
        msg_parts = []
        if stats['direct'] > 0:
            msg_parts.append(f"{stats['direct']} direkt")
        if stats['single'] > 0:
            msg_parts.append(f"{stats['single']} iterativ")
        if stats['subblocks']:
            msg_parts.append(f"Sub-Blöcke: {'+'.join(str(b) for b in stats['subblocks'])}")
        return True, solved_values, f"Block zerlegt ({', '.join(msg_parts)})"

    return False, solved_values, f"Iterative Zerlegung unvollständig"


def _solve_block_simultaneously(
    equations: List[str],
    variables: Set[str],
    known_values: Dict[str, float],
    context: dict,
    manual_initial: Dict[str, float] = None
) -> Tuple[bool, Dict[str, float], str]:
    """
    Löst einen Block von Gleichungen simultan mit fsolve.
    """
    if not equations or not variables:
        return True, {}, "Leerer Block"

    var_list = sorted(list(variables))
    n_vars = len(var_list)
    n_eqs = len(equations)

    if n_eqs != n_vars:
        return False, {}, f"Block nicht quadratisch: {n_eqs} Gleichungen, {n_vars} Unbekannte"

    # Erstelle Startvektor (manuelle Werte haben Priorität, sonst 1.0)
    x0 = []
    for var in var_list:
        x0.append(_get_initial_value(var, manual_initial))
    x0 = np.array(x0, dtype=float)

    # Erstelle Gleichungsfunktion
    def block_func(x):
        local_ctx = context.copy()
        local_ctx.update(known_values)
        local_ctx.update({var: val for var, val in zip(var_list, x)})

        results = []
        for eq in equations:
            try:
                result = eval(eq, {"__builtins__": {}}, local_ctx)
                results.append(result)
            except Exception:
                results.append(float('inf'))
        return np.array(results)

    # Versuche mehrere Startwerte
    best_solution = None
    best_residual = float('inf')

    # Generiere Variationen der Startwerte
    start_variations = [x0]
    for i in range(n_vars):
        variation = x0.copy()
        variation[i] *= 1.5
        start_variations.append(variation)
        variation = x0.copy()
        variation[i] *= 0.5
        start_variations.append(variation)

    import warnings
    for x_start in start_variations[:10]:  # Maximal 10 Versuche
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                solution, info, ier, _ = fsolve(block_func, x_start, full_output=True)

            residual = np.max(np.abs(info['fvec']))

            if ier == 1 and np.all(np.isfinite(solution)) and residual < 1e-8:
                result_dict = {var: val for var, val in zip(var_list, solution)}
                return True, result_dict, f"Block gelöst ({n_vars} Variablen)"
            elif np.all(np.isfinite(solution)) and residual < best_residual:
                best_solution = solution
                best_residual = residual
        except Exception:
            pass

    # Akzeptiere gute Näherung
    if best_solution is not None and best_residual < 1e-4:
        result_dict = {var: val for var, val in zip(var_list, best_solution)}
        return True, result_dict, f"Block gelöst (Residuum: {best_residual:.2e})"

    return False, {}, f"Block-Konvergenz fehlgeschlagen (Residuum: {best_residual:.2e})"


def _get_initial_value(var: str, manual_initial: Dict[str, float] = None) -> float:
    """Ermittelt den Startwert für eine Variable.

    Manuelle Startwerte haben Priorität, sonst wird 1.0 verwendet.
    """
    if manual_initial and var in manual_initial:
        return manual_initial[var]
    return 1.0


def _solve_single_unknown(equation: str, unknown: str, known_values: Dict[str, float],
                           context: dict, manual_initial: Dict[str, float] = None) -> Tuple[bool, float]:
    """
    Löst eine Gleichung mit einer einzelnen Unbekannten.

    Strategie:
    1. Schneller Newton-Raphson Versuch (für einfache Gleichungen)
    2. Falls nötig: Robuste Bracket-Suche mit Brent's Methode

    Returns:
        (success, value)
    """
    from scipy.optimize import brentq

    def func(x):
        local_ctx = context.copy()
        local_ctx.update(known_values)
        local_ctx[unknown] = x
        try:
            return eval(equation, {"__builtins__": {}}, local_ctx)
        except Exception:
            return float('inf')

    # === Phase 1: Schneller Newton-Raphson Versuch ===
    # Für einfache (oft lineare) Gleichungen konvergiert dies in wenigen Iterationen
    initial_guess = 1.0
    if manual_initial and unknown in manual_initial:
        initial_guess = manual_initial[unknown]

    try:
        x = initial_guess
        h = 1e-8  # Schrittweite für numerische Ableitung

        for iteration in range(30):  # Max 30 Iterationen
            fx = func(x)

            # Prüfe ob bereits Lösung gefunden
            if abs(fx) < 1e-10:
                return True, x

            # Numerische Ableitung
            fx_plus = func(x + h)
            fx_minus = func(x - h)
            dfx = (fx_plus - fx_minus) / (2 * h)

            # Prüfe ob Ableitung gültig
            if not np.isfinite(dfx) or abs(dfx) < 1e-15:
                break  # Newton funktioniert nicht, verwende Bracket-Suche

            # Newton-Schritt
            x_new = x - fx / dfx

            # Prüfe Konvergenz
            if abs(x_new - x) < 1e-10 * max(1, abs(x)):
                # Verifiziere Lösung
                if abs(func(x_new)) < 1e-8:
                    return True, x_new
                break

            # Dämpfung für große Schritte (verhindert Oszillation)
            if abs(x_new - x) > 100 * max(1, abs(x)):
                x = x + 0.5 * (x_new - x)  # Halber Schritt
            else:
                x = x_new

            # Prüfe ob Wert noch vernünftig
            if not np.isfinite(x) or abs(x) > 1e15:
                break

    except Exception:
        pass  # Newton fehlgeschlagen, verwende Bracket-Suche

    # === Phase 2: Robuste Bracket-Suche (Fallback) ===
    # Erzeuge Testpunkte mit dichter Abdeckung
    test_points = set()

    # Logarithmische Skalierung für extreme Bereiche
    for exp in range(-2, 8):
        test_points.update([10**exp, -10**exp, 0.5 * 10**exp, 2 * 10**exp])

    # Dichte Abdeckung (0-500 in 1er Schritten, 500-1000 in 10er Schritten)
    test_points.update(range(0, 501, 1))
    test_points.update(range(-500, 0, 1))
    test_points.update(range(500, 1001, 10))
    test_points.update(range(-1000, -500, 10))

    # Feinere Abdeckung im Bereich 0-10
    test_points.update([i * 0.1 for i in range(-100, 101)])

    # Gröbere Abdeckung für größere Werte
    test_points.update(range(1000, 10001, 100))
    test_points.update(range(-10000, -1000, 100))

    test_points = sorted(test_points)

    def eval_points(points):
        """Evaluiere Funktion an Punkten und gib gültige (x, f(x)) Paare zurück."""
        results = []
        for x in points:
            try:
                v = func(x)
                if np.isfinite(v) and abs(v) < 1e20:
                    results.append((x, v))
            except Exception:
                pass
        return sorted(results, key=lambda p: p[0])

    def find_brackets(points):
        """Finde Intervalle mit Vorzeichenwechsel."""
        brackets = []
        for i in range(len(points) - 1):
            x1, v1 = points[i]
            x2, v2 = points[i + 1]
            if v1 * v2 < 0:
                brackets.append((x1, x2))
        return brackets

    def refine_interval(x1, x2, depth=0):
        """Verfeinere ein Intervall adaptiv um Singularitäten zu finden."""
        if depth > 5 or abs(x2 - x1) < 1e-6:
            return []

        # Teste Mittelpunkt und weitere Punkte
        mid = (x1 + x2) / 2
        new_points = [x1 + (x2 - x1) * i / 10 for i in range(11)]
        evaluated = eval_points(new_points)
        brackets = find_brackets(evaluated)

        if brackets:
            return brackets

        # Rekursiv verfeinern wenn große Wertänderung
        if len(evaluated) >= 2:
            for i in range(len(evaluated) - 1):
                px1, pv1 = evaluated[i]
                px2, pv2 = evaluated[i + 1]
                if abs(pv2 - pv1) > 1.0:  # Große Änderung deutet auf Singularität
                    sub_brackets = refine_interval(px1, px2, depth + 1)
                    if sub_brackets:
                        return sub_brackets
        return []

    # Erste Auswertung
    valid_points = eval_points(test_points)
    brackets = find_brackets(valid_points)

    # Wenn keine Brackets gefunden, suche nach Regionen mit großen Änderungen
    if not brackets:
        # Sortiere nach Größe der Änderung (größte zuerst)
        changes = []
        for i in range(len(valid_points) - 1):
            x1, v1 = valid_points[i]
            x2, v2 = valid_points[i + 1]
            change = abs(v2 - v1)
            if change > 0.1:  # Schon moderate Änderungen untersuchen
                changes.append((change, x1, x2))
        changes.sort(reverse=True)

        for _, x1, x2 in changes[:20]:  # Top 20 Regionen untersuchen
            brackets = refine_interval(x1, x2)
            if brackets:
                break

    # Versuche alle gefundenen Brackets
    best_root = None
    best_residual = float('inf')

    for a, b in brackets:
        try:
            root = brentq(func, a, b, xtol=1e-12, rtol=1e-12)
            residual = abs(func(root))
            if np.isfinite(root) and np.isfinite(residual):
                if residual < 1e-6:  # Gute Lösung gefunden
                    return True, float(root)
                elif residual < best_residual:
                    best_root = root
                    best_residual = residual
        except Exception:
            pass

    # Beste gefundene Lösung zurückgeben wenn akzeptabel
    if best_root is not None and best_residual < 1e-4:
        return True, float(best_root)

    # Fallback: fsolve mit Standard-Startwert 1.0
    import warnings
    for x0 in [1.0, 0.1, 10.0, 100.0]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                solution, info, ier, _ = fsolve(func, x0, full_output=True)
            residual = abs(info['fvec'][0])
            if ier == 1 and np.isfinite(solution[0]) and residual < 1e-8:
                return True, float(solution[0])
        except Exception:
            pass

    return False, 0.0


def _try_sequential_evaluation(
    equations: List[str],
    variables: Set[str],
    constants: Dict[str, float]
) -> Tuple[List[str], Set[str], Dict[str, float]]:
    """
    Versucht einfache Zuweisungen und Gleichungen mit einer Unbekannten sequentiell auszuwerten.

    1. Gleichungen der Form "var = ausdruck" werden direkt ausgewertet
    2. Gleichungen mit nur einer Unbekannten werden mit fsolve gelöst

    Returns:
        remaining_equations: Verbleibende Gleichungen für den iterativen Solver
        remaining_variables: Verbleibende Unbekannte
        computed_values: Dictionary mit berechneten Werten
    """
    context = _get_eval_context()

    # Starte mit Konstanten als bekannte Werte
    known_values = constants.copy()

    remaining_equations = list(equations)
    remaining_vars = set(variables)
    computed_values = {}

    max_iterations = len(equations) * 2 + 1
    iteration = 0

    while remaining_equations and iteration < max_iterations:
        iteration += 1
        made_progress = False

        for eq in remaining_equations[:]:  # Kopie für Iteration
            # Phase 1: Versuche direkte Auswertung für Gleichungen der Form "(var) - (expr)"
            for var in list(remaining_vars):
                if eq.startswith(f"({var}) - ("):
                    expr = eq[len(f"({var}) - "):]
                    if expr.startswith("(") and expr.endswith(")"):
                        expr = expr[1:-1]

                    try:
                        local_context = context.copy()
                        local_context.update(known_values)

                        result = eval(expr, {"__builtins__": {}}, local_context)

                        if np.isfinite(result):
                            computed_values[var] = float(result)
                            known_values[var] = float(result)
                            remaining_vars.discard(var)
                            remaining_equations.remove(eq)
                            made_progress = True
                            break
                    except Exception:
                        # Diese Variable kann noch nicht berechnet werden
                        pass

            # Phase 2: Wenn keine direkte Auswertung möglich, versuche Gleichungen
            # mit nur einer Unbekannten iterativ zu lösen
            if not made_progress and eq in remaining_equations:
                unknowns = _get_equation_unknowns(eq, set(known_values.keys()), remaining_vars)
                if len(unknowns) == 1:
                    unknown = list(unknowns)[0]
                    success, value = _solve_single_unknown(eq, unknown, known_values, context)
                    if success:
                        computed_values[unknown] = value
                        known_values[unknown] = value
                        remaining_vars.discard(unknown)
                        remaining_equations.remove(eq)
                        made_progress = True

        if not made_progress:
            break

    return remaining_equations, remaining_vars, computed_values


def _try_vectorized_evaluation(equations: List[str], variables: Set[str],
                                sweep_vars: Dict[str, np.ndarray],
                                initial_values: Dict[str, float]) -> Tuple[bool, Dict[str, np.ndarray], str]:
    """
    Versucht direkte vektorisierte Auswertung für einfache Zuweisungen.

    Funktioniert wenn alle Gleichungen die Form "var = ausdruck" haben,
    wobei der Ausdruck nur von Sweep-Variablen und bereits berechneten Variablen abhängt.
    """
    n_points = len(list(sweep_vars.values())[0])
    results = {name: arr.copy() for name, arr in sweep_vars.items()}

    # Füge initiale Werte als Arrays hinzu
    for var, val in initial_values.items():
        if var in variables:
            results[var] = np.full(n_points, val)

    # Kontext für Auswertung
    context = _get_eval_context()

    # Versuche Gleichungen der Reihe nach auszuwerten
    # Gleichungen haben die Form "(left) - (right)" -> wir müssen sie umformen
    remaining_equations = list(equations)
    remaining_vars = set(variables)
    max_iterations = len(equations) + 1
    iteration = 0

    while remaining_equations and iteration < max_iterations:
        iteration += 1
        made_progress = False

        for eq in remaining_equations[:]:  # Kopie für Iteration
            # Versuche Gleichung zu parsen: "(var) - (expr)" oder "(expr) - (var)"
            # Vereinfacht: suche nach Variablen die wir berechnen können

            for var in list(remaining_vars):
                # Prüfe ob diese Variable berechnet werden kann
                # Die Gleichung sollte die Form "(var) - (something)" haben
                if eq.startswith(f"({var}) - ("):
                    # Extrahiere den Ausdruck auf der rechten Seite
                    expr = eq[len(f"({var}) - "):]
                    if expr.startswith("(") and expr.endswith(")"):
                        expr = expr[1:-1]

                    # Prüfe ob alle benötigten Variablen verfügbar sind
                    try:
                        # Erstelle lokalen Kontext mit aktuellen Ergebnissen
                        local_context = context.copy()
                        local_context.update(results)

                        # Evaluiere vektorisiert
                        result = eval(expr, {"__builtins__": {}}, local_context)

                        # Konvertiere zu Array falls nötig
                        if np.isscalar(result):
                            result = np.full(n_points, result)
                        elif isinstance(result, np.ndarray) and result.shape == ():
                            result = np.full(n_points, float(result))

                        results[var] = np.asarray(result)
                        remaining_vars.discard(var)
                        remaining_equations.remove(eq)
                        made_progress = True
                        break
                    except Exception:
                        # Diese Variable kann noch nicht berechnet werden
                        pass

        if not made_progress:
            # Keine weitere direkte Auswertung möglich
            break

    if not remaining_equations:
        return True, results, f"Vektorisierte Berechnung: {n_points} Punkte"
    else:
        return False, {}, "Vektorisierte Auswertung nicht möglich"


def solve_parametric(
    equations: List[str],
    variables: Set[str],
    sweep_vars: Dict[str, np.ndarray],
    initial_values: Dict[str, float] = None,
    progress_callback=None,
    constants: Dict[str, float] = None
) -> Tuple[bool, Dict[str, Union[float, np.ndarray]], str]:
    """
    Löst das Gleichungssystem für jeden Wert der Sweep-Variablen.

    Args:
        equations: Liste von Gleichungen in Python-Syntax
        variables: Set der Variablennamen (ohne Sweep-Variablen)
        sweep_vars: Dict mit Sweep-Variablen {name: numpy.array}
        initial_values: Dictionary mit Startwerten für den Solver
        progress_callback: Optional callback(current, total) für Fortschritt
        constants: Dictionary mit festen Werten (direkte Zuweisungen)

    Returns:
        success: True wenn alle Lösungen gefunden
        solution: Dictionary mit Variablen (Skalare oder Arrays)
        message: Status- oder Fehlermeldung
    """
    if not sweep_vars:
        # Keine Sweep-Variablen -> normale Lösung
        return solve_system(equations, variables, initial_values, constants=constants)

    if initial_values is None:
        initial_values = {}
    if constants is None:
        constants = {}

    # Versuche zuerst vektorisierte direkte Auswertung
    success, results, msg = _try_vectorized_evaluation(equations, variables, sweep_vars, initial_values)
    if success:
        return True, results, msg

    # Bestimme die Länge des Sweeps (alle Sweep-Variablen müssen gleich lang sein)
    sweep_lengths = [len(arr) for arr in sweep_vars.values()]
    if len(set(sweep_lengths)) > 1:
        return False, {}, f"Alle Sweep-Variablen müssen gleich lang sein. Gefunden: {sweep_lengths}"

    n_points = sweep_lengths[0]

    # Sortiere die zu lösenden Variablen
    var_list = sorted(list(variables))

    # Initialisiere Ergebnis-Arrays
    results = {var: np.zeros(n_points) for var in var_list}
    # Füge auch die Sweep-Variablen zum Ergebnis hinzu
    for name, arr in sweep_vars.items():
        results[name] = arr.copy()

    failed_points = []

    # Löse für jeden Sweep-Punkt mit der robusten solve_system Methode
    for i in range(n_points):
        # Setze aktuelle Sweep-Werte als Konstanten
        sweep_values = {name: float(arr[i]) for name, arr in sweep_vars.items()}

        # Kombiniere Konstanten mit Sweep-Werten
        combined_constants = constants.copy()
        combined_constants.update(sweep_values)

        try:
            # Verwende solve_system für jeden Punkt (nutzt Block-Dekomposition und Bracket-Suche)
            success, solution, msg = solve_system(
                equations, variables, initial_values, constants=combined_constants
            )

            if success:
                # Erfolg - speichere Lösung
                for var in var_list:
                    if var in solution:
                        results[var][i] = solution[var]
                    else:
                        results[var][i] = np.nan
            else:
                failed_points.append(i)
                for var in var_list:
                    results[var][i] = np.nan
        except Exception as e:
            failed_points.append(i)
            for var in var_list:
                results[var][i] = np.nan

        # Fortschritts-Callback
        if progress_callback:
            progress_callback(i + 1, n_points)

    # Füge Konstanten zum Ergebnis hinzu
    results.update(constants)

    # Zusammenfassung
    if not failed_points:
        msg = f"Parameterstudie erfolgreich: {n_points} Punkte berechnet"
        return True, results, msg
    elif len(failed_points) < n_points:
        msg = f"Parameterstudie teilweise erfolgreich: {n_points - len(failed_points)}/{n_points} Punkte berechnet"
        return True, results, msg
    else:
        return False, results, "Parameterstudie fehlgeschlagen: Keine Konvergenz"


if __name__ == "__main__":
    # Test: Lineares System
    print("Test 1: Lineares System")
    print("x + y = 10")
    print("x - y = 2")

    equations = ["(x) + (y) - (10)", "(x) - (y) - (2)"]
    variables = {'x', 'y'}

    success, solution, msg = solve_system(equations, variables)
    print(f"Erfolg: {success}")
    print(f"Nachricht: {msg}")
    print(format_solution(solution))
    print()

    # Test: Nichtlineares System
    print("Test 2: Nichtlineares System")
    print("x^2 + y^2 = 25")
    print("x * y = 12")

    equations = ["(x**2) + (y**2) - (25)", "(x) * (y) - (12)"]
    variables = {'x', 'y'}

    success, solution, msg = solve_system(equations, variables)
    print(f"Erfolg: {success}")
    print(f"Nachricht: {msg}")
    print(format_solution(solution))
