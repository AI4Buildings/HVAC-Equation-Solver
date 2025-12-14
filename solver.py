"""
Gleichungslöser für gekoppelte lineare und nichtlineare Gleichungssysteme.

Verwendet scipy.optimize.fsolve (Newton-Raphson / Levenberg-Marquardt).
Unterstützt Parameterstudien (Sweeps) mit Vektor-Variablen.
"""

import numpy as np
from scipy.optimize import fsolve
from typing import List, Set, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from numpy import exp, log, log10, sqrt, pi
from numpy import sinh, cosh, tanh

# Versuche Einheiten-Modul zu laden für unit-basierte Startwerte
try:
    from units import get_initial_from_unit
    UNITS_AVAILABLE = True
except ImportError:
    UNITS_AVAILABLE = False
    def get_initial_from_unit(unit_str):
        return 1.0


# ============================================================================
# Analysis Data Structures
# ============================================================================

@dataclass
class EquationInfo:
    """Information über eine einzelne Gleichung."""
    original: str           # Original-Gleichung (wie eingegeben)
    parsed: str             # Geparste Gleichung (Python-Syntax)
    variable: str           # Berechnete Variable
    value: float            # Berechneter Wert
    residual: float         # Residuum (sollte ~0 sein)
    category: str           # "constant", "direct", "single_unknown"


@dataclass
class BlockInfo:
    """Information über einen gekoppelten Block."""
    equations: List[str]        # Original-Gleichungen
    parsed_equations: List[str] # Geparste Gleichungen
    variables: List[str]        # Variablen im Block
    values: Dict[str, float]    # Berechnete Werte
    residuals: List[float]      # Residuen pro Gleichung
    max_residual: float         # Maximales Residuum
    block_number: int           # Block-Nummer


@dataclass
class UnitWarning:
    """Warnung für inkonsistente Einheiten einer Variable.

    Wird erzeugt, wenn eine Variable in verschiedenen Gleichungen
    unterschiedliche Einheiten hätte (z.B. kJ vs bar·m³).
    """
    variable: str                    # z.B. "W_v"
    equations: List[str]             # Beteiligte Gleichungen
    units: Dict[str, str]            # {equation: inferred_unit}
    explanation: str                 # Detaillierte Erklärung
    conversion_factor: float = 1.0   # z.B. 100 für bar*m³ vs kJ


@dataclass
class SolveAnalysis:
    """Vollständige Analyse-Daten einer Lösung."""
    constants: List[EquationInfo] = field(default_factory=list)
    direct_evals: List[EquationInfo] = field(default_factory=list)
    single_unknowns: List[EquationInfo] = field(default_factory=list)
    blocks: List[BlockInfo] = field(default_factory=list)
    solve_order: List[str] = field(default_factory=list)  # Reihenfolge der Lösungsschritte
    unit_warnings: List[UnitWarning] = field(default_factory=list)  # Einheiten-Inkonsistenzen

    def add_constant(self, original: str, parsed: str, var: str, value: float):
        """Fügt eine Konstante hinzu."""
        self.constants.append(EquationInfo(
            original=original, parsed=parsed, variable=var,
            value=value, residual=0.0, category="constant"
        ))
        self.solve_order.append(f"const:{var}")

    def add_direct(self, original: str, parsed: str, var: str, value: float, residual: float):
        """Fügt eine direkte Auswertung hinzu."""
        self.direct_evals.append(EquationInfo(
            original=original, parsed=parsed, variable=var,
            value=value, residual=residual, category="direct"
        ))
        self.solve_order.append(f"direct:{var}")

    def add_single_unknown(self, original: str, parsed: str, var: str, value: float, residual: float):
        """Fügt eine Einzelunbekannte hinzu."""
        self.single_unknowns.append(EquationInfo(
            original=original, parsed=parsed, variable=var,
            value=value, residual=residual, category="single_unknown"
        ))
        self.solve_order.append(f"single:{var}")

    def add_block(self, originals: List[str], parsed: List[str], variables: List[str],
                  values: Dict[str, float], residuals: List[float]):
        """Fügt einen Block hinzu."""
        block_num = len(self.blocks) + 1
        self.blocks.append(BlockInfo(
            equations=originals, parsed_equations=parsed, variables=variables,
            values=values, residuals=residuals,
            max_residual=max(abs(r) for r in residuals) if residuals else 0.0,
            block_number=block_num
        ))
        self.solve_order.append(f"block:{block_num}")


@dataclass
class BlockAnalysis:
    """Detaillierte Analyse der internen Block-Zerlegung.

    Wenn ein Block > 3 Variablen hat, wird er intern weiter zerlegt.
    Diese Klasse speichert die Details dieser Zerlegung.
    """
    direct_evals: List[EquationInfo] = field(default_factory=list)      # Direkte Auswertungen im Block
    single_unknowns: List[EquationInfo] = field(default_factory=list)   # Einzelne Unbekannte im Block
    sub_blocks: List[BlockInfo] = field(default_factory=list)           # Sub-Blöcke (gekoppelte Kerne)


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
    constants: Dict[str, float] = None,
    original_equations: Dict[str, str] = None,
    return_analysis: bool = False
) -> Union[Tuple[bool, Dict[str, float], str], Tuple[bool, Dict[str, float], str, SolveAnalysis]]:
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
        original_equations: Mapping parsed -> original für Anzeige
        return_analysis: Wenn True, wird SolveAnalysis als 4. Element zurückgegeben

    Returns:
        success: True wenn Lösung gefunden
        solution: Dictionary mit Variablen und ihren Werten
        message: Status- oder Fehlermeldung
        analysis: (optional) SolveAnalysis mit Debugging-Informationen
    """
    if initial_values is None:
        initial_values = {}
    if constants is None:
        constants = {}
    if original_equations is None:
        original_equations = {}

    context = _get_eval_context()
    analysis = SolveAnalysis()

    # Phase 1: Starte mit Konstanten
    known_values = constants.copy()
    remaining_equations = list(equations)
    remaining_vars = set(variables)

    # Konstanten zur Analysis hinzufügen
    for var, value in constants.items():
        orig = f"{var} = {value}"
        analysis.add_constant(orig, f"({var}) - ({value})", var, value)

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

                            # Berechne Residuum und füge zur Analysis hinzu
                            residual = _calculate_residual(eq, known_values, context)
                            orig = original_equations.get(eq, eq)
                            analysis.add_direct(orig, eq, var, float(result), residual)

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

                        # Berechne Residuum und füge zur Analysis hinzu
                        residual = _calculate_residual(eq, known_values, context)
                        orig = original_equations.get(eq, eq)
                        analysis.add_single_unknown(orig, eq, unknown, value, residual)

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
                    success, block_solution, block_msg, block_analysis = _solve_equation_block(
                        block_eqs, block_vars, known_values, context, initial_values, original_equations
                    )

                    if success:
                        # Aktualisiere bekannte Werte
                        known_values.update(block_solution)
                        remaining_vars -= block_vars

                        # Entferne gelöste Gleichungen
                        for eq in block_eqs:
                            if eq in remaining_equations:
                                remaining_equations.remove(eq)

                        # Füge zur Analysis hinzu - unterscheide ob Block intern zerlegt wurde
                        if block_analysis is not None:
                            # Block wurde intern zerlegt - verwende detaillierte Analysis
                            # Direkte Auswertungen aus dem Block
                            for eq_info in block_analysis.direct_evals:
                                analysis.direct_evals.append(eq_info)
                                analysis.solve_order.append(f"direct:{eq_info.variable}")

                            # Einzelne Unbekannte aus dem Block
                            for eq_info in block_analysis.single_unknowns:
                                analysis.single_unknowns.append(eq_info)
                                analysis.solve_order.append(f"single:{eq_info.variable}")

                            # Sub-Blöcke (der echte gekoppelte Kern)
                            for sub_block in block_analysis.sub_blocks:
                                sub_block.block_number = len(analysis.blocks) + 1
                                analysis.blocks.append(sub_block)
                                analysis.solve_order.append(f"block:{sub_block.block_number}")
                        else:
                            # Block wurde simultan gelöst - als Ganzes zur Analysis
                            block_residuals = []
                            for eq in block_eqs:
                                res = _calculate_residual(eq, known_values, context)
                                block_residuals.append(res)

                            orig_eqs = [original_equations.get(eq, eq) for eq in block_eqs]
                            analysis.add_block(
                                orig_eqs, list(block_eqs), list(block_vars),
                                block_solution, block_residuals
                            )

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

        if return_analysis:
            return True, result, msg, analysis
        return True, result, msg
    else:
        # Nicht alle Gleichungen gelöst
        msg = f"Unvollständig: {len(remaining_equations)} Gleichungen, {len(remaining_vars)} Unbekannte verbleibend"
        if return_analysis:
            return False, result, msg, analysis
        return False, result, msg


def _calculate_residual(equation: str, known_values: Dict[str, float], context: dict) -> float:
    """Berechnet das Residuum einer Gleichung mit den gegebenen Werten."""
    try:
        local_ctx = context.copy()
        local_ctx.update(known_values)
        result = eval(equation, {"__builtins__": {}}, local_ctx)
        return float(result) if np.isfinite(result) else float('inf')
    except Exception:
        return float('inf')


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
        'sqrt': sqrt, 'abs': np.abs, 'pi': pi,
        'max': max, 'min': min
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
    manual_initial: Dict[str, float] = None,
    original_equations: Dict[str, str] = None
) -> Tuple[bool, Dict[str, float], str, Optional[BlockAnalysis]]:
    """
    Löst einen Block von Gleichungen mit gemeinsamen Unbekannten.

    Bei größeren Blöcken wird zuerst versucht, den Block iterativ zu zerlegen
    und kleinere Sub-Blöcke sequentiell zu lösen.

    Args:
        manual_initial: Manuelle Startwerte (haben Priorität)
        original_equations: Mapping parsed -> original für Anzeige

    Returns:
        (success, solution_dict, message, block_analysis)
    """
    if not equations or not variables:
        return True, {}, "Leerer Block", None

    n_vars = len(variables)
    n_eqs = len(equations)

    if n_eqs != n_vars:
        return False, {}, f"Block nicht quadratisch: {n_eqs} Gleichungen, {n_vars} Unbekannte", None

    # Bei größeren Blöcken: Versuche iterative Zerlegung
    if n_vars > 3:
        success, solution, msg, block_analysis = _solve_block_iteratively(
            equations, variables, known_values, context, manual_initial, original_equations
        )
        if success:
            return success, solution, msg, block_analysis

    # Fallback: Löse den gesamten Block simultan (keine interne Zerlegung)
    success, solution, msg = _solve_block_simultaneously(
        equations, variables, known_values, context, manual_initial
    )
    return success, solution, msg, None  # Keine BlockAnalysis für simultane Lösung


def _solve_block_iteratively(
    equations: List[str],
    variables: Set[str],
    known_values: Dict[str, float],
    context: dict,
    manual_initial: Dict[str, float] = None,
    original_equations: Dict[str, str] = None
) -> Tuple[bool, Dict[str, float], str, Optional[BlockAnalysis]]:
    """
    Versucht einen Block iterativ zu lösen, indem nach jeder gelösten
    Gleichung geprüft wird, ob weitere Gleichungen direkt oder mit
    nur einer Unbekannten lösbar sind.

    Returns:
        (success, solution_dict, message, block_analysis)
    """
    if original_equations is None:
        original_equations = {}

    remaining_eqs = list(equations)
    remaining_vars = set(variables)
    local_known = known_values.copy()
    solved_values = {}
    stats = {'direct': 0, 'single': 0, 'subblocks': []}

    # BlockAnalysis für detaillierte Tracking
    block_analysis = BlockAnalysis()

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

                            # Zur BlockAnalysis hinzufügen
                            residual = _calculate_residual(eq, local_known, context)
                            orig = original_equations.get(eq, eq)
                            block_analysis.direct_evals.append(EquationInfo(
                                original=orig, parsed=eq, variable=var,
                                value=float(result), residual=residual, category="direct"
                            ))

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

                    # Zur BlockAnalysis hinzufügen
                    residual = _calculate_residual(eq, local_known, context)
                    orig = original_equations.get(eq, eq)
                    block_analysis.single_unknowns.append(EquationInfo(
                        original=orig, parsed=eq, variable=unknown,
                        value=value, residual=residual, category="single_unknown"
                    ))

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

                    # Residuen für Sub-Block berechnen
                    sub_residuals = []
                    for eq in core_eqs:
                        res = _calculate_residual(eq, local_known, context)
                        sub_residuals.append(res)
                        if eq in remaining_eqs:
                            remaining_eqs.remove(eq)

                    # Sub-Block zur BlockAnalysis hinzufügen
                    orig_eqs = [original_equations.get(eq, eq) for eq in core_eqs]
                    block_analysis.sub_blocks.append(BlockInfo(
                        equations=orig_eqs,
                        parsed_equations=list(core_eqs),
                        variables=list(core_vars),
                        values=sub_solution,
                        residuals=sub_residuals,
                        max_residual=max(abs(r) for r in sub_residuals) if sub_residuals else 0.0,
                        block_number=len(block_analysis.sub_blocks) + 1
                    ))

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
        return True, solved_values, f"Block zerlegt ({', '.join(msg_parts)})", block_analysis

    return False, solved_values, f"Iterative Zerlegung unvollständig", block_analysis


def _solve_block_simultaneously(
    equations: List[str],
    variables: Set[str],
    known_values: Dict[str, float],
    context: dict,
    manual_initial: Dict[str, float] = None
) -> Tuple[bool, Dict[str, float], str]:
    """
    Löst einen Block von Gleichungen simultan mit normalisiertem least_squares.

    Strategie:
    1. Intelligente Startwerte basierend auf Variablennamen
    2. Normalisierung: Alle Variablen auf ~1.0 skalieren
    3. least_squares mit Levenberg-Marquardt (robuster als fsolve)
    4. Relative Toleranzen für Konvergenzprüfung
    """
    from scipy.optimize import least_squares

    if not equations or not variables:
        return True, {}, "Leerer Block"

    var_list = sorted(list(variables))
    n_vars = len(var_list)
    n_eqs = len(equations)

    if n_eqs != n_vars:
        return False, {}, f"Block nicht quadratisch: {n_eqs} Gleichungen, {n_vars} Unbekannte"

    # Erstelle Startvektor mit intelligenten Startwerten
    x0 = np.array([_get_initial_value(var, manual_initial, known_values)
                   for var in var_list], dtype=float)

    # Skalierungsfaktoren = initiale Werte (damit normalisierte Variablen ~1.0 sind)
    scales = np.maximum(np.abs(x0), 1e-10)

    # Residuen-Funktion (nicht normalisiert, für Auswertung)
    def block_func(x):
        local_ctx = context.copy()
        local_ctx.update(known_values)
        local_ctx.update({var: val for var, val in zip(var_list, x)})

        results = []
        for eq in equations:
            try:
                result = eval(eq, {"__builtins__": {}}, local_ctx)
                if not np.isfinite(result):
                    result = 1e10
                results.append(result)
            except Exception:
                results.append(1e10)
        return np.array(results)

    # Normalisierte Residuen-Funktion für least_squares
    def normalized_residuals(x_norm):
        # Zurückskalieren: x_real = x_norm * scale
        x_real = x_norm * scales
        return block_func(x_real)

    # Versuche mit least_squares (robuster als fsolve)
    best_solution = None
    best_residual = float('inf')

    # Startwert-Strategien
    x0_norm = x0 / scales  # Sollte ~1.0 sein für alle Variablen

    start_variations = [
        x0_norm,
        x0_norm * 1.5,
        x0_norm * 0.5,
        x0_norm * 2.0,
        x0_norm * 0.25,
    ]

    # Zusätzliche Variationen für einzelne Variablen
    for i in range(min(n_vars, 3)):  # Max 3 zusätzliche pro Variable
        variation = x0_norm.copy()
        variation[i] *= 3.0
        start_variations.append(variation)
        variation = x0_norm.copy()
        variation[i] *= 0.1
        start_variations.append(variation)

    import warnings
    for x_start in start_variations[:15]:  # Maximal 15 Versuche
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = least_squares(
                    normalized_residuals,
                    x_start,
                    method='lm',  # Levenberg-Marquardt
                    ftol=1e-12,   # Relative Toleranz für Residuen
                    xtol=1e-12,   # Relative Toleranz für Variablen
                    max_nfev=500 * n_vars
                )

            if result.success or result.status in [1, 2, 3, 4]:
                # Zurückskalieren
                solution = result.x * scales
                residual = np.max(np.abs(result.fun))

                # Relative Toleranz: Residuum sollte klein relativ zur Lösungsgröße sein
                typical_scale = np.max(np.abs(solution)) if np.any(solution != 0) else 1.0
                relative_residual = residual / max(1.0, typical_scale)

                if relative_residual < 1e-8:
                    result_dict = {var: val for var, val in zip(var_list, solution)}
                    return True, result_dict, f"Block gelöst ({n_vars} Variablen)"

                if np.all(np.isfinite(solution)) and residual < best_residual:
                    best_solution = solution
                    best_residual = residual

        except Exception:
            pass

    # Fallback: Versuche fsolve mit verschiedenen Startwerten
    for x_start_norm in start_variations[:5]:
        try:
            x_start = x_start_norm * scales
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                solution, info, ier, _ = fsolve(block_func, x_start, full_output=True)

            residual = np.max(np.abs(info['fvec']))
            typical_scale = np.max(np.abs(solution)) if np.any(solution != 0) else 1.0

            if ier == 1 and np.all(np.isfinite(solution)) and residual < 1e-6 * max(1.0, typical_scale):
                result_dict = {var: val for var, val in zip(var_list, solution)}
                return True, result_dict, f"Block gelöst ({n_vars} Variablen)"

            if np.all(np.isfinite(solution)) and residual < best_residual:
                best_solution = solution
                best_residual = residual
        except Exception:
            pass

    # Akzeptiere gute Näherung mit relativer Toleranz
    if best_solution is not None:
        typical_scale = np.max(np.abs(best_solution)) if np.any(best_solution != 0) else 1.0
        relative_residual = best_residual / max(1.0, typical_scale)

        if relative_residual < 1e-4:
            result_dict = {var: val for var, val in zip(var_list, best_solution)}
            return True, result_dict, f"Block gelöst (rel. Residuum: {relative_residual:.2e})"

    return False, {}, f"Block-Konvergenz fehlgeschlagen (Residuum: {best_residual:.2e})"


def _get_initial_value(var: str, manual_initial: Dict[str, float] = None,
                       known_values: Dict[str, float] = None,
                       inferred_units: Dict[str, str] = None) -> float:
    """
    Ermittelt sinnvolle Startwerte basierend auf Einheiten (bevorzugt) oder Variablennamen.

    BEVORZUGT: Unit-basierte Startwerte (generisch, unabhängig von Variablennamen)
    FALLBACK: Variablennamen-Heuristik

    Für SI-Einheiten sind typische Größenordnungen:
    - Temperatur T: ~300-800 K
    - Druck p: ~100000-3000000 Pa (1-30 bar)
    - Enthalpie h: ~100000-3500000 J/kg
    - Entropie s: ~1000-8000 J/(kg·K)
    - Massenstrom m_dot: ~0.1-10 kg/s
    - Leistung W_dot, Q_dot: ~1000-1000000 W
    - Dampfqualität x: 0-1
    - Wirkungsgrad eta: 0-1
    """
    # PRIORITÄT 0: Manuelle Startwerte haben höchste Priorität
    if manual_initial and var in manual_initial:
        return manual_initial[var]

    # PRIORITÄT 1: Unit-basierte Startwerte (generisch, NICHT von Variablennamen abhängig)
    if inferred_units and var in inferred_units and UNITS_AVAILABLE:
        unit = inferred_units[var]
        if unit is not None:
            return get_initial_from_unit(unit)

    var_lower = var.lower()

    # PRIORITÄT 2: Suche nach ähnlich benannten bekannten Variablen
    # z.B. für h_5 schaue nach h_4, h_5s, h_1 etc.
    if known_values:
        # Extrahiere Basisnamen (z.B. "h" aus "h_5" oder "h_5s")
        var_base = var_lower.rstrip('0123456789')
        if var_base.endswith('_'):
            var_base = var_base[:-1]
        if var_base.endswith('s'):  # z.B. h_5s -> h_5 -> h
            var_base = var_base[:-1].rstrip('0123456789_')

        similar_vals = []
        for kv_name, kv_val in known_values.items():
            if isinstance(kv_val, (int, float)) and np.isfinite(kv_val):
                kv_lower = kv_name.lower()
                kv_base = kv_lower.rstrip('0123456789')
                if kv_base.endswith('_'):
                    kv_base = kv_base[:-1]
                if kv_base.endswith('s'):
                    kv_base = kv_base[:-1].rstrip('0123456789_')

                if kv_base == var_base and abs(kv_val) > 0.001:
                    similar_vals.append(kv_val)

        if similar_vals:
            # Verwende den Durchschnitt der ähnlichen Werte
            avg = sum(similar_vals) / len(similar_vals)
            return avg

    # Temperatur (T, T_1, t_ein, etc.) - NICHT tau, theta
    if var_lower.startswith('t') and not var_lower.startswith('tau') and not var_lower.startswith('theta'):
        # Taupunkt und Feuchtkugeltemperatur etwas niedriger
        if '_dp' in var_lower or '_wb' in var_lower:
            return 290.0  # ~17°C
        return 400.0  # ~127°C - typisch für Dampfprozesse

    # Druck (p, p_1, p_ein, etc.) - NICHT prandtl, phi
    if var_lower.startswith('p') and not var_lower.startswith('pr') and not var_lower.startswith('phi'):
        return 500000.0  # 5 bar

    # Enthalpie (h, h_1, h_ein, etc.) - NICHT humidity
    if var_lower.startswith('h') and not var_lower.startswith('hum'):
        return 1000000.0  # 1000 kJ/kg

    # Entropie (s, s_1, etc.)
    if var_lower.startswith('s') and len(var_lower) <= 4:
        return 3000.0  # 3 kJ/(kg·K)

    # Leistung (W_dot, Q_dot, P_dot, etc.)
    if '_dot' in var_lower or 'dot_' in var_lower:
        if var_lower.startswith('m'):
            return 1.0  # 1 kg/s Massenstrom
        if var_lower.startswith('v'):
            return 0.1  # 0.1 m³/s Volumenstrom
        # W_dot, Q_dot, P_dot - Leistung
        return 100000.0  # 100 kW

    # Dampfqualität (x, x_1, x_aus, etc.)
    if var_lower.startswith('x') and len(var_lower) <= 5:
        return 0.5

    # Wirkungsgrad (eta, eta_th, eta_s_i_T, etc.)
    if var_lower.startswith('eta'):
        return 0.85

    # Relative Feuchte (rh, phi)
    if var_lower.startswith('rh') or var_lower == 'phi':
        return 0.5

    # Feuchtegehalt (w als Luftfeuchte - kurze Namen)
    if var_lower == 'w' or (var_lower.startswith('w_') and len(var_lower) <= 4):
        return 0.01  # 10 g/kg

    # Dichte (rho, rho_1, etc.)
    if var_lower.startswith('rho'):
        return 1.0  # 1 kg/m³

    # Spezifisches Volumen (v, v_1, etc.) - NICHT v_dot
    if var_lower.startswith('v') and '_dot' not in var_lower and len(var_lower) <= 4:
        return 0.1  # 0.1 m³/kg

    # Innere Energie (u, u_1, etc.)
    if var_lower.startswith('u') and len(var_lower) <= 4:
        return 500000.0  # 500 kJ/kg

    # Fallback: Geometrisches Mittel aller bekannten Werte
    if known_values:
        positive_vals = [abs(v) for v in known_values.values()
                        if isinstance(v, (int, float)) and v > 0.01]
        if positive_vals:
            import math
            try:
                geo_mean = math.exp(sum(math.log(v) for v in positive_vals) / len(positive_vals))
                return geo_mean
            except (ValueError, OverflowError):
                pass

    return 1.0  # Ultimativer Fallback


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
    # Verwende intelligente Startwerte basierend auf Variablennamen
    initial_guess = _get_initial_value(unknown, manual_initial, known_values)

    try:
        x = initial_guess

        for iteration in range(30):  # Max 30 Iterationen
            fx = func(x)

            # Prüfe ob bereits Lösung gefunden
            # Absolutes Residuum muss klein sein (Gleichung = 0)
            if abs(fx) < 1e-10:
                return True, x

            # Skalierte Schrittweite für numerische Ableitung
            # Bei großen x-Werten (z.B. 1e6) brauchen wir größere h
            h = max(1e-12, 1e-8 * max(1.0, abs(x)))

            # Numerische Ableitung
            fx_plus = func(x + h)
            fx_minus = func(x - h)
            dfx = (fx_plus - fx_minus) / (2 * h)

            # Prüfe ob Ableitung gültig
            if not np.isfinite(dfx) or abs(dfx) < 1e-15:
                break  # Newton funktioniert nicht, verwende Bracket-Suche

            # Newton-Schritt
            x_new = x - fx / dfx

            # Prüfe Konvergenz (relative Änderung)
            if abs(x_new - x) < 1e-10 * max(1, abs(x)):
                # Verifiziere Lösung - absolutes Residuum muss klein sein
                fx_new = func(x_new)
                if abs(fx_new) < 1e-8:
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
    # Bestimme Skalierung basierend auf bekannten Werten
    # Bei SI-Einheiten können Werte sehr groß sein (z.B. 1e6 für Pa oder J/kg)
    max_known = max((abs(v) for v in known_values.values() if isinstance(v, (int, float))), default=1.0)
    scale = max(1.0, 10 ** (int(np.log10(max_known + 1)) - 1)) if max_known > 10 else 1.0

    # Erzeuge Testpunkte mit dichter Abdeckung
    test_points = set()

    # Logarithmische Skalierung für extreme Bereiche (erweitert bis 1e9)
    for exp in range(-2, 10):
        base = 10**exp
        test_points.update([base, -base, 0.5*base, 2*base, 5*base, -0.5*base, -2*base, -5*base])

    # Feine Abdeckung im Bereich 0-100
    test_points.update([i * 0.1 for i in range(-1000, 1001)])

    # Dichte Abdeckung im skalierten Bereich (0 bis scale)
    if scale > 1:
        for i in range(0, 1001):
            test_points.add(i * scale / 1000)
            test_points.add(-i * scale / 1000)
        # Gröbere Abdeckung bis 10*scale
        for i in range(100, 1001, 10):
            test_points.add(i * scale / 100)
            test_points.add(-i * scale / 100)

    # Standard-Abdeckung für kleinere Werte
    test_points.update(range(0, 501, 1))
    test_points.update(range(-500, 0, 1))
    test_points.update(range(500, 1001, 10))
    test_points.update(range(-1000, -500, 10))
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
                # Absolutes Residuum sollte sehr klein sein (Gleichung = 0)
                # Bei korrekter Lösung ist f(x) ≈ 0, unabhängig von der Größe von x
                if residual < 1e-8:
                    return True, float(root)
                elif residual < best_residual:
                    best_root = root
                    best_residual = residual
        except Exception:
            pass

    # Beste gefundene Lösung zurückgeben wenn akzeptabel
    # Absolutes Residuum muss klein sein - bei f(x)=0 muss f(root) ≈ 0 sein
    if best_root is not None and best_residual < 1e-6:
        return True, float(best_root)

    # Fallback: fsolve mit verschiedenen Startwerten
    # Beginne mit intelligentem Startwert basierend auf Variablennamen
    import warnings
    smart_start = _get_initial_value(unknown, manual_initial, known_values)
    fallback_starts = [smart_start, 1.0, 0.1, 10.0, 100.0, 1000.0, 10000.0]
    if scale > 1:
        fallback_starts.extend([scale, scale * 0.1, scale * 0.5, scale * 2, scale * 10])

    for x0 in fallback_starts:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                solution, info, ier, _ = fsolve(func, x0, full_output=True)
            residual = abs(info['fvec'][0])
            # Absolutes Residuum muss klein sein - Gleichung ist normiert auf f(x)=0
            # Bei korrekter Lösung sollte f(x) ≈ 0 sein
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
