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
import re
import time as _time

from parser import if_function, display_name
try:
    from units import unit_number, unit_quantity
except ImportError:  # ohne pint: Zahlenwertgleichungen nicht verfügbar
    def unit_number(x, unit):
        raise ValueError('value() benötigt das Einheiten-Modul (pint)')

    def unit_quantity(z, unit):
        raise ValueError('quantity() benötigt das Einheiten-Modul (pint)')

# Gesamtzeitlimit eines Lösungslaufs (solve_system): verschachtelte Versuche
# (Zerlegung, Tearing, simultan, Startvarianten) dürfen sich nicht zu Minuten
# aufsummieren - die GUI wäre so lange eingefroren.
SOLVE_TIME_LIMIT = 60.0
_deadline: Optional[float] = None


class SolveTimeout(BaseException):
    """
    Zeitlimit überschritten. Bewusst BaseException: die Teil-Löser fangen
    numerische Fehler mit 'except Exception' ab und dürfen das Zeitlimit
    nicht verschlucken.
    """


def _check_deadline() -> None:
    if _deadline is not None and _time.monotonic() > _deadline:
        raise SolveTimeout()

# Versuche Einheiten-Modul zu laden für unit-basierte Startwerte
try:
    from units import get_initial_from_unit, typical_value_from_unit
    UNITS_AVAILABLE = True
except ImportError:
    UNITS_AVAILABLE = False
    def get_initial_from_unit(unit_str):
        return 1.0

    def typical_value_from_unit(unit_str):
        return None

# Versuche Unit-Constraints zu laden für generische Einheiten-Inferenz aus Funktionsargumenten
try:
    from unit_constraints import propagate_all_units_complete
    UNIT_INFERENCE_AVAILABLE = True
except ImportError:
    UNIT_INFERENCE_AVAILABLE = False
    def propagate_all_units_complete(equations, known_units, max_iterations=15):
        return known_units.copy() if known_units else {}


# Ergebnis der Einheiten-Propagation je Gleichungssatz: hängt nur von den Gleichungen
# ab, nicht von den Zahlenwerten - Parameterstudien und Optimierung lösen dasselbe
# System sehr oft (die Propagation wäre sonst der größte Teil der Rechenzeit)
_UNIT_CACHE: Dict[tuple, Dict[str, str]] = {}


def _inferred_units(original_equations: Dict[str, str]) -> Dict[str, str]:
    key = tuple(original_equations.items())
    cached = _UNIT_CACHE.get(key)
    if cached is None:
        cached = propagate_all_units_complete(original_equations, {})
        if len(_UNIT_CACHE) >= 8:
            _UNIT_CACHE.clear()
        _UNIT_CACHE[key] = cached
    return dict(cached)


# Startwerte aus der Struktur (generisch, ohne Namen und ohne Einheiten): Summanden
# derselben Summe/Differenz haben dieselbe Dimension (wie bei der Einheiten-Propagation).
# Die Variablen, die so verbunden sind, bilden ein Netz (T_R - T_G1, T_G1 - T_G2,
# T_G2 - T_G3, T_G3 - T_a: T_R - T_G1 - T_G2 - T_G3 - T_a). Unbekannte eines gekoppelten
# Blocks ohne anderen Startwert starten beim Mittel ihrer Nachbarn im Netz, die bekannten
# Werte bleiben fest (Interpolation zwischen den bekannten Werten: 285.65 / 278.15 /
# 270.65 K) - verschiedene Startwerte, nicht genau auf einem Randwert (eine Differenz null
# ist ein typischer singulärer Punkt: Ra = 0, ln(dT_1/dT_2), 1/dT). Bekannte
# zusammengesetzte Summanden zählen als feste Nachbarn (sigma*T_D^4 - J_2 -> J_2).
# Zahlenliterale zählen nicht (1 in (1 - eps) ist keine Größenordnung von eps).
# Nur Iterationsstart - NICHT für die Auswahl unter mehreren Wurzeln.
_start_hints = None
# Im laufenden Lösungslauf bereits gelöste Kerne (bleiben bei Zeitlimit als Teillösung erhalten)
_partial_solution: Optional[Dict[str, float]] = None
_START_HINT_CACHE: Dict[tuple, tuple] = {}


def _additive_chains(expr: str) -> list:
    """Alle Summen/Differenzen eines Ausdrucks: je Kette die Summanden (AST-Knoten)."""
    import ast
    try:
        tree = ast.parse(expr, mode='eval').body
    except SyntaxError:
        return []
    chains = []

    def terms(node, out):
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            terms(node.left, out)
            terms(node.right, out)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            terms(node.operand, out)
        else:
            out.append(node)

    def visit(node):
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            out = []
            terms(node, out)
            chains.append(out)
            for term in out:
                visit(term)
        else:
            for child in ast.iter_child_nodes(node):
                visit(child)

    visit(tree)
    return chains


def _structural_start_hints(equations: List[str]) -> tuple:
    """(Nachbarn je Variable, zusammengesetzte Nachbar-Summanden je Variable, Klasse je Variable)."""
    key = tuple(equations)
    cached = _START_HINT_CACHE.get(key)
    if cached is not None:
        return cached
    import ast
    neighbors: Dict[str, Set[str]] = {}
    composite: Dict[str, list] = {}
    for eq in equations:
        for terms in _additive_chains(eq):
            names = [t.id for t in terms if isinstance(t, ast.Name)]
            if not names:
                continue
            others = []
            for term in terms:
                if isinstance(term, (ast.Name, ast.Constant)):
                    continue
                called = {n.func.id for n in ast.walk(term)
                          if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
                term_names = frozenset({n.id for n in ast.walk(term) if isinstance(n, ast.Name)} - called)
                if term_names:
                    others.append((compile(ast.Expression(body=term), '<summand>', 'eval'), term_names))
            for name in names:
                neighbors.setdefault(name, set()).update(n for n in names if n != name)
                composite.setdefault(name, []).extend(others)
    classes: Dict[str, int] = {}
    for start_name in neighbors:
        if start_name in classes:
            continue
        stack, label = [start_name], len(classes)
        while stack:
            name = stack.pop()
            if name not in classes:
                classes[name] = label
                stack.extend(neighbors[name] - classes.keys())
    hints = (neighbors, composite, classes)
    if len(_START_HINT_CACHE) >= 8:
        _START_HINT_CACHE.clear()
    _START_HINT_CACHE[key] = hints
    return hints


def _sum_net_scale(equation: str, known_values: Dict[str, float]) -> float:
    """
    Größenordnung GLEICHARTIGER Größen für die Rauschgrenze einer Prüfgleichung: die Größen,
    die als Summanden der Gleichung stehen, und alle über Summen mit ihnen verbundenen (gleiche
    Dimension, _start_hints) - q_x = 1e-17 aus q_x = q_1 - q_2 ist gegenüber q_1 numerisch null.
    Nicht der größte Wert beliebiger Größen: 1e5 Pa darf einen Widerspruch zwischen
    Volumenströmen von 1e-7 m^3/s nicht verdecken.
    """
    import ast
    names = set()
    for terms in _additive_chains(equation):
        names.update(t.id for t in terms if isinstance(t, ast.Name))
    if _start_hints is not None:
        _, _, classes = _start_hints
        labels = {classes[n] for n in names if n in classes}
        names.update(n for n, c in classes.items() if c in labels)
    values = [abs(float(known_values[n])) for n in names
              if n in known_values and np.isscalar(known_values[n]) and np.isfinite(known_values[n])]
    return max(values, default=0.0)


def _structural_start(var: str, known_values: Optional[Dict[str, float]]) -> Optional[float]:
    """Startwert durch Interpolation im Netz der über Summen verbundenen Größen oder None."""
    if _start_hints is None or not known_values:
        return None
    neighbors, composite, classes = _start_hints
    if var not in classes:
        return None
    label = classes[var]
    nodes = [n for n, c in classes.items() if c == label]

    def known(name):
        value = known_values.get(name)
        return value is not None and np.isscalar(value) and np.isfinite(value)

    unknown = [n for n in nodes if not known(n)]
    if var not in unknown:
        return None
    context = None
    fixed: Dict[str, List[float]] = {}         # feste Nachbarwerte je Unbekannte
    for name in unknown:
        values = [float(known_values[n]) for n in neighbors[name] if known(n)]
        for code, term_names in composite[name]:
            if not term_names <= known_values.keys():
                continue
            if context is None:
                context = _get_eval_context()
            try:
                value = float(_as_real(eval(code, {"__builtins__": {}},
                                            {**context, **{n: known_values[n] for n in term_names}})))
            except Exception:
                continue
            if np.isfinite(value):
                values.append(value)
        fixed[name] = values
    all_fixed = [v for values in fixed.values() for v in values]
    if not all_fixed:
        return None
    # Laplace im Netz: (Zahl der Nachbarn) * x_u - Summe unbekannter Nachbarn = Summe fester
    # Nachbarn; schwache Bindung an das Mittel, damit Teile ohne festen Nachbarn bestimmt sind
    index = {n: i for i, n in enumerate(unknown)}
    mean = float(np.mean(all_fixed))
    A = np.zeros((len(unknown), len(unknown)))
    b = np.zeros(len(unknown))
    for name, i in index.items():
        unknown_neighbors = [index[n] for n in neighbors[name] if n in index]
        A[i, i] = len(unknown_neighbors) + len(fixed[name]) + 1e-9
        for j in unknown_neighbors:
            A[i, j] -= 1.0
        b[i] = sum(fixed[name]) + 1e-9 * mean
    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return mean
    value = float(x[index[var]])
    return value if np.isfinite(value) else mean


def _appearance_order(equations: List[str], names) -> Dict[str, Tuple[int, int]]:
    """
    Reihenfolge des ersten Auftretens im Blatt (Gleichung, Position) - für Entscheidungen
    bei Gleichstand statt der alphabetischen Reihenfolge: Ergebnisse dürfen nicht von den
    Formelzeichen abhängen.
    """
    import re
    names = set(names)
    order: Dict[str, Tuple[int, int]] = {}
    blank = lambda m: ' ' * len(m.group(0))
    for i, eq in enumerate(equations):
        # wie _get_equation_unknowns: Texte ('h', 'water') und Schlüsselwort-NAMEN (p=, T=) sind
        # kein Vorkommen einer Variablen p bzw. T (Positionen bleiben erhalten)
        eq = re.sub(r"'[^']*'|\"[^\"]*\"", blank, eq)
        eq = re.sub(r'\b[a-zA-Z_][a-zA-Z0-9_]*\s*=(?!=)', blank, eq)
        for match in re.finditer(r'[A-Za-z_][A-Za-z0-9_]*', eq):
            name = match.group(0)
            if name in names and name not in order:
                order[name] = (i, match.start())
    big = (len(equations), 0)
    return {name: order.get(name, big) for name in names}


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
    hints: List[str] = field(default_factory=list)  # Generische Hinweise (z.B. mögliche Tippfehler)
    evaluation_errors: List[str] = field(default_factory=list)  # Funktions-/Auswertungsfehler

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


# Trigonometrische Funktionen in GRAD
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

# Chemisches Gleichgewicht (Anweisung EQUILIBRIUM, Gibbs-Minimierung)
try:
    from equilibrium import equilibrium as _equilibrium_function
    EQUILIBRIUM_AVAILABLE = True
except ImportError:
    EQUILIBRIUM_AVAILABLE = False

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


def _as_real(value):
    """
    Wertet komplexe Zwischenergebnisse als ungültig (NaN).

    Python liefert für negative Basis mit gebrochenem Exponenten komplexe
    Zahlen ((-8)**(1/3) -> 1+1.73j), z.B. Ra^(1/6) bei negativem Ra während
    der Iteration. Ohne diese Umwandlung bricht der Vorzeichenvergleich der
    Bracket-Suche mit TypeError ab.
    """
    if isinstance(value, complex) or np.iscomplexobj(value):
        if np.all(np.imag(value) == 0):
            return np.real(value)
        return float('nan')
    return value


def solve_system(
    equations: List[str],
    variables: Set[str],
    initial_values: Optional[Dict[str, float]] = None,
    initial_guess: float = 1.0,
    constants: Optional[Dict[str, float]] = None,
    original_equations: Optional[Dict[str, str]] = None,
    return_analysis: bool = False
) -> Union[Tuple[bool, Dict[str, float], str], Tuple[bool, Dict[str, float], str, SolveAnalysis]]:
    """
    Löst das Gleichungssystem (Strategie siehe _solve_system_impl) innerhalb
    des Gesamtzeitlimits SOLVE_TIME_LIMIT. Bei Überschreitung: Teillösung und
    Meldung mit den noch offenen Unbekannten (kein Einfrieren der GUI).

    Returns:
        success, solution, message[, analysis] wie _solve_system_impl
    """
    global _deadline, _start_hints, _partial_solution
    own_deadline = _deadline is None
    previous_partial = _partial_solution
    _partial_solution = {}
    if own_deadline:
        _deadline = _time.monotonic() + SOLVE_TIME_LIMIT
    state = {}
    previous_hints = _start_hints
    try:
        _start_hints = _structural_start_hints(equations)
    except Exception:
        _start_hints = None
    try:
        return _solve_system_impl(equations, variables, initial_values, initial_guess, constants,
                                  original_equations, return_analysis, _state=state)
    except SolveTimeout:
        if not own_deadline or not state:
            raise
        # gelöste Kerne eines noch laufenden Blocks gehören zur Teillösung
        result = {**_partial_solution, **dict(state['known'])}
        remaining = sorted(v for v in state['vars'] if v not in result)
        open_eqs = [eq for eq in state['eqs'] if _get_equation_unknowns(eq, set(result), set(remaining))]
        names = ', '.join(display_name(v) for v in remaining[:8]) + (', ...' if len(remaining) > 8 else '')
        msg = (f"Zeitlimit von {SOLVE_TIME_LIMIT:.0f} s überschritten: {len(open_eqs)} Gleichungen "
               f"mit {len(remaining)} Unbekannten ungelöst ({names}). Startwerte vorgeben "
               f"(Solve > Initial Values) oder das System vereinfachen")
        if return_analysis:
            return False, result, msg, state['analysis']
        return False, result, msg
    finally:
        _start_hints = previous_hints
        _partial_solution = previous_partial
        if own_deadline:
            _deadline = None


def _solve_system_impl(
    equations: List[str],
    variables: Set[str],
    initial_values: Optional[Dict[str, float]] = None,
    initial_guess: float = 1.0,
    constants: Optional[Dict[str, float]] = None,
    original_equations: Optional[Dict[str, str]] = None,
    return_analysis: bool = False,
    _state: Optional[dict] = None
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

    # Phase 0: Einheiten aus Funktionsargumenten inferieren (generischer Ansatz)
    # Bei HumidAir(h, T=T_AUL, rF=rF_AUL, p_tot=p) wird erkannt:
    #   - T_AUL hat Einheit K (weil T-Argument)
    #   - rF_AUL ist dimensionslos (weil rF-Argument)
    #   - p hat Einheit Pa (weil p_tot-Argument)
    inferred_units = {}
    if UNIT_INFERENCE_AVAILABLE and original_equations:
        inferred_units = _inferred_units(original_equations)

    # Phase 1: Starte mit Konstanten
    known_values = constants.copy()
    remaining_equations = list(equations)
    remaining_vars = set(variables)
    if _state is not None:
        # dieselben Objekte (werden nur in place verändert): Teilstand bei Zeitlimit
        _state.update(known=known_values, eqs=remaining_equations, vars=remaining_vars,
                      analysis=analysis)

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
    violated_constraints = []  # Widersprüchliche Constraint-Gleichungen
    non_unique = []            # (abhängige Gleichungen, unbestimmte Größen) gelöster Blöcke
    evaluation_errors = {}     # Gleichung -> Fehlermeldung der direkten Auswertung
    _LAST_SEARCH_ERROR.clear()  # nur Fehler dieses Lösungslaufs melden
    _DEPENDENT_CORES.clear()
    # Gescheiterte Blöcke merken: Blöcke sind Zusammenhangskomponenten der
    # Unbekannten, neue Werte aus anderen Blöcken ändern sie nicht - ein
    # erneuter (teurer) Versuch nach jedem anderen gelösten Block wäre sinnlos.
    failed_blocks = set()

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
                        result = _as_real(eval(expr, {"__builtins__": {}}, local_context))

                        if np.isfinite(result):
                            known_values[var] = float(result)
                            remaining_vars.discard(var)
                            remaining_equations.remove(eq)
                            stats['direct'] += 1
                            evaluation_errors.pop(eq, None)

                            # Berechne Residuum und füge zur Analysis hinzu
                            residual = _calculate_residual(eq, known_values, context)
                            orig = original_equations.get(eq, eq)
                            analysis.add_direct(orig, eq, var, float(result), residual)

                            made_progress = True
                            break
                        # Alle Größen bekannt, Ergebnis aber nicht definiert (0/0, ln(-1), ...)
                        evaluation_errors[eq] = (f"{var} ist nicht definiert "
                                                 f"(Ergebnis {result}, z.B. Division durch 0)")
                    except NameError:
                        pass  # Noch unbekannte Größe im Ausdruck - später erneut
                    except Exception as exc:
                        # Fehler einer Funktion (z.B. CoolProp/HumidAir) - Meldung merken
                        evaluation_errors[eq] = f"{var}: {exc}"

        # 2b: Gleichungen mit einer Unbekannten iterativ lösen
        if not made_progress:
            for eq in remaining_equations[:]:
                unknowns = _get_equation_unknowns(eq, set(known_values.keys()), remaining_vars)

                # Constraint-Gleichung: alle Variablen sind bekannt (0 Unbekannte)
                # Dies passiert z.B. bei "RWZ = (T-T0)/(T1-T0)" wenn RWZ als Konstante definiert ist
                # und T_ZUL_WT bereits aus einer anderen Gleichung gelöst wurde
                if len(unknowns) == 0:
                    # Überprüfe ob die Gleichung erfüllt ist (Residuum nahe 0)
                    residual = _calculate_residual(eq, known_values, context)
                    value_scale = _sum_net_scale(eq, known_values)
                    rel_residual = _relative_residual(eq, known_values, context,
                                                      scale_floor=1e-9 * value_scale)
                    remaining_equations.remove(eq)
                    orig = original_equations.get(eq, eq)
                    analysis.add_direct(orig, eq, "(constraint)", 0.0, residual)
                    # Verletzte Constraints (z.B. "x+1=3" UND "x+1=4") dürfen NICHT
                    # stillschweigend entfernt werden - das System ist widersprüchlich
                    if not np.isfinite(rel_residual) or rel_residual > 1e-4:
                        violated_constraints.append(
                            _describe_violation(eq, orig, known_values, constants, context))
                    made_progress = True
                    break

                if len(unknowns) == 1:
                    unknown = list(unknowns)[0]
                    success, value = _solve_single_unknown(
                        eq, unknown, known_values, context, initial_values, inferred_units
                    )
                    if not success and eq in _LAST_SEARCH_ERROR and eq not in evaluation_errors:
                        evaluation_errors[eq] = f"{unknown}: {_LAST_SEARCH_ERROR[eq]}"
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

            # Versuche ALLE Blöcke (kleinster zuerst) - ein nicht-quadratischer
            # oder nicht konvergierender Block darf lösbare Blöcke nicht blockieren
            for block_eqs, block_vars in blocks:
                block_key = frozenset(block_eqs)
                # Quadratischer Block: lösen; überbestimmter Block: quadratisches Teilsystem
                # lösen, die übrigen Gleichungen bleiben als Prüfung (Constraints) offen
                if len(block_eqs) >= len(block_vars) and block_key not in failed_blocks:
                    if len(block_eqs) == len(block_vars):
                        success, block_solution, block_msg, block_analysis = _solve_equation_block(
                            block_eqs, block_vars, known_values, context, initial_values,
                            original_equations, inferred_units
                        )
                    else:
                        (success, block_solution, block_msg, block_analysis,
                         solved_eqs) = _solve_overdetermined_block(
                            block_eqs, block_vars, known_values, context, initial_values,
                            original_equations, inferred_units)
                        if success:
                            # nur der überbestimmte Kern ist gelöst; Prüf- und Folgegleichungen
                            # bleiben offen (Constraints bzw. nächste Runde)
                            block_eqs = solved_eqs
                            block_vars = set(block_solution)
                    if not success:
                        failed_blocks.add(block_key)
                        # Block scheiterte an einem Kern mit abhängigen Gleichungen: "nicht eindeutig"
                        for dependency in _DEPENDENT_CORES:
                            if dependency not in non_unique:
                                non_unique.append(dependency)
                        _DEPENDENT_CORES.clear()
                        # Auswertungsfehler der 1-D-Suchen im Block (Tearing) wie bei Einzelgleichungen
                        # melden: ein Stoffwert außerhalb seines Bereichs ist die Ursache, nicht der Block
                        for eq in block_eqs:
                            if eq in _LAST_SEARCH_ERROR and eq not in evaluation_errors:
                                evaluation_errors[eq] = _LAST_SEARCH_ERROR[eq]
                        if block_solution:
                            # Unabhängig gelöste Kerne übernehmen; der Rest bildet einen
                            # neuen, kleineren Block (Meldung nennt dann nur ihn)
                            known_values.update(block_solution)
                            remaining_vars -= set(block_solution)
                            for eq in block_eqs:
                                if (eq in remaining_equations
                                        and not _get_equation_unknowns(eq, set(known_values), remaining_vars)
                                        and _relative_residual(eq, known_values, context) < 1e-6):
                                    remaining_equations.remove(eq)
                            if block_analysis is not None:
                                for eq_info in block_analysis.direct_evals:
                                    analysis.direct_evals.append(eq_info)
                                for eq_info in block_analysis.single_unknowns:
                                    analysis.single_unknowns.append(eq_info)
                                for sub_block in block_analysis.sub_blocks:
                                    sub_block.block_number = len(analysis.blocks) + 1
                                    analysis.blocks.append(sub_block)
                            made_progress = True
                            break

                    if success:
                        # Aktualisiere bekannte Werte
                        known_values.update(block_solution)
                        remaining_vars -= block_vars
                        # Eindeutig? (linear abhängige Gleichungen bzw. Plateau -> beliebiger Punkt)
                        try:
                            dependency = (_non_unique_solution(list(block_eqs), set(block_vars),
                                                               known_values, context)
                                          if len(block_vars) > 1 else
                                          _plateau_solution(list(block_eqs), next(iter(block_vars)),
                                                            known_values, context))
                        except Exception:
                            dependency = None
                        if dependency is not None:
                            non_unique.append(dependency)

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

                        if block_analysis is not None and (block_analysis.sub_blocks
                                                           or block_analysis.direct_evals
                                                           or block_analysis.single_unknowns):
                            # intern zerlegt: nur die echten gekoppelten Kerne zählen als Block
                            stats['blocks'].extend(len(sb.variables) for sb in block_analysis.sub_blocks)
                            stats['direct'] += len(block_analysis.direct_evals)
                            stats['single'] += len(block_analysis.single_unknowns)
                        else:
                            stats['blocks'].append(len(block_vars))
                        made_progress = True
                        break  # Nach gelöstem Block: zurück zu den sequentiellen Phasen

        if not made_progress:
            # Keine weitere Fortschritte möglich
            break

    # Ergebnis zusammenstellen
    result = known_values.copy()

    # Erstelle Statusmeldung
    if not remaining_equations:
        # Linear abhängige Gleichungen: der Solver hat einen beliebigen Punkt gefunden
        if non_unique:
            msg = _non_unique_message(non_unique[0], original_equations)
            if return_analysis:
                return False, result, msg, analysis
            return False, result, msg

        # Widersprüchliche Constraints -> KEIN Erfolg melden
        if violated_constraints:
            details = '; '.join(violated_constraints[:3])
            msg = (f"Widersprüchliches System (überbestimmt): {details}. "
                   f"Eine der vorgegebenen Größen muss stattdessen berechnet werden "
                   f"(Werte in SI-Einheiten).")
            if return_analysis:
                return False, result, msg, analysis
            return False, result, msg

        parts = []
        if stats['direct'] > 0:
            parts.append(f"{stats['direct']} direkt")
        if stats['single'] > 0:
            parts.append(f"{stats['single']} iterativ")
        if stats['blocks']:
            n_blocks = len(stats['blocks'])
            sizes = '+'.join(str(b) for b in stats['blocks'])
            parts.append(f"{n_blocks} {'Block' if n_blocks == 1 else 'Blöcke'} "
                         f"({sizes} {'Größe' if stats['blocks'] == [1] else 'Größen'})")

        msg = "Lösung gefunden"
        if parts:
            msg += f" ({', '.join(parts)})"

        if return_analysis:
            return True, result, msg, analysis
        return True, result, msg
    else:
        # Nicht alle Gleichungen gelöst
        msg = f"Unvollständig: {len(remaining_equations)} Gleichungen, {len(remaining_vars)} Unbekannte verbleibend"
        # Fehler bei der Auswertung noch offener Gleichungen (z.B. 0/0, CoolProp-Fehler)
        open_errors = _grouped_errors({eq: text for eq, text in evaluation_errors.items()
                                       if eq in remaining_equations}, original_equations)
        analysis.evaluation_errors = open_errors
        if open_errors:
            msg += ". Auswertungsfehler: " + "; ".join(open_errors[:3])
        # Erkannte Widersprüche bzw. abhängige Gleichungen gelöster Teile gehören auch bei einer
        # Teillösung in die Meldung (sonst nur die Folge, z.B. ein Auswertungsfehler danach)
        if non_unique:
            msg = _non_unique_message(non_unique[0], original_equations) + " " + msg
        if violated_constraints:
            msg = (f"Widersprüchliches System (überbestimmt): {'; '.join(violated_constraints[:3])}. "
                   + msg)
        if return_analysis:
            return False, result, msg, analysis
        return False, result, msg


def _grouped_errors(errors: Dict[str, str], original_equations: Dict[str, str]) -> List[str]:
    """
    Auswertungsfehler je Gleichung für die Meldung; dieselbe Ursache in mehreren Gleichungen (z.B.
    T außerhalb des Datenbereichs in allen Gleichungen einer Anweisung) einmal, mit allen Gleichungen.
    Der Text beginnt mit der gesuchten Größe ("x: ..."), die bei mehreren Gleichungen entfällt.
    """
    groups: Dict[str, List[Tuple[str, str]]] = {}
    for eq, text in errors.items():
        name, sep, cause = text.partition(': ')
        key = cause if sep and re.fullmatch(r'[A-Za-z_]\w*', name) else text
        groups.setdefault(key, []).append((eq, text))
    result = []
    for cause, members in groups.items():
        if len(members) == 1:
            eq, text = members[0]
            result.append(f"'{original_equations.get(eq, eq)}': {text}")
        else:
            shown = ", ".join(f"'{original_equations.get(eq, eq)}'" for eq, _ in members)
            result.append(f"{shown}: {cause}")
    return result


def _non_unique_message(dependency, original_equations: Dict[str, str]) -> str:
    """Meldung "Lösung nicht eindeutig" mit den abhängigen Gleichungen und den freien Größen."""
    dependent, free = dependency[0], dependency[1]
    lines = ", ".join(f"'{original_equations.get(eq, eq)}'" for eq in dependent)
    if len(dependency) > 2:     # Plateau: für einen ganzen Bereich einer Größe erfüllt
        return (f"Lösung nicht eindeutig: {lines} {'ist' if len(dependent) == 1 else 'sind'} für einen "
                f"ganzen Bereich von {free[0]} erfüllt "
                f"(flacher Verlauf, z.B. max/min oder eine Menge, die dort null bleibt) - {free[0]} ist "
                f"damit nicht bestimmt. Eine Vorgabe wählen, die {free[0]} festlegt.")
    return (f"Lösung nicht eindeutig: Die Gleichungen {lines} sind voneinander abhängig "
            f"(eine folgt aus den anderen) - {', '.join(free)} "
            f"{'ist' if len(free) == 1 else 'sind'} damit nicht bestimmt. Es fehlt eine "
            f"unabhängige Gleichung bzw. Vorgabe (z.B. eine Energie- statt einer zweiten "
            f"Massenbilanz).")


def _describe_violation(equation: str, original: str, values: Dict[str, float],
                        constants: Dict[str, float], context: dict) -> str:
    """
    Beschreibt eine verletzte Gleichung für die Fehlermeldung: beide Seiten mit
    Zahlenwert statt nur des Residuums, z.B.
    "q_dot ist vorgegeben (50), aus 'q_dot=U*A*dT' folgt 29.06".
    """
    import ast
    local_ctx = context.copy()
    local_ctx.update(values)

    def side_value(node):
        expr = ast.fix_missing_locations(ast.Expression(body=node))
        return float(_as_real(eval(compile(expr, '<side>', 'eval'), {"__builtins__": {}}, local_ctx)))

    try:
        tree = ast.parse(equation, mode='eval').body
        if not (isinstance(tree, ast.BinOp) and isinstance(tree.op, ast.Sub)):
            raise ValueError
        left_val, right_val = side_value(tree.left), side_value(tree.right)
    except Exception:
        residual = _calculate_residual(equation, values, context)
        return f"'{original}' verletzt (Residuum: {residual:.4g})"

    parts = _direct_assignment(equation)
    if parts and parts[0] in constants:
        return (f"{parts[0]} ist vorgegeben ({left_val:.6g}), "
                f"aus '{original}' folgt {right_val:.6g}")
    return f"'{original}' verletzt: links = {left_val:.6g}, rechts = {right_val:.6g}"


def _calculate_residual(equation: str, known_values: Dict[str, float], context: dict) -> float:
    """Berechnet das Residuum einer Gleichung mit den gegebenen Werten."""
    try:
        local_ctx = context.copy()
        local_ctx.update(known_values)
        result = _as_real(eval(equation, {"__builtins__": {}}, local_ctx))
        return float(result) if np.isfinite(result) else float('inf')
    except Exception:
        return float('inf')


def _additive_term_scale(expression: str, local_ctx: dict) -> Optional[float]:
    """
    Größenordnung einer Gleichung: das betragsgrößte additive Term-Ergebnis -
    auch innerhalb von Produkten und Quotienten.

    Für "(m1*h1 + m2*h2 - m3*h3) - (0)" ist die Skala max(|m1*h1|, |m2*h2|, |m3*h3|).
    Für "(Q_3) - (2*(0.4*(J_3 - J_1) + 0.4*(J_3 - J_2)))" zählen die inneren Terme
    (~0.8*|J|), nicht der am Lösungspunkt verschwindende Wert des Produkts.
    Damit lässt sich ein Residuum RELATIV zur Gleichungsgröße bewerten -
    unabhängig davon, ob mit J/kg (~1e6) oder Wirkungsgraden (~1) gerechnet wird.
    Gehen dagegen ALLE Terme gegen null (Asymptote, z.B. Eb(T, lambda) bei
    lambda -> unendlich), bleibt auch die Skala winzig: keine Scheinlösung.
    """
    import ast
    try:
        tree = ast.parse(expression.replace('^', '**'), mode='eval')
    except SyntaxError:
        return None

    def value(node):
        expr = ast.fix_missing_locations(ast.Expression(body=node))
        result = eval(compile(expr, '<term_scale>', 'eval'), {"__builtins__": {}}, local_ctx)
        result = _as_real(result)
        if not isinstance(result, (int, float, np.floating, np.integer)) or not np.isfinite(result):
            raise ValueError
        return abs(float(result))

    def scale(node):
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            return max(scale(node.left), scale(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return scale(node.operand)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            return max(value(node.left) * scale(node.right), scale(node.left) * value(node.right))
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            divisor = value(node.right)
            return scale(node.left) / divisor if divisor > 0 else value(node)
        return value(node)

    try:
        best = scale(tree.body)
    except Exception:
        # Teilausdruck nicht auswertbar: nur die äußeren Terme bewerten
        best = 0.0
        terms = []

        def collect(node):
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
                collect(node.left)
                collect(node.right)
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                collect(node.operand)
            else:
                terms.append(node)

        collect(tree.body)
        for term in terms:
            try:
                best = max(best, value(term))
            except Exception:
                pass

    return best if np.isfinite(best) and best > 0 else None


def _residual_and_scale(equation: str, values: Dict[str, float], context: dict) -> Tuple[float, float]:
    """(|Residuum|, Größenordnung der Terme) einer Gleichung; (inf, 0) wenn nicht auswertbar."""
    local_ctx = context.copy()
    local_ctx.update(values)
    try:
        res = _as_real(eval(equation, {"__builtins__": {}}, local_ctx))
    except Exception:
        return float('inf'), 0.0
    if not np.isfinite(res):
        return float('inf'), 0.0
    scale = _additive_term_scale(equation, local_ctx)
    if scale is None or not np.isfinite(scale):
        scale = 0.0
    return abs(float(res)), scale


def _stuck_equations(equations: List[str], values: Dict[str, float], variables, context: dict,
                     magnitudes: Optional[Dict[str, float]] = None) -> List[str]:
    """
    Nur näherungsweise erfüllte Gleichungen eines Blocks, die sich durch einen kleinen relativen
    Schritt der Blockgrößen NICHT erfüllen ließen (leer: die Näherung ist glaubwürdig).
    Das Residuum relativ zur Termgröße reicht dafür nicht: wachsen Terme mit der Lösung und heben
    sich auf (W = a - min(a, b) auf dem Plateau b > a, a -> 1e3), wird W/Termgröße beliebig klein,
    obwohl die Gleichung verletzt ist und sich lokal nicht ändert - ebenso, wenn sich der
    Widerspruch auf zwei Gleichungen verteilt (n_D = min(n_a, n_s), W = n_a - n_D je zur Hälfte).
    Prüfung am linearisierten Block (skalenunabhängig, Zeilen relativ zur Termgröße, Spalten
    relativ zur Größe): der Schritt J*dx = -r (kleinste Quadrate) muss das Residuum beseitigen
    (Rest <= 10 %) und klein sein (|dx| <= 1e-4 relativ).
    """
    variables = list(variables)
    pairs = [_residual_and_scale(eq, values, context) for eq in equations]
    if not any(np.isfinite(res) and res > 1e-10 * max(scale, 1e-300) for res, scale in pairs):
        return []

    def signed(eq, local_values):
        local_ctx = context.copy()
        local_ctx.update(local_values)
        return float(_as_real(eval(eq, {"__builtins__": {}}, local_ctx)))

    scales = [max(scale, 1e-300) for _, scale in pairs]
    try:
        r = np.array([signed(eq, values) / sc for eq, sc in zip(equations, scales)])
    except Exception:
        return list(equations)
    jacobian = np.zeros((len(equations), len(variables)))
    for j, var in enumerate(variables):
        x = float(values[var])
        magnitude = abs(x) or abs((magnitudes or {}).get(var, 0.0)) or 1.0
        step = 1e-7 * magnitude
        plus, minus = dict(values), dict(values)
        plus[var], minus[var] = x + step, x - step
        for i, (eq, sc) in enumerate(zip(equations, scales)):
            try:
                derivative = (signed(eq, plus) - signed(eq, minus)) / (2 * step)
            except Exception:
                continue
            if np.isfinite(derivative):
                jacobian[i, j] = derivative * magnitude / sc
    if not np.all(np.isfinite(r)):
        return list(equations)
    dx = np.linalg.lstsq(jacobian, -r, rcond=None)[0]
    rest = np.abs(jacobian @ dx + r)
    if np.linalg.norm(rest) <= 0.1 * np.linalg.norm(r) and np.max(np.abs(dx), initial=0.0) <= 1e-4:
        return []
    return [eq for eq, value in zip(equations, rest) if value > 1e-10] or list(equations)


def _nearest_block_solution(solutions: List[Dict[str, float]], equations: List[str], variables,
                            known_values: Dict[str, float], manual_initial=None, inferred_units=None
                            ) -> Dict[str, float]:
    """
    Unter verschiedenen gefundenen Lösungen eines Blocks die nächstgelegene - dieselbe Regel wie
    beim Tearing (Festlegung 11.10.2026): Abstand ALLER Blockgrößen zu ihren Startwerten, je relativ
    zum Startwert (ohne Strukturhinweis, wie dort); Gleichstand: größere Werte in Blattreihenfolge.
    Sonst entschiede die erste konvergierte Startvariante und damit die Reihenfolge der Gleichungen
    (Gleichgewichte mit negativen Stoffmengen statt der physikalischen, näheren Wurzel).
    """
    if len(solutions) == 1:
        return solutions[0]
    appearance = _appearance_order(equations, set(variables))
    names = sorted(variables, key=lambda v: appearance[v])
    anchors = {v: _get_initial_value(v, manual_initial, known_values, inferred_units, use_structure=False)
               for v in names}
    ranked = []
    for solution in solutions:
        vector = tuple(float(solution[v]) for v in names)
        distance = float(np.sqrt(sum(((x - anchors[v]) / (abs(anchors[v]) or 1.0)) ** 2
                                     for v, x in zip(names, vector))))
        ranked.append((distance, vector, solution))
    nearest = min(d for d, _, _ in ranked)
    tie = 1e-9 * max(nearest, 1.0)
    return max((item for item in ranked if item[0] <= nearest + tie), key=lambda item: item[1])[2]


def _distinct(solutions: List[Dict[str, float]], solution: Dict[str, float]) -> bool:
    """Ist die Lösung von allen bisherigen verschieden (relativ 1e-6 in mindestens einer Größe)?"""
    return all(any(abs(solution[v] - other[v]) > 1e-6 * max(abs(other[v]), abs(solution[v]), 1e-300)
                   for v in solution) for other in solutions)


def _polish_block_solution(equations: List[str], values: Dict[str, float], variables, context: dict,
                           magnitudes: Optional[Dict[str, float]] = None, steps: int = 3) -> Dict[str, float]:
    """
    Newton-Nachiteration einer akzeptierten Blocklösung am normierten System (Zeilen relativ zur
    Termgröße, Spalten relativ zur Größe). least_squares bzw. fsolve enden bei einem relativen
    Residuum um 1e-8 - Folgegrößen (Zielgröße einer Optimierung) erben diesen Fehler, und der
    Optimierer "findet" scheinbar bessere Werte, die Rechenfehler sind. Ein Schritt zählt nur,
    wenn das größte relative Residuum sinkt. Gibt die (verbesserten) Werte der Blockgrößen zurück.
    """
    variables = list(variables)
    current = dict(values)

    def normalized(local_values):
        pairs = [_residual_and_scale(eq, local_values, context) for eq in equations]
        if any(not np.isfinite(res) for res, _ in pairs):
            return None, None
        scales = [max(scale, 1e-300) for _, scale in pairs]
        signed = []
        for eq, sc in zip(equations, scales):
            local_ctx = context.copy()
            local_ctx.update(local_values)
            signed.append(float(_as_real(eval(eq, {"__builtins__": {}}, local_ctx))) / sc)
        return np.array(signed), scales

    try:
        r, scales = normalized(current)
        if r is None:
            return {var: current[var] for var in variables}
        for _ in range(steps):
            worst = float(np.max(np.abs(r), initial=0.0))
            if worst <= 1e-13:
                break
            jacobian = np.zeros((len(equations), len(variables)))
            magnitude = []
            for j, var in enumerate(variables):
                x = float(current[var])
                mag = abs(x) or abs((magnitudes or {}).get(var, 0.0)) or 1.0
                magnitude.append(mag)
                step = 1e-7 * mag
                plus, minus = dict(current), dict(current)
                plus[var], minus[var] = x + step, x - step
                for i, (eq, sc) in enumerate(zip(equations, scales)):
                    ctx_p, ctx_m = context.copy(), context.copy()
                    ctx_p.update(plus)
                    ctx_m.update(minus)
                    derivative = (float(_as_real(eval(eq, {"__builtins__": {}}, ctx_p)))
                                  - float(_as_real(eval(eq, {"__builtins__": {}}, ctx_m)))) / (2 * step)
                    jacobian[i, j] = derivative * mag / sc if np.isfinite(derivative) else 0.0
            dx = np.linalg.lstsq(jacobian, -r, rcond=None)[0]
            trial = dict(current)
            for j, var in enumerate(variables):
                trial[var] = float(current[var]) + dx[j] * magnitude[j]
            r_trial, _ = normalized(trial)
            if r_trial is None or not float(np.max(np.abs(r_trial), initial=0.0)) < worst:
                break
            current, r = trial, r_trial
    except Exception:
        pass
    return {var: current[var] for var in variables}


def _relative_residual(equation: str, values: Dict[str, float], context: dict,
                       scale_floor: float = 0.0) -> float:
    """
    Residuum einer Gleichung relativ zur Größenordnung ihrer Terme.

    WICHTIG: NICHT relativ zur Größe der LÖSUNG normieren - sonst würde
    Divergenz zur Asymptote (z.B. 1/(x-2) = 0 mit x -> unendlich) als
    "Lösung" akzeptiert, weil das Residuum durch |x| geteilt winzig wird.
    """
    local_ctx = context.copy()
    local_ctx.update(values)
    try:
        res = _as_real(eval(equation, {"__builtins__": {}}, local_ctx))
    except Exception:
        return float('inf')
    if not np.isfinite(res):
        return float('inf')

    scale = _additive_term_scale(equation, local_ctx)
    if scale is None or not np.isfinite(scale):
        scale = 1.0
    # Kein großzügiger Floor: Wenn ALLE Terme winzig sind (z.B. 1/(x-2) bei
    # x=1e83), muss das Residuum RELATIV zu diesen winzigen Termen klein sein -
    # sonst wird Divergenz zur Asymptote als Lösung akzeptiert.
    # scale_floor: Untergrenze aus dem Umfeld (andere Gleichungen des Blocks,
    # bekannte Werte) - sonst gilt z.B. "q_x = eps*y" mit eps = 0 und
    # q_x = 1e-17 (numerisch null) als um 100 % verletzt.
    return abs(float(res)) / max(scale, scale_floor, 1e-300)


def _get_eval_context():
    """Erstellt den Kontext für eval mit allen verfügbaren Funktionen."""
    context = {
        'sin': sin, 'cos': cos, 'tan': tan,
        'asin': asin, 'acos': acos, 'atan': atan,
        'sinh': sinh, 'cosh': cosh, 'tanh': tanh,
        'exp': exp, 'log': log, 'log10': log10,
        'sqrt': sqrt, 'abs': np.abs, 'pi': pi,
            'ceil': np.ceil, 'floor': np.floor, 'round': np.round,
        'max': max, 'min': min, 'IF': if_function,
        'value': unit_number, 'quantity': unit_quantity
    }
    if THERMO_AVAILABLE:
        context.update(THERMO_FUNCTIONS)
    if EQUILIBRIUM_AVAILABLE:
        context['equilibrium'] = _equilibrium_function
    if RADIATION_AVAILABLE:
        context.update(RADIATION_FUNCTIONS)
    if HUMID_AIR_AVAILABLE:
        context.update(HUMID_AIR_FUNCTIONS)
    return context


def _get_equation_unknowns(equation: str, known_vars: Set[str], all_vars: Set[str]) -> Set[str]:
    """Findet die Unbekannten in einer Gleichung."""
    import re
    # String-Literale entfernen ('water' darf nicht als Variable zählen)
    cleaned = re.sub(r"'[^']*'|\"[^\"]*\"", ' ', equation)
    # Keyword-Argument-NAMEN entfernen (in "HumidAir('h', T=T_1)" ist T der
    # Parametername, nicht die Nutzervariable T)
    cleaned = re.sub(r'\b[a-zA-Z_][a-zA-Z0-9_]*\s*=(?!=)', ' ', cleaned)
    found_vars = set(re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', cleaned))
    # Filtere auf die tatsächlichen Variablen
    return (found_vars & all_vars) - known_vars


def _smallest_independent_component(equations: List[str], eq_to_unknowns: Dict[str, Set[str]],
                                    appearance: Dict[str, int]) -> Optional[List[str]]:
    """
    Kleinste starke Zusammenhangskomponente des Abhängigkeitsgraphen, die von keiner anderen
    abhängt (Block-Dreiecksform: Zuordnung Gleichung <-> Unbekannte, Tarjan). None, wenn die
    Gleichungen keine vollständige Zuordnung zu ihren Unbekannten erlauben. Gleichstand der
    Größe: die im Blatt zuerst stehende Komponente.
    """
    unknowns = set()
    for eq in equations:
        unknowns.update(eq_to_unknowns[eq])
    if len(unknowns) != len(equations):
        return None
    owner: Dict[str, str] = {}

    def augment(eq, seen):
        for var in sorted(eq_to_unknowns[eq], key=lambda v: appearance.get(v, 0)):
            if var in seen:
                continue
            seen.add(var)
            if var not in owner or augment(owner[var], seen):
                owner[var] = eq
                return True
        return False

    for eq in equations:
        if not augment(eq, set()):
            return None
    depends = {eq: [owner[var] for var in eq_to_unknowns[eq] if owner[var] != eq] for eq in equations}

    # Tarjan (iterativ): Komponenten in umgekehrter topologischer Reihenfolge
    index: Dict[str, int] = {}
    low: Dict[str, int] = {}
    on_stack: Set[str] = set()
    stack: List[str] = []
    components: List[List[str]] = []
    counter = 0
    for root in equations:
        if root in index:
            continue
        work = [(root, iter(depends[root]))]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            node, successors = work[-1]
            advanced = False
            for succ in successors:
                if succ not in index:
                    index[succ] = low[succ] = counter
                    counter += 1
                    stack.append(succ)
                    on_stack.add(succ)
                    work.append((succ, iter(depends[succ])))
                    advanced = True
                    break
                if succ in on_stack:
                    low[node] = min(low[node], index[succ])
            if advanced:
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[node])
            if low[node] == index[node]:
                component = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                components.append(component)
    component_of = {eq: i for i, component in enumerate(components) for eq in component}
    position = {eq: i for i, eq in enumerate(equations)}
    independent = [component for i, component in enumerate(components)
                   if all(component_of[dep] == i for eq in component for dep in depends[eq])]
    if not independent:
        return None
    best = min(independent, key=lambda c: (len(c), min(position[eq] for eq in c)))
    return [eq for eq in equations if eq in set(best)]


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
    appearance = _appearance_order(equations, variables)

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

                for var in sorted(block_vars, key=lambda v: appearance[v]):
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

    # Exakt: Block-Dreiecksform. Jede Gleichung bekommt eine Unbekannte zugeordnet; Gleichung i
    # hängt von Gleichung j ab, wenn i die Unbekannte von j enthält. Die starken
    # Zusammenhangskomponenten dieses Graphen sind die gekoppelten Kerne (für quadratische
    # Systeme eindeutig, unabhängig von Namen und Reihenfolge); ein Kern ohne Abhängigkeit von
    # anderen hängt nur von bekannten Größen ab. Die gierige Erweiterung unten konnte
    # nachgelagerte Gleichungen in den Kern ziehen (Anode + Nachbrenner: 15 statt 11 Größen).
    core = _smallest_independent_component(equations, eq_to_unknowns, appearance)
    if core is not None:
        core_vars = set()
        for eq in core:
            core_vars.update(eq_to_unknowns[eq])
        return core, core_vars

    # Sonst (keine vollständige Zuordnung): kleinster unabhängig lösbarer Block, gierig
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

    # Gruppiere Gleichungen die gemeinsame Variablen haben.
    # WICHTIG: Reihenfolge = Eingabereihenfolge (Listen statt Sets). Mit
    # Set-Iteration hing die Gleichungsreihenfolge im Block - und damit, ob
    # least_squares konvergiert - vom zufälligen Hash-Seed des Prozesses ab:
    # dasselbe Eingabeblatt wurde mal gelöst, mal nicht.
    remaining_eqs = list(eq_to_vars.keys())
    blocks = []

    while remaining_eqs:
        current_block_eqs = set()
        current_block_vars = set()

        # Nimm erste verfügbare Gleichung (in Eingabereihenfolge)
        to_process = [remaining_eqs[0]]

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
                for other_eq in remaining_eqs:
                    if other_eq not in current_block_eqs and eq_to_vars[other_eq] & new_vars:
                        to_process.append(other_eq)

        # Block gefunden (Gleichungen in Eingabereihenfolge)
        block_list = [eq for eq in remaining_eqs if eq in current_block_eqs]
        remaining_eqs = [eq for eq in remaining_eqs if eq not in current_block_eqs]
        blocks.append((block_list, current_block_vars))

    # Sortiere nach Blockgröße (kleinste zuerst für bessere Konvergenz; stabil)
    blocks.sort(key=lambda b: len(b[1]))

    return blocks


def _direct_assignment(equation: str) -> Optional[Tuple[str, str]]:
    """
    Zerlegt eine Gleichung der Form "(var) - (ausdruck)" in (var, ausdruck),
    sonst None. Der Parser erzeugt Gleichungen immer als "({links}) - ({rechts})".
    """
    import re
    match = re.match(r'^\(([A-Za-z_][A-Za-z0-9_]*)\) - \(', equation)
    if not match or not equation.endswith(')'):
        return None
    var = match.group(1)
    return var, equation[len(f"({var}) - ("):-1]


class _LinearStep:
    """
    Kettenschritt für eine Gleichung, in der die gesuchte Größe nur linear vorkommt
    (eta = (h_1 - h_2)/(h_1 - h_2s) nach h_2, m_1 = m_7 + m_9 + m_3 nach m_3): aus zwei
    Auswertungen r(0), r(1) folgt exakt var = -r(0)/(r(1) - r(0)).
    """

    def __init__(self, var: str, equation: str):
        self.var = var
        self.equation = equation

    def value(self, local_ctx: dict) -> float:
        ctx = dict(local_ctx)
        ctx[self.var] = 0.0
        r0 = _as_real(eval(self.equation, {"__builtins__": {}}, ctx))
        ctx[self.var] = 1.0
        r1 = _as_real(eval(self.equation, {"__builtins__": {}}, ctx))
        if r1 == r0 or not (np.isfinite(r0) and np.isfinite(r1)):
            raise ZeroDivisionError("Gleichung hängt hier nicht von der Größe ab")
        return -r0 / (r1 - r0)


def _chain_eval(step, local_ctx: dict):
    """Wert eines Kettenschritts: Ausdruck (var = ausdruck) oder linear implizite Gleichung."""
    if isinstance(step, _LinearStep):
        return step.value(local_ctx)
    return eval(step, {"__builtins__": {}}, local_ctx)


def _linearity(node, var: str) -> str:
    """'none' (var kommt nicht vor), 'linear' oder 'nonlinear' - strukturell am Syntaxbaum."""
    import ast
    if isinstance(node, ast.Name):
        return 'linear' if node.id == var else 'none'
    if isinstance(node, ast.Constant):
        return 'none'
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _linearity(node.operand, var)
    if isinstance(node, ast.BinOp):
        left, right = _linearity(node.left, var), _linearity(node.right, var)
        if 'nonlinear' in (left, right):
            return 'nonlinear'
        if isinstance(node.op, (ast.Add, ast.Sub)):
            return 'linear' if 'linear' in (left, right) else 'none'
        if isinstance(node.op, ast.Mult):
            if left == 'linear' and right == 'linear':
                return 'nonlinear'
            return 'linear' if 'linear' in (left, right) else 'none'
        if isinstance(node.op, ast.Div):
            if right == 'linear':
                return 'nonlinear'
            return left
        return 'nonlinear' if 'linear' in (left, right) else 'none'
    return 'nonlinear' if any(isinstance(n, ast.Name) and n.id == var for n in ast.walk(node)) else 'none'


_LINEAR_CACHE: Dict[Tuple[str, str], bool] = {}


def _is_linear_in(equation: str, var: str) -> bool:
    """Kommt var in der Gleichung (Residuum) nur linear vor?"""
    key = (equation, var)
    if key not in _LINEAR_CACHE:
        import ast
        try:
            _LINEAR_CACHE[key] = _linearity(ast.parse(equation, mode='eval').body, var) == 'linear'
        except Exception:
            _LINEAR_CACHE[key] = False
        if len(_LINEAR_CACHE) > 20000:
            _LINEAR_CACHE.clear()
    return _LINEAR_CACHE[key]


def _find_tear_candidates(
    equations: List[str],
    variables: Set[str],
    known_vars: Set[str],
    allow_linear: bool = False
) -> List[Tuple[str, List[Tuple[str, str]], str]]:
    """
    Sucht Tearing-Variablen für einen gekoppelten Block.

    Eine Tearing-Variable v erfüllt: Wird v als bekannt angenommen, lassen sich
    alle übrigen Blockvariablen der Reihe nach DIREKT berechnen ("var = ausdruck"),
    und genau eine Gleichung bleibt als Residuum übrig. Der Block reduziert sich
    damit auf EINE Gleichung in v, die mit der robusten Bracket-Suche gelöst wird.

    Typisch für Wärmeübertragung: Filmtemperatur-Iteration (T_s schätzen ->
    T_f -> Stoffwerte -> Ra -> Nu -> Wärmestrom = Vorgabe).

    Returns:
        Liste von (tear_var, [(var, ausdruck), ...], residuum_gleichung),
        bevorzugt Variablen, die in vielen Gleichungen vorkommen.
    """
    variables = set(variables)
    eq_unknowns = {eq: _get_equation_unknowns(eq, known_vars, variables) for eq in equations}
    direct = {}
    for eq in equations:
        parts = _direct_assignment(eq)
        if parts and parts[0] in variables:
            expr_unknowns = _get_equation_unknowns(parts[1], known_vars, variables)
            if parts[0] not in expr_unknowns:
                direct[eq] = (parts[0], parts[1], expr_unknowns)

    counts = {v: sum(1 for eq in equations if v in eq_unknowns[eq]) for v in variables}
    appearance = _appearance_order(equations, variables)
    candidates = []
    for tear in sorted(variables, key=lambda v: (-counts[v], appearance[v])):
        determined = {tear}
        sequence = []
        unused = list(equations)
        progress = True
        while progress:
            progress = False
            for eq in unused:
                if eq not in direct:
                    continue
                var, expr, deps = direct[eq]
                if var not in determined and deps <= determined:
                    sequence.append((var, expr))
                    determined.add(var)
                    unused.remove(eq)
                    progress = True
                    break
            if progress or not allow_linear or len(unused) <= 1:
                continue
            # sonst: Gleichung mit genau einer offenen Größe, die nur linear vorkommt
            for eq in unused:
                open_vars = eq_unknowns[eq] - determined
                if len(open_vars) == 1 and _is_linear_in(eq, next(iter(open_vars))):
                    var = next(iter(open_vars))
                    sequence.append((var, _LinearStep(var, eq)))
                    determined.add(var)
                    unused.remove(eq)
                    progress = True
                    break
        if determined == variables and len(unused) == 1:
            candidates.append((tear, sequence, unused[0]))
    return candidates


def _solve_block_by_tearing(
    equations: List[str],
    variables: Set[str],
    known_values: Dict[str, float],
    context: dict,
    manual_initial: Optional[Dict[str, float]] = None,
    inferred_units: Optional[Dict[str, str]] = None,
    max_candidates: int = 3
) -> Tuple[bool, Dict[str, float], str]:
    """Löst einen Block über eine Tearing-Variable (siehe _find_tear_candidates)."""
    known = set(known_values.keys())
    # Zuerst nur explizite Zuweisungen (var = ausdruck); danach Ketten, in denen Gleichungen
    # mit genau einer offenen, nur linear vorkommenden Größe exakt aufgelöst werden
    # (natürliche implizite Schreibweise, z.B. eta = (h_1 - h_2)/(h_1 - h_2s))
    explicit = _find_tear_candidates(equations, variables, known)[:max_candidates]
    with_linear = [c for c in _find_tear_candidates(equations, variables, known, allow_linear=True)
                   if any(isinstance(step, _LinearStep) for _, step in c[1])][:max_candidates]
    # Mehrere Lösungen: Wahl nach dem Abstand ALLER Blockgrößen zu ihren Startwerten - nicht nur
    # der Tearing-Größe, sonst hinge die Wahl von der Schreibweise ab (u = v^2 mit v = u - 6
    # bzw. u = v + 6 riss u bzw. v auf: (4; -2) bzw. (9; 3))
    appearance = _appearance_order(equations, variables)
    block_vars = sorted(variables, key=lambda v: appearance[v])
    anchors = {v: _get_initial_value(v, manual_initial, known_values, inferred_units, use_structure=False)
               for v in block_vars}
    for tear, sequence, residual_eq in explicit + with_linear:
        def chain(x, tear=tear, sequence=sequence):
            values = dict(known_values)
            values[tear] = x
            local_ctx = context.copy()
            local_ctx.update(values)
            for var, expr in sequence:
                value = _chain_eval(expr, local_ctx)
                values[var] = value
                local_ctx[var] = value
            return values

        success, x = _solve_single_unknown(
            residual_eq, tear, known_values, context, manual_initial, inferred_units, chain=chain,
            block_anchors=(block_vars, anchors)
        )
        if not success:
            continue
        try:
            values = chain(x)
            solution = {var: float(values[var]) for var in variables}
        except Exception:
            continue
        check_values = dict(known_values)
        check_values.update(solution)
        if all(_relative_residual(eq, check_values, context) < 1e-6 for eq in equations):
            return True, solution, f"Block per Tearing gelöst ({tear})"
    return False, {}, "Tearing nicht möglich"


def _find_tear_set(
    equations: List[str],
    variables: Set[str],
    known_vars: Set[str]
) -> Optional[Tuple[List[str], List[Tuple[str, str]], List[str]]]:
    """
    Sucht eine kleine Menge von Tearing-Variablen (k >= 1): werden sie geschätzt,
    lassen sich alle übrigen Blockvariablen der Reihe nach direkt berechnen;
    genau k Gleichungen bleiben als Residuen. Der Block reduziert sich so auf
    ein k-dimensionales System (z.B. 12 -> 4 bei Radiositätsnetz + Stoffwerten
    bei Filmtemperatur), und alle Zwischengrößen sind stets konsistent.

    Greedy (generisch): zuerst Variablen ohne eigene Bestimmungsgleichung
    ("var = ausdruck" ohne var rechts), bevorzugt häufig vorkommende.

    Returns:
        (tear_vars, [(var, ausdruck), ...], residuum_gleichungen) oder None
    """
    variables = set(variables)
    direct = {}
    for eq in equations:
        parts = _direct_assignment(eq)
        if parts and parts[0] in variables:
            deps = _get_equation_unknowns(parts[1], known_vars, variables)
            if parts[0] not in deps:
                direct[eq] = (parts[0], parts[1], deps)
    has_direct = {var for var, _, _ in direct.values()}
    counts = {v: sum(1 for eq in equations if v in _get_equation_unknowns(eq, known_vars, {v}))
              for v in variables}
    appearance = _appearance_order(equations, variables)

    tears, determined, sequence, used = [], set(), [], set()

    def propagate():
        progress = True
        while progress:
            progress = False
            for eq in equations:
                if eq in used or eq not in direct:
                    continue
                var, expr, deps = direct[eq]
                if var not in determined and deps <= determined:
                    sequence.append((var, expr))
                    determined.add(var)
                    used.add(eq)
                    progress = True

    propagate()
    while determined != variables:
        open_vars = variables - determined
        tear = min(open_vars, key=lambda v: (v in has_direct, -counts[v], appearance[v]))
        tears.append(tear)
        determined.add(tear)
        propagate()
    residual_eqs = [eq for eq in equations if eq not in used]
    if len(residual_eqs) != len(tears):
        return None
    return tears, sequence, residual_eqs


def _solve_block_by_multi_tearing(
    equations: List[str],
    variables: Set[str],
    known_values: Dict[str, float],
    context: dict,
    manual_initial: Optional[Dict[str, float]] = None,
    inferred_units: Optional[Dict[str, str]] = None
) -> Tuple[bool, Dict[str, float], str]:
    """Löst einen Block über mehrere Tearing-Variablen (siehe _find_tear_set)."""
    import time
    import warnings
    from scipy.optimize import least_squares

    found = _find_tear_set(equations, variables, set(known_values.keys()))
    if found is None:
        return False, {}, "Tearing nicht möglich"
    tears, sequence, residual_eqs = found
    if len(tears) >= len(variables):
        return False, {}, "Tearing bringt keine Reduktion"

    deadline = time.monotonic() + 20.0

    def chain(x):
        values = dict(known_values)
        values.update(zip(tears, x))
        local_ctx = context.copy()
        local_ctx.update(values)
        for var, expr in sequence:
            value = _as_real(_chain_eval(expr, local_ctx))
            values[var] = value
            local_ctx[var] = value
        return values

    extra_deadline = [None]   # Zeitgrenze der Suche nach weiteren Lösungen (nach der ersten)

    class _ExtraSearchOver(BaseException):
        """Suche nach weiteren Lösungen zu lang (auch mitten in einem least_squares-Lauf)."""

    def raw_residuals(x):
        _check_deadline()
        if extra_deadline[0] is not None and time.monotonic() > extra_deadline[0]:
            raise _ExtraSearchOver()
        values = chain(x)
        local_ctx = context.copy()
        local_ctx.update(values)
        out = []
        for eq in residual_eqs:
            try:
                value = _as_real(eval(eq, {"__builtins__": {}}, local_ctx))
            except Exception:
                value = np.nan
            out.append(value if np.isfinite(value) else 1e10)
        return np.array(out, dtype=float)

    def make_residuals(weights):
        def residuals(x_norm):
            try:
                return raw_residuals(x_norm * scales) / weights
            except Exception:
                return np.full(len(residual_eqs), 1e10)
        return residuals

    def start_weights(x):
        """
        Feste Gewichte je Gleichung = Termgröße am Startpunkt (Gleichungen in
        W/m² und in K vergleichbar gewichtet). Fest, nicht mitlaufend: eine
        mitlaufende Normierung macht das Problem nicht-glatt, wenn Terme gegen
        null gehen (z.B. adiabate Wand: alle J gleich -> Termgröße 0).
        """
        try:
            values = chain(x)
        except Exception:
            return None
        found_scales = [_residual_and_scale(eq, values, context)[1] for eq in residual_eqs]
        top = max((sc for sc in found_scales if np.isfinite(sc)), default=0.0)
        if not top > 0:
            return None
        return np.array([sc if np.isfinite(sc) and sc > 1e-6 * top else 1e-6 * top
                         for sc in found_scales], dtype=float)

    x0 = np.array([_get_initial_value(v, manual_initial, known_values, inferred_units) for v in tears],
                  dtype=float)
    scales = np.maximum(np.abs(x0), 1e-10)
    starts = [x0 / scales]
    # Gleichartige Größen mit identischem Startwert: zusätzlich gestaffelt starten (in der
    # Reihenfolge im Blatt ab- und aufsteigend, nicht nach Namen), sonst sind Differenzen
    # wie T_1 - T_2 am Start exakt null (Ra = 0, (9000/Ra) -> inf)
    appearance = _appearance_order(equations, tears)
    for direction in (-1.0, 1.0):
        staggered = x0.copy()
        for value in set(x0.tolist()):
            same = sorted((i for i in range(len(tears)) if x0[i] == value), key=lambda i: appearance[tears[i]])
            if len(same) > 1:
                for rank, i in enumerate(same):
                    staggered[i] = value * (1.0 + direction * 0.003 * rank)
        if not np.array_equal(staggered, x0):
            starts.append(staggered / scales)
    starts += [np.ones(len(tears)) / scales, 0.5 * np.ones(len(tears)) / scales,
               1.5 * x0 / scales, 0.5 * x0 / scales, 2.0 * x0 / scales]
    attempts = []
    solutions: List[Dict[str, float]] = []
    for start in starts:
        attempts.append((start, None))      # absolute Residuen
        attempts.append((start, 'start'))   # fest gewichtet mit Termgrößen am Start
    converged_starts = set()
    began = time.monotonic()
    for start, weighting in attempts:
        if time.monotonic() > deadline:
            break
        if extra_deadline[0] is not None and time.monotonic() > extra_deadline[0]:
            break
        if id(start) in converged_starts:
            continue   # dieser Start hat schon eine Lösung - die andere Gewichtung findet dieselbe
        weights = np.ones(len(residual_eqs))
        if weighting == 'start':
            weights = start_weights(start * scales)
            if weights is None:
                continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # nach der ersten Lösung nur kurze Suchen nach weiteren
                result = least_squares(make_residuals(weights), start, method='lm', xtol=1e-14, ftol=1e-14,
                                       max_nfev=(60 if solutions else 300) * len(tears))
            values = chain(result.x * scales)
            solution = {var: float(values[var]) for var in variables}
        except _ExtraSearchOver:
            break
        except Exception:
            continue
        if not all(np.isfinite(v) for v in solution.values()):
            continue
        check_values = dict(known_values)
        check_values.update(solution)
        pairs = [_residual_and_scale(eq, check_values, context) for eq in equations]
        worst = max(res / max(sc, 1e-300) for res, sc in pairs)
        if worst < 1e-8 and not _stuck_equations(equations, check_values, variables, context):
            solution = _polish_block_solution(equations, check_values, sorted(variables), context)
            converged_starts.add(id(start))
            if _distinct(solutions, solution):
                solutions.append(solution)
            if extra_deadline[0] is None:
                # weitere Lösungen (für die Wahl der nächstgelegenen) höchstens zehnmal so lange suchen
                # wie bis zur ersten (mindestens 2 s): billige Blöcke durchsuchen alle Startvarianten,
                # teure (Gleichgewicht je Auswertung ~50 ms) liefen sonst je Variante 20 s
                extra_deadline[0] = time.monotonic() + max(10.0 * (time.monotonic() - began), 2.0)
    if solutions:
        # mehrere Lösungen: die den Startwerten nächste (nicht die erste konvergierte Variante)
        solution = _nearest_block_solution(solutions, equations, variables, known_values,
                                           manual_initial, inferred_units)
        return True, solution, f"Block per Tearing gelöst ({len(tears)} von {len(variables)} Variablen)"
    return False, {}, "Tearing-Iteration nicht konvergiert"


def _solve_core(
    equations: List[str],
    variables: Set[str],
    known_values: Dict[str, float],
    context: dict,
    manual_initial: Optional[Dict[str, float]] = None,
    inferred_units: Optional[Dict[str, str]] = None,
    attempted: Optional[Set[frozenset]] = None
) -> Tuple[bool, Dict[str, float], str]:
    """
    Löst einen gekoppelten Kern: zuerst per Tearing (robust gegen schlechte
    Startwerte, z.B. bei Stoffwert-Aufrufen mit unphysikalischen Werten),
    sonst simultan mit least_squares/fsolve.
    """
    if attempted is not None:
        attempted.add(frozenset(equations))
    if len(variables) >= 2:
        success, solution, msg = _solve_block_by_tearing(
            equations, variables, known_values, context, manual_initial, inferred_units
        )
        if success:
            return success, solution, msg
        success, solution, msg = _solve_block_by_multi_tearing(
            equations, variables, known_values, context, manual_initial, inferred_units
        )
        if success:
            return success, solution, msg
    return _solve_block_simultaneously(
        equations, variables, known_values, context, manual_initial, inferred_units
    )


def _plateau_solution(equations: List[str], var: str, values: Dict[str, float],
                      context: dict) -> Optional[Tuple[List[str], List[str]]]:
    """
    Größe auf einem Plateau: die Gleichungen sind am Lösungspunkt UND bei relativ verschobener Größe
    (1e-3) erfüllt (y = max(0, x - 5) + 3 mit y = 3; Kondensat n_L = 0 oberhalb des Taupunkts) - jeder
    Wert dort ist eine Lösung. Skalenfrei: erfüllt heißt Residuum <= 1e-12 der Termgröße ohne
    Untergrenze, und mindestens eine Gleichung hat Terme ungleich null - verschwinden alle Terme
    (Underflow: 1e-300*sin(y), exp(-x^2)), ist das kein Plateau. Bei x = 0 fehlt ein Maßstab für die
    Verschiebung: keine Aussage. Rückgabe wie _non_unique_solution.
    """
    x = float(values[var])
    if not np.isfinite(x) or x == 0:
        return None
    step = 1e-3 * abs(x)
    for shifted in (x + step, x - step):
        trial = dict(values)
        trial[var] = shifted
        if _all_satisfied(equations, trial, context) and _all_satisfied(equations, values, context):
            return list(equations), [var], 'plateau'
    return None


def _all_satisfied(equations: List[str], values: Dict[str, float], context: dict) -> bool:
    """Alle Gleichungen bis auf Rundung erfüllt (<= 1e-12 der Termgröße, ohne Untergrenze) und
    mindestens eine mit Termen ungleich null (sonst nur Underflow)."""
    nonzero = False
    for eq in equations:
        residual, scale = _residual_and_scale(eq, values, context)
        if not np.isfinite(residual) or residual > 1e-12 * scale:
            return False
        nonzero = nonzero or scale > 0
    return nonzero


def _non_unique_solution(equations: List[str], variables: Set[str], values: Dict[str, float],
                         context: dict) -> Optional[Tuple[List[str], List[str]]]:
    """
    Ist die Lösung eines gekoppelten Blocks eindeutig? Ist die Jacobi-Matrix am Lösungspunkt
    singulär UND bleiben alle Gleichungen erfüllt, wenn man entlang der Nullraum-Richtung
    weitergeht (eine Größe verschoben, die übrigen neu gelöst), gibt es unendlich viele
    Lösungen: eine Gleichung folgt aus den anderen (x + y = 1, 2*x + 2*y = 2). Isolierte
    Lösungen mit singulärer Matrix (Doppelwurzel) bestehen den Fortsetzungstest nicht.
    Generisch, numerisch, ohne Namen.

    Returns:
        (abhängige Gleichungen, nicht bestimmte Größen) oder None
    """
    from scipy.optimize import least_squares
    var_list = sorted(variables, key=lambda v: _appearance_order(equations, variables)[v])
    n = len(var_list)
    if n < 2 or len(equations) != n:
        return None
    x0 = np.array([float(values[v]) for v in var_list])
    if not np.all(np.isfinite(x0)):
        return None
    row_scale = []
    for eq in equations:
        res, scale = _residual_and_scale(eq, values, context)
        if not np.isfinite(res):
            return None
        row_scale.append(scale if scale > 0 else 1.0)
    row_scale = np.array(row_scale)

    def residuals(x):
        trial = dict(values)
        trial.update(zip(var_list, x))
        return np.array([_calculate_residual(eq, trial, context) for eq in equations]) / row_scale

    r0 = residuals(x0)
    if not np.all(np.isfinite(r0)):
        return None
    # Jacobi-Matrix der auf die Termgröße bezogenen Residuen; Schrittweite je Größe so, dass
    # sich die Residuen messbar ändern (Größen nahe null, z.B. eine Kontrollsumme, haben
    # keinen eigenen Maßstab)
    jac = np.empty((n, n))
    for j in range(n):
        step = 1e-7 * abs(x0[j]) if x0[j] != 0 else 1e-7
        for _ in range(12):
            x = x0.copy()
            x[j] += step
            column = (residuals(x) - r0) / step
            if np.all(np.isfinite(column)) and np.max(np.abs(column)) * step > 1e-9:
                break
            step *= 100.0
        else:
            # die Größe wirkt auf keine Gleichung: liegt sie auf einem Plateau (alle Gleichungen
            # bleiben exakt erfüllt), ist sie nicht bestimmt
            return _plateau_solution(equations, var_list[j], values, context)
        jac[:, j] = column
    # Nur die Richtungen zählen: Spalten und Zeilen auf Länge 1 (Beträge der Größen egal)
    col_norm = np.linalg.norm(jac, axis=0)
    if np.any(col_norm == 0):
        return None
    jac = jac / col_norm
    row_norm = np.linalg.norm(jac, axis=1)
    if np.any(row_norm == 0):
        return None
    jac = jac / row_norm[:, None]
    u, sigma, vt = np.linalg.svd(jac)
    if sigma[-1] > 1e-6 * sigma[0]:
        return None
    # Fortsetzung entlang der Nullraum-Richtung: eine Größe verschieben, die übrigen neu lösen
    direction = vt[-1] / col_norm             # in den ursprünglichen Größen
    k = int(np.argmax(np.abs(direction) * col_norm))
    delta = 1e-3 / col_norm[k]
    start = x0 + delta * direction / direction[k]
    others = [j for j in range(n) if j != k]

    def shifted(z):
        x = start.copy()
        x[others] = x0[others] + z / col_norm[others]
        return residuals(x)

    try:
        fit = least_squares(shifted, (start[others] - x0[others]) * col_norm[others],
                            method='lm', xtol=1e-14, ftol=1e-14, max_nfev=200 * n)
    except Exception:
        return None
    if not np.all(np.isfinite(fit.fun)) or np.max(np.abs(fit.fun)) > 1e-9:
        return None                       # isolierte Lösung - eindeutig
    left = np.abs(u[:, -1])
    weights = np.abs(vt[-1])
    dependent = [eq for eq, weight in zip(equations, left) if weight > 0.1 * left.max()]
    free = [v for v, weight in zip(var_list, weights) if weight > 0.1 * weights.max()]
    return dependent, free


def _overdetermined_part(equations: List[str], variables: Set[str], known: Set[str]
                         ) -> Tuple[List[str], Set[str]]:
    """
    Überbestimmter Teil eines Blocks (Dulmage-Mendelsohn): Gleichungen, die von nicht
    zugeordneten Gleichungen über alternierende Pfade (Gleichung -> Unbekannte -> deren
    zugeordnete Gleichung) erreichbar sind, und deren Unbekannte. Leer, wenn alle zugeordnet sind.
    """
    unknowns = {eq: _get_equation_unknowns(eq, known, variables) for eq in equations}
    owner: Dict[str, str] = {}

    def augment(eq, seen):
        for var in sorted(unknowns[eq]):
            if var in seen:
                continue
            seen.add(var)
            if var not in owner or augment(owner[var], seen):
                owner[var] = eq
                return True
        return False

    for eq in equations:
        augment(eq, set())
    matched = set(owner.values())
    queue = [eq for eq in equations if eq not in matched]
    part_eqs, part_vars = set(queue), set()
    while queue:
        eq = queue.pop()
        for var in unknowns[eq]:
            if var in part_vars:
                continue
            part_vars.add(var)
            other = owner.get(var)
            if other is not None and other not in part_eqs:
                part_eqs.add(other)
                queue.append(other)
    return [eq for eq in equations if eq in part_eqs], part_vars


def _square_subsets(equations: List[str], variables: Set[str], known: Set[str], limit: int = 8):
    """
    Quadratische Teilsysteme eines überbestimmten Blocks: Auswahlen von so vielen Gleichungen
    wie Unbekannten, die strukturell lösbar sind (perfekte Zuordnung Gleichung <-> Unbekannte).
    Reihenfolge: zuerst die Gleichungen in Blatt-Reihenfolge (spätere bleiben als Prüfung
    übrig), dann Varianten, in denen je eine der gewählten Gleichungen ersetzt wird - falls die
    erste Auswahl numerisch singulär ist (z.B. zwei gleichwertige Bilanzen). Generisch, ohne Namen.
    """
    unknowns = {eq: _get_equation_unknowns(eq, known, variables) for eq in equations}

    def matching(candidates):
        owner: Dict[str, str] = {}

        def augment(eq, seen):
            for var in sorted(unknowns[eq]):
                if var in seen:
                    continue
                seen.add(var)
                if var not in owner or augment(owner[var], seen):
                    owner[var] = eq
                    return True
            return False

        for eq in candidates:
            augment(eq, set())
        return set(owner.values()) if len(owner) == len(variables) else None

    first = matching(equations)
    if first is None:
        return
    seen_subsets = set()
    chosen = [eq for eq in equations if eq in first]
    seen_subsets.add(frozenset(chosen))
    yield chosen
    for left_out in reversed(chosen):
        if len(seen_subsets) >= limit:
            return
        alternative = matching([eq for eq in equations if eq != left_out])
        if alternative is None or frozenset(alternative) in seen_subsets:
            continue
        seen_subsets.add(frozenset(alternative))
        yield [eq for eq in equations if eq in alternative]


def _square_subset_without(equations: List[str], variables: Set[str], known: Set[str],
                           left_out: str) -> Optional[List[str]]:
    """Strukturell lösbares quadratisches Teilsystem ohne die Gleichung left_out (Blattreihenfolge)."""
    others = [eq for eq in equations if eq != left_out]
    return next(_square_subsets(others, variables, known, limit=1), None)


def _solve_overdetermined_block(equations, variables, known_values, context, manual_initial,
                                original_equations, inferred_units):
    """
    Block mit mehr Gleichungen als Unbekannten (überbestimmt): ein quadratisches Teilsystem
    lösen; die übrigen Gleichungen müssen mit dieser Lösung erfüllt sein. Widerspruchsfrei ->
    Lösung; sonst bleiben die übrigen Gleichungen offen und werden als Constraints geprüft
    (Meldung "Widersprüchliches System" mit vorgegebenem und berechnetem Wert).

    Returns:
        (success, solution, message, analysis, solved_equations)
    """
    # Nur der überbestimmte Teil (von überzähligen Gleichungen über alternierende Pfade
    # erreichbar, Dulmage-Mendelsohn): nachgelagerte Gleichungen kommen danach an die Reihe -
    # eine dort nicht erfüllbare Gleichung darf den Kern nicht verfälschen
    core_eqs, core_vars = _overdetermined_part(equations, variables, set(known_values))
    if core_eqs and len(core_eqs) < len(equations):
        equations, variables = core_eqs, core_vars
    fallback = None
    subsets = _square_subsets(equations, variables, set(known_values))
    preferred: List[List[str]] = []   # Teilsysteme ohne eine der abhängigen Gleichungen
    tried = set()
    while True:
        subset = preferred.pop(0) if preferred else next(subsets, None)
        if subset is None:
            break
        if frozenset(subset) in tried:
            continue
        tried.add(frozenset(subset))
        known_dependencies = len(_DEPENDENT_CORES)
        success, solution, msg, analysis = _solve_equation_block(
            subset, variables, known_values, context, manual_initial, original_equations,
            inferred_units)
        if not success:
            # singulärer Kern im Teilsystem: zuerst Varianten ohne eine seiner Gleichungen
            for dependency in _DEPENDENT_CORES[known_dependencies:]:
                for left_out in reversed([eq for eq in subset if eq in dependency[0]]):
                    alternative = _square_subset_without(equations, variables, set(known_values),
                                                         left_out)
                    if alternative is not None and frozenset(alternative) not in tried:
                        preferred.append(alternative)
            del _DEPENDENT_CORES[known_dependencies:]
            continue
        values = {**known_values, **solution}
        checks = [eq for eq in equations if eq not in subset]
        if all(_relative_residual(eq, values, context) < 1e-6 for eq in checks):
            return True, solution, msg, analysis, subset
        if fallback is None:
            fallback = (True, solution, msg, analysis, subset)
        # Numerisch singuläres Teilsystem (eine Bilanz folgt aus den anderen): zuerst eine der
        # abhängigen Gleichungen weglassen - sie wird zur Prüfgleichung. Die Reihenfolge der
        # strukturellen Varianten erreicht sie sonst evtl. nie (lange Kette nach den Bilanzen)
        dependency = _non_unique_solution(subset, variables, values, context)
        if dependency:
            for left_out in reversed([eq for eq in subset if eq in dependency[0]]):
                alternative = _square_subset_without(equations, variables, set(known_values), left_out)
                if alternative is not None and frozenset(alternative) not in tried:
                    preferred.append(alternative)
    if fallback is not None:
        return fallback
    return False, {}, "Überbestimmter Block ohne lösbares Teilsystem", None, equations


def _solve_equation_block(
    equations: List[str],
    variables: Set[str],
    known_values: Dict[str, float],
    context: dict,
    manual_initial: Optional[Dict[str, float]] = None,
    original_equations: Optional[Dict[str, str]] = None,
    inferred_units: Optional[Dict[str, str]] = None
) -> Tuple[bool, Dict[str, float], str, Optional[BlockAnalysis]]:
    """
    Löst einen Block von Gleichungen mit gemeinsamen Unbekannten.

    Bei größeren Blöcken wird zuerst versucht, den Block iterativ zu zerlegen
    und kleinere Sub-Blöcke sequentiell zu lösen.

    Args:
        manual_initial: Manuelle Startwerte (haben Priorität)
        original_equations: Mapping parsed -> original für Anzeige
        inferred_units: Aus Funktionsargumenten abgeleitete Einheiten {var: unit}

    Returns:
        (success, solution_dict, message, block_analysis)
    """
    if not equations or not variables:
        return True, {}, "Leerer Block", None

    n_vars = len(variables)
    n_eqs = len(equations)

    if n_eqs != n_vars:
        return False, {}, f"Block nicht quadratisch: {n_eqs} Gleichungen, {n_vars} Unbekannte", None

    attempted = set()  # bereits versuchte Kerne (kein teurer Doppelversuch)

    # Zerlegung in den gekoppelten Kern und nachgelagerte Gleichungen (auch kleine Blöcke:
    # eine nicht auswertbare Folgegleichung darf den lösbaren Kern nicht mitreißen)
    if n_vars > 2:
        success, solution, msg, block_analysis = _solve_block_iteratively(
            equations, variables, known_values, context, manual_initial, original_equations,
            inferred_units, attempted
        )
        if success:
            return success, solution, msg, block_analysis
        # Teilergebnis (unabhängig lösbare Kerne) behalten - nicht verwerfen
        partial, partial_analysis = solution, block_analysis
        if msg == _DEPENDENT_MSG:
            return False, partial, msg, partial_analysis   # singulär: simultan ebenso nicht eindeutig
        if frozenset(equations) in attempted:
            return False, partial, msg, partial_analysis  # Gesamtblock bereits als Kern versucht
    else:
        partial, partial_analysis = {}, None

    # Fallback: Löse den gesamten Block (Tearing, sonst simultan)
    success, solution, msg = _solve_core(
        equations, variables, known_values, context, manual_initial, inferred_units
    )
    if not success:
        return False, partial, msg, partial_analysis
    return success, solution, msg, None  # Keine BlockAnalysis für simultane Lösung


def _solve_block_iteratively(
    equations: List[str],
    variables: Set[str],
    known_values: Dict[str, float],
    context: dict,
    manual_initial: Optional[Dict[str, float]] = None,
    original_equations: Optional[Dict[str, str]] = None,
    inferred_units: Optional[Dict[str, str]] = None,
    attempted: Optional[Set[frozenset]] = None
) -> Tuple[bool, Dict[str, float], str, Optional[BlockAnalysis]]:
    """
    Versucht einen Block iterativ zu lösen, indem nach jeder gelösten
    Gleichung geprüft wird, ob weitere Gleichungen direkt oder mit
    nur einer Unbekannten lösbar sind.

    Args:
        equations: Liste von Gleichungen
        variables: Set der Unbekannten
        known_values: Dict bekannter Variablenwerte
        context: Evaluierungskontext
        manual_initial: Manuell vorgegebene Startwerte
        original_equations: Mapping parsed -> original für Anzeige
        inferred_units: Aus Funktionsargumenten abgeleitete Einheiten {var: unit}

    Returns:
        (success, solution_dict, message, block_analysis)
    """
    if original_equations is None:
        original_equations = {}
    if inferred_units is None:
        inferred_units = {}

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
                    eq, unknown, local_known, context, manual_initial, inferred_units
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

        # Phase 3: Finde und löse den minimalen gekoppelten Kern. Scheitert ein Kern, werden
        # seine Größen gesperrt und der nächste Kern unter den übrigen Gleichungen gesucht -
        # ein nicht lösbarer Teil (z = 1/(a - b) mit a = b) darf unabhängige Kerne nicht mitreißen
        blocked: Set[str] = set()
        while remaining_eqs and not made_progress:
            known_now = set(local_known.keys())
            open_eqs = [eq for eq in remaining_eqs
                        if not (_get_equation_unknowns(eq, known_now, remaining_vars) & blocked)]
            open_vars = remaining_vars - blocked
            if not open_eqs or not open_vars:
                break
            core_eqs, core_vars = _find_minimal_coupled_core(open_eqs, open_vars, known_now)
            if not core_vars or len(core_eqs) != len(core_vars):
                break
            success, sub_solution, _ = _solve_core(
                core_eqs, core_vars, local_known, context, manual_initial, inferred_units,
                attempted
            )
            if not success:
                blocked |= set(core_vars)
                continue
            # Ein Kern, dessen Lösung nicht eindeutig ist (Gleichungen voneinander abhängig: die
            # Schleife ist für jeden Wert erfüllt), legt seine Größen nicht fest - sein beliebiger
            # Wert machte alle nachgelagerten Kerne widersprüchlich. Abbrechen und melden; ein
            # überbestimmter Block lässt dann eine der abhängigen Gleichungen weg (Prüfgleichung)
            try:
                trial = {**local_known, **sub_solution}
                dependency = (_non_unique_solution(list(core_eqs), set(core_vars), trial, context)
                              if len(core_vars) > 1 else
                              _plateau_solution(list(core_eqs), next(iter(core_vars)), trial, context))
            except Exception:
                dependency = None
            if dependency is not None:
                _DEPENDENT_CORES.append(dependency)
                return False, solved_values, _DEPENDENT_MSG, block_analysis
            solved_values.update(sub_solution)
            local_known.update(sub_solution)
            if _partial_solution is not None:
                _partial_solution.update(sub_solution)
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
    manual_initial: Optional[Dict[str, float]] = None,
    inferred_units: Optional[Dict[str, str]] = None
) -> Tuple[bool, Dict[str, float], str]:
    """
    Löst einen Block von Gleichungen simultan mit normalisiertem least_squares.

    Strategie:
    1. Startwerte aus Einheiten bzw. der Struktur (nie aus Variablennamen)
    2. Normalisierung: Alle Variablen auf ~1.0 skalieren
    3. least_squares mit Levenberg-Marquardt (robuster als fsolve)
    4. Relative Toleranzen für Konvergenzprüfung

    Args:
        equations: Liste von Gleichungen
        variables: Set der Unbekannten
        known_values: Dict bekannter Variablenwerte
        context: Evaluierungskontext
        manual_initial: Manuell vorgegebene Startwerte
        inferred_units: Aus Funktionsargumenten abgeleitete Einheiten {var: unit}
    """
    import time
    from scipy.optimize import least_squares

    # Zeitbudget: ein nicht konvergierender Block darf die GUI nicht minutenlang
    # einfrieren (bis zu 16 Startvarianten mit Stoffwert-Aufrufen)
    deadline = time.monotonic() + 20.0

    if not equations or not variables:
        return True, {}, "Leerer Block"

    appearance = _appearance_order(equations, variables)
    var_list = sorted(variables, key=lambda v: appearance[v])
    n_vars = len(var_list)
    n_eqs = len(equations)

    if n_eqs != n_vars:
        return False, {}, f"Block nicht quadratisch: {n_eqs} Gleichungen, {n_vars} Unbekannte"

    # Erstelle Startvektor mit intelligenten Startwerten (basierend auf Einheiten)
    x0 = np.array([_get_initial_value(var, manual_initial, known_values, inferred_units)
                   for var in var_list], dtype=float)

    # Skalierungsfaktoren = initiale Werte (damit normalisierte Variablen ~1.0 sind)
    scales = np.maximum(np.abs(x0), 1e-10)

    # Residuen-Funktion (nicht normalisiert, für Auswertung)
    extra_deadline = [None]   # Suche nach weiteren Lösungen (nach der ersten), wie im Mehrfach-Tearing

    def block_func(x):
        _check_deadline()
        if time.monotonic() > deadline or (extra_deadline[0] is not None and time.monotonic() > extra_deadline[0]):
            # Zeitbudget des Blocks auch INNERHALB eines least_squares-/fsolve-Laufs: der
            # Versuch bricht ab (Exception), die Teillösung des Lösungslaufs bleibt erhalten
            raise RuntimeError("Zeitbudget des Blocks überschritten")
        local_ctx = context.copy()
        local_ctx.update(known_values)
        local_ctx.update({var: val for var, val in zip(var_list, x)})

        results = []
        for eq in equations:
            try:
                result = _as_real(eval(eq, {"__builtins__": {}}, local_ctx))
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

    # Kleine, je Größe gestaffelte relative Verschiebungen: ein Start genau auf einem
    # gegebenen Wert (T_2 = T_6 -> (T_2 - T_3)/(T_2 - T_6) = 0/0) wird so verlassen, ohne den
    # Gültigkeitsbereich zu verlassen (Faktoren 0.1 ... 3 führen absolute Temperaturen hinaus)
    stagger = np.arange(n_vars) - (n_vars - 1) / 2.0
    start_variations = [
        x0_norm,
        x0_norm * (1 + 0.02 + 0.01 * stagger),
        x0_norm * (1 - 0.02 - 0.01 * stagger),
        x0_norm * (1 + 0.06 + 0.02 * stagger),
        # Neutrale Startpunkte unabhängig von der Heuristik (der Fallback
        # "geometrisches Mittel bekannter Werte" kann weit danebenliegen, z.B.
        # m = 24000 als Exponent in N = C*Re^m -> Überlauf, keine Suchrichtung)
        np.ones(n_vars) / scales,
        0.5 * np.ones(n_vars) / scales,
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

    def block_relative_residual(x):
        """
        Max. Residuum aller Gleichungen, jeweils relativ zur Größenordnung
        ihrer Terme (NICHT zur Größe der Lösung - sonst würde Divergenz
        zur Asymptote als Lösung akzeptiert, z.B. 1/(x-2)=0 mit x=1e83).
        """
        values = dict(known_values)
        values.update(zip(var_list, x))
        pairs = [_residual_and_scale(eq, values, context) for eq in equations]
        if any(not np.isfinite(res) for res, _ in pairs):
            return float('inf')
        # Termgröße auch innerhalb von Produkten (_additive_term_scale): echte
        # Null-Gleichungen (Q = 0 = 2*(J_3 - J_1 + ...)) haben große innere Terme,
        # eine Asymptote (alle Terme -> 0) bleibt dagegen erkennbar
        relative = max(res / max(scale, 1e-300) for res, scale in pairs)
        # nur näherungsweise erfüllt: auf einem Plateau bzw. bei davonlaufender Lösung keine Lösung
        if relative < 1e-4 and _stuck_equations(equations, values, var_list, context,
                                                dict(zip(var_list, scales))):
            return float('inf')
        return relative

    def polished(x):
        """Newton-Nachiteration der akzeptierten Lösung (siehe _polish_block_solution)."""
        values = dict(known_values)
        values.update(zip(var_list, x))
        better = _polish_block_solution(equations, values, var_list, context, dict(zip(var_list, scales)))
        return np.array([better[var] for var in var_list])

    import warnings
    found_solutions: List[Dict[str, float]] = []
    began = time.monotonic()
    for index, x_start in enumerate(start_variations[:15]):  # Maximal 15 Versuche
        if time.monotonic() > deadline:
            break
        if extra_deadline[0] is not None and time.monotonic() > extra_deadline[0]:
            break
        if found_solutions and 1 <= index <= 3:
            continue   # kleine Verschiebungen nur, wenn der Start selbst nicht konvergiert (0/0 am Start)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = least_squares(
                    normalized_residuals,
                    x_start,
                    method='lm',  # Levenberg-Marquardt
                    ftol=1e-12,   # Relative Toleranz für Residuen
                    xtol=1e-12,   # Relative Toleranz für Variablen
                    # nach der ersten Lösung nur kurze Suchen nach weiteren (deterministisch)
                    max_nfev=(100 if found_solutions else 500) * n_vars
                )

            if result.success or result.status in [1, 2, 3, 4]:
                # Zurückskalieren
                solution = result.x * scales

                if not np.all(np.isfinite(solution)):
                    continue

                relative_residual = block_relative_residual(solution)

                if relative_residual < 1e-8:
                    result_dict = {var: val for var, val in zip(var_list, polished(solution))}
                    if _distinct(found_solutions, result_dict):
                        found_solutions.append(result_dict)
                    if extra_deadline[0] is None:
                        # weitere Lösungen höchstens zehnmal so lange suchen wie bis zur ersten (>= 2 s)
                        extra_deadline[0] = time.monotonic() + max(10.0 * (time.monotonic() - began), 2.0)
                    continue

                if relative_residual < best_residual:
                    best_solution = solution
                    best_residual = relative_residual

        except Exception:
            pass

    if found_solutions:
        # mehrere Lösungen: die den Startwerten nächste (nicht die erste konvergierte Variante)
        result_dict = _nearest_block_solution(found_solutions, equations, var_list, known_values,
                                              manual_initial, inferred_units)
        return True, result_dict, f"Block gelöst ({n_vars} Variablen)"

    # Fallback: Versuche fsolve mit verschiedenen Startwerten
    for x_start_norm in start_variations[:5]:
        if time.monotonic() > deadline:
            break
        try:
            x_start = x_start_norm * scales
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                solution, info, ier, _ = fsolve(block_func, x_start, full_output=True)

            if not np.all(np.isfinite(solution)):
                continue

            relative_residual = block_relative_residual(solution)

            if ier == 1 and relative_residual < 1e-8:
                result_dict = {var: val for var, val in zip(var_list, polished(solution))}
                return True, result_dict, f"Block gelöst ({n_vars} Variablen)"

            if relative_residual < best_residual:
                best_solution = solution
                best_residual = relative_residual
        except Exception:
            pass

    # Akzeptiere gute Näherung mit relativer Toleranz (relativ zur Termgröße)
    if best_solution is not None and best_residual < 1e-4:
        result_dict = {var: val for var, val in zip(var_list, polished(best_solution))}
        return True, result_dict, f"Block gelöst (rel. Residuum: {best_residual:.2e})"

    return False, {}, f"Block-Konvergenz fehlgeschlagen (rel. Residuum: {best_residual:.2e})"


def _get_initial_value(var: str, manual_initial: Optional[Dict[str, float]] = None,
                       known_values: Optional[Dict[str, float]] = None,
                       inferred_units: Optional[Dict[str, str]] = None,
                       use_structure: bool = True) -> float:
    """
    Ermittelt sinnvolle Startwerte basierend auf Einheiten - OHNE Variablennamen-Heuristik.

    Einheitenquellen (in Prioritätsreihenfolge):
    1. Manuelle Startwerte (höchste Priorität)
    2. Einheiten aus Funktionskontext (z.B. HumidAir(h, T=T_1, rF=rF_1) → T_1: K, rF_1: dimensionslos)
    3. Einheiten aus User-Definition (z.B. T = 20°C → T: K)
    4. Über Summen verbundene bekannte Größen (T_R - T_G1 -> T_G1 nahe T_R), siehe
       _start_hints (nur Iterationsstart, nicht für die Wurzelauswahl)
    5. Sonst 1.0

    WICHTIG: Keine Variablennamen-Heuristik! Die Einheit wird NUR aus dem Kontext abgeleitet,
    nicht weil eine Variable mit 'T' oder 'p' beginnt.
    """
    # PRIORITÄT 0: Manuelle Startwerte haben höchste Priorität
    if manual_initial and var in manual_initial:
        return manual_initial[var]

    # PRIORITÄT 1: Unit-basierte Startwerte (generisch, NICHT von Variablennamen abhängig)
    if inferred_units and var in inferred_units and UNITS_AVAILABLE:
        unit = inferred_units[var]
        typical = typical_value_from_unit(unit) if unit is not None else None
        if typical is not None:
            return typical

    # PRIORITÄT 2: Über Summen verbundene bekannte Größen (Struktur, siehe _start_hints) -
    # nur als Iterationsstart, nicht als Anker für die Wahl unter mehreren Wurzeln.
    # Bewusst KEINE Namensähnlichkeit (h_5 wie h_4): Startwerte und damit die Wahl unter
    # mehreren Wurzeln dürfen nicht von Formelzeichen abhängen
    if use_structure:
        structural = _structural_start(var, known_values)
        if structural is not None:
            return structural

    # KEINE Variablennamen-Heuristik mehr!
    # Die Einheit wird NUR aus dem Funktionskontext oder User-Definition abgeleitet,
    # nicht weil eine Variable mit 'T', 'p', 'h' etc. beginnt.
    #
    # Beispiel: "hoehe = 10" bekommt NICHT automatisch J/kg nur weil es mit 'h' beginnt.
    # Stattdessen muss die Einheit aus dem Kontext kommen:
    # - Funktionsargument: enthalpy(water, T=T_1, h=hoehe) → hoehe: J/kg
    # - User-Definition: hoehe = 10 m → hoehe: m

    # Sonst 1.0 - bewusst KEIN Mittel aller bekannten Werte: eine am Block unbeteiligte
    # Konstante (z = 1e6) würde sonst bestimmen, welche von mehreren Lösungen gewählt wird
    return 1.0


# Auswertungsfehler der letzten erfolglosen 1-D-Suche je Gleichung (für die Meldung)
_SEARCH_LOG: Dict[str, list] = {'ok': [], 'err': [], 'plateau': []}
_LAST_SEARCH_ERROR: Dict[str, str] = {}
# Kerne eines Blocks, deren Lösung nicht eindeutig ist (abhängige Gleichungen), je Lösungslauf
_DEPENDENT_CORES: List[Tuple] = []
_DEPENDENT_MSG = "Kern nicht eindeutig (abhängige Gleichungen)"


def _solve_single_unknown(equation: str, unknown: str, known_values: Dict[str, float],
                          context: dict, manual_initial: Optional[Dict[str, float]] = None,
                          inferred_units: Optional[Dict[str, str]] = None,
                          chain=None, block_anchors=None) -> Tuple[bool, float]:
    """
    Wie _solve_single_unknown_search; findet die Suche keine Wurzel, wird die Fehlermeldung
    des ungültigen Testpunkts gemerkt, der dem besten gültigen Punkt am nächsten liegt - dort
    endet der Definitionsbereich (z.B. "Zustand übersättigt" bei impliziter HumidAir-Gleichung).
    """
    _SEARCH_LOG['ok'], _SEARCH_LOG['err'], _SEARCH_LOG['plateau'] = [], [], []
    result = _solve_single_unknown_search(equation, unknown, known_values, context, manual_initial,
                                          inferred_units, chain, block_anchors)
    if result[0]:
        _LAST_SEARCH_ERROR.pop(equation, None)
    elif _SEARCH_LOG['plateau']:
        # Die Gleichung ist auf einem ganzen Bereich erfüllt (min/max-Plateau, n_L = 0 oberhalb des
        # Taupunkts): jeder Wert dort wäre eine Lösung - keine willkürlich gewählte melden
        x = _SEARCH_LOG['plateau'][0]
        _LAST_SEARCH_ERROR[equation] = (
            f"Lösung nicht eindeutig: die Gleichung ist für einen ganzen Bereich von {display_name(unknown)} "
            f"erfüllt (z.B. {display_name(unknown)} = {x:.6g}) - eine Vorgabe wählen, die "
            f"{display_name(unknown)} festlegt")
    elif _SEARCH_LOG['err']:
        if _SEARCH_LOG['ok']:
            best = min(_SEARCH_LOG['ok'], key=lambda item: item[1])[0]
        else:
            best = _get_initial_value(unknown, manual_initial, known_values, inferred_units)
        _LAST_SEARCH_ERROR[equation] = min(_SEARCH_LOG['err'], key=lambda item: abs(item[0] - best))[1]
        if len(_LAST_SEARCH_ERROR) > 1000:
            _LAST_SEARCH_ERROR.clear()
    _SEARCH_LOG['ok'], _SEARCH_LOG['err'], _SEARCH_LOG['plateau'] = [], [], []
    return result


def _solve_single_unknown_search(equation: str, unknown: str, known_values: Dict[str, float],
                                 context: dict, manual_initial: Optional[Dict[str, float]] = None,
                                 inferred_units: Optional[Dict[str, str]] = None,
                                 chain=None, block_anchors=None) -> Tuple[bool, float]:
    """
    Löst eine Gleichung mit einer einzelnen Unbekannten.

    Strategie:
    1. Schneller Newton-Raphson Versuch (für einfache Gleichungen)
    2. Falls nötig: Robuste Bracket-Suche mit Brent's Methode

    Args:
        equation: Gleichung in Python-Syntax (f(x) = 0 Form)
        unknown: Name der zu lösenden Variable
        known_values: Dict bekannter Variablenwerte
        context: Evaluierungskontext mit Funktionen
        manual_initial: Manuell vorgegebene Startwerte
        inferred_units: Aus Funktionsargumenten abgeleitete Einheiten {var: unit}
        chain: Optional (Tearing): Funktion x -> Dict aller Werte (bekannte Werte,
               unknown=x und daraus direkt berechnete Blockvariablen). Die
               Gleichung wird dann mit diesen Werten ausgewertet.
        block_anchors: Optional (Tearing): ([Blockgrößen in Blattreihenfolge], {Größe: Startwert}) -
               bei mehreren Wurzeln zählt der Abstand aller Blockgrößen (relativ zum Startwert)

    Returns:
        (success, value)
    """
    import time
    from scipy.optimize import brentq

    # Zeitbudget: unlösbare Gleichungen dürfen die GUI nicht minutenlang einfrieren
    deadline = time.monotonic() + 10.0

    def values_at(x):
        if chain is not None:
            return chain(x)
        values = dict(known_values)
        values[unknown] = x
        return values

    def func(x):
        _check_deadline()
        try:
            local_ctx = context.copy()
            local_ctx.update(values_at(x))
            value = _as_real(eval(equation, {"__builtins__": {}}, local_ctx))
        except Exception as exc:
            if len(_SEARCH_LOG['err']) < 20000:
                # Wert kopieren: fsolve übergibt ein Array und verändert es danach weiter - mit dem
                # Verweis stimmte der gemerkte Punkt nicht mehr (Meldung nannte die falsche Grenze)
                _SEARCH_LOG['err'].append((float(np.ravel(x)[0]), str(exc)))
            return float('inf')
        if np.all(np.isfinite(value)) and len(_SEARCH_LOG['ok']) < 20000:
            _SEARCH_LOG['ok'].append((float(np.ravel(x)[0]), float(np.max(np.abs(value)))))
        return value

    def rel_residual_at(x):
        """Residuum relativ zur Termgröße der Gleichung (skalenunabhängig)."""
        try:
            values = values_at(x)
        except Exception:
            return float('inf')
        return _relative_residual(equation, values, context)

    def continuous_sign_change(x):
        """
        Vorzeichenwechsel einer STETIGEN Funktion, auf Maschinengenauigkeit eingegrenzt: das
        Residuum ist dann Rauschen der Auswertung (Auslöschung in der Funktion, z.B. eine kleine
        Kondensatmenge = Wasser - Dampf aus NASA-Polynomen, ~1e-10 relativ), keine Polstelle und
        kein Sprung. Kriterium: daneben (relativ 1e-6) entgegengesetzte Vorzeichen; |f| an der
        Wurzel UND ganz nahe daneben (relativ 1e-11) winzig gegen den Verlauf (|f| relativ 1e-6
        daneben) - bei 1/(x-2) ist |f| an der "Wurzel" riesig, bei einem Sprung (IF, floor) ist
        eine Seite ganz nahe daneben so groß wie der Sprung.
        """
        span = abs(x) or abs(initial_guess)
        if not span:
            return False                  # kein Maßstab für die Umgebung
        h, near = 1e-6 * span, 1e-11 * span
        try:
            f_m, f_p = func(x - h), func(x + h)
            local = [abs(func(x)), abs(func(x - near)), abs(func(x + near))]
        except Exception:
            return False
        if not all(np.isfinite(v) for v in [f_m, f_p] + local) or f_m * f_p >= 0:
            return False
        return max(local) <= 1e-4 * max(abs(f_m), abs(f_p))

    def is_acceptable_root(x, tol=1e-9):
        return np.isfinite(x) and rel_residual_at(x) < tol

    def is_credible_root(x):
        """
        Prüft ob x eine ECHTE Nullstelle ist - nicht Underflow einer
        abklingenden Funktion (z.B. (x-3)*exp(-(x-3)^2) ist für x>30
        numerisch exakt 0) und nicht eine Polstelle.

        Kriterium: |f(x)| muss klein sein IM VERGLEICH zur Funktion in der
        Umgebung. An einer echten Nullstelle wächst |f| beidseitig (oder
        mindestens einseitig, z.B. an Clamping-Grenzen); auf einem
        Underflow-Plateau bleibt die Umgebung ebenfalls praktisch null.
        """
        if not np.isfinite(x):
            return False
        # Umgebung relativ zu x (bei x = 0 relativ zum Startwert): mit absolutem Mindestabstand
        # wäre bei x ~ 1e-6 jeder Punkt "isoliert" (y^2 = 4e-12 -> 7.8e-6 statt 2e-6)
        h0 = 0.05 * (abs(x) or abs(initial_guess) or 1.0)
        try:
            f_x = abs(func(x))
            f_p = abs(func(x + h0))
            f_m = abs(func(x - h0))
        except Exception:
            return False
        # Plateau: f ist im Punkt UND daneben exakt null, die Terme aber nicht (sonst Underflow wie
        # exp(-x^2)) - jeder Wert dort erfüllt die Gleichung, die Lösung ist nicht eindeutig
        if f_x == 0 and (f_p == 0 or f_m == 0):
            try:
                genuine = _residual_and_scale(equation, values_at(x), context)[1] > 0
            except Exception:
                genuine = False
            if genuine and len(_SEARCH_LOG['plateau']) < 10:
                _SEARCH_LOG['plateau'].append(x)
            return False
        # Polstellen/Eval-Fehler in der Umgebung als "sehr groß" behandeln
        f_p = min(f_p, 1e300) if np.isfinite(f_p) else 1e300
        f_m = min(f_m, 1e300) if np.isfinite(f_m) else 1e300
        f_max, f_min = max(f_p, f_m), min(f_p, f_m)

        # Fall 1: Gleichung relativ zur Termgröße erfüllt (mehrterminge
        # Gleichungen) UND Umgebung deutlich größer als das Residuum.
        # f_max > 1e-250 schließt Denormal-/Underflow-Plateaus aus.
        if rel_residual_at(x) < 1e-9 and f_max > 1e-250 and f_x <= 1e-3 * f_max:
            return True

        # Fall 2: |f| wächst BEIDSEITIG deutlich (klassische isolierte
        # Nullstelle, auch bei Ein-Term-Gleichungen wie f(x)=0)
        return f_min > 1e-250 and f_x <= 1e-6 * f_min

    # === Phase 1: Schneller Newton-Raphson Versuch ===
    # Für einfache (oft lineare) Gleichungen konvergiert dies in wenigen Iterationen
    # Startwert OHNE Geometrisches-Mittel-Fallback: bei Gleichungen mit mehreren
    # Wurzeln (z.B. sin(alpha)=0.5) würde sonst eine unbeteiligte Konstante
    # bestimmen, zu welcher Wurzel Newton konvergiert
    initial_guess = _get_initial_value(unknown, manual_initial, known_values,
                                       inferred_units, use_structure=False)

    def root_distance(root):
        """
        (Abstand zum Startpunkt, Vergleichsvektor) einer Wurzel. Beim Tearing über alle
        Blockgrößen, je relativ zu ihrem Startwert (Größen verschiedener Dimension); der Vektor
        (Blattreihenfolge) entscheidet bei gleichem Abstand: größere Werte zuerst.
        """
        if block_anchors is None or chain is None:
            return abs(root - initial_guess), (root,)
        names, anchors = block_anchors
        values = chain(root)
        vector = tuple(float(values[v]) for v in names)
        squares = (((x - anchors[v]) / (abs(anchors[v]) or 1.0)) ** 2 for v, x in zip(names, vector))
        return float(np.sqrt(sum(squares))), vector

    def nearer_root_possible(root):
        """
        Newton kann über eine nähere Wurzel springen (x^3 - 3.3*x + 1 = 0 ab 1 -> -1.95 statt
        1.64). Gibt es im Intervall Startwert ± |Wurzel - Startwert| einen weiteren
        Vorzeichenwechsel, entscheidet die Bracket-Suche (dem Startwert nächstgelegene Wurzel).
        Beim Tearing liegt jede Wurzel mit kleinerem Abstand aller Blockgrößen im Intervall
        Startwert ± Abstand * Bezugswert der Tearing-Größe.
        """
        try:
            distance = root_distance(root)[0]
        except Exception:
            return True
        if block_anchors is not None and chain is not None:
            distance *= abs(initial_guess) or 1.0
        if distance == 0 or not np.isfinite(distance):
            return False
        points = np.linspace(initial_guess - distance, initial_guess + distance, 65)
        values = []
        for point in points:
            try:
                values.append(func(point))
            except Exception:
                values.append(np.nan)
        tol = 1e-9 * max(distance, abs(root))
        for p_1, v_1, p_2, v_2 in zip(points, values, points[1:], values[1:]):
            if (np.isfinite(v_1) and np.isfinite(v_2) and v_1 * v_2 < 0
                    and not p_1 - tol <= root <= p_2 + tol):
                return True
        return False

    def polish_root(x, fx):
        """
        Akzeptierte Newton-Wurzel bis zur Maschinengenauigkeit nachiterieren (höchstens zwei
        Schritte, nur wenn |f| sinkt): akzeptiert wird schon bei 1e-9 relativ zur Termgröße -
        bei sich aufhebenden großen Termen (C*T_1 - C*T_2 mit T in K) wäre der Rest in einer
        kleinen Folgegröße (Wärmestrom) sonst ~1e-7 relativ.
        """
        for _ in range(2):
            try:
                h = max(1e-8 * (abs(x) or abs(initial_guess) or 1.0), 1e-300)
                dfx = (func(x + h) - func(x - h)) / (2 * h)
                if not np.isfinite(dfx) or dfx == 0:
                    break
                x_new = x - fx / dfx
                fx_new = func(x_new)
            except Exception:
                break
            if not (np.isfinite(fx_new) and abs(fx_new) < abs(fx)):
                break
            x, fx = x_new, fx_new
        return x

    try:
        x = initial_guess

        for iteration in range(30):  # Max 30 Iterationen
            fx = func(x)

            # Prüfe ob bereits Lösung gefunden (mit Schutz gegen
            # Underflow/Asymptote - siehe is_credible_root). Nur relativ zur Termgröße:
            # ein absolutes |f| < 1e-10 wäre bei kleinen SI-Werten immer erfüllt
            if is_acceptable_root(x):
                x = polish_root(x, fx)
                if is_credible_root(x) and not nearer_root_possible(x):
                    return True, x
                break  # Verdächtig flach bzw. nähere Wurzel möglich -> Bracket-Suche

            # Schrittweite der numerischen Ableitung relativ zu x (bei x = 0 zum Startwert)
            h = max(1e-8 * (abs(x) or abs(initial_guess) or 1.0), 1e-300)

            # Numerische Ableitung
            fx_plus = func(x + h)
            fx_minus = func(x - h)
            dfx = (fx_plus - fx_minus) / (2 * h)

            # Prüfe ob Ableitung gültig (keine absolute Schwelle: x/1e20 = 5 hat f' = 1e-20;
            # große Schritte dämpft der Schritt unten)
            if not np.isfinite(dfx) or dfx == 0:
                break  # Newton funktioniert nicht, verwende Bracket-Suche

            # Newton-Schritt
            x_new = x - fx / dfx

            # Prüfe Konvergenz (relative Änderung)
            if abs(x_new - x) < 1e-10 * max(abs(x), abs(x_new)):
                # Verifiziere Lösung (relativ zur Termgröße, Underflow-/Polstellen-sicher)
                if (is_acceptable_root(x_new) and is_credible_root(x_new)
                        and not nearer_root_possible(x_new)):
                    return True, x_new
                break

            # Dämpfung für große Schritte (verhindert Oszillation)
            if abs(x_new - x) > 100 * max(1, abs(x)):
                x = x + 0.5 * (x_new - x)  # Halber Schritt
            else:
                x = x_new

            # Prüfe ob Wert noch endlich (keine feste Obergrenze: x/1e20 = 5 hat die Lösung 5e20)
            if not np.isfinite(x):
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

    # Logarithmische Skalierung über alle Größenordnungen (1e-30 bis 1e30): intern ist alles
    # SI - Wellenlängen (~1e-6 m), Diffusionskoeffizienten (~1e-9 m²/s), Teilchenzahlen (~1e23);
    # weit vom Startwert kann Newton die Steigung nicht auflösen (2*x = 1e21 ab x = 1)
    for exp in range(-30, 31):
        base = 10.0 ** exp
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
            if time.monotonic() > deadline:
                break  # Zeitbudget erschöpft
            try:
                v = func(x)
                # keine absolute Obergrenze (K = 1/y mit K = 1e30: |f| ~ 1e30 überall); Pole
                # scheiden über das relative Residuum der Wurzel aus
                if np.isfinite(v):
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

    def find_exact_zeros(points):
        """
        Testpunkte, die EXAKT auf einer Nullstelle liegen (v1*v2 < 0 übersieht sie).
        Der Isolations-Check filtert Underflow-Plateaus aus (dort ist f(x) für
        ganze x-Bereiche numerisch exakt 0, ohne dass Nullstellen vorliegen).
        """
        return [x for x, v in points if v == 0.0 and is_credible_root(x)]

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

        # Rekursiv verfeinern wenn große RELATIVE Wertänderung
        # (absolute Schwelle wäre skalenabhängig: bei Residuen ~1e6 überall,
        # bei Residuen ~1e-3 nie erfüllt)
        if len(evaluated) >= 2:
            for i in range(len(evaluated) - 1):
                px1, pv1 = evaluated[i]
                px2, pv2 = evaluated[i + 1]
                local_scale = max(abs(pv1), abs(pv2), 1e-300)
                if abs(pv2 - pv1) > 0.5 * local_scale:  # Großer Sprung deutet auf Singularität
                    sub_brackets = refine_interval(px1, px2, depth + 1)
                    if sub_brackets:
                        return sub_brackets
        return []

    # Erste Auswertung
    valid_points = eval_points(test_points)
    brackets = find_brackets(valid_points)
    exact_zeros = find_exact_zeros(valid_points)

    # Wenn keine Brackets gefunden, suche nach Regionen mit großen Änderungen
    if not brackets and not exact_zeros:
        # Sortiere nach Größe der Änderung (größte zuerst, skalenunabhängig)
        changes = []
        for i in range(len(valid_points) - 1):
            x1, v1 = valid_points[i]
            x2, v2 = valid_points[i + 1]
            change = abs(v2 - v1)
            if change > 0:
                changes.append((change, x1, x2))
        changes.sort(reverse=True)

        for _, x1, x2 in changes[:20]:  # Top 20 Regionen untersuchen
            if time.monotonic() > deadline:
                break
            brackets = refine_interval(x1, x2)
            if brackets:
                break

    # Versuche alle gefundenen Brackets und sammle KANDIDATEN.
    # Bei mehreren Wurzeln (z.B. x^2-2x=5) wird NICHT einfach die erste
    # (= negativste) genommen, sondern die dem Startwert nächstgelegene -
    # das ist bei physikalischen Größen fast immer die gewünschte Lösung.
    candidates = [(0.0, z) for z in exact_zeros]
    fallback_root = None
    fallback_residual = float('inf')

    for a, b in brackets:
        if time.monotonic() > deadline and candidates:
            break
        try:
            # Toleranz relativ zum Intervall (absolut 1e-12 wäre bei Wurzeln ~1e-15 zu grob)
            # Maschinengenauigkeit: eine Wurzel nahe einem Knick (Taupunkt) braucht die volle Auflösung
            root = brentq(func, a, b, xtol=max(1e-14 * max(abs(a), abs(b)), 1e-300),
                          rtol=4 * np.finfo(float).eps)
            if not np.isfinite(root):
                continue
            rel = rel_residual_at(root)
            if rel < 1e-8 or continuous_sign_change(float(root)):
                candidates.append((rel, float(root)))
            elif rel < fallback_residual:
                fallback_root = float(root)
                fallback_residual = rel
        except Exception:
            pass

    if candidates:
        # Wähle die Wurzel, die dem Auswahl-Anker am nächsten liegt.
        # Der Anker ist der Startwert OHNE Geometrisches-Mittel-Fallback:
        # unbeteiligte Konstanten dürfen die Wurzelwahl nicht beeinflussen
        # (z.B. sin(alpha)=0.5 mit F=100 -> sonst 150 statt 30).
        # Tie-Break: bevorzuge die positive/größere Wurzel - bei physikalischen
        # Größen wie T, p, m_dot ist das fast immer die gewünschte.
        # Beim Tearing zählt der Abstand aller Blockgrößen (root_distance)
        ranked = []
        for _, root in candidates:
            try:
                distance, vector = root_distance(root)
            except Exception:
                continue
            if np.isfinite(distance):
                ranked.append((distance, vector, root))
        if ranked:
            # gleich weit entfernt bis auf Rundung (x^2 - 2*x = 5 ab 1: 1 ± sqrt(6)) -> positive Wurzel
            nearest = min(d for d, _, _ in ranked)
            reference = 1.0 if block_anchors is not None and chain is not None else abs(initial_guess)
            tie = 1e-9 * max(nearest, reference)
            return True, max((vector, root) for d, vector, root in ranked if d <= nearest + tie)[1]

    # Beste gefundene Näherung zurückgeben wenn akzeptabel (relativ zur Termgröße)
    if fallback_root is not None and fallback_residual < 1e-6:
        return True, fallback_root

    # Fallback: fsolve mit verschiedenen Startwerten
    # Beginne mit intelligentem Startwert basierend auf Einheiten (generisch)
    import warnings
    smart_start = _get_initial_value(unknown, manual_initial, known_values, inferred_units)
    fallback_starts = [smart_start, 1.0, 0.1, 10.0, 100.0, 1000.0, 10000.0]
    if scale > 1:
        fallback_starts.extend([scale, scale * 0.1, scale * 0.5, scale * 2, scale * 10])

    for x0 in fallback_starts:
        if time.monotonic() > deadline:
            break  # Zeitbudget erschöpft
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                solution, info, ier, _ = fsolve(func, x0, full_output=True)
            # Akzeptanz relativ zur Termgröße + Isolations-Check gegen
            # Underflow-Plateaus (absolut kleines Residuum reicht NICHT -
            # sonst würde z.B. 1/(x-2)=0 mit x=1e83 akzeptiert)
            if ier == 1 and np.isfinite(solution[0]) and is_credible_root(float(solution[0])):
                return True, float(solution[0])
        except Exception:
            pass

    return False, 0.0


def _try_vectorized_evaluation(equations: List[str], variables: Set[str],
                                sweep_vars: Dict[str, np.ndarray],
                                constants: Optional[Dict[str, float]] = None) -> Tuple[bool, Dict[str, np.ndarray], str]:
    """
    Versucht direkte vektorisierte Auswertung für einfache Zuweisungen.

    Funktioniert wenn alle Gleichungen die Form "var = ausdruck" haben,
    wobei der Ausdruck nur von Sweep-Variablen und bereits berechneten Variablen abhängt.

    WICHTIG: Nur echte KONSTANTEN werden vorbelegt - niemals Startwerte
    (initial_values sind Schätzwerte; sie als Ergebnisse einzutragen würde
    stillschweigend falsche Zahlen liefern, wenn eine Gleichung die Variable
    benutzt, bevor ihre definierende Gleichung ausgewertet wurde).
    Fehlende Variablen führen zu einem Eval-Fehler und werden im nächsten
    Durchlauf berechnet - die Reihenfolge löst sich also von selbst auf.
    """
    n_points = len(list(sweep_vars.values())[0])
    results = {name: arr.copy() for name, arr in sweep_vars.items()}

    # Füge Konstanten (echte feste Werte) als Arrays hinzu
    if constants:
        for var, val in constants.items():
            if var not in results:
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
    initial_values: Optional[Dict[str, float]] = None,
    progress_callback=None,
    constants: Optional[Dict[str, float]] = None,
    original_equations: Optional[Dict[str, str]] = None,
    point_solver=None,
    extra_result_vars: Optional[Set[str]] = None
) -> Tuple[bool, Dict[str, Union[float, np.ndarray]], str]:
    """
    Löst das Gleichungssystem für jeden Wert der Sweep-Variablen.

    point_solver: Löser je Punkt mit der Signatur von solve_system (Standard:
    solve_system; Optimierung: optimizer.optimize_system). extra_result_vars:
    weitere Größen, die der Punkt-Löser liefert (z.B. variierte Größen) - sie
    werden wie Sweep-Variablen als je Punkt verschieden behandelt.

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
        # Keine Sweep-Variablen -> ein Punkt (mit derselben Lösungsfunktion wie je Punkt)
        solve_point = point_solver or solve_system
        return solve_point(equations, variables, initial_values or {}, constants=constants,
                           original_equations=original_equations)[:3]

    if initial_values is None:
        initial_values = {}
    if constants is None:
        constants = {}

    extra_result_vars = set(extra_result_vars or ())
    solve_point = point_solver or solve_system

    # Versuche zuerst vektorisierte direkte Auswertung (mit Konstanten,
    # NICHT mit Startwerten - siehe _try_vectorized_evaluation)
    if point_solver is None:
        success, results, msg = _try_vectorized_evaluation(equations, variables, sweep_vars, constants)
    else:
        success, results, msg = False, {}, ""
    # Nicht definierte Werte (0/0, ln(-1), ...) entstehen vektorisiert still als NaN -
    # dann punktweise lösen, damit die betroffenen Punkte mit Grund gemeldet werden
    if success and any(isinstance(results.get(var), np.ndarray) and not np.all(np.isfinite(results[var]))
                       for var in variables):
        success = False
    if success:
        _collapse_sweep_independent(results, equations, variables, sweep_vars)
        results.update(constants)  # Konstanten als Einzelwerte, nicht als Arrays
        return True, results, msg

    # Bestimme die Länge des Sweeps (alle Sweep-Variablen müssen gleich lang sein)
    sweep_lengths = {name: len(arr) for name, arr in sweep_vars.items()}
    if len(set(sweep_lengths.values())) > 1:
        detail = ", ".join(f"{name} ({count} Werte)" for name, count in sweep_lengths.items())
        return False, {}, (f"Alle Parameterstudien-Variablen müssen gleich viele Werte haben: {detail}. "
                           f"Die Werte werden punktweise kombiniert - Schrittweiten anpassen.")

    n_points = next(iter(sweep_lengths.values()))

    # Sortiere die zu lösenden Variablen
    var_list = sorted(list(variables))

    # Initialisiere Ergebnis-Arrays
    results = {var: np.full(n_points, np.nan) for var in var_list + sorted(extra_result_vars)}
    # Füge auch die Sweep-Variablen zum Ergebnis hinzu
    for name, arr in sweep_vars.items():
        results[name] = arr.copy()

    failed_points = []
    first_failure = None  # (Punktnummer, Meldung) des ersten gescheiterten Punkts

    # Warm-Start: Die Lösung des Vorpunkts dient als Startwert für den
    # nächsten Punkt. Ohne das kann der Lösungszweig zwischen Sweep-Punkten
    # springen (z.B. x^2-2x=a: Punkt 1 negative Wurzel, Rest positive).
    point_initial = dict(initial_values)
    # Notbremse: nach einem gescheiterten Punkt gibt es keinen Warm-Start; scheitern
    # zwei Punkte hintereinander am Zeitlimit, scheitern meist alle weiteren auch
    # (n Punkte x 60 s). Dann abbrechen statt die GUI stundenlang zu blockieren.
    consecutive_timeouts = 0
    aborted_after = None

    # Löse für jeden Sweep-Punkt mit der robusten solve_system Methode
    for i in range(n_points):
        # Setze aktuelle Sweep-Werte als Konstanten
        sweep_values = {name: float(arr[i]) for name, arr in sweep_vars.items()}

        # Kombiniere Konstanten mit Sweep-Werten
        combined_constants = constants.copy()
        combined_constants.update(sweep_values)

        try:
            # Verwende solve_system für jeden Punkt (nutzt Block-Dekomposition und Bracket-Suche)
            success, solution, msg = solve_point(
                equations, variables, point_initial, constants=combined_constants,
                original_equations=original_equations
            )
        except Exception as exc:
            success, solution, msg = False, {}, str(exc)

        # Auch bei einem gescheiterten Punkt alle Größen übernehmen, die gelöst
        # wurden - sonst stünde dort z.B. C = m*c als NaN, obwohl wohldefiniert
        for var in var_list + sorted(extra_result_vars):
            value = solution.get(var)
            if value is not None and np.isfinite(value):
                results[var][i] = value

        if success:
            # Warm-Start für den nächsten Punkt
            warm = {var: float(solution[var]) for var in var_list
                    if var in solution and np.isfinite(solution[var])}
            point_initial = {**initial_values, **warm}
        else:
            failed_points.append(i)
            if first_failure is None:
                first_failure = (i + 1, msg)

        consecutive_timeouts = consecutive_timeouts + 1 if (not success and msg.startswith("Zeitlimit")) else 0
        if consecutive_timeouts >= 2 and i < n_points - 1:
            aborted_after = i + 1
            failed_points.extend(range(i + 1, n_points))
            if progress_callback:
                progress_callback(n_points, n_points)
            break

        # Fortschritts-Callback
        if progress_callback:
            progress_callback(i + 1, n_points)

    _collapse_sweep_independent(results, equations, variables,
                                {**sweep_vars, **{name: None for name in extra_result_vars}})

    # Füge Konstanten zum Ergebnis hinzu
    results.update(constants)

    # Zusammenfassung (Punkte für die Anzeige ab 1 gezählt)
    if not failed_points:
        msg = f"Parameterstudie erfolgreich: {n_points} Punkte berechnet"
        return True, results, msg
    points = ", ".join(str(i + 1) for i in failed_points[:8]) + (" ..." if len(failed_points) > 8 else "")
    reason = f"Punkt {first_failure[0]}: {first_failure[1]}"
    if aborted_after is not None:
        reason = (f"Abgebrochen nach Punkt {aborted_after}: Zeitlimit an zwei Punkten hintereinander "
                  f"überschritten (Startwerte vorgeben: Solve > Initial Values). {reason}")
    if len(failed_points) < n_points:
        msg = (f"Parameterstudie teilweise erfolgreich: {n_points - len(failed_points)}/{n_points} "
               f"Punkte berechnet, ohne Lösung: Punkt {points}. {reason}")
        return True, results, msg
    return False, results, f"Parameterstudie fehlgeschlagen: kein Punkt gelöst. {reason}"


def _collapse_sweep_independent(results: Dict[str, Any], equations: List[str],
                                variables: Set[str], sweep_vars: Dict[str, np.ndarray]) -> None:
    """
    Größen, die strukturell NICHT von den Sweep-Variablen abhängen (z.B. Radien
    aus gegebenen Durchmessern), als Einzelwert statt als konstantes Array.

    Strukturell (generisch): Jede Unbekannte wird über ein maximales Matching
    der Gleichung zugeordnet, die sie bestimmt; sie hängt vom Sweep ab, wenn
    diese Gleichung eine Sweep-Variable oder eine abhängige Größe enthält.
    """
    try:
        from diagnostics import _incidence, _maximum_matching
    except ImportError:
        return
    unknowns = set(variables)
    incidence = _incidence(equations, unknowns)
    eq_match, var_match = _maximum_matching(equations, incidence)
    sweep_names = set(sweep_vars)
    dependent = {v for v in unknowns if v not in var_match}  # strukturell offen: vorsichtig
    changed = True
    while changed:
        changed = False
        for var in sorted(unknowns - dependent):
            eq = var_match[var]
            if (_get_equation_unknowns(eq, set(), sweep_names)
                    or (set(incidence[eq]) - {var}) & dependent):
                dependent.add(var)
                changed = True
    for var in unknowns - dependent:
        values = results.get(var)
        if isinstance(values, np.ndarray):
            finite = values[np.isfinite(values)]
            if finite.size:
                results[var] = float(finite[0])


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
    print("\n".join(f"  {k} = {v:.6g}" for k, v in solution.items()))
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
    print("\n".join(f"  {k} = {v:.6g}" for k, v in solution.items()))
