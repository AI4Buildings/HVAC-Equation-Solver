"""
Dimensional Constraint Propagation für den HVAC Equation Solver.

Ermöglicht die automatische Erkennung von Einheiten bei impliziten Gleichungen
durch Analyse der Gleichungsstruktur und Propagation bekannter Dimensionen.

Beispiel:
    eta = (h_2 - h_1) / (h_2s - h_1)

    Wenn h_1, h_2s = kJ/kg bekannt und eta dimensionslos,
    dann muss h_2 auch kJ/kg sein.
"""

import ast
import re
from typing import Dict, Optional, Set, Tuple, Any
from dataclasses import dataclass

# Versuche pint zu importieren
try:
    import pint
    from units import ureg, normalize_unit
    PINT_AVAILABLE = True
except ImportError:
    PINT_AVAILABLE = False
    ureg = None


@dataclass
class DimensionInfo:
    """Speichert Dimensions-Information für eine Variable oder Ausdruck."""
    unit: Optional[str]  # None = unbekannt, "" = dimensionslos
    quantity: Any = None  # pint Quantity für Berechnungen

    @property
    def is_known(self) -> bool:
        return self.unit is not None

    @property
    def is_dimensionless(self) -> bool:
        return self.unit == "" or self.unit == "dimensionless"


def get_dimension_from_unit(unit_str: str) -> Any:
    """Erzeugt eine pint Quantity mit Dimension 1 für eine Einheit.

    Für Temperatur-Einheiten (°C, °F) wird delta_degC/delta_degF verwendet,
    da diese für dimensionale Analyse (z.B. T1-T2) besser geeignet sind.
    """
    if not PINT_AVAILABLE or not unit_str:
        return None
    try:
        normalized = normalize_unit(unit_str)
        # Konvertiere absolute Temperatur-Einheiten zu Delta-Einheiten für dimensionale Analyse
        # °C und °F sind Offset-Einheiten, die Probleme bei Berechnungen verursachen
        # K (Kelvin) wird ebenfalls zu delta_degC konvertiert, da 1K = 1°C Differenz
        if normalized in ('degC', 'degree_Celsius', 'celsius'):
            normalized = 'delta_degC'
        elif normalized in ('degF', 'degree_Fahrenheit', 'fahrenheit'):
            normalized = 'delta_degF'
        elif normalized in ('kelvin', 'K'):
            normalized = 'delta_degC'  # 1K Differenz = 1°C Differenz
        return ureg.Quantity(1.0, normalized)
    except:
        return None


def unit_from_quantity(quantity) -> str:
    """Extrahiert benutzerfreundliche Einheit aus pint Quantity."""
    if quantity is None or not PINT_AVAILABLE:
        return ""

    try:
        # Vereinfache zu Basiseinheiten
        base = quantity.to_base_units()
        dim = base.dimensionality

        # Bekannte Dimensionen zu HVAC-Einheiten mappen
        if dim == ureg.watt.dimensionality:
            return 'kW'  # Leistung in kW (HVAC-Standard)
        if dim == ureg.joule.dimensionality:
            return 'kJ'  # Energie in kJ (HVAC-Standard)
        if dim == ureg.pascal.dimensionality:
            return 'bar'
        if dim == ureg('kg/s').dimensionality:
            return 'kg/s'
        if dim == ureg('m^3/s').dimensionality:
            return 'm^3/s'
        if dim == ureg('m/s').dimensionality:
            return 'm/s'
        if dim == ureg('kg/m^3').dimensionality:
            return 'kg/m^3'
        if dim == ureg('J/kg').dimensionality:
            return 'kJ/kg'
        if dim == ureg('J/(kg*K)').dimensionality:
            return 'kJ/(kg*K)'
        if dim == ureg.kelvin.dimensionality:
            return 'K'
        if dim == ureg.meter.dimensionality:
            return 'm'
        if dim == ureg.kilogram.dimensionality:
            return 'kg'
        if dim == ureg.second.dimensionality:
            return 's'
        if dim == ureg('W/m^2').dimensionality:  # kg/s³ = W/m² (Wärmestromdichte)
            return 'W/m^2'
        if dim == ureg('m^2').dimensionality:
            return 'm^2'
        if dim == ureg('m^3').dimensionality:
            return 'm^3'
        if dim == ureg('m^3/kg').dimensionality:  # spezifisches Volumen
            return 'm^3/kg'
        if dim == ureg('W/m^2/K').dimensionality:  # Wärmeübergangskoeffizient
            return 'W/m^2K'
        if dim == ureg('W/m^2/K^4').dimensionality:  # Stefan-Boltzmann-Konstante
            return 'W/m^2K^4'
        if dim == ureg('m^2*K/W').dimensionality:  # Wärmedurchlasswiderstand
            return 'm^2K/W'

        # Dimensionslos
        if base.dimensionless:
            return ""

        # Fallback: String-Darstellung
        return str(base.units)
    except:
        return ""


class DimensionInferrer(ast.NodeVisitor):
    """
    AST-Visitor der Dimensionen durch einen Ausdruck propagiert.

    Regeln:
    - Addition/Subtraktion: Alle Operanden müssen gleiche Dimension haben
    - Multiplikation: Dimensionen multiplizieren sich
    - Division: Dimensionen dividieren sich
    - Potenz: Basis^n hat Dimension (dim_basis)^n
    - Funktionen: sin, cos, exp, log erfordern dimensionslose Argumente
    """

    def __init__(self, known_dimensions: Dict[str, DimensionInfo]):
        self.known = known_dimensions
        self.inferred: Dict[str, str] = {}  # Neu abgeleitete Einheiten

    def infer_from_expression(self, expr_str: str) -> Tuple[DimensionInfo, Dict[str, str]]:
        """
        Analysiert einen Ausdruck und gibt die resultierende Dimension zurück.

        Returns:
            (dimension_info, newly_inferred_units)
        """
        try:
            tree = ast.parse(expr_str, mode='eval')
            result = self.visit(tree.body)
            return result, self.inferred
        except:
            return DimensionInfo(None), {}

    def visit_Constant(self, node) -> DimensionInfo:
        """Zahlen sind dimensionslos."""
        return DimensionInfo("", ureg.Quantity(1.0, 'dimensionless') if PINT_AVAILABLE else None)

    def visit_Num(self, node) -> DimensionInfo:
        """Für ältere Python-Versionen."""
        return DimensionInfo("", ureg.Quantity(1.0, 'dimensionless') if PINT_AVAILABLE else None)

    def visit_Name(self, node) -> DimensionInfo:
        """Variable - schaue in bekannten Dimensionen nach."""
        var_name = node.id

        # Bekannte Konstanten
        if var_name in ('pi', 'e'):
            return DimensionInfo("", ureg.Quantity(1.0, 'dimensionless') if PINT_AVAILABLE else None)

        if var_name in self.known:
            return self.known[var_name]

        # Unbekannt
        return DimensionInfo(None)

    def visit_BinOp(self, node) -> DimensionInfo:
        """Binäre Operationen: +, -, *, /, **"""
        left = self.visit(node.left)
        right = self.visit(node.right)

        if isinstance(node.op, (ast.Add, ast.Sub)):
            # Addition/Subtraktion: Dimensionen müssen gleich sein
            return self._handle_add_sub(node, left, right)

        elif isinstance(node.op, ast.Mult):
            # Multiplikation: Dimensionen multiplizieren
            return self._handle_mult(left, right)

        elif isinstance(node.op, ast.Div):
            # Division: Dimensionen dividieren
            return self._handle_div(left, right)

        elif isinstance(node.op, ast.Pow):
            # Potenz - übergebe AST-Knoten für Exponent-Extraktion
            return self._handle_pow(left, right, node.right)

        return DimensionInfo(None)

    def _handle_add_sub(self, node, left: DimensionInfo, right: DimensionInfo) -> DimensionInfo:
        """Addition/Subtraktion: Dimensionen müssen gleich sein."""
        # Wenn beide bekannt, müssen sie gleich sein
        if left.is_known and right.is_known:
            # Rückgabe der bekannten Dimension
            if left.quantity is not None:
                return left
            return right

        # Wenn eine Seite bekannt, kann die andere abgeleitet werden
        if left.is_known and not right.is_known:
            # Versuche rechte Seite abzuleiten
            self._infer_from_node(node.right, left)
            return left

        if right.is_known and not left.is_known:
            # Versuche linke Seite abzuleiten
            self._infer_from_node(node.left, right)
            return right

        return DimensionInfo(None)

    def _handle_mult(self, left: DimensionInfo, right: DimensionInfo) -> DimensionInfo:
        """Multiplikation: Dimensionen multiplizieren."""
        if not PINT_AVAILABLE:
            return DimensionInfo(None)

        # Wenn beide Seiten eine Quantity haben, multipliziere sie
        if left.quantity is not None and right.quantity is not None:
            try:
                result = left.quantity * right.quantity
                unit = unit_from_quantity(result)
                return DimensionInfo(unit, result)
            except:
                pass

        # Wenn eine Seite dimensionslos UND BEKANNT (keine Quantity, aber is_known=True),
        # gib die andere zurück. z.B. epsilon * sigma -> sigma's Einheit bleibt erhalten
        # WICHTIG: Unbekannte Variablen (is_known=False) dürfen nicht als dimensionslos behandelt werden!
        if left.quantity is None and left.is_known and right.quantity is not None:
            return right
        if right.quantity is None and right.is_known and left.quantity is not None:
            return left

        return DimensionInfo(None)

    def _handle_div(self, left: DimensionInfo, right: DimensionInfo) -> DimensionInfo:
        """Division: Dimensionen dividieren."""
        if not PINT_AVAILABLE:
            return DimensionInfo(None)

        if left.quantity is not None and right.quantity is not None:
            try:
                result = left.quantity / right.quantity
                unit = unit_from_quantity(result)
                return DimensionInfo(unit, result)
            except:
                pass

        return DimensionInfo(None)

    def _extract_exponent_value(self, node: ast.AST) -> Optional[float]:
        """Extrahiert numerischen Wert aus Exponent-Knoten."""
        if node is None:
            return None
        if isinstance(node, ast.Constant):  # Python 3.8+
            return node.value if isinstance(node.value, (int, float)) else None
        if isinstance(node, ast.Num):  # Python < 3.8
            return node.n
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            # Negativer Exponent: -2 etc.
            inner = self._extract_exponent_value(node.operand)
            return -inner if inner is not None else None
        return None

    def _handle_pow(self, base: DimensionInfo, exp: DimensionInfo, exp_node: ast.AST = None) -> DimensionInfo:
        """Potenz: (dim_base)^n"""
        if not PINT_AVAILABLE:
            return DimensionInfo(None)

        # Exponent muss dimensionslos sein
        if exp.is_known and not exp.is_dimensionless:
            return DimensionInfo(None)

        if base.quantity is not None:
            # Versuche numerischen Exponenten aus AST zu extrahieren
            exp_value = self._extract_exponent_value(exp_node)
            if exp_value is not None:
                try:
                    result = base.quantity ** exp_value
                    unit = unit_from_quantity(result)
                    return DimensionInfo(unit, result)
                except:
                    pass

        return DimensionInfo(None)

    def visit_UnaryOp(self, node) -> DimensionInfo:
        """Unäre Operationen: -, +"""
        return self.visit(node.operand)

    def visit_Call(self, node) -> DimensionInfo:
        """Funktionsaufrufe."""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id

            # Trigonometrische und transzendente Funktionen
            if func_name in ('sin', 'cos', 'tan', 'asin', 'acos', 'atan',
                            'sinh', 'cosh', 'tanh', 'exp', 'log', 'log10'):
                # Argument sollte dimensionslos sein, Ergebnis ist dimensionslos
                return DimensionInfo("", ureg.Quantity(1.0, 'dimensionless') if PINT_AVAILABLE else None)

            # Strahlungs-Funktionen die dimensionslose Werte zurückgeben
            if func_name in ('Blackbody', 'blackbody', 'Blackbody_cumulative', 'blackbody_cumulative'):
                # Gibt Bruchteil (0-1) zurück, dimensionslos
                return DimensionInfo("", ureg.Quantity(1.0, 'dimensionless') if PINT_AVAILABLE else None)

            # sqrt
            if func_name == 'sqrt':
                if node.args:
                    arg_dim = self.visit(node.args[0])
                    if arg_dim.quantity is not None:
                        try:
                            result = arg_dim.quantity ** 0.5
                            unit = unit_from_quantity(result)
                            return DimensionInfo(unit, result)
                        except:
                            pass

            # abs - behält Dimension
            if func_name == 'abs':
                if node.args:
                    return self.visit(node.args[0])

        return DimensionInfo(None)

    def _infer_from_node(self, node, target_dim: DimensionInfo):
        """Versucht, unbekannte Variablen aus dem Knoten abzuleiten (rekursiv)."""
        if isinstance(node, ast.Name):
            var_name = node.id
            if var_name not in self.known and target_dim.is_known:
                self.inferred[var_name] = target_dim.unit
                # Füge zu known hinzu für weitere Propagation
                self.known[var_name] = target_dim

        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            # Bei Addition/Subtraktion: Alle Terme haben gleiche Dimension
            # Rekursiv beide Seiten inferieren
            self._infer_from_node(node.left, target_dim)
            self._infer_from_node(node.right, target_dim)

        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            # Bei Multiplikation: a * b = target
            # Wenn eine Seite bekannt ist, können wir die andere ableiten
            left_dim = self.visit(node.left)
            right_dim = self.visit(node.right)

            if left_dim.is_known and left_dim.quantity is not None and not right_dim.is_known:
                # left ist bekannt, inferiere right = target / left
                if target_dim.quantity is not None:
                    try:
                        inferred_quantity = target_dim.quantity / left_dim.quantity
                        inferred_unit = unit_from_quantity(inferred_quantity)
                        inferred_dim = DimensionInfo(inferred_unit, inferred_quantity)
                        self._infer_from_node(node.right, inferred_dim)
                    except:
                        pass
            elif right_dim.is_known and right_dim.quantity is not None and not left_dim.is_known:
                # right ist bekannt, inferiere left = target / right
                if target_dim.quantity is not None:
                    try:
                        inferred_quantity = target_dim.quantity / right_dim.quantity
                        inferred_unit = unit_from_quantity(inferred_quantity)
                        inferred_dim = DimensionInfo(inferred_unit, inferred_quantity)
                        self._infer_from_node(node.left, inferred_dim)
                    except:
                        pass

        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            # Bei Division: a / b = target
            # Wenn eine Seite bekannt ist, können wir die andere ableiten
            left_dim = self.visit(node.left)
            right_dim = self.visit(node.right)

            if left_dim.is_known and left_dim.quantity is not None and not right_dim.is_known:
                # left ist bekannt, inferiere right = left / target
                if target_dim.quantity is not None:
                    try:
                        inferred_quantity = left_dim.quantity / target_dim.quantity
                        inferred_unit = unit_from_quantity(inferred_quantity)
                        inferred_dim = DimensionInfo(inferred_unit, inferred_quantity)
                        self._infer_from_node(node.right, inferred_dim)
                    except:
                        pass
            elif right_dim.is_known and right_dim.quantity is not None and not left_dim.is_known:
                # right ist bekannt, inferiere left = target * right
                if target_dim.quantity is not None:
                    try:
                        inferred_quantity = target_dim.quantity * right_dim.quantity
                        inferred_unit = unit_from_quantity(inferred_quantity)
                        inferred_dim = DimensionInfo(inferred_unit, inferred_quantity)
                        self._infer_from_node(node.left, inferred_dim)
                    except:
                        pass

        elif isinstance(node, ast.UnaryOp):
            # Bei unären Operationen (-x, +x): Dimension bleibt gleich
            self._infer_from_node(node.operand, target_dim)

    def generic_visit(self, node):
        """Fallback für unbekannte Knoten."""
        return DimensionInfo(None)


def analyze_equation(equation: str, known_units: Dict[str, str]) -> Dict[str, str]:
    """
    Analysiert eine einzelne Gleichung und leitet neue Einheiten ab.

    Args:
        equation: Gleichung der Form "left = right" oder "(left) - (right)"
        known_units: Dict von Variablen zu ihren bekannten Einheiten

    Returns:
        Dict von neu abgeleiteten Variablen und ihren Einheiten
    """
    if not PINT_AVAILABLE:
        return {}

    # Konvertiere known_units zu DimensionInfo
    known_dims = {}
    for var, unit in known_units.items():
        quantity = get_dimension_from_unit(unit) if unit else ureg.Quantity(1.0, 'dimensionless')
        known_dims[var] = DimensionInfo(unit if unit else "", quantity)

    # Parse Gleichung
    # Format: "left = right" (mit oder ohne Leerzeichen) oder "(left) - (right)"
    left_str = None
    right_str = None

    if equation.startswith('(') and ') - (' in equation:
        # Solver-Format: (var) - (expr)
        match = re.match(r'\(([^)]+)\)\s*-\s*\((.+)\)$', equation)
        if match:
            left_str = match.group(1)
            right_str = match.group(2)
    elif '=' in equation:
        # Normales Format: left = right (mit oder ohne Leerzeichen)
        # Finde das erste = das nicht Teil von == oder != ist
        eq_pos = -1
        for i, c in enumerate(equation):
            if c == '=' and (i == 0 or equation[i-1] not in '!=<>') and (i == len(equation)-1 or equation[i+1] != '='):
                eq_pos = i
                break
        if eq_pos > 0:
            left_str = equation[:eq_pos].strip()
            right_str = equation[eq_pos+1:].strip()

    if left_str is None or right_str is None:
        return {}

    # Konvertiere ^ zu ** für Python-Parser
    left_str = left_str.replace('^', '**')
    right_str = right_str.replace('^', '**')

    inferred = {}

    # Analysiere beide Seiten
    inferrer_left = DimensionInferrer(known_dims.copy())
    inferrer_right = DimensionInferrer(known_dims.copy())

    try:
        left_dim, left_inferred = inferrer_left.infer_from_expression(left_str)
        right_dim, right_inferred = inferrer_right.infer_from_expression(right_str)

        inferred.update(left_inferred)
        inferred.update(right_inferred)

        # Gleichheits-Constraint: left = right bedeutet dim(left) = dim(right)
        # Wenn eine Seite bekannt und die andere eine einzelne Variable, kann sie abgeleitet werden

        # Prüfe ob linke Seite eine einzelne Variable ist
        try:
            left_ast = ast.parse(left_str, mode='eval')
            if isinstance(left_ast.body, ast.Name):
                left_var = left_ast.body.id
                if left_var not in known_units and right_dim.is_known:
                    inferred[left_var] = right_dim.unit
            # NEU: Prüfe ob linke Seite eine Potenz einer Variable ist (z.B. T_3^4)
            elif isinstance(left_ast.body, ast.BinOp) and isinstance(left_ast.body.op, ast.Pow):
                if isinstance(left_ast.body.left, ast.Name):
                    left_var = left_ast.body.left.id
                    if left_var not in known_units and right_dim.is_known and right_dim.quantity is not None:
                        # Extrahiere den Exponenten
                        exp_value = None
                        if isinstance(left_ast.body.right, ast.Constant):
                            exp_value = left_ast.body.right.value
                        elif isinstance(left_ast.body.right, ast.Num):
                            exp_value = left_ast.body.right.n

                        if exp_value is not None and exp_value != 0:
                            # var^n = expr → var = expr^(1/n)
                            try:
                                var_quantity = right_dim.quantity ** (1.0 / exp_value)
                                var_unit = unit_from_quantity(var_quantity)
                                if var_unit:
                                    inferred[left_var] = var_unit
                            except:
                                pass
        except:
            pass

        # Prüfe ob rechte Seite eine einzelne Variable ist
        try:
            right_ast = ast.parse(right_str, mode='eval')
            if isinstance(right_ast.body, ast.Name):
                right_var = right_ast.body.id
                if right_var not in known_units and left_dim.is_known:
                    inferred[right_var] = left_dim.unit
            # NEU: Prüfe ob rechte Seite eine Potenz einer Variable ist
            elif isinstance(right_ast.body, ast.BinOp) and isinstance(right_ast.body.op, ast.Pow):
                if isinstance(right_ast.body.left, ast.Name):
                    right_var = right_ast.body.left.id
                    if right_var not in known_units and left_dim.is_known and left_dim.quantity is not None:
                        # Extrahiere den Exponenten
                        exp_value = None
                        if isinstance(right_ast.body.right, ast.Constant):
                            exp_value = right_ast.body.right.value
                        elif isinstance(right_ast.body.right, ast.Num):
                            exp_value = right_ast.body.right.n

                        if exp_value is not None and exp_value != 0:
                            # expr = var^n → var = expr^(1/n)
                            try:
                                var_quantity = left_dim.quantity ** (1.0 / exp_value)
                                var_unit = unit_from_quantity(var_quantity)
                                if var_unit:
                                    inferred[right_var] = var_unit
                            except:
                                pass
        except:
            pass

        # Rückwärts-Inferenz für Multiplikation/Division
        # Bei left = a * b: Wenn left und b bekannt, kann a berechnet werden
        # Bei left = a / b: Wenn left und b bekannt, kann a berechnet werden
        if left_dim.is_known:
            try:
                right_ast = ast.parse(right_str, mode='eval')
                reverse_inferred = _infer_from_mult_div(right_ast.body, left_dim, known_dims)
                inferred.update(reverse_inferred)
                # Auch Addition/Subtraktion behandeln
                reverse_inferred_add = _infer_from_addition(right_ast.body, left_dim, known_dims)
                inferred.update(reverse_inferred_add)
            except:
                pass

        # NEU: Rückwärts-Inferenz für linke Seite wenn rechte Seite bekannt ist
        if right_dim.is_known:
            try:
                left_ast = ast.parse(left_str, mode='eval')
                reverse_inferred = _infer_from_mult_div(left_ast.body, right_dim, known_dims)
                inferred.update(reverse_inferred)
                # Auch Addition/Subtraktion behandeln
                reverse_inferred_add = _infer_from_addition(left_ast.body, right_dim, known_dims)
                inferred.update(reverse_inferred_add)
            except:
                pass

    except Exception as e:
        pass

    return inferred


def _infer_from_mult_div(node, target_dim: DimensionInfo, known_dims: Dict[str, DimensionInfo]) -> Dict[str, str]:
    """
    Rückwärts-Inferenz für Multiplikation/Division (rekursiv).

    Bei target = a * b: Wenn target und b bekannt, dann a = target / b
    Bei target = a / b: Wenn target und b bekannt, dann a = target * b
    """
    if not PINT_AVAILABLE:
        return {}

    inferred = {}

    if not isinstance(node, ast.BinOp):
        return inferred

    if isinstance(node.op, ast.Mult):
        # target = left * right
        left_dim = _get_dimension(node.left, known_dims)
        right_dim = _get_dimension(node.right, known_dims)

        # Wenn left unbekannt und right bekannt
        if not left_dim.is_known and right_dim.is_known and right_dim.quantity is not None:
            try:
                result_quantity = target_dim.quantity / right_dim.quantity
                unit = unit_from_quantity(result_quantity)
                new_target_dim = DimensionInfo(unit, result_quantity)

                var_name = _get_var_name(node.left)
                if var_name and unit:
                    inferred[var_name] = unit
                elif isinstance(node.left, ast.BinOp):
                    # Rekursiv: left ist auch eine Mult/Div Operation
                    sub_inferred = _infer_from_mult_div(node.left, new_target_dim, known_dims)
                    inferred.update(sub_inferred)
            except:
                pass

        # Wenn right unbekannt und left bekannt
        if not right_dim.is_known and left_dim.is_known and left_dim.quantity is not None:
            try:
                result_quantity = target_dim.quantity / left_dim.quantity
                unit = unit_from_quantity(result_quantity)
                new_target_dim = DimensionInfo(unit, result_quantity)

                var_name = _get_var_name(node.right)
                if var_name and unit:
                    inferred[var_name] = unit
                elif isinstance(node.right, ast.BinOp):
                    # Rekursiv: right ist auch eine Mult/Div Operation
                    sub_inferred = _infer_from_mult_div(node.right, new_target_dim, known_dims)
                    inferred.update(sub_inferred)
            except:
                pass

    elif isinstance(node.op, ast.Div):
        # target = left / right
        left_dim = _get_dimension(node.left, known_dims)
        right_dim = _get_dimension(node.right, known_dims)

        # Wenn left unbekannt und right bekannt: left = target * right
        if not left_dim.is_known and right_dim.is_known and right_dim.quantity is not None:
            try:
                result_quantity = target_dim.quantity * right_dim.quantity
                unit = unit_from_quantity(result_quantity)
                new_target_dim = DimensionInfo(unit, result_quantity)

                var_name = _get_var_name(node.left)
                if var_name and unit:
                    inferred[var_name] = unit
                elif isinstance(node.left, ast.BinOp):
                    # Rekursiv
                    sub_inferred = _infer_from_mult_div(node.left, new_target_dim, known_dims)
                    inferred.update(sub_inferred)
            except:
                pass

    return inferred


def _infer_from_addition(node, target_dim: DimensionInfo, known_dims: Dict[str, DimensionInfo]) -> Dict[str, str]:
    """
    Rückwärts-Inferenz für Addition/Subtraktion (rekursiv).

    Bei Addition/Subtraktion haben alle Terme die gleiche Dimension wie das Ergebnis.
    target = a + b → a und b haben beide target's Dimension
    """
    if not PINT_AVAILABLE:
        return {}

    inferred = {}

    if not isinstance(node, ast.BinOp):
        # Einzelne Variable oder Zahl
        if isinstance(node, ast.Name):
            var_name = node.id
            if var_name not in known_dims and target_dim.is_known:
                unit = target_dim.unit if target_dim.unit else ""
                if unit:
                    inferred[var_name] = unit
        return inferred

    if isinstance(node.op, (ast.Add, ast.Sub)):
        # Bei Addition/Subtraktion: Alle Terme haben gleiche Dimension
        # Rekursiv beide Seiten mit target_dim inferieren
        left_inferred = _infer_from_addition(node.left, target_dim, known_dims)
        inferred.update(left_inferred)

        right_inferred = _infer_from_addition(node.right, target_dim, known_dims)
        inferred.update(right_inferred)

    elif isinstance(node.op, ast.Mult):
        # Bei Multiplikation: a * b = target
        # Wenn b dimensionslos und bekannt → a = target
        # Wenn a dimensionslos und bekannt → b = target
        left_dim = _get_dimension(node.left, known_dims)
        right_dim = _get_dimension(node.right, known_dims)

        # Wenn left dimensionslos (bekannt), dann hat right die target Dimension
        if left_dim.is_known and left_dim.is_dimensionless and not right_dim.is_known:
            right_inferred = _infer_from_addition(node.right, target_dim, known_dims)
            inferred.update(right_inferred)

        # Wenn right dimensionslos (bekannt), dann hat left die target Dimension
        if right_dim.is_known and right_dim.is_dimensionless and not left_dim.is_known:
            left_inferred = _infer_from_addition(node.left, target_dim, known_dims)
            inferred.update(left_inferred)

        # Spezialfall: sigma * T^4 → wenn sigma bekannt, kann T abgeleitet werden
        # left hat Einheit, right ist Potenz einer Variable
        if left_dim.is_known and left_dim.quantity is not None:
            if isinstance(node.right, ast.BinOp) and isinstance(node.right.op, ast.Pow):
                if isinstance(node.right.left, ast.Name):
                    var_name = node.right.left.id
                    if var_name not in known_dims:
                        # Extrahiere Exponenten
                        exp_value = None
                        if isinstance(node.right.right, ast.Constant):
                            exp_value = node.right.right.value
                        elif isinstance(node.right.right, ast.Num):
                            exp_value = node.right.right.n

                        if exp_value is not None and exp_value != 0 and target_dim.quantity is not None:
                            try:
                                # target = left * var^exp → var^exp = target / left
                                pow_quantity = target_dim.quantity / left_dim.quantity
                                var_quantity = pow_quantity ** (1.0 / exp_value)
                                unit = unit_from_quantity(var_quantity)
                                if unit:
                                    inferred[var_name] = unit
                            except:
                                pass

        # Symmetrisch: right hat Einheit, left ist Potenz
        if right_dim.is_known and right_dim.quantity is not None:
            if isinstance(node.left, ast.BinOp) and isinstance(node.left.op, ast.Pow):
                if isinstance(node.left.left, ast.Name):
                    var_name = node.left.left.id
                    if var_name not in known_dims:
                        exp_value = None
                        if isinstance(node.left.right, ast.Constant):
                            exp_value = node.left.right.value
                        elif isinstance(node.left.right, ast.Num):
                            exp_value = node.left.right.n

                        if exp_value is not None and exp_value != 0 and target_dim.quantity is not None:
                            try:
                                pow_quantity = target_dim.quantity / right_dim.quantity
                                var_quantity = pow_quantity ** (1.0 / exp_value)
                                unit = unit_from_quantity(var_quantity)
                                if unit:
                                    inferred[var_name] = unit
                            except:
                                pass

    return inferred


def _get_dimension(node, known_dims: Dict[str, DimensionInfo]) -> DimensionInfo:
    """Berechnet die Dimension eines AST-Knotens."""
    inferrer = DimensionInferrer(known_dims.copy())
    return inferrer.visit(node)


def _get_var_name(node) -> Optional[str]:
    """Extrahiert den Variablennamen, wenn der Knoten eine einzelne Variable ist."""
    if isinstance(node, ast.Name):
        return node.id
    return None


def propagate_all_units(equations: Dict[str, str], known_units: Dict[str, str],
                        max_iterations: int = 10) -> Dict[str, str]:
    """
    Propagiert Einheiten durch alle Gleichungen mittels Fixpunkt-Iteration.

    Args:
        equations: Dict von parsed_equation zu original_equation
        known_units: Dict von Variablen zu bekannten Einheiten
        max_iterations: Maximale Anzahl Iterationen

    Returns:
        Dict von allen abgeleiteten Einheiten (neue + bekannte)
    """
    if not PINT_AVAILABLE:
        return {}

    all_units = known_units.copy()

    for iteration in range(max_iterations):
        found_new = False

        for parsed_eq, original_eq in equations.items():
            # Analysiere Original-Gleichung (lesbarer)
            newly_inferred = analyze_equation(original_eq, all_units)

            for var, unit in newly_inferred.items():
                if var not in all_units and unit is not None:
                    all_units[var] = unit
                    found_new = True

        if not found_new:
            break

    # Gib nur neue Einheiten zurück (nicht die ursprünglich bekannten)
    return {var: unit for var, unit in all_units.items() if var not in known_units}


# ============================================================================
# Unit Consistency Checking
# ============================================================================

def find_equations_for_variable(var: str, equations: Dict[str, str]) -> Dict[str, str]:
    """
    Findet alle Gleichungen, in denen eine Variable vorkommt.

    Args:
        var: Variablenname
        equations: Dict von parsed_equation zu original_equation

    Returns:
        Dict von {original_equation: parsed_equation} wo var vorkommt
    """
    result = {}
    pattern = rf'\b{re.escape(var)}\b'

    for parsed_eq, original_eq in equations.items():
        if re.search(pattern, original_eq):
            result[original_eq] = parsed_eq

    return result


def infer_unit_for_var_in_equation(var: str, equation: str, known_units: Dict[str, str]) -> Tuple[Optional[str], str]:
    """
    Leitet die Einheit für eine Variable aus einer bestimmten Gleichung ab.

    Analysiert die Gleichung und berechnet, welche Einheit die Variable haben müsste,
    damit die Gleichung dimensional konsistent ist.

    Args:
        var: Variablenname deren Einheit abgeleitet werden soll
        equation: Gleichung in der Form "left = right"
        known_units: Dict aller bekannten Einheiten (außer var)

    Returns:
        (inferred_unit, explanation)
        z.B. ("kJ", "aus Addition mit h_1 [kJ/kg]")
             ("bar*m^3", "aus Produkt p*V")
             (None, "konnte nicht abgeleitet werden")
    """
    if not PINT_AVAILABLE:
        return None, "pint nicht verfügbar"

    # Entferne var aus known_units für diese Analyse
    analysis_units = {k: v for k, v in known_units.items() if k != var}

    # Parse die Gleichung
    left_str = None
    right_str = None

    if '=' in equation:
        # Finde das = Zeichen (nicht ==)
        eq_pos = -1
        for i, c in enumerate(equation):
            if c == '=' and (i == 0 or equation[i-1] not in '!=<>') and (i == len(equation)-1 or equation[i+1] != '='):
                eq_pos = i
                break
        if eq_pos > 0:
            left_str = equation[:eq_pos].strip()
            right_str = equation[eq_pos+1:].strip()

    if left_str is None or right_str is None:
        return None, "Gleichung konnte nicht geparst werden"

    # Finde wo var steht
    var_pattern = rf'\b{re.escape(var)}\b'
    var_in_left = bool(re.search(var_pattern, left_str))
    var_in_right = bool(re.search(var_pattern, right_str))

    # Fall 1: var = ausdruck → Einheit des Ausdrucks (rohe Einheit)
    if var_in_left and not var_in_right:
        try:
            left_ast = ast.parse(left_str, mode='eval')
            if isinstance(left_ast.body, ast.Name) and left_ast.body.id == var:
                # var = expr → rohe Einheit von expr (nicht normalisiert!)
                raw_unit = _build_raw_unit_from_expression(right_str, analysis_units)
                if raw_unit is not None:
                    explanation = f"aus Zuweisung: {var} = ..."
                    return raw_unit, explanation
        except:
            pass

    # Fall 2: ausdruck = var → Einheit des Ausdrucks (rohe Einheit)
    if var_in_right and not var_in_left:
        try:
            right_ast = ast.parse(right_str, mode='eval')
            if isinstance(right_ast.body, ast.Name) and right_ast.body.id == var:
                # expr = var → rohe Einheit von expr (nicht normalisiert!)
                raw_unit = _build_raw_unit_from_expression(left_str, analysis_units)
                if raw_unit is not None:
                    explanation = f"aus Zuweisung: ... = {var}"
                    return raw_unit, explanation
        except:
            pass

    # Fall 3: var in Addition/Subtraktion → gleiche Einheit wie andere Terme
    inferred = _infer_from_additive_context(var, left_str, right_str, analysis_units)
    if inferred:
        return inferred

    # Fall 4: var in Multiplikation → aus Division mit anderen Faktoren
    inferred = _infer_from_multiplicative_context(var, left_str, right_str, analysis_units)
    if inferred:
        return inferred

    return None, "konnte Einheit nicht ableiten"


def _compute_expression_dimension(expr_str: str, known_units: Dict[str, str]) -> DimensionInfo:
    """Berechnet die Dimension eines Ausdrucks."""
    if not PINT_AVAILABLE:
        return DimensionInfo(None)

    known_dims = {}
    for v, unit in known_units.items():
        quantity = get_dimension_from_unit(unit) if unit else ureg.Quantity(1.0, 'dimensionless')
        known_dims[v] = DimensionInfo(unit if unit else "", quantity)

    inferrer = DimensionInferrer(known_dims)
    dim, _ = inferrer.infer_from_expression(expr_str)
    return dim


def _infer_from_additive_context(var: str, left_str: str, right_str: str,
                                  known_units: Dict[str, str]) -> Optional[Tuple[str, str]]:
    """
    Inferiert Einheit wenn var in Addition/Subtraktion vorkommt.

    Bei var + x = y oder x + var = y: var hat gleiche Einheit wie x und y
    """
    if not PINT_AVAILABLE:
        return None

    # Kombiniere beide Seiten zu: left - right = 0
    combined = f"({left_str}) - ({right_str})"

    try:
        tree = ast.parse(combined, mode='eval')
        terms = _extract_additive_terms(tree.body)

        known_dims = {}
        for v, unit in known_units.items():
            quantity = get_dimension_from_unit(unit) if unit else ureg.Quantity(1.0, 'dimensionless')
            known_dims[v] = DimensionInfo(unit if unit else "", quantity)

        # Finde Terme mit var und Terme ohne var (mit bekannter Einheit)
        for term in terms:
            term_str = ast.unparse(term) if hasattr(ast, 'unparse') else str(term)

            # Ist var in diesem Term?
            var_pattern = rf'\b{re.escape(var)}\b'
            if re.search(var_pattern, term_str):
                # Dieser Term enthält var - checke ob var allein steht
                if isinstance(term, ast.Name) and term.id == var:
                    # var steht allein, finde Einheit von anderen Termen
                    for other_term in terms:
                        if other_term is not term:
                            other_str = ast.unparse(other_term) if hasattr(ast, 'unparse') else str(other_term)
                            if not re.search(var_pattern, other_str):
                                dim = _compute_expression_dimension(other_str, known_units)
                                if dim.is_known:
                                    explanation = f"aus Addition/Subtraktion mit {other_str}"
                                    return dim.unit or "", explanation
                elif isinstance(term, ast.UnaryOp) and isinstance(term.operand, ast.Name) and term.operand.id == var:
                    # -var oder +var steht allein
                    for other_term in terms:
                        if other_term is not term:
                            other_str = ast.unparse(other_term) if hasattr(ast, 'unparse') else str(other_term)
                            if not re.search(var_pattern, other_str):
                                dim = _compute_expression_dimension(other_str, known_units)
                                if dim.is_known:
                                    explanation = f"aus Addition/Subtraktion mit {other_str}"
                                    return dim.unit or "", explanation
    except:
        pass

    return None


def _extract_additive_terms(node) -> list:
    """Extrahiert alle Terme einer Addition/Subtraktion."""
    terms = []

    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        terms.extend(_extract_additive_terms(node.left))
        if isinstance(node.op, ast.Sub):
            # Subtraktion: rechte Seite negieren
            terms.append(ast.UnaryOp(op=ast.USub(), operand=node.right))
        else:
            terms.extend(_extract_additive_terms(node.right))
    else:
        terms.append(node)

    return terms


def _build_raw_unit_from_expression(expr_str: str, known_units: Dict[str, str]) -> Optional[str]:
    """
    Baut eine "rohe" Einheiten-Darstellung aus einem Ausdruck.

    Im Gegensatz zu unit_from_quantity normalisiert diese Funktion NICHT,
    sondern behält die originale Struktur bei (z.B. bar*m^3 statt kJ).
    """
    try:
        tree = ast.parse(expr_str, mode='eval')
        return _build_raw_unit_from_node(tree.body, known_units)
    except:
        return None


def _build_raw_unit_from_node(node, known_units: Dict[str, str]) -> Optional[str]:
    """Rekursive Hilfsfunktion für _build_raw_unit_from_expression."""
    if isinstance(node, ast.Name):
        var = node.id
        if var in known_units:
            return known_units[var] if known_units[var] else ""
        return None

    elif isinstance(node, (ast.Constant, ast.Num)):
        return ""  # Zahlen sind dimensionslos

    elif isinstance(node, ast.UnaryOp):
        return _build_raw_unit_from_node(node.operand, known_units)

    elif isinstance(node, ast.BinOp):
        left = _build_raw_unit_from_node(node.left, known_units)
        right = _build_raw_unit_from_node(node.right, known_units)

        if left is None or right is None:
            return None

        if isinstance(node.op, ast.Mult):
            # Beide dimensionslos
            if not left and not right:
                return ""
            # Einer dimensionslos
            if not left:
                return right
            if not right:
                return left
            # Beide haben Einheiten
            return f"{left}*{right}"

        elif isinstance(node.op, ast.Div):
            if not left and not right:
                return ""
            if not right:
                return left
            if not left:
                return f"1/{right}"
            return f"{left}/{right}"

        elif isinstance(node.op, (ast.Add, ast.Sub)):
            # Bei Addition/Subtraktion: beide gleich, nimm eine
            if left:
                return left
            return right

        elif isinstance(node.op, ast.Pow):
            # Potenz: nur wenn Exponent konstant
            if left and isinstance(node.right, (ast.Constant, ast.Num)):
                exp = node.right.value if isinstance(node.right, ast.Constant) else node.right.n
                if exp == 2:
                    return f"{left}^2"
                elif exp == 0.5:
                    return f"sqrt({left})"
                return f"{left}^{exp}"
            return left if left else ""

    return None


def _infer_from_multiplicative_context(var: str, left_str: str, right_str: str,
                                        known_units: Dict[str, str]) -> Optional[Tuple[str, str]]:
    """
    Inferiert Einheit wenn var in Multiplikation/Division vorkommt.

    Bei var * x = y: var = y / x
    Bei var / x = y: var = y * x
    Bei x / var = y: var = x / y

    Gibt die "rohe" Einheit zurück (z.B. bar*m^3), nicht normalisiert.
    """
    if not PINT_AVAILABLE:
        return None

    var_pattern = rf'\b{re.escape(var)}\b'

    # Prüfe ob eine Seite die Form "var * expr" oder "expr * var" hat
    for expr_with_var, other_expr in [(left_str, right_str), (right_str, left_str)]:
        if not re.search(var_pattern, expr_with_var):
            continue
        if re.search(var_pattern, other_expr):
            continue  # var ist in beiden Seiten - komplizierter

        try:
            tree = ast.parse(expr_with_var, mode='eval')
            node = tree.body

            # var * expr
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
                if isinstance(node.left, ast.Name) and node.left.id == var:
                    # var * right_factor = other_expr → var = other_expr / right_factor
                    right_str_inner = ast.unparse(node.right) if hasattr(ast, 'unparse') else None
                    if right_str_inner:
                        # Baue rohe Einheit: other_unit / factor_unit
                        other_raw = _build_raw_unit_from_expression(other_expr, known_units)
                        factor_raw = _build_raw_unit_from_expression(right_str_inner, known_units)
                        if other_raw is not None and factor_raw is not None:
                            if not factor_raw:
                                raw_unit = other_raw
                            elif not other_raw:
                                raw_unit = f"1/{factor_raw}"
                            else:
                                raw_unit = f"{other_raw}/{factor_raw}"
                            explanation = f"aus Produkt {var} * {right_str_inner}"
                            return raw_unit, explanation

                elif isinstance(node.right, ast.Name) and node.right.id == var:
                    # left_factor * var = other_expr → var = other_expr / left_factor
                    left_str_inner = ast.unparse(node.left) if hasattr(ast, 'unparse') else None
                    if left_str_inner:
                        other_raw = _build_raw_unit_from_expression(other_expr, known_units)
                        factor_raw = _build_raw_unit_from_expression(left_str_inner, known_units)
                        if other_raw is not None and factor_raw is not None:
                            if not factor_raw:
                                raw_unit = other_raw
                            elif not other_raw:
                                raw_unit = f"1/{factor_raw}"
                            else:
                                raw_unit = f"{other_raw}/{factor_raw}"
                            explanation = f"aus Produkt {left_str_inner} * {var}"
                            return raw_unit, explanation

            # -var * expr oder expr * -var
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
                if isinstance(node.left, ast.UnaryOp) and isinstance(node.left.op, ast.USub):
                    if isinstance(node.left.operand, ast.Name) and node.left.operand.id == var:
                        # -var * right_factor = other_expr
                        right_str_inner = ast.unparse(node.right) if hasattr(ast, 'unparse') else None
                        if right_str_inner:
                            other_raw = _build_raw_unit_from_expression(other_expr, known_units)
                            factor_raw = _build_raw_unit_from_expression(right_str_inner, known_units)
                            if other_raw is not None and factor_raw is not None:
                                if not factor_raw:
                                    raw_unit = other_raw
                                elif not other_raw:
                                    raw_unit = f"1/{factor_raw}"
                                else:
                                    raw_unit = f"{other_raw}/{factor_raw}"
                                explanation = f"aus Produkt -{var} * {right_str_inner}"
                                return raw_unit, explanation

        except:
            pass

    return None


def check_unit_consistency(var: str, units_per_eq: Dict[str, str]) -> Optional['UnitWarning']:
    """
    Prüft ob alle abgeleiteten Einheiten für eine Variable kompatibel sind.

    Verwendet pint um festzustellen, ob die Einheiten konvertierbar sind
    und berechnet den Konversionsfaktor.

    Args:
        var: Variablenname
        units_per_eq: Dict von {equation: inferred_unit}

    Returns:
        UnitWarning wenn Konflikt gefunden, sonst None
    """
    from solver import UnitWarning

    if not PINT_AVAILABLE:
        return None

    if len(units_per_eq) < 2:
        return None

    # Sammle alle verschiedenen Einheiten
    unique_units = {}
    for eq, unit in units_per_eq.items():
        if unit:  # Ignoriere leere/dimensionslose
            unique_units[eq] = unit

    if len(unique_units) < 2:
        return None

    # Vergleiche alle Paare
    eqs = list(unique_units.keys())
    units = list(unique_units.values())

    # Prüfe ob alle Einheiten identisch sind (String-Vergleich)
    # und ob sie konvertierbar sind (pint-Vergleich)
    reference_unit = units[0]
    reference_qty = get_dimension_from_unit(reference_unit)

    incompatible = []
    conversion_factors = {}

    for i in range(1, len(units)):
        other_unit = units[i]

        # String-Vergleich (ignoriere Reihenfolge bei Multiplikation)
        if _units_are_identical(reference_unit, other_unit):
            continue  # Identische Einheiten - OK

        # Pint-Vergleich für Konversionsfaktor
        other_qty = get_dimension_from_unit(other_unit)

        if reference_qty is None or other_qty is None:
            # Können Einheiten nicht parsen - prüfe auf bekannte Konflikte
            factor = _check_known_unit_conflict(reference_unit, other_unit)
            if factor and factor != 1.0:
                incompatible.append((eqs[i], other_unit, factor))
                conversion_factors[eqs[i]] = factor
            continue

        try:
            # Versuche Konversion
            converted = other_qty.to(reference_qty.units)
            factor = float(converted.magnitude / reference_qty.magnitude)

            # Faktor nahe 1 = kompatibel (gleiche Einheit, nur andere Schreibweise)
            if abs(factor - 1.0) > 0.01:  # Mehr als 1% Unterschied
                # Zeige den intuitiveren Faktor (immer >= 1)
                display_factor = factor if factor >= 1.0 else 1.0 / factor
                incompatible.append((eqs[i], other_unit, display_factor))
                conversion_factors[eqs[i]] = display_factor
        except pint.DimensionalityError:
            # Verschiedene Dimensionen - sollte ein Fehler sein
            # aber prüfe auf bekannte Druck*Volumen vs Energie Fälle
            factor = _check_known_unit_conflict(reference_unit, other_unit)
            if factor and factor != 1.0:
                incompatible.append((eqs[i], other_unit, factor))
                conversion_factors[eqs[i]] = factor

    if not incompatible:
        return None

    # Erstelle Warnung
    all_equations = eqs
    explanation_parts = [f"{var} hat unterschiedliche Einheiten:"]
    explanation_parts.append(f"  • {eqs[0]}: {reference_unit}")

    for eq, unit, factor in incompatible:
        explanation_parts.append(f"  • {eq}: {unit} (Faktor {factor:.1f})")

    # Berechne maximalen Konversionsfaktor
    max_factor = max(abs(f) for f in conversion_factors.values()) if conversion_factors else 1.0

    explanation_parts.append(f"\n⚠ Achtung: Faktor {max_factor:.0f} Unterschied!")

    return UnitWarning(
        variable=var,
        equations=all_equations,
        units=unique_units,
        explanation="\n".join(explanation_parts),
        conversion_factor=max_factor
    )


def _units_are_identical(unit1: str, unit2: str) -> bool:
    """Prüft ob zwei Einheiten-Strings identisch sind (ignoriert Reihenfolge)."""
    if not unit1 and not unit2:
        return True
    if not unit1 or not unit2:
        return False

    # Normalisiere Strings
    u1 = unit1.lower().replace(' ', '').replace('^', '**')
    u2 = unit2.lower().replace(' ', '').replace('^', '**')

    if u1 == u2:
        return True

    # Versuche Teile zu extrahieren und zu vergleichen
    parts1 = set(re.split(r'[*/]', u1))
    parts2 = set(re.split(r'[*/]', u2))

    return parts1 == parts2


def _check_known_unit_conflict(unit1: str, unit2: str) -> Optional[float]:
    """
    Prüft auf bekannte Einheiten-Konflikte und gibt den Konversionsfaktor zurück.

    Bekannte Konflikte:
    - bar * m³ vs kJ: Faktor 100
    - Pa * m³ vs J: Faktor 1
    - kPa * m³ vs kJ: Faktor 1
    """
    if not PINT_AVAILABLE:
        return None

    u1_lower = unit1.lower().replace(' ', '').replace('^', '**')
    u2_lower = unit2.lower().replace(' ', '').replace('^', '**')

    # bar*m³ vs kJ
    bar_m3_patterns = ['bar*m**3', 'bar*m^3', 'm**3*bar', 'm^3*bar', 'bar*m3', 'm3*bar']
    kj_patterns = ['kj', 'kilojoule']

    is_u1_bar_m3 = any(p in u1_lower for p in bar_m3_patterns)
    is_u2_bar_m3 = any(p in u2_lower for p in bar_m3_patterns)
    is_u1_kj = any(p in u1_lower for p in kj_patterns)
    is_u2_kj = any(p in u2_lower for p in kj_patterns)

    if (is_u1_bar_m3 and is_u2_kj) or (is_u1_kj and is_u2_bar_m3):
        return 100.0

    return None


def _is_pressure_volume_energy_mismatch(unit1: str, unit2: str) -> bool:
    """Prüft ob ein Druck*Volumen vs Energie Mismatch vorliegt."""
    if not PINT_AVAILABLE:
        return False

    energy_units = {'kJ', 'J', 'kW', 'W', 'kWh', 'MJ'}
    pv_units = {'bar*m^3', 'bar*m³', 'Pa*m^3', 'kPa*m^3', 'bar·m³', 'bar·m^3'}

    u1_lower = unit1.lower().replace(' ', '')
    u2_lower = unit2.lower().replace(' ', '')

    return (any(e.lower() in u1_lower for e in energy_units) and
            any(p.lower() in u2_lower for p in pv_units)) or \
           (any(e.lower() in u2_lower for e in energy_units) and
            any(p.lower() in u1_lower for p in pv_units))


def _get_pressure_volume_factor(unit1: str, unit2: str) -> float:
    """Berechnet den Faktor zwischen Druck*Volumen und Energie."""
    if not PINT_AVAILABLE:
        return 1.0

    try:
        # bar * m³ = 100 kJ
        pv_to_kJ = ureg.Quantity(1.0, 'bar * m^3').to('kJ').magnitude
        return pv_to_kJ  # ~100
    except:
        return 100.0  # Fallback


def check_all_unit_consistency(solution: Dict[str, float],
                                equations: Dict[str, str],
                                known_units: Dict[str, str]) -> list:
    """
    Prüft die Einheiten-Konsistenz für alle Variablen in der Lösung.

    Für jede Variable wird geprüft, ob sie in verschiedenen Gleichungen
    konsistente Einheiten hat.

    Args:
        solution: Dict von {variable: value} der Lösung
        equations: Dict von {parsed_equation: original_equation}
        known_units: Dict von {variable: unit} bekannter Einheiten

    Returns:
        Liste von UnitWarning für inkonsistente Variablen
    """
    from solver import UnitWarning

    warnings = []

    if not PINT_AVAILABLE:
        return warnings

    for var in solution.keys():
        # Finde alle Gleichungen mit dieser Variable
        var_equations = find_equations_for_variable(var, equations)

        if len(var_equations) < 2:
            continue  # Nur in einer Gleichung - keine Konsistenzprüfung möglich

        # Leite Einheit für jede Gleichung ab
        units_per_eq = {}
        for original_eq, parsed_eq in var_equations.items():
            unit, explanation = infer_unit_for_var_in_equation(var, original_eq, known_units)
            if unit is not None:
                units_per_eq[original_eq] = unit

        # Prüfe Konsistenz
        warning = check_unit_consistency(var, units_per_eq)
        if warning:
            warnings.append(warning)

    return warnings


# Test
if __name__ == "__main__":
    print("=== Unit Constraint Propagation Test ===\n")

    # Test 1: Isentroper Wirkungsgrad
    print("Test 1: eta = (h_2 - h_1) / (h_2s - h_1)")
    known = {
        'h_1': 'kJ/kg',
        'h_2s': 'kJ/kg',
        'eta_s_i_T': ''  # dimensionslos
    }
    eq = "eta_s_i_T = (h_2 - h_1) / (h_2s - h_1)"
    result = analyze_equation(eq, known)
    print(f"  Bekannt: {known}")
    print(f"  Abgeleitet: {result}")
    print()

    # Test 2: Einfache Zuweisung
    print("Test 2: h_2 = h_1 + (h_2s - h_1) * eta")
    known2 = {
        'h_1': 'kJ/kg',
        'h_2s': 'kJ/kg',
        'eta': ''
    }
    eq2 = "h_2 = h_1 + (h_2s - h_1) * eta"
    result2 = analyze_equation(eq2, known2)
    print(f"  Bekannt: {known2}")
    print(f"  Abgeleitet: {result2}")
    print()

    # Test 3: Solver-Format
    print("Test 3: Solver-Format (h_2) - (h_1 + dh)")
    known3 = {
        'h_1': 'kJ/kg',
        'dh': 'kJ/kg'
    }
    eq3 = "(h_2) - (h_1 + dh)"
    result3 = analyze_equation(eq3, known3)
    print(f"  Bekannt: {known3}")
    print(f"  Abgeleitet: {result3}")
