"""
Dimensional Constraint Propagation für den HVAC Equation Solver.

Ermöglicht die automatische Erkennung von Einheiten bei impliziten Gleichungen
durch Analyse der Gleichungsstruktur und Propagation bekannter Dimensionen.

Beispiel:
    eta = (h_2 - h_1) / (h_2s - h_1)

    Wenn h_1, h_2s = kJ/kg bekannt und eta dimensionslos,
    dann muss h_2 auch kJ/kg sein.

Konventionen:
- Einheiten-Strings sind ANZEIGE-Labels (z.B. 'kW', 'kJ/kg', 'bar', 'um').
  Zahlenwerte sind intern immer SI-Basis; nur die Dimension des Labels zählt.
- Zahlenliterale in Summen sind dimensionsneutral (T_out = T_s - 2 -> K).
- Temperaturen tragen einen "Charakter": absolut (K) oder Differenz (delta_K).
  T_abs - T_abs -> delta_K, T_abs ± delta_K -> K, (T_1 + T_2)/2 -> K.
- Die Propagation ist unabhängig von der Gleichungsreihenfolge (Fixpunkt mit
  Prioritäten: Definition > Struktur > schwache Literal-Hinweise).
"""

import ast
import math
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Dict, Optional, Set, Tuple, Any, List

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
    quantity: Any = None  # pint Quantity (Betrag 1) für Dimensionsrechnung
    # Temperatur-Charakter (nur bei Dimension [temperature] relevant):
    # 1.0 = absolute Temperatur, 0.0 = Temperaturdifferenz, None = unbekannt
    weight: Optional[float] = None
    literal: bool = False          # reines Zahlenliteral (in Summen neutral)
    value: Optional[float] = None  # Zahlenwert bei Literalen
    # Temperatur aus Produkt/Quotient dimensionsbehafteter Größen (q*R, Q/(m*cp)):
    # in einer Summe mit einer bestimmten Temperatur zählt sie als Differenz
    product: bool = False

    @property
    def is_known(self) -> bool:
        return self.unit is not None

    @property
    def is_dimensionless(self) -> bool:
        if self.unit == "" or self.unit == "dimensionless":
            return True
        try:
            return self.quantity is not None and bool(self.quantity.dimensionless)
        except Exception:
            return False


def is_temperature_difference_variable(var_name: str) -> bool:
    """
    Prüft ob ein Variablenname auf eine Temperaturdifferenz hindeutet.

    Erkennungsmuster:
    - Beginnt mit "dT" (z.B. dT_N, dT_log, dT_B)
    - Beginnt mit "delta" (z.B. delta_T, deltaT)

    Args:
        var_name: Name der Variable

    Returns:
        True wenn der Name auf eine Temperaturdifferenz hindeutet
    """
    var_lower = var_name.lower()
    return var_lower.startswith('dt') or var_lower.startswith('delta')


def adjust_unit_for_variable(unit: str, var_name: str) -> str:
    """
    Passt die Einheit basierend auf dem Variablennamen an.

    Speziell für Temperaturdifferenzen: Wenn die Einheit 'K' ist und der
    Variablenname auf eine Differenz hindeutet (dT..., delta...), wird
    'delta_K' zurückgegeben.

    Args:
        unit: Die abgeleitete Einheit (z.B. 'K')
        var_name: Name der Variable

    Returns:
        Angepasste Einheit (z.B. 'delta_K' statt 'K' für Temperaturdifferenzen)
    """
    if unit == 'K' and is_temperature_difference_variable(var_name):
        return 'delta_K'
    return unit


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


# ============================================================================
# Dimensionen -> Anzeige-Labels
# ============================================================================

# Bevorzugte Labels für häufige Dimensionen (erste passende Zeile gewinnt).
# Die Labels sind reine Anzeige-Einheiten; die Werte bleiben SI-Basis.
_LABEL_ENTRIES = [
    ('W', 'kW'),
    ('J', 'kJ'),
    ('Pa', 'bar'),
    ('kg/s', 'kg/s'),
    ('m^3/s', 'm^3/s'),
    ('m/s', 'm/s'),
    ('m/s^2', 'm/s^2'),
    ('N', 'N'),
    ('kg/m^3', 'kg/m^3'),
    ('J/kg', 'kJ/kg'),
    ('J/(kg*K)', 'kJ/(kg*K)'),
    ('K', 'K'),
    ('m', 'm'),
    ('kg', 'kg'),
    ('s', 's'),
    ('W/m^2', 'W/m^2'),
    ('m^2', 'm^2'),
    ('m^3', 'm^3'),
    ('m^3/kg', 'm^3/kg'),
    ('W/(m^2*K)', 'W/(m^2*K)'),
    ('W/(m^2*K^4)', 'W/(m^2*K^4)'),
    ('m^2*K/W', 'm^2*K/W'),
    ('W/(m*K)', 'W/(m*K)'),
    ('K/W', 'K/W'),
    ('W/K', 'W/K'),
    ('J/K', 'J/K'),
    ('1/K', '1/K'),
    ('m^2/s', 'm^2/s'),
    ('Pa*s', 'Pa*s'),
    ('W/m^3', 'W/m^3'),
    ('1/m', '1/m'),
    ('1/s', '1/s'),
    ('K/m', 'K/m'),
    ('m*K/W', 'm*K/W'),
    ('kg/(m^2*s)', 'kg/(m^2*s)'),
    ('N/m', 'N/m'),
    ('K^4', 'K^4'),
    ('m^2*K', 'm^2*K'),
    ('K*s', 'K*s'),
]

# Basis-Dimensionen -> (Symbol für generische Labels, pint-Basiseinheit)
_BASE_DIMS = (
    ('[mass]', 'kg', 'kilogram'),
    ('[length]', 'm', 'meter'),
    ('[time]', 's', 'second'),
    ('[temperature]', 'K', 'kelvin'),
    ('[current]', 'A', 'ampere'),
    ('[substance]', 'mol', 'mole'),
    ('[luminosity]', 'cd', 'candela'),
)

_LABEL_TABLE: Optional[Dict[tuple, str]] = None
_CANON_CACHE: Dict[tuple, Any] = {}


def _dim_key(dim) -> tuple:
    """Hashbarer, sortierter Schlüssel für eine pint-Dimensionalität."""
    try:
        return tuple(sorted((str(k), float(v)) for k, v in dict(dim).items() if v))
    except Exception:
        return tuple()


def _label_table() -> Dict[tuple, str]:
    global _LABEL_TABLE
    if _LABEL_TABLE is None:
        table = {}
        for expr, label in _LABEL_ENTRIES:
            try:
                key = _dim_key(ureg.Quantity(1.0, expr).dimensionality)
            except Exception:
                continue
            table.setdefault(key, label)
        _LABEL_TABLE = table
    return _LABEL_TABLE


def _fmt_exp(exp: float) -> str:
    if float(exp).is_integer():
        return str(int(exp))
    return f"{exp:g}"


def _generic_label(dim) -> str:
    """Lesbares SI-Label aus Basisdimensionen, z.B. 'kg*m^2/(s^3*K)'."""
    d = dict(dim)
    num, den = [], []
    for name, sym, _ in _BASE_DIMS:
        exp = float(d.get(name, 0) or 0)
        if exp > 0:
            num.append(sym if exp == 1 else f"{sym}^{_fmt_exp(exp)}")
        elif exp < 0:
            den.append(sym if exp == -1 else f"{sym}^{_fmt_exp(-exp)}")
    num_str = '*'.join(num) if num else '1'
    if not den:
        return num_str
    den_str = '*'.join(den)
    if len(den) > 1:
        den_str = f"({den_str})"
    return f"{num_str}/{den_str}"


def unit_from_dimensionality(dim) -> str:
    """
    Mappt eine pint-Dimensionalität auf ein Anzeige-Label.

    Bekannte HVAC-Größen erhalten ihr übliches Label (kW, kJ/kg, bar, ...),
    alle anderen ein lesbares SI-Label (z.B. 'K/W', 'm^2/s', 'kg*m/(s^3*K)').
    Dimensionslos -> ''.
    """
    if dim is None or not PINT_AVAILABLE:
        return ""
    try:
        key = _dim_key(dim)
        if not key:
            return ''
        label = _label_table().get(key)
        if label is not None:
            return label
        return _generic_label(dim)
    except Exception:
        return ""


def si_label_from_dimensionality(dim) -> str:
    """
    Lesbares SI-Label einer Dimension für Meldungen (W/m^2, W, K, J/kg, ...).
    Anders als unit_from_dimensionality ohne Anzeige-Präfixe (kW, kJ/kg, bar).
    """
    if dim is None or not PINT_AVAILABLE:
        return ""
    try:
        from units import STANDARD_UNITS
        if dim == ureg.dimensionless.dimensionality:
            return "dimensionslos"
        for pint_unit, label in STANDARD_UNITS:
            if ureg.Quantity(1.0, pint_unit).dimensionality == dim:
                return label
        return unit_from_dimensionality(dim) or str(dim)
    except Exception:
        return str(dim)


def unit_from_quantity(quantity) -> str:
    """Extrahiert ein Anzeige-Label aus einer pint Quantity (nur Dimension zählt)."""
    if quantity is None or not PINT_AVAILABLE:
        return ""
    try:
        return unit_from_dimensionality(quantity.dimensionality)
    except Exception:
        return ""


def _canon_quantity(dim):
    """Quantity(1, SI-Basiseinheiten) mit der gegebenen Dimensionalität (ohne Offset)."""
    key = _dim_key(dim)
    q = _CANON_CACHE.get(key)
    if q is None:
        q = ureg.Quantity(1.0, 'dimensionless')
        known = {name: base for name, _, base in _BASE_DIMS}
        for name, exp in key:
            base = known.get(name)
            if base is None:
                return None
            q = q * ureg.Quantity(1.0, base) ** exp
        _CANON_CACHE[key] = q
    return q


# ============================================================================
# AST-basierte Dimensions-Engine
# ============================================================================

# CoolProp-Funktionen: Ausgabe-Label (SI)
_THERMO_OUT = {
    'enthalpy': 'J/kg',
    'entropy': 'J/(kg*K)',
    'pressure': 'Pa',
    'temperature': 'K',
    'density': 'kg/m^3',
    'volume': 'm^3/kg',
    'quality': '',
    'intenergy': 'J/kg',
    'cp': 'J/(kg*K)',
    'cv': 'J/(kg*K)',
    'viscosity': 'Pa*s',
    'conductivity': 'W/(m*K)',
    'soundspeed': 'm/s',
    'prandtl': '',
}

# HumidAir: Ausgabe-Label abhängig vom ersten Argument
_HUMID_OUT = {
    'h': 'J/kg',
    'w': '',          # kg/kg ist dimensionslos
    'rh': '',
    't': 'K',
    't_dp': 'K',
    't_wb': 'K',
    'rho_tot': 'kg/m^3',
    'rho_a': 'kg/m^3',
    'rho_w': 'kg/m^3',
    'p_w': 'Pa',
    'cp': 'J/(kg*K)',
    'cp_ha': 'J/(kg*K)',
    'p_tot': 'Pa',
}

# Strahlungsfunktionen: (Ausgabe-Label, Dimension für die Rechnung)
# Werte sind SI (Wien -> m, Eb -> W/m^3); die Labels sind Anzeige-Einheiten.
_RADIATION_OUT = {
    'eb': ('W/(m^2*um)', 'W/m^3'),
    'wien': ('um', 'm'),
    'wien_displacement': ('um', 'm'),
    'stefan_boltzmann': ('W/m^2', 'W/m^2'),
    'blackbody': ('', ''),
    'blackbody_cumulative': ('', ''),
}

# Positionsargumente der Strahlungsfunktionen: 'T' = Temperatur, 'L' = Wellenlänge
_RADIATION_ARGS = {
    'eb': ('T', 'L'),
    'blackbody': ('T', 'L', 'L'),
    'blackbody_cumulative': ('T', 'L'),
    'wien': ('T',),
    'wien_displacement': ('T',),
    'stefan_boltzmann': ('T',),
}

_THERMO_FUNCS = set(_THERMO_OUT)
_SPECIAL_FUNCS = _THERMO_FUNCS | {'humidair'} | set(_RADIATION_OUT)

# Mathematische Funktionen (Groß-/Kleinschreibung wie im Parser)
_DIMLESS_ARG_FUNCS = {'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
                      'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh', 'exp'}
_LOG_FUNCS = {'ln', 'log', 'log10'}
_TRANSCENDENTAL = _DIMLESS_ARG_FUNCS | _LOG_FUNCS

# Zeichen die eine Variable als "Differenz" kennzeichnen (Einheit)
_DELTA_MARKERS = ('delta',)


class _DimMismatch(Exception):
    """Inkompatible Dimensionen in einer Summe (strenger Prüfmodus)."""


class _Ctx:
    """Auswertekontext der Dimensions-Engine."""

    def __init__(self, known: Dict[str, DimensionInfo], strict: bool = False,
                 skip_names: Set[str] = frozenset()):
        self.known = known
        self.strict = strict
        self.skip_names = skip_names
        self.missing: Set[str] = set()   # unbekannte Variablen im Ausdruck
        self.opaque = False              # Name ohne auswertbare Einheit


_DIMLESS_Q = None
_TEMP_DIM = None


def _dimless_q():
    global _DIMLESS_Q
    if _DIMLESS_Q is None:
        _DIMLESS_Q = ureg.Quantity(1.0, 'dimensionless')
    return _DIMLESS_Q


def _temp_dim():
    global _TEMP_DIM
    if _TEMP_DIM is None:
        _TEMP_DIM = ureg.kelvin.dimensionality
    return _TEMP_DIM


def _is_temp_q(q) -> bool:
    try:
        return q is not None and q.dimensionality == _temp_dim()
    except Exception:
        return False


def _is_dimless_q(q) -> bool:
    try:
        return q is not None and bool(q.dimensionless)
    except Exception:
        return False


_Q_CACHE: Dict[str, Any] = {}


def _q_for(unit_expr: str):
    """Kanonische Quantity für einen (pint-lesbaren) Einheiten-Ausdruck."""
    if not unit_expr:
        return _dimless_q()
    if unit_expr in _Q_CACHE:
        return _Q_CACHE[unit_expr]
    q = get_dimension_from_unit(unit_expr)
    result = None
    if q is not None:
        try:
            result = _canon_quantity(q.dimensionality)
        except Exception:
            result = None
    if len(_Q_CACHE) > 2000:
        _Q_CACHE.clear()
    _Q_CACHE[unit_expr] = result
    return result


def _literal(value: Optional[float]) -> DimensionInfo:
    return DimensionInfo('', _dimless_q(), weight=0.0, literal=True, value=value)


def _dim(q, weight=None, label=None) -> DimensionInfo:
    """DimensionInfo aus Quantity; Label aus Dimension falls nicht vorgegeben."""
    if q is None:
        return DimensionInfo(None)
    if label is None:
        label = '' if _is_dimless_q(q) else unit_from_quantity(q)
    return DimensionInfo(label, q, weight=weight if _is_temp_q(q) else None)


def _dim_info_from_unit(unit: Optional[str], var: Optional[str] = None) -> DimensionInfo:
    """
    DimensionInfo für eine bekannte Einheit.

    Temperatur-Charakter: 'delta...' oder Variablenname dT.../delta... -> Differenz,
    sonst absolute Temperatur. Nicht auswertbare Einheiten -> 'opak' (Einheit bekannt,
    Dimension aber nicht verwendbar).
    """
    if unit is None:
        return DimensionInfo(None)
    u = unit.strip()
    if u in ('', 'dimensionless', '-', '1'):
        return DimensionInfo('', _dimless_q())
    q = _q_for(u)
    if q is None:
        return DimensionInfo(u, None)
    weight = None
    if _is_temp_q(q):
        if any(m in u.lower() for m in _DELTA_MARKERS) or \
                (var is not None and is_temperature_difference_variable(var)):
            weight = 0.0
        else:
            weight = 1.0
    return DimensionInfo(u, q, weight=weight)


def _flatten_additive(node, sign: int = 1) -> List[Tuple[int, ast.AST]]:
    """Zerlegt eine +/- Kette in (Vorzeichen, Term)-Paare (unäres Minus eingerechnet)."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _flatten_additive(node.left, sign) + _flatten_additive(node.right, sign)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
        return _flatten_additive(node.left, sign) + _flatten_additive(node.right, -sign)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return _flatten_additive(node.operand, -sign)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
        return _flatten_additive(node.operand, sign)
    return [(sign, node)]


def _func_name(node) -> Optional[str]:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        return node.func.id
    return None


def _eval(node, ctx: _Ctx) -> DimensionInfo:
    """Dimension eines AST-Knotens (unbekannt -> DimensionInfo(None))."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return _literal(float(node.value))
        return DimensionInfo(None)

    if isinstance(node, ast.Name):
        name = node.id
        if name in ctx.known:
            d = ctx.known[name]
            if d.quantity is None:
                ctx.opaque = True
            return d
        if name == 'pi':
            return _literal(math.pi)
        if name in ctx.skip_names:
            ctx.opaque = True
            return DimensionInfo(None)
        ctx.missing.add(name)
        return DimensionInfo(None)

    if isinstance(node, ast.UnaryOp):
        inner = _eval(node.operand, ctx)
        if isinstance(node.op, ast.USub):
            return _negate(inner)
        if isinstance(node.op, ast.UAdd):
            return inner
        return DimensionInfo(None)

    if isinstance(node, ast.BinOp):
        if isinstance(node.op, (ast.Add, ast.Sub)):
            return _eval_chain(_flatten_additive(node), ctx)
        left = _eval(node.left, ctx)
        right = _eval(node.right, ctx)
        if isinstance(node.op, ast.Mult):
            return _mul(left, right)
        if isinstance(node.op, ast.Div):
            return _div(left, right)
        if isinstance(node.op, ast.Pow):
            return _pow(left, right)
        return DimensionInfo(None)

    if isinstance(node, ast.Call):
        return _eval_call(node, ctx)

    return DimensionInfo(None)


def _negate(d: DimensionInfo) -> DimensionInfo:
    if d.quantity is None:
        return d
    return replace(d,
                   weight=-d.weight if d.weight is not None else None,
                   value=-d.value if d.value is not None else None)


def _eval_chain(terms, ctx: _Ctx) -> DimensionInfo:
    """Summe: Literale sind neutral; alle übrigen Terme haben dieselbe Dimension."""
    evals = [(s, _eval(n, ctx)) for s, n in terms]
    nonlit = [(s, d) for s, d in evals if not d.literal]
    if not nonlit:
        values = [d.value for _, d in evals]
        value = sum(s * v for (s, _), v in zip(evals, values)) if None not in values else None
        return _literal(value)
    known = [(s, d) for s, d in nonlit if d.quantity is not None]
    if not known:
        return DimensionInfo(None)
    ref = known[0][1]
    if ctx.strict:
        for _, d in known[1:]:
            if d.quantity.dimensionality != ref.quantity.dimensionality:
                raise _DimMismatch()
    weight = None
    if len(known) == len(nonlit) and _is_temp_q(ref.quantity):
        weight = _sum_weights(nonlit)
    if len(nonlit) == 1 and len(evals) == 1:
        return ref
    d = _dim(ref.quantity, weight)
    if weight is None and len(nonlit) == 1 and ref.product:
        d = replace(d, product=True)
    return d


def _sum_weights(items) -> Optional[float]:
    """
    Temperatur-Charakter einer Summe Σ s_i·w_i.

    Literale zählen 0. Produkt-Temperaturen (q*R, Q/(m*cp)) zählen als Differenz (0),
    sofern mindestens ein anderer Term einen bestimmten Charakter hat; sonst unbestimmt.
    """
    total = 0.0
    determined = False
    for s, d in items:
        if d is None:
            return None
        if d.literal:
            continue
        if d.weight is not None:
            total += s * d.weight
            determined = True
        elif not d.product:
            return None
    return total if determined else None


def _scale_weight(w: Optional[float], factor: DimensionInfo, divide: bool = False) -> Optional[float]:
    """Temperatur-Charakter bei Multiplikation/Division mit dimensionslosem Faktor."""
    if w is None or factor.quantity is None or not _is_dimless_q(factor.quantity):
        return None
    if factor.literal and factor.value is not None:
        if divide:
            return w / factor.value if factor.value != 0 else None
        return w * factor.value
    return 0.0 if w == 0 else None


def _mul(l: DimensionInfo, r: DimensionInfo) -> DimensionInfo:
    if l.quantity is None or r.quantity is None:
        return DimensionInfo(None)
    try:
        q = l.quantity * r.quantity
    except Exception:
        return DimensionInfo(None)
    weight = None
    product = False
    if _is_temp_q(q):
        if _is_dimless_q(l.quantity):
            weight = _scale_weight(r.weight, l)
            product = r.product and weight is None
        elif _is_dimless_q(r.quantity):
            weight = _scale_weight(l.weight, r)
            product = l.product and weight is None
        else:
            product = True
    d = _dim(q, weight)
    if product:
        d = replace(d, product=True)
    if l.literal and r.literal:
        val = l.value * r.value if l.value is not None and r.value is not None else None
        d = replace(d, literal=True, value=val, weight=0.0)
    return d


def _div(l: DimensionInfo, r: DimensionInfo) -> DimensionInfo:
    if l.quantity is None or r.quantity is None:
        return DimensionInfo(None)
    try:
        q = l.quantity / r.quantity
    except Exception:
        return DimensionInfo(None)
    weight = None
    product = False
    if _is_temp_q(q):
        if _is_dimless_q(r.quantity):
            weight = _scale_weight(l.weight, r, divide=True)
            product = l.product and weight is None
        else:
            product = True
    d = _dim(q, weight)
    if product:
        d = replace(d, product=True)
    if l.literal and r.literal:
        val = None
        if l.value is not None and r.value not in (None, 0):
            val = l.value / r.value
        d = replace(d, literal=True, value=val, weight=0.0)
    return d


def _pow(base: DimensionInfo, exp: DimensionInfo) -> DimensionInfo:
    if base.quantity is None:
        return DimensionInfo(None)
    if exp.literal and exp.value is not None:
        n = exp.value
        try:
            q = base.quantity ** n
        except Exception:
            return DimensionInfo(None)
        d = _dim(q, base.weight if n == 1 else None)
        if base.literal:
            try:
                val = base.value ** n if base.value is not None else None
                if isinstance(val, complex):
                    val = None
            except Exception:
                val = None
            d = replace(d, literal=True, value=val, weight=0.0)
        return d
    # Symbolischer Exponent: nur bei dimensionsloser Basis bestimmbar
    if exp.quantity is not None and not _is_dimless_q(exp.quantity):
        return DimensionInfo(None)
    if _is_dimless_q(base.quantity):
        return DimensionInfo('', _dimless_q(), literal=base.literal and exp.literal)
    return DimensionInfo(None)


def _special_call_dim(node) -> Optional[DimensionInfo]:
    """Dimension von CoolProp-/HumidAir-/Strahlungs-Aufrufen (Argumente egal)."""
    fname = _func_name(node)
    if fname is None:
        return None
    fl = fname.lower()
    if fl in _THERMO_OUT:
        label = _THERMO_OUT[fl]
        q = _q_for(label)
        return DimensionInfo(label, q, weight=1.0 if fl == 'temperature' else None)
    if fl == 'humidair':
        if not node.args:
            return DimensionInfo(None)
        first = node.args[0]
        prop = None
        if isinstance(first, ast.Name):
            prop = first.id.lower()
        elif isinstance(first, ast.Constant) and isinstance(first.value, str):
            prop = first.value.lower()
        if prop not in _HUMID_OUT:
            return DimensionInfo(None)
        label = _HUMID_OUT[prop]
        q = _q_for(label)
        return DimensionInfo(label, q, weight=1.0 if prop in ('t', 't_dp', 't_wb') else None)
    if fl in _RADIATION_OUT:
        label, dim_expr = _RADIATION_OUT[fl]
        return DimensionInfo(label, _q_for(dim_expr))
    return None


def _same_dimension_result(dims, ctx: _Ctx) -> DimensionInfo:
    """
    Ergebnis einer Funktion, deren Argumente alle dieselbe Dimension haben und
    die eines davon zurückgibt (max, min, Zweige von IF). Zahlenliterale sind
    dimensionsneutral; bei strenger Prüfung ist eine abweichende Dimension ein Fehler.
    """
    nonlit = [d for d in dims if not d.literal]
    known = [d for d in nonlit if d.quantity is not None]
    if not known:
        return DimensionInfo(None)
    ref = known[0]
    if ctx.strict:
        for d in known[1:]:
            if d.quantity.dimensionality != ref.quantity.dimensionality:
                raise _DimMismatch()
    weight = None
    if len(known) == len(nonlit):
        ws = {d.weight for d in nonlit}
        if len(ws) == 1:
            weight = ws.pop()
    if len(dims) == 1:
        return ref
    return _dim(ref.quantity, weight)


def _eval_call(node, ctx: _Ctx) -> DimensionInfo:
    fname = _func_name(node)
    if fname is None:
        return DimensionInfo(None)

    special = _special_call_dim(node)
    if special is not None:
        # Argumente von Stoffwert-/Strahlungsfunktionen werden nicht geprüft
        return special

    args = node.args
    if fname in _TRANSCENDENTAL:
        dims = [_eval(a, ctx) for a in args]
        lit = bool(dims) and all(d.literal for d in dims)
        return DimensionInfo('', _dimless_q(), literal=lit, weight=0.0 if lit else None)

    if fname == 'sqrt':
        if not args:
            return DimensionInfo(None)
        a = _eval(args[0], ctx)
        if a.quantity is None:
            return DimensionInfo(None)
        try:
            q = a.quantity ** 0.5
        except Exception:
            return DimensionInfo(None)
        d = _dim(q)
        if a.literal:
            val = math.sqrt(a.value) if a.value is not None and a.value >= 0 else None
            d = replace(d, literal=True, value=val, weight=0.0)
        return d

    if fname == 'abs':
        if not args:
            return DimensionInfo(None)
        a = _eval(args[0], ctx)
        if a.literal and a.value is not None:
            return replace(a, value=abs(a.value))
        return a

    if fname == 'IF':
        # IF(a, b, x, y, z): a und b werden verglichen (gleiche Dimension), das
        # Ergebnis hat die Dimension der Zweige x, y, z (wie max/min)
        if len(args) != 5:
            return DimensionInfo(None)
        compared = [d for d in (_eval(a, ctx) for a in args[:2])
                    if not d.literal and d.quantity is not None]
        if (ctx.strict and len(compared) == 2
                and compared[0].quantity.dimensionality != compared[1].quantity.dimensionality):
            raise _DimMismatch()
        dims = [_eval(a, ctx) for a in args[2:]]
        if all(d.literal for d in dims):
            cmp = [_eval(a, ctx) for a in args[:2]]
            values = [d.value for d in cmp + dims]
            if all(d.literal for d in cmp) and None not in values:
                a, b, x, y, z = values
                return _literal(x if a < b else (y if a == b else z))
            return _literal(None)
        return _same_dimension_result(dims, ctx)

    if fname in ('max', 'min'):
        dims = [_eval(a, ctx) for a in args]
        if not dims:
            return DimensionInfo(None)
        nonlit = [d for d in dims if not d.literal]
        if not nonlit:
            vals = [d.value for d in dims]
            if None in vals:
                return _literal(None)
            return _literal(max(vals) if fname == 'max' else min(vals))
        return _same_dimension_result(dims, ctx)

    # Unbekannte Funktion
    return DimensionInfo(None)


def _eval_collect(node, known: Dict[str, DimensionInfo]) -> Tuple[DimensionInfo, Set[str]]:
    """Dimension + Menge der unbekannten Variablen eines Knotens."""
    ctx = _Ctx(known)
    try:
        d = _eval(node, ctx)
    except Exception:
        d = DimensionInfo(None)
    return d, ctx.missing


# ============================================================================
# Gleichungen parsen
# ============================================================================

def _split_equation(equation: str) -> Optional[Tuple[str, str]]:
    """
    Teilt eine Gleichung am ersten '=' auf Klammerebene 0
    (nicht in Funktionsargumenten wie T=T_1, nicht in ==, <=, >=, !=).
    """
    depth = 0
    n = len(equation)
    for i, c in enumerate(equation):
        if c in '([{':
            depth += 1
        elif c in ')]}':
            depth -= 1
        elif c == '=' and depth == 0:
            prev = equation[i - 1] if i > 0 else ''
            nxt = equation[i + 1] if i + 1 < n else ''
            if prev in '!<>=' or nxt == '=':
                continue
            left = equation[:i].strip()
            right = equation[i + 1:].strip()
            if left and right:
                return left, right
            return None
    return None


def _split_solver_format(equation: str) -> Optional[Tuple[str, str]]:
    """Solver-Format '(left) - (right)' per Klammer-Zählung zerlegen."""
    if not (equation.startswith('(') and ') - (' in equation):
        return None
    depth = 0
    for i, ch in enumerate(equation):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0:
                rest = equation[i + 1:].strip()
                if rest.startswith('-'):
                    rest = rest[1:].strip()
                    if rest.startswith('(') and rest.endswith(')'):
                        return equation[1:i], rest[1:-1]
                return None
    return None


_PARSE_CACHE: Dict[str, Optional[Tuple[ast.AST, ast.AST]]] = {}


def _parse_equation(equation: str) -> Optional[Tuple[ast.AST, ast.AST]]:
    """
    Parst eine Gleichung zu (links, rechts) AST-Ausdrücken.

    Normale Gleichungen 'links = rechts' haben Vorrang; nur ohne '=' auf
    oberster Ebene wird das Solver-Format '(links) - (rechts)' erkannt.
    """
    if equation in _PARSE_CACHE:
        return _PARSE_CACHE[equation]
    result = None
    try:
        eq = _remove_comments(equation)
        sides = _split_equation(eq)
        if sides is None:
            sides = _split_solver_format(eq)
        if sides is not None:
            left = ast.parse(sides[0].replace('^', '**'), mode='eval').body
            right = ast.parse(sides[1].replace('^', '**'), mode='eval').body
            result = (left, right)
    except Exception:
        result = None
    if len(_PARSE_CACHE) > 5000:
        _PARSE_CACHE.clear()
    _PARSE_CACHE[equation] = result
    return result


# ============================================================================
# Einheiten-Inferenz (Kandidaten mit Rang: 1 = Definition, 2 = Struktur,
# 3 = schwacher Hinweis aus Zahlenliteralen)
# ============================================================================

_RANK_DEFINITION = 1
_RANK_STRUCTURE = 2
_RANK_WEAK = 3


def _add_cand(out, var: str, rank: int, d: DimensionInfo):
    if d is None or d.quantity is None:
        return
    out[var].append((rank, d))


def _target_for(q, weight=None, label=None) -> DimensionInfo:
    return _dim(q, weight, label)


def _rev(node, target: DimensionInfo, known, rank: int, out):
    """Rückwärts-Inferenz: 'node' muss die Dimension von 'target' haben."""
    if target is None or target.quantity is None:
        return
    if isinstance(node, ast.Name):
        if node.id not in known and node.id != 'pi':
            label = target.unit if target.unit is not None else None
            _add_cand(out, node.id, rank, _dim(target.quantity, target.weight, label))
        return

    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            _rev(node.operand, _negate(target), known, rank, out)
        elif isinstance(node.op, ast.UAdd):
            _rev(node.operand, target, known, rank, out)
        return

    if isinstance(node, ast.BinOp):
        if isinstance(node.op, (ast.Add, ast.Sub)):
            _chain_infer(_flatten_additive(node), [(-1, target)], known, rank, out)
            return
        l, ml = _eval_collect(node.left, known)
        r, mr = _eval_collect(node.right, known)
        tq = target.quantity
        try:
            if isinstance(node.op, ast.Mult):
                if l.quantity is not None and r.quantity is not None:
                    if ml:
                        _rev(node.left, l, known, rank, out)
                    if mr:
                        _rev(node.right, r, known, rank, out)
                elif l.quantity is not None:
                    w = _scale_weight(target.weight, l, divide=True)
                    _rev(node.right, _target_for(tq / l.quantity, w), known, rank, out)
                    if ml:
                        _rev(node.left, l, known, rank, out)
                elif r.quantity is not None:
                    w = _scale_weight(target.weight, r, divide=True)
                    _rev(node.left, _target_for(tq / r.quantity, w), known, rank, out)
                    if mr:
                        _rev(node.right, r, known, rank, out)
            elif isinstance(node.op, ast.Div):
                if l.quantity is not None and r.quantity is not None:
                    if ml:
                        _rev(node.left, l, known, rank, out)
                    if mr:
                        _rev(node.right, r, known, rank, out)
                elif r.quantity is not None:
                    w = _scale_weight(target.weight, r)
                    _rev(node.left, _target_for(tq * r.quantity, w), known, rank, out)
                    if mr:
                        _rev(node.right, r, known, rank, out)
                elif l.quantity is not None:
                    _rev(node.right, _target_for(l.quantity / tq), known, rank, out)
                    if ml:
                        _rev(node.left, l, known, rank, out)
            elif isinstance(node.op, ast.Pow):
                e, _ = _eval_collect(node.right, known)
                if l.quantity is None:
                    if e.literal and e.value not in (None, 0):
                        n = e.value
                        _rev(node.left, _target_for(tq ** (1.0 / n), target.weight if n == 1 else None),
                             known, rank, out)
                    elif _is_dimless_q(tq):
                        _rev(node.left, _dim(_dimless_q()), known, rank, out)
                elif ml:
                    _rev(node.left, l, known, rank, out)
                # Ein Exponent ist immer dimensionslos (z.B. n in (m_1/m_0)^n)
                _rev(node.right, _dim(_dimless_q()), known, rank, out)
        except Exception:
            pass
        return

    if isinstance(node, ast.Call):
        fname = _func_name(node)
        if fname == 'abs' and node.args:
            _rev(node.args[0], target, known, rank, out)
        elif fname in ('max', 'min') or (fname == 'IF' and len(node.args) == 5):
            for a in (node.args[2:] if fname == 'IF' else node.args):
                d, miss = _eval_collect(a, known)
                if miss and not d.literal:
                    _rev(a, target, known, rank, out)
            if fname == 'IF':
                # Verglichene Größen a, b haben dieselbe Dimension
                (da, ma), (db, mb) = (_eval_collect(a, known) for a in node.args[:2])
                if ma and not mb and not db.literal and db.quantity is not None:
                    _rev(node.args[0], db, known, rank, out)
                elif mb and not ma and not da.literal and da.quantity is not None:
                    _rev(node.args[1], da, known, rank, out)
        elif fname == 'sqrt' and node.args:
            a, miss = _eval_collect(node.args[0], known)
            try:
                if a.quantity is None:
                    _rev(node.args[0], _target_for(target.quantity ** 2), known, rank, out)
                elif miss:
                    _rev(node.args[0], a, known, rank, out)
            except Exception:
                pass


def _log_arg_name(node) -> Optional[str]:
    """Liefert x für Terme der Form ln(x), c*ln(x), ln(x)*c (x einfacher Name)."""
    call = None
    if _func_name(node) in _LOG_FUNCS:
        call = node
    elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Div)):
        for part in (node.left, node.right):
            if _func_name(part) in _LOG_FUNCS:
                call = part
                break
    if call is not None and call.args and isinstance(call.args[0], ast.Name):
        return call.args[0].id
    return None


def _chain_infer(terms, extras, known, rank: int, out):
    """
    Inferenz in einer Summe  Σ s_i·term_i + Σ s_e·extra_e = 0.

    Alle nicht-literalen Terme haben dieselbe Dimension. Zahlenliterale sind
    neutral; nur wenn KEIN Term eine bekannte Dimension hat, gelten sie als
    schwacher Hinweis auf 'dimensionslos' (Rang 3).
    """
    evals = []
    for s, n in terms:
        d, miss = _eval_collect(n, known)
        evals.append((s, n, d, miss))

    refs = [d for _, d in extras if d is not None and d.quantity is not None and not d.literal]
    refs += [d for _, _, d, miss in evals if d.quantity is not None and not d.literal and not miss]
    refs += [d for _, _, d, miss in evals if d.quantity is not None and not d.literal and miss]
    weak = False
    if refs:
        ref_q = refs[0].quantity
    else:
        has_literal = any(d.literal for _, _, d, _ in evals) or \
            any(d is not None and d.literal for _, d in extras)
        if not has_literal:
            ref_q = None
        else:
            ref_q = _dimless_q()
            weak = True

    if ref_q is not None:
        r = _RANK_WEAK if weak else rank
        is_temp = _is_temp_q(ref_q)
        all_items = [(s, d) for s, _, d, _ in evals] + [(s, d) for s, d in extras]
        for i, (s, n, d, miss) in enumerate(evals):
            if not miss or d.literal:
                continue
            w = None
            if is_temp and not weak:
                others = [(s2, d2) for j, (s2, d2) in enumerate(all_items) if j != i]
                total = _sum_weights(others)
                if total is not None:
                    w = -total / s
            _rev(n, _target_for(ref_q, w), known, r, out)

    # ln(T_2) - ln(T_1): Argumente von Logarithmen in einer Summe sind gleichartig
    log_args = [a for a in (_log_arg_name(n) for _, n, _, _ in evals) if a]
    if len(log_args) >= 2:
        ref = next((known[a] for a in log_args if a in known and known[a].quantity is not None), None)
        if ref is not None:
            for a in log_args:
                if a not in known:
                    _add_cand(out, a, rank, _dim(ref.quantity, ref.weight))


def _function_argument_candidates(side, known, out):
    """T=T_1 -> K, p=p_1 -> Pa, Eb(T, L) -> T: K, L: Länge ..."""
    for node in ast.walk(side):
        fname = _func_name(node)
        if fname is None:
            continue
        fl = fname.lower()
        targets = []
        if fl in _THERMO_FUNCS or fl == 'humidair':
            for kw in node.keywords:
                if kw.arg is None:
                    continue
                expected = FUNCTION_ARGUMENT_UNITS.get(kw.arg.lower())
                if expected is None:
                    continue
                q = _q_for(expected)
                weight = 1.0 if _is_temp_q(q) else None
                targets.append((kw.value, _dim(q, weight, expected if expected else '')))
        elif fl in _RADIATION_ARGS:
            for arg, kind in zip(node.args, _RADIATION_ARGS[fl]):
                if kind == 'T':
                    targets.append((arg, _dim(_q_for('K'), 1.0, 'K')))
                else:
                    targets.append((arg, _dim(_q_for('m'), None, 'um')))
        for value, target in targets:
            if isinstance(value, ast.Name):
                if value.id not in known and value.id != 'pi':
                    _add_cand(out, value.id, _RANK_DEFINITION, target)
            else:
                _, miss = _eval_collect(value, known)
                if miss:
                    _rev(value, target, known, _RANK_STRUCTURE, out)


def _transcendental_candidates(side, known, out):
    """Argumente von sin/cos/exp/... sind dimensionslos; ln(x) nur schwach."""
    for node in ast.walk(side):
        fname = _func_name(node)
        if fname not in _TRANSCENDENTAL:
            continue
        for arg in node.args:
            if isinstance(arg, ast.Name):
                if arg.id in known or arg.id == 'pi':
                    continue
                rank = _RANK_WEAK if fname in _LOG_FUNCS else _RANK_STRUCTURE
                _add_cand(out, arg.id, rank, _dim(_dimless_q()))
            else:
                _, miss = _eval_collect(arg, known)
                if miss:
                    _rev(arg, _dim(_dimless_q()), known, _RANK_STRUCTURE, out)


def _definition_candidates(left, right, known, out):
    """var = ausdruck (vollständig bekannt) -> Rang 1 mit Label des Ausdrucks."""
    for var_node, expr_node in ((left, right), (right, left)):
        d, miss = _eval_collect(expr_node, known)
        if d.quantity is None:
            continue
        if isinstance(var_node, ast.Name):
            var = var_node.id
            if var in known or var == 'pi':
                continue
            if d.literal:
                _add_cand(out, var, _RANK_WEAK, _dim(_dimless_q()))
            else:
                label = d.unit
                if label is None or _is_temp_q(d.quantity):
                    label = None
                elif _is_dimless_q(d.quantity):
                    label = ''
                cand = _dim(d.quantity, d.weight, label)
                _add_cand(out, var, _RANK_STRUCTURE if miss else _RANK_DEFINITION, cand)
        elif (isinstance(var_node, ast.BinOp) and isinstance(var_node.op, ast.Pow)
              and isinstance(var_node.left, ast.Name) and var_node.left.id not in known):
            e, _ = _eval_collect(var_node.right, known)
            if not (e.literal and e.value not in (None, 0)):
                continue
            var = var_node.left.id
            if d.literal:
                _add_cand(out, var, _RANK_WEAK, _dim(_dimless_q()))
                continue
            try:
                q = d.quantity ** (1.0 / e.value)
            except Exception:
                continue
            _add_cand(out, var, _RANK_STRUCTURE if miss else _RANK_DEFINITION, _dim(q))


def _equation_candidates(parsed, known, out):
    left, right = parsed
    for side in (left, right):
        _function_argument_candidates(side, known, out)
        _transcendental_candidates(side, known, out)
    _definition_candidates(left, right, known, out)
    # Gesamte Gleichung als Summe: links - rechts = 0
    terms = _flatten_additive(left, 1) + _flatten_additive(right, -1)
    _chain_infer(terms, [], known, _RANK_STRUCTURE, out)
    # Innere Summen (z.B. UA*(T_s - T_amb)) - Dimension aus den eigenen Termen
    for side in (left, right):
        for node in ast.walk(side):
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
                _chain_infer(_flatten_additive(node), [], known, _RANK_STRUCTURE, out)


def _gather(parsed_eqs, known) -> Dict[str, List[Tuple[int, DimensionInfo]]]:
    out = defaultdict(list)
    for parsed in parsed_eqs:
        try:
            _equation_candidates(parsed, known, out)
        except Exception:
            continue
    return {v: lst for v, lst in out.items() if v not in known and lst}


def _select(cands: List[DimensionInfo]) -> DimensionInfo:
    """Deterministische Auswahl: häufigste Dimension, dann häufigstes Label."""
    groups = defaultdict(list)
    for d in cands:
        groups[_dim_key(d.quantity.dimensionality)].append(d)
    best_key = sorted(groups, key=lambda k: (-len(groups[k]), k))[0]
    group = groups[best_key]
    labels = defaultdict(int)
    for d in group:
        labels[d.unit if d.unit is not None else ''] += 1
    label = sorted(labels, key=lambda u: (-labels[u], u))[0]
    q = group[0].quantity
    if _is_temp_q(q):
        label = 'K'
    elif _is_dimless_q(q):
        label = ''
    return DimensionInfo(label, q)


def _infer_dimensions(parsed_eqs, user_known: Dict[str, DimensionInfo],
                      max_iterations: int = 500) -> Dict[str, DimensionInfo]:
    """
    Fixpunkt-Iteration der Dimensions-Inferenz.

    In jedem Schritt werden ALLE Kandidaten aus dem aktuellen Wissensstand
    gesammelt, und nur die des besten (kleinsten) Rangs übernommen. Damit ist
    das Ergebnis unabhängig von der Gleichungsreihenfolge, und schwache Hinweise
    (Zahlenliterale) werden nur genutzt, wenn nichts Stärkeres verfügbar ist.
    Was aus schwachem Wissen folgt, bleibt schwach (zweite Phase).
    """
    strong: Dict[str, DimensionInfo] = {}
    for _ in range(max_iterations):
        known = {**user_known, **strong}
        cands = _gather(parsed_eqs, known)
        cands = {v: [c for c in lst if c[0] < _RANK_WEAK] for v, lst in cands.items()}
        cands = {v: lst for v, lst in cands.items() if lst}
        if not cands:
            break
        best = min(r for lst in cands.values() for r, _ in lst)
        for v, lst in cands.items():
            sel = [d for r, d in lst if r == best]
            if sel:
                strong[v] = _select(sel)

    weak: Dict[str, DimensionInfo] = {}
    for _ in range(max_iterations):
        known = {**user_known, **strong, **weak}
        cands = _gather(parsed_eqs, known)
        if not cands:
            break
        best = min(r for lst in cands.values() for r, _ in lst)
        for v, lst in cands.items():
            sel = [d for r, d in lst if r == best]
            if sel:
                weak[v] = _select(sel)

    return {**strong, **weak}


# ----------------------------------------------------------------------------
# Temperatur-Charakter (absolut vs. Differenz)
# ----------------------------------------------------------------------------

def _names_in(node) -> Set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _rev_weight(node, w: float, prio: int, out, known, free: Set[str]):
    """Rückwärts-Inferenz des Temperatur-Charakters (Gewicht w) in 'node'."""
    if isinstance(node, ast.Name):
        if node.id in free:
            out[node.id].append((prio, w))
        return
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            _rev_weight(node.operand, -w, prio, out, known, free)
        elif isinstance(node.op, ast.UAdd):
            _rev_weight(node.operand, w, prio, out, known, free)
        return
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, (ast.Add, ast.Sub)):
            _rev_weight_chain(_flatten_additive(node), w, prio, out, known, free)
            return
        l, _ = _eval_collect(node.left, known)
        r, _ = _eval_collect(node.right, known)
        if isinstance(node.op, ast.Mult):
            for factor, other_node, other in ((l, node.right, r), (r, node.left, l)):
                if factor.quantity is not None and _is_dimless_q(factor.quantity) and _is_temp_q(other.quantity):
                    nw = _scale_weight(w, factor, divide=True)
                    if nw is not None:
                        _rev_weight(other_node, nw, prio, out, known, free)
                    return
        elif isinstance(node.op, ast.Div):
            if r.quantity is not None and _is_dimless_q(r.quantity) and _is_temp_q(l.quantity):
                nw = _scale_weight(w, r)
                if nw is not None:
                    _rev_weight(node.left, nw, prio, out, known, free)
        elif isinstance(node.op, ast.Pow):
            e, _ = _eval_collect(node.right, known)
            if e.literal and e.value == 1:
                _rev_weight(node.left, w, prio, out, known, free)
        return
    if isinstance(node, ast.Call):
        fname = _func_name(node)
        if fname == 'abs' and node.args:
            _rev_weight(node.args[0], w, prio, out, known, free)
        elif fname in ('max', 'min'):
            for a in node.args:
                _rev_weight(a, w, prio, out, known, free)
        elif fname == 'IF' and len(node.args) == 5:
            for a in node.args[2:]:
                _rev_weight(a, w, prio, out, known, free)


def _rev_weight_chain(terms, target_w: float, prio: int, out, known, free: Set[str]):
    """Σ s_i·w_i = target_w: löst nach dem Term mit freien Variablen auf."""
    evals = [(s, n, _eval_collect(n, known)[0]) for s, n in terms]
    if not any(_is_temp_q(d.quantity) for _, _, d in evals if not d.literal):
        return
    for i, (s, n, d) in enumerate(evals):
        if not (_names_in(n) & free):
            continue
        total = _sum_weights([(s2, d2) for j, (s2, _, d2) in enumerate(evals) if j != i])
        if total is None:
            continue
        _rev_weight(n, (target_w - total) / s, prio, out, known, free)


def _weight_candidates(parsed, known, free: Set[str], out):
    left, right = parsed
    # Prio 1: Temperatur-Argumente von Stoffwert-/Strahlungsfunktionen sind absolut
    for side in (left, right):
        for node in ast.walk(side):
            fname = _func_name(node)
            if fname is None:
                continue
            fl = fname.lower()
            values = []
            if fl in _THERMO_FUNCS or fl == 'humidair':
                values = [kw.value for kw in node.keywords if kw.arg and kw.arg.lower() == 't']
            elif fl in _RADIATION_ARGS and node.args:
                values = [node.args[0]]
            for v in values:
                _rev_weight(v, 1.0, 1, out, known, free)
    # Prio 2: Gleichung als Summe links - rechts = 0
    terms = _flatten_additive(left, 1) + _flatten_additive(right, -1)
    _rev_weight_chain(terms, 0.0, 2, out, known, free)


def _resolve_temperature_weights(parsed_eqs, all_dims: Dict[str, DimensionInfo],
                                 user_known: Dict[str, DimensionInfo],
                                 max_iterations: int = 50) -> Dict[str, Optional[float]]:
    """
    Bestimmt für alle Temperatur-Variablen, ob absolut (1) oder Differenz (0).

    Vorgaben: bekannte Einheiten ('delta_K' vs. 'K') und Namenskonvention
    (dT..., delta...). Alle übrigen werden per Jacobi-Iteration aus den
    Gleichungen abgeleitet (reihenfolgeunabhängig); unbestimmt -> absolut.
    """
    fixed: Dict[str, float] = {}
    free: List[str] = []
    for v, d in all_dims.items():
        if not _is_temp_q(d.quantity):
            continue
        if v in user_known and user_known[v].weight is not None:
            fixed[v] = user_known[v].weight
        elif is_temperature_difference_variable(v):
            fixed[v] = 0.0
        else:
            free.append(v)
    free_set = set(free)
    current: Dict[str, Optional[float]] = {v: None for v in free}

    for _ in range(max_iterations):
        known = {}
        for v, d in all_dims.items():
            if v in fixed:
                known[v] = replace(d, weight=fixed[v])
            elif v in free_set:
                known[v] = replace(d, weight=current[v])
            else:
                known[v] = d
        cands = defaultdict(list)
        for parsed in parsed_eqs:
            try:
                _weight_candidates(parsed, known, free_set, cands)
            except Exception:
                continue
        new = dict(current)
        for v in free:
            lst = cands.get(v)
            if not lst:
                continue
            best = min(p for p, _ in lst)
            counts = defaultdict(int)
            for p, w in lst:
                if p == best:
                    counts[round(w, 9)] += 1
            new[v] = sorted(counts, key=lambda x: (-counts[x], x != 1.0, x != 0.0, x))[0]
        if new == current:
            break
        current = new

    return {**fixed, **current}


def _propagate(parsed_eqs, known_units: Dict[str, str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Kern der Propagation. Liefert (bekannte Einheiten, abgeleitete Einheiten)."""
    user = {v: _dim_info_from_unit(u, v) for v, u in known_units.items() if u is not None}
    inferred = _infer_dimensions(parsed_eqs, user)
    all_dims = {**user, **inferred}
    weights = _resolve_temperature_weights(parsed_eqs, all_dims, user)
    result = {}
    for v, d in inferred.items():
        label = d.unit if d.unit is not None else ''
        if _is_temp_q(d.quantity):
            label = 'delta_K' if weights.get(v) == 0 else 'K'
        elif _is_dimless_q(d.quantity):
            label = ''
        result[v] = label
    return known_units, result


# ============================================================================
# Öffentliche Hilfs-API (abwärtskompatibel)
# ============================================================================

class DimensionInferrer(ast.NodeVisitor):
    """
    Bestimmt die Dimension eines Ausdrucks (Wrapper um die AST-Dimensions-Engine).

    Regeln:
    - Addition/Subtraktion: Alle Operanden haben gleiche Dimension, Zahlenliterale
      sind neutral
    - Multiplikation/Division/Potenz: Dimensionen werden verrechnet, symbolische
      Exponenten nur bei dimensionsloser Basis
    - Funktionen: sin, cos, exp, ln ... -> dimensionslos; sqrt -> halbe Dimension;
      abs/max/min, Zweige von IF -> Dimension der Argumente; CoolProp/HumidAir/Strahlung -> SI
    - 'e' ist eine normale Variable (keine Euler-Konstante), 'pi' eine Zahl
    """

    def __init__(self, known_dimensions: Dict[str, DimensionInfo]):
        self.known = known_dimensions
        self.inferred: Dict[str, str] = {}

    def infer_from_expression(self, expr_str: str) -> Tuple[DimensionInfo, Dict[str, str]]:
        try:
            tree = ast.parse(expr_str.replace('^', '**'), mode='eval')
            return self.visit(tree.body), self.inferred
        except Exception:
            return DimensionInfo(None), {}

    def visit(self, node) -> DimensionInfo:
        try:
            return _eval(node, _Ctx(self.known))
        except Exception:
            return DimensionInfo(None)

    def generic_visit(self, node):
        return DimensionInfo(None)


def _remove_comments(equation: str) -> str:
    """Entfernt Kommentare aus einer Gleichung.

    Kommentare sind in "..." oder {...} eingeschlossen.
    """
    # Entferne "..." Kommentare
    result = re.sub(r'"[^"]*"', '', equation)
    # Entferne {...} Kommentare
    result = re.sub(r'\{[^}]*\}', '', result)
    return result.strip()



def _with_temperature_weights(known_dims: Dict[str, DimensionInfo]) -> Dict[str, DimensionInfo]:
    """Ergänzt fehlende Temperatur-Charaktere (für DimensionInfos ohne weight)."""
    result = {}
    for var, d in known_dims.items():
        if d is not None and d.weight is None and _is_temp_q(d.quantity):
            u = (d.unit or '').lower()
            w = 0.0 if ('delta' in u or is_temperature_difference_variable(var)) else 1.0
            d = replace(d, weight=w)
        result[var] = d
    return result


def _is_temperature_difference_expr(node, known_dims: Dict[str, DimensionInfo]) -> bool:
    """
    Erkennt Ausdrücke, die eine Temperatur-DIFFERENZ darstellen (delta_K):
    T1 - T2, (T1 - T2)/x, x*(T1 - T2) mit dimensionslosem x, dT_1 + dT_2 ...
    T_abs ± delta (z.B. T_1 - dT) und Mittelwerte (T1 + T2)/2 sind dagegen
    absolute Temperaturen.
    """
    if not PINT_AVAILABLE:
        return False
    try:
        d = _eval(node, _Ctx(_with_temperature_weights(known_dims)))
    except Exception:
        return False
    return _is_temp_q(d.quantity) and d.weight == 0


def analyze_equation(equation: str, known_units: Dict[str, str]) -> Dict[str, str]:
    """
    Analysiert eine einzelne Gleichung und leitet neue Einheiten ab.

    Args:
        equation: Gleichung der Form "left = right" oder "(left) - (right)"
        known_units: Dict von Variablen zu ihren bekannten Einheiten

    Returns:
        Dict von neu abgeleiteten Variablen und ihren Einheiten (Anzeige-Labels)
    """
    if not PINT_AVAILABLE:
        return {}
    parsed = _parse_equation(equation)
    if parsed is None:
        return {}
    try:
        _, inferred = _propagate([parsed], known_units)
    except Exception:
        return {}
    return {v: u for v, u in inferred.items() if v not in known_units}


def _collect_additive_terms(node) -> list:
    """
    Sammelt alle Terme einer Addition/Subtraktion-Kette.

    Beispiel: A + B - C + D wird zu [A, B, C, D]

    Traversiert den AST rekursiv und sammelt alle Operanden von + und -.
    Numerische Konstanten (0, 1, etc.) werden ignoriert, da sie dimensional neutral sind.
    """
    terms = []

    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        # Rekursiv linke und rechte Seite sammeln
        terms.extend(_collect_additive_terms(node.left))
        terms.extend(_collect_additive_terms(node.right))
    elif isinstance(node, ast.Constant):
        # Numerische Konstanten ignorieren - sie sind dimensional neutral
        # 0 kann jede Dimension haben (0 kg/s = 0 = 0 J/kg)
        pass
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        # Unäres Minus: -X → sammle X
        terms.extend(_collect_additive_terms(node.operand))
    else:
        # Anderer Ausdruck (Variable, Multiplikation, Division, Funktionsaufruf, etc.)
        terms.append(node)

    return terms


def _infer_from_additive_chain(node, known_dims: Dict[str, DimensionInfo]) -> Dict[str, str]:
    """
    Inferiert Einheiten aus einer Addition/Subtraktion-Kette.

    Bei Addition/Subtraktion haben ALLE Terme die gleiche Dimension.
    Beispiel: A + B - C = 0
      → Wenn A = kg/s bekannt, dann B = kg/s und C = kg/s

    Diese Funktion:
    1. Sammelt alle Terme der Kette (ignoriert numerische Konstanten)
    2. Berechnet die Dimension bekannter Terme
    3. Propagiert diese Dimension zu unbekannten Termen
    """
    if not PINT_AVAILABLE:
        return {}

    inferred = {}

    # Sammle alle Terme der Addition/Subtraktion-Kette
    terms = _collect_additive_terms(node)

    if not terms:
        return inferred

    # Finde die Dimension aus bekannten Termen
    known_dimension = None
    known_quantity = None

    for term in terms:
        term_dim = _get_dimension(term, known_dims)
        if term_dim.is_known and term_dim.quantity is not None:
            known_dimension = term_dim.unit
            known_quantity = term_dim.quantity
            break
        elif term_dim.is_known and term_dim.unit is not None:
            # Bekannt aber ohne quantity (z.B. dimensionslos "")
            known_dimension = term_dim.unit
            break

    # Wenn keine Dimension bekannt, versuche sie zu berechnen
    if known_dimension is None:
        for term in terms:
            try:
                # Versuche die Dimension des Terms zu berechnen
                term_str = ast.unparse(term) if hasattr(ast, 'unparse') else str(term)
                # Konvertiere known_dims zu known_units für compute_expression_dimension
                known_units = {var: dim.unit for var, dim in known_dims.items() if dim.is_known}
                dim_result, missing = compute_expression_dimension(term_str, known_units)
                if dim_result is not None and not missing:
                    known_dimension = unit_from_dimensionality(dim_result)
                    if known_dimension is not None:
                        known_quantity = get_dimension_from_unit(known_dimension)
                        break
            except:
                pass

    # Wenn immer noch keine Dimension bekannt, können wir nichts ableiten
    if known_dimension is None:
        return inferred

    # Propagiere die Dimension zu allen unbekannten Termen
    target_dim = DimensionInfo(known_dimension, known_quantity)

    for term in terms:
        term_dim = _get_dimension(term, known_dims)

        if term_dim.is_known:
            # Term hat bereits bekannte Dimension, überspringen
            continue

        # Versuche Variablen aus dem Term abzuleiten
        if isinstance(term, ast.Name):
            # Einfache Variable
            var_name = term.id
            if var_name not in known_dims:
                inferred[var_name] = known_dimension
        else:
            # Komplexer Term (z.B. m_dot*h, dT/ln(...))
            # Verwende _infer_from_mult_div für Multiplikation/Division
            if isinstance(term, ast.BinOp):
                sub_inferred = _infer_from_mult_div(term, target_dim, known_dims)
                inferred.update(sub_inferred)

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
                if var_name and unit is not None:  # Auch leere Einheit (dimensionslos) akzeptieren
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
                if var_name and unit is not None:  # Auch leere Einheit (dimensionslos) akzeptieren
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
                if var_name and unit is not None:  # Auch leere Einheit (dimensionslos) akzeptieren
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
                # Auch leere Einheit (dimensionslos) ist gültig
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

                        if exp_value is not None and isinstance(exp_value, (int, float)) and exp_value != 0 and target_dim.quantity is not None:
                            try:
                                # target = left * var^exp → var^exp = target / left
                                pow_quantity = target_dim.quantity / left_dim.quantity
                                var_quantity = pow_quantity ** (1.0 / exp_value)
                                unit = unit_from_quantity(var_quantity)
                                # '' = dimensionslos ist eine gültige Inferenz
                                if unit is not None:
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

                        if exp_value is not None and isinstance(exp_value, (int, float)) and exp_value != 0 and target_dim.quantity is not None:
                            try:
                                pow_quantity = target_dim.quantity / right_dim.quantity
                                var_quantity = pow_quantity ** (1.0 / exp_value)
                                unit = unit_from_quantity(var_quantity)
                                # '' = dimensionslos ist eine gültige Inferenz
                                if unit is not None:
                                    inferred[var_name] = unit
                            except:
                                pass

    return inferred



def _get_dimension(node, known_dims: Dict[str, DimensionInfo]) -> DimensionInfo:
    """Berechnet die Dimension eines AST-Knotens."""
    try:
        return _eval(node, _Ctx(known_dims))
    except Exception:
        return DimensionInfo(None)


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

    for _ in range(max_iterations):
        found_new = False

        for _parsed_eq, original_eq in equations.items():
            # Analysiere Original-Gleichung (lesbarer)
            newly_inferred = analyze_equation(original_eq, all_units)

            for var, unit in newly_inferred.items():
                if var not in all_units and unit is not None:
                    all_units[var] = unit
                    found_new = True

        if not found_new:
            break

    # Gib nur neue Einheiten zurück (nicht die ursprünglich bekannten)
    # Post-Processing: Passe Einheiten basierend auf Variablennamen an
    result = {}
    for var, unit in all_units.items():
        if var not in known_units:
            result[var] = adjust_unit_for_variable(unit, var) if unit else unit
    return result


# ============================================================================
# Function Argument Analysis for Bidirectional Unit Inference
# ============================================================================

# Einheiten für Funktionsargumente (SI-Einheiten)
FUNCTION_ARGUMENT_UNITS = {
    # CoolProp Thermodynamik-Argumente
    'T': 'K',           # Temperatur
    'p': 'Pa',          # Druck
    'h': 'J/kg',        # Enthalpie
    's': 'J/(kg*K)',    # Entropie
    'x': '',            # Dampfqualität (dimensionslos)
    'rho': 'kg/m^3',    # Dichte
    'd': 'kg/m^3',      # Dichte (Alias)
    'v': 'm^3/kg',      # Spezifisches Volumen
    'u': 'J/kg',        # Innere Energie

    # HumidAir Argumente
    't': 'K',           # Temperatur (HumidAir verwendet lowercase)
    'p_tot': 'Pa',      # Gesamtdruck
    'rh': '',           # Relative Feuchte (dimensionslos)
    'rf': '',           # Relative Feuchte (German: rF = relative Feuchte, dimensionslos)
    'w': '',            # Feuchtebeladung (kg/kg, oft als dimensionslos behandelt)
    'p_w': 'Pa',        # Partialdruck Wasserdampf

    # Strahlungsfunktionen - Argumente
    # Eb(T, wavelength), Blackbody(T, lambda1, lambda2), Wien(T), Stefan_Boltzmann(T)
    'wavelength': 'µm',     # Wellenlänge
    'lambda': 'µm',         # Wellenlänge (Alternative)
    'lambda1': 'µm',        # Untere Wellenlänge
    'lambda2': 'µm',        # Obere Wellenlänge
}



def infer_units_from_function_arguments(equation: str, known_units: Dict[str, str]) -> Dict[str, str]:
    """
    Leitet Einheiten aus Funktionsargumenten ab (bidirektional).

    Bei einem Aufruf wie `h = enthalpy(water, T=T_1, p=p_2)`:
    - T_1 muss Einheit K haben (weil T-Argument)
    - p_2 muss Einheit Pa haben (weil p-Argument)

    Funktioniert auch, wenn die linke Seite keine einzelne Variable ist
    (z.B. `Q/m = enthalpy(water, T=T_2, p=p) - h_1`).

    Args:
        equation: Gleichung die Funktionsaufrufe enthalten kann
        known_units: Dict bereits bekannter Einheiten

    Returns:
        Dict von neu abgeleiteten {variable: unit}
    """
    inferred = {}

    parsed = _parse_equation(equation)
    if parsed is not None:
        sides = list(parsed)
    else:
        try:
            sides = [ast.parse(_remove_comments(equation).replace('^', '**'), mode='eval').body]
        except Exception:
            return inferred

    thermo_funcs = {'enthalpy', 'entropy', 'pressure', 'temperature',
                    'density', 'volume', 'quality', 'intenergy',
                    'cp', 'cv', 'viscosity', 'conductivity', 'soundspeed', 'prandtl'}
    humid_funcs = {'humidair'}

    for side in sides:
        for node in ast.walk(side):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
                continue
            func_name = node.func.id.lower()

            if func_name in thermo_funcs or func_name in humid_funcs:
                for keyword in node.keywords:
                    arg_name = keyword.arg
                    if arg_name is None:
                        continue
                    expected_unit = FUNCTION_ARGUMENT_UNITS.get(arg_name.lower())
                    if expected_unit is None:
                        continue
                    if isinstance(keyword.value, ast.Name):
                        var_name = keyword.value.id
                        if var_name not in known_units and var_name not in inferred:
                            # Auch leere Einheiten (dimensionslos) - wichtig für rh, rf, x
                            inferred[var_name] = expected_unit

            elif func_name in _RADIATION_ARGS:
                # Positionsargumente: (T, Wellenlänge, ...)
                for arg, kind in zip(node.args, _RADIATION_ARGS[func_name]):
                    if isinstance(arg, ast.Name):
                        var_name = arg.id
                        if var_name not in known_units and var_name not in inferred:
                            inferred[var_name] = 'K' if kind == 'T' else 'um'

    return inferred


def propagate_all_units_complete(equations: Dict[str, str], known_units: Dict[str, str],
                                  max_iterations: int = 15) -> Dict[str, str]:
    """
    Vollständige Einheiten-Propagation mit allen Quellen.

    Kombiniert:
    1. Funktionsrückgabewerte (enthalpy → J/kg, HumidAir(T_dp, ...) → K, Eb → W/(m^2*um))
    2. Funktionsargumente bidirektional (T=T_1 → T_1: K)
    3. Arithmetische Constraint-Propagation (a + b = c → alle gleiche Einheit,
       Produkte/Quotienten/Potenzen rückwärts, sqrt, abs/max/min)
    4. Temperatur-Charakter: Differenzen → 'delta_K', absolute Temperaturen → 'K'

    Das Ergebnis ist unabhängig von der Reihenfolge der Gleichungen. Zahlenliterale
    in Summen sind neutral (T_out = T_s - 2 → beide K) und gelten nur dann als
    Hinweis auf 'dimensionslos', wenn sonst nichts bekannt ist.

    Args:
        equations: Dict von {parsed_equation: original_equation}
        known_units: Dict von {variable: unit} bekannter Einheiten
        max_iterations: (nur aus Kompatibilitätsgründen; die Iteration läuft bis
            zum Fixpunkt)

    Returns:
        Dict von ALLEN Einheiten (bekannte + abgeleitete). Einheiten sind
        Anzeige-Labels (z.B. 'kW', 'kJ/kg', 'bar', 'K', 'delta_K', 'W/(m^2*K)', '').
    """
    if not PINT_AVAILABLE:
        return known_units.copy()

    parsed_eqs = []
    seen = set()
    for parsed_key, original_eq in equations.items():
        source = original_eq if original_eq else parsed_key
        if source in seen:
            continue
        seen.add(source)
        parsed = _parse_equation(source)
        if parsed is None and parsed_key and parsed_key != source:
            parsed = _parse_equation(parsed_key)
        if parsed is not None:
            parsed_eqs.append(parsed)

    try:
        _, inferred = _propagate(parsed_eqs, known_units)
    except Exception:
        inferred = {}

    # Post-Processing: Temperaturdifferenzen (dT..., delta...) → delta_K statt K
    adjusted_units = {}
    for var, unit in known_units.items():
        adjusted_units[var] = adjust_unit_for_variable(unit, var) if unit else unit
    for var, unit in inferred.items():
        if var not in adjusted_units:
            adjusted_units[var] = adjust_unit_for_variable(unit, var) if unit else unit

    return adjusted_units


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

    elif isinstance(node, ast.Constant):
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
            if left and isinstance(node.right, ast.Constant) and isinstance(node.right.value, (int, float)):
                exp = node.right.value
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

                # WICHTIG: Prüfe ob der Faktor ein bekannter SI-Präfix-Faktor ist
                # Da intern alle Berechnungen in SI erfolgen, sind Faktoren wie
                # 1000 (kJ vs J, kW vs W, kPa vs Pa) oder 1e5 (bar vs Pa)
                # KEINE echten Fehler, sondern nur Anzeige-Unterschiede
                si_prefix_factors = {1000, 1e6, 1e9, 1e-3, 1e-6, 1e-9, 1e5, 1e-5}
                is_prefix_factor = any(abs(display_factor - f) < 0.01 or abs(display_factor - 1/f) < 0.01
                                       for f in si_prefix_factors if f != 0)

                # Prüfe ob beide Einheiten die gleiche physikalische Größe repräsentieren
                # (gleiche Dimensionalität = gleiche physikalische Größe)
                same_dimension = reference_qty.dimensionality == other_qty.dimensionality

                # Wenn gleiche Dimension UND bekannter Präfix-Faktor → kein echter Fehler
                if same_dimension and is_prefix_factor:
                    continue  # Überspringe - das ist nur ein Anzeige-Unterschied

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
    Prüft die Einheiten-Konsistenz für alle Gleichungen mittels dimensionaler Analyse.

    Neuer Ansatz: Statt String-Vergleich wird mit pint geprüft, ob jede Gleichung
    dimensional konsistent ist (Dimension links == Dimension rechts).

    WICHTIG: Vor der Prüfung werden Einheiten durch das Gleichungssystem propagiert,
    damit Variablen deren Einheiten aus anderen Gleichungen abgeleitet werden können
    (z.B. m_dot aus Q_dot = m_dot * c_p * dT) keine Warnung erzeugen.

    Args:
        solution: Dict von {variable: value} der Lösung
        equations: Dict von {parsed_equation: original_equation}
        known_units: Dict von {variable: unit} bekannter Einheiten

    Returns:
        Liste von UnitWarning für dimensionale Inkonsistenzen
    """
    from solver import UnitWarning

    if not PINT_AVAILABLE:
        return []

    # WICHTIG: Propagiere Einheiten durch alle Gleichungen BEVOR wir prüfen
    # Damit können Variablen wie m_dot oder Q_dot ihre Einheiten aus dem
    # Gleichungssystem ableiten, auch wenn sie nicht explizit definiert wurden
    all_units = propagate_all_units_complete(equations, known_units)

    # Komplett einheitenloses System: Wenn NIRGENDS eine echte (nicht-leere,
    # nicht-dimensionslose) Einheit bekannt ist, gibt es dimensional nichts
    # zu prüfen - Warnungen wie "Einheit unbekannt: x" wären reine Fehlalarme
    def _is_real_unit(u):
        return bool(u) and u not in ('dimensionless', '-', '???')

    if (not any(_is_real_unit(u) for u in all_units.values()) and
            not any(_is_real_unit(u) for u in known_units.values())):
        return []

    warnings = []

    for parsed_eq, original_eq in equations.items():
        error = check_equation_dimensions(original_eq, all_units)

        if error:
            if error['type'] == 'missing_units':
                # Sammle fehlende Variablen - nur warnen wenn es viele sind
                # Einzelne fehlende Variablen sind oft OK (dimensionslose Konstanten)
                missing = error['variables']
                if len(missing) > 0:
                    # Format: {equation: "fehlende Variablen: x, y, z"}
                    missing_info = f"Einheit unbekannt: {', '.join(sorted(missing))}"
                    warnings.append(UnitWarning(
                        variable='Unbekannte Einheiten',
                        equations=[original_eq],
                        units={original_eq: missing_info},
                        explanation=f"Einheit unbekannt für: {', '.join(sorted(missing))}",
                        conversion_factor=0
                    ))
            elif error['type'] == 'dimension_mismatch':
                # Format: {equation: "links: W/m^2 ≠ rechts: W"} - lesbare SI-Labels
                # statt pint-Rohtext ([mass] / [time] ** 3)
                left_unit = error.get('left_unit') or error['left_dim']
                right_unit = error.get('right_unit') or error['right_dim']
                dim_info = f"links: {left_unit} ≠ rechts: {right_unit}"
                # Variable = linke Seite, wenn dort ein reiner Variablenname steht
                lhs = _remove_comments(original_eq).split('=', 1)[0].strip()
                variable = lhs if re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', lhs) else 'Dimensionsfehler'
                warnings.append(UnitWarning(
                    variable=variable,
                    equations=[original_eq],
                    units={original_eq: dim_info},
                    explanation=f"Dimensionsfehler: {left_unit} ≠ {right_unit}",
                    conversion_factor=0
                ))

    return warnings




# ============================================================================
# Generische dimensionale Konsistenzprüfung mit pint
# ============================================================================

# Namen, die (wenn sie keine bekannte Variable sind) keine Warnung "Einheit
# unbekannt" auslösen sollen (Stoffnamen, Einheiten-Kürzel)
_CHECK_SKIP_TOKENS = {
    'water', 'steam', 'air', 'Water', 'Air',
    'R134a', 'R1234yf', 'CO2', 'Ammonia', 'Nitrogen',
    'C', 'K', 'F',
    'kg', 'g', 'm', 's', 'Pa', 'bar', 'J', 'W', 'kJ', 'kW',
}

_DIMENSION_ERROR = 'DIMENSION_ERROR_IN_EXPR'


def _expression_dimension(expr: str, unit_map: Dict[str, str]):
    """
    Dimension eines Ausdrucks für die Konsistenzprüfung.

    Returns:
        (dimensionality | 'DIMENSION_ERROR_IN_EXPR' | None, missing_vars, is_literal)
    """
    try:
        node = ast.parse(_remove_comments(expr).replace('^', '**'), mode='eval').body
    except Exception:
        return None, [], False

    names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
    known = {v: _dim_info_from_unit(unit_map[v], v)
             for v in names if v in unit_map and unit_map[v] is not None}
    skip = _CHECK_SKIP_TOKENS - set(known)

    # 1. Fehlende Variablen sammeln (ohne Abbruch bei Dimensionsfehlern)
    ctx = _Ctx(known, strict=False, skip_names=skip)
    try:
        _eval(node, ctx)
    except Exception:
        return None, [], False
    if ctx.missing:
        return None, sorted(ctx.missing), False

    # 2. Strenge Auswertung: inkompatible Summanden erkennen
    ctx = _Ctx(known, strict=True, skip_names=skip)
    try:
        d = _eval(node, ctx)
    except _DimMismatch:
        return _DIMENSION_ERROR, [], False
    except Exception:
        return None, [], False

    if d.quantity is None:
        return None, [], False
    return d.quantity.dimensionality, [], bool(d.literal)


def compute_expression_dimension(expr: str, unit_map: Dict[str, str]) -> Tuple[Any, list]:
    """
    Berechnet die Dimension eines mathematischen Ausdrucks.

    - Zahlenliterale in Summen sind dimensionsneutral (T_s - 2 hat die Dimension von T_s)
    - 'e' ist eine normale Variable, 'pi' eine Zahl
    - sqrt halbiert die Dimension, abs/max/min behalten sie
    - symbolische Exponenten nur bei dimensionsloser Basis (sonst unbestimmt)
    - Einheiten werden über normalize_unit gelesen ('W/m^2K' = W/(m²·K))

    Args:
        expr: Mathematischer Ausdruck als String, z.B. "m_zu * h_zu"
        unit_map: Dict {variable: unit_string} für alle bekannten Variablen

    Returns:
        (dimensionality, missing_vars) - pint Dimensionality und Liste fehlender Variablen
        Bei inkompatiblen Summanden: ('DIMENSION_ERROR_IN_EXPR', [])
        Bei nicht bestimmbarer Dimension: (None, missing_vars)
    """
    if not PINT_AVAILABLE:
        return None, []
    dim, missing, _ = _expression_dimension(expr, unit_map)
    return dim, missing


def check_equation_dimensions(equation: str, unit_map: Dict[str, str]) -> Optional[Dict]:
    """
    Prüft ob eine Gleichung dimensional konsistent ist.

    GENERISCHER ANSATZ: Bei "var = ausdruck" wird die Einheit von var
    aus dem Ausdruck ABGELEITET, nicht als "fehlend" gemeldet.
    Eine Seite, die nur aus Zahlen besteht (z.B. "... = 0" oder "= 0.0"),
    ist dimensionsneutral.

    Args:
        equation: Gleichung als String, z.B. "W_v + m_zu*h_zu = U_2-U_1"
        unit_map: Dict {variable: unit_string} für alle Variablen

    Returns:
        None wenn konsistent, sonst Dict mit Fehlerinfo:
        - {'type': 'missing_units', 'variables': [...], 'equation': ...}
        - {'type': 'dimension_mismatch', 'left_dim': ..., 'right_dim': ..., 'equation': ...}
    """
    if not PINT_AVAILABLE:
        return None

    sides = _split_equation(_remove_comments(equation))
    if sides is None:
        return None
    left, right = sides

    # SCHRITT 1: Dimension der RECHTEN Seite
    right_dim, right_missing, right_literal = _expression_dimension(right, unit_map)
    if right_missing:
        return {
            'type': 'missing_units',
            'variables': right_missing,
            'equation': equation
        }
    if right_dim == _DIMENSION_ERROR:
        return {
            'type': 'dimension_mismatch',
            'left_dim': '(linke Seite)',
            'right_dim': 'Inkompatible Terme werden addiert/subtrahiert',
            'equation': equation
        }

    dimensionless_dim = ureg.dimensionless.dimensionality

    # SCHRITT 2: Linke Seite eine einzelne Variable?
    if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', left):
        # FALL A: Variable ohne Einheit → erbt die Dimension der rechten Seite
        if left not in unit_map:
            return None
        # FALL B: Rechte Seite nur Zahl/dimensionslos → Konstantenzuweisung
        if right_literal or right_dim == dimensionless_dim:
            return None

    # SCHRITT 3: Linke Seite
    left_dim, left_missing, left_literal = _expression_dimension(left, unit_map)
    if left_missing:
        return {
            'type': 'missing_units',
            'variables': left_missing,
            'equation': equation
        }
    if left_dim == _DIMENSION_ERROR:
        return {
            'type': 'dimension_mismatch',
            'left_dim': 'Inkompatible Terme werden addiert/subtrahiert',
            'right_dim': '(rechte Seite)',
            'equation': equation
        }

    if left_dim is None or right_dim is None:
        return None

    # Eine Seite nur aus Zahlen (z.B. "= 0", "= 0.0") ist dimensionsneutral
    if left_literal or right_literal:
        return None

    if left_dim != right_dim:
        return {
            'type': 'dimension_mismatch',
            'left_dim': str(left_dim),
            'right_dim': str(right_dim),
            'left_unit': si_label_from_dimensionality(left_dim),
            'right_unit': si_label_from_dimensionality(right_dim),
            'equation': equation
        }

    return None  # OK - dimensional konsistent


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
