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
import numpy as np
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
    ('N/m', 'N/m'),                 # = J/m^2; Energie je Fläche aus Leistung mal Zeit: kWh/m^2
    ('K^4', 'K^4'),
    ('m^2*K', 'm^2*K'),
    ('K*s', 'K*s'),
    # molare Größen (Verbrennung, Brennstoffzelle; IG:-Stoffe) und Elektrochemie
    ('J/mol', 'kJ/mol'),
    ('J/(mol*K)', 'J/(mol*K)'),
    ('kg/mol', 'kg/kmol'),
    ('mol', 'mol'),
    ('mol/s', 'mol/s'),
    ('V', 'V'),
    ('A', 'A'),
    ('A/m^2', 'A/m^2'),
    ('C/mol', 'C/mol'),
    ('ohm', 'ohm'),
    ('ohm*m^2', 'ohm*m^2'),          # flächenbezogener Widerstand (ASR)
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
    'gibbs': 'J/kg',
    'molarmass': 'kg/kmol',      # Anzeige-Label; der Wert ist SI (kg/mol) wie überall
}

# HumidAir: Ausgabe-Label abhängig vom ersten Argument
_HUMID_OUT = {
    'h': 'J/kg',
    'w': 'kg/kg',     # dimensionslos; Label kg/kg -> Startwert 0.01 (nicht 0.5, weit übersättigt)
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
    'v': 'm^3/kg',
    'x': 'kg/kg',
    'phi': '',
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

    Temperatur-Charakter aus der Einheit (nie aus dem Namen): 'delta...' -> Differenz,
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
        # Charakter aus der Einheit, nie aus dem Namen: delta_K = Differenz, sonst absolut
        # (eine in K eingegebene Größe kann offen gelassen werden, siehe open_temperatures)
        weight = 0.0 if any(m in u.lower() for m in _DELTA_MARKERS) else 1.0
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
    # Alle Terme mit derselben Einheit (m3/h + m3/h): Ergebnis behält sie (nicht Temperaturen)
    labels = {d.unit for _, d in known}
    same = labels.pop() if len(labels) == 1 and not _is_temp_q(ref.quantity) else None
    d = _dim(ref.quantity, weight, same or None)
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


def _scaled_label(l: DimensionInfo, r: DimensionInfo, q, divide: bool = False) -> Optional[str]:
    """
    Einheit eines Produkts mit einer dimensionslosen Größe (bzw. Quotient durch eine): die des
    anderen Faktors (V_dot = n_P*V_dot_P mit V_dot_P in m^3/h -> m^3/h statt m^3/s). Sonst
    None (Label aus der Dimension). Temperaturen: Label aus dem Temperatur-Charakter.
    """
    if q is None or _is_temp_q(q):
        return None
    if _is_dimless_q(q):
        # dimensionslos: nur mit einer reinen Zahl (eta*0.5 bleibt %, eta*eta ist eine Zahl)
        if r.literal and l.unit:
            return l.unit
        if not divide and l.literal and r.unit:
            return r.unit
        return None
    if _is_dimless_q(r.quantity) and l.unit:
        return l.unit
    if not divide and _is_dimless_q(l.quantity) and r.unit:
        return r.unit
    return None


# Dimensionen, deren Anzeige die Settings bestimmen (Leistung, Energie, Druck) bzw. der
# Temperatur-Charakter - dort kein zusammengesetztes Label
_SETTINGS_DIMS = ('W', 'J', 'Pa')


def _composed_label(l: DimensionInfo, r: DimensionInfo, q, divide: bool = False) -> Optional[str]:
    """
    Einheit eines Produkts/Quotienten zweier Größen mit Einheit aus den Einheiten der Faktoren
    (gleiche Einheiten kürzen sich, wie in pint):
    - eine einzige Einheit bleibt: n = V_dot/V mit m^3/h und m^3 -> 1/h, t = V/V_dot -> h,
      d = A/L mit mm^2 und mm -> mm (sonst SI aus der Dimension: 1/s, s, m)
    - Leistung mal Zeit ist Energie: w = q*t mit W/m und h -> kWh/m (gleiche Dimension wie N),
      ebenso je Fläche und je Volumen (kWh/m^2 statt N/m, kWh/m^3 statt bar) - die Größenart
      folgt aus den Faktoren, nicht aus der Dimension
    Nicht für Temperaturen und die Settings-Größen (Leistung, Energie, Druck).
    """
    if (q is None or _is_temp_q(q) or _is_dimless_q(q) or not l.unit or not r.unit
            or l.quantity is None or r.quantity is None
            or _is_dimless_q(l.quantity) or _is_dimless_q(r.quantity)):
        return None
    try:
        if any(q.dimensionality == ureg.Quantity(1.0, u).dimensionality for u in _SETTINGS_DIMS):
            return None
        a, b = ureg.Quantity(1.0, l.unit), ureg.Quantity(1.0, r.unit)
        items = dict(((a / b) if divide else (a * b))._units)
        dims = {name: ureg.Quantity(1.0, name).dimensionality for name in items}
        if len(items) == 1:
            (name, exp), = items.items()
            if exp in (1, -1) and not _is_temp_q(ureg.Quantity(1.0, name)):
                symbol = ureg.get_symbol(name)
                return symbol if exp == 1 else f"1/{symbol}"
            return None
        power = ureg.Quantity(1.0, 'W').dimensionality
        time = ureg.Quantity(1.0, 's').dimensionality
        if (any(exp > 0 and dims[n] == power for n, exp in items.items())
                and any(exp > 0 and dims[n] == time for n, exp in items.items())):
            for k, label in ((1, 'kWh/m'), (2, 'kWh/m^2'), (3, 'kWh/m^3')):
                if q.dimensionality == ureg.Quantity(1.0, f'J/m^{k}').dimensionality:
                    return label
    except Exception:
        return None
    return None


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
    d = _dim(q, weight, _scaled_label(l, r, q) or _composed_label(l, r, q))
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
    d = _dim(q, weight, _scaled_label(l, r, q, divide=True) or _composed_label(l, r, q, divide=True))
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


def _unit_function_unit(node) -> Optional[Tuple[str, DimensionInfo]]:
    """
    value(x, 'Einheit') / quantity(z, 'Einheit'): (Funktionsname, DimensionInfo der
    Einheit). Einheit mit Nullpunkt (°C, °F) -> absolute Temperatur (Gewicht 1).
    """
    fname = _func_name(node)
    if fname not in ('value', 'quantity') or len(node.args) != 2:
        return None
    unit = node.args[1]
    if not (isinstance(unit, ast.Constant) and isinstance(unit.value, str)):
        return None
    q = _q_for(unit.value)
    if q is None:
        return fname, DimensionInfo(None)
    if _is_temp_q(q):
        origin = scale_origin(unit.value)
        return fname, _dim(q, 1.0 if origin else None)
    # die genannte Einheit ist eine ausdrückliche Angabe (wie ein Startwert mit Einheit):
    # value(Q_a, kWh/m^2) -> Q_a in kWh/m^2, nicht N/m aus der Dimension
    return fname, _dim(q, None, unit.value.strip().replace('**', '^'))


def _special_call_dim(node) -> Optional[DimensionInfo]:
    """Dimension von CoolProp-/HumidAir-/Strahlungs-Aufrufen (Argumente egal)."""
    fname = _func_name(node)
    if fname is None:
        return None
    unit_call = _unit_function_unit(node)
    if unit_call is not None:
        name, unit_dim = unit_call
        return DimensionInfo('', _dimless_q()) if name == 'value' else unit_dim
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
    # gemeinsame Eingabe-Einheit bleibt (max(V_1, V_2) mit m^3/h -> m^3/h, wie V_1 + V_2)
    labels = {d.unit for d in known}
    label = labels.pop() if len(labels) == 1 and len(known) == len(nonlit) and not _is_temp_q(ref.quantity) else None
    return _dim(ref.quantity, weight, label or None)


def _eval_call(node, ctx: _Ctx) -> DimensionInfo:
    fname = _func_name(node)
    if fname is None:
        return DimensionInfo(None)

    unit_call = _unit_function_unit(node)
    if unit_call is not None and ctx.strict:
        name, unit_dim = unit_call
        arg = _eval(node.args[0], ctx)
        expected = unit_dim.quantity if name == 'value' else _dimless_q()
        if (expected is not None and arg.quantity is not None and not arg.literal
                and arg.quantity.dimensionality != expected.dimensionality):
            raise _DimMismatch()
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

    if fname in ('abs', 'ceil', 'floor', 'round'):
        if not args:
            return DimensionInfo(None)
        a = _eval(args[0], ctx)
        if a.literal and a.value is not None:
            op = {'abs': abs, 'ceil': math.ceil, 'floor': math.floor, 'round': round}[fname]
            return replace(a, value=float(op(a.value)))
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

    if fname == 'equilibrium' and len(args) >= 6:
        # EQUILIBRIUM (Gibbs-Minimierung): Austrittsmenge in der Einheit der Eintrittsmengen
        return _same_dimension_result([_eval(a, ctx) for a in args[5:]], ctx)

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


def _quote_unit_arguments(equation: str) -> str:
    """value(v, m/s) -> value(v, 'm/s'): die Einheit ist Text, keine Rechnung mit m und s."""
    if 'value' not in equation and 'quantity' not in equation:
        return equation
    try:
        from parser import _replace_calls_balanced, UNIT_VALUE_FUNCTIONS, _convert_unit_function_call
        return _replace_calls_balanced(equation, UNIT_VALUE_FUNCTIONS, keep_case=False,
                                       converter=_convert_unit_function_call)
    except Exception:
        return equation


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
        eq = _quote_unit_arguments(_remove_comments(equation))
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
        if fname in ('abs', 'ceil', 'floor', 'round') and node.args:
            _rev(node.args[0], target, known, rank, out)
        elif fname in ('max', 'min') or (fname == 'IF' and len(node.args) == 5) or (
                fname == 'equilibrium' and len(node.args) >= 6):
            branches = {'IF': node.args[2:], 'equilibrium': node.args[5:]}.get(fname, node.args)
            for a in branches:
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


def _log_term(node) -> Optional[Tuple[str, str]]:
    """
    (Faktor, x) für Terme ln(x), c*ln(x), ln(x)*c, ln(x)/c mit x einfacher Name - der Faktor
    als Syntaxbaum-Text (gleicher Faktor = gleicher Text), sonst None.
    """
    call, factor = None, '1'
    if _func_name(node) in _LOG_FUNCS:
        call = node
    elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Div)):
        if _func_name(node.left) in _LOG_FUNCS:
            call, factor = node.left, type(node.op).__name__ + ast.dump(node.right)
        elif isinstance(node.op, ast.Mult) and _func_name(node.right) in _LOG_FUNCS:
            call, factor = node.right, 'Mult' + ast.dump(node.left)
    if call is not None and call.args and isinstance(call.args[0], ast.Name):
        return factor, call.args[0].id
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

    # c*ln(a) - c*ln(b) (gleicher Faktor, entgegengesetztes Vorzeichen) = c*ln(a/b): a/b ist
    # dimensionslos, a und b sind gleichartig - ln(T_2) - ln(T_1), R*ln(p_2) - R*ln(p_1)
    logs = [(sgn, term) for sgn, term in ((sgn, _log_term(n)) for sgn, n, _, _ in evals) if term]
    for i, (s_1, (f_1, a_1)) in enumerate(logs):
        for s_2, (f_2, a_2) in logs[i + 1:]:
            if f_1 != f_2 or s_1 != -s_2:
                continue
            for target, ref in ((a_1, a_2), (a_2, a_1)):
                if target not in known and ref in known and known[ref].quantity is not None:
                    _add_cand(out, target, rank, _dim(known[ref].quantity, known[ref].weight))


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
                expected = function_argument_unit(fl, kw.arg)
                if expected is None:
                    continue
                q = _q_for(expected)
                weight = 1.0 if _is_temp_q(q) else None
                targets.append((kw.value, _dim(q, weight, expected if expected else '')))
        elif _unit_function_unit(node) is not None:
            name, unit_dim = _unit_function_unit(node)
            if unit_dim.quantity is not None:
                # value(x, 'bar'): x ist ein Druck; quantity(z, 'bar'): z ist eine Zahl
                targets.append((node.args[0], unit_dim if name == 'value' else _dim(_dimless_q())))
        elif fl == 'equilibrium' and len(node.args) >= 6:
            targets.append((node.args[3], _dim(_q_for('K'), 1.0, 'K')))
            targets.append((node.args[4], _dim(_q_for('Pa'), None, 'Pa')))
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
                elif _is_dimless_q(d.quantity) and label in ('dimensionless', '-', '1'):
                    label = ''               # %, ‰, g/kg aus den Eingaben bleiben erhalten
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
    # Innere Summen (z.B. UA*(T_s - T_amb)) - Dimension aus den eigenen Termen. Jede Summe nur
    # EINMAL: Teile einer schon ausgewerteten Kette (die rechte Seite von T_3 = T_1 + T_2) würden
    # ihre Kandidaten in der Mehrheitswahl (_select) sonst doppelt zählen
    consumed = set()

    def mark(node):
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            consumed.add(id(node))
            mark(node.left)
            mark(node.right)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            mark(node.operand)

    for side in (left, right):
        mark(side)
    for side in (left, right):
        for node in ast.walk(side):
            if (isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub))
                    and id(node) not in consumed):
                mark(node)
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
    """
    Deterministische Auswahl: häufigste Dimension, dann häufigstes Label; bei Gleichstand das
    zuerst gefundene (Reihenfolge der Gleichungen im Blatt, nicht das Alphabet).
    """
    groups = {}
    for d in cands:
        groups.setdefault(_dim_key(d.quantity.dimensionality), []).append(d)
    group = groups[max(groups, key=lambda k: len(groups[k]))]
    labels = {}
    for d in group:
        key = d.unit if d.unit is not None else ''
        labels[key] = labels.get(key, 0) + 1
    label = max(labels, key=labels.get)
    q = group[0].quantity
    if _is_temp_q(q):
        label = 'K'
    elif _is_dimless_q(q) and label in ('dimensionless', '-', '1'):
        label = ''                       # %, ‰, g/kg aus den Eingaben bleiben erhalten
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
# Vollständige Einheiten-Ableitung (Stufe 2, Hindley-Milner nach Kennedy 1996 bzw.
# Olsson 2025 "Improved Unit Inference and Checking in Modelica")
# ----------------------------------------------------------------------------
# Jede Gleichung und jeder Teilausdruck liefert eine Einheiten-Gleichung:
#   Summe/Gleichung: u[a] = u[b];  Produkt: u[x*y] = u[x]*u[y];  Quotient: u[x]/u[y];
#   x^n (n Zahl): u[x]^n;  sqrt: u[x]^(1/2);  sin/exp/ln/x^y: Argument dimensionslos.
# Zahlen sind in Produkten dimensionslos, allein in Summen beliebig (Casella 2016).
# Logarithmiert ist das ein LINEARES Gleichungssystem in den Exponenten der SI-Basis-
# einheiten - es wird für alle Gleichungen gemeinsam exakt (Brüche) eliminiert und findet
# so auch gekoppelte Einheiten, die die lokale Propagation (Stufe 1) nicht findet:
# a*b = X, a/b = Y -> a, b; v*x = w, v*v = k; w*(w*(...) - 2*D*yd) = yd.
_BASE_DIMENSIONS = ('[length]', '[mass]', '[time]', '[current]', '[temperature]',
                    '[substance]', '[luminosity]')
_BASE_UNITS = ('m', 'kg', 's', 'A', 'K', 'mol', 'cd')


class _AnyUnit:
    """Zahlenliteral in einer Summe bzw. allein: jede Einheit möglich."""


_ANY = _AnyUnit()


def _exponent_vector(quantity):
    from fractions import Fraction
    dims = quantity.dimensionality
    return [Fraction(dims.get(b, 0)).limit_denominator(1000) for b in _BASE_DIMENSIONS]


def _quantity_from_vector(vector):
    q = 1.0 * ureg.dimensionless
    for exponent, unit in zip(vector, _BASE_UNITS):
        if exponent:
            q = q * ureg.Quantity(1.0, unit) ** float(exponent)
    return _canon_quantity(q.dimensionality)


def _complete_inference(parsed_eqs, known: Dict[str, DimensionInfo],
                        report: Optional[Dict[str, Any]] = None) -> Dict[str, DimensionInfo]:
    """
    Stufe 2: Einheiten aller noch offenen Variablen aus dem gemeinsamen linearen
    Gleichungssystem der Exponenten. Gleichungen, die den bisherigen widersprechen,
    werden ausgelassen (ihre Einheitenwarnung kommt aus der normalen Prüfung).
    """
    from fractions import Fraction
    zero = [Fraction(0)] * len(_BASE_DIMENSIONS)
    rows: List[Tuple[Dict[str, Fraction], List[Fraction]]] = []
    unknown_order: Dict[str, int] = {}
    fresh = [0]

    def const(vector):
        return ({}, list(vector))

    def variable(name):
        unknown_order.setdefault(name, len(unknown_order))
        return ({name: Fraction(1)}, list(zero))

    def new_variable():
        fresh[0] += 1
        return variable(f"#{fresh[0]}")

    def combine(a, b, factor=Fraction(1)):
        coeffs = dict(a[0])
        for name, c in b[0].items():
            coeffs[name] = coeffs.get(name, Fraction(0)) + factor * c
        return ({n: c for n, c in coeffs.items() if c != 0},
                [x + factor * y for x, y in zip(a[1], b[1])])

    def scale(a, factor):
        return ({n: c * factor for n, c in a[0].items() if c * factor != 0}, [x * factor for x in a[1]])

    def equal(a, b):
        if a is _ANY or b is _ANY:
            return
        coeffs, rhs = combine(a, b, Fraction(-1))
        if coeffs:
            rows.append((coeffs, [-x for x in rhs]))

    def literal_exponent(node):
        d = _eval_collect(node, known)[0]
        if d.literal and d.value is not None:
            return Fraction(d.value).limit_denominator(1000)
        return None

    def form(node):
        if isinstance(node, ast.Constant):
            return _ANY
        if isinstance(node, ast.Name):
            if node.id == 'pi':
                return _ANY
            d = known.get(node.id)
            if d is not None and d.quantity is not None:
                return const(_exponent_vector(d.quantity))
            return variable(node.id)
        if isinstance(node, ast.UnaryOp):
            return form(node.operand)
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, (ast.Add, ast.Sub)):
                result = _ANY
                for _, term in _flatten_additive(node):
                    f = form(term)
                    if f is _ANY:
                        continue
                    if result is _ANY:
                        result = f
                    else:
                        equal(result, f)
                return result
            if isinstance(node.op, (ast.Mult, ast.Div)):
                left, right = form(node.left), form(node.right)
                if left is _ANY and right is _ANY:
                    return _ANY
                left = const(zero) if left is _ANY else left
                right = const(zero) if right is _ANY else right
                return combine(left, right, Fraction(1) if isinstance(node.op, ast.Mult) else Fraction(-1))
            if isinstance(node.op, ast.Pow):
                base = form(node.left)
                n = literal_exponent(node.right)
                if n is not None:
                    return _ANY if base is _ANY else scale(base, n)
                # Exponent ohne festen Zahlenwert (Variable): er ist dimensionslos; das
                # Ergebnis nur bei sicher dimensionsloser Basis - sonst keine Willkür
                # (n kann eine Konstante wie 0.4 sein: y^n hätte dann die Einheit m^0.4)
                equal(form(node.right), const(zero))
                if base is _ANY or (not base[0] and not any(base[1])):
                    return const(zero)
                return new_variable()
            return new_variable()
        if isinstance(node, ast.Call):
            fname = _func_name(node)
            unit_call = _unit_function_unit(node)
            if unit_call is not None:
                name, unit_dim = unit_call
                if unit_dim.quantity is None:
                    return new_variable()
                unit_form = const(_exponent_vector(unit_dim.quantity))
                if name == 'value':
                    equal(form(node.args[0]), unit_form)
                    return const(zero)
                equal(form(node.args[0]), const(zero))
                return unit_form
            special = _special_call_dim(node)
            if special is not None:
                return const(_exponent_vector(special.quantity)) if special.quantity is not None else new_variable()
            args = node.args
            if fname in _TRANSCENDENTAL or (fname or '').lower() in ('sinh', 'cosh', 'tanh', 'exp'):
                for a in args:
                    equal(form(a), const(zero))
                return const(zero)
            if fname == 'sqrt' and args:
                f = form(args[0])
                return _ANY if f is _ANY else scale(f, Fraction(1, 2))
            if fname in ('abs', 'ceil', 'floor', 'round') and args:
                return form(args[0])
            if fname == 'equilibrium' and len(args) >= 6:
                # T absolut in K, p in Pa; Austrittsmenge wie die Eintrittsmengen
                equal(form(args[3]), const(_exponent_vector(_q_for('K'))))
                equal(form(args[4]), const(_exponent_vector(_q_for('Pa'))))
                result = _ANY
                for a in args[5:]:
                    f = form(a)
                    if f is _ANY:
                        continue
                    if result is _ANY:
                        result = f
                    else:
                        equal(result, f)
                return result if result is not _ANY else new_variable()
            if fname in ('max', 'min', 'IF'):
                branches = args[2:] if fname == 'IF' else args
                if fname == 'IF' and len(args) == 5:
                    equal(form(args[0]), form(args[1]))
                result = _ANY
                for a in branches:
                    f = form(a)
                    if f is _ANY:
                        continue
                    if result is _ANY:
                        result = f
                    else:
                        equal(result, f)
                return result
            return new_variable()
        return new_variable()

    for left, right in parsed_eqs:
        try:
            equal(form(left), form(right))
        except Exception:
            continue
    if report is not None:
        report.update(order=unknown_order, pivots={})
    if not rows:
        return {}

    # Gauss-Elimination (exakt), Zeile für Zeile; widersprüchliche Zeilen auslassen
    pivots: Dict[str, Tuple[Dict[str, Fraction], List[Fraction]]] = {}
    for coeffs, rhs in rows:
        coeffs, rhs = dict(coeffs), list(rhs)
        for name in [n for n in coeffs if n in pivots]:
            factor = coeffs.get(name)
            if not factor:
                continue
            p_coeffs, p_rhs = pivots[name]
            for n2, c2 in p_coeffs.items():
                coeffs[n2] = coeffs.get(n2, Fraction(0)) - factor * c2
            rhs = [x - factor * y for x, y in zip(rhs, p_rhs)]
            coeffs = {n: c for n, c in coeffs.items() if c != 0}
        if not coeffs:
            continue          # redundant oder widersprüchlich (0 = rhs != 0): auslassen
        pivot = min(coeffs, key=lambda n: unknown_order.get(n, 0))
        factor = coeffs[pivot]
        coeffs = {n: c / factor for n, c in coeffs.items()}
        rhs = [x / factor for x in rhs]
        for name, (p_coeffs, p_rhs) in list(pivots.items()):
            f = p_coeffs.get(pivot)
            if f:
                new_coeffs = dict(p_coeffs)
                for n2, c2 in coeffs.items():
                    new_coeffs[n2] = new_coeffs.get(n2, Fraction(0)) - f * c2
                pivots[name] = ({n: c for n, c in new_coeffs.items() if c != 0},
                                [x - f * y for x, y in zip(p_rhs, rhs)])
        pivots[pivot] = (coeffs, rhs)

    if report is not None:
        report['pivots'] = pivots
    result = {}
    for name, (coeffs, rhs) in pivots.items():
        if name.startswith('#') or set(coeffs) != {name}:
            continue      # Hilfsvariable bzw. noch von freien Einheiten abhängig
        try:
            q = _quantity_from_vector(rhs)
        except Exception:
            continue
        if q is not None:
            result[name] = _dim(q)
    return result


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
        if fname in ('abs', 'ceil', 'floor', 'round') and node.args:
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


def _product_difference_candidates(side, prio: int, out, known, free: Set[str]):
    """
    Eine Temperatur in einem Produkt/Quotienten, dessen Dimension KEINE Temperatur ist,
    ist eine Temperaturdifferenz (Konvention wie COMSOL): m*c*(T1 - T2) -> (T1 - T2)
    Differenz, mit T1 absolut also T2 absolut; Q = m*c*theta -> theta Differenz;
    h*(T_s - T_inf), g*beta*(T_s - T_inf)*L^3/nu^2. Nur Kandidaten, die zu absolut (1)
    oder Differenz (0) führen (der Mittelwert (T_1 + T_2)/2 im Produkt bleibt absolut).
    """
    for node in ast.walk(side):
        if not (isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Div))):
            continue
        whole = _eval_collect(node, known)[0]
        if whole.quantity is None or _is_temp_q(whole.quantity):
            continue
        for operand in (node.left, node.right):
            d = _eval_collect(operand, known)[0]
            if d.literal or d.quantity is None or not _is_temp_q(d.quantity):
                continue
            found = defaultdict(list)
            _rev_weight(operand, 0.0, prio, found, known, free)
            for name, candidates in found.items():
                out[name].extend((p, w) for p, w in candidates if w in (0.0, 1.0))


def _power_absolute_candidates(side, prio: int, out, known, free: Set[str]):
    """
    Eine Temperatur-Variable mit ganzzahligem Exponenten >= 2 (sigma*T^4) ist eine absolute
    Temperatur (Strahlung braucht die Kelvin-Verhältnisskala). Nur für einzelne Variablen als
    Basis - gebrochene Exponenten und Summen als Basis bleiben offen ((T_s - T_inf)^(1/3) in
    Konvektions-Korrelationen ist eine Potenz einer Differenz).
    """
    for node in ast.walk(side):
        if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow)
                and isinstance(node.left, ast.Name) and node.left.id in free):
            continue
        exponent = _eval_collect(node.right, known)[0]
        if not (exponent.literal and exponent.value is not None and float(exponent.value).is_integer()
                and exponent.value >= 2):
            continue
        base = _eval_collect(node.left, known)[0]
        if base.quantity is not None and _is_temp_q(base.quantity):
            out[node.left.id].append((prio, 1.0))


def _weight_candidates(parsed, known, free: Set[str], out, products: bool = False,
                       integral: bool = True):
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
                values = [kw.value for kw in node.keywords
                          if kw.arg and kw.arg.lower() in ('t', 't_dp', 't_wb')]
            elif fl in _RADIATION_ARGS and node.args:
                values = [node.args[0]]
            elif fl == 'equilibrium' and len(node.args) >= 6:
                values = [node.args[3]]          # Gleichgewichtstemperatur: absolut
            elif fl == 'value':
                unit_call = _unit_function_unit(node)
                if unit_call is not None and unit_call[1].weight == 1.0:
                    values = [node.args[0]]      # Zahlenwert in °C/°F: absolute Temperatur
            for v in values:
                _rev_weight(v, 1.0, 1, out, known, free)
    # Prio 2: Gleichung als Summe links - rechts = 0
    terms = _flatten_additive(left, 1) + _flatten_additive(right, -1)
    _rev_weight_chain(terms, 0.0, 2, out, known, free)
    # Prio 3: mehrere offene Temperaturgrößen in der Summe
    if integral:
        _integral_weight_candidates(terms, 3, out, known, free)
    # Prio 4 (nur für die Anzeige): Temperatur im Produkt ohne Temperatur-Dimension = Differenz
    if products:
        for side in (left, right):
            _product_difference_candidates(side, 4, out, known, free)
            _power_absolute_candidates(side, 5, out, known, free)


def _integral_weight_candidates(terms, prio: int, out, known, free: Set[str]):
    """
    Σ s_i·w_i = 0 mit mehreren offenen Temperaturgrößen: jede Temperaturgröße ist
    entweder absolut (1) oder eine Differenz (0). Ist nur eine Belegung möglich, ist sie
    bestimmt (T_2 = T_1 + x mit T_1 absolut -> x Differenz, T_2 absolut); sonst bleiben
    die Größen offen (T_out = T_s - 2: beide absolut oder beide Differenz).
    """
    import itertools
    open_names, rest = [], []
    for s, n in terms:
        d = _eval_collect(n, known)[0]
        if (isinstance(n, ast.Name) and n.id in free and d.weight is None
                and d.quantity is not None and _is_temp_q(d.quantity)):
            open_names.append((s, n.id))
        else:
            rest.append((s, d))
    if not 2 <= len(open_names) <= 4:
        return
    total = _sum_weights(rest)
    if total is None:
        return
    solutions = [combo for combo in itertools.product((0.0, 1.0), repeat=len(open_names))
                 if abs(total + sum(s * w for (s, _), w in zip(open_names, combo))) < 1e-9]
    if len(solutions) == 1:
        for (_, name), w in zip(open_names, solutions[0]):
            out[name].append((prio, w))


def _zero_point_candidates(parsed, key: str, known, free: Set[str], out, zero_point, prio: float):
    """
    Nullpunkt-Test (nur für die Anzeige, mit den Werten der Lösung): eine Gleichung zwischen
    Temperaturen bleibt gültig, wenn der Nullpunkt der Temperaturskala verschoben wird -
    absolute Temperaturen verschieben sich mit, Differenzen nicht. Mit genau einer offenen
    Temperaturgröße und sonst bekanntem Charakter folgt deren Charakter numerisch:
    dT = Q/(m*c) -> Differenz; m_2*c*T_2 = m_1*c*T_1 + m_E*c*T_E (T_1, T_E absolut,
    m_2 = m_1 + m_E) -> absolut. Nur für Gleichungen, die in allen beteiligten Temperaturen
    linear sind (nicht sigma*T^4, (T_s - T_inf)^(1/4), Stoffwertfunktionen), und nur wenn das
    Ergebnis genau absolut (1) oder Differenz (0) ist. Generisch, ohne Namen und Gleichungsform.
    """
    values, residual = zero_point
    temps = [n for n in sorted(_names_in(parsed[0]) | _names_in(parsed[1]))
             if n in known and n in values and known[n].quantity is not None
             and _is_temp_q(known[n].quantity) and not known[n].literal]
    if not any(n in free for n in temps):
        return

    def shifted(amounts):
        trial = dict(values)
        for name, amount in amounts.items():
            trial[name] = values[name] + amount
        return residual(key, trial)

    step = 1.0
    base = residual(key, values)
    if not np.isfinite(base):
        return
    for name in temps:
        d1, d2 = shifted({name: step}) - base, shifted({name: 2 * step}) - base
        if not (np.isfinite(d1) and np.isfinite(d2)) or d1 == 0:
            return
        if abs(d2 - 2 * d1) > 1e-6 * abs(d1):
            return                       # nicht linear in dieser Temperatur
    # jede freie Größe der Reihe nach, die übrigen mit ihrem (aktuellen) Charakter
    for target in temps:
        if target not in free:
            continue
        others = [n for n in temps if n != target]
        if any(known[n].weight not in (0.0, 1.0) for n in others):
            continue
        movers = [n for n in others if known[n].weight == 1.0]
        a = shifted({name: step for name in movers}) - base if movers else 0.0
        b = shifted({target: step}) - base
        w = -a / b
        for candidate in (0.0, 1.0):
            if abs(w - candidate) < 1e-6:
                out[target].append((prio, candidate))


def _resolve_temperature_weights(parsed_eqs, all_dims: Dict[str, DimensionInfo],
                                 user_known: Dict[str, DimensionInfo],
                                 max_iterations: int = 50,
                                 products: bool = False,
                                 integral: bool = True,
                                 zero_point=None, keys=None) -> Dict[str, Optional[float]]:
    """
    Bestimmt für alle Temperatur-Variablen, ob absolut (1) oder Differenz (0).

    Vorgaben: bekannte Einheiten ('delta_K' = Differenz, 'K' aus °C/°F = absolut;
    in K eingegebene Größen ohne Vorgabe, weight None). Alle übrigen werden per
    Jacobi-Iteration aus den Gleichungen abgeleitet (reihenfolgeunabhängig, ohne
    Namen); nicht bestimmbar -> None.
    """
    fixed: Dict[str, float] = {}
    free: List[str] = []
    for v, d in all_dims.items():
        if not _is_temp_q(d.quantity):
            continue
        if v in user_known and user_known[v].weight is not None:
            fixed[v] = user_known[v].weight
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
                _weight_candidates(parsed, known, free_set, cands, products, integral)
            except Exception:
                continue
        if zero_point is not None and keys is not None:
            for parsed, key in zip(parsed_eqs, keys):
                try:
                    _zero_point_candidates(parsed, key, known, free_set, cands, zero_point, 3.5)
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


def _temperature_label(weight: Optional[float], undetermined: str) -> str:
    """Anzeige-Label aus dem Temperatur-Charakter: 0 Differenz, 1 absolut, sonst offen."""
    if weight == 0:
        return 'delta_K'
    if weight == 1:
        return 'K'
    return undetermined


def _propagate(parsed_eqs, known_units: Dict[str, str], open_temperatures=frozenset(),
               undetermined: str = 'K', products: bool = False,
               zero_point=None, keys=None) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Kern der Propagation. Liefert (bekannte Einheiten, abgeleitete Einheiten).
    open_temperatures: bekannte Größen in K, deren Charakter (absolut/Differenz) aus den
    Gleichungen folgt - sie erscheinen mit dem bestimmten Label in den abgeleiteten.
    """
    user = {v: _dim_info_from_unit(u, v) for v, u in known_units.items() if u is not None}
    for v in open_temperatures:
        if v in user and _is_temp_q(user[v].quantity):
            user[v] = replace(user[v], weight=None)
    inferred = _infer_dimensions(parsed_eqs, user)
    # Stufe 2: vollständige Ableitung für alles, was die lokale Propagation offen lässt
    try:
        complete = _complete_inference(parsed_eqs, {**user, **inferred})
    except Exception:
        complete = {}
    for v, d in complete.items():
        if v not in user and (v not in inferred or inferred[v].quantity is None):
            inferred[v] = d
    all_dims = {**user, **inferred}
    weights = _resolve_temperature_weights(parsed_eqs, all_dims, user, products=products,
                                           zero_point=zero_point, keys=keys)
    result = {}
    for v, d in inferred.items():
        label = d.unit if d.unit is not None else ''
        if _is_temp_q(d.quantity):
            label = _temperature_label(weights.get(v), undetermined)
        elif _is_dimless_q(d.quantity):
            # %, ‰, g/kg aus den Eingaben bleiben (x_3 = x_1 + 0.001 mit x_1 in g/kg), sonst Zahl
            label = d.unit if d.unit and d.unit not in ('dimensionless', '-', '1') else ''
        result[v] = label
    for v in open_temperatures:
        if v in user and _is_temp_q(user[v].quantity):
            result[v] = _temperature_label(weights.get(v), undetermined)
    return known_units, result


def _factor_names(node: ast.AST) -> Set[str]:
    """Variablennamen eines Ausdrucks ohne Funktionsnamen (abs, enthalpy, ...)."""
    calls = {id(sub.func) for sub in ast.walk(node) if isinstance(sub, ast.Call)}
    return {sub.id for sub in ast.walk(node) if isinstance(sub, ast.Name) and id(sub) not in calls}


def _kelvin_law_term(node: ast.AST, known: Dict[str, DimensionInfo]) -> bool:
    """
    Summand mit einer Größe als Faktor, die keine Temperatur ist (EER*(T_c - T_0),
    Q/(m*c)): die Gleichung ist ein Gesetz in Kelvin - aus EER = T_0/(T_c - T_0)
    ausmultipliziert -, keine Summe von Temperaturwerten (kein Hinweis).
    """
    return not isinstance(node, ast.Name) and any(
        not (name in known and known[name].quantity is not None and _is_temp_q(known[name].quantity))
        for name in _factor_names(node))


def _literal_factor(node: ast.AST) -> float:
    """Zahlenfaktor eines Produkts/Quotienten (2*p*v/R -> 2, Q/(m*c) -> 1)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        factor = _literal_factor(node.operand)
        return -factor if isinstance(node.op, ast.USub) else factor
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return _literal_factor(node.left) * _literal_factor(node.right)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        right = node.right
        if isinstance(right, ast.Constant) and isinstance(right.value, (int, float)) and right.value:
            return _literal_factor(node.left) / float(right.value)
        return _literal_factor(node.left)
    return 1.0


def temperature_sum_conflicts(equations: Dict[str, str], known_units: Dict[str, str],
                              open_temperatures=frozenset()) -> List[Tuple[str, List[str]]]:
    """
    Gleichungen, in denen Temperaturen so addiert/subtrahiert werden, dass weder eine
    absolute Temperatur noch eine Differenz herauskommt - jede Temperaturgröße ist
    absolut (1, Punkt) oder Differenz (0), und die Summe geht mit keiner Belegung auf
    (Koeffizientensumme weder 1 noch 0: T_3 = T_1 + T_2). Typisch: eine Temperaturdifferenz
    in °C angegeben (T_2 = T_1 + x mit x = 10 °C, also 283.15 K). Ohne festgelegten Ursprung
    keine zulässige Operation (main: Fehler). Summanden, die eine Temperatur skalieren
    (T_1*r, 2*T_1, EER*(T_c - T_0)), sind Gesetze in Kelvin und werden nicht geprüft.
    Generisch aus der Struktur, ohne Namen.

    Returns:
        [(Originalgleichung, Temperaturgrößen der Summe mit festem Charakter), ...]
    """
    if not PINT_AVAILABLE:
        return []
    import itertools
    parsed_eqs, sources = [], []
    for parsed_key, original_eq in equations.items():
        source = original_eq if original_eq else parsed_key
        parsed = _parse_equation(source)
        if parsed is None and parsed_key and parsed_key != source:
            parsed = _parse_equation(parsed_key)
        if parsed is not None:
            parsed_eqs.append(parsed)
            sources.append(source)
    user = {v: _dim_info_from_unit(u, v) for v, u in known_units.items() if u is not None}
    for v in open_temperatures:
        if v in user and _is_temp_q(user[v].quantity):
            user[v] = replace(user[v], weight=None)
    inferred = _infer_dimensions(parsed_eqs, user)
    all_dims = {**user, **inferred}
    weights = _resolve_temperature_weights(parsed_eqs, all_dims, user)
    known = {v: (replace(d, weight=weights[v]) if v in weights else d) for v, d in all_dims.items()}
    conflicts = []
    for (left, right), source in zip(parsed_eqs, sources):
        terms = _flatten_additive(left, 1) + _flatten_additive(right, -1)
        evals = [(s, n, _eval_collect(n, known)[0]) for s, n in terms]
        if not any(d.quantity is not None and _is_temp_q(d.quantity) and not d.literal for _, _, d in evals):
            continue
        # Summand "Größe mal Temperatur/Differenz" (Q/(m*c), EER*(T_c - T_0)): Gesetz in Kelvin, sein
        # Charakter ist offen - Temperatur oder Differenz, wie eine benannte Zwischengröße
        laws = [(s, n) for s, n, d in evals if not d.literal and _kelvin_law_term(n, known)]
        law_ids = {id(n) for _, n in laws}
        # Summand, der eine Temperatur skaliert (T_1*5^0.286, 2*T_1; Gewicht weder 0 noch 1): Gesetz
        # auf der Verhältnisskala Kelvin (Isentrope, Gasgesetz - Festlegung der Eingabesprache);
        # geprüft werden Summen von Temperaturpunkten und Differenzen mit Faktor ±1
        if any(not isinstance(n, ast.Name) and id(n) not in law_ids and not d.literal and d.quantity is not None
               and _is_temp_q(d.quantity) and d.weight is not None and d.weight not in (0.0, 1.0)
               for _, n, d in evals):
            continue
        # offen: unbestimmt oder ein Gewicht, das weder absolut (1) noch Differenz (0) ist
        # (z.B. 2 = Summe zweier absoluter Temperaturen) - muss hier aufgehen
        open_names = [(s, n.id) for s, n, d in evals
                      if isinstance(n, ast.Name) and d.quantity is not None and _is_temp_q(d.quantity)
                      and (d.weight is None or d.weight not in (0.0, 1.0)) and n.id not in user]
        rest = [(s, d) for s, n, d in evals if id(n) not in law_ids
                and not (isinstance(n, ast.Name) and any(n.id == name for _, name in open_names))]
        total = _sum_weights(rest)
        # je offener Summand: Differenz (0) oder Temperatur (1, bei einem Gesetz-Summanden mal
        # seinem Zahlenfaktor: 2*p*v/R ist das Doppelte einer Temperatur)
        options = ([(s, (0.0, 1.0)) for s, _ in open_names]
                   + [(s, (0.0, _literal_factor(n))) for s, n in laws])
        if total is None or len(options) > 8:
            continue
        if not any(abs(total + sum(s * w for (s, _), w in zip(options, combo))) < 1e-9
                   for combo in itertools.product(*(weights for _, weights in options))):
            fixed = [n.id for _, n, d in evals if isinstance(n, ast.Name) and d.weight is not None
                     and d.quantity is not None and _is_temp_q(d.quantity)]
            conflicts.append((source, fixed))
    return conflicts


def numeric_value_form(equation: str, names, unit: str,
                       units: Optional[Dict[str, str]] = None) -> Optional[str]:
    """
    Dieselbe Gleichung als Zahlenwertgleichung in 'unit': jede der Größen 'names' (Temperatur-
    punkte) wird durch value(x, unit) ersetzt - damit ist der Ursprung der Skala festgelegt
    (eindeutig auf jeder Skala, unabhängig von der Eingabe in °C, °F oder K):
    T3 = T1 + T2 -> value(T3, °C) = value(T1, °C) + value(T2, °C) (50 °C bei 30 °C und 20 °C).
    Ein Summand ohne diese Größen, der eine Temperatur(differenz) ist (Q/(m*c); Dimension aus
    'units'), wird zur Zahl in K: value(Q/(m*c), K). Funktionsargumente bleiben Größen.
    """
    import copy
    parsed = _parse_equation(equation)
    if parsed is None:
        return None
    names = set(names)
    known = {v: _dim_info_from_unit(u, v) for v, u in (units or {}).items() if u is not None}

    def call(node, unit_text):
        return ast.Call(func=ast.Name(id='value', ctx=ast.Load()),
                        args=[node, ast.Name(id=unit_text, ctx=ast.Load())], keywords=[])

    def has_point(node):
        calls = {id(sub.func) for sub in ast.walk(node) if isinstance(sub, ast.Call)}
        return any(isinstance(sub, ast.Name) and sub.id in names and id(sub) not in calls
                   for sub in ast.walk(node))

    def term(node):
        if not has_point(node):
            dim = _eval_collect(node, known)[0] if known else None
            if dim is not None and not dim.literal and dim.quantity is not None and _is_temp_q(dim.quantity):
                return call(node, '__KELVIN__')
            return node
        return inner(node)

    def inner(node):
        if isinstance(node, ast.Name):
            return call(node, '__UNIT__') if node.id in names else node
        if isinstance(node, ast.BinOp):
            wrap = term if isinstance(node.op, (ast.Add, ast.Sub)) else inner
            return ast.BinOp(left=wrap(node.left), op=node.op, right=wrap(node.right))
        if isinstance(node, ast.UnaryOp):
            return ast.UnaryOp(op=node.op, operand=term(node.operand))
        return node      # Funktionsaufrufe: Argumente bleiben Größen

    try:
        sides = [ast.unparse(ast.fix_missing_locations(term(copy.deepcopy(side)))) for side in parsed]
    except Exception:
        return None
    return ' = '.join(sides).replace('**', '^').replace('__UNIT__', unit).replace('__KELVIN__', 'K')


def missing_unit_annotations(equations: Dict[str, str], known_units: Dict[str, str],
                             open_temperatures=frozenset()) -> List[Tuple[List[str], List[str]]]:
    """
    Größen, deren Einheit aus den Gleichungen nicht folgt (nur ihr Zusammenhang, z.B.
    a*b = X allein), und je Gruppe die Größen, deren Einheit man angeben muss, damit
    alle übrigen folgen: die freien Größen der exakten Elimination (Stufe 2) - ihre
    Anzahl ist die Zahl fehlender Angaben (Rang-Defekt). Generisch, ohne Namen.

    Returns:
        [(unbestimmte Größen der Gruppe, anzugebende Größen), ...] in Blatt-Reihenfolge
    """
    if not PINT_AVAILABLE:
        return []
    parsed_eqs = []
    for parsed_key, original_eq in equations.items():
        source = original_eq if original_eq else parsed_key
        parsed = _parse_equation(source)
        if parsed is None and parsed_key and parsed_key != source:
            parsed = _parse_equation(parsed_key)
        if parsed is not None:
            parsed_eqs.append(parsed)
    user = {v: _dim_info_from_unit(u, v) for v, u in known_units.items() if u is not None}
    inferred = _infer_dimensions(parsed_eqs, user)
    known = {**user, **inferred}
    report: Dict[str, Any] = {}
    complete = _complete_inference(parsed_eqs, known, report)
    order = report.get('order', {})
    pivots = report.get('pivots', {})
    determined = set(complete) | {v for v, d in known.items() if d.quantity is not None}
    unknown = [v for v in sorted(order, key=order.get) if not v.startswith('#') and v not in determined]
    if not unknown:
        return []
    parent: Dict[str, str] = {}

    def find(name):
        while parent.setdefault(name, name) != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    for pivot, (coeffs, _) in pivots.items():
        for name in coeffs:
            parent[find(name)] = find(pivot)
    groups: Dict[str, List[str]] = {}
    for name in unknown:
        groups.setdefault(find(name), []).append(name)
    result = []
    for root, members in groups.items():
        everything = sorted((n for n in order if find(n) == root), key=order.get)
        free = [n for n in everything if n not in pivots]
        suggest = [n for n in free if not n.startswith('#')]
        if len(suggest) < len(free):
            # an einer nicht bestimmbaren Hilfsgröße (y = x^n, n Variable): direkt angeben
            for name in members:
                if name in pivots and name not in suggest and any(
                        c.startswith('#') and c not in pivots for c in pivots[name][0]):
                    suggest.append(name)
        result.append((members, suggest or members[:1]))
    return result


def scale_offset_literals(equations: Dict[str, str], units: Dict[str, str]) -> List[Tuple[str, float, str]]:
    """
    Zahlen gleich dem Nullpunkt einer Temperaturskala (273.15 °C -> K, 459.67 °F -> °R; aus pint)
    in einer Summe mit einer Temperatur - die Umrechnung "T + 273.15" aus Excel. Hier sind
    Temperaturen bereits in Kelvin; die Zahl verschiebt den Nullpunkt ein zweites Mal, ohne
    Einheitenfehler. Generisch über die Skalen von pint, unabhängig von Namen.

    Returns:
        [(Originalgleichung, Zahl, Skala), ...]
    """
    if not PINT_AVAILABLE:
        return []
    origins = {}
    for scale, reference in (('°C', 'kelvin'), ('°F', 'degR')):
        try:
            unit = 'degC' if scale == '°C' else 'degF'
            origins[float(ureg.Quantity(0.0, unit).to(reference).magnitude)] = scale
        except Exception:
            continue
    kelvin = ureg.kelvin.dimensionality
    found = []
    for parsed_key, original_eq in equations.items():
        source = original_eq if original_eq else parsed_key
        parsed = _parse_equation(source)
        if parsed is None:
            continue
        hit = None
        for side in parsed:
            for node in ast.walk(side):
                if not (isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub))):
                    continue
                terms = _flatten_additive(node)
                numbers = [n.value for _, n in terms
                           if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))]
                matches = [(abs(float(v)), origins[o]) for v in numbers for o in origins
                           if abs(abs(float(v)) - o) < 1e-9 * o]
                if not matches:
                    continue
                if any(_expression_dimension(ast.unparse(n), units)[0] == kelvin
                       for _, n in terms if not isinstance(n, ast.Constant)):
                    hit = matches[0]
                    break
            if hit:
                break
        if hit:
            found.append((source, hit[0], hit[1]))
    return found


def si_number_literals(equations: Dict[str, str],
                       input_units: Dict[str, str]) -> List[Tuple[str, float, str, str, str]]:
    """
    Zahlen ohne Einheit in einer Summe mit einer Größe, die in einer Nicht-SI-Einheit
    eingegeben wurde (24 - t_Sperr mit t_Sperr = 2 h): die Zahl ist ein SI-Wert (24 s),
    gemeint ist meist die Einheit der Eingabe (24 h). Kein Einheitenfehler - Zahlen sind in
    Summen dimensionsneutral. Ausgenommen: dimensionslose Summen, Temperaturen (Differenzen
    in K sind SI) und 0. Generisch aus der Struktur, unabhängig von Namen.

    Returns:
        [(Originalgleichung, Zahl, Größe, Eingabe-Einheit, SI-Einheit), ...]
    """
    if not PINT_AVAILABLE:
        return []
    factors = {}
    for name, unit in input_units.items():
        try:
            q = ureg.Quantity(1.0, normalize_unit(unit.strip()))
            if _is_dimless_q(q) or _is_temp_q(q):
                continue
            base = q.to_base_units()
            if abs(base.magnitude - 1.0) > 1e-9:
                si = next((expr for expr, _ in _LABEL_ENTRIES
                           if ureg.Quantity(1.0, expr).dimensionality == base.dimensionality),
                          _generic_label(base.dimensionality))
                factors[name] = (unit.strip(), si)
        except Exception:
            continue
    found = []
    for parsed_key, original_eq in equations.items():
        source = original_eq if original_eq else parsed_key
        parsed = _parse_equation(source)
        if parsed is None:
            continue
        hit = None
        for node in (sub for side in parsed for sub in ast.walk(side)):
            if not (isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub))):
                continue
            terms = _flatten_additive(node)
            numbers = [float(n.value) for _, n in terms if isinstance(n, ast.Constant)
                       and isinstance(n.value, (int, float)) and not isinstance(n.value, bool) and n.value]
            names = [n.id for _, n in terms if isinstance(n, ast.Name) and n.id in factors]
            if numbers and names:
                hit = (numbers[0], names[0])
                break
        if hit:
            unit, si = factors[hit[1]]
            found.append((source, hit[0], hit[1], unit, si))
    return found


# ----------------------------------------------------------------------------
# Winkel: intern SI (rad = 1) wie alles. Zahlen ohne Einheit sind in sin/cos/tan Grad
# (Festlegung der Eingabesprache: sin(30) = 0.5), asin/acos/atan liefern Grad.
# Eine Winkelgröße (Einheit mit Winkel: deg, rad, rad/s * s, quantity(z, deg)) liest die
# Funktion dagegen in ihrer Einheit: sin(phi) mit phi = 30 deg, sin(omega*t) mit omega in rad/s.
# Welche Ausdrücke Winkelgrößen sind, folgt aus den Einheiten (Exponent des Winkels, wie die
# Dimension über Summen/Produkte), nie aus Namen; nicht bestimmbar -> Zahl (Grad).
# ----------------------------------------------------------------------------
_TRIG_FUNCS = {'sin', 'cos', 'tan'}
_INVERSE_TRIG_FUNCS = {'asin', 'acos', 'atan'}


def _numeric_literal(node) -> Optional[float]:
    """Zahlenwert eines Ausdrucks nur aus Zahlen (2, -1, 1/3), sonst None."""
    if any(isinstance(n, (ast.Name, ast.Call)) for n in ast.walk(node)):
        return None
    try:
        value = eval(compile(ast.Expression(body=node), '<exponent>', 'eval'), {'__builtins__': {}}, {})
        return float(value)
    except Exception:
        return None


def _angle_analysis(equations: List[str], declared: Dict[str, int]):
    """
    Winkel-Exponenten aller Größen und Ausdrücke (lineares System wie _complete_inference, ein
    Exponent je Größe; widersprechende Gleichungen ausgelassen - rad = 1 erlaubt u = omega*r).
    Summen: ein Winkel unter Zahlen bzw. Größen ohne Winkel macht die Summe zum Winkel (die
    übrigen Summanden sind SI-Werte in rad), unabhängig von der Reihenfolge der Summanden.

    Returns:
        (resolve, Größen-Exponenten, [(Gleichung, Aufruf-Knoten, Form, invers), ...],
         [(Gleichung, [(Summand, Form), ...]), ...])
    """
    from fractions import Fraction
    from units import angle_exponent
    rows = []
    order: Dict[str, int] = {}
    fresh = [0]
    calls = []
    sums = []

    def const(value):
        return ({}, Fraction(value))

    def variable(name):
        order.setdefault(name, len(order))
        return ({name: Fraction(1)}, Fraction(0))

    def new_variable():
        fresh[0] += 1
        return variable(f"#{fresh[0]}")

    def combine(a, b, factor=1):
        coeffs = dict(a[0])
        for name, c in b[0].items():
            coeffs[name] = coeffs.get(name, Fraction(0)) + factor * c
        return ({n: c for n, c in coeffs.items() if c != 0}, a[1] + factor * b[1])

    def equal(a, b):
        coeffs, rhs = combine(a, b, -1)
        if coeffs:
            rows.append((coeffs, -rhs))

    def unify(forms):
        forms = [f for f in forms if f is not _ANY]
        if not forms:
            return _ANY
        constants = {f[1] for f in forms if not f[0]}
        nonzero = constants - {0}
        if len(nonzero) > 1:
            reference = new_variable()          # verschiedene Winkel-Exponenten: offen
        elif constants:
            reference = const(next(iter(nonzero)) if nonzero else 0)
        else:
            reference = forms[0]
        for f in forms:
            if f[0] and f is not reference:
                equal(f, reference)
        return reference

    def unit_text(node):
        return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None

    def form(node, eq):
        if isinstance(node, ast.Constant):
            return _ANY
        if isinstance(node, ast.Name):
            if node.id == 'pi':
                return _ANY
            if node.id in declared:
                return const(declared[node.id])
            return variable(node.id)
        if isinstance(node, ast.UnaryOp):
            return form(node.operand, eq)
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, (ast.Add, ast.Sub)):
                terms = [(term, form(term, eq)) for _, term in _flatten_additive(node)]
                sums.append((eq, terms))
                return unify([f for _, f in terms])
            if isinstance(node.op, (ast.Mult, ast.Div)):
                left, right = form(node.left, eq), form(node.right, eq)
                if left is _ANY and right is _ANY:
                    return _ANY
                left = const(0) if left is _ANY else left
                right = const(0) if right is _ANY else right
                return combine(left, right, 1 if isinstance(node.op, ast.Mult) else -1)
            if isinstance(node.op, ast.Pow):
                base = form(node.left, eq)
                n = _numeric_literal(node.right)
                if base is _ANY:
                    return _ANY
                if n is None:
                    return const(0)
                n = Fraction(n).limit_denominator(1000)
                return ({k: c * n for k, c in base[0].items() if c * n != 0}, base[1] * n)
            return const(0)
        if isinstance(node, ast.Call):
            fname = _func_name(node)
            args = node.args
            if fname in ('value', 'quantity') and len(args) == 2 and unit_text(args[1]) is not None:
                exponent = angle_exponent(unit_text(args[1]))
                argument = form(args[0], eq)
                if fname == 'value' and argument is not _ANY:
                    equal(argument, const(exponent))       # value(x, deg): x ist ein Winkel
                return const(0) if fname == 'value' else const(exponent)
            if fname in _TRIG_FUNCS and len(args) == 1:
                calls.append((eq, node, form(args[0], eq), False))
                return const(0)
            if fname in _INVERSE_TRIG_FUNCS and len(args) == 1:
                result = new_variable()
                calls.append((eq, node, result, True))
                form(args[0], eq)
                return result
            if fname == 'sqrt' and len(args) == 1:
                f = form(args[0], eq)
                return f if f is _ANY else ({k: c / 2 for k, c in f[0].items()}, f[1] / 2)
            if fname in ('abs', 'ceil', 'floor', 'round') and args:
                return form(args[0], eq)
            if fname in ('max', 'min', 'IF'):
                if fname == 'IF' and len(args) == 5:
                    unify([form(args[0], eq), form(args[1], eq)])
                return unify([form(a, eq) for a in (args[2:] if fname == 'IF' else args)])
            for a in args:
                form(a, eq)
            for keyword in node.keywords:
                form(keyword.value, eq)
            return const(0)
        return const(0)

    for eq in equations:
        try:
            tree = ast.parse(eq, mode='eval').body
            unify([form(tree, eq)])
        except Exception:
            continue

    # Gauss-Elimination (exakt), widersprüchliche Zeilen auslassen (wie _complete_inference)
    pivots: Dict[str, Tuple[Dict[str, Any], Any]] = {}
    for coeffs, rhs in rows:
        coeffs = dict(coeffs)
        for name in [n for n in coeffs if n in pivots]:
            factor = coeffs.get(name)
            if not factor:
                continue
            p_coeffs, p_rhs = pivots[name]
            for n2, c2 in p_coeffs.items():
                coeffs[n2] = coeffs.get(n2, 0) - factor * c2
            rhs = rhs - factor * p_rhs
            coeffs = {n: c for n, c in coeffs.items() if c != 0}
        if not coeffs:
            continue
        pivot = min(coeffs, key=lambda n: order.get(n, 0))
        factor = coeffs[pivot]
        coeffs = {n: c / factor for n, c in coeffs.items()}
        rhs = rhs / factor
        for name, (p_coeffs, p_rhs) in list(pivots.items()):
            f = p_coeffs.get(pivot)
            if f:
                new_coeffs = dict(p_coeffs)
                for n2, c2 in coeffs.items():
                    new_coeffs[n2] = new_coeffs.get(n2, 0) - f * c2
                pivots[name] = ({n: c for n, c in new_coeffs.items() if c != 0}, p_rhs - f * rhs)
        pivots[pivot] = (coeffs, rhs)
    determined = {name: rhs for name, (coeffs, rhs) in pivots.items() if set(coeffs) == {name}}

    def resolve(f):
        """Exponent einer Form; None = nicht bestimmbar."""
        if f is _ANY:
            return None
        total = f[1]
        for name, c in f[0].items():
            if name not in determined:
                return None
            total += c * determined[name]
        return total

    exponents = {name: value for name, value in determined.items() if not name.startswith('#')}
    return resolve, exponents, calls, sums


def angle_exponents(equations: List[str], declared: Dict[str, int]) -> Dict[str, int]:
    """Winkel-Exponenten der berechneten Größen (1 = Winkel in rad), siehe _angle_analysis."""
    exponents = _angle_analysis(equations, declared)[1]
    return {name: int(value) for name, value in exponents.items() if value and value.denominator == 1}


def angle_sum_numbers(equations: Dict[str, str], declared: Dict[str, int], plain_names: Set[str],
                      units: Dict[str, str]) -> List[Tuple[str, str, str]]:
    """
    Zahlen bzw. Größen ohne Einheit in einer Summe mit einem Winkel (alpha + 90, omega*t + phi_0
    mit phi_0 = 30): Winkel sind intern rad, die Zahl ist daher ein SI-Wert in rad - auch wenn
    Grad gemeint sind. Nur für einen Hinweis; die Rechnung bleibt unverändert.

    Returns:
        [(Originalgleichung, Summand ohne Einheit, Winkel-Summand), ...] - je Gleichung einmal
    """
    resolve, _, _, sums = _angle_analysis(list(equations), declared)
    dimensionless = ureg.dimensionless.dimensionality if PINT_AVAILABLE else None
    found, seen = [], set()
    for eq, terms in sums:
        if eq in seen:
            continue
        plain = [t for t, _ in terms
                 if (isinstance(t, ast.Constant) and isinstance(t.value, (int, float))
                     and not isinstance(t.value, bool) and t.value)
                 or (isinstance(t, ast.Name) and t.id in plain_names)]
        angles = [t for t, f in terms if resolve(f) not in (None, 0)]
        if not plain or not angles:
            continue
        # vorgegebene Winkel zuerst nennen (alpha in h = 90 - alpha, nicht h)
        angles.sort(key=lambda t: not (isinstance(t, ast.Name) and declared.get(t.id)))
        for angle in angles:
            text = ast.unparse(angle)
            try:
                if _expression_dimension(text, units)[0] != dimensionless:
                    continue
            except Exception:
                continue
            seen.add(eq)
            found.append((equations[eq] or eq, ast.unparse(plain[0]), text))
            break
    return found


def angle_function_forms(equations: List[str], declared: Dict[str, int]) -> Dict[str, str]:
    """
    Gleichungen mit Winkelfunktionen einer Winkelgröße in der Schreibweise mit Zahlenwert in Grad
    (die Funktionen rechnen in Grad): sin(phi) -> sin(value(phi, 'deg')), alpha = asin(z) ->
    quantity(asin(z), 'deg'), wenn alpha eine Winkelgröße ist. {alte Gleichung: neue}.
    declared: Winkel-Exponent der vorgegebenen Größen (mit Einheit; ohne Einheit 0).
    """
    resolve, _, calls, _ = _angle_analysis(equations, declared)
    edits: Dict[str, list] = defaultdict(list)
    for eq, node, f, inverse in calls:
        exponent = resolve(f)
        if not exponent:
            continue
        if inverse:
            edits[eq].append((node.col_offset, node.end_col_offset, 'quantity(', ", 'deg')"))
        else:
            arg = node.args[0]
            edits[eq].append((arg.col_offset, arg.end_col_offset, 'value(', ", 'deg')"))
    result = {}
    for eq, items in edits.items():
        data = eq.encode('utf-8')        # ast-Positionen sind Byte-Positionen
        inserts = []
        for depth, (start, end, before, after) in enumerate(sorted(items, key=lambda e: (e[0], -e[1]))):
            inserts.append((start, depth, before))
            inserts.append((end, -depth, after))
        # von hinten einfügen; an gleicher Stelle zuerst der innere Ausdruck
        for position, _, text in sorted(inserts, key=lambda i: (i[0], i[1]), reverse=True):
            data = data[:position] + text.encode('utf-8') + data[position:]
        result[eq] = data.decode('utf-8')
    return result


def scale_origin(unit: str) -> Optional[float]:
    """Nullpunkt einer Temperatur-Einheit in K (°C 273.15, °F 255.37, K 0) aus pint."""
    if not PINT_AVAILABLE or not unit:
        return None
    try:
        return float(ureg.Quantity(0.0, normalize_unit(unit.strip())).to('kelvin').magnitude)
    except Exception:
        return None


# ============================================================================
# Öffentliche Hilfs-API (abwärtskompatibel)
# ============================================================================


def _remove_comments(equation: str) -> str:
    """Entfernt Kommentare aus einer Gleichung.

    Kommentare sind in "..." oder {...} eingeschlossen.
    """
    # Entferne "..." Kommentare
    result = re.sub(r'"[^"]*"', '', equation)
    # Entferne {...} Kommentare
    result = re.sub(r'\{[^}]*\}', '', result)
    return result.strip()


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
    return {var: unit for var, unit in all_units.items() if var not in known_units}


# ============================================================================
# Function Argument Analysis for Bidirectional Unit Inference
# ============================================================================

# Einheiten für Funktionsargumente (SI-Einheiten)
# Einheiten der Schlüsselwort-Argumente je Funktionsfamilie (klein geschrieben; gleiche Namen
# bedeuten je Familie Verschiedenes: x ist bei Stoffwerten der Dampfgehalt, bei HumidAir = w)
FUNCTION_ARGUMENT_UNITS = {
    # CoolProp Stoffwertfunktionen
    't': 'K',           # Temperatur
    'p': 'Pa',          # Druck
    'h': 'J/kg',        # Enthalpie
    's': 'J/(kg*K)',    # Entropie
    'x': '',            # Dampfgehalt (dimensionslos)
    'rho': 'kg/m^3',    # Dichte
    'd': 'kg/m^3',      # Dichte (Alias)
    'v': 'm^3/kg',      # Spezifisches Volumen
    'u': 'J/kg',        # Innere Energie
}

# HumidAir - Namen wie humid_air.INPUT_MAP (x = w, phi/rf = rh, p = p_tot)
HUMID_AIR_ARGUMENT_UNITS = {
    't': 'K', 'p_tot': 'Pa', 'p': 'Pa', 'h': 'J/kg', 'p_w': 'Pa', 't_dp': 'K', 't_wb': 'K',
    'w': 'kg/kg', 'x': 'kg/kg',     # Feuchtebeladung (dimensionslos, Label kg/kg für Startwert/Anzeige)
    'rh': '', 'rf': '', 'phi': '',  # relative Feuchte
}


def function_argument_unit(function: str, argument: str) -> Optional[str]:
    """Einheit des Schlüsselwort-Arguments 'argument' der Funktion 'function' (None: unbekannt)."""
    table = HUMID_AIR_ARGUMENT_UNITS if function.lower() == 'humidair' else FUNCTION_ARGUMENT_UNITS
    return table.get(argument.lower())


def propagate_all_units_complete(equations: Dict[str, str], known_units: Dict[str, str],
                                  max_iterations: int = 15, open_temperatures=frozenset(),
                                  undetermined_temperature: str = 'K',
                                  differences_in_products: bool = False,
                                  zero_point=None) -> Dict[str, str]:
    """
    Vollständige Einheiten-Propagation mit allen Quellen.

    Kombiniert:
    1. Funktionsrückgabewerte (enthalpy → J/kg, HumidAir(T_dp, ...) → K, Eb → W/(m^2*um))
    2. Funktionsargumente bidirektional (T=T_1 → T_1: K)
    3. Arithmetische Constraint-Propagation (a + b = c → alle gleiche Einheit,
       Produkte/Quotienten/Potenzen rückwärts, sqrt, abs/max/min)
    4. Temperatur-Charakter aus der Struktur, nie aus Namen: Differenzen → 'delta_K',
       absolute Temperaturen → 'K' (T_1 - T_2 Differenz; T_abs ± Differenz absolut;
       (T_1 + T_2)/2 absolut). Nicht bestimmbar (nur in Produkten wie Q = m*c*theta)
       → undetermined_temperature ('K' für Startwerte; die Anzeige verwendet 'delta_K',
       also K statt °C - richtig für beide Deutungen)

    Das Ergebnis ist unabhängig von der Reihenfolge der Gleichungen. Zahlenliterale
    in Summen sind neutral (T_out = T_s - 2 → beide K) und gelten nur dann als
    Hinweis auf 'dimensionslos', wenn sonst nichts bekannt ist.

    Args:
        equations: Dict von {parsed_equation: original_equation}
        known_units: Dict von {variable: unit} bekannter Einheiten
        max_iterations: (nur aus Kompatibilitätsgründen; die Iteration läuft bis
            zum Fixpunkt)
        open_temperatures: bekannte Größen in K (als K eingegeben), deren Charakter aus
            den Gleichungen folgt; ihr Label im Ergebnis ist das bestimmte ('K'/'delta_K')
        undetermined_temperature: Label für nicht bestimmbaren Charakter
        differences_in_products: Temperatur in einem Produkt ohne Temperatur-Dimension
            als Differenz werten, wo die Summen nichts entscheiden (für die Anzeige)
        zero_point: (Werte der Lösung in SI, residual(gleichung, werte)) - Nullpunkt-Test
            für den Temperatur-Charakter (nur Anzeige, `_zero_point_candidates`)

    Returns:
        Dict von ALLEN Einheiten (bekannte + abgeleitete). Einheiten sind
        Anzeige-Labels (z.B. 'kW', 'kJ/kg', 'bar', 'K', 'delta_K', 'W/(m^2*K)', '').
    """
    if not PINT_AVAILABLE:
        return known_units.copy()

    parsed_eqs, keys = [], []
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
            keys.append(parsed_key)

    try:
        _, inferred = _propagate(parsed_eqs, known_units, frozenset(open_temperatures),
                                 undetermined_temperature, differences_in_products,
                                 zero_point=zero_point, keys=keys)
    except Exception:
        inferred = {}

    result = dict(known_units)
    for var, unit in inferred.items():
        if var not in known_units or var in open_temperatures:
            result[var] = unit
    return result


# ============================================================================
# Unit Consistency Checking
# ============================================================================


def _dimensionless_sources(equations: Dict[str, str], all_units: Dict[str, str],
                           given: Set[str]) -> Dict[str, List[str]]:
    """
    Je dimensionsloser Größe die ohne Einheit EINGEGEBENEN Größen, von denen sie die
    Dimension über Summen erbt (T_B - T_s mit T_s = -10: T_B -> [T_s]); eine Eingabe
    selbst -> [sie selbst]. Summanden derselben Summe haben dieselbe Dimension.
    """
    links: Dict[str, Set[str]] = {}
    for parsed_key, original_eq in equations.items():
        parsed = _parse_equation(original_eq or parsed_key)
        if parsed is None:
            continue
        for node in (sub for side in parsed for sub in ast.walk(side)):
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
                names = [n.id for _, n in _flatten_additive(node) if isinstance(n, ast.Name)]
                for a in names:
                    links.setdefault(a, set()).update(b for b in names if b != a)
        # die ganze Gleichung als Summe (T_B = T_s + 2 wie T_B - T_s = 2; auch T_B = T_s)
        left, right = parsed
        names = [n.id for _, n in _flatten_additive(left, 1) + _flatten_additive(right, -1)
                 if isinstance(n, ast.Name)]
        for a in names:
            links.setdefault(a, set()).update(b for b in names if b != a)
    sources = {}
    for var, unit in all_units.items():
        if unit != '':
            continue
        if var in given:
            sources[var] = [var]
            continue
        seen, todo = {var}, [var]
        while todo:
            for other in links.get(todo.pop(), ()):
                if other not in seen and all_units.get(other) == '':
                    seen.add(other)
                    todo.append(other)
        sources[var] = sorted(seen & given)
    return sources


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
    sources = _dimensionless_sources(equations, all_units, {v for v, u in known_units.items() if u == ''})

    for parsed_eq, original_eq in equations.items():
        # Anzeige ohne Gleichung (EQUILIBRIUM (Zeile n): ...): die interne Form prüfen
        source = original_eq if _parse_equation(original_eq) is not None else parsed_eq
        error = check_equation_dimensions(source, all_units, sources)

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
                explanation = f"Dimensionsfehler: {left_unit} ≠ {right_unit}"
                # Variable = linke Seite, wenn dort ein reiner Variablenname steht
                lhs = _remove_comments(original_eq).split('=', 1)[0].strip()
                variable = lhs if re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', lhs) else 'Dimensionsfehler'
                detail = error.get('detail')
                if detail:
                    # Summe mit verschiedenen Einheiten: Terme und ihre Einheiten nennen
                    text, suspects = detail
                    dim_info = f"Summe mit verschiedenen Einheiten - {text}"
                    explanation = f"Dimensionsfehler: {text}"
                    if suspects:
                        variable = ', '.join(suspects)
                warnings.append(UnitWarning(
                    variable=variable,
                    equations=[original_eq],
                    units={original_eq: dim_info},
                    explanation=explanation,
                    conversion_factor=0
                ))

    # Argument mit falscher Dimension (Masse als Stoffmenge, Druck als T=): diese Warnung nennt die
    # Ursache - die Folgewarnung "Inkompatible Terme" derselben Gleichung entfällt
    arguments = _argument_dimension_warnings(equations, all_units)
    explained = {eq for w in arguments for eq in w.equations}
    if explained:
        warnings = [w for w in warnings if not (set(w.equations) & explained)]
    return _merge_same_warnings(warnings + arguments)


def _merge_same_warnings(warnings: list) -> list:
    """Warnungen mit gleichem Text (dieselbe Summe in mehreren Gleichungen, z.B. je OUT-Stoff einer
    EQUILIBRIUM-Anweisung) einmal, mit allen Gleichungen."""
    merged, index = [], {}
    for w in warnings:
        key = (w.explanation, tuple(sorted(set(w.units.values()))))
        if key in index:
            first = index[key]
            for eq in w.equations:
                if eq not in first.equations:
                    first.equations.append(eq)
            continue
        index[key] = w
        merged.append(w)
    return merged


def pretty_label(unit: str) -> str:
    """Einheit für Meldungen (degC -> °C, delta_K -> K, um -> µm, mole -> mol)."""
    if not unit:
        return '-'
    text = unit
    if ' ' in unit or unit.isalpha():
        # pint-Langform (mole / second, kelvin) -> Kurzzeichen; eingegebene Labels bleiben wie sie sind
        try:
            from units import ureg
            text = f"{ureg.Unit(unit):~}".replace(' / ', '/').replace(' * ', '*').replace(' ** ', '^')
        except Exception:
            pass
    text = text.replace('delta_K', 'K').replace('Δ°C', 'K').replace('degC', '°C').replace('degF', '°F')
    return re.sub(r'(?<![A-Za-z])um(?![A-Za-z])', 'µm', text.replace('mole', 'mol'))


def _argument_dimension_warnings(equations: Dict[str, str], all_units: Dict[str, str]) -> list:
    """
    Funktionsargumente mit bekannter, aber unpassender Dimension: T = 1 bar in
    enthalpy(water, T=T, p=p), eine Masse als Menge in EQUILIBRIUM (Stoffmenge mol bzw. mol/s
    erwartet), p in K. Zahlen ohne Einheit zählen nicht (SI-Werte); Zahlen mit Einheit prüft
    schon der Parser.
    """
    from solver import UnitWarning
    found = []
    seen = {}
    for parsed_key, original in equations.items():
        tree = _parse_equation(original) or _parse_equation(parsed_key)
        if tree is None:
            continue
        for side in tree:
            for node in ast.walk(side):
                fname = _func_name(node)
                if fname is None:
                    continue
                fl = fname.lower()
                checks = []
                if fl in _THERMO_FUNCS or fl == 'humidair':
                    for kw in node.keywords:
                        expected = function_argument_unit(fl, kw.arg) if kw.arg else None
                        if expected is not None:
                            checks.append((kw.value, expected, f"{kw.arg}="))
                elif fl == 'equilibrium' and len(node.args) >= 6:
                    checks += [(node.args[3], 'K', 'T='), (node.args[4], 'Pa', 'p=')]
                    checks += [(a, 'amount', 'Menge') for a in node.args[5:]]
                    # alle Mengen in derselben Einheit (mol neben mol/s ergäbe eine sinnlose Summe)
                    kinds = {}
                    for a in node.args[5:]:
                        expr = ast.unparse(a)
                        dim, missing = compute_expression_dimension(expr, all_units)
                        if dim is not None and not missing and not isinstance(dim, str) \
                                and dict(dim).get('[substance]') == 1:
                            kinds.setdefault(dim, expr)
                    key = (fl, 'Mengen', tuple(sorted(str(d) for d in kinds)))
                    if len(kinds) > 1 and key not in seen:
                        labels = ', '.join(f"{e} in {pretty_label(all_units.get(e) or _generic_label(d))}"
                                           for d, e in kinds.items())
                        text = (f"equilibrium(): Mengen in verschiedenen Einheiten ({labels}) - alle Mengen "
                                f"unter IN in derselben Einheit angeben (mol oder mol/s)")
                        seen[key] = UnitWarning(variable=next(iter(kinds.values())), equations=[original],
                                                units={original: text}, explanation=text, conversion_factor=0)
                        found.append(seen[key])
                    elif key in seen and original not in seen[key].equations:
                        seen[key].equations.append(original)
                elif fl in _RADIATION_ARGS:
                    for arg, kind in zip(node.args, _RADIATION_ARGS[fl]):
                        checks.append((arg, 'K' if kind == 'T' else 'm', 'T' if kind == 'T' else 'Wellenlänge'))
                for arg, expected, label in checks:
                    if isinstance(arg, ast.Constant) or (_func_name(arg) or '') in ('quantity', 'value'):
                        continue
                    expr = ast.unparse(arg)
                    dim, missing = compute_expression_dimension(expr, all_units)
                    if dim is None or missing or isinstance(dim, str):
                        continue
                    dims = dict(dim)
                    if expected == 'amount':
                        ok = dims.get('[substance]') == 1
                        wanted = "eine Stoffmenge (mol bzw. mol/s; aus einer Masse m/molarmass(IG:...))"
                    else:
                        reference = _q_for(expected)
                        ok = reference is not None and dim == reference.dimensionality
                        wanted = f"die Dimension von {pretty_label(expected) if expected else 'einer Zahl (-)'}"
                    if ok:
                        continue
                    unit = all_units.get(expr) or _generic_label(dim)
                    key = (fl, expr, label)          # einmal je Größe und Argument (nicht je OUT-Stoff)
                    if key in seen:
                        if original not in seen[key].equations:
                            seen[key].equations.append(original)
                        continue
                    text = (f"{fname}(): {label} {expr} hat die Einheit {pretty_label(unit)}, erwartet ist "
                            f"{wanted}")
                    seen[key] = UnitWarning(variable=expr, equations=[original], units={original: text},
                                            explanation=text, conversion_factor=0)
                    found.append(seen[key])
    return found


# ============================================================================
# Generische dimensionale Konsistenzprüfung mit pint
# ============================================================================

_DIMENSION_ERROR = 'DIMENSION_ERROR_IN_EXPR'


def _expression_dimension(expr: str, unit_map: Dict[str, str]):
    """
    Dimension eines Ausdrucks für die Konsistenzprüfung.

    Returns:
        (dimensionality | 'DIMENSION_ERROR_IN_EXPR' | None, missing_vars, is_literal)
    """
    try:
        node = ast.parse(_quote_unit_arguments(_remove_comments(expr)).replace('^', '**'), mode='eval').body  # value(p, bar): bar ist Text
    except Exception:
        return None, [], False

    names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
    known = {v: _dim_info_from_unit(unit_map[v], v)
             for v in names if v in unit_map and unit_map[v] is not None}
    skip = set()

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


def _incompatible_sum(expr: str, unit_map: Dict[str, str],
                      sources: Optional[Dict[str, List[str]]] = None) -> Optional[Tuple[str, List[str]]]:
    """
    Erste Summe/Differenz im Ausdruck, deren Terme verschiedene Einheiten haben - als Text
    mit der Einheit jedes Terms ("h_9 - h_11s: h_9 in J/kg, h_11s dimensionslos (ohne Einheit
    eingegeben?)") und den verdächtigen Größen (dimensionslose Größen in der Summe).

    sources: {Größe: ohne Einheit eingegebene Größen} - ist eine dimensionslose Größe der Summe
    berechnet (T_B), nennt die Meldung, woher das kommt (T_B - T_s mit T_s = -10 ohne Einheit).
    """
    try:
        node = ast.parse(_remove_comments(expr).replace('^', '**'), mode='eval').body
    except Exception:
        return None
    for sub in ast.walk(node):
        if not (isinstance(sub, ast.BinOp) and isinstance(sub.op, (ast.Add, ast.Sub))):
            continue
        parts = []
        for _, term in _flatten_additive(sub):
            text = ast.unparse(term)
            dim, _, literal = _expression_dimension(text, unit_map)
            if literal or dim is None or dim == _DIMENSION_ERROR:
                continue
            parts.append((text, term, dim))
        if len({str(d) for _, _, d in parts}) < 2:
            continue
        dimensionless = ureg.dimensionless.dimensionality
        described, suspects = [], []
        for text, term, dim in parts:
            if dim == dimensionless:
                hint = ''
                if isinstance(term, ast.Name) and unit_map.get(term.id) == '':
                    origin = (sources or {}).get(term.id)
                    if sources is None or origin == [term.id]:
                        hint = ' (ohne Einheit eingegeben?)'
                        suspects.append(term.id)
                    elif origin:
                        hint = f" (über eine Summe mit {', '.join(origin)} - ohne Einheit eingegeben?)"
                        suspects += [name for name in origin if name not in suspects]
                described.append(f"{text} dimensionslos{hint}")
            else:
                described.append(f"{text} in {si_label_from_dimensionality(dim)}")
        return f"{ast.unparse(sub)}: " + ", ".join(described), suspects
    return None


def check_equation_dimensions(equation: str, unit_map: Dict[str, str],
                              sources: Optional[Dict[str, List[str]]] = None) -> Optional[Dict]:
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

    sides = _split_equation(_quote_unit_arguments(_remove_comments(equation)))
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
            'detail': _incompatible_sum(right, unit_map, sources),
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
            'detail': _incompatible_sum(left, unit_map, sources),
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
