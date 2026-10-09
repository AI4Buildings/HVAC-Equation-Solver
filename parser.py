"""
EES-ähnlicher Gleichungsparser

Unterstützte Syntax:
- Gleichungen: x + y = 10
- Zuweisungen: T1 = 300
- Vektoren: T = 0:10:100 (start:step:end) oder T = 0:100 (start:end, step=1)
- Operatoren: +, -, *, /, ^ (Potenz)
- Funktionen: sin, cos, tan, exp, ln, log10, sqrt, abs, max, min
- Thermodynamik: enthalpy(water, T=100, p=1), density(R134a, T=25, x=1)
- Kommentare: "..." oder {...}
"""

import keyword
import re
import numpy as np
from typing import List, Set, Tuple, Dict, Union, Optional

# Einheiten-Modul (optional, falls nicht vorhanden wird ohne Einheiten gearbeitet)
try:
    from units import (parse_value_with_unit, UnitValue, unit_value_strict, UnknownUnitError,
                       check_unit_dimension)
    UNITS_AVAILABLE = True
except ImportError:
    UNITS_AVAILABLE = False
    UnitValue = None

    class UnknownUnitError(Exception):
        """Platzhalter, wenn das Einheiten-Modul fehlt."""


# Mathematische Funktionen die unterstützt werden
MATH_FUNCTIONS = {
    'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
    'sinh', 'cosh', 'tanh',
    'exp', 'ln', 'lg', 'log10', 'sqrt', 'abs',
    'pi', 'max', 'min'
}

# Thermodynamik-Funktionen (CoolProp)
THERMO_FUNCTIONS = {
    'enthalpy', 'entropy', 'density', 'volume', 'intenergy',
    'quality', 'temperature', 'pressure',
    'viscosity', 'conductivity', 'prandtl',
    'cp', 'cv', 'soundspeed'
}

# Strahlungs-Funktionen (Schwarzkörper)
RADIATION_FUNCTIONS = {
    'eb', 'blackbody', 'blackbody_cumulative', 'wien', 'stefan_boltzmann'
}

# Humid Air Functions
HUMID_AIR_FUNCTIONS = {
    'humidair'
}

# Mapping von EES-Syntax zu Python
FUNCTION_MAP = {
    'ln': 'log',      # ln -> numpy.log
    '^': '**',        # Potenz
}

# Regex für Vektor-Syntax: start:step:end oder start:end
VECTOR_PATTERN_3 = re.compile(r'^(-?\d+\.?\d*):(-?\d+\.?\d*):(-?\d+\.?\d*)$')  # start:step:end
VECTOR_PATTERN_2 = re.compile(r'^(-?\d+\.?\d*):(-?\d+\.?\d*)$')  # start:end (step=1)

# Python-Schlüsselwörter, die als Variablennamen vorkommen können (z.B. lambda
# für die Wärmeleitfähigkeit λ). eval() kann sie nicht als Namen verwenden,
# daher werden sie intern umbenannt: lambda -> _kw_lambda. Für die Anzeige
# macht display_name()/unmangle() das rückgängig.
# and/or/not bleiben Operatoren, True/False/None Konstanten.
KEYWORD_PREFIX = '_kw_'
_MANGLED_KEYWORDS = sorted(set(keyword.kwlist) - {'True', 'False', 'None', 'and', 'or', 'not'})
_KEYWORD_PATTERN = re.compile(r'\b(' + '|'.join(_MANGLED_KEYWORDS) + r')\b')
_MANGLED_PATTERN = re.compile(r'\b' + KEYWORD_PREFIX + r'(' + '|'.join(_MANGLED_KEYWORDS) + r')\b')


def mangle_keywords(text: str) -> str:
    """Benennt Python-Schlüsselwörter als Bezeichner um (lambda -> _kw_lambda)."""
    return _KEYWORD_PATTERN.sub(lambda m: KEYWORD_PREFIX + m.group(1), text)


def unmangle(text: str) -> str:
    """Macht mangle_keywords() rückgängig (für Anzeige von Namen und Gleichungen)."""
    return _MANGLED_PATTERN.sub(lambda m: m.group(1), text)


def display_name(name: str) -> str:
    """Anzeigename einer Variable (_kw_lambda -> lambda)."""
    return unmangle(name)


def parse_vector(value_str: str) -> Union[np.ndarray, None]:
    """
    Parst einen Vektor-String im MATLAB-Stil.

    Syntax:
        start:step:end  -> numpy array von start bis end mit Schrittweite step
        start:end       -> numpy array von start bis end mit Schrittweite 1

    Returns:
        numpy array oder None wenn kein Vektor-Format
    """
    value_str = value_str.strip()

    # Werteliste [v1 v2 ...] (Messdaten, per Zwischenablage eingefügt)
    if value_str.startswith('[') and value_str.endswith(']'):
        return _parse_value_list(value_str[1:-1])

    # Prüfe auf start:step:end Format
    match3 = VECTOR_PATTERN_3.match(value_str)
    if match3:
        start = float(match3.group(1))
        step = float(match3.group(2))
        end = float(match3.group(3))
        if step == 0:
            return None
        return _build_vector(start, step, end)

    # Prüfe auf start:end Format (step=1)
    match2 = VECTOR_PATTERN_2.match(value_str)
    if match2:
        start = float(match2.group(1))
        end = float(match2.group(2))
        step = 1.0 if end >= start else -1.0
        return _build_vector(start, step, end)

    return None


class VectorSyntaxError(Exception):
    """Werteliste [ ... ] ist fehlerhaft (z.B. Dezimalkomma, keine Zahl)."""


def _parse_value_list(inner: str) -> np.ndarray:
    """
    Werteliste wie "1.5 2.0 2.5" oder "1.5; 2.0; 2.5" oder "1.5, 2.0" (auch mit
    Zeilenumbrüchen/Tabulatoren, wie aus Excel/CSV kopiert). Dezimalzeichen ist
    der Punkt; ein Dezimalkomma ("1,5 2,0") wird erkannt und gemeldet.
    """
    tokens = inner.split()
    if len(tokens) > 1 and any(re.fullmatch(r'-?\d+,\d+', t) for t in tokens):
        raise VectorSyntaxError("Dezimalkomma in der Werteliste? Dezimalzahlen mit Punkt schreiben "
                                "(1.5 statt 1,5); Werte durch Leerzeichen, ';' oder ',' trennen")
    values = []
    for token in re.split(r'[\s,;]+', inner.strip()):
        if not token:
            continue
        try:
            values.append(float(token))
        except ValueError:
            raise VectorSyntaxError(f"'{token}' ist keine Zahl (Werteliste [ ... ])") from None
    if not values:
        raise VectorSyntaxError("Leere Werteliste [ ]")
    return np.array(values, dtype=float)


def _build_vector(start: float, step: float, end: float) -> Union[np.ndarray, None]:
    """
    Erzeugt einen Vektor mit MATLAB-Semantik: start, start+step, ...
    Der Endwert ist nur enthalten, wenn er exakt auf dem Raster liegt
    (0:0.3:1 -> 0, 0.3, 0.6, 0.9 - die Schrittweite wird nie verfälscht).
    """
    n_steps_exact = (end - start) / step
    if n_steps_exact < -1e-9:
        return None  # Schrittweite zeigt vom Endwert weg
    # Toleranz gegen Float-Rundung: liegt end (fast) exakt auf dem Raster?
    n_rounded = round(n_steps_exact)
    if abs(n_steps_exact - n_rounded) < 1e-9 * max(1.0, abs(n_steps_exact)):
        n_steps = int(n_rounded)
    else:
        n_steps = int(np.floor(n_steps_exact))
    return start + step * np.arange(n_steps + 1)


def is_vector_assignment(line: str, parse_units: bool = False) -> Tuple[bool, str, str, str]:
    """
    Prüft ob eine Zeile eine Vektor-Zuweisung ist.

    Args:
        line: Die zu prüfende Zeile
        parse_units: Ob Einheiten geparst werden sollen

    Returns:
        (is_vector, var_name, vector_string, unit_string)
        unit_string ist leer wenn keine Einheit gefunden wurde
    """
    if '=' not in line or (':' not in line and '[' not in line):
        return False, '', '', ''

    parts = line.split('=', 1)
    if len(parts) != 2:
        return False, '', '', ''

    left = parts[0].strip()
    right = parts[1].strip()

    # Links muss eine einfache Variable sein
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', left):
        return False, '', '', ''

    # Werteliste: x = [v1 v2 ...] [Einheit]
    list_match = re.match(r'^(\[[^\[\]]*\])\s*(.*)$', right, re.DOTALL)
    if list_match:
        unit_part = list_match.group(2).strip() if parse_units else ''
        return True, left, list_match.group(1), unit_part

    # Rechts muss Vektor-Syntax sein
    if parse_vector(right) is not None:
        return True, left, right, ''

    # Wenn Einheiten aktiviert: Versuche Einheit vom Ende abzutrennen
    if parse_units and UNITS_AVAILABLE:
        # Versuche verschiedene Trennungen: "100:20:200 °C" oder "100:20:200°C"
        # Suche nach dem letzten Zahlenwert im Vektor-Teil
        vec_unit_match = re.match(r'^(-?\d+\.?\d*):(-?\d+\.?\d*):(-?\d+\.?\d*)\s*(.+)$', right)
        if vec_unit_match:
            vec_part = f"{vec_unit_match.group(1)}:{vec_unit_match.group(2)}:{vec_unit_match.group(3)}"
            unit_part = vec_unit_match.group(4).strip()
            if parse_vector(vec_part) is not None:
                return True, left, vec_part, unit_part

        # Auch start:end Format mit Einheit
        vec_unit_match2 = re.match(r'^(-?\d+\.?\d*):(-?\d+\.?\d*)\s*(.+)$', right)
        if vec_unit_match2:
            vec_part = f"{vec_unit_match2.group(1)}:{vec_unit_match2.group(2)}"
            unit_part = vec_unit_match2.group(3).strip()
            if parse_vector(vec_part) is not None:
                return True, left, vec_part, unit_part

    return False, '', '', ''


def remove_comments(text: str) -> str:
    """Entfernt Kommentare aus dem Text.

    EES-Kommentare:
    - "..." (Anführungszeichen)
    - {...} (geschweifte Klammern)

    Zeilenumbrüche INNERHALB von Kommentaren bleiben erhalten, damit die
    Zeilennummern mit dem Originaltext übereinstimmen (sonst verrutscht nach
    einem mehrzeiligen Kommentar die Zuordnung Gleichung -> Originalzeile).
    """
    # Entferne "..." Kommentare (Zeilenumbrüche behalten)
    text = re.sub(r'"[^"]*"', lambda m: '\n' * m.group(0).count('\n'), text)
    # Entferne {...} Kommentare (auch verschachtelt, per Klammer-Zählung)
    result = []
    depth = 0
    unmatched_start = None  # Position des ersten unbalancierten '{'
    unmatched_result_len = 0  # Länge von result beim ersten unbalancierten '{'
    for i, char in enumerate(text):
        if char == '{':
            if depth == 0:
                unmatched_start = i
                unmatched_result_len = len(result)
            depth += 1
        elif char == '}':
            if depth > 0:
                depth -= 1
                if depth == 0:
                    unmatched_start = None
        elif depth == 0 or char == '\n':
            result.append(char)
    if depth > 0 and unmatched_start is not None:
        # Unbalancierter Kommentar: Text ab dem offenen '{' unverändert lassen
        result = result[:unmatched_result_len]
        result.append(text[unmatched_start:])
    return ''.join(result)


# Erwartete Einheit (Dimension) der Argumente von Stoffwert-/HumidAir-Funktionen
ARG_EXPECTED_UNITS = {
    't': 'K', 'p': 'Pa', 'p_tot': 'Pa', 'p_w': 'Pa',
    'h': 'J/kg', 'u': 'J/kg', 's': 'J/(kg*K)',
    'rho': 'kg/m^3', 'd': 'kg/m^3', 'v': 'm^3/kg',
    'x': 'dimensionless', 'rh': 'dimensionless', 'rf': 'dimensionless', 'w': 'dimensionless',
}

# Erwartete Einheiten der Positionsargumente der Strahlungsfunktionen (nach T)
RADIATION_ARG_UNITS = {
    'eb': ('K', 'm'), 'blackbody': ('K', 'm', 'm'), 'blackbody_cumulative': ('K', 'm'),
    'wien': ('K',), 'stefan_boltzmann': ('K',),
}


def _convert_arg_units(arg: str) -> str:
    """
    Converts units in a function argument to SI base units.

    Examples:
        'T=25°C' -> 'T=298.15'
        'p_tot=1bar' -> 'p_tot=100000'
        'T=T_1' -> 'T=T_1' (variable, unchanged)
        'rh=0.5' -> 'rh=0.5' (no unit)
    """
    if '=' not in arg:
        return arg

    key, value = arg.split('=', 1)
    key = key.strip()
    value = value.strip()

    # Check if value is a variable (starts with letter/underscore, no digits after unit patterns)
    if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', value):
        return arg  # Variable, keep unchanged

    # Try to parse as value with unit
    if UNITS_AVAILABLE:
        try:
            magnitude, unit_str = parse_value_with_unit(value)
        except ValueError:
            return arg  # Kein "Zahl + Einheit" (z.B. Ausdruck) -> unverändert
        if unit_str:
            # Unbekannte Einheit -> UnknownUnitError (nicht still dimensionslos)
            unit_value = unit_value_strict(magnitude, unit_str, arg)
            expected = ARG_EXPECTED_UNITS.get(key.lower())
            if expected:
                check_unit_dimension(unit_value, expected, arg)
            # Use SI base value for calculations
            return f"{key}={unit_value.si_value}"

    return arg


def _split_call_args(args_str: str) -> List[str]:
    """Teilt einen Argument-String an Kommas auf Klammertiefe 0."""
    args = []
    current_arg = ""
    paren_depth = 0

    for char in args_str:
        if char == '(':
            paren_depth += 1
            current_arg += char
        elif char == ')':
            paren_depth -= 1
            current_arg += char
        elif char == ',' and paren_depth == 0:
            args.append(current_arg.strip())
            current_arg = ""
        else:
            current_arg += char

    if current_arg.strip():
        args.append(current_arg.strip())

    return args


def _convert_call_parts(func_name: str, args_str: str, keep_case: bool = False) -> str:
    """
    Konvertiert einen Thermodynamik-/HumidAir-Aufruf von EES zu Python-Syntax:
    erstes Argument (Stoffname bzw. Output-Eigenschaft) wird gequotet,
    Einheiten in key=value Argumenten werden zu SI konvertiert.

    EES:    enthalpy(water, T=100°C, p=1bar)
    Python: enthalpy('water', T=373.15, p=100000)

    EES:    HumidAir(h, T=25°C, rh=0.5, p_tot=1bar)
    Python: HumidAir('h', T=298.15, rh=0.5, p_tot=100000)
    """
    args = _split_call_args(args_str)
    if len(args) < 1:
        return f"{func_name}({args_str})"

    # Erstes Argument (Stoffname/Output-Eigenschaft) in Anführungszeichen setzen
    first = args[0]
    if not (first.startswith("'") or first.startswith('"')):
        first = f"'{first}'"

    # Restliche Argumente (key=value Paare) - konvertiere Einheiten zu SI
    rest_args = [_convert_arg_units(arg) for arg in args[1:]]

    name = func_name if keep_case else func_name.lower()
    new_args = [first] + rest_args
    return f"{name}({', '.join(new_args)})"


def _iter_call_spans(text: str, func_names_lower: Set[str]):
    """
    Findet Aufrufe func(...) mit BALANCIERTEN Klammern (auch verschachtelt).

    Yields:
        (start, open_idx, close_idx, func_name) - Indizes im Text
    """
    for m in re.finditer(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', text):
        if m.group(1).lower() not in func_names_lower:
            continue
        open_idx = m.end() - 1
        depth = 0
        close_idx = -1
        for i in range(open_idx, len(text)):
            if text[i] == '(':
                depth += 1
            elif text[i] == ')':
                depth -= 1
                if depth == 0:
                    close_idx = i
                    break
        if close_idx < 0:
            continue  # Unbalanciert - überspringen
        yield m.start(), open_idx, close_idx, m.group(1)


def _convert_positional_call(func_name: str, args_str: str, keep_case: bool = True) -> str:
    """
    Konvertiert Einheiten in POSITIONSargumenten zu SI (Strahlungsfunktionen):
    Eb(500°C, 5µm) -> Eb(773.15, 5e-06). Argumente ohne Einheit bleiben unverändert.
    """
    args = []
    expected_units = RADIATION_ARG_UNITS.get(func_name.lower(), ())
    for index, arg in enumerate(_split_call_args(args_str)):
        converted = arg
        if (index < len(expected_units) and expected_units[index] == 'm'
                and re.fullmatch(r'\s*[+]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?\s*', arg)):
            # Zahlenliteral als Wellenlänge: >= 0.01 als µm ("Eb(1000, 5)"), kleinere
            # als m. Im Solver sind Wellenlängen danach immer SI (keine Heuristik)
            value = float(arg)
            converted = repr(value / 1e6 if value >= 0.01 else value)
        elif UNITS_AVAILABLE and '=' not in arg:
            try:
                magnitude, unit_str = parse_value_with_unit(arg)
            except ValueError:
                magnitude, unit_str = None, ''
            if unit_str:
                unit_value = unit_value_strict(magnitude, unit_str, arg)
                if index < len(expected_units):
                    check_unit_dimension(unit_value, expected_units[index], f"{func_name}({args_str})")
                converted = repr(unit_value.calc_value)
        args.append(converted)
    return f"{func_name}({', '.join(args)})"


def _replace_calls_balanced(text: str, func_names: Set[str], keep_case: bool = False,
                            converter=None) -> str:
    """
    Ersetzt alle func(...)-Aufrufe (auch verschachtelte) via _convert_call_parts
    (bzw. den übergebenen converter). Verschachtelte Aufrufe in den Argumenten
    werden zuerst konvertiert.
    """
    if converter is None:
        converter = _convert_call_parts
    names_lower = {n.lower() for n in func_names}

    # Wiederhole bis stabil: pro Durchlauf wird der erste Aufruf konvertiert,
    # dessen Konvertierung den Text tatsächlich ändert (idempotent -> terminiert).
    for _ in range(100):  # Sicherheitslimit
        changed = False
        for start, open_idx, close_idx, name in _iter_call_spans(text, names_lower):
            args_str = text[open_idx + 1:close_idx]
            # Innere Aufrufe in den Argumenten zuerst konvertieren
            args_converted = _replace_calls_balanced(args_str, func_names, keep_case, converter) \
                if any(f in args_str.lower() for f in names_lower) else args_str
            converted = converter(name, args_converted, keep_case=keep_case)
            if converted != text[start:close_idx + 1]:
                text = text[:start] + converted + text[close_idx + 1:]
                changed = True
                break
        if not changed:
            break
    return text


def convert_thermo_call(match) -> str:
    """Regex-Wrapper (Kompatibilität): konvertiert einen Thermodynamik-Aufruf."""
    return _convert_call_parts(match.group(1), match.group(2), keep_case=False)


def convert_humid_air_call(match) -> str:
    """Regex-Wrapper (Kompatibilität): konvertiert einen HumidAir-Aufruf."""
    return _convert_call_parts(match.group(1), match.group(2), keep_case=True)


def tokenize_equation(equation: str) -> str:
    """Konvertiert EES-Syntax zu Python-Syntax."""
    # Ersetze ^ durch **
    equation = equation.replace('^', '**')

    # Ersetze ln durch log (numpy), lg durch log10 (nur als Funktionsaufruf)
    equation = re.sub(r'\bln\b', 'log', equation)
    equation = re.sub(r'\blg(?=\s*\()', 'log10', equation)

    # Malpunkt als Multiplikation (0.475·10^-6)
    equation = equation.replace('·', '*').replace('⋅', '*')

    # Ersetze log10
    equation = re.sub(r'\blog10\b', 'log10', equation)

    # Konvertiere Thermodynamik-Funktionsaufrufe (balanciert, auch verschachtelt)
    equation = _replace_calls_balanced(equation, THERMO_FUNCTIONS, keep_case=False)

    # Konvertiere FeuchteLuft-Funktionsaufrufe
    equation = _replace_calls_balanced(equation, HUMID_AIR_FUNCTIONS, keep_case=True)

    # Strahlungsfunktionen: Einheiten in Positionsargumenten (Eb(500°C, 5µm))
    equation = _replace_calls_balanced(equation, RADIATION_FUNCTIONS, keep_case=True,
                                       converter=_convert_positional_call)

    return equation


def _reduce_special_calls_to_tokens(equation: str) -> str:
    """
    Ersetzt Thermodynamik-/HumidAir-Aufrufe durch die Variablen-Tokens ihrer
    Argument-WERTE (kwarg-Keys, Stoffnamen und Zahlen fallen weg).
    Verschachtelte Aufrufe werden von innen nach außen aufgelöst.

    Beispiel: "enthalpy(water, T=temperature(water, p=p1, s=s1), p=p2)" -> " p1 s1 p2 "
    """
    special_funcs = {f.lower() for f in (THERMO_FUNCTIONS | HUMID_AIR_FUNCTIONS)}

    for _ in range(100):  # Sicherheitslimit gegen Endlosschleifen
        # Suche einen INNERSTEN Aufruf (Argumente ohne weitere Spezial-Aufrufe)
        replaced = False
        for start, open_idx, close_idx, _name in _iter_call_spans(equation, special_funcs):
            args_str = equation[open_idx + 1:close_idx]
            if any(re.search(rf'\b{f}\s*\(', args_str, flags=re.IGNORECASE) for f in special_funcs):
                continue  # Enthält inneren Aufruf - der wird zuerst verarbeitet
            tokens = []
            for arg in _split_call_args(args_str)[1:]:  # erstes Argument (Stoff/Property) fällt weg
                value = arg.split('=', 1)[1] if '=' in arg else arg
                tokens.extend(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', value))
            equation = equation[:start] + ' ' + ' '.join(tokens) + ' ' + equation[close_idx + 1:]
            replaced = True
            break
        if not replaced:
            break

    return equation


def extract_variables(equation: str) -> Set[str]:
    """Extrahiert alle Variablennamen aus einer Gleichung."""
    # Ersetze komplette Thermodynamik-/HumidAir-Funktionsaufrufe durch die
    # Variablen-Tokens ihrer Argumente (balanciert, auch verschachtelt)
    temp_eq = _reduce_special_calls_to_tokens(equation)

    # Entferne Funktionsnamen aus der Suche - NUR wenn sie als Funktionen verwendet werden
    # (d.h. mit Klammern dahinter), nicht wenn sie als Variablen verwendet werden
    for func in MATH_FUNCTIONS:
        # Entferne nur func(...) Aufrufe, nicht alleinstehende func
        temp_eq = re.sub(rf'\b{func}\s*\(', '(', temp_eq)

    for func in THERMO_FUNCTIONS:
        # Entferne nur func(...) Aufrufe, nicht alleinstehende func
        temp_eq = re.sub(rf'\b{func}\s*\(', '(', temp_eq, flags=re.IGNORECASE)

    for func in RADIATION_FUNCTIONS:
        # Entferne nur func(...) Aufrufe, nicht alleinstehende func
        temp_eq = re.sub(rf'\b{func}\s*\(', '(', temp_eq, flags=re.IGNORECASE)

    for func in HUMID_AIR_FUNCTIONS:
        # Entferne nur func(...) Aufrufe, nicht alleinstehende func
        temp_eq = re.sub(rf'\b{func}\s*\(', '(', temp_eq, flags=re.IGNORECASE)

    # Finde alle Bezeichner (Variablen)
    # Variablen können Buchstaben, Zahlen und Unterstriche enthalten
    # aber nicht mit einer Zahl beginnen
    variables = set(re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', temp_eq))

    # Entferne Python-Keywords und mathematische Konstanten
    # NICHT die Funktionsnamen entfernen - sie können als Variablen verwendet werden
    # (z.B. cp = cv + R). Die Funktionsaufrufe wurden bereits oben aus temp_eq entfernt.
    # Aber pi und e sind Konstanten (keine Funktionen), daher hier filtern.
    python_keywords = {'and', 'or', 'not', 'True', 'False', 'None', 'log', 'log10'}
    math_constants = {'pi'}  # Mathematische Konstanten (keine Funktionen), e bleibt Variable

    variables -= python_keywords
    variables -= math_constants

    return variables


def _get_const_eval_context() -> dict:
    """
    Eval-Kontext für rein numerische Konstanten-Ausdrücke.
    Trigonometrie in GRAD (wie EES) - identisch für Pass 1 und Pass 2.
    """
    def _sin(x): return np.sin(np.radians(x))
    def _cos(x): return np.cos(np.radians(x))
    def _tan(x): return np.tan(np.radians(x))
    def _asin(x): return np.degrees(np.arcsin(x))
    def _acos(x): return np.degrees(np.arccos(x))
    def _atan(x): return np.degrees(np.arctan(x))

    return {
        'pi': np.pi, 'e': np.e,
        'sin': _sin, 'cos': _cos, 'tan': _tan,
        'asin': _asin, 'acos': _acos, 'atan': _atan,
        'sqrt': np.sqrt, 'log': np.log, 'log10': np.log10,
        'exp': np.exp, 'abs': abs,
        'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
        'max': max, 'min': min,
    }


def _eval_constant_expression(expr: str) -> Optional[float]:
    """
    Wertet einen rein numerischen Ausdruck aus (ohne Variablen).

    Args:
        expr: Ausdruck in Python-Syntax (bereits tokenisiert: ** statt ^, log statt ln)

    Returns:
        float-Wert oder None, wenn der Ausdruck nicht auswertbar ist
    """
    try:
        value = eval(expr, {"__builtins__": {}}, _get_const_eval_context())
        # Nur echte Zahlen - kein bool (aus Vergleichen wie "3 > 2"), keine Listen/Tupel
        if (isinstance(value, (int, float, np.floating)) and not isinstance(value, (bool, np.bool_))
                and np.isfinite(value)):
            return float(value)
    except Exception:
        pass
    return None


def _split_expression_with_unit(right: str) -> Optional[Tuple[float, str]]:
    """
    Trennt 'numerischer Ausdruck + Einheit', z.B. '10000/3600 kg/s'.

    Die Einheit muss als letztes, durch Leerzeichen getrenntes Token stehen
    und mit einem Buchstaben (oder °/µ) beginnen; der Rest muss ein rein
    numerischer Ausdruck sein. So wird 'x + 2' oder '2 * m' nie als
    Wert+Einheit fehlinterpretiert.

    Returns:
        (wert, einheit) oder None
    """
    if not UNITS_AVAILABLE:
        return None

    tokens = right.split()
    if len(tokens) < 2:
        return None

    unit_part = tokens[-1]
    expr_part = ' '.join(tokens[:-1]).strip()

    # Einheit muss mit Buchstabe/°/µ (oder 1/ für Kehrwerte) beginnen und darf kein Operator-Rest sein
    if not re.match(r'^(?:[a-zA-Z°µ]|1/)', unit_part):
        return None
    # Ausdruck muss mindestens eine Ziffer enthalten
    if not re.search(r'\d', expr_part):
        return None

    # Einheit validieren (mit Dummy-Wert 1)
    try:
        _, unit_str = parse_value_with_unit(f"1 {unit_part}")
    except (ValueError, Exception):
        return None
    if not unit_str:
        return None

    # Ausdruck muss rein numerisch auswertbar sein
    expr_python = re.sub(r'\bln\b', 'log', expr_part.replace('^', '**'))
    expr_python = re.sub(r'\blg(?=\s*\()', 'log10', expr_python).replace('·', '*').replace('⋅', '*')
    value = _eval_constant_expression(expr_python)
    if value is None:
        return None

    return value, unit_str


class EquationSyntaxError(Exception):
    """Gleichung ist syntaktisch nicht auswertbar (z.B. fehlendes '*')."""


# Namen, die in Gleichungen als Funktion aufgerufen werden dürfen
# (wie im Auswertungskontext des Solvers; Strahlungsfunktionen in beiden Schreibweisen)
CALLABLE_NAMES = (
    (MATH_FUNCTIONS - {'pi', 'ln'}) | {'log', 'log10'} | THERMO_FUNCTIONS
    | {'HumidAir', 'humidair'}
    | {'Eb', 'eb', 'Blackbody', 'blackbody', 'Blackbody_cumulative', 'blackbody_cumulative',
       'Wien', 'wien', 'Stefan_Boltzmann', 'stefan_boltzmann'}
)


def _python_to_display(text: str) -> str:
    """Python-Form einer Gleichung zurück in Eingabe-Schreibweise (für Meldungen)."""
    text = text.replace('**', '^')
    text = re.sub(r'\blog\(', 'ln(', text)
    text = re.sub(r"'([A-Za-z0-9_()]+)'", r'\1', text)  # 'water' -> water
    return unmangle(text)


def _excerpt(equation: str, offset: int) -> str:
    """Textausschnitt um eine Fundstelle (offset 0-basiert) in Eingabe-Schreibweise."""
    start = max(0, offset - 18)
    end = min(len(equation), offset + 18)
    # Nicht mitten in einem Bezeichner schneiden (z.B. "lo…" statt "ln(")
    while start > 0 and (equation[start - 1].isalnum() or equation[start - 1] == '_'):
        start -= 1
    while end < len(equation) and (equation[end].isalnum() or equation[end] == '_'):
        end += 1
    if equation[end:end + 1] == '(':
        end += 1
    before = _python_to_display(equation[start:offset])
    after = _python_to_display(equation[offset:end])
    prefix = '…' if start > 0 else ''
    suffix = '…' if end < len(equation) else ''
    return f"{prefix}{before}▶{after}{suffix}"


# Anzahl der Positionsargumente der Funktionen (Signaturprüfung beim Einlesen)
_ONE_ARGUMENT = {'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh',
                 'exp', 'log', 'log10', 'sqrt', 'abs'}
_RADIATION_ARGS = {'eb': 2, 'blackbody': 3, 'blackbody_cumulative': 2, 'wien': 1, 'stefan_boltzmann': 1}


def _known_fluid(name: str) -> bool:
    """Ist der Name ein CoolProp-Fluid oder ein Kurzname (Groß-/Kleinschreibung egal)?"""
    try:
        from thermodynamics import FLUID_ALIASES, get_available_fluids
    except ImportError:
        return True
    if name.lower() in FLUID_ALIASES or any(c in name for c in ':&['):
        return True  # Kurzname bzw. CoolProp-Spezialsyntax (INCOMP::, Gemische)
    return name.lower() in {f.lower() for f in get_available_fluids()}


def _check_call_signature(node, func_name: str, text: str, offset: int) -> None:
    """
    Prüft einen Funktionsaufruf gegen die Signatur der Funktion (generisch für
    alle Funktionen): Anzahl der Argumente, Fluid- und Eigenschaftsnamen,
    Parameternamen. Fehler werden sonst erst beim Lösen sichtbar - oder gar
    nicht ("keine numerische Lösung").
    """
    shown = func_name
    where = f"bei: {_excerpt(text, offset)}"
    n_args, keywords = len(node.args), [k.arg for k in node.keywords]
    lower = func_name.lower()

    if func_name in _ONE_ARGUMENT:
        if n_args != 1 or keywords:
            raise EquationSyntaxError(f"{_python_to_display(shown + '(')[:-1]}() erwartet genau 1 Argument, {where}")
        return
    if func_name in ('max', 'min'):
        if n_args < 2 or keywords:
            raise EquationSyntaxError(f"{func_name}() erwartet mindestens 2 Argumente, {where}")
        return
    if lower in _RADIATION_ARGS:
        if n_args != _RADIATION_ARGS[lower] or keywords:
            raise EquationSyntaxError(
                f"{func_name}() erwartet {_RADIATION_ARGS[lower]} Argument(e), gegeben {n_args}, {where}")
        return

    first = node.args[0] if node.args else None
    first_text = first.value if isinstance(first, ast_module().Constant) and isinstance(first.value, str) else None

    if lower in THERMO_FUNCTIONS:
        from thermodynamics import INPUT_MAP as THERMO_INPUTS
        if n_args != 1 or first_text is None:
            raise EquationSyntaxError(f"{lower}(fluid, ...) braucht als erstes Argument den Stoff, {where}")
        if not _known_fluid(first_text):
            raise EquationSyntaxError(
                f"Unbekanntes Fluid '{first_text}' (siehe Help > Fluid List), {where}")
        bad = [k for k in keywords if k is None or k.lower() not in THERMO_INPUTS]
        if bad:
            raise EquationSyntaxError(
                f"{lower}(): unbekannter Parameter '{bad[0]}' (gültig: T, p, h, s, x, rho, d, u, v), {where}")
        if len(keywords) != 2:
            raise EquationSyntaxError(
                f"{lower}() braucht genau 2 Zustandsgrößen (z.B. T=..., p=...), gegeben {len(keywords)}, {where}")
        return

    if lower == 'humidair':
        from humid_air import OUTPUT_MAP as HA_OUTPUTS, INPUT_MAP as HA_INPUTS
        if n_args != 1 or first_text is None or first_text.lower() not in HA_OUTPUTS:
            raise EquationSyntaxError(
                f"HumidAir: unbekannte Eigenschaft '{first_text}' (gültig: "
                f"{', '.join(HA_OUTPUTS)}), {where}")
        bad = [k for k in keywords if k is None or k.lower() not in HA_INPUTS]
        if bad:
            raise EquationSyntaxError(
                f"HumidAir(): unbekannter Parameter '{bad[0]}' (gültig: {', '.join(HA_INPUTS)}), {where}")
        if len(keywords) != 3:
            raise EquationSyntaxError(
                f"HumidAir() braucht genau 3 Zustandsgrößen (z.B. T=..., rh=..., p_tot=...), "
                f"gegeben {len(keywords)}, {where}")


def ast_module():
    import ast
    return ast


def _allowed_nodes():
    import ast
    return (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load, ast.Constant,
            ast.keyword, ast.Tuple,  # Tuple: eigene Meldung (Dezimalkomma)
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv,
            ast.USub, ast.UAdd)


_ALLOWED_NODES = _allowed_nodes()
_UNSUPPORTED_NODES = {
    'List': "Wertelisten [..] nur als eigene Zeile 'x = [v1 v2 ...] Einheit' (Parameterstudie)",
    'Subscript': "Indizes wie x[1] werden nicht unterstützt",
    'Attribute': "Punkt-Zugriff (a.b) ist nicht erlaubt - Dezimalzahlen ohne Ziffer vor dem Punkt? (0.5 statt .5)",
    'Compare': "Vergleiche (<, >, ==) werden nicht unterstützt (keine Fallunterscheidung)",
    'BoolOp': "Logische Verknüpfungen (and/or) werden nicht unterstützt",
    'IfExp': "Fallunterscheidungen (if/else) werden nicht unterstützt",
    'Dict': "Ausdruck { } wird nicht unterstützt - Kommentare in {...} müssen geschlossen sein",
    'Set': "Ausdruck { } wird nicht unterstützt - Kommentare in {...} müssen geschlossen sein",
    'Lambda': "lambda als Funktion ist nicht erlaubt",
}


def _check_equation_syntax(left: str, right: str) -> None:
    """
    Prüft beide Seiten einer Gleichung (Python-Syntax) VOR dem Lösen - generisch
    über den Syntaxbaum, ohne Annahmen über die Form einzelner Gleichungen:
    - nicht parsebar (Syntaxfehler) -> Meldung mit Fundstelle
    - Aufruf von etwas, das keine Funktion ist (Variable, Zahl, Klammer-
      ausdruck vor '(') -> Meldung mit Fundstelle
    Die Zeilennummer ergänzt parse_equations.
    """
    import ast
    for side in (left, right):
        try:
            tree = ast.parse(side.strip(), mode='eval')
        except SyntaxError as exc:
            text = side.strip()
            offset = min(len(text), max(0, (exc.offset or 1) - 1))
            hint = (" - Dezimalkomma? Dezimalzahlen mit Punkt schreiben (0.71 statt 0,71)"
                    if re.search(r'\d,\d', text) else "")
            raise EquationSyntaxError(
                f"Syntaxfehler ({exc.msg}) bei: {_excerpt(text, offset)}{hint}") from None

        text = side.strip()
        encoded = text.encode('utf-8')

        def position(node, end=False):
            offset = node.end_col_offset if end else node.col_offset
            return len(encoded[:offset].decode('utf-8', errors='ignore'))

        for node in ast.walk(tree):
            # Positivliste der erlaubten Ausdrucksformen: alles andere ist in der
            # Gleichungssprache nicht vorgesehen (und wird so auch nie an eval übergeben)
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Constant):
                # Eckige Klammern direkt nach einer Zahl: kein Index möglich -
                # Einheit in EES-Schreibweise "1 [kg/s]"
                raise EquationSyntaxError(
                    f"Eckige Klammern nach einer Zahl - Einheiten ohne Klammern schreiben "
                    f"(1 kg/s statt 1 [kg/s]); Wertelisten als eigene Zeile x = [1 2 3] Einheit, "
                    f"bei: {_excerpt(text, position(node.slice) - 1)}")
            if not isinstance(node, _ALLOWED_NODES):
                raise EquationSyntaxError(
                    f"{_UNSUPPORTED_NODES.get(type(node).__name__, 'Nicht unterstützter Ausdruck')}, "
                    f"bei: {_excerpt(text, position(node))}")
            if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float, str)):
                raise EquationSyntaxError(f"Nicht unterstützter Wert, bei: {_excerpt(text, position(node))}")
            # Komma außerhalb eines Funktionsaufrufs: in Gleichungen nie sinnvoll
            # (meist Dezimalkomma: "0,71" ergibt sonst stillschweigend ein Tupel)
            if isinstance(node, ast.Tuple):
                raise EquationSyntaxError(
                    f"Komma außerhalb eines Funktionsaufrufs - Dezimalkomma? Dezimalzahlen mit "
                    f"Punkt schreiben (0.71 statt 0,71), bei: {_excerpt(text, position(node))}")
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name) and func.id in CALLABLE_NAMES:
                _check_call_signature(node, func.id, text, position(node))
                continue
            # Fundstelle: erste '(' nach dem aufgerufenen Ausdruck (AST-Offsets sind UTF-8-Bytes)
            func_end = len(encoded[:func.end_col_offset].decode('utf-8', errors='ignore'))
            paren = text.find('(', func_end)
            func_text = ast.get_source_segment(text, func) or ''
            name = unmangle(func.id) if isinstance(func, ast.Name) else _python_to_display(func_text)
            raise EquationSyntaxError(
                f"'{name}' ist keine Funktion - vor '(' fehlt ein Operator oder der Funktionsname "
                f"ist falsch, bei: {_excerpt(text, paren if paren >= 0 else func_end)}")


def _join_bracket_lines(lines: List[str]) -> List[str]:
    """
    Fügt eine über mehrere Zeilen gehende Werteliste "x = [ ... ]" zu einer
    Zeile zusammen (z.B. eine aus Excel eingefügte Spalte). Die verbrauchten
    Zeilen werden zu Leerzeilen, damit die Zeilennummern erhalten bleiben.
    """
    result = list(lines)
    i = 0
    while i < len(result):
        line = result[i]
        if line.count('[') > line.count(']') and '=' in line:
            j = i
            joined = line
            while joined.count('[') > joined.count(']') and j + 1 < len(result):
                j += 1
                joined += ' ' + result[j]
                result[j] = ''
            result[i] = joined
            i = j
        i += 1
    return result


def _mangle_line(line: str) -> str:
    """
    Wendet mangle_keywords() auf eine Eingabezeile an - aber nicht auf eine
    Einheit auf der rechten Seite ("d = 2 in" bleibt, "in" ist dort Zoll).
    """
    if '=' not in line:
        return mangle_keywords(line)
    left, right = line.split('=', 1)
    right_stripped = right.strip()
    keep_right = False
    if UNITS_AVAILABLE and right_stripped:
        try:
            _, unit_str = parse_value_with_unit(right_stripped)
            keep_right = bool(unit_str)
        except ValueError:
            pass
        if not keep_right:
            keep_right = (_split_expression_with_unit(right_stripped) is not None or
                          is_vector_assignment(line, parse_units=True)[0])
    return mangle_keywords(left) + '=' + (right if keep_right else mangle_keywords(right))


def parse_equations(text: str, parse_units: bool = True) -> Tuple[List[str], Set[str], dict, dict, dict, dict]:
    """
    Parst den Eingabetext (siehe _parse_equations). Einlesefehler (Syntax,
    unbekannte/unpassende Einheit) werden mit der Zeilennummer ergänzt.
    """
    position = [0]
    try:
        return _parse_equations(text, parse_units, position)
    except (EquationSyntaxError, UnknownUnitError) as exc:
        raise type(exc)(f"Zeile {position[0]}: {exc}") from None


def _parse_equations(text: str, parse_units: bool, position: List[int]
                     ) -> Tuple[List[str], Set[str], dict, dict, dict, dict]:
    """
    Parst den Eingabetext und extrahiert Gleichungen und Variablen.

    Unterstützt Einheiten-Syntax: T = 15°C, m = 10g, h = 2500kJ/kg

    Args:
        text: Der zu parsende Text
        parse_units: Wenn True, werden Einheiten erkannt und verarbeitet

    Returns:
        equations: Liste von Gleichungen in Python-Syntax (als f(x) = 0 Form)
        variables: Set aller gefundenen Variablen (ohne Sweep-Variable)
        initial_values: Dict mit vorgegebenen Werten
        sweep_vars: Dict mit Vektor-Variablen {name: numpy.array}
        original_equations: Dict Mapping parsed -> original für Anzeige
        unit_values: Dict mit Einheiten-Informationen {var_name: UnitValue}
    """
    # Speichere Original-Text vor Kommentar-Entfernung für Mapping
    original_text = text

    # Entferne Kommentare (Zeilenumbrüche bleiben erhalten -> Zeilen bleiben synchron)
    text = remove_comments(text)

    # Teile in Zeilen auf; Python-Schlüsselwörter als Variablennamen intern
    # umbenennen (lambda -> _kw_lambda), auch in den Originalzeilen, damit die
    # Einheiten-Analyse dieselben Namen sieht. Anzeige: display_name()/unmangle()
    lines = [_mangle_line(line) for line in _join_bracket_lines(text.split('\n'))]
    original_lines = [mangle_keywords(line) for line in original_text.split('\n')]

    equations = []
    all_variables = set()
    initial_values = {}
    sweep_vars = {}  # Vektor-Variablen für Parameterstudien
    original_equations = {}  # Mapping: parsed -> original
    unit_values = {}  # Einheiten-Informationen für Variablen

    # ZWEI-PASS-ANSATZ: Erst alle Konstanten identifizieren, dann Gleichungen verarbeiten
    # Dies ist notwendig, da Konstanten nach Gleichungen definiert sein können
    # z.B. "RWZ = (T-T0)/(T1-T0)" gefolgt von "RWZ = 0.75"

    # Pass 1: Identifiziere alle Konstanten (direkte Zuweisungen)
    pre_constants = set()
    for line_index, line in enumerate(lines):
        position[0] = line_index + 1
        line = line.strip()
        if not line or '=' not in line:
            continue
        # Prüfe auf Vektor-Zuweisung (kein Konstant)
        is_vec, var_name, _, _ = is_vector_assignment(line, parse_units=parse_units)
        if is_vec:
            continue
        parts = line.split('=', 1)
        if len(parts) != 2:
            continue
        left = parts[0].strip()
        right = parts[1].strip()
        # Prüfe ob links eine einzelne Variable steht
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', left):
            continue
        var_name = left
        # Prüfe ob rechts eine Zahl, Zahl mit Einheit oder rein numerischer Ausdruck ist
        # Zahl mit Einheit
        if UNITS_AVAILABLE:
            try:
                magnitude, unit_str = parse_value_with_unit(right)
                if unit_str:
                    pre_constants.add(var_name)
                    continue
            except ValueError:
                pass
            # Numerischer Ausdruck mit Einheit (z.B. "10000/3600 kg/s")
            if _split_expression_with_unit(right) is not None:
                pre_constants.add(var_name)
                continue
        # Reine Zahl
        try:
            float(right)
            pre_constants.add(var_name)
            continue
        except ValueError:
            pass
        # Numerischer Ausdruck (ohne Variablen)
        # Gleicher Eval-Kontext wie Pass 2, damit beide Pässe konsistent erkennen
        right_tokenized = tokenize_equation(right)
        vars_in_right = extract_variables(right_tokenized)
        if not vars_in_right and _eval_constant_expression(right_tokenized) is not None:
            pre_constants.add(var_name)

    # Pass 2: Normale Verarbeitung
    for i, line in enumerate(lines):
        position[0] = i + 1
        line = line.strip()

        # Hole Original-Zeile (mit Kommentaren, falls vorhanden)
        original_line = original_lines[i].strip() if i < len(original_lines) else line

        # Überspringe leere Zeilen
        if not line:
            continue

        # Prüfe ob es eine Gleichung ist (enthält =)
        if '=' not in line:
            continue

        # Prüfe auf Vektor-Zuweisung (z.B. T = 0:10:100 oder T = 0:10:100 °C)
        is_vec, var_name, vec_str, vec_unit = is_vector_assignment(line, parse_units=parse_units)
        if is_vec:
            try:
                vec_array = parse_vector(vec_str)
            except VectorSyntaxError as exc:
                raise EquationSyntaxError(str(exc)) from None
            # Prüfe ob Vektor erfolgreich geparst wurde
            if vec_array is None:
                continue
            # Wenn Einheit angegeben: Konvertiere zu Standard-Berechnungseinheit
            if vec_unit and UNITS_AVAILABLE:
                try:
                    from units import UnitValue
                    # Temperaturdifferenz (dT..., delta...): Faktor, KEIN Offset -
                    # wie bei Einzelwerten (dT = 0:5:20 °C -> 0..20 K, nicht 273..293 K)
                    lower_name = var_name.lower()
                    diff_factor = {'K': 1.0, 'kelvin': 1.0, '°C': 1.0, 'C': 1.0, 'degC': 1.0,
                                   'celsius': 1.0, '°F': 5.0 / 9.0, 'degF': 5.0 / 9.0,
                                   'fahrenheit': 5.0 / 9.0}.get(vec_unit.strip())
                    if (lower_name.startswith('dt') or lower_name.startswith('delta')) and diff_factor:
                        sweep_vars[var_name] = vec_array * diff_factor
                        unit_values[var_name] = UnitValue.from_input(float(vec_array[0]) * diff_factor, 'delta_K')
                        continue
                    # Konvertiere IMMER elementweise: nur so werden Offset-Einheiten
                    # (°C -> K: +273.15) korrekt behandelt. Ein multiplikativer
                    # Faktor wäre bei 20:10:50 °C für alle Werte außer dem ersten falsch.
                    calc_values = np.array([
                        unit_value_strict(float(v), vec_unit, line).calc_value
                        for v in vec_array
                    ])
                    sweep_vars[var_name] = calc_values
                    # Speichere UnitValue für Anzeige (erster Wert)
                    unit_values[var_name] = UnitValue.from_input(float(vec_array[0]), vec_unit)
                except UnknownUnitError:
                    raise
                except Exception:
                    # Bei Fehler: verwende Original-Werte ohne Konvertierung
                    sweep_vars[var_name] = vec_array
            else:
                sweep_vars[var_name] = vec_array
            # Variable NICHT zu all_variables hinzufügen (wird separat behandelt)
            continue

        # Behandle == als Vergleich (falls jemand das schreibt)
        if '==' in line:
            line = line.replace('==', '=')

        # Teile bei = (nur das erste =)
        parts = line.split('=', 1)
        if len(parts) != 2:
            continue

        left = parts[0].strip()
        right = parts[1].strip()

        # FRÜHE PRÜFUNG: Ist links eine einzelne Variable und rechts ein Wert mit Einheit?
        # Dies muss VOR tokenize_equation passieren, da Einheiten sonst falsch geparst werden
        if UNITS_AVAILABLE and re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', left):
            try:
                from units import UnitValue
                magnitude, unit_str = None, ''
                try:
                    magnitude, unit_str = parse_value_with_unit(right)
                except ValueError:
                    pass
                if not unit_str:
                    # Numerischer Ausdruck mit Einheit (z.B. "10000/3600 kg/s")
                    expr_unit = _split_expression_with_unit(right)
                    if expr_unit is not None:
                        magnitude, unit_str = expr_unit
                if unit_str and magnitude is not None:
                    # Wert mit Einheit gefunden (z.B. "15°C", "10g", "10000/3600 kg/s")
                    var_name = left

                    # Spezialfall: Temperaturdifferenz (dT..., delta...)
                    # Temperatur-Einheiten werden als Differenz behandelt, nicht absolut
                    # (verhindert falsche Offset-Konvertierung, z.B. +273.15 bei °C)
                    var_lower = var_name.lower()
                    is_temp_diff_name = (var_lower.startswith('dt') or
                                         var_lower.startswith('delta'))
                    # Faktor Einheit -> delta_K: 1 K-Diff = 1 °C-Diff, 1 °F-Diff = 5/9 K
                    diff_factor = {
                        'K': 1.0, 'kelvin': 1.0,
                        '°C': 1.0, 'C': 1.0, 'degC': 1.0, 'celsius': 1.0,
                        '°F': 5.0 / 9.0, 'degF': 5.0 / 9.0, 'fahrenheit': 5.0 / 9.0,
                    }.get(unit_str.strip())

                    if is_temp_diff_name and diff_factor is not None:
                        # Temperaturdifferenz: keine Offset-Konvertierung
                        initial_values[var_name] = magnitude * diff_factor
                        if parse_units:
                            # Erstelle UnitValue mit delta_K als Differenz-Einheit
                            unit_values[var_name] = UnitValue.from_input(magnitude * diff_factor, 'delta_K')
                    else:
                        # Unbekannte Einheit -> UnknownUnitError (kein ValueError,
                        # wird also unten NICHT verschluckt)
                        unit_value = unit_value_strict(magnitude, unit_str, line)
                        # Verwende calc_value für Berechnungen (konvertiert zu Standard-Einheit)
                        # z.B. 10 kg/h → 0.00278 kg/s, aber 20°C bleibt 20°C
                        initial_values[var_name] = unit_value.calc_value

                        # Speichere Einheiten-Info nur wenn parse_units aktiviert
                        if parse_units:
                            unit_values[var_name] = unit_value
                    continue
            except ValueError:
                pass  # Kein gültiger Wert mit Einheit, normale Verarbeitung

        # Konvertiere zu Python-Syntax
        left = tokenize_equation(left)
        right = tokenize_equation(right)

        # Extrahiere Variablen
        vars_left = extract_variables(left)
        vars_right = extract_variables(right)

        # Prüfe ob es eine direkte Zuweisung ist (z.B. T1 = 300 oder m = 10000/3600)
        # WICHTIG: Links muss ein REINER Variablenname stehen (wie in Pass 1).
        # "x + 5 = 2", "sin(alpha) = 0.5" oder "x^2 = 9" sind GLEICHUNGEN,
        # keine Zuweisungen - sonst würde z.B. alpha = 0.5 statt 30 gesetzt!
        if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', left) and len(vars_right) == 0:
            var_name = left

            # Versuche als einfache Zahl
            try:
                value = float(right)
                initial_values[var_name] = value
                continue
            except ValueError:
                # Versuche als arithmetischen Ausdruck auszuwerten
                # (Trigonometrie in GRAD, gleicher Kontext wie Pass 1)
                value = _eval_constant_expression(right)
                if value is not None:
                    initial_values[var_name] = value
                    continue

        # Füge Variablen hinzu (nur wenn keine direkte Zuweisung)
        # WICHTIG: Entferne vor-identifizierte Konstanten (aus Pass 1)
        all_vars = vars_left | vars_right
        all_vars -= pre_constants  # Konstanten nie als Variablen zählen
        all_variables |= all_vars

        # Erstelle Gleichung in der Form: left - right = 0
        equation = f"({left}) - ({right})"
        _check_equation_syntax(left, right)
        equations.append(equation)

        # Speichere Original-Zeile für Anzeige
        original_equations[equation] = original_line

    # Entferne Sweep-Variablen aus der Variablenliste (sie sind keine Unbekannten)
    all_variables -= set(sweep_vars.keys())

    # Entferne Konstanten aus der Variablenliste (sie sind keine Unbekannten)
    all_variables -= set(initial_values.keys())

    return equations, all_variables, initial_values, sweep_vars, original_equations, unit_values


def validate_system(equations: List[str], variables: Set[str], constants: Optional[Dict[str, float]] = None) -> Tuple[bool, str]:
    """
    Validiert das Gleichungssystem.

    Prüft ob die Anzahl der Gleichungen mit der Anzahl der Unbekannten übereinstimmt.
    Zählt Constraint-Gleichungen (wo die LHS-Variable eine Konstante ist) korrekt.

    Args:
        equations: Liste der geparsten Gleichungen
        variables: Menge der Unbekannten
        constants: Dict der Konstanten (optional, für Constraint-Zählung)
    """
    n_eq = len(equations)
    n_var = len(variables)

    if n_eq == 0:
        return False, "Keine Gleichungen gefunden."

    if n_var == 0:
        return False, "Keine Variablen gefunden."

    if n_eq < n_var:
        return False, f"Unterbestimmtes System: {n_eq} Gleichungen, aber {n_var} Unbekannte.\nVariablen: {', '.join(sorted(variables))}"

    # Zähle Constraint-Gleichungen (LHS ist Konstante)
    n_constraints = 0
    if constants:
        for eq in equations:
            # Gleichungen haben die Form "(var) - (expr)"
            match = re.match(r'^\(([a-zA-Z_][a-zA-Z0-9_]*)\)\s*-\s*\(', eq)
            if match:
                lhs_var = match.group(1)
                if lhs_var in constants:
                    n_constraints += 1

    # Effektive Gleichungsanzahl = Gleichungen - Constraints
    n_effective_eq = n_eq - n_constraints

    if n_effective_eq > n_var:
        return False, f"Überbestimmtes System: {n_eq} Gleichungen ({n_constraints} Constraints), aber nur {n_var} Unbekannte.\nVariablen: {', '.join(sorted(variables))}"

    if n_constraints > 0:
        return True, f"System: {n_eq} Gleichungen ({n_constraints} Constraints), {n_var} Unbekannte."

    return True, f"System OK: {n_eq} Gleichungen, {n_var} Unbekannte."


if __name__ == "__main__":
    # Test 1: Normale Gleichungen
    print("=== Test 1: Normale Gleichungen ===")
    test_input = """
    "Dies ist ein Kommentar"
    x + y = 10
    x - y = 2
    {Noch ein Kommentar}
    """

    equations, variables, initial, sweep, originals, units_info = parse_equations(test_input)
    print("Gleichungen:", equations)
    print("Variablen:", variables)
    print("Initialwerte:", initial)
    print("Sweep-Variablen:", sweep)
    print("Original-Gleichungen:", originals)
    print(validate_system(equations, variables))
    print()

    # Test 2: Vektor-Syntax
    print("=== Test 2: Vektor-Syntax ===")
    test_vector = """
    T = 0:10:100
    p = 1
    h = enthalpy(water, T=T, p=p)
    """

    equations, variables, initial, sweep, originals, units_info = parse_equations(test_vector)
    print("Gleichungen:", equations)
    print("Variablen:", variables)
    print("Initialwerte:", initial)
    print("Sweep-Variablen:")
    for name, arr in sweep.items():
        print(f"  {name}: {arr} ({len(arr)} Werte)")
    print()

    # Test 3: Verschiedene Vektor-Formate
    print("=== Test 3: Vektor-Formate ===")
    print("0:10:100 ->", parse_vector("0:10:100"))
    print("0:100 ->", parse_vector("0:100")[:5], "... (", len(parse_vector("0:100")), "Werte)")
    print("0:0.5:5 ->", parse_vector("0:0.5:5"))
