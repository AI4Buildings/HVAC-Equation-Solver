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

import re
import numpy as np
from typing import List, Set, Tuple, Dict, Union, Optional

# Einheiten-Modul (optional, falls nicht vorhanden wird ohne Einheiten gearbeitet)
try:
    from units import parse_value_with_unit, UnitValue
    UNITS_AVAILABLE = True
except ImportError:
    UNITS_AVAILABLE = False
    UnitValue = None


# Mathematische Funktionen die unterstützt werden
MATH_FUNCTIONS = {
    'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
    'sinh', 'cosh', 'tanh',
    'exp', 'ln', 'log10', 'sqrt', 'abs',
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
    if '=' not in line or ':' not in line:
        return False, '', '', ''

    parts = line.split('=', 1)
    if len(parts) != 2:
        return False, '', '', ''

    left = parts[0].strip()
    right = parts[1].strip()

    # Links muss eine einfache Variable sein
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', left):
        return False, '', '', ''

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
    """
    # Entferne "..." Kommentare
    text = re.sub(r'"[^"]*"', '', text)
    # Entferne {...} Kommentare (auch verschachtelt, per Klammer-Zählung)
    result = []
    depth = 0
    unmatched_start = None  # Position des ersten unbalancierten '{'
    for i, char in enumerate(text):
        if char == '{':
            if depth == 0:
                unmatched_start = i
            depth += 1
        elif char == '}':
            if depth > 0:
                depth -= 1
                if depth == 0:
                    unmatched_start = None
        elif depth == 0:
            result.append(char)
    if depth > 0 and unmatched_start is not None:
        # Unbalancierter Kommentar: Text ab dem offenen '{' unverändert lassen
        result.append(text[unmatched_start:])
    return ''.join(result)


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
            if unit_str:
                from units import UnitValue
                unit_value = UnitValue.from_input(magnitude, unit_str)
                # Use SI base value for calculations
                return f"{key}={unit_value.si_value}"
        except (ValueError, Exception):
            pass

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


def _replace_calls_balanced(text: str, func_names: Set[str], keep_case: bool = False) -> str:
    """
    Ersetzt alle func(...)-Aufrufe (auch verschachtelte) via _convert_call_parts.
    Verschachtelte Aufrufe in den Argumenten werden zuerst konvertiert.
    """
    names_lower = {n.lower() for n in func_names}

    # Wiederhole bis stabil: pro Durchlauf wird der erste Aufruf konvertiert,
    # dessen Konvertierung den Text tatsächlich ändert (idempotent -> terminiert).
    for _ in range(100):  # Sicherheitslimit
        changed = False
        for start, open_idx, close_idx, name in _iter_call_spans(text, names_lower):
            args_str = text[open_idx + 1:close_idx]
            # Innere Aufrufe in den Argumenten zuerst konvertieren
            args_converted = _replace_calls_balanced(args_str, func_names, keep_case) \
                if any(f in args_str.lower() for f in names_lower) else args_str
            converted = _convert_call_parts(name, args_converted, keep_case=keep_case)
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

    # Ersetze ln durch log (numpy)
    equation = re.sub(r'\bln\b', 'log', equation)

    # Ersetze log10
    equation = re.sub(r'\blog10\b', 'log10', equation)

    # Konvertiere Thermodynamik-Funktionsaufrufe (balanciert, auch verschachtelt)
    equation = _replace_calls_balanced(equation, THERMO_FUNCTIONS, keep_case=False)

    # Konvertiere FeuchteLuft-Funktionsaufrufe
    equation = _replace_calls_balanced(equation, HUMID_AIR_FUNCTIONS, keep_case=True)

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
        if isinstance(value, (int, float, np.floating)) and np.isfinite(value):
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

    # Einheit muss mit Buchstabe/°/µ beginnen und darf kein Operator-Rest sein
    if not re.match(r'^[a-zA-Z°µ]', unit_part):
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
    value = _eval_constant_expression(expr_python)
    if value is None:
        return None

    return value, unit_str


def parse_equations(text: str, parse_units: bool = True) -> Tuple[List[str], Set[str], dict, dict, dict, dict]:
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

    # Entferne Kommentare
    text = remove_comments(text)

    # Teile in Zeilen auf
    lines = text.split('\n')
    original_lines = original_text.split('\n')

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
    for line in lines:
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
            vec_array = parse_vector(vec_str)
            # Prüfe ob Vektor erfolgreich geparst wurde
            if vec_array is None:
                continue
            # Wenn Einheit angegeben: Konvertiere zu Standard-Berechnungseinheit
            if vec_unit and UNITS_AVAILABLE:
                try:
                    from units import UnitValue
                    # Konvertiere IMMER elementweise: nur so werden Offset-Einheiten
                    # (°C -> K: +273.15) korrekt behandelt. Ein multiplikativer
                    # Faktor wäre bei 20:10:50 °C für alle Werte außer dem ersten falsch.
                    calc_values = np.array([
                        UnitValue.from_input(float(v), vec_unit).calc_value
                        for v in vec_array
                    ])
                    sweep_vars[var_name] = calc_values
                    # Speichere UnitValue für Anzeige (erster Wert)
                    unit_values[var_name] = UnitValue.from_input(float(vec_array[0]), vec_unit)
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
                        unit_value = UnitValue.from_input(magnitude, unit_str)
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
