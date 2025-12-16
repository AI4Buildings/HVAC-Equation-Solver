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
from typing import List, Set, Tuple, Dict, Union, Any

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
        # Erzeuge Array (inklusive Endwert)
        n_points = int(abs((end - start) / step)) + 1
        return np.linspace(start, end, n_points)

    # Prüfe auf start:end Format (step=1)
    match2 = VECTOR_PATTERN_2.match(value_str)
    if match2:
        start = float(match2.group(1))
        end = float(match2.group(2))
        step = 1.0 if end >= start else -1.0
        n_points = int(abs(end - start)) + 1
        return np.linspace(start, end, n_points)

    return None


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
    # Entferne {...} Kommentare
    text = re.sub(r'\{[^}]*\}', '', text)
    return text


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


def convert_thermo_call(match) -> str:
    """
    Konvertiert einen Thermodynamik-Funktionsaufruf von EES zu Python-Syntax.
    Konvertiert auch Einheiten in key=value Argumenten zu SI-Basiseinheiten.

    EES:    enthalpy(water, T=100°C, p=1bar)
    Python: enthalpy('water', T=373.15, p=100000)

    EES:    h = enthalpy(R134a, T=T1, x=1)
    Python: h = enthalpy('R134a', T=T1, x=1)
    """
    func_name = match.group(1).lower()
    args_str = match.group(2)

    # Parse die Argumente
    # Erstes Argument ist der Stoffname (ohne Anführungszeichen in EES)
    # Weitere Argumente sind key=value Paare

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

    if len(args) < 1:
        return match.group(0)  # Unverändert zurückgeben

    # Erstes Argument ist der Stoffname - in Anführungszeichen setzen
    fluid = args[0]
    # Prüfe ob bereits in Anführungszeichen
    if not (fluid.startswith("'") or fluid.startswith('"')):
        fluid = f"'{fluid}'"

    # Restliche Argumente (key=value Paare) - konvertiere Einheiten zu SI
    rest_args = []
    for arg in args[1:]:
        rest_args.append(_convert_arg_units(arg))

    # Rekonstruiere den Aufruf
    new_args = [fluid] + rest_args
    return f"{func_name}({', '.join(new_args)})"


def convert_humid_air_call(match) -> str:
    """
    Converts a HumidAir function call from EES to Python syntax.
    Also converts units in key=value arguments to SI base units.

    EES:    HumidAir(h, T=25°C, rh=0.5, p_tot=1bar)
    Python: HumidAir('h', T=298.15, rh=0.5, p_tot=100000)

    EES:    w = HumidAir(w, T=30, rh=0.6, p_tot=1)
    Python: w = HumidAir('w', T=30, rh=0.6, p_tot=1)
    """
    func_name = match.group(1)  # Behalte Groß-/Kleinschreibung
    args_str = match.group(2)

    # Parse die Argumente
    # Erstes Argument ist die Output-Eigenschaft (h, phi, x, etc.)
    # Weitere Argumente sind key=value Paare

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

    if len(args) < 1:
        return match.group(0)  # Unverändert zurückgeben

    # Erstes Argument ist die Output-Eigenschaft - in Anführungszeichen setzen
    output_prop = args[0]
    # Prüfe ob bereits in Anführungszeichen
    if not (output_prop.startswith("'") or output_prop.startswith('"')):
        output_prop = f"'{output_prop}'"

    # Restliche Argumente (key=value Paare) - konvertiere Einheiten zu SI
    rest_args = []
    for arg in args[1:]:
        rest_args.append(_convert_arg_units(arg))

    # Rekonstruiere den Aufruf
    new_args = [output_prop] + rest_args
    return f"{func_name}({', '.join(new_args)})"


def tokenize_equation(equation: str) -> str:
    """Konvertiert EES-Syntax zu Python-Syntax."""
    # Ersetze ^ durch **
    equation = equation.replace('^', '**')

    # Ersetze ln durch log (numpy)
    equation = re.sub(r'\bln\b', 'log', equation)

    # Ersetze log10
    equation = re.sub(r'\blog10\b', 'log10', equation)

    # Konvertiere Thermodynamik-Funktionsaufrufe
    # Pattern: funktionsname(argumente)
    for func in THERMO_FUNCTIONS:
        pattern = rf'\b({func})\s*\(([^)]*)\)'
        equation = re.sub(pattern, convert_thermo_call, equation, flags=re.IGNORECASE)

    # Konvertiere FeuchteLuft-Funktionsaufrufe
    for func in HUMID_AIR_FUNCTIONS:
        pattern = rf'\b({func})\s*\(([^)]*)\)'
        equation = re.sub(pattern, convert_humid_air_call, equation, flags=re.IGNORECASE)

    return equation


def extract_variables(equation: str) -> Set[str]:
    """Extrahiert alle Variablennamen aus einer Gleichung."""
    temp_eq = equation

    # Entferne komplette Thermodynamik-Funktionsaufrufe
    # Diese enthalten den Stoffnamen und key=value Parameter
    # Pattern: funktionsname('stoffname', key1=val1, key2=val2)
    for func in THERMO_FUNCTIONS:
        # Finde alle Funktionsaufrufe und extrahiere die Werte (nicht die Keys)
        pattern = rf"\b{func}\s*\([^)]*\)"
        matches = re.findall(pattern, temp_eq, flags=re.IGNORECASE)

        for match in matches:
            # Extrahiere die Werte aus key=value Paaren
            # z.B. aus "enthalpy('water', T=T1, p=p1)" -> T1, p1
            values = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*([a-zA-Z_][a-zA-Z0-9_]*|\d+\.?\d*)', match)
            # Ersetze den kompletten Funktionsaufruf durch die Werte
            replacement = ' '.join(str(v) for v in values if not v.replace('.', '').isdigit())
            temp_eq = temp_eq.replace(match, replacement)

    # Entferne komplette FeuchteLuft-Funktionsaufrufe
    # Pattern: FeuchteLuft('eigenschaft', key1=val1, key2=val2, key3=val3)
    for func in HUMID_AIR_FUNCTIONS:
        pattern = rf"\b{func}\s*\([^)]*\)"
        matches = re.findall(pattern, temp_eq, flags=re.IGNORECASE)

        for match in matches:
            # Extrahiere die Werte aus key=value Paaren
            values = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*([a-zA-Z_][a-zA-Z0-9_]*|\d+\.?\d*)', match)
            # Ersetze den kompletten Funktionsaufruf durch die Werte
            replacement = ' '.join(str(v) for v in values if not v.replace('.', '').isdigit())
            temp_eq = temp_eq.replace(match, replacement)

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
                from units import UnitValue
                magnitude, unit_str = parse_value_with_unit(right)
                if unit_str:
                    pre_constants.add(var_name)
                    continue
            except ValueError:
                pass
        # Reine Zahl
        try:
            float(right)
            pre_constants.add(var_name)
            continue
        except ValueError:
            pass
        # Numerischer Ausdruck (ohne Variablen)
        right_tokenized = tokenize_equation(right)
        vars_in_right = extract_variables(right_tokenized)
        if not vars_in_right:
            # Versuche auszuwerten
            try:
                eval(right_tokenized, {"__builtins__": {}}, {
                    'pi': np.pi, 'e': np.e,
                    'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                    'sqrt': np.sqrt, 'log': np.log, 'log10': np.log10,
                    'exp': np.exp, 'abs': abs
                })
                pre_constants.add(var_name)
            except Exception:
                pass

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
            # Wenn Einheit angegeben: Konvertiere zu Standard-Berechnungseinheit
            if vec_unit and UNITS_AVAILABLE:
                try:
                    from units import UnitValue
                    # Erstelle UnitValue für ersten Wert um Konvertierungsfaktor zu bekommen
                    first_uv = UnitValue.from_input(vec_array[0], vec_unit)
                    # Berechne Konvertierungsfaktor: calc_value / original_value
                    if first_uv.original_value != 0:
                        conversion_factor = first_uv.calc_value / first_uv.original_value
                        # Wende Faktor auf alle Werte an (für lineare Konvertierungen)
                        calc_values = vec_array * conversion_factor
                    else:
                        # Offset-Konvertierung (z.B. °C zu K): Konvertiere einzeln
                        calc_values = np.array([UnitValue.from_input(v, vec_unit).calc_value for v in vec_array])
                    sweep_vars[var_name] = calc_values
                    # Speichere UnitValue für Anzeige (erster Wert)
                    unit_values[var_name] = first_uv
                except Exception as e:
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
                magnitude, unit_str = parse_value_with_unit(right)
                if unit_str:
                    # Wert mit Einheit gefunden (z.B. "15°C", "10g", "2.5kJ/kg")
                    var_name = left

                    # Spezialfall: Temperaturdifferenz (dT..., delta...)
                    # K sollte als Differenz behandelt werden, nicht als absolute Temperatur
                    # Erkennungsmuster: Variablenname beginnt mit "dT" oder "delta"
                    var_lower = var_name.lower()
                    is_temp_diff = (
                        unit_str.upper() == 'K' and
                        (var_lower.startswith('dt') or
                         var_lower.startswith('delta'))
                    )

                    if is_temp_diff:
                        # Temperaturdifferenz: 1K = 1 delta_K
                        # Keine Konvertierung nötig, Wert bleibt gleich
                        initial_values[var_name] = magnitude
                        if parse_units:
                            # Erstelle UnitValue mit delta_K als Differenz-Einheit
                            unit_values[var_name] = UnitValue.from_input(magnitude, 'delta_K')
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
        # Das ist der Fall wenn links nur eine Variable steht
        # und rechts eine Zahl oder ein arithmetischer Ausdruck ohne Variablen
        if len(vars_left) == 1 and len(vars_right) == 0:
            var_name = list(vars_left)[0]

            # Versuche als einfache Zahl
            try:
                value = float(right)
                initial_values[var_name] = value
                continue
            except ValueError:
                # Versuche als arithmetischen Ausdruck auszuwerten
                try:
                    # Trigonometrische Funktionen in GRAD (wie EES)
                    def _sin(x): return np.sin(np.radians(x))
                    def _cos(x): return np.cos(np.radians(x))
                    def _tan(x): return np.tan(np.radians(x))
                    def _asin(x): return np.degrees(np.arcsin(x))
                    def _acos(x): return np.degrees(np.arccos(x))
                    def _atan(x): return np.degrees(np.arctan(x))

                    # Nur sichere mathematische Operationen erlauben
                    value = eval(right, {"__builtins__": {}}, {
                        'pi': np.pi, 'e': np.e,
                        'sin': _sin, 'cos': _cos, 'tan': _tan,
                        'asin': _asin, 'acos': _acos, 'atan': _atan,
                        'sqrt': np.sqrt, 'log': np.log, 'log10': np.log10,
                        'exp': np.exp, 'abs': abs,
                        'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh
                    })
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        initial_values[var_name] = float(value)
                        continue
                except Exception:
                    pass

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


def validate_system(equations: List[str], variables: Set[str], constants: Dict[str, float] = None) -> Tuple[bool, str]:
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

    equations, variables, initial, sweep, originals = parse_equations(test_input)
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

    equations, variables, initial, sweep, originals = parse_equations(test_vector)
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
