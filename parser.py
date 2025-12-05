"""
EES-ähnlicher Gleichungsparser

Unterstützte Syntax:
- Gleichungen: x + y = 10
- Zuweisungen: T1 = 300
- Vektoren: T = 0:10:100 (start:step:end) oder T = 0:100 (start:end, step=1)
- Operatoren: +, -, *, /, ^ (Potenz)
- Funktionen: sin, cos, tan, exp, ln, log10, sqrt, abs
- Thermodynamik: enthalpy(water, T=100, p=1), density(R134a, T=25, x=1)
- Kommentare: "..." oder {...}
"""

import re
import numpy as np
from typing import List, Set, Tuple, Dict, Union, Any


# Mathematische Funktionen die unterstützt werden
MATH_FUNCTIONS = {
    'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
    'sinh', 'cosh', 'tanh',
    'exp', 'ln', 'log10', 'sqrt', 'abs',
    'pi'
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


def is_vector_assignment(line: str) -> Tuple[bool, str, str]:
    """
    Prüft ob eine Zeile eine Vektor-Zuweisung ist.

    Returns:
        (is_vector, var_name, vector_string)
    """
    if '=' not in line or ':' not in line:
        return False, '', ''

    parts = line.split('=', 1)
    if len(parts) != 2:
        return False, '', ''

    left = parts[0].strip()
    right = parts[1].strip()

    # Links muss eine einfache Variable sein
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', left):
        return False, '', ''

    # Rechts muss Vektor-Syntax sein
    if parse_vector(right) is not None:
        return True, left, right

    return False, '', ''


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


def convert_thermo_call(match) -> str:
    """
    Konvertiert einen Thermodynamik-Funktionsaufruf von EES zu Python-Syntax.

    EES:    enthalpy(water, T=100, p=1)
    Python: enthalpy('water', T=100, p=1)

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

    # Restliche Argumente (key=value Paare)
    rest_args = args[1:]

    # Rekonstruiere den Aufruf
    new_args = [fluid] + rest_args
    return f"{func_name}({', '.join(new_args)})"


def convert_humid_air_call(match) -> str:
    """
    Converts a HumidAir function call from EES to Python syntax.

    EES:    HumidAir(h, T=25, rh=0.5, p_tot=1)
    Python: HumidAir('h', T=25, rh=0.5, p_tot=1)

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

    # Restliche Argumente (key=value Paare)
    rest_args = args[1:]

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

    # Entferne Python-Keywords
    # NICHT die Funktionsnamen entfernen - sie können als Variablen verwendet werden
    # (z.B. cp = cv + R). Die Funktionsaufrufe wurden bereits oben aus temp_eq entfernt.
    python_keywords = {'and', 'or', 'not', 'True', 'False', 'None', 'log', 'log10'}

    variables -= python_keywords

    return variables


def parse_equations(text: str) -> Tuple[List[str], Set[str], dict, dict]:
    """
    Parst den Eingabetext und extrahiert Gleichungen und Variablen.

    Returns:
        equations: Liste von Gleichungen in Python-Syntax (als f(x) = 0 Form)
        variables: Set aller gefundenen Variablen (ohne Sweep-Variable)
        initial_values: Dict mit vorgegebenen Werten (direkte Zuweisungen)
        sweep_vars: Dict mit Vektor-Variablen {name: numpy.array}
    """
    # Entferne Kommentare
    text = remove_comments(text)

    # Teile in Zeilen auf
    lines = text.split('\n')

    equations = []
    all_variables = set()
    initial_values = {}
    sweep_vars = {}  # Vektor-Variablen für Parameterstudien

    for line in lines:
        line = line.strip()

        # Überspringe leere Zeilen
        if not line:
            continue

        # Prüfe ob es eine Gleichung ist (enthält =)
        if '=' not in line:
            continue

        # Prüfe auf Vektor-Zuweisung (z.B. T = 0:10:100)
        is_vec, var_name, vec_str = is_vector_assignment(line)
        if is_vec:
            sweep_vars[var_name] = parse_vector(vec_str)
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
            try:
                # Versuche zuerst als einfache Zahl
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
        all_vars = vars_left | vars_right
        all_variables |= all_vars

        # Erstelle Gleichung in der Form: left - right = 0
        equation = f"({left}) - ({right})"
        equations.append(equation)

    # Entferne Sweep-Variablen aus der Variablenliste (sie sind keine Unbekannten)
    all_variables -= set(sweep_vars.keys())

    # Entferne Konstanten aus der Variablenliste (sie sind keine Unbekannten)
    all_variables -= set(initial_values.keys())

    return equations, all_variables, initial_values, sweep_vars


def validate_system(equations: List[str], variables: Set[str]) -> Tuple[bool, str]:
    """
    Validiert das Gleichungssystem.

    Prüft ob die Anzahl der Gleichungen mit der Anzahl der Unbekannten übereinstimmt.
    """
    n_eq = len(equations)
    n_var = len(variables)

    if n_eq == 0:
        return False, "Keine Gleichungen gefunden."

    if n_var == 0:
        return False, "Keine Variablen gefunden."

    if n_eq < n_var:
        return False, f"Unterbestimmtes System: {n_eq} Gleichungen, aber {n_var} Unbekannte.\nVariablen: {', '.join(sorted(variables))}"

    if n_eq > n_var:
        return False, f"Überbestimmtes System: {n_eq} Gleichungen, aber nur {n_var} Unbekannte.\nVariablen: {', '.join(sorted(variables))}"

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

    equations, variables, initial, sweep = parse_equations(test_input)
    print("Gleichungen:", equations)
    print("Variablen:", variables)
    print("Initialwerte:", initial)
    print("Sweep-Variablen:", sweep)
    print(validate_system(equations, variables))
    print()

    # Test 2: Vektor-Syntax
    print("=== Test 2: Vektor-Syntax ===")
    test_vector = """
    T = 0:10:100
    p = 1
    h = enthalpy(water, T=T, p=p)
    """

    equations, variables, initial, sweep = parse_equations(test_vector)
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
