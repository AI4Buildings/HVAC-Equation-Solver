"""
Einheiten-Modul für den HVAC Equation Solver

Verwendet pint für Einheiten-Konvertierung und -Verwaltung.

Features:
- Parsing von Werten mit Einheiten: "15°C", "10g", "2.5kJ/kg"
- Hybrid-Speicherung: SI intern, Original-Einheit merken
- Konvertierung zwischen kompatiblen Einheiten
- Automatische Einheiten für CoolProp-Funktionen
"""

import re
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional, Any
import pint

# Globale Unit Registry
ureg = pint.UnitRegistry()

# Zusätzliche Einheiten-Definitionen für HVAC
ureg.define('degC = kelvin; offset: 273.15 = °C = celsius')
ureg.define('degF = 5/9 * kelvin; offset: 255.372222 = °F = fahrenheit')

# Aliase für häufige Einheiten
ureg.define('@alias bar = Bar')
ureg.define('@alias pascal = Pa')
ureg.define('@alias joule = J')
ureg.define('@alias watt = W')
ureg.define('@alias kilogram = kg')
ureg.define('@alias meter = m')
ureg.define('@alias second = s')
ureg.define('@alias liter = L = l')


# Einheiten-Mapping für CoolProp-Funktionen (Ausgabe-Einheiten)
COOLPROP_UNITS = {
    'enthalpy': 'kJ/kg',
    'entropy': 'kJ/(kg*K)',
    'density': 'kg/m^3',
    'temperature': 'degC',
    'pressure': 'bar',
    'cp': 'kJ/(kg*K)',
    'cv': 'kJ/(kg*K)',
    'viscosity': 'Pa*s',
    'conductivity': 'W/(m*K)',
    'volume': 'm^3/kg',
    'intenergy': 'kJ/kg',
    'quality': '',  # dimensionslos
    'prandtl': '',  # dimensionslos
    'soundspeed': 'm/s',
}

# Einheiten für HumidAir-Funktionen
HUMID_AIR_UNITS = {
    'h': 'kJ/kg',
    'w': 'kg/kg',
    'rh': '',  # dimensionslos (0-1)
    't': 'degC',
    't_dp': 'degC',
    't_wb': 'degC',
    'rho_tot': 'kg/m^3',
    'rho_a': 'kg/m^3',
    'rho_w': 'kg/m^3',
    'p_w': 'bar',
}

# Einheiten für Strahlungs-Funktionen
RADIATION_UNITS = {
    'eb': 'W/(m^2*um)',              # Spektrale Emissionsleistung [W/(m²·µm)]
    'blackbody': '',                  # Anteil (dimensionslos, 0-1)
    'blackbody_cumulative': '',       # Kumulativer Anteil (dimensionslos, 0-1)
    'wien': 'um',                     # Wellenlänge maximaler Emission [µm]
    'stefan_boltzmann': 'W/m^2',      # Gesamtemission [W/m²]
}

# Standard-Einheiten für HVAC-Berechnungen
# Diese Einheiten werden für interne Berechnungen verwendet
# SI-basiert, damit Einheiten-Arithmetik automatisch funktioniert:
#   m_dot [kg/s] * Δh [kJ/kg] = Q [kJ/s] = Q [kW]
STANDARD_UNITS = {
    # Dimension -> (Ziel-Einheit, pint-kompatibel)
    '[temperature]': ('degC', 'degC'),           # Temperatur in °C (für CoolProp)
    '[mass] / [time]': ('kg/s', 'kg/s'),         # Massenstrom in kg/s (SI!)
    '[length] ** 3 / [time]': ('m^3/s', 'm^3/s'), # Volumenstrom in m³/s (SI!)
    '[mass] / [length] ** 3': ('kg/m^3', 'kg/m^3'), # Dichte in kg/m³
    '[mass] / [length] / [time] ** 2': ('bar', 'bar'),  # Druck in bar (für CoolProp)
    '[length] ** 2 / [time] ** 2': ('kJ/kg', 'kJ/kg'),  # Spez. Energie in kJ/kg
    '[length] ** 2 / [time] ** 2 / [temperature]': ('kJ/(kg*K)', 'kJ/(kg*K)'),  # Spez. Wärme
    '[mass] * [length] ** 2 / [time] ** 3': ('kW', 'kW'),  # Leistung in kW
    '[mass] * [length] ** 2 / [time] ** 2': ('kJ', 'kJ'),  # Energie in kJ
    '[mass]': ('kg', 'kg'),                      # Masse in kg
    '[length]': ('m', 'm'),                      # Länge in m
    '[time]': ('s', 's'),                        # Zeit in s
    '[time] ** -1': ('1/s', '1/s'),              # Frequenz
}


# Kompatible Einheiten für Dropdown-Menüs
COMPATIBLE_UNITS = {
    # Temperatur
    'degC': ['degC', 'K', 'degF'],
    'K': ['K', 'degC', 'degF'],
    'degF': ['degF', 'degC', 'K'],
    'celsius': ['degC', 'K', 'degF'],
    '°C': ['degC', 'K', 'degF'],
    '°F': ['degF', 'degC', 'K'],

    # Druck
    'bar': ['bar', 'Pa', 'kPa', 'MPa', 'mbar', 'atm', 'psi'],
    'Pa': ['Pa', 'kPa', 'MPa', 'bar', 'mbar', 'atm', 'psi'],
    'kPa': ['kPa', 'Pa', 'MPa', 'bar', 'mbar', 'atm', 'psi'],
    'MPa': ['MPa', 'kPa', 'Pa', 'bar', 'atm', 'psi'],
    'mbar': ['mbar', 'bar', 'Pa', 'kPa', 'atm', 'psi'],
    'atm': ['atm', 'bar', 'Pa', 'kPa', 'MPa', 'psi'],
    'psi': ['psi', 'bar', 'Pa', 'kPa', 'MPa', 'atm'],

    # Masse
    'kg': ['kg', 'g', 'mg', 't', 'lb'],
    'g': ['g', 'kg', 'mg', 't', 'lb'],
    'mg': ['mg', 'g', 'kg'],
    't': ['t', 'kg', 'lb'],
    'lb': ['lb', 'kg', 'g'],

    # Länge
    'm': ['m', 'cm', 'mm', 'km', 'inch', 'ft'],
    'cm': ['cm', 'm', 'mm', 'inch'],
    'mm': ['mm', 'cm', 'm', 'inch'],
    'km': ['km', 'm', 'mile'],
    'inch': ['inch', 'cm', 'mm', 'm', 'ft'],
    'ft': ['ft', 'm', 'inch'],

    # Zeit
    's': ['s', 'min', 'h'],
    'min': ['min', 's', 'h'],
    'h': ['h', 'min', 's'],
    'hour': ['h', 'min', 's'],

    # Energie
    'J': ['J', 'kJ', 'MJ', 'Wh', 'kWh', 'cal', 'BTU'],
    'kJ': ['kJ', 'J', 'MJ', 'Wh', 'kWh', 'cal', 'BTU'],
    'MJ': ['MJ', 'kJ', 'J', 'kWh'],
    'Wh': ['Wh', 'kWh', 'J', 'kJ'],
    'kWh': ['kWh', 'Wh', 'MJ', 'kJ', 'J'],
    'cal': ['cal', 'kcal', 'J', 'kJ'],
    'kcal': ['kcal', 'cal', 'kJ', 'J'],
    'BTU': ['BTU', 'kJ', 'J'],

    # Leistung
    'W': ['W', 'kW', 'MW', 'hp'],
    'kW': ['kW', 'W', 'MW', 'hp'],
    'MW': ['MW', 'kW', 'W'],
    'hp': ['hp', 'kW', 'W'],

    # Enthalpie / spezifische Energie
    'kJ/kg': ['kJ/kg', 'J/kg', 'BTU/lb'],
    'J/kg': ['J/kg', 'kJ/kg'],

    # Entropie / spezifische Wärme
    'kJ/(kg*K)': ['kJ/(kg*K)', 'J/(kg*K)'],
    'J/(kg*K)': ['J/(kg*K)', 'kJ/(kg*K)'],

    # Dichte
    'kg/m^3': ['kg/m^3', 'g/cm^3', 'g/L', 'kg/L'],
    'g/cm^3': ['g/cm^3', 'kg/m^3', 'g/L'],
    'g/L': ['g/L', 'kg/m^3', 'g/cm^3', 'kg/L'],

    # Volumenstrom
    'm^3/s': ['m^3/s', 'm^3/h', 'L/s', 'L/min', 'L/h'],
    'm^3/h': ['m^3/h', 'm^3/s', 'L/s', 'L/min', 'L/h'],
    'L/s': ['L/s', 'L/min', 'L/h', 'm^3/s', 'm^3/h'],
    'L/min': ['L/min', 'L/s', 'L/h', 'm^3/h'],
    'L/h': ['L/h', 'L/min', 'L/s', 'm^3/h'],

    # Massenstrom
    'kg/s': ['kg/s', 'kg/h', 'g/s', 'kg/min', 't/h'],
    'kg/h': ['kg/h', 'kg/s', 'g/s', 'kg/min', 't/h'],
    'g/s': ['g/s', 'kg/s', 'kg/h'],
    't/h': ['t/h', 'kg/h', 'kg/s'],

    # Geschwindigkeit
    'm/s': ['m/s', 'km/h', 'mph', 'ft/s'],
    'km/h': ['km/h', 'm/s', 'mph'],

    # Spezifisches Volumen
    'm^3/kg': ['m^3/kg', 'L/kg', 'cm^3/g'],

    # Viskosität
    'Pa*s': ['Pa*s', 'mPa*s', 'cP'],

    # Wärmeleitfähigkeit
    'W/(m*K)': ['W/(m*K)'],
}


# Regex für Wert mit Einheit
# Matches: "15", "15.5", "-3.14", "1.5e-3", "15°C", "100kJ/kg", "4.18kJ/(kg*K)"
VALUE_WITH_UNIT_PATTERN = re.compile(
    r'^'
    r'(-?\d+\.?\d*(?:[eE][+-]?\d+)?)'  # Zahl (inkl. wissenschaftliche Notation)
    r'\s*'                               # Optionale Leerzeichen
    r'(°?[a-zA-Z²³µ][a-zA-Z0-9²³µ°/*^()]*)?'  # Optionale Einheit
    r'$'
)

def _convert_to_standard(quantity) -> Tuple[float, str]:
    """
    Konvertiert eine pint Quantity zur Standard-Einheit für HVAC-Berechnungen.

    Args:
        quantity: pint Quantity

    Returns:
        (wert, einheit) in Standard-Einheit
    """
    try:
        # Spezialfall: µm für Strahlungsfunktionen nicht konvertieren
        # Die Blackbody/Eb/Wien-Funktionen erwarten Wellenlängen in µm
        unit_str = str(quantity.units)
        if unit_str in ('micrometer', 'um', 'µm', 'micron'):
            return (float(quantity.magnitude), 'µm')

        # Spezialfall: Kelvin für Strahlungsberechnungen nicht zu °C konvertieren
        # Stefan-Boltzmann und Planck-Funktionen benötigen absolute Temperatur
        # Wenn Benutzer explizit K angibt, soll der Wert in K bleiben
        if unit_str in ('kelvin', 'K'):
            return (float(quantity.magnitude), 'K')

        dim_str = str(quantity.dimensionality)

        # Suche passende Standard-Einheit
        for dim_pattern, (target_unit, pint_unit) in STANDARD_UNITS.items():
            if dim_str == dim_pattern:
                converted = quantity.to(pint_unit)
                return (float(converted.magnitude), target_unit)

        # Fallback: Behalte Original wenn keine Standard-Einheit definiert
        return (float(quantity.magnitude), str(quantity.units))

    except Exception:
        return (float(quantity.magnitude), str(quantity.units))


# Mapping von User-freundlichen zu pint-kompatiblen Einheiten
UNIT_ALIASES = {
    '°C': 'degC',
    '°F': 'degF',
    'celsius': 'degC',
    'fahrenheit': 'degF',
    'm³': 'm^3',
    'm²': 'm^2',
    'µm': 'micrometer',
    'µs': 'microsecond',
    # Kompakte Bruch-Notation
    'kJ/kgK': 'kJ/(kg*K)',
    'J/kgK': 'J/(kg*K)',
    'kJ/kgC': 'kJ/(kg*K)',
    'W/mK': 'W/(m*K)',
    'W/m²K': 'W/(m^2*K)',
    # Weitere Aliase
    'Bar': 'bar',
    'BAR': 'bar',
}


@dataclass
class UnitValue:
    """
    Wert mit Einheit - Hybrid-Speicherung.

    Speichert sowohl den SI-Wert für Berechnungen als auch
    den Original-Wert und die Original-Einheit für die Anzeige.
    """
    si_value: float               # Wert in SI-Basiseinheit
    si_unit: str                  # SI-Einheit als String
    original_value: float         # Original-Eingabewert
    original_unit: str            # Original-Einheit als String
    quantity: Any = field(default=None, repr=False)  # pint Quantity
    _calc_value: float = field(default=None, repr=False)  # Wert in Standard-Einheit
    _calc_unit: str = field(default='', repr=False)       # Standard-Einheit

    @classmethod
    def from_input(cls, value: float, unit_str: str) -> 'UnitValue':
        """
        Erstellt UnitValue aus Benutzereingabe.

        Args:
            value: Numerischer Wert
            unit_str: Einheit als String (z.B. "°C", "kJ/kg")

        Returns:
            UnitValue mit Konvertierung zu Standard-Einheiten
        """
        # Normalisiere Einheit
        normalized_unit = normalize_unit(unit_str)

        try:
            # Erstelle pint Quantity
            quantity = ureg.Quantity(value, normalized_unit)

            # Spezialbehandlung für Temperatur-Offset-Einheiten
            if normalized_unit in ['degC', 'degF', 'celsius', 'fahrenheit']:
                si_quantity = quantity.to('kelvin')
                si_unit = 'kelvin'
            else:
                si_quantity = quantity.to_base_units()
                si_unit = str(si_quantity.units)

            # Berechne Wert in Standard-Einheit für Berechnungen
            calc_value, calc_unit = _convert_to_standard(quantity)

            return cls(
                si_value=si_quantity.magnitude,
                si_unit=si_unit,
                original_value=value,
                original_unit=unit_str,
                quantity=quantity,
                _calc_value=calc_value,
                _calc_unit=calc_unit
            )
        except Exception as e:
            # Fallback: Einheit nicht erkannt, behandle als dimensionslos
            return cls(
                si_value=value,
                si_unit='',
                original_value=value,
                original_unit=unit_str,
                quantity=None,
                _calc_value=value,
                _calc_unit=''
            )

    @property
    def calc_value(self) -> float:
        """Wert in Standard-Einheit für Berechnungen (kg/s, °C, bar, kJ/kg, etc.)"""
        if self._calc_value is not None:
            return self._calc_value
        return self.original_value

    @property
    def calc_unit(self) -> str:
        """Standard-Einheit für Berechnungen"""
        return self._calc_unit or self.original_unit

    @classmethod
    def from_si(cls, value: float, unit_str: str) -> 'UnitValue':
        """
        Erstellt UnitValue aus SI-Wert (z.B. für CoolProp-Ergebnisse).

        Args:
            value: Wert bereits in der angegebenen Einheit
            unit_str: Einheit des Wertes (z.B. "kJ/kg")

        Returns:
            UnitValue
        """
        normalized_unit = normalize_unit(unit_str)

        try:
            quantity = value * ureg(normalized_unit)
            si_quantity = quantity.to_base_units()

            return cls(
                si_value=si_quantity.magnitude,
                si_unit=str(si_quantity.units),
                original_value=value,
                original_unit=unit_str,
                quantity=quantity
            )
        except Exception:
            return cls(
                si_value=value,
                si_unit='',
                original_value=value,
                original_unit=unit_str,
                quantity=None
            )

    @classmethod
    def dimensionless(cls, value: float) -> 'UnitValue':
        """Erstellt dimensionslosen UnitValue."""
        return cls(
            si_value=value,
            si_unit='',
            original_value=value,
            original_unit='',
            quantity=value * ureg.dimensionless
        )

    def to(self, target_unit: str) -> float:
        """
        Konvertiert zu Ziel-Einheit.

        Args:
            target_unit: Ziel-Einheit als String

        Returns:
            Numerischer Wert in Ziel-Einheit
        """
        if not target_unit:
            return self.original_value if self.original_unit else self.si_value

        if self.quantity is None:
            return self.si_value

        try:
            normalized = normalize_unit(target_unit)
            converted = self.quantity.to(normalized)
            return float(converted.magnitude)
        except Exception:
            # Fallback: versuche über SI-Wert
            try:
                if self.si_unit:
                    si_qty = ureg.Quantity(self.si_value, self.si_unit)
                    return float(si_qty.to(normalized).magnitude)
            except Exception:
                pass
            return self.si_value

    def display(self, unit: str = None) -> Tuple[float, str]:
        """
        Gibt Wert und Einheit für Anzeige zurück.

        Args:
            unit: Optionale Ziel-Einheit, sonst Original-Einheit

        Returns:
            (wert, einheit) Tuple
        """
        if unit is None:
            return (self.original_value, self.original_unit)

        return (self.to(unit), unit)

    def __repr__(self):
        if self.original_unit:
            return f"UnitValue({self.original_value} {self.original_unit})"
        return f"UnitValue({self.si_value})"


def normalize_unit(unit_str: str) -> str:
    """
    Normalisiert eine Einheit zu pint-kompatiblem Format.

    Args:
        unit_str: Einheit als String (z.B. "°C", "kJ/kgK")

    Returns:
        Pint-kompatible Einheit
    """
    if not unit_str:
        return 'dimensionless'

    # Entferne führende/trailing Leerzeichen
    unit_str = unit_str.strip()

    # Prüfe Aliase
    if unit_str in UNIT_ALIASES:
        return UNIT_ALIASES[unit_str]

    # Ersetze Unicode-Zeichen
    unit_str = unit_str.replace('°C', 'degC')
    unit_str = unit_str.replace('°F', 'degF')
    unit_str = unit_str.replace('³', '^3')
    unit_str = unit_str.replace('²', '^2')
    unit_str = unit_str.replace('µ', 'micro')

    # Normalisiere Bruch-Notation ohne Klammern
    # z.B. "kJ/kgK" -> "kJ/(kg*K)"
    if '/' in unit_str and '(' not in unit_str:
        parts = unit_str.split('/')
        if len(parts) == 2:
            numerator = parts[0]
            denominator = parts[1]

            # Prüfe ob Nenner zusammengesetzt ist (z.B. "kgK")
            # Heuristik: Wenn Großbuchstabe in der Mitte, dann aufteilen
            if len(denominator) > 1:
                # Finde Position des zweiten Großbuchstabens
                for i, c in enumerate(denominator[1:], 1):
                    if c.isupper():
                        # Teile auf: "kgK" -> "kg*K"
                        denominator = denominator[:i] + '*' + denominator[i:]
                        break

            unit_str = f"{numerator}/({denominator})"

    return unit_str


def parse_value_with_unit(text: str) -> Tuple[float, str]:
    """
    Parst einen Wert mit optionaler Einheit.

    Args:
        text: String wie "15°C", "10", "2.5kJ/kg"

    Returns:
        (wert, einheit) Tuple. Einheit ist "" wenn keine angegeben.

    Raises:
        ValueError: Wenn das Format ungültig ist
    """
    text = text.strip()

    if not text:
        raise ValueError("Leerer Wert")

    match = VALUE_WITH_UNIT_PATTERN.match(text)

    if not match:
        # Versuche nur als Zahl zu parsen
        try:
            value = float(text)
            return (value, "")
        except ValueError:
            raise ValueError(f"Ungültiges Format: {text}")

    value_str = match.group(1)
    unit_str = match.group(2) or ""

    try:
        value = float(value_str)
    except ValueError:
        raise ValueError(f"Ungültiger Zahlenwert: {value_str}")

    return (value, unit_str)


def get_compatible_units(unit_str: str) -> List[str]:
    """
    Gibt Liste kompatibler Einheiten für Dropdown zurück.

    Args:
        unit_str: Aktuelle Einheit

    Returns:
        Liste kompatibler Einheiten (inkl. aktueller)
    """
    if not unit_str:
        return ['-']

    # Normalisiere für Lookup
    normalized = normalize_unit(unit_str)

    # Suche in COMPATIBLE_UNITS
    if normalized in COMPATIBLE_UNITS:
        return COMPATIBLE_UNITS[normalized]

    # Versuche SI-Basiseinheit zu finden
    try:
        quantity = 1 * ureg(normalized)
        base_unit = str(quantity.to_base_units().units)

        # Suche kompatible Einheiten basierend auf Dimensionalität
        for key, units in COMPATIBLE_UNITS.items():
            try:
                key_quantity = 1 * ureg(normalize_unit(key))
                if quantity.dimensionality == key_quantity.dimensionality:
                    # Original-Einheit an erste Stelle
                    result = [unit_str] + [u for u in units if u != unit_str]
                    return result
            except Exception:
                continue
    except Exception:
        pass

    # Fallback: nur aktuelle Einheit
    return [unit_str]


def convert_value(value: float, from_unit: str, to_unit: str) -> float:
    """
    Konvertiert einen Wert zwischen Einheiten.

    Args:
        value: Numerischer Wert
        from_unit: Quell-Einheit
        to_unit: Ziel-Einheit

    Returns:
        Konvertierter Wert
    """
    if not from_unit or not to_unit or from_unit == to_unit:
        return value

    try:
        from_normalized = normalize_unit(from_unit)
        to_normalized = normalize_unit(to_unit)

        # Verwende Quantity für korrekte Offset-Behandlung
        quantity = ureg.Quantity(value, from_normalized)
        converted = quantity.to(to_normalized)
        return float(converted.magnitude)
    except Exception:
        return value


def get_unit_for_coolprop_function(func_name: str) -> str:
    """
    Gibt die Einheit für eine CoolProp-Funktion zurück.

    Args:
        func_name: Name der Funktion (z.B. "enthalpy", "entropy")

    Returns:
        Einheit als String oder "" für dimensionslose Größen
    """
    func_lower = func_name.lower()
    return COOLPROP_UNITS.get(func_lower, '')


def get_unit_for_humidair_property(prop_name: str) -> str:
    """
    Gibt die Einheit für eine HumidAir-Property zurück.

    Args:
        prop_name: Name der Property (z.B. "h", "w", "T")

    Returns:
        Einheit als String oder "" für dimensionslose Größen
    """
    prop_lower = prop_name.lower()
    return HUMID_AIR_UNITS.get(prop_lower, '')


def detect_unit_from_equation(equation: str, unit_values: dict = None) -> str:
    """
    Erkennt die Einheit einer Variable basierend auf der Gleichung.

    Prüft ob die Gleichung eine Thermodynamik-Funktion enthält und
    gibt die entsprechende Einheit zurück. Falls nicht, versucht
    Einheiten durch die Berechnung zu propagieren.

    Args:
        equation: Original-Gleichung (z.B. "h = enthalpy(water, T=T, p=p)")
        unit_values: Dict mit bekannten UnitValues für Variablen

    Returns:
        Einheit als String oder "" wenn keine erkannt
    """
    import re

    eq_lower = equation.lower()

    # Prüfe CoolProp-Funktionen
    for func_name, unit in COOLPROP_UNITS.items():
        # Suche nach func_name( Pattern
        if re.search(rf'\b{func_name}\s*\(', eq_lower):
            return unit

    # Prüfe HumidAir-Funktionen
    if 'humidair' in eq_lower:
        # Extrahiere die Output-Property (erstes Argument)
        match = re.search(r'humidair\s*\(\s*([a-z_]+)', eq_lower)
        if match:
            prop = match.group(1)
            return HUMID_AIR_UNITS.get(prop, '')

    # Prüfe Strahlungs-Funktionen
    for func_name, unit in RADIATION_UNITS.items():
        if re.search(rf'\b{func_name}\s*\(', eq_lower):
            return unit

    # Versuche Einheiten-Propagation durch Berechnung
    if unit_values:
        propagated = propagate_units(equation, unit_values)
        if propagated:
            return propagated

    return ''


def propagate_units(equation: str, unit_values: dict) -> str:
    """
    Propagiert Einheiten durch eine mathematische Berechnung.

    Ersetzt Variablen durch pint Quantities und berechnet die resultierende Einheit.

    Args:
        equation: Gleichung der Form "var = ausdruck"
        unit_values: Dict mit UnitValues {var_name: UnitValue}

    Returns:
        Resultierende Einheit als String oder "" wenn nicht berechenbar
    """
    import re

    # Extrahiere rechte Seite der Gleichung
    if '=' not in equation:
        return ''

    parts = equation.split('=', 1)
    if len(parts) != 2:
        return ''

    expr = parts[1].strip()

    # Erstelle Kontext mit pint Quantities für bekannte Variablen
    context = {}
    has_units = False
    has_humidity_ratio = False  # Tracke ob kg/kg (Feuchtebeladung) vorkommt

    for var_name, uv in unit_values.items():
        if uv.original_unit:
            try:
                normalized = normalize_unit(uv.original_unit)
                # Konvertiere absolute Temperatur-Einheiten zu Delta-Einheiten für dimensionale Analyse
                # °C und °F sind Offset-Einheiten, die Probleme bei Berechnungen verursachen
                # K (Kelvin) wird ebenfalls zu delta_degC konvertiert, da 1K = 1°C Differenz
                if normalized in ('degC', 'degree_Celsius', 'celsius'):
                    normalized = 'delta_degC'
                elif normalized in ('degF', 'degree_Fahrenheit', 'fahrenheit'):
                    normalized = 'delta_degF'
                elif normalized in ('kelvin', 'K'):
                    normalized = 'delta_degC'  # 1K Differenz = 1°C Differenz
                # Verwende 1 als Wert, da wir nur die Einheit berechnen
                context[var_name] = ureg.Quantity(1.0, normalized)
                has_units = True
                # Prüfe auf Feuchtebeladung (kg/kg)
                if normalized in ['kg/kg', 'kilogram/kilogram'] or uv.original_unit == 'kg/kg':
                    has_humidity_ratio = True
            except Exception:
                context[var_name] = 1.0
        else:
            context[var_name] = 1.0

    if not has_units:
        return ''

    # Konvertiere ^ zu ** für Python-Parser
    expr = expr.replace('^', '**')

    # Ersetze Funktionsaufrufe durch dimensionslose 1
    # (enthalpy, entropy etc. werden separat behandelt)
    expr_clean = re.sub(r'\b(enthalpy|entropy|density|pressure|temperature|cp|cv|quality|volume|intenergy|humidair)\s*\([^)]+\)',
                        '1', expr, flags=re.IGNORECASE)

    # Prüfe ob alle Variablen bekannt sind - wenn nicht, keine Propagation
    # (verhindert falsche Ergebnisse durch Annahme von dimensionslos)
    var_pattern = re.compile(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b')
    builtin_funcs = {'sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs', 'pi', 'e'}
    for match in var_pattern.finditer(expr_clean):
        var = match.group(1)
        if var not in context and var not in builtin_funcs:
            # Unbekannte Variable gefunden - keine sichere Propagation möglich
            return ''

    try:
        # Sichere Auswertung
        import numpy as np
        safe_context = {
            '__builtins__': {},
            'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
            'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt,
            'abs': abs, 'pi': np.pi,
        }
        safe_context.update(context)

        result = eval(expr_clean, safe_context)

        # Prüfe ob Ergebnis eine pint Quantity ist
        if hasattr(result, 'units'):
            # Vereinfache Einheit
            result = result.to_base_units()
            unit_str = str(result.units)

            # Spezialfall: Wenn Ergebnis dimensionslos und Eingabe Feuchtebeladung enthielt
            # -> Ergebnis ist auch Feuchtebeladung (kg/kg)
            if result.dimensionless and has_humidity_ratio:
                return 'kg/kg'

            # Konvertiere zu benutzerfreundlichen Einheiten
            unit_str = _simplify_unit(unit_str, result)
            return unit_str

    except Exception:
        pass

    return ''


def _simplify_unit(unit_str: str, quantity) -> str:
    """
    Konvertiert SI-Basiseinheiten zu benutzerfreundlichen Einheiten.

    Args:
        unit_str: Einheit als String
        quantity: pint Quantity

    Returns:
        Vereinfachte Einheit
    """
    try:
        # Versuche zu bekannten Einheiten zu konvertieren
        dim = quantity.dimensionality

        # Leistung: kg*m²/s³ = W
        if dim == ureg.watt.dimensionality:
            return 'kW'

        # Energie: kg*m²/s² = J
        if dim == ureg.joule.dimensionality:
            return 'kJ'

        # Druck: kg/(m*s²) = Pa
        if dim == ureg.pascal.dimensionality:
            return 'bar'

        # Massenstrom: kg/s
        if dim == ureg('kg/s').dimensionality:
            return 'kg/s'

        # Volumenstrom: m³/s
        if dim == ureg('m^3/s').dimensionality:
            return 'm^3/s'

        # Geschwindigkeit: m/s
        if dim == ureg('m/s').dimensionality:
            return 'm/s'

        # Dichte: kg/m³
        if dim == ureg('kg/m^3').dimensionality:
            return 'kg/m^3'

        # Spezifische Enthalpie/Energie: J/kg = m²/s²
        if dim == ureg('J/kg').dimensionality:
            return 'kJ/kg'

        # Spezifische Wärme: J/(kg*K)
        if dim == ureg('J/(kg*K)').dimensionality:
            return 'kJ/(kg*K)'

        # Wärmestromdichte: W/m² = kg/s³
        if dim == ureg('W/m^2').dimensionality:
            return 'W/m^2'

        # Wärmeübergangskoeffizient: W/(m²·K)
        if dim == ureg('W/m^2/K').dimensionality:
            return 'W/m^2K'

        # Stefan-Boltzmann-Konstante: W/(m²·K⁴)
        if dim == ureg('W/m^2/K^4').dimensionality:
            return 'W/m^2K^4'

        # Spezifisches Volumen / Wärmedurchlasswiderstand: m³/kg oder m²K/W
        if dim == ureg('m^3/kg').dimensionality:
            return 'm^3/kg'

        # Wärmedurchlasswiderstand: m²K/W = K·s³/kg
        if dim == ureg('m^2*K/W').dimensionality:
            return 'm^2K/W'

        # Fläche: m²
        if dim == ureg('m^2').dimensionality:
            return 'm^2'

        # Volumen: m³
        if dim == ureg('m^3').dimensionality:
            return 'm^3'

        # Temperatur (delta)
        if dim == ureg.kelvin.dimensionality:
            return 'K'

        # Länge
        if dim == ureg.meter.dimensionality:
            return 'm'

        # Masse
        if dim == ureg.kilogram.dimensionality:
            return 'kg'

        # Zeit
        if dim == ureg.second.dimensionality:
            return 's'

        # Wenn keine bekannte Einheit, versuche in Standard-Einheit zu konvertieren
        dim_str = str(dim)
        for dim_pattern, (target_unit, pint_unit) in STANDARD_UNITS.items():
            if dim_str == dim_pattern:
                return target_unit

    except Exception:
        pass

    return unit_str


def format_value_with_unit(value: float, unit: str, precision: int = 6) -> str:
    """
    Formatiert einen Wert mit Einheit für die Anzeige.

    Args:
        value: Numerischer Wert
        unit: Einheit
        precision: Anzahl signifikanter Stellen

    Returns:
        Formatierter String
    """
    if not unit:
        return f"{value:.{precision}g}"
    return f"{value:.{precision}g} {unit}"


# Test
if __name__ == "__main__":
    print("=== Units Module Test ===\n")

    # Test parse_value_with_unit
    print("Test parse_value_with_unit:")
    test_values = ["15°C", "288.15K", "10g", "2.5kJ/kg", "4.18kJ/(kg*K)", "4.18kJ/kgK", "100", "1.5e-3bar"]
    for v in test_values:
        try:
            val, unit = parse_value_with_unit(v)
            print(f"  '{v}' -> ({val}, '{unit}')")
        except ValueError as e:
            print(f"  '{v}' -> ERROR: {e}")

    print("\nTest UnitValue.from_input:")
    uv = UnitValue.from_input(15, "°C")
    print(f"  15°C -> SI: {uv.si_value:.2f} {uv.si_unit}")
    print(f"  Convert to K: {uv.to('K'):.2f}")
    print(f"  Convert to °F: {uv.to('degF'):.2f}")

    print("\nTest UnitValue.from_si (CoolProp result):")
    h = UnitValue.from_si(2676.5, "kJ/kg")
    print(f"  2676.5 kJ/kg -> {h}")
    print(f"  Compatible units: {get_compatible_units('kJ/kg')}")

    print("\nTest normalize_unit:")
    test_units = ["kJ/kgK", "°C", "m³/s", "W/m²K"]
    for u in test_units:
        print(f"  '{u}' -> '{normalize_unit(u)}'")

    print("\nTest convert_value:")
    print(f"  100 kPa -> bar: {convert_value(100, 'kPa', 'bar')}")
    print(f"  25 °C -> K: {convert_value(25, '°C', 'K')}")
    print(f"  1000 kg/h -> kg/s: {convert_value(1000, 'kg/h', 'kg/s'):.4f}")
