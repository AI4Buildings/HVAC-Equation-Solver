"""
Einheiten-Modul für den HVAC Equation Solver

Verwendet pint für Einheiten-Konvertierung und -Verwaltung.

Features:
- Parsing von Werten mit Einheiten: "15°C", "10g", "2.5kJ/kg"
- Hybrid-Speicherung: SI intern, Original-Einheit merken
- Konvertierung zwischen kompatiblen Einheiten
- Automatische Einheiten für CoolProp-Funktionen
"""

import math
import re
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional, Any
import functools
import pint

# Globale Unit Registry
ureg = pint.UnitRegistry()

# Zusätzliche Einheiten-Definitionen für HVAC
ureg.define('degC = kelvin; offset: 273.15 = °C = celsius')
ureg.define('degF = 5/9 * kelvin; offset: 233.15 + 200 / 9 = °F = fahrenheit')   # exakt wie pint

# Temperaturdifferenz-Einheit (delta_K = 1 Kelvin Differenz, identisch mit delta_degC)
# Verwendet für Variablen wie dT, delta_T, etc.
ureg.define('delta_K = delta_degC')

# Aliase für häufige Einheiten
ureg.define('@alias bar = Bar')
ureg.define('@alias pascal = Pa')
ureg.define('@alias joule = J')
ureg.define('@alias watt = W')
ureg.define('@alias kilogram = kg')
ureg.define('@alias meter = m')
ureg.define('@alias second = s')
ureg.define('@alias liter = L = l')


# Einheiten-Mapping für CoolProp-Funktionen (Ausgabe-Einheiten, alle in SI)
COOLPROP_UNITS = {
    'enthalpy': 'J/kg',       # SI: J/kg (CoolProp liefert J/kg)
    'entropy': 'J/(kg*K)',    # SI: J/(kg·K)
    'density': 'kg/m^3',
    'temperature': 'K',
    'pressure': 'Pa',         # SI: Pa (CoolProp liefert Pa)
    'cp': 'J/(kg*K)',         # SI: J/(kg·K)
    'cv': 'J/(kg*K)',         # SI: J/(kg·K)
    'viscosity': 'Pa*s',
    'conductivity': 'W/(m*K)',
    'volume': 'm^3/kg',
    'intenergy': 'J/kg',      # SI: J/kg
    'quality': '',  # dimensionslos
    'prandtl': '',  # dimensionslos
    'soundspeed': 'm/s',
}

# Einheiten für HumidAir-Funktionen (alle in SI)
HUMID_AIR_UNITS = {
    'h': 'J/kg',              # SI: J/kg (CoolProp HAPropsSI liefert J/kg)
    'w': 'kg/kg',
    'rh': '',  # dimensionslos (0-1)
    't': 'K',
    't_dp': 'K',
    't_wb': 'K',
    'rho_tot': 'kg/m^3',
    'rho_a': 'kg/m^3',
    'rho_w': 'kg/m^3',
    'p_w': 'Pa',              # SI: Pa
    'cp': 'J/(kg*K)',         # spez. Wärmekapazität je kg trockene Luft
    'cp_ha': 'J/(kg*K)',      # spez. Wärmekapazität je kg feuchte Luft
}

# Einheiten für Strahlungs-Funktionen (ANZEIGE-Einheiten; die Funktionen
# liefern intern SI: Eb in W/m³, Wien in m - gleiche Dimension wie das Label)
RADIATION_UNITS = {
    'eb': 'W/(m^2*um)',              # Spektrale Emissionsleistung, intern W/m³
    'blackbody': '',                  # Anteil (dimensionslos, 0-1)
    'blackbody_cumulative': '',       # Kumulativer Anteil (dimensionslos, 0-1)
    'wien': 'um',                     # Wellenlänge maximaler Emission, intern m
    'stefan_boltzmann': 'W/m^2',      # Gesamtemission [W/m²]
}

# Kohärente SI-Einheiten für interne Berechnungen: (pint-Einheit, Label).
# Die Auswahl erfolgt über die DIMENSION; bei gleicher Dimension gewinnt der
# erste Eintrag (z.B. J/kg vor m²/s², Pa vor J/m³, W/m³ als SI-Form von Eb).
# Alle Größen in SI, damit Einheiten-Arithmetik korrekt funktioniert:
#   p*v = R*T  →  Pa * m³/kg = J/(kg·K) * K  →  konsistent!
STANDARD_UNITS = [
    ('kelvin', 'K'),                     # Temperatur
    ('kg/s', 'kg/s'),                    # Massenstrom
    ('m^3/s', 'm^3/s'),                  # Volumenstrom
    ('kg/m^3', 'kg/m^3'),                # Dichte
    ('Pa', 'Pa'),                        # Druck
    ('J/kg', 'J/kg'),                    # Spez. Energie / Enthalpie
    ('J/(kg*K)', 'J/(kg*K)'),            # Spez. Wärme / Entropie
    ('W', 'W'),                          # Leistung
    ('J', 'J'),                          # Energie
    ('N', 'N'),                          # Kraft
    ('m/s^2', 'm/s^2'),                  # Beschleunigung
    ('kg', 'kg'),                        # Masse
    ('m', 'm'),                          # Länge (auch µm, nm, mm -> m)
    ('m^2', 'm^2'),                      # Fläche
    ('m^3', 'm^3'),                      # Volumen
    ('s', 's'),                          # Zeit
    ('1/s', '1/s'),                      # Frequenz
    ('m/s', 'm/s'),                      # Geschwindigkeit
    ('m^3/kg', 'm^3/kg'),                # Spez. Volumen
    ('Pa*s', 'Pa*s'),                    # Dynamische Viskosität
    ('m^2/s', 'm^2/s'),                  # Kinematische Viskosität, Temperaturleitfähigkeit
    ('W/m^2', 'W/m^2'),                  # Wärmestromdichte
    ('W/m^3', 'W/m^3'),                  # Spektrale Emission Eb (SI), Leistungsdichte
    ('W/(m^2*K)', 'W/(m^2*K)'),          # Wärmeübergangskoeffizient, U-Wert
    ('W/(m*K)', 'W/(m*K)'),              # Wärmeleitfähigkeit
    ('W/(m^2*K^4)', 'W/(m^2*K^4)'),      # Stefan-Boltzmann
    ('m^2*K/W', 'm^2*K/W'),              # Wärmedurchlasswiderstand
    ('K/W', 'K/W'),                      # Thermischer Widerstand
    ('W/K', 'W/K'),                      # Wärmedurchgangsfähigkeit UA
    ('J/K', 'J/K'),                      # Wärmekapazität, Entropie
    ('1/K', '1/K'),                      # Ausdehnungskoeffizient
    ('kg/(m^2*s)', 'kg/(m^2*s)'),        # Massenstromdichte
    ('1/m', '1/m'),                      # Rippenparameter m
]

# Typische Startwerte (in SI!) für die automatische Initialisierung,
# exakte Treffer auf das Einheiten-Label
UNIT_TYPICAL_VALUES = {
    'K': 350.0,                  # ~77°C - typische HVAC-Temperatur
    'delta_K': 10.0,             # Temperaturdifferenz
    'Pa': 500000.0,              # 5 bar - typischer HVAC-Druck
    'J/kg': 1000000.0,           # 1000 kJ/kg - typische Enthalpie
    'J/(kg*K)': 3000.0,          # 3 kJ/(kg·K) - typische spez. Wärmekapazität
    'kg/kg': 0.01,               # Feuchtebeladung
    'um': 5e-6,                  # Wellenlänge 5 µm (intern in m)
    'W/(m^2*um)': 1e9,           # Spektrale Emission, intern W/m³ (= 1000 W/(m²·µm))
}

# Typische Startwerte (in SI) nach Dimension, für alle anderen Labels
# (z.B. 'kW', 'kJ/kg', 'kilogram / second', 'W/m^2K')
_TYPICAL_BY_DIMENSION = [
    ('kelvin', 350.0),           # Temperatur
    ('Pa', 500000.0),            # Druck
    ('J/kg', 1000000.0),         # Spezifische Energie
    ('J/(kg*K)', 3000.0),        # Spezifische Wärme
    ('kg/s', 1.0),               # Massenstrom
    ('W', 100000.0),             # Leistung (100 kW)
    ('J', 100000.0),             # Energie (100 kJ)
    ('m^3/s', 0.1),              # Volumenstrom
    ('kg/m^3', 1.0),             # Dichte (Luft)
    ('m^3/kg', 0.5),             # Spez. Volumen
    ('Pa*s', 0.001),             # Dynamische Viskosität
    ('m^2/s', 1e-5),             # Kinematische Viskosität (Luft ~1.5e-5)
    ('W/(m*K)', 0.5),            # Wärmeleitfähigkeit
    ('m/s', 300.0),              # Schallgeschwindigkeit
    ('m', 1.0),                  # Länge
    ('m^2', 1.0),                # Fläche
    ('m^3', 1.0),                # Volumen
    ('N', 100.0),                # Kraft
    ('m/s^2', 9.81),             # Beschleunigung
    ('W/m^2', 1000.0),           # Wärmestromdichte
    ('W/m^3', 1e9),              # Spektrale Emission (SI)
    ('W/(m^2*K)', 10.0),         # Wärmeübergangskoeffizient
    ('W/(m^2*K^4)', 5.67e-8),    # Stefan-Boltzmann
    ('m^2*K/W', 0.1),            # Wärmedurchlasswiderstand
    ('K/W', 0.1),                # Thermischer Widerstand
    ('W/K', 100.0),              # UA-Wert
    ('J/K', 1000.0),             # Wärmekapazität
    ('1/K', 3.4e-3),             # Ausdehnungskoeffizient (ideales Gas ~1/T)
    ('s', 1.0),                  # Zeit
    ('1/s', 1.0),                # Frequenz
    ('kg', 1.0),                 # Masse
]

_DIMENSIONALITY_CACHE = {}


def _dimensionality(pint_unit: str):
    """Dimension einer pint-Einheit (gecacht)."""
    if pint_unit not in _DIMENSIONALITY_CACHE:
        _DIMENSIONALITY_CACHE[pint_unit] = ureg.Quantity(1.0, pint_unit).dimensionality
    return _DIMENSIONALITY_CACHE[pint_unit]


def is_absolute_temperature_unit(unit_str: Optional[str]) -> bool:
    """True für absolute Temperatureinheiten (K, °C, °F), False für delta_K & Co."""
    if not unit_str or 'delta' in unit_str.lower():
        return False
    try:
        quantity = ureg.Quantity(1.0, normalize_unit(unit_str.strip()))
    except Exception:
        return False
    return quantity.dimensionality == ureg.kelvin.dimensionality


def initial_values_from_units(variables, all_units: Dict[str, Optional[str]],
                              known_values: Dict[str, float]) -> Dict[str, float]:
    """
    Startwerte (SI) für Variablen aus ihren Einheiten - generisch, nicht namensabhängig.

    Absolute Temperaturen starten beim Mittelwert der vorgegebenen Temperaturen der
    Aufgabe statt pauschal bei 350 K: liegen alle Temperaturen z.B. zwischen -10 °C
    und 25 °C, gäbe 350 K Temperaturdifferenzen das falsche Vorzeichen
    (Ra ~ T_Raum - T_Scheibe < 0 -> Ra^(1/6) nicht reell -> keine Suchrichtung).
    """
    temperatures = [float(value) for name, value in known_values.items()
                    if isinstance(value, (int, float)) and math.isfinite(value) and value > 0
                    and is_absolute_temperature_unit(all_units.get(name))]
    mean_temperature = sum(temperatures) / len(temperatures) if temperatures else None
    result = {}
    for var in variables:
        unit = all_units.get(var)
        if unit is None:
            continue
        if mean_temperature is not None and is_absolute_temperature_unit(unit):
            result[var] = mean_temperature
        else:
            result[var] = get_initial_from_unit(unit)
    return result


def get_initial_from_unit(unit_str: str) -> float:
    """
    Liefert einen typischen Startwert (in SI) basierend auf der Einheit.

    Diese Funktion ist generisch und hängt NICHT von Variablennamen ab.
    Stattdessen wird die physikalische Größe aus der Einheit abgeleitet -
    über die DIMENSION, nicht über Teilstrings (sonst würde z.B. 'kg/s' oder
    '1/kelvin' wegen des enthaltenen 'k' den Temperatur-Startwert 350 bekommen).

    Args:
        unit_str: Einheit als String (z.B. 'K', 'Pa', 'J/kg', 'kW')

    Returns:
        Typischer Startwert für diese Einheit in SI

    Examples:
        >>> get_initial_from_unit('K')
        350.0
        >>> get_initial_from_unit('kW')
        100000.0
        >>> get_initial_from_unit('')  # dimensionslos
        0.5
    """
    if not unit_str or unit_str in ('dimensionless', '???'):
        # Dimensionslose Größen: Wirkungsgrad, Qualität, relative Feuchte
        return 0.5

    unit_normalized = unit_str.strip()

    # Exakter Treffer
    if unit_normalized in UNIT_TYPICAL_VALUES:
        return UNIT_TYPICAL_VALUES[unit_normalized]

    # Temperaturdifferenz (delta_K, delta_degC, ...): Größenordnung 10 K, nicht 350 K
    if 'delta' in unit_normalized.lower():
        return 10.0

    # Dimensionsanalyse mit pint
    try:
        quantity = ureg.Quantity(1.0, normalize_unit(unit_normalized))
    except Exception:
        return 1.0

    if quantity.dimensionless:
        # Feuchtebeladung (kg/kg, g/kg) ~ 0.01, sonst Wirkungsgrad/Qualität/rel. Feuchte
        return 0.01 if 'gram' in str(quantity.units) else 0.5

    for pint_unit, value in _TYPICAL_BY_DIMENSION:
        if quantity.dimensionality == _dimensionality(pint_unit):
            return value

    # Fallback: 1.0 ist ein neutraler Startwert
    return 1.0


# Kompatible Einheiten für Dropdown-Menüs
# Dimensionslose Anzeige: reine Zahl, Prozent, Promille, Massenverhältnis (Wassergehalt)
DIMENSIONLESS_UNITS = ['-', '%', '‰', 'g/kg']

COMPATIBLE_UNITS = {
    '%': DIMENSIONLESS_UNITS,
    '‰': DIMENSIONLESS_UNITS,
    # Temperatur; eine Differenz nur in K - °C/°F sind immer absolut (Nullpunkt)
    'delta_K': ['K'],
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
    'm': ['m', 'cm', 'mm', 'um', 'nm', 'km', 'inch', 'ft'],
    'cm': ['cm', 'm', 'mm', 'inch'],
    'mm': ['mm', 'cm', 'm', 'um', 'inch'],
    'um': ['um', 'nm', 'mm', 'm'],
    'micrometer': ['um', 'nm', 'mm', 'm'],
    'nm': ['nm', 'um', 'mm', 'm'],
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

    # Energie je Länge, Fläche, Volumen (gleiche Dimension wie N, N/m, Pa - andere Größenart)
    'kWh/m': ['kWh/m', 'Wh/m', 'kJ/m', 'MJ/m'],
    'kJ/m': ['kJ/m', 'J/m', 'kWh/m', 'MJ/m'],
    'kWh/m^2': ['kWh/m^2', 'Wh/m^2', 'kJ/m^2', 'MJ/m^2'],
    'kJ/m^2': ['kJ/m^2', 'J/m^2', 'kWh/m^2', 'MJ/m^2'],
    'kWh/m^3': ['kWh/m^3', 'Wh/m^3', 'kJ/m^3', 'MJ/m^3'],
    'kJ/m^3': ['kJ/m^3', 'J/m^3', 'MJ/m^3', 'kWh/m^3'],
    'N': ['N', 'kN'],
    'kN': ['kN', 'N'],

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
    '1/s': ['1/s', '1/h', '1/min'],      # Frequenz, Luftwechselzahl
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
# Matches: "15", "15.5", "-3.14", "1.5e-3", "15°C", "100kJ/kg", "4.18kJ/(kg*K)", "89.2 %"
_UNIT_BODY = r'(?:[a-zA-Z0-9²³µ°/*()·⋅%‰]|\^-?)*'   # Zeichen einer Einheit, auch h^-1
VALUE_WITH_UNIT_PATTERN = re.compile(
    r'^'
    r'(-?\d+\.?\d*(?:[eE][+-]?\d+)?)'  # Zahl (inkl. wissenschaftliche Notation)
    r'(?:'
    # Einheit, beginnend mit Buchstabe/°/µ (auch W/(m²·K), h^-1). Darf nicht mit
    # einem Exponenten beginnen: sonst zerlegt Backtracking "1e5*y" in 1 und e5*y
    r'\s*((?![eE][+-]?\d)°?[a-zA-Z²³µ%‰]' + _UNIT_BODY + r')'
    # oder Kehrwert-Einheit "1/h", "1/K" - nur NACH einem Leerzeichen
    # (sonst wäre "0.51/h" mehrdeutig)
    r'|\s+(1/(?:\(|°?[a-zA-Z²³µ])' + _UNIT_BODY + r')'
    r')?'
    r'$'
)

def _convert_to_standard(quantity) -> Tuple[float, str]:
    """
    Konvertiert eine pint Quantity in die kohärente SI-Einheit für Berechnungen.

    Jede Einheit wird nach SI umgerechnet - auch solche ohne benannte
    Standard-Einheit (dann in pint-Basiseinheiten). Früher wurde bei unbekannter
    Dimension der Original-Zahlenwert behalten (50 cm² -> 50, 2 L -> 2,
    1 kW/m² -> 1), was Rechenergebnisse um Zehnerpotenzen verfälschte.

    Args:
        quantity: pint Quantity

    Returns:
        (wert, einheit) in SI; einheit ist ein Label wie 'K', 'W/(m^2*K)', 'delta_K'
    """
    try:
        units_str = str(quantity.units)

        # Temperaturdifferenz: bleibt Differenz (kein Offset), Label delta_K. Zusammengesetzte
        # Einheiten mit °C/°F (°C/h, 1/°C, W/(m²°C)) rechnet pint mit der Differenz: unten nach SI
        if 'delta_' in units_str and quantity.dimensionality == _dimensionality('kelvin'):
            return (float(quantity.to('delta_degC').magnitude), 'delta_K')

        # Winkel: intern Grad, wie die Winkelfunktionen (sin(30) = 0.5); 0.5236 rad -> 30
        if quantity.dimensionless and units_str in _ANGLE_UNITS:
            return (float(quantity.to('degree').magnitude), '')

        # Dimensionslos (inkl. kg/kg, g/kg): Zahlenwert ohne Einheiten-Präfix
        if quantity.dimensionless:
            label = 'kg/kg' if 'gram' in units_str else ''
            return (float(quantity.to('dimensionless').magnitude), label)

        for pint_unit, label in STANDARD_UNITS:
            if quantity.dimensionality == _dimensionality(pint_unit):
                return (float(quantity.to(pint_unit).magnitude), label)

        # Keine benannte Standard-Einheit: trotzdem nach SI (Basiseinheiten)
        base = quantity.to_base_units()
        return (float(base.magnitude), str(base.units))

    except Exception:
        # nie unumgerechnet weiterrechnen (früher: Zahlenwert der Eingabe, z.B. 2 °C/h -> 2 K/s)
        raise UnknownUnitError(f"Einheit '{quantity.units}' lässt sich nicht nach SI umrechnen") from None


# Mapping von User-freundlichen zu pint-kompatiblen Einheiten
UNIT_ALIASES = {
    # Drehzahl n = Umdrehungen pro Zeit (DIN: 1/min), u = pi*D*n - pint rechnete rpm als
    # Winkelgeschwindigkeit 2*pi/min und U als Enzym-Einheit (µmol/min)
    'rpm': '1/minute', 'U/min': '1/minute', 'rps': '1/second', 'U/s': '1/second',
    # Newtonmeter (pint läse Nm als "number_meter")
    'Nm': 'N*m', 'kNm': 'kN*m',
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

    @classmethod
    def from_si_base(cls, si_value: float, target_unit: str) -> 'UnitValue':
        """
        Erstellt UnitValue aus SI-Basiswert und konvertiert zur Ziel-Einheit.

        Dies ist für Ergebnisse aus dem Solver gedacht, die in SI-Basiseinheiten
        vorliegen (Pa, J/kg, K) und in Benutzer-Einheiten (bar, kJ/kg, °C)
        angezeigt werden sollen.

        Args:
            si_value: Wert in SI-Basiseinheit (z.B. 3000000 für Pa)
            target_unit: Ziel-Einheit für Anzeige (z.B. "bar")

        Returns:
            UnitValue mit korrekter Konvertierung
        """
        normalized_unit = normalize_unit(target_unit)

        # Bestimme SI-Basiseinheit aus der Ziel-Einheit
        try:
            # Erstelle temporäre Quantity um SI-Basis zu ermitteln
            temp_qty = 1.0 * ureg(normalized_unit)
            si_base_unit = str(temp_qty.to_base_units().units)
            if temp_qty.dimensionless and str(temp_qty.units) in _ANGLE_UNITS:
                si_base_unit = 'degree'   # Winkel sind intern Grad (wie die Winkelfunktionen)

            # Erstelle SI-Quantity
            si_quantity = si_value * ureg(si_base_unit)

            # Konvertiere zur Ziel-Einheit
            target_quantity = si_quantity.to(normalized_unit)
            target_value = float(target_quantity.magnitude)

            # Berechne calc_value/calc_unit (Standard-Einheiten wie bar, °C, kJ/kg)
            calc_value, calc_unit = _convert_to_standard(target_quantity)

            return cls(
                si_value=si_value,
                si_unit=si_base_unit,
                original_value=target_value,
                original_unit=target_unit,
                quantity=target_quantity,
                _calc_value=calc_value,
                _calc_unit=calc_unit
            )
        except Exception as e:
            # Fallback: Versuche direkte Konvertierung mit bekannten Faktoren
            factor = 1.0
            si_base = ''

            # Bekannte Konvertierungen
            if normalized_unit == 'bar':
                factor = 1e-5  # Pa -> bar
                si_base = 'pascal'
            elif normalized_unit in ['kPa', 'kilopascal']:
                factor = 1e-3  # Pa -> kPa
                si_base = 'pascal'
            elif normalized_unit in ['MPa', 'megapascal']:
                factor = 1e-6  # Pa -> MPa
                si_base = 'pascal'
            elif normalized_unit in ['degC', 'celsius']:
                # K -> °C
                target_value = si_value - 273.15
                return cls(
                    si_value=si_value,
                    si_unit='kelvin',
                    original_value=target_value,
                    original_unit=target_unit,
                    quantity=None,
                    _calc_value=target_value,
                    _calc_unit='degC'
                )
            elif normalized_unit in ['degF', 'fahrenheit']:
                # K -> °F (Offset-Konvertierung)
                target_value = si_value * 9.0 / 5.0 - 459.67
                return cls(
                    si_value=si_value,
                    si_unit='kelvin',
                    original_value=target_value,
                    original_unit=target_unit,
                    quantity=None,
                    _calc_value=target_value,
                    _calc_unit='degF'
                )
            elif 'kilojoule' in normalized_unit or normalized_unit.startswith('kJ'):
                factor = 1e-3  # J -> kJ
                si_base = 'joule / kilogram' if '/kg' in target_unit or '/kilogram' in normalized_unit else 'joule'
            elif 'joule' in normalized_unit or normalized_unit.startswith('J'):
                factor = 1.0
                si_base = 'joule / kilogram' if '/kg' in target_unit else 'joule'

            target_value = si_value * factor
            return cls(
                si_value=si_value,
                si_unit=si_base,
                original_value=target_value,
                original_unit=target_unit,
                quantity=None,
                _calc_value=target_value,
                _calc_unit=target_unit
            )

    def to(self, target_unit: str) -> float:
        """
        Konvertiert zu Ziel-Einheit.

        Args:
            target_unit: Ziel-Einheit als String

        Returns:
            Numerischer Wert in Ziel-Einheit

        Temperaturdifferenzen (calc_unit delta_K) ohne Offset: 1 K = 1 °C Differenz.
        """
        if not target_unit:
            return self.original_value if self.original_unit else self.si_value

        if self.quantity is None:
            # Berechnete Variable ohne pint quantity
            normalized = normalize_unit(target_unit)
            if self.si_unit == 'kelvin':
                if self._calc_unit == 'delta_K':
                    # ECHTE Temperaturdifferenz: 1K-Diff = 1°C-Diff (kein Offset!)
                    if normalized in ('degF', 'fahrenheit', 'degree_Fahrenheit'):
                        return self.si_value * 9.0 / 5.0
                    return self.si_value
                # Absolute Temperatur (z.B. aus from_si_base-Offset-Fallback):
                # Offset-Konvertierung anwenden - sonst würde 350 K als
                # "350 °C" angezeigt statt 76.85 °C
                if normalized in ('degC', 'celsius', 'degree_Celsius'):
                    return self.si_value - 273.15
                if normalized in ('degF', 'fahrenheit', 'degree_Fahrenheit'):
                    return self.si_value * 9.0 / 5.0 - 459.67
            return self.si_value

        try:
            normalized = normalize_unit(target_unit)

            # Temperatur-DIFFERENZ (delta_K): 1 K-Diff = 1 °C-Diff, kein Offset!
            # (pint würde delta_K -> degC sonst als absolute Temperatur behandeln)
            if (self._calc_unit == 'delta_K' or
                    'delta' in (self.original_unit or '').lower()):
                if normalized in ('degF', 'fahrenheit', 'degree_Fahrenheit'):
                    return self.si_value * 9.0 / 5.0
                if normalized in ('degC', 'celsius', 'degree_Celsius', 'K', 'kelvin', 'delta_K'):
                    return self.si_value

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


class UnknownUnitError(Exception):
    """
    Einheit wird nicht erkannt.

    Bewusst KEIN ValueError: der Parser fängt ValueError als "kein Wert mit
    Einheit" ab und würde den Fehler sonst verschlucken.
    """


def unit_value_strict(value: float, unit_str: str, context: str = '') -> 'UnitValue':
    """
    Wie UnitValue.from_input, aber eine unbekannte Einheit ist ein FEHLER.

    UnitValue.from_input fällt bei unbekannter Einheit auf "dimensionslos" mit
    unverändertem Zahlenwert zurück - für Benutzereingaben würde das still
    falsche Ergebnisse liefern.
    """
    if re.search(r'(?<![A-Za-z])Nm(\^?3|³)', unit_str):
        where = f" in: {context}" if context else ""
        raise UnknownUnitError(f"'{unit_str}': Nm³ (Normkubikmeter) ist eine Gasmenge im Normzustand, "
                               f"kein Volumen - Volumen in m³ angeben bzw. über die Dichte im "
                               f"Normzustand umrechnen{where}")
    unit_value = UnitValue.from_input(value, unit_str)
    if unit_value.quantity is None:
        where = f" in: {context}" if context else ""
        raise UnknownUnitError(f"Unbekannte Einheit '{unit_str}'{where}")
    return unit_value


def check_unit_dimension(unit_value: 'UnitValue', expected_unit: str, context: str = '') -> None:
    """
    Prüft, ob ein Wert mit Einheit die erwartete Dimension hat (z.B. T=... muss
    eine Temperatur sein). Sonst UnknownUnitError - verhindert stille Unsinns-
    Werte wie "T=20 Grad" (pint: Giga-Radiant, dimensionslos 2e10).
    """
    if unit_value.quantity is None:
        return
    try:
        expected_dim = ureg.Quantity(1.0, normalize_unit(expected_unit)).dimensionality
        actual_dim = unit_value.quantity.dimensionality
    except Exception:
        return
    if actual_dim != expected_dim:
        where = f" in: {context}" if context else ""
        raise UnknownUnitError(
            f"Einheit '{unit_value.original_unit}' passt nicht (erwartet: {expected_unit}){where}")


# ---------------------------------------------------------------------------
# Zahlenwertgleichungen: value(x, Einheit) und quantity(z, Einheit)
# Empirische Formeln gelten oft nur für Zahlenwerte in bestimmten Einheiten
# (Heizkurve in °C, h = 5.7 + 3.8*v mit v in m/s). Intern ist alles SI; diese
# Funktionen rechnen ausdrücklich in die genannte Einheit bzw. zurück - generisch
# für jede Einheit (Nullpunkt und Faktor aus pint, auch °C, °F, bar, kW, m3/h).
# Winkel sind intern Grad (Trigonometrie in Grad), daher Bezug auf Grad.
# ---------------------------------------------------------------------------
_AFFINE_CACHE: Dict[str, Tuple[float, float]] = {}
_ANGLE_UNITS = {'degree', 'radian', 'arcminute', 'arcsecond', 'gradian', 'turn', 'revolution'}


def unit_affine(unit_str: str) -> Tuple[float, float]:
    """
    (Nullpunkt, Faktor) einer Einheit bezogen auf den internen SI-Wert:
    SI = Nullpunkt + Faktor * Zahlenwert (°C: 273.15, 1; bar: 0, 1e5; °F: 255.37, 5/9).
    Unbekannte Einheit -> UnknownUnitError.
    """
    key = unit_str.strip()
    cached = _AFFINE_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        normalized = normalize_unit(key)
        one = ureg.Quantity(1.0, normalized)
        if one.dimensionless and str(one.units) in _ANGLE_UNITS:
            result = (0.0, float(one.to('degree').magnitude))
        else:
            zero_si = float(ureg.Quantity(0.0, normalized).to_base_units().magnitude)
            one_si = float(one.to_base_units().magnitude)
            result = (zero_si, one_si - zero_si)
    except Exception:
        raise UnknownUnitError(f"Unbekannte Einheit '{key}'") from None
    if result[1] == 0:
        raise UnknownUnitError(f"Einheit '{key}' hat keinen Umrechnungsfaktor")
    _AFFINE_CACHE[key] = result
    return result


def unit_number(x, unit_str: str):
    """value(x, Einheit): Zahlenwert der Größe x (SI) in der Einheit - dimensionslos."""
    zero, scale = unit_affine(unit_str)
    return (x - zero) / scale


def unit_quantity(z, unit_str: str):
    """quantity(z, Einheit): Größe (SI) aus dem Zahlenwert z in der Einheit."""
    zero, scale = unit_affine(unit_str)
    return zero + z * scale


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
    unit_str = unit_str.replace('·', '*').replace('⋅', '*')

    # Exponent ohne '^' (m2, cm2, m3/h, kg/m3, W/m2K): Ziffern direkt nach einem
    # Buchstaben sind ein Exponent. pint kennt diese Schreibweise nicht - früher
    # wurde der Wert dann stillschweigend NICHT umgerechnet (20 cm2 -> 20 m²).
    unit_str = re.sub(r'(?<=[A-Za-z])(\d+)', r'^\1', unit_str)

    # Nenner ohne Klammern: alles nach '/' (z.B. "kJ/kgK" -> "kJ/(kg*K)")
    if '/' in unit_str and '(' not in unit_str:
        parts = unit_str.split('/')
        if len(parts) == 2:
            unit_str = f"{parts[0]}/({_denominator_units(parts[1])})"

    return unit_str


@functools.lru_cache(maxsize=512)
def _known_unit(text: str) -> bool:
    try:
        ureg.parse_units(text)
        return True
    except Exception:
        return False


def _split_known_units(text: str) -> Optional[List[str]]:
    """Zerlegt 'kgK' in bekannte Einheiten ['kg', 'K'] - längster bekannter Anfang zuerst."""
    if not text:
        return []
    for end in range(len(text), 0, -1):
        if _known_unit(text[:end]):
            rest = _split_known_units(text[end:])
            if rest is not None:
                return [text[:end]] + rest
    return None


def _denominator_units(denominator: str) -> str:
    """
    Nenner ohne Klammern als Produkt. Einen Nenner, den pint als Ganzes kennt (kWh, kPa, MJ,
    degC, m^2K), liest pint selbst - Präfixe bleiben Präfixe. Nur einen unbekannten Nenner
    (kgK, cmK, m^2degC) zerlegt die Funktion in bekannte Einheiten. Feste Schreibweise:
    'mK' ist hier m·K (W/mK), nicht Millikelvin.
    """
    if denominator == 'mK':
        return 'm*K'
    if _known_unit(denominator):
        return denominator
    parts = _split_known_units(denominator)
    return '*'.join(parts) if parts else denominator


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
    unit_str = match.group(2) or match.group(3) or ""

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
    if not unit_str or unit_str == '-':
        return list(DIMENSIONLESS_UNITS)

    # Normalisiere für Lookup
    normalized = normalize_unit(unit_str)

    # Suche in COMPATIBLE_UNITS (wie geschrieben, sonst normalisiert: kJ/m^3 -> kJ/(m^3) -
    # die Dimension allein unterscheidet Größenarten wie kJ/m^3 und bar nicht)
    for key in (unit_str.strip(), normalized):
        if key in COMPATIBLE_UNITS:
            return COMPATIBLE_UNITS[key]

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

    print("\nTest Umrechnung nach SI:")
    for value, unit in ((100, 'kPa'), (25, '°C'), (1000, 'kg/h'), (0.25, 'kg/kWh'), (2, '°C/h')):
        print(f"  {value} {unit} -> {UnitValue.from_input(value, unit).calc_value:.6g}")
