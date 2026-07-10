"""
Thermodynamik-Modul für den HVAC Equation Solver

Verwendet CoolProp für Stoffdatenberechnung mit EES-ähnlicher Syntax.

Einheiten (SI-Basiseinheiten für interne Berechnungen):
- Temperatur: K
- Druck: Pa
- Dichte: kg/m³
- Spez. Volumen: m³/kg
- Spez. Enthalpie: J/kg
- Spez. Entropie: J/(kg·K)
- Spez. innere Energie: J/kg
- Dampfgehalt (Quality): - (0-1)
- Viskosität: Pa·s
- Wärmeleitfähigkeit: W/(m·K)
- Prandtl-Zahl: -

Syntax (wie EES):
    h = enthalpy(Water, T=373.15, p=100000)  # T in K, p in Pa
    rho = density(R134a, T=298.15, x=1)
    s = entropy(Water, p=1000000, h=500000)  # p in Pa, h in J/kg
"""

import CoolProp.CoolProp as CP
from typing import Dict, List, Tuple, Optional
import re


# Hinweis: Das Mapping Funktionsname -> CoolProp-Output-Property erfolgt
# direkt in calculate_property() (if/elif-Kette).

# Mapping von EES Input-Parametern zu CoolProp
INPUT_MAP = {
    't': 'T',      # Temperatur
    'p': 'P',      # Druck
    'h': 'H',      # Enthalpie
    's': 'S',      # Entropie
    'x': 'Q',      # Quality/Dampfgehalt
    'rho': 'D',    # Dichte
    'd': 'D',      # Dichte (alternativ)
    'u': 'U',      # Innere Energie
    'v': 'V_input', # Spez. Volumen (wird zu Dichte konvertiert)
}

# Umrechnungsfaktoren: Interne Einheit -> SI-Einheit (für CoolProp)
# Da wir intern jetzt SI verwenden, ist keine Konvertierung mehr nötig!
# CoolProp verwendet: Pa, J/kg, J/(kg·K), K
TO_SI = {
    'T': lambda x: x,                # K -> K
    'P': lambda x: x,                # Pa -> Pa (bereits SI)
    'H': lambda x: x,                # J/kg -> J/kg (bereits SI)
    'S': lambda x: x,                # J/(kg·K) -> J/(kg·K) (bereits SI)
    'U': lambda x: x,                # J/kg -> J/kg (bereits SI)
    'Q': lambda x: x,                # dimensionslos
    'D': lambda x: x,                # kg/m³
}

# Umrechnungsfaktoren: SI-Einheit -> Interne Einheit
# Da wir intern jetzt SI verwenden, ist keine Konvertierung mehr nötig!
FROM_SI = {
    'T': lambda x: x,                # K -> K
    'P': lambda x: x,                # Pa -> Pa (bleibt SI)
    'H': lambda x: x,                # J/kg -> J/kg (bleibt SI)
    'S': lambda x: x,                # J/(kg·K) -> J/(kg·K) (bleibt SI)
    'U': lambda x: x,                # J/kg -> J/kg (bleibt SI)
    'Q': lambda x: x,                # dimensionslos
    'D': lambda x: x,                # kg/m³
    'V': lambda x: x,                # Pa·s (Viskosität)
    'L': lambda x: x,                # W/(m·K) (Wärmeleitfähigkeit)
    'Prandtl': lambda x: x,          # dimensionslos
    'C': lambda x: x,                # J/(kg·K) -> J/(kg·K) (bleibt SI)
    'O': lambda x: x,                # J/(kg·K) -> J/(kg·K) (bleibt SI)
    'A': lambda x: x,                # m/s
}

# Bekannte Stoffnamen und ihre CoolProp-Äquivalente
FLUID_ALIASES = {
    # Wasser
    'water': 'Water',
    'steam': 'Water',
    'h2o': 'Water',
    'wasser': 'Water',

    # Luft
    'air': 'Air',
    'luft': 'Air',

    # Kältemittel
    'r134a': 'R134a',
    'r1234yf': 'R1234yf',
    'r1234ze': 'R1234ze(E)',
    'r32': 'R32',
    'r410a': 'R410A',
    'r407c': 'R407C',
    'r404a': 'R404A',
    'r507a': 'R507A',
    'r22': 'R22',
    'r12': 'R12',
    'r290': 'R290',      # Propan
    'r600a': 'R600a',    # Isobutan
    'r717': 'R717',      # Ammoniak
    'r744': 'R744',      # CO2
    'r718': 'Water',     # Wasser als Kältemittel

    # Natürliche Kältemittel
    'ammonia': 'Ammonia',
    'ammoniak': 'Ammonia',
    'nh3': 'Ammonia',
    'co2': 'CO2',
    'propane': 'Propane',
    'propan': 'Propane',
    'isobutane': 'IsoButane',
    'isobutan': 'IsoButane',

    # Andere Gase
    'nitrogen': 'Nitrogen',
    'stickstoff': 'Nitrogen',
    'n2': 'Nitrogen',
    'oxygen': 'Oxygen',
    'sauerstoff': 'Oxygen',
    'o2': 'Oxygen',
    'hydrogen': 'Hydrogen',
    'wasserstoff': 'Hydrogen',
    'h2': 'Hydrogen',
    'helium': 'Helium',
    'he': 'Helium',
    'argon': 'Argon',
    'ar': 'Argon',
    'methane': 'Methane',
    'methan': 'Methane',
    'ch4': 'Methane',
    'ethane': 'Ethane',
    'ethan': 'Ethane',
    'c2h6': 'Ethane',

    # Weitere
    'carbondioxide': 'CO2',
    'kohlendioxid': 'CO2',
}


def get_fluid_name(name: str) -> str:
    """Konvertiert einen Stoffnamen zum CoolProp-Format."""
    name_lower = name.lower().strip()
    if name_lower in FLUID_ALIASES:
        return FLUID_ALIASES[name_lower]
    # Versuche direkten CoolProp-Namen
    return name


def get_available_fluids() -> List[str]:
    """Gibt eine Liste aller verfügbaren Fluide zurück."""
    try:
        fluids = CP.get_global_param_string('fluids_list').split(',')
        return sorted(fluids)
    except:
        return []


def get_fluid_info() -> Dict[str, List[str]]:
    """Gibt kategorisierte Fluid-Informationen zurück."""
    return {
        'Wasser/Dampf': ['Water (water, steam, h2o)'],
        'Luft': ['Air (air, luft)'],
        'Kältemittel (HFCs)': [
            'R134a', 'R32', 'R410A', 'R407C', 'R404A', 'R507A'
        ],
        'Kältemittel (HFOs)': [
            'R1234yf', 'R1234ze(E)'
        ],
        'Natürliche Kältemittel': [
            'R717/Ammonia (ammonia, nh3)',
            'R744/CO2 (co2)',
            'R290/Propane (propane, propan)',
            'R600a/IsoButane (isobutane, isobutan)',
        ],
        'Gase': [
            'Nitrogen (n2, stickstoff)',
            'Oxygen (o2, sauerstoff)',
            'Hydrogen (h2, wasserstoff)',
            'Helium (he)',
            'Argon (ar)',
            'Methane (ch4, methan)',
        ],
    }


def calculate_property(func_name: str, fluid: str, **kwargs) -> float:
    """
    Berechnet eine thermodynamische Eigenschaft.

    Args:
        func_name: Name der Funktion (enthalpy, entropy, etc.)
        fluid: Name des Fluids
        **kwargs: Zustandsgrößen (T, p, h, s, x, etc.)

    Returns:
        Berechneter Wert in EES-Einheiten
    """
    func_name = func_name.lower()

    # Bestimme Output-Property
    if func_name == 'enthalpy':
        output_prop = 'H'
    elif func_name == 'entropy':
        output_prop = 'S'
    elif func_name == 'density':
        output_prop = 'D'
    elif func_name == 'volume':
        output_prop = 'D'  # Berechne 1/D später
    elif func_name == 'intenergy':
        output_prop = 'U'
    elif func_name == 'quality':
        output_prop = 'Q'
    elif func_name == 'temperature':
        output_prop = 'T'
    elif func_name == 'pressure':
        output_prop = 'P'
    elif func_name == 'viscosity':
        output_prop = 'V'
    elif func_name == 'conductivity':
        output_prop = 'L'
    elif func_name == 'prandtl':
        output_prop = 'Prandtl'
    elif func_name == 'cp':
        output_prop = 'C'
    elif func_name == 'cv':
        output_prop = 'O'
    elif func_name == 'soundspeed':
        output_prop = 'A'
    else:
        raise ValueError(f"Unbekannte Funktion: {func_name}")

    # Konvertiere Fluid-Namen
    coolprop_fluid = get_fluid_name(fluid)

    # Parse Input-Parameter
    inputs = {}
    for key, value in kwargs.items():
        key_lower = key.lower()
        if key_lower in INPUT_MAP:
            cp_key = INPUT_MAP[key_lower]

            # Begrenze Werte auf gültige Bereiche (wichtig für iterative Löser)
            if cp_key == 'Q':  # Dampfqualität muss zwischen 0 und 1 sein
                value = max(0.0, min(1.0, float(value)))

            # Spezialfall: Volumen -> Dichte umrechnen
            if cp_key == 'V_input':
                # v [m³/kg] -> rho [kg/m³] = 1/v
                cp_key = 'D'
                value = 1.0 / float(value)

            # Konvertiere zu SI
            if cp_key in TO_SI:
                inputs[cp_key] = TO_SI[cp_key](value)
            else:
                inputs[cp_key] = value
        else:
            raise ValueError(f"Unbekannter Parameter: {key}")

    # Brauchen genau 2 unabhängige Zustandsgrößen
    if len(inputs) != 2:
        raise ValueError(f"Genau 2 Zustandsgrößen erforderlich, {len(inputs)} gegeben")

    # Erstelle CoolProp-Aufruf
    keys = list(inputs.keys())
    values = list(inputs.values())

    try:
        result_si = CP.PropsSI(output_prop, keys[0], values[0], keys[1], values[1], coolprop_fluid)
    except Exception as e:
        raise ValueError(f"CoolProp Fehler für {coolprop_fluid}: {e}")

    # Konvertiere von SI zu EES-Einheiten
    if func_name == 'volume':
        # Spez. Volumen = 1 / Dichte
        result = 1.0 / result_si
    elif output_prop in FROM_SI:
        result = FROM_SI[output_prop](result_si)
    else:
        result = result_si

    return result


# Erstelle Wrapper-Funktionen für den Solver
def enthalpy(fluid: str, **kwargs) -> float:
    """Spez. Enthalpie [J/kg]"""
    return calculate_property('enthalpy', fluid, **kwargs)

def entropy(fluid: str, **kwargs) -> float:
    """Spez. Entropie [J/(kg·K)]"""
    return calculate_property('entropy', fluid, **kwargs)

def density(fluid: str, **kwargs) -> float:
    """Dichte [kg/m³]"""
    return calculate_property('density', fluid, **kwargs)

def volume(fluid: str, **kwargs) -> float:
    """Spez. Volumen [m³/kg]"""
    return calculate_property('volume', fluid, **kwargs)

def intenergy(fluid: str, **kwargs) -> float:
    """Spez. innere Energie [J/kg]"""
    return calculate_property('intenergy', fluid, **kwargs)

def quality(fluid: str, **kwargs) -> float:
    """Dampfgehalt/Quality [-]"""
    return calculate_property('quality', fluid, **kwargs)

def temperature(fluid: str, **kwargs) -> float:
    """Temperatur [K]"""
    return calculate_property('temperature', fluid, **kwargs)

def pressure(fluid: str, **kwargs) -> float:
    """Druck [Pa]"""
    return calculate_property('pressure', fluid, **kwargs)

def viscosity(fluid: str, **kwargs) -> float:
    """Dynamische Viskosität [Pa·s]"""
    return calculate_property('viscosity', fluid, **kwargs)

def conductivity(fluid: str, **kwargs) -> float:
    """Wärmeleitfähigkeit [W/(m·K)]"""
    return calculate_property('conductivity', fluid, **kwargs)

def prandtl(fluid: str, **kwargs) -> float:
    """Prandtl-Zahl [-]"""
    return calculate_property('prandtl', fluid, **kwargs)

def cp(fluid: str, **kwargs) -> float:
    """Spez. Wärmekapazität bei konst. Druck [J/(kg·K)]"""
    return calculate_property('cp', fluid, **kwargs)

def cv(fluid: str, **kwargs) -> float:
    """Spez. Wärmekapazität bei konst. Volumen [J/(kg·K)]"""
    return calculate_property('cv', fluid, **kwargs)

def soundspeed(fluid: str, **kwargs) -> float:
    """Schallgeschwindigkeit [m/s]"""
    return calculate_property('soundspeed', fluid, **kwargs)


# Dictionary aller Thermodynamik-Funktionen für den Solver
THERMO_FUNCTIONS = {
    'enthalpy': enthalpy,
    'entropy': entropy,
    'density': density,
    'volume': volume,
    'intenergy': intenergy,
    'quality': quality,
    'temperature': temperature,
    'pressure': pressure,
    'viscosity': viscosity,
    'conductivity': conductivity,
    'prandtl': prandtl,
    'cp': cp,
    'cv': cv,
    'soundspeed': soundspeed,
}


if __name__ == "__main__":
    # Tests (alle Eingaben in SI-Basiseinheiten: T in K, p in Pa)
    print("=== Thermodynamik-Modul Tests (SI-Einheiten) ===\n")

    # Test 1: Wasser bei 373.15 K (100°C), 100000 Pa (1 bar)
    print("Test 1: Wasser bei T=373.15 K (100°C), p=100000 Pa (1 bar)")
    h = enthalpy('water', T=373.15, p=100000)
    s = entropy('water', T=373.15, p=100000)
    rho = density('water', T=373.15, p=100000)
    print(f"  h = {h:.1f} J/kg ({h/1000:.2f} kJ/kg)")
    print(f"  s = {s:.2f} J/(kg·K) ({s/1000:.4f} kJ/(kg·K))")
    print(f"  rho = {rho:.2f} kg/m³")
    print()

    # Test 2: Sattdampf (x=1) bei 373.15 K (100°C)
    print("Test 2: Sattdampf bei T=373.15 K (100°C), x=1")
    h = enthalpy('water', T=373.15, x=1)
    p = pressure('water', T=373.15, x=1)
    print(f"  h = {h:.1f} J/kg ({h/1000:.2f} kJ/kg)")
    print(f"  p = {p:.1f} Pa ({p/1e5:.4f} bar)")
    print()

    # Test 3: R134a
    print("Test 3: R134a bei T=298.15 K (25°C), x=1 (Sattdampf)")
    h = enthalpy('R134a', T=298.15, x=1)
    p = pressure('R134a', T=298.15, x=1)
    rho = density('R134a', T=298.15, x=1)
    print(f"  h = {h:.1f} J/kg ({h/1000:.2f} kJ/kg)")
    print(f"  p = {p:.1f} Pa ({p/1e5:.2f} bar)")
    print(f"  rho = {rho:.2f} kg/m³")
    print()

    # Test 4: Transporteigenschaften
    print("Test 4: Wasser bei T=323.15 K (50°C), p=100000 Pa (Transporteigenschaften)")
    mu = viscosity('water', T=323.15, p=100000)
    k = conductivity('water', T=323.15, p=100000)
    pr = prandtl('water', T=323.15, p=100000)
    print(f"  Viskosität = {mu:.6f} Pa·s")
    print(f"  Wärmeleitfähigkeit = {k:.4f} W/(m·K)")
    print(f"  Prandtl = {pr:.2f}")
