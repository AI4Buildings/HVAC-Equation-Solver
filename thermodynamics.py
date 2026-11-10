"""
Thermodynamik-Modul für den HVAC Equation Solver

Verwendet CoolProp für die Stoffdatenberechnung.

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

Syntax:
    h = enthalpy(Water, T=373.15, p=100000)  # T in K, p in Pa
    rho = density(R134a, T=298.15, x=1)
    s = entropy(Water, p=1000000, h=500000)  # p in Pa, h in J/kg

Ideale Gase und kondensierte Stoffe aus der NASA-Datenbank mit dem Präfix IG:
(ideal_gas.py), z.B. enthalpy(IG:N2, T=293.15), entropy(IG:N2, T=293.15, p=1e6).
"""

import CoolProp.CoolProp as CP
import numpy as np
import ideal_gas
from contextlib import contextmanager
from functools import lru_cache
from typing import Dict, List, Tuple, Optional
import re


# Hinweis: Das Mapping Funktionsname -> CoolProp-Output-Property erfolgt
# direkt in calculate_property() (if/elif-Kette).

# Mapping der Input-Parameter zu CoolProp
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


# ---------------------------------------------------------------------------
# Bezugszustand (REFERENCE R134a IIR). CoolProp rechnet jedes Fluid in
# seinem Standard-Bezugszustand (R134a, R32, CO2, Propan: IIR; Ammoniak, Wasser
# nicht). Ein anderer Bezugszustand verschiebt h, u und s um Konstanten -
# Differenzen sowie T, p, x, rho, cp bleiben gleich.
# ---------------------------------------------------------------------------
# Name: (Eingabe, Wert, h, s, Beschreibung) der siedenden Flüssigkeit
REFERENCE_STATES = {
    'IIR': ('T', 273.15, 200e3, 1e3, 'h = 200 kJ/kg, s = 1 kJ/(kg K) bei 0 °C'),
    'ASHRAE': ('T', 233.15, 0.0, 0.0, 'h = 0, s = 0 bei -40 °C'),
    'NBP': ('P', 101325.0, 0.0, 0.0, 'h = 0, s = 0 beim Normalsiedepunkt (1 atm)'),
    'DEFAULT': None,
}
REFERENCE_ALIASES = {'iir': 'IIR', 'ashrae': 'ASHRAE', 'ash': 'ASHRAE', 'nbp': 'NBP',
                     'default': 'DEFAULT', 'def': 'DEFAULT', 'dft': 'DEFAULT'}

# Verschiebung (dh, ds) und Name des Bezugszustands je Fluid (CoolProp-Name), je Lösungslauf
_reference_offsets: Dict[str, Tuple[float, float]] = {}
_reference_names: Dict[str, str] = {}


@lru_cache(maxsize=None)
def canonical_fluid(name: str) -> str:
    """Eindeutiger CoolProp-Name (R717 und ammonia -> Ammonia)."""
    cp_name = get_fluid_name(name)
    try:
        return CP.get_fluid_param_string(cp_name, 'name')
    except Exception:
        return cp_name


def reference_offsets(fluid: str, reference: str) -> Tuple[float, float]:
    """
    Verschiebung (dh, ds) in J/kg bzw. J/(kg K) vom CoolProp-Standard zum
    Bezugszustand IIR, ASHRAE, NBP (DEFAULT: keine). Fehler, wenn der
    Bezugspunkt nicht im Nassdampfgebiet des Fluids liegt.
    """
    key = REFERENCE_ALIASES.get(reference.lower())
    if key is None:
        raise ValueError(f"Unbekannter Bezugszustand '{reference}' - möglich: IIR, ASHRAE, NBP, DEFAULT")
    if REFERENCE_STATES[key] is None:
        return 0.0, 0.0
    input_key, value, h_ref, s_ref, text = REFERENCE_STATES[key]
    cp_name = canonical_fluid(fluid)
    try:
        if input_key == 'T':
            low, high = CP.PropsSI('Ttriple', cp_name), CP.PropsSI('Tcrit', cp_name)
        else:
            low, high = CP.PropsSI('ptriple', cp_name), CP.PropsSI('pcrit', cp_name)
        inside = low <= value < high
        if inside:
            h = CP.PropsSI('H', input_key, value, 'Q', 0, cp_name)
            s = CP.PropsSI('S', input_key, value, 'Q', 0, cp_name)
    except Exception:
        low = high = None
        inside = False
    if not inside:
        where = ''
        if low is not None:
            fmt = ((lambda t: f"{t - 273.15:.4g} °C") if input_key == 'T' else (lambda p: f"{p / 1e5:.4g} bar"))
            where = (f" (unter dem Tripelpunkt {fmt(low)})" if value < low
                     else f" (über dem kritischen Punkt {fmt(high)})")
        raise ValueError(f"Bezugszustand {key} ({text}) für {fluid} nicht möglich - "
                         f"der Bezugspunkt liegt nicht im Nassdampfgebiet{where}")
    return h_ref - h, s_ref - s


def set_reference_states(states: Dict[str, str]) -> None:
    """Bezugszustände {Fluid: IIR/ASHRAE/NBP/DEFAULT} für die folgenden Berechnungen."""
    offsets, names = {}, {}
    for fluid, reference in states.items():
        dh, ds = reference_offsets(fluid, reference)
        if dh or ds:
            offsets[canonical_fluid(fluid)] = (dh, ds)
            names[canonical_fluid(fluid)] = REFERENCE_ALIASES[reference.lower()]
    _reference_offsets.clear()
    _reference_offsets.update(offsets)
    _reference_names.clear()
    _reference_names.update(names)


# Zustände unter dem Tripelpunkt (Festlegung des Nutzers 10.11.2026: "wichtig ist, dass es richtig ist,
# und wenn nicht, dass es einen Hinweis gibt"). Im Gleichgewicht ist der Stoff dort Dampf oder fest:
# - Dampf (p höchstens der Sublimationsdruck): richtig gerechnet. CoolProp nimmt unter der
#   Tripeltemperatur T und p nicht an, wohl aber T und rho - die Dichte wird so bestimmt, dass der
#   Druck stimmt (_vapour_density).
# - Sättigung (x gegeben): Druck und Temperatur liegen auf der Sublimationslinie, dem Gleichgewicht
#   von Dampf und fester Phase - Wasser bei -10 °C: 259.9 Pa über Eis, nicht die 286 Pa der über den
#   Tripelpunkt hinaus verlängerten Siedelinie (unterkühltes Wasser, +10 %; bei -20 °C +22 %).
#   Sattdampf (x = 1) wie der Dampf oben. Daten der festen Phase führt CoolProp nur für Wasser
#   (Eis nach IAPWS, dieselben wie in HumidAir) - _sublimation_pressure.
# - Was die kondensierte Phase braucht (Flüssigkeitsgebiet, x < 1), hat keine richtigen Daten: geliefert
#   wird die unterkühlte Flüssigkeit (metastabil), solange sie thermodynamisch stabil gerechnet wird
#   (cp > 0, (dp/drho)_T > 0, nicht unter _metastable_limit) - die Oberfläche gibt dazu einen Hinweis
#   (collecting_supercooled). Weiter darunter und im Gebiet der festen Phase eine Meldung.
# Toleranz relativ: 0 °C gilt bei Wasser (Tripelpunkt 0.01 °C) nicht als "unter dem Tripelpunkt".
_TRIPLE_TOLERANCE = 1e-4

# nur in einer Phase definiert: im Nassdampfgebiet lieferte CoolProp Zahlen ohne Bedeutung
# (Wasser 100 °C: cp = 11.8 kJ/(kg K) bei x = 0.3, 2.65 bei x = 0.5)
_SINGLE_PHASE_ONLY = {'C': 'cp', 'O': 'cv', 'V': 'viscosity', 'L': 'conductivity',
                      'Prandtl': 'prandtl', 'A': 'soundspeed'}


@lru_cache(maxsize=None)
def _triple_point(coolprop_fluid: str) -> Tuple[Optional[float], Optional[float]]:
    """(Tripeltemperatur K, Tripeldruck Pa) aus CoolProp; None, wo nicht verfügbar (Gemische)."""
    result = []
    for key in ('Ttriple', 'ptriple'):
        try:
            value = float(CP.PropsSI(key, coolprop_fluid))
            result.append(value if np.isfinite(value) and value > 0 else None)
        except Exception:
            result.append(None)
    return result[0], result[1]


@lru_cache(maxsize=None)
def _has_solid_data(coolprop_fluid: str) -> bool:
    """Führt die Datenbank die feste Phase des Stoffs? CoolProp nur für Wasser (Eis Ih, IAPWS)."""
    try:
        return canonical_fluid(coolprop_fluid) == canonical_fluid('Water')
    except Exception:
        return False


def _sublimation_pressure(coolprop_fluid: str, temperature: float) -> Optional[float]:
    """Sublimationsdruck (Pa) bei T unter dem Tripelpunkt - Dampf im Gleichgewicht mit der festen
    Phase; None ohne Daten der festen Phase bzw. außerhalb ihres Bereichs."""
    if not _has_solid_data(coolprop_fluid):
        return None
    try:
        from CoolProp.HumidAirProp import HAProps_Aux
        value = float(HAProps_Aux('p_ws', float(temperature), 101325.0, 0.0)[0])
    except Exception:
        return None
    return value if np.isfinite(value) and value > 0 else None


def _sublimation_temperature(coolprop_fluid: str, pressure: float) -> Optional[float]:
    """Temperatur (K) auf der Sublimationslinie zum Druck p unter dem Tripeldruck (Reifpunkt), sonst None."""
    t_triple, _ = _triple_point(coolprop_fluid)
    if t_triple is None or not _has_solid_data(coolprop_fluid) or not pressure > 0:
        return None
    low, high = 0.4 * t_triple, t_triple
    p_low, p_high = _sublimation_pressure(coolprop_fluid, low), _sublimation_pressure(coolprop_fluid, high)
    if p_low is None or p_high is None or not p_low <= pressure <= p_high:
        return None
    from scipy.optimize import brentq
    return float(brentq(lambda T: np.log(_sublimation_pressure(coolprop_fluid, T) / pressure), low, high,
                        xtol=1e-12, rtol=1e-14))


@lru_cache(maxsize=None)
def _melting_line(coolprop_fluid: str):
    """CoolProp-Zustandsobjekt mit Schmelzdruckkurve (Wasser, CO2, N2, Propan ...) oder None."""
    try:
        state = CP.AbstractState('HEOS', coolprop_fluid)
        return state if state.has_melting_line() else None
    except Exception:
        return None


def _melting_temperature(coolprop_fluid: str, pressure: float) -> Optional[float]:
    """Schmelztemperatur (K) beim Druck p, None ohne Schmelzdruckkurve bzw. außerhalb ihres Bereichs.
    Wasser schmilzt unter Druck tiefer (200 bar: -1.54 °C): dort ist es knapp unter der
    Tripeltemperatur stabil flüssig, nicht unterkühlt."""
    state = _melting_line(coolprop_fluid)
    if state is None:
        return None
    try:
        value = float(state.melting_line(CP.iT, CP.iP, float(pressure)))
    except Exception:
        return None
    return value if np.isfinite(value) and value > 0 else None


def _vapour_density(coolprop_fluid: str, temperature: float, pressure: float) -> float:
    """
    Dichte des Dampfs bei T und p unter der Tripeltemperatur: CoolProp nimmt dort das Eingabepaar
    (T, p) nicht an ("we don't support p below ptriple when T is less than Tmin"), rechnet die
    Gasphase aber mit (T, rho) - die Dichte so, dass der Druck stimmt (Start: ideales Gas, der
    Dampf ist dort nahezu ideal).
    """
    gas_constant = 8.314462618 / float(CP.PropsSI('M', coolprop_fluid))
    density = pressure / (gas_constant * temperature)
    for _ in range(30):
        computed = float(CP.PropsSI('P', 'T', temperature, 'D', density, coolprop_fluid))
        if abs(computed - pressure) <= 1e-13 * pressure:
            break
        density *= pressure / computed
    return density


def _below_triple_error(coolprop_fluid: str, temperature: float, pressure: Optional[float], kind: str,
                        sublimation: Optional[float] = None) -> ValueError:
    """Meldung für einen Zustand unter dem Tripelpunkt, für den es keine Stoffdaten gibt."""
    t_triple, p_triple = _triple_point(coolprop_fluid)
    triple = f"{t_triple - 273.15:.4g} °C" + (f", {p_triple:.4g} Pa" if p_triple is not None else "")
    state = f"T = {temperature - 273.15:.4g} °C" + (f" und p = {pressure:.4g} Pa" if pressure is not None else "")
    verb = "liegen" if pressure is not None else "liegt"
    if kind == 'solid':
        return ValueError(
            f"{coolprop_fluid}: {state} {verb} im Gebiet der festen Phase (unter dem Tripelpunkt {triple}, "
            f"über dem Sublimationsdruck {sublimation:.4g} Pa) - für die feste Phase gibt es keine Stoffdaten")
    if kind == 'unknown':
        return ValueError(
            f"{coolprop_fluid}: {state} {verb} unter dem Tripelpunkt ({triple}) - dort ist der Stoff Dampf "
            f"oder fest; ohne Daten der festen Phase (Sublimationsdruck) rechnet die Stoffdatenbank das nicht")
    return ValueError(
        f"{coolprop_fluid}: {state} {verb} zu weit unter dem Tripelpunkt ({triple}) - dort ist der "
        f"Stoff fest, und die Stoffdaten der unterkühlten Flüssigkeit reichen nicht so weit "
        f"(außerhalb des Gültigkeitsbereichs)")


def _stable_state(coolprop_fluid: str, keys: list, values: list) -> bool:
    """Thermodynamisch stabil (cp > 0, (dp/drho)_T > 0)? Bei Sättigung zählt die Flüssigkeit."""
    state = [keys[0], values[0], keys[1], values[1]]
    if 'Q' in keys:
        state[2 * keys.index('Q') + 1] = 0.0
    try:
        heat_capacity = float(CP.PropsSI('C', *state, coolprop_fluid))
        stiffness = float(CP.PropsSI('d(P)/d(D)|T', *state, coolprop_fluid))
    except Exception:
        return False
    return bool(np.isfinite(heat_capacity) and np.isfinite(stiffness) and heat_capacity > 0 and stiffness > 0)


@lru_cache(maxsize=None)
def _metastable_limit(coolprop_fluid: str) -> float:
    """
    Tiefste Temperatur (K), bis zu der die siedende Flüssigkeit unter dem Tripelpunkt
    thermodynamisch stabil gerechnet wird (Wasser etwa -41 °C) - darunter liefert die
    Zustandsgleichung keine brauchbaren Werte mehr, auch wo ein einzelner Punkt zufällig stabil
    aussieht (Wasser -52 °C, 1 bar: Dichte 3790 kg/m^3). Schrittweise abwärts bis zum ersten
    instabilen Punkt, dann eingegrenzt; nicht tiefer als 60 % der Tripeltemperatur.
    """
    t_triple, _ = _triple_point(coolprop_fluid)
    if t_triple is None:
        return 0.0

    def stable(temperature):
        return _stable_state(coolprop_fluid, ['T', 'Q'], [temperature, 0.0])

    good, bad = t_triple, None
    for step in range(1, 41):
        candidate = t_triple * (1 - 0.01 * step)
        if stable(candidate):
            good = candidate
        else:
            bad = candidate
            break
    if bad is None:
        return good
    for _ in range(20):
        middle = 0.5 * (good + bad)
        if stable(middle):
            good = middle
        else:
            bad = middle
    return good


# Sammelliste für den Hinweis der Oberfläche: Zustände, für die statt des stabilen Zustands (fest)
# die unterkühlte Flüssigkeit geliefert wurde - (Fluid, T, Bezugstemperatur, "dem Tripelpunkt" bzw.
# "der Schmelztemperatur bei diesem Druck")
_supercooled_notes: Optional[list] = None


@contextmanager
def collecting_supercooled():
    """
    with collecting_supercooled() as notes: ... - sammelt während der Auswertung die Zustände, deren
    Wert für die unterkühlte Flüssigkeit gilt (nicht für den stabilen, festen Zustand).
    """
    global _supercooled_notes
    previous, _supercooled_notes = _supercooled_notes, []
    try:
        yield _supercooled_notes
    finally:
        _supercooled_notes = previous


def calculate_property(func_name: str, fluid: str, **kwargs) -> float:
    """
    Berechnet eine thermodynamische Eigenschaft.

    Args:
        func_name: Name der Funktion (enthalpy, entropy, etc.)
        fluid: Name des Fluids
        **kwargs: Zustandsgrößen (T, p, h, s, x, etc.)

    Returns:
        Berechneter Wert in SI-Einheiten
    """
    func_name = func_name.lower()

    # IG:-Stoffe (NASA-Datenbank) - vor CoolProp, ohne Bezugszustand (REFERENCE)
    species = ideal_gas.species_part(fluid)
    if species is not None:
        return ideal_gas.calculate_property(func_name, species, **kwargs)

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
    elif func_name == 'gibbs':
        output_prop = 'G'
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

            # Dampfgehalt: nur Rundungsfehler werden auf 0 ... 1 begrenzt; ein Wert deutlich
            # außerhalb (x = 2) ist ein Eingabefehler bzw. ein ungültiger Iterationspunkt
            if cp_key == 'Q':
                value = float(value)
                if not -1e-6 <= value <= 1 + 1e-6:
                    raise ValueError(f"Dampfgehalt x = {value:.6g} liegt außerhalb von 0 ... 1 "
                                     f"(x gilt nur im Nassdampfgebiet)")
                value = max(0.0, min(1.0, value))

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

    # Bezugszustand (REFERENCE): h, u, s des Blatts -> CoolProp-Standard
    dh, ds = (_reference_offsets.get(canonical_fluid(coolprop_fluid), (0.0, 0.0))
              if _reference_offsets else (0.0, 0.0))
    if dh or ds:
        for cp_key, shift in (('H', dh), ('U', dh), ('S', ds)):
            if cp_key in inputs:
                inputs[cp_key] = float(inputs[cp_key]) - shift

    # Brauchen genau 2 unabhängige Zustandsgrößen
    if len(inputs) != 2:
        raise ValueError(f"Genau 2 Zustandsgrößen erforderlich, {len(inputs)} gegeben")

    # Erstelle CoolProp-Aufruf
    keys = list(inputs.keys())
    values = list(inputs.values())
    vectors = any(np.ndim(value) > 0 for value in values)

    # Zustand unter dem Tripelpunkt (siehe _TRIPLE_TOLERANCE): Dampf und Sättigung über der festen
    # Phase richtig, sonst die unterkühlte Flüssigkeit (mit Hinweis) bzw. eine Meldung
    t_triple, p_triple = _triple_point(coolprop_fluid)
    below = None           # Temperatur des Zustands, wenn sie unter dem Tripelpunkt liegt
    supercooled = False    # der gelieferte Wert gilt für die unterkühlte Flüssigkeit
    direct = None          # Ergebnis ohne Zustandsberechnung (Sublimationslinie)
    reference = ('dem Tripelpunkt', t_triple)     # Bezug für den Hinweis
    if t_triple is not None:
        limit = t_triple * (1 - _TRIPLE_TOLERANCE)
        if 'T' in inputs:
            if float(np.min(inputs['T'])) < limit:
                below = float(np.min(inputs['T']))
            elif set(keys) == {'T', 'P'} and not vectors and p_triple is not None and inputs['P'] > p_triple:
                # Stoffe, deren Schmelztemperatur mit dem Druck steigt (CO2 bei 100 bar: -54.5 °C):
                # unter ihr ist der Stoff fest, auch über der Tripeltemperatur
                melting = _melting_temperature(coolprop_fluid, inputs['P'])
                if melting is not None and inputs['T'] < melting * (1 - _TRIPLE_TOLERANCE):
                    below, reference = float(inputs['T']), ('der Schmelztemperatur bei diesem Druck', melting)
        elif 'Q' in inputs and 'P' in inputs and p_triple is not None and float(np.min(inputs['P'])) < p_triple:
            if vectors:
                raise ValueError("Zustand unter dem Tripelpunkt - punktweise auswerten")
            saturation = _sublimation_temperature(coolprop_fluid, float(inputs['P']))
            if saturation is None:
                try:
                    saturation = float(CP.PropsSI('T', 'P', inputs['P'], 'Q', inputs['Q'], coolprop_fluid))
                except Exception:
                    saturation = 0.0            # nicht berechenbar: weit unter dem Tripelpunkt
            if saturation < limit:
                below = saturation
    if below is not None and vectors:
        raise ValueError("Zustand unter dem Tripelpunkt - punktweise auswerten")
    if below is not None:
        sublimation = _sublimation_pressure(coolprop_fluid, below)
        vapour_at = None                         # Dampfzustand (T, p), gerechnet über (T, rho)
        if 'Q' in inputs and sublimation is not None:
            # Sättigung über der festen Phase: p und T auf der Sublimationslinie
            vapour_fraction = float(inputs['Q'])
            inputs = {'T': below, 'Q': vapour_fraction}
            keys, values = ['T', 'Q'], [below, vapour_fraction]
            if output_prop == 'P':
                direct = sublimation
            elif output_prop == 'T':
                direct = below
            elif vapour_fraction >= 1 - 1e-9:
                vapour_at = (below, sublimation)
            else:
                supercooled = True               # siedende "Flüssigkeit": im Gleichgewicht fest
        elif 'Q' in inputs:
            supercooled = True                   # ohne Daten der festen Phase: verlängerte Siedelinie
        else:
            if set(keys) == {'T', 'P'}:
                state_pressure = float(inputs['P'])
            else:
                try:
                    state_pressure = float(CP.PropsSI('P', keys[0], values[0], keys[1], values[1], coolprop_fluid))
                except Exception:
                    state_pressure = None
            if state_pressure is None:
                supercooled = True
            elif sublimation is not None and state_pressure <= sublimation * (1 + 1e-9):
                if set(keys) == {'T', 'P'}:
                    vapour_at = (below, state_pressure)
                # sonst: Dampf, mit dem gegebenen Eingabepaar gerechnet
            elif p_triple is not None and state_pressure < p_triple:
                raise _below_triple_error(coolprop_fluid, below, state_pressure,
                                          'solid' if sublimation is not None else 'unknown', sublimation)
            else:
                # Flüssigkeitsgebiet: unter der Schmelztemperatur beim Druck des Zustands ist der
                # Stoff im Gleichgewicht fest; darüber (Wasser unter hohem Druck) stabil flüssig
                melting = _melting_temperature(coolprop_fluid, state_pressure)
                supercooled = melting is None or below < melting * (1 - 1e-9)
        if supercooled and below < _metastable_limit(coolprop_fluid):
            raise _below_triple_error(coolprop_fluid, below, inputs.get('P'), 'range')
        if vapour_at is not None:
            try:
                keys, values = ['T', 'D'], [vapour_at[0], _vapour_density(coolprop_fluid, *vapour_at)]
            except Exception:
                raise _below_triple_error(coolprop_fluid, below, vapour_at[1], 'range') from None
            if output_prop == 'P':
                direct = vapour_at[1]

    # cp, cv, Viskosität, Wärmeleitfähigkeit, Prandtl-Zahl, Schallgeschwindigkeit gibt es nur in
    # einer Phase (x = 0 bzw. x = 1 sind die Phasen an der Grenze)
    if output_prop in _SINGLE_PHASE_ONLY and set(keys) != {'T', 'P'}:
        try:
            vapour = inputs['Q'] if 'Q' in inputs else CP.PropsSI(
                'Q', keys[0], values[0], keys[1], values[1], coolprop_fluid)
            inside = (np.asarray(vapour, dtype=float) > 1e-9) & (np.asarray(vapour, dtype=float) < 1 - 1e-9)
        except Exception:
            vapour, inside = None, np.array(False)
        if np.any(inside):
            if vectors:
                raise ValueError("Zustand im Nassdampfgebiet - punktweise auswerten")
            raise ValueError(
                f"{coolprop_fluid}: {func_name} ist im Nassdampfgebiet (x = {float(vapour):.4g}) nicht "
                f"definiert - dort liegen Flüssigkeit und Dampf nebeneinander vor. Wert der siedenden "
                f"Flüssigkeit (x=0) oder des Sattdampfs (x=1) verwenden")

    try:
        result_si = direct if direct is not None else CP.PropsSI(
            output_prop, keys[0], values[0], keys[1], values[1], coolprop_fluid)
    except Exception as e:
        if below is not None or 'below ptriple' in str(e):
            raise _below_triple_error(coolprop_fluid, below if below is not None else float(inputs.get('T', 0.0)),
                                      inputs.get('P'), 'unknown' if 'below ptriple' in str(e) else 'range') from None
        if set(keys) == {'T', 'P'} and 'Saturation pressure' in str(e):
            # T und p auf der Siedelinie bestimmen den Zustand nicht (Nassdampf: x fehlt)
            raise ValueError(f"{coolprop_fluid}: T = {inputs['T'] - 273.15:.4g} °C und p = "
                             f"{inputs['P'] / 1e5:.5g} bar liegen auf der Sättigungslinie - dort legen "
                             f"T und p den Zustand nicht fest. Dampfgehalt angeben (x=1 Sattdampf, "
                             f"x=0 siedende Flüssigkeit) statt T bzw. p") from None
        note = ''
        if dh or ds:
            # Zahlen der CoolProp-Meldung sind intern verschoben (h, u, s im CoolProp-Bezug)
            note = (f" (h, u, s in dieser Meldung im CoolProp-Standardbezug; im Blatt gilt "
                    f"REFERENCE {_reference_names.get(canonical_fluid(coolprop_fluid), '')}: "
                    f"h um {dh / 1e3:+.6g} kJ/kg, s um {ds / 1e3:+.6g} kJ/(kg K) verschoben)")
        raise ValueError(f"CoolProp Fehler für {coolprop_fluid}: {e}{note}")

    if supercooled:
        if not _stable_state(coolprop_fluid, keys, values):
            raise _below_triple_error(coolprop_fluid, below, inputs.get('P'), 'range')
        if _supercooled_notes is not None:
            _supercooled_notes.append((coolprop_fluid, below, reference[1], reference[0]))

    # Dampfgehalt außerhalb des Nassdampfgebiets: CoolProp liefert je nach Eingabepaar -1 (T, p)
    # oder 10000 (rho mit u, h, s) - einheitlich -1 (wie dokumentiert, Hinweis in der Oberfläche)
    if output_prop == 'Q' and not -1e-9 <= result_si <= 1 + 1e-9:
        return -1.0

    # Ausgabe in SI (volume = 1/Dichte)
    if func_name == 'volume':
        # Spez. Volumen = 1 / Dichte
        result = 1.0 / result_si
    elif output_prop in FROM_SI:
        result = FROM_SI[output_prop](result_si)
    else:
        result = result_si
    shift = dh if output_prop in ('H', 'U') else ds if output_prop == 'S' else 0.0
    if output_prop == 'G' and (dh or ds):
        # g = h - T*s: verschiebt sich um dh - T*ds
        T_state = inputs['T'] if 'T' in inputs else CP.PropsSI(
            'T', keys[0], values[0], keys[1], values[1], coolprop_fluid)
        shift = dh - T_state * ds
    if shift:
        shifted = result + shift
        # im Bezugspunkt selbst bleibt sonst ein Rundungsrest (-1.2e-9 J/kg statt 0)
        result = 0.0 if abs(shifted) <= 1e-12 * max(abs(shift), abs(result)) else shifted

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

def gibbs(fluid: str, **kwargs) -> float:
    """Spez. freie Enthalpie g = h - T*s [J/kg]"""
    return calculate_property('gibbs', fluid, **kwargs)

def molarmass(fluid: str, **kwargs) -> float:
    """Molmasse [kg/mol] (Anzeige kg/kmol) - IG:-Stoffe und CoolProp-Fluide"""
    species = ideal_gas.species_part(fluid)
    if species is not None:
        return ideal_gas.calculate_property('molarmass', species, **kwargs)
    if kwargs:
        raise ValueError(f"molarmass({fluid}) hat keine Zustandsgrößen (nur den Stoff)")
    try:
        return CP.PropsSI('M', get_fluid_name(fluid))
    except Exception as e:
        raise ValueError(f"Molmasse von {fluid} nicht verfügbar (CoolProp: {e})") from None


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
    'gibbs': gibbs,
    'molarmass': molarmass,
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
