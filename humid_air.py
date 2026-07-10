"""
Humid Air Module for the HVAC Equation Solver

Uses CoolProp HumidAirProp for psychrometric calculations.

Units (SI base units for internal calculations):
- Temperature T: K
- Pressure p_tot: Pa
- Enthalpy h: J/kg_dry_air
- Humidity ratio w: kg_water/kg_dry_air
- Relative humidity rh: - (0-1)
- Partial pressure p_w: Pa
- Densities rho: kg/m³

Syntax:
    h = HumidAir(h, T=298.15, rh=0.5, p_tot=100000)  # T in K, p in Pa
    w = HumidAir(w, T=303.15, rh=0.6, p_tot=100000)
    T_dp = HumidAir(T_dp, T=298.15, w=0.01, p_tot=100000)
"""

import CoolProp.CoolProp as CP
from typing import Dict, Any, Optional
from scipy.optimize import brentq


# Mapping of output properties
# User-Name -> (CoolProp-Key, conversion function SI->User)
# Da wir intern SI verwenden, ist keine Konvertierung mehr nötig!
OUTPUT_MAP = {
    't': ('T', lambda x: x),                        # K -> K
    'h': ('Hda', lambda x: x),                      # J/kg -> J/kg (bleibt SI)
    'rh': ('R', lambda x: x),                       # dimensionless (0-1)
    'w': ('W', lambda x: x),                        # kg_water/kg_dry_air
    'p_w': ('psi_w', None),                         # Special handling: psi_w * P -> Pa
    'rho_tot': ('Vha', lambda x: 1/x),              # m³/kg -> kg/m³ (humid air density)
    'rho_a': ('Vda', lambda x: 1/x),                # m³/kg -> kg/m³ (dry air density)
    'rho_w': (None, None),                          # Special handling: W / Vda
    't_dp': ('Tdp', lambda x: x),                   # K -> K
    't_wb': ('Twb', lambda x: x),                   # K -> K
}

# Mapping of input parameters
# User-Name -> (CoolProp-Key, conversion function User->SI)
# Da wir intern SI verwenden (Pa, J/kg, K), ist keine Konvertierung mehr nötig!
INPUT_MAP = {
    't': ('T', lambda x: x),                        # K -> K
    'p_tot': ('P', lambda x: x),                    # Pa -> Pa (bereits SI)
    'w': ('W', lambda x: x),                        # kg_water/kg_dry_air
    'rh': ('R', lambda x: x),                       # dimensionless (0-1)
    'rf': ('R', lambda x: x),                       # rF = relative Feuchte (German alias for rh)
    'p_w': ('psi_w', None),                         # Special handling
    'h': ('Hda', lambda x: x),                      # J/kg -> J/kg (bereits SI)
}


def _resolve_dual_humidity(inputs: dict, humidity_keys: set) -> dict:
    """
    Resolves the case when two humidity properties are given.

    CoolProp doesn't support two humidity inputs directly (e.g., w and rh).
    This function finds the temperature T that satisfies both conditions
    and returns a valid input set (T, P, and one humidity property).

    Args:
        inputs: Dict with CoolProp keys and SI values (must contain P and 2 humidity props)
        humidity_keys: Set of the two humidity keys present

    Returns:
        Modified inputs dict with T instead of one humidity property
    """
    P = inputs['P']
    humidity_list = list(humidity_keys)
    key1, key2 = humidity_list[0], humidity_list[1]
    val1, val2 = inputs[key1], inputs[key2]

    # Suchbereiche in K:
    # HAPropsSI ist ca. von 135 K bis 620 K gültig. Primär wird ein
    # konservativer Bereich abgesucht, bei Misserfolg ein erweiterter.
    T_RANGES = [
        (213.15, 473.15),   # -60°C bis 200°C (konservativ)
        (143.15, 613.15),   # Fallback: nahezu voller HAPropsSI-Bereich
    ]

    def make_residual(known_key, known_val, target_key, target_val):
        """Residual: berechne target_key aus (T, P, known_key) und vergleiche."""
        def residual(T_K):
            try:
                computed = CP.HAPropsSI(target_key, 'T', T_K, 'P', P,
                                        known_key, known_val)
                return computed - target_val
            except (ValueError, RuntimeError):
                return float('nan')  # ungültiger Punkt (statt inf)
        return residual

    def find_root(residual, T_min, T_max, step=2.0):
        """
        Tastet [T_min, T_max] in `step`-K-Schritten ab und löst mit brentq
        in einem gültigen Teilintervall mit Vorzeichenwechsel.
        Gibt None zurück, wenn keine Wurzel gefunden wurde.
        """
        T_prev = T_min
        r_prev = residual(T_prev)
        while T_prev < T_max:
            T_cur = min(T_prev + step, T_max)
            r_cur = residual(T_cur)
            # Beide Punkte gültig (nicht NaN)?
            if r_prev == r_prev and r_cur == r_cur:
                if r_prev == 0.0:
                    return T_prev
                if r_cur == 0.0:
                    return T_cur
                if r_prev * r_cur < 0:
                    try:
                        return brentq(residual, T_prev, T_cur, xtol=1e-10)
                    except (ValueError, RuntimeError):
                        pass  # weiter suchen
            T_prev, r_prev = T_cur, r_cur
        return None

    # Beide Richtungen probieren: key2 aus key1 berechnen und umgekehrt.
    # Die bekannte (known) Feuchtegröße wird im Ergebnis behalten.
    attempts = [
        (make_residual(key1, val1, key2, val2), key1, val1),
        (make_residual(key2, val2, key1, val1), key2, val2),
    ]

    for T_min, T_max in T_RANGES:
        for residual, keep_key, keep_val in attempts:
            T_K = find_root(residual, T_min, T_max)
            if T_K is not None:
                return {'T': T_K, 'P': P, keep_key: keep_val}

    raise ValueError(f"Could not find consistent temperature for given humidity properties "
                     f"({key1}={val1}, {key2}={val2})")


def HumidAir(output_prop: str, **kwargs) -> float:
    """
    Calculates properties of humid air.

    All inputs and outputs use SI base units (K, Pa, J/kg).

    Args:
        output_prop: The property to calculate:
            - T: Dry bulb temperature [K]
            - h: Specific enthalpy [J/kg_dry_air]
            - rh: Relative humidity [-] (0-1)
            - w: Humidity ratio [kg_water/kg_dry_air]
            - p_w: Partial pressure of water vapor [Pa]
            - rho_tot: Density of humid air [kg/m³]
            - rho_a: Density of dry air [kg/m³]
            - rho_w: Density of water vapor [kg/m³]
            - T_dp: Dew point temperature [K]
            - T_wb: Wet bulb temperature [K]

        **kwargs: State properties to define the state (exactly 3 required):
            - T: Temperature [K]
            - p_tot: Total pressure [Pa]
            - w: Humidity ratio [kg_water/kg_dry_air]
            - rh: Relative humidity [-]
            - p_w: Partial pressure of water vapor [Pa]
            - h: Enthalpy [J/kg_dry_air]

    Returns:
        Calculated value in SI units

    Examples:
        h = HumidAir('h', T=298.15, rh=0.5, p_tot=100000)
        w = HumidAir('w', T=303.15, rh=0.6, p_tot=100000)
        T = HumidAir('T', h=50000, rh=0.5, p_tot=100000)
        T_dp = HumidAir('T_dp', T=298.15, w=0.01, p_tot=100000)
    """
    # Normalize output property (lowercase)
    output_key = output_prop.lower()

    if output_key not in OUTPUT_MAP:
        valid_outputs = ', '.join(OUTPUT_MAP.keys())
        raise ValueError(f"Unknown property '{output_prop}'. Valid values: {valid_outputs}")

    # Collect and convert input parameters
    inputs = {}
    p_tot_pa = None  # Store total pressure for p_w calculations

    for key, value in kwargs.items():
        key_lower = key.lower()

        if key_lower not in INPUT_MAP:
            valid_inputs = ', '.join(INPUT_MAP.keys())
            raise ValueError(f"Unknown parameter '{key}'. Valid parameters: {valid_inputs}")

        cp_key, converter = INPUT_MAP[key_lower]

        # Special case: p_w as input (partial pressure -> water mole fraction)
        if key_lower == 'p_w':
            # p_w will be converted later when p_tot is known
            inputs['_p_w_input'] = value  # Store temporarily
        else:
            if converter:
                inputs[cp_key] = converter(value)
            else:
                inputs[cp_key] = value

            # Store total pressure
            if key_lower == 'p_tot':
                p_tot_pa = inputs[cp_key]

    # Convert p_w to psi_w if p_w was given as input
    if '_p_w_input' in inputs:
        p_w_pa = inputs.pop('_p_w_input')  # Already in Pa (SI)
        if p_tot_pa is None:
            raise ValueError("When using p_w as input, p_tot must also be specified")
        # psi_w = p_w / p_tot
        inputs['psi_w'] = p_w_pa / p_tot_pa

    # Check that exactly 3 independent parameters are given
    # (CoolProp HumidAirProp requires 3 inputs: typically T, P, and one humidity property)
    if len(inputs) != 3:
        raise ValueError(f"Exactly 3 state properties required, {len(inputs)} given. "
                        f"Typically: T, p_tot and one humidity property (rh, w, p_w or h)")

    # Check for dual humidity inputs (CoolProp doesn't support these directly)
    # Humidity properties: W (w), R (rh), psi_w (p_w)
    humidity_keys = {'W', 'R', 'psi_w'}
    input_humidity = set(inputs.keys()) & humidity_keys

    if len(input_humidity) == 2 and 'P' in inputs:
        # Two humidity properties given - need to find T iteratively
        inputs = _resolve_dual_humidity(inputs, input_humidity)

    # Output p_w requires the total pressure as input (p_w = psi_w * p_tot)
    if output_key == 'p_w' and p_tot_pa is None:
        raise ValueError("Output 'p_w' benötigt p_tot als Input-Parameter "
                         "(p_w = psi_w * p_tot)")

    # Create CoolProp call
    keys = list(inputs.keys())
    values = list(inputs.values())

    try:
        # Special case: rho_w (water vapor density)
        if output_key == 'rho_w':
            # rho_w = W / Vda = w * rho_a
            W = CP.HAPropsSI('W', keys[0], values[0], keys[1], values[1], keys[2], values[2])
            Vda = CP.HAPropsSI('Vda', keys[0], values[0], keys[1], values[1], keys[2], values[2])
            return W / Vda

        # Special case: p_w (partial pressure of water vapor)
        # p_tot_pa is guaranteed to be set here (checked above)
        if output_key == 'p_w':
            psi_w = CP.HAPropsSI('psi_w', keys[0], values[0], keys[1], values[1], keys[2], values[2])
            # p_w = psi_w * p_tot (already in Pa, SI)
            return psi_w * p_tot_pa

        # Normal case
        cp_output_key, converter = OUTPUT_MAP[output_key]
        result_si = CP.HAPropsSI(cp_output_key, keys[0], values[0], keys[1], values[1], keys[2], values[2])

        if converter:
            return converter(result_si)
        else:
            return result_si

    except Exception as e:
        raise ValueError(f"CoolProp HumidAirProp Error: {e}")


# Wrapper function for case-insensitivity
def humidair(output_prop: str, **kwargs) -> float:
    """Alias for HumidAir (lowercase)."""
    return HumidAir(output_prop, **kwargs)


# Dictionary of all humid air functions for the solver
HUMID_AIR_FUNCTIONS = {
    'HumidAir': HumidAir,
    'humidair': humidair,
}


if __name__ == "__main__":
    # Tests (alle Eingaben in SI-Basiseinheiten: T in K, p in Pa, h in J/kg)
    print("=== Humid Air Module Tests (SI units) ===\n")

    # Test 1: Enthalpy at given temperature, relative humidity and pressure
    print("Test 1: Enthalpy at T=298.15 K (25°C), rh=0.5, p_tot=100000 Pa")
    h = HumidAir('h', T=298.15, rh=0.5, p_tot=100000)
    print(f"  h = {h:.1f} J/kg_dry_air ({h/1000:.2f} kJ/kg)")
    print()

    # Test 2: Humidity ratio
    print("Test 2: Humidity ratio at T=303.15 K (30°C), rh=0.6, p_tot=100000 Pa")
    w = HumidAir('w', T=303.15, rh=0.6, p_tot=100000)
    print(f"  w = {w:.5f} kg_water/kg_dry_air")
    print()

    # Test 3: Dew point temperature
    print("Test 3: Dew point at T=298.15 K (25°C), w=0.01, p_tot=100000 Pa")
    T_dp = HumidAir('T_dp', T=298.15, w=0.01, p_tot=100000)
    print(f"  T_dp = {T_dp:.2f} K ({T_dp - 273.15:.2f} °C)")
    print()

    # Test 4: Wet bulb temperature
    print("Test 4: Wet bulb temperature at T=303.15 K (30°C), rh=0.5, p_tot=100000 Pa")
    T_wb = HumidAir('T_wb', T=303.15, rh=0.5, p_tot=100000)
    print(f"  T_wb = {T_wb:.2f} K ({T_wb - 273.15:.2f} °C)")
    print()

    # Test 5: Densities
    print("Test 5: Densities at T=298.15 K (25°C), rh=0.5, p_tot=100000 Pa")
    rho_tot = HumidAir('rho_tot', T=298.15, rh=0.5, p_tot=100000)
    rho_a = HumidAir('rho_a', T=298.15, rh=0.5, p_tot=100000)
    rho_w = HumidAir('rho_w', T=298.15, rh=0.5, p_tot=100000)
    print(f"  rho_tot = {rho_tot:.4f} kg/m³ (humid air)")
    print(f"  rho_a = {rho_a:.4f} kg/m³ (dry air)")
    print(f"  rho_w = {rho_w:.6f} kg/m³ (water vapor)")
    print()

    # Test 6: Partial pressure
    print("Test 6: Partial pressure at T=298.15 K (25°C), rh=0.5, p_tot=100000 Pa")
    p_w = HumidAir('p_w', T=298.15, rh=0.5, p_tot=100000)
    print(f"  p_w = {p_w:.1f} Pa ({p_w/1e5:.5f} bar)")
    print()

    # Test 7: Relative humidity from humidity ratio
    print("Test 7: Relative humidity at T=298.15 K (25°C), w=0.01, p_tot=100000 Pa")
    rh = HumidAir('rh', T=298.15, w=0.01, p_tot=100000)
    print(f"  rh = {rh:.3f}")
    print()

    # Test 8: With p_w as input
    print("Test 8: Humidity ratio at T=298.15 K (25°C), p_w=1000 Pa, p_tot=100000 Pa")
    w = HumidAir('w', T=298.15, p_w=1000, p_tot=100000)
    print(f"  w = {w:.5f} kg_water/kg_dry_air")
    print()

    # Test 9: Different parameter combinations
    print("Test 9: Different parameter combinations")
    h1 = HumidAir('h', w=0.012, T=305.15, p_tot=100000)
    h2 = HumidAir('h', T=305.15, p_tot=100000, w=0.012)
    h3 = HumidAir('h', p_tot=100000, rh=0.5, T=305.15)
    print(f"  HumidAir('h', w=0.012, T=305.15, p_tot=100000) = {h1:.1f} J/kg")
    print(f"  HumidAir('h', T=305.15, p_tot=100000, w=0.012) = {h2:.1f} J/kg")
    print(f"  HumidAir('h', p_tot=100000, rh=0.5, T=305.15) = {h3:.1f} J/kg")
    print()

    # Test 10: Dual humidity inputs (T is solved iteratively)
    print("Test 10: Dual humidity inputs")
    T_a = HumidAir('T', rh=0.5, w=0.01, p_tot=100000)
    print(f"  T(rh=0.5, w=0.01, p_tot=1 bar) = {T_a:.2f} K ({T_a - 273.15:.2f} °C)")
    T_b = HumidAir('T', rh=0.3, w=0.15, p_tot=101325)
    print(f"  T(rh=0.3, w=0.15, p_tot=1 atm) = {T_b:.2f} K ({T_b - 273.15:.2f} °C)")
