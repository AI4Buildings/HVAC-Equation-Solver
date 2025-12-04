"""
Humid Air Module for the HVAC Equation Solver

Uses CoolProp HumidAirProp for psychrometric calculations.

Units:
- Temperature T: °C
- Pressure p_tot: bar
- Enthalpy h: kJ/kg_dry_air
- Humidity ratio w: kg_water/kg_dry_air
- Relative humidity rh: - (0-1)
- Partial pressure p_w: bar
- Densities rho: kg/m³

Syntax:
    h = HumidAir(h, T=25, rh=0.5, p_tot=1)
    w = HumidAir(w, T=30, rh=0.6, p_tot=1)
    T_dp = HumidAir(T_dp, T=25, w=0.01, p_tot=1)
"""

import CoolProp.CoolProp as CP
from typing import Dict, Any, Optional


# Mapping of output properties
# User-Name -> (CoolProp-Key, conversion function SI->User)
OUTPUT_MAP = {
    't': ('T', lambda x: x - 273.15),               # K -> °C (dry bulb temperature)
    'h': ('Hda', lambda x: x / 1000),               # J/kg -> kJ/kg
    'rh': ('R', lambda x: x),                       # dimensionless (0-1)
    'w': ('W', lambda x: x),                        # kg_water/kg_dry_air
    'p_w': ('psi_w', None),                         # Special handling: psi_w * P -> bar
    'rho_tot': ('Vha', lambda x: 1/x),              # m³/kg -> kg/m³ (humid air density)
    'rho_a': ('Vda', lambda x: 1/x),                # m³/kg -> kg/m³ (dry air density)
    'rho_w': (None, None),                          # Special handling: W / Vda
    't_dp': ('Tdp', lambda x: x - 273.15),          # K -> °C (dew point)
    't_wb': ('Twb', lambda x: x - 273.15),          # K -> °C (wet bulb)
}

# Mapping of input parameters
# User-Name -> (CoolProp-Key, conversion function User->SI)
INPUT_MAP = {
    't': ('T', lambda x: x + 273.15),               # °C -> K
    'p_tot': ('P', lambda x: x * 1e5),              # bar -> Pa
    'w': ('W', lambda x: x),                        # kg_water/kg_dry_air
    'rh': ('R', lambda x: x),                       # dimensionless (0-1)
    'p_w': ('psi_w', None),                         # Special handling
    'h': ('Hda', lambda x: x * 1000),               # kJ/kg -> J/kg
}


def HumidAir(output_prop: str, **kwargs) -> float:
    """
    Calculates properties of humid air.

    Args:
        output_prop: The property to calculate:
            - T: Dry bulb temperature [°C]
            - h: Specific enthalpy [kJ/kg_dry_air]
            - rh: Relative humidity [-] (0-1)
            - w: Humidity ratio [kg_water/kg_dry_air]
            - p_w: Partial pressure of water vapor [bar]
            - rho_tot: Density of humid air [kg/m³]
            - rho_a: Density of dry air [kg/m³]
            - rho_w: Density of water vapor [kg/m³]
            - T_dp: Dew point temperature [°C]
            - T_wb: Wet bulb temperature [°C]

        **kwargs: State properties to define the state (exactly 3 required):
            - T: Temperature [°C]
            - p_tot: Total pressure [bar]
            - w: Humidity ratio [kg_water/kg_dry_air]
            - rh: Relative humidity [-]
            - p_w: Partial pressure of water vapor [bar]
            - h: Enthalpy [kJ/kg_dry_air]

    Returns:
        Calculated value in the specified units

    Examples:
        h = HumidAir('h', T=25, rh=0.5, p_tot=1)
        w = HumidAir('w', T=30, rh=0.6, p_tot=1)
        T = HumidAir('T', h=50, rh=0.5, p_tot=1)
        T_dp = HumidAir('T_dp', T=25, w=0.01, p_tot=1)
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
        p_w_pa = inputs.pop('_p_w_input') * 1e5  # bar -> Pa
        if p_tot_pa is None:
            raise ValueError("When using p_w as input, p_tot must also be specified")
        # psi_w = p_w / p_tot
        inputs['psi_w'] = p_w_pa / p_tot_pa

    # Check that exactly 3 independent parameters are given
    # (CoolProp HumidAirProp requires 3 inputs: typically T, P, and one humidity property)
    if len(inputs) != 3:
        raise ValueError(f"Exactly 3 state properties required, {len(inputs)} given. "
                        f"Typically: T, p_tot and one humidity property (rh, w, p_w or h)")

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
        if output_key == 'p_w':
            psi_w = CP.HAPropsSI('psi_w', keys[0], values[0], keys[1], values[1], keys[2], values[2])
            # Get total pressure
            if p_tot_pa is None:
                p_tot_pa = CP.HAPropsSI('P', keys[0], values[0], keys[1], values[1], keys[2], values[2])
            # p_w = psi_w * p_tot (convert to bar)
            return (psi_w * p_tot_pa) / 1e5

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
    print("=== Humid Air Module Tests ===\n")

    # Test 1: Enthalpy at given temperature, relative humidity and pressure
    print("Test 1: Enthalpy at T=25°C, rh=0.5, p_tot=1 bar")
    h = HumidAir('h', T=25, rh=0.5, p_tot=1)
    print(f"  h = {h:.2f} kJ/kg_dry_air")
    print()

    # Test 2: Humidity ratio
    print("Test 2: Humidity ratio at T=30°C, rh=0.6, p_tot=1 bar")
    w = HumidAir('w', T=30, rh=0.6, p_tot=1)
    print(f"  w = {w:.5f} kg_water/kg_dry_air")
    print()

    # Test 3: Dew point temperature
    print("Test 3: Dew point at T=25°C, w=0.01, p_tot=1 bar")
    T_dp = HumidAir('T_dp', T=25, w=0.01, p_tot=1)
    print(f"  T_dp = {T_dp:.2f} °C")
    print()

    # Test 4: Wet bulb temperature
    print("Test 4: Wet bulb temperature at T=30°C, rh=0.5, p_tot=1 bar")
    T_wb = HumidAir('T_wb', T=30, rh=0.5, p_tot=1)
    print(f"  T_wb = {T_wb:.2f} °C")
    print()

    # Test 5: Densities
    print("Test 5: Densities at T=25°C, rh=0.5, p_tot=1 bar")
    rho_tot = HumidAir('rho_tot', T=25, rh=0.5, p_tot=1)
    rho_a = HumidAir('rho_a', T=25, rh=0.5, p_tot=1)
    rho_w = HumidAir('rho_w', T=25, rh=0.5, p_tot=1)
    print(f"  rho_tot = {rho_tot:.4f} kg/m³ (humid air)")
    print(f"  rho_a = {rho_a:.4f} kg/m³ (dry air)")
    print(f"  rho_w = {rho_w:.6f} kg/m³ (water vapor)")
    print()

    # Test 6: Partial pressure
    print("Test 6: Partial pressure at T=25°C, rh=0.5, p_tot=1 bar")
    p_w = HumidAir('p_w', T=25, rh=0.5, p_tot=1)
    print(f"  p_w = {p_w:.5f} bar")
    print()

    # Test 7: Relative humidity from humidity ratio
    print("Test 7: Relative humidity at T=25°C, w=0.01, p_tot=1 bar")
    rh = HumidAir('rh', T=25, w=0.01, p_tot=1)
    print(f"  rh = {rh:.3f}")
    print()

    # Test 8: With p_w as input
    print("Test 8: Humidity ratio at T=25°C, p_w=0.01 bar, p_tot=1 bar")
    w = HumidAir('w', T=25, p_w=0.01, p_tot=1)
    print(f"  w = {w:.5f} kg_water/kg_dry_air")
    print()

    # Test 9: Different parameter combinations
    print("Test 9: Different parameter combinations")
    h1 = HumidAir('h', w=0.012, T=32, p_tot=1)
    h2 = HumidAir('h', T=32, p_tot=1, w=0.012)
    h3 = HumidAir('h', p_tot=1, rh=0.5, T=32)
    print(f"  HumidAir('h', w=0.012, T=32, p_tot=1) = {h1:.2f} kJ/kg")
    print(f"  HumidAir('h', T=32, p_tot=1, w=0.012) = {h2:.2f} kJ/kg")
    print(f"  HumidAir('h', p_tot=1, rh=0.5, T=32) = {h3:.2f} kJ/kg")
