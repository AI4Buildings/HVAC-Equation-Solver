"""
Ideale Gase und kondensierte Stoffe aus der NASA-Datenbank thermo.inp

Stoffwerte mit dem Präfix IG: (auch IG::, Groß-/Kleinschreibung egal) in den
Stoffwertfunktionen; ohne Präfix rechnet weiter CoolProp:
    h = enthalpy(IG:N2, T=20 °C)               J/kg, mit Bildungsenthalpie
    s = entropy(IG:N2, T=20 °C, p=10 bar)      J/(kg K), absolute Entropie
    g = gibbs(IG:H2O(L), T=25 °C, p=1 bar)     J/kg
    M = molarmass(IG:CH4)                      kg/mol (Anzeige kg/kmol)

Daten: data/nasa/thermo.inp aus NASA CEA (https://github.com/nasa/cea, Commit 2f79a64,
unverändert), Apache License 2.0 - siehe data/nasa/LICENSE.txt und NOTICE.txt.
NASA-9-Polynome nach McBride, Zehe, Gordon: NASA Glenn Coefficients for Calculating
Thermodynamic Properties of Individual Species, NASA TP-2002-211556 (2002).
Dieses Modul (Leser und Formeln) steht wie das übrige Projekt unter MIT.

Einheiten massebezogen in SI wie bei CoolProp: h, u, g in J/kg, s, cp, cv in J/(kg K),
v in m³/kg, rho in kg/m³, Molmasse in kg/mol. h enthält die Bildungsenthalpie
(h(298.15 K) = Δh_f, Elemente im Referenzzustand 0), s ist die absolute Entropie
(dritter Hauptsatz) - Reaktions- und Verbrennungsbilanzen brauchen keinen eigenen
Bezugspunkt. Gase sind ideal: h, u, cp, cv hängen nur von T ab, s = s°(T) - R·ln(p/p°)
mit p° = 1 bar. Kondensierte Stoffe: h, s, g, cp ohne Druckeinfluss, keine Dichte.

Alle Funktionen rechnen elementweise mit numpy-Arrays (Parameterstudien).
"""

import difflib
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import brentq

# Allgemeine Gaskonstante (CODATA 2018, exakt) und Standarddruck der NASA-Daten
R_UNIVERSAL = 8.314462618       # J/(mol K)
P_STANDARD = 1e5                # Pa

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'nasa', 'thermo.inp')
DATA_SOURCE = "NASA CEA thermo.inp"

_PREFIX = re.compile(r'^\s*IG::?\s*', re.IGNORECASE)
_EXPONENTS = (-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0)

# Eingaben (Schlüsselwort -> Zustandsgröße) wie bei CoolProp; x gibt es nicht
INPUT_MAP = {'t': 'T', 'p': 'P', 'h': 'H', 'u': 'U', 's': 'S', 'v': 'V', 'rho': 'D', 'd': 'D'}
_INPUT_NAMES = {'T': 'T', 'P': 'p', 'H': 'h', 'U': 'u', 'S': 's', 'V': 'v', 'D': 'rho'}

# Funktionen: Ergebnis nur von T abhängig bzw. auch vom Druck (Gase)
_TEMPERATURE_ONLY = ('enthalpy', 'intenergy', 'cp', 'cv', 'soundspeed', 'temperature')
_PRESSURE_DEPENDENT = ('entropy', 'gibbs', 'volume', 'density', 'pressure')
GAS_FUNCTIONS = _TEMPERATURE_ONLY + _PRESSURE_DEPENDENT + ('molarmass',)
CONDENSED_FUNCTIONS = ('enthalpy', 'entropy', 'gibbs', 'cp', 'temperature', 'molarmass')


@dataclass(frozen=True)
class Species:
    """Eine Spezies aus thermo.inp (Molmasse in kg/mol, Bildungsenthalpie in J/mol)."""
    name: str
    condensed: bool
    molar_mass: float
    formation_enthalpy: float
    t_min: np.ndarray            # Temperaturbereiche [K], aufsteigend
    t_max: np.ndarray
    coefficients: np.ndarray     # je Bereich a1 ... a7, b1, b2
    elements: Tuple[Tuple[str, float], ...] = ()   # Zusammensetzung, z.B. (('C', 1.0), ('H', 4.0))

    @property
    def gas_constant(self) -> float:
        """Spezielle Gaskonstante R/M [J/(kg K)]."""
        return R_UNIVERSAL / self.molar_mass

    @property
    def label(self) -> str:
        return f"IG:{self.name}"

    def range_text(self) -> str:
        bounds = [f"{lo:g} ... {hi:g} K" for lo, hi in _merged_ranges(self.t_min, self.t_max)]
        return ", ".join(bounds)


def _merged_ranges(t_min, t_max):
    """Zusammenhängende Temperaturbereiche [(von, bis)]."""
    merged = []
    for lo, hi in zip(t_min, t_max):
        if merged and abs(lo - merged[-1][1]) <= 1e-9 * hi:
            merged[-1][1] = hi
        else:
            merged.append([lo, hi])
    return merged


def _number(text: str) -> float:
    return float(text.replace('D', 'E').replace('d', 'e'))


def parse_thermo_inp(text: str) -> Dict[str, Species]:
    """
    Liest thermo.inp (Format NASA TP-2002-211556, Anhang A; feste Spalten).

    Nach der Zeile 'thermo' folgt eine Zeile mit den Standard-Temperaturbereichen, dann
    die Spezies bis 'END PRODUCTS' und die Reaktanten bis 'END REACTANTS'. Je Spezies:
    Zeile 1 Name (erstes Token); Zeile 2 Anzahl der T-Bereiche (Spalten 1-2), Zusammensetzung
    (Spalten 11-50: 5 x (Elementsymbol 2 Zeichen, Anzahl 6 Zeichen)), Phase
    (51-52, 0 = Gas), Molmasse in g/mol (53-65), Bildungsenthalpie in J/mol (66-80);
    je T-Bereich drei Zeilen: Tmin, Tmax (1-11, 12-22) und Exponenten, a1 ... a5 (5 x 16
    Zeichen), a6, a7 (2 x 16) sowie b1 (49-64) und b2 (65-80). Spezies ohne T-Bereich
    (nur eine Bildungsenthalpie, z.B. JP-4) und leere Bereiche (Tmax <= Tmin) werden
    übersprungen; mehrere Einträge desselben Namens (Umwandlung ohne Namenswechsel,
    z.B. Fe(a) unter/über dem Curie-Punkt) ergeben eine Spezies mit allen Bereichen.
    """
    lines = text.splitlines()
    try:
        index = next(i for i, line in enumerate(lines) if line.strip().lower() == 'thermo') + 2
    except StopIteration:
        raise ValueError("thermo.inp: Zeile 'thermo' fehlt") from None
    raw: Dict[str, list] = {}
    while index < len(lines):
        line = lines[index]
        tag = line.strip().upper()
        if tag.startswith('END REACTANTS'):
            break
        if not tag or tag.startswith('END PRODUCTS') or line.startswith('!'):
            index += 1
            continue
        name = line.split()[0]
        head = lines[index + 1]
        n_ranges = int(head[0:2])
        condensed = int(head[50:52]) != 0
        molar_mass = _number(head[52:65]) / 1000.0
        formation = _number(head[65:80])
        elements = []
        for k in range(5):
            chunk = head[10 + 8 * k:18 + 8 * k]
            symbol, count = chunk[:2].strip().upper(), chunk[2:8].strip()
            if symbol and count and _number(count) != 0:
                elements.append((symbol, _number(count)))
        if n_ranges == 0:
            index += 3
            continue
        ranges = []
        for k in range(n_ranges):
            t_line, c_1, c_2 = lines[index + 2 + 3 * k:index + 5 + 3 * k]
            exponents = tuple(float(e) for e in t_line[23:63].split()[:7])
            if exponents != _EXPONENTS:
                raise ValueError(f"thermo.inp, {name}: unerwartete Exponenten {exponents}")
            t_lo, t_hi = _number(t_line[0:11]), _number(t_line[11:22])
            coefficients = ([_number(c_1[16 * j:16 * j + 16]) for j in range(5)]
                            + [_number(c_2[0:16]), _number(c_2[16:32]),
                               _number(c_2[48:64]), _number(c_2[64:80])])
            if t_hi > t_lo:
                ranges.append((t_lo, t_hi, coefficients))
        index += 2 + 3 * n_ranges
        entry = raw.setdefault(name, [condensed, molar_mass, formation, [], tuple(elements)])
        if entry[0] == condensed:
            entry[3].extend(ranges)
    species = {}
    for name, (condensed, molar_mass, formation, ranges, elements) in raw.items():
        if not ranges:
            continue
        ranges.sort(key=lambda r: r[0])
        species[name] = Species(
            name, condensed, molar_mass, formation,
            np.array([r[0] for r in ranges]), np.array([r[1] for r in ranges]),
            np.array([r[2] for r in ranges]), elements)
    return species


@lru_cache(maxsize=None)
def database() -> Dict[str, Species]:
    """Alle Spezies aus data/nasa/thermo.inp (beim ersten Aufruf eingelesen)."""
    try:
        with open(DATA_FILE, encoding='ascii') as handle:
            return parse_thermo_inp(handle.read())
    except OSError as exc:
        raise ValueError(f"NASA-Datenbank nicht gefunden ({DATA_FILE}): {exc}") from None


def species_names() -> List[str]:
    """Namen aller Spezies in der Reihenfolge der Datei."""
    return list(database())


def species_part(fluid) -> Optional[str]:
    """'IG:N2' -> 'N2' (Präfix IG: oder IG::, Groß-/Kleinschreibung egal), sonst None."""
    if not isinstance(fluid, str):
        return None
    match = _PREFIX.match(fluid)
    return fluid[match.end():].strip() if match else None


def is_ideal_gas(fluid) -> bool:
    return species_part(fluid) is not None


def _candidates(name: str) -> List[str]:
    """Spezies, die ohne Groß-/Kleinschreibung zum Namen passen."""
    lower = name.lower()
    return [n for n in database() if n.lower() == lower]


def find_species(name: str) -> Species:
    """
    Spezies zum Namen (ohne Präfix): zuerst exakt, sonst ohne Groß-/Kleinschreibung,
    wenn eindeutig. Mehrdeutig (CO/Co, CS/Cs, SN/Sn) oder unbekannt -> ValueError mit
    Vorschlägen.
    """
    data = database()
    if name in data:
        return data[name]
    candidates = _candidates(name)
    if len(candidates) == 1:
        return data[candidates[0]]
    if candidates:
        raise ValueError(f"IG:{name} ist nicht eindeutig: " + ", ".join(f"IG:{c}" for c in candidates)
                         + " - Groß-/Kleinschreibung wie in der Datenbank angeben")
    raise ValueError(f"Unbekannter Stoff IG:{name} in {DATA_SOURCE}" + _suggestions(name)
                     + " (Help > Fluid List)")


def _suggestions(name: str) -> str:
    lower = name.lower()
    names = list(database())
    by_lower = {}
    for n in names:
        by_lower.setdefault(n.lower(), n)
    found = [n for n in names if lower and lower in n.lower()][:6]
    for close in difflib.get_close_matches(lower, list(by_lower), n=6, cutoff=0.6):
        if by_lower[close] not in found:
            found.append(by_lower[close])
    return (" - ähnlich: " + ", ".join(f"IG:{n}" for n in found[:8])) if found else ""


def species_argument_count(args: Sequence[str]) -> int:
    """
    Wie viele der (an Kommas getrennten) Aufrufargumente gehören zum Stoffnamen?
    Namen mit Komma (IG:C8H18,isooctane): die längste Folge von Argumenten ohne '=',
    die zusammen eine Spezies ergibt; sonst 1.
    """
    if not args or species_part(args[0]) is None:
        return 1
    for count in range(len(args), 1, -1):
        if any('=' in a for a in args[1:count]):
            continue
        name = species_part(','.join(a.strip() for a in args[:count])) or ''
        if name in database() or len(_candidates(name)) == 1:
            return count
    return 1


# ---------------------------------------------------------------------------
# Polynome (molar, dimensionslos)
# ---------------------------------------------------------------------------

def _coefficients(sp: Species, T: np.ndarray) -> np.ndarray:
    """Koeffizienten des passenden Bereichs je Temperatur; außerhalb -> ValueError."""
    tolerance = 1e-9 * sp.t_max[-1]
    index = np.searchsorted(sp.t_max, T - tolerance, side='left')
    inside = index < len(sp.t_max)
    index = np.minimum(index, len(sp.t_max) - 1)
    inside &= T >= sp.t_min[index] - tolerance
    if not np.all(inside):
        bad = float(np.asarray(T)[~inside].flat[0]) if np.ndim(T) else float(T)
        raise ValueError(f"{sp.label}: T = {bad:.6g} K ({bad - 273.15:.6g} °C) liegt außerhalb "
                         f"des Datenbereichs {sp.range_text()} ({DATA_SOURCE})")
    return sp.coefficients[index]


def _cp_R(c, T):
    a = [c[..., k] for k in range(7)]
    return a[0] / T**2 + a[1] / T + a[2] + a[3] * T + a[4] * T**2 + a[5] * T**3 + a[6] * T**4


def _h_RT(c, T):
    a = [c[..., k] for k in range(9)]
    return (-a[0] / T**2 + a[1] * np.log(T) / T + a[2] + a[3] * T / 2 + a[4] * T**2 / 3
            + a[5] * T**3 / 4 + a[6] * T**4 / 5 + a[7] / T)


def _s_R(c, T):
    a = [c[..., k] for k in range(9)]
    return (-a[0] / (2 * T**2) - a[1] / T + a[2] * np.log(T) + a[3] * T + a[4] * T**2 / 2
            + a[5] * T**3 / 3 + a[6] * T**4 / 4 + a[8])


def _temperature_array(sp: Species, T) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    if np.iscomplexobj(T) or not np.all(np.isfinite(T)):
        raise ValueError(f"{sp.label}: ungültige Temperatur T = {T}")
    return T


def molar_cp(sp: Species, T):
    """cp in J/(mol K)."""
    T = _temperature_array(sp, T)
    return R_UNIVERSAL * _cp_R(_coefficients(sp, T), T)


def molar_enthalpy(sp: Species, T):
    """h in J/mol (mit Bildungsenthalpie)."""
    T = _temperature_array(sp, T)
    return R_UNIVERSAL * T * _h_RT(_coefficients(sp, T), T)


def molar_entropy(sp: Species, T):
    """Standardentropie s°(T) in J/(mol K) bei p° = 1 bar."""
    T = _temperature_array(sp, T)
    return R_UNIVERSAL * _s_R(_coefficients(sp, T), T)


def standard_gibbs_RT(sp: Species, T: float) -> float:
    """g°(T)/(R*T) (molar, Standarddruck p° = 1 bar, mit Bildungsenthalpie) - dimensionslos."""
    T = _temperature_array(sp, T)
    c = _coefficients(sp, T)
    return float(_h_RT(c, T) - _s_R(c, T))


# ---------------------------------------------------------------------------
# Prüfung eines Aufrufs (beim Einlesen und beim Rechnen dieselbe Regel)
# ---------------------------------------------------------------------------

def call_error(function: str, sp: Species, keys: Sequence[str]) -> Optional[str]:
    """
    Fehlermeldung für function(IG:..., keys) oder None. Gase: der Zustand folgt aus
    T (bzw. h oder u, die nur von T abhängen) und - für entropy, gibbs, volume,
    density, pressure - dem Druck p (bzw. s oder v/rho zusammen mit T). Kondensierte
    Stoffe: h, s, g, cp aus T (bzw. h oder s), ohne Druckeinfluss.
    """
    function = function.lower()
    call = f"{function}({sp.label})"
    available = CONDENSED_FUNCTIONS if sp.condensed else GAS_FUNCTIONS
    if function not in available:
        if sp.condensed:
            return (f"{call}: für kondensierte Stoffe nicht verfügbar - {DATA_SOURCE} enthält "
                    f"keine Dichte (Zustandsgleichung) fester und flüssiger Stoffe; verfügbar: "
                    f"{', '.join(CONDENSED_FUNCTIONS)}. Dichte z.B. aus CoolProp (ohne Präfix IG:)")
        return (f"{call}: für IG:-Stoffe nicht verfügbar ({DATA_SOURCE} enthält nur kalorische "
                f"Daten); verfügbar: {', '.join(GAS_FUNCTIONS)}")
    if function == 'molarmass':
        return f"{call} hat keine Zustandsgrößen (nur den Stoff)" if keys else None
    unknown = [k for k in keys if k is None or k.lower() not in INPUT_MAP]
    if unknown:
        hint = " (Dampfgehalt gibt es bei IG:-Stoffen nicht)" if str(unknown[0]).lower() == 'x' else ""
        return f"{call}: unbekannter Parameter '{unknown[0]}'{hint} (gültig: T, p, h, u, s, v, rho)"
    given = [INPUT_MAP[k.lower()] for k in keys]
    if len(set(given)) != len(given):
        return f"{call}: Zustandsgröße doppelt angegeben"
    if not given:
        return f"{call}: Zustand angeben, z.B. T=..."
    if len(given) > 2:
        return f"{call}: höchstens 2 Zustandsgrößen (z.B. T=..., p=...), gegeben {len(given)}"
    names = [_INPUT_NAMES[g] for g in given]

    if sp.condensed:
        if any(g in ('U', 'V', 'D') for g in given):
            bad = names[[g in ('U', 'V', 'D') for g in given].index(True)]
            return (f"{call}: {bad} ist für kondensierte Stoffe nicht verfügbar ({DATA_SOURCE} "
                    f"enthält keine Dichte); Zustand über T, h oder s")
        by_temperature = [n for g, n in zip(given, names) if g in ('T', 'H', 'S')]
        if len(by_temperature) > 1:
            return (f"{call}: {' und '.join(by_temperature)} hängen bei kondensierten Stoffen beide "
                    f"nur von T ab - nur eine davon angeben (p ohne Wirkung)")
        if not by_temperature:
            return f"{call}: T (oder h, s) angeben - p allein legt die Temperatur nicht fest"
        return None

    by_temperature = [n for g, n in zip(given, names) if g in ('T', 'H', 'U')]
    if len(by_temperature) > 1:
        return (f"{call}: {' und '.join(by_temperature)} hängen bei idealen Gasen beide nur von T ab "
                f"und legen den Zustand nicht zusammen fest - statt einer davon p angeben")
    others = [g for g in given if g in ('S', 'V', 'D')]
    has_temperature = bool(by_temperature) or ('P' in given and bool(others))
    has_pressure = 'P' in given or (bool(by_temperature) and bool(others))
    if not has_temperature:
        if 'P' in given:
            return f"{call}: T (oder h, u) angeben - p allein legt die Temperatur nicht fest"
        if len(names) == 1:
            return f"{call}: {names[0]} legt den Zustand nicht fest - zusätzlich T oder p angeben"
        return f"{call}: {' und '.join(names)} legen den Zustand nicht fest - statt einer davon T oder p angeben"
    if function == 'pressure' and not has_pressure:
        return f"{call}: der Druck folgt aus T zusammen mit s oder rho (bzw. v) - eine davon angeben"
    if function in _PRESSURE_DEPENDENT and not has_pressure:
        return (f"{call}: Druck p angeben - Entropie, freie Enthalpie, Dichte und Volumen eines "
                f"idealen Gases hängen vom Druck ab (es wird nicht still p = 1 bar angenommen)")
    return None


# ---------------------------------------------------------------------------
# Stoffwerte (massebezogen, SI)
# ---------------------------------------------------------------------------

def _invert(sp: Species, prop, target, what: str):
    """T mit prop(T) = target (prop monoton steigend), elementweise."""
    lo, hi = float(sp.t_min[0]), float(sp.t_max[-1])
    f_lo, f_hi = float(prop(lo)), float(prop(hi))

    def solve(value):
        value = float(value)
        if not np.isfinite(value) or not f_lo <= value <= f_hi:
            raise ValueError(f"{sp.label}: {what} = {value:.6g} liegt außerhalb des Datenbereichs "
                             f"{sp.range_text()} ({what} = {f_lo:.6g} ... {f_hi:.6g}, {DATA_SOURCE})")
        return brentq(lambda t: float(prop(t)) - value, lo, hi, xtol=1e-12, rtol=1e-14, maxiter=200)

    target = np.asarray(target, dtype=float)
    if target.ndim == 0:
        return solve(target)
    return np.vectorize(solve, otypes=[float])(target)


def _positive_pressure(sp: Species, p):
    p = np.asarray(p, dtype=float)
    if not np.all(p > 0):
        raise ValueError(f"{sp.label}: Druck p = {p} muss positiv sein")
    return p


def _state(sp: Species, inputs: Dict[str, Any], need_pressure: bool):
    """(T, p) aus den Eingaben; p None, wenn nicht nötig bzw. kondensiert."""
    R = sp.gas_constant
    T: Any = inputs.get('T')
    p: Any = inputs.get('P')
    if p is not None and not sp.condensed:
        p = _positive_pressure(sp, p)
    if T is None:
        if 'H' in inputs:
            T = _invert(sp, lambda t: R * t * _h_RT(_coefficients(sp, np.asarray(t)), t), inputs['H'], 'h')
        elif 'U' in inputs:
            T = _invert(sp, lambda t: R * t * (_h_RT(_coefficients(sp, np.asarray(t)), t) - 1),
                        inputs['U'], 'u')
        elif 'S' in inputs:
            s_standard = np.asarray(inputs['S'], dtype=float)
            if not sp.condensed:
                s_standard = s_standard + R * np.log(p / P_STANDARD)
            T = _invert(sp, lambda t: R * _s_R(_coefficients(sp, np.asarray(t)), t), s_standard, 's')
        else:
            v = inputs['V'] if 'V' in inputs else 1.0 / np.asarray(inputs['D'], dtype=float)
            T = p * np.asarray(v, dtype=float) / R
    T = _temperature_array(sp, T)
    if need_pressure and not sp.condensed and p is None:
        if 'S' in inputs:
            p = P_STANDARD * np.exp((R * _s_R(_coefficients(sp, T), T) - np.asarray(inputs['S'], float)) / R)
        else:
            v = inputs['V'] if 'V' in inputs else 1.0 / np.asarray(inputs['D'], dtype=float)
            p = R * T / np.asarray(v, dtype=float)
        p = _positive_pressure(sp, p)
    return T, p


def _result(value):
    return float(value) if np.ndim(value) == 0 else np.asarray(value, dtype=float)


def calculate_property(function: str, species: str, **kwargs):
    """
    Stoffwert function (enthalpy, entropy, gibbs, ...) der Spezies (Name ohne Präfix)
    in SI, massebezogen. Eingaben T, p, h, u, s, v, rho in SI (Skalar oder Array).
    """
    sp = find_species(species)
    function = function.lower()
    error = call_error(function, sp, list(kwargs))
    if error:
        raise ValueError(error)
    if function == 'molarmass':
        return sp.molar_mass
    inputs = {INPUT_MAP[k.lower()]: v for k, v in kwargs.items()}
    T, p = _state(sp, inputs, need_pressure=function in _PRESSURE_DEPENDENT)
    c = _coefficients(sp, T)
    R = sp.gas_constant

    if function == 'temperature':
        return _result(T)
    if function == 'pressure':
        return _result(p)
    if function == 'enthalpy':
        return _result(R * T * _h_RT(c, T))
    if function == 'intenergy':
        return _result(R * T * (_h_RT(c, T) - 1))
    if function == 'cp':
        return _result(R * _cp_R(c, T))
    if function == 'cv':
        return _result(R * (_cp_R(c, T) - 1))
    if function == 'soundspeed':
        cp_R = _cp_R(c, T)
        return _result(np.sqrt(cp_R / (cp_R - 1) * R * T))
    if function == 'volume':
        return _result(R * T / p)
    if function == 'density':
        return _result(p / (R * T))
    s = R * _s_R(c, T)
    if not sp.condensed:
        s = s - R * np.log(p / P_STANDARD)
    if function == 'entropy':
        return _result(s)
    if function == 'gibbs':
        return _result(R * T * _h_RT(c, T) - T * s)
    raise ValueError(f"Unbekannte Funktion: {function}")


if __name__ == "__main__":
    print("=== Ideale Gase (NASA thermo.inp) - Tests ===\n")
    data = database()
    n_condensed = sum(1 for sp in data.values() if sp.condensed)
    print(f"{len(data)} Spezies ({len(data) - n_condensed} Gase, {n_condensed} kondensiert)\n")

    def check(text, value, reference, tolerance):
        ok = abs(value - reference) <= tolerance
        print(f"  {'ok ' if ok else 'FEHLER'} {text}: {value:.6g} (Referenz {reference:.6g})")
        if not ok:
            raise SystemExit(1)

    T0 = 298.15
    for name, h_ref, s_ref in (('H2O', -241.826, 188.83), ('H2O(L)', -285.83, 69.95),
                               ('CO2', -393.51, 213.79), ('N2', 0.0, 191.61)):
        sp = find_species(name)
        check(f"h({name}, 25 °C) in kJ/mol", molar_enthalpy(sp, T0) / 1e3, h_ref, 0.01)
        check(f"s°({name}, 25 °C) in J/(mol K)", molar_entropy(sp, T0), s_ref, 0.02)

    F = 96485.33212
    def gibbs_molar(name, T, p=P_STANDARD):
        sp = find_species(name)
        return calculate_property('gibbs', name, T=T, p=p) * sp.molar_mass
    for product, T, E_ref in (('H2O', 298.15, 1.1845), ('H2O', 1073.15, 0.9769), ('H2O(L)', 298.15, 1.2289)):
        dG = gibbs_molar(product, T) - gibbs_molar('H2', T) - 0.5 * gibbs_molar('O2', T)
        check(f"E0 Brennstoffzelle {product}, {T - 273.15:g} °C in V", -dG / (2 * F), E_ref, 1e-3)

    h = lambda name: calculate_property('enthalpy', name, T=T0) * find_species(name).molar_mass
    H_u = -(h('CO2') + 2 * h('H2O') - h('CH4') - 2 * h('O2')) / find_species('CH4').molar_mass
    check("Heizwert CH4 in MJ/kg", H_u / 1e6, 50.03, 0.05)
    ds = calculate_property('entropy', 'N2', T=300, p=10e5) - calculate_property('entropy', 'N2', T=300, p=1e5)
    check("Δs N2, 1 -> 10 bar in J/(kg K)", ds, -find_species('N2').gas_constant * np.log(10), 1e-9)
    T_back = calculate_property('temperature', 'N2', h=calculate_property('enthalpy', 'N2', T=1234.5))
    check("T aus h (N2) in K", T_back, 1234.5, 1e-7)
    for name in ('co', 'IG_unbekannt'):
        try:
            find_species(name)
        except ValueError as exc:
            print(f"  ok  Meldung: {exc}")
    print("\nAlle Tests bestanden")
