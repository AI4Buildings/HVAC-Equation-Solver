"""
Naturkonstanten und Normwerte für das Gleichungsblatt.

Im Blatt stehen sie mit einem angehängten '#': sigma#, g#, R# ... Damit haben sie einen eigenen
Namensraum - eine Größe sigma, g oder R des Blatts bleibt frei und kann eine Konstante nicht
überschreiben. Der Parser benennt name# intern in _cn_name um (parser.mangle_keywords, Anzeige
wieder name#) und trägt jede verwendete Konstante mit SI-Wert und Einheit als Vorgabe ein - für
Solver, Einheiten-Ableitung und Anzeige ist sie damit eine gewöhnliche gegebene Größe.

Werte: SI 2019 (exakt festgelegte Konstanten) bzw. CODATA 2018; sigma, R und die Strahlungskonstanten
wie in radiation.py und ideal_gas.py (sigma#*T^4 ist exakt Stefan_Boltzmann(T)).
"""
from typing import Dict, NamedTuple, Optional

try:
    from radiation import SIGMA as _SIGMA
except ImportError:             # pragma: no cover
    _SIGMA = 5.670374419e-8
try:
    from ideal_gas import R_UNIVERSAL as _R
except ImportError:             # pragma: no cover
    _R = 8.314462618

MARKER = '#'                    # sigma# im Blatt
PREFIX = '_cn_'                 # interner Name: _cn_sigma


class Constant(NamedTuple):
    name: str                   # Name im Blatt ohne '#'
    value: float                # SI-Wert
    unit: str                   # Einheit in der Schreibweise des Blatts
    description: str            # für Menü und Hilfe (die Oberfläche ist englisch)


_TABLE = (
    Constant('sigma', _SIGMA, 'W/(m^2*K^4)', 'Stefan-Boltzmann constant'),
    Constant('g', 9.80665, 'm/s^2', 'standard gravity'),
    Constant('R', _R, 'J/(mol*K)', 'molar gas constant'),
    Constant('k', 1.380649e-23, 'J/K', 'Boltzmann constant'),
    Constant('N_A', 6.02214076e23, '1/mol', 'Avogadro constant'),
    Constant('F', 96485.33212, 'C/mol', 'Faraday constant'),
    Constant('h', 6.62607015e-34, 'J*s', 'Planck constant'),
    Constant('c', 299792458.0, 'm/s', 'speed of light in vacuum'),
    Constant('e', 1.602176634e-19, 'C', 'elementary charge'),
    Constant('C_1', 3.741771852e-16, 'W*m^2', 'first radiation constant'),
    Constant('C_2', 1.4387768775e-2, 'm*K', 'second radiation constant'),
    Constant('b', 2.897771955e-3, 'm*K', 'Wien displacement constant'),
    Constant('p_n', 101325.0, 'Pa', 'standard pressure (1 atm)'),
    Constant('T_n', 273.15, 'K', 'standard temperature (0 °C)'),
)

CONSTANTS: Dict[str, Constant] = {constant.name: constant for constant in _TABLE}


def lookup(name: str) -> Optional[Constant]:
    """Konstante zum Namen (ohne '#') oder None."""
    return CONSTANTS.get(name)


def sheet_name(name: str) -> str:
    """Schreibweise im Blatt: 'sigma' -> 'sigma#'."""
    return name + MARKER


def names_text() -> str:
    """'sigma#, g#, R#, ...' für Meldungen."""
    return ', '.join(sheet_name(constant.name) for constant in _TABLE)


def table_lines(width: int = 70) -> list:
    """Zeilen 'sigma#   5.670374419e-08 W/(m^2*K^4)   Stefan-Boltzmann constant' für die Hilfe."""
    lines = []
    for constant in _TABLE:
        line = f"  {sheet_name(constant.name):<7} {constant.value:<15.10g} {constant.unit:<12} {constant.description}"
        if len(line) > width:
            raise ValueError(f"Zeile der Konstanten-Tabelle zu lang ({len(line)} > {width}): {line}")
        lines.append(line)
    return lines


if __name__ == '__main__':
    for text in table_lines(200):
        print(text)
