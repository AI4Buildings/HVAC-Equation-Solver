"""
Generische Fehleranalyse für Gleichungssysteme.

Alle Verfahren arbeiten auf der STRUKTUR des Systems (welche Gleichung enthält
welche Unbekannte) bzw. auf den Namen - nie auf der Form einzelner Gleichungen.
Sie gelten damit für beliebige Aufgaben:

1. Strukturanalyse (Dulmage-Mendelsohn-Zerlegung des bipartiten Graphen
   Gleichungen <-> Unbekannte über ein maximales Matching):
   - unterbestimmter Teil: Unbekannte, für die Gleichungen fehlen
   - überbestimmter Teil: Gleichungen, die dieselben Unbekannten mehrfach festlegen
   - wohlbestimmter Rest: lösbar (Blockzerlegung macht der Solver)
2. Numerische Diagnose: Unbekannte, die trotz korrekter Struktur nicht gelöst wurden
3. Namens-Hinweise: Eine Unbekannte, die nur in EINER Gleichung vorkommt und
   aus ihr implizit berechnet wird, deren Name aber einem vorhandenen Namen bis
   auf Schreibweise gleicht oder aus zwei vorhandenen Namen zusammengesetzt ist
   (z.B. Tippfehler r_1h_i statt r_1/h_i). Nur Hinweis, nie Fehler - ein
   strukturell korrektes System kann einen Tippfehler nicht beweisen.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from solver import _get_equation_unknowns, _direct_assignment

try:
    from parser import unmangle
except ImportError:  # pragma: no cover
    def unmangle(text):
        return text


@dataclass
class StructureReport:
    """Ergebnis der Strukturanalyse."""
    under_equations: List[str] = field(default_factory=list)  # unterbestimmter Teil
    under_variables: List[str] = field(default_factory=list)
    over_equations: List[str] = field(default_factory=list)   # überbestimmter Teil (mit Unbekannten)
    over_variables: List[str] = field(default_factory=list)
    check_equations: List[str] = field(default_factory=list)  # Gleichungen ohne Unbekannte (Prüfgleichungen)

    @property
    def missing_equations(self) -> int:
        return len(self.under_variables) - len(self.under_equations)

    @property
    def surplus_equations(self) -> int:
        return len(self.over_equations) - len(self.over_variables)


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _incidence(equations: List[str], unknowns: Set[str]) -> Dict[str, List[str]]:
    """Gleichung -> sortierte Liste ihrer Unbekannten."""
    return {eq: sorted(_get_equation_unknowns(eq, set(), unknowns)) for eq in equations}


def _maximum_matching(equations: List[str], incidence: Dict[str, List[str]]
                      ) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Maximales Matching Gleichung <-> Unbekannte (augmentierende Pfade, BFS).
    Deterministisch: Gleichungen in Eingabereihenfolge, Unbekannte sortiert.
    """
    eq_match: Dict[str, str] = {}
    var_match: Dict[str, str] = {}
    for start in equations:
        parent: Dict[str, str] = {}
        queue = [start]
        visited: Set[str] = set()
        free_var = None
        while queue and free_var is None:
            eq = queue.pop(0)
            for var in incidence[eq]:
                if var in visited:
                    continue
                visited.add(var)
                parent[var] = eq
                if var not in var_match:
                    free_var = var
                    break
                queue.append(var_match[var])
        # Pfad umkehren (augmentieren)
        var = free_var
        while var is not None:
            eq = parent[var]
            previous = eq_match.get(eq)
            eq_match[eq] = var
            var_match[var] = eq
            var = previous if eq != start else None
    return eq_match, var_match


def _source_line_numbers(original_equations: Dict[str, str], source_text: Optional[str]) -> Dict[str, int]:
    """Gleichung -> Zeilennummer im Editortext (über die Originalzeile)."""
    if not source_text:
        return {}
    lines = [line.strip() for line in source_text.split('\n')]
    used: Set[int] = set()
    numbers = {}
    for parsed, original in original_equations.items():
        target = unmangle(original.strip())
        for index, line in enumerate(lines):
            if index not in used and line == target:
                numbers[parsed] = index + 1
                used.add(index)
                break
    return numbers


def _format_lines(equations: List[str], line_numbers: Dict[str, int],
                  original_equations: Dict[str, str]) -> str:
    """'Zeile 11, 14' bzw. die Gleichungen selbst, wenn Zeilen unbekannt sind."""
    numbers = sorted(line_numbers[eq] for eq in equations if eq in line_numbers)
    if len(numbers) == len(equations) and numbers:
        label = "Zeile" if len(numbers) == 1 else "Zeilen"
        return f"{label} {', '.join(str(n) for n in numbers)}"
    return "; ".join(f"'{unmangle(original_equations.get(eq, eq))}'" for eq in equations)


def _names(variables) -> str:
    return ", ".join(unmangle(v) for v in sorted(variables))


# ---------------------------------------------------------------------------
# 1. Strukturanalyse
# ---------------------------------------------------------------------------

def analyze_structure(equations: List[str], unknowns: Set[str]) -> StructureReport:
    """
    Dulmage-Mendelsohn-Zerlegung: teilt das System in unterbestimmten,
    überbestimmten und wohlbestimmten Teil.
    """
    unknowns = set(unknowns)
    incidence = _incidence(equations, unknowns)
    eq_match, var_match = _maximum_matching(equations, incidence)
    var_to_eqs: Dict[str, List[str]] = {v: [] for v in unknowns}
    for eq in equations:
        for var in incidence[eq]:
            var_to_eqs[var].append(eq)

    report = StructureReport()
    report.check_equations = [eq for eq in equations if not incidence[eq]]

    # Unterbestimmt: von freien (nicht zugeordneten) Unbekannten aus über
    # alternierende Pfade erreichbar
    under_vars = {v for v in unknowns if v not in var_match}
    under_eqs: Set[str] = set()
    stack = sorted(under_vars)
    while stack:
        var = stack.pop()
        for eq in var_to_eqs[var]:
            if eq not in under_eqs:
                under_eqs.add(eq)
                matched = eq_match.get(eq)
                if matched is not None and matched not in under_vars:
                    under_vars.add(matched)
                    stack.append(matched)

    # Überbestimmt: von freien Gleichungen (mit Unbekannten) aus erreichbar
    over_eqs = {eq for eq in equations if eq not in eq_match and incidence[eq]}
    over_vars: Set[str] = set()
    stack = [eq for eq in equations if eq in over_eqs]
    while stack:
        eq = stack.pop()
        for var in incidence[eq]:
            if var not in over_vars:
                over_vars.add(var)
                matched = var_match.get(var)
                if matched is not None and matched not in over_eqs:
                    over_eqs.add(matched)
                    stack.append(matched)

    report.under_equations = [eq for eq in equations if eq in under_eqs]
    report.under_variables = sorted(under_vars)
    report.over_equations = [eq for eq in equations if eq in over_eqs]
    report.over_variables = sorted(over_vars)
    return report


# ---------------------------------------------------------------------------
# 3. Namens-Hinweise
# ---------------------------------------------------------------------------

def _normalized(name: str) -> str:
    return unmangle(name).replace('_', '').lower()


def _split_into_names(name: str, names: Set[str]) -> Optional[Tuple[str, str]]:
    """name == a + b mit a, b vorhandene Namen (je mind. 2 Zeichen)."""
    for i in range(2, len(name) - 1):
        a, b = name[:i], name[i:]
        if a in names and b in names:
            return a, b
    return None


def name_hints(equations: List[str], unknowns: Set[str], known_names: Set[str],
               original_equations: Dict[str, str], source_text: Optional[str] = None) -> List[str]:
    """
    Hinweise auf mögliche Tippfehler in Variablennamen (siehe Moduldoku, Punkt 3).
    """
    line_numbers = _source_line_numbers(original_equations, source_text)
    all_names = set(unknowns) | set(known_names)
    hints = []
    for var in sorted(unknowns):
        containing = [eq for eq in equations if var in _get_equation_unknowns(eq, set(), {var})]
        if len(containing) != 1:
            continue
        eq = containing[0]
        definition = _direct_assignment(eq)
        if definition and definition[0] == var:
            continue  # Ergebnisgröße "var = ausdruck" - normal
        others = all_names - {var}
        reasons = []
        parts = _split_into_names(var, others)
        if parts:
            reasons.append(f"der Name besteht aus {unmangle(parts[0])} und {unmangle(parts[1])} "
                           f"(fehlt ein Operator?)")
        twins = sorted(n for n in others if _normalized(n) == _normalized(var))
        if twins:
            reasons.append(f"ähnelt {_names(twins)} (bis auf Schreibweise)")
        if reasons:
            where = _format_lines([eq], line_numbers, original_equations)
            hints.append(f"{unmangle(var)} kommt nur in {where} vor und wird daraus berechnet; "
                         + "; ".join(reasons) + " - Tippfehler?")
    return hints


# ---------------------------------------------------------------------------
# Meldungstexte
# ---------------------------------------------------------------------------

def describe_structure(report: StructureReport, original_equations: Dict[str, str],
                       source_text: Optional[str] = None, include_over: bool = True) -> List[str]:
    """Verständliche Meldungen zum unter-/überbestimmten Teil."""
    line_numbers = _source_line_numbers(original_equations, source_text)
    messages = []
    if report.under_variables:
        missing = report.missing_equations
        n_eq = len(report.under_equations)
        if n_eq:
            where = _format_lines(report.under_equations, line_numbers, original_equations)
            subject = "Die Gleichung in" if n_eq == 1 else "Die Gleichungen in"
            verb, reach = ("enthält", "reicht") if n_eq == 1 else ("enthalten", "reichen")
            lack = ("fehlt 1 Gleichung bzw. Vorgabe" if missing == 1
                    else f"fehlen {missing} Gleichungen bzw. Vorgaben")
            messages.append(
                f"Unterbestimmt: {subject} {where} {verb} {len(report.under_variables)} Unbekannte "
                f"({_names(report.under_variables)}), {reach} aber nur für {n_eq} - es {lack}.")
        else:
            messages.append(f"Unterbestimmt: Für {_names(report.under_variables)} gibt es keine Gleichung.")
    if include_over and report.over_equations:
        where = _format_lines(report.over_equations, line_numbers, original_equations)
        surplus = report.surplus_equations
        subject = "Die Gleichung in" if len(report.over_equations) == 1 else "Die Gleichungen in"
        verb = "legt" if len(report.over_equations) == 1 else "legen"
        extra = "1 Gleichung bzw. Vorgabe" if surplus == 1 else f"{surplus} Gleichungen bzw. Vorgaben"
        messages.append(
            f"Überbestimmt: {subject} {where} {verb} {_names(report.over_variables) or 'dieselben Größen'} "
            f"mehrfach fest ({extra} zu viel).")
    return messages


def describe_unsolved(equations: List[str], unknowns: Set[str], solution: Dict[str, float],
                      original_equations: Dict[str, str], source_text: Optional[str] = None) -> str:
    """Numerische Diagnose: welche Unbekannten wurden nicht gelöst, in welchen Gleichungen."""
    unsolved = {v for v in unknowns if v not in solution}
    if not unsolved:
        return ""
    line_numbers = _source_line_numbers(original_equations, source_text)
    involved = [eq for eq in equations if _get_equation_unknowns(eq, set(), unsolved)]
    where = _format_lines(involved, line_numbers, original_equations)
    return (f"Keine numerische Lösung für {_names(unsolved)} ({where}). Mögliche Ursachen: "
            f"keine reelle Lösung, Definitionsbereich verlassen (z.B. ln, sqrt, Stoffwerte) "
            f"oder ungünstige Startwerte (Solve → Initial Values).")


def diagnose(equations: List[str], unknowns: Set[str], constants: Dict[str, float],
             original_equations: Dict[str, str], source_text: Optional[str] = None,
             solution: Optional[Dict[str, float]] = None) -> Tuple[List[str], List[str]]:
    """
    Gesamtdiagnose.

    Returns:
        (fehler, hinweise) - Fehler: struktureller unterbestimmter Teil bzw.
        numerisch ungelöste Unbekannte; Hinweise: Namens-Hinweise.
        Ein überbestimmter Teil allein ist kein Fehler (der Solver prüft, ob die
        zusätzlichen Gleichungen erfüllt sind) und wird hier nicht gemeldet.
    """
    report = analyze_structure(equations, unknowns)
    errors = describe_structure(report, original_equations, source_text, include_over=False)
    if not errors and solution is not None:
        unsolved_text = describe_unsolved(equations, unknowns, solution, original_equations, source_text)
        if unsolved_text:
            errors.append(unsolved_text)
    hints = name_hints(equations, unknowns, set(constants), original_equations, source_text)
    return errors, hints
