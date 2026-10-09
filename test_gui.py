"""
Headless GUI-Regressionstests für main.py (EquationSolverApp).

Deckt die im GUI-Review (Oktober 2026) gefundenen und behobenen Fehler ab:
Datei-Operationen, Undo-Stack, Tastenkürzel, Startwerte-Dialog,
Re-Entrancy von solve(), Fehlermeldungen, Hilfetexte, Schriftgröße.

Ausführen mit:  python3 test_gui.py
Das Hauptfenster wird versteckt (withdraw); Dialoge von messagebox und
filedialog werden durch Attrappen ersetzt, damit nichts blockiert.
Hinweis: Der Tastenkürzel-Test holt das Fenster kurz in den Vordergrund -
währenddessen nicht tippen, sonst landen die Zeichen im Test-Editor.
"""
import faulthandler
import os
import re
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")
# Wachhund: hängt ein Test (z.B. modaler Dialog), Traceback + Abbruch
faulthandler.dump_traceback_later(600, exit=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from tkinter import filedialog, messagebox
import customtkinter as ctk

import main
from main import EquationSolverApp, FUNCTION_HELP_TEXT, FONT_SIZE_MIN, FONT_SIZE_MAX
from parser import parse_equations, display_name
from solver import solve_system

PASSED = []
FAILED = []


def check(name, cond, extra=""):
    (PASSED if cond else FAILED).append(name)
    print(("OK  " if cond else "FAIL"), name, extra)


# --- Attrappen für modale Dialoge -----------------------------------------
MESSAGES = []
for _name in ("showinfo", "showerror", "showwarning"):
    setattr(messagebox, _name, lambda *a, _n=_name, **k: MESSAGES.append((_n, a)))

_next_open_path = [""]
_next_save_path = [""]
_open_calls = []
filedialog.askopenfilename = lambda **k: (_open_calls.append(1), _next_open_path[0])[1]
filedialog.asksaveasfilename = lambda **k: _next_save_path[0]

TMP = tempfile.mkdtemp(prefix="test_gui_")

app = EquationSolverApp()
app.withdraw()
INITIAL_FONT_SIZE = app.font_size


def set_text(text):
    app.equations_text.delete("1.0", "end")
    app.equations_text.insert("1.0", text)


def get_text():
    return app.equations_text.get("1.0", "end-1c")


def solve(text):
    set_text(text)
    app.solve()


def shown_rows():
    """Variablennamen der angezeigten Ergebniszeilen (in Anzeigereihenfolge)."""
    return [row.winfo_children()[0].cget("text")
            for row in app.var_rows_container.winfo_children()]


def all_children(widget):
    for child in widget.winfo_children():
        yield child
        yield from all_children(child)


def open_dialog(show_func):
    """Öffnet einen Dialog und liefert das neue CTkToplevel-Fenster."""
    before = {str(w) for w in app.winfo_children()}
    show_func()
    app.update_idletasks()
    new = [w for w in app.winfo_children()
           if isinstance(w, ctk.CTkToplevel) and str(w) not in before]
    return new[-1] if new else None


def dialog_button(dialog, text):
    return [w for w in all_children(dialog)
            if isinstance(w, ctk.CTkButton) and w.cget("text") == text][0]


def value_entries(dialog):
    return [w for w in all_children(dialog) if isinstance(w, ctk.CTkEntry)]


def write_file(name, data: bytes):
    path = os.path.join(TMP, name)
    with open(path, "wb") as f:
        f.write(data)
    return path


# ---------------------------------------------------------------------------
print("=== #7 Konstanten + Sweep ohne Gleichungen ===")
solve("T = 20:10:50 °C\np = 1 bar")
sol = app.last_solution or {}
check("Sweep bleibt erhalten (T ist Array)",
      isinstance(sol.get("T"), np.ndarray) and len(sol["T"]) == 4)
check("Konstante bleibt erhalten (p)", abs(sol.get("p", 0) - 1e5) < 1e-6)
check("Beide Zeilen angezeigt", sorted(shown_rows()) == ["T", "p"], str(shown_rows()))

solve("p = 1 bar")
check("Nur Konstanten weiterhin OK",
      (app.last_solution or {}).get("p") == 1e5 and shown_rows() == ["p"])

# ---------------------------------------------------------------------------
print("\n=== #9 Fehlermeldung wird nicht überschrieben ===")
solve("a = 5\nb = 3\nc = a + b\na = b + 1")
info = app.info_label.cget("text")
check("Widerspruchs-Meldung sichtbar", "Widersprüch" in info, repr(info))
check("Teillösung trotzdem angezeigt", "c" in shown_rows())
check("Status: PARTIAL SOLUTION",
      "PARTIAL" in app.result_status_label.cget("text"))
check("Statusleiste: Contradictory system",
      app.status_label.cget("text") == "Contradictory system")

solve("x = 2\ny = x + 1\nz^2 = -1")
info = app.info_label.cget("text")
check("Konvergenz-Meldung des Solvers sichtbar (nicht 'Equations: ...')",
      info and not info.startswith("Equations:"), repr(info))

solve("x = 2\ny = x + 1")
check("Nach Erfolg: Info-Zeile wieder normal",
      app.info_label.cget("text").startswith("Equations:")
      and app.info_label.cget("text_color") == main.COLORS["text_dim"])

# ---------------------------------------------------------------------------
print("\n=== #10 Undo-Stack wird bei Open/New zurückgesetzt ===")
path_b = write_file("B.hes", "y = 2 {document B}\n".encode("utf-8"))
set_text("x = 1 {document A}")
_next_open_path[0] = path_b
app.open_file()
check("Open lädt Datei B", get_text() == "y = 2 {document B}\n")
app._undo()
app._undo()
check("Undo nach Open stellt Dokument A NICHT wieder her",
      get_text() == "y = 2 {document B}\n", repr(get_text()))
app.save_file()
with open(path_b, encoding="utf-8") as f:
    check("Save nach Undo überschreibt Datei B nicht", f.read() == "y = 2 {document B}\n")

set_text("x = 1 {document A}")
app.new_file()
app._undo()
check("Undo nach New stellt altes Dokument NICHT wieder her", get_text() == "")

set_text("q = 1")
app.clear_all()
app._undo()
check("Clear All bleibt per Undo rückgängig machbar", get_text() == "q = 1")

# ---------------------------------------------------------------------------
print("\n=== #11 Fehlgeschlagenes Save As ===")
app.new_file()
set_text("x = 1")
MESSAGES.clear()
_next_save_path[0] = "/nonexistent_dir_test_gui/file.hes"
app.save_file_as()
check("Fehlermeldung angezeigt", any(m[0] == "showerror" for m in MESSAGES))
check("current_file NICHT gesetzt", app.current_file is None)
check("Label NICHT 'Saved'", "Saved" not in app.file_label.cget("text"),
      app.file_label.cget("text"))

good_path = os.path.join(TMP, "ok.hes")
_next_save_path[0] = good_path
app.save_file_as()
check("Erfolgreiches Save As setzt current_file + Label",
      app.current_file == good_path and "Saved" in app.file_label.cget("text"))

# ---------------------------------------------------------------------------
print("\n=== #12 Tastenkürzel verändern den Text nicht ===")
app.deiconify()
app.update()
tb = app.equations_text._textbox
original = "T_1 = 450 °C\nh = enthalpy(water, T=T_1, p=1 bar)"
for seq in ("<Control-o>", "<Control-s>", "<Control-plus>", "<Control-minus>",
            "<Control-equal>"):
    set_text(original)
    _next_open_path[0] = ""            # Open-Dialog "abgebrochen"
    app.current_file = good_path       # Save schreibt in Temp-Datei
    tb.focus_force()
    tb.mark_set("insert", "1.7")       # Cursor mitten in "450"
    app.update()
    n_open = len(_open_calls)
    tb.event_generate(seq)
    app.update()
    check(f"{seq} im Editor lässt Text unverändert", get_text() == original,
          repr(get_text()[:30]))
    if seq == "<Control-o>":
        check("<Control-o> öffnet genau einen Dialog", len(_open_calls) == n_open + 1)
app.set_font_size(main.FONT_SIZE_DEFAULT)
app.withdraw()
app.update()

# ---------------------------------------------------------------------------
print("\n=== #13 Manuelle Startwerte ===")
app.new_file()
solve("x^2 = 9")
app.manual_initial_values = {"x": -3.0}
solve("x^2 = 9")
check("Manueller Startwert wirkt (x = -3)", abs(app.last_solution["x"] + 3) < 1e-6)
app.new_file()
check("New löscht manuelle Startwerte", app.manual_initial_values == {})
solve("x^2 = 16")
check("Nach New: Standard-Wurzel x = +4", abs(app.last_solution["x"] - 4) < 1e-6)

app.manual_initial_values = {"x": -3.0}
_next_open_path[0] = write_file("C.hes", b"x^2 = 16\n")
app.open_file()
check("Open löscht manuelle Startwerte", app.manual_initial_values == {})

# Dialog: OK ohne Eingabe speichert KEINE grauen Auto-Werte
solve("T_1 = 20 °C\np = 1 bar\nh_x = 100 kJ/kg\nh_x = enthalpy(water, T=T_x, p=p)")
dlg = open_dialog(app.show_initial_values_dialog)
entries = value_entries(dlg)
auto_expected = f"{main.get_initial_from_unit(app.inferred_units.get('T_x')):.6g}"
check("Dialog zeigt grauen Auto-Wert", len(entries) == 1 and entries[0].get() == auto_expected,
      str([e.get() for e in entries]))
check("Keine editierbare Einheiten-ComboBox mehr (manual_units entfernt)",
      not any(isinstance(w, ctk.CTkComboBox) for w in all_children(dlg)))
dialog_button(dlg, "OK").invoke()
app.update_idletasks()
check("OK ohne Eingabe speichert nichts", app.manual_initial_values == {},
      str(app.manual_initial_values))
check("Kein manual_units-Attribut", not hasattr(app, "manual_units"))

dlg = open_dialog(app.show_initial_values_dialog)
entry = value_entries(dlg)[0]
entry.delete(0, "end")
entry.insert(0, "300")
dialog_button(dlg, "OK").invoke()
app.update_idletasks()
check("Eingetippter Wert wird gespeichert", app.manual_initial_values == {"T_x": 300.0})

dlg = open_dialog(app.show_initial_values_dialog)
entry = value_entries(dlg)[0]
check("Gespeicherter Wert wird wieder angezeigt", entry.get() == "300")
entry.delete(0, "end")
entry.insert(0, "abc")
MESSAGES.clear()
dialog_button(dlg, "OK").invoke()
app.update_idletasks()
check("Ungültige Eingabe: Fehler + alte Werte bleiben",
      any(m[0] == "showerror" for m in MESSAGES)
      and app.manual_initial_values == {"T_x": 300.0})
dialog_button(dlg, "Cancel").invoke()
app.update_idletasks()
app.manual_initial_values = {}

# ---------------------------------------------------------------------------
print("\n=== #14 Kein Re-Entrancy in solve() ===")
set_text("a = 2\nb = a*3")
app.after(0, app.solve)      # zweites F5, während der erste Lauf update() ruft
app.solve()
app.update()
check("Ergebniszeilen nicht doppelt", shown_rows() == ["a", "b"], str(shown_rows()))
check("Guard nach dem Lösen zurückgesetzt", app._solving is False)
check("Solve-Button wieder aktiv", app.solve_btn.cget("state") == "normal")

calls = []
app._solving = True
set_text("q = 1")
app.new_file()
app.clear_all()
check("New/Clear während des Lösens ignoriert", get_text() == "q = 1")
app._solving = False

# ---------------------------------------------------------------------------
print("\n=== #15 Hilfetexte in SI ===")
check("Hilfe nennt keine bar/kJ-Interneinheiten",
      not re.search(r"\[(bar|kJ/kg|kJ/\(kg K\))\]", FUNCTION_HELP_TEXT))
check("Hilfe nennt [Pa] und [J/kg]",
      "Pressure [Pa]" in FUNCTION_HELP_TEXT and "enthalpy [J/kg]" in FUNCTION_HELP_TEXT)
check("Keine Beispiele mit p=1 / p_tot=1 ohne Einheit",
      not re.search(r"p(_tot)?=1\s*\)", FUNCTION_HELP_TEXT))

# Alle Funktionsaufruf-Beispiele müssen lösbar sein (mit Konstanten-Kontext)
results = {}
for block in re.split(r"\n\s*\n", FUNCTION_HELP_TEXT):
    context = []
    for line in block.splitlines():
        code = re.sub(r"\s*\{[^}]*\}\s*$", "", line.strip())
        if re.match(r"^[A-Za-z_]\w* = [-\d.]+\s*\S*$", code):
            context.append(code)                     # z.B. T_s = 500 °C
        elif re.match(r"^[A-Za-z_]\w* = \w+\(.*\)$", code) and "(...)" not in code:
            eqs, variables, consts, _, orig, _ = parse_equations("\n".join(context + [code]))
            ok, sol, msg = solve_system(eqs, variables, {}, constants=consts,
                                        original_equations=orig)
            var = code.split("=")[0].strip()
            results[code] = (ok, sol.get(var))
            check(f"Hilfe-Beispiel lösbar: {code}", ok, "" if ok else msg)
h_water = results.get("h = enthalpy(water, T=100 °C, p=1 bar)", (False, 0))[1] or 0
check("enthalpy(water, 100 °C, 1 bar) ≈ 2675.8 kJ/kg", abs(h_water - 2675766) < 100)
h_air = results.get("h = HumidAir(h, T=25 °C, rh=0.5, p_tot=1 bar)", (False, 0))[1] or 0
check("HumidAir(h, 25 °C, 50 %, 1 bar) ≈ 50.77 kJ/kg", abs(h_air - 50766) < 50)

app._insert_example()
check("Beispiel-Kopf nennt SI-Einheiten", "p[Pa], h[J/kg]" in get_text())
app.solve()
check("Eingebautes Beispiel löst", "SOLUTION FOUND" in app.result_status_label.cget("text"))

# ---------------------------------------------------------------------------
print("\n=== #16 Diverses ===")
# Kein zweites EquationSolverApp (zweiter Tk-Root) - das lässt update()
# in späteren solve()-Aufrufen sporadisch hängen; Startwert oben gemerkt
check("Standard-Schriftgröße 16", INITIAL_FONT_SIZE == 16, str(INITIAL_FONT_SIZE))
app.set_font_size(100)
check("Schriftgröße max. 36", app.font_size == FONT_SIZE_MAX == 36)
app.set_font_size(1)
check("Schriftgröße min. 6", app.font_size == FONT_SIZE_MIN == 6)
app.set_font_size(main.FONT_SIZE_DEFAULT)

solve("a = 2\nb = a*3")
app.new_file()
check("New löscht last_solution", app.last_solution is None)
solve("a = 2\nb = a*3")
app.clear_all()
check("Clear löscht last_solution", app.last_solution is None)
solve("a = 2\nb = a*3")
_next_open_path[0] = path_b
app.open_file()
check("Open löscht last_solution", app.last_solution is None)

_next_open_path[0] = write_file("latin1.txt", "T_1 = 450 °C\nT_2 = T_1 + 10\n".encode("latin-1"))
MESSAGES.clear()
app.open_file()
check("Latin-1-Datei öffnet ohne Fehler", not MESSAGES and "450 °C" in get_text(),
      repr(get_text()[:20]))
app.solve()
check("Latin-1-Datei: T_1 = 723.15 K", abs((app.last_solution or {}).get("T_1", 0) - 723.15) < 1e-9)

_next_open_path[0] = write_file("bom.hes", "﻿x = 3\n".encode("utf-8"))
app.open_file()
check("UTF-8-BOM wird entfernt", get_text() == "x = 3\n", repr(get_text()))

_next_open_path[0] = os.path.join(TMP, "does_not_exist.hes")
MESSAGES.clear()
set_text("keep me")
app.open_file()
check("Nicht lesbare Datei: Fehler, Editor unverändert",
      any(m[0] == "showerror" for m in MESSAGES) and get_text() == "keep me")

# ---------------------------------------------------------------------------
print("=== Einheiten-Anzeige (Wert und Label aus derselben Einheit) ===")


def shown(var):
    """(Wert als float, Einheiten-Label) der Ergebniszeile einer Variable."""
    text = app.value_labels[var].cget("text")
    dd = app.unit_dropdowns.get(var)
    if dd is not None:
        unit = dd.get()
    else:
        unit = next(row.winfo_children()[1].cget("text")
                    for row in app.var_rows_container.winfo_children()
                    if row.winfo_children()[0].cget("text") == display_name(var))
    try:
        value = float(text)
    except ValueError:
        value = text
    return value, unit


def close(value, ref, rtol=1e-4):
    return isinstance(value, float) and abs(value - ref) <= rtol * abs(ref)


solve("T_h_in = 90 °C\nT_c_out = 40 °C\nT_c_in = 20 °C\n"
      "dT_1 = T_h_in - T_c_out\ntheta = T_h_in - T_c_in")
check("Temperaturdifferenz dT_1 = 50 delta_K (nicht -223.15 °C)", shown("dT_1") == (50.0, "delta_K"),
      str(shown("dT_1")))
check("Temperaturdifferenz theta = 70 delta_K", shown("theta") == (70.0, "delta_K"), str(shown("theta")))

solve("R_si = 0.13 m^2*K/W\nd_1 = 20 cm\nlambda_1 = 2.3 W/mK\nT_i = 20 °C\nT_e = -10 °C\n"
      "U = 1/(R_si + d_1/lambda_1 + 0.04)\nq = U*(T_i - T_e)\nT_si = T_i - q*R_si")
v, u = shown("T_si")
check("Oberflächentemperatur T_si = 4.82 °C (absolut, nicht delta_K)", close(v, 4.82164, 1e-3) and u == "degC",
      str((v, u)))
v, u = shown("U")
check("U-Wert in W/(m²K), nicht 0.0039 kW/m²K", close(v, 3.89171) and u.startswith("W/"), str((v, u)))

solve("p = 1 bar\nT_1 = 20 °C\nm_dot = 2 kg/s\nh_1 = enthalpy(water, T=T_1, p=p)\n"
      "Q_dot = m_dot*(enthalpy(water, T=80 °C, p=p) - h_1)")
v, u = shown("Q_dot")
check("Funktion im Ausdruck: Q_dot in kW (nicht kJ/kg)", close(v, 502.096) and u == "kW", str((v, u)))

solve("T = 1000 K\nlambda_max = Wien(T)\nL = 5 µm\nE_l = Eb(T, L)")
v, u = shown("lambda_max")
check("Wien: 2.898 µm (nicht 2.9e6 µm)", close(v, 2.89777) and u == "um", str((v, u)))
v, u = shown("E_l")
check("Eb: 7139.6 W/(m²·µm)", close(v, 7139.62) and u == "W/(m^2*um)", str((v, u)))
check("Eingabe 5 µm wird als 5 µm angezeigt", shown("L") == (5.0, "µm"), str(shown("L")))

solve("P_el = 2 MW\nUA = 500 W/K\nE = 3 MJ\nx = P_el*2")
check("2 MW bleibt '2 MW' (nicht '2000 MW')", shown("P_el") == (2.0, "MW"), str(shown("P_el")))
check("500 W/K bleibt '500 W/K' (nicht 'kW/K')", shown("UA") == (500.0, "W/K"), str(shown("UA")))
check("3 MJ bleibt '3 MJ'", shown("E") == (3.0, "MJ"), str(shown("E")))
v, u = shown("x")
check("Berechnete Leistung in kW (Settings)", close(v, 4000.0) and u == "kW", str((v, u)))

solve("T = 20:10:50 °C\nh = enthalpy(water, T=T, p=1 bar)")
check("Sweep T in °C angezeigt (nicht 293→323)",
      app.value_labels["T"].cget("text") == "[4× 20→50]" and shown("T")[1] == "degC",
      app.value_labels["T"].cget("text"))
check("Sweep h in kJ/kg angezeigt", shown("h")[1] == "kJ/kg" and "84.01" in app.value_labels["h"].cget("text"),
      app.value_labels["h"].cget("text"))
plot_T, plot_unit = app._display_values("T", app.last_solution["T"])
check("Plot-Daten in Anzeige-Einheit (°C)", np.allclose(plot_T, [20, 30, 40, 50]) and plot_unit == "degC")

solve("lambda = 0.6 W/mK\nd = 0.1 m\nalpha = 2*lambda/d")
check("Schlüsselwort 'lambda' als Variable, Anzeige als 'lambda'",
      "lambda" in shown_rows() and "_kw_lambda" not in shown_rows(), str(shown_rows()))
check("alpha = 12 aus lambda berechnet", close(shown("alpha")[0], 12.0))

solve("epsilon = 0.85\nsigma = 5.67E-8 W/(m^2*K^4)\nA = 2 m^2\nT_s = 500 °C\n"
      "Q_rad = epsilon*sigma*A*T_s^4")
v, u = shown("Q_rad")
check("Blatt mit Einheiten: Konstante ohne Einheit dimensionslos -> Q_rad in kW",
      close(v, 34.4419) and u == "kW", str((v, u)))

solve("m_dot = 2.78\np_1 = 3000000\nT_1 = 723.15\nh_1 = enthalpy(water, T=T_1, p=p_1)\n"
      "h_2 = enthalpy(water, T=373.15, p=100000)\nW_dot = m_dot*(h_1 - h_2)")
check("Reines Zahlen-Blatt: W_dot ohne (falsches) kJ/kg-Label", shown("W_dot")[1] == "-",
      str(shown("W_dot")))
check("Reines Zahlen-Blatt: h_1 = enthalpy(...) weiterhin in kJ/kg", shown("h_1")[1] == "kJ/kg")

solve("T_1 = 20 °C\np = 1 bar\nh_1 = enthalpy(water, T=T_1, p=p)\nexp(z) = -1")
v, u = shown("h_1")
check("Teillösung: Werte mit Einheit (h_1 in kJ/kg)", close(v, 84.0061) and u == "kJ/kg", str((v, u)))

# ---------------------------------------------------------------------------
app.destroy()
faulthandler.cancel_dump_traceback_later()

print()
print(f"{len(PASSED)}/{len(PASSED) + len(FAILED)} Tests bestanden")
if FAILED:
    print("FEHLGESCHLAGEN:", *FAILED, sep="\n  - ")
    sys.exit(1)
print("ALLE TESTS OK")
