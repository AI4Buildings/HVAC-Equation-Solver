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

print("\n=== Zwischenablage: Einfügen, Kopieren, Ausschneiden ===")
import subprocess
# Die Tests benutzen die echte System-Zwischenablage: Textinhalt sichern und
# am Ende wiederherstellen (Bilder o.ä. lassen sich so nicht sichern)
_clipboard_backup = None
if sys.platform == "darwin":
    _clipboard_backup = subprocess.run(["pbpaste"], capture_output=True).stdout
paste_keys = ["<<Paste>>"] + (["<Command-v>"] if sys.platform == "darwin" else ["<Control-v>"])
for how in paste_keys:
    set_text("x = 1\n")
    tb.edit_reset()
    tb.focus_force()
    app.clipboard_clear()
    app.clipboard_append("A = 20 cm2\nT_1 = 20 °C")
    tb.mark_set("insert", "end")
    tb.event_generate(how)
    app.update()
    check(f"Einfügen per {how}", get_text() == "x = 1\nA = 20 cm2\nT_1 = 20 °C", repr(get_text()))
tb.event_generate("<Command-z>" if sys.platform == "darwin" else "<Control-z>")
app.update()
check("Rückgängig nach Einfügen", get_text() == "x = 1\n", repr(get_text()))
set_text("q_dot = 50 W/m2")
tb.tag_add("sel", "1.0", "1.5")
tb.event_generate("<<Copy>>")
app.update()
check("Kopieren in die Zwischenablage", app.clipboard_get() == "q_dot" and get_text() == "q_dot = 50 W/m2")
tb.tag_add("sel", "1.0", "1.8")
tb.event_generate("<<Cut>>")
app.update()
check("Ausschneiden", app.clipboard_get() == "q_dot = " and get_text() == "50 W/m2", repr(get_text()))

# Kopieren übergibt den Text fest an die System-Zwischenablage
exported = []
_original_export = main.export_to_system_clipboard
main.export_to_system_clipboard = lambda text: exported.append(text) or True
set_text("T_1 = 20 °C")
tb.tag_add("sel", "1.0", "end-1c")
tb.event_generate("<<Copy>>")
app.update()
main.export_to_system_clipboard = _original_export
check("Kopieren -> Export an System-Zwischenablage", exported == ["T_1 = 20 °C"], str(exported))

if sys.platform == "darwin":
    # Echter Ablauf: kopieren, Programm beenden, danach einfügen
    child = (
        "import sys, warnings; warnings.filterwarnings('ignore');"
        f"sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r});"
        "from main import EquationSolverApp;"
        "app = EquationSolverApp(); app.deiconify(); app.update();"
        "tb = app.equations_text._textbox;"
        "app.equations_text.insert('1.0', 'A = 20 cm² ⋅ µm'); tb.focus_force(); app.update();"
        "tb.tag_add('sel', '1.0', 'end-1c'); tb.event_generate('<<Copy>>'); app.update();"
        "app.destroy()")
    subprocess.run(["pbcopy"], input=b"vorher")
    subprocess.run([sys.executable, "-c", child], capture_output=True, timeout=60)
    after_exit = subprocess.run(["pbpaste"], capture_output=True,
                                env=dict(os.environ, LANG="en_US.UTF-8")).stdout.decode("utf-8")
    check("Zwischenablage bleibt nach Programmende erhalten", after_exit == "A = 20 cm² ⋅ µm", repr(after_exit))
    # Neu gestartetes Programm (frischer Prozess) fügt ein
    restarted = (
        "import sys, warnings; warnings.filterwarnings('ignore');"
        f"sys.path.insert(0, {os.path.dirname(os.path.abspath(__file__))!r});"
        "from main import EquationSolverApp;"
        "app = EquationSolverApp(); app.deiconify(); app.update();"
        "tb = app.equations_text._textbox; tb.focus_force(); app.update();"
        "tb.event_generate('<<Paste>>'); app.update();"
        "sys.stdout.buffer.write(app.equations_text.get('1.0', 'end-1c').encode('utf-8'));"
        "app.destroy()")
    pasted = subprocess.run([sys.executable, "-c", restarted], capture_output=True,
                            timeout=60).stdout.decode("utf-8")
    check("Nach Neustart einfügbar", pasted == "A = 20 cm² ⋅ µm", repr(pasted))

# Messdaten-Spalte (z.B. aus Excel/TXT kopiert) in eine Werteliste einfügen und lösen
set_text("T = [\n] °C\ny = 2*T")
tb.focus_force()
app.clipboard_clear()
app.clipboard_append("20\r\n25\r\n30\r\n")
tb.mark_set("insert", "1.5")
tb.event_generate("<<Paste>>")
app.update()
app.solve()
app.update()
check("Eingefügte Messdaten-Spalte als Werteliste gelöst",
      app.last_solution is not None and np.allclose(app.last_solution.get("T", []), [293.15, 298.15, 303.15]),
      get_text() + " | " + app.info_label.cget("text"))

# Zeitlimit: Meldung nennt den Abbruch, Teillösung wird angezeigt
import solver as _solver_module
_old_limit = _solver_module.SOLVE_TIME_LIMIT
_solver_module.SOLVE_TIME_LIMIT = 3.0
set_text("p = 1 bar\na = exp(b) + c^2 + 1 + enthalpy(water, T=T_1, p=p)/1e6\nb = exp(c) + a^2 + 1\n"
         "c = exp(a) + b^2 + 1\nT_1 = 300 + d\nd^2 + e^2 = -1 - a^2\ne = d + f\nf*g = 1 + a\n"
         "g = sin(f) + 1\ny = 2*k\nk = 5")
app.solve()
app.update()
_solver_module.SOLVE_TIME_LIMIT = _old_limit
check("Zeitlimit: Meldung + Teillösung in der GUI",
      "Zeitlimit (3 s)" in app.info_label.cget("text") and "y" in app.value_labels
      and "PARTIAL" in app.result_status_label.cget("text"), app.info_label.cget("text"))

if _clipboard_backup is not None:
    subprocess.run(["pbcopy"], input=_clipboard_backup)
app.set_font_size(main.FONT_SIZE_DEFAULT)
app.withdraw()
app.update()

# ---------------------------------------------------------------------------
print("\n=== #13 Manuelle Startwerte (Block {$Startwerte ... $} im Blatt) ===")
app.new_file()
solve("x^2 = 9\n{$Startwerte\nx = -3\n$}")
check("Startwert aus dem Block wirkt (x = -3)", abs(app.last_solution["x"] + 3) < 1e-6)
app.new_file()
check("New löscht manuelle Startwerte", app.manual_initial_values == {})
solve("x^2 = 16")
check("Nach New: Standard-Wurzel x = +4", abs(app.last_solution["x"] - 4) < 1e-6)
app.manual_initial_values = {"x": -3.0}
app.solve()
check("Ohne Block keine Startwerte aus dem Speicher (Text ist die Quelle)",
      abs(app.last_solution["x"] - 4) < 1e-6 and app.manual_initial_values == {})

app.manual_initial_values = {"x": -3.0}
_next_open_path[0] = write_file("C.hes", "x^2 = 16\n\n{$Startwerte\nx = -3\n$}\n".encode("utf-8"))
app.open_file()
check("Open löscht manuelle Startwerte des vorigen Blatts", app.manual_initial_values == {})
app.solve()
check("Startwert aus der geöffneten Datei wirkt (x = -4)", abs(app.last_solution["x"] + 4) < 1e-6)

# Fehler im Block: Meldung mit Zeilennummer
MESSAGES.clear()
solve("x^2 = 9\n{$Startwerte\nx = 2*y\n$}")
check("Fehler im Startwerte-Block mit Zeilennummer",
      "Zeile 3" in app.info_label.cget("text") and "Startwert" in app.info_label.cget("text"),
      app.info_label.cget("text"))

# Dialog: OK ohne Eingabe speichert KEINE grauen Auto-Werte
SHEET = "T_1 = 20 °C\np = 1 bar\nh_x = 100 kJ/kg\nh_x = enthalpy(water, T=T_x, p=p)"
solve(SHEET)
dlg = open_dialog(app.show_initial_values_dialog)
entries = value_entries(dlg)
# Auto-Startwert unbekannter Temperatur = Mittel der vorgegebenen Temperaturen (T_1 = 20 °C)
auto_expected = "293.15"
check("Dialog zeigt grauen Auto-Wert (Mittel der gegebenen Temperaturen)",
      len(entries) == 1 and entries[0].get() == auto_expected,
      str([e.get() for e in entries]))
check("Keine editierbare Einheiten-ComboBox mehr (manual_units entfernt)",
      not any(isinstance(w, ctk.CTkComboBox) for w in all_children(dlg)))
dialog_button(dlg, "OK").invoke()
app.update_idletasks()
check("OK ohne Eingabe speichert nichts", app.manual_initial_values == {} and get_text() == SHEET,
      str(app.manual_initial_values))
check("Kein manual_units-Attribut", not hasattr(app, "manual_units"))

dlg = open_dialog(app.show_initial_values_dialog)
entry = value_entries(dlg)[0]
entry.delete(0, "end")
entry.insert(0, "300")
dialog_button(dlg, "OK").invoke()
app.update_idletasks()
check("Eingetippter Wert wird gespeichert", app.manual_initial_values == {"T_x": 300.0})
check("Wert steht als Block im Blatt (SI-Wert mit Einheit)",
      get_text() == SHEET + "\n\n{$Startwerte\nT_x = 300 K\n$}\n", repr(get_text()))
app.equations_text._textbox.edit_undo()
check("Undo entfernt den Block in einem Schritt", get_text() == SHEET, repr(get_text()))
app.equations_text._textbox.edit_redo()
check("Redo stellt ihn wieder her", "T_x = 300 K" in get_text())

dlg = open_dialog(app.show_initial_values_dialog)
entry = value_entries(dlg)[0]
check("Gespeicherter Wert wird wieder angezeigt", entry.get() == "300")
entry.delete(0, "end")
entry.insert(0, "abc")
MESSAGES.clear()
before = get_text()
dialog_button(dlg, "OK").invoke()
app.update_idletasks()
check("Ungültige Eingabe: Fehler + alte Werte bleiben",
      any(m[0] == "showerror" for m in MESSAGES)
      and app.manual_initial_values == {"T_x": 300.0} and get_text() == before)
entry.delete(0, "end")
entry.insert(0, "30 °C")
dialog_button(dlg, "OK").invoke()
app.update_idletasks()
check("Wert mit Einheit: bleibt im Blatt wie eingegeben",
      abs(app.manual_initial_values["T_x"] - 303.15) < 1e-9
      and "{$Startwerte\nT_x = 30 °C\n$}" in get_text() and get_text().count("$Startwerte") == 1,
      repr(get_text()))

# Von Hand geänderter Block wird im Dialog angezeigt und beim Lösen verwendet
set_text(get_text().replace("T_x = 30 °C", "T_x = 40 °C"))
dlg = open_dialog(app.show_initial_values_dialog)
entry = value_entries(dlg)[0]
check("Von Hand geänderter Block erscheint im Dialog", entry.get() == "313.15", entry.get())
dialog_button(dlg, "OK").invoke()
app.update_idletasks()
check("Unveränderter Wert: Angabe im Blatt bleibt (40 °C)", "T_x = 40 °C" in get_text(), repr(get_text()))
app.solve()
check("Lösen mit Startwert-Block", app.last_solution is not None
      and abs(app.last_solution["T_x"] - 297.0) < 5, str(app.last_solution and app.last_solution.get("T_x")))

dlg = open_dialog(app.show_initial_values_dialog)
dialog_button(dlg, "Clear All").invoke()
dialog_button(dlg, "OK").invoke()
app.update_idletasks()
check("Alle Werte gelöscht: Block wird entfernt", get_text().rstrip() == SHEET and app.manual_initial_values == {},
      repr(get_text()))

# Speichern und Öffnen: Block bleibt erhalten
set_text("x^2 = 25")
app.solve()
dlg = open_dialog(app.show_initial_values_dialog)
entry = value_entries(dlg)[0]
entry.delete(0, "end")
entry.insert(0, "-1")
dialog_button(dlg, "OK").invoke()
app.update_idletasks()
_next_save_path[0] = os.path.join(TMP, "startwerte.hes")
app.save_file_as()
app.new_file()
_next_open_path[0] = _next_save_path[0]
app.open_file()
app.solve()
check("Startwert übersteht Speichern + Öffnen (x = -5)",
      abs(app.last_solution["x"] + 5) < 1e-6, get_text())
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

too_long = [line for line in FUNCTION_HELP_TEXT.splitlines() if len(line) > 70]
check("Hilfe: alle Zeilen <= 70 Zeichen", not too_long, str(too_long))
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


app._insert_example()
check("Beispiel-Kopf nennt SI-Einheiten", "p[Pa], h[J/kg]" in get_text())
app.solve()
check("Eingebautes Beispiel löst", "SOLUTION FOUND" in app.result_status_label.cget("text"))
check("Beispiel: Heizkurve mit value()/quantity() -> T_VL = 65 °C", shown("T_VL") == (65.0, "°C"), str(shown("T_VL")))
check("Beispiel: Spreizung in K, T_RL = 45 °C, Q = 41.9 kW",
      shown("sigma_w") == (20.0, "K") and shown("T_RL") == (45.0, "°C")
      and abs(shown("Q_dot_H")[0] - 41.9) < 1e-9, f"{shown('sigma_w')} {shown('T_RL')} {shown('Q_dot_H')}")
check("Beispiel: IF wählt turbulent (Re = 30000)", abs(app.last_solution["Nu"] - 0.023*30000**0.8*7**0.4) < 1e-9)
# Wirtschaftliche Dämmdicke analytisch: (R + s/lambda)^2 = k_E*dT*t/(a*k_ins*lambda)
s_opt = (np.sqrt(0.10*15*5000/1000/(0.08*120*0.035)) - 0.5)*0.035
check("Beispiel: Optimierung der Dämmdicke = analytisch", abs(app.last_solution["s_ins"] - s_opt) < 1e-4,
      f"{app.last_solution['s_ins']} / {s_opt}")
check("Beispiel: Energie je Fläche in kWh/m²", shown("Q_a")[1] == "kWh/m^2" and abs(shown("Q_a")[0] - 15.874) < 0.01,
      str(shown("Q_a")))
check("Beispiel: NH3 mit REFERENCE R717 IIR (h' 0 °C = 200 kJ/kg) und Carnot implizit",
      abs(shown("h_r1")[0] - 1450.274) < 0.01 and abs(shown("h_r3")[0] - 365.880) < 0.01
      and abs(app.last_solution["EER"] - 4.86227) < 1e-4 and abs(app.last_solution["EER_C"] - 263.15/45) < 1e-6,
      f"{shown('h_r1')} {shown('h_r3')} {app.last_solution.get('EER')} {app.last_solution.get('EER_C')}")
check("Beispiel: keine Einheitenwarnung, kein Hinweis",
      app.unit_warning_label.cget("text") == "" and app.hints_label.cget("text") == "",
      app.unit_warning_label.cget("text") + app.hints_label.cget("text"))

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




def close(value, ref, rtol=1e-4):
    return isinstance(value, float) and abs(value - ref) <= rtol * abs(ref)


solve("T_h_in = 90 °C\nT_c_out = 40 °C\nT_c_in = 20 °C\n"
      "dT_1 = T_h_in - T_c_out\ntheta = T_h_in - T_c_in")
check("Temperaturdifferenz dT_1 = 50 K (nicht -223.15 °C)", shown("dT_1") == (50.0, "K"),
      str(shown("dT_1")))
check("Temperaturdifferenz theta = 70 K", shown("theta") == (70.0, "K"), str(shown("theta")))

# Temperaturdifferenzen immer in K; Charakter aus der Struktur, nicht aus dem Namen
for diff, out in (("dT", "T_2"), ("a", "b")):
    solve(f"T_1 = 20 °C\n{diff} = 10 K\n{out} = T_1 + {diff}")
    check(f"Differenz in K ({diff}): {diff} = 10 K, {out} = 30 °C",
          shown(diff) == (10.0, "K") and shown(out) == (30.0, "°C"), f"{shown(diff)} {shown(out)}")
solve("T_1 = 20 °C\nx = 10 °C\nT_2 = T_1 + x")
check("Zwei Werte in °C addiert -> in Kelvin gerechnet (303.15 °C) + Hinweis",
      close(shown("T_2")[0], 303.15) and shown("T_2")[1] == "°C" and app.hints_label.cget("text") == "ⓘ HINWEISE (1)"
      and "gerechnet wird in Kelvin" in app.last_analysis.hints[0], f"{shown('T_2')} {app.hints_label.cget('text')}")
solve("Q = 41.9 kW\nm = 1 kg/s\nc = 4.19 kJ/(kg*K)\nQ = m*c*theta")
check("Temperatur im Produkt ohne Temperatur-Dimension = Differenz: theta = 10 K", shown("theta") == (10.0, "K"),
      str(shown("theta")))
# Nutzerfälle: Differenz zweier °C-Werte in K; (T1 - T2) im Produkt ist Differenz -> T2 absolut
solve("T1=10°C\nT2=20°C\nT1=T2+x")
check("T1 = T2 + x mit T1, T2 in °C: x = -10 K", shown("x") == (-10.0, "K"), str(shown("x")))
for order, expected in (("T1-T2", 75.0), ("T2-T1", 85.0)):
    solve(f"T1=80°C\nQ_dot=20kW\nm_dot=1 kg/s\nc=4 kJ/kgK\nQ_dot=m_dot*c*({order})")
    check(f"Q = m*c*({order}) mit T1 = 80 °C: T2 = {expected} °C", shown("T2") == (expected, "°C"), str(shown("T2")))
# Temperaturen immer in Kelvin - auch Summen absoluter Temperaturen (nur Hinweis), keine Skalen-Regel
for text, name, expected, hints in (("T_1=20°C\nT_2=40°C\n\n\nT_3=T_2+T_1", "T_3", 333.15, 1),
                                    ("T_1=20°C\nT_2=40°C\nT_3=T_2+T_1\nT_4=T_3+T_1", "T_4", 626.3, 1),
                                    ("T_1 = 20 °C\nT_2 = 40 °C\nT_m = (T_1 + T_2)/2", "T_m", 30.0, 0)):
    solve(text)
    v, u = shown(name)
    n_hints = len(app.last_analysis.hints) if app.last_analysis else 0
    check(f"Summe absoluter Temperaturen in Kelvin: {name} = {expected} °C, {hints} Hinweis(e)",
          close(v, expected) and u == "°C" and n_hints == hints, f"{(v, u)} {n_hints}")
solve("T_a = -5 °C\nT_VL = quantity(20 + 1.5*(20 - value(T_a, °C)), °C)")
check("Zahlenwertgleichung (Heizkurve in °C): T_VL = 57.5 °C", shown("T_VL") == (57.5, "°C"), str(shown("T_VL")))
solve("L_0 = 5 m\nn = 0.4\ny = L_0^n\nq = 2*y")
check("Einheit nicht bestimmbar -> Hinweis mit anzugebender Größe",
      app.hints_label.cget("text") == "ⓘ HINWEISE (1)" and "Einheit von q (oder von y) angeben"
      in app.last_analysis.hints[0], str(app.last_analysis.hints if app.last_analysis else None))
solve("L_0 = 5 m\nn = 0.4\ny = L_0^n\nq = 2*y\n{$Startwerte q = 2 m $}")
check("Startwert mit Einheit legt die Einheit fest (kein Hinweis)",
      shown("q")[1] == "m" and app.hints_label.cget("text") == "", str(shown("q")))
# Konstante 0 ohne Einheit: Einheit folgt aus den Gleichungen (null ist in jeder Einheit null)
solve("Q_12 = 0\nm = 2 kg\nc = 4.19 kJ/(kg*K)\nT_1 = 20 °C\nQ_12 + W_12 = m*c*(T_2 - T_1)\nW_12 = 10 kJ")
check("Konstante 0 ohne Einheit: keine Einheitenwarnung, Einheit kJ",
      app.unit_warning_label.cget("text") == "" and shown("Q_12") == (0.0, "kJ"), str(shown("Q_12")))
solve("x = 3\ny = 5 m\nz = y + x")
check("Konstante 3 ohne Einheit bleibt dimensionslos (Warnung bei y + x)", app.unit_warning_label.cget("text") != "")
# Nullpunkt-Test: Gleichung bleibt bei verschobenem Temperatur-Nullpunkt gültig
# (absolute Temperaturen verschieben sich mit, Differenzen nicht) - generisch, numerisch
for text, name, expected in (
        ("Q = 1 kJ\nm = 1 kg\nc = 0.1 kJ/(kg*K)\ndT = Q/(m*c)", "dT", (10.0, "K")),
        ("q = 50 W/m^2\nR_si = 0.13 m^2*K/W\nx = q*R_si", "x", (6.5, "K")),
        ("P = 100 W\nm = 500 kg\nc = 0.714 kJ/(kg*K)\nt = 1 h\nm*c*r = P\ny = r*t", "y", (1.0084, "K")),
        ("m_1 = 1 kg/s\nm_2 = 3 kg/s\nT_1 = 20 °C\nT_2 = 60 °C\nm_3 = m_1 + m_2\nm_3*T_3 = m_1*T_1 + m_2*T_2",
         "T_3", (50.0, "°C")),
        ("m_1 = 1 kg/s\nm_2 = 3 kg/s\nT_1 = 20 °C\nT_2 = 60 °C\nm_3 = 5 kg/s\nm_3*T_3 = m_1*T_1 + m_2*T_2",
         "T_3", (258.52, "K")),
        ("T_1 = 20 °C\nT_2 = T_1*5^0.286", "T_2", (191.3603, "°C")),
        ("sigma = 5.67e-8 W/(m^2*K^4)\nq = 500 W/m^2\nT_u = 20 °C\nq = sigma*(T_s^4 - T_u^4)", "T_s", (83.6314, "°C"))):
    solve(text)
    value, unit = shown(name)
    check(f"Nullpunkt-Test: {text.splitlines()[-1][:40]} -> {expected[0]} {expected[1]}",
          unit == expected[1] and abs(value - expected[0]) < 1e-3, f"{value} {unit}")
# Auswertungsfehler (Stoffwert außerhalb des Gültigkeitsbereichs, 0/0) erscheinen in der Meldung
for text, expected in (("p_2 = 90 bar\nh_3 = enthalpy(CO2, p=p_2, x=0)", "critical point"),
                       ("C_r = 1\nNTU = 2\neps = (1-exp(-NTU*(1-C_r)))/(1-C_r*exp(-NTU*(1-C_r)))",
                        "nicht definiert"),
                       ("T = 20 °C\np = 10 bar\nw = 0.01\nphi = HumidAir(rh, T=T, w=w, p_tot=p)",
                        "Zustand übersättigt")):
    solve(text)
    info = app.info_label.cget("text")
    check(f"GUI nennt Auswertungsfehler: {expected}", info.startswith("Auswertungsfehler:") and expected in info,
          info[:160])
solve("a = 2\nx + y = a\nx + 3*y = 4\nx + 2*y = 3")
check("GUI: überbestimmt, widerspruchsfrei -> Lösung", "SOLUTION FOUND" in app.result_status_label.cget("text")
      and shown("x")[0] == 1.0, app.info_label.cget("text"))
solve("p = 0.05 bar\nh = 2567 kJ/kg\nx = quality(water, p=p, h=h)")
check("quality() = -1 (überhitzt) -> Hinweis 'nicht im Nassdampfgebiet'",
      any("nicht im Nassdampfgebiet" in h for h in app.last_analysis.hints), str(app.last_analysis.hints))
solve("p = 1 bar\nh = 1500 kJ/kg\nx = quality(water, p=p, h=h)")
check("quality() im Nassdampfgebiet -> kein Hinweis", app.hints_label.cget("text") == "")
# Dimensionslose Ergebnisse umschaltbar (-, %, ‰, g/kg); Faktor/Summe behält die Eingabe-Einheit
solve("T = 26 °C\nphi_R = 40 %\np = 950 mbar\nx_R = HumidAir(x, T=T, phi=phi_R, p=p)\n"
      "n_P = 40\nV_dot_P = 35 m3/h\nV_dot = n_P*V_dot_P\nV_dot_2 = V_dot + V_dot_P")
app.unit_dropdowns["x_R"].set("g/kg"); app._on_unit_changed("x_R", "g/kg")
check("Wassergehalt in g/kg umschaltbar", abs(float(app.value_labels["x_R"].cget("text")) - 8.9725) < 1e-3,
      app.value_labels["x_R"].cget("text"))
check("n*V_dot_P und Summe behalten m3/h", shown("V_dot") == (1400.0, "m3/h") and shown("V_dot_2") == (1435.0, "m3/h"),
      f"{shown('V_dot')} {shown('V_dot_2')}")
solve("x_1 = 7 g/kg\nx_3 = x_1 + 0.001\neta = 80 %\neta_2 = eta*0.5\neta_3 = eta*eta")
check("Dimensionslose Einheiten bleiben (g/kg, %), Produkt zweier % ist eine Zahl",
      shown("x_3") == (8.0, "g/kg") and shown("eta_2") == (40.0, "%") and shown("eta_3")[1] == "-",
      f"{shown('x_3')} {shown('eta_2')} {shown('eta_3')}")
app._on_unit_changed("T2", "K")
solve("T1=10°C\nT2=20°C\nT1=T2+x")
app._on_unit_changed("x", "°C")
check("Differenz in °C umgestellt: ohne Offset (-10)", app.value_labels["x"].cget("text") == "-10",
      app.value_labels["x"].cget("text"))
solve("T_sun = 5800 K\nF = Blackbody_cumulative(T_sun, 4 µm)")
check("In K eingegebene absolute Temperatur (Strahlungsfunktion) nach Settings in °C",
      shown("T_sun") == (5526.85, "°C"), str(shown("T_sun")))

solve("R_si = 0.13 m^2*K/W\nd_1 = 20 cm\nlambda_1 = 2.3 W/mK\nT_i = 20 °C\nT_e = -10 °C\n"
      "U = 1/(R_si + d_1/lambda_1 + 0.04)\nq = U*(T_i - T_e)\nT_si = T_i - q*R_si")
v, u = shown("T_si")
check("Oberflächentemperatur T_si = 4.82 °C (absolut, nicht als Differenz)", close(v, 4.82164, 1e-3) and u == "°C",
      str((v, u)))
v, u = shown("U")
check("U-Wert in W/(m²K), nicht 0.0039 kW/m²K", close(v, 3.89171) and u.startswith("W/"), str((v, u)))

solve("p = 1 bar\nT_1 = 20 °C\nm_dot = 2 kg/s\nh_1 = enthalpy(water, T=T_1, p=p)\n"
      "Q_dot = m_dot*(enthalpy(water, T=80 °C, p=p) - h_1)")
v, u = shown("Q_dot")
check("Funktion im Ausdruck: Q_dot in kW (nicht kJ/kg)", close(v, 502.096) and u == "kW", str((v, u)))

solve("T = 1000 K\nlambda_max = Wien(T)\nL = 5 µm\nE_l = Eb(T, L)")
v, u = shown("lambda_max")
check("Wien: 2.898 µm (nicht 2.9e6 µm)", close(v, 2.89777) and u == "µm", str((v, u)))
v, u = shown("E_l")
check("Eb: 7139.6 W/(m²·µm)", close(v, 7139.62) and u == "W/(m^2*µm)", str((v, u)))
check("Eingabe 5 µm wird als 5 µm angezeigt", shown("L") == (5.0, "µm"), str(shown("L")))

solve("P_el = 2 MW\nUA = 500 W/K\nE = 3 MJ\nx = P_el*2")
check("2 MW bleibt '2 MW' (nicht '2000 MW')", shown("P_el") == (2.0, "MW"), str(shown("P_el")))
check("500 W/K bleibt '500 W/K' (nicht 'kW/K')", shown("UA") == (500.0, "W/K"), str(shown("UA")))
check("3 MJ bleibt '3 MJ'", shown("E") == (3.0, "MJ"), str(shown("E")))
v, u = shown("x")
check("Berechnete Leistung in kW (Settings)", close(v, 4000.0) and u == "kW", str((v, u)))

solve("T = 20:10:50 °C\nh = enthalpy(water, T=T, p=1 bar)")
check("Sweep T in °C angezeigt (nicht 293→323)",
      app.value_labels["T"].cget("text") == "[4× 20→50]" and shown("T")[1] == "°C",
      app.value_labels["T"].cget("text"))
check("Sweep h in kJ/kg angezeigt", shown("h")[1] == "kJ/kg" and "84.01" in app.value_labels["h"].cget("text"),
      app.value_labels["h"].cget("text"))
plot_T, plot_unit = app._display_values("T", app.last_solution["T"])
check("Plot-Daten in Anzeige-Einheit (°C)", np.allclose(plot_T, [20, 30, 40, 50]) and plot_unit == "°C")

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

solve("alpha=10 W/m2K\n\nlambda=0.04 W/mK\n\nA=20cm2\n\nT_1=20°C\nT_2=15°C\n\nQ_dot=A*alpha*(T_1-T_2)")
v, u = shown("Q_dot")
check("Exponent ohne ^: Q_dot = 0.1 W mit Einheit (kW-Einstellung)", close(v, 1e-4) and u == "kW", str((v, u)))
check("Eingabe 20cm2 wird als '20 cm2' angezeigt", shown("A") == (20.0, "cm2"), str(shown("A")))

ROHR = """h_e=11 W/m2K
T_e=10°C
h_i=2000 W/m2K
T_i=90 °C
lambda_1=50 W/mK
lambda_2=0.04 W/mK
r_1=0.1 m
r_2=0.104 m
q_dot=50 W/m2

r_3=r_2+s

L=1 m
U_r_1  = 1 /( r_1  *( (1/r_1/h_i) + ln(r_2/r_1)/lambda_1 + ln(r_3/r_2)/lambda_2  +  (1/r_3/h_e) ) )
q_dot=U_r_1*2*r_1*pi*L*(T_i-T_e)"""
solve(ROHR)
check("Unit-Warnung angezeigt", app.unit_warning_label.cget("text") == "⚠ UNIT WARNINGS (1)",
      app.unit_warning_label.cget("text"))
section = app.unit_warnings_content
section.grid_remove()                       # Sektion zugeklappt
app.tab_view.set("Results")
app.unit_warning_label._label.event_generate("<Button-1>")   # Klick auf das Label
app.update()
check("Klick auf UNIT WARNINGS öffnet Residuals-Tab", app.tab_view.get() == "Residuals", app.tab_view.get())
check("Klick klappt die Warnungs-Sektion auf", bool(section.winfo_manager()))
warning_texts = [w.cget("text") for w in all_children(section) if isinstance(w, ctk.CTkLabel)]
check("Warnung nennt die Variable (q_dot)", "Variable: q_dot" in warning_texts, str(warning_texts))
check("Warnung mit lesbaren Einheiten (W/m^2 vs. W)",
      any("links: W/m^2 ≠ rechts: W" in t for t in warning_texts), str(warning_texts))
check("Kein sinnloses 'Faktor 0×'", not any("Faktor" in t for t in warning_texts), str(warning_texts))
app.tab_view.set("Results")

solve(ROHR.replace(" W/m2K", "").replace(" W/mK", "").replace("°C", "").replace(" °C", "")
      .replace(" W/m2", "").replace(" m\n", "\n").replace("r_3=r_2+s", "r_3=r_2+s\ns=0.1"))
info = app.info_label.cget("text")
check("Widerspruch nennt vorgegebenen und berechneten Wert",
      "q_dot ist vorgegeben (50)" in info and "29.0642" in info and "überbestimmt" in info, info)

solve(ROHR.replace("r_1  *( (1/r_1/h_i)", "r_1  ( (1/r_1/h_i)"))
info = app.info_label.cget("text")
check("Fehlendes '*': Zeile + Fundstelle statt 'Unvollständig'",
      "Zeile 14:" in info and "'r_1' ist keine Funktion" in info and "r_1  ▶(" in info, info)

ROHR_TYPO = ROHR.replace("q_dot=50 W/m2", "q_dot=50 W").replace("(1/r_1/h_i)", "(1/r_1h_i)")
solve(ROHR_TYPO)
info = app.info_label.cget("text")
check("Tippfehler r_1h_i: Strukturdiagnose statt Zählung",
      "Unterbestimmt" in info and "r_1h_i, r_3, s" in info and "Zeilen 11, 14" in info, info)
check("Tippfehler r_1h_i: Namens-Hinweis in der Meldung", "r_1 und h_i" in info, info)

solve(ROHR_TYPO.replace("r_3=r_2+s", "r_3=r_2+s\ns=0.1"))
check("Formal lösbar mit Tippfehler: Lösung + Hinweis-Label",
      "SOLUTION FOUND" in app.result_status_label.cget("text")
      and app.hints_label.cget("text") == "ⓘ HINWEISE (1)", app.hints_label.cget("text"))
app.deiconify()
app.update()
app.tab_view.set("Results")
app.update()
app.hints_label._label.event_generate("<Button-1>")
app.update()
check("Klick auf HINWEISE öffnet Residuals-Tab", app.tab_view.get() == "Residuals", app.tab_view.get())
hint_texts = [w.cget("text") for w in all_children(app.hints_content) if isinstance(w, ctk.CTkLabel)]
check("Hinweis-Sektion nennt r_1h_i", any("r_1h_i" in t for t in hint_texts), str(hint_texts))
app.tab_view.set("Results")
app.withdraw()
app.update()

solve("a = 2\nexp(z) = -a\nb = a*3")
check("Numerisch unlösbar: verständliche Meldung statt 'Unvollständig'",
      "Keine numerische Lösung für z (Zeile 2)" in app.info_label.cget("text"), app.info_label.cget("text"))

print("\n=== Settings, Plot-Dialoge, Fluidliste ===")
solve("T = 20 °C\ndT = 10 K\nT_2 = T + dT\np = 2 bar")
check("Anzeige vorher: T_2 = 30 °C", shown("T_2") == (30.0, "°C"), str(shown("T_2")))
app.temp_display_unit.set("K")
app.pressure_display_unit.set("Pa")
app.update()
check("Settings-Wechsel aktualisiert sofort: T_2 = 303.15 K", shown("T_2") == (303.15, "K"), str(shown("T_2")))
check("Settings-Wechsel aktualisiert sofort: p = 200000 Pa", shown("p") == (200000.0, "Pa"), str(shown("p")))
app.temp_display_unit.set("degC")
app.pressure_display_unit.set("bar")
app.update()
check("Zurück auf °C", shown("T_2") == (30.0, "°C"), str(shown("T_2")))

FIGS = []
class _RecordingFigure(main.Figure):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        FIGS.append(self)
main.Figure = _RecordingFigure
solve("T = 20:20:100 °C\np_s = pressure(water, T=T, x=0)\nh = enthalpy(water, T=T, x=0)")
app.deiconify()
app.update()
dialog = open_dialog(app.show_plot_dialog)
combos = [w for w in all_children(dialog) if isinstance(w, ctk.CTkComboBox)]
boxes = {w.cget("text"): w for w in all_children(dialog) if isinstance(w, ctk.CTkCheckBox)}
combos[0].set("T")
for name, box in boxes.items():
    if name in ("p_s", "h") and not box.get():
        box.toggle()
    elif name not in ("p_s", "h", "Grid", "Legend") and box.get():
        box.toggle()
dialog_button(dialog, "Plot").invoke()
app.update()
ax = FIGS[-1].axes[0] if FIGS else None
check("New Plot Window: zwei Kurven (p_s, h) über T in °C",
      ax is not None and len(ax.lines) == 2 and np.allclose(ax.lines[0].get_xdata(), [20, 40, 60, 80, 100])
      and ax.get_xlabel() == "T [°C]", str(ax and [l.get_label() for l in ax.lines]))
check("New Plot Window: Legende mit Einheiten",
      ax is not None and ax.get_legend() is not None
      and {t.get_text() for t in ax.get_legend().get_texts()} == {"h [kJ/kg]", "p_s [bar]"},
      str(ax and ax.get_legend() and [t.get_text() for t in ax.get_legend().get_texts()]))
dialog = open_dialog(app.show_quick_plot_dialog)
combos = [w for w in all_children(dialog) if isinstance(w, ctk.CTkComboBox)]
combos[0].set("T")
combos[1].set("p_s")
dialog_button(dialog, "Plot").invoke()
app.update()
ax = FIGS[-1].axes[0]
check("Quick Plot: eine Kurve p_s [bar] über T [°C]",
      len(ax.lines) == 1 and ax.get_xlabel() == "T [°C]" and ax.get_ylabel() == "p_s [bar]",
      f"{ax.get_xlabel()} / {ax.get_ylabel()}")
for window in app.winfo_children():
    if isinstance(window, ctk.CTkToplevel):
        window.destroy()
app.withdraw()
app.update()

fluids_text = main.fluid_help_text()
import CoolProp.CoolProp as _CP
all_fluids = _CP.get_global_param_string("fluids_list").split(",")
check("Fluidliste enthält alle CoolProp-Fluide", all(f in fluids_text for f in all_fluids),
      str([f for f in all_fluids if f not in fluids_text][:5]))
check("Fluidliste enthält alle Kurznamen", all(a in fluids_text for a in __import__("thermodynamics").FLUID_ALIASES))

solve("e_1 = 0.94:-0.35:0.24\nT = 300 + 10*e_1")
check("Sweep-Anzeige: erster -> letzter Punkt (fallend)",
      app.value_labels["e_1"].cget("text") == "[3× 0.94→0.24]", app.value_labels["e_1"].cget("text"))
solve("C_r = 0.5:0.5:1.5\neps = (1-exp(-2*(1-C_r)))/(1-C_r*exp(-2*(1-C_r)))")
check("Sweep mit gescheitertem Punkt: Status PARTIAL + Grund",
      "PARTIAL" in app.result_status_label.cget("text") and "Punkt 2" in app.info_label.cget("text"),
      app.result_status_label.cget("text") + " | " + app.info_label.cget("text"))

solve("A = 20 qcm\nQ = A*2")
check("Unbekannte Einheit -> Fehlermeldung statt falscher Wert",
      "Unbekannte Einheit 'qcm'" in app.info_label.cget("text") and app.last_solution is None,
      app.info_label.cget("text"))

solve("T_1 = 20 °C\np = 1 bar\nh_1 = enthalpy(water, T=T_1, p=p)\nexp(z) = -1")
v, u = shown("h_1")
check("Teillösung: Werte mit Einheit (h_1 in kJ/kg)", close(v, 84.0061) and u == "kJ/kg", str((v, u)))

# ---------------------------------------------------------------------------
print("\n=== Optimierung (MINIMIZE/MAXIMIZE ... VARY ...) ===")
solve("y = (s - 0.0207)^2*1e4 + 1\nMINIMIZE y VARY s = 11 .. 30 mm")
v, u = shown("s")
check("Optimum in der Einheit der Grenzen (s in mm)", close(v, 20.7) and u == "mm", str((v, u)))
check("Status und Meldung der Optimierung",
      "SOLUTION FOUND" in app.result_status_label.cget("text")
      and "Minimum von y" in app.info_label.cget("text") and app.status_label.cget("text") == "Optimum found",
      app.info_label.cget("text"))
check("Residuals-Analyse beim Optimum vorhanden", app.last_analysis is not None)
check("Variierte Größe im Initial-Values-Dialog wählbar", "s" in app.known_variables)

solve("y = (s - 0.0207)^2\nMAXIMIZE y VARY s = 11 .. 30 mm")
check("Maximum am Rand wird gemeldet", "s an der Untergrenze" in app.info_label.cget("text"),
      app.info_label.cget("text"))

solve("a = 1\ns = 5 mm\ny = s^2 + a\nMINIMIZE y VARY s = 1 .. 9 mm")
check("Variierte Größe mit festem Wert -> Meldung mit Zeile",
      "Zeile 4: s hat im Blatt einen festen Wert" in app.info_label.cget("text"), app.info_label.cget("text"))
solve("y = a^2\na = 2\nMINIMIZE y VARY b = 1 .. 2")
check("Variierte Größe in keiner Gleichung -> Meldung",
      "b kommt in keiner Gleichung vor" in app.info_label.cget("text"), app.info_label.cget("text"))
solve("y = x^2\nMINIMIZE y VARY x 1 .. 2")
check("Syntaxfehler der Anweisung -> Meldung mit Zeile",
      "Zeile 2:" in app.info_label.cget("text") and "Bereich als x = a .. b" in app.info_label.cget("text"),
      app.info_label.cget("text"))
solve("y = 5 + 0*x\nMINIMIZE y VARY x = 0 .. 1")
check("Zielgröße unabhängig -> Teillösung mit Meldung",
      "PARTIAL" in app.result_status_label.cget("text") and "ändert sich nicht" in app.info_label.cget("text"),
      app.info_label.cget("text"))

solve("y = (x^2 - 4)^2 + x\nMINIMIZE y VARY x = -3 .. 3\n{$Startwerte\nx = 2\n$}")
check("Startwert im Block beim schlechteren Minimum: trotzdem globales Optimum",
      app.last_solution is not None and abs(app.last_solution["x"] + 2.0305) < 1e-3)

solve("a = [1 2 3] kg/s\nf = (m - a)^2\nMINIMIZE f VARY m = 0 .. 5 kg/s")
check("Optimum je Punkt einer Werteliste",
      isinstance(app.last_solution.get("m"), np.ndarray) and np.allclose(app.last_solution["m"], [1, 2, 3], atol=1e-4)
      and "Parametric Study: 3 points" in app.info_label.cget("text"), str(app.last_solution.get("m")))

# Bezugszustand im Blatt (REFERENCE R717 IIR) gilt nur für diesen Lauf
solve("REFERENCE R717 IIR\nh_1 = enthalpy(ammonia, T=0 °C, x=0)")
check("REFERENCE R717 IIR -> h' = 200 kJ/kg bei 0 °C", shown("h_1") == (200.0, "kJ/kg"), str(shown("h_1")))
solve("h_1 = enthalpy(ammonia, T=0 °C, x=0)")
check("Blatt ohne REFERENCE -> wieder CoolProp-Standard (345.7 kJ/kg)",
      shown("h_1")[0] is not None and abs(shown("h_1")[0] - 345.675) < 0.01, str(shown("h_1")))
solve("T_0 = -40 °C\nT_c = 25 °C\nEER_C*(T_c - T_0) = T_0")
check("Carnot implizit: EER_C*(T_c - T_0) = T_0 in Kelvin (3.587), kein Hinweis",
      abs(shown("EER_C")[0] - 3.58692) < 1e-4 and not app.last_analysis.hints, f"{shown('EER_C')}")
# Wellenlänge als Zahl ohne Einheit: SI (m) wie jede Zahl, keine µm-Deutung - Hinweis
solve("E_1 = Eb(573.15, 5)\nE_2 = Eb(573.15, 5 µm)")
check("Eb(573.15, 5) = 5 m (keine µm-Deutung) + Hinweis",
      app.last_solution["E_1"] < 1e-12 * app.last_solution["E_2"] and len(app.last_analysis.hints) == 1
      and "gilt als 5 m" in app.last_analysis.hints[0], str(app.last_analysis.hints))

# Anzeige: Startwert mit Einheit legt die Anzeige fest (q_V in kJ/m^3 statt bar), Energie je Länge,
# Rundungsrest nach Offset-Umrechnung, Zahl in Summe mit h-Eingabe -> Hinweis
solve("h_1 = 388 kJ/kg\nh_6 = 242 kJ/kg\nv_1 = 0.2367 m^3/kg\nq_V = (h_1 - h_6)/v_1\n"
      "{$Startwerte\nq_V = 600 kJ/m^3\n$}")
check("Startwert mit Einheit -> Anzeige q_V in kJ/m³", shown("q_V")[1] == "kJ/m^3"
      and abs(shown("q_V")[0] - 616.815) < 0.01, str(shown("q_V")))
solve("q_ES = 60 W/m\nt_B = 1800 h\nw_ES = q_ES*t_B")
check("Leistung je Länge mal Zeit -> 108 kWh/m (nicht N)", shown("w_ES") == (108.0, "kWh/m"), str(shown("w_ES")))
solve("T_0 = 0 °C\np_0 = pressure(ammonia, T=T_0, x=1)\nh_1 = enthalpy(ammonia, T=T_0, x=0.5)\n"
      "T_4 = temperature(ammonia, p=p_0, h=h_1)")
check("Kein Rundungsrest nach Offset-Umrechnung (T_4 = 0 °C)", shown("T_4") == (0.0, "°C"), str(shown("T_4")))
solve("Q_HL = 10 kW\nt_S = 2 h\nf = 24/(24 - t_S)\nQ_WP = Q_HL*f")
check("Zahl in Summe mit Zeit in h -> Hinweis 'gilt als 24 s'",
      any("gilt als 24 s" in h for h in app.last_analysis.hints), str(app.last_analysis.hints))

# ---------------------------------------------------------------------------
app.destroy()
faulthandler.cancel_dump_traceback_later()

print()
print(f"{len(PASSED)}/{len(PASSED) + len(FAILED)} Tests bestanden")
if FAILED:
    print("FEHLGESCHLAGEN:", *FAILED, sep="\n  - ")
    sys.exit(1)
print("ALLE TESTS OK")
