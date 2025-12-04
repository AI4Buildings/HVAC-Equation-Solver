#!/usr/bin/env python3
"""
HVAC Equation Solver - Ein EES-ähnlicher Gleichungslöser

Hauptanwendung mit grafischer Benutzeroberfläche.
"""

import os
import sys

# Unterdrücke macOS-spezifische Warnungen (NSOpenPanel/NSWindow)
if sys.platform == 'darwin':
    os.environ['TK_SILENCE_DEPRECATION'] = '1'
    # Unterdrücke Cocoa-Warnungen
    from ctypes import cdll, c_int
    try:
        libc = cdll.LoadLibrary('libc.dylib')
        # Redirect stderr to /dev/null für Cocoa-Warnungen
        devnull = os.open(os.devnull, os.O_WRONLY)
        saved_stderr = os.dup(2)
        os.dup2(devnull, 2)
        os.close(devnull)
    except Exception:
        saved_stderr = None

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, font, filedialog

# Stelle stderr wieder her nach tkinter-Import
if sys.platform == 'darwin' and 'saved_stderr' in dir() and saved_stderr is not None:
    os.dup2(saved_stderr, 2)
    os.close(saved_stderr)


def _suppress_macos_warning(func):
    """Wrapper um macOS Cocoa-Warnungen bei Dateidialogen zu unterdrücken."""
    if sys.platform != 'darwin':
        return func()

    # Unterdrücke stderr während des Dialogs
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        return func()
    finally:
        os.dup2(saved, 2)
        os.close(saved)


from parser import parse_equations, validate_system
from solver import solve_system, solve_parametric, format_solution
import numpy as np

# Versuche Thermodynamik-Modul zu laden
try:
    from thermodynamics import get_fluid_info, THERMO_FUNCTIONS
    THERMO_AVAILABLE = True
except ImportError:
    THERMO_AVAILABLE = False
    THERMO_FUNCTIONS = {}

# Versuche matplotlib zu laden
try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class EquationSolverApp:
    """Hauptanwendung für den Gleichungslöser."""

    def __init__(self, root):
        self.root = root
        self.root.title("HVAC Equation Solver")
        self.root.geometry("900x700")
        self.root.minsize(600, 400)

        # Aktueller Dateipfad
        self.current_file = None

        # Schriftgröße (Standard: 16)
        self.font_size = 16

        # Gespeicherte Variablen und manuelle Startwerte
        self.known_variables = set()
        self.manual_initial_values = {}

        # Letzte Lösung (für Plots)
        self.last_solution = None
        self.last_sweep_vars = {}

        # Konfiguriere das Hauptfenster
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Erstelle Hauptframe
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(1, weight=1)

        # Erstelle die GUI-Elemente
        self._create_menu()
        self._create_toolbar()
        self._create_paned_window()
        self._create_status_bar()

    def _create_menu(self):
        """Erstellt die Menüleiste."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File Menü
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)

        file_menu.add_command(label="New", command=self.new_file, accelerator="Ctrl+N")
        file_menu.add_command(label="Open...", command=self.open_file, accelerator="Ctrl+O")
        file_menu.add_command(label="Save", command=self.save_file, accelerator="Ctrl+S")
        file_menu.add_command(label="Save As...", command=self.save_file_as, accelerator="Ctrl+Shift+S")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit, accelerator="Alt+F4")

        # Edit Menü
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)

        edit_menu.add_command(label="Undo", command=lambda: self.equations_text.edit_undo(), accelerator="Ctrl+Z")
        edit_menu.add_command(label="Redo", command=lambda: self.equations_text.edit_redo(), accelerator="Ctrl+Y")
        edit_menu.add_separator()
        edit_menu.add_command(label="Clear All", command=self.clear_all)
        edit_menu.add_command(label="Insert Example", command=self._insert_example)

        # View Menü (Schriftgröße)
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)

        view_menu.add_command(label="Increase Font Size", command=self.increase_font_size, accelerator="Ctrl++")
        view_menu.add_command(label="Decrease Font Size", command=self.decrease_font_size, accelerator="Ctrl+-")
        view_menu.add_command(label="Reset Font Size", command=self.reset_font_size)
        view_menu.add_separator()

        # Schriftgrößen-Untermenü
        fontsize_menu = tk.Menu(view_menu, tearoff=0)
        view_menu.add_cascade(label="Font Size", menu=fontsize_menu)
        for size in [8, 10, 12, 14, 16, 18, 20, 24]:
            fontsize_menu.add_command(label=f"{size} pt", command=lambda s=size: self.set_font_size(s))

        # Solve Menü
        solve_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Solve", menu=solve_menu)

        solve_menu.add_command(label="Solve", command=self.solve, accelerator="F5")
        solve_menu.add_separator()
        solve_menu.add_command(label="Initial Values...", command=self.show_initial_values_dialog)

        # Plot Menü (nur wenn matplotlib verfügbar)
        if MATPLOTLIB_AVAILABLE:
            plot_menu = tk.Menu(menubar, tearoff=0)
            menubar.add_cascade(label="Plot", menu=plot_menu)
            plot_menu.add_command(label="New Plot Window...", command=self.show_plot_dialog)
            plot_menu.add_command(label="Quick Plot X-Y...", command=self.show_quick_plot_dialog)

        # Help Menü
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)

        help_menu.add_command(label="Function Reference", command=self.show_function_help)
        if THERMO_AVAILABLE:
            help_menu.add_command(label="Fluid List (CoolProp)", command=self.show_fluid_help)
            help_menu.add_separator()
            help_menu.add_command(label="Thermodynamic Example", command=self._insert_thermo_example)

        # Keyboard shortcuts
        self.root.bind("<Control-n>", lambda e: self.new_file())
        self.root.bind("<Control-o>", lambda e: self.open_file())
        self.root.bind("<Control-s>", lambda e: self.save_file())
        self.root.bind("<Control-S>", lambda e: self.save_file_as())
        self.root.bind("<Control-plus>", lambda e: self.increase_font_size())
        self.root.bind("<Control-minus>", lambda e: self.decrease_font_size())
        self.root.bind("<Control-equal>", lambda e: self.increase_font_size())  # Für Tastaturen ohne separates +

    def _create_toolbar(self):
        """Erstellt die Toolbar mit Buttons."""
        toolbar = ttk.Frame(self.main_frame)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        # Solve Button
        self.solve_btn = ttk.Button(
            toolbar,
            text="Solve (F5)",
            command=self.solve,
            width=15
        )
        self.solve_btn.pack(side=tk.LEFT, padx=(0, 5))

        # Clear Button
        self.clear_btn = ttk.Button(
            toolbar,
            text="Clear",
            command=self.clear_all,
            width=10
        )
        self.clear_btn.pack(side=tk.LEFT, padx=(0, 5))

        # Example Button
        self.example_btn = ttk.Button(
            toolbar,
            text="Example",
            command=self._insert_example,
            width=10
        )
        self.example_btn.pack(side=tk.LEFT)

        # Info Label
        info_label = ttk.Label(
            toolbar,
            text="Syntax: x + y = 10 | Kommentare: \"...\" oder {...}",
            foreground="gray"
        )
        info_label.pack(side=tk.RIGHT)

    def _create_paned_window(self):
        """Erstellt das geteilte Fenster mit Equations und Solution."""
        # PanedWindow für horizontale Teilung
        self.paned = ttk.PanedWindow(self.main_frame, orient=tk.HORIZONTAL)
        self.paned.grid(row=1, column=0, sticky="nsew")

        # Linker Frame: Equations Window
        left_frame = ttk.LabelFrame(self.paned, text="Equations", padding="5")
        self.paned.add(left_frame, weight=2)

        # Equations Text Widget
        self.eq_font = font.Font(family="Courier", size=self.font_size)
        self.equations_text = scrolledtext.ScrolledText(
            left_frame,
            font=self.eq_font,
            wrap=tk.NONE,
            undo=True
        )
        self.equations_text.pack(fill=tk.BOTH, expand=True)

        # Horizontale Scrollbar für Equations
        h_scroll = ttk.Scrollbar(left_frame, orient=tk.HORIZONTAL,
                                  command=self.equations_text.xview)
        h_scroll.pack(fill=tk.X)
        self.equations_text.configure(xscrollcommand=h_scroll.set)

        # Rechter Frame: Solution Window
        right_frame = ttk.LabelFrame(self.paned, text="Solution", padding="5")
        self.paned.add(right_frame, weight=1)

        # Solution Text Widget
        self.solution_text = scrolledtext.ScrolledText(
            right_frame,
            font=self.eq_font,
            wrap=tk.NONE,
            state=tk.DISABLED
        )
        self.solution_text.pack(fill=tk.BOTH, expand=True)

        # Horizontale Scrollbar für Solution
        h_scroll2 = ttk.Scrollbar(right_frame, orient=tk.HORIZONTAL,
                                   command=self.solution_text.xview)
        h_scroll2.pack(fill=tk.X)
        self.solution_text.configure(xscrollcommand=h_scroll2.set)

        # Tags für farbige Ausgabe
        self.solution_text.tag_configure("success", foreground="green")
        self.solution_text.tag_configure("error", foreground="red")
        self.solution_text.tag_configure("info", foreground="blue")

        # Keyboard shortcuts
        self.root.bind("<F5>", lambda e: self.solve())
        self.equations_text.bind("<Control-Return>", lambda e: self.solve())

    def _create_status_bar(self):
        """Erstellt die Statusleiste."""
        self.status_var = tk.StringVar(value="Bereit")
        self.status_bar = ttk.Label(
            self.main_frame,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_bar.grid(row=2, column=0, sticky="ew", pady=(10, 0))

    def set_font_size(self, size: int):
        """Setzt die Schriftgröße."""
        self.font_size = max(6, min(36, size))  # Begrenzen auf 6-36
        self.eq_font.configure(size=self.font_size)
        self.status_var.set(f"Schriftgröße: {self.font_size} pt")

    def increase_font_size(self):
        """Erhöht die Schriftgröße."""
        self.set_font_size(self.font_size + 2)

    def decrease_font_size(self):
        """Verringert die Schriftgröße."""
        self.set_font_size(self.font_size - 2)

    def reset_font_size(self):
        """Setzt die Schriftgröße auf Standard zurück."""
        self.set_font_size(12)

    def _insert_example(self):
        """Fügt ein Beispiel mit Thermodynamik ein."""
        if THERMO_AVAILABLE:
            example = '''"HVAC Equation Solver - Beispiel"
"Einheiten: T[C], p[bar], h[kJ/kg], s[kJ/(kg K)], rho[kg/m3]"

{--- Beispiel 1: Wasser Stoffdaten ---}
"Sattdampf bei 1 bar"
p_sat = 1
x_dampf = 1
h_dampf = enthalpy(water, p=p_sat, x=x_dampf)
T_sat = temperature(water, p=p_sat, x=x_dampf)
rho_dampf = density(water, p=p_sat, x=x_dampf)

"Siedendes Wasser bei 1 bar"
x_wasser = 0
h_wasser = enthalpy(water, p=p_sat, x=x_wasser)
rho_wasser = density(water, p=p_sat, x=x_wasser)

"Verdampfungsenthalpie"
delta_h_v = h_dampf - h_wasser

{--- Beispiel 2: R134a Kaeltemittel ---}
"Sattdampf bei 25 C"
T_R134a = 25
h_R134a = enthalpy(R134a, T=T_R134a, x=1)
p_R134a = pressure(R134a, T=T_R134a, x=1)
rho_R134a = density(R134a, T=T_R134a, x=1)

"Zum Loesen: F5 druecken"
'''
        else:
            example = '''"Beispiel: Nichtlineares Gleichungssystem"
"Berechnung eines rechtwinkligen Dreiecks"

{Gegebene Werte}
a = 3
b = 4

{Pythagorean theorem}
c^2 = a^2 + b^2

{Winkel berechnen}
tan(alpha) = a / b
alpha + beta = 90

"Zum Lösen: F5 drücken oder Solve-Button klicken"
'''
        self.equations_text.delete("1.0", tk.END)
        self.equations_text.insert("1.0", example)
        self.clear_solution()

    def _insert_thermo_example(self):
        """Fügt ein Thermodynamik-Beispiel ein."""
        example = '''"Beispiel: Thermodynamische Stoffdaten"
"Dampfkraftprozess - einfacher Rankine-Zyklus"

{Einheiten: T in C, p in bar, h in kJ/kg, s in kJ/(kg K)}

{Zustand 1: Kesseleintritt (Speisewasser)}
p1 = 100
T1 = 30
h1 = enthalpy(water, T=T1, p=p1)
s1 = entropy(water, T=T1, p=p1)
rho1 = density(water, T=T1, p=p1)

{Zustand 2: Kesselaustritt (Sattdampf)}
p2 = p1
x2 = 1
h2 = enthalpy(water, p=p2, x=x2)
s2 = entropy(water, p=p2, x=x2)
T2 = temperature(water, p=p2, x=x2)

{Waermeaufnahme im Kessel}
q_zu = h2 - h1

"Weitere Funktionen: density, volume, intenergy, quality"
"Transporteigenschaften: viscosity, conductivity, prandtl, cp, cv"
'''
        self.equations_text.delete("1.0", tk.END)
        self.equations_text.insert("1.0", example)
        self.clear_solution()

    def show_function_help(self):
        """Zeigt Hilfe zu verfügbaren Funktionen."""
        help_window = tk.Toplevel(self.root)
        help_window.title("Function Reference")
        help_window.geometry("650x700")

        text = scrolledtext.ScrolledText(help_window, font=("Courier", 10))
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        help_text = """=== HVAC EQUATION SOLVER - FUNCTION REFERENCE ===

MATHEMATISCHE FUNKTIONEN:
-------------------------
sin(x), cos(x), tan(x)     Trigonometrische Funktionen (x in Grad)
asin(x), acos(x), atan(x)  Inverse trig. Funktionen
sinh(x), cosh(x), tanh(x)  Hyperbolische Funktionen
exp(x)                      e^x
ln(x)                       Natürlicher Logarithmus
log10(x)                    Logarithmus zur Basis 10
sqrt(x)                     Quadratwurzel
abs(x)                      Absolutwert
pi                          Kreiszahl (3.14159...)

OPERATOREN:
-----------
+, -, *, /                  Grundrechenarten
^                           Potenz (z.B. x^2)
( )                         Klammerung

KOMMENTARE:
-----------
"Text"                      Kommentar in Anführungszeichen
{Text}                      Kommentar in geschweiften Klammern

"""
        if THERMO_AVAILABLE:
            help_text += """
THERMODYNAMISCHE FUNKTIONEN (CoolProp):
---------------------------------------
Syntax: funktion(stoff, param1=wert1, param2=wert2)

Eigenschaften:
  enthalpy(...)      Spez. Enthalpie [kJ/kg]
  entropy(...)       Spez. Entropie [kJ/(kg K)]
  density(...)       Dichte [kg/m3]
  volume(...)        Spez. Volumen [m3/kg]
  intenergy(...)     Spez. innere Energie [kJ/kg]
  quality(...)       Dampfgehalt [-] (0=Fluessigkeit, 1=Dampf)
  temperature(...)   Temperatur [C]
  pressure(...)      Druck [bar]
  cp(...)            Waermekapazitaet cp [kJ/(kg K)]
  cv(...)            Waermekapazitaet cv [kJ/(kg K)]

Transporteigenschaften:
  viscosity(...)     Dynamische Viskositaet [Pa s]
  conductivity(...)  Waermeleitfaehigkeit [W/(m K)]
  prandtl(...)       Prandtl-Zahl [-]
  soundspeed(...)    Schallgeschwindigkeit [m/s]

Zustandsgroessen (2 erforderlich):
  T = Temperatur [C]
  p = Druck [bar]
  h = Enthalpie [kJ/kg]
  s = Entropie [kJ/(kg K)]
  x = Dampfgehalt [-]
  rho, d = Dichte [kg/m3]
  u = Innere Energie [kJ/kg]

Beispiele:
  h = enthalpy(water, T=100, p=1)
  rho = density(R134a, T=25, x=1)
  T = temperature(water, p=10, h=2700)

HUMID AIR FUNCTIONS (CoolProp HumidAirProp):
--------------------------------------------
Syntax: HumidAir(property, T=..., rh=..., p_tot=...)

Output Properties (first argument):
  T        Dry bulb temperature [C]
  h        Specific enthalpy [kJ/kg_dry_air]
  rh       Relative humidity [-] (0-1)
  w        Humidity ratio [kg_water/kg_dry_air]
  p_w      Partial pressure of water vapor [bar]
  rho_tot  Density of humid air [kg/m3]
  rho_a    Density of dry air [kg/m3]
  rho_w    Density of water vapor [kg/m3]
  T_dp     Dew point temperature [C]
  T_wb     Wet bulb temperature [C]

Input Parameters (exactly 3 required):
  T        Temperature [C]
  p_tot    Total pressure [bar]
  rh       Relative humidity [-] (0-1)
  w        Humidity ratio [kg_water/kg_dry_air]
  p_w      Partial pressure water vapor [bar]
  h        Enthalpy [kJ/kg_dry_air]

Examples:
  h = HumidAir(h, T=25, rh=0.5, p_tot=1)
  w = HumidAir(w, T=30, rh=0.6, p_tot=1)
  T = HumidAir(T, h=50, rh=0.5, p_tot=1)
  T_dp = HumidAir(T_dp, T=25, w=0.01, p_tot=1)
  T_wb = HumidAir(T_wb, T=30, rh=0.5, p_tot=1)
  rho = HumidAir(rho_tot, T=25, rh=0.5, p_tot=1)
  rh = HumidAir(rh, T=25, w=0.01, p_tot=1)
"""

        help_text += """
SCHWARZKOERPER-STRAHLUNG (Planck):
----------------------------------
Eb(T, lambda)              Spektrale Emissionsleistung [W/(m2 um)]
                           T: Temperatur [C], lambda: Wellenlaenge [um]

Blackbody(T, l1, l2)       Anteil der Strahlung im Bereich [l1, l2]
                           Dimensionslos (0-1)

Wien(T)                    Wellenlaenge max. Emission [um]
                           (Wiensches Verschiebungsgesetz)

Stefan_Boltzmann(T)        Gesamte Emissionsleistung [W/m2]
                           (Stefan-Boltzmann-Gesetz: E = sigma*T^4)

Beispiele:
  E_b = Eb(1000, 2.5)                   "bei 1000 C und 2.5 um"
  f = Blackbody(5500, 0.4, 0.7)         "sichtbarer Anteil Sonne"
  lambda_max = Wien(1000)               "Wellenlaenge bei max. Emission"
  E_total = Stefan_Boltzmann(500)       "Gesamtemission bei 500 C"
"""

        text.insert("1.0", help_text)
        text.configure(state=tk.DISABLED)

    def show_fluid_help(self):
        """Zeigt Liste der verfügbaren Fluide."""
        if not THERMO_AVAILABLE:
            messagebox.showinfo("Info", "CoolProp nicht verfügbar")
            return

        help_window = tk.Toplevel(self.root)
        help_window.title("Available Fluids (CoolProp)")
        help_window.geometry("500x600")

        text = scrolledtext.ScrolledText(help_window, font=("Courier", 10))
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        help_text = """=== VERFUEGBARE FLUIDE ===

WASSER / DAMPF:
  Water, water, steam, h2o, wasser

LUFT:
  Air, air, luft

KAELTEMITTEL (HFCs):
  R134a, R32, R410A, R407C, R404A, R507A

KAELTEMITTEL (HFOs):
  R1234yf, R1234ze(E)

NATUERLICHE KAELTEMITTEL:
  R717 / Ammonia (ammonia, nh3)
  R744 / CO2 (co2)
  R290 / Propane (propane, propan)
  R600a / IsoButane (isobutane, isobutan)

GASE:
  Nitrogen (n2, stickstoff)
  Oxygen (o2, sauerstoff)
  Hydrogen (h2, wasserstoff)
  Helium (he)
  Argon (ar)
  Methane (ch4, methan)
  Ethane (c2h6, ethan)

HINWEIS:
  Die Kurzbezeichnungen (in Klammern) koennen
  alternativ verwendet werden.
  Gross-/Kleinschreibung wird ignoriert.

BEISPIELE:
  h = enthalpy(water, T=100, p=1)
  h = enthalpy(Water, T=100, p=1)
  h = enthalpy(R134a, T=25, x=1)
  h = enthalpy(ammonia, T=0, x=0)
"""

        text.insert("1.0", help_text)
        text.configure(state=tk.DISABLED)

    def clear_all(self):
        """Löscht alle Eingaben und Ausgaben."""
        self.equations_text.delete("1.0", tk.END)
        self.clear_solution()
        self.status_var.set("Bereit")

    def new_file(self):
        """Erstellt eine neue leere Datei."""
        self.equations_text.delete("1.0", tk.END)
        self.clear_solution()
        self.current_file = None
        self._update_title()
        self.status_var.set("Neue Datei")

    def open_file(self):
        """Öffnet eine Datei."""
        filetypes = [
            ("HES Files", "*.hes"),
            ("Text Files", "*.txt"),
            ("All Files", "*.*")
        ]
        filepath = _suppress_macos_warning(lambda: filedialog.askopenfilename(
            title="Datei öffnen",
            filetypes=filetypes,
            defaultextension=".hes"
        ))

        if filepath:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()

                self.equations_text.delete("1.0", tk.END)
                self.equations_text.insert("1.0", content)
                self.clear_solution()
                self.current_file = filepath
                self._update_title()
                self.status_var.set(f"Geöffnet: {filepath}")
            except Exception as e:
                messagebox.showerror("Fehler", f"Datei konnte nicht geöffnet werden:\n{e}")

    def save_file(self):
        """Speichert die aktuelle Datei."""
        if self.current_file:
            self._save_to_file(self.current_file)
        else:
            self.save_file_as()

    def save_file_as(self):
        """Speichert die Datei unter neuem Namen."""
        filetypes = [
            ("HES Files", "*.hes"),
            ("Text Files", "*.txt"),
            ("All Files", "*.*")
        ]
        filepath = _suppress_macos_warning(lambda: filedialog.asksaveasfilename(
            title="Datei speichern",
            filetypes=filetypes,
            defaultextension=".hes"
        ))

        if filepath:
            self._save_to_file(filepath)
            self.current_file = filepath
            self._update_title()

    def _save_to_file(self, filepath: str):
        """Speichert den Inhalt in eine Datei."""
        try:
            content = self.equations_text.get("1.0", tk.END)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            self.status_var.set(f"Gespeichert: {filepath}")
        except Exception as e:
            messagebox.showerror("Fehler", f"Datei konnte nicht gespeichert werden:\n{e}")

    def _update_title(self):
        """Aktualisiert den Fenstertitel."""
        if self.current_file:
            import os
            filename = os.path.basename(self.current_file)
            self.root.title(f"HVAC Equation Solver - {filename}")
        else:
            self.root.title("HVAC Equation Solver")

    def clear_solution(self):
        """Löscht nur die Solution-Ausgabe."""
        self.solution_text.configure(state=tk.NORMAL)
        self.solution_text.delete("1.0", tk.END)
        self.solution_text.configure(state=tk.DISABLED)

    def write_solution(self, text: str, tag: str = None):
        """Schreibt Text in das Solution-Fenster."""
        self.solution_text.configure(state=tk.NORMAL)
        if tag:
            self.solution_text.insert(tk.END, text, tag)
        else:
            self.solution_text.insert(tk.END, text)
        self.solution_text.configure(state=tk.DISABLED)

    def show_initial_values_dialog(self):
        """Zeigt einen Dialog zum Setzen von manuellen Startwerten."""
        if not self.known_variables:
            messagebox.showinfo(
                "Initial Values",
                "Bitte zuerst Solve drücken, damit die Variablen erkannt werden."
            )
            return

        # Erstelle Dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Initial Values")
        dialog.geometry("400x500")
        dialog.transient(self.root)
        dialog.grab_set()

        # Hauptframe mit Scrollbar
        main_frame = ttk.Frame(dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Info Label
        info_label = ttk.Label(
            main_frame,
            text="Startwerte für Variablen setzen:\n(Leer lassen für automatischen Startwert)",
            justify=tk.LEFT
        )
        info_label.pack(anchor=tk.W, pady=(0, 10))

        # Canvas mit Scrollbar für die Variablenliste
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(canvas_frame)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Eingabefelder für jede Variable
        entries = {}
        for var in sorted(self.known_variables):
            row_frame = ttk.Frame(scrollable_frame)
            row_frame.pack(fill=tk.X, pady=2)

            label = ttk.Label(row_frame, text=f"{var}:", width=20, anchor=tk.W)
            label.pack(side=tk.LEFT, padx=(0, 5))

            entry = ttk.Entry(row_frame, width=15)
            entry.pack(side=tk.LEFT)

            # Aktuellen Wert eintragen falls vorhanden
            if var in self.manual_initial_values:
                entry.insert(0, str(self.manual_initial_values[var]))

            entries[var] = entry

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))

        def apply_values():
            """Übernimmt die Werte."""
            self.manual_initial_values.clear()
            for var, entry in entries.items():
                value_str = entry.get().strip()
                if value_str:
                    try:
                        value = float(value_str)
                        self.manual_initial_values[var] = value
                    except ValueError:
                        messagebox.showerror(
                            "Fehler",
                            f"Ungültiger Wert für {var}: '{value_str}'"
                        )
                        return
            dialog.destroy()
            self.status_var.set(f"{len(self.manual_initial_values)} Startwerte gesetzt")

        def clear_all_values():
            """Löscht alle Startwerte."""
            for entry in entries.values():
                entry.delete(0, tk.END)
            self.manual_initial_values.clear()

        ttk.Button(button_frame, text="Clear All", command=clear_all_values).pack(side=tk.LEFT)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(button_frame, text="OK", command=apply_values).pack(side=tk.RIGHT)

        # Scrollrad-Unterstützung
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", on_mousewheel)

        # Cleanup beim Schließen
        def on_close():
            canvas.unbind_all("<MouseWheel>")
            dialog.destroy()

        dialog.protocol("WM_DELETE_WINDOW", on_close)

    def show_plot_dialog(self):
        """Zeigt einen Dialog zum Erstellen eines Plots mit mehreren Y-Variablen."""
        if not MATPLOTLIB_AVAILABLE:
            messagebox.showerror("Fehler", "matplotlib nicht verfügbar.\nBitte installieren: pip install matplotlib")
            return

        if self.last_solution is None:
            messagebox.showinfo("Plot", "Bitte zuerst Solve drücken, um Daten zu erhalten.")
            return

        # Prüfe ob es Array-Daten gibt (Parameterstudie)
        has_arrays = any(isinstance(v, np.ndarray) for v in self.last_solution.values())
        if not has_arrays:
            messagebox.showinfo(
                "Plot",
                "Plot erfordert eine Parameterstudie mit Vektordaten.\n\n"
                "Beispiel:\n  T = 0:10:100\n  h = enthalpy(water, T=T, x=0)"
            )
            return

        # Finde alle Array-Variablen
        array_vars = sorted([k for k, v in self.last_solution.items() if isinstance(v, np.ndarray)])

        # Erstelle Dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("New Plot")
        dialog.geometry("450x580")
        dialog.minsize(400, 550)
        dialog.transient(self.root)
        dialog.grab_set()

        main_frame = ttk.Frame(dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # X-Achse Auswahl
        x_frame = ttk.LabelFrame(main_frame, text="X-Achse", padding="5")
        x_frame.pack(fill=tk.X, pady=(0, 10))

        x_var = tk.StringVar()
        x_combo = ttk.Combobox(x_frame, textvariable=x_var, values=array_vars, state="readonly", width=30)
        x_combo.pack(fill=tk.X)
        if array_vars:
            x_combo.current(0)

        x_label_var = tk.StringVar()
        ttk.Label(x_frame, text="Label:").pack(anchor=tk.W, pady=(5, 0))
        x_label_entry = ttk.Entry(x_frame, textvariable=x_label_var, width=30)
        x_label_entry.pack(fill=tk.X)

        # Y-Achsen Auswahl (mehrere möglich)
        y_frame = ttk.LabelFrame(main_frame, text="Y-Achse(n) - Mehrfachauswahl möglich", padding="5")
        y_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Listbox mit Scrollbar für Y-Variablen
        y_list_frame = ttk.Frame(y_frame)
        y_list_frame.pack(fill=tk.BOTH, expand=True)

        y_scrollbar = ttk.Scrollbar(y_list_frame)
        y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        y_listbox = tk.Listbox(y_list_frame, selectmode=tk.MULTIPLE, height=8, yscrollcommand=y_scrollbar.set)
        y_listbox.pack(fill=tk.BOTH, expand=True)
        y_scrollbar.config(command=y_listbox.yview)

        for var in array_vars:
            y_listbox.insert(tk.END, var)

        y_label_var = tk.StringVar()
        ttk.Label(y_frame, text="Y-Label:").pack(anchor=tk.W, pady=(5, 0))
        y_label_entry = ttk.Entry(y_frame, textvariable=y_label_var, width=30)
        y_label_entry.pack(fill=tk.X)

        # Plot-Titel
        title_frame = ttk.LabelFrame(main_frame, text="Plot-Titel", padding="5")
        title_frame.pack(fill=tk.X, pady=(0, 10))

        title_var = tk.StringVar()
        title_entry = ttk.Entry(title_frame, textvariable=title_var, width=40)
        title_entry.pack(fill=tk.X)

        # Optionen
        options_frame = ttk.LabelFrame(main_frame, text="Optionen", padding="5")
        options_frame.pack(fill=tk.X, pady=(0, 10))

        grid_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Gitterlinien anzeigen", variable=grid_var).pack(anchor=tk.W)

        legend_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Legende anzeigen", variable=legend_var).pack(anchor=tk.W)

        markers_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Datenpunkte markieren", variable=markers_var).pack(anchor=tk.W)

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)

        def create_plot():
            x_name = x_var.get()
            y_indices = y_listbox.curselection()

            if not x_name:
                messagebox.showwarning("Warnung", "Bitte X-Variable auswählen")
                return
            if not y_indices:
                messagebox.showwarning("Warnung", "Bitte mindestens eine Y-Variable auswählen")
                return

            y_names = [array_vars[i] for i in y_indices]
            x_data = self.last_solution[x_name]
            y_data_list = [(name, self.last_solution[name]) for name in y_names]

            self._create_plot_window(
                x_data, y_data_list,
                x_label=x_label_var.get() or x_name,
                y_label=y_label_var.get() or (y_names[0] if len(y_names) == 1 else ""),
                title=title_var.get(),
                show_grid=grid_var.get(),
                show_legend=legend_var.get(),
                show_markers=markers_var.get()
            )
            dialog.destroy()

        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(button_frame, text="Plot", command=create_plot).pack(side=tk.RIGHT)

    def show_quick_plot_dialog(self):
        """Zeigt einen vereinfachten Dialog für schnelle X-Y Plots."""
        if not MATPLOTLIB_AVAILABLE:
            messagebox.showerror("Fehler", "matplotlib nicht verfügbar.\nBitte installieren: pip install matplotlib")
            return

        if self.last_solution is None:
            messagebox.showinfo("Plot", "Bitte zuerst Solve drücken, um Daten zu erhalten.")
            return

        # Prüfe ob es Array-Daten gibt
        has_arrays = any(isinstance(v, np.ndarray) for v in self.last_solution.values())
        if not has_arrays:
            messagebox.showinfo(
                "Plot",
                "Plot erfordert eine Parameterstudie mit Vektordaten.\n\n"
                "Beispiel:\n  T = 0:10:100\n  h = enthalpy(water, T=T, x=0)"
            )
            return

        # Finde alle Array-Variablen
        array_vars = sorted([k for k, v in self.last_solution.items() if isinstance(v, np.ndarray)])

        # Erstelle einfachen Dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Quick Plot")
        dialog.geometry("300x150")
        dialog.transient(self.root)
        dialog.grab_set()

        main_frame = ttk.Frame(dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # X-Achse
        ttk.Label(main_frame, text="X-Achse:").grid(row=0, column=0, sticky=tk.W, pady=5)
        x_var = tk.StringVar()
        x_combo = ttk.Combobox(main_frame, textvariable=x_var, values=array_vars, state="readonly", width=20)
        x_combo.grid(row=0, column=1, pady=5)
        if array_vars:
            x_combo.current(0)

        # Y-Achse
        ttk.Label(main_frame, text="Y-Achse:").grid(row=1, column=0, sticky=tk.W, pady=5)
        y_var = tk.StringVar()
        y_combo = ttk.Combobox(main_frame, textvariable=y_var, values=array_vars, state="readonly", width=20)
        y_combo.grid(row=1, column=1, pady=5)
        if len(array_vars) > 1:
            y_combo.current(1)
        elif array_vars:
            y_combo.current(0)

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=(20, 0))

        def quick_plot():
            x_name = x_var.get()
            y_name = y_var.get()

            if not x_name or not y_name:
                messagebox.showwarning("Warnung", "Bitte X und Y Variable auswählen")
                return

            x_data = self.last_solution[x_name]
            y_data_list = [(y_name, self.last_solution[y_name])]

            self._create_plot_window(
                x_data, y_data_list,
                x_label=x_name,
                y_label=y_name,
                title=f"{y_name} vs {x_name}",
                show_grid=True,
                show_legend=False,
                show_markers=False
            )
            dialog.destroy()

        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(button_frame, text="Plot", command=quick_plot).pack(side=tk.RIGHT)

    def _create_plot_window(self, x_data, y_data_list, x_label="", y_label="", title="",
                            show_grid=True, show_legend=True, show_markers=False):
        """Erstellt ein neues Plot-Fenster mit matplotlib."""
        if not MATPLOTLIB_AVAILABLE:
            return

        # Erstelle neues Fenster
        plot_window = tk.Toplevel(self.root)
        plot_window.title(f"Plot: {title}" if title else "Plot")
        plot_window.geometry("800x600")

        # Erstelle Figure und Axes
        fig = Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)

        # Farben für mehrere Linien
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        # Plotte alle Y-Daten
        marker = 'o' if show_markers else None
        for i, (name, y_data) in enumerate(y_data_list):
            color = colors[i % len(colors)]
            ax.plot(x_data, y_data, label=name, color=color, marker=marker, markersize=4)

        # Beschriftungen
        if x_label:
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_ylabel(y_label)
        if title:
            ax.set_title(title)

        # Optionen
        if show_grid:
            ax.grid(True, linestyle='--', alpha=0.7)
        if show_legend and len(y_data_list) > 1:
            ax.legend()

        # Tight layout für bessere Darstellung
        fig.tight_layout()

        # Canvas in Tkinter einbetten
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()

        # Toolbar hinzufügen (Zoom, Pan, Save, etc.) - OBEN platzieren
        toolbar_frame = ttk.Frame(plot_window)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()

        # Canvas NACH der Toolbar packen, damit es den restlichen Platz füllt
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Cleanup bei Fenster-Schließung
        def on_close():
            plt.close(fig)
            plot_window.destroy()

        plot_window.protocol("WM_DELETE_WINDOW", on_close)

    def solve(self):
        """Löst das Gleichungssystem."""
        self.clear_solution()
        self.status_var.set("Löse...")
        self.root.update()

        # Hole Gleichungen
        equations_text = self.equations_text.get("1.0", tk.END)

        try:
            # Parse Gleichungen (jetzt mit 4 Rückgabewerten)
            equations, variables, initial_values, sweep_vars = parse_equations(equations_text)

            # Speichere gefundene Variablen (inkl. Sweep-Variablen für Initial Values Dialog)
            self.known_variables = variables.copy()
            self.known_variables.update(sweep_vars.keys())

            # Speichere letzte Lösung für Plots
            self.last_solution = None
            self.last_sweep_vars = sweep_vars

            # Validiere System
            valid, msg = validate_system(equations, variables)

            self.write_solution("=== Systemanalyse ===\n", "info")
            self.write_solution(f"Gefundene Variablen: {', '.join(sorted(variables))}\n")
            self.write_solution(f"Anzahl Gleichungen: {len(equations)}\n")

            # Zeige Sweep-Variablen
            if sweep_vars:
                self.write_solution("\nParameterstudie:\n", "info")
                for name, arr in sweep_vars.items():
                    self.write_solution(f"  {name}: {arr[0]:.4g} bis {arr[-1]:.4g} ({len(arr)} Werte)\n")

            self.write_solution(f"\nStatus: {msg}\n\n")

            # Spezialfall: Keine Gleichungen, aber Konstanten berechnet
            if not valid and len(equations) == 0 and initial_values:
                self.write_solution("=== Berechnete Konstanten ===\n", "info")
                for var in sorted(initial_values.keys()):
                    val = initial_values[var]
                    if abs(val) >= 1e6 or (abs(val) < 1e-4 and val != 0):
                        self.write_solution(f"{var} = {val:.6e}\n")
                    else:
                        self.write_solution(f"{var} = {val:.6g}\n")
                self.last_solution = initial_values.copy()
                self.status_var.set("Konstanten berechnet")
                return

            if not valid:
                self.write_solution("FEHLER: ", "error")
                self.write_solution(f"{msg}\n")
                self.status_var.set("Fehler: System nicht lösbar")
                return

            # initial_values vom Parser sind Konstanten (direkte Zuweisungen)
            # manual_initial_values sind Startwerte für den Solver
            constants = initial_values.copy()

            # Manuelle Startwerte nur für unbekannte Variablen
            solver_initial = {}
            for var, val in self.manual_initial_values.items():
                if var in variables:
                    solver_initial[var] = val

            # Löse System (mit oder ohne Parameterstudie)
            if sweep_vars:
                # Parameterstudie mit Fortschrittsanzeige
                def progress_callback(current, total):
                    self.status_var.set(f"Löse... {current}/{total}")
                    self.root.update()

                success, solution, solve_msg = solve_parametric(
                    equations, variables, sweep_vars, solver_initial,
                    progress_callback=progress_callback, constants=constants
                )
            else:
                # Normale Lösung
                success, solution, solve_msg = solve_system(
                    equations, variables, solver_initial, constants=constants
                )

            self.write_solution("=== Lösung ===\n", "info")

            if success:
                self.write_solution(f"{solve_msg}\n\n", "success")
                self.write_solution(format_solution(solution) + "\n")
                self.last_solution = solution
                if sweep_vars:
                    self.status_var.set(f"Parameterstudie: {len(list(sweep_vars.values())[0])} Punkte")
                else:
                    self.status_var.set("Lösung gefunden")
            else:
                self.write_solution(f"{solve_msg}\n\n", "error")
                if solution:
                    self.write_solution("Letzte Näherung:\n")
                    self.write_solution(format_solution(solution) + "\n")
                self.status_var.set("Konvergenzproblem")

        except Exception as e:
            self.write_solution("FEHLER:\n", "error")
            self.write_solution(str(e) + "\n")
            self.status_var.set(f"Fehler: {e}")


def main():
    """Hauptfunktion."""
    root = tk.Tk()

    # Setze Style
    style = ttk.Style()
    if 'clam' in style.theme_names():
        style.theme_use('clam')

    app = EquationSolverApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
