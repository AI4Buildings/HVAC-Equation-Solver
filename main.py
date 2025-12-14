#!/usr/bin/env python3
"""
HVAC Equation Solver - Ein EES-ähnlicher Gleichungslöser

Hauptanwendung mit CustomTkinter GUI.
"""

import os
import sys

# Unterdrücke macOS-spezifische Warnungen
if sys.platform == 'darwin':
    os.environ['TK_SILENCE_DEPRECATION'] = '1'

import tkinter as tk
from tkinter import messagebox, filedialog

import customtkinter as ctk

from parser import parse_equations, validate_system
from solver import solve_system, solve_parametric, format_solution, SolveAnalysis
import numpy as np

# Versuche Units-Modul zu laden
try:
    from units import get_compatible_units, UnitValue, detect_unit_from_equation, get_initial_from_unit
    UNITS_AVAILABLE = True
except ImportError:
    UNITS_AVAILABLE = False
    def get_initial_from_unit(unit_str):
        return 1.0

# Versuche Constraint-Propagation zu laden
try:
    from unit_constraints import propagate_all_units, check_all_unit_consistency, propagate_all_units_complete
    CONSTRAINT_PROPAGATION_AVAILABLE = True
except ImportError:
    CONSTRAINT_PROPAGATION_AVAILABLE = False

# CustomTkinter Einstellungen
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

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

# Versuche CoolProp Version zu ermitteln
try:
    import CoolProp
    COOLPROP_VERSION = CoolProp.__version__
except:
    COOLPROP_VERSION = None


# Farbschema
COLORS = {
    "bg_dark": "#1a1a2e",
    "bg_frame": "#16213e",
    "bg_input": "#0f0f1a",
    "accent": "#e94560",
    "accent_hover": "#ff6b6b",
    "text": "#eaeaea",
    "text_dim": "#8892a0",
    "success": "#4ade80",
    "error": "#f87171",
    "warning": "#fbbf24",
    "info": "#60a5fa",
    "value": "#fbbf24",
    "border": "#2d3748",
}


def _suppress_macos_warning(func):
    """Wrapper um macOS Cocoa-Warnungen bei Dateidialogen zu unterdrücken."""
    if sys.platform != 'darwin':
        return func()
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2)
    os.dup2(devnull, 2)
    os.close(devnull)
    try:
        return func()
    finally:
        os.dup2(saved, 2)
        os.close(saved)


class EquationSolverApp(ctk.CTk):
    """Hauptanwendung für den Gleichungslöser."""

    def __init__(self):
        super().__init__()

        # Fenster-Konfiguration
        self.title("HVAC Equation Solver")
        self.geometry("1200x800")
        self.minsize(800, 600)

        # Setze Hintergrundfarbe
        self.configure(fg_color=COLORS["bg_dark"])

        # Aktueller Dateipfad
        self.current_file = None

        # Schriftgröße (Standard: 14)
        self.font_size = 14

        # Gespeicherte Variablen und manuelle Startwerte
        self.known_variables = set()
        self.manual_initial_values = {}

        # Letzte Lösung (für Plots und Analysis)
        self.last_solution = None
        self.last_sweep_vars = {}
        self.last_solve_stats = {}
        self.last_analysis = None
        self.current_unit_values = {}  # Einheiten-Informationen für Variablen
        self.value_labels = {}  # Referenzen auf Value-Labels (für Unit-Änderung)
        self.unit_dropdowns = {}  # Referenzen auf Unit-Dropdowns
        self.temp_display_unit = ctk.StringVar(value="degC")  # Standard-Anzeigeeinheit für Temperaturen (°C)
        self.pressure_display_unit = ctk.StringVar(value="bar")  # Standard: bar (statt Pa)
        self.energy_display_unit = ctk.StringVar(value="kJ")  # Standard: kJ (statt J)
        self.power_display_unit = ctk.StringVar(value="kW")  # Standard: kW (statt W)

        # Grid-Konfiguration
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Erstelle die GUI-Elemente
        self._create_header()
        self._create_main_content()
        self._create_statusbar()
        self._create_menu()
        self._setup_bindings()

    def _create_header(self):
        """Erstellt den Header mit Logo, Titel und Buttons."""
        header = ctk.CTkFrame(self, fg_color=COLORS["bg_frame"], corner_radius=0, height=60)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(1, weight=1)

        # Logo/Icon Frame
        logo_frame = ctk.CTkFrame(header, fg_color=COLORS["accent"], width=50, height=50, corner_radius=8)
        logo_frame.grid(row=0, column=0, padx=15, pady=8)
        logo_frame.grid_propagate(False)

        logo_label = ctk.CTkLabel(logo_frame, text="Σ", font=ctk.CTkFont(size=28, weight="bold"),
                                   text_color="white")
        logo_label.place(relx=0.5, rely=0.5, anchor="center")

        # Titel
        title_frame = ctk.CTkFrame(header, fg_color="transparent")
        title_frame.grid(row=0, column=1, sticky="w", padx=10)

        title_label = ctk.CTkLabel(title_frame, text="HVAC Equation Solver",
                                    font=ctk.CTkFont(size=20, weight="bold"),
                                    text_color=COLORS["text"])
        title_label.pack(anchor="w")

        subtitle_label = ctk.CTkLabel(title_frame, text="Thermodynamic System Analysis",
                                       font=ctk.CTkFont(size=12),
                                       text_color=COLORS["text_dim"])
        subtitle_label.pack(anchor="w")

        # Buttons Frame
        buttons_frame = ctk.CTkFrame(header, fg_color="transparent")
        buttons_frame.grid(row=0, column=2, padx=15, pady=8)

        # Solve Button (prominent)
        self.solve_btn = ctk.CTkButton(
            buttons_frame,
            text="▷ Solve",
            command=self.solve,
            width=100,
            height=36,
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.solve_btn.pack(side="left", padx=(0, 5))

        # F5 Badge
        f5_label = ctk.CTkLabel(buttons_frame, text="F5", font=ctk.CTkFont(size=10),
                                 text_color=COLORS["text_dim"],
                                 fg_color=COLORS["bg_dark"], corner_radius=4,
                                 width=24, height=18)
        f5_label.pack(side="left", padx=(0, 10))

        # Clear Button
        self.clear_btn = ctk.CTkButton(
            buttons_frame,
            text="⊘ Clear",
            command=self.clear_all,
            width=80,
            height=36,
            fg_color=COLORS["bg_dark"],
            hover_color=COLORS["border"],
            border_width=1,
            border_color=COLORS["border"],
            font=ctk.CTkFont(size=13)
        )
        self.clear_btn.pack(side="left", padx=(0, 5))

        # Examples Button
        self.example_btn = ctk.CTkButton(
            buttons_frame,
            text="☰ Examples",
            command=self._insert_example,
            width=100,
            height=36,
            fg_color=COLORS["bg_dark"],
            hover_color=COLORS["border"],
            border_width=1,
            border_color=COLORS["border"],
            font=ctk.CTkFont(size=13)
        )
        self.example_btn.pack(side="left", padx=(0, 5))

        # Settings Button
        self.settings_btn = ctk.CTkButton(
            buttons_frame,
            text="⚙",
            command=self.show_settings,
            width=36,
            height=36,
            fg_color=COLORS["bg_dark"],
            hover_color=COLORS["border"],
            border_width=1,
            border_color=COLORS["border"],
            font=ctk.CTkFont(size=18)
        )
        self.settings_btn.pack(side="left")

    def _create_main_content(self):
        """Erstellt den Hauptinhalt mit Equations und Solution Panels."""
        # Container Frame
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        content.grid_columnconfigure(0, weight=1)
        content.grid_rowconfigure(0, weight=1)

        # PanedWindow für verschiebbare Trennung (verwende tkinter.ttk)
        from tkinter import ttk

        # Style für PanedWindow im Dark Mode
        style = ttk.Style()
        style.configure("Dark.TPanedwindow", background=COLORS["bg_dark"])

        self.paned = ttk.PanedWindow(content, orient=tk.HORIZONTAL)
        self.paned.grid(row=0, column=0, sticky="nsew")

        # Linkes Panel: Equations (in eigenem Frame für PanedWindow)
        eq_container = ctk.CTkFrame(self.paned, fg_color="transparent")
        self._create_equations_panel(eq_container)
        self.paned.add(eq_container, weight=2)

        # Rechtes Panel: Solution (in eigenem Frame für PanedWindow)
        sol_container = ctk.CTkFrame(self.paned, fg_color="transparent")
        self._create_solution_panel(sol_container)
        self.paned.add(sol_container, weight=1)

    def _create_equations_panel(self, parent):
        """Erstellt das Equations Panel."""
        # Parent konfigurieren für Ausdehnung
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)

        eq_frame = ctk.CTkFrame(parent, fg_color=COLORS["bg_frame"], corner_radius=10)
        eq_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=0)
        eq_frame.grid_columnconfigure(0, weight=1)
        eq_frame.grid_rowconfigure(1, weight=1)

        # Header
        header = ctk.CTkFrame(eq_frame, fg_color="transparent", height=40)
        header.grid(row=0, column=0, sticky="ew", padx=15, pady=(10, 5))

        # Icon und Titel
        ctk.CTkLabel(header, text="T", font=ctk.CTkFont(size=16, weight="bold"),
                      text_color=COLORS["info"]).pack(side="left")
        ctk.CTkLabel(header, text="  EQUATIONS", font=ctk.CTkFont(size=13, weight="bold"),
                      text_color=COLORS["text"]).pack(side="left")

        # Syntax-Hinweis
        syntax_hint = ctk.CTkLabel(
            header,
            text='x + y = 10   ·  Kommentare: "..." oder {...}',
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_dim"]
        )
        syntax_hint.pack(side="right")

        # Text Editor
        self.equations_text = ctk.CTkTextbox(
            eq_frame,
            font=ctk.CTkFont(family="Courier", size=self.font_size),
            fg_color=COLORS["bg_input"],
            text_color=COLORS["text"],
            corner_radius=8,
            border_width=1,
            border_color=COLORS["border"],
            wrap="none"
        )
        self.equations_text.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

    def _create_solution_panel(self, parent):
        """Erstellt das Solution Panel mit TabView."""
        # Parent konfigurieren für Ausdehnung
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)

        sol_frame = ctk.CTkFrame(parent, fg_color=COLORS["bg_frame"], corner_radius=10)
        sol_frame.grid(row=0, column=0, sticky="nsew", padx=(5, 0), pady=0)
        sol_frame.grid_columnconfigure(0, weight=1)
        sol_frame.grid_rowconfigure(1, weight=1)

        # Header
        header = ctk.CTkFrame(sol_frame, fg_color="transparent", height=40)
        header.grid(row=0, column=0, sticky="ew", padx=15, pady=(10, 5))

        # Icon und Titel
        ctk.CTkLabel(header, text="☑", font=ctk.CTkFont(size=16),
                      text_color=COLORS["success"]).pack(side="left")
        ctk.CTkLabel(header, text="  SOLUTION", font=ctk.CTkFont(size=13, weight="bold"),
                      text_color=COLORS["text"]).pack(side="left")

        # TabView für Results und Residuals
        self.tab_view = ctk.CTkTabview(
            sol_frame,
            fg_color=COLORS["bg_input"],
            segmented_button_fg_color=COLORS["bg_dark"],
            segmented_button_selected_color=COLORS["accent"],
            segmented_button_unselected_color=COLORS["bg_frame"],
            corner_radius=8
        )
        self.tab_view.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

        # Results Tab
        self.results_tab = self.tab_view.add("Results")
        self._create_results_tab()

        # Residuals Tab
        self.residuals_tab = self.tab_view.add("Residuals")
        self._create_residuals_tab()

    def _create_results_tab(self):
        """Erstellt den Inhalt des Results Tabs."""
        self.results_tab.grid_columnconfigure(0, weight=1)
        self.results_tab.grid_rowconfigure(2, weight=1)

        # Status Frame
        self.status_frame = ctk.CTkFrame(self.results_tab, fg_color="transparent", height=50)
        self.status_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        # Status Label (SOLUTION FOUND / ERROR)
        self.result_status_label = ctk.CTkLabel(
            self.status_frame,
            text="",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLORS["text_dim"]
        )
        self.result_status_label.pack(side="left")

        # Unit Warning Label (⚠ UNIT WARNINGS)
        self.unit_warning_label = ctk.CTkLabel(
            self.status_frame,
            text="",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLORS["warning"]
        )
        self.unit_warning_label.pack(side="left", padx=(20, 0))

        # Stats Label (15 direct, 1 iterative)
        self.result_stats_label = ctk.CTkLabel(
            self.status_frame,
            text="",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_dim"]
        )
        self.result_stats_label.pack(side="right")

        # Info Frame (Equations: X, Unknowns: Y, Status: OK)
        self.info_frame = ctk.CTkFrame(self.results_tab, fg_color=COLORS["bg_dark"],
                                        corner_radius=6, height=35)
        self.info_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=(0, 10))

        self.info_label = ctk.CTkLabel(
            self.info_frame,
            text="",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_dim"]
        )
        self.info_label.pack(padx=10, pady=8)

        # Scrollable Frame für Variablen-Tabelle
        self.results_scroll = ctk.CTkScrollableFrame(
            self.results_tab,
            fg_color="transparent",
            corner_radius=0
        )
        self.results_scroll.grid(row=2, column=0, sticky="nsew", padx=5)
        self.results_scroll.grid_columnconfigure(0, weight=1)
        self.results_scroll.grid_columnconfigure(1, weight=0)

        # Header der Tabelle
        self.table_header = ctk.CTkFrame(self.results_scroll, fg_color="transparent")
        self.table_header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 5))

        ctk.CTkLabel(self.table_header, text="VARIABLE", font=ctk.CTkFont(size=11, weight="bold"),
                      text_color=COLORS["text_dim"], width=150, anchor="w").pack(side="left", padx=5)
        ctk.CTkLabel(self.table_header, text="VALUE", font=ctk.CTkFont(size=11, weight="bold"),
                      text_color=COLORS["text_dim"], anchor="e").pack(side="right", padx=5)

        # Container für Variablen-Zeilen
        self.var_rows_container = ctk.CTkFrame(self.results_scroll, fg_color="transparent")
        self.var_rows_container.grid(row=1, column=0, columnspan=2, sticky="nsew")

    def _create_residuals_tab(self):
        """Erstellt den Inhalt des Residuals Tabs."""
        self.residuals_tab.grid_columnconfigure(0, weight=1)
        self.residuals_tab.grid_rowconfigure(0, weight=1)

        # Scrollable Frame für Residuals
        self.residuals_scroll = ctk.CTkScrollableFrame(
            self.residuals_tab,
            fg_color="transparent",
            corner_radius=0
        )
        self.residuals_scroll.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.residuals_scroll.grid_columnconfigure(0, weight=1)

        # Container für Residuals-Sektionen
        self.residuals_content = ctk.CTkFrame(self.residuals_scroll, fg_color="transparent")
        self.residuals_content.grid(row=0, column=0, sticky="nsew")
        self.residuals_content.grid_columnconfigure(0, weight=1)

        # Placeholder Text (wird versteckt wenn Daten vorhanden)
        self.residuals_placeholder = ctk.CTkLabel(
            self.residuals_content,
            text="Run Solve to see residuals.\n\n"
                 "This tab will show:\n"
                 "• Constants\n"
                 "• Direct Evaluations\n"
                 "• Block decomposition with residuals",
            font=ctk.CTkFont(size=12),
            text_color=COLORS["text_dim"],
            justify="center"
        )
        self.residuals_placeholder.grid(row=0, column=0, pady=50)

        # Sections storage
        self.residuals_sections = []

    def _create_collapsible_section(self, parent, title: str, count: int, row: int,
                                      header_color: str = None) -> ctk.CTkFrame:
        """Erstellt eine aufklappbare Sektion für die Residuals."""
        # Main container
        section_frame = ctk.CTkFrame(parent, fg_color=COLORS["bg_frame"], corner_radius=6)
        section_frame.grid(row=row, column=0, sticky="ew", pady=3)
        section_frame.grid_columnconfigure(0, weight=1)

        # Header Frame (klickbar)
        header = ctk.CTkFrame(section_frame, fg_color="transparent", height=30)
        header.grid(row=0, column=0, sticky="ew", padx=5, pady=3)
        header.grid_columnconfigure(1, weight=1)

        # Expand/Collapse Button
        expand_var = tk.BooleanVar(value=True)
        expand_btn = ctk.CTkLabel(
            header, text="▼", width=20,
            font=ctk.CTkFont(size=10),
            text_color=header_color or COLORS["text_dim"]
        )
        expand_btn.grid(row=0, column=0, padx=(5, 0))

        # Title
        title_label = ctk.CTkLabel(
            header, text=f"{title} ({count})",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=header_color or COLORS["text"],
            anchor="w"
        )
        title_label.grid(row=0, column=1, sticky="w", padx=5)

        # Content Frame
        content_frame = ctk.CTkFrame(section_frame, fg_color="transparent")
        content_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 5))
        content_frame.grid_columnconfigure(0, weight=1)

        # Toggle function
        def toggle():
            if expand_var.get():
                content_frame.grid_remove()
                expand_btn.configure(text="▶")
                expand_var.set(False)
            else:
                content_frame.grid()
                expand_btn.configure(text="▼")
                expand_var.set(True)

        # Bind click to toggle
        expand_btn.bind("<Button-1>", lambda e: toggle())
        title_label.bind("<Button-1>", lambda e: toggle())
        header.bind("<Button-1>", lambda e: toggle())

        return content_frame

    def _add_equation_row(self, parent, equation: str, var: str, value: float, residual: float, row: int):
        """Fügt eine Zeile für eine Gleichung zu den Residuals hinzu."""
        row_frame = ctk.CTkFrame(parent, fg_color="transparent", height=22)
        row_frame.grid(row=row, column=0, sticky="ew", pady=1)
        row_frame.grid_columnconfigure(0, weight=1)

        # Residual color based on magnitude
        if abs(residual) < 1e-8:
            res_color = COLORS["success"]
        elif abs(residual) < 1e-4:
            res_color = COLORS["warning"]
        else:
            res_color = COLORS["error"]

        # Equation (truncated if too long)
        eq_display = equation if len(equation) < 50 else equation[:47] + "..."
        eq_label = ctk.CTkLabel(
            row_frame, text=eq_display,
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text"],
            anchor="w"
        )
        eq_label.grid(row=0, column=0, sticky="w")

        # Residual
        res_text = f"Res: {residual:.2E}"
        res_label = ctk.CTkLabel(
            row_frame, text=res_text,
            font=ctk.CTkFont(size=10),
            text_color=res_color,
            anchor="e"
        )
        res_label.grid(row=0, column=1, sticky="e", padx=5)

    def _add_unit_warning_row(self, parent, warning, row: int):
        """Fügt eine Zeile für eine Einheiten-Warnung hinzu."""
        from solver import UnitWarning

        # Main container für diese Warnung
        warning_frame = ctk.CTkFrame(parent, fg_color=COLORS["bg_dark"], corner_radius=4)
        warning_frame.grid(row=row, column=0, sticky="ew", pady=3)
        warning_frame.grid_columnconfigure(0, weight=1)

        # Variable Name Header
        var_frame = ctk.CTkFrame(warning_frame, fg_color="transparent")
        var_frame.pack(fill="x", padx=8, pady=(5, 2))

        ctk.CTkLabel(
            var_frame, text=f"Variable: {warning.variable}",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLORS["warning"],
            anchor="w"
        ).pack(side="left")

        # Faktor anzeigen
        if warning.conversion_factor != 1.0:
            ctk.CTkLabel(
                var_frame, text=f"Faktor {warning.conversion_factor:.0f}×",
                font=ctk.CTkFont(size=11, weight="bold"),
                text_color=COLORS["error"],
                anchor="e"
            ).pack(side="right")

        # Gleichungen mit ihren Einheiten
        for eq, unit in warning.units.items():
            eq_frame = ctk.CTkFrame(warning_frame, fg_color="transparent")
            eq_frame.pack(fill="x", padx=15, pady=1)

            # Gleichung (gekürzt)
            eq_display = eq if len(eq) < 45 else eq[:42] + "..."
            ctk.CTkLabel(
                eq_frame, text=f"• {eq_display}",
                font=ctk.CTkFont(size=10),
                text_color=COLORS["text_dim"],
                anchor="w"
            ).pack(side="left")

            # Einheit
            unit_display = unit if unit else "(dimensionslos)"
            ctk.CTkLabel(
                eq_frame, text=f"→ {unit_display}",
                font=ctk.CTkFont(size=10),
                text_color=COLORS["accent"],
                anchor="e"
            ).pack(side="right")

        # Hinweis
        hint_frame = ctk.CTkFrame(warning_frame, fg_color="transparent")
        hint_frame.pack(fill="x", padx=8, pady=(3, 5))
        ctk.CTkLabel(
            hint_frame,
            text="⚠ Prüfen Sie die Einheiten in Ihren Gleichungen!",
            font=ctk.CTkFont(size=10),
            text_color=COLORS["warning"],
            anchor="w"
        ).pack(side="left")

    def _update_residuals_tab(self, analysis: SolveAnalysis):
        """Aktualisiert den Residuals Tab mit den Lösungsdaten."""
        # Placeholder verstecken
        self.residuals_placeholder.grid_remove()

        # Alte Sektionen löschen
        for section in self.residuals_sections:
            section.destroy()
        self.residuals_sections = []

        # Residuals direkt anzeigen
        current_row = 0

        # === Unit Warnings Section ===
        if analysis.unit_warnings:
            content = self._create_collapsible_section(
                self.residuals_content, "⚠ UNIT WARNINGS", len(analysis.unit_warnings), current_row,
                header_color=COLORS["warning"]
            )
            self.residuals_sections.append(content.master)

            for i, warning in enumerate(analysis.unit_warnings):
                self._add_unit_warning_row(content, warning, i)

            current_row += 1

        # === Constants Section ===
        if analysis.constants:
            content = self._create_collapsible_section(
                self.residuals_content, "Constants", len(analysis.constants), current_row
            )
            self.residuals_sections.append(content.master)

            for i, eq_info in enumerate(analysis.constants):
                self._add_equation_row(
                    content, eq_info.original, eq_info.variable,
                    eq_info.value, eq_info.residual, i
                )
            current_row += 1

        # === Direct Evaluations Section ===
        if analysis.direct_evals:
            content = self._create_collapsible_section(
                self.residuals_content, "Direct Evaluations", len(analysis.direct_evals), current_row
            )
            self.residuals_sections.append(content.master)

            for i, eq_info in enumerate(analysis.direct_evals):
                self._add_equation_row(
                    content, eq_info.original, eq_info.variable,
                    eq_info.value, eq_info.residual, i
                )
            current_row += 1

        # === Single Unknowns Section ===
        if analysis.single_unknowns:
            content = self._create_collapsible_section(
                self.residuals_content, "Single Unknown (Iterative)", len(analysis.single_unknowns), current_row
            )
            self.residuals_sections.append(content.master)

            for i, eq_info in enumerate(analysis.single_unknowns):
                self._add_equation_row(
                    content, eq_info.original, eq_info.variable,
                    eq_info.value, eq_info.residual, i
                )
            current_row += 1

        # === Blocks Section ===
        for block in analysis.blocks:
            title = f"Block {block.block_number}"
            content = self._create_collapsible_section(
                self.residuals_content, title, len(block.equations), current_row
            )
            self.residuals_sections.append(content.master)

            # Block header mit Max-Residuum
            max_res_frame = ctk.CTkFrame(content, fg_color="transparent")
            max_res_frame.grid(row=0, column=0, sticky="ew", pady=(0, 3))

            vars_text = ", ".join(block.variables)
            if len(vars_text) > 40:
                vars_text = vars_text[:37] + "..."

            ctk.CTkLabel(
                max_res_frame, text=f"Variables: {vars_text}",
                font=ctk.CTkFont(size=10),
                text_color=COLORS["text_dim"]
            ).pack(side="left")

            max_res_color = COLORS["success"] if block.max_residual < 1e-8 else (
                COLORS["warning"] if block.max_residual < 1e-4 else COLORS["error"]
            )
            ctk.CTkLabel(
                max_res_frame, text=f"Max Res: {block.max_residual:.2E}",
                font=ctk.CTkFont(size=10, weight="bold"),
                text_color=max_res_color
            ).pack(side="right")

            # Gleichungen im Block
            for i, (eq, res) in enumerate(zip(block.equations, block.residuals)):
                self._add_equation_row(content, eq, "", 0, res, i + 1)

            current_row += 1

        # Falls keine Daten vorhanden
        if current_row == 0:
            self.residuals_placeholder.grid()

    def _create_statusbar(self):
        """Erstellt die Statusbar am unteren Rand."""
        statusbar = ctk.CTkFrame(self, fg_color=COLORS["bg_frame"], corner_radius=0, height=30)
        statusbar.grid(row=2, column=0, sticky="ew")

        # Linke Seite: Dateiname
        self.file_label = ctk.CTkLabel(
            statusbar,
            text="Unsaved",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_dim"]
        )
        self.file_label.pack(side="left", padx=15, pady=5)

        # Rechte Seite: Versionsinfo
        version_text = f"Python {sys.version_info.major}.{sys.version_info.minor}"
        if COOLPROP_VERSION:
            version_text = f"CoolProp v{COOLPROP_VERSION}  •  {version_text}"

        version_label = ctk.CTkLabel(
            statusbar,
            text=version_text,
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_dim"]
        )
        version_label.pack(side="right", padx=15, pady=5)

        # Mitte: Status
        self.status_label = ctk.CTkLabel(
            statusbar,
            text="Ready",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text"]
        )
        self.status_label.pack(pady=5)

    def _create_menu(self):
        """Erstellt die Menüleiste."""
        menubar = tk.Menu(self)
        self.configure(menu=menubar)

        # File Menü
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New", command=self.new_file, accelerator="Ctrl+N")
        file_menu.add_command(label="Open...", command=self.open_file, accelerator="Ctrl+O")
        file_menu.add_command(label="Save", command=self.save_file, accelerator="Ctrl+S")
        file_menu.add_command(label="Save As...", command=self.save_file_as)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)

        # Edit Menü
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Clear All", command=self.clear_all)
        edit_menu.add_command(label="Insert Example", command=self._insert_example)

        # View Menü
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Increase Font Size", command=self.increase_font_size, accelerator="Ctrl++")
        view_menu.add_command(label="Decrease Font Size", command=self.decrease_font_size, accelerator="Ctrl+-")

        # Solve Menü
        solve_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Solve", menu=solve_menu)
        solve_menu.add_command(label="Solve", command=self.solve, accelerator="F5")
        solve_menu.add_separator()
        solve_menu.add_command(label="Initial Values...", command=self.show_initial_values_dialog)

        # Plot Menü
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

    def _setup_bindings(self):
        """Richtet Keyboard-Shortcuts ein."""
        self.bind("<Control-n>", lambda e: self.new_file())
        self.bind("<Control-o>", lambda e: self.open_file())
        self.bind("<Control-s>", lambda e: self.save_file())
        self.bind("<F5>", lambda e: self.solve())
        self.bind("<Control-plus>", lambda e: self.increase_font_size())
        self.bind("<Control-minus>", lambda e: self.decrease_font_size())
        self.bind("<Control-equal>", lambda e: self.increase_font_size())

    # === Schriftgröße ===

    def set_font_size(self, size: int):
        """Setzt die Schriftgröße."""
        self.font_size = max(8, min(24, size))
        self.equations_text.configure(font=ctk.CTkFont(family="Courier", size=self.font_size))
        self.status_label.configure(text=f"Font size: {self.font_size}pt")

    def increase_font_size(self):
        self.set_font_size(self.font_size + 2)

    def decrease_font_size(self):
        self.set_font_size(self.font_size - 2)

    # === Datei-Operationen ===

    def new_file(self):
        """Erstellt eine neue leere Datei."""
        self.equations_text.delete("1.0", "end")
        self.clear_results()
        self.current_file = None
        self._update_file_label()
        self.status_label.configure(text="New file")

    def open_file(self):
        """Öffnet eine Datei."""
        filetypes = [("HES Files", "*.hes"), ("Text Files", "*.txt"), ("All Files", "*.*")]
        filepath = _suppress_macos_warning(lambda: filedialog.askopenfilename(
            title="Open File", filetypes=filetypes, defaultextension=".hes"
        ))
        if filepath:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.equations_text.delete("1.0", "end")
                self.equations_text.insert("1.0", content)
                self.clear_results()
                self.current_file = filepath
                self._update_file_label()
                self.status_label.configure(text=f"Opened: {os.path.basename(filepath)}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not open file:\n{e}")

    def save_file(self):
        """Speichert die aktuelle Datei."""
        if self.current_file:
            self._save_to_file(self.current_file)
        else:
            self.save_file_as()

    def save_file_as(self):
        """Speichert die Datei unter neuem Namen."""
        filetypes = [("HES Files", "*.hes"), ("Text Files", "*.txt"), ("All Files", "*.*")]
        filepath = _suppress_macos_warning(lambda: filedialog.asksaveasfilename(
            title="Save File", filetypes=filetypes, defaultextension=".hes"
        ))
        if filepath:
            self._save_to_file(filepath)
            self.current_file = filepath
            self._update_file_label()

    def _save_to_file(self, filepath: str):
        """Speichert den Inhalt in eine Datei."""
        try:
            content = self.equations_text.get("1.0", "end-1c")
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            self.status_label.configure(text=f"Saved: {os.path.basename(filepath)}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save file:\n{e}")

    def _update_file_label(self):
        """Aktualisiert das Datei-Label in der Statusbar."""
        if self.current_file:
            self.file_label.configure(text=f"✓ Saved: {os.path.basename(self.current_file)}")
        else:
            self.file_label.configure(text="Unsaved")

    # === Lösen ===

    def solve(self):
        """Löst das Gleichungssystem."""
        self.clear_results()
        self.status_label.configure(text="Solving...")
        self.update()

        equations_text = self.equations_text.get("1.0", "end-1c")

        try:
            # Parse Gleichungen mit Einheiten
            equations, variables, initial_values, sweep_vars, original_equations, unit_values = parse_equations(equations_text, parse_units=True)

            self.known_variables = variables.copy()
            self.current_unit_values = unit_values
            self.known_variables.update(sweep_vars.keys())

            self.last_solution = None
            self.last_sweep_vars = sweep_vars
            self.last_analysis = None  # Residuals-Daten

            # Validiere System
            valid, msg = validate_system(equations, variables)

            n_equations = len(equations)
            n_variables = len(variables)

            # Spezialfall: Keine Gleichungen, aber Konstanten
            if not valid and n_equations == 0 and initial_values:
                self._show_results(initial_values, "Constants only", n_equations, n_variables, "OK")
                self.last_solution = initial_values.copy()
                self.status_label.configure(text="Constants calculated")
                return

            # Spezialfall: Keine Gleichungen, aber Sweep-Variablen
            if not valid and n_equations == 0 and sweep_vars:
                # Zeige nur die Sweep-Variablen als Ergebnis
                sweep_result = {name: arr for name, arr in sweep_vars.items()}
                n_points = len(list(sweep_vars.values())[0])
                self._show_results(sweep_result, f"Parametric: {n_points} points", 0, 0, "OK")
                self.last_solution = sweep_result
                self.status_label.configure(text=f"Parametric study: {n_points} points")
                return

            if not valid:
                self._show_error(msg)
                self.status_label.configure(text="Error: System not solvable")
                return

            constants = initial_values.copy()

            # Manuelle Startwerte
            solver_initial = {}
            for var, val in self.manual_initial_values.items():
                if var in variables:
                    solver_initial[var] = val

            # NEU: Einheiten-Propagation VOR dem Lösen für bessere Startwerte
            # Dies ist der generische Ansatz, der NICHT von Variablennamen abhängt
            self.inferred_units = {}  # Speichere abgeleitete Einheiten für Dialog
            if CONSTRAINT_PROPAGATION_AVAILABLE and UNITS_AVAILABLE:
                # Sammle bekannte Einheiten aus unit_values
                known_units = {}
                for var, uv in unit_values.items():
                    if uv.calc_unit:
                        known_units[var] = uv.calc_unit

                # Führe vollständige Einheiten-Propagation durch
                all_units = propagate_all_units_complete(original_equations, known_units)
                self.inferred_units = all_units

                # Leite Startwerte aus Einheiten ab (nur für Variablen ohne manuellen Startwert)
                for var in variables:
                    if var not in solver_initial:
                        unit = all_units.get(var)
                        if unit is not None:
                            solver_initial[var] = get_initial_from_unit(unit)

            # Löse System
            if sweep_vars:
                def progress_callback(current, total):
                    self.status_label.configure(text=f"Solving... {current}/{total}")
                    self.update()

                success, solution, solve_msg = solve_parametric(
                    equations, variables, sweep_vars, solver_initial,
                    progress_callback=progress_callback, constants=constants
                )
                # Keine Residuals für Parameterstudien
                analysis = None
            else:
                result = solve_system(
                    equations, variables, solver_initial, constants=constants,
                    original_equations=original_equations, return_analysis=True
                )
                success, solution, solve_msg, analysis = result
                self.last_analysis = analysis

            # Automatische Einheiten für Thermodynamik-Funktionen erkennen
            # Zwei-Phasen-Ansatz:
            # Phase 1: Direkte Erkennung von CoolProp/HumidAir (keine Propagation)
            # Phase 2: Propagation für Berechnungen (mehrere Durchläufe)
            if UNITS_AVAILABLE and success:
                # Phase 1: Nur Thermodynamik-Funktionen (ohne Propagation)
                for var in solution.keys():
                    if var not in self.current_unit_values:
                        for parsed_eq, orig_eq in original_equations.items():
                            if parsed_eq.startswith(f"({var}) - "):
                                # Erste Phase: nur direkte Erkennung (None = keine Propagation)
                                detected_unit = detect_unit_from_equation(orig_eq, None)
                                if detected_unit:
                                    val = solution[var]
                                    # Für Arrays: Verwende ersten Wert für UnitValue (nur für Anzeige)
                                    # Verwende from_si_base() da der Wert in SI-Basiseinheit vorliegt
                                    if isinstance(val, np.ndarray):
                                        self.current_unit_values[var] = UnitValue.from_si_base(float(val[0]), detected_unit)
                                    else:
                                        self.current_unit_values[var] = UnitValue.from_si_base(val, detected_unit)
                                break

                # Phase 2: Propagation für Berechnungen (max 5 Durchläufe)
                for pass_num in range(5):
                    found_new = False
                    for var in solution.keys():
                        if var not in self.current_unit_values:
                            for parsed_eq, orig_eq in original_equations.items():
                                if parsed_eq.startswith(f"({var}) - "):
                                    # Versuche Propagation mit bekannten Einheiten
                                    detected_unit = detect_unit_from_equation(orig_eq, self.current_unit_values)
                                    if detected_unit:
                                        val = solution[var]
                                        # Für Arrays: Verwende ersten Wert für UnitValue
                                        # Verwende from_si_base() da der Wert in SI-Basiseinheit vorliegt
                                        if isinstance(val, np.ndarray):
                                            self.current_unit_values[var] = UnitValue.from_si_base(float(val[0]), detected_unit)
                                        else:
                                            self.current_unit_values[var] = UnitValue.from_si_base(val, detected_unit)
                                        found_new = True
                                    break
                    if not found_new:
                        break

                # Phase 2.5: Constraint-Propagation für implizite Gleichungen
                # Mehrere Durchläufe, da neue Einheiten weitere Ableitungen ermöglichen
                if CONSTRAINT_PROPAGATION_AVAILABLE:
                    for propagation_pass in range(5):  # Max 5 Durchläufe
                        # Verwende calc_unit (interne Einheit, z.B. K) für konsistente Propagation
                        known_units = {var: uv.calc_unit for var, uv in self.current_unit_values.items()
                                       if uv.calc_unit}
                        # Füge Konstanten ohne Einheit als dimensionslos hinzu
                        for var in constants:
                            if var not in known_units:
                                known_units[var] = ''  # dimensionslos

                        inferred = propagate_all_units(original_equations, known_units)

                        if not inferred:
                            break  # Keine neuen Einheiten gefunden

                        found_new = False
                        for var, unit in inferred.items():
                            if var in solution and unit:  # Nur wenn unit nicht leer ist
                                # Aktualisiere auch wenn schon vorhanden aber ohne Einheit
                                existing = self.current_unit_values.get(var)
                                if existing is None or not existing.original_unit:
                                    val = solution[var]
                                    # Für Arrays: Verwende ersten Wert für UnitValue
                                    # Verwende from_si_base() da der Wert in SI-Basiseinheit vorliegt
                                    if isinstance(val, np.ndarray):
                                        self.current_unit_values[var] = UnitValue.from_si_base(float(val[0]), unit)
                                    else:
                                        self.current_unit_values[var] = UnitValue.from_si_base(val, unit)
                                    found_new = True

                        if not found_new:
                            break  # Keine neuen Einheiten hinzugefügt

                # Phase 3: Einheiten-Konsistenzprüfung
                if CONSTRAINT_PROPAGATION_AVAILABLE and analysis:
                    # Verwende calc_unit für konsistente Prüfung
                    # WICHTIG: Auch leere Einheiten ('') bedeuten "dimensionslos" und müssen enthalten sein!
                    known_units = {var: uv.calc_unit for var, uv in self.current_unit_values.items()
                                   if uv.calc_unit is not None}
                    # Füge Konstanten ohne Einheit als dimensionslos hinzu
                    for var in constants:
                        if var not in known_units:
                            known_units[var] = ''
                    unit_warnings = check_all_unit_consistency(solution, original_equations, known_units)
                    if unit_warnings:
                        analysis.unit_warnings = unit_warnings

            if success:
                self._show_results(solution, solve_msg, n_equations, n_variables, "OK")
                self.last_solution = solution
                if sweep_vars:
                    self.status_label.configure(text=f"Parametric study: {len(list(sweep_vars.values())[0])} points")
                else:
                    self.status_label.configure(text="Solution found")
                    # Residuals Tab aktualisieren
                    if analysis:
                        self._update_residuals_tab(analysis)
                        # Unit Warning Label im Results Tab aktualisieren
                        if analysis.unit_warnings:
                            n_warnings = len(analysis.unit_warnings)
                            self.unit_warning_label.configure(
                                text=f"⚠ UNIT WARNINGS ({n_warnings})"
                            )
                        else:
                            self.unit_warning_label.configure(text="")
            else:
                self._show_error(solve_msg)
                if solution:
                    self._show_results(solution, "Partial solution", n_equations, n_variables, "FAIL")
                self.status_label.configure(text="Convergence problem")
                # Auch bei Fehler Residuals anzeigen
                if analysis:
                    self._update_residuals_tab(analysis)
                    # Unit Warning Label auch bei Fehler anzeigen
                    if analysis.unit_warnings:
                        n_warnings = len(analysis.unit_warnings)
                        self.unit_warning_label.configure(
                            text=f"⚠ UNIT WARNINGS ({n_warnings})"
                        )
                    else:
                        self.unit_warning_label.configure(text="")

        except Exception as e:
            self._show_error(str(e))
            self.status_label.configure(text=f"Error: {e}")

    def clear_results(self):
        """Löscht die Ergebnisanzeige."""
        # Status zurücksetzen
        self.result_status_label.configure(text="", text_color=COLORS["text_dim"])
        self.unit_warning_label.configure(text="")  # Unit Warning zurücksetzen
        self.result_stats_label.configure(text="")
        self.info_label.configure(text="")

        # Variablen-Zeilen löschen
        for widget in self.var_rows_container.winfo_children():
            widget.destroy()

        # Unit-Referenzen zurücksetzen
        self.value_labels = {}
        self.unit_dropdowns = {}
        self.current_unit_values = {}

        # Residuals Tab zurücksetzen
        for section in self.residuals_sections:
            section.destroy()
        self.residuals_sections = []
        self.residuals_placeholder.grid()

    def _show_results(self, solution: dict, solve_msg: str, n_eq: int, n_var: int, status: str):
        """Zeigt die Ergebnisse im Results Tab an."""
        # Status
        if status == "OK":
            self.result_status_label.configure(text="● SOLUTION FOUND", text_color=COLORS["success"])
        else:
            self.result_status_label.configure(text="● PARTIAL SOLUTION", text_color=COLORS["warning"])

        # Stats aus solve_msg extrahieren (falls vorhanden)
        self.result_stats_label.configure(text=solve_msg if len(solve_msg) < 40 else "")

        # Info-Zeile
        status_color = COLORS["success"] if status == "OK" else COLORS["error"]
        # Prüfe ob Parameterstudie (Arrays in Lösung)
        has_arrays = any(isinstance(v, np.ndarray) for v in solution.values())
        if has_arrays:
            n_points = max(len(v) for v in solution.values() if isinstance(v, np.ndarray))
            info_text = f"Parametric Study: {n_points} points          Use Plot menu for visualization"
        else:
            info_text = f"Equations: {n_eq}          Unknowns: {n_var}          Status: {status}"
        self.info_label.configure(text=info_text)

        # Referenzen zurücksetzen
        self.value_labels = {}
        self.unit_dropdowns = {}

        # Variablen-Tabelle
        row_idx = 0
        for var in sorted(solution.keys()):
            val = solution[var]

            # Zeile erstellen
            row = ctk.CTkFrame(self.var_rows_container, fg_color="transparent", height=28)
            row.pack(fill="x", pady=1)

            # Variable Name
            var_label = ctk.CTkLabel(
                row, text=var,
                font=ctk.CTkFont(size=12),
                text_color=COLORS["text"],
                anchor="w", width=150
            )
            var_label.pack(side="left", padx=5)

            # Prüfe ob Variable eine Einheit hat
            unit_value = self.current_unit_values.get(var)
            has_unit = UNITS_AVAILABLE and unit_value and unit_value.original_unit

            # Wert für Anzeige bestimmen
            # Prüfe Einheiten-Typen für bevorzugte Anzeige
            temp_units = {'K', 'degC', 'degF', 'kelvin', 'celsius', 'fahrenheit', '°C', '°F'}
            pressure_units = {'Pa', 'bar', 'kPa', 'MPa', 'mbar', 'atm', 'psi'}
            energy_units = {'J', 'kJ', 'J/kg', 'kJ/kg', 'J/(kg*K)', 'kJ/(kg*K)', 'J/kg/K', 'kJ/kg/K'}
            power_units = {'W', 'kW', 'MW', 'W/m^2', 'kW/m^2', 'W/m²', 'kW/m²'}

            is_temperature = has_unit and unit_value.original_unit in temp_units
            is_pressure = has_unit and unit_value.original_unit in pressure_units
            is_energy = has_unit and any(eu in unit_value.original_unit for eu in ['J/kg', 'J/(kg', 'J/m'])
            is_power = has_unit and any(pu in unit_value.original_unit for pu in ['W/m', 'W/(m'])

            # Spezifischere Prüfung für reine J und W Einheiten
            if has_unit:
                unit_str = unit_value.original_unit
                if unit_str in {'J', 'kJ', 'MJ'} or unit_str.startswith('J/') or unit_str.startswith('kJ/'):
                    is_energy = True
                if unit_str in {'W', 'kW', 'MW'} or unit_str.startswith('W/') or unit_str.startswith('kW/'):
                    is_power = True
                if unit_str in {'Pa', 'bar', 'kPa', 'MPa'}:
                    is_pressure = True

            if isinstance(val, np.ndarray):
                # Für Arrays: Zeige Bereich (min → max) für bessere Übersicht
                min_val = np.nanmin(val)
                max_val = np.nanmax(val)
                if min_val == max_val:
                    val_text = f"[{len(val)}× {min_val:.4g}]"
                else:
                    val_text = f"[{len(val)}× {min_val:.4g}→{max_val:.4g}]"
                display_val = val
            else:
                # Bei Einheiten: Bevorzugte Anzeige-Einheit verwenden
                if has_unit:
                    # Für Temperaturen: Setting-Einheit als Standard verwenden
                    if is_temperature:
                        preferred_unit = self.temp_display_unit.get()
                        display_val = unit_value.to(preferred_unit)
                    # Für Druck: bar oder Pa je nach Setting
                    elif is_pressure:
                        preferred_unit = self.pressure_display_unit.get()
                        display_val = unit_value.to(preferred_unit)
                    # Für Energie: kJ/kg oder J/kg je nach Setting
                    elif is_energy:
                        preferred_unit = self.energy_display_unit.get()
                        # Bestimme Ziel-Einheit basierend auf Original-Einheit
                        base_unit = unit_value.original_unit or unit_value.calc_unit
                        # Spezialfall: Spezifische Wärmekapazität (kJ/kgK, J/(kg*K), etc.)
                        # Diese Einheiten enthalten /kg UND K (Kelvin)
                        if 'kgK' in base_unit or 'kg*K' in base_unit or 'kg·K' in base_unit:
                            # Behalte die vollständige Einheit, nur J<->kJ wechseln
                            if preferred_unit == 'kJ':
                                if base_unit.startswith('J'):
                                    target_unit = 'k' + base_unit
                                else:
                                    target_unit = base_unit  # bereits kJ-basiert
                            else:  # J
                                if base_unit.startswith('kJ'):
                                    target_unit = base_unit[1:]  # entferne 'k'
                                else:
                                    target_unit = base_unit  # bereits J-basiert
                        elif '/(kg' in base_unit:
                            # Format wie J/(kg*K) - ersetze J durch kJ oder umgekehrt
                            if preferred_unit == 'kJ':
                                target_unit = base_unit.replace('J/', 'kJ/')
                            else:
                                target_unit = base_unit.replace('kJ/', 'J/')
                        elif '/kg' in base_unit or '/kilogram' in base_unit.lower():
                            # Einfache spezifische Energie (J/kg, kJ/kg)
                            target_unit = f'{preferred_unit}/kg'
                        else:
                            target_unit = preferred_unit
                        display_val = unit_value.to(target_unit)
                    # Für Leistung: kW oder W je nach Setting
                    elif is_power:
                        preferred_unit = self.power_display_unit.get()
                        base_unit = unit_value.original_unit or unit_value.calc_unit
                        if '/m' in base_unit or '/meter' in base_unit.lower():
                            # Einheiten wie W/m², kW/m²
                            target_unit = base_unit.replace('W/', f'{preferred_unit}/')
                            target_unit = target_unit.replace('kW/', f'{preferred_unit}/')
                        else:
                            target_unit = preferred_unit
                        display_val = unit_value.to(target_unit)
                    else:
                        display_val = unit_value.original_value
                else:
                    display_val = val

                if abs(display_val) >= 1e6 or (abs(display_val) < 1e-4 and display_val != 0):
                    val_text = f"{display_val:.6e}"
                else:
                    val_text = f"{display_val:.6g}"

            # Unit Dropdown oder Platzhalter (rechts außen, vor Value)
            if has_unit and not isinstance(val, np.ndarray):
                compatible_units = get_compatible_units(unit_value.original_unit)

                # Bevorzugte Einheit basierend auf Settings bestimmen
                if is_temperature:
                    default_unit = self.temp_display_unit.get()
                elif is_pressure:
                    default_unit = self.pressure_display_unit.get()
                elif is_energy:
                    # Konvertiere J/kg -> kJ/kg etc.
                    base_unit = unit_value.original_unit
                    if self.energy_display_unit.get() == "kJ" and base_unit.startswith('J'):
                        default_unit = 'k' + base_unit
                    elif self.energy_display_unit.get() == "J" and base_unit.startswith('kJ'):
                        default_unit = base_unit[1:]  # Remove 'k'
                    else:
                        default_unit = base_unit
                elif is_power:
                    # Konvertiere W -> kW etc.
                    base_unit = unit_value.original_unit
                    if self.power_display_unit.get() == "kW" and base_unit.startswith('W'):
                        default_unit = 'k' + base_unit
                    elif self.power_display_unit.get() == "W" and base_unit.startswith('kW'):
                        default_unit = base_unit[1:]  # Remove 'k'
                    else:
                        default_unit = base_unit
                else:
                    default_unit = unit_value.original_unit

                unit_dropdown = ctk.CTkOptionMenu(
                    row,
                    values=compatible_units,
                    width=80,
                    height=24,
                    font=ctk.CTkFont(size=11),
                    fg_color=COLORS["bg_input"],
                    button_color=COLORS["bg_frame"],
                    button_hover_color=COLORS["accent"],
                    dropdown_fg_color=COLORS["bg_frame"],
                    dropdown_hover_color=COLORS["accent"],
                    command=lambda u, v=var: self._on_unit_changed(v, u)
                )
                unit_dropdown.set(default_unit)
                unit_dropdown.pack(side="right", padx=2)
                self.unit_dropdowns[var] = unit_dropdown
            elif isinstance(val, np.ndarray) and UNITS_AVAILABLE:
                # Für Arrays: Zeige Einheit als Label (wenn bekannt), sonst "array"
                array_unit_text = unit_value.original_unit if has_unit else "array"
                unit_label = ctk.CTkLabel(
                    row, text=array_unit_text,
                    font=ctk.CTkFont(size=11),
                    text_color=COLORS["accent"] if has_unit else COLORS["text_dim"],
                    width=80, anchor="center"
                )
                unit_label.pack(side="right", padx=2)
            elif UNITS_AVAILABLE and not isinstance(val, np.ndarray):
                # Platzhalter für dimensionslose Werte (für Spaltenausrichtung)
                unit_placeholder = ctk.CTkLabel(
                    row, text="-",
                    font=ctk.CTkFont(size=11),
                    text_color=COLORS["text_dim"],
                    width=80, anchor="center"
                )
                unit_placeholder.pack(side="right", padx=2)

            # Value Label
            val_label = ctk.CTkLabel(
                row, text=val_text,
                font=ctk.CTkFont(size=12),
                text_color=COLORS["value"],
                anchor="e",
                width=120
            )
            val_label.pack(side="right", padx=5)
            self.value_labels[var] = val_label

            row_idx += 1

    def _on_unit_changed(self, var: str, new_unit: str):
        """Aktualisiert den angezeigten Wert bei Einheitenänderung."""
        if var not in self.current_unit_values or var not in self.value_labels:
            return

        unit_value = self.current_unit_values[var]
        try:
            # Konvertiere zum neuen Unit
            new_val = unit_value.to(new_unit)

            # Formatiere Wert
            if abs(new_val) >= 1e6 or (abs(new_val) < 1e-4 and new_val != 0):
                val_text = f"{new_val:.6e}"
            else:
                val_text = f"{new_val:.6g}"

            # Update Label
            self.value_labels[var].configure(text=val_text)
        except Exception as e:
            print(f"Unit conversion error for {var}: {e}")

    def _show_error(self, message: str):
        """Zeigt eine Fehlermeldung im Results Tab an."""
        self.result_status_label.configure(text="● ERROR", text_color=COLORS["error"])
        self.info_label.configure(text=message[:80] + "..." if len(message) > 80 else message)

    def clear_all(self):
        """Löscht alle Eingaben und Ausgaben."""
        self.equations_text.delete("1.0", "end")
        self.clear_results()
        self.status_label.configure(text="Ready")

    # === Dialoge ===

    def show_settings(self):
        """Zeigt den Settings Dialog."""
        dialog = ctk.CTkToplevel(self)
        dialog.title("Settings")
        dialog.geometry("360x520")  # Größer für alle Einheiten-Optionen
        dialog.transient(self)
        dialog.grab_set()

        # Font Size
        ctk.CTkLabel(dialog, text="Font Size:", font=ctk.CTkFont(size=13)).pack(pady=(15, 5))

        font_slider = ctk.CTkSlider(dialog, from_=8, to=24, number_of_steps=8,
                                     command=lambda v: self.set_font_size(int(v)))
        font_slider.set(self.font_size)
        font_slider.pack(pady=5, padx=20, fill="x")

        # Separator
        separator = ctk.CTkFrame(dialog, height=2, fg_color=COLORS["bg_frame"])
        separator.pack(fill="x", padx=20, pady=15)

        # Temperature Display Unit
        ctk.CTkLabel(dialog, text="Temperature Display:", font=ctk.CTkFont(size=13)).pack(pady=(5, 10))

        temp_unit_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        temp_unit_frame.pack(padx=20, anchor="w")

        temp_radio_k = ctk.CTkRadioButton(
            temp_unit_frame,
            text="Kelvin (K)",
            variable=self.temp_display_unit,
            value="K",
            font=ctk.CTkFont(size=12),
            fg_color=COLORS["accent"]
        )
        temp_radio_k.pack(side="left", padx=(0, 20))

        temp_radio_c = ctk.CTkRadioButton(
            temp_unit_frame,
            text="Celsius (°C)",
            variable=self.temp_display_unit,
            value="degC",
            font=ctk.CTkFont(size=12),
            fg_color=COLORS["accent"]
        )
        temp_radio_c.pack(side="left")

        ctk.CTkLabel(
            dialog,
            text="Default unit for temperature results\n(can still be changed per variable)",
            font=ctk.CTkFont(size=10),
            text_color=COLORS["text_dim"],
            justify="left"
        ).pack(padx=40, anchor="w", pady=(2, 0))

        # Separator
        separator2 = ctk.CTkFrame(dialog, height=2, fg_color=COLORS["bg_frame"])
        separator2.pack(fill="x", padx=20, pady=15)

        # Pressure Display Unit
        ctk.CTkLabel(dialog, text="Pressure Display:", font=ctk.CTkFont(size=13)).pack(pady=(5, 10))

        pressure_unit_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        pressure_unit_frame.pack(padx=20, anchor="w")

        ctk.CTkRadioButton(
            pressure_unit_frame, text="Pa", variable=self.pressure_display_unit,
            value="Pa", font=ctk.CTkFont(size=12), fg_color=COLORS["accent"]
        ).pack(side="left", padx=(0, 20))

        ctk.CTkRadioButton(
            pressure_unit_frame, text="bar", variable=self.pressure_display_unit,
            value="bar", font=ctk.CTkFont(size=12), fg_color=COLORS["accent"]
        ).pack(side="left")

        # Energy Display Unit
        ctk.CTkLabel(dialog, text="Energy Display:", font=ctk.CTkFont(size=13)).pack(pady=(15, 10))

        energy_unit_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        energy_unit_frame.pack(padx=20, anchor="w")

        ctk.CTkRadioButton(
            energy_unit_frame, text="J (J/kg, J/(kg·K))", variable=self.energy_display_unit,
            value="J", font=ctk.CTkFont(size=12), fg_color=COLORS["accent"]
        ).pack(side="left", padx=(0, 20))

        ctk.CTkRadioButton(
            energy_unit_frame, text="kJ (kJ/kg, kJ/(kg·K))", variable=self.energy_display_unit,
            value="kJ", font=ctk.CTkFont(size=12), fg_color=COLORS["accent"]
        ).pack(side="left")

        # Power Display Unit
        ctk.CTkLabel(dialog, text="Power Display:", font=ctk.CTkFont(size=13)).pack(pady=(15, 10))

        power_unit_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        power_unit_frame.pack(padx=20, anchor="w")

        ctk.CTkRadioButton(
            power_unit_frame, text="W", variable=self.power_display_unit,
            value="W", font=ctk.CTkFont(size=12), fg_color=COLORS["accent"]
        ).pack(side="left", padx=(0, 20))

        ctk.CTkRadioButton(
            power_unit_frame, text="kW", variable=self.power_display_unit,
            value="kW", font=ctk.CTkFont(size=12), fg_color=COLORS["accent"]
        ).pack(side="left")

        # Close Button
        ctk.CTkButton(dialog, text="Close", command=dialog.destroy).pack(pady=20)

    def show_initial_values_dialog(self):
        """Zeigt Dialog für manuelle Startwerte mit Einheiten-Anzeige."""
        if not self.known_variables:
            messagebox.showinfo("Initial Values", "Please run Solve first to detect variables.")
            return

        dialog = ctk.CTkToplevel(self)
        dialog.title("Initial Values")
        dialog.geometry("600x550")
        dialog.transient(self)
        dialog.grab_set()

        # Info
        ctk.CTkLabel(
            dialog,
            text="Set initial values for variables. Units are auto-detected from equations.\n(Leave empty for automatic based on unit, or enter manual value)",
            font=ctk.CTkFont(size=12),
            text_color=COLORS["text_dim"]
        ).pack(pady=10)

        # Header Row
        header_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        header_frame.pack(fill="x", padx=10, pady=(0, 5))
        ctk.CTkLabel(header_frame, text="Variable", width=120, anchor="w",
                    font=ctk.CTkFont(weight="bold")).pack(side="left")
        ctk.CTkLabel(header_frame, text="Initial Value", width=110, anchor="w",
                    font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        ctk.CTkLabel(header_frame, text="Unit", width=100, anchor="w",
                    font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)
        ctk.CTkLabel(header_frame, text="Status", width=80, anchor="w",
                    font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)

        # Scrollable Frame für Variablen
        scroll_frame = ctk.CTkScrollableFrame(dialog, fg_color="transparent")
        scroll_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Sammle alle Variablen (bekannte + ungelöste)
        all_vars = set(self.known_variables)

        # Hole abgeleitete Einheiten (falls vorhanden)
        inferred_units = getattr(self, 'inferred_units', {})

        # Hole gelöste Variablen
        solved_vars = set(self.last_solution.keys()) if self.last_solution else set()

        # Liste der verfügbaren Einheiten für Dropdown
        unit_options = ["", "K", "Pa", "J/kg", "J/(kg*K)", "kg/s", "W", "kg/m^3", "m^3/kg", "m/s", "m/s^2", "N", "kg", "m", "m^2", "m^3", "???"]

        entries = {}
        unit_vars = {}  # Speichere StringVars für Einheiten

        for var in sorted(all_vars):
            row = ctk.CTkFrame(scroll_frame, fg_color="transparent")
            row.pack(fill="x", pady=2)

            # Variable name
            ctk.CTkLabel(row, text=f"{var}:", width=120, anchor="w").pack(side="left")

            # Initial value entry
            entry = ctk.CTkEntry(row, width=100)
            entry.pack(side="left", padx=5)

            # Bestimme Standardwert
            if var in self.manual_initial_values:
                entry.insert(0, str(self.manual_initial_values[var]))
            elif var in inferred_units:
                # Zeige automatischen Startwert basierend auf Einheit
                auto_val = get_initial_from_unit(inferred_units[var])
                entry.insert(0, f"{auto_val:.6g}")
                entry.configure(text_color=COLORS["text_dim"])  # Grau für automatisch

            entries[var] = entry

            # Unit display/dropdown
            unit = inferred_units.get(var, "???")
            # Auch aus current_unit_values holen falls verfügbar
            if unit == "???" and hasattr(self, 'current_unit_values') and var in self.current_unit_values:
                unit = self.current_unit_values[var].calc_unit or "???"

            unit_var = ctk.StringVar(value=unit if unit else "-")
            unit_vars[var] = unit_var

            # Einheit als Dropdown (editierbar)
            unit_combo = ctk.CTkComboBox(row, variable=unit_var, values=unit_options, width=90)
            unit_combo.pack(side="left", padx=5)

            # Status indicator
            if var in solved_vars:
                status_text = "solved"
                status_color = COLORS["success"]
            else:
                status_text = "unsolved"
                status_color = COLORS["warning"]

            ctk.CTkLabel(row, text=status_text, width=80, anchor="w",
                        text_color=status_color).pack(side="left", padx=5)

        # Buttons
        btn_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=10)

        def apply_values():
            self.manual_initial_values.clear()
            # Aktualisiere auch die überschriebenen Einheiten
            for var, entry in entries.items():
                val_str = entry.get().strip()
                if val_str:
                    try:
                        self.manual_initial_values[var] = float(val_str)
                    except ValueError:
                        messagebox.showerror("Error", f"Invalid value for {var}: '{val_str}'")
                        return

                # Speichere überschriebene Einheiten
                if var in unit_vars:
                    new_unit = unit_vars[var].get()
                    if new_unit and new_unit not in ("-", "???"):
                        # Aktualisiere inferred_units für nächsten Solve
                        if not hasattr(self, 'manual_units'):
                            self.manual_units = {}
                        self.manual_units[var] = new_unit

            dialog.destroy()
            self.status_label.configure(text=f"{len(self.manual_initial_values)} initial values set")

        def clear_all():
            for e in entries.values():
                e.delete(0, "end")
                e.configure(text_color=COLORS["text"])

        def auto_fill():
            """Füllt alle leeren Felder mit automatischen Werten basierend auf Einheiten."""
            for var, entry in entries.items():
                if not entry.get().strip():
                    unit = unit_vars[var].get() if var in unit_vars else ""
                    if unit and unit not in ("-", "???"):
                        auto_val = get_initial_from_unit(unit)
                        entry.delete(0, "end")
                        entry.insert(0, f"{auto_val:.6g}")
                        entry.configure(text_color=COLORS["text_dim"])

        ctk.CTkButton(btn_frame, text="Clear All", command=clear_all).pack(side="left")
        ctk.CTkButton(btn_frame, text="Auto-Fill", command=auto_fill,
                     fg_color=COLORS["accent"]).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="Cancel", command=dialog.destroy).pack(side="right", padx=5)
        ctk.CTkButton(btn_frame, text="OK", command=apply_values).pack(side="right")

    def show_plot_dialog(self):
        """Zeigt Plot-Dialog."""
        if not MATPLOTLIB_AVAILABLE:
            messagebox.showerror("Error", "matplotlib not available")
            return

        if self.last_solution is None:
            messagebox.showinfo("Plot", "Please run Solve first.")
            return

        has_arrays = any(isinstance(v, np.ndarray) for v in self.last_solution.values())
        if not has_arrays:
            messagebox.showinfo("Plot", "Plot requires parametric study with vector data.")
            return

        array_vars = sorted([k for k, v in self.last_solution.items() if isinstance(v, np.ndarray)])

        dialog = ctk.CTkToplevel(self)
        dialog.title("New Plot")
        dialog.geometry("400x400")
        dialog.transient(self)
        dialog.grab_set()

        # X-Achse
        ctk.CTkLabel(dialog, text="X-Axis:", font=ctk.CTkFont(size=13)).pack(pady=(20, 5))
        x_var = ctk.StringVar(value=array_vars[0] if array_vars else "")
        x_combo = ctk.CTkComboBox(dialog, variable=x_var, values=array_vars, width=200)
        x_combo.pack()

        # Y-Achse
        ctk.CTkLabel(dialog, text="Y-Axis:", font=ctk.CTkFont(size=13)).pack(pady=(20, 5))
        y_var = ctk.StringVar(value=array_vars[1] if len(array_vars) > 1 else array_vars[0] if array_vars else "")
        y_combo = ctk.CTkComboBox(dialog, variable=y_var, values=array_vars, width=200)
        y_combo.pack()

        # Grid Option
        grid_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(dialog, text="Show grid", variable=grid_var).pack(pady=20)

        def create_plot():
            x_name, y_name = x_var.get(), y_var.get()
            if not x_name or not y_name:
                return

            x_data = self.last_solution[x_name]
            y_data = self.last_solution[y_name]

            self._create_plot_window(x_data, [(y_name, y_data)], x_name, y_name,
                                      f"{y_name} vs {x_name}", grid_var.get(), False, False)
            dialog.destroy()

        ctk.CTkButton(dialog, text="Plot", command=create_plot).pack(pady=20)

    def show_quick_plot_dialog(self):
        """Vereinfachter Plot-Dialog."""
        self.show_plot_dialog()

    def _create_plot_window(self, x_data, y_data_list, x_label="", y_label="", title="",
                            show_grid=True, show_legend=True, show_markers=False):
        """Erstellt ein Plot-Fenster."""
        if not MATPLOTLIB_AVAILABLE:
            return

        plot_window = ctk.CTkToplevel(self)
        plot_window.title(f"Plot: {title}" if title else "Plot")
        plot_window.geometry("800x600")

        fig = Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        marker = 'o' if show_markers else None

        for i, (name, y_data) in enumerate(y_data_list):
            ax.plot(x_data, y_data, label=name, color=colors[i % len(colors)], marker=marker, markersize=4)

        if x_label:
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_ylabel(y_label)
        if title:
            ax.set_title(title)
        if show_grid:
            ax.grid(True, linestyle='--', alpha=0.7)
        if show_legend and len(y_data_list) > 1:
            ax.legend()

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()

        # Toolbar
        toolbar_frame = ctk.CTkFrame(plot_window, fg_color="transparent")
        toolbar_frame.pack(side="top", fill="x")
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()

        canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    def show_function_help(self):
        """Zeigt Funktions-Hilfe."""
        dialog = ctk.CTkToplevel(self)
        dialog.title("Function Reference")
        dialog.geometry("700x750")

        text = ctk.CTkTextbox(dialog, font=ctk.CTkFont(family="Courier", size=11))
        text.pack(fill="both", expand=True, padx=10, pady=10)

        help_text = """=== HVAC EQUATION SOLVER - FUNCTION REFERENCE ===

MATHEMATICAL FUNCTIONS:
-----------------------
sin(x), cos(x), tan(x)     Trigonometric (x in degrees)
asin(x), acos(x), atan(x)  Inverse trig functions
sinh(x), cosh(x), tanh(x)  Hyperbolic functions
exp(x)                      e^x
ln(x)                       Natural logarithm
log10(x)                    Base 10 logarithm
sqrt(x)                     Square root
abs(x)                      Absolute value
pi                          Pi constant

THERMODYNAMIC FUNCTIONS (CoolProp):
-----------------------------------
Syntax: function(fluid, param1=value1, param2=value2)

Properties:
  enthalpy(...)      Specific enthalpy [kJ/kg]
  entropy(...)       Specific entropy [kJ/(kg K)]
  density(...)       Density [kg/m3]
  temperature(...)   Temperature [K]
  pressure(...)      Pressure [bar]
  quality(...)       Vapor quality [-]

State properties (2 required):
  T = Temperature [K] (internally, use 373.15K or 100°C)
  p = Pressure [bar]
  h = Enthalpy [kJ/kg]
  s = Entropy [kJ/(kg K)]
  x = Vapor quality [-]

Examples:
  h = enthalpy(water, T=373.15K, p=1)    {100°C}
  h = enthalpy(water, T=100°C, p=1)      {also valid}
  rho = density(R134a, T=298.15K, x=1)   {25°C}

HUMID AIR FUNCTIONS:
--------------------
Syntax: HumidAir(property, T=..., rh=..., p_tot=...)
Temperature T in [K] (or use °C with unit)

  h = HumidAir(h, T=298.15K, rh=0.5, p_tot=1)   {25°C}
  h = HumidAir(h, T=25°C, rh=0.5, p_tot=1)      {also valid}
  w = HumidAir(w, T=303.15K, rh=0.6, p_tot=1)   {30°C}
  T_dp = HumidAir(T_dp, T=298.15K, w=0.01, p_tot=1)

RADIATION FUNCTIONS (Blackbody):
--------------------------------
All functions: T in [K], wavelength in [um]

  Eb(T, lambda)              Spectral emissive power [W/(m2*um)]
  Blackbody(T, l1, l2)       Fraction of energy in wavelength range [-]
  Blackbody_cumulative(T, l) Cumulative fraction from 0 to l [-]
  Wien(T)                    Wavelength of max emission [um]
  Stefan_Boltzmann(T)        Total emissive power [W/m2]

Examples:
  E = Eb(573.15K, 5)                  {Spectral power at 300°C, 5um}
  E = Eb(300°C, 5)                    {also valid}
  f = Blackbody(1273.15K, 0.4, 0.7)   {Visible light fraction at 1000°C}
  lambda_max = Wien(773.15K)          {Peak wavelength at 500°C}
  E_total = Stefan_Boltzmann(373.15K) {Total emission at 100°C}

RESERVED VARIABLE NAMES (DO NOT USE):
-------------------------------------
Python keywords (cause syntax errors):
  lambda, if, else, for, while, class, def, return,
  import, from, as, try, except, with, pass, break,
  continue, and, or, not, in, is, True, False, None

Mathematical constants/functions (will be overwritten):
  pi, e, sin, cos, tan, exp, ln, sqrt, abs, max, min

Thermodynamic functions (case-insensitive):
  enthalpy, entropy, density, temperature, pressure, etc.

TIPS:
  - Use descriptive names: lambda_1 instead of lambda
  - Use subscripts: T_1, p_2, h_in, h_out
  - For wavelength: use 'L', 'wl', or 'lambda_1'
  - Euler's number: use exp(1) instead of e

PARAMETRIC STUDIES (Sweeps):
----------------------------
Syntax: variable = start:step:end [unit]

Examples:
  T = 20:5:40           {20, 25, 30, 35, 40}
  T = 20:5:40 °C        {with unit}
  p = 1:0.5:3 bar       {1, 1.5, 2, 2.5, 3 bar}

After solving: Use Plot menu for visualization
"""
        text.insert("1.0", help_text)
        text.configure(state="disabled")

    def show_fluid_help(self):
        """Zeigt Fluid-Liste."""
        dialog = ctk.CTkToplevel(self)
        dialog.title("Available Fluids")
        dialog.geometry("400x500")

        text = ctk.CTkTextbox(dialog, font=ctk.CTkFont(family="Courier", size=11))
        text.pack(fill="both", expand=True, padx=10, pady=10)

        help_text = """=== AVAILABLE FLUIDS ===

WATER / STEAM:
  Water, water, steam, h2o

AIR:
  Air, air

REFRIGERANTS (HFCs):
  R134a, R32, R410A, R407C

REFRIGERANTS (HFOs):
  R1234yf, R1234ze(E)

NATURAL REFRIGERANTS:
  R717 / Ammonia (ammonia, nh3)
  R744 / CO2 (co2)
  R290 / Propane (propane)

GASES:
  Nitrogen (n2)
  Oxygen (o2)
  Hydrogen (h2)
  Helium (he)
  Argon (ar)
  Methane (ch4)
"""
        text.insert("1.0", help_text)
        text.configure(state="disabled")

    def _insert_example(self):
        """Fügt ein Beispiel ein."""
        if THERMO_AVAILABLE:
            example = '''"HVAC Equation Solver - Example"
"Internal units: T[K], p[bar], h[kJ/kg]"

{--- Example 1: Water/Steam ---}
T_1 = 150 °C
p_1 = 5 bar
h_1 = enthalpy(water, T=T_1, p=p_1)
s_1 = entropy(water, T=T_1, p=p_1)

{Saturated steam at same pressure}
x_sat = 1
T_sat = temperature(water, p=p_1, x=x_sat)
h_sat = enthalpy(water, p=p_1, x=x_sat)

{--- Example 2: Humid Air ---}
T_air = 25 °C
rh = 0.6
p_tot = 1 bar

h_air = HumidAir(h, T=T_air, rh=rh, p_tot=p_tot)
w = HumidAir(w, T=T_air, rh=rh, p_tot=p_tot)
T_dp = HumidAir(T_dp, T=T_air, rh=rh, p_tot=p_tot)
T_wb = HumidAir(T_wb, T=T_air, rh=rh, p_tot=p_tot)

{--- Example 3: Thermal Radiation ---}
T_surface = 500 °C
epsilon = 0.85
A = 2 m^2
sigma = 5.67E-8 W/(m^2*K^4)

{Stefan-Boltzmann radiation}
Q_rad = epsilon * sigma * A * T_surface^4

{Peak wavelength (Wien's law)}
lambda_max = Wien(T_surface)

"Press F5 to solve"
'''
        else:
            example = '''"Example: Nonlinear equation system"
"Right triangle calculation"

{Given values}
a = 3
b = 4

{Pythagorean theorem}
c^2 = a^2 + b^2

{Calculate angles}
tan(alpha) = a / b
alpha + beta = 90

"Press F5 to solve"
'''
        self.equations_text.delete("1.0", "end")
        self.equations_text.insert("1.0", example)
        self.clear_results()


def main():
    """Hauptfunktion."""
    app = EquationSolverApp()
    app.mainloop()


if __name__ == "__main__":
    main()
