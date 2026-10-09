#!/bin/bash
# HVAC Equation Solver - Installation / Update der Python-Umgebung (macOS, Linux)
#
# Legt im Programmordner eine eigene Python-Umgebung (.venv) an und installiert
# die Pakete aus requirements.txt. Internet ist nur hierfür nötig - danach
# läuft das Programm offline. Erneut ausführen nach jedem Update.

cd "$(dirname "$0")" || exit 1

pause_exit() {
    echo
    read -r -p "Enter drücken zum Schließen ... " _
    exit "$1"
}

echo "=============================================="
echo " HVAC Equation Solver $(sed -n 's/^__version__ = "\(.*\)"/\1/p' version.py) - Installation"
echo "=============================================="

# Python suchen: getestete Versionen zuerst (python.org-Installer bevorzugt)
PY=""
for candidate in \
        /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
        /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 \
        python3.12 python3.13 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1 &&
       "$candidate" -c 'import sys; sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)' 2>/dev/null; then
        PY="$candidate"
        break
    fi
done

if [ -z "$PY" ]; then
    echo "FEHLER: Kein Python 3.10 oder neuer gefunden (empfohlen: Python 3.12)."
    echo "  macOS: Installer von https://www.python.org/downloads/ ausführen."
    echo "  Linux: z.B. 'sudo apt install python3 python3-venv python3-tk'"
    pause_exit 1
fi
echo "Verwende: $PY ($("$PY" --version 2>&1))"
"$PY" -c 'import sys; sys.exit(0 if sys.version_info[:2] <= (3, 13) else 1)' ||
    echo "Hinweis: Diese Python-Version ist nicht getestet (getestet: 3.12)."

# tkinter (GUI) muss in Python enthalten sein
if ! "$PY" -c 'import tkinter; tkinter.Tcl()' 2>/dev/null; then
    echo "FEHLER: In diesem Python fehlt tkinter (wird für die Oberfläche benötigt)."
    echo "  macOS: Python von python.org verwenden (Homebrew: 'brew install python-tk@3.12')."
    echo "  Linux: 'sudo apt install python3-tk'"
    pause_exit 1
fi

# Eigene Umgebung im Programmordner (nichts wird systemweit installiert)
if [ ! -x .venv/bin/python ]; then
    echo "Lege Python-Umgebung .venv an ..."
    if ! "$PY" -m venv .venv; then
        echo "FEHLER: Python-Umgebung konnte nicht angelegt werden."
        echo "  Linux: 'sudo apt install python3-venv'"
        pause_exit 1
    fi
fi

echo "Installiere Pakete (Internet nötig, einige Minuten) ..."
.venv/bin/python -m pip install --upgrade pip --quiet
if ! .venv/bin/python -m pip install -r requirements.txt; then
    echo "FEHLER: Pakete konnten nicht installiert werden (Internetverbindung prüfen)."
    echo "  Bei wiederholten Problemen den Ordner .venv löschen und erneut installieren."
    pause_exit 1
fi

if ! .venv/bin/python -c 'import numpy, scipy, CoolProp, matplotlib, pint, customtkinter, tkinter'; then
    echo "FEHLER: Prüfung der Pakete fehlgeschlagen."
    pause_exit 1
fi

echo
echo "Installation erfolgreich. Starten mit:"
echo "  macOS: Doppelklick auf starten_mac.command"
echo "  Linux: ./start.sh"
pause_exit 0
