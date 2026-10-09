#!/bin/bash
# HVAC Equation Solver - Start (macOS, Linux). Läuft ohne Internet.
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
    echo "Das Programm ist noch nicht installiert."
    echo "Bitte zuerst installieren: macOS installieren_mac.command, Linux ./install.sh"
    read -r -p "Enter drücken zum Schließen ... " _
    exit 1
fi
exec .venv/bin/python main.py
