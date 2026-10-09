@echo off
rem HVAC Equation Solver - Installation / Update der Python-Umgebung (Windows)
rem Legt im Programmordner eine eigene Python-Umgebung (.venv) an und installiert
rem die Pakete aus requirements.txt. Internet ist nur hierfuer noetig - danach
rem laeuft das Programm offline. Nach jedem Update erneut ausfuehren.
setlocal
cd /d "%~dp0"
echo ==============================================
echo  HVAC Equation Solver - Installation
echo ==============================================

rem Python suchen: Python-Launcher "py" mit getesteten Versionen, sonst "python"
set "PY="
for %%V in (3.12 3.13 3.11 3.10) do (
    if not defined PY (
        py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V"
    )
)
if not defined PY (
    python -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY goto :no_python
echo Verwende: %PY%

%PY% -c "import tkinter; tkinter.Tcl()" >nul 2>&1
if errorlevel 1 goto :no_tkinter

if exist ".venv\Scripts\python.exe" goto :install_packages
echo Lege Python-Umgebung .venv an ...
%PY% -m venv .venv
if errorlevel 1 goto :no_venv

:install_packages
echo Installiere Pakete - Internet noetig, einige Minuten ...
".venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :pip_failed
".venv\Scripts\python.exe" -c "import numpy, scipy, CoolProp, matplotlib, pint, customtkinter, tkinter"
if errorlevel 1 goto :check_failed

echo.
echo Installation erfolgreich. Starten mit Doppelklick auf starten_windows.bat
pause
exit /b 0

:no_python
echo FEHLER: Kein Python 3.10 oder neuer gefunden - empfohlen: Python 3.12.
echo Bitte von https://www.python.org/downloads/ installieren und dabei
echo "Add python.exe to PATH" anhaken. Danach diese Datei erneut starten.
pause
exit /b 1

:no_tkinter
echo FEHLER: In diesem Python fehlt tkinter - wird fuer die Oberflaeche benoetigt.
echo Bitte Python von https://www.python.org/downloads/ installieren, Option "tcl/tk" aktiviert lassen.
pause
exit /b 1

:no_venv
echo FEHLER: Python-Umgebung .venv konnte nicht angelegt werden.
pause
exit /b 1

:pip_failed
echo FEHLER: Pakete konnten nicht installiert werden - Internetverbindung pruefen.
echo Bei wiederholten Problemen den Ordner .venv loeschen und erneut installieren.
pause
exit /b 1

:check_failed
echo FEHLER: Pruefung der Pakete fehlgeschlagen.
pause
exit /b 1
