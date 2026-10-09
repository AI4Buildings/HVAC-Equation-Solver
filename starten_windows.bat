@echo off
rem HVAC Equation Solver - Start (Windows). Laeuft ohne Internet.
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" goto :not_installed
start "" ".venv\Scripts\pythonw.exe" main.py
exit /b 0

:not_installed
echo Das Programm ist noch nicht installiert.
echo Bitte zuerst installieren_windows.bat per Doppelklick ausfuehren.
pause
exit /b 1
