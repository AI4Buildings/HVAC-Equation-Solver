# HVAC Equation Solver – Installation und Update

Der HVAC Equation Solver läuft **lokal auf dem eigenen Rechner, ohne Internet**.
Eine Internetverbindung braucht man nur **einmal für die Installation** und später
für **Updates**.

Unterstützt: Windows 10/11, macOS, Linux.

---

## 1. Python installieren (einmalig)

Benötigt wird **Python 3.12** (3.10 bis 3.13 funktionieren ebenfalls).
Download: <https://www.python.org/downloads/> → dort eine Version **3.12.x** wählen.

**Windows**
1. Den „Windows installer (64-bit)“ herunterladen und starten.
2. **Wichtig:** Im ersten Fenster unten **„Add python.exe to PATH“** anhaken.
3. „Install Now“ klicken.

**macOS**
1. Den „macOS 64-bit universal2 installer“ herunterladen und durchklicken.
2. Wer Homebrew statt des python.org-Installers verwendet, braucht zusätzlich:
   `brew install python@3.12 python-tk@3.12`

**Linux (Ubuntu/Debian)**
```bash
sudo apt install python3 python3-venv python3-tk
```

---

## 2. Programm herunterladen

**Variante A – ohne Git (empfohlen):**
1. Auf der GitHub-Seite des Projekts rechts auf **„Releases“** klicken.
2. Bei der neuesten Version unter „Assets“ **„Source code (zip)“** herunterladen.
3. Die ZIP-Datei entpacken und den Ordner an einen festen Ort legen, z. B.
   - Windows: `C:\Users\<Name>\HVAC-Equation-Solver`
   - macOS/Linux: `~/HVAC-Equation-Solver`

> **Tipp:** Den Programmordner **nicht** in einen OneDrive- oder iCloud-Ordner legen
> (unter Windows ist „Dokumente“ oft mit OneDrive synchronisiert). Die Installation
> legt dort einige hundert MB an, deren Synchronisation stört.

**Variante B – mit Git:**
```bash
git clone https://github.com/AI4Buildings/HVAC-Equation-Solver.git
```

---

## 3. Installieren (einmalig, Internet nötig)

Die Installation legt im Programmordner eine eigene Python-Umgebung (Ordner `.venv`)
an und lädt die benötigten Pakete. Am Rechner wird sonst nichts verändert.
Dauer: etwa 1–5 Minuten. Am Ende erscheint **„Installation erfolgreich“**.

| System | So geht's |
|---|---|
| **Windows** | Doppelklick auf **`installieren_windows.bat`** |
| **macOS** | Doppelklick auf **`installieren_mac.command`** |
| **Linux** | Im Terminal im Programmordner: `./install.sh` |

**Windows blockiert die Datei?** Erscheint „Der Computer wurde durch Windows geschützt“:
„Weitere Informationen“ → „Trotzdem ausführen“.

**macOS blockiert die Datei?** Erscheint „… kann nicht geöffnet werden“:
- macOS 14 und älter: Rechtsklick (ctrl-Klick) auf die Datei → „Öffnen“ → „Öffnen“.
- macOS 15 und neuer: *Systemeinstellungen → Datenschutz & Sicherheit* → ganz unten
  „Dennoch öffnen“.
- Alternativ im Terminal: `bash ` tippen (mit Leerzeichen), die Datei
  `installieren_mac.command` ins Terminal-Fenster ziehen, Enter drücken.

---

## 4. Starten (ohne Internet)

| System | So geht's |
|---|---|
| **Windows** | Doppelklick auf **`starten_windows.bat`** |
| **macOS** | Doppelklick auf **`starten_mac.command`** (das Terminal-Fenster bleibt offen, solange das Programm läuft – nicht schließen) |
| **Linux** | `./start.sh` |

**Verknüpfung auf dem Schreibtisch:**
- Windows: Rechtsklick auf `starten_windows.bat` → „Senden an“ → „Desktop (Verknüpfung erstellen)“.
- macOS: Rechtsklick auf `starten_mac.command` → „Alias erzeugen“ → Alias auf den Schreibtisch ziehen.

**Welche Version habe ich?** Steht in der Titelleiste und unten rechts in der
Statusleiste („Version 4.0.0“).

---

## 5. Eigene Dateien

Eigene Gleichungsdateien (`.hes`) **außerhalb des Programmordners** speichern,
z. B. in `Dokumente/Thermodynamik`. Bei einem Update wird der Programmordner ersetzt.

---

## 6. Update auf eine neue Version

Die aktuelle Version steht auf GitHub unter **„Releases“**. Ein Update braucht Internet.

**Variante A – ohne Git:**
1. Programm schließen.
2. Neue Version herunterladen und entpacken (wie in Schritt 2).
3. Alten Programmordner löschen (vorher prüfen, dass keine eigenen Dateien darin
   liegen) und den neuen Ordner an dieselbe Stelle legen.
4. **Installation erneut ausführen** (Schritt 3).

**Variante B – mit Git:**
```bash
cd HVAC-Equation-Solver
git pull
```
Danach **Installation erneut ausführen** (Schritt 3). Sind die Pakete unverändert,
dauert das nur wenige Sekunden.

---

## 7. Probleme und Lösungen

| Meldung / Problem | Lösung |
|---|---|
| „Kein Python 3.10 oder neuer gefunden“ | Python installieren (Schritt 1). Unter Windows „Add python.exe to PATH“ nicht vergessen; danach die Installation erneut starten. |
| „In diesem Python fehlt tkinter“ | Python von python.org verwenden (macOS/Windows) bzw. `sudo apt install python3-tk` (Linux). |
| „Pakete konnten nicht installiert werden“ | Internetverbindung prüfen und erneut installieren. Hilft das nicht: den Ordner `.venv` im Programmordner löschen und neu installieren (versteckter Ordner: macOS-Finder `Cmd+Shift+.`, Windows-Explorer „Ansicht → Ausgeblendete Elemente“). |
| Windows: Doppelklick auf `starten_windows.bat`, aber nichts passiert | Installation erneut ausführen. Fehlermeldung sichtbar machen: Eingabeaufforderung im Programmordner öffnen und `.venv\Scripts\python.exe main.py` eingeben. |
| „Das Programm ist noch nicht installiert“ | Zuerst Schritt 3 ausführen. |

**Deinstallieren:** Einfach den Programmordner löschen. Python selbst kann über die
Systemsteuerung (Windows) bzw. den Programme-Ordner (macOS) entfernt werden.
