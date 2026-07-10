"""
Strahlungs-Modul für den HVAC Equation Solver

Implementiert Schwarzkörper-Strahlungsfunktionen basierend auf dem Planckschen Strahlungsgesetz.

Funktionen:
- Eb(T, lambda): Spektrale Emissionsleistung [W/(m²·µm)]
- Blackbody(T, lambda1, lambda2): Anteil der Strahlungsenergie im Wellenlängenbereich [-]

Einheiten:
- Temperatur T: K (Kelvin, wird bereits in K erwartet)
- Wellenlänge lambda: µm (Mikrometer)
- Eb: W/(m²·µm)
- Blackbody: dimensionslos (0-1)

Konstanten (CODATA, konsistent zu σ = 5.670374419e-8):
- C1 = 2·π·h·c² = 3.741771852e8 W·µm⁴/m²
- C2 = h·c/k = 1.4387768775e4 µm·K
- σ (Stefan-Boltzmann) = 5.670374419e-8 W/(m²·K⁴)

Alle Funktionen unterstützen sowohl skalare Werte als auch numpy-Arrays (vektorisiert).
"""

import math
import numpy as np
from scipy import integrate

# Physikalische Konstanten (CODATA, konsistent zur Normierung mit SIGMA)
C1 = 3.741771852e8       # Erste Strahlungskonstante [W·µm⁴/m²]
C2 = 1.4387768775e4      # Zweite Strahlungskonstante [µm·K]
SIGMA = 5.670374419e-8  # Stefan-Boltzmann Konstante [W/(m²·K⁴)]


def _ensure_kelvin(T):
    """
    Stellt sicher dass T ein numpy Array in Kelvin ist.

    Temperatur wird intern bereits in Kelvin übergeben (seit v3.1).
    Diese Funktion konvertiert nur zu numpy Array für vektorisierte Berechnung.
    """
    return np.asarray(T)


def _normalize_wavelength(wavelength):
    """
    Normalisiert Wellenlänge zu µm.

    Wenn der Wert sehr klein ist (< 0.0001), wird angenommen dass er in Metern
    angegeben ist und zu µm konvertiert (Faktor 1e6).

    Typische Wellenlängen für Wärmestrahlung: 0.1-100 µm
    """
    wavelength = np.asarray(wavelength)

    # Schwelle: 0.0001 µm = 0.1 nm (extrem kurz, unwahrscheinlich)
    # Wenn wavelength < 0.0001, ist es wahrscheinlich in Metern
    threshold = 0.0001

    if wavelength.ndim == 0:
        # Skalar
        if wavelength < threshold:
            return wavelength * 1e6  # m -> µm
        return wavelength
    else:
        # Array
        return np.where(wavelength < threshold, wavelength * 1e6, wavelength)


def Eb(T, wavelength):
    """
    Berechnet die spektrale (monochromatische) Emissionsleistung eines Schwarzkörpers.

    Basiert auf dem Planckschen Strahlungsgesetz:
    Eb_λ = C1 / (λ⁵ · (exp(C2/(λ·T)) - 1))

    Args:
        T: Temperatur in K (Skalar oder Array)
        wavelength: Wellenlänge in µm oder m (automatische Erkennung)
                   Werte < 0.0001 werden als Meter interpretiert

    Returns:
        Spektrale Emissionsleistung in W/(m²·µm)

    Beispiel:
        >>> Eb(1273.15, 3.0)  # Bei 1273.15 K (= 1000°C) und 3 µm
        36446.4...
        >>> Eb(1273.15, 3e-6)  # Bei 1273.15 K und 3e-6 m = 3 µm (gleich!)
        36446.4...
    """
    wavelength = _normalize_wavelength(wavelength)  # Auto-Konvertierung m -> µm
    T_kelvin = _ensure_kelvin(T)

    # Validierung (Skalare und Arrays)
    if np.any(T_kelvin <= 0):
        raise ValueError(f"Temperatur muss > 0 K sein (gegeben: {T} K)")
    if np.any(wavelength <= 0):
        raise ValueError(f"Wellenlänge muss > 0 sein (gegeben: {wavelength} µm)")

    # Plancksches Strahlungsgesetz
    # Eb = C1 / (λ^5 * (exp(C2/(λ*T)) - 1))
    exponent = C2 / (wavelength * T_kelvin)

    # Overflow-Schutz: setze große Exponenten auf 0
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        eb = np.where(
            exponent > 700,
            0.0,
            C1 / (wavelength**5 * (np.exp(exponent) - 1))
        )

    # Rückgabe als Skalar wenn Eingabe skalar war
    if eb.ndim == 0:
        return float(eb)
    return eb


def _blackbody_integrand(wavelength: float, T_kelvin: float) -> float:
    """Integrand für die Blackbody-Funktion (normiert auf σT⁴)."""
    if wavelength <= 0:
        return 0.0

    exponent = C2 / (wavelength * T_kelvin)

    if exponent > 700:
        return 0.0

    try:
        return C1 / (wavelength**5 * (math.exp(exponent) - 1))
    except OverflowError:
        return 0.0


def _peak_points(T_kelvin, lambda_lo, lambda_hi):
    """
    Stützpunkte um den Wien-Peak für integrate.quad.

    Bei sehr breiten Integrationsintervallen (z.B. 1e-6 bis 10000 µm) kann
    quad den schmalen Planck-Peak sonst übersehen. Gibt None zurück, wenn
    kein Stützpunkt im Intervall liegt.
    """
    lambda_peak = 2897.8 / T_kelvin  # Wien'sches Verschiebungsgesetz [µm]
    candidates = (0.2 * lambda_peak, 0.5 * lambda_peak, lambda_peak,
                  2 * lambda_peak, 5 * lambda_peak, 10 * lambda_peak,
                  50 * lambda_peak)
    points = [p for p in candidates if lambda_lo < p < lambda_hi]
    return points or None


def _blackbody_single(T_kelvin, lambda1, lambda2):
    """Berechnet Blackbody-Fraktion für einen einzelnen Temperaturwert."""
    # Bei lambda1 = 0: verwende sehr kleine untere Grenze (Singularität bei 0)
    lambda1_eff = max(lambda1, 1e-6)

    # Praktische Obergrenze für Integration:
    # Bei sehr großen Wellenlängen ist praktisch keine Strahlungsenergie mehr vorhanden.
    # Wien'sches Verschiebungsgesetz: λ_max ≈ 2898/T µm
    # Bei 100 * λ_max ist >99.9999% der Strahlung erfasst
    # Praktische Obergrenze: 10000 µm (10 mm) reicht für alle Temperaturen > -173°C
    lambda_max_practical = max(10000.0, 100 * 2898.0 / T_kelvin)
    lambda2_eff = min(lambda2, lambda_max_practical)

    # Gesamte emittierte Leistung nach Stefan-Boltzmann
    total_power = SIGMA * T_kelvin**4

    # Integriere Eb über den Wellenlängenbereich.
    # Stützpunkte um den Wien-Peak, damit quad den schmalen Peak
    # auch bei sehr breiten Intervallen nicht übersieht.
    result, error = integrate.quad(
        _blackbody_integrand,
        lambda1_eff,
        lambda2_eff,
        args=(T_kelvin,),
        limit=200,
        points=_peak_points(T_kelvin, lambda1_eff, lambda2_eff)
    )

    # Normiere auf Gesamtleistung
    fraction = result / total_power

    # Begrenze auf [0, 1]
    return max(0.0, min(1.0, fraction))


def Blackbody(T, lambda1, lambda2):
    """
    Berechnet den Anteil der Schwarzkörperstrahlung im Wellenlängenbereich [λ1, λ2].

    Der Rückgabewert ist der Bruchteil der gesamten emittierten Energie,
    der im angegebenen Wellenlängenbereich liegt:

    F(λ1→λ2) = ∫[λ1,λ2] Eb_λ dλ / (σ·T⁴)

    Args:
        T: Temperatur in K (Skalar oder Array)
        lambda1: Untere Wellenlänge in µm oder m (automatische Erkennung)
        lambda2: Obere Wellenlänge in µm oder m (automatische Erkennung)

    Returns:
        Anteil der Strahlung im Bereich (dimensionslos, 0-1)

    Beispiel:
        >>> Blackbody(5773.15, 0.4, 0.7)  # Sichtbares Licht bei Sonnentemperatur (5773 K)
        0.367...
    """
    T_kelvin = _ensure_kelvin(T)
    lambda1 = _normalize_wavelength(lambda1)  # Auto-Konvertierung m -> µm
    lambda2 = _normalize_wavelength(lambda2)  # Auto-Konvertierung m -> µm

    # Validierung (Skalare und Arrays)
    if np.any(T_kelvin <= 0):
        raise ValueError(f"Temperatur muss > 0 K sein (gegeben: {T} K)")
    if np.any(lambda1 < 0) or np.any(lambda2 <= 0):
        raise ValueError(f"Wellenlängen müssen >= 0 sein")
    if np.any(lambda1 >= lambda2):
        raise ValueError(f"lambda1 ({lambda1}) muss kleiner als lambda2 ({lambda2}) sein")

    # Skalarer Fall (Verhalten unverändert)
    if T_kelvin.ndim == 0 and lambda1.ndim == 0 and lambda2.ndim == 0:
        return _blackbody_single(float(T_kelvin), float(lambda1), float(lambda2))

    # Vektorisierter Fall: mindestens eine Eingabe ist ein Array.
    # np.vectorize broadcastet T, lambda1 und lambda2 gegeneinander.
    vectorized = np.vectorize(_blackbody_single, otypes=[float])
    return vectorized(T_kelvin, lambda1, lambda2)


def _blackbody_cumulative_single(T_kelvin, wavelength):
    """Berechnet kumulative Blackbody-Fraktion für einen einzelnen Wert."""
    # Verwende sehr kleine untere Grenze statt 0 (Singularität)
    lambda_min = 1e-6

    # Praktische Obergrenze für Integration (wie in _blackbody_single):
    # Bei 100 * λ_max (Wien) ist >99.9999% der Strahlung erfasst
    lambda_max_practical = max(10000.0, 100 * 2898.0 / T_kelvin)
    wavelength_eff = min(wavelength, lambda_max_practical)

    total_power = SIGMA * T_kelvin**4

    result, error = integrate.quad(
        _blackbody_integrand,
        lambda_min,
        wavelength_eff,
        args=(T_kelvin,),
        limit=200,
        points=_peak_points(T_kelvin, lambda_min, wavelength_eff)
    )

    fraction = result / total_power
    return max(0.0, min(1.0, fraction))


def Blackbody_cumulative(T, wavelength):
    """
    Berechnet den kumulativen Anteil der Schwarzkörperstrahlung von 0 bis λ.

    F(0→λ) = ∫[0,λ] Eb_λ dλ / (σ·T⁴)

    Args:
        T: Temperatur in K (Skalar oder Array)
        wavelength: Obere Wellenlänge in µm oder m (automatische Erkennung)

    Returns:
        Kumulativer Anteil der Strahlung (dimensionslos, 0-1)
    """
    wavelength = _normalize_wavelength(wavelength)  # Auto-Konvertierung m -> µm
    T_kelvin = _ensure_kelvin(T)

    # Validierung (Skalare und Arrays)
    if np.any(T_kelvin <= 0):
        raise ValueError(f"Temperatur muss > 0 K sein")
    if np.any(wavelength <= 0):
        raise ValueError(f"Wellenlänge muss > 0 sein")

    # Vektorisierte Berechnung
    is_T_scalar = T_kelvin.ndim == 0
    is_wl_scalar = wavelength.ndim == 0

    if is_T_scalar and is_wl_scalar:
        return _blackbody_cumulative_single(float(T_kelvin), float(wavelength))
    elif is_T_scalar:
        # T skalar, wavelength Array
        return np.array([_blackbody_cumulative_single(float(T_kelvin), wl) for wl in wavelength])
    elif is_wl_scalar:
        # T Array, wavelength skalar
        return np.array([_blackbody_cumulative_single(tk, float(wavelength)) for tk in T_kelvin])
    else:
        # Beide Arrays (müssen gleiche Länge haben)
        return np.array([_blackbody_cumulative_single(tk, wl) for tk, wl in zip(T_kelvin, wavelength)])


def Wien_displacement(T):
    """
    Berechnet die Wellenlänge maximaler Emission nach dem Wienschen Verschiebungsgesetz.

    λ_max = 2898 µm·K / T

    Args:
        T: Temperatur in K (Skalar oder Array)

    Returns:
        Wellenlänge maximaler Emission in µm
    """
    T_kelvin = _ensure_kelvin(T)

    # Validierung (Skalare und Arrays)
    if np.any(T_kelvin <= 0):
        raise ValueError(f"Temperatur muss > 0 K sein")

    # Wiensche Verschiebungskonstante
    b = 2897.8  # µm·K

    result = b / T_kelvin

    # Rückgabe als Skalar wenn Eingabe skalar war
    if result.ndim == 0:
        return float(result)
    return result


def Stefan_Boltzmann(T):
    """
    Berechnet die gesamte emittierte Strahlungsleistung eines Schwarzkörpers.

    E = σ·T⁴

    Args:
        T: Temperatur in K (Skalar oder Array)

    Returns:
        Gesamte Emissionsleistung in W/m²
    """
    T_kelvin = _ensure_kelvin(T)

    # Validierung (Skalare und Arrays)
    if np.any(T_kelvin <= 0):
        raise ValueError(f"Temperatur muss > 0 K sein")

    result = SIGMA * T_kelvin**4

    # Rückgabe als Skalar wenn Eingabe skalar war
    if result.ndim == 0:
        return float(result)
    return result


# Dictionary aller Strahlungs-Funktionen für den Solver
# Sowohl Groß- als auch Kleinschreibung unterstützen
RADIATION_FUNCTIONS = {
    'Eb': Eb,
    'eb': Eb,
    'Blackbody': Blackbody,
    'blackbody': Blackbody,
    'Blackbody_cumulative': Blackbody_cumulative,
    'blackbody_cumulative': Blackbody_cumulative,
    'Wien': Wien_displacement,
    'wien': Wien_displacement,
    'Stefan_Boltzmann': Stefan_Boltzmann,
    'stefan_boltzmann': Stefan_Boltzmann,
}


if __name__ == "__main__":
    # Tests
    print("=== Strahlungs-Modul Tests ===\n")

    # Test 1: Spektrale Emissionsleistung
    print("Test 1: Eb bei T=1273.15 K (= 1000°C)")
    for lam in [1, 2, 3, 5, 10]:
        eb = Eb(1273.15, lam)
        print(f"  Eb(1273.15 K, {lam} µm) = {eb:.2f} W/(m²·µm)")
    print()

    # Test 2: Wien'sche Verschiebung
    print("Test 2: Wien'sche Verschiebung")
    for T in [273.15, 373.15, 773.15, 1273.15, 5773.15]:
        lam_max = Wien_displacement(T)
        print(f"  λ_max({T} K) = {lam_max:.3f} µm")
    print()

    # Test 3: Stefan-Boltzmann
    print("Test 3: Stefan-Boltzmann Gesamtemission")
    for T in [273.15, 373.15, 773.15, 1273.15]:
        E = Stefan_Boltzmann(T)
        print(f"  E({T} K) = {E:.2f} W/m²")
    print()

    # Test 4: Blackbody-Fraktion (sichtbares Licht bei Sonnentemperatur)
    print("Test 4: Blackbody-Fraktion")
    T_sun = 5773.15  # K (ungefähre Sonnenoberflächentemperatur, = 5500°C)
    # Sichtbares Licht: 0.38 - 0.75 µm
    f_visible = Blackbody(T_sun, 0.38, 0.75)
    print(f"  Sichtbarer Anteil bei {T_sun} K: {f_visible*100:.1f}%")

    # Infrarot: 0.75 - 1000 µm
    f_ir = Blackbody(T_sun, 0.75, 100)
    print(f"  Infrarot-Anteil bei {T_sun} K: {f_ir*100:.1f}%")

    # UV: 0.01 - 0.38 µm
    f_uv = Blackbody(T_sun, 0.01, 0.38)
    print(f"  UV-Anteil bei {T_sun} K: {f_uv*100:.1f}%")
    print(f"  Summe: {(f_visible+f_ir+f_uv)*100:.1f}%")
