"""
Integrating Sphere Calculations
Based on LabSphere Technical Guide: A Guide to Integrating Sphere Theory and Applications
"""

import numpy as np
import pandas as pd

# Physical constants
H_PLANCK = 6.62607015e-34  # J·s
C_LIGHT = 2.99792458e8     # m/s


def sphere_surface_area(diameter_m: float) -> float:
    """Return surface area of sphere in m² given diameter in metres."""
    return np.pi * diameter_m ** 2


def port_fraction(port_diameters_m: list[float], sphere_diameter_m: float) -> float:
    """
    Calculate total port fraction f = sum(A_port) / A_sphere.

    Parameters
    ----------
    port_diameters_m : list of port diameters in metres (assuming circular ports)
    sphere_diameter_m : sphere inner diameter in metres

    Returns
    -------
    f : dimensionless port fraction
    """
    a_sphere = sphere_surface_area(sphere_diameter_m)
    total_port_area = sum(np.pi * (d / 2) ** 2 for d in port_diameters_m)
    return total_port_area / a_sphere


def sphere_multiplier(rho: np.ndarray, f: float) -> np.ndarray:
    """
    Calculate the sphere multiplier M.

    M = ρ / (1 − ρ·(1 − f))

    Parameters
    ----------
    rho : reflectance array (scalar or 1-D, values in [0, 1))
    f   : total port fraction (dimensionless)

    Returns
    -------
    M : sphere multiplier (same shape as rho)
    """
    rho = np.asarray(rho, dtype=float)
    denom = 1.0 - rho * (1.0 - f)
    # Avoid division by zero for perfect reflectors
    denom = np.where(denom == 0, np.nan, denom)
    return rho / denom


def sphere_radiance(
    phi_in: float,
    rho: np.ndarray,
    f: float,
    sphere_diameter_m: float,
) -> np.ndarray:
    """
    Spectral sphere radiance L [W · m⁻² · sr⁻¹ · nm⁻¹] (or per-nm units).

    L = M · Φ_in / (π · A_sphere)

    Parameters
    ----------
    phi_in          : input flux (W/nm, or normalised to 1 for ratio plots)
    rho             : reflectance array
    f               : total port fraction
    sphere_diameter_m : sphere inner diameter in metres

    Returns
    -------
    L : radiance array (same length as rho)
    """
    m = sphere_multiplier(rho, f)
    a_sphere = sphere_surface_area(sphere_diameter_m)
    return m * phi_in / (np.pi * a_sphere)


def output_flux_lens(
    L: np.ndarray,
    f_ratio: float,
    detector_area_m2: float,
    efficiency: np.ndarray,
) -> np.ndarray:
    """
    Output flux for lens-coupled port.

    Φ_out = L · A_det · π/(4·F²) · ε

    Parameters
    ----------
    L               : sphere radiance array
    f_ratio         : f-number of the coupling optics
    detector_area_m2: active detector area in m²
    efficiency      : optical system efficiency ε (scalar or array, [0, 1])

    Returns
    -------
    Φ_out : output flux array
    """
    solid_angle = np.pi / (4.0 * f_ratio ** 2)
    return L * detector_area_m2 * solid_angle * np.asarray(efficiency, dtype=float)


def output_flux_fiber(
    L: np.ndarray,
    na: float,
    fiber_diameter_m: float,
    r_fiber: float,
) -> np.ndarray:
    """
    Output flux for fiber-coupled port.

    Φ_out = L · A_fiber · π · NA² · (1 − R_fiber)

    Parameters
    ----------
    L               : sphere radiance array
    na              : numerical aperture of the fiber
    fiber_diameter_m: fiber core diameter in metres
    r_fiber         : Fresnel reflectivity of fiber end face [0, 1)

    Returns
    -------
    Φ_out : output flux array
    """
    a_fiber = np.pi * (fiber_diameter_m / 2.0) ** 2
    omega_fiber = np.pi * na ** 2
    return L * a_fiber * omega_fiber * (1.0 - r_fiber)


# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------

def flux_to_photon_rate(
    flux_per_nm: np.ndarray,
    wavelengths_nm: np.ndarray,
) -> np.ndarray:
    """
    Convert spectral flux [W/nm] to spectral photon flux [photons/s/nm].

    n_λ = Φ_λ · λ / (h · c)

    Parameters
    ----------
    flux_per_nm    : flux density [W/nm]
    wavelengths_nm : wavelength array [nm]

    Returns
    -------
    photon_flux : [photons/s/nm]
    """
    wavelengths_m = wavelengths_nm * 1e-9
    energy_per_photon = H_PLANCK * C_LIGHT / wavelengths_m   # J per photon
    return flux_per_nm / energy_per_photon


def photons_per_resolution_element(
    photon_flux_per_nm: np.ndarray,
    wavelengths_nm: np.ndarray,
    resolving_power: float,
) -> np.ndarray:
    """
    Convert spectral photon flux to photons per resolution element.

    Δλ = λ / R  (R = resolving power)
    N_res = n_λ · Δλ

    Parameters
    ----------
    photon_flux_per_nm : [photons/s/nm]
    wavelengths_nm     : wavelength array [nm]
    resolving_power    : R = λ/Δλ

    Returns
    -------
    photons per resolution element [photons/s]
    """
    delta_lambda = wavelengths_nm / resolving_power   # nm
    return photon_flux_per_nm * delta_lambda


# ---------------------------------------------------------------------------
# Coating / reflectance helpers
# ---------------------------------------------------------------------------

def flat_reflectance(rho_value: float, wavelengths_nm: np.ndarray) -> np.ndarray:
    """Return a flat reflectance array for the given wavelength grid."""
    return np.full_like(wavelengths_nm, rho_value, dtype=float)


def load_coating_csv(path: str, wavelength_col: str = "wavelength_nm",
                     reflectance_col: str = "reflectance") -> pd.DataFrame:
    """
    Load a coating reflectance CSV file.

    Expected columns: wavelength_nm, reflectance (0-1 scale)
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    # Try to auto-detect columns if standard names not present
    if wavelength_col not in df.columns:
        # look for column containing "wave" or "lambda" or "wl"
        for col in df.columns:
            if any(k in col for k in ("wave", "lambda", "wl", "nm")):
                df = df.rename(columns={col: wavelength_col})
                break
    if reflectance_col not in df.columns:
        for col in df.columns:
            if any(k in col for k in ("refl", "rho", "r_")):
                df = df.rename(columns={col: reflectance_col})
                break
    # If reflectance values look like percentages (> 1), divide by 100
    if df[reflectance_col].max() > 1.1:
        df[reflectance_col] /= 100.0
    return df[[wavelength_col, reflectance_col]].sort_values(wavelength_col).dropna()


def interpolate_reflectance(
    coating_df: pd.DataFrame,
    wavelengths_nm: np.ndarray,
    wavelength_col: str = "wavelength_nm",
    reflectance_col: str = "reflectance",
) -> np.ndarray:
    """Interpolate coating reflectance onto a target wavelength grid."""
    wl = coating_df[wavelength_col].values
    rho = coating_df[reflectance_col].values
    return np.interp(wavelengths_nm, wl, rho,
                     left=rho[0], right=rho[-1])


def interpolate_efficiency(
    efficiency_df: pd.DataFrame,
    wavelengths_nm: np.ndarray,
) -> np.ndarray:
    """Interpolate instrument efficiency onto a target wavelength grid."""
    cols = efficiency_df.columns.tolist()
    # Expect two columns: wavelength and efficiency
    wl_col = cols[0]
    eff_col = cols[1]
    wl = efficiency_df[wl_col].values.astype(float)
    eff = efficiency_df[eff_col].values.astype(float)
    if eff.max() > 1.1:
        eff /= 100.0
    return np.interp(wavelengths_nm, wl, eff, left=eff[0], right=eff[-1])
