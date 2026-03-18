"""
Integrating Sphere Calculator
Streamlit app based on LabSphere's Technical Guide on Integrating Spheres.
"""

import io
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sphere_calculations import (
    flat_reflectance,
    flux_to_photon_rate,
    interpolate_efficiency,
    interpolate_reflectance,
    load_coating_csv,
    output_flux_fiber,
    output_flux_lens,
    photons_per_resolution_element,
    sphere_multiplier,
    sphere_radiance,
    port_fraction,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Integrating Sphere Calculator",
    page_icon="🔵",
    layout="wide",
)

st.title("🔵 Integrating Sphere Calculator")
st.caption(
    "Based on LabSphere's *Technical Guide: A Guide to Integrating Sphere Theory "
    "and Applications*. Calculates sphere radiance, sphere multiplier, and "
    "output flux for lens- and fiber-coupled ports."
)

# ---------------------------------------------------------------------------
# Helper – locate bundled coating CSVs
# ---------------------------------------------------------------------------
COATINGS_DIR = os.path.join(os.path.dirname(__file__), "coatings")

PRESET_COATINGS: dict[str, str] = {}
if os.path.isdir(COATINGS_DIR):
    for fname in sorted(os.listdir(COATINGS_DIR)):
        if fname.endswith(".csv"):
            label = os.path.splitext(fname)[0].replace("_", " ").title()
            PRESET_COATINGS[label] = os.path.join(COATINGS_DIR, fname)

# ---------------------------------------------------------------------------
# Sidebar – inputs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Inputs")

    # ── Wavelength range ────────────────────────────────────────────────────
    with st.expander("Wavelength range", expanded=True):
        wl_range = st.slider(
            "Wavelength range (nm)",
            min_value=100,
            max_value=5000,
            value=(400, 1000),
            step=10,
        )
        wl_min, wl_max = wl_range
        wl_step = st.number_input("Step (nm)", value=2, min_value=1,
                                  max_value=50, step=1)
        wavelengths = np.arange(float(wl_min), float(wl_max) + 1e-6, float(wl_step))

    # ── Input flux ──────────────────────────────────────────────────────────
    with st.expander("Input flux", expanded=True):
        input_flux_total = st.number_input(
            "Total input flux Φ_in (W)",
            value=1.0, min_value=1e-12, format="%.4g",
        )
        flux_shape = st.selectbox(
            "Spectral distribution",
            ["Flat (equal W/nm)", "Custom CSV upload"],
        )
        flux_csv_file = None
        if flux_shape == "Custom CSV upload":
            flux_csv_file = st.file_uploader(
                "Upload flux spectrum CSV (wavelength_nm, flux_W_per_nm)",
                type=["csv"],
                key="flux_csv",
            )

    # ── Sphere geometry ──────────────────────────────────────────────────────
    with st.expander("Sphere geometry", expanded=True):
        sphere_diam_cm = st.number_input(
            "Sphere inner diameter (cm)", value=15.0, min_value=0.1,
            max_value=300.0, format="%.2f",
        )
        sphere_diam_m = sphere_diam_cm * 1e-2

        st.markdown("**Ports (diameters in cm)**")
        n_ports = st.number_input("Number of ports", value=2, min_value=1,
                                  max_value=10, step=1)
        port_diams_cm = []
        cols_ports = st.columns(2)
        for i in range(int(n_ports)):
            d = cols_ports[i % 2].number_input(
                f"Port {i+1} ∅ (cm)", value=2.5, min_value=0.01,
                max_value=sphere_diam_cm, format="%.2f", key=f"port_{i}",
            )
            port_diams_cm.append(d)
        port_diams_m = [d * 1e-2 for d in port_diams_cm]
        f_total = port_fraction(port_diams_m, sphere_diam_m)
        st.info(f"Port fraction f = {f_total:.4f}")

    # ── Reflectance ──────────────────────────────────────────────────────────
    with st.expander("Surface reflectance", expanded=True):
        rho_mode = st.radio(
            "Reflectance input mode",
            ["Scalar value", "Preset coating", "Upload CSV"],
            index=0,
        )
        coating_df = None
        if rho_mode == "Scalar value":
            rho_scalar = st.slider("Reflectance ρ", 0.80, 0.999, 0.97, 0.001,
                                   format="%.3f")
        elif rho_mode == "Preset coating":
            if not PRESET_COATINGS:
                st.warning("No preset coatings found in ./coatings/ directory.")
            preset_label = st.selectbox("Coating", list(PRESET_COATINGS.keys()))
            coating_df = load_coating_csv(PRESET_COATINGS[preset_label])
        else:  # Upload CSV
            uploaded_rho = st.file_uploader(
                "Upload reflectance CSV (wavelength_nm, reflectance)",
                type=["csv"],
                key="rho_csv",
            )
            if uploaded_rho is not None:
                coating_df = load_coating_csv(io.StringIO(uploaded_rho.read().decode()))

    # ── Output port ──────────────────────────────────────────────────────────
    with st.expander("Output port", expanded=True):
        port_type = st.selectbox("Coupling type", ["Lens", "Fiber"])

        if port_type == "Lens":
            f_ratio = st.number_input("f-ratio (f-number)", value=5.0,
                                      min_value=0.5, max_value=100.0, format="%.1f")
            det_diam_mm = st.number_input(
                "Detector active diameter (mm)", value=10.0,
                min_value=0.01, max_value=200.0, format="%.2f",
            )
            det_area_m2 = np.pi * (det_diam_mm * 1e-3 / 2) ** 2
            st.caption(f"Detector area = {det_area_m2 * 1e6:.4f} mm²")

            eff_mode = st.radio("System efficiency ε", ["Constant", "Upload CSV"],
                                key="eff_mode")
            eff_csv_file = None
            if eff_mode == "Constant":
                epsilon = st.slider("ε (0-1)", 0.01, 1.0, 0.85, 0.01)
            else:
                eff_csv_file = st.file_uploader(
                    "Upload efficiency CSV (wavelength_nm, efficiency)",
                    type=["csv"],
                    key="eff_csv",
                )

        else:  # Fiber
            fiber_na = st.number_input("Fiber NA", value=0.22,
                                       min_value=0.01, max_value=0.99, format="%.3f")
            fiber_diam_um = st.number_input(
                "Fiber core diameter (µm)", value=200.0,
                min_value=1.0, max_value=2000.0, format="%.1f",
            )
            fiber_diam_m = fiber_diam_um * 1e-6
            r_fiber = st.slider("Fiber end reflectivity R", 0.0, 0.20, 0.04, 0.01,
                                format="%.2f")

    # ── Astronomical spectrograph (bonus) ────────────────────────────────────
    with st.expander("Astronomical spectrograph (optional)", expanded=False):
        use_spectrograph = st.checkbox("Enable spectrograph output")
        if use_spectrograph:
            resolving_power = st.number_input(
                "Resolving power R = λ/Δλ", value=50000, min_value=100,
                max_value=1000000, step=1000,
            )
            inst_eff_mode = st.radio(
                "Instrument efficiency η(λ)",
                ["Constant", "Upload CSV"],
                key="inst_eff_mode",
            )
            inst_eff_csv = None
            if inst_eff_mode == "Constant":
                inst_eta = st.slider("η (0-1)", 0.01, 1.0, 0.10, 0.01,
                                     key="inst_eta")
            else:
                inst_eff_csv = st.file_uploader(
                    "Upload instrument efficiency CSV (wavelength_nm, efficiency)",
                    type=["csv"],
                    key="inst_eff_csv",
                )

    # ── Plot units ─────────────────────────────────────────────────────────
    with st.expander("Display options", expanded=True):
        unit_choice = st.selectbox(
            "Y-axis units",
            ["W/nm (normalised to input)", "photons/s/nm", "photons/s/resolution element"],
        )
        show_multiplier = st.checkbox("Show sphere multiplier M(λ)", value=True)
        show_radiance = st.checkbox("Show sphere radiance L(λ)", value=False)

# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

# Build spectral input flux Φ_in(λ) [W/nm]
if flux_shape == "Custom CSV upload" and flux_csv_file is not None:
    flux_df = pd.read_csv(flux_csv_file)
    flux_df.columns = [c.strip().lower() for c in flux_df.columns]
    wl_col_f = flux_df.columns[0]
    fl_col_f = flux_df.columns[1]
    phi_in_spectrum = np.interp(
        wavelengths,
        flux_df[wl_col_f].values.astype(float),
        flux_df[fl_col_f].values.astype(float),
        left=0, right=0,
    )
    # Normalise so total integral ≈ input_flux_total
    integral = np.trapezoid(phi_in_spectrum, wavelengths)
    if integral > 0:
        phi_in_spectrum = phi_in_spectrum * input_flux_total / integral
else:
    # Flat: equal W per nm
    span_nm = float(wl_max - wl_min)
    phi_in_spectrum = np.full_like(wavelengths, input_flux_total / span_nm)

# Build reflectance array ρ(λ)
if rho_mode == "Scalar value":
    rho_array = flat_reflectance(rho_scalar, wavelengths)
elif coating_df is not None:
    rho_array = interpolate_reflectance(coating_df, wavelengths)
else:
    st.warning("No reflectance data available – using ρ = 0.97.")
    rho_array = flat_reflectance(0.97, wavelengths)

# Sphere multiplier and radiance
M = sphere_multiplier(rho_array, f_total)
L = sphere_radiance(phi_in_spectrum, rho_array, f_total, sphere_diam_m)

# Output port flux
if port_type == "Lens":
    if eff_mode == "Constant":
        epsilon_arr = np.full_like(wavelengths, epsilon)
    else:
        if eff_csv_file is not None:
            eff_df = pd.read_csv(eff_csv_file)
            eff_df.columns = [c.strip().lower() for c in eff_df.columns]
            epsilon_arr = interpolate_efficiency(eff_df, wavelengths)
        else:
            st.warning("No efficiency CSV uploaded – using ε = 1.")
            epsilon_arr = np.ones_like(wavelengths)
    phi_out = output_flux_lens(L, f_ratio, det_area_m2, epsilon_arr)

else:  # Fiber
    phi_out = output_flux_fiber(L, fiber_na, fiber_diam_m, r_fiber)

# Ratio output / input – both numerator and denominator are in W/nm, so the ratio is dimensionless
ratio_out_in = phi_out / phi_in_spectrum

# Convert units if needed
y_label = ""
if unit_choice == "W/nm (normalised to input)":
    y_values = ratio_out_in
    y_label = "Φ_out / Φ_in  (dimensionless)"
    y_hover = ".4g"
elif unit_choice == "photons/s/nm":
    phi_out_photons = flux_to_photon_rate(phi_out, wavelengths)
    y_values = phi_out_photons
    y_label = "Spectral photon flux  [photons · s⁻¹ · nm⁻¹]"
    y_hover = ".3e"
else:  # photons per resolution element
    if not use_spectrograph:
        st.warning(
            "Enable the spectrograph section in the sidebar to use "
            "'photons/s/resolution element'."
        )
        phi_out_photons = flux_to_photon_rate(phi_out, wavelengths)
        y_values = phi_out_photons
        y_label = "Spectral photon flux  [photons · s⁻¹ · nm⁻¹]"
        y_hover = ".3e"
    else:
        phi_out_photons = flux_to_photon_rate(phi_out, wavelengths)
        if inst_eff_mode == "Constant":
            inst_eta_arr = np.full_like(wavelengths, inst_eta)
        else:
            if inst_eff_csv is not None:
                inst_df = pd.read_csv(inst_eff_csv)
                inst_df.columns = [c.strip().lower() for c in inst_df.columns]
                inst_eta_arr = interpolate_efficiency(inst_df, wavelengths)
            else:
                st.warning("No instrument efficiency CSV – using η = 1.")
                inst_eta_arr = np.ones_like(wavelengths)
        y_values = photons_per_resolution_element(
            phi_out_photons * inst_eta_arr, wavelengths, resolving_power
        )
        y_label = "Photons per resolution element  [photons · s⁻¹ · res. el.⁻¹]"
        y_hover = ".3e"

# ---------------------------------------------------------------------------
# Right pane – results and plots
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([1, 2], gap="large")

with col_left:
    st.subheader("📊 Key values")

    a_sphere_cm2 = np.pi * sphere_diam_cm ** 2
    st.metric("Sphere surface area", f"{a_sphere_cm2:.1f} cm²")
    st.metric("Port fraction f", f"{f_total:.4f}")
    st.metric("Mean sphere multiplier M̄",
              f"{np.nanmean(M):.3f}")

    if port_type == "Lens":
        omega_sr = np.pi / (4 * f_ratio ** 2)
        st.metric("Collection solid angle Ω", f"{omega_sr:.4f} sr")
    else:
        omega_fiber = np.pi * fiber_na ** 2
        a_fib_mm2 = np.pi * (fiber_diam_um / 2e3) ** 2
        st.metric("Fiber collection Ω", f"{omega_fiber:.4f} sr")
        st.metric("Fiber end area", f"{a_fib_mm2:.5f} mm²")

    mean_ratio = float(np.nanmean(ratio_out_in))
    st.metric(
        "Mean Φ_out / Φ_in",
        f"{mean_ratio:.4e}",
        help="Average output-to-input flux ratio across the wavelength range.",
    )

    total_out_W = float(np.trapezoid(phi_out, wavelengths))
    st.metric("Total output flux (integrated)", f"{total_out_W:.4e} W",
              help="∫ Φ_out(λ) dλ over the selected wavelength range.")

with col_right:
    tabs = st.tabs(["Output / Input ratio", "Sphere multiplier", "Reflectance"])

    # ── Tab 1: main output ratio plot ─────────────────────────────────────
    with tabs[0]:
        fig1 = go.Figure()
        fig1.add_trace(
            go.Scatter(
                x=wavelengths,
                y=y_values,
                mode="lines",
                name="Φ_out",
                line=dict(color="royalblue", width=2),
                hovertemplate=f"λ=%{{x:.1f}} nm<br>value=%{{y:{y_hover}}}<extra></extra>",
            )
        )
        fig1.update_layout(
            xaxis_title="Wavelength (nm)",
            yaxis_title=y_label,
            title="Output flux",
            hovermode="x unified",
            margin=dict(l=60, r=20, t=50, b=50),
        )
        st.plotly_chart(fig1, use_container_width=True)

    # ── Tab 2: sphere multiplier ──────────────────────────────────────────
    with tabs[1]:
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=wavelengths,
                y=M,
                mode="lines",
                name="M(λ)",
                line=dict(color="darkorange", width=2),
                hovertemplate="λ=%{x:.1f} nm<br>M=%{y:.3f}<extra></extra>",
            )
        )
        fig2.update_layout(
            xaxis_title="Wavelength (nm)",
            yaxis_title="Sphere multiplier M",
            title="Sphere Multiplier M(λ) = ρ / [1 − ρ·(1−f)]",
            hovermode="x unified",
            margin=dict(l=60, r=20, t=50, b=50),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 3: reflectance ────────────────────────────────────────────────
    with tabs[2]:
        fig3 = go.Figure()
        fig3.add_trace(
            go.Scatter(
                x=wavelengths,
                y=rho_array,
                mode="lines",
                name="ρ(λ)",
                line=dict(color="green", width=2),
                hovertemplate="λ=%{x:.1f} nm<br>ρ=%{y:.4f}<extra></extra>",
            )
        )
        fig3.update_layout(
            xaxis_title="Wavelength (nm)",
            yaxis_title="Reflectance ρ",
            title="Surface Reflectance ρ(λ)",
            yaxis_range=[0, 1.05],
            hovermode="x unified",
            margin=dict(l=60, r=20, t=50, b=50),
        )
        st.plotly_chart(fig3, use_container_width=True)

# ---------------------------------------------------------------------------
# Theory reference
# ---------------------------------------------------------------------------
with st.expander("📖 Theory reference", expanded=False):
    st.markdown(
        r"""
### Integrating Sphere Theory (LabSphere)

**Sphere surface area:**  $A_s = \pi d^2$  (diameter $d$)

**Port fraction:**  $f = \dfrac{\sum_i A_{\text{port},i}}{A_s}$

**Sphere multiplier:**
$$M = \frac{\rho}{1 - \rho(1 - f)}$$

**Sphere radiance (W m⁻² sr⁻¹ [nm⁻¹]):**
$$L = \frac{M \cdot \Phi_\text{in}}{\pi A_s}$$

**Lens-coupled output flux:**
$$\Phi_\text{out} = L \cdot A_\text{det} \cdot \frac{\pi}{4 F^2} \cdot \varepsilon(\lambda)$$
where $F$ = f-number, $A_\text{det}$ = detector active area, $\varepsilon$ = system efficiency.

**Fiber-coupled output flux:**
$$\Phi_\text{out} = L \cdot A_\text{fiber} \cdot \pi \cdot \text{NA}^2 \cdot (1 - R_\text{fiber})$$
where $A_\text{fiber} = \pi (d_\text{fiber}/2)^2$, NA = numerical aperture,
$R_\text{fiber}$ = Fresnel reflectivity of fiber end face.

**Photon rate:**
$$\dot{n}(\lambda) = \frac{\Phi(\lambda) \cdot \lambda}{h c}$$

**Photons per resolution element** (resolving power $\mathcal{R} = \lambda/\Delta\lambda$):
$$N_\text{res}(\lambda) = \dot{n}(\lambda) \cdot \frac{\lambda}{\mathcal{R}}$$
        """
    )
