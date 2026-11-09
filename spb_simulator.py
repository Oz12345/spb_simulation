# -*- coding: utf-8 -*-
"""
Super Pressure Balloon (SPB) Simulator  -  fixed-envelope, mass-driven helium
============================================================================

Design chain (recomputed on every input change):
    1. Envelope geometry is fixed by the diameter D:
           V_max = (pi/6) D^3        A = pi D^2           (sphere)
    2. Envelope mass from area and film areal density:
           m_env = A * (1 + seam overhead) * film_density
    3. Dry mass:   m_dry = m_env + m_payload + m_hardware
    4. BASE helium = helium needed for NEUTRAL BUOYANCY at the TARGET float
       altitude (same balance as the original simulator):
           rho_air(h_t) * V_He(h_t) = m_dry + m_He,base
       where V_He(h_t) is the volume the helium occupies at ambient T, P.
    5. Helium filling rule (separate control) adds a launch lift margin:
           m_He,launch = m_He,base * (1 + margin)
    6. Total system mass is always derived:
           m_total = m_env + m_payload + m_hardware + m_He,launch
    7. The fixed envelope then decides where the balloon really floats:
           rho_air(h_float) * V_max = m_total    (fully pressurized)
       The app compares h_float with the target and reports the envelope
       diameter that would make them coincide.

Physics modules: ISA 1976 atmosphere, 1-DOF ascent ODE (buoyancy, weight,
quadratic drag, added mass, volume capped at V_max), 2-node transient thermal
model (film + helium) for the 24 h cycle, superpressure and film line load.

Single file, no local dependencies -> deployable on Streamlit Community Cloud.
"""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# =============================================================================
# Physical constants
# =============================================================================
G0 = 9.80665              # gravitational acceleration [m/s^2]
R_U = 8.314462618         # universal gas constant [J/(mol K)]
M_AIR = 0.0289644         # molar mass of dry air [kg/mol]
M_HE = 0.0040026          # molar mass of helium [kg/mol]
SIGMA = 5.670374419e-8    # Stefan-Boltzmann [W/(m^2 K^4)]
SOLAR_C = 1361.0          # solar constant [W/m^2]
CV_HE = 3116.0            # helium Cv [J/(kg K)]
K_MOLAR = M_HE / M_AIR    # helium / air molar-mass ratio (~0.138)
P_SL, T_SL = 101325.0, 288.15

# =============================================================================
# ISA 1976 atmosphere
# =============================================================================
_LAYERS = [  # (base altitude [m], base temperature [K], lapse rate [K/m])
    (0.0, 288.15, -0.0065),
    (11000.0, 216.65, 0.0),
    (20000.0, 216.65, 0.0010),
    (32000.0, 228.65, 0.0028),
    (47000.0, 270.65, 0.0),
    (51000.0, 270.65, -0.0028),
    (71000.0, 214.65, -0.0020),
]
_GMR = G0 * M_AIR / R_U


def _layer_base_pressures():
    p = [P_SL]
    for i in range(1, len(_LAYERS)):
        h0, T0, L = _LAYERS[i - 1]
        h1 = _LAYERS[i][0]
        if L == 0.0:
            p.append(p[-1] * np.exp(-_GMR * (h1 - h0) / T0))
        else:
            p.append(p[-1] * ((T0 + L * (h1 - h0)) / T0) ** (-_GMR / L))
    return p


_P_BASE = _layer_base_pressures()


def atmosphere(h):
    """Return (T [K], P [Pa], rho [kg/m^3]) at altitude h [m] (0-86 km)."""
    h = min(max(float(h), 0.0), 86000.0)
    i = len(_LAYERS) - 1
    while h < _LAYERS[i][0]:
        i -= 1
    h0, T0, L = _LAYERS[i]
    T = T0 + L * (h - h0)
    if L == 0.0:
        P = _P_BASE[i] * np.exp(-_GMR * (h - h0) / T0)
    else:
        P = _P_BASE[i] * (T / T0) ** (-_GMR / L)
    return T, P, P * M_AIR / (R_U * T)


def altitude_for_density(rho_target):
    """Invert rho(h) = rho_target on 0-50 km. NaN if out of range."""
    f = lambda h: atmosphere(h)[2] - rho_target
    if f(0.0) < 0 or f(50000.0) > 0:
        return np.nan
    return brentq(f, 0.0, 50000.0, xtol=0.1)


def helium_volume(m_he, h, T_gas=None):
    """Volume [m^3] of m_he kg of helium at ambient pressure of altitude h."""
    T, P, _ = atmosphere(h)
    return m_he * R_U * (T if T_gas is None else T_gas) / (P * M_HE)


# =============================================================================
# Design & mass budget
# =============================================================================
@dataclass(frozen=True)
class Design:
    diameter: float          # envelope diameter [m]
    film_gsm: float          # film areal density [g/m^2]
    seam_pct: float          # seams / tapes / overlap overhead [% of area]
    payload: float           # payload mass [kg]
    hardware: float          # safety / hardware mass [kg]

    @property
    def radius(self):
        return self.diameter / 2.0

    @property
    def v_max(self):
        return np.pi / 6.0 * self.diameter ** 3

    @property
    def area(self):
        return np.pi * self.diameter ** 2

    @property
    def m_env(self):
        return self.area * (1 + self.seam_pct / 100.0) * self.film_gsm / 1000.0

    @property
    def m_dry(self):
        return self.m_env + self.payload + self.hardware


def base_helium(d: Design, h_target):
    """Helium mass for neutral buoyancy at the target altitude (original logic):
    displaced air mass of the helium volume = total system mass."""
    T, P, rho = atmosphere(h_target)

    def neutral_balance(m_he):
        V = helium_volume(m_he, h_target)
        return V * rho - (d.m_env + d.payload + d.hardware + m_he)

    hi = 10.0 * d.m_dry + 1.0
    return brentq(neutral_balance, 1e-9, hi, xtol=1e-10)


def launch_helium(m_he_base, rule, value, d: Design):
    """Helium filling rule -> launch helium mass."""
    if rule == "Lift margin on base helium":
        return m_he_base * (1.0 + value / 100.0)
    # "Free lift [% of total weight]": m_He/K = (1+f)(m_dry + m_He)
    f = value / 100.0
    return (1.0 + f) * d.m_dry / (1.0 / K_MOLAR - (1.0 + f))


def float_state(d: Design, m_he):
    """Equilibrium float of the fixed envelope (helium at ambient temperature)."""
    m_total = d.m_dry + m_he
    h = altitude_for_density(m_total / d.v_max)
    if np.isnan(h):
        return dict(h=np.nan, T=np.nan, P=np.nan, dp=np.nan)
    T, P, rho = atmosphere(h)
    m_fill = rho * d.v_max * K_MOLAR          # helium that just fills V_max
    return dict(h=h, T=T, P=P, dp=P * (m_he / m_fill - 1.0))


def required_diameter(d: Design, h_target, rule, value):
    """Envelope diameter whose float altitude equals the target (same masses,
    same film, same filling rule). NaN if none in 0.2-200 m."""
    def resid(D):
        dx = replace(d, diameter=D)
        mh = launch_helium(base_helium(dx, h_target), rule, value, dx)
        return atmosphere(h_target)[2] * dx.v_max - (dx.m_dry + mh)
    Ds = np.geomspace(0.2, 200.0, 200)
    r = np.array([resid(D) for D in Ds])
    idx = np.where(np.sign(r[:-1]) != np.sign(r[1:]))[0]
    if len(idx) == 0:
        return np.nan
    i = idx[-1]
    return brentq(resid, Ds[i], Ds[i + 1], xtol=1e-4)


# =============================================================================
# Ascent model
# =============================================================================
def ascent_rhs(t, y, m_he, m_sys, v_max, cd, c_add):
    h, v = y
    T, P, rho = atmosphere(h)
    V = min(m_he * R_U * T / (P * M_HE), v_max)
    r = (3.0 * V / (4.0 * np.pi)) ** (1.0 / 3.0)
    f_net = rho * V * G0 - m_sys * G0 - 0.5 * rho * v * abs(v) * cd * np.pi * r * r
    acc = f_net / (m_sys + c_add * rho * V)               # added (virtual) mass
    if h <= 0.0 and acc < 0.0 and v <= 0.0:               # resting on the ground
        return [0.0, 0.0]
    return [v, acc]


@st.cache_data(show_spinner=False)
def run_ascent(m_he, m_sys, v_max, cd, c_add, t_end_s):
    sol = solve_ivp(ascent_rhs, (0.0, t_end_s), [0.0, 0.0],
                    args=(m_he, m_sys, v_max, cd, c_add),
                    method="RK45", max_step=10.0, rtol=1e-6, atol=1e-6)
    t, h, v = sol.t, np.maximum(sol.y[0], 0.0), sol.y[1]
    T, P, _ = np.array([atmosphere(x) for x in h]).T
    fill = np.minimum(m_he * R_U * T / (P * M_HE) / v_max, 1.0) * 100.0
    acc = np.gradient(v, t) if len(t) > 2 else np.zeros_like(t)
    return t, h, v, acc, fill


# =============================================================================
# Thermal model at float (fixed altitude & volume)
# =============================================================================
@dataclass(frozen=True)
class ThermalEnv:
    latitude_deg: float
    day_of_year: int
    alpha: float          # film solar absorptivity
    eps: float            # film IR emissivity
    albedo: float
    tau_atm: float        # transmissivity per unit air mass above the balloon
    q_earth_ir: float     # upwelling Earth / cloud IR [W/m^2]
    t_sky: float          # effective sky temperature [K]
    h_ext: float          # external convection [W/m^2K]
    h_int: float          # internal convection [W/m^2K]
    cp_film: float        # film specific heat [J/kgK]


def direct_solar(hour, env: ThermalEnv, p_amb):
    lat = np.radians(env.latitude_deg)
    decl = np.radians(23.44) * np.sin(2.0 * np.pi * (284 + env.day_of_year) / 365.0)
    ha = np.radians((hour - 12.0) * 15.0)
    s = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha)
    el = np.arcsin(np.clip(s, -1.0, 1.0))
    if el <= 0.0:
        return 0.0, 0.0
    airmass = 1.0 / (np.sin(el) + 0.50572 * (np.degrees(el) + 6.07995) ** -1.6364)
    airmass *= p_amb / P_SL          # only the air column above the balloon
    return SOLAR_C * env.tau_atm ** airmass, np.sin(el)


def thermal_rhs(t, y, m_he, m_env, A, T_amb, p_amb, env):
    T_he, T_f = y
    i_dn, sin_el = direct_solar((t / 3600.0) % 24.0, env, p_amb)
    q_sun = env.alpha * (i_dn * A / 4.0 + env.albedo * i_dn * sin_el * A / 2.0)
    q_ir_in = env.eps * A / 2.0 * (env.q_earth_ir + SIGMA * env.t_sky ** 4)
    q_ir_out = env.eps * SIGMA * T_f ** 4 * A
    q_conv = env.h_ext * A * (T_f - T_amb)
    q_int = env.h_int * A * (T_f - T_he)
    return [q_int / (m_he * CV_HE),
            (q_sun + q_ir_in - q_ir_out - q_conv - q_int) / (m_env * env.cp_film)]


@st.cache_data(show_spinner=False)
def run_thermal(m_he, m_env, A, T_amb, p_amb, env: ThermalEnv, n_days=3):
    """Integrate n_days from ambient start; return the last (periodic) day."""
    t_end = n_days * 86400.0
    t_eval = np.linspace(t_end - 86400.0, t_end, 289)
    sol = solve_ivp(thermal_rhs, (0.0, t_end), [T_amb, T_amb],
                    args=(m_he, m_env, A, T_amb, p_amb, env),
                    method="LSODA", t_eval=t_eval, max_step=300.0, rtol=1e-6)
    return (sol.t - (t_end - 86400.0)) / 3600.0, sol.y[0], sol.y[1]


# =============================================================================
# Case evaluation
# =============================================================================
def evaluate_case(label, d: Design, m_he, h_target, env, cd, c_add, t_asc_h):
    m_sys = d.m_dry + m_he
    gross = m_he / K_MOLAR                      # buoyancy mass at launch (isothermal)
    fs = float_state(d, m_he)
    out = dict(label=label, design=d, m_he=m_he, m_sys=m_sys,
               fl_pct=(gross / m_sys - 1.0) * 100.0, fl_N=(gross - m_sys) * G0,
               V0=helium_volume(m_he, 0.0), V_tgt=helium_volume(m_he, h_target),
               float=fs, ok=True, msg="")
    if out["fl_pct"] <= 0:
        out.update(ok=False, msg="No positive free lift at launch - the balloon will not ascend.")
        return out
    out["ascent"] = run_ascent(m_he, m_sys, d.v_max, cd, c_add, t_asc_h * 3600.0)
    if np.isnan(fs["h"]):
        out.update(ok=False, msg="No float equilibrium found below 50 km.")
        return out
    hrs, T_he, T_f = run_thermal(m_he, d.m_env, d.area, fs["T"], fs["P"], env)
    dp = m_he * R_U * T_he / (d.v_max * M_HE) - fs["P"]
    out["thermal"] = dict(hours=hrs, T_he=T_he, T_f=T_f, dp=dp,
                          load=np.maximum(dp, 0.0) * d.radius / 2.0)
    return out


def summary_row(c, h_target):
    d, fs = c["design"], c["float"]
    row = {
        "Case": c["label"], "D [m]": d.diameter, "V_max [m³]": d.v_max,
        "m_env [kg]": d.m_env, "m_He [kg]": c["m_he"], "m_total [kg]": c["m_sys"],
        "Free lift [%]": c["fl_pct"], "Free lift [N]": c["fl_N"],
        "V_launch [m³]": c["V0"], "He fill @ target [%]": c["V_tgt"] / d.v_max * 100.0,
        "Float alt [km]": fs["h"] / 1000.0,
        "Float − target [km]": (fs["h"] - h_target) / 1000.0,
        "ΔP isothermal [Pa]": fs["dp"],
    }
    if "ascent" in c:
        t, h, v, *_ = c["ascent"]
        h_f = fs["h"] if np.isfinite(fs["h"]) else h[-1]
        reach = np.where(h >= 0.98 * h_f)[0]
        row["Time to float [min]"] = t[reach[0]] / 60.0 if len(reach) else np.nan
        mask = (h > 100) & (h < 0.95 * h_f) & (v > 0)
        row["Mean ascent rate [m/s]"] = v[mask].mean() if mask.any() else np.nan
    if "thermal" in c:
        th = c["thermal"]
        row.update({
            "ΔP min [Pa]": th["dp"].min(), "ΔP max [Pa]": th["dp"].max(),
            "T_He min [K]": th["T_he"].min(), "T_He max [K]": th["T_he"].max(),
            "Max film load [N/m]": th["load"].max(),
        })
    return row


# =============================================================================
# Streamlit UI  (single page, no tabs)
# =============================================================================
COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#a16207"]


def sidebar_inputs():
    sb = st.sidebar
    sb.title("🎈 SPB Inputs")

    sb.subheader("1 · Envelope geometry")
    D = sb.number_input("Envelope diameter D [m]", 0.5, 100.0, 4.6, 0.05,
                        help="Fixed design diameter. V_max = πD³/6, A = πD².")
    gsm = sb.number_input("Film areal density [g/m²]", 5.0, 300.0, 69.0, 1.0)
    seam = sb.number_input("Seam / tape overhead [% of area]", 0.0, 50.0, 0.0, 0.5)

    sb.subheader("2 · Masses")
    payload = sb.number_input("Payload mass [kg]", 0.0, 500.0, 0.0, 0.1)
    hardware = sb.number_input("Safety / hardware mass [kg]", 0.0, 200.0, 5.0, 0.1)

    sb.subheader("3 · Target float altitude")
    h_t = sb.number_input("Target float altitude [km]", 1.0, 45.0, 14.0, 0.1,
                          help="Altitude used for the neutral-buoyancy helium calculation.") * 1000.0

    sb.subheader("4 · Helium filling rule")
    rule = sb.radio("Launch helium =", ["Lift margin on base helium",
                                        "Free lift [% of total weight]"], index=0,
                    help="Base helium = neutral buoyancy at the target altitude. "
                         "The rule adds lift so the balloon leaves the ground.")
    rule_val = sb.number_input("Margin / free lift [%]", 0.0, 200.0, 20.0, 0.5)

    sb.subheader("5 · Sensitivity cases")
    he_var = sb.number_input("± helium mass [%]", 0.0, 50.0, 5.0, 0.5)
    d_var = sb.number_input("± diameter, same helium [%]", 0.0, 20.0, 2.0, 0.5)
    gsm_var = sb.number_input("± film density, same helium [%]", 0.0, 30.0, 0.0, 1.0)

    sb.subheader("6 · Ascent dynamics")
    cd = sb.number_input("Drag coefficient C_d", 0.1, 2.0, 0.47, 0.01)
    c_add = sb.number_input("Added-mass coefficient", 0.0, 1.0, 0.5, 0.05)
    t_asc = sb.number_input("Ascent simulation time [h]", 0.5, 12.0, 4.0, 0.5)

    sb.subheader("7 · Thermal & radiation")
    lat = sb.number_input("Latitude [°]", -90.0, 90.0, 32.0, 0.1)
    doy = sb.slider("Day of year", 1, 365, 355, help="355 ≈ Dec 21.")
    alpha = sb.slider("Film solar absorptivity α", 0.01, 1.0, 0.40, 0.01)
    eps = sb.slider("Film IR emissivity ε", 0.01, 1.0, 0.85, 0.01)
    albedo = sb.slider("Surface / cloud albedo", 0.0, 1.0, 0.30, 0.01)
    tau = sb.slider("Atmospheric transmissivity τ (per air mass)", 0.3, 1.0, 0.85, 0.01)
    q_ir = sb.number_input("Upwelling Earth IR [W/m²]", 50.0, 400.0, 230.0, 5.0)
    t_sky = sb.number_input("Effective sky temperature [K]", 3.0, 300.0, 216.65, 1.0)
    h_ext = sb.slider("External convection h_ext [W/m²K]", 0.1, 10.0, 0.6, 0.1)
    h_int = sb.slider("Internal convection h_int [W/m²K]", 0.1, 5.0, 2.0, 0.1)
    cp_f = sb.number_input("Film specific heat [J/kgK]", 500.0, 3000.0, 1500.0, 50.0)

    design = Design(D, gsm, seam, payload, hardware)
    env = ThermalEnv(lat, int(doy), alpha, eps, albedo, tau, q_ir, t_sky, h_ext, h_int, cp_f)
    return (design, h_t, rule, rule_val, env,
            dict(he=he_var, d=d_var, gsm=gsm_var), dict(cd=cd, c_add=c_add, t_asc=t_asc))


def build_cases(base, m_he, var):
    cases = [("Baseline", base, m_he)]
    if var["he"] > 0:
        cases += [(f"+{var['he']:g}% He", base, m_he * (1 + var["he"] / 100)),
                  (f"−{var['he']:g}% He", base, m_he * (1 - var["he"] / 100))]
    for key, attr, tag in (("d", "diameter", "D"), ("gsm", "film_gsm", "film")):
        if var[key] > 0:
            for s in (+1, -1):
                dd = replace(base, **{attr: getattr(base, attr) * (1 + s * var[key] / 100)})
                cases.append((f"{'+' if s > 0 else '−'}{var[key]:g}% {tag}", dd, m_he))
    return cases


def line_fig(title, xt, yt, hover="x unified"):
    fig = go.Figure()
    fig.update_layout(title=title, xaxis_title=xt, yaxis_title=yt, hovermode=hover,
                      legend_title_text="Case", margin=dict(l=10, r=10, t=50, b=10), height=400)
    return fig


def main():
    st.set_page_config(page_title="SPB Simulator", page_icon="🎈", layout="wide",
                       initial_sidebar_state="expanded")
    design, h_t, rule, rule_val, env, var, dyn = sidebar_inputs()

    st.title("Super Pressure Balloon Simulator")
    st.caption("Diameter → V_max & area → envelope mass → dry mass → helium for neutral buoyancy "
               "at the target altitude → + filling rule → total system mass → actual float of the "
               "fixed envelope.")

    # ---- Mass budget (all derived) ----------------------------------------
    m_he_base = base_helium(design, h_t)
    m_he = launch_helium(m_he_base, rule, rule_val, design)
    if not np.isfinite(m_he) or m_he <= 0:
        st.error("The filling rule gives no valid helium mass. Reduce the free-lift value.")
        st.stop()
    m_total_base = design.m_dry + m_he_base
    m_total = design.m_dry + m_he
    T_t, P_t, rho_t = atmosphere(h_t)
    V_base_t = helium_volume(m_he_base, h_t)
    D_req = required_diameter(design, h_t, rule, rule_val)

    base = evaluate_case("Baseline", design, m_he, h_t, env, dyn["cd"], dyn["c_add"], dyn["t_asc"])
    fs = base["float"]

    st.header("Mass budget")
    c = st.columns(6)
    c[0].metric("V_max", f"{design.v_max:,.1f} m³")
    c[1].metric("Envelope area", f"{design.area:,.1f} m²")
    c[2].metric("Envelope mass", f"{design.m_env:,.2f} kg")
    c[3].metric("Base helium (neutral @ target)", f"{m_he_base:,.3f} kg")
    c[4].metric("Launch helium", f"{m_he:,.3f} kg", f"+{m_he - m_he_base:,.3f} kg (rule)",
                delta_color="off")
    c[5].metric("Total system mass", f"{m_total:,.2f} kg",
                f"{m_total_base:,.2f} kg at neutral", delta_color="off")

    budget = pd.DataFrame({
        "Component": ["Envelope (film × area)", "Payload", "Safety / hardware",
                      "Helium – base (neutral @ target)", "Helium – filling-rule margin",
                      "TOTAL SYSTEM MASS"],
        "Mass [kg]": [design.m_env, design.payload, design.hardware, m_he_base,
                      m_he - m_he_base, m_total],
    })
    budget["Share [%]"] = budget["Mass [kg]"] / m_total * 100.0
    a, b = st.columns([1, 1])
    a.dataframe(budget.style.format({"Mass [kg]": "{:,.3f}", "Share [%]": "{:,.1f}"}),
                hide_index=True, width="stretch")
    pie = go.Figure(go.Pie(labels=budget["Component"][:-1], values=budget["Mass [kg]"][:-1],
                           hole=0.5, sort=False, marker=dict(colors=COLORS)))
    pie.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=10))
    b.plotly_chart(pie, width="stretch")

    # ---- Target altitude vs envelope ---------------------------------------
    st.header("Target altitude check")
    c = st.columns(5)
    c[0].metric("Target altitude", f"{h_t / 1000:,.2f} km")
    c[1].metric("Base He volume @ target", f"{V_base_t:,.1f} m³",
                f"{V_base_t / design.v_max * 100:,.1f}% of V_max", delta_color="off")
    c[2].metric("Actual float altitude",
                f"{fs['h'] / 1000:,.2f} km" if np.isfinite(fs["h"]) else "—",
                f"{(fs['h'] - h_t) / 1000:+,.2f} km vs target" if np.isfinite(fs["h"]) else None,
                delta_color="off")
    c[3].metric("Launch free lift", f"{base['fl_pct']:,.1f} %", f"{base['fl_N']:,.1f} N",
                delta_color="off")
    c[4].metric("Diameter to float at target",
                f"{D_req:,.2f} m" if np.isfinite(D_req) else "—",
                f"{D_req - design.diameter:+,.2f} m vs current" if np.isfinite(D_req) else None,
                delta_color="off")

    if V_base_t > design.v_max:
        st.error(f"The envelope is too small: at {h_t/1000:.1f} km the base helium needs "
                 f"{V_base_t:,.1f} m³ but V_max is {design.v_max:,.1f} m³. The balloon becomes "
                 f"pressurized and stops below the target. Increase D to ≈ {D_req:,.2f} m, "
                 "reduce mass, or lower the target.")
    elif np.isfinite(fs["h"]) and abs(fs["h"] - h_t) > 250:
        st.info(f"With D = {design.diameter:g} m the balloon floats at {fs['h']/1000:.2f} km, "
                f"{'above' if fs['h'] > h_t else 'below'} the {h_t/1000:.1f} km target. "
                f"A {D_req:,.2f} m envelope would float exactly at the target with these masses "
                "and this filling rule.")
    else:
        st.success("The envelope floats at the target altitude with these masses.")
    if not base["ok"]:
        st.error(base["msg"])

    # ---- Run cases ----------------------------------------------------------
    cases = build_cases(design, m_he, var)
    with st.spinner("Running ascent and 24 h thermal simulations…"):
        results = [base] + [evaluate_case(lbl, d, mh, h_t, env, dyn["cd"], dyn["c_add"],
                                          dyn["t_asc"]) for lbl, d, mh in cases[1:]]

    st.header("Simulation summary")
    df = pd.DataFrame([summary_row(r, h_t) for r in results]).set_index("Case")
    st.dataframe(df.style.format("{:,.2f}", na_rep="—"), width="stretch")
    st.download_button("Download summary (CSV)", df.to_csv().encode(), "spb_summary.csv", "text/csv")
    for r in results[1:]:
        if not r["ok"]:
            st.warning(f"{r['label']}: {r['msg']}")

    # ---- Ascent -------------------------------------------------------------
    st.header("Ascent profile")
    f_h = line_fig("Altitude vs time", "Time [min]", "Altitude [km]")
    f_v = line_fig("Vertical velocity vs time", "Time [min]", "Velocity [m/s]")
    f_a = line_fig("Acceleration vs time", "Time [min]", "Acceleration [m/s²]")
    f_fill = line_fig("Envelope fill V/V_max vs altitude", "Altitude [km]", "Fill [%]", "closest")
    for r, col in zip(results, COLORS * 2):
        if "ascent" not in r:
            continue
        t, h, v, acc, fill = r["ascent"]
        kw = dict(name=r["label"], line=dict(color=col))
        f_h.add_trace(go.Scatter(x=t / 60, y=h / 1000, **kw))
        f_v.add_trace(go.Scatter(x=t / 60, y=v, **kw))
        f_a.add_trace(go.Scatter(x=t / 60, y=acc, **kw))
        f_fill.add_trace(go.Scatter(x=h / 1000, y=fill, **kw))
    f_h.add_hline(y=h_t / 1000, line_dash="dash", line_color="gray",
                  annotation_text="target", annotation_position="bottom right")
    f_fill.add_vline(x=h_t / 1000, line_dash="dash", line_color="gray")
    a, b = st.columns(2)
    a.plotly_chart(f_h, width="stretch"); b.plotly_chart(f_v, width="stretch")
    a.plotly_chart(f_a, width="stretch"); b.plotly_chart(f_fill, width="stretch")

    # ---- Thermal ------------------------------------------------------------
    st.header("24-hour thermal cycle at float")
    st.caption("Third day of a 3-day integration (periodic state), local solar time; "
               "altitude and volume held at the actual float point.")
    f_T = line_fig("Helium temperature", "Hour of day", "T_He [K]")
    f_Tf = line_fig("Film temperature", "Hour of day", "T_film [K]")
    f_dp = line_fig("Superpressure ΔP = P_in − P_amb", "Hour of day", "ΔP [Pa]")
    f_N = line_fig("Film line load N = ΔP·r/2", "Hour of day", "N [N/m]")
    for r, col in zip(results, COLORS * 2):
        if "thermal" not in r:
            continue
        th = r["thermal"]
        kw = dict(name=r["label"], line=dict(color=col))
        f_T.add_trace(go.Scatter(x=th["hours"], y=th["T_he"], **kw))
        f_Tf.add_trace(go.Scatter(x=th["hours"], y=th["T_f"], **kw))
        f_dp.add_trace(go.Scatter(x=th["hours"], y=th["dp"], **kw))
        f_N.add_trace(go.Scatter(x=th["hours"], y=th["load"], **kw))
    f_dp.add_hline(y=0, line_dash="dot", line_color="gray")
    for f in (f_T, f_Tf, f_dp, f_N):
        f.update_xaxes(dtick=3, range=[0, 24])
    a, b = st.columns(2)
    a.plotly_chart(f_T, width="stretch"); b.plotly_chart(f_Tf, width="stretch")
    a.plotly_chart(f_dp, width="stretch"); b.plotly_chart(f_N, width="stretch")
    if "thermal" in base and base["thermal"]["dp"].min() < 0:
        st.warning("Baseline loses superpressure (ΔP < 0) during part of the cycle - "
                   "the balloon goes slack and descends at night.")

    # ---- Design map ---------------------------------------------------------
    st.header("Design map: effect of envelope diameter")
    st.caption("Same film, payload, hardware, target and filling rule; helium recomputed for "
               "each diameter. Dashed = current D, dotted = diameter that floats at the target.")
    Ds = np.linspace(max(0.5, design.diameter * 0.5), design.diameter * 2.0, 120)
    rows = []
    for Dx in Ds:
        dx = replace(design, diameter=Dx)
        mb = base_helium(dx, h_t)
        ml = launch_helium(mb, rule, rule_val, dx)
        fx = float_state(dx, ml)
        rows.append(dict(D=Dx, m_env=dx.m_env, m_he=ml, m_tot=dx.m_dry + ml,
                         h=fx["h"] / 1000, dp=fx["dp"]))
    mp = pd.DataFrame(rows)
    f1 = line_fig("Float altitude vs diameter", "Diameter [m]", "Float altitude [km]")
    f1.add_trace(go.Scatter(x=mp.D, y=mp.h, name="Float altitude", line=dict(color=COLORS[0])))
    f1.add_hline(y=h_t / 1000, line_dash="dash", line_color=COLORS[1], annotation_text="target")
    f2 = line_fig("Masses vs diameter", "Diameter [m]", "Mass [kg]")
    for key, name, col in (("m_env", "Envelope", 1), ("m_he", "Launch helium", 2),
                           ("m_tot", "Total system", 3)):
        f2.add_trace(go.Scatter(x=mp.D, y=mp[key], name=name, line=dict(color=COLORS[col])))
    for f in (f1, f2):
        f.add_vline(x=design.diameter, line_dash="dash", line_color="gray")
        if np.isfinite(D_req):
            f.add_vline(x=D_req, line_dash="dot", line_color=COLORS[1])
    a, b = st.columns(2)
    a.plotly_chart(f1, width="stretch"); b.plotly_chart(f2, width="stretch")

    # ---- Model notes --------------------------------------------------------
    st.header("Model notes")
    st.markdown(r"""
**Mass chain.** $V_{max}=\tfrac{\pi}{6}D^3$, $A=\pi D^2$, $m_{env}=A(1+s)\rho_{film}$,
$m_{dry}=m_{env}+m_{payload}+m_{hw}$.

**Base helium (neutral buoyancy at target $h_t$, original logic).**
$\rho_{air}(h_t)\,V_{He}(h_t)=m_{dry}+m_{He,base}$ with $V_{He}=m_{He}RT/(PM_{He})$
$\Rightarrow m_{He,base}=m_{dry}\,\dfrac{M_{He}/M_{air}}{1-M_{He}/M_{air}}\approx0.16\,m_{dry}$.
It scales with every mass component, so payload, hardware or film changes update the helium and
$m_{total}$ automatically. Because helium and air expand together, this mass is the same
for every target altitude. The target altitude decides whether the fixed envelope is big enough,
via $V_{He}(h_t)\le V_{max}$.

**Filling rule.** Lift margin: $m_{He}=m_{He,base}(1+k)$. Free lift: $m_{He}/K=(1+f)(m_{dry}+m_{He})$.
$m_{total}=m_{dry}+m_{He}$.

**Actual float of the fixed envelope.** $\rho_{air}(h_f)\,V_{max}=m_{total}$; isothermal
$\Delta P=P\,(m_{He}/m_{fill}-1)$, $m_{fill}=\rho_{air}V_{max}M_{He}/M_{air}$. The *diameter to float
at target* solves $\rho_{air}(h_t)V_{max}(D)=m_{total}(D)$.

**Ascent.** $(m+C_a\rho V)\ddot h=\rho Vg-mg-\tfrac12\rho\dot h|\dot h|C_dA_p$, $V=\min(V_{He},V_{max})$.

**Thermal.** $m_{env}c_p\dot T_f=\alpha(\tfrac A4I_{dn}+\tfrac A2aI_{dn}\sin\theta)
+\tfrac{\varepsilon A}{2}(q_{IR,E}+\sigma T_{sky}^4)-\varepsilon\sigma AT_f^4-h_{ext}A(T_f-T_{amb})-h_{int}A(T_f-T_{He})$,
$m_{He}c_v\dot T_{He}=h_{int}A(T_f-T_{He})$.

**Limitations.** Spherical shape, no film creep, no helium leakage, altitude fixed during the
thermal cycle, no slack-balloon descent when ΔP < 0, ISA atmosphere.
""")


if __name__ == "__main__":
    main()
