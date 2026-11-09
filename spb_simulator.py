# -*- coding: utf-8 -*-
"""
Super Pressure Balloon (SPB) Simulator  -  fixed-envelope design logic
=====================================================================

Design logic (envelope-driven):
    1. The envelope (jacket) diameter D is a fixed design input.
       -> V_max = (pi/6) D^3,   A = pi D^2  (sphere)
    2. Envelope mass is derived from the area and the film areal density:
       -> m_env = A * (1 + seam_overhead) * film_density
    3. Helium mass is set by the chosen fill rule (free lift, target altitude
       or target superpressure). Total system mass is then a dependent variable:
       -> m_total = m_env + m_payload + m_hardware + m_He

Physics modules:
    * ISA 1976 standard atmosphere (0-86 km, layered closed form)
    * Ascent: 1-DOF ODE with buoyancy, weight, quadratic drag and added mass,
      volume capped at V_max (slack -> fully pressurized transition)
    * Float: equilibrium altitude rho_air(h) * V_max = m_total
    * 24 h thermal cycle: 2-node transient model (film + helium) with direct
      solar, albedo, Earth IR, sky IR, external & internal convection
    * Superpressure dP(t) and film load N = dP * r / 2 (sphere membrane)

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
K_MOLAR = M_HE / M_AIR    # helium/air molar-mass ratio (~0.138)
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
    """Return (T [K], P [Pa], rho [kg/m^3]) at geometric altitude h [m] (0-86 km)."""
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
    """Invert rho(h) = rho_target on 0-50 km. Returns NaN if out of range."""
    f = lambda h: atmosphere(h)[2] - rho_target
    if f(0.0) < 0 or f(50000.0) > 0:
        return np.nan
    return brentq(f, 0.0, 50000.0, xtol=0.1)


# =============================================================================
# Design definition
# =============================================================================
@dataclass(frozen=True)
class Design:
    diameter: float          # envelope diameter [m]
    film_gsm: float          # film areal density [g/m^2]
    seam_pct: float          # seams/tapes/overlap area overhead [%]
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


def helium_from_free_lift(d: Design, fl_pct):
    """Free lift f defined as (gross buoyancy - total weight) / total weight at launch,
    helium at ambient temperature:  m_He*(M_air/M_He) = (1+f)(m_dry + m_He)."""
    f = fl_pct / 100.0
    denom = 1.0 / K_MOLAR - (1.0 + f)
    return (1.0 + f) * d.m_dry / denom


def helium_from_target_altitude(d: Design, h_m):
    """Neutral buoyancy at h with fixed V_max: m_dry + m_He = rho(h) * V_max."""
    _, _, rho = atmosphere(h_m)
    return rho * d.v_max - d.m_dry


def helium_from_target_dp(d: Design, dp_pa):
    """Find the float altitude where the isothermal superpressure equals dp, with
    m_He = rho(h)V - m_dry. Returns (m_He, h) or (nan, nan) if infeasible."""
    def resid(h):
        T, P, rho = atmosphere(h)
        m_he = rho * d.v_max - d.m_dry
        m_fill = rho * d.v_max * K_MOLAR
        return m_he / m_fill - (1.0 + dp_pa / P)
    hs = np.linspace(0.0, 45000.0, 451)
    r = np.array([resid(h) for h in hs])
    idx = np.where(np.sign(r[:-1]) != np.sign(r[1:]))[0]
    if len(idx) == 0:
        return np.nan, np.nan
    i = idx[-1]  # highest crossing = physical float point
    h = brentq(resid, hs[i], hs[i + 1], xtol=0.1)
    return helium_from_target_altitude(d, h), h


def float_state(d: Design, m_he):
    """Equilibrium float altitude and isothermal superpressure (He at ambient T)."""
    m_total = d.m_dry + m_he
    h = altitude_for_density(m_total / d.v_max)
    if np.isnan(h):
        return dict(h=np.nan, T=np.nan, P=np.nan, dp=np.nan, fill_ratio=np.nan)
    T, P, rho = atmosphere(h)
    m_fill = rho * d.v_max * K_MOLAR          # He mass that just fills V_max at ambient
    ratio = m_he / m_fill
    return dict(h=h, T=T, P=P, dp=P * (ratio - 1.0), fill_ratio=ratio)


# =============================================================================
# Ascent model
# =============================================================================
def ascent_rhs(t, y, m_he, m_sys, v_max, cd, c_add):
    h, v = y
    T, P, rho = atmosphere(h)
    V = min(m_he * R_U * T / (P * M_HE), v_max)
    r = (3.0 * V / (4.0 * np.pi)) ** (1.0 / 3.0)
    a_proj = np.pi * r * r
    f_net = rho * V * G0 - m_sys * G0 - 0.5 * rho * v * abs(v) * cd * a_proj
    m_eff = m_sys + c_add * rho * V                     # virtual (added) mass
    acc = f_net / m_eff
    if h <= 0.0 and acc < 0.0 and v <= 0.0:              # resting on the ground
        return [0.0, 0.0]
    return [v, acc]


@st.cache_data(show_spinner=False)
def run_ascent(m_he, m_sys, v_max, cd, c_add, t_end_s):
    sol = solve_ivp(ascent_rhs, (0.0, t_end_s), [0.0, 0.0],
                    args=(m_he, m_sys, v_max, cd, c_add),
                    method="RK45", max_step=10.0, rtol=1e-6, atol=1e-6)
    t, h, v = sol.t, np.maximum(sol.y[0], 0.0), sol.y[1]
    T, P, _ = np.array([atmosphere(x) for x in h]).T
    V_ideal = m_he * R_U * T / (P * M_HE)
    fill = np.minimum(V_ideal / v_max, 1.0) * 100.0
    acc = np.gradient(v, t) if len(t) > 2 else np.zeros_like(t)
    return t, h, v, acc, fill


# =============================================================================
# Thermal model (float, fixed altitude & volume)
# =============================================================================
@dataclass(frozen=True)
class ThermalEnv:
    latitude_deg: float
    day_of_year: int
    alpha: float          # film solar absorptivity
    eps: float            # film IR emissivity
    albedo: float
    tau_atm: float        # transmissivity per unit air mass above the balloon
    q_earth_ir: float     # upwelling Earth/cloud IR [W/m^2]
    t_sky: float          # effective sky temperature [K]
    h_ext: float          # external convection [W/m^2K]
    h_int: float          # internal convection [W/m^2K]
    cp_film: float        # film specific heat [J/kgK]


def solar_elevation(hour, lat_rad, doy):
    decl = np.radians(23.44) * np.sin(2.0 * np.pi * (284 + doy) / 365.0)
    ha = np.radians((hour - 12.0) * 15.0)
    s = np.sin(lat_rad) * np.sin(decl) + np.cos(lat_rad) * np.cos(decl) * np.cos(ha)
    return np.arcsin(np.clip(s, -1.0, 1.0))


def direct_solar(hour, env: ThermalEnv, p_amb):
    """Direct normal irradiance at the balloon [W/m^2] and sin(elevation)."""
    el = solar_elevation(hour, np.radians(env.latitude_deg), env.day_of_year)
    if el <= 0.0:
        return 0.0, 0.0
    el_deg = np.degrees(el)
    airmass = 1.0 / (np.sin(el) + 0.50572 * (el_deg + 6.07995) ** -1.6364)  # Kasten-Young
    airmass *= p_amb / P_SL      # only the air column above the balloon attenuates
    return SOLAR_C * env.tau_atm ** airmass, np.sin(el)


def heat_flows(t_s, T_he, T_f, A, T_amb, p_amb, env: ThermalEnv):
    hour = (t_s / 3600.0) % 24.0
    i_dn, sin_el = direct_solar(hour, env, p_amb)
    q_sun = env.alpha * (i_dn * A / 4.0                       # direct on projected area
                         + env.albedo * i_dn * sin_el * A / 2.0)  # reflected, lower hemisphere
    q_ir_in = env.eps * A / 2.0 * (env.q_earth_ir + SIGMA * env.t_sky ** 4)
    q_ir_out = env.eps * SIGMA * T_f ** 4 * A
    q_conv = env.h_ext * A * (T_f - T_amb)
    q_int = env.h_int * A * (T_f - T_he)
    return q_sun, q_ir_in, q_ir_out, q_conv, q_int


def thermal_rhs(t, y, m_he, m_env, A, T_amb, p_amb, env):
    T_he, T_f = y
    q_sun, q_in, q_out, q_conv, q_int = heat_flows(t, T_he, T_f, A, T_amb, p_amb, env)
    dT_he = q_int / (m_he * CV_HE)                          # constant-volume heating
    dT_f = (q_sun + q_in - q_out - q_conv - q_int) / (m_env * env.cp_film)
    return [dT_he, dT_f]


@st.cache_data(show_spinner=False)
def run_thermal(m_he, m_env, A, T_amb, p_amb, env: ThermalEnv, n_days=3):
    """Integrate n_days from ambient start; return the last (periodic) day."""
    t_end = n_days * 86400.0
    t_eval = np.linspace(t_end - 86400.0, t_end, 289)       # 5-min resolution
    sol = solve_ivp(thermal_rhs, (0.0, t_end), [T_amb, T_amb],
                    args=(m_he, m_env, A, T_amb, p_amb, env),
                    method="LSODA", t_eval=t_eval, max_step=300.0, rtol=1e-6)
    hours = (sol.t - (t_end - 86400.0)) / 3600.0
    return hours, sol.y[0], sol.y[1]


# =============================================================================
# Full case evaluation
# =============================================================================
def evaluate_case(label, d: Design, m_he, env: ThermalEnv, cd, c_add, t_asc_h):
    m_sys = d.m_dry + m_he
    gross = m_he / K_MOLAR                         # gross buoyancy mass at launch (isothermal)
    fl_pct = (gross / m_sys - 1.0) * 100.0
    V0 = m_he * R_U * T_SL / (P_SL * M_HE)
    fs = float_state(d, m_he)
    out = dict(label=label, design=d, m_he=m_he, m_sys=m_sys, fl_pct=fl_pct,
               fl_N=(gross - m_sys) * G0, V0=V0, float=fs, ok=True, msg="")
    if m_he <= 0 or fl_pct <= 0:
        out.update(ok=False, msg="No positive free lift at launch - balloon will not ascend.")
        return out
    out["ascent"] = run_ascent(m_he, m_sys, d.v_max, cd, c_add, t_asc_h * 3600.0)
    if np.isnan(fs["h"]):
        out.update(ok=False, msg="No float equilibrium found below 50 km.")
        return out
    hrs, T_he, T_f = run_thermal(m_he, d.m_env, d.area, fs["T"], fs["P"], env)
    # Internal pressure if the helium fills V_max; otherwise balloon is slack (dP = 0)
    P_in = m_he * R_U * T_he / (d.v_max * M_HE)
    dp = P_in - fs["P"]
    out["thermal"] = dict(hours=hrs, T_he=T_he, T_f=T_f, dp=dp,
                          load=np.maximum(dp, 0.0) * d.radius / 2.0)
    return out


def summary_row(c):
    d, fs = c["design"], c["float"]
    row = {
        "Case": c["label"], "D [m]": d.diameter, "V_max [m³]": d.v_max, "Area [m²]": d.area,
        "m_env [kg]": d.m_env, "m_He [kg]": c["m_he"], "m_total [kg]": c["m_sys"],
        "Free lift [%]": c["fl_pct"], "Free lift [N]": c["fl_N"], "V_launch [m³]": c["V0"],
        "Float alt [km]": fs["h"] / 1000.0, "ΔP isothermal [Pa]": fs["dp"],
    }
    if "ascent" in c:
        t, h, v, *_ = c["ascent"]
        h_f = fs["h"] if not np.isnan(fs["h"]) else h[-1]
        reach = np.where(h >= 0.98 * h_f)[0]
        row["Time to 98% float [min]"] = t[reach[0]] / 60.0 if len(reach) else np.nan
        mask = (h > 100) & (h < 0.95 * h_f) & (v > 0)
        row["Mean ascent rate [m/s]"] = v[mask].mean() if mask.any() else np.nan
    if "thermal" in c:
        th = c["thermal"]
        row.update({
            "ΔP min [Pa]": th["dp"].min(), "ΔP max [Pa]": th["dp"].max(),
            "T_He min [K]": th["T_he"].min(), "T_He max [K]": th["T_he"].max(),
            "Max superheat [K]": (th["T_he"] - fs["T"]).max(),
            "Max film load [N/m]": th["load"].max(),
        })
    return row


# =============================================================================
# Streamlit UI
# =============================================================================
COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#a16207"]


def sidebar_inputs():
    sb = st.sidebar
    sb.title("🎈 SPB Inputs")

    with sb.expander("Envelope (jacket) geometry", expanded=True):
        D = st.number_input("Envelope diameter D [m]", 0.5, 100.0, 4.6, 0.05,
                            help="Fixed design diameter. V_max = πD³/6, A = πD².")
        gsm = st.number_input("Film areal density [g/m²]", 5.0, 300.0, 69.0, 1.0)
        seam = st.number_input("Seam / tape overhead [% of area]", 0.0, 50.0, 0.0, 0.5,
                               help="Extra film mass from overlaps, load tapes, end fittings.")

    with sb.expander("Masses", expanded=True):
        payload = st.number_input("Payload mass [kg]", 0.0, 500.0, 0.0, 0.1)
        hardware = st.number_input("Safety / hardware mass [kg]", 0.0, 200.0, 5.0, 0.1,
                                   help="Valves, rigging, cutter, parachute, etc.")

    with sb.expander("Helium fill rule", expanded=True):
        mode = st.radio("Size helium mass by", ["Launch free lift", "Target float altitude",
                                                "Target superpressure at float"], index=0)
        fl = h_t = dp_t = None
        if mode == "Launch free lift":
            fl = st.number_input("Free lift [% of total weight]", 0.5, 200.0, 20.0, 0.5,
                                 help="(Buoyancy − weight)/weight at launch. With He at ambient "
                                      "temperature this also equals the isothermal ΔP/P at float.")
        elif mode == "Target float altitude":
            h_t = st.number_input("Target float altitude [km]", 1.0, 45.0, 14.0, 0.1) * 1000.0
        else:
            dp_t = st.number_input("Target isothermal ΔP at float [Pa]", 1.0, 50000.0, 2500.0, 50.0)

    with sb.expander("Sensitivity cases"):
        he_var = st.number_input("± helium mass [%]", 0.0, 50.0, 5.0, 0.5)
        d_var = st.number_input("± diameter (manufacturing) [%]", 0.0, 20.0, 2.0, 0.5,
                                help="Changes V_max, area AND envelope mass together.")
        gsm_var = st.number_input("± film density [%]", 0.0, 30.0, 0.0, 1.0)

    with sb.expander("Ascent dynamics"):
        cd = st.number_input("Drag coefficient C_d", 0.1, 2.0, 0.47, 0.01)
        c_add = st.number_input("Added-mass coefficient", 0.0, 1.0, 0.5, 0.05,
                                help="0.5 for a sphere; set 0 to ignore virtual mass.")
        t_asc = st.number_input("Ascent simulation time [h]", 0.5, 12.0, 4.0, 0.5)

    with sb.expander("Thermal & radiation environment"):
        lat = st.number_input("Latitude [°]", -90.0, 90.0, 32.0, 0.1)
        doy = st.slider("Day of year", 1, 365, 355, help="Sets solar declination (355 ≈ Dec 21).")
        alpha = st.slider("Film solar absorptivity α", 0.01, 1.0, 0.40, 0.01)
        eps = st.slider("Film IR emissivity ε", 0.01, 1.0, 0.85, 0.01)
        albedo = st.slider("Surface / cloud albedo", 0.0, 1.0, 0.30, 0.01)
        tau = st.slider("Atmospheric transmissivity τ (per air mass)", 0.3, 1.0, 0.85, 0.01)
        q_ir = st.number_input("Upwelling Earth IR [W/m²]", 50.0, 400.0, 230.0, 5.0)
        t_sky = st.number_input("Effective sky temperature [K]", 3.0, 300.0, 216.65, 1.0)
        h_ext = st.slider("External convection h_ext [W/m²K]", 0.1, 10.0, 0.6, 0.1)
        h_int = st.slider("Internal convection h_int [W/m²K]", 0.1, 5.0, 2.0, 0.1)
        cp_f = st.number_input("Film specific heat [J/kgK]", 500.0, 3000.0, 1500.0, 50.0)

    design = Design(D, gsm, seam, payload, hardware)
    env = ThermalEnv(lat, int(doy), alpha, eps, albedo, tau, q_ir, t_sky, h_ext, h_int, cp_f)
    fill = dict(mode=mode, fl=fl, h_t=h_t, dp_t=dp_t)
    var = dict(he=he_var, d=d_var, gsm=gsm_var)
    dyn = dict(cd=cd, c_add=c_add, t_asc=t_asc)
    return design, env, fill, var, dyn


def size_helium(d, fill):
    if fill["mode"] == "Launch free lift":
        return helium_from_free_lift(d, fill["fl"])
    if fill["mode"] == "Target float altitude":
        return helium_from_target_altitude(d, fill["h_t"])
    return helium_from_target_dp(d, fill["dp_t"])[0]


def build_cases(base, m_he, var):
    cases = [("Baseline", base, m_he)]
    if var["he"] > 0:
        cases += [(f"+{var['he']:g}% He", base, m_he * (1 + var["he"] / 100)),
                  (f"−{var['he']:g}% He", base, m_he * (1 - var["he"] / 100))]
    if var["d"] > 0:   # same helium, different envelope -> different V_max, A, m_env
        for s in (+1, -1):
            dd = replace(base, diameter=base.diameter * (1 + s * var["d"] / 100))
            cases.append((f"{'+' if s > 0 else '−'}{var['d']:g}% D", dd, m_he))
    if var["gsm"] > 0:
        for s in (+1, -1):
            dd = replace(base, film_gsm=base.film_gsm * (1 + s * var["gsm"] / 100))
            cases.append((f"{'+' if s > 0 else '−'}{var['gsm']:g}% film", dd, m_he))
    return cases


def line_fig(title, xt, yt):
    fig = go.Figure()
    fig.update_layout(title=title, xaxis_title=xt, yaxis_title=yt, hovermode="x unified",
                      legend_title_text="Case", margin=dict(l=10, r=10, t=50, b=10), height=420)
    return fig


def main():
    st.set_page_config(page_title="SPB Simulator", page_icon="🎈", layout="wide",
                       initial_sidebar_state="expanded")
    design, env, fill, var, dyn = sidebar_inputs()

    st.title("Super Pressure Balloon Simulator")
    st.caption("Fixed-envelope logic: diameter → V_max & area → envelope mass → "
               "helium (fill rule) → total system mass → float altitude & superpressure.")

    m_he = size_helium(design, fill)
    if not np.isfinite(m_he) or m_he <= 0:
        st.error("This envelope cannot satisfy the selected fill rule: the dry mass "
                 f"({design.m_dry:.2f} kg) is too heavy for D = {design.diameter:g} m at that "
                 "condition. Increase the diameter, lower the film density / masses, or relax the target.")
        st.stop()

    base = evaluate_case("Baseline", design, m_he, env, dyn["cd"], dyn["c_add"], dyn["t_asc"])
    fs = base["float"]

    # ---- Key metrics -------------------------------------------------------
    c = st.columns(6)
    c[0].metric("V_max", f"{design.v_max:,.1f} m³")
    c[1].metric("Envelope area", f"{design.area:,.1f} m²")
    c[2].metric("Envelope mass", f"{design.m_env:,.2f} kg")
    c[3].metric("Helium mass", f"{m_he:,.3f} kg")
    c[4].metric("Total system mass", f"{base['m_sys']:,.2f} kg")
    c[5].metric("Float altitude", f"{fs['h'] / 1000:,.2f} km" if np.isfinite(fs["h"]) else "—")
    c = st.columns(6)
    c[0].metric("Launch free lift", f"{base['fl_pct']:,.1f} %", f"{base['fl_N']:,.1f} N",
                delta_color="off")
    c[1].metric("Launch volume", f"{base['V0']:,.1f} m³", f"{base['V0'] / design.v_max * 100:.1f}% of V_max",
                delta_color="off")
    c[2].metric("ΔP isothermal", f"{fs['dp']:,.0f} Pa" if np.isfinite(fs["dp"]) else "—")
    if "thermal" in base:
        th = base["thermal"]
        c[3].metric("ΔP 24h min / max", f"{th['dp'].min():,.0f} / {th['dp'].max():,.0f} Pa")
        c[4].metric("Max superheat", f"{(th['T_he'] - fs['T']).max():,.1f} K")
        c[5].metric("Max film load", f"{th['load'].max():,.0f} N/m",
                    help="Membrane line load of a sphere, N = ΔP·r/2.")
        if th["dp"].min() < 0:
            st.warning("⚠️ Superpressure is lost during part of the cycle (ΔP < 0): the balloon goes "
                       "slack and will descend at night. Add helium or reduce night cooling.")
    if not base["ok"]:
        st.error(base["msg"])

    # ---- Run sensitivity cases --------------------------------------------
    cases = build_cases(design, m_he, var)
    with st.spinner("Running ascent and 24 h thermal simulations…"):
        results = [base] + [evaluate_case(lbl, d, mh, env, dyn["cd"], dyn["c_add"], dyn["t_asc"])
                            for lbl, d, mh in cases[1:]]

    tab_sum, tab_asc, tab_th, tab_map, tab_model = st.tabs(
        ["📊 Summary", "🚀 Ascent", "☀️ 24 h thermal cycle", "🗺️ Design map", "📘 Model notes"])

    with tab_sum:
        df = pd.DataFrame([summary_row(r) for r in results]).set_index("Case")
        st.dataframe(df.style.format("{:,.2f}", na_rep="—"), width="stretch")
        st.download_button("Download summary (CSV)", df.to_csv().encode(), "spb_summary.csv",
                           "text/csv")
        for r in results:
            if not r["ok"]:
                st.warning(f"{r['label']}: {r['msg']}")

    with tab_asc:
        f_h = line_fig("Altitude vs time", "Time [min]", "Altitude [km]")
        f_v = line_fig("Vertical velocity vs time", "Time [min]", "Velocity [m/s]")
        f_a = line_fig("Acceleration vs time", "Time [min]", "Acceleration [m/s²]")
        f_fill = line_fig("Envelope fill (V / V_max)", "Altitude [km]", "Fill [%]")
        for r, col in zip(results, COLORS * 2):
            if "ascent" not in r:
                continue
            t, h, v, a, fill_pct = r["ascent"]
            kw = dict(name=r["label"], line=dict(color=col))
            f_h.add_trace(go.Scatter(x=t / 60, y=h / 1000, **kw))
            f_v.add_trace(go.Scatter(x=t / 60, y=v, **kw))
            f_a.add_trace(go.Scatter(x=t / 60, y=a, **kw))
            f_fill.update_layout(hovermode="closest")
            f_fill.add_trace(go.Scatter(x=h / 1000, y=fill_pct, **kw))
        a, b = st.columns(2)
        a.plotly_chart(f_h, width="stretch"); b.plotly_chart(f_v, width="stretch")
        a.plotly_chart(f_a, width="stretch"); b.plotly_chart(f_fill, width="stretch")

    with tab_th:
        st.caption("Third day of a 3-day integration (periodic steady state). "
                   "Time is local solar time; altitude and volume held at float.")
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

    with tab_map:
        st.caption("How the chosen fill rule plays out across envelope diameters "
                   "(same film, payload and hardware). Dashed line = current design.")
        Ds = np.linspace(max(0.5, design.diameter * 0.5), design.diameter * 2.0, 120)
        rows = []
        for Dx in Ds:
            dx = replace(design, diameter=Dx)
            mh = size_helium(dx, fill)
            fsx = float_state(dx, mh) if np.isfinite(mh) and mh > 0 else None
            rows.append(dict(D=Dx, m_env=dx.m_env, m_he=mh if fsx else np.nan,
                             m_tot=dx.m_dry + mh if fsx else np.nan,
                             h=fsx["h"] / 1000 if fsx else np.nan,
                             dp=fsx["dp"] if fsx else np.nan))
        mp = pd.DataFrame(rows)
        f1 = line_fig("Float altitude vs diameter", "Diameter [m]", "Float altitude [km]")
        f1.add_trace(go.Scatter(x=mp.D, y=mp.h, name="Float altitude", line=dict(color=COLORS[0])))
        f2 = line_fig("Masses vs diameter", "Diameter [m]", "Mass [kg]")
        f2.add_trace(go.Scatter(x=mp.D, y=mp.m_env, name="Envelope", line=dict(color=COLORS[1])))
        f2.add_trace(go.Scatter(x=mp.D, y=mp.m_he, name="Helium", line=dict(color=COLORS[2])))
        f2.add_trace(go.Scatter(x=mp.D, y=mp.m_tot, name="Total", line=dict(color=COLORS[3])))
        f3 = line_fig("Isothermal superpressure vs diameter", "Diameter [m]", "ΔP [Pa]")
        f3.add_trace(go.Scatter(x=mp.D, y=mp.dp, name="ΔP", line=dict(color=COLORS[4])))
        for f in (f1, f2, f3):
            f.add_vline(x=design.diameter, line_dash="dash", line_color="gray")
        a, b = st.columns(2)
        a.plotly_chart(f1, width="stretch"); b.plotly_chart(f2, width="stretch")
        a.plotly_chart(f3, width="stretch")

    with tab_model:
        st.markdown(r"""
**Geometry & mass (sphere).** $V_{max}=\tfrac{\pi}{6}D^3$, $A=\pi D^2$,
$m_{env}=A\,(1+s)\,\rho_{film}$, $m_{total}=m_{env}+m_{payload}+m_{hw}+m_{He}$.

**Helium fill rules** (helium at ambient temperature):
* *Free lift f:* $m_{He}\,M_{air}/M_{He}=(1+f)\,(m_{dry}+m_{He})$.
* *Target altitude h:* $m_{He}=\rho_{air}(h)\,V_{max}-m_{dry}$.
* *Target superpressure:* finds $h$ where $m_{He}/m_{fill}(h)=1+\Delta P/P(h)$.

**Float.** A fully pressurized SPB floats where $\rho_{air}(h)\,V_{max}=m_{total}$.
Isothermal superpressure $\Delta P = P\,(m_{He}/m_{fill}-1)$, with $m_{fill}=\rho_{air}V_{max}M_{He}/M_{air}$
(so $\Delta P/P$ equals the launch free-lift fraction).

**Ascent.** $(m+C_a\rho V)\,\ddot h=\rho V g-mg-\tfrac12\rho\dot h|\dot h|C_dA_p$,
$V=\min(V_{gas},V_{max})$, helium assumed at ambient temperature.

**Thermal (film + helium nodes).**
$m_{env}c_p\dot T_f=\alpha(\tfrac{A}{4}I_{dn}+\tfrac{A}{2}a\,I_{dn}\sin\theta)+\tfrac{\varepsilon A}{2}(q_{IR,E}+\sigma T_{sky}^4)
-\varepsilon\sigma A T_f^4-h_{ext}A(T_f-T_{amb})-h_{int}A(T_f-T_{He})$;
$m_{He}c_v\dot T_{He}=h_{int}A(T_f-T_{He})$. $I_{dn}=S\,\tau^{m\,P/P_0}$ (Kasten–Young air mass).

**Limitations.** Spherical shape (no pumpkin lobing), no film creep/elasticity,
no helium diffusion loss, fixed float altitude during the thermal cycle,
no slack-balloon descent dynamics when ΔP < 0, ISA atmosphere only.
""")


if __name__ == "__main__":
    main()
