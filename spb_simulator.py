# -*- coding: utf-8 -*-
"""
Super Pressure Balloon (SPB) Simulator
======================================

Single-page Streamlit app built on the original SPB model (ISA atmosphere,
RK45 ascent ODE, transient film/helium thermal ODE with the original
solar / IR / convection formulation), with a fixed, user-defined envelope.

Core logic
----------
1. Envelope (user input D):
       V_max = pi D^3 / 6        A = pi D^2        m_env = A * film_density
2. Helium for neutral buoyancy at the target altitude h_t.
   A fully pressurized SPB has volume V_max, so neutral buoyancy means
       rho_air(h_t) * V_max = m_env + m_payload + m_He
   =>  m_He = rho_air(h_t) * V_max - m_env - m_payload
       m_total = m_env + m_payload + m_He
3. Simulations: ascent (helium at ambient T, volume capped at V_max) and a
   24 h thermal cycle at float giving T_He(t), dP(t) and film load N = dP r / 2.
4. Operational checks: envelope pressurized at target, launch free lift,
   ascent rate, target reached, no slack at night (dP_min > 0),
   film load below allowable / safety factor (burst).

Standalone file - deployable on Streamlit Community Cloud with requirements:
streamlit, numpy, scipy, pandas, plotly, jinja2.
"""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# =============================================================================
# Constants
# =============================================================================
G = 9.80665               # gravitational acceleration [m/s^2]
R_U = 8.314462618         # universal gas constant [J/(mol K)]
M_AIR = 0.0289644         # molar mass of air [kg/mol]
M_HE = 0.0040026          # molar mass of helium [kg/mol]
SIGMA = 5.670374419e-8    # Stefan-Boltzmann constant [W/(m^2 K^4)]
CV_HE = 3116.0            # helium specific heat at constant volume [J/(kg K)]
SOLAR_C = 1361.0          # solar constant [W/m^2]
T_SL, P_SL = 288.15, 101325.0
K_HE = M_HE / M_AIR       # helium-to-air molar mass ratio (~0.138)

COLORS = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692"]
PASS, WARN, FAIL = "✅ Pass", "⚠️ Warning", "❌ Fail"


# =============================================================================
# ISA atmosphere (U.S. Standard Atmosphere 1976)
# =============================================================================
_LAYERS = [  # base altitude [m], base temperature [K], lapse rate [K/m]
    (0.0, 288.15, -0.0065),
    (11000.0, 216.65, 0.0),
    (20000.0, 216.65, 0.0010),
    (32000.0, 228.65, 0.0028),
    (47000.0, 270.65, 0.0),
    (51000.0, 270.65, -0.0028),
    (71000.0, 214.65, -0.0020),
]
_GMR = G * M_AIR / R_U


def _base_pressures():
    p = [P_SL]
    for i in range(1, len(_LAYERS)):
        h0, T0, L = _LAYERS[i - 1]
        dh = _LAYERS[i][0] - h0
        p.append(p[-1] * (np.exp(-_GMR * dh / T0) if L == 0.0
                          else ((T0 + L * dh) / T0) ** (-_GMR / L)))
    return p


_P_BASE = _base_pressures()


def atmosphere(h):
    """(T [K], P [Pa], rho [kg/m^3]) at altitude h [m], valid 0-86 km."""
    h = min(max(float(h), 0.0), 86000.0)
    i = len(_LAYERS) - 1
    while h < _LAYERS[i][0]:
        i -= 1
    h0, T0, L = _LAYERS[i]
    T = T0 + L * (h - h0)
    P = (_P_BASE[i] * np.exp(-_GMR * (h - h0) / T0) if L == 0.0
         else _P_BASE[i] * (T / T0) ** (-_GMR / L))
    return T, P, P * M_AIR / (R_U * T)


def altitude_for_density(rho):
    """Altitude [m] where air density equals rho (0-50 km), NaN if outside."""
    f = lambda h: atmosphere(h)[2] - rho
    if f(0.0) < 0.0 or f(50000.0) > 0.0:
        return np.nan
    return brentq(f, 0.0, 50000.0, xtol=0.1)


# =============================================================================
# Envelope & helium sizing
# =============================================================================
@dataclass(frozen=True)
class Envelope:
    diameter: float      # [m]
    film_gsm: float      # [g/m^2]
    payload: float       # payload + hardware additions [kg]

    @property
    def radius(self):
        return self.diameter / 2.0

    @property
    def v_max(self):
        return np.pi * self.diameter ** 3 / 6.0

    @property
    def area(self):
        return np.pi * self.diameter ** 2

    @property
    def m_env(self):
        return self.area * self.film_gsm / 1000.0

    @property
    def m_dry(self):
        return self.m_env + self.payload


def size_for_target(env: Envelope, h_t):
    """Neutral-buoyancy helium at the target for the fixed envelope."""
    T, P, rho = atmosphere(h_t)
    m_total = rho * env.v_max                   # displaced air mass at target
    m_he = m_total - env.m_dry                  # required helium
    m_fill = env.v_max * P * M_HE / (R_U * T)   # helium that just fills V_max at target
    return dict(T=T, P=P, rho=rho, m_total=m_total, m_he=m_he, m_fill=m_fill,
                payload_max=m_total - m_fill - env.m_env,
                dp_std=P * (m_he / m_fill - 1.0))


def min_diameter(env: Envelope, h_t):
    """Smallest D that carries the payload to h_t with zero superpressure."""
    f = lambda D: size_for_target(replace(env, diameter=D), h_t)["payload_max"] - env.payload
    Ds = np.geomspace(0.2, 300.0, 300)
    r = np.array([f(D) for D in Ds])
    idx = np.where((r[:-1] < 0) & (r[1:] >= 0))[0]
    return brentq(f, Ds[idx[0]], Ds[idx[0] + 1], xtol=1e-4) if len(idx) else np.nan


def float_state(env: Envelope, m_he):
    """Equilibrium float of a given envelope + helium (helium at ambient T)."""
    h = altitude_for_density((env.m_dry + m_he) / env.v_max)
    if np.isnan(h):
        return dict(h=np.nan, T=np.nan, P=np.nan, dp_std=np.nan)
    T, P, _ = atmosphere(h)
    m_fill = env.v_max * P * M_HE / (R_U * T)
    return dict(h=h, T=T, P=P, dp_std=P * (m_he / m_fill - 1.0))


# =============================================================================
# Ascent model
# =============================================================================
def ascent_ode(t, y, m_total, m_he, v_max, cd, c_add):
    h, v = y
    T, P, rho = atmosphere(h)
    V = min(m_he * R_U * T / (P * M_HE), v_max)
    r = (3.0 * V / (4.0 * np.pi)) ** (1.0 / 3.0)
    f_lift = G * V * rho
    f_drag = 0.5 * rho * v * abs(v) * cd * np.pi * r ** 2
    a = (f_lift - m_total * G - f_drag) / (m_total + c_add * rho * V)
    if h <= 0.0 and v <= 0.0 and a < 0.0:      # sitting on the ground
        return [0.0, 0.0]
    return [v, a]


@st.cache_data(show_spinner=False)
def run_ascent(m_total, m_he, v_max, cd, c_add, t_end):
    sol = solve_ivp(ascent_ode, (0.0, t_end), [0.0, 0.1],
                    args=(m_total, m_he, v_max, cd, c_add),
                    method="RK45", max_step=15.0)
    t, h, v = sol.t, np.maximum(sol.y[0], 0.0), sol.y[1]
    fill = np.array([min(m_he * R_U * atmosphere(x)[0] / (atmosphere(x)[1] * M_HE) / v_max, 1.0)
                     for x in h]) * 100.0
    a = np.gradient(v, t) if len(t) > 2 else np.zeros_like(t)
    return t, h, v, a, fill


# =============================================================================
# 24 h thermal model (original formulation)
# =============================================================================
@dataclass(frozen=True)
class Thermal:
    latitude_deg: float
    declination_deg: float
    alpha: float
    eps: float
    albedo: float
    tau_atm: float
    t_sky: float
    h_conv: float
    h_int: float
    cp_film: float
    spinup_days: int


def thermal_ode(t, y, m_he, m_env, A, T_amb, th: Thermal):
    T_he, T_f = y
    hr = (t / 3600.0) % 24.0
    lat, dec = np.radians(th.latitude_deg), np.radians(th.declination_deg)
    ha = np.radians((hr - 12.0) * 15.0)
    sin_alt = np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(ha)
    alt = np.arcsin(np.clip(sin_alt, -1.0, 1.0))

    q_solar = 0.0
    if alt > 0.0:
        airmass = 1.0 / (np.sin(alt) + 0.50572 * (np.degrees(alt) + 6.07995) ** -1.6364)
        i_dn = SOLAR_C * th.tau_atm ** airmass
        i_diff = 0.1 * SOLAR_C * sin_alt
        ghi = i_dn * sin_alt + i_diff
        q_solar = th.alpha * (0.25 * i_dn + i_diff + th.albedo * ghi)

    Q_sun = q_solar * A
    Q_sky = th.eps * SIGMA * th.t_sky ** 4 * A
    Q_emit = th.eps * SIGMA * T_f ** 4 * A
    Q_conv = th.h_conv * A * (T_f - T_amb)
    Q_int = th.h_int * A * (T_f - T_he)
    return [Q_int / (m_he * CV_HE),
            (Q_sun + Q_sky - Q_emit - Q_conv - Q_int) / (m_env * th.cp_film)]


@st.cache_data(show_spinner=False)
def run_thermal(m_he, m_env, A, T_amb, th: Thermal):
    """Start at midnight with T_He = T_film = T_amb; optional spin-up days;
    returns the final 24 h at 15-min resolution."""
    t0 = th.spinup_days * 86400.0
    t_eval = np.linspace(t0, t0 + 86400.0, 97)
    sol = solve_ivp(thermal_ode, (0.0, t0 + 86400.0), [T_amb, T_amb],
                    args=(m_he, m_env, A, T_amb, th), method="LSODA",  # stiff film node
                    t_eval=t_eval, max_step=900.0)
    T_he = np.clip(np.interp(t_eval, sol.t, sol.y[0]), 150.0, 400.0)
    T_f = np.interp(t_eval, sol.t, sol.y[1])
    return (t_eval - t0) / 3600.0, T_he, T_f


# =============================================================================
# Case simulation
# =============================================================================
def simulate_case(label, env: Envelope, m_he, h_t, th, cd, c_add, t_asc_h):
    m_total = env.m_dry + m_he
    V0 = m_he * R_U * T_SL / (P_SL * M_HE)
    lift_N = G * V0 * atmosphere(0.0)[2]
    res = dict(label=label, env=env, m_he=m_he, m_total=m_total, V0=V0, lift_N=lift_N,
               free_lift_pct=(lift_N / (m_total * G) - 1.0) * 100.0,
               flt=float_state(env, m_he), ok=True, msg="")
    if m_he <= 0 or res["free_lift_pct"] <= 0:
        res.update(ok=False, msg="No positive free lift at launch.")
        return res
    t, h, v, a, fill = run_ascent(m_total, m_he, env.v_max, cd, c_add, t_asc_h * 3600.0)
    res["ascent"] = dict(t=t, h=h, v=v, a=a, fill=fill)
    h_ref = res["flt"]["h"] if np.isfinite(res["flt"]["h"]) else h_t
    reach = np.where(h >= 0.98 * h_ref)[0]
    sel = (h > 100.0) & (h < 0.98 * h_ref) & (v > 0.1)
    res["t_reach_min"] = t[reach[0]] / 60.0 if len(reach) else np.nan
    res["v_avg"] = float(np.mean(v[sel])) if sel.any() else np.nan
    res["h_end"] = h[-1]

    # thermal cycle at the float altitude (volume = V_max)
    h_f = res["flt"]["h"] if np.isfinite(res["flt"]["h"]) else h[-1]
    T_amb, P_amb, _ = atmosphere(h_f)
    hrs, T_he, T_f = run_thermal(m_he, env.m_env, env.area, T_amb, th)
    dp = m_he * R_U * T_he / (env.v_max * M_HE) - P_amb
    res["thermal"] = dict(hrs=hrs, T_he=T_he, T_f=T_f, dp=dp, T_amb=T_amb,
                          load=np.maximum(dp, 0.0) * env.radius / 2.0)
    return res


def summary_row(r, h_t):
    e, f = r["env"], r["flt"]
    row = dict(Case=r["label"], He_kg=r["m_he"], Casing_kg=e.m_env, Total_kg=r["m_total"],
               Weight_N=r["m_total"] * G, D_m=e.diameter, Vmax_m3=e.v_max, V0_m3=r["V0"],
               Lift_N=r["lift_N"], FreeLift_pct=r["free_lift_pct"],
               Alt_km=f["h"] / 1000.0, dAlt_km=(f["h"] - h_t) / 1000.0,
               Vavg_mps=r.get("v_avg", np.nan), tFloat_min=r.get("t_reach_min", np.nan),
               dP_std_Pa=f["dp_std"])
    if "thermal" in r:
        th = r["thermal"]
        row.update(dP_min_Pa=th["dp"].min(), dP_max_Pa=th["dp"].max(),
                   T_night_K=th["T_he"][0], T_max_K=th["T_he"].max(),
                   Load_max_Nm=th["load"].max())
    return row


def helium_window(env: Envelope, h_t, T_he_min, T_he_max, n_lim):
    """Helium range at the target: no slack at T_min, load <= n_lim at T_max."""
    T, P, _ = atmosphere(h_t)
    m_lo = P * env.v_max * M_HE / (R_U * T_he_min)
    m_hi = (P + 2.0 * n_lim / env.radius) * env.v_max * M_HE / (R_U * T_he_max)
    return m_lo, m_hi


def feasibility(env, h_t, sz, base, lim, m_lo, m_hi):
    rows = [("Envelope pressurized at target", f"{sz['m_he']:.3f} kg He",
             f"≥ {sz['m_fill']:.3f} kg", PASS, "Helium fills V_max at the target altitude.")]
    fl = base["free_lift_pct"]
    rows.append(("Launch free lift", f"{fl:.1f} %", f"≥ {lim['fl_min']:.0f} %",
                 PASS if fl >= lim["fl_min"] else WARN,
                 "OK" if fl >= lim["fl_min"] else "Low free lift - slow, wind-sensitive launch."))
    va = base.get("v_avg", np.nan)
    if np.isfinite(va):
        ok = lim["v_lo"] <= va <= lim["v_hi"]
        rows.append(("Mean ascent rate", f"{va:.2f} m/s", f"{lim['v_lo']:.1f}–{lim['v_hi']:.1f} m/s",
                     PASS if ok else WARN,
                     "OK" if ok else ("Too slow - long tropopause crossing, large drift."
                                      if va < lim["v_lo"] else
                                      "Too fast - high dynamic loads and overshoot at full inflation.")))
    reached = np.isfinite(base.get("t_reach_min", np.nan))
    rows.append(("Target reached", f"{base['h_end']/1000:.2f} km at end of ascent sim",
                 f"≥ {0.98*h_t/1000:.2f} km", PASS if reached else FAIL,
                 f"Reached in {base['t_reach_min']:.0f} min." if reached
                 else "Not reached - extend the ascent time or check lift."))
    th = base["thermal"]
    dpmin, nmax = th["dp"].min(), th["load"].max()
    n_lim = lim["n_allow"] / lim["sf"]
    rows.append(("Night superpressure (ΔP min)", f"{dpmin:,.0f} Pa", "> 0 Pa",
                 PASS if dpmin > 0 else FAIL,
                 "Stays pressurized through the night." if dpmin > 0 else
                 f"Envelope goes slack at night → altitude loss. Needs ≥ {m_lo:.3f} kg He."))
    status = PASS if nmax <= n_lim else (WARN if nmax <= lim["n_allow"] else FAIL)
    rows.append(("Peak film load (day)", f"{nmax:,.0f} N/m",
                 f"≤ {n_lim:,.0f} N/m (allowable / SF)", status,
                 {PASS: "OK", WARN: "Below allowable but inside the safety margin.",
                  FAIL: f"Over-pressure / burst risk. Keep He ≤ {m_hi:.3f} kg."}[status]))
    return rows


# =============================================================================
# UI
# =============================================================================
def sidebar():
    sb = st.sidebar
    sb.title("🎈 SPB Inputs")

    sb.subheader("1 · Envelope")
    D = sb.number_input("Envelope diameter D [m]", 0.5, 100.0, 4.6, 0.05,
                        help="V_max = πD³/6, A = πD²")
    gsm = sb.number_input("Fabric areal density [g/m²]", 5.0, 300.0, 69.0, 1.0)

    sb.subheader("2 · Mission")
    h_t = sb.number_input("Target float altitude [km]", 1.0, 45.0, 14.0, 0.1) * 1000.0
    payload = sb.number_input("Payload / hardware additions [kg]", 0.0, 1000.0, 5.0, 0.1)

    sb.subheader("3 · Structural & flight limits")
    n_allow = sb.number_input("Film allowable line load [N/m]", 100.0, 200000.0, 15000.0, 100.0,
                              help="Burst strength of film/seams per unit width.")
    sf = sb.number_input("Safety factor", 1.0, 5.0, 1.5, 0.05)
    fl_min = sb.number_input("Min launch free lift [%]", 0.0, 100.0, 10.0, 1.0)
    v_lo, v_hi = sb.slider("Acceptable mean ascent rate [m/s]", 0.5, 15.0, (2.0, 7.0), 0.1)

    sb.subheader("4 · Sensitivity cases")
    he_var = sb.number_input("± helium mass [%]", 0.0, 50.0, 5.0, 0.5)
    vol_var = sb.number_input("± max volume [%]", 0.0, 50.0, 5.0, 0.5,
                              help="Envelope tolerance; D, area and envelope mass follow V_max.")

    sb.subheader("5 · Ascent")
    cd = sb.number_input("Drag coefficient C_d", 0.1, 2.0, 0.47, 0.01)
    c_add = sb.number_input("Added-mass coefficient", 0.0, 1.0, 0.0, 0.05,
                            help="0 = original model; 0.5 = sphere virtual mass.")
    t_asc = sb.number_input("Ascent simulation time [h]", 0.5, 12.0, 5.0, 0.5)

    sb.subheader("6 · Thermal")
    cp_film = sb.number_input("Film specific heat [J/kgK]", 1000.0, 3000.0, 1500.0, 50.0)
    h_int = sb.slider("Internal convection h_internal [W/m²K]", 0.1, 5.0, 2.0, 0.1)
    h_conv = sb.slider("External convection H_CONV [W/m²K]", 0.1, 10.0, 0.6, 0.1)
    spin = sb.number_input("Thermal spin-up days", 0, 5, 0, 1,
                           help="0 = original (24 h from ambient at midnight). "
                                ">0 = show a periodic, settled day.")

    sb.subheader("7 · Radiation")
    lat = sb.number_input("Latitude [°]", -90.0, 90.0, 32.0, 0.1)
    dec = sb.number_input("Solar declination [°]", -23.44, 23.44, -23.44, 0.5,
                          help="-23.44 = Dec solstice, 0 = equinox, +23.44 = Jun solstice.")
    alpha = sb.slider("Solar absorptivity α", 0.01, 1.0, 0.40, 0.01)
    eps = sb.slider("Thermal emissivity ε", 0.01, 1.0, 0.85, 0.01)
    albedo = sb.slider("Surface albedo", 0.0, 1.0, 0.70, 0.01)
    tau = sb.slider("Atmospheric transmissivity τ", 0.1, 1.0, 0.95, 0.01)
    t_sky = sb.number_input("Sky temperature [K]", 3.0, 300.0, 216.65, 1.0)

    env = Envelope(D, gsm, payload)
    th = Thermal(lat, dec, alpha, eps, albedo, tau, t_sky, h_conv, h_int, cp_film, int(spin))
    lim = dict(n_allow=n_allow, sf=sf, fl_min=fl_min, v_lo=v_lo, v_hi=v_hi)
    return env, h_t, lim, dict(he=he_var, vol=vol_var), dict(cd=cd, c_add=c_add, t=t_asc), th


def fig(title, xt, yt, hover="x unified"):
    f = go.Figure()
    f.update_layout(title=title, xaxis_title=xt, yaxis_title=yt, hovermode=hover, height=400,
                    legend_title_text="Case", margin=dict(l=10, r=10, t=50, b=10))
    return f


def main():
    st.set_page_config(page_title="SPB Simulator", page_icon="🎈", layout="wide",
                       initial_sidebar_state="expanded")
    env, h_t, lim, var, asc, th = sidebar()

    st.title("Super Pressure Balloon Simulator")
    st.caption("Diameter → V_max, area, envelope mass → helium for neutral buoyancy at the target "
               "→ ascent → 24 h superpressure cycle → operational checks.")

    sz = size_for_target(env, h_t)
    m_he = sz["m_he"]

    # ---------------------------------------------------------------- sizing
    st.header("1 · Envelope and mass budget")
    c = st.columns(6)
    c[0].metric("V_max", f"{env.v_max:,.1f} m³")
    c[1].metric("Surface area", f"{env.area:,.1f} m²")
    c[2].metric("Envelope mass", f"{env.m_env:,.2f} kg")
    c[3].metric("Payload / additions", f"{env.payload:,.2f} kg")
    c[4].metric("Required helium", f"{m_he:,.3f} kg" if m_he > 0 else "—",
                f"fill minimum {sz['m_fill']:.3f} kg", delta_color="off")
    c[5].metric("Total system mass", f"{env.m_dry + max(m_he, 0.0):,.2f} kg",
                f"= ρ(h_t)·V_max {sz['m_total']:.2f} kg", delta_color="off")

    if m_he < sz["m_fill"]:
        st.error(f"❌ Not feasible: a {env.diameter:g} m envelope cannot float "
                 f"{env.payload:g} kg of additions at {h_t/1000:.1f} km - the helium needed for "
                 f"neutral buoyancy would not fill V_max (the balloon would not leave the ground). "
                 f"Max additions for this envelope: {max(sz['payload_max'], 0):.2f} kg. "
                 f"Minimum diameter for this payload: {min_diameter(env, h_t):.2f} m.")
        st.stop()

    budget = pd.DataFrame({"Component": ["Envelope", "Payload / additions", "Helium", "Total"],
                           "Mass [kg]": [env.m_env, env.payload, m_he, env.m_dry + m_he]})
    budget["Share [%]"] = budget["Mass [kg]"] / budget["Mass [kg]"].iloc[-1] * 100.0
    a, b = st.columns(2)
    a.dataframe(budget.style.format({"Mass [kg]": "{:,.3f}", "Share [%]": "{:,.1f}"}),
                hide_index=True, width="stretch")
    pie = go.Figure(go.Pie(labels=budget["Component"][:3], values=budget["Mass [kg]"][:3],
                           hole=0.5, sort=False, marker=dict(colors=COLORS[:3])))
    pie.update_layout(height=230, margin=dict(l=0, r=0, t=10, b=10))
    b.plotly_chart(pie, width="stretch")

    # ----------------------------------------------------------- simulations
    cases = [("Baseline", env, m_he)]
    if var["he"] > 0:
        cases += [(f"+{var['he']:g}% He", env, m_he * (1 + var["he"] / 100)),
                  (f"-{var['he']:g}% He", env, m_he * (1 - var["he"] / 100))]
    if var["vol"] > 0:
        for s in (1, -1):
            dD = (1 + s * var["vol"] / 100) ** (1 / 3)
            cases.append((f"{'+' if s > 0 else '-'}{var['vol']:g}% Vol",
                          replace(env, diameter=env.diameter * dD), m_he))
    with st.spinner("Running ascent and 24 h thermal simulations…"):
        results = [simulate_case(lbl, e, mh, h_t, th, asc["cd"], asc["c_add"], asc["t"])
                   for lbl, e, mh in cases]
    base = results[0]

    # ----------------------------------------------------------- feasibility
    st.header("2 · Flight feasibility")
    if not base["ok"]:
        st.error(base["msg"])
        st.stop()
    tb = base["thermal"]
    n_lim = lim["n_allow"] / lim["sf"]
    m_lo, m_hi = helium_window(env, h_t, tb["T_he"].min(), tb["T_he"].max(), n_lim)
    checks = feasibility(env, h_t, sz, base, lim, m_lo, m_hi)
    n_fail = sum(r[3] == FAIL for r in checks)
    n_warn = sum(r[3] == WARN for r in checks)
    if n_fail:
        st.error(f"❌ Not feasible - {n_fail} failed check(s), {n_warn} warning(s).")
    elif n_warn:
        st.warning(f"⚠️ Feasible with {n_warn} warning(s).")
    else:
        st.success("✅ Feasible: reaches the target, stays pressurized day and night, "
                   "within structural limits.")
    st.dataframe(pd.DataFrame(checks, columns=["Check", "Value", "Limit", "Status", "Comment"]),
                 hide_index=True, width="stretch")

    p_lo = sz["m_total"] - env.m_env - m_hi     # lightest payload (burst limit)
    p_hi = sz["m_total"] - env.m_env - m_lo     # heaviest payload (night slack)
    c = st.columns(4)
    c[0].metric("Helium window @ target", f"{m_lo:.3f} – {m_hi:.3f} kg",
                f"current {m_he:.3f} kg", delta_color="off",
                help="No slack at the coldest helium temperature → structural limit at the "
                     "hottest (baseline temperature extremes).")
    c[1].metric("Payload window @ target",
                f"{max(p_lo, 0):.2f} – {p_hi:.2f} kg" if p_hi > max(p_lo, 0) else "none",
                f"current {env.payload:.2f} kg", delta_color="off")
    c[2].metric("ΔP isothermal @ target", f"{sz['dp_std']:,.0f} Pa")
    c[3].metric("Min diameter (zero ΔP)", f"{min_diameter(env, h_t):.2f} m",
                f"current {env.diameter:g} m", delta_color="off")

    # --------------------------------------------------------------- summary
    st.header("3 · Simulation summary")
    df = pd.DataFrame([summary_row(r, h_t) for r in results]).set_index("Case")
    st.dataframe(df.style.format("{:,.2f}", na_rep="—"), width="stretch")
    st.download_button("Download summary (CSV)", df.to_csv().encode(), "spb_summary.csv",
                       "text/csv")
    for r in results[1:]:
        if not r["ok"]:
            st.warning(f"{r['label']}: {r['msg']}")

    # ---------------------------------------------------------------- ascent
    st.header("4 · Ascent profile")
    f_h = fig("Altitude vs time", "Time [min]", "Altitude [km]")
    f_v = fig("Velocity vs time", "Time [min]", "Velocity [m/s]")
    f_a = fig("Acceleration vs time", "Time [min]", "Acceleration [m/s²]")
    f_fill = fig("Envelope fill V/V_max vs altitude", "Altitude [km]", "Fill [%]", "closest")
    for r, col in zip(results, COLORS):
        if "ascent" not in r:
            continue
        s = r["ascent"]
        kw = dict(name=r["label"], line=dict(color=col))
        f_h.add_trace(go.Scatter(x=s["t"] / 60, y=s["h"] / 1000, **kw))
        f_v.add_trace(go.Scatter(x=s["t"] / 60, y=s["v"], **kw))
        f_a.add_trace(go.Scatter(x=s["t"] / 60, y=s["a"], **kw))
        f_fill.add_trace(go.Scatter(x=s["h"] / 1000, y=s["fill"], **kw))
    f_h.add_hline(y=h_t / 1000, line_dash="dash", line_color="gray", annotation_text="target")
    f_v.add_hrect(y0=lim["v_lo"], y1=lim["v_hi"], fillcolor="green", opacity=0.07, line_width=0)
    f_fill.add_vline(x=h_t / 1000, line_dash="dash", line_color="gray")
    for f in (f_h, f_v, f_a):
        f.update_xaxes(dtick=30)
    a, b = st.columns(2)
    a.plotly_chart(f_h, width="stretch")
    b.plotly_chart(f_v, width="stretch")
    a.plotly_chart(f_a, width="stretch")
    b.plotly_chart(f_fill, width="stretch")

    # --------------------------------------------------------------- thermal
    st.header("5 · 24-hour thermal / superpressure cycle at float")
    st.caption("t = 0 is local midnight. Volume fixed at V_max; ΔP < 0 means the envelope is slack.")
    f_T = fig("Helium temperature", "Hour of day", "T_He [K]")
    f_Tf = fig("Film temperature", "Hour of day", "T_film [K]")
    f_dp = fig("Pressure difference ΔP (internal − ambient)", "Hour of day", "ΔP [Pa]")
    f_N = fig("Film line load N = ΔP·r/2", "Hour of day", "N [N/m]")
    for r, col in zip(results, COLORS):
        if "thermal" not in r:
            continue
        t_ = r["thermal"]
        kw = dict(name=r["label"], line=dict(color=col))
        f_T.add_trace(go.Scatter(x=t_["hrs"], y=t_["T_he"], **kw))
        f_Tf.add_trace(go.Scatter(x=t_["hrs"], y=t_["T_f"], **kw))
        f_dp.add_trace(go.Scatter(x=t_["hrs"], y=t_["dp"], **kw))
        f_N.add_trace(go.Scatter(x=t_["hrs"], y=t_["load"], **kw))
    f_T.add_hline(y=tb["T_amb"], line_dash="dot", line_color="gray", annotation_text="ambient")
    f_dp.add_hline(y=0, line_dash="dot", line_color="gray", annotation_text="slack")
    f_N.add_hline(y=n_lim, line_dash="dash", line_color="#EF553B", annotation_text="allowable / SF")
    f_N.add_hline(y=lim["n_allow"], line_dash="dot", line_color="#EF553B",
                  annotation_text="allowable")
    for f in (f_T, f_Tf, f_dp, f_N):
        f.update_xaxes(dtick=3, range=[0, 24])
    a, b = st.columns(2)
    a.plotly_chart(f_T, width="stretch")
    b.plotly_chart(f_Tf, width="stretch")
    a.plotly_chart(f_dp, width="stretch")
    b.plotly_chart(f_N, width="stretch")

    # ---------------------------------------------------------- design space
    st.header("6 · Design space at the target altitude")
    st.caption("Feasible payload band vs envelope diameter (no night slack, film load ≤ allowable/SF), "
               "using the baseline helium temperature extremes.")
    Ds = np.linspace(max(0.5, env.diameter * 0.6), env.diameter * 1.8, 120)
    lo, hi = [], []
    for Dx in Ds:
        ex = replace(env, diameter=Dx)
        s2 = size_for_target(ex, h_t)
        wl, wh = helium_window(ex, h_t, tb["T_he"].min(), tb["T_he"].max(), n_lim)
        lo.append(max(s2["m_total"] - ex.m_env - wh, 0.0))
        hi.append(s2["m_total"] - ex.m_env - wl)
    lo, hi = np.array(lo), np.array(hi)
    ok = hi > lo
    f_ds = fig("Feasible payload / additions vs diameter", "Diameter [m]",
               "Payload / additions [kg]", "closest")
    f_ds.add_trace(go.Scatter(x=Ds[ok], y=hi[ok], name="Max (night slack limit)",
                              line=dict(color="#FFA15A")))
    f_ds.add_trace(go.Scatter(x=Ds[ok], y=lo[ok], name="Min (burst limit)", fill="tonexty",
                              fillcolor="rgba(0,204,150,0.15)", line=dict(color="#EF553B")))
    f_ds.add_trace(go.Scatter(x=[env.diameter], y=[env.payload], name="Current design",
                              mode="markers", marker=dict(size=12, color="black", symbol="x")))
    st.plotly_chart(f_ds, width="stretch")

    # ---------------------------------------------------------------- notes
    st.header("7 · Model notes")
    st.markdown(r"""
**Envelope.** $V_{max}=\pi D^3/6$, $A=\pi D^2$, $m_{env}=A\,\rho_{film}$.

**Helium for neutral buoyancy at the target.** Fully pressurized volume is $V_{max}$:
$\rho_{air}(h_t)V_{max}=m_{env}+m_{payload}+m_{He}$ ⇒ $m_{He}=\rho_{air}(h_t)V_{max}-m_{env}-m_{payload}$.
Feasible only if $m_{He}\ge m_{fill}=P(h_t)V_{max}M_{He}/(RT(h_t))$; otherwise the helium cannot fill
the envelope and the system has no free lift. Isothermal superpressure $\Delta P_{std}=P(m_{He}/m_{fill}-1)$.

**Ascent.** $(m+C_a\rho V)\dot v=\rho Vg-mg-\tfrac12\rho v|v|C_d\pi r^2$, $V=\min(m_{He}RT/(PM_{He}),V_{max})$,
RK45, helium at ambient temperature.

**Thermal (original model).** $q_{sun}=\alpha(\tfrac14I_{dn}+I_{diff}+a\,GHI)$, $I_{dn}=S\tau^{m}$ (Kasten–Young),
$I_{diff}=0.1S\sin\theta$, $GHI=I_{dn}\sin\theta+I_{diff}$;
$m_{env}c_p\dot T_f=A[q_{sun}+\varepsilon\sigma T_{sky}^4-\varepsilon\sigma T_f^4-H(T_f-T_{amb})-h_i(T_f-T_{He})]$,
$m_{He}c_v\dot T_{He}=h_iA(T_f-T_{He})$. $\Delta P(t)=m_{He}RT_{He}/(V_{max}M_{He})-P_{amb}$,
film load $N=\Delta P\,r/2$.

**Limitations.** Spherical envelope, no film creep, no helium leakage, altitude fixed during the
thermal cycle, ISA atmosphere.
""")


if __name__ == "__main__":
    main()
