# -*- coding: utf-8 -*-
"""
Super Pressure Balloon (SPB) Simulator  -  compressible-helium model
====================================================================

Independent inputs: envelope diameter D, target float altitude z_t,
payload/hardware additions mass, fabric density, thermal/radiation data.

Physics chain
-------------
1. Envelope:   V_max = pi D^3 / 6,   A = pi D^2,   m_env = A * fabric_density
2. Helium (neutral buoyancy at the target, fully pressurized volume V_max):
       m_env + m_payload + m_He = rho_air(z_t) * V_max
   =>  m_He = rho_air(z_t) * V_max - m_env - m_payload        (no caps)
3. Compressible helium - m_He is constant for the whole flight:
       V_He(z) = m_He * R * T(z) / (P(z) * M_He)
   Launch volume V_0 = V_He(0) < V_max: the envelope starts slack, expands
   during ascent, becomes taut at z_full where V_He(z_full) = V_max, then
   rises at constant volume until rho_air * V_max = m_total, i.e. at z_t.
   Superpressure at the target (helium at ambient T):
       dP = P(z_t) * (V_He(z_t) / V_max - 1)
4. Ascent ODE (RK45), volume V = min(V_He(z), V_max), drag, optional added
   mass. 24 h thermal ODE (LSODA) for film and helium at constant volume
   -> dP(t) and film line load N = dP r / 2.
5. Checks: lift-off, envelope taut below target, ascent rate, target reached,
   no slack at night (dP_min > 0), film load <= allowable / safety factor.

Standalone single file - ready for Streamlit Community Cloud
(requirements: streamlit, numpy, scipy, pandas, plotly, jinja2).
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
CV_HE = 3116.0            # helium Cv [J/(kg K)]
SOLAR_C = 1361.0          # solar constant [W/m^2]

COLORS = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692"]
PASS, WARN, FAIL = "✅ Pass", "⚠️ Warning", "❌ Fail"

# =============================================================================
# ISA atmosphere (US Standard Atmosphere 1976, 0-86 km)
# =============================================================================
_LAYERS = [  # base altitude [m], base temperature [K], lapse rate [K/m]
    (0.0, 288.15, -0.0065), (11000.0, 216.65, 0.0), (20000.0, 216.65, 0.0010),
    (32000.0, 228.65, 0.0028), (47000.0, 270.65, 0.0), (51000.0, 270.65, -0.0028),
    (71000.0, 214.65, -0.0020),
]
_GMR = G * M_AIR / R_U
_P_BASE = [101325.0]
for _i in range(1, len(_LAYERS)):
    _h0, _T0, _L = _LAYERS[_i - 1]
    _dh = _LAYERS[_i][0] - _h0
    _P_BASE.append(_P_BASE[-1] * (np.exp(-_GMR * _dh / _T0) if _L == 0.0
                                  else ((_T0 + _L * _dh) / _T0) ** (-_GMR / _L)))


def atmosphere(z):
    """(T [K], P [Pa], rho [kg/m^3]) at altitude z [m]."""
    z = min(max(float(z), 0.0), 86000.0)
    i = len(_LAYERS) - 1
    while z < _LAYERS[i][0]:
        i -= 1
    z0, T0, L = _LAYERS[i]
    T = T0 + L * (z - z0)
    P = (_P_BASE[i] * np.exp(-_GMR * (z - z0) / T0) if L == 0.0
         else _P_BASE[i] * (T / T0) ** (-_GMR / L))
    return T, P, P * M_AIR / (R_U * T)


def he_volume(m_he, z):
    """Free volume of m_he kg of helium at ambient T and P of altitude z (ideal gas)."""
    T, P, _ = atmosphere(z)
    return m_he * R_U * T / (P * M_HE)


def root_altitude(fun, lo=0.0, hi=50000.0):
    """Root of fun(z) on [lo, hi]; NaN if there is no sign change."""
    a, b = fun(lo), fun(hi)
    if a == 0.0:
        return lo
    if np.sign(a) == np.sign(b):
        return np.nan
    return brentq(fun, lo, hi, xtol=0.1)


# =============================================================================
# Envelope & helium sizing
# =============================================================================
@dataclass(frozen=True)
class Envelope:
    diameter: float     # [m]
    fabric_gsm: float   # [g/m^2]
    payload: float      # payload + hardware additions [kg]

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
        return self.area * self.fabric_gsm / 1000.0


def size_helium(env: Envelope, z_t):
    """Neutral-buoyancy helium at the target and its compressible-gas consequences."""
    T_t, P_t, rho_t = atmosphere(z_t)
    m_total = rho_t * env.v_max                       # displaced air mass at target
    m_he = m_total - env.m_env - env.payload          # required helium, constant in flight
    mp = max(m_he, 0.0)
    V_t = he_volume(mp, z_t)                          # free helium volume at target
    z_full = root_altitude(lambda z: he_volume(mp, z) - env.v_max) if mp > 0 else np.nan
    if mp > 0 and he_volume(mp, 0.0) >= env.v_max:
        z_full = 0.0
    gross = mp * M_AIR / M_HE                         # buoyancy [kg] while slack (constant)
    return dict(T=T_t, P=P_t, rho=rho_t, m_total=m_total, m_he=m_he,
                V0=he_volume(mp, 0.0), V_t=V_t, z_full=z_full,
                dp_iso=P_t * (V_t / env.v_max - 1.0),
                free_lift_kg=gross - m_total,
                free_lift_pct=(gross / m_total - 1.0) * 100.0)


def lift_off_limits(env: Envelope, z_t):
    """Physical lift-off boundary (V_He(z_t) = V_max): max payload for this D,
    min D for this payload."""
    def margin(e):
        _, _, rho = atmosphere(z_t)
        return rho * e.v_max * (1.0 - M_HE / M_AIR) - e.m_env - e.payload
    p_max = margin(replace(env, payload=0.0))
    Ds = np.geomspace(0.2, 300.0, 300)
    r = np.array([margin(replace(env, diameter=D)) for D in Ds])
    idx = np.where((r[:-1] < 0) & (r[1:] >= 0))[0]
    d_min = (brentq(lambda D: margin(replace(env, diameter=D)), Ds[idx[0]], Ds[idx[0] + 1],
                    xtol=1e-4) if len(idx) else np.nan)
    return p_max, d_min


def float_altitude(env: Envelope, m_he):
    """Equilibrium float of envelope + given helium when taut (helium at ambient T)."""
    m_total = env.m_env + env.payload + m_he
    return root_altitude(lambda z: atmosphere(z)[2] * env.v_max - m_total)


# =============================================================================
# Ascent model
# =============================================================================
def ascent_ode(t, y, m_total, m_he, v_max, cd, c_add):
    z, v = y
    T, P, rho = atmosphere(z)
    V = min(m_he * R_U * T / (P * M_HE), v_max)      # compressible, capped by envelope
    r = (3.0 * V / (4.0 * np.pi)) ** (1.0 / 3.0)
    f_net = rho * V * G - m_total * G - 0.5 * rho * v * abs(v) * cd * np.pi * r * r
    a = f_net / (m_total + c_add * rho * V)
    if z <= 0.0 and v <= 0.0 and a < 0.0:            # resting on the ground
        return [0.0, 0.0]
    return [v, a]


@st.cache_data(show_spinner=False)
def run_ascent(m_total, m_he, v_max, cd, c_add, t_end):
    sol = solve_ivp(ascent_ode, (0.0, t_end), [0.0, 0.1],
                    args=(m_total, m_he, v_max, cd, c_add),
                    method="RK45", max_step=10.0, rtol=1e-6, atol=1e-6)
    t, z, v = sol.t, np.maximum(sol.y[0], 0.0), sol.y[1]
    V = np.array([min(he_volume(m_he, x), v_max) for x in z])
    a = np.gradient(v, t) if len(t) > 2 else np.zeros_like(t)
    return dict(t=t, z=z, v=v, a=a, V=V)


# =============================================================================
# 24 h thermal model (film + helium nodes, constant volume at float)
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
    sin_alt = (np.sin(lat) * np.sin(dec)
               + np.cos(lat) * np.cos(dec) * np.cos(np.radians((hr - 12.0) * 15.0)))
    alt = np.arcsin(np.clip(sin_alt, -1.0, 1.0))
    q_sun = 0.0
    if alt > 0.0:
        airmass = 1.0 / (np.sin(alt) + 0.50572 * (np.degrees(alt) + 6.07995) ** -1.6364)
        i_dn = SOLAR_C * th.tau_atm ** airmass
        i_diff = 0.1 * SOLAR_C * sin_alt
        ghi = i_dn * sin_alt + i_diff
        q_sun = th.alpha * (0.25 * i_dn + i_diff + th.albedo * ghi)
    Q_int = th.h_int * A * (T_f - T_he)
    Q_film = A * (q_sun + th.eps * SIGMA * (th.t_sky ** 4 - T_f ** 4)
                  - th.h_conv * (T_f - T_amb)) - Q_int
    return [Q_int / (m_he * CV_HE), Q_film / (m_env * th.cp_film)]


@st.cache_data(show_spinner=False)
def run_thermal(m_he, m_env, A, T_amb, th: Thermal):
    """Midnight start at ambient temperature, optional spin-up days, last 24 h returned."""
    t0 = th.spinup_days * 86400.0
    t_eval = np.linspace(t0, t0 + 86400.0, 97)
    sol = solve_ivp(thermal_ode, (0.0, t0 + 86400.0), [T_amb, T_amb],
                    args=(m_he, m_env, A, T_amb, th), method="LSODA",
                    t_eval=t_eval, max_step=900.0, rtol=1e-6, atol=1e-6)
    T_he = np.interp(t_eval, sol.t, sol.y[0])
    T_f = np.interp(t_eval, sol.t, sol.y[1])
    return (t_eval - t0) / 3600.0, T_he, T_f


# =============================================================================
# Case simulation
# =============================================================================
def simulate(label, env: Envelope, m_he, z_t, th, cd, c_add, t_asc_h):
    m_total = env.m_env + env.payload + m_he
    gross = max(m_he, 0.0) * M_AIR / M_HE
    res = dict(label=label, env=env, m_he=m_he, m_total=m_total,
               V0=he_volume(max(m_he, 0.0), 0.0),
               free_lift_pct=(gross / m_total - 1.0) * 100.0 if m_total > 0 else np.nan,
               z_float=float_altitude(env, m_he) if m_he > 0 else np.nan)
    if m_he <= 0 or gross <= m_total:
        res["msg"] = "No lift-off: helium buoyancy is below the system weight."
        return res

    asc = run_ascent(m_total, m_he, env.v_max, cd, c_add, t_asc_h * 3600.0)
    res["ascent"] = asc
    z_ref = res["z_float"] if np.isfinite(res["z_float"]) else z_t
    reach = np.where(asc["z"] >= 0.98 * z_ref)[0]
    sel = (asc["z"] > 100.0) & (asc["z"] < 0.95 * z_ref) & (asc["v"] > 0.1)
    full = np.where(asc["V"] >= 0.999 * env.v_max)[0]
    res.update(t_reach=asc["t"][reach[0]] / 60.0 if len(reach) else np.nan,
               v_avg=float(asc["v"][sel].mean()) if sel.any() else np.nan,
               z_end=float(asc["z"][-1]),
               z_full=float(asc["z"][full[0]]) if len(full) else np.nan)

    z_f = res["z_float"] if np.isfinite(res["z_float"]) else res["z_end"]
    T_amb, P_amb, _ = atmosphere(z_f)
    hrs, T_he, T_f = run_thermal(m_he, env.m_env, env.area, T_amb, th)
    dp = m_he * R_U * T_he / (env.v_max * M_HE) - P_amb      # constant volume V_max
    res["thermal"] = dict(hrs=hrs, T_he=T_he, T_f=T_f, dp=dp, T_amb=T_amb,
                          load=np.maximum(dp, 0.0) * env.radius / 2.0)
    return res


def summary_row(r, z_t):
    e = r["env"]
    row = {"Case": r["label"], "D [m]": e.diameter, "V_max [m³]": e.v_max,
           "m_env [kg]": e.m_env, "m_payload [kg]": e.payload, "m_He [kg]": r["m_he"],
           "m_total [kg]": r["m_total"], "V_0 launch [m³]": r["V0"],
           "V_0/V_max [%]": r["V0"] / e.v_max * 100.0, "Free lift [%]": r["free_lift_pct"],
           "z_full [km]": r.get("z_full", np.nan) / 1000.0,
           "Float [km]": r["z_float"] / 1000.0,
           "Float − target [km]": (r["z_float"] - z_t) / 1000.0,
           "Time to float [min]": r.get("t_reach", np.nan),
           "Mean ascent [m/s]": r.get("v_avg", np.nan)}
    if "thermal" in r:
        t = r["thermal"]
        row.update({"ΔP min [Pa]": t["dp"].min(), "ΔP max [Pa]": t["dp"].max(),
                    "T_He min [K]": t["T_he"].min(), "T_He max [K]": t["T_he"].max(),
                    "N max [N/m]": t["load"].max()})
    return row


def helium_window(env: Envelope, z_t, T_he_min, T_he_max, n_lim):
    """Helium range at the target: P_in(T_min) >= P_amb and N(T_max) <= n_lim."""
    _, P, _ = atmosphere(z_t)
    m_lo = P * env.v_max * M_HE / (R_U * T_he_min)
    m_hi = (P + 2.0 * n_lim / env.radius) * env.v_max * M_HE / (R_U * T_he_max)
    return m_lo, m_hi


def feasibility_table(env, z_t, sz, base, lim, m_lo, m_hi):
    rows = []
    lift_ok = sz["m_he"] > 0 and sz["free_lift_pct"] > 0
    if lift_ok:
        msg = "Helium buoyancy exceeds the system weight."
    else:
        p_max, d_min = lift_off_limits(env, z_t)
        msg = (f"The neutral-buoyancy helium ({sz['m_he']:.3f} kg) would occupy only "
               f"{sz['V_t']:.1f} m³ at the target (< V_max {env.v_max:.1f} m³), so it cannot lift "
               f"the system. Physical limit for this D: payload ≤ {max(p_max, 0):.2f} kg; "
               f"for this payload: D ≥ {d_min:.2f} m.")
    rows.append(("Lift-off (launch free lift > 0)", f"{sz['free_lift_pct']:.1f} %", "> 0 %",
                 PASS if lift_ok else FAIL, msg))
    if not lift_ok:
        return rows
    fl = sz["free_lift_pct"]
    rows.append(("Launch free lift margin", f"{fl:.1f} %", f"≥ {lim['fl_min']:.0f} %",
                 PASS if fl >= lim["fl_min"] else WARN,
                 "OK" if fl >= lim["fl_min"] else "Low - slow, wind-sensitive launch."))
    rows.append(("Envelope taut below target (z_full)", f"{sz['z_full']/1000:.2f} km",
                 f"≤ {z_t/1000:.2f} km", PASS,
                 f"Slack at launch (V₀ = {sz['V0']:.2f} m³, {sz['V0']/env.v_max*100:.1f}% of V_max), "
                 f"taut at {sz['z_full']/1000:.2f} km, then rises at constant volume to the target."))
    va = base.get("v_avg", np.nan)
    if np.isfinite(va):
        ok = lim["v_lo"] <= va <= lim["v_hi"]
        rows.append(("Mean ascent rate", f"{va:.2f} m/s",
                     f"{lim['v_lo']:.1f}–{lim['v_hi']:.1f} m/s", PASS if ok else WARN,
                     "OK" if ok else ("Too slow - long tropopause crossing, large drift."
                                      if va < lim["v_lo"] else
                                      "Too fast - high dynamic load and overshoot at full inflation.")))
    reached = np.isfinite(base.get("t_reach", np.nan))
    rows.append(("Target reached in ascent sim", f"{base['z_end']/1000:.2f} km at end",
                 f"≥ {0.98*z_t/1000:.2f} km", PASS if reached else FAIL,
                 f"Reached in {base['t_reach']:.0f} min." if reached
                 else "Not reached - extend the ascent time."))
    t = base["thermal"]
    n_lim = lim["n_allow"] / lim["sf"]
    dpmin, nmax = t["dp"].min(), t["load"].max()
    rows.append(("Night superpressure ΔP min", f"{dpmin:,.0f} Pa", "> 0 Pa",
                 PASS if dpmin > 0 else FAIL,
                 "Pressurized all night." if dpmin > 0 else
                 f"Envelope goes slack at night → descent. Needs ≥ {m_lo:.3f} kg He at this D."))
    s = PASS if nmax <= n_lim else (WARN if nmax <= lim["n_allow"] else FAIL)
    rows.append(("Peak film load (day)", f"{nmax:,.0f} N/m", f"≤ {n_lim:,.0f} N/m (allow/SF)", s,
                 {PASS: "OK", WARN: "Below allowable, inside the safety margin.",
                  FAIL: f"Burst risk. Helium must stay ≤ {m_hi:.3f} kg at this D."}[s]))
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
    z_t = sb.number_input("Target float altitude [km]", 1.0, 45.0, 14.0, 0.1) * 1000.0
    payload = sb.number_input("Payload / hardware additions [kg]", 0.0, 10000.0, 5.0, 0.1)

    sb.subheader("3 · Structural & flight limits")
    n_allow = sb.number_input("Film allowable line load [N/m]", 100.0, 500000.0, 15000.0, 100.0,
                              help="Burst strength of film / seams per unit width.")
    sf = sb.number_input("Safety factor", 1.0, 5.0, 1.5, 0.05)
    fl_min = sb.number_input("Min launch free lift [%]", 0.0, 500.0, 10.0, 1.0)
    v_lo, v_hi = sb.slider("Acceptable mean ascent rate [m/s]", 0.5, 15.0, (2.0, 7.0), 0.1)

    sb.subheader("4 · Sensitivity cases")
    he_var = sb.number_input("± helium mass [%]", 0.0, 50.0, 5.0, 0.5)
    vol_var = sb.number_input("± max volume [%]", 0.0, 50.0, 5.0, 0.5,
                              help="Envelope tolerance at the same helium mass.")

    sb.subheader("5 · Ascent")
    cd = sb.number_input("Drag coefficient C_d", 0.1, 2.0, 0.47, 0.01)
    c_add = sb.number_input("Added-mass coefficient", 0.0, 1.0, 0.0, 0.05,
                            help="0 = baseline model; 0.5 = sphere virtual mass.")
    t_asc = sb.number_input("Ascent simulation time [h]", 0.5, 12.0, 5.0, 0.5)

    sb.subheader("6 · Thermal")
    cp_film = sb.number_input("Film specific heat [J/kgK]", 1000.0, 3000.0, 1500.0, 50.0)
    h_int = sb.slider("Internal convection [W/m²K]", 0.1, 5.0, 2.0, 0.1)
    h_conv = sb.slider("External convection [W/m²K]", 0.1, 10.0, 0.6, 0.1)
    spin = sb.number_input("Thermal spin-up days", 0, 5, 0, 1,
                           help="0 = 24 h from ambient at midnight; >0 = settled periodic day.")

    sb.subheader("7 · Radiation")
    lat = sb.number_input("Latitude [°]", -90.0, 90.0, 32.0, 0.1)
    dec = sb.number_input("Solar declination [°]", -23.44, 23.44, -23.44, 0.5)
    alpha = sb.slider("Solar absorptivity α", 0.01, 1.0, 0.40, 0.01)
    eps = sb.slider("Thermal emissivity ε", 0.01, 1.0, 0.85, 0.01)
    albedo = sb.slider("Surface albedo", 0.0, 1.0, 0.70, 0.01)
    tau = sb.slider("Atmospheric transmissivity τ", 0.1, 1.0, 0.95, 0.01)
    t_sky = sb.number_input("Sky temperature [K]", 3.0, 300.0, 216.65, 1.0)

    return (Envelope(D, gsm, payload), z_t,
            dict(n_allow=n_allow, sf=sf, fl_min=fl_min, v_lo=v_lo, v_hi=v_hi),
            dict(he=he_var, vol=vol_var), dict(cd=cd, c_add=c_add, t=t_asc),
            Thermal(lat, dec, alpha, eps, albedo, tau, t_sky, h_conv, h_int, cp_film, int(spin)))


def fig(title, xt, yt, hover="x unified"):
    f = go.Figure()
    f.update_layout(title=title, xaxis_title=xt, yaxis_title=yt, hovermode=hover, height=400,
                    legend_title_text="Case", margin=dict(l=10, r=10, t=50, b=10))
    return f


def show_grid(figs):
    cols = st.columns(2)
    for i, f in enumerate(figs):
        cols[i % 2].plotly_chart(f, width="stretch")


def main():
    st.set_page_config(page_title="SPB Simulator", page_icon="🎈", layout="wide",
                       initial_sidebar_state="expanded")
    env, z_t, lim, var, asc, th = sidebar()
    st.title("Super Pressure Balloon Simulator")
    st.caption("D → V_max, m_env  •  m_He = ρ(z_t)·V_max − m_env − m_payload  •  "
               "compressible helium: slack at launch → taut at z_full → float at z_t")

    sz = size_helium(env, z_t)
    m_he = sz["m_he"]

    # ------------------------------------------------------------ 1. sizing
    st.header("1 · Mass budget and helium")
    c = st.columns(6)
    c[0].metric("V_max", f"{env.v_max:,.2f} m³", f"A = {env.area:,.1f} m²", delta_color="off")
    c[1].metric("Envelope mass", f"{env.m_env:,.3f} kg")
    c[2].metric("Payload / additions", f"{env.payload:,.3f} kg")
    c[3].metric("Helium mass (constant)", f"{m_he:,.3f} kg")
    c[4].metric("Total system mass", f"{sz['m_total']:,.3f} kg",
                f"= ρ(z_t)·V_max, ρ = {sz['rho']:.4f} kg/m³", delta_color="off")
    c[5].metric("Launch free lift", f"{sz['free_lift_pct']:,.1f} %",
                f"{sz['free_lift_kg'] * G:,.1f} N", delta_color="off")
    c = st.columns(4)
    c[0].metric("Launch volume V₀", f"{sz['V0']:,.2f} m³",
                f"{sz['V0'] / env.v_max * 100:,.1f}% of V_max", delta_color="off")
    c[1].metric("Envelope taut at z_full",
                f"{sz['z_full'] / 1000:,.2f} km" if np.isfinite(sz["z_full"]) else "never")
    c[2].metric("Free He volume @ target", f"{sz['V_t']:,.2f} m³",
                f"{sz['V_t'] / env.v_max * 100:,.1f}% of V_max", delta_color="off")
    c[3].metric("Isothermal ΔP @ target", f"{sz['dp_iso']:,.0f} Pa")

    zs = np.linspace(0.0, max(z_t * 1.25, 5000.0), 300)
    Vfree = np.array([he_volume(max(m_he, 0.0), z) for z in zs])
    f_vol = fig("Compressible helium: volume vs altitude (constant mass)",
                "Altitude [km]", "Volume [m³]", "closest")
    f_vol.add_trace(go.Scatter(x=zs / 1000, y=Vfree, name="Free helium volume V_He(z)",
                               line=dict(color=COLORS[0], dash="dot")))
    f_vol.add_trace(go.Scatter(x=zs / 1000, y=np.minimum(Vfree, env.v_max),
                               name="Envelope volume min(V_He, V_max)",
                               line=dict(color=COLORS[2], width=3)))
    f_vol.add_hline(y=env.v_max, line_dash="dash", line_color="gray", annotation_text="V_max")
    f_vol.add_vline(x=z_t / 1000, line_dash="dash", line_color=COLORS[1], annotation_text="target")
    f_vol.update_yaxes(range=[0, env.v_max * 1.6])
    budget = pd.DataFrame({"Component": ["Envelope", "Payload / additions", "Helium", "Total"],
                           "Mass [kg]": [env.m_env, env.payload, m_he, sz["m_total"]]})
    budget["Share [%]"] = budget["Mass [kg]"] / sz["m_total"] * 100.0
    a, b = st.columns([1, 2])
    a.dataframe(budget.style.format({"Mass [kg]": "{:,.3f}", "Share [%]": "{:,.1f}"}),
                hide_index=True, width="stretch")
    b.plotly_chart(f_vol, width="stretch")

    # ------------------------------------------------------- simulations
    cases = [("Baseline", env, m_he)]
    if var["he"] > 0:
        cases += [(f"+{var['he']:g}% He", env, m_he * (1 + var["he"] / 100)),
                  (f"−{var['he']:g}% He", env, m_he * (1 - var["he"] / 100))]
    if var["vol"] > 0:
        for s in (1, -1):
            k = (1 + s * var["vol"] / 100) ** (1 / 3)
            cases.append((f"{'+' if s > 0 else '−'}{var['vol']:g}% Vol",
                          replace(env, diameter=env.diameter * k), m_he))
    with st.spinner("Running ascent and 24 h thermal simulations…"):
        results = [simulate(lbl, e, mh, z_t, th, asc["cd"], asc["c_add"], asc["t"])
                   for lbl, e, mh in cases]
    base = results[0]

    # ------------------------------------------------------- 2. feasibility
    st.header("2 · Flight feasibility")
    n_lim = lim["n_allow"] / lim["sf"]
    if "thermal" in base:
        T_min, T_max = base["thermal"]["T_he"].min(), base["thermal"]["T_he"].max()
    else:
        T_min = T_max = sz["T"]
    m_lo, m_hi = helium_window(env, z_t, T_min, T_max, n_lim)
    checks = feasibility_table(env, z_t, sz, base, lim, m_lo, m_hi)
    n_fail = sum(r[3] == FAIL for r in checks)
    n_warn = sum(r[3] == WARN for r in checks)
    if n_fail:
        st.error(f"❌ Not feasible - {n_fail} failed check(s), {n_warn} warning(s).")
    elif n_warn:
        st.warning(f"⚠️ Feasible with {n_warn} warning(s).")
    else:
        st.success("✅ Feasible: lifts off, reaches the target, pressurized day and night, "
                   "within structural limits.")
    st.dataframe(pd.DataFrame(checks, columns=["Check", "Value", "Limit", "Status", "Comment"]),
                 hide_index=True, width="stretch")
    if "thermal" in base:
        p_lo = sz["m_total"] - env.m_env - m_hi
        p_hi = sz["m_total"] - env.m_env - m_lo
        c = st.columns(2)
        c[0].metric("Helium window @ target (this D)", f"{m_lo:.3f} – {m_hi:.3f} kg",
                    f"required {m_he:.3f} kg", delta_color="off",
                    help="Night pressurization (coldest He) → structural limit (hottest He).")
        c[1].metric("Payload window @ target (this D)",
                    f"{max(p_lo, 0):.2f} – {p_hi:.2f} kg" if p_hi > max(p_lo, 0) else "none",
                    f"current {env.payload:.2f} kg", delta_color="off")

    # ------------------------------------------------------- 3. summary
    st.header("3 · Simulation summary")
    df = pd.DataFrame([summary_row(r, z_t) for r in results]).set_index("Case")
    st.dataframe(df.style.format("{:,.2f}", na_rep="—"), width="stretch")
    st.download_button("Download summary (CSV)", df.to_csv().encode(), "spb_summary.csv",
                       "text/csv")
    for r in results:
        if "msg" in r:
            st.warning(f"{r['label']}: {r['msg']}")

    # ------------------------------------------------------- 4. ascent
    st.header("4 · Ascent")
    f_z = fig("Altitude vs time", "Time [min]", "Altitude [km]")
    f_v = fig("Velocity vs time", "Time [min]", "Velocity [m/s]")
    f_a = fig("Acceleration vs time", "Time [min]", "Acceleration [m/s²]")
    f_V = fig("Balloon volume vs time", "Time [min]", "Volume [m³]")
    for r, col in zip(results, COLORS):
        if "ascent" not in r:
            continue
        s = r["ascent"]
        kw = dict(name=r["label"], line=dict(color=col))
        f_z.add_trace(go.Scatter(x=s["t"] / 60, y=s["z"] / 1000, **kw))
        f_v.add_trace(go.Scatter(x=s["t"] / 60, y=s["v"], **kw))
        f_a.add_trace(go.Scatter(x=s["t"] / 60, y=s["a"], **kw))
        f_V.add_trace(go.Scatter(x=s["t"] / 60, y=s["V"], **kw))
    f_z.add_hline(y=z_t / 1000, line_dash="dash", line_color="gray", annotation_text="target")
    f_v.add_hrect(y0=lim["v_lo"], y1=lim["v_hi"], fillcolor="green", opacity=0.07, line_width=0)
    f_V.add_hline(y=env.v_max, line_dash="dash", line_color="gray", annotation_text="V_max")
    for f in (f_z, f_v, f_a, f_V):
        f.update_xaxes(dtick=30)
    show_grid([f_z, f_v, f_a, f_V])

    # ------------------------------------------------------- 5. thermal
    st.header("5 · 24-hour thermal / superpressure cycle at float")
    st.caption("t = 0 is local midnight. Volume = V_max; ΔP < 0 means the envelope is slack.")
    f_T = fig("Helium temperature", "Hour of day", "T_He [K]")
    f_Tf = fig("Film temperature", "Hour of day", "T_film [K]")
    f_dp = fig("Superpressure ΔP = P_in − P_amb", "Hour of day", "ΔP [Pa]")
    f_N = fig("Film line load N = ΔP·r/2", "Hour of day", "N [N/m]")
    for r, col in zip(results, COLORS):
        if "thermal" not in r:
            continue
        t = r["thermal"]
        kw = dict(name=r["label"], line=dict(color=col))
        f_T.add_trace(go.Scatter(x=t["hrs"], y=t["T_he"], **kw))
        f_Tf.add_trace(go.Scatter(x=t["hrs"], y=t["T_f"], **kw))
        f_dp.add_trace(go.Scatter(x=t["hrs"], y=t["dp"], **kw))
        f_N.add_trace(go.Scatter(x=t["hrs"], y=t["load"], **kw))
    if "thermal" in base:
        f_T.add_hline(y=base["thermal"]["T_amb"], line_dash="dot", line_color="gray",
                      annotation_text="ambient")
    f_dp.add_hline(y=0, line_dash="dot", line_color="gray", annotation_text="slack")
    f_N.add_hline(y=n_lim, line_dash="dash", line_color=COLORS[1], annotation_text="allowable / SF")
    f_N.add_hline(y=lim["n_allow"], line_dash="dot", line_color=COLORS[1],
                  annotation_text="allowable")
    for f in (f_T, f_Tf, f_dp, f_N):
        f.update_xaxes(dtick=3, range=[0, 24])
    show_grid([f_T, f_Tf, f_dp, f_N])

    # ------------------------------------------------------- 6. design space
    st.header("6 · Design space at the target altitude")
    st.caption("Helium recomputed from neutral buoyancy for every point. Night/burst limits use "
               "the baseline helium temperature extremes.")
    p_axis = np.linspace(0.0, max(env.payload * 2.0, 1.0), 150)
    he_axis = sz["m_total"] - env.m_env - p_axis
    m_fill = sz["m_total"] * M_HE / M_AIR             # helium that just fills V_max at target
    f1 = fig(f"Required helium vs payload (D = {env.diameter:g} m)",
             "Payload / additions [kg]", "Helium [kg]", "closest")
    f1.add_trace(go.Scatter(x=p_axis, y=he_axis, name="Required helium",
                            line=dict(color=COLORS[0], width=3)))
    f1.add_hline(y=m_fill, line_dash="dash", line_color="gray", annotation_text="lift-off limit")
    f1.add_hline(y=m_lo, line_dash="dot", line_color=COLORS[4], annotation_text="night slack limit")
    f1.add_hline(y=m_hi, line_dash="dot", line_color=COLORS[1], annotation_text="burst limit")
    f1.add_vline(x=env.payload, line_dash="dash", line_color="black")

    Ds = np.linspace(max(0.5, env.diameter * 0.6), env.diameter * 1.8, 120)
    lo, hi = [], []
    for Dx in Ds:
        ex = replace(env, diameter=Dx)
        cap = atmosphere(z_t)[2] * ex.v_max
        wl, wh = helium_window(ex, z_t, T_min, T_max, n_lim)
        lo.append(max(cap - ex.m_env - wh, 0.0))
        hi.append(cap - ex.m_env - wl)
    lo, hi = np.array(lo), np.array(hi)
    ok = hi > lo
    f2 = fig("Feasible payload band vs diameter", "Diameter [m]", "Payload / additions [kg]",
             "closest")
    f2.add_trace(go.Scatter(x=Ds[ok], y=hi[ok], name="Max (night slack)",
                            line=dict(color=COLORS[4])))
    f2.add_trace(go.Scatter(x=Ds[ok], y=lo[ok], name="Min (burst)", fill="tonexty",
                            fillcolor="rgba(0,204,150,0.15)", line=dict(color=COLORS[1])))
    f2.add_trace(go.Scatter(x=[env.diameter], y=[env.payload], name="Current design",
                            mode="markers", marker=dict(size=12, color="black", symbol="x")))
    show_grid([f1, f2])

    # ------------------------------------------------------- 7. notes
    st.header("7 · Model notes")
    st.markdown(r"""
**Envelope.** $V_{max}=\pi D^3/6$, $A=\pi D^2$, $m_{env}=A\,\rho_{fabric}$.

**Helium (neutral buoyancy at the target).**
$m_{env}+m_{payload}+m_{He}=\rho_{air}(z_t)\,V_{max}\;\Rightarrow\;m_{He}=\rho_{air}(z_t)V_{max}-m_{env}-m_{payload}$.
No caps or ground-volume limits are applied.

**Compressible helium.** $m_{He}$ is constant; $V_{He}(z)=m_{He}RT(z)/(P(z)M_{He})$.
At launch $V_0=V_{He}(0)<V_{max}$ (slack). While slack, buoyancy $=m_{He}M_{air}/M_{He}$ is constant with
altitude; the envelope becomes taut at $z_{full}$ ($V_{He}=V_{max}$), then buoyancy $\rho_{air}V_{max}$ falls
with altitude until it equals $m_{total}$ at $z_t$. Lift-off requires $m_{He}M_{air}/M_{He}>m_{total}$, which
is the same condition as $V_{He}(z_t)>V_{max}$ (i.e. $z_{full}<z_t$) - a consequence of the gas law, not an
imposed cap. Isothermal $\Delta P(z_t)=P(z_t)\,(V_{He}(z_t)/V_{max}-1)$.

**Ascent (RK45).** $(m+C_a\rho V)\dot v=\rho Vg-mg-\tfrac12\rho v|v|C_d\pi r^2$, $V=\min(V_{He},V_{max})$.

**Thermal (LSODA).** $q_{sun}=\alpha(\tfrac14I_{dn}+I_{diff}+a\,GHI)$, $I_{dn}=S\tau^{m}$ (Kasten–Young),
$I_{diff}=0.1S\sin\theta$; $m_{env}c_p\dot T_f=A[q_{sun}+\varepsilon\sigma(T_{sky}^4-T_f^4)-H(T_f-T_{amb})]-h_iA(T_f-T_{He})$,
$m_{He}c_v\dot T_{He}=h_iA(T_f-T_{He})$. $\Delta P(t)=m_{He}RT_{He}/(V_{max}M_{He})-P_{amb}$, $N=\Delta P\,r/2$.

**Limitations.** Spherical envelope, helium at ambient temperature during ascent, no film creep,
no helium leakage, fixed altitude during the thermal cycle, ISA atmosphere.
""")


if __name__ == "__main__":
    main()
