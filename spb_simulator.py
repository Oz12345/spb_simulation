# -*- coding: utf-8 -*-
"""
Super Pressure Balloon (SPB) Simulator  -  target-altitude-driven design
=======================================================================

Design chain (recomputed on every input change):
    1. Fixed envelope:   V_max = (pi/6) D^3,   A = pi D^2          (sphere)
    2. Envelope mass:    m_env = A (1 + seam overhead) * film_density
    3. Dry mass:         m_dry = m_env + m_extra   (payload + all hardware)
    4. TARGET ALTITUDE drives the helium. A fully pressurized SPB floats where
           rho_air(h_t) * V_max = m_total           (Archimedes, fixed volume)
       so the required helium is
           m_He = rho_air(h_t) * V_max - m_dry
       and the total system mass m_total = m_dry + m_He.
    5. Feasibility at the target:
           * helium must at least fill V_max at h_t  ->  m_He >= m_fill
             (equivalent to m_dry <= rho V (1 - M_He/M_air) = max dry mass)
           * night: helium must stay pressurized (dP_min > 0), else slack/descent
           * day:   film line load N = dP r / 2 must stay below allowable / SF
           * launch free lift and ascent rate must be within operational limits
           * the ascent simulation must reach the target within the sim time

Physics modules: ISA 1976 atmosphere, 1-DOF ascent ODE (buoyancy, weight,
quadratic drag, added mass, volume capped at V_max), 2-node transient thermal
model (film + helium) for the 24 h cycle at float.

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


# =============================================================================
# Design, helium sizing and feasibility envelope
# =============================================================================
@dataclass(frozen=True)
class Design:
    diameter: float          # envelope diameter [m]
    film_gsm: float          # film areal density [g/m^2]
    seam_pct: float          # seams / tapes overhead [% of area]
    extra: float             # payload + all hardware / additions [kg]

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
        return self.m_env + self.extra


def target_sizing(d: Design, h_t):
    """Helium and mass budget that make the fixed envelope float at h_t."""
    T, P, rho = atmosphere(h_t)
    m_total = rho * d.v_max                       # buoyancy capacity at target
    m_he = m_total - d.m_dry                      # required helium
    m_fill = rho * d.v_max * K_MOLAR              # helium that just fills V_max at h_t
    m_dry_max = m_total - m_fill                  # = rho V (1 - K)
    return dict(T=T, P=P, rho=rho, m_total=m_total, m_he=m_he, m_fill=m_fill,
                m_dry_max=m_dry_max, extra_max=m_dry_max - d.m_env,
                dp_iso=P * (m_he / m_fill - 1.0) if m_he > 0 else np.nan)


def helium_limits(d: Design, h_t, T_he_min, T_he_max, n_allow):
    """Helium window at h_t from the thermal extremes (constant-volume gas):
    night: P_in(T_min) >= P_amb         -> m_He >= m_lo
    day:   (P_in(T_max) - P_amb) r/2 <= N_allow  -> m_He <= m_hi"""
    T, P, rho = atmosphere(h_t)
    m_lo = P * d.v_max * M_HE / (R_U * T_he_min)
    dp_allow = 2.0 * n_allow / d.radius
    m_hi = (P + dp_allow) * d.v_max * M_HE / (R_U * T_he_max)
    return m_lo, m_hi


def float_state(d: Design, m_he):
    """Where a given envelope + helium really floats (helium at ambient T)."""
    m_total = d.m_dry + m_he
    h = altitude_for_density(m_total / d.v_max)
    if np.isnan(h):
        return dict(h=np.nan, T=np.nan, P=np.nan, dp=np.nan)
    T, P, rho = atmosphere(h)
    return dict(h=h, T=T, P=P, dp=P * (m_he / (rho * d.v_max * K_MOLAR) - 1.0))


def min_diameter(d: Design, h_t):
    """Smallest envelope that can carry m_extra to h_t (zero superpressure)."""
    f = lambda D: target_sizing(replace(d, diameter=D), h_t)["extra_max"] - d.extra
    Ds = np.geomspace(0.2, 300.0, 300)
    r = np.array([f(D) for D in Ds])
    idx = np.where((r[:-1] < 0) & (r[1:] >= 0))[0]
    return brentq(f, Ds[idx[0]], Ds[idx[0] + 1], xtol=1e-4) if len(idx) else np.nan


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
    alpha: float
    eps: float
    albedo: float
    tau_atm: float
    q_earth_ir: float
    t_sky: float
    h_ext: float
    h_int: float
    cp_film: float


def direct_solar(hour, env: ThermalEnv, p_amb):
    lat = np.radians(env.latitude_deg)
    decl = np.radians(23.44) * np.sin(2.0 * np.pi * (284 + env.day_of_year) / 365.0)
    ha = np.radians((hour - 12.0) * 15.0)
    s = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(ha)
    el = np.arcsin(np.clip(s, -1.0, 1.0))
    if el <= 0.0:
        return 0.0, 0.0
    airmass = 1.0 / (np.sin(el) + 0.50572 * (np.degrees(el) + 6.07995) ** -1.6364)
    airmass *= p_amb / P_SL
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
    t_end = n_days * 86400.0
    t_eval = np.linspace(t_end - 86400.0, t_end, 289)
    sol = solve_ivp(thermal_rhs, (0.0, t_end), [T_amb, T_amb],
                    args=(m_he, m_env, A, T_amb, p_amb, env),
                    method="LSODA", t_eval=t_eval, max_step=300.0, rtol=1e-6)
    return (sol.t - (t_end - 86400.0)) / 3600.0, sol.y[0], sol.y[1]


# =============================================================================
# Case evaluation
# =============================================================================
def evaluate_case(label, d: Design, m_he, env, cd, c_add, t_asc_h):
    m_sys = d.m_dry + m_he
    gross = m_he / K_MOLAR
    fs = float_state(d, m_he)
    out = dict(label=label, design=d, m_he=m_he, m_sys=m_sys,
               fl_pct=(gross / m_sys - 1.0) * 100.0, fl_N=(gross - m_sys) * G0,
               float=fs, ok=True, msg="")
    if m_he <= 0 or out["fl_pct"] <= 0:
        out.update(ok=False, msg="No positive free lift at launch - balloon cannot ascend.")
        return out
    out["ascent"] = run_ascent(m_he, m_sys, d.v_max, cd, c_add, t_asc_h * 3600.0)
    if np.isnan(fs["h"]):
        out.update(ok=False, msg="No float equilibrium below 50 km.")
        return out
    hrs, T_he, T_f = run_thermal(m_he, d.m_env, d.area, fs["T"], fs["P"], env)
    dp = m_he * R_U * T_he / (d.v_max * M_HE) - fs["P"]
    out["thermal"] = dict(hours=hrs, T_he=T_he, T_f=T_f, dp=dp,
                          load=np.maximum(dp, 0.0) * d.radius / 2.0)
    return out


def ascent_metrics(c, h_ref):
    """Time to reach 98 % of h_ref, mean / max ascent rate, full-inflation altitude."""
    if "ascent" not in c:
        return dict(t_reach=np.nan, v_mean=np.nan, v_max=np.nan, h_full=np.nan, h_end=np.nan)
    t, h, v, _, fill = c["ascent"]
    reach = np.where(h >= 0.98 * h_ref)[0]
    mask = (h > 100) & (h < 0.95 * h_ref) & (v > 0)
    full = np.where(fill >= 99.9)[0]
    return dict(t_reach=t[reach[0]] / 60.0 if len(reach) else np.nan,
                v_mean=v[mask].mean() if mask.any() else np.nan,
                v_max=v.max(), h_full=h[full[0]] if len(full) else np.nan, h_end=h[-1])


def summary_row(c, h_t):
    d, fs = c["design"], c["float"]
    am = ascent_metrics(c, fs["h"] if np.isfinite(fs["h"]) else h_t)
    row = {
        "Case": c["label"], "D [m]": d.diameter, "V_max [m³]": d.v_max,
        "m_env [kg]": d.m_env, "m_He [kg]": c["m_he"], "m_total [kg]": c["m_sys"],
        "Free lift [%]": c["fl_pct"], "Float alt [km]": fs["h"] / 1000.0,
        "Float − target [km]": (fs["h"] - h_t) / 1000.0,
        "Time to float [min]": am["t_reach"], "Mean ascent [m/s]": am["v_mean"],
        "ΔP isothermal [Pa]": fs["dp"],
    }
    if "thermal" in c:
        th = c["thermal"]
        row.update({"ΔP min [Pa]": th["dp"].min(), "ΔP max [Pa]": th["dp"].max(),
                    "T_He min [K]": th["T_he"].min(), "T_He max [K]": th["T_he"].max(),
                    "Max film load [N/m]": th["load"].max()})
    return row


# =============================================================================
# Streamlit UI  (single page)
# =============================================================================
COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#a16207"]
PASS, WARN, FAIL = "✅ Pass", "⚠️ Warning", "❌ Fail"


def sidebar_inputs():
    sb = st.sidebar
    sb.title("🎈 SPB Inputs")

    sb.subheader("1 · Target float altitude")
    h_t = sb.number_input("Target float altitude [km]", 1.0, 45.0, 14.0, 0.1,
                          help="Drives the required helium mass and all feasibility checks.") * 1000.0

    sb.subheader("2 · Envelope")
    D = sb.number_input("Envelope diameter D [m]", 0.5, 100.0, 4.6, 0.05,
                        help="Fixed design diameter. V_max = πD³/6, A = πD².")
    gsm = sb.number_input("Film areal density [g/m²]", 5.0, 300.0, 69.0, 1.0)
    seam = sb.number_input("Seam / tape overhead [% of area]", 0.0, 50.0, 0.0, 0.5)

    sb.subheader("3 · Mass additions")
    extra = sb.number_input("Payload + hardware additions [kg]", 0.0, 1000.0, 5.0, 0.1,
                            help="Everything carried besides the envelope and helium: payload, "
                                 "valves, rigging, cutter, parachute, ballast…")

    sb.subheader("4 · Structural & flight limits")
    n_allow = sb.number_input("Film allowable line load [N/m]", 100.0, 100000.0, 8000.0, 100.0,
                              help="Envelope seam/film strength per unit width (burst limit).")
    sf = sb.number_input("Structural safety factor", 1.0, 5.0, 1.5, 0.05)
    fl_min = sb.number_input("Min launch free lift [%]", 0.0, 100.0, 10.0, 1.0)
    v_lo, v_hi = sb.slider("Acceptable mean ascent rate [m/s]", 0.5, 15.0, (2.0, 7.0), 0.1)

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
    alpha = sb.slider("Film solar absorptivity α", 0.01, 1.0, 0.15, 0.01)
    eps = sb.slider("Film IR emissivity ε", 0.01, 1.0, 0.85, 0.01)
    albedo = sb.slider("Surface / cloud albedo", 0.0, 1.0, 0.30, 0.01)
    tau = sb.slider("Atmospheric transmissivity τ (per air mass)", 0.3, 1.0, 0.85, 0.01)
    q_ir = sb.number_input("Upwelling Earth IR [W/m²]", 50.0, 400.0, 230.0, 5.0)
    t_sky = sb.number_input("Effective sky temperature [K]", 3.0, 300.0, 216.65, 1.0)
    h_ext = sb.slider("External convection h_ext [W/m²K]", 0.1, 10.0, 0.6, 0.1)
    h_int = sb.slider("Internal convection h_int [W/m²K]", 0.1, 5.0, 2.0, 0.1)
    cp_f = sb.number_input("Film specific heat [J/kgK]", 500.0, 3000.0, 1500.0, 50.0)

    design = Design(D, gsm, seam, extra)
    env = ThermalEnv(lat, int(doy), alpha, eps, albedo, tau, q_ir, t_sky, h_ext, h_int, cp_f)
    limits = dict(n_allow=n_allow, sf=sf, fl_min=fl_min, v_lo=v_lo, v_hi=v_hi)
    return (design, h_t, limits, env,
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


def feasibility_checks(d, h_t, sz, base, am, lim, m_lo, m_hi):
    """Return list of (check, value, limit, status, comment)."""
    rows = []
    n_lim = lim["n_allow"] / lim["sf"]

    # 1 - can it float at the target at all?
    if sz["m_he"] <= 0:
        rows.append(("Lift capacity at target", f"m_dry {d.m_dry:.2f} kg",
                     f"≤ {sz['m_total']:.2f} kg", FAIL,
                     "Dry mass exceeds the total buoyancy of V_max at the target."))
        return rows
    st_fill = PASS if sz["m_he"] >= sz["m_fill"] else FAIL
    rows.append(("Envelope pressurized at target", f"{sz['m_he']:.3f} kg He",
                 f"≥ {sz['m_fill']:.3f} kg (fill)", st_fill,
                 "OK - helium fills V_max at the target." if st_fill == PASS else
                 "Too heavy: helium cannot fill V_max - the balloon would not lift off / "
                 "reach the target. Reduce mass or enlarge D."))
    if st_fill == FAIL:
        return rows

    # 2 - launch & ascent
    fl = base["fl_pct"]
    rows.append(("Launch free lift", f"{fl:.1f} %", f"≥ {lim['fl_min']:.0f} %",
                 PASS if fl >= lim["fl_min"] else WARN,
                 "OK" if fl >= lim["fl_min"] else "Low free lift: slow, wind-sensitive launch."))
    vm = am["v_mean"]
    if np.isfinite(vm):
        s = PASS if lim["v_lo"] <= vm <= lim["v_hi"] else WARN
        msg = ("OK" if s == PASS else
               "Too slow - long cold tropopause crossing, large drift." if vm < lim["v_lo"] else
               "Too fast - high dynamic load and overshoot at full inflation.")
        rows.append(("Mean ascent rate", f"{vm:.2f} m/s",
                     f"{lim['v_lo']:.1f}–{lim['v_hi']:.1f} m/s", s, msg))
    reached = np.isfinite(am["t_reach"])
    rows.append(("Reaches target in sim time", f"{am['h_end']/1000:.2f} km at end",
                 f"≥ {0.98*h_t/1000:.2f} km", PASS if reached else FAIL,
                 f"Reached in {am['t_reach']:.0f} min." if reached else
                 "Did not reach the target - extend sim time or increase lift."))
    if np.isfinite(am["h_full"]):
        rows.append(("Full-inflation altitude", f"{am['h_full']/1000:.2f} km",
                     f"< {h_t/1000:.2f} km", PASS,
                     "Envelope becomes taut below the target, then floats on superpressure."))

    # 3 - float: superpressure & structure
    if "thermal" in base:
        th = base["thermal"]
        dpmin, nmax = th["dp"].min(), th["load"].max()
        rows.append(("Night superpressure (ΔP min)", f"{dpmin:,.0f} Pa", "> 0 Pa",
                     PASS if dpmin > 0 else FAIL,
                     "Stays pressurized all night." if dpmin > 0 else
                     f"Goes slack at night → altitude loss. Need ≥ {m_lo:.3f} kg He."))
        rows.append(("Peak film load (day)", f"{nmax:,.0f} N/m", f"≤ {n_lim:,.0f} N/m (allow/SF)",
                     PASS if nmax <= n_lim else (WARN if nmax <= lim["n_allow"] else FAIL),
                     "OK" if nmax <= n_lim else
                     ("Inside allowable but below the safety factor." if nmax <= lim["n_allow"]
                      else f"Over-pressure / burst risk. Keep He ≤ {m_hi:.3f} kg.")))
    return rows


def main():
    st.set_page_config(page_title="SPB Simulator", page_icon="🎈", layout="wide",
                       initial_sidebar_state="expanded")
    design, h_t, lim, env, var, dyn = sidebar_inputs()

    st.title("Super Pressure Balloon Simulator")
    st.caption("Target altitude + fixed envelope → buoyancy capacity ρ(h)·V_max → required helium "
               "= capacity − dry mass → ascent, 24 h superpressure and structural checks.")

    sz = target_sizing(design, h_t)
    m_he = sz["m_he"]

    # ---- Mass budget --------------------------------------------------------
    st.header("Mass budget for the target altitude")
    c = st.columns(6)
    c[0].metric("Target altitude", f"{h_t/1000:,.2f} km",
                f"ρ = {sz['rho']:.4f} kg/m³", delta_color="off")
    c[1].metric("V_max / area", f"{design.v_max:,.1f} m³", f"{design.area:,.1f} m²",
                delta_color="off")
    c[2].metric("Envelope mass", f"{design.m_env:,.2f} kg")
    c[3].metric("Dry mass (env + additions)", f"{design.m_dry:,.2f} kg")
    c[4].metric("Required helium", f"{m_he:,.3f} kg" if m_he > 0 else "—",
                f"min to fill V_max: {sz['m_fill']:.3f} kg", delta_color="off")
    c[5].metric("Total system mass", f"{design.m_dry + max(m_he, 0):,.2f} kg",
                f"capacity ρ·V_max = {sz['m_total']:.2f} kg", delta_color="off")

    if m_he <= 0 or m_he < sz["m_fill"]:
        dmin = min_diameter(design, h_t)
        st.error(f"❌ Not feasible: the {design.diameter:g} m envelope cannot carry "
                 f"{design.extra:g} kg of additions to {h_t/1000:.1f} km. Maximum additions for this "
                 f"envelope: {max(sz['extra_max'], 0):.2f} kg. Minimum diameter for this load: "
                 f"{dmin:.2f} m.")
        st.stop()

    base = evaluate_case("Baseline", design, m_he, env, dyn["cd"], dyn["c_add"], dyn["t_asc"])
    am = ascent_metrics(base, h_t)
    th = base.get("thermal")
    T_lo, T_hi = (th["T_he"].min(), th["T_he"].max()) if th else (sz["T"], sz["T"])
    m_lo, m_hi = helium_limits(design, h_t, T_lo, T_hi, lim["n_allow"] / lim["sf"])

    budget = pd.DataFrame({
        "Component": ["Envelope (film × area)", "Payload + hardware additions",
                      "Helium (target-driven)", "TOTAL SYSTEM MASS"],
        "Mass [kg]": [design.m_env, design.extra, m_he, design.m_dry + m_he]})
    budget["Share [%]"] = budget["Mass [kg]"] / budget["Mass [kg]"].iloc[-1] * 100.0
    a, b = st.columns(2)
    a.dataframe(budget.style.format({"Mass [kg]": "{:,.3f}", "Share [%]": "{:,.1f}"}),
                hide_index=True, width="stretch")
    pie = go.Figure(go.Pie(labels=budget["Component"][:-1], values=budget["Mass [kg]"][:-1],
                           hole=0.5, sort=False, marker=dict(colors=COLORS)))
    pie.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=10))
    b.plotly_chart(pie, width="stretch")

    # ---- Feasibility --------------------------------------------------------
    st.header("Flight feasibility")
    checks = feasibility_checks(design, h_t, sz, base, am, lim, m_lo, m_hi)
    n_fail = sum(r[3] == FAIL for r in checks)
    n_warn = sum(r[3] == WARN for r in checks)
    if n_fail:
        st.error(f"❌ Design NOT feasible - {n_fail} failed check(s), {n_warn} warning(s).")
    elif n_warn:
        st.warning(f"⚠️ Feasible with {n_warn} warning(s).")
    else:
        st.success("✅ Design feasible: reaches the target, stays pressurized and within structural limits.")
    st.dataframe(pd.DataFrame(checks, columns=["Check", "Value", "Limit", "Status", "Comment"]),
                 hide_index=True, width="stretch")

    ex_lo = sz["m_total"] - design.m_env - m_hi     # lightest additions (burst limit)
    ex_hi = sz["m_total"] - design.m_env - m_lo     # heaviest additions (night slack)
    c = st.columns(4)
    c[0].metric("Helium window @ target", f"{m_lo:.3f} – {m_hi:.3f} kg",
                f"current {m_he:.3f} kg", delta_color="off",
                help="Night pressurization (min) to structural limit (max), using the baseline "
                     "helium temperature extremes.")
    c[1].metric("Additions window @ target", f"{max(ex_lo, 0):.2f} – {ex_hi:.2f} kg"
                if ex_hi > max(ex_lo, 0) else "none",
                f"current {design.extra:.2f} kg", delta_color="off")
    c[2].metric("Isothermal ΔP @ target", f"{sz['dp_iso']:,.0f} Pa")
    c[3].metric("Min diameter (zero ΔP)", f"{min_diameter(design, h_t):.2f} m",
                f"current {design.diameter:g} m", delta_color="off")

    # ---- Cases --------------------------------------------------------------
    cases = build_cases(design, m_he, var)
    with st.spinner("Running ascent and 24 h thermal simulations…"):
        results = [base] + [evaluate_case(lbl, d, mh, env, dyn["cd"], dyn["c_add"], dyn["t_asc"])
                            for lbl, d, mh in cases[1:]]
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
    f_v.add_hrect(y0=lim["v_lo"], y1=lim["v_hi"], fillcolor="green", opacity=0.07, line_width=0)
    f_fill.add_vline(x=h_t / 1000, line_dash="dash", line_color="gray")
    a, b = st.columns(2)
    a.plotly_chart(f_h, width="stretch"); b.plotly_chart(f_v, width="stretch")
    a.plotly_chart(f_a, width="stretch"); b.plotly_chart(f_fill, width="stretch")

    # ---- Thermal ------------------------------------------------------------
    st.header("24-hour thermal cycle at float")
    st.caption("Third day of a 3-day integration, local solar time; altitude and volume held "
               "at each case's float point.")
    n_lim = lim["n_allow"] / lim["sf"]
    f_T = line_fig("Helium temperature", "Hour of day", "T_He [K]")
    f_Tf = line_fig("Film temperature", "Hour of day", "T_film [K]")
    f_dp = line_fig("Superpressure ΔP = P_in − P_amb", "Hour of day", "ΔP [Pa]")
    f_N = line_fig("Film line load N = ΔP·r/2", "Hour of day", "N [N/m]")
    for r, col in zip(results, COLORS * 2):
        if "thermal" not in r:
            continue
        t_ = r["thermal"]
        kw = dict(name=r["label"], line=dict(color=col))
        f_T.add_trace(go.Scatter(x=t_["hours"], y=t_["T_he"], **kw))
        f_Tf.add_trace(go.Scatter(x=t_["hours"], y=t_["T_f"], **kw))
        f_dp.add_trace(go.Scatter(x=t_["hours"], y=t_["dp"], **kw))
        f_N.add_trace(go.Scatter(x=t_["hours"], y=t_["load"], **kw))
    f_dp.add_hline(y=0, line_dash="dot", line_color="gray", annotation_text="slack")
    f_N.add_hline(y=n_lim, line_dash="dash", line_color=COLORS[1], annotation_text="allow / SF")
    f_N.add_hline(y=lim["n_allow"], line_dash="dot", line_color=COLORS[1],
                  annotation_text="allowable")
    for f in (f_T, f_Tf, f_dp, f_N):
        f.update_xaxes(dtick=3, range=[0, 24])
    a, b = st.columns(2)
    a.plotly_chart(f_T, width="stretch"); b.plotly_chart(f_Tf, width="stretch")
    a.plotly_chart(f_dp, width="stretch"); b.plotly_chart(f_N, width="stretch")

    # ---- Design space -------------------------------------------------------
    st.header("Design space for the target altitude")
    st.caption("Approximate: uses the baseline helium temperature extremes. Green band = feasible.")
    # (a) vs additions mass, fixed D
    exs = np.linspace(0.0, max(sz["extra_max"] * 1.15, design.extra * 1.5, 1.0), 150)
    he_req = sz["m_total"] - design.m_env - exs
    f1 = line_fig(f"Helium required vs additions (D = {design.diameter:g} m)",
                  "Payload + hardware additions [kg]", "Helium [kg]", "closest")
    f1.add_trace(go.Scatter(x=exs, y=he_req, name="Required He", line=dict(color=COLORS[0])))
    f1.add_hline(y=m_lo, line_dash="dot", line_color=COLORS[4], annotation_text="night slack limit")
    f1.add_hline(y=m_hi, line_dash="dot", line_color=COLORS[1], annotation_text="structural limit")
    if ex_hi > max(ex_lo, 0):
        f1.add_vrect(x0=max(ex_lo, 0), x1=ex_hi, fillcolor="green", opacity=0.08, line_width=0)
    f1.add_vline(x=design.extra, line_dash="dash", line_color="gray")
    # (b) additions window vs diameter
    Ds = np.linspace(max(0.5, design.diameter * 0.6), design.diameter * 1.8, 120)
    lo_l, hi_l = [], []
    for Dx in Ds:
        dx = replace(design, diameter=Dx)
        s2 = target_sizing(dx, h_t)
        a_lo, a_hi = helium_limits(dx, h_t, T_lo, T_hi, lim["n_allow"] / lim["sf"])
        lo_l.append(max(s2["m_total"] - dx.m_env - a_hi, 0.0))
        hi_l.append(s2["m_total"] - dx.m_env - a_lo)
    lo_l, hi_l = np.array(lo_l), np.array(hi_l)
    ok = hi_l > lo_l
    f2 = line_fig("Feasible additions vs envelope diameter", "Diameter [m]",
                  "Payload + hardware additions [kg]", "closest")
    f2.add_trace(go.Scatter(x=Ds[ok], y=hi_l[ok], name="Max (night slack)",
                            line=dict(color=COLORS[4])))
    f2.add_trace(go.Scatter(x=Ds[ok], y=lo_l[ok], name="Min (structural)", fill="tonexty",
                            fillcolor="rgba(22,163,74,0.12)", line=dict(color=COLORS[1])))
    f2.add_trace(go.Scatter(x=[design.diameter], y=[design.extra], name="Current design",
                            mode="markers", marker=dict(size=12, color="black", symbol="x")))
    a, b = st.columns(2)
    a.plotly_chart(f1, width="stretch"); b.plotly_chart(f2, width="stretch")

    # ---- Model notes --------------------------------------------------------
    st.header("Model notes")
    st.markdown(r"""
**Mass chain.** $V_{max}=\tfrac{\pi}{6}D^3$, $A=\pi D^2$, $m_{env}=A(1+s)\rho_{film}$,
$m_{dry}=m_{env}+m_{add}$.

**Target-driven helium.** A pressurized SPB has fixed volume, so it floats where
$\rho_{air}(h_t)V_{max}=m_{total}$. Hence $m_{He}=\rho_{air}(h_t)V_{max}-m_{dry}$ and
$m_{total}=m_{dry}+m_{He}$. For a fixed envelope and target, the buoyancy capacity
$\rho V_{max}$ is fixed: heavier additions mean less helium and lower superpressure, until the
helium can no longer fill $V_{max}$ ($m_{He}<m_{fill}=\rho V_{max}M_{He}/M_{air}$), at which
point the target is unreachable. Raising the target or the additions changes the helium; the
total system mass follows the target altitude.

**Superpressure.** Isothermal $\Delta P=P(m_{He}/m_{fill}-1)$; diurnal
$\Delta P(t)=m_{He}RT_{He}(t)/(V_{max}M_{He})-P$. Film line load $N=\Delta P\,r/2$.
Helium window: $m_{lo}=PV M_{He}/(RT_{He,min})$, $m_{hi}=(P+2N_{allow}/(SF\,r))VM_{He}/(RT_{He,max})$.

**Ascent.** $(m+C_a\rho V)\ddot h=\rho Vg-mg-\tfrac12\rho\dot h|\dot h|C_dA_p$, $V=\min(V_{He},V_{max})$.

**Thermal.** $m_{env}c_p\dot T_f=\alpha(\tfrac A4I_{dn}+\tfrac A2aI_{dn}\sin\theta)
+\tfrac{\varepsilon A}{2}(q_{IR,E}+\sigma T_{sky}^4)-\varepsilon\sigma AT_f^4-h_{ext}A(T_f-T_{amb})-h_{int}A(T_f-T_{He})$,
$m_{He}c_v\dot T_{He}=h_{int}A(T_f-T_{He})$.

**Limitations.** Spherical envelope, no film creep or elasticity, no helium leakage, altitude
fixed during the thermal cycle, ISA atmosphere, no wind or tropopause cold-soak during ascent.
""")


if __name__ == "__main__":
    main()
