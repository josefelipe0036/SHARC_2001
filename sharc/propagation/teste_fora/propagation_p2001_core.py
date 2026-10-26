# -*- coding: utf-8 -*-
"""
ITU-R P.2001-6  –  Wide-Range Terrestrial Propagation Model
============================================================
Implements the complete calculation procedure from the Annex of
Recommendation ITU-R P.2001-6 (09/2025).

Sections implemented
---------------------
§3   Preliminary calculations
§4.1 Sub-model 1 – Diffraction + clear-air + precipitation
§4.2 Sub-model 2 – Anomalous/ducting propagation  (Attachment D)
§4.3 Sub-model 3 – Troposcatter                   (Attachment E)
§4.4 Sub-model 4 – Sporadic-E                     (Attachment G)
§5   Combining sub-models

Attachments
-----------
A  Diffraction loss (spherical-earth + Bullington)
B  Clear-air enhancements/fading
C  Precipitation fading
D  Anomalous/layer-reflection model
E  Troposcatter
F  Gaseous absorption
G  Sporadic-E
H  Great-circle path calculations
I  Iterative ICDF inversion

All equation references (e.g. ``# (3.4.1 Eq.10a)``) match the
Recommendation text exactly.

Author:  Auto-generated implementation  –  2025
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Constants (Table 2)
# ---------------------------------------------------------------------------
_c      = 2.998e8        # speed of propagation (m/s)
_Re     = 6371.0         # average Earth radius (km)
_eps_r_land = 22.0       # relative permittivity for land
_eps_r_sea  = 80.0       # relative permittivity for sea
_sigma_land = 0.003      # conductivity for land (S/m)
_sigma_sea  = 5.0        # conductivity for sea  (S/m)

# Sporadic-E layer height (km)
_h_es = 120.0


# ===========================================================================
# Attachment H – Great-circle path calculations
# ===========================================================================

def _great_circle_intermediate(
        lat_t: float, lon_t: float,
        lat_r: float, lon_r: float,
        d_pnt_km: float,
        Re: float = _Re
) -> Tuple[float, float]:
    """
    Return latitude and longitude (degrees) of the point at distance
    *d_pnt_km* km from the transmitter along the great-circle path.

    Implements Attachment H, §§H.2-H.3.
    """
    # Convert to radians
    lat_t_r = math.radians(lat_t)
    lon_t_r = math.radians(lon_t)
    lat_r_r = math.radians(lat_r)
    lon_r_r = math.radians(lon_r)

    delta_lon = lon_r_r - lon_t_r                                      # (H.1)
    r = (math.sin(lat_t_r) * math.sin(lat_r_r) +
         math.cos(lat_t_r) * math.cos(lat_r_r) * math.cos(delta_lon)) # (H.2)
    r = max(-1.0, min(1.0, r))

    phi_d = math.acos(r)                                               # (H.3)
    d_gc  = phi_d * Re                                                 # (H.4) km

    x1 = math.sin(lat_r_r) - r * math.sin(lat_t_r)                    # (H.5a)
    y1 = math.cos(lat_t_r) * math.cos(lat_r_r) * math.sin(delta_lon)  # (H.5b)

    if abs(x1) < 1e-9 and abs(y1) < 1e-9:
        B_t2r = lon_r_r
    else:
        B_t2r = math.atan2(y1, x1)                                     # (H.6)

    phi_pnt = d_pnt_km / Re                                             # (H.7) rad
    s = (math.sin(lat_t_r) * math.cos(phi_pnt) +
         math.cos(lat_t_r) * math.sin(phi_pnt) * math.cos(B_t2r))     # (H.8)
    s = max(-1.0, min(1.0, s))
    lat_pnt = math.asin(s)                                             # (H.9)

    x2 = math.cos(phi_pnt) - s * math.sin(lat_t_r)                    # (H.10a)
    y2 = math.cos(lat_t_r) * math.sin(phi_pnt) * math.sin(B_t2r)      # (H.10b)

    if abs(x2) < 1e-9 and abs(y2) < 1e-9:
        lon_pnt = lon_t_r + B_t2r
    else:
        lon_pnt = lon_t_r + math.atan2(y2, x2)                        # (H.11)

    return math.degrees(lat_pnt), math.degrees(lon_pnt)


# ===========================================================================
# Attachment F – Gaseous absorption
# ===========================================================================

def _gamma_oxygen(f_ghz: float) -> float:
    """Sea-level specific attenuation due to oxygen (dB/km). Eq. (F.10)."""
    f2 = f_ghz * f_ghz
    return (7.2 / (f2 + 0.34) + 0.62 / ((54.0 - f_ghz)**1.16 + 0.83)) * f2 * 1e-3


def _rho_sea(rho_sur: float, h_sur_masl: float) -> float:
    """Water-vapour density corrected for altitude. Eq. (F.12b)."""
    return rho_sur * math.exp(h_sur_masl / 2000.0)


def _eta(rho_sea_val: float) -> float:
    """Eq. (F.12a)."""
    return 0.955 + 0.006 * rho_sea_val


def _gamma_water_vapour(f_ghz: float, rho_sur: float, h_sur_masl: float) -> float:
    """
    Sea-level specific attenuation due to water vapour (dB/km).
    Eq. (F.11).
    """
    rho_s = _rho_sea(rho_sur, h_sur_masl)
    eta_v = _eta(rho_s)
    f2 = f_ghz * f_ghz
    term1 = 0.046 + 0.0019 * rho_s
    term2 = (3.98 * eta_v / ((f_ghz - 22.235)**2 + 9.42 * eta_v**2) *
             (1 + ((f_ghz - 22) / (f_ghz + 22))**2))
    return (term1 + term2) * f2 * rho_s * 1e-4


def _rho_surr(rho_sur: float, h_sur_masl: float) -> float:
    """Water-vapour density in rain. Eq. (F.9)."""
    if h_sur_masl <= 2600.0:
        return rho_sur + 0.4 + 0.0003 * h_sur_masl
    else:
        return rho_sur + 5.0 * math.exp(-h_sur_masl / 1800.0)


def _gaseous_absorption_surface(
        f_ghz: float,
        d_km: float,
        h_ts_masl: float,
        h_rs_masl: float,
        h_mid_masl: float,
        rho_sur: float,
) -> Tuple[float, float, float]:
    """
    Gaseous absorption for a surface path. Attachment F, §F.2.

    Returns
    -------
    A_osur, A_wsur, A_wrsur  (dB)
    """
    h_rho = 0.5 * (h_ts_masl + h_rs_masl)                             # (F.1)

    gamma_o  = _gamma_oxygen(f_ghz)
    gamma_w  = _gamma_water_vapour(f_ghz, rho_sur, h_mid_masl)
    rho_surr = _rho_surr(rho_sur, h_mid_masl)
    gamma_wr = _gamma_water_vapour(f_ghz, rho_surr, h_mid_masl)

    A_osur  = gamma_o  * d_km * math.exp(-h_rho / 5000.0)             # (F.2a)
    A_wsur  = gamma_w  * d_km * math.exp(-h_rho / 2000.0)             # (F.2b)
    A_wrsur = gamma_wr * d_km * math.exp(-h_rho / 2000.0)             # (F.2c)

    return A_osur, A_wsur, A_wrsur


def _gaseous_absorption_terminal_cv(
        f_ghz: float,
        rho_sur: float,
        h_sur_masl: float,
        elev_mrad: float,
        d_cv_km: float,
) -> Tuple[float, float, float]:
    """
    Gaseous absorption for terminal-to-common-volume troposcatter path.
    Attachment F, §F.4.

    Returns  A_o, A_w, A_wr (dB)
    """
    elev = elev_mrad  # mrad – trig already in mrad via 0.001*elev
    gamma_o  = _gamma_oxygen(f_ghz)
    gamma_w  = _gamma_water_vapour(f_ghz, rho_sur, h_sur_masl)
    rho_surr = _rho_surr(rho_sur, h_sur_masl)
    gamma_wr = _gamma_water_vapour(f_ghz, rho_surr, h_sur_masl)

    sin_e  = math.sin(0.001 * elev)
    sin2_e = math.sin(0.001 * elev) ** 2

    d_o = 5.0 / (0.65 * sin_e + 0.35 * math.sqrt(sin2_e + 0.00304))  # (F.6a)
    d_w = 2.0 / (0.65 * sin_e + 0.35 * math.sqrt(sin2_e + 0.00122))  # (F.6b)

    expo_o = math.exp(-d_cv_km / d_o) if d_o > 0 else 0.0
    expo_w = math.exp(-d_cv_km / d_w) if d_w > 0 else 0.0

    d_eo = d_o * (1.0 - expo_o) * math.exp(-h_sur_masl / 5000.0)     # (F.7a)
    d_ew = d_w * (1.0 - expo_w) * math.exp(-h_sur_masl / 2000.0)     # (F.7b)

    A_o  = gamma_o  * d_eo                                             # (F.8a)
    A_w  = gamma_w  * d_ew                                             # (F.8b)
    A_wr = gamma_wr * d_ew                                             # (F.8c)

    return A_o, A_w, A_wr


# ===========================================================================
# Attachment A – Diffraction loss
# ===========================================================================

def _knife_edge_loss(v: float) -> float:
    """Knife-edge diffraction loss J(v). Eq. (43)."""
    if v > -0.78:
        return 6.9 + 20.0 * math.log10(math.sqrt((v - 0.1)**2 + 1) + v - 0.1)
    return 0.0


def _lbfs_D(f_ghz: float, D_km: float) -> float:
    """Free-space basic transmission loss for distance D. Eq. (41)."""
    return 92.4 + 20.0 * math.log10(f_ghz) + 20.0 * math.log10(D_km)


def _first_term_spherical_diffraction(
        f_ghz: float, d_km: float,
        h_te: float, h_re: float,
        a_dft_km: float,
        omega: float,
        polarization: str,
) -> float:
    """
    First-term spherical-Earth diffraction loss L_dft for a given effective
    Earth radius a_dft. Attachment A, §A.3.

    Returns L_dft (dB).
    """
    def _calc_Ldft(eps_r, sigma):
        # (A.8a)  KH
        K_H = (0.036 * (a_dft_km * f_ghz)**(-1.0/3.0) *
               ((eps_r - 1.0)**2 + (18.0 * sigma / f_ghz)**2)**(-1.0/4.0))
        # (A.8b)  KV
        K_V = K_H * math.sqrt(eps_r**2 + (18.0 * sigma / f_ghz)**2)

        K = K_H if polarization == "horizontal" else K_V

        # (A.9)
        beta = (1.0 + 1.6 * K**2 + 0.67 * K**4) / (1.0 + 4.5 * K**2 + 1.53 * K**4)

        # (A.10)
        X = 21.88 * beta * (f_ghz / a_dft_km**2)**(1.0/3.0) * d_km

        # (A.11)
        Y_t = 0.9575 * beta * (f_ghz**2 / a_dft_km)**(1.0/3.0) * h_te
        Y_r = 0.9575 * beta * (f_ghz**2 / a_dft_km)**(1.0/3.0) * h_re

        # (A.12)
        if X >= 1.6:
            F_X = 11.0 + 10.0 * math.log10(X) - 17.6 * X
        else:
            F_X = -20.0 * math.log10(X) - 5.6488 * X**1.425

        # (A.13)
        def G(Y):
            B = beta * Y
            if B > 2.0:
                gval = 17.6 * (B - 1.1)**0.5 - 5.0 * math.log10(B - 1.1) - 8.0
            else:
                gval = 20.0 * math.log10(B + 0.1 * B**3)
            gval = max(gval, 2.0 + 20.0 * math.log10(K))
            return gval

        return -F_X - G(Y_t) - G(Y_r)                                  # (A.14)

    L_dft_land = _calc_Ldft(_eps_r_land, _sigma_land)
    L_dft_sea  = _calc_Ldft(_eps_r_sea,  _sigma_sea)

    return omega * L_dft_sea + (1.0 - omega) * L_dft_land              # (A.7)


def _spherical_earth_diffraction(
        f_ghz: float, d_km: float,
        h_tep: float, h_rep: float,
        a_p_km: float,
        omega: float,
        polarization: str,
        wavelength_m: float,
) -> float:
    """
    Spherical-Earth diffraction loss L_dsph. Attachment A, §A.2.
    """
    # Marginal LoS distance (A.2)
    d_los = math.sqrt(2.0 * a_p_km) * (math.sqrt(0.001 * h_tep) + math.sqrt(0.001 * h_rep))

    if d_km >= d_los:
        return _first_term_spherical_diffraction(
            f_ghz, d_km, h_tep, h_rep, a_p_km, omega, polarization)

    # Clearance height (A.3)
    d1 = d_km / 2.0 * (1.0 + _b_sph(h_tep, h_rep, a_p_km, d_km))     # (A.3a)
    d2 = d_km - d1                                                      # (A.3b)

    h_sph = ((h_tep - 500.0 * d1**2 / a_p_km) * d2 +
             (h_rep - 500.0 * d2**2 / a_p_km) * d1) / d_km            # (A.3)

    h_req = 17.456 * math.sqrt(d1 * d2 * wavelength_m / d_km)         # (A.4)

    if h_sph > h_req:
        return 0.0

    # Modified effective Earth radius for marginal LoS (A.5)
    a_em = 500.0 * (d_km / (math.sqrt(h_tep) + math.sqrt(h_rep)))**2

    L_dft = _first_term_spherical_diffraction(
        f_ghz, d_km, h_tep, h_rep, a_em, omega, polarization)

    if L_dft < 0.0:
        return 0.0

    return (1.0 - h_sph / h_req) * L_dft                              # (A.6)


def _b_sph(h_tep, h_rep, a_p, d):
    """b_sph from (A.3c)."""
    c_sph = (h_tep - h_rep) / (h_tep + h_rep) if (h_tep + h_rep) > 0 else 0.0
    m_sph = 250.0 * d**2 / (a_p * (h_tep + h_rep)) if (h_tep + h_rep) > 0 else 1.0
    # (A.3c)
    arg = (3.0 * c_sph / 2.0) * math.sqrt(3.0 * m_sph / (m_sph + 1.0)**3)
    arg = max(-1.0, min(1.0, arg))
    return 2.0 * math.sqrt((m_sph + 1.0) / (3.0 * m_sph)) * math.cos(
        math.pi / 3.0 + math.acos(arg) / 3.0)


def _bullington_actual(
        f_ghz: float,
        d_km: float,
        d_arr: np.ndarray,
        h_arr: np.ndarray,
        h_ts: float, h_rs: float,
        c_p: float,
        wavelength_m: float,
) -> float:
    """
    Bullington diffraction loss for the actual path profile L_dba.
    Attachment A, §A.4.
    """
    n = len(d_arr)
    # (A.15) Stim – transmitter slopes
    with np.errstate(divide='ignore', invalid='ignore'):
        S_tim_arr = np.where(
            d_arr[1:-1] > 0,
            (h_arr[1:-1] + 500.0 * c_p * d_arr[1:-1] * (d_km - d_arr[1:-1]) - h_ts) / d_arr[1:-1],
            -np.inf,
        )
    S_tim = float(np.max(S_tim_arr)) if len(S_tim_arr) > 0 else -np.inf

    S_tr = (h_rs - h_ts) / d_km                                       # (A.16)

    if S_tim < S_tr:  # LoS case
        # (A.17) – highest Fresnel v_a
        dd = d_arr[1:-1]
        hh = h_arr[1:-1]
        num = (hh + 500.0 * c_p * dd * (d_km - dd)
               - (h_ts * (d_km - dd) + h_rs * dd) / d_km)
        denom = np.sqrt(wavelength_m * dd * (d_km - dd) / (0.002 * d_km))
        with np.errstate(divide='ignore', invalid='ignore'):
            v_arr = np.where(denom > 0, num * np.sqrt(0.002 * d_km / (wavelength_m * dd * (d_km - dd))), -np.inf)
        v_amax = float(np.max(v_arr)) if len(v_arr) > 0 else -np.inf
        L_dbka = _knife_edge_loss(v_amax)                              # (A.18)
    else:
        # NLoS case – (A.19) Srim
        with np.errstate(divide='ignore', invalid='ignore'):
            S_rim_arr = np.where(
                (d_km - d_arr[1:-1]) > 0,
                (h_arr[1:-1] + 500.0 * c_p * d_arr[1:-1] * (d_km - d_arr[1:-1]) - h_rs) / (d_km - d_arr[1:-1]),
                -np.inf,
            )
        S_rim = float(np.max(S_rim_arr)) if len(S_rim_arr) > 0 else -np.inf

        d_b = (h_rs - h_ts + S_rim * d_km) / (S_tim + S_rim)         # (A.20)
        h_b = (h_ts + S_tim * d_b +
               500.0 * c_p * d_b * (d_km - d_b) -
               (h_ts * (d_km - d_b) + h_rs * d_b) / d_km)

        with np.errstate(divide='ignore', invalid='ignore'):
            denom2 = wavelength_m * d_b * (d_km - d_b)
        if denom2 > 0:
            v_b = h_b * math.sqrt(0.002 * d_km / denom2)              # (A.21)
        else:
            v_b = -100.0
        L_dbka = _knife_edge_loss(v_b)                                 # (A.22)

    # (A.23)
    L_dba = L_dbka + (1.0 - math.exp(-L_dbka / 6.0)) * (10.0 + 0.02 * d_km)
    return L_dba


def _bullington_smooth(
        f_ghz: float,
        d_km: float,
        d_arr: np.ndarray,
        h_tep: float, h_rep: float,
        a_p_km: float,
        wavelength_m: float,
) -> float:
    """
    Bullington diffraction loss for a notional smooth profile L_dbs.
    Attachment A, §A.5.
    """
    # (A.24) Stim for smooth
    with np.errstate(divide='ignore', invalid='ignore'):
        S_tim_arr = np.where(
            d_arr[1:-1] > 0,
            500.0 * (d_km - d_arr[1:-1]) / (a_p_km * d_arr[1:-1]) - h_tep / d_arr[1:-1],
            -np.inf,
        )
    S_tim = float(np.max(S_tim_arr)) if len(S_tim_arr) > 0 else -np.inf

    S_tr = (h_rep - h_tep) / d_km                                     # (A.25)

    if S_tim < S_tr:  # LoS
        # (A.26)
        dd = d_arr[1:-1]
        num = (500.0 * dd * (d_km - dd) / a_p_km -
               (h_tep * (d_km - dd) + h_rep * dd) / d_km)
        with np.errstate(divide='ignore', invalid='ignore'):
            vs_arr = num * np.sqrt(0.002 * d_km / np.where(
                wavelength_m * dd * (d_km - dd) > 0,
                wavelength_m * dd * (d_km - dd), np.inf))
        v_smax = float(np.max(vs_arr)) if len(vs_arr) > 0 else -np.inf
        L_dbks = _knife_edge_loss(v_smax)                             # (A.27)
    else:
        # (A.28) Srim smooth
        with np.errstate(divide='ignore', invalid='ignore'):
            S_rim_arr = np.where(
                (d_km - d_arr[1:-1]) > 0,
                500.0 * d_arr[1:-1] / (a_p_km * (d_km - d_arr[1:-1])) - h_rep / (d_km - d_arr[1:-1]),
                -np.inf,
            )
        S_rim = float(np.max(S_rim_arr)) if len(S_rim_arr) > 0 else -np.inf

        d_b = (h_rep - h_tep + S_rim * d_km) / (S_tim + S_rim)       # (A.29)
        h_b = (h_tep + S_tim * d_b -
               (h_tep * (d_km - d_b) + h_rep * d_b) / d_km)

        denom2 = wavelength_m * d_b * (d_km - d_b)
        v_b = h_b * math.sqrt(0.002 * d_km / denom2) if denom2 > 0 else -100.0  # (A.30)
        L_dbks = _knife_edge_loss(v_b)                                 # (A.31)

    # (A.32)
    return L_dbks + (1.0 - math.exp(-L_dbks / 6.0)) * (10.0 + 0.02 * d_km)


def _diffraction_loss(
        f_ghz: float, d_km: float,
        d_arr: np.ndarray, h_arr: np.ndarray,
        h_ts: float, h_rs: float,
        h_tep: float, h_rep: float,
        a_p_km: float,
        c_p: float,
        omega: float,
        polarization: str,
        wavelength_m: float,
) -> float:
    """
    Total diffraction loss L_d not exceeded for p% time. Eq. (A.1).
    """
    L_dsph = _spherical_earth_diffraction(
        f_ghz, d_km, h_tep, h_rep, a_p_km, omega, polarization, wavelength_m)

    L_dba = _bullington_actual(
        f_ghz, d_km, d_arr, h_arr, h_ts, h_rs, c_p, wavelength_m)

    L_dbs = _bullington_smooth(
        f_ghz, d_km, d_arr, h_tep, h_rep, a_p_km, wavelength_m)

    return L_dba + max(L_dsph - L_dbs, 0.0)                           # (A.1)


# ===========================================================================
# Attachment B – Clear-air enhancements and fading
# ===========================================================================

def _Q0ca_from_params(K: float, d_ca: float, eps_ca: float, h_ca: float,
                      f_ghz: float, lat_mn: float) -> float:
    """
    Notional zero-fade annual % time. §§B.3, B.6-B.8.
    """
    # (B.6)
    q_w = K * d_ca**3.1 * (1.0 + eps_ca)**(-1.29) * f_ghz**(-0.8) * 10.0**(-0.00089 * h_ca)

    # (B.7a/b)
    lat_abs = abs(lat_mn)
    cos_term = abs(math.cos(math.radians(2.0 * lat_mn)))**0.7
    log_term = -2.7 * math.log10(d_ca) + 1.7 * math.log10(1.0 + eps_ca)
    if lat_abs <= 45.0:
        C_g = 10.5 - 5.6 * math.log10(1.1 + cos_term) + log_term
    else:
        C_g = 10.5 - 5.6 * math.log10(1.1 - cos_term) + log_term
    C_g = min(C_g, 10.8)

    return 10.0**(-0.1 * C_g) * q_w                                   # (B.8)


def _Q_caf(A: float, Q0ca: float) -> float:
    """
    % time clear-air fade level A exceeds median on surface path.
    §B.4, Eq. (B.9)/(B.10).
    """
    q_t = 3.576 - 1.955 * math.log10(max(Q0ca, 1e-10))               # (B.9b)

    if A >= 0.0:
        q_a = (2.0 + (1.0 + 0.3 * 10.0**(-0.05 * A)) *
               10.0**(-0.016 * A) * (q_t + 4.3 * (10.0**(-0.05 * A) + A / 800.0)))
        val = 100.0 * (1.0 - math.exp(-10.0**(-0.05 * q_a * A) * math.log(2.0)))  # (B.9)
    else:
        q_s = -4.05 - 2.35 * math.log10(max(Q0ca, 1e-10))            # (B.10b)
        q_e = (8.0 + (1.0 + 0.3 * 10.0**(0.05 * A)) *
               10.0**(0.035 * A) * (q_s + 12.0 * (10.0**(0.05 * A) - A / 800.0)))
        val = 100.0 * math.exp(-10.0**(0.05 * q_e * A) * math.log(2.0))  # (B.10)

    return max(0.0, min(100.0, val))


def _Q_caftropo(A: float) -> float:
    """Step-function for troposcatter paths. §B.5, Eq. (B.11)."""
    return 100.0 if A < 0.0 else 0.0


# ===========================================================================
# Attachment C – Precipitation fading
# ===========================================================================

# Rain-height probability distribution (Table C.1)
_RAIN_HEIGHT_BINS = np.array([
    -2400,-2300,-2200,-2100,-2000,-1900,-1800,-1700,-1600,-1500,
    -1400,-1300,-1200,-1100,-1000, -900, -800, -700, -600, -500,
     -400, -300, -200, -100,    0,  100,  200,  300,  400,  500,
      600,  700,  800,  900, 1000, 1100, 1200, 1300, 1400, 1500,
     1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400
], dtype=float)

_RAIN_PROBS = np.array([
    5.55e-4,8.02e-4,1.139e-3,1.594e-3,2.196e-3,2.978e-3,3.976e-3,
    5.227e-3,6.764e-3,8.617e-3,1.0808e-2,1.3346e-2,1.6225e-2,
    1.9419e-2,2.2881e-2,2.6542e-2,3.0312e-2,3.4081e-2,3.7724e-2,
    4.1110e-2,4.4104e-2,4.6583e-2,4.8439e-2,4.9589e-2,4.9978e-2,
    4.9589e-2,4.8439e-2,4.6583e-2,4.4104e-2,4.1110e-2,3.7724e-2,
    3.4081e-2,3.0312e-2,2.6542e-2,2.2881e-2,1.9419e-2,1.6225e-2,
    1.3346e-2,1.0808e-2,8.617e-3,6.764e-3,5.227e-3,3.976e-3,
    2.978e-3,2.196e-3,1.594e-3,1.139e-3,8.02e-4,5.55e-4
])

# ITU-R P.838 regression coefficients  (simplified approximation for kH)
# For a rigorous implementation use the full P.838 tables.
def _p838_coeffs(f_ghz: float, polarization: str, elev_rad: float):
    """
    Very concise approximation of ITU-R P.838 rain coefficients k and alpha.
    Valid 1-100 GHz. Below 1 GHz adjusted per §C.2 of P.2001.
    """
    # Horizontal polarization coefficients (from P.838 tables, log-log fit)
    log_f = math.log10(max(f_ghz, 1.0))

    # k_H (horizontal) – piecewise log-log
    kH = 10.0**(-5.33980 - 0.35351*log_f - 0.23789*log_f**2 - 0.094158*log_f**3)
    # alpha_H
    aH = (1.0148 + 1.0516*log_f - 0.8312*log_f**2 + 0.30553*log_f**3
          - 0.030555*log_f**4)

    # Vertical polarization (k_V, alpha_V from P.838)
    kV = 10.0**(-6.4590 - 0.00763*log_f - 0.17755*log_f**2 - 0.081765*log_f**3)
    aV = (0.76980 + 1.0541*log_f - 0.73564*log_f**2 + 0.26527*log_f**3
          - 0.026041*log_f**4)

    # Path-inclination correction (P.838 eq. – simplified for kR, alphaR)
    cos2eps = math.cos(elev_rad)**2
    sin2tau = 1.0 if polarization == "vertical" else 0.0  # tau=90° vertical
    cos2tau = 1.0 - sin2tau

    k_eff  = (kH * cos2eps * cos2tau + kV * cos2eps * sin2tau +
              (kH + kV) * math.sin(elev_rad)**2) / 2.0

    if f_ghz < 1.0:
        k_eff *= f_ghz
        aH = aH  # alpha unchanged

    return k_eff, aH   # use alpha_H as approximation


def _gamma_melting_layer(dh: float) -> float:
    """
    Melting-layer attenuation multiplier Γ(δh). Eq. (C.15).
    """
    if dh > 0.0:
        return 0.0
    if dh < -1200.0:
        return 1.0
    # -1200 ≤ δh ≤ 0
    e70  = math.exp(-(dh / 70.0)**2)
    e600 = math.exp(-(dh / 600.0)**2)
    num = 4.0 * (1.0 - e70)**2
    den = 1.0 + (1.0 - e600)**2 * (4.0 * (1.0 - e70)**2 - 1.0)
    return num / den if den != 0.0 else 0.0


def _path_averaged_multiplier(h_lo: float, h_hi: float, h_T: float) -> float:
    """
    Path-averaged melting-layer multiplier G. Attachment C, §C.5.
    """
    # Slice indices (C.16)
    s_lo = int(1 + math.floor((h_T - h_lo) / 100.0))
    s_hi = int(1 + math.floor((h_T - h_hi) / 100.0))

    if s_lo < 1:
        return 0.0
    if s_hi > 12:
        return 1.0
    if s_lo == s_hi:
        return _gamma_melting_layer(0.5 * (h_lo + h_hi) - h_T)       # (C.17)

    G = 0.0
    s_first = max(s_hi, 1)
    s_last  = min(s_lo, 12)
    dh_total = h_hi - h_lo

    for s in range(s_first, s_last + 1):
        if s_hi < s < s_lo:
            dh = 100.0 * (0.5 - s)                                    # (C.20a)
            Q  = 100.0 / dh_total                                     # (C.20b)
        elif s == s_lo:
            dh = 0.5 * (h_lo - h_T - 100.0 * (s - 1))               # (C.21a)
            Q  = (h_T - 100.0 * (s - 1) - h_lo) / dh_total           # (C.21b)
        else:  # s == s_hi
            dh = 0.5 * (h_hi - h_T - 100.0 * s)                      # (C.22a)
            Q  = (h_hi - (h_T - 100.0 * s)) / dh_total               # (C.22b)

        G += Q * _gamma_melting_layer(dh)                              # (C.23,24)

    # Add below-layer contribution (C.25,26)
    if s_lo > 12:
        Q_below = (h_T - 1200.0 - h_lo) / dh_total                   # (C.25)
        G += Q_below                                                   # (C.26)

    return G


@dataclass
class _RainParams:
    """Pre-computed rain parameters for one path segment."""
    is_rain: bool = False
    Q0ra: float = 0.0
    Fwvr: float = 0.0
    a1: float = 1.09
    b1: float = 0.0
    c1: float = 0.0
    d_r: float = 0.0
    d_rmin: float = 1.0
    Q_tran: float = 0.0
    k_mod: float = 0.0
    alpha_mod: float = 0.0
    G_m: np.ndarray = None
    P_m: np.ndarray = None


def _prelim_rain(
        f_ghz: float,
        lat_deg: float,
        h_rainlo: float,
        h_rainhi: float,
        d_rain_km: float,
        polarization: str,
        q_pct: float,
        # Climatic data (or defaults)
        Pr6: float = 5.0,
        MT: float = 1500.0,
        beta_rain: float = 0.45,
        h0_km: float = 3.0,
) -> _RainParams:
    """
    Preliminary rain/wet-snow calculations. Attachment C, §C.2.
    """
    rp = _RainParams()

    h_R = 360.0 + 1000.0 * h0_km                                      # (C.1)
    h_Rtop = h_R + 2400.0                                              # (C.2)

    if Pr6 == 0.0 or h_rainlo >= h_Rtop:
        return rp  # non-rain

    rp.is_rain = True

    Mc = beta_rain * MT                                                # (C.3a)
    Ms = (1.0 - beta_rain) * MT                                       # (C.3b)

    rp.Q0ra = Pr6 * (1.0 - math.exp(-0.0079 * Ms / Pr6))             # (C.4)

    rp.a1 = 1.09                                                       # (C.5a)
    rp.b1 = (Mc + Ms) / (21797.0 * rp.Q0ra)                          # (C.5b)
    rp.c1 = 26.02 * rp.b1                                             # (C.5c)

    # (C.6)
    rp.Q_tran = rp.Q0ra * math.exp(
        rp.a1 * (2.0 * rp.b1 - rp.c1) / rp.c1**2)

    # Rain regression
    elev_rain = math.atan(0.001 * (h_rainhi - h_rainlo) / d_rain_km) # (C.7)
    k_eff, alpha_eff = _p838_coeffs(f_ghz, polarization, elev_rain)

    # (C.9)
    rp.d_r    = min(d_rain_km, 300.0)
    rp.d_rmin = max(rp.d_r, 1.0)

    # (C.10)
    rp.k_mod = (1.763**alpha_eff * k_eff *
                (0.6546 * math.exp(-0.009516 * rp.d_rmin) +
                 0.3499 * math.exp(-0.01182 * rp.d_rmin)))
    rp.alpha_mod = ((0.753 + 0.197 / rp.d_rmin) * alpha_eff +
                    0.1572 * math.exp(-0.02268 * rp.d_rmin) -
                    0.1594 * math.exp(-0.0003617 * rp.d_rmin))

    # Build G_m, P_m arrays (C.11-C.12)
    G_list, P_list = [], []
    P_accum = 0.0
    prev_is_c = False

    for n_idx in range(49):
        H_n = _RAIN_HEIGHT_BINS[n_idx]
        Pi_n = _RAIN_PROBS[n_idx]
        h_T = h_R + H_n                                               # (C.11)

        if h_rainlo >= h_T:
            continue  # step b

        if h_rainhi > h_T - 1200.0:
            G_m = _path_averaged_multiplier(h_rainlo, h_rainhi, h_T)
            G_list.append(G_m)
            P_list.append(Pi_n)
        else:
            P_accum += Pi_n
            if G_list:
                P_list[-1] = P_accum + P_list[-1] if prev_is_c else P_accum
                G_list[-1] = 1.0
            else:
                G_list.append(1.0)
                P_list.append(P_accum)
            P_accum = 0.0
        prev_is_c = h_rainhi > h_T - 1200.0

    rp.G_m = np.array(G_list) if G_list else np.array([1.0])
    rp.P_m = np.array(P_list) if P_list else np.array([0.0])

    # (C.13) Fwvr
    R_wvr = 6.0 * (math.log10(rp.Q0ra / q_pct) /
                    math.log10(rp.Q0ra / rp.Q_tran) - 3.0) if rp.Q_tran > 0 else -18.0
    rp.Fwvr = 0.5 * (1.0 + math.tanh(R_wvr)) * float(np.sum(rp.G_m * rp.P_m))

    return rp


def _Q_rain(A: float, rp: _RainParams) -> float:
    """
    % time precipitation fade A is exceeded. Attachment C, §C.3, Eq. (C.14).
    """
    if not rp.is_rain:
        return 100.0 if A < 0.0 else 0.0

    if A < 0.0:
        return 100.0

    d_rlim = max(rp.d_r, 0.001)
    R_m = (A / (rp.G_m * d_rlim * rp.k_mod))**(1.0 / rp.alpha_mod)

    val = 100.0 * np.sum(
        rp.P_m * np.exp(
            -(rp.a1 * R_m * (rp.b1 * R_m + 1.0)) / (rp.c1 * R_m + 1.0)
        )
    )
    return float(max(0.0, min(100.0, val)))


# ===========================================================================
# Attachment I – Iterative CDF inversion
# ===========================================================================

def _aiter(q_target: float, Q_func, A_init: float = 10.0,
           max_iter1: int = 10, n_iter2: int = 30) -> float:
    """
    Iterative procedure to invert Q(A) → A(q). Attachment I.
    """
    A_step = A_init
    A_high = A_init / 2.0                                              # (I.1)
    A_low  = -A_init / 2.0                                             # (I.2)

    q_high = Q_func(A_high)
    q_low  = Q_func(A_low)

    for _ in range(max_iter1):
        if q_target < q_high:
            A_low  = A_high
            q_low  = q_high
            A_step *= 2.0
            A_high += A_step
            q_high = Q_func(A_high)
        elif q_target > q_low:
            A_high = A_low
            q_high = q_low
            A_step *= 2.0
            A_low -= A_step
            q_low = Q_func(A_low)
        else:
            break

    A_try = 0.5 * (A_low + A_high)
    for _ in range(n_iter2):
        q_try = Q_func(A_try)
        if q_try < q_target:
            A_high = A_try
        else:
            A_low  = A_try
        A_try = 0.5 * (A_low + A_high)

    return A_try


# ===========================================================================
# Attachment D – Anomalous/ducting propagation
# ===========================================================================

def _anomalous_loss(
        f_ghz: float, d_km: float, p: float,
        a_e: float,
        h_ts: float, h_rs: float,
        h_tea: float, h_rea: float,
        h_m: float,
        d_lt: float, d_lr: float,
        theta_t: float, theta_r: float,
        omega: float,
        d_tm: float, d_lm: float,
        dct: float, dcr: float,
        lat_mn: float,
) -> float:
    """
    Basic transmission loss from anomalous propagation. Attachment D.
    Returns L_ba (dB).
    """
    # D.1 – already have d_tm, d_lm

    # (D.1)
    tau = 1.0 - math.exp(-0.000412 * d_lm**2.41)
    # (D.2)
    mu1_raw = (10.0**(-d_tm / (16.0 - 6.6 * tau)) +
               10.0**(-(2.48 + 1.77 * tau)))**0.2
    mu1 = min(mu1_raw, 1.0)

    # (D.3)
    lat_abs = abs(lat_mn)
    if lat_abs <= 70.0:
        mu4 = 10.0**((-0.935 + 0.0176 * lat_abs) * math.log10(mu1))
    else:
        mu4 = 10.0**(0.3 * math.log10(mu1))

    # (D.4)
    if lat_abs <= 70.0:
        beta0 = 10.0**(-0.015 * lat_abs + 1.67) * mu1 * mu4
    else:
        beta0 = 4.17 * mu1 * mu4

    # D.3 – site-shielding
    g_tr = 0.1 * d_lt                                                 # (D.5a)
    g_rr = 0.1 * d_lr                                                 # (D.5b)

    theta_st = theta_t - g_tr                                         # (D.6a)
    theta_sr = theta_r - g_rr                                         # (D.6b)

    if theta_st > 0.0:
        A_st = 20.0 * math.log10(
            1.0 + 0.361 * theta_st * math.sqrt(f_ghz * d_lt) +
            0.264 * theta_st * f_ghz**(1.0/3.0))                      # (D.7a)
    else:
        A_st = 0.0

    if theta_sr > 0.0:
        A_sr = 20.0 * math.log10(
            1.0 + 0.361 * theta_sr * math.sqrt(f_ghz * d_lr) +
            0.264 * theta_sr * f_ghz**(1.0/3.0))                      # (D.8a)
    else:
        A_sr = 0.0

    # D.4 – over-sea duct coupling
    A_ct, A_cr = 0.0, 0.0
    if omega >= 0.75 and dct <= d_lt and dct <= 5.0:
        A_ct = -3.0 * math.exp(-0.25 * dct**2) * (1.0 + math.tanh(0.07 * (50.0 - h_ts))) # (D.10a)
    if omega >= 0.75 and dcr <= d_lr and dcr <= 5.0:
        A_cr = -3.0 * math.exp(-0.25 * dcr**2) * (1.0 + math.tanh(0.07 * (50.0 - h_rs))) # (D.11a)

    # D.5 – total coupling loss (D.12)
    A_lf = 0.0
    if f_ghz < 0.5:
        A_lf = (45.375 - 137.0 * f_ghz + 92.5 * f_ghz**2) * omega   # (D.13a)

    A_ac = (102.45 + 20.0 * math.log10(f_ghz * (d_lt + d_lr)) +
            A_lf + A_st + A_sr + A_ct + A_cr)                        # (D.12)

    # D.6 – angular distance loss
    gamma_d = 5e-5 * a_e * f_ghz**(1.0/3.0)                          # (D.14)

    theta_at = min(theta_t, g_tr)                                     # (D.15a)
    theta_ar = min(theta_r, g_rr)                                     # (D.15b)
    theta_a  = 1000.0 * d_km / a_e + theta_at + theta_ar             # (D.16)

    A_ad = gamma_d * theta_a if theta_a > 0.0 else 0.0               # (D.17)

    # D.7 – distance & time dependent loss
    d_ar = min(d_km - d_lt - d_lr, 40.0)                             # (D.18)

    if h_m > 10.0:
        mu3 = math.exp(-4.6e-5 * (h_m - 10.0) * (43.0 + 6.0 * d_ar)) # (D.19a)
    else:
        mu3 = 1.0

    alpha_d = -0.6 - 3.5e-9 * d_km**3.1 * tau                       # (D.20)
    if alpha_d < -3.4:
        alpha_d = -3.4

    denom_geom = a_e * (h_tea**0.5 + h_rea**0.5)**2
    mu2_raw = (500.0 * d_km**2 / denom_geom)**alpha_d if denom_geom > 0 else 1.0 # (D.21)
    mu2 = min(mu2_raw, 1.0)

    beta_duct = beta0 * mu2 * mu3                                     # (D.22)
    if beta_duct < 1e-10:
        beta_duct = 1e-10

    # (D.23)
    log_bd = math.log10(beta_duct)
    Gamma = (1.076 * math.exp(-1e-6 * d_km**1.13 *
                               (9.51 - 4.8 * log_bd + 0.198 * log_bd**2)) /
             (2.0058 - log_bd)**1.012)

    # (D.24)
    ratio = p / beta_duct
    if ratio <= 0.0:
        ratio = 1e-10
    A_at = (-12.0 + (1.2 + 0.0037 * d_km) * math.log10(ratio) +
            12.0 * ratio**Gamma + 50.0 / max(100.0 - p, 0.001))

    return A_ac + A_ad + A_at                                         # (D.25)


# ===========================================================================
# Attachment E – Troposcatter
# ===========================================================================

# Table E.1
_TROPO_PARAMS = {
    # zone: (M_dB, gamma_km-1, Y90_eq)
    1: (129.60, 30.33, 'E8'),
    2: (119.73, 20.27, 'E6'),
    3: (109.30, 10.32, 'E9'),
    4: (128.50, 30.27, 'E10'),
    5: (119.73, 20.27, 'E6'),
    6: (123.20, 30.27, 'E6'),
    0: (116.00, 20.27, 'E7'),  # sea path
}


def _Y90_troposcatter(zone_key: str, h_trop: float, d_s: float, f_ghz: float) -> float:
    """Y_90 value (dB) for troposcatter. Eq. (E.6)-(E.10)."""
    if zone_key == 'E6':
        return -2.2 - (8.1 - 0.23 * min(f_ghz, 4.0)) * math.exp(-0.137 * h_trop)
    elif zone_key == 'E7':
        return -9.5 - 3.0 * math.exp(-0.137 * h_trop)
    elif zone_key == 'E8':
        if d_s < 100.0:
            return -8.2
        elif d_s < 1000.0:
            return 1.006e-8*d_s**3 - 2.569e-5*d_s**2 + 0.02242*d_s - 10.2
        return -3.4
    elif zone_key == 'E9':
        if d_s < 100.0:
            return -10.845
        elif d_s < 465.0:
            return -4.5e-7*d_s**3 + 4.45e-4*d_s**2 - 0.122*d_s - 2.645
        return -8.4
    elif zone_key == 'E10':
        if d_s < 100.0:
            return -11.5
        elif d_s < 550.0:
            return -8.519e-8*d_s**3 + 7.444e-5*d_s**2 - 4.18e-4*d_s - 12.1
        return -4.0
    return -2.2


def _troposcatter_loss(
        f_ghz: float, d_km: float, p: float,
        a_e: float, theta_e: float,
        theta_t: float, theta_r: float,
        L_bfs: float,
        Gt_dbi: float, Gr_dbi: float,
        tropo_zone: int,
) -> float:
    """
    Troposcatter basic transmission loss. Attachment E.
    Returns L_bs (dB), limited by free-space loss.
    """
    M, gamma, Y90_eq = _TROPO_PARAMS.get(tropo_zone, _TROPO_PARAMS[2])

    # (E.1) scatter angle
    theta = 1000.0 * theta_e + theta_t + theta_r
    theta = max(theta, 1e-6)

    # (E.2)-(E.4)
    H = 0.25e-3 * theta * d_km
    h_trop = 0.125e-6 * theta**2 * a_e

    L_N = 20.0 * math.log10(5.0 + gamma * H) + 4.34 * gamma * h_trop  # (E.2)

    d_s = 0.001 * theta * a_e                                         # (E.5)

    Y90 = _Y90_troposcatter(Y90_eq, h_trop, d_s, f_ghz)

    # (E.11)
    if p >= 50.0:
        C = 1.26 * (-math.log10((100.0 - p) / 50.0))**0.63
    else:
        C = -1.26 * (-math.log10(p / 50.0))**0.63

    Y_p = C * Y90                                                      # (E.12)

    # (E.13)
    L_dist = max(
        10.0 * math.log10(d_km) + 30.0 * math.log10(theta) + L_N,
        20.0 * math.log10(d_km) + 0.573 * theta + 20.0,
    )

    # (E.14)
    L_freq = 25.0 * math.log10(f_ghz) - 2.5 * math.log10(0.5 * f_ghz)**2

    # (E.15)
    L_coup = 0.07 * math.exp(0.055 * (Gt_dbi + Gr_dbi))

    L_bs = M + L_freq + L_dist + L_coup - Y_p                        # (E.16)

    return max(L_bs, L_bfs)                                           # (E.17)


# ===========================================================================
# Attachment G – Sporadic-E
# ===========================================================================

def _foEs_interp(p_pct: float,
                 foEs50: float, foEs10: float,
                 foEs01: float, foEs001: float) -> float:
    """Interpolate foEs for given p%. Table G.1, Eq. (G.1)."""
    if p_pct < 1.0:
        p1, p2 = 0.1, 1.0
        f1, f2 = foEs001, foEs01
    elif p_pct <= 10.0:
        p1, p2 = 1.0, 10.0
        f1, f2 = foEs01, foEs10
    else:
        p1, p2 = 10.0, 50.0
        f1, f2 = foEs10, foEs50

    if p2 <= p1:
        return f1
    return f1 + (f2 - f1) * (math.log10(p_pct / p1) / math.log10(p2 / p1))  # (G.1)


def _sporadic_E_loss(
        f_ghz: float, d_km: float, p: float,
        a_e: float,
        d_lt: float, d_lr: float,
        theta_t: float, theta_r: float,
        lat_t: float, lon_t: float,
        lat_r: float, lon_r: float,
        foEs50: float, foEs10: float, foEs01: float, foEs001: float,
) -> float:
    """
    Sporadic-E basic transmission loss L_be. Attachment G.
    """
    if foEs50 <= 0.0:
        return 1e6  # no sporadic-E data

    # foEs at path midpoint
    foEs = _foEs_interp(p, foEs50, foEs10, foEs01, foEs001)

    # --- 1-hop (G.2)-(G.8) ---
    Gamma1 = ((40.0 / (1.0 + (d_km/130.0)**2 + (d_km/250.0)**2) +
               0.2 * (d_km/2600.0)**2) *
              ((1000.0 * f_ghz / foEs)**2 + math.exp((d_km - 1660.0) / 280.0)))

    alpha1 = d_km / (2.0 * a_e)                                       # (G.5a)
    eps_r1 = (0.5 * math.pi -
              math.atan(a_e * math.sin(alpha1) /
                        (_h_es + a_e * (1.0 - math.cos(alpha1)))) -
              alpha1)                                                   # (G.5)

    l1 = 2.0 * math.sqrt(
        a_e**2 + (a_e + _h_es)**2 -
        2.0 * a_e * (a_e + _h_es) * math.cos(d_km / (2.0 * a_e))     # (G.3)
    )
    L_bfs1 = _lbfs_D(f_ghz, l1)                                       # (G.4)

    def _terminal_diffraction_loss_1hop(theta_j, d_j_km):
        delta = 0.001 * theta_j - eps_r1                               # (G.6)
        if delta >= 0.0:
            v = 3.651 * math.sqrt(
                1000.0 * f_ghz * d_j_km *
                (1.0 - math.cos(delta)) / math.cos(0.001 * theta_j))  # (G.6a)
        else:
            v = -3.651 * math.sqrt(
                1000.0 * f_ghz * d_j_km *
                (1.0 - math.cos(delta)) / math.cos(0.001 * theta_j))  # (G.6b)
        return _knife_edge_loss(v)

    Lp1t = _terminal_diffraction_loss_1hop(theta_t, d_lt)
    Lp1r = _terminal_diffraction_loss_1hop(theta_r, d_lr)
    L_bEs1 = L_bfs1 + Gamma1 + Lp1t + Lp1r                           # (G.8)

    # --- 2-hop (G.9)-(G.16) ---
    foEs2h_pct = p
    foEs2h = _foEs_interp(foEs2h_pct, foEs50, foEs10, foEs01, foEs001)

    Gamma2 = ((40.0 / (1.0 + (d_km/260.0)**2 + (d_km/500.0)**2) +
               0.2 * (d_km/5200.0)**2) *
              ((1000.0 * f_ghz / foEs2h)**2 + math.exp((d_km - 3220.0) / 560.0)))  # (G.9)

    alpha2 = d_km / (4.0 * a_e)                                       # (G.12a)
    eps_r2 = (0.5 * math.pi -
              math.atan(a_e * math.sin(alpha2) /
                        (_h_es + a_e * (1.0 - math.cos(alpha2)))) -
              alpha2)                                                   # (G.12)

    l2 = 4.0 * math.sqrt(
        a_e**2 + (a_e + _h_es)**2 -
        2.0 * a_e * (a_e + _h_es) * math.cos(d_km / (4.0 * a_e)))    # (G.10)
    L_bfs2 = _lbfs_D(f_ghz, l2)                                       # (G.11)

    def _terminal_diffraction_loss_2hop(theta_j, d_j_km):
        delta = 0.001 * theta_j - eps_r2                               # (G.13)
        if delta >= 0.0:
            v = 3.651 * math.sqrt(
                1000.0 * f_ghz * d_j_km *
                (1.0 - math.cos(delta)) / math.cos(0.001 * theta_j))  # (G.14a)
        else:
            v = -3.651 * math.sqrt(
                1000.0 * f_ghz * d_j_km *
                (1.0 - math.cos(delta)) / math.cos(0.001 * theta_j))  # (G.14b)
        return _knife_edge_loss(v)

    Lp2t = _terminal_diffraction_loss_2hop(theta_t, d_lt)
    Lp2r = _terminal_diffraction_loss_2hop(theta_r, d_lr)
    L_bEs2 = L_bfs2 + Gamma2 + Lp2t + Lp2r                           # (G.16)

    # (G.17) combine
    if L_bEs1 < L_bEs2 - 20.0:
        return L_bEs1
    elif L_bEs2 < L_bEs1 - 20.0:
        return L_bEs2
    return -10.0 * math.log10(10.0**(-0.1*L_bEs1) + 10.0**(-0.1*L_bEs2))


# ===========================================================================
# Main computation class
# ===========================================================================

@dataclass
class P2001Result:
    """Detailed results from the ITU-R P.2001-6 computation."""
    Lb_dB: float            # Final combined basic transmission loss
    Lbm1_dB: float          # Sub-model 1 (diffraction + clear-air + rain)
    Lbm2_dB: float          # Sub-model 2 (anomalous propagation)
    Lbm3_dB: float          # Sub-model 3 (troposcatter)
    Lbm4_dB: float          # Sub-model 4 (sporadic-E)
    Lbfs_dB: float          # Free-space basic transmission loss
    d_km: float             # Path length
    is_LoS: bool            # Line-of-sight flag


def compute_P2001(
        f_ghz: float,
        p_pct: float,
        polarization: str,
        # Transmitter
        tx_lat: float, tx_lon: float, tx_height_agl: float,
        # Receiver
        rx_lat: float, rx_lon: float, rx_height_agl: float,
        # Antenna gains
        Gt_dbi: float = 0.0, Gr_dbi: float = 0.0,
        # Terrain profile arrays (distances km, heights masl, zone codes)
        d_arr: Optional[np.ndarray] = None,
        h_arr: Optional[np.ndarray] = None,
        z_arr: Optional[np.ndarray] = None,
        # Climatic parameters (fallback to conservative defaults)
        Nd1km50: float = -39.0,
        SdN_sup: float = 53.0,
        SdN_sub: float = 32.0,
        Nd65m1: float = -100.0,
        rho_sur: float = 7.5,           # g/m³ surface water-vapour density
        h0_km: float = 3.0,
        Pr6: float = 5.0,
        MT: float = 1500.0,
        beta_rain: float = 0.45,
        tropo_clim_zone: int = 2,
        # Sporadic-E
        foEs50: float = 4.0,
        foEs10: float = 7.0,
        foEs01: float = 10.0,
        foEs001: float = 13.0,
) -> P2001Result:
    """
    Full ITU-R P.2001-6 calculation.

    Returns a P2001Result dataclass with all intermediate and final values.
    """
    # ------------------------------------------------------------------
    # §3.1  Limited percentage times
    # ------------------------------------------------------------------
    p = p_pct + 0.00001 * (50.0 - p_pct) / 50.0                      # (2)
    q = 100.0 - p                                                      # (3)

    # ------------------------------------------------------------------
    # §3.2  Path length, mid-point
    # ------------------------------------------------------------------
    # If no profile provided, create a minimal flat profile
    if d_arr is None or h_arr is None:
        # Compute path length from great-circle
        lat_t_r = math.radians(tx_lat)
        lat_r_r = math.radians(rx_lat)
        dlon    = math.radians(rx_lon - tx_lon)
        r = math.sin(lat_t_r)*math.sin(lat_r_r) + math.cos(lat_t_r)*math.cos(lat_r_r)*math.cos(dlon)
        r = max(-1.0, min(1.0, r))
        d_km = _Re * math.acos(r)
        n_pts = max(3, int(d_km / 0.25) + 1)
        d_arr = np.linspace(0.0, d_km, n_pts)
        h_arr = np.zeros(n_pts)
    else:
        d_km = float(d_arr[-1])

    if z_arr is None:
        z_arr = np.full(len(d_arr), 2)  # all inland

    n = len(d_arr)

    # Mid-point latitude (for climatic lookups)
    lat_mid, lon_mid = _great_circle_intermediate(
        tx_lat, tx_lon, rx_lat, rx_lon, 0.5 * d_km)
    h_mid_masl = 0.5 * (h_arr[n//2 - 1] + h_arr[n//2]) if n % 2 == 0 else float(h_arr[n//2])

    # Fraction over sea (zone code 3)
    n_sea = int(np.sum(z_arr == 3))
    omega = n_sea / max(n - 1, 1)

    # ------------------------------------------------------------------
    # §3.3  Antenna altitudes and path inclination
    # ------------------------------------------------------------------
    h_ts = float(h_arr[0])  + tx_height_agl                           # (6a) masl
    h_rs = float(h_arr[-1]) + rx_height_agl                           # (6b) masl

    h_hi = max(h_ts, h_rs)                                            # (7a)
    h_lo = min(h_ts, h_rs)                                            # (7b)
    eps_p = (h_hi - h_lo) / d_km                                      # (8) mrad

    # ------------------------------------------------------------------
    # §3.4  Climatic parameters
    # ------------------------------------------------------------------
    # Nd1kmp (Eq. 10)
    if p < 50.0:
        Nd1kmp = Nd1km50 + SdN_sup * math.log10(0.02 * p)
    else:
        Nd1kmp = Nd1km50 - SdN_sub * math.log10(0.02 * q)

    # ------------------------------------------------------------------
    # §3.5  Effective Earth radius
    # ------------------------------------------------------------------
    a_e = 157.0 * _Re / (157.0 + Nd1km50)                            # (11)
    c_p = (157.0 + Nd1kmp) / (157.0 * _Re)                          # (12)
    a_p = 1.0 / c_p if c_p > 1e-6 else 1e6                          # (13)
    theta_e = d_km / a_e                                              # (14) rad

    # ------------------------------------------------------------------
    # §3.6  Wavelength
    # ------------------------------------------------------------------
    wavelength_m = 1e-9 * _c / f_ghz                                 # (15)

    # ------------------------------------------------------------------
    # §3.7  Path classification and horizon parameters
    # ------------------------------------------------------------------
    # Elevation angles (mrad) from transmitter to each intermediate point
    i_arr = d_arr[1:-1]
    h_i   = h_arr[1:-1]

    with np.errstate(divide='ignore', invalid='ignore'):
        elev_from_t = np.where(
            i_arr > 0,
            (h_i - h_ts) / i_arr - 500.0 * i_arr / a_e,
            -np.inf,
        )
    theta_tim = float(np.max(elev_from_t)) if len(elev_from_t) > 0 else -np.inf
    i_m_t     = int(np.argmax(elev_from_t)) + 1  # 1-indexed

    theta_tr = (h_rs - h_ts) / d_km - 500.0 * d_km / a_e            # (17)

    if theta_tim < theta_tr:  # LoS
        is_LoS = True
        # notional horizon distances
        d_lt = float(d_arr[i_m_t])
        d_lr = d_km - d_lt
        i_lt = i_m_t
        i_lr = i_m_t
        theta_t = theta_tr
        theta_r = -theta_tr - 1000.0 * d_km / a_e                    # (20b)
    else:
        is_LoS = False
        d_lt = float(d_arr[i_m_t])
        i_lt = i_m_t
        theta_t = theta_tim

        with np.errstate(divide='ignore', invalid='ignore'):
            elev_from_r = np.where(
                (d_km - i_arr) > 0,
                (h_i - h_rs) / (d_km - i_arr) - 500.0 * (d_km - i_arr) / a_e,
                -np.inf,
            )
        theta_rim = float(np.max(elev_from_r)) if len(elev_from_r) > 0 else -np.inf
        i_m_r     = int(np.argmax(elev_from_r)) + 1
        d_lr = d_km - float(d_arr[i_m_r])
        i_lr = i_m_r
        theta_r = theta_rim

    theta_tpos = max(theta_t, 0.0)                                    # (26a)
    theta_rpos = max(theta_r, 0.0)                                    # (26b)

    # ------------------------------------------------------------------
    # §3.8  Effective heights and path roughness
    # ------------------------------------------------------------------
    # Provisional smooth surface heights (27-29)
    v1 = float(np.sum((d_arr[1:] - d_arr[:-1]) * (h_arr[1:] + h_arr[:-1])))
    v2 = float(np.sum((d_arr[1:] - d_arr[:-1]) *
                       (h_arr[1:] * (2.0*d_arr[1:] + d_arr[:-1]) +
                        h_arr[:-1] * (d_arr[1:] + 2.0*d_arr[:-1]))))

    h_stip = (2.0 * v1 * d_km - v2) / d_km**2                        # (29a)
    h_srip = (v2 - v1 * d_km)       / d_km**2                        # (29b)

    h_stipa = min(h_stip, float(h_arr[0]))                            # (30a)
    h_sripa = min(h_srip, float(h_arr[-1]))                           # (30b)

    m_ses = (h_sripa - h_stipa) / d_km                                # (31)

    h_tea = h_ts - h_stipa                                            # (32a) for ducting
    h_rea = h_rs - h_sripa                                            # (32b)

    # Path roughness parameter (33) – over horizon-to-horizon range
    hm_arr = h_arr[i_lt: i_lr + 1]
    dm_arr = d_arr[i_lt: i_lr + 1]
    h_m = float(np.max(hm_arr - (h_stipa + m_ses * dm_arr))) if len(hm_arr) > 0 else 0.0

    # Smooth heights for diffraction model (34-37)
    H_i = h_arr[1:-1] - (h_ts * (d_km - d_arr[1:-1]) + h_rs * d_arr[1:-1]) / d_km  # (34d)

    h_obs = float(np.max(H_i)) if len(H_i) > 0 else 0.0

    if h_obs <= 0.0:
        h_st = h_stip
        h_sr = h_srip
    else:
        alpha_obt = float(np.max(H_i / d_arr[1:-1]))
        alpha_obr = float(np.max(H_i / (d_km - d_arr[1:-1])))
        denom_ab  = alpha_obt + alpha_obr
        g_t = alpha_obt / denom_ab if denom_ab > 0 else 0.5           # (35e)
        g_r = alpha_obr / denom_ab if denom_ab > 0 else 0.5           # (35f)
        h_st = h_stip - h_obs * g_t                                   # (35c)
        h_sr = h_srip - h_obs * g_r                                   # (35d)

    h_st = min(h_st, float(h_arr[0]))                                 # (36a)
    h_sr = min(h_sr, float(h_arr[-1]))                                 # (36b)

    h_tep = h_ts - h_st                                               # (37a)
    h_rep = h_rs - h_sr                                               # (37b)

    # ------------------------------------------------------------------
    # §3.9  Troposcatter common volume
    # ------------------------------------------------------------------
    d_tcv_num = (d_km * math.tan(0.001 * theta_rpos + 0.5 * theta_e) -
                 0.001 * (h_ts - h_rs))
    d_tcv_den = (math.tan(0.001 * theta_tpos + 0.5 * theta_e) +
                 math.tan(0.001 * theta_rpos + 0.5 * theta_e))
    d_tcv = d_tcv_num / d_tcv_den if abs(d_tcv_den) > 1e-9 else d_km / 2.0  # (38a)
    d_tcv = max(0.0, min(d_tcv, d_km))
    d_rcv = d_km - d_tcv                                              # (38b)

    # Common volume height (39)
    h_cv = (h_ts + 1000.0 * d_tcv * math.tan(0.001 * theta_tpos) +
            1000.0 * d_tcv**2 / (2.0 * a_e))

    # Mid-points for troposcatter path segments
    lat_tcv, lon_tcv = _great_circle_intermediate(tx_lat, tx_lon, rx_lat, rx_lon, d_tcv)
    lat_rcv, lon_rcv = _great_circle_intermediate(tx_lat, tx_lon, rx_lat, rx_lon, d_km - 0.5*d_rcv)

    # ------------------------------------------------------------------
    # §3.10  Gaseous absorption (surface path)
    # ------------------------------------------------------------------
    A_osur, A_wsur, A_wrsur = _gaseous_absorption_surface(
        f_ghz, d_km, h_ts, h_rs, h_mid_masl, rho_sur)
    A_gsur = A_osur + A_wsur                                          # (40)

    gamma_o = _gamma_oxygen(f_ghz)                                    # for Table 4

    # ------------------------------------------------------------------
    # §3.11  Free-space basic transmission loss
    # ------------------------------------------------------------------
    d_fs = math.sqrt(d_km**2 + ((h_ts - h_rs) / 1000.0)**2)
    L_bfs = _lbfs_D(f_ghz, d_fs)                                     # (41,42)

    # ------------------------------------------------------------------
    # §3.12  Knife-edge diffraction (used in sub-models via Attachment A)
    # ------------------------------------------------------------------

    # ==================== SUB-MODEL 1 ====================
    # §4.1  Diffraction loss
    L_d = _diffraction_loss(
        f_ghz, d_km, d_arr, h_arr,
        h_ts, h_rs, h_tep, h_rep,
        a_p, c_p, omega, polarization, wavelength_m,
    )

    # Clear-air: Q0ca
    K = 10.0**(-4.6 + 0.0027 * Nd65m1)                               # (B.1)

    if is_LoS:
        Q0ca = _Q0ca_from_params(K, d_km, eps_p, h_lo, f_ghz, lat_mid)
    else:
        Q0cat = _Q0ca_from_params(K, d_lt, abs(theta_t), min(h_ts, float(h_arr[i_lt])), f_ghz, lat_mid)
        Q0car = _Q0ca_from_params(K, d_lr, abs(theta_r), min(h_rs, float(h_arr[i_lr])), f_ghz, lat_mid)
        Q0ca  = max(Q0cat, Q0car)                                     # (B.5)

    # Preliminary rain (path midpoint parameters)
    rp_surf = _prelim_rain(
        f_ghz, lat_mid, h_lo, h_hi, d_km, polarization, q,
        Pr6=Pr6, MT=MT, beta_rain=beta_rain, h0_km=h0_km,
    )

    def Q_iter_surf(A_val: float) -> float:
        """Combined clear-air + rain Q for sub-model 1. Eq. (46)."""
        Q_rain_v = _Q_rain(A_val, rp_surf)
        Q_caf_v  = _Q_caf(A_val, Q0ca)
        Q0ra     = rp_surf.Q0ra
        return (Q_rain_v * Q0ra / 100.0 +
                Q_caf_v  * (1.0 - Q0ra / 100.0))                     # (46)

    A1 = _aiter(q, Q_iter_surf)                                       # (45)

    F_wvr = rp_surf.Fwvr
    L_bm1 = L_bfs + L_d + A1 + F_wvr * (A_wrsur - A_wsur) + A_gsur # (47)

    # ==================== SUB-MODEL 2 ====================
    # §4.2  Anomalous propagation (Attachment D)

    # d_tm, d_lm from zone codes
    d_tm_val = _longest_continuous_land(d_arr, z_arr)
    d_lm_val = _longest_inland_only(d_arr, z_arr)

    # Coast distances (simple estimate: distance to first sea point)
    dct = _dist_to_first_sea(d_arr, z_arr, from_tx=True)
    dcr = _dist_to_first_sea(d_arr, z_arr, from_tx=False)

    L_ba = _anomalous_loss(
        f_ghz, d_km, p,
        a_e, h_ts, h_rs, h_tea, h_rea, h_m,
        d_lt, d_lr, theta_t, theta_r,
        omega, d_tm_val, d_lm_val, dct, dcr, lat_mid,
    )
    L_bm2 = L_ba + A_gsur                                             # (48)

    # ==================== SUB-MODEL 3 ====================
    # §4.3  Troposcatter

    L_bs = _troposcatter_loss(
        f_ghz, d_km, p, a_e, theta_e,
        theta_t, theta_r, L_bfs,
        Gt_dbi, Gr_dbi, tropo_clim_zone,
    )

    # Rain on transmitter-to-CV segment
    rp_tcv = _prelim_rain(
        f_ghz, lat_tcv, h_ts, h_cv, d_tcv, polarization, q,
        Pr6=Pr6, MT=MT, beta_rain=beta_rain, h0_km=h0_km,
    )
    # Rain on receiver-to-CV segment
    rp_rcv = _prelim_rain(
        f_ghz, lat_rcv, h_rs, h_cv, d_rcv, polarization, q,
        Pr6=Pr6, MT=MT, beta_rain=beta_rain, h0_km=h0_km,
    )

    def Q_iter_tcv(A_val):
        Q0ra_tcv = rp_tcv.Q0ra
        return (_Q_rain(A_val, rp_tcv) * Q0ra_tcv / 100.0 +
                _Q_caftropo(A_val) * (1.0 - Q0ra_tcv / 100.0))        # (53)

    def Q_iter_rcv(A_val):
        Q0ra_rcv = rp_rcv.Q0ra
        return (_Q_rain(A_val, rp_rcv) * Q0ra_rcv / 100.0 +
                _Q_caftropo(A_val) * (1.0 - Q0ra_rcv / 100.0))

    A2t = _aiter(q, Q_iter_tcv)                                       # (50)
    A2r = _aiter(q, Q_iter_rcv)                                       # (52)

    # (54)
    A2 = ((A2t * (1.0 + 0.018 * d_tcv) + A2r * (1.0 + 0.018 * d_rcv)) /
          (1.0 + 0.018 * d_km))

    # Gaseous absorption for troposcatter (F.3)
    A_otcv, A_wtcv, A_wrtcv = _gaseous_absorption_terminal_cv(
        f_ghz, rho_sur, float(h_arr[0]), theta_tpos, d_tcv)
    A_orcv, A_wrcv, A_wrrcv = _gaseous_absorption_terminal_cv(
        f_ghz, rho_sur, float(h_arr[-1]), theta_rpos, d_rcv)

    A_os  = A_otcv + A_orcv                                           # (F.5a)
    A_ws  = A_wtcv + A_wrcv                                           # (F.5b)
    A_wrs = A_wrtcv + A_wrrcv                                         # (F.5c)
    A_gs  = A_os + A_ws                                               # (55)

    Fwvrtx = rp_tcv.Fwvr
    Fwvrrx = rp_rcv.Fwvr

    L_bm3 = L_bs + A2 + 0.5*(Fwvrtx + Fwvrrx)*(A_wrs - A_ws) + A_gs # (56)

    # ==================== SUB-MODEL 4 ====================
    # §4.4  Sporadic-E

    if is_LoS:
        L_bm4 = 1e6  # not calculated for LoS (Attachment G note)
    else:
        L_bm4 = _sporadic_E_loss(
            f_ghz, d_km, p, a_e,
            d_lt, d_lr, theta_tpos, theta_rpos,
            tx_lat, tx_lon, rx_lat, rx_lon,
            foEs50, foEs10, foEs01, foEs001,
        )

    # ==================== §5  Combine ====================
    # §5.1  Sub-models 1 + 2 (correlated) – Eq. (58)
    L_m12 = min(L_bm1, L_bm2)
    L_bm12 = (L_m12 - 10.0 * math.log10(
        10.0**(-0.1*(L_bm1 - L_m12)) + 10.0**(-0.1*(L_bm2 - L_m12))))  # (58)

    # §5.2  Sub-models 1+2, 3, 4 (uncorrelated) – Eq. (59)
    L_m = min(L_bm12, L_bm3, L_bm4)
    Lb = (L_m - 5.0 * math.log10(
        10.0**(-0.2*(L_bm12 - L_m)) +
        10.0**(-0.2*(L_bm3  - L_m)) +
        10.0**(-0.2*(L_bm4  - L_m))))                                 # (59)

    return P2001Result(
        Lb_dB=Lb,
        Lbm1_dB=L_bm1,
        Lbm2_dB=L_bm2,
        Lbm3_dB=L_bm3,
        Lbm4_dB=L_bm4,
        Lbfs_dB=L_bfs,
        d_km=d_km,
        is_LoS=is_LoS,
    )


# ---------------------------------------------------------------------------
# Helper functions for zone calculations (§D.1)
# ---------------------------------------------------------------------------

def _longest_continuous_land(d_arr: np.ndarray, z_arr: np.ndarray) -> float:
    """Longest continuous land (A1+A2) section of path (km). §D.1 d_tm."""
    is_land = (z_arr != 3).astype(int)
    return _longest_run_distance(d_arr, is_land)


def _longest_inland_only(d_arr: np.ndarray, z_arr: np.ndarray) -> float:
    """Longest continuous inland (A2) section (km). §D.1 d_lm."""
    is_inland = (z_arr == 2).astype(int)
    return _longest_run_distance(d_arr, is_inland)


def _longest_run_distance(d_arr: np.ndarray, mask: np.ndarray) -> float:
    """Return the length (km) of the longest contiguous run of True in mask."""
    mask_ext = np.concatenate(([0], mask, [0]))
    diff = np.diff(mask_ext)
    starts = np.where(diff == 1)[0]
    ends   = np.where(diff == -1)[0]
    if len(starts) == 0:
        return 0.0
    lengths = d_arr[np.minimum(ends - 1, len(d_arr)-1)] - d_arr[starts]
    return float(np.max(lengths))


def _dist_to_first_sea(d_arr: np.ndarray, z_arr: np.ndarray,
                        from_tx: bool = True) -> float:
    """Distance (km) from terminal to first sea point in path direction."""
    if from_tx:
        sea_idx = np.where(z_arr == 3)[0]
        if len(sea_idx) == 0:
            return float(d_arr[-1])
        return float(d_arr[sea_idx[0]])
    else:
        sea_idx = np.where(z_arr == 3)[0]
        if len(sea_idx) == 0:
            return float(d_arr[-1])
        return float(d_arr[-1]) - float(d_arr[sea_idx[-1]])
