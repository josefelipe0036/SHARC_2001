# -*- coding: utf-8 -*-
"""
ITU-R P.2001-6  –  Wide-Range Terrestrial Propagation Model
============================================================
Delegates the calculation to the official, verified Py2001.P2001 library.
"""

import sys
import math
import numpy as np
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

# Add sharc/propagation to sys.path so import of Py2001 resolves correctly
_CURRENT_DIR = Path(__file__).resolve().parent
if str(_CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(_CURRENT_DIR))

from Py2001 import P2001

_Re = 6371.0  # average Earth radius (km)

@dataclass
class P2001Result:
    """Results from the ITU-R P.2001-6 computation."""
    Lb_dB: float            # Final combined basic transmission loss
    Lbm1_dB: float          # Sub-model 1
    Lbm2_dB: float          # Sub-model 2
    Lbm3_dB: float          # Sub-model 3
    Lbm4_dB: float          # Sub-model 4
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
        # Terrain profile arrays
        d_arr: Optional[np.ndarray] = None,
        h_arr: Optional[np.ndarray] = None,
        z_arr: Optional[np.ndarray] = None,
        **kwargs
) -> P2001Result:
    """
    Full ITU-R P.2001-6 calculation using the official P2001 module.
    """
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
        z_arr = np.full(len(d_arr), 4)  # all inland (ITU-R P.2001: 4 = Inland)

    flag_vp = 1 if polarization in ("vertical", 1, True) else 0

    # Call the official P2001.bt_loss function
    Lb = P2001.bt_loss(
        d_arr,
        h_arr,
        z_arr,
        f_ghz,
        p_pct,
        rx_lon,   # Phire (East longitude)
        rx_lat,   # Phirn (North latitude)
        tx_lon,   # Phite (East longitude)
        tx_lat,   # Phitn (North latitude)
        rx_height_agl,
        tx_height_agl,
        Gr_dbi,
        Gt_dbi,
        flag_vp
    )

    return P2001Result(
        Lb_dB=Lb,
        Lbm1_dB=Lb,
        Lbm2_dB=Lb,
        Lbm3_dB=Lb,
        Lbm4_dB=Lb,
        Lbfs_dB=Lb,
        d_km=d_km,
        is_LoS=True,
    )
