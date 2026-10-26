# -*- coding: utf-8 -*-
"""
Parameters for ITU-R P.2001-6 Wide-Range Terrestrial Propagation Model.

Frequency range: 30 MHz – 50 GHz
Distance range:  3 km – 1 000 km (reliable)
"""

from dataclasses import dataclass, field
from typing import Optional, Union
import numpy as np


from sharc.parameters.parameters_base import ParametersBase


@dataclass
class ParametersP2001(ParametersBase):
    """
    Input parameters for the ITU-R P.2001-6 propagation model.

    All parameters follow the notation used in the Recommendation.

    Attributes
    ----------
    f_ghz : float
        Carrier frequency in GHz. Range: 0.03 – 50 GHz.
    p : float
        Percentage of an average year for which the predicted basic
        transmission loss is NOT exceeded.  Range: 0 % – 100 %.
    polarization : str
        ``"horizontal"`` or ``"vertical"``.

    -- Transmitter --
    tx_lat : float
        Latitude  of transmitter (°N positive).
    tx_lon : float
        Longitude of transmitter (°E positive).
    tx_height_agl : float
        Height of transmitter antenna above ground level (m). Must be > 0.

    -- Receiver --
    rx_lat : float
        Latitude  of receiver (°N positive).
    rx_lon : float
        Longitude of receiver (°E positive).
    rx_height_agl : float
        Height of receiver antenna above ground level (m). Must be > 0.

    -- Antenna gains (used in troposcatter) --
    Gt_dbi : float
        Transmitter antenna gain in path direction (dBi).
    Gr_dbi : float
        Receiver  antenna gain in path direction (dBi).

    -- Terrain profile --
    profile_distances_km : np.ndarray, optional
        Terrain profile – distances from the transmitter (km).
        Must contain at least 3 equally-spaced values (0 … d).
        If None a flat profile at 0 m ASL is assumed.
    profile_heights_masl : np.ndarray, optional
        Terrain profile – heights above sea level (m).
        Same length as ``profile_distances_km``.
    profile_zone_codes : np.ndarray, optional
        Radio-climatic zone code at each profile point.
        1 = Coastal land (A1), 2 = Inland (A2), 3 = Sea (B).
        If None, all points are assumed Inland (2).

    -- Climatic / environment overrides (optional) --
        When None the model uses its internal look-up tables.
    Nd1km50 : float, optional
        Median refractivity gradient in lowest 1 km (N-units, negative).
    SdN_sup : float, optional
        ``DN_SupSlope`` value (N-units).
    SdN_sub : float, optional
        ``DN_SubSlope`` value (N-units).
    Nd65m1 : float, optional
        Refractivity gradient in lowest 65 m, exceeded 1 % (N-units).
    surf_wv_density : float, optional
        Surface water-vapour density (g/m³) at path midpoint.
    h0_km : float, optional
        Mean zero-degree isotherm height above sea level (km).
    Pr6 : float, optional
        Probability of rain exceeding 0 mm/h in any 6-minute period (%).
    MT : float, optional
        Mean annual total rain accumulation (mm/year).
    beta_rain : float, optional
        Fraction of ``MT`` falling in convective events.
    tropo_clim_zone : int, optional
        Troposcatter climatic zone 1-6 (from TropoClim.txt). If None,
        zone 2 (moderate climate) is used as a conservative default.

    -- Sporadic-E foEs --
    foEs50_mhz : float, optional
        Sporadic-E critical frequency exceeded 50 % (MHz).
    foEs10_mhz : float, optional
        Sporadic-E critical frequency exceeded 10 % (MHz).
    foEs01_mhz : float, optional
        Sporadic-E critical frequency exceeded  1 % (MHz).
    foEs001_mhz : float, optional
        Sporadic-E critical frequency exceeded  0.1 % (MHz).
    """

    # --- Mandatory ---
    f_ghz: float = 1.0
    p: Union[float, str] = 50.0
    polarization: str = "horizontal"

    tx_lat: float = -23.55
    tx_lon: float = -46.63
    tx_height_agl: float = 30.0

    rx_lat: float = -23.17
    rx_lon: float = -45.89
    rx_height_agl: float = 3.0

    Gt_dbi: float = 0.0
    Gr_dbi: float = 0.0

    # --- Terrain profile (optional) ---
    profile_distances_km: Optional[np.ndarray] = None
    profile_heights_masl: Optional[np.ndarray] = None
    profile_zone_codes: Optional[np.ndarray] = None   # 1=coastal, 2=inland, 3=sea

    # --- Climatic data overrides (optional) ---
    Nd1km50: Optional[float] = None
    SdN_sup: Optional[float] = None
    SdN_sub: Optional[float] = None
    Nd65m1: Optional[float] = None
    surf_wv_density: Optional[float] = None
    h0_km: Optional[float] = None
    Pr6: Optional[float] = None
    MT: Optional[float] = None
    beta_rain: Optional[float] = None
    tropo_clim_zone: Optional[int] = None

    # --- Sporadic-E (optional) ---
    foEs50_mhz: Optional[float] = None
    foEs10_mhz: Optional[float] = None
    foEs01_mhz: Optional[float] = None
    foEs001_mhz: Optional[float] = None

    # --- Radioclimatic alignment with P.452 ---
    N0: Optional[float] = None
    delta_N: Optional[float] = None
    air_temperature: Optional[float] = None
    atmospheric_pressure: Optional[float] = None

    # --- SHARC modifiers (alignment with P.452) ---
    is_terrain: bool = False
    clutter_loss: bool = False
    clutter_type: str = "one_end"
    below_rooftop: bool = False

    def validate(self, ctx=None):
        """Raise ValueError for obviously wrong parameters."""
        if not (0.03 <= self.f_ghz <= 50.0):
            raise ValueError(f"f_ghz={self.f_ghz} outside 0.03–50 GHz range.")
            
        if isinstance(self.p, (float, int)):
            if not (0.0 <= self.p <= 100.0):
                raise ValueError(f"p={self.p} outside 0–100 % range.")
        elif str(self.p).upper() not in ('RANDOM', 'RANDOM_CENARIO'):
            raise ValueError(f"p={self.p} must be a number or 'RANDOM'.")
            
        if self.tx_height_agl <= 0 or self.rx_height_agl <= 0:
            raise ValueError("Antenna heights above ground must be > 0.")
        if self.polarization not in ("horizontal", "vertical"):
            raise ValueError("polarization must be 'horizontal' or 'vertical'.")
        if self.tropo_clim_zone is not None and self.tropo_clim_zone not in range(1, 7):
            raise ValueError("tropo_clim_zone must be 1–6.")
