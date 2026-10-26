# -*- coding: utf-8 -*-
"""
SHARC propagation adapter for ITU-R P.2001-6.

Integrates the P.2001-6 wide-range terrestrial propagation model into
the SHARC simulation framework, following the same interface as
``PropagationClearAir`` (P.452) and ``PropagationFreeSpace``.

Usage in propagation_factory.py
---------------------------------
Add the following block inside ``PropagationFactory.create_propagation``:

    elif channel_model == "P2001":
        return PropagationP2001(
            random_number_gen, param_system.param_p2001)

Then add ``param_p2001: ParametersP2001 = field(default_factory=ParametersP2001)``
to the relevant system parameter dataclass.
"""

import numpy as np
from multipledispatch import dispatch

from sharc.propagation.propagation import Propagation
from sharc.station_manager import StationManager
from sharc.parameters.parameters import Parameters

# Import our P.2001-6 engine
try:
    from sharc.propagation.propagation_p2001_core import (
        compute_P2001,
        P2001Result,
    )
    from sharc.parameters.parameters_p2001 import ParametersP2001
except ImportError:
    # Allow standalone use / testing when SHARC is not installed
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from propagation_p2001_core import compute_P2001, P2001Result
    from parameters_p2001 import ParametersP2001


class PropagationP2001(Propagation):
    """
    ITU-R P.2001-6 wide-range terrestrial propagation model.

    Frequency range: 30 MHz – 50 GHz
    Distance range : 3 km – 1 000 km (most reliable)
    Time %        : 0 – 100 %

    The model combines four sub-models:
      1. Diffraction + clear-air + precipitation
      2. Anomalous (ducting / layer-reflection)
      3. Troposcatter + precipitation
      4. Sporadic-E

    Parameters
    ----------
    random_number_gen : np.random.RandomState
        RNG (not used internally but required by SHARC base class).
    model_params : ParametersP2001
        All model configuration. See ``ParametersP2001`` for documentation.
    """

    def __init__(
        self,
        random_number_gen: np.random.RandomState,
        model_params: ParametersP2001,
    ):
        super().__init__(random_number_gen)
        self.params = model_params
        model_params.validate()

    # ------------------------------------------------------------------
    # SHARC interface – vectorised get_loss
    # ------------------------------------------------------------------

    @dispatch(Parameters, float, StationManager, StationManager,
              np.ndarray, np.ndarray)
    def get_loss(
        self,
        params: Parameters,
        frequency: float,
        station_a: StationManager,
        station_b: StationManager,
        station_a_gains=None,
        station_b_gains=None,
    ) -> np.ndarray:
        """
        Compute basic transmission loss matrix (dB) between all station pairs.

        Parameters
        ----------
        params : Parameters
            SHARC simulation parameters.
        frequency : float
            Carrier frequency in MHz (SHARC convention).
        station_a : StationManager
            First group of stations (rows in output matrix).
        station_b : StationManager
            Second group of stations (columns in output matrix).
        station_a_gains : np.ndarray, optional
            Antenna gains for station_a (dBi) – passed to troposcatter.
        station_b_gains : np.ndarray, optional
            Antenna gains for station_b (dBi).

        Returns
        -------
        np.ndarray
            Loss matrix of shape (station_a.num_stations, station_b.num_stations).
        """
        f_ghz = frequency * 1e-3  # MHz → GHz

        na = station_a.num_stations
        nb = station_b.num_stations
        loss = np.full((na, nb), fill_value=np.inf)

        p   = self.params.p
        pol = self.params.polarization

        # Unpack climatic parameters from model_params
        kw = dict(
            Nd1km50       = self.params.Nd1km50        if self.params.Nd1km50        is not None else -39.0,
            SdN_sup       = self.params.SdN_sup        if self.params.SdN_sup        is not None else 53.0,
            SdN_sub       = self.params.SdN_sub        if self.params.SdN_sub        is not None else 32.0,
            Nd65m1        = self.params.Nd65m1         if self.params.Nd65m1         is not None else -100.0,
            rho_sur       = self.params.surf_wv_density if self.params.surf_wv_density is not None else 7.5,
            h0_km         = self.params.h0_km          if self.params.h0_km          is not None else 3.0,
            Pr6           = self.params.Pr6            if self.params.Pr6            is not None else 5.0,
            MT            = self.params.MT             if self.params.MT             is not None else 1500.0,
            beta_rain     = self.params.beta_rain      if self.params.beta_rain      is not None else 0.45,
            tropo_clim_zone = self.params.tropo_clim_zone if self.params.tropo_clim_zone is not None else 2,
            foEs50        = self.params.foEs50_mhz     if self.params.foEs50_mhz     is not None else 4.0,
            foEs10        = self.params.foEs10_mhz     if self.params.foEs10_mhz     is not None else 7.0,
            foEs01        = self.params.foEs01_mhz     if self.params.foEs01_mhz     is not None else 10.0,
            foEs001       = self.params.foEs001_mhz    if self.params.foEs001_mhz    is not None else 13.0,
        )

        # Profile override from params (or None → auto flat profile)
        d_arr = self.params.profile_distances_km
        h_arr = self.params.profile_heights_masl
        z_arr = self.params.profile_zone_codes

        for i in range(na):
            if station_a_gains is not None:
                # Se houver apenas 1 ganho para múltiplas estações, usa o índice 0
                idx_a = i if len(station_a_gains) > i else 0
                val_a = station_a_gains[idx_a]
                Gt = float(val_a.flat[0]) if hasattr(val_a, 'flat') else float(val_a)
            else:
                Gt = 0.0

            for j in range(nb):
                if station_b_gains is not None:
                    # Correção para o IndexError: verifica o tamanho do array de ganhos
                    idx_b = j if len(station_b_gains) > j else 0
                    val_b = station_b_gains[idx_b]
                    Gr = float(val_b.flat[0]) if hasattr(val_b, 'flat') else float(val_b)
                else:
                    Gr = 0.0

                # Conversão de coordenadas cartesianas para geográficas
                tx_lat, tx_lon = self._xy_to_latlon(
                    station_a.x[i], station_a.y[i],
                    self.params.tx_lat, self.params.tx_lon,
                )
                rx_lat, rx_lon = self._xy_to_latlon(
                    station_b.x[j], station_b.y[j],
                    self.params.rx_lat, self.params.rx_lon,
                )

                # Altura das antenas
                tx_h = float(station_a.height[i]) if station_a.height[i] > 0 else self.params.tx_height_agl
                rx_h = float(station_b.height[j]) if station_b.height[j] > 0 else self.params.rx_height_agl

                try:
                    result = compute_P2001(
                        f_ghz=f_ghz, p_pct=p, polarization=pol,
                        tx_lat=tx_lat, tx_lon=tx_lon, tx_height_agl=tx_h,
                        rx_lat=rx_lat, rx_lon=rx_lon, rx_height_agl=rx_h,
                        Gt_dbi=Gt, Gr_dbi=Gr,
                        d_arr=d_arr, h_arr=h_arr, z_arr=z_arr,
                        **kw,
                    )
                    loss[i, j] = result.Lb_dB
                except Exception as exc:
                    loss[i, j] = self._fallback_fspl(
                        f_ghz,
                        station_a.x[i], station_a.y[i], station_a.z[i],
                        station_b.x[j], station_b.y[j], station_b.z[j]
                    )
        return loss

    # ------------------------------------------------------------------
    # Direct scalar API (useful for testing / campaigns)
    # ------------------------------------------------------------------

    def get_basic_transmission_loss(
        self,
        f_ghz: float,
        p_pct: float,
        tx_lat: float, tx_lon: float, tx_height_m: float,
        rx_lat: float, rx_lon: float, rx_height_m: float,
        Gt_dbi: float = 0.0, Gr_dbi: float = 0.0,
        d_arr: np.ndarray = None,
        h_arr: np.ndarray = None,
        z_arr: np.ndarray = None,
    ) -> P2001Result:
        """
        Convenience wrapper returning a full P2001Result for one link.

        Parameters
        ----------
        f_ghz : float
            Frequency in GHz.
        p_pct : float
            Time percentage (0-100).
        tx_lat, tx_lon : float
            Transmitter position (degrees).
        tx_height_m : float
            Transmitter antenna height above ground (m).
        rx_lat, rx_lon : float
            Receiver position (degrees).
        rx_height_m : float
            Receiver antenna height above ground (m).
        Gt_dbi, Gr_dbi : float
            Antenna gains in the path direction (dBi).
        d_arr, h_arr, z_arr : np.ndarray, optional
            Terrain profile arrays.

        Returns
        -------
        P2001Result
        """
        kw = dict(
            Nd1km50       = self.params.Nd1km50        if self.params.Nd1km50        is not None else -39.0,
            SdN_sup       = self.params.SdN_sup        if self.params.SdN_sup        is not None else 53.0,
            SdN_sub       = self.params.SdN_sub        if self.params.SdN_sub        is not None else 32.0,
            Nd65m1        = self.params.Nd65m1         if self.params.Nd65m1         is not None else -100.0,
            rho_sur       = self.params.surf_wv_density if self.params.surf_wv_density is not None else 7.5,
            h0_km         = self.params.h0_km          if self.params.h0_km          is not None else 3.0,
            Pr6           = self.params.Pr6            if self.params.Pr6            is not None else 5.0,
            MT            = self.params.MT             if self.params.MT             is not None else 1500.0,
            beta_rain     = self.params.beta_rain      if self.params.beta_rain      is not None else 0.45,
            tropo_clim_zone = self.params.tropo_clim_zone if self.params.tropo_clim_zone is not None else 2,
            foEs50        = self.params.foEs50_mhz     if self.params.foEs50_mhz     is not None else 4.0,
            foEs10        = self.params.foEs10_mhz     if self.params.foEs10_mhz     is not None else 7.0,
            foEs01        = self.params.foEs01_mhz     if self.params.foEs01_mhz     is not None else 10.0,
            foEs001       = self.params.foEs001_mhz    if self.params.foEs001_mhz    is not None else 13.0,
        )
        return compute_P2001(
            f_ghz=f_ghz,
            p_pct=p_pct,
            polarization=self.params.polarization,
            tx_lat=tx_lat, tx_lon=tx_lon, tx_height_agl=tx_height_m,
            rx_lat=rx_lat, rx_lon=rx_lon, rx_height_agl=rx_height_m,
            Gt_dbi=Gt_dbi, Gr_dbi=Gr_dbi,
            d_arr=d_arr, h_arr=h_arr, z_arr=z_arr,
            **kw,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _xy_to_latlon(x_m: float, y_m: float,
                       ref_lat: float, ref_lon: float) -> tuple:
        """
        Convert local Cartesian (m) to lat/lon (degrees) relative to a
        reference point.  Uses flat-Earth approximation – adequate for
        the SHARC cell sizes (< 100 km).
        """
        R_earth = 6371000.0  # m
        lat = ref_lat + np.degrees(y_m / R_earth)
        lon = ref_lon + np.degrees(x_m / (R_earth * np.cos(np.radians(ref_lat))))
        return float(lat), float(lon)

    @staticmethod
    def _fallback_fspl(f_ghz: float,
                        x_a: float, y_a: float, z_a: float,
                        x_b: float, y_b: float, z_b: float) -> float:
        """Free-space path loss fallback (dB)."""
        d_m = np.sqrt((x_a-x_b)**2 + (y_a-y_b)**2 + (z_a-z_b)**2)
        d_m = max(d_m, 1.0)
        d_km = d_m / 1000.0
        return 92.4 + 20.0 * np.log10(f_ghz) + 20.0 * np.log10(d_km)
