# -*- coding: utf-8 -*-
"""Implements ITU-R P.452 clear-air propagation model and related calculations."""
import numpy as np
import logging
import logging.config
from multipledispatch import dispatch

from sharc.propagation.propagation import Propagation
from sharc.station_manager import StationManager
from sharc.parameters.parameters import Parameters
from sharc.parameters.parameters_p452 import ParametersP452
from sharc.propagation.clear_air_452_aux import p676_ga
from sharc.propagation.clear_air_452_aux import inv_cum_norm
from sharc.support.enumerations import StationType
from sharc.propagation.propagation_clutter_loss import PropagationClutterLoss
from sharc.propagation.propagation_building_entry_loss import PropagationBuildingEntryLoss

class PropagationClearAir(Propagation):
    """Basic transmission loss due to free-space propagation and attenuation by atmospheric gases."""

    def __init__(
            self,
            random_number_gen: np.random.RandomState,
            model_params: ParametersP452):
        super().__init__(random_number_gen)
        self.clutter = PropagationClutterLoss(random_number_gen)
        self.building_entry = PropagationBuildingEntryLoss(self.random_number_gen)
        self.building_loss = 20
        self.model_params = model_params

    @staticmethod
    def closs_corr(f, d, h, zone, htg, hrg, ha_t, ha_r, dk_t, dk_r):
        """Apply clutter loss correction according to ITU-R P.452-16."""
        index1 = 1
        index2 = d.size
        htgc, hrgc = htg, hrg
        Aht, Ahr = 0, 0

        if len(ha_t) > 0 and np.max(ha_t) > htg:
            ha, dk = np.max(ha_t), np.max(dk_t)
            Ffc = 0.25 + 0.375 * (1 + np.tanh(7.5 * (f - 0.5)))
            Aht = 10.25 * Ffc * np.exp(-dk) * (1 - np.tanh(6 * (htg / ha - 0.625))) - 0.33
            kk = np.nonzero(d >= dk)
            index1 = int(kk[0][0]) + 1 if kk[0].size > 0 else d.size
            htgc = ha

        if len(ha_r) > 0 and np.max(ha_r) > hrg:
            ha, dk = np.max(ha_r), np.max(dk_r)
            Ffc = 0.25 + 0.375 * (1 + np.tanh(7.5 * (f - 0.5)))
            Ahr = 10.25 * Ffc * np.exp(-dk) * (1 - np.tanh(6 * (hrg / ha - 0.625))) - 0.33
            kk = np.nonzero(d <= d[-1] - dk)
            index2 = int(kk[0][-1]) + 1 if kk[0].size > 0 else 1
            hrgc = ha

        if (index2 - index1 < 3):
            raise ValueError("closs_corr: sum of clutter distances larger than path length.")

        return d[index1-1:index2] - d[index1-1], h[index1-1:index2], zone[index1-1:index2], htgc, hrgc, Aht, Ahr

    @staticmethod
    def longest_cont_dist(d, zone, zone_r):
        """Return the longest continuous distance in a given zone."""
        dm = 0
        aux = (zone == 1) + (zone == 2) if zone_r == 12 else (zone == zone_r)

        aux = np.append(0, np.append(aux, 0))
        aux = np.diff(aux)
        start = np.where(aux == 1)[0]
        stop = np.where(aux == -1)[0] - 1

        n = start.size
        for i in range(n):
            delta = 0
            if (d[stop[i]] < d[-1]):
                delta = delta + (d[stop[i] + 1] - d[stop[i]]) / 2.0
            if (d[start[i]] > 0):
                delta = delta + (d[stop[i]] - d[stop[i] - 1]) / 2.0
            dm = max(d[stop[i]] - d[start[i]] + delta, dm)

        return dm

    @staticmethod
    def beta0(phi, dtm, dlm):
        """Calculate the time percentage beta0 for anomalous propagation."""
        tau = 1 - np.exp(-(4.12 * 1e-4 * dlm ** 2.41))
        mu1 = (10 ** (-dtm / (16 - 6.6 * tau)) + 10 ** (-5 * (0.496 + 0.354 * tau))) ** 0.2
        mu1[np.nonzero(mu1 > 1)] = 1

        if abs(phi) <= 70:
            mu4 = 10 ** ((-0.935 + 0.0176 * abs(phi)) * np.log10(mu1))
            b0 = 10 ** (-0.015 * abs(phi) + 1.67) * mu1 * mu4
        else:
            mu4 = 10 ** (0.3 * np.log10(mu1))
            b0 = 4.17 * mu1 * mu4
        return b0

    @staticmethod
    def earth_rad_eff(DN):
        """Calculate the effective Earth radii (ae, ab)."""
        k50 = 157 / (157 - DN)
        return 6371 * k50, 6371 * 3

    @staticmethod
    def smooth_earth_heights(d, h, htg, hrg, ae, f):
        """Compute smooth Earth heights and related parameters."""
        n = d.size
        dtot = d[-1]
        hts, hrs = h[0] + htg, h[-1] + hrg

        v1 = np.sum(np.diff(d) * (h[1:] + h[:-1]))
        v2 = np.sum(np.diff(d) * (h[1:] * (2 * d[1:] + d[:-1]) + h[:-1] * (d[1:] + 2 * d[:-1])))
        hst, hsr = (2 * v1 * dtot - v2) / dtot ** 2, (v2 - v1 * dtot) / dtot ** 2

        HH = h - (hts * (dtot - d) + hrs * d) / dtot
        hobs = np.max(HH[1:-1])
        alpha_obt = np.max(HH[1:-1] / d[1:-1])
        alpha_obr = np.max(HH[1:-1] / (dtot - d[1:-1]))
        
        gt, gr = alpha_obt / (alpha_obt + alpha_obr), alpha_obr / (alpha_obt + alpha_obr)
        hstp, hsrp = (hst - hobs * gt, hsr - hobs * gr) if hobs > 0 else (hst, hsr)
        hstd, hsrd = min(hstp, h[0]), min(hsrp, h[-1])

        ii = np.arange(1, n - 1)
        theta = 1000 * np.arctan((h[ii] - hts) / (1000 * d[ii]) - d[ii] / (2 * ae))
        theta_t = np.max(theta)
        
        theta_td = 1000 * np.arctan((hrs - hts) / (1000 * dtot) - dtot / (2 * ae))
        theta_rd = 1000 * np.arctan((hts - hrs) / (1000 * dtot) - dtot / (2 * ae))
        pathtype = 2 if theta_t > theta_td else 1

        lt = int(np.argmax(theta)) + 1
        dlt = d[lt]

        theta_rx = 1000 * np.arctan((h[ii] - hrs) / (1000 * (dtot - d[ii])) - (dtot - d[ii]) / (2 * ae))
        theta_r = np.max(theta_rx)
        lr = int(np.argmax(theta_rx)) + 1
        dlr = dtot - d[lr]

        if pathtype == 1:
            theta_t, theta_r = theta_td, theta_rd
            lamb, Ce = 0.3 / f, 1 / ae
            nu = (h[ii] + 500 * Ce * d[ii] * (dtot - d[ii]) - (hts * (dtot - d[ii]) + hrs * d[ii]) / dtot) * np.sqrt(0.002 * dtot / (lamb * d[ii] * (dtot - d[ii])))
            lt = int(np.argmax(nu)) + 1
            dlt, dlr = d[lt], dtot - d[lt]
            kindex = np.nonzero(dlr <= dtot - d[ii])
            lr = int(kindex[0][-1]) + 1 if kindex[0].size > 0 else 1

        theta_tot = 1e3 * dtot / ae + theta_t + theta_r

        hst, hsr = min(hst, h[0]), min(hsr, h[-1])
        m = (hsr - hst) / dtot
        hte, hre = htg + h[0] - hst, hrg + h[-1] - hsr

        ii_range = np.arange(lt, lr + 1)
        hm = np.max(h[ii_range] - (hst + m * d[ii_range])) if ii_range.size > 0 else 0

        return hst, hsr, hstd, hsrd, hte, hre, hm, dlt, dlr, theta_t, theta_r, theta_tot, pathtype

    @staticmethod
    def path_fraction(d, zone, zone_r):
        """Calculate the fraction of the path in a given zone as per ITU-R P.452."""
        dm = 0
        aux = np.nonzero(zone == zone_r)
        start, stop = np.atleast_1d(aux[0]), np.atleast_1d(aux[-1])

        for i in range(start.size):
            delta = 0
            if (d[stop[i]] < d[-1]):
                delta += (d[stop[i] + 1] - d[stop[i]]) / 2.0
            if (d[start[i]] > 0):
                delta += (d[stop[i]] - d[stop[i] - 1]) / 2.0
            dm += d[stop[i]] - d[start[i]] + delta

        return dm / (d[-1] - d[0])

    @staticmethod
    def pl_los(d, f, p, b0, w, T, press, dlt, dlr):
        """Calculate the basic transmission loss not exceeded for time percentage p due to LoS propagation."""
        rho = 7.5 + 2.5 * w
        [g_0, g_w] = p676_ga(f, press, rho, T, True)
        Ag = (g_0 + g_w) * d

        Lbfsg = 92.5 + 20.0 * np.log10(f) + 20.0 * np.log10(d) + Ag
        Esp = 2.6 * (1 - np.exp(-0.1 * (dlt + dlr))) * np.log10(p / 50)
        Esb = 2.6 * (1 - np.exp(-0.1 * (dlt + dlr))) * np.log10(b0 / 50)

        return Lbfsg, Lbfsg + Esp, Lbfsg + Esb

    @staticmethod
    def tl_tropo(dtot, theta, f, p, T, press, N0, Gt, Gr):
        """Calculate the basic transmission loss due to troposcatter."""
        Lf = 25 * np.log10(f) - 2.5 * (np.log10(f / 2)) ** 2
        Lc = 0.051 * np.exp(0.055 * (Gt + Gr))
        [g_0, g_w] = p676_ga(f, press, 3, T, True)
        Ag = (g_0 + g_w) * dtot

        return 190 + Lf + 20 * np.log10(dtot) + 0.573 * theta - 0.15 * N0 + Lc + Ag - 10.1 * (-np.log10(p / 50)) ** (0.7)

    @staticmethod
    def tl_anomalous(dtot, dlt, dlr, dct, dcr, dlm, hts, hrs, hte, hre, hm, theta_t, theta_r, f, p, T, press, omega, ae, b0):
        """Calculate the basic transmission loss due to anomalous propagation."""
        Alf = 45.375 - 137.0 * f + 92.5 * f * f if f < 0.5 else 0

        theta_t1, theta_r1 = theta_t - 0.1 * dlt, theta_r - 0.1 * dlr
        Ast = 20 * np.log10(1 + 0.361 * theta_t1 * np.sqrt(f * dlt)) + 0.264 * theta_t1 * f ** (1 / 3) if theta_t1 > 0 else 0
        Asr = 20 * np.log10(1 + 0.361 * theta_r1 * np.sqrt(f * dlr)) + 0.264 * theta_r1 * f ** (1 / 3) if theta_r1 > 0 else 0

        Act = -3 * np.exp(-0.25 * dct * dct) * (1 + np.tanh(0.07 * (50 - hts))) if dct <= 5 and dct <= dlt and omega >= 0.75 else 0
        Acr = -3 * np.exp(-0.25 * dcr * dcr) * (1 + np.tanh(0.07 * (50 - hrs))) if dcr <= 5 and dcr <= dlr and omega >= 0.75 else 0

        gamma_d = 5e-5 * ae * f ** (1 / 3)
        theta_t1 = min(theta_t, 0.1 * dlt)
        theta_r1 = min(theta_r, 0.1 * dlr)

        theta1 = 1e3 * dtot / ae + theta_t1 + theta_r1
        dI = min(dtot - dlt - dlr, 40)
        mu3 = np.exp(-4.6e-5 * (hm - 10) * (43 + 6 * dI)) if hm > 10 else 1

        tau = 1 - np.exp(-(4.12e-4 * dlm ** 2.41))
        alpha = max(-3.4, -0.6 - 3.5 * 1e-9 * dtot ** (3.1) * tau)
        mu2 = min(1, (500 / ae * dtot ** 2 / (np.sqrt(hte) + np.sqrt(hre)) ** 2) ** alpha)

        beta = b0 * mu2 * mu3
        Gamma = 1.076 / (2.0058 - np.log10(beta)) ** 1.012 * np.exp(-(9.51 - 4.8 * np.log10(beta) + 0.198 * (np.log10(beta)) ** 2) * 1e-6 * dtot ** 1.13)

        Ap = -12 + (1.2 + 3.7e-3 * dtot) * np.log10(p / beta) + 12 * (p / beta) ** Gamma
        Adp = gamma_d * theta1 + Ap

        [g_0, g_w] = p676_ga(f, press, 7.5 + 2.5 * omega, T, True)
        Ag = (g_0 + g_w) * dtot

        Af = 102.45 + 20 * np.log10(f) + 20 * np.log10(dlt + dlr) + Alf + Ast + Asr + Act + Acr
        return Af + Adp + Ag

    @staticmethod
    def dl_bull(d, h, hts, hrs, ap, f):
        """Calculate Bullington diffraction loss for a given path profile."""
        Ce, lamb, dtot = 1 / ap, 0.3 / f, d[-1] - d[0]
        di, hi = d[1:-1], h[1:-1]

        Stim = np.max((hi + 500 * Ce * di * (dtot - di) - hts) / di)
        Str = (hrs - hts) / dtot

        if Stim < Str: 
            numax = np.max((hi + 500 * Ce * di * (dtot - di) - (hts * (dtot - di) + hrs * di) / dtot) * np.sqrt(0.002 * dtot / (lamb * di * (dtot - di))))
            Luc = 6.9 + 20 * np.log10(np.sqrt((numax - 0.1) ** 2 + 1) + numax - 0.1) if numax > -0.78 else 0
        else:
            Srim = np.max((hi + 500 * Ce * di * (dtot - di) - hrs) / (dtot - di))
            dbp = (hrs - hts + Srim * dtot) / (Stim + Srim)
            nub = (hts + Stim * dbp - (hts * (dtot - dbp) + hrs * dbp) / dtot) * np.sqrt(0.002 * dtot / (lamb * dbp * (dtot - dbp)))
            Luc = 6.9 + 20 * np.log10(np.sqrt((nub - 0.1) ** 2 + 1) + nub - 0.1) if nub > -0.78 else 0

        return Luc + (1 - np.exp(-Luc / 6.0)) * (10 + 0.02 * dtot)

    @staticmethod
    def dl_se_ft_inner(epsr, sigma, d, hte, hre, adft, f):
        """Calculate the first-term part of the spherical-Earth diffraction loss."""
        # CORREÇÃO BLINDADA: Garante que sejam floats escalares puros para evitar o TypeError de 0-dimensionais no NumPy
        f = float(np.asarray(f).item())
        adft = float(np.asarray(adft).item())
        d = float(np.asarray(d).item())
        hte = float(np.asarray(hte).item())
        hre = float(np.asarray(hre).item())

        K = np.empty(2)
        K[0] = 0.036 * (adft * f) ** (-1 / 3) * ((epsr - 1) ** 2 + (18 * sigma / f) ** 2) ** (-1 / 4)
        K[1] = K[0] * (epsr ** 2 + (18 * sigma / f) ** 2) ** (1 / 2)

        beta_dft = (1 + 1.6 * K ** 2 + 0.67 * K**4) / (1 + 4.5 * K ** 2 + 1.53 * K ** 4)
        X = 21.88 * beta_dft * (f / adft ** 2) ** (1 / 3) * d
        Yt = 0.9575 * beta_dft * (f ** 2 / adft) ** (1 / 3) * hte
        Yr = 0.9575 * beta_dft * (f ** 2 / adft) ** (1 / 3) * hre

        Fx = np.where(X >= 1.6, 11 + 10 * np.log10(X) - 17.6 * X, -20 * np.log10(X) - 5.6488 * (X) ** 1.425)

        Bt, Br = beta_dft * Yt, beta_dft * Yr
        GYt = np.where(Bt > 2, 17.6 * (Bt - 1.1) ** 0.5 - 5 * np.log10(Bt - 1.1) - 8, 20 * np.log10(Bt + 0.1 * Bt ** 3))
        GYr = np.where(Br > 2, 17.6 * (Br - 1.1) ** 0.5 - 5 * np.log10(Br - 1.1) - 8, 20 * np.log10(Br + 0.1 * Br ** 3))

        GYt = np.maximum(GYt, 2 + 20 * np.log10(K))
        GYr = np.maximum(GYr, 2 + 20 * np.log10(K))

        return -Fx - GYt - GYr

    @staticmethod
    def dl_se_ft(d, hte, hre, adft, f, omega):
        """Calculate the first-term spherical-Earth diffraction loss over land and sea."""
        Ldft_land = PropagationClearAir.dl_se_ft_inner(22, 0.003, d, hte, hre, adft, f)
        Ldft_sea = PropagationClearAir.dl_se_ft_inner(80, 5, d, hte, hre, adft, f)
        return omega * Ldft_sea + (1 - omega) * Ldft_land

    @staticmethod
    def dl_se(d, hte, hre, ap, f, omega):
        """Calculate the spherical-Earth diffraction loss for a given path."""
        lamb = 0.3 / f
        dlos = np.sqrt(2 * ap) * (np.sqrt(0.001 * hte) + np.sqrt(0.001 * hre))

        if d >= dlos:
            Ldsph = PropagationClearAir.dl_se_ft(d, hte, hre, ap, f, omega)
        else:
            c = (hte - hre) / (hte + hre)
            m = 250 * d * d / (ap * (hte + hre))
            b = 2 * np.sqrt((m + 1) / (3 * m)) * np.cos(np.pi / 3 + 1 / 3 * np.arccos(3 * c / 2 * np.sqrt(3 * m / (m + 1) ** 3)))

            dse1 = d / 2 * (1 + b)
            dse2 = d - dse1

            hse = ((hte - 500 * dse1 * dse1 / ap) * dse2 + (hre - 500 * dse2 * dse2 / ap) * dse1) / d
            hreq = 17.456 * np.sqrt(dse1 * dse2 * lamb / d)

            if hse > hreq:
                Ldsph = np.array([0, 0])
            else:
                aem = 500 * (d / (np.sqrt(hte) + np.sqrt(hre)))**2
                Ldft = PropagationClearAir.dl_se_ft(d, hte, hre, aem, f, omega)
                Ldsph = np.array([0, 0]) if (Ldft < 0).any() else (1 - hse / hreq) * Ldft

        return Ldsph

    @staticmethod
    def dl_delta_bull(d, h, hts, hrs, hstd, hsrd, ap, f, omega):
        """Calculate the general diffraction loss using the Bullington and spherical-Earth methods."""
        Lbulla = PropagationClearAir.dl_bull(d, h, hts, hrs, ap, f)
        hts1, hrs1, h1 = hts - hstd, hrs - hsrd, np.zeros(h.size)
        Lbulls = PropagationClearAir.dl_bull(d, h1, hts1, hrs1, ap, f)

        Ldsph = PropagationClearAir.dl_se(d[-1] - d[0], hts1, hrs1, ap, f, omega)
        
        Ld = np.empty(2)
        Ld[0] = Lbulla + max(Ldsph[0] - Lbulls, 0)
        Ld[1] = Lbulla + max(Ldsph[1] - Lbulls, 0)
        return Ld

    @staticmethod
    def dl_p(d, h, hts, hrs, hstd, hsrd, f, omega, p, b0, DN):
        """Calculate the diffraction loss not exceeded for a given time percentage using ITU-R P.452."""
        ae, ab = PropagationClearAir.earth_rad_eff(DN)
        Ld50 = PropagationClearAir.dl_delta_bull(d, h, hts, hrs, hstd, hsrd, ae, f, omega)

        if p == 50:
            return Ld50, Ld50
        
        Ldb = PropagationClearAir.dl_delta_bull(d, h, hts, hrs, hstd, hsrd, ab, f, omega)
        Fi = inv_cum_norm(p / 100) / inv_cum_norm(b0 / 100) if p > b0 else 1
        return Ld50 + Fi * (Ldb - Ld50), Ld50

    @dispatch(Parameters, float, StationManager, StationManager, np.ndarray, np.ndarray)
    def get_loss(self, params: Parameters, frequency: float, station_a: StationManager, station_b: StationManager, station_a_gains=None, station_b_gains=None) -> np.array:
        """Wrapper function for the get_loss method to fit the Propagation ABC class interface."""
        distance = station_a.get_3d_distance_to(station_b) * (1e-3)
        frequency_array = frequency * np.ones(distance.shape) * (1e-3)
        indoor_stations = np.tile(station_b.indoor, (station_a.num_stations, 1))
        elevation = station_b.get_elevation(station_a)

        tx_gain, rx_gain = (station_a_gains, station_b_gains) if params.imt.interfered_with else (station_b_gains, station_a_gains)
        return self.get_loss(distance, frequency_array, indoor_stations, elevation, tx_gain, rx_gain)

    @dispatch(np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    def get_loss(self, distance: np.ndarray, frequency: np.ndarray, indoor_stations: np.ndarray, elevation: np.ndarray, tx_gain: np.ndarray, rx_gain: np.ndarray) -> np.array:
        """Calculates the loss according to P.452"""
        frequency_unique = np.unique(frequency)
        if len(frequency_unique) > 1:
            raise ValueError("different frequencies not supported in P.452")

        # CORREÇÃO PRINCIPAL: Assegurando que todos os parâmetros atmosféricos e de frequência sejam escalares puros (floats).
        # Evita a propagação de arrays 1D pelo código quebrando o NumPy na formulação de Terra Esférica e Bullington.
        f_scalar = float(frequency_unique.item())
        Ph = float(np.asarray(self.model_params.atmospheric_pressure).item())
        T = float(np.asarray(self.model_params.air_temperature).item())
        Dct = float(np.asarray(self.model_params.Dct).item())
        Dcr = float(np.asarray(self.model_params.Dcr).item())
        Hte = float(np.asarray(self.model_params.Hte).item())
        Hre = float(np.asarray(self.model_params.Hre).item())
        N0 = float(np.asarray(self.model_params.N0).item())
        deltaN = float(np.asarray(self.model_params.delta_N).item())

        if self.model_params.percentage_p == 'RANDOM':
            p = 50 * self.random_number_gen.rand(distance.size)
        elif self.model_params.percentage_p == 'RANDOM_CENARIO':
            p = 50 * self.random_number_gen.rand(1) * np.ones(distance.size)
        else:
            p = float(self.model_params.percentage_p) * np.ones(distance.size)

        tx_lat, rx_lat = self.model_params.tx_lat, self.model_params.rx_lat
        tx_gain, rx_gain = np.ravel(tx_gain), np.ravel(rx_gain)

        terrain_d = getattr(self.model_params, "terrain_d", None)
        terrain_h = getattr(self.model_params, "terrain_h", None)
        use_profile = (terrain_d is not None and terrain_h is not None and len(terrain_d) > 1 and len(terrain_h) == len(terrain_d))

        if use_profile:
            profile_d, profile_h = terrain_d.copy(), terrain_h.copy()
            num_links, profile_length = distance.shape[1], profile_d.size
            d, h = np.zeros((num_links, profile_length)), np.zeros((num_links, profile_length))

            for ii in range(num_links):
                scale = distance[0, ii] / profile_d[-1]
                d[ii,:], h[ii,:] = profile_d * scale, profile_h
            num_dists = distance.shape[1]

        elif self.model_params.is_terrain:
            rng = np.random.default_rng(42)
            nu_h, mu_h, sigma_h, mu_d, sigma_d = 1.525, 0, 24.52, 1.06, 0.814
            num_dists, total_dist = distance.size, float(np.min(distance))

            while True:
                d_vals, h_vals = [0.0], [0.0]
                while d_vals[-1] < total_dist:
                    step = rng.lognormal(mean=mu_d, sigma=sigma_d)
                    next_d = d_vals[-1] + step

                    if next_d >= total_dist:
                        d_vals.append(total_dist)
                        h_vals.append(0.0)
                        break

                    d_vals.append(next_d)
                    h_vals.append(mu_h + sigma_h * rng.standard_t(df=nu_h))

                if len(d_vals) > 3:
                    break
                
            profile_d, profile_h = np.array(d_vals), np.array(h_vals)
            profile_length = len(profile_d)
            d, h = np.zeros((num_dists, profile_length)), np.zeros((num_dists, profile_length))

            for ii in range(num_dists):
                d[ii, :], h[ii, :] = profile_d, profile_h
                d[ii, -1], h[ii, -1] = distance[0, ii], 0.0
        else:
            profile_length, num_dists = 100, distance.size
            d = np.empty([num_dists, profile_length])
            for ii in range(num_dists):
                d[ii, :] = np.linspace(0, distance[0][ii], profile_length)
            h = np.zeros(d.shape)

        ha_t, ha_r, dk_t, dk_r = [], [], [], []
        phi_path = (tx_lat + rx_lat) / 2
        dtm, dlm, zone = np.empty(num_dists), np.empty(num_dists), np.ones(profile_length) * 2

        for index in range(num_dists):
            dtm[index] = self.longest_cont_dist(d[index, :], zone, 12)
            dlm[index] = self.longest_cont_dist(d[index, :], zone, 2)

        b0 = self.beta0(phi_path, dtm, dlm)
        ae, ab = self.earth_rad_eff(deltaN)
        omega = self.path_fraction(d.transpose(), zone, 3)

        Lb = np.empty([1, num_dists])
        Ce, THETA, KSI, dsw, kappa = 1 / ae, 0.3, 0.8, 20, 0.5

        for ii in range(num_dists):
            # Usando f_scalar em todas as chamadas numéricas
            [dc, hc, zonec, htg, hrg, Aht, Ahr] = self.closs_corr(f_scalar, d[ii, :], h[ii, :], zone, Hte, Hre, ha_t, ha_r, dk_t, dk_r)
            d[ii, :], h[ii, :] = dc, hc

            [hst, hsr, hstd, hsrd, hte, hre, hm, dlt, dlr, theta_t, theta_r, theta_tot, pathtype] = self.smooth_earth_heights(d[ii, :], h[ii, :], htg, hrg, ae, f_scalar)

            dtot = d[ii, -1] - d[ii, 0]
            hts, hrs = hc[0] + htg, hc[-1] + hrg

            if len(d[ii]) < 4:
                raise ValueError("tl_p452: path profile requires at least 4 points.")

            di, hi = d[ii, 1:-1], h[ii, 1:-1]
            Stim, Str = max((hi + 500 * Ce * di * (dtot - di) - hts) / di), (hrs - hts) / dtot

            Fj = 1.0 - 0.5 * (1.0 + np.tanh(3.0 * KSI * (Stim - Str) / THETA))
            Fk = 1.0 - 0.5 * (1.0 + np.tanh(3.0 * kappa * (dtot - dsw) / dsw))

            [Lbfsg, Lb0p, Lb0b] = self.pl_los(dtot, f_scalar, p[ii], b0[ii], omega[ii], T, Ph, dlt, dlr)
            [Ldp, Ld50] = self.dl_p(d[ii], h[ii], hts, hrs, hstd, hsrd, f_scalar, omega[ii], p[ii], b0[ii], deltaN)

            Lbd50, Lbd = Lbfsg + Ld50, Lb0p + Ldp
            Lminb0p = Lb0p + (1 - omega[ii]) * Ldp

            if p[ii] >= b0[ii]:
                Fi = inv_cum_norm(p[ii] / 100) / inv_cum_norm(b0[ii] / 100)
                Lminb0p = Lbd50 + (Lb0b + (1 - omega[ii]) * Ldp - Lbd50) * Fi

            Lba = self.tl_anomalous(dtot, dlt, dlr, Dct, Dcr, dlm[ii], hts, hrs, hte, hre, hm, theta_t, theta_r, f_scalar, p[ii], T, Ph, omega[ii], ae, b0[ii])

            Lminbap = 2.5 * np.log(np.exp(Lba / 2.5) + np.exp(Lb0p / 2.5))
            Lbda = Lminbap + (Lbd - Lminbap) * Fk if (Lbd >= Lminbap).any() else Lbd
            Lbam = Lbda + (Lminb0p - Lbda) * Fj

            Lbs = self.tl_tropo(dtot, theta_tot, f_scalar, p[ii], T, Ph, N0, tx_gain[ii], rx_gain[ii])
            Lb_pol = -5 * np.log10(10 ** (-0.2 * Lbs) + 10 ** (-0.2 * Lbam)) + Aht + Ahr

            if self.model_params.polarization.lower() == "horizontal":
                Lb[0, ii] = Lb_pol[0]
            elif self.model_params.polarization.lower() == "vertical":
                Lb[0, ii] = Lb_pol[1]
            else:
                raise ValueError("invalid polarization")

        if self.model_params.clutter_loss:
            # Aqui mantemos a array original de frequency porque o modelo de clutter pode requerer o mesmo formato de 'distance'
            clutter_loss = self.clutter.get_loss(
                frequency=frequency * 1000,
                distance=distance * 1000,
                clutter_scenario="terrestrial",
                clutter_type=self.model_params.clutter_type,
                below_rooftop=self.model_params.below_rooftop
            )
        else:
            clutter_loss = np.zeros(distance.shape)

        b_loss = np.transpose(self.building_entry.get_loss(frequency, elevation))
        return Lb + clutter_loss + (b_loss * indoor_stations)