"""
Plasmonic wakefields in single- and double-walled carbon nanotubes (SWCNT,
DWCNT), linearized hydrodynamic model, closed-form gamma->0+ limit.

Implements Sections 3-4 of:
Martin-Luna & Resta-Lopez, "Excitation of Plasmonic Wakefields in
Multi-Walled Carbon Nanotubes: A Hydrodynamic Approach",
DOI: 10.5772/intechopen.114270

Unlike the graphene case, this chapter gives an ANALYTIC formula for the
gamma->0+ limit (Eqs. 17-18 for SWCNT, Eqs. 27-32 for DWCNT): the wakefield
is a discrete sum over the resonant wavenumbers k_m (roots of the plasma
resonance condition k*v = omega_m(k)), instead of a numerical integral over
a narrow k-space resonance. This is inherently well-behaved numerically
(no grid-convergence issue like the graphene case) -- root-finding for a
handful of k_m per angular mode m, done to whatever precision you like.

All internal computation in atomic units; helper functions convert to/from
nm, GV/m, units of c.
"""

import numpy as np
from scipy.special import iv, kv, ivp, kvp
from scipy.optimize import brentq

# ---------------------------------------------------------------- constants
C_AU = 137.035999084
BOHR_TO_NM = 0.0529177210903
NM_TO_AU = 1.0 / BOHR_TO_NM
FIELD_AU_TO_GVPM = 514.2207
N_G_AU = 4 * 0.107   # graphite electron-gas surface density (a.u.), Sec. 2


def _g(r1, r2, m, k):
    """g(r1,r2;m,k) = 4*pi*I_m(|k|*r_min)*K_m(|k|*r_max)   (Eq. 5)."""
    rmin, rmax = min(r1, r2), max(r1, r2)
    ak = abs(k)
    return 4 * np.pi * iv(m, ak * rmin) * kv(m, ak * rmax)


def _dg_dr(r_fixed_a, r, m, k):
    """d/dr of g(r_fixed_a, r; m, k), i.e. derivative w.r.t. the SECOND
    argument, needed for the transverse wakefield Wr. r_fixed_a is the tube
    radius a (the density lives there); r is the observation radius."""
    ak = abs(k)
    if r < r_fixed_a:
        # r is the "min" argument -> I_m(k r) is the r-dependent factor
        return 4 * np.pi * ak * ivp(m, ak * r) * kv(m, ak * r_fixed_a)
    else:
        # r is the "max" argument -> K_m(k r) is the r-dependent factor
        return 4 * np.pi * ak * iv(m, ak * r_fixed_a) * kvp(m, ak * r)


# ============================================================ SWCNT (Sec.3)
class SWCNT:
    """Single-walled carbon nanotube, on-axis driving charge by default.

    Parameters (physical units)
    ----------------------------
    a_nm : tube radius, nm
    v_over_c : driver velocity / c
    n0_over_ng : surface density in units of n_g = 4*0.107 a.u.
    Q : driver charge (protron = +1)
    r0_nm, phi0 : driver radial position (nm) and angle (rad); r0=0 excites
        only the m=0 mode (paper: "the fundamental mode is the only mode
        that is excited" on-axis).
    """

    def __init__(self, a_nm=0.36, v_over_c=0.05, n0_over_ng=1.0, Q=1.0,
                 r0_nm=0.0, phi0=0.0):
        self.a = a_nm * NM_TO_AU
        self.v = v_over_c * C_AU
        self.n0 = n0_over_ng * N_G_AU
        self.Q = Q
        self.r0 = r0_nm * NM_TO_AU
        self.phi0 = phi0

        self.alpha = self.n0  # NOTE: alpha = vF^2/2 = (2*pi*n0)/2 = pi*n0 -- see _omega2
        self.beta = 0.25
        self.Omega_p2 = 4 * np.pi * self.n0 / self.a   # Omega_p^2 (Eq. 16)

    def _omega2(self, m, k):
        """omega_m^2(k), Eq. (16)."""
        q2 = k**2 + m**2 / self.a**2
        alpha = np.pi * self.n0          # alpha = vF^2/2, vF^2 = 2*pi*n0
        return (alpha * q2 + self.beta * q2**2 +
                self.Omega_p2 * self.a**2 * q2 * kv(m, abs(k) * self.a) * iv(m, abs(k) * self.a))

    def _Z(self, m, k):
        return (k * self.v)**2 - self._omega2(m, k)

    def resonances(self, m, k_max=3.0, n_scan=4000):
        """Find all positive roots k_m of Z_m(k)=0 in (0, k_max] by
        scan + bisection. Returns a (possibly empty) list of floats."""
        ks = np.linspace(1e-6, k_max, n_scan)
        Z = np.array([self._Z(m, k) for k in ks])
        roots = []
        for i in range(len(ks) - 1):
            if Z[i] == 0:
                roots.append(ks[i])
            elif Z[i] * Z[i + 1] < 0:
                roots.append(brentq(lambda k: self._Z(m, k), ks[i], ks[i + 1]))
        return roots

    def _dZdk(self, m, k, h=1e-6):
        return (self._Z(m, k + h) - self._Z(m, k - h)) / (2 * h)

    # -- wakefields (gamma->0+ analytic limit), Eqs. (17)-(18) ------------
    def wakefields(self, r_nm, zeta_nm, phi=0.0, m_max=6, k_max=3.0):
        """
        Returns (Wz, Wr) in GV/m at radial position r_nm, azimuth phi (rad),
        as arrays over the zeta_nm array (nm, comoving coordinate).
        Sums angular modes m = -m_max..m_max (m and -m give identical
        physics for a point charge; the sum is real by construction).
        """
        r = r_nm * NM_TO_AU
        zeta = np.asarray(zeta_nm) * NM_TO_AU
        Wz = np.zeros_like(zeta)
        Wr = np.zeros_like(zeta)

        for m in range(-m_max, m_max + 1):
            mm = abs(m)
            for km in self.resonances(mm, k_max=k_max):
                dZdk = self._dZdk(mm, km)
                if dZdk == 0:
                    continue
                phase = np.cos(m * (phi - self.phi0))
                q2 = km**2 + mm**2 / self.a**2

                pref_common = self.Omega_p2 * self.a**2 * q2 / abs(dZdk)
                g_a_r0 = _g(self.a, self.r0, mm, km)

                # Wz,Im : Eq. (17)
                g_a_r = _g(self.a, r, mm, km)
                amp_z = -self.Q / (8 * np.pi**2) * km * pref_common * g_a_r0 * g_a_r
                Wz += phase * amp_z * np.cos(km * zeta)

                # Wr,Im : Eq. (18)
                dg_a_r = _dg_dr(self.a, r, mm, km)
                amp_r = -self.Q / (8 * np.pi**2) * pref_common * g_a_r0 * dg_a_r
                Wr += phase * amp_r * np.sin(km * zeta)

        return Wz * FIELD_AU_TO_GVPM, Wr * FIELD_AU_TO_GVPM

    # -- 2D map in a (zeta, x) cut plane, x = signed radius (phi=0 for x>0,
    #    phi=pi for x<0) -- the CNT analogue of the graphene (zeta,z) map.
    def wakefields_zeta_x(self, zeta_nm, x_nm, m_max=6, k_max=3.0):
        """
        Returns Wz_map, Wx_map (GV/m), shape (len(x_nm), len(zeta_nm)).
        Wx_map is the LONGITUDINAL-plane-projected radial field: positive x
        means radially outward on the x>0 side, and the sign is flipped on
        the x<0 side so that a genuinely outward-pointing field always shows
        the same color convention on both sides of the tube axis (like the
        two graphene layers appearing as mirrored horizontal lines).
        """
        zeta = np.asarray(zeta_nm) * NM_TO_AU
        x = np.asarray(x_nm) * NM_TO_AU
        Wz_map = np.zeros((len(x), len(zeta)))
        Wr_map = np.zeros((len(x), len(zeta)))

        r_abs = np.abs(x)
        sign_conv = np.where(x >= 0, 1.0, -1.0)
        phi_of_x = np.where(x >= 0, 0.0, np.pi)

        for m in range(-m_max, m_max + 1):
            mm = abs(m)
            for km in self.resonances(mm, k_max=k_max):
                dZdk = self._dZdk(mm, km)
                if dZdk == 0:
                    continue
                q2 = km**2 + mm**2 / self.a**2
                pref_common = self.Omega_p2 * self.a**2 * q2 / abs(dZdk)
                g_a_r0 = _g(self.a, self.r0, mm, km)
                cos_kz = np.cos(km * zeta)
                sin_kz = np.sin(km * zeta)

                for ix in range(len(x)):
                    r = r_abs[ix]
                    phase = np.cos(m * (phi_of_x[ix] - self.phi0))

                    g_a_r = _g(self.a, r, mm, km)
                    amp_z = -self.Q / (8 * np.pi**2) * km * pref_common * g_a_r0 * g_a_r
                    Wz_map[ix, :] += phase * amp_z * cos_kz

                    dg_a_r = _dg_dr(self.a, r, mm, km)
                    amp_r = -self.Q / (8 * np.pi**2) * pref_common * g_a_r0 * dg_a_r
                    Wr_map[ix, :] += sign_conv[ix] * phase * amp_r * sin_kz

        return Wz_map * FIELD_AU_TO_GVPM, Wr_map * FIELD_AU_TO_GVPM
class DWCNT:
    """Double-walled carbon nanotube, Eqs. (20)-(32)."""

    def __init__(self, a1_nm=0.36, a2_nm=0.7, v_over_c=0.05,
                 n0_over_ng=1.0, Q=1.0, r0_nm=0.0, phi0=0.0):
        self.a = [a1_nm * NM_TO_AU, a2_nm * NM_TO_AU]
        self.v = v_over_c * C_AU
        self.n0 = n0_over_ng * N_G_AU
        self.Q = Q
        self.r0 = r0_nm * NM_TO_AU
        self.phi0 = phi0
        self.beta = 0.25
        self.alpha = np.pi * self.n0

    def _omega_j2(self, j, m, k):
        """omega_j^2(m,k), Eq. (21), j=0,1 (wall index)."""
        a = self.a[j]
        q2 = k**2 + m**2 / a**2
        return (self.alpha * q2 + self.beta * q2**2 +
                self.n0 * a * q2 * _g(a, a, m, k))

    def _Delta2(self, m, k):
        """Delta^2, Eq. (22)."""
        a1, a2 = self.a
        return (self.n0**2 * a1 * a2 * (k**2 + m**2 / a1**2) * (k**2 + m**2 / a2**2)
                * _g(a1, a2, m, k)**2)

    def _omega_pm2(self, sign, m, k):
        w1_2 = self._omega_j2(0, m, k)
        w2_2 = self._omega_j2(1, m, k)
        disc = np.sqrt(((w1_2 - w2_2) / 2)**2 + self._Delta2(m, k))
        return (w1_2 + w2_2) / 2 + sign * disc

    def _Z_pm(self, sign, m, k):
        return (k * self.v)**2 - self._omega_pm2(sign, m, k)

    def _dZdk(self, sign, m, k, h=1e-6):
        return (self._Z_pm(sign, m, k + h) - self._Z_pm(sign, m, k - h)) / (2 * h)

    def resonances(self, sign, m, k_max=3.0, n_scan=4000):
        ks = np.linspace(1e-6, k_max, n_scan)
        Z = np.array([self._Z_pm(sign, m, k) for k in ks])
        roots = []
        for i in range(len(ks) - 1):
            if Z[i] * Z[i + 1] < 0:
                roots.append(brentq(lambda k: self._Z_pm(sign, m, k), ks[i], ks[i + 1]))
        return roots

    def _N_j(self, j, m, k, omega):
        """N_1, N_2 (Eq. 24), needed to weight each wall's contribution.

        NOTE: self._omega_j2(j,m,k) already includes the G_jj self-term
        (Eq. 21 bakes it in), so S1 = omega^2 - omega_j2(0) already equals
        (S1_raw - G11) in the paper's notation -- do NOT subtract G_jj again
        here, or you double-count it (this was a real bug, caught by
        cross-checking the magnitude against Fig. 8a of the paper).
        """
        def Bp(jj):
            aj = self.a[jj]
            q2 = k**2 + m**2 / aj**2
            return self.n0 * aj * q2 * 2 * np.pi * self.Q * _g(aj, self.r0, m, k) * \
                np.exp(-1j * m * self.phi0)

        G12 = self.n0 * self.a[0] * (k**2 + m**2 / self.a[0]**2) * _g(self.a[0], self.a[1], m, k)
        G21 = self.n0 * self.a[1] * (k**2 + m**2 / self.a[1]**2) * _g(self.a[1], self.a[0], m, k)

        # S1, S2 here already equal (S_raw - G_jj) -- see note above.
        S1 = omega**2 - self._omega_j2(0, m, k)
        S2 = omega**2 - self._omega_j2(1, m, k)

        B1p, B2p = Bp(0), Bp(1)
        if j == 0:
            return S2 * B1p + G12 * B2p
        else:
            return S1 * B2p + G21 * B1p

    def wakefields(self, r_nm, zeta_nm, phi=0.0, m_max=4, k_max=3.0):
        r = r_nm * NM_TO_AU
        zeta = np.asarray(zeta_nm) * NM_TO_AU
        Wz = np.zeros_like(zeta, dtype=complex)
        Wr = np.zeros_like(zeta, dtype=complex)

        for m in range(-m_max, m_max + 1):
            for sign in (+1, -1):
                for km in self.resonances(sign, abs(m), k_max=k_max):
                    dZdk = self._dZdk(sign, abs(m), km)
                    if dZdk == 0:
                        continue
                    omega = km * self.v
                    phase = np.exp(1j * m * (phi - self.phi0))
                    prefac_z = km / (2 * np.pi)**2 / abs(dZdk)
                    prefac_r = 1.0 / (2 * np.pi)**2 / abs(dZdk)

                    sumj_z = 0j
                    sumj_r = 0j
                    for j in range(2):
                        aj = self.a[j]
                        Nj = self._N_j(j, abs(m), km, omega)
                        sumj_z += _g(r, aj, abs(m), km) * aj * Nj
                        sumj_r += _dg_dr(aj, r, abs(m), km) * aj * Nj

                    Wz += phase * prefac_z * sumj_z * np.cos(km * zeta)
                    Wr += phase * prefac_r * sumj_r * np.sin(km * zeta)

        return Wz.real * FIELD_AU_TO_GVPM, Wr.real * FIELD_AU_TO_GVPM

    def wakefields_zeta_x(self, zeta_nm, x_nm, m_max=4, k_max=3.0):
        """DWCNT analogue of SWCNT.wakefields_zeta_x -- see its docstring."""
        zeta = np.asarray(zeta_nm) * NM_TO_AU
        x = np.asarray(x_nm) * NM_TO_AU
        Wz_map = np.zeros((len(x), len(zeta)), dtype=complex)
        Wr_map = np.zeros((len(x), len(zeta)), dtype=complex)

        r_abs = np.abs(x)
        sign_conv = np.where(x >= 0, 1.0, -1.0)
        phi_of_x = np.where(x >= 0, 0.0, np.pi)

        for m in range(-m_max, m_max + 1):
            for sign in (+1, -1):
                for km in self.resonances(sign, abs(m), k_max=k_max):
                    dZdk = self._dZdk(sign, abs(m), km)
                    if dZdk == 0:
                        continue
                    omega = km * self.v
                    prefac_z = km / (2 * np.pi)**2 / abs(dZdk)
                    prefac_r = 1.0 / (2 * np.pi)**2 / abs(dZdk)
                    cos_kz = np.cos(km * zeta)
                    sin_kz = np.sin(km * zeta)

                    Nj = [self._N_j(j, abs(m), km, omega) for j in range(2)]

                    for ix in range(len(x)):
                        r = r_abs[ix]
                        phase = np.exp(1j * m * (phi_of_x[ix] - self.phi0))

                        sumj_z = sum(_g(r, self.a[j], abs(m), km) * self.a[j] * Nj[j]
                                     for j in range(2))
                        sumj_r = sum(_dg_dr(self.a[j], r, abs(m), km) * self.a[j] * Nj[j]
                                     for j in range(2))

                        Wz_map[ix, :] += phase * prefac_z * sumj_z * cos_kz
                        Wr_map[ix, :] += sign_conv[ix] * phase * prefac_r * sumj_r * sin_kz

        return Wz_map.real * FIELD_AU_TO_GVPM, Wr_map.real * FIELD_AU_TO_GVPM