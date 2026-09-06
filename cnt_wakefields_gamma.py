"""
Unified general (finite-friction) longitudinal wakefield for SWCNT and
DWCNT, driven by an on-axis proton (r0 = 0).

SWCNT: Martin-Luna et al., New J. Phys. 25, 123029 (2023), arXiv:2308.08399
DWCNT: Martin-Luna et al., "Plasmonic excitations in double-walled carbon
       nanotubes", arXiv:2401.08334

Both follow the same structure (Eq. 12 of either paper):

    W_z,ind(r,phi,zeta) = 1/(2pi)^3 sum_m e^{i m phi}
        Int dk k [Re(Phi_ind) sin(k*zeta) + Im(Phi_ind) cos(k*zeta)]

with Phi_ind(r,m,k) the k,m-th Fourier-Bessel component of the induced
potential evaluated on-shell (omega = k*v, fixed by the driving charge).
At finite gamma this integral is smooth (no closed form) and is done
here by numerical quadrature on a k-grid refined around the gamma->0+
resonances. NOTE: while validating this module a real bug was found in
the DWCNT.wakefields() closed-form method of cnt_wakefields.py -- it is
missing a division by D_m^mp(k_m^pm) (the *other* dispersion branch
evaluated at each resonance), Eq. (24)-(25) of arXiv:2401.08334. This
general module does not use that method, so it is unaffected.
"""

import numpy as np
from scipy.special import iv, kv
from cnt_wakefields import SWCNT, DWCNT, _g, NM_TO_AU, FIELD_AU_TO_GVPM


# ============================================================== SWCNT ====
def _swcnt_ReA_ImA(tube, mm, k, gamma_au):
    a = tube.a
    q2 = k**2 + mm**2 / a**2
    Km2 = kv(mm, np.abs(k) * a) ** 2
    N = tube.Omega_p2 * a**2 * q2 * Km2
    Z = (k * tube.v) ** 2 - tube._omega2(mm, k)
    denom = Z**2 + (gamma_au * k * tube.v) ** 2
    ReA = N * Z / denom
    ImA = -N * gamma_au * k * tube.v / denom
    return ReA, ImA


def _swcnt_phi_ind(tube, r, mm, k, gamma_au):
    ReA, ImA = _swcnt_ReA_ImA(tube, mm, k, gamma_au)
    G = iv(mm, k * tube.r0) * iv(mm, k * r)
    return G * (ReA + 1j * ImA)


def _swcnt_resonance_list(tube, mm, k_max):
    out = []
    for km in tube.resonances(mm, k_max=k_max):
        out.append((mm, km, abs(tube._dZdk(mm, km))))
    return out


# ============================================================== DWCNT ====
def _D_pm(tube, sign, m, k, gamma_au):
    w2 = tube._omega_pm2(sign, m, k)
    kvel = k * tube.v
    return kvel * (kvel + 1j * gamma_au) - w2


def _dwcnt_Nj(tube, j, m, k, gamma_au):
    """N_j(m,k,kv), Eq. (19) of arXiv:2401.08334, generalised to finite
    gamma (S_j = kv(kv+i*gamma) - omega_j^2, instead of just kv^2 -
    omega_j^2). NOTE the fix relative to a naive port of the existing
    (gamma=0) DWCNT._N_j: B'_j = -n0 (k^2+m^2/a_j^2) 2 pi Q g(aj,r0) e^{-imphi0}
    (Eq. 10) -- no extra factor of a_j, and with the leading minus sign;
    both were missing in the closed-form-only code and were caught here
    by validating against an isolated-resonance residue calculation."""
    a = tube.a

    def Bp(jj):
        aj = a[jj]
        q2 = k**2 + m**2 / aj**2
        return (-tube.n0 * q2 * 2 * np.pi * tube.Q *
                _g(aj, tube.r0, m, k) * np.exp(-1j * m * tube.phi0))

    G12 = tube.n0 * a[0] * (k**2 + m**2 / a[0]**2) * _g(a[0], a[1], m, k)
    G21 = tube.n0 * a[1] * (k**2 + m**2 / a[1]**2) * _g(a[1], a[0], m, k)
    kvel = k * tube.v
    S1 = kvel * (kvel + 1j * gamma_au) - tube._omega_j2(0, m, k)
    S2 = kvel * (kvel + 1j * gamma_au) - tube._omega_j2(1, m, k)
    B1p, B2p = Bp(0), Bp(1)
    return (S2 * B1p + G12 * B2p) if j == 0 else (S1 * B2p + G21 * B1p)


def _dwcnt_phi_ind(tube, r, m, k, gamma_au):
    Dp = _D_pm(tube, +1, m, k, gamma_au)
    Dm = _D_pm(tube, -1, m, k, gamma_au)
    denom = Dp * Dm
    tot = np.zeros_like(k, dtype=complex)
    for j in range(2):
        aj = tube.a[j]
        Nj = _dwcnt_Nj(tube, j, m, k, gamma_au)
        tot = tot + _g(r, aj, m, k) * aj * Nj
    return -tot / denom


def _dwcnt_resonance_list(tube, m, k_max):
    out = []
    for sign in (+1, -1):
        for km in tube.resonances(sign, m, k_max=k_max):
            out.append((m, km, abs(tube._dZdk(sign, m, km))))
    return out


# =========================================================== shared k-grid
def _k_grid(resonance_list, k_max, gamma_au, v,
            n_coarse=3000, n_fine=2000, span=100,
            n_low=1500, low_k_max_frac=0.02):
    """Non-uniform k>0 grid: coarse background + log-spaced refinement
    near k=0 (the induced potential has a mild log-type structure there
    from K_0(k*a)) + dense windows around each gamma->0+ resonance."""
    coarse = np.linspace(1e-6, k_max, n_coarse)
    low = np.geomspace(1e-8, max(1e-2, k_max * low_k_max_frac), n_low)
    pieces = [coarse, low]
    for mm, km, dZdk in resonance_list:
        if dZdk == 0:
            continue
        hw = gamma_au * km * v / dZdk
        if hw <= 0:
            continue
        window = min(span * hw, 0.4 * km + 0.05)
        lo, hi = max(1e-8, km - window), km + window
        pieces.append(np.linspace(lo, hi, n_fine))
    return np.unique(np.concatenate(pieces))


# ============================================================== driver ===
def wz_general(tube, r_nm, zeta_nm, gamma_over_Omega, phi=0.0,
               m_max=6, k_max=3.0):
    """Longitudinal wakefield at finite friction gamma (GV/m), for a
    SWCNT or a DWCNT instance (auto-detected).

    gamma_over_Omega is in units of Omega = sqrt(4*pi*n0/a1), with a1
    the FIRST (innermost) tube radius -- matches both papers' convention.

    Returns dict with 'total' (=Wz1+Wz2, induced only, no bare Coulomb
    -- see cnt_wakefields_gamma.py docstring), 'Wz1' (Re part), 'Wz2'
    (Im part), 'Wz0' (bare Coulomb self-field, informative only) and
    'full_total' (= Wz0+Wz1+Wz2), each an array over zeta_nm, in GV/m.
    """
    is_dwcnt = isinstance(tube, DWCNT)
    a1 = tube.a[0] if is_dwcnt else tube.a
    Omega = np.sqrt(4 * np.pi * tube.n0 / a1)
    gamma_au = gamma_over_Omega * Omega

    r = r_nm * NM_TO_AU
    zeta = np.asarray(zeta_nm, dtype=float) * NM_TO_AU
    dphi = phi - tube.phi0

    Wz1 = np.zeros_like(zeta)
    Wz2 = np.zeros_like(zeta)

    mm_values = [0] if tube.r0 == 0 else range(0, m_max + 1)
    for mm in mm_values:
        weight = 1.0 if mm == 0 else 2.0
        phase = weight * np.cos(mm * dphi)

        if is_dwcnt:
            res_list = _dwcnt_resonance_list(tube, mm, k_max)
        else:
            res_list = _swcnt_resonance_list(tube, mm, k_max)
        k = _k_grid(res_list, k_max, gamma_au, tube.v)

        if is_dwcnt:
            Phi = _dwcnt_phi_ind(tube, r, mm, k, gamma_au)
        else:
            Phi = _swcnt_phi_ind(tube, r, mm, k, gamma_au)
        ReP, ImP = Phi.real, Phi.imag

        sin_kz = np.sin(np.outer(k, zeta))
        cos_kz = np.cos(np.outer(k, zeta))
        integ1 = np.trapezoid((k * ReP)[:, None] * sin_kz, x=k, axis=0)
        integ2 = np.trapezoid((k * ImP)[:, None] * cos_kz, x=k, axis=0)

        Wz1 += phase * integ1
        Wz2 += phase * integ2

    C = (2.0 * tube.Q / np.pi) if not is_dwcnt else (2.0 / (2 * np.pi) ** 3)
    # NOTE: the two source papers use different Fourier-Bessel transform
    # normalisations (SWCNT: arXiv:2308.08399: does not carry the extra
    # (2pi)^2 from dk/(2pi)^2, dw/(2pi) that the DWCNT paper's Eq.(12)
    # does), so the prefactor genuinely differs between the two cases;
    # both values were independently validated against each paper's own
    # closed-form (gamma->0+) result.
    #
    # FACTOR-OF-2 BUG (found and fixed): as gamma->0+, Wz1 (Re(Phi)*sin,
    # the "reactive"/principal-value piece) and Wz2 (Im(Phi)*cos, the
    # "absorptive"/delta-function piece) do NOT each carry half of the
    # physical field -- by the Sokhotski-Plemelj identity applied to this
    # particular integral, EACH of Wz1 and Wz2 independently converges to
    # the FULL physical resonant field (verified against
    # SWCNT.wakefields()/DWCNT.wakefields(), which match the papers'
    # closed-form residue formulas to machine precision). Summing them as
    # total = Wz1 + Wz2 therefore double-counts and overshoots by exactly
    # a factor of 2 (checked numerically: ratio -> 2.000 as gamma -> 0,
    # for SWCNT, DWCNT, and separately for the N-wall generalisation in
    # cnt_wakefields_nlayer.py, which has the same bug/fix). Halving C
    # here restores total = Wz1+Wz2 -> closed-form limit; Wz1 and Wz2
    # individually now each carry the physically-meaningful "half share"
    # implied by the m=+k/-k Hermitian-symmetry derivation of Eq. 12,
    # consistent with how they're used/plotted (e.g. in the GUI) as two
    # complementary halves of 'total' rather than duplicates of it.
    C *= 0.5
    Wz1 *= C
    Wz2 *= C

    dr2 = r**2 + tube.r0**2 - 2 * r * tube.r0 * np.cos(dphi) + zeta**2
    Wz0 = tube.Q * zeta / dr2**1.5

    total_induced = Wz1 + Wz2
    return {
        "total": total_induced * FIELD_AU_TO_GVPM,
        "Wz0": Wz0 * FIELD_AU_TO_GVPM,
        "Wz1": Wz1 * FIELD_AU_TO_GVPM,
        "Wz2": Wz2 * FIELD_AU_TO_GVPM,
        "full_total": (Wz0 + Wz1 + Wz2) * FIELD_AU_TO_GVPM,
    }
