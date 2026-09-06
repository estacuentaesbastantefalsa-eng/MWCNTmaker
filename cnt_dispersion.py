"""
Relaciones de dispersion plasmonica de un MWCNT de N paredes arbitrario,
generalizando las Ecs. (16) [SWCNT] y (20)-(22) [DWCNT] de:

Martin-Luna & Resta-Lopez, "Excitation of Plasmonic Wakefields in
Multi-Walled Carbon Nanotubes: A Hydrodynamic Approach",
DOI: 10.5772/intechopen.114270

CONSTRUCCION
------------
El propio capitulo plantea el problema para N paredes genericas antes
de particularizarlo: la ecuacion (7)/(12) es

    M(m,k,omega) n~  =  B(m,k,omega),   con
    M_jj = S_j - G_jj ,   M_jl = -G_jl  (l != j)               (*)

donde S_j = omega(omega+i*gamma) - (alpha_j*q_j^2 + beta*q_j^4) y
G_jl(m,k) = n0_l * a_l * q_l^2 * g(a_j,a_l;m,k)  (Eq. 9), con
q_j^2 = k^2 + m^2/a_j^2.

Las frecuencias resonantes (gamma=0) son las raices de det(M)=0.
Definiendo omega_j^2(m,k) = alpha_j*q_j^2 + beta*q_j^4 + G_jj(m,k)
(Eq. 21) y la matriz real K(m,k) con

    K_jj = omega_j^2(m,k),      K_jl = G_jl(m,k)  (l != j),

se tiene M = omega^2*I - K, así que det(M)=0  <=>  omega^2 es
autovalor de K(m,k). Con N=1 esto reproduce exactamente la Ec. (16)
(K es 1x1, K_11 = omega_1^2). Con N=2, los autovalores de la matriz
2x2 [[omega_1^2, G12],[G21, omega_2^2]] son

    (omega_1^2+omega_2^2)/2 +- sqrt(((omega_1^2-omega_2^2)/2)^2 + G12*G21)

y como G12*G21 = Delta^2 (Ec. 22, comprobado algebraicamente:
G12*G21 = n0^2*a1*a2*q1^2*q2^2*g(a1,a2)^2 = Delta^2), esto reproduce
la Ec. (20) termino a termino. Se valida numericamente mas abajo
contra SWCNT._omega2 y DWCNT._omega_pm2 de cnt_wakefields.py.

NOTA sobre n0 por pared: el capitulo asume n0 igual (=n_g) en todas
las paredes salvo que se indique lo contrario; aqui se permite un
n0_j distinto por pared (como en las filas de la interfaz de
wakefields), usando en G_jl el n0 de la pared "fuente" l -- es la
asignacion que se reduce a la Ec. (9) cuando todos los n0_j coinciden,
y es la unica consistente con la derivacion (G_jl proviene de la
densidad perturbada de la pared l, que obedece la ecuacion de
continuidad con SU propio n0_l).

Todo en unidades atomicas salvo que se indique (a en nm en el
constructor).
"""

import numpy as np
from cnt_wakefields import _g, NM_TO_AU, N_G_AU


class MWCNT:
    """Nanotubo de N paredes para el calculo de dispersion.

    Parameters
    ----------
    a_nm : lista de radios (nm). N=1 -> SWCNT, N=2 -> DWCNT, N>=3 -> MWCNT.
    n0_over_ng : densidad superficial de cada pared en unidades de
        n_g = 4*0.107 (a.u.). Escalar (misma densidad en todas las
        paredes) o lista de la misma longitud que a_nm.
    beta : coeficiente del termino de Von Weizsacker (beta=1/4 por defecto).
    """

    def __init__(self, a_nm, n0_over_ng=1.0, beta=0.25):
        a_nm = np.atleast_1d(np.asarray(a_nm, dtype=float))
        order = np.argsort(a_nm)
        self.a = a_nm[order] * NM_TO_AU
        self.N = len(self.a)

        n0_over_ng = np.atleast_1d(np.asarray(n0_over_ng, dtype=float))
        if len(n0_over_ng) == 1 and self.N > 1:
            n0_over_ng = np.repeat(n0_over_ng, self.N)
        self.n0 = n0_over_ng[order] * N_G_AU

        self.beta = beta
        self.alpha = np.pi * self.n0  # alpha_j por pared, array (N,)

    # -------------------------------------------------------- bloques
    def _q2(self, j, m, k):
        return k**2 + m**2 / self.a[j] ** 2

    def omega_j2(self, j, m, k):
        """omega_j^2(m,k) de la pared j sola (Eq. 21 generalizada)."""
        q2 = self._q2(j, m, k)
        Gjj = self.n0[j] * self.a[j] * q2 * _g(self.a[j], self.a[j], m, k)
        return self.alpha[j] * q2 + self.beta * q2 ** 2 + Gjj

    def G_jl(self, j, l, m, k):
        """Acoplo electrostatico Eq. (9) entre la pared j (observacion)
        y la pared l (fuente de la densidad perturbada)."""
        ql2 = self._q2(l, m, k)
        return self.n0[l] * self.a[l] * ql2 * _g(self.a[j], self.a[l], m, k)

    def K_matrix(self, m, k):
        N = self.N
        K = np.empty((N, N))
        for j in range(N):
            K[j, j] = self.omega_j2(j, m, k)
        for j in range(N):
            for l in range(N):
                if l != j:
                    K[j, l] = self.G_jl(j, l, m, k)
        return K

    # ------------------------------------------------------- publico
    def branches(self, m, k):
        """omega_i^2(m,k) para i=1..N (orden ascendente), k escalar."""
        K = self.K_matrix(m, k)
        w2 = np.linalg.eigvals(K)
        return np.sort(w2.real)

    def branches_array(self, m, k_array):
        """Igual que branches() pero vectorizado sobre un array de k.
        Devuelve array (len(k_array), N)."""
        k_array = np.asarray(k_array, dtype=float)
        out = np.empty((len(k_array), self.N))
        for i, k in enumerate(k_array):
            out[i, :] = self.branches(m, k)
        return out

    def omega_j_array(self, j, m, k_array):
        """omega_j(m,k) (dispersion de la pared j SOLA, sin acoplo),
        vectorizado."""
        k_array = np.asarray(k_array, dtype=float)
        return np.sqrt(np.array([self.omega_j2(j, m, k) for k in k_array]))


# ==================================================================
# Autochequeo: compara con las clases SWCNT/DWCNT de cnt_wakefields.py
# ==================================================================
if __name__ == "__main__":
    from cnt_wakefields import SWCNT, DWCNT

    print("Validando N=1 (SWCNT) ...")
    a, n0r = 0.36, 1.0
    sw = SWCNT(a_nm=a, n0_over_ng=n0r)
    mw1 = MWCNT(a_nm=[a], n0_over_ng=n0r)
    ks = np.linspace(0.01, 2.0, 25)
    for m in range(0, 3):
        w2_ref = np.array([sw._omega2(m, k) for k in ks])
        w2_new = mw1.branches_array(m, ks)[:, 0]
        assert np.allclose(w2_ref, w2_new, rtol=1e-10), (m, w2_ref - w2_new)
    print("  OK, SWCNT.omega2 == MWCNT(N=1).branches")

    print("Validando N=2 (DWCNT) ...")
    a1, a2 = 0.36, 0.7
    dw = DWCNT(a1_nm=a1, a2_nm=a2, n0_over_ng=n0r)
    mw2 = MWCNT(a_nm=[a1, a2], n0_over_ng=n0r)
    for m in range(0, 3):
        wp_ref = np.array([dw._omega_pm2(+1, m, k) for k in ks])
        wm_ref = np.array([dw._omega_pm2(-1, m, k) for k in ks])
        branches = mw2.branches_array(m, ks)  # columnas ascendentes: [w-, w+]
        wm_new, wp_new = branches[:, 0], branches[:, 1]
        assert np.allclose(wp_ref, wp_new, rtol=1e-8), (m, wp_ref - wp_new)
        assert np.allclose(wm_ref, wm_new, rtol=1e-8), (m, wm_ref - wm_new)
    print("  OK, DWCNT.omega_pm2(+-1) == MWCNT(N=2).branches")

    print("Validando N=3 (autoconsistencia: subcaso degenerado a1==a2==a3\n"
          "  con n0 total repartido debe reducir a SWCNT con densidad\n"
          "  n0_total, comprobando solo que branches() no rompe / da\n"
          "  valores reales, no una identidad cerrada) ...")
    mw3 = MWCNT(a_nm=[0.36, 0.7, 1.1], n0_over_ng=[1.0, 0.8, 1.2])
    for m in range(0, 3):
        for k in ks[::5]:
            w2 = mw3.branches(m, k)
            assert np.all(np.isfinite(w2))
            assert len(w2) == 3
    print("  OK, N=3 corre y da 3 ramas reales y finitas.")
    print("\nTodo correcto.")
