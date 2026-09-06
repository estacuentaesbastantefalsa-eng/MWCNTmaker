"""
Wakefield longitudinal en un CNT multipared (MWCNT, N paredes arbitrarias),
a friccion finita, generalizando cnt_wakefields.DWCNT / cnt_wakefields_gamma
al caso de N paredes siguiendo el MISMO patron que
graphene_wakefield_nlayer.py (sistema lineal NxN de acoplos por modo,
resuelto numericamente) en vez de perseguir una formula cerrada tipo
Eqs. 24-32 de arXiv:2401.08334 generalizada a N paredes.

------------------------------------------------------------------------
De DWCNT (2 paredes) a MWCNT (N paredes): el diccionario de traduccion
------------------------------------------------------------------------
En DWCNT (cnt_wakefields.py) el sistema que se resuelve, para cada modo
angular m y numero de onda k, es:

    S_1 n_1 - G_12 n_2 = B'_1
   -G_21 n_1 + S_2 n_2 = B'_2

con

    S_j(m,k,omega)  = omega^2 - omega_j^2(m,k)
                     = omega^2 - [alpha_j q_j^2 + beta q_j^4 + G_jj(m,k)]
    G_jl(m,k)        = n0_j * a_j * q_j^2 * g(a_j,a_l;m,k)      (l != j)
    B'_j(m,k)        = -n0_j * q_j^2 * 2*pi*Q * g(a_j,r0;m,k) * e^{-i m phi0}
    q_j^2            = k^2 + m^2/a_j^2
    g(r1,r2;m,k)     = 4*pi*I_m(k r_min) K_m(k r_max)             (Eq. 5)

y el potencial inducido es

    Phi_ind(r,m,k) = - sum_j g(r,a_j;m,k) * a_j * n_j(m,k)

Esto es EXACTAMENTE la misma estructura que el sistema NxN de
graphene_wakefield_nlayer.py,

    S_j(k,w) n_j - sum_l G_jl(k) n_l = B_j(k,w),

solo que:
  * el indice de capa -> indice de pared (radio a_j en vez de altura z_j),
  * el kernel plano exp(-k|zj-zl|) -> el kernel cilindrico g(a_j,a_l;m,k),
  * hay una suma extra sobre el modo angular m (irrelevante si r0=0, donde
    solo sobrevive m=0, igual que en SWCNT/DWCNT).

A friccion finita gamma_j, S_j se generaliza igual que en el caso plano:

    S_j(m,k,omega,gamma_j) = omega*(omega + i*gamma_j) - alpha_j q_j^2 - beta q_j^4

(el "omega^2" de gamma=0 es el caso limite omega*(omega+i*0)).

Con esto el sistema NxN

    M_jl = S_j delta_jl - G_jl (1-delta_jl),      M n = B'

se resuelve numericamente en cada (m,k) con np.linalg.solve (bateado sobre
k), exactamente como _solve_layers() en graphene_wakefield_nlayer.py.

------------------------------------------------------------------------
Por que NO se generaliza la formula cerrada gamma->0+
------------------------------------------------------------------------
Para N=2, el limite gamma->0+ (Eqs. 24-25 del paper DWCNT) se obtiene por
residuos: en cada resonancia k_m^(+/-) (raiz de Z_+ o de Z_-), el otro
factor D_other = Z_mp(k) (la OTRA rama) aparece en el denominador. Para N
paredes, el "otro factor" en cada resonancia pasa a ser el producto de las
N-1 ramas restantes evaluadas ahi -- combinatoriamente mas caro y menos
robusto numericamente (raices casi degeneradas, multiplicidad, etc.) que
simplemente integrar en k con friccion finita pequena, que es un limite
suave y bien definido (igual que ya se hace en cnt_wakefields_gamma.py y
en graphene_wakefield_nlayer.py, que NUNCA usa formula cerrada). Por eso
este modulo solo ofrece la ruta numerica de friccion finita.

Como validacion, MWCNT con N=1 pared reproduce SWCNT.wakefields()/
wz_general(SWCNT) y con N=2 paredes reproduce DWCNT/wz_general(DWCNT)
(ver test al final del fichero, ejecutado durante el desarrollo).
"""

import numpy as np
from scipy.optimize import brentq

from cnt_wakefields import _g, NM_TO_AU, FIELD_AU_TO_GVPM, C_AU, N_G_AU
from cnt_wakefields_gamma import _k_grid
class MWCNT:
    """CNT de N paredes (N arbitrario >= 1), driver on/off-axis.

    Parameters (physical units)
    ----------------------------
    a_nm : sequence of float
        Radios de las N paredes, en nm (cualquier orden; se ordenan).
    v_over_c : velocidad del driver / c.
    n0_over_ng : float o secuencia de N floats
        Densidad superficial de cada pared, en unidades de
        n_g = 4*0.107 a.u. (mismo n_g que SWCNT/DWCNT). Un solo float se
        difunde (broadcast) a todas las paredes.
    gamma_over_Omega_default : friccion por defecto (ver wz_general_nlayer;
        normalmente se pasa gamma_over_Omega directamente a esa funcion en
        vez de aqui).
    Q : carga del driver (proton = +1).
    r0_nm, phi0 : posicion radial (nm) y azimutal (rad) del driver;
        r0=0 excita solo el modo m=0 (igual que en SWCNT/DWCNT).
    """

    def __init__(self, a_nm, v_over_c=0.05, n0_over_ng=1.0, Q=1.0,
                 r0_nm=0.0, phi0=0.0):
        a_nm = np.atleast_1d(np.asarray(a_nm, dtype=float))
        order = np.argsort(a_nm)
        self.a = a_nm[order] * NM_TO_AU
        self.N = len(self.a)

        # broadcast_to funciona igual con escalar, lista de 1 o de N
        # elementos; antes, una lista de 1 elemento con N=1 pared se colaba
        # por la rama float(n0_over_ng), que fallaba (float() no acepta listas)
        n0_over_ng = np.broadcast_to(
            np.atleast_1d(np.asarray(n0_over_ng, dtype=float)), (self.N,))[order]
        self.n0 = n0_over_ng * N_G_AU          # (N,) -- n0_j, una por pared
        self.alpha = np.pi * self.n0            # (N,) -- alpha_j = vFj^2/2 = pi*n0_j
        self.beta = 0.25

        self.v = v_over_c * C_AU
        self.Q = Q
        self.r0 = r0_nm * NM_TO_AU
        self.phi0 = phi0

    # ------------------------------------------------------------ k-space
    def _build_M_B(self, m, k, gamma_au):
        """Construye M (N,N) y B (N,) para cada k del array k (batched).
        gamma_au: escalar o array de N (friccion por pared)."""
        k = np.asarray(k, dtype=float)
        nk = k.size
        N = self.N
        gamma_au_arr = np.atleast_1d(np.asarray(gamma_au, dtype=float))
        gamma_arr = np.broadcast_to(gamma_au_arr, (N,)) if gamma_au_arr.size > 1 \
            else np.full(N, float(gamma_au_arr.reshape(-1)[0]))

        omega = k * self.v
        M = np.zeros((nk, N, N), dtype=complex)
        B = np.zeros((nk, N), dtype=complex)

        q2 = k[:, None] ** 2 + (m ** 2) / (self.a[None, :] ** 2)   # (nk,N)

        for j in range(N):
            aj = self.a[j]
            q2j = q2[:, j]
            Sj = omega * (omega + 1j * gamma_arr[j]) - self.alpha[j] * q2j - self.beta * q2j ** 2
            B[:, j] = -self.n0[j] * q2j * 2 * np.pi * self.Q * \
                _g(aj, self.r0, m, k) * np.exp(-1j * m * self.phi0)
            for l in range(N):
                al = self.a[l]
                Gjl = self.n0[j] * aj * q2j * _g(aj, al, m, k)
                if l == j:
                    # auto-acoplo G_jj (Eq. 21: omega_j^2 = alpha q^2 +
                    # beta q^4 + G_jj YA incluye este termino) -- se resta
                    # de la diagonal exactamente igual que en
                    # graphene_wakefield_nlayer.py (M[j,j] = Sj - Gjj), NO
                    # se deja Sj solo.
                    M[:, j, j] = Sj - Gjl
                else:
                    M[:, j, l] = -Gjl

        return M, B

    def _solve_walls(self, m, k, gamma_au):
        """n_j(m,k), j=0..N-1, solucion del sistema NxN. Shape (nk,N)."""
        M, B = self._build_M_B(m, k, gamma_au)
        return np.linalg.solve(M, B[..., None])[..., 0]

    def _det0(self, m, k):
        """det M a gamma=0 (real), usado solo para localizar resonancias.
        Acepta k escalar o array; devuelve la misma forma."""
        k_arr = np.atleast_1d(np.asarray(k, dtype=float))
        M, _ = self._build_M_B(m, k_arr, np.zeros(self.N))
        d = np.linalg.det(M).real
        return d[0] if np.ndim(k) == 0 else d

    def resonances(self, m, k_max=3.0, n_scan=4000):
        """Raices positivas de det(M(m,k,gamma=0))=0 en (0,k_max]. Cada raiz
        es una resonancia de UNA de las N ramas (generaliza Z=0 de SWCNT y
        Z_+ = 0 / Z_- = 0 de DWCNT: para N=1,2 este determinante coincide,
        salvo un factor, con Z(k) / Z_+(k)*Z_-(k))."""
        ks = np.linspace(1e-6, k_max, n_scan)
        Dk = self._det0(m, ks)
        roots = []
        for i in range(len(ks) - 1):
            if Dk[i] == 0:
                roots.append(ks[i])
            elif Dk[i] * Dk[i + 1] < 0:
                roots.append(brentq(lambda kk: self._det0(m, kk), ks[i], ks[i + 1]))
        return roots

    def _dDdk(self, m, k, h=1e-6):
        return float((self._det0(m, k + h) - self._det0(m, k - h)) / (2 * h))

    # -------------------------------------------------- residuo en km ----
    def resonance_residue(self, m, km, h=1e-6):
        """Datos del residuo del sistema NxN en la resonancia k=km (raiz de
        det M(m,k,gamma=0)=0), usados para construir la formula cerrada
        gamma->0+ (wakefields_closed). No requiere descomponer en "ramas"
        (a diferencia de DWCNT.wakefields, que usa D_other = la otra rama
        Z_mp): basta con el vector nulo derecho/izquierdo de M(km), validos
        para cualquier N.

        Cerca de km, M(k)^{-1} B(k) ~ vR (vL^T B) / [(k-km) A], con
        A = vL^T (dM/dk) vR -- formula estandar de perturbacion de
        autovalores simples, invariante bajo reescalado de vR o vL por
        separado (por eso no importa la normalizacion que use la SVD).

        Returns: vR, vL (vectores, (N,)), B0 (fuente en km, (N,)),
                 A (escalar real), Bw (escalar real, "peso" de friccion).
        """
        M0, B0 = self._build_M_B(m, np.array([km]), np.zeros(self.N))
        M0 = M0[0]
        B0 = B0[0]
        U, S, Vh = np.linalg.svd(M0)
        vR = Vh[-1, :].conj()
        vL = U[:, -1].conj()
        Mp, _ = self._build_M_B(m, np.array([km + h]), np.zeros(self.N))
        Mm, _ = self._build_M_B(m, np.array([km - h]), np.zeros(self.N))
        dMdk = (Mp[0] - Mm[0]) / (2 * h)
        A = (vL @ dMdk @ vR).real
        omega = km * self.v
        Bw = (omega * (vL @ vR)).real
        return vR, vL, B0, A, Bw


# =========================================================== phi inducido
def _nlayer_phi_ind(tube, r, m, k, gamma_au):
    """Phi_ind(r,m,k) = -sum_j g(r,a_j;m,k) * a_j * n_j(m,k)."""
    n = tube._solve_walls(m, k, gamma_au)          # (nk, N) complex
    tot = np.zeros_like(np.asarray(k, dtype=float), dtype=complex)
    for j in range(tube.N):
        aj = tube.a[j]
        tot = tot + _g(r, aj, m, k) * aj * n[:, j]
    return -tot


def _nlayer_resonance_list(tube, m, k_max):
    out = []
    for km in tube.resonances(m, k_max=k_max):
        out.append((m, km, abs(tube._dDdk(m, km))))
    return out


# ======================================================= relacion de dispersion
def dispersion_branches(tube, m, k, with_restoring=True):
    """Relacion de dispersion INTRINSECA del tubo, omega_b(m,k) para las N
    ramas hibridas de las N paredes (independiente de v: es una propiedad
    del gas de electrones + geometria, no del driver).

    A gamma=0, el sistema lineal de _build_M_B es
        M(m,k,omega) = omega^2 * I - D(m,k),   con
        D_jj(m,k) = [alpha_j q_j^2 + beta q_j^4] * with_restoring + G_jj(m,k)
        D_jl(m,k) = G_jl(m,k)                                        (l!=j)
    (compruebese: M_jj = Sj - Gjj = omega^2 - alpha_j q_j^2 - beta q_j^4 -
    Gjj = omega^2 - D_jj; M_jl = -Gjl = -D_jl para l!=j -- exactamente la
    misma D_jl/Gjl de _build_M_B, solo que aqui SE RESUELVE EN omega en vez
    de fijarlo a k*v). det(M)=0 <=> omega^2 es autovalor de D(m,k) -- las N
    ramas se obtienen diagonalizando D(m,k) para cada k (sin busqueda de
    raices).

    with_restoring=True (modelo completo, el que usa el resto del modulo):
    incluye alpha_j*q_j^2 (presion de Fermi/degeneracion, vF_j^2/2 * q^2) y
    beta*q_j^4 (potencial cuantico de Bohm) -- las "frecuencias
    restauradoras" del modelo hidrodinamico, ademas del acoplo
    electrostatico G_jl.
    with_restoring=False (modelo "ideal", sin esos terminos): solo queda el
    acoplo electrostatico G_jl -- el analogo de un plasmon de superficie
    puramente coulombiano. Para m=0, q_j^2=k^2 (sin termino de curvatura
    m^2/a_j^2), asi que TODOS los terminos de D (con o sin restauracion) se
    anulan cuando k->0 -- ambas curvas bajan a omega=0 en k=0 para m=0;
    donde difieren es en la FORMA de esa subida (la parte de restauracion
    domina a k pequeno frente al termino electrostatico, que lleva un log
    de K_0(ka) ~ -ln(ka)).

    Para N=1 (SWCNT) esto reproduce exactamente SWCNT._omega2(m,k) (con
    with_restoring=True); para N=2 (DWCNT), las dos ramas reproducen
    DWCNT._omega_pm2(+1/-1,m,k) -- verificado numericamente.

    Devuelve omega array (N, len(k)) en u.a., ramas ordenadas de menor a
    mayor en cada k (parte real de los autovalores de D; deberian ser
    reales para un sistema fisicamente estable -- se descarta la parte
    imaginaria residual de redondeo).
    """
    k = np.atleast_1d(np.asarray(k, dtype=float))
    N = tube.N
    nk = k.size
    q2 = k[:, None] ** 2 + (m ** 2) / (tube.a[None, :] ** 2)   # (nk, N)

    D = np.zeros((nk, N, N), dtype=float)
    for j in range(N):
        aj = tube.a[j]
        q2j = q2[:, j]
        Gjj = tube.n0[j] * aj * q2j * _g(aj, aj, m, k)
        diag = Gjj.copy()
        if with_restoring:
            diag = diag + tube.alpha[j] * q2j + tube.beta * q2j ** 2
        D[:, j, j] = diag
        for l in range(N):
            if l == j:
                continue
            al = tube.a[l]
            D[:, j, l] = tube.n0[j] * aj * q2j * _g(aj, al, m, k)

    eig = np.linalg.eigvals(D)              # (nk, N), en general complejo
    eig = np.sort(np.clip(eig.real, 0.0, None), axis=1)
    return np.sqrt(eig).T                    # (N, nk)


# ============================================================== driver ===
def wz_general_nlayer(tube, r_nm, zeta_nm, gamma_over_Omega, phi=0.0,
                       m_max=6, k_max=3.0):
    """Wakefield longitudinal a friccion finita gamma (GV/m) para un
    MWCNT (N paredes). Firma y convenciones identicas a
    cnt_wakefields_gamma.wz_general (SWCNT/DWCNT), del que es la
    generalizacion directa a N paredes.

    gamma_over_Omega esta en unidades de Omega = sqrt(4*pi*n0_1/a_1), con
    a_1, n0_1 el radio y densidad de la pared MAS INTERNA (misma
    convencion que SWCNT/DWCNT). gamma se aplica por igual a todas las
    paredes (gamma_j = gamma_over_Omega * Omega para todo j).

    Returns dict con 'total' (=Wz1+Wz2, solo inducido), 'Wz1' (parte Re),
    'Wz2' (parte Im), 'Wz0' (Coulomb desnudo, informativo) y 'full_total'
    (=Wz0+Wz1+Wz2), cada uno un array sobre zeta_nm, en GV/m.
    """
    a1 = tube.a[0]
    n0_1 = tube.n0[0]
    Omega = np.sqrt(4 * np.pi * n0_1 / a1)
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

        res_list = _nlayer_resonance_list(tube, mm, k_max)
        k = _k_grid(res_list, k_max, gamma_au, tube.v)

        Phi = _nlayer_phi_ind(tube, r, mm, k, gamma_au)
        ReP, ImP = Phi.real, Phi.imag

        sin_kz = np.sin(np.outer(k, zeta))
        cos_kz = np.cos(np.outer(k, zeta))
        integ1 = np.trapezoid((k * ReP)[:, None] * sin_kz, x=k, axis=0)
        integ2 = np.trapezoid((k * ImP)[:, None] * cos_kz, x=k, axis=0)

        Wz1 += phase * integ1
        Wz2 += phase * integ2

    # misma normalizacion que el caso DWCNT (ver nota en
    # cnt_wakefields_gamma.wz_general): esta es la generalizacion directa
    # de esa derivacion (Eq. 12 de arXiv:2401.08334) a N paredes.
    #
    # BUG DE FACTOR 2 (encontrado y corregido): igual que en
    # cnt_wakefields_gamma.wz_general, Wz1 (parte Re, "reactiva") y Wz2
    # (parte Im, "absortiva") NO son dos mitades complementarias del
    # campo -- en el limite gamma->0+, CADA UNA por separado converge ya
    # al campo fisico completo (verificado frente a wakefields_closed(),
    # que reproduce SWCNT.wakefields()/DWCNT.wakefields() a precision de
    # maquina). Sumarlas como total=Wz1+Wz2 duplicaba el resultado
    # (ratio numerico -> 2.000 al hacer gamma->0). Se corrige aqui
    # dividiendo C entre 2.
    C = 2.0 / (2 * np.pi) ** 3
    C *= 0.5
    Wz1 *= C
    Wz2 *= C

    dr2 = r ** 2 + tube.r0 ** 2 - 2 * r * tube.r0 * np.cos(dphi) + zeta ** 2
    Wz0 = tube.Q * zeta / dr2 ** 1.5

    total_induced = Wz1 + Wz2
    return {
        "total": total_induced * FIELD_AU_TO_GVPM,
        "Wz0": Wz0 * FIELD_AU_TO_GVPM,
        "Wz1": Wz1 * FIELD_AU_TO_GVPM,
        "Wz2": Wz2 * FIELD_AU_TO_GVPM,
        "full_total": (Wz0 + Wz1 + Wz2) * FIELD_AU_TO_GVPM,
    }


# =================================================== limite cerrado gamma->0+
def wakefields_closed(tube, r_nm, zeta_nm, phi=0.0, m_max=6, k_max=3.0):
    """Limite cerrado gamma->0+ EXACTO para un MWCNT de N paredes
    arbitrario -- generaliza SWCNT.wakefields()/DWCNT.wakefields() (Eqs.
    17-18 / 24-32) sin necesitar el "producto de las N-1 ramas restantes"
    en cada resonancia (que es la generalizacion combinatoriamente cara
    descartada en el docstring del modulo). En su lugar, se usa la formula
    estandar de residuos de un sistema lineal cerca de un autovalor simple
    (ver MWCNT.resonance_residue), que es valida para N arbitrario.

    ------------------------------------------------------------------
    Derivacion (resumen)
    ------------------------------------------------------------------
    Cerca de una resonancia k=km (raiz de det M(m,k,gamma=0)=0), el
    potencial inducido se comporta como un polo simple en k:

        Phi_ind(k,gamma) ~ NUM(km) / [A(km)*(k-km) + i*gamma*B(km)]

    con A = vL^T (dM/dk) vR, B = omega(km)*(vL^T vR) y
    NUM(km) = -[sum_j g(r,aj;m,km) aj (vR)_j] * (vL^T B'(km))
    (vR, vL: vectores nulos dcha./izda. de M(km); B'(km): termino fuente).

    Sustituyendo esta forma de polo en la integral general (la misma que
    evalua wz_general_nlayer numericamente) y tomando gamma->0+ con la
    identidad estandar eps/(x^2+eps^2) -> pi*sign(eps)*delta(x), la parte
    oscilante par (Ax, PV=0 por simetria) se anula y solo sobrevive el
    termino tipo delta -- dando, por resonancia:

        contribucion a Wz(zeta) = -pi*km*sign(B)/|A| * Re[NUM * e^{i km zeta}]

    Validado (ver notas de desarrollo): para N=1 y N=2 esta formula
    reproduce SWCNT.wakefields()/DWCNT.wakefields() a precision de
    maquina (phi0=0; con phi0!=0 hay una diferencia de convencion de fase
    heredada de una inconsistencia YA EXISTENTE entre esos dos metodos en
    cnt_wakefields.py -- este modulo sigue la convencion de DWCNT, que es
    la misma que usa wz_general_nlayer). Para N>=3 coincide con el limite
    numerico (wz_general_nlayer a gamma muy pequeno) dentro del error de
    cuadratura esperado.

    A diferencia de wz_general_nlayer, esta funcion NO discretiza k: cada
    resonancia se evalua exactamente en cualquier zeta, por grande que
    sea, sin aliasing ni coste adicional -- ver la explicacion sobre el
    problema de muestreo en k para zeta grande.

    Returns: array Wz(zeta) en GV/m (solo campo inducido; el Coulomb
    desnudo Wz0 no esta incluido, igual que en wz_general_nlayer["Wz1"]+
    ["Wz2"], que es la cantidad a la que esto converge).
    """
    r = r_nm * NM_TO_AU
    zeta = np.asarray(zeta_nm, dtype=float) * NM_TO_AU
    dphi = phi - tube.phi0

    Wz = np.zeros_like(zeta)
    mm_values = [0] if tube.r0 == 0 else range(0, m_max + 1)
    for mm in mm_values:
        weight = 1.0 if mm == 0 else 2.0
        phase = weight * np.cos(mm * dphi)

        for km in tube.resonances(mm, k_max=k_max):
            vR, vL, B0, A, Bw = tube.resonance_residue(mm, km)
            if A == 0 or Bw == 0:
                continue
            tot = 0j
            for j in range(tube.N):
                aj = tube.a[j]
                tot += _g(r, aj, mm, km) * aj * vR[j]
            NUM = -tot * (vL @ B0)

            term = -np.pi * km * np.sign(Bw) / abs(A) * (NUM * np.exp(1j * km * zeta)).real
            Wz += phase * term

    C = 2.0 / (2 * np.pi) ** 3        # misma normalizacion que wz_general_nlayer
    return Wz * C * FIELD_AU_TO_GVPM
