# mechsp/eff_dynamics.py

from dataclasses import dataclass
from typing import Callable
import numpy as np

from .fields import FieldDesign
from .magnetics import dipole_Bz, grad_Bz_analytic, hessian_Bz_analytic

@dataclass
class EffModel:
    gradK: Callable[[np.ndarray], np.ndarray]   # ∇K_eff(q)
    N_eff: Callable[[np.ndarray, np.ndarray], np.ndarray]  # N_eff(q) possibly using dot{q} if you refine later
    M_eff: Callable[[np.ndarray], np.ndarray]   # M_eff(q)
    c_damp: float                               # isotropic damping coefficient

def build_eff_model(design: FieldDesign, m_ball: float, kappa: float = None, c_damp: float = 0.12) -> EffModel:
    """
    Fast averaged model with vectorized analytic derivatives and optional damping.

    N_eff approximation used: N_eff(q) = gamma(q) * J, where
        gamma(q) = m_b * B0(q) * sin(delta) * ||∇phi(q)||.
    ∇phi computed from the complex phasor S(q) = Σ A_i b_i(q), with
        A_i = Irot_amp_i * exp(j * Irot_phase_i).
    """
    I2 = np.eye(2)
    J = np.array([[0.0, -1.0],
                  [1.0,  0.0]])

    # ----- prepack coil arrays for vectorized ops -----
    coil_xy = design.coil_xy
    h = design.h
    scale = design.scale
    m_b = m_ball

    # complex phasors for rotation
    A = design.Irot_amp * np.exp(1j * design.Irot_phase)
    A_re = np.real(A)  # (N,)
    A_im = np.imag(A)

    # HF weights for B1 (phase kept via cos(theta) as in FieldDesign.B1)
    HF_w = design.Ihf_amp * np.cos(design.Ihf_phase)  # (N,)

    # delta (phase lag)
    if kappa is not None and design.omega is not None and design.omega < kappa:
        delta = np.arcsin(design.omega / kappa)
    else:
        delta = 0.2

    # ---------- gradK(q) = - m_b * ∇Bz0(q) ----------
    I0 = design.I0  # (N,)

    
    def gradK(q: np.ndarray) -> np.ndarray:
        # ∇K_eff = - m_b * Σ I0_i ∇b_i(q)
        G_all = grad_Bz_analytic(q, coil_xy, h, scale=scale)    # (N,2) because q is (2,), coil_xy is (N,2)
        return -m_b * (I0[:, None] * G_all).sum(axis=0)


    # ---------- N_eff(q) via analytic ∇phi from complex phasor ----------
    # S(q) = Σ A_i b_i(q) = a(q) + j b(q).
    # ∇a(q) = Σ Re(A_i) ∇b_i(q),  ∇b(q) = Σ Im(A_i) ∇b_i(q)
    def N_eff(q: np.ndarray, v: np.ndarray = None) -> np.ndarray:
        # B(q) basis values
        b_vals = np.array([ # (N,)
            # evaluate Bz basis for all coils at this q
            # we vectorize via a single pass over coils using broadcasting in magnetics if available
            # here we call dipole grad only once and reconstruct a,b via scalar Bz
            # but to keep it light, compute Bz for all coils inline:
            # we reuse grad_Bz_analytic to avoid adding dipole_Bz_vectorized; a tiny cost, but OK.
            ], dtype=float)

        # Efficient: compute S(q) and ∇a, ∇b in one pass via grad_Bz_analytic
        Q = np.repeat(q[None, :], coil_xy.shape[0], axis=0)       # (N,2)
        # scalar Bz values per coil:
        # We'll recompute using the closed form quickly:
        rx = Q[:,0] - coil_xy[:,0]
        ry = Q[:,1] - coil_xy[:,1]
        rz = h
        R2 = rx*rx + ry*ry + rz*rz
        Rm3 = R2**(-1.5)
        Rm5 = R2**(-2.5)
        Bz_i = scale*( 3.0*rz*rz*Rm5 - Rm3 )     # (N,)

        # S(q) = Σ A_i b_i(q) = a + j b
        Bz_i = dipole_Bz(q, coil_xy, h, scale=scale)            # (N,)
        a = float(np.dot(A_re, Bz_i))
        b = float(np.dot(A_im, Bz_i))
        S2 = a*a + b*b
        if S2 < 1e-16:
            return np.zeros((2, 2))

        # ∇a = Σ Re(A_i) ∇b_i,  ∇b = Σ Im(A_i) ∇b_i
        G_all = grad_Bz_analytic(q, coil_xy, h, scale=scale)    # (N,2)
        grad_a = (A_re[:, None] * G_all).sum(axis=0)            # (2,)
        grad_b = (A_im[:, None] * G_all).sum(axis=0)            # (2,)

        grad_phi = (a*grad_b - b*grad_a) / S2
        B0 = np.sqrt(S2)
        gamma = m_b * B0 * np.sin(delta) * np.linalg.norm(grad_phi)
        return gamma * J


    # ---------- M_eff(q) via analytic Hessian of B1 ----------
    c_mass = (design.eps**2) / (2.0 * (design.Omega**2)) if design.Omega > 0 else 0.0

    def M_eff(q: np.ndarray) -> np.ndarray:
        if c_mass == 0.0 or np.all(HF_w == 0.0):
            return m_b * I2
        H_all = hessian_Bz_analytic(q, coil_xy, h, scale=scale) # (N,2,2)
        H_sum = (HF_w[:, None, None] * H_all).sum(axis=0)
        return m_b * I2 + c_mass * H_sum


    return EffModel(gradK=gradK, N_eff=N_eff, M_eff=M_eff, c_damp=c_damp)