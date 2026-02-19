# mechsp/synthesis/synthesis_rot.py
import numpy as np
from typing import Dict, Tuple
from ..magnetics import dipole_Bz  # we use scalar basis b_i(q); combine coils with complex weights

def build_rot_design_matrix(sample_xy: np.ndarray,
                            coil_xy: np.ndarray, h: float,
                            scale: float = 1.0) -> np.ndarray:
    """
    Build real-valued matrix for complex LS:
    Re{Σ A_i b_i(q_j)} ≈ Re{T_j},  Im{Σ A_i b_i(q_j)} ≈ Im{T_j}
    We construct:
       [ Re(B)  -Im(B) ] [Re(A)] = [Re(T)]
       [ Im(B)   Re(B) ] [Im(A)]   [Im(T)]
    where B_ji = b_i(q_j).
    """
    J = sample_xy.shape[0]
    N = coil_xy.shape[0]
    B = np.zeros((J, N))
    # B_z basis values
    for i in range(N):
        B[:, i] = dipole_Bz(sample_xy, coil_xy[i], h, scale=scale)
    # Build block real system
    # X * [Re(A); Im(A)] ≈ [Re(T); Im(T)]
    X = np.block([
        [ B,           -np.zeros_like(B) ],  # we'll inject Im with zeros below
        [ np.zeros_like(B),  B           ]
    ])
    return B, X

def solve_rotating_phasors(sample_xy: np.ndarray,
                           B0_des: np.ndarray,       # (J,) desired magnitude map
                           phi_des: np.ndarray,      # (J,) desired phase map
                           coil_xy: np.ndarray, h: float,
                           lam: float = 1e-4, scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Solve complex LS for coil phasors A_i such that Σ A_i b_i(q_j) ≈ B0(q_j) e^{j phi(q_j)}.
    Returns:
      Irot_amp (N,), Irot_phase (N,), diagnostics
    """
    assert B0_des.shape[0] == sample_xy.shape[0] == phi_des.shape[0]
    J = sample_xy.shape[0]
    N = coil_xy.shape[0]

    B, X = build_rot_design_matrix(sample_xy, coil_xy, h, scale=scale)

    # Target complex T
    T = B0_des * np.exp(1j * phi_des)
    ReT = np.real(T)
    ImT = np.imag(T)
    y = np.concatenate([ReT, ImT], axis=0)  # (2J,)

    # Include lam as Tikhonov on Re(A) and Im(A)
    X_aug = np.vstack([X, np.sqrt(lam)*np.eye(2*N)])
    y_aug = np.concatenate([y, np.zeros(2*N)])

    # Solve normal equations
    ATA = X_aug.T @ X_aug
    ATy = X_aug.T @ y_aug
    sol = np.linalg.solve(ATA, ATy)

    A_re = sol[:N]
    A_im = sol[N:]
    A = A_re + 1j*A_im

    Irot_amp = np.abs(A)
    Irot_phase = np.angle(A)

    # diagnostics (fit error magnitude)
    T_hat = B @ A_re + 1j*(B @ A_im)
    fit_rel = np.linalg.norm(T_hat - T) / max(1.0, np.linalg.norm(T))

    return Irot_amp, Irot_phase, {"rel_fit_err": fit_rel, "cond_hint": np.linalg.cond(ATA)}