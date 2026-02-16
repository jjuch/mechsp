# mechsp/synthesis_hf.py
import numpy as np
from typing import Dict, Tuple

from ..magnetics import grad_Bz_analytic

def numeric_hessian_bz(sample_xy: np.ndarray, coil_xy_i: np.ndarray, h: float,
                       scale: float = 1.0, eps: float = 1e-4) -> np.ndarray:
    """
    Compute Hessian D^2 b_i(q) numerically (2x2) at all sample points (J).
    Returns H: (J, 2, 2)
    """
    # central differences on ∂/∂x, ∂/∂y of ∇Bz
    ex = np.array([eps, 0.0])
    ey = np.array([0.0, eps])

    # G    = grad_Bz_analytic(sample_xy, coil_xy_i, h, scale=scale)  # (J,2)
    G_xp = grad_Bz_analytic(sample_xy + ex, coil_xy_i, h, scale=scale)
    G_xm = grad_Bz_analytic(sample_xy - ex, coil_xy_i, h, scale=scale)
    G_yp = grad_Bz_analytic(sample_xy + ey, coil_xy_i, h, scale=scale)
    G_ym = grad_Bz_analytic(sample_xy - ey, coil_xy_i, h, scale=scale)

    dGdx = (G_xp - G_xm) / (2*eps)   # (J,2)
    dGdy = (G_yp - G_ym) / (2*eps)   # (J,2)

    # Hessian entries: Hxx = ∂^2Bz/∂x^2, Hxy = ∂^2Bz/∂x∂y, Hyx = ∂^2Bz/∂y∂x, Hyy = ∂^2Bz/∂y^2
    H = np.zeros((sample_xy.shape[0], 2, 2))
    H[:, 0, 0] = dGdx[:, 0]  # ∂/∂x (∂Bz/∂x)
    H[:, 0, 1] = dGdy[:, 0]  # ∂/∂y (∂Bz/∂x)
    H[:, 1, 0] = dGdx[:, 1]  # ∂/∂x (∂Bz/∂y)
    H[:, 1, 1] = dGdy[:, 1]  # ∂/∂y (∂Bz/∂y)
    return H

def build_hessian_design(sample_xy: np.ndarray,
                         coil_xy: np.ndarray, h: float,
                         scale: float = 1.0) -> np.ndarray:
    """
    Build linear map Hmat * beta ≈ hess_vec_target, where:
      - beta: (N,)
      - Hmat: stacks J blocks, each maps beta -> vec(H(q_j)) with order [xx, xy, yx, yy]
    Returns Hmat: shape (4J, N)
    """
    J = sample_xy.shape[0]
    N = coil_xy.shape[0]
    Hmat = np.zeros((4*J, N))
    for i in range(N):
        H_i = numeric_hessian_bz(sample_xy, coil_xy[i], h, scale=scale)  # (J,2,2)
        # flatten into order [xx, xy, yx, yy]
        Hvec = np.stack([H_i[:,0,0], H_i[:,0,1], H_i[:,1,0], H_i[:,1,1]], axis=1)  # (J,4)
        Hmat[:, i] = Hvec.reshape(-1)  # (4J,)
    return Hmat

def solve_hf_modulation(sample_xy: np.ndarray,
                        DeltaM_des: np.ndarray,       # (J,2,2) desired inertia correction SPD
                        coil_xy: np.ndarray, h: float,
                        lam_H: float = 1e-3, scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Solve Hmat * beta ≈ vec(DeltaM_des), ridge regularization.
    Returns:
      Ihf_amp:  (N,)   # use beta as amplitude scaling; phase=0 initially
      Ihf_phase:(N,)   # zeros for now
      diagnostics
    """
    J = sample_xy.shape[0]
    assert DeltaM_des.shape == (J,2,2)
    Hmat = build_hessian_design(sample_xy, coil_xy, h, scale=scale)
    # target vector: order [xx, xy, yx, yy]
    tgt = np.stack([DeltaM_des[:,0,0],
                    DeltaM_des[:,0,1],
                    DeltaM_des[:,1,0],
                    DeltaM_des[:,1,1]], axis=1).reshape(-1)

    # ridge solve
    A_aug = np.vstack([Hmat, np.sqrt(lam_H)*np.eye(Hmat.shape[1])])
    y_aug = np.concatenate([tgt, np.zeros(Hmat.shape[1])])
    ATA = A_aug.T @ A_aug
    ATy = A_aug.T @ y_aug
    beta = np.linalg.solve(ATA, ATy)

    # diagnostics
    residual = np.linalg.norm(Hmat @ beta - tgt) / max(1.0, np.linalg.norm(tgt))
    cond = np.linalg.cond(ATA)

    Ihf_amp   = beta.copy()
    Ihf_phase = np.zeros_like(beta)  # set phases later if needed

    return Ihf_amp, Ihf_phase, {"rel_fit_err": residual, "cond_hint": cond}