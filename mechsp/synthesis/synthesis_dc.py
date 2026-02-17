# mechsp/synthesis/synthesis_dc.py
import numpy as np
from typing import Optional, Dict, Tuple

try:
    from scipy.optimize import lsq_linear
    _HAS_SCIPY_OPT = True
except Exception:
    _HAS_SCIPY_OPT = False

from ..magnetics import grad_Bz_analytic, hessian_Bz_analytic

def build_dc_basis_forces(sample_xy: np.ndarray, coil_xy: np.ndarray, h: float,
                          marble_moment: float = 1.0, scale: float = 1.0) -> np.ndarray:
    """
    Assemble matrix A mapping DC coil currents I0 to forces at sample points.
    A shape: (2J, N); rows are [Fx(q1), Fy(q1), Fx(q2), Fy(q2), ...]
    """
    J = sample_xy.shape[0]
    N = coil_xy.shape[0]
    A = np.zeros((2*J, N))
    for i in range(N):
        G = marble_moment * grad_Bz_analytic(sample_xy, coil_xy[i], h, scale=scale)  # (J,2)
        A[0::2, i] = G[:, 0]
        A[1::2, i] = G[:, 1]
    return A

def solve_dc_currents(sample_xy: np.ndarray,
                      F_des: np.ndarray,                  # (J,2)
                      coil_xy: np.ndarray, h: float,
                      lam: float = 1e-4,
                      weights: Optional[np.ndarray] = None,   # (J,) per-sample weights
                      bounds: Optional[Tuple[float, float]] = None,
                      marble_moment: float = 1.0,
                      scale: float = 1.0) -> Tuple[np.ndarray, Dict]:
    """
    Solve min ||W^{1/2}(A I0 - y)||^2 + lam||I0||^2, with optional bounds.
      - sample_xy: (J,2) points
      - F_des:     (J,2) desired forces at the sample points
      - bounds:    (lo, hi) for |I0_i| <= hi (symmetric box), or None
    Returns:
      I0: (N,) DC currents
      diag: dict with 'cond', 'rel_fit_err', 'method'
    """
    A = build_dc_basis_forces(sample_xy, coil_xy, h, marble_moment, scale)
    y = F_des.reshape(-1)

    # weights
    if weights is not None:
        assert weights.shape[0] == sample_xy.shape[0]
        W = np.repeat(weights, 2)
        A_w = A * W[:, None]
        y_w = y * W
    else:
        A_w, y_w = A, y

    # ridge via augmentation
    sqrt_lam = np.sqrt(lam) if lam > 0 else 0.0
    if lam > 0:
        A_aug = np.vstack([A_w, sqrt_lam*np.eye(A.shape[1])])
        y_aug = np.concatenate([y_w, np.zeros(A.shape[1])])
    else:
        A_aug, y_aug = A_w, y_w

    if bounds is not None and _HAS_SCIPY_OPT:
        lo, hi = bounds
        res = lsq_linear(A_aug, y_aug, bounds=(lo, hi), lsmr_tol='auto', max_iter=1000)
        I0 = res.x
        method = "lsq_linear(bounded)"
    else:
        # normal equations
        ATA = A_aug.T @ A_aug
        ATy = A_aug.T @ y_aug
        I0 = np.linalg.solve(ATA, ATy)
        method = "ridge"

    # diagnostics
    s = np.linalg.svd(A, compute_uv=False)
    cond = (s[0]/s[-1]) if s[-1] > 0 else np.inf
    rel = np.linalg.norm(A @ I0 - y) / max(1.0, np.linalg.norm(y))

    return I0, {"cond": cond, "rel_fit_err": rel, "method": method}

def solve_dc_currents_anchored(sample_xy: np.ndarray,
                               F_des: np.ndarray,                  # (J,2)
                               coil_xy: np.ndarray, h: float,
                               m_ball: float,
                               q_goal: np.ndarray,
                               k_stiff: float,                     # desired local stiffness (K = k I)
                               lam: float = 1e-4,
                               w_region: Optional[np.ndarray] = None,  # (J,) regional weights
                               w_goalF: float = 50.0,              # weight for force-zero at goal
                               w_goalH: float = 10.0,              # weight for curvature at goal
                               bounds: Optional[Tuple[float,float]] = None,
                               scale: float = 1.0
                               ) -> Tuple[np.ndarray, Dict]:
    """
    Anchored DC solve that enforces:
      (i) Force at goal is zero,
     (ii) Local curvature at goal matches K = k_stiff * I_2 (i.e., J_F = -k I).

    We form a single ridge LS:
      min ||W^{1/2}(A I - y)||^2 + w_goalF ||G I - 0||^2 + w_goalH ||H I - vec(-k I)||^2 + lam||I||^2

    Returns:
      I0: (N,), diag
    """
    J = sample_xy.shape[0]
    N = coil_xy.shape[0]

    # Base force map: A I ≈ y
    A = build_dc_basis_forces(sample_xy, coil_xy, h, marble_moment= m_ball, scale=scale)  # (2J,N)
    y = F_des.reshape(-1)                                                                  # (2J,)

    if w_region is not None:
        assert w_region.shape[0] == J
        Wv = np.repeat(w_region, 2)                 # (2J,)
        A_w = A * Wv[:, None]
        y_w = y * Wv
    else:
        A_w, y_w = A, y

    # Goal force block: G I = 0
    G = m_ball * grad_Bz_analytic(q_goal, coil_xy, h, scale=scale)   # (N,2)
    G = G.T                                                          # (2,N) mapping I -> F(qg)
    # Flatten to (2, N) with weight
    A_goalF = np.sqrt(w_goalF) * G
    y_goalF = np.zeros(2)

    # Goal curvature block: H I = vec(-k I)
    H_all = m_ball * hessian_Bz_analytic(q_goal, coil_xy, h, scale=scale)   # (N,2,2)
    # Build (4,N) mapping I -> vec(J_F) in order [xx, xy, yx, yy]
    H = np.stack([H_all[:,0,0], H_all[:,0,1], H_all[:,1,0], H_all[:,1,1]], axis=0)  # (4,N)
    A_goalH = np.sqrt(w_goalH) * H
    y_goalH = np.sqrt(w_goalH) * np.array([-k_stiff, 0.0, 0.0, -k_stiff])           # vec(-k I)

    # Ridge augmentation
    A_ridge = np.sqrt(lam) * np.eye(N)
    y_ridge = np.zeros(N)

    # Stack all blocks
    A_big = np.vstack([A_w, A_goalF, A_goalH, A_ridge])  # ((2J)+2+4+N, N)
    y_big = np.concatenate([y_w, y_goalF, y_goalH, y_ridge])

    # Solve normal equations (bounded option optional)
    ATA = A_big.T @ A_big
    ATy = A_big.T @ y_big
    I0 = np.linalg.solve(ATA, ATy)

    # Diagnostics
    rel = np.linalg.norm(A @ I0 - y) / max(1.0, np.linalg.norm(y))
    Fg = (G @ I0)  # force at goal
    JF = (H @ I0).reshape(2,2)  # Jacobian at goal
    diag = {
        "rel_fit_err": rel,
        "F_goal": Fg,
        "JF_goal": JF,
        "cond": np.linalg.cond(ATA)
    }
    return I0, diag