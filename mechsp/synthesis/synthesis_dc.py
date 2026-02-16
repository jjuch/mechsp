# mechsp/synthesis_dc.py
import numpy as np
from typing import Optional, Dict, Tuple

try:
    from scipy.optimize import lsq_linear
    _HAS_SCIPY_OPT = True
except Exception:
    _HAS_SCIPY_OPT = False

from ..magnetics import grad_Bz_analytic

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