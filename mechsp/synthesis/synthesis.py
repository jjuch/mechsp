# mechsp/synthesis.py
import numpy as np
from ..magnetics import grad_Bz, dipole_Bz, grad_Bz_analytic

try:
    from scipy.linalg import cho_factor, cho_solve
    _HAS_SCIPY_LINALG = True
except Exception:
    _HAS_SCIPY_LINALG = False

try:
    from scipy.optimize import lsq_linear
    _HAS_SCIPY_OPT = True
except Exception:
    _HAS_SCIPY_OPT = False


def build_basis_forces(sample_xy, coil_xy, h, marble_moment=1.0, scale=1.0):
    """
    Assemble basis force matrix A for linear mapping I -> F(sample_xy).
    For each coil i, basis force at q is f_i(q) = marble_moment * ∇Bz_i(q).
    Returns A shape (2J, N).
    """
    J = sample_xy.shape[0]
    N = coil_xy.shape[0]
    A = np.zeros((2 * J, N))
    for i in range(N):
        G = marble_moment * grad_Bz_analytic(sample_xy, coil_xy[i], h, scale=scale)  # (J,2)
        A[0::2, i] = G[:, 0]
        A[1::2, i] = G[:, 1]
    return A


def synthesize_currents(sample_xy, F_target, coil_xy, h,
                        lam=1e-6, marble_moment=1.0, scale=1.0):
    """
    Ridge LS: min_I ||A I - y||^2 + lam ||I||^2
    F_target: (J,2) desired forces at sample points.
    Returns I (N,), and conditioning diagnostics.
    """
    A = build_basis_forces(sample_xy, coil_xy, h, marble_moment, scale)
    y = F_target.reshape(-1)  # (2J,)
    # Normal equations (fine for prototyping):
    ATA = A.T @ A + lam * np.eye(A.shape[1])
    ATy = A.T @ y
    I = np.linalg.solve(ATA, ATy)
    # diagnostics
    s = np.linalg.svd(A, compute_uv=False)
    cond = (s[0] / s[-1]) if s[-1] > 0 else np.inf
    fit_err = np.linalg.norm(A @ I - y) / max(1.0, np.linalg.norm(y))
    return I, {"cond": cond, "rel_fit_err": fit_err}


def make_quadratic_desired_field(sample_xy, q_goal, k=5.0):
    """
    F_des(q) = -k (q - q_goal)
    """
    return -k * (sample_xy - q_goal[None, :])


def mix_with_circulation(F_des, x_mix):
    """
    F_target = (1 - x) F_des + x R F_des, with R = [[0,-1],[1,0]]
    """
    R = np.array([[0.0, -1.0],
                  [1.0,  0.0]])
    F_rot = (F_des @ R.T)  # rotate each vector by +90°
    return (1.0 - x_mix) * F_des + x_mix * F_rot


def reconstruct_potential_on_grid(grid_xy, coil_xy, h, I, marble_moment=1.0, scale=1.0):
    """
    K_env(q) = - m * sum_i I_i * Bz_i(q); up to a constant factor for visualization.
    Returns K_env values at each grid point, shape (Npoints,).
    """
    K = np.zeros(grid_xy.shape[0])
    for Ii, cxy in zip(I, coil_xy):
        K -= marble_moment * Ii * dipole_Bz(grid_xy, cxy, h, scale=scale)
    return K


def synthesize_currents_weighted(sample_xy, F_target, coil_xy, h,
                                 q_goal, sigma=0.05,  # meters; adjust to focus region
                                 lam=1e-6,
                                 marble_moment=1.0, scale=1.0):
    """
    Weighted ridge LS with radial weights centered at q_goal.
    Weight per sample j: w_j = exp(-||q_j - q_goal||^2 / (2 sigma^2))
    Then duplicate for x/y rows.

    Returns I, diagnostics
    """
    A = build_basis_forces(sample_xy, coil_xy, h, marble_moment, scale)
    y = F_target.reshape(-1)  # (2J,)

    # compute per-sample weights then duplicate for x/y rows
    d = np.linalg.norm(sample_xy - q_goal[None, :], axis=1)
    w = np.exp(- (d**2) / (2.0 * (sigma**2) + 1e-12))  # (J,)
    W = np.repeat(w, 2)  # (2J,)

    # Form normal equations with weights
    # A^T W A  and  A^T W y
    # (no need to build diag(W) explicitly)
    AW = A * W[:, None]      # (2J,N), weighted rows
    ATA = A.T @ AW + lam * np.eye(A.shape[1])
    ATy = A.T @ (W * y)

    # Solve (prefer Cholesky if SciPy available)
    if _HAS_SCIPY_LINALG:
        c, low = cho_factor(ATA, overwrite_a=False, check_finite=False)
        I = cho_solve((c, low), ATy, check_finite=False)
    else:
        I = np.linalg.solve(ATA, ATy)

    # diagnostics
    s = np.linalg.svd(A, compute_uv=False)
    cond = (s[0] / s[-1]) if s[-1] > 0 else np.inf
    rel_fit = np.linalg.norm((A @ I - y) * np.sqrt(W)) / max(1.0, np.linalg.norm(y * np.sqrt(W)))
    return I, {"cond": cond, "rel_fit_err_w": rel_fit, "sigma": sigma}

def synthesize_currents_bounded(sample_xy, F_target, coil_xy, h,
                                lam=1e-6, Imax=1.0,
                                marble_moment=1.0, scale=1.0,
                                max_iter_pg=500, lr=1e-2):
    """
    Ridge LS with box constraints |I_i| <= Imax.
    If SciPy is available, uses lsq_linear on the augmented system.
    Otherwise, fallback to a projected gradient on the quadratic cost.

    Returns I, diagnostics
    """
    A = build_basis_forces(sample_xy, coil_xy, h, marble_moment, scale)
    y = F_target.reshape(-1)

    # Augment for ridge: minimize ||A I - y||^2 + lam ||I||^2
    # Equivalent to least squares on [A; sqrt(lam) I] I ≈ [y; 0]
    if lam > 0:
        sqrt_lam = np.sqrt(lam)
        A_aug = np.vstack([A, sqrt_lam * np.eye(A.shape[1])])
        y_aug = np.concatenate([y, np.zeros(A.shape[1])])
    else:
        A_aug, y_aug = A, y

    if _HAS_SCIPY_OPT:
        bounds = (-Imax * np.ones(A.shape[1]), Imax * np.ones(A.shape[1]))
        res = lsq_linear(A_aug, y_aug, bounds=bounds, lsmr_tol='auto', verbose=0, max_iter=500)
        I = res.x
        fit = np.linalg.norm(A @ I - y) / max(1.0, np.linalg.norm(y))
        return I, {"method": "lsq_linear", "rel_fit_err": fit, "status": res.status}
    else:
        # Projected gradient fallback
        N = A.shape[1]
        I = np.zeros(N)
        # Precompute ATA, ATy for speed
        ATA = A.T @ A + lam * np.eye(N)
        ATy = A.T @ y
        for _ in range(max_iter_pg):
            grad = (ATA @ I) - ATy        # gradient of 0.5*||A I - y||^2 + 0.5*lam||I||^2
            I = I - lr * grad             # gradient step
            I = np.clip(I, -Imax, Imax)   # projection onto box
        fit = np.linalg.norm(A @ I - y) / max(1.0, np.linalg.norm(y))
        return I, {"method": "proj_grad", "rel_fit_err": fit, "steps": max_iter_pg}