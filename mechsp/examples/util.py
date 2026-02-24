# mechsp/examples/util.py
import os, json, hashlib
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.sparse import csc_matrix
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

from mechsp.magnetics import dipole_Bz, grad_Bz_analytic
from mechsp.synthesis.synthesis import build_basis_forces


# ---------------------------------------------------------------------
# 0) Small helpers
# ---------------------------------------------------------------------
def _sha256_bytes(arr: np.ndarray, ndigits: int = 9) -> str:
    """Stable numeric digest of an array (rounded to reduce floating noise)."""
    if arr is None:
        return "none"
    a = np.asarray(arr, dtype=float)
    a = np.round(a, decimals=ndigits)
    return hashlib.sha256(a.tobytes()).hexdigest()

def _sha256_dict(d: Dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode("utf-8")).hexdigest()

def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True); return 
    

# ---------------------------------------------------------------------
# 1) I0 cache: load / verify / save
# ---------------------------------------------------------------------
@dataclass
class I0CacheMeta:
    key: str
    rel_fit_err: float
    params: Dict

def _cache_path(cache_dir: str, key: str) -> str:
    return os.path.join(cache_dir, f"I0_{key}.npz")

def save_I0_cache(cache_dir: str, key: str, I0: np.ndarray, meta: I0CacheMeta):
    ensure_dir(cache_dir)
    np.savez(_cache_path(cache_dir, key), I0=I0, rel_fit_err=meta.rel_fit_err, params=json.dumps(meta.params))


def load_I0_cache(cache_dir: str, key: str) -> Optional[Tuple[np.ndarray, I0CacheMeta]]:
    path = _cache_path(cache_dir, key)
    if not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=True)
    I0 = data["I0"]
    rel_fit_err = float(data["rel_fit_err"])
    params = json.loads(str(data["params"]))
    meta = I0CacheMeta(key=key, rel_fit_err=rel_fit_err, params=params)
    return I0, meta


def compute_rel_fit_err(I0: np.ndarray, sample_xy: np.ndarray, 
                        F_des: np.ndarray, coil_xy: np.ndarray, h: float, m_ball: float, scale: float = 1.0) -> float:
    """Compute relative LS fit error ||A I0 - y|| / ||y|| for given I0."""
    A = build_basis_forces(sample_xy, coil_xy, h, marble_moment=m_ball, scale=scale)  # (2J,N)
    y = F_des.reshape(-1)
    num = np.linalg.norm(A @ I0 - y)
    den = max(1.0, np.linalg.norm(y))
    return float(num / den)


def make_I0_cache_key(coil_xy: np.ndarray, h: float, L: float, 
                      sample_xy: np.ndarray, q_goal: np.ndarray, k: float, mu: float, lam: float, Imax: Optional[float], r0: float, thin_factor: int, m_ball: float, scale: float,
                      solver: str = "osqp") -> str:
    # Encode only what affects the DC bowl (K_eff). We’re not changing K anymore.
    d = dict(
        coil_xy=_sha256_bytes(coil_xy),
        h=round(float(h), 9),
        L=round(float(L), 9),
        S=_sha256_bytes(sample_xy),          # grid geometry
        q_goal=_sha256_bytes(q_goal),
        k=round(float(k), 9),                # desired linear stiffness
        mu=round(float(mu), 9), lam=round(float(lam), 12),
        Imax=None if Imax is None else round(float(Imax), 9),
        r0=round(float(r0), 9), thin=thin_factor,
        m_ball=round(float(m_ball), 9), scale=round(float(scale), 9),
        solver=solver.lower(),
    )
    return _sha256_dict(d)


# ---------------------------------------------------------------------
# 2) Fit scalar target B1(q) to coils for HF inertia shaping (simple LS)
# ---------------------------------------------------------------------
def fit_scalar_field_to_coils(sample_xy: np.ndarray, B_target: np.ndarray,
                              coil_xy: np.ndarray, h: float, scale: float = 1.0,
                              lam_ridge: float = 1e-6) -> np.ndarray:
    """
    Solve min ||B(sample_xy)*w - B_target||^2 + lam ||w||^2
    where B(q) = [b_1(q) ... b_N(q)] with b_i = dipole_Bz at coil i.
    Returns w (N,)
    """
    Bmat = dipole_Bz(sample_xy, coil_xy, h, scale=scale)  # (J,N)
    # normal equation: (B^T B + lam I) w = B^T y
    if _HAS_SCIPY:
        BT = csc_matrix(Bmat).T
        P = (BT @ csc_matrix(Bmat)) + lam_ridge * csc_matrix(np.eye(coil_xy.shape[0]))
        q = (BT @ csc_matrix(B_target).reshape((-1, 1))).toarray().ravel()
        w = np.linalg.solve(P.toarray(), q)
    else:
        P = (Bmat.T @ Bmat) + lam_ridge * np.eye(coil_xy.shape[0])
        q = Bmat.T @ B_target
        w = np.linalg.solve(P, q)
    return w


# ---------------------------------------------------------------------
# 3) Grid generation & plotting utilities
# ---------------------------------------------------------------------
def make_grid(L: float, Jside: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(-L/2, L/2, Jside)
    ys = np.linspace(-L/2, L/2, Jside)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    S = np.stack([X.ravel(), Y.ravel()], axis=1)
    return X, Y, S

def plot_scalar_map(X, Y, Z, title="", cmap="viridis", cbar_label="", ax=None):
    ax = ax if ax is not None else plt.gca()
    im = ax.imshow(Z, origin='lower',
                   extent=[X.min(), X.max(), Y.min(), Y.max()],
                   cmap=cmap, aspect='equal')
    ax.set_title(title); ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    cb = plt.colorbar(im, ax=ax); cb.set_label(cbar_label)
    return ax



# ---------------------------------------------------------------------
# 4) Visualize desired & realized M_eff and N_eff
# ---------------------------------------------------------------------
def compute_Meff_maps(eff_model, X, Y) -> Dict[str, np.ndarray]:
    """
    Returns dict of maps on (J,J):
      lam_min, lam_max, anisotropy = lam_max/lam_min
    """
    JY, JX = Y.shape
    lam_min = np.zeros_like(X)
    lam_max = np.zeros_like(X)
    for j in range(JY):
        for i in range(JX):
            q = np.array([X[j, i], Y[j, i]])
            M = eff_model.M_eff(q)
            vals = np.linalg.eigvalsh(M)
            lam_min[j, i] = vals[0]
            lam_max[j, i] = vals[1]
    aniso = np.divide(lam_max, lam_min, out=np.ones_like(lam_max), where=lam_min>1e-16)
    return dict(lam_min=lam_min, lam_max=lam_max, anisotropy=aniso)


def compute_Neff_maps(eff_model, X, Y) -> Dict[str, np.ndarray]:
    """
    For each (x,y), compute a scalar 'gamma' magnitude of N_eff.
    For a 2x2 skew matrix ~ gamma*J, gamma ≈ N[1,0] (or -N[0,1]).
    Also return |N|_F for completeness.
    """
    JY, JX = Y.shape
    gamma = np.zeros_like(X)
    fnorm = np.zeros_like(X)
    for j in range(JY):
        for i in range(JX):
            q = np.array([X[j, i], Y[j, i]])
            N = eff_model.N_eff(q, np.array([1.0, 0.0]))  # v is unused in your implementation
            gamma[j, i] = abs(N[1, 0])
            fnorm[j, i] = np.sqrt(np.sum(N*N))
    return dict(gamma=gamma, fnorm=fnorm)


def plot_Meff_suite(eff_model, L: float, Jside: int = 60, suptitle: str = "M_eff maps"):
    X, Y, _ = make_grid(L, Jside)
    mm = compute_Meff_maps(eff_model, X, Y)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.5))
    plot_scalar_map(X, Y, mm['lam_min'], "λ_min(M_eff)", "viridis", "λ_min", ax=axs[0])
    plot_scalar_map(X, Y, mm['lam_max'], "λ_max(M_eff)", "viridis", "λ_max", ax=axs[1])
    plot_scalar_map(X, Y, mm['anisotropy'], "Anisotropy (λ_max/λ_min)", "magma", "ratio", ax=axs[2])
    fig.suptitle(suptitle); plt.tight_layout()
    return fig, axs


def plot_Neff_suite(eff_model, L: float, Jside: int = 60, suptitle: str = "N_eff maps"):
    X, Y, _ = make_grid(L, Jside)
    nm = compute_Neff_maps(eff_model, X, Y)
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.5))
    plot_scalar_map(X, Y, nm['gamma'], "|gamma(q)| from N_eff", "plasma", "|γ|", ax=axs[0])
    plot_scalar_map(X, Y, nm['fnorm'], "||N_eff||_F", "cividis", "Frobenius norm", ax=axs[1])
    fig.suptitle(suptitle); plt.tight_layout()
    return fig, axs


# ---------------------------------------------------------------------
# 5) Convenience: compute potential proxy field V_k,eff on grid
# ---------------------------------------------------------------------
def compute_Vk_eff_grid(design, m_ball: float, q_goal: np.ndarray, L: float, Jside: int = 80):
    """
    Potential-energy proxy field
       V_k,eff(X,Y) = 0.5 * <∇K_eff(q), q - q_goal>
    on a Jside x Jside grid (diagnostic / visualization).
    """
    X, Y, S = make_grid(L, Jside)
    G_all = grad_Bz_analytic(S, design.coil_xy, design.h, scale=design.scale)  # (J,N,2)
    I0 = design.I0.reshape(1, -1, 1)
    gradK = -m_ball * np.sum(I0 * G_all, axis=1)  # (J,2)
    dq = S - q_goal[None, :]
    V_flat = 0.5 * np.einsum('ij,ij->i', gradK, dq)
    return X, Y, V_flat.reshape(Jside, Jside)

    
# ---------------------------------------------------------------------
# 6) Caching wrapper: compute or reuse I0
# ---------------------------------------------------------------------
def compute_I0_with_cache(cache_dir: str,
                          sample_xy: np.ndarray, F_des: np.ndarray,
                          coil_xy: np.ndarray, h: float, L: float,
                          q_goal: np.ndarray, k: float, mu: float,
                          m_ball: float, lam: float, Imax: Optional[float],
                          scale: float, r0: float, thin_factor: int,
                          solve_dc_fn, *, solver: str = "osqp",
                          eps_abs: float = 1e-5, eps_rel: float = 1e-5, polish: bool = True,
                          time_limit: Optional[float] = None,
                          warm_start: Optional[np.ndarray] = None,
                          verbose: bool = True, tol_match: float = 5e-4):
    """
    1) Build key from setup parameters
    2) Try to load I0 from cache
    3) Verify rel_fit_err against current (A,y); if within tol, reuse
    4) Else compute via solve_dc_fn(...), save to cache
    """
    key = make_I0_cache_key(coil_xy, h, L, sample_xy, q_goal, k, mu, lam, Imax, r0, thin_factor, m_ball, scale, solver)
    loaded = load_I0_cache(cache_dir, key)
    if loaded is not None:
        I0_cached, meta = loaded
        rel_now = compute_rel_fit_err(I0_cached, sample_xy, F_des, coil_xy, h, m_ball, scale)
        if verbose:
            print(f"[I0 cache] key={key[:10]}..., cached_rel={meta.rel_fit_err:.3e}, now_rel={rel_now:.3e}")
        if abs(rel_now - meta.rel_fit_err) <= tol_match:
            if verbose: print("[I0 cache] Using cached I0 ✓")
            return I0_cached, dict(rel_fit_err=rel_now, cached=True, key=key)

        if verbose:
            print("[I0 cache] Cache mismatch; recomputing currents…")

    # Compute with provided solver function
    I0, diag = solve_dc_fn(
        sample_xy=sample_xy, F_des=F_des,
        coil_xy=coil_xy, h=h,
        q_goal=q_goal, mu=mu,
        m_ball=m_ball, lam=lam, Imax=Imax, scale=scale,
        r0=r0, thin_factor=thin_factor,
        solver=solver,
        eps_abs=eps_abs, eps_rel=eps_rel, polish=polish,
        time_limit=time_limit, x0=warm_start,
        verbose=verbose
    )
    meta = I0CacheMeta(key=key, rel_fit_err=float(diag.get("rel_fit_err", np.nan)),
                       params=dict(k=k, mu=mu, lam=lam, Imax=Imax, r0=r0, thin=thin_factor))
    save_I0_cache(cache_dir, key, I0, meta)
    if verbose:
        print(f"[I0 cache] Saved I0 (key={key[:10]}...), rel_fit_err={meta.rel_fit_err:.3e}")
    return I0, dict(rel_fit_err=meta.rel_fit_err, cached=False, key=key)


# ---------------------------------------------------------------------
# 7) Build a local SPD anisotropic ΔM target on a stencil
# ---------------------------------------------------------------------

def build_deltaM_target(
    q_center: np.ndarray,
    *,
    peak_ratio: float,      # desired peak ΔM ≈ peak_ratio * m_ball
    m_ball: float,
    sigma_x: float,         # Gaussian std in x
    sigma_y: float,         # Gaussian std in y
    anisotropy: float = 2.0,# ΔM principal ratio λx/λy (>1)
    grid_halfwidth: float = 0.03,
    step: float = 0.004,
    freeze_goal: Optional[np.ndarray] = None,
    freeze_r: float = 0.012,
    avoid_center: Optional[np.ndarray] = None,
    avoid_r_in: float = 0.0,
    avoid_r_out: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      S_hf:   (J,2) local stencil points around q_center
      ΔM_des: (J,2,2) SPD anisotropic target (principal axes aligned with world x/y)

    The footprint is Gaussian in space and tapered:
      - zero near the goal (if freeze_goal provided),
      - zero inside a moat around an obstacle 'avoid_center'.
    """
    # Local grid
    xs = np.arange(-grid_halfwidth, grid_halfwidth + 1e-12, step)
    ys = np.arange(-grid_halfwidth, grid_halfwidth + 1e-12, step)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    S = np.stack([X.ravel(), Y.ravel()], axis=1) + q_center[None, :]

    # Gaussian footprint
    invSig = np.diag([1.0/(sigma_x**2 + 1e-18), 1.0/(sigma_y**2 + 1e-18)])
    d = S - q_center[None, :]
    e = np.einsum('ij,jk,ik->i', d, invSig, d)
    g = np.exp(-0.5 * e)  # (J,)

    # Tapers
    if freeze_goal is not None:
        dg = np.linalg.norm(S - freeze_goal[None, :], axis=1)
        # 0 near goal within freeze_r, rise to 1 by ~1.5*freeze_r
        u = np.clip((dg - freeze_r) / max(1e-12, 0.5*freeze_r), 0.0, 1.0)
        s_goal = 10*u**3 - 15*u**4 + 6*u**5
        g *= s_goal

    if avoid_center is not None and avoid_r_out > avoid_r_in:
        da = np.linalg.norm(S - avoid_center[None, :], axis=1) 
        # 0 inside r_in, rise to 1 by r_out
        u = np.clip((da - avoid_r_in) / max(1e-12, (avoid_r_out - avoid_r_in)), 0.0, 1.0)
        s_avoid = 10*u**3 - 15*u**4 + 6*u**5
        g *= s_avoid

    # Anisotropic SPD principal values (aligned with x,y)
    lam_y = 1.0
    lam_x = anisotropy * lam_y
    peak = peak_ratio * m_ball

    J = S.shape[0]
    DeltaM = np.zeros((J, 2, 2))
    for j in range(J):
        A = g[j] * peak * np.diag([lam_x, lam_y])  # SPD 2x2
        DeltaM[j, :, :] = A
    return S, DeltaM
