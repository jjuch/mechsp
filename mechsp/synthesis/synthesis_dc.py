# mechsp/synthesis/synthesis_dc.py
import numpy as np
from numpy.linalg import cond as dense_cond
from typing import Optional, Dict, Tuple

try:
    from scipy.optimize import lsq_linear, minimize, LinearConstraint, Bounds
    from scipy.sparse import csc_matrix, csr_matrix, vstack as sp_vstack, eye as sp_eye, diags as sp_diags, triu as sp_triu, issparse, tril as sp_tril
    _HAS_SCIPY_OPT = True
except Exception:
    _HAS_SCIPY_OPT = False

from ..magnetics import grad_Bz_analytic, hessian_Bz_analytic, dipole_Bz
from .synthesis import build_basis_forces

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

def solve_dc_currents_potsep(
    sample_xy: np.ndarray,
    F_des: np.ndarray,            # (J,2)
    coil_xy: np.ndarray,
    h: float,
    q_goal: np.ndarray,
    mu: float,
    *,
    m_ball: float = 1.0,
    lam: float = 1e-4,
    Imax: Optional[float] = None,
    scale: float = 1.0,
    r0: float = 0.0,              # exclude inner disk (m) from inequalities
    thin_factor: int = 1,         # keep every k-th inequality row (speed) - speed up by choosing 2 or 3
    solver: str = "osqp",         # 'osqp' (default) or 'highs'
    # OSQP-specific knobs (ignored by 'highs'):
    eps_abs: float = 1e-5,
    eps_rel: float = 1e-5,
    polish: bool = True,
    time_limit: Optional[float] = None,
    x0: Optional[np.ndarray] = None,   # warm-start currents
    verbose: Optional[bool] = True,
    logging: Optional[bool] = False,
) -> Tuple[np.ndarray, Dict]:

    """
    DC solve with POTENTIAL-SEPARATION inequalities (vectorized), using either:
      - OSQP (default)  -> fast ADMM QP solver with warm start & polishing
      - HiGHS ('highs') -> active-set / IPM QP via highspy

    Problem:
      minimize  0.5 * ||A I0 - y||^2 + 0.5 * lam * ||I0||^2
      subject to
         grad K(q_goal) = 0                (linear equalities)
         K(q) - K(q_goal) >= (mu/2)||q-q_goal||^2   (linear inequalities)
         |I0_i| <= Imax (optional box bounds)

    Returns:
      I0 : (N,) currents
      diag : dict with fit error and solver diagnostics
    """
    # ---------------------------
    # Setup
    # ---------------------------
    P = sample_xy            # (J,2)
    J = P.shape[0]
    N = coil_xy.shape[0]

    # Force regression matrix A (2J x N) and RHS y
    A = build_basis_forces(P, coil_xy, h, marble_moment=m_ball, scale=scale)  # (2J, N)
    y = F_des.reshape(-1)  # (2J,)
    print("Built basis forces ...")

    # Equality constraints at goal: grad K(qg) = 0  =>  sum I0_i * ∇b_i(qg) = 0
    # grad K = -m_b * sum I_i ∇b_i  => we impose sum I_i ∇b_i = 0  (equivalently scaling)
    G = m_ball * grad_Bz_analytic(q_goal, coil_xy, h, scale=scale)  # (N,2)
    A_eq = G.T.copy()                 # (2, N)
    b_eq = np.zeros(2, dtype=float)
    print("Built equality constraints...")

    # ---------------------------
    # Potential-separation INEQUALITIES (vectorized)
    # ---------------------------


    # B_all[j,i] = b_i(q_j), B_goal[i] = b_i(q_goal)
    B_all = dipole_Bz(P, coil_xy, h, scale=scale)           # (J, N)
    B_goal = dipole_Bz(q_goal, coil_xy, h, scale=scale)     # (N,)

    # A_ineq(j,:) = m_b * (B_j - B_g), b_ineq(j) = -0.5 * mu * ||q_j - q_g||^2
    A_ineq_full = m_ball * (B_all - B_goal[None, :])        # (J, N)
    d = P - q_goal[None, :]
    r2 = np.einsum('ij,ij->i', d, d)                        # (J,)
    b_ineq_full = -0.5 * mu * r2

    # Exclude inner disk r < r0 and thin constraints if requested
    mask = r2 > (r0 * r0)
    if thin_factor > 1:
        # Apply thinning only after masking to preserve near-goal coverage
        idx = np.flatnonzero(mask)[::thin_factor]
        A_ineq = A_ineq_full[idx, :]
        b_ineq = b_ineq_full[idx]
    else:
        A_ineq = A_ineq_full[mask, :]
        b_ineq = b_ineq_full[mask]

    if A_ineq.shape[0] == 0:
        raise RuntimeError("No valid inequality constraints — increase domain or reduce r0.")

    print("Built potential-separation inequalities...")

    # ---------------------------
    # Quadratic objective pieces
    # ---------------------------
    # 0.5 * ||A I0 - y||^2 + 0.5 * lam ||I0||^2  ==>
    # P = A^T A + lam I   (symmetric PSD),  q = -A^T y
    A_sp = csc_matrix(A)
    P_qp = (A_sp.T @ A_sp) + lam * sp_eye(N, format='csc')
    q_qp = -(A_sp.T @ csc_matrix(y).reshape((-1,1))).toarray().ravel()
    print("Built quadratic objective pieces...")
 
    # ---------------------------
    # Branch by solver
    # ---------------------------
    if solver.lower() == "osqp":    
        try:
            import osqp 
        except Exception as e:
            raise ImportError(
                "OSQP is required for solver='osqp'. Install with `pip install osqp`."
            ) from e

        # OSQP canonical:  minimize 0.5 x^T P x + q^T x,  s.t.  l <= A_osqp x <= u
        # Stack rows: [A_eq; A_ineq; (+I if bounds); (-I if bounds)]
        rows = []
        lvec = []
        uvec = []

        # Equalities: l=u=b_eq
        Aeq_sp = csc_matrix(A_eq)
        rows.append(Aeq_sp)
        lvec.append(b_eq)
        uvec.append(b_eq)

        # Inequalities: (-inf, b_ineq]
        Aineq_sp = csc_matrix(A_ineq)
        rows.append(Aineq_sp)
        lvec.append(-np.inf * np.ones_like(b_ineq))
        uvec.append(b_ineq)

        # Variable bounds (box): |I0_i| <= Imax  ->  -I x <= Imax  and  I x <= Imax
        if Imax is not None and np.isfinite(Imax):
            I_sp = sp_eye(N, format='csc')
            rows.append(I_sp)  #  I x <= Imax
            lvec.append(-np.inf * np.ones(N))
            uvec.append(Imax * np.ones(N))

            rows.append(-I_sp) # -I x <= Imax
            lvec.append(-np.inf * np.ones(N))
            uvec.append(Imax * np.ones(N))

        A_osqp = sp_vstack(rows).tocsc()
        l_osqp = np.concatenate(lvec)
        u_osqp = np.concatenate(uvec)

        # OSQP expects upper-triangular of P
        P_triu = csc_matrix(np.triu(P_qp.toarray()))

        prob = osqp.OSQP()
        prob.setup(
            P=P_triu, q=q_qp, A=A_osqp, l=l_osqp, u=u_osqp,
            eps_abs=eps_abs, eps_rel=eps_rel, polish=polish,
            max_iter=8000, verbose=False,
            time_limit=(float(time_limit) if time_limit is not None else 1e10)
        )

        if x0 is not None and x0.shape == (N,):
            prob.warm_start(x=x0)   # Warm start is effective for param sweeps.
        print("Solving...")
        res = prob.solve()
        print("Stopped Solving...")
        I0 = np.array(res.x, dtype=float).ravel()

        # Diagnostics
        fit_err = np.linalg.norm(A @ I0 - y) / max(1.0, np.linalg.norm(y))
        diag = dict(
            rel_fit_err=fit_err,
            status=res.info.status, # Solved, primal infeasible, dual infeasible, or maximum iterations reached (see OSQP documentation)
            obj_val=res.info.obj_val, # Final objective value
            prim_res=res.info.prim_res, # Primal residual norm: violation of constraints (should be small)
            dual_res=res.info.dual_res, # Dual residual norm: stationarity violation (should be small, mitigation: adaptive rho/more iterations/lower tolerance)
            niter=res.info.iter # Number of iterations (is > 2000: adjust mu/add constraint thinning/tune rho)
        )

        if verbose:
            summarise_dc_qp_diagnostics(
                solver="osqp",
                I0=I0,
                diag=diag,
                A_force=A, y_force=y,
                A_eq=A_eq, b_eq=b_eq,
                A_ineq=A_ineq, b_ineq=b_ineq,
                lam=lam,
                P_qp=P_qp,
            )

        return I0, diag
    
    elif solver.lower() == "highs":
    # HiGHS QP via highspy (active-set / IPM). QP form: min 0.5 x^T Q x + c^T x,  LB <= A x <= UB,  LBx <= x <= UBx.
        try:
            import highspy as highs
        except Exception as e:
            raise ImportError(
                "HiGHS Python API (highspy) is required for solver='highs'. Install with `pip install highspy`."
            ) from e

        # Build linear-constraint block: stack equalities and inequalities
        # Equalities as rows with lb=ub=b_eq; Inequalities as rows with lb=-inf, ub=b_ineq
        A_lin = sp_vstack([csc_matrix(A_eq), csc_matrix(A_ineq)]).tocsr()
        lbA = np.concatenate([b_eq, -np.inf * np.ones(A_ineq.shape[0])])
        ubA = np.concatenate([b_eq, b_ineq])

        # Variable bounds
        if Imax is None or not np.isfinite(Imax):
            lbx = -np.inf * np.ones(N)
            ubx =  np.inf * np.ones(N)
        else:
            lbx = -float(Imax) * np.ones(N)
            ubx =  float(Imax) * np.ones(N)

        # HiGHS model
        h = highs.Highs()
        h.setOptionValue("output_flag", logging)   # Start log through 'logging' flag
        if logging:
            h.setOptionValue("log_file", "highs.log")
        status = h.setOptionValue("user_objective_scale", -24)
        print("Status: ", status)

        # Convert to CSC for column-wise pass (HiGHS accepts standard sparse formats)
        A_csc = A_lin.tocsc()
        A_csc.sort_indices()

        # Linear costs and sense
        # Objective: 0.5 x^T P x + q^T x  (P = P_qp, q = q_qp)
        # Highs requires lower triangular of Q:
        Q_tril = sp_tril(P_qp, k=0, format='csc')
        Q_tril.sort_indices()

        # Build HighsModel
        model = highs.HighsModel()

        # Column (variable) data
        model.lp_.num_col_ = N
        model.lp_.col_cost_ = q_qp.astype(float).tolist()
        model.lp_.col_lower_ = lbx.astype(float).tolist()
        model.lp_.col_upper_ = ubx.astype(float).tolist()
        model.lp_.sense_ = highs.ObjSense.kMinimize # make the sense explicit

        # Row (constraint) data
        model.lp_.num_row_ = A_csc.shape[0]
        model.lp_.row_lower_ = lbA.astype(float).tolist()
        model.lp_.row_upper_ = ubA.astype(float).tolist()

        # Sparse A matrix in column-wise format
        model.lp_.a_matrix_.start_  = A_csc.indptr.astype(np.int64).tolist()
        model.lp_.a_matrix_.index_  = A_csc.indices.astype(np.int32).tolist()
        model.lp_.a_matrix_.value_  = A_csc.data.astype(float).tolist()
        model.lp_.a_matrix_.format_ = highs.MatrixFormat.kColwise

        # Quadratic (upper triangular) in CSC
        model.hessian_.dim_    = N
        model.hessian_.start_  = Q_tril.indptr.astype(np.int64).tolist()
        model.hessian_.index_  = Q_tril.indices.astype(np.int32).tolist()
        model.hessian_.value_  = Q_tril.data.astype(float).tolist()
        model.hessian_.format_ = highs.HessianFormat.kTriangular

        
    #     print("nnz(Q) =", Q_tril.nnz, "  nnz(A) =", A_csc.nnz,
    #   "  rows:", A_csc.shape[0], "  cols:", N)
    #     exit()


        h.passModel(model)     # Pass full QP model to HiGHS.
        print("Solving...")
        run_status = h.run()
        print("Done solving...")

        sol = h.getSolution()
        I0 = np.array(sol.col_value, dtype=float)
        
        # Diagnostics
        fit_err = np.linalg.norm(A @ I0 - y) / max(1.0, np.linalg.norm(y))
        info = h.getInfo()
        diag = dict(
            rel_fit_err=fit_err,
            status=info.primal_solution_status,  # Optimal / Infeasible / unbounded
            obj_val=info.objective_function_value, # Objective value
            niter=getattr(info, "qp_iteration_count", None)
        )

        
        info  = h.getInfo()
        sol   = h.getSolution()
        mstat = h.getModelStatus()   # <- model solution status (Optimal/Infeasible/…)

        print("ModelStatus           :", mstat)               # solution status
        print("Simplex iters         :", info.simplex_iteration_count)
        print("IPM iters             :", getattr(info, "ipm_iteration_count", None))
        print("QP iters              :", getattr(info, "qp_iteration_count", None))
        print("Objective value       :", info.objective_function_value)


        if verbose:
            summarise_dc_qp_diagnostics(
                solver="highs",
                I0=I0,
                diag=diag,
                A_force=A, y_force=y,
                A_eq=A_eq, b_eq=b_eq,
                A_ineq=A_ineq, b_ineq=b_ineq,
                lam=lam,
                P_qp=P_qp,
            )

        return I0, diag
    
    else:
        raise ValueError("Solver must be 'osqp' or 'highs'.")
    

def _as_dense(M):
    return M.toarray() if issparse(M) else np.asarray(M)


def summarise_dc_qp_diagnostics(
    solver: str,
    I0: np.ndarray,
    diag: dict,
    *,
    A_force: np.ndarray,          # (2J,N) build_basis_forces(...)
    y_force: np.ndarray,          # (2J,)
    A_eq: np.ndarray,             # (2,N)   equalities at goal
    b_eq: np.ndarray,             # (2,)
    A_ineq: np.ndarray,           # (M,N)   potential-sep rows after masking/thinning
    b_ineq: np.ndarray,           # (M,)
    lam: float,
    print_matrix_cond: bool = True,
    P_qp: np.ndarray = None       # optional: pass P = A^T A + lam I if already built
):
    """
    Prints a standardised summary:
      - solver name & status
      - objective value (if available), iterations (if available)
      - force fit error (rel)
      - equality and inequality residual checks
      - primal/dual residuals for OSQP
      - condition numbers cond(A) and cond(P)
    """
    N = I0.size
    solver = solver.lower()

    # --- Fit quality (force LS residual)
    r_force = A_force @ I0 - y_force
    rel_fit = np.linalg.norm(r_force) / max(1.0, np.linalg.norm(y_force))

    # --- Equality residual at goal
    eq_res = A_eq @ I0 - b_eq
    eq_res_norm = np.linalg.norm(eq_res, ord=2)

    # --- Inequality slack (want A_ineq x <= b_ineq)
    if A_ineq is not None and A_ineq.size > 0:
        ineq_slack = b_ineq - (A_ineq @ I0)
        worst_viol = float(np.maximum(0.0, -ineq_slack).max())  # positive means violation
        min_slack  = float(ineq_slack.min())                    # most active constraint (≤0 is active)
    else:
        worst_viol = 0.0
        min_slack = np.nan

    # --- Condition numbers
    cond_A = None
    cond_P = None
    if print_matrix_cond:
        A = _as_dense(A_force)
        # NB: A can be tall; cond(A) is on 2-norm; for large sizes this is a costly dense op.
        # For speed: you can estimate via randomized SVD if needed.
        try:
            cond_A = float(dense_cond(A))
        except Exception:
            cond_A = None

        if P_qp is None:
            # P = A^T A + lam I
            P_qp = (A.T @ A) + lam * np.eye(N)
        else:
            P_qp = _as_dense(P_qp)

        try:
            cond_P = float(dense_cond(P_qp))
        except Exception:
            cond_P = None

    # --- Print
    line = "-" * 62
    print(line)
    print(f"DC QP Diagnostics  |  solver={solver}")
    print(line)

    # Common fields
    if "status" in diag:
        print(f"status       : {diag['status']}")
    if "obj_val" in diag:
        print(f"objective    : {diag['obj_val']:.6e}")
    if "niter" in diag and diag["niter"] is not None:
        print(f"iterations   : {diag['niter']}")

    print(f"rel_fit_err  : {rel_fit:.3e}   (||A I0 - y|| / ||y||)")

    # Goal equalities & inequalities
    print(f"eq_res_norm  : {eq_res_norm:.3e}   (||A_eq I0 - b_eq||_2)")
    print(f"ineq_max_vio : {worst_viol:.3e}   (max(0, A_ineq I0 - b_ineq))")
    print(f"ineq_min_slk : {min_slack:.3e}   (min(b_ineq - A_ineq I0))")

    
    # Solver-specific extras
    if solver == "osqp":
        # OSQP uses primal/dual residuals & polishing. 
        pri_res = diag.get("prim_res", None)
        dua_res = diag.get("dual_res", None)
        if pri_res is not None:
            print(f"prim_res      : {pri_res:.3e}   (constraint residual)")
        if dua_res is not None:
            print(f"dual_res      : {dua_res:.3e}   (stationarity residual)")

    # Conditioning metrics
    if print_matrix_cond:
        print("---- conditioning (for context) ----")
        if cond_A is not None:
            print(f"cond(A)      : {cond_A:.3e}")
        else:
            print("cond(A)      : (not computed)")
        if cond_P is not None:
            print(f"cond(P)      : {cond_P:.3e}   (P = A^T A + lam I)")
        else:
            print("cond(P)      : (not computed)")
    
    print(line)