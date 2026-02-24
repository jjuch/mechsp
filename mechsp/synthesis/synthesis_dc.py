# mechsp/synthesis/synthesis_dc.py
import numpy as np
from numpy.linalg import cond as dense_cond
from typing import Optional, Dict, Tuple

try:
    from scipy.sparse import csc_matrix, vstack as sp_vstack, eye as sp_eye, issparse, tril as sp_tril
    _HAS_SCIPY_OPT = True
except Exception:
    _HAS_SCIPY_OPT = False

from ..magnetics import grad_Bz_analytic, dipole_Bz
from .synthesis import build_basis_forces


def solve_dc_currents(
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


def approx_Hess_gradK(eff_model, q, h=None):
    """
    Central-difference Hessian of gradK at q (2D).
    Returns a (2,2) array H with H[i,j] = ∂ gradK_i / ∂ q_j.
    """
    q = np.asarray(q, dtype=float)
    if h is None:
        # step ~ sqrt(machine_eps) * (1 + |q|)   (rule-of-thumb)
        h = 1e-6 * (1.0 + np.linalg.norm(q))
        h = max(h, 1e-7)

    H = np.zeros((2, 2), dtype=float)
    e1 = np.array([1.0, 0.0])
    e2 = np.array([0.0, 1.0])

    # column 0: ∂/∂x of gradK
    g_plus  = eff_model.gradK(q + h*e1)
    g_minus = eff_model.gradK(q - h*e1)
    H[:, 0] = (g_plus - g_minus) / (2.0*h)

    # column 1: ∂/∂y of gradK
    g_plus  = eff_model.gradK(q + h*e2)
    g_minus = eff_model.gradK(q - h*e2)
    H[:, 1] = (g_plus - g_minus) / (2.0*h)

    return H
