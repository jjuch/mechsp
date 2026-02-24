# mechsp/examples/effective_minimal_demo.py
"""
Minimal demonstration of the averaged (effective) model.

Steps:
 1) Build a coil grid (n x m)
 2) Synthesize DC currents for K_eff (quadratic well to a goal)
 3) Optional local M_eff bump via HF (B1 fit)
 4) Optional short N_eff steering arc
 5) Build FieldDesign and effective model
 6) Integrate averaged dynamics

Run:
  python -m mechsp.examples.effective_minimal_demo
"""

import numpy as np
import sys, os
import matplotlib.pyplot as plt
try:
    from scipy.integrate import solve_ivp
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

from mechsp.synthesis.synthesis_dc import solve_dc_currents, approx_Hess_gradK
from mechsp.synthesis.synthesis import build_basis_forces
from mechsp.synthesis.synthesis_rot import solve_rotating_phasors
from mechsp.synthesis.synthesis_hf import solve_hf_modulation
from mechsp.fields import FieldDesign
from mechsp.eff_dynamics import build_eff_model
from mechsp.magnetics import grad_Bz_analytic, dipole_Bz

from mechsp.examples.util import (
    make_grid, compute_Vk_eff_grid, plot_scalar_map,
    compute_Meff_maps, plot_Meff_suite,
    compute_Neff_maps, plot_Neff_suite,
    fit_scalar_field_to_coils, compute_I0_with_cache, build_deltaM_target
)

# ----------------------------
# Experiment toggles
# ----------------------------
USE_I0_CACHE = True       # use/load a local cache for I0 (recompute only if fit changed)
USE_M = True             # add M_eff bump (anisotropic inertia)
USE_N = True             # add N_eff steering arc
PLOT_M_DESIRED = True     # visualize desired M bump footprint
PLOT_N_DESIRED = True     # visualize desired N window
PLOT_M_REALIZED = True    # maps of realized M_eff
PLOT_N_REALIZED = True    # maps of realized N_eff
RUN_TRUE_SIM = True # Run the ground-truth with prescribed M, N, K: without magnetic approximations


# ----------------------------
# Scene setup (domain + coils)
# ----------------------------
def make_coils(L=0.20, n=25, m=25, h=0.02):
    """
    Square domain [-L/2, L/2]^2, coil plane at z=-h.
    Returns:
      coil_xy (N,2), h, and a helper to generate sample grids.
    """
    dx = L / (n + 1)
    dy = L / (m + 1)
    xs = (np.arange(n) + 1) * dx - L/2
    ys = (np.arange(m) + 1) * dy - L/2
    XX, YY = np.meshgrid(xs, ys, indexing='xy')
    coil_xy = np.stack([XX.ravel(), YY.ravel()], axis=1)
    return coil_xy, h, L


def sample_domain(L, Jside=100):
    xs = np.linspace(-L/2, L/2, Jside)
    ys = np.linspace(-L/2, L/2, Jside)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    S = np.stack([X.ravel(), Y.ravel()], axis=1)
    return S, (X, Y)


# ----------------------------
# Steering windows for N_eff
# ----------------------------
def smooth_step(u): 
    """
    C^k smooth step (process reaction curve) from 0 to 1:
    u in [0,1] -> sigma(u) in [0,1].
    """
    u = np.clip(u, 0.0, 1.0)
    return 10*u**3 - 15*u**4 + 6*u**5  # C^2

def phase_taper(r, r0, r1):
    """
    Taper s(r) of smooth_step that equals 0 for r<=r0, rises smoothly to 1 by r>=r1.
    """
    return smooth_step((r - r0) / max(1e-12, (r1 - r0)))


def window_arc(q, q_obs, r_in=0.02, r_out=0.05, angle_center=-np.pi/2, angle_halfwidth=np.pi/6):
    """
    Spatial window ~1 on an arc 'south' of the obstacle and ~0 elsewhere.
    angle_center = -pi/2 points downward (+y negative)
    """
    d = q - q_obs
    r = np.linalg.norm(d)
    if r < 1e-12: return 0.0
    ang = np.arctan2(d[1], d[0])  # (-pi, pi]
    # radial window
    wr = phase_taper(r, r_in, r_out) * (1 - phase_taper(r, r_out, r_out + 0.01))
    # angular window (cosine lobe)
    dang = np.arctan2(np.sin(ang - angle_center), np.cos(ang - angle_center))
    wa = np.clip(1.0 - (abs(dang) / angle_halfwidth), 0.0, 1.0)
    return float(wr * wa)


# ----------------------------
# Main
# ----------------------------
def main():
    # --- geometry & params ---
    coil_xy, h, L = make_coils(L=0.20, n=25, m=25, h=0.02)
    m_ball = 0.015  # kg
    q_goal = np.array([0.06, -0.03])
    q0 = np.array([-0.07, 0.06])
    v0 = np.zeros(2)
    Jside = 100 # samples in x and y direction

    # Obstacle (for windowing visual intuition; not simulated as barrier here)
    q_obs = np.array([0.0, 0.02])  # put “north” of goal for the swoop
    r_obs = 0.015
    
    # --- DC synthesis for K_eff (linear spring toward goal)---
    S, (X, Y) = sample_domain(L, Jside=Jside)
    k = 1.0 # N/m
    F_des = -k * (S - q_goal[None, :])  # Desired force
    
    # I0 cache directory
    cache_dir = os.path.join(os.path.dirname(__file__), ".cache_i0")

    if USE_I0_CACHE:
        I0, diag_dc = compute_I0_with_cache(
            cache_dir=cache_dir,
            sample_xy=S, F_des=F_des, coil_xy=coil_xy, h=h, L=L,
            q_goal=q_goal, k=k, mu=0.5*k, m_ball=m_ball, lam=1e-4, Imax=1.0,
            scale=1.0, r0=0.015, thin_factor=1,
            solve_dc_fn=solve_dc_currents, solver="osqp", eps_abs=1e-5, eps_rel=1e-5, polish=True,
            time_limit=None, warm_start=None, verbose=True, tol_match=5e-4            
        )
    else:
        I0, diag_dc = solve_dc_currents(
            sample_xy=S, F_des=F_des, coil_xy=coil_xy, h=h,
            q_goal=q_goal, mu=0.5*k, m_ball=m_ball, lam=1e-4, Imax=1.0,
            scale=1.0, r0=0.015, thin_factor=1,
            solver="osqp", eps_abs=1e-5, eps_rel=1e-5, polish=True,
            time_limit=None, x0=None, verbose=True
        )

    print(f"[DC] rel_fit={diag_dc['rel_fit_err']:.3e}  (cached={diag_dc.get('cached', False)})")
    
    
    # --- Desired M bump via HF ---
    # Averaging leverage: ΔM ∝ (eps^2)/(2*Omega^2). Use parameters that make the bump visible while keeping a clear scale separation.
    Ihf_amp = np.zeros_like(I0)
    Ihf_phase = np.zeros_like(I0)
    Omega = 2*np.pi*120.0 # (>> 10 Hz slow dynamics)
    eps = 0.70 # Sizeable HF amplitude
    c_mass = (eps**2)/(2.0*Omega**2)
    print(f"[HF] c_mass = eps^2 / (2 Omega^2) = {c_mass:.3e}")

    if USE_M:
        # Local SPD bump above obstacle, principal axes aligned with x/y, λx > λy
        q_m = q_obs + np.array([0.0, 0.015]) # Center of bump above obstacle
        S_hf, DeltaM_des = build_deltaM_target(
            q_center=q_m, peak_ratio=0.12, m_ball=m_ball,
            sigma_x=0.020, sigma_y=0.008, anisotropy=2.5,
            grid_halfwidth=0.03, step=0.004,
            freeze_goal=q_goal, freeze_r=0.012,
            avoid_center=q_obs, avoid_r_in=r_obs+0.004, avoid_r_out=r_obs+0.012
        )

        Ihf_amp, Ihf_phase, diag_hf = solve_hf_modulation(
            sample_xy=S_hf,
            DeltaM_des=DeltaM_des, #(J, 2, 2) SPD targets
            coil_xy=coil_xy, h=h,
            lam_H=1e-3, scale=1.0
        )
        print(f"[HF] rel_fit={diag_hf.get('rel_fit_err',np.nan):.3e}")

        if PLOT_M_DESIRED:
            Xd, Yd, Sd = make_grid(L, 120)
            
            def trace_deltaM(P):
                # interpolate coarse target — for visualization only
                d = np.linalg.norm(P - q_m[None,:], axis=1)
                g = np.exp(-0.5*((P[:,0]-q_m[0])**2/(0.020**2) + (P[:,1]-q_m[1])**2/(0.008**2)))
                # rough trace profile (scaled)
                return 0.12*m_ball * (1 + (2.5-1))*0.5 * g
            Tr = trace_deltaM(Sd).reshape(120,120)

            plt.figure(figsize=(5,4))
            plot_scalar_map(Xd, Yd, Tr, title="Desired ΔM footprint (trace, approx.)", cbar_label="tr(ΔM)")
            plt.scatter([q_obs[0]], [q_obs[1]], s=60, c='r', marker='x', label='obstacle')
            plt.scatter([q0[0]], [q0[1]], s=60, c='b', marker='x', label='start')
            plt.scatter([q_goal[0]], [q_goal[1]], s=60, c='k', marker='x', label='goal')
            plt.legend(); plt.tight_layout()

            
    # --- Desired N arc  ---
    # Phase/amp taper: freeze near goal; strong only on a small arc “south” of obstacle
    omega  = 2*np.pi*10.0
    kappa  = 4.0 * omega   # small lag: sin(delta)=omega/kappa; delta ≈ 14°
    Irot_amp   = np.zeros_like(I0)
    Irot_phase = np.zeros_like(I0)

    if USE_N:   
        # Build amplitude only on a short arc “south” of the obstacle
        S_rot = S[::4]
        B0_des = np.array([window_arc(p, q_obs, r_in=r_obs+0.010, r_out=r_obs+0.040, angle_center=-np.pi/2, angle_halfwidth=np.pi/5) for p in S_rot])
        # Phase map: freeze near goal (∇φ=0) and follow the angular coordinate elsewhere
        d_rot = S_rot - q_goal[None, :]
        phi_raw = np.arctan2(d_rot[:, 1], d_rot[:, 0]) # [-pi, pi]
        r_goal = np.linalg.norm(d_rot, axis=1)
        s_phase = phase_taper(r_goal, 0.010, 0.018)
        phi_des = s_phase * phi_raw

        Irot_amp, Irot_phase, diag_rot = solve_rotating_phasors(
            sample_xy=S_rot,
            B0_des=B0_des, phi_des=phi_des,
            coil_xy=coil_xy, h=h,
            lam=1e-3, scale=1.0
        )
        print(f"[ROT] rel_fit={diag_rot.get('rel_fit_err',np.nan):.3e}")

        if PLOT_N_DESIRED:
            Xw, Yw, Sw = make_grid(L, 120)
            Wmap = np.array([window_arc(p, q_obs, r_in=r_obs+0.010, r_out=r_obs+0.040, angle_center=-np.pi/2, angle_halfwidth=np.pi/5) for p in Sw]).reshape(120, 120)
            plt.figure(figsize=(5, 4))
            plot_scalar_map(Xw, Yw, Wmap, title="N_eff steering window (desired)", cbar_label="window")
            plt.scatter([q_obs[0]], [q_obs[1]], s=60, c='r', marker='x', label='obstacle')
            plt.scatter([q0[0]], [q0[1]], s=60, c='b', marker='x', label='start')
            plt.scatter([q_goal[0]], [q_goal[1]], s=60, c='k', marker='x', label='goal')
            plt.legend(); plt.tight_layout()

    if RUN_TRUE_SIM:
        I2 = np.eye(2)
        # Potential energy force
        def gradK_true(q):
            return k * (q - q_goal)
        
        # Inertia force
        q_m_true = q_obs + np.array([0.0, 0.015])
        sigx, sigy = 0.020, 0.008
        lam_ratio = 2.5 # λx / λy
        peak_ratio = 0.12 # ΔM_peak ≈ peak_ratio * m_ball

        def DeltaM_true(q):
            d = q - q_m_true
            g = np.exp(-0.5 * ((d[0]/sigx)**2 + (d[1]/sigy)**2))
            # zero near goal & inside obstacle moat
            s_goal  = 1.0 - phase_taper(np.linalg.norm(q - q_goal), 0.010, 0.018)
            s_avoid = phase_taper(np.linalg.norm(q - q_obs ), r_obs+0.004, r_obs+0.012)
            g *= s_goal * s_avoid
            lam_y = 1.0
            lam_x = lam_ratio * lam_y
            return (peak_ratio * m_ball) * np.diag([lam_x, lam_y]) * g
        
        def M_true(q): return m_ball*I2 + DeltaM_true(q)

        # Velocity forces: γ(q) J on a short arc (south)
        def gamma_true(q):
            rwin = phase_taper(np.linalg.norm(q - q_obs), r_obs+0.010, r_obs+0.040) * (1 - phase_taper(np.linalg.norm(q - q_obs), r_obs+0.040, r_obs+0.050))
            # angular lobe around -pi/2
            ang = np.arctan2((q - q_obs)[1], (q - q_obs)[0])
            dang = np.arctan2(np.sin(ang + np.pi/2), np.cos(ang + np.pi/2))
            awin = np.clip(1 - np.abs(dang)/(np.pi/5), 0.0, 1.0)
            # small magnitude; tuned so bend is visible but not dominant
            return 2.0 * rwin * awin   # [1/s], scale as needed

        Jskew = np.array([[0.0, -1.0],[1.0, 0.0]])
        def N_true(q): return gamma_true(q) * Jskew

        c_damp = 0.07
        # --- RHS (true)
        def true_rhs(t, y):
            q = y[0:2]; v = y[2:4]
            M = M_true(q)
            rhs = - gradK_true(q) - c_damp * v - N_true(q) @ v
            a = np.linalg.solve(M, rhs)
            return np.array([v[0], v[1], a[0], a[1]])

        def true_jac_quasi(t, y):
            q = y[0:2]; v = y[2:4]
            # same quasi-Newton structure, with Hessian of K_true = k I
            A_q = - np.linalg.inv(M_true(q)) @ (k * I2)  # since H_K_true = k I
            
            A_v =   np.linalg.inv(M_true(q)) @ (-c_damp*I2 - N_true(q))
            J = np.zeros((4,4))
            J[0:2, 2:4] = I2
            J[2:4, 0:2] = A_q
            J[2:4, 2:4] = A_v
            return J

        # integrate with the same stiff settings
        rtol = 1e-6
        atol = np.array([1e-8, 1e-8, 1e-6, 1e-6])
        y0 = np.hstack([q0, v0])
        t_final = 20.0 # simulation time
        sol_true = solve_ivp(
            true_rhs, (0.0, t_final), y0,
            method="Radau", jac=true_jac_quasi,
            rtol=rtol, atol=atol, max_step=0.01, first_step=0.002
        )

        
        Q_true = sol_true.y[0:2, :]
        t_true = sol_true.t

        # overlay on your heatmap
        plt.figure()
        # plot_scalar_map(Xp, Yp, Vk, title="V_k,eff with trajectories", cbar_label="V")
        # plt.plot(Q[0, :],      Q[1, :],      'r-', lw=2, label='eff (magnetic)')
        plt.plot(Q_true[0, :], Q_true[1, :], 'b--', lw=2, label='true (prescribed)')
        plt.plot([q0[0]],[q0[1]], 'ko', ms=4); plt.plot([q_goal[0]],[q_goal[1]], 'k*', ms=8)
        plt.legend(); plt.tight_layout()
        plt.show()





    # --- Build FieldDesign and effective model ---
    design = FieldDesign(
        coil_xy=coil_xy,
        h=h,
        I0=I0,
        Irot_amp=Irot_amp,
        Irot_phase=Irot_phase,
        omega=omega,
        Ihf_amp=Ihf_amp,
        Ihf_phase=Ihf_phase,
        Omega=Omega,
        eps=eps,
        marble_moment=1.0,
        scale=1.0
    )

    eff = build_eff_model(design, m_ball=m_ball, kappa=kappa, c_damp=0.07)

    def scan_meff_ring(R=0.03, K=60):
        """
        Scan M_eff eigenvalues on a ring to ensure SPD and modest anisotropy
        K (int) : Number of points for theta parameter to describe the ring
        R (float) : Radius of the ring
        """
        thetas = np.linspace(0, 2*np.pi, K, endpoint=False)
        pts = q_obs + R*np.c_[np.cos(thetas), np.sin(thetas)]
        lmins, lmaxs = [], []
        for p in pts:
            L = np.linalg.eigvalsh(eff.M_eff(p))
            lmins.append(L[0]); lmaxs.append(L[1])
        return min(lmins), max(lmaxs)
    
    lam_min, lam_max = scan_meff_ring()
    print(f"[Meff] λ_min={lam_min:.3e}, λ_max={lam_max:.3e}")

    
    # --- Realized M_eff / N_eff plots
    if PLOT_M_REALIZED:
        plot_Meff_suite(eff, L, Jside=70, suptitle="Realized M_eff")
    if PLOT_N_REALIZED:
        plot_Neff_suite(eff, L, Jside=70, suptitle="Realized N_eff")


    # --- Simulate averaged dynamics: M_eff(q) qdd = -N_eff(q) qd - gradK(q) (- optional damping) ---
    y0 = np.hstack([q0, v0])
    t_final = 20.0 # simulation time

    def wN(q):
        if not USE_N: return 0.0
        return window_arc(q, q_obs, r_in=r_obs + 0.010, r_out=r_obs + 0.040, angle_center=-np.pi/2, angle_halfwidth=np.pi/5)

    def eff_rhs(t, y):
        print(f"{t} / {t_final}", end='\r')
        q = y[0:2]; v = y[2:4]
        M = eff.M_eff(q) if USE_M else m_ball*np.eye(2)
        rhs = - eff.gradK(q) - eff.c_damp * v 
        if USE_N:
            rhs -= wN(q) *  (eff.N_eff(q, v) @ v)
        a = np.linalg.solve(M, rhs)
        return np.array([v[0], v[1], a[0], a[1]])
    
    def eff_jac(t, y):
        """
        Quasi-Newton Jacobian of f = [v; a(q,v)]:
        ∂f/∂y = [[0, I],
                [∂a/∂q, ∂a/∂v]]
        We approximate:
        ∂a/∂v ≈ M^{-1} * ( -c*I - N(q) )     # drop ∂M^{-1}/∂q term
        ∂a/∂q ≈ M^{-1} * ( -∂(gradK)/∂q - ∂N/∂q @ v )
        For speed/simplicity, you can start with:
        ∂a/∂q ≈ - M^{-1} * H_K(q)            # Hessian of K_eff
        ∂a/∂v as above without ∂N/∂q term.
        """
        
        q = y[0:2]; v = y[2:4]
        I2 = np.eye(2)

        M  = eff.M_eff(q) if USE_M else m_ball*I2
        Minv = np.linalg.inv(M)

        HK = approx_Hess_gradK(q)  # tiny central FD Hessian of gradK

        #  N term at q (skew); ∂N/∂q is small for gentle tapers -> ignore initially
        Nq = eff.N_eff(q, v) if USE_N else np.zeros((2,2))

        A_q = - Minv @ HK                   # (2,2)
        A_v =   Minv @ (-eff.c_damp*np.eye(2) - Nq)

        J = np.zeros((4,4))
        J[0:2, 2:4] = I2        # ∂(qdot)/∂v
        J[2:4, 0:2] = A_q
        J[2:4, 2:4] = A_v
        return J



    if _HAS_SCIPY:
        rtol = 1e-6
        atol = np.array([1e-8, 1e-8, 1e-6, 1e-6])
        method = 'Radau' if USE_M or USE_N else 'LSODA'
        jac = eff_jac if USE_M or USE_N else None
        sol = solve_ivp(eff_rhs, (0.0, t_final), y0, method=method,
                        jac=jac, max_step=0.01, rtol=rtol, atol=atol, first_step=0.002, dense_output=False)
        Q = sol.y[0:2, :]
        dQ = sol.y[2:4, :]
        t = sol.t
        print(f"[SIM] steps={Q.shape[1]}, q(T)={Q[:, -1]}")        
    else:
        # fallback semi-implicit Euler
        dt = 0.005
        steps = int(np.ceil(t_final / dt))
        t = np.linspace(0, t_final, steps)
        y = y0.copy()
        Q = np.zeros((2, steps)); dQ = np.zeros((2, steps))
        for k_step in range(steps):
            dydt = eff_rhs(0.0, y)
            # semi-implicit on velocity
            y[2:4] += dt * dydt[2:4]
            y[0:2] += dt * y[2:4]
            Q[:, k_step] = y[0:2]; dQ[:, k_step] = y[2:4]
        print(f"[SIM fallback] q(T)={y[0:2]}")

    print("Done.")

    # --- Potential proxy & trajectory overlay
    Xp, Yp, Vk = compute_Vk_eff_grid(design, m_ball, q_goal, L, Jside=Jside)
    fig = plt.figure(figsize=(11, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(Xp, Yp, Vk, cmap='viridis', linewidth=0, antialiased=True, alpha=0.9)
    ax1.scatter([q_goal[0]], [q_goal[1]], [float(np.nanmax(Vk))], s=70, c='k', marker='*')
    ax1.set_title("Potential-energy proxy V_k,eff"); ax1.set_xlabel('x'); ax1.set_ylabel('y')

    ax2 = fig.add_subplot(122)
    plot_scalar_map(Xp, Yp, Vk, title="V_k,eff heatmap", cbar_label="V", ax=ax2)
    ax2.plot(Q[0, :], Q[1, :], "r-", lw=2, label="Trajectory")
    ax2.plot([q0[0]], [q0[1]], "ro")
    ax2.plot([q_goal[0]], [q_goal[1]], "k*")
    ax2.plot([q_obs[0]], [q_obs[1]], "b*")
    ax2.legend(); plt.tight_layout()
    
    # --- Time traces
    plt.figure(figsize=(10,4))
    plt.subplot(121); plt.plot(t, Q[0, :], label='x(t)'); plt.plot(t, Q[1, :], label='y(t)'); plt.legend(); plt.xlabel('t [s]'); plt.ylabel('q')
    plt.subplot(122); plt.plot(t, dQ[0, :], label='vx'); plt.plot(t, dQ[1, :], label='vy'); plt.legend(); plt.xlabel('t [s]'); plt.ylabel('v')
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    sys.exit(main())