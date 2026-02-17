# mechsp/examples/effective_minimal_demo.py
"""
Minimal demonstration of the averaged (effective) model.

Steps:
 1) Build a coil grid (n x m)
 2) Synthesize DC currents for K_eff (quadratic well to a goal)
 3) Synthesize rotating phasors for a simple swirl (phi=atan2)
 4) Skip HF (set ΔM=0) in this minimal demo
 5) Build FieldDesign and effective model
 6) Integrate averaged dynamics with solve_ivp

Run:
  python -m mechsp.examples.effective_minimal_demo
"""

import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

try:
    from scipy.integrate import solve_ivp
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

from mechsp.synthesis.synthesis_dc import solve_dc_currents, solve_dc_currents_anchored
from mechsp.synthesis.synthesis_rot import solve_rotating_phasors
# HF left out here for minimal demo
from mechsp.fields import FieldDesign
from mechsp.eff_dynamics import build_eff_model
from mechsp.magnetics import grad_Bz_analytic


def make_coils(L=0.20, n=14, m=14, h=0.02):
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


def sample_domain(L, Jside=28):
    xs = np.linspace(-L/2, L/2, Jside)
    ys = np.linspace(-L/2, L/2, Jside)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    S = np.stack([X.ravel(), Y.ravel()], axis=1)
    return S, (X, Y)


def compute_Vk_eff_grid(design, m_ball, q_goal, L, Jside=50):
    """
    Compute the potential energy field V_{k,eff}(x,y) = ∇K_eff(q) · (q - q_goal) over a Jside x Jside grid.

    Returns
    -------
    X, Y : (Jside,Jside) grids of coordinates
    K_p   : (Jside,Jside) scalar field
    """
    S, (X, Y) = sample_domain(L, Jside=Jside)

    # 2) Compute ∇Bz0 for all points against all coils in one broadcasted call:
    #    grad_Bz_analytic(S, coil_xy, h) -> (J, N, 2)
    G_all = grad_Bz_analytic(S, design.coil_xy, design.h, scale=design.scale)   # (J,N,2)

    # 3) ∇K_eff(q) = - m_b * Σ_i I0_i ∇b_i(q)
    I0 = design.I0.reshape(1, -1, 1)            # (1,N,1) for broadcasting
    gradK = -m_ball * np.sum(I0 * G_all, axis=1)   # (J,2)

    # 4) Dot with (q - q_goal) to get scalar F_{k,eff}
    dq = S - q_goal[None, :]             # (J,2)
    K_p_flat = 0.5 * np.einsum('ij,ij->i', gradK, dq)  # (J,) Potential energy
    
    K_p = K_p_flat.reshape(Jside, Jside)
    return X, Y, K_p

def plot_potEn_eff_surface(X, Y, Vk, q_goal, title="Scalar field", q=None):
    """
    3-D surface of the potential energy + heatmap with optional trajectory overlay.
    X,Y,Vk : (J,J)
    Q     : (T,2) trajectory points in (x,y)
    q_goal: (2,) goal point (optional)
    """

    fig = plt.figure(figsize=(11,5))

    # Left: 3D surface
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(X, Y, Vk, cmap='viridis', linewidth=0, antialiased=True, alpha=0.9)
    ax.scatter([q_goal[0]], [q_goal[1]], [float(np.nanmax(Vk))], s=70, c='k', marker='*')
    if q is not None:
        xi = np.searchsorted(np.sort(np.unique(X[0, :])), q[0]) - 1
        yi = np.searchsorted(np.sort(np.unique(Y[0, :])), q[1]) - 1
        xi = np.clip(xi, 0, X.shape[1] - 1)
        yi = np.clip(yi, 0, Y.shape[0] - 1)
        Vkq = Vk[yi, xi]
        ax.plot(q[0], q[1], Vkq, 'r-', lw=2.0, label='trajectory')
    ax.set_title(title)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_zlabel('V_{k,eff}')
    ax.legend()

    # Right: heatmap
    ax2 = fig.add_subplot(122)
    im = ax2.imshow(Vk, origin='lower',
                    extent=[X.min(), X.max(), Y.min(), Y.max()],
                    cmap='viridis', aspect='equal')
    ax2.plot([q_goal[0]], [q_goal[1]], 'k*', ms=10)
    if q is not None:
        ax2.plot(q[0], q[1], 'r-', lw=2.0)
        ax2.plot(q[0][0], q[1][0], 'ro', ms=5, label='start')
        ax2.plot(q[0][-1], q[1][-1], 'r*', ms=8, label='end')
    ax2.set_title('Heatmap of F_{k,eff}')
    ax2.set_xlabel('x [m]')
    ax2.set_ylabel('y [m]')
    ax2.legend()
    cb = fig.colorbar(im, ax=ax2)
    cb.set_label('V_{k,eff}')

    plt.tight_layout()
    plt.show()




def main():
    # --- geometry & params ---
    coil_xy, h, L = make_coils(L=0.20, n=16, m=16, h=0.02)
    m_ball = 0.015  # kg (example)
    q_goal = np.array([0.06, -0.03])
    q0 = np.array([-0.07, 0.07])
    v0 = np.zeros(2)
    Jside = 100 # 30 by 30 grid

    # --- DC synthesis for K_eff ---
    S, (X, Y) = sample_domain(L, Jside=Jside)

    # Desired linear field
    k = 20.0
    F_des = -k * (S - q_goal[None, :])  # linear "spring" toward goal
    q_minus_g = S - q_goal[None, :]             # (J,2)
    Vdes_flat = np.einsum('ij,ij->i', k*np.ones(q_minus_g.shape), q_minus_g**2)  # (J,)
    Vdes = Vdes_flat.reshape(Jside, Jside)
    # plot_potEn_eff_surface(X, Y, Vdes, q_goal)
    
    # Region weights: emphasise vicinity of q_goal
    d = np.linalg.norm(S - q_goal[None, :], axis=1)
    sigma = 0.05 # 5cm
    w_region = np.exp(-(d**2) / (2*sigma**2))

    # I0, diag_dc = solve_dc_currents_anchored(
    #     sample_xy=S, F_des=F_des,
    #     coil_xy = coil_xy, h=h, m_ball=m_ball, q_goal=q_goal, k_stiff=k, lam=1e-4, w_region=w_region, w_goalF=50, w_goalH=10.0, scale=1.0)
    I0, diag_dc = solve_dc_currents(
        sample_xy=S, F_des=F_des,
        coil_xy = coil_xy, h=h, lam=1e-4, scale=1.0) 
    print(f"[DC anchored] rel_fit={diag_dc['rel_fit_err']:.3e}, cond≈{diag_dc['cond']:.2e}")
    # print(f"   Force at goal  ≈ {diag_dc['F_goal']}")
    # print(f"   JF at goal     ≈\n{diag_dc['JF_goal']}")


    # --- Rotating field for swirl (N_eff) ---# 
    # Helper functions
    def smooth_step(u, k=6):
        """
        C^k smooth step from 0 to 1:
        u in [0,1] -> sigma(u) in [0,1]
        Here use a simple polynomial 10u^3 - 15u^4 + 6u^5 (C^2), good enough for our need.
        """
        u = np.clip(u, 0.0, 1.0)
        return 10*u**3 - 15*u**4 + 6*u**5

    def phase_taper(r, r0, r1):
        """
        Taper s(r) that equals 0 for r<=r0, rises smoothly to 1 by r>=r1.
        """
        return smooth_step((r - r0) / max(1e-12, (r1 - r0)))

    def B0_profile(r, L, beta=0.8, r0=0.015, r1=0.04, p=3):
        """
        B0(r) ~ (r/r1)^p inside, saturating smoothly to beta*(r/(L/2)) outside.
        Ensures B0=O(r^p) near 0; pick p>=3 for fast decay of gamma.
        """
        # inner polynomial
        inner = (r / max(r1,1e-9))**p
        # outer gentle rise (kept small)
        outer = beta * (r / (L/2))
        s = phase_taper(r, r0, r1)  # 0..1
        return (1 - s)*inner + s*outer
    
    # actual definitions
    dxy = S - q_goal[None, :]
    r = np.linalg.norm(dxy, axis=1)
    phi_raw = np.arctan2(dxy[:,1], dxy[:,0])

    # phase taper: freeze phase inside r<=r0 so ∇phi=0 there
    r0, r1 = 0.012, 0.03  # 1.2cm inner carpet, 3cm full phase on
    s_phase = phase_taper(r, r0, r1)
    phi_des = s_phase * phi_raw          # fades to 0 near goal

    # B0 with higher-order vanish near goal, mild outside
    B0_des = B0_profile(r, L, beta=0.6, r0=r0, r1=r1, p=5)
    B0_des = np.zeros(B0_des.shape)

    # VERY IMPORTANT: include q_goal itself as a sample with B0=0 and any phi (ignored by solver)
    S_plus = np.vstack([q_goal[None,:], S])
    B0_plus = np.concatenate([[0.0], B0_des])
    phi_plus = np.concatenate([[0.0], phi_des])
    phi_plus = np.zeros(phi_plus.shape)

    Irot_amp, Irot_phase, diag_rot = solve_rotating_phasors(S_plus, B0_plus, phi_plus, coil_xy, h, lam=1e-3)
    print(f"[ROT] rel_fit={diag_rot['rel_fit_err']:.3e}")

    # --- HF: skip (ΔM=0) for the minimal demo ---
    Ihf_amp = np.zeros_like(I0)
    Ihf_phase = np.zeros_like(I0)

    # --- Frequencies ---
    omega = 2*np.pi*10.0   # 10 Hz rotation (example)
    Omega = 2*np.pi*500.0  # 500 Hz HF (unused here)
    eps = 0.1              # HF small parameter (unused here)

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

    # choose a dipole lag rate kappa > omega, for delta=arcsin(omega/kappa)
    kappa = 5.0 * omega  # previously 2.0; now smaller sin(delta)
    eff = build_eff_model(design, m_ball=m_ball, kappa=kappa, c_damp=0.05)

    # --- Simulate averaged dynamics: M_eff(q) qdd = -N_eff(q) qd - gradK(q) ---
    y0 = np.hstack([q0, v0])
    t_final = 20.0 # simulation time

    def eff_rhs(t, y):
        print(f"{t} / {t_final}", end='\r')
        q = y[0:2]; v = y[2:4]
        M = eff.M_eff(q)
        rhs = - eff.gradK(q) - eff.c_damp * v #- eff.N_eff(q, v) @ v
        a = np.linalg.solve(M, rhs)
        return np.array([v[0], v[1], a[0], a[1]])



    if _HAS_SCIPY:
        sol = solve_ivp(eff_rhs, (0.0, t_final), y0, method='LSODA', max_step=0.01, rtol=1e-6, atol=1e-9)
        Q = sol.y[0:2, :].T
        t = sol.t
        print(f"[SIM] steps={Q.shape[0]}, q(T)={Q[-1]}")
        Q = Q.T
        dQ = sol.y[2:4, :]
    else:
        # fallback semi-implicit Euler
        dt = 0.005
        steps = int(np.ceil(t_final / dt))
        t = np.linspace(0, t_final, steps)
        y = y0.copy()
        for _ in range(steps):
            dydt = eff_rhs(0.0, y)
            # semi-implicit on velocity
            y[2:4] += dt * dydt[2:4]
            y[0:2] += dt * y[2:4]
        Q = y[0:2]
        dQ = y[2:4]
        print(f"[SIM fallback] q(T)={y[0:2]}")

    
    probe = [q_goal,
            q_goal + np.array([r0/2, 0]),
            q_goal + np.array([r0,   0]),
            q_goal + np.array([r1,   0])]
    for p in probe:
        Ne = eff.N_eff(p, np.array([1.0,0.0]))
        print(f"q={p}, ||N_eff||_F={np.linalg.norm(Ne):.3e}") # You want 0, ~0, very small, small respectively.

    print("Done.")
    # Check potential energy well
    X, Y, Vk = compute_Vk_eff_grid(design, m_ball, q_goal, L, Jside=Jside)
    plot_potEn_eff_surface(X, Y, Vk, q_goal, q=Q)

    plt.figure()
    plt.subplot(121)
    plt.plot(t, Q[0, :], label='x(t)')
    plt.plot(t, Q[1, :], label='y(t)')
    plt.xlabel('t [s]')
    plt.legend()
    plt.subplot(122)
    plt.plot(t, dQ[0, :], label='dx(t)')
    plt.plot(t, dQ[1, :], label='dy(t)')
    plt.xlabel('t [s]')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    sys.exit(main())