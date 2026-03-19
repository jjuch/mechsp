"""
Figure generator for the second‑order obstacle-aware navigation model.

Rules/laws:
  • 'none'          -> NoMagnetic()
  • 'const'         -> ConstMagnetic()
  • 'dp'            -> PowerMagnetic()         (b(d)=kB d^p φ_far)
  • 'sine'          -> SineMagnetic()          (b(d,θ)=kB d^p φ_far * [(1-S)+S sinθ])
  • 'sine_rphase'   -> SineRPhaseMagnetic()    (b(d,θ,r)=kB d^p w_ann(r) φ_far * sin(θ+φ(r)),
                                                with auto-selection of (r1,r2))

This file only:
  1) Instantiates a scenario.
  2) Builds a few laws.
  3) Generates the figures.

All heavy lifting is done in navigate_roundObstacle.py
"""

from __future__ import annotations
import os, numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import List, Tuple
from navigate_roundObstacle import (
    Params, Obstacle, SecondOrderSystem,
    NoMagnetic, ConstMagnetic, PowerMagnetic, SineMagnetic, SineRPhaseMagnetic, TiltedSineMagnetic, SignedPowerMagnetic,
    dist_n_t, boundary_compliance, _initial_positions_aligned_with_goal
)

os.makedirs("figs", exist_ok=True)
SAVE = True

# -------------------------
# Scenario
# -------------------------

def make_default_params() -> Params:
    obs = Obstacle(c=np.array([0.0, 0.0]), r=0.5)
    prm = Params(
        qg=np.array([1.20, 1.00]),
        obs=obs,
        m0=1.0,
        alpha=1.2,
        eps=0.05,
        eps_b=0.02,
        p=2.0,
        c_damp=0.60,
        kB=100.0,
        k_psi=1.0,
        d_far=0.35,
        q_far=2.0
    )
    return prm

# -------------------------
# Figures
# -------------------------

def figA_invariance(prm: Params, systems: List[Tuple[str, SecondOrderSystem]], save_as: str):
    """
    Fig A: grazing boundary test – fraction n·a>=0 vs distance d. Only the goal‑opposing semicircle is evaluated (risk-aware).
    """
    print("Figure A")
    Ds = np.geomspace(1e-3, 0.35, 32)
    vt_list = (0.2, 0.5, 0.8)

    fig, ax = plt.subplots(1, 1, figsize=(6.8, 4.2))
    for label, sys in systems:
        print(f"label: {label}")
        # frac = boundary_compliance(sys, Ds, vt_list, sector_only=True)
        # For a clean A: we also plot the curve vs d by sampling per ring:
        fracs_per_d = []
        for i, d in enumerate(Ds):
            print(f"{i} / {len(Ds)}", end="\r")
            fracs_per_d.append(boundary_compliance(sys, [d], vt_list, sector_only=True))
        ax.plot(Ds, fracs_per_d, label=label)
    ax.set_xscale('log'); ax.set_ylim(0, 1.05)
    ax.set_xlabel("distance d [m]"); ax.set_ylabel("fraction with n·a ≥ 0 (grazing)")
    ax.set_title("Boundary test (goal‑opposing semicircle)")
    ax.grid(True, which='both', alpha=0.3); ax.legend(fontsize=9)
    plt.tight_layout()
    if SAVE: plt.savefig(save_as, dpi=180); plt.close(fig)
    else:    plt.show()

def figB_trajectories(prm: Params, systems: List[Tuple[str, SecondOrderSystem]], save_as: str):
    """
    Fig B: trajectory overlays (acceleration streamlines as background).
    """
    print("Figure B")
    XMIN, XMAX, YMIN, YMAX = -2.0, 2.0, -2.0, 2.0
    starts = _initial_positions_aligned_with_goal(prm, num=5)

    ncol = len(systems)
    fig, axes = plt.subplots(1, ncol, figsize=(4.6*ncol, 4.4))
    if ncol == 1: axes = [axes]
    for ax, (label, sys) in zip(axes, systems):
        print(f"label: {label}")
        # background (accel streamlines for grazing vt=0.6)
        XX,YY,U,V = _acc_field_on_grid(sys, vt_stream=0.6, Nx=60, Ny=60,
                                       XMIN=XMIN, XMAX=XMAX, YMIN=YMIN, YMAX=YMAX)
        sp = np.sqrt(U**2+V**2)
        ax.streamplot(XX,YY,U,V, color=np.clip(sp,0,3), density=1.0, linewidth=0.6, cmap='viridis')

        # obstacle & goal
        ax.add_patch(Circle(prm.obs.c, prm.obs.r, color='k', alpha=0.15))
        ax.plot(prm.qg[0], prm.qg[1], 'r*', ms=11)

        # trajectories (small tangential v0)
        for i, q0 in enumerate(starts):
            print(f"{i} / {len(starts)}: {q0}")
            _,_,t = dist_n_t(q0, prm.obs)
            v0 = 0.05*t
            _, Xs, _ = sys.simulate(q0, v0, h=0.05, tmax=40.0)
            Q = Xs[:,:2]
            ax.plot(Q[:,0], Q[:,1], '-', lw=2)
            ax.plot(q0[0], q0[1], 'ko', ms=3)

        ax.set_title(label, fontsize=10)
        ax.set_xlim([XMIN,XMAX]); ax.set_ylim([YMIN,YMAX]); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if SAVE: plt.savefig(save_as, dpi=180); plt.close(fig)
    else:    plt.show()

def figC_ring_accels(prm: Params, systems: List[Tuple[str, SecondOrderSystem]], save_as: str):
    """
    Fig C: ring‑averaged ⟨max(0,n·a)⟩ and ⟨|t·a|⟩ on the goal‑opposing semicircle at grazing.
    """
    print("Figure C")
    Ds = np.geomspace(1e-3, 0.35, 32)
    vt = 0.6
    thetas = np.linspace(0, 2*np.pi, 240, endpoint=False)

    def in_goal_opp(q: np.ndarray) -> bool:
        return ((q - prm.obs.c) @ (prm.qg - prm.obs.c)) < 0.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for label, sys in systems:
        print(f"label: {label}")
        avg_na, avg_ta = [], []
        for i, d in enumerate(Ds):
            print(f"{i} / {len(Ds)}", end='\r')
            nas, tas = [], []
            for th in thetas:
                q = prm.obs.c + (prm.obs.r + d)*np.array([np.cos(th), np.sin(th)])
                if not in_goal_opp(q): continue
                dd, n, t = dist_n_t(q, prm.obs)
                v = vt*t
                M = sys.M_of_q(q); Minv = np.linalg.inv(M)
                Cv = sys.C_times_qdot(q, v); N = sys.N_of_q(q)
                a  = Minv @ (N @ v - Cv - prm.c_damp*(M @ v) - sys.grad_psi(q))
                nas.append(max(0.0, float(n @ a)))
                tas.append(abs(float(t @ a)))
            avg_na.append(np.mean(nas) if nas else 0.0)
            avg_ta.append(np.mean(tas) if tas else 0.0)
        axes[0].plot(Ds, avg_na, label=label)
        axes[1].plot(Ds, avg_ta, label=label)

    axes[0].set_xscale('log'); axes[1].set_xscale('log')
    axes[0].set_title('⟨max(0, n·a)⟩ (grazing, goal‑opposing semicircle)')
    axes[1].set_title('⟨|t·a|⟩ (grazing, goal‑opposing semicircle)')
    for a in axes:
        a.set_xlabel("distance d [m]"); a.grid(True, which='both', alpha=0.3); a.legend(fontsize=9)
    axes[0].set_ylabel("outward normal accel. [m/s²]")
    axes[1].set_ylabel("tangential accel. [m/s²]")
    plt.tight_layout()
    if SAVE: plt.savefig(save_as, dpi=180); plt.close(fig)
    else:    plt.show()


# -------------------------
# Utilities for figures
# -------------------------

def _acc_field_on_grid(system: SecondOrderSystem, vt_stream: float,
                       Nx: int, Ny: int, XMIN: float, XMAX: float, YMIN: float, YMAX: float):
    xs = np.linspace(XMIN, XMAX, Nx); ys = np.linspace(YMIN, YMAX, Ny)
    XX, YY = np.meshgrid(xs, ys)
    U = np.zeros_like(XX); V = np.zeros_like(YY)
    for i in range(Nx):
        for j in range(Ny):
            q = np.array([XX[j,i], YY[j,i]])
            d,_,t = dist_n_t(q, system.prm.obs)
            if d <= 0: continue
            v = vt_stream * t
            a = system.rhs(np.hstack([q, v]))[2:]
            U[j,i], V[j,i] = a
    return XX, YY, U, V


# -------------------------
# Main
# -------------------------

def main():
    prm = make_default_params()
    # overwrite default parameters
    prm.alpha = 0.00
    prm.kB = 700
    prm.d_far=0.15
    prm.eps_b = 0.10

    # Laws:
    sys_none  = SecondOrderSystem(prm, NoMagnetic(prm))
    sys_const = SecondOrderSystem(prm, ConstMagnetic(prm))
    sys_dp    = SecondOrderSystem(prm, PowerMagnetic(prm))
    sys_sine  = SecondOrderSystem(prm, SineMagnetic(prm, d_sw=0.00, w_sw=0.00))
    sys_dpsigned = SecondOrderSystem(prm, SignedPowerMagnetic(prm))
    sys_sine_rphase = SecondOrderSystem(prm, SineRPhaseMagnetic(prm, phi_max=np.pi/2,
                                                                sigma_frac=0.35,
                                                                use_tanh=False,
                                                                k=4.0,
                                                                sector_only=True, r1=None, r2=None))
    vn_list = (0.5, 1.0, 1.5) # head-on speeds
    kB_fixed = prm.kB
    r2_fixed = prm.obs.r + 0.5

    law_signed = SignedPowerMagnetic(prm)
    law_signed.design_headon_tradeoff(vn_list,
                                      kB_fixed=kB_fixed, r2_fixed=r2_fixed, simulate=True)
    exit()
    
    law = TiltedSineMagnetic(prm,
                            phi_max=np.pi/8, 
                             sigma_frac=0.45, 
                             use_tanh=False, 
                             gamma=2.0, 
                             w_min=0.0, 
                             eps0=0.0, 
                             sigma_cap=0.10, 
                             kB_cap_rel=0.20, 
                             theta_cap=np.pi/6,
                             delta_cap=0.07)
    sys_tilted_sine = SecondOrderSystem(prm, b_law=law)

    # You can comment modes in/out as needed
    systems = [
        ("none",          sys_none),
        # ("const",         sys_const),
        # ("dp",            sys_dp),
        # ("sine",          sys_sine),
        ("sine_rphase",   sys_sine_rphase),
        ("sine_tilted", sys_tilted_sine)
    ]

    # figA_invariance(prm, systems, save_as="figs/FigA_invariance.png")
    # figB_trajectories(prm, systems, save_as="figs/FigB_trajectories.png")
    # figC_ring_accels(prm, systems, save_as="figs/FigC_ring_accels.png")
    # exit()

    # Check curvatures
    # law = SineRPhaseMagnetic(phi_max=np.pi/6, sigma_frac=0.35, use_tanh=False, sector_only=True, r1=None, r2=None) #r1=prm.obs.r + 0.15, r2=prm.obs.r + 0.75)

    

    show_traj = True
    if show_traj:
        traj = law.plot_trajectories(save_as=None,
                                    plot=True,
                                    q0=None,
                                    n_traj=3)
    else:
        traj = None

    for mode in ['tangent', 'headon']:
        law.predict_RA(mode=mode,
                       vt_list=(0.5, 1.0, 1.5),
                       vn_list=(0.5, 1.0, 1.5),
                       n_theta=240,
                       with_trajectories=True,
                       trajectories=traj,
                       n_traj=10,
                       axis_choice='both'
                       )
        

    law.plot_curvature_maps(save_as=False, # name: "figs/curv_maps_sine_rphase.png"
                            xlim=(-2,2), 
                            ylim=(-2, 2), 
                            vts=(0.6,), 
                            with_trajectories=show_traj,
                            trajectories=traj,
                            n_traj=10, 
                            plot_radii=True) 
    
    law.plot_grazing_normal_maps(save_as=None, # name: "figs/grazing_na_sine_rphase.png" 
                                vts=(0.6,), 
                                with_trajectories=show_traj, 
                                trajectories=traj,
                                axis_gap=0.0, 
                                plot_radii=True) # ignore +-0.10 rad around the goal axis if the angular window zeros ther

if __name__ == "__main__":
    main()