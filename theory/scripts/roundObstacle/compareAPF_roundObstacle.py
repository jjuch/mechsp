"""
Side-by-side comparison of APF vs Magnetic around a round obstacle.

What this script does:
 1) Builds two systems with the *same* mass, damping, goal, obstacle.
    - APF: SecondOrderSystemAPF  (attractive + repulsive)
    - Magnetic: SignedPowerMagnetic (lossless velocity rotation)
 2) Simulates a small set of initial conditions (aligned with goal axis, and off-axis).
 3) Plots:
      (A) Trajectories with velocity-magnitude coloring (APF vs Magnetic).
      (B) Speed-versus-time overlays for representative starts.
      (C) APF potential contours + candidate local minima markers.
"""


from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle
from matplotlib.colors import Normalize
from typing import Tuple

# our ground truth
from navigate_roundObstacle import (
    Params, Obstacle, SecondOrderSystem,
    SignedPowerMagnetic, dist_n_t, _initial_positions_aligned_with_goal
)

# APF implementation
from APF_roundObstacle import SecondOrderSystemAPF


def make_default_params() -> Params:
    obs = Obstacle(c=np.array([0.0, 0.0]), r=0.5)
    prm = Params(
        qg=np.array([1.20, 1.00]),
        obs=obs,
        m0=1.0,
        alpha=0.0,      # <= Euclidean 
        eps=0.05,
        eps_b=0.10,     # used by other laws; here not critical
        p=2.0,
        c_damp=0.9,     # dissipation common to both strategies
        kB=700.0,       # strong magnetic gain 
        k_psi=1.0,
        d_far=0.15,     # short decay length (near-field focus)
        q_far=2.0
    )
    return prm

def simulate_paths(system, starts: np.ndarray, vn: float = 0.0, 
                   vt: float = 0.05, h: float = 0.05, tmax: float = 40.0):
    """
    Run a batch of paths with small tangential v0 (as in your figures).
    Returns lists of arrays Q_i (positions), V_i (velocities), T_i (time).
    """
    Qs, Vs, Ts = [], [], []
    for i, q0 in enumerate(starts):
        print(f"{i + 1} / {len(starts)}: {q0}")
        d, n, t = dist_n_t(q0, system.prm.obs)
        v0 = vn * n + vt * t
        T, Xs, _ = system.simulate(q0, v0, h=h, tmax=tmax)
        Qs.append(Xs[:, :2]); Vs.append(Xs[:, 2:4]); Ts.append(T)
    return Qs, Vs, Ts


def plot_trajectories_with_speed(ax, prm, Qs, Vs, label=None, norm: Normalize | None = None, cmap: str = 'viridis'):
    """
    Draw colored trajectories with speed magnitude on 'ax'.
    """
    if norm is None:
        # Fallback: compute a local norm if not provided
        speeds = [np.linalg.norm(V, axis=1) for V in Vs if len(V)]
        if speeds:
            vmin = min(s.min() for s in speeds)
            vmax = max(s.max() for s in speeds)
            norm = Normalize(vmin=vmin, vmax=vmax)
        else:
            norm = Normalize(vmin=0.0, vmax=1.0)
        
    for Q, V in zip(Qs, Vs):
        if len(Q) < 2: continue
        s = np.linalg.norm(V, axis=1)
        seg = np.concatenate([Q[:-1, None, :], Q[1:, None, :]], axis=1)

        lc = LineCollection(seg, cmap=cmap, norm=norm)
        lc.set_array(s[:-1]); lc.set_linewidth(2.0)
        ax.add_collection(lc)
        # start marker
        ax.plot(Q[0,0], Q[0,1], 'ko', ms=3)

    # obstacle and goal
    ax.add_patch(Circle(prm.obs.c, prm.obs.r, facecolor='w', alpha=0.7, edgecolor='k'))
    ax.plot(prm.qg[0], prm.qg[1], 'y*', ms=14)

    if label:
        ax.set_title(label)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.25)


def compare_APF_vs_Magnetic(save: bool=False):
    prm = make_default_params()

    # --- Build the two systems ---
    # APF: tune eta_rep and d0 so APF "feels" obstacle well before contact
    apf = SecondOrderSystemAPF(prm,
                               k_att=prm.k_psi*0.5,  # if prm.k_psi: keep same attractive gain
                               d0=0.5,           # influence radius 0.35 (m)
                               d_eps=0.03,
                               smooth_cap=1e-2,
                               U_wall=2.25,
                               F_cap=30)

    # Magnetic: SignedPowerMagnetic (lossless curvature) using your ground-truth PRM
    mag = SecondOrderSystem(prm, SignedPowerMagnetic(prm))

    # --- Initial conditions ---
    starts = _initial_positions_aligned_with_goal(prm, num=5)

    # Add a strictly head-on start on the goal axis (near r2)
    r2_extra = prm.obs.r + 0.85  # far enough to show path differences
    q_axis = prm.obs.c + r2_extra * ((prm.qg - prm.obs.c)/np.linalg.norm(prm.qg - prm.obs.c))
    starts = np.vstack([starts, q_axis[None, :]])


    # --- Simulations ---
    # small tangential seed (as in your figures)
    Qs_apf, Vs_apf, Ts_apf = simulate_paths(apf, starts, vn=0.0, vt=0.05, h=0.05, tmax=40.0)
    Qs_mag, Vs_mag, Ts_mag = simulate_paths(mag, starts, vn=0.0, vt=0.05, h=0.05, tmax=40.0)

    # --- Figure (A): Trajectories with velocity color (APF vs Magnetic) ---
    fig = plt.figure(figsize=(12.5, 5.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.045, 1.0], wspace=0.06)

    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 2])
    cax = fig.add_subplot(gs[0, 1])
    
    all_speeds = []
    for Vs in (Vs_apf, Vs_mag):
        for V in Vs:
            if len(V):
                all_speeds.append(np.linalg.norm(V, axis=1))

    if all_speeds:
        vmin = min(s.min() for s in all_speeds)
        vmax = max(s.max() for s in all_speeds)
    else:
        vmin, vmax = 0.0, 1.0
    
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = 'viridis'

    plot_trajectories_with_speed(axL, prm, Qs_apf, Vs_apf, label="APF", norm=norm, cmap=cmap)
    plot_trajectories_with_speed(axR, prm, Qs_mag, Vs_mag, label="Magnetic ", norm=norm, cmap=cmap)

    for ax in (axL, axR):
        ax.set_xlim(-2.0, 2.0); ax.set_ylim(-2.0, 2.0)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
    cbar.set_label('speed |v| [m/s]')

    # plt.tight_layout()
    if save:
        plt.savefig(f"figs/compare_trajectories_APF_vs_Magnetic.png", dpi=180)
        plt.close(fig)
    else:
        plt.show()

        
    # --- Figure (B): Speed vs time overlays for two representative starts ---
    # pick (i) strictly head-on q_axis and (ii) a mid off-axis start
    head_on_idx = len(starts) - 1
    mid_idx = len(starts) // 2

    def speed_trace(Ts, Vs, idx):
        return Ts[idx], np.linalg.norm(Vs[idx], axis=1)
    
    T_ah, v_ah = speed_trace(Ts_apf, Vs_apf, head_on_idx)
    T_mh, v_mh = speed_trace(Ts_mag, Vs_mag, head_on_idx)
    T_am, v_am = speed_trace(Ts_apf, Vs_apf, mid_idx)
    T_mm, v_mm = speed_trace(Ts_mag, Vs_mag, mid_idx)

    fig2, axes2 = plt.subplots(1, 2, figsize=(12.5, 5.6))
    axes2[0].plot(T_ah, v_ah, 'r-', lw=2, label='APF')
    axes2[0].plot(T_mh, v_mh, 'b-', lw=2, label='Magnetic')
    axes2[0].set_title("Speed vs time (head-on)")
    axes2[0].set_xlabel('t [s]'); axes2[0].set_ylabel('|v| [m/s]'); axes2[0].grid(True, alpha=0.3); axes2[0].legend()

    axes2[1].plot(T_am, v_am, 'r-', lw=2, label='APF')
    axes2[1].plot(T_mm, v_mm, 'b-', lw=2, label='Magnetic')
    axes2[1].set_title("Speed vs time (off-axis)")
    axes2[1].set_xlabel('t [s]'); axes2[1].set_ylabel('|v| [m/s]'); axes2[1].grid(True, alpha=0.3); axes2[1].legend()

    plt.tight_layout()
    if save:
        plt.savefig("figs/compare_speed_traces.png", dpi=180)
        plt.close(fig2)
    else:
        plt.show()
        
    # --- (C) APF potential map + candidate minima markers ---
    if save:
        save_as = "figs/APF_potential_and_minima.png"
    else:
        save_as = None

    apf.plot_potential_and_minima(xlim=(-2, 2), ylim=(-2, 2), Nx=160, Ny=160,
                                  grad_thresh=1e-2,
                                  save_as=save_as)

    print("Generated:")
    print("  figs/compare_trajectories_APF_vs_Magnetic.png")
    print("  figs/compare_speed_traces.png")
    print("  figs/APF_potential_and_minima.png")


if __name__ == "__main__":
    import os
    os.makedirs("figs", exist_ok=True)
    compare_APF_vs_Magnetic(save=False)

