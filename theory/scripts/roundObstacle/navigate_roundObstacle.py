"""
Minimal library for the 2‑D second‑order navigation model with round obstacle-aware metric and gyroscopic two-form. Includes:
  • Geometry utilities.
  • SecondOrderSystem (Levi–Civita, Rayleigh damping).
  • Magnetic laws: None, Const, Power(dp), Sine (θ), TwistedSine, SineRPhase (θ + φ(r)).
  • Diagnostics: boundary compliance, annulus auto-selection for SineRPhase.
  • Lightweight simulation (RK4).

Author: Jasper Juchem
"""

from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from typing import Callable, Iterable, Tuple, List, Dict, Optional, Literal
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from PyQt5.QtCore import QTimer

# -------------------------
# Constants & helpers
# -------------------------

J = np.array([[0.0, -1.0],
              [1.0,  0.0]], dtype=float)

@dataclass
class Obstacle:
    """Round obstacle."""
    c: np.ndarray  # center (2,)
    r: float       # radius

@dataclass
class Params:
    """
    Global parameters of the navigation problem.
    """
    qg: np.ndarray        # goal
    obs: Obstacle         # obstacle
    m0: float = 1.0       # base mass
    alpha: float = 1.2    # metric normal amplification gain
    eps: float = 0.05     # metric regularization (m)
    eps_b: float = 0.02   # magnetic base level near obstacle (d -> 0)
    p: float = 2.0        # exponent for distance law d^p (p>1)
    c_damp: float = 0.9   # Rayleigh damping coefficient
    kB: float = 1.0       # magnetic gain base
    k_psi: float = 1.0    # goal potential stiffness
    d_far: float = 3.5    # far field filter - magnitude
    q_far: float = 2.0    # far field filer - exponent

# -------------------------
# Geometry
# -------------------------

def dist_n_t(q: np.ndarray, obs: Obstacle) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Distance to the obstacle boundary and the local (n,t) frame.

    Returns:
        d: signed distance to boundary (d>0 outside).
        n: outward normal at q (unit).
        t: unit tangent = J n.
    """
    qc = q - obs.c
    r = np.linalg.norm(qc)
    d = r - obs.r
    if r < 1e-12:
        n = np.array([1.0, 0.0])
    else:
        n = qc / r
    t = J @ n
    return d, n, t

def s_of_d(d: float, eps: float) -> float:
    """Metric radial profile s(d)=1/(d^2+eps^2), clamped for d<0."""
    return 1.0 / ((max(d, 0.0))**2 + eps**2)

def theta_rel_goal(q: np.ndarray, qg: np.ndarray, c: np.ndarray) -> float:
    """
    Angle between (q-c) and (qg-c); θ=0 on obstacle-goal axis (counterclockwise positive).
    """
    qc = q - c
    r = np.linalg.norm(qc)
    if r < 1e-12:
        return 0.0
    eg = (qg - c)
    eg = eg / max(np.linalg.norm(eg), 1e-12)
    cos_th = (qc @ eg) / r
    sin_th = (qc @ (J @ eg)) / r
    return float(np.arctan2(sin_th, cos_th))

def phi_window_far(d: float, d_on: float = 0.35, q: float = 2.0) -> float:
    """
    Far-field window for b(d): ~1 near obstacle, decays after d_on.
    """
    return float(np.exp(- (max(d, 0.0)/d_on)**q))

# -------------------------
# System (Levi–Civita + damping)
# -------------------------

class SecondOrderSystem:
    """
    Second-order Lagrangian system with:
      M(q) = m0 I + alpha s(d) n n^T,
      ψ(q) = 0.5*k_psi*||q-qg||^2,
      N(q) = b(q)*J (gyroscopic two-form),
      EOM: M qdd + C(q,qd) qd + c_damp M qd + ∇ψ = N(q) qd.
    """
    def __init__(self, prm: Params, b_law: "MagneticLaw"):
        self.prm = prm
        self.b_law = b_law

    # ---- metric & connection ----
    def M_of_q(self, q: np.ndarray) -> np.ndarray:
        d, n, _ = dist_n_t(q, self.prm.obs)
        s = s_of_d(d, self.prm.eps)
        M = self.prm.m0*np.eye(2) + self.prm.alpha*s*np.outer(n, n)
        return 0.5*(M+M.T)

    def partial_M(self, q: np.ndarray, axis: int = 0, h: float = 2e-4) -> np.ndarray:
        e = np.array([1.0, 0.0]) if axis == 0 else np.array([0.0, 1.0])
        return (self.M_of_q(q + h*e) - self.M_of_q(q - h*e)) / (2*h)

    def christoffel(self, q: np.ndarray) -> np.ndarray:
        M = self.M_of_q(q)
        Minv = np.linalg.inv(M)
        dMx, dMy = self.partial_M(q, 0), self.partial_M(q, 1)
        dM = np.stack([dMx, dMy], axis=0)  # (2,2,2)
        G = np.zeros((2,2,2))
        # Γ^i_{jk} = (1/2) Σ_l M^{il}(∂_j M_{lk} + ∂_k M_{lj} - ∂_l M_{jk})
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    s = 0.0
                    for l in range(2):
                        term = dM[j,l,k] + dM[k,l,j] - dM[l,j,k]
                        s += Minv[i,l]*term
                    G[i,j,k] = 0.5*s
        return G

    def C_times_qdot(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        G = self.christoffel(q)
        # (C(q,v) v)^i = Σ_{j,k} Γ^i_{jk} v^j v^k
        a = np.zeros(2)
        for i in range(2):
            a[i] = v @ G[i] @ v
        return a

    def grad_psi(self, q: np.ndarray) -> np.ndarray:
        return self.prm.k_psi*(q - self.prm.qg)

    # ---- gyroscopic two-form ----
    def N_of_q(self, q: np.ndarray) -> np.ndarray:
        return self.b_law.b_scalar(q) * J

    # ---- dynamics ----
    def rhs(self, x: np.ndarray) -> np.ndarray:
        q = x[:2]; v = x[2:]
        M = self.M_of_q(q); Minv = np.linalg.inv(M)
        Cv = self.C_times_qdot(q, v)
        N  = self.N_of_q(q)
        a  = Minv @ (N @ v - Cv - self.prm.c_damp*(M @ v) - self.grad_psi(q))
        return np.hstack([v, a])

    # ---- energy ----
    def energy(self, x: np.ndarray) -> float:
        q = x[:2]; v = x[2:]
        M = self.M_of_q(q)
        return 0.5*float(v.T @ (M @ v)) + 0.5*self.prm.k_psi*np.linalg.norm(q - self.prm.qg)**2

    # ---- integrator ----
    def rk4(self, x: np.ndarray, h: float) -> np.ndarray:
        f = self.rhs
        k1 = f(x)
        k2 = f(x + 0.5*h*k1)
        k3 = f(x + 0.5*h*k2)
        k4 = f(x + h*k3)
        return x + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    def simulate(self, q0: np.ndarray, v0: np.ndarray,
                 h: float = 1e-3, tmax: float = 20.0, tol: float = 1e-3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            T: (N,), X: (N,4), E: (N,)
        """
        x = np.hstack([q0, v0])
        T = [0.0]; X=[x.copy()]; E=[self.energy(x)]
        for k in range(int(tmax/h)):
            print(f"Simulation step: {k + 1} / {int(tmax/h)}", end="\r", flush=True)
            d,_,_ = dist_n_t(x[:2], self.prm.obs)
            if d < 0:  # collided
                break
            x = self.rk4(x, h)
            T.append(T[-1]+h); X.append(x.copy()); E.append(self.energy(x))
            if np.linalg.norm(x[:2] - self.prm.qg) < tol and np.all(np.abs(x[2:4]) < 1e-4):
                break
        return np.asarray(T), np.asarray(X), np.asarray(E)
    
def _initial_positions_aligned_with_goal(prm: Params, num: int = 5) -> np.ndarray:
    """
    Create a short line segment of starting positions orthogonal to the obstacle-goal axis,
    behind the obstacle, so all starts “face” the hard sector.
    """
    r  = prm.obs.r
    c  = prm.obs.c
    eg = (prm.qg - c)/np.linalg.norm(prm.qg - c)
    base = c - 2.5*r*eg  # go “behind” the obstacle wrt goal
    eperp = np.array([eg[1], -eg[0]])
    alphas = np.linspace(-1.5*r, 1.5*r, num)
    return np.array([base + a*eperp for a in alphas])

def _signed_kappa(v: np.ndarray, a: np.ndarray) -> float:
    sp = np.linalg.norm(v)
    if sp < 1e-10: return 0.0
    return float((v[0]*a[1] - v[1]*a[0]) / (sp**3))

# -------------------------
# Magnetic laws (Strategy)
# -------------------------

class MagneticLaw(ABC):
    """Strategy base for N(q)=b(q)*J."""
    name: str = "base"

    def __init__(self, prm):
        self.prm = prm

    @abstractmethod
    def b_scalar(self, q: np.ndarray) -> float:
        ...

    @staticmethod
    def _agg(vals: list[float]) -> float:
        """
        Per-pixel aggregation over vts: RMS magnitude with mean-sign to avoid random sign flips.
        """
        if not vals: return np.nan
        arr = np.asarray(vals, float)
        return float(np.sqrt(np.mean(arr**2))) * np.sign(np.mean(arr))

    
    def plot_trajectories(self,
                          save_as: str | None = None,
                          plot: bool = True,
                          xlim: Tuple[float, float] = (-2.0, 2.0),
                          ylim: Tuple[float, float] = (-2.0, 2.0),
                          grid: Tuple[int, int] = (90, 90),
                          q0: np.ndarray | None = None,
                          vn: float = 0.00,
                          vt: float = 0.05,
                          n_traj: int = 5) -> tuple[np.ndarray, list, list, list]:
        """
        Plot the simulated trajectories for a magnetic law.

        Args:
            prm:   Params (scenario).
            save_as: path to PNG; if None it might be shown instead (see 'plot').
            plot: show a plot of the trajectories with the color indicating the velocity
            xlim,ylim: plot domain.
            grid: (Nx,Ny) sampling grid.
            vn, vt: normal and tangential initial velocity relative to the obstacle.
            n_traj: number of starting points.
        
        Returns:
            q0: initial positions, size n_traj
            Q_all: n_traj position vectors times N x 2, with N the time vector's length
            V_all: n_traj velocity vectors times N x 2, with N the time vector's length
            T_all: n_traj time vectors.
        """
        prm = self.prm
        system = SecondOrderSystem(prm, self)

        if q0 is None:
            q0 = _initial_positions_aligned_with_goal(prm, num=n_traj)

        Q_all = []
        V_all = []
        T_all = []
        for i, qi in enumerate(q0):
            print(f"q0: {i + 1} / {len(q0)}", flush=True)
            _, n, t = dist_n_t(qi, prm.obs)
            v0 = vn * n + vt * t
            T, Xs, _ = system.simulate(qi, v0, h=0.05, tmax=40.0)
            print("\033[F", end="", flush=True) # go up one line, not robust, because it breaks if other things are printed.
            Q = Xs[:, :2]
            V = Xs[:, 2:4]
            Q_all.append(Q)
            V_all.append(V)
            T_all.append(T)

        generate_figure = plot or save_as is not None

        if generate_figure:
            # Calculate velocities
            vel_mag = [np.linalg.norm(v, axis=1) for v in V_all]
            v_min = np.min([v.min() for v in vel_mag])
            v_max = np.max([v.max() for v in vel_mag])

            fig, ax = plt.subplots(figsize=(10, 7))

            for i, (qi, Q, v) in enumerate(zip(q0, Q_all, vel_mag)):
                # make segments of (x, y) to (x+1, y+1)
                pts = Q.reshape(-1, 1, 2)
                seg = np.concatenate([pts[:-1], pts[1:]], axis=1)

                # make linecollection and couple velocity of point (x, y) to line segment
                lc = LineCollection(seg, cmap='jet', norm=plt.Normalize(v_min, v_max))
                lc.set_array(v[:-1])
                lc.set_linewidth(2)
                ax.add_collection(lc)
                sc = ax.scatter(Q[:,0], Q[:,1], c=v, cmap='jet', s=5, vmin=v_min, vmax=v_max, edgecolor='none')

                # add obstacle
                ax.add_patch(Circle(prm.obs.c, prm.obs.r, facecolor='w', alpha=0.6, edgecolor='k', linewidth=1.0))
                
                # add initial point
                ax.plot(qi[0], qi[1], 'ko', ms=3, zorder=6)

            # colorbar
            sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(v_min, v_max))
            fig.colorbar(sm, ax=ax, label='Velocity (Magnitude) [m/s]')
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')

            ax.plot(prm.qg[0], prm.qg[1], 'y*', ms=10, zorder=50)
            ax.set_aspect('equal')
            plt.tight_layout(); plt.grid(True, alpha=0.2)
            if save_as: plt.savefig(save_as, dpi=180); plt.close(fig)
            else: plt.show()

        return q0, Q_all, V_all, T_all
    
    @staticmethod
    def add_trajectories(trajectories, fig, axes):
        print("Trajectories added...")
        q0, Q_all, V_all, T_all = trajectories

        # obtain the global min and max value of the velocities  
        vel_mag = [np.linalg.norm(v, axis=1) for v in V_all]
        v_min = np.min([v.min() for v in vel_mag])
        v_max = np.max([v.max() for v in vel_mag])

        all_scatters = [[] for _ in range(len(axes.flatten()))]
        all_annots = []

        for j, ax in enumerate(axes.flatten()):
            for i, (qi, Q, v) in enumerate(zip(q0, Q_all, vel_mag)):
                # plot trajectory
                ax.plot(Q[:, 0], Q[:, 1], color='gray', alpha=0.2, lw=1)
                sc = ax.scatter(Q[:,0], Q[:,1], c=v, cmap='jet', s=10, vmin=v_min, vmax=v_max, edgecolor='k', linewidth=0.5, zorder=5)
                all_scatters[j].append(sc)

                # plot initial positions
                ax.plot(qi[0], qi[1], 'ko', ms=3, zorder=6)
            
            # make invisble empty annotation
            annot = ax.annotate("", xy=(0,0), xytext=(10,10), textcoords="offset points", bbox=dict(boxstyle="round", fc="w", alpha=0.8, ec="gray"), arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"), zorder=20, clip_on=False)
            annot.set_visible(False)
            all_annots.append(annot)

        # add hover tooltips to show velocity - time sensitivity included
        hover_timer = QTimer()
        hover_timer.setSingleShot(True)
        last_event = None

        def update_tooltip():
            global last_event
            if not last_event or not last_event.inaxes: return
            
            try:
                ax_idx = axes.flatten().tolist().index(last_event.inaxes)
            except ValueError:
                return
            scatters = all_scatters[ax_idx]
            annot = all_annots[ax_idx]

            for i, sc in enumerate(scatters):
                cont, ind = sc.contains(last_event) # check whether it is on a point
                if cont:
                    # update position and text
                    pos = sc.get_offsets()[ind["ind"][0]]
                    annot.xy = pos
                    speed_val = vel_mag[i][ind["ind"][0]]
                    annot.set_text(f"|v| = {speed_val:.2f} m/s")
                    annot.set_visible(True)
                    fig.canvas.draw_idle() # forced update
                    return
        
        def on_mouse_move(event):
            global last_event
            last_event = event

            # Hide all annotations immediately at movement
            for ann in all_annots:
                if ann.get_visible():
                    ann.set_visible(False)
            fig.canvas.draw_idle()
            
            # start timer: show tooltip if mouse still for 200ms
            hover_timer.stop()
            if event.inaxes:
                hover_timer.start(100)
            
        fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)
        hover_timer.timeout.connect(update_tooltip)
        
        return fig, axes

    

    def plot_curvature_maps(self,
                            save_as: str | None = None,
                            xlim: Tuple[float, float] = (-2.0, 2.0),
                            ylim: Tuple[float, float] = (-2.0, 2.0),
                            grid: Tuple[int, int] = (90, 90),
                            vts: Tuple[float, ...] = (0.6,),
                            with_trajectories: bool = True,
                            trajectories: tuple[np.ndarray, list, list, list] | None = None,
                            n_traj: int = 5,
                            plot_radii: bool = False) -> tuple[np.ndarray, list]:
        """
        Plot a 2×2 grid of signed curvature fields:
          [ total | magnetic ]
          [ geom  | goal     ]
        and optionally overlay trajectories.

        Args:
            prm:   Params (scenario).
            save_as: path to PNG; if None ⇒ show() instead.
            xlim,ylim: plot domain.
            grid: (Nx,Ny) sampling grid.
            vts: grazing speeds to aggregate (RMS with mean sign).
            with_trajectories: overlay trajectories from initial positions aligned with goal.
            trajectories: pre-simulated trajectories with starts and the simulated time series. Generate with 'plot_trajectories'.
            n_traj: number of starting points.
            plot_radii: optional, plot the circles with radius r1 and r2.

        Returns:
            q0: the starting positions of the simulated trajectories. If ´with_trajectories´ is False, None is returned.
            Q_all: a list of np.ndarray's with the simulated positions. If ´with_trajectories´is False, None is returned.
            V_all: a list of np.ndarray's with the simulated velocities. If ´with_trajectories´is False, None is returned.
            T_all: a list of np.ndarray's with the simulated time vectors. If ´with_trajectories´is False, None is returned.
        """
        prm = self.prm
        system = SecondOrderSystem(prm, self)

        Nx, Ny = grid
        xs = np.linspace(xlim[0], xlim[1], Nx)
        ys = np.linspace(ylim[0], ylim[1], Ny)
        XX, YY = np.meshgrid(xs, ys)

        K_tot  = np.full_like(XX, np.nan, dtype=float)
        K_B    = np.full_like(XX, np.nan, dtype=float)
        K_geom = np.full_like(XX, np.nan, dtype=float)
        K_goal = np.full_like(XX, np.nan, dtype=float)

        for i in range(Nx):
            for j in range(Ny):
                q = np.array([XX[j,i], YY[j,i]])
                d, n, t = dist_n_t(q, prm.obs)
                if d <= 0:  # inside obstacle
                    continue

                kt_list, kB_list, kGm_list, kGl_list = [], [], [], []
                for vt in vts:
                    v = vt * t
                    M = system.M_of_q(q); Minv = np.linalg.inv(M)
                    Cv = system.C_times_qdot(q, v); N = system.N_of_q(q)
                    g  = system.grad_psi(q)

                    a_B    = Minv @ (N @ v)
                    a_geom = - Minv @ (Cv)
                    a_goal = - Minv @ (g)
                    a_tot  = a_B + a_geom + a_goal - prm.c_damp * v  # damping adds zero signed curvature

                    kB  = _signed_kappa(v, a_B)
                    kGm = _signed_kappa(v, a_geom)
                    kGl = _signed_kappa(v, a_goal)
                    kT  = _signed_kappa(v, a_tot)

                    kt_list.append(kT); kB_list.append(kB); kGm_list.append(kGm); kGl_list.append(kGl)
                
                K_tot[j,i]  = self._agg(kt_list)
                K_B[j,i]    = self._agg(kB_list)
                K_geom[j,i] = self._agg(kGm_list)
                K_goal[j,i] = self._agg(kGl_list)

        # color scaling (diverging, centered at 0)
        stack = np.vstack([K_tot.ravel(), K_B.ravel(),
                           K_geom.ravel(), K_goal.ravel()])
        kmax = np.nanmax(stack, axis=1)
        kmax = [max(k, 1e-4) for k in kmax]
        kmin = np.nanmin(stack, axis=1)
        kmin = [min(k, -1e-4) for k in kmin]

        fig, axes = plt.subplots(2, 2, figsize=(11.2, 9.2))
        mats   = [K_tot, K_B, K_geom, K_goal]
        titles = [r'$\kappa_{\rm s,tot}$', r'$\kappa_{\rm s,B}$',
                  r'$\kappa_{\rm s,geom}$', r'$\kappa_{\rm s,goal}$']

        for i, (ax, M, T, kma, kmi) in enumerate(zip(axes.flatten(), mats, titles, kmax, kmin)):
            norm = TwoSlopeNorm(vcenter=0.0, vmin=kmi, vmax=kma)
            im = ax.imshow(M, origin='lower', extent=[xlim[0], xlim[1], ylim[0], ylim[1]], cmap='RdBu_r', norm=norm)
            ax.add_patch(Circle(prm.obs.c, prm.obs.r, facecolor='w', alpha=0.6, edgecolor='k', linewidth=1.0))

            if plot_radii:
                try:
                    ax.add_patch(Circle(prm.obs.c, self.r1, facecolor='none', alpha=0.6, edgecolor='g', linewidth=1.0))
                    ax.add_patch(Circle(prm.obs.c, self.r2, facecolor='none', alpha=0.6, edgecolor='g', linewidth=1.0))
                except AttributeError: 
                    # self.r1 or self.r2 does not exist
                    pass

            # add colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='RdBu_r'),
                            ax=ax, cax=cax, shrink=0.92, pad=0.02)
            cbar.set_label('signed curvature  [1/m]')
            ax.set_aspect('equal'); ax.set_title(T); ax.grid(True, alpha=0.2)

        # overlay trajectories
        if with_trajectories:
            if trajectories is None:
                trajectories = self.plot_trajectories(plot=False, xlim=xlim, ylim=ylim, grid=grid, q0=None, vn=0.0, vt=0.05, n_traj=n_traj)

            q0, Q_all, V_all, T_all = trajectories

            # add trajectories with annotation of speed
            self.add_trajectories(trajectories, fig, axes)
     
        else:
            q0 = None; Q_all = None; V_all = None; T_all = None
        
        for ax in axes.flatten():
            ax.plot(prm.qg[0], prm.qg[1], 'y*', ms=20, zorder=19)

        plt.tight_layout()
        if save_as: plt.savefig(save_as, dpi=180); plt.close(fig)
        else: plt.show()

        return q0, Q_all, V_all, T_all


    def plot_grazing_normal_maps(self, 
                                 save_as: str | None = None,
                                 xlim: tuple[float,float] = (-2.0, 2.0),
                                 ylim: tuple[float,float] = (-2.0, 2.0),
                                 grid: tuple[int,int] = (90, 90),
                                 vts: tuple[float,...] = (0.6,),
                                 with_trajectories: bool = True,
                                 trajectories: tuple[np.ndarray, list, list, list] | None = None,
                                 n_traj: int = 5,
                                 axis_gap: float | None = None,
                                 goal_conditioned: bool = True,
                                 plot_radii:bool = False) -> tuple[np.ndarray, list]:
        
        """
        Visual debugger for the grazing boundary condition n·a on a grid.
        Renders a 2×2 panel (signed, diverging colormap):
          [ n·a_tot | n·a_B ]
          [ n·a_geom| n·a_goal ]
        Each pixel q uses grazing v = vt * t(q) for vt in vts (RMS magnitude with mean sign).

        Args:
            prm: scenario parameters.
            save_as: PNG path (if None: show()).
            xlim,ylim: domain in meters.
            grid: (Nx,Ny) sampling resolution.
            vts: list of grazing speeds to aggregate.
            with_trajectories: overlay trajectories from _initial_positions_aligned_with_goal.
            trajectories: pre-simulated trajectories with starts and the simulated time series. Generate with 'plot_trajectories'.
            n_traj: how many starting points.
            axis_gap: optionally ignore a wedge |θ|<axis_gap (rad) around the goal axis to avoid window=0 pixels dominating (useful for sine/sine_rphase).
            goal_conditioned : bool, use the Nagumo‑like, policy‑conditioned grazing speed, or in the case of False use v = v_t * t.
            plot_radii : optional, plot the circles with radius r1 and r2. 

        Returns:
            q0: the starting positions of the simulated trajectories. If ´with_trajectories´ is False, None is returned.
            Q_all: a list of np.ndarray's with the simulated positions. If ´with_trajectories´is False, None is returned.
            V_all: a list of np.ndarray's with the simulated velocities. If ´with_trajectories´is False, None is returned.
            T_all: a list of np.ndarray's with the simulated time vectors. If ´with_trajectories´is False, None is returned.
        """
        prm = self.prm
        system = SecondOrderSystem(prm, self)

        Nx, Ny = grid
        xs = np.linspace(xlim[0], xlim[1], Nx)
        ys = np.linspace(ylim[0], ylim[1], Ny)
        XX, YY = np.meshgrid(xs, ys)

        NA_tot = np.full_like(XX, np.nan, dtype=float)
        NA_B = np.full_like(XX, np.nan, dtype=float)
        NA_geom = np.full_like(XX, np.nan, dtype=float)
        NA_goal = np.full_like(XX, np.nan, dtype=float)

        for i in range(Nx):
            for j in range(Ny):
                q = np.array([XX[j, i], YY[j, i]])
                d, n, t = dist_n_t(q, prm.obs)
                if d <= 0: # inside obstacle
                    continue
                if axis_gap is not None:
                    th = theta_rel_goal(q, prm.qg, prm.obs.c)
                    if abs(th) < axis_gap or abs(abs(th) - np.pi) < axis_gap:
                        continue
                
                nat_list, nab_list, nageom_list, nagoal_list = [], [], [], []
                for vt in vts:
                    if goal_conditioned:
                        # choose the tangential direction that makes progress toward the goal
                        sgn_goal = np.sign( t @ (-system.grad_psi(q)) )  # projection of -∇ψ on t
                        v = vt * ( sgn_goal * t if sgn_goal != 0 else t )
                    else:
                        v = vt * t

                    M = system.M_of_q(q); Minv = np.linalg.inv(M)
                    Cv=  system.C_times_qdot(q, v)
                    N = system.N_of_q(q)
                    g = system.grad_psi(q)

                    aB = Minv @ (N @ v)
                    ageom = -Minv @ Cv
                    agoal = -Minv @ g
                    a = aB + ageom + agoal - prm.c_damp * v

                    nab_list.append(float(n @ aB))
                    nageom_list.append(float(n @ ageom))
                    nagoal_list.append(float(n @ agoal))
                    nat_list.append(float(n @ a))

                NA_tot[j,i] = self._agg(nat_list)
                NA_B[j,i] = self._agg(nab_list)
                NA_geom[j,i] = self._agg(nageom_list)
                NA_goal[j,i] = self._agg(nagoal_list)

        stack = np.vstack([NA_tot.ravel(), NA_B.ravel(),
                           NA_geom.ravel(), NA_goal.ravel()])
        namax = np.nanmax(stack, axis=1)
        namax = [max(na, 1e-4) for na in namax]
        namin = np.nanmin(stack, axis=1)
        namin = [min(na, -1e-4) for na in namin]

        fig, axes = plt.subplots(2, 2, figsize=(11.2, 9.2))
        mats = [NA_tot, NA_B, NA_geom, NA_goal]
        titles = [r'$n\!\cdot a_{\rm tot}$', r'$n\!\cdot a_{B}$',
                  r'$n\!\cdot a_{\rm geom}$', r'$n\!\cdot a_{\rm goal}$']
        
        for ax, M, T, nami, nama in zip(axes.flatten(), mats, titles, namin, namax):
            norm = TwoSlopeNorm(vcenter=0.0, vmin=nami, vmax=nama)
            im = ax.imshow(M, origin='lower', extent=[xlim[0], xlim[1], ylim[0], ylim[1]], cmap='RdBu_r', norm=norm)
            ax.add_patch(Circle(prm.obs.c, prm.obs.r, facecolor='w', alpha=0.6, edgecolor='k', linewidth=1.0))
            if plot_radii:
                try:
                    ax.add_patch(Circle(prm.obs.c, self.r1, facecolor='none', alpha=0.6, edgecolor='g', linewidth=1.0))
                    ax.add_patch(Circle(prm.obs.c, self.r2, facecolor='none', alpha=0.6, edgecolor='g', linewidth=1.0))
                except AttributeError:
                    # self.r1 or self.r2 does not exist
                    pass

            # zero contour for visual boundary of sign flip
            try:
                cs = ax.contour(XX, YY, M, levels=[0.0], colors='g', linewidths=1.5, alpha=0.8)
            except Exception:
                pass

            # add colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='RdBu_r'), ax=ax, cax=cax, shrink=0.92, pad=0.02)
            cbar.set_label(r'Grazing condition $<n\cdot a>$ [m/s²]')
            ax.set_aspect('equal'); ax.set_title(T); ax.grid(True, alpha=0.2)

        if with_trajectories:
            if trajectories is None:
                trajectories = self.plot_trajectories(plot=False, xlim=xlim, ylim=ylim, grid=grid, q0=None, vn=0.0, vt=0.05, n_traj=n_traj)

            q0, Q_all, V_all, T_all = trajectories

            # add trajectories with annotation of speed
            print("Adding Trajectories...", end="\r")
            self.add_trajectories(trajectories, fig, axes)
        else:
            q0 = None; Q_all = None; V_all = None; T_all = None

        for ax in axes.flatten():
            ax.plot(prm.qg[0], prm.qg[1], 'y*', ms=20, zorder=50)

        plt.tight_layout()
        if save_as: plt.savefig(save_as, dpi=180); plt.close(fig)
        else: plt.show()

        return q0, Q_all, V_all, T_all
    
    def predict_RA(self,
                   mode: Literal['tangent', 'headon'] = 'headon',
                   vt_list: Tuple[float,...] = (0.5, 1.0, 1.5),
                   vn_list: Tuple[float,...] = (0.5, 1.0, 1.5),
                   n_theta: int = 180,
                   axis_choice: Literal['goal', 'opp', 'both'] = 'both',
                   plot: bool = True,
                   with_trajectories: bool = True,
                   trajectories: tuple[np.ndarray, list, list, list] | None = None,
                   n_traj: int = 5,
                   save_as: str | None = None):
        """
        Predict (conservatively, without simulating) which launch conditions at r=r2
        are safe (do not hit the obstacle), using closed-form integrals of b(q).

        mode='tangent'  : sweep launch angle θ0 ∈ [-π,π), speeds vt_list (grazing).
                          Safety margin: Δψ_B^tang - ψ_req(Γ).
        mode='headon'   : evaluate axis launches (goal / goal-opposing) with normal
                          speeds vn_list. Safety margin: Δψ_B^head - π/2.

        Returns:
            dict with grids and safe masks; and produces an explanatory plot if plot=True.

        """
        prm = self.prm
        # --- ensure ring r2 is defined (only tilted sine and sine_rphase) ---
        r_obs = prm.obs.r
        r1 = getattr(self, 'r1', None)
        r2 = getattr(self, 'r2', None)
        if (r1 is None) or (r2 is None):
            raise SyntaxError("Only valid for tilted sine and sine_rphase.")
        
        d2 = float(r2 - r_obs)
        m0 = prm.m0 # TODO: include if alpha != 0

        # --- helper: integrate b along angle on ring (for 'tangent' mode) ---
        def angle_integral_b(theta0: float, dtheta:float) -> float:
            """
            Integrate b(r2,theta) over theta ∈ [theta0, theta0 + dtheta] (signed dtheta). Uses trapezoidal rule on a small grid – not a time simulation of the ODE, just an integral of the analytic b_scalar.
            """
            n_int = max(32, int(abs(dtheta) / (np.pi/180))) # ~ 1.0 deg steps
            thetas = np.linspace(theta0, theta0 + dtheta, n_int)
            qs = prm.obs.c + r2 * np.column_stack([np.cos(thetas), np.sin(thetas)])
            vals = np.array([self.b_scalar(q) for q in qs], dtype=float)
            return float(np.trapz(vals, thetas))
        
        # ---- helper: integrate b along radius on axis ray ('headon' mode)---
        def radial_integral_b(theta_axis: float) -> float:
            """
            Integrate b(r, theta_axis) over r in [r_obs, r2].
            """
            n_int = 256 # 512
            rs = np.linspace(r_obs, r2, n_int)
            qs = prm.obs.c + np.column_stack([rs * np.cos(theta_axis), rs * np.sin(theta_axis)])
            vals = np.array([self.b_scalar(q) for q in qs], dtype=float)
            return float(np.trapz(vals, rs))
        
        out = dict(mode=mode, r1=r1, r2=r2, safe_mask=None)

        if mode == 'tangent':
            # sample theta0 uniformly around the circle
            thetas0 = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
            # determine active angular sector length Δtheta_\Gamma for each theta0
            beta_allow = getattr(self, 'beta_allow', np.pi/2) if hasattr(self, '_W_theta') else np.pi/2
            dtheta_sector = 2.0 * beta_allow

            margin = np.zeros((len(vt_list), len(thetas0)))
            for i_vt, vt in enumerate(vt_list):
                for j, th0 in enumerate(thetas0):
                    print(f"[{len(vt_list)} | {len(thetas0)} // {i_vt + 1} | {j + 1}]", end='\r')
                    # int(b dtheta) over sector; \grad psi_B^tang = (r2/(m0*vt)) int(b dtheta)
                    Ib = angle_integral_b(th0, dtheta_sector) 
                    dpsi = (r2 / (m0 * vt)) * Ib
                    # required clear-angle to avoid inward drift through sector
                    dpsi_req = max(0.0, d2 / (r2 * dtheta_sector))
                    margin[i_vt, j] = dpsi - dpsi_req

            # build boolean mask of safe (margin >= 0)
            safe_mask = (margin >= 0.0)
            out.update(dict(theta0=thetas0, vt_list=vt_list, margin=margin, safe_mask=safe_mask))

            if plot:
                self._plot_RA_tangent(thetas0, vt_list, margin, safe_mask, r2, with_trajectories, trajectories, n_traj, save_as)

        elif mode == 'headon':
            # choose axis angle to test
            axes = []
            if axis_choice in ('goal', 'both'):
                axes.append(0.0) # theta = 0 goal axis
            if axis_choice in ('opp', 'both'):
                axes.append(np.pi) # theta = pi goal-opposing axis

            margin = np.zeros((len(vn_list), len(axes)))
            for i_vn, vn in enumerate(vn_list):
                for j, th_ax in enumerate(axes):
                    print(f"[{len(vn_list)} | {len(axes)} // {i_vn + 1} | {j + 1}]", end='\r')
                    # int(b dr) for R to r2; \grad \psi_B^head = (1/m0*vn)) * int(b dr)
                    Ib_r = radial_integral_b(th_ax)
                    dpsi = (1.0 / (m0*vn)) * Ib_r
                    margin[i_vn, j] = dpsi - (0.5*np.pi) # need at least Pi/2

            safe_mask = (margin >= 0.0)
            out.update(dict(axes=np.array(axes), vn_list=vn_list, margin=margin, safe_mask=safe_mask))

            if plot:
                self._plot_RA_headon(np.array(axes), vn_list, margin, safe_mask, r2, with_trajectories, trajectories, n_traj, save_as)

        else:
            raise ValueError("Mode must be 'tangent' or 'headon'.")
        
        return out
    
    # --- plotting helpers for RA predictor ----
    def _plot_RA_tangent(self, thetas0, vt_list, margin, safe_mask, r2, with_trajectories, trajectories, n_traj, save_as):
        prm = self.prm
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))

        # heatmap margin(vt, theta)
        im = axes[0].imshow(margin, aspect='auto', origin='lower', extent=[thetas0[0], thetas0[-1], vt_list[0], vt_list[-1]], cmap='RdBu_r', vmin=-np.max(abs(margin)), vmax=np.max(abs(margin)))

        axes[0].set_xlabel(r'Launch angle $\theta_0$ on $r=r_2$')
        axes[0].set_ylabel(r'$v_t$ [m/s]')
        axes[0].set_title(r'$\Delta\psi^{\rm tang}_B - \psi_{\rm req}$  (safe ≥ 0)')

        # add colorbar
        divider = make_axes_locatable(axes[0])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-np.max(abs(margin)), vmax=np.max(abs(margin)))
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='RdBu_r'),
                            ax=axes[0], cax=cax, shrink=0.92, pad=0.02)
        cbar.set_label('margin [rad]')
        
        # map view: safe/unsafe rays at r=r2
        ax = axes[1]
        ths = thetas0
        xs = prm.obs.c[0] + r2*np.cos(ths)
        ys = prm.obs.c[1] + r2*np.sin(ths)
        ax.add_patch(Circle(prm.obs.c, prm.obs.r, facecolor='w', alpha=0.6,
                            edgecolor='k', linewidth=1))
        ax.plot(prm.qg[0], prm.qg[1], 'y*', ms=16, zorder=10)
        # mark safe sectors for a representative vt (e.g., middle index)
        mid = len(vt_list)//2
        for j, th in enumerate(ths):
            col = 'tab:green' if safe_mask[mid, j] else 'tab:red'
            ax.plot([prm.obs.c[0], xs[j]], [prm.obs.c[1], ys[j]], color=col, alpha=0.6, lw=1.6)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
        ax.set_title(f'safe rays at r2 (vt≈{vt_list[mid]:.2f} m/s)')
        plt.tight_layout()

        if with_trajectories:
            if trajectories is None:
                trajectories = self.plot_trajectories(plot=False, q0=None, vn=0.0, vt=0.05, n_traj=n_traj)

            # q0, Q_all, V_all, T_all = trajectories

            # add trajectories with annotation of speed
            print("Adding Trajectories...", end="\r")
            self.add_trajectories(trajectories, fig, np.array([ax]))

        if save_as: 
            plt.savefig(save_as.replace('.png', '_tangent_RA.png'), dpi=180)
            plt.close(fig)
        else:
            plt.show()

    def _plot_RA_headon(self, axes_list, vn_list, margin, safe_mask, r2, with_trajectories, trajectories, n_traj, save_as):
        prm = self.prm

        fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))

        # heatmap: margin (vn, axis)
        ax0 = axes[0]
        im = ax0.imshow(margin, aspect="auto", origin='lower', extent=[0, len(axes_list) - 1, vn_list[0], vn_list[-1]], cmap='RdBu_r', vmin=-np.max(abs(margin)), vmax=np.max(abs(margin)))
        ax0.set_xticks(range(len(axes_list)))
        ax0.set_xticklabels(['goal' if abs(a) < 1e-6 else 'opp' for a in axes_list])
        ax0.set_ylabel(r'$v_n$ [m/s]')
        ax0.set_title(r'$\Delta\psi^{\rm head}_B - \pi/2$ (safe ≥ 0)')

        # add colorbar
        divider = make_axes_locatable(axes[0])
        cax = divider.append_axes("right", size="5%", pad=0.1)
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-np.max(abs(margin)), vmax=np.max(abs(margin)))
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='RdBu_r'),
                            ax=axes[0], cax=cax, shrink=0.92, pad=0.02)
        cbar.set_label('margin [rad]')

        # map view: draw axis rays and color by safety for a representative v_n
        mid = len(vn_list) // 2
        ax = axes[1]
        ax.add_patch(Circle(prm.obs.c, prm.obs.r, facecolor='w', alpha=0.6,
                            edgecolor='k', linewidth=1))
        ax.plot(prm.qg[0], prm.qg[1], 'y*', ms=16, zorder=10)
        for j, th in enumerate(axes_list):
            col = 'tab:green' if safe_mask[mid, j] else 'tab:red'
            x = prm.obs.c[0] + r2*np.cos(th)
            y = prm.obs.c[1] + r2*np.sin(th)
            ax.plot([prm.obs.c[0], x], [prm.obs.c[1], y], color=col, lw=2)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
        ax.set_title(f'head-on safety at r2 (vn = {vn_list[mid]:.2f} m/s)')
        plt.tight_layout()

        if with_trajectories:
            if trajectories is None:
                trajectories = self.plot_trajectories(plot=False, q0=None, vn=0.0, vt=0.05, n_traj=n_traj)

            # q0, Q_all, V_all, T_all = trajectories

            # add trajectories with annotation of speed
            print("Adding Trajectories...", end="\r")
            self.add_trajectories(trajectories, fig, np.array([ax]))

        if save_as: 
            plt.savefig(save_as.replace('.png', '_headon_RA.png'), dpi=180)
            plt.close(fig)
        else:
            plt.show()


class RoundMagneticLaw(MagneticLaw):
    name = "baseRound"

    def __init__(self, 
                 prm: Params,
                 r1: Optional[float] = None,
                 r2: Optional[float] = None,
                 tau_far: Optional[float] = 1e-3,
                 tau_near: Optional[float] = 1e-3,
                 min_width: Optional[float] = 0.04,
                 max_width: Optional[float] = 0.50):
        super().__init__(prm)
        self.r1 = r1; self.r2 = r2
        self._annulus_ready = (r1 is not None and r2 is not None)

        if not self._annulus_ready:
            self.auto_annulus(tau_far, tau_near, min_width, max_width)

    # ---- auto selection of (r1,r2) ----    
    def auto_annulus(self,
                     tau_far: float=1e-3,
                     tau_near: float = 1e-3,
                     min_width: float = 0.04,
                     max_width: float = 0.50,
                     verbose: bool = False) -> tuple[float,float]:
        
        """
        Simple analytic estimator for (r1,r2):
          • r2 from φ_far(d2)=tau_far. d_on,q: φ_far(d, q)=exp(-(d/d_on)^q) parameters.
          • r1 from d1^p = tau_near  => d1 = tau_near^(1/p).
        Clamps the annulus width to [min_width, max_width].

        Args:
            tau_far: far-field cutoff (0<tau_far<1).
            tau_near: near-boundary cutoff for d^p (0<tau_near<1).
            min_width,max_width: thickness clamp (meters).
            verbose: print r1 and r2.

        Returns:
            (r1,r2)
        """
        prm = self.prm
        # outer
        d2 = prm.d_far * (-np.log(max(tau_far, 1e-12))) ** (1.0/max(prm.q_far, 1e-6))
        # inner
        d1 = max(1e-6, tau_near**(1.0/max(prm.p, 1e-6)))

        # convert to radii
        r_obs = prm.obs.r
        r2 = r_obs + d2
        r1 = r_obs + d1
            
        # ensure thickness in [min_width, max_width]
        width = r2 - r1
        if width < min_width:
            # push outward if possible
            center = 0.5*(r1 + r2)
            r1 = center - 0.5*min_width
            r2 = center + 0.5*min_width
            if r1 < r_obs + 1e-3:
                r1 = r_obs + 1e-3
                r2 = r1 + min_width
        elif width > max_width:
            center = 0.5*(r1 + r2)
            r1 = center - 0.5*max_width
            r2 = center + 0.5*max_width

        self.r1, self.r2 = float(r1), float(r2)
        self._annulus_ready = True
        if verbose:
            print(f" r1: {r1}\n r2:{r2}", flush=True)
        return self.r1, self.r2


class NoMagnetic(MagneticLaw):
    name = "none"
    
    @classmethod
    def b_scalar(cls, q: np.ndarray) -> float:
        return 0.0 * q

class ConstMagnetic(MagneticLaw):
    """
    b(d)=kB * φ_far(d).
    """
    name = "const"

    def __init__(self, prm: Params):
        super().__init__(prm)

    def b_scalar(self, q: np.ndarray) -> float:
        prm = self.prm
        d,_,_ = dist_n_t(q, prm.obs)
        return prm.kB * phi_window_far(d, d_on=prm.d_far, q=prm.q_far)

class PowerMagnetic(RoundMagneticLaw):
    """
    b(d)=kB * d^p * φ_far(d).
    """
    name = "dp"

    def __init__(self, 
                 prm: Params,
                 r1: Optional[float] = None,
                 r2: Optional[float] = None):
        super().__init__(prm, r1, r2)
        self.cst_law = ConstMagnetic(self.prm)


    def b_scalar(self, q: np.ndarray) -> float:
        prm = self.prm
        d,_,_ = dist_n_t(q, prm.obs)
        cst = self.cst_law.b_scalar(q)
        
        return cst * ((max(d,1e-6) + prm.eps_b)**prm.p)

class SignedPowerMagnetic(RoundMagneticLaw):
    """
    
    """
    pass

class SineMagnetic(RoundMagneticLaw):
    """
    Single-lobe sine in angle: b(d,θ)=kB d^p φ_far(d) * [(1-S(d)) + S(d) sin θ],
    where S(d)=switch that keeps outward sign in the very near field.
    """
    name = "sine"

    def __init__(self, 
                 prm: Params,
                 r1: Optional[float] = None,
                 r2: Optional[float] = None,
                 d_sw: Optional[float] = 0.06, 
                 w_sw: Optional[float] = 0.02):
        self.d_sw = d_sw
        self.w_sw = w_sw
        super().__init__(prm, r1, r2)
        self.power_law = PowerMagnetic(self.prm, self.r1, self.r2)

    def _switch(self, d: float) -> float:
        """
        smooth radial switch S(d) : 0 near obstacle, -> 1 after d_sw
        """
        x = (max(d,0.0) - self.d_sw) / max(self.w_sw, 1e-6)
        return 0.5*(1.0 + np.tanh(x))

    def b_scalar(self, q: np.ndarray) -> float:
        prm = self.prm
        d,_,_ = dist_n_t(q, prm.obs)
        S = self._switch(d)
        th = theta_rel_goal(q, prm.qg, prm.obs.c)
        base = self.power_law.b_scalar(q)
        return base * ((1.0 - S) + S*np.sin(th))
    

class TiltedSineMagnetic(RoundMagneticLaw):
    """
    Tilted (biased) phase-swept sine on a thin annulus, plus an optional 'knee-cap':
        b_total = b_tilted_sine + b_kneecap,
    with:
        b_tilted(q) = kB d^p * S_ff(d) * A(r) * [ eps0 + a1(r) sin(θ+φ(r)) ] * W(θ),
        b_kneecap(q) = kB_cap * d^p * S_ff_cap(d) * A_cap(r) * W_cap(theta) + s_outward,

    where:
      • S_ff(d)  : far-field window φ_far(d).
      • A(r)     : annular Gaussian centered at r_m=(r1+r2)/2 with width σ_r.
      • W(θ)     : angular weight with floor w_min (aims the hard semicircle; removes axis corridors).
      • φ(r)     : phase sweep (same as in SineRPhaseMagnetic).
      • eps0     : smooth DC bias to avoid sign flips / net-zero along radial rays.
      • Knee-cap : a tiny, single-sign bump centered at r_cap > r2, narrow wedge around the obstacle-goal axis.
      
    Use auto_annulus(prm, ...) once to set (r1,r2) or set them manually.
    """
    name = "tilted_sine"


    def __init__(self,
                 prm: Params,
                 r1: Optional[float] = None,
                 r2: Optional[float] = None,
                 phi_max: float = np.pi/3,
                 sigma_frac: float = 0.35,
                 use_tanh: bool = False,
                 k_phase: float = 4.0,
                 gamma: float = 2.0, w_min: float = 0.0,
                 eps0: float = 0.15,
                 enable_knee: bool = True,
                 delta_cap: float = 0.15, # knee-cap center offset: r_cap = r2 + delta_cap
                 sigma_cap: float = 0.02, # knee-cap radial width (m)
                 kB_cap_rel: float = 0.15, # knee-cap amplitude relative to kB
                 theta_cap: float = np.pi/6, # half-width of angular width (rad) around theta=0
                 q_far_cap: float = 2.0,
                 d_far_cap: float = 0.45
                 ):
        self.phi_max = phi_max
        self.sigma_frac = sigma_frac
        self.use_tanh = use_tanh
        self.k_phase = k_phase
        self.gamma, self.w_min = gamma, w_min
        self.eps0 = eps0
        self._annulus_ready = (r1 is not None and r2 is not None)
        
        self.enable_knee = enable_knee
        self.delta_cap, self.sigma_cap = delta_cap, sigma_cap
        self.kB_cap_rel = kB_cap_rel
        self.theta_cap = theta_cap
        self.q_far_cap, self.d_far_cap = q_far_cap, d_far_cap

        super().__init__(prm)
        self.base_law = PowerMagnetic(prm, self.r1, self.r2)


    # ---- helper pieces ----
    @staticmethod
    def _A_gauss(r: float, r1: float, r2: float, sigma_frac= float) -> float:
        rm = 0.5 * (r1 + r2)
        sigma = sigma_frac * max(r2 - r1, 1e-9)
        return float(np.exp(-((r - rm)**2)/(2*sigma**2)))
    
    def _A_gauss_main(self, r):
        return self._A_gauss(r, self.r1, self.r2, self.sigma_frac)
    
    @staticmethod
    def _wrap_pi(th: float) -> float:
        """Wrap angle to (-pi, pi]."""
        return (th + np.pi) % (2*np.pi) - np.pi

    def _W_theta(self, th: float, use_beta_allow: bool = True, beta_allow: float = np.pi/2) -> float:
        """
        Angular weight that favors the goal-opposing side (±π) with a controllable
        angular reach measured from the perpendicular axis (±π/2) toward ±π.

        Args
        ----
        th : float
            Angle w.r.t. obstacle→goal axis. th=0 points toward the goal; th=±π is goal-opposing.
        use_beta_allow : bool, default False
            Whether you want to use the feature of beta_allow.
        beta_allow : float, default π/2
            Allowed angular extent (in radians) from the perpendicular axis toward the goal-opposing axis.
            Range: (0, π/2]. Smaller values restrict the active region to a thin wedge just beyond ±π/2.

        Returns
        -------
        float
            A smooth weight in [w_min, 1], with:
            • ≈ w_min on the goal side and deep in the far side (beyond the allowed reach),
            • raised smoothly from w_min→1 within the allowed wedge of width beta_allow,
            • symmetric across the two far-side halves (around ±π).
        """
        if use_beta_allow:
            th = self._wrap_pi(th)
            delta_opp = abs(self._wrap_pi(th - np.pi)) # distance to +pi (same as -pi after wrap)
            delta = max(0.0, np.pi/2 - delta_opp) # delta = 0 at +-pi/2 and everywhere on goal side; increases to pi/2

            # Normalize by the allowed reach beta_allow
            beta = max(1e-9, min(beta_allow, np.pi/2))
            u = np.clip(delta/beta, 0.0, 1.0)

            # smooth raised cosine ramp on [0, beta]; u = 0 -> w_min (at perpendicular); u=1 -> 1.0 (at inner edge of allowed wedge)
            bump = (0.5* (1.0 - np.cos(np.pi * u))) ** self.gamma
        else:
            bump = (0.5 * (1.0 - np.cos(np.pi)))**self.gamma

        return float(self.w_min + (1.0 - self.w_min) * bump)
    
    def _phi_r(self, r: float) -> float:
        rm = 0.5 * (self.r1 + self.r2)
        s = (r - rm) / max(0.5*(self.r2 - self.r1), 1e-9)
        s = float(np.clip(s, -1.0, 1.0))
        if self.use_tanh:
            return self.phi_max*np.tanh(self.k_phase*s) / np.tanh(self.k_phase)
        return self.phi_max*s
    
    # ---- helpers knee-cap -----
    def _S_ff_cap(self, d: float) -> float:
        return phi_window_far(d, d_on=self.d_far_cap, q=self.q_far_cap)
    
    def _A_gauss_cap(self, r: float) -> float:
        # r2 - r1 = 2 | 0.5*(r1 + r2) = r_cap
        r_cap = float(self.r2) + self.delta_cap
        return self._A_gauss(r, r_cap-0.5, r_cap+0.5, self.sigma_cap)
    
    def _W_cap_theta(self, th: float) -> float:
        # raised cosine wedge around theta=0 (gaol axis)
        def bump(x): # C^1 compact bump in [-theta_cap, theta_cap]
            if abs(x) > self.theta_cap: return 0.0
            u = 0.5 * (1 + np.cos(np.pi*x/self.theta_cap)) # cos window
            return float(u*u) # slightly sharper
        val = bump(abs(abs(th) - np.pi))
        return val
    
    def kneecap(self,
                delta_cap: float | None = None,
                sigma_cap: float | None = None) -> tuple[float, float]:
        """Return (r_cap, sigma_cap) for the knee-cap: r_cap = r2 + delta_cap"""
        if delta_cap is not None: self.delta_cap = delta_cap
        if sigma_cap is not None: self.sigma_cap = sigma_cap
        return float(self.r2) + self.delta_cap, self.sigma_cap
    

    def b_scalar(self, q: np.ndarray) -> float:
        prm = self.prm
        if not self._annulus_ready:
            self.auto_annulus(tau_far=1e-2, tau_near=1e-3, min_width=0.04, max_width=0.50, verbose=False)

        d, _, _ = dist_n_t(q, prm.obs)
        qc = q - prm.obs.c; r = np.linalg.norm(qc)
        th = theta_rel_goal(q, prm.qg, prm.obs.c)
        A = self._A_gauss_main(r)
        W = self._W_theta(th, beta_allow=np.pi/3)
        phi = self._phi_r(r)

        a1 = A # Sine lobe on the annulus
        base = self.base_law.b_scalar(q)

        field_main = base * (self.eps0 + a1*np.sin(th + phi)) * W

        if not self.enable_knee:
            return field_main
        
        # outward normal grazing requires b <= 0 in the knee cap
        s_outward = -1.0
        S_ff_cap = self._S_ff_cap(d)
        A_cap = self._A_gauss_cap(r)
        W_cap = self._W_cap_theta(th)

        field_knee = (self.kB_cap_rel * prm.kB) * (max(d, 1e-6)**prm.p) * S_ff_cap * A_cap * W_cap * s_outward

        return field_main + field_knee



class SineRPhaseMagnetic(RoundMagneticLaw):
    """
    Radially phase-swept sine on an annulus r∈[r1,r2]:
        b(d,θ,r) = kB d^p w_ann(r) φ_far(d) sin( θ + φ(r) ),
        with φ(r1)=-φ_max, φ(rm)=0, φ(r2)=+φ_max, rm=(r1+r2)/2.

    (r1,r2) can be auto-selected from a diagnostic.
    """
    name = "sine_rphase"

    def __init__(self,
                 prm: Params,
                 r1: Optional[float] = None,
                 r2: Optional[float] = None,
                 phi_max: float = np.pi/2,
                 sigma_frac: float = 0.35,
                 use_tanh: bool = False,
                 k: float = 4.0,
                 sector_only: bool = True):

        self.phi_max = phi_max
        self.sigma_frac = sigma_frac
        self.use_tanh = use_tanh
        self.k = k
        self.sector_only = sector_only
        self._annulus_ready = (r1 is not None and r2 is not None)

        super().__init__(prm, r1, r2)
        self.base_law = PowerMagnetic(prm, self.r1, self.r2)

    # ---- annulus utils ----
    @staticmethod
    def _w_ann(theta: float, gamma: float = 2.0) -> float:
        """
        Smooth angular window that kills the donut on the goal side. theta is measured from the obstacle -  goal axis.
        """
        return float(((1.0 - np.cos(theta)) * 0.5)**gamma)

    def _phi_of_r(self, r: float) -> float:
        r1 = self.r1; r2 = self.r2
        rm = 0.5*(r1 + r2)
        s  = (r - rm) / max(0.5*(r2 - r1), 1e-9) # s \in [-1, 1]
        s  = float(np.clip(s, -1.0, 1.0))
        if self.use_tanh:
            return self.phi_max * np.tanh(self.k*s) / np.tanh(self.k)
        return self.phi_max * s

    # ---- law ----
    def b_scalar(self, q: np.ndarray) -> float:
        # ensure annulus
        # d_on, q_far = 0.35, 2.0 # phi_window_far parameters
        prm = self.prm
        if self.r1 is None or self.r2 is None:
            self.auto_annulus(tau_far=1e-2, tau_near=1e-2, min_width=0.04, max_width=0.50, verbose=False)
            
        qc = q - prm.obs.c; r = np.linalg.norm(qc)
        th = theta_rel_goal(q, prm.qg, prm.obs.c)

        w_ann = self._w_ann(th)
        phi_r = self._phi_of_r(r)

        base = self.base_law.b_scalar(q)
        return base * np.sin(th + phi_r) * w_ann

# -------------------------
# Diagnostics
# -------------------------

def boundary_compliance(system: SecondOrderSystem,
                        Ds: Iterable[float],
                        vt_list: Iterable[float],
                        sector_only: bool = True) -> float:
    """
    Average fraction of ring samples (angles x speeds) that satisfy n·a >= 0 at grazing.

    Args:
        system: SecondOrderSystem with the desired magnetic law.
        Ds: distances to scan from obstacle boundary (m).
        vt_list: list of grazing speeds.
        sector_only: if True, only evaluate the goal-opposing semicircle.

    Returns:
        mean fraction over Ds.
    """
    prm = system.prm
    def in_goal_opposing_sector(q: np.ndarray) -> bool:
        return ((q - prm.obs.c) @ (prm.qg - prm.obs.c)) < 0.0

    thetas = np.linspace(0, 2*np.pi, 240, endpoint=False)
    fracs = []
    for d in Ds:
        ok = 0.0; tot = 0.0
        for th in thetas:
            q = prm.obs.c + (prm.obs.r + d)*np.array([np.cos(th), np.sin(th)])
            if sector_only and not in_goal_opposing_sector(q):  # focus where it matters
                continue
            dd, n, t = dist_n_t(q, prm.obs)
            for vt in vt_list:
                v = vt * t
                M = system.M_of_q(q); Minv = np.linalg.inv(M)
                Cv = system.C_times_qdot(q, v); N = system.N_of_q(q)
                a  = Minv @ (N @ v - Cv - prm.c_damp*(M @ v) - system.grad_psi(q))
                ok += 1.0 if (n @ a >= -1e-9) else 0.0
                tot += 1.0
        fracs.append(ok/max(tot,1e-12))
    return float(np.mean(fracs))