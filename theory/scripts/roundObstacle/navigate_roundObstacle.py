"""
Minimal library for the 2‑D second‑order navigation model with round obstacle-aware metric and gyroscopic two-form. Includes:
  • Geometry utilities.
  • SecondOrderSystem (Levi–Civita, Rayleigh damping).
  • Magnetic laws: None, Const, Power(dp), Sine (θ), SineRPhase (θ + φ(r)).
  • Diagnostics: boundary compliance, annulus auto-selection for SineRPhase.
  • Lightweight simulation (RK4).

Author: Jasper Juchem
"""

from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from typing import Callable, Iterable, Tuple, List, Dict, Optional
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
        return self.b_law.b_scalar(q, self.prm) * J

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
            print(f"{k} / {int(tmax/h)}", end="\r")
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

def _signed_kappa(v: np.ndarray, a: np.npdarray) -> float:
    sp = np.linalg.norm(v)
    if sp < 1e-10: return 0.0
    return float((v[0]*a[1] - v[1]*a[0]) / (sp**3))

# -------------------------
# Magnetic laws (Strategy)
# -------------------------

class MagneticLaw(ABC):
    """Strategy base for N(q)=b(q)*J."""
    name: str = "base"
    @abstractmethod
    def b_scalar(self, q: np.ndarray, prm: Params) -> float:
        ...

    @staticmethod
    def _agg(vals: list[float]) -> float:
        """
        Per-pixel aggregation over vts: RMS magnitude with mean-sign to avoid random sign flips.
        """
        if not vals: return np.nan
        arr = np.asarray(vals, float)
        return float(np.sqrt(np.mean(arr**2))) * np.sign(np.mean(arr))
    

    def plot_curvature_maps(self,
                            prm:Params,
                            save_as: str | None = None,
                            xlim: Tuple[float, float] = (-2.0, 2.0),
                            ylim: Tuple[float, float] = (-2.0, 2.0),
                            grid: Tuple[int, int] = (90, 90),
                            vts: Tuple[float, ...] = (0.6,),
                            with_trajectories: bool = True,
                            trajectories: tuple[np.ndarray, list] | None = None,
                            n_traj: int = 5) -> tuple[np.ndarray, list]:
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
            trajectories: pre-simulated trajectories with starts and the simulated time series.
            n_traj: number of starting points.

        Returns:
            starts: the starting positions of the simulated trajectories. If ´with_trajectories´ is False, None is returned.
            Q_all: a list of np.ndarray's with the simulated positions. If ´with_trajectories´is False, None is returned.
        """
        system = SecondOrderSystem(prm, self)

        Nx, Ny = grid
        xs = np.linspace(xlim[0], xlim[1], Nx)
        ys = np.linspace(ylim[0], ylim[1], Ny)
        XX, YY = np.meshgrid(xs, ys)

        K_tot  = np.full_like(XX, np.nan, dtype=float)
        K_B    = np.full_like(XX, np.nan, dtype=float)
        K_geom = np.full_like(XX, np.nan, dtype=float)
        K_goal = np.full_like(XX, np.nan, dtype=float)


        # Per-pixel aggregation over vts: RMS magnitude with mean-sign to avoid random sign flips.
        # def _agg(vals: List[float]) -> float:
        #     if not vals: return np.nan
        #     vals = np.asarray(vals, dtype=float)
        #     return float(np.sqrt(np.mean(vals**2))) * np.sign(np.mean(vals))

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
        vmax = np.nanpercentile(stack, 99, axis=1)
        vmax = [max(v, 1e-4) for v in vmax]
        vmin = np.nanmin(stack, axis=1)
        vmin = [min(v, -1e-4) for v in vmin]

        fig, axes = plt.subplots(2, 2, figsize=(11.2, 9.2))
        mats   = [K_tot, K_B, K_geom, K_goal]
        titles = [r'$\kappa_{\rm s,tot}$', r'$\kappa_{\rm s,B}$',
                  r'$\kappa_{\rm s,geom}$', r'$\kappa_{\rm s,goal}$']

        for ax, M, T, vma, vmi in zip(axes.flatten(), mats, titles, vmax, vmin):
            norm = TwoSlopeNorm(vcenter=0.0, vmin=vmi, vmax=vma)
            im = ax.imshow(M, origin='lower', extent=[xlim[0], xlim[1], ylim[0], ylim[1]], cmap='RdBu_r', norm=norm)
            ax.add_patch(Circle(prm.obs.c, prm.obs.r, facecolor='w', alpha=0.6,
                                   edgecolor='k', linewidth=1.0))

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
                starts = _initial_positions_aligned_with_goal(prm, num=n_traj)
                Q_all = []
            else:
                starts, Q_all = trajectories

            for i, q0 in enumerate(starts):
                print(f"{i + 1} / {len(starts)}", end="\r")
                if trajectories is None:
                    _, _, t = dist_n_t(q0, prm.obs)
                    v0 = 0.05 * t
                    _, Xs, _ = system.simulate(q0, v0, h=0.05, tmax=40.0)
                    Q = Xs[:, :2]
                    Q_all.append(Q)
                else:
                    Q = Q_all[i]
                for ax in axes.flatten():
                    ax.plot(Q[:,0], Q[:,1], '-', lw=2, color='k', alpha=0.85, zorder=5)
                    ax.plot(q0[0], q0[1], 'ko', ms=3, zorder=6)
        else:
            Q_all = None; starts = None
        
        for ax in axes.flatten():
            ax.plot(prm.qg[0], prm.qg[1], 'y*', ms=20, zorder=50)

        plt.tight_layout()
        if save_as: plt.savefig(save_as, dpi=180); plt.close(fig)
        else:       plt.show()

        return starts, Q_all


    def plot_grazing_normal_maps(self, 
                                 prm: Params,
                                 save_as: str | None = None,
                                 xlim: tuple[float,float] = (-2.0, 2.0),
                                 ylim: tuple[float,float] = (-2.0, 2.0),
                                 grid: tuple[int,int] = (90, 90),
                                 vts: tuple[float,...] = (0.6,),
                                 with_trajectories: bool = True,
                                 trajectories: tuple[np.ndarray, list] | None = None,
                                 n_traj: int = 5,
                                 axis_gap: float | None = None) -> tuple[np.ndarray, list]:
        
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
            trajectories: pre-simulated trajectories with starts and the simulated time series.
            n_traj: how many starting points.
            axis_gap: optionally ignore a wedge |θ|<axis_gap (rad) around the goal axis to avoid window=0 pixels dominating (useful for sine/sine_rphase).

        Returns:
            starts: the starting positions of the simulated trajectories. If ´with_trajectories´ is False, None is returned.
            Q_all: a list of np.ndarray's with the simulated positions. If ´with_trajectories´is False, None is returned.
        """
        system = SecondOrderSystem(prm, self)

        Nx, Ny = grid
        xs = np.linspace(xlim[0], ylim[1], Nx)
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
        vmax = np.nanpercentile(stack, 99, axis=1)
        vmax = [max(v, 1e-4) for v in vmax]
        vmin = np.nanmin(stack, axis=1)
        vmin = [min(v, -1e-4) for v in vmin]

        fig, axes = plt.subplots(2, 2, figsize=(11.2, 9.2))
        mats = [NA_tot, NA_B, NA_geom, NA_goal]
        titles = [r'$n\!\cdot a_{\rm tot}$', r'$n\!\cdot a_{B}$',
                  r'$n\!\cdot a_{\rm geom}$', r'$n\!\cdot a_{\rm goal}$']
        
        for ax, M, T, vmi, vma in zip(axes.flatten(), mats, titles, vmin, vmax):
            norm = TwoSlopeNorm(vcenter=0.0, vmin=vmi, vmax=vma)
            im = ax.imshow(M, origin='lower', extent=[xlim[0], xlim[1], ylim[0], ylim[1]], cmap='RdBu_r', norm=norm)
            ax.add_patch(Circle(prm.obs.c, prm.obs.r, facecolor='w', alpha=0.6, edgecolor='k', linewidth=1.0))

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
                starts = _initial_positions_aligned_with_goal(prm, num=n_traj)
                Q_all = []
            else:
                starts, Q_all = trajectories
            
            for i, q0 in enumerate(starts):
                if trajectories is None:
                    _,_, t = dist_n_t(q0, prm.obs)
                    v0 = 0.05*t
                    _, Xs, _ = system.simulate(q0, v0, h=0.05, tmax=40.0)
                    Q = Xs[:,:2]
                    Q_all.append(Q)
                else:
                    Q = Q_all[i]
                for ax in axes.flatten():
                    ax.plot(Q[:,0], Q[:,1], '-', lw=2, color='k', alpha=0.85, zorder=5)
                    ax.plot(q0[0], q0[1],'ko', ms=3, zorder=6)
        else:
            Q_all = None; starts = None

        for ax in axes.flatten():
            ax.plot(prm.qg[0], prm.qg[1], 'y*', ms=20, zorder=50)

        plt.tight_layout()
        if save_as: plt.savefig(save_as, dpi=180); plt.close(fig)
        else: plt.show()

        return starts, Q_all
    

class RoundMagneticLaw(MagneticLaw):
    name = "baseRound"

    def __init__(self, 
                 r1: Optional[float] = None,
                 r2: Optional[float] = None):
        self.r1 = r1; self.r2 = r2
        self._annulus_ready = (r1 is not None and r2 is not None)

    # ---- auto selection of (r1,r2) ----    
    def auto_annulus(self,
                     prm: Params,
                     tau_far: float=0.25,
                     tau_near: float = 1e-3,
                     min_width: float = 0.04,
                     max_width: float = 0.50) -> tuple[float,float]:
        
        """
        Simple analytic estimator for (r1,r2):
          • r2 from φ_far(d2)=tau_far. d_on,q: φ_far(d, q)=exp(-(d/d_on)^q) parameters.
          • r1 from d1^p = tau_near  => d1 = tau_near^(1/p).
        Clamps the annulus width to [min_width, max_width].

        Args:
            tau_far: far-field cutoff (0<tau_far<1).
            tau_near: near-boundary cutoff for d^p (0<tau_near<1).
            min_width,max_width: thickness clamp (meters).

        Returns:
            (r1,r2)
        """
        # outer
        d2 = prm.d_far * (-np.log(max(tau_far, 1e-12))) ** (1.0/max(prm.q_far, 1e-6))
        # inner
        d1 = max(1e-6, tau_near**(1.0/max(prm.p, 1e-6)))

        # convert to radii
        r_obs = prm.obs.r
        r2 = r_obs + d2
        r1 = r_obs + d1
        # print(f"r1: {d1} | r2: {d2}")

        
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
        return self.r1, self.r2


class NoMagnetic(MagneticLaw):
    name = "none"
    def b_scalar(self, q: np.ndarray, prm: Params) -> float:
        return 0.0

class ConstMagnetic(MagneticLaw):
    """
    b(d)=kB * φ_far(d).
    """
    name = "const"
    def b_scalar(self, q: np.ndarray, prm: Params) -> float:
        d,_,_ = dist_n_t(q, prm.obs)
        return prm.kB * phi_window_far(d)

class PowerMagnetic(MagneticLaw):
    """
    b(d)=kB * d^p * φ_far(d).
    """
    name = "dp"
    def b_scalar(self, q: np.ndarray, prm: Params) -> float:
        d,_,_ = dist_n_t(q, prm.obs)
        return prm.kB * (max(d,1e-6)**prm.p) * phi_window_far(d)

class SineMagnetic(MagneticLaw):
    """
    Single-lobe sine in angle: b(d,θ)=kB d^p φ_far(d) * [(1-S(d)) + S(d) sin θ],
    where S(d)=switch that keeps outward sign in the very near field.
    """
    name = "sine"

    def __init__(self, d_sw: float = 0.06, w_sw: float = 0.02):
        self.d_sw = d_sw
        self.w_sw = w_sw

    def _switch(self, d: float) -> float:
        """
        smooth radial switch S(d) : 0 near obstacle, -> 1 after d_sw
        """
        x = (max(d,0.0) - self.d_sw) / max(self.w_sw, 1e-6)
        return 0.5*(1.0 + np.tanh(x))

    def b_scalar(self, q: np.ndarray, prm: Params) -> float:
        d,_,_ = dist_n_t(q, prm.obs)
        S = self._switch(d)
        th = theta_rel_goal(q, prm.qg, prm.obs.c)
        base = prm.kB * (max(d,1e-6)**prm.p) * phi_window_far(d)
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
        self.r1 = r1; self.r2 = r2 
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


    # ---- helper pieces ----
    @ staticmethod
    def _A_gauss(r: float, r1: float, r2: float, sigma_frac= float) -> float:
        rm = 0.5 * (r1 + r2)
        sigma = sigma_frac * max(r2 - r1, 1e-9)
        return float(np.exp(-((r - rm)**2)/(2*sigma**2)))
    
    def _A_gauss_main(self, r):
        return self._A_gauss(r, self.r1, self.r2, self.sigma_frac)

    def _W_theta(self, th: float) -> float:
        return float(self.w_min + (1.0 - self.w_min) * ((1.0 - np.cos(th))*0.5)**self.gamma)
    
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
                prm: Params,
                delta_cap: float | None = None,
                sigma_cap: float | None = None) -> tuple[float, float]:
        """Return (r_cap, sigma_cap) for the knee-cap: r_cap = r2 + delta_cap"""
        if delta_cap is not None: self.delta_cap = delta_cap
        if sigma_cap is not None: self.sigma_cap = sigma_cap
        return float(self.r2) + self.delta_cap, self.sigma_cap
    

    def b_scalar(self, q: np.ndarray, prm: Params) -> float:
        if not self._annulus_ready:
            self.auto_annulus(prm, tau_far=1e-2, tau_near=1e-2, min_width=0.04, max_width=0.50)

        d, _, _ = dist_n_t(q, prm.obs)
        qc = q - prm.obs.c; r = np.linalg.norm(qc)
        th = theta_rel_goal(q, prm.qg, prm.obs.c)
        A = self._A_gauss_main(r)
        W = self._W_theta(th)
        phi = self._phi_r(r)

        a1 = A # Sine lobe on the annulus
        base = prm.kB * (max(d,1e-6)**prm.p) * phi_window_far(d)

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
                 r1: Optional[float] = None,
                 r2: Optional[float] = None,
                 phi_max: float = np.pi/2,
                 sigma_frac: float = 0.35,
                 use_tanh: bool = False,
                 k: float = 4.0,
                 sector_only: bool = True):
        self.r1 = r1
        self.r2 = r2
        self.phi_max = phi_max
        self.sigma_frac = sigma_frac
        self.use_tanh = use_tanh
        self.k = k
        self.sector_only = sector_only
        # populated after calling auto_annulus(...) if r1/r2 are None:
        self._annulus_ready = False

    # ---- annulus utils ----
    @staticmethod
    def _w_ann(theta: float, gamma: float = 2.0) -> float:
        """
        Smooth angular window that kills the donut on the goal side. theta is measured from the obstacle -  goal axis.
        """
        return float(((1.0 - np.cos(theta)) * 0.5)**gamma)

    def _phi_of_r(self, r: float, r1: float, r2: float) -> float:
        rm = 0.5*(r1+r2)
        s  = (r - rm) / max(0.5*(r2 - r1), 1e-9) # s \in [-1, 1]
        s  = float(np.clip(s, -1.0, 1.0))
        if self.use_tanh:
            return self.phi_max * np.tanh(self.k*s) / np.tanh(self.k)
        return self.phi_max * s

    # ---- law ----
    def b_scalar(self, q: np.ndarray, prm: Params) -> float:
        # ensure annulus
        # d_on, q_far = 0.35, 2.0 # phi_window_far parameters
        if self.r1 is None or self.r2 is None:
            self.auto_annulus(prm, tau_far=1e-2, tau_near=1e-2, min_width=0.04, max_width=0.50)
        
        r1 = self.r1
        r2 = self.r2
            
        d,_,_ = dist_n_t(q, prm.obs)
        qc = q - prm.obs.c; r = np.linalg.norm(qc)
        th = theta_rel_goal(q, prm.qg, prm.obs.c)

        w_ann = self._w_ann(th)
        phi_r = self._phi_of_r(r, r1, r2)

        base = prm.kB * (max(d,1e-6)**prm.p) * w_ann * phi_window_far(d, d_on=prm.d_far, q=prm.q_far)
        return base * np.sin(th + phi_r)

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