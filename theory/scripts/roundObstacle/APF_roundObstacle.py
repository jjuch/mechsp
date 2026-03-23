"""
APF (Artificial Potential Field) strategy for a round obstacle, reusing the
ground-truth model in navigate_roundObstacle.py.

- SecondOrderSystemAPF inherits from SecondOrderSystem and overrides grad_psi(...)
  so that the total gradient equals ∇(U_att + U_rep).
- Attractive: U_att = 0.5 * k_att * ||q - qg||^2        -> ∇U_att = k_att (q - qg)
- Repulsive : U_rep(d) = 0.5 * eta_rep * (1/d - 1/d0)^2  for d < d0 ; 0 otherwise
              ∇U_rep  = eta_rep * (1/d - 1/d0) * (1/d^2) * n, C^0; smoothed at d0.

Figures & comparisons are created by compareAPF_roundObstacle.py
"""


from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import Tuple, Optional

# Import your ground truth
from navigate_roundObstacle import (
    Params, Obstacle, SecondOrderSystem,
    dist_n_t, theta_rel_goal, phi_window_far, _initial_positions_aligned_with_goal
)


class SecondOrderSystemAPF(SecondOrderSystem):
    """
    Second-order system using an Artificial Potential Field (APF):
        M(q) qdd + C(q,qd) qd + c_damp M qd + ∇(U_att + U_rep) = 0

    Inherits all geometry/metric/damping from SecondOrderSystem, but overrides
    grad_psi(q) to return ∇(U_att + U_rep). The base rhs() subtracts grad_psi(q),
    yielding -∇U as a force, consistent with Lagrangian sign conventions.

    Parameters
    ----------
    prm : Params
        Scenario parameters (obstacle, goal, etc.).
    k_att : float
        Attractive gain; defaults to prm.k_psi for consistency with your code.
    eta_rep : float
        Repulsive gain (η in APF literature).
    d0 : float
        Repulsive influence distance (meters): active if 0 < d < d0.
    d_eps : float
        Regularization for 1/d near the wall (avoids blow-up).
    smooth_cap : float
        Width (meters) of a raised-cosine smoothing at the cutoff d0 for C¹ continuity.
    """

    def __init__(self,
                 prm: Params,
                 k_att: Optional[float] = None,
                 d0: float = 0.35,
                 d_eps: float = 5e-3,
                 smooth_cap: float = 1e-2,
                 U_wall: float = 2.25, # target wall energy
                 F_cap:float = 30.0, # smooth cap on ||grad U_rep||
                 eta_rep: Optional[float] = None):
        # Pass a "no-magnetic" strategy to the parent; APF acts via ∇U only.
        from navigate_roundObstacle import NoMagnetic
        super().__init__(prm, NoMagnetic(prm))
        self.k_att = prm.k_psi if (k_att is None) else float(k_att)
        self.d0 = float(d0)
        self.d_eps = float(d_eps)
        self.smooth_cap = float(smooth_cap)
        self.F_cap = float(F_cap)

        # auto-tune eta_rep from U_wall unless explicitly provided
        if eta_rep is None:
            # eta = 2* U_wall / ((1/d_eps - 1/d0)^2)
            denom = (1.0/max(self.d_eps, 1e-9) - 1.0/max(self.d0, 1e-9))
            self.eta_rep = float(max(2.0 * U_wall / max(denom * denom, 1e-12), 1e-6))
        else:
            self.eta_rep = float(eta_rep)
        
    # ------------ APF core (potentials and gradients) -----------------

    def U_att(self, q: np.ndarray) -> float:
        return 0.5 * self.k_att * np.linalg.norm(q - self.prm.qg)**2

    def grad_U_att(self, q: np.ndarray) -> np.ndarray:
        return self.k_att * (q - self.prm.qg)
    
    def _rep_window(self, d: float) -> float:
        """Raised-cosine smooth activation near d≈d0 (C¹). Returns alpha(d)∈[0,1]."""
        if d <= 0.0:
            return 1.0
        if d >= self.d0 + self.smooth_cap:
            return 0.0
        if d <= self.d0 - self.smooth_cap:
            return 1.0
        # Smoothly transition on [d0 - w, d0 + w]
        x = (d - (self.d0 - self.smooth_cap)) / (2.0*self.smooth_cap)
        return 0.5 * (1.0 + np.cos(np.pi * x))  # falls from 1->0

    def U_rep(self, q: np.ndarray) -> float:
        d, _, _ = dist_n_t(q, self.prm.obs)
        if d <= 0.0:
            d_use = self.d_eps  # inside: saturate
            return 0.5 * self.eta_rep * (1.0/d_use - 1.0/self.d0)**2
        if d >= self.d0 + self.smooth_cap:
            return 0.0
        alpha = self._rep_window(d)
        d_use = max(d, self.d_eps)
        return 0.5 * self.eta_rep * ((1.0/d_use - 1.0/self.d0)**2) * alpha


    def grad_U_rep(self, q: np.ndarray) -> np.ndarray:
        d, n, _ = dist_n_t(q, self.prm.obs)

        #outside influence, no repulsion
        if d >= self.d0 + self.smooth_cap:
            return np.zeros(2)
        
        # choose a well-defined distance
        d_use = self.d_eps if d <= 0.0 else max(d, self.d_eps)
        alpha = 1.0 if d <= 0.0 else self._rep_window(d)

        # scalar core shape (always >= 0 over (0, d0))
        core = self.eta_rep * (1.0/d_use - 1.0/self.d0) * (1.0/(d_use**2))  # >=0

        
        # We want the PHYSICAL FORCE  F_rep = -∇U_rep  to be outward = +…*n.
        # Cap the *force* smoothly, then convert back to a gradient by negation.
        F_rep_scalar = alpha * core                       # outward magnitude (uncapped)
        F_rep_capped = self.F_cap * np.tanh(F_rep_scalar / max(self.F_cap, 1e-9))  # outward

        # Gradient = - Force
        grad = - F_rep_capped * n                         # **inward** gradient
        return grad



    # ---- override the "potential gradient" that the parent subtracts in rhs() ----
    def grad_psi(self, q: np.ndarray) -> np.ndarray:
        return self.grad_U_att(q) + self.grad_U_rep(q)

    # ---------------- convenience utilities for plots -----------------

    def U_total(self, q: np.ndarray) -> float:
        return self.U_att(q) + self.U_rep(q)

    def grid_potential(self,
                       xlim: Tuple[float, float] = (-2.0, 2.0),
                       ylim: Tuple[float, float] = (-2.0, 2.0),
                       Nx: int = 120, Ny: int = 120):
        xs = np.linspace(xlim[0], xlim[1], Nx)
        ys = np.linspace(ylim[0], ylim[1], Ny)
        XX, YY = np.meshgrid(xs, ys)
        UU = np.full_like(XX, np.nan, dtype=float)
        G = np.full(XX.shape + (2,), np.nan, dtype=float)
        for i in range(Nx):
            for j in range(Ny):
                q = np.array([XX[j, i], YY[j, i]])
                d, _, _ = dist_n_t(q, self.prm.obs)
                if d <= 0:
                    continue
                UU[j, i] = self.U_total(q)
                G[j, i, :] = self.grad_psi(q)
        return XX, YY, UU, G

    def plot_potential_and_minima(self,
                                  xlim=(-2, 2), ylim=(-2, 2),
                                  Nx=140, Ny=140,
                                  grad_thresh=1e-2,
                                  save_as: Optional[str] = None):
        XX, YY, UU, G = self.grid_potential(xlim, ylim, Nx, Ny)
        fig, ax = plt.subplots(1, 1, figsize=(6.6, 5.6))
        cs = ax.contourf(XX, YY, UU, levels=20, cmap='magma')
        fig.colorbar(cs, ax=ax, label='U_APF(q)')
        # mark "candidate minima": small ||grad|| away from goal
        GM = np.linalg.norm(G, axis=2)
        mask = (GM < grad_thresh)
        # do not mark the goal neighborhood
        goal_mask = (np.hypot(XX - self.prm.qg[0], YY - self.prm.qg[1]) > 0.2)
        pts = np.column_stack([XX[mask & goal_mask], YY[mask & goal_mask]])
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], s=15, c='w', edgecolor='k',
                       label='candidate local minima', zorder=5)
            ax.legend(loc='upper right')
        # obstacle & goal
        ax.add_patch(Circle(self.prm.obs.c, self.prm.obs.r, facecolor='w', alpha=0.7, edgecolor='k'))
        ax.plot(self.prm.qg[0], self.prm.qg[1], 'y*', ms=14)
        ax.set_aspect('equal'); ax.set_xlim(xlim); ax.set_ylim(ylim); ax.grid(True, alpha=0.2)
        ax.set_title("APF potential (attractive + repulsive)")
        plt.tight_layout()
        if save_as:
            plt.savefig(save_as, dpi=180); plt.close(fig)
        else:
            plt.show()
