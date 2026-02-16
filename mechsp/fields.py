# mechsp/fields.py
"""
FieldDesign: container for a single, multi-frequency field design.

This dataclass encapsulates:
  - Coil geometry (positions, depth)
  - DC currents for the conservative potential (K)
  - Rotating-field phasors (amplitude, phase) and frequency (omega) for N_eff
  - High-frequency modulation (amplitude, phase) and frequency (Omega) for M_eff

It also provides convenience evaluators to reconstruct:
  - Bz0(q):   the DC (static) vertical field
  - grad_Bz0(q)
  - B0_and_phi(q): rotating-field amplitude & spatial phase at q
  - B1(q):   HF shape function used to build M_eff via Hessian(D^2 B1)
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np

from .magnetics import dipole_Bz, grad_Bz_analytic


@dataclass
class FieldDesign:
    """
    Parameters
    ----------
    coil_xy : (N,2) ndarray
        Coil centers in meters [(x_i, y_i)]
    h : float
        Coil plane depth below z=0 (meters, positive)
    I0 : (N,) ndarray
        DC currents (A) for K_eff
    Irot_amp : (N,) ndarray
        Rotating-field current amplitudes (A) for N_eff
    Irot_phase : (N,) ndarray
        Rotating-field current phases (rad) for N_eff
    omega : float
        Rotating-field frequency (rad/s)
    Ihf_amp : (N,) ndarray
        HF current amplitudes (A) for M_eff (shape function)
    Ihf_phase : (N,) ndarray
        HF current phases (rad)
    Omega : float
        High-frequency modulation (rad/s)
    eps : float
        Non-dimensional HF scaling ε (used in M_eff = m I + (ε^2/2Ω^2) D^2 B1)
    marble_moment : float
        Magnetic dipole magnitude of the ball used in force computation (arbitrary scale)
    scale : float
        Dipole scale (absorbs μ0/4π etc.), keep =1.0 for normalized basis
    """
    coil_xy: np.ndarray
    h: float
    I0: np.ndarray
    Irot_amp: np.ndarray
    Irot_phase: np.ndarray
    omega: float
    Ihf_amp: np.ndarray
    Ihf_phase: np.ndarray
    Omega: float
    eps: float
    marble_moment: float = 1.0
    scale: float = 1.0

    # ---------- DC (K_eff) ----------
    def Bz0(self, q: np.ndarray) -> float:
        """Return DC vertical field at point q=(x,y,)."""
        total = 0.0
        for Ii, cxy in zip(self.I0, self.coil_xy):
            total += Ii * dipole_Bz(q[None, :], cxy, self.h, scale=self.scale)[0]
        return total

    def grad_Bz0(self, q: np.ndarray) -> np.ndarray:
        """Return ∇Bz0(q) as (2,) ndarray."""
        G = np.zeros(2)
        for Ii, cxy in zip(self.I0, self.coil_xy):
            G += Ii * grad_Bz_analytic(q[None, :], cxy, self.h, scale=self.scale)[0]
        return G

    # ---------- Rotating (N_eff) ----------
    def B0_and_phi(self, q: np.ndarray) -> Tuple[float, float]:
        """
        Return (B0(q), phi(q)) from the rotating-field phasors.
        We form complex phasor at q: Sum_i A_i b_i(q), where A_i = Irot_amp_i * exp(j * phase_i).
        """
        A = self.Irot_amp * np.exp(1j * self.Irot_phase)
        acc = 0.0 + 0.0j
        for Ai, cxy in zip(A, self.coil_xy):
            b = dipole_Bz(q[None, :], cxy, self.h, scale=self.scale)[0]
            acc += Ai * b
        B0 = np.abs(acc)
        phi = float(np.angle(acc))
        return B0, phi

    # ---------- High-frequency (M_eff) ----------
    def B1(self, q: np.ndarray) -> float:
        """
        Return HF shape B1(q) = Sum_i Ihf_amp_i * cos(Ihf_phase_i) * b_i(q) for simplicity.
        TODO: extend to full complex phasors if needed later
        """
        total = 0.0
        for Ai, thetai, cxy in zip(self.Ihf_amp, self.Ihf_phase, self.coil_xy):
            b = dipole_Bz(q[None, :], cxy, self.h, scale=self.scale)[0]
            total += Ai * np.cos(thetai) * b
        return total