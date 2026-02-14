# mechsp/barrier.py
import numpy as np

def edge_barrier_force(q, L, k_edge=5.0, margin=0.02, p=2):
    """
    Soft wall force near the square boundary ±L/2.
    margin: how far from edge the barrier ramps up (m).
    k_edge: strength.
    p: steepness exponent.

    Potential idea (component-wise):
      For x:
        d = L/2 - |x|
        if d < margin: add ∂/∂x [ k_edge * ( (margin/d)^p ) ]
      Similar for y.
    We smooth and cap to avoid infinities.
    """
    x, y = q[0], q[1]
    half = L / 2.0

    def one_axis(val):
        d = half - abs(val)
        if d <= 0:
            # outside; push hard inward
            sgn = -np.sign(val)
            return sgn * k_edge * (1.0 + (abs(val)-half)/1e-3)
        if d >= margin:
            return 0.0
        # inside the margin: power-law rise
        # F = -∂K/∂val; with K ~ (margin/d)^p
        sgn = -np.sign(val)                 # direction towards center
        # derivative wrt val through d = half - |val| gives: dK/dval ~ p*(margin^p)*|val|' / d^{p+1}
        # |val|' = sign(val)
        # Compose signs carefully so force points inward
        denom = max(d, 1e-6)
        mag = k_edge * (margin ** p) * p * (1.0 / (denom ** (p+1)))
        return sgn * mag

    Fx = one_axis(x)
    Fy = one_axis(y)
    return np.array([Fx, Fy])