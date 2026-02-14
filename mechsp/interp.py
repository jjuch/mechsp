# mechsp/interp.py
import numpy as np
from .dynamics import force_from_currents
from .magnetics import grad_Bz_analytic

try:
    from scipy.interpolate import RegularGridInterpolator
    _HAS_SCIPY_INTERP = True
except Exception:
    _HAS_SCIPY_INTERP = False




def precompute_force_grid(coil_xy, h, I, L, Gside=80, marble_moment=1.0, scale=1.0):
    """
    Vectorized grid: compute U,V on (gy,gx).
    """
    gx = np.linspace(-L/2, L/2, Gside)
    gy = np.linspace(-L/2, L/2, Gside)
    X, Y = np.meshgrid(gx, gy, indexing='xy')           # (G,G)
    P = np.stack([X.ravel(), Y.ravel()], axis=1)         # (G*G,2)

    # For each coil, get ∇Bz at all points in one go, then sum with I
    U = np.zeros(P.shape[0], dtype=float)
    V = np.zeros(P.shape[0], dtype=float)
    for Ii, cxy in zip(I, coil_xy):
        G = marble_moment * grad_Bz_analytic(P, cxy, h, scale=scale)  # (G*G,2)
        U += Ii * G[:, 0]
        V += Ii * G[:, 1]

    U = U.reshape(gy.size, gx.size)  # (G,G)
    V = V.reshape(gy.size, gx.size)
    return gx, gy, U, V



def make_force_interpolator(gx, gy, U, V):
    """
    Build a callable F(q) using compiled regular-grid interpolation.
    If SciPy not available, falls back to bilinear manual interpolation.
    """
    if _HAS_SCIPY_INTERP:
        Fx = RegularGridInterpolator((gy, gx), U, bounds_error=False, fill_value=None)  # note (y,x) order
        Fy = RegularGridInterpolator((gy, gx), V, bounds_error=False, fill_value=None)

        def F_interp(q):
            # q is (2,) -> needs [[y,x]]
            pt = np.array([[q[1], q[0]]])
            fx = Fx(pt)[0]
            fy = Fy(pt)[0]
            return np.array([fx, fy])

        return F_interp
    else:
        # simple bilinear fallback
        def F_interp(q):
            x, y = q[0], q[1]
            # indices
            if x <= gx[0] or x >= gx[-1] or y <= gy[0] or y >= gy[-1]:
                # outside grid: zero force
                return np.zeros(2)
            ix = np.searchsorted(gx, x) - 1
            iy = np.searchsorted(gy, y) - 1
            x0, x1 = gx[ix], gx[ix+1]
            y0, y1 = gy[iy], gy[iy+1]
            tx = (x - x0) / (x1 - x0)
            ty = (y - y0) / (y1 - y0)

            # bilinear interpolation
            def bilerp(M):
                m00 = M[iy, ix]
                m01 = M[iy, ix+1]
                m10 = M[iy+1, ix]
                m11 = M[iy+1, ix+1]
                return (1-ty)*((1-tx)*m00 + tx*m01) + ty*((1-tx)*m10 + tx*m11)

            return np.array([bilerp(U), bilerp(V)])

        return F_interp