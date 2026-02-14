# mechsp/magnetics.py
import numpy as np

def dipole_Bz(xy, src_xy, h, scale=1.0):
    """
    Bz at (x,y,0) from a z-oriented ideal dipole at (xs, ys, -h).
    scale absorbs μ0/(4π) * m_dipole; for prototyping we keep scale=1.
    xy: (N,2) points on plane z=0
    src_xy: (2,) coil center
    h: positive distance below plane
    Returns: (N,) Bz values
    """
    r = xy - src_xy[None, :]
    rx, ry = r[:, 0], r[:, 1]
    rz = h * np.ones_like(rx)
    R2 = rx**2 + ry**2 + rz**2
    # Bz component of dipole field aligned with +z (up to scale)
    Bz = scale * ((3.0 * rz * rz) / (R2 ** 2.5) - 1.0 / (R2 ** 1.5))
    return Bz


def grad_Bz_analytic(xy, src_xy, h, scale=1.0):
    """
    Vectorized closed-form ∇Bz. xy: (N,2), src_xy: (2,), returns (N,2).
    """
    r = xy - src_xy[None, :]
    rx, ry = r[:, 0], r[:, 1]
    rz = h
    R2 = rx**2 + ry**2 + rz**2
    Rm5 = R2**(-2.5)
    Rm7 = R2**(-3.5)
    common = 3.0 * scale * (Rm5 - 5.0 * rz * rz * Rm7)
    dBx = common * rx
    dBy = common * ry
    return np.stack([dBx, dBy], axis=1)


def grad_Bz(xy, src_xy, h, scale=1.0, eps=1e-4):
    """
    Finite-difference ∇Bz = [∂Bz/∂x, ∂Bz/∂y] at xy from a dipole at (src_xy, -h).
    Returns: (N,2)
    """
    ex = np.array([eps, 0.0])
    ey = np.array([0.0, eps])
    Bz_xp = dipole_Bz(xy + ex, src_xy, h, scale)
    Bz_xm = dipole_Bz(xy - ex, src_xy, h, scale)
    Bz_yp = dipole_Bz(xy + ey, src_xy, h, scale)
    Bz_ym = dipole_Bz(xy - ey, src_xy, h, scale)
    dBx = (Bz_xp - Bz_xm) / (2 * eps)
    dBy = (Bz_yp - Bz_ym) / (2 * eps)
    return np.stack([dBx, dBy], axis=1)