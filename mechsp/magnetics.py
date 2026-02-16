# mechsp/magnetics.py
import numpy as np

def _pairwise_geometry(xy: np.ndarray, src_xy: np.ndarray, h: float):
    """
    Normalize inputs and build pairwise geometry for broadcasting.

    Parameters
    ----------
    xy : (2,) or (J,2)
        Evaluation point(s) on z=0 plane.
    src_xy : (2,) or (N,2)
        Coil/dipole location(s) on z=-h plane.
    h : float
        Depth of the dipoles.

    Returns
    -------
    rx, ry : (J,N) arrays
    R2     : (J,N) array
    flags  : dict
        {'xy_was_1d': bool, 'src_was_1d': bool, 'J': int, 'N': int}
    """
    xy = np.asarray(xy, dtype=float)
    src_xy = np.asarray(src_xy, dtype=float)

    xy_was_1d = (xy.ndim == 1)
    src_was_1d = (src_xy.ndim == 1)

    if xy_was_1d:
        xy = xy.reshape(1, 2)        # (1,2) for broadcasting
    if src_was_1d:
        src_xy = src_xy.reshape(1, 2) # (1,2)

    J = xy.shape[0]
    N = src_xy.shape[0]

    # pairwise differences
    rx = xy[:, None, 0] - src_xy[None, :, 0]  # (J,N)
    ry = xy[:, None, 1] - src_xy[None, :, 1]  # (J,N)
    rz = float(h)

    R2 = rx*rx + ry*ry + rz*rz
    # avoid division by zero; physically you never sit on a coil center
    R2 = np.maximum(R2, 1e-18)

    flags = dict(xy_was_1d=xy_was_1d, src_was_1d=src_was_1d, J=J, N=N)
    return rx, ry, R2, rz, flags


def dipole_Bz(xy: np.ndarray, src_xy: np.ndarray, h: float, scale: float = 1.0):
    """
    Vertical component Bz at point(s) xy from z-oriented dipole(s) at src_xy (depth -h).

    Parameters
    ----------
    xy : (2,) or (J,2)
    src_xy : (2,) or (N,2)
    h : float
    scale : float

    Returns
    -------
    Bz :
        - (J,) if src_xy is (2,)
        - (N,) if xy is (2,)
        - (J,N) if both are arrays
    """
    rx, ry, R2, rz, f = _pairwise_geometry(xy, src_xy, h)
    Rm3 = R2**(-1.5)
    Rm5 = R2**(-2.5)
    Bz = scale * (3.0*rz*rz*Rm5 - Rm3)  # (J,N)

    # squeeze shape for backward compatibility
    if f['src_was_1d'] and not f['xy_was_1d']:
        return Bz[:, 0]           # (J,)
    elif f['xy_was_1d'] and not f['src_was_1d']:
        return Bz[0, :]           # (N,)
    else:
        return Bz                 # (J,N)


def grad_Bz_analytic(xy: np.ndarray, src_xy: np.ndarray, h: float, scale: float = 1.0):
    """
    Analytic gradient ∇Bz at xy from z-oriented dipole(s) at src_xy.

    Parameters
    ----------
    xy : (2,) or (J,2)
    src_xy : (2,) or (N,2)
    h : float
    scale : float

    Returns
    -------
    grad :
        - (J,2) if src_xy is (2,)
        - (N,2) if xy is (2,)
        - (J,N,2) if both are arrays
    """
    rx, ry, R2, rz, f = _pairwise_geometry(xy, src_xy, h)
    Rm5 = R2**(-2.5)
    Rm7 = R2**(-3.5)

    A = (Rm5 - 5.0*rz*rz*Rm7)          # (J,N)
    dBx = 3.0*scale*rx*A               # (J,N)
    dBy = 3.0*scale*ry*A               # (J,N)
    G = np.stack([dBx, dBy], axis=-1)  # (J,N,2)

    # squeeze
    if f['src_was_1d'] and not f['xy_was_1d']:
        return G[:, 0, :]              # (J,2)
    elif f['xy_was_1d'] and not f['src_was_1d']:
        return G[0, :, :]              # (N,2)
    else:
        return G                       # (J,N,2)


def hessian_Bz_analytic(xy: np.ndarray, src_xy: np.ndarray, h: float, scale: float = 1.0):
    """
    Analytic Hessian D^2 Bz at xy from z-oriented dipole(s) at src_xy.

    Parameters
    ----------
    xy : (2,) or (J,2)
    src_xy : (2,) or (N,2)
    h : float
    scale : float

    Returns
    -------
    H :
        - (J,2,2) if src_xy is (2,)
        - (N,2,2) if xy is (2,)
        - (J,N,2,2) if both are arrays

    Order is [[dxx, dxy], [dxy, dyy]].
    """
    rx, ry, R2, rz, f = _pairwise_geometry(xy, src_xy, h)
    Rm5 = R2**(-2.5)
    Rm7 = R2**(-3.5)
    Rm9 = R2**(-4.5)

    A = (Rm5 - 5.0*rz*rz*Rm7)          # (J,N)
    C = (-5.0*Rm7 + 35.0*rz*rz*Rm9)    # (J,N)

    dxx = 3.0*scale*(A + rx*rx*C)      # (J,N)
    dyy = 3.0*scale*(A + ry*ry*C)      # (J,N)
    dxy = 3.0*scale*(rx*ry*C)          # (J,N)

    H = np.zeros(dxx.shape + (2, 2))
    H[..., 0, 0] = dxx
    H[..., 1, 1] = dyy
    H[..., 0, 1] = dxy
    H[..., 1, 0] = dxy

    # squeeze
    if f['src_was_1d'] and not f['xy_was_1d']:
        return H[:, 0, :, :]           # (J,2,2)
    elif f['xy_was_1d'] and not f['src_was_1d']:
        return H[0, :, :, :]           # (N,2,2)
    else:
        return H                       # (J,N,2,2)