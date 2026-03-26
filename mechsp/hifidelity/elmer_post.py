"""
Post-processing utilities: Maxwell stress integration and analytic dipole field.
"""
from __future__ import annotations
import numpy as np
from typing import Tuple
try:
    import meshio  # for VTU reading
    from meshio._mesh import Mesh
except Exception:
    meshio = None

MU0 = 4e-7 * np.pi # magnetic permeability of free space

def get_total_force_on_ball(mesh: Mesh, ball_surf_tag: int | None = None) -> Tuple[float, float, float]:
    target_data = []

    if ball_surf_tag is None:
        tags = set(np.hstack(mesh.cell_data["GeometryIds"]))
        ball_surf_tag = tags[0]

    for k, cell_block in enumerate(mesh.cells):
        print(f"Block {k}: Type = {cell_block.type}, number of Elements: {len(cell_block.data)}")
        print(mesh.point_data.keys())


    if 'nodal force' not in mesh.point_data:
        print(f"No Nodal Forces found. Check whether 'Calculate Nodal Forces' is True in .sif.")
        return (None, None, None)
    
    forces = mesh.point_data['nodal force']
    
    total_force = np.sum(forces, axis=0)
    return total_force


def load_vtu(vtu_path: str) -> Mesh:
    """
    Load vtu file with meshio
    """
    if meshio is None:
        raise RuntimeError("meshio is not available. Install meshio to read VTU files.")
    m = meshio.read(vtu_path)
    tags = set(np.hstack(m.cell_data["GeometryIds"]))
    print("all tags: ", tags)
    return m


def maxwell_stress_force(pts: np.ndarray,
                         normals: np.ndarray, 
                         B: np.ndarray) -> np.ndarray:
    """
    Integrate Maxwell stress over a closed surface (approx. using nodal data).
    Continuous MST traction for magnetostatics in vacuum:
        t = (1/mu0) * [ (B·n) B - 0.5 |B|^2 n ]
    We approximate the surface integral by a lumped node area using Voronoi weights (here: simple uniform proxy).
    """
    # Very rough area estimate: total area via convex hull not available -> infer radius and 4πr^2 / N per node
    c = pts.mean(axis=0)
    r = np.mean(np.linalg.norm(pts - c, axis=1))
    A_node = 4*np.pi*r*r / pts.shape[0]
    t = (1.0/MU0) * ((np.sum(B*normals, axis=1)[:,None] * B) - 0.5*np.sum(B*B, axis=1)[:,None]*normals)
    F = np.sum(t * A_node, axis=0)
    return F

# ---------- Analytic dipole field helpers ----------
def dipole_B_vector(q: np.ndarray,
                    src: np.ndarray, 
                    m: float, 
                    axis: np.ndarray = np.array([0,0,1.0])) -> np.ndarray:
    """
    Magnetic field of a point dipole at src with moment maxis (SI units up to global scale).
    B(r) = mu0/(4π) [ 3 r (m·r) / r^5 - m / r^3 ] 
    """
    r = q - src
    r2 = np.dot(r, r)
    r5 = (r2**2.5) if r2 > 0 else 1e-24
    r3 = (r2**1.5) if r2 > 0 else 1e-18
    mvec = m * axis
    mdotr = np.dot(mvec, r)
    return (MU0/(4*np.pi)) * ( (3.0 *mdotr/r5) * r - mvec/r3 )

def synthetic_ball_surface_B(qc: np.ndarray,
                            R: float, 
                            coils_xyz: np.ndarray,
                            m: float, 
                            n_theta=40, 
                            n_phi=80) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Synthesize B on a spherical surface from a set of dipoles (for MST self‑check).
    Returns (points, normals, B).
    """
    thetas = np.linspace(1e-3, np.pi-1e-3, n_theta)
    phis = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    P = []
    N = []
    Bv = []
    for th in thetas:
        for ph in phis:
            n = np.array([np.sin(th)*np.cos(ph), np.sin(th)*np.sin(ph), np.cos(th)])
            p = qc + R*n
            B = np.zeros(3)
            for src in coils_xyz:
                B += dipole_B_vector(p, src, m, axis=np.array([0, 0, 1.0]))
            P.append(p); N.append(n); Bv.append(B)
    return np.array(P), np.array(N), np.array(Bv)