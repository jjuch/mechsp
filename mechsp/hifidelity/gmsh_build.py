"""
Gmsh geometry/mesh builder for the DC verification case.
Generates:

Coil blocks (rectangular prisms) located at coil_xy in the plane z=-h
An iron sphere (ball) centered at q_ball
An air box that encloses everything
Physical groups for: air, coils, ball, outer boundary, ball surface

NOTE: Requires gmsh Python API. Install via pip install gmsh or use the official SDK.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Sequence

try:
    import gmsh  # type: ignore
except Exception:
    gmsh = None

def build_mesh(out_msh: str, *,
               coil_xy: np.ndarray,
               h: float,
               q_ball: Sequence[float],
               ball_r: float,
               L: float,
               coil_size: Tuple[float, float, float] = (0.004, 0.004, 0.003),
               lc_air: float = 0.01,
               lc_ball: float = 0.0005,
               lc_coil: float = 0.001,
               make_shell: bool = False,) -> None:
    """
    Create a 3D mesh with coils, ball, air.
    Parameters
    ----------
    out_msh : str
        Path to write Gmsh .msh file (version 4 recommended)
    coil_xy : (N,2) ndarray
        Coil centers in meters
    h : float
        Depth of coil plane (ball on z=0; coils centered at z=-h)
    q_ball : (3,) sequence
        Ball center (x,y,z)
    ball_r : float
        Ball radius (m)
    L : float
        Air box half‑size -> box spans [-L/2, L/2]^2 in x,y and [-(L/2+h), (L/2)] in z
    coil_size : (dx,dy,dz)
        Rectangular coil block size (meters)
    lc_air, lc_ball, lc_coil : float
        Target mesh sizes in regions
    make_shell : bool
        If True, create a thin air shell around the ball for MST integration (optional)
    """
    if gmsh is None:
        raise RuntimeError("gmsh Python API is not available. Install gmsh and rerun.")

    gmsh.initialize()
    gmsh.model.add("mechsp_dc")
    occ = gmsh.model.occ

    # --- Air box ---
    air = occ.addBox(-L/2, -L/2, -L/2 - h, L, L, L + h)  # include coil plane below

    # --- Ball ---
    qx, qy, qz = q_ball
    ball = occ.addSphere(qx, qy, qz, ball_r)

    # --- Coils ---
    dx, dy, dz = coil_size
    coils = []
    for (cx, cy) in coil_xy:
        coils.append(occ.addBox(cx - dx/2, cy - dy/2, -h - dz/2, dx, dy, dz))


    # Cut the ball and coils out of the air such that the mesh connects(conformal mesh)
    all_volumes = [(3, air), (3, ball)] + [(3, coil) for coil in coils]
    out, out_map = occ.fragment(all_volumes, [])
    # Synchronize the OCC CAD with the current gmsh model
    occ.synchronize()
    
    # Get new tags - fragment changes original tags:
    # out_map[0] is new tag for air, out_map[1] for ball, rest for coils
    new_air = out_map[0][0][1]
    new_ball = out_map[1][0][1]
    new_coils = [m[0][1] for m in out_map[2:]]

    # Tag materials
    gmsh.model.addPhysicalGroup(3, [new_air], tag=100, name="air")
    gmsh.model.addPhysicalGroup(3, [new_ball], tag=200, name="ball")
    if new_coils:
        gmsh.model.addPhysicalGroup(3, new_coils, tag=300, name="coils")

    # Surfaces for post‑processing
    ball_surfs = gmsh.model.getBoundary([(3, new_ball)], oriented=False, combined=False)
    ball_surf_tags = [s[1] for s in ball_surfs if s[0] == 2]
    if ball_surf_tags:
        gmsh.model.addPhysicalGroup(2, ball_surf_tags, tag=210, name="ball_surface")

    # gmsh.fltk.run()

    # Outer boundary (for potential reference)
    air_surfs = gmsh.model.getBoundary([(3, new_air)], oriented=False, combined=False)
    air_surf_tags = [s[1] for s in air_surfs if s[0] == 2]
    if air_surf_tags:
        gmsh.model.addPhysicalGroup(2, air_surf_tags, tag=110, name="outer")

    # Mesh sizes
    ball_points = gmsh.model.getBoundary([(3, new_ball)], oriented=False, combined=False, recursive=False)
    ball_point_tags = [p[1] for p in ball_points if p[0] == 0]
    print(ball_point_tags)
    gmsh.model.mesh.setSize(ball_point_tags, lc_ball)

    # for c_tag in new_coils:
    #     coil_points = gmsh.model.getBoundary([(3, c_tag)], oriented=False, combined=False, recursive=True)
    #     coil_point_tags = [p[1] for p in coil_points if p[0] == 0]
    #     gmsh.model.mesh.setSize(coil_point_tags, lc_coil)
    
    all_points = gmsh.model.getEntities(0)
    gmsh.model.mesh.setSize(all_points, lc_air) # overwrite all points without specification 

    gmsh.model.mesh.generate(3)


    gmsh.write(out_msh)


    gmsh.finalize()

    
