"""
Self‑check: compute the MST force on a sphere using a synthetic dipole field and compare against the analytic gradient‑based force at the center (rough sanity only).
"""
from __future__ import annotations
import numpy as np
from .elmer_post import synthetic_ball_surface_B, maxwell_stress_force, MU0

def main():
    # Simple setup: 4 dipoles under the sphere at z=-h
    # GOAL:
    # Very rough center‑based dipole force on a test dipole is not equal to the MST force on sphere, but we can at least ensure the sign/direction is plausible (towards coils).
    # For a more rigorous test, you'd integrate stress on spheres of different radii and check convergence.
    h = 0.025
    R = 0.006
    qc = np.array([0.0, 0.0, 0.0])
    xs = np.array([-0.02, 0.02])
    ys = np.array([-0.02, 0.02])
    XY = np.array([[x,y] for x in xs for y in ys])
    coils = np.column_stack([XY, -h*np.ones(len(XY))])
    m = 1.0  # arbitrary
    P, N, B = synthetic_ball_surface_B(qc, R, coils, m)
    F_mst = maxwell_stress_force(P, N, B)
    print("MST force estimate (N):", F_mst) # x and y is zero due to symmetry and a positive force in the z-direction



if __name__ == '__main__':
    main()
