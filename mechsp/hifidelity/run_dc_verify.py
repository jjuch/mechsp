"""
End‑to‑end DC verification runner:

Build mesh with Gmsh
Convert to Elmer mesh and run solver
Load VTU and compute MST force on ball
Compare with simplified analytic force from existing mechsp.magnetics model
"""
from __future__ import annotations
import argparse, numpy as np, os
from pathlib import Path
import glob
from .gmsh_build import build_mesh
from .elmer_case import DCCase
from .elmer_post import load_vtu, get_total_force_on_ball, maxwell_stress_force
from ..magnetics import grad_Bz_analytic

print("run_dc_verify: ", __name__)
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--L', type=float, default=0.20)
    ap.add_argument('--n', type=int, default=12)
    ap.add_argument('--m', type=int, default=12)
    ap.add_argument('--h', type=float, default=0.02)
    ap.add_argument('--ball_r', type=float, default=0.006)
    ap.add_argument('--mu_r_ball', type=float, default=400.0)
    ap.add_argument('--I0_file', type=str, help='Optional numpy file (N,) of DC coil currents; else uniform 1A')
    ap.add_argument('--Nturns', type=int, default=50)
    ap.add_argument('--coil_dx', type=float, default=0.01)
    ap.add_argument('--coil_dy', type=float, default=0.01)
    ap.add_argument('--coil_dz', type=float, default=0.01)
    ap.add_argument('--outdir', type=str, default='out/elmer_dc')
    return ap.parse_args()

def make_grid(L: float, n: int, m_: int, h: float):
    dx = L / (n + 1)
    dy = L / (m_ + 1)
    xs = (np.arange(n) + 1) * dx - L/2
    ys = (np.arange(m_) + 1) * dy - L/2
    XX, YY = np.meshgrid(xs, ys, indexing='xy')
    coil_xy = np.stack([XX.ravel(), YY.ravel()], axis=1)
    return coil_xy

def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    coil_xy = make_grid(args.L, args.n, args.m, args.h)
    I0 = None
    if args.I0_file and Path(args.I0_file).exists():
        I0 = np.load(args.I0_file)
        assert I0.shape[0] == coil_xy.shape[0]

    msh_path = outdir / 'geom'
    # build_mesh(
    #     str(msh_path) + '.msh',
    #     coil_xy=coil_xy,
    #     h=args.h,
    #     q_ball=(0.0, 0.0, 0.0),
    #     ball_r=args.ball_r,
    #     L=args.L,
    #     coil_size=(args.coil_dx, args.coil_dy, args.coil_dz),
    # )
    # exit()

    # For now use a single I per coil region; if I0 is provided, you could split coil regions by sign/amp.
    case = DCCase(
        workdir=outdir,
        mesh_msh=msh_path,
        mu_r_ball=args.mu_r_ball,
        I0_per_coil=float(1.0 if I0 is None else float(np.mean(I0))),
        Nturns=args.Nturns,
        coil_area=args.coil_dx*args.coil_dy
    )
    case.write_and_run()

    # Post: read results.vtu and integrate MST on ball surface
    vtu_path = outdir / 'mesh'
    vtu_files = glob.glob(str(vtu_path / "results_ball_surface_bc*.vtu"))
    if vtu_files:
        latest_vtu = sorted(vtu_files)[-1]
        print("Processing VTU file: ", str(latest_vtu))
    else:
        raise ValueError("No VTU file found.")
    
    m = load_vtu(str(latest_vtu))
    F = get_total_force_on_ball(m, ball_surf_tag=102)
    print('Net force on ball (N) from Elmer MST:', F)

    
    # Simplified analytic DC force at the ball center (as in repo): F = m_b * sum_i I_i * grad Bz_i(q)\n    m_b = 1.0  
    # Set your marble magnetic moment scale (same as in your repo defaults)
    q_center = np.array([0.0, 0.0])
    G = grad_Bz_analytic(q_center, coil_xy, args.h, scale=1.0)  # shape (N,2)
    if I0 is None:
        I0 = np.ones(coil_xy.shape[0])
    F_simpl = (I0[:,None] * G).sum(axis=0) * args.m
    print('Simplified DC force at center (N, analytic dipole gradient):', F_simpl)


if __name__ == '__main__':
    print("Runninig DC verify")
    main()
    
