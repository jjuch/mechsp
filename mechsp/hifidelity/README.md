
# Multiphysics FEM Verification (Elmer + Gmsh)

This package contains the initial high‑fidelity verification pipeline.

**What it does now - ONLY DC**
- Builds a 3D geometry with: coil blocks, an iron sphere (ball), and an air box (optionally with a thin shell).
- Generates an Elmer **.sif** input (magneto–quasistatic with DC currents).
- Runs Elmer in batch (requires `ElmerSolver` in PATH).
- Post‑processes results (VTU) to compute **net force** on the sphere via the **Maxwell Stress Tensor (MST)**.
- Compares with the simplified analytical/dipole model already used in the repo.

**What it will do next**
- Extend to time‑harmonic AC cases (rotating field) by emitting a `Frequency`/`Angular Frequency` in the same template.
- Optional ALE/moving mesh co‑sim with Project Chrono.

> Dependencies (install with your package manager):
> - `gmsh` (>=4.10, Python API available)
> - `ElmerSolver` (ElmerFEM)
> - Python packages: `numpy`, `meshio` (for VTU), `scipy` (optional), `pyyaml` (optional)

Run the smoke test (DC analytic vs MST using synthetic dipole field):

```bash
python -m mechsp.hifidelity.mst_selfcheck
```
Run a full Elmer solve after you have gmsh and ElmerSolver installed:

```bash
python -m mechsp.hifidelity.run_dc_verify \
  --L 0.20 --n 12 --m 12 --h 0.02 --ball_r 0.006 --mu_r_ball 400 \
  --I0_file path/to/I0.npy --Nturns 50 --coil_dx 0.004 --coil_dy 0.004 --coil_dz 0.003 \
  --outdir out/elmer_dc
```

## Gmsh geometry/mesh builder
Generates:

Coil blocks (rectangular prisms) located at coil_xy in the plane z=-h
An iron sphere (ball) centered at q_ball
An air box that encloses everything
Physical groups for: air, coils, ball, outer boundary, ball surface

NOTE: Requires gmsh Python API. Install via pip install gmsh or use the official SDK.

