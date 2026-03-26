"""
Elmer .sif writer and case runner for DC magneto‑quasistatic verification.
We use Elmer's MagnetoDynamics solver in steady (DC) mode with impressed
current densities in coil volumes.
This module emits:

mesh directory (converted from .msh via ElmerGrid — run externally or via subprocess)
.sif file with materials (air, ball with mu_r), coil regions with Jz source
requests for saving B,H fields and force postprocessing data

Note: You must have ElmerGrid and ElmerSolver available on PATH.
"""
from __future__ import annotations
import os, subprocess, shutil, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict

ELMERGRID = shutil.which("ElmerGrid") or "ElmerGrid"
ELMERSOLVER = shutil.which("ElmerSolver") or "ElmerSolver"

@dataclass
class DCCase:
    workdir: Path
    mesh_msh: Path
    mu_r_ball: float = 5e3
    sigma_ball: float = 1.0e7  # conductivity (S/m);
    mu_r_air: float = 1.0
    sigma_air: float = 1.0e-12
    coil_region_tag: int = 300
    air_region_tag: int = 100
    ball_region_tag: int = 200
    ball_surf_tag: int = 210
    air_surf_tag: int = 110
    I0_per_coil: float = 1.0
    Nturns: int = 50
    coil_area: float = 1e-6  # m^2 (dxdy), used to compute J = NI/area in z

    def _sif_text(self) -> str:
        Jz = self.Nturns * self.I0_per_coil / max(self.coil_area, 1e-12)
        # Jz = 1.0e3
        return f"""Check Keywords "Warn"
! Magneto-quasistatic DC in A-formulation
Header
    Mesh DB "." "mesh"
    Include Path "."
    Results Directory "."
End

Simulation
    Coordinate System = Cartesian 3D
    Max Output Level = 5
    Simulation Type = Steady State
    Steady State Max Iterations = 1
    Output Intervals = 1
    Post File = "results.vtu"
    Output File = "case.dat"
    Simulation Timing = Logical True
End

Constants
    ! SI units
    Permittivity of Vacuum = Real 8.854187817e-12 ! F/m
    Permeability of Vacuum = Real 1.2566370614e-6 ! H/m
End

! --- BODY DEFINITIONS ---
Body 1
    Target Bodies(1) = {self.air_region_tag}
    Name = "air_body"
    Geometry Id = Integer {self.air_region_tag}
    Equation = 1
    Material = 1
End
Body 2
    Target Bodies(1) = {self.ball_region_tag}
    Name = "ball_body"
    Geometry Id = Integer {self.ball_region_tag}
    Equation = 1
    Material = 2
    Calculate Magnetic Force = Logical True
End
Body 3
    Target Bodies(1) = {self.coil_region_tag}
    Name = "coils_body"
    Geometry Id = Integer {self.coil_region_tag}
    Equation = 1
    Material = 1
    Body Force = 1
End

! --- MATERIALS ---
Material 1
    Name = "air"
    Relative Permeability = Real {self.mu_r_air}
    Electric Conductivity = Real {self.sigma_air}
End
Material 2
    Name = "ball"
    Relative Permeability = Real {self.mu_r_ball}
    Electric Conductivity = Real {self.sigma_ball}
End

! --- BODY FORCE ---
Body Force 1
    name = "CurrentSource"
    Current Density 1 = Real 0.0
    Current Density 2 = Real 0.0
    Current Density 3 = Real {Jz}
End

! --- Components ---
Component 1
    Name = "ball-force"
    Calculate Magnetic Force = Logical True
    Master Boundaries = {self.ball_surf_tag}
    Master Bodies = {self.ball_region_tag}
End


! --- SOLVER: MagnetoDynamics ---
Solver 1
    Equation = "MGDynamics"
    Procedure = "MagnetoDynamics" "WhitneyAVSolver"
    Variable = "A"
    Linear System Solver = Iterative
    Linear System Iterative Method = BiCGStabl
    Linear System Preconditioning = ILU2
    Linear System Max Iterations = 2000
    Linear System Convergence Tolerance = 1.0e-5
    Fix Input Current Density = Logical True
    Use Tree Gauge = Logical True
    BiCGstabl polynomial degree = Integer 4
    Automated Source Projection BCs = Logical True
End

! --- SOLVER: Calculate fields ---
Solver 2
    Equation = "MGDynamicsCalc"
    Procedure = "MagnetoDynamics" "MagnetoDynamicsCalcFields"
    Potential Variable = String "A"
    Angular Frequency = Real 0.0
    Calculate Magnetic Field Strength = Logical True
    Calculate Maxwell Stress = Logical True
    Calculate Nodal Forces = Logical True
    Calculate Elemental Fields = Logical True
End

! --- SAVING FIELDS ---
Solver 3
    Equation = "ResultOutput"
    Procedure = "ResultOutputSolve" "ResultOutputSolver"
    Output File Name = "results"

    Vtu Format = Logical True
    Vtu Names = Logical True
    Save Geometry Ids = Logical True
    Save Boundary Values = Logical True
    Save Nodal Fields = Logical True
    Save Elemental Fields = Logical True
    Show Variables = Logical True
    Vtu Part Collection = Logical True
End

! --- EQUATION ---
Equation 1
    Active Solvers(3) = 1 2 3
End

! --- BOUNDARY CONDITIONS ---
! Set A = 0 on outer boundary for reference (Dirichlet)
Boundary Condition 1
    Target Boundaries(1) = {self.air_surf_tag}
    Name = "outer_bc"
    A {{e}} = Real 0.0
End

! Add boundary conditions of the ball
Boundary Condition 2
    Target Boundaries(1) = {self.ball_surf_tag}
    Geomtery Id = Integer {self.ball_surf_tag}
    Name = "ball_surface_bc"
End
"""

    def write_and_run(self) -> None:
        self.workdir.mkdir(parents=True, exist_ok=True)
        # Convert mesh: Gmsh -> Elmer
        meshdir = self.workdir / "mesh"
        if meshdir.exists():
            shutil.rmtree(meshdir)
        cmd_grid = [ELMERGRID, "14", "2", str(self.mesh_msh), "-out", str(meshdir)]
        print("Running:", " ".join(cmd_grid))
        subprocess.check_call(cmd_grid, cwd=self.workdir)
        sif = self.workdir / "case.sif"
        log = self.workdir / "log.txt"
        sif.write_text(self._sif_text())
        cmd_solve = [ELMERSOLVER, str(sif), "-noorder"]
        print("Running:", " ".join(cmd_solve))
        subprocess.check_call(cmd_solve, cwd=self.workdir)