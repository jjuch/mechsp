#!/bin/bash
python -m mechsp.hifidelity.run_dc_verify \
  --L 0.20 --n 12 --m 12 --h 0.02 --ball_r 0.006 --mu_r_ball 400 \
  --I0_file path/to/I0.npy --Nturns 50 --coil_dx 0.01 --coil_dy 0.01 --coil_dz 0.01 \
  --outdir C:/Users/jjuchem/Documents/PostDoc/MSCA/mechsp/out/elmer_dc
