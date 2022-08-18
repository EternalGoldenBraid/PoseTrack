#!/bin/bash

# Render 
if [ $# -ne 6 ]; then
  echo "Invalid syntax."
  echo "Syntax: render.sh <recorded_object_name> <cad_model> <recording filename> <output filename> <icp_track_max_iters> <icp_ove6d_max_iters>"
else

  # Without icp refinement 
  python scripts/captured/run_ove6d_icp_o3d_point2plane_captured.py \
    --object_name $1 -o $2 -fin $3 -fout $4 --save --icp_track_max_iters $5 --icp_ove6d_max_iters $6

  ## With icp refinement
  #python scripts/captured/run_ove6d_icp_o3d_point2plane_captured.py \
  #  -o box_synth --object_name $1 -fin $2 -fout $3 --save -icp --icp_track_max_iters $4 --icp_ove6d_max_iters $5
  #python scripts/captured/run_ove6d_icp_o3d_point2plane_captured.py \
  #  -o box --object_name $1 -fin $2 -fout $3 --save -icp --icp_track_max_iters $4 --icp_ove6d_max_iters $5
fi
