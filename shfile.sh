#!/bin/sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/pkgs_local/cuda-10.2/lib64
export PATH=$PATH:/pkgs_local/cuda-10.2/bin
python num_run_all_er_magellan.py