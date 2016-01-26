#!/bin/bash

# export filepath=/net/scratch1/guofan/share/ultra-sigma/sigma1e4-mime100-4000-track/tracer/
export filepath=/net/scratch2/guofan/VPIC-trinity/turbulent-sheet3D-mixing-trinity/tracer

export particle=electron
mpirun -np 1 ./h5trajectory -d ${filepath} -o data/${particle}s_2.h5p -n 10 \
    -p ${particle} -r 1

# export particle=ion
# mpirun -np 128 ./h5trajectory -d ${filepath} -o data/${particle}s_2.h5p -n 1000 \
# -p ${particle} -r 10
