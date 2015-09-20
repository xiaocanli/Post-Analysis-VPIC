#!/bin/bash

# export particle=electrons
export particle=ion
export filepath=/net/scratch1/guofan/share/ultra-sigma/sigma1e4-mime100-4000-track/tracer/

mpirun -np 128 ./h5trajectory -d ${filepath} -o data/${particle}s.h5p -n 1000 \
-p ${particle}
