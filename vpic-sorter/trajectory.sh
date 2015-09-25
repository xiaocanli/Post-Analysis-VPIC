#!/bin/bash

export particle=electron
# export particle=ion
export filepath=/net/scratch1/guofan/share/ultra-sigma/sigma1e4-mime100-4000-track/tracer/

mpirun -np 128 ./h5trajectory -d ${filepath} -o data/${particle}s_2.h5p -n 1000 \
-p ${particle} -r 10

export particle=ion
mpirun -np 128 ./h5trajectory -d ${filepath} -o data/${particle}s_2.h5p -n 1000 \
-p ${particle} -r 10
