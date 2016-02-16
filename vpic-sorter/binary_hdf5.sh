#!/bin/bash

export filepath=/net/scratch1/sli/trinity-run1/tracer

export particle=electron
# tstep_max=28
tstep_max=2996
tstep_min=14
tinterval=14
ncpus=256
echo "Maximum time step:" $tstep_max
echo "Time interval:" $tinterval

mpirun -np 16 ./binary_to_hdf5 --tmin=$tstep_min --tmax=$tstep_max \
    --tinterval=$tinterval --filepath=$filepath --species=${particle} \
    --ncpus=$ncpus
