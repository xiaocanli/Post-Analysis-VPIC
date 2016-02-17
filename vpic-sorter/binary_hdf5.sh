#!/bin/bash

export filepath=/net/scratch1/hli/trinity-run1
export fpath_binary=$filepath/tracer
export fpath_hdf5=$filepath/tracer_hdf5

export particle=electron
tstep_max=14
# tstep_max=2996
tstep_min=14
tinterval=14
ncpus=1024
# ncpus=32768

# Create the directories
if [ ! -d "$fpath_hdf5" ]; then
    mkdir $fpath_hdf5
fi
i=$tstep_min
while [ "$i" -le "$tstep_max" ]; do
    if [ ! -d "$fpath_hdf5/T.$i" ]; then
        mkdir $fpath_hdf5/T.$i
    fi
    i=$(($i+$tinterval))
done

echo "Minimum and maximum time step:" $tstep_min $tstep_max
echo "Time interval:" $tinterval
echo "Particle species:" $particle
echo "Binary file path:" $filepath
echo "Number of CPUs used in PIC simulation:" $ncpus

mpirun -np 16 ./binary_to_hdf5 --tmin=$tstep_min --tmax=$tstep_max \
    --tinterval=$tinterval --fpath_binary=$fpath_binary \
    --fpath_hdf5=$fpath_hdf5 --species=${particle} --ncpus=$ncpus
