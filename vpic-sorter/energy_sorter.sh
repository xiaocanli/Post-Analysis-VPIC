#!/bin/bash

# export filepath=/scratch3/guofan/open3d-h5tracer/tracer
# export filepath=/net/scratch1/guofan/share/ultra-sigma/sigma1e4-mime100-4000-track/tracer
export filepath=/net/scratch2/guofan/turbulent-sheet3D-mixing-trinity/tracer

# Find the maximum time step
export tstep_max=-1
for D in `find $filepath ! -path $filepath -type d`
do
    arr=( $(echo $D | awk -F "." '{print $2}') )
    tstep_tmp=${arr[0]}
    if [ $tstep_tmp -gt $tstep_max ]
    then
        tstep_max=$tstep_tmp
    fi
done

tstep=$tstep_max

export particle=electron

# echo $filepath/T.$tstep/electron_tracer_sorted.h5p
mpirun -np 64 ./h5group-sorter -f $filepath/T.$tstep/${particle}_tracer.h5p \
-o $filepath/T.$tstep/${particle}_tracer_energy_sorted.h5p \
-g /Step#$tstep -m $filepath/T.$tstep/grid_metadata_${particle}_tracer.h5p \
-k 8 -a attribute
