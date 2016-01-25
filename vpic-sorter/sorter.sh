#!/bin/bash

# export filepath=/scratch3/guofan/open3d-h5tracer/tracer
# export filepath=/net/scratch1/guofan/share/ultra-sigma/sigma1e4-mime100-4000-track/tracer
export filepath=/net/scratch2/guofan/turbulent-sheet3D-mixing-trinity/tracer
export particle=electron

for D in `find $filepath ! -path $filepath -type d`
do
    # echo $D
    arr=( $(echo $D | awk -F "." '{print $2}') )
    tstep=${arr[0]}
    # if [ $tstep -eq 1002 ]
    # then
    echo $filepath/T.$tstep/${particle}_tracer_sorted.h5p
    mpirun -np 64 ./h5group-sorter -f $filepath/T.$tstep/${particle}_tracer.h5p \
    -o $filepath/T.$tstep/${particle}_tracer_sorted.h5p \
    -g /Step\#$tstep -m $filepath/T.$tstep/grid_metadata_${particle}_tracer.h5p \
    -k 7 -a attribute
    # fi
done
