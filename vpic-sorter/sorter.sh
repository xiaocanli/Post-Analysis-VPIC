#!/bin/bash

# export filepath=/scratch3/guofan/open3d-h5tracer/tracer
export filepath=/net/scratch1/guofan/share/ultra-sigma/sigma1e4-mime100-4000-track/tracer

for D in `find $filepath ! -path $filepath -type d`
do
    # echo $D
    arr=( $(echo $D | awk -F "." '{print $2}') )
    tstep=${arr[0]}
    echo $filepath/T.$tstep/electron_tracer.h5p
    mpirun -np 128 ./h5group-sorter -f $filepath/T.$tstep/electron_tracer.h5p -g /Step#$tstep -m $filepath/T.$tstep/grid_metadata_electron_tracer.h5p -o $filepath/T.$tstep/electron_tracer_sorted.h5p -k 7 -a attribute
done
