#!/bin/bash

export filepath=/net/scratch2/guofan/VPIC-trinity/turbulent-sheet3D-mixing-trinity/tracer

# Find the maximum time step
export tstep_max=-1
export ct=0
for D in `find $filepath ! -path $filepath -type d`
do
    arr=( $(echo $D | awk -F "." '{print $2}') )
    tstep_tmp=${arr[0]}
    if [ $tstep_tmp -gt $tstep_max ]
    then
        tstep_max=$tstep_tmp
    fi
    tsteps[ct]=$tstep_tmp
    ct=$ct+1
done

tstep=$tstep_max

# From http://stackoverflow.com/a/11789688
IFS=$'\n' tsorted=($(sort -n <<<"${tsteps[*]}"))
# printf "[%s]\n" "${sorted[@]}"
# Time interval
tinterval=`expr ${tsorted[1]} - ${tsorted[0]}`

export particle=electron
tstep_max=12

# # echo $filepath/T.$tstep/electron_tracer_sorted.h5p
mpirun -np 16 ./h5group-sorter -f $filepath/T.$tstep/${particle}_tracer.h5p \
-o $filepath/T.$tstep/${particle}_tracer_energy_sorted.h5p \
-g /Step#$tstep -m $filepath/T.$tstep/grid_metadata_${particle}_tracer.h5p \
-k 8 -a attribute --tmax=$tstep_max --tinterval=$tinterval -w \
--filepath=$filepath --species=${particle} -p
