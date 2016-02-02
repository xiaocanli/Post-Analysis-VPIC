#!/bin/bash

# export filepath=/net/scratch2/guofan/VPIC-trinity/turbulent-sheet3D-mixing-trinity/tracer
export filepath=/net/scratch3/guofan/turbulent-sheet3D-mixing-trinity-Jan29-test/tracer

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
# tstep_max=126
# tinterval=6
echo "Maximum time step:" $tstep_max
echo "Time interval:" $tinterval

key_index=8 # Sort by energy
tstep=$tstep_max
mpirun -np 16 ./h5group-sorter -f $filepath/T.$tstep/${particle}_tracer.h5p \
-o $filepath/T.$tstep/${particle}_tracer_energy_sorted.h5p \
-g /Step#$tstep -m $filepath/T.$tstep/grid_metadata_${particle}_tracer.h5p \
-k $key_index -a attribute --tmax=$tstep_max --tinterval=$tinterval \
--filepath=$filepath --species=${particle}

key_index=7 # sort by particle tag
mpirun -np 16 ./h5group-sorter -f $filepath/T.$tstep/${particle}_tracer.h5p \
-o $filepath/T.$tstep/${particle}_tracer_energy_sorted.h5p \
-g /Step#$tstep -m $filepath/T.$tstep/grid_metadata_${particle}_tracer.h5p \
-k $key_index -a attribute --tmax=$tstep_max --tinterval=$tinterval \
--filepath=$filepath --species=${particle} -p -q -w\
--filename_traj=data/${particle}s_2.h5p --nptl_traj=10 --ratio_emax=10
