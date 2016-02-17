#!/bin/bash

export filepath=/net/scratch3/xiaocanli/turbulent-sheet3D-mixing-trinity-Feb16-test
export fpath_binary=$filepath/tracer
export fpath_hdf5=$filepath/tracer_hdf5

# # Find the maximum time step
# export tstep_max=-1
# export ct=0
# for D in `find $filepath ! -path $filepath -type d`
# do
#     arr=( $(echo $D | awk -F "." '{print $2}') )
#     tstep_tmp=${arr[0]}
#     if [ $tstep_tmp -gt $tstep_max ]
#     then
#         tstep_max=$tstep_tmp
#     fi
#     tsteps[ct]=$tstep_tmp
#     ct=$ct+1
# done
# tstep=$tstep_max

# # From http://stackoverflow.com/a/11789688
# IFS=$'\n' tsorted=($(sort -n <<<"${tsteps[*]}"))
# # printf "[%s]\n" "${sorted[@]}"

# # Time interval
# tinterval=`expr ${tsorted[1]} - ${tsorted[0]}`
# # The minimum time step
# tstep_min=$tsorted[0]

tstep_max=8000
tstep_min=0
tinterval=5

ncpus=256
dataset_num=8
export particle=electron

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

mpirun -np 64 ./binary_to_hdf5 --tmin=$tstep_min --tmax=$tstep_max \
    --tinterval=$tinterval --fpath_binary=$fpath_binary \
    --fpath_hdf5=$fpath_hdf5 --species=${particle} --ncpus=$ncpus \
    --dataset_num=$dataset_num
