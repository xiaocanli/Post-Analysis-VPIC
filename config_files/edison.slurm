#!/bin/bash -l
#SBATCH -p regular
#SBATCH --qos premium
#SBATCH -N 4
#SBATCH -t 16:00:00
#SBATCH -J traj
#SBATCH -o traj.o%j
#SBATCH -L SCRATCH,project

#Edison has 24 cores per compute node
source module_cray.sh
tstart=0
tend=0
tinterval=13
export particle=e
export root_dir=/scratch3/scratchdirs/guofan/open3d-full/
export dir_tracer_hdf5=tracer
export tracer_file=electron_tracer_reduced_sorted.h5p
export meta_file=grid_metadata_electron_tracer_reduced.h5p
echo "Maximum time step:" $tstart
echo "Time interval:" $tinterval

srun -n 66 ./parallel_hdf5 -tf -ro -ts $tstart -te $tend -ti $tinterval \
-sp $particle -ft $tracer_file -fm $meta_file -rp  $root_dir \
-dt $dir_tracer_hdf5
