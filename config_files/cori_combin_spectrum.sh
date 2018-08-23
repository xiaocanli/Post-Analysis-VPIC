#!/bin/bash
#
#SBATCH -q premium
#SBATCH -N 4
#SBATCH -t 8:00:00
#SBATCH -C haswell
#SBATCH -o combine%j.out
#SBATCH -e combine%j.err
#SBATCH -J combine
#SBATCH -A m3122
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=phyxiaolee@gmail.com
#SBATCH -L SCRATCH,project

##DW persistentdw name=reconnection
##DW stage_in source=/global/cscratch1/sd/xiaocan/3D-Lx150-bg0.2-150ppc-2048KNL/input destination=$DW_PERSISTENT_STRIPED_reconnection/input type=directory
##DW stage_out source=$DW_PERSISTENT_STRIPED_reconnection/input/data destination=/global/cscratch1/sd/xiaocan/3D-Lx150-bg0.2-150ppc-2048KNL/data type=directory


##### These are shell commands
date
# module swap craype-haswell craype-mic-knl
# module load lustre-default
module load dws
module load cray-hdf5-parallel
module list

#sbcast --compress --force --fanout=8 -t 240 ./untar.sh /tmp/untar.sh &  # Untar after stage in.
#sbcast --compress --force --fanout=8 -t 240 ./reconnection.Linux /tmp/reconnection.Linux &
#sbcast --compress --force --fanout=8 -t 240 ./tar.sh /tmp/tar.sh &
#wait

#cd $DW_JOB_STRIPED
##chmod +x ./ff-coll.Linux
##cd $DW_PERSISTENT_STRIPED_FFCOLL
#mkdir KNL64-PROP
#cd KNL64-PROP

# mkdir particle
# lfs setstripe -S 8388608 -c 72 particle
# mkdir spectrum
# lfs setstripe -S 8388608 -c 72 spectrum

#cd $DW_PERSISTENT_STRIPED_reconnection/input
#cd $SLURM_SUBMIT_DIR

# export OMP_NUM_THREADS=4
# export OMP_PLACES=threads
# export OMP_PROC_BIND=spread

time srun -n 128 -N 4 -c 2 --cpu_bind=cores ./combine_spectrum_hdf5.exec -rp /global/cscratch1/sd/xiaocan/3D-Lx150-bg0.2-150ppc-2048KNL/ -ts 0 -te 88680 -ti 2217
# wait

date
echo 'Done'
