#!/bin/bash
#
#SBATCH -q premium
#SBATCH -N 64
#SBATCH -t 8:00:00
#SBATCH -C haswell
#SBATCH -o translate%j.out
#SBATCH -e translate%j.err
#SBATCH -J translate
#SBATCH -A m3122
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=phyxiaolee@gmail.com
#SBATCH -L SCRATCH,project

#DW persistentdw name=reconnection
#DW stage_out source=$DW_PERSISTENT_STRIPED_reconnection/input/data destination=/global/cscratch1/sd/xiaocan/3D-Lx150-bg0.2-150ppc-2048KNL/data type=directory

##### These are shell commands
date
module swap craype-mic-knl craype-haswell  # Make sure that we are not using KNL libraries
# module load lustre-default
module load dws
module list

export RUNDIR_SCRATCH=/global/cscratch1/sd/xiaocan/3D-Lx150-bg0.2-150ppc-2048KNL
mkdir -p $RUNDIR_SCRATCH/data
lfs setstripe -S 16777216 -c 32 $RUNDIR_SCRATCH/data

cp -r $RUNDIR_SCRATCH/input $DW_PERSISTENT_STRIPED_reconnection/

cd $DW_PERSISTENT_STRIPED_reconnection/input

# time srun -n 2048 -N 64 -c 2 --cpu_bind=cores ./translate.exec -rp /global/cscratch1/sd/xiaocan/3D-Lx150-bg0.2-150ppc-2048KNL/ -fd -sd -nf 32
time srun -n 2048 -N 64 -c 2 --cpu_bind=cores ./translate.exec -rp $DW_PERSISTENT_STRIPED_reconnection/input/ -fd -sd -nf 32
# wait

date
echo 'Done'
