#!/bin/bash
#
#SBATCH -q premium
#SBATCH -N 16
#SBATCH -t 2:00:00
#SBATCH -C haswell
#SBATCH -o energization%j.out
#SBATCH -e energization%j.err
#SBATCH -J energization
#SBATCH -A m3122
##SBATCH -A m2407
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=phyxiaolee@gmail.com
#SBATCH -L SCRATCH,project

##DW jobdw capacity=204800GB access_mode=striped type=scratch pool=wlm_pool

##DW persistentdw name=reconnection
##DW stage_in source=/global/cscratch1/sd/xiaocan/3D-Lx150-bg0.2-150ppc-2048KNL/particle destination=$DW_PERSISTENT_STRIPED_reconnection/particle type=directory
##DW stage_in source=/global/cscratch1/sd/xiaocan/3D-Lx150-bg0.2-150ppc-2048KNL/data destination=$DW_PERSISTENT_STRIPED_reconnection/data type=directory
##DW stage_in source=/global/cscratch1/sd/xiaocan/3D-Lx150-bg0.2-150ppc-2048KNL/Makefile destination=$DW_PERSISTENT_STRIPED_reconnection/Makefile type=file
##DW stage_in source=/global/cscratch1/sd/xiaocan/3D-Lx150-bg0.2-150ppc-2048KNL/reconnection.cc destination=$DW_PERSISTENT_STRIPED_reconnection/reconnection.cc type=file
##DW stage_in source=/global/cscratch1/sd/xiaocan/3D-Lx150-bg0.2-150ppc-2048KNL/info destination=$DW_PERSISTENT_STRIPED_reconnection/info type=file
##DW stage_in source=/global/cscratch1/sd/xiaocan/3D-Lx150-bg0.2-150ppc-2048KNL/info.bin destination=$DW_PERSISTENT_STRIPED_reconnection/info.bin type=file

#DW persistentdw name=magrec_bg

##### These are shell commands
date
module swap craype-mic-knl craype-haswell # make sure that we are using Haswell libraries
# module load lustre-default
module load dws
module load cray-hdf5-parallel
module list

# ./energization_cori.sh
./energetic_density.sh

date
echo 'Done'
