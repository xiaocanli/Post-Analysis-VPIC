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

##### These are shell commands
date
# module swap craype-haswell craype-mic-knl
# module load lustre-default
module load dws
module load cray-hdf5-parallel
module list

time srun -n 128 -N 4 -c 2 --cpu_bind=cores ../combine_spectrum_hdf5.exec -rp /global/cscratch1/sd/xiaocan/3D-Lx150-bg1.0-150ppc-2048KNL/ -ts 0 -te 88680 -ti 2217
# wait

date
echo 'Done'
