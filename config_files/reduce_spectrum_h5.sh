#!/bin/bash

ana_path=/net/scratch3/xiaocanli/pic_analysis
ana_config=$ana_path/config_files/analysis_config.dat
conf=$ana_path/config_files/conf.dat

ndata=1003  # energy bins + 3 magnetic field components

reduce_spectrum () {
    cd $ana_path
    mpirun -np ${14} ./reduce_spectrum_hdf5.exec -rp $1 -ts $2 \
        -te $3 -ti $4 -px $5 -py $6 -pz $7 -nx $8 -ny $9 -nz ${10} \
        -rx ${11} -ry ${12} -rz ${13} -nd $ndata
}

runs_path=/net/scratch3/xiaocanli/reconnection/Cori_runs

# run_name=2D-Lx150-bg0.2-150ppc-16KNL
run_name=2D-Lx150-bg1.0-150ppc-16KNL
rootpath=$runs_path/$run_name/
tstart=0
tend=82720
# tend=0
tinterval=2068
pic_mpi_sizex=512
pic_mpi_sizey=1
pic_mpi_sizez=2
nzonex=1
nzoney=1
nzonez=80
nreducex=8
nreducey=1
nreducez=6
mpi_size=1
# reduce_spectrum $rootpath $tstart $tend $tinterval \
#                 $pic_mpi_sizex $pic_mpi_sizey $pic_mpi_sizez \
#                 $nzonex $nzoney $nzonez \
#                 $nreducex $nreducey $nreducez $mpi_size

# run_name=3D-Lx150-bg0.2-150ppc-2048KNL
run_name=3D-Lx150-bg1.0-150ppc-2048KNL
rootpath=$runs_path/$run_name/
# tstart=0
tstart=86463
tend=88680
# tend=0
tinterval=2217
pic_mpi_sizex=256
pic_mpi_sizey=256
pic_mpi_sizez=2
nzonex=1
nzoney=1
nzonez=80
nreducex=4
nreducey=8
nreducez=6
mpi_size=32
reduce_spectrum $rootpath $tstart $tend $tinterval \
                $pic_mpi_sizex $pic_mpi_sizey $pic_mpi_sizez \
                $nzonex $nzoney $nzonez \
                $nreducex $nreducey $nreducez $mpi_size
