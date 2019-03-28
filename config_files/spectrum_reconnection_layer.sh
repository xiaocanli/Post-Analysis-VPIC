#!/bin/bash

ana_path=/net/scratch3/xiaocanli/pic_analysis
ana_config=$ana_path/config_files/analysis_config.dat
conf=$ana_path/config_files/conf.dat

get_spectrum () {
    cd $ana_path
    srun -n ${23} spectrum_reconnection_layer.exec -rp $1 -ts $2 \
        -te $3 -ti $4 -px $5 -py $6 -pz $7 -nx $8 -ny $9 -nz ${10} \
        -nd ${11} -tx ${12} -ty ${13} -tz ${14} -pe ${15} -dt ${16} \
        -el ${17} -eh ${18} -ve ${19} -ip ${20} -op ${21} -is ${22} \
        -rz
}

runs_path=/net/scratch3/xiaocanli/reconnection/Cori_runs

run_name=2D-Lx150-bg0.2-150ppc-16KNL
rootpath=$runs_path/$run_name/
# tstart=0
tstart=2068
# tstart=82720
# tend=2068
tend=82720
tinterval=2068
pic_mpi_sizex=512 # PIC MPI topology
pic_mpi_sizey=1
pic_mpi_sizez=2
nzonex=1 # Number of zones in each PIC MPI rank
nzoney=1
nzonez=80
ndata=1003 # energy bins + 3 magnetic field components
topox=1 # MPI topology for current analysis
topoy=1
topoz=1
particle_energy=10
density_threshold=1E-3
emin=1E-6
emax=1E4
vthe=0.1
input_path=spectrum_reorganize
output_path=spectrum_reconnection_layer
input_suffix=h5
mpi_size=1
# get_spectrum $rootpath $tstart $tend $tinterval \
#              $pic_mpi_sizex $pic_mpi_sizey $pic_mpi_sizez \
#              $nzonex $nzoney $nzonez \
#              $nreducex $nreducey $nreducez $ndata \
#              $topox $topoy $topoz \
#              $particle_energy $density_threshold \
#              $emin $emax $vthe \
#              $input_path $output_path \
#              $input_suffix $mpi_size

run_name=2D-Lx150-bg1.0-150ppc-16KNL
rootpath=$runs_path/$run_name/
# get_spectrum $rootpath $tstart $tend $tinterval \
#              $pic_mpi_sizex $pic_mpi_sizey $pic_mpi_sizez \
#              $nzonex $nzoney $nzonez \
#              $nreducex $nreducey $nreducez $ndata \
#              $topox $topoy $topoz \
#              $particle_energy $density_threshold \
#              $emin $emax $vthe \
#              $input_path $output_path \
#              $input_suffix $mpi_size

run_name=3D-Lx150-bg0.2-150ppc-2048KNL
rootpath=$runs_path/$run_name/
# tstart=0
tstart=2217
# tstart=88680
# tend=0
tend=88680
tinterval=2217
pic_mpi_sizex=256
pic_mpi_sizey=256
pic_mpi_sizez=2
nzonex=1 # Number of zones in each PIC MPI rank
nzoney=1
nzonez=80
ndata=1003 # energy bins + 3 magnetic field components
topox=1 # MPI topology for current analysis
topoy=32
topoz=1
particle_energy=10
density_threshold=1E-3
emin=1E-6
emax=1E4
vthe=0.1
input_path=spectrum_reorganize
output_path=spectrum_reconnection_layer
input_suffix=h5
mpi_size=32
# get_spectrum $rootpath $tstart $tend $tinterval \
#              $pic_mpi_sizex $pic_mpi_sizey $pic_mpi_sizez \
#              $nzonex $nzoney $nzonez \
#              $nreducex $nreducey $nreducez $ndata \
#              $topox $topoy $topoz \
#              $particle_energy $density_threshold \
#              $emin $emax $vthe \
#              $input_path $output_path \
#              $input_suffix $mpi_size

run_name=3D-Lx150-bg1.0-150ppc-2048KNL
rootpath=$runs_path/$run_name/
get_spectrum $rootpath $tstart $tend $tinterval \
             $pic_mpi_sizex $pic_mpi_sizey $pic_mpi_sizez \
             $nzonex $nzoney $nzonez \
             $nreducex $nreducey $nreducez $ndata \
             $topox $topoy $topoz \
             $particle_energy $density_threshold \
             $emin $emax $vthe \
             $input_path $output_path \
             $input_suffix $mpi_size
