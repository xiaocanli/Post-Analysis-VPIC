#!/bin/bash

ana_path=/net/scratch3/xiaocanli/pic_analysis
ana_config=$ana_path/config_files/analysis_config.dat
conf=$ana_path/config_files/conf.dat

reduce_spectrum () {
    cd $ana_path
    srun -n ${22} ./reduce_organized_spectrum_hdf5.exec -rp $1 -ts $2 \
        -te $3 -ti $4 -px $5 -py $6 -pz $7 -nx $8 -ny $9 -nz ${10} \
        -rx ${11} -ry ${12} -rz ${13} -nd ${14} -tx ${15} -ty ${16} \
        -tz ${17} -ip ${18} -op ${19} -is ${20} -os ${21}
}

runs_path=/net/scratch3/xiaocanli/reconnection/Cori_runs

# run_name=2D-Lx150-bg0.2-150ppc-16KNL
run_name=2D-Lx150-bg1.0-150ppc-16KNL
rootpath=$runs_path/$run_name/
tstart=0
tend=0
# tend=82720
tinterval=2068
pic_mpi_sizex=512 # PIC MPI topology
pic_mpi_sizey=1
pic_mpi_sizez=2
nzonex=1 # Number of zones in each PIC MPI rank
nzoney=1
nzonez=80
nreducex=8 # Reduce factor along each direction
nreducey=1
nreducez=6
ndata=1003 # energy bins + 3 magnetic field components
topox=1 # MPI topology for current analysis
topoy=1
topoz=1
input_path=spectrum_reorganize
output_path=spectrum_reduced
input_suffix=h5
output_suffix=h5
mpi_size=1
# reduce_spectrum $rootpath $tstart $tend $tinterval \
#                 $pic_mpi_sizex $pic_mpi_sizey $pic_mpi_sizez \
#                 $nzonex $nzoney $nzonez \
#                 $nreducex $nreducey $nreducez $ndata \
#                 $topox $topoy $topoz $input_path $output_path \
#                 $input_suffix $output_suffix $mpi_size

run_name=3D-Lx150-bg0.2-150ppc-2048KNL
# run_name=3D-Lx150-bg1.0-150ppc-2048KNL
rootpath=$runs_path/$run_name/
tstart=0
# tend=0
tend=88680
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
ndata=1003 # energy bins + 3 magnetic field components
topox=1    # MPI topology for current analysis
topoy=32
topoz=1
input_path=spectrum_reorganize
output_path=spectrum_reduced
input_suffix=h5
output_suffix=h5
mpi_size=32

# Spectra along x
nreducex=1
nreducey=256
nreducez=160
topox=32    # MPI topology for current analysis
topoy=1
topoz=1
input_path=spectrum_reorganize
output_path=spectrum_along_x

reduce_spectrum $rootpath $tstart $tend $tinterval \
                $pic_mpi_sizex $pic_mpi_sizey $pic_mpi_sizez \
                $nzonex $nzoney $nzonez \
                $nreducex $nreducey $nreducez $ndata \
                $topox $topoy $topoz $input_path $output_path \
                $input_suffix $output_suffix $mpi_size

run_name=3D-Lx150-bg1.0-150ppc-2048KNL
rootpath=$runs_path/$run_name/

reduce_spectrum $rootpath $tstart $tend $tinterval \
                $pic_mpi_sizex $pic_mpi_sizey $pic_mpi_sizez \
                $nzonex $nzoney $nzonez \
                $nreducex $nreducey $nreducez $ndata \
                $topox $topoy $topoz $input_path $output_path \
                $input_suffix $output_suffix $mpi_size
