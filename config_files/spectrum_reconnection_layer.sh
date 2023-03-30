#!/bin/bash

ana_path=/global/u2/x/xiaocan/pic_analysis
ana_config=$ana_path/config_files/analysis_config.dat
conf=$ana_path/config_files/conf.dat

get_spectrum () {
    cd $ana_path
    srun -n ${23} spectrum_reconnection_layer.exec -rp $1 -ts $2 \
        -te $3 -ti $4 -px $5 -py $6 -pz $7 -nx $8 -ny $9 -nz ${10} \
        -nd ${11} -tx ${12} -ty ${13} -tz ${14} -pe ${15} -dt ${16} \
        -el ${17} -eh ${18} -ve ${19} -ip ${20} -op ${21} -is ${22}
        # -rz
}

runs_path=/global/cscratch1/sd/xiaocan/low_beta_3D
deck_name=reconnection.cc

get_max_spectrum_step () {
    tstep_max=-1
    ct=0
    runpath=$1
    for D in `find $runpath/spectrum ! -path $runpath/spectrum -type d`
    do
        arr=( $(echo $D | awk -F "." '{print $3}') )
        tstep_tmp=${arr[0]}
        if [ $tstep_tmp -gt $tstep_max ]
        then
            tstep_max=$tstep_tmp
        fi
        ct=$ct+1
    done
    echo $tstep_max
}

get_spectrum_rec_layer () {
    run_name=$1
    runpath=$runs_path/$run_name
    if grep -i spectrum_interval $runpath/info
    then
        spectrum_interval=$(grep -i spectrum_interval $runpath/info | cut -d"=" -f2 )
    else
        spectrum_interval=$(grep -i fields_interval $runpath/info | cut -d"=" -f2 )
    fi
    tend=$(get_max_spectrum_step $runpath $spectrum_interval )
    pic_mpi_size=$(grep -i nproc $runpath/info | cut -d"=" -f2 )
    nx=$(grep -m1 -i nx $runpath/info | cut -d"=" -f2 | awk -F"E" 'BEGIN{OFMT="%10.10f"} {print $1 * (10 ^ $2)}' )
    ny=$(grep -m1 -i ny $runpath/info | cut -d"=" -f2 | awk -F"E" 'BEGIN{OFMT="%10.10f"} {print $1 * (10 ^ $2)}' )
    nz=$(grep -m1 -i nz $runpath/info | cut -d"=" -f2 | awk -F"E" 'BEGIN{OFMT="%10.10f"} {print $1 * (10 ^ $2)}' )
    nx_zone=$(grep -i nx_zone $runpath/info | cut -d"=" -f2 )
    ny_zone=$(grep -i ny_zone $runpath/info | cut -d"=" -f2 )
    nz_zone=$(grep -i nz_zone $runpath/info | cut -d"=" -f2 )
    topox_pic=$(grep -m1 -i "  double topology_x = " $runpath/$deck_name | cut -d"=" -f2 | cut -d";" -f1)
    topoy_pic=$(grep -m1 -i "  double topology_y = " $runpath/$deck_name | cut -d"=" -f2 | cut -d";" -f1)
    topoz_pic=$(grep -m1 -i "  double topology_z = " $runpath/$deck_name | cut -d"=" -f2 | cut -d";" -f1)
    nx_rank=$(( nx / topox_pic ))
    ny_rank=$(( ny / topoy_pic ))
    nz_rank=$(( nz / topoz_pic ))
    nzonex=$(( nx_rank / nx_zone )) # Number of zones in each PIC MPI rank
    nzoney=$(( ny_rank / ny_zone ))
    nzonez=$(( nz_rank / nz_zone ))
    nzones=$(( nzonex *  nzoney * nzonez ))
    nbins=$(grep -i nbins $runpath/info | cut -d"=" -f2 )
    ndata=$(( nbins + 3 )) # including 3 components of B-field
    emin_spect=$(grep -i emin_spect $runpath/info | cut -d"=" -f2 )
    emax_spect=$(grep -i emax_spect $runpath/info | cut -d"=" -f2 )
    vthe=$(grep -i vthe\/c $runpath/info | cut -d"=" -f2 )
    echo $runpath $tstart $tend $spectrum_interval \
                 $topox_pic $topoy_pic $topoz_pic \
                 $nzonex $nzoney $nzonez $ndata \
                 $topox $topoy $topoz \
                 $particle_energy $density_threshold \
                 $emin_spect $emax_spect $vthe \
                 $input_path $output_path \
                 $input_suffix $mpi_size
    get_spectrum $runpath $tstart $tend $spectrum_interval \
                 $topox_pic $topoy_pic $topoz_pic \
                 $nzonex $nzoney $nzonez $ndata \
                 $topox $topoy $topoz \
                 $particle_energy $density_threshold \
                 $emin_spect $emax_spect $vthe \
                 $input_path $output_path \
                 $input_suffix $mpi_size
}

tstart=0
topox=1 # MPI topology for current analysis
topoy=1
topoz=1
mpi_size=$(( topox * topoy * topoz ))
particle_energy=10
density_threshold=1E-3
input_path=spectrum_reorganize
output_path=spectrum_reconnection_layer
input_suffix=h5
for bg in 0.2 0.4 0.6 1.0
do
    run_name=2D-Lx150-bg$bg-150ppc
    get_spectrum_rec_layer $run_name
done

run_name=2D-Lx150-bg0.2-150ppc-beta008
get_spectrum_rec_layer $run_name

# topoy=32
# mpi_size=$(( topox * topoy * topoz ))
# for bg in 0.2 0.4 0.6 1.0
# do
#     run_name=3D-Lx150-bg$bg-150ppc-2048KNL
#     get_spectrum_rec_layer $run_name
# done

# run_name=3D-Lx150-bg0.2-150ppc-beta008
# get_spectrum_rec_layer $run_name
