#!/bin/bash

ana_path=/net/scratch3/xiaocanli/pic_analysis
ana_config=$ana_path/config_files/analysis_config.dat
conf=$ana_path/config_files/conf.dat
ch_inductive () {
    sed -i -e "s/\(inductive = \).*/\1$1/" $ana_config
}
ch_tp2 () {
    sed -i -e "s/\(tp2 = \).*/\1$1/" $ana_config
}

ch_htt () {
    sed -i -e "s/\(httx = \).*,\(.*\)/\1$1,\2/" $conf
    sed -i -e "s/\(htty = \).*,\(.*\)/\1$2,\2/" $conf
    sed -i -e "s/\(httz = \).*,\(.*\)/\1$3,\2/" $conf
}

mpi_size=128
node_size=4
mpi_sizex=8
mpi_sizey=8
mpi_sizez=2

combine_spectrum () {
    cd $ana_path
    srun -n $mpi_size ./combine_spectrum_binary.exec \
        -rp $1 -op $2 -ts $3 -te $4 -ti $5 -mx $mpi_sizex \
        -my $mpi_sizey -mz $mpi_sizez
}

runs_path=/net/scratch3/stanier/CORI-RUN1/
run_name=LOCAL-SPECTRA-NEW
rootpath=$runs_path/$run_name/
output_path=/net/scratch3/xiaocanli/reconnection/NERSC_ADAM/$run_name/
mkdir -p $output_path
# tstart=0
tstart=16392
# tend=0
tend=147528
tinterval=2732

combine_spectrum $rootpath $output_path $tstart $tend $tinterval
