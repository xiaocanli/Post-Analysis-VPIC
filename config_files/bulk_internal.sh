#!/bin/bash

ana_path=/net/scratch3/xiaocanli/pic_analysis
ana_config=$ana_path/config_files/analysis_config.dat
conf=$ana_path/config_files/conf.dat

ch_tp2 () {
    sed -i -e "s/\(tp2 = \).*/\1$1/" $ana_config
}

mpi_size=128

run_bulk () {
    cd $ana_path
    mkdir -p data/bulk_internal_energy/$2
    mpirun -np $mpi_size ./bulk_internal_energy.exec -rp $1 -sp e
    mv data/bulk_internal_energy/*dat data/bulk_internal_energy/$2
    mpirun -np $mpi_size ./bulk_internal_energy.exec -rp $1 -sp i
    mv data/bulk_internal_energy/*dat data/bulk_internal_energy/$2
}

root_path=/net/scratch4/xiaocanli/reconnection/mime400
run_name=mime400_beta002_bg00
run_path=$root_path/$run_name/
ch_tp2 106
run_bulk $run_path $run_name
