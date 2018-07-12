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

mpi_size=16

run_dissipation () {
    cd $ana_path
    mpirun -np $mpi_size ./particle_energization.exec -rp $1 -tf \
        -ts $4 -te $5 -ti $6 -fi $7 -sp $2 -de data -dh data \
        -ph -pa -cs -cg -mm -pt -ps
    mkdir -p data/particle_interp/$3
    mv data/particle_interp/*.gda data/particle_interp/$3
    mpirun -np $mpi_size ./fluid_energization.exec -rp $1 -sp $2 -pp 1
    mkdir -p data/fluid_energization/$3
    mv data/fluid_energization/*.gda data/fluid_energization/$3
}

runs_path=/net/scratch3/xiaocanli/reconnection/Cori_runs

run_name=test_2d_1
rootpath=$runs_path/$run_name/
ch_tp2 3
tstart=5170
tend=82720
tinterval=5170
fields_interval=517
run_dissipation $rootpath e $run_name $tstart $tend $tinterval $fields_interval
run_dissipation $rootpath i $run_name $tstart $tend $tinterval $fields_interval
