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

mpi_size=64
node_size=2
fd_tinterval=1
nbinx=128

run_dissipation () {
    cd $ana_path
    srun -n $mpi_size -N $node_size ./particle_energization_io.exec -rp $1 -tf \
        -ts $4 -te $5 -ti $6 -fi $7 -sp $2 -de data -dh data \
        -ft $fd_tinterval -nx $nbinx
        # -ph -pr -ci -ft $fd_tinterval
    mkdir -p data/particle_interp/$3
    mv data/particle_interp/*.gda data/particle_interp/$3
    # srun -n $mpi_size -N $node_size ./fluid_energization.exec -rp $1 -sp $2 \
    #     -pp 1 -ft $fd_tinterval -et -pa
    # mkdir -p data/fluid_energization/$3
    # mv data/fluid_energization/*.gda data/fluid_energization/$3
}

run_dissipation_h5 () {
    cd $ana_path
    srun -n $mpi_size -N $node_size ./particle_energization_io.exec -rp $1 -tf \
        -ts $4 -te $5 -ti $6 -fi $7 -sp $2 -de data -dh data \
        -ph -pr -ft $fd_tinterval -nx $nbinx
        # -ph -pr -ci -ft $fd_tinterval -nx $nbinx
    mkdir -p data/particle_interp/$3
    mv data/particle_interp/*.gda data/particle_interp/$3
    srun -n $mpi_size -N $node_size ./fluid_energization.exec -rp $1 -sp $2 \
        -pp 1 -ft $fd_tinterval -et -pa
    mkdir -p data/fluid_energization/$3
    mv data/fluid_energization/*.gda data/fluid_energization/$3
}

runs_path=/net/scratch3/xiaocanli/reconnection/mime25

run_name=mime25_beta002_bg00
rootpath=$runs_path/$run_name/
ch_tp2 201
tstart=2060
tend=41200
# tstart=14420
# tend=14420
tinterval=2060
fields_interval=206
# run_dissipation $rootpath e $run_name $tstart $tend $tinterval $fields_interval
run_dissipation $rootpath i $run_name $tstart $tend $tinterval $fields_interval

run_name=mime25_beta002_bg02
rootpath=$runs_path/$run_name/
ch_tp2 201
tstart=2060
tend=41200
# tstart=14420
# tend=16480
tinterval=2060
fields_interval=206
# run_dissipation $rootpath e $run_name $tstart $tend $tinterval $fields_interval
run_dissipation $rootpath i $run_name $tstart $tend $tinterval $fields_interval

run_name=mime25_beta002_bg04
rootpath=$runs_path/$run_name/
ch_tp2 201
tstart=2060
tend=41200
# tstart=10300
# tend=12360
tinterval=2060
fields_interval=206
# run_dissipation $rootpath e $run_name $tstart $tend $tinterval $fields_interval
run_dissipation $rootpath i $run_name $tstart $tend $tinterval $fields_interval

run_name=mime25_beta002_bg08
rootpath=$runs_path/$run_name/
ch_tp2 201
tstart=2060
tend=41200
# tstart=10300
# tend=12360
tinterval=2060
fields_interval=206
# run_dissipation $rootpath e $run_name $tstart $tend $tinterval $fields_interval
run_dissipation $rootpath i $run_name $tstart $tend $tinterval $fields_interval


# Runs with different thermal and Alfven speeds
runs_path=/net/scratch4/xiaocanli/reconnection/mime25_high

run_name=mime25_beta002_bg00_high
rootpath=$runs_path/$run_name/
ch_tp2 114
tstart=33100
tend=364100
tinterval=33100
fields_interval=3310
# run_dissipation_h5 $rootpath e $run_name $tstart $tend $tinterval $fields_interval
# run_dissipation_h5 $rootpath i $run_name $tstart $tend $tinterval $fields_interval

run_name=mime25_beta002_bg02_high
rootpath=$runs_path/$run_name/
ch_tp2 114
tstart=33100
tend=364100
tinterval=33100
fields_interval=3310
# run_dissipation_h5 $rootpath e $run_name $tstart $tend $tinterval $fields_interval
# run_dissipation_h5 $rootpath i $run_name $tstart $tend $tinterval $fields_interval

run_name=mime25_beta002_bg04_high
rootpath=$runs_path/$run_name/
ch_tp2 148
tstart=33100
tend=463400
tinterval=33100
fields_interval=3310
# run_dissipation_h5 $rootpath e $run_name $tstart $tend $tinterval $fields_interval
# run_dissipation_h5 $rootpath i $run_name $tstart $tend $tinterval $fields_interval

run_name=mime25_beta002_bg08_high
rootpath=$runs_path/$run_name/
ch_tp2 123
tstart=33100
tend=397200
tinterval=33100
fields_interval=3310
# run_dissipation_h5 $rootpath e $run_name $tstart $tend $tinterval $fields_interval
# run_dissipation_h5 $rootpath i $run_name $tstart $tend $tinterval $fields_interval
