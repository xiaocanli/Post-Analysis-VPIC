#!/bin/bash

ana_path=/global/u2/x/xiaocan/pic_analysis
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

# mpi_size=2048
# node_size=64
mpi_size=512
node_size=16
# mpi_size=32
# node_size=1
# mpi_size=128
# node_size=4
# mpi_size=256
# node_size=8
# mpi_size=512
# node_size=16
fd_tinterval=1

run_dissipation () {
    cd $ana_path
    # srun -n $mpi_size -N $node_size -c 2 ./particle_energization_io.exec -rp $1 -tf \
    #     -ts $4 -te $5 -ti $6 -fi $7 -sp $2 -de data -dh data \
    #     -ph -pr -ci -ft $fd_tinterval
    # mkdir -p data/particle_interp/$3
    # mv data/particle_interp/*.gda data/particle_interp/$3
    srun -n $mpi_size -N $node_size -c 2 ./fluid_energization.exec -rp $1 -sp $2 \
        -pp 1 -ft $fd_tinterval
    mkdir -p data/fluid_energization/$3
    mv data/fluid_energization/*.gda data/fluid_energization/$3
}

runs_path=/global/cscratch1/sd/xiaocan

run_name=3D-Lx150-bg0.2-150ppc-2048KNL
rootpath=$runs_path/$run_name/
# rootpath=$DW_PERSISTENT_STRIPED_reconnection/
ch_tp2 41
tstart=2217
tend=88680
tinterval=2217
fields_interval=2217
run_dissipation $rootpath e $run_name $tstart $tend $tinterval $fields_interval
# run_dissipation $rootpath i $run_name $tstart $tend $tinterval $fields_interval

run_name=2D-Lx150-bg0.2-150ppc-16KNL
rootpath=$runs_path/$run_name/
ch_tp2 41
tstart=2068
tend=82720
# tend=2068
tinterval=2068
fields_interval=2068
# run_dissipation $rootpath e $run_name $tstart $tend $tinterval $fields_interval
# run_dissipation $rootpath i $run_name $tstart $tend $tinterval $fields_interval

run_name=2D-Lx150-bg0.2-150ppc-16KNL
rootpath=$DW_JOB_STRIPED/
ch_tp2 41
tstart=2068
tend=2068
tinterval=2068
fields_interval=2068
# run_dissipation $rootpath e $run_name $tstart $tend $tinterval $fields_interval
# run_dissipation $rootpath i $run_name $tstart $tend $tinterval $fields_interval
