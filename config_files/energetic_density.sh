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

mpi_size=512
node_size=16
# mpi_size=32
# node_size=1

run_code() {
    cd $ana_path
    srun -n $mpi_size -N $node_size -c 2 ./energetic_particle_density.exec \
        -rp $1 -sp $2 -ts $3 -te $4 -ti $5 -fi $6 -ph -pr \
        -rf $7 -se $8 -pb $9 -nb ${10}
        # -rp $1 -sp $2 -ts $3 -te $4 -ti $5 -fi $6 -ph -pr -ci \
}

runs_path=/global/cscratch1/sd/xiaocan

run_name=3D-Lx150-bg0.2-150ppc-2048KNL
# rootpath=$runs_path/$run_name/
rootpath=$DW_PERSISTENT_STRIPED_reconnection/
ch_tp2 41
tstart=2217
# tend=2217
tend=88680
tinterval=2217
fields_interval=2217
reduce_factor=4
starting_ene=10
power_base=2
nbands=7
# run_code $rootpath e $tstart $tend $tinterval $fields_interval \
#     $reduce_factor $starting_ene $power_base $nbands
# run_code $rootpath i $tstart $tend $tinterval $fields_interval \
#     $reduce_factor $starting_ene $power_base $nbands

run_name=2D-Lx150-bg0.2-150ppc-16KNL
rootpath=$runs_path/$run_name/
# rootpath=$DW_PERSISTENT_STRIPED_reconnection/
ch_tp2 41
tstart=2068
# tend=2068
tend=82720
tinterval=2068
fields_interval=2068
reduce_factor=1
starting_ene=10
power_base=2
nbands=7
# run_code $rootpath e $tstart $tend $tinterval $fields_interval \
#     $reduce_factor $starting_ene $power_base $nbands
# run_code $rootpath i $tstart $tend $tinterval $fields_interval \
#     $reduce_factor $starting_ene $power_base $nbands

run_name=3D-Lx150-bg1.0-150ppc-2048KNL
rootpath=$DW_PERSISTENT_STRIPED_magrec_bg/input/
ch_tp2 41
tstart=2217
# tend=2217
tend=88680
tinterval=2217
fields_interval=2217
reduce_factor=4
starting_ene=10
power_base=2
nbands=7
run_code $rootpath e $tstart $tend $tinterval $fields_interval \
    $reduce_factor $starting_ene $power_base $nbands
run_code $rootpath i $tstart $tend $tinterval $fields_interval \
    $reduce_factor $starting_ene $power_base $nbands
