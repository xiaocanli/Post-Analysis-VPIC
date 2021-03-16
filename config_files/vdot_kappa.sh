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

run_vdot_kappa () {
    cd $ana_path
    # srun -n $mpi_size -N $node_size -c 2 ./vdot_kappa.exec -rp $1
    # srun -n $mpi_size -N $node_size -c 2 ./vdot_kappa.exec -rp $1 -sp e -wn
    srun -n $mpi_size -N $node_size -c 2 ./vdot_kappa.exec -rp $1 \
        -nb $2 -kl $3 -kh $4 -we
    mkdir -p data/vkappa_dist/$5
    mv data/vkappa_dist/*.gda data/vkappa_dist/$5
}

runs_path=/global/cscratch1/sd/xiaocan

run_name=3D-Lx150-bg0.2-150ppc-2048KNL
rootpath=$runs_path/$run_name/
# rootpath=$DW_PERSISTENT_STRIPED_reconnection/
ch_tp2 41
nbins=100
vkappa_min=1E-8
vkappa_max=1E2
run_vdot_kappa $rootpath $nbins $vkappa_min $vkappa_max $run_name

run_name=3D-Lx150-bg1.0-150ppc-2048KNL
rootpath=$runs_path/$run_name/
# rootpath=$DW_PERSISTENT_STRIPED_reconnection/
ch_tp2 41
nbins=100
vkappa_min=1E-8
vkappa_max=1E2
run_vdot_kappa $rootpath $nbins $vkappa_min $vkappa_max $run_name
