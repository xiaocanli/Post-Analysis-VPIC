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

combine_spectrum_single_species () {
    cd $ana_path
    srun -n $mpi_size ./combine_spectrum_hdf5.exec \
         -rp $1 -sp $2 --pic_mpi_size $3 --nzones $4 \
         --ndata $5 -ts $6 -te $7 -ti $8
}

combine_spectrum () {
    cd $ana_path
    for sp in electron ion
    do
        combine_spectrum_single_species $1 $sp $2 $3 $4 $5 $6 $7
    done
}

combine_spectrum_multi_species () {
    cd $ana_path
    for sp in electron ion
    do
        echo $sp
        combine_spectrum_single_species $1 $sp $2 $3 $4 $5 $6 $7
    done

    for sp in electron_wo_epara ion_wo_epara electron_wo_eparay ion_wo_eparay
    do
        echo $sp
        combine_spectrum_single_species $1 $sp $2 $3 $4 $5 $6 $7
    done

    for sp in electron_wo_egtb ion_wo_egtb electron_egtb ion_egtb
    do
        echo $sp
        combine_spectrum_single_species $1 $sp $2 $3 $4 $5 $6 $7
    done

    for sp in electron_egtb_egtb ion_egtb_egtb
    do
        echo $sp
        combine_spectrum_single_species $1 $sp $2 $3 $4 $5 $6 $7
    done

    # for sp in electron_wo_eperp ion_wo_eperp
    # do
    #     echo $sp
    #     combine_spectrum_single_species $1 $sp $2 $3 $4 $5 $6 $7
    # done
}

runs_path=/net/scratch4/xiaocanli/reconnection/power_law_index/
run_name=high_sigma_test
rootpath=$runs_path/$run_name/
pic_mpi_size=2048
nzones=256
ndata=1003
tstart=0
tend=32960
tinterval=206
# combine_spectrum $rootpath $pic_mpi_size $nzones $ndata \
#                  $tstart $tend $tinterval
# combine_spectrum_multi_species $rootpath $pic_mpi_size $nzones $ndata \
#                                $tstart $tend $tinterval
