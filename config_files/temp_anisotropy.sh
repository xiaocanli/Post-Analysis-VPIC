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

mpi_size=2

run_temp_anisotropy () {
    cd $ana_path
    srun -n $mpi_size ./temperature_anisotropy.exec -rp $1 -sp $2 \
        -nt $3 -nb $4 -nr $5 -tl $6 -th $7 -bl $8 -bh $9 \
        -rl ${10} -rh ${11}
    mkdir -p data/temperature_anisotropy/${12}
    mv data/temperature_anisotropy/*.gda data/temperature_anisotropy/${12}
}

runs_path=/net/scratch3/xiaocanli/reconnection/mime25

run_name=mime25_beta002_bg08
rootpath=$runs_path/$run_name/
ch_tp2 201
nbins_temp=700
nbins_beta=600
nbins_tratio=400
temp_min=1E-5
temp_max=100.0
beta_min=1E-3
beta_max=1000.0
tratio_min=1E-2
tratio_max=100.0
run_temp_anisotropy $rootpath e $nbins_temp $nbins_beta $nbins_tratio \
                    $temp_min $temp_max $beta_min $beta_max \
                    $tratio_min $tratio_max $run_name
