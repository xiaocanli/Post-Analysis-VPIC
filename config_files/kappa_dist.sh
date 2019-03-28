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

mpi_size=32

run_kappa_dist () {
    cd $ana_path
    srun -n $mpi_size ./kappa_dist.exec -rp $1 -nk $2 -kl $3 -kh $4
    mkdir -p data/kappa_dist/$5
    mv data/kappa_dist/*.gda data/kappa_dist/$5
}

runs_path=/net/scratch3/xiaocanli/reconnection/mime25

run_name=mime25_beta002_bg00
rootpath=$runs_path/$run_name/
ch_tp2 201
nbins_kappa=500
kappa_min=1E-2
kappa_max=1E3
run_kappa_dist $rootpath $nbins_kappa $kappa_min $kappa_max $run_name

run_name=mime25_beta002_bg02
rootpath=$runs_path/$run_name/
run_kappa_dist $rootpath $nbins_kappa $kappa_min $kappa_max $run_name

run_name=mime25_beta002_bg04
rootpath=$runs_path/$run_name/
run_kappa_dist $rootpath $nbins_kappa $kappa_min $kappa_max $run_name

run_name=mime25_beta002_bg08
rootpath=$runs_path/$run_name/
run_kappa_dist $rootpath $nbins_kappa $kappa_min $kappa_max $run_name

runs_path=/net/scratch3/xiaocanli/reconnection/mime100

run_name=mime100_beta002_bg00
rootpath=$runs_path/$run_name/
ch_tp2 196
nbins_kappa=500
kappa_min=1E-2
kappa_max=1E3
run_kappa_dist $rootpath $nbins_kappa $kappa_min $kappa_max $run_name

run_name=mime100_beta002_bg02
rootpath=$runs_path/$run_name/
ch_tp2 200
run_kappa_dist $rootpath $nbins_kappa $kappa_min $kappa_max $run_name

run_name=mime100_beta002_bg04
rootpath=$runs_path/$run_name/
ch_tp2 201
run_kappa_dist $rootpath $nbins_kappa $kappa_min $kappa_max $run_name

run_name=mime100_beta002_bg08
rootpath=$runs_path/$run_name/
ch_tp2 201
run_kappa_dist $rootpath $nbins_kappa $kappa_min $kappa_max $run_name

runs_path=/net/scratch4/xiaocanli/reconnection/mime25_high

run_name=mime25_beta002_bg00_high
rootpath=$runs_path/$run_name/
ch_tp2 114
nbins_kappa=500
kappa_min=1E-2
kappa_max=1E3
run_kappa_dist $rootpath $nbins_kappa $kappa_min $kappa_max $run_name

run_name=mime25_beta002_bg02_high
rootpath=$runs_path/$run_name/
ch_tp2 114
run_kappa_dist $rootpath $nbins_kappa $kappa_min $kappa_max $run_name

run_name=mime25_beta002_bg04_high
rootpath=$runs_path/$run_name/
ch_tp2 148
run_kappa_dist $rootpath $nbins_kappa $kappa_min $kappa_max $run_name

run_name=mime25_beta002_bg08_high
rootpath=$runs_path/$run_name/
ch_tp2 123
run_kappa_dist $rootpath $nbins_kappa $kappa_min $kappa_max $run_name

runs_path=/net/scratch4/xiaocanli/reconnection/mime100_high

run_name=mime100_beta002_bg00_high
rootpath=$runs_path/$run_name/
ch_tp2 118
nbins_kappa=500
kappa_min=1E-2
kappa_max=1E3
run_kappa_dist $rootpath $nbins_kappa $kappa_min $kappa_max $run_name

run_name=mime100_beta002_bg02_high
rootpath=$runs_path/$run_name/
ch_tp2 112
run_kappa_dist $rootpath $nbins_kappa $kappa_min $kappa_max $run_name

run_name=mime100_beta002_bg04_high
rootpath=$runs_path/$run_name/
ch_tp2 114
run_kappa_dist $rootpath $nbins_kappa $kappa_min $kappa_max $run_name

run_name=mime100_beta002_bg08_high
rootpath=$runs_path/$run_name/
ch_tp2 119
run_kappa_dist $rootpath $nbins_kappa $kappa_min $kappa_max $run_name

runs_path=/net/scratch4/xiaocanli/reconnection/mime400

run_name=mime400_beta002_bg00
rootpath=$runs_path/$run_name/
ch_tp2 106
nbins_kappa=500
kappa_min=1E-2
kappa_max=1E3
run_kappa_dist $rootpath $nbins_kappa $kappa_min $kappa_max $run_name

run_name=mime400_beta002_bg02
rootpath=$runs_path/$run_name/
ch_tp2 108
run_kappa_dist $rootpath $nbins_kappa $kappa_min $kappa_max $run_name

run_name=mime400_beta002_bg04
rootpath=$runs_path/$run_name/
ch_tp2 107
run_kappa_dist $rootpath $nbins_kappa $kappa_min $kappa_max $run_name

run_name=mime400_beta002_bg08
rootpath=$runs_path/$run_name/
ch_tp2 111
run_kappa_dist $rootpath $nbins_kappa $kappa_min $kappa_max $run_name
