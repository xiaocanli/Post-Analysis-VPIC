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

combine_spectrum () {
    cd $ana_path
    srun -n $mpi_size ./combine_spectrum_hdf5.exec \
         -rp $1 --pic_mpi_size $2 --nzones $3 --ndata $4 -ts $5 -te $6 -ti $7
}

runs_path=/net/scratch4/xiaocanli/reconnection/mime25_high/
run_name=mime25_beta002_bg00_high
rootpath=$runs_path/$run_name/
pic_mpi_size=4096
nzones=1
ndata=1003
tstart=0
tend=374030
tinterval=3310
# combine_spectrum $rootpath $pic_mpi_size $nzones $ndata \
#                  $tstart $tend $tinterval

run_name=mime25_beta002_bg02_high
rootpath=$runs_path/$run_name/
tend=374030
# combine_spectrum $rootpath $pic_mpi_size $nzones $ndata \
#                  $tstart $tend $tinterval

run_name=mime25_beta002_bg04_high
rootpath=$runs_path/$run_name/
tend=486570
combine_spectrum $rootpath $pic_mpi_size $nzones $ndata \
                 $tstart $tend $tinterval

run_name=mime25_beta002_bg08_high
rootpath=$runs_path/$run_name/
tend=403820
combine_spectrum $rootpath $pic_mpi_size $nzones $ndata \
                 $tstart $tend $tinterval

runs_path=/net/scratch4/xiaocanli/reconnection/mime100_high/
run_name=mime100_beta002_bg00_high
rootpath=$runs_path/$run_name/
pic_mpi_size=4096
nzones=1
ndata=1003
tstart=0
tend=387270
tinterval=3310
combine_spectrum $rootpath $pic_mpi_size $nzones $ndata \
                 $tstart $tend $tinterval

run_name=mime100_beta002_bg02_high
rootpath=$runs_path/$run_name/
tend=367410
combine_spectrum $rootpath $pic_mpi_size $nzones $ndata \
                 $tstart $tend $tinterval

run_name=mime100_beta002_bg04_high
rootpath=$runs_path/$run_name/
tend=374030
combine_spectrum $rootpath $pic_mpi_size $nzones $ndata \
                 $tstart $tend $tinterval

run_name=mime100_beta002_bg08_high
rootpath=$runs_path/$run_name/
tend=390580
# combine_spectrum $rootpath $pic_mpi_size $nzones $ndata \
#                  $tstart $tend $tinterval
