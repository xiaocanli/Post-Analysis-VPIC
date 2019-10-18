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

mpi_size=1
node_size=1
fd_tinterval=1
nbinx=128
nbins=80
emin=1E-4
emax=1E4
nbins_high=20
emin_high=1E0
emax_high=1E4
nzone_x=1
nzone_y=1
nzone_z=80

# remove -hf flag if you are using translated binary fields and hydro data
run_dissipation_h5 () {
    cd $ana_path
    srun -n $mpi_size -N $node_size ./particle_energization_io.exec -rp $1 -tf \
        -ts $4 -te $5 -ti $6 -fi $7 -sp $2 -de data -dh data \
        -ph -pr -ft $fd_tinterval -nb $nbins -el $emin -eh $emax \
        -nx $nbinx -nh $nbins_high -eb $emin_high -et $emax_high \
        -zx $nzone_x -zy $nzone_y -zz $nzone_z -hf
        # -ph -pr -ci -ft $fd_tinterval -nx $nbinx
    mkdir -p data/particle_interp/$3
    mv data/particle_interp/*.gda data/particle_interp/$3
    srun -n $mpi_size -N $node_size ./fluid_energization.exec -rp $1 -sp $2 \
        -pp 1 -ft $fd_tinterval -et -pa -hf
    mkdir -p data/fluid_energization/$3
    mv data/fluid_energization/*.gda data/fluid_energization/$3
}

runs_path=/net/scratch4/xiaocanli/reconnection/

run_name=dump_hdf5_test
rootpath=$runs_path/$run_name/
ch_tp2 41
tstart=1810
tend=72400
# tend=3620
tinterval=1810
fields_interval=1810
run_dissipation_h5 $rootpath e $run_name $tstart $tend $tinterval $fields_interval
run_dissipation_h5 $rootpath i $run_name $tstart $tend $tinterval $fields_interval
