#!/bin/bash

ana_path=/net/scratch3/xiaocanli/pic_analysis
ana_config=$ana_path/config_files/analysis_config.dat
conf=$ana_path/config_files/conf.dat

ch_tp2 () {
    sed -i -e "s/\(tp2 = \).*/\1$1/" $ana_config
}

run_bulk () {
    cd $ana_path
    mkdir -p data/bulk_internal_energy/$2
    mpirun -np 32 ./bulk_internal_energy.exec -rp $1 -sp e
    mv data/bulk_internal_energy/*dat data/bulk_internal_energy/$2
    mpirun -np 32 ./bulk_internal_energy.exec -rp $1 -sp i
    mv data/bulk_internal_energy/*dat data/bulk_internal_energy/$2
}

# run_name=mime25_beta002_bg00
# run_path=/net/scratch3/xiaocanli/reconnection/mime25/$run_name/
# ch_tp2 201
# run_bulk $run_path $run_name

# run_name=mime25_beta002_bg02
# run_path=/net/scratch3/xiaocanli/reconnection/mime25/$run_name/
# ch_tp2 201
# run_bulk $run_path $run_name

# run_name=mime25_beta002_bg04
# run_path=/net/scratch3/xiaocanli/reconnection/mime25/$run_name/
# ch_tp2 201
# run_bulk $run_path $run_name

# run_name=mime25_beta002_bg08
# run_path=/net/scratch3/xiaocanli/reconnection/mime25/$run_name/
# ch_tp2 201
# run_bulk $run_path $run_name

# run_name=mime100_beta002_bg00
# run_path=/net/scratch3/xiaocanli/reconnection/mime100/$run_name/
# ch_tp2 196
# run_bulk $run_path $run_name

# run_name=mime100_beta002_bg02
# run_path=/net/scratch3/xiaocanli/reconnection/mime100/$run_name/
# ch_tp2 200
# run_bulk $run_path $run_name

# run_name=mime100_beta002_bg04
# run_path=/net/scratch3/xiaocanli/reconnection/mime100/$run_name/
# ch_tp2 201
# run_bulk $run_path $run_name

# run_name=mime100_beta002_bg08
# run_path=/net/scratch3/xiaocanli/reconnection/mime100/$run_name/
# ch_tp2 201
# run_bulk $run_path $run_name

# run_name=mime400_beta002_bg00
# run_path=/net/scratch3/xiaocanli/reconnection/mime400/$run_name/
# ch_tp2 103
# run_bulk $run_path $run_name

# run_name=mime400_beta002_bg02
# run_path=/net/scratch3/xiaocanli/reconnection/mime400/$run_name/
# ch_tp2 103
# run_bulk $run_path $run_name

# run_name=mime400_beta002_bg04
# run_path=/net/scratch3/xiaocanli/reconnection/mime400/$run_name/
# ch_tp2 121
# run_bulk $run_path $run_name

# run_name=mime400_beta002_bg08
# run_path=/net/scratch3/xiaocanli/reconnection/mime400/$run_name/
# ch_tp2 110
# run_bulk $run_path $run_name

# root_path=/net/scratch4/xiaocanli/reconnection/mime25_high
# run_name=mime25_beta002_bg00_high
# run_path=$root_path/$run_name/
# ch_tp2 114
# run_bulk $run_path $run_name

# run_name=mime25_beta002_bg02_high
# run_path=$root_path/$run_name/
# ch_tp2 114
# run_bulk $run_path $run_name

# run_name=mime25_beta002_bg04_high
# run_path=$root_path/$run_name/
# ch_tp2 148
# run_bulk $run_path $run_name

# run_name=mime25_beta002_bg08_high
# run_path=$root_path/$run_name/
# ch_tp2 123
# run_bulk $run_path $run_name

root_path=/net/scratch4/xiaocanli/reconnection/mime100_high
# run_name=mime100_beta002_bg00_high
# run_path=$root_path/$run_name/
# ch_tp2 118
# run_bulk $run_path $run_name

# run_name=mime100_beta002_bg02_high
# run_path=$root_path/$run_name/
# ch_tp2 112
# run_bulk $run_path $run_name

# run_name=mime100_beta002_bg04_high
# run_path=$root_path/$run_name/
# ch_tp2 114
# run_bulk $run_path $run_name

run_name=mime100_beta002_bg08_high
run_path=$root_path/$run_name/
ch_tp2 119
run_bulk $run_path $run_name
