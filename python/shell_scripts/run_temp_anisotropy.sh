#!/bin/sh

cd ../

temp_anisotropy () {
    python temperature_anisotropy.py --run_name $1 --run_dir $2 \
        $3 --multi_frames --time_loop --tstart $4 --tend $5 \
        --species $6
}

temp_anisotropy_multi () {
    temp_anisotropy $1 $2 --temp_dist $3 $4 $5
    temp_anisotropy $1 $2 --beta_dist $3 $4 $5
    temp_anisotropy $1 $2 --tratio_dist $3 $4 $5
    temp_anisotropy $1 $2 --tratio_beta $3 $4 $5
}

root_path=/net/scratch3/xiaocanli/reconnection/Cori_runs/
run_name=3D-Lx150-bg0.2-150ppc-2048KNL
run_dir=$root_path/$run_name/
temp_anisotropy_multi $run_name $run_dir 0 40 e
temp_anisotropy_multi $run_name $run_dir 0 40 i

run_name=3D-Lx150-bg1.0-150ppc-2048KNL
run_dir=$root_path/$run_name/
temp_anisotropy_multi $run_name $run_dir 0 40 e
temp_anisotropy_multi $run_name $run_dir 0 40 i
