#!/bin/bash

ana_path=pic_analysis
ana_config=config_files/analysis_config.dat
conf=config_files/conf.dat

run_parspec () {
    cd ../
    mpirun -np 32 ./parspec.exec -rp $1
    mv spectrum/ data/spectra/$2
    cd config_files
}

run_name=mime25_beta002_guide00_frequent_dump
run_path=/net/scratch3/xiaocanli/reconnection/frequent_dump/$run_name/
run_parspec $run_path $run_name

run_name=mime25_beta002_guide05_frequent_dump
run_path=/net/scratch3/xiaocanli/reconnection/frequent_dump/$run_name/
run_parspec $run_path $run_name

run_name=mime25_beta002_guide10_frequent_dump
run_path=/net/scratch3/xiaocanli/reconnection/frequent_dump/$run_name/
run_parspec $run_path $run_name

run_name=mime25_beta008_guide00_frequent_dump
run_path=/net/scratch3/xiaocanli/reconnection/frequent_dump/$run_name/
run_parspec $run_path $run_name

run_name=mime25_beta032_guide00_frequent_dump
run_path=/net/scratch3/xiaocanli/reconnection/frequent_dump/$run_name/
run_parspec $run_path $run_name
