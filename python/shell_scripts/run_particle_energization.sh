root_path=/net/scratch3/xiaocanli/reconnection/mime25/
cd ../

ptl_ene () {
    pic_run=mime${1}_beta002_bg$2
    run_name=mime${1}_beta002_bg$2
    python high_mass_ratio.py --run_name $run_name $3 --multi_frames --time_loop \
        --tstart $4 --tend $5 --species $6
}

ptl_ene_25 () {
    ptl_ene 25 00 $1 1 20 e
    ptl_ene 25 02 $1 1 20 e
    ptl_ene 25 04 $1 1 20 e
    ptl_ene 25 08 $1 1 20 e
    ptl_ene 25 00 $1 1 20 i
    ptl_ene 25 02 $1 1 20 i
    ptl_ene 25 04 $1 1 20 i
    ptl_ene 25 08 $1 1 20 i
}

ptl_ene_100 () {
    ptl_ene 100 00 $1 1 19 e
    ptl_ene 100 02 $1 1 19 e
    ptl_ene 100 04 $1 1 20 e
    ptl_ene 100 08 $1 1 20 e
    ptl_ene 100 00 $1 1 19 i
    ptl_ene 100 02 $1 1 19 i
    ptl_ene 100 04 $1 1 20 i
    ptl_ene 100 08 $1 1 20 i
}

ptl_ene_400 () {
    ptl_ene 400 00 $1 1 10 e
    ptl_ene 400 02 $1 1 10 e
    ptl_ene 400 04 $1 1 12 e
    ptl_ene 400 08 $1 1 10 e
    ptl_ene 400 00 $1 1 10 i
    ptl_ene 400 02 $1 1 10 i
    ptl_ene 400 04 $1 1 12 i
    ptl_ene 400 08 $1 1 10 i
}

ptl_ene_all () {
    ptl_ene_25 $1
    ptl_ene_100 $1
    ptl_ene_400 $1
}

ptl_ene_all --para_perp
ptl_ene_all --comp_shear
ptl_ene_all --drifts
ptl_ene_all --model_ene
