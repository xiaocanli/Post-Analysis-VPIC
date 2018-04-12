top_dir=/net/scratch3/xiaocanli/reconnection/frequent_dump

plot_particle_compression () {
    python particle_compression.py --species i --run_dir $top_dir/$1/ --run_name $1 --only_plotting --multi_frames
    # python particle_compression.py --species e --run_dir $top_dir/$1/ --run_name $1
    # python particle_compression.py --species i --run_dir $top_dir/$1/ --run_name $1
}

plot_fluid_compression () {
    python compression.py --species e --run_dir $top_dir/$1/ --run_name $1 --multi_frames
    python compression.py --species i --run_dir $top_dir/$1/ --run_name $1 --multi_frames
}

plot_apjl () {
    python apjl_plots.py --species e --run_dir $top_dir/$1/ --run_name $1
}

run_name=mime25_beta002_guide00_frequent_dump
plot_particle_compression $run_name
# run_name=mime25_beta002_guide02_frequent_dump
# plot_particle_compression $run_name
# run_name=mime25_beta002_guide05_frequent_dump
# plot_particle_compression $run_name
# run_name=mime25_beta002_guide10_frequent_dump
# plot_particle_compression $run_name
# run_name=mime25_beta008_guide00_frequent_dump
# plot_particle_compression $run_name
# run_name=mime25_beta032_guide00_frequent_dump
# plot_particle_compression $run_name

# run_name=mime25_beta002_guide00_frequent_dump
# plot_fluid_compression $run_name
# run_name=mime25_beta002_guide02_frequent_dump
# plot_fluid_compression $run_name
# run_name=mime25_beta002_guide05_frequent_dump
# plot_fluid_compression $run_name
# run_name=mime25_beta002_guide10_frequent_dump
# plot_fluid_compression $run_name
# run_name=mime25_beta008_guide00_frequent_dump
# plot_fluid_compression $run_name
# run_name=mime25_beta032_guide00_frequent_dump
# plot_fluid_compression $run_name

# run_name=mime25_beta002_guide00_frequent_dump
# python smooth_fields.py --run_dir $top_dir/$run_name/ --run_name $run_name

# run_name=mime25_beta002_guide00_frequent_dump
# plot_apjl $run_name
# run_name=mime25_beta002_guide02_frequent_dump
# plot_apjl $run_name
# run_name=mime25_beta002_guide05_frequent_dump
# plot_apjl $run_name
# run_name=mime25_beta002_guide10_frequent_dump
# plot_apjl $run_name
# run_name=mime25_beta008_guide00_frequent_dump
# plot_apjl $run_name
# run_name=mime25_beta032_guide00_frequent_dump
# plot_apjl $run_name
