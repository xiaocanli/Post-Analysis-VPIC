root_path=/net/scratch3/xiaocanli/reconnection/mime100/

run_name=mime100_beta002_bg00
python combine_energy_spectrum.py --run_dir $root_path/$run_name/ --run_name $run_name --multi_frames

run_name=mime100_beta002_bg02
python combine_energy_spectrum.py --run_dir $root_path/$run_name/ --run_name $run_name --multi_frames

run_name=mime100_beta002_bg04
python combine_energy_spectrum.py --run_dir $root_path/$run_name/ --run_name $run_name --multi_frames

run_name=mime100_beta002_bg08
python combine_energy_spectrum.py --run_dir $root_path/$run_name/ --run_name $run_name --multi_frames
