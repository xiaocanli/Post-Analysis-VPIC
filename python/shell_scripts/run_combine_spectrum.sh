root_path=/net/scratch3/xiaocanli/reconnection/mime25/
cd ../

pic_run=mime25_beta002_bg00
run_name=mime25_beta002_bg00_lx100
python combine_energy_spectrum.py --run_dir $root_path/$pic_run/ --run_name $run_name --multi_frames

pic_run=mime25_beta002_bg02
run_name=mime25_beta002_bg02_lx100
python combine_energy_spectrum.py --run_dir $root_path/$pic_run/ --run_name $run_name --multi_frames

pic_run=mime25_beta002_bg04
run_name=mime25_beta002_bg04_lx100
python combine_energy_spectrum.py --run_dir $root_path/$pic_run/ --run_name $run_name --multi_frames

pic_run=mime25_beta002_bg08
run_name=mime25_beta002_bg08_lx100
python combine_energy_spectrum.py --run_dir $root_path/$pic_run/ --run_name $run_name --multi_frames
