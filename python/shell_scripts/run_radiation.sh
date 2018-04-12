root_path=/net/scratch2/xiaocanli/vpic_radiation/reconnection/grizzly/guide_field_scaling_16000_8000

# run_name=sigma4E4_bg00_rad_vthe100_cool100
# python radiation_cooling.py --run_dir $root_path/$run_name/ --run_name $run_name
# run_name=sigma4E4_bg02_rad_vthe100_cool100
# python radiation_cooling.py --run_dir $root_path/$run_name/ --run_name $run_name
# run_name=sigma4E4_bg05_rad_vthe100_cool100
# python radiation_cooling.py --run_dir $root_path/$run_name/ --run_name $run_name
run_name=sigma4E4_bg10_rad_vthe100_cool100
python radiation_cooling.py --run_dir $root_path/$run_name/ --run_name $run_name
