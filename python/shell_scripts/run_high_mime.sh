#!/bin/sh

cd ../

power_spect () {
    python high_mass_ratio.py --calc_mag_power \
        --multi_frames --tstart 0 --tend 100 --bg $1 --mime $2
    python high_mass_ratio.py --calc_mag_power --const_va \
        --multi_frames --tstart 0 --tend 100 --bg $1 --mime $2
}

power_spect_bg () {
    power_spect 0.0 $1
    power_spect 0.2 $1
    power_spect 0.4 $1
    power_spect 0.8 $1
}

power_spect_bg 25 
power_spect_bg 100
power_spect_bg 400 
