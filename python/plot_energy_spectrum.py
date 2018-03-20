"""
Plotting particle energy spectrum
"""
import argparse
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from json_functions import read_data_from_json
from shell_functions import mkdir_p

mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

FONT = {'family': 'serif',
        'color': 'black',
        'weight': 'normal',
        'size': 24}


def plot_energy_spectrum(run_name, spect_info, species='e'):
    """Plot particle energy spectrum

    Args:
        run_name: PIC simulation run name
        species: 'e' for electrons, 'H' for ions
        spect_info: dictionary for spectra information
    """
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    if species == 'h':
        species = 'H'
    fig = plt.figure(figsize=[7, 5])
    rect = [0.14, 0.16, 0.82, 0.8]
    ax = fig.add_axes(rect)
    ntf = pic_info.ntf
    if species == 'e':
        vth = pic_info.vthe
    else:
        vth = pic_info.vthi
    gama = 1.0 / math.sqrt(1.0 - 3 * vth**2)
    eth = gama - 1.0
    emin_log = math.log10(spect_info["emin"])
    emax_log = math.log10(spect_info["emax"])
    elog = 10**(np.linspace(emin_log, emax_log, spect_info["nbins"]))
    elog /= eth
    for tframe in range(ntf):
        print("Time frame: %d" % tframe)
        fdir = '../data/spectra/' + run_name + '/'
        fname = fdir + 'spectrum-' + species.lower() + '.' + str(tframe)
        flog = np.fromfile(fname)
        color = plt.cm.jet(tframe/float(ntf), 1)
        ax.loglog(elog, flog, linewidth=2, color=color)

    if species == 'e':
        ax.set_xlim([3E-1, 1E3])
        ax.set_ylim([1E1, 1E12])
    else:
        ax.set_xlim([3E-1, 1E3])
        ax.set_ylim([1E4, 1E15])
    ax.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$',
                  fontdict=FONT, fontsize=20)
    ax.set_ylabel(r'$f(\varepsilon)$', fontdict=FONT, fontsize=20)
    ax.tick_params(labelsize=16)
    fpath = "../img/spectra/"
    mkdir_p(fpath)
    fname = fpath + "spect_time_" + run_name + "_" + species + ".pdf"
    fig.savefig(fname)
    plt.show()


def get_cmd_args():
    """Get command line arguments """
    default_run_name = 'mime400_beta002_bg08'
    parser = argparse.ArgumentParser(description='Plotting particle spectra')
    parser.add_argument('--species', action="store", default='e',
                        help='particle species')
    parser.add_argument('--run_name', action="store", default=default_run_name,
                        help='run name')
    return parser.parse_args()


def main():
    """business logic for when running this module as the primary one!"""
    args = get_cmd_args()
    spect_info = {"nbins": 800,
                  "emin": 1E-5,
                  "emax": 1E3}
    plot_energy_spectrum(args.run_name, spect_info, species=args.species)


if __name__ == "__main__":
    main()
