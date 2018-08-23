"""
Plotting particle energy spectrum
"""
import argparse
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import palettable

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
    ntf = spect_info["tmax"]
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
    nptot = pic_info.nx * pic_info.ny * pic_info.nz * pic_info.nppc
    for tframe in range(ntf):
        print("Time frame: %d" % tframe)
        fdir = '../data/spectra/' + run_name + '/'
        fname = fdir + 'spectrum-' + species.lower() + '.' + str(tframe)
        flog = np.fromfile(fname)
        flog /= nptot
        color = plt.cm.jet(tframe/float(ntf), 1)
        ax.loglog(elog, flog, linewidth=2, color=color)

    if species == 'e':
        ax.set_xlim([3E-1, 5E2])
        ax.set_ylim([1E-9, 1E2])
    else:
        ax.set_xlim([3E-1, 2E3])
        # ax.set_ylim([1E-8, 1E4])
    ax.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$',
                  fontdict=FONT, fontsize=20)
    ax.set_ylabel(r'$f(\varepsilon)$', fontdict=FONT, fontsize=20)
    ax.tick_params(labelsize=16)
    fpath = "../img/spectra/"
    mkdir_p(fpath)
    fname = fpath + "spect_time_" + run_name + "_" + species + ".pdf"
    fig.savefig(fname)
    plt.show()


def energy_spectrum_multi(bg, spect_info, species='e'):
    """Plot energy spectra for runs with different guide field
    """
    if species == 'h':
        species = 'H'
    bg_str = str(int(bg * 10)).zfill(2)
    mimes = np.asarray([25, 100, 400])
    tmaxs = np.asarray([114, 114, 102])
    fig = plt.figure(figsize=[7, 5])
    ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    ax.set_prop_cycle('color', COLORS)
    for mime, tmax in zip(mimes, tmaxs):
        run_name = "mime" + str(mime) + "_beta002_bg" + bg_str
        picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        if species == 'e':
            vth = pic_info.vthe
        else:
            vth = pic_info.vthi
        gama = 1.0 / math.sqrt(1.0 - 3 * vth**2)
        eth = gama - 1.0

        fdir = '../data/spectra/' + run_name + '/'
        fname = fdir + 'spectrum-' + species.lower() + '.' + str(tmax)
        flog = np.fromfile(fname)

        emin_log = math.log10(spect_info["emin"])
        emax_log = math.log10(spect_info["emax"])
        elog = 10**(np.linspace(emin_log, emax_log, spect_info["nbins"]))
        flog *= np.gradient(elog)
        elog /= eth
        flog /= np.gradient(elog)
        nptot = pic_info.nx * pic_info.ny * pic_info.nz * pic_info.nppc

        flog /= nptot
        ltext = r"$m_i/m_e=" + str(mime) + "$"
        ax.loglog(elog, flog, linewidth=3, label=ltext)

    ax.legend(loc=3, prop={'size': 16}, ncol=1,
              shadow=False, fancybox=False, frameon=False)
    if species == 'e':
        ax.set_xlim([3E-1, 5E2])
    else:
        ax.set_xlim([3E-1, 1E3])
    ax.set_ylim([1E-11, 1E0])
    ax.set_yticks(np.logspace(-10, 0, num=6))
    ax.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$',
                  fontdict=FONT, fontsize=20)
    ax.set_ylabel(r'$f(\varepsilon)$', fontdict=FONT, fontsize=20)
    ax.tick_params(labelsize=16)
    fpath = "../img/img_high_mime/spectra/"
    mkdir_p(fpath)
    fname = fpath + "spect_bg" + bg_str + "_" + species + ".pdf"
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
    parser.add_argument('--tmax', action="store", default='100', type=int,
                        help='maximum time step')
    parser.add_argument('--mime', action="store", default='25', type=int,
                        help='ion-to-electron mass ratio')
    parser.add_argument('--bg', action="store", default='0.0', type=float,
                        help='ion-to-electron mass ratio')
    parser.add_argument('--multi_runs', action="store_true", default=False,
                        help='whether analyzing multiple runs')
    return parser.parse_args()


def main():
    """business logic for when running this module as the primary one!"""
    args = get_cmd_args()
    spect_info = {"nbins": 800,
                  "emin": 1E-5,
                  "emax": 1E3,
                  "tmax": args.tmax}
    if args.multi_runs:
        energy_spectrum_multi(args.bg, spect_info, species=args.species)
    else:
        plot_energy_spectrum(args.run_name, spect_info, species=args.species)


if __name__ == "__main__":
    main()
