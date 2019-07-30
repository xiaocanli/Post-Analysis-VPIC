"""
#!/usr/bin/env python3
"""
from __future__ import print_function

import argparse
import itertools
import json
import math
import multiprocessing

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import palettable
from matplotlib.colors import LogNorm, SymLogNorm
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy.interpolate import interp1d

import fitting_funcs
import pic_information
from contour_plots import read_2d_fields
from joblib import Parallel, delayed
from json_functions import read_data_from_json
from shell_functions import mkdir_p

plt.style.use("seaborn-deep")
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = \
[r"\usepackage{amsmath, bm}",
 r"\DeclareMathAlphabet{\mathsfit}{\encodingdefault}{\sfdefault}{m}{sl}",
 r"\SetMathAlphabet{\mathsfit}{bold}{\encodingdefault}{\sfdefault}{bx}{sl}",
 r"\newcommand{\tensorsym}[1]{\bm{\mathsfit{#1}}}"]
COLORS = palettable.colorbrewer.qualitative.Set1_9.mpl_colors

def find_nearest(array, value):
    """Find nearest value in an array
    """
    idx = (np.abs(array-value)).argmin()
    return (idx, array[idx])


def energy_conversion(plot_config):
    """Plot energy conversion

    Args:
        plot_config: plotting configuration
    """
    pic_run = plot_config["pic_run"]
    root_dir = "/net/scratch3/xiaocanli/reconnection/Cori_runs/"
    pic_run_dir = root_dir + pic_run + '/'
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    fig = plt.figure(figsize=[3.25, 2.5])
    w1, h1 = 0.78, 0.78
    xs, ys = 0.96 - w1, 0.96 - h1
    ax = fig.add_axes([xs, ys, w1, h1])
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    ax.set_prop_cycle('color', COLORS)

    tenergy = pic_info.tenergy
    enorm = pic_info.ene_magnetic[0]

    ene_bx = pic_info.ene_bx
    ene_by = pic_info.ene_by
    ene_bz = pic_info.ene_bz
    ene_magnetic = pic_info.ene_magnetic
    ene_electric = pic_info.ene_electric
    kene_e = pic_info.kene_e
    kene_i = pic_info.kene_i

    etot = ene_magnetic + ene_electric + kene_e + kene_i
    dene_mag = ene_magnetic[-1] - ene_magnetic[0]
    dene_ele = ene_electric[-1] - ene_electric[0]
    print("Energy conservation: %e" % ((etot[-1] - etot[0]) / etot[0]))
    print("Energy conversion: %e" % (dene_mag / ene_magnetic[0]))
    print("Electron gain: %e" % ((kene_e[-1] - kene_e[0]) / abs(dene_mag)))
    print("Ion gain: %e" % ((kene_i[-1] - kene_i[0]) / abs(dene_mag)))
    print("Electric: %e" % (dene_ele / abs(dene_mag)))

    ene_bx /= enorm
    ene_by /= enorm
    ene_bz /= enorm
    ene_magnetic /= enorm
    kene_e /= enorm
    kene_i /= enorm

    ax.plot(tenergy, ene_magnetic, linewidth=1, linestyle='-')
    ax.plot(tenergy, kene_e, linewidth=1, linestyle='-')
    ax.plot(tenergy, kene_i, linewidth=1, linestyle='-')

    ax.text(0.03, 0.6, "magnetic", color=COLORS[0], fontsize=12,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)
    ax.text(0.03, 0.5, "electron", color=COLORS[1], fontsize=12,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)
    ax.text(0.03, 0.4, "ion", color=COLORS[2], fontsize=12,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top=False)
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in', left=False)
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlim([0, 260])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel(r'$t\Omega_{ci}$', fontsize=12)
    ax.set_ylabel(r'$\text{Energy}/\varepsilon_{B0}$', fontsize=12)
    ax.tick_params(labelsize=10)

    fdir = '../img/cori_3d/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + 'econv.pdf'
    fig.savefig(fname)
    plt.show()


def get_cmd_args():
    """Get command line arguments
    """
    default_pic_run = '3D-sigmae100-Lx125-bg0.0-100ppc-1024KNL'
    default_pic_run_dir = ('/net/scratch3/xiaocanli/reconnection/Cori_runs/' +
                           default_pic_run + '/')
    parser = argparse.ArgumentParser(description='Analysis for Cori 3D runs')
    parser.add_argument('--pic_run', action="store",
                        default=default_pic_run, help='PIC run name')
    parser.add_argument('--pic_run_dir', action="store",
                        default=default_pic_run_dir, help='PIC run directory')
    parser.add_argument('--species', action="store",
                        default="e", help='Particle species')
    parser.add_argument('--tframe', action="store", default='20', type=int,
                        help='Time frame')
    parser.add_argument('--multi_frames', action="store_true", default=False,
                        help='whether to analyze multiple frames')
    parser.add_argument('--time_loop', action="store_true", default=False,
                        help='whether to use a time loop to analyze multiple frames')
    parser.add_argument('--tstart', action="store", default='0', type=int,
                        help='starting time frame')
    parser.add_argument('--tend', action="store", default='40', type=int,
                        help='ending time frame')
    parser.add_argument('--bg', action="store", default='0.2', type=float,
                        help='Guide field strength')
    parser.add_argument('--econv', action="store_true", default=False,
                        help='whether to plot energy conversion')
    parser.add_argument('--show_plot', action="store_true", default=False,
                        help='whether to show plot')
    return parser.parse_args()


def analysis_single_frames(plot_config, args):
    """Analysis for multiple time frames
    """
    tframe = args.tframe
    if args.econv:
        energy_conversion(plot_config)


def process_input(plot_config, args, tframe):
    """process one time frame"""
    plot_config["tframe"] = tframe


def analysis_multi_frames(plot_config, args):
    """Analysis for multiple time frames
    """
    tframes = range(plot_config["tstart"], plot_config["tend"] + 1)
    pass


def main():
    """business logic for when running this module as the primary one!"""
    args = get_cmd_args()
    plot_config = {}
    plot_config["pic_run"] = args.pic_run
    plot_config["pic_run_dir"] = args.pic_run_dir
    plot_config["tframe"] = args.tframe
    plot_config["tstart"] = args.tstart
    plot_config["tend"] = args.tend
    plot_config["species"] = args.species
    plot_config["bg"] = args.bg
    if args.multi_frames:
        analysis_multi_frames(plot_config, args)
    else:
        analysis_single_frames(plot_config, args)


if __name__ == "__main__":
    main()
