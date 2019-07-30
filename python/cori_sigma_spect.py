#!/usr/bin/env python3
"""
Particle energy spectrum for the Cori runs
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
from scipy.optimize import curve_fit

import fitting_funcs
import pic_information
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


def plot_spectrum(plot_config):
    """Plot spectrum for all time frames for a single run

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    tstart = plot_config["tstart"]
    tend = plot_config["tend"]
    pic_run = plot_config["pic_run"]
    root_dir = "/net/scratch3/xiaocanli/reconnection/Cori_runs/"
    pic_run_dir = root_dir + pic_run + "/"
    species = plot_config["species"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    if species == 'e':
        vth = pic_info.vthe
        sname = 'e'
    else:
        vth = pic_info.vthi
        sname = 'i'
    ebins = np.logspace(-4, 6, 1000)
    if species in ['i', 'ion', 'proton']:
        ebins *= pic_info.mime
    dt_particles = pic_info.dt_particles  # in 1/wci
    nframes = tend - tstart + 1
    dtf = math.ceil(pic_info.dt_particles / 0.1) * 0.1

    fig = plt.figure(figsize=[7, 5])
    rect = [0.13, 0.16, 0.7, 0.8]
    ax = fig.add_axes(rect)
    for tframe in range(tstart, tend + 1):
        print("Time frame: %d" % tframe)
        tindex = pic_info.eparticle_interval * tframe
        fname = (pic_run_dir + "spectrum_combined/spectrum_" +
                 species + "_" + str(tindex) + ".dat")
        spect = np.fromfile(fname, dtype=np.float32)
        ndata, = spect.shape
        spect[3:] /= np.gradient(ebins)
        spect[spect == 0] = np.nan
        ax.loglog(ebins, spect[3:], linewidth=1,
                  color = plt.cm.Spectral_r((tframe - tstart)/float(nframes), 1))
    if species == 'e':
        pindex = -2.5
        power_index = "{%0.1f}" % pindex
        pname = r'$\propto (\gamma - 1)^{' + power_index + '}$'
        fpower = 1E13*ebins**pindex
        ax.loglog(ebins, fpower, linewidth=1, color='k', linestyle='--')
    else:
        pindex = -3.5
        power_index = "{%0.1f}" % pindex
        pname = r'$\propto (\gamma - 1)^{' + power_index + '}$'
        fpower = 1E15*ebins**pindex
        ax.loglog(ebins, fpower, linewidth=1, color='k', linestyle='--')
    ax.text(0.94, 0.85, pname, color='k', fontsize=20,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='right', verticalalignment='center',
            transform=ax.transAxes)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top=True)
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in', left=True)
    ax.tick_params(axis='y', which='major', direction='in')
    if species in ['e', 'electron']:
        ax.set_xlim([1E-1, 1E3])
    else:
        ax.set_xlim([1E-1, 1E3])
    if '3D' in pic_run:
        ax.set_ylim([1E0, 1E12])
        ax.set_yticks(np.logspace(0, 10, num=6))
    else:
        ax.set_ylim([1E-1, 1E9])
        ax.set_yticks(np.logspace(-1, 9, num=6))
    text1 = r'$(\gamma - 1)m_' + species + r'c^2$'
    ax.set_xlabel(text1, fontsize=20)
    ax.set_ylabel(r'$f(\gamma - 1)$', fontsize=20)
    ax.tick_params(labelsize=16)
    rect_cbar = np.copy(rect)
    rect_cbar[0] += rect[2] + 0.02
    rect_cbar[2] = 0.03
    cax = fig.add_axes(rect_cbar)
    colormap = plt.cm.get_cmap('jet', tend - tstart + 1)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Spectral_r,
                               norm=plt.Normalize(vmin=tstart * dtf,
                                                  vmax=tend * dtf))
    cax.tick_params(axis='x', which='major', direction='in')
    # fake up the array of the scalar mappable. Urgh...
    sm._A = []
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(r'$t\Omega_{ci}$', fontsize=16)
    cbar.ax.tick_params(labelsize=12)
    fdir = '../img/cori_3d/spectrum/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + 'spectrum_' + species + '_2.pdf'
    fig.savefig(fname)
    plt.show()


def plot_spectrum_both(plot_config, show_plot=True):
    """Plot spectrum both species

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    tstart = plot_config["tstart"]
    tend = plot_config["tend"]
    tframe = plot_config["tframe"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    species = plot_config["species"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    vthe = pic_info.vthe
    gama = 1.0 / math.sqrt(1.0 - 3 * vthe**2)
    ethe = gama - 1.0
    vthi = pic_info.vthi
    gama = 1.0 / math.sqrt(1.0 - 3 * vthi**2)
    ethi = gama - 1.0
    ebins = np.logspace(-6, 4, 1000)
    ebins_e = ebins / ethe
    ebins_i = ebins / ethi
    dt_particles = pic_info.dt_particles  # in 1/wci
    nframes = tend - tstart + 1
    dtf = math.ceil(pic_info.dt_particles / 0.1) * 0.1

    fig = plt.figure(figsize=[3.5, 2.5])
    rect = [0.16, 0.16, 0.8, 0.8]
    ax = fig.add_axes(rect)
    ax.set_prop_cycle('color', COLORS)
    # for tframe in range(tstart_plot, tend_plot + 1):
    print("Time frame: %d" % tframe)
    tindex = pic_info.particle_interval * tframe
    fname = (pic_run_dir + "spectrum_combined/spectrum_e_" + str(tindex) + ".dat")
    spect_e = np.fromfile(fname, dtype=np.float32)
    fname = (pic_run_dir + "spectrum_combined/spectrum_i_" + str(tindex) + ".dat")
    spect_i = np.fromfile(fname, dtype=np.float32)
    spect_e[3:] /= np.gradient(ebins_e)
    spect_i[3:] /= np.gradient(ebins_i)

    # initial thermal distribution
    fname = (pic_run_dir + "spectrum_combined/spectrum_e_0.dat")
    spect_init = np.fromfile(fname, dtype=np.float32)
    ndata, = spect_init.shape
    spect_init[3:] /= np.gradient(ebins_e)

    ax.loglog(ebins_e, spect_init[3:], linewidth=1, color='k',
              linestyle='--', label='initial')
    ax.loglog(ebins_e, spect_e[3:], linewidth=1, label='electron')
    ax.loglog(ebins_i, spect_i[3:], linewidth=1, label='ion')
    ax.legend(loc=3, prop={'size': 10}, ncol=1,
              shadow=False, fancybox=False, frameon=False)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top='on')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in', left='on')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlim([1E-1, 2E3])
    ax.set_ylim([1E-1, 2E12])
    ax.set_yticks(np.logspace(-1, 11, num=7))
    ax.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$', fontsize=10)
    ax.set_ylabel(r'$f(\varepsilon)$', fontsize=10)
    ax.tick_params(labelsize=8)
    fdir = '../img/cori_3d/spectrum/' + pic_run + '/both_species/'
    mkdir_p(fdir)
    fname = fdir + 'spectrum_both_' + str(tframe) + '.pdf'
    fig.savefig(fname)
    if show_plot:
        plt.show()
    else:
        plt.close()


def compare_spectrum(plot_config):
    """Compare 2D and 3D spectra

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    bg = plot_config["bg"]
    root_dir = "/net/scratch3/xiaocanli/reconnection/Cori_runs/"
    pic_runs = ["2D-sigmae100-Lx125-bg0.0-100ppc-15gr"]
    pic_runs.append("3D-sigmae100-Lx125-bg0.0-100ppc-1024KNL")
    tframes = np.asarray([0, 20, 40, 52])
    fig = plt.figure(figsize=[7, 5])
    rect = [0.14, 0.14, 0.8, 0.8]
    ax = fig.add_axes(rect)
    # colors = np.copy(COLORS)
    # colors[5] = colors[6]
    colors = palettable.tableau.Tableau_10.mpl_colors
    for irun, pic_run in enumerate(pic_runs):
        pic_run_dir = root_dir + pic_run + "/"
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        if species == 'e':
            vth = pic_info.vthe
        else:
            vth = pic_info.vthi
        ebins = np.logspace(-4, 6, 1000)
        if species in ['i', 'ion', 'proton']:
            ebins *= pic_info.mime

        if irun == 0:
            fnorm = 1
            lstyle = '--'
        else:
            fnorm = pic_info.ny
            lstyle = '-'

        tinterval = pic_info.eparticle_interval
        dtf = math.ceil(pic_info.dt_fields / 0.1) * 0.1
        pdim = "2D" if "2D" in pic_run else "3D"
        for iframe, tframe in enumerate(tframes):
            tindex = tinterval * tframe
            fname = (pic_run_dir + "spectrum_combined/spectrum_" +
                     species + "_" + str(tindex) + ".dat")
            spect = np.fromfile(fname, dtype=np.float32)
            ndata, = spect.shape
            spect[3:] /= np.gradient(ebins) * fnorm
            if iframe == len(tframes) - 1:
                eindex_20, ene = find_nearest(ebins, 20)
            spect[spect == 0] = np.nan
            if iframe > 0:
                ax.loglog(ebins, spect[3:], linewidth=1,
                          linestyle=lstyle, color=colors[iframe - 1])
            else:
                if irun == 1:
                    ax.loglog(ebins, spect[3:], linewidth=1,
                              linestyle='--', color='k')

    if species == 'e':
        fpower = 1E10 * ebins**-2.5
        power_index = "{%0.1f}" % -2.5
        pname = r'$\sim \varepsilon^{' + power_index + '}$'
        ax.loglog(ebins[548:648], fpower[548:648], linewidth=1, color='k')
    else:
        fpower = 2E12 * ebins**-3.5
        power_index = "{%0.1f}" % -3.5
        pname = r'$\sim \varepsilon^{' + power_index + '}$'
        ax.loglog(ebins[368:468], fpower[368:468], linewidth=1, color='k')
    ax.text(0.95, 0.7, pname, color='k', fontsize=20,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='right', verticalalignment='center',
            transform=ax.transAxes)

    ax.plot([0, 10], [0, 0], linestyle="--", color='k',
            linewidth=1, label='2D')
    ax.plot([0, 10], [0, 0], linestyle="-", color='k',
            linewidth=1, label='3D')
    ax.legend(loc=3, prop={'size': 20}, ncol=1,
              shadow=False, fancybox=False, frameon=False)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top=True)
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    if species in ['e', 'electron']:
        ax.set_xlim([1E-1, 2E3])
    else:
        ax.set_xlim([1E-1, 2E3])
    ax.set_ylim([1E-5, 1E9])
    text1 = r'$(\gamma - 1)m_' + species + r'c^2$'
    ax.set_xlabel(text1, fontsize=20)
    ax.set_ylabel(r'$f(\gamma - 1)$', fontsize=20)
    ax.tick_params(labelsize=16)

    if species == 'e':
        xpos = [0.4, 0.7, 0.87, 0.94]
    else:
        xpos = [0.4, 0.67, 0.86, 0.93]
    text1 = r'$t\Omega_{ci}=0$'
    ax.text(xpos[0], 0.10, text1, color='k', fontsize=16,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='bottom',
            transform=ax.transAxes)
    text2 = r'$' + str(int(tframes[1]*10)) + '$'
    ax.text(xpos[1], 0.10, text2, color=colors[0], fontsize=16,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='bottom',
            transform=ax.transAxes)
    text3 = r'$' + str(int(tframes[2]*10)) + '$'
    ax.text(xpos[2], 0.10, text3, color=colors[1], fontsize=16,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='bottom',
            transform=ax.transAxes)
    text4 = r'$' + str(int(tframes[3]*10)) + '$'
    ax.text(xpos[3], 0.10, text4, color=colors[2], fontsize=16,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes)

    fdir = '../img/cori_sigma/spectrum/'
    mkdir_p(fdir)
    fname = fdir + species + 'spect_32.pdf'
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
    parser.add_argument('--tend', action="store", default='52', type=int,
                        help='ending time frame')
    parser.add_argument('--bg', action="store", default='0.2', type=float,
                        help='Normalized guide field strength')
    parser.add_argument('--whole_spectrum', action="store_true", default=False,
                        help='whether to plot spectrum in the whole box')
    parser.add_argument('--binary', action="store_true", default=False,
                        help='whether spectrum in binary format')
    parser.add_argument('--single_run', action="store_true", default=False,
                        help="whether to plot for a single run")
    parser.add_argument('--show_plot', action="store_true", default=False,
                        help='whether to show plot')
    parser.add_argument('--both_species', action="store_true", default=False,
                        help='whether to plot spectra for both species')
    parser.add_argument('--compare_spectrum', action="store_true", default=False,
                        help='whether to compare 2D and 3D spectra')
    return parser.parse_args()


def analysis_single_frames(plot_config, args):
    """Analysis for multiple time frames
    """
    tframe = args.tframe
    if args.whole_spectrum:
        if args.single_run:
            plot_spectrum(plot_config)
    elif args.both_species:
        plot_spectrum_both(plot_config)
    elif args.compare_spectrum:
        compare_spectrum(plot_config)


def process_input(plot_config, args, tframe):
    """process one time frame"""
    plot_config["tframe"] = tframe
    # if args.mom_spectrum:
        # plot_momentum_spectrum_single(plot_config, show_plot=False)


def analysis_multi_frames(plot_config, args):
    """Analysis for multiple time frames
    """
    tframes = range(plot_config["tstart"], plot_config["tend"] + 1)
    if args.time_loop:
        for tframe in tframes:
            print("Time frame: %d" % tframe)
            plot_config["tframe"] = tframe
            if args.both_species:
                plot_spectrum_both(plot_config, show_plot=False)
    else:
        ncores = multiprocessing.cpu_count()
        ncores = 8
        Parallel(n_jobs=ncores)(delayed(process_input)(plot_config, args, tframe)
                                for tframe in tframes)


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
    plot_config["binary"] = args.binary
    if args.multi_frames:
        analysis_multi_frames(plot_config, args)
    else:
        analysis_single_frames(plot_config, args)


if __name__ == "__main__":
    main()
