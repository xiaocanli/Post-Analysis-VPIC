"""
Analysis procedures for temperature anisotropy
"""
import argparse
import itertools
import math
import multiprocessing
import os.path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import palettable
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
from scipy import signal
from scipy.interpolate import interp1d, interp2d
from scipy.ndimage.filters import median_filter, gaussian_filter

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

FONT = {'family': 'serif',
        'color': 'black',
        'weight': 'normal',
        'size': 24}


def plot_temp_dist(plot_config, show_plot=True):
    """
    Plot temperature distribution
    """
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    species = plot_config["species"]
    tframe = plot_config["tframe"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    tinit = pic_info.vthe**2

    fdir = '../data/temperature_anisotropy/' + pic_run + '/'
    fname = fdir + 'ftpara_' + species + '_' + str(tframe) + '.gda'
    fdata = np.fromfile(fname, dtype=np.float32)
    nbins_temp = int(fdata[0])
    temp_bins_edge = fdata[1:nbins_temp+2]
    temp_bins_mid = 0.5 * (temp_bins_edge[1:] + temp_bins_edge[:-1])
    temp_bins_mid /= tinit
    dtemp = np.diff(temp_bins_edge)
    ftpara = fdata[nbins_temp+2:] / dtemp

    fname = fdir + 'ftperp_' + species + '_' + str(tframe) + '.gda'
    fdata = np.fromfile(fname, dtype=np.float32)
    ftperp = fdata[nbins_temp+2:] / dtemp

    fig = plt.figure(figsize=[3.25, 2.5])
    rect = [0.16, 0.16, 0.79, 0.8]
    ax = fig.add_axes(rect)
    ax.set_prop_cycle('color', COLORS)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', bottom=True, top=True)
    ax.tick_params(axis='x', which='major', direction='in', bottom=True, top=True)
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')

    ax.loglog(temp_bins_mid, ftpara, linewidth=1, label=r'$T_\parallel$')
    ax.loglog(temp_bins_mid, ftperp, linewidth=1, label=r'$T_\perp$')
    ax.legend(loc=1, prop={'size': 10}, ncol=1,
              shadow=False, fancybox=False, frameon=False)
    ax.set_xlim([1E-1, 1E2])
    ax.tick_params(labelsize=8)
    ax.set_xlabel(r'$T/T_0$', fontsize=10)
    ax.set_ylabel(r'$f(T)$', fontsize=10)
    fdir = '../img/temp_dist/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + 'temp_dist_' + species + '_' + str(tframe) + '.pdf'
    fig.savefig(fname)

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_beta_dist(plot_config, show_plot=True):
    """
    Plot plasma beta distribution
    """
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    species = plot_config["species"]
    tframe = plot_config["tframe"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    tinit = pic_info.vthe**2
    beta_init = tinit * 2 / pic_info.b0**2

    fdir = '../data/temperature_anisotropy/' + pic_run + '/'
    fname = fdir + 'fbpara_' + species + '_' + str(tframe) + '.gda'
    fdata = np.fromfile(fname, dtype=np.float32)
    nbins_beta = int(fdata[0])
    beta_bins_edge = fdata[1:nbins_beta+2]
    beta_bins_mid = 0.5 * (beta_bins_edge[1:] + beta_bins_edge[:-1])
    beta_bins_mid /= beta_init
    dtemp = np.diff(beta_bins_edge)
    fbpara = fdata[nbins_beta+2:] / dtemp

    fname = fdir + 'fbperp_' + species + '_' + str(tframe) + '.gda'
    fdata = np.fromfile(fname, dtype=np.float32)
    fbperp = fdata[nbins_beta+2:] / dtemp

    fig = plt.figure(figsize=[3.25, 2.5])
    rect = [0.16, 0.16, 0.79, 0.8]
    ax = fig.add_axes(rect)
    ax.set_prop_cycle('color', COLORS)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', bottom=True, top=True)
    ax.tick_params(axis='x', which='major', direction='in', bottom=True, top=True)
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')

    ax.loglog(beta_bins_mid, fbpara, linewidth=1, label=r'$\beta_\parallel$')
    ax.loglog(beta_bins_mid, fbperp, linewidth=1, label=r'$\beta_\perp$')
    ax.legend(loc=1, prop={'size': 10}, ncol=1,
              shadow=False, fancybox=False, frameon=False)
    ax.set_xlim([1E-1, 1E5])
    ax.tick_params(labelsize=8)
    ax.set_xlabel(r'$\beta/\beta_0$', fontsize=10)
    ax.set_ylabel(r'$f(\beta)$', fontsize=10)
    fdir = '../img/beta_dist/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + 'beta_dist_' + species + '_' + str(tframe) + '.pdf'
    fig.savefig(fname)

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_tratio_dist(plot_config, show_plot=True):
    """
    Plot the distribution of the ratio between Tperp and Tpara
    """
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    species = plot_config["species"]
    tframe = plot_config["tframe"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)

    fdir = '../data/temperature_anisotropy/' + pic_run + '/'
    fname = fdir + 'ftratio_' + species + '_' + str(tframe) + '.gda'
    fdata = np.fromfile(fname, dtype=np.float32)
    nbins_tratio = int(fdata[0])
    tratio_bins_edge = fdata[1:nbins_tratio+2]
    tratio_bins_mid = 0.5 * (tratio_bins_edge[1:] + tratio_bins_edge[:-1])
    dtemp = np.diff(tratio_bins_edge)
    ftratio = fdata[nbins_tratio+2:] / dtemp

    fig = plt.figure(figsize=[3.25, 2.5])
    rect = [0.16, 0.16, 0.79, 0.8]
    ax = fig.add_axes(rect)
    ax.set_prop_cycle('color', COLORS)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', bottom=True, top=True)
    ax.tick_params(axis='x', which='major', direction='in', bottom=True, top=True)
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')

    ax.loglog(tratio_bins_mid, ftratio, linewidth=1)
    ax.set_xlim([5E-2, 2E1])
    ax.tick_params(labelsize=8)
    ax.set_xlabel(r'$T_\perp/T_\parallel$', fontsize=10)
    ax.set_ylabel(r'$f(T_\perp/T_\parallel)$', fontsize=10)
    fdir = '../img/tratio_dist/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + 'tratio_dist_' + species + '_' + str(tframe) + '.pdf'
    fig.savefig(fname)

    if show_plot:
        plt.show()
    else:
        plt.close()


@plt.FuncFormatter
def fake_log(x, pos):
    'The two args are the value and tick position'
    return r'$10^{%d}$' % (x)


def contour_tratio_beta(plot_config, show_plot=True):
    """
    Plot contour of temperature ratio and parallel plasma beta
    """
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    species = plot_config["species"]
    tframe = plot_config["tframe"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)

    fdir = '../data/temperature_anisotropy/' + pic_run + '/'
    fname = fdir + 'ftratio_bpara_' + species + '_' + str(tframe) + '.gda'
    fdata = np.fromfile(fname, dtype=np.float32)
    nbins_tratio = int(fdata[0])
    nbins_beta = int(fdata[1])
    tratio_bins_edge = fdata[2:nbins_tratio+3]
    beta_bins_edge = fdata[nbins_tratio+3:nbins_beta+nbins_tratio+4]
    tratio_bins_mid = 0.5 * (tratio_bins_edge[1:] + tratio_bins_edge[:-1])
    dtratio = np.diff(tratio_bins_edge)
    beta_bins_mid = 0.5 * (beta_bins_edge[1:] + beta_bins_edge[:-1])
    dbeta = np.diff(beta_bins_edge)
    ftratio_beta = fdata[nbins_beta+nbins_tratio+4:]
    ftratio_beta = ftratio_beta.reshape([nbins_beta, nbins_tratio]).T
    delta = np.dot(dtratio[:, None], dbeta[None, :])
    ftratio_beta /= delta
    # ftratio_beta[ftratio_beta <= 0] = np.nan
    tratio_min = tratio_bins_mid[0]
    tratio_max = tratio_bins_mid[-1]
    beta_min = beta_bins_mid[0]
    beta_max = beta_bins_mid[-1]

    # According to Bale et al. 2009,
    # T_\perp / T_\parallel = 1 + a/(\beta_\parallel - \beta_0)^b.
    # For mirror instability threshold, (a, b, \beta_0) = (0.77, 0.76, -0.016)
    # For oblique firehose instability threshold, (a, b, \beta_0) = (-1.4, 1.0, -0.11)
    tratio_firhose = 1 - 1.4 / (beta_bins_mid + 0.11)
    tratio_mirror = 1 + 0.77 / (beta_bins_mid + 0.016)**0.76
    tratio_firhose[tratio_firhose < 0] = np.nan

    fig = plt.figure(figsize=[3.25, 2.5])
    rect = [0.18, 0.16, 0.67, 0.8]
    ax = fig.add_axes(rect)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', bottom=True, top=True)
    ax.tick_params(axis='x', which='major', direction='in', bottom=True, top=True)
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')

    im = ax.pcolormesh(beta_bins_mid, tratio_bins_mid, ftratio_beta,
                       cmap=plt.cm.jet, norm=LogNorm(vmin=1E0, vmax=1E12))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.plot(beta_bins_mid, tratio_firhose, color='k', linestyle='--')
    ax.plot(beta_bins_mid, tratio_mirror, color='k', linestyle='--')
    ax.set_xlim([1E-3, 1E3])
    ax.set_ylim([1E-2, 1E2])
    ax.tick_params(labelsize=8)
    ax.set_xlabel(r'$\beta_\parallel$', fontsize=10)
    ax.set_ylabel(r'$T_\perp/T_\parallel$', fontsize=10)

    rect_cbar = np.copy(rect)
    rect_cbar[0] += rect[2] + 0.02
    rect_cbar[2] = 0.02
    cbar_ax = fig.add_axes(rect_cbar)
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=8)

    fdir = '../img/brazil_plot/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + 'tratio_bpara_' + species + '_' + str(tframe) + '.jpg'
    fig.savefig(fname, dpi=300)

    if show_plot:
        plt.show()
    else:
        plt.close()


def get_cmd_args():
    """Get command line arguments
    """
    default_run_name = 'mime25_beta002_bg00'
    default_run_dir = ('/net/scratch3/xiaocanli/reconnection/mime25/' +
                       'mime25_beta002_bg00/')
    parser = argparse.ArgumentParser(description='High-mass-ratio runs')
    parser.add_argument('--run_dir', action="store", default=default_run_dir,
                        help='run directory')
    parser.add_argument('--run_name', action="store", default=default_run_name,
                        help='run name')
    parser.add_argument('--multi_frames', action="store_true", default=False,
                        help='whether analyzing multiple frames')
    parser.add_argument('--time_loop', action="store_true", default=False,
                        help='whether analyzing multiple frames using a time loop')
    parser.add_argument('--species', action="store", default='e',
                        help='particle species')
    parser.add_argument('--tframe', action="store", default='30', type=int,
                        help='Time frame for fields')
    parser.add_argument('--tstart', action="store", default='0', type=int,
                        help='Starting time frame')
    parser.add_argument('--tend', action="store", default='40', type=int,
                        help='Ending time frame')
    parser.add_argument('--temp_dist', action="store_true", default=False,
                        help='whether to plot temperature distribution')
    parser.add_argument('--beta_dist', action="store_true", default=False,
                        help='whether to plot plasma beta distribution')
    parser.add_argument('--tratio_dist', action="store_true", default=False,
                        help='whether to plot temperature ratio distribution')
    parser.add_argument('--tratio_beta', action="store_true", default=False,
                        help=('whether to plot contour of temperature ' +
                              'ratio vs. parallel plasma beta'))
    return parser.parse_args()


def analysis_single_frame(plot_config, args):
    """Analysis for single time frame
    """
    if args.temp_dist:
        plot_temp_dist(plot_config)
    elif args.beta_dist:
        plot_beta_dist(plot_config)
    elif args.tratio_dist:
        plot_tratio_dist(plot_config)
    elif args.tratio_beta:
        contour_tratio_beta(plot_config)


def process_input(args, plot_config, tframe):
    """process one time frame"""
    plot_config["tframe"] = tframe
    print("Time frame %d" % tframe)


def analysis_multi_frames(plot_config, args):
    """Analysis for multiple time frames
    """
    tframes = range(args.tstart, args.tend + 1)
    if args.time_loop:
        for tframe in tframes:
            plot_config["tframe"] = tframe
            if args.temp_dist:
                plot_temp_dist(plot_config, show_plot=False)
            elif args.beta_dist:
                plot_beta_dist(plot_config, show_plot=False)
            elif args.tratio_dist:
                plot_tratio_dist(plot_config, show_plot=False)
            elif args.tratio_beta:
                contour_tratio_beta(plot_config, show_plot=False)
    else:
        ncores = multiprocessing.cpu_count()
        ncores = 8
        Parallel(n_jobs=ncores)(delayed(process_input)(args, plot_config, tframe)
                                for tframe in tframes)


def main():
    """business logic for when running this module as the primary one!"""
    args = get_cmd_args()
    plot_config = {}
    plot_config["pic_run"] = args.run_name
    plot_config["pic_run_dir"] = args.run_dir
    plot_config["tframe"] = args.tframe
    plot_config["tstart"] = args.tstart
    plot_config["tend"] = args.tend
    plot_config["species"] = args.species
    if args.multi_frames:
        analysis_multi_frames(plot_config, args)
    else:
        analysis_single_frame(plot_config, args)


if __name__ == "__main__":
    main()
