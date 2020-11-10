#!/usr/bin/env python3
"""
Magnetotail problem
"""
from __future__ import print_function

import argparse
import collections
import itertools
import json
import math
import multiprocessing
import os

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import palettable
from matplotlib.colors import LogNorm, SymLogNorm
from scipy import signal, integrate
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.ndimage.filters import median_filter, gaussian_filter
from scipy.special import erf

import fitting_funcs
import pic_information
from contour_plots import read_2d_fields
from joblib import Parallel, delayed
from json_functions import read_data_from_json
from pic_information import get_variable_value
from shell_functions import mkdir_p

plt.style.use("seaborn-deep")
mpl.rc('text', usetex=True)
mpl.rcParams["text.latex.preamble"] = \
        (r"\usepackage{amsmath, bm}" +
         r"\DeclareMathAlphabet{\mathsfit}{\encodingdefault}{\sfdefault}{m}{sl}" +
         r"\SetMathAlphabet{\mathsfit}{bold}{\encodingdefault}{\sfdefault}{bx}{sl}" +
         r"\newcommand{\tensorsym}[1]{\bm{\mathsfit{#1}}}")
COLORS = palettable.colorbrewer.qualitative.Set1_9.mpl_colors


def div0(a, b):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0]

    From: http://stackoverflow.com/a/35696047/2561161

    """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[~np.isfinite(c)] = 0  # -inf inf NaN
    return c


def find_nearest(array, value):
    """Find nearest value in an array
    """
    idx = (np.abs(array-value)).argmin()
    return (idx, array[idx])


def get_vpic_info(pic_run_dir):
    """Get information of the VPIC simulation
    """
    with open(pic_run_dir + '/info') as f:
        content = f.readlines()
    f.close()
    vpic_info = {}
    for line in content[1:]:
        if "=" in line:
            line_splits = line.split("=")
        elif ":" in line:
            line_splits = line.split(":")

        tail = line_splits[1].split("\n")
        vpic_info[line_splits[0].strip()] = float(tail[0])
    return vpic_info


def plot_jy(plot_config, show_plot=True):
    """Plot current density
    """
    tframe = plot_config["tframe"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    vpic_info = get_vpic_info(pic_run_dir)
    smime = math.sqrt(vpic_info["mi/me"])
    nx = int(vpic_info["nx"])
    nz = int(vpic_info["nz"])
    lx_di = vpic_info["Lx/di"]
    lz_di = vpic_info["Lz/di"]
    lx_de = lx_di * smime
    lz_de = lz_di * smime
    xmin, xmax = 0, lx_de
    zmin, zmax = -0.5 * lz_de, 0.5 * lz_de
    xmin_di, xmax_di = 0, lx_di
    zmin_di, zmax_di = -0.5*lz_di, 0.5*lz_di

    fname = pic_run_dir + "data/jy.gda"
    doffset = nz * nx * 4 * tframe
    jy = np.fromfile(fname, dtype='float32', offset=doffset,
                     count=nx*nz).reshape([nz, nx])

    len0 = 10
    fig = plt.figure(figsize=[len0, len0*lz_de/lx_de])
    rect = [0.11, 0.14, 0.78, 0.78]
    ax = fig.add_axes(rect)
    im1 = ax.imshow(jy,
                    extent=[xmin_di, xmax_di, zmin_di, zmax_di],
                    vmin=-2E-2, vmax=2E-2,
                    cmap=plt.cm.coolwarm, aspect='auto',
                    origin='lower', interpolation='bicubic')
    # Magnetic field lines
    fname = pic_run_dir + "data/bx.gda"
    bx = np.fromfile(fname, dtype='float32', offset=doffset,
                     count=nx*nz).reshape([nz, nx])
    fname = pic_run_dir + "data/bz.gda"
    bz = np.fromfile(fname, dtype='float32', offset=doffset,
                     count=nx*nz).reshape([nz, nx])
    xgrid = np.linspace(xmin_di, xmax_di, nx)
    zgrid = np.linspace(zmin_di, zmax_di, nz)
    xmesh, zmesh = np.meshgrid(xgrid, zgrid)
    ax.streamplot(xmesh, zmesh, bx, bz, color='k', linewidth=1)

    ax.set_xlim([xmin_di, xmax_di])
    ax.set_ylim([zmin_di, zmax_di])
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlabel(r'$x/d_i$', fontsize=20)
    ax.set_ylabel(r'$z/d_i$', fontsize=20)
    ax.tick_params(labelsize=16)
    rect_cbar = np.copy(rect)
    rect_cbar[0] += rect[2] + 0.015
    rect_cbar[2] = 0.02
    rect_cbar[1] += rect[3] * 0.25
    rect_cbar[3] = rect[3] * 0.5
    cbar_ax = fig.add_axes(rect_cbar)
    cbar = fig.colorbar(im1, cax=cbar_ax, extend='both')
    cbar_ax.tick_params(bottom=False, top=False, left=False, right=True)
    cbar_ax.tick_params(axis='y', which='major', direction='out')
    cbar_ax.tick_params(axis='y', which='minor', direction='in', right=False)
    cbar_ax.set_title(r'$j_y$', fontsize=20)
    cbar.ax.tick_params(labelsize=12)
    img_dir = '../img/tail_problem/jy/' + pic_run + '/'
    mkdir_p(img_dir)
    fname = img_dir + "jy_" + str(tframe) + ".jpg"
    fig.savefig(fname, dpi=200)

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_bfield(plot_config, show_plot=True):
    """Plot magnetic field
    """
    tframe = plot_config["tframe"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    vpic_info = get_vpic_info(pic_run_dir)
    smime = math.sqrt(vpic_info["mi/me"])
    nx = int(vpic_info["nx"])
    nz = int(vpic_info["nz"])
    lx_di = vpic_info["Lx/di"]
    lz_di = vpic_info["Lz/di"]
    lx_de = lx_di * smime
    lz_de = lz_di * smime
    xmin, xmax = 0, lx_de
    zmin, zmax = -0.5 * lz_de, 0.5 * lz_de
    xmin_di, xmax_di = 0, lx_di
    zmin_di, zmax_di = -0.5*lz_di, 0.5*lz_di

    doffset = nz * nx * 4 * tframe
    fname = pic_run_dir + "data/bx.gda"
    bvec = {}
    bvec["bx"] = np.fromfile(fname, dtype='float32', offset=doffset,
                             count=nx*nz).reshape([nz, nx])
    fname = pic_run_dir + "data/by.gda"
    bvec["by"] = np.fromfile(fname, dtype='float32', offset=doffset,
                             count=nx*nz).reshape([nz, nx])
    fname = pic_run_dir + "data/bz.gda"
    bvec["bz"] = np.fromfile(fname, dtype='float32', offset=doffset,
                             count=nx*nz).reshape([nz, nx])
    xgrid = np.linspace(xmin_di, xmax_di, nx)
    zgrid = np.linspace(zmin_di, zmax_di, nz)
    xmesh, zmesh = np.meshgrid(xgrid, zgrid)

    bvec["absb"] = np.sqrt(bvec["bx"]**2 + bvec["by"]**2 + bvec["bz"]**2)
    fig = plt.figure(figsize=[5, 8])
    rect = [0.14, 0.75, 0.72, 0.2]
    hgap, vgap = 0.02, 0.02
    var_names = [r"$B_x$", r"$B_y$", r"$B_z$", r"$|\boldsymbol{B}|$"]
    colors = ['k', 'k', 'k', 'k']
    for ivar, var in enumerate(bvec):
        ax = fig.add_axes(rect)
        cmap = plt.cm.coolwarm if ivar < 3 else plt.cm.plasma
        dmax = 0.5
        dmin = -dmax if ivar < 3 else 0
        im1 = ax.imshow(bvec[var],
                        extent=[xmin_di, xmax_di, zmin_di, zmax_di],
                        vmin=dmin, vmax=dmax,
                        cmap=cmap, aspect='auto',
                        origin='lower', interpolation='none')
        ax.streamplot(xmesh, zmesh, bvec["bx"], bvec["bz"], color='k', linewidth=1)
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')
        ax.set_xlim([xmin_di, xmax_di])
        ax.set_ylim([zmin_di, zmax_di])
        if ivar == 3:
            ax.set_xlabel(r'$x/d_i$', fontsize=16)
        else:
            ax.tick_params(axis='x', labelbottom=False)
        ax.set_ylabel(r'$z/d_i$', fontsize=16)
        ax.tick_params(labelsize=12)
        ax.text(0.02, 0.85, var_names[ivar], color=colors[ivar], fontsize=16,
                bbox=dict(facecolor='w', alpha=0.7, edgecolor='k', pad=3.0),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)
        rect_cbar = np.copy(rect)
        rect_cbar[0] += rect[2] + 0.01
        rect_cbar[2] = 0.02
        cbar_ax = fig.add_axes(rect_cbar)
        cbar = fig.colorbar(im1, cax=cbar_ax, extend='both')
        cbar_ax.tick_params(bottom=False, top=False, left=False, right=True)
        cbar_ax.tick_params(axis='y', which='major', direction='out')
        cbar_ax.tick_params(axis='y', which='minor', direction='in', right=False)
        cbar.ax.tick_params(labelsize=12)
        rect[1] -= rect[3] + vgap
    img_dir = '../img/tail_problem/bfield/' + pic_run + '/'
    mkdir_p(img_dir)
    fname = img_dir + "bfield_" + str(tframe) + ".jpg"
    fig.savefig(fname, dpi=200)

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_efield(plot_config, show_plot=True):
    """Plot electric field
    """
    tframe = plot_config["tframe"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    vpic_info = get_vpic_info(pic_run_dir)
    smime = math.sqrt(vpic_info["mi/me"])
    nx = int(vpic_info["nx"])
    nz = int(vpic_info["nz"])
    lx_di = vpic_info["Lx/di"]
    lz_di = vpic_info["Lz/di"]
    lx_de = lx_di * smime
    lz_de = lz_di * smime
    xmin, xmax = 0, lx_de
    zmin, zmax = -0.5 * lz_de, 0.5 * lz_de
    xmin_di, xmax_di = 0, lx_di
    zmin_di, zmax_di = -0.5*lz_di, 0.5*lz_di

    doffset = nz * nx * 4 * tframe
    evec = {}
    fname = pic_run_dir + "data/ex.gda"
    evec["ex"] = np.fromfile(fname, dtype='float32', offset=doffset,
                             count=nx*nz).reshape([nz, nx])
    fname = pic_run_dir + "data/ey.gda"
    evec["ey"] = np.fromfile(fname, dtype='float32', offset=doffset,
                             count=nx*nz).reshape([nz, nx])
    fname = pic_run_dir + "data/ez.gda"
    evec["ez"] = np.fromfile(fname, dtype='float32', offset=doffset,
                             count=nx*nz).reshape([nz, nx])
    sigma = 5
    evec["ex"] = gaussian_filter(evec["ex"], sigma=sigma)
    evec["ey"] = gaussian_filter(evec["ey"], sigma=sigma)
    evec["ez"] = gaussian_filter(evec["ez"], sigma=sigma)
    fname = pic_run_dir + "data/bx.gda"
    bx = np.fromfile(fname, dtype='float32', offset=doffset,
                     count=nx*nz).reshape([nz, nx])
    fname = pic_run_dir + "data/bz.gda"
    bz = np.fromfile(fname, dtype='float32', offset=doffset,
                     count=nx*nz).reshape([nz, nx])
    xgrid = np.linspace(xmin_di, xmax_di, nx)
    zgrid = np.linspace(zmin_di, zmax_di, nz)
    xmesh, zmesh = np.meshgrid(xgrid, zgrid)

    evec["absb"] = np.sqrt(evec["ex"]**2 + evec["ey"]**2 + evec["ez"]**2)
    fig = plt.figure(figsize=[5, 8])
    rect = [0.14, 0.75, 0.70, 0.2]
    hgap, vgap = 0.02, 0.02
    var_names = [r"$E_x$", r"$E_y$", r"$E_z$", r"$|\boldsymbol{E}|$"]
    colors = ['k', 'k', 'k', 'k']
    for ivar, var in enumerate(evec):
        ax = fig.add_axes(rect)
        cmap = plt.cm.coolwarm if ivar < 3 else plt.cm.plasma
        dmax = 0.01
        dmin = -dmax if ivar < 3 else 0
        im1 = ax.imshow(evec[var],
                        extent=[xmin_di, xmax_di, zmin_di, zmax_di],
                        vmin=dmin, vmax=dmax,
                        cmap=cmap, aspect='auto',
                        origin='lower', interpolation='none')
        ax.streamplot(xmesh, zmesh, bx, bz, color='k', linewidth=1)
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')
        ax.set_xlim([xmin_di, xmax_di])
        ax.set_ylim([zmin_di, zmax_di])
        if ivar == 3:
            ax.set_xlabel(r'$x/d_i$', fontsize=16)
        else:
            ax.tick_params(axis='x', labelbottom=False)
        ax.set_ylabel(r'$z/d_i$', fontsize=16)
        ax.tick_params(labelsize=12)
        ax.text(0.02, 0.85, var_names[ivar], color=colors[ivar], fontsize=16,
                bbox=dict(facecolor='w', alpha=0.7, edgecolor='k', pad=3.0),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)
        rect_cbar = np.copy(rect)
        rect_cbar[0] += rect[2] + 0.01
        rect_cbar[2] = 0.02
        cbar_ax = fig.add_axes(rect_cbar)
        cbar = fig.colorbar(im1, cax=cbar_ax, extend='both')
        cbar_ax.tick_params(bottom=False, top=False, left=False, right=True)
        cbar_ax.tick_params(axis='y', which='major', direction='out')
        cbar_ax.tick_params(axis='y', which='minor', direction='in', right=False)
        cbar.ax.tick_params(labelsize=12)
        rect[1] -= rect[3] + vgap
    img_dir = '../img/tail_problem/efield/' + pic_run + '/'
    mkdir_p(img_dir)
    fname = img_dir + "efield_" + str(tframe) + ".jpg"
    fig.savefig(fname, dpi=200)

    if show_plot:
        plt.show()
    else:
        plt.close()


def get_cmd_args():
    """Get command line arguments
    """
    default_pic_run = 'tail_close'
    default_pic_run_dir = ('/net/scratch4/xiaocan/for_yihsin/' + default_pic_run + '/')
    parser = argparse.ArgumentParser(description='Analysis for trans-relativistic runs')
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
    parser.add_argument('--show_plot', action="store_true", default=False,
                        help='whether to show plot')
    parser.add_argument('--plot_jy', action="store_true", default=False,
                        help='whether to plot current density along y')
    parser.add_argument('--plot_bfield', action="store_true", default=False,
                        help='whether to plot magnetic field')
    parser.add_argument('--plot_efield', action="store_true", default=False,
                        help='whether to plot electric field')
    return parser.parse_args()


def analysis_single_frames(plot_config, args):
    """Analysis for multiple time frames
    """
    tframe = args.tframe
    if args.plot_jy:
        plot_jy(plot_config, args.show_plot)
    elif args.plot_bfield:
        plot_bfield(plot_config, args.show_plot)
    elif args.plot_efield:
        plot_efield(plot_config, args.show_plot)


def process_input(plot_config, args, tframe):
    """process one time frame"""
    plot_config["tframe"] = tframe


def analysis_multi_frames(plot_config, args):
    """Analysis for multiple time frames
    """
    tframes = range(plot_config["tstart"], plot_config["tend"] + 1)
    nframes = len(tframes)
    if args.time_loop:
        for tframe in tframes:
            print("Time frame: %d" % tframe)
            plot_config["tframe"] = tframe
            if args.plot_jy:
                plot_jy(plot_config, show_plot=False)
            elif args.plot_bfield:
                plot_bfield(plot_config, show_plot=False)
            elif args.plot_efield:
                plot_efield(plot_config, show_plot=False)
    else:
        ncores = multiprocessing.cpu_count()
        ncores = 4
        if ncores > nframes:
            ncores = nframes
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
    if args.multi_frames:
        analysis_multi_frames(plot_config, args)
    else:
        analysis_single_frames(plot_config, args)


if __name__ == "__main__":
    main()
