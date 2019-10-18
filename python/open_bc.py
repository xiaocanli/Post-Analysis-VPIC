#!/usr/bin/env python3
"""
Analysis procedures for open-boundary runs
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


def div0(a, b):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0]

    From: http://stackoverflow.com/a/35696047/2561161

    """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[~np.isfinite(c)] = 0  # -inf inf NaN
    return c


def rho_bands_2d(plot_config, show_plot=True):
    """Plot densities for energetic particles in the 2D simulation
    """
    pic_run = plot_config["pic_run"]
    species = plot_config["species"]
    tframe = plot_config["tframe"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_run_dir = pic_info.run_dir
    if species == 'e':
        vth = pic_info.vthe
    else:
        vth = pic_info.vthi
    nbins = 1000
    ndata = nbins + 3  # including magnetic field
    tindex = pic_info.particle_interval * tframe
    gama = 1.0 / math.sqrt(1.0 - 3 * vth**2)
    eth = gama - 1.0
    ebins = np.logspace(-6, 4, nbins)
    ebins /= eth
    xmin, xmax = 0, pic_info.lx_di
    zmin, zmax = -pic_info.lz_di * 0.5, pic_info.lz_di * 0.5
    nx, nz = pic_info.nx, pic_info.nz
    smime = math.sqrt(pic_info.mime)
    dx_de = pic_info.dx_di * smime
    dy_de = pic_info.dy_di * smime
    dz_de = pic_info.dz_di * smime
    kwargs = {"current_time": tframe,
              "xl": 0, "xr": pic_info.lx_di,
              "zb": -pic_info.lz_di, "zt": pic_info.lz_di}
    # fname = pic_run_dir + "data/Ay.gda"
    # x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/bx.gda"
    x, z, bx = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/by.gda"
    x, z, by = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/bz.gda"
    x, z, bz = read_2d_fields(pic_info, fname, **kwargs)
    ib = 1.0/np.sqrt(bx**2 + by**2 + bz**2)
    bx = bx * ib
    by = by * ib
    bz = bz * ib
    kappax = (bx * np.gradient(bx, axis=1) / dx_de +
              bz * np.gradient(bx, axis=0) / dz_de)
    kappay = (bx * np.gradient(by, axis=1) / dx_de +
              bz * np.gradient(by, axis=0) / dz_de)
    kappaz = (bx * np.gradient(bz, axis=1) / dx_de +
              bz * np.gradient(bz, axis=0) / dz_de)

    fname = pic_run_dir + "data/vex.gda"
    x, z, vex = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/vey.gda"
    x, z, vey = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/vez.gda"
    x, z, vez = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/vix.gda"
    x, z, vix = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/viy.gda"
    x, z, viy = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/viz.gda"
    x, z, viz = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/ne.gda"
    x, z, ne = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/ni.gda"
    x, z, ni = read_2d_fields(pic_info, fname, **kwargs)
    inrho = 1.0 / (ne + ni*pic_info.mime)
    vx = (ne*vex + ni*vix*pic_info.mime) * inrho
    vy = (ne*vey + ni*viy*pic_info.mime) * inrho
    vz = (ne*vez + ni*viz*pic_info.mime) * inrho
    vdot_kappa = vx * kappax + vy * kappay + vz * kappaz

    nbands = 7
    nreduce = 16
    nxr = pic_info.nx // nreduce
    nzr = pic_info.nz // nreduce
    ntot = np.zeros((nzr, nxr))
    nhigh = np.zeros((nzr, nxr))
    fig = plt.figure(figsize=[14, 7])
    rect0 = [0.055, 0.75, 0.4, 0.20]
    hgap, vgap = 0.09, 0.025
    if species == 'e':
        nmins = [1E-1, 1E-2, 5E-3, 1E-3, 6E-5, 2E-5, 6E-6]
        nmaxs = [5E0, 1E0, 5E-1, 1E-1, 6E-3, 2E-3, 6E-4]
    else:
        nmins = [1E-1, 2E-2, 1E-2, 5E-3, 5E-4, 5E-5, 1.2E-5]
        nmaxs = [5E0, 2E0, 1E0, 5E-1, 5E-2, 5E-3, 1.2E-3]
    nmins = np.asarray(nmins) / 10
    nmaxs = np.asarray(nmaxs) / 10
    axs = []
    rects = []
    nrows = (nbands + 1) // 2
    for iband in range(nbands+1):
        row = iband % nrows
        col = iband // nrows
        if row == 0:
            rect = np.copy(rect0)
            rect[0] += (rect[2] + hgap) * col
        ax = fig.add_axes(rect)
        ax.set_ylim([-20, 20])
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')
        if row < nrows - 1:
            ax.tick_params(axis='x', labelbottom=False)
        else:
            ax.set_xlabel(r'$x/d_i$', fontsize=16)
        if col == 0:
            ax.set_ylabel(r'$z/d_i$', fontsize=16)
        ax.tick_params(labelsize=12)
        axs.append(ax)
        rects.append(np.copy(rect))
        rect[1] -= rect[3] + vgap

    band_break = 4
    for iband in range(nbands):
        print("Energy band: %d" % iband)
        if iband < band_break:
            ax = axs[iband]
            rect = rects[iband]
        else:
            ax = axs[iband+1]
            rect = rects[iband+1]
        fname = (pic_run_dir + "data-smooth2/n" + species + "_" +
                 str(iband) + "_" + str(tindex) + ".gda")
        nrho = np.fromfile(fname, dtype=np.float32)
        nrho = nrho.reshape((nzr, nxr))
        if iband >= 5:
            nhigh += nrho
        ntot += nrho
        nmin, nmax = nmins[iband], nmaxs[iband]
        p1 = ax.imshow(nrho + 1E-10,
                       extent=[xmin, xmax, zmin, zmax],
                       norm = LogNorm(vmin=nmin, vmax=nmax),
                       cmap=plt.cm.inferno, aspect='auto',
                       origin='lower', interpolation='bicubic')
        # ax.contour(x, z, Ay, colors='w', linewidths=0.5)
        if iband == 0:
            label1 = r'$n(\varepsilon < 10\varepsilon_\text{th})$'
        elif iband > 0 and iband < nbands-1:
            label1 = (r'$n(' + str(2**(iband-1)*10) + r'\varepsilon_\text{th} < ' +
                      r'\varepsilon < ' + str(2**iband*10) + r'\varepsilon_\text{th})$')
        else:
            label1 = (r'$n(\varepsilon > ' + str(2**(nbands-2)*10) +
                      r'\varepsilon_\text{th})$')
        ax.text(0.98, 0.87, label1, color='k', fontsize=16,
                bbox=dict(facecolor='w', alpha=0.75,
                          edgecolor='none', boxstyle="round,pad=0.1"),
                horizontalalignment='right',
                verticalalignment='center',
                transform=ax.transAxes)
        twci = math.ceil((tframe * pic_info.dt_fields) / 0.1) * 0.1
        text1 = r'$t\Omega_{ci}=' + ("{%0.0f}" % twci) + '$'
        if iband == 0:
            # ax.set_title(text1, fontsize=16)
            xpos = (rect[2] + hgap * 0.5) / rect[2]
            ax.text(xpos, 1.1, text1, color='k', fontsize=16,
                    bbox=dict(facecolor='w', alpha=0.75,
                              edgecolor='none', boxstyle="round,pad=0.1"),
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes)

        rect_cbar = np.copy(rect)
        rect_cbar[0] += rect[2] + 0.01
        rect_cbar[2] = 0.007
        cbar_ax = fig.add_axes(rect_cbar)
        cbar = fig.colorbar(p1, cax=cbar_ax, extend='both')
        cbar.ax.tick_params(labelsize=12)

    ax = axs[band_break]
    rect = rects[band_break]
    vmin, vmax = -1.0, 1.0
    knorm = 100
    p1 = ax.imshow(vdot_kappa*knorm,
                   extent=[xmin, xmax, zmin, zmax],
                   vmin=vmin, vmax=vmax,
                   cmap=plt.cm.seismic, aspect='auto',
                   origin='lower', interpolation='bicubic')
    # ax.contour(x, z, Ay, colors='k', linewidths=0.5)
    ax.tick_params(labelsize=12)
    label1 = r'$' + str(knorm) + r'\boldsymbol{v}\cdot\boldsymbol{\kappa}$'
    ax.text(0.98, 0.87, label1, color='w', fontsize=16,
            bbox=dict(facecolor='k', alpha=0.5,
                      edgecolor='none', boxstyle="round,pad=0.1"),
            horizontalalignment='right',
            verticalalignment='center',
            transform=ax.transAxes)
    rect_cbar = np.copy(rect)
    rect_cbar[0] += rect[2] + 0.01
    rect_cbar[2] = 0.007
    cbar_ax = fig.add_axes(rect_cbar)
    cbar = fig.colorbar(p1, cax=cbar_ax, extend='both')
    cbar.set_ticks(np.linspace(-1, 1, num=5))
    cbar.ax.tick_params(labelsize=12)

    fdir = '../img/open_bc/rho_bands_2d/' + pic_run + '/'
    mkdir_p(fdir)
    fname = (fdir + 'nrho_bands_' + species + '_' + str(tframe) + ".jpg")
    fig.savefig(fname, dpi=200)

    if show_plot:
        plt.show()
    else:
        plt.close()


def get_cmd_args():
    """Get command line arguments
    """
    default_pic_run = 'local_perturb_master'
    default_pic_run_dir = ('/net/scratch4/xiaocanli/reconnection/psp_reconnection/' +
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
    parser.add_argument('--show_plot', action="store_true", default=False,
                        help='whether to show plot')
    parser.add_argument('--rho_bands_2d', action="store_true", default=False,
                        help=("whether to plot densities if different " +
                              "energy bands for the 2D simulation"))
    return parser.parse_args()


def analysis_single_frames(plot_config, args):
    """Analysis for multiple time frames
    """
    tframe = args.tframe
    if args.rho_bands_2d:
        rho_bands_2d(plot_config)


def process_input(plot_config, args, tframe):
    """process one time frame"""
    plot_config["tframe"] = tframe
    if args.rho_bands_2d:
        rho_bands_2d(plot_config, show_plot=False)


def analysis_multi_frames(plot_config, args):
    """Analysis for multiple time frames
    """
    tframes = range(plot_config["tstart"], plot_config["tend"] + 1)
    if args.time_loop:
        for tframe in tframes:
            print("Time frame: %d" % tframe)
            plot_config["tframe"] = tframe
            if args.rho_bands_2d:
                rho_bands_2d(plot_config, show_plot=False)
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
    if args.multi_frames:
        analysis_multi_frames(plot_config, args)
    else:
        analysis_single_frames(plot_config, args)


if __name__ == "__main__":
    main()
