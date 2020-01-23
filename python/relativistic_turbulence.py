#!/usr/bin/env python3
"""
Analysis procedures for relativistic turbulence runs
"""
from __future__ import print_function

import argparse
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
from scipy.optimize import curve_fit

import fitting_funcs
import pic_information
from contour_plots import read_2d_fields
from joblib import Parallel, delayed
from json_functions import read_data_from_json
from pic_information import get_variable_value
from shell_functions import mkdir_p

plt.style.use("seaborn-deep")
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = \
[r"\usepackage{amsmath, bm}",
 r"\DeclareMathAlphabet{\mathsfit}{\encodingdefault}{\sfdefault}{m}{sl}",
 r"\SetMathAlphabet{\mathsfit}{bold}{\encodingdefault}{\sfdefault}{bx}{sl}",
 r"\newcommand{\tensorsym}[1]{\bm{\mathsfit{#1}}}"]
COLORS = palettable.colorbrewer.qualitative.Set1_9.mpl_colors


def div0(a, b):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0]

    From: http://stackoverflow.com/a/35696047/2561161

    """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[~np.isfinite(c)] = 0  # -inf inf NaN
    return c


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


def energy_conversion(plot_config):
    """Plot energy conversion

    Args:
        plot_config: plotting configuration
    """
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    vpic_info = get_vpic_info(pic_run_dir)
    tenergy = pic_info.tenergy
    dt_energy = vpic_info["energies_interval"] * vpic_info["dt*wpe"]
    tenergy = np.arange(len(tenergy)) * dt_energy
    enorm = pic_info.ene_magnetic[0]

    ene_bx = pic_info.ene_bx
    ene_by = pic_info.ene_by
    ene_bz = pic_info.ene_bz
    ene_magnetic = pic_info.ene_magnetic
    ene_electric = pic_info.ene_electric
    kene_e = pic_info.kene_e
    kene_i = pic_info.kene_i

    etot = ene_magnetic + ene_electric + kene_e + kene_i
    print("Energy conservation: %e" % ((etot[-1] - etot[0]) / etot[0]))
    print("Energy conversion: %e" %
          ((ene_magnetic[-1] - ene_magnetic[0]) / ene_magnetic[0]))
    print("deltaB/B0: %e" % math.sqrt((ene_bx[0] + ene_bz[0]) / ene_by[0]))

    ene_bx /= enorm
    ene_by /= enorm
    ene_bz /= enorm
    ene_magnetic /= enorm
    ene_electric /= enorm
    kene_e /= enorm
    kene_i /= enorm

    dene_magnetic = np.gradient(ene_magnetic) / dt_energy
    dene_electric = np.gradient(ene_electric) / dt_energy
    dkene_e = np.gradient(kene_e) / dt_energy
    dkene_i = np.gradient(kene_i) / dt_energy

    fig = plt.figure(figsize=[5, 3.5])
    rect = [0.15, 0.16, 0.8, 0.8]
    ax = fig.add_axes(rect)
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    ax.set_prop_cycle('color', COLORS)

    ax.plot(tenergy, ene_magnetic, linewidth=1, linestyle='-', label='magnetic')
    ax.plot(tenergy, ene_electric, linewidth=1, linestyle='-', label='electric')
    ax.plot(tenergy, kene_e, linewidth=1, linestyle='-', label='electron')
    ax.plot(tenergy, kene_i, linewidth=1, linestyle='-', label='ion')

    ax.legend(loc=6, prop={'size': 12}, ncol=1,
              shadow=False, fancybox=False, frameon=False)
    ax.tick_params(bottom=True, top=False, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlim([0, tenergy[-1]])
    ax.set_ylim([0, 1.00])
    ax.set_xlabel(r'$t\omega_{pe}$', fontsize=16)
    ax.set_ylabel(r'$\text{Energy}/\varepsilon_{B0}$', fontsize=16)
    ax.tick_params(labelsize=12)

    fdir = '../img/relativistic_turbulence/'
    mkdir_p(fdir)
    fname = fdir + 'econv_' + pic_run + '.pdf'
    fig.savefig(fname)

    fig = plt.figure(figsize=[5, 3.5])
    rect = [0.15, 0.16, 0.8, 0.8]
    ax = fig.add_axes(rect)
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    ax.set_prop_cycle('color', COLORS)

    ax.plot(tenergy, -dene_magnetic, linewidth=1, linestyle='-', label='-magnetic')
    ax.plot(tenergy, dene_electric, linewidth=1, linestyle='-', label='electric')
    ax.plot(tenergy, dkene_e, linewidth=1, linestyle='-', label='electron')
    ax.plot(tenergy, dkene_i, linewidth=1, linestyle='-', label='ion')

    ax.legend(loc=1, prop={'size': 12}, ncol=1,
              shadow=False, fancybox=False, frameon=False)
    ax.tick_params(bottom=True, top=False, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlim([0, tenergy[-1]])
    ax.set_ylim([-5E-4, 2E-3])
    ax.set_xlabel(r'$t\omega_{pe}$', fontsize=16)
    ax.set_ylabel(r'$\text{Energy}/\varepsilon_{B0}$', fontsize=16)
    ax.tick_params(labelsize=12)

    fdir = '../img/relativistic_turbulence/'
    mkdir_p(fdir)
    fname = fdir + 'deconv_' + pic_run + '.pdf'
    fig.savefig(fname)

    plt.show()


def find_nearest(array, value):
    """Find nearest value in an array
    """
    idx = (np.abs(array-value)).argmin()
    return (idx, array[idx])


def plot_spectrum_multi(plot_config, show_plot=True):
    """Plot particle energy spectrum for multiple time frames

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    ntf = len(os.listdir(pic_run_dir + "/spectrum"))
    emin, emax = 1E-4, 1E6
    nbins = 1000
    dloge = (math.log10(emax) - math.log10(emin)) / (nbins - 1)
    emin0 = 10**(math.log10(emin) - dloge)
    ebins = np.logspace(math.log10(emin0), math.log10(emax), nbins+1)
    ebins_mid = (ebins[:-1] + ebins[1:]) * 0.5
    debins = np.diff(ebins)
    nptot = pic_info.nx * pic_info.ny * pic_info.nz * pic_info.nppc
    if species in ["e", "electron"]:
        sname = "electron"
    else:
        sname = "Ion"

    tframes = range(ntf)
    nframes = len(tframes)
    flogs = np.zeros((nframes, nbins))

    fig = plt.figure(figsize=[7, 5])
    rect = [0.14, 0.16, 0.82, 0.8]
    ax = fig.add_axes(rect)
    vpic_info = get_vpic_info(pic_run_dir)
    spectrum_interval = int(vpic_info["spectrum_interval"])
    for iframe, tframe in enumerate(tframes):
        print("Time frame: %d" % tframe)
        tindex = tframe * spectrum_interval
        fdir = pic_run_dir + '/spectrum_combined/'
        fname = fdir + 'spectrum_' + sname + '_' + str(tindex) + '.dat'
        flog = np.fromfile(fname, dtype=np.float32)
        espect = flog[3:] / debins / nptot  # the first 3 are magnetic field
        color = plt.cm.jet(tframe/float(ntf), 1)
        flogs[iframe, :] = espect
        ax.loglog(ebins_mid, espect, linewidth=1, color=color)

    pindex = -2.7
    es, _ = find_nearest(ebins_mid, 40)
    ee, _ = find_nearest(ebins_mid, 400)
    fpower = ebins_mid**pindex
    norm = espect[es] * 3 / fpower[es]
    fpower *= norm
    power_index = "{%0.1f}" % pindex
    ax.loglog(ebins_mid[es:ee], fpower[es:ee], linewidth=1, color='k')
    pname = r'$\propto (\gamma-1)^{' + power_index + '}$'
    ax.text(0.78, 0.65, pname, color='k', fontsize=16, rotation=0,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)

    if species == 'e':
        ax.set_xlim([1E-2, 1E3])
        ax.set_ylim([1E-8, 1E1])
    else:
        ax.set_xlim([1E-5, 5E0])
        ax.set_ylim([1E-6, 5E3])
    ax.tick_params(bottom=True, top=True, left=True, right=False)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlabel(r'$\gamma - 1$', fontsize=20)
    ax.set_ylabel(r'$f(\gamma - 1)$', fontsize=20)
    ax.tick_params(labelsize=16)
    ename = 'electron' if species == 'e' else 'ion'
    fpath = "../img/relativistic_turbulence/spectra/" + ename + "/"
    mkdir_p(fpath)
    fname = fpath + "spect_time_" + pic_run + "_" + species + ".pdf"
    fig.savefig(fname)
    if show_plot:
        plt.show()
    else:
        plt.close()


def check_density(plot_config, show_plot=True):
    """Whether to check maximum density

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    tframe = plot_config["tframe"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    kwargs = {"current_time": tframe,
              "xl": 0, "xr": pic_info.lx_di,
              "zb": -pic_info.lz_di*0.5, "zt": pic_info.lz_di*0.5}
    fname = pic_run_dir + "data/ne.gda"
    tframes = range(0, pic_info.ntf, 10)
    nframe = len(tframes)
    nmaxs = np.zeros(nframe)
    for iframe, tframe in enumerate(tframes):
        kwargs["current_time"] = tframe
        x, z, ne = read_2d_fields(pic_info, fname, **kwargs)
        nz, nx = ne.shape
        ne_z1 = np.mean(ne[:nz//2, :], axis=0)
        ne_z2 = np.mean(ne[nz//2:, :], axis=0)
        nmaxs[iframe] = max([np.max(ne_z1), np.max(ne_z2)])
    fig = plt.figure(figsize=[7, 5])
    rect = [0.12, 0.16, 0.70, 0.8]
    ax = fig.add_axes(rect)
    ax.plot(nmaxs, linewidth=2)
    plt.show()


def read_var(group, dset_name, sz):
    """Read data from a HDF5 group

    Args:
        group: one HDF5 group
        var: the dataset name
        sz: the size of the data
    """
    dset = group[dset_name]
    fdata = np.zeros(sz, dtype=dset.dtype)
    dset.read_direct(fdata)
    return fdata


def plot_four_panels(datah, datav, fname, ixyz, twpe10, sigma_e,
                     var="wpara_wperp", show_plot=True, wpara_wperp=False):
    """plot 4 panels of histograms of energization terms

    Args:
        datah: data for horizontal axes
        datav: data for vertical axes
        fname: filename to save the figure
        ixyz: to indicate which component to plot
        twpe10: 10*t*wpe
        sigma_e: electron magnetization factor
        var(optional): variable name
        show_plot(optional): whether to show plot
        wpara_wperp(optional): whether to plot wpara and wperp
    """
    fig = plt.figure(figsize=[5.5, 5])
    rect0 = [0.12, 0.54, 0.37, 0.39]
    hgap, vgap = 0.035, 0.035
    rect = np.copy(rect0)
    nbins = 128
    dene_min, dene_max = -1, 3
    drange = [[dene_min, dene_max], [dene_min, dene_max]]
    vmin, vmax = 1E0, 1E3
    for row, col in itertools.product(range(2), range(2)):
        if row == 0 and col == 0:
            cond = np.logical_and(datah < 0, datav > 0)
        elif row == 0 and col == 1:
            cond = np.logical_and(datah > 0, datav > 0)
        elif row == 1 and col == 0:
            cond = np.logical_and(datah < 0, datav < 0)
        elif row == 1 and col == 1:
            cond = np.logical_and(datah > 0, datav < 0)

        hist, _, _ = np.histogram2d(np.log10(np.abs(datav[cond])),
                                    np.log10(np.abs(datah[cond])),
                                    bins=nbins, range=drange)
        rect[0] = rect0[0] + col * (rect0[2] + hgap)
        rect[1] = rect0[1] - row * (rect0[3] + vgap)
        ax = fig.add_axes(rect)
        p1 = ax.imshow(hist,
                       extent=[dene_min, dene_max, dene_min, dene_max],
                       norm = LogNorm(vmin=vmin, vmax=vmax),
                       cmap=plt.cm.viridis, aspect='auto',
                       origin='lower', interpolation='nearest')
        xsign = '> 0' if col else '< 0'
        ysign = '< 0' if row else '> 0'
        labels = set_labels(ixyz, xsign, ysign, var)
        text_diag = labels["xtext"] + r'$=$' + labels["ytext"]
        ax.plot([dene_min, dene_max], [dene_min, dene_max],
                color='k', linewidth=1, linestyle='--', label=text_diag)
        if wpara_wperp:
            wpara_bins = np.linspace(dene_min, dene_max, 10000)
            wperp = sigma_e*0.5 - 10**wpara_bins * (col - 0.5) * 2
            wperp *= -(row - 0.5) * 2
            cond = wperp > 0
            wperp_bins = np.log10(wperp[cond])
            text1 = r"$W = \sigma_e/2$"
            ax.plot(wpara_bins[cond], wperp_bins, color='k', linewidth=1,
                    linestyle='-', label=text1)
            pos = (row, col)
            if pos == (1, 0):
                ax.legend(loc=4, prop={'size': 10}, ncol=1,
                          shadow=False, fancybox=False, frameon=False)
            # angle = -60 if row else 0
            # if pos == (0, 0):
            #     xpos, ypos = 0.7, 0.64
            # elif pos == (0, 1):
            #     xpos, ypos = 0.05, 0.64
            # elif pos == (1, 1):
            #     xpos, ypos = 0.65, 0.4
            # if pos != (1, 0):
            #     ax.text(xpos, ypos, text1, color='k', fontsize=12, rotation=angle,
            #             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            #             horizontalalignment='left', verticalalignment='center',
            #             transform=ax.transAxes)
        else:
            ax.plot([dene_min, dene_max],
                    [math.log10(sigma_e*0.5), math.log10(sigma_e*0.5)],
                    color='k', linewidth=1, linestyle='-', label=r'$\sigma_e/2$')
            ax.plot([math.log10(sigma_e*0.5), math.log10(sigma_e*0.5)], [-1, 3],
                    color='k', linewidth=1, linestyle='-')
            xpos = 0.87 if col else 0.05
            ypos = 0.35 if row else 0.64
            # ax.text(xpos, ypos, r'$\sigma_e/2$', color='k', fontsize=12,
            #         bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            #         horizontalalignment='left', verticalalignment='center',
            #         transform=ax.transAxes)
            xpos = 0.63 if col else 0.33
            ypos = 0.08 if row else 0.9
            # ax.text(xpos, ypos, r'$\sigma_e/2$', color='k', fontsize=12,
            #         rotation=90,
            #         bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            #         horizontalalignment='left', verticalalignment='center',
            #         transform=ax.transAxes)
            pos = (row, col)
            if pos == (1, 0):
                ax.legend(loc=4, prop={'size': 10}, ncol=1,
                          shadow=False, fancybox=False, frameon=False)
        xlabel = labels["xlabel"]
        ylabel = labels["ylabel"]
        if row == 1:
            ax.set_xlabel(xlabel, fontsize=16)
        else:
            ax.tick_params(axis='x', labelbottom=False)
        if col == 0:
            ax.set_ylabel(ylabel, fontsize=16)
        else:
            ax.tick_params(axis='y', labelleft=False)

        # angle = -45 if row == col else 45
        # pos = (row, col)
        # if pos == (0, 0):
        #     xpos, ypos = 0.03, 0.7
        # elif pos == (0, 1):
        #     xpos, ypos = 0.6, 0.68
        # elif pos == (1, 0):
        #     xpos, ypos = 0.19, 0.25
        # elif pos == (1, 1):
        #     xpos, ypos = 0.5, 0.24
        # ax.text(xpos, ypos, text_diag, color='k', fontsize=12, rotation=angle,
        #         bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        #         horizontalalignment='left', verticalalignment='center',
        #         transform=ax.transAxes)
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')
        xlim = [-1, 3] if col else [3, -1]
        ax.set_xlim(xlim)
        ylim = [3, -1] if row else [-1, 3]
        ax.set_ylim(ylim)
        ax.tick_params(labelsize=12)
        if row and col:
            rect_cbar = np.copy(rect)
            rect_cbar[0] += rect[2] + 0.02
            rect_cbar[1] += (rect[3] + vgap) * 0.5
            rect_cbar[2] = 0.02
            cbar_ax = fig.add_axes(rect_cbar)
            cbar = fig.colorbar(p1, cax=cbar_ax, extend='both')
            cbar_ax.tick_params(bottom=False, top=False, left=False, right=True)
            cbar_ax.tick_params(axis='y', which='major', direction='out')
            cbar_ax.tick_params(axis='y', which='minor', direction='in', right=False)
            cbar_ax.set_title(r'$N_e$', fontsize=16)
            cbar.ax.tick_params(labelsize=12)
        twpe = math.ceil(twpe10) * 0.1
        text1 = r'$t\omega_{pe}=' + ("{%0.0f}" % twpe) + '$'
        fig.suptitle(text1, fontsize=16)
    fig.savefig(fname)

    if show_plot:
        plt.show()
    else:
        plt.close()


def create_fig_dir(pic_run, var='wpara_wperp'):
    """Create directory for saving figures

    Args:
        pic_run: run name of the PIC simulation
        var(optional): variable name
    """
    top_dir = '../img/relativistic_turbulence/'
    var_name = var + '_four'
    fig_dir = top_dir + var_name + '/' + pic_run + '/'
    mkdir_p(fig_dir)
    return fig_dir


def set_fig_name(fig_dir, ixyz, istep, species='e', var='wpara_wperp'):
    """Save figure name

    Args:
        fig_dir: directory for saving figures
        ixyz: different components
        istep: time step
        species(optional): particle species ('e' or 'i')
        var(optional): variable name
    """
    xyz = ['x', 'y', 'z']
    fig_name = fig_dir + var + '_4_'
    if ixyz < 3:
        fig_name += xyz[ixyz]
    else:
        fig_name += 'tot'

    fig_name += '_' + species + '_' + str(istep) + '.pdf'
    return fig_name


def set_labels(ixyz, xsign, ysign, var='wpara_wperp'):
    """Set labels for different variables

    Args:
        ixyz: different components
        xsign, ysign: to indicate sign
        var(optional): variable name
    """
    labels = {}
    xyz = ['x', 'y', 'z']
    if var == 'wpara_wperp':
        if ixyz < 3:
            labels["xlabel"] = (r'$\log(' + '|W_{\parallel,' +
                                xyz[ixyz] + '}' + xsign + '|)$')
            labels["ylabel"] = (r'$\log(' + '|W_{\perp,' +
                                xyz[ixyz] + '}' + ysign + '|)$')
            labels["xtext"] = r'$' + '|W_{\parallel,' + xyz[ixyz] + '}' + '|$'
            labels["ytext"] = r'$' + '|W_{\perp,' + xyz[ixyz] + '}' + '|$'
        else:
            labels["xlabel"] = r'$\log(' + '|W_\parallel' + xsign + '|)$'
            labels["ylabel"] = r'$\log(' + '|W_\perp' + ysign + '|)$'
            labels["xtext"] = r'$' + '|W_\parallel' + '|$'
            labels["ytext"] = r'$' + '|W_\perp' + '|$'
    elif var == 'wpara_dgamma':
        if ixyz < 3:
            labels["xlabel"] = (r'$\log(' + '|W_{\parallel,' +
                                xyz[ixyz] + '}' + xsign + '|)$')
            labels["ylabel"] = (r'$\log(' + '|\Delta\gamma' + ysign + '|)$')
            labels["xtext"] = r'$' + '|W_{\parallel,' + xyz[ixyz] + '}' + '|$'
            labels["ytext"] = r'$' + '|\Delta\gamma|$'
        else:
            labels["xlabel"] = r'$\log(' + '|W_\parallel' + xsign + '|)$'
            labels["ylabel"] = r'$\log(' + '|\Delta\gamma' + ysign + '|)$'
            labels["xtext"] = r'$' + '|W_\parallel' + '|$'
            labels["ytext"] = r'$' + '|\Delta\gamma' + '|$'
    elif var == 'wperp_dgamma':
        if ixyz < 3:
            labels["xlabel"] = (r'$\log(' + '|W_{\perp,' +
                                xyz[ixyz] + '}' + xsign + '|)$')
            labels["ylabel"] = (r'$\log(' + '|\Delta\gamma' + ysign + '|)$')
            labels["xtext"] = r'$' + '|W_{\perp,' + xyz[ixyz] + '}' + '|$'
            labels["ytext"] = r'$' + '|\Delta\gamma|$'
        else:
            labels["xlabel"] = r'$\log(' + '|W_\perp' + xsign + '|)$'
            labels["ylabel"] = r'$\log(' + '|\Delta\gamma' + ysign + '|)$'
            labels["xtext"] = r'$' + '|W_\perp' + '|$'
            labels["ytext"] = r'$' + '|\Delta\gamma' + '|$'
    elif var == 'wtot_dgamma':
        if ixyz < 3:
            labels["xlabel"] = r'$\log(' + '|W_{' + xyz[ixyz] + '}' + xsign + '|)$'
            labels["ylabel"] = r'$\log(' + '|\Delta\gamma' + ysign + '|)$'
            labels["xtext"] = r'$' + '|W_{' + xyz[ixyz] + '}' + '|$'
            labels["ytext"] = r'$' + '|\Delta\gamma|$'
        else:
            labels["xlabel"] = r'$\log(' + '|W' + xsign + '|)$'
            labels["ylabel"] = r'$\log(' + '|\Delta\gamma' + ysign + '|)$'
            labels["xtext"] = r'$' + '|W' + '|$'
            labels["ytext"] = r'$' + '|\Delta\gamma' + '|$'

    return labels


def calc_wpara_wperp(plot_config, show_plot=True):
    """
    Calculate energization due to parallel and perpendicular electric field when
    particle energy reach sigma_e/2

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    pic_run_dir = plot_config["pic_run_dir"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    dtwpe_tracer = pic_info.dtwpe * pic_info.tracer_interval
    sigma_e = 5.524770e+02
    fname = pic_run_dir + '/info'
    with open(fname) as f:
        content = f.readlines()
    f.close()
    half_thickness_cs, _ = get_variable_value('L/de', 0, content)

    tracer_dir = pic_run_dir + 'tracer/tracer1/'
    nframes = len(os.listdir(tracer_dir))
    # nframes = 20
    plot_interval = plot_config["plot_interval"]

    if species in ["e", "electron"]:
        sname = "electron"
        pmass = 1.0
    else:
        sname = "H"
        pmass = pic_info.mime

    fname = tracer_dir + 'T.0/electron_tracer_qtag_sorted.h5p'
    with h5py.File(fname, 'r') as fh:
        group = fh['Step#0']
        dset = group['dX']
        nptl, = dset.shape
        ptl = {}
        for dset in group:
            dset = str(dset)
            ptl[str(dset)] = read_var(group, dset, nptl)

    gamma0 = np.zeros(nptl)
    dgamma = np.zeros(nptl)
    dgamma_pre = np.zeros(nptl)
    dgamma_pos = np.zeros(nptl)
    dene_para = np.zeros([4, nptl])
    dene_perp = np.zeros([4, nptl])
    dene_para_cross = np.zeros([4, nptl])
    dene_perp_cross = np.zeros([4, nptl])

    gamma0 = np.sqrt(1.0 + ptl["Ux"]**2 + ptl["Uy"]**2 + ptl["Uz"]**2)
    cross_half_sigmae = gamma0 > sigma_e * 0.5

    fdir = '../data/relativistic_turbulence/wpara_wperp/' + pic_run + '/'
    mkdir_p(fdir)

    for tframe in range(nframes):
        print("Time frame %d of %d" % (tframe, nframes))
        tindex0 = tframe * pic_info.tracer_interval * plot_config["nsteps"]
        fname = (tracer_dir + 'T.' + str(tindex0) + '/' +
                 sname + '_tracer_qtag_sorted.h5p')
        fh = h5py.File(fname, 'r')
        for step in range(plot_config["nsteps"]):
            tindex = step * pic_info.tracer_interval + tindex0
            print("Time index: %d" % tindex)
            gname = 'Step#' + str(tindex)
            if not gname in fh:
                break
            group = fh[gname]
            ptl = {}
            for dset in group:
                dset = str(dset)
                ptl[str(dset)] = read_var(group, dset, nptl)
            ux = ptl["Ux"]
            uy = ptl["Uy"]
            uz = ptl["Uz"]
            gamma = np.sqrt(1.0 + ux**2 + uy**2 + uz**2)
            dgamma = gamma - gamma0
            dgamma_pos = np.copy(dgamma)
            ib2 = 1.0 / (ptl["Bx"]**2 + ptl["By"]**2 + ptl["Bz"]**2)
            edotb = ptl["Ex"] * ptl["Bx"] + ptl["Ey"] * ptl["By"] + ptl["Ez"] * ptl["Bz"]
            eparax = edotb * ptl["Bx"] * ib2
            eparay = edotb * ptl["By"] * ib2
            eparaz = edotb * ptl["Bz"] * ib2
            eperpx = ptl["Ex"] - eparax
            eperpy = ptl["Ey"] - eparay
            eperpz = ptl["Ez"] - eparaz
            igamma = 1.0 / gamma
            vx = ux * igamma
            vy = uy * igamma
            vz = uz * igamma
            dene_para[0, :] += -vx * eparax
            dene_para[1, :] += -vy * eparay
            dene_para[2, :] += -vz * eparaz
            dene_para[3, :] = np.sum(dene_para[:3, :], axis=0)
            dene_perp[0, :] += -vx * eperpx
            dene_perp[1, :] += -vy * eperpy
            dene_perp[2, :] += -vz * eperpz
            dene_perp[3, :] = np.sum(dene_perp[:3, :], axis=0)
            dene_para *= dtwpe_tracer
            dene_perp *= dtwpe_tracer
            cond = np.logical_and(dgamma_pre < 0.5 * sigma_e, dgamma_pos > 0.5 * sigma_e)
            cond = np.logical_and(cond, np.logical_not(cross_half_sigmae))
            dene_para_cross[:, cond] = dene_para[:, cond]
            dene_perp_cross[:, cond] = dene_perp[:, cond]
            dgamma_pre = np.copy(dgamma_pos)

            istep = tindex // pic_info.tracer_interval
            if istep % plot_interval == 0:
                fname = fdir + 'wpara_cross_' + str(istep) + '.dat'
                dene_para_cross.tofile(fname)
                fname = fdir + 'wperp_cross_' + str(istep) + '.dat'
                dene_perp_cross.tofile(fname)
                fname = fdir + 'wpara_' + str(istep) + '.dat'
                dene_para.tofile(fname)
                fname = fdir + 'wperp_' + str(istep) + '.dat'
                dene_perp.tofile(fname)
                fdata = cross_half_sigmae.astype(int)
                fname = fdir + 'cross_half_sigma_' + str(istep) + '.dat'
                fdata.tofile(fname)

            cross_half_sigmae = np.logical_or(cond, cross_half_sigmae)
            dene_para /= dtwpe_tracer
            dene_perp /= dtwpe_tracer
        fh.close()


def plot_wpara_wperp(plot_config, show_plot=True):
    """
    Plot energization due to parallel and perpendicular electric field when
    particle energy reach sigma_e/2

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    vpic_info = get_vpic_info(pic_run_dir)
    dtwpe_tracer = vpic_info["tracer_interval"] * vpic_info["dt*wpe"]
    sigma_e = 1.0 / vpic_info["wpe/wce"]**2

    tracer_dir = pic_run_dir + 'tracer/tracer1/'
    nsteps = plot_config["nsteps"]
    plot_interval = plot_config["plot_interval"]
    nframes = len(os.listdir(tracer_dir)) * nsteps
    plot_interval = plot_config["plot_interval"]

    if plot_config["wpp_hdf5"]:
        data_dir = pic_run_dir + 'wpara_wperp_2nd_pass/'
    else:
        data_dir = '../data/relativistic_turbulence/wpara_wperp/' + pic_run + '/'

    if plot_config["wpp_hdf5"]:
        fdir = pic_run_dir + 'wpara_wperp_2nd_pass/'
        file_list = os.listdir(fdir)
        file_list.sort()
        tframes = []
        for file_name in file_list:
            fsplit = file_name.split(".")
            tindex = int(fsplit[0].split("_")[-1])
            tframes.append(tindex)
    else:
        tframes = range(0, nframes, plot_interval)
    ntf = len(tframes)
    ncross = np.zeros(ntf)
    number_larger_wperp = np.zeros(ntf)
    number_larger_wpara = np.zeros(ntf)
    wperp_avg = np.zeros(ntf)
    wpara_avg = np.zeros(ntf)
    wperp_tot = np.zeros(ntf)
    wpara_tot = np.zeros(ntf)
    ttracer = np.asarray(tframes) * dtwpe_tracer

    fig1 = plt.figure(figsize=[5, 3.5])
    rect = [0.17, 0.16, 0.78, 0.8]
    ax1 = fig1.add_axes(rect)
    fig2 = plt.figure(figsize=[5, 3.5])
    ax2 = fig2.add_axes(rect)
    fig3 = plt.figure(figsize=[5, 3.5])
    ax3 = fig3.add_axes(rect)

    if species in ["e", "electron"]:
        sname = "electron"
    else:
        sname = "ion"

    ptl = {}
    for iframe, tframe in enumerate(tframes):
        print("Time frame: %d" % tframe)
        if plot_config["wpp_hdf5"]:
            fname = (data_dir + 'wpara_wperp_' + sname + '_' +
                     str(tframe).zfill(6) + '.h5')
            with h5py.File(fname, 'r') as fh:
                for dset in fh:
                    dset_name = str(dset)
                    dset = fh[dset_name]
                    fdata = np.zeros(dset.shape, dtype=dset.dtype)
                    dset.read_direct(fdata)
                    ptl[dset_name] = np.copy(fdata)
            ncross[iframe] = np.sum(ptl["cross_half_sigmae"])
            cond = ptl["cross_half_sigmae"].astype(bool)
            fdata_para = ptl["wpara_cross"][cond]
            fdata_perp = ptl["wperp_cross"][cond]
        else:
            fname = data_dir + 'wpara_cross_' + str(tframe) + '.dat'
            dene_para_cross = np.fromfile(fname).reshape([4, -1])
            fname = data_dir + 'wperp_cross_' + str(tframe) + '.dat'
            dene_perp_cross = np.fromfile(fname).reshape([4, -1])
            fname = data_dir + 'cross_half_sigma_' + str(tframe) + '.dat'
            cross_half_sigma = np.fromfile(fname, dtype=int)
            ncross[iframe] = np.sum(cross_half_sigma)
            cond = cross_half_sigma.astype(bool)
            fdata_para = dene_para_cross[3, cond]
            fdata_perp = dene_perp_cross[3, cond]
        cond1 = fdata_para > fdata_perp
        number_larger_wpara[iframe] = np.sum(cond1)
        cond1 = fdata_para < fdata_perp
        number_larger_wperp[iframe] = np.sum(cond1)
        if ncross[iframe]:
            wpara_avg[iframe] = np.mean(fdata_para)
            wperp_avg[iframe] = np.mean(fdata_perp)
        wpara_tot[iframe] = np.sum(fdata_para)
        wperp_tot[iframe] = np.sum(fdata_perp)

    COLORS = palettable.tableau.Tableau_10.mpl_colors
    ax1.set_prop_cycle('color', COLORS)
    label1 = r'$N_e(W_\parallel > W_\perp)$'
    label2 = r'$N_e(W_\parallel < W_\perp)$'
    ax1.plot(ttracer, number_larger_wpara, linewidth=2, label=label1)
    ax1.plot(ttracer, number_larger_wperp, linewidth=2, label=label2)
    print("Percentage for Wperp > Wpara in the end: %f" %
          (number_larger_wperp[-1]/ncross[-1]))
    print("Percentage for Wpara > Wperp in the end: %f" %
          (number_larger_wpara[-1]/ncross[-1]))
    ax1.legend(loc=2, prop={'size': 12}, ncol=1,
               shadow=False, fancybox=False, frameon=False)
    ax1.set_xlim([ttracer.min(), ttracer.max()])
    ax1.tick_params(bottom=True, top=True, left=True, right=True)
    ax1.tick_params(axis='x', which='minor', direction='in')
    ax1.tick_params(axis='x', which='major', direction='in')
    ax1.tick_params(axis='y', which='minor', direction='in')
    ax1.tick_params(axis='y', which='major', direction='in')
    ax1.set_xlabel(r'$t\omega_{pe}$', fontsize=16)
    ax1.set_ylabel(r'$N_e$', fontsize=16)
    ax1.tick_params(labelsize=12)

    COLORS = palettable.tableau.Tableau_10.mpl_colors
    ax2.set_prop_cycle('color', COLORS)
    label1 = r'$\left<W_\parallel\right>$'
    label2 = r'$\left<W_\perp\right>$'
    ax2.plot(ttracer, wpara_avg, linewidth=2, label=label1)
    ax2.plot(ttracer, wperp_avg, linewidth=2, label=label2)
    wavg = wpara_avg + wperp_avg
    print("Percentage of Wpara in the end: %f" % (wpara_avg[-1]/wavg[-1]))
    print("Percentage of Wperp in the end: %f" % (wperp_avg[-1]/wavg[-1]))
    ax2.legend(loc=4, prop={'size': 12}, ncol=1,
               shadow=False, fancybox=False, frameon=False)
    ax2.set_xlim([ttracer.min(), ttracer.max()])
    ax2.tick_params(bottom=True, top=True, left=True, right=True)
    ax2.tick_params(axis='x', which='minor', direction='in')
    ax2.tick_params(axis='x', which='major', direction='in')
    ax2.tick_params(axis='y', which='minor', direction='in')
    ax2.tick_params(axis='y', which='major', direction='in')
    ax2.set_xlabel(r'$t\omega_{pe}$', fontsize=16)
    ax2.set_ylabel(r'$\left<W\right>$', fontsize=16)
    ax2.tick_params(labelsize=12)

    COLORS = palettable.tableau.Tableau_10.mpl_colors
    ax3.set_prop_cycle('color', COLORS)
    label1 = r'$\sum W_\parallel/\sigma_e$'
    label2 = r'$\sum W_\perp/\sigma_e$'
    ax3.plot(ttracer, wpara_tot/sigma_e, linewidth=2, label=label1)
    ax3.plot(ttracer, wperp_tot/sigma_e, linewidth=2, label=label2)
    wtot = wpara_tot + wperp_tot
    print("Percentage of Wpara in the end: %f" % (wpara_tot[-1]/wtot[-1]))
    print("Percentage of Wperp in the end: %f" % (wperp_tot[-1]/wtot[-1]))
    ax3.legend(loc=2, prop={'size': 12}, ncol=1,
               shadow=False, fancybox=False, frameon=False)
    ax3.set_xlim([ttracer.min(), ttracer.max()])
    ax3.tick_params(bottom=True, top=True, left=True, right=True)
    ax3.tick_params(axis='x', which='minor', direction='in')
    ax3.tick_params(axis='x', which='major', direction='in')
    ax3.tick_params(axis='y', which='minor', direction='in')
    ax3.tick_params(axis='y', which='major', direction='in')
    ax3.set_xlabel(r'$t\omega_{pe}$', fontsize=16)
    ax3.set_ylabel(r'$\sum W/\sigma_e$', fontsize=16)
    ax3.tick_params(labelsize=12)


    fdir = '../img/relativistic_turbulence/wpara_wperp/' + pic_run + '/'
    mkdir_p(fdir)

    fname = fdir + 'num_wpara_wperp.pdf'
    fig1.savefig(fname)

    fname = fdir + 'ene_wpara_wperp.pdf'
    fig2.savefig(fname)

    fname = fdir + 'ene_wpara_wperp_tot.pdf'
    fig3.savefig(fname)

    plt.show()
    # plt.close('all')


def compare_spectrum(plot_config, show_plot=True):
    """Compare spectrum for different injection mechanism

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    pic_run_dir = plot_config["pic_run_dir"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    vpic_info = get_vpic_info(pic_run_dir)
    dtwpe_tracer = vpic_info["tracer_interval"] * vpic_info["dt*wpe"]
    sigma_e = 1.0 / vpic_info["wpe/wce"]**2

    tracer_dir = pic_run_dir + 'tracer/tracer1/'
    nsteps = plot_config["nsteps"]
    plot_interval = plot_config["plot_interval"]
    nframes = len(os.listdir(tracer_dir)) * nsteps

    if species in ["e", "electron"]:
        sname = "electron"
    else:
        sname = "ion"

    if plot_config["wpp_hdf5"]:
        fdir = pic_run_dir + 'wpara_wperp_2nd_pass/'
        file_list = os.listdir(fdir)
        file_list.sort()
        fsplit = file_list[-1].split(".")
        tframe = int(fsplit[0].split("_")[-1])
        fname = (fdir + 'wpara_wperp_' + sname + '_' +
                 str(tframe).zfill(6) + '.h5')
        ptl = {}
        with h5py.File(fname, 'r') as fh:
            for dset in fh:
                dset_name = str(dset)
                dset = fh[dset_name]
                fdata = np.zeros(dset.shape, dtype=dset.dtype)
                dset.read_direct(fdata)
                ptl[dset_name] = np.copy(fdata)
        cond_cross = ptl["cross_half_sigmae"].astype(bool)
        fdata_para = ptl["wpara_cross"][cond_cross]
        fdata_perp = ptl["wperp_cross"][cond_cross]
    else:
        tframe = (nframes - 1) // plot_interval * plot_interval
        fname = fdir + 'wpara_cross_' + str(tframe) + '.dat'
        dene_para_cross = np.fromfile(fname).reshape([4, -1])
        fname = fdir + 'wperp_cross_' + str(tframe) + '.dat'
        dene_perp_cross = np.fromfile(fname).reshape([4, -1])
        fname = fdir + 'cross_half_sigma_' + str(tframe) + '.dat'
        cross_half_sigma = np.fromfile(fname, dtype=int)
        cond_cross = cross_half_sigma.astype(bool)
        fdata_para = dene_para_cross[3, cond_cross]
        fdata_perp = dene_perp_cross[3, cond_cross]
    cond1 = fdata_para > fdata_perp
    cond2 = fdata_para < fdata_perp

    if plot_config["wpp_hdf5"]:
        vpic_info = get_vpic_info(pic_run_dir)
        tracer_file_interval = int(vpic_info["tracer_file_interval"])
        tindex = tframe
        tindex_file = (tindex // tracer_file_interval) * tracer_file_interval
    else:
        tindex = tframe * pic_info.tracer_interval
        tinterval_file = plot_config["nsteps"] * pic_info.tracer_interval
        tindex_file = tindex // (tinterval_file) * tinterval_file
    fname = tracer_dir + 'T.' + str(tindex_file) + '/electron_tracer_qtag_sorted.h5p'
    with h5py.File(fname, 'r') as fh:
        group = fh['Step#' + str(tindex)]
        dset = group['dX']
        nptl, = dset.shape
        ptl = {}
        ptl_all = {}
        for dset in group:
            dset = str(dset)
            fdata = read_var(group, dset, nptl)
            ptl_all[str(dset)] = fdata
            ptl[str(dset)] = fdata[cond_cross]
    ux = ptl["Ux"][cond1]
    uy = ptl["Uy"][cond1]
    uz = ptl["Uz"][cond1]
    gamma1 = np.sqrt(1 + ux**2 + uy**2 + uz**2)
    ux = ptl["Ux"][cond2]
    uy = ptl["Uy"][cond2]
    uz = ptl["Uz"][cond2]
    gamma2 = np.sqrt(1 + ux**2 + uy**2 + uz**2)
    gamma_all = np.sqrt(1.0 + ptl_all["Ux"]**2 + ptl_all["Uy"]**2 + ptl_all["Uz"]**2)

    emin, emax = 1E-4, 1E6
    nbins = 100
    dloge = (math.log10(emax) - math.log10(emin)) / (nbins - 1)
    emin0 = 10**(math.log10(emin) - dloge)
    ebins = np.logspace(math.log10(emin0), math.log10(emax), nbins+1)
    ebins_mid = (ebins[:-1] + ebins[1:]) * 0.5
    debins = np.diff(ebins)

    spect1, _ = np.histogram(gamma1 - 1, bins=ebins)
    spect2, _ = np.histogram(gamma2 - 1, bins=ebins)
    spect_all, _ = np.histogram(gamma_all - 1, bins=ebins)

    fig = plt.figure(figsize=[5, 3.5])
    rect = [0.15, 0.16, 0.8, 0.8]
    ax = fig.add_axes(rect)
    COLORS = palettable.colorbrewer.qualitative.Set1_9.mpl_colors
    ax.loglog(ebins_mid, spect1/debins, linewidth=2,
              label=r'$W_\parallel > W_\perp$', color=COLORS[0])
    ax.loglog(ebins_mid, spect2/debins, linewidth=2,
              label=r'$W_\parallel < W_\perp$', color='k')
    ax.loglog(ebins_mid, spect_all/debins, linewidth=1, color='k',
              linestyle='--', label='All tracers')

    ax.legend(loc=1, prop={'size': 12}, ncol=1,
             shadow=False, fancybox=False, frameon=False)
    ax.set_xlim([1E0, 1E3])
    ax.set_ylim([1E-2, 1E6])
    ax.set_yticks(np.logspace(-2, 4, num=4))
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlabel(r'$\gamma - 1$', fontsize=16)
    ax.set_ylabel(r'$f(\gamma - 1)$', fontsize=16)
    ax.tick_params(labelsize=12)
    fdir = '../img/relativistic_turbulence/wpara_wperp/' + pic_run + '/'
    mkdir_p(fdir)

    fname = fdir + 'spect_wpara_wperp.pdf'
    fig.savefig(fname)

    plt.show()


def plot_spect_species(plot_config, show_plot=True):
    """plot energy spectrum for different test-particle species

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    tstart = plot_config["tstart"]
    tend = plot_config["tend"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    if species in ["e", "electron"]:
        sname = "electron"
        pmass = 1.0
    else:
        sname = "ion"
        pmass = pic_info.mime
    sp_list = [sname]
    sp_name_list = ["Regular " + sname]
    test_species = ["wo_epara", "wo_eparay", "wo_egtb", "egtb", "egtb_egtb"]
    species_name = [r"$E_\parallel=0$", r"$E_{\parallel y}=0$",
                    r"$E=0 \text{ when } E>B$", "tracers",
                    r"tracers passing $E>B$"]
    for tsp, tsp_name in zip(test_species, species_name):
        sp_list.append(sname + '_' + tsp)
        sp_name_list.append("Test " + sname + ": " + tsp_name)
    norms = [1, 10, 10, 10, 10, 10, 10]
    ntot = pic_info.nx * pic_info.nz * pic_info.nppc
    emin, emax = 1E-4, 1E6
    nbins = 1000
    dloge = (math.log10(emax) - math.log10(emin)) / (nbins - 1)
    emin0 = 10**(math.log10(emin) - dloge)
    ebins = np.logspace(math.log10(emin0), math.log10(emax), nbins+1)
    ebins_mid = (ebins[:-1] + ebins[1:]) * 0.5
    ebins_mid *= pmass
    debins = np.diff(ebins)
    espect_elb = np.zeros(nbins)
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    vpic_info = get_vpic_info(pic_run_dir)
    spectrum_interval = int(vpic_info["spectrum_interval"])
    dtwpe_spectrum = math.ceil(spectrum_interval * vpic_info["dt*wpe"] / 0.1) * 0.1
    for tframe in range(tstart, tend+1):
        print("Time frame: %d" % tframe)
        tindex = tframe * spectrum_interval
        fig = plt.figure(figsize=[7, 5])
        rect = [0.12, 0.12, 0.82, 0.8]
        ax = fig.add_axes(rect)
        ax.set_prop_cycle('color', COLORS)
        isp = 0
        for sp, sp_name in zip(sp_list, sp_name_list):
            fname = (pic_run_dir + "spectrum_combined/spectrum_" + sp +
                     "_" + str(tindex) + ".dat")
            flog = np.fromfile(fname, dtype=np.float32)
            espect = flog[3:]  # the first 3 are magnetic field
            espect /= ntot
            if sp != sname + "_egtb":
                ax.loglog(ebins_mid, espect*norms[isp], linewidth=2, label=sp_name)
            else:
                espect_elb = np.copy(espect)
            if sp == sname + "_egtb_egtb":
                espect_elb -= espect
            isp += 1
        sp_name = r"Test " + sname + r": tracers never passing $E>B$"
        ax.loglog(ebins_mid, espect_elb*norms[isp], linewidth=2, label=sp_name)
        ax.legend(loc=3, prop={'size': 12}, ncol=1,
                 shadow=False, fancybox=False, frameon=False)
        ax.set_xlim([1E-2, 1E3])
        ax.set_ylim([1E-8, 1E-1])
        if species in ["e", "electron"]:
            ax.set_xlabel(r'$\gamma - 1$', fontsize=16)
        else:
            ax.set_xlabel(r'$(\gamma - 1)m_i/m_e$', fontsize=16)
        ax.set_ylabel(r'$(\gamma - 1)f(\gamma - 1)$', fontsize=16)
        ax.tick_params(labelsize=12)
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')
        twpe = tframe * dtwpe_spectrum
        text1 = r'$t\omega_{pe}=' + str(twpe) + '$'
        ax.set_title(text1, fontsize=16)
        fdir = '../img/relativistic_turbulence/spect_species/' + pic_run + '/'
        mkdir_p(fdir)
        fname = fdir + 'spects_' + species + '_' + str(tframe) + '.pdf'
        fig.savefig(fname)
        if show_plot:
            plt.show()
        else:
            plt.close()


def egain_after_injection(plot_config, show_plot=True):
    """
    Plot energy gain after injection

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    pic_run_dir = plot_config["pic_run_dir"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    vpic_info = get_vpic_info(pic_run_dir)
    dtwpe_tracer = vpic_info["tracer_interval"] * vpic_info["dt*wpe"]
    sigma_e = 1.0 / vpic_info["wpe/wce"]**2

    tracer_dir = pic_run_dir + 'tracer/tracer1/'
    nsteps = plot_config["nsteps"]
    plot_interval = plot_config["plot_interval"]
    nframes = len(os.listdir(tracer_dir)) * nsteps

    if plot_config["wpp_hdf5"]:
        data_dir = pic_run_dir + 'wpara_wperp_2nd_pass/'
    else:
        data_dir = '../data/relativistic_turbulence/wpara_wperp/' + pic_run + '/'

    if plot_config["wpp_hdf5"]:
        fdir = pic_run_dir + 'wpara_wperp_2nd_pass/'
        file_list = os.listdir(fdir)
        file_list.sort()
        tframes = []
        for file_name in file_list:
            fsplit = file_name.split(".")
            tindex = int(fsplit[0].split("_")[-1])
            tframes.append(tindex)
    else:
        tframes = range(0, nframes, plot_interval)
    ntf = len(tframes)
    ncross = np.zeros(ntf)
    number_larger_wperp = np.zeros(ntf)
    number_larger_wpara = np.zeros(ntf)
    wperp_avg = np.zeros(ntf)
    wpara_avg = np.zeros(ntf)
    wperp_post_avg = np.zeros(ntf)
    wpara_post_avg = np.zeros(ntf)
    ttracer = np.asarray(tframes) * dtwpe_tracer

    fig1 = plt.figure(figsize=[5, 3.5])
    rect = [0.15, 0.16, 0.8, 0.8]
    ax1 = fig1.add_axes(rect)

    if species in ["e", "electron"]:
        sname = "electron"
    else:
        sname = "H"

    ptl = {}
    for iframe, tframe in enumerate(tframes):
        print("Time frame: %d" % tframe)
        if plot_config["wpp_hdf5"]:
            fname = (data_dir + 'wpara_wperp_' + sname + '_' +
                     str(tframe).zfill(6) + '.h5')
            with h5py.File(fname, 'r') as fh:
                for dset in fh:
                    dset_name = str(dset)
                    dset = fh[dset_name]
                    fdata = np.zeros(dset.shape, dtype=dset.dtype)
                    dset.read_direct(fdata)
                    ptl[dset_name] = np.copy(fdata)
            ncross[iframe] = np.sum(ptl["cross_half_sigmae"])
            cond = ptl["cross_half_sigmae"].astype(bool)
            fdata_para = ptl["wpara_cross"][cond]
            fdata_perp = ptl["wperp_cross"][cond]
            fdata_para_tot = ptl["wpara"][cond]
            fdata_perp_tot = ptl["wperp"][cond]
        else:
            fname = data_dir + 'wpara_cross_' + str(tframe) + '.dat'
            dene_para_cross = np.fromfile(fname).reshape([4, -1])
            fname = data_dir + 'wperp_cross_' + str(tframe) + '.dat'
            dene_perp_cross = np.fromfile(fname).reshape([4, -1])
            fname = data_dir + 'wpara_' + str(tframe) + '.dat'
            dene_para = np.fromfile(fname).reshape([4, -1])
            fname = data_dir + 'wperp_' + str(tframe) + '.dat'
            dene_perp = np.fromfile(fname).reshape([4, -1])
            fname = data_dir + 'cross_half_sigma_' + str(tframe) + '.dat'
            cross_half_sigma = np.fromfile(fname, dtype=int)
            ncross[iframe] = np.sum(cross_half_sigma)
            cond = cross_half_sigma.astype(bool)
            fdata_para = dene_para_cross[3, cond]
            fdata_perp = dene_perp_cross[3, cond]
            fdata_para_tot = dene_para[3, cond]
            fdata_perp_tot = dene_perp[3, cond]
        cond1 = fdata_para > fdata_perp
        number_larger_wpara[iframe] = np.sum(cond1)
        cond1 = fdata_para < fdata_perp
        number_larger_wperp[iframe] = np.sum(cond1)
        if ncross[iframe]:
            wpara_avg[iframe] = np.mean(fdata_para)
            wperp_avg[iframe] = np.mean(fdata_perp)
            wpara_post_avg[iframe] = np.mean(fdata_para_tot) - wpara_avg[iframe]
            wperp_post_avg[iframe] = np.mean(fdata_perp_tot) - wperp_avg[iframe]

    COLORS = palettable.colorbrewer.qualitative.Set1_9.mpl_colors
    ax1.set_prop_cycle('color', COLORS)
    label1 = r'$\left<W_\parallel\right>$'
    label2 = r'$\left<W_\perp\right>$'
    ax1.plot(ttracer, wpara_post_avg, linewidth=2, label=label1, color=COLORS[0])
    ax1.plot(ttracer, wperp_post_avg, linewidth=2, label=label2, color='k')
    ax1.legend(loc=4, prop={'size': 12}, ncol=1,
               shadow=False, fancybox=False, frameon=False)
    ax1.set_xlim([ttracer.min(), ttracer.max()])
    ax1.tick_params(bottom=True, top=True, left=True, right=True)
    ax1.tick_params(axis='x', which='minor', direction='in')
    ax1.tick_params(axis='x', which='major', direction='in')
    ax1.tick_params(axis='y', which='minor', direction='in')
    ax1.tick_params(axis='y', which='major', direction='in')
    ax1.set_xlabel(r'$t\omega_{pe}$', fontsize=16)
    ax1.set_ylabel(r'$\left<W\right>$ after crossing $\sigma_e/2$', fontsize=16)
    ax1.tick_params(labelsize=12)

    fdir = '../img/relativistic_turbulence/wpara_wperp_after_crossing/' + pic_run + '/'
    mkdir_p(fdir)

    fname = fdir + 'wpara_wperp_post.pdf'
    fig1.savefig(fname)

    plt.show()
    # plt.close()


def calc_wpara_wperp_1st(plot_config, show_plot=True):
    """
    Calculate wpara and wperp for runs with a larger number of time steps
    This is the first pass. The whole time series is separated into different
    pieces, so it can be proceeded in parallel.

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    pic_run_dir = plot_config["pic_run_dir"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)

    vpic_info = get_vpic_info(pic_run_dir)
    tracer_interval = int(vpic_info["tracer_interval"])
    tracer_file_interval = int(vpic_info["tracer_file_interval"])
    wpe_wce = vpic_info["wpe/wce"]
    dtwpe_tracer = tracer_interval * vpic_info["dt*wpe"]

    if species in ["e", "electron"]:
        sname = "electron"
    else:
        sname = "H"

    tframe = plot_config["tframe"]
    tracer_dir = pic_run_dir + 'tracer/tracer1/'
    fname = tracer_dir + 'T.0/electron_tracer_qtag_sorted.h5p'
    with h5py.File(fname, 'r') as fh:
        group = fh['Step#0']
        dset = group['dX']
        nptl, = dset.shape
        ptl = {}
        for dset in group:
            dset = str(dset)
            ptl[str(dset)] = read_var(group, dset, nptl)

    gamma0 = np.zeros(nptl)
    dgamma = np.zeros(nptl)
    dgamma_pre = np.zeros(nptl)
    dgamma_pos = np.zeros(nptl)
    dene_para = np.zeros([4, nptl])
    dene_perp = np.zeros([4, nptl])
    dene_para_cross = np.zeros([4, nptl])
    dene_perp_cross = np.zeros([4, nptl])

    gamma0 = np.sqrt(1.0 + ptl["Ux"]**2 + ptl["Uy"]**2 + ptl["Uz"]**2)
    sigma_e = 1.0 / wpe_wce**2
    cross_half_sigmae = gamma0 > sigma_e * 0.5

    fdir = pic_run_dir + 'wpara_wperp_1st_pass/'
    mkdir_p(fdir)

    tindex0 = tframe * tracer_file_interval
    fname = (tracer_dir + 'T.' + str(tindex0) + '/' +
             sname + '_tracer_qtag_sorted.h5p')
    fh = h5py.File(fname, 'r')
    nframes_in_file = len(fh)
    for iframe in range(nframes_in_file):
        tindex = iframe * tracer_interval + tindex0
        print("Time index: %d" % tindex)
        gname = 'Step#' + str(tindex)
        if not gname in fh:  # only possible for the last tracer directory
            break
        group = fh[gname]
        ptl = {}
        for dset in group:
            dset = str(dset)
            ptl[str(dset)] = read_var(group, dset, nptl)
        ux = ptl["Ux"]
        uy = ptl["Uy"]
        uz = ptl["Uz"]
        gamma = np.sqrt(1.0 + ux**2 + uy**2 + uz**2)
        dgamma = gamma - gamma0
        dgamma_pos = np.copy(dgamma)
        ib2 = 1.0 / (ptl["Bx"]**2 + ptl["By"]**2 + ptl["Bz"]**2)
        edotb = ptl["Ex"] * ptl["Bx"] + ptl["Ey"] * ptl["By"] + ptl["Ez"] * ptl["Bz"]
        eparax = edotb * ptl["Bx"] * ib2
        eparay = edotb * ptl["By"] * ib2
        eparaz = edotb * ptl["Bz"] * ib2
        eperpx = ptl["Ex"] - eparax
        eperpy = ptl["Ey"] - eparay
        eperpz = ptl["Ez"] - eparaz
        igamma = 1.0 / gamma
        vx = ux * igamma
        vy = uy * igamma
        vz = uz * igamma
        dene_para[0, :] += -vx * eparax
        dene_para[1, :] += -vy * eparay
        dene_para[2, :] += -vz * eparaz
        dene_para[3, :] = np.sum(dene_para[:3, :], axis=0)
        dene_perp[0, :] += -vx * eperpx
        dene_perp[1, :] += -vy * eperpy
        dene_perp[2, :] += -vz * eperpz
        dene_perp[3, :] = np.sum(dene_perp[:3, :], axis=0)
        # Has not crossed previously but crossed at this time step
        cond = np.logical_and(dgamma_pre < 0.5 * sigma_e, dgamma_pos > 0.5 * sigma_e)
        cond = np.logical_and(cond, np.logical_not(cross_half_sigmae))
        dene_para_cross[:, cond] = dene_para[:, cond]
        dene_perp_cross[:, cond] = dene_perp[:, cond]
        dgamma_pre = np.copy(dgamma_pos)

        iframe_g = tindex // tracer_interval
        plot_interval = plot_config["plot_interval"]
        if iframe_g % plot_interval == 0 or iframe == nframes_in_file - 1:
            fname = fdir + "wpara_wperp_" + sname + "_" + str(tindex).zfill(6) + '.h5'
            with h5py.File(fname, 'w') as fh_out:
                fh_out.create_dataset('wpara_cross', (nptl, ),
                                      data=dene_para_cross[3, :]*dtwpe_tracer)
                fh_out.create_dataset('wperp_cross', (nptl, ),
                                      data=dene_perp_cross[3, :]*dtwpe_tracer)
                fh_out.create_dataset('wpara', (nptl, ), data=dene_para[3, :]*dtwpe_tracer)
                fh_out.create_dataset('wperp', (nptl, ), data=dene_perp[3, :]*dtwpe_tracer)
                fh_out.create_dataset('dgamma', (nptl, ), data=dgamma)
                fdata = cross_half_sigmae.astype(int)
                fh_out.create_dataset('cross_half_sigmae', (nptl, ), data=fdata)
        cross_half_sigmae = np.logical_or(cond, cross_half_sigmae)
    fh.close()


def calc_wpara_wperp_2nd(plot_config, show_plot=True):
    """
    Calculate wpara and wperp for runs with a larger number of time steps
    This is the second pass.

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    pic_run_dir = plot_config["pic_run_dir"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)

    vpic_info = get_vpic_info(pic_run_dir)
    tracer_interval = int(vpic_info["tracer_interval"])
    tracer_file_interval = int(vpic_info["tracer_file_interval"])
    wpe_wce = vpic_info["wpe/wce"]
    dtwpe_tracer = tracer_interval * vpic_info["dt*wpe"]

    if species in ["e", "electron"]:
        sname = "electron"
    else:
        sname = "H"

    tracer_dir = pic_run_dir + 'tracer/tracer1/'
    fname = tracer_dir + 'T.0/' + sname + '_tracer_qtag_sorted.h5p'
    with h5py.File(fname, 'r') as fh:
        group = fh['Step#0']
        dset = group['dX']
        nptl, = dset.shape
        ptl = {}
        for dset in group:
            dset = str(dset)
            ptl[str(dset)] = read_var(group, dset, nptl)
    gamma0 = np.sqrt(1.0 + ptl["Ux"]**2 + ptl["Uy"]**2 + ptl["Uz"]**2)
    sigma_e = 1.0 / wpe_wce**2
    cross_half_sigmae = gamma0 > sigma_e * 0.5

    wpara = np.zeros(nptl)
    wperp = np.zeros(nptl)
    wpara0 = np.zeros(nptl)
    wperp0 = np.zeros(nptl)
    wpara_cross = np.zeros(nptl)
    wperp_cross = np.zeros(nptl)
    ptl = {}
    ptl["wpara"] = np.zeros(nptl)
    ptl["wperp"] = np.zeros(nptl)
    ptl["wpara_cross"] = np.zeros(nptl)
    ptl["wperp_cross"] = np.zeros(nptl)
    ptl["dgamma"] = np.zeros(nptl, np.float32)
    ptl["cross_half_sigmae"] = np.zeros(nptl, dtype=np.int)

    fdir = pic_run_dir + 'wpara_wperp_1st_pass/'
    file_list = os.listdir(fdir)
    file_list.sort()

    fdir_out = pic_run_dir + 'wpara_wperp_2nd_pass/'
    mkdir_p(fdir_out)

    for file_name in file_list:
        print("File name: %s" % file_name)
        fsplit = file_name.split(".")
        tindex = int(fsplit[0].split("_")[-1])
        fname = fdir + file_name
        with h5py.File(fname, 'r') as fh:
            for dset in fh:
                dset_name = str(dset)
                dset = fh[dset_name]
                dset.read_direct(ptl[dset_name])
        wpara = wpara0 + ptl["wpara"]
        wperp = wperp0 + ptl["wperp"]
        cond = np.logical_and(ptl["cross_half_sigmae"].astype(bool),
                              np.logical_not(cross_half_sigmae))
        wpara_cross[cond] = ptl["wpara_cross"][cond] + wpara0[cond]
        wperp_cross[cond] = ptl["wperp_cross"][cond] + wperp0[cond]
        cross_half_sigmae = np.logical_or(cond, cross_half_sigmae)
        fname = fdir_out + file_name
        if (tindex + tracer_interval) % tracer_file_interval == 0:
            wpara0 += ptl["wpara"]
            wperp0 += ptl["wperp"]
        with h5py.File(fname, 'w') as fh_out:
            fh_out.create_dataset('wpara_cross', (nptl, ), data=wpara_cross)
            fh_out.create_dataset('wperp_cross', (nptl, ), data=wperp_cross)
            fh_out.create_dataset('wpara', (nptl, ), data=wpara)
            fh_out.create_dataset('wperp', (nptl, ), data=wperp)
            fh_out.create_dataset('dgamma', (nptl, ), data=ptl["dgamma"])
            fdata = cross_half_sigmae.astype(int)
            fh_out.create_dataset('cross_half_sigmae', (nptl, ), data=fdata)


def compare_energization_four_h5(plot_config, show_plot=True):
    """Compare two energization terms in four panels

    Data is already calculated and saved in HDF5 format
    4 panels for positive and negative values

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    pic_run_dir = plot_config["pic_run_dir"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    vpic_info = get_vpic_info(pic_run_dir)
    sigma_e = 1.0 / vpic_info["wpe/wce"]**2

    if species in ["e", "electron"]:
        sname = "electron"
    else:
        sname = "H"

    tracer_dir = pic_run_dir + 'tracer/tracer1/'
    fname = tracer_dir + 'T.0/' + sname + '_tracer_qtag_sorted.h5p'
    with h5py.File(fname, 'r') as fh:
        group = fh['Step#0']
        dset = group['dX']
        nptl, = dset.shape

    fig_dir1 = create_fig_dir(pic_run, 'wpara_wperp')
    fig_dir2 = create_fig_dir(pic_run, 'wpara_dgamma')
    fig_dir3 = create_fig_dir(pic_run, 'wperp_dgamma')

    fdir = pic_run_dir + 'wpara_wperp_2nd_pass/'
    file_list = os.listdir(fdir)
    file_list.sort()

    ptl = {}
    ptl["wpara"] = np.zeros(nptl)
    ptl["wperp"] = np.zeros(nptl)
    ptl["dgamma"] = np.zeros(nptl, np.float32)
    for file_name in file_list:
        print("File name: %s" % file_name)
        fsplit = file_name.split(".")
        tindex = int(fsplit[0].split("_")[-1])
        fname = fdir + file_name
        with h5py.File(fname, 'r') as fh:
            for dset_name in ["wpara", "wperp", "dgamma"]:
                dset = fh[dset_name]
                dset.read_direct(ptl[dset_name])

            fig_name1 = set_fig_name(fig_dir1, 3, tindex, species, "wpara_wperp")
            fig_name2 = set_fig_name(fig_dir2, 3, tindex, species, "wpara_dgamma")
            fig_name3 = set_fig_name(fig_dir3, 3, tindex, species, "wperp_dgamma")
            twpe10 = tindex * vpic_info["dt*wpe"] / 0.1
            plot_four_panels(ptl["wpara"], ptl["wperp"], fig_name1, 3, twpe10, sigma_e,
                             "wpara_wperp", show_plot=False, wpara_wperp=True)
            plot_four_panels(ptl["wpara"], ptl["dgamma"], fig_name2, 3, twpe10, sigma_e,
                             "wpara_dgamma", show_plot=False)
            plot_four_panels(ptl["wperp"], ptl["dgamma"], fig_name3, 3, twpe10, sigma_e,
                             "wperp_dgamma", show_plot=False)


def plot_trajectory(plot_config, show_plot=True):
    """Plot particle trajectory
    """
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    tracer_dir = "/net/scratch3/xiaocanli/vpic-sorter/data/relativistic_turbulence/"
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    dtwpe_tracer = pic_info.dtwpe * pic_info.tracer_interval
    fname = tracer_dir + pic_run + "/electrons_ntraj500_1emax.h5p"
    fh = h5py.File(fname, 'r')
    particle_tags = list(fh.keys())
    nptl = len(particle_tags)
    print("Total number of particles: %d" % nptl)
    group = fh[particle_tags[0]]
    dset = group['dX']
    nframes, = dset.shape
    ptl = {}
    for dset_name in group:
        dset = group[dset_name]
        ptl[str(dset_name)] = np.zeros(dset.shape, dset.dtype)
        dset.read_direct(ptl[str(dset_name)])
    gamma0 = 1.0 / np.sqrt(1 + ptl["Ux"]**2 + ptl["Uy"]**2 + ptl["Uz"]**2)
    ttracer = np.arange(0, nframes) * dtwpe_tracer
    tmin, tmax = ttracer[0], ttracer[-1]

    semilogy = False

    if semilogy:
        img_dir = '../img/relativistic_turbulence/tracer_traj_log/' + pic_run + '/'
    else:
        img_dir = '../img/relativistic_turbulence/tracer_traj/' + pic_run + '/'
    mkdir_p(img_dir)
    if species in ["e", "electron"]:
        sname = "electron"
    else:
        sname = "Ion"

    for iptl in range(nptl):
    # for iptl in range(1, 2):
        print("Particle: %d of %d" % (iptl, nptl))
        group = fh[particle_tags[iptl]]
        for dset_name in group:
            dset = group[dset_name]
            dset.read_direct(ptl[str(dset_name)])
        gamma = np.sqrt(1 + ptl["Ux"]**2 + ptl["Uy"]**2 + ptl["Uz"]**2)
        dgamma = gamma - gamma0
        igamma = 1.0 / gamma
        vx = ptl["Ux"] * igamma
        vy = ptl["Uy"] * igamma
        vz = ptl["Uz"] * igamma
        x = ptl["dX"]
        y = ptl["dY"]
        z = ptl["dZ"]
        ex = ptl["Ex"]
        ey = ptl["Ey"]
        ez = ptl["Ez"]
        bx = ptl["Bx"]
        by = ptl["By"]
        bz = ptl["Bz"]
        edotb = ex*bx + ey*by + ez*bz
        ib2 = div0(1.0, bx**2 + by**2 + bz**2)
        eparax = edotb * bx * ib2
        eparay = edotb * by * ib2
        eparaz = edotb * bz * ib2
        wtot = np.cumsum(-(ex*vx + ey*vy + ez*vz)) * dtwpe_tracer
        wpara = np.cumsum(-(eparax * vx + eparay * vy + eparaz * vz)) * dtwpe_tracer
        wperp = wtot - wpara
        fig = plt.figure(figsize=[5, 3.5])
        rect = [0.14, 0.16, 0.82, 0.8]
        ax = fig.add_axes(rect)
        COLORS = palettable.tableau.Tableau_10.mpl_colors
        ax.set_prop_cycle('color', COLORS)
        if semilogy:
            ax.semilogy(ttracer, wpara, linewidth=2, label=r'$W_\parallel$')
            ax.semilogy(ttracer, wperp, linewidth=2, label=r'$W_\perp$')
            # ax.semilogy(ttracer, wpara + wperp, linewidth=2,
            #             label=r'$W_\parallel + $' + r'$W_\perp$')
            ax.semilogy(ttracer, dgamma, linewidth=2, label=r'$\Delta\gamma$')
        else:
            ax.plot(ttracer, wpara, linewidth=2, label=r'$W_\parallel$')
            ax.plot(ttracer, wperp, linewidth=2, label=r'$W_\perp$')
            # ax.plot(ttracer, wpara + wperp, linewidth=2,
            #         label=r'$W_\parallel + $' + r'$W_\perp$')
            ax.plot(ttracer, dgamma, linewidth=2, label=r'$\Delta\gamma$')
        ax.set_xlim([tmin, tmax])
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')
        ax.set_xlabel(r'$t\omega_{pe}$', fontsize=16)
        ax.set_ylabel('Energy change', fontsize=16)
        ax.tick_params(labelsize=12)
        ax.set_xlim([0, 1.58E3])
        if semilogy:
            ax.set_ylim([1E-1, 3E3])
            ax.legend(loc=4, prop={'size': 12}, ncol=1,
                     shadow=False, fancybox=False, frameon=False)
        else:
            ax.legend(loc=6, prop={'size': 12}, ncol=1,
                     shadow=False, fancybox=False, frameon=False)
        fname = img_dir + sname + "_tracer_" + str(iptl) + ".pdf"
        fig.savefig(fname)

        plt.close()
        # plt.show()

    fh.close()


def plot_absj(plot_config, show_plot=True):
    """Plot current density
    """
    tframe = plot_config["tframe"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    vpic_info = get_vpic_info(pic_run_dir)
    fields_interval = int(vpic_info["fields_interval"])
    tindex = fields_interval * tframe
    smime = math.sqrt(pic_info.mime)
    lx_de = pic_info.lx_di * smime
    lz_de = pic_info.lz_di * smime
    xmin, xmax = 0, lx_de
    zmin, zmax = -0.5 * lz_de, 0.5 * lz_de

    fname = (pic_run_dir + "hydro_hdf5/T." + str(tindex) +
             "/hydro_electron_" + str(tindex) + ".h5")
    je = {}
    with h5py.File(fname, 'r') as fh:
        group = fh["Timestep_" + str(tindex)]
        for var in ["jx", "jy", "jz"]:
            dset = group[var]
            je[var] = np.zeros(dset.shape, dtype=dset.dtype)
            dset.read_direct(je[var])

    fname = (pic_run_dir + "hydro_hdf5/T." + str(tindex) +
             "/hydro_ion_" + str(tindex) + ".h5")
    ji = {}
    with h5py.File(fname, 'r') as fh:
        group = fh["Timestep_" + str(tindex)]
        for var in ["jx", "jy", "jz"]:
            dset = group[var]
            ji[var] = np.zeros(dset.shape, dtype=dset.dtype)
            dset.read_direct(ji[var])

    absj = np.squeeze(np.sqrt((je["jx"] + ji["jx"])**2 +
                              (je["jy"] + ji["jy"])**2 +
                              (je["jz"] + ji["jz"])**2))
    fig = plt.figure(figsize=[10, 9.5])
    rect = [0.1, 0.08, 0.82, 0.86]
    ax = fig.add_axes(rect)
    im1 = ax.imshow(absj.T,
                    extent=[xmin, xmax, zmin, zmax],
                    vmin=0, vmax=5,
                    cmap=plt.cm.viridis, aspect='auto',
                    origin='lower', interpolation='bicubic')
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlabel(r'$x/d_e$', fontsize=20)
    ax.set_ylabel(r'$z/d_e$', fontsize=20)
    ax.tick_params(labelsize=16)
    rect_cbar = np.copy(rect)
    rect_cbar[0] += rect[2] + 0.01
    rect_cbar[2] = 0.02
    cbar_ax = fig.add_axes(rect_cbar)
    cbar = fig.colorbar(im1, cax=cbar_ax)
    cbar_ax.tick_params(bottom=False, top=False, left=False, right=True)
    cbar_ax.tick_params(axis='y', which='major', direction='out')
    cbar_ax.tick_params(axis='y', which='minor', direction='in', right=False)
    cbar_ax.set_title(r'$|\boldsymbol{J}|$', fontsize=20)
    cbar.ax.tick_params(labelsize=12)
    twpe = math.ceil(tindex * pic_info.dtwpe / 0.1) * 0.1
    text1 = r'$t\omega_{pe}=' + ("{%0.0f}" % twpe) + '$'
    fig.suptitle(text1, fontsize=20)
    img_dir = '../img/relativistic_turbulence/absj/' + pic_run + '/'
    mkdir_p(img_dir)
    fname = img_dir + "absj_" + str(tframe) + ".jpg"
    fig.savefig(fname, dpi=200)

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_absj_tri(plot_config, show_plot=True):
    """Plot current density
    """
    tframe = plot_config["tframe"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    tracer_dir = "/net/scratch3/xiaocanli/vpic-sorter/data/trans-relativistic/"
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    fields_interval = pic_info.fields_interval
    smime = math.sqrt(pic_info.mime)
    lx_de = pic_info.lx_di * smime
    lz_de = pic_info.lz_di * smime
    xmin, xmax = 0, lx_de
    zmin, zmax = -0.5 * lz_de, 0.5 * lz_de

    tframes = [33, 67, 139]
    fig = plt.figure(figsize=[5, 5])
    rect = [0.13, 0.7, 0.73, 0.26]
    hgap, vgap = 0.03, 0.03

    nframes = len(tframes)

    for iframe, tframe in enumerate(tframes):
        tindex = fields_interval * tframe
        fname = (pic_run_dir + "hydro_hdf5/T." + str(tindex) +
                 "/hydro_electron_" + str(tindex) + ".h5")
        je = {}
        with h5py.File(fname, 'r') as fh:
            group = fh["Timestep_" + str(tindex)]
            for var in ["jx", "jy", "jz"]:
                dset = group[var]
                je[var] = np.zeros(dset.shape, dtype=dset.dtype)
                dset.read_direct(je[var])

        fname = (pic_run_dir + "hydro_hdf5/T." + str(tindex) +
                 "/hydro_ion_" + str(tindex) + ".h5")
        ji = {}
        with h5py.File(fname, 'r') as fh:
            group = fh["Timestep_" + str(tindex)]
            for var in ["jx", "jy", "jz"]:
                dset = group[var]
                ji[var] = np.zeros(dset.shape, dtype=dset.dtype)
                dset.read_direct(ji[var])

        absj = np.squeeze(np.sqrt((je["jx"] + ji["jx"])**2 +
                                  (je["jy"] + ji["jy"])**2 +
                                  (je["jz"] + ji["jz"])**2))
        ax = fig.add_axes(rect)
        im1 = ax.imshow(absj.T,
                        extent=[xmin, xmax, zmin, zmax],
                        # vmin=0, vmax=5,
                        norm = LogNorm(vmin=0.1, vmax=10),
                        cmap=plt.cm.viridis, aspect='auto',
                        origin='lower',
                        interpolation='bicubic')
        # ax.tick_params(bottom=True, top=True, left=True, right=True)
        # ax.tick_params(axis='x', which='minor', direction='in')
        # ax.tick_params(axis='x', which='major', direction='in')
        # ax.tick_params(axis='y', which='minor', direction='in')
        # ax.tick_params(axis='y', which='major', direction='in')
        if iframe == nframes - 1:
            ax.set_xlabel(r'$x/d_e$', fontsize=16)
        else:
            ax.tick_params(axis='x', labelbottom=False)
        ax.set_ylabel(r'$z/d_e$', fontsize=16, labelpad=-7)
        twpe = math.ceil(tindex * pic_info.dtwpe / 0.1) * 0.1
        text1 = r'$t\omega_{pe}=' + ("{%0.0f}" % twpe) + '$'
        ax.text(0.03, 0.85, text1, color='w', fontsize=12,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)
        ax.tick_params(labelsize=12)
        ax.set_ylim([-500, 500])
        rect[1] -= rect[3] + vgap
    rect[1] += rect[3] + vgap
    rect_cbar = np.copy(rect)
    rect_cbar[1] += rect[3] * 0.5
    rect_cbar[3] = rect[3] * 2 + vgap * 2
    rect_cbar[0] += rect[2] + hgap
    rect_cbar[2] = 0.02
    cbar_ax = fig.add_axes(rect_cbar)
    cbar = fig.colorbar(im1, cax=cbar_ax, extend='both')
    cbar_ax.tick_params(bottom=False, top=False, left=False, right=True)
    cbar_ax.tick_params(axis='y', which='major', direction='out')
    cbar_ax.tick_params(axis='y', which='minor', direction='in', right=False)
    cbar_ax.set_title(r'$|\boldsymbol{J}|$', fontsize=16)
    cbar.ax.tick_params(labelsize=12)
    img_dir = '../img/relativistic_turbulence/absj_tri/' + pic_run + '/'
    mkdir_p(img_dir)
    fname = img_dir + "absj_tri.pdf"
    fig.savefig(fname, dpi=200)

    if show_plot:
        plt.show()
    else:
        plt.close()


def get_cmd_args():
    """Get command line arguments
    """
    default_pic_run = 'test_2d'
    default_pic_run_dir = ('/net/scratch4/xiaocanli/relativistic_turbulence/' +
                           default_pic_run + '/')
    parser = argparse.ArgumentParser(description='Analysis for relativistic turbulence runs')
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
    parser.add_argument('--plot_interval', action="store", default='100', type=int,
                        help='plot only for every plot_interval frames')
    parser.add_argument('--nsteps', action="store", default='1', type=int,
                        help='number of steps that are saved in the same file')
    parser.add_argument('--var_four', action="store", default='wpara_wperp',
                        help='variable for four-panel plot')
    parser.add_argument('--all_frames', action="store_true", default=False,
                        help='whether to analyze all frames')
    parser.add_argument('--show_plot', action="store_true", default=False,
                        help='whether to show plot')
    parser.add_argument('--econv', action="store_true", default=False,
                        help='whether to plot energy conversion')
    parser.add_argument('--plot_spect', action="store_true", default=False,
                        help='whether to plot particle energy spectrum')
    parser.add_argument('--check_density', action="store_true", default=False,
                        help='whether to check maximum density')
    parser.add_argument('--wpara_wperp_four_h5', action="store_true", default=False,
                        help='whether to compare energization due to parallel' +
                        ' and perpendicular electric field for 4 different cases')
    parser.add_argument('--wpara_wperp', action="store_true", default=False,
                        help='whether to calculate energization due to parallel' +
                        ' and perpendicular electric field when particle energy'
                        ' reaches sigma_e/2')
    parser.add_argument('--plot_wpara_wperp', action="store_true", default=False,
                        help='whether to plot energization due to parallel' +
                        ' and perpendicular electric field when particle energy'
                        ' reaches sigma_e/2')
    parser.add_argument('--comp_spect', action="store_true", default=False,
                        help='whether to compare spectrum for different injection')
    parser.add_argument('--spect_species', action="store_true", default=False,
                        help='energy spectrum for different species')
    parser.add_argument('--egain_post', action="store_true", default=False,
                        help='energy gain after injection')
    parser.add_argument('--calc_wpp_1st', action="store_true", default=False,
                        help='whether to calculate wpara and wperp for ' +
                        'a large number of time steps (first pass)')
    parser.add_argument('--calc_wpp_2nd', action="store_true", default=False,
                        help='whether to calculate wpara and wperp for ' +
                        'a large number of time steps (second pass)')
    parser.add_argument('--wpp_hdf5', action="store_true", default=False,
                        help='whether to wpara and wperp are saved in a HDF5 file')
    parser.add_argument('--plot_traj', action="store_true", default=False,
                        help='whether to plot tracer particle trajectory')
    parser.add_argument('--plot_absj', action="store_true", default=False,
                        help='whether to plot current density')
    parser.add_argument('--plot_absj_tri', action="store_true", default=False,
                        help='whether to plot three frames of current density')
    return parser.parse_args()


def analysis_single_frames(plot_config, args):
    """Analysis for multiple time frames
    """
    tframe = args.tframe
    if args.econv:
        energy_conversion(plot_config)
    elif args.plot_spect:
        if args.all_frames:
            plot_spectrum_multi(plot_config)
    elif args.check_density:
        check_density(plot_config)
    elif args.wpara_wperp_four_h5:
        compare_energization_four_h5(plot_config)
    elif args.wpara_wperp:
        calc_wpara_wperp(plot_config)
    elif args.calc_wpp_1st:
        calc_wpara_wperp_1st(plot_config)
    if args.calc_wpp_2nd:
        calc_wpara_wperp_2nd(plot_config)
    elif args.plot_wpara_wperp:
        plot_wpara_wperp(plot_config)
    elif args.comp_spect:
        compare_spectrum(plot_config)
    elif args.spect_species:
        plot_spect_species(plot_config, args.show_plot)
    elif args.egain_post:
        egain_after_injection(plot_config, args.show_plot)
    elif args.plot_traj:
        plot_trajectory(plot_config, args.show_plot)
    elif args.plot_absj:
        plot_absj(plot_config, args.show_plot)
    elif args.plot_absj_tri:
        plot_absj_tri(plot_config, args.show_plot)


def process_input(plot_config, args, tframe):
    """process one time frame"""
    plot_config["tframe"] = tframe
    if args.calc_wpp_1st:
        calc_wpara_wperp_1st(plot_config)
    elif args.plot_absj:
        plot_absj(plot_config, show_plot=False)


def analysis_multi_frames(plot_config, args):
    """Analysis for multiple time frames
    """
    tframes = range(plot_config["tstart"], plot_config["tend"] + 1)
    nframes = len(tframes)
    if args.time_loop:
        for tframe in tframes:
            print("Time frame: %d" % tframe)
            plot_config["tframe"] = tframe
            if args.plot_absj:
                plot_absj(plot_config, show_plot=False)
    else:
        ncores = multiprocessing.cpu_count()
        ncores = 36
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
    plot_config["plot_interval"] = args.plot_interval
    plot_config["nsteps"] = args.nsteps
    plot_config["species"] = args.species
    plot_config["var_four"] = args.var_four
    plot_config["wpp_hdf5"] = args.wpp_hdf5
    if args.multi_frames:
        analysis_multi_frames(plot_config, args)
    else:
        analysis_single_frames(plot_config, args)


if __name__ == "__main__":
    main()
