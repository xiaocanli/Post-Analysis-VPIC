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


def plot_spectrum_multi(plot_config):
    """Plot spectrum for multiple time frames

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    bg = plot_config["bg"]
    root_dir = "/net/scratch3/xiaocanli/reconnection/Cori_runs/"
    pic_runs = ["2D-Lx150-bg" + str(bg) + "-150ppc-16KNL"]
    pic_runs.append("3D-Lx150-bg" + str(bg) + "-150ppc-2048KNL")
    tstarts = [[0, 12], [0, 20]]
    tends = [[12, 40], [20, 40]]
    cbar_ticks = [[np.linspace(tstarts[0][0], tends[0][0], 4),
                   np.linspace(tstarts[0][1], tends[0][1], 5)],
                  [np.linspace(tstarts[1][0], tends[1][0], 5),
                   np.linspace(tstarts[1][1], tends[1][1], 5)]]
    pindex = [[-3.6, -6.0], [-4.0, -4.6]]
    pnorm = [[1E11, 2E16], [4E15, 5E16]]
    plow = [[538, 588], [558, 568]]
    phigh = [[638, 688], [658, 668]]
    fig = plt.figure(figsize=[7, 5])
    rect0 = [0.11, 0.54, 0.41, 0.4]
    hgap, vgap = 0.04, 0.03
    rect_cbar0 = [0.14, 0.63, 0.16, 0.02]
    emax = 1E3 if species == "e" else 2E3
    colormap = plt.cm.Spectral_r
    for irun, pic_run in enumerate(pic_runs):
        pic_run_dir = root_dir + pic_run + "/"
        rect = np.copy(rect0)
        rect[0] += irun * (rect[2] + hgap)
        ax1 = fig.add_axes(rect)
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        if species == 'e':
            vth = pic_info.vthe
        else:
            vth = pic_info.vthi
        gama = 1.0 / math.sqrt(1.0 - 3 * vth**2)
        eth = gama - 1.0
        ebins = np.logspace(-6, 4, 1000)
        ebins /= eth

        tinterval = pic_info.particle_interval
        dtf = math.ceil(pic_info.dt_fields / 0.1) * 0.1
        tstart, tend = tstarts[irun][0], tends[irun][0]
        ntp =  tend - tstart + 1
        for tframe in range(tstart, tend):
            tindex = tinterval * tframe
            fname = (pic_run_dir + "spectrum_combined/spectrum_" +
                     species + "_" + str(tindex) + ".dat")
            spect = np.fromfile(fname, dtype=np.float32)
            ndata, = spect.shape
            spect[3:] /= np.gradient(ebins)
            color = colormap((tframe - tstart)/float(ntp), 1)
            ax1.loglog(ebins, spect[3:], linewidth=1, color=color)

        if species == 'e':
            fpower = pnorm[irun][0] * ebins**pindex[irun][0]
            power_index = "{%0.1f}" % pindex[irun][0]
            pname = r'$\propto \varepsilon^{' + power_index + '}$'
            ax1.loglog(ebins[plow[irun][0]:phigh[irun][0]],
                       fpower[plow[irun][0]:phigh[irun][0]],
                       linewidth=1, color='k', label=pname)
            ax1.text(0.9, 0.67, pname, color='k', fontsize=12,
                     bbox=dict(facecolor='none', alpha=1.0,
                               edgecolor='none', pad=10.0),
                     horizontalalignment='right', verticalalignment='center',
                     transform=ax1.transAxes)

        ax1.tick_params(bottom=True, top=True, left=True, right=True)
        ax1.tick_params(axis='x', which='minor', direction='in', top='on')
        ax1.tick_params(axis='x', which='major', direction='in')
        ax1.tick_params(axis='y', which='minor', direction='in')
        ax1.tick_params(axis='y', which='major', direction='in')
        ax1.set_xlim([1E-1, emax])
        if irun == 0:
            ax1.set_ylim([1E-1, 1E9])
        else:
            ax1.set_ylim([1E-1, 1E13])
        ax1.tick_params(axis='x', labelbottom='off')
        if irun == 0:
            ax1.set_ylabel(r'$f(\varepsilon)$', fontsize=12)
        else:
            ax1.tick_params(axis='y', labelleft='off')
        ax1.tick_params(labelsize=10)
        rect_cbar = np.copy(rect_cbar0)
        rect_cbar[0] += (rect[2] + hgap) * irun
        cax = fig.add_axes(rect_cbar)
        sm = plt.cm.ScalarMappable(cmap=colormap,
                                   norm=plt.Normalize(vmin=tstart * dtf,
                                                      vmax=tend * dtf))
        cax.tick_params(axis='x', which='major', direction='in')
        # fake up the array of the scalar mappable. Urgh...
        sm._A = []
        cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
        cbar.set_label(r'$t\Omega_{ci}$', fontsize=10)
        cbar.set_ticks(cbar_ticks[irun][0] * dtf)
        cbar.ax.tick_params(labelsize=8)

        pdim = "2D" if "2D" in pic_run else "3D"
        ax1.set_title(pdim, color='k', fontsize=16)
        # ax1.text(-0.25, 0.5, pdim, color='k',
        #          fontsize=16, rotation='vertical',
        #          bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        #          horizontalalignment='center', verticalalignment='center',
        #          transform=ax1.transAxes)

        rect[1] -= rect[3] + vgap
        ax2 = fig.add_axes(rect)

        tinterval = pic_info.particle_interval
        tstart, tend = tstarts[irun][1], tends[irun][1]
        ntp =  tend - tstart + 1
        for tframe in range(tstart, tend):
            tindex = tinterval * tframe
            fname = (pic_run_dir + "spectrum_combined/spectrum_" +
                     species + "_" + str(tindex) + ".dat")
            spect = np.fromfile(fname, dtype=np.float32)
            ndata, = spect.shape
            spect[3:] /= np.gradient(ebins)
            color = colormap((tframe - tstart)/float(ntp), 1)
            ax2.loglog(ebins, spect[3:], linewidth=1, color=color)

        if species == 'e':
            fpower = pnorm[irun][1] * ebins**pindex[irun][1]
            power_index = "{%0.1f}" % pindex[irun][1]
            pname = r'$\propto \varepsilon^{' + power_index + '}$'
            ax2.loglog(ebins[plow[irun][1]:phigh[irun][1]],
                       fpower[plow[irun][1]:phigh[irun][1]],
                       linewidth=1, color='k', label=pname)
            ax2.text(0.9, 0.67, pname, color='k', fontsize=12,
                     bbox=dict(facecolor='none', alpha=1.0,
                               edgecolor='none', pad=10.0),
                     horizontalalignment='right', verticalalignment='center',
                     transform=ax2.transAxes)

        ax2.set_xlim([1E-1, emax])
        if irun == 0:
            ax2.set_ylim([1E-1, 1E9])
        else:
            ax2.set_ylim([1E-1, 1E13])
        ax2.tick_params(bottom=True, top=True, left=True, right=True)
        ax2.tick_params(axis='x', which='minor', direction='in', top='on')
        ax2.tick_params(axis='x', which='major', direction='in')
        ax2.tick_params(axis='y', which='minor', direction='in', left='on')
        ax2.tick_params(axis='y', which='major', direction='in')

        if irun == 0:
            ax2.set_ylabel(r'$f(\varepsilon)$', fontsize=12)
        else:
            ax2.tick_params(axis='y', labelleft='off')
        ax2.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$', fontsize=12)
        ax2.tick_params(labelsize=10)
        rect_cbar[1] -= rect[3] + vgap
        cax = fig.add_axes(rect_cbar)
        cax.tick_params(axis='x', which='major', direction='in')
        sm = plt.cm.ScalarMappable(cmap=colormap,
                                   norm=plt.Normalize(vmin=tstart * dtf,
                                                      vmax=tend * dtf))
        # fake up the array of the scalar mappable. Urgh...
        sm._A = []
        cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
        cbar.set_label(r'$t\Omega_{ci}$', fontsize=10)
        cbar.set_ticks(cbar_ticks[irun][1] * dtf)
        cbar.ax.tick_params(labelsize=8)

    fdir = '../img/cori_3d/spectrum/'
    mkdir_p(fdir)
    fname = fdir + species + 'spect_bg' + str(int(bg*10)).zfill(2) + '.pdf'
    fig.savefig(fname)

    plt.show()


def plot_spectrum_single(plot_config, show_plot=True):
    """Plot spectrum for each time frame

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    tstart = plot_config["tstart"]
    tend = plot_config["tend"]
    root_dir = "/net/scratch3/xiaocanli/reconnection/Cori_runs/"
    pic_runs = ["2D-Lx150-bg" + str(plot_config["bg"]) + "-150ppc-16KNL"]
    pic_runs.append("3D-Lx150-bg" + str(plot_config["bg"]) + "-150ppc-2048KNL")
    pic_infos = []
    fnorm = []
    for irun, pic_run in enumerate(pic_runs):
        pic_run_dir = root_dir + pic_run + "/"
        species = plot_config["species"]
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        pic_infos.append(pic_info)
        fnorm.append(pic_info.nppc * pic_info.nx * pic_info.ny * pic_info.nz)

    fnorm = np.asarray(fnorm)
    fnorm /= np.min(fnorm)

    for tframe in range(tstart, tend + 1):
        print("Time frame: %d" % tframe)
        fig = plt.figure(figsize=[7, 5])
        w1, h1 = 0.83, 0.8
        xs, ys = 0.96 - w1, 0.96 - h1
        ax = fig.add_axes([xs, ys, w1, h1])
        for irun, pic_run in enumerate(pic_runs):
            pic_run_dir = root_dir + pic_run + "/"
            if species == 'e':
                vth = pic_infos[irun].vthe
            else:
                vth = pic_infos[irun].vthi
            gama = 1.0 / math.sqrt(1.0 - 3 * vth**2)
            eth = gama - 1.0
            ebins = np.logspace(-6, 4, 1000)
            ebins /= eth
            dt_particles = pic_infos[irun].dt_particles  # in 1/wci

            tindex = pic_infos[irun].particle_interval * tframe
            fname = (pic_run_dir + "spectrum_combined/spectrum_" +
                     species + "_" + str(tindex) + ".dat")
            spect = np.fromfile(fname, dtype=np.float32)
            ndata, = spect.shape
            spect[3:] /= np.gradient(ebins)
            label = '2D' if '2D' in pic_run else '3D'
            ax.loglog(ebins, spect[3:] / fnorm[irun], linewidth=2,
                      color=COLORS[irun], label=label)
        ax.legend(loc=3, prop={'size': 16}, ncol=1,
                  shadow=False, fancybox=False, frameon=False)
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in', top='on')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in', left='on')
        ax.tick_params(axis='y', which='major', direction='in')
        ax.set_xlim([1E-1, 2E3])
        ax.set_ylim([1E-4, 1E9])
        ax.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$',
                      fontsize=20)
        ax.set_ylabel(r'$f(\varepsilon)$', fontsize=20)
        ax.set_yticks(np.logspace(-3, 9, num=7))
        ax.tick_params(labelsize=16)
        dtp = math.ceil(dt_particles * tframe / 0.1) * 0.1
        text = str(dtp) + r'$\Omega_{ci}^{-1}$'
        ax.text(0.98, 0.9, text, color='k', fontsize=20,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='right', verticalalignment='center',
                transform=ax.transAxes)
        fdir = '../img/cori_3d/spectrum/compare_2d_3d/'
        fdir += 'Bg' + str(plot_config["bg"]) + '/'
        mkdir_p(fdir)
        fname = fdir + "spect2d_" + species + "_" + str(tframe) + ".pdf"
        fig.savefig(fname)
        if show_plot:
            plt.show()
        else:
            plt.close()


def plot_spectrum_pub(plot_config, show_plot=True):
    """Plot energy spectrum for all time frames for publication
    """
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    root_dir = "/net/scratch3/xiaocanli/reconnection/Cori_runs/"
    bg = plot_config["bg"]
    bg_str = str(int(bg * 10)).zfill(2)
    pic_runs = ["2D-Lx150-bg" + str(bg) + "-150ppc-16KNL"]
    pic_runs.append("3D-Lx150-bg" + str(bg) + "-150ppc-2048KNL")
    species = plot_config["species"]
    tframe = plot_config["tframe"]
    tstart = plot_config["tstart"]
    tend = plot_config["tend"]
    if species == 'e':
        vth = 0.1
        sname = 'electron'
    else:
        vth = 0.02
        sname = 'ion'
    gama = 1.0 / math.sqrt(1.0 - 3 * vth**2)
    eth = gama - 1.0
    emin, emax = 1E-6, 1E4
    nband = 1000
    dloge = (math.log10(emax) - math.log10(emin)) / (nband - 1)
    emin0 = 10**(math.log10(emin) - dloge)
    ebins = np.logspace(math.log10(emin0), math.log10(emax), nband+1)
    pbins = np.sqrt((ebins + 1)**2 - 1)
    pbins /= np.sqrt((eth + 1)**2 - 1)
    pbins_mid = (pbins[:-1] + pbins[1:]) * 0.5
    dpbins = np.diff(pbins)
    ebins /= eth
    ebins_mid = (ebins[:-1] + ebins[1:]) * 0.5
    debins = np.diff(ebins)

    # tstarts = [[0, 12], [0, 20]]
    # tends = [[12, 40], [20, 40]]
    # angle = [[55, 60], [50, 45]]
    # ypos = [[0.55, 0.61], [0.71, 0.7]]
    # norms = [[2, 1], [2, 1]]
    # pene1 = [[20, 25], [25, 25]]
    # pene2 = [[200, 250], [250, 250]]
    # cbar_ticks = [[np.linspace(tstarts[0][0], tends[0][0], 3),
    #                np.linspace(tstarts[0][1], tends[0][1], 5)],
    #               [np.linspace(tstarts[1][0], tends[1][0], 3),
    #                np.linspace(tstarts[1][1], tends[1][1], 5)]]
    power_indices = np.zeros((2, tend - tstart + 1))

    # Energy indices to get power-law spectra
    eindices = np.zeros((2, 2), dtype=np.int)
    eindices[0, 0], ene = find_nearest(ebins_mid, 30)
    eindices[0, 1], ene = find_nearest(ebins_mid, 100)
    eindices[1, 0], ene = find_nearest(ebins_mid, 30)
    eindices[1, 1], ene = find_nearest(ebins_mid, 100)
    tstart_power = 6  # frame to start fitting power-laws
    ts_spect, te_spect = 1, 20

    fig = plt.figure(figsize=[3.5, 2.5])
    rect = [0.16, 0.16, 0.69, 0.8]
    ax = fig.add_axes(rect)

    for irun, pic_run in enumerate(pic_runs):
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        for tframe in range(tstart, tend + 1):
            print("Time frame: %d" % tframe)
            pic_run_dir = root_dir + pic_run + "/"
            tindex = pic_info.fields_interval * tframe
            fname = (pic_run_dir + "spectrum_combined/spectrum_" +
                     species + "_" + str(tindex) + ".dat")
            # fname = (pic_run_dir + "spectrum_reconnection_layer/"
            #          "spectrum_layer_" + sname + "_" + str(tindex) + ".dat")
            spect = np.fromfile(fname, dtype=np.float32)
            espect = spect[3:] / debins
            if irun == 1 and tframe >= ts_spect and tframe <= te_spect:
                nframes = te_spect - ts_spect + 1
                color = plt.cm.tab20c_r((tframe - ts_spect + 0.5)/float(nframes), 1)
                ax.loglog(ebins_mid, espect, linewidth=1, color=color)
            # Power-law fitting
            if tframe >= tstart_power:
                ein1, ein2 = eindices[irun, 0], eindices[irun, 1]
                popt, pcov = curve_fit(fitting_funcs.func_line,
                                       np.log10(ebins_mid[ein1:ein2]),
                                       np.log10(espect[ein1:ein2]))
                fpower = fitting_funcs.func_line(np.log10(ebins_mid), popt[0], popt[1])
                fpower = 10**fpower
                power_indices[irun, tframe - tstart] = popt[0]
                # Plot fitted power-law for one time frame
                if irun == 1 and tframe == te_spect and species == 'e':
                    pindex1, ene = find_nearest(ebins_mid, 50)
                    pnorm = espect[pindex1] / fpower[pindex1] * 3
                    fpower *= pnorm
                    pindex1, ene = find_nearest(ebins_mid, 25)
                    pindex2, ene = find_nearest(ebins_mid, 250)
                    ax.loglog(ebins_mid[pindex1:pindex2], fpower[pindex1:pindex2],
                              linewidth=0.5, linestyle='--', color='k')
                    power_index = "{%0.1f}" % popt[0]
                    pname = r'$\propto \varepsilon^{' + power_index + '}$'
                    ax.text(0.68, 0.65, pname, color='k', fontsize=10, rotation=-50,
                            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                            horizontalalignment='left', verticalalignment='center',
                            transform=ax.transAxes)
    ax.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$', fontsize=10)
    ax.set_ylabel(r'$f(\varepsilon)$', fontsize=10)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top=False)
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlim([1E-1, 1E3])
    ax.set_ylim([1E0, 2E12])
    ax.tick_params(labelsize=8)
    ax.set_yticks(np.logspace(0, 12, num=4))

    # Plot power-law indices
    rect1 = [0.23, 0.32, 0.30, 0.3]
    ax1 = fig.add_axes(rect1)
    ax1.tick_params(bottom=True, top=True, left=True, right=True)
    ax1.tick_params(axis='x', which='minor', direction='in', top=False)
    ax1.tick_params(axis='x', which='major', direction='in')
    ax1.tick_params(axis='y', which='minor', direction='in')
    ax1.tick_params(axis='y', which='major', direction='in')
    twci = np.arange(tend - tstart + 1) * pic_info.dt_fields
    ax1.plot(twci[tstart_power:], -power_indices[0, tstart_power:],
             color='k', linestyle='--', linewidth=1)
    ax1.plot(twci[tstart_power:], -power_indices[1, tstart_power:],
             color='k', linestyle='-', linewidth=1)
    ax1.set_ylabel(r'$p$', fontsize=10, labelpad=-2)
    ax1.set_xlabel(r'$t\Omega_{ci}$', fontsize=10)
    ax1.set_xlim([0, 400])
    ax1.tick_params(labelsize=8)
    ax1.text(0.5, 0.76, '2D', color='k', fontsize=10,
             bbox=dict(facecolor='none', alpha=1.0,
                       edgecolor='none', pad=10.0),
             horizontalalignment='right', verticalalignment='center',
             transform=ax1.transAxes)
    ax1.text(0.9, 0.56, '3D', color='k', fontsize=10,
             bbox=dict(facecolor='none', alpha=1.0,
                       edgecolor='none', pad=10.0),
             horizontalalignment='right', verticalalignment='center',
             transform=ax1.transAxes)

    # Colorbar
    rect_cbar = np.copy(rect)
    rect_cbar[0] = rect[0] + rect[2] + 0.01
    rect_cbar[2] = 0.02
    cax = fig.add_axes(rect_cbar)
    dtf = math.ceil(pic_info.dt_fields / 0.1) * 0.1
    sm = plt.cm.ScalarMappable(cmap=plt.cm.tab20c_r,
                               norm=plt.Normalize(vmin=0.5*dtf,
                                                  vmax=(te_spect + 0.5) * dtf))
    cax.tick_params(axis='y', which='major', direction='in')
    sm._A = []
    cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
    cbar.set_label(r'$t\Omega_{ci}$', fontsize=10)
    ticks = np.linspace(0, te_spect, 6) * dtf
    ticks = np.concatenate(([10], ticks))
    cbar.set_ticks(ticks)
    cax.tick_params(labelrotation=90)
    # cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(),
    #                         rotation='vertical')
    cbar.ax.tick_params(labelsize=8)
    cax.tick_params(axis='y', which='major', direction='out')
    cax.yaxis.set_ticks_position('right')
    cax.yaxis.set_label_position('right')

    fdir = '../img/cori_3d/espect/'
    mkdir_p(fdir)
    fname = fdir + "espect_bg" + bg_str + '_' + species + ".pdf"
    fig.savefig(fname)
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_spectrum_pub_23(plot_config, show_plot=True):
    """Plot energy spectrum for all time frames for publication
    """
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    root_dir = "/net/scratch3/xiaocanli/reconnection/Cori_runs/"
    bg = plot_config["bg"]
    bg_str = str(int(bg * 10)).zfill(2)
    pic_runs = ["2D-Lx150-bg" + str(bg) + "-150ppc-16KNL"]
    pic_runs.append("3D-Lx150-bg" + str(bg) + "-150ppc-2048KNL")
    species = plot_config["species"]
    tframe = plot_config["tframe"]
    tstart = plot_config["tstart"]
    tend = plot_config["tend"]
    if species == 'e':
        vth = 0.1
        sname = 'electron'
    else:
        vth = 0.02
        sname = 'ion'
    gama = 1.0 / math.sqrt(1.0 - 3 * vth**2)
    eth = gama - 1.0
    emin, emax = 1E-6, 1E4
    nband = 1000
    dloge = (math.log10(emax) - math.log10(emin)) / (nband - 1)
    emin0 = 10**(math.log10(emin) - dloge)
    ebins = np.logspace(math.log10(emin0), math.log10(emax), nband+1)
    pbins = np.sqrt((ebins + 1)**2 - 1)
    pbins /= np.sqrt((eth + 1)**2 - 1)
    pbins_mid = (pbins[:-1] + pbins[1:]) * 0.5
    dpbins = np.diff(pbins)
    ebins /= eth
    ebins_mid = (ebins[:-1] + ebins[1:]) * 0.5
    debins = np.diff(ebins)

    power_indices = np.zeros((2, tend - tstart + 1))
    tframes_23 = [10, 20, 40]
    lstyles = ['--', '-', ':']

    # Energy indices to get power-law spectra
    eindices = np.zeros((2, 2), dtype=np.int)
    eindices[0, 0], ene = find_nearest(ebins_mid, 30)
    eindices[0, 1], ene = find_nearest(ebins_mid, 100)
    eindices[1, 0], ene = find_nearest(ebins_mid, 30)
    eindices[1, 1], ene = find_nearest(ebins_mid, 100)
    tstart_power = 6  # frame to start fitting power-laws
    ts_spect, te_spect = 1, 20

    fig = plt.figure(figsize=[3.5, 2.5])
    rect = [0.16, 0.16, 0.69, 0.8]
    ax = fig.add_axes(rect)
    rect1 = [0.25, 0.25, 0.35, 0.35]
    ax1 = fig.add_axes(rect1)

    picinfo_fname = '../data/pic_info/pic_info_' + pic_runs[1] + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    fnorm = pic_info.ny

    COLORS = palettable.tableau.Tableau_10.mpl_colors
    for irun, pic_run in enumerate(pic_runs):
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        iframe = 0
        for tframe in range(tstart, tend + 1):
            print("Time frame: %d" % tframe)
            pic_run_dir = root_dir + pic_run + "/"
            tindex = pic_info.fields_interval * tframe
            fname = (pic_run_dir + "spectrum_combined/spectrum_" +
                     species + "_" + str(tindex) + ".dat")
            # fname = (pic_run_dir + "spectrum_reconnection_layer/"
            #          "spectrum_layer_" + sname + "_" + str(tindex) + ".dat")
            spect = np.fromfile(fname, dtype=np.float32)
            espect = spect[3:] / debins
            if irun == 1 and tframe >= ts_spect and tframe <= te_spect:
                nframes = te_spect - ts_spect + 1
                # color = plt.cm.tab20c_r((tframe - ts_spect + 0.5)/float(nframes), 1)
                color = plt.cm.plasma_r((tframe - ts_spect + 0.5)/float(nframes), 1)
                ax.loglog(ebins_mid, espect, linewidth=0.5, color=color)
            # Power-law fitting
            if tframe >= tstart_power:
                ein1, ein2 = eindices[irun, 0], eindices[irun, 1]
                popt, pcov = curve_fit(fitting_funcs.func_line,
                                       np.log10(ebins_mid[ein1:ein2]),
                                       np.log10(espect[ein1:ein2]))
                fpower = fitting_funcs.func_line(np.log10(ebins_mid), popt[0], popt[1])
                fpower = 10**fpower
                power_indices[irun, tframe - tstart] = popt[0]
                # Plot fitted power-law for one time frame
                if irun == 1 and tframe == te_spect and species == 'e':
                    pindex1, ene = find_nearest(ebins_mid, 50)
                    pnorm = espect[pindex1] / fpower[pindex1] * 3
                    fpower *= pnorm
                    pindex1, ene = find_nearest(ebins_mid, 25)
                    pindex2, ene = find_nearest(ebins_mid, 250)
                    ax.loglog(ebins_mid[pindex1:pindex2], fpower[pindex1:pindex2],
                              linewidth=0.5, linestyle='-', color='k')
                    power_index = "{%0.1f}" % popt[0]
                    pname = r'$\propto \varepsilon^{' + power_index + '}$'
                    ax.text(0.68, 0.65, pname, color='k', fontsize=10, rotation=-50,
                            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                            horizontalalignment='left', verticalalignment='center',
                            transform=ax.transAxes)
            if tframe in tframes_23:
                lstyle = '-' if irun == 1 else ':'
                espect[espect == 0] = np.nan
                ax1.loglog(ebins_mid, espect, linewidth=0.5, color=COLORS[iframe],
                           linestyle=lstyle)
                iframe += 1
    ax.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$', fontsize=10)
    ax.set_ylabel(r'$f(\varepsilon)$', fontsize=10)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top=False)
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlim([1E-1, 1E3])
    ax.set_ylim([1E0, 2E12])
    ax.tick_params(labelsize=8)
    ax.set_yticks(np.logspace(0, 12, num=4))

    # for i in range(3):
    #     text1 = r"$t\Omega_{ci}=" + str(tframes_23[i]) + "$"
    #     ax1.loglog([0, 0], [1, 2], linewidth=0.5, color='k',
    #                linestyle=lstyles[i], label=text1)
    # ax1.legend(loc=3, prop={'size': 6}, ncol=1,
    #            shadow=False, fancybox=False, frameon=False)
    ax1.tick_params(bottom=True, top=True, left=True, right=True)
    ax1.tick_params(axis='x', which='minor', direction='in', top=False)
    ax1.tick_params(axis='x', which='major', direction='in')
    ax1.tick_params(axis='y', which='minor', direction='in')
    ax1.tick_params(axis='y', which='major', direction='in')
    ax1.set_xlim([1E0, 1E3])
    ax1.set_ylim([1E-1, 2E12])
    ax1.tick_params(labelsize=6)
    ax1.set_yticks(np.logspace(0, 12, num=4))
    ax1.text(0.03, 0.86, '3D', color='k', fontsize=6,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='left', verticalalignment='top',
             transform=ax1.transAxes)
    ax1.text(0.03, 0.63, '2D', color='k', fontsize=6,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='left', verticalalignment='top',
             transform=ax1.transAxes)
    ypos0 = 0.8
    ax1.text(0.53, ypos0, '$t\Omega_{ci}=$', color='k', fontsize=6,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='left', verticalalignment='center',
             transform=ax1.transAxes)
    dtf = math.ceil(pic_info.dt_fields / 0.1) * 0.1
    for iframe, tframe in enumerate(tframes_23):
        ypos = ypos0 + (1 - iframe) * 0.12
        text1 = r"$" + str(int(tframe*dtf)) + "$"
        ax1.text(0.83, ypos, text1, color=COLORS[iframe], fontsize=6,
                 bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                 horizontalalignment='left', verticalalignment='center',
                 transform=ax1.transAxes)

    # Colorbar
    rect_cbar = np.copy(rect)
    rect_cbar[0] = rect[0] + rect[2] + 0.01
    rect_cbar[2] = 0.02
    cax = fig.add_axes(rect_cbar)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma_r,
                               norm=plt.Normalize(vmin=0.5*dtf,
                                                  vmax=(te_spect + 0.5) * dtf))
    cax.tick_params(axis='y', which='major', direction='in')
    sm._A = []
    cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
    cbar.set_label(r'$t\Omega_{ci}$', fontsize=10)
    ticks = np.linspace(0, te_spect, 6) * dtf
    ticks = np.concatenate(([10], ticks))
    cbar.set_ticks(ticks)
    cax.tick_params(labelrotation=90)
    # cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(),
    #                         rotation='vertical')
    cbar.ax.tick_params(labelsize=8)
    cax.tick_params(axis='y', which='major', direction='out')
    cax.yaxis.set_ticks_position('right')
    cax.yaxis.set_label_position('right')

    fdir = '../img/cori_3d/espect/'
    mkdir_p(fdir)
    fname = fdir + "espect_bg" + bg_str + '_' + species + "_23.pdf"
    fig.savefig(fname)
    plt.show()


def plot_spectrum(plot_config):
    """Plot spectrum for all time frames for a single run

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    tstart = plot_config["tstart"]
    tend = plot_config["tend"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    species = plot_config["species"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    if species == 'e':
        vth = pic_info.vthe
    else:
        vth = pic_info.vthi
    # gama = 1.0 / math.sqrt(1.0 - 3 * vth**2)
    gama = math.sqrt(1.0 + 3 * vth**2)
    eth = gama - 1.0
    ebins = np.logspace(-6, 4, 1000)
    ebins /= eth
    dt_particles = pic_info.dt_particles  # in 1/wci
    nframes = tend - tstart + 1
    nplots = 4
    nframes_plot = nframes // nplots
    dtf = math.ceil(pic_info.dt_particles / 0.1) * 0.1

    for iplot in range(nplots):
        fig = plt.figure(figsize=[7, 5])
        rect = [0.13, 0.16, 0.7, 0.8]
        ax = fig.add_axes(rect)
        tstart_plot = nframes_plot * iplot
        if iplot == nplots - 1:
            tend_plot = tend
        else:
            tend_plot = tstart_plot + nframes_plot
        for tframe in range(tstart_plot, tend_plot + 1):
            print("Time frame: %d" % tframe)
            tindex = pic_info.particle_interval * tframe
            fname = (pic_run_dir + "spectrum_combined/spectrum_" +
                     species + "_" + str(tindex) + ".dat")
            spect = np.fromfile(fname, dtype=np.float32)
            ndata, = spect.shape
            spect[3:] /= np.gradient(ebins)
            ax.loglog(ebins, spect[3:], linewidth=2,
                      color = plt.cm.Spectral_r((tframe - tstart_plot)/float(nframes_plot), 1))
        ax.grid(True)
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in', top='on')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in', left='on')
        ax.tick_params(axis='y', which='major', direction='in')
        if species in ['e', 'electron']:
            ax.set_xlim([1E-1, 2E3])
        else:
            ax.set_xlim([1E-1, 2E3])
        if '3D' in pic_run:
            ax.set_ylim([1E-2, 1E12])
            ax.set_yticks(np.logspace(-1, 11, num=7))
        else:
            ax.set_ylim([1E-1, 1E9])
            ax.set_yticks(np.logspace(-1, 9, num=6))
        ax.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$',
                      fontsize=20)
        ax.set_ylabel(r'$f(\varepsilon)$', fontsize=20)
        ax.tick_params(labelsize=16)
        rect_cbar = np.copy(rect)
        rect_cbar[0] += rect[2] + 0.02
        rect_cbar[2] = 0.03
        cax = fig.add_axes(rect_cbar)
        colormap = plt.cm.get_cmap('jet', tend_plot - tstart_plot + 1)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Spectral_r,
                                   norm=plt.Normalize(vmin=tstart_plot * dtf,
                                                      vmax=tend_plot * dtf))
        cax.tick_params(axis='x', which='major', direction='in')
        # fake up the array of the scalar mappable. Urgh...
        sm._A = []
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label(r'$t\Omega_{ci}$', fontsize=16)
        cbar.ax.tick_params(labelsize=12)
        fdir = '../img/cori_3d/spectrum/' + pic_run + '/'
        mkdir_p(fdir)
        fname = fdir + 'spectrum_' + species + '_' + str(iplot) + '.pdf'
        fig.savefig(fname)
    plt.show()


def spectrum_reconnection_layer(plot_config):
    """Plot spectra in the reconnection layer for all time frames for a single run

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    tstart = plot_config["tstart"]
    tend = plot_config["tend"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    species = plot_config["species"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    if species in ['e', 'electron']:
        vth = pic_info.vthe
        sname = 'electron'
    else:
        vth = pic_info.vthi
        sname = 'ion'
    gama = 1.0 / math.sqrt(1.0 - 3 * vth**2)
    eth = gama - 1.0
    ebins = np.logspace(-6, 4, 1000)
    ebins /= eth
    dt_particles = pic_info.dt_particles  # in 1/wci
    nframes = tend - tstart + 1
    nplots = 4
    nframes_plot = nframes // nplots
    dtf = math.ceil(pic_info.dt_particles / 0.1) * 0.1

    for iplot in range(nplots):
        fig = plt.figure(figsize=[7, 5])
        rect = [0.13, 0.16, 0.7, 0.8]
        ax = fig.add_axes(rect)
        tstart_plot = nframes_plot * iplot + 1
        if iplot == nplots - 1:
            tend_plot = tend
        else:
            tend_plot = tstart_plot + nframes_plot
        for tframe in range(tstart_plot, tend_plot + 1):
            print("Time frame: %d" % tframe)
            tindex = pic_info.particle_interval * tframe
            fname = (pic_run_dir + "spectrum_reconnection_layer/spectrum_layer_" +
                     sname + "_" + str(tindex) + ".dat")
            spect = np.fromfile(fname, dtype=np.float32)
            ndata, = spect.shape
            spect[3:] /= np.gradient(ebins)
            if tframe > 1:
                tindex = pic_info.particle_interval * (tframe - 1)
                fname = (pic_run_dir + "spectrum_reconnection_layer/spectrum_layer_" +
                         sname + "_" + str(tindex) + ".dat")
                spect1 = np.fromfile(fname, dtype=np.float32)
                spect1[3:] /= np.gradient(ebins)
                ax.loglog(ebins, spect[3:] - spect1[3:], linewidth=2,
                          color = plt.cm.Spectral_r((tframe - tstart_plot)/float(nframes_plot), 1))
        ax.grid(True)
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in', top=True)
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in', left=True)
        ax.tick_params(axis='y', which='major', direction='in')
        if species in ['e', 'electron']:
            ax.set_xlim([1E-1, 2E3])
        else:
            ax.set_xlim([1E-1, 2E3])
        if '3D' in pic_run:
            ax.set_ylim([1E-2, 1E12])
            ax.set_yticks(np.logspace(-1, 11, num=7))
        else:
            ax.set_ylim([1E-1, 1E9])
            ax.set_yticks(np.logspace(-1, 9, num=6))
        ax.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$',
                      fontsize=20)
        ax.set_ylabel(r'$f(\varepsilon)$', fontsize=20)
        ax.tick_params(labelsize=16)
        rect_cbar = np.copy(rect)
        rect_cbar[0] += rect[2] + 0.02
        rect_cbar[2] = 0.03
        cax = fig.add_axes(rect_cbar)
        colormap = plt.cm.get_cmap('jet', tend_plot - tstart_plot + 1)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Spectral_r,
                                   norm=plt.Normalize(vmin=tstart_plot * dtf,
                                                      vmax=tend_plot * dtf))
        cax.tick_params(axis='x', which='major', direction='in')
        # fake up the array of the scalar mappable. Urgh...
        sm._A = []
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label(r'$t\Omega_{ci}$', fontsize=16)
        cbar.ax.tick_params(labelsize=12)
        fdir = '../img/cori_3d/spectrum/' + pic_run + '/'
        mkdir_p(fdir)
        # fname = fdir + 'spectrum_' + species + '_' + str(iplot) + '.pdf'
        # fig.savefig(fname)
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


def plot_local_spectrum(plot_config):
    """Plot local spectrum
    """
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    species = plot_config["species"]
    tframe = plot_config["tframe"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
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
    nxr, nyr, nzr = pic_info.nx//2, pic_info.ny//2, pic_info.nz//2
    nslicex, nslicey, nslicez = 64, 32, 28
    box_size = 24
    box_size_h = box_size // 2
    shiftz = (nzr - (nslicez - 2) * box_size) // 2
    midx = np.linspace(box_size_h - 1, nxr - box_size_h - 1, nslicex, dtype=int)
    midy = np.linspace(box_size_h - 1, nyr - box_size_h - 1, nslicey, dtype=int)
    midz = np.linspace(box_size_h + shiftz - 1, nzr - box_size_h - shiftz - 1,
                       nslicez - 2, dtype=int)
    xslices = np.asarray([0, 13, 25, 36])
    yboxes = np.asarray([4, 12, 20, 28])
    z0, z1 = nslicez//2 - 1, 9
    dx_di = pic_info.dx_di * 2  # smoothed data
    dy_di = pic_info.dy_di * 2
    dz_di = pic_info.dy_di * 2
    xdi = midx[xslices] * dx_di
    xmin, xmax = 0, pic_info.lx_di
    ymin, ymax = -pic_info.ly_di * 0.5, pic_info.ly_di * 0.5
    zmin, zmax = -pic_info.lz_di * 0.5, pic_info.lz_di * 0.5
    jmin, jmax = 0.0, 0.4
    ydi = midy[yboxes] * dy_di + ymin
    z0_di = midz[z0] * dz_di + zmin
    z1_di = midz[z1] * dz_di + zmin

    # initial thermal distribution
    fname = (pic_run_dir + "spectrum_combined/spectrum_" + species + "_0.dat")
    spect_init = np.fromfile(fname, dtype=np.float32)
    ndata, = spect_init.shape
    spect_init[3:] /= np.gradient(ebins)
    spect_init[3:] /= (pic_info.nx * pic_info.ny * pic_info.nz / box_size**3 / 8)

    fig = plt.figure(figsize=[5, 3])
    w1, h1 = 0.78, 0.75
    xs, ys = 0.96 - w1, 0.96 - h1
    ax = fig.add_axes([xs, ys, w1, h1])
    ax.set_prop_cycle('color', COLORS)
    fname = (pic_run_dir + "spectrum_reduced/spectrum_" +
             species + "_" + str(tindex) + ".dat")
    spect = np.fromfile(fname, dtype=np.float32)
    sz, = spect.shape
    npoints = sz//ndata
    spect = spect.reshape((npoints, ndata))
    print(spect.shape)
    print(np.sum(spect[:, 3:]))
    spect[:, 3:] /= np.gradient(ebins)
    # ix, iz = 0, 9
    ix, iz = 13, 13
    for ibox, iy in enumerate(yboxes):
        cindex = iz * nslicex * nslicey + iy * nslicex  + ix
        ax.loglog(ebins, spect[cindex, 3:], linewidth=2,
                  label=str(ibox + 1))
    ax.loglog(ebins, spect_init[3:], linewidth=2, linestyle='--', color='k',
              label='Initial')
    pindex = -4.0
    power_index = "{%0.1f}" % pindex
    pname = r'$\sim \varepsilon^{' + power_index + '}$'
    fpower = 1E12*ebins**pindex
    if species == 'e':
        es, ee = 588, 688
    else:
        es, ee = 438, 538
    if species == 'e':
        ax.loglog(ebins[es:ee], fpower[es:ee], color='k', linewidth=2)
        ax.text(0.98, 0.6, pname, color='k', fontsize=20,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='right', verticalalignment='center',
                transform=ax.transAxes)
    ax.legend(loc=3, prop={'size': 16}, ncol=1,
              shadow=False, fancybox=False, frameon=False)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top='on')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in', left='on')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlim([1E-1, 2E3])
    ax.set_ylim([1E-1, 2E7])
    ax.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$',
                  fontsize=20)
    ax.set_ylabel(r'$f(\varepsilon)$', fontsize=20)
    ax.tick_params(labelsize=16)
    # fdir = "../img/cori_3d/spect_local/"
    # mkdir_p(fdir)
    # fname = (fdir + "spect_local_" + species + "_" + str(tframe) +
    #          "_ix" + str(ix) + "_iz" + str(iz) + ".pdf")
    # fig.savefig(fname)
    plt.show()


def plot_box(center, length, ax, color):
    """Plot a box in figure
    """
    xl = center[0] - length / 2
    xr = center[0] + length / 2
    yb = center[1] - length / 2
    yt = center[1] + length / 2
    xbox = [xl, xr, xr, xl, xl]
    ybox = [yb, yb, yt, yt, yb]
    ax.plot(xbox, ybox, color=color, linewidth=1)


def plot_local_spectrum2d(plot_config):
    """Plot local spectrum for the 2D simulation
    """
    root_dir = "/net/scratch3/xiaocanli/reconnection/Cori_runs/"
    pic_run = "2D-Lx150-bg0.2-150ppc-16KNL"
    pic_run_dir = root_dir + pic_run + '/'
    species = plot_config["species"]
    tframe = plot_config["tframe"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
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
    nxr, nyr, nzr = pic_info.nx, pic_info.ny, pic_info.nz
    nslicex, nslicey, nslicez = 64, 32, 28
    box_size = 48
    box_size_h = box_size // 2
    shiftz = (nzr - (nslicez - 2) * box_size) // 2
    midx = np.linspace(box_size_h - 1, nxr - box_size_h - 1, nslicex, dtype=int)
    midy = np.linspace(box_size_h - 1, nyr - box_size_h - 1, nslicey, dtype=int)
    midz = np.linspace(box_size_h + shiftz - 1, nzr - box_size_h - shiftz - 1,
                       nslicez - 2, dtype=int)
    z0, z1 = nslicez//2 - 1, 9
    dx_di = pic_info.dx_di
    dy_di = pic_info.dy_di
    dz_di = pic_info.dy_di
    xmin, xmax = 0, pic_info.lx_di
    ymin, ymax = -pic_info.ly_di * 0.5, pic_info.ly_di * 0.5
    zmin, zmax = -pic_info.lz_di * 0.5, pic_info.lz_di * 0.5
    jmin, jmax = 0.0, 0.4
    z0_di = midz[z0] * dz_di + zmin
    z1_di = midz[z1] * dz_di + zmin

    # initial thermal distribution
    fname = (pic_run_dir + "spectrum_combined/spectrum_" + species + "_0.dat")
    spect_init = np.fromfile(fname, dtype=np.float32)
    ndata, = spect_init.shape
    spect_init[3:] /= np.gradient(ebins)
    spect_init[3:] /= (pic_info.nx * pic_info.nz / box_size**2)

    fname = (pic_run_dir + "spectrum_reduced/spectrum_" +
             species + "_" + str(tindex) + ".dat")
    spect = np.fromfile(fname, dtype=np.float32)
    sz, = spect.shape
    npoints = sz//ndata
    spect = spect.reshape((npoints, ndata))
    print("Spectrum data shape: ", spect.shape)
    spect[:, 3:] /= np.gradient(ebins)

    fname = pic_run_dir + "data/absJ.gda"
    absj = np.fromfile(fname, dtype=np.float32)
    absj = absj.reshape((-1, pic_info.nz, pic_info.nx))

    if tframe == 5:
        xboxes_list = [np.asarray([2, 6, 10, 14]),
                       np.asarray([17, 21, 24, 27]),
                       np.asarray([28, 32, 36, 40]),
                       np.asarray([41, 44, 47, 49])]
    if tframe == 6:
        xboxes_list = [np.asarray([2, 6, 10, 14]),
                       np.asarray([17, 20, 22, 25]),
                       np.asarray([29, 33, 37, 41])]
    elif tframe == 7:
        xboxes_list = [np.asarray([2, 6, 10, 14]),
                       np.asarray([17, 20, 23, 26]),
                       np.asarray([28, 32, 36, 40]),
                       np.asarray([44, 47, 50, 53])]
    elif tframe == 8:
        xboxes_list = [np.asarray([2, 6, 10, 13]),
                       np.asarray([17, 20, 23, 27]),
                       np.asarray([29, 31, 36, 41]),
                       np.asarray([44, 47, 50, 53])]
    elif tframe == 9:
        xboxes_list = [np.asarray([2, 6, 10, 13]),
                       np.asarray([23, 27, 29, 32]),
                       np.asarray([42, 45, 48, 51])]
    elif tframe == 10:
        xboxes_list = [np.asarray([2, 6, 10, 13]),
                       np.asarray([23, 27, 29, 32]),
                       np.asarray([42, 45, 48, 51]),
                       np.asarray([53, 55, 58, 61])]
    elif tframe == 11:
        xboxes_list = [np.asarray([2, 5, 10, 13]),
                       np.asarray([23, 27, 29, 32]),
                       np.asarray([42, 45, 48, 51]),
                       np.asarray([53, 55, 58, 61])]
    elif tframe == 12:
        xboxes_list = [np.asarray([2, 6, 10, 14]),
                       np.asarray([17, 21, 24, 27]),
                       np.asarray([28, 32, 36, 40]),
                       np.asarray([41, 44, 47, 49])]
    elif tframe == 13:
        xboxes_list = [np.asarray([2, 5, 10, 13]),
                       np.asarray([23, 27, 29, 32]),
                       np.asarray([42, 45, 48, 51]),
                       np.asarray([53, 55, 58, 61])]
    elif tframe == 14:
        xboxes_list = [np.asarray([2, 5, 10, 13]),
                       np.asarray([23, 26, 29, 32]),
                       np.asarray([42, 45, 48, 51]),
                       np.asarray([55, 57, 59, 61])]
    elif tframe == 15:
        xboxes_list = [np.asarray([2, 5, 10, 13]),
                       np.asarray([23, 26, 29, 32]),
                       np.asarray([42, 45, 48, 51]),
                       np.asarray([55, 57, 59, 61])]
    elif tframe == 16:
        xboxes_list = [np.asarray([2, 5, 10, 13]),
                       np.asarray([23, 26, 29, 32]),
                       np.asarray([42, 45, 48, 51]),
                       np.asarray([55, 57, 59, 61])]
    elif tframe == 18:
        xboxes_list = [np.asarray([2, 5, 10, 13]),
                       np.asarray([26, 29, 32, 35]),
                       np.asarray([42, 45, 48, 51]),
                       np.asarray([53, 55, 58, 61])]
    elif tframe == 20:
        xboxes_list = [np.asarray([2, 5, 7, 10]),
                       np.asarray([25, 28, 31, 34]),
                       np.asarray([42, 45, 48, 50]),
                       np.asarray([53, 55, 58, 61])]
    elif tframe == 22:
        xboxes_list = [np.asarray([2, 5, 7, 10]),
                       np.asarray([24, 27, 30, 33]),
                       np.asarray([42, 45, 48, 50]),
                       np.asarray([53, 55, 58, 61])]
    elif tframe == 24:
        xboxes_list = [np.asarray([2, 5, 7, 10]),
                       np.asarray([23, 26, 29, 32]),
                       np.asarray([42, 45, 48, 50]),
                       np.asarray([53, 55, 58, 61])]
    elif tframe == 26:
        xboxes_list = [np.asarray([7, 10, 13, 16]),
                       np.asarray([21, 24, 27, 30]),
                       np.asarray([42, 45, 48, 50]),
                       np.asarray([53, 55, 58, 61])]
    elif tframe == 28:
        xboxes_list = [np.asarray([7, 10, 12, 15]),
                       np.asarray([19, 22, 25, 28]),
                       np.asarray([42, 45, 48, 50]),
                       np.asarray([52, 55, 58, 61])]
    elif tframe == 30:
        xboxes_list = [np.asarray([7, 10, 12, 15]),
                       np.asarray([19, 22, 25, 28]),
                       np.asarray([42, 45, 48, 50]),
                       np.asarray([52, 55, 58, 61])]
    elif tframe == 32:
        xboxes_list = [np.asarray([7, 10, 12, 15]),
                       np.asarray([19, 22, 25, 28]),
                       np.asarray([42, 45, 48, 50]),
                       np.asarray([52, 55, 58, 61])]
    elif tframe == 34:
        xboxes_list = [np.asarray([7, 10, 12, 15]),
                       np.asarray([19, 22, 25, 28]),
                       np.asarray([42, 45, 48, 50]),
                       np.asarray([52, 55, 58, 61])]
    elif tframe == 36:
        xboxes_list = [np.asarray([7, 10, 12, 15]),
                       np.asarray([19, 22, 25, 28]),
                       np.asarray([42, 45, 48, 50]),
                       np.asarray([52, 55, 58, 61])]
    elif tframe == 38:
        xboxes_list = [np.asarray([7, 10, 12, 15]),
                       np.asarray([19, 22, 25, 28]),
                       np.asarray([42, 45, 48, 50]),
                       np.asarray([52, 55, 58, 61])]
    for xboxes in xboxes_list:
        xdi = midx[xboxes] * dx_di + xmin
        fig = plt.figure(figsize=[12, 3])
        rect1 = [0.08, 0.25, 0.57, 0.7]
        ax = fig.add_axes(rect1)
        p1 = ax.imshow(absj[tframe, :, :], extent=[xmin, xmax, zmin, zmax],
                       vmin=jmin, vmax=jmax,
                       cmap=plt.cm.coolwarm, aspect='auto',
                       origin='lower', interpolation='bicubic')
        nxboxes = len(xboxes)
        COLORS = palettable.colorbrewer.diverging.RdYlGn_4.mpl_colors
        for ix in range(len(xboxes)):
            color = COLORS[ix]
            plot_box([xdi[ix], z0_di], dx_di * box_size, ax, color)
        ax.set_ylim([-20, 20])
        ax.set_xlabel(r'$x/d_i$', fontsize=20)
        ax.set_ylabel(r'$z/d_i$', fontsize=20)
        ax.tick_params(labelsize=16)
        twci = "{%0.1f}" % (math.ceil((pic_info.dt_fields) * tframe / 0.1) * 0.1)
        text1 = r"$|J| (t\Omega_{ci}=" + twci + ")$"
        ax.text(0.02, 0.85, text1, color='k', fontsize=20,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)

        rect1[0] += rect1[2] + 0.08
        rect1[2] = 0.25
        ax = fig.add_axes(rect1)
        for ibox, ix in enumerate(xboxes):
            cindex = z0 * nslicex + ix
            color = COLORS[ibox]
            ax.loglog(ebins, spect[cindex, 3:], linewidth=2,
                      color=color, label=str(ibox + 1))
        ax.loglog(ebins, spect_init[3:], linewidth=2, linestyle='--', color='k',
                  label='Initial')
        pindex = -3.6
        power_index = "{%0.1f}" % pindex
        pname = r'$\sim \varepsilon^{' + power_index + '}$'
        fpower = 5E9*ebins**pindex
        if species == 'e':
            es, ee = 548, 638
        else:
            es, ee = 438, 538
        if species == 'e':
            ax.loglog(ebins[es:ee], fpower[es:ee], color='k', linewidth=2)
            ax.text(0.98, 0.7, pname, color='k', fontsize=20,
                    bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                    horizontalalignment='right', verticalalignment='center',
                    transform=ax.transAxes)
        # ax.legend(loc=3, prop={'size': 16}, ncol=1,
        #           shadow=False, fancybox=False, frameon=False)
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in', top='on')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in', left='on')
        ax.tick_params(axis='y', which='major', direction='in')
        ax.set_xlim([1E-1, 5E2])
        ax.set_ylim([1E0, 2E6])
        ax.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$',
                      fontsize=20)
        ax.set_ylabel(r'$f(\varepsilon)$', fontsize=20)
        ax.tick_params(labelsize=16)
        fdir = "../img/cori_3d/spect_local_2d/"
        mkdir_p(fdir)
        loc = "_ix"
        for ix in xboxes:
            loc += "_" + str(ix)
        loc += "_iz_" + str(z0)
        fname = (fdir + "spect_local_" + species + "_t" + str(tframe) +
                 loc + ".pdf")
        fig.savefig(fname)
        # plt.close()
    plt.show()


def absj_local_spect(plot_config):
    """Plot current density with local spectrum
    """
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    species = plot_config["species"]
    tframe = plot_config["tframe"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    if species == 'e':
        vth = pic_info.vthe
        sname = 'electron'
    else:
        vth = pic_info.vthi
        sname = 'ion'
    nbins = 1000
    ndata = nbins + 3  # including magnetic field
    tindex = pic_info.particle_interval * tframe
    gama = 1.0 / math.sqrt(1.0 - 3 * vth**2)
    eth = gama - 1.0
    ebins = np.logspace(-6, 4, nbins)
    ebins /= eth
    nxr, nyr, nzr = pic_info.nx//2, pic_info.ny//2, pic_info.nz//2
    nslicex, nslicey, nslicez = 64, 32, 28
    box_size = 24
    box_size_h = box_size // 2
    shiftz = (nzr - (nslicez - 2) * box_size) // 2
    midx = np.linspace(box_size_h - 1, nxr - box_size_h - 1, nslicex, dtype=int)
    midy = np.linspace(box_size_h - 1, nyr - box_size_h - 1, nslicey, dtype=int)
    midz = np.linspace(box_size_h + shiftz - 1, nzr - box_size_h - shiftz - 1,
                       nslicez - 2, dtype=int)
    dx_di = pic_info.dx_di * 2  # smoothed data
    dy_di = pic_info.dy_di * 2
    dz_di = pic_info.dy_di * 2
    xmin, xmax = 0, pic_info.lx_di
    ymin, ymax = -pic_info.ly_di * 0.5, pic_info.ly_di * 0.5
    zmin, zmax = -pic_info.lz_di * 0.5, pic_info.lz_di * 0.5
    jmin, jmax = 0.0, 0.4
    xcut = np.argwhere(midx==947)[0][0]
    ycut = np.argwhere(midy==731)[0][0]
    zcut = 13
    xde = (midx[xcut] * dx_di + xmin) * math.sqrt(pic_info.mime)
    yde = (midy[ycut] * dy_di + ymin) * math.sqrt(pic_info.mime)
    zde = (midz[zcut] * dz_di + zmin) * math.sqrt(pic_info.mime)
    print(xde, yde, zde)
    print(dx_di * 24)

    xcuts = np.arange(xcut+7, nslicex-5, 4)
    ycuts = np.arange(2, ycut, 5)
    xboxes = midx[xcuts]
    yboxes = midy[ycuts]
    zbox = midz[zcut]

    xde = (midx[xcuts] * dx_di + xmin) * math.sqrt(pic_info.mime)
    yde = (midy[ycuts] * dy_di + ymin) * math.sqrt(pic_info.mime)
    print(xde)
    print(yde)

    fname = pic_run_dir + "data-smooth/absJ_" + str(tindex) + ".gda"
    absj = np.fromfile(fname, dtype=np.float32)
    absj = absj.reshape((nzr, nyr, nxr))

    fdir = '../img/cori_3d/'
    mkdir_p(fdir)

    fig = plt.figure(figsize=[10, 7])
    rect = [0.06, 0.47, 0.6, 0.45]
    hgap, vgap = 0.02, 0.02
    ax = fig.add_axes(rect)
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    fdata = absj[nzr//2, :, :]
    p1 = ax.imshow(fdata, extent=[0, nxr-1, 0, nyr-1],
                   vmin=jmin, vmax=jmax,
                   cmap=plt.cm.viridis, aspect='auto',
                   origin='lower', interpolation='bicubic')
    ax.plot([midx[xcut], midx[xcut]], [0, nyr-1], color='w', linewidth=0.5)
    ax.plot([0, nxr-1], [midy[ycut], midy[ycut]], color='w', linewidth=0.5)
    for ix, xpos in enumerate(xboxes):
        plot_box([xpos, midy[ycut]], box_size, ax, COLORS[ix])
    for iy, ypos in enumerate(yboxes):
        plot_box([midx[xcut], ypos], box_size, ax, COLORS[iy])
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.tick_params(axis='x', labelbottom=False)
    ax.set_ylabel(r'$y$', fontsize=12)
    ax.tick_params(labelsize=10)

    rect1 = np.copy(rect)
    rect1[3] = 0.36
    rect1[1] -= rect1[3] + vgap
    ax = fig.add_axes(rect1)
    fdata = absj[:, midy[ycut], :]
    p1 = ax.imshow(fdata, extent=[0, nxr-1, 0, nzr-1],
                   vmin=jmin, vmax=jmax,
                   cmap=plt.cm.viridis, aspect='auto',
                   origin='lower', interpolation='bicubic')
    ax.plot([midx[xcut], midx[xcut]], [0, nzr-1], color='w', linewidth=0.5)
    ax.plot([0, nxr-1], [zbox, zbox], color='w', linewidth=0.5)
    for xpos in xboxes:
        plot_box([xpos, zbox], box_size, ax, 'w')
    ax.set_xlabel(r'$x$', fontsize=12)
    ax.set_ylabel(r'$z$', fontsize=12)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')

    rect[0] += rect[2] + hgap
    rect[2] = 0.3
    ax = fig.add_axes(rect)
    fdata = absj[:, :, midx[xcut]].T
    p1 = ax.imshow(fdata, extent=[0, nzr-1, 0, nyr-1],
                   vmin=jmin, vmax=jmax,
                   cmap=plt.cm.viridis, aspect='auto',
                   origin='lower', interpolation='bicubic')
    ax.plot([0, nzr-1], [midy[ycut], midy[ycut]], color='w', linewidth=0.5)
    ax.plot([zbox, zbox], [0, nyr-1], color='w', linewidth=0.5)
    for ypos in yboxes:
        plot_box([zbox, ypos], box_size, ax, 'w')
    ax.tick_params(axis='y', labelleft=False)
    ax.set_xlabel(r'$z$', fontsize=12)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')

    # rect_cbar = np.copy(rect)
    # rect_cbar[0] = rect[0] + rect[2] + hgap
    # rect_cbar[2] = 0.02
    # cbar_ax = fig.add_axes(rect_cbar)
    # cbar = fig.colorbar(p1, cax=cbar_ax)
    # cbar.ax.tick_params(labelsize=10)
    # cbar_ax.set_title(r'$J/J_0$', fontsize=12)

    # initial thermal distribution
    fname = (pic_run_dir + "spectrum_combined/spectrum_" + species + "_0.dat")
    spect_init = np.fromfile(fname, dtype=np.float32)
    ndata, = spect_init.shape
    spect_init[3:] /= np.gradient(ebins)
    spect_init[3:] /= (pic_info.nx * pic_info.ny * pic_info.nz / box_size**3 / 8)

    fig = plt.figure(figsize=[3.5, 2.5])
    rect = [0.16, 0.16, 0.69, 0.8]
    ax = fig.add_axes(rect)
    ax.set_prop_cycle('color', COLORS)
    if plot_config['binary']:
        fname = (pic_run_dir + "spectrum_reduced/spectrum_" +
                 species + "_" + str(tindex) + ".dat")
        spect = np.fromfile(fname, dtype=np.float32)
        sz, = spect.shape
        npoints = sz//ndata
        spect = spect.reshape((npoints, ndata))
        print("Spectral data size: %d, %d" % (npoints, ndata))
        spect[:, 3:] /= np.gradient(ebins)
        for xindex, ix in enumerate(xcuts):
            cindex = zcut * nslicex * nslicey + ycut * nslicex  + ix
            ax.loglog(ebins, spect[cindex, 3:], linewidth=1,
                      color=COLORS[xindex])
        for yindex, iy in enumerate(ycuts):
            cindex = zcut * nslicex * nslicey + iy * nslicex  + xcut
            ax.loglog(ebins, spect[cindex, 3:], linewidth=1,
                      linestyle='--', color=COLORS[yindex])
    else:
        fname = (pic_run_dir + "spectrum_reduced/spectrum_" +
                 sname + "_" + str(tindex) + ".h5")
        with h5py.File(fname, 'r') as fh:
            dset = fh['spectrum']
            sz = dset.shape
            spect = np.zeros(sz)
            dset.read_direct(spect)
        for xindex, ix in enumerate(xcuts):
            fspect_local = spect[zcut, ycut, ix, 3:] / np.gradient(ebins)
            ax.loglog(ebins, fspect_local, linewidth=1, color=COLORS[xindex])
        # for yindex, iy in enumerate(ycuts):
        #     fspect_local = spect[zcut, iy, xcut, 3:] / np.gradient(ebins)
        #     ax.loglog(ebins, fspect_local, linewidth=1,
        #               linestyle='--', color=COLORS[yindex])
    for i in range(4):
        ypos = 0.3 - i * 0.08
        text1 = 'Box' + str(i+1)
        ax.text(0.05, ypos, text1, color=COLORS[i], fontsize=10,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=0.0),
                horizontalalignment='left', verticalalignment='bottom',
                transform=ax.transAxes)
    ax.loglog(ebins, spect_init[3:], linewidth=0.5, linestyle='--',
              color='k', label='initial')
    pindex = -4.0
    power_index = "{%0.1f}" % pindex
    pname = r'$\propto \varepsilon^{' + power_index + '}$'
    fpower = 5E11*ebins**pindex
    if species == 'e':
        es, ee = 568, 668
    else:
        es, ee = 438, 538
    if species == 'e':
        ax.loglog(ebins[es:ee], fpower[es:ee], color='k', linewidth=0.5)
        ax.text(0.85, 0.58, pname, color='k', fontsize=10, rotation=-65,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='right', verticalalignment='center',
                transform=ax.transAxes)
        ax.text(0.52, 0.07, "initial", color='k', fontsize=10,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='right', verticalalignment='center',
                transform=ax.transAxes)
    # ax.text(0.97, 0.9, '(c)', color='k', fontsize=10,
    #         bbox=dict(facecolor='none', alpha=1.0,
    #                   edgecolor='none', pad=10.0),
    #         horizontalalignment='right', verticalalignment='center',
    #         transform=ax.transAxes)
    ax.set_yticks(np.logspace(0, 6, num=4))
    ax.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$', fontsize=10)
    ax.set_ylabel(r'$f(\varepsilon)$', fontsize=10)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top=False)
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in', left=False)
    ax.tick_params(axis='y', which='major', direction='in')
    ax.tick_params(labelsize=8)
    ax.set_xlim([1E-1, 1E3])
    ax.set_ylim([1E-1, 2E7])

    fdir = '../img/cori_3d/espect/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + "espect_local_" + species + "_new.pdf"
    fig.savefig(fname)
    plt.show()


def plot_momentum_spectrum(plot_config, show_plot=True):
    """Plot momentum spectrum for each time frame
    """
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    species = plot_config["species"]
    tframe = plot_config["tframe"]
    tstart = plot_config["tstart"]
    tend = plot_config["tend"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    if species == 'e':
        vth = 0.1
    else:
        vth = 0.02
    gama = 1.0 / math.sqrt(1.0 - 3 * vth**2)
    eth = gama - 1.0
    emin, emax = 1E-6, 1E4
    nband = 1000
    dloge = (math.log10(emax) - math.log10(emin)) / (nband - 1)
    emin0 = 10**(math.log10(emin) - dloge)
    ebins = np.logspace(math.log10(emin0), math.log10(emax), nband+1)
    pbins = np.sqrt((ebins + 1)**2 - 1)
    pbins /= np.sqrt((eth + 1)**2 - 1)
    pbins_mid = (pbins[:-1] + pbins[1:]) * 0.5
    dpbins = np.diff(pbins)
    ebins /= eth
    ebins_mid = (ebins[:-1] + ebins[1:]) * 0.5
    debins = np.diff(ebins)
    particle_interval = pic_info.dt_particles  # in 1/wci

    # separate all frame into 4 plots
    nframes = tend - tstart + 1
    nplots = 4
    nframes_plot = nframes // nplots
    dtf = math.ceil(pic_info.dt_particles / 0.1) * 0.1

    for iplot in range(nplots):
        fig = plt.figure(figsize=[7, 5])
        rect = [0.13, 0.16, 0.7, 0.8]
        ax = fig.add_axes(rect)
        tstart_plot = nframes_plot * iplot
        if iplot == nplots - 1:
            tend_plot = tend
        else:
            tend_plot = tstart_plot + nframes_plot
        for tframe in range(tstart_plot, tend_plot + 1):
            print("Time frame: %d" % tframe)
            tindex = pic_info.fields_interval * tframe
            fname = (pic_run_dir + "spectrum_combined/spectrum_" +
                     species + "_" + str(tindex) + ".dat")
            spect = np.fromfile(fname, dtype=np.float32)
            pspect = spect[3:] / dpbins
            ax.loglog(pbins_mid, pspect, linewidth=1,
                      color = plt.cm.Spectral_r((tframe - tstart_plot)/float(nframes_plot), 1))
        pindex1, mom = find_nearest(pbins_mid, 7)
        pindex2, mom = find_nearest(pbins_mid, 13)
        popt, pcov = curve_fit(fitting_funcs.func_line,
                               np.log10(pbins_mid[pindex1:pindex2]),
                               np.log10(pspect[pindex1:pindex2]))
        fpower = fitting_funcs.func_line(np.log10(pbins_mid), popt[0], popt[1])
        fpower = 10**fpower
        power_index = "{%0.2f}" % popt[0]
        pname = r'$\propto p^{' + power_index + '}$'
        ax.text(0.6, 0.8, pname, color='k', fontsize=20,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)
        pindex_10, mom = find_nearest(pbins_mid, 10)
        pnorm = pspect[pindex_10] / fpower[pindex_10]
        fpower *= pnorm * 5
        pindex_5, mom = find_nearest(pbins_mid, 5)
        pindex_40, mom = find_nearest(pbins_mid, 40)
        ax.loglog(pbins_mid[pindex_5:pindex_40], fpower[pindex_5:pindex_40],
                  linewidth=1, linestyle='--', color='k')
        # ax.loglog(pbins_mid[pindex1], pspect[pindex1], marker='.',
        #           markersize=10, linestyle='None', color='b')
        # ax.loglog(pbins_mid[pindex2], pspect[pindex2], marker='.',
        #           markersize=10, linestyle='None', color='b')
        ax.set_xlabel(r'$p/p_\text{th}$', fontsize=20)
        ax.set_ylabel(r'$f(p)$', fontsize=20)
        ax.set_xlim([5E-1, 1E2])
        if '3D' in pic_run:
            ax.set_ylim([1E0, 1E12])
            ax.set_yticks(np.logspace(0, 12, num=7))
        else:
            ax.set_ylim([1E0, 1E9])
            ax.set_yticks(np.logspace(1, 9, num=5))

        ax.tick_params(labelsize=16)

        # colorbar
        rect_cbar = np.copy(rect)
        rect_cbar[0] += rect[2] + 0.02
        rect_cbar[2] = 0.03
        cax = fig.add_axes(rect_cbar)
        colormap = plt.cm.get_cmap('jet', tend_plot - tstart_plot + 1)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Spectral_r,
                                   norm=plt.Normalize(vmin=tstart_plot * dtf,
                                                      vmax=tend_plot * dtf))
        cax.tick_params(axis='x', which='major', direction='in')
        # fake up the array of the scalar mappable. Urgh...
        sm._A = []
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label(r'$t\Omega_{ci}$', fontsize=16)
        cbar.ax.tick_params(labelsize=12)

        fdir = '../img/cori_3d/momentum_spectrum/' + pic_run + '/'
        mkdir_p(fdir)
        fname = fdir + 'pspect_' + species + '_' + str(iplot) + '.pdf'
        fig.savefig(fname)
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_momentum_spectrum_multi(plot_config, show_plot=True):
    """Plot momentum spectrum for all time frames
    """
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    bg = plot_config["bg"]
    root_dir = "/net/scratch3/xiaocanli/reconnection/Cori_runs/"
    pic_runs = ["2D-Lx150-bg" + str(bg) + "-150ppc-16KNL"]
    pic_runs.append("3D-Lx150-bg" + str(bg) + "-150ppc-2048KNL")
    species = plot_config["species"]
    tframe = plot_config["tframe"]
    tstart = plot_config["tstart"]
    tend = plot_config["tend"]
    if species == 'e':
        vth = 0.1
    else:
        vth = 0.02
    gama = 1.0 / math.sqrt(1.0 - 3 * vth**2)
    eth = gama - 1.0
    emin, emax = 1E-6, 1E4
    nband = 1000
    dloge = (math.log10(emax) - math.log10(emin)) / (nband - 1)
    emin0 = 10**(math.log10(emin) - dloge)
    ebins = np.logspace(math.log10(emin0), math.log10(emax), nband+1)
    pbins = np.sqrt((ebins + 1)**2 - 1)
    pbins /= np.sqrt((eth + 1)**2 - 1)
    pbins_mid = (pbins[:-1] + pbins[1:]) * 0.5
    dpbins = np.diff(pbins)
    ebins /= eth
    ebins_mid = (ebins[:-1] + ebins[1:]) * 0.5
    debins = np.diff(ebins)
    particle_interval = 10.0  # in 1/wci
    tend = 20

    tstarts = [[0, 12], [0, 20]]
    tends = [[12, 40], [20, 40]]
    angle = [[47, 60], [43, 45]]
    ypos = [[0.6, 0.61], [0.76, 0.7]]
    norms = [[2, 5], [2, 5]]
    pene1 = [[5, 6], [5, 6]]
    pene2 = [[30, 30], [30, 40]]
    cbar_ticks = [[np.linspace(tstarts[0][0], tends[0][0], 3),
                   np.linspace(tstarts[0][1], tends[0][1], 5)],
                  [np.linspace(tstarts[1][0], tends[1][0], 3),
                   np.linspace(tstarts[1][1], tends[1][1], 5)]]

    fig = plt.figure(figsize=[3.2, 4.0])
    rect = [[0.17, 0.55, 0.78, 0.42],
            [0.20, 0.58, 0.25, 0.15]]
    for irun, pic_run in enumerate(pic_runs):
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        for i in range(2):
            tstart, tend = tstarts[irun][i], tends[irun][i]
            nframes = tend - tstart + 1
            ax = fig.add_axes(rect[i])
            for tframe in range(tstart, tend + 1):
                print("Time frame: %d" % tframe)
                pic_run_dir = root_dir + pic_run + "/"
                tindex = pic_info.fields_interval * tframe
                fname = (pic_run_dir + "spectrum_combined/spectrum_" +
                         species + "_" + str(tindex) + ".dat")
                spect = np.fromfile(fname, dtype=np.float32)
                pspect = spect[3:] / dpbins
                color = plt.cm.Spectral_r((tframe - tstart)/float(nframes), 1)
                ax.loglog(pbins_mid, pspect, linewidth=1, color=color)
                if tframe == tend and species == 'e':
                    pindex1, mom = find_nearest(pbins_mid, 7)
                    pindex2, mom = find_nearest(pbins_mid, 13)
                    popt, pcov = curve_fit(fitting_funcs.func_line,
                                           np.log10(pbins_mid[pindex1:pindex2]),
                                           np.log10(pspect[pindex1:pindex2]))
                    fpower = fitting_funcs.func_line(np.log10(pbins_mid), popt[0], popt[1])
                    fpower = 10**fpower
                    power_index = "{%0.1f}" % popt[0]
                    pname = r'$\propto p^{' + power_index + '}$'
                    fsize = 10 if i == 0 else 6
                    ax.text(0.52, ypos[irun][i], pname, color='k', fontsize=fsize,
                            rotation=-angle[irun][i],
                            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                            horizontalalignment='left', verticalalignment='center',
                            transform=ax.transAxes)
                    pindex_10, mom = find_nearest(pbins_mid, 10)
                    pnorm = pspect[pindex_10] / fpower[pindex_10]
                    fpower *= pnorm * norms[irun][i]
                    pindex1, mom = find_nearest(pbins_mid, pene1[irun][i])
                    pindex2, mom = find_nearest(pbins_mid, pene2[irun][i])
                    ax.loglog(pbins_mid[pindex1:pindex2], fpower[pindex1:pindex2],
                              linewidth=0.5, linestyle='--', color='k')
            if i == 0:
                if irun == 0:
                    ax.tick_params(axis='x', labelbottom='off')
                else:
                    ax.set_xlabel(r'$p/p_\text{th}$', fontsize=10)
                ax.set_ylabel(r'$f(p)$', fontsize=10)
            else:
                ax.tick_params(axis='x', labelbottom='off')
                ax.tick_params(axis='y', labelleft='off')
            ax.tick_params(bottom=True, top=True, left=True, right=True)
            ax.tick_params(axis='x', which='minor', direction='in', top='on')
            ax.tick_params(axis='x', which='major', direction='in')
            ax.tick_params(axis='y', which='minor', direction='in')
            ax.tick_params(axis='y', which='major', direction='in')
            ax.set_xlim([5E-1, 1E2])
            if irun == 0:
                ax.set_ylim([1E0, 2E9])
                ax.set_yticks(np.logspace(0, 9, num=4))
            else:
                ax.set_ylim([1E0, 2E12])
                ax.set_yticks(np.logspace(0, 12, num=4))

            ax.tick_params(labelsize=8)

            if i == 0:
                pdim = "2D" if "2D" in pic_run else "3D"
                ax.text(0.97, 0.9, pdim, color='k', fontsize=10,
                        bbox=dict(facecolor='none', alpha=1.0,
                                  edgecolor='none', pad=10.0),
                        horizontalalignment='right', verticalalignment='center',
                        transform=ax.transAxes)
                rect_cbar = np.copy(rect[i])
                rect_cbar[0] = rect[i][0] + rect[i][2] - 0.05
                rect_cbar[1] = rect[i][1] + rect[i][3] * 0.4
                rect_cbar[2] = 0.03
                rect_cbar[3] = rect[i][3] * 0.4
                cax = fig.add_axes(rect_cbar)
                dtf = math.ceil(pic_info.dt_fields / 0.1) * 0.1
                sm = plt.cm.ScalarMappable(cmap=plt.cm.Spectral_r,
                                           norm=plt.Normalize(vmin=tstart * dtf,
                                                              vmax=tend * dtf))
                cax.tick_params(axis='y', which='major', direction='in')
                # fake up the array of the scalar mappable. Urgh...
                sm._A = []
                cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
                cbar.set_label(r'$t\Omega_{ci}$', fontsize=8)
                cbar.set_ticks(cbar_ticks[irun][0] * dtf)
                cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(),
                                        rotation='vertical')
                cbar.ax.tick_params(labelsize=8)
                cax.yaxis.set_ticks_position('left')
                cax.yaxis.set_label_position('left')
            else:
                ax.text(0.05, 0.05, 'Latter', color='k', fontsize=8,
                        bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                        horizontalalignment='left', verticalalignment='bottom',
                        transform=ax.transAxes)

            rect[i][1] -= rect[0][3] + 0.02
    fdir = '../img/cori_3d/momentum_spectrum/'
    mkdir_p(fdir)
    fname = fdir + "pspect_" + species + "_bg" + str(int(bg*10)).zfill(2) + ".pdf"
    fig.savefig(fname)
    if show_plot:
        plt.show()
    else:
        plt.close()


def fit_thermal_core(ene, f):
    """Fit to get the thermal core of the particle distribution.

    Fit the thermal core of the particle distribution.
    The thermal core is fitted as a Maxwellian distribution.

    Args:
        ene: the energy bins array.
        f: the particle flux distribution.

    Returns:
        fthermal: thermal part of the particle distribution.
    """
    print('Fitting to get the thermal core of the particle distribution')
    estart = 0
    ng = 3
    kernel = np.ones(ng) / float(ng)
    fnew = np.convolve(f, kernel, 'same')
    nshift = 10  # grids shift for fitting thermal core.
    eend = np.argmax(fnew) + nshift
    popt, pcov = curve_fit(fitting_funcs.func_maxwellian,
                           ene[estart:eend], f[estart:eend])
    fthermal = fitting_funcs.func_maxwellian(ene, popt[0], popt[1])
    print('Energy with maximum flux: %f' % ene[eend - 10])
    print('Energy with maximum flux in fitted thermal core: %f' % (0.5 / popt[1]))
    return fthermal


def accumulated_particle_info(ene, f):
    """
    Get the accumulated particle number and total energy from
    the distribution function.

    Args:
        ene: the energy bins array.
        f: the energy distribution array.
    Returns:
        nacc_ene: the accumulated particle number with energy.
        eacc_ene: the accumulated particle total energy with energy.
    """
    nbins, = f.shape
    dlogE = (math.log10(max(ene)) - math.log10(min(ene))) / nbins
    nacc_ene = np.zeros(nbins)
    eacc_ene = np.zeros(nbins)
    nacc_ene[0] = f[0] * ene[0]
    eacc_ene[0] = 0.5 * f[0] * ene[0]**2
    for i in range(1, nbins):
        nacc_ene[i] = f[i] * (ene[i] + ene[i - 1]) * 0.5 + nacc_ene[i - 1]
        eacc_ene[i] = 0.5 * f[i] * (ene[i] - ene[i - 1]) * (
            ene[i] + ene[i - 1])
        eacc_ene[i] += eacc_ene[i - 1]
    nacc_ene *= dlogE
    eacc_ene *= dlogE
    return (nacc_ene, eacc_ene)


def compare_spectrum(plot_config):
    """Compare 2D and 3D spectra

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    bg = plot_config["bg"]
    root_dir = "/net/scratch3/xiaocanli/reconnection/Cori_runs/"
    pic_runs = ["2D-Lx150-bg" + str(bg) + "-150ppc-16KNL"]
    pic_runs.append("3D-Lx150-bg" + str(bg) + "-150ppc-2048KNL")
    if bg == 0.2:
        tframes = np.asarray([0, 5, 8, 20])
    elif bg == 1.0:
        tframes = np.asarray([0, 5, 8, 30])
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
        gama = 1.0 / math.sqrt(1.0 - 3 * vth**2)
        eth = gama - 1.0
        ebins = np.logspace(-6, 4, 1000)
        ebins /= eth
        print(eth)

        if irun == 0:
            fnorm = 1
            lstyle = '--'
        else:
            fnorm = pic_info.ny
            lstyle = '-'

        tinterval = pic_info.particle_interval
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
                nacc, eacc = accumulated_particle_info(ebins, spect[3:])
                print("Number fraction (E > 20Eth, %s): %0.3f" %
                      (pdim, (nacc[-1] - nacc[eindex_20])/nacc[-1]))
                print("Energy fraction (E > 20Eth, %s): %0.3f" %
                      (pdim, (eacc[-1] - eacc[eindex_20])/eacc[-1]))
            spect[spect == 0] = np.nan
            if iframe > 0:
                ax.loglog(ebins, spect[3:], linewidth=2,
                          linestyle=lstyle, color=colors[iframe - 1])
            else:
                if irun == 1:
                    ax.loglog(ebins, spect[3:], linewidth=1,
                              linestyle='--', color='k')

    if species == 'e':
        if bg == 0.2:
            fpower = 2E12 * ebins**-4
            power_index = "{%0.1f}" % -4.0
        elif bg == 1.0:
            fpower = 1E13 * ebins**-4.5
            power_index = "{%0.1f}" % -4.5
        pname = r'$\sim \varepsilon^{' + power_index + '}$'
        ax.loglog(ebins[558:658], fpower[558:658], linewidth=1, color='k')
    else:
        if bg == 0.2:
            fpower = 1E12 * ebins**-3.5
            power_index = "{%0.1f}" % -3.5
        elif bg == 1.0:
            fpower = 5E11 * ebins**-3.5
            power_index = "{%0.1f}" % -3.5
        pname = r'$\sim \varepsilon^{' + power_index + '}$'
        ax.loglog(ebins[428:528], fpower[428:528], linewidth=1, color='k')
    ax.text(0.85, 0.7, pname, color='k', fontsize=20,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='right', verticalalignment='center',
            transform=ax.transAxes)

    ax.plot([0, 10], [0, 0], linestyle="--", color='k',
            linewidth=2, label='2D')
    ax.plot([0, 10], [0, 0], linestyle="-", color='k',
            linewidth=2, label='3D')
    ax.legend(loc=3, prop={'size': 20}, ncol=1,
              shadow=False, fancybox=False, frameon=False)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top=True)
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    if species == 'e':
        ax.set_xlim([1E-1, 1E3])
    else:
        ax.set_xlim([1E-1, 2E3])
    ax.set_ylim([1E-3, 1E9])
    ax.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$', fontsize=20)
    ax.set_ylabel(r'$f(\varepsilon)$', fontsize=20)
    ax.tick_params(labelsize=16)

    if species == 'e':
        if bg == 0.2:
            xpos = [0.4, 0.85, 0.92]
        elif bg == 1.0:
            xpos = [0.5, 0.82, 0.94]
    else:
        if bg == 0.2:
            xpos = [0.45, 0.83, 0.95]
        elif bg == 1.0:
            xpos = [0.45, 0.78, 0.93]
    text1 = r'$t\Omega_{ci}=0$'
    ax.text(xpos[0], 0.02, text1, color='k', fontsize=16,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='bottom',
            transform=ax.transAxes)
    text2 = r'$' + str(int(tframes[1]*10)) + '$'
    ax.text(xpos[1], 0.02, text2, color=colors[0], fontsize=16,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='bottom',
            transform=ax.transAxes)
    text3 = r'$' + str(int(tframes[2]*10)) + '$'
    ax.text(xpos[2], 0.02, text3, color=colors[1], fontsize=16,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='bottom',
            transform=ax.transAxes)

    # fthermal = fit_thermal_core(ebins, spect[3:])
    # fnonthermal = spect[3:] - fthermal
    # rect = [0.2, 0.25, 0.31, 0.4]
    # ax1 = fig.add_axes(rect)
    # ax1.loglog(ebins, spect[3:], linewidth=3,
    #            linestyle=lstyle, color=colors[iframe])
    # ax1.loglog(ebins, fthermal, linewidth=1, linestyle='--', color='k')
    # ax1.loglog(ebins, fnonthermal, linewidth=1, linestyle='-',
    #            color=colors[1])
    # nacc, eacc = accumulated_particle_info(ebins, spect[3:])
    # nacc_thermal, eacc_thermal = accumulated_particle_info(ebins, fthermal)
    # nacc, eacc = accumulated_particle_info(ebins, spect[3:])
    # nacc_nthermal, eacc_nthermal = accumulated_particle_info(ebins, fnonthermal)
    # print("Thermal and non-thermal number fraction: %0.2f, %0.2f" %
    #       (nacc_thermal[-1]/nacc[-1], nacc_nthermal[-1]/nacc[-1]))
    # print("Thermal and non-thermal energy fraction: %0.2f, %0.2f" %
    #       (eacc_thermal[-1]/eacc[-1], eacc_nthermal[-1]/eacc[-1]))
    # ax1.text(0.02, 0.22, 'all particles', color=colors[iframe], fontsize=12,
    #          bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
    #          horizontalalignment='left', verticalalignment='bottom',
    #          transform=ax1.transAxes)
    # ax1.text(0.02, 0.12, 'thermal', color='k', fontsize=12,
    #          bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
    #          horizontalalignment='left', verticalalignment='bottom',
    #          transform=ax1.transAxes)
    # ax1.text(0.02, 0.02, 'non-thermal', color=colors[1], fontsize=12,
    #          bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
    #          horizontalalignment='left', verticalalignment='bottom',
    #          transform=ax1.transAxes)
    # ax1.set_xlim([1E-1, 1E3])
    # ax1.set_ylim([1E-3, 1E9])

    fdir = '../img/cori_3d/spectrum/'
    mkdir_p(fdir)
    fname = fdir + species + 'spect_32_bg' + str(int(bg*10)).zfill(2) + '.pdf'
    fig.savefig(fname)

    plt.show()


def spectrum_along_x(plot_config):
    """plot spectrum along x

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    bg = plot_config["bg"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    tframe = plot_config["tframe"]
    fig = plt.figure(figsize=[7, 5])
    rect = [0.14, 0.14, 0.8, 0.8]
    ax = fig.add_axes(rect)
    colors = np.copy(COLORS)
    colors[5] = colors[6]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    if species == 'e':
        vth = pic_info.vthe
        sname = 'electron'
    else:
        vth = pic_info.vthi
        sname = 'ion'
    gama = 1.0 / math.sqrt(1.0 - 3 * vth**2)
    eth = gama - 1.0
    ebins = np.logspace(-6, 4, 1000)
    ebins /= eth

    tinterval = pic_info.particle_interval
    dtf = math.ceil(pic_info.dt_fields / 0.1) * 0.1
    pdim = "2D" if "2D" in pic_run else "3D"
    tindex = tinterval * tframe
    fname = (pic_run_dir + "spectrum_along_x/spectrum_" +
             sname + "_" + str(tindex) + ".h5")

    with h5py.File(fname, 'r') as fh:
        dset = fh['spectrum']
        sz = dset.shape
        fdata = np.zeros(sz)
        dset.read_direct(fdata)
    fdata = np.squeeze(fdata)
    fdata[:, 3:] /= np.gradient(ebins)
    nx, ndata = fdata.shape
    ixs, ixe = 0, 64
    # for ix in range(ixs, ixe, 4):
    #     color = plt.cm.jet((ix - ixs)/float(ixe-ixs), 1)
        # ax.loglog(ebins, fdata[ix, 3:].T, linewidth=1,
        #           color=color)
    ax.loglog(ebins, np.sum(fdata[ixs:ixe, 3:], axis=0), linewidth=2)
    ax.loglog(ebins, np.sum(fdata[64:128, 3:], axis=0), linewidth=2)
    ax.loglog(ebins, np.sum(fdata[128:192, 3:], axis=0), linewidth=2)
    ax.loglog(ebins, np.sum(fdata[192:256, 3:], axis=0), linewidth=2)
    ax.loglog(ebins, np.sum(fdata[:, 3:], axis=0), linewidth=2)
    ax.grid(True)
    if species == 'e':
        ax.set_xlim([1E-1, 1E3])
    else:
        ax.set_xlim([1E-1, 2E3])
    # ax.set_ylim([1E-2, 1E10])
    ax.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$', fontsize=20)
    ax.set_ylabel(r'$f(\varepsilon)$', fontsize=20)
    ax.tick_params(labelsize=16)

    fdir = '../img/cori_3d/spectrum/'
    mkdir_p(fdir)
    fname = fdir + species + 'spect_32_bg' + str(int(bg*10)).zfill(2) + '.pdf'
    # fig.savefig(fname)

    plt.show()


def get_cmd_args():
    """Get command line arguments
    """
    default_pic_run = '3D-Lx150-bg0.2-150ppc-2048KNL'
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
                        help='Normalized guide field strength')
    parser.add_argument('--whole_spectrum', action="store_true", default=False,
                        help='whether to plot spectrum in the whole box')
    parser.add_argument('--binary', action="store_true", default=False,
                        help='whether spectrum in binary format')
    parser.add_argument('--spectrum_pub', action="store_true", default=False,
                        help='whether to plot spectrum for publication')
    parser.add_argument('--spectrum_pub_23', action="store_true", default=False,
                        help='whether to plot spectrum for publication')
    parser.add_argument('--all_frames', action="store_true", default=False,
                        help='whether to analyze all frames')
    parser.add_argument('--local_spectrum', action="store_true", default=False,
                        help='whether to plot local spectrum')
    parser.add_argument('--local_spectrum2d', action="store_true", default=False,
                        help='whether to plot local spectrum for the 2D simulation')
    parser.add_argument('--mom_spectrum', action="store_true", default=False,
                        help='whether to plot momentum spectrum')
    parser.add_argument('--compare_spectrum', action="store_true", default=False,
                        help='whether to compare 2D and 3D spectra')
    parser.add_argument('--single_run', action="store_true", default=False,
                        help="whether to plot for a single run")
    parser.add_argument('--show_plot', action="store_true", default=False,
                        help='whether to show plot')
    parser.add_argument('--both_species', action="store_true", default=False,
                        help='whether to plot spectra for both species')
    parser.add_argument('--spect_along_x', action="store_true", default=False,
                        help='whether to plot spectra along x')
    parser.add_argument('--spect_rec_layer', action="store_true", default=False,
                        help='whether to plot spectra only in reconnection layer')
    parser.add_argument('--absj_local_spect', action="store_true", default=False,
                        help='whether to plot current density with local spectrum')
    return parser.parse_args()


def analysis_single_frames(plot_config, args):
    """Analysis for multiple time frames
    """
    tframe = args.tframe
    if args.whole_spectrum:
        if args.single_run:
            plot_spectrum(plot_config)
        else:
            if args.all_frames:
                plot_spectrum_multi(plot_config)
            else:
                plot_spectrum_single(plot_config, args.show_plot)
    elif args.compare_spectrum:
        compare_spectrum(plot_config)
    elif args.mom_spectrum:
        if args.single_run:
            plot_momentum_spectrum(plot_config)
        else:
            if args.all_frames:
                plot_momentum_spectrum_multi(plot_config)
    elif args.spectrum_pub:
        plot_spectrum_pub(plot_config)
    elif args.spectrum_pub_23:
        plot_spectrum_pub_23(plot_config)
    elif args.local_spectrum:
        plot_local_spectrum(plot_config)
    elif args.local_spectrum2d:
        plot_local_spectrum2d(plot_config)
    elif args.both_species:
        plot_spectrum_both(plot_config)
    elif args.spect_along_x:
        spectrum_along_x(plot_config)
    elif args.spect_rec_layer:
        spectrum_reconnection_layer(plot_config)
    elif args.absj_local_spect:
        absj_local_spect(plot_config)


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
            # if args.mom_spectrum:
            #     plot_momentum_spectrum_single(plot_config, show_plot=False)
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
