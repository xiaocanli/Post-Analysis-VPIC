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
    pic_run_dir = plot_config["pic_run_dir"]
    pic_runs = ["2D-Lx150-bg0.2-150ppc-16KNL", "3D-Lx150-bg0.2-150ppc-2048KNL"]
    fig = plt.figure(figsize=[3.25, 2.5])
    w1, h1 = 0.78, 0.78
    xs, ys = 0.96 - w1, 0.96 - h1
    ax = fig.add_axes([xs, ys, w1, h1])

    for pic_run in pic_runs:
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
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
        print("Energy conservation: %e" % ((etot[-1] - etot[0]) / etot[0]))
        print("Energy conversion: %e" %
              ((ene_magnetic[-1] - ene_magnetic[0]) / ene_magnetic[0]))

        ene_bx /= enorm
        ene_by /= enorm
        ene_bz /= enorm
        ene_magnetic /= enorm
        kene_e /= enorm
        kene_i /= enorm

        pdim = "2D" if "2D" in pic_run else "3D"
        color = COLORS[0] if "2D" in pic_run else COLORS[1]

        ax.plot(tenergy, ene_magnetic, linewidth=1, linestyle='--', color=color)
        ax.plot(tenergy, kene_e, linewidth=1, linestyle='-', color=color)
        ax.plot(tenergy, kene_i, linewidth=1, linestyle='-.', color=color)

    ax.text(0.8, 0.9, "2D", color=COLORS[0], fontsize=12,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes)
    ax.text(0.9, 0.9, "3D", color=COLORS[1], fontsize=12,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes)
    ax.text(0.98, 0.55, "magnetic", color='k', fontsize=12,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='right', verticalalignment='bottom',
            transform=ax.transAxes)
    ax.text(0.98, 0.22, "ion", color='k', fontsize=12,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='right', verticalalignment='bottom',
            transform=ax.transAxes)
    ax.text(0.98, 0.04, "electron", color='k', fontsize=12,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='right', verticalalignment='bottom',
            transform=ax.transAxes)
    ax.tick_params(bottom=True, top=False, left=True, right=True)
    ax.set_xlim([0, 400])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel(r'$t\Omega_{ci}$', fontsize=12)
    ax.set_ylabel(r'$\text{Energy}/\varepsilon_{B0}$', fontsize=12)
    ax.tick_params(labelsize=10)

    fdir = '../img/cori_3d/'
    mkdir_p(fdir)
    fname = fdir + 'econv.pdf'
    fig.savefig(fname)
    plt.show()


def plot_spectrum(plot_config):
    """Plot spectrum for all time frames

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    root_dir = "/net/scratch3/xiaocanli/reconnection/Cori_runs/"
    pic_runs = ["2D-Lx150-bg0.2-150ppc-16KNL", "3D-Lx150-bg0.2-150ppc-2048KNL"]
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

    fdir = '../img/cori_3d/'
    mkdir_p(fdir)
    fname = fdir + species + 'spect.pdf'
    fig.savefig(fname)

    plt.show()


def plot_spectrum_pub(plot_config, show_plot=True):
    """Plot energy spectrum for all time frames for publication
    """
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    root_dir = "/net/scratch3/xiaocanli/reconnection/Cori_runs/"
    pic_runs = ["2D-Lx150-bg0.2-150ppc-16KNL", "3D-Lx150-bg0.2-150ppc-2048KNL"]
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

    tstarts = [[0, 12], [0, 20]]
    tends = [[12, 40], [20, 40]]
    angle = [[55, 60], [50, 45]]
    ypos = [[0.55, 0.61], [0.71, 0.7]]
    norms = [[2, 1], [2, 1]]
    pene1 = [[20, 25], [25, 25]]
    pene2 = [[200, 250], [250, 250]]
    cbar_ticks = [[np.linspace(tstarts[0][0], tends[0][0], 3),
                   np.linspace(tstarts[0][1], tends[0][1], 5)],
                  [np.linspace(tstarts[1][0], tends[1][0], 3),
                   np.linspace(tstarts[1][1], tends[1][1], 5)]]
    power_indices = np.zeros((2, tend - tstart + 1))
    twci = np.arange(tend - tstart + 1) * 10
    tstart0 = tstart

    fig = plt.figure(figsize=[4.8, 2.0])
    rect = [[0.1, 0.2, 0.4, 0.75],
            [0.16, 0.35, 0.15, 0.25]]
    for irun, pic_run in enumerate(pic_runs):
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        for i in range(2):
            tstart, tend = tstarts[irun][i], tends[irun][i]
            nframes = tend - tstart + 1
            if not (irun == 0 and i == 1):
                ax = fig.add_axes(rect[i])
            for tframe in range(tstart, tend + 1):
                print("Time frame: %d" % tframe)
                pic_run_dir = root_dir + pic_run + "/"
                tindex = pic_info.fields_interval * tframe
                fname = (pic_run_dir + "spectrum_combined/spectrum_" +
                         species + "_" + str(tindex) + ".dat")
                spect = np.fromfile(fname, dtype=np.float32)
                espect = spect[3:] / debins
                color = plt.cm.Spectral_r((tframe - tstart)/float(nframes), 1)
                if i == 0:
                    ax.loglog(ebins_mid, espect, linewidth=1, color=color)
                if tframe > 5:
                    if irun == 0:
                        if i == 0:
                            pindex1, ene = find_nearest(ebins_mid, 25)
                        else:
                            pindex1, ene = find_nearest(ebins_mid, 50)
                        pindex2, ene = find_nearest(ebins_mid, 100)
                    else:
                        pindex1, ene = find_nearest(ebins_mid, 30)
                        pindex2, ene = find_nearest(ebins_mid, 100)
                    popt, pcov = curve_fit(fitting_funcs.func_line,
                                           np.log10(ebins_mid[pindex1:pindex2]),
                                           np.log10(espect[pindex1:pindex2]))
                    fpower = fitting_funcs.func_line(np.log10(ebins_mid), popt[0], popt[1])
                    fpower = 10**fpower
                    power_indices[irun, tframe - tstart0] = popt[0]
                if tframe == tend and species == 'e' and i == 0:
                    power_index = "{%0.1f}" % popt[0]
                    pname = r'$\propto \varepsilon^{' + power_index + '}$'
                    fsize = 10 if i == 0 else 6
                    ax.text(0.62, ypos[irun][i], pname, color='k', fontsize=fsize,
                            rotation=-angle[irun][i],
                            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                            horizontalalignment='left', verticalalignment='center',
                            transform=ax.transAxes)
                    pindex1, ene = find_nearest(ebins_mid, 50)
                    pnorm = espect[pindex1] / fpower[pindex1]
                    fpower *= pnorm * norms[irun][i]
                    pindex1, ene = find_nearest(ebins_mid, pene1[irun][i])
                    pindex2, ene = find_nearest(ebins_mid, pene2[irun][i])
                    ax.loglog(ebins_mid[pindex1:pindex2], fpower[pindex1:pindex2],
                              linewidth=0.5, linestyle='--', color='k')
            if i == 0:
                if irun == 0:
                    ax.set_ylabel(r'$f(\varepsilon)$', fontsize=10)
                ax.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$', fontsize=10)
            ax.tick_params(bottom=True, top=True, left=True, right=True)
            ax.tick_params(axis='x', which='minor', direction='in', top='off')
            ax.tick_params(axis='x', which='major', direction='in')
            ax.tick_params(axis='y', which='minor', direction='in')
            ax.tick_params(axis='y', which='major', direction='in')
            if i == 0:
                ax.set_xlim([1E-1, 1E3])
                if irun == 0:
                    ax.set_ylim([1E0, 2E9])
                    ax.set_yticks(np.logspace(0, 9, num=4))
                else:
                    ax.set_ylim([1E0, 2E12])
                    ax.set_yticks(np.logspace(0, 12, num=4))

            # Power-law indices
            if i == 1 and irun == 1:
                print(power_indices)
                ax.plot(twci[6:], -power_indices[0, 6:], color='k',
                        linestyle='--', linewidth=1)
                ax.plot(twci[6:], -power_indices[1, 6:], color='k',
                        linestyle='-', linewidth=1)
                ax.set_ylabel(r'$p$', fontsize=8)
                ax.set_xlabel(r'$t\Omega_{ci}$', fontsize=8)
                ax.set_xlim([0, 400])
                ax.tick_params(labelsize=6)
                ax.text(0.5, 0.7, '2D', color='k', fontsize=8,
                        bbox=dict(facecolor='none', alpha=1.0,
                                  edgecolor='none', pad=10.0),
                        horizontalalignment='right', verticalalignment='center',
                        transform=ax.transAxes)
                ax.text(0.9, 0.52, '3D', color='k', fontsize=8,
                        bbox=dict(facecolor='none', alpha=1.0,
                                  edgecolor='none', pad=10.0),
                        horizontalalignment='right', verticalalignment='center',
                        transform=ax.transAxes)
            else:
                ax.tick_params(labelsize=8)

            if i == 0:
                pdim = "2D" if "2D" in pic_run else "3D"
                # pdim = "(a) 2D" if "2D" in pic_run else "(b) 3D"
                ax.text(0.97, 0.9, pdim, color='k', fontsize=10,
                        bbox=dict(facecolor='none', alpha=1.0,
                                  edgecolor='none', pad=10.0),
                        horizontalalignment='right', verticalalignment='center',
                        transform=ax.transAxes)
                rect_cbar = np.copy(rect[i])
                rect_cbar[0] = rect[i][0] + 0.02
                rect_cbar[1] = rect[i][1] + rect[i][3] * 0.58
                rect_cbar[2] = 0.02
                rect_cbar[3] = rect[i][3] * 0.3
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
                cbar.ax.tick_params(labelsize=6)
                cax.yaxis.set_ticks_position('right')
                cax.yaxis.set_label_position('right')

            rect[i][0] += rect[0][2] + 0.07
    fdir = '../img/cori_3d/espect/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + "espect_" + species + ".pdf"
    fig.savefig(fname)
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_spectrum_single(plot_config, show_plot=True):
    """Plot spectrum for each time frame

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    tstart = plot_config["tstart"]
    tend = plot_config["tend"]
    root_dir = "/net/scratch3/xiaocanli/reconnection/Cori_runs/"
    pic_runs = ["2D-Lx150-bg0.2-150ppc-16KNL", "3D-Lx150-bg0.2-150ppc-2048KNL"]
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
            ax.loglog(ebins, spect[3:] / fnorm[irun], linewidth=2,
                      color=COLORS[irun])
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
        text = str(dt_particles * tframe) + r'$\Omega_{ci}^{-1}$'
        ax.text(0.98, 0.9, text, color='k', fontsize=32,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='right', verticalalignment='center',
                transform=ax.transAxes)
        fdir = '../img/cori_3d/spectrum_single/'
        mkdir_p(fdir)
        fname = fdir + "spect_" + species + "_" + str(tframe) + ".pdf"
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
    fdir = "../img/cori_3d/spect_local/"
    mkdir_p(fdir)
    fname = (fdir + "spect_local_" + species + "_" + str(tframe) +
             "_ix" + str(ix) + "_iz" + str(iz) + ".pdf")
    fig.savefig(fname)
    plt.show()


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
        plt.close()
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
    particle_interval = 10.0  # in 1/wci

    fig = plt.figure(figsize=[7, 5])
    w1, h1 = 0.83, 0.8
    xs, ys = 0.96 - w1, 0.96 - h1
    ax = fig.add_axes([xs, ys, w1, h1])
    tindex = pic_info.fields_interval * tframe
    fname = (pic_run_dir + "spectrum_combined/spectrum_" +
             species + "_" + str(tindex) + ".dat")
    spect = np.fromfile(fname, dtype=np.float32)
    pspect = spect[3:] / dpbins
    ax.loglog(pbins_mid, pspect, linewidth=1, color='k')
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
    ax.set_ylim([1E0, 1E12])

    ax.set_yticks(np.logspace(0, 12, num=7))
    ax.tick_params(labelsize=16)
    text = str(particle_interval * tframe) + r'$\Omega_{ci}^{-1}$'
    ax.text(0.98, 0.9, text, color='k', fontsize=20,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='right', verticalalignment='center',
            transform=ax.transAxes)
    fdir = '../img/cori_3d/pspect/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + "pspect_" + species + "_" + str(tframe) + ".pdf"
    fig.savefig(fname)
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_momentum_spectrum_all(plot_config, show_plot=True):
    """Plot momentum spectrum for all time frames
    """
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    root_dir = "/net/scratch3/xiaocanli/reconnection/Cori_runs/"
    pic_runs = ["2D-Lx150-bg0.2-150ppc-16KNL", "3D-Lx150-bg0.2-150ppc-2048KNL"]
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
    fdir = '../img/cori_3d/pspect/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + "pspect_" + species + ".pdf"
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
    root_dir = "/net/scratch3/xiaocanli/reconnection/Cori_runs/"
    pic_runs = ["2D-Lx150-bg0.2-150ppc-16KNL", "3D-Lx150-bg0.2-150ppc-2048KNL"]
    tframes = np.asarray([0, 4, 8, 12, 20])
    fig = plt.figure(figsize=[7, 5])
    rect = [0.14, 0.14, 0.8, 0.8]
    ax = fig.add_axes(rect)
    colors = np.copy(COLORS)
    colors[5] = colors[6]
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
            ax.loglog(ebins, spect[3:], linewidth=2,
                      linestyle=lstyle, color=colors[iframe])

    fpower =  2E12 * ebins**-4
    power_index = "{%0.1f}" % -4.0
    pname = r'$\sim \varepsilon^{' + power_index + '}$'
    ax.loglog(ebins[558:658], fpower[558:658], linewidth=1, color='k')
    ax.text(0.85, 0.7, pname, color='k', fontsize=20,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='right', verticalalignment='center',
            transform=ax.transAxes)

    ax.plot([0, 10], [0, 0], linestyle="--", color='k',
            linewidth=2, label='2D')
    ax.plot([0, 10], [0, 0], linestyle="-", color='k',
            linewidth=2, label='3D')
    ax.legend(loc=1, prop={'size': 20}, ncol=1,
              shadow=False, fancybox=False, frameon=False)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top='on')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlim([1E-1, 1E3])
    ax.set_ylim([1E-3, 1E9])
    ax.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$', fontsize=20)
    ax.set_ylabel(r'$f(\varepsilon)$', fontsize=20)
    ax.tick_params(labelsize=16)

    text1 = r'$t\Omega_{ci}=0$'
    ax.text(0.5, 0.02, text1, color=colors[0], fontsize=16,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='bottom',
            transform=ax.transAxes)
    ax.text(0.64, 0.02, r'$40$', color=colors[1], fontsize=16,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='bottom',
            transform=ax.transAxes)
    ax.text(0.85, 0.02, r'$80$', color=colors[2], fontsize=16,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='bottom',
            transform=ax.transAxes)
    ax.text(0.928, 0.02, r'$120$', color=colors[3], fontsize=16, rotation=-80,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='bottom',
            transform=ax.transAxes)
    ax.text(0.97, 0.02, r'$200$', color=colors[4], fontsize=16, rotation=-80,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='bottom',
            transform=ax.transAxes)

    fthermal = fit_thermal_core(ebins, spect[3:])
    fnonthermal = spect[3:] - fthermal
    rect = [0.2, 0.25, 0.31, 0.4]
    ax1 = fig.add_axes(rect)
    ax1.loglog(ebins, spect[3:], linewidth=3,
               linestyle=lstyle, color=colors[iframe])
    ax1.loglog(ebins, fthermal, linewidth=1, linestyle='--', color='k')
    ax1.loglog(ebins, fnonthermal, linewidth=1, linestyle='-',
               color=colors[1])
    nacc, eacc = accumulated_particle_info(ebins, spect[3:])
    nacc_thermal, eacc_thermal = accumulated_particle_info(ebins, fthermal)
    nacc, eacc = accumulated_particle_info(ebins, spect[3:])
    nacc_nthermal, eacc_nthermal = accumulated_particle_info(ebins, fnonthermal)
    print("Thermal and non-thermal number fraction: %0.2f, %0.2f" %
          (nacc_thermal[-1]/nacc[-1], nacc_nthermal[-1]/nacc[-1]))
    print("Thermal and non-thermal energy fraction: %0.2f, %0.2f" %
          (eacc_thermal[-1]/eacc[-1], eacc_nthermal[-1]/eacc[-1]))
    ax1.text(0.02, 0.22, 'all particles', color=colors[iframe], fontsize=12,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='left', verticalalignment='bottom',
             transform=ax1.transAxes)
    ax1.text(0.02, 0.12, 'thermal', color='k', fontsize=12,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='left', verticalalignment='bottom',
             transform=ax1.transAxes)
    ax1.text(0.02, 0.02, 'non-thermal', color=colors[1], fontsize=12,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='left', verticalalignment='bottom',
             transform=ax1.transAxes)
    ax1.set_xlim([1E-1, 1E3])
    ax1.set_ylim([1E-3, 1E9])

    fdir = '../img/cori_3d/'
    mkdir_p(fdir)
    fname = fdir + species + 'spect_32.pdf'
    fig.savefig(fname)

    plt.show()


def plot_jslice(plot_config):
    """Plot slices of current density
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
    xmin, xmax = 0, pic_info.lx_di
    ymin, ymax = -pic_info.ly_di * 0.5, pic_info.ly_di * 0.5
    zmin, zmax = -pic_info.lz_di * 0.5, pic_info.lz_di * 0.5
    jmin, jmax = 0.0, 0.4

    fname = pic_run_dir + "data-smooth/absJ_" + str(tindex) + ".gda"
    absj = np.fromfile(fname, dtype=np.float32)
    absj = absj.reshape((nzr, nyr, nxr))

    fdir = '../img/cori_3d/absJ/tframe_' + str(tframe) + '/'
    mkdir_p(fdir)

    for iz in midz:
        print("z-slice %d" % iz)
        fig = plt.figure(figsize=[9, 4])
        rect = [0.10, 0.16, 0.75, 0.8]
        ax = fig.add_axes(rect)
        p1 = ax.imshow(absj[iz, :, :], extent=[xmin, xmax, ymin, ymax],
                       vmin=jmin, vmax=jmax,
                       cmap=plt.cm.coolwarm, aspect='auto',
                       origin='lower', interpolation='bicubic')
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')
        ax.set_xlabel(r'$x/d_i$', fontsize=20)
        ax.set_ylabel(r'$y/d_i$', fontsize=20)
        ax.tick_params(labelsize=16)

        rect_cbar = np.copy(rect)
        rect_cbar[0] += rect[2] + 0.02
        rect_cbar[2] = 0.02
        cbar_ax = fig.add_axes(rect_cbar)
        cbar = fig.colorbar(p1, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=16)
        cbar_ax.set_ylabel(r'$|J|$', fontsize=24)
        iz_str = str(iz).zfill(4)
        fname = fdir + 'absJ_xy_' + str(tframe) + "_" + iz_str + ".jpg"
        fig.savefig(fname, dpi=200)
        plt.close()

    for iy in midy:
        print("y-slice %d" % iy)
        fig = plt.figure(figsize=[9, 4])
        rect = [0.10, 0.16, 0.75, 0.8]
        ax = fig.add_axes(rect)
        p1 = ax.imshow(absj[:, iy, :], extent=[xmin, xmax, zmin, zmax],
                       vmin=jmin, vmax=jmax,
                       cmap=plt.cm.coolwarm, aspect='auto',
                       origin='lower', interpolation='bicubic')
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')
        ax.set_xlabel(r'$x/d_i$', fontsize=20)
        ax.set_ylabel(r'$z/d_i$', fontsize=20)
        ax.tick_params(labelsize=16)

        rect_cbar = np.copy(rect)
        rect_cbar[0] += rect[2] + 0.02
        rect_cbar[2] = 0.02
        cbar_ax = fig.add_axes(rect_cbar)
        cbar = fig.colorbar(p1, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=16)
        cbar_ax.set_ylabel(r'$|J|$', fontsize=24)
        iy_str = str(iy).zfill(4)
        fname = fdir + 'absJ_xz_' + str(tframe) + "_" + iy_str + ".jpg"
        fig.savefig(fname, dpi=200)
        plt.close()

    for ix in midx:
        print("x-slice %d" % ix)
        fig = plt.figure(figsize=[7, 5])
        rect = [0.12, 0.16, 0.70, 0.8]
        ax = fig.add_axes(rect)
        p1 = ax.imshow(absj[:, :, ix], extent=[ymin, ymax, zmin, zmax],
                       vmin=jmin, vmax=jmax,
                       cmap=plt.cm.coolwarm, aspect='auto',
                       origin='lower', interpolation='bicubic')
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')
        ax.set_xlabel(r'$y/d_i$', fontsize=20)
        ax.set_ylabel(r'$z/d_i$', fontsize=20)
        ax.tick_params(labelsize=16)

        rect_cbar = np.copy(rect)
        rect_cbar[0] += rect[2] + 0.02
        rect_cbar[2] = 0.02
        cbar_ax = fig.add_axes(rect_cbar)
        cbar = fig.colorbar(p1, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=16)
        cbar_ax.set_ylabel(r'$|J|$', fontsize=24)
        ix_str = str(ix).zfill(4)
        fname = fdir + 'absJ_yz_' + str(tframe) + "_" + ix_str + ".jpg"
        fig.savefig(fname, dpi=200)
        plt.close()

    # plt.show()


def plot_absj_2d(plot_config, show_plot=True):
    """Plot current density of the 2D simulation
    """
    tframe = plot_config["tframe"]
    pic_run = "2D-Lx150-bg0.2-150ppc-16KNL"
    root_dir = "/net/scratch3/xiaocanli/reconnection/Cori_runs/"
    pic_run_dir = root_dir + pic_run + "/"
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
    nx, nz = pic_info.nx, pic_info.nz
    ntf = pic_info.ntf
    xmin, xmax = 0, pic_info.lx_di
    ymin, ymax = -pic_info.ly_di * 0.5, pic_info.ly_di * 0.5
    zmin, zmax = -pic_info.lz_di * 0.5, pic_info.lz_di * 0.5
    jmin, jmax = 0.0, 0.4

    fname = pic_run_dir + "data/absJ.gda"
    absj = np.fromfile(fname, dtype=np.float32)
    absj = absj.reshape((ntf, nz, nx))

    fdir = '../img/cori_3d/absJ_2d/'
    mkdir_p(fdir)

    fig = plt.figure(figsize=[3.25, 1.5])
    rect = [0.18, 0.28, 0.68, 0.65]
    ax = fig.add_axes(rect)
    p1 = ax.imshow(absj[tframe, :, :],
                   extent=[xmin, xmax, zmin, zmax],
                   vmin=jmin, vmax=jmax,
                   cmap=plt.cm.coolwarm, aspect='auto',
                   origin='lower', interpolation='bicubic')
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlabel(r'$x/d_i$', fontsize=12)
    ax.set_ylabel(r'$z/d_i$', fontsize=12)
    ax.tick_params(labelsize=10)
    text1 = r'$|\boldsymbol{J}|/J_0$'
    ax.text(0.02, 0.85, text1, color='w', fontsize=12,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)

    rect_cbar = np.copy(rect)
    rect_cbar[0] += rect[2] + 0.02
    rect_cbar[2] = 0.02
    cbar_ax = fig.add_axes(rect_cbar)
    cbar_ax.tick_params(axis='y', which='major', direction='in')
    cbar = fig.colorbar(p1, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_ticks((np.linspace(0, 0.4, 5)))
    fname = fdir + 'absJ_' + str(tframe) + ".jpg"
    fig.savefig(fname, dpi=300)
    fname = fdir + 'absJ_' + str(tframe) + ".pdf"
    fig.savefig(fname, dpi=300)
    if show_plot:
        plt.show()
    else:
        plt.close()


def absj_2d_pub(plot_config, show_plot=True):
    """Plot current density of the 2D simulation for publication
    """
    tframe = plot_config["tframe"]
    pic_run = "2D-Lx150-bg0.2-150ppc-16KNL"
    root_dir = "/net/scratch3/xiaocanli/reconnection/Cori_runs/"
    pic_run_dir = root_dir + pic_run + "/"
    species = plot_config["species"]
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
    nx, nz = pic_info.nx, pic_info.nz
    ntf = pic_info.ntf
    xmin, xmax = 0, pic_info.lx_di
    ymin, ymax = -pic_info.ly_di * 0.5, pic_info.ly_di * 0.5
    zmin, zmax = -pic_info.lz_di * 0.5, pic_info.lz_di * 0.5
    xgrid = np.linspace(xmin, xmax, pic_info.nx)
    zgrid = np.linspace(zmin, zmax, pic_info.nz)
    jmin, jmax = 0.0, 0.4

    fname = pic_run_dir + "data/absJ.gda"
    absj = np.fromfile(fname, dtype=np.float32)
    absj = absj.reshape((ntf, nz, nx))

    fname = pic_run_dir + "data/Ay.gda"
    Ay = np.fromfile(fname, dtype=np.float32)
    Ay = Ay.reshape((ntf, nz, nx))

    fdir = '../img/cori_3d/absJ_2d/'
    mkdir_p(fdir)

    colormap = plt.cm.coolwarm
    tframe1, tframe2 = 8, 20
    fig = plt.figure(figsize=[3.5, 2.8])
    rect = [0.09, 0.55, 0.75, 0.41]
    hgap, vgap = 0.05, 0.02
    ax1 = fig.add_axes(rect)
    p1 = ax1.imshow(absj[tframe1, :, :],
                   extent=[xmin, xmax, zmin, zmax],
                   vmin=jmin, vmax=jmax,
                   cmap=colormap, aspect='auto',
                   origin='lower', interpolation='bicubic')
    ax1.contour(xgrid, zgrid, Ay[tframe1, :, :], colors='k', linewidths=0.5)
    ax1.tick_params(bottom=True, top=True, left=True, right=True)
    ax1.tick_params(axis='x', which='minor', direction='in')
    ax1.tick_params(axis='x', which='major', direction='in')
    ax1.tick_params(axis='y', which='minor', direction='in')
    ax1.tick_params(axis='y', which='major', direction='in')
    ax1.tick_params(axis='x', labelbottom='off')
    ax1.tick_params(axis='y', labelleft='off')
    ax1.tick_params(labelsize=10)
    twci = math.ceil((tframe1 * pic_info.dt_fields) / 0.1) * 0.1
    text1 = r'$t\Omega_{ci}=' + ("{%0.0f}" % twci) + '$'
    ax1.text(0.02, 0.85, text1, color='w', fontsize=10,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='left', verticalalignment='center',
             transform=ax1.transAxes)
    ax1.annotate(s='', xy=(-0.02, -0.02), xytext=(-0.02, 1.02), xycoords='axes fraction',
                 arrowprops=dict(arrowstyle='<->',
                                 linestyle='dashed', linewidth=0.5))
    ax1.text(-0.05, rect[1] + rect[3]*0.5 - 0.26, r'$L_z=62.5d_i$',
             rotation=90, color='k', fontsize=10,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='center', verticalalignment='center',
             transform=ax1.transAxes)

    rect[1] -= rect[3] + vgap
    ax2 = fig.add_axes(rect)
    p1 = ax2.imshow(absj[tframe2, :, :],
                   extent=[xmin, xmax, zmin, zmax],
                   vmin=jmin, vmax=jmax,
                   cmap=colormap, aspect='auto',
                   origin='lower', interpolation='bicubic')
    ax2.contour(xgrid, zgrid, Ay[tframe2, :, :], colors='k', linewidths=0.5)
    ax2.tick_params(bottom=True, top=True, left=True, right=True)
    ax2.tick_params(axis='x', which='minor', direction='in')
    ax2.tick_params(axis='x', which='major', direction='in')
    ax2.tick_params(axis='y', which='minor', direction='in')
    ax2.tick_params(axis='y', which='major', direction='in')
    ax2.tick_params(axis='x', labelbottom='off')
    ax2.tick_params(axis='y', labelleft='off')
    ax2.tick_params(labelsize=10)
    twci = math.ceil((tframe2 * pic_info.dt_fields) / 0.1) * 0.1
    text2 = r'$t\Omega_{ci}=' + ("{%0.0f}" % twci) + '$'
    ax2.text(0.02, 0.85, text2, color='w', fontsize=10,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='left', verticalalignment='center',
             transform=ax2.transAxes)
    ax2.annotate(s='', xy=(-0.01,-0.05), xytext=(1.02,-0.05), xycoords='axes fraction',
                 arrowprops=dict(arrowstyle='<->', linestyle='dashed', linewidth=0.5))
    ax2.text(rect[0] + rect[2]*0.5, -0.1, r'$L_x=150d_i$', color='k', fontsize=10,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='center', verticalalignment='top',
             transform=ax2.transAxes)
    ax2.annotate(s='', xy=(-0.02, -0.02), xytext=(-0.02, 1.02), xycoords='axes fraction',
                 arrowprops=dict(arrowstyle='<->', linestyle='dashed', linewidth=0.5))
    ax2.text(-0.05, rect[1] + rect[3]*0.5 + 0.18, r'$L_z=62.5d_i$',
             rotation=90, color='k', fontsize=10,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='center', verticalalignment='center',
             transform=ax2.transAxes)

    rect_cbar = np.copy(rect)
    rect_cbar[0] += rect[2] + hgap
    rect_cbar[1] = rect[1] + (rect[3] + vgap) * 0.5
    rect_cbar[2] = 0.02
    cbar_ax = fig.add_axes(rect_cbar)
    cbar_ax.tick_params(axis='y', which='major', direction='in')
    cbar = fig.colorbar(p1, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_ticks((np.linspace(0, 0.4, 5)))
    cbar_ax.set_title(r'$J/J_0$', fontsize=10)
    fname = fdir + 'absJ_' + str(tframe1) + "_" + str(tframe2) + ".pdf"
    fig.savefig(fname, dpi=300)
    if show_plot:
        plt.show()
    else:
        plt.close()


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


def plot_jslice_box(plot_config):
    """Plot slices of current density with indicated box region
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
    xslices = np.asarray([0, 13, 25, 37])
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

    fname = pic_run_dir + "data-smooth/absJ_" + str(tindex) + ".gda"
    absj = np.fromfile(fname, dtype=np.float32)
    absj = absj.reshape((nzr, nyr, nxr))

    # initial thermal distribution
    fname = (pic_run_dir + "spectrum_combined/spectrum_" + species + "_0.dat")
    spect_init = np.fromfile(fname, dtype=np.float32)
    ndata, = spect_init.shape
    spect_init[3:] /= np.gradient(ebins)
    spect_init[3:] /= (pic_info.nx * pic_info.ny * pic_info.nz / box_size**3 / 8)

    fdir = '../img/cori_3d/'
    mkdir_p(fdir)

    fig = plt.figure(figsize=[7, 6.125])
    rect = [0.09, 0.77, 0.45, 0.21]
    hgap, vgap = 0.02, 0.02
    rect1 = np.copy(rect)
    rect1[0] += rect[2] + 0.19
    rect1[2] = 0.25

    nslices = len(xslices)
    for islice, ix in enumerate(xslices):
        ax = fig.add_axes(rect)
        print("x-slice %d" % ix)
        p1 = ax.imshow(absj[:, :, midx[ix]], extent=[ymin, ymax, zmin, zmax],
                       vmin=jmin, vmax=jmax,
                       cmap=plt.cm.binary, aspect='auto',
                       origin='lower', interpolation='bicubic')
        ax.set_ylim([-15, 15])
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')
        if islice == nslices - 1:
            ax.set_xlabel(r'$y/d_i$', fontsize=12)
        else:
            ax.tick_params(axis='x', labelbottom='off')
        ax.set_ylabel(r'$z/d_i$', fontsize=12)
        ax.tick_params(labelsize=10)

        text1 = r'$x=' + ("{%0.1f}" % xdi[islice]) + 'd_i$'
        ax.text(0.02, 0.85, text1, color='k', fontsize=12,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)
        if islice == 0:
            for iy in range(len(yboxes)):
                color = COLORS[iy]
                plot_box([ydi[iy], z1_di], dx_di * box_size, ax, color=color)
        else:
            for iy in range(len(yboxes)):
                color = COLORS[iy]
                plot_box([ydi[iy], z0_di], dx_di * box_size, ax, color=color)

        ax1 = fig.add_axes(rect1)
        ax1.set_prop_cycle('color', COLORS)
        fname = (pic_run_dir + "spectrum_reduced/spectrum_" +
                 species + "_" + str(tindex) + ".dat")
        spect = np.fromfile(fname, dtype=np.float32)
        sz, = spect.shape
        npoints = sz//ndata
        spect = spect.reshape((npoints, ndata))
        print("Spectral data size: %d, %d" % (npoints, ndata))
        spect[:, 3:] /= np.gradient(ebins)
        if islice == 0:
            for iy in yboxes:
                cindex = z1 * nslicex * nslicey + iy * nslicex  + ix
                ax1.loglog(ebins, spect[cindex, 3:], linewidth=1)
        else:
            for iy in yboxes:
                cindex = z0 * nslicex * nslicey + iy * nslicex  + ix
                ax1.loglog(ebins, spect[cindex, 3:], linewidth=1)
        if islice == nslices - 1:
            ax1.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$',
                          fontsize=12)
        else:
            ax1.tick_params(axis='x', labelbottom='off')
        ax1.loglog(ebins, spect_init[3:], linewidth=1, linestyle='--',
                   color='k', label='Initial')
        pindex = -4.0
        power_index = "{%0.1f}" % pindex
        pname = r'$\propto \varepsilon^{' + power_index + '}$'
        fpower = 1E12*ebins**pindex
        if species == 'e':
            es, ee = 588, 688
        else:
            es, ee = 438, 538
        if species == 'e':
            ax1.loglog(ebins[es:ee], fpower[es:ee], color='k', linewidth=1)
            ax1.text(0.92, 0.58, pname, color='k', fontsize=12, rotation=-60,
                     bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                     horizontalalignment='right', verticalalignment='center',
                     transform=ax1.transAxes)
            ax1.text(0.5, 0.05, "Initial", color='k', fontsize=12,
                     bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                     horizontalalignment='right', verticalalignment='center',
                     transform=ax1.transAxes)
        ax1.set_xlim([1E-1, 1E3])
        ax1.set_ylim([1E-1, 2E7])
        ax1.set_ylabel(r'$f(\varepsilon)$', fontsize=12)
        ax1.tick_params(labelsize=10)

        rect[1] -= rect[3] + vgap
        rect1[1] -= rect1[3] + vgap

    rect[1] += (rect[3] + vgap) * 2
    rect_cbar = np.copy(rect)
    rect_cbar[0] = rect[0] + rect[2] + hgap
    rect_cbar[2] = 0.02
    rect_cbar[3] = rect[3] * 2  + vgap * 1
    cbar_ax = fig.add_axes(rect_cbar)
    cbar = fig.colorbar(p1, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=10)
    cbar_ax.set_title(r'$J/J_0$', fontsize=12)
    ix_str = str(ix).zfill(4)

    fname = fdir + 'absJ_yz_boxes.jpg'
    fig.savefig(fname, dpi=300)
    fname = fdir + 'absJ_yz_boxes.pdf'
    fig.savefig(fname, dpi=300)
    plt.show()


def plot_absj_spect(plot_config):
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
    dx_di = pic_info.dx_di * 2  # smoothed data
    dy_di = pic_info.dy_di * 2
    dz_di = pic_info.dy_di * 2
    xmin, xmax = 0, pic_info.lx_di
    ymin, ymax = -pic_info.ly_di * 0.5, pic_info.ly_di * 0.5
    zmin, zmax = -pic_info.lz_di * 0.5, pic_info.lz_di * 0.5
    jmin, jmax = 0.0, 0.4
    xcut = np.argwhere(midx==947)[0][0]
    ycut = np.argwhere(midy==347)[0][0]
    zcut = 13
    xde = (midx[xcut] * dx_di + xmin) * math.sqrt(pic_info.mime)
    yde = (midy[ycut] * dy_di + ymin) * math.sqrt(pic_info.mime)
    zde = (midz[zcut] * dz_di + zmin) * math.sqrt(pic_info.mime)
    print(xde, yde, zde)
    print(dx_di * 24)

    xcuts = np.arange(xcut+10, nslicex-5, 4)
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
    ax.tick_params(axis='x', labelbottom='off')
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
    ax.tick_params(axis='y', labelleft='off')
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

    fig = plt.figure(figsize=[2.4, 2])
    rect = [0.15, 0.2, 0.8, 0.75]
    ax = fig.add_axes(rect)
    ax.set_prop_cycle('color', COLORS)
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
    ax.loglog(ebins, spect_init[3:], linewidth=1, linestyle='--',
              color='k', label='initial')
    pindex = -4.0
    power_index = "{%0.1f}" % pindex
    pname = r'$\propto \varepsilon^{' + power_index + '}$'
    fpower = 1E12*ebins**pindex
    if species == 'e':
        es, ee = 568, 668
    else:
        es, ee = 438, 538
    if species == 'e':
        ax.loglog(ebins[es:ee], fpower[es:ee], color='k', linewidth=1)
        ax.text(0.92, 0.58, pname, color='k', fontsize=10, rotation=-60,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='right', verticalalignment='center',
                transform=ax.transAxes)
        ax.text(0.5, 0.07, "initial", color='k', fontsize=10,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='right', verticalalignment='center',
                transform=ax.transAxes)
    ax.text(0.97, 0.9, '(c)', color='k', fontsize=10,
            bbox=dict(facecolor='none', alpha=1.0,
                      edgecolor='none', pad=10.0),
            horizontalalignment='right', verticalalignment='center',
            transform=ax.transAxes)
    ax.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$', fontsize=10)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.tick_params(labelsize=8)
    ax.set_xlim([1E-1, 1E3])
    ax.set_ylim([1E0, 2E7])

    fdir = '../img/cori_3d/espect/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + "espect_local_" + species + ".pdf"
    fig.savefig(fname)
    plt.show()


def calc_absj_dist(plot_config):
    """calculate the current density distribution
    """
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    tframe = plot_config["tframe"]
    species = plot_config["species"]
    tframe = plot_config["tframe"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    jmin, jmax = 0.0, 2.0
    nbins = 200
    jbins = np.linspace(jmin, jmax, nbins + 1)
    jbins_mid = (jbins[:-1] + jbins[1:]) * 0.5

    tindex = pic_info.particle_interval * tframe
    fname = pic_run_dir + "data-smooth/absJ_" + str(tindex) + ".gda"
    absj = np.fromfile(fname, dtype=np.float32)
    jdist, bin_edges = np.histogram(absj, bins=jbins)

    jarray = np.vstack((jbins_mid, jdist))

    fdir = '../data/cori_3d/absj_dist/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + "absj_dist_" + str(tframe) + ".dat"
    jarray.tofile(fname)


def plot_absj_dist(plot_config):
    """plot the current density distribution
    """
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    tframe = plot_config["tframe"]
    tstart = plot_config["tstart"]
    tend = plot_config["tend"]
    species = plot_config["species"]
    nframes = tend - tstart + 1
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    fdir = '../data/cori_3d/absj_dist/' + pic_run + '/'
    fig = plt.figure(figsize=[7, 5])
    rect = [0.12, 0.15, 0.8, 0.8]
    ax = fig.add_axes(rect)
    for tframe in range(tstart, tend + 1):
        fname = fdir + "absj_dist_" + str(tframe) + ".dat"
        jarray = np.fromfile(fname)
        nbins = jarray.shape[0] // 2
        jbins = jarray[:nbins]
        jdist = jarray[nbins:]
        color = plt.cm.seismic((tframe - tstart)/float(nframes), 1)
        ax.semilogy(jbins, jdist, color=color)
    plt.show()


def calc_abse_dist(plot_config):
    """calculate the electric field distribution
    """
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    tframe = plot_config["tframe"]
    species = plot_config["species"]
    tframe = plot_config["tframe"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)

    tindex = pic_info.particle_interval * tframe
    fname = pic_run_dir + "data-smooth/ex_" + str(tindex) + ".gda"
    ex = np.fromfile(fname, dtype=np.float32)
    fname = pic_run_dir + "data-smooth/ey_" + str(tindex) + ".gda"
    ey = np.fromfile(fname, dtype=np.float32)
    fname = pic_run_dir + "data-smooth/ez_" + str(tindex) + ".gda"
    ez = np.fromfile(fname, dtype=np.float32)
    abse = np.sqrt(ex**2 + ey**2 + ez**2)
    emin, emax = 0.0, 0.3
    nbins = 300
    ebins = np.linspace(emin, emax, nbins + 1)
    ebins_mid = (ebins[:-1] + ebins[1:]) * 0.5
    edist, bin_edges = np.histogram(abse, bins=ebins)

    earray = np.vstack((ebins_mid, edist))

    fdir = '../data/cori_3d/abse_dist/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + "abse_dist_" + str(tframe) + ".dat"
    earray.tofile(fname)


def plot_abse_dist(plot_config):
    """plot the electric field distribution
    """
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    tframe = plot_config["tframe"]
    tstart = plot_config["tstart"]
    tend = plot_config["tend"]
    species = plot_config["species"]
    nframes = tend - tstart + 1
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    fdir = '../data/cori_3d/abse_dist/' + pic_run + '/'
    fig = plt.figure(figsize=[7, 5])
    rect = [0.12, 0.15, 0.8, 0.8]
    ax = fig.add_axes(rect)
    nacc = np.zeros(nframes)
    for tframe in range(tstart, tend + 1):
        fname = fdir + "abse_dist_" + str(tframe) + ".dat"
        earray = np.fromfile(fname)
        nbins = earray.shape[0] // 2
        ebins = earray[:nbins]
        edist = earray[nbins:]
        color = plt.cm.seismic((tframe - tstart)/float(nframes), 1)
        ax.semilogy(ebins, edist, color=color)
        nacc[tframe - tstart] = np.sum(edist[ebins > 0.06])

    fig = plt.figure(figsize=[7, 5])
    rect = [0.12, 0.15, 0.8, 0.8]
    ax = fig.add_axes(rect)
    ax.plot(nacc)

    plt.show()


def rho_profile(plot_config, show_plot=True):
    """Plot number density profile
    """
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    species = plot_config["species"]
    tframe = plot_config["tframe"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    nxr, nyr, nzr = pic_info.nx//2, pic_info.ny//2, pic_info.nz//2
    xmin, xmax = 0, pic_info.lx_di
    ymin, ymax = -pic_info.ly_di * 0.5, pic_info.ly_di * 0.5
    zmin, zmax = -pic_info.lz_di * 0.5, pic_info.lz_di * 0.5
    tindex = tframe * pic_info.fields_interval
    fname = pic_run_dir + "data-smooth/n" + species + "_" + str(tindex) + ".gda"
    nrho = np.fromfile(fname, dtype=np.float32)
    nrho = nrho.reshape((nzr, nyr, nxr))

    nrho_xz = np.mean(nrho, axis=1)
    fig = plt.figure(figsize=[7, 3.5])
    rect = [0.12, 0.15, 0.75, 0.8]
    ax = fig.add_axes(rect)
    p1 = ax.imshow(nrho_xz, extent=[xmin, xmax, zmin, zmax],
                   vmin=0.5, vmax=2.2,
                   cmap=plt.cm.viridis, aspect='auto',
                   origin='lower', interpolation='bicubic')
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlabel(r'$x/d_i$', fontsize=16)
    ax.set_ylabel(r'$z/d_i$', fontsize=16)
    ax.tick_params(labelsize=12)

    rect_cbar = np.copy(rect)
    rect_cbar[0] += rect[2] + 0.02
    rect_cbar[2] = 0.02
    cbar_ax = fig.add_axes(rect_cbar)
    cbar_ax.tick_params(axis='y', which='major', direction='in')
    cbar = fig.colorbar(p1, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=12)
    label=r'$n_' + species + '/n_0$'
    cbar_ax.set_ylabel(label, fontsize=16)

    fdir = '../img/cori_3d/rho_xz_3d/'
    mkdir_p(fdir)
    fname = fdir + 'rho_xz_' + str(tframe) + '.pdf'
    fig.savefig(fname, dpi=300)

    if show_plot:
        plt.show()
    else:
        plt.close()


def absb_profile(plot_config, show_plot=True):
    """Plot the profile of the magnitude of magnetic field
    """
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    species = plot_config["species"]
    tframe = plot_config["tframe"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    nxr, nyr, nzr = pic_info.nx//2, pic_info.ny//2, pic_info.nz//2
    xmin, xmax = 0, pic_info.lx_di
    ymin, ymax = -pic_info.ly_di * 0.5, pic_info.ly_di * 0.5
    zmin, zmax = -pic_info.lz_di * 0.5, pic_info.lz_di * 0.5
    tindex = tframe * pic_info.fields_interval
    fname = pic_run_dir + "data-smooth/bx_" + str(tindex) + ".gda"
    bx = np.fromfile(fname, dtype=np.float32)
    fname = pic_run_dir + "data-smooth/by_" + str(tindex) + ".gda"
    by = np.fromfile(fname, dtype=np.float32)
    fname = pic_run_dir + "data-smooth/bz_" + str(tindex) + ".gda"
    bz = np.fromfile(fname, dtype=np.float32)
    absb = np.sqrt(bx**2 + by**2 + bz**2)
    absb = absb.reshape((nzr, nyr, nxr))

    absb_xz = np.mean(absb, axis=1)
    fig = plt.figure(figsize=[7, 3.5])
    rect = [0.12, 0.15, 0.75, 0.8]
    ax = fig.add_axes(rect)
    ax.plot(absb_xz[:, 0])
    # p1 = ax.imshow(absb_xz, extent=[xmin, xmax, zmin, zmax],
    #                vmin=0.5, vmax=1.5,
    #                cmap=plt.cm.viridis, aspect='auto',
    #                origin='lower', interpolation='bicubic')
    # ax.tick_params(bottom=True, top=True, left=True, right=True)
    # ax.tick_params(axis='x', which='minor', direction='in')
    # ax.tick_params(axis='x', which='major', direction='in')
    # ax.tick_params(axis='y', which='minor', direction='in')
    # ax.tick_params(axis='y', which='major', direction='in')
    # ax.set_xlabel(r'$x/d_i$', fontsize=16)
    # ax.set_ylabel(r'$z/d_i$', fontsize=16)
    # ax.tick_params(labelsize=12)

    # rect_cbar = np.copy(rect)
    # rect_cbar[0] += rect[2] + 0.02
    # rect_cbar[2] = 0.02
    # cbar_ax = fig.add_axes(rect_cbar)
    # cbar_ax.tick_params(axis='y', which='major', direction='in')
    # cbar = fig.colorbar(p1, cax=cbar_ax)
    # cbar.ax.tick_params(labelsize=12)
    # label=r'$B/B_0$'
    # cbar_ax.set_ylabel(label, fontsize=16)

    # fdir = '../img/cori_3d/rho_xz_3d/'
    # mkdir_p(fdir)
    # fname = fdir + 'rho_xz_' + str(tframe) + '.pdf'
    # fig.savefig(fname, dpi=300)

    if show_plot:
        plt.show()
    else:
        plt.close()


def compression_factor(plot_config):
    """Calculate compression factor
    """
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    species = plot_config["species"]
    tframe = plot_config["tframe"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    nxr, nyr, nzr = pic_info.nx//2, pic_info.ny//2, pic_info.nz//2
    tindex = tframe * pic_info.fields_interval
    fname = pic_run_dir + "data-smooth/n" + species + "_" + str(tindex) + ".gda"
    nrho = np.fromfile(fname, dtype=np.float32)
    nrho = nrho.reshape((nzr, nyr, nxr))
    fname = pic_run_dir + "data-smooth/bx_" + str(tindex) + ".gda"
    bx = np.fromfile(fname, dtype=np.float32)
    bx = bx.reshape((nzr, nyr, nxr))
    fname = pic_run_dir + "data-smooth/by_" + str(tindex) + ".gda"
    by = np.fromfile(fname, dtype=np.float32)
    by = by.reshape((nzr, nyr, nxr))
    fname = pic_run_dir + "data-smooth/bz_" + str(tindex) + ".gda"
    bz = np.fromfile(fname, dtype=np.float32)
    bz = bz.reshape((nzr, nyr, nxr))

    # fname = pic_run_dir + "data-smooth/ex_" + str(tindex) + ".gda"
    # ex = np.fromfile(fname, dtype=np.float32)
    # ex = ex.reshape((nzr, nyr, nxr))
    # fname = pic_run_dir + "data-smooth/ey_" + str(tindex) + ".gda"
    # ey = np.fromfile(fname, dtype=np.float32)
    # ey = ey.reshape((nzr, nyr, nxr))
    # fname = pic_run_dir + "data-smooth/ez_" + str(tindex) + ".gda"
    # ez = np.fromfile(fname, dtype=np.float32)
    # ez = ez.reshape((nzr, nyr, nxr))

    # ib2 = 1.0 / (bx**2 + by**2 + bz**2)
    # exb_x = (ey * bz - ez * by) * ib2
    # exb_y = (ez * bx - ex * bz) * ib2
    # exb_z = (ex * by - ey * bx) * ib2

    # del ex, ey, ez, bx, by, bz, ib2

    # comp = nrho * (np.gradient(exb_x, axis=2) +
    #                np.gradient(exb_y, axis=1) +
    #                np.gradient(exb_z, axis=0)) / 3
    # adve = -(exb_x * np.gradient(nrho, axis=2) +
    #          exb_y * np.gradient(nrho, axis=1) +
    #          exb_z * np.gradient(nrho, axis=0))
    # comp_factor = div0(adve, comp)
    # bins = np.linspace(-8, 8, 100)
    # hist, bin_edges = np.histogram(comp_factor, bins=bins)

    ib = 1.0 / np.sqrt(bx**2 + by**2 + bz**2)
    bx *= ib
    by *= ib
    bz *= ib
    kappax = (bx * np.gradient(bx, axis=2) +
              by * np.gradient(bx, axis=1) +
              bz * np.gradient(bx, axis=0))
    kappay = (bx * np.gradient(by, axis=2) +
              by * np.gradient(by, axis=1) +
              bz * np.gradient(by, axis=0))
    kappaz = (bx * np.gradient(bz, axis=2) +
              by * np.gradient(bz, axis=1) +
              bz * np.gradient(bz, axis=0))
    kappa = np.sqrt(kappax**2 + kappay**2 + kappaz**2)
    # kappa = np.copy(kappax)
    del kappax, kappay, kappaz
    ilen = np.gradient(nrho, axis=2) / nrho

    comp_factor = div0(ilen, kappa)
    bins = np.linspace(-8, 8, 100)
    hist, bin_edges = np.histogram(comp_factor, bins=bins)

    plt.plot(bins[:-1], hist, linewidth=2)
    # bins = np.linspace(0, 5, 100)
    # hist, bin_edges = np.histogram(nrho, bins=bins)
    # plt.plot(bins[:-1], hist, linewidth=2)

    plt.show()


def plotj_box_2d(plot_config):
    """Plot current density with indicated box region of the 2D run
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
    nslicex, nslicez = 64, 28
    box_size = 48
    box_size_h = box_size // 2
    nx, nz = pic_info.nx, pic_info.nz
    shiftz = (nz - (nslicez - 2) * box_size) // 2
    midx = np.linspace(box_size_h - 1, nx - box_size_h - 1, nslicex, dtype=int)
    midz = np.linspace(box_size_h + shiftz - 1, nz - box_size_h - shiftz - 1,
                       nslicez - 2, dtype=int)
    tframes = np.asarray([5, 10, 15, 25])
    xboxes = np.asarray([4, 12, 20, 28])
    dx_di = pic_info.dx_di
    dy_di = pic_info.dy_di
    dz_di = pic_info.dy_di
    xmin, xmax = 0, pic_info.lx_di
    ymin, ymax = -pic_info.ly_di * 0.5, pic_info.ly_di * 0.5
    zmin, zmax = -pic_info.lz_di * 0.5, pic_info.lz_di * 0.5
    jmin, jmax = 0.0, 0.4
    xdi = midx[xboxes] * dx_di + xmin
    z0 = nslicez//2 - 1
    z0_di = midz[z0] * dz_di + zmin

    fname = pic_run_dir + "data/absJ.gda"
    absj = np.fromfile(fname, dtype=np.float32)
    absj = absj.reshape((-1, nz, nx))

    fdir = '../img/cori_3d/'
    mkdir_p(fdir)

    fig = plt.figure(figsize=[10, 10])
    rect = [0.09, 0.76, 0.77, 0.21]
    hgap, vgap = 0.02, 0.02

    nframes = len(tframes)
    for iframe, tframe in enumerate(tframes):
        ax = fig.add_axes(rect)
        print("Time frame %d" % tframe)
        p1 = ax.imshow(absj[tframe, :, :], extent=[xmin, xmax, zmin, zmax],
                       vmin=jmin, vmax=jmax,
                       cmap=plt.cm.coolwarm, aspect='auto',
                       origin='lower', interpolation='bicubic')
        ax.set_ylim([-20, 20])
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')
        if iframe == nframes - 1:
            ax.set_xlabel(r'$x/d_i$', fontsize=20)
        else:
            ax.tick_params(axis='x', labelbottom='off')
        ax.set_ylabel(r'$z/d_i$', fontsize=20)
        ax.tick_params(labelsize=16)

        twci = math.ceil((tframe * pic_info.dt_fields) / 0.1) * 0.1
        text1 = r'$t\Omega_{ci}=' + ("{%0.0f}" % twci) + '$'
        ax.text(0.02, 0.85, text1, color='k', fontsize=24,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)
        for ix in range(len(xboxes)):
            plot_box([xdi[ix], z0_di], dx_di * box_size, ax, 'k')

        rect[1] -= rect[3] + vgap

    rect[1] += rect[3] + vgap
    rect_cbar = np.copy(rect)
    rect_cbar[0] = rect[0] + rect[2] + hgap
    rect_cbar[2] = 0.02
    rect_cbar[3] = rect[3] * 4  + vgap * 3
    cbar_ax = fig.add_axes(rect_cbar)
    cbar = fig.colorbar(p1, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=16)
    cbar_ax.set_ylabel(r'$|J|$', fontsize=24)

    # fname = fdir + 'absJ_yz_boxes.jpg'
    # fig.savefig(fname, dpi=200)
    plt.show()


def fluid_energization(plot_config, show_plot=True):
    """Plot fluid energization

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    pic_run_dir = plot_config["pic_run_dir"]
    pic_runs = ["2D-Lx150-bg0.2-150ppc-16KNL", "3D-Lx150-bg0.2-150ppc-2048KNL"]
    if species == 'e':
        ylim = [-0.2, 1.0]
    else:
        ylim = [-0.6, 1.7]
    fig1 = plt.figure(figsize=[7, 2.5])
    box1 = [0.09, 0.18, 0.41, 0.68]
    axs1 = []
    fig2 = plt.figure(figsize=[7, 2.5])
    box2 = [0.09, 0.18, 0.41, 0.68]
    axs2 = []
    fig3 = plt.figure(figsize=[7, 2.5])
    box3 = [0.09, 0.18, 0.41, 0.68]
    axs3 = []
    for irun, pic_run in enumerate(pic_runs):
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        if "3D" in pic_run:
            enorm = pic_info.ny
        else:
            enorm = 1.0
        tfields = pic_info.tfields
        tenergy = pic_info.tenergy
        fname = "../data/fluid_energization/" + pic_run + "/"
        fname += "emf_ptensor_" + species + '.gda'
        fluid_ene = np.fromfile(fname, dtype=np.float32)
        nvar = int(fluid_ene[0])
        nframes = int(fluid_ene[1])
        fluid_ene[2:] /= enorm
        curv_drift_dote = fluid_ene[2:nframes+2]
        bulk_curv_dote = fluid_ene[nframes+2:2*nframes+2]
        grad_drift_dote = fluid_ene[2*nframes+2:3*nframes+2]
        magnetization_dote = fluid_ene[3*nframes+2:4*nframes+2]
        comp_ene = fluid_ene[4*nframes+2:5*nframes+2]
        shear_ene = fluid_ene[5*nframes+2:6*nframes+2]
        ptensor_ene = fluid_ene[6*nframes+2:7*nframes+2]
        pgyro_ene = fluid_ene[7*nframes+2:8*nframes+2]

        fname = "../data/fluid_energization/" + pic_run + "/"
        fname += "para_perp_acc_" + species + '.gda'
        fluid_ene = np.fromfile(fname, dtype=np.float32)
        nvar = int(fluid_ene[0])
        nframes = int(fluid_ene[1])
        fluid_ene[2:] /= enorm
        acc_drift_dote_t = fluid_ene[2:nframes+2]
        acc_drift_dote_s = fluid_ene[nframes+2:2*nframes+2]
        acc_drift_dote = acc_drift_dote_t + acc_drift_dote_s
        epara_ene = fluid_ene[2*nframes+2:3*nframes+2]
        eperp_ene = fluid_ene[3*nframes+2:4*nframes+2]
        acc_drift_dote[-1] = acc_drift_dote[-2]

        jperp_dote = curv_drift_dote + grad_drift_dote + magnetization_dote
        jagy_dote = ptensor_ene - jperp_dote
        if species == 'e':
            dkene = pic_info.dkene_e
        else:
            dkene = pic_info.dkene_i
        dkene /= enorm

        ax = fig1.add_axes(box1)
        axs1.append(ax)
        COLORS = palettable.tableau.Tableau_10.mpl_colors
        ax.set_prop_cycle('color', COLORS)
        label2 = (r'$\boldsymbol{j}_{' + species + '\perp}' +
                  r'\cdot\boldsymbol{E}_\perp$')
        ax.plot(tfields, eperp_ene, linewidth=1, label=label2)
        label1 = (r'$\boldsymbol{j}_{' + species + '\parallel}' +
                  r'\cdot\boldsymbol{E}_\parallel$')
        ax.plot(tfields, epara_ene, linewidth=1, label=label1)
        label6 = r'$dK_' + species + '/dt$'
        ax.plot(tenergy, dkene, linewidth=1, label=label6)
        label3 = r'$(\nabla\cdot\tensorsym{P}_' + species + r')\cdot\boldsymbol{v}_E$'
        ax.plot(tfields, ptensor_ene, linewidth=1, label=label3)
        ax.plot([0, tenergy.max()], [0, 0], color='k', linestyle='--')
        ax.set_xlim([0, np.max(tfields)])
        ax.set_ylim(ylim)
        ax.tick_params(labelsize=10)
        if irun == 0:
            ax.set_ylabel('Energization', fontsize=12)
        # else:
        #     ax.tick_params(axis='y', labelleft='off')
        pdim = "2D" if "2D" in pic_run else "3D"
        ax.text(0.02, 0.9, pdim, color='k', fontsize=12,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)
        ax.set_xlabel(r'$t\Omega_{ci}$', fontsize=12)

        ax = fig2.add_axes(box1)
        axs2.append(ax)
        # COLORS = palettable.colorbrewer.qualitative.Set1_9.mpl_colors
        ax.set_prop_cycle('color', COLORS)
        label2 = (r'$\boldsymbol{j}_{' + species + '\perp}' +
                  r'\cdot\boldsymbol{E}_\perp$')
        ax.plot(tfields, eperp_ene, linewidth=1, label=label2)
        ax.plot(tfields, curv_drift_dote, linewidth=1, label='Curvature')
        # ax.plot(tfields, bulk_curv_dote, linewidth=1, label='Bulk Curvature')
        ax.plot(tfields, grad_drift_dote, linewidth=1, label='Gradient')
        ax.plot(tfields, magnetization_dote, linewidth=1, label='Magnetization')
        # ax.plot(tfields, acc_drift_dote, linewidth=1, label='Polarization')
        jdote_sum = (curv_drift_dote + grad_drift_dote +
                     magnetization_dote + jagy_dote + acc_drift_dote)
        # ax.plot(tfields, jdote_sum, linewidth=1)
        ax.plot([0, tenergy.max()], [0, 0], color='k', linestyle='--')
        ax.set_xlim([0, np.max(tfields)])
        ax.set_ylim(ylim)
        ax.tick_params(labelsize=10)
        if irun == 0:
            ax.set_ylabel('Energization', fontsize=12)
        # else:
        #     ax.tick_params(axis='y', labelleft='off')
        pdim = "2D" if "2D" in pic_run else "3D"
        ax.text(0.02, 0.9, pdim, color='k', fontsize=12,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)
        ax.set_xlabel(r'$t\Omega_{ci}$', fontsize=12)

        ax = fig3.add_axes(box1)
        axs3.append(ax)
        # COLORS = palettable.colorbrewer.qualitative.Set1_9.mpl_colors
        ax.set_prop_cycle('color', COLORS)
        label2 = (r'$\boldsymbol{j}_{' + species + '\perp}' +
                  r'\cdot\boldsymbol{E}_\perp' + '$')
        ax.plot(tfields, eperp_ene, linewidth=1, label=label2)
        ax.plot(tfields, comp_ene, linewidth=1, label='Compression')
        ax.plot(tfields, shear_ene, linewidth=1, label='Shear')
        # label2 = (r'$\boldsymbol{j}_{' + species + '\perp}' +
        #           r'\cdot\boldsymbol{E}_\perp -' + 'n_' + species +
        #           'm_' + species + r'(d\boldsymbol{u}_' + species +
        #           r'/dt)\cdot\boldsymbol{v}_E$')
        # ax.plot(tfields, eperp_ene - acc_drift_dote, linewidth=1, label=label2)
        label4 = (r'$\boldsymbol{j}_{' + species + r'-\text{agy}}' +
                  r'\cdot\boldsymbol{E}_\perp$')
        ax.plot(tfields, jagy_dote, linewidth=1, label=label4)
        # jdote_sum = comp_ene + shear_ene + jagy_dote
        # ax.plot(tfields, jdote_sum, linewidth=1)
        ax.plot([0, tenergy.max()], [0, 0], color='k', linestyle='--')
        ax.set_xlim([0, np.max(tfields)])
        ax.set_ylim(ylim)
        ax.tick_params(labelsize=10)
        if irun == 0:
            ax.set_ylabel('Energization', fontsize=12)
        # else:
        #     ax.tick_params(axis='y', labelleft='off')
        pdim = "2D" if "2D" in pic_run else "3D"
        ax.text(0.02, 0.9, pdim, color='k', fontsize=12,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)
        ax.set_xlabel(r'$t\Omega_{ci}$', fontsize=12)

        box1[0] += box1[2] + 0.07

    axs1[0].legend(loc='upper center', prop={'size': 12}, ncol=4,
                   bbox_to_anchor=(1.1, 1.22),
                   shadow=False, fancybox=False, frameon=False)
    axs2[0].legend(loc='upper center', prop={'size': 12}, ncol=4,
                   bbox_to_anchor=(1.1, 1.22),
                   shadow=False, fancybox=False, frameon=False)
    axs3[0].legend(loc='upper center', prop={'size': 12}, ncol=4,
                   bbox_to_anchor=(1.1, 1.22),
                   shadow=False, fancybox=False, frameon=False)
    fdir = '../img/cori_3d/fluid_energization/'
    mkdir_p(fdir)
    fname = fdir + 'fluid_ene_' + species + '.pdf'
    fig1.savefig(fname)

    fdir = '../img/cori_3d/fluid_energization/'
    mkdir_p(fdir)
    fname = fdir + 'fluid_drift_' + species + '.pdf'
    fig2.savefig(fname)

    fdir = '../img/cori_3d/fluid_energization/'
    mkdir_p(fdir)
    fname = fdir + 'fluid_comp_shear_' + species + '.pdf'
    fig3.savefig(fname)

    if show_plot:
        plt.show()
    else:
        plt.close('all')


def fluid_energization_pub(plot_config, show_plot=True):
    """Plot fluid energization for publication

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    pic_run_dir = plot_config["pic_run_dir"]
    pic_runs = ["3D-Lx150-bg0.2-150ppc-2048KNL", "2D-Lx150-bg0.2-150ppc-16KNL"]
    if species == 'e':
        ylim = [-0.2, 1.0]
    else:
        ylim = [-0.6, 1.7]
    fig1 = plt.figure(figsize=[3.25, 2.5])
    box1 = [[0.2, 0.18, 0.76, 0.75],
            [0.6, 0.55, 0.33, 0.33]]
    axs1 = []
    fig2 = plt.figure(figsize=[3.25, 2.5])
    box2 = [[0.2, 0.18, 0.76, 0.75],
            [0.6, 0.55, 0.33, 0.33]]
    axs2 = []
    fig3 = plt.figure(figsize=[3.25, 2.5])
    box3 = [0.09, 0.18, 0.41, 0.68]
    axs3 = []
    for irun, pic_run in enumerate(pic_runs):
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        if "3D" in pic_run:
            enorm = pic_info.ny
        else:
            enorm = 1.0
        tfields = pic_info.tfields
        tenergy = pic_info.tenergy
        fname = "../data/fluid_energization/" + pic_run + "/"
        fname += "emf_ptensor_" + species + '.gda'
        fluid_ene = np.fromfile(fname, dtype=np.float32)
        nvar = int(fluid_ene[0])
        nframes = int(fluid_ene[1])
        fluid_ene[2:] /= enorm
        curv_drift_dote = fluid_ene[2:nframes+2]
        bulk_curv_dote = fluid_ene[nframes+2:2*nframes+2]
        grad_drift_dote = fluid_ene[2*nframes+2:3*nframes+2]
        magnetization_dote = fluid_ene[3*nframes+2:4*nframes+2]
        comp_ene = fluid_ene[4*nframes+2:5*nframes+2]
        shear_ene = fluid_ene[5*nframes+2:6*nframes+2]
        ptensor_ene = fluid_ene[6*nframes+2:7*nframes+2]
        pgyro_ene = fluid_ene[7*nframes+2:8*nframes+2]

        fname = "../data/fluid_energization/" + pic_run + "/"
        fname += "para_perp_acc_" + species + '.gda'
        fluid_ene = np.fromfile(fname, dtype=np.float32)
        nvar = int(fluid_ene[0])
        nframes = int(fluid_ene[1])
        fluid_ene[2:] /= enorm
        acc_drift_dote_t = fluid_ene[2:nframes+2]
        acc_drift_dote_s = fluid_ene[nframes+2:2*nframes+2]
        acc_drift_dote = acc_drift_dote_t + acc_drift_dote_s
        epara_ene = fluid_ene[2*nframes+2:3*nframes+2]
        eperp_ene = fluid_ene[3*nframes+2:4*nframes+2]
        acc_drift_dote[-1] = acc_drift_dote[-2]

        jperp_dote = curv_drift_dote + grad_drift_dote + magnetization_dote
        jagy_dote = ptensor_ene - jperp_dote
        if species == 'e':
            dkene = pic_info.dkene_e
        else:
            dkene = pic_info.dkene_i
        dkene /= enorm

        ax = fig1.add_axes(box1[irun])
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')
        axs1.append(ax)
        COLORS = palettable.tableau.Tableau_10.mpl_colors
        ax.set_prop_cycle('color', COLORS)
        label2 = (r'$\boldsymbol{j}_{' + species + '\perp}' +
                  r'\cdot\boldsymbol{E}_\perp$')
        ax.plot(tfields, eperp_ene, linewidth=1, label=label2)
        label1 = (r'$\boldsymbol{j}_{' + species + '\parallel}' +
                  r'\cdot\boldsymbol{E}_\parallel$')
        ax.plot(tfields, epara_ene, linewidth=1, label=label1)
        label6 = r'$dK_' + species + '/dt$'
        ax.plot(tenergy, dkene, linewidth=1, label=label6)
        label3 = r'$(\nabla\cdot\tensorsym{P}_' + species + r')\cdot\boldsymbol{v}_E$'
        ax.plot(tfields, ptensor_ene, linewidth=1, label=label3)
        ax.plot([0, tenergy.max()], [0, 0], color='k',
                linewidth=0.5, linestyle='--')
        ax.set_xlim([0, np.max(tfields)])
        ax.set_ylim(ylim)
        ax.tick_params(labelsize=8)
        pdim = "2D" if "2D" in pic_run else "3D"
        ypos = 0.93 if "3D" in pic_run else 0.85
        ax.text(0.02, ypos, pdim, color='k', fontsize=10,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)
        if irun == 0:
            ax.text(0.04, 0.82, label3, color=COLORS[3], fontsize=8,
                    bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                    horizontalalignment='left', verticalalignment='center',
                    transform=ax.transAxes)
            ax.text(0.04, 0.71, label2, color=COLORS[0], fontsize=8,
                    bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                    horizontalalignment='left', verticalalignment='center',
                    transform=ax.transAxes)
            ax.text(0.04, 0.6, label6, color=COLORS[2], fontsize=8,
                    bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                    horizontalalignment='left', verticalalignment='center',
                    transform=ax.transAxes)
            ax.text(0.8, 0.15, label1, color=COLORS[1], fontsize=8,
                    bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                    horizontalalignment='left', verticalalignment='center',
                    transform=ax.transAxes)
        if irun == 0:
            ax.set_xlabel(r'$t\Omega_{ci}$', fontsize=10)
            ax.set_ylabel('Energization', fontsize=10)
        else:
            ax.tick_params(axis='x', labelbottom='off')
            ax.tick_params(axis='y', labelleft='off')

        ax = fig2.add_axes(box1[irun])
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')
        axs2.append(ax)
        ax.set_prop_cycle('color', COLORS)
        label2 = (r'$\boldsymbol{j}_{' + species + '\perp}' +
                  r'\cdot\boldsymbol{E}_\perp$')
        ax.plot(tfields, eperp_ene, linewidth=1, label=label2)
        ax.plot(tfields, curv_drift_dote, linewidth=1, label='Curvature')
        # ax.plot(tfields, bulk_curv_dote, linewidth=1, label='Bulk Curvature')
        ax.plot(tfields, grad_drift_dote, linewidth=1, label='Gradient')
        ax.plot(tfields, magnetization_dote, linewidth=1, label='Magnetization')
        # ax.plot(tfields, acc_drift_dote, linewidth=1, label='Polarization')
        jdote_sum = (curv_drift_dote + grad_drift_dote +
                     magnetization_dote + jagy_dote + acc_drift_dote)
        # ax.plot(tfields, jdote_sum, linewidth=1)
        label1 = (r'$\boldsymbol{j}_{' + species + '\parallel}' +
                  r'\cdot\boldsymbol{E}_\parallel$')
        ax.plot(tfields, epara_ene, linewidth=1, label=label1)
        ax.plot([0, tenergy.max()], [0, 0], color='k',
                linewidth=0.5, linestyle='--')
        ax.set_xlim([0, np.max(tfields)])
        ax.set_ylim(ylim)
        ax.tick_params(labelsize=8)
        pdim = "2D" if "2D" in pic_run else "3D"
        ypos = 0.93 if "3D" in pic_run else 0.85
        ax.text(0.02, ypos, pdim, color='k', fontsize=10,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)
        if irun == 0:
            ax.text(0.04, 0.82, label2, color=COLORS[0], fontsize=8,
                    bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                    horizontalalignment='left', verticalalignment='center',
                    transform=ax.transAxes)
            ax.text(0.04, 0.74, 'Curvature', color=COLORS[1], fontsize=8,
                    bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                    horizontalalignment='left', verticalalignment='center',
                    transform=ax.transAxes)
            ax.text(0.52, 0.07, 'Gradient', color=COLORS[2], fontsize=8,
                    bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                    horizontalalignment='left', verticalalignment='center',
                    transform=ax.transAxes)
            ax.text(0.1, 0.07, 'Magnetization', color=COLORS[3], fontsize=8,
                    bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                    horizontalalignment='left', verticalalignment='center',
                    transform=ax.transAxes)
            ax.text(0.17, 0.25, label1, color=COLORS[4], fontsize=8,
                    bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                    horizontalalignment='left', verticalalignment='center',
                    transform=ax.transAxes)
        if irun == 0:
            ax.set_xlabel(r'$t\Omega_{ci}$', fontsize=10)
            ax.set_ylabel('Energization', fontsize=12)
        else:
            ax.tick_params(axis='x', labelbottom='off')
            ax.tick_params(axis='y', labelleft='off')

        ax = fig3.add_axes(box1[irun])
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')
        axs3.append(ax)
        # COLORS = palettable.colorbrewer.qualitative.Set1_9.mpl_colors
        ax.set_prop_cycle('color', COLORS)
        label2 = (r'$\boldsymbol{j}_{' + species + '\perp}' +
                  r'\cdot\boldsymbol{E}_\perp' + '$')
        ax.plot(tfields, eperp_ene, linewidth=1, label=label2)
        ax.plot(tfields, comp_ene, linewidth=1, label='Compression')
        ax.plot(tfields, shear_ene, linewidth=1, label='Shear')
        # label2 = (r'$\boldsymbol{j}_{' + species + '\perp}' +
        #           r'\cdot\boldsymbol{E}_\perp -' + 'n_' + species +
        #           'm_' + species + r'(d\boldsymbol{u}_' + species +
        #           r'/dt)\cdot\boldsymbol{v}_E$')
        # ax.plot(tfields, eperp_ene - acc_drift_dote, linewidth=1, label=label2)
        label4 = (r'$\boldsymbol{j}_{' + species + r'-\text{agy}}' +
                  r'\cdot\boldsymbol{E}_\perp$')
        ax.plot(tfields, jagy_dote, linewidth=1, label=label4)
        # jdote_sum = comp_ene + shear_ene + jagy_dote
        # ax.plot(tfields, jdote_sum, linewidth=1)
        ax.plot([0, tenergy.max()], [0, 0], color='k',
                linewidth=0.5, linestyle='--')
        ax.set_xlim([0, np.max(tfields)])
        ax.set_ylim(ylim)
        ax.tick_params(labelsize=8)
        pdim = "2D" if "2D" in pic_run else "3D"
        ypos = 0.93 if "3D" in pic_run else 0.85
        ax.text(0.02, ypos, pdim, color='k', fontsize=10,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)
        if irun == 0:
            ax.text(0.04, 0.82, label2, color=COLORS[0], fontsize=8,
                    bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                    horizontalalignment='left', verticalalignment='center',
                    transform=ax.transAxes)
            ax.text(0.04, 0.74, 'Compression', color=COLORS[1], fontsize=8,
                    bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                    horizontalalignment='left', verticalalignment='center',
                    transform=ax.transAxes)
            ax.text(0.25, 0.34, 'Shear', color=COLORS[2], fontsize=8,
                    bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                    horizontalalignment='left', verticalalignment='center',
                    transform=ax.transAxes)
            ax.text(0.7, 0.1, label4, color=COLORS[3], fontsize=8,
                    bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                    horizontalalignment='left', verticalalignment='center',
                    transform=ax.transAxes)
        if irun == 0:
            ax.set_xlabel(r'$t\Omega_{ci}$', fontsize=10)
            # ax.set_ylabel('Energization', fontsize=10)
        else:
            ax.tick_params(axis='x', labelbottom='off')
            ax.tick_params(axis='y', labelleft='off')

    fdir = '../img/cori_3d/fluid_energization/'
    mkdir_p(fdir)
    fname = fdir + 'fluid_ene_pub_' + species + '.pdf'
    fig1.savefig(fname)

    fdir = '../img/cori_3d/fluid_energization/'
    mkdir_p(fdir)
    fname = fdir + 'fluid_drift_pub_' + species + '.pdf'
    fig2.savefig(fname)

    fdir = '../img/cori_3d/fluid_energization/'
    mkdir_p(fdir)
    fname = fdir + 'fluid_comp_shear_pub_' + species + '.pdf'
    fig3.savefig(fname)

    if show_plot:
        plt.show()
    else:
        plt.close('all')


def div0(a, b):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0]

    From: http://stackoverflow.com/a/35696047/2561161

    """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[~np.isfinite(c)] = 0  # -inf inf NaN
    return c


def get_plot_setup(plot_type, species):
    """Get plotting setup for different type
    """
    if plot_type == 'total':
        if species == 'e':
            ylims = [[-0.001, 0.003], [-0.0005, 0.001],
                     [-0.0002, 0.0004], [-0.0002, 0.0004]]
        else:
            ylims = [[-0.001, 0.003], [-0.0005, 0.002],
                     [-0.0002, 0.0005], [-0.0002, 0.0004]]
        data_indices = [1, 2]
    if plot_type == 'perpendicular':
        if species == 'e':
            ylims = [[-0.001, 0.003], [-0.0005, 0.001],
                     [-0.0002, 0.0004], [-0.0002, 0.0004]]
        else:
            ylims = [[-0.001, 0.003], [-0.0005, 0.002],
                     [-0.0002, 0.0005], [-0.0002, 0.0004]]
        data_indices = [2]
    if plot_type == 'parallel':
        if species == 'e':
            ylims = [[-0.001, 0.003], [-0.0005, 0.001],
                     [-0.0002, 0.0004], [-0.0002, 0.0004]]
        else:
            ylims = [[-0.001, 0.003], [-0.0005, 0.002],
                     [-0.0002, 0.0005], [-0.0002, 0.0004]]
        data_indices = [1]
    elif plot_type == 'curvature':
        if species == 'e':
            ylims = [[-0.001, 0.004], [-0.0005, 0.0015],
                     [-0.0002, 0.0004], [-0.0002, 0.0005]]
        else:
            ylims = [[-0.001, 0.004], [-0.0005, 0.0015],
                     [-0.0002, 0.0003], [-0.0002, 0.0004]]
        data_indices = [5]
    elif plot_type == 'gradient':
        if species == 'e':
            ylims = [[-0.0002, 0.00025], [-0.0002, 0.00025],
                     [-0.0001, 0.0001], [-0.0001, 0.0001]]
        else:
            ylims = [[-0.0002, 0.00025], [-0.0002, 0.00025],
                     [-0.0001, 0.0001], [-0.0001, 0.0001]]
        data_indices = [6]
    elif plot_type == 'inertial':
        if species == 'e':
            ylims = [[-0.00075, 0.00025], [-0.0002, 0.0001],
                     [-0.0001, 0.0001], [-0.0002, 0.0001]]
        else:
            ylims = [[-0.0012, 0.0004], [-0.0008, 0.0001],
                     [-0.0001, 0.0001], [-0.0002, 0.0001]]
        data_indices = [11, 12]
    elif plot_type == 'polarization':
        if species == 'e':
            ylims = [[-0.00025, 0.00025], [-0.00025, 0.00025],
                     [-0.0001, 0.0001], [-0.0002, 0.0001]]
        else:
            ylims = [[-0.001, 0.002], [-0.0005, 0.0015],
                     [-0.0002, 0.0005], [-0.0002, 0.00025]]
        data_indices = [15, 16]
    elif plot_type == 'parallel_drift':
        if species == 'e':
            ylims = [[-0.00025, 0.0004], [-0.00005, 0.0001],
                     [-0.00005, 0.0001], [-0.0002, 0.0001]]
        else:
            ylims = [[-0.00002, 0.0002], [-0.00005, 0.0001],
                     [-0.00005, 0.0001], [-0.00005, 0.00005]]
        data_indices = [7]
    elif plot_type == 'mu':
        if species == 'e':
            ylims = [[-0.0003, 0.0002], [-0.0002, 0.0005],
                     [-0.0001, 0.0003], [-0.0001, 0.0003]]
        else:
            ylims = [[-0.0003, 0.0002], [-0.0002, 0.0005],
                     [-0.0001, 0.0003], [-0.0001, 0.0003]]
        data_indices = [8]
    elif plot_type == 'compression':
        if species == 'e':
            ylims = [[-0.001, 0.003], [-0.0005, 0.001],
                     [-0.0002, 0.0004], [-0.0002, 0.0004]]
        else:
            ylims = [[-0.001, 0.003], [-0.0005, 0.002],
                     [-0.0002, 0.0005], [-0.0002, 0.0004]]
        data_indices = [3]
    elif plot_type == 'shear':
        if species == 'e':
            ylims = [[-0.0005, 0.0015], [-0.00025, 0.0005],
                     [-0.0001, 0.0002], [-0.0001, 0.0002]]
        else:
            ylims = [[-0.0005, 0.0015], [-0.00025, 0.0005],
                     [-0.0001, 0.0002], [-0.0001, 0.0002]]
        data_indices = [4]

    return (ylims, data_indices)


def particle_energization(plot_config):
    """Particle-based energization

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    tframe = plot_config["tframe"]
    pic_run_dir = plot_config["pic_run_dir"]
    pic_runs = ["2D-Lx150-bg0.2-150ppc-16KNL", "3D-Lx150-bg0.2-150ppc-2048KNL"]

    ylims, data_indices = get_plot_setup(plot_config["plot_type"], species)

    tstarts = [6, 10, 20, 30]
    tends = [10, 20, 30, 40]
    nplots = len(tstarts)

    fnorm = 1E-3
    for iplot in range(nplots):
        tstart = tstarts[iplot]
        tend = tends[iplot]
        ylim = np.asarray(ylims[iplot]) / fnorm
        fig1 = plt.figure(figsize=[4.8, 2.0])
        box1 = [0.14, 0.2, 0.36, 0.75]
        axs1 = []

        nframes = tend - tstart

        for irun, pic_run in enumerate(pic_runs):
            fpath = "../data/particle_interp/" + pic_run + "/"
            picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
            pic_info = read_data_from_json(picinfo_fname)
            dtp = math.ceil(pic_info.dt_particles / 0.1) * 0.1
            ax = fig1.add_axes(box1)
            for tframe in range(tstart, tend + 1):
                tstep = tframe * pic_info.particle_interval
                tframe_fluid = tstep // pic_info.fields_interval
                fname = fpath + "particle_energization_" + species + "_" + str(tstep) + ".gda"
                fdata = np.fromfile(fname, dtype=np.float32)
                nbins = int(fdata[0])
                nvar = int(fdata[1])
                ebins = fdata[2:nbins+2]
                fbins = fdata[nbins+2:].reshape((nvar, nbins))

                if species == 'i':
                    ebins *= pic_info.mime  # ebins are actually gamma
                if species == 'e':
                    fbins[1:, :] = div0(fbins[1:, :], fbins[0, :])
                else:
                    fbins[1:, :] = div0(fbins[1:, :], fbins[0, :] * pic_info.mime)

                # normalized with thermal energy
                if species == 'e':
                    vth = pic_info.vthe
                else:
                    vth = pic_info.vthi
                gama = 1.0 / math.sqrt(1.0 - 3 * vth**2)
                eth = gama - 1.0
                if species == 'i':
                    eth *= pic_info.mime

                ebins /= eth

                color = plt.cm.Spectral_r((tframe - tstart)/float(nframes), 1)
                fdata = np.zeros(nbins)
                for idata in data_indices:
                    fdata += fbins[idata, :]
                ax.semilogx(ebins, fdata/fnorm, linewidth=1, color=color)
            if species == 'e':
                if "2D" in pic_run:
                    ax.set_xlim([1E0, 200])
                else:
                    ax.set_xlim([1E0, 200])
            else:
                if "2D" in pic_run:
                    ax.set_xlim([1E0, 500])
                else:
                    ax.set_xlim([1E0, 1000])
            ax.set_ylim(ylim)
            box1[0] += box1[2] + 0.02
            if irun == 0:
                # ax.set_ylabel('Acceleration Rate', fontsize=12)
                ax.set_ylabel(r'$\left<\nu^\text{I}_\text{COM}\right>/10^{-3}$', fontsize=10)
            else:
                ax.tick_params(axis='y', labelleft='off')

            ax.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$', fontsize=10)
            ax.tick_params(labelsize=8)

            pdim = "2D" if "2D" in pic_run else "3D"
            ax.text(0.02, 0.9, pdim, color='k', fontsize=10,
                    bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                    horizontalalignment='left', verticalalignment='center',
                    transform=ax.transAxes)
            ax.plot(ax.get_xlim(), [0, 0], linestyle='--', color='k')
            ax.tick_params(bottom=True, top=False, left=True, right=False)
            ax.tick_params(axis='x', which='minor', direction='in')
            ax.tick_params(axis='x', which='major', direction='in')
            ax.tick_params(axis='y', which='minor', direction='in')
            ax.tick_params(axis='y', which='major', direction='in')

            if irun == 1:
                box1[0] -= box1[2] + 0.02
                rect_cbar = np.copy(box1)
                rect_cbar[0] += box1[2] + 0.01
                rect_cbar[2] = 0.015
                cax = fig1.add_axes(rect_cbar)
                cax.tick_params(axis='y', which='major', direction='in')
                sm = plt.cm.ScalarMappable(cmap=plt.cm.Spectral_r,
                                           norm=plt.Normalize(vmin=tstart * dtp,
                                                              vmax=tend * dtp))
                # fake up the array of the scalar mappable. Urgh...
                sm._A = []
                cbar = fig1.colorbar(sm, cax=cax)
                cbar.set_label(r'$t\Omega_{ci}$', fontsize=12)
                cbar.set_ticks((np.linspace(tstart, tend, tend - tstart + 1) * dtp))
                cbar.ax.tick_params(labelsize=10)

        fdir = '../img/cori_3d/particle_energization/'
        mkdir_p(fdir)
        fname = (fdir + 'particle_' + plot_config["plot_type"] + '_' +
                 species + '_' + str(iplot) + '.pdf')
        fig1.savefig(fname)

    plt.show()


def fit_particle_energization(plot_config):
    """Fit particle-based energization

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    tframe = plot_config["tframe"]
    pic_run_dir = plot_config["pic_run_dir"]
    pic_runs = ["2D-Lx150-bg0.2-150ppc-16KNL", "3D-Lx150-bg0.2-150ppc-2048KNL"]

    ylims, data_indices = get_plot_setup(plot_config["plot_type"], species)

    tstarts = [6, 10, 20, 30]
    tends = [10, 20, 30, 40]
    nplots = len(tstarts)

    tstart, tend = 1, 40
    nframes = tend - tstart + 1
    slope = np.zeros((2, nframes))
    alpha0 = np.zeros((2, nframes))
    tframe_chosen = [10, 18]

    fig1 = plt.figure(figsize=[3.5, 5])
    box1 = [0.16, 0.58, 0.8, 0.4]
    ax = fig1.add_axes(box1)

    for irun, pic_run in enumerate(pic_runs):
        fpath = "../data/particle_interp/" + pic_run + "/"
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        dtp = math.ceil(pic_info.dt_particles / 0.1) * 0.1
        tfields = np.arange(tstart, tend+1) * dtp
        for tframe in range(tstart, tend + 1):
            tstep = tframe * pic_info.particle_interval
            tframe_fluid = tstep // pic_info.fields_interval
            fname = fpath + "particle_energization_" + species + "_" + str(tstep) + ".gda"
            fdata = np.fromfile(fname, dtype=np.float32)
            nbins = int(fdata[0])
            nvar = int(fdata[1])
            ebins = fdata[2:nbins+2]
            fbins = fdata[nbins+2:].reshape((nvar, nbins))

            if species == 'i':
                ebins *= pic_info.mime  # ebins are actually gamma
            if species == 'e':
                fbins[1:, :] = div0(fbins[1:, :], fbins[0, :])
            else:
                fbins[1:, :] = div0(fbins[1:, :], fbins[0, :] * pic_info.mime)

            # normalized with thermal energy
            if species == 'e':
                vth = pic_info.vthe
            else:
                vth = pic_info.vthi
            gama = 1.0 / math.sqrt(1.0 - 3 * vth**2)
            eth = gama - 1.0
            if species == 'i':
                eth *= pic_info.mime

            ebins /= eth
            pindex1, ene1 = find_nearest(ebins, 25)
            if '2D' in pic_run:
                pindex2, ene2 = find_nearest(ebins, 150)
            else:
                pindex2, ene2 = find_nearest(ebins, 200)

            color = plt.cm.Spectral_r((tframe - tstart)/float(nframes), 1)
            fdata = np.zeros(nbins)
            for idata in data_indices:
                fdata += fbins[idata, :]

            # fit particle energization
            fnorm = 1E-3
            erange = ebins[pindex1:pindex2+1]
            frange = fdata[pindex1:pindex2+1] / fnorm

            # elog = np.log10(erange)
            elog = erange
            fit_fun = np.polyfit(elog, frange, 1)
            slope[irun, tframe - tstart] = fit_fun[0]
            alpha0[irun, tframe - tstart] = fit_fun[1]
            p = np.poly1d(fit_fun)
            f_fit = p(elog)

            color = COLORS[4] if '2D' in pic_run else COLORS[1]
            if tframe in tframe_chosen:
                lstyle = '-' if tframe == tframe_chosen[0] else '--'
                ax.plot(ebins, fdata/fnorm, linewidth=1,
                            color=color, linestyle=lstyle)
                # if '3D' in pic_run and tframe == tframe_chosen[0]:
                    # ax.plot(erange, f_fit, linewidth=1,
                    #         color='k', linestyle=':')
                    # alpha_norm = "{%0.2f}" % (fit_fun[1] / 0.1)
                    # text1 = r"$\alpha_0 = " + alpha_norm + r"\times 10^{-4}$"
                    # ax.text(0.5, 0.93, text1, color='k', fontsize=10,
                    #         bbox=dict(facecolor='none', alpha=1.0,
                    #                   edgecolor='none', pad=10.0),
                    #         horizontalalignment='center',
                    #         verticalalignment='center',
                    #         transform=ax.transAxes)
                    # snorm = "{%0.1f}" % (fit_fun[0] / 1E-4)
                    # text1 = r"$s = " + snorm + r"\times 10^{-4}$"
                    # # text1 = r"$s = " + ("{%0.4f}" % (fit_fun[0])) + "$"
                    # ax.text(0.5, 0.73, text1, color='k', fontsize=10,
                    #         bbox=dict(facecolor='none', alpha=1.0,
                    #                   edgecolor='none', pad=10.0),
                    #         horizontalalignment='center',
                    #         verticalalignment='center',
                    #         transform=ax.transAxes)

            # fig1 = plt.figure(figsize=[7, 5])
            # box1 = [0.14, 0.14, 0.8, 0.8]
            # ax = fig1.add_axes(box1)
            # ax.semilogx(erange, frange, linewidth=2, color=color)
            # ax.semilogx(erange, f_fit, linewidth=2, color=color)
            # # ax.semilogx(ebins, fdata, linewidth=1, color=color)
            # if species == 'e':
            #     ax.set_xlim([20, 200])
            #     # ax.set_xlim([1E0, 200])
            # else:
            #     ax.set_xlim([1E0, 500])
            # # ax.set_ylim(ylims[0])
            # plt.show()
    ax.text(0.1, 0.93, "2D", color=COLORS[4], fontsize=10,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes)
    ax.text(0.17, 0.93, "3D", color=COLORS[1], fontsize=10,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes)

    ax.text(0.37, 0.48, r'$t\Omega_{ci}=100$', color='k', fontsize=8,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes)
    ax.text(0.37, 0.28, r'$t\Omega_{ci}=180$', color='k', fontsize=8,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes)

    ax.tick_params(bottom=True, top=False, left=True, right=False)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$', fontsize=10)
    ax.set_ylabel(r'$\alpha\omega_{pe}^{-1}/10^{-3}$', fontsize=10)
    ax.tick_params(labelsize=8)
    if species == 'e':
        ax.set_xlim([0, 200])
    else:
        ax.set_xlim([1E0, 500])
    ax.plot(ax.get_xlim(), [0, 0], linestyle='--',
            color='k', linewidth=0.5)
    ax.set_ylim([-0.2, 1.0])
    # ax.set_ylim([-0.4, 0.2])

    box1[1] -= box1[3] + 0.1
    ax = fig1.add_axes(box1)
    slope /= 1E-2
    ts = 5
    ax.plot(tfields[ts:], slope[0, ts:], marker='v', markersize=3,
            linestyle='None', color=COLORS[4])
    ax.plot(tfields[ts:], slope[1, ts:], marker='o', markersize=3,
            linestyle='None', color=COLORS[1])
    tmin, tmax = tfields[ts - 1], tfields.max()
    ax.set_xlim([tmin, tmax])
    ax.set_ylim([-0.3, 0.7])
    ax.plot([tmin, tmax], [0, 0], linewidth=0.5, linestyle='--', color='k')
    ax.set_xlabel(r'$t\Omega_{ci}$', fontsize=10)
    ax.set_ylabel(r'$s/10^{-2}$', fontsize=10)
    ax.tick_params(bottom=True, top=False, left=True, right=False)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.tick_params(labelsize=8)

    ax.text(0.1, 0.93, "2D", color=COLORS[4], fontsize=10,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes)
    ax.text(0.17, 0.93, "3D", color=COLORS[1], fontsize=10,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes)

    box2 = [0.56, 0.3, 0.35, 0.15]
    ax1 = fig1.add_axes(box2)
    ax1.plot(tfields[ts:], alpha0[0, ts:], marker='v', markersize=3,
             linestyle='-', color=COLORS[4], linewidth=1)
    ax1.plot(tfields[ts:], alpha0[1, ts:], marker='o', markersize=3,
             linestyle='-', color=COLORS[1], linewidth=1)
    tmin, tmax = tfields[ts - 1], tfields.max()
    ax1.set_xlim([tmin, tmax])
    ax1.plot([tmin, tmax], [0, 0], linewidth=0.5, linestyle='--', color='k')
    ax1.tick_params(bottom=True, top=False, left=True, right=False)
    ax1.tick_params(axis='x', which='minor', direction='in')
    ax1.tick_params(axis='x', which='major', direction='in')
    ax1.tick_params(axis='y', which='minor', direction='in')
    ax1.tick_params(axis='y', which='major', direction='in')
    # ax1.tick_params(axis='x', labelbottom='off')
    ax1.set_ylabel(r'$\alpha_0\omega_{pe}^{-1}/10^{-3}$', fontsize=8)
    ax1.tick_params(labelsize=8)

    fdir = '../img/cori_3d/particle_energization/'
    mkdir_p(fdir)
    fname = (fdir + 'particle_' + plot_config["plot_type"] + '_' +
             species + '.pdf')
    fig1.savefig(fname)

    plt.show()


def acceleration_rate(plot_config):
    """Get accelerate data

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    tframe = plot_config["tframe"]
    tstart = plot_config["tstart"]
    tend = plot_config["tend"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]

    fpath = "../data/particle_interp/" + pic_run + "/"
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    fdir = "../data/acceleration_rate/" + pic_run + "/"
    mkdir_p(fdir)
    for tframe in range(tstart, tend + 1):
        tstep = tframe * pic_info.particle_interval
        tframe_fluid = tstep // pic_info.fields_interval
        fname = fpath + "particle_energization_" + species + "_" + str(tstep) + ".gda"
        fdata = np.fromfile(fname, dtype=np.float32)
        nbins = int(fdata[0])
        nvar = int(fdata[1])
        ebins = fdata[2:nbins+2]
        fbins = fdata[nbins+2:].reshape((nvar, nbins))

        if species == 'i':
            ebins *= pic_info.mime  # ebins are actually gamma
        if species == 'e':
            fbins[1:, :] = div0(fbins[1:, :], fbins[0, :])
        else:
            fbins[1:, :] = div0(fbins[1:, :], fbins[0, :] * pic_info.mime)

        # normalized with thermal energy
        if species == 'e':
            vth = pic_info.vthe
        else:
            vth = pic_info.vthi
        gama = 1.0 / math.sqrt(1.0 - 3 * vth**2)
        eth = gama - 1.0
        if species == 'i':
            eth *= pic_info.mime

        # ebins /= eth

        fdata = np.zeros((2, nbins))
        fdata[0, :] = ebins
        fdata[1, :] = fbins[1, :] + fbins[2, :]
        fname = fdir + "acc_rate_" + str(tframe) + ".dat"
        fdata.tofile(fname)


def analytical_fan(plot_config):
    """Calculate and plot Fan's analytical expression

    Reference: Guo et al. PRL 113, 155005 (2014)
    """
    ene = np.logspace(1.3, 3.3, 200)
    ene_sqrt = np.sqrt(ene)
    alpha_tau = 5.0
    alpha_tau_h = alpha_tau * 0.5
    exp_atau = math.exp(-alpha_tau)
    exp_atauh = math.exp(-alpha_tau_h)
    f = ((erf(ene_sqrt) - erf(ene_sqrt * exp_atauh)) / ene +
         2 * (exp_atauh * np.exp(-ene * exp_atau) - np.exp(-ene)) /
         math.sqrt(math.pi) / ene_sqrt)
    plt.loglog(ene, f)
    plt.show()


def energetic_rho(plot_config):
    """Plot densities for energetic particles
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
    nxr, nyr, nzr = pic_info.nx//4, pic_info.ny//4, pic_info.nz//4
    nslicex, nslicey, nslicez = 64, 32, 28
    box_size = 12
    box_size_h = box_size // 2
    shiftz = (nzr - (nslicez - 2) * box_size) // 2
    midx = np.linspace(box_size_h - 1, nxr - box_size_h - 1, nslicex, dtype=int)
    midy = np.linspace(box_size_h - 1, nyr - box_size_h - 1, nslicey, dtype=int)
    midz = np.linspace(box_size_h + shiftz - 1, nzr - box_size_h - shiftz - 1,
                       nslicez - 2, dtype=int)
    xmin, xmax = 0, pic_info.lx_di
    ymin, ymax = -pic_info.ly_di * 0.5, pic_info.ly_di * 0.5
    zmin, zmax = -pic_info.lz_di * 0.5, pic_info.lz_di * 0.5
    nmin, nmax = 0.02, 0.2

    nbands = 7
    ntot = np.zeros((nzr, nyr, nxr))
    nhigh = np.zeros((nzr, nyr, nxr))
    for iband in range(nbands):
        print("Energy band: %d" % iband)
        fname = (pic_run_dir + "data-smooth2/n" + species + "_" +
                 str(iband) + "_" + str(tindex) + ".gda")
        nrho = np.fromfile(fname, dtype=np.float32)
        nrho = nrho.reshape((nzr, nyr, nxr))
        if iband >= 2:
            nhigh += nrho
        ntot += nrho

    fraction_h = nhigh / ntot
    fdir = '../img/cori_3d/energetic_rho_3d/tframe_' + str(tframe) + '/'
    mkdir_p(fdir)

    for iz in midz:
        print("z-slice %d" % iz)
        fig = plt.figure(figsize=[9, 4])
        rect = [0.10, 0.16, 0.75, 0.8]
        ax = fig.add_axes(rect)
        p1 = ax.imshow(fraction_h[iz, :, :],
                       extent=[xmin, xmax, ymin, ymax],
                       vmin=nmin, vmax=nmax,
                       cmap=plt.cm.viridis, aspect='auto',
                       origin='lower', interpolation='bicubic')
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')
        ax.set_xlabel(r'$x/d_i$', fontsize=20)
        ax.set_ylabel(r'$y/d_i$', fontsize=20)
        ax.tick_params(labelsize=16)

        rect_cbar = np.copy(rect)
        rect_cbar[0] += rect[2] + 0.02
        rect_cbar[2] = 0.02
        cbar_ax = fig.add_axes(rect_cbar)
        cbar = fig.colorbar(p1, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=16)
        cbar_ax.set_ylabel(r'$n(\varepsilon > 20\varepsilon_\text{th})/n_\text{tot}$',
                           fontsize=24)
        iz_str = str(iz).zfill(4)
        fname = (fdir + 'nhigh_' + species + '_xy_' +
                 str(tframe) + "_" + iz_str + ".jpg")
        fig.savefig(fname, dpi=200)
        plt.close()

    for iy in midy:
        print("y-slice %d" % iy)
        fig = plt.figure(figsize=[9, 4])
        rect = [0.10, 0.16, 0.75, 0.8]
        ax = fig.add_axes(rect)
        p1 = ax.imshow(fraction_h[:, iy, :],
                       extent=[xmin, xmax, zmin, zmax],
                       vmin=nmin, vmax=nmax,
                       cmap=plt.cm.viridis, aspect='auto',
                       origin='lower', interpolation='bicubic')
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')
        ax.set_xlabel(r'$x/d_i$', fontsize=20)
        ax.set_ylabel(r'$z/d_i$', fontsize=20)
        ax.tick_params(labelsize=16)

        rect_cbar = np.copy(rect)
        rect_cbar[0] += rect[2] + 0.02
        rect_cbar[2] = 0.02
        cbar_ax = fig.add_axes(rect_cbar)
        cbar = fig.colorbar(p1, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=16)
        cbar_ax.set_ylabel(r'$n(\varepsilon > 20\varepsilon_\text{th})/n_\text{tot}$',
                           fontsize=24)
        iy_str = str(iy).zfill(4)
        fname = (fdir + 'nhigh_' + species + '_xz_' +
                 str(tframe) + "_" + iy_str + ".jpg")
        fig.savefig(fname, dpi=200)
        plt.close()

    for ix in midx:
        print("x-slice %d" % ix)
        fig = plt.figure(figsize=[7, 5])
        rect = [0.12, 0.16, 0.70, 0.8]
        ax = fig.add_axes(rect)
        p1 = ax.imshow(fraction_h[:, :, ix],
                       extent=[ymin, ymax, zmin, zmax],
                       vmin=nmin, vmax=nmax,
                       cmap=plt.cm.viridis, aspect='auto',
                       origin='lower', interpolation='bicubic')
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')
        ax.set_xlabel(r'$y/d_i$', fontsize=20)
        ax.set_ylabel(r'$z/d_i$', fontsize=20)
        ax.tick_params(labelsize=16)

        rect_cbar = np.copy(rect)
        rect_cbar[0] += rect[2] + 0.02
        rect_cbar[2] = 0.02
        cbar_ax = fig.add_axes(rect_cbar)
        cbar = fig.colorbar(p1, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=16)
        cbar_ax.set_ylabel(r'$n(\varepsilon > 20\varepsilon_\text{th})/n_\text{tot}$',
                           fontsize=24)
        ix_str = str(ix).zfill(4)
        fname = (fdir + 'nhigh_' + species + '_yz_' +
                 str(tframe) + "_" + ix_str + ".jpg")
        fig.savefig(fname, dpi=200)
        plt.close()


def energetic_rho_2d(plot_config):
    """Plot densities for energetic particles in the 2D simulation
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
    xmin, xmax = 0, pic_info.lx_di
    zmin, zmax = -pic_info.lz_di * 0.5, pic_info.lz_di * 0.5
    nmin, nmax = 0.02, 0.2
    nx, nz = pic_info.nx, pic_info.nz

    nbands = 7
    ntot = np.zeros((nz, nx))
    nhigh = np.zeros((nz, nx))
    for iband in range(nbands):
        print("Energy band: %d" % iband)
        fname = (pic_run_dir + "data-smooth2/n" + species + "_" +
                 str(iband) + "_" + str(tindex) + ".gda")
        nrho = np.fromfile(fname, dtype=np.float32)
        nrho = nrho.reshape((nz, nx))
        if iband >= 2:
            nhigh += nrho
        ntot += nrho

    fraction_h = nhigh / ntot
    fdir = '../img/cori_3d/energetic_rho_2d/'
    mkdir_p(fdir)

    fig = plt.figure(figsize=[9, 4])
    rect = [0.10, 0.16, 0.75, 0.8]
    ax = fig.add_axes(rect)
    p1 = ax.imshow(fraction_h[:, :],
                   extent=[xmin, xmax, zmin, zmax],
                   vmin=nmin, vmax=nmax,
                   cmap=plt.cm.viridis, aspect='auto',
                   origin='lower', interpolation='bicubic')
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlabel(r'$x/d_i$', fontsize=20)
    ax.set_ylabel(r'$z/d_i$', fontsize=20)
    ax.tick_params(labelsize=16)

    rect_cbar = np.copy(rect)
    rect_cbar[0] += rect[2] + 0.02
    rect_cbar[2] = 0.02
    cbar_ax = fig.add_axes(rect_cbar)
    cbar = fig.colorbar(p1, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=16)
    cbar_ax.set_ylabel(r'$n(\varepsilon > 20\varepsilon_\text{th})/n_\text{tot}$',
                       fontsize=24)
    fname = (fdir + 'nhigh_' + species + '_' + str(tframe) + ".jpg")
    fig.savefig(fname, dpi=200)
    plt.close()


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
    parser.add_argument('--whole_spectrum', action="store_true", default=False,
                        help='whether to plot spectrum in the whole box')
    parser.add_argument('--spectrum_pub', action="store_true", default=False,
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
    parser.add_argument('--econv', action="store_true", default=False,
                        help='whether to plot energy conversion')
    parser.add_argument('--show_plot', action="store_true", default=False,
                        help='whether to show plot')
    parser.add_argument('--jslice', action="store_true", default=False,
                        help='whether to plot slices of current density')
    parser.add_argument('--absj_2d', action="store_true", default=False,
                        help='whether to plot the current density of the 2D simulation')
    parser.add_argument('--absj_2d_pub', action="store_true", default=False,
                        help=('whether to plot the current density of' +
                              'the 2d simulation for publication'))
    parser.add_argument('--jslice_box', action="store_true", default=False,
                        help='whether to plot slices of current density with boxes')
    parser.add_argument('--absj_spect', action="store_true", default=False,
                        help='whether to plot current density with local spectrum')
    parser.add_argument('--j2d_box', action="store_true", default=False,
                        help='whether to plot current density with boxes in 2D')
    parser.add_argument('--fluid_ene', action="store_true", default=False,
                        help='whether to plot fluid energization')
    parser.add_argument('--fluid_ene_pub', action="store_true", default=False,
                        help='whether to plot fluid energization for publication')
    parser.add_argument('--particle_ene', action="store_true", default=False,
                        help='whether to plot particle energization')
    parser.add_argument('--fit_particle_ene', action="store_true", default=False,
                        help='whether to fit particle energization')
    parser.add_argument('--plot_type', action="store", default='total', type=str,
                        help='Particle plot type')
    parser.add_argument('--acc_rate', action="store_true", default=False,
                        help='whether to get accelerate data')
    parser.add_argument('--analytical_fan', action="store_true", default=False,
                        help="whether to calculate Fan's analytical expression")
    parser.add_argument('--energetic_rho', action="store_true", default=False,
                        help="whether to plot densities for energetic particles")
    parser.add_argument('--energetic_rho_2d', action="store_true", default=False,
                        help=("whether to plot densities for energetic " +
                              "particles for the 2D simulation"))
    parser.add_argument('--rho_profile', action="store_true", default=False,
                        help="whether to plot densities profile")
    parser.add_argument('--absb_profile', action="store_true", default=False,
                        help="whether to plot magnetic field magnitude")
    parser.add_argument('--comp_factor', action="store_true", default=False,
                        help="whether to compression factor")
    parser.add_argument('--calc_absj_dist', action="store_true", default=False,
                        help="whether to calculate current density distribution")
    parser.add_argument('--plot_absj_dist', action="store_true", default=False,
                        help="whether to plot current density distribution")
    parser.add_argument('--calc_abse_dist', action="store_true", default=False,
                        help="whether to calculate electric field distribution")
    parser.add_argument('--plot_abse_dist', action="store_true", default=False,
                        help="whether to plot electric field distribution")
    return parser.parse_args()


def analysis_single_frames(plot_config, args):
    """Analysis for multiple time frames
    """
    tframe = args.tframe
    if args.whole_spectrum:
        if args.all_frames:
            plot_spectrum(plot_config)
        else:
            plot_spectrum_single(plot_config, args.show_plot)
    elif args.spectrum_pub:
        plot_spectrum_pub(plot_config)
    elif args.local_spectrum:
        plot_local_spectrum(plot_config)
    elif args.local_spectrum2d:
        plot_local_spectrum2d(plot_config)
    elif args.mom_spectrum:
        if args.all_frames:
            plot_momentum_spectrum_all(plot_config)
        else:
            plot_momentum_spectrum(plot_config)
    elif args.compare_spectrum:
        compare_spectrum(plot_config)
    elif args.econv:
        energy_conversion(plot_config)
    elif args.jslice:
        plot_jslice(plot_config)
    elif args.absj_2d:
        plot_absj_2d(plot_config)
    elif args.absj_2d_pub:
        absj_2d_pub(plot_config)
    elif args.jslice_box:
        plot_jslice_box(plot_config)
    elif args.absj_spect:
        plot_absj_spect(plot_config)
    elif args.j2d_box:
        plotj_box_2d(plot_config)
    elif args.fluid_ene:
        fluid_energization(plot_config)
    elif args.fluid_ene_pub:
        fluid_energization_pub(plot_config)
    elif args.particle_ene:
        particle_energization(plot_config)
    elif args.fit_particle_ene:
        fit_particle_energization(plot_config)
    elif args.acc_rate:
        acceleration_rate(plot_config)
    elif args.analytical_fan:
        analytical_fan(plot_config)
    elif args.energetic_rho:
        energetic_rho(plot_config)
    elif args.energetic_rho_2d:
        energetic_rho_2d(plot_config)
    elif args.rho_profile:
        rho_profile(plot_config)
    elif args.absb_profile:
        absb_profile(plot_config)
    elif args.comp_factor:
        compression_factor(plot_config)
    elif args.calc_absj_dist:
        calc_absj_dist(plot_config)
    elif args.plot_absj_dist:
        plot_absj_dist(plot_config)
    elif args.calc_abse_dist:
        calc_abse_dist(plot_config)
    elif args.plot_abse_dist:
        plot_abse_dist(plot_config)


def process_input(plot_config, args, tframe):
    """process one time frame"""
    plot_config["tframe"] = tframe
    if args.energetic_rho:
        energetic_rho(plot_config)
    elif args.energetic_rho_2d:
        energetic_rho_2d(plot_config)
    elif args.absj_2d:
        plot_absj_2d(plot_config, show_plot=False)
    elif args.mom_spectrum:
        plot_momentum_spectrum(plot_config, show_plot=False)
    elif args.calc_absj_dist:
        calc_absj_dist(plot_config)
    elif args.calc_abse_dist:
        calc_abse_dist(plot_config)


def analysis_multi_frames(plot_config, args):
    """Analysis for multiple time frames
    """
    tframes = range(plot_config["tstart"], plot_config["tend"] + 1)
    if args.time_loop:
        for tframe in tframes:
            print("Time frame: %d" % tframe)
            plot_config["tframe"] = tframe
            if args.absj_2d:
                plot_absj_2d(plot_config, show_plot=False)
            elif args.mom_spectrum:
                plot_momentum_spectrum(plot_config, show_plot=False)
            elif args.rho_profile:
                rho_profile(plot_config, show_plot=False)
            elif args.absb_profile:
                absb_profile(plot_config, show_plot=False)
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
    plot_config["plot_type"] = args.plot_type
    if args.multi_frames:
        analysis_multi_frames(plot_config, args)
    else:
        analysis_single_frames(plot_config, args)


if __name__ == "__main__":
    main()
