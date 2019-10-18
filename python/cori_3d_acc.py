#!/usr/bin/env python3
"""
Analysis on particle acceleration
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
from scipy import signal
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy.interpolate import interp1d, interp2d, RegularGridInterpolator
from scipy.ndimage.filters import median_filter, gaussian_filter

import fitting_funcs
import pic_information
from contour_plots import read_2d_fields
from dolointerpolation import MultilinearInterpolator
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


def acceleration_rate(plot_config, show_plot=True):
    """Particle-based energization acceleration rate

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    tframe = plot_config["tframe"]
    tstart = plot_config["tstart"]
    tend = plot_config["tend"]
    bg = plot_config['bg']
    bg_str = str(int(bg * 10)).zfill(2)
    pic_run_dir = plot_config["pic_run_dir"]
    bg = plot_config["bg"]
    pic_runs = ["3D-Lx150-bg" + str(bg) + "-150ppc-2048KNL"]
    pic_runs.append("2D-Lx150-bg" + str(bg) + "-150ppc-16KNL")

    fnorm = 1E-3
    fig1 = plt.figure(figsize=[16, 10])
    rect0 = [0.05, 0.75, 0.15, 0.2]
    hgap, vgap = 0.04, 0.02
    row, col = 4, 5
    axs = []
    for i in range(row):
        rect = np.copy(rect0)
        rect[1] -= (rect[3] + vgap) * i
        for j in range(col):
            ax = fig1.add_axes(rect)
            ax.set_prop_cycle('color', COLORS)
            rect[0] += rect[2] + hgap
            if i < row - 1:
                ax.tick_params(axis='x', labelbottom='off')
            else:
                ax.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$', fontsize=12)
            ax.tick_params(bottom=True, top=False, left=True, right=True)
            ax.tick_params(axis='x', which='minor', direction='in', top='on')
            ax.tick_params(axis='x', which='major', direction='in', top='on')
            ax.tick_params(axis='y', which='minor', direction='in')
            ax.tick_params(axis='y', which='major', direction='in')
            ax.set_xlim([1E0, 2E2])
            axs.append(ax)
    labels = [r'$\boldsymbol{E}_\parallel$', r'$\boldsymbol{E}_\perp$',
              'Compression', 'Shear', 'Curvature', 'Gradient',
              'Parallel drift', r'$\mu$ conservation', 'Polar-time',
              'Polar-spatial', 'Inertial-time', 'Inertial-spatial',
              'Polar-fluid-time', 'Polar-fluid-spatial',
              'Polar-time-v', 'Polar-spatial-v', 'Ptensor',
              r'$\boldsymbol{E}_\parallel + \boldsymbol{E}_\perp$',
              'Compression + Shear', 'Curvature + Gradient']

    ntot = np.zeros(len(pic_runs))
    ptl_weights = np.zeros(len(pic_runs))
    for irun, pic_run in enumerate(pic_runs):
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        nptl = pic_info.nx * pic_info.ny * pic_info.nz * pic_info.nppc
        ntot[irun] = nptl / pic_info.stride_particle_dump
        lx_di = pic_info.lx_di
        ly_di = pic_info.lz_di
        lz_di = pic_info.ly_di
        smime = math.sqrt(pic_info.mime)
        # assuming n0 = 1
        weight = (lx_di * ly_di * lz_di * smime**3) / nptl
        ptl_weights[irun] = weight
    fnorm = ntot.max() / ntot

    for irun, pic_run in enumerate(pic_runs):
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        dtp = math.ceil(pic_info.dt_particles / 0.1) * 0.1
        tstep = tframe * pic_info.particle_interval
        tframe_fluid = tstep // pic_info.fields_interval
        fpath = "../data/particle_interp/" + pic_run + "/"
        fname = fpath + "particle_energization_" + species + "_" + str(tstep) + ".gda"
        fdata = np.fromfile(fname, dtype=np.float32)
        nbins = int(fdata[0])
        nbinx = int(fdata[1])
        nvar = int(fdata[2])
        ebins = fdata[3:nbins+3]
        fbins = np.sum(fdata[nbins+3:].reshape((nvar, nbinx, nbins)), axis=1)

        if species == 'i':
            ebins *= pic_info.mime  # ebins are actually gamma
        fbins[1:, :] = div0(fbins[1:, :], fbins[0, :])
        dee = fbins[18:, :] - fbins[1:18, :]**2

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
        emin, emax = 1E0, 2E2
        es, _ = find_nearest(ebins, emin)
        ee, _ = find_nearest(ebins, emax)

        ymax = np.max(fbins[1:6, es:ee+1])
        for iplot in range(17):
            ax = axs[iplot]
            ax.semilogx(ebins[es:ee+1], fbins[iplot+1, es:ee+1],
                        marker='o', markersize=4, linestyle='-', linewidth=1)
            ax.grid(True)

            ax.set_ylim([-5E-4, ymax])

            ax.text(0.05, 0.05, labels[iplot], color='k', fontsize=12,
                    bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                    horizontalalignment='left', verticalalignment='bottom',
                    transform=ax.transAxes)
            if iplot == 2:
                ax.text(0.4, 0.95, '3D', color=COLORS[0], fontsize=16,
                        bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                        horizontalalignment='center', verticalalignment='top',
                        transform=ax.transAxes)
                ax.text(0.6, 0.95, '2D', color=COLORS[1], fontsize=16,
                        bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                        horizontalalignment='center', verticalalignment='top',
                        transform=ax.transAxes)
                text1 = r'$t\Omega_{ci}=' + str(tframe*10) + '$'
                ax.set_title(text1, fontsize=20)
            ax.plot(ax.get_xlim(), [0, 0], color='k', linestyle='--')
            ax.tick_params(labelsize=10)
        ax = axs[17]
        ax.semilogx(ebins[es:ee+1], fbins[1, es:ee+1] + fbins[2, es:ee+1],
                    marker='o', markersize=4, linestyle='-', linewidth=1)
        ax.grid(True)
        ax.set_ylim([-5E-4, ymax])
        ax.text(0.05, 0.05, labels[17], color='k', fontsize=12,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='bottom',
                transform=ax.transAxes)

        ax = axs[18]
        ax.semilogx(ebins[es:ee+1], fbins[3, es:ee+1] + fbins[4, es:ee+1],
                    marker='o', markersize=4, linestyle='-', linewidth=1)
        ax.grid(True)
        ax.set_ylim([-5E-4, ymax])
        ax.text(0.05, 0.05, labels[18], color='k', fontsize=12,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='bottom',
                transform=ax.transAxes)

        ax = axs[19]
        ax.semilogx(ebins[es:ee+1], fbins[5, es:ee+1] + fbins[6, es:ee+1],
                    marker='o', markersize=4, linestyle='-', linewidth=1)
        ax.grid(True)
        ax.set_ylim([-5E-4, ymax])
        ax.text(0.05, 0.05, labels[19], color='k', fontsize=12,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='bottom',
                transform=ax.transAxes)

    fdir = '../img/cori_3d/acceleration_rates/bg' + bg_str + '/'
    mkdir_p(fdir)
    fname = fdir + 'acc_rates_' + str(tframe) + '.pdf'
    fig1.savefig(fname)

    if show_plot:
        plt.show()
    else:
        plt.close('all')


def acceleration_rate_std(plot_config, show_plot=True):
    """The standard deviation of acceleration rates

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    tframe = plot_config["tframe"]
    tstart = plot_config["tstart"]
    tend = plot_config["tend"]
    bg = plot_config['bg']
    bg_str = str(int(bg * 10)).zfill(2)
    pic_run_dir = plot_config["pic_run_dir"]
    bg = plot_config["bg"]
    pic_runs = ["3D-Lx150-bg" + str(bg) + "-150ppc-2048KNL"]
    pic_runs.append("2D-Lx150-bg" + str(bg) + "-150ppc-16KNL")

    fnorm = 1E-3
    fig1 = plt.figure(figsize=[16, 10])
    rect0 = [0.05, 0.75, 0.15, 0.2]
    hgap, vgap = 0.04, 0.02
    row, col = 4, 5
    axs = []
    for i in range(row):
        rect = np.copy(rect0)
        rect[1] -= (rect[3] + vgap) * i
        for j in range(col):
            ax = fig1.add_axes(rect)
            ax.set_prop_cycle('color', COLORS)
            rect[0] += rect[2] + hgap
            if i < row - 1:
                ax.tick_params(axis='x', labelbottom=False)
            else:
                ax.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$', fontsize=12)
            ax.tick_params(bottom=True, top=False, left=True, right=True)
            ax.tick_params(axis='x', which='minor', direction='in', top=True)
            ax.tick_params(axis='x', which='major', direction='in', top=True)
            ax.tick_params(axis='y', which='minor', direction='in')
            ax.tick_params(axis='y', which='major', direction='in')
            # ax.set_xlim([1E0, 5E2])
            axs.append(ax)
    labels = [r'$\boldsymbol{E}_\parallel$', r'$\boldsymbol{E}_\perp$',
              'Compression', 'Shear', 'Curvature', 'Gradient',
              'Parallel drift', r'$\mu$ conservation', 'Polar-time',
              'Polar-spatial', 'Inertial-time', 'Inertial-spatial',
              'Polar-fluid-time', 'Polar-fluid-spatial',
              'Polar-time-v', 'Polar-spatial-v', 'Ptensor',
              r'$\boldsymbol{E}_\parallel + \boldsymbol{E}_\perp$',
              'Compression + Shear', 'Curvature + Gradient']

    ntot = np.zeros(len(pic_runs))
    ptl_weights = np.zeros(len(pic_runs))
    for irun, pic_run in enumerate(pic_runs):
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        nptl = pic_info.nx * pic_info.ny * pic_info.nz * pic_info.nppc
        ntot[irun] = nptl / pic_info.stride_particle_dump
        lx_di = pic_info.lx_di
        ly_di = pic_info.lz_di
        lz_di = pic_info.ly_di
        smime = math.sqrt(pic_info.mime)
        # assuming n0 = 1
        weight = (lx_di * ly_di * lz_di * smime**3) / nptl
        ptl_weights[irun] = weight
    fnorm = ntot.max() / ntot

    for irun, pic_run in enumerate(pic_runs):
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        dtp = math.ceil(pic_info.dt_particles / 0.1) * 0.1
        tstep = tframe * pic_info.particle_interval
        tframe_fluid = tstep // pic_info.fields_interval
        fpath = "../data/particle_interp/" + pic_run + "/"
        fname = fpath + "particle_energization_" + species + "_" + str(tstep) + ".gda"
        fdata = np.fromfile(fname, dtype=np.float32)
        nbins = int(fdata[0])
        nbinx = int(fdata[1])
        nvar = int(fdata[2])
        ebins = fdata[3:nbins+3]
        gamma = ebins + 1
        fbins = np.sum(fdata[nbins+3:].reshape((nvar, nbinx, nbins)), axis=1)
        pbins = np.sqrt((ebins + 1)**2 - 1)
        pbins_mid = (pbins[:-1] + pbins[1:]) * 0.5
        dpbins = np.diff(pbins)

        if species == 'i':
            ebins *= pic_info.mime  # ebins are actually gamma
        fbins[1:, :] = div0(fbins[1:, :], fbins[0, :])
        dee = fbins[18:, :] - fbins[1:18, :]**2
        dgamma = fbins[1:18, :] * ebins
        dgamma2 = fbins[18:, :] * ebins**2
        dp = gamma * dgamma / np.sqrt(gamma**2 - 1)
        dp2 = dgamma2 + 2 * dgamma
        dpp = (dp2 - dp**2) / pbins**2

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
        emin, emax = 1E0, 5E2
        es, _ = find_nearest(ebins, emin)
        ee, _ = find_nearest(ebins, emax)

        ymax = np.max(dee[:5, es:ee+1])
        for iplot in range(17):
            ax = axs[iplot]
            # ax.loglog(ebins[es:ee+1], dee[iplot, es:ee+1],
            #           marker='o', markersize=4, linestyle='-', linewidth=1)
            ax.loglog(pbins, dpp[iplot],
                      marker='o', markersize=4, linestyle='-', linewidth=1)
            ax.grid(True)

            # ax.set_ylim([-5E-4, ymax])

            ax.text(0.05, 0.05, labels[iplot], color='k', fontsize=12,
                    bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                    horizontalalignment='left', verticalalignment='bottom',
                    transform=ax.transAxes)
            if iplot == 2:
                ax.text(0.4, 0.95, '3D', color=COLORS[0], fontsize=16,
                        bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                        horizontalalignment='center', verticalalignment='top',
                        transform=ax.transAxes)
                ax.text(0.6, 0.95, '2D', color=COLORS[1], fontsize=16,
                        bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                        horizontalalignment='center', verticalalignment='top',
                        transform=ax.transAxes)
                text1 = r'$t\Omega_{ci}=' + str(tframe*10) + '$'
                ax.set_title(text1, fontsize=20)
            # ax.plot(ax.get_xlim(), [0, 0], color='k', linestyle='--')
            ax.tick_params(labelsize=10)
        # ax = axs[17]
        # ax.semilogx(ebins[es:ee+1], fbins[1, es:ee+1] + fbins[2, es:ee+1],
        #             marker='o', markersize=4, linestyle='-', linewidth=1)
        # ax.grid(True)
        # ax.set_ylim([-5E-4, ymax])
        # ax.text(0.05, 0.05, labels[17], color='k', fontsize=12,
        #         bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        #         horizontalalignment='left', verticalalignment='bottom',
        #         transform=ax.transAxes)

        # ax = axs[18]
        # ax.semilogx(ebins[es:ee+1], fbins[3, es:ee+1] + fbins[4, es:ee+1],
        #             marker='o', markersize=4, linestyle='-', linewidth=1)
        # ax.grid(True)
        # ax.set_ylim([-5E-4, ymax])
        # ax.text(0.05, 0.05, labels[18], color='k', fontsize=12,
        #         bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        #         horizontalalignment='left', verticalalignment='bottom',
        #         transform=ax.transAxes)

        # ax = axs[19]
        # ax.semilogx(ebins[es:ee+1], fbins[5, es:ee+1] + fbins[6, es:ee+1],
        #             marker='o', markersize=4, linestyle='-', linewidth=1)
        # ax.grid(True)
        # ax.set_ylim([-5E-4, ymax])
        # ax.text(0.05, 0.05, labels[19], color='k', fontsize=12,
        #         bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        #         horizontalalignment='left', verticalalignment='bottom',
        #         transform=ax.transAxes)

    # fdir = '../img/cori_3d/acceleration_rates/bg' + bg_str + '/'
    # mkdir_p(fdir)
    # fname = fdir + 'acc_rates_' + str(tframe) + '.pdf'
    # fig1.savefig(fname)

    if show_plot:
        plt.show()
    else:
        plt.close('all')


def acceleration_rate_distribution(plot_config, show_plot=True):
    """Particle-based acceleration rate distribution

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    tframe = plot_config["tframe"]
    tstart = plot_config["tstart"]
    tend = plot_config["tend"]
    pic_run_dir = plot_config["pic_run_dir"]
    bg = plot_config["bg"]
    pic_runs = ["3D-Lx150-bg" + str(bg) + "-150ppc-2048KNL"]
    pic_runs.append("2D-Lx150-bg" + str(bg) + "-150ppc-16KNL")

    fnorm = 1E-3
    fig1 = plt.figure(figsize=[10, 4.0])
    rect = [0.10, 0.2, 0.36, 0.75]
    hgap, vgap = 0.08, 0.02

    ntot = np.zeros(len(pic_runs))
    ptl_weights = np.zeros(len(pic_runs))
    for irun, pic_run in enumerate(pic_runs):
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        nptl = pic_info.nx * pic_info.ny * pic_info.nz * pic_info.nppc
        ntot[irun] = nptl / pic_info.stride_particle_dump
        lx_di = pic_info.lx_di
        ly_di = pic_info.lz_di
        lz_di = pic_info.ly_di
        smime = math.sqrt(pic_info.mime)
        # assuming n0 = 1
        weight = (lx_di * ly_di * lz_di * smime**3) / nptl
        ptl_weights[irun] = weight
    fnorm = ntot.max() / ntot

    for irun, pic_run in enumerate(pic_runs):
        ax = fig1.add_axes(rect)
        rect[0] += rect[2] + hgap
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        dtp = math.ceil(pic_info.dt_particles / 0.1) * 0.1
        tstep = tframe * pic_info.particle_interval
        tframe_fluid = tstep // pic_info.fields_interval
        fpath = "../data/particle_interp/" + pic_run + "/"
        fname = fpath + "particle_energization_" + species + "_" + str(tstep) + ".gda"
        fdata = np.fromfile(fname, dtype=np.float32)
        nbins = int(fdata[0])
        nbinx = int(fdata[1])
        nvar = int(fdata[2])
        ebins = fdata[3:nbins+3]
        fbins = np.sum(fdata[nbins+3:].reshape((nvar, nbinx, nbins)), axis=1)

        if species == 'i':
            ebins *= pic_info.mime  # ebins are actually gamma
        fbins[1:, :] = div0(fbins[1:, :], fbins[0, :])
        dee = fbins[18:, :] - fbins[1:18, :]**2

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

        ax.semilogx(ebins, fbins[5, :])
        # ax.loglog(ebins, dee.T)

        # the distributions of particle acceleration rates
        fname = fpath + "acc_rate_dist_" + species + "_" + str(tstep) + ".gda"
        fdata = np.fromfile(fname, dtype=np.float32)
        nalpha = int(fdata[0]) # number of bins of the rates
        nbins = int(fdata[1])  # number of bins along x
        nbinx = int(fdata[2])  # number of energy bins
        nvar = int(fdata[3])   # number of variables
        ebins = fdata[4:nbins+4]  # energy bins
        alpha_bins = fdata[nbins+4:nbins+nalpha+4]  # acceleration rates bins
        ebins /= eth
        # variables: 0-Epara, 1-Eperp, 2-compression, 3-shear, 4-curvature,
        #   5-gradient, 6-para_drift, 7-mu, 8-polar_time, 9-polar_spatial,
        #   10-inertial_time, 11-inertial_spatial, 12-polar_fluid_time,
        #   13-polar_fluid_spatial, 14-polar_time_v, 15-polar_spatial_v,
        #   16-ptensor
        fdist_x = fdata[nbins+nalpha+4:].reshape((nvar, nbinx, nbins, (nalpha+1)*4))
        fdist = np.sum(fdist_x, axis=1)
        alpha = div0(np.sum(fdist[:, :, (nalpha+1)*2:], axis=2),
                     np.sum(fdist[:, :, :(nalpha+1)*2], axis=2))
        debins = np.diff(ebins)
        fdist = fdist * fnorm[irun]
        # ax.semilogx(ebins, alpha[1, :])
        # ax.set_xlim([1E-2, 1E0])
        dalpha_log = math.log10(alpha_bins[1]) - math.log10(alpha_bins[0])
        alpha_min = 10**(math.log10(alpha_bins[0]) - dalpha_log)
        alpha_max = 10**(math.log10(alpha_bins[-1]) + dalpha_log)
        alpha_bins_sym = np.zeros((nalpha+1)*2+1)
        alpha_bins_sym[0] = -alpha_max
        alpha_bins_sym[1:nalpha+1] = -alpha_bins[::-1]
        alpha_bins_sym[nalpha+1] = 0
        alpha_bins_sym[nalpha+2:-1] = alpha_bins
        alpha_bins_sym[-1] = alpha_max
        dalpha = np.diff(alpha_bins_sym)
        alpha_bins_mid = 0.5 * (alpha_bins_sym[1:] + alpha_bins_sym[:-1])
        alpha_dist = fdist[:, :, :(nalpha+1)*2]
        # alpha_dist /= ptl_weights[irun]
        alpha_dist_fold = alpha_dist[:, :, nalpha+1:] - alpha_dist[:, :, nalpha::-1]
        alpha_dist_sum = np.sum(alpha_dist_fold * alpha_bins_mid[nalpha+1:], axis=2)
        acc_rate = div0(alpha_dist_sum, np.sum(alpha_dist, axis=2))
        # ax.semilogx(ebins, acc_rate[4, :])

        vmin, vmax = 1E-4, 1E0
        pdata = (alpha_dist_fold[4, :, :] * alpha_bins_mid[nalpha+1:]).T
        # pdata = alpha_dist[4, :, :] * np.abs(alpha_bins_mid)
        pdata[pdata==0] = 1E-8
        f = interp2d(ebins, alpha_bins_mid[nalpha+1:], pdata, kind='cubic')
        alpha_bins_new = np.logspace(math.log10(alpha_bins_mid[nalpha+1]),
                                     math.log10(alpha_bins_mid[-1]),
                                     4*nalpha+1)
        ebins_new = np.logspace(math.log10(ebins[0]),
                                math.log10(ebins[-1]),
                                4*(nbins-1)+1)
        pdata_new = f(ebins_new, alpha_bins_new)
        # Xn, Yn = np.meshgrid(ebins, alpha_bins_mid[nalpha+1:])
        Xn, Yn = np.meshgrid(ebins_new, alpha_bins_new)

        # ax.plot(pdata[:, 32] * 1E4)
        # ax.plot(alpha_dist[4, 32, nalpha+1:] + alpha_dist[4, 32, nalpha::-1])

        ax.set_xscale('log')
        ax.set_yscale('log')
        img = ax.pcolormesh(Xn, Yn, pdata_new, cmap=plt.cm.coolwarm,
                            # norm = LogNorm(vmin=vmin, vmax=vmax)
                           norm=SymLogNorm(linthresh=0.0001, linscale=0.01,
                                           vmin=-1.0, vmax=1.0),)

        ax.plot(alpha_dist[4, :, nalpha+1:].T + alpha_dist[4, :, nalpha::-1].T,
                alpha_bins_mid[nalpha+1:])
        ax.set_xlim([1E-1, 5E2])
        ax.set_ylim([1E-7, 1E-1])

    rect[0] -= rect[2] + hgap
    rect_cbar = np.copy(rect)
    rect_cbar[0] += rect[2] + 0.02
    rect_cbar[2] = 0.015
    # cbar_ax = fig1.add_axes(rect_cbar)
    # cbar = fig1.colorbar(img, cax=cbar_ax, extend='both')
    if show_plot:
        plt.show()
    else:
        plt.close('all')


def plot_anisotropy(plot_config, show_plot=True):
    """Plot particle-based pressure anisotropy

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    tframe = plot_config["tframe"]
    tstart = plot_config["tstart"]
    tend = plot_config["tend"]
    bg = plot_config['bg']
    bg_str = str(int(bg * 10)).zfill(2)
    pic_run_dir = plot_config["pic_run_dir"]
    bg = plot_config["bg"]
    pic_runs = ["3D-Lx150-bg" + str(bg) + "-150ppc-2048KNL"]
    pic_runs.append("2D-Lx150-bg" + str(bg) + "-150ppc-16KNL")

    fnorm = 1E-3
    fig = plt.figure(figsize=[3.5, 2.5])
    rect = [0.16, 0.16, 0.8, 0.8]
    ax = fig.add_axes(rect)
    hgap, vgap = 0.04, 0.02
    row, col = 4, 5
    axs = []

    for irun, pic_run in enumerate(pic_runs):
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        dtp = math.ceil(pic_info.dt_particles / 0.1) * 0.1
        tstep = tframe * pic_info.particle_interval
        tframe_fluid = tstep // pic_info.fields_interval
        fpath = "../data/particle_interp/" + pic_run + "/"
        fname = fpath + "anisotropy_" + species + "_" + str(tstep) + ".gda"
        fdata = np.fromfile(fname, dtype=np.float32)
        nvar = int(fdata[0])
        nbins = int(fdata[1])
        nbinx = int(fdata[2])
        ebins = fdata[3:nbins+3]
        color = COLORS[4] if '2D' in pic_run else COLORS[1]
        fbins = np.sum(fdata[nbins+3:].reshape((nbinx, nbins, nvar)), axis=0).T

        # fbins[1:, :] = div0(fbins[1:, :], fbins[0, :])

        # normalized with thermal energy
        if species == 'e':
            vth = pic_info.vthe
        else:
            vth = pic_info.vthi
        gama = 1.0 / math.sqrt(1.0 - 3 * vth**2)
        eth = gama - 1.0
        ebins /= eth
        ax.semilogx(ebins, div0(fbins[1, :], fbins[2, :]), linewidth=1,
                    color=color)
        # ax.loglog(ebins, fbins[1, :])
        # ax.loglog(ebins, fbins[2, :])

    ax.set_xlim([1E0, 5E2])
    ax.plot(ax.get_xlim(), [1, 1], color='k', linewidth=1, linestyle='--')
    ax.tick_params(labelsize=8)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top=True)
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in', left=True)
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$', fontsize=10)

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_anisotropy_pub(plot_config, show_plot=True):
    """Plot particle-based pressure anisotropy for publication

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    tframe = plot_config["tframe"]
    tstart = plot_config["tstart"]
    tend = plot_config["tend"]
    bg = plot_config['bg']
    bg_str = str(int(bg * 10)).zfill(2)
    bg = plot_config["bg"]
    pic_runs = ["3D-Lx150-bg" + str(bg) + "-150ppc-2048KNL"]
    pic_runs.append("2D-Lx150-bg" + str(bg) + "-150ppc-16KNL")
    root_dir = "/net/scratch3/xiaocanli/reconnection/Cori_runs/"
    tframes = [10, 15, 20]
    lstyles = ['-', '--', ':']
    fig = plt.figure(figsize=[3.5, 2.5])
    rect = [0.16, 0.16, 0.8, 0.8]
    hgap, vgap = 0.04, 0.02
    ax = fig.add_axes(rect)
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    ax.set_prop_cycle('color', COLORS)

    for irun, pic_run in enumerate(pic_runs):
        # color = COLORS[1] if irun else COLORS[0]
        lstyle = lstyles[1] if irun else lstyles[0]
        pic_run_dir = root_dir + pic_run + '/'

        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        dtp = math.ceil(pic_info.dt_particles / 0.1) * 0.1

        for iframe, tframe in enumerate(tframes):
            tstep = tframe * pic_info.particle_interval
            fpath = "../data/particle_interp/" + pic_run + "/"
            fname = fpath + "anisotropy_" + species + "_" + str(tstep) + ".gda"
            fdata = np.fromfile(fname, dtype=np.float32)
            nvar = int(fdata[0])
            nbins = int(fdata[1])
            nbinx = int(fdata[2])
            ebins = fdata[3:nbins+3]
            fbins = np.sum(fdata[nbins+3:].reshape((nbinx, nbins, nvar)), axis=0).T

            # normalized with thermal energy
            if species == 'e':
                vth = pic_info.vthe
            else:
                vth = pic_info.vthi
            gama = 1.0 / math.sqrt(1.0 - 3 * vth**2)
            eth = gama - 1.0
            ebins /= eth

            ax.plot(ebins, div0(fbins[1, :], fbins[2, :]), linewidth=1,
                    linestyle=lstyle, color=COLORS[iframe])

    ax.set_xlim([0, 250])
    ax.set_ylim([0.8, 3.6])
    ax.plot(ax.get_xlim(), [1, 1], color='k', linewidth=0.5, linestyle='-')
    ax.tick_params(labelsize=8)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top=True)
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in', left=True)
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$', fontsize=10)
    ax.set_ylabel('Anisotropy', fontsize=10)
    # ax.text(0.74, 0.92, "2D", color=COLORS[1], fontsize=10,
    #         bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
    #         horizontalalignment='center', verticalalignment='center',
    #         transform=ax.transAxes)
    # ax.text(0.90, 0.92, "3D", color=COLORS[0], fontsize=10,
    #         bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
    #         horizontalalignment='center', verticalalignment='center',
    #         transform=ax.transAxes)
    # ax.text(0.82, 0.92, "vs.", color='k', fontsize=10,
    #         bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
    #         horizontalalignment='center', verticalalignment='center',
    #         transform=ax.transAxes)

    ypos0 = 0.65
    ax.text(0.20, ypos0, '$t\Omega_{ci}=$', color='k', fontsize=10,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)
    dtf = math.ceil(pic_info.dt_fields / 0.1) * 0.1
    for iframe, tframe in enumerate(tframes):
        ypos = ypos0 + (1 - iframe) * 0.08
        text1 = r"$" + str(int(tframe*dtf)) + "$"
        ax.text(0.35, ypos, text1, color=COLORS[iframe], fontsize=10,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)

    ax.plot([-10, -1], [0, 0], color='k', linewidth=1,
            linestyle=lstyles[0], label='3D')
    ax.plot([-10, -1], [1, 1], color='k', linewidth=1,
            linestyle=lstyles[1], label='2D')
    ax.legend(loc=1, prop={'size': 10}, ncol=1,
              shadow=False, fancybox=True, frameon=True)

    fdir = '../img/cori_3d/'
    mkdir_p(fdir)
    fname = fdir + 'anisotropy_' + 'bg' + bg_str + '_' + species + '.pdf'
    fig.savefig(fname)

    if show_plot:
        plt.show()
    else:
        plt.close()


def calc_vexb_kappa_2d(plot_config):
    """Get the vexb dot magnetic curvature for the 2D simulations
    """
    bg = plot_config["bg"]
    tframe = plot_config["tframe"]
    pic_run = "2D-Lx150-bg" + str(bg) + "-150ppc-16KNL"
    root_dir = "/net/scratch3/xiaocanli/reconnection/Cori_runs/"
    pic_run_dir = root_dir + pic_run + "/"
    species = plot_config["species"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    tindex = tframe * pic_info.fields_interval
    pic_topox = pic_info.topology_x
    pic_topoy = pic_info.topology_y
    pic_topoz = pic_info.topology_z
    nx, ny, nz = pic_info.nx, pic_info.ny, pic_info.nz
    xmin, xmax = 0, pic_info.lx_di
    ymin, ymax = -pic_info.ly_di * 0.5, pic_info.ly_di * 0.5
    zmin, zmax = -pic_info.lz_di * 0.5, pic_info.lz_di * 0.5
    smime = math.sqrt(pic_info.mime)
    dx_de = pic_info.dx_di * smime
    dy_de = pic_info.dy_di * smime
    dz_de = pic_info.dz_di * smime
    kwargs = {"current_time": tframe,
              "xl": 0, "xr": pic_info.lx_di,
              "zb": -pic_info.lz_di, "zt": pic_info.lz_di}
    fname = pic_run_dir + "data/bx.gda"
    x, z, bx = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/by.gda"
    x, z, by = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/bz.gda"
    x, z, bz = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/ex.gda"
    x, z, ex = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/ey.gda"
    x, z, ey = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/ez.gda"
    x, z, ez = read_2d_fields(pic_info, fname, **kwargs)
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
    vexb_x = (ey * bz - ez * by) * ib
    vexb_y = (ez * bx - ex * bz) * ib
    vexb_z = (ex * by - ey * bx) * ib

    vexb_kappa = vexb_x * kappax + vexb_y * kappay + vexb_z * kappaz

    fdir = pic_run_dir + "data/"
    fname = fdir + 'vexb_kappa.gda'
    size_one_frame = pic_info.nx * pic_info.nz * 4
    with open(fname, 'a+') as f:
        offset = size_one_frame * tframe
        f.seek(offset, os.SEEK_SET)
        vexb_kappa.tofile(f)


def vkappa_dist_2d(plot_config):
    """Get the distribution of vdot_kappa for the 2D simulations
    """
    bg = plot_config["bg"]
    tframe = plot_config["tframe"]
    pic_run = "2D-Lx150-bg" + str(bg) + "-150ppc-16KNL"
    root_dir = "/net/scratch3/xiaocanli/reconnection/Cori_runs/"
    pic_run_dir = root_dir + pic_run + "/"
    species = plot_config["species"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    tindex = tframe * pic_info.fields_interval
    pic_topox = pic_info.topology_x
    pic_topoy = pic_info.topology_y
    pic_topoz = pic_info.topology_z
    nx, ny, nz = pic_info.nx, pic_info.ny, pic_info.nz
    xmin, xmax = 0, pic_info.lx_di
    ymin, ymax = -pic_info.ly_di * 0.5, pic_info.ly_di * 0.5
    zmin, zmax = -pic_info.lz_di * 0.5, pic_info.lz_di * 0.5
    smime = math.sqrt(pic_info.mime)
    dx_de = pic_info.dx_di * smime
    dy_de = pic_info.dy_di * smime
    dz_de = pic_info.dz_di * smime
    kwargs = {"current_time": tframe,
              "xl": 0, "xr": pic_info.lx_di,
              "zb": -pic_info.lz_di, "zt": pic_info.lz_di}
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

    nbins = 100
    vkappa_min, vkappa_max = 1E-8, 1E2
    vkappa_min_log = math.log10(vkappa_min)
    vkappa_max_log = math.log10(vkappa_max)
    dvkappa_log = (vkappa_max_log - vkappa_min_log) / nbins
    vkappa_bins_edge = np.zeros(nbins*2 + 5)
    vkappa_bins_edge[nbins+3:] = np.logspace(vkappa_min_log,
                                             vkappa_max_log + dvkappa_log,
                                             nbins+2)
    vkappa_bins_edge[:nbins+2] = -vkappa_bins_edge[-1:nbins+2:-1]
    vkappa_bins_edge[nbins+2] = 0
    fvkappa, _ = np.histogram(vdot_kappa, bins=vkappa_bins_edge)

    fdist = np.concatenate(([nbins*2 + 4], vkappa_bins_edge))
    fdist = np.concatenate((fdist, fvkappa))
    fdist = fdist.astype(np.float32)

    fdir = '../data/vkappa_dist/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + 'vkappa_dist_' + str(tindex) + '.gda'
    fdist.tofile(fname)


def reorganize_vkappa_dist_3d(plot_config):
    """Re-organize the distribution of vdot_kappa for the 3D simulations

    There are a couple of bugs when calculating the distribution, but they
    can be fixed by post-processing.
    """
    bg = plot_config["bg"]
    tframe = plot_config["tframe"]
    pic_run = "3D-Lx150-bg" + str(bg) + "-150ppc-2048KNL"
    root_dir = "/net/scratch3/xiaocanli/reconnection/Cori_runs/"
    pic_run_dir = root_dir + pic_run + "/"
    species = plot_config["species"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    tindex = tframe * pic_info.fields_interval
    fdir = '../data/vkappa_dist/test/' + pic_run + '/'
    fname = fdir + 'vkappa_dist_' + str(tindex) + '.gda'
    fdata = np.fromfile(fname, dtype=np.float32)
    nbins = int(fdata[0])
    vkappa_bins_edge = fdata[1:nbins+2]
    vkappa_bins_edge[:nbins//2] = -vkappa_bins_edge[:nbins//2]
    fvkappa = fdata[-nbins:]

    fdist = np.concatenate(([nbins], vkappa_bins_edge))
    fdist = np.concatenate((fdist, fvkappa))
    fdist = fdist.astype(np.float32)

    fdir = '../data/vkappa_dist/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + 'vkappa_dist_' + str(tindex) + '.gda'
    fdist.tofile(fname)


def plot_vkappa_dist(plot_config):
    """Plot the distribution of vdot_kappa
    """
    bg = plot_config["bg"]
    tframe = plot_config["tframe"]
    pic_runs = ["2D-Lx150-bg" + str(bg) + "-150ppc-16KNL",
                "3D-Lx150-bg" + str(bg) + "-150ppc-2048KNL"]
    root_dir = "/net/scratch3/xiaocanli/reconnection/Cori_runs/"
    fnorm = np.zeros(len(pic_runs))
    for irun, pic_run in enumerate(pic_runs):
        pic_run_dir = root_dir + pic_run + "/"
        species = plot_config["species"]
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        fnorm[irun] = pic_info.ny
    fnorm = fnorm.max() / fnorm

    fig = plt.figure(figsize=[7, 5])
    rect = [0.15, 0.15, 0.8, 0.8]
    ax = fig.add_axes(rect)
    for irun, pic_run in enumerate(pic_runs):
        pic_run_dir = root_dir + pic_run + "/"
        species = plot_config["species"]
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        tindex = tframe * pic_info.fields_interval
        fdir = '../data/vkappa_dist/' + pic_run + '/'
        fname = fdir + 'vkappa_dist_' + str(tindex) + '.gda'
        fdata = np.fromfile(fname, dtype=np.float32)
        nbins = int(fdata[0])
        vkappa_bins_edge = fdata[1:nbins+2]
        dvkappa = np.diff(vkappa_bins_edge)
        vkappa_bins_mid = 0.5 * (vkappa_bins_edge[:-1] + vkappa_bins_edge[1:])
        fvkappa = fdata[nbins+2:] * fnorm[irun]
        # fvkappa *= np.abs(vkappa_bins_mid)
        fvkappa /= dvkappa
        # print(np.sum(fvkappa*vkappa_bins_mid) / np.sum(fvkappa))
        text1 = '2D' if '2D' in pic_run else '3D'
        ax.loglog(vkappa_bins_mid[nbins//2:], fvkappa[nbins//2:],
                  color=COLORS[irun], label=text1)
        ax.loglog(vkappa_bins_mid[nbins//2:], fvkappa[nbins//2-1::-1],
                  color=COLORS[irun], linestyle='--')
        # ax.loglog(vkappa_bins_mid[nbins//2:], np.cumsum(fvkappa[nbins//2:]),
        #           color=COLORS[irun], label=text1)
        # ax.loglog(vkappa_bins_mid[nbins//2:], np.cumsum(fvkappa[nbins//2-1::-1]),
        #           color=COLORS[irun], linestyle='--')
    ax.legend(loc=1, prop={'size': 16}, ncol=1,
              shadow=False, fancybox=False, frameon=False)
    ax.set_xlim([1E-5, 1E1])
    ax.tick_params(bottom=True, top=False, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top=True)
    ax.tick_params(axis='x', which='major', direction='in', top=True)
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.tick_params(labelsize=16)

    plt.show()


def vkappa_dist_3d(plot_config):
    """Plot the distribution of vdot_kappa
    """
    bg = plot_config["bg"]
    bg_str = str(int(bg * 10)).zfill(2)
    tframe = plot_config["tframe"]
    tstart = plot_config["tstart"]
    tend = plot_config["tend"]
    species = plot_config["species"]
    pic_run = "3D-Lx150-bg" + str(bg) + "-150ppc-2048KNL"
    root_dir = "/net/scratch3/xiaocanli/reconnection/Cori_runs/"
    pic_run_dir = root_dir + pic_run + "/"
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    tindex = tframe * pic_info.fields_interval
    nframes = tend - tstart + 1

    fig = plt.figure(figsize=[3.5, 2.5])
    rect = [0.16, 0.16, 0.8, 0.8]
    ax = fig.add_axes(rect)
    fdir = '../data/vkappa_dist/' + pic_run + '/'
    for tframe in range(tstart, tend + 1):
        tindex = tframe * pic_info.fields_interval
        fname = fdir + 'vexb_kappa_dist_' + str(tindex) + '.gda'
        fdata = np.fromfile(fname, dtype=np.float32)
        nbins = int(fdata[0])
        vkappa_bins_edge = fdata[1:nbins+2]
        dvkappa = np.diff(vkappa_bins_edge)
        vkappa_bins_mid = 0.5 * (vkappa_bins_edge[:-1] + vkappa_bins_edge[1:])
        fvkappa = fdata[nbins+2:]
        # fvkappa *= np.abs(vkappa_bins_mid)
        fvkappa /= dvkappa
        fdata = div0(fvkappa[nbins//2:], fvkappa[nbins//2-1::-1]) - 1
        color = plt.cm.jet((tframe - tstart + 0.5)/float(nframes), 1)
        ax.semilogx(vkappa_bins_mid[nbins//2:], fdata, color=color,
                    linewidth=1)
        # ax.legend(loc=1, prop={'size': 16}, ncol=1,
        #           shadow=False, fancybox=False, frameon=False)
        text1 = r'\boldsymbol{v}_{\boldsymbol{E}}\cdot\boldsymbol{\kappa}'
        if tframe == 10:
            rect1 = [0.27, 0.50, 0.31, 0.33]
            ax1 = fig.add_axes(rect1)
            ax1.tick_params(bottom=True, top=False, left=True, right=True)
            ax1.tick_params(axis='x', which='minor', direction='in', top=True)
            ax1.tick_params(axis='x', which='major', direction='in', top=True)
            ax1.tick_params(axis='y', which='minor', direction='in')
            ax1.tick_params(axis='y', which='major', direction='in')
            ax1.tick_params(labelsize=8)
            p1, = ax1.loglog(vkappa_bins_mid[nbins//2:], fvkappa[nbins//2:],
                             color=COLORS[0], linewidth=1)
            ax1.loglog(vkappa_bins_mid[nbins//2:], fvkappa[nbins//2-1::-1],
                       color=COLORS[1], linewidth=1)
            ax1.set_xlim([1E-4, 1E0])
            ax1.set_ylim([1E3, 1E14])
            ax1.text(0.43, 0.77, r'$t\Omega_{ci}=100$', color='k', fontsize=8,
                     bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                     horizontalalignment='left', verticalalignment='bottom',
                     transform=ax1.transAxes)
            text2 = r'$f(' + text1 + '>0)$'
            ax1.text(0.35, 0.2, text2, color=COLORS[0], fontsize=8, rotation=-35,
                     bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                     horizontalalignment='left', verticalalignment='bottom',
                     transform=ax1.transAxes)
            text2 = r'$f(' + text1 + '<0)$'
            ax1.text(0.25, 0.0, text2, color=COLORS[1], fontsize=8, rotation=-40,
                     bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                     horizontalalignment='left', verticalalignment='bottom',
                     transform=ax1.transAxes)

    # ax.grid(True)
    ax.set_xlim([1E-5, 1E-2])
    ax.set_ylim([-0.5, 1E0])
    # ax.plot([0.002, 0.002], ax.get_ylim(), color='k',
    #         linewidth=0.5, linestyle='--')
    # ax.plot([0.0005, 0.0005], ax.get_ylim(), color='k',
    #         linewidth=0.5, linestyle='--')
    ax.plot(ax.get_xlim(), [0, 0], color='k', linewidth=1, linestyle='--')
    # ax.fill_between(ax.get_xlim(), -0.1, 0.1, color='gray', alpha=0.5,
    #                 edgecolor='none')
    ax.fill_betweenx(ax.get_ylim(), 0.0005, 0.002, color='gray', alpha=0.5,
                     edgecolor='none')
    ax.tick_params(bottom=True, top=False, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top=True)
    ax.tick_params(axis='x', which='major', direction='in', top=True)
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    xlabel = r'$|' + text1 + '|$'
    ax.set_xlabel(xlabel, fontsize=10)
    ylabel = r'$f(' + text1 + '>0)/' + 'f(' + text1 + '<0) - 1' + '$'
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(labelsize=8)

    rect_cbar = [0.2, 0.25, 0.35, 0.03]
    cax = fig.add_axes(rect_cbar)
    dtf = math.ceil(pic_info.dt_fields / 0.1) * 0.1
    sm = plt.cm.ScalarMappable(cmap=plt.cm.jet,
                               norm=plt.Normalize(vmin=tstart*dtf,
                                                  vmax=tend*dtf))
    cax.tick_params(axis='x', which='major', direction='in')
    sm._A = []
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_label(r'$t\Omega_{ci}$', fontsize=10)
    ticks = [70, 200, 300, 400]
    cbar.set_ticks(ticks)
    # cax.tick_params(labelrotation=90)
    # cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(),
    #                         rotation='vertical')
    cbar.ax.tick_params(labelsize=8)
    cax.xaxis.set_label_position('top')

    fdir = '../img/cori_3d/vdot_kappa_dist/'
    mkdir_p(fdir)
    fname = fdir + 'vdot_kappa_dist_' + 'bg' + bg_str + '_' + species + '.pdf'
    fig.savefig(fname)

    plt.show()


def compare_escape_rate(plot_config):
    """Compare the escape rate for different threshold
    """
    bg = plot_config['bg']
    bg_str = str(int(bg * 10)).zfill(2)
    bg = plot_config["bg"]
    pic_run = "3D-Lx150-bg" + str(bg) + "-150ppc-2048KNL"
    root_dir = "/net/scratch3/xiaocanli/reconnection/Cori_runs/"
    pic_run_dir = root_dir + pic_run + '/'
    species = plot_config["species"]
    tframe = plot_config["tframe"]
    tstart = plot_config["tstart"]
    tend = plot_config["tend"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    dtf = pic_info.fields_interval * pic_info.dtwpe
    dtwci = dtf * pic_info.dtwci / pic_info.dtwpe

    nframes = tend - tstart + 1
    twci = np.linspace(tstart, tend, nframes) * dtwci
    acc_rates = np.zeros(nframes)
    acc_rates_rest = np.zeros(nframes)
    nhigh_tot = np.zeros((2, nframes))
    acc_rates2 = np.zeros((3, nframes))
    acc_rates_rest2 = np.zeros((3, nframes))
    nhigh_tot2 = np.zeros((2, nframes))
    anisotropy = np.zeros((6, nframes))

    fig = plt.figure(figsize=[3.5, 2.5])
    box1 = [0.13, 0.16, 0.76, 0.8]
    ax = fig.add_axes(box1)
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    ax.set_prop_cycle('color', COLORS)

    thresholds = [0.000125, 0.00025, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.016]
    for threshold in thresholds[:5]:
        tname = str(threshold)
        tname = tname.replace('.', '_')
        fdir = '../data/cori_3d/acc_rate/' + pic_run + '/threshold_' + tname + '/'
        for tframe in range(tstart, tend + 1):
            fname = fdir + 'acc_rate_' + species + '_' + str(tframe) + '.dat'
            fdata = np.fromfile(fname)
            acc_rates[tframe - tstart] = fdata[0]
            acc_rates_rest[tframe - tstart] = fdata[1]
            nhigh_tot[:, tframe - tstart] = fdata[2:]
            fname = fdir + 'acc_rate_particle_' + species + '_' + str(tframe) + '.dat'
            fdata = np.fromfile(fname)
            acc_rates2[0, tframe - tstart] = fdata[0]
            acc_rates2[1, tframe - tstart] = fdata[1]
            acc_rates2[2, tframe - tstart] = fdata[2]
            acc_rates_rest2[0, tframe - tstart] = fdata[3]
            acc_rates_rest2[1, tframe - tstart] = fdata[4]
            acc_rates_rest2[2, tframe - tstart] = fdata[5]
            nhigh_tot2[:, tframe - tstart] = fdata[6:]
            fname = fdir + 'anisotropy_' + species + '_' + str(tframe) + '.dat'
            anisotropy[:, tframe - tstart] = np.fromfile(fname)

        esc_rates = div0(np.gradient(nhigh_tot[1, :], dtf), nhigh_tot[0, :])
        esc_rates[:6] = 0
        esc_rates2 = div0(np.gradient(nhigh_tot2[1, :], dtf), nhigh_tot2[0, :])
        esc_rates2[:6] = 0
        aniso = div0(anisotropy[0, :], anisotropy[1, :])
        kpara_ratio = aniso / (aniso + 2)

        fnorm = 100
        p1, = ax.plot(twci, esc_rates * fnorm, linewidth=1)
        ax.plot(twci, esc_rates2 * fnorm, linewidth=1,
                linestyle='--', color=p1.get_color())
    ax.set_xlim([0, twci.max()])
    ax.tick_params(bottom=True, top=False, left=True, right=False)
    ax.tick_params(axis='x', which='minor', direction='in', top=True)
    ax.tick_params(axis='x', which='major', direction='in', top=True)
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlabel(r'$t\Omega_{ci}$', fontsize=10)
    ax.set_ylabel('Rates', fontsize=10)
    ax.tick_params(labelsize=8)

    fdir = '../img/cori_3d/'
    mkdir_p(fdir)
    fname = fdir + 'spect_index_' + 'bg' + bg_str + '_' + species + '.pdf'
    # fig.savefig(fname)

    plt.show()


def comp_vsingle_vexb(plot_config):
    """compare vsingle and vexb when calculating vdot_kappa
    """
    bg = plot_config["bg"]
    tframe = plot_config["tframe"]
    pic_run = "3D-Lx150-bg" + str(bg) + "-150ppc-2048KNL"
    root_dir = "/net/scratch3/xiaocanli/reconnection/Cori_runs/"

    fig = plt.figure(figsize=[7, 5])
    rect = [0.15, 0.15, 0.8, 0.8]
    ax = fig.add_axes(rect)
    pic_run_dir = root_dir + pic_run + "/"
    species = plot_config["species"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    tindex = tframe * pic_info.fields_interval
    fdir = '../data/vkappa_dist/' + pic_run + '/'
    fname = fdir + 'vkappa_dist_' + str(tindex) + '.gda'
    fdata = np.fromfile(fname, dtype=np.float32)
    nbins = int(fdata[0])
    vkappa_bins_edge = fdata[1:nbins+2]
    dvkappa = np.diff(vkappa_bins_edge)
    vkappa_bins_mid = 0.5 * (vkappa_bins_edge[:-1] + vkappa_bins_edge[1:])
    fvkappa = fdata[nbins+2:]
    # fvkappa *= np.abs(vkappa_bins_mid)
    # fvkappa /= dvkappa
    ax.loglog(vkappa_bins_mid[nbins//2:], fvkappa[nbins//2:],
              color=COLORS[0], label='Single Fluid V')
    ax.loglog(vkappa_bins_mid[nbins//2:], fvkappa[nbins//2-1::-1],
              color=COLORS[0], linestyle='--')

    fname = fdir + 'vexb_kappa_dist_' + str(tindex) + '.gda'
    fdata = np.fromfile(fname, dtype=np.float32)
    nbins = int(fdata[0])
    vkappa_bins_edge = fdata[1:nbins+2]
    dvkappa = np.diff(vkappa_bins_edge)
    vkappa_bins_mid = 0.5 * (vkappa_bins_edge[:-1] + vkappa_bins_edge[1:])
    fvkappa = fdata[nbins+2:]
    # fvkappa *= np.abs(vkappa_bins_mid)
    # fvkappa /= dvkappa
    ax.loglog(vkappa_bins_mid[nbins//2:], fvkappa[nbins//2:],
              color=COLORS[1], label='ExB Drift')
    ax.loglog(vkappa_bins_mid[nbins//2:], fvkappa[nbins//2-1::-1],
              color=COLORS[1], linestyle='--')

    ax.legend(loc=1, prop={'size': 16}, ncol=1,
              shadow=False, fancybox=False, frameon=False)
    ax.set_xlim([1E-5, 1E2])
    ax.tick_params(bottom=True, top=False, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top=True)
    ax.tick_params(axis='x', which='major', direction='in', top=True)
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.tick_params(labelsize=16)

    plt.show()


def avg_acceleration_rate(plot_config):
    """Get the average acceleration rate for particles in exhausts
    """
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    species = plot_config["species"]
    tframe = plot_config["tframe"]
    tstart = plot_config["tstart"]
    tend = plot_config["tend"]
    threshold = plot_config["threshold"]
    tname = str(threshold)
    tname = tname.replace('.', '_')
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    dtf = pic_info.fields_interval * pic_info.dtwpe
    nx, ny, nz = pic_info.nx, pic_info.ny, pic_info.nz
    nxr2, nyr2, nzr2 = nx // 2, ny // 2, nz // 2
    nxr4, nyr4, nzr4 = nx // 4, ny // 4, nz // 4
    nxr12 = nx // 12
    nyr6 = ny // 6
    nzr8 = nz // 8
    x_di = pic_info.x_di
    y_di = pic_info.y_di
    z_di = pic_info.z_di
    xr4_di = x_di[2::4]
    yr4_di = y_di[2::4]
    zr4_di = z_di[2::4]
    xr12_di = x_di[6::12]
    yr6_di = y_di[3::6]
    zr8_di = z_di[4::8]

    nframes = tend - tstart + 1
    tindex = tframe * pic_info.fields_interval
    fname = pic_run_dir + "data-smooth/vexb_kappa_" + str(tindex) + ".gda"
    vkappa = np.fromfile(fname, dtype=np.float32)
    vkappa = vkappa.reshape((nzr2, nyr2, nxr2))
    vkappa = vkappa.reshape((nzr4, 2, nyr4, 2, nxr4, 2))
    vkappa_r = np.mean(np.mean(np.mean(vkappa, axis=5), axis=3), axis=1)
    nhigh = np.zeros((nzr4, nyr4, nxr4))
    nbands = 7
    low_band = 3
    for iband in range(low_band, nbands):
        print("Energy band: %d" % iband)
        fname = (pic_run_dir + "data-smooth2/n" + species + "_" +
                 str(iband) + "_" + str(tindex) + ".gda")
        nrho = np.fromfile(fname, dtype=np.float32)
        nhigh += nrho.reshape((nzr4, nyr4, nxr4))
    nhigh_x = np.sum(np.sum(nhigh, axis=1), axis=0)
    acc_x = np.sum(np.sum(nhigh*vkappa_r, axis=1), axis=0)
    acc_rate_x = div0(acc_x, nhigh_x)

    # Acceleration rate based curvature drift calculated from fluid
    cond1 = np.abs(vkappa_r) > threshold  # major acceleration regions
    cond2 = np.logical_not(cond1) # other regions
    fdata = np.zeros(4)
    nhigh_sum1 = np.sum(nhigh[cond1])
    nhigh_sum2 = np.sum(nhigh[cond2])
    if nhigh_sum1:
        fdata[0] = np.sum(nhigh[cond1] * vkappa_r[cond1]) / nhigh_sum1
    if nhigh_sum2:
        fdata[1] = np.sum(nhigh[cond2] * vkappa_r[cond2]) / nhigh_sum2
    fdata[2] = nhigh_sum1
    fdata[3] = nhigh_sum2

    print("Acceleration rate: %f" % fdata[0])
    print("Acceleration rate for the rest of particles: %f" % fdata[1])
    ratio = fdata[1] / fdata[0] if fdata[0] else 0
    print("Ratio: %f" % ratio)

    odir = '../data/cori_3d/acc_rate/' + pic_run + '/threshold_' + tname + '/'
    mkdir_p(odir)
    fname = odir + 'acc_rate_' + species + '_' + str(tframe) + '.dat'
    fdata.tofile(fname)

    # Acceleration rates calculated from particles
    fdir = pic_run_dir + 'spatial_acceleration_rates/'
    fname = fdir + 'spatial_acc_rates_' + species + '_' + str(tindex) + '.h5'
    fh = h5py.File(fname, 'r')
    dset = fh['particle_distribution']
    pdist = np.zeros(dset.shape, dtype=dset.dtype)
    dset.read_direct(pdist)
    grp = fh['acc_rates']
    dset = grp['Epara']
    fene_epara = np.zeros(dset.shape, dtype=dset.dtype)
    dset.read_direct(fene_epara)
    dset = grp['Eperp']
    fene_eperp = np.zeros(dset.shape, dtype=dset.dtype)
    dset.read_direct(fene_eperp)
    dset = grp['Curvature']
    fene_curv = np.zeros(dset.shape, dtype=dset.dtype)
    dset.read_direct(fene_curv)
    fh.close()
    ie = 14
    pdist_high_sum = np.sum(pdist[:, :, :, ie:], axis=3)
    fene_epara_sum = np.sum(fene_epara[:, :, :, ie:], axis=3)
    fene_eperp_sum = np.sum(fene_eperp[:, :, :, ie:], axis=3)
    fene_curv_sum = np.sum(fene_curv[:, :, :, ie:], axis=3)
    zv, yv, xv = np.meshgrid(zr4_di, yr4_di, xr4_di, indexing='ij')

    # prepare data for MultilinearInterpolator
    _, _, _, nbands = fene_curv.shape
    smin_h = [zr8_di[0], yr6_di[0], xr12_di[0]]
    smax_h = [zr8_di[-1], yr6_di[-1], xr12_di[-1]]
    orders = [nzr8, nyr6, nxr12]
    coord = np.vstack([zv.flatten(), yv.flatten(), xv.flatten()])
    fn = MultilinearInterpolator(smin_h, smax_h, orders)

    # interpolate the acceleration for high-energy particles
    fn.set_values(np.atleast_2d(pdist_high_sum.flatten()))
    pdist_high_new = fn(coord).reshape([nzr4, nyr4, nxr4])
    fn.set_values(np.atleast_2d(fene_epara_sum.flatten()))
    fene_epara_new = fn(coord).reshape([nzr4, nyr4, nxr4])
    fn.set_values(np.atleast_2d(fene_eperp_sum.flatten()))
    fene_eperp_new = fn(coord).reshape([nzr4, nyr4, nxr4])
    fn.set_values(np.atleast_2d(fene_curv_sum.flatten()))
    fene_curv_new = fn(coord).reshape([nzr4, nyr4, nxr4])

    # plt.imshow(fene_curv_new[:, 0, :],
    #            vmin=-1E-4, vmax=1E-4,
    #            cmap=plt.cm.seismic, aspect='auto',
    #            origin='lower', interpolation='none')
    # plt.show()

    fdata = np.zeros(8)
    pdist_sum1 = np.sum(pdist_high_new[cond1])
    pdist_sum2 = np.sum(pdist_high_new[cond2])
    if pdist_sum1:
        fdata[0] = np.sum(fene_epara_new[cond1]) / pdist_sum1
        fdata[1] = np.sum(fene_eperp_new[cond1]) / pdist_sum1
        fdata[2] = np.sum(fene_curv_new[cond1]) / pdist_sum1
    if pdist_sum2:
        fdata[3] = np.sum(fene_epara_new[cond2]) / pdist_sum2
        fdata[4] = np.sum(fene_eperp_new[cond2]) / pdist_sum2
        fdata[5] = np.sum(fene_curv_new[cond2]) / pdist_sum2
    fdata[6] = pdist_sum1
    fdata[7] = pdist_sum2
    del pdist_high_sum, fene_epara_sum, fene_eperp_sum, fene_curv_sum
    del pdist_high_new, fene_epara_new, fene_eperp_new, fene_curv_new

    fname = odir + 'acc_rate_particle_' + species + '_' + str(tframe) + '.dat'
    fdata.tofile(fname)

    # interpolate the acceleration for all energy bands
    fdata = np.zeros([8, nbands])
    for iband in range(nbands):
        fn.set_values(np.atleast_2d(pdist[:, :, :, iband].flatten()))
        fnew = fn(coord).reshape([nzr4, nyr4, nxr4])
        fdata[6, iband] = np.sum(fnew[cond1])
        fdata[7, iband] = np.sum(fnew[cond2])
    del pdist

    for iband in range(nbands):
        fn.set_values(np.atleast_2d(fene_epara[:, :, :, iband].flatten()))
        fnew = fn(coord).reshape([nzr4, nyr4, nxr4])
        if fdata[6, iband]:
            fdata[0, iband] = np.sum(fnew[cond1]) / fdata[6, iband]
        if fdata[7, iband]:
            fdata[3, iband] = np.sum(fnew[cond2]) / fdata[7, iband]
    del fene_epara

    for iband in range(nbands):
        fn.set_values(np.atleast_2d(fene_eperp[:, :, :, iband].flatten()))
        fnew = fn(coord).reshape([nzr4, nyr4, nxr4])
        if fdata[6, iband]:
            fdata[1, iband] = np.sum(fnew[cond1]) / fdata[6, iband]
        if fdata[7, iband]:
            fdata[4, iband] = np.sum(fnew[cond2]) / fdata[7, iband]
    del fene_eperp

    for iband in range(nbands):
        fn.set_values(np.atleast_2d(fene_curv[:, :, :, iband].flatten()))
        fnew = fn(coord).reshape([nzr4, nyr4, nxr4])
        # plt.imshow(fnew[:, 134, :],
        #            vmin=-1E-5, vmax=1E-5,
        #            cmap=plt.cm.seismic, aspect='auto',
        #            origin='lower', interpolation='bicubic')
        # plt.show()
        if fdata[6, iband]:
            fdata[2, iband] = np.sum(fnew[cond1]) / fdata[6, iband]
        if fdata[7, iband]:
            fdata[5, iband] = np.sum(fnew[cond2]) / fdata[7, iband]
    del fene_curv

    fname = odir + 'acc_rate_band' + species + '_' + str(tframe) + '.dat'
    fdata.tofile(fname)

    # # Anisotropy calculated from particles
    # fdir = pic_run_dir + 'spatial_anisotropy/'
    # fname = fdir + 'spatial_aniso_' + species + '_' + str(tindex) + '.h5'
    # fh = h5py.File(fname, 'r')
    # grp = fh['anisotropy']
    # dset = grp['ppara']
    # ppara = np.zeros(dset.shape, dtype=dset.dtype)
    # dset.read_direct(ppara)
    # dset = grp['pperp']
    # pperp = np.zeros(dset.shape, dtype=dset.dtype)
    # dset.read_direct(pperp)
    # fh.close()

    # # interpolate the anisotropy for high-energy particles
    # ppara_high_sum = np.sum(ppara[:, :, :, ie:], axis=3)
    # pperp_high_sum = np.sum(pperp[:, :, :, ie:], axis=3)
    # fn.set_values(np.atleast_2d(ppara_high_sum.flatten()))
    # ppara_high_new = fn(coord).reshape([nzr4, nyr4, nxr4])
    # fn.set_values(np.atleast_2d(pperp_high_sum.flatten()))
    # pperp_high_new = fn(coord).reshape([nzr4, nyr4, nxr4])
    # aniso = np.zeros(6)
    # if pdist_sum1:
    #     aniso[0] = np.sum(ppara_high_new[cond1]) / pdist_sum1
    #     aniso[1] = np.sum(pperp_high_new[cond1]) / pdist_sum1
    # if pdist_sum2:
    #     aniso[2] = np.sum(ppara_high_new[cond2]) / pdist_sum2
    #     aniso[3] = np.sum(pperp_high_new[cond2]) / pdist_sum2
    # aniso[4] = pdist_sum1
    # aniso[5] = pdist_sum2
    # del ppara_high_sum, pperp_high_sum
    # del ppara_high_new, pperp_high_new

    # fname = odir + 'anisotropy_' + species + '_' + str(tframe) + '.dat'
    # aniso.tofile(fname)

    # # interpolate the acceleration for all energy bands
    # aniso = np.zeros([6, nbands])
    # for iband in range(nbands):
    #     fn.set_values(np.atleast_2d(ppara[:, :, :, iband].flatten()))
    #     fnew = fn(coord).reshape([nzr4, nyr4, nxr4])
    #     if fdata[6, iband]:
    #         aniso[0, iband] = np.sum(fnew[cond1]) / fdata[6, iband]
    #     if fdata[7, iband]:
    #         aniso[1, iband] = np.sum(fnew[cond2]) / fdata[7, iband]
    # del ppara

    # for iband in range(nbands):
    #     fn.set_values(np.atleast_2d(pperp[:, :, :, iband].flatten()))
    #     fnew = fn(coord).reshape([nzr4, nyr4, nxr4])
    #     if fdata[6, iband]:
    #         aniso[0, iband] = np.sum(fnew[cond1]) / fdata[6, iband]
    #     if fdata[7, iband]:
    #         aniso[1, iband] = np.sum(fnew[cond2]) / fdata[7, iband]
    # del pperp, fnew, coord, fn

    # fname = odir + 'anisotropy_band_' + species + '_' + str(tframe) + '.dat'
    # fdata.tofile(fname)


def plot_acc_esc(plot_config):
    """Plot acceleration and escape rates
    """
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    species = plot_config["species"]
    tframe = plot_config["tframe"]
    tstart = plot_config["tstart"]
    tend = plot_config["tend"]
    threshold = plot_config["threshold"]
    tname = str(threshold)
    tname = tname.replace('.', '_')
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    dtf = pic_info.fields_interval * pic_info.dtwpe
    gama = 1.0 / math.sqrt(1.0 - 3 * pic_info.vthe**2)
    ethe = gama - 1.0
    if species in ['e', 'electron']:
        vth = pic_info.vthe
        sname = 'electron'
    else:
        vth = pic_info.vthi
        sname = 'ion'
    ncells = pic_info.nx * pic_info.ny * pic_info.nz
    smime = math.sqrt(pic_info.mime)
    lx_de = pic_info.lx_di * smime
    ly_de = pic_info.ly_di * smime
    lz_de = pic_info.lz_di * smime
    vol = lx_de * ly_de * lz_de
    weight = vol / (ncells * pic_info.nppc)
    ncells_local = pic_info.nx_zone * pic_info.ny_zone * pic_info.nz_zone
    # we interpolate the local distribution from 256*256*160 to 768*384*320
    norm = 4**3 / ncells_local

    nframes = tend - tstart + 1
    nbands = 21
    acc_rates = np.zeros(nframes)
    acc_rates_rest = np.zeros(nframes)
    nhigh_tot = np.zeros((2, nframes))
    acc_rates2 = np.zeros((3, nframes))
    acc_rates_rest2 = np.zeros((3, nframes))
    nhigh_tot2 = np.zeros((2, nframes))
    anisotropy = np.zeros((6, nframes))
    acc_rates_band = np.zeros((3, nframes, nbands))
    acc_rates_rest_band = np.zeros((3, nframes, nbands))
    nhigh_band = np.zeros((2, nframes, nbands))
    ntot_rec_layer = np.zeros(nframes)

    nbins_high = nbands - 1
    emin_high, emax_high = 1E-3, 1E1  # for high-energy particles
    emin_high_log = math.log10(emin_high)
    emax_high_log = math.log10(emax_high)
    delog = (emax_high_log - emin_high_log) / nbins_high
    emax_high_adjusted = 10**(emax_high_log + delog)
    ebins_high = np.logspace(emin_high_log, math.log10(emax_high_adjusted),
                             nbins_high+2)
    ebins_mid = 0.5 * (ebins_high[:-1] + ebins_high[1:])
    ebins_mid /= ethe
    # for i, ene in enumerate(ebins_mid):
    #     print(i, ene)
    for i, ene in enumerate(ebins_high/ethe):
        print(i, ene)

    fdir = '../data/cori_3d/acc_rate/' + pic_run + '/threshold_' + tname + '/'
    for tframe in range(tstart, tend + 1):
        tshift = tframe - tstart
        fname = fdir + 'acc_rate_' + species + '_' + str(tframe) + '.dat'
        fdata = np.fromfile(fname)
        acc_rates[tshift] = fdata[0]
        acc_rates_rest[tshift] = fdata[1]
        nhigh_tot[:, tshift] = fdata[2:]
        fname = fdir + 'acc_rate_particle_' + species + '_' + str(tframe) + '.dat'
        fdata = np.fromfile(fname)
        acc_rates2[:3, tshift] = fdata[:3]
        acc_rates_rest2[:3, tshift] = fdata[3:6]
        nhigh_tot2[:, tshift] = fdata[6:]
        fname = fdir + 'anisotropy_' + species + '_' + str(tframe) + '.dat'
        anisotropy[:, tshift] = np.fromfile(fname)
        fname = fdir + 'acc_rate_band' + species + '_' + str(tframe) + '.dat'
        fdata = np.fromfile(fname).reshape((8, nbands))
        acc_rates_band[:3, tshift, :] = fdata[:3, :]
        acc_rates_rest_band[:3, tshift, :] = fdata[3:6, :]
        nhigh_band[:, tshift, :] = fdata[6:, :]
        tindex = pic_info.particle_interval * tframe
        fname = (pic_run_dir + "spectrum_reconnection_layer/spectrum_layer_" +
                 sname + "_" + str(tindex) + ".dat")
        spect = np.fromfile(fname, dtype=np.float32)
        ntot_rec_layer[tshift] = np.sum(spect)

    esc_rates = div0(np.gradient(nhigh_tot[1, :], dtf), nhigh_tot[0, :])
    esc_rates[:6] = 0
    esc_rates2 = div0(np.gradient(nhigh_tot2[1, :], dtf), nhigh_tot2[0, :])
    esc_rates2[:6] = 0
    esc_rates_band = div0(np.gradient(nhigh_band[1, :, :], dtf, axis=0),
                          nhigh_band[0, :, :])
    # esc_rates_band = div0(np.gradient(nhigh_band[0, :, :], dtf, axis=0),
    #                       nhigh_band[0, :, :])
    sband = 10
    esc_rates3 = div0(np.gradient(np.sum(nhigh_band[1, :, sband:], axis=1), dtf),
                      np.sum(nhigh_band[0, :, sband:], axis=1))
    esc_rates3[:6] = 0
    aniso = div0(anisotropy[0, :], anisotropy[1, :])
    kpara_ratio = aniso / (aniso + 2)

    ntot_rec_layer *= weight
    ntot_main_acc = np.sum(nhigh_band[0, :, :], axis=1) * norm
    ntot_rest = ntot_rec_layer / pic_info.stride_particle_dump - ntot_main_acc
    esc_rates4 = div0(np.gradient(ntot_rest, dtf), ntot_main_acc)
    esc_rates4[:6] = 0

    fig = plt.figure(figsize=[7, 5])
    box1 = [0.13, 0.18, 0.82, 0.78]
    ax = fig.add_axes(box1)
    # ax.plot(esc_rates_band[6:].T)
    # ax.plot(div0(acc_rates2[2], acc_rates_rest2[2]))
    # ax.plot(np.gradient(nhigh_band[0, :, :], axis=0))
    # ax.plot(ntot_main_acc)
    # ax.plot(np.gradient(nhigh_tot2, axis=1).T)
    # ax.plot(esc_rates)
    # ax.plot(esc_rates2)
    # ax.plot(esc_rates3)
    # ax.plot(esc_rates4)
    # for tframe in range(30, 40):
    #     p1, = ax.plot([10, 20], [acc_rates2[2, tframe], acc_rates2[2, tframe]],
    #                   linestyle='--')
    #     ax.plot(acc_rates_band[2, tframe, :], color=p1.get_color())
    #     # p1, = ax.plot([10, 20], [esc_rates2[tframe], esc_rates2[tframe]],
    #     #               linestyle='--')
    #     # ax.plot(esc_rates_band[tframe, :], color=p1.get_color())
    #     # ax.plot(div0(esc_rates_band[tframe, :], acc_rates_band[2, tframe, :]))
    #     # p1, = ax.semilogy(nhigh_band[0, tframe, :], linestyle='--')
    #     # p2, = ax.semilogy(nhigh_band[1, tframe, :], color=p1.get_color())
    # ax.set_xlim([10, 20])
    # ax.set_ylim([0, 4E-3])
    ax.tick_params(bottom=True, top=False, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top=True)
    ax.tick_params(axis='x', which='major', direction='in', top=True)
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')

    plt.show()


def spectral_index_pub(plot_config):
    """Plot spectral index for publication
    """
    bg = plot_config['bg']
    bg_str = str(int(bg * 10)).zfill(2)
    bg = plot_config["bg"]
    pic_run = "3D-Lx150-bg" + str(bg) + "-150ppc-2048KNL"
    root_dir = "/net/scratch3/xiaocanli/reconnection/Cori_runs/"
    pic_run_dir = root_dir + pic_run + '/'
    species = plot_config["species"]
    tframe = plot_config["tframe"]
    tstart = plot_config["tstart"]
    tend = plot_config["tend"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    dtf = pic_info.fields_interval * pic_info.dtwpe
    dtwci = dtf * pic_info.dtwci / pic_info.dtwpe
    if species in ['e', 'electron']:
        vth = pic_info.vthe
        sname = 'electron'
    else:
        vth = pic_info.vthi
        sname = 'ion'
    ncells = pic_info.nx * pic_info.ny * pic_info.nz
    smime = math.sqrt(pic_info.mime)
    lx_de = pic_info.lx_di * smime
    ly_de = pic_info.ly_di * smime
    lz_de = pic_info.lz_di * smime
    vol = lx_de * ly_de * lz_de
    weight = vol / (ncells * pic_info.nppc)
    ncells_local = pic_info.nx_zone * pic_info.ny_zone * pic_info.nz_zone
    # we interpolate the local distribution from 256*256*160 to 768*384*320
    norm = 4**3 / ncells_local

    nframes = tend - tstart + 1
    twci = np.linspace(tstart, tend, nframes) * dtwci
    acc_rates = np.zeros(nframes)
    acc_rates_rest = np.zeros(nframes)
    nhigh_tot = np.zeros((2, nframes))
    acc_rates2 = np.zeros((3, nframes))
    acc_rates_rest2 = np.zeros((3, nframes))
    nhigh_tot2 = np.zeros((2, nframes))
    anisotropy = np.zeros((6, nframes))
    nbands = 21
    acc_rates_band = np.zeros((3, nframes, nbands))
    acc_rates_rest_band = np.zeros((3, nframes, nbands))
    nhigh_band = np.zeros((2, nframes, nbands))
    ntot_rec_layer = np.zeros(nframes)

    thresholds = [0.0005, 0.001, 0.002]
    fig = plt.figure(figsize=[3.5, 5.0])
    rect = [0.13, 0.54, 0.82, 0.43]
    ax = fig.add_axes(rect)
    rect[1] -= rect[3] + 0.02
    ax1 = fig.add_axes(rect)
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    ax.set_prop_cycle('color', COLORS)
    ax1.set_prop_cycle('color', COLORS)
    for ith, threshold in enumerate(thresholds):
        tname = str(threshold)
        tname = tname.replace('.', '_')
        fdir = '../data/cori_3d/acc_rate/' + pic_run + '/threshold_' + tname + '/'
        for tframe in range(tstart, tend + 1):
            tshift = tframe - tstart
            fname = fdir + 'acc_rate_' + species + '_' + str(tframe) + '.dat'
            fdata = np.fromfile(fname)
            acc_rates[tshift] = fdata[0]
            acc_rates_rest[tshift] = fdata[1]
            nhigh_tot[:, tshift] = fdata[2:]
            fname = fdir + 'acc_rate_particle_' + species + '_' + str(tframe) + '.dat'
            fdata = np.fromfile(fname)
            acc_rates2[0, tshift] = fdata[0]
            acc_rates2[1, tshift] = fdata[1]
            acc_rates2[2, tshift] = fdata[2]
            acc_rates_rest2[0, tshift] = fdata[3]
            acc_rates_rest2[1, tshift] = fdata[4]
            acc_rates_rest2[2, tshift] = fdata[5]
            nhigh_tot2[:, tshift] = fdata[6:]
            fname = fdir + 'anisotropy_' + species + '_' + str(tframe) + '.dat'
            anisotropy[:, tshift] = np.fromfile(fname)
            fname = fdir + 'acc_rate_band' + species + '_' + str(tframe) + '.dat'
            fdata = np.fromfile(fname).reshape((8, nbands))
            acc_rates_band[:, tshift, :] = fdata[:3, :]
            acc_rates_rest_band[:3, tshift, :] = fdata[3:6, :]
            nhigh_band[:, tshift, :] = fdata[6:, :]
            tindex = pic_info.particle_interval * tframe
            fname = (pic_run_dir + "spectrum_reconnection_layer/spectrum_layer_" +
                     sname + "_" + str(tindex) + ".dat")
            spect = np.fromfile(fname, dtype=np.float32)
            ntot_rec_layer[tshift] = np.sum(spect)

        # Escape rate based on particle band data
        esc_rates = div0(np.gradient(nhigh_tot[1, :], dtf), nhigh_tot[0, :])
        esc_rates[:6] = 0

        # Escape rate based
        ene_shift = np.exp((acc_rates_rest2[0, :] + acc_rates_rest2[1, :]) * dtf) - 1
        flux_increase = (1 + ene_shift)**4 - 1
        nhigh_escape = np.gradient(nhigh_tot2[1, :], dtf)
        # nhigh_escape = np.gradient(nhigh_tot2[1, :] * (1 - flux_increase), dtf)
        esc_rates2 = div0(nhigh_escape, nhigh_tot2[0, :])
        # esc_rates2[:6] = 0
        aniso = div0(anisotropy[0, :], anisotropy[1, :])
        kpara_ratio = aniso / (aniso + 2)

        # Escape rate based on banded particle data in and out of
        # the major acceleration region
        sband = 14
        esc_rates3 = div0(np.gradient(np.sum(nhigh_band[1, :, sband:], axis=1), dtf),
                          np.sum(nhigh_band[0, :, sband:], axis=1))
        esc_rates3[:5] = 0
        acc_rates3 = np.mean(acc_rates_band[:, :, 12:-2], axis=2)

        # Escape rate based on all particles in the reconnection layer
        ntot_rec_layer *= weight
        ntot_main_acc = np.sum(nhigh_band[0, :, :], axis=1) * norm
        ntot_rest = ntot_rec_layer / pic_info.stride_particle_dump - ntot_main_acc
        esc_rates4 = div0(np.gradient(ntot_rest, dtf), ntot_main_acc)
        esc_rates4[:6] = 0

        # acc_rates2[:, :5] = 0

        if ith == 1:
            fnorm = 100
            ax.plot(twci, esc_rates2 * fnorm, linewidth=1)
            ax.plot(twci, (acc_rates2[0, :] + acc_rates2[1, :]) * 3 * fnorm,
                    linewidth=1)
        ax1.plot(twci, div0(esc_rates2, acc_rates2[0, :] + acc_rates2[1, :]) + 1,
                 linewidth=1, marker='o', markersize=4)

    # ax.set_xlim([0, twci.max()])
    ax.set_xlim([0, 250])
    ax.set_ylim([0, 0.6])
    ax.tick_params(bottom=True, top=False, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top=True)
    ax.tick_params(axis='x', which='major', direction='in', top=True)
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_ylabel(r'Rates/$10^{-2}$', fontsize=10)
    ax.tick_params(labelsize=8)
    ax.text(0.48, 0.05, r'$1/\tau_\text{esc}$', color=COLORS[0], fontsize=10,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='right', verticalalignment='bottom',
            transform=ax.transAxes)
    ax.text(0.52, 0.05, r'$3\alpha$', color=COLORS[1], fontsize=10,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes)
    ax.tick_params(axis='x', labelbottom=False)

    ax1.set_xlim(ax.get_xlim())
    ax1.set_ylim([0.5, 10])
    ax1.plot(ax1.get_xlim(), [4, 4], linewidth=0.5, linestyle='--', color='k')
    ax1.tick_params(bottom=True, top=True, left=True, right=True)
    ax1.tick_params(axis='x', which='minor', direction='in', top=True)
    ax1.tick_params(axis='x', which='major', direction='in', top=True)
    ax1.tick_params(axis='y', which='minor', direction='in')
    ax1.tick_params(axis='y', which='major', direction='in')
    text1 = r'$1+(\alpha\tau_\text{esc})^{-1}$'
    ax1.set_ylabel(text1, fontsize=10)
    ax1.tick_params(labelsize=8)
    # ax1.text(0.6, 0.67, text1, color=COLORS[2], fontsize=10,
    #          bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
    #          horizontalalignment='left', verticalalignment='bottom',
    #          transform=ax1.transAxes)
    text1 = r'$|\boldsymbol{v}_{\boldsymbol{E}}\cdot\boldsymbol{\kappa}|>$'
    ax1.text(0.43, 0.79, text1, color='k', fontsize=10,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='center', verticalalignment='bottom',
             transform=ax1.transAxes)
    ax1.text(0.6, 0.87, '0.0005', color=COLORS[0], fontsize=10,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='center', verticalalignment='bottom',
             transform=ax1.transAxes)
    ax1.text(0.6, 0.79, '0.001', color=COLORS[1], fontsize=10,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='center', verticalalignment='bottom',
             transform=ax1.transAxes)
    ax1.text(0.6, 0.71, '0.002', color=COLORS[2], fontsize=10,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='center', verticalalignment='bottom',
             transform=ax1.transAxes)
    ax1.set_xlabel(r'$t\Omega_{ci}$', fontsize=10)

    # v.kappa distribution
    tframe = 10
    tindex = tframe * pic_info.fields_interval
    fdir = '../data/vkappa_dist/' + pic_run + '/'
    fname = fdir + 'vexb_kappa_dist_' + str(tindex) + '.gda'
    fdata = np.fromfile(fname, dtype=np.float32)
    nbins = int(fdata[0])
    vkappa_bins_edge = fdata[1:nbins+2]
    dvkappa = np.diff(vkappa_bins_edge)
    vkappa_bins_mid = 0.5 * (vkappa_bins_edge[:-1] + vkappa_bins_edge[1:])
    fvkappa = fdata[nbins+2:]
    # fvkappa *= np.abs(vkappa_bins_mid)
    fvkappa /= dvkappa
    rect1 = [0.55, 0.8, 0.37, 0.15]
    ax2 = fig.add_axes(rect1)
    ax2.tick_params(bottom=True, top=False, left=True, right=True)
    ax2.tick_params(axis='x', which='minor', direction='in', top=False, bottom=False)
    ax2.tick_params(axis='x', which='major', direction='in', top=True)
    ax2.tick_params(axis='y', which='minor', direction='in', left=False)
    ax2.tick_params(axis='y', which='major', direction='in')
    ax2.tick_params(labelsize=8)
    p1, = ax2.loglog(vkappa_bins_mid[nbins//2:], fvkappa[nbins//2:],
                     color='k', linewidth=1)
    ax2.loglog(vkappa_bins_mid[nbins//2:], fvkappa[nbins//2-1::-1],
               color='k', linewidth=1, linestyle='--')
    ax2.set_xlim([1E-4, 1E0])
    ax2.set_ylim([1E3, 1E14])
    ax2.plot([1E-3, 1E-3], ax2.get_ylim(), color='k', linewidth=0.5,
             linestyle=':')
    ax2.text(0.53, 0.77, r'$t\Omega_{ci}=100$', color='k', fontsize=8,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='left', verticalalignment='bottom',
             transform=ax2.transAxes)
    text1 = r'\boldsymbol{v}_{\boldsymbol{E}}\cdot\boldsymbol{\kappa}'
    text2 = r'$f(' + text1 + '>0)$'
    ax2.text(0.35, 0.25, text2, color='k', fontsize=8, rotation=-29,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='left', verticalalignment='bottom',
             transform=ax2.transAxes)
    text2 = r'$f(' + text1 + '<0)$'
    ax2.text(0.25, 0.0, text2, color='k', fontsize=8, rotation=-33,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='left', verticalalignment='bottom',
             transform=ax2.transAxes)
    xlabel = r'$|' + text1 + '|$'
    ax2.set_xlabel(xlabel, fontsize=8)
    ax2.tick_params(labelsize=6)

    fdir = '../img/cori_3d/'
    mkdir_p(fdir)
    fname = fdir + 'spect_index_' + 'bg' + bg_str + '_' + species + '.pdf'
    fig.savefig(fname)

    plt.show()


def solve_energy_equation(plot_config):
    """Solve the energy continuity equation
    """
    bg = plot_config['bg']
    bg_str = str(int(bg * 10)).zfill(2)
    bg = plot_config["bg"]
    pic_run = "3D-Lx150-bg" + str(bg) + "-150ppc-2048KNL"
    root_dir = "/net/scratch3/xiaocanli/reconnection/Cori_runs/"
    pic_run_dir = root_dir + pic_run + '/'
    species = plot_config["species"]
    tframe = plot_config["tframe"]
    tstart = plot_config["tstart"]
    tend = plot_config["tend"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    dtf = pic_info.fields_interval * pic_info.dtwpe
    dtwci = dtf * pic_info.dtwci / pic_info.dtwpe
    if species in ['e', 'electron']:
        vth = pic_info.vthe
        sname = 'electron'
    else:
        vth = pic_info.vthi
        sname = 'ion'
    ncells = pic_info.nx * pic_info.ny * pic_info.nz
    smime = math.sqrt(pic_info.mime)
    lx_de = pic_info.lx_di * smime
    ly_de = pic_info.ly_di * smime
    lz_de = pic_info.lz_di * smime
    vol = lx_de * ly_de * lz_de
    weight = vol / (ncells * pic_info.nppc)
    ncells_local = pic_info.nx_zone * pic_info.ny_zone * pic_info.nz_zone
    # we interpolate the local distribution from 256*256*160 to 768*384*320
    norm = 4**3 / ncells_local

    nx, = pic_info.x_di.shape
    ny, = pic_info.y_di.shape
    nz, = pic_info.z_di.shape
    nxr2 = nx // 2
    nyr2 = ny // 2
    nzr2 = nz // 2
    b0 = pic_info.b0
    va = pic_info.dtwce * math.sqrt(1.0 / pic_info.mime) / pic_info.dtwpe
    nframes = tend - tstart + 1
    tframes = np.asarray(range(tstart, tend + 1))
    twci = tframes * math.ceil(pic_info.dt_fields)

    vthe = pic_info.vthe
    gama = 1.0 / math.sqrt(1.0 - 3 * vthe**2)
    ethe = gama - 1.0
    vthi = pic_info.vthi
    gama = 1.0 / math.sqrt(1.0 - 3 * vthi**2)
    ethi = gama - 1.0
    nbins = 1000
    ebins = np.logspace(-6, 4, nbins)
    ebins_e = ebins / ethe
    ebins_i = ebins / ethi

    # reconnection rate as particle injection rate
    bx_fluxes = np.zeros([2, nframes, nxr2])
    for tframe in tframes:
        fdir = '../data/cori_3d/bx_flux/' + pic_run + '/'
        fname = fdir + 'bx_flux_' + str(tframe) + '.dat'
        bx_flux = np.fromfile(fname)
        bx_fluxes[:, tframe-tstart, :] = bx_flux.reshape([2, -1])
    bx_flux_diff = np.gradient(bx_fluxes, axis=1) / dtf
    rrate_mean = np.mean(bx_flux_diff, axis=2).T
    rrate_std = np.std(bx_flux_diff, axis=2).T
    rrate_norm = b0 * pic_info.ly_di * math.sqrt(pic_info.mime)
    rrate_mean = np.abs(rrate_mean / rrate_norm)
    rrate = np.mean(rrate_mean, axis=1)

    # particle acceleration and escape rate
    acc_rates2 = np.zeros((3, nframes))
    acc_rates_rest2 = np.zeros((3, nframes))
    nhigh_tot2 = np.zeros((2, nframes))
    nbins_high = 20
    acc_rates_band = np.zeros((3, nbins_high+1, nframes))
    acc_rates_rest_band = np.zeros((3, nbins_high+1, nframes))
    nhigh_tot_band = np.zeros((2, nbins_high+1, nframes))
    nbands = nbins_high + 1
    nhigh_band = np.zeros((2, nframes, nbands))
    ntot_rec_layer = np.zeros(nframes)

    # thresholds = [0.0005, 0.001, 0.002]
    thresholds = [0.001]
    for ith, threshold in enumerate(thresholds):
        tname = str(threshold)
        tname = tname.replace('.', '_')
        fdir = '../data/cori_3d/acc_rate/' + pic_run + '/threshold_' + tname + '/'
        for tframe in range(tstart, tend + 1):
            fname = fdir + 'acc_rate_particle_' + species + '_' + str(tframe) + '.dat'
            fdata = np.fromfile(fname)
            iframe = tframe - tstart
            acc_rates2[0, iframe] = fdata[0]
            acc_rates2[1, iframe] = fdata[1]
            acc_rates2[2, iframe] = fdata[2]
            acc_rates_rest2[0, iframe] = fdata[3]
            acc_rates_rest2[1, iframe] = fdata[4]
            acc_rates_rest2[2, iframe] = fdata[5]
            nhigh_tot2[:, iframe] = fdata[6:]
            fname = fdir + 'acc_rate_band' + species + '_' + str(tframe) + '.dat'
            fdata = np.fromfile(fname).reshape([8, -1])
            acc_rates_band[0, :, iframe] = fdata[0]
            acc_rates_band[1, :, iframe] = fdata[1]
            acc_rates_band[2, :, iframe] = fdata[2]
            acc_rates_rest_band[0, :, iframe] = fdata[3]
            acc_rates_rest_band[1, :, iframe] = fdata[4]
            acc_rates_rest_band[2, :, iframe] = fdata[5]
            nhigh_tot_band[:, :, iframe] = fdata[6:]
            nhigh_band[:, iframe, :] = fdata[6:, :]
            tindex = pic_info.particle_interval * tframe
            fname = (pic_run_dir + "spectrum_reconnection_layer/spectrum_layer_" +
                     sname + "_" + str(tindex) + ".dat")
            spect = np.fromfile(fname, dtype=np.float32)
            ntot_rec_layer[iframe] = np.sum(spect)

        esc_rates2 = div0(np.gradient(nhigh_tot2[1, :], dtf), nhigh_tot2[0, :])
        esc_rates2[:6] = 0
        esc_rates_band = div0(np.gradient(nhigh_tot_band[1, :, :], dtf, axis=1),
                              nhigh_tot_band[0, :, :])
        acc_rates_band[:, :, :6] = 0
        esc_rates_band[:, :6] = 0

        ntot_rec_layer *= weight
        ntot_main_acc = np.sum(nhigh_band[0, :, :], axis=1) * norm
        ntot_rest = ntot_rec_layer / pic_info.stride_particle_dump - ntot_main_acc
        esc_rates4 = div0(np.gradient(ntot_rest, dtf), ntot_main_acc)
        esc_rates4[:6] = 0

        emin_high, emax_high = 1E-3, 1E1  # for high-energy particles
        emin_high_log = math.log10(emin_high)
        emax_high_log = math.log10(emax_high)
        delog = (emax_high_log - emin_high_log) / nbins_high
        emax_high_adjusted = 10**(emax_high_log + delog)
        ebins_high = np.logspace(emin_high_log, math.log10(emax_high_adjusted),
                                 nbins_high+2)
        ebins_mid = 0.5 * (ebins_high[:-1] + ebins_high[1:])
        ebins_mid /= ethe

        esc_rates_band[-1] = esc_rates_band[-2]
        # plt.semilogx(ebins_mid, esc_rates_band)
        # plt.xlim([1E-1, 1E3])
        # plt.show()

        ene_start = np.argmax(ebins_e > ebins_mid[0])
        ene_end = np.argmin(ebins_e < ebins_mid[-1])
        ene01, _ = find_nearest(ebins_e, 0.01)
        acc_rates_band_new = np.zeros([nframes, nbins])
        esc_rates_band_new = np.zeros([nframes, nbins])
        for tframe in range(nframes):
            # f = interp1d(ebins_mid, acc_rates_band[2, :, tframe], kind='linear')
            f = interp1d(ebins_mid,
                         acc_rates_band[0, :, tframe] + acc_rates_band[1, :, tframe],
                         kind='linear')
            acc_rates_band_new[tframe, ene_start:ene_end] = f(ebins_e[ene_start:ene_end])
            f = interp1d(ebins_mid, esc_rates_band[:, tframe], kind='linear')
            esc_rates_band_new[tframe, ene_start:ene_end] = f(ebins_e[ene_start:ene_end])
            nbins_extend = ene_start - ene01
            acc_rates_band_new[tframe, :ene_start] = acc_rates_band_new[tframe, ene_start]
            pindex = np.argmax(esc_rates_band_new[tframe] > 0) + 6
            nbins_extend = pindex - ene01
            esc_rates_band_new[tframe, ene01:pindex] = esc_rates_band_new[tframe, pindex]
            # esc_rates_band_new[tframe, ene01:pindex] = \
            #         esc_rates_band_new[tframe, pindex] * np.linspace(0, 1, nbins_extend)
        # plt.semilogx(ebins_e, acc_rates_band_new.T)
        # plt.xlim([1E-2, 1E3])
        # plt.show()

        fthermal = fitting_funcs.func_maxwellian(ebins, 1E12, 1.0/(ethe))
        finit = np.copy(fthermal)
        f = np.zeros(nbins)
        dtf = pic_info.dt_fields * pic_info.dtwpe / pic_info.dtwci
        ntf, = esc_rates2.shape
        stride = 1000
        ntf = 40
        nframes_new = stride * ntf
        dtf /= stride

        # interpolate the rates
        told = np.linspace(1, ntf, ntf)
        tnew = np.linspace(1, ntf, nframes_new)
        fin = interp1d(told, acc_rates2[0, :ntf] + acc_rates2[1, :ntf], kind='linear')
        acc_rate_new = fin(tnew)
        fin = interp1d(told, esc_rates2[:ntf], kind='linear')
        esc_rate_new = fin(tnew)
        fin = interp1d(told, esc_rates4[:ntf], kind='linear')
        esc_rates4_new = fin(tnew)
        fin = interp1d(told, rrate[:ntf], kind='linear')
        rrate_new = fin(tnew)
        fin = interp2d(ebins_e, told, acc_rates_band_new[:ntf])
        acc_rates_band_interp = fin(ebins_e, tnew)
        fin = interp2d(ebins_e, told, esc_rates_band_new[:ntf])
        esc_rates_band_interp = fin(ebins_e, tnew)

        pindex1, ene1 = find_nearest(ebins_e, 0.1)
        pindex2, ene2 = find_nearest(ebins_e, 10)
        pindex3, ene3 = find_nearest(ebins_e, 200)
        pindex4, ene4 = find_nearest(ebins_e, 1000)
        # acc_rates2[:] = 0.0008
        # esc_rates2 = acc_rates2[2, :] * 3
        ntot = np.zeros([3, nframes_new])
        fesc = np.zeros(nbins)
        finj = np.zeros(nbins)

        fig = plt.figure(figsize=[7, 5])
        rect = [0.13, 0.13, 0.8, 0.8]
        ax = fig.add_axes(rect)

        for tframe in range(0, nframes_new):
            tframe_adjusted = tframe // stride
            # df = acc_rate_new[tframe] * div0(np.gradient(ebins_e * f),
            #                                  np.gradient(ebins_e))
            # df[pindex2-30:pindex2+30] *= 4.0*np.sin(np.linspace(0, math.pi, 60)) + 1.0
            df = div0(np.gradient(acc_rates_band_interp[tframe] * ebins_e * f),
                      np.gradient(ebins_e))
            # fesc_single = f * esc_rates_band_interp[tframe] * dtf
            fesc_single = f * esc_rates4_new[tframe] * dtf
            fesc += fesc_single
            ntot[0, tframe] = ntot[0, tframe-1] + np.sum(fesc * np.gradient(ebins_e))
            finj += finit * rrate_new[tframe] * dtf
            ntot[1, tframe] = ntot[1, tframe-1] + np.sum(finj * np.gradient(ebins_e))
            f += (-df + finit * rrate_new[tframe]) * dtf - fesc_single
            ntot[2, tframe] = ntot[2, tframe-1] + np.sum(f * np.gradient(ebins_e))
            # f += (-df - f * esc_rate_new[tframe]) * dtf
            # f += (finit * rrate_new[tframe]) * dtf
            # f += -(f * esc_rate_new[tframe]) * dtf

            ftot = f + fesc
            if tframe % stride == 0:
                color = plt.cm.jet((tframe//stride)/float(ntf), 1)
                ax.loglog(ebins_e, f, color=color)
                # ax.loglog(ebins_e, fesc, color=color)
                # ax.loglog(ebins_e, ftot, color=color)
        pindex = -3.5
        fpower = ebins_e**pindex * 2E16
        ax.loglog(ebins_e, fpower, color='k')
    fth = finit * ftot[np.argmax(ftot)] / finit[np.argmax(ftot)]
    fnth = ftot - fth
    ax.set_xlim([1E-1, 1E3])
    ax.set_ylim([1E2, 1E13])
    ax.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$', fontsize=16)
    ax.set_ylabel(r'$f(\varepsilon)$', fontsize=16)
    ax.tick_params(labelsize=12)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top=True)
    ax.tick_params(axis='x', which='major', direction='in', top=True)
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    power_index = "{%0.1f}" % pindex
    pname = r'$\propto \varepsilon^{' + power_index + '}$'
    ax.text(0.78, 0.80, pname, color='k', fontsize=16,
            bbox=dict(facecolor='none', alpha=1,
                      edgecolor='none', boxstyle="round,pad=0.1"),
            horizontalalignment='right', verticalalignment='center',
            transform=ax.transAxes)
    fdir = '../img/cori_3d/energy_equation/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + 'espect_' + str(threshold)[2:] + '.pdf'
    fig.savefig(fname)
    plt.show()


def plot_vkappa_3d(plot_config, show_plot=True):
    """Plot vdot_kappa in the 3D simulation
    """
    root_dir = "/net/scratch3/xiaocanli/reconnection/Cori_runs/"
    bg = plot_config["bg"]
    bg_str = str(int(bg * 10)).zfill(2)
    pic_run = "3D-Lx150-bg" + str(bg) + "-150ppc-2048KNL"
    pic_run_dir = root_dir + pic_run + '/'
    species = plot_config["species"]
    tframe = plot_config["tframe"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    stride_particle_dump = pic_info.stride_particle_dump
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
    nx, ny, nz = pic_info.nx, pic_info.ny, pic_info.nz
    nxr2, nyr2, nzr2 = nx // 2, ny // 2, nz // 2
    nxr4, nyr4, nzr4 = nx // 4, ny // 4, nz // 4
    smime = math.sqrt(pic_info.mime)
    x_di = pic_info.x_di
    y_di = pic_info.y_di
    z_di = pic_info.z_di
    xr2_di = x_di[1::2]
    yr2_di = y_di[1::2]
    zr2_di = z_di[1::2]
    fname = pic_run_dir + "data-smooth/vexb_kappa_" + str(tindex) + ".gda"
    vdot_kappa = np.fromfile(fname, dtype=np.float32)
    vdot_kappa = vdot_kappa.reshape((nzr2, nyr2, nxr2))

    ng = 5
    kernel = np.ones((ng,ng)) / float(ng*ng)
    # for yslice in range(0, nyr2, 4):
    for yslice in range(520, 521):
        fig = plt.figure(figsize=[14, 7])
        rect = [0.1, 0.1, 0.8, 0.8]
        ax = fig.add_axes(rect)
        vmin, vmax = -1.0, 1.0
        knorm = 100 if bg_str == '02' else 400
        fdata = vdot_kappa[:, yslice, :]*knorm
        fdata = signal.convolve2d(fdata, kernel, mode='same')
        p1 = ax.imshow(np.abs(fdata),
                       extent=[xmin, xmax, zmin, zmax],
                       vmin=vmin, vmax=vmax,
                       cmap=plt.cm.seismic, aspect='auto',
                       origin='lower', interpolation='bicubic')
        cs = ax.contour(xr2_di, zr2_di, np.abs(fdata), colors='k',
                        linewidths=0.5, levels=[0.1])
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

        # fname = fdir + 'nrho_bands_' + species + '_yslice_' + str(yslice) + ".jpg"
        # fig.savefig(fname, dpi=200)

        if show_plot:
            plt.show()
        else:
            plt.close()


def spatial_acc_rates(plot_config, show_plot=True):
    """
    """
    root_dir = "/net/scratch3/xiaocanli/reconnection/Cori_runs/"
    bg = plot_config["bg"]
    bg_str = str(int(bg * 10)).zfill(2)
    pic_run = "3D-Lx150-bg" + str(bg) + "-150ppc-2048KNL"
    pic_run_dir = root_dir + pic_run + '/'
    species = plot_config["species"]
    tframe = plot_config["tframe"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    stride_particle_dump = pic_info.stride_particle_dump
    if species == 'e':
        vth = pic_info.vthe
    else:
        vth = pic_info.vthi
    nbins = 1000
    ndata = nbins + 3  # including magnetic field
    tindex = pic_info.particle_interval * tframe
    gama = 1.0 / math.sqrt(1.0 - 3 * vth**2)
    eth = gama - 1.0
    nbins = 15
    ebins_edge = np.logspace(-2, 1, nbins+1)
    debins = np.diff(ebins_edge)
    ebins_edge /= eth
    ebins_mid = 0.5 * (ebins_edge[1:] + ebins_edge[:-1])
    tindex = tframe * pic_info.particle_interval
    fdir = pic_run_dir + 'spatial_acceleration_rates/'
    fname = fdir + 'spatial_acc_rates_' + species + '_' + str(tindex) + '.h5'
    fh = h5py.File(fname, 'r')
    dset = fh['particle_distribution']
    pdist = np.zeros(dset.shape, dtype=dset.dtype)
    dset.read_direct(pdist)
    grp = fh['acc_rates']
    dset = grp['Epara']
    fene_epara = np.zeros(dset.shape, dtype=dset.dtype)
    dset.read_direct(fene_epara)
    dset = grp['Eperp']
    fene_eperp = np.zeros(dset.shape, dtype=dset.dtype)
    dset.read_direct(fene_eperp)
    fh.close()

    # fbins = np.sum(np.sum(np.sum(pdist, axis=2), axis=1), axis=0)
    # fene_tot = np.sum(np.sum(np.sum(fene_epara+fene_eperp, axis=2), axis=1), axis=0)
    # plt.semilogx(ebins_mid, fene_tot[1:] / fbins[1:])
    # plt.show()

    fene_tot = np.sum(fene_eperp[:, :, :, 9:], axis=3)
    pdist_tot = np.sum(pdist[:, :, :, 9:], axis=3)
    fene_tot = div0(fene_tot, pdist_tot)
    ng = 3
    kernel = np.ones((ng,ng)) / float(ng*ng)
    vmin, vmax = -1.0, 1.0
    fdata = fene_tot[:, 173, :] * 100
    fdata = signal.convolve2d(fdata, kernel, mode='same')
    p1 = plt.imshow(fdata,
                    # extent=[xmin, xmax, zmin, zmax],
                    vmin=vmin, vmax=vmax,
                    cmap=plt.cm.seismic, aspect='auto',
                    origin='lower')
    plt.show()


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


def fit_particle_energization(plot_config):
    """Fit particle-based energization

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    tframe = plot_config["tframe"]
    pic_run_dir = plot_config["pic_run_dir"]
    bg = plot_config["bg"]
    bg_str = str(int(bg * 10)).zfill(2)
    pic_runs = ["3D-Lx150-bg" + str(bg) + "-150ppc-2048KNL"]
    pic_runs.append("2D-Lx150-bg" + str(bg) + "-150ppc-16KNL")

    ylims, data_indices = get_plot_setup(plot_config["plot_type"], species)

    tstarts = [6, 10, 20, 30]
    tends = [10, 20, 30, 40]
    nplots = len(tstarts)

    tstart, tend = 1, 40
    nframes = tend - tstart + 1
    slope = np.zeros((2, nframes))
    alpha0 = np.zeros((2, nframes))
    # tframe_chosen = [10, 18]
    tframe_chosen = np.linspace(10, 15, 2)

    # fig1 = plt.figure(figsize=[3.5, 5])
    # box1 = [0.16, 0.58, 0.8, 0.4]
    fig1 = plt.figure(figsize=[3.5, 2.5])
    box1 = [0.16, 0.16, 0.8, 0.8]
    ax = fig1.add_axes(box1)
    COLORS = palettable.tableau.Tableau_10.mpl_colors

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
            nbinx = int(fdata[1])
            nvar = int(fdata[2])
            ebins = fdata[3:nbins+3]
            fbins = np.sum(fdata[nbins+3:].reshape((nvar, nbinx, nbins)), axis=1)

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
            pindex1, ene1 = find_nearest(ebins, 50)
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

            color = COLORS[0] if '2D' in pic_run else COLORS[1]
            if tframe in tframe_chosen:
                lstyle = '-' if tframe == tframe_chosen[0] else '--'
                ax.plot(ebins, fdata/fnorm, linewidth=1, color=color,
                        linestyle=lstyle, marker='o', markersize=4)
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
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top=True)
    ax.tick_params(axis='x', which='major', direction='in', top=True)
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.text(0.90, 0.93, "2D", color=COLORS[0], fontsize=10,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='right', verticalalignment='center',
            transform=ax.transAxes)
    ax.text(0.98, 0.93, "3D", color=COLORS[1], fontsize=10,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='right', verticalalignment='center',
            transform=ax.transAxes)

    ax.text(0.57, 0.58, r'$t\Omega_{ci}=100$', color='k', fontsize=10,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes)
    ax.text(0.57, 0.36, r'$t\Omega_{ci}=150$', color='k', fontsize=10,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes)

    ax.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$', fontsize=10)
    ax.set_ylabel(r'$\alpha\omega_{pe}^{-1}/10^{-3}$', fontsize=10)
    ax.tick_params(labelsize=8)
    if species == 'e':
        ax.set_xlim([0, 250])
    else:
        ax.set_xlim([1E0, 500])
    ax.plot(ax.get_xlim(), [0, 0], linestyle='--',
            color='k', linewidth=0.5)
    ax.set_ylim([-0.2, 1.2])
    # ax.set_ylim([-0.4, 0.2])

    # box1[1] -= box1[3] + 0.1
    # ax = fig1.add_axes(box1)
    # slope /= 1E-2
    # ts = 5
    # ax.plot(tfields[ts:], slope[0, ts:], marker='v', markersize=3,
    #         linestyle='None', color=COLORS[0])
    # ax.plot(tfields[ts:], slope[1, ts:], marker='o', markersize=3,
    #         linestyle='None', color=COLORS[1])
    # tmin, tmax = tfields[ts - 1], tfields.max()
    # ax.set_xlim([tmin, tmax])
    # ax.set_ylim([-0.3, 0.7])
    # ax.plot([tmin, tmax], [0, 0], linewidth=0.5, linestyle='--', color='k')
    # ax.set_xlabel(r'$t\Omega_{ci}$', fontsize=10)
    # ax.set_ylabel(r'$s/10^{-2}$', fontsize=10)
    # ax.tick_params(bottom=True, top=False, left=True, right=False)
    # ax.tick_params(axis='x', which='minor', direction='in')
    # ax.tick_params(axis='x', which='major', direction='in')
    # ax.tick_params(axis='y', which='minor', direction='in')
    # ax.tick_params(axis='y', which='major', direction='in')
    # ax.tick_params(labelsize=8)

    # ax.text(0.1, 0.93, "2D", color=COLORS[0], fontsize=10,
    #         bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
    #         horizontalalignment='center', verticalalignment='center',
    #         transform=ax.transAxes)
    # ax.text(0.17, 0.93, "3D", color=COLORS[1], fontsize=10,
    #         bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
    #         horizontalalignment='center', verticalalignment='center',
    #         transform=ax.transAxes)

    # box2 = [0.56, 0.3, 0.35, 0.15]
    # ax1 = fig1.add_axes(box2)
    # ax1.plot(tfields[ts:], alpha0[0, ts:], marker='v', markersize=3,
    #          linestyle='-', color=COLORS[0], linewidth=1)
    # ax1.plot(tfields[ts:], alpha0[1, ts:], marker='o', markersize=3,
    #          linestyle='-', color=COLORS[1], linewidth=1)
    # tmin, tmax = tfields[ts - 1], tfields.max()
    # ax1.set_xlim([tmin, tmax])
    # ax1.plot([tmin, tmax], [0, 0], linewidth=0.5, linestyle='--', color='k')
    # ax1.tick_params(bottom=True, top=False, left=True, right=False)
    # ax1.tick_params(axis='x', which='minor', direction='in')
    # ax1.tick_params(axis='x', which='major', direction='in')
    # ax1.tick_params(axis='y', which='minor', direction='in')
    # ax1.tick_params(axis='y', which='major', direction='in')
    # # ax1.tick_params(axis='x', labelbottom='off')
    # ax1.set_ylabel(r'$\alpha_0\omega_{pe}^{-1}/10^{-3}$', fontsize=8)
    # ax1.tick_params(labelsize=8)

    fdir = '../img/cori_3d/particle_energization/'
    mkdir_p(fdir)
    fname = (fdir + 'particle_' + plot_config["plot_type"] + '_' +
             'bg' + bg_str + '_' + species + '.pdf')
    fig1.savefig(fname)

    plt.show()


def acc_rate_pub(plot_config):
    """Plot acceleration rate for publication

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    tframe = plot_config["tframe"]
    pic_run_dir = plot_config["pic_run_dir"]
    bg = plot_config["bg"]
    bg_str = str(int(bg * 10)).zfill(2)
    pic_runs = ["3D-Lx150-bg" + str(bg) + "-150ppc-2048KNL"]
    # pic_runs.append("2D-Lx150-bg" + str(bg) + "-150ppc-16KNL")
    pic_runs.append("2D-Lx150-bg" + str(bg) + "-150ppc-large-nppc")

    tstarts = [6, 10, 20, 30]
    tends = [10, 20, 30, 40]
    nplots = len(tstarts)

    tstart, tend = 1, 40
    nframes = tend - tstart + 1
    slope = np.zeros((2, nframes))
    alpha0 = np.zeros((2, nframes))
    tframe_chosen = np.linspace(10, 15, 2)

    fig1 = plt.figure(figsize=[3.5, 2.5])
    box1 = [0.16, 0.16, 0.8, 0.8]
    ax = fig1.add_axes(box1)
    # box2 = [0.53, 0.61, 0.4, 0.3]
    fig2 = plt.figure(figsize=[3.5, 2.5])
    ax1 = fig2.add_axes(box1)
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    ax1.set_prop_cycle('color', COLORS)

    for irun, pic_run in enumerate(pic_runs):
        fpath = "../data/particle_interp/" + pic_run + "/"
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        dtp = math.ceil(pic_info.dt_particles / 0.1) * 0.1
        tfields = np.arange(tstart, tend+1) * dtp
        # normalized with thermal energy
        if species == 'e':
            vth = pic_info.vthe
        else:
            vth = pic_info.vthi
        gama = 1.0 / math.sqrt(1.0 - 3 * vth**2)
        eth = gama - 1.0
        if species == 'i':
            eth *= pic_info.mime
        # time loop
        for tframe in tframe_chosen:
            tstep = tframe * pic_info.particle_interval
            tframe_fluid = tstep // pic_info.fields_interval
            fname = fpath + "particle_energization_" + species + "_" + str(int(tstep)) + ".gda"
            fdata = np.fromfile(fname, dtype=np.float32)
            nbins = int(fdata[0])
            nbinx = int(fdata[1])
            nvar = int(fdata[2])
            ebins = fdata[3:nbins+3]
            fbins = np.sum(fdata[nbins+3:].reshape((nvar, nbinx, nbins)), axis=1)

            if species == 'i':
                ebins *= pic_info.mime  # ebins are actually gamma
            if species == 'e':
                fbins[1:, :] = div0(fbins[1:, :], fbins[0, :])
            else:
                fbins[1:, :] = div0(fbins[1:, :], fbins[0, :] * pic_info.mime)

            ebins /= eth
            pindex1, ene1 = find_nearest(ebins, 50)
            if '2D' in pic_run:
                pindex2, ene2 = find_nearest(ebins, 150)
            else:
                pindex2, ene2 = find_nearest(ebins, 200)

            ylims, data_indices = get_plot_setup("total", species)
            fdata = np.zeros(nbins)
            for idata in data_indices:
                fdata += fbins[idata, :]

            fnorm = 1E-3
            color = COLORS[1] if '2D' in pic_run else COLORS[0]
            lstyle = '-' if tframe == tframe_chosen[0] else '--'
            ax.plot(ebins, fdata/fnorm, linewidth=1, color=color,
                    linestyle=lstyle, marker='o', markersize=4)
            if irun == 0 and tframe == tframe_chosen[0]:
                ax1.plot(ebins, fdata/fnorm, linewidth=1,
                         linestyle=lstyle, marker='o', markersize=4)
                ylims, data_indices = get_plot_setup("curvature", species)
                fdata = np.zeros(nbins)
                for idata in data_indices:
                    fdata += fbins[idata, :]
                ax1.plot(ebins, fdata/fnorm, linewidth=1,
                         linestyle=lstyle, marker='o', markersize=4)
                ylims, data_indices = get_plot_setup("parallel", species)
                fdata = np.zeros(nbins)
                for idata in data_indices:
                    fdata += fbins[idata, :]
                ax1.plot(ebins, fdata/fnorm, linewidth=1,
                         linestyle=lstyle, marker='o', markersize=4)
                ylims, data_indices = get_plot_setup("gradient", species)
                fdata = np.zeros(nbins)
                for idata in data_indices:
                    fdata += fbins[idata, :]
                ax1.plot(ebins, fdata/fnorm, linewidth=1,
                         linestyle=lstyle, marker='o', markersize=4)

    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top=True)
    ax.tick_params(axis='x', which='major', direction='in', top=True)
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.text(0.20, 0.92, "2D", color=COLORS[1], fontsize=10,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes)
    ax.text(0.36, 0.92, "3D", color=COLORS[0], fontsize=10,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes)
    ax.text(0.28, 0.92, "vs.", color='k', fontsize=10,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes)

    ax.text(0.40, 0.46, r'$t\Omega_{ci}=100$', color='k', fontsize=8,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes)
    ax.text(0.40, 0.35, r'$t\Omega_{ci}=150$', color='k', fontsize=8,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes)

    ax.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$', fontsize=10)
    ax.set_ylabel(r'$10^3\alpha\omega_{pe}^{-1}$', fontsize=10)
    ax.tick_params(labelsize=8)
    if species == 'e':
        ax.set_xlim([0, 250])
    else:
        ax.set_xlim([1E0, 500])
    ax.plot(ax.get_xlim(), [0, 0], linestyle='--',
            color='k', linewidth=0.5)
    ax.set_ylim([-0.2, 1.0])
    # ax.set_ylim([-0.4, 0.2])

    ax1.tick_params(bottom=True, top=True, left=True, right=True)
    ax1.tick_params(axis='x', which='minor', direction='in', top=True)
    ax1.tick_params(axis='x', which='major', direction='in', top=True)
    ax1.tick_params(axis='y', which='minor', direction='in')
    ax1.tick_params(axis='y', which='major', direction='in')
    if species == 'e':
        ax1.set_xlim([0, 250])
    else:
        ax1.set_xlim([1E0, 500])
    ax1.plot(ax1.get_xlim(), [0, 0], linestyle='--', color='k', linewidth=0.5)
    ax1.set_ylim([-0.4, 1.2])
    ax1.tick_params(labelsize=8)
    ax1.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$', fontsize=10)
    ax1.set_ylabel(r'$10^3\alpha\omega_{pe}^{-1}$', fontsize=10)
    ax1.text(0.95, 0.48, "Total", color=COLORS[0], fontsize=10,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='right', verticalalignment='center',
             transform=ax1.transAxes)
    ax1.text(0.95, 0.65, "Curvature Drift", color=COLORS[1], fontsize=10,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='right', verticalalignment='center',
             transform=ax1.transAxes)
    ax1.text(0.95, 0.25, r"$\boldsymbol{E}_\parallel$", color=COLORS[2], fontsize=10,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='right', verticalalignment='center',
             transform=ax1.transAxes)
    ax1.text(0.95, 0.14, "Gradient Drift", color=COLORS[3], fontsize=10,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='right', verticalalignment='center',
             transform=ax1.transAxes)
    ax1.text(0.97, 0.93, r'3D: $t\Omega_{ci}=150$', color='k', fontsize=10,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='right', verticalalignment='top',
             transform=ax1.transAxes)

    fdir = '../img/cori_3d/particle_energization_pub/'
    mkdir_p(fdir)
    fname = fdir + 'particle_total_' + 'bg' + bg_str + '_' + species + '.pdf'
    fig1.savefig(fname)
    fname = fdir + 'arate_3d_' + 'bg' + bg_str + '_' + species + '.pdf'
    fig2.savefig(fname)

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
                        help='Guide field strength')
    parser.add_argument('--threshold', action="store", default='0.001', type=float,
                        help='Guide field strength')
    parser.add_argument('--show_plot', action="store_true", default=False,
                        help='whether to show plot')
    parser.add_argument('--acc_rate', action="store_true", default=False,
                        help='whether to plot acceleration rate')
    parser.add_argument('--acc_rate_std', action="store_true", default=False,
                        help='whether to plot the standard deviation of the acceleration rates')
    parser.add_argument('--acc_rate_dist', action="store_true", default=False,
                        help='whether to plot acceleration rate distribution')
    parser.add_argument('--plot_type', action="store", default='total', type=str,
                        help='Particle plot type')
    parser.add_argument('--multi_types', action="store_true", default=False,
                        help='Multiple particle plot types')
    parser.add_argument('--anisotropy', action="store_true", default=False,
                        help='whether to plot pressure anisotropy')
    parser.add_argument('--anisotropy_pub', action="store_true", default=False,
                        help='whether to plot pressure anisotropy for publication')
    parser.add_argument('--calc_vexb_kappa_2d', action="store_true", default=False,
                        help='whether to calculate vexb dot magnetic curvature')
    parser.add_argument('--vkappa_dist_2d', action="store_true", default=False,
                        help='whether to calculate the distribution of magnetic curvature')
    parser.add_argument('--reorg_vkappa_dist_3d', action="store_true", default=False,
                        help='whether to reorganize vkappa_dist for 3D')
    parser.add_argument('--plot_vkappa_dist', action="store_true", default=False,
                        help='whether to plot vkappa_dist')
    parser.add_argument('--vkappa_dist_3d', action="store_true", default=False,
                        help='whether to plot the distribution of magnetic curvature in 3D')
    parser.add_argument('--avg_acc_rate', action="store_true", default=False,
                        help='whether to get average acceleration rate')
    parser.add_argument('--plot_vkappa_3d', action="store_true", default=False,
                        help='whether to plot vdot_kappa in the 3D simulations')
    parser.add_argument('--plot_acc_esc', action="store_true", default=False,
                        help='whether to plot acceleration and escape rates')
    parser.add_argument('--comp_vsingle_vexb', action="store_true", default=False,
                        help='whether to compare vsingle and vexb for calculating vdot_kappa')
    parser.add_argument('--fit_particle_ene', action="store_true", default=False,
                        help='whether to fit particle energization')
    parser.add_argument('--test', action="store_true", default=False,
                        help='whether to test')
    parser.add_argument('--spect_index_pub', action="store_true", default=False,
                        help='whether to plot spectral index for publication')
    parser.add_argument('--comp_esc_rate', action="store_true", default=False,
                        help='whether to escape rate for different threshold')
    parser.add_argument('--energy_equation', action="store_true", default=False,
                        help='whether to solve the energy continuity equation')
    parser.add_argument('--acc_rate_pub', action="store_true", default=False,
                        help='whether to acceleration rate for publication')
    return parser.parse_args()


def analysis_single_frames(plot_config, args):
    """Analysis for multiple time frames
    """
    tframe = args.tframe
    if args.acc_rate:
        acceleration_rate(plot_config, show_plot=True)
    if args.acc_rate_std:
        acceleration_rate_std(plot_config, show_plot=True)
    elif args.acc_rate_dist:
        acceleration_rate_distribution(plot_config, show_plot=True)
    elif args.anisotropy:
        plot_anisotropy(plot_config, show_plot=True)
    elif args.anisotropy_pub:
        plot_anisotropy_pub(plot_config, show_plot=True)
    elif args.acc_rate_pub:
        acc_rate_pub(plot_config)
    elif args.calc_vexb_kappa_2d:
        calc_vexb_kappa_2d(plot_config)
    elif args.vkappa_dist_2d:
        vkappa_dist_2d(plot_config)
    elif args.reorg_vkappa_dist_3d:
        reorganize_vkappa_dist_3d(plot_config)
    elif args.plot_vkappa_dist:
        plot_vkappa_dist(plot_config)
    elif args.vkappa_dist_3d:
        vkappa_dist_3d(plot_config)
    elif args.avg_acc_rate:
        avg_acceleration_rate(plot_config)
    elif args.plot_vkappa_3d:
        plot_vkappa_3d(plot_config)
    elif args.plot_acc_esc:
        plot_acc_esc(plot_config)
    elif args.comp_vsingle_vexb:
        comp_vsingle_vexb(plot_config)
    elif args.fit_particle_ene:
        fit_particle_energization(plot_config)
    elif args.test:
        spatial_acc_rates(plot_config)
    elif args.spect_index_pub:
        spectral_index_pub(plot_config)
    elif args.comp_esc_rate:
        compare_escape_rate(plot_config)
    elif args.energy_equation:
        solve_energy_equation(plot_config)


def process_input(plot_config, args, tframe):
    """process one time frame"""
    plot_config["tframe"] = tframe
    if args.vkappa_dist_2d:
        vkappa_dist_2d(plot_config)
    elif args.reorg_vkappa_dist_3d:
        reorganize_vkappa_dist_3d(plot_config)
    elif args.avg_acc_rate:
        avg_acceleration_rate(plot_config)


def analysis_multi_frames(plot_config, args):
    """Analysis for multiple time frames
    """
    tframes = range(plot_config["tstart"], plot_config["tend"] + 1)
    if args.time_loop:
        for tframe in tframes:
            print("Time frame: %d" % tframe)
            plot_config["tframe"] = tframe
            if args.acc_rate:
                acceleration_rate(plot_config, show_plot=False)
            elif args.calc_vexb_kappa_2d:
                calc_vexb_kappa_2d(plot_config)
    else:
        # ncores = multiprocessing.cpu_count()
        ncores = 4
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
    plot_config["bg"] = args.bg
    plot_config["threshold"] = args.threshold
    if args.multi_frames:
        analysis_multi_frames(plot_config, args)
    else:
        analysis_single_frames(plot_config, args)


if __name__ == "__main__":
    main()
