#!/usr/bin/env python3
"""
Power spectrum of fields for the Cori runs
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
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
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


def read_power_spectrum(fname):
    """Read power spectrum data
    """
    fdata = np.fromfile(fname, dtype=np.float32)
    sz, = fdata.shape
    kbins, fk = fdata[:sz//2], fdata[sz//2:]
    return kbins, fk


def var_labels():
    """Labels for different variables
    """
    labels = {"ne": "n_e",
              "ni": "n_i",
              "bx": "B_x",
              "by": "B_y",
              "bz": "B_z",
              "ex": "E_x",
              "ey": "E_y",
              "ez": "E_z",
              "vex": "v_{ex}",
              "vey": "v_{ey}",
              "vez": "v_{ez}",
              "vix": "v_{ix}",
              "viy": "v_{iy}",
              "viz": "v_{iz}"}
    return labels


def plot_power_spectrum(plot_config, show_plot=True):
    """Plot power spectrum for one variable at one time frame
    Args:
        plot_config: plot configuration
    """
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    tframe = plot_config["tframe"]
    var_name = plot_config["var_name"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    tindex = tframe * pic_info.fields_interval
    fdir = '../data/power_spectrum/' + pic_run + '/power_spectrum_' + var_name + '/'
    labels = var_labels()
    var_label = labels[var_name]
    fname = fdir + var_name + str(tindex) + '.kx'
    kx, fkx = read_power_spectrum(fname)
    fname = fdir + var_name + str(tindex) + '.ky'
    ky, fky = read_power_spectrum(fname)
    fname = fdir + var_name + str(tindex) + '.kz'
    kz, fkz = read_power_spectrum(fname)
    fname = fdir + var_name + str(tindex) + '.para'
    kpara, fkpara = read_power_spectrum(fname)
    fname = fdir + var_name + str(tindex) + '.perp'
    kperp, fkperp = read_power_spectrum(fname)

    smime = math.sqrt(pic_info.mime)
    dx_de = pic_info.dx_di * smime
    dy_de = pic_info.dy_di * smime
    dz_de = pic_info.dz_di * smime
    k1, k2 = 0.02, 0.5
    nkbins = 100

    idx = (np.abs(kx-k1)).argmin()
    idy = (np.abs(ky-k1)).argmin()
    idz = (np.abs(kz-k1)).argmin()
    fnorm = max(fkx[idx], fky[idy], fkz[idz]) * 2

    kpower = np.logspace(math.log10(k1), math.log10(k2), 100)
    fpower1 = kpower**(-5/3)
    fpower1 *= fnorm / fpower1[0]
    fpower2 = kpower**-1.5
    fpower2 *= fpower1[0] / fpower2[0]
    fpower3 = kpower**-2
    fpower3 *= fpower1[0] / fpower3[0]

    fig = plt.figure(figsize=[7, 5])
    rect = [0.15, 0.16, 0.8, 0.8]
    ax = fig.add_axes(rect)
    ax.set_prop_cycle('color', COLORS)
    ax.loglog(kx, fkx, linewidth=2, label=r'$k_x$')
    ax.loglog(ky, fky, linewidth=2, label=r'$k_y$')
    ax.loglog(kz, fkz, linewidth=2, label=r'$k_z$')
    ax.loglog(kpower, fpower1, linewidth=1, color='k',
              linestyle='--', label=r'$\sim k^{-5/3}$')
    ax.loglog(kpower, fpower2, linewidth=1, color='k',
              linestyle='-.', label=r'$\sim k^{-3/2}$')
    ax.loglog(kpower, fpower3, linewidth=1, color='k',
              linestyle=':', label=r'$\sim k^{-2}$')
    ax.legend(loc=1, prop={'size': 16}, ncol=1,
              shadow=False, fancybox=False, frameon=False)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top=True)
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in', left=True)
    ax.tick_params(axis='y', which='major', direction='in')
    text1 = r'$' + var_label + '$'
    ax.text(0.02, 0.05, text1, color='k', fontsize=20,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)
    ax.set_xlim([1E-2, 1E1])
    # ax.set_ylim([1E-7, 2E-1])
    ax.set_xlabel(r'$kd_e$', fontsize=20)
    ax.set_ylabel(r'$E_{' + var_label + '}(k)$', fontsize=20)
    ax.tick_params(labelsize=16)

    fdir = '../img/power_spectrum/' + pic_run + '/' + var_name + '/'
    mkdir_p(fdir)
    fname = fdir + var_name + '_xyz_' + str(tframe) + '.pdf'
    fig.savefig(fname)

    fig = plt.figure(figsize=[7, 5])
    rect = [0.15, 0.16, 0.8, 0.8]
    ax = fig.add_axes(rect)
    ax.set_prop_cycle('color', COLORS)
    ax.loglog(kpara, fkpara, linewidth=2, label=r'$k_\parallel$')
    ax.loglog(kperp, fkperp, linewidth=2, label=r'$k_\perp$')
    ax.loglog(kpower, fpower1, linewidth=1, color='k',
              linestyle='--', label=r'$\sim k^{-5/3}$')
    ax.loglog(kpower, fpower2, linewidth=1, color='k',
              linestyle='-.', label=r'$\sim k^{-3/2}$')
    ax.loglog(kpower, fpower3, linewidth=1, color='k',
              linestyle=':', label=r'$\sim k^{-2}$')
    ax.legend(loc=1, prop={'size': 16}, ncol=1,
              shadow=False, fancybox=False, frameon=False)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top=True)
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in', left=True)
    ax.tick_params(axis='y', which='major', direction='in')
    text1 = r'$' + var_label + '$'
    ax.text(0.02, 0.05, text1, color='k', fontsize=20,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)
    ax.set_xlim([1E-2, 1E1])
    # ax.set_ylim([1E-7, 2E-1])
    ax.set_xlabel(r'$kd_e$', fontsize=20)
    ax.set_ylabel(r'$E_{' + var_label + '}(k)$', fontsize=20)
    ax.tick_params(labelsize=16)
    fname = fdir + var_name + '_para_perp_' + str(tframe) + '.pdf'
    fig.savefig(fname)

    if show_plot:
        plt.show()
    else:
        plt.close('all')


def plot_power_spectrum_pub(plot_config, show_plot=True):
    """Plot power spectrum for one variable at one time frame for publication
    Args:
        plot_config: plot configuration
    """
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    tframe = plot_config["tframe"]
    var_name = plot_config["var_name"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    tindex = tframe * pic_info.fields_interval
    fdir = '../data/power_spectrum/' + pic_run + '/power_spectrum_' + var_name + '/'
    labels = var_labels()
    var_label = labels[var_name]
    fname = fdir + var_name + str(tindex) + '.kx'
    kx, fkx = read_power_spectrum(fname)
    fname = fdir + var_name + str(tindex) + '.ky'
    ky, fky = read_power_spectrum(fname)
    fname = fdir + var_name + str(tindex) + '.kz'
    kz, fkz = read_power_spectrum(fname)
    fname = fdir + var_name + str(tindex) + '.para'
    kpara, fkpara = read_power_spectrum(fname)
    fname = fdir + var_name + str(tindex) + '.perp'
    kperp, fkperp = read_power_spectrum(fname)

    k1, k2 = 0.02, 0.5
    nkbins = 100

    idx = (np.abs(kx-k1)).argmin()
    idy = (np.abs(ky-k1)).argmin()
    idz = (np.abs(kz-k1)).argmin()
    fnorm = max(fkx[idx], fky[idy], fkz[idz]) * 2

    kpower = np.logspace(math.log10(k1), math.log10(k2), 100)
    fpower1 = kpower**(-5/3)
    fpower1 *= fnorm / fpower1[0]
    fpower2 = kpower**-1.5
    fpower2 *= fpower1[0] / fpower2[0]
    fpower3 = kpower**-2
    fpower3 *= fpower1[0] / fpower3[0]

    fig = plt.figure(figsize=[3.5, 2.5])
    rect = [0.17, 0.16, 0.78, 0.8]
    ax = fig.add_axes(rect)
    ax.set_prop_cycle('color', COLORS)
    ax.loglog(kpara, fkpara, linewidth=1, label=r'$k_\parallel$')
    ax.loglog(kperp, fkperp, linewidth=1, label=r'$k_\perp$')
    ax.loglog(kpower, fpower3, linewidth=1, color='k',
              linestyle=':', label=r'$\sim k^{-2}$')
    ax.legend(loc=1, prop={'size': 10}, ncol=1,
              shadow=False, fancybox=False, frameon=False)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top=True)
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in', left=False)
    ax.tick_params(axis='y', which='major', direction='in')
    twci = math.ceil((tframe * pic_info.dt_fields) / 0.1) * 0.1
    text1 = r'$t\Omega_{ci}=' + ("{%0.0f}" % twci) + '$'
    ax.text(0.02, 0.05, text1, color='k', fontsize=10,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)
    ax.set_xlim([1E-2, 1E1])
    ax.set_ylim([1E-7, 2E-1])
    ax.set_yticks((np.logspace(-7, -1, 4)))
    ax.set_xlabel(r'$kd_e$', fontsize=10)
    ax.set_ylabel(r'$E_{' + var_label + '}(k)$', fontsize=10)
    ax.tick_params(labelsize=8)
    fdir = '../img/power_spectrum_pub/' + pic_run + '/' + var_name + '/'
    mkdir_p(fdir)
    fname = fdir + var_name + '_para_perp_' + str(tframe) + '.pdf'
    fig.savefig(fname)

    if show_plot:
        plt.show()
    else:
        plt.close('all')


def magnetic_power_spectrum(plot_config, show_plot=True):
    """Plot power spectrum of magnetic field
    Args:
        plot_config: plot configuration
    """
    tframe = plot_config["tframe"]
    bg = plot_config["bg"]
    pic_runs = ["2D-Lx150-bg" + str(bg) + "-150ppc-16KNL"]
    pic_runs.append("3D-Lx150-bg" + str(bg) + "-150ppc-2048KNL")
    pic_run = pic_runs[1]
    root_dir = "/net/scratch3/xiaocanli/reconnection/Cori_runs/"
    pic_run_dir = root_dir + pic_run + "/"
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    k1, k2 = 0.03, 1.0
    nkbins = 100
    pindex = -2.7
    kpower = np.logspace(math.log10(k1), math.log10(k2), 100)
    fpower3 = kpower**pindex / 1E4

    fig = plt.figure(figsize=[3.5, 2.5])
    rect = [0.17, 0.16, 0.78, 0.8]
    ax = fig.add_axes(rect)
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    ax.set_prop_cycle('color', COLORS)
    tframes = range(10, 36, 5)

    for tframe in tframes:
        tindex = tframe * pic_info.fields_interval
        for ivar, var in enumerate(["bx", "by", "bz"]):
            fdir = ('../data/power_spectrum/' + pic_run +
                    '/power_spectrum_' + var + '/')
            fname = fdir + var + str(tindex) + '.para'
            kpara, fdata = read_power_spectrum(fname)
            if ivar > 0:
                fkpara += fdata
            else:
                fkpara = fdata
            fname = fdir + var + str(tindex) + '.perp'
            kperp, fdata = read_power_spectrum(fname)
            if ivar > 0:
                fkperp += fdata
            else:
                fkperp = fdata
        ax.loglog(kperp, fkperp, linewidth=1, label=r'$k_\perp$')

    label1 = r'$\propto k_\perp^{' + str(pindex) + '}$'
    ax.loglog(kpower, fpower3, linewidth=0.5, color='k',
              linestyle='--', label=label1)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top=False)
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in', left=False)
    ax.tick_params(axis='y', which='major', direction='in')
    twci = math.ceil((tframe * pic_info.dt_fields) / 0.1) * 0.1
    # text1 = r'$t\Omega_{ci}=' + ("{%0.0f}" % twci) + '$'
    ax.text(0.7, 0.47, label1, color='k', fontsize=10,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)
    ax.set_xlim([1E-2, 5E0])
    # ax.set_ylim([1E-7, 2E-1])
    ax.set_yticks((np.logspace(-7, -1, 4)))
    ax.set_xlabel(r'$k_\perp d_e$', fontsize=10)
    ax.set_ylabel(r'$E_B(k_\perp)$', fontsize=10)
    ax.tick_params(labelsize=8)

    # Embedded plot for energy evolution
    rect1 = [0.29, 0.29, 0.30, 0.28]
    ax1 = fig.add_axes(rect1)
    ax1.tick_params(bottom=True, top=True, left=True, right=True)
    ax1.tick_params(axis='x', which='minor', direction='in', top=False)
    ax1.tick_params(axis='x', which='major', direction='in')
    ax1.tick_params(axis='y', which='minor', direction='in')
    ax1.tick_params(axis='y', which='major', direction='in')

    for irun, pic_run in enumerate(pic_runs):
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        enorm = pic_info.ene_magnetic[0]
        ene_bx = pic_info.ene_bx
        ene_by = pic_info.ene_by
        ene_bz = pic_info.ene_bz
        ene_magnetic = pic_info.ene_magnetic
        ene_electric = pic_info.ene_electric
        kene_e = pic_info.kene_e
        kene_i = pic_info.kene_i
        ene_bx /= enorm
        ene_by /= enorm
        ene_bz /= enorm
        ene_magnetic /= enorm
        kene_e /= enorm
        kene_i /= enorm
        tenergy = pic_info.tenergy

        lstyle = '-' if '3D' in pic_run else '--'
        ax1.plot(tenergy, ene_magnetic, linewidth=1,
                 linestyle=lstyle, color='k')
    ax1.set_ylim([0.66, 1.02])
    # for iframe, tframe in enumerate(tframes):
    #     twci = tframe * pic_info.dt_fields
    #     ax1.plot([twci, twci], ax1.get_ylim(), linewidth=0.5,
    #              linestyle=':', color=COLORS[iframe])
    tframes = np.asarray(tframes) * pic_info.dt_fields
    nframe, = tframes.shape
    ax1.scatter(tframes, [0.7]*nframe, c=COLORS[:nframe],
                marker='x', s=10, linewidth=0.5)
    ax1.text(0.6, 0.28, "2D", color='k', fontsize=6,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='center', verticalalignment='center',
             transform=ax1.transAxes)
    ax1.text(0.6, 0.52, "3D", color='k', fontsize=6,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='center', verticalalignment='center',
             transform=ax1.transAxes)
    ax1.tick_params(labelsize=6)
    ax1.set_xlabel(r'$t\Omega_{ci}$', fontsize=6)
    ax1.set_ylabel(r'$\varepsilon_B/\varepsilon_{B0}$', fontsize=6)
    ax1.set_xlim([0, 400])

    fdir = '../img/cori_3d/power_spectrum_pub/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + 'mag_perp.pdf'
    fig.savefig(fname)

    if show_plot:
        plt.show()
    else:
        plt.close('all')


def pspect_mag_vel(plot_config, show_plot=True):
    """Plot power spectrum for magnetic field and velocity field
    Args:
        plot_config: plot configuration
    """
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    tframe = plot_config["tframe"]
    component = plot_config["component"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    tindex = tframe * pic_info.fields_interval
    var_name = 'vi' + component
    fdir = '../data/power_spectrum/' + pic_run + '/power_spectrum_' + var_name + '/'
    fname = fdir + var_name + str(tindex) + '.para'
    kpara, fkpara_v = read_power_spectrum(fname)
    fname = fdir + var_name + str(tindex) + '.perp'
    kperp, fkperp_v = read_power_spectrum(fname)
    # fkpara_v *= pic_info.mime * 0.5
    # fkperp_v *= pic_info.mime * 0.5

    var_name = 'b' + component
    fdir = '../data/power_spectrum/' + pic_run + '/power_spectrum_' + var_name + '/'
    fname = fdir + var_name + str(tindex) + '.para'
    kpara, fkpara_b = read_power_spectrum(fname)
    fname = fdir + var_name + str(tindex) + '.perp'
    kperp, fkperp_b = read_power_spectrum(fname)

    k1, k2 = 0.05, 0.5
    nkbins = 100

    id_para = (np.abs(kpara-k1)).argmin()
    id_perp = (np.abs(kperp-k1)).argmin()
    fnorm1 = max(fkpara_b[id_para], fkperp_b[id_perp]) * 2
    fnorm2 = max(fkpara_v[id_para], fkperp_v[id_perp]) * 4

    kpower = np.logspace(math.log10(k1), math.log10(k2), 100)
    fpower1 = kpower**(-5/3)
    fpower1 /= fpower1[0]
    fpower2 = kpower**-1.5
    fpower2 *= fpower1[0] / fpower2[0]
    fpower3 = kpower**-2
    fpower3 *= fpower1[0] / fpower3[0]

    fig = plt.figure(figsize=[7, 5])
    rect = [0.15, 0.16, 0.8, 0.8]
    ax = fig.add_axes(rect)
    ax.set_prop_cycle('color', COLORS)
    p1, = ax.loglog(kpara, fkpara_b, linewidth=2, label=r'$E_B(k_\parallel)$')
    p2, = ax.loglog(kperp, fkperp_b, linewidth=2, label=r'$E_B(k_\perp)$')
    p3, = ax.loglog(kpara, fkpara_v, linewidth=2, linestyle='--',
                    color=p1.get_color(), label=r'$E_V(k_\parallel)$')
    p4, = ax.loglog(kperp, fkperp_v, linewidth=2, linestyle='--',
                    color=p2.get_color(), label=r'$E_V(k_\perp)$')
    ax.loglog(kpower, fpower1 * fnorm1, linewidth=1, color='k',
              linestyle='--', label=r'$\sim k^{-5/3}$')
    ax.loglog(kpower, fpower2 * fnorm1, linewidth=1, color='k',
              linestyle='-.', label=r'$\sim k^{-3/2}$')
    ax.loglog(kpower, fpower3 * fnorm1, linewidth=1, color='k',
              linestyle=':', label=r'$\sim k^{-2}$')
    ax.loglog(kpower, fpower1 * fnorm2, linewidth=1, color='k', linestyle='--')
    ax.loglog(kpower, fpower2 * fnorm2, linewidth=1, color='k', linestyle='-.')
    ax.loglog(kpower, fpower3 * fnorm2, linewidth=1, color='k', linestyle=':')
    ax.legend(loc=1, prop={'size': 16}, ncol=1,
              shadow=False, fancybox=False, frameon=False)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top='on')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in', left='on')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlim([1E-2, 1E1])
    ax.set_ylim([5E-9, 2E-1])
    ax.set_xlabel(r'$kd_e$', fontsize=20)
    ax.set_ylabel(r'$E(k)$', fontsize=20)
    ax.tick_params(labelsize=16)
    fdir = '../img/power_spectrum/' + pic_run + '/mag_vel/'
    mkdir_p(fdir)
    fname = fdir + 'bvel_' + component + '_para_perp_' + str(tframe) + '.pdf'
    fig.savefig(fname)

    if show_plot:
        plt.show()
    else:
        plt.close('all')


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
    parser.add_argument('--var_name', action="store", default="ne",
                        help='variable name')
    parser.add_argument('--single_var', action="store_true", default=False,
                        help='whether to plot power spectrum for a single variable')
    parser.add_argument('--single_var_pub', action="store_true", default=False,
                        help='whether to plot power spectrum for a single ' +
                             'variable for publication')
    parser.add_argument('--mag_vel', action="store_true", default=False,
                        help='whether to plot power spectrum for for B and V')
    parser.add_argument('--mag_power', action="store_true", default=False,
                        help='whether to plot power spectrum of magnetic field')
    parser.add_argument('--component', action="store", default="x",
                        help='which component (x/y/z)')
    return parser.parse_args()


def analysis_single_frames(plot_config, args):
    """Analysis for multiple time frames
    """
    tframe = args.tframe
    if args.single_var:
        plot_power_spectrum(plot_config)
    if args.single_var_pub:
        plot_power_spectrum_pub(plot_config)
    elif args.mag_vel:
        pspect_mag_vel(plot_config)
    elif args.mag_power:
        magnetic_power_spectrum(plot_config)


def process_input(plot_config, args, tframe):
    """process one time frame"""
    plot_config["tframe"] = tframe
    if args.single_var:
        plot_power_spectrum(plot_config, show_plot=False)


def analysis_multi_frames(plot_config, args):
    """Analysis for multiple time frames
    """
    tframes = range(plot_config["tstart"], plot_config["tend"] + 1)
    if args.time_loop:
        for tframe in tframes:
            print("Time frame: %d" % tframe)
            plot_config["tframe"] = tframe
            if args.single_var:
                plot_power_spectrum(plot_config, show_plot=False)
            elif args.single_var_pub:
                plot_power_spectrum_pub(plot_config, show_plot=False)
            elif args.mag_vel:
                pspect_mag_vel(plot_config, show_plot=False)
    else:
        ncores = multiprocessing.cpu_count()
        ncores = 32
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
    plot_config["var_name"] = args.var_name
    plot_config["component"] = args.component
    plot_config["bg"] = args.bg
    if args.multi_frames:
        analysis_multi_frames(plot_config, args)
    else:
        analysis_single_frames(plot_config, args)


if __name__ == "__main__":
    main()
