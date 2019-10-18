#!/usr/bin/env python3
"""
Particle energy spectrum for runs to determine power-law indices
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
from scipy.optimize import curve_fit
from scipy import signal

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
    ntf = pic_info.ntf
    emin, emax = 1E-4, 1E6
    nbins = 1000
    dloge = (math.log10(emax) - math.log10(emin)) / (nbins - 1)
    emin0 = 10**(math.log10(emin) - dloge)
    ebins = np.logspace(math.log10(emin0), math.log10(emax), nbins+1)
    ebins_mid = (ebins[:-1] + ebins[1:]) * 0.5
    debins = np.diff(ebins)
    nptot = pic_info.nx * pic_info.ny * pic_info.nz * pic_info.nppc

    tframes = range(ntf)
    nframes = len(tframes)
    flogs = np.zeros((nframes, nbins))
    if species in ["e", "electron"]:
        sname = "electron"
    else:
        sname = "Ion"

    fig = plt.figure(figsize=[7, 5])
    rect = [0.14, 0.16, 0.82, 0.8]
    ax = fig.add_axes(rect)
    for iframe, tframe in enumerate(tframes):
        print("Time frame: %d" % tframe)
        tindex = tframe * pic_info.fields_interval
        fdir = pic_run_dir + '/spectrum_combined/'
        # fname = fdir + 'spectrum_' + species.lower() + '_' + str(tindex) + '.dat'
        fname = fdir + 'spectrum_' + sname + '_' + str(tindex) + '.dat'
        flog = np.fromfile(fname, dtype=np.float32)
        espect = flog[3:] / debins / nptot  # the first 3 are magnetic field
        color = plt.cm.jet(tframe/float(ntf), 1)
        flogs[iframe, :] = espect
        ax.loglog(ebins_mid, espect, linewidth=1, color=color)

    pindex = -2.7
    fpower = ebins_mid**pindex * 5E0
    es, _ = find_nearest(ebins_mid, 10)
    ee, _ = find_nearest(ebins_mid, 100)
    ax.loglog(ebins_mid[es:ee], fpower[es:ee], linewidth=1, color='k')
    power_index = "{%0.1f}" % pindex
    pname = r'$\propto \varepsilon^{' + power_index + '}$'

    if species == 'e':
        ax.set_xlim([1E-1, 2E3])
        ax.set_ylim([1E-9, 1E0])
    else:
        ax.set_xlim([1E-1, 2E3])
        ax.set_ylim([1E-9, 1E0])
    ax.tick_params(bottom=True, top=True, left=True, right=False)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlabel(r'$\gamma - 1$', fontsize=16)
    ax.set_ylabel(r'$f(\gamma - 1)$', fontsize=16)
    ax.tick_params(labelsize=12)
    ename = 'electron' if species == 'e' else 'ion'
    fdir = '../img/power_law_index/spectrum/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + 'spectrum_' + species + '.pdf'
    fig.savefig(fname)

    # dflogs = np.gradient(flogs, axis=0)
    # dflogs[dflogs<=0] = np.nan
    # for iframe, tframe in enumerate(tframes):
    #     print("Time frame: %d" % tframe)
    #     fig = plt.figure(figsize=[7, 5])
    #     rect = [0.14, 0.16, 0.82, 0.8]
    #     ax = fig.add_axes(rect)
    #     ax.loglog(ebins_mid, dflogs[iframe, :], color='k')
    #     pindex = -1.5
    #     fpower = ebins_mid**pindex
    #     es, _ = find_nearest(ebins_mid, 10)
    #     ee, _ = find_nearest(ebins_mid, 200)
    #     ax.loglog(ebins_mid[es:ee], fpower[es:ee]*5E-3, linewidth=1, color='r')
    #     power_index = "{%0.1f}" % pindex
    #     pname = r'$\propto \varepsilon^{' + power_index + '}$'
    #     if species == 'e':
    #         ax.set_xlim([1E-1, 2E3])
    #         ax.set_ylim([1E-9, 1E-3])
    #     else:
    #         ax.set_xlim([1E-1, 2E3])
    #         ax.set_ylim([1E-9, 1E0])
    #     ax.tick_params(bottom=True, top=True, left=True, right=False)
    #     ax.tick_params(axis='x', which='minor', direction='in')
    #     ax.tick_params(axis='x', which='major', direction='in')
    #     ax.tick_params(axis='y', which='minor', direction='in')
    #     ax.tick_params(axis='y', which='major', direction='in')
    #     ax.set_xlabel(r'$\gamma - 1$', fontsize=16)
    #     ax.set_ylabel(r'$f(\gamma - 1)$', fontsize=16)
    #     ax.tick_params(labelsize=12)
    #     ename = 'electron' if species == 'e' else 'ion'
    #     fdir = '../img/power_law_index/spectrum/' + pic_run + '/'
    #     mkdir_p(fdir)
    #     fname = fdir + 'diff_spectrum_' + species + '_' + str(tframe) + '.pdf'
    #     fig.savefig(fname)
    #     plt.close()

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


def fluid_energization(plot_config, show_plot=True):
    """Plot fluid energization

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    if species == 'e':
        ylim = [-700, 700]
    else:
        ylim = [-700, 700]
    fig1 = plt.figure(figsize=[7, 2.5])
    box1 = [0.1, 0.18, 0.85, 0.68]
    axs1 = []
    fig2 = plt.figure(figsize=[7, 2.5])
    box2 = [0.1, 0.18, 0.85, 0.68]
    axs2 = []
    fig3 = plt.figure(figsize=[7, 2.5])
    box3 = [0.1, 0.18, 0.85, 0.68]
    axs3 = []

    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    if "3D" in pic_run:
        enorm = pic_info.ny
    else:
        enorm = 1.0
    tfields = pic_info.tfields * pic_info.dtwpe / pic_info.dtwce
    tenergy = pic_info.tenergy * pic_info.dtwpe / pic_info.dtwce
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
    epara_ene = fluid_ene[4*nframes+2:5*nframes+2]
    eperp_ene = fluid_ene[5*nframes+2:6*nframes+2]
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
    ax.set_ylabel('Energization', fontsize=12)
    pdim = "2D" if "2D" in pic_run else "3D"
    ax.text(0.02, 0.9, pdim, color='k', fontsize=12,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)
    ax.set_xlabel(r'$t\omega_{pe}$', fontsize=12)

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
    ax.set_ylabel('Energization', fontsize=12)
    pdim = "2D" if "2D" in pic_run else "3D"
    ax.text(0.02, 0.9, pdim, color='k', fontsize=12,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)
    ax.set_xlabel(r'$t\omega_{pe}$', fontsize=12)

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
    ax.set_ylabel('Energization', fontsize=12)
    pdim = "2D" if "2D" in pic_run else "3D"
    ax.text(0.02, 0.9, pdim, color='k', fontsize=12,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)
    ax.set_xlabel(r'$t\omega_{pe}$', fontsize=12)

    box1[0] += box1[2] + 0.07

    axs1[0].legend(loc='upper left', prop={'size': 12}, ncol=4,
                   bbox_to_anchor=(0, 1.22),
                   shadow=False, fancybox=False, frameon=False)
    axs2[0].legend(loc='upper left', prop={'size': 12}, ncol=4,
                   bbox_to_anchor=(0, 1.22),
                   shadow=False, fancybox=False, frameon=False)
    axs3[0].legend(loc='upper left', prop={'size': 12}, ncol=4,
                   bbox_to_anchor=(0, 1.22),
                   shadow=False, fancybox=False, frameon=False)
    fdir = '../img/power_law_index/fluid_energization/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + 'fluid_ene_' + species + '.pdf'
    fig1.savefig(fname)

    fdir = '../img/power_law_index/fluid_energization/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + 'fluid_drift_' + species + '.pdf'
    fig2.savefig(fname)

    fdir = '../img/power_law_index/fluid_energization/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + 'fluid_comp_shear_' + species + '.pdf'
    fig3.savefig(fname)

    if show_plot:
        plt.show()
    else:
        plt.close('all')


def get_plot_setup(plot_type, species):
    """Get plotting setup for different type
    """
    if plot_type == 'total':
        if species == 'e':
            ylims = [[-0.01, 0.02], [-0.005, 0.01],
                     [-0.002, 0.005], [-0.002, 0.005]]
        else:
            ylims = [[-0.001, 0.003], [-0.0005, 0.002],
                     [-0.0002, 0.0005], [-0.0002, 0.0004]]
        data_indices = [1, 2]
    if plot_type == 'perpendicular':
        if species == 'e':
            ylims = [[-0.01, 0.02], [-0.005, 0.01],
                     [-0.002, 0.005], [-0.002, 0.005]]
        else:
            ylims = [[-0.001, 0.003], [-0.0005, 0.002],
                     [-0.0002, 0.0005], [-0.0002, 0.0004]]
        data_indices = [2]
    if plot_type == 'parallel':
        if species == 'e':
            ylims = [[-0.01, 0.02], [-0.005, 0.01],
                     [-0.002, 0.005], [-0.002, 0.005]]
        else:
            ylims = [[-0.001, 0.003], [-0.0005, 0.002],
                     [-0.0002, 0.0005], [-0.0002, 0.0004]]
        data_indices = [1]
    elif plot_type == 'curvature':
        if species == 'e':
            ylims = [[-0.01, 0.04], [-0.005, 0.02],
                     [-0.002, 0.01], [-0.002, 0.01]]
        else:
            ylims = [[-0.001, 0.004], [-0.0005, 0.0015],
                     [-0.0002, 0.0003], [-0.0002, 0.0004]]
        data_indices = [5]
    elif plot_type == 'gradient':
        if species == 'e':
            ylims = [[-0.01, 0.04], [-0.005, 0.02],
                     [-0.002, 0.01], [-0.002, 0.01]]
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


def particle_energization_multi(plot_config):
    """Plot particle-based energization for multiple types

    Args:
        plot_config: plotting configuration
    """
    plot_config1 = plot_config.copy()
    plot_types = ["total", "perpendicular", "parallel",
                  "curvature", "gradient", "inertial",
                  "polarization", "parallel_drift", "mu",
                  "compression", "shear"]
    for ptype in plot_types:
        plot_config1["plot_type"] = ptype
        particle_energization(plot_config1, show_plot=False)


def particle_energization(plot_config, show_plot=True):
    """Particle-based energization

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    tframe = plot_config["tframe"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]

    ylims, data_indices = get_plot_setup(plot_config["plot_type"], species)

    tstarts = [10, 51, 91, 130]
    tends = [50, 90, 130, 160]
    nplots = len(tstarts)

    fnorm = 1E-3
    for iplot in range(nplots):
        tstart = tstarts[iplot]
        tend = tends[iplot]
        ylim = np.asarray(ylims[iplot]) / fnorm
        fig1 = plt.figure(figsize=[9.6, 4.0])
        box1 = [0.14, 0.2, 0.8, 0.75]
        axs1 = []

        nframes = tend - tstart

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
            nbinx = int(fdata[1])
            nvar = int(fdata[2])
            ebins = fdata[3:nbins+3]
            fbins = np.sum(fdata[nbins+3:].reshape((nvar, nbinx, nbins)), axis=1)

            if species == 'i':
                ebins *= pic_info.mime  # ebins are actually gamma
            fbins[1:, :] = div0(fbins[1:, :], fbins[0, :])

            color = plt.cm.jet((tframe - tstart)/float(nframes), 1)
            fdata = np.zeros(nbins)
            for idata in data_indices:
                fdata += fbins[idata, :]
            ax.semilogx(ebins, fdata/fnorm, linewidth=1, color=color)
        if species == 'e':
            ax.set_xlim([1E0, 2000])
        else:
            ax.set_xlim([1E0, 2000])
        ax.set_ylim(ylim)
        box1[0] += box1[2] + 0.02
        ax.set_ylabel(r'$\left<\nu^\text{I}_\text{COM}\right>/10^{-3}$', fontsize=10)

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

        # bg_str = str(int(bg * 10)).zfill(2)
        # fdir = '../img/cori_3d/particle_energization/bg' + bg_str + '/'
        # mkdir_p(fdir)
        # fname = (fdir + 'particle_' + plot_config["plot_type"] + '_' +
        #          species + '_' + str(iplot) + '.pdf')
        # fig1.savefig(fname)

    if show_plot:
        plt.show()
    else:
        plt.close('all')


def calc_vexb_kappa(plot_config):
    """Get the vexb dot magnetic curvature for the 2D simulations
    """
    tframe = plot_config["tframe"]
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
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
    kappa = np.sqrt(kappax**2 + kappay**2 + kappaz**2)
    kmin, kmax = 1E-6, 1E2
    nbins = 80
    kbins = np.logspace(math.log10(kmin), math.log10(kmax), nbins+1)
    kdist, _ = np.histogram(kappa, bins=kbins)
    fdata = np.zeros(nbins + 3)
    fdata[0] = kmin
    fdata[1] = kmax
    fdata[2] = nbins
    fdata[3:] = kdist
    fdir = '../data/power_law_index/kappa_dist/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + 'kappa_dist_' + str(tframe) + '.dat'
    fdata.tofile(fname)

    vexb_x = (ey * bz - ez * by) * ib
    vexb_y = (ez * bx - ex * bz) * ib
    vexb_z = (ex * by - ey * bx) * ib
    vexb_kappa = vexb_x * kappax + vexb_y * kappay + vexb_z * kappaz
    vkmin, vkmax = 1E-5, 1E3
    nbins = 80
    vkbins = np.zeros(2*nbins+3)
    fbins = np.logspace(math.log10(vkmin), math.log10(vkmax), nbins+1)
    vkbins[:nbins+1] = -fbins[::-1]
    vkbins[nbins+2:]= fbins
    vkdist, _ = np.histogram(vexb_kappa, bins=vkbins)
    fdata = np.zeros(2*nbins + 5)
    fdata[0] = vkmin
    fdata[1] = vkmax
    fdata[2] = nbins
    fdata[3:] = vkdist
    fdir = '../data/power_law_index/vexb_kappa_dist/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + 'vexb_kappa_dist_' + str(tframe) + '.dat'
    fdata.tofile(fname)
    # fbins_mid = 0.5 * (fbins[1:] + fbins[:-1])
    # vkdist = vkdist / np.diff(vkbins)
    # plt.loglog(fbins_mid, vkdist[nbins+2:])
    # plt.loglog(fbins_mid[::-1], vkdist[:nbins])
    # plt.show()

    fdir = pic_run_dir + "data/"
    fname = fdir + 'vexb_kappa.gda'
    size_one_frame = pic_info.nx * pic_info.nz * 4
    with open(fname, 'a+') as f:
        offset = size_one_frame * tframe
        f.seek(offset, os.SEEK_SET)
        vexb_kappa.tofile(f)


def plot_vexb_kappa(plot_config, show_plot=True):
    """
    Plot the distribution of vexb dot magnetic curvature for the 2D simulations
    """
    tframe = plot_config["tframe"]
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    tindex = tframe * pic_info.fields_interval
    fdir = '../data/power_law_index/vexb_kappa_dist/' + pic_run + '/'
    fname = fdir + 'vexb_kappa_dist_' + str(tframe) + '.dat'
    fdata = np.fromfile(fname)
    vkmin = fdata[0]
    vkmax = fdata[1]
    nbins = fdata[2]
    vkdist = fdata[3:]
    nbins = 80
    vkbins = np.zeros(2*nbins+3)
    fbins = np.logspace(math.log10(vkmin), math.log10(vkmax), nbins+1)
    vkbins[:nbins+1] = -fbins[::-1]
    vkbins[nbins+2:]= fbins
    vkbins_mid = 0.5 * (vkbins[1:] + vkbins[:-1])
    vkdist = div0(vkdist, np.diff(vkbins))
    fig = plt.figure(figsize=[7, 8])
    rect = [0.12, 0.35, 0.82, 0.6]
    ax = fig.add_axes(rect)
    ax.set_prop_cycle('color', COLORS)
    text1 = r'\boldsymbol{v}_{\boldsymbol{E}}\cdot\boldsymbol{\kappa}'
    label1 = r'$' + text1 + ' > 0$'
    ax.loglog(vkbins_mid[nbins+1:], vkdist[nbins+1:], label=label1)
    label2 = r'$' + text1 + ' < 0$'
    ax.loglog(vkbins_mid[nbins+1:], np.flip(vkdist[:nbins+1]), label=label2)
    ax.legend(loc=3, prop={'size': 16}, ncol=1,
             shadow=False, fancybox=False, frameon=False)
    ax.tick_params(bottom=True, top=True, left=True, right=False)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.tick_params(labelsize=12)
    xlabel = r'$|' + text1 + r'|$'
    ax.set_xlabel(xlabel, fontsize=16)
    ylabel = r'$f(|' + text1 + r'|)$'
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_xlim([1E-5, 1E2])
    ax.set_ylim([1E-1, 1E11])
    vsh = plot_config["vkappa_threshold"]
    ax.plot([vsh, vsh], ax.get_ylim(), color='k',
            linewidth=1, linestyle='-')
    ax.plot([vsh*0.5, vsh*0.5], ax.get_ylim(), color='k',
            linewidth=1, linestyle='--')
    ax.plot([vsh*2, vsh*2], ax.get_ylim(), color='k',
            linewidth=1, linestyle='--')
    dtwpe_fields = math.ceil(pic_info.fields_interval * pic_info.dtwpe / 0.1) * 0.1
    twpe = tframe * dtwpe_fields
    text = r'$t\omega_{pe}=' + str(twpe) + '$'
    ax.set_title(text, fontsize=16)

    rect[3] = 0.2
    rect[1] -= rect[3] + 0.07
    xmin, xmax = 0, pic_info.lx_di
    zmin, zmax = -pic_info.lz_di * 0.25, pic_info.lz_di * 0.25
    kwargs = {"current_time": tframe,
              "xl": xmin, "xr": xmax,
              "zb": zmin, "zt": zmax}
    fname = pic_run_dir + "data/vexb_kappa.gda"
    x, z, vexb_kappa = read_2d_fields(pic_info, fname, **kwargs)
    vmin, vmax = -vsh*25, vsh*25
    ax = fig.add_axes(rect)
    ng = 5
    kernel = np.ones((ng,ng)) / float(ng*ng)
    vexb_kappa = signal.convolve2d(vexb_kappa, kernel, mode='same')
    p2 = ax.imshow(vexb_kappa, extent=[xmin, xmax, zmin, zmax],
                   vmin=vmin, vmax=vmax,
                   cmap=plt.cm.seismic, aspect='auto',
                   origin='lower', interpolation='none')
    levels = [vsh]
    cs = ax.contour(x, z, np.abs(vexb_kappa), colors='k',
                    linewidths=0.5, levels=levels)
    ax.tick_params(bottom=True, top=True, left=True, right=False)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.tick_params(labelsize=12)
    ax.set_xlabel(r'$x/d_i$', fontsize=16)
    ax.set_ylabel(r'$z/d_i$', fontsize=16)
    rect_cbar = [0.72, 0.24, 0.2, 0.01]
    cbar_ax = fig.add_axes(rect_cbar)
    cbar = fig.colorbar(p2, cax=cbar_ax, orientation='horizontal',
                        extend='both')
    cbar.ax.tick_params(labelsize=16)
    lablel = r'$' + text1 + r'$'
    cbar_ax.set_xlabel(lablel, fontsize=16)
    fdir = '../img/power_law_index/vexb_kappa_dist/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + 'vexb_kappa_dist_' + species + '_' + str(tframe) + '.jpg'
    fig.savefig(fname, dpi=400)

    if show_plot:
        plt.show()
    else:
        plt.close()


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
    else:
        sname = "Ion"
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
    emin, emax = 1E-4, 1E6
    nbins = 1000
    dloge = (math.log10(emax) - math.log10(emin)) / (nbins - 1)
    emin0 = 10**(math.log10(emin) - dloge)
    ebins = np.logspace(math.log10(emin0), math.log10(emax), nbins+1)
    ebins_mid = (ebins[:-1] + ebins[1:]) * 0.5
    debins = np.diff(ebins)
    dtwpe_fields = math.ceil(pic_info.fields_interval * pic_info.dtwpe / 0.1) * 0.1
    espect_elb = np.zeros(nbins)
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    for tframe in range(tstart, tend+1):
        print("Time frame: %d" % tframe)
        tindex = tframe * pic_info.fields_interval
        fig = plt.figure(figsize=[7, 5])
        rect = [0.12, 0.12, 0.82, 0.8]
        ax = fig.add_axes(rect)
        ax.set_prop_cycle('color', COLORS)
        isp = 0
        for sp, sp_name in zip(sp_list, sp_name_list):
            print(sp)
            fname = (pic_run_dir + "spectrum_combined/spectrum_" + sp +
                     "_" + str(tindex) + ".dat")
            flog = np.fromfile(fname, dtype=np.float32)
            espect = flog[3:]  # the first 3 are magnetic field
            if sp != sname + "_egtb":
                ax.loglog(ebins_mid, espect*norms[isp], linewidth=2, label=sp_name)
            else:
                espect_elb = np.copy(espect)
            if sp == sname + "_egtb_egtb":
                espect_elb -= espect
            isp += 1
        sp_name = r"tracers passing $E\leq B$"
        ax.loglog(ebins_mid, espect_elb*norms[isp], linewidth=2, label=sp_name)
        ax.legend(loc=3, prop={'size': 12}, ncol=1,
                 shadow=False, fancybox=False, frameon=False)
        ax.set_xlim([1E-1, 2E3])
        ax.set_ylim([1E3, 5E7])
        ax.set_xlabel(r'$\gamma - 1$', fontsize=16)
        ax.set_ylabel(r'$(\gamma - 1)f(\gamma - 1)$', fontsize=16)
        ax.tick_params(labelsize=12)
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')
        twpe = tframe * dtwpe_fields
        text1 = r'$t\omega_{pe}=' + str(twpe) + '$'
        ax.set_title(text1, fontsize=16)
        fdir = '../img/power_law_index/spect_species/' + pic_run + '/'
        mkdir_p(fdir)
        fname = fdir + 'spects_' + species + '_' + str(tframe) + '.pdf'
        fig.savefig(fname)
        if show_plot:
            plt.show()
        else:
            plt.close()


def acc_esc_rate(plot_config, show_plot=True):
    """calculate particle acceleration and escape rates

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    tframe = plot_config["tframe"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    alpha_min = plot_config["vkappa_threshold"]
    ntf = pic_info.ntf
    emin = 10
    emax = 100
    # ntf = 41
    # emin = 0.6

    fpath = "../data/particle_interp/" + pic_run + "/"
    tindex = pic_info.particle_interval
    fname = fpath + "acc_rate_dist_" + species + "_" + str(tindex) + ".gda"
    fdata = np.fromfile(fname, dtype=np.float32)
    nvar = int(fdata[2])   # number of variables
    # nvar = int(fdata[3])   # number of variables

    nhigh_acc_t = np.zeros(ntf)
    nhigh_esc_t = np.zeros(ntf)
    arate_acc_t = np.zeros([nvar, ntf])
    arate_esc_t = np.zeros([nvar, ntf])

    for tframe in range(1, ntf):
        print("Time frame: %d" % tframe)
        tindex = tframe * pic_info.particle_interval
        fname = fpath + "acc_rate_dist_" + species + "_" + str(tindex) + ".gda"
        fdata = np.fromfile(fname, dtype=np.float32)
        nalpha = int(fdata[0]) # number of bins of the rates
        nbins = int(fdata[1])  # number of bins along x
        nvar = int(fdata[2])   # number of variables
        ebins = fdata[3:nbins+3]  # energy bins
        alpha_bins = fdata[nbins+3:nbins+nalpha+3]  # acceleration rates bins
        alpha_bins_mid = 0.5 * (alpha_bins[1:] + alpha_bins[:-1])
        # variables: 0-Epara, 1-Eperp, 2-compression, 3-shear, 4-curvature,
        #   5-gradient, 6-para_drift, 7-mu, 8-polar_time, 9-polar_spatial,
        #   10-inertial_time, 11-inertial_spatial, 12-polar_fluid_time,
        #   13-polar_fluid_spatial, 14-polar_time_v, 15-polar_spatial_v,
        #   16-ptensor
        fdist = fdata[nbins+nalpha+3:].reshape((nvar, nbins, (nalpha+1)*4))
        # fdata = fdist[:, :, (nalpha+1)*2:]
        # fdata1 = np.sum(fdata, axis=2) * ebins
        # print(np.sum(fdata1, axis=1))

        # fname = fpath + "acc_rate_dist_" + species + "_" + str(tindex) + ".gda"
        # fdata = np.fromfile(fname, dtype=np.float32)
        # nalpha = int(fdata[0]) # number of bins of the rates
        # nbins = int(fdata[1])  # number of bins along x
        # nbinx = int(fdata[2])  # number of energy bins
        # nvar = int(fdata[3])   # number of variables
        # ebins = fdata[4:nbins+4]  # energy bins
        # alpha_bins = fdata[nbins+4:nbins+nalpha+4]  # acceleration rates bins
        # fdist_x = fdata[nbins+nalpha+4:].reshape((nvar, nbinx, nbins, (nalpha+1)*4))
        # fdist = np.sum(fdist_x, axis=1)

        es, _ = find_nearest(ebins, emin)
        ee, _ = find_nearest(ebins, emax)
        alpha_s, _ = find_nearest(alpha_bins, alpha_min)
        fdist_high = np.sum(fdist[:, es:ee, :], axis=1)
        nalpha0 = nalpha + 1
        nptl_bins = fdist_high[:, :nalpha0*2]
        dene_bins = fdist_high[:, nalpha0*2:]
        alpha_bins = div0(dene_bins, nptl_bins)
        fnptl = nptl_bins[4]
        fdene = dene_bins[4]
        pos_range = np.arange(nalpha0+1, nalpha0*2-1)
        neg_range = np.arange(nalpha0-2, 0, -1)
        fnptl_pos = fnptl[pos_range]
        fnptl_neg = fnptl[neg_range]
        fdene_pos = fdene[pos_range]
        fdene_neg = fdene[neg_range]
        # plt.loglog(alpha_bins_mid, fnptl_pos)
        # plt.loglog(alpha_bins_mid, fnptl_neg)
        nhigh_acc_t[tframe] = np.sum(fnptl_pos[alpha_s:] + fnptl_neg[alpha_s:])
        nhigh_esc_t[tframe] = np.sum(fnptl_pos + fnptl_neg) - nhigh_acc_t[tframe]

        fnptl_var_pos = nptl_bins[:, pos_range]
        fnptl_var_neg = nptl_bins[:, neg_range]
        fdene_var_pos = dene_bins[:, pos_range]
        fdene_var_neg = dene_bins[:, neg_range]

        arate_acc_t[:, tframe] = np.sum(fdene_var_pos[:, alpha_s:] +
                                        fdene_var_neg[:, alpha_s:], axis=1)
        arate_esc_t[:, tframe] = (np.sum(fdene_var_pos + fdene_var_neg, axis=1) -
                                  arate_acc_t[:, tframe])

    dtwpe_fields = pic_info.particle_interval * pic_info.dtwpe
    tfields = np.arange(0, ntf) * dtwpe_fields
    tmin, tmax = tfields[0], tfields[-1]
    esc_rate = div0(np.gradient(nhigh_esc_t, dtwpe_fields), nhigh_acc_t)
    acc_rate = div0(arate_acc_t[4, :], nhigh_acc_t)
    acc_rate1 = div0(arate_acc_t[4, :] + arate_acc_t[5, :], nhigh_acc_t)
    acc_rate2 = div0(arate_esc_t[1, :], nhigh_esc_t)
    pindex = 1 + div0(esc_rate, acc_rate)
    # plt.plot(esc_rate, color='r')
    # plt.plot(acc_rate, color='b')
    # plt.plot(acc_rate1, color='r')
    # plt.plot(acc_rate2, color='b')
    fig = plt.figure(figsize=[7, 5])
    rect = [0.10, 0.12, 0.86, 0.84]
    ax = fig.add_axes(rect)
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    ax.set_prop_cycle('color', COLORS)
    ax.plot(tfields, pindex, marker='o', linewidth=2)
    ax.plot([tmin, tmax], [1, 1], color='k', linewidth=1, linestyle='--')
    ax.plot([tmin, tmax], [2, 2], color='k', linewidth=1, linestyle='--')
    ax.set_xlim([tmin, tmax])
    ax.set_ylim([0, 5])
    ax.tick_params(bottom=True, top=True, left=True, right=False)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlabel(r'$t\omega_{pe}$', fontsize=16)
    text1 = r'$1+(\alpha\tau_\text{esc})^{-1}$'
    ax.set_ylabel(text1, fontsize=16)
    ax.tick_params(labelsize=12)
    fdir = '../img/power_law_index/power_index/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + 'power_index_' + species + '.pdf'
    fig.savefig(fname)
    plt.show()


def rates_based_vkappa(plot_config, show_plot=True):
    """calculate particle acceleration and escape rates

    This is based on rates bins with vkappa, either particle-based or grid-based

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    tframe = plot_config["tframe"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    alpha_min = plot_config["vkappa_threshold"]
    ntf = pic_info.ntf
    emin = 10
    emax = 100
    # ntf = 41
    # emin = 0.6

    fpath = "../data/particle_interp/" + pic_run + "/"
    tindex = pic_info.particle_interval
    fname = fpath + "acc_rate_dist_" + species + "_" + str(tindex) + ".gda"
    fdata = np.fromfile(fname, dtype=np.float32)
    nvar = int(fdata[2])   # number of variables
    # nvar = int(fdata[3])   # number of variables

    nhigh_acc_t = np.zeros(ntf)
    nhigh_esc_t = np.zeros(ntf)
    arate_acc_t = np.zeros([nvar, ntf])
    arate_esc_t = np.zeros([nvar, ntf])

    for tframe in range(1, ntf):
        print("Time frame: %d" % tframe)
        tindex = tframe * pic_info.particle_interval
        fname = fpath + "acc_rate_dist_vkappa_" + species + "_" + str(tindex) + ".gda"
        fdata = np.fromfile(fname, dtype=np.float32)
        nalpha = int(fdata[0]) # number of bins of the rates
        nbins = int(fdata[1])  # number of bins along x
        nvar = int(fdata[2])   # number of variables
        ebins = fdata[3:nbins+3]  # energy bins
        alpha_bins = fdata[nbins+3:nbins+nalpha+3]  # acceleration rates bins
        alpha_bins_mid = 0.5 * (alpha_bins[1:] + alpha_bins[:-1])
        # variables: 0-Epara, 1-Eperp, 2-compression, 3-shear, 4-curvature,
        #   5-gradient, 6-para_drift, 7-mu, 8-polar_time, 9-polar_spatial,
        #   10-inertial_time, 11-inertial_spatial, 12-polar_fluid_time,
        #   13-polar_fluid_spatial, 14-polar_time_v, 15-polar_spatial_v,
        #   16-ptensor
        fdist = fdata[nbins+nalpha+3:].reshape((nvar+1, nbins, (nalpha+1)*2))

        es, _ = find_nearest(ebins, emin)
        ee, _ = find_nearest(ebins, emax)
        alpha_s, _ = find_nearest(alpha_bins, alpha_min)
        fdist_high = np.sum(fdist[:, es:ee, :], axis=1)
        nalpha0 = nalpha + 1
        nptl_bins = fdist_high[0, :]
        dene_bins = fdist_high[1:, :]
        alpha_bins = div0(dene_bins, nptl_bins)
        fnptl = np.copy(nptl_bins)
        fdene = dene_bins[4]
        pos_range = np.arange(nalpha0+1, nalpha0*2-1)
        neg_range = np.arange(nalpha0-2, 0, -1)
        fnptl_pos = fnptl[pos_range]
        fnptl_neg = fnptl[neg_range]
        fdene_pos = fdene[pos_range]
        fdene_neg = fdene[neg_range]
        # plt.loglog(alpha_bins_mid, fnptl_pos)
        # plt.loglog(alpha_bins_mid, fnptl_neg)
        nhigh_acc_t[tframe] = np.sum(fnptl_pos[alpha_s:] + fnptl_neg[alpha_s:])
        nhigh_esc_t[tframe] = np.sum(fnptl_pos + fnptl_neg) - nhigh_acc_t[tframe]

        fdene_var_pos = dene_bins[:, pos_range]
        fdene_var_neg = dene_bins[:, neg_range]

        arate_acc_t[:, tframe] = np.sum(fdene_var_pos[:, alpha_s:] +
                                        fdene_var_neg[:, alpha_s:], axis=1)
        arate_esc_t[:, tframe] = (np.sum(fdene_var_pos + fdene_var_neg, axis=1) -
                                  arate_acc_t[:, tframe])

    dtwpe_fields = pic_info.particle_interval * pic_info.dtwpe
    tfields = np.arange(0, ntf) * dtwpe_fields
    tmin, tmax = tfields[0], tfields[-1]
    esc_rate = div0(np.gradient(nhigh_esc_t, dtwpe_fields), nhigh_acc_t)
    acc_rate = div0(arate_acc_t[4, :], nhigh_acc_t)
    acc_rate1 = div0(arate_acc_t[4, :] + arate_acc_t[5, :], nhigh_acc_t)
    acc_rate2 = div0(arate_esc_t[1, :], nhigh_esc_t)
    pindex = 1 + div0(esc_rate, acc_rate)
    # plt.plot(esc_rate, color='r')
    # plt.plot(acc_rate, color='b')
    # plt.plot(acc_rate1, color='r')
    # plt.plot(acc_rate2, color='b')
    fig = plt.figure(figsize=[7, 5])
    rect = [0.10, 0.12, 0.86, 0.84]
    ax = fig.add_axes(rect)
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    ax.set_prop_cycle('color', COLORS)
    ax.plot(tfields, pindex, marker='o', linewidth=2)
    ax.plot([tmin, tmax], [1, 1], color='k', linewidth=1, linestyle='--')
    ax.plot([tmin, tmax], [2, 2], color='k', linewidth=1, linestyle='--')
    ax.set_xlim([tmin, tmax])
    ax.set_ylim([0, 5])
    ax.tick_params(bottom=True, top=True, left=True, right=False)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlabel(r'$t\omega_{pe}$', fontsize=16)
    text1 = r'$1+(\alpha\tau_\text{esc})^{-1}$'
    ax.set_ylabel(text1, fontsize=16)
    ax.tick_params(labelsize=12)
    fdir = '../img/power_law_index/power_index/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + 'power_index_' + species + '.pdf'
    fig.savefig(fname)
    plt.show()


def get_cmd_args():
    """Get command line arguments
    """
    # default_pic_run = 'sigmae100_bg005_800de_Lde60_triggered'
    default_pic_run = 'sigmae25_bg005_800de_triggered'
    default_pic_run_dir = ('/net/scratch4/xiaocanli/reconnection/power_law_index/' +
                           default_pic_run + '/')
    parser = argparse.ArgumentParser(description='Analysis for runs to determine power-law indices')
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
    parser.add_argument('--vkappa_threshold', action="store", default='4E-2',
                        type=float, help='the threshold for vexb_dot_kappa')
    parser.add_argument('--all_frames', action="store_true", default=False,
                        help='whether to analyze all frames')
    parser.add_argument('--show_plot', action="store_true", default=False,
                        help='whether to show plot')
    parser.add_argument('--plot_spect', action="store_true", default=False,
                        help='whether to plot particle energy spectrum')
    parser.add_argument('--check_density', action="store_true", default=False,
                        help='whether to check maximum density')
    parser.add_argument('--fluid_ene', action="store_true", default=False,
                        help='whether to plot fluid energization')
    parser.add_argument('--particle_ene', action="store_true", default=False,
                        help='whether to plot particle energization')
    parser.add_argument('--plot_type', action="store", default='total', type=str,
                        help='Particle plot type')
    parser.add_argument('--multi_types', action="store_true", default=False,
                        help='Multiple particle plot types')
    parser.add_argument('--calc_vexb_kappa', action="store_true", default=False,
                        help='whether to calculate vexb dot magnetic curvature')
    parser.add_argument('--plot_vexb_kappa', action="store_true", default=False,
                        help='whether to plot vexb dot magnetic curvature')
    parser.add_argument('--spect_species', action="store_true", default=False,
                        help='energy spectrum for different species')
    parser.add_argument('--acc_esc_rate', action="store_true", default=False,
                        help='calculate particle acceleration and escape rates')
    parser.add_argument('--rates_vkappa', action="store_true", default=False,
                        help='calculate acceleration and escape rates based ' +
                        'distributions of accelerations rate binned with vkappa')
    return parser.parse_args()


def analysis_single_frames(plot_config, args):
    """Analysis for multiple time frames
    """
    tframe = args.tframe
    if args.plot_spect:
        if args.all_frames:
            plot_spectrum_multi(plot_config)
    elif args.check_density:
        check_density(plot_config)
    elif args.fluid_ene:
        fluid_energization(plot_config)
    elif args.particle_ene:
        if args.multi_types:
            particle_energization_multi(plot_config)
        else:
            particle_energization(plot_config)
    elif args.calc_vexb_kappa:
        calc_vexb_kappa(plot_config)
    elif args.plot_vexb_kappa:
        plot_vexb_kappa(plot_config)
    elif args.spect_species:
        plot_spect_species(plot_config, args.show_plot)
    elif args.acc_esc_rate:
        acc_esc_rate(plot_config, args.show_plot)
    elif args.rates_vkappa:
        rates_based_vkappa(plot_config, args.show_plot)


def process_input(plot_config, args, tframe):
    """process one time frame"""
    plot_config["tframe"] = tframe


def analysis_multi_frames(plot_config, args):
    """Analysis for multiple time frames
    """
    tframes = range(plot_config["tstart"], plot_config["tend"] + 1)
    if args.time_loop:
        for tframe in tframes:
            print("Time frame: %d" % tframe)
            plot_config["tframe"] = tframe
            if args.calc_vexb_kappa:
                calc_vexb_kappa(plot_config)
            elif args.plot_vexb_kappa:
                plot_vexb_kappa(plot_config, show_plot=False)
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
    plot_config["vkappa_threshold"] = args.vkappa_threshold
    if args.multi_frames:
        analysis_multi_frames(plot_config, args)
    else:
        analysis_single_frames(plot_config, args)


if __name__ == "__main__":
    main()
