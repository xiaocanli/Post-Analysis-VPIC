#!/usr/bin/env python3
"""
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


def energy_conversion(plot_config):
    """Plot energy conversion

    Args:
        plot_config: plotting configuration
    """
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    fig = plt.figure(figsize=[3.25, 2.5])
    w1, h1 = 0.78, 0.78
    xs, ys = 0.96 - w1, 0.96 - h1
    ax = fig.add_axes([xs, ys, w1, h1])

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

    ax.plot(tenergy, ene_magnetic, linewidth=1, linestyle='-', label='magnetic')
    ax.plot(tenergy, kene_e, linewidth=1, linestyle='-', label='electron')
    ax.plot(tenergy, kene_i, linewidth=1, linestyle='-', label='ion')

    ax.legend(loc=6, prop={'size': 12}, ncol=1,
              shadow=False, fancybox=False, frameon=False)
    ax.tick_params(bottom=True, top=False, left=True, right=True)
    ax.set_xlim([0, 400])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel(r'$t\Omega_{ci}$', fontsize=12)
    ax.set_ylabel(r'$\text{Energy}/\varepsilon_{B0}$', fontsize=12)
    ax.tick_params(labelsize=10)

    fdir = '../img/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + 'econv.pdf'
    fig.savefig(fname)
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
        ylim = [-0.2, 1.0]
    else:
        ylim = [-0.6, 1.7]
    fig1 = plt.figure(figsize=[7, 2.5])
    box1 = [0.09, 0.18, 0.85, 0.68]
    axs1 = []
    fig2 = plt.figure(figsize=[7, 2.5])
    box2 = [0.09, 0.18, 0.85, 0.68]
    axs2 = []
    fig3 = plt.figure(figsize=[7, 2.5])
    box3 = [0.09, 0.18, 0.85, 0.68]
    axs3 = []

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
    # ax.set_ylim(ylim)
    ax.tick_params(labelsize=10)
    ax.set_ylabel('Energization', fontsize=12)
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
    # ax.set_ylim(ylim)
    ax.tick_params(labelsize=10)
    ax.set_ylabel('Energization', fontsize=12)
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
    # ax.set_ylim(ylim)
    ax.tick_params(labelsize=10)
    ax.set_ylabel('Energization', fontsize=12)
    pdim = "2D" if "2D" in pic_run else "3D"
    ax.text(0.02, 0.9, pdim, color='k', fontsize=12,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)
    ax.set_xlabel(r'$t\Omega_{ci}$', fontsize=12)

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
    fdir = '../img/cori_3d/fluid_energization/'
    mkdir_p(fdir)
    fname = fdir + 'fluid_ene_' + species + '.pdf'
    # fig1.savefig(fname)

    fdir = '../img/cori_3d/fluid_energization/'
    mkdir_p(fdir)
    fname = fdir + 'fluid_drift_' + species + '.pdf'
    # fig2.savefig(fname)

    fdir = '../img/cori_3d/fluid_energization/'
    mkdir_p(fdir)
    fname = fdir + 'fluid_comp_shear_' + species + '.pdf'
    # fig3.savefig(fname)

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

    tstarts = [6, 10, 20, 30]
    tends = [10, 20, 30, 40]
    nplots = len(tstarts)

    fnorm = 1E-3
    for iplot in range(nplots):
        tstart = tstarts[iplot]
        tend = tends[iplot]
        ylim = np.asarray(ylims[iplot]) / fnorm
        fig1 = plt.figure(figsize=[4.8, 2.0])
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
        ax.set_ylabel(r'$\left<\nu^\text{I}_\text{COM}\right>/10^{-3}$', fontsize=10)
        ax.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$', fontsize=10)
        ax.tick_params(labelsize=8)

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


def get_cmd_args():
    """Get command line arguments
    """
    default_pic_run = '3D-Lx150-bg1.0-150ppc-2048KNL'
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
    parser.add_argument('--econv', action="store_true", default=False,
                        help='whether to plot energy conversion')
    parser.add_argument('--show_plot', action="store_true", default=False,
                        help='whether to show plot')
    parser.add_argument('--fluid_ene', action="store_true", default=False,
                        help='whether to plot fluid energization')
    parser.add_argument('--particle_ene', action="store_true", default=False,
                        help='whether to plot particle energization')
    parser.add_argument('--plot_type', action="store", default='total', type=str,
                        help='Particle plot type')
    parser.add_argument('--multi_types', action="store_true", default=False,
                        help='Multiple particle plot types')
    return parser.parse_args()


def analysis_single_frames(plot_config, args):
    """Analysis for multiple time frames
    """
    tframe = args.tframe
    if args.econv:
        energy_conversion(plot_config)
    elif args.fluid_ene:
        fluid_energization(plot_config)
    elif args.particle_ene:
        particle_energization(plot_config)


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
