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
    emin, emax = 1E-6, 1E4
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
    for iframe, tframe in enumerate(tframes):
        print("Time frame: %d" % tframe)
        tindex = tframe * pic_info.fields_interval
        fdir = pic_run_dir + '/spectrum_combined/'
        fname = fdir + 'spectrum_' + sname + '_' + str(tindex) + '.dat'
        flog = np.fromfile(fname, dtype=np.float32)
        espect = flog[3:] / debins / nptot  # the first 3 are magnetic field
        color = plt.cm.jet(tframe/float(ntf), 1)
        flogs[iframe, :] = espect
        ax.loglog(ebins_mid, espect, linewidth=1, color=color)

    if species == 'e':
        ax.set_xlim([1E-1, 1E4])
        ax.set_ylim([1E-9, 1E0])
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
    # fpath = "../img/img_high_mime/spectra/" + ename + "/"
    # mkdir_p(fpath)
    # fname = fpath + "spect_time_" + run_name + "_" + species + ".pdf"
    # fig.savefig(fname)
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


def cumulative_energization(plot_config, show_plot=True):
    """calculate the cumulative energization of tracer particles

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
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
    # nframes = 21
    plot_interval = plot_config["plot_interval"]

    if species in ["e", "electron"]:
        sname = "electron"
        pmass = 1.0
    else:
        sname = "H"
        pmass = pic_info.mime

    fname = tracer_dir + 'T.0/' + sname + '_tracer_qtag_sorted.h5p'
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
    dene_para = np.zeros([4, nptl])
    dene_perp = np.zeros([4, nptl])

    gamma0 = np.sqrt(1.0 + ptl["Ux"]**2 + ptl["Uy"]**2 + ptl["Uz"]**2)
    cond_exclude_cs = np.abs(ptl["dZ"]) > half_thickness_cs
    dgamma_min, dgamma_max = 0, 4
    dene_min, dene_max = 0, 4
    nbins = 128
    drange = [[dgamma_min, dgamma_max], [dene_min, dene_max]]
    ebins = np.logspace(dene_min, dene_max, nbins+1)
    ebins_mid = 0.5 * (ebins[1:] + ebins[:-1])
    vmin, vmax = 1E0, 1E2
    xyz = ['x', 'y', 'z']

    fdir = '../img/trans_relativistic/hist_dene_dgamma/' + pic_run + '/'
    if plot_config["exclude_cs"]:
        fdir += 'nocs/'
    else:
        fdir += 'all/'
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
            istep = tindex // pic_info.tracer_interval
            if istep % plot_interval == 0 and istep != 0:
                for para, ixyz in itertools.product(range(2), range(4)):
                    if para:
                        dene = dene_para[ixyz, :]
                        comp = '\parallel'
                        cname = 'para'
                    else:
                        dene = dene_perp[ixyz, :]
                        comp = '\perp'
                        cname = 'perp'
                    fig = plt.figure(figsize=[10, 5])
                    rect = [0.08, 0.12, 0.4, 0.8]
                    hgap, vgap = 0.03, 0.03
                    ax1 = fig.add_axes(rect)
                    cond1 = dgamma > 0
                    cond2 = dene < 0
                    cond = np.logical_and(cond1, cond2)
                    if plot_config["exclude_cs"]:
                        cond = np.logical_and(cond, cond_exclude_cs)
                    nptl_cond = len(dgamma[cond])
                    hist, ubins_edges, ubins_edges = np.histogram2d(np.log10(dgamma[cond]),
                                                                    np.log10(-dene[cond]),
                                                                    bins=nbins, range=drange)
                    hist1d_1, _ = np.histogram(-dene[cond2], bins=ebins)
                    hist1d_2, _ = np.histogram(-dene[cond2], bins=ebins, weights=dgamma[cond2])
                    dgamma_avg = div0(hist1d_2, hist1d_1)
                    p1 = ax1.imshow(hist,
                                    extent=[dgamma_min, dgamma_max, dene_min, dene_max],
                                    norm = LogNorm(vmin=vmin, vmax=vmax),
                                    cmap=plt.cm.viridis, aspect='auto',
                                    origin='lower', interpolation='nearest')
                    ax1.plot(np.log10(ebins_mid), np.log10(dgamma_avg), color=COLORS[0],
                             label=r'$\left<\Delta\gamma\right>$')
                    if ixyz < 3:
                        xlabel = r'$\log(' + '|W_{' + comp + ',' + xyz[ixyz] + '}<0|' + ')$'
                    else:
                        xlabel = r'$\log(' + '|W_{' + comp + '}<0|' + ')$'
                    ax1.plot([0, 4], [0, 4], color='k', linewidth=1, linestyle='--',
                            label=r'$\Delta\gamma=$' + xlabel)
                    ax1.plot([0, 4], [math.log10(sigma_e*0.5), math.log10(sigma_e*0.5)],
                            color='k', linewidth=1, linestyle='-',
                            label=r'$\Delta\gamma=\sigma_e/2$')
                    ax1.legend(loc=1, prop={'size': 12}, ncol=1,
                              shadow=False, fancybox=False, frameon=False)
                    ax1.tick_params(bottom=True, top=True, left=True, right=True)
                    ax1.tick_params(axis='x', which='minor', direction='in')
                    ax1.tick_params(axis='x', which='major', direction='in')
                    ax1.tick_params(axis='y', which='minor', direction='in')
                    ax1.tick_params(axis='y', which='major', direction='in')
                    ax1.set_xlim([4, 0])
                    ax1.set_ylim([0, 4])
                    ax1.set_xlabel(xlabel, fontsize=16)
                    ax1.set_ylabel(r'$\log(\Delta\gamma)$', fontsize=16)
                    ax1.tick_params(labelsize=12)

                    rect[0] += rect[2] + hgap
                    ax2 = fig.add_axes(rect)
                    cond1 = dgamma > 0
                    cond2 = dene > 0
                    cond = np.logical_and(cond1, cond2)
                    if plot_config["exclude_cs"]:
                        cond = np.logical_and(cond, cond_exclude_cs)
                    nptl_cond = len(dgamma[cond])
                    hist, ubins_edges, ubins_edges = np.histogram2d(np.log10(dgamma[cond]),
                                                                    np.log10(dene[cond]),
                                                                    bins=nbins, range=drange)
                    hist1d_1, _ = np.histogram(dene[cond2], bins=ebins)
                    hist1d_2, _ = np.histogram(dene[cond2], bins=ebins, weights=dgamma[cond2])
                    dgamma_avg = div0(hist1d_2, hist1d_1)
                    p1 = ax2.imshow(hist,
                                    extent=[dgamma_min, dgamma_max, dene_min, dene_max],
                                    norm = LogNorm(vmin=vmin, vmax=vmax),
                                    cmap=plt.cm.viridis, aspect='auto',
                                    origin='lower', interpolation='nearest')
                    ax2.plot(np.log10(ebins_mid), np.log10(dgamma_avg), color=COLORS[0],
                             label=r'$\left<\Delta\gamma\right>$')
                    if ixyz < 3:
                        xlabel = r'$\log(' + '|W_{' + comp + ',' + xyz[ixyz] + '}>0|' + ')$'
                    else:
                        xlabel = r'$\log(' + '|W_{' + comp + '}>0|' + ')$'
                    ax2.plot([0, 4], [0, 4], color='k', linewidth=1, linestyle='--',
                            label=r'$\Delta\gamma=$' + xlabel)
                    ax2.plot([0, 4], [math.log10(sigma_e*0.5), math.log10(sigma_e*0.5)],
                            color='k', linewidth=1, linestyle='-',
                            label=r'$\Delta\gamma=\sigma_e/2$')
                    ax2.legend(loc=2, prop={'size': 12}, ncol=1,
                              shadow=False, fancybox=False, frameon=False)
                    ax2.tick_params(bottom=True, top=True, left=True, right=True)
                    ax2.tick_params(axis='x', which='minor', direction='in')
                    ax2.tick_params(axis='x', which='major', direction='in')
                    ax2.tick_params(axis='y', which='minor', direction='in')
                    ax2.tick_params(axis='y', which='major', direction='in')
                    ax2.set_xlim([0, 4])
                    ax2.set_ylim([0, 4])
                    ax2.set_xlabel(xlabel, fontsize=16)
                    ax2.tick_params(axis='y', labelleft=False)
                    ax2.tick_params(labelsize=12)

                    rect_cbar = np.copy(rect)
                    rect_cbar[0] += rect[2] + 0.01
                    rect_cbar[2] = 0.02
                    cbar_ax = fig.add_axes(rect_cbar)
                    cbar = fig.colorbar(p1, cax=cbar_ax, extend='both')
                    cbar_ax.tick_params(bottom=False, top=False, left=False, right=True)
                    cbar_ax.tick_params(axis='y', which='major', direction='out')
                    cbar_ax.tick_params(axis='y', which='minor', direction='in', right=False)
                    cbar_ax.set_title(r'$N_e$', fontsize=16)
                    cbar.ax.tick_params(labelsize=12)
                    twpe = math.ceil(tindex * pic_info.dtwpe / 0.1) * 0.1
                    text1 = r'$t\omega_{pe}=' + ("{%0.0f}" % twpe) + '$'
                    fig.suptitle(text1, fontsize=16)
                    fname = fdir + 'hist_dw_' + cname + '_'
                    if plot_config["exclude_cs"]:
                        fname += 'nocs_'
                    if ixyz < 3:
                        fname += xyz[ixyz]
                    else:
                        fname += 'tot'
                    fname += '_' + species + '_' + str(istep) + '.pdf'
                    fig.savefig(fname)
                    plt.close()
                    # plt.show()
            dene_para /= dtwpe_tracer
            dene_perp /= dtwpe_tracer
        fh.close()


def compare_dw_para_perp(plot_config, show_plot=True):
    """Compare energization due to parallel and perpendicular electric field

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
    # nframes = 21
    plot_interval = plot_config["plot_interval"]

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
    dene_para = np.zeros([4, nptl])
    dene_perp = np.zeros([4, nptl])

    gamma0 = np.sqrt(1.0 + ptl["Ux"]**2 + ptl["Uy"]**2 + ptl["Uz"]**2)
    cond_exclude_cs = np.abs(ptl["dZ"]) > half_thickness_cs
    dgamma_min, dgamma_max = 0, 4
    dene_min, dene_max = 0, 4
    nbins = 128
    drange = [[dgamma_min, dgamma_max], [dene_min, dene_max]]
    ebins = np.logspace(dene_min, dene_max, nbins+1)
    ebins_mid = 0.5 * (ebins[1:] + ebins[:-1])
    vmin, vmax = 1E0, 1E2
    xyz = ['x', 'y', 'z']

    fdir = '../img/trans_relativistic/hist_dene_para_perp/' + pic_run + '/'
    if plot_config["exclude_cs"]:
        fdir += 'nocs/'
    else:
        fdir += 'all/'
    mkdir_p(fdir)

    for tframe in range(nframes):
        print("Time frame: %d" % tframe)
        tindex = tframe * pic_info.tracer_interval
        fname = tracer_dir + 'T.' + str(tindex) + '/electron_tracer_qtag_sorted.h5p'
        fh = h5py.File(fname, 'r')
        group = fh['Step#' + str(tindex)]
        ptl = {}
        for dset in group:
            dset = str(dset)
            ptl[str(dset)] = read_var(group, dset, nptl)
        fh.close()
        ux = ptl["Ux"]
        uy = ptl["Uy"]
        uz = ptl["Uz"]
        gamma = np.sqrt(1.0 + ux**2 + uy**2 + uz**2)
        dgamma = gamma - gamma0
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
        if tframe % plot_interval == 0 and tframe != 0:
            for ixyz in range(4):
                cond = np.logical_and(dene_para[ixyz, :] > 0, dene_perp[ixyz, :] > 0)
                if plot_config["exclude_cs"]:
                    cond = np.logical_and(cond, cond_exclude_cs)
                nptl_cond = len(dgamma[cond])
                hist, ubins_edges, ubins_edges = np.histogram2d(np.log10(dene_perp[ixyz, cond]),
                                                                np.log10(dene_para[ixyz, cond]),
                                                                bins=nbins, range=drange)
                fig = plt.figure(figsize=[6, 5])
                rect = [0.12, 0.12, 0.76, 0.8]
                ax = fig.add_axes(rect)
                p1 = ax.imshow(hist,
                               extent=[dgamma_min, dgamma_max, dene_min, dene_max],
                               norm = LogNorm(vmin=vmin, vmax=vmax),
                               cmap=plt.cm.viridis, aspect='auto',
                               origin='lower', interpolation='nearest')
                if ixyz < 3:
                    xlabel = r'$\log(' + 'W_{\parallel,' + xyz[ixyz] + '}' + ')$'
                    ylabel = r'$\log(' + 'W_{\perp,' + xyz[ixyz] + '}' + ')$'
                else:
                    xlabel = r'$\log(' + 'W_\parallel' + ')$'
                    ylabel = r'$\log(' + 'W_\perp' + ')$'
                ax.plot([0, 4], [0, 4], color='k', linewidth=1, linestyle='--',
                        label=xlabel + r'$=$' + ylabel)
                ax.plot([0, 4], [math.log10(sigma_e*0.5), math.log10(sigma_e*0.5)],
                        color='k', linewidth=1, linestyle='-',
                        label=ylabel + r'$=\sigma_e/2$')
                ax.plot([math.log10(sigma_e*0.5), math.log10(sigma_e*0.5)], [0, 4],
                        color='k', linewidth=1, linestyle=':',
                        label=xlabel + r'$=\sigma_e/2$')
                ax.legend(loc=2, prop={'size': 12}, ncol=1,
                          shadow=False, fancybox=False, frameon=False)
                ax.tick_params(bottom=True, top=True, left=True, right=True)
                ax.tick_params(axis='x', which='minor', direction='in')
                ax.tick_params(axis='x', which='major', direction='in')
                ax.tick_params(axis='y', which='minor', direction='in')
                ax.tick_params(axis='y', which='major', direction='in')
                ax.set_xlim([0, 4])
                ax.set_ylim([0, 4])
                ax.set_xlabel(xlabel, fontsize=16)
                ax.set_ylabel(ylabel, fontsize=16)
                ax.tick_params(labelsize=12)
                rect_cbar = np.copy(rect)
                rect_cbar[0] += rect[2] + 0.02
                rect_cbar[2] = 0.03
                cbar_ax = fig.add_axes(rect_cbar)
                cbar = fig.colorbar(p1, cax=cbar_ax, extend='both')
                cbar_ax.tick_params(bottom=False, top=False, left=False, right=True)
                cbar_ax.tick_params(axis='y', which='major', direction='out')
                cbar_ax.tick_params(axis='y', which='minor', direction='in', right=False)
                cbar_ax.set_title(r'$N_e$', fontsize=16)
                cbar.ax.tick_params(labelsize=12)
                twpe = math.ceil(tindex * pic_info.dtwpe / 0.1) * 0.1
                text1 = r'$t\omega_{pe}=' + ("{%0.0f}" % twpe) + '$'
                ax.set_title(text1, fontsize=16)
                fname = fdir + 'hist_dw_'
                if plot_config["exclude_cs"]:
                    fname += 'nocs_'
                if ixyz < 3:
                    fname += xyz[ixyz]
                else:
                    fname += 'tot'
                fname += '_' + species + '_' + str(tframe) + '.pdf'
                fig.savefig(fname)
                # plt.close()
                plt.show()
        dene_para /= dtwpe_tracer
        dene_perp /= dtwpe_tracer


def compare_wpara_wperp_four(plot_config, show_plot=True):
    """Compare energization due to parallel and perpendicular electric field

    4 panels for different conditions are plotted

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
    # nframes = 21
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
    dene_para = np.zeros([4, nptl])
    dene_perp = np.zeros([4, nptl])

    gamma0 = np.sqrt(1.0 + ptl["Ux"]**2 + ptl["Uy"]**2 + ptl["Uz"]**2)
    cond_exclude_cs = np.abs(ptl["dZ"]) > half_thickness_cs
    dgamma_min, dgamma_max = 0, 4
    dene_min, dene_max = 0, 4
    nbins = 128
    drange = [[dgamma_min, dgamma_max], [dene_min, dene_max]]
    ebins = np.logspace(dene_min, dene_max, nbins+1)
    ebins_mid = 0.5 * (ebins[1:] + ebins[:-1])
    vmin, vmax = 1E0, 1E2
    xyz = ['x', 'y', 'z']

    fdir = '../img/trans_relativistic/wpara_wperp_four/' + pic_run + '/'
    if plot_config["exclude_cs"]:
        fdir += 'nocs/'
    else:
        fdir += 'all/'
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
            rect0 = [0.07, 0.52, 0.41, 0.43]
            hgap, vgap = 0.02, 0.02
            istep = tindex // pic_info.tracer_interval
            if istep % plot_interval == 0 and istep != 0:
                for ixyz in range(4):
                    fig = plt.figure(figsize=[10.5, 10])
                    rect = np.copy(rect0)
                    for row, col in itertools.product(range(2), range(2)):
                        if row == 0 and col == 0:
                            cond = np.logical_and(dene_para[ixyz, :] < 0, dene_perp[ixyz, :] > 0)
                        elif row == 0 and col == 1:
                            cond = np.logical_and(dene_para[ixyz, :] > 0, dene_perp[ixyz, :] > 0)
                        elif row == 1 and col == 0:
                            cond = np.logical_and(dene_para[ixyz, :] < 0, dene_perp[ixyz, :] < 0)
                        elif row == 1 and col == 1:
                            cond = np.logical_and(dene_para[ixyz, :] > 0, dene_perp[ixyz, :] < 0)

                        if plot_config["exclude_cs"]:
                            cond = np.logical_and(cond, cond_exclude_cs)
                        nptl_cond = len(dgamma[cond])
                        hist, _, _ = np.histogram2d(np.log10(np.abs(dene_perp[ixyz, cond])),
                                                    np.log10(np.abs(dene_para[ixyz, cond])),
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
                        if ixyz < 3:
                            xlabel = (r'$\log(' + '|W_{\parallel,' + xyz[ixyz] + '}' +
                                      xsign + '|)$')
                            ylabel = (r'$\log(' + '|W_{\perp,' + xyz[ixyz] + '}' +
                                      ysign + '|)$')
                        else:
                            xlabel = (r'$\log(' + '|W_\parallel' + xsign + '|)$')
                            ylabel = (r'$\log(' + '|W_\perp' + ysign + '|)$')
                        ax.plot([0, 4], [0, 4], color='k', linewidth=1, linestyle='--')
                        ax.plot([0, 4], [math.log10(sigma_e*0.5), math.log10(sigma_e*0.5)],
                                color='k', linewidth=1, linestyle='-')
                        ax.plot([math.log10(sigma_e*0.5), math.log10(sigma_e*0.5)], [0, 4],
                                color='k', linewidth=1, linestyle='-')
                        xpos = 0.87 if col else 0.05
                        ypos = 0.35 if row else 0.64
                        ax.text(xpos, ypos, r'$\sigma_e/2$', color='k', fontsize=12,
                                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                                horizontalalignment='left', verticalalignment='center',
                                transform=ax.transAxes)
                        xpos = 0.63 if col else 0.33
                        ypos = 0.08 if row else 0.9
                        ax.text(xpos, ypos, r'$\sigma_e/2$', color='k', fontsize=12,
                                rotation=90,
                                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                                horizontalalignment='left', verticalalignment='center',
                                transform=ax.transAxes)
                        if row == 1:
                            ax.set_xlabel(xlabel, fontsize=16)
                        else:
                            ax.tick_params(axis='x', labelbottom=False)
                        if col == 0:
                            ax.set_ylabel(ylabel, fontsize=16)
                        else:
                            ax.tick_params(axis='y', labelleft=False)

                        if ixyz < 3:
                            xlabel = r'$' + '|W_{\parallel,' + xyz[ixyz] + '}' + '|$'
                            ylabel = r'$' + '|W_{\perp,' + xyz[ixyz] + '}' + '|$'
                        else:
                            xlabel = r'$' + '|W_\parallel' + '|$'
                            ylabel = r'$' + '|W_\perp' + '|$'
                        text1 = xlabel + r'$=$' + ylabel
                        angle = -45 if row == col else 45
                        xpos = 0.75 if col else 0.19
                        ypos = 0.21 if row else 0.77
                        ax.text(xpos, ypos, text1, color='k', fontsize=12, rotation=angle,
                                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                                horizontalalignment='left', verticalalignment='center',
                                transform=ax.transAxes)
                        ax.tick_params(bottom=True, top=True, left=True, right=True)
                        ax.tick_params(axis='x', which='minor', direction='in')
                        ax.tick_params(axis='x', which='major', direction='in')
                        ax.tick_params(axis='y', which='minor', direction='in')
                        ax.tick_params(axis='y', which='major', direction='in')
                        xlim = [0, 4] if col else [4, 0]
                        ax.set_xlim(xlim)
                        ylim = [4, 0] if row else [0, 4]
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
                        twpe = math.ceil(tindex * pic_info.dtwpe / 0.1) * 0.1
                        text1 = r'$t\omega_{pe}=' + ("{%0.0f}" % twpe) + '$'
                        fig.suptitle(text1, fontsize=16)
                    fname = fdir + 'wpara_wperp_4_'
                    if plot_config["exclude_cs"]:
                        fname += 'nocs_'
                    if ixyz < 3:
                        fname += xyz[ixyz]
                    else:
                        fname += 'tot'
                    fname += '_' + species + '_' + str(istep) + '.pdf'
                    fig.savefig(fname)
                    plt.close()
                    # plt.show()
            dene_para /= dtwpe_tracer
            dene_perp /= dtwpe_tracer
        fh.close()


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
    cond_exclude_cs = np.abs(ptl["dZ"]) > half_thickness_cs

    fdir = '../data/trans_relativistic/wpara_wperp/' + pic_run + '/'
    if plot_config["exclude_cs"]:
        fdir += 'nocs/'
    else:
        fdir += 'all/'
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
            if plot_config["exclude_cs"]:
                cond = np.logical_and(cond, cond_exclude_cs)
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
    pic_run_dir = plot_config["pic_run_dir"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    dtwpe_tracer = pic_info.dtwpe * pic_info.tracer_interval

    tracer_dir = pic_run_dir + 'tracer/tracer1/'
    nframes = len(os.listdir(tracer_dir))
    plot_interval = plot_config["plot_interval"]

    root_dir = '../data/trans_relativistic/wpara_wperp/' + pic_run + '/'
    fdir_all = root_dir + 'all/'
    fdir_nocs = root_dir + 'nocs/'

    tframes = range(0, nframes, plot_interval)
    ntf = len(tframes)
    ncross = np.zeros(ntf)
    number_larger_wperp = np.zeros(ntf)
    number_larger_wpara = np.zeros(ntf)
    wperp_avg = np.zeros(ntf)
    wpara_avg = np.zeros(ntf)
    ttracer = np.arange(ntf) * dtwpe_tracer * plot_interval

    fig1 = plt.figure(figsize=[7, 5])
    rect = [0.12, 0.16, 0.83, 0.8]
    ax1 = fig1.add_axes(rect)
    fig2 = plt.figure(figsize=[7, 5])
    ax2 = fig2.add_axes(rect)

    for fdir in [fdir_all, fdir_nocs]:
        for iframe, tframe in enumerate(tframes):
            print("Time frame: %d" % tframe)
            fname = fdir + 'wpara_cross_' + str(tframe) + '.dat'
            dene_para_cross = np.fromfile(fname).reshape([4, -1])
            fname = fdir + 'wperp_cross_' + str(tframe) + '.dat'
            dene_perp_cross = np.fromfile(fname).reshape([4, -1])
            fname = fdir + 'cross_half_sigma_' + str(tframe) + '.dat'
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

        COLORS = palettable.tableau.Tableau_10.mpl_colors
        ax1.set_prop_cycle('color', COLORS)
        lstyle = '-' if fdir == fdir_all else '--'
        plabel = '(w/o CS particles)' if fdir == fdir_nocs else ''
        label1 = r'$N_e(W_\parallel > W_\perp)$' + plabel
        label2 = r'$N_e(W_\parallel < W_\perp)$'+ plabel
        ax1.plot(ttracer, number_larger_wpara, linewidth=2,
                 linestyle=lstyle, label=label1)
        ax1.plot(ttracer, number_larger_wperp, linewidth=2,
                 linestyle=lstyle, label=label2)
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
        label1 = r'$\left<W_\parallel\right>$' + plabel
        label2 = r'$\left<W_\perp\right>$' + plabel
        ax2.plot(ttracer, wpara_avg, linewidth=2, linestyle=lstyle, label=label1)
        ax2.plot(ttracer, wperp_avg, linewidth=2, linestyle=lstyle, label=label2)
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

    fdir = '../img/trans_relativistic/wpara_wperp/' + pic_run + '/'
    mkdir_p(fdir)

    fname = fdir + 'num_wpara_wperp.pdf'
    fig1.savefig(fname)

    fname = fdir + 'ene_wpara_wperp.pdf'
    fig2.savefig(fname)

    # plt.show()
    plt.close('all')


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
    dtwpe_tracer = pic_info.dtwpe * pic_info.tracer_interval
    sigma_e = 5.524770e+02
    fname = pic_run_dir + '/info'
    with open(fname) as f:
        content = f.readlines()
    f.close()
    half_thickness_cs, _ = get_variable_value('L/de', 0, content)

    tracer_dir = pic_run_dir + 'tracer/tracer1/'
    nframes = len(os.listdir(tracer_dir))
    plot_interval = plot_config["plot_interval"]

    fdir = '../data/trans_relativistic/wpara_wperp/' + pic_run + '/'
    if plot_config["exclude_cs"]:
        fdir += 'nocs/'
    else:
        fdir += 'all/'

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
    tindex = tframe * pic_info.tracer_interval
    fname = tracer_dir + 'T.' + str(tindex) + '/electron_tracer_qtag_sorted.h5p'
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

    emin, emax = 1E-6, 1E4
    nbins = 100
    dloge = (math.log10(emax) - math.log10(emin)) / (nbins - 1)
    emin0 = 10**(math.log10(emin) - dloge)
    ebins = np.logspace(math.log10(emin0), math.log10(emax), nbins+1)
    ebins_mid = (ebins[:-1] + ebins[1:]) * 0.5
    debins = np.diff(ebins)

    spect1, _ = np.histogram(gamma1 - 1, bins=ebins)
    spect2, _ = np.histogram(gamma2 - 1, bins=ebins)
    spect_all, _ = np.histogram(gamma_all - 1, bins=ebins)

    fig = plt.figure(figsize=[7, 5])
    rect = [0.12, 0.16, 0.83, 0.8]
    ax = fig.add_axes(rect)
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    ax.set_prop_cycle('color', COLORS)
    ax.loglog(ebins_mid, spect1/debins, linewidth=2, label=r'$W_\parallel > W_\perp$')
    ax.loglog(ebins_mid, spect2/debins, linewidth=2, label=r'$W_\parallel < W_\perp$')
    ax.loglog(ebins_mid, spect_all/debins, linewidth=1, color='k',
              linestyle='--', label='All tracers')

    # fdir = pic_run_dir + "spectrum_combined/spectrum_e_69613.dat"
    # fdata = np.fromfile(fdir, dtype=np.float32)
    # emin, emax = 1E-6, 1E4
    # nbins = 1000
    # dloge = (math.log10(emax) - math.log10(emin)) / (nbins - 1)
    # emin0 = 10**(math.log10(emin) - dloge)
    # ebins = np.logspace(math.log10(emin0), math.log10(emax), nbins+1)
    # ebins_mid = (ebins[:-1] + ebins[1:]) * 0.5
    # debins = np.diff(ebins)
    # particle_select = 10000;
    # espect = fdata[3:] / debins / particle_select
    # ax.loglog(ebins_mid, espect, linewidth=2, label='All electrons')

    ax.legend(loc=1, prop={'size': 12}, ncol=1,
             shadow=False, fancybox=False, frameon=False)
    ax.set_xlim([1E0, 1E4])
    ax.set_ylim([1E-3, 1E5])
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlabel(r'$\gamma - 1$', fontsize=16)
    ax.set_ylabel(r'$f(\gamma - 1)$', fontsize=16)
    ax.tick_params(labelsize=12)
    fdir = '../img/trans_relativistic/wpara_wperp/' + pic_run + '/'
    if plot_config["exclude_cs"]:
        fdir += 'nocs/'
    else:
        fdir += 'all/'
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
    emin, emax = 1E-6, 1E4
    nbins = 1000
    dloge = (math.log10(emax) - math.log10(emin)) / (nbins - 1)
    emin0 = 10**(math.log10(emin) - dloge)
    ebins = np.logspace(math.log10(emin0), math.log10(emax), nbins+1)
    ebins_mid = (ebins[:-1] + ebins[1:]) * 0.5
    ebins_mid *= pmass
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
        sp_name = r"Test " + sname + r": tracers never passing $E>B$"
        ax.loglog(ebins_mid, espect_elb*norms[isp], linewidth=2, label=sp_name)
        ax.legend(loc=3, prop={'size': 12}, ncol=1,
                 shadow=False, fancybox=False, frameon=False)
        ax.set_xlim([1E0, 5E3])
        ax.set_ylim([1E1, 5E7])
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
        twpe = tframe * dtwpe_fields
        text1 = r'$t\omega_{pe}=' + str(twpe) + '$'
        ax.set_title(text1, fontsize=16)
        fdir = '../img/trans_relativistic/spect_species/' + pic_run + '/'
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
    dtwpe_tracer = pic_info.dtwpe * pic_info.tracer_interval

    tracer_dir = pic_run_dir + 'tracer/tracer1/'
    nframes = len(os.listdir(tracer_dir))
    plot_interval = plot_config["plot_interval"]

    root_dir = '../data/trans_relativistic/wpara_wperp/' + pic_run + '/'
    fdir_all = root_dir + 'all/'
    fdir_nocs = root_dir + 'nocs/'

    tframes = range(0, nframes, plot_interval)
    ntf = len(tframes)
    ncross = np.zeros(ntf)
    number_larger_wperp = np.zeros(ntf)
    number_larger_wpara = np.zeros(ntf)
    wperp_avg = np.zeros(ntf)
    wpara_avg = np.zeros(ntf)
    wperp_post_avg = np.zeros(ntf)
    wpara_post_avg = np.zeros(ntf)
    ttracer = np.arange(ntf) * dtwpe_tracer * plot_interval

    fig1 = plt.figure(figsize=[7, 5])
    rect = [0.12, 0.16, 0.83, 0.8]
    ax1 = fig1.add_axes(rect)

    for fdir in [fdir_all, fdir_nocs]:
        for iframe, tframe in enumerate(tframes):
            print("Time frame: %d" % tframe)
            fname = fdir + 'wpara_cross_' + str(tframe) + '.dat'
            dene_para_cross = np.fromfile(fname).reshape([4, -1])
            fname = fdir + 'wperp_cross_' + str(tframe) + '.dat'
            dene_perp_cross = np.fromfile(fname).reshape([4, -1])
            fname = fdir + 'wpara_' + str(tframe) + '.dat'
            dene_para = np.fromfile(fname).reshape([4, -1])
            fname = fdir + 'wperp_' + str(tframe) + '.dat'
            dene_perp = np.fromfile(fname).reshape([4, -1])
            fname = fdir + 'cross_half_sigma_' + str(tframe) + '.dat'
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

        COLORS = palettable.tableau.Tableau_10.mpl_colors
        ax1.set_prop_cycle('color', COLORS)
        lstyle = '-' if fdir == fdir_all else '--'
        plabel = '(w/o CS particles)' if fdir == fdir_nocs else ''
        label1 = r'$\left<W_\parallel\right>$' + plabel
        label2 = r'$\left<W_\perp\right>$' + plabel
        ax1.plot(ttracer, wpara_post_avg, linewidth=2,
                 linestyle=lstyle, label=label1)
        ax1.plot(ttracer, wperp_post_avg, linewidth=2,
                 linestyle=lstyle, label=label2)
        ax1.legend(loc=2, prop={'size': 12}, ncol=1,
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

    fdir = '../img/trans_relativistic/wpara_wperp_after_crossing/' + pic_run + '/'
    mkdir_p(fdir)

    fname = fdir + 'wpara_wperp_post.pdf'
    fig1.savefig(fname)

    # plt.show()
    plt.close()


def get_cmd_args():
    """Get command line arguments
    """
    default_pic_run = 'sigma03_bg01_2700de_Lde14_triggered_new'
    default_pic_run_dir = ('/net/scratch4/xiaocanli/reconnection/trans-relativistic/' +
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
    parser.add_argument('--plot_interval', action="store", default='20', type=int,
                        help='plot only for every plot_interval frames')
    parser.add_argument('--nsteps', action="store", default='1', type=int,
                        help='number of steps that are saved in the same file')
    parser.add_argument('--all_frames', action="store_true", default=False,
                        help='whether to analyze all frames')
    parser.add_argument('--show_plot', action="store_true", default=False,
                        help='whether to show plot')
    parser.add_argument('--plot_spect', action="store_true", default=False,
                        help='whether to plot particle energy spectrum')
    parser.add_argument('--check_density', action="store_true", default=False,
                        help='whether to check maximum density')
    parser.add_argument('--cumsum_ene', action="store_true", default=False,
                        help='whether to calculate cumulative energization')
    parser.add_argument('--exclude_cs', action="store_true", default=False,
                        help='whether to exclude particles in current sheet')
    parser.add_argument('--dw_para_perp', action="store_true", default=False,
                        help='whether to compare energization due to parallel' +
                        ' and perpendicular electric field')
    parser.add_argument('--wpara_wperp_four', action="store_true", default=False,
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
    return parser.parse_args()


def analysis_single_frames(plot_config, args):
    """Analysis for multiple time frames
    """
    tframe = args.tframe
    # test(plot_config)
    if args.plot_spect:
        if args.all_frames:
            plot_spectrum_multi(plot_config)
    elif args.check_density:
        check_density(plot_config)
    elif args.cumsum_ene:
        cumulative_energization(plot_config)
    elif args.dw_para_perp:
        compare_dw_para_perp(plot_config)
    elif args.wpara_wperp_four:
        compare_wpara_wperp_four(plot_config)
    elif args.wpara_wperp:
        calc_wpara_wperp(plot_config)
    elif args.plot_wpara_wperp:
        plot_wpara_wperp(plot_config)
    elif args.comp_spect:
        compare_spectrum(plot_config)
    elif args.spect_species:
        plot_spect_species(plot_config, args.show_plot)
    elif args.egain_post:
        egain_after_injection(plot_config, args.show_plot)


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


def test(plot_config):
    """Compare energization due to parallel and perpendicular electric field

    4 panels for different conditions are plotted

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
    # nframes = 21
    plot_interval = plot_config["plot_interval"]

    fname = tracer_dir + 'T.0/electron_tracer_qtag_sorted.h5p'
    with h5py.File(fname, 'r') as fh:
        group = fh['Step#0']
        dset = group['dX']
        nptl, = dset.shape
        ptl = {}
        for dset in group:
            dset = str(dset)
            ptl[str(dset)] = read_var(group, dset, nptl)

    # gamma0 = np.zeros(nptl)
    # dgamma = np.zeros(nptl)
    # dene_para = np.zeros([4, nptl])
    # dene_perp = np.zeros([4, nptl])

    # gamma0 = np.sqrt(1.0 + ptl["Ux"]**2 + ptl["Uy"]**2 + ptl["Uz"]**2)
    # cond_exclude_cs = np.abs(ptl["dZ"]) > half_thickness_cs
    # dgamma_min, dgamma_max = 0, 4
    # dene_min, dene_max = 0, 4
    # nbins = 128
    # drange = [[dgamma_min, dgamma_max], [dene_min, dene_max]]
    # ebins = np.logspace(dene_min, dene_max, nbins+1)
    # ebins_mid = 0.5 * (ebins[1:] + ebins[:-1])
    # vmin, vmax = 1E0, 1E2
    # xyz = ['x', 'y', 'z']

    # fdir = '../img/trans_relativistic/wpara_wperp_four/' + pic_run + '/'
    # if plot_config["exclude_cs"]:
    #     fdir += 'nocs/'
    # else:
    #     fdir += 'all/'
    # mkdir_p(fdir)

    nframes = 500
    ptl0 = np.zeros([nframes, 10])

    for tframe in range(nframes):
        print("Time frame: %d" % tframe)
        tindex = tframe * pic_info.tracer_interval
        fname = tracer_dir + 'T.' + str(tindex) + '/electron_tracer_qtag_sorted.h5p'
        fh = h5py.File(fname, 'r')
        group = fh['Step#' + str(tindex)]
        ptl = {}
        for dset in group:
            dset = str(dset)
            fdata = read_var(group, dset, nptl)
            ptl[str(dset)] = fdata[133555]
        fh.close()
        ptl0[tframe, 0] = ptl["Bx"]
        ptl0[tframe, 1] = ptl["By"]
        ptl0[tframe, 2] = ptl["Bz"]
        ptl0[tframe, 3] = ptl["Ex"]
        ptl0[tframe, 4] = ptl["Ey"]
        ptl0[tframe, 5] = ptl["Ez"]
        ptl0[tframe, 6] = ptl["Ux"]
        ptl0[tframe, 7] = ptl["Uy"]
        ptl0[tframe, 8] = ptl["Uz"]
        ptl0[tframe, 9] = ptl["q"]
    gamma = np.sqrt(1 + np.sum(ptl0[:, 6:9]**2, axis=1))
    vx = ptl0[:, 6] / gamma
    vy = ptl0[:, 7] / gamma
    vz = ptl0[:, 8] / gamma
    edotb = np.sum(ptl0[:, 0:3] * ptl0[:, 3:6], axis=1)
    ib2 = 1./np.sum(ptl0[:, 0:3]**2, axis=1)
    b2 = np.sum(ptl0[:, 0:3]**2, axis=1)

    eparax = edotb * ptl0[:, 0] * ib2
    eparay = edotb * ptl0[:, 1] * ib2
    eparaz = edotb * ptl0[:, 2] * ib2
    eperpx = ptl0[:, 3] - eparax
    eperpy = ptl0[:, 4] - eparay
    eperpz = ptl0[:, 5] - eparaz

    dene_para = -(vx * eparax + vy * eparay + vz * eparaz)
    dene_perp = -(vx * eperpx + vy * eperpy + vz * eperpz)

    dene_tot = np.cumsum(dene_para)*dtwpe_tracer + np.cumsum(dene_perp)*dtwpe_tracer

    plt.plot(gamma[0] + np.cumsum(dene_para)*dtwpe_tracer, color='r')
    plt.plot(gamma[0] + np.cumsum(dene_perp)*dtwpe_tracer, color='g')
    plt.plot(gamma[0] + np.cumsum(dene_para)*dtwpe_tracer + np.cumsum(dene_perp)*dtwpe_tracer, color='b')
    plt.plot(gamma, color='k')
    # plt.plot((gamma[0] + dene_tot)/(gamma), color='k')
    plt.show()

    # print(dene_para, dene_perp)

    # print(eparax**2 + eparay**2 + eparaz**2)
    # print(eperpx**2 + eperpy**2 + eperpz**2)
    # print(np.sum(ptl0[:, 3:6]**2, axis=1))
    # print(eparax, eperpx)
    # print(eparay, eperpy)
    # print(eparaz, eperpz)


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
    plot_config["exclude_cs"] = args.exclude_cs
    if args.multi_frames:
        analysis_multi_frames(plot_config, args)
    else:
        analysis_single_frames(plot_config, args)


if __name__ == "__main__":
    main()
