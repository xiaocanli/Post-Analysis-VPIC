#!/usr/bin/env python3
"""
Reconnection rate problem
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
from scipy import signal, integrate
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.ndimage.filters import median_filter, gaussian_filter

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


def plot_absj(plot_config, show_plot=True):
    """Plot current density
    """
    tframe = plot_config["tframe"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    fields_interval = pic_info.fields_interval
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

    xgrid = np.linspace(xmin, xmax, pic_info.nx)
    zgrid = np.linspace(zmin, zmax, pic_info.nz)

    absj = np.squeeze(np.sqrt((je["jx"] + ji["jx"])**2 +
                              (je["jy"] + ji["jy"])**2 +
                              (je["jz"] + ji["jz"])**2))
    len0 = 10
    fig = plt.figure(figsize=[len0, len0*lz_de/lx_de])
    rect = [0.12, 0.14, 0.78, 0.78]
    ax = fig.add_axes(rect)
    im1 = ax.imshow(absj.T,
                    extent=[xmin, xmax, zmin, zmax],
                    vmin=0, vmax=0.05,
                    cmap=plt.cm.viridis, aspect='auto',
                    origin='lower', interpolation='bicubic')
    # Magnetic field lines
    if "open" in pic_run or "test" in pic_run:
        fname = (pic_run_dir + "field_hdf5/T." + str(tindex) +
                 "/fields_" + str(tindex) + ".h5")
        bvec = {}
        with h5py.File(fname, 'r') as fh:
            group = fh["Timestep_" + str(tindex)]
            for var in ["cbx", "cbz"]:
                dset = group[var]
                bvec[var] = np.zeros(dset.shape, dtype=dset.dtype)
                dset.read_direct(bvec[var])
        xmesh, zmesh = np.meshgrid(xgrid, zgrid)
        ax.streamplot(xmesh, zmesh, np.squeeze(bvec["cbx"]).T,
                      np.squeeze(bvec["cbz"]).T, color='w',
                      linewidth=0.5)
    else:
        pass
        # kwargs = {"current_time": tframe,
        #           "xl": 0, "xr": pic_info.lx_di,
        #           "zb": -pic_info.lz_di, "zt": pic_info.lz_di}
        # fname = pic_run_dir + "data/Ay.gda"
        # x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
        # ax.contour(xgrid, zgrid, Ay, colors='k', linewidths=0.5,
        #            levels=np.linspace(np.min(Ay), np.max(Ay), 20))

    fdir = '../data/rate_problem/exhaust_boundary/' + pic_run + '/'
    fname = fdir + 'xz_top_' + str(tframe) + '.dat'
    xz = np.fromfile(fname).reshape([2, -1])
    xlist_top = xz[0, :]
    zlist_top = xz[1, :] + zmin
    fname = fdir + 'xz_bot_' + str(tframe) + '.dat'
    xz = np.fromfile(fname).reshape([2, -1])
    xlist_bot = xz[0, :]
    zlist_bot = xz[1, :] + zmin
    ax.plot(xlist_top, zlist_top, linewidth=2, color=COLORS[0])
    ax.plot(xlist_bot, zlist_bot, linewidth=2, color=COLORS[0])

    # fdir = '../data/rate_problem/rrate_bflux/' + pic_run + '/'
    # fname = fdir + 'xz_close_' + str(tframe) + '.dat'
    # xz = np.fromfile(fname).reshape([2, -1])
    # xlist = xz[0, :]
    # zlist = xz[1, :] + zmin
    # ax.plot(xlist, zlist, linewidth=1, linestyle='--', color=COLORS[1])

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([zmin, zmax])
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
    rect_cbar[1] += rect[3] * 0.25
    rect_cbar[3] = rect[3] * 0.5
    cbar_ax = fig.add_axes(rect_cbar)
    cbar = fig.colorbar(im1, cax=cbar_ax, extend='max')
    cbar_ax.tick_params(bottom=False, top=False, left=False, right=True)
    cbar_ax.tick_params(axis='y', which='major', direction='out')
    cbar_ax.tick_params(axis='y', which='minor', direction='in', right=False)
    cbar_ax.set_title(r'$|\boldsymbol{J}|$', fontsize=20)
    cbar.ax.tick_params(labelsize=12)
    twpe = math.ceil(tindex * pic_info.dtwpe / 0.1) * 0.1
    text1 = r'$t\omega_{pe}=' + ("{%0.0f}" % twpe) + '$'
    fig.suptitle(text1, fontsize=20)
    img_dir = '../img/rate_problem/absj/' + pic_run + '/'
    mkdir_p(img_dir)
    fname = img_dir + "absj_" + str(tframe) + ".jpg"
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
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    fields_interval = pic_info.fields_interval
    tindex = fields_interval * tframe
    smime = math.sqrt(pic_info.mime)
    lx_de = pic_info.lx_di * smime
    lz_de = pic_info.lz_di * smime
    xmin, xmax = 0, lx_de
    zmin, zmax = -0.5 * lz_de, 0.5 * lz_de

    fname = (pic_run_dir + "field_hdf5/T." + str(tindex) +
             "/fields_" + str(tindex) + ".h5")
    bvec = {}
    with h5py.File(fname, 'r') as fh:
        group = fh["Timestep_" + str(tindex)]
        for var in ["cbx", "cby", "cbz"]:
            dset = group[var]
            bvec[var] = np.zeros(dset.shape, dtype=dset.dtype)
            dset.read_direct(bvec[var])

    bvec["absb"] = np.squeeze(np.sqrt(bvec["cbx"]**2 +
                                      bvec["cby"]**2 +
                                      bvec["cbz"]**2))
    fig = plt.figure(figsize=[5, 8])
    rect = [0.14, 0.75, 0.73, 0.2]
    hgap, vgap = 0.02, 0.02
    var_names = [r"$B_x$", r"$B_y$", r"$B_z$", r"$|\boldsymbol{B}|$"]
    colors = ['w', 'k', 'k', 'k']
    for ivar, var in enumerate(bvec):
        ax = fig.add_axes(rect)
        cmap = plt.cm.seismic if "cb" in var else plt.cm.plasma
        dmax = 0.35
        dmin = -dmax if "cb" in var else 0
        im1 = ax.imshow(np.squeeze(bvec[var]).T,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=dmin, vmax=dmax,
                        cmap=cmap, aspect='auto',
                        origin='lower', interpolation='none')
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')
        if ivar == 3:
            ax.set_xlabel(r'$x/d_e$', fontsize=16)
        else:
            ax.tick_params(axis='x', labelbottom=False)
        ax.set_ylabel(r'$z/d_e$', fontsize=16)
        ax.tick_params(labelsize=12)
        ax.text(0.02, 0.85, var_names[ivar], color=colors[ivar], fontsize=16,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
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
    twpe = math.ceil(tindex * pic_info.dtwpe / 0.1) * 0.1
    text1 = r'$t\omega_{pe}=' + ("{%0.0f}" % twpe) + '$'
    fig.suptitle(text1, fontsize=20)
    img_dir = '../img/rate_problem/bfield/' + pic_run + '/'
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
    print("Time Frame: %d" % tframe)
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    fields_interval = pic_info.fields_interval
    tindex = fields_interval * tframe
    smime = math.sqrt(pic_info.mime)
    lx_de = pic_info.lx_di * smime
    lz_de = pic_info.lz_di * smime
    xmin, xmax = 0, lx_de
    zmin, zmax = -0.5 * lz_de, 0.5 * lz_de

    fname = (pic_run_dir + "field_hdf5/T." + str(tindex) +
             "/fields_" + str(tindex) + ".h5")
    evec = {}
    with h5py.File(fname, 'r') as fh:
        group = fh["Timestep_" + str(tindex)]
        for var in ["ex", "ey", "ez"]:
            dset = group[var]
            evec[var] = np.zeros(dset.shape, dtype=dset.dtype)
            dset.read_direct(evec[var])

    evec["abse"] = np.sqrt(evec["ex"]**2 + evec["ey"]**2 + evec["ez"]**2)
    fig = plt.figure(figsize=[5, 8])
    rect = [0.14, 0.75, 0.68, 0.2]
    hgap, vgap = 0.02, 0.02
    for ivar, var in enumerate(evec):
        ax = fig.add_axes(rect)
        cmap = plt.cm.plasma if var == "abse" else plt.cm.seismic
        if "Tbe_Te_01" in pic_run:
            dmax = 0.015
        elif "Tbe_Te_1" in pic_run:
            if "Tbe_Te_10" in pic_run:
                dmax = 0.04
            else:
                dmax = 0.015
        else:
            dmax = 0.04
        dmin = 0 if var == "abse" else -dmax
        fdata = gaussian_filter(np.squeeze(evec[var]).T, sigma=3)
        im1 = ax.imshow(fdata,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=dmin, vmax=dmax,
                        cmap=cmap, aspect='auto',
                        origin='lower', interpolation='none')
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')
        if ivar == 3:
            ax.set_xlabel(r'$x/d_e$', fontsize=16)
        else:
            ax.tick_params(axis='x', labelbottom=False)
        ax.set_ylabel(r'$z/d_e$', fontsize=16)
        ax.tick_params(labelsize=12)
        rect_cbar = np.copy(rect)
        if var == "abse":
            rect_cbar[0] += rect[2] + 0.01
            rect_cbar[2] = 0.02
            cbar_ax = fig.add_axes(rect_cbar)
            cbar = fig.colorbar(im1, cax=cbar_ax, extend="max")
        elif var == "ey":
            rect_cbar[0] += rect[2] + 0.01
            rect_cbar[2] = 0.02
            rect_cbar[1] -= rect[3] * 0.5 + vgap
            rect_cbar[3] = (rect[3] + vgap) * 2
            cbar_ax = fig.add_axes(rect_cbar)
            cbar = fig.colorbar(im1, cax=cbar_ax, extend="both")
        if var in ["abse", "ey"]:
            cbar_ax.tick_params(bottom=False, top=False, left=False, right=True)
            cbar_ax.tick_params(axis='y', which='major', direction='out')
            cbar_ax.tick_params(axis='y', which='minor', direction='in', right=False)
            cbar.ax.tick_params(labelsize=12)
        rect[1] -= rect[3] + vgap
    twpe = math.ceil(tindex * pic_info.dtwpe / 0.1) * 0.1
    text1 = r'$t\omega_{pe}=' + ("{%0.0f}" % twpe) + '$'
    fig.suptitle(text1, fontsize=20)
    img_dir = '../img/rate_problem/efield/' + pic_run + '/'
    mkdir_p(img_dir)
    fname = img_dir + "efield_" + str(tframe) + ".jpg"
    fig.savefig(fname, dpi=200)

    if show_plot:
        plt.show()
    else:
        plt.close()

    plt.plot(evec["ey"][:, 0, 0])
    plt.show()


def plot_pressure_tensor(plot_config, show_plot=True):
    """Plot pressure tensor
    """
    tframe = plot_config["tframe"]
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    fields_interval = pic_info.fields_interval
    tindex = fields_interval * tframe
    smime = math.sqrt(pic_info.mime)
    lx_de = pic_info.lx_di * smime
    lz_de = pic_info.lz_di * smime
    xmin, xmax = 0, lx_de
    zmin, zmax = -0.5 * lz_de, 0.5 * lz_de
    xgrid = np.linspace(xmin, xmax, pic_info.nx)
    zgrid = np.linspace(zmin, zmax, pic_info.nz)
    vpic_info = get_vpic_info(pic_run_dir)
    vthe = vpic_info["vtheb/c"]
    n0 = vpic_info["n0"]
    pback = n0 * vthe**2
    if "Tbe_Te_01" in pic_run:
        pmin, pmax = 0.5, 20.0
    elif "Tbe_Te_1" in pic_run:
        pmin, pmax = 0.5, 2.0
        if "Tbe_Te_10" in pic_run:
            pmin, pmax = 0.9, 1.1
    else:
        pmin = pback * 0.1
        pmax = pback * 10.0
    if species in ["e", "electron"]:
        sname = "electron"
        pmass = 1.0
    else:
        sname = "ion"
        pmass = pic_info.mime

    fname = (pic_run_dir + "hydro_hdf5/T." + str(tindex) +
             "/hydro_" + sname + "_" + str(tindex) + ".h5")
    hydro = {}
    with h5py.File(fname, 'r') as fh:
        group = fh["Timestep_" + str(tindex)]
        for var in group:
            dset = group[var]
            hydro[var] = np.zeros(dset.shape, dtype=dset.dtype)
            dset.read_direct(hydro[var])

    irho = 1.0 / hydro["rho"]
    nrho = np.abs(hydro["rho"])
    vx = hydro["jx"] * irho
    vy = hydro["jy"] * irho
    vz = hydro["jz"] * irho
    sigma = 9
    hydro["txx"] = gaussian_filter(np.squeeze(hydro["txx"] - vx * hydro["px"]), sigma=sigma)
    hydro["tyy"] = gaussian_filter(np.squeeze(hydro["tyy"] - vy * hydro["py"]), sigma=sigma)
    hydro["tzz"] = gaussian_filter(np.squeeze(hydro["tzz"] - vz * hydro["pz"]), sigma=sigma)
    hydro["tyx"] = gaussian_filter(np.squeeze(hydro["txy"] - vx * hydro["py"]), sigma=sigma)
    hydro["txz"] = gaussian_filter(np.squeeze(hydro["tzx"] - vz * hydro["px"]), sigma=sigma)
    hydro["tzy"] = gaussian_filter(np.squeeze(hydro["tyz"] - vy * hydro["pz"]), sigma=sigma)
    hydro["txy"] = gaussian_filter(np.squeeze(hydro["txy"] - vy * hydro["px"]), sigma=sigma)
    hydro["tyz"] = gaussian_filter(np.squeeze(hydro["tyz"] - vz * hydro["py"]), sigma=sigma)
    hydro["tzx"] = gaussian_filter(np.squeeze(hydro["tzx"] - vx * hydro["pz"]), sigma=sigma)

    pvars = ["txx", "tyy", "tzz", "txy", "txz", "tyz", "tyx", "tzx", "tzy"]
    pnames = [r"$P_{xx}$", r"$P_{yy}$", r"$P_{zz}$",
              r"$P_{xy}$", r"$P_{xz}$", r"$P_{yz}$",
              r"$P_{yx}$", r"$P_{zx}$", r"$P_{zy}$"]
    fig = plt.figure(figsize=[12, 5])
    rect0 = [0.07, 0.65, 0.27, 0.25]
    hgap, vgap = 0.01, 0.02
    nrow = ncol = 3
    for ivar, var in enumerate(pvars):
        irow = ivar // ncol
        icol = ivar % ncol
        if icol == 0:
            rect = np.copy(rect0)
            rect[1] -= (rect[3] + hgap) * irow
        ax = fig.add_axes(rect)
        if irow == 0:
            dmin, dmax = pback*pmin, pback*pmax
            cmap = plt.cm.plasma
        else:
            if species in ["e", "electron"]:
                dmin, dmax = -pback*0.2, pback*0.2
            else:
                dmin, dmax = -pback, pback
            cmap = plt.cm.seismic
        im1 = ax.imshow(hydro[var].T,
                        extent=[xmin, xmax, zmin, zmax],
                        # norm = LogNorm(vmin=dmin, vmax=dmax),
                        vmin=dmin, vmax=dmax,
                        cmap=cmap, aspect='auto',
                        origin='lower', interpolation='bicubic')
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')
        if irow == 2:
            ax.set_xlabel(r'$x/d_e$', fontsize=16)
        else:
            ax.tick_params(axis='x', labelbottom=False)
        if icol == 0:
            ax.set_ylabel(r'$z/d_e$', fontsize=16)
        else:
            ax.tick_params(axis='y', labelleft=False)
        ax.tick_params(labelsize=12)
        if icol == 2:
            rect_cbar = np.copy(rect)
            rect_cbar[0] += rect[2] + 0.01
            rect_cbar[2] = 0.01
            cbar_ax = fig.add_axes(rect_cbar)
            cbar = fig.colorbar(im1, cax=cbar_ax, extend='both')
            cbar_ax.tick_params(bottom=False, top=False, left=False, right=True)
            cbar_ax.tick_params(axis='y', which='major', direction='out')
            cbar_ax.tick_params(axis='y', which='minor', direction='in', right=False)
            cbar.ax.tick_params(labelsize=12)
        tcolor = 'w' if irow == 0 else 'k'
        ax.text(0.02, 0.85, pnames[ivar], color=tcolor, fontsize=16,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)
        rect[0] += rect[2] + hgap
    twpe = math.ceil(tindex * pic_info.dtwpe / 0.1) * 0.1
    text1 = r'$t\omega_{pe}=' + ("{%0.0f}" % twpe) + '$'
    fig.suptitle(text1, fontsize=20)
    img_dir = '../img/rate_problem/pressure_tensor/' + pic_run + '/'
    mkdir_p(img_dir)
    fname = img_dir + "ptensor_" + species + "_" + str(tframe) + ".jpg"
    fig.savefig(fname, dpi=200)

    lx_de = pic_info.lx_di * smime
    lz_de = pic_info.lz_di * smime
    xmin, xmax = 0, lx_de
    zmin, zmax = -0.5 * lz_de, 0.5 * lz_de
    nx, nz = pic_info.nx, pic_info.nz
    xgrid = np.linspace(xmin, xmax, nx)
    zgrid = np.linspace(zmin, zmax, nz)
    dx_de = lx_de / nx
    dz_de = lz_de / nz
    if "open" in pic_run:
        fdir = '../data/rate_problem/exhaust_boundary/' + pic_run + '/'
        fname = fdir + 'xz_top_' + str(tframe) + '.dat'
        xz = np.fromfile(fname).reshape([2, -1])
        xlist_top = xz[0, :]
        zlist_top = xz[1, :] + zmin
        fname = fdir + 'xz_bot_' + str(tframe) + '.dat'
        xz = np.fromfile(fname).reshape([2, -1])
        xlist_bot = xz[0, :]
        zlist_bot = xz[1, :] + zmin
        x0 = 0.5 * (xlist_bot[np.argmax(zlist_bot)] +
                    xlist_top[np.argmin(zlist_top)])
        ix_xpoint = int(x0 / dx_de)
    else:
        ix_xpoint = nx // 2

    # Vertical and horizontal cuts
    fig1 = plt.figure(figsize=[7, 5])
    rect = [0.12, 0.15, 0.8, 0.8]
    ax1 = fig1.add_axes(rect)
    ax1.set_prop_cycle('color', COLORS)
    fig2 = plt.figure(figsize=[7, 5])
    ax2 = fig2.add_axes(rect)
    ax2.set_prop_cycle('color', COLORS)
    nx, nz = pic_info.nx, pic_info.nz
    ng = 5
    kernel = np.ones(ng) / float(ng)
    for ivar, var in enumerate(["txx", "tyy", "tzz"]):
        fdata_x = hydro[var][:, nz//2]
        fdata_z = hydro[var][ix_xpoint, :]
        ax1.plot(xgrid, fdata_x, label=pnames[ivar])
        ax2.plot(zgrid, fdata_z, label=pnames[ivar])
    ax1.tick_params(bottom=True, top=True, left=True, right=True)
    ax1.tick_params(axis='x', which='minor', direction='in')
    ax1.tick_params(axis='x', which='major', direction='in')
    ax1.tick_params(axis='y', which='minor', direction='in')
    ax1.tick_params(axis='y', which='major', direction='in')
    ax2.tick_params(bottom=True, top=True, left=True, right=True)
    ax2.tick_params(axis='x', which='minor', direction='in')
    ax2.tick_params(axis='x', which='major', direction='in')
    ax2.tick_params(axis='y', which='minor', direction='in')
    ax2.tick_params(axis='y', which='major', direction='in')
    ax1.set_xlim([xmin, xmax])
    ax2.set_xlim([zmin, zmax])
    ax1.tick_params(labelsize=12)
    ax2.tick_params(labelsize=12)
    ax1.set_xlabel(r'$x/d_e$', fontsize=16)
    ax2.set_xlabel(r'$z/d_e$', fontsize=16)
    ax1.legend(loc=4, prop={'size': 16}, ncol=1,
               shadow=False, fancybox=False, frameon=False)
    ax2.legend(loc=1, prop={'size': 16}, ncol=1,
               shadow=False, fancybox=False, frameon=False)
    fname = img_dir + "ptensor_hcuts_" + species + "_" + str(tframe) + ".pdf"
    fig1.savefig(fname)
    fname = img_dir + "ptensor_vcuts_" + species + "_" + str(tframe) + ".pdf"
    fig2.savefig(fname)

    # Parallel and perpendicular pressure
    fname = (pic_run_dir + "field_hdf5/T." + str(tindex) +
             "/fields_" + str(tindex) + ".h5")
    bvec = {}
    with h5py.File(fname, 'r') as fh:
        group = fh["Timestep_" + str(tindex)]
        for var in ["cbx", "cby", "cbz"]:
            dset = group[var]
            bvec[var] = np.zeros(dset.shape, dtype=dset.dtype)
            dset.read_direct(bvec[var])

    bvec["cbx"] = gaussian_filter(np.squeeze(bvec["cbx"]), sigma=sigma)
    bvec["cby"] = gaussian_filter(np.squeeze(bvec["cby"]), sigma=sigma)
    bvec["cbz"] = gaussian_filter(np.squeeze(bvec["cbz"]), sigma=sigma)
    absB = np.sqrt(bvec["cbx"]**2 + bvec["cby"]**2 + bvec["cbz"]**2)

    ppara = (hydro["txx"] * bvec["cbx"]**2 +
             hydro["tyy"] * bvec["cby"]**2 +
             hydro["tzz"] * bvec["cbz"]**2 +
             (hydro["txy"] + hydro["tyx"]) * bvec["cbx"] * bvec["cby"] +
             (hydro["txz"] + hydro["tzx"]) * bvec["cbx"] * bvec["cbz"] +
             (hydro["tyz"] + hydro["tzy"]) * bvec["cby"] * bvec["cbz"])
    ppara /= absB * absB
    pperp = 0.5 * (hydro["txx"] + hydro["tyy"] + hydro["tzz"] - ppara)

    fig = plt.figure(figsize=[7, 5])
    rect = [0.12, 0.55, 0.76, 0.4]
    hgap, vgap = 0.02, 0.02
    ax = fig.add_axes(rect)
    dmin, dmax = pback*pmin, pback*pmax
    cmap = plt.cm.plasma
    im1 = ax.imshow(ppara.T,
                    extent=[xmin, xmax, zmin, zmax],
                    vmin=dmin, vmax=dmax,
                    cmap=cmap, aspect='auto',
                    origin='lower', interpolation='bicubic')
    ax.tick_params(axis='x', labelbottom=False)
    ax.set_ylabel(r'$z/d_e$', fontsize=16)
    ax.tick_params(labelsize=12)
    ax.text(0.02, 0.85, r"$P_\parallel$", color='w', fontsize=16,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)

    rect[1] -= rect[3] + vgap
    ax = fig.add_axes(rect)
    dmin, dmax = pback*pmin, pback*pmax
    cmap = plt.cm.plasma
    im1 = ax.imshow(pperp.T,
                    extent=[xmin, xmax, zmin, zmax],
                    vmin=dmin, vmax=dmax,
                    cmap=cmap, aspect='auto',
                    origin='lower', interpolation='bicubic')
    ax.set_xlabel(r'$x/d_e$', fontsize=16)
    ax.set_ylabel(r'$z/d_e$', fontsize=16)
    ax.tick_params(labelsize=12)
    ax.text(0.02, 0.85, r"$P_\perp$", color='w', fontsize=16,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)

    rect_cbar = np.copy(rect)
    rect_cbar[0] += rect[2] + 0.01
    rect_cbar[2] = 0.02
    rect_cbar[3] = rect[3] * 2 + vgap
    cbar_ax = fig.add_axes(rect_cbar)
    cbar = fig.colorbar(im1, cax=cbar_ax, extend='both')
    cbar_ax.tick_params(bottom=False, top=False, left=False, right=True)
    cbar_ax.tick_params(axis='y', which='major', direction='out')
    cbar_ax.tick_params(axis='y', which='minor', direction='in', right=False)
    cbar.ax.tick_params(labelsize=12)
    fname = img_dir + "ppara_pperp_" + species + "_" + str(tframe) + ".jpg"
    fig.savefig(fname, dpi=200)

    # Vertical and horizontal cuts
    fig1 = plt.figure(figsize=[7, 5])
    rect = [0.12, 0.15, 0.8, 0.8]
    ax1 = fig1.add_axes(rect)
    ax1.set_prop_cycle('color', COLORS)
    fig2 = plt.figure(figsize=[7, 5])
    ax2 = fig2.add_axes(rect)
    ax2.set_prop_cycle('color', COLORS)
    nx, nz = pic_info.nx, pic_info.nz
    ng = 5
    kernel = np.ones(ng) / float(ng)
    fdata_x = ppara[:, nz//2]
    fdata_z = ppara[nx//2, :]
    ax1.plot(xgrid, fdata_x, label=r"$P_\parallel$")
    ax2.plot(zgrid, fdata_z, label=r"$P_\parallel$")
    fdata_x = pperp[:, nz//2]
    fdata_z = pperp[nx//2, :]
    ax1.plot(xgrid, fdata_x, label=r"$P_\perp$")
    ax2.plot(zgrid, fdata_z, label=r"$P_\perp$")
    ax1.tick_params(bottom=True, top=True, left=True, right=True)
    ax1.tick_params(axis='x', which='minor', direction='in')
    ax1.tick_params(axis='x', which='major', direction='in')
    ax1.tick_params(axis='y', which='minor', direction='in')
    ax1.tick_params(axis='y', which='major', direction='in')
    ax2.tick_params(bottom=True, top=True, left=True, right=True)
    ax2.tick_params(axis='x', which='minor', direction='in')
    ax2.tick_params(axis='x', which='major', direction='in')
    ax2.tick_params(axis='y', which='minor', direction='in')
    ax2.tick_params(axis='y', which='major', direction='in')
    ax1.set_xlim([xmin, xmax])
    ax2.set_xlim([zmin, zmax])
    ax1.tick_params(labelsize=12)
    ax2.tick_params(labelsize=12)
    ax1.set_xlabel(r'$x/d_e$', fontsize=16)
    ax2.set_xlabel(r'$z/d_e$', fontsize=16)
    ax1.legend(loc=4, prop={'size': 16}, ncol=1,
               shadow=False, fancybox=False, frameon=False)
    ax2.legend(loc=1, prop={'size': 16}, ncol=1,
               shadow=False, fancybox=False, frameon=False)
    fname = img_dir + "ppara_pperp_hcuts_" + species + "_" + str(tframe) + ".pdf"
    fig1.savefig(fname)
    fname = img_dir + "ppara_pperp_vcuts_" + species + "_" + str(tframe) + ".pdf"
    fig2.savefig(fname)

    if show_plot:
        plt.show()
    else:
        plt.close("all")


def plot_reconnection_rate_2d(plot_config):
    """Plot reconnection rate for the 2D simulation

    Args:
        run_dir: the run root directory
        run_name: PIC run name
    """
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    ntf = pic_info.ntf
    phi = np.zeros(ntf)
    fname = pic_run_dir + 'data/Ay.gda'
    for tframe in range(ntf):
        kwargs = {"current_time": tframe,
                  "xl": 0, "xr": pic_info.lx_di,
                  "zb": -pic_info.lz_di*0.5, "zt": pic_info.lz_di*0.5}
        x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
        nz, = z.shape
        max_ay = np.max(Ay[nz // 2 - 1:nz // 2 + 2, :])
        min_ay = np.min(Ay[nz // 2 - 1:nz // 2 + 2, :])
        # max_ay = np.max(Ay[nz // 2, :])
        # min_ay = np.min(Ay[nz // 2, :])
        phi[tframe] = max_ay - min_ay
    nk = 3
    # phi = signal.medfilt(phi, kernel_size=nk)
    dtwpe = pic_info.dtwpe
    dtwce = pic_info.dtwce
    dtwci = pic_info.dtwci
    mime = pic_info.mime
    dtf_wpe = pic_info.dt_fields * dtwpe / dtwci
    reconnection_rate = np.gradient(phi) / dtf_wpe
    b0 = pic_info.b0
    va = dtwce * math.sqrt(1.0 / mime) / dtwpe
    reconnection_rate /= b0 * va
    # reconnection_rate[-1] = reconnection_rate[-2]
    tfields = pic_info.tfields

    COLORS = palettable.tableau.Tableau_10.mpl_colors
    fig = plt.figure(figsize=[7, 5])
    rect = [0.12, 0.12, 0.83, 0.8]
    ax = fig.add_axes(rect)
    ax.plot(tfields, reconnection_rate, linestyle='-',
            marker='o', color=COLORS[0])
    ax.set_xlabel(r'$t\Omega_{ci}$', fontsize=16)
    ax.set_ylabel(r'$R$', fontsize=16)
    ax.tick_params(bottom=True, top=False, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top=True)
    ax.tick_params(axis='x', which='major', direction='in', top=True)
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.tick_params(labelsize=12)

    fdir = '../data/rate_problem/rrate/'
    mkdir_p(fdir)
    fname = fdir + 'rrate_' + pic_run + '.dat'
    reconnection_rate.tofile(fname)

    fdir = '../img/rate_problem/rrate/'
    mkdir_p(fdir)
    fname = fdir + 'rrate_' + pic_run + '.pdf'
    fig.savefig(fname)

    plt.show()


def calc_rrate_vin(plot_config, show_plot=True):
    """Calculate reconnection rate for the inflow velocity
    """
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    ntf = pic_info.ntf
    fields_interval = pic_info.fields_interval
    dtwpe = pic_info.dtwpe
    dtwce = pic_info.dtwce
    va = dtwce * math.sqrt(1.0 / pic_info.mime) / dtwpe
    print("Alfven speed/c: %f" % va)

    rrate_vin = np.zeros(ntf)

    for tframe in range(ntf):
        print("Time frame %d of %d" % (tframe, ntf))
        tindex = fields_interval * tframe
        rho_vel = {}
        for species in ["e", "i"]:
            sname = "electron" if species == 'e' else "ion"
            fname = (pic_run_dir + "hydro_hdf5/T." + str(tindex) +
                     "/hydro_" + sname + "_" + str(tindex) + ".h5")
            hydro = {}
            with h5py.File(fname, 'r') as fh:
                group = fh["Timestep_" + str(tindex)]
                for var in ["rho", "jz"]:
                    dset = group[var]
                    hydro[var] = np.zeros(dset.shape, dtype=dset.dtype)
                    dset.read_direct(hydro[var])
            irho = 1.0 / hydro["rho"]
            var = "v" + species
            rho_vel[var+"z"] = np.squeeze(hydro["jz"] * irho)
            rho_vel["n"+species] = np.abs(np.squeeze(hydro["rho"]))
        rho = rho_vel["ne"] + rho_vel["ni"] + pic_info.mime
        irho = 1.0 / rho
        # vsx = (rho_vel["ne"] * rho_vel["vex"] +
        #        rho_vel["ni"] * rho_vel["vix"] * pic_info.mime) * irho
        # vsy = (rho_vel["ne"] * rho_vel["vey"] +
        #        rho_vel["ni"] * rho_vel["viy"] * pic_info.mime) * irho
        vsz = (rho_vel["ne"] * rho_vel["vez"] +
               rho_vel["ni"] * rho_vel["viz"] * pic_info.mime) * irho
        nx, nz = vsz.shape
        sigma = 9
        # vsx = gaussian_filter(vsx, sigma=sigma)
        # vsy = gaussian_filter(vsy, sigma=sigma)
        vsz = gaussian_filter(vsz, sigma=sigma)
        rrate_vin[tframe] = np.max(vsz[nx//2, :]) / va
        # plt.plot(vsz[nx//2, :])
        # plt.show()

    # fdir = '../data/rate_problem/rrate/'
    # fname = fdir + 'rrate_' + pic_run + '.dat'
    # reconnection_rate = np.fromfile(fname)
    # plt.plot(reconnection_rate)

    # plt.plot(rrate_vin)

    plt.show()


def get_bfield_pressure(plot_config):
    """Get magnetic field and pressure tensor
    """
    tframe = plot_config["tframe"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    fields_interval = pic_info.fields_interval
    tindex = fields_interval * tframe
    vpic_info = get_vpic_info(pic_run_dir)
    vthe = vpic_info["vtheb/c"]
    n0 = vpic_info["n0"]
    pnorm = n0 * vthe**2

    vecb_pre = {}

    # Magnetic field
    fname = (pic_run_dir + "field_hdf5/T." + str(tindex) +
             "/fields_" + str(tindex) + ".h5")
    bvec = {}
    with h5py.File(fname, 'r') as fh:
        group = fh["Timestep_" + str(tindex)]
        for var in ["cbx", "cby", "cbz"]:
            dset = group[var]
            bvec[var] = np.zeros(dset.shape, dtype=dset.dtype)
            dset.read_direct(bvec[var])

    vecb_pre["bx"] = np.squeeze(bvec["cbx"])
    vecb_pre["by"] = np.squeeze(bvec["cby"])
    vecb_pre["bz"] = np.squeeze(bvec["cbz"])
    absB = np.sqrt(vecb_pre["bx"]**2 + vecb_pre["by"]**2 + vecb_pre["bz"]**2)

    for species in ["e", "i"]:
        sname = "electron" if species == 'e' else "ion"
        fname = (pic_run_dir + "hydro_hdf5/T." + str(tindex) +
                 "/hydro_" + sname + "_" + str(tindex) + ".h5")
        hydro = {}
        with h5py.File(fname, 'r') as fh:
            group = fh["Timestep_" + str(tindex)]
            for var in group:
                dset = group[var]
                hydro[var] = np.zeros(dset.shape, dtype=dset.dtype)
                dset.read_direct(hydro[var])

        irho = 1.0 / hydro["rho"]
        vx = hydro["jx"] * irho
        vy = hydro["jy"] * irho
        vz = hydro["jz"] * irho
        var = "v" + species
        vecb_pre[var+"x"] = np.squeeze(vx)
        vecb_pre[var+"y"] = np.squeeze(vy)
        vecb_pre[var+"z"] = np.squeeze(vz)
        vecb_pre["n"+species] = np.squeeze(np.abs(hydro["rho"]))
        vpar = "p" + species
        vecb_pre[vpar+"xx"] = np.squeeze(hydro["txx"] - vx * hydro["px"])
        vecb_pre[vpar+"yy"] = np.squeeze(hydro["tyy"] - vy * hydro["py"])
        vecb_pre[vpar+"zz"] = np.squeeze(hydro["tzz"] - vz * hydro["pz"])
        vecb_pre[vpar+"yx"] = np.squeeze(hydro["txy"] - vx * hydro["py"])
        vecb_pre[vpar+"xz"] = np.squeeze(hydro["tzx"] - vz * hydro["px"])
        vecb_pre[vpar+"zy"] = np.squeeze(hydro["tyz"] - vy * hydro["pz"])
        vecb_pre[vpar+"xy"] = np.squeeze(hydro["txy"] - vy * hydro["px"])
        vecb_pre[vpar+"yz"] = np.squeeze(hydro["tyz"] - vz * hydro["py"])
        vecb_pre[vpar+"zx"] = np.squeeze(hydro["tzx"] - vx * hydro["pz"])

        # Parallel and perpendicular pressure
        vname1 = vpar + "para"
        vecb_pre[vname1] = (vecb_pre[vpar+"xx"] * vecb_pre["bx"]**2 +
                            vecb_pre[vpar+"yy"] * vecb_pre["by"]**2 +
                            vecb_pre[vpar+"zz"] * vecb_pre["bz"]**2 +
                            (vecb_pre[vpar+"xy"] + vecb_pre[vpar+"yx"]) * vecb_pre["bx"] * vecb_pre["by"] +
                            (vecb_pre[vpar+"xz"] + vecb_pre[vpar+"zx"]) * vecb_pre["bx"] * vecb_pre["bz"] +
                            (vecb_pre[vpar+"yz"] + vecb_pre[vpar+"zy"]) * vecb_pre["by"] * vecb_pre["bz"])
        vecb_pre[vname1] /= absB * absB
        vname2 = vpar + "perp"
        vecb_pre[vname2] = 0.5 * (vecb_pre[vpar+"xx"] + vecb_pre[vpar+"yy"] +
                                  vecb_pre[vpar+"zz"] - vecb_pre[vname1])

    return vecb_pre


def inflow_balance(plot_config, show_plot=True):
    """Force balance in the inflow region
    """
    tframe = plot_config["tframe"]
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    fields_interval = pic_info.fields_interval
    tindex = fields_interval * tframe
    smime = math.sqrt(pic_info.mime)
    lx_de = pic_info.lx_di * smime
    lz_de = pic_info.lz_di * smime
    xmin, xmax = 0, lx_de
    zmin, zmax = -0.5 * lz_de, 0.5 * lz_de
    nx, nz = pic_info.nx, pic_info.nz
    xgrid = np.linspace(xmin, xmax, nx)
    zgrid = np.linspace(zmin, zmax, nz)
    dx_de = lx_de / nx
    dz_de = lz_de / nz

    # Find X-point
    if "open" in pic_run:
        fdir = '../data/rate_problem/exhaust_boundary/' + pic_run + '/'
        fname = fdir + 'xz_top_' + str(tframe) + '.dat'
        xz = np.fromfile(fname).reshape([2, -1])
        xlist_top = xz[0, :]
        zlist_top = xz[1, :] + zmin
        fname = fdir + 'xz_bot_' + str(tframe) + '.dat'
        xz = np.fromfile(fname).reshape([2, -1])
        xlist_bot = xz[0, :]
        zlist_bot = xz[1, :] + zmin
        x0 = 0.5 * (xlist_bot[np.argmax(zlist_bot)] +
                    xlist_top[np.argmin(zlist_top)])
        ix_xpoint = int(x0 / dx_de)
    else:
        kwargs = {"current_time": tframe,
                  "xl": 0, "xr": pic_info.lx_di,
                  "zb": -pic_info.lz_di, "zt": pic_info.lz_di}
        fname = pic_run_dir + "data/Ay.gda"
        x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
        day_dx = np.gradient(Ay, axis=1)
        day_dz = np.gradient(Ay, axis=0)
        day_dxx = np.gradient(day_dx, axis=1)
        day_dzz = np.gradient(day_dz, axis=0)
        ix_xpoint = np.argmin(Ay[nz//2, :])

    bvec_pre = get_bfield_pressure(plot_config)
    sigma = 9
    bvec_pre["bx"] = gaussian_filter(bvec_pre["bx"], sigma=sigma)
    bvec_pre["by"] = gaussian_filter(bvec_pre["by"], sigma=sigma)
    bvec_pre["bz"] = gaussian_filter(bvec_pre["bz"], sigma=sigma)
    bvec_pre["pezz"] = gaussian_filter(bvec_pre["pezz"], sigma=sigma)
    bvec_pre["pizz"] = gaussian_filter(bvec_pre["pizz"], sigma=sigma)
    bvec_pre["pepara"] = gaussian_filter(bvec_pre["pepara"], sigma=sigma)
    bvec_pre["pipara"] = gaussian_filter(bvec_pre["pipara"], sigma=sigma)
    bvec_pre["peperp"] = gaussian_filter(bvec_pre["peperp"], sigma=sigma)
    bvec_pre["piperp"] = gaussian_filter(bvec_pre["piperp"], sigma=sigma)
    b2 = bvec_pre["bx"]**2 + bvec_pre["by"]**2 + bvec_pre["bz"]**2
    bgb = bvec_pre["bx"] * np.gradient(bvec_pre["bz"], axis=0) / dx_de
    bgb += bvec_pre["bz"] * np.gradient(bvec_pre["bz"], axis=1) / dz_de
    # dbz_dx = np.diff(bvec_pre["bz"], axis=0) / dx_de
    # bgb = np.zeros([nx, nz])
    # bgb[1:, :-1] = bvec_pre["bx"][1:, :-1] * (dbz_dx[:, 1:] + dbz_dx[:, :-1]) * 0.5
    # bgb_integrate = integrate.cumtrapz(bgb, axis=1, initial=0) * dz_de
    bgb_cumsum = np.cumsum(bgb, axis=1) * dz_de

    fig1 = plt.figure(figsize=[7, 6])
    rect = [0.12, 0.25, 0.8, 0.7]
    ax1 = fig1.add_axes(rect)
    ax1.set_prop_cycle('color', COLORS)
    xcut = ix_xpoint
    fdata = b2[xcut, :]*0.5
    ax1.plot(zgrid, fdata, label=r"$B^2/8\pi$")
    fdata = bvec_pre["pezz"][xcut, :]
    ax1.plot(zgrid, fdata, label=r"$P_{e,zz}$")
    fdata = bvec_pre["pizz"][xcut, :]
    ax1.plot(zgrid, fdata, label=r"$P_{i,zz}$")
    fdata = bvec_pre["pezz"][xcut, :] + bvec_pre["pizz"][xcut, :]
    ax1.plot(zgrid, fdata, label=r"$P_{zz}$")
    lname = r"$-\int\boldsymbol{B}\cdot\nabla B_z/4\pi dz$"
    fdata = -bgb_cumsum[xcut, :]
    ax1.plot(zgrid, fdata, label=lname)
    ax1.set_xlim([zmin, zmax])
    ax1.legend(loc=8, bbox_to_anchor=(0.5, -0.35), prop={'size': 16},
               ncol=3, shadow=False, fancybox=False, frameon=False)
    ax1.tick_params(bottom=True, top=True, left=True, right=True)
    ax1.tick_params(axis='x', which='minor', direction='in')
    ax1.tick_params(axis='x', which='major', direction='in')
    ax1.tick_params(axis='y', which='minor', direction='in')
    ax1.tick_params(axis='y', which='major', direction='in')
    ax1.set_xlabel(r'$z/d_e$', fontsize=20)
    ax1.tick_params(labelsize=16)
    fdir = '../img/rate_problem/inflow_balance/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + 'inbalance_t' + str(tframe) + '_x' + str(ix_xpoint) + '_1.pdf'
    fig1.savefig(fname)

    # Anisotropy parameter
    epsilon = 1 - (bvec_pre["pepara"] + bvec_pre["pipara"] -
                   bvec_pre["peperp"] - bvec_pre["piperp"]) / b2

    # fig1 = plt.figure(figsize=[7, 5])
    # rect = [0.12, 0.15, 0.8, 0.8]
    # ax = fig1.add_axes(rect)
    # im1 = ax.imshow(epsilon.T,
    #                 extent=[xmin, xmax, zmin, zmax],
    #                 vmin=-1.0, vmax=1.0,
    #                 cmap=plt.cm.seismic, aspect='auto',
    #                 origin='lower', interpolation='bicubic')
    # ax.plot(zgrid, epsilon[xcut, :])
    # ax.plot(xgrid, epsilon[:, nz//2])
    # ppara = bvec_pre["pepara"] + bvec_pre["pipara"]
    # pperp = bvec_pre["peperp"] + bvec_pre["piperp"]
    fig = plt.figure(figsize=[7, 6])
    rect = [0.12, 0.58, 0.76, 0.4]
    hgap, vgap = 0.07, 0.09
    ax = fig.add_axes(rect)
    cmap = plt.cm.seismic
    dmin, dmax = -10, 10
    im1 = ax.imshow(-(epsilon.T - 1),
                    extent=[xmin, xmax, zmin, zmax],
                    norm = SymLogNorm(linthresh=0.03, linscale=0.03,
                                      vmin=dmin, vmax=dmax),
                    # vmin=dmin, vmax=dmax,
                    cmap=cmap, aspect='auto',
                    origin='lower', interpolation='bicubic')
    ax.plot([xgrid[ix_xpoint], xgrid[ix_xpoint]], [zmin, zmax], color='k',
            linewidth=1)
    ax.set_xlabel(r'$x/d_e$', fontsize=16)
    ax.set_ylabel(r'$z/d_e$', fontsize=16)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.tick_params(labelsize=12)
    text1 = r"$4\pi(P_\parallel-P_\perp)/B^2$"
    ax.text(0.02, 0.85, text1, color='k', fontsize=16,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)
    rect_cbar = np.copy(rect)
    rect_cbar[0] += rect[2] + 0.01
    rect_cbar[2] = 0.02
    cbar_ax = fig.add_axes(rect_cbar)
    cbar = fig.colorbar(im1, cax=cbar_ax, extend='both')
    cbar.set_ticks([-10, -1, -0.1, 0.1, 1, 10])
    cbar_ax.tick_params(bottom=False, top=False, left=False, right=True)
    cbar_ax.tick_params(axis='y', which='major', direction='out')
    cbar_ax.tick_params(axis='y', which='minor', direction='in', right=False)
    cbar.ax.tick_params(labelsize=12)

    rect[1] -= rect[3] + vgap
    rect[2] = (rect[2] - hgap) / 2
    ax = fig.add_axes(rect)
    ax.set_yscale("symlog")
    fdata = epsilon[xcut, :]
    ax.plot(zgrid, fdata)
    ax.plot([zmin, zmax], [1, 1], linestyle='--', color='k')
    ax.set_xlim([zmin, zmax])
    ax.set_xlabel(r'$z/d_e$', fontsize=16)
    label = r'$1-4\pi(P_\parallel-P_\perp)/B^2$'
    ax.set_ylabel(label, fontsize=16)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.tick_params(labelsize=12)

    rect[0] += rect[2] + hgap
    ax = fig.add_axes(rect)
    ax.set_yscale("symlog")
    fdata = epsilon[:, nz//2]
    ax.plot(xgrid, fdata)
    ax.plot([xmin, xmax], [1, 1], linestyle='--', color='k')
    ax.set_xlim([xmin, xmax])
    ax.set_xlabel(r'$x/d_e$', fontsize=16)
    label = r'$1-4\pi(P_\parallel-P_\perp)/B^2$'
    # ax.set_ylabel(label, fontsize=16)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.tick_params(labelsize=12)
    img_dir_p = '../img/rate_problem/anisotropy/' + pic_run + '/'
    mkdir_p(img_dir_p)
    fname = img_dir_p + "anisotropy_" + str(tframe) + ".pdf"
    fig.savefig(fname, dpi=200)

    tension = epsilon * bgb
    tension += (np.gradient(epsilon, axis=0) * bvec_pre["bx"] / dx_de +
                np.gradient(epsilon, axis=1) * bvec_pre["bz"] / dz_de) * bvec_pre["bz"]
    tension_cumsum = np.cumsum(tension, axis=1) * dz_de
    fig1 = plt.figure(figsize=[7, 6])
    rect = [0.12, 0.25, 0.8, 0.7]
    ax1 = fig1.add_axes(rect)
    ax1.set_prop_cycle('color', COLORS)
    fdata = b2[xcut, :]*0.5
    ax1.plot(zgrid, fdata, label=r"$B^2/8\pi$")
    fdata = bvec_pre["peperp"][xcut, :]
    ax1.plot(zgrid, fdata, label=r"$P_{e,\perp}$")
    fdata = bvec_pre["piperp"][xcut, :]
    ax1.plot(zgrid, fdata, label=r"$P_{i,\perp}$")
    fdata = bvec_pre["peperp"][xcut, :] + bvec_pre["piperp"][xcut, :]
    ax1.plot(zgrid, fdata, label=r"$P_{\perp}$")
    lname = "Tension"
    fdata = -tension_cumsum[xcut, :]
    p1, = ax1.plot(zgrid, fdata, label=lname)
    lname = r"$-\int\boldsymbol{B}\cdot\nabla B_z/4\pi dz$"
    fdata = -bgb_cumsum[xcut, :]
    ax1.plot(zgrid, fdata, linestyle='--', color=p1.get_color(), label=lname)
    ax1.set_xlim([zmin, zmax])
    ax1.legend(loc=8, bbox_to_anchor=(0.5, -0.35), prop={'size': 16},
               ncol=3, shadow=False, fancybox=False, frameon=False)

    nz1 = 200
    izs, ize = nz//2 - nz1, nz//2 + nz1

    ylim = ax1.get_ylim()
    ax1.plot([zgrid[izs], zgrid[izs]], ylim, linestyle='--',
             linewidth=1, color='k')
    ax1.plot([zgrid[ize], zgrid[ize]], ylim, linestyle='--',
             linewidth=1, color='k')
    ax1.set_ylim(ylim)
    b0 = pic_info.b0
    print(bvec_pre["bx"][ix_xpoint, [izs, ize]] / b0)

    ax1.tick_params(bottom=True, top=True, left=True, right=True)
    ax1.tick_params(axis='x', which='minor', direction='in')
    ax1.tick_params(axis='x', which='major', direction='in')
    ax1.tick_params(axis='y', which='minor', direction='in')
    ax1.tick_params(axis='y', which='major', direction='in')
    ax1.set_xlabel(r'$z/d_e$', fontsize=20)
    ax1.tick_params(labelsize=16)
    fdir = '../img/rate_problem/inflow_balance/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + 'inbalance_t' + str(tframe) + '_x' + str(ix_xpoint) + '_2.pdf'
    fig1.savefig(fname)

    if show_plot:
        plt.show()
    else:
        plt.close("all")


def outflow_balance(plot_config, show_plot=True):
    """Force balance in the outflow region
    """
    tframe = plot_config["tframe"]
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    fields_interval = pic_info.fields_interval
    tindex = fields_interval * tframe
    smime = math.sqrt(pic_info.mime)
    lx_de = pic_info.lx_di * smime
    lz_de = pic_info.lz_di * smime
    xmin, xmax = 0, lx_de
    zmin, zmax = -0.5 * lz_de, 0.5 * lz_de
    nx, nz = pic_info.nx, pic_info.nz
    xgrid = np.linspace(xmin, xmax, nx)
    zgrid = np.linspace(zmin, zmax, nz)
    dx_de = lx_de / nx
    dz_de = lz_de / nz

    bvec_pre = get_bfield_pressure(plot_config)
    rho = bvec_pre["ne"] + bvec_pre["ni"] + pic_info.mime
    irho = 1.0 / rho
    vsx = (bvec_pre["ne"] * bvec_pre["vex"] +
           bvec_pre["ni"] * bvec_pre["vix"] * pic_info.mime) * irho
    vsy = (bvec_pre["ne"] * bvec_pre["vey"] +
           bvec_pre["ni"] * bvec_pre["viy"] * pic_info.mime) * irho
    vsz = (bvec_pre["ne"] * bvec_pre["vez"] +
           bvec_pre["ni"] * bvec_pre["viz"] * pic_info.mime) * irho
    sigma = 9
    b2 = bvec_pre["bx"]**2 + bvec_pre["by"]**2 + bvec_pre["bz"]**2
    # epsilon = 1 - (bvec_pre["pepara"] + bvec_pre["pipara"] -
    #                bvec_pre["peperp"] - bvec_pre["piperp"]) / b2
    # epsilon = gaussian_filter(epsilon, sigma=sigma)
    rho = gaussian_filter(rho, sigma=sigma)
    vsx = gaussian_filter(vsx, sigma=sigma)
    vsy = gaussian_filter(vsy, sigma=sigma)
    vsz = gaussian_filter(vsz, sigma=sigma)
    bvec_pre["bx"] = gaussian_filter(bvec_pre["bx"], sigma=sigma)
    bvec_pre["by"] = gaussian_filter(bvec_pre["by"], sigma=sigma)
    bvec_pre["bz"] = gaussian_filter(bvec_pre["bz"], sigma=sigma)
    bvec_pre["pezz"] = gaussian_filter(bvec_pre["pezz"], sigma=sigma)
    bvec_pre["pizz"] = gaussian_filter(bvec_pre["pizz"], sigma=sigma)
    bvec_pre["pepara"] = gaussian_filter(bvec_pre["pepara"], sigma=sigma)
    bvec_pre["pipara"] = gaussian_filter(bvec_pre["pipara"], sigma=sigma)
    bvec_pre["peperp"] = gaussian_filter(bvec_pre["peperp"], sigma=sigma)
    bvec_pre["piperp"] = gaussian_filter(bvec_pre["piperp"], sigma=sigma)
    b2 = bvec_pre["bx"]**2 + bvec_pre["by"]**2 + bvec_pre["bz"]**2

    # Anisotropy parameter
    ppara = bvec_pre["pepara"] + bvec_pre["pipara"]
    pperp = bvec_pre["peperp"] + bvec_pre["piperp"]
    epsilon = 1 - (ppara - pperp) / b2

    tension = epsilon * (np.gradient(bvec_pre["bx"], axis=0) * bvec_pre["bx"] / dx_de +
                         np.gradient(bvec_pre["bx"], axis=1) * bvec_pre["bz"] / dz_de)
    tension += (np.gradient(epsilon, axis=0) * bvec_pre["bx"] / dx_de +
                np.gradient(epsilon, axis=1) * bvec_pre["bz"] / dz_de) * bvec_pre["bx"]
    tension0 = (np.gradient(bvec_pre["bx"], axis=0) * bvec_pre["bx"] / dx_de +
                np.gradient(bvec_pre["bx"], axis=1) * bvec_pre["bz"] / dz_de)

    dbulk = rho * (vsx * np.gradient(vsx, axis=0) / dx_de +
                   vsz * np.gradient(vsx, axis=1) / dz_de)
    db2 = 0.5 * np.gradient(b2, axis=0) / dx_de
    dpperp = np.gradient(pperp, axis=0) / dx_de
    forcex = tension - db2 - dpperp
    forcex0 = tension0 - db2 - dpperp

    tension_mean = np.mean(tension, axis=1)
    tension0_mean = np.mean(tension0, axis=1)
    tension_cumsum = np.cumsum(tension, axis=0) * dx_de
    tension0_cumsum = np.cumsum(tension0, axis=0) * dx_de

    # Exhaust boundary
    fdir = '../data/rate_problem/exhaust_boundary/' + pic_run + '/'
    fname = fdir + 'xz_top_' + str(tframe) + '.dat'
    xz = np.fromfile(fname).reshape([2, -1])
    xlist_top = xz[0, :]
    zlist_top = xz[1, :] + zmin
    f = interp1d(xlist_top, zlist_top)
    ztop = f(xgrid)
    iz_top = np.floor((ztop - zmin) / dz_de).astype(int)
    dz_top = (ztop - zmin) / dz_de - iz_top
    fname = fdir + 'xz_bot_' + str(tframe) + '.dat'
    xz = np.fromfile(fname).reshape([2, -1])
    xlist_bot = xz[0, ::-1]
    zlist_bot = xz[1, ::-1] + zmin
    f = interp1d(xlist_bot, zlist_bot)
    zbot = f(xgrid)
    iz_bot = np.ceil((zbot - zmin) / dz_de).astype(int)
    dz_bot = iz_bot - (zbot - zmin) / dz_de

    # X-point
    x_xp = 0.5 * (xlist_bot[np.argmax(zlist_bot)] +
                  xlist_top[np.argmin(zlist_top)])
    ix_xp = int(x_xp / dx_de)

    # The work done by the forces starting from the X-point
    work_force = np.zeros([5, nx])
    for ix in range(nx):
        work_force[0, ix] = np.sum(db2[ix, iz_bot[ix]:iz_top[ix]+1])
        work_force[1, ix] = np.sum(dpperp[ix, iz_bot[ix]:iz_top[ix]+1])
        work_force[2, ix] = np.sum(dbulk[ix, iz_bot[ix]:iz_top[ix]+1])
        work_force[3, ix] = np.sum(tension[ix, iz_bot[ix]:iz_top[ix]+1])
        work_force[4, ix] = np.sum(tension0[ix, iz_bot[ix]:iz_top[ix]+1])
        work_force[0, ix] += db2[ix, iz_bot[ix]-1] * dz_bot[ix]
        work_force[0, ix] += db2[ix, iz_top[ix]+1] * dz_top[ix]
        work_force[1, ix] += dpperp[ix, iz_bot[ix]-1] * dz_bot[ix]
        work_force[1, ix] += dpperp[ix, iz_top[ix]+1] * dz_top[ix]
        work_force[2, ix] += dbulk[ix, iz_bot[ix]-1] * dz_bot[ix]
        work_force[2, ix] += dbulk[ix, iz_top[ix]+1] * dz_top[ix]
        work_force[3, ix] += tension[ix, iz_bot[ix]-1] * dz_bot[ix]
        work_force[3, ix] += tension[ix, iz_top[ix]+1] * dz_top[ix]
        work_force[4, ix] += tension0[ix, iz_bot[ix]-1] * dz_bot[ix]
        work_force[4, ix] += tension0[ix, iz_top[ix]+1] * dz_top[ix]

    work_force_int = np.zeros(work_force.shape)
    # All forces are 0 at the X-point
    for i in range(5):
        work_force_int[:, ix_xp::-1] = np.cumsum(work_force[:, ix_xp::-1], axis=1)
        work_force_int[:, ix_xp:] = np.cumsum(work_force[:, ix_xp:], axis=1)
    work_force_int *= dx_de * dz_de

    fig = plt.figure(figsize=[8, 6])
    rect = [0.1, 0.50, 0.85, 0.37]
    vgap = 0.02
    ax = fig.add_axes(rect)
    COLORS = palettable.colorbrewer.qualitative.Set1_9.mpl_colors
    ax.set_prop_cycle('color', COLORS)
    lname1 = r"$\int_{x_0}^x\partial(B^2/8\pi)/\partial x |dx|$"
    p1, = ax.plot(xgrid, work_force_int[0], label=lname1)
    lname2 = r"$\int_{x_0}^x\partial P_{\perp}/\partial x |dx|$"
    p2, = ax.plot(xgrid, work_force_int[1], label=lname2)
    lname3 = r"$\int_{x_0}^x\partial(\rho v_x^2/2)/\partial x |dx|$"
    p3, = ax.plot(xgrid, work_force_int[2], label=lname3)
    lname4 = r"$\int_{x_0}^x\text{Tension }|dx|$"
    p4, = ax.plot(xgrid, work_force_int[3], label=lname4)
    lname5 = r"$\int_{x0}^x\boldsymbol{B}\cdot\nabla B_x/4\pi |dx|$"
    p5, = ax.plot(xgrid, work_force_int[4], color=p4.get_color(),
                  linestyle='--', label=lname5)
    ylim = ax.get_ylim()
    ax.plot([x_xp, x_xp], ylim, color='k', linestyle='--', linewidth=1)
    ax.plot([xmin, xmax], [0, 0], color='k', linestyle='--', linewidth=1)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim(ylim)
    ax.legend(loc="upper center", prop={'size': 12}, ncol=3,
              bbox_to_anchor=(0.5, 1.33),
              shadow=False, fancybox=False, frameon=False)
    xpos = (x_xp - xmin) / lx_de + 0.01
    ax.text(xpos, 0.85, r"$x_0$", color='k', fontsize=16,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.tick_params(axis='x', labelbottom=False)
    ax.tick_params(labelsize=16)

    rect[1] -= rect[3] + vgap
    ax1 = fig.add_axes(rect)
    ax1.set_prop_cycle('color', COLORS)
    ax1.plot(xgrid, work_force_int[2], color=p3.get_color(), label=lname3)
    lname = lname4 + r"$-$" + lname1 + r"$-$" + lname2
    fdata = work_force_int[3] - work_force_int[0] - work_force_int[1]
    ax1.plot(xgrid, fdata, color=p3.get_color(), linestyle='--', label=lname)
    fdata = work_force_int[4] - work_force_int[3]
    lname = lname5 + r"$-$" + lname4
    ax1.plot(xgrid, fdata, color='k', label=lname)
    ax1.plot([xmin, xmax], [0, 0], color='k', linestyle='--', linewidth=1)
    ax1.set_xlim([xmin, xmax])
    ax1.legend(loc="best", prop={'size': 12}, ncol=1,
               shadow=False, fancybox=False, frameon=False)
    ax1.tick_params(bottom=True, top=True, left=True, right=True)
    ax1.tick_params(axis='x', which='minor', direction='in')
    ax1.tick_params(axis='x', which='major', direction='in')
    ax1.tick_params(axis='y', which='minor', direction='in')
    ax1.tick_params(axis='y', which='major', direction='in')
    ax1.set_xlabel(r'$x/d_e$', fontsize=20)
    ax1.tick_params(labelsize=16)

    img_dir = '../img/rate_problem/outflow_balance/' + pic_run + '/'
    mkdir_p(img_dir)
    fname = img_dir + "outflow_balance_" + str(tframe) + ".pdf"
    fig.savefig(fname)

    fig = plt.figure(figsize=[7, 5])
    rect = [0.12, 0.15, 0.82, 0.7]
    ax = fig.add_axes(rect)
    ax.set_prop_cycle('color', COLORS)
    ax.plot(xgrid, work_force_int[3] / work_force_int[4], linewidth=2)
    ax.grid()
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([0.5, 1.2])
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlabel(r'$x/d_e$', fontsize=20)
    ax.tick_params(labelsize=16)
    lname = (r"$\int_{x_0}^x\text{Tension }|dx|/" +
             r"\int_{x0}^x\boldsymbol{B}\cdot\nabla B_x/4\pi |dx|$")
    fig.suptitle(lname, fontsize=20)
    fname = img_dir + "epsilon_avg_" + str(tframe) + ".pdf"
    fig.savefig(fname)

    if show_plot:
        plt.show()
    else:
        plt.close()


def outflow_balance_center(plot_config, show_plot=True):
    """Force balance in the outflow in the center of the current sheet
    """
    tframe = plot_config["tframe"]
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    fields_interval = pic_info.fields_interval
    tindex = fields_interval * tframe
    smime = math.sqrt(pic_info.mime)
    lx_de = pic_info.lx_di * smime
    lz_de = pic_info.lz_di * smime
    xmin, xmax = 0, lx_de
    zmin, zmax = -0.5 * lz_de, 0.5 * lz_de
    nx, nz = pic_info.nx, pic_info.nz
    xgrid = np.linspace(xmin, xmax, nx)
    zgrid = np.linspace(zmin, zmax, nz)
    dx_de = lx_de / nx
    dz_de = lz_de / nz

    bvec_pre = get_bfield_pressure(plot_config)
    rho = bvec_pre["ne"] + bvec_pre["ni"] + pic_info.mime
    irho = 1.0 / rho
    vsx = (bvec_pre["ne"] * bvec_pre["vex"] +
           bvec_pre["ni"] * bvec_pre["vix"] * pic_info.mime) * irho
    vsy = (bvec_pre["ne"] * bvec_pre["vey"] +
           bvec_pre["ni"] * bvec_pre["viy"] * pic_info.mime) * irho
    vsz = (bvec_pre["ne"] * bvec_pre["vez"] +
           bvec_pre["ni"] * bvec_pre["viz"] * pic_info.mime) * irho
    sigma = 9
    b2 = bvec_pre["bx"]**2 + bvec_pre["by"]**2 + bvec_pre["bz"]**2
    rho = gaussian_filter(rho, sigma=sigma)
    vsx = gaussian_filter(vsx, sigma=sigma)
    vsy = gaussian_filter(vsy, sigma=sigma)
    vsz = gaussian_filter(vsz, sigma=sigma)
    bvec_pre["bx"] = gaussian_filter(bvec_pre["bx"], sigma=sigma)
    bvec_pre["by"] = gaussian_filter(bvec_pre["by"], sigma=sigma)
    bvec_pre["bz"] = gaussian_filter(bvec_pre["bz"], sigma=sigma)
    bvec_pre["pezz"] = gaussian_filter(bvec_pre["pezz"], sigma=sigma)
    bvec_pre["pizz"] = gaussian_filter(bvec_pre["pizz"], sigma=sigma)
    bvec_pre["pepara"] = gaussian_filter(bvec_pre["pepara"], sigma=sigma)
    bvec_pre["pipara"] = gaussian_filter(bvec_pre["pipara"], sigma=sigma)
    bvec_pre["peperp"] = gaussian_filter(bvec_pre["peperp"], sigma=sigma)
    bvec_pre["piperp"] = gaussian_filter(bvec_pre["piperp"], sigma=sigma)
    b2 = bvec_pre["bx"]**2 + bvec_pre["by"]**2 + bvec_pre["bz"]**2

    # Anisotropy parameter
    ppara = bvec_pre["pepara"] + bvec_pre["pipara"]
    pperp = bvec_pre["peperp"] + bvec_pre["piperp"]
    epsilon = 1 - (ppara - pperp) / b2

    tension = epsilon * (np.gradient(bvec_pre["bx"], axis=0) * bvec_pre["bx"] / dx_de +
                         np.gradient(bvec_pre["bx"], axis=1) * bvec_pre["bz"] / dz_de)
    tension += (np.gradient(epsilon, axis=0) * bvec_pre["bx"] / dx_de +
                np.gradient(epsilon, axis=1) * bvec_pre["bz"] / dz_de) * bvec_pre["bx"]
    tension0 = (np.gradient(bvec_pre["bx"], axis=0) * bvec_pre["bx"] / dx_de +
                np.gradient(bvec_pre["bx"], axis=1) * bvec_pre["bz"] / dz_de)

    dbulk = rho * (vsx * np.gradient(vsx, axis=0) / dx_de +
                   vsz * np.gradient(vsx, axis=1) / dz_de)
    db2 = 0.5 * np.gradient(b2, axis=0) / dx_de
    dpperp = np.gradient(pperp, axis=0) / dx_de
    forcex = tension - db2 - dpperp
    forcex0 = tension0 - db2 - dpperp

    tension_mean = np.mean(tension, axis=1)
    tension0_mean = np.mean(tension0, axis=1)
    tension_cumsum = np.cumsum(tension, axis=0) * dx_de
    tension0_cumsum = np.cumsum(tension0, axis=0) * dx_de

    # Exhaust boundary
    fdir = '../data/rate_problem/exhaust_boundary/' + pic_run + '/'
    fname = fdir + 'xz_top_' + str(tframe) + '.dat'
    xz = np.fromfile(fname).reshape([2, -1])
    xlist_top = xz[0, :]
    zlist_top = xz[1, :] + zmin
    f = interp1d(xlist_top, zlist_top)
    ztop = f(xgrid)
    iz_top = np.floor((ztop - zmin) / dz_de).astype(int)
    dz_top = (ztop - zmin) / dz_de - iz_top
    fname = fdir + 'xz_bot_' + str(tframe) + '.dat'
    xz = np.fromfile(fname).reshape([2, -1])
    xlist_bot = xz[0, ::-1]
    zlist_bot = xz[1, ::-1] + zmin
    f = interp1d(xlist_bot, zlist_bot)
    zbot = f(xgrid)
    iz_bot = np.ceil((zbot - zmin) / dz_de).astype(int)
    dz_bot = iz_bot - (zbot - zmin) / dz_de

    # X-point
    x_xp = 0.5 * (xlist_bot[np.argmax(zlist_bot)] +
                  xlist_top[np.argmin(zlist_top)])
    ix_xp = int(x_xp / dx_de)

    # The work done by the forces starting from the X-point
    work_force = np.zeros([5, nx])
    work_force[0, :] = db2[:, nz//2]
    work_force[1, :] = dpperp[:, nz//2]
    work_force[2, :] = dbulk[:, nz//2]
    work_force[3, :] = tension[:, nz//2]
    work_force[4, :] = tension0[:, nz//2]

    work_force_int = np.zeros(work_force.shape)
    # All forces are 0 at the X-point
    for i in range(5):
        work_force_int[:, ix_xp::-1] = np.cumsum(work_force[:, ix_xp::-1], axis=1)
        work_force_int[:, ix_xp:] = np.cumsum(work_force[:, ix_xp:], axis=1)
    work_force_int *= dx_de * dz_de

    fig = plt.figure(figsize=[8, 6])
    rect = [0.1, 0.50, 0.85, 0.37]
    vgap = 0.02
    ax = fig.add_axes(rect)
    COLORS = palettable.colorbrewer.qualitative.Set1_9.mpl_colors
    ax.set_prop_cycle('color', COLORS)
    lname1 = r"$\int_{x_0}^x\partial(B^2/8\pi)/\partial x |dx|$"
    p1, = ax.plot(xgrid, work_force_int[0], label=lname1)
    lname2 = r"$\int_{x_0}^x\partial P_{\perp}/\partial x |dx|$"
    p2, = ax.plot(xgrid, work_force_int[1], label=lname2)
    lname3 = r"$\int_{x_0}^x\partial(\rho v_x^2/2)/\partial x |dx|$"
    p3, = ax.plot(xgrid, work_force_int[2], label=lname3)
    lname4 = r"$\int_{x_0}^x\text{Tension }|dx|$"
    p4, = ax.plot(xgrid, work_force_int[3], label=lname4)
    lname5 = r"$\int_{x0}^x\boldsymbol{B}\cdot\nabla B_x/4\pi |dx|$"
    p5, = ax.plot(xgrid, work_force_int[4], color=p4.get_color(),
                  linestyle='--', label=lname5)
    ylim = ax.get_ylim()
    ax.plot([x_xp, x_xp], ylim, color='k', linestyle='--', linewidth=1)
    ax.plot([xmin, xmax], [0, 0], color='k', linestyle='--', linewidth=1)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim(ylim)
    ax.legend(loc="upper center", prop={'size': 12}, ncol=3,
              bbox_to_anchor=(0.5, 1.33),
              shadow=False, fancybox=False, frameon=False)
    xpos = (x_xp - xmin) / lx_de + 0.01
    ax.text(xpos, 0.85, r"$x_0$", color='k', fontsize=16,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.tick_params(axis='x', labelbottom=False)
    ax.tick_params(labelsize=16)

    rect[1] -= rect[3] + vgap
    ax1 = fig.add_axes(rect)
    ax1.set_prop_cycle('color', COLORS)
    ax1.plot(xgrid, work_force_int[2], color=p3.get_color(), label=lname3)
    lname = lname4 + r"$-$" + lname1 + r"$-$" + lname2
    fdata = work_force_int[3] - work_force_int[0] - work_force_int[1]
    ax1.plot(xgrid, fdata, color=p3.get_color(), linestyle='--', label=lname)
    fdata = work_force_int[4] - work_force_int[3]
    lname = lname5 + r"$-$" + lname4
    ax1.plot(xgrid, fdata, color='k', label=lname)
    ax1.plot([xmin, xmax], [0, 0], color='k', linestyle='--', linewidth=1)
    ax1.set_xlim([xmin, xmax])
    ax1.legend(loc="best", prop={'size': 12}, ncol=1,
               shadow=False, fancybox=False, frameon=False)
    ax1.tick_params(bottom=True, top=True, left=True, right=True)
    ax1.tick_params(axis='x', which='minor', direction='in')
    ax1.tick_params(axis='x', which='major', direction='in')
    ax1.tick_params(axis='y', which='minor', direction='in')
    ax1.tick_params(axis='y', which='major', direction='in')
    ax1.set_xlabel(r'$x/d_e$', fontsize=20)
    ax1.tick_params(labelsize=16)

    img_dir = '../img/rate_problem/outflow_balance/' + pic_run + '/'
    mkdir_p(img_dir)
    fname = img_dir + "outflow_balance_" + str(tframe) + ".pdf"
    fig.savefig(fname)

    fig = plt.figure(figsize=[7, 5])
    rect = [0.12, 0.15, 0.82, 0.7]
    ax = fig.add_axes(rect)
    ax.set_prop_cycle('color', COLORS)
    ax.plot(xgrid, work_force_int[3] / work_force_int[4], linewidth=2)
    ax.grid()
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([0.5, 1.2])
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlabel(r'$x/d_e$', fontsize=20)
    ax.tick_params(labelsize=16)
    lname = (r"$\int_{x_0}^x\text{Tension }|dx|/" +
             r"\int_{x0}^x\boldsymbol{B}\cdot\nabla B_x/4\pi |dx|$")
    fig.suptitle(lname, fontsize=20)
    # fname = img_dir + "epsilon_avg_" + str(tframe) + ".pdf"
    # fig.savefig(fname)

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_vout(plot_config, show_plot=True):
    """Plot outflow velocity
    """
    tframe = plot_config["tframe"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    fields_interval = pic_info.fields_interval
    tindex = fields_interval * tframe
    vpic_info = get_vpic_info(pic_run_dir)
    mime = pic_info.mime
    smime = math.sqrt(mime)
    lx_de = pic_info.lx_di * smime
    lz_de = pic_info.lz_di * smime
    xmin, xmax = 0, lx_de
    zmin, zmax = -0.5 * lz_de, 0.5 * lz_de
    xgrid = np.linspace(xmin, xmax, pic_info.nx)
    dtwpe = pic_info.dtwpe
    dtwce = pic_info.dtwce
    nb_n0 = vpic_info["nb/n0"]
    va = dtwce * math.sqrt(1.0 / mime) / dtwpe / math.sqrt(nb_n0)
    print("Alfven speed/c: %f" % va)

    rho_vel = {}

    for species in ["e", "i"]:
        sname = "electron" if species == 'e' else "ion"
        fname = (pic_run_dir + "hydro_hdf5/T." + str(tindex) +
                 "/hydro_" + sname + "_" + str(tindex) + ".h5")
        hydro = {}
        with h5py.File(fname, 'r') as fh:
            group = fh["Timestep_" + str(tindex)]
            for var in group:
                dset = group[var]
                hydro[var] = np.zeros(dset.shape, dtype=dset.dtype)
                dset.read_direct(hydro[var])

        irho = 1.0 / hydro["rho"]
        vx = hydro["jx"] * irho
        vy = hydro["jy"] * irho
        vz = hydro["jz"] * irho
        var = "v" + species
        rho_vel[var+"x"] = np.squeeze(vx)
        rho_vel[var+"y"] = np.squeeze(vy)
        rho_vel[var+"z"] = np.squeeze(vz)
        rho_vel["n"+species] = np.squeeze(np.abs(hydro["rho"]))

    irho = 1.0 / (rho_vel["ne"] + rho_vel["ni"] * mime)
    vsx = (rho_vel["ne"] * rho_vel["vex"] +
           rho_vel["ni"] * rho_vel["vix"] * mime) * irho
    vsy = (rho_vel["ne"] * rho_vel["vey"] +
           rho_vel["ni"] * rho_vel["viy"] * mime) * irho
    vsz = (rho_vel["ne"] * rho_vel["vez"] +
           rho_vel["ni"] * rho_vel["viz"] * mime) * irho
    nx, nz = vsx.shape
    fig = plt.figure(figsize=[7, 5])
    rect = [0.12, 0.15, 0.82, 0.8]
    ax = fig.add_axes(rect)
    ax.set_prop_cycle('color', COLORS)
    sigma = 5
    fdata = gaussian_filter(vsx[:, nz//2], sigma=sigma)
    ax.plot(xgrid, fdata/va, linewidth=2, label=r"$v_x/v_A$")
    fdata = gaussian_filter(vsy[:, nz//2], sigma=sigma)
    ax.plot(xgrid, fdata/va, linewidth=2, label=r"$v_y/v_A$")
    fdata = gaussian_filter(vsz[:, nz//2], sigma=sigma)
    ax.plot(xgrid, fdata/va, linewidth=2, label=r"$v_z/v_A$")
    ax.set_xlim([xmin, xmax])
    ax.legend(loc=4, prop={'size': 16}, ncol=1,
              shadow=False, fancybox=False, frameon=False)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlabel(r'$x/d_e$', fontsize=20)
    ax.tick_params(labelsize=16)
    img_dir = '../img/rate_problem/vout_cut/' + pic_run + '/'
    mkdir_p(img_dir)
    fname = img_dir + "vout_cut_" + str(tframe) + ".pdf"
    fig.savefig(fname)

    fig = plt.figure(figsize=[7, 6])
    rect = [0.15, 0.12, 0.7, 0.8]
    ax = fig.add_axes(rect)
    fdata = gaussian_filter(vsx, sigma=sigma)
    im1 = ax.imshow(fdata.T/va, cmap=plt.cm.seismic,
                    extent=[xmin, xmax, zmin, zmax],
                    vmin=-0.5, vmax=0.5, aspect='auto',
                    origin='lower', interpolation='bicubic')
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.tick_params(labelsize=16)
    ax.set_xlabel(r'$x/d_e$', fontsize=20)
    ax.set_ylabel(r'$z/d_e$', fontsize=20)
    rect_cbar = np.copy(rect)
    rect_cbar[0] += rect[2] + 0.02
    rect_cbar[2] = 0.02
    cbar_ax = fig.add_axes(rect_cbar)
    cbar = fig.colorbar(im1, cax=cbar_ax, extend='both')
    cbar_ax.tick_params(bottom=False, top=False, left=False, right=True)
    cbar_ax.tick_params(axis='y', which='major', direction='out')
    cbar_ax.tick_params(axis='y', which='minor', direction='in', right=False)
    cbar.ax.tick_params(labelsize=12)
    cbar_ax.set_title(r'$v_x/v_A$', fontsize=20)
    fname = img_dir + "vx_" + str(tframe) + ".jpg"
    fig.savefig(fname, dpi=200)

    if show_plot:
        plt.show()
    else:
        plt.close("all")


def plot_density(plot_config, show_plot=True):
    """Force balance in the inflow region
    """
    tframe = plot_config["tframe"]
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    fields_interval = pic_info.fields_interval
    tindex = fields_interval * tframe
    smime = math.sqrt(pic_info.mime)
    lx_de = pic_info.lx_di * smime
    lz_de = pic_info.lz_di * smime
    xmin, xmax = 0, lx_de
    zmin, zmax = -0.5 * lz_de, 0.5 * lz_de
    nx, nz = pic_info.nx, pic_info.nz
    xgrid = np.linspace(xmin, xmax, nx)
    zgrid = np.linspace(zmin, zmax, nz)
    dx_de = lx_de / nx
    dz_de = lz_de / nz

    rhos = {}
    for var in ["electron", "ion"]:
        fname = (pic_run_dir + "hydro_hdf5/T." + str(tindex) +
                 "/hydro_" + var + "_" + str(tindex) + ".h5")
        with h5py.File(fname, 'r') as fh:
            group = fh["Timestep_" + str(tindex)]
            dset = group["rho"]
            rhos[var] = np.zeros(dset.shape, dtype=dset.dtype)
            dset.read_direct(rhos[var])
        rhos[var] = gaussian_filter(np.abs(np.squeeze(rhos[var])), sigma=9)

    # Find X-point
    fdir = '../data/rate_problem/exhaust_boundary/' + pic_run + '/'
    fname = fdir + 'xz_top_' + str(tframe) + '.dat'
    xz = np.fromfile(fname).reshape([2, -1])
    xlist_top = xz[0, :]
    zlist_top = xz[1, :] + zmin
    fname = fdir + 'xz_bot_' + str(tframe) + '.dat'
    xz = np.fromfile(fname).reshape([2, -1])
    xlist_bot = xz[0, :]
    zlist_bot = xz[1, :] + zmin
    x0 = 0.5 * (xlist_bot[np.argmax(zlist_bot)] +
                xlist_top[np.argmin(zlist_top)])
    ix_xpoint = int(x0 / dx_de)

    xcut = ix_xpoint
    fig = plt.figure(figsize=[9, 8])
    rect0 = [0.09, 0.70, 0.5, 0.28]
    rect = np.copy(rect0)
    hgap, vgap = 0.12, 0.03
    cmap = plt.cm.viridis
    for ivar, var in enumerate(rhos):
        ax = fig.add_axes(rect)
        print("Maximum density: %f" % rhos[var].max())
        im1 = ax.imshow(rhos[var].T,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=0.5, vmax=4,
                        cmap=cmap, aspect='auto',
                        origin='lower', interpolation='bicubic')
        ax.plot([xgrid[xcut], xgrid[xcut]], [zmin, zmax], linewidth=1,
                linestyle='--', color='w')
        ax.tick_params(axis='x', labelbottom=False)
        ax.set_ylabel(r'$z/d_e$', fontsize=16)
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')
        ax.tick_params(labelsize=12)
        species = 'e' if var == "electron" else "i"
        text1 = r"$n_" + species + "$"
        ax.text(0.02, 0.85, text1, color='k', fontsize=16,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)
        rect[1] -= rect[3] + vgap
    rect[1] += rect[3] + vgap
    rect_cbar = np.copy(rect)
    rect_cbar[0] += rect[2] + 0.01
    rect_cbar[1] = rect[1] + rect[3] * 0.5
    rect_cbar[2] = 0.015
    rect_cbar[3] = rect[3] + vgap
    cbar_ax = fig.add_axes(rect_cbar)
    cbar = fig.colorbar(im1, cax=cbar_ax, extend='both')
    cbar_ax.tick_params(bottom=False, top=False, left=False, right=True)
    cbar_ax.tick_params(axis='y', which='major', direction='out')
    cbar_ax.tick_params(axis='y', which='minor', direction='in', right=False)
    cbar.ax.tick_params(labelsize=12)

    rect[1] -= rect[3] + vgap
    ax = fig.add_axes(rect)
    ax.plot(xgrid, rhos["electron"][:, nz//2], label=r"$n_e$")
    ax.plot(xgrid, rhos["ion"][:, nz//2], label=r"$n_i$")
    ax.plot([xmin, xmax], [1, 1], linewidth=1, linestyle='--', color='k')
    ax.set_xlim([xmin, xmax])
    ax.set_xlabel(r'$x/d_e$', fontsize=16)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.tick_params(labelsize=12)
    ax.legend(loc=4, prop={'size': 16}, ncol=1,
              shadow=False, fancybox=False, frameon=False)

    rect = np.copy(rect0)
    rect[0] += rect[2] + hgap
    rect[2] = 0.26
    ax1 = fig.add_axes(rect)
    p1, = ax1.plot(rhos["electron"][xcut, :], zgrid)
    ax1.set_ylim([zmin, zmax])
    ax1.tick_params(bottom=True, top=True, left=True, right=True)
    ax1.tick_params(axis='x', which='minor', direction='in')
    ax1.tick_params(axis='x', which='major', direction='in')
    ax1.tick_params(axis='y', which='minor', direction='in')
    ax1.tick_params(axis='y', which='major', direction='in')
    ax1.tick_params(labelsize=12)
    ax1.tick_params(axis='x', labelbottom=False)

    rect[1] -= rect[3] + vgap
    ax2 = fig.add_axes(rect)
    p2, = ax2.plot(rhos["ion"][xcut, :], zgrid)
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim([zmin, zmax])
    ax2.tick_params(bottom=True, top=True, left=True, right=True)
    ax2.tick_params(axis='x', which='minor', direction='in')
    ax2.tick_params(axis='x', which='major', direction='in')
    ax2.tick_params(axis='y', which='minor', direction='in')
    ax2.tick_params(axis='y', which='major', direction='in')
    ax2.tick_params(labelsize=12)

    img_dir_p = '../img/rate_problem/nrho/' + pic_run + '/'
    mkdir_p(img_dir_p)
    fname = img_dir_p + "nrho_" + str(tframe) + ".pdf"
    fig.savefig(fname, dpi=200)

    if show_plot:
        plt.show()
    else:
        plt.close()


def middle_step_rk4(x, z, nx, nz, dx, dz, Bx, Bz):
    """Middle step of Runge-Kutta method to trace the magnetic field line.

    Args:
        x, z: the coordinates of current point.
        nx, nz: the dimensions of the data.
        Bx, Bz: the magnetic field arrays.
    """
    ix1 = int(math.floor(x / dx))
    iz1 = int(math.floor(z / dz))
    offsetx = x / dx - ix1
    offsetz = z / dz - iz1
    ix2 = ix1 + 1
    iz2 = iz1 + 1
    v1 = (1.0 - offsetx) * (1.0 - offsetz)
    v2 = offsetx * (1.0 - offsetz)
    v3 = offsetx * offsetz
    v4 = (1.0 - offsetx) * offsetz
    if ix1 < nx and ix2 < nx and iz1 < nz and iz2 < nz:
        bx = (Bx[iz1, ix1] * v1 + Bx[iz1, ix2] * v2 +
              Bx[iz2, ix2] * v3 + Bx[iz2, ix1] * v4)
        bz = (Bz[iz1, ix1] * v1 + Bz[iz1, ix2] * v2 +
              Bz[iz2, ix2] * v3 + Bz[iz2, ix1] * v4)
        absB = math.sqrt(bx**2 + bz**2)
        deltax1 = bx / absB
        deltaz1 = bz / absB
    else:
        if ix1 >= nx:
            ix1 = nx - 1
        if iz1 >= nz:
            iz1 = nz - 1
        bx = Bx[iz1, ix1]
        bz = Bz[iz1, ix1]
    absB = math.sqrt(bx**2 + bz**2)
    deltax1 = bx / absB
    deltaz1 = bz / absB
    return (deltax1, deltaz1, bx)


def trace_field_line(bvec, pic_info, x0, z0):
    """Tracer magnetic field line

    Args:
        bvec: magnetic field
        pic_info: PIC simulation information
        x0, z0: starting point of field line tracing (in de)
    """
    nz, nx = bvec["cbx"].shape
    smime = math.sqrt(pic_info.mime)
    lx_de = pic_info.lx_di * smime
    lz_de = pic_info.lz_di * smime
    xmin, xmax = 0, lx_de
    zmin, zmax = -0.5 * lz_de, 0.5 * lz_de
    xgrid = np.linspace(xmin, xmax, pic_info.nx)
    zgrid = np.linspace(zmin, zmax, pic_info.nz)
    dx_de = pic_info.dx_di * smime
    dz_de = pic_info.dz_di * smime
    i = int(x0 / dx_de)
    k = int((z0 - zmin) / dz_de)

    nstep = 0
    deltas = math.sqrt(dx_de**2 + dz_de**2) * 0.1
    hds = deltas * 0.5
    total_lengh = 0
    xs = x0
    zs = z0 - zmin
    xlist = [xs]
    zlist = [zs]
    x, z = xs, zs
    xcond = x >= xgrid[0] and x <= xgrid[-1]
    zcond = z >= 0 and z <= zgrid[-1] - zgrid[0]
    while xcond and zcond and total_lengh < 5 * lx_de:
        deltax1, deltaz1, _ = middle_step_rk4(x, z, nx, nz, dx_de, dz_de,
                                              bvec["cbx"], bvec["cbz"])
        x1 = x + deltax1 * hds
        z1 = z + deltaz1 * hds
        deltax2, deltaz2, _ = middle_step_rk4(x1, z1, nx, nz, dx_de, dz_de,
                                              bvec["cbx"], bvec["cbz"])
        x2 = x + deltax2 * hds
        z2 = z + deltaz2 * hds
        deltax3, deltaz3, _ = middle_step_rk4(x2, z2, nx, nz, dx_de, dz_de,
                                              bvec["cbx"], bvec["cbz"])
        x3 = x + deltax3 * deltas
        z3 = z + deltaz3 * deltas
        deltax4, deltaz4, _ = middle_step_rk4(x3, z3, nx, nz, dx_de, dz_de,
                                              bvec["cbx"], bvec["cbz"])
        x += deltas/6 * (deltax1 + 2*deltax2 + 2*deltax3 + deltax4)
        z += deltas/6 * (deltaz1 + 2*deltaz2 + 2*deltaz3 + deltaz4)
        total_lengh += deltas
        xlist.append(x)
        zlist.append(z)
        nstep += 1
        length = math.sqrt((x-xs)**2 + (z-zs)**2)
        if length < dx_de and nstep > 20:
            break
        xcond = x >= xgrid[0] and x <= xgrid[-1]
        zcond = z >= 0 and z <= zgrid[-1] - zgrid[0]

    _, _, bx = middle_step_rk4(xs, zs, nx, nz, dx_de, dz_de,
                               bvec["cbx"], bvec["cbz"])
    xlist = np.asarray(xlist)
    zlist = np.asarray(zlist)
    return (xlist, zlist, bx)


def calc_rrate_bflux(plot_config, show_plot=True):
    """Calculate reconnection rate based magnetic flux
    """
    pic_run = plot_config["pic_run"]
    tframe = plot_config["tframe"]
    pic_run_dir = plot_config["pic_run_dir"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    smime = math.sqrt(pic_info.mime)
    lx_de = pic_info.lx_di * smime
    lz_de = pic_info.lz_di * smime
    xmin, xmax = 0, lx_de
    zmin, zmax = -0.5 * lz_de, 0.5 * lz_de
    nx = pic_info.nx
    nz = pic_info.nz
    xgrid = np.linspace(xmin, xmax, pic_info.nx)
    zgrid = np.linspace(zmin, zmax, pic_info.nz)
    dx_de = pic_info.dx_di * smime
    dz_de = pic_info.dz_di * smime
    ntf = pic_info.ntf
    fields_interval = pic_info.fields_interval
    dtwpe = pic_info.dtwpe
    dtwce = pic_info.dtwce
    va = dtwce * math.sqrt(1.0 / pic_info.mime) / dtwpe

    print("Time frame %d of %d" % (tframe, ntf))
    tindex = fields_interval * tframe
    fname = (pic_run_dir + "field_hdf5/T." + str(tindex) +
             "/fields_" + str(tindex) + ".h5")
    bvec = {}
    with h5py.File(fname, 'r') as fh:
        group = fh["Timestep_" + str(tindex)]
        for var in ["cbx", "cbz"]:
            dset = group[var]
            bvec[var] = np.zeros(dset.shape, dtype=dset.dtype)
            dset.read_direct(bvec[var])
            bvec[var] = np.squeeze(bvec[var]).T

    z1, z2 = 0, zmax
    while (z2 - z1) > 0.1 * dz_de:
        zmid = (z1 + z2) * 0.5
        # print("Starting z-position: %f" % zmid)
        xlist, zlist, bx = trace_field_line(bvec, pic_info, 0, zmid)
        if xlist[-1] > xmax:
            z2 = zmid
        else:
            z1 = zmid

    xlist, zlist, _ = trace_field_line(bvec, pic_info, 0, z2)
    # plt.plot(xlist, zlist)
    # plt.show()

    iz_close = math.floor((z1 - zmin) / dz_de)
    dz_close = 1.0 - (z1 - zmin) / dz_de + iz_close
    bflux = np.sum(bvec["cbx"][iz_close+1:, 0]) + bvec["cbx"][iz_close, 0] * dz_close
    bflux *= dz_de

    fdir = '../data/rate_problem/rrate_bflux/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + 'rrate_bflux_' + str(tframe) + '.dat'
    bflux = np.asarray([bflux])
    bflux.tofile(fname)

    xz = np.asarray([xlist, zlist])
    fname = fdir + 'xz_close_' + str(tframe) + '.dat'
    xz.tofile(fname)


def get_edrive_params(plot_config):
    """Get driving parameters in the inflow region
    """
    pic_run_dir = plot_config["pic_run_dir"]
    deck_file = pic_run_dir + "reconnection.cc"
    with open(deck_file) as fh:
        content = fh.readlines()
    line_no = 0
    cond1 = True
    cond2 = True
    cond3 = True
    cond = cond1 or cond2 or cond3
    while cond:
        line_no += 1
        cond1 = "edrive" not in content[line_no]
        line = content[line_no].lstrip()
        if len(line) > 0:
            cond2 = line[0] == '/'
            cond3 = "=" not in line
        cond = cond1 or cond2 or cond3
    line_splits = content[line_no].split("=")
    edrive = float(line_splits[1].split("*")[0])

    line_no = 0
    cond1 = True
    cond2 = True
    cond3 = True
    cond = cond1 or cond2 or cond3
    while cond:
        line_no += 1
        cond1 = "tdrive" not in content[line_no]
        line = content[line_no].lstrip()
        if len(line) > 0:
            cond2 = line[0] == '/'
            cond3 = "=" not in line
        cond = cond1 or cond2 or cond3
    line_splits = content[line_no].split("=")
    tdrive = float(line_splits[1].split(";")[0])

    return (edrive, tdrive)


def plot_rrate_bflux(plot_config, show_plot=True):
    """Plot reconnection rate based magnetic flux
    """
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    vpic_info = get_vpic_info(pic_run_dir)
    nb_n0 = vpic_info["nb/n0"]
    dtwpe = pic_info.dtwpe
    dtwce = pic_info.dtwce
    va = dtwce * math.sqrt(1.0 / pic_info.mime) / dtwpe / math.sqrt(nb_n0)
    wpe_wce = dtwpe / dtwce
    b0 = pic_info.b0
    fields_interval = pic_info.fields_interval
    dtf = pic_info.dtwpe * fields_interval
    ntf = pic_info.ntf
    tfields = np.arange(ntf) * dtf
    tfields_wci = np.arange(ntf) * pic_info.dtwci * fields_interval
    bflux = np.zeros(ntf)
    for tframe in range(ntf):
        fdir = '../data/rate_problem/rrate_bflux/' + pic_run + '/'
        fname = fdir + 'rrate_bflux_' + str(tframe) + '.dat'
        fdata = np.fromfile(fname)
        bflux[tframe] = fdata[0]

    edrive, tdrive = get_edrive_params(plot_config)
    vin = edrive * (1.0 - np.exp(-tfields/tdrive)) / wpe_wce / b0

    rrate_bflux = -np.gradient(bflux) / dtf
    rrate_bflux /= va * b0
    if "open" in pic_run or "test" in pic_run:
        rrate_bflux += vin
    fig1 = plt.figure(figsize=[7, 5])
    rect = [0.12, 0.15, 0.8, 0.8]
    ax = fig1.add_axes(rect)
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    ax.set_prop_cycle('color', COLORS)
    ax.plot(tfields_wci, rrate_bflux, marker='o', label="Rate from magnetic flux")

    # if ("open" not in pic_run) and ("test" not in pic_run):
    #     fdir = '../data/rate_problem/rrate/'
    #     fname = fdir + 'rrate_' + pic_run + '.dat'
    #     reconnection_rate = np.fromfile(fname)
    #     ax.plot(tfields_wci, reconnection_rate, label=r"Rate from $A_y$")
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlim([tfields_wci[0], tfields_wci[-1]])
    # ax.set_ylim([0, 0.15])
    ax.tick_params(labelsize=12)
    ax.set_xlabel(r'$t\Omega_{ci}$', fontsize=16)
    ax.legend(loc=4, prop={'size': 16}, ncol=1,
              shadow=False, fancybox=False, frameon=False)

    fdir = '../img/rate_problem/rrate/'
    mkdir_p(fdir)
    fname = fdir + 'rrate_bflux_' + pic_run + '.pdf'
    fig1.savefig(fname)

    plt.show()


def open_angle(plot_config, show_plot=True):
    """Plot exhaust open angle
    """
    tframe = plot_config["tframe"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    fields_interval = pic_info.fields_interval
    tindex = fields_interval * tframe
    smime = math.sqrt(pic_info.mime)
    lx_de = pic_info.lx_di * smime
    lz_de = pic_info.lz_di * smime
    xmin, xmax = 0, lx_de
    zmin, zmax = -0.5 * lz_de, 0.5 * lz_de
    dx_de = lx_de / pic_info.nx
    dz_de = lz_de / pic_info.nz

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

    xgrid = np.linspace(xmin, xmax, pic_info.nx)
    zgrid = np.linspace(zmin, zmax, pic_info.nz)

    absj = np.squeeze(np.sqrt((je["jx"] + ji["jx"])**2 +
                              (je["jy"] + ji["jy"])**2 +
                              (je["jz"] + ji["jz"])**2))
    fig = plt.figure(figsize=[10, 10*lz_de/lx_de])
    rect = [0.11, 0.14, 0.8, 0.78]
    ax = fig.add_axes(rect)
    im1 = ax.imshow(absj.T,
                    extent=[xmin, xmax, zmin, zmax],
                    vmin=0, vmax=0.06,
                    cmap=plt.cm.viridis, aspect='auto',
                    origin='lower', interpolation='bicubic')
    # Magnetic field lines
    fname = (pic_run_dir + "field_hdf5/T." + str(tindex) +
             "/fields_" + str(tindex) + ".h5")
    bvec = {}
    with h5py.File(fname, 'r') as fh:
        group = fh["Timestep_" + str(tindex)]
        for var in ["cbx", "cbz"]:
            dset = group[var]
            bvec[var] = np.zeros(dset.shape, dtype=dset.dtype)
            dset.read_direct(bvec[var])
            bvec[var] = np.squeeze(bvec[var]).T
    xmesh, zmesh = np.meshgrid(xgrid, zgrid)
    ax.streamplot(xmesh, zmesh, bvec["cbx"], bvec["cbz"],
                  color='w', linewidth=0.5)

    fdir = '../data/rate_problem/exhaust_boundary/' + pic_run + '/'
    fname = fdir + 'xz_top_' + str(tframe) + '.dat'
    xz = np.fromfile(fname).reshape([2, -1])
    xlist_top = xz[0, :]
    zlist_top = xz[1, :] + zmin
    fname = fdir + 'xz_bot_' + str(tframe) + '.dat'
    xz = np.fromfile(fname).reshape([2, -1])
    xlist_bot = xz[0, :]
    zlist_bot = xz[1, :] + zmin
    x0 = 0.5 * (xlist_bot[np.argmax(zlist_bot)] +
                xlist_top[np.argmin(zlist_top)])
    ix_xpoint = int(x0 / dx_de)
    z0 = zlist_top[np.argmin(zlist_top)]

    ax.plot(xlist_top, zlist_top, linewidth=2, color=COLORS[0])
    ax.plot(xlist_bot, zlist_bot, linewidth=2, color=COLORS[0])

    if pic_run == "mime100_Tbe_Te_01_small_open_thick":
        ix0 = np.argmax(xlist)
        x0 = xlist[ix0]
        z0 = zlist[ix0]
        length = 60.0
        angle0 = 10.0
        angle = (180 - angle0) * math.pi / 180
        x1 = x0 + length * math.cos(angle)
        z1 = z0 + length * math.sin(angle)
        ax.plot([x0, x1], [z0, z1], color='k', linestyle='--')
        ax.plot([x0, x1], [0, 0], color='k', linestyle='--')
        text1 = r"$" + ("{%0.1f}" % angle0) +  "^\circ$"
        ax.text(0.3, 0.5, text1, color='k', fontsize=16,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='right', verticalalignment='bottom',
                transform=ax.transAxes)
    elif pic_run == "mime100_Tbe_Te_1_small_open":
        length = 60.0
        angle0 = 13.5
        angle = angle0 * math.pi / 180
        x1 = x0 + length * math.cos(angle)
        z1 = z0 + length * math.sin(angle)
        ax.plot([x0, x1], [z0, z1], color='k', linestyle='--')
        ax.plot([x0, x1], [0, 0], color='k', linestyle='--')
        text1 = r"$" + ("{%0.1f}" % angle0) +  "^\circ$"
        ax.text(0.62, 0.5, text1, color='k', fontsize=16,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='bottom',
                transform=ax.transAxes)
    elif pic_run == "mime100_Tbe_Te_10_small_open_weak":
        length = 60.0
        angle0 = 14.0
        angle = (180 - angle0) * math.pi / 180
        x1 = x0 + length * math.cos(angle)
        z1 = z0 + length * math.sin(angle)
        ax.plot([x0, x1], [z0, z1], color='k', linestyle='--')
        ax.plot([x0, x1], [0, 0], color='k', linestyle='--')
        text1 = r"$" + ("{%0.1f}" % angle0) +  "^\circ$"
        ax.text(0.38, 0.5, text1, color='k', fontsize=16,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='right', verticalalignment='bottom',
                transform=ax.transAxes)
    elif pic_run == "mime400_nb_n0_002":
        length = 300.0
        angle0 = 20.0
        angle = (180 - angle0) * math.pi / 180
        x1 = x0 + length * math.cos(angle)
        z1 = z0 + length * math.sin(angle)
        ax.plot([x0, x1], [z0, z1], color='k', linestyle='--')
        ax.plot([x0, x1], [0, 0], color='k', linestyle='--')
        text1 = r"$" + ("{%0.1f}" % angle0) +  "^\circ$"
        ax.text(0.38, 0.5, text1, color='k', fontsize=16,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='right', verticalalignment='bottom',
                transform=ax.transAxes)
    elif pic_run == "mime400_nb_n0_02":
        length = 300.0
        angle0 = 10.0
        angle = (180 - angle0) * math.pi / 180
        x1 = x0 + length * math.cos(angle)
        z1 = z0 + length * math.sin(angle)
        ax.plot([x0, x1], [z0, z1], color='k', linestyle='--')
        ax.plot([x0, x1], [0, 0], color='k', linestyle='--')
        text1 = r"$" + ("{%0.1f}" % angle0) +  "^\circ$"
        ax.text(0.38, 0.5, text1, color='k', fontsize=16,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='right', verticalalignment='bottom',
                transform=ax.transAxes)
    elif pic_run == "mime400_nb_n0_1":
        length = 300.0
        angle0 = 10.0
        angle = (180 - angle0) * math.pi / 180
        x1 = x0 + length * math.cos(angle)
        z1 = z0 + length * math.sin(angle)
        ax.plot([x0, x1], [z0, z1], color='k', linestyle='--')
        ax.plot([x0, x1], [0, 0], color='k', linestyle='--')
        text1 = r"$" + ("{%0.1f}" % angle0) +  "^\circ$"
        ax.text(0.38, 0.5, text1, color='k', fontsize=16,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='right', verticalalignment='bottom',
                transform=ax.transAxes)

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([zmin, zmax])
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlabel(r'$x/d_e$', fontsize=20)
    ax.set_ylabel(r'$z/d_e$', fontsize=20)
    ax.tick_params(labelsize=16)
    rect_cbar = np.copy(rect)
    rect_cbar[0] += rect[2] + 0.02
    rect_cbar[2] = 0.015
    rect_cbar[1] += 0.15 * rect[3]
    rect_cbar[3] = rect[3] * 0.7
    cbar_ax = fig.add_axes(rect_cbar)
    cbar = fig.colorbar(im1, cax=cbar_ax, extend='max')
    cbar_ax.tick_params(bottom=False, top=False, left=False, right=True)
    cbar_ax.tick_params(axis='y', which='major', direction='out')
    cbar_ax.tick_params(axis='y', which='minor', direction='in', right=False)
    cbar_ax.set_title(r'$|\boldsymbol{J}|$', fontsize=20)
    cbar.ax.tick_params(labelsize=12)
    twpe = math.ceil(tindex * pic_info.dtwpe / 0.1) * 0.1
    text1 = r'$t\omega_{pe}=' + ("{%0.0f}" % twpe) + '$'
    fig.suptitle(text1, fontsize=20)
    img_dir = '../img/rate_problem/absj_angle/' + pic_run + '/'
    mkdir_p(img_dir)
    fname = img_dir + "absj_angle_" + str(tframe) + ".jpg"
    fig.savefig(fname, dpi=200)

    if show_plot:
        plt.show()
    else:
        plt.close()


def get_exhaust_boundary(plot_config, show_plot=True):
    """Get the boundary of reconnection exhaust
    """
    pic_run = plot_config["pic_run"]
    tframe = plot_config["tframe"]
    pic_run_dir = plot_config["pic_run_dir"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    smime = math.sqrt(pic_info.mime)
    lx_de = pic_info.lx_di * smime
    lz_de = pic_info.lz_di * smime
    xmin, xmax = 0, lx_de
    zmin, zmax = -0.5 * lz_de, 0.5 * lz_de
    nx = pic_info.nx
    nz = pic_info.nz
    xgrid = np.linspace(xmin, xmax, pic_info.nx)
    zgrid = np.linspace(zmin, zmax, pic_info.nz)
    dx_de = pic_info.dx_di * smime
    dz_de = pic_info.dz_di * smime
    ntf = pic_info.ntf
    fields_interval = pic_info.fields_interval
    dtwpe = pic_info.dtwpe
    dtwce = pic_info.dtwce
    va = dtwce * math.sqrt(1.0 / pic_info.mime) / dtwpe
    vpic_info = get_vpic_info(pic_run_dir)
    lde = vpic_info["L/de"]

    print("Time frame %d of %d" % (tframe, ntf))
    tindex = fields_interval * tframe
    fname = (pic_run_dir + "field_hdf5/T." + str(tindex) +
             "/fields_" + str(tindex) + ".h5")
    bvec = {}
    with h5py.File(fname, 'r') as fh:
        group = fh["Timestep_" + str(tindex)]
        for var in ["cbx", "cbz"]:
            dset = group[var]
            bvec[var] = np.zeros(dset.shape, dtype=dset.dtype)
            dset.read_direct(bvec[var])
            bvec[var] = np.squeeze(bvec[var]).T

    z1, z2 = 0, zmax
    while (z2 - z1) > 0.1 * dz_de:
        zmid = (z1 + z2) * 0.5
        # print("Starting z-position: %f" % zmid)
        xlist, zlist, _ = trace_field_line(bvec, pic_info, 0, zmid)
        if xlist[-1] > xmax:
            z2 = zmid
        else:
            z1 = zmid
    xlist_top, zlist_top, _ = trace_field_line(bvec, pic_info, 0, z2)

    z1, z2 = 0, zmin
    while (z1 - z2) > 0.1 * dz_de:
        zmid = (z1 + z2) * 0.5
        # print("Starting z-position: %f" % zmid)
        xlist, zlist, _ = trace_field_line(bvec, pic_info, xmax, zmid)
        if xlist[-1] < xmin:
            z2 = zmid
        else:
            z1 = zmid
    xlist_bot, zlist_bot, _ = trace_field_line(bvec, pic_info, xmax, z2)

#     plt.plot(xlist_top, zlist_top)
#     plt.plot(xlist_bot, zlist_bot)
#     plt.show()

    fdir = '../data/rate_problem/exhaust_boundary/' + pic_run + '/'
    mkdir_p(fdir)

    xz = np.asarray([xlist_top, zlist_top])
    fname = fdir + 'xz_top_' + str(tframe) + '.dat'
    xz.tofile(fname)
    xz = np.asarray([xlist_bot, zlist_bot])
    fname = fdir + 'xz_bot_' + str(tframe) + '.dat'
    xz.tofile(fname)


def inflow_pressure(plot_config, show_plot=True):
    """pressure in the inflow region
    """
    tframe = plot_config["tframe"]
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    fields_interval = pic_info.fields_interval
    tindex = fields_interval * tframe
    smime = math.sqrt(pic_info.mime)
    lx_de = pic_info.lx_di * smime
    lz_de = pic_info.lz_di * smime
    xmin, xmax = 0, lx_de
    zmin, zmax = -0.5 * lz_de, 0.5 * lz_de
    nx, nz = pic_info.nx, pic_info.nz
    xgrid = np.linspace(xmin, xmax, nx)
    zgrid = np.linspace(zmin, zmax, nz)
    dx_de = lx_de / nx
    dz_de = lz_de / nz

    # Find X-point
    if "open" in pic_run:
        fdir = '../data/rate_problem/exhaust_boundary/' + pic_run + '/'
        fname = fdir + 'xz_top_' + str(tframe) + '.dat'
        xz = np.fromfile(fname).reshape([2, -1])
        xlist_top = xz[0, :]
        zlist_top = xz[1, :] + zmin
        fname = fdir + 'xz_bot_' + str(tframe) + '.dat'
        xz = np.fromfile(fname).reshape([2, -1])
        xlist_bot = xz[0, :]
        zlist_bot = xz[1, :] + zmin
        x0 = 0.5 * (xlist_bot[np.argmax(zlist_bot)] +
                    xlist_top[np.argmin(zlist_top)])
        ix_xpoint = int(x0 / dx_de)
    else:
        kwargs = {"current_time": tframe,
                  "xl": 0, "xr": pic_info.lx_di,
                  "zb": -0.5*pic_info.lz_di, "zt": 0.5*pic_info.lz_di}
        fname = pic_run_dir + "data/Ay.gda"
        x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
        day_dx = np.gradient(Ay, axis=1)
        day_dz = np.gradient(Ay, axis=0)
        day_dxx = np.gradient(day_dx, axis=1)
        day_dzz = np.gradient(day_dz, axis=0)
        ix_xpoint = np.argmin(Ay[nz//2, :])

    vpic_info = get_vpic_info(pic_run_dir)
    vthe = vpic_info["vtheb/c"]
    n0 = vpic_info["n0"]
    b0 = vpic_info["b0"]
    Te = vpic_info["Te"]
    Tbe_Te = vpic_info["Tbe/Te"]
    nb_n0 = vpic_info["nb/n0"]
    nb = n0 * nb_n0
    p0 = nb * Te * Tbe_Te

    bvec_pre = get_bfield_pressure(plot_config)
    sigma = 9
    bvec_pre["bx"] = gaussian_filter(bvec_pre["bx"], sigma=sigma).T / b0
    bvec_pre["by"] = gaussian_filter(bvec_pre["by"], sigma=sigma).T / b0
    bvec_pre["bz"] = gaussian_filter(bvec_pre["bz"], sigma=sigma).T / b0
    bvec_pre["ne"] = gaussian_filter(bvec_pre["ne"], sigma=sigma).T / nb
    bvec_pre["ni"] = gaussian_filter(bvec_pre["ni"], sigma=sigma).T / nb
    bvec_pre["pepara"] = gaussian_filter(bvec_pre["pepara"], sigma=sigma).T / p0
    bvec_pre["pipara"] = gaussian_filter(bvec_pre["pipara"], sigma=sigma).T / p0
    bvec_pre["peperp"] = gaussian_filter(bvec_pre["peperp"], sigma=sigma).T / p0
    bvec_pre["piperp"] = gaussian_filter(bvec_pre["piperp"], sigma=sigma).T / p0
    b2 = bvec_pre["bx"]**2 + bvec_pre["by"]**2 + bvec_pre["bz"]**2
    ib2 = 1.0 / b2
    b = np.sqrt(b2)
    # CGL
    peperp = bvec_pre["ne"] * b
    pepara = bvec_pre["ne"]**3 * ib2
    piperp = bvec_pre["ni"] * b
    pipara = bvec_pre["ni"]**3 * ib2
    # Egedal & Le
    alpha = bvec_pre["ne"]**3 * ib2
    peperp2 = (bvec_pre["ne"] / (1 + alpha) +
               bvec_pre["ne"] * b / (1 + 1/alpha))
    pepara2 = (bvec_pre["ne"] / (1 + 0.5*alpha) +
               math.pi * alpha / (6.0 * (1 + 0.5/alpha)))
    alpha = bvec_pre["ni"]**3 * ib2
    piperp2 = (bvec_pre["ni"] / (1 + alpha) +
               bvec_pre["ni"] * b / (1 + 1/alpha))
    pipara2 = (bvec_pre["ni"] / (1 + 0.5*alpha) +
               math.pi * alpha / (6.0 * (1 + 0.5/alpha)))
    epsilon = 1 - (bvec_pre["pepara"] + bvec_pre["pipara"] -
                   bvec_pre["peperp"] - bvec_pre["piperp"]) * ib2
    epsilon1 = 1 - (pepara + pipara - peperp - piperp) * ib2

    fig = plt.figure(figsize=[10, 10])
    rect0 = [0.08, 0.76, 0.4, 0.2]
    hgap, vgap = 0.03, 0.04
    cmap = plt.cm.seismic
    axs = []
    for col in range(2):
        for row in range(4):
            rect = np.copy(rect0)
            rect[0] += col * (rect[2] + vgap)
            rect[1] -= row * (rect[3] + hgap)
            ax = fig.add_axes(rect)
            axs.append(ax)
            ax.plot([xgrid[ix_xpoint], xgrid[ix_xpoint]], [zmin, zmax],
                    color='k', linewidth=1)
            if col == 0:
                ax.set_ylabel(r'$z/d_e$', fontsize=16)
            else:
                ax.tick_params(axis='y', labelleft=False)
            if row == 3:
                ax.set_xlabel(r'$x/d_e$', fontsize=16)
            else:
                ax.tick_params(axis='x', labelbottom=False)
            ax.tick_params(bottom=True, top=True, left=True, right=True)
            ax.tick_params(axis='x', which='minor', direction='in')
            ax.tick_params(axis='x', which='major', direction='in')
            ax.tick_params(axis='y', which='minor', direction='in')
            ax.tick_params(axis='y', which='major', direction='in')
            ax.tick_params(labelsize=12)
    fdata = (pepara - bvec_pre["pepara"]) / bvec_pre["pepara"]
    im1 = axs[0].imshow(fdata,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=-0.5, vmax=0.5,
                        cmap=cmap, aspect='auto',
                        origin='lower', interpolation='bicubic')
    text1 = r"$(P_{e\parallel}^\prime-P_{e\parallel})/P_{e\parallel}$"
    axs[0].text(0.02, 0.85, text1, color='w', fontsize=16,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='center',
                transform=axs[0].transAxes)
    axs[0].set_title('CGL', fontsize=20)

    fdata = (peperp - bvec_pre["peperp"]) / bvec_pre["peperp"]
    im1 = axs[1].imshow(fdata,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=-0.5, vmax=0.5,
                        cmap=cmap, aspect='auto',
                        origin='lower', interpolation='bicubic')
    text1 = r"$(P_{e\perp}^\prime-P_{e\perp})/P_{e\perp}$"
    axs[1].text(0.02, 0.85, text1, color='k', fontsize=16,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='center',
                transform=axs[1].transAxes)

    fdata = (pipara - bvec_pre["pipara"]) / bvec_pre["pipara"]
    im1 = axs[2].imshow(fdata,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=-0.5, vmax=0.5,
                        cmap=cmap, aspect='auto',
                        origin='lower', interpolation='bicubic')
    text1 = r"$(P_{i\parallel}^\prime-P_{i\parallel})/P_{i\parallel}$"
    axs[2].text(0.02, 0.85, text1, color='w', fontsize=16,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='center',
                transform=axs[2].transAxes)

    fdata = (piperp - bvec_pre["piperp"]) / bvec_pre["piperp"]
    im1 = axs[3].imshow(fdata,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=-0.5, vmax=0.5,
                        cmap=cmap, aspect='auto',
                        origin='lower', interpolation='bicubic')
    text1 = r"$(P_{i\perp}^\prime-P_{i\perp})/P_{i\perp}$"
    axs[3].text(0.02, 0.85, text1, color='k', fontsize=16,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='center',
                transform=axs[3].transAxes)

    fdata = (pepara2 - bvec_pre["pepara"]) / bvec_pre["pepara"]
    im1 = axs[4].imshow(fdata,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=-0.5, vmax=0.5,
                        cmap=cmap, aspect='auto',
                        origin='lower', interpolation='bicubic')
    text1 = r"$(P_{e\parallel}^\prime-P_{e\parallel})/P_{e\parallel}$"
    axs[4].text(0.02, 0.85, text1, color='k', fontsize=16,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='center',
                transform=axs[4].transAxes)
    axs[4].set_title('Egedal \& Le', fontsize=20)

    fdata = (peperp2 - bvec_pre["peperp"]) / bvec_pre["peperp"]
    im1 = axs[5].imshow(fdata,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=-0.5, vmax=0.5,
                        cmap=cmap, aspect='auto',
                        origin='lower', interpolation='bicubic')
    text1 = r"$(P_{e\perp}^\prime-P_{e\perp})/P_{e\perp}$"
    axs[5].text(0.02, 0.85, text1, color='k', fontsize=16,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='center',
                transform=axs[5].transAxes)

    fdata = (pipara2 - bvec_pre["pipara"]) / bvec_pre["pipara"]
    im1 = axs[6].imshow(fdata,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=-0.5, vmax=0.5,
                        cmap=cmap, aspect='auto',
                        origin='lower', interpolation='bicubic')
    text1 = r"$(P_{i\parallel}^\prime-P_{i\parallel})/P_{i\parallel}$"
    axs[6].text(0.02, 0.85, text1, color='k', fontsize=16,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='center',
                transform=axs[6].transAxes)

    fdata = (piperp2 - bvec_pre["piperp"]) / bvec_pre["piperp"]
    im1 = axs[7].imshow(fdata,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=-0.5, vmax=0.5,
                        cmap=cmap, aspect='auto',
                        origin='lower', interpolation='bicubic')
    text1 = r"$(P_{i\perp}^\prime-P_{i\perp})/P_{i\perp}$"
    axs[7].text(0.02, 0.85, text1, color='k', fontsize=16,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='center',
                transform=axs[7].transAxes)

    rect_cbar = np.copy(rect0)
    rect_cbar[0] += rect0[2] * 2 + vgap + 0.01
    rect_cbar[2] = 0.015
    rect_cbar[1] -= (rect0[3] + hgap) * 2
    rect_cbar[3] = rect0[3] * 2 + hgap
    cbar_ax = fig.add_axes(rect_cbar)
    cbar = fig.colorbar(im1, cax=cbar_ax, extend='both')
    cbar_ax.tick_params(bottom=False, top=False, left=False, right=True)
    cbar_ax.tick_params(axis='y', which='major', direction='out')
    cbar_ax.tick_params(axis='y', which='minor', direction='in', right=False)
    cbar.ax.tick_params(labelsize=12)

    img_dir = '../img/rate_problem/inflow_pressure/' + pic_run + '/'
    mkdir_p(img_dir)
    fname = img_dir + "pres_" + str(tframe) + ".jpg"
    fig.savefig(fname)

    if show_plot:
        plt.show()
    else:
        plt.close("all")


def calc_bxm(plot_config, show_plot=True):
    """The x-component of magnetic field in the upstream of the diffusion region
    """
    tframe = plot_config["tframe"]
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    fields_interval = pic_info.fields_interval
    tindex = fields_interval * tframe
    smime = math.sqrt(pic_info.mime)
    lx_de = pic_info.lx_di * smime
    lz_de = pic_info.lz_di * smime
    xmin, xmax = 0, lx_de
    zmin, zmax = -0.5 * lz_de, 0.5 * lz_de
    nx, nz = pic_info.nx, pic_info.nz
    xgrid = np.linspace(xmin, xmax, nx)
    zgrid = np.linspace(zmin, zmax, nz)
    dx_de = lx_de / nx
    dz_de = lz_de / nz

    # Find X-point
    if "open" in pic_run:
        fdir = '../data/rate_problem/exhaust_boundary/' + pic_run + '/'
        fname = fdir + 'xz_top_' + str(tframe) + '.dat'
        xz = np.fromfile(fname).reshape([2, -1])
        xlist_top = xz[0, :]
        zlist_top = xz[1, :] + zmin
        fname = fdir + 'xz_bot_' + str(tframe) + '.dat'
        xz = np.fromfile(fname).reshape([2, -1])
        xlist_bot = xz[0, :]
        zlist_bot = xz[1, :] + zmin
        x0 = 0.5 * (xlist_bot[np.argmax(zlist_bot)] +
                    xlist_top[np.argmin(zlist_top)])
        ix_xpoint = int(x0 / dx_de)
    else:
        kwargs = {"current_time": tframe,
                  "xl": 0, "xr": pic_info.lx_di,
                  "zb": -0.5*pic_info.lz_di, "zt": 0.5*pic_info.lz_di}
        fname = pic_run_dir + "data/Ay.gda"
        x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
        day_dx = np.gradient(Ay, axis=1)
        day_dz = np.gradient(Ay, axis=0)
        day_dxx = np.gradient(day_dx, axis=1)
        day_dzz = np.gradient(day_dz, axis=0)
        ix_xpoint = np.argmin(Ay[nz//2, :])

    # Magnetic field
    fname = (pic_run_dir + "field_hdf5/T." + str(tindex) +
             "/fields_" + str(tindex) + ".h5")
    emf = {}
    with h5py.File(fname, 'r') as fh:
        group = fh["Timestep_" + str(tindex)]
        for var in ["cbx", "cbz", "ey"]:
            dset = group[var]
            emf[var] = np.zeros(dset.shape, dtype=dset.dtype)
            dset.read_direct(emf[var])

    for var in emf:
        emf[var] = np.squeeze(emf[var])

    fname = (pic_run_dir + "hydro_hdf5/T." + str(tindex) +
             "/hydro_ion_" + str(tindex) + ".h5")
    hydro = {}
    with h5py.File(fname, 'r') as fh:
        group = fh["Timestep_" + str(tindex)]
        for var in ["jx", "jz", "rho"]:
            dset = group[var]
            hydro[var] = np.zeros(dset.shape, dtype=dset.dtype)
            dset.read_direct(hydro[var])
    vx = np.squeeze(hydro["jx"] / hydro["rho"])
    vz = np.squeeze(hydro["jz"] / hydro["rho"])

    vxb_y = emf["cbz"] * vx - emf["cbx"] * vz
    eres = emf["ey"] - vxb_y
    eres = gaussian_filter(eres, sigma=9)

    nz1 = 80
    izs, ize = nz//2 - nz1, nz//2 + nz1
    fig1 = plt.figure(figsize=[7, 5])
    rect = [0.12, 0.15, 0.8, 0.8]
    ax = fig1.add_axes(rect)
    ax.plot(eres[ix_xpoint, :])
    ax.plot([izs, ize], eres[ix_xpoint, [izs, ize]], linestyle='none',
            marker='o', markersize=10)
    b0 = pic_info.b0
    print(emf["cbx"][ix_xpoint, [izs, ize]] / b0)

    if show_plot:
        plt.show()
    else:
        plt.close("all")


def get_cmd_args():
    """Get command line arguments
    """
    # default_pic_run = 'mime100_Tbe_Te_01'
    # default_pic_run = 'mime100_Tbe_Te_1'
    # default_pic_run = 'mime100_Tbe_Te_10'
    # default_pic_run = 'mime100_Tbe_Te_01_small_open_thick'
    # default_pic_run = 'mime100_Tbe_Te_1_small_open'
    # default_pic_run = 'mime100_Tbe_Te_10_small_open_weak'
    # default_pic_run = 'mime100_nb_n0_001'
    # default_pic_run = 'mime100_nb_n0_01'
    # default_pic_run = 'mime100_nb_n0_1'
    # default_pic_run = 'mime400_nb_n0_002'
    # default_pic_run = 'mime400_nb_n0_02'
    # default_pic_run = 'mime400_nb_n0_1'
    # default_pic_run = 'mime400_nb_n0_1_new'
    default_pic_run = 'mime400_Tbe_Te_20'
    # default_pic_run = 'mime400_Tbe_Te_20_new'
    # default_pic_run = 'mime1_Tbe_Te_001_open_new'
    # default_pic_run = 'mime1_Tbe_Te_20_open_new'
    # default_pic_run = 'high_beta_test'
    default_pic_run_dir = ('/net/scratch4/xiaocan/reconnection_rate/' + default_pic_run + '/')
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
    parser.add_argument('--plot_absj', action="store_true", default=False,
                        help='whether to plot current density')
    parser.add_argument('--plot_bfield', action="store_true", default=False,
                        help='whether to plot magnetic field')
    parser.add_argument('--plot_efield', action="store_true", default=False,
                        help='whether to plot electric field')
    parser.add_argument('--plot_ptensor', action="store_true", default=False,
                        help='whether to plot pressure tensor')
    parser.add_argument('--plot_rrate', action="store_true", default=False,
                        help="whether to plot magnetic reconnection rate")
    parser.add_argument('--inflow_balance', action="store_true", default=False,
                        help="whether analyzing inflow balance")
    parser.add_argument('--outflow_balance', action="store_true", default=False,
                        help="whether analyzing inflow balance")
    parser.add_argument('--outflow_balance_center', action="store_true", default=False,
                        help="whether analyzing inflow balance in the center of the current sheet")
    parser.add_argument('--plot_vout', action="store_true", default=False,
                        help="whether plotting outflow velocity")
    parser.add_argument('--plot_density', action="store_true", default=False,
                        help="whether plotting number density")
    parser.add_argument('--rrate_vin', action="store_true", default=False,
                        help="whether calculating reconnection rate using inflow velocity")
    parser.add_argument('--rrate_bflux', action="store_true", default=False,
                        help="whether calculating reconnection rate using magnetic flux")
    parser.add_argument('--plot_rrate_bflux', action="store_true", default=False,
                        help="whether plotting reconnection rate using magnetic flux")
    parser.add_argument('--open_angle', action="store_true", default=False,
                        help='whether to plot exhaust open angle')
    parser.add_argument('--exhaust_boundary', action="store_true", default=False,
                        help='whether to get the boundary of reconnection exhaust')
    parser.add_argument('--inflow_pressure', action="store_true", default=False,
                        help="whether analyzing inflow pressure")
    parser.add_argument('--calc_bxm', action="store_true", default=False,
                        help="whether calculating Bxm")
    return parser.parse_args()


def analysis_single_frames(plot_config, args):
    """Analysis for multiple time frames
    """
    tframe = args.tframe
    if args.plot_absj:
        plot_absj(plot_config, args.show_plot)
    elif args.plot_bfield:
        plot_bfield(plot_config, args.show_plot)
    elif args.plot_efield:
        plot_efield(plot_config, args.show_plot)
    elif args.plot_ptensor:
        plot_pressure_tensor(plot_config, args.show_plot)
    elif args.plot_rrate:
        plot_reconnection_rate_2d(plot_config)
    elif args.inflow_balance:
        inflow_balance(plot_config, args.show_plot)
    elif args.outflow_balance:
        outflow_balance(plot_config, args.show_plot)
    elif args.outflow_balance_center:
        outflow_balance_center(plot_config, args.show_plot)
    elif args.plot_vout:
        plot_vout(plot_config, args.show_plot)
    elif args.plot_density:
        plot_density(plot_config, args.show_plot)
    elif args.rrate_vin:
        calc_rrate_vin(plot_config, args.show_plot)
    elif args.rrate_bflux:
        calc_rrate_bflux(plot_config, args.show_plot)
    elif args.plot_rrate_bflux:
        plot_rrate_bflux(plot_config, args.show_plot)
    elif args.open_angle:
        open_angle(plot_config, args.show_plot)
    elif args.exhaust_boundary:
        get_exhaust_boundary(plot_config, args.show_plot)
    elif args.inflow_pressure:
        inflow_pressure(plot_config, args.show_plot)
    elif args.calc_bxm:
        calc_bxm(plot_config, args.show_plot)


def process_input(plot_config, args, tframe):
    """process one time frame"""
    plot_config["tframe"] = tframe
    if args.rrate_bflux:
        calc_rrate_bflux(plot_config, args.show_plot)
    elif args.exhaust_boundary:
        get_exhaust_boundary(plot_config, args.show_plot)


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
            elif args.plot_bfield:
                plot_bfield(plot_config, show_plot=False)
            elif args.plot_ptensor:
                plot_pressure_tensor(plot_config, show_plot=False)
            elif args.plot_vout:
                plot_vout(plot_config, show_plot=False)
            elif args.plot_density:
                plot_density(plot_config, show_plot=False)
            elif args.inflow_balance:
                inflow_balance(plot_config, show_plot=False)
            elif args.outflow_balance:
                outflow_balance(plot_config, show_plot=False)
            elif args.plot_bfield:
                plot_bfield(plot_config, show_plot=False)
            elif args.plot_efield:
                plot_efield(plot_config, show_plot=False)
            elif args.inflow_pressure:
                inflow_pressure(plot_config, show_plot=False)
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
    plot_config["species"] = args.species
    if args.multi_frames:
        analysis_multi_frames(plot_config, args)
    else:
        analysis_single_frames(plot_config, args)


if __name__ == "__main__":
    main()
