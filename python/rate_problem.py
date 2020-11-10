#!/usr/bin/env python3
"""
Reconnection rate problem
"""
from __future__ import print_function

import argparse
import collections
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
from scipy.special import erf

import fitting_funcs
import pic_information
from contour_plots import read_2d_fields
from joblib import Parallel, delayed
from json_functions import read_data_from_json
from pic_information import get_variable_value
from shell_functions import mkdir_p

plt.style.use("seaborn-deep")
mpl.rc('text', usetex=True)
mpl.rcParams["text.latex.preamble"] = \
        (r"\usepackage{amsmath, bm}" +
         r"\DeclareMathAlphabet{\mathsfit}{\encodingdefault}{\sfdefault}{m}{sl}" +
         r"\SetMathAlphabet{\mathsfit}{bold}{\encodingdefault}{\sfdefault}{bx}{sl}" +
         r"\newcommand{\tensorsym}[1]{\bm{\mathsfit{#1}}}")
COLORS = palettable.colorbrewer.qualitative.Set1_9.mpl_colors


def div0(a, b):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0]

    From: http://stackoverflow.com/a/35696047/2561161

    """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[~np.isfinite(c)] = 0  # -inf inf NaN
    return c


def find_nearest(array, value):
    """Find nearest value in an array
    """
    idx = (np.abs(array-value)).argmin()
    return (idx, array[idx])


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


def get_bfield_pressure(plot_config, box=[0, 0, 1, 1]):
    """Get magnetic field and pressure tensor
    """
    tframe = plot_config["tframe"]
    pic_run = plot_config["pic_run"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_run_dir = pic_info.run_dir
    fields_interval = pic_info.fields_interval
    tindex = fields_interval * tframe
    vpic_info = get_vpic_info(pic_run_dir)
    vthe = vpic_info["vtheb/c"]
    n0 = vpic_info["n0"]
    pnorm = n0 * vthe**2
    if box == [0, 0, 1, 1]:
        xs, xe = 0, pic_info.nx
        zs, ze = 0, pic_info.nz
    else:
        xs, zs, xe, ze = box

    vecb_pre = {}

    # Magnetic field
    fname = (pic_run_dir + "field_hdf5/T." + str(tindex) +
             "/fields_" + str(tindex) + ".h5")
    with h5py.File(fname, 'r') as fh:
        group = fh["Timestep_" + str(tindex)]
        for var in ["cbx", "cby", "cbz"]:
            dset = group[var]
            var_name = var[1:]
            vecb_pre[var_name]= dset[xs:xe, 0, zs:ze]

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
                hydro[var]= dset[xs:xe, 0, zs:ze]

        irho = 1.0 / hydro["rho"]
        vx = hydro["jx"] * irho
        vy = hydro["jy"] * irho
        vz = hydro["jz"] * irho
        var = "v" + species
        vecb_pre[var+"x"] = vx
        vecb_pre[var+"y"] = vy
        vecb_pre[var+"z"] = vz
        vecb_pre["n"+species] = np.abs(hydro["rho"])
        vpar = "p" + species
        vecb_pre[vpar+"xx"] = hydro["txx"] - vx * hydro["px"]
        vecb_pre[vpar+"yy"] = hydro["tyy"] - vy * hydro["py"]
        vecb_pre[vpar+"zz"] = hydro["tzz"] - vz * hydro["pz"]
        vecb_pre[vpar+"yx"] = hydro["txy"] - vx * hydro["py"]
        vecb_pre[vpar+"xz"] = hydro["tzx"] - vz * hydro["px"]
        vecb_pre[vpar+"zy"] = hydro["tyz"] - vy * hydro["pz"]
        vecb_pre[vpar+"xy"] = hydro["txy"] - vy * hydro["px"]
        vecb_pre[vpar+"yz"] = hydro["tyz"] - vz * hydro["py"]
        vecb_pre[vpar+"zx"] = hydro["tzx"] - vx * hydro["pz"]

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


def plot_absj(plot_config, show_plot=True):
    """Plot current density
    """
    tframe = plot_config["tframe"]
    pic_run = plot_config["pic_run"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_run_dir = pic_info.run_dir
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
                    vmin=0, vmax=0.08,
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
        kwargs = {"current_time": tframe,
                  "xl": 0, "xr": pic_info.lx_di,
                  "zb": -0.5*pic_info.lz_di, "zt": 0.5*pic_info.lz_di}
        fname = pic_run_dir + "data/Ay.gda"
        x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
        ax.contour(xgrid, zgrid, Ay, colors='k', linewidths=0.5,
                   levels=np.linspace(np.min(Ay), np.max(Ay), 20))

    fdir = '../data/rate_problem/exhaust_boundary/' + pic_run + '/'
    fname = fdir + 'xz_top_' + str(tframe) + '.dat'
    xz = np.fromfile(fname).reshape([2, -1])
    xlist_top = xz[0, :]
    zlist_top = xz[1, :] + zmin
    fname = fdir + 'xz_bot_' + str(tframe) + '.dat'
    xz = np.fromfile(fname).reshape([2, -1])
    xlist_bot = xz[0, :]
    zlist_bot = xz[1, :] + zmin
    ax.plot(xlist_top, zlist_top, linewidth=1, color=COLORS[0], linestyle='-')
    ax.plot(xlist_bot, zlist_bot, linewidth=1, color=COLORS[0], linestyle='-')

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


def plot_jy(plot_config, show_plot=True):
    """Plot jy
    """
    tframe = plot_config["tframe"]
    pic_run = plot_config["pic_run"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_run_dir = pic_info.run_dir
    fields_interval = pic_info.fields_interval
    tindex = fields_interval * tframe
    smime = math.sqrt(pic_info.mime)
    lx_de = pic_info.lx_di * smime
    lz_de = pic_info.lz_di * smime
    xmin, xmax = 0, lx_de
    zmin, zmax = -0.5 * lz_de, 0.5 * lz_de

    fname = (pic_run_dir + "hydro_hdf5/T." + str(tindex) +
             "/hydro_electron_" + str(tindex) + ".h5")
    with h5py.File(fname, 'r') as fh:
        group = fh["Timestep_" + str(tindex)]
        dset = group["jy"]
        jey = np.zeros(dset.shape, dtype=dset.dtype)
        dset.read_direct(jey)

    fname = (pic_run_dir + "hydro_hdf5/T." + str(tindex) +
             "/hydro_ion_" + str(tindex) + ".h5")
    with h5py.File(fname, 'r') as fh:
        group = fh["Timestep_" + str(tindex)]
        dset = group["jy"]
        jiy = np.zeros(dset.shape, dtype=dset.dtype)
        dset.read_direct(jiy)

    xgrid = np.linspace(xmin, xmax, pic_info.nx)
    zgrid = np.linspace(zmin, zmax, pic_info.nz)

    jy = np.squeeze(jey + jiy)
    len0 = 10
    fig = plt.figure(figsize=[len0, len0*lz_de/lx_de])
    rect = [0.12, 0.14, 0.78, 0.78]
    ax = fig.add_axes(rect)
    im1 = ax.imshow(jy.T,
                    extent=[xmin, xmax, zmin, zmax],
                    vmin=-0.08, vmax=0.08,
                    cmap=plt.cm.seismic, aspect='auto',
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
        kwargs = {"current_time": tframe,
                  "xl": 0, "xr": pic_info.lx_di,
                  "zb": -0.5*pic_info.lz_di, "zt": 0.5*pic_info.lz_di}
        fname = pic_run_dir + "data/Ay.gda"
        x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
        ax.contour(xgrid, zgrid, Ay, colors='k', linewidths=0.5,
                   levels=np.linspace(np.min(Ay), np.max(Ay), 20))

    fdir = '../data/rate_problem/exhaust_boundary/' + pic_run + '/'
    fname = fdir + 'xz_top_' + str(tframe) + '.dat'
    xz = np.fromfile(fname).reshape([2, -1])
    xlist_top = xz[0, :]
    zlist_top = xz[1, :] + zmin
    fname = fdir + 'xz_bot_' + str(tframe) + '.dat'
    xz = np.fromfile(fname).reshape([2, -1])
    xlist_bot = xz[0, :]
    zlist_bot = xz[1, :] + zmin
    ax.plot(xlist_top, zlist_top, linewidth=1, color=COLORS[0], linestyle='-')
    ax.plot(xlist_bot, zlist_bot, linewidth=1, color=COLORS[0], linestyle='-')

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
    cbar = fig.colorbar(im1, cax=cbar_ax, extend='both')
    cbar_ax.tick_params(bottom=False, top=False, left=False, right=True)
    cbar_ax.tick_params(axis='y', which='major', direction='out')
    cbar_ax.tick_params(axis='y', which='minor', direction='in', right=False)
    cbar_ax.set_title(r'$j_y$', fontsize=20)
    cbar.ax.tick_params(labelsize=12)
    twpe = math.ceil(tindex * pic_info.dtwpe / 0.1) * 0.1
    text1 = r'$t\omega_{pe}=' + ("{%0.0f}" % twpe) + '$'
    fig.suptitle(text1, fontsize=20)
    img_dir = '../img/rate_problem/jy/' + pic_run + '/'
    mkdir_p(img_dir)
    fname = img_dir + "jy_" + str(tframe) + ".jpg"
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
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_run_dir = pic_info.run_dir
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


def plot_bz_xcut_beta(plot_config, show_plot=True):
    """Plot Bz along x for runs with different beta
    """
    pic_runs = ["mime400_Tb_T0_025",
                "mime400_nb_n0_1",
                "mime400_Tb_T0_10",
                "mime400_Tb_T0_40"]
    tframe = plot_config["tframe"]
    fig = plt.figure(figsize=[7, 5])
    rect = [0.15, 0.12, 0.8, 0.8]
    ax = fig.add_axes(rect)
    ax.set_prop_cycle('color', COLORS)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlabel(r'$x/d_e$', fontsize=16)
    ax.tick_params(labelsize=12)
    labels = [r"$\beta=0.25$", r"$\beta=1$", r"$\beta=10$", r"$\beta=40$"]
    for irun, pic_run in enumerate(pic_runs):
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        pic_run_dir = pic_info.run_dir
        fields_interval = pic_info.fields_interval
        tindex = fields_interval * tframe
        smime = math.sqrt(pic_info.mime)
        lx_de = pic_info.lx_di * smime
        lz_de = pic_info.lz_di * smime
        xmin, xmax = 0, lx_de
        zmin, zmax = -0.5 * lz_de, 0.5 * lz_de
        nx = pic_info.nx
        nz = pic_info.nz
        xgrid = np.linspace(xmin, xmax, nx)
        zgrid = np.linspace(zmin, zmax, nz)

        fname = (pic_run_dir + "field_hdf5/T." + str(tindex) +
                 "/fields_" + str(tindex) + ".h5")
        with h5py.File(fname, 'r') as fh:
            group = fh["Timestep_" + str(tindex)]
            dset = group["cbz"]
            bz = np.zeros(dset.shape, dtype=dset.dtype)
            dset.read_direct(bz)

        bz = np.squeeze(bz)
        fdata = gaussian_filter(bz[:, nz//2], sigma=5)
        ax.plot(xgrid, fdata, linewidth=2, label=labels[irun])
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([-0.12, 0.12])
    ax.set_ylabel(r'$B_z$', fontsize=16)
    ax.legend(loc=2, prop={'size': 16}, ncol=1,
              shadow=False, fancybox=False, frameon=False)
    twpe = math.ceil(tindex * pic_info.dtwpe / 0.1) * 0.1
    text1 = r'$t\omega_{pe}=' + ("{%0.0f}" % twpe) + '$'
    ax.set_title(text1, fontsize=20)
    img_dir = '../img/rate_problem/bz_xcut_beta/'
    mkdir_p(img_dir)
    fname = img_dir + "bz_xcut_beta_" + str(tframe) + ".pdf"
    fig.savefig(fname)

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_pres_xcut_beta(plot_config, show_plot=True):
    """Plot pressure along x for runs with different beta
    """
    pic_runs = ["mime400_Tb_T0_025",
                "mime400_nb_n0_1",
                "mime400_Tb_T0_10",
                "mime400_Tb_T0_40"]
    tframe = plot_config["tframe"]
    fig = plt.figure(figsize=[7, 8])
    rect = [0.15, 0.76, 0.8, 0.19]
    hgap, vgap = 0.02, 0.03
    nvar = 4
    axs = []
    for ivar in range(nvar):
        ax = fig.add_axes(rect)
        axs.append(ax)
        rect[1] -= rect[3] + vgap
        ax.set_prop_cycle('color', COLORS)
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')
        if ivar == nvar - 1:
            ax.set_xlabel(r'$x/d_e$', fontsize=16)
        ax.tick_params(labelsize=12)
    axs[0].set_ylabel(r'$\Delta P_{e\parallel}$', fontsize=16)
    axs[1].set_ylabel(r'$\Delta P_{e\perp}$', fontsize=16)
    axs[2].set_ylabel(r'$\Delta P_{i\parallel}$', fontsize=16)
    axs[3].set_ylabel(r'$\Delta P_{i\perp}$', fontsize=16)
    labels = [r"$\beta=0.25$", r"$\beta=1$", r"$\beta=10$", r"$\beta=40$"]
    for irun, pic_run in enumerate(pic_runs):
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        pic_run_dir = pic_info.run_dir
        fields_interval = pic_info.fields_interval
        tindex = fields_interval * tframe
        smime = math.sqrt(pic_info.mime)
        lx_de = pic_info.lx_di * smime
        lz_de = pic_info.lz_di * smime
        xmin, xmax = 0, lx_de
        zmin, zmax = -0.5 * lz_de, 0.5 * lz_de
        nx = pic_info.nx
        nz = pic_info.nz
        dx_de = lx_de / nx
        dz_de = lz_de / nz
        xgrid = np.linspace(xmin, xmax, nx)
        zgrid = np.linspace(zmin, zmax, nz)
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

        plot_config["pic_run"] = pic_run
        plot_config["pic_run_dir"] = pic_run_dir
        bvec_pre = get_bfield_pressure(plot_config)
        sigma = 5
        bvec_pre["pepara"] = gaussian_filter(np.squeeze(bvec_pre["pepara"]), sigma=sigma)
        bvec_pre["pipara"] = gaussian_filter(np.squeeze(bvec_pre["pipara"]), sigma=sigma)
        bvec_pre["peperp"] = gaussian_filter(np.squeeze(bvec_pre["peperp"]), sigma=sigma)
        bvec_pre["piperp"] = gaussian_filter(np.squeeze(bvec_pre["piperp"]), sigma=sigma)

        fdata = bvec_pre["pepara"][:, nz//2] - bvec_pre["pepara"][ix_xpoint, nz//2]
        axs[0].plot(xgrid, fdata, linewidth=2, label=labels[irun])
        fdata = bvec_pre["peperp"][:, nz//2] - bvec_pre["peperp"][ix_xpoint, nz//2]
        axs[1].plot(xgrid, fdata, linewidth=2, label=labels[irun])
        fdata = bvec_pre["pipara"][:, nz//2] - bvec_pre["pipara"][ix_xpoint, nz//2]
        axs[2].plot(xgrid, fdata, linewidth=2, label=labels[irun])
        fdata = bvec_pre["piperp"][:, nz//2] - bvec_pre["piperp"][ix_xpoint, nz//2]
        axs[3].plot(xgrid, fdata, linewidth=2, label=labels[irun])
    for ivar, ax in enumerate(axs):
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([-0.05, 0.25])
        ax.grid(True)
        if ivar == 0:
            ax.legend(loc=2, prop={'size': 16}, ncol=3,
                      shadow=False, fancybox=False, frameon=False)
    twpe = math.ceil(tindex * pic_info.dtwpe / 0.1) * 0.1
    text1 = r'$t\omega_{pe}=' + ("{%0.0f}" % twpe) + '$'
    fig.suptitle(text1, fontsize=20)
    img_dir = '../img/rate_problem/pres_xcut_beta/'
    mkdir_p(img_dir)
    fname = img_dir + "pres_xcut_beta_" + str(tframe) + ".pdf"
    fig.savefig(fname)

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_pres(plot_config, show_plot=True):
    """Plot pressure along
    """
    pic_run = plot_config["pic_run"]
    tframe = plot_config["tframe"]
    fig = plt.figure(figsize=[7, 7])
    rect = [0.12, 0.75, 0.77, 0.19]
    hgap, vgap = 0.02, 0.03
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_run_dir = pic_info.run_dir
    fields_interval = pic_info.fields_interval
    tindex = fields_interval * tframe
    smime = math.sqrt(pic_info.mime)
    lx_de = pic_info.lx_di * smime
    lz_de = pic_info.lz_di * smime
    nx = pic_info.nx
    nz = pic_info.nz
    xs = nx//2 - nx//4
    xe = nx//2 + nx//4
    zs = nz//2 - nz//16
    ze = nz//2 + nz//16
    dx_de = lx_de / nx
    dz_de = lz_de / nz
    xmin = xs * dz_de
    xmax = xe * dz_de
    zmin = zs * dz_de - lz_de * 0.5
    zmax = ze * dz_de - lz_de * 0.5

    vpic_info = get_vpic_info(pic_run_dir)
    n0 = vpic_info["n0"]
    b0 = vpic_info["b0"]
    Te = vpic_info["Te"]
    Tbe_Te = vpic_info["Tbe/Te"]
    nb_n0 = vpic_info["nb/n0"]
    nb = n0 * nb_n0
    p0 = nb * Te * Tbe_Te

    nvar = 4
    axs = []
    for ivar in range(nvar):
        ax = fig.add_axes(rect)
        axs.append(ax)
        rect[1] -= rect[3] + vgap
        ax.set_prop_cycle('color', COLORS)
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')
        if ivar == nvar - 1:
            ax.set_xlabel(r'$x/d_e$', fontsize=16)
        ax.set_ylabel(r'$z/d_e$', fontsize=16)
        ax.tick_params(labelsize=12)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([zmin, zmax])

    plot_config["pic_run"] = pic_run
    plot_config["pic_run_dir"] = pic_run_dir
    bvec_pre = get_bfield_pressure(plot_config, box=[xs, zs, xe, ze])
    sigma = 5
    bvec_pre["pepara"] = gaussian_filter(np.squeeze(bvec_pre["pepara"]), sigma=sigma)
    bvec_pre["pipara"] = gaussian_filter(np.squeeze(bvec_pre["pipara"]), sigma=sigma)
    bvec_pre["peperp"] = gaussian_filter(np.squeeze(bvec_pre["peperp"]), sigma=sigma)
    bvec_pre["piperp"] = gaussian_filter(np.squeeze(bvec_pre["piperp"]), sigma=sigma)

    if "mime400_Tb_T0_025" in pic_run:
        dmin, dmax = 1, 8
    elif "mime400_Tb_T0_10" in pic_run:
        dmin, dmax = 1, 1.2
    elif "mime400_nb_n0_1" in pic_run or "mime400_Tb_T0_1" in pic_run:
        dmin, dmax = 1, 4
    elif "mime400_Tb_T0_40" in pic_run:
        dmin, dmax = 1, 1.05
    cmap = plt.cm.jet
    im1 = axs[0].imshow(bvec_pre["pepara"].T / p0,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=dmin, vmax=dmax,
                        cmap=cmap, aspect='auto',
                        origin='lower', interpolation='bicubic')
    im2 = axs[1].imshow(bvec_pre["peperp"].T / p0,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=dmin, vmax=dmax,
                        cmap=cmap, aspect='auto',
                        origin='lower', interpolation='bicubic')
    im3 = axs[2].imshow(bvec_pre["pipara"].T / p0,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=dmin, vmax=dmax,
                        cmap=cmap, aspect='auto',
                        origin='lower', interpolation='bicubic')
    im4 = axs[3].imshow(bvec_pre["piperp"].T / p0,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=dmin, vmax=dmax,
                        cmap=cmap, aspect='auto',
                        origin='lower', interpolation='bicubic')

    rect_cbar = np.copy(rect)
    rect_cbar[0] += rect[2] + 0.01
    rect_cbar[2] = 0.02
    rect_cbar[1] += (rect[3] + vgap) * 2
    rect_cbar[3] = rect[3] * 2 + vgap
    cbar_ax = fig.add_axes(rect_cbar)
    cbar = fig.colorbar(im1, cax=cbar_ax, extend='both')
    cbar_ax.tick_params(bottom=False, top=False, left=False, right=True)
    cbar_ax.tick_params(axis='y', which='major', direction='out')
    cbar_ax.tick_params(axis='y', which='minor', direction='in', right=False)
    cbar.ax.tick_params(labelsize=12)

    xmin_di = xmin / smime
    xmax_di = xmax / smime
    zmin_di = zmin / smime
    zmax_di = zmax / smime
    kwargs = {"current_time": tframe,
              "xl": xmin_di, "xr": xmax_di,
              "zb": zmin_di, "zt": zmax_di}
    fname = pic_run_dir + "data/Ay.gda"
    x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
    xde = x * smime
    zde = z * smime
    texts = [r'$P_{e\parallel}$', r'$P_{e\perp}$',
             r'$P_{i\parallel}$', r'$P_{i\perp}$']
    for iax, ax in enumerate(axs):
        ax.contour(xde, zde, Ay, colors='w', linewidths=0.5,
                   levels=np.linspace(np.min(Ay), np.max(Ay), 10))
        ax.text(0.02, 0.85, texts[iax], color='w', fontsize=16,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)

    twpe = math.ceil(tindex * pic_info.dtwpe / 0.1) * 0.1
    text1 = r'$t\omega_{pe}=' + ("{%0.0f}" % twpe) + '$'
    fig.suptitle(text1, fontsize=20)
    img_dir = '../img/rate_problem/pres/' + pic_run + '/'
    mkdir_p(img_dir)
    fname = img_dir + "pres_" + str(tframe) + ".jpg"
    fig.savefig(fname, dpi=200)

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_pres_avg(plot_config, show_plot=True):
    """Plot the average pressure along x
    """
    pic_run = plot_config["pic_run"]
    tframe = plot_config["tframe"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_run_dir = pic_info.run_dir
    fields_interval = pic_info.fields_interval
    tindex = fields_interval * tframe
    smime = math.sqrt(pic_info.mime)
    lx_de = pic_info.lx_di * smime
    lz_de = pic_info.lz_di * smime
    nx = pic_info.nx
    nz = pic_info.nz
    xgrid = np.linspace(0, lx_de, nx)
    zgrid = np.linspace(-0.5*lz_de, lz_de, nz)
    xs = nx//2 - nx//4
    xe = nx//2 + nx//4
    zs = nz//2 - nz//16
    ze = nz//2 + nz//16
    nxs = xe - xs
    nzs = ze - zs
    dx_de = lx_de / nx
    dz_de = lz_de / nz
    xmin = xs * dz_de
    xmax = xe * dz_de
    zmin = zs * dz_de - lz_de * 0.5
    zmax = ze * dz_de - lz_de * 0.5

    vpic_info = get_vpic_info(pic_run_dir)
    n0 = vpic_info["n0"]
    b0 = vpic_info["b0"]
    Te = vpic_info["Te"]
    Tbe_Te = vpic_info["Tbe/Te"]
    nb_n0 = vpic_info["nb/n0"]
    nb = n0 * nb_n0
    p0 = nb * Te * Tbe_Te

    plot_config["pic_run"] = pic_run
    plot_config["pic_run_dir"] = pic_run_dir
    bvec_pre = get_bfield_pressure(plot_config, box=[xs, zs, xe, ze])
    sigma = 5
    bvec_pre["pepara"] = gaussian_filter(np.squeeze(bvec_pre["pepara"]), sigma=sigma)
    bvec_pre["pipara"] = gaussian_filter(np.squeeze(bvec_pre["pipara"]), sigma=sigma)
    bvec_pre["peperp"] = gaussian_filter(np.squeeze(bvec_pre["peperp"]), sigma=sigma)
    bvec_pre["piperp"] = gaussian_filter(np.squeeze(bvec_pre["piperp"]), sigma=sigma)
    bvec_pre["ne"] = gaussian_filter(np.squeeze(bvec_pre["ne"]), sigma=sigma)
    bvec_pre["ni"] = gaussian_filter(np.squeeze(bvec_pre["ni"]), sigma=sigma)

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
    ix_xp = int(x_xp / dx_de) - xs

    # Averaged pressure in the exhaust
    nvar = 4
    pres_avg = np.zeros([nvar, nxs])
    navg = np.zeros(nxs)
    bavg = np.zeros(nxs)
    absB = np.sqrt(bvec_pre["bx"]**2 + bvec_pre["by"]**2 + bvec_pre["bz"]**2)
    for ix in range(nxs):
        ixs = ix + xs
        iz1 = iz_bot[ixs] - zs
        iz2 = iz_top[ixs] - zs
        pres_avg[0, ix] = np.mean(bvec_pre["pepara"][ix, iz1:iz2+1])
        pres_avg[1, ix] = np.mean(bvec_pre["peperp"][ix, iz1:iz2+1])
        pres_avg[2, ix] = np.mean(bvec_pre["pipara"][ix, iz1:iz2+1])
        pres_avg[3, ix] = np.mean(bvec_pre["piperp"][ix, iz1:iz2+1])
        navg[ix] = np.mean(bvec_pre["ni"][ix, iz1:iz2+1])
        bavg[ix] = np.mean(absB[ix, iz1:iz2+1])

    fig = plt.figure(figsize=[7, 8])
    rect = [0.13, 0.54, 0.75, 0.41]
    hgap, vgap = 0.02, 0.04

    ax = fig.add_axes(rect)
    ax.set_prop_cycle('color', COLORS)
    xgrid = np.arange(xs, xe) * dx_de
    p1, = ax.plot(xgrid, pres_avg[2] - p0, label=r'$\Delta P_{i\parallel}$')
    p2, = ax.plot(xgrid, pres_avg[3] - p0, label=r'$\Delta P_{i\perp}$')
    ax1 = ax.twinx()
    ax1.plot(xgrid, navg - nb, label=r'$\Delta n_i$', color=COLORS[2])
    ax1.plot(xgrid, bavg, label=r'$B$', color=COLORS[3])

    ax.legend(loc=3, prop={'size': 16}, ncol=1,
              shadow=False, fancybox=False, frameon=False)
    ax1.legend(loc=4, prop={'size': 16}, ncol=1,
               shadow=False, fancybox=False, frameon=False)

    ax.set_ylabel(r'$\Delta P_i$', fontsize=16)
    ax1.set_ylabel(r'$\Delta n_i, B$', fontsize=16)
    ax.tick_params(bottom=True, top=True, left=True, right=False)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.tick_params(labelsize=12)
    ax1.tick_params(labelsize=12)
    ax.set_xlim([xmin, xmax])

    rect[1] -= rect[3] + vgap
    ax = fig.add_axes(rect)
    ax.set_prop_cycle('color', COLORS)
    xgrid = np.arange(xs, xe) * dx_de
    p1, = ax.plot(xgrid, pres_avg[0] - p0, label=r'$\Delta P_{e\parallel}$')
    p2, = ax.plot(xgrid, pres_avg[1] - p0, label=r'$\Delta P_{e\perp}$')
    ax1 = ax.twinx()
    ax1.plot(xgrid, navg - nb, label=r'$\Delta n_e$', color=COLORS[2])
    ax1.plot(xgrid, bavg, label=r'$B$', color=COLORS[3])

    ax.legend(loc=3, prop={'size': 16}, ncol=1,
              shadow=False, fancybox=False, frameon=False)
    ax1.legend(loc=4, prop={'size': 16}, ncol=1,
               shadow=False, fancybox=False, frameon=False)

    ax.set_xlabel(r'$x/d_e$', fontsize=16)
    ax.set_ylabel(r'$\Delta P_e$', fontsize=16)
    ax1.set_ylabel(r'$\Delta n_e, B$', fontsize=16)
    ax.tick_params(bottom=True, top=True, left=True, right=False)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.tick_params(labelsize=12)
    ax1.tick_params(labelsize=12)
    ax.set_xlim([xmin, xmax])

    twpe = math.ceil(tindex * pic_info.dtwpe / 0.1) * 0.1
    text1 = r'$t\omega_{pe}=' + ("{%0.0f}" % twpe) + '$'
    fig.suptitle(text1, fontsize=20)
    img_dir = '../img/rate_problem/pres_avg/' + pic_run + '/'
    mkdir_p(img_dir)
    fname = img_dir + "pres_" + str(tframe) + ".jpg"
    fig.savefig(fname, dpi=200)

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_pres_inflow_cut(plot_config, show_plot=True):
    """Plot the pressure cut along z
    """
    pic_run = plot_config["pic_run"]
    tframe = plot_config["tframe"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_run_dir = pic_info.run_dir
    fields_interval = pic_info.fields_interval
    tindex = fields_interval * tframe
    smime = math.sqrt(pic_info.mime)
    lx_de = pic_info.lx_di * smime
    lz_de = pic_info.lz_di * smime
    nx = pic_info.nx
    nz = pic_info.nz
    xgrid = np.linspace(0, lx_de, nx)
    zgrid = np.linspace(-0.5*lz_de, lz_de, nz)
    xs = nx//2 - nx//32
    xe = nx//2 + nx//32
    zs = nz//2 - nz//4
    ze = nz//2 + nz//4
    nxs = xe - xs
    nzs = ze - zs
    dx_de = lx_de / nx
    dz_de = lz_de / nz
    xmin = xs * dz_de
    xmax = xe * dz_de
    zmin = zs * dz_de - lz_de * 0.5
    zmax = ze * dz_de - lz_de * 0.5

    vpic_info = get_vpic_info(pic_run_dir)
    n0 = vpic_info["n0"]
    b0 = vpic_info["b0"]
    Te = vpic_info["Te"]
    Tbe_Te = vpic_info["Tbe/Te"]
    nb_n0 = vpic_info["nb/n0"]
    nb = n0 * nb_n0
    p0 = nb * Te * Tbe_Te

    plot_config["pic_run"] = pic_run
    plot_config["pic_run_dir"] = pic_run_dir
    bvec_pre = get_bfield_pressure(plot_config, box=[xs, zs, xe, ze])
    sigma = 5
    bvec_pre["pepara"] = gaussian_filter(np.squeeze(bvec_pre["pepara"]), sigma=sigma)
    bvec_pre["pipara"] = gaussian_filter(np.squeeze(bvec_pre["pipara"]), sigma=sigma)
    bvec_pre["peperp"] = gaussian_filter(np.squeeze(bvec_pre["peperp"]), sigma=sigma)
    bvec_pre["piperp"] = gaussian_filter(np.squeeze(bvec_pre["piperp"]), sigma=sigma)
    bvec_pre["ne"] = gaussian_filter(np.squeeze(bvec_pre["ne"]), sigma=sigma)
    bvec_pre["ni"] = gaussian_filter(np.squeeze(bvec_pre["ni"]), sigma=sigma)

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
    ix_xp = int(x_xp / dx_de) - xs

    # Averaged pressure in the exhaust
    nvar = 4
    pres_cut = np.zeros([nvar, nzs])
    ncut = np.zeros(nzs)
    bcut = np.zeros(nzs)
    absB = np.sqrt(bvec_pre["bx"]**2 + bvec_pre["by"]**2 + bvec_pre["bz"]**2)
    pres_cut[0] = bvec_pre["pepara"][ix_xp, :]
    pres_cut[1] = bvec_pre["peperp"][ix_xp, :]
    pres_cut[2] = bvec_pre["pipara"][ix_xp, :]
    pres_cut[3] = bvec_pre["piperp"][ix_xp, :]
    pixx = bvec_pre["pixx"][ix_xp, :]
    ncut = bvec_pre["ni"][ix_xp, :]
    bcut = absB[ix_xp, :]

    fig = plt.figure(figsize=[7, 8])
    rect = [0.13, 0.54, 0.75, 0.41]
    hgap, vgap = 0.02, 0.04

    ax = fig.add_axes(rect)
    ax.set_prop_cycle('color', COLORS)
    zgrid = np.arange(zs, ze) * dz_de - 0.5 * lx_de
    p1, = ax.plot(zgrid, pres_cut[2]/p0, label=r'$P_{i\parallel}/P_0$')
    p2, = ax.plot(zgrid, pres_cut[3]/p0, label=r'$P_{i\perp}/P_0$')
    ax1 = ax.twinx()
    ax1.plot(zgrid, ncut/nb, label=r'$n_i/n_b$', color=COLORS[2])
    # ax1.plot(zgrid, bcut/b0, label=r'$B/B_0$', color=COLORS[3])

    ax.legend(loc=2, prop={'size': 16}, ncol=1,
              shadow=False, fancybox=False, frameon=False)
    ax1.legend(loc=1, prop={'size': 16}, ncol=1,
               shadow=False, fancybox=False, frameon=False)

    ax.set_ylabel(r'$P_i/P_0$', fontsize=16)
    ax1.set_ylabel(r'$n_i/n_b, B/B_0$', fontsize=16)
    ax.tick_params(bottom=True, top=True, left=True, right=False)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.tick_params(labelsize=12)
    ax1.tick_params(labelsize=12)
    ax.set_xlim([zmin, zmax])

    rect[1] -= rect[3] + vgap
    ax = fig.add_axes(rect)
    ax.set_prop_cycle('color', COLORS)
    p1, = ax.plot(zgrid, pres_cut[0]/p0, label=r'$P_{i\parallel}/P_0$')
    p2, = ax.plot(zgrid, pres_cut[1]/p0, label=r'$P_{i\perp}/P_0$')
    ax1 = ax.twinx()
    ax1.plot(zgrid, ncut/nb, label=r'$n_i/n_b$', color=COLORS[2])
    # ax1.plot(zgrid, bcut/b0, label=r'$B/B_0$', color=COLORS[3])

    ax.legend(loc=2, prop={'size': 16}, ncol=1,
              shadow=False, fancybox=False, frameon=False)
    ax1.legend(loc=1, prop={'size': 16}, ncol=1,
               shadow=False, fancybox=False, frameon=False)

    ax.set_xlabel(r'$x/d_e$', fontsize=16)
    ax.set_ylabel(r'$P_e/P_0$', fontsize=16)
    ax1.set_ylabel(r'$n_e/n_b, B/B_0$', fontsize=16)
    ax.tick_params(bottom=True, top=True, left=True, right=False)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.tick_params(labelsize=12)
    ax1.tick_params(labelsize=12)
    ax.set_xlim([zmin, zmax])

    twpe = math.ceil(tindex * pic_info.dtwpe / 0.1) * 0.1
    text1 = r'$t\omega_{pe}=' + ("{%0.0f}" % twpe) + '$'
    fig.suptitle(text1, fontsize=20)
    img_dir = '../img/rate_problem/pres_cut/' + pic_run + '/'
    mkdir_p(img_dir)
    fname = img_dir + "pres_" + str(tframe) + ".pdf"
    fig.savefig(fname)

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_temperature(plot_config, show_plot=True):
    """Plot temperature
    """
    pic_run = plot_config["pic_run"]
    tframe = plot_config["tframe"]
    fig = plt.figure(figsize=[7, 7])
    rect = [0.12, 0.75, 0.75, 0.19]
    hgap, vgap = 0.02, 0.03
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_run_dir = pic_info.run_dir
    fields_interval = pic_info.fields_interval
    tindex = fields_interval * tframe
    smime = math.sqrt(pic_info.mime)
    lx_de = pic_info.lx_di * smime
    lz_de = pic_info.lz_di * smime
    nx = pic_info.nx
    nz = pic_info.nz
    xs = nx//2 - nx//4
    xe = nx//2 + nx//4
    zs = nz//2 - nz//16
    ze = nz//2 + nz//16
    dx_de = lx_de / nx
    dz_de = lz_de / nz
    xmin = xs * dz_de
    xmax = xe * dz_de
    zmin = zs * dz_de - lz_de * 0.5
    zmax = ze * dz_de - lz_de * 0.5

    vpic_info = get_vpic_info(pic_run_dir)
    n0 = vpic_info["n0"]
    b0 = vpic_info["b0"]
    Te = vpic_info["Te"]
    Tbe_Te = vpic_info["Tbe/Te"]
    nb_n0 = vpic_info["nb/n0"]
    nb = n0 * nb_n0
    p0 = nb * Te * Tbe_Te
    Tb = Te * Tbe_Te

    nvar = 4
    axs = []
    for ivar in range(nvar):
        ax = fig.add_axes(rect)
        axs.append(ax)
        rect[1] -= rect[3] + vgap
        ax.set_prop_cycle('color', COLORS)
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')
        if ivar == nvar - 1:
            ax.set_xlabel(r'$x/d_e$', fontsize=16)
        ax.set_ylabel(r'$z/d_e$', fontsize=16)
        ax.tick_params(labelsize=12)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([zmin, zmax])

    plot_config["pic_run"] = pic_run
    plot_config["pic_run_dir"] = pic_run_dir
    bvec_pre = get_bfield_pressure(plot_config, box=[xs, zs, xe, ze])
    sigma = 5
    bvec_pre["ne"] = gaussian_filter(np.squeeze(bvec_pre["ne"]), sigma=sigma)
    bvec_pre["ni"] = gaussian_filter(np.squeeze(bvec_pre["ni"]), sigma=sigma)
    bvec_pre["pepara"] = gaussian_filter(np.squeeze(bvec_pre["pepara"]), sigma=sigma)
    bvec_pre["pipara"] = gaussian_filter(np.squeeze(bvec_pre["pipara"]), sigma=sigma)
    bvec_pre["peperp"] = gaussian_filter(np.squeeze(bvec_pre["peperp"]), sigma=sigma)
    bvec_pre["piperp"] = gaussian_filter(np.squeeze(bvec_pre["piperp"]), sigma=sigma)

    if pic_run == "mime400_Tb_T0_025":
        dmin, dmax = -0.05, 0.05
    elif pic_run == "mime400_nb_n0_1" or pic_run == "mime400_Tb_T0_1":
        dmin, dmax = -0.05, 0.05
    elif "mime400_Tb_T0_10" in pic_run:
        dmin, dmax = -0.05, 0.05
    elif "mime400_Tb_T0_40" in pic_run:
        dmin, dmax = -0.05, 0.05
    cmap = plt.cm.coolwarm
    fdata = bvec_pre["pepara"][xs:xe, zs:ze].T / bvec_pre["ne"][xs:xe, zs:ze].T
    im1 = axs[0].imshow(fdata - Tb,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=dmin, vmax=dmax,
                        cmap=cmap, aspect='auto',
                        origin='lower', interpolation='bicubic')
    fdata = bvec_pre["peperp"][xs:xe, zs:ze].T / bvec_pre["ne"][xs:xe, zs:ze].T
    im2 = axs[1].imshow(fdata - Tb,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=dmin, vmax=dmax,
                        cmap=cmap, aspect='auto',
                        origin='lower', interpolation='bicubic')
    fdata = bvec_pre["pipara"][xs:xe, zs:ze].T / bvec_pre["ni"][xs:xe, zs:ze].T
    im3 = axs[2].imshow(fdata - Tb,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=dmin, vmax=dmax,
                        cmap=cmap, aspect='auto',
                        origin='lower', interpolation='bicubic')
    fdata = bvec_pre["piperp"][xs:xe, zs:ze].T / bvec_pre["ni"][xs:xe, zs:ze].T
    im4 = axs[3].imshow(fdata - Tb,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=dmin, vmax=dmax,
                        cmap=cmap, aspect='auto',
                        origin='lower', interpolation='bicubic')

    rect_cbar = np.copy(rect)
    rect_cbar[0] += rect[2] + 0.01
    rect_cbar[2] = 0.02
    rect_cbar[1] += (rect[3] + vgap) * 2
    rect_cbar[3] = rect[3] * 2 + vgap
    cbar_ax = fig.add_axes(rect_cbar)
    cbar = fig.colorbar(im1, cax=cbar_ax, extend='both')
    cbar_ax.tick_params(bottom=False, top=False, left=False, right=True)
    cbar_ax.tick_params(axis='y', which='major', direction='out')
    cbar_ax.tick_params(axis='y', which='minor', direction='in', right=False)
    cbar.ax.tick_params(labelsize=12)

    xmin_di = xmin / smime
    xmax_di = xmax / smime
    zmin_di = zmin / smime
    zmax_di = zmax / smime
    kwargs = {"current_time": tframe,
              "xl": xmin_di, "xr": xmax_di,
              "zb": zmin_di, "zt": zmax_di}
    fname = pic_run_dir + "data/Ay.gda"
    x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
    xde = x * smime
    zde = z * smime
    texts = [r'$T_{e\parallel}-T_b$', r'$T_{e\perp}-T_b$',
             r'$T_{i\parallel}-T_b$', r'$T_{i\perp}-T_b$']
    for iax, ax in enumerate(axs):
        ax.contour(xde, zde, Ay, colors='grey', linewidths=0.5,
                   levels=np.linspace(np.min(Ay), np.max(Ay), 10))
        ax.text(0.02, 0.85, texts[iax], color='k', fontsize=16,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)

    # twpe = math.ceil(tindex * pic_info.dtwpe / 0.1) * 0.1
    # text1 = r'$t\omega_{pe}=' + ("{%0.0f}" % twpe) + '$'
    # fig.suptitle(text1, fontsize=20)
    # img_dir = '../img/rate_problem/pres/' + pic_run + '/'
    # mkdir_p(img_dir)
    # fname = img_dir + "pres_" + str(tframe) + ".jpg"
    # fig.savefig(fname, dpi=200)

    if show_plot:
        plt.show()
    else:
        plt.close()


def gradx_pressure(plot_config, show_plot=True):
    """Plot the gradient of temperature along x-direction
    """
    pic_run = plot_config["pic_run"]
    tframe = plot_config["tframe"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_run_dir = pic_info.run_dir
    fields_interval = pic_info.fields_interval
    tindex = fields_interval * tframe
    smime = math.sqrt(pic_info.mime)
    lx_de = pic_info.lx_di * smime
    lz_de = pic_info.lz_di * smime
    nx = pic_info.nx
    nz = pic_info.nz
    xs = nx//2 - nx//4
    xe = nx//2 + nx//4
    zs = nz//2 - nz//16
    ze = nz//2 + nz//16
    dx_de = lx_de / nx
    dz_de = lz_de / nz
    xmin = xs * dz_de
    xmax = xe * dz_de
    zmin = zs * dz_de - lz_de * 0.5
    zmax = ze * dz_de - lz_de * 0.5

    vpic_info = get_vpic_info(pic_run_dir)
    n0 = vpic_info["n0"]
    b0 = vpic_info["b0"]
    Te = vpic_info["Te"]
    Tbe_Te = vpic_info["Tbe/Te"]
    nb_n0 = vpic_info["nb/n0"]
    nb = n0 * nb_n0
    p0 = nb * Te * Tbe_Te
    Tb = Te * Tbe_Te

    fig = plt.figure(figsize=[14, 7])
    rect0 = [0.06, 0.75, 0.4, 0.19]
    hgap, vgap = 0.04, 0.03
    nvar = 4
    axs = []
    for col in range(2):
        rect = np.copy(rect0)
        rect[0] += (rect[2] + hgap) * col
        for ivar in range(nvar):
            ax = fig.add_axes(rect)
            axs.append(ax)
            rect[1] -= rect[3] + vgap
            ax.set_prop_cycle('color', COLORS)
            ax.tick_params(bottom=True, top=True, left=True, right=True)
            ax.tick_params(axis='x', which='minor', direction='in')
            ax.tick_params(axis='x', which='major', direction='in')
            ax.tick_params(axis='y', which='minor', direction='in')
            ax.tick_params(axis='y', which='major', direction='in')
            if ivar == nvar - 1:
                ax.set_xlabel(r'$x/d_e$', fontsize=16)
            if col == 0:
                ax.set_ylabel(r'$z/d_e$', fontsize=16)
            ax.tick_params(labelsize=12)
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([zmin, zmax])

    plot_config["pic_run"] = pic_run
    plot_config["pic_run_dir"] = pic_run_dir
    bvec_pre = get_bfield_pressure(plot_config, box=[xs, zs, xe, ze])
    sigma = 5
    bvec_pre["ne"] = gaussian_filter(np.squeeze(bvec_pre["ne"]), sigma=sigma)
    bvec_pre["ni"] = gaussian_filter(np.squeeze(bvec_pre["ni"]), sigma=sigma)
    bvec_pre["pepara"] = gaussian_filter(np.squeeze(bvec_pre["pepara"]), sigma=sigma)
    bvec_pre["pipara"] = gaussian_filter(np.squeeze(bvec_pre["pipara"]), sigma=sigma)
    bvec_pre["peperp"] = gaussian_filter(np.squeeze(bvec_pre["peperp"]), sigma=sigma)
    bvec_pre["piperp"] = gaussian_filter(np.squeeze(bvec_pre["piperp"]), sigma=sigma)
    Te_para = bvec_pre["pepara"] / bvec_pre["ne"]
    Te_perp = bvec_pre["peperp"] / bvec_pre["ne"]
    Ti_para = bvec_pre["pipara"] / bvec_pre["ni"]
    Ti_perp = bvec_pre["piperp"] / bvec_pre["ni"]

    if pic_run == "mime400_Tb_T0_025":
        dmin, dmax = -0.01, 0.01
    elif pic_run == "mime400_nb_n0_1" or pic_run == "mime400_Tb_T0_1":
        dmin, dmax = -0.01, 0.01
    elif "mime400_Tb_T0_10" in pic_run:
        dmin, dmax = -0.01, 0.01
    elif "mime400_Tb_T0_40" in pic_run:
        dmin, dmax = -0.01, 0.01
    cmap = plt.cm.coolwarm
    fdata = (np.gradient(Te_para[xs:xe, zs:ze], dx_de, axis=0) *
             bvec_pre["ne"][xs:xe, zs:ze] /
             bvec_pre["pepara"][xs:xe, zs:ze])
    im1 = axs[0].imshow(fdata.T,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=dmin, vmax=dmax,
                        cmap=cmap, aspect='auto',
                        origin='lower', interpolation='bicubic')
    fdata = (Te_para[xs:xe, zs:ze] *
             np.gradient(bvec_pre["ne"][xs:xe, zs:ze], dx_de, axis=0) /
             bvec_pre["pepara"][xs:xe, zs:ze])
    im2 = axs[1].imshow(fdata.T,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=dmin, vmax=dmax,
                        cmap=cmap, aspect='auto',
                        origin='lower', interpolation='bicubic')
    fdata = (np.gradient(Te_perp[xs:xe, zs:ze], dx_de, axis=0) *
             bvec_pre["ne"][xs:xe, zs:ze] /
             bvec_pre["peperp"][xs:xe, zs:ze])
    im3 = axs[2].imshow(fdata.T,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=dmin, vmax=dmax,
                        cmap=cmap, aspect='auto',
                        origin='lower', interpolation='bicubic')
    fdata = (Te_perp[xs:xe, zs:ze] *
             np.gradient(bvec_pre["ne"][xs:xe, zs:ze], dx_de, axis=0) /
             bvec_pre["peperp"][xs:xe, zs:ze])
    im4 = axs[3].imshow(fdata.T,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=dmin, vmax=dmax,
                        cmap=cmap, aspect='auto',
                        origin='lower', interpolation='bicubic')

    fdata = (np.gradient(Ti_para[xs:xe, zs:ze], dx_de, axis=0) *
             bvec_pre["ni"][xs:xe, zs:ze] /
             bvec_pre["pipara"][xs:xe, zs:ze])
    im1 = axs[4].imshow(fdata.T,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=dmin, vmax=dmax,
                        cmap=cmap, aspect='auto',
                        origin='lower', interpolation='bicubic')
    fdata = (Ti_para[xs:xe, zs:ze] *
             np.gradient(bvec_pre["ni"][xs:xe, zs:ze], dx_de, axis=0) /
             bvec_pre["pipara"][xs:xe, zs:ze])
    im2 = axs[5].imshow(fdata.T,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=dmin, vmax=dmax,
                        cmap=cmap, aspect='auto',
                        origin='lower', interpolation='bicubic')
    fdata = (np.gradient(Ti_perp[xs:xe, zs:ze], dx_de, axis=0) *
             bvec_pre["ni"][xs:xe, zs:ze] /
             bvec_pre["piperp"][xs:xe, zs:ze])
    im3 = axs[6].imshow(fdata.T,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=dmin, vmax=dmax,
                        cmap=cmap, aspect='auto',
                        origin='lower', interpolation='bicubic')
    fdata = (Ti_perp[xs:xe, zs:ze] *
             np.gradient(bvec_pre["ni"][xs:xe, zs:ze], dx_de, axis=0) /
             bvec_pre["piperp"][xs:xe, zs:ze])
    im4 = axs[7].imshow(fdata.T,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=dmin, vmax=dmax,
                        cmap=cmap, aspect='auto',
                        origin='lower', interpolation='bicubic')

    rect_cbar = np.copy(rect)
    rect_cbar[0] += rect[2] + 0.01
    rect_cbar[2] = 0.01
    rect_cbar[1] += (rect[3] + vgap) * 2
    rect_cbar[3] = rect[3] * 2 + vgap
    cbar_ax = fig.add_axes(rect_cbar)
    cbar = fig.colorbar(im1, cax=cbar_ax, extend='both')
    cbar_ax.tick_params(bottom=False, top=False, left=False, right=True)
    cbar_ax.tick_params(axis='y', which='major', direction='out')
    cbar_ax.tick_params(axis='y', which='minor', direction='in', right=False)
    cbar.ax.tick_params(labelsize=12)

    xmin_di = xmin / smime
    xmax_di = xmax / smime
    zmin_di = zmin / smime
    zmax_di = zmax / smime
    kwargs = {"current_time": tframe,
              "xl": xmin_di, "xr": xmax_di,
              "zb": zmin_di, "zt": zmax_di}
    fname = pic_run_dir + "data/Ay.gda"
    x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
    xde = x * smime
    zde = z * smime
    texts = [r'$n_e\nabla T_{e\parallel}$',
             r'$\nabla n_eT_{e\parallel}$',
             r'$n_e\nabla T_{e\perp}$',
             r'$\nabla n_eT_{e\perp}$',
             r'$n_i\nabla T_{i\parallel}$',
             r'$\nabla n_iT_{i\parallel}$',
             r'$n_i\nabla T_{i\perp}$',
             r'$\nabla n_iT_{i\perp}$']
    for iax, ax in enumerate(axs):
        ax.contour(xde, zde, Ay, colors='grey', linewidths=0.5,
                   levels=np.linspace(np.min(Ay), np.max(Ay), 10))
        ax.text(0.02, 0.85, texts[iax], color='k', fontsize=16,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)

    twpe = math.ceil(tindex * pic_info.dtwpe / 0.1) * 0.1
    text1 = r'$t\omega_{pe}=' + ("{%0.0f}" % twpe) + '$'
    fig.suptitle(text1, fontsize=20)
    img_dir = '../img/rate_problem/gradx_p/' + pic_run + '/'
    mkdir_p(img_dir)
    fname = img_dir + "gradx_p_" + str(tframe) + ".jpg"
    fig.savefig(fname, dpi=200)

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_density_xcut_beta(plot_config, show_plot=True):
    """Plot number density along x for runs with different beta
    """
    pic_runs = ["mime400_Tb_T0_025",
                "mime400_nb_n0_1",
                "mime400_Tb_T0_10",
                "mime400_Tb_T0_40"]
    tframe = plot_config["tframe"]
    fig = plt.figure(figsize=[7, 5])
    rect = [0.15, 0.12, 0.8, 0.8]
    ax = fig.add_axes(rect)
    ax.set_prop_cycle('color', COLORS)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlabel(r'$x/d_e$', fontsize=16)
    ax.tick_params(labelsize=12)
    labels = [r"$\beta=0.25$", r"$\beta=1$", r"$\beta=10$", r"$\beta=40$"]
    for irun, pic_run in enumerate(pic_runs):
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        pic_run_dir = pic_info.run_dir
        fields_interval = pic_info.fields_interval
        tindex = fields_interval * tframe
        smime = math.sqrt(pic_info.mime)
        lx_de = pic_info.lx_di * smime
        lz_de = pic_info.lz_di * smime
        xmin, xmax = 0, lx_de
        zmin, zmax = -0.5 * lz_de, 0.5 * lz_de
        nx = pic_info.nx
        nz = pic_info.nz
        xgrid = np.linspace(xmin, xmax, nx)
        zgrid = np.linspace(zmin, zmax, nz)

        fname = (pic_run_dir + "hydro_hdf5/T." + str(tindex) +
                 "/hydro_ion_" + str(tindex) + ".h5")
        with h5py.File(fname, 'r') as fh:
            group = fh["Timestep_" + str(tindex)]
            dset = group["rho"]
            ni = np.zeros(dset.shape, dtype=dset.dtype)
            dset.read_direct(ni)
        ni = gaussian_filter(np.squeeze(ni), sigma=5)
        fdata = ni[:, nz//2]
        ax.plot(xgrid, fdata, linewidth=2, label=labels[irun])
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([0.5, 3])
    ax.grid(True)
    ax.set_ylabel(r'$n_i$', fontsize=16)
    ax.legend(loc=3, prop={'size': 16}, ncol=1,
              shadow=False, fancybox=False, frameon=False)
    twpe = math.ceil(tindex * pic_info.dtwpe / 0.1) * 0.1
    text1 = r'$t\omega_{pe}=' + ("{%0.0f}" % twpe) + '$'
    ax.set_title(text1, fontsize=20)
    img_dir = '../img/rate_problem/n_xcut_beta/'
    mkdir_p(img_dir)
    fname = img_dir + "n_xcut_beta_" + str(tframe) + ".pdf"
    fig.savefig(fname)

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_pn_xcut_beta(plot_config, show_plot=True):
    """Plot number density along x for runs with different beta
    """
    pic_runs = ["mime400_Tb_T0_025",
                "mime400_nb_n0_1",
                "mime400_Tb_T0_10",
                "mime400_Tb_T0_40"]
    tframe = plot_config["tframe"]
    fig = plt.figure(figsize=[7, 5])
    rect = [0.15, 0.12, 0.8, 0.8]
    ax = fig.add_axes(rect)
    ax.set_prop_cycle('color', COLORS)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlabel(r'$x/d_e$', fontsize=16)
    ax.tick_params(labelsize=12)
    labels = [r"$\beta=0.25$", r"$\beta=1$", r"$\beta=10$", r"$\beta=40$"]
    for irun, pic_run in enumerate(pic_runs):
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        pic_run_dir = pic_info.run_dir
        fields_interval = pic_info.fields_interval
        tindex = fields_interval * tframe
        smime = math.sqrt(pic_info.mime)
        lx_de = pic_info.lx_di * smime
        lz_de = pic_info.lz_di * smime
        xmin, xmax = 0, lx_de
        zmin, zmax = -0.5 * lz_de, 0.5 * lz_de
        nx = pic_info.nx
        nz = pic_info.nz
        xgrid = np.linspace(xmin, xmax, nx)
        zgrid = np.linspace(zmin, zmax, nz)
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
        dx_de = lx_de / nx
        dz_de = lz_de / nz
        ix_xpoint = int(x0 / dx_de)

        plot_config["pic_run"] = pic_run
        plot_config["pic_run_dir"] = pic_run_dir
        bvec_pre = get_bfield_pressure(plot_config)
        sigma = 5
        bvec_pre["piperp"] = gaussian_filter(np.squeeze(bvec_pre["piperp"]), sigma=sigma)
        bvec_pre["ni"] = gaussian_filter(np.squeeze(bvec_pre["ni"]), sigma=sigma)
        n0 = bvec_pre["ni"][ix_xpoint, nz//2]
        p0 = bvec_pre["piperp"][ix_xpoint, nz//2]
        fdata = bvec_pre["ni"][:, nz//2] * bvec_pre["piperp"][ix_xpoint, nz//2] / n0
        fdata -= bvec_pre["piperp"][ix_xpoint, nz//2]
        ax.plot(xgrid, fdata, linewidth=2, label=labels[irun])
    ax.set_xlim([xmin, xmax])
    # ax.set_ylim([0.5, 3])
    ax.grid(True)
    ax.set_ylabel(r'$p_x\Delta n_i$', fontsize=16)
    ax.legend(loc=3, prop={'size': 16}, ncol=1,
              shadow=False, fancybox=False, frameon=False)
    twpe = math.ceil(tindex * pic_info.dtwpe / 0.1) * 0.1
    text1 = r'$t\omega_{pe}=' + ("{%0.0f}" % twpe) + '$'
    ax.set_title(text1, fontsize=20)
    # img_dir = '../img/rate_problem/pn_xcut_beta/'
    # mkdir_p(img_dir)
    # fname = img_dir + "pn_xcut_beta_" + str(tframe) + ".pdf"
    # fig.savefig(fname)

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_bx_zcut_beta(plot_config, show_plot=True):
    """Plot Bx along z for runs with different beta
    """
    pic_runs = ["mime400_Tb_T0_025",
                "mime400_nb_n0_1",
                "mime400_Tb_T0_10",
                "mime400_Tb_T0_40"]
    tframe = plot_config["tframe"]
    fig = plt.figure(figsize=[7, 10])
    rect = [0.15, 0.8, 0.8, 0.15]
    hgap, vgap = 0.02, 0.03
    ncut = 5
    axs = []
    for icut in range(ncut):
        ax = fig.add_axes(rect)
        axs.append(ax)
        rect[1] -= rect[3] + vgap
        ax.set_prop_cycle('color', COLORS)
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')
        if icut == ncut - 1:
            ax.set_xlabel(r'$z/d_e$', fontsize=16)
        ax.set_ylabel(r'$B_x$', fontsize=16)
        ax.tick_params(labelsize=12)
    labels = [r"$\beta=0.25$", r"$\beta=1$", r"$\beta=10$", r"$\beta=40$"]
    for irun, pic_run in enumerate(pic_runs):
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        pic_run_dir = pic_info.run_dir
        fields_interval = pic_info.fields_interval
        tindex = fields_interval * tframe
        smime = math.sqrt(pic_info.mime)
        lx_de = pic_info.lx_di * smime
        lz_de = pic_info.lz_di * smime
        xmin, xmax = 0, lx_de
        zmin, zmax = -0.5 * lz_de, 0.5 * lz_de
        nx = pic_info.nx
        nz = pic_info.nz
        xgrid = np.linspace(xmin, xmax, nx)
        zgrid = np.linspace(zmin, zmax, nz)
        nz_di = int(pic_info.nz / pic_info.lz_di)
        dx_de = lx_de / nx
        dz_de = lz_de / nz

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

        fname = (pic_run_dir + "field_hdf5/T." + str(tindex) +
                 "/fields_" + str(tindex) + ".h5")
        with h5py.File(fname, 'r') as fh:
            group = fh["Timestep_" + str(tindex)]
            dset = group["cbx"]
            bx = np.zeros(dset.shape, dtype=dset.dtype)
            dset.read_direct(bx)
        bx = np.squeeze(bx)

        for icut in range(ncut):
            ix = ix_xpoint + icut * nz_di * 2
            fdata = gaussian_filter(bx[ix, :], sigma=5)
            axs[icut].plot(zgrid, fdata, linewidth=2, label=labels[irun])
    for icut, ax in enumerate(axs):
        if icut == ncut//2:
            ax.legend(loc=4, prop={'size': 16}, ncol=1,
                      shadow=False, fancybox=False, frameon=False)
        text1 = r"$\Delta x=" + str(icut*2) + "d_i$"
        ax.text(0.02, 0.85, text1, color='k', fontsize=16,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)
        ax.set_xlim([zmin*0.1, zmax*0.1])
        ax.set_ylim([-0.55, 0.55])
    twpe = math.ceil(tindex * pic_info.dtwpe / 0.1) * 0.1
    text1 = r'$t\omega_{pe}=' + ("{%0.0f}" % twpe) + '$'
    fig.suptitle(text1, fontsize=20)
    img_dir = '../img/rate_problem/bx_zcut_beta/'
    mkdir_p(img_dir)
    fname = img_dir + "bx_zcut_beta_" + str(tframe) + ".pdf"
    fig.savefig(fname)

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_vx_zcut_beta(plot_config, show_plot=True):
    """Plot outflow velocity along z for runs with different beta
    """
    pic_runs = ["mime400_Tb_T0_025",
                "mime400_nb_n0_1",
                "mime400_Tb_T0_10",
                "mime400_Tb_T0_40"]
    tframe = plot_config["tframe"]
    fig = plt.figure(figsize=[7, 10])
    rect = [0.15, 0.8, 0.8, 0.15]
    hgap, vgap = 0.02, 0.03
    ncut = 5
    axs = []
    for icut in range(ncut):
        ax = fig.add_axes(rect)
        axs.append(ax)
        rect[1] -= rect[3] + vgap
        ax.set_prop_cycle('color', COLORS)
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')
        if icut == ncut - 1:
            ax.set_xlabel(r'$z/d_e$', fontsize=16)
        ax.set_ylabel(r'$V_x$', fontsize=16)
        ax.tick_params(labelsize=12)
    labels = [r"$\beta=0.25$", r"$\beta=1$", r"$\beta=10$", r"$\beta=40$"]
    for irun, pic_run in enumerate(pic_runs):
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        pic_run_dir = pic_info.run_dir
        fields_interval = pic_info.fields_interval
        tindex = fields_interval * tframe
        mime = pic_info.mime
        smime = math.sqrt(mime)
        lx_de = pic_info.lx_di * smime
        lz_de = pic_info.lz_di * smime
        xmin, xmax = 0, lx_de
        zmin, zmax = -0.5 * lz_de, 0.5 * lz_de
        nx = pic_info.nx
        nz = pic_info.nz
        xgrid = np.linspace(xmin, xmax, nx)
        zgrid = np.linspace(zmin, zmax, nz)
        nz_di = int(pic_info.nz / pic_info.lz_di)
        dx_de = lx_de / nx
        dz_de = lz_de / nz

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
            var = "v" + species
            rho_vel[var+"x"] = np.squeeze(vx)
            rho_vel["n"+species] = np.squeeze(np.abs(hydro["rho"]))

        irho = 1.0 / (rho_vel["ne"] + rho_vel["ni"] * mime)
        vsx = (rho_vel["ne"] * rho_vel["vex"] +
               rho_vel["ni"] * rho_vel["vix"] * mime) * irho

        dtwpe = pic_info.dtwpe
        dtwce = pic_info.dtwce
        vpic_info = get_vpic_info(pic_run_dir)
        nb_n0 = vpic_info["nb/n0"]
        va = dtwce * math.sqrt(1.0 / mime) / dtwpe / math.sqrt(nb_n0)
        for icut in range(ncut):
            ix = ix_xpoint + icut * nz_di * 2
            fdata = gaussian_filter(vsx[ix, :], sigma=5) / va
            axs[icut].plot(zgrid, fdata, linewidth=2, label=labels[irun])
    for icut, ax in enumerate(axs):
        if icut == ncut//2:
            ax.legend(loc=4, prop={'size': 16}, ncol=1,
                      shadow=False, fancybox=False, frameon=False)
        text1 = r"$\Delta x=" + str(icut*2) + "d_i$"
        ax.text(0.02, 0.85, text1, color='k', fontsize=16,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)
        ax.set_xlim([zmin*0.1, zmax*0.1])
        ax.set_ylim([-0.25, 0.25])
    twpe = math.ceil(tindex * pic_info.dtwpe / 0.1) * 0.1
    text1 = r'$t\omega_{pe}=' + ("{%0.0f}" % twpe) + '$'
    fig.suptitle(text1, fontsize=20)
    img_dir = '../img/rate_problem/vx_zcut_beta/'
    mkdir_p(img_dir)
    fname = img_dir + "vx_zcut_beta_" + str(tframe) + ".pdf"
    fig.savefig(fname)

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
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_run_dir = pic_info.run_dir
    ntf = pic_info.ntf
    phi = np.zeros(ntf)
    fname = pic_run_dir + 'data/Ay.gda'
    for tframe in range(ntf):
        kwargs = {"current_time": tframe,
                  "xl": 0, "xr": pic_info.lx_di,
                  "zb": -pic_info.lz_di*0.5, "zt": pic_info.lz_di*0.5}
        x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
        nz, = z.shape
        # max_ay = np.max(Ay[nz // 2 - 1:nz // 2 + 2, :])
        # min_ay = np.min(Ay[nz // 2 - 1:nz // 2 + 2, :])
        max_ay = np.max(Ay[nz // 2, :])
        min_ay = np.min(Ay[nz // 2, :])
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


def inflow_balance(plot_config, show_plot=True):
    """Force balance in the inflow region
    """
    tframe = plot_config["tframe"]
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_run_dir = pic_info.run_dir
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
    sigma = 3
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
    bvec_pre["pexx"] = gaussian_filter(bvec_pre["pexx"], sigma=sigma)
    bvec_pre["pexz"] = gaussian_filter(bvec_pre["pexz"], sigma=sigma)
    bvec_pre["pixx"] = gaussian_filter(bvec_pre["pixx"], sigma=sigma)
    bvec_pre["pixz"] = gaussian_filter(bvec_pre["pixz"], sigma=sigma)
    b2 = bvec_pre["bx"]**2 + bvec_pre["by"]**2 + bvec_pre["bz"]**2

    # Anisotropy parameter
    ppara = bvec_pre["pepara"] + bvec_pre["pipara"]
    pperp = bvec_pre["peperp"] + bvec_pre["piperp"]
    epsilon = 1 - (ppara - pperp) / b2

    tension = epsilon * (np.gradient(bvec_pre["bx"], axis=0) * bvec_pre["bx"] / dx_de +
                         np.gradient(bvec_pre["bx"], axis=1) * bvec_pre["bz"] / dz_de)
    tension += (np.gradient(epsilon, axis=0) * bvec_pre["bx"] / dx_de +
                np.gradient(epsilon, axis=1) * bvec_pre["bz"] / dz_de) * bvec_pre["bx"]
    tension0_x = np.gradient(bvec_pre["bx"], axis=0) * bvec_pre["bx"] / dx_de
    tension0_z = np.gradient(bvec_pre["bx"], axis=1) * bvec_pre["bz"] / dz_de
    tension0 = tension0_x + tension0_z

    dbulk = rho * (vsx * np.gradient(vsx, axis=0) / dx_de +
                   vsz * np.gradient(vsx, axis=1) / dz_de)
    db2 = 0.5 * np.gradient(b2, axis=0) / dx_de
    dpperp = np.gradient(pperp, axis=0) / dx_de
    divpx = (np.gradient(bvec_pre["pexx"], dx_de, axis=0) +
             np.gradient(bvec_pre["pexz"], dz_de, axis=1) +
             np.gradient(bvec_pre["pixx"], dx_de, axis=0) +
             np.gradient(bvec_pre["pixz"], dz_de, axis=1))
    divpx_gyrotropic = dpperp + tension0 - tension

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
    nvar = 9
    work_force = np.zeros([nvar, nx])
    for ix in range(nx):
        work_force[0, ix] = np.sum(db2[ix, iz_bot[ix]:iz_top[ix]+1])
        work_force[1, ix] = np.sum(dpperp[ix, iz_bot[ix]:iz_top[ix]+1])
        work_force[2, ix] = np.sum(dbulk[ix, iz_bot[ix]:iz_top[ix]+1])
        work_force[3, ix] = np.sum(tension[ix, iz_bot[ix]:iz_top[ix]+1])
        work_force[4, ix] = np.sum(tension0[ix, iz_bot[ix]:iz_top[ix]+1])
        work_force[5, ix] = np.sum(tension0_x[ix, iz_bot[ix]:iz_top[ix]+1])
        work_force[6, ix] = np.sum(tension0_z[ix, iz_bot[ix]:iz_top[ix]+1])
        work_force[7, ix] = np.sum(divpx[ix, iz_bot[ix]:iz_top[ix]+1])
        work_force[8, ix] = np.sum(divpx_gyrotropic[ix, iz_bot[ix]:iz_top[ix]+1])
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
        work_force[5, ix] += tension0_x[ix, iz_bot[ix]-1] * dz_bot[ix]
        work_force[5, ix] += tension0_x[ix, iz_top[ix]+1] * dz_top[ix]
        work_force[6, ix] += tension0_z[ix, iz_bot[ix]-1] * dz_bot[ix]
        work_force[6, ix] += tension0_z[ix, iz_top[ix]+1] * dz_top[ix]
        work_force[7, ix] += divpx[ix, iz_bot[ix]-1] * dz_bot[ix]
        work_force[7, ix] += divpx[ix, iz_top[ix]+1] * dz_top[ix]
        work_force[8, ix] += divpx_gyrotropic[ix, iz_bot[ix]-1] * dz_bot[ix]
        work_force[8, ix] += divpx_gyrotropic[ix, iz_top[ix]+1] * dz_top[ix]

    work_force_int = np.zeros(work_force.shape)
    # All forces are 0 at the X-point
    for i in range(nvar):
        work_force_int[:, ix_xp::-1] = np.cumsum(work_force[:, ix_xp::-1], axis=1)
        work_force_int[:, ix_xp:] = np.cumsum(work_force[:, ix_xp:], axis=1)
    work_force_int *= dx_de * dz_de

    fig = plt.figure(figsize=[8, 6])
    rect = [0.1, 0.50, 0.85, 0.37]
    vgap = 0.02
    ax = fig.add_axes(rect)
    COLORS = palettable.tableau.Tableau_10.mpl_colors
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
    lname6 = r"$\nabla\cdot\boldsymbol{P}$"
    p6, = ax.plot(xgrid, work_force_int[7], label=lname6)
    lname7 = r"$\nabla\cdot\boldsymbol{P}_\text{gyro}$"
    p7, = ax.plot(xgrid, work_force_int[8], label=lname7)
    # p6, = ax.plot(xgrid, work_force_int[5], color='k',
    #               linestyle='-', label="tension-x")
    # p7, = ax.plot(xgrid, work_force_int[6], color='k',
    #               linestyle='--', label="tension-x")
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
    rect = [0.12, 0.15, 0.82, 0.8]
    ax = fig.add_axes(rect)
    ax.set_prop_cycle('color', COLORS)
    ax.plot(xgrid, work_force_int[3] / work_force_int[4], linewidth=2,
            label="Tension")
    ax.plot(xgrid, -work_force_int[0] / work_force_int[4], linewidth=2,
            label="-Magnetic pressure")
    ax.plot(xgrid, work_force_int[2] / work_force_int[4], linewidth=2,
            label="-Bulk")
    ax.plot(xgrid, work_force_int[1] / work_force_int[4], linewidth=2,
            label="-GradP")
    ax.legend(loc=1, prop={'size': 12}, ncol=1,
              shadow=False, fancybox=False, frameon=False)
    ax.grid()
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([-0.1, 1.2])
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlabel(r'$x/d_e$', fontsize=20)
    ax.tick_params(labelsize=16)
    # lname = (r"$\int_{x_0}^x\text{Tension }|dx|/" +
    #          r"\int_{x0}^x\boldsymbol{B}\cdot\nabla B_x/4\pi |dx|$")
    # fig.suptitle(lname, fontsize=20)
    fname = img_dir + "epsilon_avg_" + str(tframe) + ".pdf"
    fig.savefig(fname)

    if show_plot:
        plt.show()
    else:
        plt.close("all")


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
    tension0_x = np.gradient(bvec_pre["bx"], axis=0) * bvec_pre["bx"] / dx_de
    tension0_z = np.gradient(bvec_pre["bx"], axis=1) * bvec_pre["bz"] / dz_de
    tension0 = tension0_x + tension0_z

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
    nvar = 7
    work_force = np.zeros([nvar, nx])
    work_force[0, :] = db2[:, nz//2]
    work_force[1, :] = dpperp[:, nz//2]
    work_force[2, :] = dbulk[:, nz//2]
    work_force[3, :] = tension[:, nz//2]
    work_force[4, :] = tension0[:, nz//2]
    work_force[5, :] = tension0_x[:, nz//2]
    work_force[6, :] = tension0_z[:, nz//2]

    work_force_int = np.zeros(work_force.shape)
    # All forces are 0 at the X-point
    for i in range(nvar):
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
    p6, = ax.plot(xgrid, work_force_int[5], color='k',
                  linestyle='-', label="tension-x")
    p7, = ax.plot(xgrid, work_force_int[6], color='k',
                  linestyle='--', label="tension-x")
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
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_run_dir = pic_info.run_dir
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
    zgrid = np.linspace(zmin, zmax, pic_info.nz)
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
    rect = [0.13, 0.15, 0.82, 0.8]
    ax = fig.add_axes(rect)
    ax.set_prop_cycle('color', COLORS)
    sigma = 5
    fdata = gaussian_filter(vsx[:, nz//2], sigma=sigma)
    ax.plot(xgrid, fdata/va, linewidth=2, label=r"$v_x/v_A$")
    # fdata = gaussian_filter(vsy[:, nz//2], sigma=sigma)
    # ax.plot(xgrid, fdata/va, linewidth=2, label=r"$v_y/v_A$")
    # fdata = gaussian_filter(vsz[:, nz//2], sigma=sigma)
    # ax.plot(xgrid, fdata/va, linewidth=2, label=r"$v_z/v_A$")
    ax.plot([xmin, xmax], [0, 0], linewidth=1, linestyle='--', color='k')
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([-0.6, 0.6])
    # ax.legend(loc=4, prop={'size': 16}, ncol=1,
    #           shadow=False, fancybox=False, frameon=False)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlabel(r'$x/d_e$', fontsize=20)
    ax.set_ylabel(r'$v_x/v_A$', fontsize=20)
    ax.tick_params(labelsize=16)
    ax.grid(True)
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
    if "open" in pic_run:
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
                      np.squeeze(bvec["cbz"]).T, color='k',
                      linewidth=0.5)
    else:
        kwargs = {"current_time": tframe,
                  "xl": 0, "xr": pic_info.lx_di,
                  "zb": -0.5*pic_info.lz_di, "zt": 0.5*pic_info.lz_di}
        fname = pic_run_dir + "data/Ay.gda"
        x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
        ax.contour(xgrid, zgrid, Ay, colors='k', linewidths=0.5,
                   levels=np.linspace(np.min(Ay), np.max(Ay), 20))
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([zmin, zmax])
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
    """Particle number density
    """
    tframe = plot_config["tframe"]
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_run_dir = pic_info.run_dir
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


def plot_density_cut(plot_config, show_plot=True):
    """Particle number density and cuts along two directions
    """
    tframe = plot_config["tframe"]
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_run_dir = pic_info.run_dir
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
    fig = plt.figure(figsize=[7*lx_de/lz_de, 6])
    rect = [0.15, 0.12, 0.7, 0.8]
    ax = fig.add_axes(rect)
    hgap, vgap = 0.12, 0.03
    cmap = plt.cm.viridis
    im1 = ax.imshow(rhos["ion"].T,
                    extent=[xmin, xmax, zmin, zmax],
                    vmin=0.5, vmax=3,
                    cmap=cmap, aspect='auto',
                    origin='lower', interpolation='bicubic')
    if "open" in pic_run:
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
                      np.squeeze(bvec["cbz"]).T, color='k',
                      linewidth=0.5)
    else:
        kwargs = {"current_time": tframe,
                  "xl": 0, "xr": pic_info.lx_di,
                  "zb": -0.5*pic_info.lz_di, "zt": 0.5*pic_info.lz_di}
        fname = pic_run_dir + "data/Ay.gda"
        x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
        ax.contour(xgrid, zgrid, Ay, colors='k', linewidths=0.5,
                   levels=np.linspace(np.min(Ay), np.max(Ay), 20))
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([zmin, zmax])
    ax.set_xlabel(r'$x/d_e$', fontsize=16)
    ax.set_ylabel(r'$z/d_e$', fontsize=16)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.tick_params(labelsize=12)
    rect[1] -= rect[3] + vgap
    rect[1] += rect[3] + vgap
    rect_cbar = np.copy(rect)
    rect_cbar[0] += rect[2] + 0.01
    rect_cbar[2] = 0.02 / (lx_de / lz_de)
    cbar_ax = fig.add_axes(rect_cbar)
    cbar = fig.colorbar(im1, cax=cbar_ax, extend='both')
    cbar_ax.tick_params(bottom=False, top=False, left=False, right=True)
    cbar_ax.tick_params(axis='y', which='major', direction='out')
    cbar_ax.tick_params(axis='y', which='minor', direction='in', right=False)
    cbar_ax.set_title(r'$n_i$', fontsize=20)
    cbar.ax.tick_params(labelsize=12)

    img_dir_p = '../img/rate_problem/nrho/' + pic_run + '/'
    mkdir_p(img_dir_p)
    fname = img_dir_p + "nrho_" + str(tframe) + ".pdf"
    fig.savefig(fname, dpi=200)

    fig = plt.figure(figsize=[7, 5])
    rect = [0.15, 0.12, 0.8, 0.8]
    ax = fig.add_axes(rect)
    ax.plot(xgrid, rhos["ion"][:, nz//2], label=r"$n_i$")
    ax.plot([xmin, xmax], [1, 1], linewidth=1, linestyle='--', color='k')
    ax.set_xlim([xmin, xmax])
    ax.set_xlabel(r'$x/d_e$', fontsize=16)
    ax.set_ylabel(r'$n_i$', fontsize=16)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.tick_params(labelsize=12)
    fname = img_dir_p + "nrho_x_" + str(tframe) + ".pdf"
    fig.savefig(fname)

    fig = plt.figure(figsize=[7, 5])
    rect = [0.15, 0.12, 0.8, 0.8]
    ax = fig.add_axes(rect)
    shift = 128
    p1, = ax.plot(zgrid[shift:-shift], rhos["ion"][xcut, shift:-shift])
    zmin1 = zmin + dz_de * shift
    zmax1 = zmax - dz_de * shift
    ax.set_xlim([zmin1, zmax1])
    ax.set_xlabel(r'$z/d_e$', fontsize=16)
    ax.set_ylabel(r'$n_i$', fontsize=16)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.tick_params(labelsize=12)
    fname = img_dir_p + "nrho_z_" + str(tframe) + ".pdf"
    fig.savefig(fname)

    if show_plot:
        plt.show()
    else:
        plt.close('all')


def middle_step_rk4(x, z, nx, nz, dx, dz, Bx, Bz):
    """Middle step of Runge-Kutta method to trace the magnetic field line.

    Args:
        x, z: the coordinates of current point.
        nx, nz: the dimensions of the data.
        Bx, Bz: the magnetic field arrays.
    """
    # for Bx at face center
    xf = x / dx
    zf = (z - 0.5 * dz) / dz
    ix1 = int(xf)
    iz1 = int(zf)
    dxl = xf - ix1  # left
    dxr = 1 - dxl   # right
    dzb = zf - iz1  # bottom
    dzt = 1 - dzb   # top
    weights = np.zeros((2, 2))
    if ix1 < nx - 1:
        if zf < 0 or zf > nz - 2:
            bx = Bx[iz1, ix1] * dxr + Bx[iz1, ix1+1] * dxl
        else:
            weights[0, 0] = dxr * dzt
            weights[0, 1] = dxl * dzt
            weights[1, 0] = dxr * dzb
            weights[1, 1] = dxl * dzb
            bx = np.sum(Bx[iz1:iz1+2, ix1:ix1+2] * weights)
    else:
        if zf < 0 or zf > nz - 2:
            bx = Bx[iz1, nx-1]
        else:
            bx = Bx[iz1, nx-1] * dzt + Bx[iz1+1, nx-1] * dzb

    # for Bz at face center
    xf = (x - 0.5 * dx) / dx
    zf = z / dz
    ix1 = int(xf)
    iz1 = int(zf)
    dxl = xf - ix1  # left
    dxr = 1 - dxl   # right
    dzb = zf - iz1  # bottom
    dzt = 1 - dzb   # top
    if iz1 < nz - 1:
        if xf < 0 or xf > nx - 2:
            bz = Bz[iz1, ix1] * dzt + Bz[iz1+1, ix1] * dzb
        else:
            weights[0, 0] = dxr * dzt
            weights[0, 1] = dxl * dzt
            weights[1, 0] = dxr * dzb
            weights[1, 1] = dxl * dzb
            bz = np.sum(Bz[iz1:iz1+2, ix1:ix1+2] * weights)
    else:
        if xf < 0 or xf > nx - 2:
            bz = Bz[nz-1, ix1]
        else:
            bz = Bz[nz-1, ix1] * dxr + Bz[nz-1, ix1+1] * dxl

    # if ix1 < nx and ix2 < nx and iz1 < nz and iz2 < nz:
    #     bx = (Bx[iz1, ix1] * v1 + Bx[iz1, ix2] * v2 +
    #           Bx[iz2, ix2] * v3 + Bx[iz2, ix1] * v4)
    #     bz = (Bz[iz1, ix1] * v1 + Bz[iz1, ix2] * v2 +
    #           Bz[iz2, ix2] * v3 + Bz[iz2, ix1] * v4)
    #     absB = math.sqrt(bx**2 + bz**2)
    #     deltax1 = bx / absB
    #     deltaz1 = bz / absB
    # else:
    #     if ix1 >= nx:
    #         ix1 = nx - 1
    #     if iz1 >= nz:
    #         iz1 = nz - 1
    #     bx = Bx[iz1, ix1]
    #     bz = Bz[iz1, ix1]
    ib = 1.0 / math.sqrt(bx**2 + bz**2)
    deltax1 = bx * ib
    deltaz1 = bz * ib
    return (deltax1, deltaz1, bx, bz)


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
        deltax1, deltaz1, _, _ = middle_step_rk4(x, z, nx, nz, dx_de, dz_de,
                                                 bvec["cbx"], bvec["cbz"])
        x1 = x + deltax1 * hds
        z1 = z + deltaz1 * hds
        deltax2, deltaz2, _, _ = middle_step_rk4(x1, z1, nx, nz, dx_de, dz_de,
                                                 bvec["cbx"], bvec["cbz"])
        x2 = x + deltax2 * hds
        z2 = z + deltaz2 * hds
        deltax3, deltaz3, _, _ = middle_step_rk4(x2, z2, nx, nz, dx_de, dz_de,
                                                 bvec["cbx"], bvec["cbz"])
        x3 = x + deltax3 * deltas
        z3 = z + deltaz3 * deltas
        deltax4, deltaz4, bx, bz = middle_step_rk4(x3, z3, nx, nz, dx_de, dz_de,
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

    _, _, bx, bz = middle_step_rk4(xs, zs, nx, nz, dx_de, dz_de,
                                   bvec["cbx"], bvec["cbz"])
    xlist = np.asarray(xlist)
    zlist = np.asarray(zlist)
    return (xlist, zlist, bx)


def calc_rrate_bflux(plot_config, show_plot=True):
    """Calculate reconnection rate based magnetic flux
    """
    pic_run = plot_config["pic_run"]
    tframe = plot_config["tframe"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_run_dir = pic_info.run_dir
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
    vpic_info = get_vpic_info(pic_run_dir)
    fields_interval = int(vpic_info["fields_interval"])
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
    while (z2 - z1) > 0.05 * dz_de:
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

    f = interp1d(xlist, zlist)
    znew = f(xgrid)
    bflux_x = np.zeros(nx)
    for ix in range(nx):
        iz_close = math.floor(znew[ix] / dz_de)
        dz_close = 1.0 - znew[ix] / dz_de + iz_close
        bflux_x[ix] = np.sum(bvec["cbx"][iz_close+1:, ix])
        bflux_x[ix] += bvec["cbx"][iz_close, ix] * dz_close
    bflux_x *= dz_de

    fdir = '../data/rate_problem/rrate_bflux/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + 'rrate_bflux_' + str(tframe) + '.dat'
    bflux = np.asarray([bflux])
    bflux.tofile(fname)

    fname = fdir + 'rrate_bflux_x_' + str(tframe) + '.dat'
    bflux_x.tofile(fname)

    xz = np.asarray([xlist, zlist])
    fname = fdir + 'xz_close_' + str(tframe) + '.dat'
    xz.tofile(fname)


def get_edrive_params(pic_run_dir):
    """Get driving parameters in the inflow region
    """
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
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_run_dir = pic_info.run_dir
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

    edrive, tdrive = get_edrive_params(pic_run_dir)
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
    ax.plot(tfields_wci, rrate_bflux, marker='o',
            label="Rate from magnetic flux")

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


def plot_rrate_bflux_beta(plot_config, bg=0.0, show_plot=True):
    """Plot reconnection rate based magnetic flux for runs with different beta
    """
    if plot_config["open_boundary"]:
        pic_runs = ["mime400_nb_n0_002",
                    "mime400_nb_n0_02",
                    "mime400_nb_n0_1_new",
                    "mime400_Tbe_Te_20"]
        betas = [0.02, 0.2, 1.0, 20.0]
    else:
        pic_runs = ["mime400_Tb_T0_025",
                    "mime400_Tb_T0_1",
                    "mime400_Tb_T0_10_weak",
                    "mime400_Tb_T0_40_nppc450_old"]
        betas = [0.25, 1.0, 10, 40]
    fig1 = plt.figure(figsize=[7, 5])
    rect = [0.11, 0.12, 0.83, 0.83]
    ax = fig1.add_axes(rect)
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    ax.set_prop_cycle('color', COLORS)
    if bg > 0.99:  # one or a few integer times of B0
        bg_str = "_bg" + str(int(bg))
    elif bg > 0.01:  # between 0 to B0
        bg_str = "_bg" + str(int(bg*10)).zfill(2)
    else:
        bg_str = ""
    for irun, pic_run in enumerate(pic_runs):
        pic_run += bg_str
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        pic_run_dir = pic_info.run_dir
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

        edrive, tdrive = get_edrive_params(pic_run_dir)
        vin = edrive * (1.0 - np.exp(-tfields/tdrive)) / wpe_wce / b0

        rrate_bflux = -np.gradient(bflux) / dtf
        rrate_bflux /= va * b0
        if "open" in pic_run or "test" in pic_run:
            rrate_bflux += vin
        print("Maximum rate: %f" % rrate_bflux.max())
        ax.plot(tfields_wci, rrate_bflux, marker='o',
                label=r"$\beta=" + str(betas[irun]) + "$")

    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    if plot_config["open_boundary"]:
        ax.set_xlim([0, 50])
    else:
        ax.set_xlim([0, 47])
    if plot_config["open_boundary"]:
        ax.set_ylim([0, 0.16])
    else:
        ax.set_ylim([0, 0.11])
    ax.tick_params(labelsize=12)
    ax.set_xlabel(r'$t\Omega_{ci}$', fontsize=16)
    ax.set_ylabel(r'$E_R$', fontsize=16)
    ax.legend(loc=2, prop={'size': 16}, ncol=1,
              shadow=False, fancybox=False, frameon=False)

    fdir = '../img/rate_problem/rrate/'
    mkdir_p(fdir)
    if plot_config["open_boundary"]:
        fname = fdir + 'rrate_bflux_beta_open.pdf'
    else:
        fname = fdir + 'rrate_bflux_beta' + bg_str + '.pdf'
    fig1.savefig(fname)

    plt.show()


def calc_open_angle(plot_config, show_plot=True):
    """Calculate exhaust open angle
    """
    tframe = plot_config["tframe"]
    pic_run = plot_config["pic_run"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_run_dir = pic_info.run_dir
    fields_interval = pic_info.fields_interval
    tindex = fields_interval * tframe
    smime = math.sqrt(pic_info.mime)
    lx_de = pic_info.lx_di * smime
    lz_de = pic_info.lz_di * smime
    xmin, xmax = 0, lx_de
    zmin, zmax = -0.5 * lz_de, 0.5 * lz_de
    dx_de = lx_de / pic_info.nx
    dz_de = lz_de / pic_info.nz

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
    iz_top = np.argmin(zlist_top)
    iz_bot = np.argmax(zlist_bot)
    xtop = xlist_top[iz_top]
    ztop = zlist_top[iz_top]
    if pic_run == "mime400_Tb_T0_025":
        shift = 200
        theta_top = np.arctan((zlist_top - ztop) / (xlist_top - xtop + 1E-10)) * 180 / math.pi
        theta_max1 = theta_top[iz_top+shift:].max()
        theta_min1 = -theta_top[:iz_top-shift].min()
        xbot = xlist_bot[iz_bot]
        zbot = zlist_bot[iz_bot]
        theta_bot = np.arctan((zlist_bot - zbot) / (xlist_bot - xbot + 1E-10)) * 180 / math.pi
        theta_max2 = theta_bot[iz_bot+shift:].max()
        theta_min2 = -theta_bot[:iz_bot-shift].min()
    else:
        shift = 2000
        f = np.polyfit(xlist_top[iz_top:iz_top+shift], zlist_top[iz_top:iz_top+shift], 1)
        theta_max1 = math.atan(f[0]) * 180 / math.pi
        f = np.polyfit(xlist_top[iz_top-shift:iz_top], zlist_top[iz_top-shift:iz_top], 1)
        theta_min1 = -math.atan(f[0]) * 180 / math.pi
        f = np.polyfit(xlist_bot[iz_bot:iz_bot+shift], zlist_bot[iz_bot:iz_bot+shift], 1)
        theta_max2 = math.atan(f[0]) * 180 / math.pi
        f = np.polyfit(xlist_bot[iz_bot-shift:iz_bot], zlist_bot[iz_bot-shift:iz_bot], 1)
        theta_min2 = -math.atan(f[0]) * 180 / math.pi

    theta_min = 0.5*(theta_min1 + theta_min2)
    theta_max = 0.5*(theta_max1 + theta_max2)

    if pic_run == "mime400_Tb_T0_025":
        angle0 = (theta_max1 + theta_min2) * 0.5
        # angle0 = np.max([theta_min1, theta_max1, theta_min2, theta_max2])
    else:
        angle0 = (theta_min + theta_max) * 0.5

    fdir = '../data/rate_problem/open_angle/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + "open_angle_" + str(tframe) + ".dat"
    np.asarray([angle0]).tofile(fname)


def plot_open_angle(plot_config, bg=0.0, show_plot=True):
    """Plot exhaust open angle
    """
    pic_runs = ["mime400_Tb_T0_025",
                "mime400_Tb_T0_1",
                "mime400_Tb_T0_10_weak",
                "mime400_Tb_T0_40_nppc450"]
    betas = [0.25, 1.0, 10, 40]
    fig = plt.figure(figsize=[7, 5])
    rect = [0.11, 0.12, 0.83, 0.83]
    ax = fig.add_axes(rect)
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    ax.set_prop_cycle('color', COLORS)
    if bg > 0.99:  # one or a few integer times of B0
        bg_str = "_bg" + str(int(bg))
    elif bg > 0.01:  # between 0 to B0
        bg_str = "_bg" + str(int(bg*10)).zfill(2)
    else:
        bg_str = ""
    for irun, pic_run in enumerate(pic_runs):
        pic_run += bg_str
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        pic_run_dir = pic_info.run_dir
        b0 = pic_info.b0
        fdir = '../data/rate_problem/open_angle/' + pic_run + '/'
        nframes = len(os.listdir(fdir))
        open_angle = np.zeros(nframes)
        vpic_info = get_vpic_info(pic_run_dir)
        fields_interval = vpic_info["fields_interval"]
        tfields_wci = np.arange(nframes) * pic_info.dtwci * fields_interval
        for tframe in range(nframes):
            fname = fdir + "open_angle_" + str(tframe) + ".dat"
            fdata = np.fromfile(fname)
            open_angle[tframe] = fdata[0]
        ax.plot(tfields_wci, open_angle, linewidth=2,
                label=r"$\beta=" + str(betas[irun]) + "$")

    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlim([0, 47])
    ax.set_ylim([0, 28])
    ax.tick_params(labelsize=12)
    ax.legend(loc=2, prop={'size': 16}, ncol=1,
              shadow=False, fancybox=False, frameon=False)
    ax.set_xlabel(r'$t\Omega_{ci}$', fontsize=16)
    ax.set_ylabel(r'Opening Angle ($^\circ$)', fontsize=16)

    img_dir = '../img/rate_problem/open_angle/'
    mkdir_p(img_dir)
    fname = img_dir + "open_angle_beta" + bg_str + ".pdf"
    fig.savefig(fname)

    plt.show()


def open_angle(plot_config, show_plot=True):
    """Plot exhaust open angle
    """
    tframe = plot_config["tframe"]
    pic_run = plot_config["pic_run"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_run_dir = pic_info.run_dir
    fields_interval = pic_info.fields_interval
    tindex = fields_interval * tframe
    smime = math.sqrt(pic_info.mime)
    lx_de = pic_info.lx_di * smime
    lz_de = pic_info.lz_di * smime
    xmin, xmax = 0, lx_de
    zmin, zmax = -0.5 * lz_de, 0.5 * lz_de
    dx_de = lx_de / pic_info.nx
    dz_de = lz_de / pic_info.nz
    nx = pic_info.nx
    nz = pic_info.nz
    xs = nx//2 - nx//4
    xe = nx//2 + nx//4
    zs = nz//2 - nz//8
    ze = nz//2 + nz//8
    dx_de = lx_de / nx
    dz_de = lz_de / nz
    nxs = xe - xs
    nzs = ze - zs
    xmin = xs * dz_de
    xmax = xe * dz_de
    zmin_pic = -lz_de * 0.5
    zmin = zs * dz_de + zmin_pic
    zmax = ze * dz_de + zmin_pic

    fname = (pic_run_dir + "hydro_hdf5/T." + str(tindex) +
             "/hydro_electron_" + str(tindex) + ".h5")
    je = {}
    with h5py.File(fname, 'r') as fh:
        group = fh["Timestep_" + str(tindex)]
        for var in ["jx", "jy", "jz"]:
            dset = group[var]
            je[var]= dset[xs:xe, 0, zs:ze]

    fname = (pic_run_dir + "hydro_hdf5/T." + str(tindex) +
             "/hydro_ion_" + str(tindex) + ".h5")
    ji = {}
    with h5py.File(fname, 'r') as fh:
        group = fh["Timestep_" + str(tindex)]
        for var in ["jx", "jy", "jz"]:
            dset = group[var]
            ji[var]= dset[xs:xe, 0, zs:ze]

    xgrid = np.linspace(xmin, xmax, nxs)
    zgrid = np.linspace(zmin, zmax, nzs)

    absj = np.squeeze(np.sqrt((je["jx"] + ji["jx"])**2 +
                              (je["jy"] + ji["jy"])**2 +
                              (je["jz"] + ji["jz"])**2))
    fig = plt.figure(figsize=[7, 3.5])
    rect = [0.12, 0.16, 0.77, 0.75]
    ax = fig.add_axes(rect)
    im1 = ax.imshow(absj.T,
                    extent=[xmin, xmax, zmin, zmax],
                    vmin=0, vmax=0.08,
                    cmap=plt.cm.viridis, aspect='auto',
                    origin='lower', interpolation='bicubic')
    # # Magnetic field lines
    # fname = (pic_run_dir + "field_hdf5/T." + str(tindex) +
    #          "/fields_" + str(tindex) + ".h5")
    # bvec = {}
    # with h5py.File(fname, 'r') as fh:
    #     group = fh["Timestep_" + str(tindex)]
    #     for var in ["cbx", "cbz"]:
    #         dset = group[var]
    #         bvec[var]= dset[xs:xe, 0, zs:ze]
    # xmesh, zmesh = np.meshgrid(xgrid, zgrid)
    # ax.streamplot(xmesh, zmesh, bvec["cbx"].T, bvec["cbz"].T,
    #               color='w', linewidth=0.5)
    xmin_di = xmin / smime
    xmax_di = xmax / smime
    zmin_di = zmin / smime
    zmax_di = zmax / smime
    kwargs = {"current_time": tframe,
              "xl": xmin_di, "xr": xmax_di,
              "zb": zmin_di, "zt": zmax_di}
    fname = pic_run_dir + "data/Ay.gda"
    x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
    xde = x * smime
    zde = z * smime
    ax.contour(xde, zde, Ay, colors='w', linewidths=0.5,
               levels=np.linspace(np.min(Ay), np.max(Ay), 8))

    fdir = '../data/rate_problem/exhaust_boundary/' + pic_run + '/'
    fname = fdir + 'xz_top_' + str(tframe) + '.dat'
    xz = np.fromfile(fname).reshape([2, -1])
    xlist_top = xz[0, :]
    zlist_top = xz[1, :] + zmin_pic
    fname = fdir + 'xz_bot_' + str(tframe) + '.dat'
    xz = np.fromfile(fname).reshape([2, -1])
    xlist_bot = xz[0, :]
    zlist_bot = xz[1, :] + zmin_pic
    x0 = 0.5 * (xlist_bot[np.argmax(zlist_bot)] +
                xlist_top[np.argmin(zlist_top)])
    ix_xpoint = int(x0 / dx_de)
    z0 = zlist_top[np.argmin(zlist_top)]
    iz_top = np.argmin(zlist_top)
    iz_bot = np.argmax(zlist_bot)
    xtop = xlist_top[iz_top]
    ztop = zlist_top[iz_top]

    if pic_run == "mime400_Tb_T0_025":
        shift = 200
        theta_top = np.arctan((zlist_top - ztop) / (xlist_top - xtop + 1E-10)) * 180 / math.pi
        theta_max1 = theta_top[iz_top+shift:].max()
        theta_min1 = -theta_top[:iz_top-shift].min()
        xbot = xlist_bot[iz_bot]
        zbot = zlist_bot[iz_bot]
        theta_bot = np.arctan((zlist_bot - zbot) / (xlist_bot - xbot + 1E-10)) * 180 / math.pi
        theta_max2 = theta_bot[iz_bot+shift:].max()
        theta_min2 = -theta_bot[:iz_bot-shift].min()
    else:
        shift = 2000
        f = np.polyfit(xlist_top[iz_top:iz_top+shift], zlist_top[iz_top:iz_top+shift], 1)
        theta_max1 = math.atan(f[0]) * 180 / math.pi
        f = np.polyfit(xlist_top[iz_top-shift:iz_top], zlist_top[iz_top-shift:iz_top], 1)
        theta_min1 = -math.atan(f[0]) * 180 / math.pi
        f = np.polyfit(xlist_bot[iz_bot:iz_bot+shift], zlist_bot[iz_bot:iz_bot+shift], 1)
        theta_max2 = math.atan(f[0]) * 180 / math.pi
        f = np.polyfit(xlist_bot[iz_bot-shift:iz_bot], zlist_bot[iz_bot-shift:iz_bot], 1)
        theta_min2 = -math.atan(f[0]) * 180 / math.pi

    theta_min = 0.5*(theta_min1 + theta_min2)
    theta_max = 0.5*(theta_max1 + theta_max2)
    if pic_run == "mime400_Tb_T0_025":
        angle0 = (theta_max1 + theta_min2) * 0.5
    else:
        angle0 = (theta_min + theta_max) * 0.5

    # ax.plot(xlist_top[iz_top-shift:iz_top+shift],
    #         zlist_top[iz_top-shift:iz_top+shift],
    #         linewidth=2, color=COLORS[0])
    # ax.plot(xlist_bot[iz_bot-shift:iz_bot+shift],
    #         zlist_bot[iz_bot-shift:iz_bot+shift],
    #         linewidth=2, color=COLORS[0])
    ax.plot(xlist_top, zlist_top, linewidth=2, color=COLORS[0])
    ax.plot(xlist_bot, zlist_bot, linewidth=2, color=COLORS[0])

    length = 500.0
    x1 = x0 + 0.5 * length * math.cos(angle0*math.pi/180)
    x2 = x0 - 0.5 * length * math.cos(angle0*math.pi/180)
    z1 = z0 + 0.5 * length * math.sin(angle0*math.pi/180)
    z2 = z0 - 0.5 * length * math.sin(angle0*math.pi/180)
    ax.plot([x1, x2], [z1, z2], color='k', linestyle='--', linewidth=1)
    ax.plot([x1, x2], [-z1, -z2], color='k', linestyle='--', linewidth=1)
    ax.plot([x1, x2], [0, 0], color='k', linestyle='--', linewidth=1)
    text1 = r"$" + ("{%0.1f}" % angle0) +  "^\circ$"
    ax.text(0.7, 0.5, text1, color='k', fontsize=16,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='right', verticalalignment='bottom',
            transform=ax.transAxes)

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([zmin, zmax])
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    # ax.tick_params(axis='x', which='minor', direction='in')
    # ax.tick_params(axis='x', which='major', direction='in')
    # ax.tick_params(axis='y', which='minor', direction='in')
    # ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlabel(r'$x/d_e$', fontsize=16)
    ax.set_ylabel(r'$z/d_e$', fontsize=16)
    ax.tick_params(labelsize=12)
    rect_cbar = np.copy(rect)
    rect_cbar[0] += rect[2] + 0.03
    rect_cbar[2] = 0.015
    rect_cbar[3] = rect[3]
    cbar_ax = fig.add_axes(rect_cbar)
    cbar = fig.colorbar(im1, cax=cbar_ax, extend='max')
    cbar_ax.tick_params(bottom=False, top=False, left=False, right=True)
    cbar_ax.tick_params(axis='y', which='major', direction='out')
    cbar_ax.tick_params(axis='y', which='minor', direction='in', right=False)
    cbar_ax.set_title(r'$|\boldsymbol{J}|$', fontsize=16)
    cbar.ax.tick_params(labelsize=12)
    twpe = math.ceil(tindex * pic_info.dtwpe / 0.1) * 0.1
    text1 = r'$t\omega_{pe}=' + ("{%0.0f}" % twpe) + '$'
    fig.suptitle(text1, fontsize=16)
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
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_run_dir = pic_info.run_dir
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

    # Top
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
    ztop = z2

    # Magnetic flux in the top half of box
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

    # Bottom
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
    zbot = z2

    # plt.plot(xlist_top, zlist_top)
    # plt.plot(xlist_bot, zlist_bot)
    # plt.show()

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
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_run_dir = pic_info.run_dir
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


def firehose_parameter(plot_config, show_plot=True):
    """firehose instability parameter
    """
    tframe = plot_config["tframe"]
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_run_dir = pic_info.run_dir
    fields_interval = pic_info.fields_interval
    tindex = fields_interval * tframe
    smime = math.sqrt(pic_info.mime)
    lx_de = pic_info.lx_di * smime
    lz_de = pic_info.lz_di * smime
    nx, nz = pic_info.nx, pic_info.nz
    xs = nx//2 - nx//4
    xe = nx//2 + nx//4
    zs = nz//2 - nz//16
    ze = nz//2 + nz//16
    dx_de = lx_de / nx
    dz_de = lz_de / nz
    xmin = xs * dz_de
    xmax = xe * dz_de
    zmin = zs * dz_de - lz_de * 0.5
    zmax = ze * dz_de - lz_de * 0.5

    vpic_info = get_vpic_info(pic_run_dir)
    vthe = vpic_info["vtheb/c"]
    n0 = vpic_info["n0"]
    b0 = vpic_info["b0"]
    Te = vpic_info["Te"]
    Tbe_Te = vpic_info["Tbe/Te"]
    nb_n0 = vpic_info["nb/n0"]
    nb = n0 * nb_n0
    p0 = nb * Te * Tbe_Te

    bvec_pre = get_bfield_pressure(plot_config, box=[xs, zs, xe, ze])
    sigma = 3
    bvec_pre["bx"] = gaussian_filter(bvec_pre["bx"], sigma=sigma).T
    bvec_pre["by"] = gaussian_filter(bvec_pre["by"], sigma=sigma).T
    bvec_pre["bz"] = gaussian_filter(bvec_pre["bz"], sigma=sigma).T
    bvec_pre["pepara"] = gaussian_filter(bvec_pre["pepara"], sigma=sigma).T
    bvec_pre["pipara"] = gaussian_filter(bvec_pre["pipara"], sigma=sigma).T
    bvec_pre["peperp"] = gaussian_filter(bvec_pre["peperp"], sigma=sigma).T
    bvec_pre["piperp"] = gaussian_filter(bvec_pre["piperp"], sigma=sigma).T
    b2 = bvec_pre["bx"]**2 + bvec_pre["by"]**2 + bvec_pre["bz"]**2
    ppara = bvec_pre["pepara"] + bvec_pre["pipara"]
    pperp = bvec_pre["peperp"] + bvec_pre["piperp"]
    firehose = 1 - (ppara - pperp) / b2

    fig = plt.figure(figsize=[7, 3.5])
    rect = [0.12, 0.17, 0.78, 0.73]
    cmap = plt.cm.RdGy
    ax = fig.add_axes(rect)
    im1 = ax.imshow(firehose,
                    extent=[xmin, xmax, zmin, zmax],
                    vmin=0, vmax=1.0,
                    cmap=cmap, aspect='auto',
                    origin='lower', interpolation='bicubic')

    xmin_di = xmin / smime
    xmax_di = xmax / smime
    zmin_di = zmin / smime
    zmax_di = zmax / smime
    kwargs = {"current_time": tframe,
              "xl": xmin_di, "xr": xmax_di,
              "zb": zmin_di, "zt": zmax_di}
    fname = pic_run_dir + "data/Ay.gda"
    x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
    xde = x * smime
    zde = z * smime
    ax.contour(xde, zde, Ay, colors='w', linewidths=0.5,
               levels=np.linspace(np.min(Ay), np.max(Ay), 10))

    # fdir = '../data/rate_problem/exhaust_boundary/' + pic_run + '/'
    # fname = fdir + 'xz_top_' + str(tframe) + '.dat'
    # xz = np.fromfile(fname).reshape([2, -1])
    # xlist_top = xz[0, :]
    # zlist_top = xz[1, :] - 0.5 * lz_de
    # fname = fdir + 'xz_bot_' + str(tframe) + '.dat'
    # xz = np.fromfile(fname).reshape([2, -1])
    # xlist_bot = xz[0, :]
    # zlist_bot = xz[1, :] - 0.5 * lz_de
    # ax.plot(xlist_top, zlist_top, linewidth=1, color=COLORS[0], linestyle='-')
    # ax.plot(xlist_bot, zlist_bot, linewidth=1, color=COLORS[0], linestyle='-')

    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlabel(r'$x/d_e$', fontsize=20)
    ax.set_ylabel(r'$z/d_e$', fontsize=20)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([zmin, zmax])
    ax.tick_params(labelsize=16)

    rect_cbar = np.copy(rect)
    rect_cbar[0] += rect[2] + 0.01
    rect_cbar[2] = 0.015
    cbar_ax = fig.add_axes(rect_cbar)
    cbar = fig.colorbar(im1, cax=cbar_ax, extend='both')
    cbar_ax.tick_params(bottom=False, top=False, left=False, right=True)
    cbar_ax.tick_params(axis='y', which='major', direction='out')
    cbar_ax.tick_params(axis='y', which='minor', direction='in', right=False)
    cbar.ax.tick_params(labelsize=12)
    cbar_ax.set_title(r'$\epsilon$', fontsize=20)

    twpe = math.ceil(tindex * pic_info.dtwpe / 0.1) * 0.1
    text1 = r'$t\omega_{pe}=' + ("{%0.0f}" % twpe) + '$'
    fig.suptitle(text1, fontsize=20)

    img_dir = '../img/rate_problem/firehose/' + pic_run + '/'
    mkdir_p(img_dir)
    fname = img_dir + "firehose_" + str(tframe) + ".jpg"
    fig.savefig(fname, dpi=200)

    if show_plot:
        plt.show()
    else:
        plt.close("all")


def firehose_parameter_zcut(plot_config, show_plot=True):
    """firehose instability parameter cut along z
    """
    tframe = plot_config["tframe"]
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_run_dir = pic_info.run_dir
    fields_interval = pic_info.fields_interval
    tindex = fields_interval * tframe
    smime = math.sqrt(pic_info.mime)
    lx_de = pic_info.lx_di * smime
    lz_de = pic_info.lz_di * smime
    nx, nz = pic_info.nx, pic_info.nz
    xs = nx//2 - nx//8
    xe = nx//2 + nx//8
    zs = nz//2 - nz//4
    ze = nz//2 + nz//4
    nxs = xe - xs
    nzs = ze - zs
    dx_de = lx_de / nx
    dz_de = lz_de / nz
    xmin = xs * dz_de
    xmax = xe * dz_de
    zmin = zs * dz_de - lz_de * 0.5
    zmax = ze * dz_de - lz_de * 0.5

    vpic_info = get_vpic_info(pic_run_dir)
    vthe = vpic_info["vtheb/c"]
    n0 = vpic_info["n0"]
    b0 = vpic_info["b0"]
    Te = vpic_info["Te"]
    Tbe_Te = vpic_info["Tbe/Te"]
    nb_n0 = vpic_info["nb/n0"]
    nb = n0 * nb_n0
    p0 = nb * Te * Tbe_Te
    beta0 = Tbe_Te

    bvec_pre = get_bfield_pressure(plot_config, box=[xs, zs, xe, ze])
    sigma = 3
    bvec_pre["bx"] = gaussian_filter(bvec_pre["bx"], sigma=sigma)
    bvec_pre["by"] = gaussian_filter(bvec_pre["by"], sigma=sigma)
    bvec_pre["bz"] = gaussian_filter(bvec_pre["bz"], sigma=sigma)
    bvec_pre["pepara"] = gaussian_filter(bvec_pre["pepara"], sigma=sigma)
    bvec_pre["pipara"] = gaussian_filter(bvec_pre["pipara"], sigma=sigma)
    bvec_pre["peperp"] = gaussian_filter(bvec_pre["peperp"], sigma=sigma)
    bvec_pre["piperp"] = gaussian_filter(bvec_pre["piperp"], sigma=sigma)
    b2 = bvec_pre["bx"]**2 + bvec_pre["by"]**2 + bvec_pre["bz"]**2
    ppara = bvec_pre["pepara"] + bvec_pre["pipara"]
    pperp = bvec_pre["peperp"] + bvec_pre["piperp"]
    firehose = 1 - (ppara - pperp) / b2
    bx = np.abs(bvec_pre["bx"] / b0)
    # firehose_model = 1 + 0.5 * beta0 * (1/bx - 1/bx**4)
    # Because we didn't use relativistic injection. The electron thermal
    # pressure is smaller than expected.
    firehose_model = 1 + 0.5 * beta0 * (1/bx - 1/bx**2)

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
    ix_xp = int(x0 / dx_de) - xs

    fig = plt.figure(figsize=[7, 5])
    rect = [0.12, 0.17, 0.78, 0.73]
    ax = fig.add_axes(rect)
    ax.set_prop_cycle('color', COLORS)
    zgrid = np.linspace(zmin, zmax, nzs)
    ax.plot(zgrid, firehose[ix_xp, :], label="Simulation")
    ax.plot(zgrid, firehose_model[ix_xp, :], label="Model")
    ax.plot([zgrid[0], zgrid[-1]], [1, 1], color='k', linestyle='--')

    ax.legend(loc=2, prop={'size': 16}, ncol=1,
              shadow=False, fancybox=False, frameon=False)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlabel(r'$z/d_e$', fontsize=20)
    ax.set_ylabel(r'$\epsilon$', fontsize=20)
    ax.set_xlim([zmin, zmax])
    ax.set_ylim([0.8, 1.2])
    ax.tick_params(labelsize=16)

    twpe = math.ceil(tindex * pic_info.dtwpe / 0.1) * 0.1
    text1 = r'$t\omega_{pe}=' + ("{%0.0f}" % twpe) + '$'
    fig.suptitle(text1, fontsize=20)

    img_dir = '../img/rate_problem/firehose_zcut/' + pic_run + '/'
    mkdir_p(img_dir)
    fname = img_dir + "firehose_zcut_" + str(tframe) + ".pdf"
    fig.savefig(fname)

    if show_plot:
        plt.show()
    else:
        plt.close("all")


def pxyz_zcut(plot_config, show_plot=True):
    """pxx, pyy, and pzz cut along z
    """
    tframe = plot_config["tframe"]
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_run_dir = pic_info.run_dir
    fields_interval = pic_info.fields_interval
    tindex = fields_interval * tframe
    smime = math.sqrt(pic_info.mime)
    lx_de = pic_info.lx_di * smime
    lz_de = pic_info.lz_di * smime
    nx, nz = pic_info.nx, pic_info.nz
    xs = nx//2 - nx//4
    xe = nx//2 + nx//4
    zs = nz//2 - nz//16
    ze = nz//2 + nz//16
    nxs = xe - xs
    nzs = ze - zs
    dx_de = lx_de / nx
    dz_de = lz_de / nz
    xmin = xs * dz_de
    xmax = xe * dz_de
    zmin = zs * dz_de - lz_de * 0.5
    zmax = ze * dz_de - lz_de * 0.5

    vpic_info = get_vpic_info(pic_run_dir)
    vthe = vpic_info["vtheb/c"]
    n0 = vpic_info["n0"]
    b0 = vpic_info["b0"]
    Te = vpic_info["Te"]
    Tbe_Te = vpic_info["Tbe/Te"]
    nb_n0 = vpic_info["nb/n0"]
    nb = n0 * nb_n0
    p0 = nb * Te * Tbe_Te
    beta0 = Tbe_Te

    bvec_pre = get_bfield_pressure(plot_config, box=[xs, zs, xe, ze])
    sigma = 3
    bvec_pre["bx"] = gaussian_filter(bvec_pre["bx"], sigma=sigma)
    bvec_pre["by"] = gaussian_filter(bvec_pre["by"], sigma=sigma)
    bvec_pre["bz"] = gaussian_filter(bvec_pre["bz"], sigma=sigma)
    pexx = gaussian_filter(bvec_pre["pexx"], sigma=sigma)
    pixx = gaussian_filter(bvec_pre["pixx"], sigma=sigma)
    peyy = gaussian_filter(bvec_pre["peyy"], sigma=sigma)
    piyy = gaussian_filter(bvec_pre["piyy"], sigma=sigma)
    pezz = gaussian_filter(bvec_pre["pezz"], sigma=sigma)
    pizz = gaussian_filter(bvec_pre["pizz"], sigma=sigma)
    absB = np.sqrt(bvec_pre["bx"]**2 + bvec_pre["by"]**2 + bvec_pre["bz"]**2) / b0
    pexx_model = p0 / absB**2 * absB**2
    pixx_model = p0 / absB**2 * absB**2
    peyy_model = p0 * absB
    piyy_model = p0 * absB
    pezz_model = p0 * absB
    pizz_model = p0 * absB

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
    ix_xp = int(x0 / dx_de) - xs

    fig = plt.figure(figsize=[7, 7])
    rect = [0.12, 0.68, 0.8, 0.26]
    hgap, vgap = 0.02, 0.03
    nvar = 3
    axs = []
    for ivar in range(nvar):
        ax = fig.add_axes(rect)
        ax.set_prop_cycle('color', COLORS)
        axs.append(ax)
        rect[1] -= rect[3] + vgap
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')
        ax.set_xlim([zmin, zmax])
        ax.set_ylim([0, 2])
        ax.tick_params(labelsize=12)
        if ivar == nvar - 1:
            ax.set_xlabel(r'$z/d_e$', fontsize=16)
        else:
            ax.tick_params(axis='x', labelbottom=False)

    zgrid = np.linspace(zmin, zmax, nzs)
    p1, = axs[0].plot(zgrid, pexx[ix_xp, :]/p0, label="Simulation (e)")
    p2, = axs[0].plot(zgrid, pixx[ix_xp, :]/p0, label="Simulation (i)")
    axs[0].plot(zgrid, pixx_model[ix_xp, :]/p0, linestyle="--", label="Model")
    axs[0].plot([zgrid[0], zgrid[-1]], [1, 1], color='k', linestyle='--')

    p1, = axs[1].plot(zgrid, peyy[ix_xp, :]/p0, label="Simulation (e)")
    p2, = axs[1].plot(zgrid, piyy[ix_xp, :]/p0, label="Simulation (i)")
    axs[1].plot(zgrid, piyy_model[ix_xp, :]/p0, linestyle="--", label="Model")
    axs[1].plot([zgrid[0], zgrid[-1]], [1, 1], color='k', linestyle='--')

    p1, = axs[2].plot(zgrid, pezz[ix_xp, :]/p0, label="Simulation (e)")
    p2, = axs[2].plot(zgrid, pizz[ix_xp, :]/p0, label="Simulation (i)")
    axs[2].plot(zgrid, pizz_model[ix_xp, :]/p0, linestyle="--", label="Model")
    axs[2].plot([zgrid[0], zgrid[-1]], [1, 1], color='k', linestyle='--')

    axs[0].legend(loc=9, bbox_to_anchor=(0.5, 1.2),
                  prop={'size': 12}, ncol=3,
                  shadow=False, fancybox=False, frameon=False)
    axs[0].set_ylabel(r'$P_{xx}/P_0$', fontsize=16)
    axs[1].set_ylabel(r'$P_{yy}/P_0$', fontsize=16)
    axs[2].set_ylabel(r'$P_{zz}/P_0$', fontsize=16)

    img_dir = '../img/rate_problem/pxyz_zcut/' + pic_run + '/'
    mkdir_p(img_dir)
    fname = img_dir + "pxyz_zcut_" + str(tframe) + ".pdf"
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
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_run_dir = pic_info.run_dir
    plot_config["pic_run_dir"] = pic_run_dir
    fields_interval = pic_info.fields_interval
    b0 = pic_info.b0
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

    bvec_pre = get_bfield_pressure(plot_config)
    ptot = (bvec_pre["pepara"] + 2 * bvec_pre["peperp"] +
            bvec_pre["pipara"] + 2 * bvec_pre["piperp"]) / 3
    bx = bvec_pre["bx"]
    sigma = 3
    ptot = gaussian_filter(ptot, sigma=sigma)
    bx = gaussian_filter(bx, sigma=sigma)
    pcut = ptot[ix_xpoint, :]
    bcut = bx[ix_xpoint, :]

    # Electron diffusion region
    ng_de = int(1.0 / dx_de)
    iz1 = nz//2 - ng_de*2
    iz2 = nz//2 + ng_de*2
    bcut_d = bcut[iz1:iz2+1]
    f = np.polyfit(zgrid[iz1:iz2+1], bcut_d, 1)
    p = np.poly1d(f)
    bfit = p(zgrid)
    bdiff = bcut - bfit
    iz_min1 = np.argmax(bdiff < -0.5*b0)
    iz_min2 = np.argmin(bdiff > 0.5*b0)
    iz_min1_old = iz_min1
    iz_min2_old = iz_min2

    if tframe == 0:
        bxm_edr = pic_info.b0
    else:
        bxm1 = np.mean(bcut[iz_min1])
        bxm2 = np.mean(bcut[iz_min2])
        bxm_edr = (bxm1 - bxm2) * 0.5
    fdir = '../data/rate_problem/bxm_edr/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + 'bxm_' + str(tframe) + '.dat'
    fdata = np.asarray([bxm_edr])
    fdata.tofile(fname)

    # Ion diffusion region (upper)
    iz1 = iz_min1
    iz2 = iz_min1 + ng_de*10
    bcut_d = bcut[iz1:iz2+1]
    f = np.polyfit(zgrid[iz1:iz2+1], bcut_d, 1)
    p = np.poly1d(f)
    bfit1 = p(zgrid)
    bdiff = bcut - bfit1
    iz_min1 += np.argmax(bdiff[iz_min1:] < -0.05*b0)

    # Ion diffusion region (lower)
    iz1 = iz_min2 - ng_de*10
    iz2 = iz_min2
    bcut_d = bcut[iz1:iz2+1]
    f = np.polyfit(zgrid[iz1:iz2+1], bcut_d, 1)
    p = np.poly1d(f)
    bfit2 = p(zgrid)
    bdiff = bcut - bfit2
    iz_min2 = np.argmin(bdiff[:iz2+1] > 0.05*b0)

    shift = 4
    # p0 = np.mean(pcut[nz//2-shift:nz//2+shift]) - np.min(pcut)
    # bxm = math.sqrt(p0 * 2)
    # ng_di = int(math.sqrt(pic_info.mime) / dx_de)
    # nl = ng_di * 3  # only consider neighbors
    # iz_min1 = np.argmin(pcut[nz//2:nz//2+nl]) + nz//2
    # iz_min2 = np.argmin(pcut[nz//2-nl:nz//2]) + nz//2 - nl
    if tframe == 0:
        bxm = pic_info.b0
    else:
        iz1 = iz_min1 - shift
        iz2 = iz_min1 + shift + 1
        if iz2 > nz:
            iz2 = nz
        bxm1 = np.mean(bcut[iz1:iz2])
        iz1 = iz_min2 - shift
        iz2 = iz_min2 + shift + 1
        if iz1 < 0:
            iz1 = 0
        bxm2 = np.mean(bcut[iz1:iz2])
        bxm = (bxm1 - bxm2) * 0.5
    fdir = '../data/rate_problem/bxm/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + 'bxm_' + str(tframe) + '.dat'
    fdata = np.asarray([bxm])
    fdata.tofile(fname)

    # fig = plt.figure(figsize=[7, 5])
    # rect = [0.12, 0.15, 0.8, 0.8]
    # ax = fig.add_axes(rect)
    # ax.set_prop_cycle('color', COLORS)
    # ax.plot(zgrid, bcut)
    # p1, = ax.plot([zgrid[iz_min1_old]], [bcut[iz_min1_old]],
    #               linestyle='none', marker='o')
    # ax.plot([zgrid[iz_min2_old]], [bcut[iz_min2_old]],
    #         linestyle='none', marker='o', color=p1.get_color())
    # p2, = ax.plot([zgrid[iz_min1]], [bcut[iz_min1]],
    #               linestyle='none', marker='o')
    # ax.plot([zgrid[iz_min2]], [bcut[iz_min2]],
    #         linestyle='none', marker='o', color=p2.get_color())
    # ax.plot(zgrid, bfit1, color=p2.get_color())
    # ax.plot(zgrid, bfit2, color=p2.get_color())
    # ax.tick_params(bottom=True, top=True, left=True, right=True)
    # ax.tick_params(axis='x', which='minor', direction='in')
    # ax.tick_params(axis='x', which='major', direction='in')
    # ax.tick_params(axis='y', which='minor', direction='in')
    # ax.tick_params(axis='y', which='major', direction='in')
    # ax.set_xlim([zmin, zmax])
    # ax.set_ylim([-1.1*b0, 1.1*b0])
    # ax.tick_params(labelsize=12)
    # ax.set_xlabel(r'$z/d_e$', fontsize=16)
    # img_dir = '../img/rate_problem/calc_bxm/' + pic_run + '/'
    # mkdir_p(img_dir)
    # fname = img_dir + "bfit_" + str(tframe) + ".pdf"
    # fig.savefig(fname)

    # fig = plt.figure(figsize=[7, 5])
    # rect = [0.12, 0.15, 0.8, 0.8]
    # ax = fig.add_axes(rect)
    # ax.set_prop_cycle('color', COLORS)
    # ax.plot(zgrid, pcut, linewidth=2, label=r'$P$')
    # ax.plot(zgrid, 0.5*bcut**2, linewidth=2, label=r'$B_x^2/8\pi$')
    # ylim = ax.get_ylim()
    # ax.plot([zgrid[iz_min1], zgrid[iz_min1]], ylim, linewidth=1,
    #         linestyle='--', color='k')
    # ax.plot([zgrid[iz_min2], zgrid[iz_min2]], ylim, linewidth=1,
    #         linestyle='--', color='k')
    # ax.set_ylim(ylim)
    # ax.tick_params(bottom=True, top=True, left=True, right=True)
    # ax.tick_params(axis='x', which='minor', direction='in')
    # ax.tick_params(axis='x', which='major', direction='in')
    # ax.tick_params(axis='y', which='minor', direction='in')
    # ax.tick_params(axis='y', which='major', direction='in')
    # ax.set_xlim([zmin, zmax])
    # ax.tick_params(labelsize=12)
    # ax.legend(loc=7, prop={'size': 16}, ncol=1,
    #           shadow=False, fancybox=False, frameon=False)
    # ax.set_xlabel(r'$z/d_e$', fontsize=16)
    # img_dir = '../img/rate_problem/calc_bxm/' + pic_run + '/'
    # mkdir_p(img_dir)
    # fname = img_dir + "calc_bxm_" + str(tframe) + ".pdf"
    # fig.savefig(fname)

    # fig = plt.figure(figsize=[7, 5])
    # rect = [0.12, 0.15, 0.8, 0.8]
    # ax = fig.add_axes(rect)
    # ax.set_prop_cycle('color', COLORS)
    # ax.plot(zgrid, bcut, linewidth=2, label=r'$B_x$')
    # ylim = ax.get_ylim()
    # ax.plot([zgrid[iz_min1], zgrid[iz_min1]], ylim, linewidth=1,
    #         linestyle='--', color='k')
    # ax.plot([zgrid[iz_min2], zgrid[iz_min2]], ylim, linewidth=1,
    #         linestyle='--', color='k')
    # ax.set_ylim(ylim)
    # ax.tick_params(bottom=True, top=True, left=True, right=True)
    # ax.tick_params(axis='x', which='minor', direction='in')
    # ax.tick_params(axis='x', which='major', direction='in')
    # ax.tick_params(axis='y', which='minor', direction='in')
    # ax.tick_params(axis='y', which='major', direction='in')
    # ax.set_xlim([zmin, zmax])
    # ax.tick_params(labelsize=12)
    # ax.legend(loc=7, prop={'size': 16}, ncol=1,
    #           shadow=False, fancybox=False, frameon=False)
    # ax.set_xlabel(r'$z/d_e$', fontsize=16)
    # img_dir = '../img/rate_problem/calc_bxm/' + pic_run + '/'
    # mkdir_p(img_dir)
    # fname = img_dir + "bx_" + str(tframe) + ".pdf"
    # fig.savefig(fname)

    # if show_plot:
    #     plt.show()
    # else:
    #     plt.close("all")


def calc_bxm_fix(plot_config, show_plot=True):
    """Calculate bxm at fix position

    di or rho_i from the X-point
    """
    tframe = plot_config["tframe"]
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_run_dir = pic_info.run_dir
    plot_config["pic_run_dir"] = pic_run_dir
    fields_interval = pic_info.fields_interval
    nx = pic_info.nx
    nz = pic_info.nz
    smime = math.sqrt(pic_info.mime)
    lx_de = pic_info.lx_di * smime
    lz_de = pic_info.lz_di * smime
    xs = nx//2 - nx//4
    xe = nx//2 + nx//4
    zs = nz//2 - nz//16
    ze = nz//2 + nz//16
    nxs = xe - xs
    nzs = ze - zs
    dx_de = lx_de / nx
    dz_de = lz_de / nz
    xmin = xs * dz_de
    xmax = xe * dz_de
    zmin = zs * dz_de - lz_de * 0.5
    zmax = ze * dz_de - lz_de * 0.5

    vpic_info = get_vpic_info(pic_run_dir)
    b0 = pic_info.b0
    di = smime
    vthi = vpic_info["vthib/c"]
    wci_wpe = vpic_info["dt*wci"] / vpic_info["dt*wpe"]
    rhoi = vthi / wci_wpe
    dmax = max(di, rhoi)
    dnz = int(dmax / dz_de)

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
    ix_xp = int(x0 / dx_de) - xs

    tindex = fields_interval * tframe
    fname = (pic_run_dir + "field_hdf5/T." + str(tindex) +
             "/fields_" + str(tindex) + ".h5")
    with h5py.File(fname, 'r') as fh:
        group = fh["Timestep_" + str(tindex)]
        dset = group["cbx"]
        bx = dset[xs:xe, 0, zs:ze]

    bxm = 0.5 * (bx[ix_xp, nzs//2 + dnz] - bx[ix_xp, nzs//2 - dnz])
    fdir = '../data/rate_problem/bxm/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + 'bxm_fix_' + str(tframe) + '.dat'
    fdata = np.asarray([bxm])
    fdata.tofile(fname)

    # fig = plt.figure(figsize=[7, 5])
    # rect = [0.12, 0.15, 0.8, 0.8]
    # ax = fig.add_axes(rect)
    # ax.set_prop_cycle('color', COLORS)
    # ax.plot(zgrid, bcut)
    # p1, = ax.plot([zgrid[iz_min1_old]], [bcut[iz_min1_old]],
    #               linestyle='none', marker='o')
    # ax.plot([zgrid[iz_min2_old]], [bcut[iz_min2_old]],
    #         linestyle='none', marker='o', color=p1.get_color())
    # p2, = ax.plot([zgrid[iz_min1]], [bcut[iz_min1]],
    #               linestyle='none', marker='o')
    # ax.plot([zgrid[iz_min2]], [bcut[iz_min2]],
    #         linestyle='none', marker='o', color=p2.get_color())
    # ax.plot(zgrid, bfit1, color=p2.get_color())
    # ax.plot(zgrid, bfit2, color=p2.get_color())
    # ax.tick_params(bottom=True, top=True, left=True, right=True)
    # ax.tick_params(axis='x', which='minor', direction='in')
    # ax.tick_params(axis='x', which='major', direction='in')
    # ax.tick_params(axis='y', which='minor', direction='in')
    # ax.tick_params(axis='y', which='major', direction='in')
    # ax.set_xlim([zmin, zmax])
    # ax.set_ylim([-1.1*b0, 1.1*b0])
    # ax.tick_params(labelsize=12)
    # ax.set_xlabel(r'$z/d_e$', fontsize=16)
    # img_dir = '../img/rate_problem/calc_bxm/' + pic_run + '/'
    # mkdir_p(img_dir)
    # fname = img_dir + "bfit_" + str(tframe) + ".pdf"
    # fig.savefig(fname)

    # fig = plt.figure(figsize=[7, 5])
    # rect = [0.12, 0.15, 0.8, 0.8]
    # ax = fig.add_axes(rect)
    # ax.set_prop_cycle('color', COLORS)
    # ax.plot(zgrid, pcut, linewidth=2, label=r'$P$')
    # ax.plot(zgrid, 0.5*bcut**2, linewidth=2, label=r'$B_x^2/8\pi$')
    # ylim = ax.get_ylim()
    # ax.plot([zgrid[iz_min1], zgrid[iz_min1]], ylim, linewidth=1,
    #         linestyle='--', color='k')
    # ax.plot([zgrid[iz_min2], zgrid[iz_min2]], ylim, linewidth=1,
    #         linestyle='--', color='k')
    # ax.set_ylim(ylim)
    # ax.tick_params(bottom=True, top=True, left=True, right=True)
    # ax.tick_params(axis='x', which='minor', direction='in')
    # ax.tick_params(axis='x', which='major', direction='in')
    # ax.tick_params(axis='y', which='minor', direction='in')
    # ax.tick_params(axis='y', which='major', direction='in')
    # ax.set_xlim([zmin, zmax])
    # ax.tick_params(labelsize=12)
    # ax.legend(loc=7, prop={'size': 16}, ncol=1,
    #           shadow=False, fancybox=False, frameon=False)
    # ax.set_xlabel(r'$z/d_e$', fontsize=16)
    # img_dir = '../img/rate_problem/calc_bxm/' + pic_run + '/'
    # mkdir_p(img_dir)
    # fname = img_dir + "calc_bxm_" + str(tframe) + ".pdf"
    # fig.savefig(fname)

    # fig = plt.figure(figsize=[7, 5])
    # rect = [0.12, 0.15, 0.8, 0.8]
    # ax = fig.add_axes(rect)
    # ax.set_prop_cycle('color', COLORS)
    # ax.plot(zgrid, bcut, linewidth=2, label=r'$B_x$')
    # ylim = ax.get_ylim()
    # ax.plot([zgrid[iz_min1], zgrid[iz_min1]], ylim, linewidth=1,
    #         linestyle='--', color='k')
    # ax.plot([zgrid[iz_min2], zgrid[iz_min2]], ylim, linewidth=1,
    #         linestyle='--', color='k')
    # ax.set_ylim(ylim)
    # ax.tick_params(bottom=True, top=True, left=True, right=True)
    # ax.tick_params(axis='x', which='minor', direction='in')
    # ax.tick_params(axis='x', which='major', direction='in')
    # ax.tick_params(axis='y', which='minor', direction='in')
    # ax.tick_params(axis='y', which='major', direction='in')
    # ax.set_xlim([zmin, zmax])
    # ax.tick_params(labelsize=12)
    # ax.legend(loc=7, prop={'size': 16}, ncol=1,
    #           shadow=False, fancybox=False, frameon=False)
    # ax.set_xlabel(r'$z/d_e$', fontsize=16)
    # img_dir = '../img/rate_problem/calc_bxm/' + pic_run + '/'
    # mkdir_p(img_dir)
    # fname = img_dir + "bx_" + str(tframe) + ".pdf"
    # fig.savefig(fname)

    # if show_plot:
    #     plt.show()
    # else:
    #     plt.close("all")


def plot_bxm(plot_config, show_plot=True):
    """
    """
    pic_run = plot_config["pic_run"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_run_dir = pic_info.run_dir
    b0 = pic_info.b0
    fdir = '../data/rate_problem/bxm/' + pic_run + '/'
    tstart = plot_config["tstart"]
    tend = plot_config["tend"]
    nframes = len(os.listdir(fdir)) // 2
    bxm = np.zeros(nframes)
    vpic_info = get_vpic_info(pic_run_dir)
    fields_interval = vpic_info["fields_interval"]
    tfields_wci = np.arange(nframes) * pic_info.dtwci * fields_interval
    tmin, tmax = tfields_wci.min(), tfields_wci.max()
    for tframe in range(nframes):
        fname = fdir + 'bxm_' + str(tframe) + '.dat'
        fdata = np.fromfile(fname)
        bxm[tframe-tstart] = fdata[0]

    fig = plt.figure(figsize=[7, 5])
    rect = [0.12, 0.15, 0.8, 0.8]
    ax = fig.add_axes(rect)
    ax.set_prop_cycle('color', COLORS)
    ax.plot(tfields_wci, bxm/b0, linewidth=2)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlim([tmin, tmax])
    ax.tick_params(labelsize=12)
    ax.set_xlabel(r'$t\Omega_{ci}$', fontsize=16)
    ax.set_ylabel(r'$B_{xm}/B_0$', fontsize=16)

    img_dir = '../img/rate_problem/bxm/'
    mkdir_p(img_dir)
    fname = img_dir + "bxm_" + pic_run + ".pdf"
    fig.savefig(fname)

    plt.show()


def plot_bx_edr(plot_config, show_plot=True):
    """Plot magnetic field upstream of the electron diffusion region
    """
    pic_runs = ["mime400_Tb_T0_025",
                "mime1836_Tb_T0_025"]
    mimes = [400, 1836]
    fig = plt.figure(figsize=[7, 5])
    rect = [0.12, 0.15, 0.8, 0.8]
    ax = fig.add_axes(rect)
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    ax.set_prop_cycle('color', COLORS)
    for irun, pic_run in enumerate(pic_runs):
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        pic_run_dir = pic_info.run_dir
        b0 = pic_info.b0
        fdir = '../data/rate_problem/bxm_edr/' + pic_run + '/'
        nframes = len(os.listdir(fdir))
        bxm = np.zeros(nframes)
        vpic_info = get_vpic_info(pic_run_dir)
        fields_interval = vpic_info["fields_interval"]
        tfields_wci = np.arange(nframes) * pic_info.dtwci * fields_interval
        tmin, tmax = tfields_wci.min(), tfields_wci.max()
        for tframe in range(nframes):
            fname = fdir + 'bxm_' + str(tframe) + '.dat'
            fdata = np.fromfile(fname)
            bxm[tframe] = fdata[0]
        ax.plot(tfields_wci, bxm/b0, linewidth=2,
                label=r"$m_i/m_e=" + str(mimes[irun]) + "$")

    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlim([tmin, 20])
    ax.tick_params(labelsize=12)
    ax.legend(loc=1, prop={'size': 16}, ncol=1,
              shadow=False, fancybox=False, frameon=False)
    ax.set_xlabel(r'$t\Omega_{ci}$', fontsize=16)
    ax.set_ylabel(r'$B_{xm}/B_0$', fontsize=16)

    img_dir = '../img/rate_problem/bx_edr/'
    mkdir_p(img_dir)
    fname = img_dir + "bx_edr.pdf"
    fig.savefig(fname)

    plt.show()


def plot_bxm(plot_config, show_plot=True):
    """
    """
    pic_run = plot_config["pic_run"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_run_dir = pic_info.run_dir
    b0 = pic_info.b0
    fdir = '../data/rate_problem/bxm/' + pic_run + '/'
    tstart = plot_config["tstart"]
    tend = plot_config["tend"]
    nframes = len(os.listdir(fdir)) // 2
    bxm = np.zeros(nframes)
    vpic_info = get_vpic_info(pic_run_dir)
    fields_interval = vpic_info["fields_interval"]
    tfields_wci = np.arange(nframes) * pic_info.dtwci * fields_interval
    tmin, tmax = tfields_wci.min(), tfields_wci.max()
    for tframe in range(nframes):
        fname = fdir + 'bxm_' + str(tframe) + '.dat'
        fdata = np.fromfile(fname)
        bxm[tframe-tstart] = fdata[0]

    fig = plt.figure(figsize=[7, 5])
    rect = [0.12, 0.15, 0.8, 0.8]
    ax = fig.add_axes(rect)
    ax.set_prop_cycle('color', COLORS)
    ax.plot(tfields_wci, bxm/b0, linewidth=2)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlim([tmin, tmax])
    ax.tick_params(labelsize=12)
    ax.set_xlabel(r'$t\Omega_{ci}$', fontsize=16)
    ax.set_ylabel(r'$B_{xm}/B_0$', fontsize=16)

    img_dir = '../img/rate_problem/bxm/'
    mkdir_p(img_dir)
    fname = img_dir + "bxm_" + pic_run + ".pdf"
    fig.savefig(fname)

    plt.show()


def plot_bxm_beta(plot_config, show_plot=True):
    """
    """
    pic_runs = ["mime400_Tb_T0_025",
                "mime400_Tb_T0_1",
                "mime400_Tb_T0_10_weak",
                "mime400_Tb_T0_40_nppc450"]
    betas = [0.25, 1.0, 10, 40]
    fig = plt.figure(figsize=[7, 5])
    rect = [0.12, 0.15, 0.8, 0.8]
    ax = fig.add_axes(rect)
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    ax.set_prop_cycle('color', COLORS)
    fixz = True
    for irun, pic_run in enumerate(pic_runs):
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        pic_run_dir = pic_info.run_dir
        b0 = pic_info.b0
        fdir = '../data/rate_problem/bxm/' + pic_run + '/'
        nframes = len(os.listdir(fdir)) // 2
        bxm = np.zeros(nframes)
        vpic_info = get_vpic_info(pic_run_dir)
        fields_interval = vpic_info["fields_interval"]
        tfields_wci = np.arange(nframes) * pic_info.dtwci * fields_interval
        tmin, tmax = tfields_wci.min(), tfields_wci.max()
        for tframe in range(nframes):
            if fixz:
                fname = fdir + 'bxm_fix_' + str(tframe) + '.dat'
            else:
                fname = fdir + 'bxm_' + str(tframe) + '.dat'
            fdata = np.fromfile(fname)
            bxm[tframe] = fdata[0]
        ax.plot(tfields_wci, bxm/b0, linewidth=2,
                label=r"$\beta=" + str(betas[irun]) + "$")

    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlim([tmin, tmax])
    ax.tick_params(labelsize=12)
    ax.legend(loc=3, prop={'size': 16}, ncol=1,
              shadow=False, fancybox=False, frameon=False)
    ax.set_xlabel(r'$t\Omega_{ci}$', fontsize=16)
    ax.set_ylabel(r'$B_{xm}/B_0$', fontsize=16)

    img_dir = '../img/rate_problem/bxm/'
    mkdir_p(img_dir)
    if fixz:
        fname = img_dir + "bxm_beta_fixz.pdf"
    else:
        fname = img_dir + "bxm_beta.pdf"
    fig.savefig(fname)

    plt.show()


def plot_energy_conversion(plot_config, show_plot=True):
    """Plot energy conversion for different runs
    """
    pic_runs = ["mime400_Tb_T0_025",
                "mime400_nb_n0_1",
                "mime400_Tb_T0_10",
                "mime400_Tb_T0_40"]
    tframe = plot_config["tframe"]
    fig = plt.figure(figsize=[7, 5])
    rect = [0.15, 0.12, 0.8, 0.8]
    ax = fig.add_axes(rect)
    ax.set_prop_cycle('color', COLORS)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlabel(r'$t\Omega_{ci}$', fontsize=16)
    ax.tick_params(labelsize=12)
    labels = [r"$\beta=0.25$", r"$\beta=1$", r"$\beta=10$", r"$\beta=40$"]
    for irun, pic_run in enumerate(pic_runs):
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        pic_run_dir = pic_info.run_dir
        tenergy = pic_info.tenergy * pic_info.dtwci / pic_info.dtwpe
        enorm = pic_info.ene_magnetic[0]
        dt_ene = pic_info.dtwpe * pic_info.energy_interval
        tenergy *= pic_info.dtwpe / pic_info.dtwci

        ene_bx = pic_info.ene_bx
        ene_by = pic_info.ene_by
        ene_bz = pic_info.ene_bz
        ene_magnetic = pic_info.ene_magnetic
        ene_electric = pic_info.ene_electric
        kene_e = pic_info.kene_e
        kene_i = pic_info.kene_i

        # etot = ene_magnetic + ene_electric + kene_e + kene_i
        # print("Energy conservation: %e" % ((etot[-1] - etot[0]) / etot[0]))
        # print("Energy conversion: %e" %
        #       ((ene_magnetic[-1] - ene_magnetic[0]) / ene_magnetic[0]))

        ene_bx /= enorm
        ene_by /= enorm
        ene_bz /= enorm
        ene_magnetic /= enorm
        ene_electric /= enorm
        kene_e /= enorm
        kene_i /= enorm

        p1, = ax.plot(tenergy, kene_e - kene_e[0], linewidth=2, label=labels[irun])
        p2, = ax.plot(tenergy, kene_i - kene_i[0], linewidth=2,
                      linestyle='--', color=p1.get_color())
    # ax.set_xlim([xmin, xmax])
    ax.grid(True)
    ax.legend(loc='best', prop={'size': 16}, ncol=1,
              shadow=False, fancybox=False, frameon=False)
    # img_dir = '../img/rate_problem/n_xcut_beta/'
    # mkdir_p(img_dir)
    # fname = img_dir + "n_xcut_beta_" + str(tframe) + ".pdf"
    # fig.savefig(fname)

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_jxb_x(plot_config, show_plot=True):
    """Plot the x-component of jxB
    """
    tframe = plot_config["tframe"]
    pic_run = plot_config["pic_run"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_run_dir = pic_info.run_dir
    fields_interval = pic_info.fields_interval
    tindex = fields_interval * tframe
    smime = math.sqrt(pic_info.mime)
    lx_de = pic_info.lx_di * smime
    lz_de = pic_info.lz_di * smime
    nx = pic_info.nx
    nz = pic_info.nz
    xs = nx//2 - nx//4
    xe = nx//2 + nx//4
    zs = nz//2 - nz//16
    ze = nz//2 + nz//16
    dx_de = lx_de / nx
    dz_de = lz_de / nz
    xmin = xs * dz_de
    xmax = xe * dz_de
    zmin = zs * dz_de - lz_de * 0.5
    zmax = ze * dz_de - lz_de * 0.5

    fname = (pic_run_dir + "hydro_hdf5/T." + str(tindex) +
             "/hydro_electron_" + str(tindex) + ".h5")
    je = {}
    with h5py.File(fname, 'r') as fh:
        group = fh["Timestep_" + str(tindex)]
        for var in ["jx", "jy", "jz"]:
            dset = group[var]
            je[var]= dset[xs:xe, 0, zs:ze]

    fname = (pic_run_dir + "hydro_hdf5/T." + str(tindex) +
             "/hydro_ion_" + str(tindex) + ".h5")
    ji = {}
    with h5py.File(fname, 'r') as fh:
        group = fh["Timestep_" + str(tindex)]
        for var in ["jy", "jz"]:
            dset = group[var]
            ji[var]= dset[xs:xe, 0, zs:ze]

    jy = np.squeeze(je["jy"] + ji["jy"])
    jz = np.squeeze(je["jz"] + ji["jz"])

    fname = (pic_run_dir + "field_hdf5/T." + str(tindex) +
             "/fields_" + str(tindex) + ".h5")
    bvec = {}
    with h5py.File(fname, 'r') as fh:
        group = fh["Timestep_" + str(tindex)]
        for var in ["cbx", "cby", "cbz"]:
            dset = group[var]
            bvec[var]= dset[xs:xe, 0, zs:ze]
    bx = np.squeeze(bvec["cbx"])
    by = np.squeeze(bvec["cby"])
    bz = np.squeeze(bvec["cbz"])
    b2 = bx**2 + by**2 + bz**2
    tension0_x = np.gradient(bx, dx_de, axis=0) * bx
    tension0_z = np.gradient(bx, dz_de, axis=1) * bz
    gradx_b2 = np.gradient(b2, dx_de, axis=0)
    tension0 = tension0_x + tension0_z

    jxb_x1 = jy * bz
    jxb_x2 = jz * by
    jxb_x = jxb_x1 - jxb_x2

    sigma = 3
    gradx_b2 = gaussian_filter(gradx_b2, sigma=sigma)
    tension0 = gaussian_filter(tension0, sigma=sigma)
    jxb_x1 = gaussian_filter(jxb_x1, sigma=sigma)
    jxb_x2 = gaussian_filter(jxb_x2, sigma=sigma)
    jxb_x = gaussian_filter(jxb_x, sigma=sigma)

    xmin_di = xmin / smime
    xmax_di = xmax / smime
    zmin_di = zmin / smime
    zmax_di = zmax / smime
    kwargs = {"current_time": tframe,
              "xl": xmin_di, "xr": xmax_di,
              "zb": zmin_di, "zt": zmax_di}
    fname = pic_run_dir + "data/Ay.gda"
    x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
    xde = x * smime
    zde = z * smime

    fig = plt.figure(figsize=[7, 11])
    rect = [0.12, 0.83, 0.76, 0.13]
    hgap, vgap = 0.02, 0.025
    nvar = 6
    var_names = [r"$j_yB_z$", r"$-j_zB_y$",
                 r"$(\boldsymbol{j}\times\boldsymbol{B})_x$",
                 r"$(\boldsymbol{B}\cdot\nabla)\boldsymbol{B}$",
                 r"$-\nabla(B^2/2)$"]
    axs = []
    for ivar in range(nvar):
        ax = fig.add_axes(rect)
        axs.append(ax)
        rect[1] -= rect[3] + vgap
        ax.set_prop_cycle('color', COLORS)
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')
        if ivar < nvar - 1:
            ax.contour(xde, zde, Ay, colors='k', linewidths=0.5,
                       levels=np.linspace(np.min(Ay), np.max(Ay), 10))
            ax.text(0.02, 0.85, var_names[ivar], color='k', fontsize=16,
                    bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                    horizontalalignment='left', verticalalignment='center',
                    transform=ax.transAxes)
        if ivar == nvar-1:
            ax.set_xlabel(r'$x/d_e$', fontsize=16)
        if ivar < nvar - 1:
            ax.set_ylabel(r'$z/d_e$', fontsize=16)
        else:
            ax.set_ylabel('Accumulation', fontsize=16)
        ax.tick_params(labelsize=12)
        ax.set_xlim([xmin, xmax])
        if ivar < nvar - 1:
            ax.set_ylim([zmin, zmax])

    dmax = 3E-3
    dmin = -dmax
    cmap = plt.cm.seismic
    im1 = axs[0].imshow(jxb_x1.T,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=dmin, vmax=dmax,
                        cmap=cmap, aspect='auto',
                        origin='lower', interpolation='bicubic')
    im2 = axs[1].imshow(jxb_x2.T,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=dmin, vmax=dmax,
                        cmap=cmap, aspect='auto',
                        origin='lower', interpolation='bicubic')
    im3 = axs[2].imshow(jxb_x.T,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=dmin, vmax=dmax,
                        cmap=cmap, aspect='auto',
                        origin='lower', interpolation='bicubic')
    im4 = axs[3].imshow(tension0.T,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=dmin, vmax=dmax,
                        cmap=cmap, aspect='auto',
                        origin='lower', interpolation='bicubic')
    im5 = axs[4].imshow(-0.5 * gradx_b2.T,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=dmin, vmax=dmax,
                        cmap=cmap, aspect='auto',
                        origin='lower', interpolation='bicubic')

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
    ix_xp = int(x0 / dx_de) - xs

    xgrid = np.arange(xs, xe) * dx_de
    dv = dx_de * dz_de
    work_force = np.zeros([nvar, xe-xs])
    work_force_int = np.zeros([nvar, xe-xs])
    work_force[0] = np.sum(jxb_x1, axis=1)
    work_force[1] = np.sum(jxb_x2, axis=1)
    work_force[2] = np.sum(jxb_x, axis=1)
    work_force[3] = np.sum(tension0, axis=1)
    work_force[4] = np.sum(-0.5*gradx_b2, axis=1)
    for ivar in range(nvar - 1):
        work_force_int[:, ix_xp::-1] = np.cumsum(work_force[:, ix_xp::-1], axis=1)
        work_force_int[:, ix_xp:] = np.cumsum(work_force[:, ix_xp:], axis=1)
        axs[5].plot(xgrid, work_force_int[ivar], label=var_names[ivar])
    axs[5].legend(loc=4, prop={'size': 10}, ncol=2,
                  shadow=False, fancybox=False, frameon=False)

    rect_cbar = np.copy(rect)
    rect_cbar[0] += rect[2] + 0.02
    rect_cbar[2] = 0.02
    rect_cbar[1] += rect[3] * 2.5 + vgap
    rect_cbar[3] = (rect[3] + vgap) * 3
    cbar_ax = fig.add_axes(rect_cbar)
    cbar = fig.colorbar(im1, cax=cbar_ax, extend='both')
    cbar_ax.tick_params(bottom=False, top=False, left=False, right=True)
    cbar_ax.tick_params(axis='y', which='major', direction='out')
    cbar_ax.tick_params(axis='y', which='minor', direction='in', right=False)
    twpe = math.ceil(tindex * pic_info.dtwpe / 0.1) * 0.1
    text1 = r'$t\omega_{pe}=' + ("{%0.0f}" % twpe) + '$'
    fig.suptitle(text1, fontsize=20)
    img_dir = '../img/rate_problem/jxb_x/' + pic_run + '/'
    mkdir_p(img_dir)
    fname = img_dir + "jxb_x_" + str(tframe) + ".jpg"
    fig.savefig(fname, dpi=200)

    if show_plot:
        plt.show()
    else:
        plt.close()


def fluid_energization(plot_config, show_plot=True):
    """Plot fluid energization

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_run_dir = pic_info.run_dir
    if species == 'e':
        ylim = [-700, 700]
    else:
        ylim = [-700, 700]
    fig1 = plt.figure(figsize=[9, 3.0])
    box1 = [0.1, 0.18, 0.85, 0.68]
    axs1 = []
    # fig2 = plt.figure(figsize=[9, 3.0])
    # axs2 = []
    # fig3 = plt.figure(figsize=[9, 3.0])
    # axs3 = []
    # fig4 = plt.figure(figsize=[9, 3.0])
    # axs4 = []

    tfields = pic_info.tfields
    tenergy = pic_info.tenergy
    fname = "../data/fluid_energization/" + pic_run + "/"
    fname += "emf_ptensor_" + species + '.gda'
    fluid_ene = np.fromfile(fname, dtype=np.float32)
    nvar = int(fluid_ene[0])
    nframes = int(fluid_ene[1])
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
    acc_drift_dote_t_perp = fluid_ene[2:nframes+2]
    acc_drift_dote_t_para = fluid_ene[nframes+2:2*nframes+2]
    acc_drift_dote_s_perp = fluid_ene[2*nframes+2:3*nframes+2]
    acc_drift_dote_s_para = fluid_ene[3*nframes+2:4*nframes+2]
    acc_drift_dote_t = acc_drift_dote_t_para + acc_drift_dote_t_perp
    acc_drift_dote_s = acc_drift_dote_s_para + acc_drift_dote_s_perp
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

    if nframes < pic_info.ntf:
        tfields_adjust = tfields[:(nframes-pic_info.ntf)]
    else:
        tfields_adjust = tfields

    ax = fig1.add_axes(box1)
    axs1.append(ax)
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    ax.set_prop_cycle('color', COLORS)
    label2 = (r'$\boldsymbol{j}_{' + species + '\perp}' +
              r'\cdot\boldsymbol{E}_\perp$')
    ax.plot(tfields_adjust, eperp_ene, linewidth=1, label=label2)
    label1 = (r'$\boldsymbol{j}_{' + species + '\parallel}' +
              r'\cdot\boldsymbol{E}_\parallel$')
    ax.plot(tfields_adjust, epara_ene, linewidth=1, label=label1)
    label6 = r'$dK_' + species + '/dt$'
    ax.plot(tenergy, dkene, linewidth=1, label=label6)
    label3 = r'$(\nabla\cdot\tensorsym{P}_' + species + r')\cdot\boldsymbol{v}_E$'
    ax.plot(tfields_adjust, ptensor_ene, linewidth=1, label=label3)
    ax.plot([0, tenergy.max()], [0, 0], color='k', linestyle='--')
    ax.set_xlim([0, np.max(tfields_adjust)])
    # ax.set_ylim(ylim)
    ax.tick_params(labelsize=10)
    ax.set_ylabel('Energization', fontsize=12)
    ax.set_xlabel(r'$t\Omega_{ci}$', fontsize=12)

#     ax = fig2.add_axes(box1)
#     axs2.append(ax)
#     # COLORS = palettable.colorbrewer.qualitative.Set1_9.mpl_colors
#     ax.set_prop_cycle('color', COLORS)
#     label2 = (r'$\boldsymbol{j}_{' + species + '\perp}' +
#               r'\cdot\boldsymbol{E}_\perp$')
#     ax.plot(tfields_adjust, eperp_ene, linewidth=1, label=label2)
#     ax.plot(tfields_adjust, curv_drift_dote, linewidth=1, label='Curvature')
#     ax.plot(tfields_adjust, grad_drift_dote, linewidth=1, label='Gradient')
#     ax.plot(tfields_adjust, magnetization_dote, linewidth=1, label='Magnetization')
#     ax.plot([0, tenergy.max()], [0, 0], color='k', linestyle='--')
#     ax.set_xlim([0, np.max(tfields_adjust)])
#     # ax.set_ylim(ylim)
#     ax.tick_params(labelsize=10)
#     ax.set_ylabel('Energization', fontsize=12)
#     ax.set_xlabel(r'$t\omega_{pe}$', fontsize=12)

#     ax = fig3.add_axes(box1)
#     axs3.append(ax)
#     # COLORS = palettable.colorbrewer.qualitative.Set1_9.mpl_colors
#     ax.set_prop_cycle('color', COLORS)
#     label2 = (r'$\boldsymbol{j}_{' + species + '\perp}' +
#               r'\cdot\boldsymbol{E}_\perp' + '$')
#     ax.plot(tfields_adjust, eperp_ene, linewidth=1, label=label2)
#     ax.plot(tfields_adjust, comp_ene, linewidth=1, label='Compression')
#     ax.plot(tfields_adjust, shear_ene, linewidth=1, label='Shear')
#     # label2 = (r'$\boldsymbol{j}_{' + species + '\perp}' +
#     #           r'\cdot\boldsymbol{E}_\perp -' + 'n_' + species +
#     #           'm_' + species + r'(d\boldsymbol{u}_' + species +
#     #           r'/dt)\cdot\boldsymbol{v}_E$')
#     # ax.plot(tfields_adjust, eperp_ene - acc_drift_dote, linewidth=1, label=label2)
#     label4 = (r'$\boldsymbol{j}_{' + species + r'-\text{agy}}' +
#               r'\cdot\boldsymbol{E}_\perp$')
#     ax.plot(tfields_adjust, jagy_dote, linewidth=1, label=label4)
#     # jdote_sum = comp_ene + shear_ene + jagy_dote
#     # ax.plot(tfields_adjust, jdote_sum, linewidth=1)
#     ax.plot([0, tenergy.max()], [0, 0], color='k', linestyle='--')
#     ax.set_xlim([0, np.max(tfields_adjust)])
#     # ax.set_ylim(ylim)
#     ax.tick_params(labelsize=10)
#     ax.set_ylabel('Energization', fontsize=12)
#     ax.set_xlabel(r'$t\omega_{pe}$', fontsize=12)

#     ax = fig4.add_axes(box1)
#     axs4.append(ax)
#     ax.set_prop_cycle('color', COLORS)
#     label2 = (r'$\boldsymbol{j}_{' + species + '\perp}' +
#               r'\cdot\boldsymbol{E}_\perp$')
#     ax.plot(tfields_adjust, eperp_ene, linewidth=1, label=label2)
#     jdote_sum = (curv_drift_dote + grad_drift_dote + magnetization_dote +
#                  jagy_dote + acc_drift_dote)
#     ax.plot(tfields_adjust, jdote_sum, linewidth=1,
#             label='Drifts+Magnetization+Agyrotropic')
#     ax.plot(tfields_adjust, acc_drift_dote, linewidth=1, label='Flow inertial')
#     ax.plot([0, tenergy.max()], [0, 0], color='k', linestyle='--')
#     ax.set_xlim([0, np.max(tfields_adjust)])
#     # ax.set_ylim(ylim)
#     ax.tick_params(labelsize=10)
#     ax.set_ylabel('Energization', fontsize=12)
#     ax.set_xlabel(r'$t\omega_{pe}$', fontsize=12)

    axs1[0].legend(loc='upper center', prop={'size': 12}, ncol=4,
                   bbox_to_anchor=(0.5, 1.22),
                   shadow=False, fancybox=False, frameon=False)
    # axs2[0].legend(loc='upper center', prop={'size': 12}, ncol=4,
    #                bbox_to_anchor=(0.5, 1.22),
    #                shadow=False, fancybox=False, frameon=False)
    # axs3[0].legend(loc='upper center', prop={'size': 12}, ncol=4,
    #                bbox_to_anchor=(0.5, 1.22),
    #                shadow=False, fancybox=False, frameon=False)
    # axs4[0].legend(loc='upper center', prop={'size': 12}, ncol=4,
    #                bbox_to_anchor=(0.5, 1.22),
    #                shadow=False, fancybox=False, frameon=False)
    # fdir = '../img/power_law_index/fluid_energization/' + pic_run + '/'
    # mkdir_p(fdir)
    # fname = fdir + 'fluid_ene_' + species + '.pdf'
    # fig1.savefig(fname)
    # fname = fdir + 'fluid_drift_' + species + '.pdf'
    # fig2.savefig(fname)
    # fname = fdir + 'fluid_comp_shear_' + species + '.pdf'
    # fig3.savefig(fname)
    # fname = fdir + 'polar_total_perp' + species + '.pdf'
    # fig4.savefig(fname)

    if show_plot:
        plt.show()
    else:
        plt.close('all')


def fluid_energization_2d(plot_config, show_plot=True):
    """Plot 2D fluid energization terms
    """
    pic_run = plot_config["pic_run"]
    tframe = plot_config["tframe"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_run_dir = pic_info.run_dir
    fields_interval = pic_info.fields_interval
    tindex = fields_interval * tframe
    smime = math.sqrt(pic_info.mime)
    lx_de = pic_info.lx_di * smime
    lz_de = pic_info.lz_di * smime
    nx = pic_info.nx
    nz = pic_info.nz
    xs = nx//2 - nx//4
    xe = nx//2 + nx//4
    zs = nz//2 - nz//16
    ze = nz//2 + nz//16
    dx_de = lx_de / nx
    dz_de = lz_de / nz
    nxs = xe - xs
    nzs = ze - zs
    xmin = xs * dz_de
    xmax = xe * dz_de
    zmin = zs * dz_de - lz_de * 0.5
    zmax = ze * dz_de - lz_de * 0.5

    vpic_info = get_vpic_info(pic_run_dir)
    n0 = vpic_info["n0"]
    b0 = vpic_info["b0"]
    Te = vpic_info["Te"]
    Tbe_Te = vpic_info["Tbe/Te"]
    nb_n0 = vpic_info["nb/n0"]
    nb = n0 * nb_n0
    p0 = nb * Te * Tbe_Te
    Tb = Te * Tbe_Te

    plot_config["pic_run"] = pic_run
    plot_config["pic_run_dir"] = pic_run_dir
    start = (xs, 0, zs)
    count = (nxs, 1, nzs)

    sigma = 3
    fname = (pic_run_dir + "field_hdf5/T." + str(tindex) +
             "/fields_" + str(tindex) + ".h5")
    eb = {}
    with h5py.File(fname, 'r') as fh:
        group = fh["Timestep_" + str(tindex)]
        for var in group:
            dset = group[var]
            eb[var] = gaussian_filter(dset[xs:xe, 0, zs:ze], sigma=sigma)

    species = plot_config["species"]
    sname = "electron" if species == 'e' else "ion"
    fname = (pic_run_dir + "hydro_hdf5/T." + str(tindex) +
             "/hydro_" + sname + "_" + str(tindex) + ".h5")
    hydro = {}
    with h5py.File(fname, 'r') as fh:
        group = fh["Timestep_" + str(tindex)]
        for var in group:
            dset = group[var]
            hydro[var] = gaussian_filter(dset[xs:xe, 0, zs:ze], sigma=sigma)

    b2 = eb["cbx"]**2 + eb["cby"]**2 + eb["cbz"]**2
    jdotb = (hydro["jx"] * eb["cbx"] +
             hydro["jy"] * eb["cby"] +
             hydro["jz"] * eb["cbz"])
    jpara_dote = (eb["cbx"] * eb["ex"] +
                  eb["cby"] * eb["ey"] +
                  eb["cbz"] * eb["ez"]) * jdotb / b2
    jdote = (hydro["jx"] * eb["ex"] +
             hydro["jy"] * eb["ey"] +
             hydro["jz"] * eb["ez"])
    jperp_dote = jdote - jpara_dote

    # Energization associated with pressure tensor
    ib2 = 1.0 / b2
    vexb_x = (eb["ey"] * eb["cbz"] - eb["ez"] * eb["cby"]) * ib2
    vexb_y = (eb["ez"] * eb["cbx"] - eb["ex"] * eb["cbz"]) * ib2
    vexb_z = (eb["ex"] * eb["cby"] - eb["ey"] * eb["cbx"]) * ib2
    irho = 1.0 / hydro["rho"]
    vx = hydro["jx"] * irho
    vy = hydro["jy"] * irho
    vz = hydro["jz"] * irho
    pxx = hydro["txx"] - vx * hydro["px"]
    pyy = hydro["tyy"] - vy * hydro["py"]
    pzz = hydro["tzz"] - vz * hydro["pz"]
    pyx = hydro["txy"] - vx * hydro["py"]
    pxz = hydro["tzx"] - vz * hydro["px"]
    pzy = hydro["tyz"] - vy * hydro["pz"]
    pxy = hydro["txy"] - vy * hydro["px"]
    pyz = hydro["tyz"] - vz * hydro["py"]
    pzx = hydro["tzx"] - vx * hydro["pz"]
    divpt_vexb = ((np.gradient(pxx, dx_de, axis=0) +
                   np.gradient(pxz, dz_de, axis=1)) * vexb_x +
                  (np.gradient(pyx, dx_de, axis=0) +
                   np.gradient(pyz, dz_de, axis=1)) * vexb_y +
                  (np.gradient(pzx, dx_de, axis=0) +
                   np.gradient(pzz, dz_de, axis=1)) * vexb_z)
    divpt_v = ((np.gradient(pxx, dx_de, axis=0) +
                np.gradient(pxz, dz_de, axis=1)) * vx +
               (np.gradient(pyx, dx_de, axis=0) +
                np.gradient(pyz, dz_de, axis=1)) * vy +
               (np.gradient(pzx, dx_de, axis=0) +
                np.gradient(pzz, dz_de, axis=1)) * vz)
    pvx = pxx * vx + pxy * vy + pxz * vz
    pvz = pzx * vx + pzy * vy + pzz * vz
    div_pv = (np.gradient(pvx, dx_de, axis=0) +
              np.gradient(pvz, dx_de, axis=1))

    ppara = (pxx * eb["cbx"]**2 +
             pyy * eb["cby"]**2 +
             pzz * eb["cbz"]**2 +
             (pxy + pyx) * eb["cbx"] * eb["cby"] +
             (pxz + pzx) * eb["cbx"] * eb["cbz"] +
             (pyz + pzy) * eb["cby"] * eb["cbz"]) * ib2
    pperp = 0.5 * (pxx + pyy + pzz - ppara)
    pscalar = (ppara + 2 * pperp) / 3
    ib = np.sqrt(ib2)
    absB = 1 / ib
    bx = eb["cbx"] * ib
    by = eb["cby"] * ib
    bz = eb["cbz"] * ib
    dvx_dx = np.gradient(vx, dx_de, axis=0)
    dvy_dx = np.gradient(vy, dx_de, axis=0)
    dvz_dx = np.gradient(vz, dx_de, axis=0)
    dvx_dz = np.gradient(vx, dz_de, axis=1)
    dvy_dz = np.gradient(vy, dz_de, axis=1)
    dvz_dz = np.gradient(vz, dz_de, axis=1)
    divv = dvx_dx + dvz_dz
    divv3 = divv / 3
    shear = ((dvx_dx - divv3) * bx**2 +
             (dvz_dz - divv3) * bz**2 +
             (-divv3) * by**2 +
             dvy_dx * bx * by + dvy_dz * by * bz +
             (dvx_dz + dvz_dx) * bx * bz)
    pdivv = -pscalar * divv
    pshear = (pperp - ppara) * shear

    # Energization associated with curvature drift and gradient drift
    kappax = (bx * np.gradient(bx, dx_de, axis=0) +
              bz * np.gradient(bx, dz_de, axis=1))
    kappay = (bx * np.gradient(by, dx_de, axis=0) +
              bz * np.gradient(by, dz_de, axis=1))
    kappaz = (bx * np.gradient(bz, dx_de, axis=0) +
              bz * np.gradient(bz, dz_de, axis=1))
    jcurv_dote = (kappax * vexb_x +
                  kappay * vexb_y +
                  kappaz * vexb_z) * ppara
    gradb_x = np.gradient(absB, dx_de, axis=0)
    gradb_z = np.gradient(absB, dz_de, axis=1)
    jgrad_dote = (gradb_x * vexb_x +
                  gradb_z * vexb_z) * ib * pperp

    dmax = 1E-4
    dmin = -dmax
    fig = plt.figure(figsize=[7, 13])
    rect = [0.12, 0.87, 0.72, 0.1]
    hgap, vgap = 0.02, 0.018
    nvar = 8
    axs = []
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    for ivar in range(nvar):
        ax = fig.add_axes(rect)
        axs.append(ax)
        rect[1] -= rect[3] + vgap
        ax.set_prop_cycle('color', COLORS)
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')
        if ivar == nvar - 1:
            ax.set_xlabel(r'$x/d_e$', fontsize=16)
        if ivar < nvar - 1:
            ax.set_ylabel(r'$z/d_e$', fontsize=16)
        else:
            ax.set_ylabel('Accumulation', fontsize=16)
        ax.tick_params(labelsize=12)
        ax.set_xlim([xmin, xmax])
        if ivar < nvar - 1:
            ax.set_ylim([zmin, zmax])

    cmap = plt.cm.seismic
    im1 = axs[0].imshow(jpara_dote.T,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=dmin, vmax=dmax,
                        cmap=cmap, aspect='auto',
                        origin='lower', interpolation='bicubic')
    im2 = axs[1].imshow(jperp_dote.T,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=dmin, vmax=dmax,
                        cmap=cmap, aspect='auto',
                        origin='lower', interpolation='bicubic')
    im3 = axs[2].imshow(jdote.T,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=dmin, vmax=dmax,
                        cmap=cmap, aspect='auto',
                        origin='lower', interpolation='bicubic')
    # im4 = axs[3].imshow(divpt_vexb.T,
    #                     extent=[xmin, xmax, zmin, zmax],
    #                     vmin=dmin, vmax=dmax,
    #                     cmap=cmap, aspect='auto',
    #                     origin='lower', interpolation='bicubic')
    im4 = axs[3].imshow(divpt_v.T,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=dmin, vmax=dmax,
                        cmap=cmap, aspect='auto',
                        origin='lower', interpolation='bicubic')
    im5 = axs[4].imshow(pdivv.T,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=dmin, vmax=dmax,
                        cmap=cmap, aspect='auto',
                        origin='lower', interpolation='bicubic')
    im6 = axs[5].imshow(pshear.T,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=dmin, vmax=dmax,
                        cmap=cmap, aspect='auto',
                        origin='lower', interpolation='bicubic')
    im7 = axs[6].imshow(div_pv.T,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=dmin, vmax=dmax,
                        cmap=cmap, aspect='auto',
                        origin='lower', interpolation='bicubic')
    jpara_dote_zsum = np.sum(jpara_dote, axis=1)
    jperp_dote_zsum = np.sum(jperp_dote, axis=1)
    jdote_zsum = np.sum(jdote, axis=1)
    divpt_vexb_zsum = np.sum(divpt_vexb, axis=1)
    divpt_v_zsum = np.sum(divpt_v, axis=1)
    pdivv_zsum = np.sum(pdivv, axis=1)
    pshear_zsum = np.sum(pshear, axis=1)
    div_pv_zsum = np.sum(div_pv, axis=1)
    # jcurv_dote_zsum = np.sum(jcurv_dote, axis=1)
    # jgrad_dote_zsum = np.sum(jgrad_dote, axis=1)

    # jpara_dote_xsum = np.cumsum(jpara_dote_zsum)
    # jperp_dote_xsum = np.cumsum(jperp_dote_zsum)
    # jdote_xsum = np.cumsum(jdote_zsum)
    # divpt_vexb_xsum = np.cumsum(divpt_vexb_zsum)
    # divpt_v_xsum = np.cumsum(divpt_v_zsum)
    # pdivv_xsum = np.cumsum(pdivv_zsum)
    # pshear_xsum = np.cumsum(pshear_zsum)
    # div_pv_xsum = np.cumsum(div_pv_zsum)
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
    ix_xp = int(x0 / dx_de) - xs

    jpara_dote_xsum = np.zeros(nxs)
    jperp_dote_xsum = np.zeros(nxs)
    jdote_xsum = np.zeros(nxs)
    divpt_vexb_xsum = np.zeros(nxs)
    divpt_v_xsum = np.zeros(nxs)
    pdivv_xsum = np.zeros(nxs)
    pshear_xsum = np.zeros(nxs)
    div_pv_xsum = np.zeros(nxs)
    jpara_dote_xsum[ix_xp::-1] = np.cumsum(jpara_dote_zsum[ix_xp::-1])
    jpara_dote_xsum[ix_xp:] = np.cumsum(jpara_dote_zsum[ix_xp:])
    jperp_dote_xsum[ix_xp::-1] = np.cumsum(jperp_dote_zsum[ix_xp::-1])
    jperp_dote_xsum[ix_xp:] = np.cumsum(jperp_dote_zsum[ix_xp:])
    jdote_xsum[ix_xp::-1] = np.cumsum(jdote_zsum[ix_xp::-1])
    jdote_xsum[ix_xp:] = np.cumsum(jdote_zsum[ix_xp:])
    divpt_vexb_xsum[ix_xp::-1] = np.cumsum(divpt_vexb_zsum[ix_xp::-1])
    divpt_vexb_xsum[ix_xp:] = np.cumsum(divpt_vexb_zsum[ix_xp:])
    divpt_v_xsum[ix_xp::-1] = np.cumsum(divpt_v_zsum[ix_xp::-1])
    divpt_v_xsum[ix_xp:] = np.cumsum(divpt_v_zsum[ix_xp:])
    pdivv_xsum[ix_xp::-1] = np.cumsum(pdivv_zsum[ix_xp::-1])
    pdivv_xsum[ix_xp:] = np.cumsum(pdivv_zsum[ix_xp:])
    pshear_xsum[ix_xp::-1] = np.cumsum(pshear_zsum[ix_xp::-1])
    pshear_xsum[ix_xp:] = np.cumsum(pshear_zsum[ix_xp:])
    div_pv_xsum[ix_xp::-1] = np.cumsum(div_pv_zsum[ix_xp::-1])
    div_pv_xsum[ix_xp:] = np.cumsum(div_pv_zsum[ix_xp:])
    xgrid = np.arange(xs, xe) * dx_de
    axs[nvar-1].plot(xgrid, jpara_dote_xsum)
    axs[nvar-1].plot(xgrid, jperp_dote_xsum)
    axs[nvar-1].plot(xgrid, jdote_xsum)
    # axs[nvar-1].plot(xgrid, divpt_vexb_xsum)
    axs[nvar-1].plot(xgrid, divpt_v_xsum)
    # axs[nvar-1].plot(xgrid, jcurv_dote_xsum)
    # axs[nvar-1].plot(xgrid, jgrad_dote_xsum)
    # axs[nvar-1].plot(xgrid, pdivv_xsum)
    # axs[nvar-1].plot(xgrid, pshear_xsum)
    # axs[nvar-1].plot(xgrid, div_pv_xsum)
    axs[nvar-1].plot([xmin, xmax], [0, 0], color='k', linestyle='--')

    rect_cbar = np.copy(rect)
    rect_cbar[0] += rect[2] + 0.01
    rect_cbar[2] = 0.02
    rect_cbar[1] += (rect[3] + vgap) * 4
    rect_cbar[3] = rect[3] * 3 + vgap * 2
    cbar_ax = fig.add_axes(rect_cbar)
    cbar = fig.colorbar(im1, cax=cbar_ax, extend='both')
    cbar_ax.tick_params(bottom=False, top=False, left=False, right=True)
    cbar_ax.tick_params(axis='y', which='major', direction='out')
    cbar_ax.tick_params(axis='y', which='minor', direction='in', right=False)
    cbar.ax.tick_params(labelsize=12)

    xmin_di = xmin / smime
    xmax_di = xmax / smime
    zmin_di = zmin / smime
    zmax_di = zmax / smime
    kwargs = {"current_time": tframe,
              "xl": xmin_di, "xr": xmax_di,
              "zb": zmin_di, "zt": zmax_di}
    fname = pic_run_dir + "data/Ay.gda"
    x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
    xde = x * smime
    zde = z * smime
    texts = [r'$\boldsymbol{j}_{' + species + r'\parallel}\cdot\boldsymbol{E}$',
             r'$\boldsymbol{j}_{' + species + r'\perp}\cdot\boldsymbol{E}$',
             r'$\boldsymbol{j}_{' + species + r'}\cdot\boldsymbol{E}$',
             r'$(\nabla\cdot\boldsymbol{P})\cdot\boldsymbol{v}$',
             r'$-p\nabla\cdot\boldsymbol{v}$',
             r'$-(p_\parallel-p_\perp)b_ib_j\sigma_{ij}$',
             r'$\nabla\cdot(\mathcal{P}\cdot\boldsymbol{v})$']
    for iax, ax in enumerate(axs[:-1]):
        ax.contour(xde, zde, Ay, colors='k', linewidths=0.5,
                   levels=np.linspace(np.min(Ay), np.max(Ay), 10))
        ax.text(0.03, 0.8, texts[iax], color=COLORS[iax], fontsize=16,
                bbox=dict(facecolor='w', alpha=1.0, boxstyle='round',
                          edgecolor='grey', pad=0.2),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)

    twpe = math.ceil(tindex * pic_info.dtwpe / 0.1) * 0.1
    text1 = r'$t\omega_{pe}=' + ("{%0.0f}" % twpe) + '$'
    fig.suptitle(text1, fontsize=20)
    img_dir = '../img/rate_problem/fluid_ene_2d/' + pic_run + '/'
    mkdir_p(img_dir)
    fname = img_dir + "fluid_ene_2d_" + species + "_" + str(tframe) + ".jpg"
    fig.savefig(fname, dpi=200)

    if show_plot:
        plt.show()
    else:
        plt.close()


def compression_energization(plot_config, show_plot=True):
    """Plot energization terms associated with compression or shear
    """
    pic_run = plot_config["pic_run"]
    tframe = plot_config["tframe"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_run_dir = pic_info.run_dir
    fields_interval = pic_info.fields_interval
    tindex = fields_interval * tframe
    smime = math.sqrt(pic_info.mime)
    lx_de = pic_info.lx_di * smime
    lz_de = pic_info.lz_di * smime
    nx = pic_info.nx
    nz = pic_info.nz
    xs = nx//2 - nx//4
    xe = nx//2 + nx//4
    zs = nz//2 - nz//16
    ze = nz//2 + nz//16
    dx_de = lx_de / nx
    dz_de = lz_de / nz
    nxs = xe - xs
    nzs = ze - zs
    xmin = xs * dz_de
    xmax = xe * dz_de
    zmin = zs * dz_de - lz_de * 0.5
    zmax = ze * dz_de - lz_de * 0.5

    vpic_info = get_vpic_info(pic_run_dir)
    n0 = vpic_info["n0"]
    b0 = vpic_info["b0"]
    Te = vpic_info["Te"]
    Tbe_Te = vpic_info["Tbe/Te"]
    nb_n0 = vpic_info["nb/n0"]
    nb = n0 * nb_n0
    p0 = nb * Te * Tbe_Te
    Tb = Te * Tbe_Te

    plot_config["pic_run"] = pic_run
    plot_config["pic_run_dir"] = pic_run_dir
    start = (xs, 0, zs)
    count = (nxs, 1, nzs)

    sigma = 3
    fname = (pic_run_dir + "field_hdf5/T." + str(tindex) +
             "/fields_" + str(tindex) + ".h5")
    eb = {}
    with h5py.File(fname, 'r') as fh:
        group = fh["Timestep_" + str(tindex)]
        for var in group:
            dset = group[var]
            eb[var] = gaussian_filter(dset[xs:xe, 0, zs:ze], sigma=sigma)

    species = plot_config["species"]
    sname = "electron" if species == 'e' else "ion"
    fname = (pic_run_dir + "hydro_hdf5/T." + str(tindex) +
             "/hydro_" + sname + "_" + str(tindex) + ".h5")
    hydro = {}
    with h5py.File(fname, 'r') as fh:
        group = fh["Timestep_" + str(tindex)]
        for var in group:
            dset = group[var]
            hydro[var] = gaussian_filter(dset[xs:xe, 0, zs:ze], sigma=sigma)

    b2 = eb["cbx"]**2 + eb["cby"]**2 + eb["cbz"]**2
    jdotb = (hydro["jx"] * eb["cbx"] +
             hydro["jy"] * eb["cby"] +
             hydro["jz"] * eb["cbz"])
    jpara_dote = (eb["cbx"] * eb["ex"] +
                  eb["cby"] * eb["ey"] +
                  eb["cbz"] * eb["ez"]) * jdotb / b2
    jdote = (hydro["jx"] * eb["ex"] +
             hydro["jy"] * eb["ey"] +
             hydro["jz"] * eb["ez"])
    jperp_dote = jdote - jpara_dote

    # Energization associated with pressure tensor
    ib2 = 1.0 / b2
    vexb_x = (eb["ey"] * eb["cbz"] - eb["ez"] * eb["cby"]) * ib2
    vexb_y = (eb["ez"] * eb["cbx"] - eb["ex"] * eb["cbz"]) * ib2
    vexb_z = (eb["ex"] * eb["cby"] - eb["ey"] * eb["cbx"]) * ib2
    irho = 1.0 / hydro["rho"]
    vx = hydro["jx"] * irho
    vy = hydro["jy"] * irho
    vz = hydro["jz"] * irho
    pxx = hydro["txx"] - vx * hydro["px"]
    pyy = hydro["tyy"] - vy * hydro["py"]
    pzz = hydro["tzz"] - vz * hydro["pz"]
    pyx = hydro["txy"] - vx * hydro["py"]
    pxz = hydro["tzx"] - vz * hydro["px"]
    pzy = hydro["tyz"] - vy * hydro["pz"]
    pxy = hydro["txy"] - vy * hydro["px"]
    pyz = hydro["tyz"] - vz * hydro["py"]
    pzx = hydro["tzx"] - vx * hydro["pz"]
    divpt_vexb = ((np.gradient(pxx, dx_de, axis=0) +
                   np.gradient(pxz, dz_de, axis=1)) * vexb_x +
                  (np.gradient(pyx, dx_de, axis=0) +
                   np.gradient(pyz, dz_de, axis=1)) * vexb_y +
                  (np.gradient(pzx, dx_de, axis=0) +
                   np.gradient(pzz, dz_de, axis=1)) * vexb_z)
    divpt_v = ((np.gradient(pxx, dx_de, axis=0) +
                np.gradient(pxz, dz_de, axis=1)) * vx +
               (np.gradient(pyx, dx_de, axis=0) +
                np.gradient(pyz, dz_de, axis=1)) * vy +
               (np.gradient(pzx, dx_de, axis=0) +
                np.gradient(pzz, dz_de, axis=1)) * vz)
    pvx = pxx * vx + pxy * vy + pxz * vz
    pvz = pzx * vx + pzy * vy + pzz * vz
    div_pv = (np.gradient(pvx, dx_de, axis=0) +
              np.gradient(pvz, dx_de, axis=1))
    pgradv = (pxx * np.gradient(vx, dx_de, axis=0) +
              pxy * np.gradient(vy, dx_de, axis=0) +
              pxz * np.gradient(vz, dx_de, axis=0) +
              pzx * np.gradient(vx, dz_de, axis=1) +
              pzy * np.gradient(vy, dz_de, axis=1) +
              pzz * np.gradient(vz, dz_de, axis=1))

    ppara = (pxx * eb["cbx"]**2 +
             pyy * eb["cby"]**2 +
             pzz * eb["cbz"]**2 +
             (pxy + pyx) * eb["cbx"] * eb["cby"] +
             (pxz + pzx) * eb["cbx"] * eb["cbz"] +
             (pyz + pzy) * eb["cby"] * eb["cbz"]) * ib2
    pperp = 0.5 * (pxx + pyy + pzz - ppara)
    pscalar = (ppara + 2 * pperp) / 3
    ib = np.sqrt(ib2)
    absB = 1 / ib
    bx = eb["cbx"] * ib
    by = eb["cby"] * ib
    bz = eb["cbz"] * ib
    dvx_dx = np.gradient(vx, dx_de, axis=0)
    dvy_dx = np.gradient(vy, dx_de, axis=0)
    dvz_dx = np.gradient(vz, dx_de, axis=0)
    dvx_dz = np.gradient(vx, dz_de, axis=1)
    dvy_dz = np.gradient(vy, dz_de, axis=1)
    dvz_dz = np.gradient(vz, dz_de, axis=1)
    divv = dvx_dx + dvz_dz
    divv3 = divv / 3
    shear = ((dvx_dx - divv3) * bx**2 +
             (dvz_dz - divv3) * bz**2 +
             (-divv3) * by**2 +
             dvy_dx * bx * by + dvy_dz * by * bz +
             (dvx_dz + dvz_dx) * bx * bz)
    pdivv = -pscalar * divv
    pshear = (pperp - ppara) * shear

    dmax = 1E-4
    dmin = -dmax
    fig = plt.figure(figsize=[7, 11])
    rect = [0.12, 0.84, 0.72, 0.11]
    hgap, vgap = 0.02, 0.02
    nvar = 7
    axs = []
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    for ivar in range(nvar):
        ax = fig.add_axes(rect)
        axs.append(ax)
        rect[1] -= rect[3] + vgap
        ax.set_prop_cycle('color', COLORS)
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')
        if ivar == nvar - 1:
            ax.set_xlabel(r'$x/d_e$', fontsize=16)
        if ivar < nvar - 1:
            ax.set_ylabel(r'$z/d_e$', fontsize=16)
        else:
            ax.set_ylabel('Accumulation', fontsize=16)
        ax.tick_params(labelsize=12)
        ax.set_xlim([xmin, xmax])
        if ivar < nvar - 1:
            ax.set_ylim([zmin, zmax])

    cmap = plt.cm.seismic
    im1 = axs[0].imshow(jdote.T,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=dmin, vmax=dmax,
                        cmap=cmap, aspect='auto',
                        origin='lower', interpolation='bicubic')
    im2 = axs[1].imshow(divpt_v.T,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=dmin, vmax=dmax,
                        cmap=cmap, aspect='auto',
                        origin='lower', interpolation='bicubic')
    im3 = axs[2].imshow(pdivv.T,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=dmin, vmax=dmax,
                        cmap=cmap, aspect='auto',
                        origin='lower', interpolation='bicubic')
    im4 = axs[3].imshow(pshear.T,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=dmin, vmax=dmax,
                        cmap=cmap, aspect='auto',
                        origin='lower', interpolation='bicubic')
    im5 = axs[4].imshow(div_pv.T,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=dmin, vmax=dmax,
                        cmap=cmap, aspect='auto',
                        origin='lower', interpolation='bicubic')
    im6 = axs[5].imshow(-pgradv.T,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=dmin, vmax=dmax,
                        cmap=cmap, aspect='auto',
                        origin='lower', interpolation='bicubic')
    jdote_zsum = np.sum(jdote, axis=1)
    divpt_v_zsum = np.sum(divpt_v, axis=1)
    pdivv_zsum = np.sum(pdivv, axis=1)
    pshear_zsum = np.sum(pshear, axis=1)
    div_pv_zsum = np.sum(div_pv, axis=1)
    pgradv_zsum = np.sum(pgradv, axis=1)

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
    ix_xp = int(x0 / dx_de) - xs

    jdote_xsum = np.zeros(nxs)
    divpt_v_xsum = np.zeros(nxs)
    pdivv_xsum = np.zeros(nxs)
    pshear_xsum = np.zeros(nxs)
    div_pv_xsum = np.zeros(nxs)
    pgradv_xsum = np.zeros(nxs)
    jdote_xsum[ix_xp::-1] = np.cumsum(jdote_zsum[ix_xp::-1])
    jdote_xsum[ix_xp:] = np.cumsum(jdote_zsum[ix_xp:])
    divpt_v_xsum[ix_xp::-1] = np.cumsum(divpt_v_zsum[ix_xp::-1])
    divpt_v_xsum[ix_xp:] = np.cumsum(divpt_v_zsum[ix_xp:])
    pdivv_xsum[ix_xp::-1] = np.cumsum(pdivv_zsum[ix_xp::-1])
    pdivv_xsum[ix_xp:] = np.cumsum(pdivv_zsum[ix_xp:])
    pshear_xsum[ix_xp::-1] = np.cumsum(pshear_zsum[ix_xp::-1])
    pshear_xsum[ix_xp:] = np.cumsum(pshear_zsum[ix_xp:])
    div_pv_xsum[ix_xp::-1] = np.cumsum(div_pv_zsum[ix_xp::-1])
    div_pv_xsum[ix_xp:] = np.cumsum(div_pv_zsum[ix_xp:])
    pgradv_xsum[ix_xp::-1] = np.cumsum(pgradv_zsum[ix_xp::-1])
    pgradv_xsum[ix_xp:] = np.cumsum(pgradv_zsum[ix_xp:])
    xgrid = np.arange(xs, xe) * dx_de
    axs[nvar-1].plot(xgrid, jdote_xsum)
    axs[nvar-1].plot(xgrid, divpt_v_xsum)
    axs[nvar-1].plot(xgrid, pdivv_xsum)
    axs[nvar-1].plot(xgrid, pshear_xsum)
    axs[nvar-1].plot(xgrid, div_pv_xsum)
    axs[nvar-1].plot(xgrid, -pgradv_xsum)
    axs[nvar-1].plot([xmin, xmax], [0, 0], color='k', linestyle='--')

    rect_cbar = np.copy(rect)
    rect_cbar[0] += rect[2] + 0.01
    rect_cbar[2] = 0.02
    rect_cbar[1] += rect[3] * 3.5 + vgap * 3
    rect_cbar[3] = rect[3] * 3 + vgap * 3
    cbar_ax = fig.add_axes(rect_cbar)
    cbar = fig.colorbar(im1, cax=cbar_ax, extend='both')
    cbar_ax.tick_params(bottom=False, top=False, left=False, right=True)
    cbar_ax.tick_params(axis='y', which='major', direction='out')
    cbar_ax.tick_params(axis='y', which='minor', direction='in', right=False)
    cbar.ax.tick_params(labelsize=12)

    xmin_di = xmin / smime
    xmax_di = xmax / smime
    zmin_di = zmin / smime
    zmax_di = zmax / smime
    kwargs = {"current_time": tframe,
              "xl": xmin_di, "xr": xmax_di,
              "zb": zmin_di, "zt": zmax_di}
    fname = pic_run_dir + "data/Ay.gda"
    x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
    xde = x * smime
    zde = z * smime
    texts = [r'$\boldsymbol{j}_{' + species + r'}\cdot\boldsymbol{E}$',
             r'$(\nabla\cdot\boldsymbol{P})\cdot\boldsymbol{v}$',
             r'$-p\nabla\cdot\boldsymbol{v}$',
             r'$-(p_\parallel-p_\perp)b_ib_j\sigma_{ij}$',
             r'$\nabla\cdot(\mathcal{P}\cdot\boldsymbol{v})$',
             r'$-\mathcal{P}:\nabla\boldsymbol{v}$']
    for iax, ax in enumerate(axs[:-1]):
        ax.contour(xde, zde, Ay, colors='k', linewidths=0.5,
                   levels=np.linspace(np.min(Ay), np.max(Ay), 10))
        ax.text(0.03, 0.8, texts[iax], color=COLORS[iax], fontsize=16,
                bbox=dict(facecolor='w', alpha=1.0, boxstyle='round',
                          edgecolor='grey', pad=0.2),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)

    twpe = math.ceil(tindex * pic_info.dtwpe / 0.1) * 0.1
    text1 = r'$t\omega_{pe}=' + ("{%0.0f}" % twpe) + '$'
    fig.suptitle(text1, fontsize=20)
    img_dir = '../img/rate_problem/comp_ene/' + pic_run + '/'
    mkdir_p(img_dir)
    fname = img_dir + "comp_ene_" + species + "_" + str(tframe) + ".jpg"
    fig.savefig(fname, dpi=200)

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_density_xline(plot_config, show_plot=True):
    """Plot density near X-line
    """
    pic_run = plot_config["pic_run"]
    tframe = plot_config["tframe"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_run_dir = pic_info.run_dir
    fields_interval = pic_info.fields_interval
    tindex = fields_interval * tframe
    smime = math.sqrt(pic_info.mime)
    lx_de = pic_info.lx_di * smime
    lz_de = pic_info.lz_di * smime
    nx = pic_info.nx
    nz = pic_info.nz
    xs = nx//2 - nx//8
    xe = nx//2 + nx//8
    zs = nz//2 - nz//32
    ze = nz//2 + nz//32
    dx_de = lx_de / nx
    dz_de = lz_de / nz
    nxs = xe - xs
    nzs = ze - zs
    xmin = xs * dz_de
    xmax = xe * dz_de
    zmin = zs * dz_de - lz_de * 0.5
    zmax = ze * dz_de - lz_de * 0.5
    zmin_pic = -lz_de * 0.5
    zmin = zs * dz_de - lz_de * 0.5
    zmax = ze * dz_de - lz_de * 0.5

    vpic_info = get_vpic_info(pic_run_dir)
    n0 = vpic_info["n0"]
    b0 = vpic_info["b0"]
    Te = vpic_info["Te"]
    Tbe_Te = vpic_info["Tbe/Te"]
    nb_n0 = vpic_info["nb/n0"]
    nb = n0 * nb_n0
    p0 = nb * Te * Tbe_Te
    Tb = Te * Tbe_Te
    beta0 = Tbe_Te

    plot_config["pic_run"] = pic_run
    plot_config["pic_run_dir"] = pic_run_dir
    start = (xs, 0, zs)
    count = (nxs, 1, nzs)

    rho = {}
    for species in ['e', 'i']:
        sname = "electron" if species == 'e' else "ion"
        fname = (pic_run_dir + "hydro_hdf5/T." + str(tindex) +
                 "/hydro_electron_" + str(tindex) + ".h5")
        with h5py.File(fname, 'r') as fh:
            group = fh["Timestep_" + str(tindex)]
            dset = group["rho"]
            var_name = "n" + species
            rho[var_name] = gaussian_filter(np.abs(dset[xs:xe, 0, zs:ze]),
                                            sigma=3)

    # Outflow velocity
    rho_vel = {}
    for species in ["e", "i"]:
        sname = "electron" if species == 'e' else "ion"
        fname = (pic_run_dir + "hydro_hdf5/T." + str(tindex) +
                 "/hydro_" + sname + "_" + str(tindex) + ".h5")
        hydro = {}
        with h5py.File(fname, 'r') as fh:
            group = fh["Timestep_" + str(tindex)]
            for var in ["rho", "jx"]:
                dset = group[var]
                hydro[var]= dset[xs:xe, 0, zs:ze]

        irho = 1.0 / hydro["rho"]
        vx = hydro["jx"] * irho
        var = "v" + species
        rho_vel[var+"x"] = np.squeeze(vx)
        rho_vel["n"+species] = np.squeeze(np.abs(hydro["rho"]))

    mime = pic_info.mime
    irho = 1.0 / (rho_vel["ne"] + rho_vel["ni"] * mime)
    vsx = (rho_vel["ne"] * rho_vel["vex"] +
           rho_vel["ni"] * rho_vel["vix"] * mime) * irho

    dtwpe = pic_info.dtwpe
    dtwce = pic_info.dtwce
    vpic_info = get_vpic_info(pic_run_dir)
    nb_n0 = vpic_info["nb/n0"]
    va = dtwce * math.sqrt(1.0 / mime) / dtwpe / math.sqrt(nb_n0)

    # Find X-point
    fdir = '../data/rate_problem/exhaust_boundary/' + pic_run + '/'
    fname = fdir + 'xz_top_' + str(tframe) + '.dat'
    xz = np.fromfile(fname).reshape([2, -1])
    xlist_top = xz[0, :]
    zlist_top = xz[1, :] + zmin_pic
    fname = fdir + 'xz_bot_' + str(tframe) + '.dat'
    xz = np.fromfile(fname).reshape([2, -1])
    xlist_bot = xz[0, :]
    zlist_bot = xz[1, :] + zmin_pic
    x0 = 0.5 * (xlist_bot[np.argmax(zlist_bot)] +
                xlist_top[np.argmin(zlist_top)])
    ix_xp = int(x0 / dx_de) - xs

    dmin, dmax = 0.5, 3
    fig = plt.figure(figsize=[7, 7])
    rect = [0.12, 0.76, 0.78, 0.18]
    hgap, vgap = 0.02, 0.04
    nvar = 3
    axs = []
    for ivar in range(nvar):
        if ivar == nvar - 1:
            rect1 = np.copy(rect)
            rect1[3] = 0.42
            rect1[1] -= rect1[3] - rect[3]
            ax = fig.add_axes(rect1)
        else:
            ax = fig.add_axes(rect)
        axs.append(ax)
        rect[1] -= rect[3] + vgap
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')

    cmap = plt.cm.jet
    im1 = axs[0].imshow(rho["ne"].T,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=dmin, vmax=dmax,
                        cmap=cmap, aspect='auto',
                        origin='lower', interpolation='bicubic')
    im2 = axs[1].imshow(rho["ni"].T,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=dmin, vmax=dmax,
                        cmap=cmap, aspect='auto',
                        origin='lower', interpolation='bicubic')
    rect_cbar = np.copy(rect)
    rect_cbar[0] += rect[2] + 0.01
    rect_cbar[2] = 0.02
    rect_cbar[1] += (rect[3] + vgap) * 2
    rect_cbar[3] = rect[3]*2 + vgap
    cbar_ax = fig.add_axes(rect_cbar)
    cbar = fig.colorbar(im1, cax=cbar_ax, extend='both')
    cbar_ax.tick_params(bottom=False, top=False, left=False, right=True)
    cbar_ax.tick_params(axis='y', which='major', direction='out')
    cbar_ax.tick_params(axis='y', which='minor', direction='in', right=False)
    cbar.ax.tick_params(labelsize=12)
    xde = np.linspace(xmin, xmax, nxs)
    betav = np.sqrt(2/beta0) * np.abs(vsx[:, nzs//2]) / va
    dn_fermi = 1 + erf(betav)
    axs[2].plot(xde, rho["ne"][:, nzs//2]/rho["ne"][ix_xp, nzs//2])
    axs[2].plot(xde, rho["ni"][:, nzs//2]/rho["ni"][ix_xp, nzs//2])
    axs[2].plot(xde, dn_fermi)
    axs[2].grid(True)

    xmin_di = xmin / smime
    xmax_di = xmax / smime
    zmin_di = zmin / smime
    zmax_di = zmax / smime
    kwargs = {"current_time": tframe,
              "xl": xmin_di, "xr": xmax_di,
              "zb": zmin_di, "zt": zmax_di}
    fname = pic_run_dir + "data/Ay.gda"
    x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
    xde = x * smime
    zde = z * smime
    texts = [r"$n_e$", r"$n_i$", "Cut"]
    for iax, ax in enumerate(axs):
        if iax < nvar - 1:
            ax.contour(xde, zde, Ay, colors='k', linewidths=0.5,
                       levels=np.linspace(np.min(Ay), np.max(Ay), 10))
            ax.text(0.02, 0.85, texts[iax], color='k', fontsize=16,
                    bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                    horizontalalignment='left', verticalalignment='center',
                    transform=ax.transAxes)
        if iax == nvar - 1:
            ax.set_xlabel(r'$x/d_e$', fontsize=16)
        if iax < nvar - 1:
            ax.set_ylabel(r'$z/d_e$', fontsize=16)
        else:
            ax.set_ylabel('Cut', fontsize=16)
        ax.tick_params(labelsize=12)
        ax.set_xlim([xmin, xmax])
        if iax < nvar - 1:
            ax.set_ylim([zmin, zmax])
        else:
            ax.set_ylim([dmin, dmax])

    twpe = math.ceil(tindex * pic_info.dtwpe / 0.1) * 0.1
    text1 = r'$t\omega_{pe}=' + ("{%0.0f}" % twpe) + '$'
    fig.suptitle(text1, fontsize=20)
    img_dir = '../img/rate_problem/rhox/' + pic_run + '/'
    mkdir_p(img_dir)
    fname = img_dir + "rhox_" + str(tframe) + ".jpg"
    fig.savefig(fname, dpi=400)

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_density_fieldline(plot_config, show_plot=True):
    """Plot density along field line
    """
    pic_run = plot_config["pic_run"]
    tframe = plot_config["tframe"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_run_dir = pic_info.run_dir
    fields_interval = pic_info.fields_interval
    tindex = fields_interval * tframe
    smime = math.sqrt(pic_info.mime)
    lx_de = pic_info.lx_di * smime
    lz_de = pic_info.lz_di * smime
    nx = pic_info.nx
    nz = pic_info.nz
    xs = nx//2 - nx//4
    xe = nx//2 + nx//4
    zs = nz//2 - nz//16
    ze = nz//2 + nz//16
    dx_de = lx_de / nx
    dz_de = lz_de / nz
    nxs = xe - xs
    nzs = ze - zs
    xmin = xs * dz_de
    xmax = xe * dz_de
    zmin_pic = -lz_de * 0.5
    zmin = zs * dz_de - lz_de * 0.5
    zmax = ze * dz_de - lz_de * 0.5

    vpic_info = get_vpic_info(pic_run_dir)
    n0 = vpic_info["n0"]
    b0 = vpic_info["b0"]
    Te = vpic_info["Te"]
    Tbe_Te = vpic_info["Tbe/Te"]
    nb_n0 = vpic_info["nb/n0"]
    nb = n0 * nb_n0
    p0 = nb * Te * Tbe_Te
    Tb = Te * Tbe_Te

    plot_config["pic_run"] = pic_run
    plot_config["pic_run_dir"] = pic_run_dir
    start = (xs, 0, zs)
    count = (nxs, 1, nzs)

    rho = {}
    for species in ['e', 'i']:
        sname = "electron" if species == 'e' else "ion"
        fname = (pic_run_dir + "hydro_hdf5/T." + str(tindex) +
                 "/hydro_electron_" + str(tindex) + ".h5")
        with h5py.File(fname, 'r') as fh:
            group = fh["Timestep_" + str(tindex)]
            dset = group["rho"]
            var_name = "n" + species
            rho[var_name] = gaussian_filter(np.abs(dset[xs:xe, 0, zs:ze]),
                                            sigma=3)

    dmin, dmax = 0.5, 3
    fig = plt.figure(figsize=[7, 7])
    rect = [0.12, 0.76, 0.78, 0.18]
    hgap, vgap = 0.02, 0.04
    nvar = 2
    axs = []
    for ivar in range(nvar):
        if ivar == nvar - 1:
            rect1 = np.copy(rect)
            rect1[3] = 0.6
            rect1[1] -= rect1[3] - rect[3]
            ax = fig.add_axes(rect1)
        else:
            ax = fig.add_axes(rect)
        axs.append(ax)
        rect[1] -= rect[3] + vgap
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')
        ax.tick_params(labelsize=12)

    cmap = plt.cm.jet
    im1 = axs[0].imshow(rho["ni"].T,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=dmin, vmax=dmax,
                        cmap=cmap, aspect='auto',
                        origin='lower', interpolation='bicubic')
    rect_cbar = np.copy(rect)
    rect_cbar[0] += rect[2] + 0.01
    rect_cbar[2] = 0.02
    rect_cbar[1] += (rect[3] + vgap) * 2
    cbar_ax = fig.add_axes(rect_cbar)
    cbar = fig.colorbar(im1, cax=cbar_ax, extend='both')
    cbar_ax.tick_params(bottom=False, top=False, left=False, right=True)
    cbar_ax.tick_params(axis='y', which='major', direction='out')
    cbar_ax.tick_params(axis='y', which='minor', direction='in', right=False)
    cbar.ax.tick_params(labelsize=12)

    # Find X-point
    fdir = '../data/rate_problem/exhaust_boundary/' + pic_run + '/'
    fname = fdir + 'xz_top_' + str(tframe) + '.dat'
    xz = np.fromfile(fname).reshape([2, -1])
    xlist_top = xz[0, :]
    zlist_top = xz[1, :] + zmin_pic
    fname = fdir + 'xz_bot_' + str(tframe) + '.dat'
    xz = np.fromfile(fname).reshape([2, -1])
    xlist_bot = xz[0, :]
    zlist_bot = xz[1, :] + zmin_pic
    x0 = 0.5 * (xlist_bot[np.argmax(zlist_bot)] +
                xlist_top[np.argmin(zlist_top)])
    ix_xpoint = int(x0 / dx_de)
    axs[0].plot(xlist_top, zlist_top)
    axs[0].plot(xlist_bot, zlist_bot)

    xmin_di = xmin / smime
    xmax_di = xmax / smime
    zmin_di = zmin / smime
    zmax_di = zmax / smime
    kwargs = {"current_time": tframe,
              "xl": xmin_di, "xr": xmax_di,
              "zb": zmin_di, "zt": zmax_di}
    fname = pic_run_dir + "data/Ay.gda"
    x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
    xde = x * smime
    zde = z * smime
    texts = [r"$n_i$", "Along field line"]

    cs = axs[0].contour(xde, zde, Ay, colors='w', linewidths=0.5, levels=[364])
    axs[0].text(0.02, 0.85, r"$n_i$", color='k', fontsize=16,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)
    axs[0].set_ylabel(r'$z/d_e$', fontsize=16)
    axs[0].set_ylabel('Cut', fontsize=16)
    axs[0].set_xlim([xmin, xmax])
    axs[0].set_ylim([zmin, zmax])

    xdata = []
    zdata = []
    for cl in cs.collections:
        sz = len(cl.get_paths())
        for p in cl.get_paths():
            v = p.vertices
            xdata.append(v[:, 0])
            zdata.append(v[:, 1])
    cond = np.logical_and(xdata[0] > xmin, xdata[0] < xmax)
    x = xdata[0][cond] - xmin
    z = zdata[0][cond] - zmin
    x1 = np.floor(x / dx_de).astype(np.int)
    z1 = np.floor(z / dz_de).astype(np.int)
    x2 = x1 + 1
    z2 = z1 + 1
    x2[x2 > nxs - 1] = nxs - 1
    z2[z2 > nzs - 1] = nzs - 1
    dx = x / dx_de - x1
    dz = z / dz_de - z1
    v1 = (1.0 - dx) * (1.0 - dz)
    v2 = dx * (1.0 - dz)
    v3 = (1.0 - dx) * dz
    v4 = dx * dz
    nline = (rho["ni"][x1, z1] * v1 +
             rho["ni"][x2, z1] * v2 +
             rho["ni"][x1, z2] * v3 +
             rho["ni"][x2, z2] * v4)
    axs[1].plot(x + xmin, nline)
    axs[1].grid(True)
    axs[1].set_xlabel(r'$x/d_e$', fontsize=16)
    axs[1].set_ylabel(r'$z/d_e$', fontsize=16)
    axs[1].set_ylabel('Along field line', fontsize=16)
    axs[1].tick_params(labelsize=12)
    axs[1].set_xlim([xmin, xmax])

    # twpe = math.ceil(tindex * pic_info.dtwpe / 0.1) * 0.1
    # text1 = r'$t\omega_{pe}=' + ("{%0.0f}" % twpe) + '$'
    # fig.suptitle(text1, fontsize=20)
    # img_dir = '../img/rate_problem/rhox/' + pic_run + '/'
    # mkdir_p(img_dir)
    # fname = img_dir + "rhox_" + str(tframe) + ".jpg"
    # fig.savefig(fname, dpi=400)

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_alfven_speed(plot_config, show_plot=True):
    """Plot the Alfven speed near X-line
    """
    pic_run = plot_config["pic_run"]
    tframe = plot_config["tframe"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_run_dir = pic_info.run_dir
    fields_interval = pic_info.fields_interval
    tindex = fields_interval * tframe
    smime = math.sqrt(pic_info.mime)
    lx_de = pic_info.lx_di * smime
    lz_de = pic_info.lz_di * smime
    nx = pic_info.nx
    nz = pic_info.nz
    xs = nx//2 - nx//4
    xe = nx//2 + nx//4
    zs = nz//2 - nz//16
    ze = nz//2 + nz//16
    dx_de = lx_de / nx
    dz_de = lz_de / nz
    nxs = xe - xs
    nzs = ze - zs
    xmin = xs * dz_de
    xmax = xe * dz_de
    zmin = zs * dz_de - lz_de * 0.5
    zmax = ze * dz_de - lz_de * 0.5

    vpic_info = get_vpic_info(pic_run_dir)
    n0 = vpic_info["n0"]
    b0 = vpic_info["b0"]
    Te = vpic_info["Te"]
    Tbe_Te = vpic_info["Tbe/Te"]
    nb_n0 = vpic_info["nb/n0"]
    nb = n0 * nb_n0
    p0 = nb * Te * Tbe_Te
    Tb = Te * Tbe_Te
    dtwpe = pic_info.dtwpe
    dtwce = pic_info.dtwce
    va0 = dtwce * math.sqrt(1.0 / pic_info.mime) / dtwpe / math.sqrt(nb_n0)

    plot_config["pic_run"] = pic_run
    plot_config["pic_run_dir"] = pic_run_dir
    start = (xs, 0, zs)
    count = (nxs, 1, nzs)

    fname = (pic_run_dir + "field_hdf5/T." + str(tindex) +
             "/fields_" + str(tindex) + ".h5")
    bvec = {}
    with h5py.File(fname, 'r') as fh:
        group = fh["Timestep_" + str(tindex)]
        for var in ["cbx", "cby", "cbz"]:
            dset = group[var]
            bvec[var]= np.abs(dset[xs:xe, 0, zs:ze])
    absB = np.sqrt(bvec["cbx"]**2 + bvec["cby"]**2 + bvec["cbz"]**2)

    fname = (pic_run_dir + "hydro_hdf5/T." + str(tindex) +
             "/hydro_ion_" + str(tindex) + ".h5")
    with h5py.File(fname, 'r') as fh:
        group = fh["Timestep_" + str(tindex)]
        dset = group["rho"]
        ni = np.abs(dset[xs:xe, 0, zs:ze])

    va = absB / np.sqrt(ni * pic_info.mime)

    dmin, dmax = 0.5, 3
    fig = plt.figure(figsize=[7, 6])
    rect = [0.12, 0.7, 0.78, 0.25]
    hgap, vgap = 0.02, 0.05
    nvar = 3
    axs = []
    cbar_axs = []
    for ivar in range(nvar):
        ax = fig.add_axes(rect)
        axs.append(ax)
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')
        rect_cbar = np.copy(rect)
        rect_cbar[0] += rect[2] + 0.01
        rect_cbar[2] = 0.015
        cbar_axs.append(fig.add_axes(rect_cbar))
        rect[1] -= rect[3] + vgap

    cmap = plt.cm.jet
    im1 = axs[0].imshow(absB.T / b0,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=0, vmax=2,
                        cmap=plt.cm.seismic, aspect='auto',
                        origin='lower', interpolation='bicubic')
    im2 = axs[1].imshow(ni.T / nb,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=0.5, vmax=3,
                        cmap=plt.cm.jet, aspect='auto',
                        origin='lower', interpolation='bicubic')
    im3 = axs[2].imshow(va.T / va0,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=0, vmax=1,
                        cmap=plt.cm.coolwarm, aspect='auto',
                        origin='lower', interpolation='bicubic')

    for i, im in enumerate([im1, im2, im3]):
        cbar = fig.colorbar(im, cax=cbar_axs[i], extend='both')
        cbar_axs[i].tick_params(bottom=False, top=False, left=False, right=True)
        cbar_axs[i].tick_params(axis='y', which='major', direction='out')
        cbar_axs[i].tick_params(axis='y', which='minor', direction='in', right=False)
        cbar.ax.tick_params(labelsize=12)

    xmin_di = xmin / smime
    xmax_di = xmax / smime
    zmin_di = zmin / smime
    zmax_di = zmax / smime
    kwargs = {"current_time": tframe,
              "xl": xmin_di, "xr": xmax_di,
              "zb": zmin_di, "zt": zmax_di}
    fname = pic_run_dir + "data/Ay.gda"
    x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
    xde = x * smime
    zde = z * smime
    texts = [r"$B/B_0$", r"$n_i/n_0$", r"$v_A/v_{A0}$"]
    for iax, ax in enumerate(axs):
        ax.contour(xde, zde, Ay, colors='k', linewidths=0.5,
                   levels=np.linspace(np.min(Ay), np.max(Ay), 10))
        ax.text(0.02, 0.85, texts[iax], color='k', fontsize=16,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)
        if iax == nvar - 1:
            ax.set_xlabel(r'$x/d_e$', fontsize=16)
        ax.set_ylabel(r'$z/d_e$', fontsize=16)
        ax.tick_params(labelsize=12)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([zmin, zmax])

    twpe = math.ceil(tindex * pic_info.dtwpe / 0.1) * 0.1
    text1 = r'$t\omega_{pe}=' + ("{%0.0f}" % twpe) + '$'
    fig.suptitle(text1, fontsize=20)
    img_dir = '../img/rate_problem/va/' + pic_run + '/'
    mkdir_p(img_dir)
    fname = img_dir + "va_" + str(tframe) + ".jpg"
    fig.savefig(fname, dpi=200)

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_absj_vout(plot_config, show_plot=True):
    """Plot current density and outflow velocity
    """
    tframe = plot_config["tframe"]
    pic_runs = ["mime400_Tb_T0_025", "mime400_Tb_T0_10_weak"]
    fig = plt.figure(figsize=[7, 8])
    rect0 = [0.12, 0.74, 0.75, 0.2]
    hgap, vgap = 0.02, 0.02
    axs = []
    cbar_axs = []
    nvar = 4
    for i in range(nvar):
        rect = np.copy(rect0)
        rect[1] = rect0[1] - (rect0[3] + vgap) * i
        ax = fig.add_axes(rect)
        axs.append(ax)
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')
        ax.set_ylabel(r'$z/d_i$', fontsize=16)
        if i == nvar - 1:
            ax.set_xlabel(r'$x/d_i$', fontsize=16)
        else:
            ax.tick_params(axis='x', labelbottom=False)
        ax.tick_params(labelsize=12)

        if i % 2 == 0:
            rect_cbar = np.copy(rect)
            rect_cbar[0] += rect[2] + 0.05
            rect_cbar[2] = 0.015
            rect_cbar[1] -= rect[3]*0.5 + vgap
            rect_cbar[3] = rect[3] + vgap
            cbar_ax = fig.add_axes(rect_cbar)
            cbar_axs.append(cbar_ax)

    for irun, pic_run in enumerate(pic_runs):
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        pic_run_dir = pic_info.run_dir
        fields_interval = pic_info.fields_interval
        tindex = fields_interval * tframe
        mime = pic_info.mime
        smime = math.sqrt(mime)
        lx_de = pic_info.lx_di * smime
        lz_de = pic_info.lz_di * smime
        dx_de = lx_de / pic_info.nx
        dz_de = lz_de / pic_info.nz
        nx = pic_info.nx
        nz = pic_info.nz
        xs = nx//2 - nx//4
        xe = nx//2 + nx//4
        zs = nz//2 - nz//16
        ze = nz//2 + nz//16
        nxs = xe - xs
        nzs = ze - zs
        xmin = xs * dz_de
        xmax = xe * dz_de
        zmin_pic = -lz_de * 0.5
        zmin = zs * dz_de + zmin_pic
        zmax = ze * dz_de + zmin_pic

        fname = (pic_run_dir + "hydro_hdf5/T." + str(tindex) +
                 "/hydro_electron_" + str(tindex) + ".h5")
        je = {}
        with h5py.File(fname, 'r') as fh:
            group = fh["Timestep_" + str(tindex)]
            for var in ["jx", "jy", "jz"]:
                dset = group[var]
                je[var]= dset[xs:xe, 0, zs:ze]

        fname = (pic_run_dir + "hydro_hdf5/T." + str(tindex) +
                 "/hydro_ion_" + str(tindex) + ".h5")
        ji = {}
        with h5py.File(fname, 'r') as fh:
            group = fh["Timestep_" + str(tindex)]
            for var in ["jx", "jy", "jz"]:
                dset = group[var]
                ji[var]= dset[xs:xe, 0, zs:ze]
        absj = np.squeeze(np.sqrt((je["jx"] + ji["jx"])**2 +
                                  (je["jy"] + ji["jy"])**2 +
                                  (je["jz"] + ji["jz"])**2))

        # Magnetic field lines
        xmin_di = xmin / smime
        xmax_di = xmax / smime
        zmin_di = zmin / smime
        zmax_di = zmax / smime
        kwargs = {"current_time": tframe,
                  "xl": xmin_di, "xr": xmax_di,
                  "zb": zmin_di, "zt": zmax_di}
        fname = pic_run_dir + "data/Ay.gda"
        x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
        xde = x * smime
        zde = z * smime
        axs[irun].contour(x, z, Ay, colors='w', linewidths=0.5,
                          levels=np.linspace(np.min(Ay), np.max(Ay), 8))

        im1 = axs[irun].imshow(absj.T,
                               extent=[xmin_di, xmax_di, zmin_di, zmax_di],
                               vmin=0, vmax=0.08,
                               cmap=plt.cm.viridis, aspect='auto',
                               origin='lower', interpolation='bicubic')

        fdir = '../data/rate_problem/exhaust_boundary/' + pic_run + '/'
        fname = fdir + 'xz_top_' + str(tframe) + '.dat'
        xz = np.fromfile(fname).reshape([2, -1])
        xlist_top = xz[0, :]
        zlist_top = xz[1, :] + zmin_pic
        fname = fdir + 'xz_bot_' + str(tframe) + '.dat'
        xz = np.fromfile(fname).reshape([2, -1])
        xlist_bot = xz[0, :]
        zlist_bot = xz[1, :] + zmin_pic
        x0 = 0.5 * (xlist_bot[np.argmax(zlist_bot)] +
                    xlist_top[np.argmin(zlist_top)])
        ix_xpoint = int(x0 / dx_de)
        z0 = zlist_top[np.argmin(zlist_top)]
        iz_top = np.argmin(zlist_top)
        iz_bot = np.argmax(zlist_bot)
        xtop = xlist_top[iz_top]
        ztop = zlist_top[iz_top]

        if pic_run == "mime400_Tb_T0_025":
            shift = 200
            theta_top = np.arctan((zlist_top - ztop) / (xlist_top - xtop + 1E-10)) * 180 / math.pi
            theta_max1 = theta_top[iz_top+shift:].max()
            theta_min1 = -theta_top[:iz_top-shift].min()
            xbot = xlist_bot[iz_bot]
            zbot = zlist_bot[iz_bot]
            theta_bot = np.arctan((zlist_bot - zbot) / (xlist_bot - xbot + 1E-10)) * 180 / math.pi
            theta_max2 = theta_bot[iz_bot+shift:].max()
            theta_min2 = -theta_bot[:iz_bot-shift].min()
        else:
            shift = 2000
            f = np.polyfit(xlist_top[iz_top:iz_top+shift], zlist_top[iz_top:iz_top+shift], 1)
            theta_max1 = math.atan(f[0]) * 180 / math.pi
            f = np.polyfit(xlist_top[iz_top-shift:iz_top], zlist_top[iz_top-shift:iz_top], 1)
            theta_min1 = -math.atan(f[0]) * 180 / math.pi
            f = np.polyfit(xlist_bot[iz_bot:iz_bot+shift], zlist_bot[iz_bot:iz_bot+shift], 1)
            theta_max2 = math.atan(f[0]) * 180 / math.pi
            f = np.polyfit(xlist_bot[iz_bot-shift:iz_bot], zlist_bot[iz_bot-shift:iz_bot], 1)
            theta_min2 = -math.atan(f[0]) * 180 / math.pi

        theta_min = 0.5*(theta_min1 + theta_min2)
        theta_max = 0.5*(theta_max1 + theta_max2)
        if pic_run == "mime400_Tb_T0_025":
            angle0 = (theta_max1 + theta_min2) * 0.5
        else:
            angle0 = (theta_min + theta_max) * 0.5

        axs[irun].plot(xlist_top/smime, zlist_top/smime, linewidth=1, color=COLORS[0])
        axs[irun].plot(xlist_bot/smime, zlist_bot/smime, linewidth=1, color=COLORS[0])

        length = 200.0
        x1 = x0 + 0.5 * length * math.cos(angle0*math.pi/180)
        x2 = x0 - 0.5 * length * math.cos(angle0*math.pi/180)
        z1 = z0 + 0.5 * length * math.sin(angle0*math.pi/180)
        z2 = z0 - 0.5 * length * math.sin(angle0*math.pi/180)
        x1 /= smime
        x2 /= smime
        z1 /= smime
        z2 /= smime
        axs[irun].plot([x1, x2], [z1, z2], color='k', linestyle='--', linewidth=1)
        axs[irun].plot([x1, x2], [-z1, -z2], color='k', linestyle='--', linewidth=1)
        # axs[irun].plot([x1, x2], [0, 0], color='k', linestyle='--', linewidth=1)
        angs = np.linspace(-angle0, angle0, 50) * math.pi / 180
        l0 = 75
        xarc = l0 * np.cos(angs) + x0
        yarc = l0 * np.sin(angs) + z0
        xarc /= smime
        yarc /= smime
        axs[irun].plot(xarc, yarc, linewidth=1, color='k')
        text1 = r"$" + ("{%0.1f}" % (2*angle0)) +  "^\circ$"
        axs[irun].text(0.7, 0.5, text1, color='k', fontsize=16,
                       bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                       horizontalalignment='right', verticalalignment='center',
                       transform=axs[irun].transAxes)

        if irun == 0:
            cbar = fig.colorbar(im1, cax=cbar_axs[0], extend='max')
            cbar_axs[0].tick_params(bottom=False, top=False, left=False, right=True)
            cbar_axs[0].tick_params(axis='y', which='major', direction='out')
            cbar_axs[0].tick_params(axis='y', which='minor', direction='in', right=False)
            cbar_axs[0].set_title(r'$|\boldsymbol{j}|/j_0$', fontsize=16)
            cbar.ax.tick_params(labelsize=12)

        # Outflow velocity
        rho_vel = {}
        for species in ["e", "i"]:
            sname = "electron" if species == 'e' else "ion"
            fname = (pic_run_dir + "hydro_hdf5/T." + str(tindex) +
                     "/hydro_" + sname + "_" + str(tindex) + ".h5")
            hydro = {}
            with h5py.File(fname, 'r') as fh:
                group = fh["Timestep_" + str(tindex)]
                for var in ["rho", "jx"]:
                    dset = group[var]
                    hydro[var]= dset[xs:xe, 0, zs:ze]

            irho = 1.0 / hydro["rho"]
            vx = hydro["jx"] * irho
            var = "v" + species
            rho_vel[var+"x"] = np.squeeze(vx)
            rho_vel["n"+species] = np.squeeze(np.abs(hydro["rho"]))

        irho = 1.0 / (rho_vel["ne"] + rho_vel["ni"] * mime)
        vsx = (rho_vel["ne"] * rho_vel["vex"] +
               rho_vel["ni"] * rho_vel["vix"] * mime) * irho

        dtwpe = pic_info.dtwpe
        dtwce = pic_info.dtwce
        vpic_info = get_vpic_info(pic_run_dir)
        nb_n0 = vpic_info["nb/n0"]
        va = dtwce * math.sqrt(1.0 / mime) / dtwpe / math.sqrt(nb_n0)

        im1 = axs[irun+2].imshow(vsx.T/va, cmap=plt.cm.seismic,
                                 extent=[xmin_di, xmax_di, zmin_di, zmax_di],
                                 vmin=-0.5, vmax=0.5, aspect='auto',
                                 origin='lower', interpolation='bicubic')
        axs[irun+2].contour(x, z, Ay, colors='k', linewidths=0.5,
                            levels=np.linspace(np.min(Ay), np.max(Ay), 8))

        if irun == 0:
            cbar = fig.colorbar(im1, cax=cbar_axs[1], extend='both')
            cbar_axs[1].tick_params(bottom=False, top=False, left=False, right=True)
            cbar_axs[1].tick_params(axis='y', which='major', direction='out')
            cbar_axs[1].tick_params(axis='y', which='minor', direction='in', right=False)
            cbar_axs[1].set_title(r'$V_x/v_{A0}$', fontsize=16)
            cbar.ax.tick_params(labelsize=12)

    for ax in axs:
        ax.set_xlim([xmin_di, xmax_di])
        ax.set_ylim([zmin_di, zmax_di])
    axs[0].text(0.02, 0.85, r"$\beta=0.25$", color='k', fontsize=16,
                bbox=dict(facecolor='w', alpha=0.7, edgecolor='k', pad=3.0),
                horizontalalignment='left', verticalalignment='center',
                transform=axs[0].transAxes)
    axs[1].text(0.02, 0.85, r"$\beta=10$", color='k', fontsize=16,
                bbox=dict(facecolor='w', alpha=0.7, edgecolor='k', pad=3.0),
                horizontalalignment='left', verticalalignment='center',
                transform=axs[1].transAxes)
    axs[2].text(0.02, 0.85, r"$\beta=0.25$", color='k', fontsize=16,
                bbox=dict(facecolor='w', alpha=0.7, edgecolor='k', pad=3.0),
                horizontalalignment='left', verticalalignment='center',
                transform=axs[2].transAxes)
    axs[3].text(0.02, 0.85, r"$\beta=10$", color='k', fontsize=16,
                bbox=dict(facecolor='w', alpha=0.7, edgecolor='k', pad=3.0),
                horizontalalignment='left', verticalalignment='center',
                transform=axs[3].transAxes)

    axs[0].text(-0.11, 0.85, "(a)", color="k", fontsize=20,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='center', verticalalignment='center',
                transform=axs[0].transAxes)
    axs[1].text(-0.11, 0.85, "(b)", color="k", fontsize=20,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='center', verticalalignment='center',
                transform=axs[1].transAxes)
    axs[2].text(-0.11, 0.85, "(c)", color="k", fontsize=20,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='center', verticalalignment='center',
                transform=axs[2].transAxes)
    axs[3].text(-0.11, 0.85, "(d)", color="k", fontsize=20,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='center', verticalalignment='center',
                transform=axs[3].transAxes)


    twpe = math.ceil(tindex * pic_info.dtwpe / 0.1) * 0.1
    twci = twpe / pic_info.wpe_wce / pic_info.mime
    text1 = r'$t\Omega_{ci}=' + ("{%0.0f}" % twci) + '$'
    fig.suptitle(text1, fontsize=16)
    img_dir = '../img/rate_problem/pub/'
    mkdir_p(img_dir)
    fname = img_dir + "absj_vout_" + str(tframe) + ".pdf"
    fig.savefig(fname, dpi=300)

    if show_plot:
        plt.show()
    else:
        plt.close()


def calc_peak_vout(plot_config, show_plot=True):
    """Calculate exhaust open angle
    """
    tframe = plot_config["tframe"]
    pic_run = plot_config["pic_run"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_run_dir = pic_info.run_dir
    fields_interval = pic_info.fields_interval
    tindex = fields_interval * tframe
    mime = pic_info.mime
    smime = math.sqrt(mime)
    lx_de = pic_info.lx_di * smime
    lz_de = pic_info.lz_di * smime
    dx_de = lx_de / pic_info.nx
    dz_de = lz_de / pic_info.nz
    nx = pic_info.nx
    nz = pic_info.nz
    nx_10di = int(nx * 10 / pic_info.lx_di)
    xs = nx//2 - nx_10di
    xe = nx//2 + nx_10di
    zs = nz//2 - nz//16
    ze = nz//2 + nz//16
    nxs = xe - xs
    nzs = ze - zs
    xmin = xs * dz_de
    xmax = xe * dz_de
    zmin_pic = -lz_de * 0.5
    zmin = zs * dz_de + zmin_pic
    zmax = ze * dz_de + zmin_pic

    # Peak outflow velocity near the X-point
    rho_vel = {}
    for species in ["e", "i"]:
        sname = "electron" if species == 'e' else "ion"
        fname = (pic_run_dir + "hydro_hdf5/T." + str(tindex) +
                 "/hydro_" + sname + "_" + str(tindex) + ".h5")
        hydro = {}
        with h5py.File(fname, 'r') as fh:
            group = fh["Timestep_" + str(tindex)]
            for var in ["rho", "jx"]:
                dset = group[var]
                hydro[var]= dset[xs:xe, 0, zs:ze]

        irho = 1.0 / hydro["rho"]
        vx = hydro["jx"] * irho
        var = "v" + species
        rho_vel[var+"x"] = np.squeeze(vx)
        rho_vel["n"+species] = np.squeeze(np.abs(hydro["rho"]))

    irho = 1.0 / (rho_vel["ne"] + rho_vel["ni"] * mime)
    vsx = (rho_vel["ne"] * rho_vel["vex"] +
           rho_vel["ni"] * rho_vel["vix"] * mime) * irho
    vsx = gaussian_filter(vsx, sigma=5)

    dtwpe = pic_info.dtwpe
    dtwce = pic_info.dtwce
    vpic_info = get_vpic_info(pic_run_dir)
    nb_n0 = vpic_info["nb/n0"]
    va = dtwce * math.sqrt(1.0 / mime) / dtwpe / math.sqrt(nb_n0)
    vsx_peak = np.max(np.abs(vsx)) / va

    fdir = '../data/rate_problem/vout_peak/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + "vout_peak_" + str(tframe) + ".dat"
    np.asarray([vsx_peak]).tofile(fname)


def simulation_evolution(plot_config, bg=0.0, show_plot=True):
    """Plot rate, opening angle, peak velocity, Bxm
    """
    pic_runs = ["mime400_Tb_T0_025",
                "mime400_Tb_T0_1",
                "mime400_Tb_T0_10_weak",
                "mime400_Tb_T0_40_nppc450"]
    open_angles = [33, 30, 25, 24]
    betas = [0.25, 1, 10, 40]
    nruns = len(pic_runs)
    if bg > 0.99:  # one or a few integer times of B0
        bg_str = "_bg" + str(int(bg))
    elif bg > 0.01:  # between 0 to B0
        bg_str = "_bg" + str(int(bg*10)).zfill(2)
    else:
        bg_str = ""
    fig = plt.figure(figsize=[7, 5])
    rect0 = [0.11, 0.53, 0.37, 0.4]
    hgap, vgap = 0.12, 0.03
    COLORS = palettable.tableau.Tableau_10.mpl_colors

    # model results
    beta0 = betas[0]
    input_dir = "../data/rate_problem/rate_model/cgl/"
    fname = input_dir + "dz_dx_" + str(beta0) + ".dat"
    dz_dxs = np.fromfile(fname)
    open_angle = np.arctan(dz_dxs) * 180 * 2 / math.pi
    open_index = []
    for angle in open_angles:
        index, _ = find_nearest(open_angle, angle)
        open_index.append(index)
    rate_model = np.zeros(nruns)
    vout_model = np.zeros(nruns)
    bxm_model = np.zeros(nruns)
    for irun, beta0 in enumerate(betas):
        fname = input_dir + "bxm_" + str(beta0) + ".dat"
        bxm = np.fromfile(fname)
        fname = input_dir + "vout_" + str(beta0) + ".dat"
        vout = np.fromfile(fname)
        fname = input_dir + "nout_" + str(beta0) + ".dat"
        nout = np.fromfile(fname)
        nin = np.ones(bxm.shape)  # incompressible
        vin = vout * nout * dz_dxs / nin
        rate = vin * bxm
        rate_model[irun] = rate[open_index[irun]]
        vout_model[irun] = vout[open_index[irun]]
        bxm_model[irun] = bxm[open_index[irun]]

    # reconnection rate
    ax = fig.add_axes(rect0)
    ax.set_prop_cycle('color', COLORS)
    for irun, pic_run in enumerate(pic_runs):
        pic_run += bg_str
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        pic_run_dir = pic_info.run_dir
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

        edrive, tdrive = get_edrive_params(pic_run_dir)
        vin = edrive * (1.0 - np.exp(-tfields/tdrive)) / wpe_wce / b0

        rrate_bflux = -np.gradient(bflux) / dtf
        rrate_bflux /= va * b0
        if "open" in pic_run or "test" in pic_run:
            rrate_bflux += vin
        print("Maximum rate: %f" % rrate_bflux.max())
        p1, = ax.plot(tfields_wci, rrate_bflux, linewidth=1,
                      label=r"$\beta=" + str(betas[irun]) + "$")
        ax.plot([tfields_wci[0], tfields_wci[-1]],
                [rate_model[irun], rate_model[irun]], color=p1.get_color(),
                linewidth=1, linestyle='--')

    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlim([0, 47])
    ax.tick_params(labelsize=12)
    ax.tick_params(axis='x', labelbottom=False)
    ax.set_ylabel(r'$E_R$', fontsize=16)
    xpos = 1.0 + 0.5 * hgap / rect0[2]
    ax.legend(loc=9, bbox_to_anchor=(xpos, 1.2),
              prop={'size': 12}, ncol=4,
              shadow=False, fancybox=False, frameon=False)
    ax.text(0.04, 0.9, "(a)", color="k", fontsize=20,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)

    # opening angle
    rect = np.copy(rect0)
    rect[0] += rect[2] + hgap
    ax = fig.add_axes(rect)
    ax.set_prop_cycle('color', COLORS)
    for irun, pic_run in enumerate(pic_runs):
        pic_run += bg_str
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        pic_run_dir = pic_info.run_dir
        b0 = pic_info.b0
        fdir = '../data/rate_problem/open_angle/' + pic_run + '/'
        nframes = len(os.listdir(fdir))
        open_angle = np.zeros(nframes)
        vpic_info = get_vpic_info(pic_run_dir)
        fields_interval = vpic_info["fields_interval"]
        tfields_wci = np.arange(nframes) * pic_info.dtwci * fields_interval
        for tframe in range(nframes):
            fname = fdir + "open_angle_" + str(tframe) + ".dat"
            fdata = np.fromfile(fname)
            open_angle[tframe] = fdata[0]
        p1, = ax.plot(tfields_wci, open_angle*2, linewidth=1,
                      label=r"$\beta=" + str(betas[irun]) + "$")
        # ax.plot([tfields_wci[0], tfields_wci[-1]],
        #         [open_angles[irun], open_angles[irun]], color=p1.get_color(),
        #         linewidth=1, linestyle='--')

    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlim([0, 47])
    ax.set_ylim([0, 55])
    ax.tick_params(labelsize=12)
    ax.tick_params(axis='x', labelbottom=False)
    ax.set_ylabel(r'Opening Angle ($^\circ$)', fontsize=16)
    ax.text(0.96, 0.9, "(b)", color="k", fontsize=20,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='right', verticalalignment='center',
            transform=ax.transAxes)

    # peak outflow velocity near the X-point
    rect = np.copy(rect0)
    rect[1] -= rect[3] + vgap
    ax = fig.add_axes(rect)
    ax.set_prop_cycle('color', COLORS)
    for irun, pic_run in enumerate(pic_runs):
        pic_run += bg_str
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        pic_run_dir = pic_info.run_dir
        b0 = pic_info.b0
        fdir = '../data/rate_problem/vout_peak/' + pic_run + '/'
        nframes = len(os.listdir(fdir))
        vout_peak = np.zeros(nframes)
        vpic_info = get_vpic_info(pic_run_dir)
        fields_interval = vpic_info["fields_interval"]
        tfields_wci = np.arange(nframes) * pic_info.dtwci * fields_interval
        for tframe in range(nframes):
            fname = fdir + "vout_peak_" + str(tframe) + ".dat"
            fdata = np.fromfile(fname)
            vout_peak[tframe] = fdata[0]
        p1, = ax.plot(tfields_wci, vout_peak, linewidth=1,
                      label=r"$\beta=" + str(betas[irun]) + "$")
        ax.plot([tfields_wci[0], tfields_wci[-1]],
                [vout_model[irun], vout_model[irun]], color=p1.get_color(),
                linewidth=1, linestyle='--')

    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlim([0, 47])
    ax.tick_params(labelsize=12)
    ax.set_xlabel(r'$t\Omega_{ci}$', fontsize=16)
    ax.set_ylabel(r'$|V_x|_\text{peak}/v_{A0}$', fontsize=16)
    ax.text(0.04, 0.9, "(c)", color="k", fontsize=20,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)

    # Bxm
    rect[0] += rect[2] + hgap
    ax = fig.add_axes(rect)
    ax.set_prop_cycle('color', COLORS)
    for irun, pic_run in enumerate(pic_runs):
        pic_run += bg_str
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        pic_run_dir = pic_info.run_dir
        b0 = pic_info.b0
        fdir = '../data/rate_problem/bxm/' + pic_run + '/'
        nframes = len(os.listdir(fdir)) // 2
        bxm = np.zeros(nframes)
        vpic_info = get_vpic_info(pic_run_dir)
        fields_interval = vpic_info["fields_interval"]
        tfields_wci = np.arange(nframes) * pic_info.dtwci * fields_interval
        for tframe in range(nframes):
            fname = fdir + 'bxm_' + str(tframe) + '.dat'
            fdata = np.fromfile(fname)
            bxm[tframe] = fdata[0]
        p1, = ax.plot(tfields_wci, bxm/b0, linewidth=1,
                      label=r"$\beta=" + str(betas[irun]) + "$")
        ax.plot([tfields_wci[0], tfields_wci[-1]],
                [bxm_model[irun], bxm_model[irun]], color=p1.get_color(),
                linewidth=1, linestyle='--')
    ax.text(0.96, 0.9, "(d)", color="k", fontsize=20,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='right', verticalalignment='center',
            transform=ax.transAxes)

    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlim([0, 47])
    ax.tick_params(labelsize=12)
    ax.set_xlabel(r'$t\Omega_{ci}$', fontsize=16)
    ax.set_ylabel(r'$B_{xm}/B_0$', fontsize=16)

    img_dir = '../img/rate_problem/pub/'
    mkdir_p(img_dir)
    fname = img_dir + "simulation" + bg_str + "_2.pdf"
    fig.savefig(fname)

    plt.show()


def simulation_evolution2(plot_config, bg=0.0, show_plot=True):
    """Plot rate, opening angle, peak velocity, Bxm
    """
    pic_runs = ["mime400_Tb_T0_025",
                "mime400_Tb_T0_1",
                "mime400_Tb_T0_10_weak",
                "mime400_Tb_T0_40_nppc450_old"]
    open_angles = [33, 30, 25, 24]
    betas = [0.25, 1, 10, 40]
    nruns = len(pic_runs)
    if bg > 0.99:  # one or a few integer times of B0
        bg_str = "_bg" + str(int(bg))
    elif bg > 0.01:  # between 0 to B0
        bg_str = "_bg" + str(int(bg*10)).zfill(2)
    else:
        bg_str = ""
    fig = plt.figure(figsize=[7, 5])
    rect0 = [0.11, 0.53, 0.37, 0.4]
    hgap, vgap = 0.12, 0.03
    COLORS = palettable.tableau.Tableau_10.mpl_colors

    # model results
    beta0 = betas[0]
    input_dir = "../data/rate_problem/rate_model/cgl/"
    fname = input_dir + "dz_dx_" + str(beta0) + ".dat"
    dz_dxs = np.fromfile(fname)
    open_angle = np.arctan(dz_dxs) * 180 * 2 / math.pi
    open_index = []
    for angle in open_angles:
        index, _ = find_nearest(open_angle, angle)
        open_index.append(index)
    rate_model = np.zeros(nruns)
    vout_model = np.zeros(nruns)
    bxm_model = np.zeros(nruns)
    for irun, beta0 in enumerate(betas):
        fname = input_dir + "bxm_" + str(beta0) + ".dat"
        bxm = np.fromfile(fname)
        fname = input_dir + "vout_" + str(beta0) + ".dat"
        vout = np.fromfile(fname)
        fname = input_dir + "nout_" + str(beta0) + ".dat"
        nout = np.fromfile(fname)
        nin = np.ones(bxm.shape)  # incompressible
        vin = vout * nout * dz_dxs / nin
        rate = vin * bxm
        rate_model[irun] = rate[open_index[irun]]
        vout_model[irun] = vout[open_index[irun]]
        bxm_model[irun] = bxm[open_index[irun]]

    # opening angle
    ax = fig.add_axes(rect0)
    ax.set_prop_cycle('color', COLORS)
    for irun, pic_run in enumerate(pic_runs):
        pic_run += bg_str
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        pic_run_dir = pic_info.run_dir
        b0 = pic_info.b0
        fdir = '../data/rate_problem/open_angle/' + pic_run + '/'
        nframes = len(os.listdir(fdir))
        open_angle = np.zeros(nframes)
        vpic_info = get_vpic_info(pic_run_dir)
        fields_interval = vpic_info["fields_interval"]
        tfields_wci = np.arange(nframes) * pic_info.dtwci * fields_interval
        for tframe in range(nframes):
            fname = fdir + "open_angle_" + str(tframe) + ".dat"
            fdata = np.fromfile(fname)
            open_angle[tframe] = fdata[0]
        p1, = ax.plot(tfields_wci, open_angle*2, linewidth=1,
                      label=r"$\beta=" + str(betas[irun]) + "$")
        # ax.plot([tfields_wci[0], tfields_wci[-1]],
        #         [open_angles[irun], open_angles[irun]], color=p1.get_color(),
        #         linewidth=1, linestyle='--')

    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlim([0, 47])
    ax.set_ylim([0, 55])
    ax.tick_params(labelsize=12)
    ax.tick_params(axis='x', labelbottom=False)
    ax.set_ylabel(r'Opening Angle ($^\circ$)', fontsize=16)
    ax.text(0.04, 0.9, "(a)", color="k", fontsize=20,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)
    xpos = 1.0 + 0.5 * hgap / rect0[2]
    ax.legend(loc=9, bbox_to_anchor=(xpos, 1.2),
              prop={'size': 12}, ncol=4,
              shadow=False, fancybox=False, frameon=False)

    # Bxm
    rect = np.copy(rect0)
    rect[0] += rect[2] + hgap
    ax = fig.add_axes(rect)
    ax.set_prop_cycle('color', COLORS)
    for irun, pic_run in enumerate(pic_runs):
        pic_run += bg_str
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        pic_run_dir = pic_info.run_dir
        b0 = pic_info.b0
        fdir = '../data/rate_problem/bxm/' + pic_run + '/'
        nframes = len(os.listdir(fdir)) // 2
        bxm = np.zeros(nframes)
        vpic_info = get_vpic_info(pic_run_dir)
        fields_interval = vpic_info["fields_interval"]
        tfields_wci = np.arange(nframes) * pic_info.dtwci * fields_interval
        for tframe in range(nframes):
            fname = fdir + 'bxm_' + str(tframe) + '.dat'
            fdata = np.fromfile(fname)
            bxm[tframe] = fdata[0]
        p1, = ax.plot(tfields_wci, bxm/b0, linewidth=1,
                      label=r"$\beta=" + str(betas[irun]) + "$")
        # ax.plot([tfields_wci[0], tfields_wci[-1]],
        #         [bxm_model[irun], bxm_model[irun]], color=p1.get_color(),
        #         linewidth=1, linestyle='--')
    ax.tick_params(axis='x', labelbottom=False)
    ax.text(0.96, 0.9, "(b)", color="k", fontsize=20,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='right', verticalalignment='center',
            transform=ax.transAxes)

    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlim([0, 47])
    ax.tick_params(labelsize=12)
    ax.set_ylabel(r'$B_{xm}/B_0$', fontsize=16)

    # peak outflow velocity near the X-point
    rect = np.copy(rect0)
    rect[1] -= rect[3] + vgap
    ax = fig.add_axes(rect)
    ax.set_prop_cycle('color', COLORS)
    for irun, pic_run in enumerate(pic_runs):
        pic_run += bg_str
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        pic_run_dir = pic_info.run_dir
        b0 = pic_info.b0
        fdir = '../data/rate_problem/vout_peak/' + pic_run + '/'
        nframes = len(os.listdir(fdir))
        vout_peak = np.zeros(nframes)
        vpic_info = get_vpic_info(pic_run_dir)
        fields_interval = vpic_info["fields_interval"]
        tfields_wci = np.arange(nframes) * pic_info.dtwci * fields_interval
        for tframe in range(nframes):
            fname = fdir + "vout_peak_" + str(tframe) + ".dat"
            fdata = np.fromfile(fname)
            vout_peak[tframe] = fdata[0]
        p1, = ax.plot(tfields_wci, vout_peak, linewidth=1,
                      label=r"$\beta=" + str(betas[irun]) + "$")
        # ax.plot([tfields_wci[0], tfields_wci[-1]],
        #         [vout_model[irun], vout_model[irun]], color=p1.get_color(),
        #         linewidth=1, linestyle='--')

    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlim([0, 47])
    ax.tick_params(labelsize=12)
    ax.set_xlabel(r'$t\Omega_{ci}$', fontsize=16)
    ax.set_ylabel(r'$|V_x|_\text{peak}/v_{A0}$', fontsize=16)
    ax.text(0.04, 0.9, "(c)", color="k", fontsize=20,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)

    # reconnection rate
    rect[0] += rect[2] + hgap
    ax = fig.add_axes(rect)
    ax.set_prop_cycle('color', COLORS)
    for irun, pic_run in enumerate(pic_runs):
        pic_run += bg_str
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        pic_run_dir = pic_info.run_dir
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

        edrive, tdrive = get_edrive_params(pic_run_dir)
        vin = edrive * (1.0 - np.exp(-tfields/tdrive)) / wpe_wce / b0

        rrate_bflux = -np.gradient(bflux) / dtf
        rrate_bflux /= va * b0
        if "open" in pic_run or "test" in pic_run:
            rrate_bflux += vin
        print("Maximum rate: %f" % rrate_bflux.max())
        p1, = ax.plot(tfields_wci, rrate_bflux, linewidth=1,
                      label=r"$\beta=" + str(betas[irun]) + "$")
        # ax.plot([tfields_wci[0], tfields_wci[-1]],
        #         [rate_model[irun], rate_model[irun]], color=p1.get_color(),
        #         linewidth=1, linestyle='--')

    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlim([0, 47])
    ax.tick_params(labelsize=12)
    ax.set_xlabel(r'$t\Omega_{ci}$', fontsize=16)
    ax.set_ylabel(r'$E_R$', fontsize=16)
    ax.text(0.96, 0.9, "(d)", color="k", fontsize=20,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='right', verticalalignment='center',
            transform=ax.transAxes)

    img_dir = '../img/rate_problem/pub/'
    mkdir_p(img_dir)
    fname = img_dir + "simulation" + bg_str + "_2.pdf"
    fig.savefig(fname)

    plt.show()


def calc_rates_low_beta(bxm_bx0, sigma0, nonrec=False):
    """
    Calculate reconnection rates and other quantities for low-beta conditions
    """
    sigmam = bxm_bx0**2 * sigma0
    dz_dx = math.sqrt((1-bxm_bx0)/(1+bxm_bx0))
    bzm_bxm = dz_dx
    delta_l = dz_dx
    d2 = delta_l**2
    vout_m = math.sqrt((1-d2)*sigmam / (1 + (1-d2)*sigmam))
    va0 = math.sqrt(sigma0 / (1 + sigma0))
    vam = math.sqrt(sigmam / (1 + sigmam))
    if nonrec:
        rec0 = bxm_bx0**2 * dz_dx * math.sqrt(1-dz_dx**2)
    else:
        rec0 = bzm_bxm * bxm_bx0 * vout_m / va0
    recm = bzm_bxm * vout_m / vam
    vinm = recm * vam
    open_angle = math.atan(dz_dx) * 180 / math.pi
    print("Reconnection rate: %f" % rec0)
    print("Reconnection rate (microscale magnetic field): %f" % recm)
    print("Inflow velocity/va0: %f" % (vinm/va0))
    print("Outflow velocity/va0: %f" % (vout_m/va0))
    print("Opening angle: %f" % open_angle)
    print("")


def calc_rates_high_beta(bxm_bx0, sigma0, beta0, epsilon):
    """
    Calculate reconnection rates and other quantities for high-beta conditions
    """
    sigmam = bxm_bx0**2 * sigma0
    dz_dx = math.sqrt(((1+beta0) - beta0*bxm_bx0 - bxm_bx0**2) /
            (epsilon * (1+bxm_bx0)**2))
    bzm_bxm = dz_dx
    A = epsilon - dz_dx**2 - beta0 * dz_dx / bxm_bx0
    vout_m = math.sqrt(A*sigmam / (1 + A*sigmam))
    va0 = math.sqrt(sigma0 / (1 + sigma0))
    vam = math.sqrt(sigmam / (1 + sigmam))
    rec0 = dz_dx * bxm_bx0**2 * math.sqrt(A * (1 + sigma0) / (1 + A * sigmam))
    recm = bzm_bxm * vout_m / vam
    vinm = recm * vam
    open_angle = math.atan(dz_dx) * 180 / math.pi
    print("Reconnection rate: %f" % rec0)
    print("Reconnection rate (microscale magnetic field): %f" % recm)
    print("Inflow velocity/va0: %f" % (vinm/va0))
    print("Outflow velocity/va0: %f" % (vout_m/va0))
    print("Opening angle: %f" % open_angle)
    print("")


def plot_rates_scaling(sigma0, beta0):
    """
    Plot the scaling of reconnection rates and other quantities for high-beta conditions
    """
    bxm_bx0 = np.linspace(1E-5, 1, 10000)
    sigmam = bxm_bx0**2 * sigma0
    bh = (1 + bxm_bx0) * 0.5
    epsilon = 1 - 0.5 * beta0 * (1/bh**2 - bh) / bh**2
    dz_dx = np.sqrt(((1+beta0) - beta0*bxm_bx0 - bxm_bx0**2) /
            (epsilon * (1+bxm_bx0)**2))
    bzm_bxm = dz_dx
    A = epsilon - dz_dx**2 - beta0 * dz_dx / bxm_bx0
    vout_m = np.sqrt(A*sigmam / (1 + A*sigmam))
    vout_m[np.isnan(vout_m)] = 0
    va0 = np.sqrt(sigma0 / (1 + sigma0))
    vam = np.sqrt(sigmam / (1 + sigmam))
    rec0 = dz_dx * bxm_bx0**2 * np.sqrt(A * (1 + sigma0) / (1 + A * sigmam))
    rec0[np.isnan(rec0)] = 0
    recm = bzm_bxm * vout_m / vam
    vinm = recm * vam
    open_angle = np.arctan(dz_dx) * 180 / math.pi
    fig1 = plt.figure(figsize=[16, 9])
    rect0 = [0.06, 0.55, 0.26, 0.38]
    hgap, vgap = 0.06, 0.06
    rect = np.copy(rect0)
    ax = fig1.add_axes(rect)
    ax.plot(dz_dx, rec0, label=r'$R_0$')
    ax.plot(dz_dx, recm, label=r'$R_m$')
    ax.set_xlim([0, 1.2])
    ax.set_ylim([0, 1.0])
    ax.grid()
    ax.tick_params(labelsize=12)
    ax.legend(loc='best', prop={'size': 16}, ncol=1,
              shadow=False, fancybox=True, frameon=True)

    rect[0] += rect[2] + hgap
    ax = fig1.add_axes(rect)
    ax.plot(dz_dx, bxm_bx0, label=r'$B_{xm}/B_{x0}$')
    ax.set_xlim([0, 1.2])
    ax.grid()
    ax.tick_params(labelsize=12)
    ax.legend(loc='best', prop={'size': 16}, ncol=1,
              shadow=False, fancybox=True, frameon=True)

    rect[0] += rect[2] + hgap
    ax = fig1.add_axes(rect)
    ax.plot(dz_dx, A, label=r'$A$')
    ax.set_xlim([0, 1.2])
    ax.set_ylim([-0.5, 1.1])
    ax.grid()
    ax.tick_params(labelsize=12)
    ax.legend(loc='best', prop={'size': 16}, ncol=1,
              shadow=False, fancybox=True, frameon=True)

    rect = np.copy(rect0)
    rect[1] -= rect[3] + vgap
    ax = fig1.add_axes(rect)
    COLORS = palettable.colorbrewer.qualitative.Set1_9.mpl_colors
    ax.set_prop_cycle('color', COLORS)
    p1, = ax.plot(dz_dx, vout_m/va0, label=r'$V_{out, m}/V_{A0}$')
    p2, = ax.plot(dz_dx, vinm/va0, label=r'$V_{in, m}/V_{A0}$')
    ax.plot(dz_dx, vout_m/vam, label=r'$V_{out, m}/V_{Am}$',
            color=p1.get_color(), linestyle='--')
    ax.plot(dz_dx, vinm/vam, label=r'$V_{in, m}/V_{Am}$',
            color=p2.get_color(), linestyle='--')
    ax.set_xlim([0, 1.2])
    ax.set_ylim([0, 1.1])
    ax.grid()
    ax.set_xlabel(r'$\Delta z/\Delta x$', fontsize=20)
    ax.tick_params(labelsize=12)
    ax.legend(loc='best', prop={'size': 16}, ncol=1,
              shadow=False, fancybox=True, frameon=True)

    rect[0] += rect[2] + hgap
    ax = fig1.add_axes(rect)
    ax.plot(dz_dx, open_angle, label='Opening angle')
    ax.set_xlim([0, 1.2])
    ax.grid()
    ax.tick_params(labelsize=12)
    ax.set_xlabel(r'$\Delta z/\Delta x$', fontsize=20)
    ax.legend(loc='best', prop={'size': 16}, ncol=1,
              shadow=False, fancybox=True, frameon=True)

    rect[0] += rect[2] + hgap
    ax = fig1.add_axes(rect)
    ax.plot(dz_dx, epsilon, label=r'$\epsilon$')
    ax.set_xlim([0, 1.2])
    ax.grid()
    ax.tick_params(labelsize=12)
    ax.set_xlabel(r'$\Delta z/\Delta x$', fontsize=20)
    ax.legend(loc='best', prop={'size': 16}, ncol=1,
              shadow=False, fancybox=True, frameon=True)

    text1 = r"$\beta_0=" + str(beta0) + "$"
    fig1.suptitle(text1, fontsize=20)

    fname = "img/rate_scaling_beta0_" + str(beta0) + ".pdf"
    fig1.savefig(fname)

    plt.show()


def rate_model(beta0, le_closure=False):
    """Reconnection rate model

    Args:
        beta0: plasma beta in the inflow region
        le_closure: whether to use Le-Egedal closure
    """
    nbins = 100
    dz_dxs = np.linspace(0, 1, nbins)
    bxm = np.zeros(nbins)
    nout = np.zeros(nbins)
    epsilon = 1E-12
    sbeta = math.sqrt(2*beta0/math.pi)
    for ibin, dz_dx in enumerate(dz_dxs):
        if le_closure:
            b1 = 1.0
            f = 1.0
            while abs(f) > epsilon:  # Newton's method
                f1 = (1.0 - b1**3) / (1.0 + b1)
                f2 = (1.0 - b1**2) / beta0
                f3 = (1.0 + b1)**2 * dz_dx**2 / beta0
                tmp1 = 8.0 / (5.0 + b1)
                tmp2 = (1.0 + b1) * math.pi / (2.0 + b1) / 3
                tmp3 = -(4.0 + (1.0 + b1)**2) / (3.0 + b1)
                f4 = -(1.0 + b1) * (tmp1 + tmp2 + tmp3) * dz_dx**2 / 2
                f = f1 + f2 - f3 - f4
                df1 = -(2.0 * b1 + 1) / (1.0 + b1)
                df2 = -2.0 * b1 / beta0
                df3 = 2 * (1.0 + b1) * dz_dx**2 / beta0
                df4 = -(tmp1 + tmp2 + tmp3) * dz_dx**2 / 2
                tmp1 = -8.0 / (5.0 + b1)**2
                tmp2 = math.pi / (2.0 + b1)**2 / 3
                tmp3 = -(b1**2 + 6.0*b1 + 1) / (3.0 + b1)**2
                df5 = -(1.0 + b1) * (tmp1 + tmp2 + tmp3) * dz_dx**2 / 2
                df = df1 + df2 - df3 - df4 - df5
                b1 -= f / df
            bxm[ibin] = b1
        else:
            tmp1 = 1 + beta0
            tmp2 = (1 + 0.5 * beta0) * dz_dx**2
            tmp3 = (1 - 0.5 * beta0) * dz_dx**2
            bxm[ibin] = (tmp1 - tmp3) / (tmp1 + tmp2)
        n1 = 1.0
        f = 1.0
        while abs(f) > epsilon:  # Newton's method
            tmp1 = 10 * bxm[ibin]**2 * (dz_dx**2 - 1) / n1
            tmp2 = math.sqrt(8*beta0/math.pi - tmp1)
            vo = -2 * sbeta / 5 + tmp2 /5
            dvdn = bxm[ibin]**2 * (dz_dx**2 - 1) / (n1**2 * tmp2)
            df1 = bxm[ibin]**2 + 5 * beta0 * (1 + 2*bxm[ibin]) * bxm[ibin] / 12
            f1 = df1 * n1
            f2 = n1**3 * vo**2 * dz_dx**2 / (2 * bxm[ibin])
            f3 = 5 * beta0 * (1 + 2*bxm[ibin]) * bxm[ibin]**2 / 12
            f4 = 5 * n1 * vo * sbeta * bxm[ibin] / 2
            f5 = 3 * n1 * vo**2 * bxm[ibin]
            f6 = bxm[ibin]**3 * dz_dx**2
            df2 = (dz_dx**2/(2*bxm[ibin])) * (3 * n1**2 * vo**2 +
                                              2 * n1**3 * vo * dvdn)
            df3 = 5 * sbeta * (vo + n1*dvdn) * bxm[ibin] / 2
            df4 = 3 * vo * (vo + 2*n1*dvdn) * bxm[ibin]
            f = f1 + f2 - f3 - f4 - f5 - f6
            df = df1 + df2 - df3 - df4
            n1 -= f / df
        nout[ibin] = n1

    # Save the data
    if le_closure:
        fdir = "data/le/"
    else:
        fdir = "data/cgl/"
    mkdir_p(fdir)
    fname = fdir + "dz_dx_" + str(beta0) + ".dat"
    dz_dxs.tofile(fname)
    fname = fdir + "bxm_" + str(beta0) + ".dat"
    bxm.tofile(fname)
    fname = fdir + "nout_" + str(beta0) + ".dat"
    nout.tofile(fname)

    # Plot quantities with different normalization
    nin = np.copy(bxm)  # Assumption
    tmp1 = 10 * bxm**2 * (dz_dxs**2 - 1) / nout
    tmp2 = np.sqrt(8*beta0/math.pi - tmp1)
    vout = -2 * math.sqrt(2*beta0/math.pi) / 5 + tmp2 /5
    vin = vout * nout * dz_dxs / nin
    rate = vin * bxm

    # Alfven speed using B and n upstream of the in diffusion region
    vam = bxm / np.sqrt(nin)
    vin_m = vin / vam
    vout_m = vout / vam
    rate_m = vin_m
    open_angle = np.arctan(dz_dxs) * 180 / math.pi

    if le_closure:
        fdir = "img/le/"
    else:
        fdir = "img/cgl/"
    mkdir_p(fdir)

    fig = plt.figure(figsize=[7, 5])
    rect = [0.12, 0.15, 0.8, 0.8]
    hgap, vgap = 0.06, 0.06
    ax = fig.add_axes(rect)
    p1, = ax.plot(open_angle, rate, label=r'$v_\text{in}B_{xm}/v_{A0}B_0$')
    ax.plot(open_angle, rate_m, linestyle='--', color=p1.get_color(),
            label=r'$v_\text{in}/v_{Am}$')
    ax.legend(loc=2, prop={'size': 16}, ncol=1,
              shadow=False, fancybox=True, frameon=True)
    ax.set_xlim([0, 45])
    ax.set_xlabel(r'Opening Angle', fontsize=16)
    ax.set_ylabel(r'$E_R$', fontsize=16)
    ax.grid(True)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(labelsize=12)
    beta_str = str(beta0).replace(".", "_")
    fname = fdir + 'rate_' + beta_str + '.pdf'
    fig.savefig(fname)

    fig = plt.figure(figsize=[7, 5])
    rect = [0.12, 0.15, 0.8, 0.8]
    hgap, vgap = 0.06, 0.06
    ax = fig.add_axes(rect)
    p1, = ax.plot(open_angle, vin, label=r'$v_\text{in}/v_{A0}$')
    ax.plot(open_angle, vin_m, linestyle='--', color=p1.get_color(),
            label=r'$v_\text{in}/v_{Am}$')
    p2, = ax.plot(open_angle, vout, label=r'$v_\text{out}/v_{A0}$')
    ax.plot(open_angle, vout_m, linestyle='--', color=p2.get_color(),
            label=r'$v_\text{out}/v_{Am}$')
    ax.legend(loc=6, prop={'size': 16}, ncol=1,
              shadow=False, fancybox=True, frameon=True)
    ax.set_xlim([0, 45])
    ax.set_xlabel(r'Opening Angle', fontsize=16)
    ax.grid(True)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(labelsize=12)
    beta_str = str(beta0).replace(".", "_")
    fname = fdir + 'vin_out_' + beta_str + '.pdf'
    fig.savefig(fname)

    plt.close("all")
    # plt.show()


def rate_model_fermi(beta0, le_closure=False):
    """Reconnection rate model with Fermi heating

    Args:
        beta0: plasma beta in the inflow region
        le_closure: whether to use Le-Egedal closure
    """
    nbins = 100
    dz_dxs = np.linspace(0, 1, nbins)
    bxm = np.zeros(nbins)
    nout = np.zeros(nbins)
    epsilon = 1E-12
    sbeta = math.sqrt(2*beta0/math.pi)
    for ibin, dz_dx in enumerate(dz_dxs):
        if le_closure:
            b1 = 1.0
            f = 1.0
            while abs(f) > epsilon:  # Newton's method
                f1 = (1.0 - b1**3) / (1.0 + b1)
                f2 = (1.0 - b1**2) / beta0
                f3 = (1.0 + b1)**2 * dz_dx**2 / beta0
                tmp1 = 8.0 / (5.0 + b1)
                tmp2 = (1.0 + b1) * math.pi / (2.0 + b1) / 3
                tmp3 = -(4.0 + (1.0 + b1)**2) / (3.0 + b1)
                f4 = -(1.0 + b1) * (tmp1 + tmp2 + tmp3) * dz_dx**2 / 2
                f = f1 + f2 - f3 - f4
                df1 = -(2.0 * b1 + 1) / (1.0 + b1)
                df2 = -2.0 * b1 / beta0
                df3 = 2 * (1.0 + b1) * dz_dx**2 / beta0
                df4 = -(tmp1 + tmp2 + tmp3) * dz_dx**2 / 2
                tmp1 = -8.0 / (5.0 + b1)**2
                tmp2 = math.pi / (2.0 + b1)**2 / 3
                tmp3 = -(b1**2 + 6.0*b1 + 1) / (3.0 + b1)**2
                df5 = -(1.0 + b1) * (tmp1 + tmp2 + tmp3) * dz_dx**2 / 2
                df = df1 + df2 - df3 - df4 - df5
                b1 -= f / df
            bxm[ibin] = b1
        else:
            tmp1 = 1 + beta0
            tmp2 = (1 + 0.5 * beta0) * dz_dx**2
            tmp3 = (1 - 0.5 * beta0) * dz_dx**2
            bxm[ibin] = (tmp1 - tmp3) / (tmp1 + tmp2)
        n1 = 1.0
        f = 1.0
        # firehose = 1.0 - (1.0 / bxm[ibin] - 1.0) * beta0 / 2.0
        firehose = 1.0
        while abs(f) > epsilon:  # Newton's method
            # tmp1 = 66 * bxm[ibin]**2 * (dz_dx**2 - firehose) / n1
            # tmp3 = 8*beta0/math.pi - tmp1
            # if tmp3 < 0:
            #     n1 = 0.0
            #     break
            # else:
            #     tmp2 = math.sqrt(tmp3)
            # vo = -2 * sbeta / 11 + tmp2 / 11
            # dvdn = 3 * bxm[ibin]**2 * (dz_dx**2 - firehose) / (n1**2 * tmp2)
            tmp1 = 11 * n1 + 3 * bxm[ibin]
            c = 8*beta0*n1**2/math.pi - 6 * tmp1 * bxm[ibin]**2 * (dz_dx**2 - firehose)
            if c < 0:
                n1 = 0.0
                break
            vo = (-2 * sbeta * n1 + math.sqrt(c)) / tmp1
            tmp3 = 8*beta0*n1/math.pi - 33*bxm[ibin]**2 * (dz_dx**2 - firehose)
            dvdn = (-2 * sbeta + tmp3/math.sqrt(c)) / tmp1
            dvdn -= 11 * (-2 * sbeta * n1 + math.sqrt(c)) / tmp1**2
            df1 = bxm[ibin]**2 + 5 * beta0 * (1 + 2*bxm[ibin]) * bxm[ibin] / 12
            f1 = df1 * n1
            f2 = n1**3 * vo**2 * dz_dx**2 / (2 * bxm[ibin])
            f3 = 5 * beta0 * (1 + 2*bxm[ibin]) * bxm[ibin]**2 / 12
            f4 = 5 * n1 * vo * sbeta * bxm[ibin] / 6
            f5 = 13 * n1 * vo**2 * bxm[ibin] / 6
            f6 = bxm[ibin]**3 * dz_dx**2
            df2 = (dz_dx**2/(2*bxm[ibin])) * (3 * n1**2 * vo**2 +
                                              2 * n1**3 * vo * dvdn)
            df3 = 5 * sbeta * (vo + n1*dvdn) * bxm[ibin] / 6
            df4 = 13 * vo * (vo + 2*n1*dvdn) * bxm[ibin] / 6
            f = f1 + f2 - f3 - f4 - f5 - f6
            df = df1 + df2 - df3 - df4
            n1 -= f / df
        nout[ibin] = n1

    tmp1 = div0(10 * bxm**2 * (dz_dxs**2 - 1), nout)
    tmp2 = np.sqrt(8*beta0/math.pi - tmp1)
    vout = -2 * math.sqrt(2*beta0/math.pi) / 5 + tmp2 /5

    # Save the data
    if le_closure:
        fdir = "data/le/"
    else:
        fdir = "data/cgl/"
    mkdir_p(fdir)
    fname = fdir + "dz_dx_" + str(beta0) + ".dat"
    dz_dxs.tofile(fname)
    fname = fdir + "bxm_" + str(beta0) + ".dat"
    bxm.tofile(fname)
    fname = fdir + "nout_" + str(beta0) + ".dat"
    nout.tofile(fname)
    fname = fdir + "vout_" + str(beta0) + ".dat"
    vout.tofile(fname)

    # Plot quantities with different normalization
    nin = np.copy(bxm)  # Assumption
    tmp1 = div0(10 * bxm**2 * (dz_dxs**2 - 1), nout)
    tmp2 = np.sqrt(8*beta0/math.pi - tmp1)
    vout = -2 * math.sqrt(2*beta0/math.pi) / 5 + tmp2 /5
    vin = vout * nout * dz_dxs / nin
    rate = vin * bxm

    # Alfven speed using B and n upstream of the in diffusion region
    vam = bxm / np.sqrt(nin)
    vin_m = vin / vam
    vout_m = vout / vam
    rate_m = vin_m
    open_angle = np.arctan(dz_dxs) * 180 / math.pi

    if le_closure:
        fdir = "img/le/"
    else:
        fdir = "img/cgl/"
    mkdir_p(fdir)

    fig = plt.figure(figsize=[7, 5])
    rect = [0.12, 0.15, 0.8, 0.8]
    hgap, vgap = 0.06, 0.06
    ax = fig.add_axes(rect)
    p1, = ax.plot(open_angle, rate, label=r'$v_\text{in}B_{xm}/v_{A0}B_0$')
    ax.plot(open_angle, rate_m, linestyle='--', color=p1.get_color(),
            label=r'$v_\text{in}/v_{Am}$')
    ax.legend(loc=2, prop={'size': 16}, ncol=1,
              shadow=False, fancybox=True, frameon=True)
    ax.set_xlim([0, 45])
    ax.set_xlabel(r'Opening Angle', fontsize=16)
    ax.set_ylabel(r'$E_R$', fontsize=16)
    ax.grid(True)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(labelsize=12)
    beta_str = str(beta0).replace(".", "_")
    fname = fdir + 'rate_' + beta_str + '.pdf'
    fig.savefig(fname)

    fig = plt.figure(figsize=[7, 5])
    rect = [0.12, 0.15, 0.8, 0.8]
    hgap, vgap = 0.06, 0.06
    ax = fig.add_axes(rect)
    p1, = ax.plot(open_angle, vin, label=r'$v_\text{in}/v_{A0}$')
    ax.plot(open_angle, vin_m, linestyle='--', color=p1.get_color(),
            label=r'$v_\text{in}/v_{Am}$')
    p2, = ax.plot(open_angle, vout, label=r'$v_\text{out}/v_{A0}$')
    ax.plot(open_angle, vout_m, linestyle='--', color=p2.get_color(),
            label=r'$v_\text{out}/v_{Am}$')
    ax.legend(loc=6, prop={'size': 16}, ncol=1,
              shadow=False, fancybox=True, frameon=True)
    ax.set_xlim([0, 45])
    ax.set_xlabel(r'Opening Angle', fontsize=16)
    ax.grid(True)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(labelsize=12)
    beta_str = str(beta0).replace(".", "_")
    fname = fdir + 'vin_out_' + beta_str + '.pdf'
    fig.savefig(fname)

    fig = plt.figure(figsize=[7, 5])
    rect = [0.12, 0.15, 0.8, 0.8]
    hgap, vgap = 0.06, 0.06
    ax = fig.add_axes(rect)
    p1, = ax.plot(open_angle, nout/nin,
                  label=r'$n_\text{out}/n_\text{in}$')
    ax.set_xlim([0, 45])
    ax.set_xlabel(r'Opening Angle', fontsize=16)
    ax.set_xlabel(r'$n_\text{out}/n_\text{in}$', fontsize=16)
    ax.grid(True)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(labelsize=12)
    beta_str = str(beta0).replace(".", "_")
    fname = fdir + 'nout_in_' + beta_str + '.pdf'
    fig.savefig(fname)

    plt.close("all")
    # plt.show()


def rate_model_fermi_incomp(beta0, le_closure=False):
    """Reconnection rate model with Fermi heating and incompressibility

    Args:
        beta0: plasma beta in the inflow region
        le_closure: whether to use Le-Egedal closure
    """
    print("Plasma beta: %f" % beta0)
    nbins = 10000
    dz_dxs = np.linspace(0, 1, nbins)
    bxm = np.zeros(nbins)
    nout = np.zeros(nbins)
    vout = np.zeros(nbins)
    firehose = np.zeros(nbins)
    epsilon = 1E-12
    sbeta = math.sqrt(2*beta0/math.pi)
    sbeta_h = sbeta * 0.5
    for ibin, dz_dx in enumerate(dz_dxs):
        if le_closure:
            b1 = 1.0
            f = 1.0
            while abs(f) > epsilon:  # Newton's method
                pass
            bxm[ibin] = b1
        else:
            b1 = 1.0
            f = 1.0
            while abs(f) > epsilon:  # Newton's method
                bh = (1 + b1) / 2
                f1 = (1 + beta0) - beta0 * b1 - b1**2
                b2 = (1 + b1)**2
                # f2 = (b2 - 8 * beta0 / b2 + beta0 * (1 + b1)) * dz_dx**2
                f2 = (b2 - 2 * beta0 + beta0 * (1 + b1)) * dz_dx**2  # constant Ppara
                df1 = -beta0 - 2 * b1
                # df2 = (2*(1 + b1) + 16*beta0/(1+b1)**3 + beta0) * dz_dx**2
                df2 = (2*(1 + b1) + beta0) * dz_dx**2  # constant Ppara
                f = f1 - f2
                df = df1 - df2
                b1 -= f / df
            bxm[ibin] = b1
            bh = (1 + b1) / 2
    # firehose = 1 + 0.5 * beta0 * (bxm**-1 - bxm**-4)
    firehose = 1 + 0.5 * beta0 * (bxm**-1 - bxm**-2)  # constant Ppara
    # firehose = 1 + 0.25 * beta0 * (bxm**-1 - bxm**-2)  # constant Ppara and electron pressure
    # a = 5.0 * np.ones(nbins)
    # b = 2 * sbeta * np.ones(nbins)
    # c = 3 * bxm**2 * (dz_dxs**2 - firehose)
    # tmp = b**2 - 4*a*c
    # cond = np.logical_and(tmp > 0, c < 0)
    # vout = np.zeros(nbins)
    # vout[cond] = (-b[cond] + np.sqrt(tmp[cond])) / (2 * a[cond])
    for ibin, dz_dx in enumerate(dz_dxs):
        v1 = 1.0
        f = 1.0
        nloops = 0
        find_root = True
        while abs(f) > epsilon:  # Newton's method
            betav = math.sqrt(2/beta0) * v1
            # f1 = v1**2/2 + 0.5 * bxm[ibin]**2 * (dz_dx**2 - firehose[ibin]*0.6)
            f1 = 0.5 * v1**2 * (1 + erf(betav))
            f1 += 0.5 * bxm[ibin]**2 * (dz_dx**2 - firehose[ibin])
            ebetav = math.exp(-betav**2)
            # f2 = (v1**2 + (v1**2 + 3*beta0/4) * erf(betav) +
            #       v1 * sbeta_h * ebetav) / 3
            f2 = (v1**2 + (v1**2 + beta0/4) * erf(betav) +
                  v1 * sbeta_h * ebetav)
            f = f1 + f2
            tmp = 2 / math.sqrt(math.pi)
            isbeta = math.sqrt(2/beta0)
            df1 = v1 * (1 + erf(betav)) + 0.5 * v1**2 * tmp * ebetav * isbeta
            # df2 = (2*v1 + 2*v1*erf(betav) +
            #        (v1**2 + 3*beta0/4) * tmp * ebetav * isbeta +
            #        sbeta_h * ebetav * (1 - 4*v1**2/beta0)) / 3
            df2 = (2*v1 + 2*v1*erf(betav) +
                   (v1**2 + beta0/4) * tmp * ebetav * isbeta +
                   sbeta_h * ebetav * (1 - 4*v1**2/beta0))
            df = df1 + df2
            v1 -= f / df
            nloops += 1
            if nloops > 1000 or v1 < 0:
                find_root = False
                break
        if find_root:
            vout[ibin] = v1
            betav = math.sqrt(2/beta0) * v1
            nout[ibin] = 1 + erf(betav)

    # Save the data
    if le_closure:
        fdir = "../data/rate_problem/rate_model/le/"
    else:
        fdir = "../data/rate_problem/rate_model/cgl/"
    mkdir_p(fdir)
    fname = fdir + "dz_dx_" + str(beta0) + ".dat"
    dz_dxs.tofile(fname)
    fname = fdir + "bxm_" + str(beta0) + ".dat"
    bxm.tofile(fname)
    fname = fdir + "nout_" + str(beta0) + ".dat"
    nout.tofile(fname)
    fname = fdir + "vout_" + str(beta0) + ".dat"
    vout.tofile(fname)
    fname = fdir + "firehose_" + str(beta0) + ".dat"
    firehose.tofile(fname)

    # Plot quantities with different normalization
    nin = np.zeros(nbins)
    nin = 1.0
    vin = vout * nout * dz_dxs / nin
    rate = vin * bxm

    # Alfven speed using B and n upstream of the in diffusion region
    vam = bxm / np.sqrt(nin)
    vin_m = div0(vin, vam)
    vout_m = div0(vout, vam)
    rate_m = vin_m
    open_angle = np.arctan(dz_dxs) * 180 / math.pi

    if le_closure:
        fdir = "../img/rate_problem/rate_model/le/"
    else:
        fdir = "../img/rate_problem/rate_model/cgl/"
    mkdir_p(fdir)

    fig = plt.figure(figsize=[7, 5])
    rect = [0.12, 0.15, 0.8, 0.8]
    hgap, vgap = 0.06, 0.06
    ax = fig.add_axes(rect)
    p1, = ax.plot(open_angle, rate, label=r'$v_\text{in}B_{xm}/v_{A0}B_0$')
    ax.plot(open_angle, rate_m, linestyle='--', color=p1.get_color(),
            label=r'$v_\text{in}/v_{Am}$')
    ax.legend(loc=2, prop={'size': 16}, ncol=1,
              shadow=False, fancybox=True, frameon=True)
    ax.set_xlim([0, 45])
    ax.set_xlabel(r'Opening Angle', fontsize=16)
    ax.set_ylabel(r'$E_R$', fontsize=16)
    ax.grid(True)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(labelsize=12)
    beta_str = str(beta0).replace(".", "_")
    fname = fdir + 'rate_' + beta_str + '.pdf'
    fig.savefig(fname)

    fig = plt.figure(figsize=[7, 5])
    rect = [0.12, 0.15, 0.8, 0.8]
    hgap, vgap = 0.06, 0.06
    ax = fig.add_axes(rect)
    p1, = ax.plot(open_angle, vin, label=r'$v_\text{in}/v_{A0}$')
    ax.plot(open_angle, vin_m, linestyle='--', color=p1.get_color(),
            label=r'$v_\text{in}/v_{Am}$')
    p2, = ax.plot(open_angle, vout, label=r'$v_\text{out}/v_{A0}$')
    ax.plot(open_angle, vout_m, linestyle='--', color=p2.get_color(),
            label=r'$v_\text{out}/v_{Am}$')
    ax.legend(loc=6, prop={'size': 16}, ncol=1,
              shadow=False, fancybox=True, frameon=True)
    ax.set_xlim([0, 45])
    ax.set_xlabel(r'Opening Angle', fontsize=16)
    ax.grid(True)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(labelsize=12)
    beta_str = str(beta0).replace(".", "_")
    fname = fdir + 'vin_out_' + beta_str + '.pdf'
    fig.savefig(fname)

    fig = plt.figure(figsize=[7, 5])
    rect = [0.12, 0.15, 0.8, 0.8]
    hgap, vgap = 0.06, 0.06
    ax = fig.add_axes(rect)
    p1, = ax.plot(open_angle, nout/nin,
                  label=r'$n_\text{out}/n_\text{in}$')
    ax.set_xlim([0, 45])
    ax.set_xlabel(r'Opening Angle', fontsize=16)
    ax.set_xlabel(r'$n_\text{out}/n_\text{in}$', fontsize=16)
    ax.grid(True)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(labelsize=12)
    beta_str = str(beta0).replace(".", "_")
    fname = fdir + 'nout_in_' + beta_str + '.pdf'
    fig.savefig(fname)

    plt.close("all")
    # plt.show()


def plot_rate_model(le_closure=False):
    """Plot results calculated from reconnection rate model

    Args:
        le_closure: whether to use Le-Egedal closure
    """
    if le_closure:
        fdir = "../data/rate_problem/rate_model/le/"
    else:
        fdir = "../data/rate_problem/rate_model/cgl/"
    betas = [0.25, 1, 10, 40, 1E-3, 100]

    figs = []
    axs = []
    nvar = 4

    for i in range(nvar):
        fig = plt.figure(figsize=[7, 5])
        rect = [0.12, 0.15, 0.8, 0.8]
        hgap, vgap = 0.06, 0.06
        ax = fig.add_axes(rect)
        ax.set_xlim([0, 45])
        ax.set_xlabel(r'Opening Angle', fontsize=16)
        ax.grid(True)
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(labelsize=12)
        figs.append(fig)
        axs.append(ax)
    for beta0 in betas:
        fname = fdir + "dz_dx_" + str(beta0) + ".dat"
        dz_dxs = np.fromfile(fname)
        fname = fdir + "bxm_" + str(beta0) + ".dat"
        bxm = np.fromfile(fname)
        fname = fdir + "nout_" + str(beta0) + ".dat"
        nout = np.fromfile(fname)
        fname = fdir + "vout_" + str(beta0) + ".dat"
        vout = np.fromfile(fname)
        # fname = fdir + "firehose_" + str(beta0) + ".dat"
        # firehose = np.fromfile(fname)
        nin = bxm  # Assumption

        vin = vout * nout * dz_dxs / nin
        rate = vin * bxm
        # rate = bxm * dz_dxs * vout

        # Alfven speed using B and n upstream of the in diffusion region
        vam = bxm / np.sqrt(nin)
        vin_m = vin / vam
        vout_m = vout / vam
        rate_m = vin_m

        open_angle = np.arctan(dz_dxs) * 180 / math.pi
        label1 = r"$\beta=" + str(beta0) + "$"
        axs[0].plot(open_angle, rate, label=label1)
        axs[1].plot(open_angle, bxm, label=label1)
        axs[2].plot(open_angle, nout, label=label1)
        axs[3].plot(open_angle, vout, label=label1)

    axs[0].set_ylabel(r'$E_R$', fontsize=16)
    axs[1].set_ylabel(r'$B_{xm}/B_0$', fontsize=16)
    axs[2].set_ylabel(r'$n_\text{out}/n_0$', fontsize=16)
    axs[3].set_ylabel(r'$v_\text{out}/v_{A0}$', fontsize=16)

    if le_closure:
        fdir = "../img/rate_problem/rate_model/le/"
    else:
        fdir = "../img/rate_problem/rate_model/cgl/"
    mkdir_p(fdir)

    vnames = ['rate', 'bxm', 'nout', 'vout']
    for i in range(nvar):
        axs[i].legend(loc='best', prop={'size': 16}, ncol=1,
                      shadow=False, fancybox=True, frameon=True)
        fname = fdir + vnames[i] + '_beta.pdf'
        figs[i].savefig(fname)

    plt.show()


def calc_bxm_analytical(beta0, le_closure=False):
    """calculate bxm analytically

    Args:
        beta0: plasma beta in the inflow region
        le_closure: whether to use Le-Egedal closure
    """
    nbins = 100
    dz_dxs = np.linspace(0, 1, nbins)
    bxm = np.zeros(nbins)
    nout = np.zeros(nbins)
    epsilon = 1E-12
    sbeta = math.sqrt(2*beta0/math.pi)
    for ibin, dz_dx in enumerate(dz_dxs):
        b1 = 1.0
        f = 1.0
        while abs(f) > epsilon:  # Newton's method
            nm = (b1**2 + 10*b1 + 1) / (b1**2 + 4*b1 + 7)
            nh = (b1**2 + 7*b1 + 4) / (b1**2 + 4*b1 + 7)
            bh = (1 + b1) / 2
            f1 = 1 - b1**2 + beta0 * (1 - b1*nm)
            tmp1 = 1 - 0.5 * beta0 * (nh**3/bh**4 - nh/bh)
            f2 = tmp1 * dz_dx**2 * (1 + b1)**2
            f = f1 - f2
            dnm_dbxm = (-6*b1**2 + 12*b1 + 66) / (b1**2 + 4*b1 + 7)**2
            dnh_dbxm = 0.5 * dnm_dbxm
            dbh_dbxm = 0.5
            df1 = -2 * b1
            df2 = -beta0 * (nm + b1 * dnm_dbxm)
            df3 = tmp1 * dz_dx**2 * 2 * (1 + b1)
            tmp1 = (3*nh**2 * dnh_dbxm*bh - 4*nh**3*dbh_dbxm) / bh**5
            tmp2 = (dnh_dbxm*bh - nh*dbh_dbxm) / bh**2
            df4 = (-0.5 * beta0 * dz_dx**2 * (1 + b1)**2) * (tmp1 - tmp2)
            df = df1 + df2 - df3 - df4
            b1 -= f / df
        bxm[ibin] = b1
    open_angle = np.arctan(dz_dxs) * 180 / math.pi
    nm = (bxm**2 + 10*bxm + 1) / (bxm**2 + 4*bxm + 7)
    plt.plot(open_angle, nm)
    plt.grid(True)
    plt.show()


def peak_rate_beta(le_closure=False):
    """Peak reconnection rate changing with plasma beta

    Args:
        le_closure: whether to use Le-Egedal closure
    """
    nbins = 10000
    dz_dxs = np.linspace(0, 1, nbins)
    open_angle = np.arctan(dz_dxs) * 180 / math.pi
    bxm = np.zeros(nbins)
    nbeta = 70
    nfirehose = 11
    rate_peak = np.zeros((nfirehose, nbeta))
    bxm_peak = np.zeros((nfirehose, nbeta))
    vout_peak = np.zeros((nfirehose, nbeta))
    firehose_peak = np.zeros((nfirehose, nbeta))
    angle_peak = np.zeros((nfirehose, nbeta))
    epsilon = 1E-12
    betas = np.logspace(-4, math.log10(2000), nbeta)
    for ibeta, beta0 in enumerate(betas):
        print(ibeta, beta0)
        sbeta = math.sqrt(2*beta0/math.pi)
        sbeta_h = sbeta * 0.5
        for ibin, dz_dx in enumerate(dz_dxs):
            if le_closure:
                b1 = 1.0
                f = 1.0
                while abs(f) > epsilon:  # Newton's method
                    pass
                bxm[ibin] = b1
            else:
                b1 = 1.0
                f = 1.0
                while abs(f) > epsilon:  # Newton's method
                    bh = (1 + b1) / 2
                    f1 = (1 + beta0) - beta0 * b1 - b1**2
                    b2 = (1 + b1)**2
                    # f2 = (b2 - 8 * beta0 / b2 + beta0 * (1 + b1)) * dz_dx**2
                    f2 = (b2 - 2 * beta0 + beta0 * (1 + b1)) * dz_dx**2  # constant Ppara
                    df1 = -beta0 - 2 * b1
                    # df2 = (2*(1 + b1) + 16*beta0/(1+b1)**3 + beta0) * dz_dx**2
                    df2 = (2*(1 + b1) + beta0) * dz_dx**2  # constant Ppara
                    f = f1 - f2
                    df = df1 - df2
                    b1 -= f / df
                bxm[ibin] = b1
        # firehose = 1 + 0.5 * beta0 * (bxm**-1 - bxm**-4)
        firehose = 1 + 0.5 * beta0 * (bxm**-1 - bxm**-2)  # constant Ppara
        # firehose = 1 + 0.25 * beta0 * (bxm**-1 - bxm**-2)  # constant Ppara and electron pressure
        vout = np.zeros(nbins)
        # a = 5.0 * np.ones(nbins)
        # b = 2 * sbeta * np.ones(nbins)
        # c = 3 * bxm**2 * (dz_dxs**2 - firehose)
        # tmp = b**2 - 4*a*c
        # cond = np.logical_and(tmp > 0, c < 0)
        # vout[cond] = (-b[cond] + np.sqrt(tmp[cond])) / (2 * a[cond])
        for i in range(nfirehose-1, nfirehose):
            firehose_norm = 0.1 * i
            print("firehose_norm: %f" % firehose_norm)
            for ibin, dz_dx in enumerate(dz_dxs):
                v1 = 1.0
                f = 1.0
                nloops = 0
                find_root = True
                while abs(f) > epsilon:  # Newton's method
                    betav = math.sqrt(2/beta0) * v1
                    # f1 = v1**2/2 + 0.5 * bxm[ibin]**2 * (dz_dx**2 - firehose[ibin]*0.6)
                    f1 = 0.5 * v1**2 * (1 + erf(betav))
                    f1 += 0.5 * bxm[ibin]**2 * (dz_dx**2 - firehose[ibin])
                    ebetav = math.exp(-betav**2)
                    # f2 = (v1**2 + (v1**2 + 3*beta0/4) * erf(betav) +
                    #       v1 * sbeta_h * ebetav) / 3
                    f2 = (v1**2 + (v1**2 + beta0/4) * erf(betav) +
                          v1 * sbeta_h * ebetav)
                    f = f1 + f2
                    tmp = 2 / math.sqrt(math.pi)
                    isbeta = math.sqrt(2/beta0)
                    df1 = v1 * (1 + erf(betav)) + 0.5 * v1**2 * tmp * ebetav * isbeta
                    # df2 = (2*v1 + 2*v1*erf(betav) +
                    #        (v1**2 + 3*beta0/4) * tmp * ebetav * isbeta +
                    #        sbeta_h * ebetav * (1 - 4*v1**2/beta0)) / 3
                    df2 = (2*v1 + 2*v1*erf(betav) +
                           (v1**2 + beta0/4) * tmp * ebetav * isbeta +
                           sbeta_h * ebetav * (1 - 4*v1**2/beta0))
                    df = df1 + df2
                    v1 -= f / df
                    nloops += 1
                    if nloops > 1000 or v1 < 0:
                        find_root = False
                        break
                if find_root:
                    vout[ibin] = v1
            # rate = vout * bxm * dz_dxs
            betav = math.sqrt(2/beta0) * vout
            rate = vout * bxm * dz_dxs * (1 + erf(betav))
            ipeak = np.argmax(rate)
            bxm_peak[i,ibeta] = bxm[ipeak]
            vout_peak[i,ibeta] = vout[ipeak]
            firehose_peak[i,ibeta] = firehose[ipeak]
            angle_peak[i,ibeta] = open_angle[ipeak]
            rate_peak[i,ibeta] = rate[ipeak]

    for i in range(nfirehose):
        # Save the data
        if le_closure:
            fdir = "../data/rate_problem/rate_model/le/"
        else:
            fdir = "../data/rate_problem/rate_model/cgl/"
        firehose_norm = 0.1 * i
        fdir += "firehose_norm_" + str(int(firehose_norm*10)) + "/"
        mkdir_p(fdir)
        fname = fdir + "betas.dat"
        betas.tofile(fname)
        fname = fdir + "rate_peak_beta.dat"
        rate_peak[i].tofile(fname)
        fname = fdir + "bxm_peak_beta.dat"
        bxm_peak[i].tofile(fname)
        fname = fdir + "vout_peak_beta.dat"
        vout_peak[i].tofile(fname)
        fname = fdir + "firehose_peak_beta.dat"
        firehose_peak[i].tofile(fname)
        fname = fdir + "angle_peak_beta.dat"
        angle_peak[i].tofile(fname)


def plot_peak_rate_beta(le_closure=False):
    """Plot peak reconnection rate changing with plasma beta

    Args:
        le_closure: whether to use Le-Egedal closure
    """
    nfirehose = 11
    if le_closure:
        input_dir = "../data/rate_problem/rate_model/le/"
    else:
        input_dir = "../data/rate_problem/rate_model/cgl/"
    fig = plt.figure(figsize=[7, 5])
    rect = [0.13, 0.15, 0.8, 0.8]
    hgap, vgap = 0.06, 0.06
    ax = fig.add_axes(rect)
    for i in [6, 8, 10]:
        firehose_norm = 0.1 * i
        print("firehose_norm: %f" % firehose_norm)
        fdir = input_dir + "firehose_norm_" + str(int(firehose_norm*10)) + "/"
        fname = fdir + "betas.dat"
        betas = np.fromfile(fname)
        fname = fdir + "rate_peak_beta.dat"
        rate = np.fromfile(fname)
        ax.semilogx(betas, rate, color='k', alpha=0.5)

    beta_s = [0.25, 1, 10, 40]
    rate_s = [0.11, 0.08, 0.05, 0.04]

    COLORS = palettable.tableau.Tableau_10.mpl_colors
    ax.scatter(beta_s, rate_s, color=COLORS[:4], s=100)
    norms = [10, 5, 7, 7]
    for i, beta in enumerate(beta_s):
        txt = r"$\beta=" + str(beta) + "$"
        ax.annotate(txt, (beta/norms[i], rate_s[i]-0.003),
                    color=COLORS[i], fontsize=16)
    ax.text(0.02, 0.93, r"$\epsilon_2=\epsilon(B_{xm})$", color="k", fontsize=12,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes)
    ax.text(0.02, 0.82, r"$\epsilon_2=0.8\epsilon(B_{xm})$", color="k", fontsize=12,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes)
    ax.text(0.02, 0.69, r"$\epsilon_2=0.6\epsilon(B_{xm})$", color="k", fontsize=12,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='top',
            transform=ax.transAxes)
    ax.set_xlabel(r'$\beta$', fontsize=16)
    ax.set_ylabel(r'$E_R$', fontsize=16)
    ax.set_xlim([1E-3, betas.max()])
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(labelsize=12)
    img_dir = '../img/rate_problem/pub/'
    mkdir_p(img_dir)
    fname = img_dir + "rate_beta.pdf"
    fig.savefig(fname)
    plt.show()


def plot_rate_model_pub(le_closure=False):
    """Plot results calculated from reconnection rate model for paper

    Args:
        le_closure: whether to use Le-Egedal closure
    """
    if le_closure:
        fdir = "../data/rate_problem/rate_model/le/"
    else:
        fdir = "../data/rate_problem/rate_model/cgl/"
    betas = [0.25, 1, 10, 40, 1E-3, 100]

    fig = plt.figure(figsize=[7, 10])
    rect0 = [0.13, 0.69, 0.8, 0.28]
    hgap, vgap = 0.02, 0.03

    axs = []
    nvar = 3

    COLORS = palettable.tableau.Tableau_10.mpl_colors
    for i in range(nvar):
        rect = np.copy(rect0)
        rect[1] = rect0[1] - i * (rect[3] + vgap)
        ax = fig.add_axes(rect)
        ax.set_prop_cycle('color', COLORS)
        ax.set_xlim([0, 45])
        if i == nvar - 1:
            ax.set_xlabel('Opening Angle', fontsize=16)
        else:
            ax.tick_params(axis='x', labelbottom=False)
        ax.grid(True)
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(labelsize=12)
        axs.append(ax)
    for beta0 in betas:
        fname = fdir + "dz_dx_" + str(beta0) + ".dat"
        dz_dxs = np.fromfile(fname)
        fname = fdir + "bxm_" + str(beta0) + ".dat"
        bxm = np.fromfile(fname)
        fname = fdir + "vout_" + str(beta0) + ".dat"
        vout = np.fromfile(fname)
        nin = np.ones(bxm.shape)  # incompressible
        nout = np.ones(bxm.shape)  # incompressible

        vin = vout * nout * dz_dxs / nin
        rate = vin * bxm
        rate = bxm * dz_dxs * vout

        # Alfven speed using B and n upstream of the in diffusion region
        vam = bxm / np.sqrt(nin)
        vin_m = vin / vam
        vout_m = vout / vam
        rate_m = vin_m

        open_angle = np.arctan(dz_dxs) * 180 / math.pi
        label1 = r"$\beta=" + str(beta0) + "$"
        axs[0].plot(open_angle, vout, label=label1)
        axs[1].plot(open_angle, bxm, label=label1)
        axs[2].plot(open_angle, rate, label=label1)

    # pic_runs = ["mime400_Tb_T0_025",
    #             "mime400_Tb_T0_1",
    #             "mime400_Tb_T0_10_weak",
    #             "mime400_Tb_T0_40_nppc450"]
    # for irun, pic_run in enumerate(pic_runs):
    #     picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    #     pic_info = read_data_from_json(picinfo_fname)
    #     pic_run_dir = pic_info.run_dir
    #     vpic_info = get_vpic_info(pic_run_dir)
    #     nb_n0 = vpic_info["nb/n0"]
    #     dtwpe = pic_info.dtwpe
    #     dtwce = pic_info.dtwce
    #     va = dtwce * math.sqrt(1.0 / pic_info.mime) / dtwpe / math.sqrt(nb_n0)
    #     wpe_wce = dtwpe / dtwce
    #     b0 = pic_info.b0
    #     fields_interval = pic_info.fields_interval
    #     dtf = pic_info.dtwpe * fields_interval
    #     ntf = pic_info.ntf
    #     tfields = np.arange(ntf) * dtf
    #     tfields_wci = np.arange(ntf) * pic_info.dtwci * fields_interval
    #     bflux = np.zeros(ntf)
    #     for tframe in range(ntf):
    #         fdir = '../data/rate_problem/rrate_bflux/' + pic_run + '/'
    #         fname = fdir + 'rrate_bflux_' + str(tframe) + '.dat'
    #         fdata = np.fromfile(fname)
    #         bflux[tframe] = fdata[0]
    #     fdir = '../data/rate_problem/open_angle/' + pic_run + '/'
    #     nframes = len(os.listdir(fdir))
    #     open_angle = np.zeros(nframes)
    #     for tframe in range(nframes):
    #         fname = fdir + "open_angle_" + str(tframe) + ".dat"
    #         fdata = np.fromfile(fname)
    #         open_angle[tframe] = fdata[0]

    #     rrate_bflux = -np.gradient(bflux) / dtf
    #     rrate_bflux /= va * b0
    #     ax.scatter(open_angle, rrate_bflux, color=COLORS[irun])

    axs[1].legend(loc=3, prop={'size': 16}, ncol=1,
                  shadow=False, fancybox=True, frameon=True)

    axs[0].set_ylabel(r'$v_\text{out}/v_{A0}$', fontsize=16)
    axs[1].set_ylabel(r'$B_{xm}/B_0$', fontsize=16)
    axs[2].set_ylabel(r'$E_R$', fontsize=16)

    axs[0].text(-0.12, 0.95, "(a)", color="k", fontsize=20,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='center', verticalalignment='center',
                transform=axs[0].transAxes)
    axs[1].text(-0.12, 0.95, "(b)", color="k", fontsize=20,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='center', verticalalignment='center',
                transform=axs[1].transAxes)
    axs[2].text(-0.12, 0.95, "(c)", color="k", fontsize=20,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='center', verticalalignment='center',
                transform=axs[2].transAxes)

    img_dir = '../img/rate_problem/pub/'
    mkdir_p(img_dir)
    fname = img_dir + "model.pdf"
    fig.savefig(fname)

    plt.show()


def plot_rate_model_pub2(le_closure=False):
    """Plot results calculated from reconnection rate model for paper

    Args:
        le_closure: whether to use Le-Egedal closure
    """
    if le_closure:
        input_dir = "../data/rate_problem/rate_model/le/"
    else:
        input_dir = "../data/rate_problem/rate_model/cgl/"
    betas = [1E-3, 0.25, 1, 10, 40, 100]

    fig = plt.figure(figsize=[7, 6])
    rect0 = [0.11, 0.55, 0.37, 0.35]
    hgap, vgap = 0.12, 0.11

    axs = []
    nvar = 4

    COLORS = palettable.tableau.Tableau_10.mpl_colors
    colors2 = np.copy(COLORS)
    colors2[0] = COLORS[4]
    colors2[1:5] = COLORS[:4]
    for i in range(nvar):
        rect = np.copy(rect0)
        row = i // 2
        col = i % 2
        rect[0] = rect0[0] + col * (rect[2] + hgap)
        rect[1] = rect0[1] - row * (rect[3] + vgap)
        ax = fig.add_axes(rect)
        ax.set_prop_cycle('color', colors2)
        ax.set_xlim([0, 90])
        if i == nvar - 1:
            ax.set_xlabel(r'$\beta$', fontsize=16)
        else:
            ax.set_xlabel('Opening Angle ($^\circ$)', fontsize=16)
        ax.grid(True)
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')
        ax.tick_params(labelsize=12)
        axs.append(ax)
    for beta0 in betas:
        fname = input_dir + "dz_dx_" + str(beta0) + ".dat"
        dz_dxs = np.fromfile(fname)
        fname = input_dir + "bxm_" + str(beta0) + ".dat"
        bxm = np.fromfile(fname)
        fname = input_dir + "vout_" + str(beta0) + ".dat"
        vout = np.fromfile(fname)
        fname = input_dir + "nout_" + str(beta0) + ".dat"
        nout = np.fromfile(fname)
        nin = np.ones(bxm.shape)  # incompressible
        # nout = np.ones(bxm.shape)  # incompressible

        vin = vout * nout * dz_dxs / nin
        rate = vin * bxm
        # rate = bxm * dz_dxs * vout

        # Alfven speed using B and n upstream of the in diffusion region
        vam = bxm / np.sqrt(nin)
        vin_m = vin / vam
        vout_m = vout / vam
        rate_m = vin_m

        open_angle = np.arctan(dz_dxs) * 180 * 2 / math.pi
        label1 = r"$\beta_0=" + str(beta0) + "$"
        axs[0].plot(open_angle, bxm, label=label1, linewidth=1)
        axs[1].plot(open_angle, vout, label=label1, linewidth=1)
        axs[2].plot(open_angle, rate, label=label1, linewidth=1)

    xpos = 1.0 + 0.5 * hgap / rect0[2]
    axs[0].legend(loc=9, bbox_to_anchor=(xpos, 1.3),
                  prop={'size': 12}, ncol=3,
                  shadow=False, fancybox=False, frameon=False)

    axs[0].set_ylabel(r'$B_{xm}/B_{x0}$', fontsize=16)
    axs[1].set_ylabel(r'$v_\text{out}/v_{A0}$', fontsize=16)
    axs[2].set_ylabel(r'$E_R$', fontsize=16)

    # Peak rate
    # for i in [4, 6, 8, 10]:
    for i in [10]:
        firehose_norm = 0.1 * i
        print("firehose_norm: %f" % firehose_norm)
        fdir = input_dir + "firehose_norm_" + str(int(firehose_norm*10)) + "/"
        fname = fdir + "betas.dat"
        betas = np.fromfile(fname)
        fname = fdir + "rate_peak_beta.dat"
        rate = np.fromfile(fname)
        if i == 6:
            ax.semilogx(betas, rate, color='k', linewidth=1)
        else:
            # ax.semilogx(betas, rate, color='k', alpha=0.5, linestyle='--')
            ax.semilogx(betas, rate, color='k', linewidth=1)

    beta_s = [0.25, 1, 10, 40]
    # rate_s = [0.11, 0.08, 0.05, 0.04]
    rate_s = [0.11, 0.08, 0.045, 0.038]

    COLORS = palettable.tableau.Tableau_10.mpl_colors
    axs[3].scatter(beta_s, rate_s, color=COLORS[:4], s=50)
    norms = [0.02, 0.05, 1.5, 1.5]
    for i, beta in enumerate(beta_s):
        txt = r"$\beta=" + str(beta) + "$"
        if i == 2:
            ypos = rate_s[i] + 0.005
        else:
            ypos = rate_s[i]
        axs[3].annotate(txt, (beta*norms[i], ypos),
                        color=COLORS[i], fontsize=12)
    # axs[3].text(0.02, 0.93, r"$\epsilon_2=\epsilon(B_{xm})$", color="k", fontsize=16,
    #             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
    #             horizontalalignment='left', verticalalignment='top',
    #             transform=axs[3].transAxes)
    # axs[3].text(0.02, 0.82, r"$\epsilon_2=0.8\epsilon(B_{xm})$", color="k", fontsize=16,
    #             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
    #             horizontalalignment='left', verticalalignment='top',
    #             transform=axs[3].transAxes)
    # axs[3].text(0.02, 0.69, r"$\epsilon_2=0.6\epsilon(B_{xm})$", color="k", fontsize=16,
    #             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
    #             horizontalalignment='left', verticalalignment='top',
    #             transform=axs[3].transAxes)
    # axs[3].text(0.02, 0.53, r"$\epsilon_2=0.4\epsilon(B_{xm})$", color="k", fontsize=16,
    #             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
    #             horizontalalignment='left', verticalalignment='top',
    #             transform=axs[3].transAxes)
    axs[3].set_xlabel(r'$\beta$', fontsize=16)
    axs[3].set_ylabel(r'Peak $E_R$', fontsize=16)
    axs[3].set_xlim([1E-3, betas.max()])
    axs[3].tick_params(bottom=True, top=True, left=True, right=True)
    axs[3].tick_params(labelsize=12)

    xpos, ypos = -0.2, 0.9
    axs[0].text(xpos, ypos, "(a)", color="k", fontsize=20,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='center', verticalalignment='center',
                transform=axs[0].transAxes)
    axs[1].text(xpos, ypos, "(b)", color="k", fontsize=20,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='center', verticalalignment='center',
                transform=axs[1].transAxes)
    axs[2].text(xpos, ypos, "(c)", color="k", fontsize=20,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='center', verticalalignment='center',
                transform=axs[2].transAxes)
    axs[3].text(xpos, ypos, "(d)", color="k", fontsize=20,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='center', verticalalignment='center',
                transform=axs[3].transAxes)

    img_dir = '../img/rate_problem/pub/'
    mkdir_p(img_dir)
    fname = img_dir + "model2.pdf"
    fig.savefig(fname)

    plt.show()


def plot_rate_model_pub3(le_closure=False):
    """Plot results calculated from reconnection rate model for paper

    Args:
        le_closure: whether to use Le-Egedal closure
    """
    if le_closure:
        input_dir = "../data/rate_problem/rate_model/le/"
    else:
        input_dir = "../data/rate_problem/rate_model/cgl/"
    betas = [1E-3, 0.25, 1, 10, 40, 100]

    fig = plt.figure(figsize=[7, 6])
    rect0 = [0.11, 0.55, 0.37, 0.35]
    hgap, vgap = 0.12, 0.11

    axs = []
    nvar = 4

    COLORS = palettable.tableau.Tableau_10.mpl_colors
    colors2 = np.copy(COLORS)
    colors2[0] = COLORS[4]
    colors2[1:5] = COLORS[:4]
    colors2[5] = COLORS[7]
    print(np.asarray(COLORS[0])*256)
    for i in range(nvar):
        rect = np.copy(rect0)
        row = i // 2
        col = i % 2
        rect[0] = rect0[0] + col * (rect[2] + hgap)
        rect[1] = rect0[1] - row * (rect[3] + vgap)
        ax = fig.add_axes(rect)
        ax.set_prop_cycle('color', colors2)
        ax.set_xlim([0, 90])
        if i == nvar - 1:
            ax.set_xlabel(r'$\beta$', fontsize=16)
        else:
            ax.set_xlabel('Opening Angle ($^\circ$)', fontsize=16)
        ax.grid(True)
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')
        ax.tick_params(labelsize=12)
        axs.append(ax)
    for beta0 in betas:
        fname = input_dir + "dz_dx_" + str(beta0) + ".dat"
        dz_dxs = np.fromfile(fname)
        fname = input_dir + "bxm_" + str(beta0) + ".dat"
        bxm = np.fromfile(fname)
        fname = input_dir + "vout_" + str(beta0) + ".dat"
        vout = np.fromfile(fname)
        fname = input_dir + "nout_" + str(beta0) + ".dat"
        nout = np.fromfile(fname)
        nin = np.ones(bxm.shape)  # incompressible
        # nout = np.ones(bxm.shape)  # incompressible

        vin = vout * nout * dz_dxs / nin
        rate = vin * bxm
        # rate = bxm * dz_dxs * vout

        # Alfven speed using B and n upstream of the in diffusion region
        vam = bxm / np.sqrt(nin)
        vin_m = vin / vam
        vout_m = vout / vam
        rate_m = vin_m

        open_angle = np.arctan(dz_dxs) * 180 * 2 / math.pi
        label1 = r"$\beta_0=" + str(beta0) + "$"
        axs[0].plot(open_angle, bxm, label=label1, linewidth=1)
        axs[1].plot(open_angle, vout, label=label1, linewidth=1)
        axs[2].plot(open_angle, rate, label=label1, linewidth=1)

    pic_runs = ["mime400_Tb_T0_025",
                "mime400_Tb_T0_1",
                "mime400_Tb_T0_10_weak",
                "mime400_Tb_T0_40_nppc450_old"]
    bxm_mean = []
    vout_mean = []
    rate_mean = []
    angle_mean = []
    bxm_err_min = []
    bxm_err_max = []
    bxm_err_max = []
    vout_err_min = []
    vout_err_max = []
    rate_err_min = []
    rate_err_max = []
    angle_err_min = []
    angle_err_max = []
    for irun, pic_run in enumerate(pic_runs):
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        pic_run_dir = pic_info.run_dir
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
        fdir = '../data/rate_problem/open_angle/' + pic_run + '/'
        fdir_bxm = '../data/rate_problem/bxm/' + pic_run + '/'
        fdir_vout = '../data/rate_problem/vout_peak/' + pic_run + '/'
        nframes = len(os.listdir(fdir))
        open_angle = np.zeros(nframes)
        bxm = np.zeros(nframes)
        vout_peak = np.zeros(nframes)
        for tframe in range(nframes):
            fname = fdir + "open_angle_" + str(tframe) + ".dat"
            fdata = np.fromfile(fname)
            open_angle[tframe] = fdata[0]
            fname = fdir_bxm + 'bxm_' + str(tframe) + '.dat'
            fdata = np.fromfile(fname)
            bxm[tframe] = fdata[0]
            fname = fdir_vout + "vout_peak_" + str(tframe) + ".dat"
            fdata = np.fromfile(fname)
            vout_peak[tframe] = fdata[0]

        rrate_bflux = -np.gradient(bflux) / dtf
        rrate_bflux /= va * b0
        tpeak = np.argmax(rrate_bflux)
        tmax = min(47, nframes)
        # Angle
        dmean = np.mean(open_angle[tpeak:tmax]*2)
        angle_mean.append(dmean)
        angle_err_min.append(dmean - np.min(open_angle[tpeak:tmax]*2))
        angle_err_max.append(np.max(open_angle[tpeak:tmax]*2) - dmean)
        # Bxm
        dmean = np.mean(bxm[tpeak:tmax])
        bxm_mean.append(dmean / b0)
        bxm_err_min.append(dmean/b0 - np.min(bxm[tpeak:tmax]) / b0)
        bxm_err_max.append(np.max(bxm[tpeak:tmax]) / b0 - dmean/b0)
        # Vout
        dmean = np.mean(vout_peak[tpeak:tmax])
        vout_mean.append(dmean)
        vout_err_min.append(dmean - np.min(vout_peak[tpeak:tmax]))
        vout_err_max.append(np.max(vout_peak[tpeak:tmax]) - dmean)
        # Rate
        dmean = np.mean(rrate_bflux[tpeak:tmax])
        rate_mean.append(dmean)
        rate_err_min.append(dmean - np.min(rrate_bflux[tpeak:tmax]))
        rate_err_max.append(np.max(rrate_bflux[tpeak:tmax]) - dmean)
    for irun, pic_run in enumerate(pic_runs):
        axs[0].errorbar([angle_mean[irun]], [bxm_mean[irun]],
                        [[bxm_err_min[irun]], [bxm_err_max[irun]]],
                        [[angle_err_min[irun]], [angle_err_max[irun]]],
                        color=COLORS[irun], lw=1, capsize=2, capthick=1,
                        marker='o')
        axs[1].errorbar([angle_mean[irun]], [vout_mean[irun]],
                        [[vout_err_min[irun]], [vout_err_max[irun]]],
                        [[angle_err_min[irun]], [angle_err_max[irun]]],
                        color=COLORS[irun], lw=1, capsize=2, capthick=1,
                        marker='o')
        axs[2].errorbar([angle_mean[irun]], [rate_mean[irun]],
                        [[rate_err_min[irun]], [rate_err_max[irun]]],
                        [[angle_err_min[irun]], [angle_err_max[irun]]],
                        color=COLORS[irun], lw=1, capsize=2, capthick=1,
                        marker='o')

    xpos = 1.0 + 0.5 * hgap / rect0[2]
    axs[0].legend(loc=9, bbox_to_anchor=(xpos, 1.3),
                  prop={'size': 12}, ncol=3,
                  shadow=False, fancybox=False, frameon=False)

    axs[0].set_ylabel(r'$B_{xm}/B_{x0}$', fontsize=16)
    axs[1].set_ylabel(r'$v_\text{out}/v_{A0}$', fontsize=16)
    axs[2].set_ylabel(r'$E_R$', fontsize=16)

    # Peak rate
    # for i in [4, 6, 8, 10]:
    for i in [10]:
        firehose_norm = 0.1 * i
        print("firehose_norm: %f" % firehose_norm)
        fdir = input_dir + "firehose_norm_" + str(int(firehose_norm*10)) + "/"
        fname = fdir + "betas.dat"
        betas = np.fromfile(fname)
        fname = fdir + "rate_peak_beta.dat"
        rate = np.fromfile(fname)
        if i == 6:
            axs[3].semilogx(betas/2, rate, color='k', linewidth=1,
                            label=r"Peak $E_R$ (model)")
        else:
            # ax.semilogx(betas, rate, color='k', alpha=0.5, linestyle='--')
            axs[3].semilogx(betas/2, rate, color='k', linewidth=1,
                            label=r"Peak $E_R$ (model)")
    # rate_scaling = 0.08 / betas**0.25
    rate_scaling = 0.1 / (betas/2)**0.5
    axs[3].semilogx(betas/2, rate_scaling, color='k', linewidth=1,
                    linestyle="--", label=r"$0.1/\sqrt{\beta_{i0}}$")
    ax.legend(loc=1, prop={'size': 12}, ncol=1,
              shadow=False, fancybox=True, frameon=True)

    beta_s = [0.25, 1, 10, 40]
    # rate_s = [0.11, 0.08, 0.05, 0.04]
    rate_s = [0.11, 0.08, 0.045, 0.038]

    COLORS = palettable.tableau.Tableau_10.mpl_colors
    # axs[3].scatter(beta_s, rate_s, color=COLORS[:4], s=50)
    norms = [0.02, 0.05, 1.5, 1.5]
    # for i, beta in enumerate(beta_s):
    #     txt = r"$\beta=" + str(beta) + "$"
    #     if i == 2:
    #         ypos = rate_s[i] + 0.005
    #     else:
    #         ypos = rate_s[i]
    #     axs[3].annotate(txt, (beta*norms[i], ypos),
    #                     color=COLORS[i], fontsize=12)
    for irun, pic_run in enumerate(pic_runs):
        axs[3].errorbar([beta_s[irun]/2], [rate_mean[irun]],
                        [[rate_err_min[irun]], [rate_err_max[irun]]],
                        color=COLORS[irun], lw=1, capsize=2, capthick=1,
                        marker='o')
    # axs[3].text(0.35, 0.97, r"Peak $E_R$ (model)", color="k", fontsize=16,
    #             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
    #             horizontalalignment='left', verticalalignment='top',
    #             transform=axs[3].transAxes)
    # axs[3].text(0.02, 0.93, r"$\epsilon_2=\epsilon(B_{xm})$", color="k", fontsize=16,
    #             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
    #             horizontalalignment='left', verticalalignment='top',
    #             transform=axs[3].transAxes)
    # axs[3].text(0.02, 0.82, r"$\epsilon_2=0.8\epsilon(B_{xm})$", color="k", fontsize=16,
    #             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
    #             horizontalalignment='left', verticalalignment='top',
    #             transform=axs[3].transAxes)
    # axs[3].text(0.02, 0.69, r"$\epsilon_2=0.6\epsilon(B_{xm})$", color="k", fontsize=16,
    #             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
    #             horizontalalignment='left', verticalalignment='top',
    #             transform=axs[3].transAxes)
    # axs[3].text(0.02, 0.53, r"$\epsilon_2=0.4\epsilon(B_{xm})$", color="k", fontsize=16,
    #             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
    #             horizontalalignment='left', verticalalignment='top',
    #             transform=axs[3].transAxes)
    axs[3].set_xlabel(r'$\beta_{i0}$', fontsize=16)
    axs[3].set_ylabel(r'$E_R$', fontsize=16)
    axs[3].set_xlim([1E-3, 1E3])
    axs[3].set_ylim([-0.01, 0.18])
    axs[3].tick_params(bottom=True, top=True, left=True, right=True)
    axs[3].tick_params(labelsize=12)

    xpos, ypos = -0.2, 0.9
    axs[0].text(xpos, ypos, "(a)", color="k", fontsize=20,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='center', verticalalignment='center',
                transform=axs[0].transAxes)
    axs[1].text(xpos, ypos, "(b)", color="k", fontsize=20,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='center', verticalalignment='center',
                transform=axs[1].transAxes)
    axs[2].text(xpos, ypos, "(c)", color="k", fontsize=20,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='center', verticalalignment='center',
                transform=axs[2].transAxes)
    axs[3].text(xpos, ypos, "(d)", color="k", fontsize=20,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='center', verticalalignment='center',
                transform=axs[3].transAxes)

    img_dir = '../img/rate_problem/pub/'
    mkdir_p(img_dir)
    fname = img_dir + "model3.pdf"
    fig.savefig(fname)

    plt.show()


def outflow_heating_fermi(plot_config, show_plot=True):
    """Heating in outflow due to Fermi mechanism
    """
    pic_run = plot_config["pic_run"]
    tframe = plot_config["tframe"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_run_dir = pic_info.run_dir
    fields_interval = pic_info.fields_interval
    tindex = fields_interval * tframe
    mime = pic_info.mime
    smime = math.sqrt(mime)
    lx_de = pic_info.lx_di * smime
    lz_de = pic_info.lz_di * smime
    nx = pic_info.nx
    nz = pic_info.nz
    xgrid = np.linspace(0, lx_de, nx)
    zgrid = np.linspace(-0.5*lz_de, lz_de, nz)
    xs = nx//2 - nx//4
    xe = nx//2 + nx//4
    zs = nz//2 - nz//16
    ze = nz//2 + nz//16
    nxs = xe - xs
    nzs = ze - zs
    dx_de = lx_de / nx
    dz_de = lz_de / nz
    xmin = xs * dz_de
    xmax = xe * dz_de
    zmin = zs * dz_de - lz_de * 0.5
    zmax = ze * dz_de - lz_de * 0.5

    vpic_info = get_vpic_info(pic_run_dir)
    n0 = vpic_info["n0"]
    b0 = vpic_info["b0"]
    Te = vpic_info["Te"]
    Tbe_Te = vpic_info["Tbe/Te"]
    nb_n0 = vpic_info["nb/n0"]
    nb = n0 * nb_n0
    p0 = nb * Te * Tbe_Te
    beta0 = 4 * p0 / b0**2

    plot_config["pic_run"] = pic_run
    plot_config["pic_run_dir"] = pic_run_dir
    bvec_pre = get_bfield_pressure(plot_config, box=[xs, zs, xe, ze])
    sigma = 5
    bvec_pre["pepara"] = gaussian_filter(bvec_pre["pepara"], sigma=sigma)
    bvec_pre["pipara"] = gaussian_filter(bvec_pre["pipara"], sigma=sigma)
    bvec_pre["peperp"] = gaussian_filter(bvec_pre["peperp"], sigma=sigma)
    bvec_pre["piperp"] = gaussian_filter(bvec_pre["piperp"], sigma=sigma)
    bvec_pre["ne"] = gaussian_filter(bvec_pre["ne"], sigma=sigma)
    bvec_pre["ni"] = gaussian_filter(bvec_pre["ni"], sigma=sigma)

    # Outflow velocity
    rho_vel = {}
    for species in ["e", "i"]:
        sname = "electron" if species == 'e' else "ion"
        fname = (pic_run_dir + "hydro_hdf5/T." + str(tindex) +
                 "/hydro_" + sname + "_" + str(tindex) + ".h5")
        hydro = {}
        with h5py.File(fname, 'r') as fh:
            group = fh["Timestep_" + str(tindex)]
            for var in ["rho", "jx"]:
                dset = group[var]
                hydro[var]= dset[xs:xe, 0, zs:ze]

        irho = 1.0 / hydro["rho"]
        vx = hydro["jx"] * irho
        var = "v" + species
        rho_vel[var+"x"] = np.squeeze(vx)
        rho_vel["n"+species] = np.squeeze(np.abs(hydro["rho"]))

    irho = 1.0 / (rho_vel["ne"] + rho_vel["ni"] * mime)
    vsx = (rho_vel["ne"] * rho_vel["vex"] +
           rho_vel["ni"] * rho_vel["vix"] * mime) * irho
    vsx = gaussian_filter(vsx, sigma=sigma)

    dtwpe = pic_info.dtwpe
    dtwce = pic_info.dtwce
    va = dtwce * math.sqrt(1.0 / mime) / dtwpe / math.sqrt(nb_n0)

    vsx_cut = np.abs(vsx[:, nzs//2] / va)
    tmp = math.sqrt(2/beta0) * vsx_cut
    sbeta_h = math.sqrt(0.5*beta0/math.pi)
    du_fermi = 0.5 * (vsx_cut**2 +
                      (vsx_cut**2 + 3*beta0/4) * erf(tmp) +
                      vsx_cut * sbeta_h * np.exp(-tmp**2))
    dp_fermi = du_fermi / 3
    dp_norm = nb * mime * va**2
    dp_fermi *= dp_norm
    dp_fermi /= p0

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
    ix_xp = int(x_xp / dx_de) - xs

    # Averaged pressure in the exhaust
    nvar = 4
    pres_avg = np.zeros([nvar, nxs])
    navg = np.zeros(nxs)
    bavg = np.zeros(nxs)
    absB = np.sqrt(bvec_pre["bx"]**2 + bvec_pre["by"]**2 + bvec_pre["bz"]**2)
    for ix in range(nxs):
        ixs = ix + xs
        iz1 = iz_bot[ixs] - zs
        iz2 = iz_top[ixs] - zs
        pres_avg[0, ix] = np.mean(bvec_pre["pepara"][ix, iz1:iz2+1])
        pres_avg[1, ix] = np.mean(bvec_pre["peperp"][ix, iz1:iz2+1])
        pres_avg[2, ix] = np.mean(bvec_pre["pipara"][ix, iz1:iz2+1])
        pres_avg[3, ix] = np.mean(bvec_pre["piperp"][ix, iz1:iz2+1])
        navg[ix] = np.mean(bvec_pre["ni"][ix, iz1:iz2+1])
        bavg[ix] = np.mean(absB[ix, iz1:iz2+1])

    # Internal energy
    eint_e = (pres_avg[0] + 2 * pres_avg[1]) / 2
    eint_i = (pres_avg[2] + 2 * pres_avg[3]) / 2

    pres_cut = np.zeros([nvar, nxs])
    pres_cut[0, :] = bvec_pre["pepara"][:, nzs//2]
    pres_cut[1, :] = bvec_pre["peperp"][:, nzs//2]
    pres_cut[2, :] = bvec_pre["pipara"][:, nzs//2]
    pres_cut[3, :] = bvec_pre["piperp"][:, nzs//2]

    fig = plt.figure(figsize=[7, 5])
    rect = [0.13, 0.13, 0.75, 0.8]
    hgap, vgap = 0.02, 0.04

    ax = fig.add_axes(rect)
    ax.set_prop_cycle('color', COLORS)
    xgrid = np.arange(xs, xe) * dx_de
    p1, = ax.plot(xgrid, pres_avg[3]/p0 - pres_avg[3].min()/p0,
                  label=r'$\Delta P_{i\perp}$')
    p1, = ax.plot(xgrid, pres_cut[3]/p0 - pres_avg[3, ix_xp]/p0,
                  label=r'$\Delta P_{i\perp}$')
    p2, = ax.plot(xgrid, dp_fermi, label='Fermi Heating')
    # p1, = ax.plot(xgrid, eint_i/p0 - eint_i.min()/p0,
    #               label=r'$\Delta U$')
    p2, = ax.plot(xgrid, du_fermi, label='Fermi Heating')

    ax.legend(loc=3, prop={'size': 16}, ncol=1,
              shadow=False, fancybox=False, frameon=False)

    ax.set_xlabel(r'$x/d_e$', fontsize=16)
    ax.tick_params(bottom=True, top=True, left=True, right=False)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.tick_params(labelsize=12)
    ax.set_xlim([xmin, xmax])

    twpe = math.ceil(tindex * pic_info.dtwpe / 0.1) * 0.1
    text1 = r'$t\omega_{pe}=' + ("{%0.0f}" % twpe) + '$'
    fig.suptitle(text1, fontsize=20)
    # img_dir = '../img/rate_problem/pperp_heating/' + pic_run + '/'
    # mkdir_p(img_dir)
    # fname = img_dir + "pperp_heating_" + str(tframe) + ".pdf"
    img_dir = '../img/rate_problem/pperp_heating_cut/' + pic_run + '/'
    mkdir_p(img_dir)
    fname = img_dir + "pperp_heating_cut_" + str(tframe) + ".pdf"
    fig.savefig(fname)

    if show_plot:
        plt.show()
    else:
        plt.close()


def average_pressure_pub(plot_config, show_plot=True):
    """Average pressure in the outflow region
    """
    pic_run = plot_config["pic_run"]
    tframe = plot_config["tframe"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_run_dir = pic_info.run_dir
    fields_interval = pic_info.fields_interval
    tindex = fields_interval * tframe
    mime = pic_info.mime
    smime = math.sqrt(mime)
    lx_de = pic_info.lx_di * smime
    lz_de = pic_info.lz_di * smime
    nx = pic_info.nx
    nz = pic_info.nz
    xgrid = np.linspace(0, lx_de, nx)
    zgrid = np.linspace(-0.5*lz_de, lz_de, nz)
    xs = nx//2 - nx//4
    xe = nx//2 + nx//4
    zs = nz//2 - nz//16
    ze = nz//2 + nz//16
    nxs = xe - xs
    nzs = ze - zs
    dx_de = lx_de / nx
    dz_de = lz_de / nz
    xmin = xs * dz_de
    xmax = xe * dz_de
    zmin_pic = -lz_de * 0.5
    zmin = zs * dz_de + zmin_pic
    zmax = ze * dz_de + zmin_pic

    vpic_info = get_vpic_info(pic_run_dir)
    n0 = vpic_info["n0"]
    b0 = vpic_info["b0"]
    Te = vpic_info["Te"]
    Tbe_Te = vpic_info["Tbe/Te"]
    nb_n0 = vpic_info["nb/n0"]
    nb = n0 * nb_n0
    p0 = nb * Te * Tbe_Te
    beta0 = 4 * p0 / b0**2

    plot_config["pic_run"] = pic_run
    plot_config["pic_run_dir"] = pic_run_dir
    bvec_pre = get_bfield_pressure(plot_config, box=[xs, zs, xe, ze])
    sigma = 5
    bvec_pre["pepara"] = gaussian_filter(bvec_pre["pepara"], sigma=sigma)
    bvec_pre["pipara"] = gaussian_filter(bvec_pre["pipara"], sigma=sigma)
    bvec_pre["peperp"] = gaussian_filter(bvec_pre["peperp"], sigma=sigma)
    bvec_pre["piperp"] = gaussian_filter(bvec_pre["piperp"], sigma=sigma)
    bvec_pre["ne"] = gaussian_filter(bvec_pre["ne"], sigma=sigma)
    bvec_pre["ni"] = gaussian_filter(bvec_pre["ni"], sigma=sigma)

    # Outflow velocity
    rho_vel = {}
    for species in ["e", "i"]:
        sname = "electron" if species == 'e' else "ion"
        fname = (pic_run_dir + "hydro_hdf5/T." + str(tindex) +
                 "/hydro_" + sname + "_" + str(tindex) + ".h5")
        hydro = {}
        with h5py.File(fname, 'r') as fh:
            group = fh["Timestep_" + str(tindex)]
            for var in ["rho", "jx"]:
                dset = group[var]
                hydro[var]= dset[xs:xe, 0, zs:ze]

        irho = 1.0 / hydro["rho"]
        vx = hydro["jx"] * irho
        var = "v" + species
        rho_vel[var+"x"] = np.squeeze(vx)
        rho_vel["n"+species] = np.squeeze(np.abs(hydro["rho"]))

    irho = 1.0 / (rho_vel["ne"] + rho_vel["ni"] * mime)
    vsx = (rho_vel["ne"] * rho_vel["vex"] +
           rho_vel["ni"] * rho_vel["vix"] * mime) * irho
    vsx = gaussian_filter(vsx, sigma=sigma)

    dtwpe = pic_info.dtwpe
    dtwce = pic_info.dtwce
    va = dtwce * math.sqrt(1.0 / mime) / dtwpe / math.sqrt(nb_n0)

    # Bxm
    fdir = '../data/rate_problem/bxm/' + pic_run + '/'
    fname = fdir + 'bxm_' + str(tframe) + '.dat'
    fdata = np.fromfile(fname)
    b0 = pic_info.b0
    bxm = fdata[0] / b0
    firehose = 1 - beta0 * (bxm**-2 - bxm) / bxm**2 / 2

    vsx_cut = np.abs(vsx[:, nzs//2] / va)
    tmp = math.sqrt(2/beta0) * vsx_cut
    sbeta_h = math.sqrt(0.5*beta0/math.pi)
    dp_fermi = (vsx_cut**2 +
                (vsx_cut**2 + 3*beta0/4) * erf(tmp) +
                vsx_cut * sbeta_h * np.exp(-tmp**2)) / 6
    dp_norm = nb * mime * va**2
    dp_fermi *= dp_norm
    dp_fermi /= p0

    # Exhaust boundary
    fdir = '../data/rate_problem/exhaust_boundary/' + pic_run + '/'
    fname = fdir + 'xz_top_' + str(tframe) + '.dat'
    xz = np.fromfile(fname).reshape([2, -1])
    xlist_top = xz[0, :]
    zlist_top = xz[1, :] + zmin_pic
    f = interp1d(xlist_top, zlist_top)
    ztop = f(xgrid)
    iz_top = np.floor((ztop - zmin_pic) / dz_de).astype(int)
    dz_top = (ztop - zmin_pic) / dz_de - iz_top
    fname = fdir + 'xz_bot_' + str(tframe) + '.dat'
    xz = np.fromfile(fname).reshape([2, -1])
    xlist_bot = xz[0, ::-1]
    zlist_bot = xz[1, ::-1] + zmin_pic
    f = interp1d(xlist_bot, zlist_bot)
    zbot = f(xgrid)
    iz_bot = np.ceil((zbot - zmin_pic) / dz_de).astype(int)
    dz_bot = iz_bot - (zbot - zmin_pic) / dz_de

    # X-point
    x_xp = 0.5 * (xlist_bot[np.argmax(zlist_bot)] +
                  xlist_top[np.argmin(zlist_top)])
    ix_xp = int(x_xp / dx_de) - xs

    # Averaged pressure in the exhaust
    nvar = 4
    pres_avg = np.zeros([nvar, nxs])
    navg = np.zeros(nxs)
    bavg = np.zeros(nxs)
    absB = np.sqrt(bvec_pre["bx"]**2 + bvec_pre["by"]**2 + bvec_pre["bz"]**2)
    for ix in range(nxs):
        ixs = ix + xs
        iz1 = iz_bot[ixs] - zs
        iz2 = iz_top[ixs] - zs
        pres_avg[0, ix] = np.mean(bvec_pre["pepara"][ix, iz1:iz2+1])
        pres_avg[1, ix] = np.mean(bvec_pre["peperp"][ix, iz1:iz2+1])
        pres_avg[2, ix] = np.mean(bvec_pre["pipara"][ix, iz1:iz2+1])
        pres_avg[3, ix] = np.mean(bvec_pre["piperp"][ix, iz1:iz2+1])
        navg[ix] = np.mean(bvec_pre["ni"][ix, iz1:iz2+1])
        bavg[ix] = np.mean(absB[ix, iz1:iz2+1])

    pixx = bvec_pre["pixx"][:, nzs//2]
    piyy = bvec_pre["piyy"][:, nzs//2]
    pizz = bvec_pre["pizz"][:, nzs//2]
    pixx = gaussian_filter(pixx, sigma=5)
    piyy = gaussian_filter(piyy, sigma=5)
    pizz = gaussian_filter(pizz, sigma=5)

    fig = plt.figure(figsize=[7, 7])
    rect = [0.13, 0.75, 0.75, 0.2]
    hgap, vgap = 0.02, 0.03

    axs = []
    nvar = 3

    for i in range(nvar):
        ax = fig.add_axes(rect)
        if i == nvar - 1:
            ax.set_xlabel(r'$x/d_e$', fontsize=16)
        else:
            ax.tick_params(axis='x', labelbottom=False)
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(labelsize=12)
        ax.set_xlim([xmin, xmax])
        axs.append(ax)
        if i == 1:
            rect_cbar = np.copy(rect)
            rect_cbar[0] += rect[2] + 0.02
            rect_cbar[2] = 0.02
            rect_cbar[3] = rect[3] * 2 + vgap
            cbar_ax = fig.add_axes(rect_cbar)
        if i == nvar - 2:
            rect[3] = 0.4
        rect[1] -= rect[3] + vgap
    xmin_di = xmin / smime
    xmax_di = xmax / smime
    zmin_di = zmin / smime
    zmax_di = zmax / smime
    kwargs = {"current_time": tframe,
              "xl": xmin_di, "xr": xmax_di,
              "zb": zmin_di, "zt": zmax_di}
    fname = pic_run_dir + "data/Ay.gda"
    x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
    xde = x * smime
    zde = z * smime
    if pic_run == "mime400_Tb_T0_025":
        dmin, dmax = 1, 8
    elif pic_run == "mime400_nb_n0_1" or pic_run == "mime400_Tb_T0_1":
        dmin, dmax = 1, 4
    elif "mime400_Tb_T0_10" in pic_run:
        dmin, dmax = 1, 1.2
    elif "mime400_Tb_T0_40" in pic_run:
        dmin, dmax = 1, 1.05
    im0 = axs[0].imshow(bvec_pre["pipara"].T / p0,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=dmin, vmax=dmax,
                        cmap=plt.cm.viridis, aspect='auto',
                        origin='lower', interpolation='bicubic')
    im1 = axs[1].imshow(bvec_pre["piperp"].T / p0,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=dmin, vmax=dmax,
                        cmap=plt.cm.viridis, aspect='auto',
                        origin='lower', interpolation='bicubic')
    axs[0].text(0.02, 0.85, r"$P_{i\parallel}/P_0$", color='k', fontsize=16,
                bbox=dict(facecolor='w', alpha=0.7, edgecolor='k', pad=3.0),
                horizontalalignment='left', verticalalignment='center',
                transform=axs[0].transAxes)
    axs[1].text(0.02, 0.85, r"$P_{i\perp}/P_0$", color='k', fontsize=16,
                bbox=dict(facecolor='w', alpha=0.7, edgecolor='k', pad=3.0),
                horizontalalignment='left', verticalalignment='center',
                transform=axs[1].transAxes)
    axs[0].contour(xde, zde, Ay, colors='w', linewidths=0.5,
                   levels=np.linspace(np.min(Ay), np.max(Ay), 8))
    axs[1].contour(xde, zde, Ay, colors='w', linewidths=0.5,
                   levels=np.linspace(np.min(Ay), np.max(Ay), 8))
    COLORS = palettable.colorbrewer.qualitative.Set1_9.mpl_colors
    axs[0].plot(xlist_top, zlist_top, linewidth=1, color=COLORS[0])
    axs[0].plot(xlist_bot, zlist_bot, linewidth=1, color=COLORS[0])
    axs[1].plot(xlist_top, zlist_top, linewidth=1, color=COLORS[0])
    axs[1].plot(xlist_bot, zlist_bot, linewidth=1, color=COLORS[0])
    cbar = fig.colorbar(im1, cax=cbar_ax, extend='both')
    cbar.set_ticks([1, 1.05, 1.1, 1.15, 1.2])
    cbar_ax.tick_params(bottom=False, top=False, left=False, right=True)
    cbar_ax.tick_params(axis='y', which='major', direction='out')
    cbar_ax.tick_params(axis='y', which='minor', direction='in', right=False)
    cbar.ax.tick_params(labelsize=12)

    xgrid = np.arange(xs, xe) * dx_de
    # COLORS = palettable.tableau.Tableau_10.mpl_colors
    COLORS = palettable.colorbrewer.qualitative.Set1_9.mpl_colors
    ax = axs[2]
    ax.set_prop_cycle('color', COLORS)
    p1, = ax.plot(xgrid, pres_avg[2]/p0 - pres_avg[2, ix_xp]/p0,
                  label=r'$\Delta P_{i\parallel}$')
    p2, = ax.plot(xgrid, pres_avg[3]/p0 - pres_avg[3, ix_xp]/p0,
                  label=r'$\Delta P_{i\perp}$')
    p3, = ax.plot(xgrid, dp_fermi, label='Fermi Heating')
    ax.text(0.9, 0.82, r"$\left<\Delta P_{i\parallel}\right>_z$",
            color=COLORS[0], fontsize=16,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes)
    ax.text(0.9, 0.20, r"$\left<\Delta P_{i\perp}\right>_z$",
            color=COLORS[1], fontsize=16,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes)
    ax.text(0.9, 0.55, "Fermi",
            color=COLORS[2], fontsize=16,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes)

    axs[0].text(-0.12, 0.85, "(a)", color="k", fontsize=20,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='center', verticalalignment='center',
                transform=axs[0].transAxes)
    axs[1].text(-0.12, 0.85, "(b)", color="k", fontsize=20,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='center', verticalalignment='center',
                transform=axs[1].transAxes)
    axs[2].text(-0.12, 0.9, "(c)", color="k", fontsize=20,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='center', verticalalignment='center',
                transform=axs[2].transAxes)

    twpe = math.ceil(tindex * pic_info.dtwpe / 0.1) * 0.1
    text1 = r'$t\omega_{pe}=' + ("{%0.0f}" % twpe) + '$'
    fig.suptitle(text1, fontsize=16)
    img_dir = '../img/rate_problem/pub/'
    mkdir_p(img_dir)
    fname = img_dir + "pi_" + str(tframe) + ".pdf"
    fig.savefig(fname, dpi=300)

    if show_plot:
        plt.show()
    else:
        plt.close()


def average_pressure_pub2(plot_config, show_plot=True):
    """Average pressure in the outflow region
    """
    pic_run = plot_config["pic_run"]
    tframe = plot_config["tframe"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_run_dir = pic_info.run_dir
    fields_interval = pic_info.fields_interval
    tindex = fields_interval * tframe
    mime = pic_info.mime
    smime = math.sqrt(mime)
    lx_de = pic_info.lx_di * smime
    lz_de = pic_info.lz_di * smime
    nx = pic_info.nx
    nz = pic_info.nz
    xgrid = np.linspace(0, lx_de, nx)
    zgrid = np.linspace(-0.5*lz_de, lz_de, nz)
    xs = nx//2 - nx//4
    xe = nx//2 + nx//4
    zs = nz//2 - nz//16
    ze = nz//2 + nz//16
    nxs = xe - xs
    nzs = ze - zs
    dx_de = lx_de / nx
    dz_de = lz_de / nz
    xmin = xs * dz_de
    xmax = xe * dz_de
    zmin_pic = -lz_de * 0.5
    zmin = zs * dz_de + zmin_pic
    zmax = ze * dz_de + zmin_pic

    vpic_info = get_vpic_info(pic_run_dir)
    n0 = vpic_info["n0"]
    b0 = vpic_info["b0"]
    Te = vpic_info["Te"]
    Tbe_Te = vpic_info["Tbe/Te"]
    nb_n0 = vpic_info["nb/n0"]
    nb = n0 * nb_n0
    p0 = nb * Te * Tbe_Te
    beta0 = 4 * p0 / b0**2

    plot_config["pic_run"] = pic_run
    plot_config["pic_run_dir"] = pic_run_dir
    bvec_pre = get_bfield_pressure(plot_config, box=[xs, zs, xe, ze])
    sigma = 5
    bvec_pre["pepara"] = gaussian_filter(bvec_pre["pepara"], sigma=sigma)
    bvec_pre["pipara"] = gaussian_filter(bvec_pre["pipara"], sigma=sigma)
    bvec_pre["peperp"] = gaussian_filter(bvec_pre["peperp"], sigma=sigma)
    bvec_pre["piperp"] = gaussian_filter(bvec_pre["piperp"], sigma=sigma)
    bvec_pre["ne"] = gaussian_filter(bvec_pre["ne"], sigma=sigma)
    bvec_pre["ni"] = gaussian_filter(bvec_pre["ni"], sigma=sigma)
    b2 = bvec_pre["bx"]**2 + bvec_pre["by"]**2 + bvec_pre["bz"]**2
    epsilon = 1 - (bvec_pre["pepara"] + bvec_pre["pipara"] -
                   bvec_pre["peperp"] - bvec_pre["piperp"]) / b2

    # Outflow velocity
    rho_vel = {}
    for species in ["e", "i"]:
        sname = "electron" if species == 'e' else "ion"
        fname = (pic_run_dir + "hydro_hdf5/T." + str(tindex) +
                 "/hydro_" + sname + "_" + str(tindex) + ".h5")
        hydro = {}
        with h5py.File(fname, 'r') as fh:
            group = fh["Timestep_" + str(tindex)]
            for var in ["rho", "jx"]:
                dset = group[var]
                hydro[var]= dset[xs:xe, 0, zs:ze]

        irho = 1.0 / hydro["rho"]
        vx = hydro["jx"] * irho
        var = "v" + species
        rho_vel[var+"x"] = np.squeeze(vx)
        rho_vel["n"+species] = np.squeeze(np.abs(hydro["rho"]))

    irho = 1.0 / (rho_vel["ne"] + rho_vel["ni"] * mime)
    vsx = (rho_vel["ne"] * rho_vel["vex"] +
           rho_vel["ni"] * rho_vel["vix"] * mime) * irho
    vsx = gaussian_filter(vsx, sigma=sigma)

    dtwpe = pic_info.dtwpe
    dtwce = pic_info.dtwce
    va = dtwce * math.sqrt(1.0 / mime) / dtwpe / math.sqrt(nb_n0)

    # Bxm
    fdir = '../data/rate_problem/bxm/' + pic_run + '/'
    fname = fdir + 'bxm_' + str(tframe) + '.dat'
    fdata = np.fromfile(fname)
    b0 = pic_info.b0
    bxm = fdata[0] / b0
    firehose = 1 - beta0 * (bxm**-2 - bxm) / bxm**2 / 2

    vsx_cut = np.abs(vsx[:, nzs//2] / va)
    tmp = math.sqrt(2/beta0) * vsx_cut
    sbeta_h = math.sqrt(0.5*beta0/math.pi)
    dp_fermi = (vsx_cut**2 * (1+erf(tmp)) +
                beta0 * erf(tmp) / 4 +
                vsx_cut * sbeta_h * np.exp(-tmp**2))
    dp_norm = nb * mime * va**2
    dp_fermi *= dp_norm
    dp_fermi /= p0

    # Exhaust boundary
    fdir = '../data/rate_problem/exhaust_boundary/' + pic_run + '/'
    fname = fdir + 'xz_top_' + str(tframe) + '.dat'
    xz = np.fromfile(fname).reshape([2, -1])
    xlist_top = xz[0, :]
    zlist_top = xz[1, :] + zmin_pic
    f = interp1d(xlist_top, zlist_top)
    ztop = f(xgrid)
    iz_top = np.floor((ztop - zmin_pic) / dz_de).astype(int)
    dz_top = (ztop - zmin_pic) / dz_de - iz_top
    fname = fdir + 'xz_bot_' + str(tframe) + '.dat'
    xz = np.fromfile(fname).reshape([2, -1])
    xlist_bot = xz[0, ::-1]
    zlist_bot = xz[1, ::-1] + zmin_pic
    f = interp1d(xlist_bot, zlist_bot)
    zbot = f(xgrid)
    iz_bot = np.ceil((zbot - zmin_pic) / dz_de).astype(int)
    dz_bot = iz_bot - (zbot - zmin_pic) / dz_de

    # X-point
    x_xp = 0.5 * (xlist_bot[np.argmax(zlist_bot)] +
                  xlist_top[np.argmin(zlist_top)])
    ix_xp = int(x_xp / dx_de) - xs

    # Averaged pressure in the exhaust
    nvar = 4
    pres_avg = np.zeros([nvar, nxs])
    navg = np.zeros(nxs)
    bavg = np.zeros(nxs)
    absB = np.sqrt(bvec_pre["bx"]**2 + bvec_pre["by"]**2 + bvec_pre["bz"]**2)
    for ix in range(nxs):
        ixs = ix + xs
        iz1 = iz_bot[ixs] - zs
        iz2 = iz_top[ixs] - zs
        pres_avg[0, ix] = np.mean(bvec_pre["pepara"][ix, iz1:iz2+1])
        pres_avg[1, ix] = np.mean(bvec_pre["peperp"][ix, iz1:iz2+1])
        pres_avg[2, ix] = np.mean(bvec_pre["pipara"][ix, iz1:iz2+1])
        pres_avg[3, ix] = np.mean(bvec_pre["piperp"][ix, iz1:iz2+1])
        navg[ix] = np.mean(bvec_pre["ni"][ix, iz1:iz2+1])
        bavg[ix] = np.mean(absB[ix, iz1:iz2+1])

    pixx = bvec_pre["pixx"]
    piyy = bvec_pre["piyy"]
    pizz = bvec_pre["pizz"]
    pixx = gaussian_filter(pixx, sigma=3)
    piyy = gaussian_filter(piyy, sigma=3)
    pizz = gaussian_filter(pizz, sigma=3)
    pixx_cut = pixx[:, nzs//2]
    piyy_cut = piyy[:, nzs//2]
    pizz_cut = pizz[:, nzs//2]

    fig = plt.figure(figsize=[7, 7])
    rect = [0.13, 0.75, 0.75, 0.2]
    hgap, vgap = 0.02, 0.03

    axs = []
    cbar_axs = []
    nvar = 3

    for i in range(nvar):
        ax = fig.add_axes(rect)
        if i == nvar - 1:
            ax.set_xlabel(r'$x/d_e$', fontsize=16)
        else:
            ax.tick_params(axis='x', labelbottom=False)
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(labelsize=12)
        ax.set_xlim([xmin, xmax])
        axs.append(ax)
        if i < nvar - 1:
            rect_cbar = np.copy(rect)
            rect_cbar[0] += rect[2] + 0.02
            rect_cbar[2] = 0.02
            rect_cbar[3] = rect[3]
            cbar_axs.append(fig.add_axes(rect_cbar))
        if i == nvar - 2:
            rect[3] = 0.4
        rect[1] -= rect[3] + vgap
    xmin_di = xmin / smime
    xmax_di = xmax / smime
    zmin_di = zmin / smime
    zmax_di = zmax / smime
    kwargs = {"current_time": tframe,
              "xl": xmin_di, "xr": xmax_di,
              "zb": zmin_di, "zt": zmax_di}
    fname = pic_run_dir + "data/Ay.gda"
    x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
    xde = x * smime
    zde = z * smime
    if pic_run == "mime400_Tb_T0_025":
        dmin, dmax = 1, 8
    elif pic_run == "mime400_nb_n0_1" or pic_run == "mime400_Tb_T0_1":
        dmin, dmax = 1, 4
    elif "mime400_Tb_T0_10" in pic_run:
        dmin, dmax = 1, 1.2
    elif "mime400_Tb_T0_40" in pic_run:
        dmin, dmax = 1, 1.05
    cmap = mpl.cm.inferno_r
    bounds = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')
    im0 = axs[0].imshow(epsilon.T,
                        extent=[xmin, xmax, zmin, zmax],
                        norm=norm,
                        cmap=cmap, aspect='auto',
                        origin='lower', interpolation='bicubic')
    axs[0].text(0.02, 0.85, r"$\epsilon$", color='k', fontsize=16,
                bbox=dict(facecolor='w', alpha=0.7, edgecolor='k', pad=3.0),
                horizontalalignment='left', verticalalignment='center',
                transform=axs[0].transAxes)
    axs[0].contour(xde, zde, Ay, colors='w', linewidths=0.5,
                   levels=np.linspace(np.min(Ay), np.max(Ay), 8))
    # cbar = fig.colorbar(im0, cax=cbar_axs[0], extend='both')
    cbar = mpl.colorbar.ColorbarBase(cbar_axs[0], cmap=cmap, norm=norm, extend="both")
    cbar_axs[1].tick_params(bottom=False, top=False, left=False, right=True)
    cbar_axs[1].tick_params(axis='y', which='major', direction='out')
    cbar_axs[1].tick_params(axis='y', which='minor', direction='in', right=False)
    cbar.ax.tick_params(labelsize=12)
    im1 = axs[1].imshow(pixx.T / p0,
                        extent=[xmin, xmax, zmin, zmax],
                        vmin=dmin, vmax=dmax,
                        cmap=plt.cm.viridis, aspect='auto',
                        origin='lower', interpolation='bicubic')
    axs[1].text(0.02, 0.85, r"$P_{ixx}/P_{i0}$", color='k', fontsize=16,
                bbox=dict(facecolor='w', alpha=0.7, edgecolor='k', pad=3.0),
                horizontalalignment='left', verticalalignment='center',
                transform=axs[1].transAxes)
    axs[1].contour(xde, zde, Ay, colors='w', linewidths=0.5,
                   levels=np.linspace(np.min(Ay), np.max(Ay), 8))
    COLORS = palettable.colorbrewer.qualitative.Set1_9.mpl_colors
    # axs[0].plot(xlist_top, zlist_top, linewidth=1, color=COLORS[0])
    # axs[0].plot(xlist_bot, zlist_bot, linewidth=1, color=COLORS[0])
    cbar = fig.colorbar(im1, cax=cbar_axs[1], extend='both')
    cbar.set_ticks([1, 1.05, 1.1, 1.15, 1.2])
    cbar_axs[1].tick_params(bottom=False, top=False, left=False, right=True)
    cbar_axs[1].tick_params(axis='y', which='major', direction='out')
    cbar_axs[1].tick_params(axis='y', which='minor', direction='in', right=False)
    cbar.ax.tick_params(labelsize=12)

    xgrid = np.arange(xs, xe) * dx_de
    # COLORS = palettable.tableau.Tableau_10.mpl_colors
    COLORS = palettable.colorbrewer.qualitative.Set1_9.mpl_colors
    ax = axs[2]
    ax.set_prop_cycle('color', COLORS)
    p1, = ax.plot(xgrid, pixx_cut/p0 - pixx_cut[ix_xp]/p0, label=r'$\Delta P_{ixx}$')
    p2, = ax.plot(xgrid, dp_fermi, label='Fermi Heating')
    ax.text(0.9, 0.13, r"$\Delta P_{ixx}/P_{i0}$",
            color=COLORS[0], fontsize=16,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='bottom',
            transform=ax.transAxes)
    ax.text(0.9, 0.03, "Fermi",
            color=COLORS[1], fontsize=16,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='bottom',
            transform=ax.transAxes)

    axs[0].text(-0.12, 0.86, "(a)", color="k", fontsize=20,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='center', verticalalignment='center',
                transform=axs[0].transAxes)
    axs[1].text(-0.12, 0.86, "(b)", color="k", fontsize=20,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='center', verticalalignment='center',
                transform=axs[1].transAxes)
    axs[2].text(-0.12, 0.88, "(c)", color="k", fontsize=20,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='center', verticalalignment='center',
                transform=axs[2].transAxes)

    twpe = math.ceil(tindex * pic_info.dtwpe / 0.1) * 0.1
    text1 = r'$t\omega_{pe}=' + ("{%0.0f}" % twpe) + '$'
    fig.suptitle(text1, fontsize=16)
    img_dir = '../img/rate_problem/pub/'
    mkdir_p(img_dir)
    # fname = img_dir + "pi_" + str(tframe) + "_2.pdf"
    # fig.savefig(fname, dpi=300)

    if show_plot:
        plt.show()
    else:
        plt.close()


def average_pressure_pub3(plot_config, show_plot=True):
    """Average pressure in the outflow region
    """
    pic_run = plot_config["pic_run"]
    tframe = plot_config["tframe"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_run_dir = pic_info.run_dir
    fields_interval = pic_info.fields_interval
    tindex = fields_interval * tframe
    mime = pic_info.mime
    smime = math.sqrt(mime)
    lx_de = pic_info.lx_di * smime
    lz_de = pic_info.lz_di * smime
    nx = pic_info.nx
    nz = pic_info.nz
    xgrid = np.linspace(0, lx_de, nx)
    zgrid = np.linspace(-0.5*lz_de, lz_de, nz)
    xs = nx//2 - nx//8
    xe = nx//2 + nx//8
    zs = nz//2 - nz//8
    ze = nz//2 + nz//8
    nxs = xe - xs
    nzs = ze - zs
    dx_de = lx_de / nx
    dz_de = lz_de / nz
    xmin = xs * dz_de
    xmax = xe * dz_de
    zmin_pic = -lz_de * 0.5
    zmin = zs * dz_de + zmin_pic
    zmax = ze * dz_de + zmin_pic

    vpic_info = get_vpic_info(pic_run_dir)
    n0 = vpic_info["n0"]
    b0 = vpic_info["b0"]
    Te = vpic_info["Te"]
    Tbe_Te = vpic_info["Tbe/Te"]
    nb_n0 = vpic_info["nb/n0"]
    nb = n0 * nb_n0
    p0 = nb * Te * Tbe_Te
    beta0 = 4 * p0 / b0**2
    if pic_run == "mime400_Tb_T0_1":
        pe0 = 0.878 * p0
        pi0 = p0
    elif pic_run == "mime400_Tb_T0_10_weak":
        pe0 = 0.998 * p0
        pi0 = 0.9946 * p0
    else:
        pe0 = p0
        pi0 = p0

    plot_config["pic_run"] = pic_run
    plot_config["pic_run_dir"] = pic_run_dir
    bvec_pre = get_bfield_pressure(plot_config, box=[xs, zs, xe, ze])
    sigma = 3
    bvec_pre["pepara"] = gaussian_filter(bvec_pre["pepara"], sigma=sigma)
    bvec_pre["pipara"] = gaussian_filter(bvec_pre["pipara"], sigma=sigma)
    bvec_pre["peperp"] = gaussian_filter(bvec_pre["peperp"], sigma=sigma)
    bvec_pre["piperp"] = gaussian_filter(bvec_pre["piperp"], sigma=sigma)
    bvec_pre["ne"] = gaussian_filter(bvec_pre["ne"], sigma=sigma)
    bvec_pre["ni"] = gaussian_filter(bvec_pre["ni"], sigma=sigma)
    bx = np.abs(gaussian_filter(bvec_pre["bx"], sigma=sigma)) / b0
    b2 = bvec_pre["bx"]**2 + bvec_pre["by"]**2 + bvec_pre["bz"]**2
    b2 = gaussian_filter(b2, sigma=sigma)
    absB = np.sqrt(b2)
    epsilon = 1 - (bvec_pre["pepara"] + bvec_pre["pipara"] -
                   bvec_pre["peperp"] - bvec_pre["piperp"]) / b2
    pxx_model = np.ones(b2.shape) * p0
    pyy_model = p0 * absB / b0
    pzz_model = p0 * absB / b0
    firehose_model = 1 + 0.5 * beta0 * (1/bx - 1/bx**2)
    # firehose_model = 1 + 0.25 * beta0 * (b0/absB - b0**2/b2)

    # Outflow velocity
    rho_vel = {}
    for species in ["e", "i"]:
        sname = "electron" if species == 'e' else "ion"
        fname = (pic_run_dir + "hydro_hdf5/T." + str(tindex) +
                 "/hydro_" + sname + "_" + str(tindex) + ".h5")
        hydro = {}
        with h5py.File(fname, 'r') as fh:
            group = fh["Timestep_" + str(tindex)]
            for var in ["rho", "jx"]:
                dset = group[var]
                hydro[var]= dset[xs:xe, 0, zs:ze]

        irho = 1.0 / hydro["rho"]
        vx = hydro["jx"] * irho
        var = "v" + species
        rho_vel[var+"x"] = np.squeeze(vx)
        rho_vel["n"+species] = np.squeeze(np.abs(hydro["rho"]))

    irho = 1.0 / (rho_vel["ne"] + rho_vel["ni"] * mime)
    vsx = (rho_vel["ne"] * rho_vel["vex"] +
           rho_vel["ni"] * rho_vel["vix"] * mime) * irho
    vsx = gaussian_filter(vsx, sigma=sigma)

    dtwpe = pic_info.dtwpe
    dtwce = pic_info.dtwce
    va = dtwce * math.sqrt(1.0 / mime) / dtwpe / math.sqrt(nb_n0)

    # Bxm
    fdir = '../data/rate_problem/bxm/' + pic_run + '/'
    fname = fdir + 'bxm_' + str(tframe) + '.dat'
    fdata = np.fromfile(fname)
    b0 = pic_info.b0
    bxm = fdata[0] / b0
    firehose = 1 - beta0 * (bxm**-2 - bxm) / bxm**2 / 2

    vsx_cut = np.abs(vsx[:, nzs//2] / va)
    tmp = math.sqrt(2/beta0) * vsx_cut
    sbeta_h = math.sqrt(0.5*beta0/math.pi)
    dp_fermi = (vsx_cut**2 * (1+erf(tmp)) +
                beta0 * erf(tmp) / 4 +
                vsx_cut * sbeta_h * np.exp(-tmp**2))
    dp_norm = nb * mime * va**2
    dp_fermi *= dp_norm
    dp_fermi /= p0

    # Exhaust boundary
    fdir = '../data/rate_problem/exhaust_boundary/' + pic_run + '/'
    fname = fdir + 'xz_top_' + str(tframe) + '.dat'
    xz = np.fromfile(fname).reshape([2, -1])
    xlist_top = xz[0, :]
    zlist_top = xz[1, :] + zmin_pic
    f = interp1d(xlist_top, zlist_top)
    ztop = f(xgrid)
    iz_top = np.floor((ztop - zmin_pic) / dz_de).astype(int)
    dz_top = (ztop - zmin_pic) / dz_de - iz_top
    fname = fdir + 'xz_bot_' + str(tframe) + '.dat'
    xz = np.fromfile(fname).reshape([2, -1])
    xlist_bot = xz[0, ::-1]
    zlist_bot = xz[1, ::-1] + zmin_pic
    f = interp1d(xlist_bot, zlist_bot)
    zbot = f(xgrid)
    iz_bot = np.ceil((zbot - zmin_pic) / dz_de).astype(int)
    dz_bot = iz_bot - (zbot - zmin_pic) / dz_de

    # X-point
    x_xp = 0.5 * (xlist_bot[np.argmax(zlist_bot)] +
                  xlist_top[np.argmin(zlist_top)])
    ix_xp = int(x_xp / dx_de) - xs

    pixx = gaussian_filter(bvec_pre["pixx"], sigma=3)
    piyy = gaussian_filter(bvec_pre["piyy"], sigma=3)
    pizz = gaussian_filter(bvec_pre["pizz"], sigma=3)
    pexx = gaussian_filter(bvec_pre["pexx"], sigma=3)
    peyy = gaussian_filter(bvec_pre["peyy"], sigma=3)
    pezz = gaussian_filter(bvec_pre["pezz"], sigma=3)
    pexx_cut = pexx[:, nzs//2]
    peyy_cut = peyy[:, nzs//2]
    pezz_cut = pezz[:, nzs//2]
    pixx_cut = pixx[:, nzs//2]
    piyy_cut = piyy[:, nzs//2]
    pizz_cut = pizz[:, nzs//2]
    # nz_di = int(pic_info.nz / pic_info.lz_di)
    # zsc = nzs//2 - nz_di // 2
    # zec = nzs//2 + nz_di // 2
    # pexx_cut = np.mean(pexx[:, zsc:zec], axis=1)
    # peyy_cut = np.mean(peyy[:, zsc:zec], axis=1)
    # pezz_cut = np.mean(pezz[:, zsc:zec], axis=1)
    # pixx_cut = np.mean(pixx[:, zsc:zec], axis=1)
    # piyy_cut = np.mean(piyy[:, zsc:zec], axis=1)
    # pizz_cut = np.mean(pizz[:, zsc:zec], axis=1)

    pexx_zcut = pexx[ix_xp, :]
    peyy_zcut = peyy[ix_xp, :]
    pezz_zcut = pezz[ix_xp, :]
    pixx_zcut = pixx[ix_xp, :]
    piyy_zcut = piyy[ix_xp, :]
    pizz_zcut = pizz[ix_xp, :]
    pxx_zcut = pexx[ix_xp, :] + pixx[ix_xp, :]
    pyy_zcut = peyy[ix_xp, :] + piyy[ix_xp, :]
    pzz_zcut = pezz[ix_xp, :] + pizz[ix_xp, :]
    epsilon_zcut = epsilon[ix_xp, :]

    fig = plt.figure(figsize=[7, 9])
    hgap, vgap = 0.05, 0.02
    rect0 = [0.12, 0.64, 0.4, 0.31]

    axs = []
    cbar_axs = []
    nvar = 5

    xmin_di = xmin / smime
    xmax_di = xmax / smime
    zmin_di = zmin / smime
    zmax_di = zmax / smime
    xgrid = np.arange(xs, xe) * dx_de
    zgrid = np.arange(zs, ze) * dz_de + zmin_pic
    xgrid_di = xgrid / smime
    zgrid_di = zgrid / smime
    for i in range(nvar):
        if i == 0:
            rect = np.copy(rect0)
        elif i == 2:
            rect[1] += rect[3] + vgap
            rect[3] = 0.22
            rect[1] -= rect[3] + vgap
        elif i == 3:
            rect = np.copy(rect0)
            rect[0] += rect[2] + hgap
            # rect[2] -= 0.02
        ax = fig.add_axes(rect)
        if i < 2:
            ax.tick_params(axis='x', labelbottom=False)
        elif i == 2:
            ax.set_xlabel(r'$x/d_i$', fontsize=16)
        elif i < nvar - 1:
            ax.tick_params(axis='x', labelbottom=False)
        if i in [3, 4]:
            ax.tick_params(axis='y', labelleft=False)
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(labelsize=12)
        if i < 3:
            ax.set_xlim([xmin_di, xmax_di])
        else:
            ax.set_ylim([zmin_di, zmax_di])
        axs.append(ax)
        if i in [0, 1]:
            rect_cbar = np.copy(rect)
            rect_cbar[0] += 0.05
            rect_cbar[2] = rect[2] - 0.1
            rect_cbar[1] += 0.05
            rect_cbar[3] = 0.01
            cbar_axs.append(fig.add_axes(rect_cbar))
            ax.set_ylabel(r'$z/d_i$', fontsize=16)
        rect[1] -= rect[3] + vgap
    COLORS = palettable.colorbrewer.qualitative.Set1_9.mpl_colors

    kwargs = {"current_time": tframe,
              "xl": xmin_di, "xr": xmax_di,
              "zb": zmin_di, "zt": zmax_di}
    fname = pic_run_dir + "data/Ay.gda"
    x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
    xde = x * smime
    zde = z * smime
    if pic_run == "mime400_Tb_T0_025":
        dmin, dmax = 1, 8
    elif pic_run == "mime400_nb_n0_1" or pic_run == "mime400_Tb_T0_1":
        dmin, dmax = 1, 3
    elif "mime400_Tb_T0_10" in pic_run:
        dmin, dmax = 1, 1.2
    elif "mime400_Tb_T0_40" in pic_run:
        dmin, dmax = 1, 1.05
    cmap = mpl.cm.inferno_r
    bounds = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')
    im0 = axs[0].imshow(epsilon.T,
                        extent=[xmin_di, xmax_di, zmin_di, zmax_di],
                        norm=norm,
                        cmap=cmap, aspect='auto',
                        origin='lower', interpolation='bicubic')
    axs[0].plot([xgrid_di[ix_xp], xgrid_di[ix_xp]],
                [zmin_di, zmax_di], color='w', alpha=0.7,
                linewidth=1, linestyle='--')
    im1 = axs[1].imshow(pixx.T / pi0,
                        extent=[xmin_di, xmax_di, zmin_di, zmax_di],
                        vmin=dmin, vmax=dmax,
                        cmap=plt.cm.viridis, aspect='auto',
                        origin='lower', interpolation='bicubic')
    axs[1].plot([xgrid_di[ix_xp], xgrid_di[ix_xp]],
                [zmin_di, zmax_di], color='w', alpha=0.7,
                linewidth=1, linestyle='--')
    axs[1].plot([xmin_di, xmax_di],
                [0, 0], color='w', alpha=0.7,
                linewidth=1, linestyle='--')
    labels = [r"$\varepsilon$", r"$P_{ixx}/P_{i0}$"]
    for iax in range(0, 2):
        axs[iax].text(0.05, 0.9, labels[iax], color='k', fontsize=16,
                      bbox=dict(facecolor='w', alpha=0.7, edgecolor='k', pad=3.0),
                      horizontalalignment='left', verticalalignment='center',
                      transform=axs[iax].transAxes)
        axs[iax].contour(x, z, Ay, colors='w', linewidths=1, alpha=0.5,
                         levels=np.linspace(np.min(Ay), np.max(Ay), 8))
    cbar = fig.colorbar(im0, cax=cbar_axs[0], extend='both', orientation="horizontal")
    cbar.ax.tick_params(labelsize=12, color='w')
    cbar.ax.xaxis.set_tick_params(color='w')
    cbar.outline.set_edgecolor('w')
    plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='w')
    cbar = fig.colorbar(im1, cax=cbar_axs[1], extend='both', orientation="horizontal")
    cbar.ax.tick_params(labelsize=12, color='w')
    cbar.ax.xaxis.set_tick_params(color='w')
    cbar.outline.set_edgecolor('w')
    plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='w')

    # COLORS = palettable.tableau.Tableau_10.mpl_colors
    COLORS = palettable.colorbrewer.qualitative.Set1_9.mpl_colors
    ax = axs[2]
    ax.set_prop_cycle('color', COLORS)
    p1, = ax.plot(xgrid_di, pixx_cut/p0 - np.min(pixx_cut)/p0)
    p2, = ax.plot(xgrid_di, pexx_cut/p0 - np.min(pexx_cut)/p0)
    p3, = ax.plot(xgrid_di, dp_fermi, label='Fermi heating (ion)', color='k')
    # ax.legend(loc=9, prop={'size': 16}, ncol=1,
    #           shadow=False, fancybox=False, frameon=False)
    ax.text(0.5, 0.85, r"$\Delta P_{ixx}/P_{i0}$ (model)",
            color='k', fontsize=16,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='bottom',
            transform=ax.transAxes)
    ax.text(0.5, 0.7, r"$\Delta P_{ixx}/P_{i0}$",
            color=COLORS[0], fontsize=16,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='bottom',
            transform=ax.transAxes)
    ax.text(0.5, 0.55, r"$\Delta P_{exx}/P_{e0}$",
            color=COLORS[1], fontsize=16,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='bottom',
            transform=ax.transAxes)

    iax = 3

    axs[iax].set_prop_cycle('color', COLORS)
    axs[iax].plot(epsilon_zcut, zgrid_di, color='k', label=r"$\varepsilon$")
    axs[iax].plot(firehose_model[ix_xp, :], zgrid_di, color='k',
                  linestyle=':', label=r'$\varepsilon$(model)')
    axs[iax].plot([1, 1], [zgrid_di[0], zgrid_di[-1]],
                  linewidth=1, linestyle='--',color='k')
    axs[iax].set_xlim([0.5, 1.2])
    axs[iax].legend(loc=2, prop={'size': 16}, ncol=1,
                    shadow=False, fancybox=False, frameon=False)
    xlim = axs[iax].get_xlim()
    z0 = 2.5
    axs[iax].fill_between(xlim, -z0, z0, alpha=0.2, color='grey')
    axs[iax].set_xlim(xlim)

    axs[iax+1].set_prop_cycle('color', COLORS)
    axs[iax+1].plot(pixx_zcut/pi0, zgrid_di, label=r'$P_{ixx}/P_{i0}$')
    axs[iax+1].plot(pexx_zcut/pe0, zgrid_di, label=r'$P_{exx}/P_{e0}$')
    axs[iax+1].plot([1, 1], [zgrid_di[0], zgrid_di[-1]],
                    linewidth=1, linestyle='--',color='k')

    # axs[iax+2].set_prop_cycle('color', COLORS)
    axs[iax+1].plot(pizz_zcut/pi0, zgrid_di, label=r'$P_{izz}/P_{i0}$')
    axs[iax+1].plot(pezz_zcut/pe0, zgrid_di, label=r'$P_{ezz}/P_{e0}$')
    axs[iax+1].plot(pzz_model[ix_xp, :]/p0, zgrid_di, label=r"$P_\perp/P_0$ (model)")
    axs[iax+1].plot([1, 1], [zgrid_di[0], zgrid_di[-1]],
                    linewidth=1, linestyle='--',color='k')
    axs[iax+1].set_xlim(axs[iax].get_xlim())
    axs[iax+1].legend(loc=8, bbox_to_anchor=(0.5, -0.8),
                      prop={'size': 16}, ncol=1,
                      shadow=False, fancybox=False, frameon=True)

    xlim = axs[iax+1].get_xlim()
    axs[iax+1].fill_between(xlim, -z0, z0, alpha=0.2, color='grey')
    axs[iax+1].set_xlim(xlim)

    labels = ["a", "c", "e", "b", "d"]
    for iax, ax in enumerate(axs):
        if iax < 3:
            xpos = -0.2
        else:
            xpos = 0.9
        ax.text(xpos, 0.9, "("+labels[iax]+")", color="k", fontsize=20,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)

    twpe = math.ceil(tindex * pic_info.dtwpe / 0.1) * 0.1
    twci = twpe / pic_info.wpe_wce / pic_info.mime
    text1 = r'$t\Omega_{ci}=' + ("{%0.0f}" % twci) + '$'
    fig.suptitle(text1, fontsize=16)
    img_dir = '../img/rate_problem/pub/'
    mkdir_p(img_dir)
    fname = img_dir + "pres_" + str(tframe) + ".pdf"
    fig.savefig(fname, dpi=300)

    if show_plot:
        plt.show()
    else:
        plt.close()


def read_boilerplate(fh):
    """Read boilerplate of a file

    Args:
        fh: file handler
    """
    offset = 0
    sizearr = np.memmap(fh, dtype='int8', mode='r', offset=offset, shape=(5), order='F')
    offset += 5
    cafevar = np.memmap(fh, dtype='int16', mode='r', offset=offset, shape=(1), order='F')
    offset += 2
    deadbeefvar = np.memmap(fh, dtype='int32', mode='r', offset=offset, shape=(1), order='F')
    offset += 4
    realone = np.memmap(fh, dtype='float32', mode='r', offset=offset, shape=(1), order='F')
    offset += 4
    doubleone = np.memmap(fh, dtype='float64', mode='r', offset=offset, shape=(1), order='F')


def read_particle_header(fh):
    """Read particle file header

    Args:
        fh: file handler.
    """
    offset = 23  # the size of the boilerplate is 23
    tmp1 = np.memmap(fh, dtype='int32', mode='r', offset=offset, shape=(6), order='F')
    offset += 6 * 4
    tmp2 = np.memmap(fh, dtype='float32', mode='r', offset=offset, shape=(10), order='F')
    offset += 10 * 4
    tmp3 = np.memmap(fh, dtype='int32', mode='r', offset=offset, shape=(4), order='F')
    v0header = collections.namedtuple("v0header",
                                      ["version", "type", "nt",
                                       "nx", "ny", "nz", "dt",
                                       "dx", "dy", "dz",
                                       "x0", "y0", "z0",
                                       "cvac", "eps0", "damp",
                                       "rank", "ndom", "spid", "spqm"])
    v0 = v0header(version=tmp1[0], type=tmp1[1], nt=tmp1[2],
                  nx=tmp1[3], ny=tmp1[4], nz=tmp1[5], dt=tmp2[0],
                  dx=tmp2[1], dy=tmp2[2], dz=tmp2[3],
                  x0=tmp2[4], y0=tmp2[5], z0=tmp2[6],
                  cvac=tmp2[7], eps0=tmp2[8], damp=tmp2[9],
                  rank=tmp3[0], ndom=tmp3[1], spid=tmp3[2], spqm=tmp3[3])
    header_particle = collections.namedtuple("header_particle",
                                             ["size", "ndim", "dim"])
    offset += 4 * 4
    tmp4 = np.memmap(fh, dtype='int32', mode='r', offset=offset, shape=(3), order='F')
    pheader = header_particle(size=tmp4[0], ndim=tmp4[1], dim=tmp4[2])
    offset += 3 * 4
    return (v0, pheader, offset)


def read_particle_data(fname):
    """Read particle information from a file.

    Args:
        fname: file name.
    """
    fh = open(fname, 'r')
    read_boilerplate(fh)
    v0, pheader, offset = read_particle_header(fh)
    nptl = pheader.dim
    particle_type = np.dtype([('dxyz', np.float32, 3), ('icell', np.int32),
                              ('u', np.float32, 3), ('q', np.float32)])
    fh.seek(offset, os.SEEK_SET)
    data = np.fromfile(fh, dtype=particle_type, count=nptl)
    fh.close()
    return (v0, pheader, data)


def calc_velocity_distribution(v0, pheader, ptl, pic_info, corners,
                               nbins, ptl_mass=1, pmax=1.0):
    """Calculate particle velocity distribution

    Args:
        v0: the header info for the grid.
        pheader: the header info for the particles.
        pic_info: namedtuple for the PIC simulation information.
        corners: the corners of the box in di.
        nbins: number of bins in each dimension.
    """
    dx = ptl['dxyz'][:, 0]
    dy = ptl['dxyz'][:, 1]
    dz = ptl['dxyz'][:, 2]
    icell = ptl['icell']
    ux = ptl['u'][:, 0] * ptl_mass
    uy = ptl['u'][:, 1] * ptl_mass
    uz = ptl['u'][:, 2] * ptl_mass

    nx = v0.nx + 2
    ny = v0.ny + 2
    nz = v0.nz + 2
    iz = icell // (nx * ny)
    iy = (icell - iz * nx * ny) // nx
    ix = icell - iz * nx * ny - iy * nx

    z = v0.z0 + ((iz - 1.0) + (dz + 1.0) * 0.5) * v0.dz
    y = v0.y0 + ((iy - 1.0) + (dy + 1.0) * 0.5) * v0.dy
    x = v0.x0 + ((ix - 1.0) + (dx + 1.0) * 0.5) * v0.dx

    # de -> di
    smime = math.sqrt(pic_info.mime)
    x /= smime
    y /= smime
    z /= smime

    mask = ((x >= corners[0][0]) & (x <= corners[0][1]) &
            (y >= corners[1][0]) & (y <= corners[1][1]) &
            (z >= corners[2][0]) & (z <= corners[2][1]))
    ux_d = ux[mask]
    uy_d = uy[mask]
    uz_d = uz[mask]

    # Assumes that magnetic field is along the z-direction
    upara = uz_d
    uperp = np.sqrt(ux_d * ux_d + uy_d * uy_d)
    upara_abs = np.abs(uz_d)
    utot = np.sqrt(ux_d * ux_d + uy_d * uy_d + uz_d * uz_d)

    drange = [[-pmax, pmax], [-pmax, pmax]]
    hist_xy, ubins_edges, _ = np.histogram2d(uy_d, ux_d, bins=nbins, range=drange)
    hist_xz, ubins_edges, _ = np.histogram2d(uz_d, ux_d, bins=nbins, range=drange)
    hist_yz, ubins_edges, _ = np.histogram2d(uz_d, uy_d, bins=nbins, range=drange)
    drange = [[-pmax, pmax], [0, pmax]]
    hist_para_perp, upara_edges, uperp_edges = np.histogram2d(upara, uperp,
                                                              bins=[nbins, nbins / 2],
                                                              range=drange)

    # 1D
    pmin = 1E-4
    pmin_log, pmax_log = math.log10(pmin), math.log10(pmax)
    pbins_log = 10**np.linspace(pmin_log, pmax_log, nbins)
    ppara_dist, pedge = np.histogram(upara_abs, bins=pbins_log)
    pperp_dist, pedge = np.histogram(uperp, bins=pbins_log)
    pdist, pedge = np.histogram(utot, bins=pbins_log)

    hists = {'hist_xy': hist_xy,
             'hist_xz': hist_xz,
             'hist_yz': hist_yz,
             'hist_para_perp': hist_para_perp,
             'ppara_dist': ppara_dist,
             'pperp_dist': pperp_dist,
             'pdist': pdist}
    bins = {'pbins_long': ubins_edges,
            'pbins_short': uperp_edges,
            'pbins_log': pbins_log}

    return (hists, bins)


def particle_distribution(plot_config, show_plot=True):
    """Get and plot particle distribution
    """
    pic_run = plot_config["pic_run"]
    tframe = plot_config["tframe"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_run_dir = pic_info.run_dir
    topox = pic_info.topology_x
    topoz = pic_info.topology_z
    vpic_info = get_vpic_info(pic_run_dir)
    mime = pic_info.mime
    smime = math.sqrt(mime)
    lx_de = pic_info.lx_di * smime
    lz_de = pic_info.lz_di * smime
    dx_rank = lx_de / topox
    xmin, xmax = 0, lx_de
    zmin, zmax = -0.5 * lz_de, 0.5 * lz_de
    nx = pic_info.nx
    nz = pic_info.nz
    dx_de = lx_de / nx
    dz_de = lz_de / nz

    nx_zone = nz_zone = 48
    nzones_z = pic_info.nz // nz_zone
    dz_zone = nx_zone * dz_de
    nranks_x = int(nx_zone / (nx / topox))

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
    ix_xp = int(x0 / dx_de)

    ncells_di = int(pic_info.nx / pic_info.lx_di)
    zmid = (zmin + zmax) / 2
    nzones_z = 10
    zbins = np.arange(nzones_z+1) - nzones_z/2 - 0.5
    zbins *= dz_zone
    nbins = 32
    hist_xy = np.zeros((nzones_z, nbins, nbins))
    hist_xz = np.zeros((nzones_z, nbins, nbins))
    hist_yz = np.zeros((nzones_z, nbins, nbins))
    vthi = vpic_info["vthib/c"]
    vmin, vmax = -4*vthi, 4*vthi
    vmin_norm, vmax_norm = vmin / vthi, vmax / vthi
    vbins = np.linspace(vmin, vmax, nbins+1)

    tindex = tframe * int(vpic_info["eparticle_interval"])
    dir_name = pic_run_dir + 'particle/T.' + str(tindex) + '/'
    fbase = dir_name + 'hparticle' + '.' + str(tindex) + '.'
    fdir = '../img/rate_problem/vel_dist/' + pic_run + '/'
    fdir += "tframe_" + str(tframe) + "/"
    mkdir_p(fdir)
    for xdi in range(6):
    # for xdi in range(1, 2):
        ix = ix_xp + ncells_di * xdi
        mpi_rankx = math.floor(ix * dx_de / dx_rank)
        rankx_s = mpi_rankx - nranks_x//2
        rankx_e = mpi_rankx + nranks_x//2
        for mpi_iz in range(topoz):
            for mpi_ix in range(rankx_s, rankx_e):
                mpi_rank = mpi_iz * topox + mpi_ix
                print(mpi_rank)
                fname = fbase + str(mpi_rank)
                v0, pheader, ptl = read_particle_data(fname)
                dz = ptl['dxyz'][:, 2]
                icell = ptl['icell']
                ux = ptl['u'][:, 0]
                uy = ptl['u'][:, 1]
                uz = ptl['u'][:, 2]
                nx = v0.nx + 2
                ny = v0.ny + 2
                iz = icell // (nx * ny)
                z = v0.z0 + ((iz - 1.0) + (dz + 1.0) * 0.5) * v0.dz
                hxy, edges = np.histogramdd((z, ux, uy),
                                            bins=(zbins, vbins, vbins))
                hxz, edges = np.histogramdd((z, ux, uz),
                                            bins=(zbins, vbins, vbins))
                hyz, edges = np.histogramdd((z, uy, uz),
                                            bins=(zbins, vbins, vbins))
                hist_xy += hxy
                hist_xz += hxz
                hist_yz += hyz

        out_dir = fdir + "xdi_" + str(xdi) + "/"
        mkdir_p(out_dir)
        cmap = plt.cm.jet
        # for izone in range(nzones_z//2, nzones_z//2+1):
        for izone in range(nzones_z):
            print("Zone %d of %d" % (izone, nzones_z))
            fig = plt.figure(figsize=[5, 12])
            rect = [0.15, 0.7, 0.72, 0.28]
            hgap, vgap = 0.02, 0.04
            ax1 = fig.add_axes(rect)
            dmin, dmax = 1, 1.5E3
            im1 = ax1.imshow(hist_xy[izone].T,
                             extent=[vmin_norm, vmax_norm, vmin_norm, vmax_norm],
                             # vmin=dmin, vmax=dmax,
                             cmap=cmap, aspect='auto',
                             origin='lower', interpolation='bicubic')
            rect_cbar = np.copy(rect)
            rect_cbar[0] += rect[2] + 0.01
            rect_cbar[2] = 0.02
            cbar_ax = fig.add_axes(rect_cbar)
            cbar = fig.colorbar(im1, cax=cbar_ax, extend='max')
            cbar_ax.tick_params(bottom=False, top=False, left=False, right=True)
            cbar_ax.tick_params(axis='y', which='major', direction='out')
            cbar_ax.tick_params(axis='y', which='minor', direction='in', right=False)
            cbar.ax.tick_params(labelsize=12)
            rect[1] -= rect[3] + vgap
            ax2 = fig.add_axes(rect)
            im2 = ax2.imshow(hist_xz[izone].T,
                             extent=[vmin_norm, vmax_norm, vmin_norm, vmax_norm],
                             # vmin=dmin, vmax=dmax,
                             cmap=cmap, aspect='auto',
                             origin='lower', interpolation='bicubic')
            rect_cbar = np.copy(rect)
            rect_cbar[0] += rect[2] + 0.01
            rect_cbar[2] = 0.02
            cbar_ax = fig.add_axes(rect_cbar)
            cbar = fig.colorbar(im2, cax=cbar_ax, extend='max')
            cbar_ax.tick_params(bottom=False, top=False, left=False, right=True)
            cbar_ax.tick_params(axis='y', which='major', direction='out')
            cbar_ax.tick_params(axis='y', which='minor', direction='in', right=False)
            cbar.ax.tick_params(labelsize=12)
            rect[1] -= rect[3] + vgap
            ax3 = fig.add_axes(rect)
            im3 = ax3.imshow(hist_yz[izone].T,
                             extent=[vmin_norm, vmax_norm, vmin_norm, vmax_norm],
                             # vmin=dmin, vmax=dmax,
                             cmap=cmap, aspect='auto',
                             origin='lower', interpolation='bicubic')
            rect_cbar = np.copy(rect)
            rect_cbar[0] += rect[2] + 0.01
            rect_cbar[2] = 0.02
            cbar_ax = fig.add_axes(rect_cbar)
            cbar = fig.colorbar(im3, cax=cbar_ax, extend='max')
            cbar_ax.tick_params(bottom=False, top=False, left=False, right=True)
            cbar_ax.tick_params(axis='y', which='major', direction='out')
            cbar_ax.tick_params(axis='y', which='minor', direction='in', right=False)
            cbar.ax.tick_params(labelsize=12)
            for ax in [ax1, ax2, ax3]:
                ax.plot([0, 0], [vmin_norm, vmax_norm], color='w',
                        linewidth=1, linestyle='dashed')
                ax.plot([vmin_norm, vmax_norm], [0, 0], color='w',
                        linewidth=1, linestyle='dashed')
                ax.tick_params(bottom=True, top=True, left=True, right=True)
                ax.tick_params(axis='x', which='minor', direction='in')
                ax.tick_params(axis='x', which='major', direction='in')
                ax.tick_params(axis='y', which='minor', direction='in')
                ax.tick_params(axis='y', which='major', direction='in')
                ax.tick_params(labelsize=12)
            ax1.set_xlabel(r'$v_{ix}/v_\text{thi}$', fontsize=16)
            ax1.set_ylabel(r'$v_{iy}/v_\text{thi}$', fontsize=16)
            ax2.set_xlabel(r'$v_{ix}/v_\text{thi}$', fontsize=16)
            ax2.set_ylabel(r'$v_{iz}/v_\text{thi}$', fontsize=16)
            ax3.set_xlabel(r'$v_{iy}/v_\text{thi}$', fontsize=16)
            ax3.set_ylabel(r'$v_{iz}/v_\text{thi}$', fontsize=16)
            fname = out_dir + "vdist_" + str(izone) + ".jpg"
            fig.savefig(fname)
            plt.close()
            # plt.show()


def plot_bx_inflow(plot_config, show_plot=True):
    """Plot Bx in the inflow region
    """
    pic_runs = ["mime400_Tb_T0_025",
                "mime1836_Tb_T0_025"]
    tframe = plot_config["tframe"]
    fig = plt.figure(figsize=[7, 5])
    rect = [0.15, 0.12, 0.8, 0.8]
    ax = fig.add_axes(rect)
    ax.set_prop_cycle('color', COLORS)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlabel(r'$z/d_e$', fontsize=16)
    ax.set_ylabel(r'$B_x$', fontsize=16)
    ax.tick_params(labelsize=12)
    labels = [r"$m_i/m_e=400$", r"$m_i/m_e=1836$"]
    for irun, pic_run in enumerate(pic_runs):
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        pic_run_dir = pic_info.run_dir
        fields_interval = pic_info.fields_interval
        tindex = fields_interval * tframe
        if irun == 1:
            tindex *= 4
        smime = math.sqrt(pic_info.mime)
        lx_de = pic_info.lx_di * smime
        lz_de = pic_info.lz_di * smime
        xmin, xmax = 0, lx_de
        zmin, zmax = -0.5 * lz_de, 0.5 * lz_de
        nx = pic_info.nx
        nz = pic_info.nz
        xgrid = np.linspace(xmin, xmax, nx)
        zgrid = np.linspace(zmin, zmax, nz)
        nz_di = int(pic_info.nz / pic_info.lz_di)
        dx_de = lx_de / nx
        dz_de = lz_de / nz

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
        ix_xp = int(x0 / dx_de)

        fname = (pic_run_dir + "field_hdf5/T." + str(tindex) +
                 "/fields_" + str(tindex) + ".h5")
        with h5py.File(fname, 'r') as fh:
            group = fh["Timestep_" + str(tindex)]
            dset = group["cbx"]
            bx = np.zeros(dset.shape, dtype=dset.dtype)
            dset.read_direct(bx)
        bx = np.squeeze(bx)

        bx_cut = bx[ix_xp, :]
        # print(bx_cut.min(), bx_cut.max())
        fdata = gaussian_filter(bx[ix_xp, :], sigma=3)
        ax.plot(zgrid, fdata, linewidth=2, label=labels[irun])
    ax.legend(loc=4, prop={'size': 16}, ncol=1,
              shadow=False, fancybox=False, frameon=True)
    ax.grid(True)
    ax.set_xlim([-30, 30])
    ax.set_ylim([-0.55, 0.55])
    twci = math.ceil(tindex * pic_info.dtwci / 0.1) * 0.1
    text1 = r'$t\Omega_{ci}=' + ("{%0.0f}" % twci) + '$'
    fig.suptitle(text1, fontsize=20)
    img_dir = '../img/rate_problem/bx_inflow/'
    mkdir_p(img_dir)
    fname = img_dir + "bx_inflow_" + str(tframe) + ".pdf"
    fig.savefig(fname)

    if show_plot:
        plt.show()
    else:
        plt.close()


def get_cmd_args():
    """Get command line arguments
    """
    # default_pic_run = 'mime400_Tb_T0_025'
    # default_pic_run = 'mime400_nb_n0_1'
    # default_pic_run = 'mime400_Tb_T0_1'
    # default_pic_run = 'mime400_Tb_T0_10'
    default_pic_run = 'mime400_Tb_T0_10_weak'
    # default_pic_run = 'mime400_Tb_T0_40'
    # default_pic_run = 'mime400_Tb_T0_40_nppc450'
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
    parser.add_argument('--bg', action="store", default='0.0', type=float,
                        help='guide field strength')
    parser.add_argument('--multi_runs', action="store_true", default=False,
                        help='whether to analyze multiple runs')
    parser.add_argument('--show_plot', action="store_true", default=False,
                        help='whether to show plot')
    parser.add_argument('--plot_absj', action="store_true", default=False,
                        help='whether to plot current density')
    parser.add_argument('--plot_jy', action="store_true", default=False,
                        help='whether to plot jy')
    parser.add_argument('--plot_bfield', action="store_true", default=False,
                        help='whether to plot magnetic field')
    parser.add_argument('--plot_bz_xcut', action="store_true", default=False,
                        help='whether to plot Bz cut along x in simulations with different beta')
    parser.add_argument('--plot_bx_zcut', action="store_true", default=False,
                        help='whether to plot Bx cut along z in simulations with different beta')
    parser.add_argument('--plot_vx_zcut', action="store_true", default=False,
                        help='whether to plot velocity cut along z in simulations')
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
    parser.add_argument('--calc_bxm_fix', action="store_true", default=False,
                        help="whether calculating Bxm at a fix z")
    parser.add_argument('--plot_bxm', action="store_true", default=False,
                        help="whether plotting Bxm")
    parser.add_argument('--plot_bxm_beta', action="store_true", default=False,
                        help="whether plotting Bxm for runs with different beta")
    parser.add_argument('--open_boundary', action="store_true", default=False,
                        help="whether runs are with open boundary")
    parser.add_argument('--plot_p_xcut', action="store_true", default=False,
                        help='whether to plot p cut along x in simulations with different beta')
    parser.add_argument('--plot_n_xcut', action="store_true", default=False,
                        help='whether to plot density cut along x in simulations with different beta')
    parser.add_argument('--plot_pn_xcut', action="store_true", default=False,
                        help='whether to plot change of pperp due to density change')
    parser.add_argument('--plot_econv', action="store_true", default=False,
                        help='whether to plot energy conversion')
    parser.add_argument('--firehose', action="store_true", default=False,
                        help='whether to firehose parameter')
    parser.add_argument('--firehose_zcut', action="store_true", default=False,
                        help='whether to firehose parameter cut along z')
    parser.add_argument('--pxyz', action="store_true", default=False,
                        help='whether to pxx, pyy, pzz cut along z')
    parser.add_argument('--plot_pres', action="store_true", default=False,
                        help='whether to plot pressure')
    parser.add_argument('--plot_temp', action="store_true", default=False,
                        help='whether to plot temperature')
    parser.add_argument('--gradx_p', action="store_true", default=False,
                        help='whether to plot the gradient of pressure along x')
    parser.add_argument('--jxb_x', action="store_true", default=False,
                        help='whether to plot the x-component of jxB')
    parser.add_argument('--fluid_ene', action="store_true", default=False,
                        help='whether to plot fluid energization')
    parser.add_argument('--fluid_ene_2d', action="store_true", default=False,
                        help='whether to plot 2D fluid energization')
    parser.add_argument('--comp_ene', action="store_true", default=False,
                        help='whether to plot compression energization')
    parser.add_argument('--rhox', action="store_true", default=False,
                        help='plot density around X-line')
    parser.add_argument('--plot_va', action="store_true", default=False,
                        help='plot Alfven speed')
    parser.add_argument('--pres_avg', action="store_true", default=False,
                        help='plot averaged pressure in the outflow region')
    parser.add_argument('--pres_in_cut', action="store_true", default=False,
                        help='plot pressure cut in the inflow region')
    parser.add_argument('--nfieldline', action="store_true", default=False,
                        help='plot density along field line')
    parser.add_argument('--calc_angle', action="store_true", default=False,
                        help='calculate the exhaust opening angle')
    parser.add_argument('--plot_angle', action="store_true", default=False,
                        help='plot the exhaust opening angle')
    parser.add_argument('--absj_vout', action="store_true", default=False,
                        help='plot current density and outflow velocity')
    parser.add_argument('--calc_peak_vout', action="store_true", default=False,
                        help='calculate peak outflow velocity')
    parser.add_argument('--evolution', action="store_true", default=False,
                        help='plot quantities evolving with simulation')
    parser.add_argument('--beta', action="store", default='0.25', type=float,
                        help='Plasma beta in the inflow region')
    parser.add_argument('--rates_low_beta', action="store_true", default=False,
                        help='whether calculates rate for low-beta plasmas')
    parser.add_argument('--rates_high_beta', action="store_true", default=False,
                        help='whether calculates rate for high-beta plasmas')
    parser.add_argument('--rates_scaling', action="store_true", default=False,
                        help='whether calculates rates scaling with beta')
    parser.add_argument('--rate_model', action="store_true", default=False,
                        help='whether calculates rates based on analytical models')
    parser.add_argument('--plot_rate_model', action="store_true", default=False,
                        help='whether plots rates based on analytical models')
    parser.add_argument('--rate_model_pub', action="store_true", default=False,
                        help='whether plots rates based on analytical models')
    parser.add_argument('--peak_rate', action="store_true", default=False,
                        help='whether to calculate peak rates')
    parser.add_argument('--plot_peak_rate', action="store_true", default=False,
                        help='whether plots peak rates')
    parser.add_argument('--outflow_heating', action="store_true", default=False,
                        help='Heating in outflow')
    parser.add_argument('--pres_avg_pub', action="store_true", default=False,
                        help='Average pressure in the outflow region for publication')
    parser.add_argument('--ptl_dist', action="store_true", default=False,
                        help='Particle distribution')
    parser.add_argument('--bx_inflow', action="store_true", default=False,
                        help='Compare Bx in the inflow region')
    parser.add_argument('--bx_edr', action="store_true", default=False,
                        help='Plot Bx upstream of the electron diffusion region')
    return parser.parse_args()


def analysis_single_frames(plot_config, args):
    """Analysis for multiple time frames
    """
    tframe = args.tframe
    if args.plot_absj:
        plot_absj(plot_config, args.show_plot)
    if args.plot_jy:
        plot_jy(plot_config, args.show_plot)
    elif args.plot_bfield:
        plot_bfield(plot_config, args.show_plot)
    elif args.plot_bz_xcut:
        plot_bz_xcut_beta(plot_config, args.show_plot)
    elif args.plot_bx_zcut:
        plot_bx_zcut_beta(plot_config, args.show_plot)
    elif args.plot_vx_zcut:
        plot_vx_zcut_beta(plot_config, args.show_plot)
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
        if args.open_boundary:
            plot_density(plot_config, args.show_plot)
        else:
            plot_density_cut(plot_config, args.show_plot)
    elif args.rrate_vin:
        calc_rrate_vin(plot_config, args.show_plot)
    elif args.rrate_bflux:
        calc_rrate_bflux(plot_config, args.show_plot)
    elif args.plot_rrate_bflux:
        if args.multi_runs:
            plot_rrate_bflux_beta(plot_config, args.bg, args.show_plot)
        else:
            plot_rrate_bflux(plot_config, args.show_plot)
    elif args.open_angle:
        open_angle(plot_config, args.show_plot)
    elif args.exhaust_boundary:
        get_exhaust_boundary(plot_config, args.show_plot)
    elif args.inflow_pressure:
        inflow_pressure(plot_config, args.show_plot)
    elif args.calc_bxm:
        calc_bxm(plot_config, args.show_plot)
    elif args.calc_bxm_fix:
        calc_bxm_fix(plot_config, args.show_plot)
    elif args.plot_bxm:
        plot_bxm(plot_config, args.show_plot)
    elif args.plot_bxm_beta:
        plot_bxm_beta(plot_config, args.show_plot)
    elif args.plot_p_xcut:
        plot_pres_xcut_beta(plot_config, args.show_plot)
    elif args.plot_pn_xcut:
        plot_pn_xcut_beta(plot_config, args.show_plot)
    elif args.plot_n_xcut:
        plot_density_xcut_beta(plot_config, args.show_plot)
    elif args.plot_econv:
        plot_energy_conversion(plot_config, args.show_plot)
    elif args.firehose:
        firehose_parameter(plot_config, args.show_plot)
    elif args.firehose_zcut:
        firehose_parameter_zcut(plot_config, args.show_plot)
    elif args.pxyz:
        pxyz_zcut(plot_config, args.show_plot)
    elif args.plot_pres:
        plot_pres(plot_config, args.show_plot)
    elif args.pres_avg:
        plot_pres_avg(plot_config, args.show_plot)
    elif args.pres_in_cut:
        plot_pres_inflow_cut(plot_config, args.show_plot)
    elif args.plot_temp:
        plot_temperature(plot_config, args.show_plot)
    elif args.gradx_p:
        gradx_pressure(plot_config, args.show_plot)
    elif args.jxb_x:
        plot_jxb_x(plot_config, args.show_plot)
    elif args.fluid_ene:
        fluid_energization(plot_config, args.show_plot)
    elif args.fluid_ene_2d:
        fluid_energization_2d(plot_config, args.show_plot)
    elif args.comp_ene:
        compression_energization(plot_config, args.show_plot)
    elif args.rhox:
        plot_density_xline(plot_config, args.show_plot)
    elif args.plot_va:
        plot_alfven_speed(plot_config, args.show_plot)
    elif args.nfieldline:
        plot_density_fieldline(plot_config, args.show_plot)
    elif args.calc_angle:
        calc_open_angle(plot_config, args.show_plot)
    elif args.plot_angle:
        plot_open_angle(plot_config, args.bg, args.show_plot)
    elif args.absj_vout:
        plot_absj_vout(plot_config, args.show_plot)
    elif args.calc_peak_vout:
        calc_peak_vout(plot_config, args.show_plot)
    elif args.evolution:
        # simulation_evolution(plot_config, args.bg, args.show_plot)
        simulation_evolution2(plot_config, args.bg, args.show_plot)
    if args.rates_low_beta:
        calc_rates_low_beta(0.22, 89)  # Liu et al. 2017 PRL
        calc_rates_low_beta(0.6, 0.25/8, nonrec=True)
    elif args.rates_high_beta:
        calc_rates_high_beta(0.72, 0.25**2/100, 0.1, 1.0)
        calc_rates_high_beta(0.75, 0.25**2/100, 1.0, 0.9)
        calc_rates_high_beta(0.9, 0.25**2/100, 10.0, 0.8)
        calc_rates_high_beta(0.62, 0.5**2/(400*0.02), 0.02, 1.0)
        calc_rates_high_beta(0.65, 0.5**2/(400*0.2), 0.2, 0.9)
    elif args.rates_scaling:
        plot_rates_scaling(0.5**2/400, 0.25)
        plot_rates_scaling(0.5**2/400, 1.0)
        plot_rates_scaling(0.5**2/400, 10.0)
        plot_rates_scaling(0.5**2/400, 40.0)
    elif args.rate_model:
        betas = [1E-3, 0.25, 1, 10, 40, 100]
        # betas = [1.0]
        for beta in betas:
            # rate_model(beta)
            # rate_model(beta, le_closure=True)
            # rate_model_fermi(beta)
            # rate_model_fermi(beta, le_closure=True)
            rate_model_fermi_incomp(beta)
            # calc_bxm_analytical(beta)
    elif args.plot_rate_model:
        plot_rate_model(le_closure=False)
        # plot_rate_model(le_closure=True)
    elif args.rate_model_pub:
        # plot_rate_model_pub(le_closure=False)
        # plot_rate_model_pub2(le_closure=False)
        plot_rate_model_pub3(le_closure=False)
    elif args.peak_rate:
        peak_rate_beta(le_closure=False)
    elif args.plot_peak_rate:
        plot_peak_rate_beta(le_closure=False)
    elif args.outflow_heating:
        outflow_heating_fermi(plot_config, args.show_plot)
    elif args.pres_avg_pub:
        # average_pressure_pub(plot_config, args.show_plot)
        # average_pressure_pub2(plot_config, args.show_plot)
        average_pressure_pub3(plot_config, args.show_plot)
    elif args.ptl_dist:
        particle_distribution(plot_config, args.show_plot)
    elif args.bx_inflow:
        plot_bx_inflow(plot_config, args.show_plot)
    elif args.bx_edr:
        plot_bx_edr(plot_config, args.show_plot)


def process_input(plot_config, args, tframe):
    """process one time frame"""
    plot_config["tframe"] = tframe
    if args.rrate_bflux:
        calc_rrate_bflux(plot_config, show_plot=False)
    elif args.exhaust_boundary:
        get_exhaust_boundary(plot_config, show_plot=False)
    elif args.calc_bxm:
        calc_bxm(plot_config, show_plot=False)
    elif args.calc_bxm_fix:
        calc_bxm_fix(plot_config, show_plot=False)
    elif args.calc_angle:
        calc_open_angle(plot_config, show_plot=False)
    elif args.calc_peak_vout:
        calc_peak_vout(plot_config, show_plot=False)


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
            if args.plot_jy:
                plot_jy(plot_config, show_plot=False)
            elif args.plot_bfield:
                plot_bfield(plot_config, show_plot=False)
            elif args.plot_ptensor:
                plot_pressure_tensor(plot_config, show_plot=False)
            elif args.plot_vout:
                plot_vout(plot_config, show_plot=False)
            elif args.plot_density:
                if args.open_boundary:
                    plot_density(plot_config, args.show_plot)
                else:
                    plot_density_cut(plot_config, args.show_plot)
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
            elif args.open_angle:
                open_angle(plot_config, show_plot=False)
            elif args.calc_bxm:
                calc_bxm(plot_config, show_plot=False)
            elif args.plot_bz_xcut:
                plot_bz_xcut_beta(plot_config, show_plot=False)
            elif args.plot_bx_zcut:
                plot_bx_zcut_beta(plot_config, show_plot=False)
            elif args.plot_p_xcut:
                plot_pres_xcut_beta(plot_config, show_plot=False)
            elif args.plot_n_xcut:
                plot_density_xcut_beta(plot_config, show_plot=False)
            elif args.plot_pres:
                plot_pres(plot_config, show_plot=False)
            elif args.firehose:
                firehose_parameter(plot_config, show_plot=False)
            elif args.firehose_zcut:
                firehose_parameter_zcut(plot_config, show_plot=False)
            elif args.gradx_p:
                gradx_pressure(plot_config, show_plot=False)
            elif args.jxb_x:
                plot_jxb_x(plot_config, show_plot=False)
            elif args.rhox:
                plot_density_xline(plot_config, show_plot=False)
            elif args.plot_va:
                plot_alfven_speed(plot_config, show_plot=False)
            elif args.pres_avg:
                plot_pres_avg(plot_config, show_plot=False)
            elif args.fluid_ene_2d:
                fluid_energization_2d(plot_config, show_plot=False)
            elif args.comp_ene:
                compression_energization(plot_config, show_plot=False)
            elif args.pres_in_cut:
                plot_pres_inflow_cut(plot_config, show_plot=False)
            elif args.outflow_heating:
                outflow_heating_fermi(plot_config, show_plot=False)
            elif args.pxyz:
                pxyz_zcut(plot_config, show_plot=False)
            elif args.bx_inflow:
                plot_bx_inflow(plot_config, show_plot=False)
    else:
        ncores = multiprocessing.cpu_count()
        ncores = 4
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
    plot_config["open_boundary"] = args.open_boundary
    if args.multi_frames:
        analysis_multi_frames(plot_config, args)
    else:
        analysis_single_frames(plot_config, args)


if __name__ == "__main__":
    main()
