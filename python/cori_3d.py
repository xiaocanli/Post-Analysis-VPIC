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
    fig = plt.figure(figsize=[7, 5])
    w1, h1 = 0.83, 0.8
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

        ene_bx /= enorm
        ene_by /= enorm
        ene_bz /= enorm
        ene_magnetic /= enorm
        kene_e /= enorm
        kene_i /= enorm

        line_style = "--" if "2D" in pic_run else "-"
        pdim = "2D" if "2D" in pic_run else "3D"

        ax.plot(tenergy, ene_magnetic, linewidth=2, linestyle=line_style,
                color=COLORS[0], label=r'$\varepsilon_B$' + '(' + pdim + ')')
        ax.plot(tenergy, kene_e, linewidth=2, linestyle=line_style,
                color=COLORS[1], label=r'$\varepsilon_e$' + '(' + pdim + ')')
        ax.plot(tenergy, kene_i, linewidth=2, linestyle=line_style,
                color=COLORS[2], label=r'$\varepsilon_i$' + '(' + pdim + ')')

    ax.tick_params(bottom=True, top=False, left=True, right=True)
    ax.legend(loc=6, prop={'size': 16}, ncol=1,
              shadow=False, fancybox=False, frameon=False)
    ax.set_xlim([0, 400])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel(r'$t\Omega_{ci}$', fontsize=20)
    ax.set_ylabel(r'$\text{Energy}/\varepsilon_{B0}$', fontsize=20)
    ax.tick_params(labelsize=16)

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
    cbar_ticks = [[np.linspace(tstarts[0][0], tends[0][0], 6),
                   np.linspace(15, tends[0][1], 6)],
                  [np.linspace(tstarts[1][0], tends[1][0], 5),
                   np.linspace(tstarts[1][1], tends[1][1], 5)]]
    pindex = [[-3.6, -6.0], [-4.0, -4.6]]
    pnorm = [[1E11, 2E16], [4E15, 5E16]]
    plow = [[538, 588], [558, 568]]
    phigh = [[638, 688], [658, 668]]
    fig = plt.figure(figsize=[14, 9])
    rect0 = [0.10, 0.54, 0.36, 0.44]
    emax = 1E3 if species == "e" else 2E3
    for irun, pic_run in enumerate(pic_runs):
        pic_run_dir = root_dir + pic_run + "/"
        rect = np.copy(rect0)
        rect[1] -= irun * (rect[3] + 0.02)
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
            color = plt.cm.jet((tframe - tstart)/float(ntp), 1)
            ax1.loglog(ebins, spect[3:], linewidth=2, color=color)

        if species == 'e':
            fpower = pnorm[irun][0] * ebins**pindex[irun][0]
            power_index = "{%0.1f}" % pindex[irun][0]
            pname = r'$\sim \varepsilon^{' + power_index + '}$'
            ax1.loglog(ebins[plow[irun][0]:phigh[irun][0]],
                       fpower[plow[irun][0]:phigh[irun][0]],
                       linewidth=2, color='k', label=pname)
            ax1.legend(loc=3, prop={'size': 20}, ncol=1,
                       shadow=False, fancybox=False, frameon=False)

        ax1.tick_params(bottom=True, top=True, left=True, right=True)
        ax1.tick_params(axis='x', which='minor', direction='in', top='on')
        ax1.tick_params(axis='x', which='major', direction='in')
        ax1.tick_params(axis='y', which='minor', direction='in')
        ax1.tick_params(axis='y', which='major', direction='in')
        ax1.set_xlim([1E-1, emax])
        if irun == 0:
            ax1.set_ylim([1E-1, 1E9])
        else:
            ax1.set_ylim([1E-1, 1E12])
        if irun == 0:
            ax1.tick_params(axis='x', labelbottom='off')
        else:
            ax1.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$', fontsize=20)
        ax1.set_ylabel(r'$f(\varepsilon)$', fontsize=20)
        ax1.tick_params(labelsize=16)
        rect_cbar = np.copy(rect)
        rect_cbar[0] += rect[2] + 0.01
        rect_cbar[2] = 0.01
        cax = fig.add_axes(rect_cbar)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.jet,
                                   norm=plt.Normalize(vmin=tstart * dtf,
                                                      vmax=tend * dtf))
        # fake up the array of the scalar mappable. Urgh...
        sm._A = []
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label(r'$t\Omega_{ci}$', fontsize=20)
        cbar.set_ticks(cbar_ticks[irun][0] * dtf)
        cbar.ax.tick_params(labelsize=16)

        pdim = "2D" if "2D" in pic_run else "3D"
        ax1.text(-0.2, 0.5, pdim, color='k',
                 fontsize=24, rotation='vertical',
                 bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax1.transAxes)

        rect[0] += rect[2] + 0.09
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
            color = plt.cm.jet((tframe - tstart)/float(ntp), 1)
            ax2.loglog(ebins, spect[3:], linewidth=2, color=color)

        if species == 'e':
            fpower = pnorm[irun][1] * ebins**pindex[irun][1]
            power_index = "{%0.1f}" % pindex[irun][1]
            pname = r'$\sim \varepsilon^{' + power_index + '}$'
            ax2.loglog(ebins[plow[irun][1]:phigh[irun][1]],
                       fpower[plow[irun][1]:phigh[irun][1]],
                       linewidth=2, color='k', label=pname)
            ax2.legend(loc=3, prop={'size': 20}, ncol=1,
                       shadow=False, fancybox=False, frameon=False)

        ax2.set_xlim([1E-1, emax])
        if irun == 0:
            ax2.set_ylim([1E-1, 1E9])
        else:
            ax2.set_ylim([1E-1, 1E12])
        ax2.tick_params(bottom=True, top=True, left=True, right=True)
        ax2.tick_params(axis='x', which='minor', direction='in', top='on')
        ax2.tick_params(axis='x', which='major', direction='in')
        ax2.tick_params(axis='y', which='minor', direction='in', left='on')
        ax2.tick_params(axis='y', which='major', direction='in')

        if irun == 0:
            ax2.tick_params(axis='x', labelbottom='off')
        else:
            ax2.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$', fontsize=20)
        ax2.tick_params(axis='y', labelleft='off')
        ax2.tick_params(labelsize=16)
        rect_cbar = np.copy(rect)
        rect_cbar[0] += rect[2] + 0.01
        rect_cbar[2] = 0.01
        cax = fig.add_axes(rect_cbar)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.jet,
                                   norm=plt.Normalize(vmin=tstart * dtf,
                                                      vmax=tend * dtf))
        # fake up the array of the scalar mappable. Urgh...
        sm._A = []
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label(r'$t\Omega_{ci}$', fontsize=20)
        cbar.set_ticks(cbar_ticks[irun][1] * dtf)
        cbar.ax.tick_params(labelsize=16)

    fdir = '../img/cori_3d/'
    mkdir_p(fdir)
    fname = fdir + species + 'spect.pdf'
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
    ix, iz = 0, 9
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


def plot_momentum_spectrum(species):
    """Plot momentum spectrum for each time frame
    """
    if species == 'e':
        vth = 0.1
    else:
        vth = 0.02
    gama = 1.0 / math.sqrt(1.0 - 3 * vth**2)
    eth = gama - 1.0
    ebins = np.logspace(-6, 4, 1000)
    pbins = np.sqrt((ebins + 1)**2 - 1)
    pbins /= np.sqrt((eth + 1)**2 - 1)
    particle_interval = 10.0  # in 1/wci

    tstart = 0
    tend = 88680
    tinterval = 2217
    ntp = (tend - tstart) // tinterval + 1
    # for tframe in range(ntp):
    for tframe in range(40, 41):
        fig = plt.figure(figsize=[7, 5])
        w1, h1 = 0.83, 0.8
        xs, ys = 0.96 - w1, 0.96 - h1
        ax = fig.add_axes([xs, ys, w1, h1])
        tindex = tinterval * tframe
        fname = "spectrum_combined/spectrum_" + species + "_" + str(tindex) + ".dat"
        spect = np.fromfile(fname, dtype=np.float32)
        ndata, = spect.shape
        spect[3:] /= np.gradient(pbins)
        color = plt.cm.jet(tframe/float(ntp), 1)
        ax.loglog(pbins, spect[3:], linewidth=3, color='k')
        fpower = 1E16*pbins**-6.2
        ax.loglog(pbins, fpower, linewidth=3, color='k')
        ax.set_xlabel(r'$p/p_\text{th}$', fontsize=20)
        ax.set_ylabel(r'$f(p)$', fontsize=20)
        ax.set_xlim([1E-1, 2E2])
        ax.set_ylim([1E-1, 1E12])
        ax.set_yticks(np.logspace(0, 12, num=7))
        ax.tick_params(labelsize=16)
        text = str(particle_interval * tframe) + r'$\Omega_{ci}^{-1}$'
        ax.text(0.7, 0.9, text, color='k', fontsize=32,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)
        # plt.show()
        fname = "img/mspect_" + species + "_" + str(tframe) + ".pdf"
        fig.savefig(fname)
        plt.close()


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
    """Plot slices of current density with indicated box regionx
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

    fdir = '../img/cori_3d/'
    mkdir_p(fdir)

    fig = plt.figure(figsize=[7, 10])
    rect = [0.12, 0.76, 0.70, 0.21]
    hgap, vgap = 0.02, 0.02

    nslices = len(xslices)
    for islice, ix in enumerate(xslices):
        ax = fig.add_axes(rect)
        print("x-slice %d" % ix)
        p1 = ax.imshow(absj[:, :, midx[ix]], extent=[ymin, ymax, zmin, zmax],
                       vmin=jmin, vmax=jmax,
                       cmap=plt.cm.coolwarm, aspect='auto',
                       origin='lower', interpolation='bicubic')
        ax.set_ylim([-15, 15])
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')
        if islice == nslices - 1:
            ax.set_xlabel(r'$y/d_i$', fontsize=20)
        else:
            ax.tick_params(axis='x', labelbottom='off')
        ax.set_ylabel(r'$z/d_i$', fontsize=20)
        ax.tick_params(labelsize=16)

        text1 = r'$x=' + ("{%0.1f}" % xdi[islice]) + 'd_i$'
        ax.text(0.02, 0.85, text1, color='k', fontsize=24,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)
        if islice == 0:
            for iy in range(len(yboxes)):
                plot_box([ydi[iy], z1_di], dx_di * box_size, ax, 'k')
        else:
            for iy in range(len(yboxes)):
                plot_box([ydi[iy], z0_di], dx_di * box_size, ax, 'k')

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
    ix_str = str(ix).zfill(4)

    fname = fdir + 'absJ_yz_boxes.jpg'
    fig.savefig(fname, dpi=200)
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
    fig1 = plt.figure(figsize=[12, 4])
    box1 = [0.08, 0.15, 0.42, 0.7]
    axs1 = []
    fig2 = plt.figure(figsize=[12, 4])
    box2 = [0.08, 0.15, 0.42, 0.7]
    axs2 = []
    fig3 = plt.figure(figsize=[12, 4])
    box3 = [0.08, 0.15, 0.42, 0.7]
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
        label1 = (r'$\boldsymbol{j}_{' + species + '\parallel}' +
                  r'\cdot\boldsymbol{E}_\parallel$')
        ax.plot(tfields, epara_ene, linewidth=2, label=label1)
        label2 = (r'$\boldsymbol{j}_{' + species + '\perp}' +
                  r'\cdot\boldsymbol{E}_\perp$')
        ax.plot(tfields, eperp_ene, linewidth=2, label=label2)
        label3 = r'$(\nabla\cdot\tensorsym{P}_' + species + r')\cdot\boldsymbol{v}_E$'
        ax.plot(tfields, ptensor_ene, linewidth=2, label=label3)
        label4 = (r'$\boldsymbol{j}_{' + species + r'-\text{agy}}' +
                  r'\cdot\boldsymbol{E}_\perp$')
        label6 = r'$dK_' + species + '/dt$'
        ax.plot(tenergy, dkene, linewidth=2, label=label6)
        ax.plot([0, tenergy.max()], [0, 0], color='k', linestyle='--')
        ax.set_xlim([0, np.max(tfields)])
        ax.set_ylim(ylim)
        ax.tick_params(labelsize=16)
        if irun == 0:
            ax.set_ylabel('Energization', fontsize=20)
        # else:
        #     ax.tick_params(axis='y', labelleft='off')
        pdim = "2D" if "2D" in pic_run else "3D"
        ax.text(0.02, 0.9, pdim, color='k', fontsize=24,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)
        ax.set_xlabel(r'$t\Omega_{ci}$', fontsize=20)

        ax = fig2.add_axes(box1)
        axs2.append(ax)
        COLORS = palettable.colorbrewer.qualitative.Set1_9.mpl_colors
        ax.set_prop_cycle('color', COLORS)
        ax.plot(tfields, curv_drift_dote, linewidth=2, label='Curvature')
        # ax.plot(tfields, bulk_curv_dote, linewidth=2, label='Bulk Curvature')
        ax.plot(tfields, grad_drift_dote, linewidth=2, label='Gradient')
        ax.plot(tfields, magnetization_dote, linewidth=2, label='Magnetization')
        # ax.plot(tfields, acc_drift_dote, linewidth=2, label='Polarization')
        label2 = (r'$\boldsymbol{j}_{' + species + '\perp}' +
                  r'\cdot\boldsymbol{E}_\perp$')
        ax.plot(tfields, eperp_ene, linewidth=2, label=label2)
        jdote_sum = (curv_drift_dote + grad_drift_dote +
                     magnetization_dote + jagy_dote + acc_drift_dote)
        # ax.plot(tfields, jdote_sum, linewidth=2)
        ax.plot([0, tenergy.max()], [0, 0], color='k', linestyle='--')
        ax.set_xlim([0, np.max(tfields)])
        ax.set_ylim(ylim)
        ax.tick_params(labelsize=16)
        if irun == 0:
            ax.set_ylabel('Energization', fontsize=20)
        # else:
        #     ax.tick_params(axis='y', labelleft='off')
        pdim = "2D" if "2D" in pic_run else "3D"
        ax.text(0.02, 0.9, pdim, color='k', fontsize=24,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)
        ax.set_xlabel(r'$t\Omega_{ci}$', fontsize=20)

        ax = fig3.add_axes(box1)
        axs3.append(ax)
        COLORS = palettable.colorbrewer.qualitative.Set1_9.mpl_colors
        ax.set_prop_cycle('color', COLORS)
        ax.plot(tfields, comp_ene, linewidth=2, label='Compression')
        ax.plot(tfields, shear_ene, linewidth=2, label='Shear')
        # label2 = (r'$\boldsymbol{j}_{' + species + '\perp}' +
        #           r'\cdot\boldsymbol{E}_\perp -' + 'n_' + species +
        #           'm_' + species + r'(d\boldsymbol{u}_' + species +
        #           r'/dt)\cdot\boldsymbol{v}_E$')
        # ax.plot(tfields, eperp_ene - acc_drift_dote, linewidth=2, label=label2)
        label2 = (r'$\boldsymbol{j}_{' + species + '\perp}' +
                  r'\cdot\boldsymbol{E}_\perp' + '$')
        ax.plot(tfields, eperp_ene, linewidth=2, label=label2)
        label4 = (r'$\boldsymbol{j}_{' + species + r'-\text{agy}}' +
                  r'\cdot\boldsymbol{E}_\perp$')
        ax.plot(tfields, jagy_dote, linewidth=2, label=label4)
        # jdote_sum = comp_ene + shear_ene + jagy_dote
        # ax.plot(tfields, jdote_sum, linewidth=2)
        ax.plot([0, tenergy.max()], [0, 0], color='k', linestyle='--')
        ax.set_xlim([0, np.max(tfields)])
        ax.set_ylim(ylim)
        ax.tick_params(labelsize=16)
        if irun == 0:
            ax.set_ylabel('Energization', fontsize=20)
        # else:
        #     ax.tick_params(axis='y', labelleft='off')
        pdim = "2D" if "2D" in pic_run else "3D"
        ax.text(0.02, 0.9, pdim, color='k', fontsize=24,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)
        ax.set_xlabel(r'$t\Omega_{ci}$', fontsize=20)

        box1[0] += box1[2] + 0.05

    axs1[0].legend(loc='upper center', prop={'size': 20}, ncol=4,
                   bbox_to_anchor=(1.1, 1.22),
                   shadow=False, fancybox=False, frameon=False)
    axs2[0].legend(loc='upper center', prop={'size': 20}, ncol=4,
                   bbox_to_anchor=(1.1, 1.22),
                   shadow=False, fancybox=False, frameon=False)
    axs3[0].legend(loc='upper center', prop={'size': 20}, ncol=4,
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


def div0(a, b):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0]

    From: http://stackoverflow.com/a/35696047/2561161

    """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[~np.isfinite(c)] = 0  # -inf inf NaN
    return c


def particle_energization(plot_config):
    """Particle-based energization

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    tframe = plot_config["tframe"]
    pic_run_dir = plot_config["pic_run_dir"]
    pic_runs = ["2D-Lx150-bg0.2-150ppc-16KNL", "3D-Lx150-bg0.2-150ppc-2048KNL"]

    tstarts = [6, 10, 20, 30]
    tends = [10, 20, 30, 40]
    nplots = len(tstarts)

    if plot_config["plot_type"] == 'total':
        if species == 'e':
            ylims = [[-0.001, 0.003], [-0.0005, 0.001],
                     [-0.0002, 0.0004], [-0.0002, 0.0004]]
        else:
            ylims = [[-0.001, 0.003], [-0.0005, 0.002],
                     [-0.0002, 0.0005], [-0.0002, 0.0004]]
        data_indices = [1, 2]
    if plot_config["plot_type"] == 'curvature':
        if species == 'e':
            ylims = [[-0.001, 0.004], [-0.0005, 0.0015],
                     [-0.0002, 0.0004], [-0.0002, 0.0005]]
        else:
            ylims = [[-0.001, 0.004], [-0.0005, 0.0015],
                     [-0.0002, 0.0003], [-0.0002, 0.0004]]
        data_indices = [5]
    if plot_config["plot_type"] == 'gradient':
        if species == 'e':
            ylims = [[-0.0002, 0.00025], [-0.0002, 0.00025],
                     [-0.0001, 0.0001], [-0.0001, 0.0001]]
        else:
            ylims = [[-0.0002, 0.00025], [-0.0002, 0.00025],
                     [-0.0001, 0.0001], [-0.0001, 0.0001]]
        data_indices = [6]
    if plot_config["plot_type"] == 'inertial':
        if species == 'e':
            ylims = [[-0.00075, 0.00025], [-0.0002, 0.0001],
                     [-0.0001, 0.0001], [-0.0002, 0.0001]]
        else:
            ylims = [[-0.0012, 0.0004], [-0.0008, 0.0001],
                     [-0.0001, 0.0001], [-0.0002, 0.0001]]
        data_indices = [11, 12]
    if plot_config["plot_type"] == 'polarization':
        if species == 'e':
            ylims = [[-0.00025, 0.00025], [-0.00025, 0.00025],
                     [-0.0001, 0.0001], [-0.0002, 0.0001]]
        else:
            ylims = [[-0.001, 0.002], [-0.0005, 0.0015],
                     [-0.0002, 0.0005], [-0.0002, 0.00025]]
        data_indices = [15, 16]
    if plot_config["plot_type"] == 'parallel_drift':
        if species == 'e':
            ylims = [[-0.00025, 0.0004], [-0.00005, 0.0001],
                     [-0.00005, 0.0001], [-0.0002, 0.0001]]
        else:
            ylims = [[-0.00002, 0.0002], [-0.00005, 0.0001],
                     [-0.00005, 0.0001], [-0.00005, 0.00005]]
        data_indices = [7]
    if plot_config["plot_type"] == 'mu':
        if species == 'e':
            ylims = [[-0.0003, 0.0002], [-0.0002, 0.0005],
                     [-0.0001, 0.0003], [-0.0001, 0.0003]]
        else:
            ylims = [[-0.0003, 0.0002], [-0.0002, 0.0005],
                     [-0.0001, 0.0003], [-0.0001, 0.0003]]
        data_indices = [8]
    if plot_config["plot_type"] == 'compression':
        if species == 'e':
            ylims = [[-0.001, 0.003], [-0.0005, 0.001],
                     [-0.0002, 0.0004], [-0.0002, 0.0004]]
        else:
            ylims = [[-0.001, 0.003], [-0.0005, 0.002],
                     [-0.0002, 0.0005], [-0.0002, 0.0004]]
        data_indices = [3]
    if plot_config["plot_type"] == 'shear':
        if species == 'e':
            ylims = [[-0.0005, 0.0015], [-0.00025, 0.0005],
                     [-0.0001, 0.0002], [-0.0001, 0.0002]]
        else:
            ylims = [[-0.0005, 0.0015], [-0.00025, 0.0005],
                     [-0.0001, 0.0002], [-0.0001, 0.0002]]
        data_indices = [4]


    for iplot in range(nplots):
        tstart = tstarts[iplot]
        tend = tends[iplot]
        ylim = ylims[iplot]
        fig1 = plt.figure(figsize=[12, 4])
        box1 = [0.12, 0.18, 0.39, 0.75]
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

                color = plt.cm.jet((tframe - tstart)/float(nframes), 1)
                fdata = np.zeros(nbins)
                for idata in data_indices:
                    fdata += fbins[idata, :]
                ax.semilogx(ebins, fdata, linewidth=2, color=color)
            if species == 'e':
                if "2D" in pic_run:
                    ax.set_xlim([1E0, 200])
                else:
                    ax.set_xlim([1E0, 500])
            else:
                if "2D" in pic_run:
                    ax.set_xlim([1E0, 500])
                else:
                    ax.set_xlim([1E0, 1000])
            ax.set_ylim(ylim)
            box1[0] += box1[2] + 0.02
            if irun == 0:
                ax.set_ylabel('Acceleration Rate', fontsize=20)
            else:
                ax.tick_params(axis='y', labelleft='off')

            ax.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$', fontsize=20)
            ax.tick_params(labelsize=16)

            pdim = "2D" if "2D" in pic_run else "3D"
            ax.text(0.02, 0.9, pdim, color='k', fontsize=24,
                    bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                    horizontalalignment='left', verticalalignment='center',
                    transform=ax.transAxes)
            ax.plot(ax.get_xlim(), [0, 0], linestyle='--', color='k')

            if irun == 1:
                box1[0] -= box1[2] + 0.02
                rect_cbar = np.copy(box1)
                rect_cbar[0] += box1[2] + 0.01
                rect_cbar[2] = 0.01
                cax = fig1.add_axes(rect_cbar)
                sm = plt.cm.ScalarMappable(cmap=plt.cm.jet,
                                           norm=plt.Normalize(vmin=tstart * dtp,
                                                              vmax=tend * dtp))
                # fake up the array of the scalar mappable. Urgh...
                sm._A = []
                cbar = fig1.colorbar(sm, cax=cax)
                cbar.set_label(r'$t\Omega_{ci}$', fontsize=20)
                cbar.set_ticks((np.linspace(tstart, tend, tend - tstart + 1) * dtp))
                cbar.ax.tick_params(labelsize=16)

        fdir = '../img/cori_3d/particle_energization/'
        mkdir_p(fdir)
        fname = (fdir + 'particle_' + plot_config["plot_type"] + '_' +
                 species + '_' + str(iplot) + '.pdf')
        fig1.savefig(fname)

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
    parser.add_argument('--tstart', action="store", default='0', type=int,
                        help='starting time frame')
    parser.add_argument('--tend', action="store", default='40', type=int,
                        help='ending time frame')
    parser.add_argument('--whole_spectrum', action="store_true", default=False,
                        help='whether to plot spectrum in the whole box')
    parser.add_argument('--all_frames', action="store_true", default=False,
                        help='whether to analyze all frames')
    parser.add_argument('--local_spectrum', action="store_true", default=False,
                        help='whether to plot local spectrum')
    parser.add_argument('--mom_spectrum', action="store_true", default=False,
                        help='whether to plot momentum spectrum')
    parser.add_argument('--econv', action="store_true", default=False,
                        help='whether to plot energy conversion')
    parser.add_argument('--show_plot', action="store_true", default=False,
                        help='whether to show plot')
    parser.add_argument('--jslice', action="store_true", default=False,
                        help='whether to plot slices of current density')
    parser.add_argument('--jslice_box', action="store_true", default=False,
                        help='whether to plot slices of current density with boxes')
    parser.add_argument('--fluid_ene', action="store_true", default=False,
                        help='whether to plot fluid energization')
    parser.add_argument('--particle_ene', action="store_true", default=False,
                        help='whether to plot particle energization')
    parser.add_argument('--plot_type', action="store", default='total', type=str,
                        help='Particle plot type')
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
    elif args.local_spectrum:
        plot_local_spectrum(plot_config)
    elif args.mom_spectrum:
        plot_momentum_spectrum(args.species)
    elif args.econv:
        energy_conversion(plot_config)
    elif args.jslice:
        plot_jslice(plot_config)
    elif args.jslice_box:
        plot_jslice_box(plot_config)
    elif args.fluid_ene:
        fluid_energization(plot_config)
    elif args.particle_ene:
        particle_energization(plot_config)


def process_input(plot_config, args, tframe):
    """process one time frame"""
    plot_config["tframe"] = tframe
    pass


def analysis_multi_frames(plot_config, args):
    """Analysis for multiple time frames
    """
    tframes = range(plot_config["tmin"], plot_config["tmax"] + 1)
    ncores = multiprocessing.cpu_count()
    # Parallel(n_jobs=ncores)(delayed(process_input)(plot_config, args, tframe)
    #                         for tframe in tframes)


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
