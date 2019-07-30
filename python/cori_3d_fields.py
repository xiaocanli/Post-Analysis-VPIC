#!/usr/bin/env python3
"""
Fields plot for the Cori 3D runs
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
from evtk.hl import gridToVTK, pointsToVTK
from matplotlib.colors import LogNorm, SymLogNorm
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy.interpolate import interp1d, interp2d, RectBivariateSpline

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

    fdir = '../img/cori_3d/absJ/' + pic_run + '/tframe_' + str(tframe) + '/'
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


def reconnection_layer(plot_config, show_plot=True):
    """Get reconnection layer boundary
    """
    bg = plot_config["bg"]
    bg_str = str(int(bg * 10)).zfill(2)
    tframe = plot_config["tframe"]
    pic_run = "3D-Lx150-bg" + str(bg) + "-150ppc-2048KNL"
    root_dir = "/net/scratch3/xiaocanli/reconnection/Cori_runs/"
    pic_run_dir = root_dir + pic_run + "/"
    species = plot_config["species"]
    tframe = plot_config["tframe"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    dz_zone = pic_info.nz_zone * pic_info.dz_di

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
    pic_topox = pic_info.topology_x
    pic_topoy = pic_info.topology_y
    pic_topoz = pic_info.topology_z
    nx, ny, nz = pic_info.nx, pic_info.ny, pic_info.nz
    nxr2, nyr2, nzr2 = nx // 2, ny // 2, nz // 2
    nxr4, nyr4, nzr4 = nx // 4, ny // 4, nz // 4
    nx_pic = nx // pic_topox
    ny_pic = ny // pic_topoy
    nz_pic = nz // pic_topoz
    xmin, xmax = 0, pic_info.lx_di
    ymin, ymax = -pic_info.ly_di * 0.5, pic_info.ly_di * 0.5
    zmin, zmax = -pic_info.lz_di * 0.5, pic_info.lz_di * 0.5
    nmin, nmax = 0.5, 2.0
    jmin, jmax = 0.0, 0.4

    fname = pic_run_dir + "data-smooth/n" + species + "_" + str(tindex) + ".gda"
    nrho = np.fromfile(fname, dtype=np.float32)
    nrho = nrho.reshape((nzr2, nyr2, nxr2))
    fname = pic_run_dir + "data-smooth/absJ_" + str(tindex) + ".gda"
    absJ = np.fromfile(fname, dtype=np.float32)
    absJ = absJ.reshape((nzr2, nyr2, nxr2))
    x_di = pic_info.x_di
    y_di = pic_info.y_di
    z_di = pic_info.z_di
    xr2_di = x_di[1::2]
    yr2_di = y_di[1::2]
    zr2_di = z_di[1::2]
    xr4_di = x_di[2::4]
    yr4_di = y_di[2::4]
    zr4_di = z_di[2::4]

    nbands = 7
    nhigh = np.zeros((nzr4, nyr4, nxr4))
    nrhos = []
    for iband in range(nbands):
        print("Energy band: %d" % iband)
        fname = (pic_run_dir + "data-smooth2/n" + species + "_" +
                 str(iband) + "_" + str(tindex) + ".gda")
        nrho = np.fromfile(fname, dtype=np.float32)
        nrho = nrho.reshape((nzr4, nyr4, nxr4))
        nrhos.append(nrho)
        if iband >= 1:
            nhigh += nrho

    ng = 1
    kernel = np.ones((ng,ng)) / float(ng*ng)

    fig = plt.figure(figsize=[9, 4])
    rect = [0.10, 0.16, 0.75, 0.8]
    ax = fig.add_axes(rect)
    cs1_surface = np.zeros([nyr4, nx])
    cs2_surface = np.zeros([nyr4, nx])
    cs1_new = np.zeros([nx, 2])
    cs2_new = np.zeros([nx, 2])
    cs1_new[:, 0] = x_di
    cs2_new[:, 0] = x_di
    fdir = '../img/cori_3d/reconnection_layer/' + pic_run + '/'
    fdir += 'tframe_' + str(tframe) + '/'
    # mkdir_p(fdir)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([zmin, zmax])
    tframe_tran = 7 if bg_str == '02' else 9
    for iy in range(0, nyr4):
        print("y-slice %d" % iy)
        if tframe <= tframe_tran:
            fdata = absJ[1::2, iy*2, 1::2]
            levels = [0.03]
        else:
            fdata = nhigh[:, iy, :]
            levels = [1E-5]
        cs = ax.contour(xr4_di, zr4_di, fdata, colors='k',
                        linewidths=0.5, levels=levels)
        for cl in cs.collections:
            cs_lengths = np.zeros(len(cl.get_paths()))
            for index, p in enumerate(cl.get_paths()):
                cs_lengths[index] = len(p.vertices)
        sorted_indices = np.argsort(cs_lengths)
        cs1 = cl.get_paths()[sorted_indices[-1]].vertices
        cs2 = cl.get_paths()[sorted_indices[-2]].vertices
        plt.cla()
        # p1 = ax.imshow(fdata, extent=[xmin, xmax, zmin, zmax],
        #                vmin=0, vmax=0.4,
        #                cmap=plt.cm.coolwarm, aspect='auto',
        #                origin='lower', interpolation='bicubic')
        # ax.plot(cs1[:, 0], cs1[:, 1], color='k')
        # ax.plot(cs2[:, 0], cs2[:, 1], color='k')
        f = interp1d(cs1[:, 0], cs1[:, 1])
        cs1_new[3:-3, 1] = f(cs1_new[3:-3, 0])
        cs1_new[:3] = cs1_new[3]
        cs1_new[-3:] = cs1_new[-4]
        f = interp1d(cs2[:, 0], cs2[:, 1])
        cs2_new[3:-3, 1] = f(cs2_new[3:-3, 0])
        cs2_new[:3] = cs2_new[3]
        cs2_new[-3:] = cs2_new[-4]
        if np.all(cs1_new[:, 1] > cs2_new[:, 1]):
            cs1_surface[iy, :] = cs1_new[:, 1]
            cs2_surface[iy, :] = cs2_new[:, 1]
        else:
            cs1_surface[iy, :] = cs2_new[:, 1]
            cs2_surface[iy, :] = cs1_new[:, 1]

        # ax.plot(cs1_new[:, 0], cs1_new[:, 1], color='k')
        # ax.plot(cs2_new[:, 0], cs2_new[:, 1], color='k')
        # fname = fdir + 'yslice_' + str(iy) + '.pdf'
        # fig.savefig(fname)
        # plt.show()

    plt.close()

    X, Y = np.meshgrid(x_di, yr4_di)
    X_new, Y_new = np.meshgrid(x_di, y_di)
    f = RectBivariateSpline(yr4_di, x_di, cs1_surface)
    cs1_surface_new = f(y_di, x_di)
    f = RectBivariateSpline(yr4_di, x_di, cs2_surface)
    cs2_surface_new = f(y_di, x_di)

    # save data
    fdir = pic_run_dir + 'reconnection_layer/'
    mkdir_p(fdir)
    fname = fdir + 'rec_layer_' + str(tframe) + '.h5'
    fh = h5py.File(fname, 'w')
    grp = fh.create_group('rec_layer')
    grp.create_dataset('Top', (ny, nx), data=cs1_surface_new)
    grp.create_dataset('Bottom', (ny, nx), data=cs2_surface_new)
    grp = fh.create_group('rec_layer_r2')
    fdata = np.mean(np.mean(cs1_surface_new.reshape([nyr2, 2, nxr2, 2]),
                            axis=3), axis=1)
    grp.create_dataset('Top', (nyr2, nxr2), data=fdata)
    fdata = np.mean(np.mean(cs2_surface_new.reshape([nyr2, 2, nxr2, 2]),
                            axis=3), axis=1)
    grp.create_dataset('Bottom', (nyr2, nxr2), data=fdata)
    grp = fh.create_group('rec_layer_r4')
    fdata = np.mean(np.mean(cs1_surface_new.reshape([nyr4, 4, nxr4, 4]),
                            axis=3), axis=1)
    grp.create_dataset('Top', (nyr4, nxr4), data=fdata)
    fdata = np.mean(np.mean(cs2_surface_new.reshape([nyr4, 4, nxr4, 4]),
                            axis=3), axis=1)
    grp.create_dataset('Bottom', (nyr4, nxr4), data=fdata)
    grp = fh.create_group('rec_layer_pic')
    fdata = np.mean(np.mean(cs1_surface_new.reshape([pic_topoy, ny_pic, pic_topox, nx_pic]),
                            axis=3), axis=1)
    grp.create_dataset('Top', (pic_topoy, pic_topox), data=fdata)
    fdata = np.mean(np.mean(cs2_surface_new.reshape([pic_topoy, ny_pic, pic_topox, nx_pic]),
                            axis=3), axis=1)
    grp.create_dataset('Bottom', (pic_topoy, pic_topox), data=fdata)
    grp = fh.create_group('rec_layer_zone')
    fdata = np.mean(np.mean(cs1_surface_new.reshape([pic_topoy, ny_pic, pic_topox, nx_pic]),
                            axis=3), axis=1)
    fdata = np.ceil((fdata - zmin) / dz_zone).astype(np.int32)
    grp.create_dataset('Top', (pic_topoy, pic_topox), data=fdata)
    fdata = np.mean(np.mean(cs2_surface_new.reshape([pic_topoy, ny_pic, pic_topox, nx_pic]),
                            axis=3), axis=1)
    fdata = np.floor((fdata - zmin) / dz_zone).astype(np.int32)
    grp.create_dataset('Bottom', (pic_topoy, pic_topox), data=fdata)
    fh.close()


def plot_reconnection_layer(plot_config, show_plot=True):
    """Plot reconnection layer boundary for the 3D simulations
    """
    bg = plot_config["bg"]
    tframe = plot_config["tframe"]
    pic_run = "3D-Lx150-bg" + str(bg) + "-150ppc-2048KNL"
    root_dir = "/net/scratch3/xiaocanli/reconnection/Cori_runs/"
    pic_run_dir = root_dir + pic_run + "/"
    species = plot_config["species"]
    tframe = plot_config["tframe"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    x_di = pic_info.x_di
    y_di = pic_info.y_di
    z_di = pic_info.z_di
    nx, = x_di.shape
    ny, = y_di.shape
    X_new, Y_new = np.meshgrid(x_di, y_di)
    cs1_surface_new = np.zeros((ny, nx))
    cs2_surface_new = np.zeros((ny, nx))

    fdir = pic_run_dir + 'reconnection_layer/'
    fname = fdir + 'rec_layer_' + str(tframe) + '.h5'
    fh = h5py.File(fname, 'r')
    grp = fh['rec_layer']
    grp['Top'].read_direct(cs1_surface_new)
    grp['Bottom'].read_direct(cs2_surface_new)
    fh.close()

    fig = plt.figure(figsize=[7, 5])
    rect = [0.10, 0.16, 0.75, 0.8]
    ax = fig.add_axes(rect, projection='3d')
    # ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X_new, Y_new, cs1_surface_new, cmap=plt.cm.coolwarm,
                           linewidth=0, antialiased=False)
    surf = ax.plot_surface(X_new, Y_new, cs2_surface_new, cmap=plt.cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.set_xlim([x_di.min(), x_di.max()])
    ax.set_ylim([y_di.min(), y_di.max()])
    ax.set_zlim([z_di.min(), z_di.max()])

    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlabel(r'$x/d_i$', fontsize=20)
    ax.set_ylabel(r'$y/d_i$', fontsize=20)
    ax.set_zlabel(r'$z/d_i$', fontsize=20)
    ax.tick_params(labelsize=16)
    fdir = '../img/cori_3d/reconnection_layer/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + 'rec_layer_' + str(tframe) + '.jpg'
    fig.savefig(fname, dpi=300)

    if show_plot:
        plt.show()
    else:
        plt.close()


def convert_layer_vtk(plot_config, show_plot=True):
    """Convert reconnection layer boundary data to vtk format
    """
    bg = plot_config["bg"]
    tframe = plot_config["tframe"]
    pic_run = "3D-Lx150-bg" + str(bg) + "-150ppc-2048KNL"
    root_dir = "/net/scratch3/xiaocanli/reconnection/Cori_runs/"
    pic_run_dir = root_dir + pic_run + "/"
    species = plot_config["species"]
    tframe = plot_config["tframe"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    x_di = pic_info.x_di
    y_di = pic_info.y_di
    z_di = pic_info.z_di
    nx, = x_di.shape
    ny, = y_di.shape
    xmesh, ymesh = np.meshgrid(x_di, y_di)
    cs1_surface = np.zeros((ny, nx))
    cs2_surface = np.zeros((ny, nx))
    fdata = np.zeros((ny, nx, 1))
    xdata = xmesh.flatten()
    ydata = ymesh.flatten()
    smime = math.sqrt(pic_info.mime)
    xmesh = np.atleast_3d(xmesh) * smime
    ymesh = np.atleast_3d(ymesh) * smime

    print("Time frame: %d" % tframe)
    fdir = pic_run_dir + 'reconnection_layer/'
    if tframe > 0:
        fname = fdir + 'rec_layer_' + str(tframe) + '.h5'
        fh = h5py.File(fname, 'r')
        grp = fh['rec_layer']
        grp['Top'].read_direct(cs1_surface)
        grp['Bottom'].read_direct(cs2_surface)
        fh.close()
    else:
        cs1_surface[:, :] = smime  # in de
        cs2_surface[:, :] = -smime  # in de

    cs1_surface_3d = np.atleast_3d(cs1_surface) * smime
    cs2_surface_3d = np.atleast_3d(cs2_surface) * smime

    tindex = tframe * pic_info.fields_interval
    fname = fdir + 'rec_layer_top_' + str(tindex)
    gridToVTK(fname, xmesh, ymesh, cs1_surface_3d,
              cellData = {"top" : fdata})
    fname = fdir + 'rec_layer_bottom_' + str(tindex)
    gridToVTK(fname, xmesh, ymesh, cs2_surface_3d,
              cellData = {"bottom" : fdata})


def reconnection_layer_2d(plot_config, show_plot=True):
    """Get reconnection layer boundary
    """
    bg = plot_config["bg"]
    tframe = plot_config["tframe"]
    pic_run = "2D-Lx150-bg" + str(bg) + "-150ppc-16KNL"
    root_dir = "/net/scratch3/xiaocanli/reconnection/Cori_runs/"
    pic_run_dir = root_dir + pic_run + "/"
    species = plot_config["species"]
    tframe = plot_config["tframe"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_topox = pic_info.topology_x
    pic_topoy = pic_info.topology_y
    pic_topoz = pic_info.topology_z
    nx, ny, nz = pic_info.nx, pic_info.ny, pic_info.nz
    nxr2, nyr2, nzr2 = nx // 2, ny // 2, nz // 2
    nxr4, nyr4, nzr4 = nx // 4, ny // 4, nz // 4
    nx_pic = nx // pic_topox
    ny_pic = ny // pic_topoy
    nz_pic = nz // pic_topoz
    xmin, xmax = 0, pic_info.lx_di
    ymin, ymax = -pic_info.ly_di * 0.5, pic_info.ly_di * 0.5
    zmin, zmax = -pic_info.lz_di * 0.5, pic_info.lz_di * 0.5
    x_di = pic_info.x_di
    y_di = pic_info.y_di
    z_di = pic_info.z_di
    dz_zone = pic_info.nz_zone * pic_info.dz_di
    kwargs = {"current_time": tframe,
              "xl": 0, "xr": pic_info.lx_di,
              "zb": -pic_info.lz_di, "zt": pic_info.lz_di}
    fname = pic_run_dir + "data/Ay.gda"
    x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/absJ.gda"
    x, z, absJ = read_2d_fields(pic_info, fname, **kwargs)
    fig = plt.figure(figsize=[7, 3.5])
    box1 = [0.13, 0.18, 0.82, 0.78]
    ax = fig.add_axes(box1)
    level_target = 10
    for level in range(50, int(Ay.max())):
        cs = ax.contour(x, z, Ay, colors='k', linewidths=0.5, levels=[level])
        cl, = cs.collections
        cl, = cs.collections
        cs1 = cl.get_paths()[0].vertices
        cs2 = cl.get_paths()[1].vertices
        if len(cl.get_paths()) > 2:
            level_target = level - 1
            break
        else:
            if cs1[:, 0].max() <= cs2[:, 0].min() or cs2[:, 0].max() <= cs1[:, 0].min():
                level_target = level - 1
                break
    plt.cla()
    cs = ax.contour(x, z, Ay, colors='k', linewidths=0.5, levels=[level_target])
    cl, = cs.collections
    cs1 = cl.get_paths()[0].vertices
    cs2 = cl.get_paths()[1].vertices
    cs1_new = np.zeros(nx)
    cs2_new = np.zeros(nx)
    f = interp1d(cs1[:, 0], cs1[:, 1])
    cs1_new[1:-1] = f(x_di[1:-1])
    cs1_new[0] = cs1_new[1]
    cs1_new[-2] = cs1_new[-1]
    f = interp1d(cs2[:, 0], cs2[:, 1])
    cs2_new[1:-1] = f(x_di[1:-1])
    cs2_new[0] = cs2_new[1]
    cs2_new[-2] = cs2_new[-1]
    plt.cla()
    ax.plot(x_di, cs1_new, color='k', linestyle='--')
    ax.plot(x_di, cs2_new, color='k')
    p1 = ax.imshow(absJ, extent=[xmin, xmax, zmin, zmax],
                   vmin=0, vmax=0.4,
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
    fdir = '../img/cori_3d/reconnection_layer/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + 'rec_layer_' + str(tframe) + '.jpg'
    fig.savefig(fname, dpi=300)
    if show_plot:
        plt.show()
    else:
        plt.close()

    # save data
    fdir = pic_run_dir + 'reconnection_layer/'
    mkdir_p(fdir)
    fname = fdir + 'rec_layer_' + str(tframe) + '.h5'
    fh = h5py.File(fname, 'w')
    grp = fh.create_group('rec_layer')
    grp.create_dataset('Top', (1, nx), data=cs2_new)
    grp.create_dataset('Bottom', (1, nx), data=cs1_new)
    grp = fh.create_group('rec_layer_r2')
    fdata = np.mean(cs2_new.reshape([nxr2, 2]), axis=1)
    grp.create_dataset('Top', (1, nxr2), data=fdata)
    fdata = np.mean(cs1_new.reshape([nxr2, 2]), axis=1)
    grp.create_dataset('Bottom', (1, nxr2), data=fdata)
    grp = fh.create_group('rec_layer_r4')
    fdata = np.mean(cs2_new.reshape([nxr4, 4]), axis=1)
    grp.create_dataset('Top', (1, nxr4), data=fdata)
    fdata = np.mean(cs1_new.reshape([nxr4, 4]), axis=1)
    grp.create_dataset('Bottom', (1, nxr4), data=fdata)
    grp = fh.create_group('rec_layer_pic')
    fdata = np.mean(cs2_new.reshape([pic_topox, nx_pic]), axis=1)
    grp.create_dataset('Top', (1, pic_topox), data=fdata)
    fdata = np.mean(cs1_new.reshape([pic_topox, nx_pic]), axis=1)
    grp.create_dataset('Bottom', (1, pic_topox), data=fdata)
    grp = fh.create_group('rec_layer_zone')
    fdata = np.mean(cs2_new.reshape([pic_topox, nx_pic]), axis=1)
    fdata = np.ceil((fdata - zmin) / dz_zone).astype(np.int32)
    grp.create_dataset('Top', (1, pic_topox), data=fdata)
    fdata = np.mean(cs1_new.reshape([pic_topox, nx_pic]), axis=1)
    fdata = np.floor((fdata - zmin) / dz_zone).astype(np.int32)
    grp.create_dataset('Bottom', (1, pic_topox), data=fdata)
    fh.close()


def calc_magnetic_flux(plot_config):
    """calculate magnetic flux in the inflow region for the 3D simulations
    """
    bg = plot_config["bg"]
    tframe = plot_config["tframe"]
    pic_run = "3D-Lx150-bg" + str(bg) + "-150ppc-2048KNL"
    root_dir = "/net/scratch3/xiaocanli/reconnection/Cori_runs/"
    pic_run_dir = root_dir + pic_run + "/"
    species = plot_config["species"]
    tframe = plot_config["tframe"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    x_di = pic_info.x_di
    y_di = pic_info.y_di
    z_di = pic_info.z_di
    x_di_r2 = x_di[1::2]
    y_di_r2 = y_di[1::2]
    z_di_r2 = z_di[1::2]
    dx_di_r2 = pic_info.dx_di * 2  # reduced grid
    dy_di_r2 = pic_info.dy_di * 2
    dz_di_r2 = pic_info.dz_di * 2
    zmin, zmax = z_di[0], z_di[-1]
    nx, = x_di.shape
    ny, = y_di.shape
    nz, = z_di.shape
    nxr2 = nx // 2
    nyr2 = ny // 2
    nzr2 = nz // 2
    X_new, Y_new = np.meshgrid(x_di_r2, y_di_r2)
    cs1_surface_new = np.zeros((nyr2, nxr2))
    cs2_surface_new = np.zeros((nyr2, nxr2))

    fdir = pic_run_dir + 'reconnection_layer/'
    mkdir_p(fdir)
    fname = fdir + 'rec_layer_' + str(tframe) + '.h5'
    fh = h5py.File(fname, 'r')
    grp = fh['rec_layer_r2']
    grp['Top'].read_direct(cs1_surface_new)
    grp['Bottom'].read_direct(cs2_surface_new)
    fh.close()
    cs1_surface_new = np.ceil((cs1_surface_new - zmin) / dz_di_r2).astype(np.int)
    cs2_surface_new = np.floor((cs2_surface_new - zmin) / dz_di_r2).astype(np.int)

    tindex = tframe * pic_info.fields_interval
    fname = pic_run_dir + "data-smooth/bx_" + str(tindex) + ".gda"
    bx = np.fromfile(fname, dtype=np.float32)
    bx = bx.reshape((nzr2, nyr2, nxr2))

    zgrid = np.linspace(0, nzr2-1, nzr2, dtype=np.int)
    zmesh = np.zeros((nzr2, nyr2, nxr2))
    for iz in range(nzr2):
        zmesh[iz, :, :] = zgrid[iz]

    bx_flux = np.zeros([2, nxr2])

    cond = zmesh > cs1_surface_new
    bx_cond = bx * cond
    cell_area = dy_di_r2 * dz_di_r2 * pic_info.mime  # in de^2
    bx_flux[0, :] = np.sum(np.sum(bx_cond, axis=1), axis=0) * cell_area

    cond = zmesh < cs2_surface_new
    bx_cond = bx * cond
    bx_flux[1, :] = np.sum(np.sum(bx_cond, axis=1), axis=0) * cell_area

    fdir = '../data/cori_3d/bx_flux/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + 'bx_flux_' + str(tframe) + '.dat'
    bx_flux.tofile(fname)


def plot_reconnection_rate(plot_config, show_plot=True):
    """calculate magnetic flux in the inflow region for the 3D simulations
    """
    bg = plot_config["bg"]
    tframe = plot_config["tframe"]
    pic_run = "3D-Lx150-bg" + str(bg) + "-150ppc-2048KNL"
    root_dir = "/net/scratch3/xiaocanli/reconnection/Cori_runs/"
    pic_run_dir = root_dir + pic_run + "/"
    species = plot_config["species"]
    tstart = plot_config["tstart"]
    tend = plot_config["tend"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    nx, = pic_info.x_di.shape
    ny, = pic_info.y_di.shape
    nz, = pic_info.z_di.shape
    nxr2 = nx // 2
    nyr2 = ny // 2
    nzr2 = nz // 2
    dtf = pic_info.fields_interval * pic_info.dtwpe
    b0 = pic_info.b0
    va = pic_info.dtwce * math.sqrt(1.0 / pic_info.mime) / pic_info.dtwpe

    nframes = tend - tstart + 1
    tframes = np.asarray(range(tstart, tend + 1))
    twci = tframes * math.ceil(pic_info.dt_fields)
    bx_fluxes = np.zeros([2, nframes, nxr2])

    for tframe in tframes:
        fdir = '../data/cori_3d/bx_flux/' + pic_run + '/'
        fname = fdir + 'bx_flux_' + str(tframe) + '.dat'
        bx_flux = np.fromfile(fname)
        bx_fluxes[:, tframe-tstart, :] = bx_flux.reshape([2, -1])

    bx_flux_diff = np.gradient(bx_fluxes, axis=1) / dtf
    rrate_mean = np.mean(bx_flux_diff, axis=2).T
    rrate_std = np.std(bx_flux_diff, axis=2).T
    rrate_norm = b0 * va * pic_info.ly_di * math.sqrt(pic_info.mime)
    rrate_mean /= rrate_norm
    rrate_std /= rrate_norm

    COLORS = palettable.tableau.Tableau_10.mpl_colors
    fig = plt.figure(figsize=[7, 5])
    rect = [0.12, 0.12, 0.83, 0.8]
    ax = fig.add_axes(rect)
    ax.errorbar(twci, np.abs(rrate_mean[:, 0]), yerr=rrate_std[:, 0],
                linestyle='None', marker='o', color=COLORS[0], capsize=3)
    ax.errorbar(twci, np.abs(rrate_mean[:, 1]), yerr=rrate_std[:, 0],
                linestyle='None', marker='o', color=COLORS[1], capsize=3)
    ax.set_xlabel(r'$t\Omega_{ci}$', fontsize=16)
    ax.set_ylabel(r'$R$', fontsize=16)
    ax.tick_params(bottom=True, top=False, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top=True)
    ax.tick_params(axis='x', which='major', direction='in', top=True)
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.tick_params(labelsize=12)
    ax.text(0.95, 0.87, 'Top', color=COLORS[0], fontsize=16,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='right', verticalalignment='bottom',
            transform=ax.transAxes)
    ax.text(0.95, 0.8, 'Bottom', color=COLORS[1], fontsize=16,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='right', verticalalignment='bottom',
            transform=ax.transAxes)

    fdir = '../img/cori_3d/rrate/'
    mkdir_p(fdir)
    bg_str = str(int(bg * 10)).zfill(2)
    fname = fdir + 'rrate_bg_' + bg_str + '.pdf'
    fig.savefig(fname)

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_reconnection_rate_2d(plot_config):
    """Plot reconnection rate for the 2D simulation

    Args:
        run_dir: the run root directory
        run_name: PIC run name
    """
    bg = plot_config['bg']
    bg_str = str(int(bg * 10)).zfill(2)
    pic_run = "2D-Lx150-bg" + str(bg) + "-150ppc-16KNL-long"
    root_dir = "/net/scratch3/xiaocanli/reconnection/Cori_runs/"
    pic_run_dir = root_dir + pic_run + "/"
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    ntf = pic_info.ntf
    phi = np.zeros(ntf)
    fname = pic_run_dir + 'data/Ay.gda'
    for tframe in range(ntf):
        kwargs = {"current_time": tframe,
                  "xl": 0, "xr": pic_info.lx_di,
                  "zb": -pic_info.lz_di*0.1, "zt": pic_info.lz_di*0.1}
        x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
        nz, = z.shape
        max_ay = np.max(Ay[nz // 2 - 1:nz // 2 + 1, :])
        min_ay = np.min(Ay[nz // 2 - 1:nz // 2 + 1, :])
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

    fdir = '../img/cori_3d/rrate/'
    mkdir_p(fdir)
    bg_str = str(int(bg * 10)).zfill(2)
    fname = fdir + 'rrate_bg_' + bg_str + '_2d.pdf'
    fig.savefig(fname)

    plt.show()


def plot_absj_2d(plot_config, show_plot=True):
    """Plot current density of the 2D simulation
    """
    tframe = plot_config["tframe"]
    bg = plot_config['bg']
    bg_str = str(int(bg * 10)).zfill(2)
    pic_run = "2D-Lx150-bg" + str(bg) + "-150ppc-16KNL"
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

    fdir = '../img/cori_3d/absJ_2d/bg' + bg_str + '/'
    mkdir_p(fdir)

    fig = plt.figure(figsize=[3.25, 1.5])
    rect = [0.16, 0.28, 0.72, 0.65]
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
    ax.set_xlabel(r'$x/d_i$', fontsize=10)
    ax.set_ylabel(r'$z/d_i$', fontsize=10)
    ax.tick_params(labelsize=8)
    text1 = r'$|\boldsymbol{J}|/J_0$'
    ax.text(0.02, 0.85, text1, color='w', fontsize=10,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)

    rect_cbar = np.copy(rect)
    rect_cbar[0] += rect[2] + 0.02
    rect_cbar[2] = 0.02
    cbar_ax = fig.add_axes(rect_cbar)
    cbar_ax.tick_params(axis='y', which='major', direction='in')
    cbar = fig.colorbar(p1, cax=cbar_ax, extend='max')
    cbar.ax.tick_params(labelsize=8)
    cbar.set_ticks((np.linspace(0, 0.4, 5)))
    fname = fdir + 'absJ_' + str(tframe) + ".jpg"
    fig.savefig(fname, dpi=400)
    fname = fdir + 'absJ_' + str(tframe) + ".pdf"
    fig.savefig(fname, dpi=400)
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
    bg = plot_config['bg']
    bg_str = str(int(bg * 10)).zfill(2)
    pic_run = "3D-Lx150-bg" + str(bg) + "-150ppc-2048KNL"
    root_dir = "/net/scratch3/xiaocanli/reconnection/Cori_runs/"
    pic_run_dir = root_dir + pic_run + "/"
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
    nmin = 0.5
    if bg_str == '02':
        nmax = 2.2
    if bg_str == '10':
        nmax = 1.6

    p1 = ax.imshow(nrho_xz, extent=[xmin, xmax, zmin, zmax],
                   vmin=nmin, vmax=nmax,
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

    fdir = '../img/cori_3d/rho_xz_3d/' + pic_run + '/'
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


def turbulent_field(plot_config):
    """calculate the turbulent component of a field
    """
    bg = plot_config["bg"]
    bg_str = str(int(bg * 10)).zfill(2)
    tframe = plot_config["tframe"]
    pic_run = "3D-Lx150-bg" + str(bg) + "-150ppc-2048KNL"
    root_dir = "/net/scratch3/xiaocanli/reconnection/Cori_runs/"
    pic_run_dir = root_dir + pic_run + "/"
    tframe = plot_config["tframe"]
    species = plot_config["species"]
    tframe = plot_config["tframe"]
    var = plot_config["var"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    nx, ny, nz = pic_info.nx, pic_info.ny, pic_info.nz
    nxr2, nyr2, nzr2 = nx // 2, ny // 2, nz // 2
    smime = math.sqrt(pic_info.mime)
    dx_de = pic_info.dx_di * smime
    dy_de = pic_info.dy_di * smime
    dz_de = pic_info.dz_di * smime
    dvol = dx_de * dy_de * dz_de * 8

    tindex = pic_info.particle_interval * tframe
    field_dir = pic_run_dir + "data-smooth/"
    if 'nv' in var:
        fname = field_dir + "n" + species + "_" + str(tindex) + ".gda"
        rho = np.fromfile(fname, dtype=np.float32)
        fname = field_dir + "v" + species + var[-1] + "_" + str(tindex) + ".gda"
        fdata = np.fromfile(fname, dtype=np.float32) * rho
        field_name = "n" + species + "_v" + species + var[-1]
    else:
        fname = field_dir + var + "_" + str(tindex) + ".gda"
        fdata = np.fromfile(fname, dtype=np.float32)
        field_name = var
    fdata = fdata.reshape([nzr2, nyr2, nxr2])
    fdata_xz = np.mean(fdata, axis=1)
    if 'nv' in var:
        rho = rho.reshape([nzr2, nyr2, nxr2])
        rho_xz = np.mean(rho, axis=1)
    for iy in range(nyr2):
        fdata[:, iy, :] -= fdata_xz
        if 'nv' in var:
            rho[:, iy, :] -= rho_xz
    fname = field_dir + 'd' + field_name + "_" + str(tindex) + '.gda'
    fdata.tofile(fname)

    if 'nv' in var:
        bulk_ene = 0.5 * np.sum(fdata_xz**2/rho_xz) * dvol * nyr2
        turb_ene = 0.0
        for iy in range(nyr2):
            turb_ene += 0.5 * np.sum(fdata[:, iy, :]**2/rho_xz)
        turb_ene *= dvol
    else:
        bulk_ene = 0.5 * np.sum(fdata_xz**2) * dvol * nyr2
        turb_ene = 0.5 * np.sum(fdata**2) * dvol

    fene = np.asarray([bulk_ene, turb_ene])
    fdir = '../data/cori_3d/turb_field/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + field_name + "_" + str(tframe) + ".dat"
    fene.tofile(fname)


def plot_turbulent_energy(plot_config):
    """calculate the turbulent field energy
    """
    bg = plot_config["bg"]
    bg_str = str(int(bg * 10)).zfill(2)
    tframe = plot_config["tframe"]
    pic_run = "3D-Lx150-bg" + str(bg) + "-150ppc-2048KNL"
    root_dir = "/net/scratch3/xiaocanli/reconnection/Cori_runs/"
    pic_run_dir = root_dir + pic_run + "/"
    tstart = plot_config["tstart"]
    tend = plot_config["tend"]
    species = plot_config["species"]
    tframe = plot_config["tframe"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    vol = pic_info.lx_di * pic_info.ly_di * pic_info.lz_di * (pic_info.mime)**1.5
    ene_norm = 0.5 * pic_info.b0**2 * vol
    fdir = '../data/cori_3d/turb_field/' + pic_run + '/'
    nframes = tend - tstart + 1
    twci = np.linspace(tstart, tend, nframes) * pic_info.dt_fields
    fene = np.zeros([9, nframes, 2])
    field_vars = ['bx', 'by', 'bz',
                  'ne_vex', 'ne_vey', 'ne_vez',
                  'ni_vix', 'ni_viy', 'ni_viz']
    for tframe in range(tstart, tend + 1):
        for ivar, var in enumerate(field_vars):
            fname = fdir + var + "_" + str(tframe) + ".dat"
            fdata = np.fromfile(fname)
            if 'ni' in var:
                fdata *= pic_info.mime
            fene[ivar, tframe - tstart, :] = fdata
    fene = fene.T / ene_norm

    COLORS = palettable.tableau.Tableau_10.mpl_colors
    fig = plt.figure(figsize=[7, 5])
    rect = [0.12, 0.12, 0.83, 0.8]
    ax = fig.add_axes(rect)
    plt.plot(fene[:, :, 1].T / ene_norm)
    linestyles = ['-', '--', ':']
    for ivar in range(3):
        for i in range(3):
            ax.plot(twci, fene[1, :, 3*ivar + i], linewidth=2,
                    linestyle=linestyles[ivar], color=COLORS[i])
    ax.set_xlim([twci.min(), twci.max()])
    ax.set_xlabel(r'$t\Omega_{ci}$', fontsize=16)
    ax.set_ylabel(r'$R$', fontsize=16)
    ax.tick_params(bottom=True, top=False, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top=True)
    ax.tick_params(axis='x', which='major', direction='in', top=True)
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.tick_params(labelsize=12)
    plt.show()


def resave_exponentiation(plot_config):
    """resave the exponentiation factor
    """
    bg = plot_config["bg"]
    bg_str = str(int(bg * 10)).zfill(2)
    tframe = plot_config["tframe"]
    pic_run = "3D-Lx150-bg" + str(bg) + "-150ppc-2048KNL"
    root_dir = "/net/scratch3/xiaocanli/reconnection/Cori_runs/"
    pic_run_dir = root_dir + pic_run + "/"
    tframe = plot_config["tframe"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    nxr, nyr, nzr = pic_info.nx//2, pic_info.ny//2, pic_info.nz//2

    fname = pic_run_dir + 'exponentiation/ftle_' + str(tframe).zfill(4) + '.csv'
    fdata = np.genfromtxt(fname, skip_header=1, delimiter=',')
    fdata1 = fdata[fdata[:, -1].argsort()]
    fdata2 = fdata1[fdata1[:, 3].argsort(kind='stable')]
    fdata3 = fdata2.reshape([nxr, nzr, -1])
    fname = pic_run_dir + 'exponentiation/ftle_' + str(tframe).zfill(4) + '.dat'
    fdata3.tofile(fname)


def plot_exponentiation(plot_config):
    """plot the exponentiation factor
    """
    bg = plot_config["bg"]
    bg_str = str(int(bg * 10)).zfill(2)
    tframe = plot_config["tframe"]
    pic_run = "3D-Lx150-bg" + str(bg) + "-150ppc-2048KNL"
    root_dir = "/net/scratch3/xiaocanli/reconnection/Cori_runs/"
    pic_run_dir = root_dir + pic_run + "/"
    tframe = plot_config["tframe"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    nxr, nyr, nzr = pic_info.nx//2, pic_info.ny//2, pic_info.nz//2
    xmin, xmax = 0, pic_info.lx_di
    ymin, ymax = -pic_info.ly_di * 0.5, pic_info.ly_di * 0.5
    zmin, zmax = -pic_info.lz_di * 0.5, pic_info.lz_di * 0.5
    x_di = pic_info.x_di
    y_di = pic_info.y_di
    z_di = pic_info.z_di
    nx, = x_di.shape
    ny, = y_di.shape
    xmesh, ymesh = np.meshgrid(x_di, y_di)

    cmap = plt.cm.viridis

    tframe = 8
    fname = pic_run_dir + 'exponentiation/ftle_' + str(tframe).zfill(4) + '.dat'
    fdata = np.fromfile(fname)
    fdata = fdata.reshape([nxr, nzr, -1])
    fig = plt.figure(figsize=[3.5, 2.5])
    rect0 = [0.15, 0.58, 0.75, 0.38]
    hgap, vgap = 0.02, 0.05
    rect = np.copy(rect0)
    ax = fig.add_axes(rect)
    p1 = ax.imshow(fdata[:, :, 1].T,
                   extent=[xmin, xmax, ymin, ymax],
                   vmin=0, vmax=6,
                   cmap=cmap, aspect='auto',
                   origin='lower', interpolation='bicubic')
    ax.set_ylim([-20, 20])
    ax.tick_params(bottom=True, top=False, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top=True)
    ax.tick_params(axis='x', which='major', direction='in', top=True)
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.tick_params(axis='x', labelbottom=False)
    ax.set_ylabel(r'$y/d_i$', fontsize=10)
    ax.tick_params(labelsize=8)
    twci = math.ceil((tframe * pic_info.dt_fields) / 0.1) * 0.1
    text1 = r'$t\Omega_{ci}=' + ("{%0.0f}" % twci) + '$'
    ax.text(0.98, 0.1, text1, color='w', fontsize=10,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='right', verticalalignment='center',
            transform=ax.transAxes)
    fdir = pic_run_dir + 'reconnection_layer/'
    cs1_surface1 = np.zeros((ny, nx))
    cs2_surface1 = np.zeros((ny, nx))
    fname = fdir + 'rec_layer_' + str(tframe) + '.h5'
    fh = h5py.File(fname, 'r')
    grp = fh['rec_layer']
    grp['Top'].read_direct(cs1_surface1)
    grp['Bottom'].read_direct(cs2_surface1)
    fh.close()
    ax.plot(xmesh[0, :], cs1_surface1[0, :], linewidth=0.5, color='w')
    ax.plot(xmesh[0, :], cs2_surface1[0, :], linewidth=0.5, color='w')

    tframe = 20
    fname = pic_run_dir + 'exponentiation/ftle_' + str(tframe).zfill(4) + '.dat'
    fdata = np.fromfile(fname)
    fdata = fdata.reshape([nxr, nzr, -1])
    rect[1] -= rect[3] + vgap
    ax1 = fig.add_axes(rect)
    p1 = ax1.imshow(fdata[:, :, 1].T,
                   extent=[xmin, xmax, ymin, ymax],
                   vmin=0, vmax=6,
                   cmap=cmap, aspect='auto',
                   origin='lower', interpolation='bicubic')
    ax1.set_ylim([-20, 20])
    ax1.tick_params(bottom=True, top=False, left=True, right=True)
    ax1.tick_params(axis='x', which='minor', direction='in', top=True)
    ax1.tick_params(axis='x', which='major', direction='in', top=True)
    ax1.tick_params(axis='y', which='minor', direction='in')
    ax1.tick_params(axis='y', which='major', direction='in')
    ax1.set_xlabel(r'$x/d_i$', fontsize=10)
    ax1.set_ylabel(r'$y/d_i$', fontsize=10)
    ax1.tick_params(labelsize=8)
    twci = math.ceil((tframe * pic_info.dt_fields) / 0.1) * 0.1
    text1 = r'$t\Omega_{ci}=' + ("{%0.0f}" % twci) + '$'
    ax1.text(0.98, 0.1, text1, color='w', fontsize=10,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='right', verticalalignment='center',
             transform=ax1.transAxes)
    cs1_surface2 = np.zeros((ny, nx))
    cs2_surface2 = np.zeros((ny, nx))
    fname = fdir + 'rec_layer_' + str(tframe) + '.h5'
    fh = h5py.File(fname, 'r')
    grp = fh['rec_layer']
    grp['Top'].read_direct(cs1_surface2)
    grp['Bottom'].read_direct(cs2_surface2)
    fh.close()
    ax1.plot(xmesh[0, :], cs1_surface2[0, :], linewidth=0.5, color='w')
    ax1.plot(xmesh[0, :], cs2_surface2[0, :], linewidth=0.5, color='w')

    rect_cbar = np.copy(rect)
    rect_cbar[0] += rect[2] + hgap
    rect_cbar[1] += rect[3]*0.5
    rect_cbar[2] = 0.02
    rect_cbar[3] = rect[3] + vgap
    cbar_ax = fig.add_axes(rect_cbar)
    cbar = fig.colorbar(p1, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=8)
    cbar_ax.set_title(r'$\sigma$', fontsize=10)
    cbar_ax.tick_params(axis='y', which='major', direction='in')

    fdir = '../img/cori_3d/'
    mkdir_p(fdir)
    fname = fdir + 'ftle_bg' + bg_str + '.pdf'
    fig.savefig(fname, dpi=400)
    plt.show()


def plot_exponentiation_pub(plot_config):
    """plot the exponentiation factor for publication
    """
    bg = plot_config["bg"]
    bg_str = str(int(bg * 10)).zfill(2)
    tframe = plot_config["tframe"]
    pic_run = "3D-Lx150-bg" + str(bg) + "-150ppc-2048KNL"
    root_dir = "/net/scratch3/xiaocanli/reconnection/Cori_runs/"
    pic_run_dir = root_dir + pic_run + "/"
    tframe = plot_config["tframe"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    nxr, nyr, nzr = pic_info.nx//2, pic_info.ny//2, pic_info.nz//2
    xmin, xmax = 0, pic_info.lx_di
    ymin, ymax = -pic_info.ly_di * 0.5, pic_info.ly_di * 0.5
    zmin, zmax = -pic_info.lz_di * 0.5, pic_info.lz_di * 0.5
    x_di = pic_info.x_di
    y_di = pic_info.y_di
    z_di = pic_info.z_di
    nx, = x_di.shape
    ny, = y_di.shape
    xmesh, ymesh = np.meshgrid(x_di, y_di)

    # cmap = plt.cm.Reds
    # print(np.asarray(cmap(192))*256)

    cmap = plt.cm.gist_heat_r

    fig = plt.figure(figsize=[3.5, 1.2])
    rect0 = [0.12, 0.28, 0.79, 0.58]
    hgap, vgap = 0.02, 0.05
    rect = np.copy(rect0)
    ax1 = fig.add_axes(rect)

    tframe = 15
    fname = pic_run_dir + 'exponentiation/ftle_' + str(tframe).zfill(4) + '.dat'
    fdata = np.fromfile(fname)
    fdata = fdata.reshape([nxr, nzr, -1])
    p1 = ax1.imshow(fdata[:, :, 1].T,
                   extent=[xmin, xmax, ymin, ymax],
                   vmin=0, vmax=6,
                   cmap=cmap, aspect='auto',
                   origin='lower', interpolation='none')
    ax1.set_ylim([-20, 20])
    ax1.tick_params(bottom=True, top=False, left=True, right=True)
    ax1.tick_params(axis='x', which='minor', direction='in', top=True)
    ax1.tick_params(axis='x', which='major', direction='in', top=True)
    ax1.tick_params(axis='y', which='minor', direction='in')
    ax1.tick_params(axis='y', which='major', direction='in')
    # ax1.set_xlabel(r'$x/d_i$', fontsize=8)
    # ax1.set_ylabel(r'$z/d_i$', fontsize=8, labelpad=-2)
    ax1.tick_params(labelsize=6)
    twci = math.ceil((tframe * pic_info.dt_fields) / 0.1) * 0.1
    text1 = r'$t\Omega_{ci}=' + ("{%0.0f}" % twci) + '$'
    # ax1.text(0.98, 0.1, text1, color='w', fontsize=10,
    #          bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
    #          horizontalalignment='right', verticalalignment='center',
    #          transform=ax1.transAxes)

    ax1.plot([59, 61], [-0.4, -0.4], color='w',
             linestyle='-', linewidth=0.25)

    rect_cbar = np.copy(rect)
    rect_cbar[0] += rect[2] + hgap
    rect_cbar[2] = 0.02
    rect_cbar[3] = rect[3]
    cbar_ax = fig.add_axes(rect_cbar)
    cbar = fig.colorbar(p1, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=6)
    # cbar_ax.set_title(r'$\sigma$', fontsize=8)
    cbar_ax.tick_params(axis='y', which='major', direction='in')

    fdir = '../img/cori_3d/'
    mkdir_p(fdir)
    fname = fdir + 'ftle_bg' + bg_str + '_' + str(tframe) + '.pdf'
    fig.savefig(fname, dpi=800)
    plt.show()


def ene_dist_vkappa(plot_config, show_plot=True):
    """
    Plot the spatial distribution of high-energy particle and vdot_kappa
    """
    tframe = plot_config["tframe"]
    bg = plot_config["bg"]
    bg_str = str(int(bg * 10)).zfill(2)
    pic_run = "3D-Lx150-bg" + str(bg) + "-150ppc-2048KNL"
    if bg_str == '02':
        pic_run += "-tracking"
    root_dir = "/net/scratch3/xiaocanli/reconnection/Cori_runs/"
    pic_run_dir = root_dir + pic_run + '/'
    species = plot_config['species']
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    smime = math.sqrt(pic_info.mime)
    lx_de = pic_info.lx_di * smime
    ly_de = pic_info.ly_di * smime
    lz_de = pic_info.lz_di * smime
    x_di = pic_info.x_di
    y_di = pic_info.y_di
    z_di = pic_info.z_di
    xr2_di = x_di[1::2]
    yr2_di = y_di[1::2]
    zr2_di = z_di[1::2]

    pic_run = "3D-Lx150-bg" + str(bg) + "-150ppc-2048KNL"
    pic_run_dir = root_dir + pic_run + '/'
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
    # kene /= eth
    xmin, xmax = 0, pic_info.lx_di
    zmin, zmax = -pic_info.lz_di * 0.5, pic_info.lz_di * 0.5
    nx, ny, nz = pic_info.nx, pic_info.ny, pic_info.nz
    nxr2, nyr2, nzr2 = nx // 2, ny // 2, nz // 2
    nxr4, nyr4, nzr4 = nx // 4, ny // 4, nz // 4
    smime = math.sqrt(pic_info.mime)
    fname = pic_run_dir + "data-smooth/vexb_kappa_" + str(tindex) + ".gda"
    vdot_kappa = np.fromfile(fname, dtype=np.float32)
    vdot_kappa = vdot_kappa.reshape((nzr2, nyr2, nxr2))

    nrhos = []
    nbands = 7
    for iband in range(nbands):
        fname = (pic_run_dir + "data-smooth2/n" + species + "_" +
                 str(iband) + "_" + str(tindex) + ".gda")
        nrho = np.fromfile(fname, dtype=np.float32)
        nrho = nrho.reshape((nzr4, nyr4, nxr4)) * stride_particle_dump
        nrhos.append(nrho)

    yslice = 440
    print("y-position of slice: %f" % (yslice*pic_info.dy_di*2 - 0.5*pic_info.ly_di))

    nmin, nmax = 1E-4, 1E-2
    knorm = 100 if bg_str == '02' else 400

    fig = plt.figure(figsize=[7, 2.2])
    rect0 = [0.07, 0.6, 0.42, 0.35]
    hgap, vgap = 0.01, 0.07

    # 2D spatial distribution of high-energy particles
    pic_run = "2D-Lx150-bg" + str(bg) + "-150ppc-16KNL"
    pic_run_dir = root_dir + pic_run + '/'
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    nx, nz = pic_info.nx, pic_info.nz
    iband = 4
    tindex = tframe * pic_info.fields_interval
    fname = (pic_run_dir + "data-smooth2/n" + species + "_" +
             str(iband) + "_" + str(tindex) + ".gda")
    nhigh = np.fromfile(fname, dtype=np.float32)
    nhigh = nhigh.reshape((nz, nx))
    rect = np.copy(rect0)
    ax1 = fig.add_axes(rect)
    p1 = ax1.imshow(nhigh + 1E-10,
                    extent=[xmin, xmax, zmin, zmax],
                    norm = LogNorm(vmin=nmin, vmax=nmax),
                    cmap=plt.cm.plasma, aspect='auto',
                    origin='lower', interpolation='bicubic')
    ax1.tick_params(bottom=True, top=True, left=True, right=True)
    ax1.tick_params(axis='x', which='minor', direction='in')
    ax1.tick_params(axis='x', which='major', direction='in')
    ax1.tick_params(axis='y', which='minor', direction='in')
    ax1.tick_params(axis='y', which='major', direction='in')
    ax1.set_ylabel('$z/d_i$', fontsize=10, labelpad=0)
    ax1.tick_params(labelsize=8)
    ax1.tick_params(axis='x', labelbottom=False)
    ax1.set_ylim([-20, 20])
    label1 = (r'2D: $n(' + str(2**(iband-1)*10) + r'\varepsilon_\text{th} < ' +
              r'\varepsilon < ' + str(2**iband*10) + r'\varepsilon_\text{th})$')
    ax1.text(0.97, 0.8, label1, color='w', fontsize=8,
             bbox=dict(facecolor='none', alpha=0.75,
                       edgecolor='none', boxstyle="round,pad=0.1"),
             horizontalalignment='right',
             verticalalignment='center',
             transform=ax1.transAxes)

    # 2D spatial distribution of vdot_kappa
    kwargs = {"current_time": tframe,
              "xl": 0, "xr": pic_info.lx_di,
              "zb": -pic_info.lz_di, "zt": pic_info.lz_di}
    fname = pic_run_dir + "data/vexb_kappa.gda"
    x, z, vexb_kappa = read_2d_fields(pic_info, fname, **kwargs)
    rect[1] -= vgap + rect[3]
    ax2 = fig.add_axes(rect)
    vexb_kappa = vexb_kappa * 100
    vmin, vmax = -1.0, 1.0
    p2 = ax2.imshow(vexb_kappa, extent=[xmin, xmax, zmin, zmax],
                    vmin=vmin, vmax=vmax,
                    cmap=plt.cm.seismic, aspect='auto',
                    origin='lower', interpolation='bicubic')
    ax2.tick_params(bottom=True, top=True, left=True, right=True)
    ax2.tick_params(axis='x', which='minor', direction='in')
    ax2.tick_params(axis='x', which='major', direction='in')
    ax2.tick_params(axis='y', which='minor', direction='in')
    ax2.tick_params(axis='y', which='major', direction='in')
    ax2.set_xlabel('$x/d_i$', fontsize=10)
    ax2.set_ylabel('$z/d_i$', fontsize=10, labelpad=0)
    ax2.tick_params(labelsize=8)
    ax2.set_ylim([-20, 20])
    label2 = r'2D: $' + str(knorm) + r'\boldsymbol{v}_{\boldsymbol{E}}\cdot\boldsymbol{\kappa}$'
    ax2.text(0.97, 0.8, label2, color='k', fontsize=8,
             bbox=dict(facecolor='none', alpha=0.5,
                       edgecolor='none', boxstyle="round,pad=0.2"),
             horizontalalignment='right',
             verticalalignment='center',
             transform=ax2.transAxes)

    # 3D spatial distribution of high-energy particles
    rect = np.copy(rect0)
    rect[0] += hgap + rect[2]
    ax3 = fig.add_axes(rect)
    iband = 4
    label3 = (r'3D: $n(' + str(2**(iband-1)*10) + r'\varepsilon_\text{th} < ' +
              r'\varepsilon < ' + str(2**iband*10) + r'\varepsilon_\text{th})$')
    p3 = ax3.imshow(nrhos[iband][:, yslice//2, :] + 1E-10,
                    extent=[xmin, xmax, zmin, zmax],
                    norm = LogNorm(vmin=nmin, vmax=nmax),
                    cmap=plt.cm.plasma, aspect='auto',
                    origin='lower', interpolation='bicubic')
    ax3.tick_params(bottom=True, top=True, left=True, right=True)
    ax3.tick_params(axis='x', which='minor', direction='in')
    ax3.tick_params(axis='x', which='major', direction='in')
    ax3.tick_params(axis='y', which='minor', direction='in')
    ax3.tick_params(axis='y', which='major', direction='in')
    ax3.tick_params(labelsize=8)
    ax3.tick_params(axis='x', labelbottom=False)
    ax3.tick_params(axis='y', labelleft=False)
    ax3.set_ylim([-20, 20])
    ax3.text(0.97, 0.8, label3, color='w', fontsize=8,
             bbox=dict(facecolor='none', alpha=0.75,
                       edgecolor='none', boxstyle="round,pad=0.1"),
             horizontalalignment='right',
             verticalalignment='center',
             transform=ax3.transAxes)
    rect_cbar = np.copy(rect)
    rect_cbar[0] += rect[2] + 0.01
    rect_cbar[2] = 0.01
    cbar_ax = fig.add_axes(rect_cbar)
    cbar = fig.colorbar(p3, cax=cbar_ax, extend='both')
    cbar_ax.tick_params(bottom=False, top=False, left=False, right=True)
    cbar_ax.tick_params(axis='y', which='major', direction='out')
    cbar_ax.tick_params(axis='y', which='minor', direction='in', right=False)
    cbar.ax.tick_params(labelsize=8)

    # 3D spatial distribution of vdot_kappa
    rect[1] -= vgap + rect[3]
    ax4 = fig.add_axes(rect)
    vmin, vmax = -1.0, 1.0
    fdata = vdot_kappa[:, yslice, :]*knorm
    ng = 3
    kernel = np.ones((ng,ng)) / float(ng*ng)
    fdata = signal.convolve2d(fdata, kernel, mode='same')
    p4 = ax4.imshow(fdata, extent=[xmin, xmax, zmin, zmax],
                    vmin=vmin, vmax=vmax,
                    cmap=plt.cm.seismic, aspect='auto',
                    origin='lower', interpolation='bicubic')
    cs = ax4.contour(xr2_di, zr2_di, np.abs(fdata), colors='k',
                     linewidths=0.25, levels=[0.1])
    ax4.tick_params(bottom=True, top=True, left=True, right=True)
    ax4.tick_params(axis='x', which='minor', direction='in')
    ax4.tick_params(axis='x', which='major', direction='in')
    ax4.tick_params(axis='y', which='minor', direction='in')
    ax4.tick_params(axis='y', which='major', direction='in')
    ax4.set_xlabel('$x/d_i$', fontsize=10)
    ax4.tick_params(axis='y', labelleft=False)
    ax4.tick_params(labelsize=8)
    ax4.set_ylim([-20, 20])
    label1 = r'3D: $' + str(knorm) + r'\boldsymbol{v}_{\boldsymbol{E}}\cdot\boldsymbol{\kappa}$'
    ax4.text(0.97, 0.8, label1, color='k', fontsize=8,
             bbox=dict(facecolor='none', alpha=0.5,
                       edgecolor='none', boxstyle="round,pad=0.2"),
             horizontalalignment='right',
             verticalalignment='center',
             transform=ax4.transAxes)
    rect_cbar = np.copy(rect)
    rect_cbar[0] += rect[2] + 0.01
    rect_cbar[2] = 0.01
    cbar_ax = fig.add_axes(rect_cbar)
    cbar = fig.colorbar(p4, cax=cbar_ax, extend='both')
    cbar_ax.tick_params(axis='y', which='major', direction='out')
    cbar_ax.tick_params(axis='y', which='minor', direction='in')
    cbar.ax.tick_params(labelsize=8)

    fdir = '../img/cori_3d/nhigh_vkappa/' + pic_run + '/tframe_' + str(tframe) + '/'
    mkdir_p(fdir)
    fname = fdir + 'nhigh_vkappa_' + species + '_yslice_' + str(yslice) + ".pdf"
    fig.savefig(fname, dpi=400)

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_profile_2d(plot_config, show_plot=True):
    """Plot profiles of the 2D fields
    """
    tframe = plot_config["tframe"]
    bg = plot_config["bg"]
    bg_str = str(int(bg * 10)).zfill(2)
    pic_run = "2D-Lx150-bg" + str(bg) + "-150ppc-16KNL-long"
    root_dir = "/net/scratch3/xiaocanli/reconnection/Cori_runs/"
    pic_run_dir = root_dir + pic_run + "/"
    species = plot_config["species"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    vthe = pic_info.vthe
    vthi = pic_info.vthi
    nbins = 1000
    ndata = nbins + 3  # including magnetic field
    tindex = pic_info.particle_interval * tframe
    gama = 1.0 / math.sqrt(1.0 - 2*vthe**2)
    eth_e = gama - 1.0
    gama = 1.0 / math.sqrt(1.0 - 2*vthi**2)
    eth_i = (gama - 1.0) * pic_info.mime
    nx, nz = pic_info.nx, pic_info.nz
    ntf = pic_info.ntf
    xmin, xmax = 0, pic_info.lx_di
    ymin, ymax = -pic_info.ly_di * 0.5, pic_info.ly_di * 0.5
    zmin, zmax = -pic_info.lz_di * 0.5, pic_info.lz_di * 0.5
    b0 = pic_info.b0
    va = pic_info.dtwce * math.sqrt(1.0 / pic_info.mime) / pic_info.dtwpe
    enorm = b0 * va

    kwargs = {"current_time": tframe,
              "xl": 0, "xr": pic_info.lx_di,
              "zb": -pic_info.lz_di*0.5, "zt": pic_info.lz_di*0.5}
    fname = pic_run_dir + "data/Ay.gda"
    x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
    fdata = {}

    fname = pic_run_dir + "data/jx.gda"
    x, z, fdata["jx"] = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/jy.gda"
    x, z, fdata["jy"] = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/jz.gda"
    x, z, fdata["jz"] = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/absJ.gda"
    x, z, fdata["absj"] = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/bx.gda"
    x, z, fdata["Bx"] = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/by.gda"
    x, z, fdata["By"] = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/bz.gda"
    x, z, fdata["Bz"] = read_2d_fields(pic_info, fname, **kwargs)
    fdata["absB"] = np.sqrt(fdata["Bx"]**2 + fdata["By"]**2 + fdata["Bz"]**2)
    ib2 = 1.0 / fdata["absB"]**2
    fname = pic_run_dir + "data/ex.gda"
    x, z, fdata["Ex"] = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/ey.gda"
    x, z, fdata["Ey"] = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/ez.gda"
    x, z, fdata["Ez"] = read_2d_fields(pic_info, fname, **kwargs)
    fdata["Ex"] = fdata["Ex"] / enorm
    fdata["Ey"] = fdata["Ey"] / enorm
    fdata["Ez"] = fdata["Ez"] / enorm

    # ion
    fname = pic_run_dir + "data/vix.gda"
    x, z, fdata["vix"] = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/viy.gda"
    x, z, fdata["viy"] = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/viz.gda"
    x, z, fdata["viz"] = read_2d_fields(pic_info, fname, **kwargs)
    fdata["vix"] = fdata["vix"] / va
    fdata["viy"] = fdata["viy"] / va
    fdata["viz"] = fdata["viz"] / va
    fname = pic_run_dir + "data/ni.gda"
    x, z, fdata["ni"] = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/pi-xx.gda"
    x, z, pxx = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/pi-yy.gda"
    x, z, pyy = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/pi-zz.gda"
    x, z, pzz = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/pi-xy.gda"
    x, z, pxy = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/pi-xz.gda"
    x, z, pxz = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/pi-yz.gda"
    x, z, pyz = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/pi-yx.gda"
    x, z, pyx = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/pi-zx.gda"
    x, z, pzx = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/pi-zy.gda"
    x, z, pzy = read_2d_fields(pic_info, fname, **kwargs)
    pscalar_i = (pxx + pyy + pzz) / 3.0
    ppara_i = (pxx * fdata["Bx"]**2 + pyy * fdata["By"]**2 +
               pzz * fdata["Bz"]**2 + (pxy + pyx) * fdata["Bx"] * fdata["By"] +
               (pxz + pzx) * fdata["Bx"] * fdata["Bz"] +
               (pyz + pzy) * fdata["By"] * fdata["Bz"])
    ppara_i *= ib2
    pperp_i = (pscalar_i * 3 - ppara_i) * 0.5
    fdata["Tpara_i"] = ppara_i / fdata["ni"] / eth_i
    fdata["Tperp_i"] = pperp_i / fdata["ni"] / eth_i
    fdata["anisotropy_i"] = ppara_i / pperp_i
    fdata["Ti"] = pscalar_i / fdata["ni"] / eth_i

    # electron
    fname = pic_run_dir + "data/vex.gda"
    x, z, fdata["vex"] = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/vey.gda"
    x, z, fdata["vey"] = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/vez.gda"
    x, z, fdata["vez"] = read_2d_fields(pic_info, fname, **kwargs)
    fdata["vex"] = fdata["vex"] / va
    fdata["vey"] = fdata["vey"] / va
    fdata["vez"] = fdata["vez"] / va
    fname = pic_run_dir + "data/ne.gda"
    x, z, fdata["ne"] = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/pe-xx.gda"
    x, z, pxx = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/pe-yy.gda"
    x, z, pyy = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/pe-zz.gda"
    x, z, pzz = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/pe-xy.gda"
    x, z, pxy = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/pe-xz.gda"
    x, z, pxz = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/pe-yz.gda"
    x, z, pyz = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/pe-yx.gda"
    x, z, pyx = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/pe-zx.gda"
    x, z, pzx = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/pe-zy.gda"
    x, z, pzy = read_2d_fields(pic_info, fname, **kwargs)
    pscalar_e = (pxx + pyy + pzz) / 3.0
    ppara_e = (pxx * fdata["Bx"]**2 + pyy * fdata["By"]**2 +
               pzz * fdata["Bz"]**2 + (pxy + pyx) * fdata["Bx"] * fdata["By"] +
               (pxz + pzx) * fdata["Bx"] * fdata["Bz"] +
               (pyz + pzy) * fdata["By"] * fdata["Bz"])
    ppara_e *= ib2
    pperp_e = (pscalar_e * 3 - ppara_e) * 0.5
    fdata["Tpara_e"] = ppara_e / fdata["ne"] / eth_e
    fdata["Tperp_e"] = pperp_e / fdata["ne"] / eth_e
    fdata["anisotropy_e"] = ppara_e / pperp_e
    fdata["Te"] = pscalar_e / fdata["ne"] / eth_e

    nvar = len(fdata)
    var_config = {"jx": {"dmin_max": [-0.4, 0.4],
                         "cmap": plt.cm.coolwarm,
                         "base": 0.0,
                         "label": ""},
                  "jy": {"dmin_max": [-0.4, 0.4],
                         "cmap": plt.cm.coolwarm,
                         "base": 0.0,
                         "label": r"$j_x, j_y, j_z$"},
                  "jz": {"dmin_max": [-0.4, 0.4],
                         "cmap": plt.cm.coolwarm,
                         "base": 0.0,
                         "label": ""},
                  "absj": {"dmin_max": [0.0, 0.4],
                           "cmap": plt.cm.viridis,
                           "base": 0.0,
                           "label": r"$|\boldsymbol{J}|/J_0$"},
                  "Bx": {"dmin_max": [-1.0, 1.0],
                         "cmap": plt.cm.coolwarm,
                         "base": 0.0,
                         "label": ""},
                  "By": {"dmin_max": [-1.0, 1.0],
                         "cmap": plt.cm.coolwarm,
                         "base": 0.0,
                         "label": r"$B_x, B_y, B_z$"},
                  "Bz": {"dmin_max": [-1.0, 1.0],
                         "cmap": plt.cm.coolwarm,
                         "base": 0.0,
                         "label": ""},
                  "absB": {"dmin_max": [0.0, 2.0],
                           "cmap": plt.cm.magma,
                           "base": 1.0,
                           "label": r"$|\boldsymbol{B}|$"},
                  "Ex": {"dmin_max": [-0.5, 0.5],
                         "cmap": plt.cm.coolwarm,
                         "base": 0.0,
                         "label": ""},
                  "Ey": {"dmin_max": [-0.5, 0.5],
                         "cmap": plt.cm.coolwarm,
                         "base": 0.0,
                         "label": r"$E_x, E_y, E_z$"},
                  "Ez": {"dmin_max": [-0.5, 0.5],
                         "cmap": plt.cm.coolwarm,
                         "base": 0.0,
                         "label": ""},
                  "vix": {"dmin_max": [-0.5, 0.5],
                          "cmap": plt.cm.coolwarm,
                          "base": 0.0,
                          "label": ""},
                  "viy": {"dmin_max": [-0.5, 0.5],
                          "cmap": plt.cm.coolwarm,
                          "base": 0.0,
                          "label": r"$V_{ix}, V_{iy}, V_{iz}/V_A$"},
                  "viz": {"dmin_max": [-0.5, 0.5],
                          "cmap": plt.cm.coolwarm,
                          "base": 0.0,
                          "label": ""},
                  "ni": {"dmin_max": [0, 3.0],
                         "cmap": plt.cm.inferno,
                         "base": 1.0,
                         "label": r"$n_i$"},
                  "Tpara_i": {"dmin_max": [0, 20.0],
                              "cmap": plt.cm.plasma,
                              "base": 1.0,
                              "label": r"$T_{i\parallel}$"},
                  "Tperp_i": {"dmin_max": [0, 20.0],
                              "cmap": plt.cm.plasma,
                              "base": 1.0,
                              "label": r"$T_{i\perp}$"},
                  "anisotropy_i": {"dmin_max": [0, 2.0],
                                   "cmap": plt.cm.seismic,
                                   "base": 1.0,
                                   "label": r"$T_{i\parallel}/T_{i\perp}$"},
                  "Ti": {"dmin_max": [0, 20.0],
                         "cmap": plt.cm.plasma,
                         "base": 1.0,
                         "label": r"$T_i$"},
                  "vex": {"dmin_max": [-0.5, 0.5],
                          "cmap": plt.cm.coolwarm,
                          "base": 0.0,
                          "label": ""},
                  "vey": {"dmin_max": [-0.5, 0.5],
                          "cmap": plt.cm.coolwarm,
                          "base": 0.0,
                          "label": r"$V_{ex}, V_{ey}, V_{ez}/V_A$"},
                  "vez": {"dmin_max": [-0.5, 0.5],
                          "cmap": plt.cm.coolwarm,
                          "base": 0.0,
                          "label": ""},
                  "ne": {"dmin_max": [0, 3.0],
                         "cmap": plt.cm.inferno,
                         "base": 1.0,
                         "label": r"$n_e$"},
                  "Tpara_e": {"dmin_max": [0, 20.0],
                              "cmap": plt.cm.plasma,
                              "base": 1.0,
                              "label": r"$T_{e\parallel}$"},
                  "Tperp_e": {"dmin_max": [0, 20.0],
                              "cmap": plt.cm.plasma,
                              "base": 1.0,
                              "label": r"$T_{e\perp}$"},
                  "anisotropy_e": {"dmin_max": [0, 2.0],
                                   "cmap": plt.cm.seismic,
                                   "base": 1.0,
                                   "label": r"$T_{e\parallel}/T_{e\perp}$"},
                  "Te": {"dmin_max": [0, 20.0],
                         "cmap": plt.cm.plasma,
                         "base": 1.0,
                         "label": r"$T_e$"}
                }
    fig1 = plt.figure(figsize=[16, 12])
    rect0 = [0.05, 0.9, 0.25, 0.07]
    rect = np.copy(rect0)
    hgap, vgap = 0.07, 0.015
    ilabel = 0
    nrows = 11
    xposes = np.linspace(5, xmax-5, 15)
    for ivar, var in enumerate(fdata):
        if var in ["jx", "vix", "vex"]:
            rect = np.copy(rect0)
            rect[0] += (hgap + rect0[2]) * ((ivar + 1) // 7)
        ax = fig1.add_axes(rect)
        dmin_max = var_config[var]["dmin_max"]
        cmap = var_config[var]["cmap"]
        p1 = ax.imshow(fdata[var],
                       extent=[xmin, xmax, zmin, zmax],
                       vmin=dmin_max[0], vmax=dmin_max[1],
                       cmap=cmap, aspect='auto',
                       origin='lower', interpolation='bicubic')
        ax.contour(x, z, Ay, colors='k', linewidths=0.5)
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')
        ax.tick_params(labelsize=12)
        twci = math.ceil((tframe * pic_info.dt_fields) / 0.1) * 0.1
        ax.set_ylim([-20, 20])
        for xpos in xposes:
            ax.plot([xpos, xpos], [-20, 20], color='w', linewidth=0.5,
                    linestyle='--')
        if ivar in [nrows-1, nrows+7, nrows+15]:
            ax.set_xlabel(r'$x/d_i$', fontsize=16)
        else:
            ax.tick_params(axis='x', labelbottom=False)
        if ivar < nrows:
            ax.set_ylabel(r'$z/d_i$', fontsize=16)
        else:
            ax.tick_params(axis='y', labelleft=False)
        label = var_config[var]["label"]
        if label:
            rect_cbar = np.copy(rect)
            rect_cbar[0] += rect[2] + 0.005
            rect_cbar[2] = 0.005
            cbar_ax = fig1.add_axes(rect_cbar)
            cbar = fig1.colorbar(p1, cax=cbar_ax)
            cbar.ax.tick_params(labelsize=12)
            cbar_ax.set_ylabel(label, fontsize=16)
        if var == 'vix':
            text1 = r'$t\Omega_{ci}=' + ("{%0.0f}" % twci) + '$'
            ax.set_title(text1, fontsize=16)
        rect[1] -= vgap + rect[3]

    fdir = '../img/cori_3d/profile_2d/' + pic_run + '/tframe' + str(tframe) + '/'
    mkdir_p(fdir)
    fname = fdir + "profile_2d.jpg"
    fig1.savefig(fname)
    if show_plot:
        plt.show()
    else:
        plt.close()

    # cuts
    ng = 5
    kernel = np.ones(ng) / float(ng)
    xp = (xposes - xmin) / pic_info.dx_di
    ixs1 = np.floor(xp)
    ixs2 = np.ceil(xp)
    for ix1, ix2, xdi in zip(ixs1, ixs2, xposes):
        fig2 = plt.figure(figsize=[16, 12])
        for ivar, var in enumerate(fdata):
            if var in ["jx", "vix", "vex"]:
                rect = np.copy(rect0)
                rect[0] += (hgap + rect0[2]) * ((ivar + 1) // 7)
            ax = fig2.add_axes(rect)
            ax.tick_params(bottom=True, top=True, left=True, right=True)
            ax.tick_params(axis='x', which='minor', direction='in')
            ax.tick_params(axis='x', which='major', direction='in')
            ax.tick_params(axis='y', which='minor', direction='in')
            ax.tick_params(axis='y', which='major', direction='in')
            ax.tick_params(labelsize=12)
            twci = math.ceil((tframe * pic_info.dt_fields) / 0.1) * 0.1
            ax.set_xlim([zmin, zmax])
            cut = (fdata[var][:, int(ix1)] + fdata[var][:, int(ix2)]) * 0.5
            cut_new = np.convolve(cut, kernel, 'same')
            ax.plot(z, cut, color=COLORS[0], linewidth=1)
            ax.plot(z, cut_new, color=COLORS[1], linewidth=2)
            base = var_config[var]["base"]
            ax.plot([zmin, zmax], [base, base], color='k',
                    linewidth=0.5, linestyle='--')
            if ivar in [nrows-1, nrows+7, nrows+15]:
                ax.set_xlabel(r'$z/d_i$', fontsize=16)
            else:
                ax.tick_params(axis='x', labelbottom=False)
            label = var_config[var]["label"]
            if label:
                ax.set_ylabel(label, fontsize=16)
            if var == 'vix':
                text1 = r'$t\Omega_{ci}=' + ("{%0.0f}" % twci) + '$'
                ax.set_title(text1, fontsize=16)
            rect[1] -= vgap + rect[3]
        fname = fdir + "cut_x" + str(int(xdi)) + ".pdf"
        fig2.savefig(fname)
        plt.close()

    if show_plot:
        plt.show()
    else:
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
    parser.add_argument('--bg', action="store", default='0.2', type=float,
                        help='Guide field strength')
    parser.add_argument('--var', action="store", default='bx',
                        help='variable name of a field')
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
    parser.add_argument('--rho_profile', action="store_true", default=False,
                        help="whether to plot densities profile")
    parser.add_argument('--absb_profile', action="store_true", default=False,
                        help="whether to plot magnetic field magnitude")
    parser.add_argument('--calc_absj_dist', action="store_true", default=False,
                        help="whether to calculate current density distribution")
    parser.add_argument('--plot_absj_dist', action="store_true", default=False,
                        help="whether to plot current density distribution")
    parser.add_argument('--calc_abse_dist', action="store_true", default=False,
                        help="whether to calculate electric field distribution")
    parser.add_argument('--plot_abse_dist', action="store_true", default=False,
                        help="whether to plot electric field distribution")
    parser.add_argument('--reconnection_layer', action="store_true", default=False,
                        help="whether to get reconnection layer boundary")
    parser.add_argument('--plot_reconnection_layer', action="store_true", default=False,
                        help="whether to plot reconnection layer boundary")
    parser.add_argument('--reconnection_layer_2d', action="store_true", default=False,
                        help="whether to get reconnection layer boundary")
    parser.add_argument('--convert_layer_vtk', action="store_true", default=False,
                        help="whether to convert reconnection layer data to vtk")
    parser.add_argument('--magnetic_flux', action="store_true", default=False,
                        help="whether to magnetic flux in the inflow region")
    parser.add_argument('--plot_rrate', action="store_true", default=False,
                        help="whether to plot magnetic reconnection rate")
    parser.add_argument('--plot_rrate_2d', action="store_true", default=False,
                        help="whether to plot magnetic reconnection rate for 2D")
    parser.add_argument('--turb_field', action="store_true", default=False,
                        help="whether to calculate turbulent field")
    parser.add_argument('--plot_turb_ene', action="store_true", default=False,
                        help="whether to plot turbulent field energy")
    parser.add_argument('--resave_exponentiation', action="store_true", default=False,
                        help="whether to resave the exponentiation factor")
    parser.add_argument('--plot_exponentiation', action="store_true", default=False,
                        help="whether to plot the exponentiation factor")
    parser.add_argument('--plot_exponentiation_pub', action="store_true", default=False,
                        help="whether to plot the exponentiation factor for publication")
    parser.add_argument('--ene_dist_vkappa', action="store_true", default=False,
                        help="whether to plot spatial distribution of high-energy particle " +
                        "and vdot_kappa")
    parser.add_argument('--profile_2d', action="store_true", default=False,
                        help="whether to plot fields profile in the 2D simulation")
    return parser.parse_args()


def analysis_single_frames(plot_config, args):
    """Analysis for multiple time frames
    """
    tframe = args.tframe
    if args.jslice:
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
    elif args.rho_profile:
        rho_profile(plot_config)
    elif args.absb_profile:
        absb_profile(plot_config)
    elif args.calc_absj_dist:
        calc_absj_dist(plot_config)
    elif args.plot_absj_dist:
        plot_absj_dist(plot_config)
    elif args.calc_abse_dist:
        calc_abse_dist(plot_config)
    elif args.plot_abse_dist:
        plot_abse_dist(plot_config)
    elif args.reconnection_layer:
        reconnection_layer(plot_config)
    elif args.plot_reconnection_layer:
        plot_reconnection_layer(plot_config)
    elif args.reconnection_layer_2d:
        reconnection_layer_2d(plot_config)
    elif args.convert_layer_vtk:
        convert_layer_vtk(plot_config)
    elif args.magnetic_flux:
        calc_magnetic_flux(plot_config)
    elif args.plot_rrate:
        plot_reconnection_rate(plot_config)
    elif args.plot_rrate_2d:
        plot_reconnection_rate_2d(plot_config)
    elif args.turb_field:
        turbulent_field(plot_config)
    elif args.plot_turb_ene:
        plot_turbulent_energy(plot_config)
    elif args.resave_exponentiation:
        resave_exponentiation(plot_config)
    elif args.plot_exponentiation:
        plot_exponentiation(plot_config)
    elif args.plot_exponentiation_pub:
        plot_exponentiation_pub(plot_config)
    elif args.ene_dist_vkappa:
        ene_dist_vkappa(plot_config)
    elif args.profile_2d:
        plot_profile_2d(plot_config)


def process_input(plot_config, args, tframe):
    """process one time frame"""
    plot_config["tframe"] = tframe
    if args.absj_2d:
        plot_absj_2d(plot_config, show_plot=False)
    elif args.calc_absj_dist:
        calc_absj_dist(plot_config)
    elif args.calc_abse_dist:
        calc_abse_dist(plot_config)
    elif args.jslice:
        plot_jslice(plot_config)
    elif args.reconnection_layer:
        reconnection_layer(plot_config, show_plot=False)
    elif args.magnetic_flux:
        calc_magnetic_flux(plot_config)
    elif args.turb_field:
        turbulent_field(plot_config)
    elif args.convert_layer_vtk:
        convert_layer_vtk(plot_config)
    elif args.profile_2d:
        plot_profile_2d(plot_config, show_plot=False)


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
            elif args.rho_profile:
                rho_profile(plot_config, show_plot=False)
            elif args.absb_profile:
                absb_profile(plot_config, show_plot=False)
            elif args.jslice:
                plot_jslice(plot_config)
            elif args.reconnection_layer_2d:
                reconnection_layer_2d(plot_config, show_plot=False)
            elif args.plot_reconnection_layer:
                plot_reconnection_layer(plot_config, show_plot=False)
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
    plot_config["var"] = args.var
    if args.multi_frames:
        analysis_multi_frames(plot_config, args)
    else:
        analysis_single_frames(plot_config, args)


if __name__ == "__main__":
    main()
