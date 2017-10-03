"""
Analysis procedures for 2D contour plots.
"""
import collections
import math
import multiprocessing
import os
import os.path
import struct

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from matplotlib import rc
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
from scipy.fftpack import fft2, fftshift, ifft2
from scipy.ndimage.filters import generic_filter as gf

import color_maps as cm
import colormap.colormaps as cmaps
import pic_information
from energy_conversion import read_data_from_json
from runs_name_path import ApJ_long_paper_runs
from shell_functions import mkdir_p

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

font = {
    'family': 'serif',
    #'color'  : 'darkred',
    'color': 'black',
    'weight': 'normal',
    'size': 24,
}
mpl.rcParams['contour.negative_linestyle'] = 'solid'


def read_2d_fields(pic_info, fname, current_time, xl, xr, zb, zt):
    """Read 2D fields data from file.
    
    Args:
        pic_info: namedtuple for the PIC simulation information.
        fname: the filename.
        current_time: current time frame.
        xl, xr: left and right x position in di (ion skin length).
        zb, zt: top and bottom z position in di.
    """
    print 'Reading data from ', fname
    print 'The spatial range (di): ', \
            'x_left = ', xl, 'x_right = ', xr, \
            'z_bottom = ', zb, 'z_top = ', zt
    nx = pic_info.nx
    nz = pic_info.nz
    x_di = pic_info.x_di
    z_di = pic_info.z_di
    dx_di = pic_info.dx_di
    dz_di = pic_info.dz_di
    xmin = np.min(x_di)
    xmax = np.max(x_di)
    zmin = np.min(z_di)
    zmax = np.max(z_di)
    if (xl <= xmin):
        xl_index = 0
    else:
        xl_index = int(math.floor((xl - xmin) / dx_di))
    if (xr >= xmax):
        xr_index = nx - 1
    else:
        xr_index = int(math.ceil((xr - xmin) / dx_di))
    if (zb <= zmin):
        zb_index = 0
    else:
        zb_index = int(math.floor((zb - zmin) / dz_di))
    if (zt >= zmax):
        zt_index = nz - 1
    else:
        zt_index = int(math.ceil((zt - zmin) / dz_di))
    nx1 = xr_index - xl_index + 1
    nz1 = zt_index - zb_index + 1
    fp = np.zeros((nz1, nx1), dtype=np.float32)
    offset = nx * nz * current_time * 4 + zb_index * nx * 4 + xl_index * 4
    for k in range(nz1):
        fp[k, :] = np.memmap(
            fname,
            dtype='float32',
            mode='r',
            offset=offset,
            shape=(nx1),
            order='F')
        offset += nx * 4
    return (x_di[xl_index:xr_index + 1], z_di[zb_index:zt_index + 1], fp)


def plot_2d_contour(x, z, field_data, ax, fig, is_cbar=1, **kwargs):
    """Plot contour of 2D fields.

    Args:
        x, z: the x and z coordinates for the field data.
        field_data: the 2D field data set.
        ax: axes object.
        fig: figure object.
        is_cbar: whether to plot colorbar. Default is yes.
    Returns:
        p1: plot object.
        cbar: color bar object.
    """
    xmin = np.min(x)
    xmax = np.max(x)
    zmin = np.min(z)
    zmax = np.max(z)
    nx, = x.shape
    nz, = z.shape
    if (kwargs and "xstep" in kwargs and "zstep" in kwargs):
        data = field_data[0:nz:kwargs["zstep"], 0:nx:kwargs["xstep"]]
    else:
        data = field_data
    print "Maximum and minimum of the data: ", np.max(data), np.min(data)
    if (kwargs and "vmin" in kwargs and "vmax" in kwargs):
        p1 = ax.imshow(
            data,
            cmap=plt.cm.jet,
            extent=[xmin, xmax, zmin, zmax],
            aspect='auto',
            origin='lower',
            vmin=kwargs["vmin"],
            vmax=kwargs["vmax"],
            interpolation='bicubic')
    else:
        p1 = ax.imshow(
            data,
            cmap=plt.cm.jet,
            extent=[xmin, xmax, zmin, zmax],
            aspect='auto',
            origin='lower',
            interpolation='spline16')
    # Log scale plot
    if (kwargs and "is_log" in kwargs and kwargs["is_log"] == True):
        p1.norm = LogNorm(vmin=kwargs["vmin"], vmax=kwargs["vmax"])
    ax.tick_params(labelsize=16)
    ax.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    #ax.set_title(r'$\beta_e$', fontsize=24)

    if is_cbar == 1:
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.05)
        cbar = fig.colorbar(p1, cax=cax)
        cbar.ax.tick_params(labelsize=16)
        return (p1, cbar)
    else:
        return p1


def plot_jy(pic_info, species, current_time):
    """Plot out-of-plane current density.

    Args:
        pic_info: namedtuple for the PIC simulation information.
        species: 'e' for electrons, 'i' for ions.
        current_time: current time frame.
    """
    print(current_time)
    kwargs = {
        "current_time": current_time,
        "xl": 0,
        "xr": 200,
        "zb": -20,
        "zt": 20
    }
    x, z, jy = read_2d_fields(pic_info, "../../data/jy.gda", **kwargs)
    x, z, Ay = read_2d_fields(pic_info, "../../data/Ay.gda", **kwargs)
    nx, = x.shape
    nz, = z.shape
    width = 0.75
    height = 0.7
    xs = 0.12
    ys = 0.9 - height
    fig = plt.figure(figsize=[10, 4])
    ax1 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep": 1, "zstep": 1, "vmin": -0.5, "vmax": 0.5}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, jy, ax1, fig, **kwargs_plot)
    p1.set_cmap(plt.cm.get_cmap('seismic'))
    # p1.set_cmap(cmaps.inferno)
    ax1.contour(
        x[0:nx:xstep],
        z[0:nz:zstep],
        Ay[0:nz:zstep, 0:nx:xstep],
        colors='black',
        linewidths=0.5)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax1.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
    ax1.tick_params(labelsize=24)
    cbar1.ax.set_ylabel(r'$j_y$', fontdict=font, fontsize=24)
    cbar1.set_ticks(np.arange(-0.4, 0.5, 0.2))
    cbar1.ax.tick_params(labelsize=24)

    t_wci = current_time * pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(t_wci) + '/\Omega_{ci}$'
    ax1.set_title(title, fontdict=font, fontsize=24)

    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    if not os.path.isdir('../img/img_jy2/'):
        os.makedirs('../img/img_jy2/')
    fname = '../img/img_jy2/jy_' + species + '_' + \
            str(current_time).zfill(3) + '.jpg'
    fig.savefig(fname, dpi=200)

    plt.show()
    # plt.close()


def plot_absB_jy(pic_info, species, current_time):
    """Plot magnetic field strength and out-of-plane current density.

    Args:
        pic_info: namedtuple for the PIC simulation information.
        species: 'e' for electrons, 'i' for ions.
        current_time: current time frame.
    """
    print(current_time)
    kwargs = {
        "current_time": current_time,
        "xl": 0,
        "xr": 200,
        "zb": -30,
        "zt": 30
    }
    x, z, jy = read_2d_fields(pic_info, "../../data/jy.gda", **kwargs)
    x, z, absB = read_2d_fields(pic_info, "../../data/absB.gda", **kwargs)
    x, z, Ay = read_2d_fields(pic_info, "../../data/Ay.gda", **kwargs)
    nx, = x.shape
    nz, = z.shape
    width = 0.8
    height = 0.37
    xs = 0.12
    ys = 0.92 - height
    gap = 0.05
    fig = plt.figure(figsize=[7, 5])
    ax1 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep": 1, "zstep": 1, "vmin": 0, "vmax": 2.0}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, absB, ax1, fig, **kwargs_plot)
    # p1.set_cmap(plt.cm.get_cmap('seismic'))
    p1.set_cmap(cmaps.plasma)
    ax1.contour(
        x[0:nx:xstep],
        z[0:nz:zstep],
        Ay[0:nz:zstep, 0:nx:xstep],
        colors='black',
        linewidths=0.5)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax1.tick_params(labelsize=16)
    ax1.tick_params(axis='x', labelbottom='off')
    # cbar1.set_ticks(np.arange(0, 1.0, 0.2))
    cbar1.ax.tick_params(labelsize=16)
    ax1.text(
        0.02,
        0.8,
        r'$|\mathbf{B}|$',
        color='w',
        fontsize=20,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='left',
        verticalalignment='center',
        transform=ax1.transAxes)

    ys -= height + gap
    ax2 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep": 1, "zstep": 1, "vmin": -0.5, "vmax": 0.5}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p2, cbar2 = plot_2d_contour(x, z, jy, ax2, fig, **kwargs_plot)
    p2.set_cmap(plt.cm.get_cmap('seismic'))
    ax2.contour(
        x[0:nx:xstep],
        z[0:nz:zstep],
        Ay[0:nz:zstep, 0:nx:xstep],
        colors='black',
        linewidths=0.5)
    ax2.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    ax2.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax2.tick_params(labelsize=16)
    cbar2.set_ticks(np.arange(-0.4, 0.5, 0.2))
    cbar2.ax.tick_params(labelsize=16)
    ax2.text(
        0.02,
        0.8,
        r'$j_y$',
        color='k',
        fontsize=20,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='left',
        verticalalignment='center',
        transform=ax2.transAxes)

    t_wci = current_time * pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(t_wci) + '/\Omega_{ci}$'
    ax1.set_title(title, fontdict=font, fontsize=20)

    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    dir = '../img/img_absB_jy/'
    if not os.path.isdir(dir):
        os.makedirs(dir)
    fname = dir + 'absB_jy_' + str(current_time).zfill(3) + '.jpg'
    fig.savefig(fname, dpi=200)

    # plt.show()
    plt.close()


def plot_by_multi():
    """Plot out-of-plane magnetic field for multiple runs.
    """
    base_dirs, run_names = ApJ_long_paper_runs()
    print run_names
    ct1, ct2, ct3 = 20, 21, 11
    kwargs = {"current_time": ct1, "xl": 0, "xr": 200, "zb": -10, "zt": 10}
    base_dir = base_dirs[2]
    run_name = run_names[2]
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    fname1 = base_dir + 'data/jy.gda'
    fname2 = base_dir + 'data/Ay.gda'
    x, z, by1 = read_2d_fields(pic_info, fname1, **kwargs)
    x, z, Ay1 = read_2d_fields(pic_info, fname2, **kwargs)
    kwargs["current_time"] = ct2
    base_dir = base_dirs[5]
    run_name = run_names[5]
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    fname1 = base_dir + 'data/by.gda'
    fname2 = base_dir + 'data/Ay.gda'
    x, z, by2 = read_2d_fields(pic_info, fname1, **kwargs)
    x, z, Ay2 = read_2d_fields(pic_info, fname2, **kwargs)
    kwargs["current_time"] = ct3
    base_dir = base_dirs[6]
    run_name = run_names[6]
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    fname1 = base_dir + 'data/by.gda'
    fname2 = base_dir + 'data/Ay.gda'
    x, z, by3 = read_2d_fields(pic_info, fname1, **kwargs)
    x, z, Ay3 = read_2d_fields(pic_info, fname2, **kwargs)
    nx, = x.shape
    nz, = z.shape
    width = 0.75
    height = 0.24
    gap = 0.05
    xs = 0.12
    ys = 0.96 - height
    w1, h1 = 7, 5
    fig = plt.figure(figsize=[w1, h1])

    ax1 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep": 2, "zstep": 2, "vmin": -1.0, "vmax": 1.0}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1 = plot_2d_contour(x, z, by1, ax1, fig, is_cbar=0, **kwargs_plot)
    xs1 = xs + width * 1.02
    ys1 = ys - 2 * (height + gap)
    width1 = width * 0.04
    height1 = 3 * height + 2 * gap
    cax = fig.add_axes([xs1, ys1, width1, height1])
    cbar1 = fig.colorbar(p1, cax=cax)
    cbar1.ax.tick_params(labelsize=16)
    p1.set_cmap(plt.cm.get_cmap('bwr'))
    levels = np.linspace(np.max(Ay1), np.min(Ay1), 10)
    ax1.contour(
        x[0:nx:xstep],
        z[0:nz:zstep],
        Ay1[0:nz:zstep, 0:nx:xstep],
        colors='black',
        linewidths=0.5,
        levels=levels)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax1.tick_params(labelsize=16)
    ax1.tick_params(axis='x', labelbottom='off')
    cbar1.set_ticks(np.arange(-0.8, 0.9, 0.4))
    cbar1.ax.tick_params(labelsize=16)
    t_wci = ct1 * pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(t_wci) + '/\Omega_{ci}$'
    ax1.text(
        0.02,
        0.8,
        title,
        color='k',
        fontsize=20,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='left',
        verticalalignment='center',
        transform=ax1.transAxes)

    ys -= height + gap
    ax2 = fig.add_axes([xs, ys, width, height])
    p2 = plot_2d_contour(x, z, by2, ax2, fig, is_cbar=0, **kwargs_plot)
    p2.set_cmap(plt.cm.get_cmap('bwr'))
    levels = np.linspace(np.max(Ay2), np.min(Ay2), 10)
    ax2.contour(
        x[0:nx:xstep],
        z[0:nz:zstep],
        Ay2[0:nz:zstep, 0:nx:xstep],
        colors='black',
        linewidths=0.5,
        levels=levels)
    ax2.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax2.tick_params(labelsize=16)
    ax2.tick_params(axis='x', labelbottom='off')
    t_wci = ct2 * pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(t_wci) + '/\Omega_{ci}$'
    ax2.text(
        0.02,
        0.8,
        title,
        color='k',
        fontsize=20,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='left',
        verticalalignment='center',
        transform=ax2.transAxes)

    ys -= height + gap
    ax3 = fig.add_axes([xs, ys, width, height])
    p3 = plot_2d_contour(x, z, by3, ax3, fig, is_cbar=0, **kwargs_plot)
    p3.set_cmap(plt.cm.get_cmap('bwr'))
    levels = np.linspace(np.max(Ay3), np.min(Ay3), 10)
    ax3.contour(
        x[0:nx:xstep],
        z[0:nz:zstep],
        Ay3[0:nz:zstep, 0:nx:xstep],
        colors='black',
        linewidths=0.5,
        levels=levels)
    ax3.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax3.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    ax3.tick_params(labelsize=16)
    t_wci = ct3 * pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(t_wci) + '/\Omega_{ci}$'
    ax3.text(
        0.02,
        0.8,
        title,
        color='k',
        fontsize=20,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='left',
        verticalalignment='center',
        transform=ax3.transAxes)

    # minor_ticks = np.arange(0, 200, 5)
    # ax3.set_xticks(minor_ticks, minor=True)
    # ax3.grid(which='both')

    # if not os.path.isdir('../img/'):
    #     os.makedirs('../img/')
    # fname = '../img/by_time.eps'
    # fig.savefig(fname)

    plt.show()
    # plt.close()


def plot_by(pic_info):
    """Plot out-of-plane magnetic field.
    """
    ct1, ct2, ct3 = 10, 20, 35
    kwargs = {"current_time": ct1, "xl": 0, "xr": 200, "zb": -10, "zt": 10}
    x, z, by1 = read_2d_fields(pic_info, "../../data/by.gda", **kwargs)
    x, z, Ay1 = read_2d_fields(pic_info, "../../data/Ay.gda", **kwargs)
    kwargs["current_time"] = ct2
    x, z, by2 = read_2d_fields(pic_info, "../../data/by.gda", **kwargs)
    x, z, Ay2 = read_2d_fields(pic_info, "../../data/Ay.gda", **kwargs)
    kwargs["current_time"] = ct3
    x, z, by3 = read_2d_fields(pic_info, "../../data/by.gda", **kwargs)
    x, z, Ay3 = read_2d_fields(pic_info, "../../data/Ay.gda", **kwargs)
    nx, = x.shape
    nz, = z.shape
    width = 0.75
    height = 0.24
    gap = 0.05
    xs = 0.12
    ys = 0.96 - height
    w1, h1 = 7, 5
    fig = plt.figure(figsize=[w1, h1])

    ax1 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep": 2, "zstep": 2, "vmin": -1.0, "vmax": 1.0}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1 = plot_2d_contour(x, z, by1, ax1, fig, is_cbar=0, **kwargs_plot)
    xs1 = xs + width * 1.02
    ys1 = ys - 2 * (height + gap)
    width1 = width * 0.04
    height1 = 3 * height + 2 * gap
    cax = fig.add_axes([xs1, ys1, width1, height1])
    cbar1 = fig.colorbar(p1, cax=cax)
    cbar1.ax.tick_params(labelsize=16)
    p1.set_cmap(plt.cm.get_cmap('bwr'))
    ax1.contour(
        x[0:nx:xstep],
        z[0:nz:zstep],
        Ay1[0:nz:zstep, 0:nx:xstep],
        colors='black',
        linewidths=0.5)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax1.tick_params(labelsize=16)
    ax1.tick_params(axis='x', labelbottom='off')
    cbar1.set_ticks(np.arange(-0.8, 0.9, 0.4))
    cbar1.ax.tick_params(labelsize=16)
    t_wci = ct1 * pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(t_wci) + '/\Omega_{ci}$'
    ax1.text(
        0.02,
        0.8,
        title,
        color='k',
        fontsize=20,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='left',
        verticalalignment='center',
        transform=ax1.transAxes)

    ys -= height + gap
    ax2 = fig.add_axes([xs, ys, width, height])
    p2 = plot_2d_contour(x, z, by2, ax2, fig, is_cbar=0, **kwargs_plot)
    p2.set_cmap(plt.cm.get_cmap('bwr'))
    ax2.contour(
        x[0:nx:xstep],
        z[0:nz:zstep],
        Ay2[0:nz:zstep, 0:nx:xstep],
        colors='black',
        linewidths=0.5)
    ax2.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax2.tick_params(labelsize=16)
    ax2.tick_params(axis='x', labelbottom='off')
    t_wci = ct2 * pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(t_wci) + '/\Omega_{ci}$'
    ax2.text(
        0.02,
        0.8,
        title,
        color='k',
        fontsize=20,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='left',
        verticalalignment='center',
        transform=ax2.transAxes)

    ys -= height + gap
    ax3 = fig.add_axes([xs, ys, width, height])
    p3 = plot_2d_contour(x, z, by3, ax3, fig, is_cbar=0, **kwargs_plot)
    p3.set_cmap(plt.cm.get_cmap('bwr'))
    ax3.contour(
        x[0:nx:xstep],
        z[0:nz:zstep],
        Ay3[0:nz:zstep, 0:nx:xstep],
        colors='black',
        linewidths=0.5)
    ax3.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax3.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    ax3.tick_params(labelsize=16)
    t_wci = ct3 * pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(t_wci) + '/\Omega_{ci}$'
    ax3.text(
        0.02,
        0.8,
        title,
        color='k',
        fontsize=20,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='left',
        verticalalignment='center',
        transform=ax3.transAxes)

    # minor_ticks = np.arange(0, 200, 5)
    # ax3.set_xticks(minor_ticks, minor=True)
    # ax3.grid(which='both')

    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    fname = '../img/by_time.eps'
    fig.savefig(fname)

    plt.show()
    # plt.close()


def plot_number_density(pic_info,
                        species,
                        ct,
                        run_name,
                        shock_pos,
                        base_dir='../../',
                        single_file=True):
    """Plot plasma beta and number density.

    Args:
        pic_info: namedtuple for the PIC simulation information.
        species: 'e' for electrons, 'i' for ions.
        ct current time frame.
    """
    xmin, xmax = 0, pic_info.lx_di
    xmax = 105
    zmin, zmax = -0.5 * pic_info.lz_di, 0.5 * pic_info.lz_di
    if single_file:
        kwargs = {
            "current_time": ct,
            "xl": xmin,
            "xr": xmax,
            "zb": zmin,
            "zt": zmax
        }
        fname = base_dir + 'data1/n' + species + '.gda'
        x, z, num_rho = read_2d_fields(pic_info, fname, **kwargs)
        fname_Ay = base_dir + 'data1/Ay.gda'
        if os.path.isfile(fname_Ay):
            x, z, Ay = read_2d_fields(pic_info, fname_Ay, **kwargs)
    else:
        kwargs = {
            "current_time": 0,
            "xl": xmin,
            "xr": xmax,
            "zb": zmin,
            "zt": zmax
        }
        fields_interval = pic_info.fields_interval
        tframe = str(fields_interval * ct)
        fname = base_dir + 'data/n' + species + '_' + tframe + '.gda'
        x, z, num_rho = read_2d_fields(pic_info, fname, **kwargs)
        fname_Ay = base_dir + 'data/Ay_' + tframe + '.gda'
        if os.path.isfile(fname_Ay):
            x, z, Ay = read_2d_fields(pic_info, fname_Ay, **kwargs)
    nx, = x.shape
    nz, = z.shape
    nrho_cum = np.sum(num_rho, axis=0) / nz
    xm = x[shock_pos]

    w1, h1 = 0.7, 0.52
    xs, ys = 0.15, 0.94 - h1
    gap = 0.05

    width, height = 10, 12
    fig = plt.figure(figsize=[10, 12])
    ax1 = fig.add_axes([xs, ys, w1, h1])
    # kwargs_plot = {"xstep":2, "zstep":2, "is_log":True, "vmin":0.1, "vmax":10}
    kwargs_plot = {"xstep": 2, "zstep": 2, "vmin": 0.5, "vmax": 5}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, num_rho, ax1, fig, **kwargs_plot)
    p1.set_cmap(plt.cm.jet)
    nlevels = 15
    if os.path.isfile(fname_Ay):
        levels = np.linspace(np.min(Ay), np.max(Ay), nlevels)
        ax1.contour(
            x[0:nx:xstep],
            z[0:nz:zstep],
            Ay[0:nz:zstep, 0:nx:xstep],
            colors='black',
            linewidths=0.5,
            levels=levels)
    ax1.set_xlim([xmin, xmax])
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax1.tick_params(labelsize=24)
    lname = r'$n_' + species + '$'
    cbar1.ax.set_ylabel(lname, fontdict=font, fontsize=24)
    cbar1.ax.tick_params(labelsize=24)

    ax1.plot([xm, xm], [zmin, zmax], color='white', linestyle='--')

    t_wci = ct * pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(t_wci) + '/\Omega_{ci}$'
    ax1.set_title(title, fontdict=font, fontsize=24)

    h2 = 0.3
    ys -= gap + h2
    w2 = w1 * 0.98 - 0.05 / width
    ax2 = fig.add_axes([xs, ys, w2, h2])
    ax2.plot(x, nrho_cum, linewidth=2, color='k')
    ax2.set_xlim([xmin, xmax])
    ax2.set_ylim([0.5, 4.5])
    ax2.plot([xm, xm], ax2.get_ylim(), color='k', linestyle='--')
    ax2.tick_params(labelsize=24)
    ax2.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)

    fig_dir = '../img/img_number_densities/' + run_name + '/'
    mkdir_p(fig_dir)
    fname = fig_dir + '/nrho_linear_' + species + '_' + str(ct).zfill(
        3) + '.jpg'
    # fig.savefig(fname)

    plt.show()
    # plt.close()


def plot_vx(pic_info, species, ct, run_name, shock_pos, base_dir='../../'):
    """Plot vx

    Args:
        pic_info: namedtuple for the PIC simulation information.
        species: 'e' for electrons, 'i' for ions.
        ct current time frame.
    """
    xmin, xmax = 0, pic_info.lx_di
    xmax = 105
    zmin, zmax = -0.5 * pic_info.lz_di, 0.5 * pic_info.lz_di
    kwargs = {
        "current_time": ct,
        "xl": xmin,
        "xr": xmax,
        "zb": zmin,
        "zt": zmax
    }
    fname = base_dir + 'data1/v' + species + 'x.gda'
    x, z, vx = read_2d_fields(pic_info, fname, **kwargs)
    fname = base_dir + 'data1/Ay.gda'
    x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
    nx, = x.shape
    nz, = z.shape
    vx_cum = np.sum(vx, axis=0) / nz
    vx_grad = np.abs(np.gradient(vx_cum))
    xm = x[shock_pos]

    w1, h1 = 0.7, 0.52
    xs, ys = 0.15, 0.94 - h1
    gap = 0.05

    width, height = 10, 12
    fig = plt.figure(figsize=[10, 12])
    ax1 = fig.add_axes([xs, ys, w1, h1])
    kwargs_plot = {"xstep": 2, "zstep": 2, "vmin": -0.1, "vmax": 0.1}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, vx, ax1, fig, **kwargs_plot)
    p1.set_cmap(plt.cm.jet)
    nlevels = 20
    levels = np.linspace(np.min(Ay), np.max(Ay), nlevels)
    ax1.contour(
        x[0:nx:xstep],
        z[0:nz:zstep],
        Ay[0:nz:zstep, 0:nx:xstep],
        colors='black',
        linewidths=0.5,
        levels=levels)
    ax1.set_xlim([xmin, xmax])
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax1.tick_params(labelsize=24)
    lname = r'$v_{' + species + 'x}$'
    cbar1.ax.set_ylabel(lname, fontdict=font, fontsize=24)
    cbar1.ax.tick_params(labelsize=24)

    ax1.plot([xm, xm], [zmin, zmax], color='k', linestyle='--')

    t_wci = ct * pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(t_wci) + '/\Omega_{ci}$'
    ax1.set_title(title, fontdict=font, fontsize=24)

    h2 = 0.3
    ys -= gap + h2
    w2 = w1 * 0.98 - 0.05 / width
    ax2 = fig.add_axes([xs, ys, w2, h2])
    ax2.plot(x, vx_cum, linewidth=2, color='k')
    ax2.set_xlim([xmin, xmax])
    ax2.set_ylim([-0.25, 0.10])
    ax2.plot([xm, xm], ax2.get_ylim(), color='k', linestyle='--')
    ax2.tick_params(labelsize=24)
    ax2.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)

    fig_dir = '../img/img_velocity/' + run_name + '/'
    mkdir_p(fig_dir)
    fname = fig_dir + '/v' + species + 'x_' + str(ct).zfill(3) + '.jpg'
    fig.savefig(fname)

    # plt.show()
    plt.close()


def get_anisotropy_data(pic_info, species, ct, rootpath='../../'):
    """
    Args:
        pic_info: namedtuple for the PIC simulation information.
        species: 'e' for electrons, 'i' for ions.
        ct: current time frame.
        rootpath: the root path of a run.
    """
    kwargs = {"current_time": ct, "xl": 0, "xr": 200, "zb": -20, "zt": 20}
    fname = rootpath + 'data/p' + species + '-xx.gda'
    x, z, pxx = read_2d_fields(pic_info, fname, **kwargs)
    fname = rootpath + 'data/p' + species + '-yy.gda'
    x, z, pyy = read_2d_fields(pic_info, fname, **kwargs)
    fname = rootpath + 'data/p' + species + '-zz.gda'
    x, z, pzz = read_2d_fields(pic_info, fname, **kwargs)
    fname = rootpath + 'data/p' + species + '-xy.gda'
    x, z, pxy = read_2d_fields(pic_info, fname, **kwargs)
    fname = rootpath + 'data/p' + species + '-xz.gda'
    x, z, pxz = read_2d_fields(pic_info, fname, **kwargs)
    fname = rootpath + 'data/p' + species + '-yz.gda'
    x, z, pyz = read_2d_fields(pic_info, fname, **kwargs)
    x, z, bx = read_2d_fields(pic_info, rootpath + "data/bx.gda", **kwargs)
    x, z, by = read_2d_fields(pic_info, rootpath + "data/by.gda", **kwargs)
    x, z, bz = read_2d_fields(pic_info, rootpath + "data/bz.gda", **kwargs)
    x, z, absB = read_2d_fields(pic_info, rootpath + "data/absB.gda", **kwargs)
    x, z, Ay = read_2d_fields(pic_info, rootpath + "data/Ay.gda", **kwargs)
    ppara = pxx*bx*bx + pyy*by*by + pzz*bz*bz + \
            pxy*bx*by*2.0 + pxz*bx*bz*2.0 + pyz*by*bz*2.0
    ppara /= absB * absB
    pperp = 0.5 * (pxx + pyy + pzz - ppara)
    return (ppara, pperp, Ay, x, z)


def plot_anisotropy(pic_info, ct):
    """Plot pressure anisotropy.

    Args:
        pic_info: namedtuple for the PIC simulation information.
        ct: current time frame.
    """
    ppara_e, pperp_e, Ay, x, z = get_anisotropy_data(pic_info, 'e', ct)
    ppara_i, pperp_i, Ay, x, z = get_anisotropy_data(pic_info, 'i', ct)
    nx, = x.shape
    nz, = z.shape
    width = 0.8
    height = 0.37
    xs = 0.12
    ys = 0.92 - height
    gap = 0.05

    fig = plt.figure(figsize=[7, 5])
    ax1 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {
        "xstep": 1,
        "zstep": 1,
        "is_log": True,
        "vmin": 0.1,
        "vmax": 10
    }
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, ppara_e / pperp_e, ax1, fig, **
                                kwargs_plot)
    p1.set_cmap(plt.cm.seismic)
    ax1.contour(
        x[0:nx:xstep],
        z[0:nz:zstep],
        Ay[0:nz:zstep, 0:nx:xstep],
        colors='black',
        linewidths=0.5)
    ax1.tick_params(axis='x', labelbottom='off')
    ax1.text(
        0.1,
        0.9,
        r'$p_{e\parallel}/p_{e\perp}$',
        color='k',
        fontsize=20,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='left',
        verticalalignment='center',
        transform=ax1.transAxes)

    ys -= height + gap
    ax2 = fig.add_axes([xs, ys, width, height])
    p2, cbar2 = plot_2d_contour(x, z, ppara_i / pperp_i, ax2, fig, **
                                kwargs_plot)
    p2.set_cmap(plt.cm.seismic)
    ax2.contour(
        x[0:nx:xstep],
        z[0:nz:zstep],
        Ay[0:nz:zstep, 0:nx:xstep],
        colors='black',
        linewidths=0.5)
    ax2.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    ax2.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    ax2.text(
        0.1,
        0.9,
        r'$p_{i\parallel}/p_{i\perp}$',
        color='k',
        fontsize=20,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='left',
        verticalalignment='center',
        transform=ax2.transAxes)

    t_wci = ct * pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(t_wci) + '/\Omega_{ci}$'
    ax1.set_title(title, fontdict=font, fontsize=20)

    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    dir = '../img/anisotropy/'
    if not os.path.isdir(dir):
        os.makedirs(dir)
    fname = dir + 'anisotropy_' + str(ct).zfill(3) + '.jpg'
    fig.savefig(fname, dpi=200)
    plt.close()

    # plt.show()


def plot_anisotropy_multi(species):
    """Plot pressure anisotropy for multiple runs.

    Args:
        species: 'e' for electrons, 'i' for ions.
    """
    width = 0.8
    height = 0.1
    xs = 0.12
    ys = 0.92 - height
    gap = 0.02
    fig = plt.figure(figsize=[7, 12])

    ct = 20
    base_dirs, run_names = ApJ_long_paper_runs()
    for base_dir, run_name in zip(base_dirs, run_names):
        picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        ppara, pperp, Ay, x, z = \
                get_anisotropy_data(pic_info, species, ct, base_dir)
        nx, = x.shape
        nz, = z.shape
        ax1 = fig.add_axes([xs, ys, width, height])
        kwargs_plot = {
            "xstep": 2,
            "zstep": 2,
            "is_log": True,
            "vmin": 0.1,
            "vmax": 10
        }
        xstep = kwargs_plot["xstep"]
        zstep = kwargs_plot["zstep"]
        p1, cbar1 = plot_2d_contour(x, z, ppara / pperp, ax1, fig, **
                                    kwargs_plot)
        p1.set_cmap(plt.cm.seismic)
        ax1.contour(
            x[0:nx:xstep],
            z[0:nz:zstep],
            Ay[0:nz:zstep, 0:nx:xstep],
            colors='black',
            linewidths=0.5)
        ax1.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
        ys -= height + gap

    plt.show()


def plot_beta_rho(pic_info):
    """Plot plasma beta and number density.

    Args:
        pic_info: namedtuple for the PIC simulation information.
    """
    kwargs = {"current_time": 20, "xl": 0, "xr": 200, "zb": -10, "zt": 10}
    x, z, pexx = read_2d_fields(pic_info, "../../data/pe-xx.gda", **kwargs)
    x, z, peyy = read_2d_fields(pic_info, "../../data/pe-yy.gda", **kwargs)
    x, z, pezz = read_2d_fields(pic_info, "../../data/pe-zz.gda", **kwargs)
    x, z, absB = read_2d_fields(pic_info, "../../data/absB.gda", **kwargs)
    x, z, eEB05 = read_2d_fields(pic_info, "../../data/eEB05.gda", **kwargs)
    x, z, Ay = read_2d_fields(pic_info, "../../data/Ay.gda", **kwargs)
    nx, = x.shape
    nz, = z.shape
    beta_e = (pexx + peyy + pezz) * 2 / (3 * absB**2)
    width = 0.8
    height = 0.3
    xs = 0.12
    ys = 0.9 - height
    fig = plt.figure(figsize=[7, 4])
    ax1 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {
        "xstep": 2,
        "zstep": 2,
        "is_log": True,
        "vmin": 0.01,
        "vmax": 10
    }
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, beta_e, ax1, fig, **kwargs_plot)
    p1.set_cmap(cmaps.plasma)
    ax1.contour(
        x[0:nx:xstep],
        z[0:nz:zstep],
        Ay[0:nz:zstep, 0:nx:xstep],
        colors='white',
        linewidths=0.5)
    ax1.tick_params(axis='x', labelbottom='off')
    ax1.set_title(r'$\beta_e$', fontsize=24)

    ys -= height + 0.15
    ax2 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep": 2, "zstep": 2}
    p2, cbar2 = plot_2d_contour(x, z, eEB05, ax2, fig, **kwargs_plot)
    # p2.set_cmap(cmaps.magma)
    # p2.set_cmap(cmaps.inferno)
    p2.set_cmap(cmaps.plasma)
    # p2.set_cmap(cmaps.viridis)
    ax2.contour(
        x[0:nx:xstep],
        z[0:nz:zstep],
        Ay[0:nz:zstep, 0:nx:xstep],
        colors='white',
        linewidths=0.5)
    ax2.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    ax2.set_title(r'$n_{acc}/n_e$', fontsize=24)
    cbar2.set_ticks(np.arange(0.2, 1.0, 0.2))
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    fig.savefig('../img/beta_e_ne_2.eps')
    plt.show()


def plot_jdote_2d(pic_info):
    """Plot jdotE due to drift current.
    Args:
        pic_info: namedtuple for the PIC simulation information.
    """
    kwargs = {"current_time": 40, "xl": 0, "xr": 200, "zb": -10, "zt": 10}
    x, z, jcpara_dote = read_2d_fields(
        pic_info, "../data1/jcpara_dote00_e.gda", **kwargs)
    x, z, jgrad_dote = read_2d_fields(pic_info, "../data1/jgrad_dote00_e.gda",
                                      **kwargs)
    x, z, agyp = read_2d_fields(pic_info, "../data1/agyrotropy00_e.gda",
                                **kwargs)
    x, z, Ay = read_2d_fields(pic_info, "../data/Ay.gda", **kwargs)
    jdote_norm = 1E3
    jcpara_dote *= jdote_norm
    jgrad_dote *= jdote_norm
    nx, = x.shape
    nz, = z.shape
    width = 0.78
    height = 0.25
    xs = 0.14
    xe = 0.94 - xs
    ys = 0.96 - height
    fig = plt.figure(figsize=(7, 5))
    ax1 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep": 2, "zstep": 2, "vmin": -1, "vmax": 1}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, jcpara_dote, ax1, fig, **kwargs_plot)
    p1.set_cmap(cmaps.viridis)
    ax1.contour(
        x[0:nx:xstep],
        z[0:nz:zstep],
        Ay[0:nz:zstep, 0:nx:xstep],
        colors='black',
        linewidths=0.5)
    cbar1.set_ticks(np.arange(-0.8, 1.0, 0.4))
    ax1.tick_params(axis='x', labelbottom='off')
    ax1.text(
        5,
        5.2,
        r'$\mathbf{j}_c\cdot\mathbf{E}$',
        color='blue',
        fontsize=24,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0))

    ys -= height + 0.035
    ax2 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep": 2, "zstep": 2, "vmin": -1, "vmax": 1}
    p2, cbar2 = plot_2d_contour(x, z, jcpara_dote, ax2, fig, **kwargs_plot)
    p2.set_cmap(cmaps.viridis)
    ax2.contour(
        x[0:nx:xstep],
        z[0:nz:zstep],
        Ay[0:nz:zstep, 0:nx:xstep],
        colors='black',
        linewidths=0.5)
    cbar2.set_ticks(np.arange(-0.8, 1.0, 0.4))
    ax2.tick_params(axis='x', labelbottom='off')
    ax2.text(
        5,
        5,
        r'$\mathbf{j}_g\cdot\mathbf{E}$',
        color='green',
        fontsize=24,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0))

    ys -= height + 0.035
    ax3 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep": 2, "zstep": 2, "vmin": 0, "vmax": 1.5}
    p3, cbar3 = plot_2d_contour(x, z, agyp, ax3, fig, **kwargs_plot)
    p3.set_cmap(cmaps.viridis)
    ax3.contour(
        x[0:nx:xstep],
        z[0:nz:zstep],
        Ay[0:nz:zstep, 0:nx:xstep],
        colors='black',
        linewidths=0.5)
    cbar3.set_ticks(np.arange(0, 1.6, 0.4))
    ax3.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    ax3.text(
        5,
        5,
        r'$A_e$',
        color='black',
        fontsize=24,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0))

    plt.show()


def plot_phi_parallel(ct, pic_info):
    """Plot parallel potential.
    Args:
        ct: current time frame.
        pic_info: namedtuple for the PIC simulation information.
    """
    kwargs = {"current_time": ct, "xl": 0, "xr": 200, "zb": -20, "zt": 20}
    x, z, phi_para = read_2d_fields(pic_info, "../../data1/phi_para.gda",
                                    **kwargs)
    x, z, Ay = read_2d_fields(pic_info, "../../data/Ay.gda", **kwargs)

    # phi_para_new = phi_para
    nk = 7
    phi_para_new = signal.medfilt2d(phi_para, kernel_size=(nk, nk))
    # phi_para_new = signal.wiener(phi_para, mysize=5)
    # ng = 9
    # kernel = np.ones((ng,ng)) / float(ng*ng)
    # phi_para_new = signal.convolve2d(phi_para, kernel)

    # fft_transform = np.fft.fft2(phi_para)
    # power_spectrum = np.abs(fft_transform)**2
    # scaled_power_spectrum = np.log10(power_spectrum)
    # scaled_ps0 = scaled_power_spectrum - np.max(scaled_power_spectrum)
    # # print np.min(scaled_ps0), np.max(scaled_ps0)

    # freqx = np.fft.fftfreq(x.size)
    # freqz = np.fft.fftfreq(z.size)
    # xv, zv = np.meshgrid(freqx, freqz)
    # #print np.max(freqx), np.min(freqx)
    # #m = np.ma.masked_where(scaled_ps0 > -8.8, scaled_ps0)
    # #new_fft_spectrum = np.ma.masked_array(fft_transform, m.mask)
    # new_fft_spectrum = fft_transform.copy()
    # #new_fft_spectrum[scaled_ps0 < -1.3] = 0.0
    # print np.max(xv), np.min(xv)
    # new_fft_spectrum[xv > 0.02] = 0.0
    # new_fft_spectrum[xv < -0.02] = 0.0
    # new_fft_spectrum[zv > 0.02] = 0.0
    # new_fft_spectrum[zv < -0.02] = 0.0

    # data = np.fft.ifft2(new_fft_spectrum)

    nx, = x.shape
    nz, = z.shape
    width = 0.85
    height = 0.85
    xs = 0.1
    xe = 0.95 - xs
    ys = 0.96 - height
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep": 1, "zstep": 1, "vmin": -0.05, "vmax": 0.05}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    im1, cbar1 = plot_2d_contour(x, z, phi_para_new, ax1, fig, **kwargs_plot)
    #im1 = plt.imshow(data.real, vmin=-0.1, vmax=0.1)
    im1.set_cmap(plt.cm.seismic)

    ax1.contour(
        x[0:nx:xstep],
        z[0:nz:zstep],
        Ay[0:nz:zstep, 0:nx:xstep],
        colors='black',
        linewidths=0.5,
        levels=np.arange(np.min(Ay), np.max(Ay), 15))
    # cbar1.set_ticks(np.arange(-0.2, 0.2, 0.05))
    #ax1.tick_params(axis='x', labelbottom='off')
    ax1.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    ax1.set_ylabel(r'$y/d_i$', fontdict=font, fontsize=20)

    #plt.close()
    plt.show()


def plot_Ey(pic_info, species, current_time):
    """Plot out-of-plane current density.

    Args:
        pic_info: namedtuple for the PIC simulation information.
        species: 'e' for electrons, 'i' for ions.
        current_time: current time frame.
    """
    print(current_time)
    kwargs = {
        "current_time": current_time,
        "xl": 0,
        "xr": 200,
        "zb": -50,
        "zt": 50
    }
    x, z, ey = read_2d_fields(pic_info, "../../data/ey.gda", **kwargs)
    x, z, Ay = read_2d_fields(pic_info, "../../data/Ay.gda", **kwargs)
    nx, = x.shape
    nz, = z.shape
    width = 0.75
    height = 0.7
    xs = 0.12
    ys = 0.9 - height
    fig = plt.figure(figsize=[10, 4])
    ax1 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep": 1, "zstep": 1, "vmin": -0.1, "vmax": 0.1}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, ey, ax1, fig, **kwargs_plot)
    p1.set_cmap(cmaps.plasma)
    ax1.contour(
        x[0:nx:xstep],
        z[0:nz:zstep],
        Ay[0:nz:zstep, 0:nx:xstep],
        colors='black',
        linewidths=0.5)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax1.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
    ax1.tick_params(labelsize=24)
    cbar1.ax.set_ylabel(r'$E_y$', fontdict=font, fontsize=24)
    cbar1.set_ticks(np.arange(-0.08, 0.1, 0.04))
    cbar1.ax.tick_params(labelsize=24)

    t_wci = current_time * pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(t_wci) + '/\Omega_{ci}$'
    ax1.set_title(title, fontdict=font, fontsize=24)

    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    if not os.path.isdir('../img/img_ey/'):
        os.makedirs('../img/img_ey/')
    fname = '../img/img_ey/ey_' + str(current_time).zfill(3) + '.jpg'
    fig.savefig(fname, dpi=200)

    plt.show()
    # plt.close()


def plot_jy_Ey(pic_info, species, current_time):
    """Plot out-of-plane current density.

    Args:
        pic_info: namedtuple for the PIC simulation information.
        species: 'e' for electrons, 'i' for ions.
        current_time: current time frame.
    """
    print(current_time)
    kwargs = {
        "current_time": current_time,
        "xl": 0,
        "xr": 200,
        "zb": -50,
        "zt": 50
    }
    x, z, ey = read_2d_fields(pic_info, "../data/ey.gda", **kwargs)
    x, z, jy = read_2d_fields(pic_info, "../data/jy.gda", **kwargs)
    x, z, Ay = read_2d_fields(pic_info, "../data/Ay.gda", **kwargs)
    nx, = x.shape
    nz, = z.shape
    width = 0.75
    height = 0.7
    xs = 0.12
    ys = 0.9 - height
    fig = plt.figure(figsize=[10, 4])
    ax1 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep": 1, "zstep": 1, "vmin": -0.01, "vmax": 0.01}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, jy * ey, ax1, fig, **kwargs_plot)
    p1.set_cmap(plt.cm.seismic)
    ax1.contour(
        x[0:nx:xstep],
        z[0:nz:zstep],
        Ay[0:nz:zstep, 0:nx:xstep],
        colors='black',
        linewidths=0.5)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax1.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
    ax1.tick_params(labelsize=24)
    cbar1.ax.set_ylabel(r'$j_yE_y$', fontdict=font, fontsize=24)
    cbar1.set_ticks(np.arange(-0.01, 0.015, 0.01))
    cbar1.ax.tick_params(labelsize=24)

    t_wci = current_time * pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(t_wci) + '/\Omega_{ci}$'
    ax1.set_title(title, fontdict=font, fontsize=24)

    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    if not os.path.isdir('../img/img_jy_ey/'):
        os.makedirs('../img/img_jy_ey/')
    fname = '../img/img_jy_ey/jy_ey' + '_' + str(current_time).zfill(
        3) + '.jpg'
    fig.savefig(fname, dpi=200)

    #plt.show()
    plt.close()


def plot_jpolar_dote(pic_info, species, current_time):
    """Plot out-of-plane current density.

    Args:
        pic_info: namedtuple for the PIC simulation information.
        species: 'e' for electrons, 'i' for ions.
        current_time: current time frame.
    """
    print(current_time)
    kwargs = {
        "current_time": current_time,
        "xl": 0,
        "xr": 400,
        "zb": -100,
        "zt": 100
    }
    x, z, jpolar_dote = read_2d_fields(
        pic_info, "../../data1/jpolar_dote00_e.gda", **kwargs)
    # x, z, Ay = read_2d_fields(pic_info, "../../data/Ay.gda", **kwargs) 
    nx, = x.shape
    nz, = z.shape
    width = 0.75
    height = 0.7
    xs = 0.12
    ys = 0.9 - height
    fig = plt.figure(figsize=[10, 4])
    ax1 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep": 1, "zstep": 1, "vmin": -1, "vmax": 1}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, jpolar_dote, ax1, fig, **kwargs_plot)
    p1.set_cmap(plt.cm.seismic)
    # ax1.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep], 
    #         colors='black', linewidths=0.5)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax1.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
    ax1.tick_params(labelsize=24)
    # cbar1.ax.set_ylabel(r'$j_yE_y$', fontdict=font, fontsize=24)
    # cbar1.set_ticks(np.arange(-0.01, 0.015, 0.01))
    cbar1.ax.tick_params(labelsize=24)
    plt.show()


def plot_epara(pic_info, species, current_time):
    kwargs = {
        "current_time": current_time,
        "xl": 0,
        "xr": 200,
        "zb": -10,
        "zt": 10
    }
    x, z, bx = read_2d_fields(pic_info, "../../data/bx.gda", **kwargs)
    x, z, by = read_2d_fields(pic_info, "../../data/by.gda", **kwargs)
    x, z, bz = read_2d_fields(pic_info, "../../data/bz.gda", **kwargs)
    x, z, absB = read_2d_fields(pic_info, "../../data/absB.gda", **kwargs)
    x, z, ex = read_2d_fields(pic_info, "../../data/ex.gda", **kwargs)
    x, z, ey = read_2d_fields(pic_info, "../../data/ey.gda", **kwargs)
    x, z, ez = read_2d_fields(pic_info, "../../data/ez.gda", **kwargs)
    x, z, Ay = read_2d_fields(pic_info, "../../data/Ay.gda", **kwargs)

    absE = np.sqrt(ex * ex + ey * ey + ez * ez)
    epara = (ex * bx + ey * by + ez * bz) / absB
    eperp = np.sqrt(absE * absE - epara * epara)
    ng = 3
    kernel = np.ones((ng, ng)) / float(ng * ng)
    epara = signal.convolve2d(epara, kernel)
    eperp = signal.convolve2d(eperp, kernel)

    nx, = x.shape
    nz, = z.shape
    width = 0.79
    height = 0.37
    xs = 0.12
    ys = 0.92 - height
    gap = 0.05

    fig = plt.figure(figsize=[7, 5])
    ax1 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep": 2, "zstep": 2, "vmin": 0, "vmax": 0.1}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, eperp, ax1, fig, **kwargs_plot)
    # p1.set_cmap(cmaps.inferno)
    p1.set_cmap(plt.cm.get_cmap('hot'))
    Ay_min = np.min(Ay)
    Ay_max = np.max(Ay)
    levels = np.linspace(Ay_min, Ay_max, 10)
    ax1.contour(
        x[0:nx:xstep],
        z[0:nz:zstep],
        Ay[0:nz:zstep, 0:nx:xstep],
        colors='white',
        linewidths=0.5)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    # ax1.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
    ax1.tick_params(axis='x', labelbottom='off')
    ax1.tick_params(labelsize=16)
    # cbar1.ax.set_ylabel(r'$E_\perp$', fontdict=font, fontsize=20)
    cbar1.set_ticks(np.arange(0, 0.1, 0.03))
    cbar1.ax.tick_params(labelsize=16)
    ax1.text(
        0.02,
        0.8,
        r'$E_\perp$',
        color='w',
        fontsize=20,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='left',
        verticalalignment='center',
        transform=ax1.transAxes)

    ys -= height + gap
    ax2 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep": 1, "zstep": 1, "vmin": -0.05, "vmax": 0.05}
    p2, cbar2 = plot_2d_contour(x, z, epara, ax2, fig, **kwargs_plot)
    p2.set_cmap(plt.cm.seismic)
    # p2.set_cmap(cmaps.plasma)
    ax2.contour(
        x[0:nx:xstep],
        z[0:nz:zstep],
        Ay[0:nz:zstep, 0:nx:xstep],
        colors='black',
        linewidths=0.5)
    ax2.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax2.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    ax2.tick_params(labelsize=16)
    # cbar2.ax.set_ylabel(r'$E_\parallel$', fontdict=font, fontsize=24)
    cbar2.set_ticks(np.arange(-0.04, 0.05, 0.02))
    cbar2.ax.tick_params(labelsize=16)
    ax2.text(
        0.02,
        0.8,
        r'$E_\parallel$',
        color='k',
        fontsize=20,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='left',
        verticalalignment='center',
        transform=ax2.transAxes)

    t_wci = current_time * pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(t_wci) + '/\Omega_{ci}$'
    ax1.set_title(title, fontdict=font, fontsize=20)

    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    dir = '../img/img_epara/'
    if not os.path.isdir(dir):
        os.makedirs(dir)
    # fname = dir + 'epara_perp' + '_' + str(current_time).zfill(3) + '.jpg'
    # fig.savefig(fname, dpi=200)
    fname = dir + 'epara_perp' + '_' + str(current_time).zfill(3) + '.eps'
    fig.savefig(fname)

    plt.show()
    # plt.close()


def plot_diff_fields(pic_info, species, current_time):
    """Plot the differential of the fields.

    Args:
        pic_info: namedtuple for the PIC simulation information.
        species: 'e' for electrons, 'i' for ions.
        current_time: current time frame.
    """
    print(current_time)
    kwargs = {
        "current_time": current_time,
        "xl": 0,
        "xr": 200,
        "zb": -20,
        "zt": 20
    }
    x, z, data = read_2d_fields(pic_info, "../../data/absB.gda", **kwargs)
    x, z, Ay = read_2d_fields(pic_info, "../../data/Ay.gda", **kwargs)
    nx, = x.shape
    nz, = z.shape
    width = 0.75
    height = 0.7
    xs = 0.12
    ys = 0.9 - height
    fig = plt.figure(figsize=[10, 4])
    ax1 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep": 1, "zstep": 1, "vmin": -0.01, "vmax": 0.01}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    data_new = np.zeros((nz, nx))
    # data_new[:, 0:nx-1] = data[:, 1:nx] - data[:, 0:nx-1]
    data_new[0:nz - 1, :] = data[1:nz, :] - data[0:nz - 1, :]
    ng = 5
    kernel = np.ones((ng, ng)) / float(ng * ng)
    data_new = signal.convolve2d(data_new, kernel)
    p1, cbar1 = plot_2d_contour(x, z, data_new, ax1, fig, **kwargs_plot)
    p1.set_cmap(plt.cm.get_cmap('seismic'))
    # p1.set_cmap(cmaps.inferno)
    ax1.contour(
        x[0:nx:xstep],
        z[0:nz:zstep],
        Ay[0:nz:zstep, 0:nx:xstep],
        colors='black',
        linewidths=0.5)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax1.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
    ax1.tick_params(labelsize=24)
    cbar1.ax.set_ylabel(r'$j_y$', fontdict=font, fontsize=24)
    cbar1.set_ticks(np.arange(-0.4, 0.5, 0.2))
    cbar1.ax.tick_params(labelsize=24)

    t_wci = current_time * pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(t_wci) + '/\Omega_{ci}$'
    ax1.set_title(title, fontdict=font, fontsize=24)

    plt.show()


def plot_jpara_perp(pic_info, species, current_time):
    """Plot the energy conversion from parallel and perpendicular direction.

    Args:
        pic_info: namedtuple for the PIC simulation information.
        species: 'e' for electrons, 'i' for ions.
        current_time: current time frame.
    """
    print(current_time)
    kwargs = {
        "current_time": current_time,
        "xl": 0,
        "xr": 200,
        "zb": -20,
        "zt": 20
    }
    x, z, data = read_2d_fields(pic_info, "../../data1/jqnvperp_dote00_e.gda",
                                **kwargs)
    x, z, Ay = read_2d_fields(pic_info, "../../data/Ay.gda", **kwargs)
    nx, = x.shape
    nz, = z.shape
    width = 0.75
    height = 0.7
    xs = 0.12
    ys = 0.9 - height
    fig = plt.figure(figsize=[10, 4])
    ax1 = fig.add_axes([xs, ys, width, height])
    dmax = -0.0005
    kwargs_plot = {"xstep": 1, "zstep": 1, "vmin": -dmax, "vmax": dmax}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    data_new = np.zeros((nz, nx))
    ng = 5
    kernel = np.ones((ng, ng)) / float(ng * ng)
    data_new = signal.convolve2d(data, kernel)
    p1, cbar1 = plot_2d_contour(x, z, data_new, ax1, fig, **kwargs_plot)
    p1.set_cmap(plt.cm.get_cmap('seismic'))
    # p1.set_cmap(cmaps.inferno)
    ax1.contour(
        x[0:nx:xstep],
        z[0:nz:zstep],
        Ay[0:nz:zstep, 0:nx:xstep],
        colors='black',
        linewidths=0.5)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax1.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
    ax1.tick_params(labelsize=24)
    # cbar1.ax.set_ylabel(r'$j_y$', fontdict=font, fontsize=24)
    # cbar1.set_ticks(np.arange(-0.4, 0.5, 0.2))
    cbar1.ax.tick_params(labelsize=24)

    t_wci = current_time * pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(t_wci) + '/\Omega_{ci}$'
    ax1.set_title(title, fontdict=font, fontsize=24)

    plt.show()


def plot_ux(pic_info, species, current_time):
    """Plot the in-plane bulk velocity field.

    Args:
        pic_info: namedtuple for the PIC simulation information.
        species: 'e' for electrons, 'i' for ions.
        current_time: current time frame.
    """
    print(current_time)
    kwargs = {
        "current_time": current_time,
        "xl": 0,
        "xr": 200,
        "zb": -50,
        "zt": 50
    }
    fname = "../../data/vex.gda"
    if not os.path.isfile(fname):
        fname = "../../data/uex.gda"
    x, z, uex = read_2d_fields(pic_info, fname, **kwargs)
    x, z, ne = read_2d_fields(pic_info, "../../data/ne.gda", **kwargs)
    fname = "../../data/vix.gda"
    if not os.path.isfile(fname):
        fname = "../../data/uix.gda"
    x, z, uix = read_2d_fields(pic_info, fname, **kwargs)
    x, z, ni = read_2d_fields(pic_info, "../../data/ni.gda", **kwargs)
    x, z, Ay = read_2d_fields(pic_info, "../../data/Ay.gda", **kwargs)
    wpe_wce = pic_info.dtwce / pic_info.dtwpe
    va = wpe_wce / math.sqrt(pic_info.mime)  # Alfven speed of inflow region
    ux = (uex * ne + uix * ni * pic_info.mime) / (ne + ni * pic_info.mime)
    ux /= va
    nx, = x.shape
    nz, = z.shape
    width = 0.79
    height = 0.37
    xs = 0.13
    ys = 0.92 - height
    gap = 0.05
    # width = 0.75
    # height = 0.4
    # xs = 0.15
    # ys = 0.92 - height
    fig = plt.figure(figsize=[7, 5])
    ax1 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep": 2, "zstep": 2, "vmin": -1.0, "vmax": 1.0}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, ux, ax1, fig, **kwargs_plot)
    p1.set_cmap(plt.cm.get_cmap('seismic'))
    # p1.set_cmap(cmaps.inferno)
    ax1.contour(
        x[0:nx:xstep],
        z[0:nz:zstep],
        Ay[0:nz:zstep, 0:nx:xstep],
        colors='black',
        linewidths=0.5)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax1.tick_params(axis='x', labelbottom='off')
    ax1.tick_params(labelsize=16)
    cbar1.set_ticks(np.arange(-0.8, 1.0, 0.4))
    cbar1.ax.tick_params(labelsize=16)
    ax1.text(
        0.02,
        0.8,
        r'$u_x/V_A$',
        color='k',
        fontsize=20,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='left',
        verticalalignment='center',
        transform=ax1.transAxes)
    ax1.plot(
        [np.min(x), np.max(x)], [0, 0], linestyle='--', color='k', linewidth=2)

    t_wci = current_time * pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(t_wci) + '/\Omega_{ci}$'
    ax1.set_title(title, fontdict=font, fontsize=20)

    gap = 0.06
    ys0 = 0.15
    height0 = ys - gap - ys0
    w1, h1 = fig.get_size_inches()
    width0 = width * 0.98 - 0.05 / w1
    ax2 = fig.add_axes([xs, ys0, width0, height0])
    ax2.plot(x, ux[nz / 2, :], color='k', linewidth=1)
    ax2.plot([np.min(x), np.max(x)], [0, 0], linestyle='--', color='k')
    ax2.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    ax2.set_ylabel(r'$u_x/V_A$', fontdict=font, fontsize=20)
    ax2.tick_params(labelsize=16)
    ax2.set_ylim([-1, 1])

    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    dir = '../img/velocity/'
    if not os.path.isdir(dir):
        os.makedirs(dir)
    fname = dir + 'vx_' + str(current_time) + '.eps'
    fig.savefig(fname)

    plt.show()
    # plt.close()


def plot_uy(pic_info, current_time):
    """Plot the out-of-plane bulk velocity field.

    Args:
        pic_info: namedtuple for the PIC simulation information.
        current_time: current time frame.
    """
    print(current_time)
    kwargs = {
        "current_time": current_time,
        "xl": 0,
        "xr": 200,
        "zb": -50,
        "zt": 50
    }
    fname = "../../data/vey.gda"
    if not os.path.isfile(fname):
        fname = "../../data/uey.gda"
    x, z, uey = read_2d_fields(pic_info, fname, **kwargs)
    fname = "../../data/viy.gda"
    if not os.path.isfile(fname):
        fname = "../../data/uiy.gda"
    x, z, uiy = read_2d_fields(pic_info, fname, **kwargs)
    x, z, Ay = read_2d_fields(pic_info, "../../data/Ay.gda", **kwargs)
    wpe_wce = pic_info.dtwce / pic_info.dtwpe
    va = wpe_wce / math.sqrt(pic_info.mime)  # Alfven speed of inflow region
    uey /= va
    uiy /= va
    nx, = x.shape
    nz, = z.shape
    width = 0.79
    height = 0.37
    xs = 0.13
    ys = 0.92 - height
    gap = 0.05
    fig = plt.figure(figsize=[7, 5])
    ax1 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep": 2, "zstep": 2, "vmin": -0.5, "vmax": 0.5}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, uey, ax1, fig, **kwargs_plot)
    p1.set_cmap(plt.cm.get_cmap('seismic'))
    # p1.set_cmap(cmaps.inferno)
    ax1.contour(
        x[0:nx:xstep],
        z[0:nz:zstep],
        Ay[0:nz:zstep, 0:nx:xstep],
        colors='black',
        linewidths=0.5)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax1.tick_params(axis='x', labelbottom='off')
    ax1.tick_params(labelsize=16)
    cbar1.set_ticks(np.arange(-0.4, 0.5, 0.2))
    cbar1.ax.tick_params(labelsize=16)
    ax1.text(
        0.02,
        0.8,
        r'$u_{ey}/V_A$',
        color='k',
        fontsize=20,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='left',
        verticalalignment='center',
        transform=ax1.transAxes)

    ys -= height + gap
    ax2 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep": 1, "zstep": 1, "vmin": -0.5, "vmax": 0.5}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p2, cbar2 = plot_2d_contour(x, z, uiy, ax2, fig, **kwargs_plot)
    p2.set_cmap(plt.cm.get_cmap('seismic'))
    # p1.set_cmap(cmaps.inferno)
    ax2.contour(
        x[0:nx:xstep],
        z[0:nz:zstep],
        Ay[0:nz:zstep, 0:nx:xstep],
        colors='black',
        linewidths=0.5)
    ax2.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    ax2.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax2.tick_params(labelsize=16)
    cbar2.set_ticks(np.arange(-0.4, 0.5, 0.2))
    cbar2.ax.tick_params(labelsize=16)
    ax2.text(
        0.02,
        0.8,
        r'$u_{iy}/V_A$',
        color='k',
        fontsize=20,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='left',
        verticalalignment='center',
        transform=ax2.transAxes)

    t_wci = current_time * pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(t_wci) + '/\Omega_{ci}$'
    ax1.set_title(title, fontdict=font, fontsize=20)

    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    dir = '../img/velocity/'
    if not os.path.isdir(dir):
        os.makedirs(dir)
    fname = dir + 'vy_' + str(current_time) + '.eps'
    fig.savefig(fname)

    plt.show()
    # plt.close()


def locate_shock(pic_info, ct, run_name, base_dir='../../', single_file=True):
    """Locate the location of shocks

    Args:
        pic_info: namedtuple for the PIC simulation information.
        species: 'e' for electrons, 'i' for ions.
        ct current time frame.
    """
    xmin, xmax = 0, pic_info.lx_di
    xmax = 105
    zmin, zmax = -0.5 * pic_info.lz_di, 0.5 * pic_info.lz_di
    if single_file:
        kwargs = {
            "current_time": ct,
            "xl": xmin,
            "xr": xmax,
            "zb": zmin,
            "zt": zmax
        }
        fname = base_dir + 'data1/ne.gda'
        x, z, ne = read_2d_fields(pic_info, fname, **kwargs)
        fname = base_dir + 'data1/ni.gda'
        x, z, ni = read_2d_fields(pic_info, fname, **kwargs)
        fname = base_dir + 'data1/vex.gda'
        x, z, vex = read_2d_fields(pic_info, fname, **kwargs)
        fname = base_dir + 'data1/vix.gda'
        x, z, vix = read_2d_fields(pic_info, fname, **kwargs)
        fname_Ay = base_dir + 'data1/Ay.gda'
        if os.path.isfile(fname_Ay):
            x, z, Ay = read_2d_fields(pic_info, fname_Ay, **kwargs)
    else:
        kwargs = {
            "current_time": 0,
            "xl": xmin,
            "xr": xmax,
            "zb": zmin,
            "zt": zmax
        }
        fields_interval = pic_info.fields_interval
        tframe = str(fields_interval * ct)
        fname = base_dir + 'data/ne_' + tframe + '.gda'
        x, z, ne = read_2d_fields(pic_info, fname, **kwargs)
        fname = base_dir + 'data/ni_' + tframe + '.gda'
        x, z, ni = read_2d_fields(pic_info, fname, **kwargs)
        fname = base_dir + 'data/vex_' + tframe + '.gda'
        x, z, vex = read_2d_fields(pic_info, fname, **kwargs)
        fname = base_dir + 'data/vix_' + tframe + '.gda'
        x, z, vix = read_2d_fields(pic_info, fname, **kwargs)
        fname_Ay = base_dir + 'data/Ay_' + tframe + '.gda'
        if os.path.isfile(fname_Ay):
            x, z, Ay = read_2d_fields(pic_info, fname_Ay, **kwargs)
    nx, = x.shape
    nz, = z.shape
    xs = 0
    ne_cum = np.sum(ne, axis=0) / nz
    ne_grad = np.abs(np.gradient(ne_cum))
    max_index1 = np.argmax(ne_grad[xs:])
    ni_cum = np.sum(ni, axis=0) / nz
    ni_grad = np.abs(np.gradient(ni_cum))
    max_index2 = np.argmax(ni_grad[xs:])
    vex_cum = np.sum(vex, axis=0) / nz
    vex_grad = np.abs(np.gradient(vex_cum))
    max_index3 = np.argmax(vex_grad[xs:])
    vix_cum = np.sum(vix, axis=0) / nz
    vix_grad = np.abs(np.gradient(vix_cum))
    max_index4 = np.argmax(vix_grad[xs:])
    max_indices = [max_index1, max_index2, max_index3, max_index4]
    imax = max(max_indices)
    imin = min(max_indices)
    shock_xindex = (sum(max_indices) - imax - imin) / 2
    fdir = '../data/shock_pos/'
    mkdir_p(fdir)
    fname = fdir + 'shock_pos_' + str(ct) + '.txt'
    # imax = max(max_indices)
    np.savetxt(fname, [shock_xindex])

    w1, h1 = 0.75, 0.76
    xs, ys = 0.13, 0.92 - h1
    gap = 0.05

    width, height = 7, 5
    fig = plt.figure(figsize=[width, height])
    ax1 = fig.add_axes([xs, ys, w1, h1])
    kwargs_plot = {"xstep": 2, "zstep": 2, "vmin": 0.5, "vmax": 5}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, ni, ax1, fig, **kwargs_plot)
    p1.set_cmap(plt.cm.viridis)
    nlevels = 15
    if os.path.isfile(fname_Ay):
        levels = np.linspace(np.min(Ay), np.max(Ay), nlevels)
        ax1.contour(
            x[0:nx:xstep],
            z[0:nz:zstep],
            Ay[0:nz:zstep, 0:nx:xstep],
            colors='black',
            linewidths=0.5,
            levels=levels)

    xm = x[shock_xindex]
    shift = 1.0
    ax1.plot([xm, xm], [zmin, zmax], color='white', linewidth=1, linestyle='-')
    xs = xm + shift
    ax1.plot(
        [xs, xs], [zmin, zmax], color='white', linewidth=1, linestyle='--')
    xe = xm - shift
    ax1.plot(
        [xe, xe], [zmin, zmax], color='white', linewidth=1, linestyle='--')
    shift = 5.0
    xs = xm + shift
    ax1.plot([xs, xs], [zmin, zmax], color='red', linewidth=1, linestyle='--')
    xe = xm - shift
    ax1.plot([xe, xe], [zmin, zmax], color='red', linewidth=1, linestyle='--')

    ax1.set_xlim([xmin, xmax])
    ax1.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax1.tick_params(labelsize=16)
    lname = r'$n_i$'
    cbar1.ax.set_ylabel(lname, fontdict=font, fontsize=20)
    cbar1.ax.tick_params(labelsize=16)

    t_wci = ct * pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(t_wci) + '/\Omega_{ci}$'
    ax1.set_title(title, fontdict=font, fontsize=20)

    fig_dir = '../img/img_shock_pos/' + run_name + '/'
    mkdir_p(fig_dir)
    fname = fig_dir + '/nrho_linear_i_' + str(ct).zfill(3) + '.jpg'
    fig.savefig(fname, dpi=200)

    # plt.show()
    plt.close()


def combine_shock_files(ntf, run_name):
    """Combine all shock location files at different time frame

    The shock position is saved in different file because Parallel
    in joblib cannot shock one global array
    
    """
    fdir = '../data/shock_pos/'
    shock_loc = np.zeros(ntf - 1)
    for ct in range(ntf - 1):
        print ct
        fname = fdir + 'shock_pos_' + str(ct) + '.txt'
        shock_loc[ct] = np.genfromtxt(fname)
    shock_loc[0] = 0
    tframe = np.arange(ntf - 1)
    z = np.polyfit(tframe, shock_loc, 1)
    p = np.poly1d(z)
    shock_loc_fitted = p(tframe)
    plt.plot(shock_loc, linewidth=2)
    plt.plot(shock_loc_fitted, linewidth=2)
    plt.show()
    fname = fdir + 'shock_pos_' + run_name + '.txt'
    np.savetxt(fname, shock_loc_fitted, fmt='%d')


def find_closest(A, target):
    #A must be sorted
    idx = A.searchsorted(target)
    idx = np.clip(idx, 1, len(A) - 1)
    left = A[idx - 1]
    right = A[idx]
    idx -= target - left < right - target
    return idx


def plot_pressure(pic_info, species, ct, run_name, xm, base_dir='../../'):
    """Plot plasma pressure

    Args:
        pic_info: namedtuple for the PIC simulation information.
        species: 'e' for electrons, 'i' for ions.
        ct current time frame.
    """
    xmin, xmax = xm - 5, xm + 5
    zmin, zmax = -0.5 * pic_info.lz_di, 0.5 * pic_info.lz_di
    kwargs = {
        "current_time": ct,
        "xl": xmin,
        "xr": xmax,
        "zb": zmin,
        "zt": zmax
    }
    fname = base_dir + 'data1/p' + species + '-xx.gda'
    x, z, pxx = read_2d_fields(pic_info, fname, **kwargs)
    # fname = base_dir + 'data1/p' + species + '-yy.gda'
    # x, z, pyy = read_2d_fields(pic_info, fname, **kwargs) 
    # fname = base_dir + 'data1/p' + species + '-zz.gda'
    # x, z, pzz = read_2d_fields(pic_info, fname, **kwargs) 
    # fname = base_dir + 'data1/p' + species + '-xy.gda'
    # x, z, pxy = read_2d_fields(pic_info, fname, **kwargs) 
    # fname = base_dir + 'data1/p' + species + '-xz.gda'
    # x, z, pxz = read_2d_fields(pic_info, fname, **kwargs) 
    # fname = base_dir + 'data1/p' + species + '-yz.gda'
    # x, z, pyz = read_2d_fields(pic_info, fname, **kwargs) 
    # fname = base_dir + 'data1/bx.gda'
    # x, z, bx = read_2d_fields(pic_info, fname, **kwargs) 
    # fname = base_dir + 'data1/by.gda'
    # x, z, by = read_2d_fields(pic_info, fname, **kwargs) 
    # fname = base_dir + 'data1/bz.gda'
    # x, z, bz = read_2d_fields(pic_info, fname, **kwargs) 
    # fname = base_dir + 'data1/absB.gda'
    # x, z, absB = read_2d_fields(pic_info, fname, **kwargs) 
    fname = base_dir + 'data1/Ay.gda'
    x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
    # ppara = pxx*bx*bx + pyy*by*by + pzz*bz*bz + \
    #         pxy*bx*by*2.0 + pxz*bx*bz*2.0 + pyz*by*bz*2.0
    # ppara /= absB * absB
    # pperp = 0.5 * (pxx+pyy+pzz-ppara)
    # pscalar = (ppara + 2 * pperp) / 3
    nx, = x.shape
    nz, = z.shape

    fdata = pxx

    xcut = xm + 0.5
    xindex = find_closest(x, xcut)
    fdata_cut = fdata[:, xindex]
    # fdata_cut_smooth = signal.savgol_filter(fdata_cut, 11, 2)
    ng = 11
    kernel = np.ones(ng) / float(ng)
    fdata_cut_smooth = np.convolve(fdata_cut, kernel, 'same')

    w1, h1 = 0.4, 0.8
    xs, ys = 0.2, 0.94 - h1
    gap = 0.05

    width, height = 5, 12
    fig = plt.figure(figsize=[width, height])
    ax1 = fig.add_axes([xs, ys, w1, h1])
    cbar_min, cbar_max = 0.025, 0.5
    kwargs_plot = {
        "xstep": 2,
        "zstep": 2,
        "is_cbar": False,
        "vmin": cbar_min,
        "vmax": cbar_max
    }
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1 = plot_2d_contour(x, z, fdata, ax1, fig, **kwargs_plot)
    p1.set_cmap(plt.cm.jet)
    nlevels = 15
    levels = np.linspace(np.min(Ay), np.max(Ay), nlevels)
    ax1.contour(
        x[0:nx:xstep],
        z[0:nz:zstep],
        Ay[0:nz:zstep, 0:nx:xstep],
        colors='black',
        linewidths=0.5,
        levels=levels)
    ax1.set_xlim([xmin, xmax])
    ax1.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax1.tick_params(labelsize=16)

    hcbar = 0.03
    wcbar = 0.9
    xs1 = 0.05
    ys1 = ys - 0.06 - hcbar
    cax = fig.add_axes([xs, ys1, w1, hcbar])
    cbar = fig.colorbar(p1, cax=cax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=16)
    cbar.set_ticks(np.arange(0.1, cbar_max + 0.01, 0.2))

    ax1.plot([xcut, xcut], [zmin, zmax], color='white', linestyle='--')

    t_wci = ct * pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(t_wci) + '/\Omega_{ci}$'
    ax1.set_title(title, fontdict=font, fontsize=24)

    xs += w1 + 0.05
    w2 = 0.3
    ax2 = fig.add_axes([xs, ys, w2, h1])
    # ax2.plot(fdata_cut, z, linewidth=1, color='k')
    ax2.plot(fdata_cut_smooth, z, linewidth=2, color='r')
    print np.min(fdata_cut), np.max(fdata_cut)
    ax2.set_xlim([0.0, 0.12])
    ax2.set_ylim([zmin, zmax])
    ax2.xaxis.set_ticks([0.0, 0.05, 0.1])
    ax2.tick_params(labelsize=16)
    ax2.tick_params(axis='y', labelleft='off')

    # fig_dir = '../img/img_number_densities/' + run_name + '/'
    # mkdir_p(fig_dir)
    # fname = fig_dir + '/nrho_linear_' + species + '_' + str(ct).zfill(3) + '.jpg'
    # fig.savefig(fname)

    plt.show()
    # plt.close()


def plot_magnetic_field(pic_info, ct, run_name, xm, base_dir='../../'):
    """Plot magnetic field 

    Args:
        pic_info: namedtuple for the PIC simulation information.
        species: 'e' for electrons, 'i' for ions.
        ct current time frame.
    """
    xmin, xmax = xm - 5, xm + 5
    zmin, zmax = -0.5 * pic_info.lz_di, 0.5 * pic_info.lz_di
    kwargs = {
        "current_time": ct,
        "xl": xmin,
        "xr": xmax,
        "zb": zmin,
        "zt": zmax
    }
    fname = base_dir + 'data1/bx.gda'
    x, z, bx = read_2d_fields(pic_info, fname, **kwargs)
    fname = base_dir + 'data1/by.gda'
    x, z, by = read_2d_fields(pic_info, fname, **kwargs)
    fname = base_dir + 'data1/bz.gda'
    x, z, bz = read_2d_fields(pic_info, fname, **kwargs)
    fname = base_dir + 'data1/Ay.gda'
    x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
    nx, = x.shape
    nz, = z.shape

    xcut = xm + 0.5
    xindex = find_closest(x, xcut)
    ng = 11
    kernel = np.ones(ng) / float(ng)
    bx_cut = bx[:, xindex]
    by_cut = by[:, xindex]
    bz_cut = bz[:, xindex]
    # bx_cut_smooth = np.convolve(bx_cut, kernel, 'same')
    # by_cut_smooth = np.convolve(by_cut, kernel, 'same')
    # bz_cut_smooth = np.convolve(bz_cut, kernel, 'same')

    w1, h1 = 0.2, 0.8
    xs, ys = 0.1, 0.94 - h1
    xs0, ys0 = xs, ys
    gap = 0.03
    cmap = plt.cm.seismic
    cbar_min, cbar_max = -1.0, 1.0

    width, height = 10, 12
    fig = plt.figure(figsize=[width, height])
    ax11 = fig.add_axes([xs, ys, w1, h1])
    kwargs_plot = {
        "xstep": 2,
        "zstep": 2,
        "is_cbar": False,
        "vmin": cbar_min,
        "vmax": cbar_max
    }
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1 = plot_2d_contour(x, z, bx, ax11, fig, **kwargs_plot)
    p1.set_cmap(cmap)
    nlevels = 15
    levels = np.linspace(np.min(Ay), np.max(Ay), nlevels)
    ax11.contour(
        x[0:nx:xstep],
        z[0:nz:zstep],
        Ay[0:nz:zstep, 0:nx:xstep],
        colors='black',
        linewidths=0.5,
        levels=levels)
    ax11.set_xlim([xmin, xmax])
    ax11.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
    ax11.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax11.tick_params(labelsize=16)

    xs += w1 + gap
    ax12 = fig.add_axes([xs, ys, w1, h1])
    p2 = plot_2d_contour(x, z, by, ax12, fig, **kwargs_plot)
    p2.set_cmap(cmap)
    ax12.contour(
        x[0:nx:xstep],
        z[0:nz:zstep],
        Ay[0:nz:zstep, 0:nx:xstep],
        colors='black',
        linewidths=0.5,
        levels=levels)
    ax12.set_xlim([xmin, xmax])
    ax12.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
    ax12.set_ylabel('')
    ax12.tick_params(axis='y', labelleft='off')
    ax12.tick_params(labelsize=16)

    xs += w1 + gap
    ax13 = fig.add_axes([xs, ys, w1, h1])
    p3 = plot_2d_contour(x, z, bz, ax13, fig, **kwargs_plot)
    p3.set_cmap(cmap)
    ax13.contour(
        x[0:nx:xstep],
        z[0:nz:zstep],
        Ay[0:nz:zstep, 0:nx:xstep],
        colors='black',
        linewidths=0.5,
        levels=levels)
    ax13.set_xlim([xmin, xmax])
    ax13.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
    ax13.set_ylabel('')
    ax13.tick_params(axis='y', labelleft='off')
    ax13.tick_params(labelsize=16)

    ax11.plot([xcut, xcut], [zmin, zmax], color='k', linestyle='--')
    ax12.plot([xcut, xcut], [zmin, zmax], color='k', linestyle='--')
    ax13.plot([xcut, xcut], [zmin, zmax], color='w', linestyle='--')

    hcbar = 0.02
    wcbar = w1 * 3 + gap * 2
    ys1 = ys - 0.06 - hcbar
    cax = fig.add_axes([xs0, ys1, wcbar, hcbar])
    cbar = fig.colorbar(p1, cax=cax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=16)

    t_wci = ct * pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(t_wci) + '/\Omega_{ci}$'
    ax12.set_title(title, fontdict=font, fontsize=24)

    ax11.text(
        0.98,
        0.95,
        r'$B_x$',
        color='k',
        fontsize=24,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='right',
        verticalalignment='bottom',
        transform=ax11.transAxes)
    ax12.text(
        0.98,
        0.95,
        r'$B_y$',
        color='k',
        fontsize=24,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='right',
        verticalalignment='bottom',
        transform=ax12.transAxes)
    ax13.text(
        0.98,
        0.95,
        r'$B_z$',
        color='w',
        fontsize=24,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='right',
        verticalalignment='bottom',
        transform=ax13.transAxes)

    xs += w1 + gap
    w2 = 0.2
    ax2 = fig.add_axes([xs, ys, w2, h1])
    # ax2.plot(fdata_cut, z, linewidth=1, color='k')
    ax2.plot(bx_cut, z, linewidth=2, color='r')
    ax2.plot(by_cut, z, linewidth=2, color='g')
    ax2.plot(bz_cut, z, linewidth=2, color='b')
    ax2.set_ylim([zmin, zmax])
    ax2.set_xlim([-1.2, 1.2])
    ax2.xaxis.set_ticks([-1.2, -0.6, 0.0, 0.6, 1.2])
    ax2.tick_params(labelsize=16)
    ax2.tick_params(axis='y', labelleft='off')

    ax2.text(
        0.3,
        1.0,
        r'$B_x$',
        color='r',
        fontsize=24,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax2.transAxes)
    ax2.text(
        0.5,
        1.0,
        r'$B_y$',
        color='g',
        fontsize=24,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax2.transAxes)
    ax2.text(
        0.7,
        1.0,
        r'$B_z$',
        color='b',
        fontsize=24,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax2.transAxes)

    fig_dir = '../img/img_bfields/' + run_name + '/'
    mkdir_p(fig_dir)
    fname = fig_dir + '/bf_' + str(ct).zfill(3) + '.jpg'
    fig.savefig(fname)

    # plt.show()
    plt.close()


def plot_electric_field(pic_info, ct, run_name, xm, base_dir='../../'):
    """Plot electric field 

    Args:
        pic_info: namedtuple for the PIC simulation information.
        species: 'e' for electrons, 'i' for ions.
        ct current time frame.
    """
    xmin, xmax = xm - 5, xm + 5
    zmin, zmax = -0.5 * pic_info.lz_di, 0.5 * pic_info.lz_di
    kwargs = {
        "current_time": ct,
        "xl": xmin,
        "xr": xmax,
        "zb": zmin,
        "zt": zmax
    }
    fname = base_dir + 'data1/ex.gda'
    x, z, ex = read_2d_fields(pic_info, fname, **kwargs)
    fname = base_dir + 'data1/ey.gda'
    x, z, ey = read_2d_fields(pic_info, fname, **kwargs)
    fname = base_dir + 'data1/ez.gda'
    x, z, ez = read_2d_fields(pic_info, fname, **kwargs)
    fname = base_dir + 'data1/Ay.gda'
    x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
    nx, = x.shape
    nz, = z.shape

    ng = 5
    kernel = np.ones((ng, ng)) / float(ng * ng)
    ex = signal.convolve2d(ex, kernel, 'same')
    ey = signal.convolve2d(ey, kernel, 'same')
    ez = signal.convolve2d(ez, kernel, 'same')

    xcut = xm + 0.5
    xindex = find_closest(x, xcut)
    ng = 11
    kernel = np.ones(ng) / float(ng)
    ex_cut = ex[:, xindex]
    ex_cut_smooth = np.convolve(ex_cut, kernel, 'same')
    ey_cut = ey[:, xindex]
    ey_cut_smooth = np.convolve(ey_cut, kernel, 'same')
    ez_cut = ez[:, xindex]
    ez_cut_smooth = np.convolve(ez_cut, kernel, 'same')

    w1, h1 = 0.2, 0.8
    xs, ys = 0.1, 0.94 - h1
    xs0, ys0 = xs, ys
    gap = 0.03
    cmap = plt.cm.seismic

    width, height = 10, 12
    fig = plt.figure(figsize=[width, height])
    ax11 = fig.add_axes([xs, ys, w1, h1])
    cbar_min, cbar_max = -0.2, 0.2
    kwargs_plot = {
        "xstep": 2,
        "zstep": 2,
        "is_cbar": False,
        "vmin": cbar_min,
        "vmax": cbar_max
    }
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1 = plot_2d_contour(x, z, ex, ax11, fig, **kwargs_plot)
    p1.set_cmap(cmap)
    nlevels = 15
    levels = np.linspace(np.min(Ay), np.max(Ay), nlevels)
    ax11.contour(
        x[0:nx:xstep],
        z[0:nz:zstep],
        Ay[0:nz:zstep, 0:nx:xstep],
        colors='black',
        linewidths=0.5,
        levels=levels)
    ax11.set_xlim([xmin, xmax])
    ax11.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
    ax11.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax11.tick_params(labelsize=16)

    xs += w1 + gap
    ax12 = fig.add_axes([xs, ys, w1, h1])
    p2 = plot_2d_contour(x, z, ey, ax12, fig, **kwargs_plot)
    p2.set_cmap(cmap)
    ax12.contour(
        x[0:nx:xstep],
        z[0:nz:zstep],
        Ay[0:nz:zstep, 0:nx:xstep],
        colors='black',
        linewidths=0.5,
        levels=levels)
    ax12.set_xlim([xmin, xmax])
    ax12.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
    ax12.set_ylabel('')
    ax12.tick_params(axis='y', labelleft='off')
    ax12.tick_params(labelsize=16)

    xs += w1 + gap
    ax13 = fig.add_axes([xs, ys, w1, h1])
    p3 = plot_2d_contour(x, z, ez, ax13, fig, **kwargs_plot)
    p3.set_cmap(cmap)
    ax13.contour(
        x[0:nx:xstep],
        z[0:nz:zstep],
        Ay[0:nz:zstep, 0:nx:xstep],
        colors='black',
        linewidths=0.5,
        levels=levels)
    ax13.set_xlim([xmin, xmax])
    ax13.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
    ax13.set_ylabel('')
    ax13.tick_params(axis='y', labelleft='off')
    ax13.tick_params(labelsize=16)

    ax11.plot([xcut, xcut], [zmin, zmax], color='k', linestyle='--')
    ax12.plot([xcut, xcut], [zmin, zmax], color='k', linestyle='--')
    ax13.plot([xcut, xcut], [zmin, zmax], color='k', linestyle='--')

    hcbar = 0.02
    wcbar = w1 * 3 + gap * 2
    ys1 = ys - 0.06 - hcbar
    cax = fig.add_axes([xs0, ys1, wcbar, hcbar])
    cbar = fig.colorbar(p1, cax=cax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=16)

    t_wci = ct * pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(t_wci) + '/\Omega_{ci}$'
    ax12.set_title(title, fontdict=font, fontsize=24)

    ax11.text(
        0.98,
        0.95,
        r'$E_x$',
        color='k',
        fontsize=24,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='right',
        verticalalignment='bottom',
        transform=ax11.transAxes)
    ax12.text(
        0.98,
        0.95,
        r'$E_y$',
        color='k',
        fontsize=24,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='right',
        verticalalignment='bottom',
        transform=ax12.transAxes)
    ax13.text(
        0.98,
        0.95,
        r'$E_z$',
        color='k',
        fontsize=24,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='right',
        verticalalignment='bottom',
        transform=ax13.transAxes)

    xs += w1 + gap
    w2 = 0.2
    ax2 = fig.add_axes([xs, ys, w2, h1])
    # ax2.plot(fdata_cut, z, linewidth=1, color='k')
    ax2.plot(ex_cut, z, linewidth=2, color='r')
    ax2.plot(ey_cut, z, linewidth=2, color='g')
    ax2.plot(ez_cut, z, linewidth=2, color='b')
    ax2.set_ylim([zmin, zmax])
    ax2.set_xlim([-0.15, 0.15])
    ax2.xaxis.set_ticks([-0.1, 0.0, 0.1])
    ax2.tick_params(labelsize=16)
    ax2.tick_params(axis='y', labelleft='off')

    ax2.text(
        0.3,
        1.0,
        r'$E_x$',
        color='r',
        fontsize=24,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax2.transAxes)
    ax2.text(
        0.5,
        1.0,
        r'$E_y$',
        color='g',
        fontsize=24,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax2.transAxes)
    ax2.text(
        0.7,
        1.0,
        r'$E_z$',
        color='b',
        fontsize=24,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax2.transAxes)

    fig_dir = '../img/img_efields/' + run_name + '/'
    mkdir_p(fig_dir)
    fname = fig_dir + '/ef_' + str(ct).zfill(3) + '.jpg'
    fig.savefig(fname)

    # plt.show()
    plt.close()


def plot_velocity_field(pic_info,
                        species,
                        ct,
                        run_name,
                        xm,
                        base_dir='../../',
                        single_file=True):
    """Plot velocity field 

    Args:
        pic_info: namedtuple for the PIC simulation information.
        species: 'e' for electrons, 'i' for ions.
        ct: current time frame.
        base_dir: the base directory of the run
        single_file: whether all time frames are saved in the same file
    """
    xmin, xmax = xm - 5, xm + 5
    zmin, zmax = -0.5 * pic_info.lz_di, 0.5 * pic_info.lz_di
    if single_file:
        kwargs = {
            "current_time": ct,
            "xl": xmin,
            "xr": xmax,
            "zb": zmin,
            "zt": zmax
        }
        fname = base_dir + 'data1/v' + species + 'x.gda'
        x, z, vx = read_2d_fields(pic_info, fname, **kwargs)
        fname = base_dir + 'data1/v' + species + 'y.gda'
        x, z, vy = read_2d_fields(pic_info, fname, **kwargs)
        fname = base_dir + 'data1/v' + species + 'z.gda'
        x, z, vz = read_2d_fields(pic_info, fname, **kwargs)
        fname = base_dir + 'data1/Ay.gda'
        x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
    else:
        kwargs = {
            "current_time": 0,
            "xl": xmin,
            "xr": xmax,
            "zb": zmin,
            "zt": zmax
        }
        fields_interval = pic_info.fields_interval
        tframe = str(fields_interval * ct)
        fname = base_dir + 'data/v' + species + 'x_' + tframe + '.gda'
        x, z, vx = read_2d_fields(pic_info, fname, **kwargs)
        fname = base_dir + 'data/v' + species + 'y_' + tframe + '.gda'
        x, z, vy = read_2d_fields(pic_info, fname, **kwargs)
        fname = base_dir + 'data/v' + species + 'z_' + tframe + '.gda'
        x, z, vz = read_2d_fields(pic_info, fname, **kwargs)
        # fname = base_dir + 'data/Ay.gda'
        # x, z, Ay = read_2d_fields(pic_info, fname, **kwargs) 
    nx, = x.shape
    nz, = z.shape

    xcut = xm
    xindex = find_closest(x, xcut)
    ng = 11
    kernel = np.ones(ng) / float(ng)
    vx_cut = vx[:, xindex]
    vx_cut_smooth = np.convolve(vx_cut, kernel, 'same')
    vy_cut = vy[:, xindex]
    vy_cut_smooth = np.convolve(vy_cut, kernel, 'same')
    vz_cut = vz[:, xindex]
    vz_cut_smooth = np.convolve(vz_cut, kernel, 'same')

    w1, h1 = 0.2, 0.8
    xs, ys = 0.1, 0.94 - h1
    xs0, ys0 = xs, ys
    gap = 0.03
    cmap = plt.cm.seismic
    cbar_min, cbar_max = -0.2, 0.2

    width, height = 10, 12
    fig = plt.figure(figsize=[width, height])
    ax11 = fig.add_axes([xs, ys, w1, h1])
    kwargs_plot = {
        "xstep": 2,
        "zstep": 2,
        "is_cbar": False,
        "vmin": cbar_min,
        "vmax": cbar_max
    }
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1 = plot_2d_contour(x, z, vx, ax11, fig, **kwargs_plot)
    p1.set_cmap(cmap)
    nlevels = 15
    # levels = np.linspace(np.min(Ay), np.max(Ay), nlevels)
    # ax11.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep], 
    #         colors='black', linewidths=0.5, levels=levels)
    ax11.set_xlim([xmin, xmax])
    ax11.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
    ax11.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax11.tick_params(labelsize=16)

    xs += w1 + gap
    ax12 = fig.add_axes([xs, ys, w1, h1])
    p2 = plot_2d_contour(x, z, vy, ax12, fig, **kwargs_plot)
    p2.set_cmap(cmap)
    # ax12.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep], 
    #         colors='black', linewidths=0.5, levels=levels)
    ax12.set_xlim([xmin, xmax])
    ax12.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
    ax12.set_ylabel('')
    ax12.tick_params(axis='y', labelleft='off')
    ax12.tick_params(labelsize=16)

    xs += w1 + gap
    ax13 = fig.add_axes([xs, ys, w1, h1])
    p3 = plot_2d_contour(x, z, vz, ax13, fig, **kwargs_plot)
    p3.set_cmap(cmap)
    # ax13.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep], 
    #         colors='black', linewidths=0.5, levels=levels)
    ax13.set_xlim([xmin, xmax])
    ax13.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
    ax13.set_ylabel('')
    ax13.tick_params(axis='y', labelleft='off')
    ax13.tick_params(labelsize=16)

    ax11.plot([xcut, xcut], [zmin, zmax], color='k', linestyle='-')
    ax12.plot([xcut, xcut], [zmin, zmax], color='k', linestyle='-')
    ax13.plot([xcut, xcut], [zmin, zmax], color='k', linestyle='-')
    xcut1 = xcut + 1.0
    ax11.plot([xcut1, xcut1], [zmin, zmax], color='k', linestyle='--')
    ax12.plot([xcut1, xcut1], [zmin, zmax], color='k', linestyle='--')
    ax13.plot([xcut1, xcut1], [zmin, zmax], color='k', linestyle='--')
    xcut2 = xcut - 1.0
    ax11.plot([xcut2, xcut2], [zmin, zmax], color='k', linestyle='--')
    ax12.plot([xcut2, xcut2], [zmin, zmax], color='k', linestyle='--')
    ax13.plot([xcut2, xcut2], [zmin, zmax], color='k', linestyle='--')

    hcbar = 0.02
    wcbar = w1 * 3 + gap * 2
    ys1 = ys - 0.06 - hcbar
    cax = fig.add_axes([xs0, ys1, wcbar, hcbar])
    cbar = fig.colorbar(p1, cax=cax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=16)

    t_wci = ct * pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(t_wci) + '/\Omega_{ci}$'
    ax12.set_title(title, fontdict=font, fontsize=24)

    fname = '$V_{' + species + 'x}$'
    ax11.text(
        0.98,
        0.95,
        fname,
        color='w',
        fontsize=24,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='right',
        verticalalignment='bottom',
        transform=ax11.transAxes)
    fname = '$V_{' + species + 'y}$'
    ax12.text(
        0.98,
        0.95,
        fname,
        color='k',
        fontsize=24,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='right',
        verticalalignment='bottom',
        transform=ax12.transAxes)
    fname = '$V_{' + species + 'z}$'
    ax13.text(
        0.98,
        0.95,
        fname,
        color='k',
        fontsize=24,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='right',
        verticalalignment='bottom',
        transform=ax13.transAxes)

    xs += w1 + gap
    w2 = 0.2
    ax2 = fig.add_axes([xs, ys, w2, h1])
    # ax2.plot(fdata_cut, z, linewidth=1, color='k')
    ax2.plot(vx_cut, z, linewidth=2, color='r')
    ax2.plot(vy_cut, z, linewidth=2, color='g')
    ax2.plot(vz_cut, z, linewidth=2, color='b')
    ax2.set_ylim([zmin, zmax])
    ax2.set_xlim([-0.2, 0.15])
    ax2.xaxis.set_ticks([-0.2, -0.1, 0.0, 0.1])
    ax2.tick_params(labelsize=16)
    ax2.tick_params(axis='y', labelleft='off')

    fname = '$V_{' + species + 'x}$'
    ax2.text(
        0.2,
        1.0,
        fname,
        color='r',
        fontsize=24,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax2.transAxes)
    fname = '$V_{' + species + 'y}$'
    ax2.text(
        0.5,
        1.0,
        fname,
        color='g',
        fontsize=24,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax2.transAxes)
    fname = '$V_{' + species + 'z}$'
    ax2.text(
        0.8,
        1.0,
        fname,
        color='b',
        fontsize=24,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='center',
        verticalalignment='bottom',
        transform=ax2.transAxes)

    fig_dir = '../img/img_velocity_shock/' + run_name + '/'
    mkdir_p(fig_dir)
    fname = fig_dir + '/v' + species + '_' + str(ct).zfill(3) + '.jpg'
    fig.savefig(fname, dpi=200)

    # plt.show()
    plt.close()


def plot_energy_band(pic_info,
                     species,
                     ct,
                     run_name,
                     shock_pos,
                     base_dir='../../'):
    """Plot number density of different energy band

    Args:
        pic_info: namedtuple for the PIC simulation information.
        species: 'e' for electrons, 'i' for ions.
        ct current time frame.
    """
    xmin, xmax = 0, pic_info.lx_di
    xmax = 105
    zmin, zmax = -0.5 * pic_info.lz_di, 0.5 * pic_info.lz_di
    kwargs = {
        "current_time": ct,
        "xl": xmin,
        "xr": xmax,
        "zb": zmin,
        "zt": zmax
    }
    fname = base_dir + 'data1/n' + species + '.gda'
    x, z, num_rho = read_2d_fields(pic_info, fname, **kwargs)
    fname = base_dir + 'data1/' + species + 'EB05.gda'
    x, z, eband = read_2d_fields(pic_info, fname, **kwargs)
    fname = base_dir + 'data1/Ay.gda'
    x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
    nx, = x.shape
    nz, = z.shape
    num_rho *= eband
    nrho_cum = np.sum(num_rho, axis=0) / nz
    xm = x[shock_pos]

    w1, h1 = 0.7, 0.52
    xs, ys = 0.15, 0.94 - h1
    gap = 0.05

    width, height = 10, 12
    fig = plt.figure(figsize=[10, 12])
    ax1 = fig.add_axes([xs, ys, w1, h1])
    # kwargs_plot = {"xstep":2, "zstep":2, "is_log":True, "vmin":0.1, "vmax":10}
    kwargs_plot = {"xstep": 2, "zstep": 2, "vmin": 0.5, "vmax": 5}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, num_rho, ax1, fig, **kwargs_plot)
    p1.set_cmap(plt.cm.jet)
    nlevels = 15
    levels = np.linspace(np.min(Ay), np.max(Ay), nlevels)
    ax1.contour(
        x[0:nx:xstep],
        z[0:nz:zstep],
        Ay[0:nz:zstep, 0:nx:xstep],
        colors='black',
        linewidths=0.5,
        levels=levels)
    ax1.set_xlim([xmin, xmax])
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax1.tick_params(labelsize=24)
    lname = r'$n_' + species + '$'
    cbar1.ax.set_ylabel(lname, fontdict=font, fontsize=24)
    cbar1.ax.tick_params(labelsize=24)

    ax1.plot([xm, xm], [zmin, zmax], color='white', linestyle='--')

    t_wci = ct * pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(t_wci) + '/\Omega_{ci}$'
    ax1.set_title(title, fontdict=font, fontsize=24)

    h2 = 0.3
    ys -= gap + h2
    w2 = w1 * 0.98 - 0.05 / width
    ax2 = fig.add_axes([xs, ys, w2, h2])
    ax2.plot(x, nrho_cum, linewidth=2, color='k')
    ax2.set_xlim([xmin, xmax])
    # ax2.set_ylim([0.5, 4.5])
    ax2.plot([xm, xm], ax2.get_ylim(), color='k', linestyle='--')
    ax2.tick_params(labelsize=24)
    ax2.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)

    # fig_dir = '../img/img_number_densities/' + run_name + '/'
    # mkdir_p(fig_dir)
    # fname = fig_dir + '/nrho_linear_' + species + '_' + str(ct).zfill(3) + '.jpg'
    # fig.savefig(fname)

    plt.show()
    # plt.close()


def plot_current_density_2d(pic_info,
                            species,
                            ct,
                            run_name,
                            xshock,
                            base_dir='../../',
                            single_file=True):
    """Plot current density 2D contour

    Args:
        pic_info: namedtuple for the PIC simulation information.
        species: 'e' for electrons, 'i' for ions.
        ct: current time frame.
    """
    xmin, xmax = 0, pic_info.lx_di
    xmax = 105
    zmin, zmax = -0.5 * pic_info.lz_di, 0.5 * pic_info.lz_di
    if single_file:
        kwargs = {
            "current_time": ct,
            "xl": xmin,
            "xr": xmax,
            "zb": zmin,
            "zt": zmax
        }
        fname = base_dir + 'data1/n' + species + '.gda'
        x, z, num_rho = read_2d_fields(pic_info, fname, **kwargs)
        fname_Ay = base_dir + 'data1/Ay.gda'
        if os.path.isfile(fname_Ay):
            x, z, Ay = read_2d_fields(pic_info, fname_Ay, **kwargs)
    else:
        kwargs = {
            "current_time": 0,
            "xl": xmin,
            "xr": xmax,
            "zb": zmin,
            "zt": zmax
        }
        fields_interval = pic_info.fields_interval
        tframe = str(fields_interval * ct)
        fname = base_dir + 'data/jy_' + tframe + '.gda'
        x, z, num_rho = read_2d_fields(pic_info, fname, **kwargs)
        fname_Ay = base_dir + 'data/Ay_' + tframe + '.gda'
        if os.path.isfile(fname_Ay):
            x, z, Ay = read_2d_fields(pic_info, fname_Ay, **kwargs)
    nx, = x.shape
    nz, = z.shape
    nrho_cum = np.sum(num_rho, axis=0) / nz

    w1, h1 = 0.75, 0.76
    xs, ys = 0.13, 0.92 - h1
    gap = 0.05

    width, height = 7, 5
    fig = plt.figure(figsize=[width, height])
    ax1 = fig.add_axes([xs, ys, w1, h1])
    # kwargs_plot = {"xstep":2, "zstep":2, "is_log":True, "vmin":0.1, "vmax":10}
    kwargs_plot = {"xstep": 2, "zstep": 2, "vmin": -0.5, "vmax": 0.5}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, num_rho, ax1, fig, **kwargs_plot)
    p1.set_cmap(plt.cm.seismic)
    nlevels = 15
    if os.path.isfile(fname_Ay):
        levels = np.linspace(np.min(Ay), np.max(Ay), nlevels)
        ax1.contour(
            x[0:nx:xstep],
            z[0:nz:zstep],
            Ay[0:nz:zstep, 0:nx:xstep],
            colors='black',
            linewidths=0.5,
            levels=levels)
    ax1.set_xlim([xmin, xmax])
    ax1.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax1.tick_params(labelsize=16)
    lname = r'$j_y$'
    cbar1.ax.set_ylabel(lname, fontdict=font, fontsize=20)
    cbar1.ax.tick_params(labelsize=16)

    shift = 1.0
    xm = xshock
    ax1.plot([xm, xm], [zmin, zmax], color='k', linewidth=1, linestyle='-')
    xs = xm + shift
    ax1.plot([xs, xs], [zmin, zmax], color='k', linewidth=1, linestyle='--')
    xe = xm - shift
    ax1.plot([xe, xe], [zmin, zmax], color='k', linewidth=1, linestyle='--')
    shift = 5.0
    xs = xm + shift
    ax1.plot([xs, xs], [zmin, zmax], color='k', linewidth=2, linestyle='--')
    xe = xm - shift
    ax1.plot([xe, xe], [zmin, zmax], color='k', linewidth=2, linestyle='--')

    t_wci = ct * pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(t_wci) + '/\Omega_{ci}$'
    ax1.set_title(title, fontdict=font, fontsize=20)

    fig_dir = '../img/img_current_densities/' + run_name + '/'
    mkdir_p(fig_dir)
    fname = fig_dir + '/jy_' + str(ct).zfill(3) + '.jpg'
    fig.savefig(fname, dpi=200)

    # plt.show()
    plt.close()


def plot_number_density_2d(pic_info,
                           species,
                           ct,
                           run_name,
                           xshock,
                           base_dir='../../',
                           single_file=True):
    """Plot number density 2D contour

    Args:
        pic_info: namedtuple for the PIC simulation information.
        species: 'e' for electrons, 'i' for ions.
        ct: current time frame.
        run_name: run name
        xshock: shock position along the x-direction in di
    """
    xmin, xmax = 0, pic_info.lx_di
    xmax = 105
    zmin, zmax = -0.5 * pic_info.lz_di, 0.5 * pic_info.lz_di
    if single_file:
        kwargs = {
            "current_time": ct,
            "xl": xmin,
            "xr": xmax,
            "zb": zmin,
            "zt": zmax
        }
        fname = base_dir + 'data1/n' + species + '.gda'
        x, z, num_rho = read_2d_fields(pic_info, fname, **kwargs)
        fname_Ay = base_dir + 'data1/Ay.gda'
        if os.path.isfile(fname_Ay):
            x, z, Ay = read_2d_fields(pic_info, fname_Ay, **kwargs)
    else:
        kwargs = {
            "current_time": 0,
            "xl": xmin,
            "xr": xmax,
            "zb": zmin,
            "zt": zmax
        }
        fields_interval = pic_info.fields_interval
        tframe = str(fields_interval * ct)
        fname = base_dir + 'data/n' + species + '_' + tframe + '.gda'
        x, z, num_rho = read_2d_fields(pic_info, fname, **kwargs)
        fname_Ay = base_dir + 'data/Ay_' + tframe + '.gda'
        if os.path.isfile(fname_Ay):
            x, z, Ay = read_2d_fields(pic_info, fname_Ay, **kwargs)
    nx, = x.shape
    nz, = z.shape
    nrho_cum = np.sum(num_rho, axis=0) / nz

    w1, h1 = 0.75, 0.76
    xs, ys = 0.13, 0.92 - h1
    gap = 0.05

    width, height = 7, 5
    fig = plt.figure(figsize=[width, height])
    ax1 = fig.add_axes([xs, ys, w1, h1])
    # kwargs_plot = {"xstep":2, "zstep":2, "is_log":True, "vmin":0.1, "vmax":10}
    kwargs_plot = {"xstep": 2, "zstep": 2, "vmin": 0.5, "vmax": 5}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, num_rho, ax1, fig, **kwargs_plot)
    p1.set_cmap(plt.cm.viridis)
    nlevels = 15
    if os.path.isfile(fname_Ay):
        levels = np.linspace(np.min(Ay), np.max(Ay), nlevels)
        ax1.contour(
            x[0:nx:xstep],
            z[0:nz:zstep],
            Ay[0:nz:zstep, 0:nx:xstep],
            colors='black',
            linewidths=0.5,
            levels=levels)
    ax1.set_xlim([xmin, xmax])
    ax1.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax1.tick_params(labelsize=16)
    lname = r'$n_' + species + '$'
    cbar1.ax.set_ylabel(lname, fontdict=font, fontsize=20)
    cbar1.ax.tick_params(labelsize=16)

    shift = 1.0
    xm = xshock
    ax1.plot([xm, xm], [zmin, zmax], color='white', linewidth=1, linestyle='-')
    xs = xm + shift
    ax1.plot(
        [xs, xs], [zmin, zmax], color='white', linewidth=1, linestyle='--')
    xe = xm - shift
    ax1.plot(
        [xe, xe], [zmin, zmax], color='white', linewidth=1, linestyle='--')
    shift = 5.0
    xs = xm + shift
    ax1.plot([xs, xs], [zmin, zmax], color='red', linewidth=1, linestyle='--')
    xe = xm - shift
    ax1.plot([xe, xe], [zmin, zmax], color='red', linewidth=1, linestyle='--')

    t_wci = ct * pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(t_wci) + '/\Omega_{ci}$'
    ax1.set_title(title, fontdict=font, fontsize=20)

    fig_dir = '../img/img_number_densities/' + run_name + '/'
    mkdir_p(fig_dir)
    fname = fig_dir + '/nrho_linear_' + species + '_' + str(ct).zfill(
        3) + '.jpg'
    fig.savefig(fname, dpi=200)

    # plt.show()
    plt.close()


if __name__ == "__main__":
    # pic_info = pic_information.get_pic_info('../../')
    # ntp = pic_info.ntp
    # plot_beta_rho(pic_info)
    # plot_jdote_2d(pic_info)
    # plot_anistropy(pic_info, 'e')
    # plot_phi_parallel(29, pic_info)
    # maps = sorted(m for m in plt.cm.datad if not m.endswith("_r")) # nmaps = len(maps) + 1 # print nmaps
    # for i in range(200):
    #     # plot_number_density(pic_info, 'e', i)
    #     # plot_jy(pic_info, 'e', i)
    #     # plot_Ey(pic_info, 'e', i)
    #     plot_anisotropy(pic_info, i)
    # plot_number_density(pic_info, 'e', 40)
    # plot_jy(pic_info, 'e', 120)
    # for i in range(90, 170, 10):
    #     plot_absB_jy(pic_info, 'e', i)
    # plot_by(pic_info)
    # plot_by_multi()
    # plot_ux(pic_info, 'e', 160)
    # plot_uy(pic_info, 160)
    # plot_diff_fields(pic_info, 'e', 120)
    # plot_jpara_perp(pic_info, 'e', 120)
    # plot_Ey(pic_info, 'e', 40)
    # plot_jy_Ey(pic_info, 'e', 40)
    # for i in range(pic_info.ntf):
    #     plot_Ey(pic_info, 'e', i)

    # for i in range(pic_info.ntf):
    #     plot_jy_Ey(pic_info, 'e', i)
    # plot_jpolar_dote(pic_info, 'e', 30)
    # plot_epara(pic_info, 'e', 35)
    # plot_anisotropy_multi('e')
    # base_dir = '/net/scratch3/xiaocanli/2D-90-Mach4-sheet4-multi/'
    # run_name = '2D-90-Mach4-sheet4-multi'
    base_dir = '/net/scratch2/guofan/for_Senbei/2D-90-Mach4-sheet6-3/'
    run_name = '2D-90-Mach4-sheet6-3'
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    ntf = pic_info.ntf
    print ntf
    ct = ntf - 2
    # ct = 270
    cts = range(ntf - 1)
    xmin, xmax = 0, 105
    zmin, zmax = -0.5 * pic_info.lz_di, 0.5 * pic_info.lz_di
    # kwargs = {"current_time":ct, "xl":xmin, "xr":xmax, "zb":zmin, "zt":zmax}
    # fname = base_dir + 'data1/vex.gda'
    # x, z, pxx = read_2d_fields(pic_info, fname, **kwargs) 
    kwargs = {
        "current_time": 0,
        "xl": xmin,
        "xr": xmax,
        "zb": zmin,
        "zt": zmax
    }
    fields_interval = pic_info.fields_interval
    tframe = str(fields_interval * ct)
    fname = base_dir + 'data/vex_' + tframe + '.gda'
    x, z, pxx = read_2d_fields(pic_info, fname, **kwargs)
    fname = '../data/shock_pos/shock_pos_' + run_name + '.txt'
    shock_loc = np.genfromtxt(fname, dtype=np.int32)
    xm = x[shock_loc[ct]]

    def processInput(ct, species):
        print ct
        xm = x[shock_loc[ct]]
        # plot_number_density(pic_info, 'i', ct, run_name, shock_loc[ct], base_dir)
        # plot_vx(pic_info, 'i', ct, run_name, shock_loc[ct], base_dir)
        # locate_shock(pic_info, ct, run_name, base_dir, single_file=False)
        # plot_electric_field(pic_info, ct, run_name, xm, base_dir)
        # plot_magnetic_field(pic_info, ct, run_name, xm, base_dir)
        plot_velocity_field(
            pic_info, species, ct, run_name, xm, base_dir, single_file=False)
        # plot_number_density_2d(pic_info, species, ct, run_name, xm, base_dir,
        #                     single_file=False)
        # plot_current_density_2d(pic_info, species, ct, run_name, xm, base_dir,
        #                     single_file=False)

    num_cores = multiprocessing.cpu_count()
    # Parallel(n_jobs=num_cores)(delayed(processInput)(ct, 'e') for ct in cts)
    Parallel(n_jobs=num_cores)(delayed(processInput)(ct, 'i') for ct in cts)
    # combine_shock_files(ntf, run_name)
    # plot_number_density(pic_info, 'e', ct, run_name, shock_loc[ct], base_dir)
    # plot_number_density_2d(pic_info, 'i', ct, run_name, xm, base_dir,
    #                     single_file=False)
    # plot_current_density_2d(pic_info, 'i', ct, run_name, xm, base_dir,
    #                     single_file=False)
    # plot_vx(pic_info, 'i', ct, run_name, shock_loc[ct], base_dir)
    # plot_electric_field(pic_info, ct, run_name, shock_loc[ct], base_dir)
    # plot_pressure(pic_info, 'e', ct, run_name, xm, base_dir)
    # plot_electric_field(pic_info, ct, run_name, xm, base_dir)
    # plot_magnetic_field(pic_info, ct, run_name, xm, base_dir)
    # plot_velocity_field(pic_info, 'i', ct, run_name, xm, base_dir)
    # plot_velocity_field(pic_info, 'i', ct, run_name, xm, base_dir,
    #                     single_file=False)
    # plot_energy_band(pic_info, 'e', ct, run_name, shock_loc[ct], base_dir)
    # locate_shock(pic_info, ct, run_name, base_dir, single_file=False)
