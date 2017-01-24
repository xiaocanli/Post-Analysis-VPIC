"""
Analysis procedures for 2D contour plots.
"""
import collections
import math
import os
import os.path
import struct

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
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


def read_3d_fields(pic_info, fname):
    """Read 3D fields data from file.
    
    Args:
        pic_info: namedtuple for the PIC simulation information.
        fname: the filename.
        current_time: current time frame.
        xl, xr: left and right x position in di (ion skin length).
        zb, zt: top and bottom z position in di.
    """
    print 'Reading data from ', fname
    nx = pic_info.nx
    ny = pic_info.ny
    nz = pic_info.nz
    fp = np.zeros((nx, ny, nz))
    fp = np.memmap(
        fname,
        dtype='float32',
        mode='r',
        offset=0,
        shape=(nx, ny, nz),
        order='C')
    return fp


def plot_3d_fields(pic_info, fname, iy, tidex):
    fdata = read_3d_fields(pic_info, fname)
    xmin = 0
    xmax = pic_info.lx_di
    ymin = -0.5 * pic_info.ly_di
    ymax = 0.5 * pic_info.ly_di
    zmin = -0.5 * pic_info.lz_di
    zmax = 0.5 * pic_info.lz_di
    nx = pic_info.nx
    ny = pic_info.ny
    nz = pic_info.nz
    width = 0.7
    height = 0.8
    xs = 0.15
    ys = 0.95 - height
    fig = plt.figure(figsize=[7, 5])
    ax = fig.add_axes([xs, ys, width, height])
    data = fdata[nz / 2, :, :]
    print(np.min(data), np.max(data))
    vmin, vmax = -0.1, 0.1
    p1 = ax.imshow(
        data,
        cmap=plt.cm.seismic,
        extent=[xmin, xmax, ymin, ymax],
        aspect='auto',
        origin='lower',
        vmin=vmin,
        vmax=vmax,
        interpolation='bicubic')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(p1, cax=cax)
    cbar.ax.tick_params(labelsize=16)
    ax.set_ylabel(r'$y/d_i$', fontdict=font, fontsize=24)
    ax.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
    ax.tick_params(labelsize=16)
    fname = r'$t\omega_{pe} = ' + str(tindex) + '$'
    ax.text(
        0.05,
        0.9,
        fname,
        color='k',
        fontsize=32,
        horizontalalignment='left',
        verticalalignment='center',
        transform=ax.transAxes)
    cbar.ax.set_ylabel(r'$j_z$', fontdict=font, fontsize=24)


def get_emf(pic_info, root_dir, tindex):
    """Get electromagnetic fields
    """
    fname = root_dir + 'data/ex_' + str(tindex) + '.gda'
    ex = read_3d_fields(pic_info, fname)
    fname = root_dir + 'data/ey_' + str(tindex) + '.gda'
    ey = read_3d_fields(pic_info, fname)
    fname = root_dir + 'data/ez_' + str(tindex) + '.gda'
    ez = read_3d_fields(pic_info, fname)
    fname = root_dir + 'data/bx_' + str(tindex) + '.gda'
    bx = read_3d_fields(pic_info, fname)
    fname = root_dir + 'data/by_' + str(tindex) + '.gda'
    by = read_3d_fields(pic_info, fname)
    fname = root_dir + 'data/bz_' + str(tindex) + '.gda'
    bz = read_3d_fields(pic_info, fname)
    fname = root_dir + 'data/absB_' + str(tindex) + '.gda'
    absB = read_3d_fields(pic_info, fname)
    return (ex, ey, ez, bx, by, bz, absB)


def plot_j_jdote(pic_info, root_dir, iy, tindex, tint):
    fname = root_dir + 'data/jx_' + str(tindex) + '.gda'
    jx = read_3d_fields(pic_info, fname)
    fname = root_dir + 'data/jy_' + str(tindex) + '.gda'
    jy = read_3d_fields(pic_info, fname)
    fname = root_dir + 'data/jz_' + str(tindex) + '.gda'
    jz = read_3d_fields(pic_info, fname)
    fname = root_dir + 'data/absJ_' + str(tindex) + '.gda'
    absJ = read_3d_fields(pic_info, fname)
    ex, ey, ez, bx, by, bz, absB = get_emf(pic_info, root_dir, tindex)
    ex1, ey1, ez1, bx1, by1, bz1, absB1 = \
            get_emf(pic_info, root_dir, tindex - tint)
    ex2, ey2, ez2, bx2, by2, bz2, absB2 = \
            get_emf(pic_info, root_dir, tindex + tint)
    nx = pic_info.nx
    ny = pic_info.ny
    nz = pic_info.nz
    jx0 = jx[nz / 2, :, :]
    jy0 = jy[nz / 2, :, :]
    jz0 = jz[nz / 2, :, :]
    absJ0 = absJ[nz / 2, :, :]
    ex0 = ex[nz / 2, :, :]
    ey0 = ey[nz / 2, :, :]
    ez0 = ez[nz / 2, :, :]
    ex1 = ex1[nz / 2, :, :]
    ey1 = ey1[nz / 2, :, :]
    ez1 = ez1[nz / 2, :, :]
    ex2 = ex2[nz / 2, :, :]
    ey2 = ey2[nz / 2, :, :]
    ez2 = ez2[nz / 2, :, :]
    bx0 = bx[nz / 2, :, :]
    by0 = by[nz / 2, :, :]
    bz0 = bz[nz / 2, :, :]
    bx1 = bx1[nz / 2, :, :]
    by1 = by1[nz / 2, :, :]
    bz1 = bz1[nz / 2, :, :]
    bx2 = bx2[nz / 2, :, :]
    by2 = by2[nz / 2, :, :]
    bz2 = bz2[nz / 2, :, :]
    absB0 = absB[nz / 2, :, :]
    absB1 = absB1[nz / 2, :, :]
    absB2 = absB2[nz / 2, :, :]

    edotb = ex0 * bx0 + ey0 * by0 + ez0 * bz0
    edotb1 = ex1 * bx1 + ey1 * by1 + ez1 * bz1
    edotb2 = ex2 * bx2 + ey2 * by2 + ez2 * bz2
    epara = edotb / absB0
    eperp = np.sqrt(ex0 * ex0 + ey0 * ey0 + ez0 * ez0 - epara * epara)
    epara1 = edotb1 / absB1
    epara2 = edotb2 / absB2
    epara = (epara + epara1 + epara2) / 3.0
    jdote = jx0 * ex0 + jy0 * ey0 + jz0 * ez0
    xmin = 0
    xmax = pic_info.lx_di
    ymin = -0.5 * pic_info.ly_di
    ymax = 0.5 * pic_info.ly_di
    zmin = -0.5 * pic_info.lz_di
    zmax = 0.5 * pic_info.lz_di
    width = 0.21
    height = 0.8
    xs = 0.07
    ys = 0.95 - height
    fig = plt.figure(figsize=[14, 4])
    ax = fig.add_axes([xs, ys, width, height])
    ng = 3
    kernel = np.ones((ng, ng)) / float(ng * ng)
    jdote = signal.convolve2d(jdote, kernel, 'same')
    print(np.min(jdote), np.max(jdote))
    p1 = ax.imshow(
        jdote,
        cmap=plt.cm.seismic,
        extent=[xmin, xmax, ymin, ymax],
        aspect='auto',
        origin='lower',
        vmin=-0.01,
        vmax=0.01,
        interpolation='bicubic')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(p1, cax=cax)
    cbar.ax.tick_params(labelsize=16)
    ax.set_ylabel(r'$y/d_i$', fontdict=font, fontsize=20)
    ax.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    ax.tick_params(labelsize=16)
    fname = r'$t\omega_{pe} = ' + str(tindex) + '$'
    cbar.ax.set_ylabel(
        r'$\boldsymbol{j}\cdot\boldsymbol{E}$', fontdict=font, fontsize=20)
    center = [9.0, 0]
    length = 1.0
    plot_box(center, length, ax, 'k')
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))

    xs += width + 0.12
    ax1 = fig.add_axes([xs, ys, width, height])
    p1 = ax1.imshow(
        absJ0,
        cmap=plt.cm.jet,
        extent=[xmin, xmax, ymin, ymax],
        aspect='auto',
        origin='lower',
        vmin=0,
        vmax=0.3,
        interpolation='bicubic')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar1 = fig.colorbar(p1, cax=cax)
    cbar1.ax.tick_params(labelsize=16)
    ax1.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    ax1.tick_params(labelsize=16)
    dtwpe = pic_info.dtwpe
    fname = r'$t\omega_{pe} = ' + str(int(tindex * dtwpe)) + '$'
    ax1.text(
        -0.45,
        -0.11,
        fname,
        color='k',
        fontsize=32,
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax1.transAxes)
    cbar1.ax.set_ylabel(r'$|\boldsymbol{j}|$', fontdict=font, fontsize=20)
    plot_box(center, length, ax1, 'k')
    ax1.set_xlim((xmin, xmax))
    ax1.set_ylim((ymin, ymax))

    xs += width + 0.1
    ax2 = fig.add_axes([xs, ys, width, height])
    p1 = ax2.imshow(
        eperp,
        cmap=plt.cm.seismic,
        extent=[xmin, xmax, ymin, ymax],
        aspect='auto',
        origin='lower',
        # vmin=-0.05, vmax=0.05,
        vmin=0,
        vmax=0.2,
        interpolation='bicubic')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar1 = fig.colorbar(p1, cax=cax)
    cbar1.ax.tick_params(labelsize=16)
    ax2.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    ax2.tick_params(labelsize=16)
    dtwpe = pic_info.dtwpe
    cbar1.ax.set_ylabel(r'$E_\parallel$', fontdict=font, fontsize=20)
    plot_box(center, length, ax2, 'k')
    ax2.set_xlim((xmin, xmax))
    ax2.set_ylim((ymin, ymax))


def plot_vsingle(pic_info, root_dir, iy, tindex, tint):
    fname = root_dir + 'data/vex_' + str(tindex) + '.gda'
    vex = read_3d_fields(pic_info, fname)
    fname = root_dir + 'data/vey_' + str(tindex) + '.gda'
    vey = read_3d_fields(pic_info, fname)
    fname = root_dir + 'data/vez_' + str(tindex) + '.gda'
    vez = read_3d_fields(pic_info, fname)
    fname = root_dir + 'data/ne_' + str(tindex) + '.gda'
    ne = read_3d_fields(pic_info, fname)
    fname = root_dir + 'data/vix_' + str(tindex) + '.gda'
    vix = read_3d_fields(pic_info, fname)
    fname = root_dir + 'data/viy_' + str(tindex) + '.gda'
    viy = read_3d_fields(pic_info, fname)
    fname = root_dir + 'data/viz_' + str(tindex) + '.gda'
    viz = read_3d_fields(pic_info, fname)
    fname = root_dir + 'data/ni_' + str(tindex) + '.gda'
    ni = read_3d_fields(pic_info, fname)
    nx = pic_info.nx
    ny = pic_info.ny
    nz = pic_info.nz
    vex1 = vex[nz / 2, :, :]
    vey1 = vey[nz / 2, :, :]
    vez1 = vez[nz / 2, :, :]
    ne1 = ne[nz / 2, :, :]
    vix1 = vix[nz / 2, :, :]
    viy1 = viy[nz / 2, :, :]
    viz1 = viz[nz / 2, :, :]
    ni1 = ni[nz / 2, :, :]
    mime = pic_info.mime
    vx = (vex1 * ne1 + vix1 * ni1 * mime) / (ne1 + ni1 * mime)
    vy = (vey1 * ne1 + viy1 * ni1 * mime) / (ne1 + ni1 * mime)
    vz = (vez1 * ne1 + viz1 * ni1 * mime) / (ne1 + ni1 * mime)
    data1 = vx
    data2 = vy
    data3 = vz
    dmin, dmax = -0.2, 0.2
    xmin = 0
    xmax = pic_info.lx_di
    ymin = -0.5 * pic_info.ly_di
    ymax = 0.5 * pic_info.ly_di
    zmin = -0.5 * pic_info.lz_di
    zmax = 0.5 * pic_info.lz_di
    width = 0.21
    height = 0.8
    xs = 0.07
    ys = 0.95 - height
    fig = plt.figure(figsize=[14, 4])
    ax = fig.add_axes([xs, ys, width, height])
    cmap = plt.cm.RdBu_r
    p1 = ax.imshow(
        data1,
        cmap=cmap,
        extent=[xmin, xmax, ymin, ymax],
        aspect='auto',
        origin='lower',
        vmin=dmin,
        vmax=dmax,
        interpolation='bicubic')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(p1, cax=cax)
    cbar.ax.tick_params(labelsize=16)
    ax.set_ylabel(r'$y/d_i$', fontdict=font, fontsize=20)
    ax.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    ax.tick_params(labelsize=16)
    fname = r'$t\omega_{pe} = ' + str(tindex) + '$'
    cbar.ax.set_ylabel(r'$v_x$', fontdict=font, fontsize=20)
    center = [9.0, 0]
    length = 1.0
    plot_box(center, length, ax, 'k')
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))

    xs += width + 0.12
    ax1 = fig.add_axes([xs, ys, width, height])
    p1 = ax1.imshow(
        data2,
        cmap=cmap,
        extent=[xmin, xmax, ymin, ymax],
        aspect='auto',
        origin='lower',
        vmin=dmin,
        vmax=dmax,
        interpolation='bicubic')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar1 = fig.colorbar(p1, cax=cax)
    cbar1.ax.tick_params(labelsize=16)
    ax1.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    ax1.tick_params(labelsize=16)
    dtwpe = pic_info.dtwpe
    fname = r'$t\omega_{pe} = ' + str(int(tindex * dtwpe)) + '$'
    ax1.text(
        -0.45,
        -0.11,
        fname,
        color='k',
        fontsize=32,
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax1.transAxes)
    cbar1.ax.set_ylabel(r'$v_y$', fontdict=font, fontsize=20)
    plot_box(center, length, ax1, 'k')
    ax1.set_xlim((xmin, xmax))
    ax1.set_ylim((ymin, ymax))

    xs += width + 0.1
    ax2 = fig.add_axes([xs, ys, width, height])
    p1 = ax2.imshow(
        data3,
        cmap=cmap,
        extent=[xmin, xmax, ymin, ymax],
        aspect='auto',
        origin='lower',
        vmin=dmin,
        vmax=dmax,
        interpolation='bicubic')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar1 = fig.colorbar(p1, cax=cax)
    cbar1.ax.tick_params(labelsize=16)
    ax2.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    ax2.tick_params(labelsize=16)
    dtwpe = pic_info.dtwpe
    cbar1.ax.set_ylabel(r'$v_z$', fontdict=font, fontsize=20)
    plot_box(center, length, ax2, 'k')
    ax2.set_xlim((xmin, xmax))
    ax2.set_ylim((ymin, ymax))


def plot_box(center, length, ax, color):
    xl = center[0] - length / 2
    xr = center[0] + length / 2
    yb = center[1] - length / 2
    yt = center[1] + length / 2
    xbox = [xl, xr, xr, xl, xl]
    ybox = [yb, yb, yt, yt, yb]
    ax.plot(xbox, ybox, color=color, linewidth=2)


def calc_parallel_potential(pic_info, root_dir, iy, tidex):
    fname = root_dir + 'data/bx_' + str(tindex) + '.gda'
    bx = read_3d_fields(pic_info, fname)
    fname = root_dir + 'data/by_' + str(tindex) + '.gda'
    by = read_3d_fields(pic_info, fname)
    fname = root_dir + 'data/bz_' + str(tindex) + '.gda'
    bz = read_3d_fields(pic_info, fname)
    fname = root_dir + 'data/absB_' + str(tindex) + '.gda'
    absB = read_3d_fields(pic_info, fname)
    fname = root_dir + 'data/ex_' + str(tindex) + '.gda'
    ex = read_3d_fields(pic_info, fname)
    fname = root_dir + 'data/ey_' + str(tindex) + '.gda'
    ey = read_3d_fields(pic_info, fname)
    fname = root_dir + 'data/ez_' + str(tindex) + '.gda'
    ez = read_3d_fields(pic_info, fname)
    nx = pic_info.nx
    ny = pic_info.ny
    nz = pic_info.nz
    dz = pic_info.dz_di * math.sqrt(pic_info.mime)

    edotb = (
        ex * bx + ey * by + ez * bz) / np.sqrt(bx * bx + by * by + bz * bz)
    edotb_cum = np.sum(edotb, axis=0) * dz

    xmin = 0
    xmax = pic_info.lx_di
    ymin = -0.5 * pic_info.ly_di
    ymax = 0.5 * pic_info.ly_di
    zmin = -0.5 * pic_info.lz_di
    zmax = 0.5 * pic_info.lz_di
    width = 0.64
    height = 0.8
    xs = 0.17
    ys = 0.95 - height
    fig = plt.figure(figsize=[7.5, 6])
    ax = fig.add_axes([xs, ys, width, height])
    print(np.min(edotb_cum), np.max(edotb_cum))
    dmin = '%.2f' % np.min(edotb_cum)
    dmax = '%.2f' % np.max(edotb_cum)
    print dmin, dmax
    p1 = ax.imshow(
        edotb_cum,
        cmap=plt.cm.seismic,
        extent=[xmin, xmax, ymin, ymax],
        aspect='auto',
        origin='lower',
        vmin=-1.0,
        vmax=1.0,
        interpolation='bicubic')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(p1, cax=cax)
    cbar.ax.tick_params(labelsize=16)
    ax.set_ylabel(r'$y/d_i$', fontdict=font, fontsize=24)
    ax.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
    ax.tick_params(labelsize=16)
    fname = r'$t\omega_{pe} = ' + str(tindex) + '$'
    cbar.ax.set_ylabel(
        r'$\int_0^{z_\text{max}}E_\parallel dz$', fontdict=font, fontsize=24)
    fname = 'max: ' + dmax
    ax.text(
        0.8,
        -0.08,
        fname,
        color='k',
        fontsize=20,
        horizontalalignment='left',
        verticalalignment='center',
        transform=ax.transAxes)
    fname = 'min: ' + dmin
    ax.text(
        0.8,
        -0.14,
        fname,
        color='k',
        fontsize=20,
        horizontalalignment='left',
        verticalalignment='center',
        transform=ax.transAxes)


def plot_epara(pic_info, root_dir, iy, tidex):
    fname = root_dir + 'data/bx_' + str(tindex) + '.gda'
    bx = read_3d_fields(pic_info, fname)
    fname = root_dir + 'data/by_' + str(tindex) + '.gda'
    by = read_3d_fields(pic_info, fname)
    fname = root_dir + 'data/bz_' + str(tindex) + '.gda'
    bz = read_3d_fields(pic_info, fname)
    fname = root_dir + 'data/absB_' + str(tindex) + '.gda'
    absB = read_3d_fields(pic_info, fname)
    fname = root_dir + 'data/ex_' + str(tindex) + '.gda'
    ex = read_3d_fields(pic_info, fname)
    fname = root_dir + 'data/ey_' + str(tindex) + '.gda'
    ey = read_3d_fields(pic_info, fname)
    fname = root_dir + 'data/ez_' + str(tindex) + '.gda'
    ez = read_3d_fields(pic_info, fname)
    nx = pic_info.nx
    ny = pic_info.ny
    nz = pic_info.nz
    bx0 = bx[nz / 2, :, :]
    by0 = by[nz / 2, :, :]
    bz0 = bz[nz / 2, :, :]
    absB0 = absB[nz / 2, :, :]
    ex0 = ex[nz / 2, :, :]
    ey0 = ey[nz / 2, :, :]
    ez0 = ez[nz / 2, :, :]

    edotb = ex0 * bx0 + ey0 * by0 + ez0 * bz0
    edotb_para = edotb / absB0
    xmin = 0
    xmax = pic_info.lx_di
    ymin = -0.5 * pic_info.ly_di
    ymax = 0.5 * pic_info.ly_di
    zmin = -0.5 * pic_info.lz_di
    zmax = 0.5 * pic_info.lz_di
    width = 0.35
    height = 0.8
    xs = 0.09
    ys = 0.95 - height
    fig = plt.figure(figsize=[14, 5])
    ax = fig.add_axes([xs, ys, width, height])
    ng = 3
    kernel = np.ones((ng, ng)) / float(ng * ng)
    edotb_para = signal.convolve2d(edotb_para, kernel, 'same')
    print(np.min(edotb_para), np.max(edotb_para))
    p1 = ax.imshow(
        edotb_para,
        cmap=plt.cm.seismic,
        extent=[xmin, xmax, ymin, ymax],
        aspect='auto',
        origin='lower',
        vmin=-0.1,
        vmax=0.1,
        interpolation='bicubic')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(p1, cax=cax)
    cbar.ax.tick_params(labelsize=16)
    ax.set_ylabel(r'$y/d_i$', fontdict=font, fontsize=20)
    ax.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    ax.tick_params(labelsize=16)
    fname = r'$t\omega_{pe} = ' + str(tindex) + '$'
    cbar.ax.set_ylabel(
        r'$\boldsymbol{j}\cdot\boldsymbol{E}$', fontdict=font, fontsize=20)


if __name__ == "__main__":
    root_dir = '../../'
    pic_info = pic_information.get_pic_info(root_dir)
    fint = pic_info.fields_interval
    ny = pic_info.ny
    ct = fint * 10
    ntf = pic_info.ntf
    print ntf
    itf = 25
    for ct in range(itf, itf + 1):
        # for ct in range(ntf):
        print ct
        tindex = ct * fint
        # fname = root_dir + 'data/jz_' + str(tindex) + '.gda'
        # # plot_3d_fields(pic_info, fname, ny/2, tindex)
        plot_vsingle(pic_info, root_dir, ny / 2, tindex, fint)
        plot_j_jdote(pic_info, root_dir, ny / 2, tindex, fint)
        # if not os.path.isdir('../img/'):
        #     os.makedirs('../img/')
        # fig_dir = '../img/jy/'
        # if not os.path.isdir(fig_dir):
        #     os.makedirs(fig_dir)
        # fname = fig_dir + 'jy_' + str(ct).zfill(3) + '.jpg'
        # plt.savefig(fname, dpi=200)
        # fname = root_dir + 'data/ez_' + str(tindex) + '.gda'
        # plot_3d_fields(pic_info, fname, ny/2, tindex)
        # if not os.path.isdir('../img/'):
        #     os.makedirs('../img/')
        # fig_dir = '../img/ey/'
        # if not os.path.isdir(fig_dir):
        #     os.makedirs(fig_dir)
        # fname = fig_dir + 'ey_' + str(ct).zfill(3) + '.jpg'
        # plot_epara(pic_info, root_dir, ny/2, tindex)
        # plt.savefig(fname, dpi=200)
        # plt.close()
        # fig_dir = '../img/phi_parallel/'
        # if not os.path.isdir(fig_dir):
        #     os.makedirs(fig_dir)
        # fname = fig_dir + 'phi_' + str(ct).zfill(3) + '.jpg'
        # calc_parallel_potential(pic_info, root_dir, ny/2, tindex)
        # plt.savefig(fname, dpi=200)
        # plt.close()
    plt.show()
