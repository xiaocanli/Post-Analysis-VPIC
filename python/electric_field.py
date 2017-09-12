"""
Functions and classes for 2D contour plots of fields.
"""
import collections
import math
import os
import os.path
import pprint
import re
import stat
import struct
import sys
import itertools
from itertools import groupby
from os import listdir
from os.path import isfile, join

import functools
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from joblib import Parallel, delayed
from matplotlib import rc
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
from scipy import optimize
from scipy.fftpack import fft2, fftshift, ifft2
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.ndimage.filters import generic_filter as gf

import color_maps as cm
import colormap.colormaps as cmaps
import palettable
import pic_information
from energy_conversion import read_data_from_json
from contour_plots import plot_2d_contour, read_2d_fields

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

font = {
    'family': 'serif',
    # 'color':'darkred',
    'color': 'black',
    'weight': 'normal',
    'size': 24,
}

colors = palettable.colorbrewer.qualitative.Set1_9.mpl_colors

# colors = palettable.colorbrewer.qualitative.Dark2_8.mpl_colors


def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]


def plot_epara_eperp(pic_info, ct, root_dir='../../'):
    kwargs = {"current_time": ct, "xl": 50, "xr": 150, "zb": -20, "zt": 20}
    fname = root_dir + 'data/bx.gda'
    x, z, bx = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/by.gda'
    x, z, by = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/bz.gda'
    x, z, bz = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/absB.gda'
    x, z, absB = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/ex.gda'
    x, z, ex = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/ey.gda'
    x, z, ey = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/ez.gda'
    x, z, ez = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/Ay.gda'
    x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)

    fname = root_dir + 'data/pe-xx.gda'
    x, z, pexx = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/pe-xy.gda'
    x, z, pexy = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/pe-xz.gda'
    x, z, pexz = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/pe-yy.gda'
    x, z, peyy = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/pe-yz.gda'
    x, z, peyz = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/pe-zz.gda'
    x, z, pezz = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/ne.gda'
    x, z, ne = read_2d_fields(pic_info, fname, **kwargs)

    absE = np.sqrt(ex * ex + ey * ey + ez * ez)
    epara = (ex * bx + ey * by + ez * bz) / absB
    eperp = np.sqrt(absE * absE - epara * epara)
    ng = 3
    kernel = np.ones((ng, ng)) / float(ng * ng)
    epara = signal.convolve2d(epara, kernel, 'same')
    eperp = signal.convolve2d(eperp, kernel, 'same')
    ey = signal.convolve2d(ey, kernel, 'same')

    dx = x[1] - x[0]
    dz = z[1] - z[0]
    dpex = np.gradient(pexx, dx, axis=1) + np.gradient(pexz, dz, axis=0)
    dpey = np.gradient(pexy, dx, axis=1) + np.gradient(peyz, dz, axis=0)
    dpez = np.gradient(pexz, dx, axis=1) + np.gradient(pezz, dz, axis=0)
    dpex /= ne
    dpey /= ne
    dpez /= ne
    dpe  = np.sqrt(dpex**2 + dpey**2 + dpez**2)

    wpe_wce = pic_info.dtwce / pic_info.dtwpe
    va = wpe_wce / math.sqrt(pic_info.mime)  # Alfven speed of inflow region
    b0 = pic_info.b0
    e0 = va * b0
    epara /= 0.5 * e0
    eperp /= 0.5 * e0
    ey /= 0.5 * e0

    nx, = x.shape
    nz, = z.shape
    xs, ys = 0.1, 0.7
    w1, h1 = 0.85, 0.28
    gap = 0.03

    fig = plt.figure(figsize=[10, 10])
    ax1 = fig.add_axes([xs, ys, w1, h1])
    kwargs_plot = {"xstep":1, "zstep":1, "vmin":-0.5, "vmax":0.5}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, ey, ax1, fig, **kwargs_plot)
    # p1.set_cmap(cmaps.inferno)
    p1.set_cmap(plt.cm.get_cmap('seismic'))
    Ay_min = np.min(Ay)
    Ay_max = np.max(Ay)
    levels = np.linspace(Ay_min, Ay_max, 10)
    ax1.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep],
            colors='k', linewidths=0.5)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax1.tick_params(axis='x', labelbottom='off')
    ax1.tick_params(labelsize=16)
    cbar1.set_ticks(np.arange(-0.5, 0.6, 0.5))
    cbar1.ax.tick_params(labelsize=16)
    ax1.text(0.02, 0.8, r'$E_y$', color='k', fontsize=20,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                pad=10.0), horizontalalignment='left',
            verticalalignment='center', transform = ax1.transAxes)

    ys -= h1 + gap
    ax2 = fig.add_axes([xs, ys, w1, h1])
    kwargs_plot = {"xstep":1, "zstep":1, "vmin":-0.5, "vmax":0.5}
    p2, cbar2 = plot_2d_contour(x, z, epara, ax2, fig, **kwargs_plot)
    p2.set_cmap(plt.cm.seismic)
    ax2.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep],
            colors='black', linewidths=0.5)
    ax2.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax2.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    ax2.tick_params(labelsize=16)
    cbar2.set_ticks(np.arange(-0.5, 0.6, 0.5))
    cbar2.ax.tick_params(labelsize=16)
    ax2.text(0.02, 0.8, r'$E_\parallel$', color='k', fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax2.transAxes)

    ys -= h1 + gap
    ax3 = fig.add_axes([xs, ys, w1, h1])
    kwargs_plot = {"xstep":1, "zstep":1, "vmin":-0.5, "vmax":0.5}
    p3, cbar3 = plot_2d_contour(x, z, dpez, ax3, fig, **kwargs_plot)
    p3.set_cmap(plt.cm.seismic)
    ax3.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep],
            colors='black', linewidths=0.5)
    ax3.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax3.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    ax3.tick_params(labelsize=16)
    cbar3.set_ticks(np.arange(-0.5, 0.6, 0.5))
    cbar3.ax.tick_params(labelsize=16)
    ax3.text(0.02, 0.8, r'$E_\parallel$', color='k', fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax3.transAxes)

    plt.show()
    # plt.close()


def plot_epara(pic_info, ct, root_dir='../../'):
    kwargs = {"current_time": ct, "xl": 50, "xr": 150, "zb": -20, "zt": 20}
    fname = root_dir + 'data/bx.gda'
    x, z, bx = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/by.gda'
    x, z, by = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/bz.gda'
    x, z, bz = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/absB.gda'
    x, z, absB = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/ex.gda'
    x, z, ex = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/ey.gda'
    x, z, ey = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/ez.gda'
    x, z, ez = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/Ay.gda'
    x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)

    fname = root_dir + 'data/pe-xx.gda'
    x, z, pexx = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/pe-xy.gda'
    x, z, pexy = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/pe-xz.gda'
    x, z, pexz = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/pe-yy.gda'
    x, z, peyy = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/pe-yz.gda'
    x, z, peyz = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/pe-zz.gda'
    x, z, pezz = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/ne.gda'
    x, z, ne = read_2d_fields(pic_info, fname, **kwargs)

    absE = np.sqrt(ex * ex + ey * ey + ez * ez)
    epara = (ex * bx + ey * by + ez * bz) / absB
    eperp = np.sqrt(absE * absE - epara * epara)
    ng = 3
    kernel = np.ones((ng, ng)) / float(ng * ng)
    eparax = signal.convolve2d(epara*bx/absB, kernel, 'same')
    eparay = signal.convolve2d(epara*by/absB, kernel, 'same')
    eparaz = signal.convolve2d(epara*bz/absB, kernel, 'same')
    epara = signal.convolve2d(epara, kernel, 'same')
    eperp = signal.convolve2d(eperp, kernel, 'same')
    ey = signal.convolve2d(ey, kernel, 'same')

    dx = x[1] - x[0]
    dz = z[1] - z[0]
    dpex = np.gradient(pexx, dx, axis=1) + np.gradient(pexz, dz, axis=0)
    dpey = np.gradient(pexy, dx, axis=1) + np.gradient(peyz, dz, axis=0)
    dpez = np.gradient(pexz, dx, axis=1) + np.gradient(pezz, dz, axis=0)
    dpex /= -ne
    dpey /= -ne
    dpez /= -ne
    dpe  = np.sqrt(dpex**2 + dpey**2 + dpez**2)
    dpex = signal.convolve2d(dpex, kernel, 'same')
    dpey = signal.convolve2d(dpey, kernel, 'same')
    dpez = signal.convolve2d(dpez, kernel, 'same')

    wpe_wce = pic_info.dtwce / pic_info.dtwpe
    va = wpe_wce / math.sqrt(pic_info.mime)  # Alfven speed of inflow region
    b0 = pic_info.b0
    e0 = va * b0
    epara /= e0
    eperp /= e0
    eparax /= e0
    eparay /= e0
    eparaz /= e0
    dpex /= e0
    dpey /= e0
    dpez /= e0
    ey /= e0

    nx, = x.shape
    nz, = z.shape
    xs, ys = 0.1, 0.7
    w1, h1 = 0.85, 0.28
    gap = 0.03

    # divP_e
    fig = plt.figure(figsize=[10, 10])
    ax1 = fig.add_axes([xs, ys, w1, h1])
    vmin, vmax = -0.2, 0.2
    kwargs_plot = {"xstep":1, "zstep":1, "vmin":vmin, "vmax":vmax}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, dpex, ax1, fig, **kwargs_plot)
    # p1.set_cmap(cmaps.inferno)
    p1.set_cmap(plt.cm.get_cmap('seismic'))
    Ay_min = np.min(Ay)
    Ay_max = np.max(Ay)
    levels = np.linspace(Ay_min, Ay_max, 10)
    ax1.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep],
            colors='k', linewidths=0.5)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax1.tick_params(axis='x', labelbottom='off')
    ax1.tick_params(labelsize=16)
    cbar1.set_ticks(np.arange(vmin, vmax+0.1, 0.1))
    cbar1.ax.tick_params(labelsize=16)
    ax1.text(0.02, 0.8, r'$E_y$', color='k', fontsize=20,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                pad=10.0), horizontalalignment='left',
            verticalalignment='center', transform = ax1.transAxes)

    ys -= h1 + gap
    ax2 = fig.add_axes([xs, ys, w1, h1])
    p2, cbar2 = plot_2d_contour(x, z, dpey, ax2, fig, **kwargs_plot)
    p2.set_cmap(plt.cm.seismic)
    ax2.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep],
            colors='black', linewidths=0.5)
    ax2.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax2.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    ax2.tick_params(labelsize=16)
    cbar2.set_ticks(np.arange(vmin, vmax+0.1, 0.1))
    cbar2.ax.tick_params(labelsize=16)
    ax2.text(0.02, 0.8, r'$E_\parallel$', color='k', fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax2.transAxes)

    ys -= h1 + gap
    ax3 = fig.add_axes([xs, ys, w1, h1])
    p3, cbar3 = plot_2d_contour(x, z, dpez, ax3, fig, **kwargs_plot)
    p3.set_cmap(plt.cm.seismic)
    ax3.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep],
            colors='black', linewidths=0.5)
    ax3.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax3.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    ax3.tick_params(labelsize=16)
    cbar3.set_ticks(np.arange(vmin, vmax+0.1, 0.1))
    cbar3.ax.tick_params(labelsize=16)
    ax3.text(0.02, 0.8, r'$E_\parallel$', color='k', fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax3.transAxes)

    # Epara
    fig = plt.figure(figsize=[10, 10])
    xs, ys = 0.1, 0.7
    w1, h1 = 0.85, 0.28
    ax1 = fig.add_axes([xs, ys, w1, h1])
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, eparax, ax1, fig, **kwargs_plot)
    # p1.set_cmap(cmaps.inferno)
    p1.set_cmap(plt.cm.get_cmap('seismic'))
    Ay_min = np.min(Ay)
    Ay_max = np.max(Ay)
    levels = np.linspace(Ay_min, Ay_max, 10)
    ax1.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep],
            colors='k', linewidths=0.5)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax1.tick_params(axis='x', labelbottom='off')
    ax1.tick_params(labelsize=16)
    cbar1.set_ticks(np.arange(vmin, vmax+0.1, 0.1))
    cbar1.ax.tick_params(labelsize=16)
    ax1.text(0.02, 0.8, r'$E_y$', color='k', fontsize=20,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                pad=10.0), horizontalalignment='left',
            verticalalignment='center', transform = ax1.transAxes)

    ys -= h1 + gap
    ax2 = fig.add_axes([xs, ys, w1, h1])
    p2, cbar2 = plot_2d_contour(x, z, eparay, ax2, fig, **kwargs_plot)
    p2.set_cmap(plt.cm.seismic)
    ax2.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep],
            colors='black', linewidths=0.5)
    ax2.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax2.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    ax2.tick_params(labelsize=16)
    cbar2.set_ticks(np.arange(vmin, vmax+0.1, 0.1))
    cbar2.ax.tick_params(labelsize=16)
    ax2.text(0.02, 0.8, r'$E_\parallel$', color='k', fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax2.transAxes)

    ys -= h1 + gap
    ax3 = fig.add_axes([xs, ys, w1, h1])
    p3, cbar3 = plot_2d_contour(x, z, eparaz, ax3, fig, **kwargs_plot)
    p3.set_cmap(plt.cm.seismic)
    ax3.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep],
            colors='black', linewidths=0.5)
    ax3.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax3.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    ax3.tick_params(labelsize=16)
    cbar3.set_ticks(np.arange(vmin, vmax+0.1, 0.1))
    cbar3.ax.tick_params(labelsize=16)
    ax3.text(0.02, 0.8, r'$E_\parallel$', color='k', fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax3.transAxes)

    plt.show()
    # plt.close()


def plot_jdote(pic_info, ct, root_dir='../../'):
    xl, xr = 50, 150
    kwargs = {"current_time": ct, "xl": xl, "xr": xr, "zb": -20, "zt": 20}
    fname = root_dir + 'data/bx.gda'
    x, z, bx = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/by.gda'
    x, z, by = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/bz.gda'
    x, z, bz = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/absB.gda'
    x, z, absB = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/ex.gda'
    x, z, ex = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/ey.gda'
    x, z, ey = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/ez.gda'
    x, z, ez = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/Ay.gda'
    x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)

    fname = root_dir + 'data/pe-xx.gda'
    x, z, pexx = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/pe-xy.gda'
    x, z, pexy = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/pe-xz.gda'
    x, z, pexz = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/pe-yy.gda'
    x, z, peyy = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/pe-yz.gda'
    x, z, peyz = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/pe-zz.gda'
    x, z, pezz = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/ne.gda'
    x, z, ne = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/vex.gda'
    x, z, vex = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/vey.gda'
    x, z, vey = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/vez.gda'
    x, z, vez = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/ni.gda'
    x, z, ni = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/vix.gda'
    x, z, vix = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/viy.gda'
    x, z, viy = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/viz.gda'
    x, z, viz = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/jx.gda'
    x, z, jx = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/jy.gda'
    x, z, jy = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/jz.gda'
    x, z, jz = read_2d_fields(pic_info, fname, **kwargs)

    # irho = 1.0 / (ne + ni * pic_info.mime)
    # vx = (vex*ne + vix*ni*pic_info.mime) * irho
    # vy = (vey*ne + viy*ni*pic_info.mime) * irho
    # vz = (vez*ne + viz*ni*pic_info.mime) * irho
    vx = vix
    vy = viy
    vz = viz
    vdotb = (vx * bx + vy * by + vz * bz) / absB**2
    vparax = vdotb * vx
    vparay = vdotb * vy
    vparaz = vdotb * vz
    vperpx = vx - vparax
    vperpy = vy - vparay
    vperpz = vz - vparaz

    einx = by*vz - bz*vy
    einy = bz*vx - bx*vz
    einz = bx*vy - by*vx

    smime = math.sqrt(pic_info.mime)
    dx = (x[1] - x[0]) * smime
    dz = (z[1] - z[0]) * smime
    dpex = np.gradient(pexx, dx, axis=1) + np.gradient(pexz, dz, axis=0)
    dpey = np.gradient(pexy, dx, axis=1) + np.gradient(peyz, dz, axis=0)
    dpez = np.gradient(pexz, dx, axis=1) + np.gradient(pezz, dz, axis=0)

    divne = np.gradient(ne*vex, dx, axis=1) + np.gradient(ne*vez, dz, axis=0)
    inertialex = np.gradient(ne*vex*vex, dx, axis=1) + np.gradient(ne*vez*vex, dz, axis=0)
    inertialey = np.gradient(ne*vex*vey, dx, axis=1) + np.gradient(ne*vez*vey, dz, axis=0)
    inertialez = np.gradient(ne*vex*vez, dx, axis=1) + np.gradient(ne*vez*vez, dz, axis=0)
    inertialex21 = np.gradient(vx*jx, dx, axis=1) + np.gradient(vz*jx, dz, axis=0)
    inertialey21 = np.gradient(vx*jy, dx, axis=1) + np.gradient(vz*jy, dz, axis=0)
    inertialez21 = np.gradient(vx*jz, dx, axis=1) + np.gradient(vz*jz, dz, axis=0)
    inertialex22 = np.gradient(jx*vx, dx, axis=1) + np.gradient(jz*vx, dz, axis=0)
    inertialey22 = np.gradient(jx*vy, dx, axis=1) + np.gradient(jz*vy, dz, axis=0)
    inertialez22 = np.gradient(jx*vz, dx, axis=1) + np.gradient(jz*vz, dz, axis=0)
    inertialex23 = np.gradient(jx*jx/ne, dx, axis=1) + np.gradient(jz*jx/ne, dz, axis=0)
    inertialey23 = np.gradient(jx*jy/ne, dx, axis=1) + np.gradient(jz*jy/ne, dz, axis=0)
    inertialez23 = np.gradient(jx*jz/ne, dx, axis=1) + np.gradient(jz*jz/ne, dz, axis=0)

    inertialex2 = inertialex21 + inertialex22 - inertialex23
    inertialey2 = inertialey21 + inertialey22 - inertialey23
    inertialez2 = inertialez21 + inertialez22 - inertialez23

    inertialex2 *= -1
    inertialey2 *= -1
    inertialez2 *= -1

    jdote_e = -ne * (ex * vex + ey * vey + ez * vez)
    jdote_i = ni * (ex * vix + ey * viy + ez * viz)
    jdot_dpe = dpex * vex + dpey * vey + dpez * vez
    jdot_ine = inertialex * vex + inertialey * vey + inertialez * vez
    jdot_ine2 = inertialex2 * vex + inertialey2 * vey + inertialez2 * vez
    jdot_ine21 = inertialex21 * vex + inertialey21 * vey + inertialez21 * vez
    jdot_ine22 = inertialex22 * vex + inertialey22 * vey + inertialez22 * vez
    jdot_ine23 = inertialex23 * vex + inertialey23 * vey + inertialez23 * vez
    dpe_dotv = dpex * vx + dpey * vy + dpez * vz
    dpe_dotj = -(dpex * jx + dpey * jy + dpez * jz) / ne
    jdot_ein = jx * einx + jy * einy + jz * einz
    vdot_inere = inertialex * vx + inertialey * vy + inertialez * vz
    jdot_inere = -(inertialex * jx + inertialey * jy + inertialez * jz) / ne
    vdot_inere2 = inertialex2 * vx + inertialey2 * vy + inertialez2 * vz
    jdot_inere2 = -(inertialex2 * jx + inertialey2 * jy + inertialez2 * jz) / ne
    dpe_dotvpara = dpex * vparax + dpey * vparay + dpez * vparaz
    dpe_dotvperp = dpex * vperpx + dpey * vperpy + dpez * vperpz

    print "jdote electron: ", np.sum(jdote_e)
    print "jdote ion ", np.sum(jdote_i)
    print "jdot_dpe: ", np.sum(jdot_dpe)
    print "jdot_ine: ", np.sum(jdot_ine)
    print "jdot_ine2: ", np.sum(jdot_ine2)
    print "jdot_ine21: ", np.sum(jdot_ine21)
    print "jdot_ine22: ", np.sum(jdot_ine22)
    print "jdot_ine23: ", np.sum(jdot_ine23)
    print "dpe_dotv:", np.sum(dpe_dotv)
    print "dpe_dotvpara:", np.sum(dpe_dotvpara)
    print "dpe_dotvperp:", np.sum(dpe_dotvperp)
    print "dpe_dotj:", np.sum(dpe_dotj)
    print "jdot_ein:", np.sum(jdot_ein)
    print "vdot_inere:", np.sum(vdot_inere)
    print "jdot_inere:", np.sum(jdot_inere)
    print "vdot_inere2:", np.sum(vdot_inere2)
    print "jdot_inere2:", np.sum(jdot_inere2)

    ng = 3
    kernel = np.ones((ng, ng)) / float(ng * ng)
    jdote_e = signal.convolve2d(jdote_e, kernel, 'same')
    jdote_i = signal.convolve2d(jdote_i, kernel, 'same')
    jdot_dpe = signal.convolve2d(jdot_dpe, kernel, 'same')
    jdot_ine = signal.convolve2d(jdot_ine, kernel, 'same')
    dpe_dotv = signal.convolve2d(dpe_dotv, kernel, 'same')
    dpe_dotj = signal.convolve2d(dpe_dotj, kernel, 'same')
    jdot_ein = signal.convolve2d(jdot_ein, kernel, 'same')
    vdot_inere = signal.convolve2d(vdot_inere, kernel, 'same')
    jdot_inere = signal.convolve2d(jdot_inere, kernel, 'same')

    wpe_wce = pic_info.dtwce / pic_info.dtwpe
    va = wpe_wce / math.sqrt(pic_info.mime)

    fdata1e = dpe_dotv
    fdata2e = dpe_dotj
    fdata3e = jdot_dpe
    fdata4e = jdot_ine
    fdata5e = jdote_e

    fdata1i = -dpe_dotv
    fdata2i = jdot_ein
    fdata3i = -vdot_inere
    fdata4i = jdot_inere
    fdata5i = jdote_i

    # horizontal cut
    # zcut = 0
    # idz = (np.abs(z-zcut)).argmin()
    # fdata1_cut = fdata1[idz]
    # fdata2_cut = fdata2[idz]

    fdata1e_csum = np.cumsum(np.sum(fdata1e, axis=0))
    fdata2e_csum = np.cumsum(np.sum(fdata2e, axis=0))
    fdata3e_csum = np.cumsum(np.sum(fdata3e, axis=0))
    fdata4e_csum = np.cumsum(np.sum(fdata4e, axis=0))
    fdata5e_csum = np.cumsum(np.sum(fdata5e, axis=0))

    fdata1i_csum = np.cumsum(np.sum(fdata1i, axis=0))
    fdata2i_csum = np.cumsum(np.sum(fdata2i, axis=0))
    fdata3i_csum = np.cumsum(np.sum(fdata3i, axis=0))
    fdata4i_csum = np.cumsum(np.sum(fdata4i, axis=0))
    fdata5i_csum = np.cumsum(np.sum(fdata5i, axis=0))

    fdata1e_line = fdata1e_csum
    fdata2e_line = fdata2e_csum
    fdata3e_line = fdata3e_csum
    fdata4e_line = fdata4e_csum
    fdata5e_line = fdata5e_csum

    fdata1i_line = fdata1i_csum
    fdata2i_line = fdata2i_csum
    fdata3i_line = fdata3i_csum
    fdata4i_line = fdata4i_csum
    fdata5i_line = fdata5i_csum

    nx, = x.shape
    nz, = z.shape
    xs0, ys0 = 0.05, 0.85
    w1, h1 = 0.40, 0.14
    vgap = 0.02
    hgap = 0.10

    fig = plt.figure(figsize=[20, 20])
    xs = xs0
    ys = ys0
    ax1 = fig.add_axes([xs, ys, w1, h1])
    vmin, vmax = -0.01, 0.01
    crange = np.arange(vmin, vmax+0.005, 0.005)
    kwargs_plot = {"xstep":1, "zstep":1, "vmin":vmin, "vmax":vmax}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, fdata1e, ax1, fig, **kwargs_plot)
    # p1.set_cmap(cmaps.inferno)
    p1.set_cmap(plt.cm.get_cmap('seismic'))
    Ay_min = np.min(Ay)
    Ay_max = np.max(Ay)
    levels = np.linspace(Ay_min, Ay_max, 10)
    ax1.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep],
            colors='k', linewidths=0.5)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax1.tick_params(axis='x', labelbottom='off')
    ax1.tick_params(labelsize=16)
    cbar1.set_ticks(crange)
    cbar1.ax.tick_params(labelsize=16)
    text1 = r'$(\nabla\cdot\mathcal{P}_e)\cdot\boldsymbol{v}$'
    ax1.text(0.02, 0.8, text1, color=colors[0], fontsize=20,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                pad=10.0), horizontalalignment='left',
            verticalalignment='center', transform = ax1.transAxes)

    ys = ys0 -h1 - vgap
    ax2 = fig.add_axes([xs, ys, w1, h1])
    p2, cbar2 = plot_2d_contour(x, z, fdata2e, ax2, fig, **kwargs_plot)
    p2.set_cmap(plt.cm.seismic)
    ax2.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep],
            colors='black', linewidths=0.5)
    ax2.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax2.tick_params(axis='x', labelbottom='off')
    ax2.tick_params(labelsize=16)
    cbar2.set_ticks(crange)
    cbar2.ax.tick_params(labelsize=16)
    text2 = r'$-(\nabla\cdot\mathcal{P}_e)\cdot\boldsymbol{j}/ne$'
    ax2.text(0.02, 0.8, text2, color=colors[1], fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax2.transAxes)

    ys -= h1 + vgap
    ax3 = fig.add_axes([xs, ys, w1, h1])
    p3, cbar3 = plot_2d_contour(x, z, fdata3e, ax3, fig, **kwargs_plot)
    p3.set_cmap(plt.cm.seismic)
    ax3.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep],
            colors='black', linewidths=0.5)
    ax3.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax3.tick_params(axis='x', labelbottom='off')
    ax3.tick_params(labelsize=16)
    cbar3.set_ticks(crange)
    cbar3.ax.tick_params(labelsize=16)
    text3 = r'$(\nabla\cdot\mathcal{P}_e)\cdot\boldsymbol{v}_e$'
    ax3.text(0.02, 0.8, text3, color=colors[2], fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax3.transAxes)

    ys -= h1 + vgap
    ax4 = fig.add_axes([xs, ys, w1, h1])
    p4, cbar4 = plot_2d_contour(x, z, fdata4e, ax4, fig, **kwargs_plot)
    p4.set_cmap(plt.cm.seismic)
    ax4.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep],
            colors='black', linewidths=0.5)
    ax4.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax4.tick_params(axis='x', labelbottom='off')
    ax4.tick_params(labelsize=16)
    cbar4.set_ticks(crange)
    cbar4.ax.tick_params(labelsize=16)
    text4 = r'$(\nabla\cdot(n\boldsymbol{v}_e\boldsymbol{v}_e))\cdot\boldsymbol{v}_em_e$'
    ax4.text(0.02, 0.8, text4, color=colors[3], fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax4.transAxes)

    ys -= h1 + vgap
    ax5 = fig.add_axes([xs, ys, w1, h1])
    p5, cbar5 = plot_2d_contour(x, z, fdata5e, ax5, fig, **kwargs_plot)
    p5.set_cmap(plt.cm.seismic)
    ax5.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep],
            colors='black', linewidths=0.5)
    ax5.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax5.tick_params(axis='x', labelbottom='off')
    ax5.tick_params(labelsize=16)
    cbar5.set_ticks(crange)
    cbar5.ax.tick_params(labelsize=16)
    text5 = r'$\boldsymbol{j}_e\cdot\boldsymbol{E}$'
    ax5.text(0.02, 0.8, text5, color=colors[4], fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax5.transAxes)

    ys -= h1 + vgap
    width, height = fig.get_size_inches()
    w2 = w1 * 0.98 - 0.05 / width
    ax6 = fig.add_axes([xs, ys, w2, h1])
    ax6.set_color_cycle(colors)
    ax6.plot(x, fdata1e_line, linewidth=2)
    ax6.plot(x, fdata2e_line, linewidth=2)
    ax6.plot(x, fdata3e_line, linewidth=2)
    ax6.plot(x, fdata4e_line, linewidth=2)
    ax6.plot(x, fdata5e_line, linewidth=2)
    ax6.set_xlim(ax2.get_xlim())
    # ax3.set_ylim([vmin, vmax])
    ax6.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    ax6.tick_params(labelsize=16)

    xs += w1 + hgap
    ys = ys0
    ax1 = fig.add_axes([xs, ys, w1, h1])
    vmin, vmax = -0.01, 0.01
    crange = np.arange(vmin, vmax+0.005, 0.005)
    kwargs_plot = {"xstep":1, "zstep":1, "vmin":vmin, "vmax":vmax}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, fdata1i, ax1, fig, **kwargs_plot)
    # p1.set_cmap(cmaps.inferno)
    p1.set_cmap(plt.cm.get_cmap('seismic'))
    Ay_min = np.min(Ay)
    Ay_max = np.max(Ay)
    levels = np.linspace(Ay_min, Ay_max, 10)
    ax1.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep],
            colors='k', linewidths=0.5)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax1.tick_params(axis='x', labelbottom='off')
    ax1.tick_params(labelsize=16)
    cbar1.set_ticks(crange)
    cbar1.ax.tick_params(labelsize=16)
    text1 = r'$-(\nabla\cdot\mathcal{P}_e)\cdot\boldsymbol{v}$'
    ax1.text(0.02, 0.8, text1, color=colors[0], fontsize=20,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                pad=10.0), horizontalalignment='left',
            verticalalignment='center', transform = ax1.transAxes)

    ys = ys0 -h1 - vgap
    ax2 = fig.add_axes([xs, ys, w1, h1])
    p2, cbar2 = plot_2d_contour(x, z, fdata2i, ax2, fig, **kwargs_plot)
    p2.set_cmap(plt.cm.seismic)
    ax2.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep],
            colors='black', linewidths=0.5)
    ax2.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax2.tick_params(axis='x', labelbottom='off')
    ax2.tick_params(labelsize=16)
    cbar2.set_ticks(crange)
    cbar2.ax.tick_params(labelsize=16)
    text2 = r'$(\boldsymbol{j}\times\boldsymbol{B})\cdot\boldsymbol{v}$'
    ax2.text(0.02, 0.8, text2, color=colors[1], fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax2.transAxes)

    ys -= h1 + vgap
    ax3 = fig.add_axes([xs, ys, w1, h1])
    p3, cbar3 = plot_2d_contour(x, z, fdata3i, ax3, fig, **kwargs_plot)
    p3.set_cmap(plt.cm.seismic)
    ax3.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep],
            colors='black', linewidths=0.5)
    ax3.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax3.tick_params(axis='x', labelbottom='off')
    ax3.tick_params(labelsize=16)
    cbar3.set_ticks(crange)
    cbar3.ax.tick_params(labelsize=16)
    text3 = r'$-(\nabla\cdot(n\boldsymbol{v}_e\boldsymbol{v}_e))\cdot\boldsymbol{v}m_e$'
    ax3.text(0.02, 0.8, text3, color=colors[2], fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax3.transAxes)

    ys -= h1 + vgap
    ax4 = fig.add_axes([xs, ys, w1, h1])
    p4, cbar4 = plot_2d_contour(x, z, fdata4i, ax4, fig, **kwargs_plot)
    p4.set_cmap(plt.cm.seismic)
    ax4.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep],
            colors='black', linewidths=0.5)
    ax4.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax4.tick_params(axis='x', labelbottom='off')
    ax4.tick_params(labelsize=16)
    cbar4.set_ticks(crange)
    cbar4.ax.tick_params(labelsize=16)
    text4 = r'$-(\nabla\cdot(n\boldsymbol{v}_e\boldsymbol{v}_e))\cdot\boldsymbol{j}m_e/ne$'
    ax4.text(0.02, 0.8, text4, color=colors[3], fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax4.transAxes)

    ys -= h1 + vgap
    ax5 = fig.add_axes([xs, ys, w1, h1])
    p5, cbar5 = plot_2d_contour(x, z, fdata5i, ax5, fig, **kwargs_plot)
    p5.set_cmap(plt.cm.seismic)
    ax5.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep],
            colors='black', linewidths=0.5)
    ax5.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax5.tick_params(axis='x', labelbottom='off')
    ax5.tick_params(labelsize=16)
    cbar5.set_ticks(crange)
    cbar5.ax.tick_params(labelsize=16)
    text5 = r'$\boldsymbol{j}_i\cdot\boldsymbol{E}$'
    ax5.text(0.02, 0.8, text5, color=colors[4], fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax5.transAxes)

    ys -= h1 + vgap
    width, height = fig.get_size_inches()
    w2 = w1 * 0.98 - 0.05 / width
    ax6 = fig.add_axes([xs, ys, w2, h1])
    ax6.set_color_cycle(colors)
    ax6.plot(x, fdata1i_line, linewidth=2)
    ax6.plot(x, fdata2i_line, linewidth=2)
    ax6.plot(x, fdata3i_line, linewidth=2)
    ax6.plot(x, fdata4i_line, linewidth=2)
    ax6.plot(x, fdata5i_line, linewidth=2)
    ax6.set_xlim(ax2.get_xlim())
    # ax3.set_ylim([vmin, vmax])
    ax6.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    ax6.tick_params(labelsize=16)

    plt.show()


def j_curlB(pic_info, ct, root_dir='../../'):
    """
    Compare j and curB
    """
    xl, xr = 50, 150
    kwargs = {"current_time": ct, "xl": xl, "xr": xr, "zb": -20, "zt": 20}
    fname = root_dir + 'data/bx.gda'
    x, z, bx = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/by.gda'
    x, z, by = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/bz.gda'
    x, z, bz = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/absB.gda'
    x, z, absB = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/Ay.gda'
    x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)

    fname = root_dir + 'data/jx.gda'
    x, z, jx = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/jy.gda'
    x, z, jy = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/jz.gda'
    x, z, jz = read_2d_fields(pic_info, fname, **kwargs)

    smime = math.sqrt(pic_info.mime)
    dx = (x[1] - x[0])*smime
    dz = (z[1] - z[0])*smime

    cbx = -np.gradient(by, dz, axis=0)
    cby = np.gradient(bx, dz, axis=0) - np.gradient(bz, dx, axis=1)
    cbz = np.gradient(by, dx, axis=1)

    ng = 3
    kernel = np.ones((ng, ng)) / float(ng * ng)
    cbx = signal.convolve2d(cbx, kernel, 'same')
    cby = signal.convolve2d(cby, kernel, 'same')
    cbz = signal.convolve2d(cbz, kernel, 'same')

    fdata1 = cbz
    fdata2 = jz

    # horizontal cut
    zcut = 0
    idz = (np.abs(z-zcut)).argmin()
    fdata1_cut = fdata1[idz]
    fdata2_cut = fdata2[idz]
    fdata1_line = fdata1_cut
    fdata2_line = fdata2_cut

    nx, = x.shape
    nz, = z.shape
    xs, ys = 0.1, 0.7
    w1, h1 = 0.85, 0.28
    gap = 0.03

    fig = plt.figure(figsize=[10, 10])
    ax1 = fig.add_axes([xs, ys, w1, h1])
    kwargs_plot = {"xstep":1, "zstep":1, "vmin":-0.5, "vmax":0.5}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, fdata1, ax1, fig, **kwargs_plot)
    # p1.set_cmap(cmaps.inferno)
    p1.set_cmap(plt.cm.get_cmap('seismic'))
    Ay_min = np.min(Ay)
    Ay_max = np.max(Ay)
    levels = np.linspace(Ay_min, Ay_max, 10)
    ax1.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep],
            colors='k', linewidths=0.5)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax1.tick_params(axis='x', labelbottom='off')
    ax1.tick_params(labelsize=16)
    cbar1.set_ticks(np.arange(-0.5, 0.6, 0.5))
    cbar1.ax.tick_params(labelsize=16)
    ax1.text(0.02, 0.8, r'$E_y$', color='k', fontsize=20,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                pad=10.0), horizontalalignment='left',
            verticalalignment='center', transform = ax1.transAxes)

    ys -= h1 + gap
    ax2 = fig.add_axes([xs, ys, w1, h1])
    kwargs_plot = {"xstep":1, "zstep":1, "vmin":-0.5, "vmax":0.5}
    p2, cbar2 = plot_2d_contour(x, z, fdata2, ax2, fig, **kwargs_plot)
    p2.set_cmap(plt.cm.seismic)
    ax2.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep],
            colors='black', linewidths=0.5)
    ax2.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax2.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    ax2.tick_params(labelsize=16)
    cbar2.set_ticks(np.arange(-0.5, 0.6, 0.5))
    cbar2.ax.tick_params(labelsize=16)
    ax2.text(0.02, 0.8, r'$E_\parallel$', color='k', fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax2.transAxes)

    ys -= h1 + gap
    width, height = fig.get_size_inches()
    w2 = w1 * 0.98 - 0.05 / width
    ax3 = fig.add_axes([xs, ys, w2, h1])
    ax3.set_color_cycle(colors)
    ax3.plot(x, fdata1_line, linewidth=2)
    ax3.plot(x, fdata2_line, linewidth=2)
    ax3.set_xlim(ax2.get_xlim())
    ax3.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    ax3.tick_params(labelsize=16)

    plt.show()
    # plt.close()


def plot_ohm_efield(pic_info, ct, root_dir='../../'):
    """Plot electric field in generalized Ohm's law
    """
    kwargs = {"current_time": ct, "xl": 50, "xr": 150, "zb": -20, "zt": 20}
    fname = root_dir + 'data/bx.gda'
    x, z, bx = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/by.gda'
    x, z, by = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/bz.gda'
    x, z, bz = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/absB.gda'
    x, z, absB = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/ex.gda'
    x, z, ex = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/ey.gda'
    x, z, ey = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/ez.gda'
    x, z, ez = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/Ay.gda'
    x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)

    fname = root_dir + 'data/jx.gda'
    x, z, jx = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/jy.gda'
    x, z, jy = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/jz.gda'
    x, z, jz = read_2d_fields(pic_info, fname, **kwargs)

    fname = root_dir + 'data1/vx.gda'
    x, z, vx = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data1/vy.gda'
    x, z, vy = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data1/vz.gda'
    x, z, vz = read_2d_fields(pic_info, fname, **kwargs)

    fname = root_dir + 'data/pe-xx.gda'
    x, z, pexx = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/pe-xy.gda'
    x, z, pexy = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/pe-xz.gda'
    x, z, pexz = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/pe-yy.gda'
    x, z, peyy = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/pe-yz.gda'
    x, z, peyz = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/pe-zz.gda'
    x, z, pezz = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/ne.gda'
    x, z, ne = read_2d_fields(pic_info, fname, **kwargs)

    einx = vz * by - vy * bz
    einy = vx * bz - vz * bx
    einz = vy * bx - vx * by

    hallx = jy * bz - jz * by
    hally = jz * bx - jx * bz
    hallz = jx * by - jy * bx

    absE = np.sqrt(ex * ex + ey * ey + ez * ez)
    epara = (ex * bx + ey * by + ez * bz) / absB
    eperp = np.sqrt(absE * absE - epara * epara)
    ng = 5
    kernel = np.ones((ng, ng)) / float(ng * ng)
    epara = signal.convolve2d(epara, kernel, 'same')
    eperp = signal.convolve2d(eperp, kernel, 'same')
    ex = signal.convolve2d(ex, kernel, 'same')
    ey = signal.convolve2d(ey, kernel, 'same')
    ez = signal.convolve2d(ez, kernel, 'same')

    dx = x[1] - x[0]
    dz = z[1] - z[0]
    dpex = np.gradient(pexx, dx, axis=1) + np.gradient(pexz, dz, axis=0)
    dpey = np.gradient(pexy, dx, axis=1) + np.gradient(peyz, dz, axis=0)
    dpez = np.gradient(pexz, dx, axis=1) + np.gradient(pezz, dz, axis=0)
    dpex /= ne
    dpey /= ne
    dpez /= ne
    hallx /= ne
    hally /= ne
    hallz /= ne
    dpe  = np.sqrt(dpex**2 + dpey**2 + dpez**2)
    dpex = signal.convolve2d(dpex, kernel, 'same')
    dpey = signal.convolve2d(dpey, kernel, 'same')
    dpez = signal.convolve2d(dpez, kernel, 'same')

    wpe_wce = pic_info.dtwce / pic_info.dtwpe
    va = wpe_wce / math.sqrt(pic_info.mime)  # Alfven speed of inflow region
    b0 = pic_info.b0
    e0 = va * b0
    epara /= 0.5 * e0
    eperp /= 0.5 * e0
    ex /= 0.5 * e0
    ey /= 0.5 * e0
    ez /= 0.5 * e0
    einx /= 0.5 * e0
    einy /= 0.5 * e0
    einz /= 0.5 * e0
    dpex /= 0.5 * e0
    dpey /= 0.5 * e0
    dpez /= 0.5 * e0
    hallx /= 0.5 * e0
    hally /= 0.5 * e0
    hallz /= 0.5 * e0


    text_ex = r'$E_x$'
    text_ey = r'$E_y$'
    text_ez = r'$E_z$'
    text_vxbx = r'$-(\boldsymbol{v}\times\boldsymbol{B})_x$'
    text_vxby = r'$-(\boldsymbol{v}\times\boldsymbol{B})_y$'
    text_vxbz = r'$-(\boldsymbol{v}\times\boldsymbol{B})_z$'
    text_dpex = r'$-(\nabla\cdot\mathcal{P}_e)_x/n_ee$'
    text_dpey = r'$-(\nabla\cdot\mathcal{P}_e)_y/n_ee$'
    text_dpez = r'$-(\nabla\cdot\mathcal{P}_e)_z/n_ee$'
    text_hallx = r'$(\boldsymbol{j}\times\boldsymbol{B})_x/n_ee$'
    text_hally = r'$(\boldsymbol{j}\times\boldsymbol{B})_y/n_ee$'
    text_hallz = r'$(\boldsymbol{j}\times\boldsymbol{B})_z/n_ee$'

    fdata11 = ex
    fdata12 = einx
    fdata13 = dpex
    fdata14 = hallx
    text11 = text_ex
    text12 = text_vxbx
    text13 = text_dpex
    text14 = text_hallx

    fdata21 = ey
    fdata22 = einy
    fdata23 = dpey
    fdata24 = hally
    text21 = text_ey
    text22 = text_vxby
    text23 = text_dpey
    text24 = text_hally

    fdata31 = ez
    fdata32 = einz
    fdata33 = dpez
    fdata34 = hallz
    text31 = text_ez
    text32 = text_vxbz
    text33 = text_dpez
    text34 = text_hallz

    nx, = x.shape
    nz, = z.shape
    xs0, ys0 = 0.05, 0.78
    w1, h1 = 0.28, 0.2
    vgap, hgap = 0.03, 0.03

    xs = xs0
    ys = ys0

    fig = plt.figure(figsize=[24, 12])
    ax1 = fig.add_axes([xs, ys, w1, h1])
    kwargs_plot = {"xstep":1, "zstep":1, "vmin":-0.5, "vmax":0.5}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, fdata11, ax1, fig, **kwargs_plot)
    # p1.set_cmap(cmaps.inferno)
    p1.set_cmap(plt.cm.get_cmap('seismic'))
    Ay_min = np.min(Ay)
    Ay_max = np.max(Ay)
    levels = np.linspace(Ay_min, Ay_max, 10)
    ax1.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep],
            colors='k', linewidths=0.5)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax1.tick_params(axis='x', labelbottom='off')
    ax1.tick_params(labelsize=16)
    cbar1.set_ticks(np.arange(-0.5, 0.6, 0.5))
    cbar1.ax.tick_params(labelsize=16)
    ax1.text(0.02, 0.8, text11, color='k', fontsize=20,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                pad=10.0), horizontalalignment='left',
            verticalalignment='center', transform = ax1.transAxes)

    ys -= h1 + vgap
    ax2 = fig.add_axes([xs, ys, w1, h1])
    kwargs_plot = {"xstep":1, "zstep":1, "vmin":-0.5, "vmax":0.5}
    p2, cbar2 = plot_2d_contour(x, z, fdata12, ax2, fig, **kwargs_plot)
    p2.set_cmap(plt.cm.seismic)
    ax2.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep],
            colors='black', linewidths=0.5)
    ax2.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax2.tick_params(axis='x', labelbottom='off')
    ax2.tick_params(labelsize=16)
    cbar2.set_ticks(np.arange(-0.5, 0.6, 0.5))
    cbar2.ax.tick_params(labelsize=16)
    ax2.text(0.02, 0.8, text12, color='k', fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax2.transAxes)

    ys -= h1 + vgap
    ax3 = fig.add_axes([xs, ys, w1, h1])
    kwargs_plot = {"xstep":1, "zstep":1, "vmin":-0.5, "vmax":0.5}
    p3, cbar3 = plot_2d_contour(x, z, fdata13, ax3, fig, **kwargs_plot)
    p3.set_cmap(plt.cm.seismic)
    ax3.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep],
            colors='black', linewidths=0.5)
    ax3.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax3.tick_params(axis='x', labelbottom='off')
    ax3.tick_params(labelsize=16)
    cbar3.set_ticks(np.arange(-0.5, 0.6, 0.5))
    cbar3.ax.tick_params(labelsize=16)
    ax3.text(0.02, 0.8, text13, color='k', fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax3.transAxes)

    ys -= h1 + vgap
    ax4 = fig.add_axes([xs, ys, w1, h1])
    kwargs_plot = {"xstep":1, "zstep":1, "vmin":-0.5, "vmax":0.5}
    p4, cbar4 = plot_2d_contour(x, z, fdata14, ax4, fig, **kwargs_plot)
    p4.set_cmap(plt.cm.seismic)
    ax4.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep],
            colors='black', linewidths=0.5)
    ax4.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax4.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    ax4.tick_params(labelsize=16)
    cbar4.set_ticks(np.arange(-0.5, 0.6, 0.5))
    cbar4.ax.tick_params(labelsize=16)
    ax4.text(0.02, 0.8, text14, color='k', fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax4.transAxes)

    xs = xs0 + hgap + w1
    ys = ys0

    ax1 = fig.add_axes([xs, ys, w1, h1])
    kwargs_plot = {"xstep":1, "zstep":1, "vmin":-0.5, "vmax":0.5}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, fdata21, ax1, fig, **kwargs_plot)
    # p1.set_cmap(cmaps.inferno)
    p1.set_cmap(plt.cm.get_cmap('seismic'))
    Ay_min = np.min(Ay)
    Ay_max = np.max(Ay)
    levels = np.linspace(Ay_min, Ay_max, 10)
    ax1.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep],
            colors='k', linewidths=0.5)
    ax1.set_ylabel('', fontdict=font, fontsize=20)
    ax1.tick_params(axis='x', labelbottom='off')
    ax1.tick_params(axis='y', labelleft='off')
    ax1.tick_params(labelsize=16)
    cbar1.set_ticks(np.arange(-0.5, 0.6, 0.5))
    cbar1.ax.tick_params(labelsize=16)
    ax1.text(0.02, 0.8, text21, color='k', fontsize=20,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                pad=10.0), horizontalalignment='left',
            verticalalignment='center', transform = ax1.transAxes)

    ys -= h1 + vgap
    ax2 = fig.add_axes([xs, ys, w1, h1])
    kwargs_plot = {"xstep":1, "zstep":1, "vmin":-0.5, "vmax":0.5}
    p2, cbar2 = plot_2d_contour(x, z, fdata22, ax2, fig, **kwargs_plot)
    p2.set_cmap(plt.cm.seismic)
    ax2.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep],
            colors='black', linewidths=0.5)
    ax2.set_ylabel('', fontdict=font, fontsize=20)
    ax2.tick_params(axis='x', labelbottom='off')
    ax2.tick_params(axis='y', labelleft='off')
    ax2.tick_params(labelsize=16)
    cbar2.set_ticks(np.arange(-0.5, 0.6, 0.5))
    cbar2.ax.tick_params(labelsize=16)
    ax2.text(0.02, 0.8, text22, color='k', fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax2.transAxes)

    ys -= h1 + vgap
    ax3 = fig.add_axes([xs, ys, w1, h1])
    kwargs_plot = {"xstep":1, "zstep":1, "vmin":-0.5, "vmax":0.5}
    p3, cbar3 = plot_2d_contour(x, z, fdata23, ax3, fig, **kwargs_plot)
    p3.set_cmap(plt.cm.seismic)
    ax3.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep],
            colors='black', linewidths=0.5)
    ax3.set_ylabel('', fontdict=font, fontsize=20)
    ax3.tick_params(axis='x', labelbottom='off')
    ax3.tick_params(axis='y', labelleft='off')
    ax3.tick_params(labelsize=16)
    cbar3.set_ticks(np.arange(-0.5, 0.6, 0.5))
    cbar3.ax.tick_params(labelsize=16)
    ax3.text(0.02, 0.8, text23, color='k', fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax3.transAxes)

    ys -= h1 + vgap
    ax4 = fig.add_axes([xs, ys, w1, h1])
    kwargs_plot = {"xstep":1, "zstep":1, "vmin":-0.5, "vmax":0.5}
    p4, cbar4 = plot_2d_contour(x, z, fdata24, ax4, fig, **kwargs_plot)
    p4.set_cmap(plt.cm.seismic)
    ax4.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep],
            colors='black', linewidths=0.5)
    ax4.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    ax4.set_ylabel('', fontdict=font, fontsize=20)
    ax4.tick_params(axis='y', labelleft='off')
    ax4.tick_params(labelsize=16)
    cbar4.set_ticks(np.arange(-0.5, 0.6, 0.5))
    cbar4.ax.tick_params(labelsize=16)
    ax4.text(0.02, 0.8, text24, color='k', fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax4.transAxes)

    xs = xs0 + 2 * hgap + 2 * w1
    ys = ys0

    ax1 = fig.add_axes([xs, ys, w1, h1])
    kwargs_plot = {"xstep":1, "zstep":1, "vmin":-0.5, "vmax":0.5}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, fdata31, ax1, fig, **kwargs_plot)
    # p1.set_cmap(cmaps.inferno)
    p1.set_cmap(plt.cm.get_cmap('seismic'))
    Ay_min = np.min(Ay)
    Ay_max = np.max(Ay)
    levels = np.linspace(Ay_min, Ay_max, 10)
    ax1.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep],
            colors='k', linewidths=0.5)
    ax1.set_ylabel('', fontdict=font, fontsize=20)
    ax1.tick_params(axis='x', labelbottom='off')
    ax1.tick_params(axis='y', labelleft='off')
    ax1.tick_params(labelsize=16)
    cbar1.set_ticks(np.arange(-0.5, 0.6, 0.5))
    cbar1.ax.tick_params(labelsize=16)
    ax1.text(0.02, 0.8, text31, color='k', fontsize=20,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                pad=10.0), horizontalalignment='left',
            verticalalignment='center', transform = ax1.transAxes)

    ys -= h1 + vgap
    ax2 = fig.add_axes([xs, ys, w1, h1])
    kwargs_plot = {"xstep":1, "zstep":1, "vmin":-0.5, "vmax":0.5}
    p2, cbar2 = plot_2d_contour(x, z, fdata32, ax2, fig, **kwargs_plot)
    p2.set_cmap(plt.cm.seismic)
    ax2.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep],
            colors='black', linewidths=0.5)
    ax2.set_ylabel('', fontdict=font, fontsize=20)
    ax2.tick_params(axis='x', labelbottom='off')
    ax2.tick_params(axis='y', labelleft='off')
    ax2.tick_params(labelsize=16)
    cbar2.set_ticks(np.arange(-0.5, 0.6, 0.5))
    cbar2.ax.tick_params(labelsize=16)
    ax2.text(0.02, 0.8, text32, color='k', fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax2.transAxes)

    ys -= h1 + vgap
    ax3 = fig.add_axes([xs, ys, w1, h1])
    kwargs_plot = {"xstep":1, "zstep":1, "vmin":-0.5, "vmax":0.5}
    p3, cbar3 = plot_2d_contour(x, z, fdata33, ax3, fig, **kwargs_plot)
    p3.set_cmap(plt.cm.seismic)
    ax3.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep],
            colors='black', linewidths=0.5)
    ax3.set_ylabel('', fontdict=font, fontsize=20)
    ax3.tick_params(axis='x', labelbottom='off')
    ax3.tick_params(axis='y', labelleft='off')
    ax3.tick_params(labelsize=16)
    cbar3.set_ticks(np.arange(-0.5, 0.6, 0.5))
    cbar3.ax.tick_params(labelsize=16)
    ax3.text(0.02, 0.8, text33, color='k', fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax3.transAxes)

    ys -= h1 + vgap
    ax4 = fig.add_axes([xs, ys, w1, h1])
    kwargs_plot = {"xstep":1, "zstep":1, "vmin":-0.5, "vmax":0.5}
    p4, cbar4 = plot_2d_contour(x, z, fdata34, ax4, fig, **kwargs_plot)
    p4.set_cmap(plt.cm.seismic)
    ax4.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep],
            colors='black', linewidths=0.5)
    ax4.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    ax4.set_ylabel('', fontdict=font, fontsize=20)
    ax4.tick_params(axis='y', labelleft='off')
    ax4.tick_params(labelsize=16)
    cbar4.set_ticks(np.arange(-0.5, 0.6, 0.5))
    cbar4.ax.tick_params(labelsize=16)
    ax4.text(0.02, 0.8, text34, color='k', fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax4.transAxes)


    plt.show()
    # plt.close()


if __name__ == "__main__":
    cmdargs = sys.argv
    if (len(cmdargs) > 2):
        base_dir = cmdargs[1]
        run_name = cmdargs[2]
    else:
        base_dir = '/net/scratch3/xiaocanli/reconnection/mime25-sigma1-beta002-guide00-200-100/'
        run_name = 'mime25_beta002_guide00'
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    # plot_epara_eperp(pic_info, 25, base_dir)
    plot_ohm_efield(pic_info, 25, base_dir)
    # plot_epara(pic_info, 25, base_dir)
    # plot_jdote(pic_info, 25, base_dir)
    # j_curlB(pic_info, 25, base_dir)
