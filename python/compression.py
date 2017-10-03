"""
Analysis procedures for compression related terms.
"""
import argparse
import collections
import math
import multiprocessing
import os
import os.path
import struct
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import seaborn as sns
import simplejson as json
from joblib import Parallel, delayed
from matplotlib import rc
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
from scipy.fftpack import fft2, fftshift, ifft2
from scipy.interpolate import (RectBivariateSpline, RegularGridInterpolator,
                               interp1d, spline)
from scipy.ndimage.filters import generic_filter as gf
from scipy.ndimage.filters import median_filter, gaussian_filter

import palettable
import pic_information
from contour_plots import find_closest, plot_2d_contour, read_2d_fields
from energy_conversion import read_data_from_json, read_jdote_data
from particle_compression import read_fields, read_hydro_velocity_density
from runs_name_path import ApJ_long_paper_runs
from serialize_json import data_to_json, json_to_data
from shell_functions import mkdir_p

style.use(['seaborn-white', 'seaborn-paper'])
# rc('font', **{'family': 'serif', 'serif': ["Times", "Palatino", "serif"]})
# rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc("font", family="Times New Roman")
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

colors_Set1_9 = palettable.colorbrewer.qualitative.Set1_9.mpl_colors
colors_Dark2_8 = palettable.colorbrewer.qualitative.Dark2_8.mpl_colors
colors_Paired_12 = palettable.colorbrewer.qualitative.Paired_12.mpl_colors
colors_Tableau_10 = palettable.tableau.Tableau_10.mpl_colors
colors_GreenOrange_6 = palettable.tableau.GreenOrange_6.mpl_colors

font = {
    'family': 'serif',
    # 'color': 'darkred',
    'color': 'black',
    'weight': 'normal',
    'size': 24,
}

def plot_compression(pic_info, species, current_time):
    """Plot compression related terms.

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
    fname = "../../data1/vdot_div_ptensor00_" + species + ".gda"
    x, z, vdot_div_ptensor = read_2d_fields(pic_info, fname, **kwargs)
    fname = "../../data1/pdiv_u00_" + species + ".gda"
    x, z, pdiv_u = read_2d_fields(pic_info, fname, **kwargs)
    fname = "../../data1/div_u00_" + species + ".gda"
    x, z, div_u = read_2d_fields(pic_info, fname, **kwargs)
    fname = "../../data1/pshear00_" + species + ".gda"
    x, z, pshear = read_2d_fields(pic_info, fname, **kwargs)
    fname = "../../data1/div_vdot_ptensor00_" + species + ".gda"
    x, z, div_vdot_ptensor = read_2d_fields(pic_info, fname, **kwargs)
    x, z, Ay = read_2d_fields(pic_info, "../../data/Ay.gda", **kwargs)
    fname = '../../data/u' + species + 'x.gda'
    x, z, ux = read_2d_fields(pic_info, fname, **kwargs)
    fname = '../../data/u' + species + 'y.gda'
    x, z, uy = read_2d_fields(pic_info, fname, **kwargs)
    fname = '../../data/u' + species + 'z.gda'
    x, z, uz = read_2d_fields(pic_info, fname, **kwargs)
    x, z, ex = read_2d_fields(pic_info, '../../data/ex.gda', **kwargs)
    x, z, ey = read_2d_fields(pic_info, '../../data/ey.gda', **kwargs)
    x, z, ez = read_2d_fields(pic_info, '../../data/ez.gda', **kwargs)
    fname = '../../data/n' + species + '.gda'
    x, z, nrho = read_2d_fields(pic_info, fname, **kwargs)
    if species == 'e':
        jdote = -(ux * ex + uy * ey + uz * ez) * nrho
    else:
        jdote = (ux * ex + uy * ey + uz * ez) * nrho

    pdiv_u_sum = np.sum(pdiv_u, axis=0)
    pdiv_u_cum = np.cumsum(pdiv_u_sum)
    pshear_sum = np.sum(pshear, axis=0)
    pshear_cum = np.cumsum(pshear_sum)
    pcomp1_sum = np.sum(div_vdot_ptensor, axis=0)
    pcomp1_cum = np.cumsum(pcomp1_sum)
    data4 = pdiv_u + pshear + div_vdot_ptensor
    pcomp2_sum = np.sum(data4, axis=0)
    pcomp2_cum = np.cumsum(pcomp2_sum)
    pcomp3_sum = np.sum(vdot_div_ptensor, axis=0)
    pcomp3_cum = np.cumsum(pcomp3_sum)
    jdote_sum = np.sum(jdote, axis=0)
    jdote_cum = np.cumsum(jdote_sum)

    nx, = x.shape
    nz, = z.shape
    zl = nz / 4
    zt = nz - zl
    nk = 5
    div_u_new = signal.medfilt2d(div_u[zl:zt, :], kernel_size=(nk, nk))
    pdiv_u_new = signal.medfilt2d(pdiv_u[zl:zt, :], kernel_size=(nk, nk))
    pshear_new = signal.medfilt2d(pshear[zl:zt, :], kernel_size=(nk, nk))
    vdot_div_ptensor_new = signal.medfilt2d(
        vdot_div_ptensor[zl:zt, :], kernel_size=(nk, nk))
    div_vdot_ptensor_new = signal.medfilt2d(
        div_vdot_ptensor[zl:zt, :], kernel_size=(nk, nk))
    jdote_new = signal.medfilt2d(jdote[zl:zt, :], kernel_size=(nk, nk))
    data4_new = pdiv_u_new + pshear_new + div_vdot_ptensor_new

    width = 0.75
    height = 0.11
    xs = 0.12
    ys = 0.98 - height
    gap = 0.025

    vmin = -0.005
    vmax = 0.005
    fig = plt.figure(figsize=[10, 14])
    ax1 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep": 1, "zstep": 1, "vmin": vmin, "vmax": vmax}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z[zl:zt], pdiv_u_new, ax1, fig,
                                **kwargs_plot)
    p1.set_cmap(plt.cm.seismic)
    ax1.contour(
        x[0:nx:xstep],
        z[zl:zt:zstep],
        Ay[zl:zt:zstep, 0:nx:xstep],
        colors='black',
        linewidths=0.5)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax1.tick_params(labelsize=20)
    ax1.tick_params(axis='x', labelbottom='off')
    cbar1.set_ticks(np.arange(-0.004, 0.005, 0.002))
    cbar1.ax.tick_params(labelsize=20)
    fname1 = r'$-p\nabla\cdot\mathbf{u}$'
    ax1.text(
        0.02,
        0.8,
        fname1,
        color='red',
        fontsize=24,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='left',
        verticalalignment='center',
        transform=ax1.transAxes)

    ys -= height + gap
    ax2 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep": 1, "zstep": 1, "vmin": vmin, "vmax": vmax}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p2, cbar2 = plot_2d_contour(x, z[zl:zt], pshear_new, ax2, fig,
                                **kwargs_plot)
    p2.set_cmap(plt.cm.seismic)
    ax2.contour(
        x[0:nx:xstep],
        z[zl:zt:zstep],
        Ay[zl:zt:zstep, 0:nx:xstep],
        colors='black',
        linewidths=0.5)
    ax2.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax2.tick_params(labelsize=20)
    ax2.tick_params(axis='x', labelbottom='off')
    cbar2.set_ticks(np.arange(-0.004, 0.005, 0.002))
    cbar2.ax.tick_params(labelsize=20)
    fname2 = r'$-(p_\parallel - p_\perp)b_ib_j\sigma_{ij}$'
    ax2.text(
        0.02,
        0.8,
        fname2,
        color='green',
        fontsize=24,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='left',
        verticalalignment='center',
        transform=ax2.transAxes)

    ys -= height + gap
    ax3 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep": 1, "zstep": 1, "vmin": vmin, "vmax": vmax}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p3, cbar3 = plot_2d_contour(x, z[zl:zt], div_vdot_ptensor_new, ax3, fig,
                                **kwargs_plot)
    p3.set_cmap(plt.cm.seismic)
    ax3.contour(
        x[0:nx:xstep],
        z[zl:zt:zstep],
        Ay[zl:zt:zstep, 0:nx:xstep],
        colors='black',
        linewidths=0.5)
    ax3.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax3.tick_params(labelsize=20)
    ax3.tick_params(axis='x', labelbottom='off')
    cbar3.set_ticks(np.arange(-0.004, 0.005, 0.002))
    cbar3.ax.tick_params(labelsize=20)
    fname3 = r'$\nabla\cdot(\mathbf{u}\cdot\mathcal{P})$'
    ax3.text(
        0.02,
        0.8,
        fname3,
        color='blue',
        fontsize=24,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='left',
        verticalalignment='center',
        transform=ax3.transAxes)

    ys -= height + gap
    ax4 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep": 1, "zstep": 1, "vmin": vmin, "vmax": vmax}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p4, cbar4 = plot_2d_contour(x, z[zl:zt], data4_new, ax4, fig,
                                **kwargs_plot)
    p4.set_cmap(plt.cm.seismic)
    ax4.contour(
        x[0:nx:xstep],
        z[zl:zt:zstep],
        Ay[zl:zt:zstep, 0:nx:xstep],
        colors='black',
        linewidths=0.5)
    ax4.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax4.tick_params(labelsize=20)
    ax4.tick_params(axis='x', labelbottom='off')
    cbar4.set_ticks(np.arange(-0.004, 0.005, 0.002))
    cbar4.ax.tick_params(labelsize=20)
    fname4 = fname3 + fname1 + fname2
    ax4.text(
        0.02,
        0.8,
        fname4,
        color='darkred',
        fontsize=24,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='left',
        verticalalignment='center',
        transform=ax4.transAxes)

    ys -= height + gap
    ax5 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep": 1, "zstep": 1, "vmin": vmin, "vmax": vmax}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p5, cbar5 = plot_2d_contour(x, z[zl:zt], vdot_div_ptensor_new, ax5, fig,
                                **kwargs_plot)
    p5.set_cmap(plt.cm.seismic)
    ax5.contour(
        x[0:nx:xstep],
        z[zl:zt:zstep],
        Ay[zl:zt:zstep, 0:nx:xstep],
        colors='black',
        linewidths=0.5)
    ax5.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax5.tick_params(labelsize=20)
    ax5.tick_params(axis='x', labelbottom='off')
    cbar5.set_ticks(np.arange(-0.004, 0.005, 0.002))
    cbar5.ax.tick_params(labelsize=20)
    ax5.text(
        0.02,
        0.8,
        r'$\mathbf{u}\cdot(\nabla\cdot\mathcal{P})$',
        color='black',
        fontsize=24,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='left',
        verticalalignment='center',
        transform=ax5.transAxes)

    ys -= height + gap
    ax6 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep": 1, "zstep": 1, "vmin": vmin, "vmax": vmax}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p6, cbar6 = plot_2d_contour(x, z[zl:zt], jdote_new, ax6, fig,
                                **kwargs_plot)
    p6.set_cmap(plt.cm.seismic)
    ax6.contour(
        x[0:nx:xstep],
        z[zl:zt:zstep],
        Ay[zl:zt:zstep, 0:nx:xstep],
        colors='black',
        linewidths=0.5)
    ax6.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax6.tick_params(labelsize=20)
    ax6.tick_params(axis='x', labelbottom='off')
    cbar6.set_ticks(np.arange(-0.004, 0.005, 0.002))
    cbar6.ax.tick_params(labelsize=20)
    fname6 = r'$' + '\mathbf{j}_' + species + '\cdot\mathbf{E}' + '$'
    ax6.text(
        0.02,
        0.8,
        fname6,
        color='black',
        fontsize=24,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='left',
        verticalalignment='center',
        transform=ax6.transAxes)

    ys -= height + gap
    w1, h1 = fig.get_size_inches()
    width1 = width * 0.98 - 0.05 / w1
    ax7 = fig.add_axes([xs, ys, width1, height])
    ax7.plot(x, pdiv_u_sum, linewidth=2, color='r')
    ax7.plot(x, pshear_sum, linewidth=2, color='g')
    ax7.plot(x, pcomp1_sum, linewidth=2, color='b')
    ax7.plot(x, pcomp2_sum, linewidth=2, color='darkred')
    ax7.plot(x, pcomp3_sum, linewidth=2, color='k')
    ax7.plot(x, jdote_sum, linewidth=2, color='k', linestyle='-.')
    xmax = np.max(x)
    xmin = np.min(x)
    # ax7.set_ylim([-0.2, 0.2])
    ax7.plot([xmin, xmax], [0, 0], color='k', linestyle='--')
    ax7.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax7.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
    ax7.tick_params(labelsize=20)

    # width = 0.75
    # height = 0.73
    # xs = 0.12
    # ys = 0.96 - height
    # fig = plt.figure(figsize=[10,3])
    # ax1 = fig.add_axes([xs, ys, width, height])
    # kwargs_plot = {"xstep":1, "zstep":1, "vmin":-0.1, "vmax":0.1}
    # xstep = kwargs_plot["xstep"]
    # zstep = kwargs_plot["zstep"]
    # p1, cbar1 = plot_2d_contour(x, z, div_u, ax1, fig, **kwargs_plot)
    # p1.set_cmap(plt.cm.seismic)
    # ax1.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep],
    #         colors='black', linewidths=0.5)
    # ax1.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
    # ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    # ax1.tick_params(labelsize=20)
    # cbar1.ax.tick_params(labelsize=20)

    plt.show()
    # if not os.path.isdir('../img/'):
    #     os.makedirs('../img/')
    # if not os.path.isdir('../img/img_compression/'):
    #     os.makedirs('../img/img_compression/')
    # fname = 'compression' + str(current_time).zfill(3) + \
    #         '_' + species + '.jpg'
    # fname = '../img/img_compression/' + fname
    # fig.savefig(fname)
    # plt.close()


def plot_compression_cut(pic_info, species, current_time):
    """Plot compression related terms.

    Args:
        pic_info: namedtuple for the PIC simulation information.
        species: 'e' for electrons, 'i' for ions.
        current_time: current time frame.
    """
    zmin, zmax = -15, 15
    xmin = xmax = 140
    kwargs = {
        "current_time": current_time,
        "xl": xmin,
        "xr": xmax,
        "zb": zmin,
        "zt": zmax
    }
    fname = "../../data1/vdot_div_ptensor00_" + species + ".gda"
    x, z, vdot_div_ptensor = read_2d_fields(pic_info, fname, **kwargs)
    fname = "../../data1/pdiv_u00_" + species + ".gda"
    x, z, pdiv_u = read_2d_fields(pic_info, fname, **kwargs)
    fname = "../../data1/div_u00_" + species + ".gda"
    x, z, div_u = read_2d_fields(pic_info, fname, **kwargs)
    fname = "../../data1/pshear00_" + species + ".gda"
    x, z, pshear = read_2d_fields(pic_info, fname, **kwargs)
    fname = "../../data1/div_vdot_ptensor00_" + species + ".gda"
    x, z, div_vdot_ptensor = read_2d_fields(pic_info, fname, **kwargs)
    fname = '../../data/u' + species + 'x.gda'
    x, z, ux = read_2d_fields(pic_info, fname, **kwargs)
    fname = '../../data/u' + species + 'y.gda'
    x, z, uy = read_2d_fields(pic_info, fname, **kwargs)
    fname = '../../data/u' + species + 'z.gda'
    x, z, uz = read_2d_fields(pic_info, fname, **kwargs)
    x, z, ex = read_2d_fields(pic_info, '../../data/ex.gda', **kwargs)
    x, z, ey = read_2d_fields(pic_info, '../../data/ey.gda', **kwargs)
    x, z, ez = read_2d_fields(pic_info, '../../data/ez.gda', **kwargs)
    fname = '../../data/n' + species + '.gda'
    x, z, nrho = read_2d_fields(pic_info, fname, **kwargs)
    nx, = x.shape
    nz, = z.shape
    if species == 'e':
        je = -(ux * ex + uy * ey + uz * ez) * nrho
    else:
        je = (ux * ex + uy * ey + uz * ez) * nrho

    pdiv_u_cum = np.cumsum(pdiv_u[:, 0])
    pshear_cum = np.cumsum(pshear[:, 0])
    vdot_div_ptensor_cum = np.cumsum(vdot_div_ptensor[:, 0])
    div_vdot_ptensor_cum = np.cumsum(div_vdot_ptensor[:, 0])
    je_cum = np.cumsum(je[:, 0])

    znew = np.linspace(zmin, zmax, nz * 10)
    pdiv_u_new = spline(z, pdiv_u[:, 0], znew)
    pshear_new = spline(z, pshear[:, 0], znew)
    div_vdot_ptensor_new = spline(z, div_vdot_ptensor[:, 0], znew)
    vdot_div_ptensor_new = spline(z, vdot_div_ptensor[:, 0], znew)
    je_new = spline(z, je[:, 0], znew)

    pdiv_u_new = spline(z, pdiv_u_cum, znew)
    pshear_new = spline(z, pshear_cum, znew)
    div_vdot_ptensor_new = spline(z, div_vdot_ptensor_cum, znew)
    vdot_div_ptensor_new = spline(z, vdot_div_ptensor_cum, znew)
    je_new = spline(z, je_cum, znew)

    width = 0.88
    height = 0.8
    xs = 0.08
    ys = 0.96 - height

    fig = plt.figure(figsize=[14, 5])
    ax1 = fig.add_axes([xs, ys, width, height])
    label1 = r'$-p\nabla\cdot\mathbf{u}$'
    label2 = r'$-(p_\parallel - p_\perp)b_ib_j\sigma_{ij}$'
    label3 = r'$\nabla\cdot(\mathbf{u}\cdot\mathcal{P})$'
    label4 = label3 + label1 + label2
    label5 = r'$\mathbf{u}\cdot(\nabla\cdot\mathcal{P})$'
    label6 = r'$\mathbf{j}\cdot\mathbf{E}$'
    # signal.medfilt(pdiv_u[:, 0], kernel_size=5)
    # p1 = ax1.plot(znew, pdiv_u_new, linewidth=2, color='r', label=label1)
    # p2 = ax1.plot(znew, pshear_new, linewidth=2, color='g', label=label2)
    # p3 = ax1.plot(znew, div_vdot_ptensor_new, linewidth=2,
    #         color='b', label=label3)
    p4 = ax1.plot(
        znew,
        pdiv_u_new + pshear_new + div_vdot_ptensor_new,
        linewidth=2,
        color='r',
        label=label4)
    p5 = ax1.plot(
        znew, vdot_div_ptensor_new, linewidth=2, color='g', label=label5)
    p6 = ax1.plot(
        znew, je_new, linewidth=2, color='b', linestyle='-', label=label6)
    ax1.set_xlabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax1.set_xlim([zmin, zmax])
    ax1.tick_params(labelsize=20)
    ax1.legend(
        loc=2,
        prop={'size': 20},
        ncol=1,
        shadow=False,
        fancybox=False,
        frameon=False)
    plt.show()
    # if not os.path.isdir('../img/'):
    #     os.makedirs('../img/')
    # if not os.path.isdir('../img/img_compression/'):
    #     os.makedirs('../img/img_compression/')
    # fname = 'compression' + str(current_time).zfill(3) + \
    #         '_' + species + '.jpg'
    # fname = '../img/img_compression/' + fname
    # fig.savefig(fname)
    # plt.close()


def angle_current(pic_info, current_time):
    """Angle between calculated current and simulation current.

    Args:
        pic_info: namedtuple for the PIC simulation information.
        current_time: current time frame.
    """
    kwargs = {
        "current_time": current_time,
        "xl": 0,
        "xr": 200,
        "zb": -15,
        "zt": 15
    }
    fname = "../../data/jx.gda"
    x, z, jx = read_2d_fields(pic_info, fname, **kwargs)
    fname = "../../data/jy.gda"
    x, z, jy = read_2d_fields(pic_info, fname, **kwargs)
    fname = "../../data/jz.gda"
    x, z, jz = read_2d_fields(pic_info, fname, **kwargs)
    fname = "../../data/uex.gda"
    x, z, uex = read_2d_fields(pic_info, fname, **kwargs)
    fname = "../../data/uey.gda"
    x, z, uey = read_2d_fields(pic_info, fname, **kwargs)
    fname = "../../data/uez.gda"
    x, z, uez = read_2d_fields(pic_info, fname, **kwargs)
    fname = "../../data/uix.gda"
    x, z, uix = read_2d_fields(pic_info, fname, **kwargs)
    fname = "../../data/uiy.gda"
    x, z, uiy = read_2d_fields(pic_info, fname, **kwargs)
    fname = "../../data/uiz.gda"
    x, z, uiz = read_2d_fields(pic_info, fname, **kwargs)
    fname = "../../data/ne.gda"
    x, z, ne = read_2d_fields(pic_info, fname, **kwargs)
    fname = "../../data/ni.gda"
    x, z, ni = read_2d_fields(pic_info, fname, **kwargs)
    x, z, Ay = read_2d_fields(pic_info, "../../data/Ay.gda", **kwargs)

    mime = pic_info.mime
    jx1 = -uex * ne + uix * ni
    jy1 = -uey * ne + uiy * ni
    jz1 = -uez * ne + uiz * ni
    absJ = np.sqrt(jx**2 + jy**2 + jz**2)
    absJ1 = np.sqrt(jx1**2 + jy1**2 + jz1**2) + 1.0E-15
    ang_current = np.arccos((jx1 * jx + jy1 * jy + jz1 * jz) / (absJ * absJ1))

    ang_current = ang_current * 180 / math.pi

    nx, = x.shape
    nz, = z.shape
    width = 0.75
    height = 0.7
    xs = 0.12
    ys = 0.9 - height
    fig = plt.figure(figsize=[10, 4])
    ax1 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep": 1, "zstep": 1}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, ang_current, ax1, fig, **kwargs_plot)
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
    cbar1.ax.set_ylabel(
        r'$\theta(\mathbf{j}, \mathbf{u}$)', fontdict=font, fontsize=24)
    cbar1.ax.tick_params(labelsize=24)

    t_wci = current_time * pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(t_wci) + '/\Omega_{ci}$'
    ax1.set_title(title, fontdict=font, fontsize=24)

    fig = plt.figure(figsize=[7, 5])
    w1, h1 = 0.8, 0.8
    xs, ys = 0.96 - w1, 0.96 - h1
    ax2 = fig.add_axes([xs, ys, w1, h1])
    ang_bins = np.arange(180)
    hist, bin_edges = np.histogram(ang_current, bins=ang_bins, density=True)
    p2 = ax2.plot(hist, linewidth=2)
    ax2.tick_params(labelsize=20)
    ax2.set_xlabel(r'$\theta$', fontdict=font, fontsize=24)
    ax2.set_ylabel(r'$f(\theta)$', fontdict=font, fontsize=24)

    plt.show()
    # plt.close()


def read_compression_data(pic_info, fdir, species):
    """
    """
    ntf = pic_info.ntf
    fname = fdir + "compression00_" + species + ".gda"
    fh = open(fname, 'r')
    data = fh.read()
    fh.close()
    compression_data = np.zeros((ntf, 6))
    index_start = 0
    index_end = 4
    ndset = 6
    print ntf
    for ct in range(ntf):
        for i in range(ndset):
            compression_data[ct, i], = \
                struct.unpack('f', data[index_start:index_end])
            index_start = index_end
            index_end += 4
    div_u = compression_data[:, 0]
    pdiv_u = compression_data[:, 1]
    div_usingle = compression_data[:, 2]
    div_upara_usingle = compression_data[:, 3]
    pdiv_usingle = compression_data[:, 4]
    pdiv_upara_usingle = compression_data[:, 5]

    fname = fdir + "compression00_exb_" + species + ".gda"
    fh = open(fname, 'r')
    data = fh.read()
    fh.close()
    compression_data = np.zeros((ntf, 6))
    index_start = 0
    index_end = 4
    ndset = 6
    print ntf
    for ct in range(ntf):
        for i in range(ndset):
            compression_data[ct, i], = \
                struct.unpack('f', data[index_start:index_end])
            index_start = index_end
            index_end += 4
    div_usingle_exb = compression_data[:, 2]
    div_upara_usingle_exb = compression_data[:, 3]
    pdiv_usingle_exb = compression_data[:, 4]
    pdiv_upara_usingle_exb = compression_data[:, 5]

    fname = fdir + "shear00_" + species + ".gda"
    fh = open(fname, 'r')
    data = fh.read()
    fh.close()
    shear_data = np.zeros((ntf, ndset))
    index_start = 0
    index_end = 4
    for ct in range(ntf):
        for i in range(ndset):
            shear_data[ct, i], = \
                struct.unpack('f', data[index_start:index_end])
            index_start = index_end
            index_end += 4
    bbsigma = shear_data[:, 0]
    pshear = shear_data[:, 1]
    bbsigma_single = shear_data[:, 2]
    bbsigma_para_usingle = shear_data[:, 3]
    pshear_single = shear_data[:, 4]
    pshear_para_usingle = shear_data[:, 5]

    fname = fdir + "shear00_exb_" + species + ".gda"
    fh = open(fname, 'r')
    data = fh.read()
    fh.close()
    shear_data = np.zeros((ntf, ndset))
    index_start = 0
    index_end = 4
    for ct in range(ntf):
        for i in range(ndset):
            shear_data[ct, i], = \
                struct.unpack('f', data[index_start:index_end])
            index_start = index_end
            index_end += 4
    bbsigma_single_exb = shear_data[:, 2]
    bbsigma_para_usingle_exb = shear_data[:, 3]
    pshear_single_exb = shear_data[:, 4]
    pshear_para_usingle_exb = shear_data[:, 5]

    fname = fdir + "div_vdot_ptensor00_" + species + ".gda"
    fh = open(fname, 'r')
    data = fh.read()
    fh.close()
    data1 = np.zeros((ntf))
    index_start = 0
    index_end = 4
    for ct in range(ntf):
        data1[ct], = struct.unpack('f', data[index_start:index_end])
        index_start = index_end
        index_end += 4
    div_vdot_ptensor = data1[:]

    fname = fdir + "vdot_div_ptensor00_" + species + ".gda"
    fh = open(fname, 'r')
    data = fh.read()
    fh.close()
    data1 = np.zeros((ntf))
    index_start = 0
    index_end = 4
    for ct in range(ntf):
        data1[ct], = struct.unpack('f', data[index_start:index_end])
        index_start = index_end
        index_end += 4
    vdot_div_ptensor = data1[:]

    dtwpe = pic_info.dtwpe
    dtwci = pic_info.dtwci
    dt_fields = pic_info.dt_fields * dtwpe / dtwci
    pdiv_u_cum = np.cumsum(pdiv_u) * dt_fields
    pshear_cum = np.cumsum(pshear) * dt_fields
    pdiv_usingle_cum = np.cumsum(pdiv_usingle) * dt_fields
    pdiv_upara_usingle_cum = np.cumsum(pdiv_upara_usingle) * dt_fields
    pshear_single_cum = np.cumsum(pshear_single) * dt_fields
    pshear_para_usingle_cum = np.cumsum(pshear_para_usingle) * dt_fields

    pdiv_usingle_exb_cum = np.cumsum(pdiv_usingle_exb) * dt_fields
    pdiv_upara_usingle_exb_cum = np.cumsum(pdiv_upara_usingle_exb) * dt_fields
    pshear_single_exb_cum = np.cumsum(pshear_single_exb) * dt_fields
    pshear_para_usingle_exb_cum = np.cumsum(pshear_para_usingle_exb) * dt_fields

    div_vdot_ptensor_cum = np.cumsum(div_vdot_ptensor) * dt_fields
    vdot_div_ptensor_cum = np.cumsum(vdot_div_ptensor) * dt_fields

    pdiv_uperp_usingle = pdiv_usingle - pdiv_upara_usingle
    pshear_perp_usingle = pshear_single - pshear_para_usingle
    pdiv_uperp_usingle_cum = pdiv_usingle_cum - pdiv_upara_usingle_cum
    pshear_perp_usingle_cum = pshear_single_cum - pshear_para_usingle_cum

    pdiv_uperp_usingle_exb = pdiv_usingle_exb - pdiv_upara_usingle_exb
    pshear_perp_usingle_exb = pshear_single_exb - pshear_para_usingle_exb
    pdiv_uperp_usingle_exb_cum = pdiv_usingle_exb_cum - pdiv_upara_usingle_exb_cum
    pshear_perp_usingle_exb_cum = np.cumsum(pshear_perp_usingle_exb) * dt_fields

    compression_collection = collections.namedtuple('compression_collection', [
        'div_u', 'pdiv_u', 'div_usingle', 'div_upara_usingle', 'pdiv_usingle',
        'pdiv_upara_usingle', 'pdiv_uperp_usingle', 'bbsigma', 'pshear',
        'bbsigma_single', 'bbsigma_para_usingle', 'pshear_single',
        'pshear_para_usingle', 'pshear_perp_usingle', 'div_vdot_ptensor',
        'vdot_div_ptensor', 'pdiv_u_cum', 'pshear_cum', 'pdiv_usingle_cum',
        'pdiv_upara_usingle_cum', 'pdiv_uperp_usingle_cum', 'pshear_single_cum',
        'pshear_para_usingle_cum', 'pshear_perp_usingle_cum',  'div_vdot_ptensor_cum',
        'vdot_div_ptensor_cum', 'div_usingle_exb', 'div_upara_usingle_exb',
        'bbsigma_single_exb', 'bbsigma_para_usingle_exb', 'pdiv_usingle_exb',
        'pdiv_upara_usingle_exb', 'pdiv_uperp_usingle_exb', 'pshear_single_exb',
        'pshear_para_usingle_exb', 'pshear_perp_usingle_exb', 'pdiv_usingle_exb_cum',
        'pdiv_upara_usingle_exb_cum', 'pdiv_uperp_usingle_exb_cum', 'pshear_single_exb_cum',
        'pshear_para_usingle_exb_cum', 'pshear_perp_usingle_exb_cum'
        ])
    compression_data = compression_collection(div_u, pdiv_u, div_usingle,
            div_upara_usingle, pdiv_usingle, pdiv_upara_usingle,
            pdiv_uperp_usingle, bbsigma, pshear, bbsigma_single,
            bbsigma_para_usingle, pshear_single, pshear_para_usingle,
            pshear_perp_usingle, div_vdot_ptensor, vdot_div_ptensor,
            pdiv_u_cum, pshear_cum, pdiv_usingle_cum, pdiv_upara_usingle_cum,
            pdiv_uperp_usingle_cum, pshear_single_cum, pshear_para_usingle_cum,
            pshear_perp_usingle_cum,  div_vdot_ptensor_cum, vdot_div_ptensor_cum,
            div_usingle_exb, div_upara_usingle_exb, bbsigma_single_exb,
            bbsigma_para_usingle_exb, pdiv_usingle_exb, pdiv_upara_usingle_exb,
            pdiv_uperp_usingle_exb, pshear_single_exb, pshear_para_usingle_exb,
            pshear_perp_usingle_exb, pdiv_usingle_exb_cum,
            pdiv_upara_usingle_exb_cum, pdiv_uperp_usingle_exb_cum,
            pshear_single_exb_cum, pshear_para_usingle_exb_cum,
            pshear_perp_usingle_exb_cum)
    return compression_data


def compression_time(pic_info, species, jdote, ylim1, root_dir='../data/'):
    """The time evolution of compression related terms.

    Args:
        pic_info: namedtuple for the PIC simulation information.
        species: 'e' for electrons, 'i' for ions.
    """
    tfields = pic_info.tfields
    read_compression_data(pic_info, root_dir, species)

    # jdote = read_jdote_data(species)
    jpolar_dote = jdote.jpolar_dote
    jpolar_dote_int = jdote.jpolar_dote_int
    jqnudote = jdote.jqnupara_dote + jdote.jqnuperp_dote
    jqnudote_cum = jdote.jqnupara_dote_int + jdote.jqnuperp_dote_int
    # jqnudote -= jpolar_dote
    # jqnudote_cum -= jpolar_dote_int
    jqnudote_cum /= enorm

    fig = plt.figure(figsize=[7, 5])
    w1, h1 = 0.8, 0.4
    xs, ys = 0.96 - w1, 0.96 - h1
    ax = fig.add_axes([xs, ys, w1, h1])
    label1 = r'$-p\nabla\cdot\boldsymbol{V}_\perp$'
    label2 = r'$-(p_\parallel - p_\perp)b_ib_j\sigma_{ij}$'
    label3 = r'$\nabla\cdot(\mathcal{P}\cdot\mathbf{u})$'
    label4 = label3 + label1 + label2
    label5 = r'$\mathbf{u}\cdot(\nabla\cdot\mathcal{P})$'
    label6 = r'$\mathbf{j}_' + species + '\cdot\mathbf{E}$'
    p1 = ax.plot(
        tfields, pdiv_uperp_usingle, linewidth=2, color='r', label=label1)
    p2 = ax.plot(
        tfields, pshear_perp_usingle, linewidth=2, color='g', label=label2)
    p3 = ax.plot(
        tfields, div_vdot_ptensor, linewidth=2, color='b', label=label3)
    p4 = ax.plot(
        tfields,
        pdiv_u + pshear + div_vdot_ptensor,
        linewidth=2,
        color='darkred',
        label=label4)
    p5 = ax.plot(
        tfields, vdot_div_ptensor, linewidth=2, color='k', label=label5)
    p6 = ax.plot(
        tfields,
        jqnudote,
        linewidth=2,
        color='k',
        linestyle='--',
        label=label6)
    ax.set_ylabel(r'$d\varepsilon_c/dt$', fontdict=font, fontsize=20)
    ax.tick_params(axis='x', labelbottom='off')
    ax.tick_params(labelsize=16)
    tmax = min(np.max(pic_info.tfields), 800)
    ax.set_xlim([0, 800])
    ax.set_ylim(ylim1)

    ax.text(
        0.65,
        0.7,
        label1,
        color='red',
        fontsize=20,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='left',
        verticalalignment='center',
        transform=ax.transAxes)
    ax.text(
        0.65,
        0.9,
        label2,
        color='green',
        fontsize=20,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='left',
        verticalalignment='center',
        transform=ax.transAxes)
    # ax.text(0.6, 0.7, label3, color='blue', fontsize=20,
    #         bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
    #         horizontalalignment='left', verticalalignment='center',
    #         transform=ax.transAxes)
    # ax.text(0.75, 0.7, label5, color='black', fontsize=20,
    #         bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
    #         horizontalalignment='left', verticalalignment='center',
    #         transform=ax.transAxes)
    ax.text(
        0.8,
        0.07,
        label4,
        color='k',
        fontsize=20,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='left',
        verticalalignment='center',
        transform=ax.transAxes)

    ys -= h1 + 0.05
    ax1 = fig.add_axes([xs, ys, w1, h1])
    p1 = ax1.plot(tfields, pdiv_uperp_usingle_cum, linewidth=2, color='r')
    p2 = ax1.plot(tfields, pshear_perp_usingle_cum, linewidth=2, color='g')
    p3 = ax1.plot(tfields, div_vdot_ptensor_cum, linewidth=2, color='b')
    p3 = ax1.plot(
        tfields,
        pdiv_u_cum + pshear_cum + div_vdot_ptensor_cum,
        linewidth=2,
        color='darkred')
    p5 = ax1.plot(tfields, vdot_div_ptensor_cum, linewidth=2, color='k')
    p6 = ax1.plot(
        tfields,
        jqnudote_cum,
        linewidth=2,
        color='k',
        linestyle='--',
        label=label6)
    ax1.set_xlabel(r'$t\Omega_{ci}$', fontdict=font, fontsize=20)
    ax1.set_ylabel(r'$\varepsilon_c$', fontdict=font, fontsize=20)
    ax1.tick_params(labelsize=16)
    # ax1.legend(loc=2, prop={'size': 20}, ncol=1,
    #            shadow=False, fancybox=False, frameon=False)
    ax1.set_xlim(ax.get_xlim())
    # ax1.set_ylim(ylim2)
    # if not os.path.isdir('../img/'):
    #     os.makedirs('../img/')
    # fname = '../img/compressional_' + species + '.eps'
    # fig.savefig(fname)
    plt.show()


def density_ratio(pic_info, current_time):
    """Electron and ion density ratio.

    Args:
        pic_info: namedtuple for the PIC simulation information.
        current_time: current time frame.
    """
    kwargs = {
        "current_time": current_time,
        "xl": 0,
        "xr": 200,
        "zb": -15,
        "zt": 15
    }
    fname = "../../data/ne.gda"
    x, z, ne = read_2d_fields(pic_info, fname, **kwargs)
    fname = "../../data/ni.gda"
    x, z, ni = read_2d_fields(pic_info, fname, **kwargs)
    x, z, Ay = read_2d_fields(pic_info, "../../data/Ay.gda", **kwargs)

    nx, = x.shape
    nz, = z.shape
    width = 0.75
    height = 0.7
    xs = 0.12
    ys = 0.9 - height
    fig = plt.figure(figsize=[10, 4])
    ax1 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep": 1, "zstep": 1, "vmin": 0.5, "vmax": 1.5}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, ne / ni, ax1, fig, **kwargs_plot)
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
    cbar1.ax.set_ylabel(r'$n_e/n_i$', fontdict=font, fontsize=24)
    cbar1.ax.tick_params(labelsize=24)

    t_wci = current_time * pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(t_wci) + '/\Omega_{ci}$'
    ax1.set_title(title, fontdict=font, fontsize=24)

    # plt.show()
    dir = '../img/img_density_ratio/'
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    if not os.path.isdir(dir):
        os.makedirs(dir)
    fname = 'density_ratio' + str(current_time).zfill(3) + '.jpg'
    fname = dir + fname
    fig.savefig(fname, dpi=300)
    plt.close()


def plot_compression_shear(pic_info, species, current_time):
    """
    Plot compression heating and shear heating terms, compared with j.E

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
    fname = "../../data1/pdiv_u00_" + species + ".gda"
    x, z, pdiv_u = read_2d_fields(pic_info, fname, **kwargs)
    fname = "../../data1/pshear00_" + species + ".gda"
    x, z, pshear = read_2d_fields(pic_info, fname, **kwargs)
    x, z, Ay = read_2d_fields(pic_info, "../../data/Ay.gda", **kwargs)
    fname = '../../data/u' + species + 'x.gda'
    x, z, ux = read_2d_fields(pic_info, fname, **kwargs)
    fname = '../../data/u' + species + 'y.gda'
    x, z, uy = read_2d_fields(pic_info, fname, **kwargs)
    fname = '../../data/u' + species + 'z.gda'
    x, z, uz = read_2d_fields(pic_info, fname, **kwargs)
    x, z, ex = read_2d_fields(pic_info, '../../data/ex.gda', **kwargs)
    x, z, ey = read_2d_fields(pic_info, '../../data/ey.gda', **kwargs)
    x, z, ez = read_2d_fields(pic_info, '../../data/ez.gda', **kwargs)
    fname = '../../data/n' + species + '.gda'
    x, z, nrho = read_2d_fields(pic_info, fname, **kwargs)
    if species == 'e':
        jdote = -(ux * ex + uy * ey + uz * ez) * nrho
    else:
        jdote = (ux * ex + uy * ey + uz * ez) * nrho

    pdiv_u_sum = np.sum(pdiv_u, axis=0)
    pdiv_u_cum = np.cumsum(pdiv_u_sum)
    pshear_sum = np.sum(pshear, axis=0)
    pshear_cum = np.cumsum(pshear_sum)
    shear_comp_sum = pdiv_u_sum + pshear_sum
    shear_comp_cum = pdiv_u_cum + pshear_cum
    jdote_sum = np.sum(jdote, axis=0)
    jdote_cum = np.cumsum(jdote_sum)

    nx, = x.shape
    nz, = z.shape
    zl = nz / 4
    zt = nz - zl

    nk = 5
    pdiv_u_new = signal.medfilt2d(pdiv_u, kernel_size=(nk, nk))
    pshear_new = signal.medfilt2d(pshear, kernel_size=(nk, nk))
    jdote_new = signal.medfilt2d(jdote, kernel_size=(nk, nk))
    shear_comp_new = pdiv_u_new + pshear_new

    width = 0.75
    height = 0.2
    xs = 0.12
    ys = 0.98 - height
    gap = 0.025

    fig = plt.figure(figsize=[10, 14])
    ax1 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep": 1, "zstep": 1, "vmin": -0.01, "vmax": 0.01}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, pdiv_u_new, ax1, fig, **kwargs_plot)
    p1.set_cmap(plt.cm.seismic)
    ax1.contour(
        x[0:nx:xstep],
        z[0:nz:zstep],
        Ay[0:nz:zstep, 0:nx:xstep],
        colors='black',
        linewidths=0.5)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax1.tick_params(labelsize=20)
    ax1.tick_params(axis='x', labelbottom='off')
    cbar1.set_ticks(np.arange(-0.01, 0.015, 0.01))
    cbar1.ax.tick_params(labelsize=20)
    fname1 = r'$-p\nabla\cdot\mathbf{u}$'
    ax1.text(
        0.02,
        0.8,
        fname1,
        color='red',
        fontsize=24,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='left',
        verticalalignment='center',
        transform=ax1.transAxes)

    ys -= height + gap
    ax2 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep": 1, "zstep": 1, "vmin": -0.01, "vmax": 0.01}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p2, cbar2 = plot_2d_contour(x, z, pshear_new, ax2, fig, **kwargs_plot)
    p2.set_cmap(plt.cm.seismic)
    ax2.contour(
        x[0:nx:xstep],
        z[0:nz:zstep],
        Ay[0:nz:zstep, 0:nx:xstep],
        colors='black',
        linewidths=0.5)
    ax2.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax2.tick_params(labelsize=20)
    ax2.tick_params(axis='x', labelbottom='off')
    cbar2.set_ticks(np.arange(-0.01, 0.015, 0.01))
    cbar2.ax.tick_params(labelsize=20)
    fname2 = r'$-(p_\parallel - p_\perp)b_ib_j\sigma_{ij}$'
    ax2.text(
        0.02,
        0.8,
        fname2,
        color='green',
        fontsize=24,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='left',
        verticalalignment='center',
        transform=ax2.transAxes)

    ys -= height + gap
    ax6 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep": 1, "zstep": 1, "vmin": -0.01, "vmax": 0.01}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p6, cbar6 = plot_2d_contour(x, z, jdote_new, ax6, fig, **kwargs_plot)
    p6.set_cmap(plt.cm.seismic)
    ax6.contour(
        x[0:nx:xstep],
        z[0:nz:zstep],
        Ay[0:nz:zstep, 0:nx:xstep],
        colors='black',
        linewidths=0.5)
    ax6.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax6.tick_params(labelsize=20)
    ax6.tick_params(axis='x', labelbottom='off')
    cbar6.set_ticks(np.arange(-0.01, 0.015, 0.01))
    cbar6.ax.tick_params(labelsize=20)
    fname6 = r'$' + '\mathbf{j}_' + species + '\cdot\mathbf{E}' + '$'
    ax6.text(
        0.02,
        0.8,
        fname6,
        color='black',
        fontsize=24,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='left',
        verticalalignment='center',
        transform=ax6.transAxes)

    ys -= height + gap
    w1, h1 = fig.get_size_inches()
    width1 = width * 0.98 - 0.05 / w1
    ax7 = fig.add_axes([xs, ys, width1, height])
    ax7.plot(x, pdiv_u_cum, linewidth=2, color='r')
    ax7.plot(x, pshear_cum, linewidth=2, color='g')
    ax7.plot(x, shear_comp_cum, linewidth=2, color='b')
    ax7.plot(x, jdote_cum, linewidth=2, color='k', linestyle='-.')
    xmax = np.max(x)
    xmin = np.min(x)
    # ax7.set_ylim([-0.2, 0.2])
    ax7.plot([xmin, xmax], [0, 0], color='k', linestyle='--')
    ax7.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax7.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
    ax7.tick_params(labelsize=20)

    # width = 0.75
    # height = 0.73
    # xs = 0.12
    # ys = 0.96 - height
    # fig = plt.figure(figsize=[10,3])
    # ax1 = fig.add_axes([xs, ys, width, height])
    # kwargs_plot = {"xstep":1, "zstep":1, "vmin":-0.1, "vmax":0.1}
    # xstep = kwargs_plot["xstep"]
    # zstep = kwargs_plot["zstep"]
    # p1, cbar1 = plot_2d_contour(x, z, div_u, ax1, fig, **kwargs_plot)
    # p1.set_cmap(plt.cm.seismic)
    # ax1.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep],
    #         colors='black', linewidths=0.5)
    # ax1.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
    # ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    # ax1.tick_params(labelsize=20)
    # cbar1.ax.tick_params(labelsize=20)

    plt.show()
    # if not os.path.isdir('../img/'):
    #     os.makedirs('../img/')
    # if not os.path.isdir('../img/img_compression/'):
    #     os.makedirs('../img/img_compression/')
    # fname = 'compression' + str(current_time).zfill(3) + \
    #         '_' + species + '.jpg'
    # fname = '../img/img_compression/' + fname
    # fig.savefig(fname)
    # plt.close()


def calc_usingle(pic_info, root_dir, current_time):
    """
    """
    kwargs = {
        "current_time": current_time,
        "xl": 0,
        "xr": 200,
        "zb": -20,
        "zt": 20
    }
    fname = root_dir + "data/vex.gda"
    x, z, vex = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/vey.gda"
    x, z, vey = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/vez.gda"
    x, z, vez = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/vix.gda"
    x, z, vix = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/viy.gda"
    x, z, viy = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/viz.gda"
    x, z, viz = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/ne.gda"
    x, z, ne = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/ni.gda"
    x, z, ni = read_2d_fields(pic_info, fname, **kwargs)

    mime = pic_info.mime
    irho = 1.0 / (ne + ni * mime)
    vx = (vex * ne + vix * ni * mime) * irho
    vy = (vey * ne + viy * ni * mime) * irho
    vz = (vez * ne + viz * ni * mime) * irho

    return (vx, vy, vz)


def plot_compression_shear_single(pic_info, root_dir, run_name, species, current_time):
    """
    Plot compression heating and shear heating terms using single fluid velocity

    Args:
        pic_info: namedtuple for the PIC simulation information.
        root_dir: simulation root directory
        species: 'e' for electrons, 'i' for ions.
        current_time: current time frame.
    """
    print(current_time)
    kwargs = {"current_time": current_time,
              "xl": 0, "xr": 200, "zb": -20, "zt": 20
              }
    fname = root_dir + "data1/pdiv_vpara_vsingle00_" + species + ".gda"
    x, z, pdiv_vpara_vsingle = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data1/pshear_para_vsingle00_" + species + ".gda"
    x, z, pshear_vpara_vsingle = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data1/pdiv_vsingle00_" + species + ".gda"
    x, z, pdiv_vsingle = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data1/pshear_vsingle00_" + species + ".gda"
    x, z, pshear_vsingle = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/Ay.gda"
    x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/v" + species + "x.gda"
    x, z, vx = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/v" + species + "y.gda"
    x, z, vy = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/v" + species + "z.gda"
    x, z, vz = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/bx.gda"
    x, z, bx = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/by.gda"
    x, z, by = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/bz.gda"
    x, z, bz = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/absB.gda"
    x, z, absB = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/ex.gda"
    x, z, ex = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/ey.gda"
    x, z, ey = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/ez.gda"
    x, z, ez = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/n" + species + ".gda"
    x, z, nrho = read_2d_fields(pic_info, fname, **kwargs)

    fname = root_dir + "data/p" + species + "-xx.gda"
    x, z, pxx = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/p" + species + "-xy.gda"
    x, z, pxy = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/p" + species + "-xz.gda"
    x, z, pxz = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/p" + species + "-yy.gda"
    x, z, pyy = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/p" + species + "-yz.gda"
    x, z, pyz = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/p" + species + "-zz.gda"
    x, z, pzz = read_2d_fields(pic_info, fname, **kwargs)
    ppara = pxx*bx*bx + pyy*by*by + pzz*bz*bz + \
            pxy*bx*by*2.0 + pxz*bx*bz*2.0 + pyz*by*bz*2.0
    ppara /= absB * absB
    pperp = 0.5 * (pxx + pyy + pzz - ppara)
    vsx, vsy, vsz = calc_usingle(pic_info, root_dir, current_time)

    nx, = x.shape
    nz, = z.shape

    smime = math.sqrt(pic_info.mime)
    dx = pic_info.dx_di * smime
    dz = pic_info.dz_di * smime
    
    ib2 = 1.0 / absB**2
    vxb = vx * bx + vy * by + vz * bz
    vparax = vxb * bx * ib2
    vparay = vxb * by * ib2
    vparaz = vxb * bz * ib2
    vperpx = vx - vparax
    vperpy = vy - vparay
    vperpz = vz - vparaz
    vsxb = vsx * bx + vsy * by + vsz * bz
    vsparax = vsxb * bx * ib2
    vsparay = vsxb * by * ib2
    vsparaz = vsxb * bz * ib2
    vsperpx = vsx - vsparax
    vsperpy = vsy - vsparay
    vsperpz = vsz - vsparaz

    div_pperp_vperp = np.gradient(pperp * vsperpx, dx, axis=1) + \
                      np.gradient(pperp * vsperpz, dz, axis=0)

    charge = -1 if species == 'e' else 1
    jqnuperp_dote = charge * nrho * (vperpx * ex + vperpy * ey + vperpz * ez)

    pdiv_vperp_vsingle = pdiv_vsingle - pdiv_vpara_vsingle
    pshear_vperp_vsingle = pshear_vsingle - pshear_vpara_vsingle

    # fdata1 = pdiv_vperp_vsingle + pshear_vpara_vsingle + div_pperp_vperp
    fdata1 = pdiv_vperp_vsingle + pshear_vpara_vsingle
    fdata2 = div_pperp_vperp
    fdata3 = jqnuperp_dote
    dv = pic_info.dx_di * pic_info.dz_di * pic_info.mime
    print(np.sum(fdata1)*dv, np.sum(fdata2)*dv, np.sum(fdata3)*dv)
    fdata1_cum = np.cumsum(np.sum(fdata1, axis=0)) * dx * dz
    fdata2_cum = np.cumsum(np.sum(fdata2, axis=0)) * dx * dz
    fdata3_cum = np.cumsum(np.sum(fdata3, axis=0)) * dx * dz

    ng = 5
    kernel = np.ones((ng, ng)) / float(ng * ng)
    fdata1 = signal.convolve2d(fdata1, kernel, 'same')
    fdata2 = signal.convolve2d(fdata2, kernel, 'same')
    fdata3 = signal.convolve2d(fdata3, kernel, 'same')

    nx, = x.shape
    nz, = z.shape
    xs0, ys0 = 0.12, 0.76
    w1, h1 = 0.80, 0.21
    vgap = 0.02
    hgap = 0.10

    fig = plt.figure(figsize=[10, 10])
    xs = xs0
    ys = ys0
    ax1 = fig.add_axes([xs, ys, w1, h1])
    vmin, vmax = -0.005, 0.005
    crange = np.arange(vmin, vmax+0.005, 0.005)
    kwargs_plot = {"xstep":1, "zstep":1, "vmin":vmin, "vmax":vmax}
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
    cbar1.set_ticks(crange)
    cbar1.ax.tick_params(labelsize=16)
    text1 = r'$-p_e\nabla\cdot\boldsymbol{u}_\perp$'
    text1 += r'$-(p_{e\parallel}-p_{e\perp})b_ib_j\sigma_{ij}$'
    ax1.text(0.02, 0.8, text1, color='r', fontsize=20,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                pad=10.0), horizontalalignment='left',
            verticalalignment='center', transform = ax1.transAxes)

    ys = ys0 -h1 - vgap
    ax2 = fig.add_axes([xs, ys, w1, h1])
    p2, cbar2 = plot_2d_contour(x, z, fdata2, ax2, fig, **kwargs_plot)
    p2.set_cmap(plt.cm.seismic)
    ax2.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep],
            colors='black', linewidths=0.5)
    ax2.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax2.tick_params(axis='x', labelbottom='off')
    ax2.tick_params(labelsize=16)
    cbar2.set_ticks(crange)
    cbar2.ax.tick_params(labelsize=16)
    text2 = r'$\nabla\cdot(p_{e\perp}\boldsymbol{u}_\perp)$'
    ax2.text(0.02, 0.8, text2, color='b', fontsize=20,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                pad=10.0), horizontalalignment='left',
            verticalalignment='center', transform = ax2.transAxes)

    ys -= h1 + vgap
    ax3 = fig.add_axes([xs, ys, w1, h1])
    p3, cbar3 = plot_2d_contour(x, z, fdata3, ax3, fig, **kwargs_plot)
    p3.set_cmap(plt.cm.seismic)
    ax3.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep],
            colors='black', linewidths=0.5)
    ax3.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax3.tick_params(axis='x', labelbottom='off')
    ax3.tick_params(labelsize=16)
    cbar3.set_ticks(crange)
    cbar3.ax.tick_params(labelsize=16)
    text3 = r'$\boldsymbol{j}_{e\perp}\cdot\boldsymbol{E}_\perp$'
    ax3.text(0.02, 0.8, text3, color='k', fontsize=20,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                pad=10.0), horizontalalignment='left',
            verticalalignment='center', transform = ax3.transAxes)

    ys -= h1 + vgap
    ax4 = fig.add_axes([xs, ys, w1, h1])
    ax4.plot(x, fdata1_cum, linewidth=2, color='r')
    ax4.plot(x, fdata2_cum, linewidth=2, color='b')
    ax4.plot(x, fdata3_cum, linewidth=2, color='k')
    ax4.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    ax4.tick_params(labelsize=16)

    fdir = '../img/compression/' + run_name + '/'
    mkdir_p(fdir)
    fname = fdir + 'comp_jdote_' + str(current_time) + '.jpg'
    fig.savefig(fname, dpi=200)

    plt.show()


def plot_shear(pic_info, species, current_time):
    """
    Plot shear heating terms.

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
    fname = "../../data1/pshear00_" + species + ".gda"
    x, z, pshear = read_2d_fields(pic_info, fname, **kwargs)
    fname = "../../data1/bbsigma00_" + species + ".gda"
    x, z, bbsigma = read_2d_fields(pic_info, fname, **kwargs)
    fname = "../../data1/ppara00_" + species + ".gda"
    x, z, ppara = read_2d_fields(pic_info, fname, **kwargs)
    fname = "../../data1/pperp00_" + species + ".gda"
    x, z, pperp = read_2d_fields(pic_info, fname, **kwargs)
    x, z, Ay = read_2d_fields(pic_info, "../../data/Ay.gda", **kwargs)

    nx, = x.shape
    nz, = z.shape
    nk = 5
    # pshear_new = signal.medfilt2d(pshear, kernel_size=(nk,nk))
    # bbsigma_new = signal.medfilt2d(bbsigma, kernel_size=(nk,nk))
    kernel = np.ones((nk, nk)) / float(nk * nk)
    pshear_new = signal.convolve2d(pshear, kernel, mode='same')
    bbsigma_new = signal.convolve2d(bbsigma, kernel, mode='same')
    pshear_sum = np.sum(pshear_new, axis=0)
    pshear_cum = np.cumsum(pshear_sum)

    width = 0.78
    height = 0.19
    xs = 0.12
    ys = 0.97 - height
    gap = 0.04

    fig = plt.figure(figsize=[10, 8])
    ax1 = fig.add_axes([xs, ys, width, height])
    if species == 'e':
        vmin, vmax = -0.04, 0.04
    else:
        vmin, vmax = -0.02, 0.02
    kwargs_plot = {"xstep": 1, "zstep": 1, "vmin": vmin, "vmax": vmax}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, bbsigma_new, ax1, fig, **kwargs_plot)
    p1.set_cmap(plt.cm.get_cmap('seismic'))
    ax1.contour(
        x[0:nx:xstep],
        z[0:nz:zstep],
        Ay[0:nz:zstep, 0:nx:xstep],
        colors='black',
        linewidths=0.5)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax1.tick_params(labelsize=20)
    ax1.tick_params(axis='x', labelbottom='off')
    cbar1.set_ticks(np.arange(vmin, vmax + 0.01, 0.02))
    cbar1.ax.tick_params(labelsize=20)
    fname1 = r'$b_ib_j\sigma_{ij}$'
    ax1.text(
        0.02,
        0.8,
        fname1,
        color='red',
        fontsize=24,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='left',
        verticalalignment='center',
        transform=ax1.transAxes)

    ys -= height + gap
    ax2 = fig.add_axes([xs, ys, width, height])
    if species == 'e':
        vmin, vmax = -0.4, 0.4
    else:
        vmin, vmax = -0.8, 0.8
    kwargs_plot = {"xstep": 1, "zstep": 1, "vmin": vmin, "vmax": vmax}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p2, cbar2 = plot_2d_contour(x, z, -ppara + pperp, ax2, fig, **kwargs_plot)
    p2.set_cmap(plt.cm.seismic)
    ax2.contour(
        x[0:nx:xstep],
        z[0:nz:zstep],
        Ay[0:nz:zstep, 0:nx:xstep],
        colors='black',
        linewidths=0.5)
    ax2.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax2.tick_params(labelsize=20)
    if species == 'e':
        cbar2.set_ticks(np.arange(vmin, vmax + 0.1, 0.2))
    else:
        cbar2.set_ticks(np.arange(vmin, vmax + 0.1, 0.4))
    cbar2.ax.tick_params(labelsize=20)
    ax2.tick_params(axis='x', labelbottom='off')
    fname2 = r'$-(p_\parallel - p_\perp)$'
    ax2.text(
        0.02,
        0.8,
        fname2,
        color='blue',
        fontsize=24,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='left',
        verticalalignment='center',
        transform=ax2.transAxes)

    ys -= height + gap
    ax3 = fig.add_axes([xs, ys, width, height])
    if species == 'e':
        vmin, vmax = -0.002, 0.002
    else:
        vmin, vmax = -0.004, 0.004
    kwargs_plot = {"xstep": 1, "zstep": 1, "vmin": vmin, "vmax": vmax}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p3, cbar3 = plot_2d_contour(x, z, pshear_new, ax3, fig, **kwargs_plot)
    p3.set_cmap(plt.cm.seismic)
    ax3.contour(
        x[0:nx:xstep],
        z[0:nz:zstep],
        Ay[0:nz:zstep, 0:nx:xstep],
        colors='black',
        linewidths=0.5)
    ax3.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax3.tick_params(labelsize=20)
    ax3.tick_params(axis='x', labelbottom='off')
    cbar3.set_ticks(np.arange(vmin, vmax + 0.001, 0.002))
    cbar3.ax.tick_params(labelsize=20)
    fname2 = r'$-(p_\parallel - p_\perp)b_ib_j\sigma_{ij}$'
    ax3.text(
        0.02,
        0.8,
        fname2,
        color='green',
        fontsize=24,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='left',
        verticalalignment='center',
        transform=ax3.transAxes)

    ys -= height + gap
    w1, h1 = fig.get_size_inches()
    width1 = width * 0.98 - 0.05 / w1
    ax4 = fig.add_axes([xs, ys, width1, height])
    p4 = ax4.plot(x, pshear_sum, color='green', linewidth=1)
    p41 = ax4.plot(
        [np.min(x), np.max(x)], [0, 0], color='black', linestyle='--')
    ax4.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
    ax4.set_ylabel(
        r'$-(p_\parallel - p_\perp)b_ib_j\sigma_{ij}$',
        fontdict=font,
        fontsize=24)
    ax4.tick_params(labelsize=20)

    # plt.show()
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    if not os.path.isdir('../img/img_compression/'):
        os.makedirs('../img/img_compression/')
    dir = '../img/img_compression/shear_only/'
    if not os.path.isdir(dir):
        os.makedirs(dir)
    fname = 'shear' + str(current_time).zfill(3) + '_' + species + '.jpg'
    fname = dir + fname
    fig.savefig(fname, dpi=400)
    plt.close()


def plot_compression_only(pic_info, species, current_time):
    """
    Plot compressional heating terms.

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
    fname = "../../data1/div_u00_" + species + ".gda"
    x, z, div_u = read_2d_fields(pic_info, fname, **kwargs)
    fname = "../../data1/pdiv_u00_" + species + ".gda"
    x, z, pdiv_u = read_2d_fields(pic_info, fname, **kwargs)
    fname = "../../data1/ppara00_" + species + ".gda"
    x, z, ppara = read_2d_fields(pic_info, fname, **kwargs)
    fname = "../../data1/pperp00_" + species + ".gda"
    x, z, pperp = read_2d_fields(pic_info, fname, **kwargs)
    x, z, Ay = read_2d_fields(pic_info, "../../data/Ay.gda", **kwargs)
    pscalar = (ppara + 2 * pperp) / 3.0

    nx, = x.shape
    nz, = z.shape
    nk = 5
    # div_u_new = signal.medfilt2d(div_u, kernel_size=(nk,nk))
    # pdiv_u_new = signal.medfilt2d(pdiv_u, kernel_size=(nk,nk))
    kernel = np.ones((nk, nk)) / float(nk * nk)
    div_u_new = signal.convolve2d(div_u, kernel, mode='same')
    pdiv_u_new = signal.convolve2d(pdiv_u, kernel, mode='same')
    pdiv_u_sum = np.sum(pdiv_u_new, axis=0)
    pdiv_u_cum = np.cumsum(pdiv_u_sum)

    width = 0.78
    height = 0.19
    xs = 0.12
    ys = 0.97 - height
    gap = 0.04

    fig = plt.figure(figsize=[10, 8])
    ax1 = fig.add_axes([xs, ys, width, height])
    if species == 'e':
        vmin, vmax = -0.04, 0.04
    else:
        vmin, vmax = -0.02, 0.02
    kwargs_plot = {"xstep": 1, "zstep": 1, "vmin": vmin, "vmax": vmax}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, div_u_new, ax1, fig, **kwargs_plot)
    p1.set_cmap(plt.cm.get_cmap('seismic'))
    ax1.contour(
        x[0:nx:xstep],
        z[0:nz:zstep],
        Ay[0:nz:zstep, 0:nx:xstep],
        colors='black',
        linewidths=0.5)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax1.tick_params(labelsize=20)
    ax1.tick_params(axis='x', labelbottom='off')
    cbar1.set_ticks(np.arange(vmin, vmax + 0.01, 0.02))
    cbar1.ax.tick_params(labelsize=20)
    fname1 = r'$\nabla\cdot\mathbf{u}$'
    ax1.text(
        0.02,
        0.8,
        fname1,
        color='red',
        fontsize=24,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='left',
        verticalalignment='center',
        transform=ax1.transAxes)

    ys -= height + gap
    ax2 = fig.add_axes([xs, ys, width, height])
    if species == 'e':
        vmax = 0.6
    else:
        vmax = 1.0
    kwargs_plot = {"xstep": 1, "zstep": 1, "vmin": 0, "vmax": vmax}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p2, cbar2 = plot_2d_contour(x, z, pscalar, ax2, fig, **kwargs_plot)
    # p2.set_cmap(plt.cm.seismic)
    ax2.contour(
        x[0:nx:xstep],
        z[0:nz:zstep],
        Ay[0:nz:zstep, 0:nx:xstep],
        colors='white',
        linewidths=0.5)
    ax2.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax2.tick_params(labelsize=20)
    cbar2.set_ticks(np.arange(0, vmax + 0.1, 0.2))
    cbar2.ax.tick_params(labelsize=20)
    ax2.tick_params(axis='x', labelbottom='off')
    fname2 = r'$p$'
    ax2.text(
        0.02,
        0.8,
        fname2,
        color='red',
        fontsize=24,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='left',
        verticalalignment='center',
        transform=ax2.transAxes)

    ys -= height + gap
    ax3 = fig.add_axes([xs, ys, width, height])
    if species == 'e':
        vmin, vmax = -0.004, 0.004
    else:
        vmin, vmax = -0.002, 0.002
    kwargs_plot = {"xstep": 1, "zstep": 1, "vmin": vmin, "vmax": vmax}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p3, cbar3 = plot_2d_contour(x, z, pdiv_u_new, ax3, fig, **kwargs_plot)
    p3.set_cmap(plt.cm.seismic)
    ax3.contour(
        x[0:nx:xstep],
        z[0:nz:zstep],
        Ay[0:nz:zstep, 0:nx:xstep],
        colors='black',
        linewidths=0.5)
    ax3.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax3.tick_params(labelsize=20)
    ax3.tick_params(axis='x', labelbottom='off')
    cbar3.set_ticks(np.arange(vmin, vmax + 0.001, 0.002))
    cbar3.ax.tick_params(labelsize=20)
    fname2 = r'$-p\nabla\cdot\mathbf{u}$'
    ax3.text(
        0.02,
        0.8,
        fname2,
        color='green',
        fontsize=24,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='left',
        verticalalignment='center',
        transform=ax3.transAxes)

    ys -= height + gap
    w1, h1 = fig.get_size_inches()
    width1 = width * 0.98 - 0.05 / w1
    ax4 = fig.add_axes([xs, ys, width1, height])
    p4 = ax4.plot(x, pdiv_u_sum, color='green', linewidth=1)
    p41 = ax4.plot(
        [np.min(x), np.max(x)], [0, 0], color='black', linestyle='--')
    ax4.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
    ax4.set_ylabel(r'$-p\nabla\cdot\mathbf{u}$', fontdict=font, fontsize=24)
    ax4.tick_params(labelsize=20)

    # plt.show()
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    if not os.path.isdir('../img/img_compression/'):
        os.makedirs('../img/img_compression/')
    dir = '../img/img_compression/compression_only/'
    if not os.path.isdir(dir):
        os.makedirs(dir)
    fname = 'compression' + str(current_time).zfill(3) + '_' + species + '.jpg'
    fname = dir + fname
    fig.savefig(fname, dpi=400)
    plt.close()


def plot_velocity_field(pic_info, species, current_time):
    """
    Plot velocity field.

    Args:
        pic_info: namedtuple for the PIC simulation information.
        species: 'e' for electrons, 'i' for ions.
        current_time: current time frame.
    """
    print(current_time)
    zb, zt = -20, 20
    xl, xr = 0, 200
    kwargs = {
        "current_time": current_time,
        "xl": xl,
        "xr": xr,
        "zb": zb,
        "zt": zt
    }
    fname = "../../data/u" + species + "x.gda"
    x, z, ux = read_2d_fields(pic_info, fname, **kwargs)
    fname = "../../data/u" + species + "z.gda"
    x, z, uz = read_2d_fields(pic_info, fname, **kwargs)
    x, z, Ay = read_2d_fields(pic_info, "../../data/Ay.gda", **kwargs)
    # X, Z = np.meshgrid(x, z)
    speed = np.sqrt(ux**2 + uz**2)
    nx, = x.shape
    nz, = z.shape

    width = 0.88
    height = 0.85
    xs = 0.06
    ys = 0.96 - height
    gap = 0.04

    fig = plt.figure(figsize=[20, 8])
    ax = fig.add_axes([xs, ys, width, height])
    p1 = ax.streamplot(
        x,
        z,
        ux,
        uz,
        color=speed,
        linewidth=1,
        density=5.0,
        cmap=plt.cm.jet,
        arrowsize=1.0)
    kwargs_plot = {"xstep": 2, "zstep": 2}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    ax.contour(
        x[0:nx:xstep],
        z[0:nz:zstep],
        Ay[0:nz:zstep, 0:nx:xstep],
        colors='black',
        linewidths=0.5)
    ax.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
    ax.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax.set_xlim([xl, xr])
    ax.set_ylim([zb, zt])
    ax.tick_params(labelsize=20)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    cbar = fig.colorbar(p1.lines, cax=cax)
    cbar.ax.tick_params(labelsize=20)
    fname = r'$u_' + species + '$'
    cbar.ax.set_ylabel(fname, fontdict=font, fontsize=24)

    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    if not os.path.isdir('../img/img_velocity_field/'):
        os.makedirs('../img/img_velocity_field/')
    fname = 'u' + species + '_' + str(current_time).zfill(3) + '.jpg'
    fname = '../img/img_velocity_field/' + fname
    fig.savefig(fname)
    # plt.show()
    plt.close()


def plot_velocity_components(pic_info, species, current_time):
    """
    Plot the 2D contour of the 3 components of the velocity field.

    Args:
        pic_info: namedtuple for the PIC simulation information.
        species: 'e' for electrons, 'i' for ions.
        current_time: current time frame.
    """
    print(current_time)
    zb, zt = -20, 20
    xl, xr = 0, 200
    kwargs = {
        "current_time": current_time,
        "xl": xl,
        "xr": xr,
        "zb": zb,
        "zt": zt
    }
    fname = "../../data/u" + species + "x.gda"
    x, z, ux = read_2d_fields(pic_info, fname, **kwargs)
    fname = "../../data/u" + species + "y.gda"
    x, z, uy = read_2d_fields(pic_info, fname, **kwargs)
    fname = "../../data/u" + species + "z.gda"
    x, z, uz = read_2d_fields(pic_info, fname, **kwargs)
    x, z, Ay = read_2d_fields(pic_info, "../../data/Ay.gda", **kwargs)
    nx, = x.shape
    nz, = z.shape

    width = 0.8
    height = 0.26
    xs = 0.12
    ys = 0.96 - height
    gap = 0.04

    fig = plt.figure(figsize=[10, 8])
    ax1 = fig.add_axes([xs, ys, width, height])
    if species == 'e':
        vmin, vmax = -0.3, 0.3
    else:
        vmin, vmax = -0.2, 0.2
    kwargs_plot = {"xstep": 1, "zstep": 1, "vmin": vmin, "vmax": vmax}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, ux, ax1, fig, **kwargs_plot)
    p1.set_cmap(plt.cm.get_cmap('seismic'))
    ax1.contour(
        x[0:nx:xstep],
        z[0:nz:zstep],
        Ay[0:nz:zstep, 0:nx:xstep],
        colors='black',
        linewidths=0.5)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax1.tick_params(labelsize=20)
    ax1.tick_params(axis='x', labelbottom='off')
    cbar1.set_ticks(np.arange(vmin, vmax + 0.1, 0.1))
    cbar1.ax.tick_params(labelsize=20)
    fname1 = r'$u_x$'
    ax1.text(
        0.02,
        0.8,
        fname1,
        color='black',
        fontsize=24,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='left',
        verticalalignment='center',
        transform=ax1.transAxes)

    ys -= height + gap
    ax2 = fig.add_axes([xs, ys, width, height])
    if species == 'e':
        vmin, vmax = -0.3, 0.3
    else:
        vmin, vmax = -0.2, 0.2
    kwargs_plot = {"xstep": 1, "zstep": 1, "vmin": vmin, "vmax": vmax}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p2, cbar2 = plot_2d_contour(x, z, uy, ax2, fig, **kwargs_plot)
    p2.set_cmap(plt.cm.seismic)
    ax2.contour(
        x[0:nx:xstep],
        z[0:nz:zstep],
        Ay[0:nz:zstep, 0:nx:xstep],
        colors='white',
        linewidths=0.5)
    ax2.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax2.tick_params(labelsize=20)
    cbar2.set_ticks(np.arange(vmin, vmax + 0.1, 0.1))
    cbar2.ax.tick_params(labelsize=20)
    ax2.tick_params(axis='x', labelbottom='off')
    fname2 = r'$u_y$'
    ax2.text(
        0.02,
        0.8,
        fname2,
        color='black',
        fontsize=24,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='left',
        verticalalignment='center',
        transform=ax2.transAxes)

    ys -= height + gap
    ax3 = fig.add_axes([xs, ys, width, height])
    if species == 'e':
        vmin, vmax = -0.3, 0.3
    else:
        vmin, vmax = -0.2, 0.2
    kwargs_plot = {"xstep": 1, "zstep": 1, "vmin": vmin, "vmax": vmax}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p3, cbar3 = plot_2d_contour(x, z, uz, ax3, fig, **kwargs_plot)
    p3.set_cmap(plt.cm.seismic)
    ax3.contour(
        x[0:nx:xstep],
        z[0:nz:zstep],
        Ay[0:nz:zstep, 0:nx:xstep],
        colors='black',
        linewidths=0.5)
    ax3.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
    ax3.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax3.tick_params(labelsize=20)
    cbar3.set_ticks(np.arange(vmin, vmax + 0.1, 0.1))
    cbar3.ax.tick_params(labelsize=20)
    fname2 = r'$u_z$'
    ax3.text(
        0.02,
        0.8,
        fname2,
        color='black',
        fontsize=24,
        bbox=dict(
            facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='left',
        verticalalignment='center',
        transform=ax3.transAxes)

    # plt.show()
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    if not os.path.isdir('../img/img_uxyz/'):
        os.makedirs('../img/img_uxyz/')
    fname = 'u' + species + '_' + str(current_time).zfill(3) + '.jpg'
    fname = '../img/img_uxyz/' + fname
    fig.savefig(fname)
    plt.close()


def move_compression():
    if not os.path.isdir('../data/'):
        os.makedirs('../data/')
    dir = '../data/compression/'
    if not os.path.isdir(dir):
        os.makedirs(dir)
    run_dir, run_names = ApJ_long_paper_runs()
    for run_dir, run_name in zip(run_dir, run_names):
        fpath = dir + run_name
        if not os.path.isdir(fpath):
            os.makedirs(fpath)
        command = "cp " + run_dir + "/pic_analysis/data/compression00* " + \
                  fpath
        os.system(command)
        command = "cp " + run_dir + "/pic_analysis/data/shear00* " + fpath
        os.system(command)
        command = "cp " + run_dir + \
                  "/pic_analysis/data/div_vdot_ptensor00* " + \
                  fpath
        os.system(command)
        command = "cp " + run_dir + \
                  "/pic_analysis/data/vdot_div_ptensor00* " + \
                  fpath
        os.system(command)


def plot_compression_time_multi(species):
    """Plot time evolution of compression and shear heating for multiple runs

    Args:
        species: particle species
    """
    dir = '../data/compression/'
    dir_jdote = '../data/jdote_data/'
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    odir = '../img/compression/'
    if not os.path.isdir(odir):
        os.makedirs(odir)
    run_dir, run_names = ApJ_long_paper_runs()
    nrun = len(run_names)
    ylim1 = np.zeros((nrun, 2))
    if species == 'e':
        ylim1[0, :] = -0.05, 0.15
        ylim1[1, :] = -0.3, 1.1
        ylim1[2, :] = -1.0, 5
        ylim1[3, :] = -10.0, 30.0
        ylim1[4, :] = -2.0, 5.0
        ylim1[5, :] = -0.1, 0.2
        ylim1[6, :] = -0.5, 1.1
        ylim1[7, :] = -3.0, 6.0
        ylim1[8, :] = -1.0, 5.0
    else:
        ylim1[0, :] = -0.1, 0.25
        ylim1[1, :] = -0.6, 2.2
        ylim1[2, :] = -2.0, 10
        ylim1[3, :] = -20.0, 60.0
        ylim1[4, :] = -4.0, 13.0
        ylim1[5, :] = -0.2, 0.4
        ylim1[6, :] = -1.0, 2.4
        ylim1[7, :] = -5.0, 15.0
        ylim1[8, :] = -3.0, 7.0
    for i in range(6, 7):
        run_name = run_names[i]
        picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
        jdote_fname = dir_jdote + 'jdote_' + run_name + '_' + species + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        jdote_data = read_data_from_json(jdote_fname)
        fpath_comp = '../data/compression/' + run_name + '/'
        compression_time(pic_info, species, jdote_data, ylim1[i, :],
                         fpath_comp)
        # oname = odir + 'compression_' + run_name + '_' + species + '.eps'
        oname = odir + 'compression_' + run_name + '_wjp_' + species + '.eps'
        plt.savefig(oname)
        plt.show()
        # plt.close()


def plot_div_v(current_time, species):
    """Plot the divergence of velocity field
    """
    print(current_time)
    spath = '/net/scratch2/xiaocanli/'
    kwargs = {
        "current_time": current_time,
        "xl": 0,
        "xr": 200,
        "zb": -20,
        "zt": 20
    }
    # rpath =  spath + 'mime25-sigma1-beta002-200-100-noperturb/'
    # rname = 'mime25_beta002_noperturb'
    # rpath =  spath + 'mime25-sigma1-beta002-guide1-200-100/'
    # rname = 'mime25_beta002_guide1'
    rpath = spath + 'mime25-guide0-beta001-200-100-sigma033/'
    rname = 'mime25_beta002_sigma033'
    picinfo_fname = '../data/pic_info/pic_info_' + rname + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    # fname = rpath + "data1/div_v00_" + species + ".gda"
    # fname = rpath + "data1/pdiv_v00_" + species + ".gda"
    fname = rpath + "data1/pshear00_" + species + ".gda"
    # fname = rpath + "data1/bbsigma00_" + species + ".gda"
    x, z, div_v = read_2d_fields(pic_info, fname, **kwargs)
    fname = rpath + "data/Ay.gda"
    x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
    nk = 7
    kernel = np.ones((nk, nk)) / float(nk * nk)
    div_v = signal.convolve2d(div_v, kernel, mode='same')
    # vmin, vmax = -0.005, 0.005
    dnorm = 5.0e-4
    div_v /= dnorm
    vmin, vmax = -1.0, 1.0
    width = 0.75
    height = 0.7
    xs = 0.12
    ys = 0.9 - height
    fig = plt.figure(figsize=[10, 4])
    ax1 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep": 1, "zstep": 1}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    kwargs_plot = {"xstep": 1, "zstep": 1, "vmin": vmin, "vmax": vmax}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, div_v, ax1, fig, **kwargs_plot)
    p1.set_cmap(plt.cm.seismic)
    ax1.contour(x, z, Ay, colors='black', linewidths=0.5)
    ax1.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax1.tick_params(labelsize=20)
    cbar1.set_ticks(np.arange(-0.004, 0.005, 0.002))
    fig.savefig('../img/bbsigma.jpg', dpi=300)
    # fig.savefig('../img/divv.jpg', dpi=300)
    # cbar1.ax.tick_params(labelsize=20)
    plt.show()


def calc_compression(run_dir, pic_info):
    """

    Args:
        run_dir: the run directory
        pic_info: namedtuple of the PIC simulation information
    """
    finterval = pic_info.fields_interval
    ct = 65
    tindex = finterval * ct
    kwargs = {
        "current_time": 0,
        "xl": 0,
        "xr": 200,
        "zb": -50,
        "zt": 50
    }
    nx = pic_info.nx
    nz = pic_info.nz
    smime = math.sqrt(pic_info.mime)
    dx = pic_info.dx_di * smime
    dz = pic_info.dz_di * smime
    fname = run_dir + 'data-ave/uix-ave_' + str(tindex) + '.gda'
    x, z, uix = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + 'data-ave/uiz-ave_' + str(tindex) + '.gda'
    x, z, uiz = read_2d_fields(pic_info, fname, **kwargs)
    divu = np.zeros((nz, nx))
    divu[:nz-1, :nx-1] = np.diff(uix, axis=1)[:nz-1, :nx-1] / dx
    divu[:nz-1, :nx-1] += np.diff(uiz, axis=0)[:nz-1, :nx-1] / dz
    ng = 5
    kernel = np.ones((ng, ng)) / float(ng * ng)
    divu = signal.convolve2d(divu, kernel, 'same')
    dmin, dmax = np.min(divu), np.max(divu)
    dbins = np.linspace(dmin, dmax, 100)
    hist, bin_edges = np.histogram(divu, bins=dbins)

    # plt.semilogy(bin_edges[:-1], hist, linewidth=2)

    xmin, xmax = np.min(x), np.max(x)
    zmin, zmax = np.min(z), np.max(z)
    vmin, vmax = -0.05, 0.05

    fig = plt.figure(figsize=[10, 5])
    xs, ys = 0.1, 0.15
    w1, h1 = 0.85, 0.8
    ax1 = fig.add_axes([xs, ys, w1, h1])
    p1 = ax1.imshow(
        divu,
        cmap=plt.cm.seismic,
        extent=[xmin, xmax, zmin, zmax],
        aspect='auto',
        origin='lower',
        vmin=vmin,
        vmax=vmax,
        interpolation='bicubic')
    ax1.tick_params(labelsize=16)
    ax1.set_xlabel(r'$x/d_i$', fontsize=20)
    ax1.set_ylabel(r'$z/d_i$', fontsize=20)
    plt.show()


def save_compression_json_single(pic_info, run_name):
    """Save compression data as json file for a single run
    """
    fdir = '../data/compression/'
    mkdir_p(fdir)

    cdir = fdir + run_name + '/'
    compression_data = read_compression_data(pic_info, cdir, 'e')
    fname = fdir + 'compression_' + run_name + '_e.json'
    compression_json = data_to_json(compression_data)
    with open(fname, 'w') as f:
        json.dump(compression_json, f)
    compression_data = read_compression_data(pic_info, cdir, 'i')
    fname = fdir + 'compression_' + run_name + '_i.json'
    compression_json = data_to_json(compression_data)
    with open(fname, 'w') as f:
        json.dump(compression_json, f)


def plot_compression_time(pic_info, run_name, species):
    """Plot the time evolution of compression-related terms.

    Args:
        pic_info: namedtuple for the PIC simulation information
        run_name: simulation run name
        species: 'e' for electrons, 'i' for ions
    """
    tfields = pic_info.tfields
    fdir = '../data/compression/'
    cdata_name = fdir + 'compression_' + run_name + '_' + species + '.json'
    cdata = read_data_from_json(cdata_name)
    jdote_name = '../data/jdote_data/jdote_' + run_name + '_' + species + '.json'
    jdote = read_data_from_json(jdote_name)

    # jpolar_dote = jdote.jpolar_dote
    fdir = '../data/jpolar_dote/'
    fname = fdir + 'jpolar_dote_' + run_name + '_' + species + '.dat'
    jpolar_dote = np.fromfile(fname)
    dtf_wpe = pic_info.dt_fields * pic_info.dtwpe / pic_info.dtwci
    jpolar_dote_int = np.cumsum(jpolar_dote) * dtf_wpe
    jqnudote = jdote.jqnupara_dote + jdote.jqnuperp_dote
    jqnudote_cum = jdote.jqnupara_dote_int + jdote.jqnuperp_dote_int
    nt, = tfields.shape
    print("-----------------------------------------------------------------------")
    print("frame pdiv_perp pshear_perp jpara_dote jperp_dote jagy_dote jpolar_dote")
    print("-----------------------------------------------------------------------")
    for tframe in range(0, nt, 10):
        print("%3d %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f" % (tframe,
              cdata.pdiv_uperp_usingle_exb[tframe], \
              cdata.pshear_perp_usingle_exb[tframe], \
              jdote.jqnupara_dote[tframe],
              jdote.jqnuperp_dote[tframe],
              jdote.jagy_dote[tframe],
              jpolar_dote[tframe]))

    fdata1 = cdata.pdiv_uperp_usingle_exb + cdata.pshear_perp_usingle_exb
    # fdata1 += jdote.jagy_dote
    fdata2 = jdote.jqnuperp_dote
    fdata2 -= jpolar_dote
    fdata2 -= jdote.jagy_dote
    f = interp1d(tfields, fdata1, kind='slinear')
    t_new = np.linspace(tfields[0], tfields[-1], 5000)
    fdata1_new = f(t_new)
    
    fig = plt.figure(figsize=[7, 5])
    w1, h1 = 0.8, 0.4
    xs, ys = 0.96 - w1, 0.96 - h1
    ax = fig.add_axes([xs, ys, w1, h1])
    label1 = r'$-p\nabla\cdot\boldsymbol{V}_\perp$'
    label2 = r'$-(p_\parallel - p_\perp)b_ib_j\sigma_{ij}$'
    label3 = r'$\nabla\cdot(\mathcal{P}\cdot\mathbf{u})$'
    label4 = label3 + label1 + label2
    label5 = r'$\mathbf{u}\cdot(\nabla\cdot\mathcal{P})$'
    label6 = r'$\mathbf{j}_' + species + '\cdot\mathbf{E}$'
    p1 = ax.plot(tfields, fdata1, linewidth=2, color='r', label=label1)
    p1 = ax.plot(tfields, jpolar_dote, linewidth=2, color='b', label=label1)
    # p11 = ax.plot(t_new, fdata1_new)
    # p1 = ax.plot(tfields, cdata.pdiv_upara_usingle, linewidth=2, color='r',
    #         label=label1, linestyle='--')
    # p2 = ax.plot(tfields, cdata.pshear_perp_usingle, linewidth=2, color='g', label=label2)
    # p3 = ax.plot(tfields, cdata.div_vdot_ptensor, linewidth=2, color='b', label=label3)
    # p4 = ax.plot(tfields, cdata.pdiv_u + cdata.pshear + cdata.div_vdot_ptensor, linewidth=2,
    #     color='darkred', label=label4)
    # p5 = ax.plot(tfields, cdata.vdot_div_ptensor, linewidth=2, color='k', label=label5)
    # p6 = ax.plot(tfields, jqnudote, linewidth=2, color='k', linestyle='--', label=label6)
    p6 = ax.plot(tfields, fdata2, linewidth=2, color='k',
            linestyle='--', label=label6)
    ax.set_ylabel(r'$d\varepsilon_c/dt$', fontdict=font, fontsize=20)
    ax.tick_params(axis='x', labelbottom='off')
    ax.tick_params(labelsize=16)
    tmax = min(np.max(pic_info.tfields), 800)
    ax.set_xlim([0, 800])
    # ax.set_ylim(ylim1)


    ax.text(0.65, 0.7, label1, color='red', fontsize=20,
        bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
    ax.text(0.65, 0.9, label2, color='green', fontsize=20,
        bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
    # ax.text(0.6, 0.7, label3, color='blue', fontsize=20,
    #         bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
    #         horizontalalignment='left', verticalalignment='center',
    #         transform=ax.transAxes)
    # ax.text(0.75, 0.7, label5, color='black', fontsize=20,
    #         bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
    #         horizontalalignment='left', verticalalignment='center',
    #         transform=ax.transAxes)
    ax.text(0.8, 0.07, label4, color='k', fontsize=20,
        bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)

    fdata1 = cdata.pdiv_uperp_usingle_exb_cum + cdata.pshear_perp_usingle_exb_cum
    # fdata1 += jdote.jagy_dote_int
    fdata2 = jdote.jqnuperp_dote_int
    fdata2 -= jpolar_dote_int
    fdata2 -= jdote.jagy_dote_int
    # print fdata1[-1] / fdata2[-1]

    ys -= h1 + 0.05
    ax1 = fig.add_axes([xs, ys, w1, h1])
    # p1 = ax1.plot(tfields, cdata.pdiv_upara_usingle_cum, linewidth=2, color='r')
    p1 = ax1.plot(tfields, fdata1, linewidth=2, color='r')
    # p2 = ax1.plot(tfields, cdata.pshear_perp_usingle_cum, linewidth=2, color='g')
    # p3 = ax1.plot(tfields, cdata.div_vdot_ptensor_cum, linewidth=2, color='b')
    # p3 = ax1.plot(tfields, pdiv_u_cum + pshear_cum + div_vdot_ptensor_cum,
    #     linewidth=2, color='darkred')
    # p5 = ax1.plot(tfields, vdot_div_ptensor_cum, linewidth=2, color='k')
    # p6 = ax1.plot(tfields, jqnudote_cum, linewidth=2, color='k',
    #     linestyle='--', label=label6)
    p6 = ax1.plot(tfields, fdata2, linewidth=2, color='k',
        linestyle='--', label=label6)
    ax1.set_xlabel(r'$t\Omega_{ci}$', fontdict=font, fontsize=20)
    ax1.set_ylabel(r'$\varepsilon_c$', fontdict=font, fontsize=20)
    ax1.tick_params(labelsize=16)
    # ax1.legend(loc=2, prop={'size': 20}, ncol=1,
    #            shadow=False, fancybox=False, frameon=False)
    ax1.set_xlim(ax.get_xlim())
    # ax1.set_ylim(ylim2)
    # if not os.path.isdir('../img/'):
    #     os.makedirs('../img/')
    # fname = '../img/compressional_' + species + '.eps'
    # fig.savefig(fname)
    plt.show()


def plot_compression_time_both(pic_info, run_name):
    """Plot the time evolution of compression-related terms for both species

    Args:
        pic_info: namedtuple for the PIC simulation information
        run_name: simulation run name
    """
    tfields = pic_info.tfields
    fdir = '../data/compression/'
    cdata_name = fdir + 'compression_' + run_name + '_e.json'
    cdata_e = read_data_from_json(cdata_name)
    cdata_name = fdir + 'compression_' + run_name + '_i.json'
    cdata_i = read_data_from_json(cdata_name)
    jdote_name = '../data/jdote_data/jdote_' + run_name + '_e.json'
    jdote_e = read_data_from_json(jdote_name)
    jdote_name = '../data/jdote_data/jdote_' + run_name + '_i.json'
    jdote_i = read_data_from_json(jdote_name)
    jdote_name = '../data/jdote_data/jdote_in_' + run_name + '_e.json'
    jdote_in_e = read_data_from_json(jdote_name)
    jdote_name = '../data/jdote_data/jdote_in_' + run_name + '_i.json'
    jdote_in_i = read_data_from_json(jdote_name)

    colors = colors_Set1_9[:5] + colors_Set1_9[6:] # remove yellow color

    fig = plt.figure(figsize=[8, 6])
    w1, h1 = 0.81, 0.32
    xs, ys = 0.96 - w1, 0.80 - h1
    ax = fig.add_axes([xs, ys, w1, h1])
    ax.set_prop_cycle('color', colors)
    label1 = r'$-p_s\nabla\cdot\boldsymbol{v}_E$'
    label2 = r'$-(p_{s\parallel} - p_{s\perp})b_ib_j\sigma_{ij}$'
    label3 = r'$\nabla\cdot(\mathcal{P}\cdot\mathbf{u})$'
    label4 = label3 + label1 + label2
    label5 = r'$\mathbf{u}\cdot(\nabla\cdot\mathcal{P})$'
    label6 = r'$\boldsymbol{j}_{s\perp}\cdot\boldsymbol{E}_\perp$'
    label7 = r'$n_sm_s(d\boldsymbol{u}_s/dt)\cdot\boldsymbol{v}_E$'
    label8 = r'$\boldsymbol{j}_\text{agy}\cdot\boldsymbol{E}_\perp$'
    label61 = label6 + r'$ - $' + label7 + r'$ - $' + label8

    fdir = '../data/jpolar_dote/'
    fname = fdir + 'jpolar_dote_' + run_name + '_e.dat'
    jpolar_dote = np.fromfile(fname)
    jpolar_dote[-1] = jpolar_dote[-2] # remove boundary spikes

    p1 = ax.plot(tfields, cdata_e.pdiv_uperp_usingle_exb, linewidth=2, label=label1)
    p2 = ax.plot(tfields, cdata_e.pshear_perp_usingle_exb, linewidth=2, label=label2)
    p4 = ax.plot(tfields, jpolar_dote, linewidth=2, label=label7)
    # fdata = jdote_in_e.jqnuperp_dote - jdote_in_e.jpolar_dote
    # jpolar_dote = jdote_e.jpolar_dote

    fdata = jdote_e.jqnuperp_dote
    fdata -= jpolar_dote
    fdata -= jdote_e.jagy_dote
    fdata[0] = fdata[1] # remove boundary spikes
    p3 = ax.plot(tfields, fdata, linewidth=2, label=label61, color=colors[1], linestyle='-',
                 marker='o', markersize=5)
    p12 = ax.plot(tfields, cdata_e.pdiv_uperp_usingle_exb + cdata_e.pshear_perp_usingle_exb,
                  linewidth=2, label=label1 + label2, color='k')
    p5 = ax.plot(tfields, jdote_e.jagy_dote, linewidth=2, label=label8)
    ax.set_ylabel(r'$d\varepsilon_e/dt$', fontsize=20)
    ax.tick_params(axis='x', labelbottom='off')
    ax.tick_params(labelsize=16)
    tmax = min(np.max(pic_info.tfields), 600)
    ax.set_xlim([0, 600])
    # ax.set_ylim([-0.2, 0.8])
    # ax.set_ylim([-0.05, 0.12])
    ax.legend(loc='upper center', prop={'size': 16}, ncol=2,
            bbox_to_anchor=(0.48, 1.68),
            # bbox_to_anchor=(0.5, 1.4),
            shadow=False, fancybox=False, frameon=False)

    ax.text(0.98, 0.8, 'electron', color='k', fontsize=20,
        bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)

    ys -= h1 + 0.05
    ax1 = fig.add_axes([xs, ys, w1, h1])
    ax1.set_prop_cycle('color', colors)

    # jpolar_dote = jdote_i.jpolar_dote
    fdir = '../data/jpolar_dote/'
    fname = fdir + 'jpolar_dote_' + run_name + '_i.dat'
    jpolar_dote = np.fromfile(fname)
    jpolar_dote[-1] = jpolar_dote[-2] # remove boundary spikes

    p1 = ax1.plot(tfields, cdata_i.pdiv_uperp_usingle_exb, linewidth=2, label=label1)
    p2 = ax1.plot(tfields, cdata_i.pshear_perp_usingle_exb, linewidth=2, label=label1)
    p4 = ax1.plot(tfields, jpolar_dote, linewidth=2, label=label7)
    # fdata = jdote_in_i.jqnuperp_dote - jdote_in_i.jpolar_dote
    fdata = jdote_i.jqnuperp_dote
    fdata -= jpolar_dote
    fdata -= jdote_i.jagy_dote
    p3 = ax1.plot(tfields, fdata, linewidth=2, label=label61, color=colors[1], linestyle='-',
                 marker='o', markersize=5)
    p12 = ax1.plot(tfields, cdata_i.pdiv_uperp_usingle_exb + cdata_i.pshear_perp_usingle_exb,
                  linewidth=2, label=label1 + label2, color='k')
    p5 = ax1.plot(tfields, jdote_i.jagy_dote, linewidth=2, label=label8)
    ax1.set_xlabel(r'$t\Omega_{ci}$', fontsize=20)
    ax1.set_ylabel(r'$d\varepsilon_i/dt$', fontsize=20)
    ax1.tick_params(labelsize=16)
    ax1.set_xlim(ax.get_xlim())
    # ax1.set_ylim([-0.05, 0.12])
    ax1.text(0.98, 0.8, 'ion', color='k', fontsize=20,
        bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='right', verticalalignment='bottom',
        transform=ax1.transAxes)
    fdir = '../img/compression/'
    mkdir_p(fdir)
    fname = fdir + 'compression_both_' + run_name + '.eps'
    fig.savefig(fname)
    plt.show()


def compression_ratio_apjl_runs():
    """Compression ratio for all ApJL runs
    """
    run_names = ['mime25_beta002_guide00_frequent_dump',
                 'mime25_beta002_guide02_frequent_dump',
                 'mime25_beta002_guide05_frequent_dump',
                 'mime25_beta002_guide10_frequent_dump',
                 'mime25_beta002_guide00_frequent_dump',
                 'mime25_beta008_guide00_frequent_dump',
                 'mime25_beta032_guide00_frequent_dump']
    nrun = len(run_names)
    ratios = np.zeros((nrun, 4))
    i = 0
    for run_name in run_names:
        picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        tfields = pic_info.tfields
        ct = find_closest(tfields, 500)
        fdir = '../data/compression/'
        cdata_name = fdir + 'compression_' + run_name + '_e.json'
        cdata_e = read_data_from_json(cdata_name)
        cdata_name = fdir + 'compression_' + run_name + '_i.json'
        cdata_i = read_data_from_json(cdata_name)
        jdote_name = '../data/jdote_data/jdote_' + run_name + '_e.json'
        jdote_e = read_data_from_json(jdote_name)
        jdote_name = '../data/jdote_data/jdote_' + run_name + '_i.json'
        jdote_i = read_data_from_json(jdote_name)
        jdote_name = '../data/jdote_data/jdote_in_' + run_name + '_e.json'
        jdote_in_e = read_data_from_json(jdote_name)
        jdote_name = '../data/jdote_data/jdote_in_' + run_name + '_i.json'
        jdote_in_i = read_data_from_json(jdote_name)
        # jdote_e = jdote_e.jqnuperp_dote_int[ct] - jdote_e.jpolar_dote_int[ct]
        # jdote_i = jdote_i.jqnuperp_dote_int[ct] - jdote_i.jpolar_dote_int[ct]
        jdote_e = jdote_e.jqnuperp_dote_int[ct]
        jdote_i = jdote_i.jqnuperp_dote_int[ct]
        ratios[i, 0] = cdata_e.pdiv_uperp_usingle_cum[ct] / jdote_e 
        ratios[i, 1] = cdata_e.pshear_perp_usingle_cum[ct] / jdote_e
        ratios[i, 2] = cdata_i.pdiv_uperp_usingle_cum[ct] / jdote_i
        ratios[i, 3] = cdata_i.pshear_perp_usingle_cum[ct] / jdote_i
        i += 1

    text0 = r'$/\boldsymbol{j}_{e\perp}\cdot\boldsymbol{E}_\perp$'
    text1 = r'$-p_e\nabla\cdot\boldsymbol{u}_\perp$' + text0
    text2 = r'$-(p_{e\parallel}-p_{e\perp})b_ib_j\sigma_{ij}$' + text0
    text0 = r'$/\boldsymbol{j}_{i\perp}\cdot\boldsymbol{E}_\perp$'
    text3 = r'$-p_i\nabla\cdot\boldsymbol{u}_\perp$' + text0
    text4 = r'$-(p_{i\parallel}-p_{i\perp})b_ib_j\sigma_{ij}$' + text0

    # Runs with different guide field
    bg = [0, 0.2, 0.5, 1.0]
    fig = plt.figure(figsize=[7, 5])
    xs, ys = 0.13, 0.16
    w1, h1 = 0.8, 0.8
    ax = fig.add_axes([xs, ys, w1, h1])
    ax.plot(bg, ratios[:4, 0], linewidth=2, linestyle='-',
            color='r', marker=".", markersize=20, label=text1)
    ax.plot(bg, ratios[:4, 1], linewidth=2, linestyle='--',
            color='r', marker=".", markersize=20, label=text2)
    ax.plot(bg, ratios[:4, 2], linewidth=2, linestyle='-',
            color='b', marker=".", markersize=20, label=text3)
    ax.plot(bg, ratios[:4, 3], linewidth=2, linestyle='--',
            color='b', marker=".", markersize=20, label=text4)
    ax.set_xlabel(r'$B_g/B_0$', fontdict=font, fontsize=20)
    ax.set_ylabel('Fraction', fontdict=font, fontsize=20)
    ax.tick_params(labelsize=16)
    ax.set_xlim([-0.1, 1.1])
    # ax.set_ylim([0, 1])
    ax.legend(loc=3, prop={'size': 16}, ncol=1,
              shadow=False, fancybox=False, frameon=False)
    fdir = '../img/compression/'
    mkdir_p(fdir)
    fig.savefig(fdir + 'comp_frac_bg.eps')

    # Runs with different plasma beta
    bg = [0.02, 0.08, 0.32]
    fig = plt.figure(figsize=[7, 5])
    xs, ys = 0.13, 0.16
    w1, h1 = 0.8, 0.8
    ax = fig.add_axes([xs, ys, w1, h1])
    ax.semilogx(bg, ratios[4:, 0], linewidth=2, linestyle='-',
            color='r', marker=".", markersize=20, label=text1)
    ax.semilogx(bg, ratios[4:, 1], linewidth=2, linestyle='--',
            color='r', marker=".", markersize=20, label=text2)
    ax.semilogx(bg, ratios[4:, 2], linewidth=2, linestyle='-',
            color='b', marker=".", markersize=20, label=text3)
    ax.semilogx(bg, ratios[4:, 3], linewidth=2, linestyle='--',
            color='b', marker=".", markersize=20, label=text4)
    ax.set_xlabel(r'$\beta_e$', fontdict=font, fontsize=20)
    ax.set_ylabel('Fraction', fontdict=font, fontsize=20)
    ax.tick_params(labelsize=16)
    ax.set_xlim([0.01, 0.5])
    # ax.set_ylim([0, 1])
    ax.legend(loc=2, prop={'size': 16}, ncol=1,
              shadow=False, fancybox=False, frameon=False)
    fdir = '../img/compression/'
    mkdir_p(fdir)
    fig.savefig(fdir + 'comp_frac_beta.eps')
    plt.show()


def jdote_calculation_test(pic_info, run_dir, current_time):
    """Test on the methods to calculate jdote

    Hydro fields, electric field and magnetic field are all staggered.
    The operation between them may change with the calculation method.
    """
    kwargs = {"current_time": current_time,
              "xl": 0, "xr": 200, "zb": -50, "zt": 50 }
    fname = run_dir + "data/vex.gda"
    x, z, vex = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/vey.gda"
    x, z, vey = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/vez.gda"
    x, z, vez = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/ne.gda"
    x, z, ne = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/ex.gda"
    x, z, ex = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/ey.gda"
    x, z, ey = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/ez.gda"
    x, z, ez = read_2d_fields(pic_info, fname, **kwargs)

    smime = math.sqrt(pic_info.mime)
    dx = pic_info.dx_di * smime
    dz = pic_info.dz_di * smime
    jdote1 = -ne * (ex * vex + ey * vey + ez * vez)
    jdote1_sum = np.sum(jdote1) * dx * dz

    nx = pic_info.nx / pic_info.topology_x
    ny = pic_info.ny / pic_info.topology_y
    nz = pic_info.nz / pic_info.topology_z
    nx2 = nx + 2
    ny2 = ny + 2
    nz2 = nz + 2
    dsize = nx2 * ny2 * nz2
    nprocs = pic_info.topology_x * pic_info.topology_y * pic_info.topology_z
    tindex = pic_info.fields_interval * current_time
    hydro_dir = run_dir + 'hydro/T.' + str(tindex) + '/'
    fields_dir = run_dir + 'fields/T.' + str(tindex) + '/'
    ehydro_name = hydro_dir + 'ehydro.' + str(tindex)
    field_name = fields_dir + 'fields.' + str(tindex)
    jdote2_sum = 0
    data_shape = ex.shape
    # vex1 = np.zeros(data_shape)
    # vey1 = np.zeros(data_shape)
    # vez1 = np.zeros(data_shape)
    # ne1 = np.zeros(data_shape)
    # ex1 = np.zeros(data_shape)
    # ey1 = np.zeros(data_shape)
    # ez1 = np.zeros(data_shape)
    for rank in range(nprocs):
        print("Rank: %d" % rank)
        ix = rank % pic_info.topology_x
        iz = rank / pic_info.topology_x
        ix1 = ix * nx
        ix2 = (ix + 1) * nx
        iz1 = iz * nz
        iz2 = (iz + 1) * nz
        fname = ehydro_name + '.' + str(rank)
        (vx, vy, vz, nrho) = read_hydro_velocity_density(fname, nx2, ny2, nz2, dsize)
        # vex1[iz1:iz2, ix1:ix2] = vx[1:-1, 1:-1]
        # vey1[iz1:iz2, ix1:ix2] = vy[1:-1, 1:-1]
        # vez1[iz1:iz2, ix1:ix2] = vz[1:-1, 1:-1]
        # ne1[iz1:iz2, ix1:ix2] = nrho[1:-1, 1:-1]
        fname = field_name + '.' + str(rank)
        (v0, pheader, fields) = read_fields(fname, nx2, ny2, nz2, dsize)
        # ex1[iz1:iz2, ix1:ix2] = fields[0, 1:-1, 1, 1:-1]
        # ey1[iz1:iz2, ix1:ix2] = fields[1, 1:-1, 1, 1:-1]
        # ez1[iz1:iz2, ix1:ix2] = fields[2, 1:-1, 1, 1:-1]
        ex = fields[0, :, 1, :]
        ey = fields[1, :, 1, :]
        ez = fields[2, :, 1, :]
        # jdote2_sum += np.sum((ex[1:-1, 1:-1]*vx[1:-1, 1:-1] +
        #                       ey[1:-1, 1:-1]*vy[1:-1, 1:-1] +
        #                       ez[1:-1, 1:-1]*vz[1:-1, 1:-1])) * dx * dz
        jdote2_sum += np.sum(0.5*(ex[1:-1, 1:-1] + ex[1:-1, 2:]) *
            (vx[1:-1, 1:-1] + vx[2:, 1:-1] + vx[1:-1, 2:] + vx[2:, 2:])*0.25 +
            0.25*(ey[1:-1, 1:-1] + ey[2:, 1:-1] + ey[1:-1:, 2:] + ey[2::, 2:]) *
            0.25*(vy[1:-1, 1:-1] + vy[2:, 1:-1] + vy[1:-1:, 2:] + vy[2::, 2:]) +
            0.5*(ez[1:-1, 1:-1] + ez[2:, 1:-1]) *
            0.25*(vz[1:-1, 1:-1] + vz[2:, 1:-1] + vz[1:-1:, 2:] + vz[2::, 2:])) * dx * dz
    # vex1 /= -ne1
    # vey1 /= -ne1
    # vez1 /= -ne1
    # ne1 = np.abs(ne1)
    # print(np.min(vex - vex1), np.max(vex - vex1))
    # print(np.min(vey - vey1), np.max(vey - vey1))
    # print(np.min(vez - vez1), np.max(vez - vez1))
    # print(np.min(ex - ex1), np.max(ex - ex1))
    # print(np.min(ey - ey1), np.max(ey - ey1))
    # print(np.min(ez - ez1), np.max(ez - ez1))
    # print(np.min(ne - ne1), np.max(ne - ne1))
    # jdote2 = -(ex1 * vex1 + ey1 * vey1 + ez1 * vez1) * ne1
    # jdote2_sum = np.sum(jdote2) * dx * dz

    print("Total jdote at %d: %f %f" % (current_time, jdote1_sum, jdote2_sum))


def calc_jpolar_dote(pic_info, current_time, run_dir, species):
    """Calculate the energy conversion due to polarization drift (inertial term)
    """
    kwargs = {"current_time": current_time,
              "xl": 0, "xr": 200, "zb": -50, "zt": 50 }
    fname = run_dir + "data/v" + species + "x.gda"
    x, z, vx = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/v" + species + "y.gda"
    x, z, vy = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/v" + species + "z.gda"
    x, z, vz = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/n" + species + ".gda"
    x, z, nrho = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/u" + species + "x.gda"
    x, z, ux = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/u" + species + "y.gda"
    x, z, uy = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/u" + species + "z.gda"
    x, z, uz = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/ke-" + species + ".gda"
    x, z, ke = read_2d_fields(pic_info, fname, **kwargs)

    fname = run_dir + "data/v" + species + "x_pre.gda"
    x, z, vx_pre = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/v" + species + "y_pre.gda"
    x, z, vy_pre = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/v" + species + "z_pre.gda"
    x, z, vz_pre = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/n" + species + "_pre.gda"
    x, z, nrho_pre = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/u" + species + "x_pre.gda"
    x, z, ux_pre = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/u" + species + "y_pre.gda"
    x, z, uy_pre = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/u" + species + "z_pre.gda"
    x, z, uz_pre = read_2d_fields(pic_info, fname, **kwargs)

    fname = run_dir + "data/v" + species + "x_post.gda"
    x, z, vx_post = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/v" + species + "y_post.gda"
    x, z, vy_post = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/v" + species + "z_post.gda"
    x, z, vz_post = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/n" + species + "_post.gda"
    x, z, nrho_post = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/u" + species + "x_post.gda"
    x, z, ux_post = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/u" + species + "y_post.gda"
    x, z, uy_post = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/u" + species + "z_post.gda"
    x, z, uz_post = read_2d_fields(pic_info, fname, **kwargs)

    fname = run_dir + "data/ex.gda"
    x, z, ex = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/ey.gda"
    x, z, ey = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/ez.gda"
    x, z, ez = read_2d_fields(pic_info, fname, **kwargs)

    fname = run_dir + "data/ex_pre.gda"
    x, z, ex_pre = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/ey_pre.gda"
    x, z, ey_pre = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/ez_pre.gda"
    x, z, ez_pre = read_2d_fields(pic_info, fname, **kwargs)

    fname = run_dir + "data/ex_post.gda"
    x, z, ex_post = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/ey_post.gda"
    x, z, ey_post = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/ez_post.gda"
    x, z, ez_post = read_2d_fields(pic_info, fname, **kwargs)

    fname = run_dir + "data/bx.gda"
    x, z, bx = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/by.gda"
    x, z, by = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/bz.gda"
    x, z, bz = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/absB.gda"
    x, z, absB = read_2d_fields(pic_info, fname, **kwargs)

    fname = run_dir + "data/bx_pre.gda"
    x, z, bx_pre = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/by_pre.gda"
    x, z, by_pre = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/bz_pre.gda"
    x, z, bz_pre = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/absB_pre.gda"
    x, z, absB_pre = read_2d_fields(pic_info, fname, **kwargs)

    fname = run_dir + "data/bx_post.gda"
    x, z, bx_post = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/by_post.gda"
    x, z, by_post = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/bz_post.gda"
    x, z, bz_post = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/absB_post.gda"
    x, z, absB_post = read_2d_fields(pic_info, fname, **kwargs)

    if species == 'e':
        pmass = 1.0
        charge = -1.0
    else:
        pmass = pic_info.mime
        charge = 1.0
    idt = 0.5 / pic_info.dtwpe

    ib2 = 1.0 / absB**2
    ib2_pre = 1.0 / absB_pre**2
    ib2_post = 1.0 / absB_post**2

    sigma = 3
    ex = gaussian_filter(ex, sigma)
    ey = gaussian_filter(ey, sigma)
    ez = gaussian_filter(ez, sigma)
    ex_pre = gaussian_filter(ex_pre, sigma)
    ey_pre = gaussian_filter(ey_pre, sigma)
    ez_pre = gaussian_filter(ez_pre, sigma)
    ex_post = gaussian_filter(ex_post, sigma)
    ey_post = gaussian_filter(ey_post, sigma)
    ez_post = gaussian_filter(ez_post, sigma)

    vexb_x = (ey * bz - ez * by) * ib2
    vexb_y = (ez * bx - ex * bz) * ib2
    vexb_z = (ex * by - ey * bx) * ib2

    vexb_pre_x = (ey_pre * bz_pre - ez_pre * by_pre) * ib2_pre
    vexb_pre_y = (ez_pre * bx_pre - ex_pre * bz_pre) * ib2_pre
    vexb_pre_z = (ex_pre * by_pre - ey_pre * bx_pre) * ib2_pre
    vexb_post_x = (ey_post * bz_post - ez_post * by_post) * ib2_post
    vexb_post_y = (ez_post * bx_post - ex_post * bz_post) * ib2_post
    vexb_post_z = (ex_post * by_post - ey_post * bx_post) * ib2_post

    # vexb_x = gaussian_filter(vexb_x, sigma)
    # vexb_y = gaussian_filter(vexb_y, sigma)
    # vexb_z = gaussian_filter(vexb_z, sigma)
    # vexb_pre_x = gaussian_filter(vexb_pre_x, sigma)
    # vexb_pre_y = gaussian_filter(vexb_pre_y, sigma)
    # vexb_pre_z = gaussian_filter(vexb_pre_z, sigma)
    # vexb_post_x = gaussian_filter(vexb_post_x, sigma)
    # vexb_post_y = gaussian_filter(vexb_post_y, sigma)
    # vexb_post_z = gaussian_filter(vexb_post_z, sigma)

    smime = math.sqrt(pic_info.mime)
    dx = pic_info.dx_di * smime
    dz = pic_info.dz_di * smime
    dv = dx * dz

    dt = pic_info.dtwpe * 2
    idt = 1.0 / dt

    tmpx = (ux_post - ux_pre) * idt
    tmpy = (uy_post - uy_pre) * idt
    tmpz = (uz_post - uz_pre) * idt
    tmpx += vx * np.gradient(ux, dx, axis=1) + vz * np.gradient(ux, dz, axis=0)
    tmpy += vx * np.gradient(uy, dx, axis=1) + vz * np.gradient(uy, dz, axis=0)
    tmpz += vx * np.gradient(uz, dx, axis=1) + vz * np.gradient(uz, dz, axis=0)

    vdotB = vx * bx + vy * by + vz * bz
    vx_para = vdotB * bx * ib2
    vy_para = vdotB * by * ib2
    vz_para = vdotB * bz * ib2

    vx_perp = vx - vx_para
    vy_perp = vy - vy_para
    vz_perp = vz - vz_para

    # jpolar_dote = tmpx * vexb_x + tmpy * vexb_y + tmpz * vexb_z
    # jpolar_dote = (tmpx * vx_para + tmpy * vy_para + tmpz * vz_para) * nrho * pmass
    jpolar_dote = (tmpx * vx_perp + tmpy * vy_perp + tmpz * vz_perp) * nrho * pmass
    # jpolar_x = by * tmpz - bz * tmpy
    # jpolar_y = bz * tmpx - bx * tmpz
    # jpolar_z = bx * tmpy - by * tmpx
    # jpolar_dote = (jpolar_x * ex + jpolar_y * ey +
    #                jpolar_z * ez) * ib2

    # udot_vexb = ux * vexb_x + uy * vexb_y + uz * vexb_z
    # udot_vexb_pre = (ux_pre * vexb_pre_x + uy_pre * vexb_pre_y +
    #                  uz_pre * vexb_pre_z)
    # udot_vexb_post = (ux_post * vexb_post_x + uy_post * vexb_post_y +
    #                   uz_post * vexb_post_z)
    # jpolar_dote = (udot_vexb_post - udot_vexb_pre) * idt
    # jpolar_dote += (vx * np.gradient(udot_vexb, dx, axis=1) +
    #                 vz * np.gradient(udot_vexb, dz, axis=0))
    # dvex_dt = ((vexb_post_x - vexb_pre_x) * idt +
    #            vx * np.gradient(vexb_x, dx, axis=1) +
    #            vz * np.gradient(vexb_x, dz, axis=0))
    # dvey_dt = ((vexb_post_y - vexb_pre_y) * idt +
    #            vx * np.gradient(vexb_y, dx, axis=1) +
    #            vz * np.gradient(vexb_y, dz, axis=0))
    # dvez_dt = ((vexb_post_z - vexb_pre_z) * idt +
    #            vx * np.gradient(vexb_z, dx, axis=1) +
    #            vz * np.gradient(vexb_z, dz, axis=0))
    # jpolar_dote -= ux * dvex_dt + uy * dvey_dt + uz * dvez_dt
    # # jpolar_dote = ux * dvex_dt + uy * dvey_dt + uz * dvez_dt
    # jpolar_dote *= nrho * pmass

    # ib2_pre = 1.0 / (bx_pre**2 + by_pre**2 + bz_pre**2)
    # ib2_post = 1.0 / (bx_post**2 + by_post**2 + bz_post**2)
    # edotb = ex * bx + ey * by + ez * bz
    # ex_perp = ex - edotb * bx * ib2
    # ey_perp = ey - edotb * by * ib2
    # ez_perp = ez - edotb * bz * ib2
    # edotb_post = ex_post * bx_post + ey_post * by_post + ez_post * bz_post
    # ex_perp_post = ex - edotb_post * bx_post * ib2_post
    # ey_perp_post = ey - edotb_post * by_post * ib2_post
    # ez_perp_post = ez - edotb_post * bz_post * ib2_post
    # edotb_pre = ex_pre * bx_pre + ey_pre * by_pre + ez_pre * bz_pre
    # ex_perp_pre = ex - edotb_pre * bx_pre * ib2_pre
    # ey_perp_pre = ey - edotb_pre * by_pre * ib2_pre
    # ez_perp_pre = ez - edotb_pre * bz_pre * ib2_pre

    # dex_perp_dt = (ex_perp_post - ex_perp_pre) * idt
    # dey_perp_dt = (ey_perp_post - ey_perp_pre) * idt
    # dez_perp_dt = (ez_perp_post - ez_perp_pre) * idt
    # dex_perp_dt += (vx * np.gradient(ex_perp, dx, axis=1) +
    #                 vz * np.gradient(ex_perp, dz, axis=0))
    # dey_perp_dt += (vx * np.gradient(ey_perp, dx, axis=1) +
    #                 vz * np.gradient(ey_perp, dz, axis=0))
    # dez_perp_dt += (vx * np.gradient(ez_perp, dx, axis=1) +
    #                 vz * np.gradient(ez_perp, dz, axis=0))
    # tmp = (ke + nrho * pmass) * ib2
    # vpx = dex_perp_dt * tmp
    # vpy = dey_perp_dt * tmp
    # vpz = dez_perp_dt * tmp

    # jpolar_dote = vpx * ex + vpy * ey + vpz * ez

#     dpx_dt = (nrho_post * ux_post - nrho_pre * ux_pre) * idt
#     dpy_dt = (nrho_post * uy_post - nrho_pre * uy_pre) * idt
#     dpz_dt = (nrho_post * uz_post - nrho_pre * uz_pre) * idt

#     div_nvu_x = np.gradient(nrho * vx * ux, dx, axis=1) + np.gradient(nrho * vx * uz, dz, axis=0)
#     div_nvu_y = np.gradient(nrho * vy * ux, dx, axis=1) + np.gradient(nrho * vy * uz, dz, axis=0)
#     div_nvu_z = np.gradient(nrho * vz * ux, dx, axis=1) + np.gradient(nrho * vz * uz, dz, axis=0)

#     # jpolar_dote = div_nvu_x * vexb_x + div_nvu_y * vexb_y + div_nvu_z * vexb_z
#     # jpolar_dote += dpx_dt * vexb_x + dpy_dt * vexb_y + dpz_dt * vexb_z
#     jpolar_dote = dpx_dt * vexb_x + dpy_dt * vexb_y + dpz_dt * vexb_z
#     jpolar_dote *= pmass

    print("Min and max: %f %f" % (np.min(jpolar_dote), np.max(jpolar_dote)))
    jpolar_dote_tot = np.sum(jpolar_dote) * dv
    print(jpolar_dote_tot)
    vmin = -0.1
    vmax = -vmin
    # plt.imshow(jpolar_dote, cmap=plt.cm.seismic, vmin=vmin, vmax=vmax)
    jpolar_dote_cum = np.cumsum(np.sum(jpolar_dote, axis=0))
    plt.plot(x, jpolar_dote_cum * dv, linewidth=2)
    plt.show()

    return jpolar_dote_tot


def calc_comperssional_heating(pic_info, current_time, run_dir, species):
    """Calculate the energy conversion due to compression 
    """
    kwargs = {"current_time": current_time,
              "xl": 0, "xr": 200, "zb": -50, "zt": 50 }
    fname = run_dir + "data/ex.gda"
    x, z, ex = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/ey.gda"
    x, z, ey = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/ez.gda"
    x, z, ez = read_2d_fields(pic_info, fname, **kwargs)

    fname = run_dir + "data/bx.gda"
    x, z, bx = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/by.gda"
    x, z, by = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/bz.gda"
    x, z, bz = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/absB.gda"
    x, z, absB = read_2d_fields(pic_info, fname, **kwargs)

    fname = run_dir + "data/p" + species + "-xx.gda"
    x, z, pxx = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/p" + species + "-yy.gda"
    x, z, pyy = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/p" + species + "-zz.gda"
    x, z, pzz = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/p" + species + "-xy.gda"
    x, z, pxy = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/p" + species + "-xz.gda"
    x, z, pxz = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/p" + species + "-yz.gda"
    x, z, pyz = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/p" + species + "-yx.gda"
    x, z, pyx = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/p" + species + "-zx.gda"
    x, z, pzx = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/p" + species + "-zy.gda"
    x, z, pzy = read_2d_fields(pic_info, fname, **kwargs)

    smime = math.sqrt(pic_info.mime)
    dx = pic_info.dx_di * smime
    dz = pic_info.dz_di * smime
    dv = dx * dz

    ib2 = 1.0 / absB**2
    vx_exb = (ey * bz - ez * by) * ib2
    vy_exb = (ez * bx - ex * bz) * ib2
    vz_exb = (ex * by - ey * bx) * ib2
    divv = np.gradient(vx_exb, dx, axis=1) + \
           np.gradient(vz_exb, dz, axis=0)
    sigmaxx = np.gradient(vx_exb, dx, axis=1) - divv / 3.0
    sigmayy = -divv / 3.0
    sigmazz = np.gradient(vz_exb, dz, axis=0) - divv / 3.0
    sigmaxy = 0.5 * np.gradient(vy_exb, dx, axis=1)
    sigmaxz = 0.5 * (np.gradient(vz_exb, dx, axis=1) + 
                     np.gradient(vx_exb, dz, axis=0))
    sigmayz = 0.5 * np.gradient(vy_exb, dz, axis=0)

    bbsigma = sigmaxx * bx**2 + sigmayy * by**2 + sigmazz * bz**2 + \
              2.0 * sigmaxy * bx * by + 2.0 * sigmaxz * bx * bz + \
              2.0 * sigmayz * by * bz
    bbsigma *= ib2

    pscalar = (pxx + pyy + pzz) / 3.0
    ppara = pxx * bx**2 + pyy * by**2 + pzz * bz**2 + \
            (pxy + pyx) * bx * by + (pxz + pzx) * bx * bz + \
            (pyz + pzy) * by * bz
    ppara *= ib2
    pperp = (pscalar * 3 - ppara) * 0.5
    ecov_comp = -np.sum(pscalar*divv) * dv
    ecov_shear = np.sum((pperp - ppara) * bbsigma) * dv

    divpx = np.gradient(pxx, dx, axis=1) + np.gradient(pxz, dz, axis=0)
    divpy = np.gradient(pyx, dx, axis=1) + np.gradient(pyz, dz, axis=0)
    divpz = np.gradient(pzx, dx, axis=1) + np.gradient(pzz, dz, axis=0)
    ecov = divpx * vx_exb + divpy * vy_exb + divpz * vz_exb
    ecov_pre1 = np.sum(ecov) * dv

    dvx_dx = np.gradient(vx_exb, dx, axis=1)
    dvy_dx = np.gradient(vy_exb, dx, axis=1)
    dvz_dx = np.gradient(vz_exb, dx, axis=1)
    dvx_dz = np.gradient(vx_exb, dz, axis=0)
    dvy_dz = np.gradient(vy_exb, dz, axis=0)
    dvz_dz = np.gradient(vz_exb, dz, axis=0)
    ecov = pxx * dvx_dx + pxy * dvy_dx + pxz * dvz_dx + \
           pzx * dvx_dz + pzy * dvy_dz + pzz * dvz_dz
    ecov_pre2 = np.sum(ecov) * dv

    print(ecov_comp, ecov_shear, ecov_pre1, ecov_pre2)


def calc_para_perp_heating(pic_info, current_time, run_dir, species):
    """Calculate the energy conversion due to para and perp E-field 
    """
    kwargs = {"current_time": current_time,
              "xl": 0, "xr": 200, "zb": -50, "zt": 50 }
    fname = run_dir + "data/ex.gda"
    x, z, ex = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/ey.gda"
    x, z, ey = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/ez.gda"
    x, z, ez = read_2d_fields(pic_info, fname, **kwargs)

    fname = run_dir + "data/bx.gda"
    x, z, bx = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/by.gda"
    x, z, by = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/bz.gda"
    x, z, bz = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/absB.gda"
    x, z, absB = read_2d_fields(pic_info, fname, **kwargs)

    fname = run_dir + "data/v" + species + "x.gda"
    x, z, vx = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/v" + species + "y.gda"
    x, z, vy = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/v" + species + "z.gda"
    x, z, vz = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/n" + species + ".gda"
    x, z, nrho = read_2d_fields(pic_info, fname, **kwargs)

    smime = math.sqrt(pic_info.mime)
    dx = (x[1] - x[0]) * smime
    dz = (z[1] - z[0]) * smime
    dxh = 0.5 * dx
    dzh = 0.5 * dz
    lx = pic_info.lx_di * smime
    lz = pic_info.lz_di * smime
    nx = pic_info.nx
    nz = pic_info.nz
    x1 = np.linspace(-dxh, lx + dxh, nx + 2)
    x2 = np.linspace(-dx, lx, nx + 2)
    z1 = np.linspace(-dzh - 0.5 * lz, 0.5 * lz + dzh, nz + 2)
    z2 = np.linspace(-dz - 0.5 * lz, 0.5 * lz, nz + 2)

    f_ex = RectBivariateSpline(x1[1:-1], z2[1:-1], ex.T)
    f_ey = RectBivariateSpline(x2[1:-1], z2[1:-1], ey.T)
    f_ez = RectBivariateSpline(x2[1:-1], z1[1:-1], ez.T)
    f_bx = RectBivariateSpline(x2[1:-1], z1[1:-1], bx.T)
    f_by = RectBivariateSpline(x1[1:-1], z1[1:-1], by.T)
    f_bz = RectBivariateSpline(x1[1:-1], z2[1:-1], bz.T)
    f_vx = RectBivariateSpline(x2[1:-1], z2[1:-1], vx.T)
    f_vy = RectBivariateSpline(x2[1:-1], z2[1:-1], vy.T)
    f_vz = RectBivariateSpline(x2[1:-1], z2[1:-1], vz.T)
    f_nrho = RectBivariateSpline(x2[1:-1], z2[1:-1], nrho.T)

    xv, zv = np.meshgrid(x1[1:-1], z1[1:-1])

    ex = f_ex(xv, zv, grid=False)
    ey = f_ey(xv, zv, grid=False)
    ez = f_ez(xv, zv, grid=False)
    bx = f_bx(xv, zv, grid=False)
    by = f_by(xv, zv, grid=False)
    bz = f_bz(xv, zv, grid=False)
    vx = f_vx(xv, zv, grid=False)
    vy = f_vy(xv, zv, grid=False)
    vz = f_vz(xv, zv, grid=False)
    nrho = f_nrho(xv, zv, grid=False)

    absB = np.sqrt(bx**2 + by**2 + bz**2)

    charge = -1 if species == 'e' else 1.0

    ib2 = 1.0 / absB**2
    vdotB = vx * bx + vy * by + vz * bz
    vparax = vdotB * bx * ib2
    vparay = vdotB * by * ib2
    vparaz = vdotB * bz * ib2

    dv = dx * dz
    jdote_para = charge * nrho * (vparax * ex + vparay * ey + vparaz * ez)
    jdote_perp = charge * nrho * (vx * ex + vy * ey + vz * ez) - jdote_para

    jdote_para_tot = np.sum(jdote_para) * dv
    jdote_perp_tot = np.sum(jdote_perp) * dv

    print(jdote_para_tot, jdote_perp_tot)

    jdotes = np.asarray([jdote_para_tot, jdote_perp_tot])

    return jdotes


def calc_jpolar_dote_multi(pic_info, run_dir, run_name, species):
    """
    """
    ntf = pic_info.ntf
    # jdote = np.zeros(ntf)
    jdote = np.zeros((ntf, 2))
    calc_jpolar_dote(pic_info, 30, run_dir, species)
    # calc_para_perp_heating(pic_info, 10, run_dir, species)
    # calc_comperssional_heating(pic_info, 30, run_dir, species)
    # for ct in range(ntf):
    #     print(ct)
    #     # jdote[ct] = calc_jpolar_dote(pic_info, ct, run_dir, species)
    #     jdote[ct] = calc_para_perp_heating(pic_info, ct, run_dir, species)

    # # var = 'jpolar_dote'
    # var = 'jpara_perp_dote'
    # fdir = '../data/jdote_data/' + var + '/'
    # mkdir_p(fdir)
    # fname = fdir + var + '_' + run_name + '_' + species + '.dat'
    # jdote.tofile(fname)
    # jdote = np.fromfile(fname)
    # jdote = jdote.reshape((ntf, 2))
    # fig, ax = plt.subplots()
    # ax.plot(jdote, linewidth=2)
    # ax.plot(np.sum(jdote, axis=1), linewidth=2)
    # ax.set_ylim([-0.15, 0.15])
    # plt.show()


def calc_jpolar_dote_continusou_dump(picinfo_fname, current_time, run_dir, species):
    """Calculate the energy conversion due to polarization drift (inertial term)

    The field data are continuous 3 frames, so the calculation is more accurate
    """
    pic_info = read_data_from_json(picinfo_fname)
    nx = pic_info.nx
    nz = pic_info.nz
    fname = run_dir + "data/u" + species + "x_post.gda"
    statinfo = os.stat(fname)
    file_size = statinfo.st_size
    required_size = (current_time + 1) * nx * nz * 4

    if file_size < required_size:
        idt = 1.0 / pic_info.dtwpe
    else:
        idt = 0.5 / pic_info.dtwpe

    print("Time frame: %d" % current_time)
    kwargs = {"current_time": current_time,
              "xl": 0, "xr": 200, "zb": -50, "zt": 50 }
    fname = run_dir + "data/v" + species + "x.gda"
    x, z, vx = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/v" + species + "y.gda"
    x, z, vy = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/v" + species + "z.gda"
    x, z, vz = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/n" + species + ".gda"
    x, z, nrho = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/u" + species + "x.gda"
    x, z, ux = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/u" + species + "y.gda"
    x, z, uy = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/u" + species + "z.gda"
    x, z, uz = read_2d_fields(pic_info, fname, **kwargs)

    fname = run_dir + "data/u" + species + "x_pre.gda"
    x, z, ux_pre = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/u" + species + "y_pre.gda"
    x, z, uy_pre = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/u" + species + "z_pre.gda"
    x, z, uz_pre = read_2d_fields(pic_info, fname, **kwargs)

    if file_size < required_size:
        ux_post = np.copy(ux)
        uy_post = np.copy(uy)
        uz_post = np.copy(uz)
    else:
        fname = run_dir + "data/u" + species + "x_post.gda"
        x, z, ux_post = read_2d_fields(pic_info, fname, **kwargs)
        fname = run_dir + "data/u" + species + "y_post.gda"
        x, z, uy_post = read_2d_fields(pic_info, fname, **kwargs)
        fname = run_dir + "data/u" + species + "z_post.gda"
        x, z, uz_post = read_2d_fields(pic_info, fname, **kwargs)

    fname = run_dir + "data/ex.gda"
    x, z, ex = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/ey.gda"
    x, z, ey = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/ez.gda"
    x, z, ez = read_2d_fields(pic_info, fname, **kwargs)

    fname = run_dir + "data/bx.gda"
    x, z, bx = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/by.gda"
    x, z, by = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/bz.gda"
    x, z, bz = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/absB.gda"
    x, z, absB = read_2d_fields(pic_info, fname, **kwargs)

    if species == 'e':
        pmass = 1.0
        charge = -1.0
    else:
        pmass = pic_info.mime
        charge = 1.0

    ib2 = 1.0 / absB**2

    smime = math.sqrt(pic_info.mime)
    dx = pic_info.dx_di * smime
    dz = pic_info.dz_di * smime
    dv = dx * dz

    tmpx = (ux_post - ux_pre) * idt
    tmpy = (uy_post - uy_pre) * idt
    tmpz = (uz_post - uz_pre) * idt
    tmpx += vx * np.gradient(ux, dx, axis=1) + vz * np.gradient(ux, dz, axis=0)
    tmpy += vx * np.gradient(uy, dx, axis=1) + vz * np.gradient(uy, dz, axis=0)
    tmpz += vx * np.gradient(uz, dx, axis=1) + vz * np.gradient(uz, dz, axis=0)
    jpolar_x = by * tmpz - bz * tmpy
    jpolar_y = bz * tmpx - bx * tmpz
    jpolar_z = bx * tmpy - by * tmpx
    jpolar_dote = (jpolar_x * ex + jpolar_y * ey + jpolar_z * ez) * ib2 * nrho * pmass

    print("Min and max: %f %f" % (np.min(jpolar_dote), np.max(jpolar_dote)))
    jpolar_dote_tot = np.sum(jpolar_dote) * dv
    print(jpolar_dote_tot)

    return jpolar_dote_tot


def calc_jpolar_dote_continusou_dump_multi(picinfo_fname, run_dir, run_name,
                                           species):
    """Calculate the energy conversion due to polarization drift
    """
    pic_info = read_data_from_json(picinfo_fname)
    ntf = pic_info.ntf
    cts = range(ntf)
    fdir = '../data/jpolar_dote/'
    mkdir_p(fdir)
    ncores = multiprocessing.cpu_count()
    fdata = Parallel(n_jobs=ncores)(delayed(calc_jpolar_dote_continusou_dump)(
            picinfo_fname, ct, run_dir, species) for ct in cts)
    jpolar_dote = np.asarray(fdata)
    fname = fdir + 'jpolar_dote_' + run_name + '_' + species + '.dat'
    jpolar_dote.tofile(fname)


def compare_fluid_particle_energization(pic_info, run_name):
    """Compare fluid based and particle based energization terms
    """
    fdir = '../data/compression/'
    cdata_name = fdir + 'compression_' + run_name + '_e.json'
    cdata_e = read_data_from_json(cdata_name)
    cdata_name = fdir + 'compression_' + run_name + '_i.json'
    cdata_i = read_data_from_json(cdata_name)
    jdote_name = '../data/jdote_data/jdote_' + run_name + '_e.json'
    jdote_e = read_data_from_json(jdote_name)
    jdote_name = '../data/jdote_data/jdote_' + run_name + '_i.json'
    jdote_i = read_data_from_json(jdote_name)

    fdir = '../data/jpolar_dote/'
    fname = fdir + 'jpolar_dote_' + run_name + '_e.dat'
    jpolar_dote_e = np.fromfile(fname)
    jpolar_dote_e[-1] = jpolar_dote_e[-2] # remove boundary spikes
    
    fname = fdir + 'jpolar_dote_' + run_name + '_i.dat'
    jpolar_dote_i = np.fromfile(fname)
    jpolar_dote_i[-1] = jpolar_dote_i[-2] # remove boundary spikes
    
    tratio = pic_info.particle_interval / pic_info.fields_interval
    tfields = pic_info.tfields
    tparticles = pic_info.tparticles

    ntp = pic_info.ntp
    pinterval = pic_info.particle_interval
    nbins = 60
    fdir = '../data/particle_compression/' + run_name + '/'
    fname = fdir + 'hists_e.' + str(pinterval) + '.all'
    fdata = np.fromfile(fname)
    sz, = fdata.shape
    nvar = sz / nbins
    pene_e = np.zeros((nvar, ntp - 1))
    pene_i = np.zeros((nvar, ntp - 1))
    for ct in range(1, ntp):
        tindex = ct * pinterval
        fname = fdir + 'hists_e.' + str(tindex) + '.all'
        fdata = np.fromfile(fname)
        sz, = fdata.shape
        nvar = sz / nbins
        fdata = fdata.reshape((nvar, nbins))
        pene_e[:, ct - 1] = np.sum(fdata, axis=1)

        fname = fdir + 'hists_i.' + str(tindex) + '.all'
        fdata = np.fromfile(fname)
        sz, = fdata.shape
        nvar = sz / nbins
        fdata = fdata.reshape((nvar, nbins))
        pene_i[:, ct - 1] = np.sum(fdata, axis=1)

    label1 = r'$-e\boldsymbol{v}_\parallel\cdot\boldsymbol{E}_\parallel$'
    label2 = r'$-e\boldsymbol{v}_\perp\cdot\boldsymbol{E}_\perp$'
    label3 = r'$-p\nabla\cdot\boldsymbol{v}_E$'
    label4 = r'$-(p_\parallel-p_\perp)b_ib_j\sigma_{ij}$'
    label5 = r'$-\mathbf{P}:\nabla\boldsymbol{v}_E$'
    label6 = r'$m_e(d\boldsymbol{u}_e/dt)\cdot\boldsymbol{v}_E$'
    label7 = label5 + r'$+$' + label6

    w0, h0 = 0.41, 0.11
    xs0, ys0 = 0.08, 0.95 - h0
    vgap, hgap = 0.02, 0.07

    xs, ys = xs0, ys0

    fig = plt.figure(figsize=[10, 14])
    ax11 = fig.add_axes([xs, ys, w0, h0])
    plt.tick_params(labelsize=16)
    fdata = jdote_e.jqnupara_dote
    fdata[0] = fdata[1] # avoiding spikes
    ax11.plot(tfields, fdata, linewidth=2)
    ax11.plot(tparticles[1:], pene_e[0, :], linestyle='None', marker='o',
             color='r', markersize=8)
    ax11.tick_params(axis='x', labelbottom='off')
    ax11.text(0.98, 0.95, label1, color='k', fontsize=16,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='right', verticalalignment='top',
             transform=ax11.transAxes)
    ax11.set_title('Electron', fontdict=font, fontsize=24)

    ys -= h0 + vgap
    ax12 = fig.add_axes([xs, ys, w0, h0])
    plt.tick_params(labelsize=16)
    fdata = jdote_e.jqnuperp_dote
    fdata[0] = fdata[1] # avoiding spikes
    ax12.plot(tfields, jdote_e.jqnuperp_dote, linewidth=2)
    ax12.plot(tparticles[1:], pene_e[1, :], linestyle='None', marker='o',
             color='r', markersize=8)
    ax12.tick_params(axis='x', labelbottom='off')
    ax12.text(0.98, 0.95, label2, color='k', fontsize=16,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='right', verticalalignment='top',
             transform=ax12.transAxes)

    ys -= h0 + vgap
    ax13 = fig.add_axes([xs, ys, w0, h0])
    plt.tick_params(labelsize=16)
    ax13.plot(tfields, cdata_e.pdiv_uperp_usingle_exb, linewidth=2)
    ax13.plot(tparticles[1:], pene_e[3, :], linestyle='None', marker='o',
             color='r', markersize=8)
    ax13.tick_params(axis='x', labelbottom='off')
    ax13.text(0.98, 0.95, label3, color='k', fontsize=16,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='right', verticalalignment='top',
             transform=ax13.transAxes)

    ys -= h0 + vgap
    ax14 = fig.add_axes([xs, ys, w0, h0])
    plt.tick_params(labelsize=16)
    ax14.plot(tfields, cdata_e.pshear_perp_usingle_exb, linewidth=2)
    ax14.plot(tparticles[1:], pene_e[4, :], linestyle='None', marker='o',
             color='r', markersize=8)
    ax14.tick_params(axis='x', labelbottom='off')
    ax14.text(0.98, 0.95, label4, color='k', fontsize=16,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='right', verticalalignment='top',
             transform=ax14.transAxes)

    ys -= h0 + vgap
    ax15 = fig.add_axes([xs, ys, w0, h0])
    plt.tick_params(labelsize=16)
    fdata = cdata_e.pdiv_uperp_usingle_exb + cdata_e.pshear_perp_usingle_exb + \
            jdote_e.jagy_dote
    ax15.plot(tfields, fdata, linewidth=2)
    ax15.plot(tparticles[1:], pene_e[5, :], linestyle='None', marker='o',
             color='r', markersize=8)
    ax15.tick_params(axis='x', labelbottom='off')
    ax15.text(0.98, 0.95, label5, color='k', fontsize=16,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='right', verticalalignment='top',
             transform=ax15.transAxes)

    ys -= h0 + vgap
    ax16 = fig.add_axes([xs, ys, w0, h0])
    plt.tick_params(labelsize=16)
    ax16.plot(tfields, jpolar_dote_e, linewidth=2)
    ax16.plot(tparticles[1:], pene_e[6, :], linestyle='None', marker='o',
             color='r', markersize=8)
    ax16.tick_params(axis='x', labelbottom='off')
    ax16.text(0.98, 0.95, label6, color='k', fontsize=16,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='right', verticalalignment='top',
             transform=ax16.transAxes)

    ys -= h0 + vgap
    ax17 = fig.add_axes([xs, ys, w0, h0])
    plt.tick_params(labelsize=16)
    ax17.plot(tfields, jdote_e.jqnuperp_dote, linewidth=2)
    ax17.plot(tparticles[1:], pene_e[5, :] + pene_e[6, :], linestyle='None',
              marker='o', color='r', markersize=8)
    ax17.text(0.98, 0.95, label7, color='k', fontsize=16,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='right', verticalalignment='top',
             transform=ax17.transAxes)
    ax17.set_xlabel(r'$t\Omega_{ci}$', fontdict=font, fontsize=20)

    label1 = r'$e\boldsymbol{v}_\parallel\cdot\boldsymbol{E}_\parallel$'
    label2 = r'$e\boldsymbol{v}_\perp\cdot\boldsymbol{E}_\perp$'
    label6 = r'$m_i(d\boldsymbol{u}_e/dt)\cdot\boldsymbol{v}_E$'
    label7 = label5 + r'$+$' + label6

    ys = ys0
    xs += hgap + w0

    ax21 = fig.add_axes([xs, ys, w0, h0])
    plt.tick_params(labelsize=16)
    fdata = jdote_i.jqnupara_dote
    fdata[0] = fdata[1] # avoiding spikes
    ax21.plot(tfields, fdata, linewidth=2)
    ax21.plot(tparticles[1:], pene_i[0, :], linestyle='None', marker='o',
             color='r', markersize=8)
    ax21.tick_params(axis='x', labelbottom='off')
    ax21.text(0.98, 0.95, label1, color='k', fontsize=16,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='right', verticalalignment='top',
             transform=ax21.transAxes)
    ax21.set_title('Ion', fontdict=font, fontsize=24)

    ys -= h0 + vgap
    ax22 = fig.add_axes([xs, ys, w0, h0])
    plt.tick_params(labelsize=16)
    fdata = jdote_i.jqnuperp_dote
    fdata[0] = fdata[1] # avoiding spikes
    ax22.plot(tfields, fdata, linewidth=2)
    ax22.plot(tparticles[1:], pene_i[1, :], linestyle='None', marker='o',
             color='r', markersize=8)
    ax22.tick_params(axis='x', labelbottom='off')
    ax22.text(0.98, 0.95, label2, color='k', fontsize=16,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='right', verticalalignment='top',
             transform=ax22.transAxes)

    ys -= h0 + vgap
    ax23 = fig.add_axes([xs, ys, w0, h0])
    plt.tick_params(labelsize=16)
    ax23.plot(tfields, cdata_i.pdiv_uperp_usingle_exb, linewidth=2)
    ax23.plot(tparticles[1:], pene_i[3, :], linestyle='None', marker='o',
             color='r', markersize=8)
    ax23.tick_params(axis='x', labelbottom='off')
    ax23.text(0.98, 0.95, label3, color='k', fontsize=16,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='right', verticalalignment='top',
             transform=ax23.transAxes)

    ys -= h0 + vgap
    ax24 = fig.add_axes([xs, ys, w0, h0])
    plt.tick_params(labelsize=16)
    ax24.plot(tfields, cdata_i.pshear_perp_usingle_exb, linewidth=2)
    ax24.plot(tparticles[1:], pene_i[4, :], linestyle='None', marker='o',
             color='r', markersize=8)
    ax24.tick_params(axis='x', labelbottom='off')
    ax24.text(0.98, 0.95, label4, color='k', fontsize=16,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='right', verticalalignment='top',
             transform=ax24.transAxes)

    ys -= h0 + vgap
    ax25 = fig.add_axes([xs, ys, w0, h0])
    plt.tick_params(labelsize=16)
    fdata = cdata_i.pdiv_uperp_usingle_exb + cdata_i.pshear_perp_usingle_exb + \
            jdote_i.jagy_dote
    ax25.plot(tfields, fdata, linewidth=2)
    ax25.plot(tparticles[1:], pene_i[5, :], linestyle='None', marker='o',
             color='r', markersize=8)
    ax25.tick_params(axis='x', labelbottom='off')
    ax25.text(0.98, 0.95, label5, color='k', fontsize=16,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='right', verticalalignment='top',
             transform=ax25.transAxes)

    ys -= h0 + vgap
    ax26 = fig.add_axes([xs, ys, w0, h0])
    plt.tick_params(labelsize=16)
    ax26.plot(tfields, jpolar_dote_i, linewidth=2)
    ax26.plot(tparticles[1:], pene_i[6, :], linestyle='None', marker='o',
             color='r', markersize=8)
    ax26.tick_params(axis='x', labelbottom='off')
    ax26.text(0.98, 0.95, label6, color='k', fontsize=16,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='right', verticalalignment='top',
             transform=ax26.transAxes)

    ys -= h0 + vgap
    ax27 = fig.add_axes([xs, ys, w0, h0])
    plt.tick_params(labelsize=16)
    ax27.plot(tfields, jdote_i.jqnuperp_dote, linewidth=2)
    ax27.plot(tparticles[1:], pene_i[5, :] + pene_i[6, :], linestyle='None',
              marker='o', color='r', markersize=8)
    ax27.text(0.98, 0.95, label7, color='k', fontsize=16,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='right', verticalalignment='top',
             transform=ax27.transAxes)
    ax27.set_xlabel(r'$t\Omega_{ci}$', fontdict=font, fontsize=20)

    fdir = '../img/fluid_particle_compression/'
    mkdir_p(fdir)
    fname = fdir + 'fluid_particle_compression_' + run_name + '.eps'
    fig.savefig(fname)

    plt.show()

    # hist_de_para_i = fdata[0, :]
    # hist_de_perp_i = fdata[1, :]
    # hist_pdivv_i = fdata[2, :]
    # hist_pdiv_vperp_i = fdata[3, :]
    # hist_pshear_i = fdata[4, :]
    # hist_ptensor_dv_i = fdata[5, :]
    # hist_de_dudt_i = fdata[6, :]


def find_min_max_energization_terms(run_name, species):
    """Find the minimum and maximum value of energization terms
    """
    fdir = '../data/compression/'
    cdata_name = fdir + 'compression_' + run_name + '_' + species + '.json'
    cdata = read_data_from_json(cdata_name)
    jdote_name = '../data/jdote_data/jdote_' + run_name + '_' + species + '.json'
    jdata = read_data_from_json(jdote_name)
    ecov_max = max(np.max(cdata.pdiv_uperp_usingle_exb),
                   np.max(cdata.pdiv_upara_usingle),
                   np.max(cdata.pdiv_usingle_exb),
                   np.max(cdata.pshear_single_exb),
                   np.max(cdata.pshear_para_usingle_exb),
                   np.max(cdata.pshear_perp_usingle_exb),
                   np.max(jdata.jqnuperp_dote),
                   np.max(jdata.jqnupara_dote),
                   np.max(jdata.jpolar_dote),
                   np.max(jdata.jagy_dote))
    ecov_min = min(np.min(cdata.pdiv_uperp_usingle_exb),
                   np.min(cdata.pdiv_upara_usingle),
                   np.min(cdata.pdiv_usingle_exb),
                   np.min(cdata.pshear_single_exb),
                   np.min(cdata.pshear_para_usingle_exb),
                   np.min(cdata.pshear_perp_usingle_exb),
                   np.min(jdata.jqnuperp_dote),
                   np.min(jdata.jqnupara_dote),
                   np.min(jdata.jpolar_dote),
                   np.min(jdata.jagy_dote))
    if ecov_max > 0:
        ecov_max *= 1.2
    else:
        ecov_max *= 0.8

    if ecov_min < 0:
        ecov_min *= 1.2
    else:
        ecov_min *= 0.8

    return (ecov_min, ecov_max)


def plot_energization_terms(pic_info, root_dir, run_name, species, current_time,
                            ecov_min, ecov_max):
    """
    Plot energization terms

    Args:
        pic_info: namedtuple for the PIC simulation information.
        root_dir: simulation root directory
        species: 'e' for electrons, 'i' for ions.
        current_time: current time frame.
        ecov_min: minimum energization
        ecov_max: maximum energization
    """
    print("Time frame: %d" % current_time)
    if species == 'e':
        charge = -1.0
        pmass = 1.0
    else:
        charge = 1.0
        pmass = pic_info.mime
    smime = math.sqrt(pic_info.mime)
    dx = pic_info.dx_di * smime
    dz = pic_info.dz_di * smime
    kwargs = {"current_time": current_time,
              "xl": 0, "xr": 200, "zb": -50, "zt": 50
              }
    fname = root_dir + "data/v" + species + "x.gda"
    x, z, vx = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/v" + species + "y.gda"
    x, z, vy = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/v" + species + "z.gda"
    x, z, vz = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/u" + species + "x.gda"
    x, z, ux = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/u" + species + "y.gda"
    x, z, uy = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/u" + species + "z.gda"
    x, z, uz = read_2d_fields(pic_info, fname, **kwargs)
    dux_dt = vx * np.gradient(ux, dx, axis=1) + \
             vz * np.gradient(ux, dz, axis=0)
    duy_dt = vx * np.gradient(uy, dx, axis=1) + \
             vz * np.gradient(uy, dz, axis=0)
    duz_dt = vx * np.gradient(uz, dx, axis=1) + \
             vz * np.gradient(uz, dz, axis=0)

    fname = root_dir + "data/bx.gda"
    x, z, bx = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/by.gda"
    x, z, by = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/bz.gda"
    x, z, bz = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/absB.gda"
    x, z, absB = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/ex.gda"
    x, z, ex = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/ey.gda"
    x, z, ey = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/ez.gda"
    x, z, ez = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/n" + species + ".gda"
    x, z, nrho = read_2d_fields(pic_info, fname, **kwargs)

    ib2 = 1.0 / absB**2
    vdotB = vx * bx + vy * by + vz * bz
    jdote = charge * nrho * (vx * ex + vy * ey + vz * ez)
    jpara_dote = charge * nrho * vdotB * (bx * ex + by * ey + bz * ez) * ib2
    jperp_dote = jdote - jpara_dote
    vexb_x = (ey * bz - ez * by) * ib2
    vexb_y = (ez * bx - ex * bz) * ib2
    vexb_z = (ex * by - ey * bx) * ib2
    del vx, vy, vz
    del ex, ey, ez, vdotB, absB

    fname = root_dir + "data/p" + species + "-xx.gda"
    x, z, pxx = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/p" + species + "-xy.gda"
    x, z, pxy = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/p" + species + "-xz.gda"
    x, z, pxz = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/p" + species + "-yx.gda"
    x, z, pyx = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/p" + species + "-yy.gda"
    x, z, pyy = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/p" + species + "-yz.gda"
    x, z, pyz = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/p" + species + "-zx.gda"
    x, z, pzx = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/p" + species + "-zy.gda"
    x, z, pzy = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/p" + species + "-zz.gda"
    x, z, pzz = read_2d_fields(pic_info, fname, **kwargs)

    pscalar = (pxx + pyy + pzz) / 3.0
    ppara = (pxx * bx**2 + pyy * by**2 + pzz * bz**2 +
             (pxy + pyx) * bx * by + (pxz + pzx) * bx * bz +
             (pyz + pzy) * by * bz) * ib2
    pperp = 0.5 * (3 * pscalar - ppara)

    dvperpx_dx = np.gradient(vexb_x, dx, axis=1)
    dvperpy_dx = np.gradient(vexb_y, dx, axis=1)
    dvperpz_dx = np.gradient(vexb_z, dx, axis=1)
    dvperpx_dz = np.gradient(vexb_x, dz, axis=0)
    dvperpy_dz = np.gradient(vexb_y, dz, axis=0)
    dvperpz_dz = np.gradient(vexb_z, dz, axis=0)
    div_vperp = dvperpx_dx + dvperpz_dz
    pdiv_vperp = -pscalar * div_vperp
    pshear = (dvperpx_dx - (1./3.) * div_vperp) * bx**2 + \
            (-(1./3.) * div_vperp) * by**2 + \
            (dvperpz_dz - (1./3.) * div_vperp) * bz**2 + \
            dvperpy_dx * bx * by + \
            (dvperpx_dz + dvperpz_dx) * bx * bz + dvperpy_dz * by * bz
    pshear *= (pperp - ppara) * ib2
    ptensor_dv = -(pxx * dvperpx_dx + pyx * dvperpy_dx + pzx * dvperpz_dx)
    ptensor_dv -= pxz * dvperpx_dz + pyz * dvperpy_dz + pzz * dvperpz_dz
    del pscalar, div_vperp
    del dvperpx_dx, dvperpy_dx, dvperpz_dx
    del dvperpx_dz, dvperpy_dz, dvperpz_dz

    div_pperp_vperp = np.gradient(pperp * vexb_x, dx, axis=1) + \
                      np.gradient(pperp * vexb_z, dz, axis=0)
    div_ptensor_vperp = np.gradient(pxx * vexb_x + pyx * vexb_y +
                                    pzx * vexb_z, dx, axis=1) + \
                        np.gradient(pxz * vexb_x + pyz * vexb_y +
                                    pzz * vexb_z, dz, axis=0)
    div_ptensor_dot_vperp = (np.gradient(pxx, dx, axis=1) +
                             np.gradient(pxz, dz, axis=0)) * vexb_x + \
                            (np.gradient(pyx, dx, axis=1) +
                             np.gradient(pyz, dz, axis=0)) * vexb_y + \
                            (np.gradient(pzx, dx, axis=1) +
                             np.gradient(pzz, dz, axis=0)) * vexb_z
    pre_diff = ppara - pperp
    div_pmag_dot_vperp = (np.gradient(pperp + pre_diff * bx**2 * ib2, dx, axis=1) +
                          np.gradient(pre_diff * bx * bz * ib2, dz, axis=0)) * vexb_x + \
                         (np.gradient(pre_diff * bx * by * ib2, dx, axis=1) +
                          np.gradient(pre_diff * by * bz * ib2, dz, axis=0)) * vexb_y + \
                         (np.gradient(pre_diff * bx * bz * ib2, dx, axis=1) +
                          np.gradient(pperp + pre_diff * bz**2 * ib2, dz, axis=0)) * vexb_z
    jagy_dote = div_ptensor_dot_vperp - div_pmag_dot_vperp
    del pxx, pxy, pxz, pyx, pyy, pyz, pzx, pzy, pzz
    del bx, by, bz, ib2, pre_diff

    nx = pic_info.nx
    nz = pic_info.nz
    fname = run_dir + "data/u" + species + "x_post.gda"
    statinfo = os.stat(fname)
    file_size = statinfo.st_size
    required_size = (current_time + 1) * nx * nz * 4

    if file_size < required_size:
        idt = 1.0 / pic_info.dtwpe
    else:
        idt = 0.5 / pic_info.dtwpe

    fname = run_dir + "data/u" + species + "x_pre.gda"
    x, z, ux_pre = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/u" + species + "y_pre.gda"
    x, z, uy_pre = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/u" + species + "z_pre.gda"
    x, z, uz_pre = read_2d_fields(pic_info, fname, **kwargs)

    if file_size < required_size:
        ux_post = np.copy(ux)
        uy_post = np.copy(uy)
        uz_post = np.copy(uz)
    else:
        fname = run_dir + "data/u" + species + "x_post.gda"
        x, z, ux_post = read_2d_fields(pic_info, fname, **kwargs)
        fname = run_dir + "data/u" + species + "y_post.gda"
        x, z, uy_post = read_2d_fields(pic_info, fname, **kwargs)
        fname = run_dir + "data/u" + species + "z_post.gda"
        x, z, uz_post = read_2d_fields(pic_info, fname, **kwargs)

    dux_dt += (ux_post - ux_pre) * idt
    duy_dt += (uy_post - uy_pre) * idt
    duz_dt += (uz_post - uz_pre) * idt

    jpolar_dote = pmass * nrho * (dux_dt * vexb_x + duy_dt * vexb_y + duz_dt * vexb_z)

    del ux, uy, uz, nrho
    del ux_pre, uy_pre, uz_pre
    del ux_post, uy_post, uz_post
    del vexb_x, vexb_y, vexb_z

    # fname = root_dir + "data/Ay.gda"
    # x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)

    wpe_wce = pic_info.dtwce / pic_info.dtwpe
    va = wpe_wce / math.sqrt(pic_info.mime)  # Alfven speed of inflow region
    b0 = pic_info.b0
    norm = va**2 * b0

    jpara_dote /= norm
    jperp_dote /= norm
    jdote /= norm
    pdiv_vperp /= norm
    pshear /= norm
    jpolar_dote /= norm
    jagy_dote /= norm
    ptensor_dv /= norm
    div_ptensor_dot_vperp /= norm
    div_pmag_dot_vperp /= norm
    div_ptensor_vperp /= norm
    div_pperp_vperp /= norm

    sigma = 3
    jpara_dote = gaussian_filter(jpara_dote, sigma)
    jperp_dote = gaussian_filter(jperp_dote, sigma)
    jdote = gaussian_filter(jdote, sigma)
    pdiv_vperp = gaussian_filter(pdiv_vperp, sigma)
    pshear = gaussian_filter(pshear, sigma)
    jpolar_dote = gaussian_filter(jpolar_dote, sigma)
    jagy_dote = gaussian_filter(jagy_dote, sigma)
    ptensor_dv = gaussian_filter(ptensor_dv, sigma)
    div_ptensor_dot_vperp = gaussian_filter(div_ptensor_dot_vperp, sigma)
    div_pmag_dot_vperp = gaussian_filter(div_pmag_dot_vperp, sigma)
    div_ptensor_vperp = gaussian_filter(div_ptensor_vperp, sigma)
    div_pperp_vperp = gaussian_filter(div_pperp_vperp, sigma)

    fname = root_dir + "data/Ay.gda"
    x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)

    xmin, xmax = 0, pic_info.lx_di
    zmin, zmax = -0.5 * pic_info.lz_di, 0.5 * pic_info.lz_di

    # w0, h0 = 0.41, 0.11
    w0, h0 = 0.27, 0.16
    xs0, ys0 = 0.08, 0.95 - h0
    vgap, hgap = 0.02, 0.04

    vmax1 = 1.0E-1
    vmin1 = -vmax1
    dv = dx * dz
    def plot_one_energization(fdata, ax, text, text_color, label_bottom='on',
                              label_left='on', ylabel=False, ay_color='k'):
        plt.tick_params(labelsize=16)
        ax.imshow(fdata, vmin=vmin1, vmax=vmax1, cmap=plt.cm.seismic,
                  extent=[xmin, xmax, zmin, zmax], aspect='auto',
                  origin='lower', interpolation='bicubic')
        ax.tick_params(axis='x', labelbottom=label_bottom)
        ax.tick_params(axis='y', labelleft=label_left)
        if ylabel:
            ax.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
        ax.contour(x, z, Ay, colors=ay_color, linewidths=0.5)
        ax.text(0.02, 0.85, text, color=text_color, fontsize=16,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)

    def plot_one_column(fig, xstart, ystart, fdata, texts, ylabel=False):
        xs, ys = xstart, ystart
        colors = colors_Set1_9
        axs = []
        for i in range(4):
            ax1 = fig.add_axes([xs, ys, w0, h0])
            plot_one_energization(fdata[i], ax1, texts[i], colors[i],
                                  label_bottom='off', ylabel=ylabel)
            axs.append(ax1)
            ys -= h0 + vgap
        ax5 = fig.add_axes([xs, ys, w0, h0])
        ax5.set_prop_cycle('color', colors)
        for i in range(4):
            fdata_cumsum = np.cumsum(np.sum(fdata[i], axis=0)) * dv
            ax5.plot(x, fdata_cumsum, linewidth=2)
        ax5.plot([xmin, xmax], [0, 0], color='k', linewidth=1, linestyle='--')
        ax5.tick_params(labelsize=16)
        ax5.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
        if ylabel:
            ax5.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
        ax5.set_ylim([ecov_min/norm, ecov_max/norm])
        axs.append(ax5)
        return axs

    fig = plt.figure(figsize=[15, 11])
    xs, ys = xs0, ys0
    label1 = r'$\boldsymbol{j}_{s\perp}\cdot\boldsymbol{E}_\perp$'
    label2 = r'$-p_s\nabla\cdot\boldsymbol{v}_E$'
    label3 = r'$-(p_{s\parallel} - p_{s\perp})b_ib_j\sigma_{ij}$'
    label4 = r'$-\mathbf{P}:\nabla\boldsymbol{v}_E$'
    fdata = [jperp_dote, pdiv_vperp, pshear, ptensor_dv]
    texts = [label1, label2, label3, label4]
    axs1 = plot_one_column(fig, xs, ys, fdata, texts, ylabel=True)

    xs, ys = xs0 + hgap + w0, ys0
    label1 = r'$\boldsymbol{j}_{s}\cdot\boldsymbol{E}$'
    label2 = r'$(\nabla\cdot\mathbf{P})\cdot\boldsymbol{v}_E$'
    label3 = r'$(\nabla\cdot\mathbf{P}_\text{mag})\cdot\boldsymbol{v}_E$'
    label4 = r'$n_sm_s(d\boldsymbol{u}_s/dt)\cdot\boldsymbol{v}_E$'
    fdata = [jdote, div_ptensor_dot_vperp, div_pmag_dot_vperp, jpolar_dote]
    texts = [label1, label2, label3, label4]
    axs2 = plot_one_column(fig, xs, ys, fdata, texts, ylabel=False)

    xs, ys = xs0 + 2 * (hgap + w0), ys0
    label1 = r'$\boldsymbol{j}_{s\parallel}\cdot\boldsymbol{E}_\parallel$'
    label2 = r'$\nabla\cdot(\mathbf{P}\cdot\boldsymbol{v}_E)$'
    label3 = r'$\boldsymbol{j}_\text{agy}\cdot\boldsymbol{E}$'
    label4 = r'$\nabla\cdot(p_\perp\boldsymbol{v}_E)$'
    fdata = [jpara_dote, div_ptensor_vperp, jagy_dote, div_pperp_vperp]
    texts = [label1, label2, label3, label4]
    axs3 = plot_one_column(fig, xs, ys, fdata, texts, ylabel=False)

    t_wci = current_time * pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(t_wci) + '\Omega_{ci}^{-1}$'
    axs2[0].set_title(title, fontdict=font, fontsize=24)

    fdir = '../img/energization_terms_ptl/' + run_name + '/'
    mkdir_p(fdir)
    fname = fdir + 'energization_terms_' + str(current_time) + '_' + species + '.jpg'
    fig.savefig(fname, dpi=200)


def generate_energization_terms_table():
    """Generate a table of energization terms

    Args:
        pic_info: namedtuple for the PIC simulation information
    """
    run_names = ['mime25_beta002_guide00_frequent_dump', 'mime25_beta008_guide00_frequent_dump',
                 'mime25_beta032_guide00_frequent_dump', 'mime25_beta002_guide02_frequent_dump',
                 'mime25_beta002_guide05_frequent_dump', 'mime25_beta002_guide10_frequent_dump']
    nrun = len(run_names)
    nvar = 6
    ene_terms = np.zeros((nrun, nvar * 2))
    for irun, run_name in enumerate(run_names):
        picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        tfields = pic_info.tfields
        tenergy = pic_info.tenergy
        ct = find_closest(tfields, 600)
        cte = find_closest(tenergy, 600)
        dv = pic_info.dx_di * pic_info.dz_di * pic_info.mime
        dene_e = pic_info.kene_e[cte] - pic_info.kene_e[0]
        dene_i = pic_info.kene_i[cte] - pic_info.kene_i[0]
        fdir = '../data/compression/'
        cdata_name = fdir + 'compression_' + run_name + '_e.json'
        cdata_e = read_data_from_json(cdata_name)
        cdata_name = fdir + 'compression_' + run_name + '_i.json'
        cdata_i = read_data_from_json(cdata_name)
        jdote_name = '../data/jdote_data/jdote_' + run_name + '_e.json'
        jdote_e = read_data_from_json(jdote_name)
        jdote_name = '../data/jdote_data/jdote_' + run_name + '_i.json'
        jdote_i = read_data_from_json(jdote_name)
        ene_terms[irun, 0] = jdote_e.jqnupara_dote_int[ct]
        ene_terms[irun, 1] = jdote_e.jqnuperp_dote_int[ct]
        ene_terms[irun, 2] = cdata_e.pdiv_uperp_usingle_exb_cum[ct]
        ene_terms[irun, 3] = cdata_e.pshear_perp_usingle_exb_cum[ct]
        ene_terms[irun, 4] = jdote_e.jpolar_dote_int[ct]
        ene_terms[irun, 5] = jdote_e.jagy_dote_int[ct]
        ene_terms[irun, 6] = jdote_i.jqnupara_dote_int[ct]
        ene_terms[irun, 7] = jdote_i.jqnuperp_dote_int[ct]
        ene_terms[irun, 8] = cdata_i.pdiv_uperp_usingle_exb_cum[ct]
        ene_terms[irun, 9] = cdata_i.pshear_perp_usingle_exb_cum[ct]
        ene_terms[irun, 10] = jdote_i.jpolar_dote_int[ct]
        ene_terms[irun, 11] = jdote_i.jagy_dote_int[ct]
        ene_terms[irun, :nvar] /= dene_e
        ene_terms[irun, nvar:] /= dene_i

    for ivar in range(nvar * 2):
        print("%6.2f &" * nrun % tuple(ene_terms[:, ivar]))


def calc_vexb(pic_info, root_dir, current_time):
    """Calculate ExB drift velocity

    Args:
        pic_info: namedtuple for the PIC simulation information.
        root_dir: simulation root directory
        current_time: current time frame.
    """
    print("Time frame: %d" % current_time)
    kwargs = {"current_time": current_time,
              "xl": 0, "xr": 200, "zb": -50, "zt": 50
              }
    fname = root_dir + "data/ex.gda"
    x, z, ex = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/ey.gda"
    x, z, ey = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/ez.gda"
    x, z, ez = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/bx.gda"
    x, z, bx = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/by.gda"
    x, z, by = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/bz.gda"
    x, z, bz = read_2d_fields(pic_info, fname, **kwargs)
    
    ib2 = 1.0 / (bx**2 + by**2 + bz**2)
    vx = (ey * bz - ez * by) * ib2
    vy = (ez * bx - ex * bz) * ib2
    vz = (ex * by - ey * bx) * ib2
    sigma = 2
    vx = gaussian_filter(vx, sigma)
    vy = gaussian_filter(vy, sigma)
    vz = gaussian_filter(vz, sigma)

    return (x, z, vx, vy, vz)


def plot_nrho_velocity(pic_info, root_dir, run_name, current_time):
    """
    Plot number density and velocity

    Args:
        pic_info: namedtuple for the PIC simulation information.
        root_dir: simulation root directory
        current_time: current time frame.
    """
    print("Time frame: %d" % current_time)
    if species == 'e':
        charge = -1.0
        pmass = 1.0
    else:
        charge = 1.0
        pmass = pic_info.mime
    smime = math.sqrt(pic_info.mime)
    dx = pic_info.dx_di * smime
    dz = pic_info.dz_di * smime
    kwargs = {"current_time": current_time,
              "xl": 0, "xr": 200, "zb": -50, "zt": 50
              }
    x, z, vx, vy, vz = calc_vexb(pic_info, root_dir, current_time)
    fname = root_dir + "data/ne.gda"
    x, z, ne = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/ni.gda"
    x, z, ni = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/Ay.gda"
    x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
    
    wpe_wce = pic_info.dtwce / pic_info.dtwpe
    va = wpe_wce / math.sqrt(pic_info.mime)  # Alfven speed of inflow region
    vx /= va # normalize with Alfven speed
    vy /= va
    vz /= va

    xmin, xmax = np.min(x), np.max(x)
    zmin, zmax = np.min(z), np.max(z)

    # w0, h0 = 0.41, 0.11
    w0, h0 = 0.73, 0.16
    xs0, ys0 = 0.14, 0.985 - h0
    vgap, hgap = 0.03, 0.04

    def plot_one_field(fdata, ax, text, text_color, label_bottom='on',
                       label_left='on', ylabel=False, vmin=0, vmax=10,
                       colormap=plt.cm.seismic, xs=xs0, ys=ys0, ay_color='k'):
        plt.tick_params(labelsize=16)
        p1 = ax.imshow(fdata, vmin=vmin, vmax=vmax, cmap=colormap,
                       extent=[xmin, xmax, zmin, zmax], aspect='auto',
                       origin='lower', interpolation='bicubic')
        ax.tick_params(axis='x', labelbottom=label_bottom)
        ax.tick_params(axis='y', labelleft=label_left)
        if ylabel:
            ax.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
        ax.contour(x, z, Ay, colors=ay_color, linewidths=0.5)
        ax.text(0.02, 0.85, text, color=text_color, fontsize=20,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)
        xs1 = xs + w0 * 1.02
        w1 = w0 * 0.04
        cax = fig.add_axes([xs1, ys, w1, h0])
        cbar = fig.colorbar(p1, cax=cax)
        cbar.ax.tick_params(labelsize=16)
        return cbar

    fig = plt.figure(figsize=[7, 10])
    xs, ys = xs0, ys0
    ax1 = fig.add_axes([xs, ys, w0, h0])
    text1 = r'$n_e$'
    print("min and max of electron density: %f %f" % (np.min(ne), np.max(ne)))
    nmin, nmax = 0.5, 3.0
    cbar1 = plot_one_field(ne, ax1, text1, 'w', label_bottom='off',
                           label_left='on', ylabel=True, vmin=nmin, vmax=nmax,
                           colormap=plt.cm.viridis, xs=xs, ys=ys, ay_color='w')
    cbar1.set_ticks(np.arange(nmin, nmax + 0.5, 0.5))

    ys -= h0 + vgap
    ax2 = fig.add_axes([xs, ys, w0, h0])
    text2 = r'$n_i$'
    print("min and max of ion density: %f %f" % (np.min(ne), np.max(ne)))
    nmin, nmax = 0.5, 3.0
    cbar2 = plot_one_field(ni, ax2, text2, 'w', label_bottom='off',
                           label_left='on', ylabel=True, vmin=nmin, vmax=nmax,
                           colormap=plt.cm.viridis, xs=xs, ys=ys, ay_color='w')
    cbar2.set_ticks(np.arange(nmin, nmax + 0.5, 0.5))

    ys -= h0 + vgap
    ax3 = fig.add_axes([xs, ys, w0, h0])
    vmin, vmax = -1.0, 1.0
    text3 = r'$v_{Ex}$'
    print("min and max of vx: %f %f" % (np.min(vx), np.max(vx)))
    cbar3 = plot_one_field(vx, ax3, text3, 'k', label_bottom='off',
                           label_left='on', ylabel=True, vmin=vmin, vmax=vmax,
                           colormap=plt.cm.seismic, xs=xs, ys=ys)
    cbar3.set_ticks(np.arange(vmin, vmax + 0.5, 0.5))

    ys -= h0 + vgap
    ax4 = fig.add_axes([xs, ys, w0, h0])
    text4 = r'$v_{Ey}$'
    print("min and max of vy: %f %f" % (np.min(vy), np.max(vy)))
    cbar4 = plot_one_field(vy, ax4, text4, 'k', label_bottom='off',
                           label_left='on', ylabel=True, vmin=vmin, vmax=vmax,
                           colormap=plt.cm.seismic, xs=xs, ys=ys)
    cbar4.set_ticks(np.arange(vmin, vmax + 0.5, 0.5))

    ys -= h0 + vgap
    ax5 = fig.add_axes([xs, ys, w0, h0])
    text5 = r'$v_{Ez}$'
    print("min and max of vz: %f %f" % (np.min(vz), np.max(vz)))
    cbar5 = plot_one_field(vz, ax5, text5, 'k', label_bottom='on',
                           label_left='on', ylabel=True, vmin=vmin, vmax=vmax,
                           colormap=plt.cm.seismic, xs=xs, ys=ys)
    ax5.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    cbar5.set_ticks(np.arange(vmin, vmax + 0.5, 0.5))

    fdir = '../img/nrho_velocity/' + run_name + '/'
    mkdir_p(fdir)
    fname = fdir + 'nrho_vel_' + str(current_time) + '.jpg'
    fig.savefig(fname, dpi=200)
    plt.close()
    # plt.show()


def calc_ppara_pperp_pscalar(pic_info, root_dir, current_time,
                             species='e'):
    """Calculate parallel and perpendicular pressure and scalar pressure

    Args:
        pic_info: namedtuple for the PIC simulation information.
        root_dir: simulation root directory
        current_time: current time frame.
    """
    print("Time frame: %d" % current_time)
    kwargs = {"current_time": current_time,
              "xl": 0, "xr": 200, "zb": -50, "zt": 50
              }
    fname = root_dir + "data/bx.gda"
    x, z, bx = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/by.gda"
    x, z, by = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/bz.gda"
    x, z, bz = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/p" + species + "-xx.gda"
    x, z, pxx = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/p" + species + "-yy.gda"
    x, z, pyy = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/p" + species + "-zz.gda"
    x, z, pzz = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/p" + species + "-xy.gda"
    x, z, pxy = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/p" + species + "-xz.gda"
    x, z, pxz = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/p" + species + "-yz.gda"
    x, z, pyz = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/p" + species + "-yx.gda"
    x, z, pyx = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/p" + species + "-zx.gda"
    x, z, pzx = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/p" + species + "-zy.gda"
    x, z, pzy = read_2d_fields(pic_info, fname, **kwargs)

    pscalar = (pxx + pyy + pzz) / 3.0
    ppara = pxx * bx**2 + pyy * by**2 + pzz * bz**2 + \
            (pxy + pyx) * bx * by + (pxz + pzx) * bx * bz + \
            (pyz + pzy) * by * bz
    ppara /= bx**2 + by**2 + bz**2
    pperp = (pscalar * 3 - ppara) * 0.5

    return (ppara, pperp, pscalar)


def plot_compresion_of_vexb(pic_info, root_dir, run_name, current_time,
                            species='e'):
    """Plot the compression of the ExB drift velocity

    Args:
        pic_info: namedtuple for the PIC simulation information
        root_dir: simulation root directory
        run_name: simulation run name
        current_time: current time frame
    """
    print("Time frame: %d" % current_time)
    if species == 'e':
        charge = -1.0
        pmass = 1.0
    else:
        charge = 1.0
        pmass = pic_info.mime
    smime = math.sqrt(pic_info.mime)
    dx = pic_info.dx_di * smime
    dz = pic_info.dz_di * smime
    kwargs = {"current_time": current_time,
              "xl": 0, "xr": 200, "zb": -50, "zt": 50
              }
    x, z, vx, vy, vz = calc_vexb(pic_info, root_dir, current_time)
    ppara, pperp, pscalar = calc_ppara_pperp_pscalar(
            pic_info, root_dir, current_time, species)
    fname = root_dir + "data/Ay.gda"
    x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
    xmin, xmax = np.min(x), np.max(x)
    zmin, zmax = np.min(z), np.max(z)
    div_vx = np.gradient(vx, dx, axis=1)
    div_vz = np.gradient(vz, dz, axis=0)
    # w0, h0 = 0.41, 0.11
    w0, h0 = 0.70, 0.26
    xs0, ys0 = 0.14, 0.96 - h0
    vgap, hgap = 0.03, 0.04

    def plot_one_field(fdata, ax, text, text_color, label_bottom='on',
                       label_left='on', ylabel=False, vmin=0, vmax=10,
                       colormap=plt.cm.seismic, xs=xs0, ys=ys0, ay_color='k'):
        plt.tick_params(labelsize=16)
        p1 = ax.imshow(fdata, vmin=vmin, vmax=vmax, cmap=colormap,
                       extent=[xmin, xmax, zmin, zmax], aspect='auto',
                       origin='lower', interpolation='bicubic')
        ax.tick_params(axis='x', labelbottom=label_bottom)
        ax.tick_params(axis='y', labelleft=label_left)
        if ylabel:
            ax.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
        ax.contour(x, z, Ay, colors=ay_color, linewidths=0.5)
        ax.text(0.02, 0.85, text, color=text_color, fontsize=20,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)
        xs1 = xs + w0 * 1.02
        w1 = w0 * 0.04
        cax = fig.add_axes([xs1, ys, w1, h0])
        cbar = fig.colorbar(p1, cax=cax)
        cbar.ax.tick_params(labelsize=16)
        return cbar

    dv = dx * dz
    pdiv_vx = -pscalar * div_vx
    pdiv_vz = -pscalar * div_vz
    pdiv_vx_cumsum = np.cumsum(np.sum(pdiv_vx, axis=0)) * dv
    pdiv_vz_cumsum = np.cumsum(np.sum(pdiv_vz, axis=0)) * dv

    fig = plt.figure(figsize=[7, 7])
    xs, ys = xs0, ys0
    ax1 = fig.add_axes([xs, ys, w0, h0])
    text1 = r'$-p\partial v_x/\partial x$'
    print("min and max of pdiv_vx: %f %f" % (np.min(pdiv_vx), np.max(pdiv_vx)))
    nmin, nmax = -1E-3, 1E-3
    cbar1 = plot_one_field(pdiv_vx, ax1, text1, 'r', label_bottom='off',
                           label_left='on', ylabel=True, vmin=nmin, vmax=nmax,
                           colormap=plt.cm.seismic, xs=xs, ys=ys, ay_color='k')
    cbar1.set_ticks(np.arange(nmin, nmax + 1E-3, 1E-3))

    ys -= h0 + vgap
    ax2 = fig.add_axes([xs, ys, w0, h0])
    text2 = r'$-p\partial v_z/\partial z$'
    print("min and max of pdiv_vz: %f %f" % (np.min(pdiv_vz), np.max(pdiv_vz)))
    cbar2 = plot_one_field(pdiv_vz, ax2, text2, 'b', label_bottom='off',
                           label_left='on', ylabel=True, vmin=nmin, vmax=nmax,
                           colormap=plt.cm.seismic, xs=xs, ys=ys, ay_color='k')
    cbar2.set_ticks(np.arange(nmin, nmax + 1E-3, 1E-3))

    ys -= h0 + vgap
    ax3 = fig.add_axes([xs, ys, w0, h0])
    ax3.plot(x, pdiv_vx_cumsum, linewidth=2, color='r')
    ax3.plot(x, pdiv_vz_cumsum, linewidth=2, color='b')
    ax3.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    ax3.tick_params(labelsize=16)

    fdir = '../img/compression_of_vexb/' + run_name + '/'
    mkdir_p(fdir)
    fname = fdir + 'compression_of_vexb_' + str(current_time)
    fname += '_' + species + '.jpg'
    fig.savefig(fname, dpi=200)
    # plt.close()
    # plt.show()


def calc_bbsigma(pic_info, root_dir, current_time):
    """Calculate b_ib_j\sigma_{ij} using ExB drift velocity

    Args:
        pic_info: namedtuple for the PIC simulation information.
        root_dir: simulation root directory
        current_time: current time frame.
    """
    print("Time frame: %d" % current_time)
    kwargs = {"current_time": current_time,
              "xl": 0, "xr": 200, "zb": -50, "zt": 50
              }
    fname = root_dir + "data/ex.gda"
    x, z, ex = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/ey.gda"
    x, z, ey = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/ez.gda"
    x, z, ez = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/bx.gda"
    x, z, bx = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/by.gda"
    x, z, by = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/bz.gda"
    x, z, bz = read_2d_fields(pic_info, fname, **kwargs)
    
    smime = math.sqrt(pic_info.mime)
    dx = pic_info.dx_di * smime
    dz = pic_info.dz_di * smime
    
    ib2 = 1.0 / (bx**2 + by**2 + bz**2)
    vx = (ey * bz - ez * by) * ib2
    vy = (ez * bx - ex * bz) * ib2
    vz = (ex * by - ey * bx) * ib2
    sigma = 2
    vx = gaussian_filter(vx, sigma)
    vy = gaussian_filter(vy, sigma)
    vz = gaussian_filter(vz, sigma)

    divv = np.gradient(vx, dx, axis=1) + np.gradient(vz, dz, axis=0)
    bbsigmaxx = (np.gradient(vx, dx, axis=1) - divv / 3.0) * bx**2 * ib2
    bbsigmayy = (-divv / 3.0) * by**2 * ib2
    bbsigmazz = (np.gradient(vz, dz, axis=0) - divv / 3.0) * bz**2 * ib2
    bbsigmaxy = np.gradient(vy, dx, axis=1) * bx * by * ib2
    bbsigmaxz = ((np.gradient(vz, dx, axis=1) +
                  np.gradient(vx, dz, axis=0))) * bx * bz * ib2
    bbsigmayz = np.gradient(vy, dz, axis=0) * by * bz * ib2

    return (bbsigmaxx, bbsigmayy, bbsigmazz, bbsigmaxy, bbsigmaxz, bbsigmayz)


def plot_shear_of_vexb(pic_info, root_dir, run_name, current_time,
                            species='e'):
    """Plot the shear of the ExB drift velocity

    Args:
        pic_info: namedtuple for the PIC simulation information
        root_dir: simulation root directory
        run_name: simulation run name
        current_time: current time frame
    """
    print("Time frame: %d" % current_time)
    if species == 'e':
        charge = -1.0
        pmass = 1.0
    else:
        charge = 1.0
        pmass = pic_info.mime
    smime = math.sqrt(pic_info.mime)
    dx = pic_info.dx_di * smime
    dz = pic_info.dz_di * smime
    kwargs = {"current_time": current_time,
              "xl": 0, "xr": 200, "zb": -50, "zt": 50
              }
    bbsigmaxx, bbsigmayy, bbsigmazz, bbsigmaxy, bbsigmaxz, bbsigmayz = \
            calc_bbsigma(pic_info, root_dir, current_time)
    ppara, pperp, pscalar = calc_ppara_pperp_pscalar(
            pic_info, root_dir, current_time, species)
    fname = root_dir + "data/Ay.gda"
    x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
    xmin, xmax = np.min(x), np.max(x)
    zmin, zmax = np.min(z), np.max(z)
    # w0, h0 = 0.41, 0.11
    w0, h0 = 0.69, 0.10
    xs0, ys0 = 0.14, 0.98 - h0
    vgap, hgap = 0.018, 0.03

    def plot_one_field(fdata, ax, text, text_color, label_bottom='on',
                       label_left='on', ylabel=False, vmin=0, vmax=10,
                       colormap=plt.cm.seismic, xs=xs0, ys=ys0, ay_color='k'):
        plt.tick_params(labelsize=16)
        p1 = ax.imshow(fdata, vmin=vmin, vmax=vmax, cmap=colormap,
                       extent=[xmin, xmax, zmin, zmax], aspect='auto',
                       origin='lower', interpolation='bicubic')
        ax.tick_params(axis='x', labelbottom=label_bottom)
        ax.tick_params(axis='y', labelleft=label_left)
        if ylabel:
            ax.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
        ax.contour(x, z, Ay, colors=ay_color, linewidths=0.5)
        ax.text(0.02, 0.85, text, color=text_color, fontsize=20,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)
        xs1 = xs + w0 * 1.02
        w1 = w0 * 0.04
        cax = fig.add_axes([xs1, ys, w1, h0])
        cbar = fig.colorbar(p1, cax=cax)
        cbar.ax.tick_params(labelsize=16)
        return cbar

    dv = dx * dz
    pdiff = pperp - ppara
    pshear_xx = pdiff * bbsigmaxx
    pshear_yy = pdiff * bbsigmayy
    pshear_zz = pdiff * bbsigmazz
    pshear_xy = pdiff * bbsigmaxy
    pshear_xz = pdiff * bbsigmaxz
    pshear_yz = pdiff * bbsigmayz
    pshear_xx_cumsum = np.cumsum(np.sum(pshear_xx, axis=0)) * dv
    pshear_yy_cumsum = np.cumsum(np.sum(pshear_yy, axis=0)) * dv
    pshear_zz_cumsum = np.cumsum(np.sum(pshear_zz, axis=0)) * dv
    pshear_xy_cumsum = np.cumsum(np.sum(pshear_xy, axis=0)) * dv
    pshear_xz_cumsum = np.cumsum(np.sum(pshear_xz, axis=0)) * dv
    pshear_yz_cumsum = np.cumsum(np.sum(pshear_yz, axis=0)) * dv

    fig = plt.figure(figsize=[7, 11])
    xs, ys = xs0, ys0
    ax1 = fig.add_axes([xs, ys, w0, h0])
    text1 = r'$(p_\perp - p_\parallel)b_xb_x\sigma_{xx}$'
    dmax = 5E-4
    nmin, nmax = -dmax, dmax
    cbar1 = plot_one_field(pshear_xx, ax1, text1, 'r', label_bottom='off',
                           label_left='on', ylabel=True, vmin=nmin, vmax=nmax,
                           colormap=plt.cm.seismic, xs=xs, ys=ys, ay_color='k')
    cbar1.set_ticks(np.arange(nmin, nmax + dmax, dmax))
    ys -= h0 + vgap
    ax2 = fig.add_axes([xs, ys, w0, h0])
    text2 = r'$(p_\perp - p_\parallel)b_yb_y\sigma_{yy}$'
    cbar2 = plot_one_field(pshear_yy, ax2, text2, 'g', label_bottom='off',
                           label_left='on', ylabel=True, vmin=nmin, vmax=nmax,
                           colormap=plt.cm.seismic, xs=xs, ys=ys, ay_color='k')
    cbar2.set_ticks(np.arange(nmin, nmax + dmax, dmax))
    ys -= h0 + vgap
    ax3 = fig.add_axes([xs, ys, w0, h0])
    text3 = r'$(p_\perp - p_\parallel)b_zb_z\sigma_{zz}$'
    cbar3 = plot_one_field(pshear_zz, ax3, text3, 'b', label_bottom='off',
                           label_left='on', ylabel=True, vmin=nmin, vmax=nmax,
                           colormap=plt.cm.seismic, xs=xs, ys=ys, ay_color='k')
    cbar3.set_ticks(np.arange(nmin, nmax + dmax, dmax))
    ys -= h0 + vgap
    ax4 = fig.add_axes([xs, ys, w0, h0])
    text4 = r'$(p_\perp - p_\parallel)b_xb_y\sigma_{xy}$'
    cbar4 = plot_one_field(pshear_xy, ax4, text4, 'r', label_bottom='off',
                           label_left='on', ylabel=True, vmin=nmin, vmax=nmax,
                           colormap=plt.cm.seismic, xs=xs, ys=ys, ay_color='k')
    cbar4.set_ticks(np.arange(nmin, nmax + dmax, dmax))
    ys -= h0 + vgap
    ax5 = fig.add_axes([xs, ys, w0, h0])
    text5 = r'$(p_\perp - p_\parallel)b_xb_z\sigma_{xz}$'
    cbar5 = plot_one_field(pshear_xz, ax5, text5, 'g', label_bottom='off',
                           label_left='on', ylabel=True, vmin=nmin, vmax=nmax,
                           colormap=plt.cm.seismic, xs=xs, ys=ys, ay_color='k')
    cbar5.set_ticks(np.arange(nmin, nmax + dmax, dmax))
    ys -= h0 + vgap
    ax6 = fig.add_axes([xs, ys, w0, h0])
    text6 = r'$(p_\perp - p_\parallel)b_yb_z\sigma_{yz}$'
    cbar6 = plot_one_field(pshear_yz, ax6, text6, 'b', label_bottom='off',
                           label_left='on', ylabel=True, vmin=nmin, vmax=nmax,
                           colormap=plt.cm.seismic, xs=xs, ys=ys, ay_color='k')
    cbar6.set_ticks(np.arange(nmin, nmax + dmax, dmax))
    ys -= h0 + vgap
    ax7 = fig.add_axes([xs, ys, w0, h0])
    text7 = r'$p_\parallel - p_\perp$'
    nmin, nmax = -0.5, 0.5
    print("min and max of pdiff: %f %f" % (np.min(pdiff), np.max(pdiff)))
    cbar7 = plot_one_field(pdiff, ax7, text7, 'k', label_bottom='off',
                           label_left='on', ylabel=True, vmin=nmin, vmax=nmax,
                           colormap=plt.cm.seismic, xs=xs, ys=ys, ay_color='k')
    cbar7.set_ticks(np.arange(nmin, nmax + 0.5, 0.5))
    ys -= h0 + vgap
    ax8 = fig.add_axes([xs, ys, w0, h0])
    ax8.plot(x, pshear_xx_cumsum, linewidth=2, color='r')
    ax8.plot(x, pshear_yy_cumsum, linewidth=2, color='g')
    ax8.plot(x, pshear_zz_cumsum, linewidth=2, color='b')
    ax8.plot(x, pshear_xy_cumsum, linewidth=2, color='r', linestyle='--')
    ax8.plot(x, pshear_xz_cumsum, linewidth=2, color='g', linestyle='--')
    ax8.plot(x, pshear_yz_cumsum, linewidth=2, color='b', linestyle='--')
    ax8.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    ax8.tick_params(labelsize=16)

    fdir = '../img/shear_of_vexb/' + run_name + '/'
    mkdir_p(fdir)
    fname = fdir + 'shear_of_vexb_' + str(current_time)
    fname += '_' + species + '.jpg'
    fig.savefig(fname, dpi=200)
    plt.close()
    # plt.show()


def get_cmd_args():
    """Get command line arguments
    """
    default_run_name = 'mime25_beta008_guide00_frequent_dump'
    default_run_dir = '/net/scratch3/xiaocanli/reconnection/frequent_dump/' + \
            'mime25_beta008_guide00_frequent_dump/'
    parser = argparse.ArgumentParser(description='Compression analysis based on fluids')
    parser.add_argument('--species', action="store", default='e',
                        help='particle species')
    parser.add_argument('--run_dir', action="store", default=default_run_dir,
                        help='run directory')
    parser.add_argument('--run_name', action="store", default=default_run_name,
                        help='run name')
    parser.add_argument('--tframe_fields', action="store", default='30', type=int,
                        help='Time frame for fields')
    parser.add_argument('--multi_frames', action="store_true", default=False,
                        help='whether analyzing multiple frames')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_cmd_args()
    run_name = args.run_name
    run_dir = args.run_dir
    species = args.species
    tframe_fields = args.tframe_fields
    multi_frames = args.multi_frames
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    tratio = pic_info.particle_interval / pic_info.fields_interval
    # save_compression_json_single(pic_info, run_name)
    # plot_compression_time(pic_info, run_name, 'e')
    # plot_compression_time(pic_info, run_name, 'i')
    # generate_energization_terms_table()
    calc_jpolar_dote_multi(pic_info, run_dir, run_name, 'i')
    # calc_jpolar_dote_continusou_dump_multi(picinfo_fname, run_dir, run_name, 'e')
    # calc_jpolar_dote_continusou_dump_multi(picinfo_fname, run_dir, run_name, 'i')
    # jdote_calculation_test(pic_info, run_dir, 60)
    # compression_ratio_apjl_runs()
    # plot_compression_time_both(pic_info, run_name)
    # plot_compression_shear_single(pic_info, run_dir, run_name, 'e', 25)
    # compare_fluid_particle_energization(pic_info, run_name)
    # ncores = multiprocessing.cpu_count()
    # ncores = 10
    # ecov_min, ecov_max = find_min_max_energization_terms(run_name, species)
    # cts = range(pic_info.ntp)
    # def processInput(job_id):
    #     print job_id
    #     ct = job_id
    #     tframe_fields = (ct + 1) * tratio
    #     plot_energization_terms(pic_info, run_dir, run_name, species,
    #                             tframe_fields, ecov_min, ecov_max)
    #     plt.close()
    # if multi_frames:
    #     Parallel(n_jobs=ncores)(delayed(processInput)(ct) for ct in cts)
    # else:
    #     plot_energization_terms(pic_info, run_dir, run_name, species,
    #                             tframe_fields, ecov_min, ecov_max)
    #     plt.show()
    # plot_nrho_velocity(pic_info, run_dir, run_name, 50)
    # plot_compresion_of_vexb(pic_info, run_dir, run_name, 50, 'i')
    # plot_shear_of_vexb(pic_info, run_dir, run_name, 50, 'i')
    # cts = range(pic_info.ntp)
    # for ct in cts:
    #     tframe_fields = (ct + 1) * tratio
    #     # plot_nrho_velocity(pic_info, run_dir, run_name, tframe_fields)
    #     plot_compresion_of_vexb(pic_info, run_dir, run_name, tframe_fields,
    #                             species)
    #     plot_shear_of_vexb(pic_info, run_dir, run_name, tframe_fields,
    #                             species)
    # run_dir = '../../'
    # pic_info = pic_information.get_pic_info(run_dir)
    # ntp = pic_info.ntp
    # for i in range(pic_info.ntf):
    #     plot_compression(pic_info, 'i', i)
    # plot_compression(pic_info, 'e', 40)
    # plot_shear(pic_info, 'e', 40)
    # for ct in range(pic_info.ntf):
    #     plot_shear(pic_info, 'i', ct)
    # plot_compression_only(pic_info, 'i', 40)
    # for ct in range(pic_info.ntf):
    #     plot_compression_only(pic_info, 'e', ct)
    # plot_velocity_field(pic_info, 'e', 15)
    # for ct in range(pic_info.ntf):
    #     plot_velocity_field(pic_info, 'e', ct)
    # for ct in range(pic_info.ntf):
    #     plot_velocity_field(pic_info, 'i', ct)
    # plot_compression_shear(pic_info, 'e', 24)
    # plot_compression_cut(pic_info, 'i', 12)
    # angle_current(pic_info, 12)
    # species = 'e'
    # jdote = read_jdote_data(species)
    # compression_time(pic_info, species, jdote, [-1.0, 2])
    # density_ratio(pic_info, 8)
    # for ct in range(pic_info.ntf):
    #     density_ratio(pic_info, ct)
    # plot_velocity_components(pic_info, 'e', 40)
    # for ct in range(pic_info.ntf):
    #     plot_velocity_components(pic_info, 'e', ct)
    # for ct in range(pic_info.ntf):
    #     plot_velocity_components(pic_info, 'i', ct)
    # move_compression()
    # plot_compression_time_multi('i')
    # calc_compression(run_dir, pic_info)

