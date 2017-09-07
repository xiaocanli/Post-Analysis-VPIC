"""
Analysis procedures for compression related terms.
"""
import collections
import math
import os
import os.path
import struct
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import simplejson as json
from matplotlib import rc
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
from scipy.fftpack import fft2, fftshift, ifft2
from scipy.interpolate import spline, interp1d
from scipy.ndimage.filters import generic_filter as gf

import pic_information
from contour_plots import plot_2d_contour, read_2d_fields, find_closest
from energy_conversion import read_data_from_json, read_jdote_data
from runs_name_path import ApJ_long_paper_runs
from serialize_json import data_to_json, json_to_data
from shell_functions import mkdir_p

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

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
    div_vdot_ptensor_cum = np.cumsum(div_vdot_ptensor) * dt_fields
    vdot_div_ptensor_cum = np.cumsum(vdot_div_ptensor) * dt_fields

    pdiv_uperp_usingle = pdiv_usingle - pdiv_upara_usingle
    pshear_perp_usingle = pshear_single - pshear_para_usingle
    pdiv_uperp_usingle_cum = pdiv_usingle_cum - pdiv_upara_usingle_cum
    pshear_perp_usingle_cum = pshear_single_cum - pshear_para_usingle_cum

    compression_collection = collections.namedtuple('compression_collection', [
        'div_u', 'pdiv_u', 'div_usingle', 'div_upara_usingle', 'pdiv_usingle',
        'pdiv_upara_usingle', 'pdiv_uperp_usingle', 'bbsigma', 'pshear',
        'bbsigma_single', 'bbsigma_para_usingle', 'pshear_single',
        'pshear_para_usingle', 'pshear_perp_usingle', 'div_vdot_ptensor',
        'vdot_div_ptensor', 'pdiv_u_cum', 'pshear_cum', 'pdiv_usingle_cum',
        'pdiv_upara_usingle_cum', 'pdiv_uperp_usingle_cum', 'pshear_single_cum',
        'pshear_para_usingle_cum', 'pshear_perp_usingle_cum',  'div_vdot_ptensor_cum',
        'vdot_div_ptensor_cum'
        ])
    compression_data = compression_collection(div_u, pdiv_u, div_usingle,
            div_upara_usingle, pdiv_usingle, pdiv_upara_usingle,
            pdiv_uperp_usingle, bbsigma, pshear, bbsigma_single,
            bbsigma_para_usingle, pshear_single, pshear_para_usingle,
            pshear_perp_usingle, div_vdot_ptensor, vdot_div_ptensor,
            pdiv_u_cum, pshear_cum, pdiv_usingle_cum, pdiv_upara_usingle_cum,
            pdiv_uperp_usingle_cum, pshear_single_cum, pshear_para_usingle_cum,
            pshear_perp_usingle_cum,  div_vdot_ptensor_cum, vdot_div_ptensor_cum)
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
    kwargs = {
        "current_time": current_time,
        "xl": 0,
        "xr": 200,
        "zb": -20,
        "zt": 20
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
    base_dirs, run_names = ApJ_long_paper_runs()
    for base_dir, run_name in zip(base_dirs, run_names):
        fpath = dir + run_name
        if not os.path.isdir(fpath):
            os.makedirs(fpath)
        command = "cp " + base_dir + "/pic_analysis/data/compression00* " + \
                  fpath
        os.system(command)
        command = "cp " + base_dir + "/pic_analysis/data/shear00* " + fpath
        os.system(command)
        command = "cp " + base_dir + \
                  "/pic_analysis/data/div_vdot_ptensor00* " + \
                  fpath
        os.system(command)
        command = "cp " + base_dir + \
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
    base_dirs, run_names = ApJ_long_paper_runs()
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

    jpolar_dote = jdote.jpolar_dote
    jpolar_dote_int = jdote.jpolar_dote_int
    jqnudote = jdote.jqnupara_dote + jdote.jqnuperp_dote
    jqnudote_cum = jdote.jqnupara_dote_int + jdote.jqnuperp_dote_int
    # jqnudote -= jpolar_dote
    # jqnudote_cum -= jpolar_dote_int
    print cdata.pdiv_uperp_usingle[49]
    print jdote.jqnuperp_dote[49]

    fdata1 = cdata.pdiv_uperp_usingle + cdata.pshear_perp_usingle
    fdata2 = jdote.jqnuperp_dote
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
    p11 = ax.plot(t_new, fdata1_new)
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

    fdata1 = cdata.pdiv_uperp_usingle_cum + cdata.pshear_perp_usingle_cum
    fdata2 = jdote.jqnuperp_dote_int
    print fdata1[-1] / fdata2[-1]

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

    fig = plt.figure(figsize=[7, 6])
    w1, h1 = 0.81, 0.33
    xs, ys = 0.96 - w1, 0.84 - h1
    ax = fig.add_axes([xs, ys, w1, h1])
    label1 = r'$-p_s\nabla\cdot\boldsymbol{u}_\perp$'
    label2 = r'$-(p_{s\parallel} - p_{s\perp})b_ib_j\sigma_{ij}$'
    label3 = r'$\nabla\cdot(\mathcal{P}\cdot\mathbf{u})$'
    label4 = label3 + label1 + label2
    label5 = r'$\mathbf{u}\cdot(\nabla\cdot\mathcal{P})$'
    label6 = r'$\boldsymbol{j}_{s\perp}\cdot\boldsymbol{E}_\perp$'
    p1 = ax.plot(tfields, cdata_e.pdiv_uperp_usingle,
                 linewidth=2, color='r', label=label1)
    p2 = ax.plot(tfields, cdata_e.pshear_perp_usingle,
                 linewidth=2, color='g', label=label2)
    p12 = ax.plot(tfields, cdata_e.pdiv_uperp_usingle + cdata_e.pshear_perp_usingle,
                 linewidth=2, color='k', label='Sum')
    # fdata = jdote_in_e.jqnuperp_dote - jdote_in_e.jpolar_dote
    fdata = jdote_e.jqnuperp_dote - jdote_e.jpolar_dote
    # fdata = jdote_e.jqnuperp_dote
    p3 = ax.plot(tfields, fdata, linewidth=2, color='k', linestyle='--', label=label6)
    ax.set_ylabel(r'$d\varepsilon_e/dt$', fontdict=font, fontsize=20)
    ax.tick_params(axis='x', labelbottom='off')
    ax.tick_params(labelsize=16)
    tmax = min(np.max(pic_info.tfields), 800)
    ax.set_xlim([0, 800])
    # ax.set_ylim([-0.2, 0.8])
    # ax.set_ylim([-0.05, 0.12])
    ax.legend(loc='upper center', prop={'size': 20}, ncol=2,
            bbox_to_anchor=(0.5, 1.5),
            # bbox_to_anchor=(0.5, 1.4),
            shadow=False, fancybox=False, frameon=False)

    ax.text(0.95, 0.8, 'electron', color='k', fontsize=20,
        bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes)

    ys -= h1 + 0.05
    ax1 = fig.add_axes([xs, ys, w1, h1])
    p1 = ax1.plot(tfields, cdata_i.pdiv_uperp_usingle,
                 linewidth=2, color='r', label=label1)
    p2 = ax1.plot(tfields, cdata_i.pshear_perp_usingle,
                 linewidth=2, color='g', label=label1)
    p12 = ax1.plot(tfields, cdata_i.pdiv_uperp_usingle + cdata_i.pshear_perp_usingle,
                 linewidth=2, color='k', label=label1)
    # fdata = jdote_in_i.jqnuperp_dote - jdote_in_i.jpolar_dote
    # fdata = jdote_in_e.jqnuperp_dote - jdote_in_e.jpolar_dote
    fdata = jdote_i.jqnuperp_dote - jdote_i.jpolar_dote
    # fdata = jdote_i.jqnuperp_dote
    p3 = ax1.plot(tfields, fdata, linewidth=2, color='k',
            linestyle='--', label=label6)
    ax1.set_xlabel(r'$t\Omega_{ci}$', fontdict=font, fontsize=20)
    ax1.set_ylabel(r'$d\varepsilon_i/dt$', fontdict=font, fontsize=20)
    ax1.tick_params(labelsize=16)
    ax1.set_xlim(ax.get_xlim())
    # ax1.set_ylim([-0.05, 0.12])
    ax1.text(0.95, 0.8, 'ion', color='k', fontsize=20,
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
    run_names = ['mime25_beta002_guide00',
                 'mime25_beta002_guide02',
                 'mime25_beta002_guide05',
                 'mime25_beta002_guide10',
                 'mime25_beta002_guide00',
                 'mime25_beta008_guide00',
                 'mime25_beta032_guide00']
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


if __name__ == "__main__":
    cmdargs = sys.argv
    if (len(cmdargs) > 2):
        base_dir = cmdargs[1]
        run_name = cmdargs[2]
    else:
        base_dir = '/net/scratch2/guofan/sigma1-mime25-beta001-average/'
        run_name = 'sigma1-mime25-beta001-average'
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    # # save_compression_json_single(pic_info, run_name)
    # plot_compression_time(pic_info, run_name, 'e')
    compression_ratio_apjl_runs()
    # plot_compression_time_both(pic_info, run_name)
    # plot_compression_shear_single(pic_info, base_dir, run_name, 'e', 25)
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
