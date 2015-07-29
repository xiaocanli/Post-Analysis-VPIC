"""
Analysis procedures for 2D contour plots.
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy.ndimage.filters import generic_filter as gf
from scipy import signal
from scipy.fftpack import fft2, ifft2, fftshift
import math
import os.path
import struct
import collections
import pic_information
from contour_plots import read_2d_fields, plot_2d_contour
from energy_conversion import read_jdote_data

mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

font = {'family' : 'serif',
        #'color'  : 'darkred',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 24,
        }

def plot_compression(pic_info, species, current_time):
    """Plot out-of-plane current density.

    Args:
        pic_info: namedtuple for the PIC simulation information.
        species: 'e' for electrons, 'i' for ions.
        current_time: current time frame.
    """
    print(current_time)
    kwargs = {"current_time":current_time, "xl":0, "xr":200, "zb":-50, "zt":50}
    fname = "../../data1/udot_div_ptensor00_" + species + ".gda"
    x, z, udot_div_ptensor = read_2d_fields(pic_info, fname, **kwargs) 
    fname = "../../data1/pdiv_u00_" + species + ".gda"
    x, z, pdiv_u = read_2d_fields(pic_info, fname, **kwargs) 
    fname = "../../data1/div_u00_" + species + ".gda"
    x, z, div_u = read_2d_fields(pic_info, fname, **kwargs) 
    fname = "../../data1/pshear00_" + species + ".gda"
    x, z, pshear = read_2d_fields(pic_info, fname, **kwargs) 
    fname = "../../data1/div_udot_ptensor00_" + species + ".gda"
    x, z, div_udot_ptensor = read_2d_fields(pic_info, fname, **kwargs) 
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
    if species == 'e':
        je = - (ux*ex + uy*ey + uz*ez)
    else:
        je = ux*ex + uy*ey + uz*ez

    div_u = signal.medfilt2d(div_u, kernel_size=(5,5))
    pdiv_u = signal.medfilt2d(pdiv_u, kernel_size=(5,5))
    pshear = signal.medfilt2d(pshear, kernel_size=(5,5))
    udot_div_ptensor = signal.medfilt2d(udot_div_ptensor, kernel_size=(5,5))
    div_udot_ptensor = signal.medfilt2d(div_udot_ptensor, kernel_size=(5,5))
    je = signal.medfilt2d(je, kernel_size=(5,5))

    nx, = x.shape
    nz, = z.shape
    width = 0.75
    height = 0.11
    xs = 0.12
    ys = 0.98 - height
    gap = 0.025

    fig = plt.figure(figsize=[10,14])
    ax1 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep":1, "zstep":1, "vmin":-0.01, "vmax":0.01}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, pdiv_u, ax1, fig, **kwargs_plot)
    p1.set_cmap(plt.cm.seismic)
    ax1.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep], 
            colors='black', linewidths=0.5)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax1.tick_params(labelsize=20)
    ax1.tick_params(axis='x', labelbottom='off')
    cbar1.set_ticks(np.arange(-0.01, 0.015, 0.01))
    cbar1.ax.tick_params(labelsize=20)
    pdiv_u_cum = np.cumsum(np.sum(pdiv_u, axis=0))
    fname1 = r'$-p\nabla\cdot\mathbf{u}$'
    ax1.text(0.02, 0.8, fname1, color='red', fontsize=24, 
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform = ax1.transAxes)

    ys -= height + gap
    ax2 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep":1, "zstep":1, "vmin":-0.01, "vmax":0.01}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p2, cbar2 = plot_2d_contour(x, z, pshear, ax2, fig, **kwargs_plot)
    p2.set_cmap(plt.cm.seismic)
    ax2.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep], 
            colors='black', linewidths=0.5)
    ax2.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax2.tick_params(labelsize=20)
    ax2.tick_params(axis='x', labelbottom='off')
    cbar2.set_ticks(np.arange(-0.01, 0.015, 0.01))
    cbar2.ax.tick_params(labelsize=20)
    pshear_cum = np.cumsum(np.sum(pshear, axis=0))
    fname2 = r'$-(p_\parallel - p_\perp)b_ib_j\sigma_{ij}$'
    ax2.text(0.02, 0.8, fname2, color='green', fontsize=24, 
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform = ax2.transAxes)
    
    ys -= height + gap
    ax3 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep":1, "zstep":1, "vmin":-0.01, "vmax":0.01}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p3, cbar3 = plot_2d_contour(x, z, div_udot_ptensor, 
            ax3, fig, **kwargs_plot)
    p3.set_cmap(plt.cm.seismic)
    ax3.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep], 
            colors='black', linewidths=0.5)
    ax3.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax3.tick_params(labelsize=20)
    ax3.tick_params(axis='x', labelbottom='off')
    cbar3.set_ticks(np.arange(-0.01, 0.015, 0.01))
    cbar3.ax.tick_params(labelsize=20)
    pcomp1 = np.cumsum(np.sum(div_udot_ptensor, axis=0))
    fname3 = r'$\nabla\cdot(\mathbf{u}\cdot\mathcal{P})$'
    ax3.text(0.02, 0.8, fname3, color='blue', fontsize=24, 
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform = ax3.transAxes)

    ys -= height + gap
    ax4 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep":1, "zstep":1, "vmin":-0.01, "vmax":0.01}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    data4 = pdiv_u + pshear + div_udot_ptensor
    p4, cbar4 = plot_2d_contour(x, z, data4, ax4, fig, **kwargs_plot)
    p4.set_cmap(plt.cm.seismic)
    ax4.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep], 
            colors='black', linewidths=0.5)
    ax4.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax4.tick_params(labelsize=20)
    ax4.tick_params(axis='x', labelbottom='off')
    cbar4.set_ticks(np.arange(-0.01, 0.015, 0.01))
    cbar4.ax.tick_params(labelsize=20)
    pcomp2 = np.cumsum(np.sum(data4, axis=0))
    fname4 = fname3 + fname1 + fname2
    ax4.text(0.02, 0.8, fname4, color='darkred', fontsize=24, 
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform = ax4.transAxes)

    ys -= height + gap
    ax5 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep":1, "zstep":1, "vmin":-0.01, "vmax":0.01}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p5, cbar5 = plot_2d_contour(x, z, udot_div_ptensor, ax5, fig, **kwargs_plot)
    p5.set_cmap(plt.cm.seismic)
    ax5.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep], 
            colors='black', linewidths=0.5)
    ax5.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax5.tick_params(labelsize=20)
    ax5.tick_params(axis='x', labelbottom='off')
    cbar5.set_ticks(np.arange(-0.01, 0.015, 0.01))
    cbar5.ax.tick_params(labelsize=20)
    pcomp3 = np.cumsum(np.sum(udot_div_ptensor, axis=0))
    ax5.text(0.02, 0.8, r'$\mathbf{u}\cdot(\nabla\cdot\mathcal{P})$',
            color='black', fontsize=24, 
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform = ax5.transAxes)

    ys -= height + gap
    ax6 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep":1, "zstep":1, "vmin":-0.01, "vmax":0.01}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p6, cbar6 = plot_2d_contour(x, z, je, ax6, fig, **kwargs_plot)
    p6.set_cmap(plt.cm.seismic)
    ax6.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep], 
            colors='black', linewidths=0.5)
    ax6.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax6.tick_params(labelsize=20)
    ax6.tick_params(axis='x', labelbottom='off')
    cbar6.set_ticks(np.arange(-0.01, 0.015, 0.01))
    cbar6.ax.tick_params(labelsize=20)
    je_cum = np.cumsum(np.sum(je, axis=0))
    fname6 = r'$' + '\mathbf{j}_' + species + '\cdot\mathbf{E}' + '$'
    ax6.text(0.02, 0.8, fname6, color='black', fontsize=24, 
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform = ax6.transAxes)

    ys -= height + gap
    ax7 = fig.add_axes([xs, ys, width, height])
    ax7.plot(x, pdiv_u_cum, linewidth=2, color='r')
    ax7.plot(x, pshear_cum, linewidth=2, color='g')
    ax7.plot(x, pcomp1, linewidth=2, color='b')
    ax7.plot(x, pcomp2, linewidth=2, color='darkred')
    ax7.plot(x, pcomp3, linewidth=2, color='k')
    ax7.plot(x, je_cum, linewidth=2, color='k', linestyle='-.')
    xmax = np.max(x)
    xmin = np.min(x)
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
    
    # plt.show()
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    if not os.path.isdir('../img/img_compression/'):
        os.makedirs('../img/img_compression/')
    fname = 'compression' + str(current_time).zfill(3) + '_' + species + '.jpg'
    fname = '../img/img_compression/' + fname
    fig.savefig(fname)
    plt.close()


def angle_current(pic_info, current_time):
    """Angle between calculated current and simulation current.

    Args:
        pic_info: namedtuple for the PIC simulation information.
        current_time: current time frame.
    """
    kwargs = {"current_time":current_time, "xl":0, "xr":200, "zb":-15, "zt":15}
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
    ang_current = np.arccos((jx1*jx + jy1*jy + jz1*jz) / (absJ * absJ1))

    ang_current = ang_current * 180 / math.pi 

    nx, = x.shape
    nz, = z.shape
    width = 0.75
    height = 0.7
    xs = 0.12
    ys = 0.9 - height
    fig = plt.figure(figsize=[10,4])
    ax1 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep":1, "zstep":1}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, ang_current, ax1, fig, **kwargs_plot)
    p1.set_cmap(plt.cm.seismic)
    ax1.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep], 
            colors='black', linewidths=0.5)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax1.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
    ax1.tick_params(labelsize=24)
    cbar1.ax.set_ylabel(r'$\theta(\mathbf{j}, \mathbf{u}$)',
            fontdict=font, fontsize=24)
    cbar1.ax.tick_params(labelsize=24)
    
    t_wci = current_time*pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(t_wci) + '/\Omega_{ci}$'
    ax1.set_title(title, fontdict=font, fontsize=24)

    fig = plt.figure(figsize=[7, 5])
    w1, h1 = 0.8, 0.8
    xs, ys = 0.96 - w1, 0.96 - h1
    ax2 = fig.add_axes([xs, ys, w1, h1])
    ang_bins = bins=np.arange(180)
    hist, bin_edges = np.histogram(ang_current, bins=ang_bins, density=True)
    p2 = ax2.plot(hist, linewidth=2)
    ax2.tick_params(labelsize=20)
    ax2.set_xlabel(r'$\theta$', fontdict=font, fontsize=24)
    ax2.set_ylabel(r'$f(\theta)$', fontdict=font, fontsize=24)

    plt.show()
    # plt.close()

def bulk_energy(pic_info, species, current_time):
    """Angle between calculated current and simulation current.

    Args:
        pic_info: namedtuple for the PIC simulation information.
        species: 'e' for electrons, 'i' for ions.
        current_time: current time frame.
    """
    kwargs = {"current_time":current_time, "xl":0, "xr":200, "zb":-15, "zt":15}
    fname = "../../data/u" + species + "x.gda"
    x, z, ux = read_2d_fields(pic_info, fname, **kwargs) 
    fname = "../../data/u" + species + "y.gda"
    x, z, uy = read_2d_fields(pic_info, fname, **kwargs) 
    fname = "../../data/u" + species + "z.gda"
    x, z, uz = read_2d_fields(pic_info, fname, **kwargs) 
    fname = "../../data/n" + species + ".gda"
    x, z, nrho = read_2d_fields(pic_info, fname, **kwargs) 
    fname = "../../data/p" + species + "-xx.gda"
    x, z, pxx = read_2d_fields(pic_info, fname, **kwargs) 
    fname = "../../data/p" + species + "-yy.gda"
    x, z, pyy = read_2d_fields(pic_info, fname, **kwargs) 
    fname = "../../data/p" + species + "-zz.gda"
    x, z, pzz = read_2d_fields(pic_info, fname, **kwargs) 
    x, z, Ay = read_2d_fields(pic_info, "../../data/Ay.gda", **kwargs) 

    if species == 'e':
        ptl_mass = 1.0
    else:
        ptl_mass = pic_info.mime

    internal_ene = (pxx + pyy + pzz) * 0.5
    bulk_ene = 0.5 * ptl_mass * nrho * (ux**2 + uy**2 + uz**2)

    nx, = x.shape
    nz, = z.shape
    width = 0.75
    height = 0.7
    xs = 0.12
    ys = 0.9 - height
    fig = plt.figure(figsize=[10,4])
    ax1 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep":1, "zstep":1, "is_log":True, "vmin":0.1, "vmax":10.0}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, bulk_ene/internal_ene,
            ax1, fig, **kwargs_plot)
    p1.set_cmap(plt.cm.seismic)
    ax1.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep], 
            colors='black', linewidths=0.5)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax1.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
    ax1.tick_params(labelsize=24)
    cbar1.ax.set_ylabel(r'$K/u$',
            fontdict=font, fontsize=24)
    cbar1.ax.tick_params(labelsize=24)
    
    t_wci = current_time*pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(t_wci) + '/\Omega_{ci}$'
    ax1.set_title(title, fontdict=font, fontsize=24)

    # plt.show()
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    if not os.path.isdir('../img/img_bulk_internal/'):
        os.makedirs('../img/img_bulk_internal/')
    dir = '../img/img_bulk_internal/'
    fname = 'bulk_internal' + str(current_time).zfill(3) + '_' + species + '.jpg'
    fname = dir + fname
    fig.savefig(fname)
    plt.close()


def compression_time(pic_info, species):
    """The time evolution of compression related terms.

    Args:
        pic_info: namedtuple for the PIC simulation information.
        species: 'e' for electrons, 'i' for ions.
    """
    ntf = pic_info.ntf
    tfields = pic_info.tfields
    fname = "../data/compression00_" + species + ".gda"
    fh = open(fname, 'r')
    data = fh.read()
    fh.close()
    compression_data = np.zeros((ntf, 2))
    index_start = 0
    index_end = 4
    for ct in range(ntf):
        for i in range(2):
            compression_data[ct, i], = \
                    struct.unpack('f', data[index_start:index_end])
            index_start = index_end
            index_end += 4
    div_u = compression_data[:, 0]
    pdiv_u = compression_data[:, 1]

    fname = "../data/shear00_" + species + ".gda"
    fh = open(fname, 'r')
    data = fh.read()
    fh.close()
    shear_data = np.zeros((ntf, 2))
    index_start = 0
    index_end = 4
    for ct in range(ntf):
        for i in range(2):
            shear_data[ct, i], = \
                    struct.unpack('f', data[index_start:index_end])
            index_start = index_end
            index_end += 4
    bbsigma = shear_data[:, 0]
    pshear = shear_data[:, 1]

    fname = "../data/div_udot_ptensor00_" + species + ".gda"
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
    div_udot_ptensor = data1[:]

    fname = "../data/udot_div_ptensor00_" + species + ".gda"
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
    udot_div_ptensor = data1[:]

    ene_bx = pic_info.ene_bx
    enorm = ene_bx[0]
    dtwpe = pic_info.dtwpe
    dtwci = pic_info.dtwci
    dt_fields = pic_info.dt_fields * dtwpe / dtwci
    pdiv_u_cum = np.cumsum(pdiv_u) * dt_fields
    pshear_cum = np.cumsum(pshear) * dt_fields
    div_udot_ptensor_cum = np.cumsum(div_udot_ptensor) * dt_fields
    udot_div_ptensor_cum = np.cumsum(udot_div_ptensor) * dt_fields
    pdiv_u_cum /= enorm
    pshear_cum /= enorm
    div_udot_ptensor_cum /= enorm
    udot_div_ptensor_cum /= enorm

    jdote = read_jdote_data(species)
    jqnudote = jdote.jqnupara_dote + jdote.jqnuperp_dote
    jqnudote_cum = jdote.jqnupara_dote_int + jdote.jqnuperp_dote_int
    jqnudote_cum /= enorm

    fig = plt.figure(figsize=[7, 5])
    w1, h1 = 0.8, 0.4
    xs, ys = 0.96-w1, 0.96-h1
    ax = fig.add_axes([xs, ys, w1, h1])
    label1 = r'$-p\nabla\cdot\mathbf{u}$'
    label2 = r'$-(p_\parallel - p_\perp)b_ib_j\sigma_{ij}$'
    label3 = r'$\nabla\cdot(\mathbf{u}\cdot\mathcal{P})$'
    label4 = label3 + label1 + label2
    label5 = r'$\mathbf{u}\cdot(\nabla\cdot\mathcal{P})$'
    label6 = r'$\mathbf{j}\cdot\mathbf{E}$'
    p1 = ax.plot(tfields, pdiv_u, linewidth=2, color='r', label=label1)
    p2 = ax.plot(tfields, pshear, linewidth=2, color='g', label=label2)
    p3 = ax.plot(tfields, div_udot_ptensor, linewidth=2,
            color='b', label=label3)
    p4 = ax.plot(tfields, pdiv_u + pshear + div_udot_ptensor,
            linewidth=2, color='darkred', label=label4)
    p5 = ax.plot(tfields, udot_div_ptensor, linewidth=2, color='k',
            label=label5)
    p6 = ax.plot(tfields, jqnudote, linewidth=2, color='k', linestyle='--',
            label=label6)
    ax.set_ylabel(r'$d\varepsilon_c/dt$', fontdict=font, fontsize=20)
    ax.tick_params(axis='x', labelbottom='off')
    ax.tick_params(labelsize=16)

    ax.text(0.45, 0.9, label1, color='red', fontsize=20, 
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform = ax.transAxes)
    ax.text(0.65, 0.9, label2, color='green', fontsize=20, 
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform = ax.transAxes)
    ax.text(0.5, 0.7, label3, color='blue', fontsize=20, 
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform = ax.transAxes)
    ax.text(0.75, 0.7, label5, color='black', fontsize=20, 
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform = ax.transAxes)
    ax.text(0.1, 0.05, label4, color='darkred', fontsize=20, 
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform = ax.transAxes)

    ys -= h1 + 0.05
    ax1 = fig.add_axes([xs, ys, w1, h1])
    p1 = ax1.plot(tfields, pdiv_u_cum, linewidth=2, color='r')
    p2 = ax1.plot(tfields, pshear_cum, linewidth=2, color='g')
    p3 = ax1.plot(tfields, div_udot_ptensor_cum, linewidth=2, color='b')
    p3 = ax1.plot(tfields, pdiv_u_cum + pshear_cum + div_udot_ptensor_cum,
            linewidth=2, color='darkred')
    p5 = ax1.plot(tfields, udot_div_ptensor_cum, linewidth=2, color='k')
    p6 = ax1.plot(tfields, jqnudote_cum, linewidth=2, color='k',
            linestyle='--', label=label6)
    ax1.set_xlabel(r'$t\Omega_{ci}$', fontdict=font, fontsize=20)
    ax1.set_ylabel(r'$\varepsilon_c$', fontdict=font, fontsize=20)
    ax1.tick_params(labelsize=16)
    ax1.legend(loc=2, prop={'size':20}, ncol=1,
            shadow=False, fancybox=False, frameon=False)
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    fname = '../img/compressional_' + species + '.eps'
    fig.savefig(fname)
    plt.show()


def density_ratio(pic_info, current_time):
    """Electron and ion density ratio.

    Args:
        pic_info: namedtuple for the PIC simulation information.
        current_time: current time frame.
    """
    kwargs = {"current_time":current_time, "xl":0, "xr":200, "zb":-15, "zt":15}
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
    fig = plt.figure(figsize=[10,4])
    ax1 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep":1, "zstep":1, "vmin":0.5, "vmax":1.5}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, ne/ni,
            ax1, fig, **kwargs_plot)
    p1.set_cmap(plt.cm.seismic)
    ax1.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep], 
            colors='black', linewidths=0.5)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax1.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
    ax1.tick_params(labelsize=24)
    cbar1.ax.set_ylabel(r'$n_e/n_i$',
            fontdict=font, fontsize=24)
    cbar1.ax.tick_params(labelsize=24)
    
    t_wci = current_time*pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(t_wci) + '/\Omega_{ci}$'
    ax1.set_title(title, fontdict=font, fontsize=24)

    plt.show()
    # dir = '../img/img_density_ratio/'
    # if not os.path.isdir('../img/'):
    #     os.makedirs('../img/')
    # if not os.path.isdir(dir):
    #     os.makedirs(dir)
    # fname = 'density_ratio' + str(current_time).zfill(3) + '.jpg'
    # fname = dir + fname
    # fig.savefig(fname)
    # plt.close()

if __name__ == "__main__":
    pic_info = pic_information.get_pic_info('../../')
    ntp = pic_info.ntp
    # for i in range(pic_info.ntf):
    #     plot_compression(pic_info, 'e', i)
    # plot_compression(pic_info, 'e', 12)
    # angle_current(pic_info, 12)
    # bulk_energy(pic_info, 'e', 12)
    # for ct in range(pic_info.ntf):
    #     bulk_energy(pic_info, 'i', ct)
    # compression_time(pic_info, 'i')
    density_ratio(pic_info, 40)
