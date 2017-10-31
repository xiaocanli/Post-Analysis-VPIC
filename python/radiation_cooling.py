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
from dolointerpolation import MultilinearInterpolator
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

def plot_magentic_field_one_frame(run_dir, run_name, tframe):
    """
    """
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    kwargs = {"current_time": tframe, "xl": 0, "xr": pic_info.lx_di,
              "zb": -0.5 * pic_info.lz_di, "zt": 0.5 * pic_info.lz_di}
    size_one_frame = pic_info.nx * pic_info.nz * 4
    fname = run_dir + "data/bx.gda"
    x, z, bx = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/by.gda"
    x, z, by = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/bz.gda"
    x, z, bz = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/absB.gda"
    x, z, absB = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/Ay.gda"
    x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
    xmin, xmax = np.min(x), np.max(x)
    zmin, zmax = np.min(z), np.max(z)

    w0, h0 = 0.78, 0.2
    xs0, ys0 = 0.09, 0.95 - h0
    vgap, hgap = 0.02, 0.02

    vmax1 = 3.0E2
    vmin1 = -vmax1
    def plot_one_field(fdata, ax, text, text_color, label_bottom='on',
                       label_left='on', ylabel=False, ay_color='k',
                       vmin=vmin1, vmax=vmax1, cmap1=plt.cm.seismic,
                       log_scale=False):
        plt.tick_params(labelsize=16)
        if log_scale:
            im1 = ax.imshow(fdata, cmap=cmap1,
                            extent=[xmin, xmax, zmin, zmax], aspect='auto',
                            origin='lower', interpolation='bicubic')
            im1.norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            im1 = ax.imshow(fdata, cmap=cmap1, vmin=vmin, vmax=vmax,
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
        return im1
    fig = plt.figure(figsize=[12, 12])
    xs, ys = xs0, ys0
    ax1 = fig.add_axes([xs, ys, w0, h0])
    im1 = plot_one_field(bx, ax1, r'$B_x$', 'w', label_bottom='off',
                         label_left='on', ylabel=True, ay_color='k')
    ys -= h0 + vgap
    ax2 = fig.add_axes([xs, ys, w0, h0])
    im2 = plot_one_field(by, ax2, r'$B_y$', 'k', label_bottom='off',
                         label_left='on', ylabel=True, ay_color='k')
    ys -= h0 + vgap
    ax3 = fig.add_axes([xs, ys, w0, h0])
    im3 = plot_one_field(bz, ax3, r'$B_z$', 'k', label_bottom='off',
                         label_left='on', ylabel=True, ay_color='k')
    xs1 = xs + w0 + hgap
    w1 = 0.03
    h1 = 3 * h0 + 2 * vgap
    cax1 = fig.add_axes([xs1, ys, w1, h1])
    cbar1 = fig.colorbar(im1, cax=cax1)
    cbar1.ax.tick_params(labelsize=16)
    ys -= h0 + vgap
    ax4 = fig.add_axes([xs, ys, w0, h0])
    im4 = plot_one_field(absB, ax4, r'$B$', 'k', label_bottom='on',
                         label_left='on', ylabel=True, ay_color='k',
                         vmin=10, vmax=300, cmap1=plt.cm.viridis)
    ax4.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    cax2 = fig.add_axes([xs1, ys, w1, h0])
    cbar2 = fig.colorbar(im4, cax=cax2)
    cbar2.ax.tick_params(labelsize=16)

    t_wci = tframe * pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(t_wci) + '\Omega_{ci}^{-1}$'
    title += ' (frame: %d)' % tframe
    ax1.set_title(title, fontdict=font, fontsize=24)

    fdir = '../img/radiation_cooling/magnetic_field/' + run_name + '/'
    mkdir_p(fdir)
    fname = fdir + 'bfields_' + str(tframe) + '.jpg'
    fig.savefig(fname, dpi=200)

    plt.close()
    # plt.show()


def plot_nrho_momentum(run_dir, run_name, tframe):
    """
    """
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    kwargs = {"current_time": tframe, "xl": 0, "xr": pic_info.lx_di,
              "zb": -0.5 * pic_info.lz_di, "zt": 0.5 * pic_info.lz_di}
    size_one_frame = pic_info.nx * pic_info.nz * 4
    # fname = run_dir + "data/uex.gda"
    # x, z, uex = read_2d_fields(pic_info, fname, **kwargs)
    # fname = run_dir + "data/uey.gda"
    # x, z, uey = read_2d_fields(pic_info, fname, **kwargs)
    # fname = run_dir + "data/uez.gda"
    # x, z, uez = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/vex.gda"
    x, z, vex = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/vey.gda"
    x, z, vey = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/vez.gda"
    x, z, vez = read_2d_fields(pic_info, fname, **kwargs)
    # fname = run_dir + "data/uix.gda"
    # x, z, uix = read_2d_fields(pic_info, fname, **kwargs)
    # fname = run_dir + "data/uiy.gda"
    # x, z, uiy = read_2d_fields(pic_info, fname, **kwargs)
    # fname = run_dir + "data/uiz.gda"
    # x, z, uiz = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/ne.gda"
    x, z, ne = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/ni.gda"
    x, z, ni = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/Ay.gda"
    x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
    xmin, xmax = np.min(x), np.max(x)
    zmin, zmax = np.min(z), np.max(z)

    smime = math.sqrt(pic_info.mime)
    dx = pic_info.dx_di * smime
    dz = pic_info.dz_di * smime

    curv_vex = -np.gradient(vey, dz, axis=0)
    curv_vey = np.gradient(vex, dz, axis=0) - np.gradient(vez, dx, axis=1)
    curv_vez = np.gradient(vey, dx, axis=1)
    curv_vex = gaussian_filter(curv_vex, 3)
    curv_vey = gaussian_filter(curv_vey, 3)
    curv_vez = gaussian_filter(curv_vez, 3)

    xv, zv = np.meshgrid(x, z)

    w0, h0 = 0.78, 0.2
    xs0, ys0 = 0.09, 0.95 - h0
    vgap, hgap = 0.02, 0.02

    vmax1 = 1.0E-2
    vmin1 = -vmax1
    def plot_one_field(fdata, ax, text, text_color, label_bottom='on',
                       label_left='on', ylabel=False, ay_color='k',
                       vmin=vmin1, vmax=vmax1, cmap1=plt.cm.seismic,
                       log_scale=False):
        plt.tick_params(labelsize=16)
        if log_scale:
            im1 = ax.imshow(fdata, cmap=cmap1,
                            extent=[xmin, xmax, zmin, zmax], aspect='auto',
                            origin='lower', interpolation='bicubic')
            im1.norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            im1 = ax.imshow(fdata, cmap=cmap1, vmin=vmin, vmax=vmax,
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
        return im1
    fig = plt.figure(figsize=[12, 12])
    xs, ys = xs0, ys0
    ax1 = fig.add_axes([xs, ys, w0, h0])
    im1 = plot_one_field(curv_vex, ax1, r'$n_em_eu_{ex}$', 'k',
                         label_bottom='off', label_left='on', ylabel=True,
                         ay_color='k')
    # strm = ax1.streamplot(xv, zv, vex, vez, linewidth=1, color=vex,
    #                       density=[2,2], cmap=plt.cm.binary)
    ys -= h0 + vgap
    ax2 = fig.add_axes([xs, ys, w0, h0])
    im2 = plot_one_field(curv_vey, ax2, r'$n_em_eu_{ey}$', 'k',
                         label_bottom='off', label_left='on', ylabel=True,
                         ay_color='k')
    ys -= h0 + vgap
    ax3 = fig.add_axes([xs, ys, w0, h0])
    im3 = plot_one_field(curv_vez, ax3, r'$n_em_eu_{ez}$', 'k',
                         label_bottom='off', label_left='on', ylabel=True,
                         ay_color='k')
    # xs1 = xs + w0 + hgap
    # w1 = 0.03
    # h1 = 3 * h0 + 2 * vgap
    # cax1 = fig.add_axes([xs1, ys, w1, h1])
    # cbar1 = fig.colorbar(im1, cax=cax1)
    # cbar1.ax.tick_params(labelsize=16)
    # ys -= h0 + vgap
    # ax4 = fig.add_axes([xs, ys, w0, h0])
    # im4 = plot_one_field(absB, ax4, r'$B$', 'k', label_bottom='on',
    #                      label_left='on', ylabel=True, ay_color='k',
    #                      vmin=10, vmax=300, cmap1=plt.cm.viridis)
    # ax4.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    # cax2 = fig.add_axes([xs1, ys, w1, h0])
    # cbar2 = fig.colorbar(im4, cax=cax2)
    # cbar2.ax.tick_params(labelsize=16)

    t_wci = tframe * pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(t_wci) + '\Omega_{ci}^{-1}$'
    title += ' (frame: %d)' % tframe
    ax1.set_title(title, fontdict=font, fontsize=24)

    # fdir = '../img/radiation_cooling/magnetic_field/' + run_name + '/'
    # mkdir_p(fdir)
    # fname = fdir + 'bfields_' + str(tframe) + '.jpg'
    # fig.savefig(fname, dpi=200)

    # plt.close()
    plt.show()


def plot_particle_distribution_2d(run_dir, run_name, ct):
    """
    """
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    mpi_sizex = pic_info.topology_x
    mpi_sizez = pic_info.topology_z
    nbins = 1000
    ndata = nbins + 3  # including bx, by, bz

    rank = 0
    tindex = ct * pic_info.fields_interval
    fname_pre = (run_dir + 'hydro/T.' + str(tindex) +
                 '/spectrum-ehydro.' + str(tindex))
    fname = fname_pre + '.' + str(rank)
    fdata = np.fromfile(fname, dtype=np.float32)
    sz, = fdata.shape
    nzone = sz / ndata

    fname = run_dir + 'distributions_2d/dist_3e3.' + str(ct)
    nrho_3e3 = np.fromfile(fname)
    nrho_3e3 = nrho_3e3.reshape((mpi_sizex, nzone*mpi_sizez))
    fname = run_dir + 'distributions_2d/dist_1e4.' + str(ct)
    nrho_1e4 = np.fromfile(fname)
    nrho_1e4 = nrho_1e4.reshape((mpi_sizex, nzone*mpi_sizez))
    fname = run_dir + 'distributions_2d/dist_3e4.' + str(ct)
    nrho_3e4 = np.fromfile(fname)
    nrho_3e4 = nrho_3e4.reshape((mpi_sizex, nzone*mpi_sizez))

    kwargs = {"current_time": ct, "xl": 0, "xr": 1000, "zb": -250, "zt": 250}
    fname = run_dir + 'data/absB.gda'
    x, z, absB = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + 'data/Ay.gda'
    x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
    nx, = x.shape
    nz, = z.shape

    smime = math.sqrt(pic_info.mime)
    x *= smime # di -> de
    z *= smime

    xmin, xmax = 0, pic_info.lx_di * smime
    lz_de = pic_info.lz_di * smime
    zmin, zmax = -0.5 * lz_de, 0.5 * lz_de
    vmin, vmax = 1E2, 1E5

    fig = plt.figure(figsize=[8, 12])
    xs0, ys0 = 0.14, 0.74
    w1, h1 = 0.76, 0.205
    gap = 0.02
    ax1 = fig.add_axes([xs0, ys0, w1, h1])
    print(np.min(nrho_3e3), np.max(nrho_3e3))
    p1 = ax1.imshow(nrho_3e3.T, cmap=plt.cm.jet,
            extent=[xmin, xmax, zmin, zmax],
            aspect='auto', origin='lower',
            norm = LogNorm(vmin=vmin, vmax=vmax),
            interpolation='bicubic')
    ax1.tick_params(labelsize=16)
    ax1.tick_params(axis='x', labelbottom='off')
    nlevels = 20
    levels = np.linspace(np.min(Ay), np.max(Ay), nlevels)
    ax1.contour(x, z, Ay, colors='k', levels=levels, linewidths=0.5)
    ax1.set_ylabel(r'$z/d_e$', fontsize=20)
    fname1 = r'$N(2\times 10^3<E<4.5\times 10^3)$'
    ax1.text(0.02, 0.9, fname1, color='k', fontsize=20,
        bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='left', verticalalignment='center',
        transform=ax1.transAxes)

    ys = ys0 - gap - h1
    ax2 = fig.add_axes([xs0, ys, w1, h1])
    print(np.min(nrho_1e4), np.max(nrho_1e4))
    p2 = ax2.imshow(nrho_1e4.T, cmap=plt.cm.jet,
            extent=[xmin, xmax, zmin, zmax],
            aspect='auto', origin='lower',
            norm = LogNorm(vmin=vmin, vmax=vmax),
            interpolation='bicubic')
    ax2.tick_params(labelsize=16)
    ax2.tick_params(axis='x', labelbottom='off')
    ax2.contour(x, z, Ay, colors='black', levels=levels, linewidths=0.5)
    ax2.set_ylabel(r'$z/d_e$', fontsize=20)
    fname2 = r'$N(10^4/1.5<E<1.5\times 10^4)$'
    ax2.text(0.02, 0.9, fname2, color='black', fontsize=20,
        bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='left', verticalalignment='center',
        transform=ax2.transAxes)

    ys -= gap + h1
    ax3 = fig.add_axes([xs0, ys, w1, h1])
    print(np.min(nrho_3e4), np.max(nrho_3e4))
    p3 = ax3.imshow(nrho_3e4.T, cmap=plt.cm.jet,
            extent=[xmin, xmax, zmin, zmax],
            aspect='auto', origin='lower',
            norm = LogNorm(vmin=vmin, vmax=vmax),
            interpolation='bicubic')
    ax3.tick_params(labelsize=16)
    ax3.tick_params(axis='x', labelbottom='off')
    ax3.contour(x, z, Ay, colors='black', levels=levels, linewidths=0.5)
    ax3.set_ylabel(r'$z/d_e$', fontsize=20)
    fname3 = r'$N(2\times 10^4<E<4.5\times 10^4)$'
    ax3.text(0.02, 0.9, fname3, color='black', fontsize=20,
        bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='left', verticalalignment='center',
        transform=ax3.transAxes)

    xs = xs0 + w1 + 0.01
    cax = fig.add_axes([xs, ys, 0.02, 3*h1+2*gap])
    cbar = fig.colorbar(p3, cax=cax)
    cbar.ax.tick_params(labelsize=16)

    bmin, bmax = 10, 320

    ys -= gap + h1
    ax4 = fig.add_axes([xs0, ys, w1, h1])
    print(np.min(absB), np.max(absB))
    p4 = ax4.imshow(absB, cmap=plt.cm.jet,
            extent=[xmin, xmax, zmin, zmax],
            aspect='auto', origin='lower',
            vmin=bmin, vmax=bmax,
            interpolation='bicubic')
    ax4.tick_params(labelsize=16)
    ax4.contour(x, z, Ay, colors='black', levels=levels, linewidths=0.5)
    ax4.set_xlabel(r'$x/d_e$', fontsize=20)
    ax4.set_ylabel(r'$z/d_e$', fontsize=20)
    fname4 = r'$B$'
    ax4.text(0.02, 0.9, fname4, color='black', fontsize=20,
        bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='left', verticalalignment='center',
        transform=ax4.transAxes)

    xs = xs0 + w1 + 0.01
    cax1 = fig.add_axes([xs, ys, 0.02, h1])
    cbar1 = fig.colorbar(p4, cax=cax1)
    cbar1.ax.tick_params(labelsize=16)

    t_wci = ct * pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(t_wci) + '\Omega_{ci}^{-1}$'
    title += ' (frame: %d)' % ct
    ax1.set_title(title, fontdict=font, fontsize=24)

    fdir = '../img/radiation_cooling/nene_b/' + run_name + '/'
    mkdir_p(fdir)
    fname = fdir + 'nene_b_' + str(ct) + '.jpg'
    fig.savefig(fname, dpi=200)

    plt.close()
    # plt.show()


def contour_raidation(run_dir, run_name, tframe):
    """
    """
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    tframe_str = str(tframe + 9).zfill(4)
    fname = run_dir + 'map/' + 'totflux' + tframe_str + '.dat'
    tot_flux = np.genfromtxt(fname)
    fname = run_dir + 'map/' + 'polangl' + tframe_str + '.dat'
    pol_angl = np.genfromtxt(fname)
    fname = run_dir + 'map/' + 'polflux' + tframe_str + '.dat'
    pol_flux = np.genfromtxt(fname)
    kwargs = {"current_time": tframe, "xl": 0, "xr": pic_info.lx_di,
              "zb": -0.5 * pic_info.lz_di, "zt": 0.5 * pic_info.lz_di}
    fname = run_dir + 'data/Ay.gda'
    x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
    nx, = x.shape
    nz, = z.shape
    tot_flux[np.isnan(tot_flux)] = 0.0
    pol_flux[np.isnan(pol_flux)] = 0.0
    tot_flux[tot_flux < 1E-21] = 0.0
    pol_flux[pol_flux < 1E-21] = 0.0
    tot_flux = np.fliplr(tot_flux)
    pol_angl = np.fliplr(pol_angl)
    pol_flux = np.fliplr(pol_flux)
    tot_flux = np.flipud(tot_flux)
    pol_angl = np.flipud(pol_angl)
    pol_flux = np.flipud(pol_flux)
    smime = math.sqrt(pic_info.mime)
    x *= smime
    z *= smime
    lx_de = pic_info.lx_di * smime
    xmin, xmax = 0, lx_de
    lz_de = pic_info.lz_di * smime
    zmin, zmax = -0.5 * lz_de, 0.5 * lz_de
    fig = plt.figure(figsize=[8, 4])
    xs0, ys0 = 0.15, 0.18
    w1, h1 = 0.74, 0.75
    ax1 = fig.add_axes([xs0, ys0, w1, h1])
    vmin = np.min(tot_flux)
    vmax = np.max(tot_flux)
    print("Min and max of total flux: %e %e" % (vmin, vmax))
    vmin = 1E-21
    vmax = 1E-18
    p1 = ax1.imshow(tot_flux.T, cmap=plt.cm.Oranges,
                    extent=[xmin, xmax, zmin, zmax],
                    # vmin=vmin, vmax=vmax,
                    norm = LogNorm(vmin=vmin, vmax=vmax),
                    aspect='auto', origin='lower',
                    interpolation='bicubic')
    nlevels = 20
    levels = np.linspace(np.min(Ay), np.max(Ay), nlevels)
    ax1.contour(x, z, Ay, colors='k', levels=levels, linewidths=0.5)
    ax1.tick_params(labelsize=16)
    ax1.set_xlabel(r'$x/d_e$', fontsize=20)
    ax1.set_ylabel(r'$z/d_e$', fontsize=20)

    quiveropts = dict(color='black', headlength=0, pivot='middle', scale=0.3,
                      linewidth=.5, units='width', headwidth=1,
                      headaxislength=0)
    nx_rad, nz_rad = tot_flux.shape
    x_rad = np.linspace(0, lx_de, nx_rad)
    z_rad = np.linspace(-0.5 * lz_de, 0.5 * lz_de, nz_rad)
    X, Z = np.meshgrid(x_rad, z_rad)
    U = np.transpose(pol_flux*np.sin(pol_angl))
    V = np.transpose(pol_flux*np.cos(pol_angl))
    s = 1
    Q = ax1.quiver(X[::s, ::s], Z[::s, ::s], U[::s, ::s], V[::s, ::s],
                   **quiveropts)

    xs = xs0 + w1 + 0.01
    cax1 = fig.add_axes([xs, ys0, 0.02, h1])
    cbar1 = fig.colorbar(p1, cax=cax1)
    cbar1.ax.tick_params(labelsize=16)

    fdir = '../img/radiation_cooling/pa_pd_flux/' + run_name + '/'
    mkdir_p(fdir)
    fname = fdir + 'pa_pd_flux_' + str(tframe) + '.jpg'
    fig.savefig(fname, dpi=200)
    plt.close()
    # plt.show()


def get_cmd_args():
    """Get command line arguments
    """
    default_run_name = 'sigma4E4_bg00_rad_vthe100_cool100'
    default_run_dir = ('/net/scratch2/xiaocanli/vpic_radiation/reconnection/' +
                       'grizzly/cooling_scaling_16000_8000/' +
                       'sigma4E4_bg00_rad_vthe100_cool100/')
    parser = argparse.ArgumentParser(description='Radiation cooling analysis')
    parser.add_argument('--species', action="store", default='e',
                        help='particle species')
    parser.add_argument('--run_dir', action="store", default=default_run_dir,
                        help='run directory')
    parser.add_argument('--run_name', action="store", default=default_run_name,
                        help='run name')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_cmd_args()
    run_name = args.run_name
    run_dir = args.run_dir
    species = args.species
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    # plot_magentic_field_one_frame(run_dir, run_name, 30)
    # plot_particle_distribution_2d(run_dir, run_name, 30)
    # contour_raidation(run_dir, run_name, 60)
    plot_nrho_momentum(run_dir, run_name, 50)
    def processInput(job_id):
        print job_id
        tframe = job_id
        # plot_magentic_field_one_frame(run_dir, run_name, tframe)
        # plot_particle_distribution_2d(run_dir, run_name, tframe)
        contour_raidation(run_dir, run_name, tframe)
    cts = range(pic_info.ntf)
    tratio = pic_info.particle_interval / pic_info.fields_interval
    # cts = range(0, pic_info.ntf, tratio)
    ncores = multiprocessing.cpu_count()
    # Parallel(n_jobs=ncores)(delayed(processInput)(ct) for ct in cts)
    # for ct in cts:
    #     print("Time frame: %d" % ct)
    #     contour_raidation(run_dir, run_name, ct)
