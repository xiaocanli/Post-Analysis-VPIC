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
import fitting_funcs
import spectrum_fitting as spect_fit
from compression import calc_vexb, calc_ppara_pperp_pscalar
from contour_plots import find_closest, plot_2d_contour, read_2d_fields
from dolointerpolation import MultilinearInterpolator
from energy_conversion import read_data_from_json, read_jdote_data
from particle_compression import read_fields, read_hydro_velocity_density
from runs_name_path import ApJ_long_paper_runs
from serialize_json import data_to_json, json_to_data
from shell_functions import mkdir_p

style.use(['seaborn-white', 'seaborn-paper', 'seaborn-ticks'])
# style.use(['seaborn-white', 'seaborn-paper'])
# style.use(['classic'])
# rc('font', **{'family': 'serif', 'serif': ["Times", "Palatino", "serif"]})
# rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc("font", family="Times New Roman")
# rc("font", family="Computer Modern")
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

def plot_ne_velocity(pic_info, root_dir, run_name, current_time):
    """
    Plot electron number density and ExB drift velocity

    Args:
        pic_info: namedtuple for the PIC simulation information.
        root_dir: simulation root directory
        current_time: current time frame.
    """
    print("Time frame: %d" % current_time)
    smime = math.sqrt(pic_info.mime)
    dx = pic_info.dx_di * smime
    dz = pic_info.dz_di * smime
    lx_de = pic_info.lx_di * smime
    lz_de = pic_info.lz_di * smime
    kwargs = {"current_time": current_time, "xl": 0, "xr": lx_de,
              "zb": -0.5 * lx_de, "zt": 0.5 * lx_de}
    x, z, vx, vy, vz = calc_vexb(pic_info, root_dir, current_time)
    fname = root_dir + "data/ne.gda"
    x, z, ne = read_2d_fields(pic_info, fname, **kwargs)
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
    w0, h0 = 0.73, 0.2
    xs0, ys0 = 0.14, 0.98 - h0
    vgap, hgap = 0.03, 0.04

    def plot_one_field(fdata, ax, text, text_color, label_bottom='on',
                       label_left='on', ylabel=False, vmin=0, vmax=10,
                       colormap=plt.cm.seismic, xs=xs0, ys=ys0, ay_color='k',
                       color_bar=False):
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
        if color_bar:
            xs1 = xs + w0 * 1.02
            w1 = w0 * 0.04
            cax = fig.add_axes([xs1, ys, w1, h0])
            cbar = fig.colorbar(p1, cax=cax)
            cbar.ax.tick_params(labelsize=16)
            return (p1, cbar)
        else:
            return p1

    fig = plt.figure(figsize=[7, 8])
    xs, ys = xs0, ys0
    ax1 = fig.add_axes([xs, ys, w0, h0])
    text1 = r'$n_e$'
    print("min and max of electron density: %f %f" % (np.min(ne), np.max(ne)))
    nmin, nmax = 0.5, 3.0
    p1, cbar1 = plot_one_field(ne, ax1, text1, 'w', label_bottom='off',
                               label_left='on', ylabel=True, vmin=nmin,
                               vmax=nmax, colormap=plt.cm.viridis, xs=xs, ys=ys,
                               ay_color='w', color_bar=True)
    cbar1.set_ticks(np.arange(nmin, nmax + 0.5, 0.5))

    ys -= h0 + vgap
    ax2 = fig.add_axes([xs, ys, w0, h0])
    vmin, vmax = -1.0, 1.0
    text2 = r'$v_{Ex}$'
    print("min and max of vx: %f %f" % (np.min(vx), np.max(vx)))
    p2 = plot_one_field(vx, ax2, text2, 'k', label_bottom='off',
                        label_left='on', ylabel=True, vmin=vmin, vmax=vmax,
                        colormap=plt.cm.seismic, xs=xs, ys=ys)
    xs1 = xs + w0 * 1.02
    w1 = w0 * 0.04
    ys1 = ys - 2 * (h0 + vgap)
    h1 = 3 * h0 + 2 * vgap
    cax2 = fig.add_axes([xs1, ys1, w1, h1])
    cbar2 = fig.colorbar(p2, cax=cax2)
    cbar2.ax.tick_params(labelsize=16)

    ys -= h0 + vgap
    ax3 = fig.add_axes([xs, ys, w0, h0])
    text3 = r'$v_{Ey}$'
    print("min and max of vy: %f %f" % (np.min(vy), np.max(vy)))
    p3 = plot_one_field(vy, ax3, text3, 'k', label_bottom='off',
                        label_left='on', ylabel=True, vmin=vmin, vmax=vmax,
                        colormap=plt.cm.seismic, xs=xs, ys=ys)

    ys -= h0 + vgap
    ax4 = fig.add_axes([xs, ys, w0, h0])
    text4 = r'$v_{Ez}$'
    print("min and max of vz: %f %f" % (np.min(vz), np.max(vz)))
    p3 = plot_one_field(vz, ax4, text4, 'k', label_bottom='on',
                        label_left='on', ylabel=True, vmin=vmin, vmax=vmax,
                        colormap=plt.cm.seismic, xs=xs, ys=ys)
    ax4.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)

    nz, nx = vx.shape
    fig1 = plt.figure(figsize=[7, 5])
    w1, h1 = 0.84, 0.8
    xs, ys = 0.12, 0.96 - h1
    ax1 = fig1.add_axes([xs, ys, w1, h1])
    x1 = nx // 4
    x2 = 70 * nx // 200
    ax1.plot(x[x1:x2], vx[nz/2, x1:x2], linewidth=2, label=r'$v_{Ex}$')
    # ax1.plot(x, vy[nz/2, :], linewidth=2, label=r'$v_{Ey}$')
    # ax1.plot(x, vz[nz/2, :], linewidth=2, label=r'$v_{Ez}$')
    ax1.legend(loc=3, prop={'size': 16}, ncol=2,
               shadow=False, fancybox=False, frameon=False)
    ax1.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    ax1.set_xlim([x[x1], x[x2]])
    ax1.tick_params(labelsize=16)

    fdir = '../img/img_apjl/ne_velocity2/' + run_name + '/'
    mkdir_p(fdir)
    # fname = fdir + 'nrho_vel_' + str(current_time) + '.jpg'
    # fig.savefig(fname, dpi=200)
    fname = fdir + 'nrho_vel_' + str(current_time) + '.pdf'
    # fig.savefig(fname)
    # plt.close()
    plt.show()


def plot_compression_time_electron(pic_info):
    """Plot the time evolution of compression-related terms for electron

    Args:
        pic_info: namedtuple for the PIC simulation information
        run_name: simulation run name
    """
    run_name = 'mime25_beta002_guide00_frequent_dump'
    tfields = pic_info.tfields
    fdir = '../data/compression/'
    cdata_name = fdir + 'compression_' + run_name + '_e.json'
    cdata_e = read_data_from_json(cdata_name)
    jdote_name = '../data/jdote_data/jdote_' + run_name + '_e.json'
    jdote_e = read_data_from_json(jdote_name)

    colors = colors_Set1_9[:5] + colors_Set1_9[6:] # remove yellow color

    fig = plt.figure(figsize=[8, 6])
    w1, h1 = 0.83, 0.32
    xs, ys = 0.96 - w1, 0.80 - h1
    ax = fig.add_axes([xs, ys, w1, h1])
    ax.set_prop_cycle('color', colors)
    label1 = r'$-p_e\nabla\cdot\boldsymbol{v}_E$'
    label2 = r'$-(p_{e\parallel} - p_{e\perp})b_ib_j\sigma_{ij}$'
    label3 = r'$\nabla\cdot(\mathcal{P}\cdot\mathbf{u})$'
    label4 = label3 + label1 + label2
    label5 = r'$\mathbf{u}\cdot(\nabla\cdot\mathcal{P})$'
    label6 = r'$\boldsymbol{j}_{e\perp}\cdot\boldsymbol{E}_\perp$'
    label7 = r'$n_em_e(d\boldsymbol{u}_e/dt)\cdot\boldsymbol{v}_E$'
    label8 = r'$\boldsymbol{j}_{e-\text{agy}}\cdot\boldsymbol{E}_\perp$'
    label61 = label6 + r'$ - $' + label7 + r'$ - $' + label8

    fdir = '../data/jpolar_dote/'
    fname = fdir + 'jpolar_dote_' + run_name + '_e.dat'
    jpolar_dote = np.fromfile(fname)
    jpolar_dote[-1] = jpolar_dote[-2] # remove boundary spikes

    p1 = ax.plot(tfields, cdata_e.pdiv_uperp_usingle_exb, linewidth=2, label=label1)
    p2 = ax.plot(tfields, cdata_e.pshear_perp_usingle_exb, linewidth=2, label=label2)
    p4 = ax.plot(tfields, jpolar_dote, linewidth=2, label=label7)

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

    text1 = r'$\beta_e=0.02, B_g=0$'
    ax.text(0.98, 0.8, text1, color='k', fontsize=20,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='right', verticalalignment='bottom',
            transform=ax.transAxes)

    run_name = 'mime25_beta002_guide05_frequent_dump'
    tfields = pic_info.tfields
    fdir = '../data/compression/'
    cdata_name = fdir + 'compression_' + run_name + '_e.json'
    cdata_e = read_data_from_json(cdata_name)
    jdote_name = '../data/jdote_data/jdote_' + run_name + '_e.json'
    jdote_e = read_data_from_json(jdote_name)

    fdir = '../data/jpolar_dote/'
    fname = fdir + 'jpolar_dote_' + run_name + '_e.dat'
    jpolar_dote = np.fromfile(fname)
    jpolar_dote[-1] = jpolar_dote[-2] # remove boundary spikes

    ys -= h1 + 0.04
    ax1 = fig.add_axes([xs, ys, w1, h1])
    ax1.set_prop_cycle('color', colors)
    p1 = ax1.plot(tfields, cdata_e.pdiv_uperp_usingle_exb, linewidth=2, label=label1)
    p2 = ax1.plot(tfields, cdata_e.pshear_perp_usingle_exb, linewidth=2, label=label2)
    p4 = ax1.plot(tfields, jpolar_dote, linewidth=2, label=label7)

    fdata = jdote_e.jqnuperp_dote
    fdata -= jpolar_dote
    fdata -= jdote_e.jagy_dote
    fdata[0] = fdata[1] # remove boundary spikes
    p3 = ax1.plot(tfields, fdata, linewidth=2, label=label61, color=colors[1], linestyle='-',
                 marker='o', markersize=5)
    p12 = ax1.plot(tfields, cdata_e.pdiv_uperp_usingle_exb + cdata_e.pshear_perp_usingle_exb,
                  linewidth=2, label=label1 + label2, color='k')
    p5 = ax1.plot(tfields, jdote_e.jagy_dote, linewidth=2, label=label8)
    ax1.set_xlabel(r'$t\Omega_{ci}$', fontsize=20)
    ax1.set_ylabel(r'$d\varepsilon_e/dt$', fontsize=20)
    ax1.tick_params(labelsize=16)
    tmax = min(np.max(pic_info.tfields), 600)
    ax1.set_xlim([0, 600])

    text1 = r'$\beta_e=0.02, B_g=0.5B_0$'
    ax1.text(0.98, 0.8, text1, color='k', fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='right', verticalalignment='bottom',
             transform=ax1.transAxes)

    fname = '../img/img_apjl/comp_bg00_bg05.eps'
    fig.savefig(fname)

    plt.show()


def calc_compressional_energization_terms(pic_info, root_dir, tframe,
                                          xl, xr, zb, zt):
    """
    """
    smime = math.sqrt(pic_info.mime)
    dx = pic_info.dx_di * smime
    dz = pic_info.dz_di * smime
    kwargs = {"current_time": tframe, "xl": xl, "xr": xr,
              "zb": zb, "zt": zt}
    fname = root_dir + "data/bx.gda"
    x, z, bx = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/by.gda"
    x, z, by = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/bz.gda"
    x, z, bz = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/ex.gda"
    x, z, ex = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/ey.gda"
    x, z, ey = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/ez.gda"
    x, z, ez = read_2d_fields(pic_info, fname, **kwargs)

    ib2 = 1.0 / (bx**2 + by**2 + bz**2)
    vx = (ey * bz - ez * by) * ib2
    vy = (ez * bx - ex * bz) * ib2
    vz = (ex * by - ey * bx) * ib2

    fname = root_dir + "data/vex.gda"
    x, z, vex = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/vey.gda"
    x, z, vey = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/vez.gda"
    x, z, vez = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/ne.gda"
    x, z, ne = read_2d_fields(pic_info, fname, **kwargs)
    vdot_b = vex * bx + vey * by + vez * bz
    jpara_dote = -ne * vdot_b * ib2 * (ex * bx + ey * by + ez * bz)
    jperp_dote = -ne * (vex * ex + vey * ey + vez * ez) - jpara_dote

    del ex, ey, ez, vex, vey, vez

    speices = 'e'
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
    ppara = (pxx * bx**2 + pyy * by**2 + pzz * bz**2 +
             (pxy + pyx) * bx * by + (pxz + pzx) * bx * bz +
             (pyz + pzy) * by * bz) * ib2
    pperp = (pscalar * 3 - ppara) * 0.5

    divv = np.gradient(vx, dx, axis=1) + np.gradient(vz, dz, axis=0)
    pdivv = -pscalar * divv
    pshear = ((np.gradient(vx, dx, axis=1) - divv / 3.0) * bx**2 +
              (-divv / 3.0) * by**2 +
              (np.gradient(vz, dz, axis=0) - divv / 3.0) * bz**2 +
              np.gradient(vy, dx, axis=1) * bx * by +
              ((np.gradient(vz, dx, axis=1) +
                np.gradient(vx, dz, axis=0))) * bx * bz +
              np.gradient(vy, dz, axis=0) * by * bz) * (pperp - ppara) * ib2
    dv = dx * dz
    print("pdivv, pshear, jperp_dote: %f %f %f" % (np.sum(pdivv) * dv,
                                                   np.sum(pshear) * dv,
                                                   np.sum(jperp_dote) * dv))
    
    jpara_dote_cum = np.cumsum(np.sum(jpara_dote, axis=0)) * dv
    jperp_dote_cum = np.cumsum(np.sum(jperp_dote, axis=0)) * dv
    pdivv_cum = np.cumsum(np.sum(pdivv, axis=0)) * dv
    pshear_cum = np.cumsum(np.sum(pshear, axis=0)) * dv
    del bx, by, bz, ib2, ppara, pperp, vx, vy, vz

    fdata_2d = {"jpara_dote": jpara_dote, "jperp_dote": jperp_dote,
                "pdivv": pdivv, "pshear": pshear}
    fdata_cumsum = {"jpara_dote_cum": jpara_dote_cum,
                    "jperp_dote_cum": jperp_dote_cum,
                    "pdivv_cum": pdivv_cum, "pshear_cum": pshear_cum}

    return (fdata_2d, fdata_cumsum)


def plot_compression_shear_2d(pic_info, root_dir, run_name, tframe):
    """
    Plot 2D contour of compressional and shear energization terms

    Args:
        pic_info: namedtuple for the PIC simulation information.
        root_dir: simulation root directory
        tframe: current time frame.
    """
    print("Time frame: %d" % tframe)
    smime = math.sqrt(pic_info.mime)
    lx_di = pic_info.lx_di
    lz_di = pic_info.lz_di
    xl, xr = 0, lx_di
    zb, zt = -20, 20
    fdata_2d, fdata_cumsum = calc_compressional_energization_terms(
            pic_info, root_dir, tframe, xl, xr, zb, zt)
    jpara_dote = fdata_2d["jpara_dote"]
    jperp_dote = fdata_2d["jperp_dote"]
    pdivv = fdata_2d["pdivv"]
    pshear = fdata_2d["pshear"]
    jpara_dote_cum = fdata_cumsum["jpara_dote_cum"]
    jperp_dote_cum = fdata_cumsum["jperp_dote_cum"]
    pdivv_cum = fdata_cumsum["pdivv_cum"]
    pshear_cum = fdata_cumsum["pshear_cum"]
    wpe_wce = pic_info.dtwce / pic_info.dtwpe
    va = wpe_wce / math.sqrt(pic_info.mime)  # Alfven speed
    enorm = va**2

    jpara_dote /= enorm
    jperp_dote /= enorm
    pdivv /= enorm
    pshear /= enorm

    fname = root_dir + "data/Ay.gda"
    kwargs = {"current_time": tframe, "xl": xl, "xr": xr,
              "zb": zb, "zt": zt}
    x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)

    xmin, xmax = np.min(x), np.max(x)
    zmin, zmax = np.min(z), np.max(z)

    w0, h0 = 0.72, 0.155
    xs0, ys0 = 0.14, 0.98 - h0
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
        ax.text(0.98, 0.85, text, color=text_color, fontsize=16,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='right', verticalalignment='center',
                transform=ax.transAxes)
        return p1

    # smooth the data
    sigma = 3
    jpara_dote = gaussian_filter(jpara_dote, sigma)
    jperp_dote = gaussian_filter(jperp_dote, sigma)
    pdivv = gaussian_filter(pdivv, sigma)
    pshear = gaussian_filter(pshear, sigma)

    dmax = 0.03
    dmin = -dmax

    colors = colors_Set1_9
    fig = plt.figure(figsize=[7, 8])
    xs, ys = xs0, ys0
    ax1 = fig.add_axes([xs, ys, w0, h0])
    text1 = r'$\boldsymbol{j}_{e\parallel}\cdot\boldsymbol{E}_\parallel$'
    print("Min and max of jpara_dote: %f %f" % (np.min(jpara_dote),
                                                np.max(jpara_dote)))
    p1 = plot_one_field(jpara_dote, ax1, text1, colors[2], label_bottom='off',
                        label_left='on', ylabel=True, vmin=dmin, vmax=dmax,
                        colormap=plt.cm.seismic, xs=xs, ys=ys, ay_color='k')
    xs1 = xs + w0 * 1.02
    w1 = w0 * 0.04
    ys1 = ys - 3 * (h0 + vgap)
    h1 = 4 * h0 + 3 * vgap
    cax = fig.add_axes([xs1, ys1, w1, h1])
    cbar1 = fig.colorbar(p1, cax=cax)
    cbar1.ax.tick_params(labelsize=16)
    cbar1.set_ticks(np.arange(dmin, dmax + dmax, dmax))

    ys -= h0 + vgap
    ax2 = fig.add_axes([xs, ys, w0, h0])
    text2 = r'$\boldsymbol{j}_{e\perp}\cdot\boldsymbol{E}_\perp$'
    print("Min and max of jperp_dote: %f %f" % (np.min(jperp_dote),
                                                np.max(jperp_dote)))
    p2 = plot_one_field(jperp_dote, ax2, text2, 'k', label_bottom='off',
                        label_left='on', ylabel=True, vmin=dmin, vmax=dmax,
                        colormap=plt.cm.seismic, xs=xs, ys=ys, ay_color='k')

    ys -= h0 + vgap
    ax3 = fig.add_axes([xs, ys, w0, h0])
    text3 = r'$-p_e\nabla\cdot\boldsymbol{v}_E$'
    print("Min and max of pdivv: %f %f" % (np.min(pdivv), np.max(pdivv)))
    p3 = plot_one_field(pdivv, ax3, text3, colors[0], label_bottom='off',
                        label_left='on', ylabel=True, vmin=dmin, vmax=dmax,
                        colormap=plt.cm.seismic, xs=xs, ys=ys, ay_color='k')

    ys -= h0 + vgap
    ax4 = fig.add_axes([xs, ys, w0, h0])
    text4 = r'$-(p_{e\parallel}-p_{e\perp})b_ib_j\sigma_{ij}$'
    print("Min and max of pshear: %f %f" % (np.min(pshear), np.max(pshear)))
    p4 = plot_one_field(pshear, ax4, text4, colors[1], label_bottom='off',
                        label_left='on', ylabel=True, vmin=dmin, vmax=dmax,
                        colormap=plt.cm.seismic, xs=xs, ys=ys, ay_color='k')

    ys -= h0 + vgap
    ax5 = fig.add_axes([xs, ys, w0, h0])
    ax5.set_prop_cycle('color', colors)
    ax5.plot(x, jperp_dote_cum, linewidth=2, color='k')
    ax5.plot(x, pdivv_cum, linewidth=2, color=colors[0])
    ax5.plot(x, pshear_cum, linewidth=2, color=colors[1])
    ax5.plot(x, jpara_dote_cum, linewidth=2, color=colors[2])
    ax5.plot([0, lx_di], [0, 0], linestyle='--', color='k')
    ax5.tick_params(labelsize=16)
    ax5.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    text5 = 'Cumulative sum along $x$'
    ax5.text(0.5, 0.85, text5, color='k', fontsize=16,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='center', verticalalignment='center',
             transform=ax5.transAxes)

    fdir = '../img/img_apjl/comp_terms2/' + run_name + '/'
    mkdir_p(fdir)
    # fname = fdir + 'comp_terms_' + str(tframe) + '.jpg'
    # fig.savefig(fname, dpi=200)
    fname = fdir + 'comp_terms_' + str(tframe) + '.pdf'
    # fig.savefig(fname)
    # plt.close()

    plt.show()


def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return (idx, array[idx])


def compression_ratio_high_energy(species):
    """
    """
    run_names = ['mime25_beta002_guide00_frequent_dump',
                 'mime25_beta002_guide02_frequent_dump',
                 'mime25_beta002_guide05_frequent_dump',
                 'mime25_beta002_guide10_frequent_dump',
                 'mime25_beta008_guide00_frequent_dump',
                 'mime25_beta032_guide00_frequent_dump']
    nrun = len(run_names)
    nbins = 60
    drange = [[1, 1.1], [0, 1]]
    picinfo_fname = '../data/pic_info/pic_info_' + run_names[0] + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    ebins = np.logspace(-4, 2, nbins + 1)
    ratio_high_ene = np.zeros((nrun, 6, pic_info.ntp-1))
    if species == 'e':
        charge = r'$-e$'
        vth = pic_info.vthe
    else:
        charge = r'$e$'
        vth = pic_info.vthi
    colors = colors_Set1_9[:5] + colors_Set1_9[6:] # remove yellow color
    gama = 1.0 / math.sqrt(1.0 - 3 * vth**2)
    eth = gama - 1.0
    for irun, run_name in enumerate(run_names):
        fdir = '../data/particle_compression/' + run_name + '/'
        picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        for ct in range(pic_info.ntp-1):
            tindex = (ct + 1) * pic_info.particle_interval
            fname = fdir + 'hists_' + species + '.' + str(tindex) + '.all'
            fdata = np.fromfile(fname)
            sz, = fdata.shape
            nvar = sz / nbins
            fdata = fdata.reshape((nvar, nbins))
            hist_de_para = fdata[0, :]
            hist_de_perp = fdata[1, :]
            hist_pdivv = fdata[2, :]
            hist_pdiv_vperp = fdata[3, :]
            hist_pshear = fdata[4, :]
            hist_ptensor_dv = fdata[5, :]
            hist_de_dudt = fdata[6, :]
            hist_de_cons_mu = fdata[7, :]
            hist_div_ptensor_vperp = fdata[8, :]
            hist_div_pperp_vperp = fdata[9, :]
            hist_nptl = fdata[-1, :]
            eindex10, eth10 = find_nearest(ebins, eth * 20)
            de_para10 = np.sum(hist_de_para[eindex10:])
            de_perp10 = np.sum(hist_de_perp[eindex10:])
            de_tot10 = de_para10 + de_perp10
            pdivv10 = np.sum(hist_pdiv_vperp[eindex10:])
            pshear10 = np.sum(hist_pshear[eindex10:])
            ptensor_dv10 = np.sum(hist_ptensor_dv[eindex10:])
            de_dudt10 = np.sum(hist_de_dudt[eindex10:])
            ratio_high_ene[irun, 0, ct] = de_para10 / de_tot10
            ratio_high_ene[irun, 1, ct] = de_perp10 / de_tot10
            ratio_high_ene[irun, 2, ct] = pdivv10 / de_tot10
            ratio_high_ene[irun, 3, ct] = pshear10 / de_tot10
            ratio_high_ene[irun, 4, ct] = ptensor_dv10 / de_tot10
            ratio_high_ene[irun, 5, ct] = de_dudt10 / de_tot10

    fig = plt.figure(figsize=[7, 5])
    w1, h1 = 0.84, 0.8
    xs, ys = 0.12, 0.96 - h1
    ax1 = fig.add_axes([xs, ys, w1, h1])
    colors = colors_Set1_9[:5] + colors_Set1_9[6:] # remove yellow color
    ax1.set_prop_cycle('color', colors)
    markers = ['o', 's', '^', 'd', 'h', '^']
    runs = ['B1/G1', 'G2', 'G3', 'G4', 'B2']
    ntp = pic_info.ntp
    tparticles = np.arange(1, ntp-1) * pic_info.dt_particles
    for irun in range(nrun-1):
        ax1.plot(tparticles[:12],
                 # ratio_high_ene[irun, 4, :12],
                 ratio_high_ene[irun, 2, :12] + ratio_high_ene[irun, 3, :12],
                 linewidth=2, linestyle='None', marker=markers[irun],
                 markersize=10, label=runs[irun])
    ax1.set_xlim([0, 650])
    ax1.set_ylim([-2, 3])
    ax1.plot(ax1.get_xlim(), [1, 1], color='k', linestyle='--')
    ax1.plot(ax1.get_xlim(), [0.5, 0.5], color='k', linestyle='--')
    ax1.plot(ax1.get_xlim(), [0, 0], color='k', linestyle='--')
    ax1.tick_params(labelsize=16)
    ax1.set_xlabel(r'$t\Omega_{ci}$', fontdict=font, fontsize=20)
    text1 = r'$\boldsymbol{j}_e\cdot\boldsymbol{E}$'
    text2 = r'$-p\nabla\cdot\boldsymbol{v}_E$'
    text3 = r'$-(p_{\parallel}-p_{\perp})b_ib_j\sigma_{ij}$'
    texty = r'$($' + text2 + text3 + r'$)/$' + text1
    ax1.set_ylabel(texty, fontdict=font, fontsize=20)
    ax1.legend(loc=3, prop={'size': 16}, ncol=2,
               shadow=False, fancybox=False, frameon=True)

    fdir = '../img/img_apjl/comp_terms2/'
    mkdir_p(fdir)
    fname = fdir + 'comp_terms_runs.pdf'
    fig.savefig(fname)

    plt.show()


def compression_ratio_high_energy_test(species):
    """
    """
    run_names = ['mime25_beta002_guide00_frequent_dump',
                 'mime25_beta002_guide02_frequent_dump',
                 'mime25_beta002_guide05_frequent_dump',
                 'mime25_beta002_guide10_frequent_dump',
                 'mime25_beta008_guide00_frequent_dump',
                 'mime25_beta032_guide00_frequent_dump']
    nrun = len(run_names)
    nbins = 60
    drange = [[1, 1.1], [0, 1]]
    picinfo_fname = '../data/pic_info/pic_info_' + run_names[0] + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    ebins = np.logspace(-4, 2, nbins + 1)
    ratio_high_ene = np.zeros((nrun, 6, pic_info.ntp-1))
    jdote_norm = np.zeros((nrun, pic_info.ntp-1))
    if species == 'e':
        charge = r'$-e$'
        vth = pic_info.vthe
    else:
        charge = r'$e$'
        vth = pic_info.vthi
    colors = colors_Set1_9[:5] + colors_Set1_9[6:] # remove yellow color
    gama = 1.0 / math.sqrt(1.0 - 3 * vth**2)
    eth = gama - 1.0
    for irun, run_name in enumerate(run_names):
        fdir = '../data/particle_compression/' + run_name + '/'
        picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        jdote_name = '../data/jdote_data/jdote_' + run_name + '_e.json'
        jdote_e = read_data_from_json(jdote_name)
        jdote_tot = jdote_e.jqnupara_dote + jdote_e.jqnuperp_dote
        jdote_max = np.max(jdote_tot)
        for ct in range(pic_info.ntp-1):
            tindex = (ct + 1) * pic_info.particle_interval
            fname = fdir + 'hists_' + species + '.' + str(tindex) + '.all'
            fdata = np.fromfile(fname)
            sz, = fdata.shape
            nvar = sz / nbins
            fdata = fdata.reshape((nvar, nbins))
            hist_de_para = fdata[0, :]
            hist_de_perp = fdata[1, :]
            hist_pdivv = fdata[2, :]
            hist_pdiv_vperp = fdata[3, :]
            hist_pshear = fdata[4, :]
            hist_ptensor_dv = fdata[5, :]
            hist_de_dudt = fdata[6, :]
            hist_de_cons_mu = fdata[7, :]
            hist_div_ptensor_vperp = fdata[8, :]
            hist_div_pperp_vperp = fdata[9, :]
            hist_nptl = fdata[-1, :]
            eindex10, eth10 = find_nearest(ebins, eth * 10)
            de_para10 = np.sum(hist_de_para[eindex10:])
            de_perp10 = np.sum(hist_de_perp[eindex10:])
            de_tot10 = de_para10 + de_perp10
            pdivv10 = np.sum(hist_pdiv_vperp[eindex10:])
            pshear10 = np.sum(hist_pshear[eindex10:])
            ptensor_dv10 = np.sum(hist_ptensor_dv[eindex10:])
            de_dudt10 = np.sum(hist_de_dudt[eindex10:])
            ratio_high_ene[irun, 0, ct] = de_para10 / de_tot10
            ratio_high_ene[irun, 1, ct] = de_perp10 / de_tot10
            ratio_high_ene[irun, 2, ct] = pdivv10 / de_tot10
            ratio_high_ene[irun, 3, ct] = pshear10 / de_tot10
            ratio_high_ene[irun, 4, ct] = ptensor_dv10 / de_tot10
            ratio_high_ene[irun, 5, ct] = de_dudt10 / de_tot10
            jdote_norm[irun, ct] = np.sum(hist_de_para + hist_de_perp) / jdote_max

    fig = plt.figure(figsize=[7, 5])
    w1, h1 = 0.7, 0.82
    xs, ys = 0.15, 0.96 - h1
    ax1 = fig.add_axes([xs, ys, w1, h1])
    # colors = colors_Set1_9[:5] + colors_Set1_9[6:] # remove yellow color
    # ax1.set_prop_cycle('color', colors)
    markers = ['o', 'v', '^', 'd', 'h', '^']
    runs = ['B1/G1', 'G2', 'G3', 'G4', 'B2']
    ntp = pic_info.ntp
    maxt = 11
    tparticles = np.arange(1, ntp-1) * pic_info.dt_particles
    cmap = plt.cm.seismic
    # # extract all colors from the .jet map
    # cmaplist = [cmap(i) for i in range(cmap.N)]
    # # force the first color entry to be grey
    # cmaplist[0] = (.5,.5,.5,1.0)
    # # create the new map
    # cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    # define the bins and normalize
    tframes = np.linspace(1,maxt,maxt) * pic_info.dt_particles
    bounds = np.linspace(0.5,maxt+0.5,maxt+1) * pic_info.dt_particles
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    colors = cmap(np.linspace(0.5, maxt-0.5, maxt) / float(maxt), 1)

    for irun in range(nrun-1):
        index = np.ones((maxt,), dtype=np.int) * irun + 0.5
        comp_shear_ratio = (ratio_high_ene[irun, 2, :maxt] +
                            ratio_high_ene[irun, 3, :maxt])
        # ax1.scatter(index, comp_shear_ratio, c=tframes,
        #             cmap=cmap, norm=norm, marker=markers[irun],
        #             label=runs[irun], s=100)
        ax1.scatter(index, comp_shear_ratio, color=colors, edgecolor='black',
                    marker=markers[irun], label=runs[irun], s=100)
        # ax1.errorbar(index, comp_shear_ratio, xerr=jdote_norm[irun, :maxt],
        #              fmt='o')
    ax1.set_xlim([0, nrun-1])
    ax1.set_ylim([-1, 2.5])
    ax1.plot(ax1.get_xlim(), [1, 1], color='k', linestyle='--')
    ax1.plot(ax1.get_xlim(), [0.5, 0.5], color='k', linestyle='--')
    ax1.plot(ax1.get_xlim(), [0, 0], color='k', linestyle='--')
    ax1.set_xticks(np.linspace(0.5, nrun-1.5, nrun-1))
    ax1.set_xticklabels(runs)
    ax1.tick_params(labelsize=16)
    text1 = r'$\boldsymbol{j}_e\cdot\boldsymbol{E}$'
    text2 = r'$-p\nabla\cdot\boldsymbol{v}_E$'
    text3 = r'$-(p_{\parallel}-p_{\perp})b_ib_j\sigma_{ij}$'
    texty = r'$($' + text2 + text3 + r'$)/$' + text1
    ax1.set_ylabel(texty, fontdict=font, fontsize=20)
    # ax1.legend(loc=3, prop={'size': 16}, ncol=2,
    #            shadow=False, fancybox=False, frameon=True)

    xs = xs + w1 + 0.01
    cax = fig.add_axes([xs, ys, 0.03, h1])
    cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm,
                                     spacing='proportional', ticks=tframes,
                                     boundaries=bounds, format='%1i')
    cbar.set_label(r'$t\Omega_{ci}$', fontdict=font, fontsize=20)
    cbar.ax.tick_params(labelsize=16)

    fdir = '../img/img_apjl/comp_terms2/'
    mkdir_p(fdir)
    fname = fdir + 'comp_terms_runs.pdf'
    fig.savefig(fname)

    plt.show()


def compression_ratio_high_energy_test2(species):
    """
    """
    run_names = ['mime25_beta002_guide00_frequent_dump',
                 'mime25_beta002_guide02_frequent_dump',
                 'mime25_beta002_guide05_frequent_dump',
                 'mime25_beta002_guide10_frequent_dump',
                 'mime25_beta008_guide00_frequent_dump',
                 'mime25_beta032_guide00_frequent_dump']
    nrun = len(run_names)
    nbins = 60
    drange = [[1, 1.1], [0, 1]]
    picinfo_fname = '../data/pic_info/pic_info_' + run_names[0] + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    ebins = np.logspace(-4, 2, nbins + 1)
    ratio_high_ene = np.zeros((nrun, 6, pic_info.ntp-1))
    jdote_norm = np.zeros((nrun, pic_info.ntp-1))
    if species == 'e':
        charge = r'$-e$'
        vth = pic_info.vthe
    else:
        charge = r'$e$'
        vth = pic_info.vthi
    colors = colors_Set1_9[:5] + colors_Set1_9[6:] # remove yellow color
    gama = 1.0 / math.sqrt(1.0 - 3 * vth**2)
    eth = gama - 1.0
    for irun, run_name in enumerate(run_names):
        fdir = '../data/particle_compression/' + run_name + '/'
        picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        jdote_name = '../data/jdote_data/jdote_' + run_name + '_e.json'
        jdote_e = read_data_from_json(jdote_name)
        jdote_tot = jdote_e.jqnupara_dote + jdote_e.jqnuperp_dote
        jdote_max = np.max(jdote_tot)
        for ct in range(pic_info.ntp-1):
            tindex = (ct + 1) * pic_info.particle_interval
            fname = fdir + 'hists_' + species + '.' + str(tindex) + '.all'
            fdata = np.fromfile(fname)
            sz, = fdata.shape
            nvar = sz / nbins
            fdata = fdata.reshape((nvar, nbins))
            hist_de_para = fdata[0, :]
            hist_de_perp = fdata[1, :]
            hist_pdivv = fdata[2, :]
            hist_pdiv_vperp = fdata[3, :]
            hist_pshear = fdata[4, :]
            hist_ptensor_dv = fdata[5, :]
            hist_de_dudt = fdata[6, :]
            hist_de_cons_mu = fdata[7, :]
            hist_div_ptensor_vperp = fdata[8, :]
            hist_div_pperp_vperp = fdata[9, :]
            hist_nptl = fdata[-1, :]
            eindex10, eth10 = find_nearest(ebins, eth * 10)
            de_para10 = np.sum(hist_de_para[eindex10:])
            de_perp10 = np.sum(hist_de_perp[eindex10:])
            de_tot10 = de_para10 + de_perp10
            pdivv10 = np.sum(hist_pdiv_vperp[eindex10:])
            pshear10 = np.sum(hist_pshear[eindex10:])
            ptensor_dv10 = np.sum(hist_ptensor_dv[eindex10:])
            de_dudt10 = np.sum(hist_de_dudt[eindex10:])
            ratio_high_ene[irun, 0, ct] = de_para10
            ratio_high_ene[irun, 1, ct] = de_perp10
            ratio_high_ene[irun, 2, ct] = pdivv10
            ratio_high_ene[irun, 3, ct] = pshear10
            ratio_high_ene[irun, 4, ct] = ptensor_dv10
            ratio_high_ene[irun, 5, ct] = de_dudt10

    ntp = pic_info.ntp
    maxt = 12
    tparticles = np.arange(1, ntp-1) * pic_info.dt_particles
    colors = colors_Set1_9[:5] + colors_Set1_9[6:] # remove yellow color
    text1 = r'$\boldsymbol{j}_e\cdot\boldsymbol{E}$'
    text2 = r'$-p\nabla\cdot\boldsymbol{v}_E$'
    text3 = r'$-(p_{\parallel}-p_{\perp})b_ib_j\sigma_{ij}$'
    locs = [1, 3, 1, 1, 1, 3]
    for irun in range(nrun):
        fig = plt.figure(figsize=[7, 5])
        w1, h1 = 0.8, 0.82
        xs, ys = 0.15, 0.96 - h1
        ax1 = fig.add_axes([xs, ys, w1, h1])

        jdote_tot = ratio_high_ene[irun, 0, :maxt] + ratio_high_ene[irun, 1, :maxt]
        comp_shear = ratio_high_ene[irun, 2, :maxt] + ratio_high_ene[irun, 3, :maxt]
        ax1.plot(tparticles[:maxt], jdote_tot, linewidth=2, color='k',
                 label='Total energization')
        ax1.plot(tparticles[:maxt], ratio_high_ene[irun, 2, :maxt],
                 linewidth=2, color=colors[0], label='Compression',
                 marker='o', markersize=10)
        ax1.plot(tparticles[:maxt], ratio_high_ene[irun, 3, :maxt],
                 linewidth=2, color=colors[1], label='Shear',
                 marker='s', markersize=10)
        ax1.plot(tparticles[:maxt], comp_shear, linewidth=2,
                 linestyle='--', color='k', label='Compression + Shear')

        ax1.set_xlim([tparticles[0], tparticles[maxt-1]])
        ax1.plot(ax1.get_xlim(), [0, 0], color='k', linestyle='--')
        ax1.tick_params(labelsize=16)
        ax1.set_xlabel(r'$t\Omega_{ci}$', fontsize=20)
        ax1.set_ylabel('Energization', fontdict=font, fontsize=20)
        ax1.legend(loc=locs[irun], prop={'size': 16}, ncol=1,
                   shadow=False, fancybox=False, frameon=False)

        fdir = '../img/img_apjl/comp_terms2/'
        mkdir_p(fdir)
        fname = fdir + 'comp_terms_run' + str(irun) + '.pdf'
        fig.savefig(fname)

    plt.show()


def plot_compression_shear_g1(pic_info, root_dir, run_name, tframe):
    """
    Plot 2D contour of compressional and shear energization terms for
    the run without guide field

    Args:
        pic_info: namedtuple for the PIC simulation information.
        root_dir: simulation root directory
        tframe: current time frame.
    """
    print("Time frame: %d" % tframe)
    smime = math.sqrt(pic_info.mime)
    lx_di = pic_info.lx_di
    lz_di = pic_info.lz_di
    # xl, xr = 0, 0.3*lx_di
    xl, xr = 0, 70
    zb, zt = -15, 15
    fdata_2d, fdata_cumsum = calc_compressional_energization_terms(
            pic_info, root_dir, tframe, xl, xr, zb, zt)
    jpara_dote = fdata_2d["jpara_dote"]
    jperp_dote = fdata_2d["jperp_dote"]
    pdivv = fdata_2d["pdivv"]
    pshear = fdata_2d["pshear"]
    jpara_dote_cum = fdata_cumsum["jpara_dote_cum"]
    jperp_dote_cum = fdata_cumsum["jperp_dote_cum"]
    pdivv_cum = fdata_cumsum["pdivv_cum"]
    pshear_cum = fdata_cumsum["pshear_cum"]
    wpe_wce = pic_info.dtwce / pic_info.dtwpe
    va = wpe_wce / math.sqrt(pic_info.mime)  # Alfven speed
    enorm = va**2

    jpara_dote /= enorm
    jperp_dote /= enorm
    pdivv /= enorm
    pshear /= enorm

    fname = root_dir + "data/Ay.gda"
    kwargs = {"current_time": tframe, "xl": xl, "xr": xr,
              "zb": zb, "zt": zt}
    x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)

    xmin, xmax = np.min(x), np.max(x)
    zmin, zmax = np.min(z), np.max(z)

    w0, h0 = 0.72, 0.185
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
        ax.text(0.98, 0.90, text, color=text_color, fontsize=16,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='right', verticalalignment='top',
                transform=ax.transAxes)
        ax.yaxis.set_ticks(np.arange(-10, 11, 10))
        ax.plot([30, 30], [zb, zt], linestyle='--', color='k')
        return p1

    # smooth the data
    sigma = 3
    jpara_dote = gaussian_filter(jpara_dote, sigma)
    jperp_dote = gaussian_filter(jperp_dote, sigma)
    pdivv = gaussian_filter(pdivv, sigma)
    pshear = gaussian_filter(pshear, sigma)

    dmax = 0.05
    dmin = -dmax

    colors = colors_Set1_9
    fig = plt.figure(figsize=[7, 5])
    xs, ys = xs0, ys0
    ax1 = fig.add_axes([xs, ys, w0, h0])
    text1 = r'$\boldsymbol{j}_{e\perp}\cdot\boldsymbol{E}_\perp$'
    print("Min and max of jperp_dote: %f %f" % (np.min(jperp_dote),
                                                np.max(jperp_dote)))
    p1 = plot_one_field(jperp_dote, ax1, text1, 'k', label_bottom='off',
                        label_left='on', ylabel=True, vmin=dmin, vmax=dmax,
                        colormap=plt.cm.seismic, xs=xs, ys=ys, ay_color='k')
    xs1 = xs + w0 * 1.02
    w1 = w0 * 0.04
    ys1 = ys - 3 * (h0 + vgap)
    h1 = 4 * h0 + 3 * vgap
    cax = fig.add_axes([xs1, ys1, w1, h1])
    cbar1 = fig.colorbar(p1, cax=cax)
    cbar1.ax.tick_params(labelsize=16)
    cbar1.set_ticks(np.arange(dmin, dmax + 0.05, 0.05))

    ys -= h0 + vgap
    ax2 = fig.add_axes([xs, ys, w0, h0])
    text2 = r'$-p_e\nabla\cdot\boldsymbol{v}_E$'
    print("Min and max of pdivv: %f %f" % (np.min(pdivv), np.max(pdivv)))
    p2 = plot_one_field(pdivv, ax2, text2, colors[0], label_bottom='off',
                        label_left='on', ylabel=True, vmin=dmin, vmax=dmax,
                        colormap=plt.cm.seismic, xs=xs, ys=ys, ay_color='k')

    ys -= h0 + vgap
    ax3 = fig.add_axes([xs, ys, w0, h0])
    text3 = r'$-(p_{e\parallel}-p_{e\perp})b_ib_j\sigma_{ij}$'
    print("Min and max of pshear: %f %f" % (np.min(pshear), np.max(pshear)))
    p3 = plot_one_field(pshear, ax3, text3, colors[1], label_bottom='off',
                        label_left='on', ylabel=True, vmin=dmin, vmax=dmax,
                        colormap=plt.cm.seismic, xs=xs, ys=ys, ay_color='k')

    ys -= h0 + vgap
    ax4 = fig.add_axes([xs, ys, w0, h0])
    # ax4.set_prop_cycle('color', colors)
    # ax4.plot(x, jpara_dote_cum, linewidth=2)
    ax4.plot(x, jperp_dote_cum, linewidth=2, color='k')
    ax4.plot(x, pdivv_cum, linewidth=2, color=colors[0])
    ax4.plot(x, pshear_cum, linewidth=2, color=colors[1])
    ax4.set_xlim(ax1.get_xlim())
    # ax4.plot(ax4.get_xlim(), [0, 0], linestyle='--', color='k')
    ax4.tick_params(labelsize=16)
    ax4.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    text4 = 'Cumulative sum along $x$'
    ax4.text(0.02, 0.85, text4, color='k', fontsize=16,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='left', verticalalignment='center',
             transform=ax4.transAxes)
    ax4.yaxis.set_ticks(np.arange(0, 3, 1))
    # ax4.yaxis.set_ticks(np.arange(0, 1.1, 0.5))

    fdir = '../img/img_apjl/comp_terms2/' + run_name + '/'
    mkdir_p(fdir)
    fname = fdir + 'comp_terms_' + str(tframe) + '.pdf'
    fig.savefig(fname)

    ix, xcut = find_nearest(x, 30)
    jperp_dote_cut = jperp_dote[:, ix]
    pdivv_cut = pdivv[:, ix]
    pshear_cut = pshear[:, ix]

    fig = plt.figure(figsize=[7, 5])
    xs, ys = 0.12, 0.15
    w0, h0 = 0.8, 0.8
    ax1 = fig.add_axes([xs, ys, w0, h0])
    # ax1.set_prop_cycle('color', colors)
    ax1.plot(z, jperp_dote_cut, linewidth=2, label=text1, color='k')
    ax1.plot(z, pdivv_cut, linewidth=2, label=text2, color=colors[0])
    ax1.plot(z, pshear_cut, linewidth=2, label=text3, color=colors[1])
    ax1.set_xlim([-5, 5])
    ax1.tick_params(labelsize=16)
    ax1.set_xlabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax1.legend(loc=2, prop={'size': 16}, ncol=1,
               shadow=False, fancybox=False, frameon=False)
    fname = fdir + 'comp_terms_cut_' + str(tframe) + '.pdf'
    fig.savefig(fname)

    # plt.close()

    plt.show()


def plot_energy_spectrum_runs(species):
    """Plot final energy spectrum for different runs

    """
    fig = plt.figure(figsize=[7, 5])
    xs, ys = 0.15, 0.15
    w1, h1 = 0.8, 0.8
    ax = fig.add_axes([xs, ys, w1, h1])
    colors = colors_Set1_9[:5] + colors_Set1_9[6:] # remove yellow color
    ax.set_color_cycle(colors)

    def plot_spectrum_single_run(run_name, label1):
        picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        if (species == 'e'):
            vth = pic_info.vthe
        else:
            vth = pic_info.vthi
        gama = 1.0 / math.sqrt(1.0 - 3 * vth**2)
        eth = gama - 1.0

        fdir = '../data/spectra/' + run_name + '/'
        n0 = pic_info.nx * pic_info.ny * pic_info.nz * pic_info.nppc
        # ct = 1
        # fname = fdir + 'spectrum-' + species + '.1'
        # file_exist = os.path.isfile(fname)
        # while file_exist:
        #     ct += 1
        #     fname = fdir + 'spectrum-' + species + '.' + str(ct)
        #     file_exist = os.path.isfile(fname)
        ct = 12

        fname = fdir + 'spectrum-' + species + '.' + str(ct)
        elin, flin, elog, flog_e = spect_fit.get_energy_distribution(fname, n0)
        elog_norm_e = spect_fit.get_normalized_energy(species, elog, pic_info)
        nacc, eacc = spect_fit.accumulated_particle_info(elog, flog_e)
        flog_e /= nacc[-1]

        f_intial = fitting_funcs.func_maxwellian(elog, n0, 1.5 / eth)
        nacc, eacc = spect_fit.accumulated_particle_info(elog, f_intial)
        f_intial /= nacc[-1]

        ax.loglog(elog_norm_e, flog_e, linewidth=3, label=label1)
        return (elog_norm_e, f_intial)

    elog_norm_e, f_intial = plot_spectrum_single_run(
            "mime25_beta002_guide00_frequent_dump",
            r"$\beta_e=0.02, B_g=0$")
    elog_norm_e, f_intial = plot_spectrum_single_run(
            "mime25_beta002_guide02_frequent_dump",
            r"$\beta_e=0.02, B_g=0.2B_0$")
    elog_norm_e, f_intial = plot_spectrum_single_run(
            "mime25_beta002_guide05_frequent_dump",
            r"$\beta_e=0.02, B_g=0.5B_0$")
    elog_norm_e, f_intial = plot_spectrum_single_run(
            "mime25_beta002_guide10_frequent_dump",
            r"$\beta_e=0.02, B_g=B_0$")
    elog_norm_e, f_intial = plot_spectrum_single_run(
            "mime25_beta008_guide00_frequent_dump",
            r"$\beta_e=0.08, B_g=0$")
    elog_norm_e, f_intial = plot_spectrum_single_run(
            "mime25_beta032_guide00_frequent_dump",
            r"$\beta_e=0.32, B_g=0$")

    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.loglog(elog_norm_e, f_intial, linewidth=1, color='k',
              linestyle='--', label='initial')

    ax.set_xlim([5E-2, 5E2])
    ax.set_ylim([1E-8, 2E2])
    ax.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$',
                  fontdict=font, fontsize=20)
    ax.set_ylabel(r'$f(\varepsilon)$', fontdict=font, fontsize=20)
    ax.tick_params(labelsize=16)

    text1 = r'$\varepsilon > 20\varepsilon_\text{th}$'
    ax.text(0.95, 0.8, text1, color='k', fontsize=20,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='right', verticalalignment='center',
            transform=ax.transAxes)

    colors_plot = [colors[0], colors[1], 'k']
    leg = ax.legend(loc=3, prop={'size': 16}, ncol=1,
            shadow=False, fancybox=False, frameon=False)
    # for color, text in zip(colors_plot, leg.get_texts()):
    #     text.set_color(color)
    ax.axvspan(20, 5E2, alpha=0.2, color='grey')
    # ax.grid(True)

    img_path = '../img/img_apjl/'
    mkdir_p(img_path)
    fname = img_path + 'spectra_runs.pdf'
    fig.savefig(fname)

    plt.show()


def energetic_particle_number_energy(species):
    """High energy particles fractions

    """
    def energetic_particle_fractions(run_name):
        picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        if (species == 'e'):
            vth = pic_info.vthe
        else:
            vth = pic_info.vthi
        gama = 1.0 / math.sqrt(1.0 - 3 * vth**2)
        eth = gama - 1.0

        fdir = '../data/spectra/' + run_name + '/'
        n0 = pic_info.nx * pic_info.ny * pic_info.nz * pic_info.nppc
        ct = 12

        fname = fdir + 'spectrum-' + species + '.' + str(ct)
        elin, flin, elog, flog_e = spect_fit.get_energy_distribution(fname, n0)
        elog_norm_e = spect_fit.get_normalized_energy(species, elog, pic_info)
        nacc, eacc = spect_fit.accumulated_particle_info(elog, flog_e)
        eh_index = find_closest(elog_norm_e, 100)
        nfrac = (nacc[-1] - nacc[eh_index]) / nacc[-1]
        efrac = (eacc[-1] - eacc[eh_index]) / eacc[-1]

        return (nfrac, efrac)

    runs = ['mime25_beta002_guide00_frequent_dump',
            'mime25_beta002_guide02_frequent_dump',
            'mime25_beta002_guide05_frequent_dump',
            'mime25_beta002_guide10_frequent_dump',
            'mime25_beta008_guide00_frequent_dump',
            'mime25_beta032_guide00_frequent_dump']

    nrun = len(runs)
    nfracs = np.zeros(nrun)
    efracs = np.zeros(nrun)
    for irun, run in enumerate(runs):
        nfracs[irun], efracs[irun] = energetic_particle_fractions(run)

    print(nfracs)
    print(efracs)

    colors = colors_Set1_9[:5] + colors_Set1_9[6:] # remove yellow color

    x = [0, 0.2, 0.5, 1.0, 1.5, 2.0]
    fig = plt.figure(figsize=[7, 5])
    xs, ys = 0.12, 0.58
    w1, h1 = 0.83, 0.38
    ax1 = fig.add_axes([xs, ys, w1, h1])
    ax1.set_prop_cycle('color', colors)
    ax1.plot(x[0:4], nfracs[0:4], color='k', label=r'$\beta_e=0.02$')
    ax1.scatter(x[0:4], nfracs[0:4], color=colors[:nrun], s=80)
    ax1.scatter([0], nfracs[4:5], color=colors[:nrun], marker='o',
                c=colors[4], s=80)
    ax1.scatter([0], nfracs[5:], color=colors[:nrun], marker='o',
                c=colors[5], s=80)
    ax1.yaxis.set_ticks(np.linspace(0, 0.06, 4))
    ax1.set_xlim([-0.1, 1.1])
    ax1.tick_params(axis='x', which='major', direction='in')
    ax1.tick_params(axis='y', which='major', direction='in')
    ax1.tick_params(labelsize=16)
    ax1.tick_params(axis='x', labelbottom='off')
    text1 = r'Number fraction for $\varepsilon > 20\varepsilon_\text{th}$'
    ax1.text(0.98, 0.82, text1, color='k', fontsize=20,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='right', verticalalignment='center',
            transform=ax1.transAxes)
    ax1.legend(loc=4, prop={'size': 16}, ncol=1,
               shadow=False, fancybox=False, frameon=False)
    ys -= h1 + 0.05
    ax2 = fig.add_axes([xs, ys, w1, h1])
    ax2.set_prop_cycle('color', colors)
    ax2.plot(x[0:4], efracs[0:4], color='k', label=r'$\beta_e=0.02$')
    ax2.scatter(x[0:4], efracs[0:4], color=colors[:nrun], s=80)
    ax2.scatter([0], efracs[4:5], color=colors[:nrun], marker='o',
                c=colors[4], s=80)
    ax2.scatter([0], efracs[5:], color=colors[:nrun], marker='o',
                c=colors[5], s=80)
    ax2.yaxis.set_ticks(np.linspace(0, 0.3, 4))
    ax2.set_xlim([-0.1, 1.1])
    ax2.tick_params(axis='x', which='major', direction='in')
    ax2.tick_params(axis='y', which='major', direction='in')
    ax2.tick_params(labelsize=16)
    ax2.set_xlabel(r'$B_g/B_0$', fontdict=font, fontsize=20)
    text2 = r'Energy fraction for $\varepsilon > 20\varepsilon_\text{th}$'
    ax2.text(0.98, 0.82, text2, color='k', fontsize=20,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='right', verticalalignment='center',
            transform=ax2.transAxes)
    ax2.legend(loc=4, prop={'size': 16}, ncol=1,
               shadow=False, fancybox=False, frameon=False)

    img_path = '../img/img_apjl/'
    mkdir_p(img_path)
    fname = img_path + 'fraction_runs.pdf'
    fig.savefig(fname)

    plt.show()


def plot_current_density(pic_info, root_dir, run_name):
    """
    Plot current density

    Args:
        pic_info: namedtuple for the PIC simulation information.
        root_dir: simulation root directory
    """
    smime = math.sqrt(pic_info.mime)
    dx = pic_info.dx_di * smime
    dz = pic_info.dz_di * smime
    lx_de = pic_info.lx_di * smime
    lz_de = pic_info.lz_di * smime
    zb, zt = -12, 12
    current_time = 12
    kwargs = {"current_time": current_time, "xl": 0, "xr": lx_de,
              "zb": zb, "zt": zt}
    fname = root_dir + "data/jy.gda"
    x, z, jy1 = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/Ay.gda"
    x, z, Ay1 = read_2d_fields(pic_info, fname, **kwargs)

    current_time = 30
    kwargs = {"current_time": current_time, "xl": 0, "xr": lx_de,
              "zb": zb, "zt": zt}
    fname = root_dir + "data/jy.gda"
    x, z, jy2 = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/Ay.gda"
    x, z, Ay2 = read_2d_fields(pic_info, fname, **kwargs)
    
    wpe_wce = pic_info.dtwce / pic_info.dtwpe
    va = wpe_wce / math.sqrt(pic_info.mime)  # Alfven speed of inflow region

    xmin, xmax = np.min(x), np.max(x)
    zmin, zmax = np.min(z), np.max(z)

    # w0, h0 = 0.41, 0.11
    w0, h0 = 0.85, 0.37
    xs0, ys0 = 0.07, 0.96 - h0
    vgap, hgap = 0.05, 0.04

    def plot_one_field(fdata, ax, text, text_color, label_bottom='on',
                       label_left='on', ylabel=False, vmin=0, vmax=10,
                       colormap=plt.cm.seismic, xs=xs0, ys=ys0, ay_color='k',
                       color_bar=False):
        plt.tick_params(labelsize=20)
        p1 = ax.imshow(fdata, vmin=vmin, vmax=vmax, cmap=colormap,
                       extent=[xmin, xmax, zmin, zmax], aspect='auto',
                       origin='lower', interpolation='bicubic')
        ax.tick_params(axis='x', labelbottom=label_bottom)
        ax.tick_params(axis='y', labelleft=label_left)
        if ylabel:
            ax.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=32)
        ax.text(0.02, 0.75, text, color=text_color, fontsize=32,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)
        if color_bar:
            xs1 = xs + w0 * 1.01
            w1 = w0 * 0.02
            cax = fig.add_axes([xs1, ys, w1, h0])
            cbar = fig.colorbar(p1, cax=cax)
            cbar.ax.tick_params(labelsize=20)
            return (p1, cbar)
        else:
            return p1

    fig = plt.figure(figsize=[20, 5])
    xs, ys = xs0, ys0
    ax1 = fig.add_axes([xs, ys, w0, h0])
    text1 = r'$j_y: t\Omega_{ci}=60$'
    print("min and max of jy1: %f %f" % (np.min(jy1), np.max(jy1)))
    nmin, nmax = -0.2, 0.6
    p1 = plot_one_field(jy1, ax1, text1, 'w', label_bottom='off',
                        label_left='on', ylabel=True, vmin=nmin,
                        vmax=nmax, colormap=plt.cm.inferno, xs=xs, ys=ys,
                        ay_color='w', color_bar=False)
    ax1.contour(x, z, Ay1, colors='w', linewidths=0.5)
    xs1 = xs + w0 * 1.01
    w1 = w0 * 0.02
    ys1 = ys - (h0 + vgap)
    h1 = 2 * h0 + vgap
    cax1 = fig.add_axes([xs1, ys1, w1, h1])
    cbar1 = fig.colorbar(p1, cax=cax1)
    cbar1.ax.tick_params(labelsize=20)
    cbar1.set_ticks(np.arange(nmin, nmax + 0.2, 0.2))

    ys -= h0 + vgap
    ax2 = fig.add_axes([xs, ys, w0, h0])
    vmin, vmax = -0.2, 0.6
    text2 = r'$j_y: t\Omega_{ci}=150$'
    print("min and max of jy2: %f %f" % (np.min(jy2), np.max(jy2)))
    p2 = plot_one_field(jy2, ax2, text2, 'w', label_bottom='on',
                        label_left='on', ylabel=True, vmin=vmin, vmax=vmax,
                        colormap=plt.cm.inferno, xs=xs, ys=ys,
                        ay_color='w', color_bar=False)
    ax2.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=32)
    ax2.contour(x, z, Ay2, colors='w', linewidths=0.5)

    fdir = '../img/img_agu17/'
    mkdir_p(fdir)
    fname = fdir + 'jys.pdf'
    fig.savefig(fname)
    plt.show()


def plot_momentum_distribution(run_name, species):
    """
    """
    nbins = 300
    drange = [[1, 1.1], [0, 1]]
    pbins = np.logspace(-2, 1, nbins + 1)

    # normalized with thermal energy
    if (species == 'e'):
        vth = pic_info.vthe
    else:
        vth = pic_info.vthi
    gama = 1.0 / math.sqrt(1.0 - 3 * vth**2)
    eth = gama - 1.0
    pth = math.sqrt(gama**2 - 1)
    pbins /= pth

    xs, ys = 0.15, 0.15
    w1, h1 = 0.8, 0.8
    fig = plt.figure(figsize=[7, 5])
    ax1 = fig.add_axes([xs, ys, w1, h1])
    colors = colors_Set1_9[:5] + colors_Set1_9[6:] # remove yellow color
    ax1.set_prop_cycle('color', colors)

    fdir = '../data/momentum_distribution/' + run_name + '/'
    # for tframe in range(1, pic_info.ntp-1):
    for tframe in range(1, 10, 2):
        tindex = tframe * pic_info.particle_interval
        fname = fdir + 'pdists_' + species + '.' + str(tindex) + '.all'
        fdata = np.fromfile(fname)
        sz, = fdata.shape
        nvar = sz / nbins
        fdata = fdata.reshape((nvar, nbins))
        ppara_dist = fdata[0, :]
        pperp_dist = fdata[1, :]
        p1, = ax1.semilogy(pbins[:-1], ppara_dist, linewidth=2, linestyle='-',
                           label=r'$p_\parallel$')
        ax1.semilogy(pbins[:-1], pperp_dist, linewidth=2, linestyle='--',
                     color=p1.get_color(), label=r'$p_\perp$')

    ax1.set_xlim([0, 40])
    ax1.set_xlabel(r'$p/p_\text{th}$', fontdict=font, fontsize=20)
    ax1.set_ylabel('momentum distribution', fontdict=font, fontsize=20)
    ax1.tick_params(labelsize=16)
    leg = ax1.legend(loc=1, prop={'size': 16}, ncol=1,
                     shadow=False, fancybox=False, frameon=False)
    fdir = '../img/img_apjl/pdist/' + run_name + '/'
    mkdir_p(fdir)
    # fname = fdir + 'pdist_' + species + '_' + str(tindex) + '.eps'
    # fig.savefig(fname)
    # plt.close()
    plt.show()


def div0(a, b):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0]

    From: http://stackoverflow.com/a/35696047/2561161

    """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[~np.isfinite(c)] = 0  # -inf inf NaN
    return c


def plot_anisotropy_distribution_1d(run_name, species):
    """
    """
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pint = pic_info.particle_interval
    ntp = pic_info.ntp
    if species == 'e':
        pmass = 1.0
    else:
        pmass = pic_info.mime

    nebins = 60
    nabins = 20
    ebins = np.logspace(-4, 2, nebins) / math.sqrt(pmass)
    abins = np.logspace(-1, 1, nabins)

    # normalized with thermal energy
    if (species == 'e'):
        vth = pic_info.vthe
    else:
        vth = pic_info.vthi
    gama = 1.0 / math.sqrt(1.0 - 3 * vth**2)
    eth = gama - 1.0
    ebins /= eth
    xmin, xmax = ebins[0], ebins[-1]
    ymin, ymax = abins[0], abins[-1]

    xs, ys = 0.15, 0.15
    w1, h1 = 0.8, 0.8
    fig = plt.figure(figsize=[7, 5])

    ts, te = 2, 11
    nt = te - ts
    # Using contourf to provide my colorbar info, then clearing the figure
    Z = [[0,0],[0,0]]
    levels = range(ts,te+1,1)
    CS3 = plt.contourf(Z, levels, cmap=plt.cm.Reds_r)
    plt.clf()

    ax1 = fig.add_axes([xs, ys, w1, h1])
    colors = colors_Set1_9[:5] + colors_Set1_9[6:] # remove yellow color
    ax1.set_prop_cycle('color', colors)

    # fdir = '../data/momentum_distribution/' + run_name + '/'
    fdir = '../data/anisotropy_distribution/' + run_name + '/'

    tframe = 3
    tindex = pint * tframe
    fname = fdir + 'edists_' + species + '.' + str(tindex) + '.all'
    fdata1 = np.fromfile(fname)
    fdata1 = fdata1.reshape((2, nebins))
    ax1.semilogx(ebins, div0(fdata1[0, :], fdata1[1, :]),
                 linewidth=2, color='k', label=r'$t\Omega_{ci}=150$')
    tframe = 5
    tindex = pint * tframe
    fname = fdir + 'edists_' + species + '.' + str(tindex) + '.all'
    fdata1 = np.fromfile(fname)
    fdata1 = fdata1.reshape((2, nebins))
    ax1.semilogx(ebins, div0(fdata1[0, :], fdata1[1, :]),
                 linewidth=2, color='k', linestyle='--',
                 label=r'$t\Omega_{ci}=250$')

    ax1.legend(loc=2, prop={'size': 20}, ncol=1,
               shadow=False, fancybox=False, frameon=False)

    xmin, xmax = 0.1, 500
    ax1.plot([xmin, xmax], [2, 2], linewidth=0.5, linestyle='--', color='k')
    ax1.plot([xmin, xmax], [1.5, 1.5], linewidth=0.5, linestyle='--', color='k')
    ax1.plot([xmin, xmax], [1, 1], linewidth=0.5, linestyle='--', color='k')
    ax1.tick_params(axis='x', which='minor', direction='in')
    ax1.tick_params(axis='x', which='major', direction='in')
    ax1.tick_params(axis='y', which='minor', direction='in')
    ax1.tick_params(axis='y', which='major', direction='in')
    ax1.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$', fontdict=font, fontsize=20)
    # ylabel1 = r'$p_\parallel v_\parallel/(0.5p_\perp v_\perp)$'
    ylabel1 = 'Anisotropy'
    ax1.set_ylabel(ylabel1, fontdict=font, fontsize=20)
    ax1.tick_params(labelsize=16)
    ax1.set_xlim([xmin, xmax])
    # ax1.set_ylim([0.5, 2])

    fdir = '../img/img_apjl/'
    mkdir_p(fdir)
    fname = fdir + 'anisotropy_' + species + '_' + run_name + '.eps'
    fig.savefig(fname)
    # plt.close()
    plt.show()


def get_cmd_args():
    """Get command line arguments """
    default_run_name = 'mime25_beta002_guide05_frequent_dump'
    default_run_dir = '/net/scratch3/xiaocanli/reconnection/frequent_dump/' + \
            'mime25_beta002_guide05_frequent_dump/'
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
    # plot_ne_velocity(pic_info, run_dir, run_name, 30)
    # plot_current_density(pic_info, run_dir, run_name)
    plot_compression_time_electron(pic_info)
    # plot_compression_shear_2d(pic_info, run_dir, run_name, 30)
    # plot_compression_shear_g1(pic_info, run_dir, run_name, 30)
    # compression_ratio_high_energy('e')
    # compression_ratio_high_energy_test('e')
    # compression_ratio_high_energy_test2('e')
    # plot_energy_spectrum_runs('e')
    # energetic_particle_number_energy('e')
    # plot_momentum_distribution(run_name, 'e')
    # plot_anisotropy_distribution_1d(run_name, species)
    cts = range(pic_info.ntp)
    for ct in cts:
        tframe_fields = (ct + 1) * tratio
        tindex = (ct + 1) * pic_info.particle_interval
        # plot_ne_velocity(pic_info, run_dir, run_name, tframe_fields)
        # plot_compression_shear_2d(pic_info, run_dir, run_name, tframe_fields)
