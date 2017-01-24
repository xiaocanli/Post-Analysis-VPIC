"""
Analysis procedures for energy conversion.
"""
import collections
import math
import os.path
import re
import struct

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import simplejson as json
from matplotlib import rc
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d

import colormap.colormaps as cmaps
import palettable
import pic_information
from contour_plots import plot_2d_contour, read_2d_fields
from energy_conversion import calc_jdotes_fraction_multi
from fields_plot import *
from pic_information import list_pic_info_dir
from runs_name_path import ApJ_long_paper_runs, guide_field_runs
from serialize_json import data_to_json, json_to_data

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

colors = palettable.colorbrewer.qualitative.Set1_9.mpl_colors

font = {'family' : 'serif',
        #'color'  : 'darkred',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 24,
        }

def plot_energy_evolution(pic_info):
    """Plot energy time evolution.

    Plot time evolution of magnetic, electric, electron and ion kinetic
    energies.

    Args:
        pic_info: the PIC simulation information.
    """
    tenergy = pic_info.tenergy
    ene_electric = pic_info.ene_electric
    ene_magnetic = pic_info.ene_magnetic
    kene_e = pic_info.kene_e
    kene_i = pic_info.kene_i
    ene_bx = pic_info.ene_bx
    ene_by = pic_info.ene_by
    ene_bz = pic_info.ene_bz

    enorm = ene_magnetic[0]

    fig = plt.figure(figsize=[7, 5])
    xs, ys = 0.13, 0.13
    w1, h1 = 0.8, 0.8
    ax = fig.add_axes([xs, ys, w1, h1])
    ax.set_color_cycle(colors)
    p1, = ax.plot(tenergy, ene_magnetic/enorm, linewidth=2,
            label=r'$\varepsilon_{b}$')
    p2, = ax.plot(tenergy, kene_i/enorm, linewidth=2, label=r'$K_i$')
    p3, = ax.plot(tenergy, kene_e/enorm, linewidth=2, label=r'$K_e$')
    p4, = ax.plot(tenergy, 100*ene_electric/enorm, linewidth=2,
            label=r'$100\varepsilon_{e}$')
    # ax.set_xlim([0, np.max(tenergy)])
    ax.set_xlim([0, np.max(tenergy)])
    ax.set_ylim([0, 1.05])

    ax.set_xlabel(r'$t\Omega_{ci}$', fontdict=font, fontsize=24)
    ax.set_ylabel(r'Energy/$\varepsilon_{b0}$', fontdict=font, fontsize=24)
    leg = ax.legend(loc=1, prop={'size':20}, ncol=2,
            shadow=False, fancybox=False, frameon=False)
    for color,text in zip(colors, leg.get_texts()):
            text.set_color(color)

    # ax.text(0.5, 0.8, r'$\varepsilon_{b}$',
    #         color='blue', fontsize=24,
    #         bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
    #         horizontalalignment='center', verticalalignment='center',
    #         transform = ax.transAxes)
    # ax.text(0.7, 0.8, r'$\varepsilon_e$', color='m', fontsize=24,
    #         bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
    #         horizontalalignment='center', verticalalignment='center',
    #         transform = ax.transAxes)
    # ax.text(0.5, 0.5, r'$K_e$', color='red', fontsize=24,
    #         bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
    #         horizontalalignment='center', verticalalignment='center',
    #         transform = ax.transAxes)
    # ax.text(0.7, 0.5, r'$K_i$', color='green', fontsize=24,
    #         bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
    #         horizontalalignment='center', verticalalignment='center',
    #         transform = ax.transAxes)

    plt.tick_params(labelsize=20)
    #plt.savefig('pic_ene.eps')

    print('The dissipated magnetic energy: %5.3f' % (1.0 - ene_magnetic[-1]/enorm))
    print('Energy gain to the initial magnetic energy: %5.3f, %5.3f' %
            ((kene_e[-1]-kene_e[0])/enorm, (kene_i[-1]-kene_i[0])/enorm))
    print('Initial kene_e and kene_i to the initial magnetic energy: %5.3f, %5.3f' %
            (kene_e[0]/enorm, kene_i[0]/enorm))
    print('Final kene_e and kene_i to the initial magnetic energy: %5.3f, %5.3f' %
            (kene_e[-1]/enorm, kene_i[-1]/enorm))
    init_ene = pic_info.ene_electric[0] + pic_info.ene_magnetic[0] + \
               kene_e[0] + kene_i[0]
    final_ene = pic_info.ene_electric[-1] + pic_info.ene_magnetic[-1] + \
               kene_e[-1] + kene_i[-1]
    print('Energy conservation: %5.3f' % (final_ene / init_ene))
    # plt.show()


def plot_energy_evolution_multi():
    """Plot energy evolution for multiple runs.
    """
    dir = '../data/pic_info/'
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    odir = '../img/ene_evolution/'
    if not os.path.isdir(odir):
        os.makedirs(odir)
    fnames = list_pic_info_dir(dir)
    for fname in fnames:
        rname = fname.replace(".json", ".eps")
        oname = rname.replace("pic_info", "enes")
        oname = odir + oname
        fname = dir + fname
        pic_info = read_data_from_json(fname)
        plot_energy_evolution(pic_info)
        plt.savefig(oname)
        plt.close()


def read_data_from_json(fname):
    """Read jdote data from a json file

    Args:
        fname: file name of the json file of the jdote data.
    """
    with open(fname, 'r') as json_file:
        data = json_to_data(json.load(json_file))
    print("Reading %s" % fname)
    return data


def plot_dke(pic_info, species, ax):
    """Plot the electron energy change
    """
    if species == 'e':
        kene= pic_info.kene_e
    else:
        kene= pic_info.kene_e
    tenergy = pic_info.tenergy
    p1, = ax.plot(tenergy, kene/kene[0], linewidth=2)


def plot_dke_multi():
    """Plot multiple electron energy change
    """
    dir = '../data/pic_info/'
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    odir = '../img/thesis/'
    if not os.path.isdir(odir):
        os.makedirs(odir)
    fnames = list_pic_info_dir(dir)
    fig = plt.figure(figsize=[7, 5])
    xs, ys = 0.13, 0.13
    w1, h1 = 0.8, 0.8
    ax = fig.add_axes([xs, ys, w1, h1])
    ax.set_color_cycle(colors)
    pnames = [fnames[7], fnames[0], fnames[8], fnames[2]]
    for fname in pnames:
        rname = fname.replace(".json", ".eps")
        oname = rname.replace("pic_info", "dke")
        oname = odir + oname
        fname = dir + fname
        pic_info = read_data_from_json(fname)
        plot_dke(pic_info, 'e', ax)
        # plt.savefig(oname)
        # plt.close()
    ax.plot([0, 1200], [1, 1], linestyle='--', color='k')
    ax.set_xlabel(r'$t\Omega_{ci}$', fontdict=font, fontsize=24)
    ax.set_ylabel(r'Energy/$\varepsilon_{b0}$', fontdict=font, fontsize=24)
    plt.tick_params(labelsize=16)
    plt.show()


def plot_jy_multi():
    """Plot the jy for multiple runs
    """
    base_dirs, run_names = ApJ_long_paper_runs()
    bdir = base_dirs[0]
    run_name = run_names[0]
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    kwargs = {"current_time":11, "xl":0, "xr":200, "zb":-10, "zt":10}
    fname = bdir + 'data/jy.gda'
    x, z, jy1 = read_2d_fields(pic_info, fname, **kwargs)
    fname = bdir + 'data/Ay.gda'
    x, z, Ay1 = read_2d_fields(pic_info, fname, **kwargs)
    bdir = base_dirs[1]
    run_name = run_names[1]
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    fname = bdir + 'data/jy.gda'
    x, z, jy2 = read_2d_fields(pic_info, fname, **kwargs)
    fname = bdir + 'data/Ay.gda'
    x, z, Ay2 = read_2d_fields(pic_info, fname, **kwargs)
    bdir = base_dirs[3]
    run_name = run_names[3]
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    fname = bdir + 'data/jy.gda'
    x, z, jy3 = read_2d_fields(pic_info, fname, **kwargs)
    fname = bdir + 'data/Ay.gda'
    x, z, Ay3 = read_2d_fields(pic_info, fname, **kwargs)
    nx, = x.shape
    nz, = z.shape

    width = 0.74
    height = 0.25
    xs = 0.13
    ys = 0.97 - height
    gap = 0.05
    fig = plt.figure(figsize=[7,5])
    ax1 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep":1, "zstep":1, "vmin":-0.5, "vmax":0.5}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, jy1, ax1, fig, **kwargs_plot)
    p1.set_cmap(plt.cm.get_cmap('seismic'))
    ax1.contour(x[0:nx:xstep], z[0:nz:zstep], Ay1[0:nz:zstep, 0:nx:xstep], 
            colors='black', linewidths=0.5)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax1.tick_params(labelsize=16)
    cbar1.ax.set_ylabel(r'$j_y$', fontdict=font, fontsize=20)
    cbar1.set_ticks(np.arange(-0.4, 0.5, 0.2))
    cbar1.ax.tick_params(labelsize=16)
    ax1.tick_params(axis='x', labelbottom='off')

    ys -= height + gap
    ax2 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep":1, "zstep":1, "vmin":-0.5, "vmax":0.5}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p2, cbar2 = plot_2d_contour(x, z, jy2, ax2, fig, **kwargs_plot)
    p2.set_cmap(plt.cm.get_cmap('seismic'))
    ax2.contour(x[0:nx:xstep], z[0:nz:zstep], Ay2[0:nz:zstep, 0:nx:xstep], 
            colors='black', linewidths=0.5)
    ax2.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax2.tick_params(labelsize=16)
    cbar2.ax.set_ylabel(r'$j_y$', fontdict=font, fontsize=20)
    cbar2.set_ticks(np.arange(-0.4, 0.5, 0.2))
    cbar2.ax.tick_params(labelsize=16)
    ax2.tick_params(axis='x', labelbottom='off')

    ys -= height + gap
    ax3 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep":1, "zstep":1, "vmin":-1.0, "vmax":1.0}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p3, cbar3 = plot_2d_contour(x, z, jy3, ax3, fig, **kwargs_plot)
    p3.set_cmap(plt.cm.get_cmap('seismic'))
    ax3.contour(x[0:nx:xstep], z[0:nz:zstep], Ay3[0:nz:zstep, 0:nx:xstep], 
            colors='black', linewidths=0.5)
    ax3.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    ax3.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax3.tick_params(labelsize=16)
    cbar3.ax.set_ylabel(r'$j_y$', fontdict=font, fontsize=20)
    cbar3.set_ticks(np.arange(-1.0, 1.1, 0.5))
    cbar3.ax.tick_params(labelsize=16)

    ax1.text(0.02, 0.78, r'R8 $\beta_e=0.2$', color='k', fontsize=20,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform = ax1.transAxes)
    ax2.text(0.02, 0.78, r'R7 $\beta_e=0.07$', color='k', fontsize=20,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform = ax2.transAxes)
    ax3.text(0.02, 0.78, r'R6 $\beta_e=0.007$', color='k', fontsize=20,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform = ax3.transAxes)

    fig.savefig('../img/jy_multi.jpg', dpi=300)

    plt.show()


def plot_jy_guide():
    """Plot jy contour for the runs with guide field

    Args:
        run_name: the name of this run.
        root_dir: the root directory of this run.
        pic_info: PIC simulation information in a namedtuple.
    """
    cts = [40, 20, 20, 20, 30]
    var = 'ez'
    vmin, vmax = -0.2, 0.2
    ticks = np.arange(-0.2, 0.25, 0.1)
    cmap = plt.cm.get_cmap('seismic')
    label = r'$E_z$'
    # # cmap = cmaps.inferno
    # var = 'jy'
    # vmin, vmax = -0.2, 0.5
    # ticks = np.arange(-0.2, 0.6, 0.2)
    # cmap = plt.cm.get_cmap('jet')
    # label = r'$j_y$'
    # var = 'by'
    # vmin, vmax = -0.2, 0.5
    # ticks = np.arange(-0.2, 0.6, 0.2)
    # cmap = plt.cm.get_cmap('jet')
    # label = r'$B_y$'
    base_dirs, run_names = guide_field_runs()
    bdir = base_dirs[0]
    run_name = run_names[0]
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    xl, xr = 50, 150
    kwargs = {"current_time":cts[0], "xl":xl, "xr":xr, "zb":-10, "zt":10}
    fname = bdir + 'data/' + var + '.gda'
    x, z, jy1 = read_2d_fields(pic_info, fname, **kwargs)
    fname = bdir + 'data/Ay.gda'
    x, z, Ay1 = read_2d_fields(pic_info, fname, **kwargs)
    kwargs = {"current_time":cts[1], "xl":xl, "xr":xr, "zb":-10, "zt":10}
    bdir = base_dirs[1]
    run_name = run_names[1]
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    fname = bdir + 'data/' + var + '.gda'
    x, z, jy2 = read_2d_fields(pic_info, fname, **kwargs)
    fname = bdir + 'data/Ay.gda'
    x, z, Ay2 = read_2d_fields(pic_info, fname, **kwargs)
    kwargs = {"current_time":cts[2], "xl":xl, "xr":xr, "zb":-10, "zt":10}
    bdir = base_dirs[2]
    run_name = run_names[2]
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    fname = bdir + 'data/' + var + '.gda'
    x, z, jy3 = read_2d_fields(pic_info, fname, **kwargs)
    fname = bdir + 'data/Ay.gda'
    x, z, Ay3 = read_2d_fields(pic_info, fname, **kwargs)
    kwargs = {"current_time":cts[3], "xl":xl, "xr":xr, "zb":-10, "zt":10}
    bdir = base_dirs[3]
    run_name = run_names[3]
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    fname = bdir + 'data/' + var + '.gda'
    x, z, jy4 = read_2d_fields(pic_info, fname, **kwargs)
    fname = bdir + 'data/Ay.gda'
    x, z, Ay4 = read_2d_fields(pic_info, fname, **kwargs)

    kwargs = {"current_time":cts[4], "xl":xl, "xr":xr, "zb":-10, "zt":10}
    bdir = base_dirs[4]
    run_name = run_names[4]
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    fname = bdir + 'data/' + var + '.gda'
    x, z, jy5 = read_2d_fields(pic_info, fname, **kwargs)
    fname = bdir + 'data/Ay.gda'
    x, z, Ay5 = read_2d_fields(pic_info, fname, **kwargs)
    nx, = x.shape
    nz, = z.shape

    width = 0.8
    height = 0.16
    xs = 0.1
    ys = 0.98 - height
    gap = 0.025
    fig = plt.figure(figsize=[10,8])
    ax1 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep":1, "zstep":1, "vmin":vmin, "vmax":vmax}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, jy1, ax1, fig, **kwargs_plot)
    p1.set_cmap(cmap)
    ax1.contour(x[0:nx:xstep], z[0:nz:zstep], Ay1[0:nz:zstep, 0:nx:xstep], 
            colors='black', linewidths=0.5)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax1.tick_params(labelsize=16)
    cbar1.ax.set_ylabel(label, fontdict=font, fontsize=20)
    cbar1.set_ticks(ticks)
    cbar1.ax.tick_params(labelsize=16)
    ax1.tick_params(axis='x', labelbottom='off')

    ys -= height + gap
    ax2 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep":1, "zstep":1, "vmin":vmin, "vmax":vmax}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p2, cbar2 = plot_2d_contour(x, z, jy2, ax2, fig, **kwargs_plot)
    p2.set_cmap(cmap)
    ax2.contour(x[0:nx:xstep], z[0:nz:zstep], Ay2[0:nz:zstep, 0:nx:xstep], 
            colors='black', linewidths=0.5)
    ax2.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax2.tick_params(labelsize=16)
    cbar2.ax.set_ylabel(label, fontdict=font, fontsize=20)
    cbar2.set_ticks(ticks)
    cbar2.ax.tick_params(labelsize=16)
    ax2.tick_params(axis='x', labelbottom='off')

    ys -= height + gap
    ax3 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep":1, "zstep":1, "vmin":vmin, "vmax":vmax}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p3, cbar3 = plot_2d_contour(x, z, jy3, ax3, fig, **kwargs_plot)
    p3.set_cmap(cmap)
    ax3.contour(x[0:nx:xstep], z[0:nz:zstep], Ay3[0:nz:zstep, 0:nx:xstep], 
            colors='black', linewidths=0.5)
    # ax3.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    ax3.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax3.tick_params(labelsize=16)
    cbar3.ax.set_ylabel(label, fontdict=font, fontsize=20)
    cbar3.set_ticks(ticks)
    cbar3.ax.tick_params(labelsize=16)
    ax3.tick_params(axis='x', labelbottom='off')

    ys -= height + gap
    ax4 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep":1, "zstep":1, "vmin":vmin, "vmax":vmax}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p4, cbar4 = plot_2d_contour(x, z, jy4, ax4, fig, **kwargs_plot)
    p4.set_cmap(cmap)
    ax4.contour(x[0:nx:xstep], z[0:nz:zstep], Ay4[0:nz:zstep, 0:nx:xstep], 
            colors='black', linewidths=0.5)
    # ax4.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    ax4.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax4.tick_params(labelsize=16)
    cbar4.ax.set_ylabel(label, fontdict=font, fontsize=20)
    cbar4.set_ticks(ticks)
    cbar4.ax.tick_params(labelsize=16)
    ax4.tick_params(axis='x', labelbottom='off')

    ys -= height + gap
    ax5 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep":1, "zstep":1, "vmin":vmin, "vmax":vmax}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p5, cbar5 = plot_2d_contour(x, z, jy5, ax5, fig, **kwargs_plot)
    p5.set_cmap(cmap)
    ax5.contour(x[0:nx:xstep], z[0:nz:zstep], Ay5[0:nz:zstep, 0:nx:xstep], 
            colors='black', linewidths=0.5)
    ax5.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    ax5.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax5.tick_params(labelsize=16)
    cbar5.ax.set_ylabel(label, fontdict=font, fontsize=20)
    cbar5.set_ticks(ticks)
    cbar5.ax.tick_params(labelsize=16)

    axs = [ax1, ax2, ax3, ax4, ax5]
    bgs = [0.0, 0.2, 0.5, 1.0, 4.0]
    for ax, bg in zip(axs, bgs):
        tname = r'$B_g = ' + str(bg) + '$'
        ax.text(0.02, 0.8, tname, color='k', fontsize=24,
                bbox=dict(facecolor='none', alpha=1.0,
                          edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='center',
                transform = ax.transAxes)

    fname = '../img/thesis/' + var + '.jpg'
    fig.savefig(fname, dpi=200)

    plt.show()


def get_jdrifts(jdotes):
    jdotes = jdotes.reshape((5, 16))
    jc = jdotes[:, 0]
    jm = jdotes[:, 2]
    jg = jdotes[:, 3]
    jp = jdotes[:, 5]
    jpara = jdotes[:, 11]
    jperp = jdotes[:, 12]
    ja = jdotes[:, 14]
    return (jc, jm, jg, jp, jpara, jperp, ja)


def plot_jdotes_runs():
    """Plot jdote for multiple runs
    """
    # jdotes = calc_jdotes_fraction_multi('e')
    # jdotes = np.array(jdotes)
    fname_e = '../data/jdotes-guide-e.dat'
    # jdotes.tofile(fname_e)
    # jdotes = calc_jdotes_fraction_multi('i')
    # jdotes = np.array(jdotes)
    fname_i = '../data/jdotes-guide-i.dat'
    # jdotes.tofile(fname_i)
    jdotes_e = np.fromfile(fname_e)
    jdotes_i = np.fromfile(fname_i)
    jc_e, jm_e, jg_e, jp_e, jpara_e, jperp_e, ja_e = \
            get_jdrifts(jdotes_e)
    jc_i, jm_i, jg_i, jp_i, jpara_i, jperp_i, ja_i = \
            get_jdrifts(jdotes_i)
    bg = [0, 0.2, 0.5, 1.0, 4.0]
    fig = plt.figure(figsize=[7, 5])
    xs, ys = 0.13, 0.16
    w1, h1 = 0.8, 0.8
    ax = fig.add_axes([xs, ys, w1, h1])
    ax.scatter(bg, jpara_e, s=100, color='r')
    ax.scatter(bg, jperp_e, s=100, color='r', marker=">")
    ax.scatter(bg, jpara_i, s=100, color='b')
    ax.scatter(bg, jperp_i, s=100, color='b', marker=">")
    ax.set_xlabel(r'$B_g/B_0$', fontdict=font, fontsize=24)
    ax.tick_params(labelsize=20)
    ax.set_xlim([-0.1, 4.5])
    ax.set_ylim([0, 1])
    fig.savefig('../img/jpp-dote-guide.eps')
    plt.show()


if __name__ == "__main__":
    # plot_energy_evolution(pic_info)
    # plot_dke_multi()
    # plot_jy_multi()
    # plot_jy_guide()
    plot_jdotes_runs()
