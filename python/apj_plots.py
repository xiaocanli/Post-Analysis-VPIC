"""
Functions and classes for 2D contour plots of fields.
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LogNorm
from matplotlib import rc
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
import color_maps as cm
import colormap.colormaps as cmaps
from runs_name_path import ApJ_long_paper_runs
from energy_conversion import read_data_from_json
from contour_plots import read_2d_fields, plot_2d_contour
from pic_information import list_pic_info_dir
import palettable
import sys
from fields_plot import *
from spectrum_fitting import calc_nonthermal_fraction
from energy_conversion import calc_jdotes_fraction_multi

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

font = {'family': 'serif',
        # 'color':'darkred',
        'color': 'black',
        'weight': 'normal',
        'size': 24,
        }

colors = palettable.colorbrewer.qualitative.Set1_9.mpl_colors
# colors = palettable.colorbrewer.qualitative.Dark2_8.mpl_colors


def plot_by_time(run_name, root_dir, pic_info):
    """Plot by contour at multiple time frames

    Args:
        run_name: the name of this run.
        root_dir: the root directory of this run.
        pic_info: PIC simulation information in a namedtuple.
    """
    ct = 80
    nt = 3
    contour_color = ['k'] * nt
    vmin = [-1.0] * nt
    vmax = [1.0] * nt
    xs, ys = 0.18, 0.7
    w1, h1 = 0.68, 0.28
    fig_sizes = (5, 6)
    axis_pos = [xs, ys, w1, h1]
    gaps = [0.1, 0.02]
    nxp, nzp = 1, nt
    cts = [60, 152.5, 800]
    cts = np.asarray(cts)
    var_names = []
    for i in range(nt):
        var_name = r'$t=' + str(cts[i]) + r'/\Omega_{ci}$'
        var_names.append(var_name)
    cts /= pic_info.dt_fields
    cts = np.asarray(cts-1, dtype=int)
    colormaps = ['seismic'] * nt
    text_colors = ['k'] * nt
    xstep, zstep = 2, 2
    is_logs = [False] * nt
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    dir = '../img/img_apj/'
    if not os.path.isdir(dir):
        os.makedirs(dir)
    fig_dir = dir + run_name + '/'
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    kwargs = {"current_time": ct, "xl": 0, "xr": 200, "zb": -50, "zt": 50}
    fname1 = root_dir + 'data/by.gda'
    fname2 = root_dir + 'data/Ay.gda'
    fdata = []
    Ay_data = []
    for i in range(nt):
        kwargs["current_time"] = cts[i]
        x, z, data = read_2d_fields(pic_info, fname1, **kwargs)
        x, z, Ay = read_2d_fields(pic_info, fname2, **kwargs)
        fdata.append(data)
        Ay_data.append(Ay)
    fname = 'by_time'
    kwargs_plots = {'current_time': ct, 'x': x, 'z': z, 'Ay': Ay_data,
                    'fdata': fdata, 'contour_color': contour_color,
                    'colormaps': colormaps, 'vmin': vmin, 'vmax': vmax,
                    'var_names': var_names, 'axis_pos': axis_pos, 'gaps': gaps,
                    'fig_sizes': fig_sizes, 'text_colors': text_colors,
                    'nxp': nxp, 'nzp': nzp, 'xstep': xstep, 'zstep': zstep,
                    'is_logs': is_logs, 'fname': fname, 'fig_dir': fig_dir,
                    'is_multi_Ay': True, 'save_eps': True}
    by_plot = PlotMultiplePanels(**kwargs_plots)
    for cbar in by_plot.cbar:
        cbar.set_ticks(np.arange(-0.8, 0.9, 0.4))
    xbox = [104, 134, 134, 104, 104]
    zbox = [-15, -15, 15, 15, -15]
    by_plot.ax[1].plot(xbox, zbox, color='k')
    by_plot.save_figures()
    plt.show()


def plot_vx_time(run_name, root_dir, pic_info):
    """Plot jdote due to different drift current

    Args:
        run_name: the name of this run.
        root_dir: the root directory of this run.
        pic_info: PIC simulation information in a namedtuple.
    """
    ct = 0
    nt = 2
    contour_color = ['k'] * nt
    vmin = [-1.0] * nt
    vmax = [1.0] * nt
    xs, ys = 0.18, 0.7
    w1, h1 = 0.68, 0.28
    fig_sizes = (5, 6)
    axis_pos = [xs, ys, w1, h1]
    gaps = [0.1, 0.02]
    nxp, nzp = 1, nt
    cts = [60, 152.5]
    cts = np.asarray(cts)
    var_names = []
    for i in range(nt):
        var_name = r'$t=' + str(cts[i]) + r'/\Omega_{ci}$'
        var_names.append(var_name)
    cts /= pic_info.dt_fields
    cts = np.asarray(cts-1, dtype=int)
    colormaps = ['seismic'] * nt
    text_colors = [colors[0], colors[1]]
    xstep, zstep = 2, 2
    is_logs = [False] * nt
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    dir = '../img/img_apj/'
    if not os.path.isdir(dir):
        os.makedirs(dir)
    fig_dir = dir + run_name + '/'
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    kwargs = {"current_time": ct, "xl": 0, "xr": 200, "zb": -50, "zt": 50}
    fname11 = root_dir + 'data/vex.gda'
    if not os.path.isfile(fname11):
        fname11 = root_dir + 'data/uex.gda'
    fname12 = root_dir + 'data/ne.gda'
    fname21 = root_dir + 'data/vix.gda'
    if not os.path.isfile(fname21):
        fname21 = root_dir + 'data/uix.gda'
    fname22 = root_dir + 'data/ni.gda'
    fname3 = root_dir + 'data/Ay.gda'
    mime = pic_info.mime
    fdata = []
    fdata_1d = []
    Ay_data = []
    wpe_wce = pic_info.dtwce / pic_info.dtwpe
    va = wpe_wce / math.sqrt(pic_info.mime)  # Alfven speed of inflow region
    for i in range(nt):
        kwargs["current_time"] = cts[i]
        x, z, vex = read_2d_fields(pic_info, fname11, **kwargs)
        x, z, ne = read_2d_fields(pic_info, fname12, **kwargs)
        x, z, vix = read_2d_fields(pic_info, fname21, **kwargs)
        x, z, ni = read_2d_fields(pic_info, fname22, **kwargs)
        ux = (ne*vex + ni*vix*mime) / (ne + ni*mime)
        ux /= va
        x, z, Ay = read_2d_fields(pic_info, fname3, **kwargs)
        nx, = x.shape
        nz, = z.shape
        fdata.append(ux)
        fdata_1d.append(ux[nz/2, :])
        Ay_data.append(Ay)
    fname = 'vx_time'
    fdata = np.asarray(fdata)
    fdata_1d = np.asarray(fdata_1d)
    fname = 'vx_time'
    bottom_panel = True
    xlim = [0, 200]
    zlim = [-50, 50]
    save_eps = True
    kwargs_plots = {'current_time': ct, 'x': x, 'z': z, 'Ay': Ay_data,
                    'fdata': fdata, 'contour_color': contour_color,
                    'colormaps': colormaps, 'vmin': vmin, 'vmax': vmax,
                    'var_names': var_names, 'axis_pos': axis_pos, 'gaps': gaps,
                    'fig_sizes': fig_sizes, 'text_colors': text_colors,
                    'nxp': nxp, 'nzp': nzp, 'xstep': xstep, 'zstep': zstep,
                    'is_logs': is_logs, 'fname': fname, 'fig_dir': fig_dir,
                    'bottom_panel': bottom_panel, 'fdata_1d': fdata_1d,
                    'xlim': xlim, 'zlim': zlim, 'is_multi_Ay': True,
                    'save_eps': save_eps}
    vx_plot = PlotMultiplePanels(**kwargs_plots)
    for cbar in vx_plot.cbar:
        cbar.set_ticks(np.arange(-0.8, 0.9, 0.4))
    vx_plot.ax1d.set_ylabel(r'$v_x/v_A$', fontdict=font, fontsize=20)
    xmin = np.min(x)
    xmax = np.max(x)
    zmin = np.min(z)
    zmax = np.max(z)
    z0 = z[nz/2]
    vx_plot.ax[0].plot([xmin, xmax], [z0, z0], linestyle='--', color='k')
    vx_plot.ax[1].plot([xmin, xmax], [z0, z0], linestyle='--', color='k')
    # x0 = 127
    # vx_plot.ax[1].plot([x0, x0], [zmin, zmax], linestyle='--', color='k')
    xbox = [104, 134, 134, 104, 104]
    zbox = [-15, -15, 15, 15, -15]
    vx_plot.ax[1].plot(xbox, zbox, color='k')
    vx_plot.save_figures()
    plt.show()


def plot_epara_eperp(pic_info, ct, root_dir='../../'):
    kwargs = {"current_time": ct, "xl": 50, "xr": 150, "zb": -10, "zt": 10}
    # kwargs = {"current_time": ct, "xl": 0, "xr": 200, "zb": -50, "zt": 50}
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

    absE = np.sqrt(ex*ex + ey*ey + ez*ez)
    epara = (ex*bx + ey*by + ez*bz) / absB
    eperp = np.sqrt(absE*absE - epara*epara)
    ng = 3
    kernel = np.ones((ng, ng)) / float(ng*ng)
    epara = signal.convolve2d(epara, kernel)
    eperp = signal.convolve2d(eperp, kernel)
    ey = signal.convolve2d(ey, kernel, 'same')

    wpe_wce = pic_info.dtwce / pic_info.dtwpe
    va = wpe_wce / math.sqrt(pic_info.mime)  # Alfven speed of inflow region
    b0 = pic_info.b0
    e0 = va * b0
    epara /= 0.5*e0
    ey /= 0.5*e0

    contour_color = ['k'] * 2
    vmin = [-1.0] * 2
    vmax = [1.0] * 2
    xs, ys = 0.18, 0.58
    w1, h1 = 0.68, 0.38
    fig_sizes = (5, 4)
    axis_pos = [xs, ys, w1, h1]
    gaps = [0.1, 0.05]
    nxp, nzp = 1, 2
    var_names = [r'$E_\parallel$', r'$E_y$']
    colormaps = ['seismic', 'seismic']
    text_colors = ['k', 'k']
    xstep, zstep = 2, 2
    is_logs = [False] * 2
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    dir = '../img/img_apj/'
    if not os.path.isdir(dir):
        os.makedirs(dir)
    fig_dir = dir + run_name + '/'
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)

    fdata = [epara, ey]
    Ay_data = Ay
    fname = 'epara_ey'
    fdata = np.asarray(fdata)
    xlim = [50, 150]
    zlim = [-10, 10]
    # xlim = [0, 200]
    # zlim = [-50, 50]
    save_eps = True
    kwargs_plots = {'current_time': ct, 'x': x, 'z': z, 'Ay': Ay_data,
                    'fdata': fdata, 'contour_color': contour_color,
                    'colormaps': colormaps, 'vmin': vmin, 'vmax': vmax,
                    'var_names': var_names, 'axis_pos': axis_pos, 'gaps': gaps,
                    'fig_sizes': fig_sizes, 'text_colors': text_colors,
                    'nxp': nxp, 'nzp': nzp, 'xstep': xstep, 'zstep': zstep,
                    'is_logs': is_logs, 'fname': fname, 'fig_dir': fig_dir,
                    'xlim': xlim, 'zlim': zlim, 'is_multi_Ay': False,
                    'save_eps': save_eps}
    vx_plot = PlotMultiplePanels(**kwargs_plots)
    for cbar in vx_plot.cbar:
        cbar.set_ticks(np.arange(-0.8, 0.9, 0.4))
    vx_plot.save_figures()
    plt.show()

    # nx, = x.shape
    # nz, = z.shape
    # width = 0.73
    # height = 0.36
    # xs = 0.15
    # ys = 0.92 - height
    # gap = 0.05

    # fig = plt.figure(figsize=[7, 5])
    # ax1 = fig.add_axes([xs, ys, width, height])
    # kwargs_plot = {"xstep":2, "zstep":2, "vmin":-0.1, "vmax":0.1}
    # xstep = kwargs_plot["xstep"]
    # zstep = kwargs_plot["zstep"]
    # p1, cbar1 = plot_2d_contour(x, z, ey, ax1, fig, **kwargs_plot)
    # # p1.set_cmap(cmaps.inferno)
    # p1.set_cmap(plt.cm.get_cmap('seismic'))
    # Ay_min = np.min(Ay)
    # Ay_max = np.max(Ay)
    # levels = np.linspace(Ay_min, Ay_max, 10)
    # ax1.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep],
    #         colors='k', linewidths=0.5)
    # ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    # # ax1.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
    # ax1.tick_params(axis='x', labelbottom='off')
    # ax1.tick_params(labelsize=16)
    # # cbar1.ax.set_ylabel(r'$E_\perp$', fontdict=font, fontsize=20)
    # cbar1.set_ticks(np.arange(-0.08, 0.1, 0.04))
    # cbar1.ax.tick_params(labelsize=16)
    # ax1.text(0.02, 0.8, r'$E_y$', color='k', fontsize=20,
    #         bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
    #             pad=10.0), horizontalalignment='left',
    #         verticalalignment='center', transform = ax1.transAxes)

    # ys -= height + gap
    # ax2 = fig.add_axes([xs, ys, width, height])
    # kwargs_plot = {"xstep":1, "zstep":1, "vmin":-0.05, "vmax":0.05}
    # p2, cbar2 = plot_2d_contour(x, z, epara, ax2, fig, **kwargs_plot)
    # p2.set_cmap(plt.cm.seismic)
    # # p2.set_cmap(cmaps.plasma)
    # ax2.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep],
    #         colors='black', linewidths=0.5)
    # ax2.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    # ax2.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    # ax2.tick_params(labelsize=16)
    # # cbar2.ax.set_ylabel(r'$E_\parallel$', fontdict=font, fontsize=24)
    # cbar2.set_ticks(np.arange(-0.04, 0.05, 0.02))
    # cbar2.ax.tick_params(labelsize=16)
    # ax2.text(0.02, 0.8, r'$E_\parallel$', color='k', fontsize=20,
    #          bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
    #                    pad=10.0), horizontalalignment='left',
    #          verticalalignment='center', transform=ax2.transAxes)

    # # dtf = pic_info.dt_fields
    # dtf = 2.5
    # t_wci = (ct + 1) * dtf
    # title = r'$t = ' + "{:10.1f}".format(t_wci) + '/\Omega_{ci}$'
    # ax1.set_title(title, fontdict=font, fontsize=20)

    # if not os.path.isdir('../img/'):
    #     os.makedirs('../img/')
    # dir = '../img/img_apj/'
    # if not os.path.isdir(dir):
    #     os.makedirs(dir)
    # fname = dir + 'epara_perp' + '_' + str(ct).zfill(3) + '.eps'
    # fig.savefig(fname)

    plt.show()
    # plt.close()


def plot_jpara_dote(run_name, root_dir, pic_info, species):
    """Plot jdote due to different parallel current

    Args:
        run_name: the name of this run.
        root_dir: the root directory of this run.
        pic_info: PIC simulation information in a namedtuple.
        species: particle species.
    """
    ct = 0
    nj = 1
    contour_color = ['k'] * nj
    vmin = [-1.0] * nj
    vmax = [1.0] * nj
    xs, ys = 0.11, 0.59
    w1, h1 = 0.8, 0.38
    axis_pos = [xs, ys, w1, h1]
    gaps = [0.1, 0.05]
    fig_sizes = (8, 4)
    nxp, nzp = 1, nj
    var_sym = ['\parallel']
    var_names = []
    for var in var_sym:
        var_name = r'$\boldsymbol{j}_' + var + r'\cdot\boldsymbol{E}$'
        var_names.append(var_name)
    colormaps = ['seismic'] * nj
    text_colors = colors[0:nj]
    xstep, zstep = 2, 2
    is_logs = [False] * nj
    wpe_wce = pic_info.dtwce / pic_info.dtwpe
    va = wpe_wce / math.sqrt(pic_info.mime)  # Alfven speed of inflow region
    b0 = pic_info.b0
    j0 = 0.1 * va**2 * b0
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    dir = '../img/img_jpara_dote/'
    if not os.path.isdir(dir):
        os.makedirs(dir)
    fig_dir = dir + run_name + '/'
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    kwargs = {"current_time": ct, "xl": 0, "xr": 200, "zb": -50, "zt": 50}
    fnames = []
    fname = root_dir + 'data1/jqnvpara_dote00_' + species + '.gda'
    fnames.append(fname)
    dv = pic_info.dx_di * pic_info.dz_di * pic_info.mime
    ng = 3
    kernel = np.ones((ng, ng)) / float(ng*ng)
    fdata = []
    fdata_1d = []
    for fname in fnames:
        x, z, data = read_2d_fields(pic_info, fname, **kwargs)
        fdata_cum = np.cumsum(np.sum(data, axis=0)) * dv
        fdata_1d.append(fdata_cum)
        data_new = signal.convolve2d(data, kernel, 'same')
        fdata.append(data_new)
    fdata = np.asarray(fdata)
    fdata_1d = np.asarray(fdata_1d)
    fdata /= j0  # Normalization
    fname2 = root_dir + 'data/Ay.gda'
    x, z, Ay = read_2d_fields(pic_info, fname2, **kwargs)
    fname = 'jpara_dote_' + species
    bottom_panel = True
    xlim = [0, 200]
    zlim = [-25, 25]
    kwargs_plots = {'current_time': ct, 'x': x, 'z': z, 'Ay': Ay,
                    'fdata': fdata, 'contour_color': contour_color,
                    'colormaps': colormaps, 'vmin': vmin, 'vmax': vmax,
                    'var_names': var_names, 'axis_pos': axis_pos, 'gaps': gaps,
                    'fig_sizes': fig_sizes, 'text_colors': text_colors,
                    'nxp': nxp, 'nzp': nzp, 'xstep': xstep, 'zstep': zstep,
                    'is_logs': is_logs, 'fname': fname, 'fig_dir': fig_dir,
                    'bottom_panel': bottom_panel, 'fdata_1d': fdata_1d,
                    'xlim': xlim, 'zlim': zlim}
    jdote_plot = PlotMultiplePanels(**kwargs_plots)
    for ct in range(1, pic_info.ntf):
        kwargs["current_time"] = ct
        fdata = []
        fdata_1d = []
        for fname in fnames:
            x, z, data = read_2d_fields(pic_info, fname, **kwargs)
            fdata_cum = np.cumsum(np.sum(data, axis=0)) * dv
            fdata_1d.append(fdata_cum)
            data_new = signal.convolve2d(data, kernel, 'same')
            fdata.append(data_new)
        fdata = np.asarray(fdata)
        fdata_1d = np.asarray(fdata_1d)
        fdata /= j0  # Normalization
        x, z, Ay = read_2d_fields(pic_info, fname2, **kwargs)
        jdote_plot.update_plot_1d(fdata_1d)
        jdote_plot.update_fields(ct, fdata, Ay)

    plt.close()
    # plt.show()


def plot_jdotes_fields(run_name, root_dir, pic_info, species, ct, srange,
                       axis_pos, gaps, fig_sizes):
    """Plot jdote due to different drift current

    Args:
        run_name: the name of this run.
        root_dir: the root directory of this run.
        pic_info: PIC simulation information in a namedtuple.
        species: particle species.
        ct: current time frame
        srange: spatial range
    """
    nj = 5
    contour_color = ['k'] * nj
    vmin = [-1.0] * nj
    vmax = [1.0] * nj
    nxp, nzp = 1, nj
    var_sym = ['c', 'g', 'm', 'p', 'a', '\parallel', '\perp']
    var_names = []
    for var in var_sym:
        var_name = r'$\boldsymbol{j}_' + var + r'\cdot\boldsymbol{E}$'
        var_names.append(var_name)
    colormaps = ['seismic'] * nj
    text_colors = ['b', 'g', 'r', 'c', 'm']
    xstep, zstep = 1, 1
    is_logs = [False] * nj
    wpe_wce = pic_info.dtwce / pic_info.dtwpe
    va = wpe_wce / math.sqrt(pic_info.mime)  # Alfven speed of inflow region
    b0 = pic_info.b0
    j0 = 0.1 * va**2 * b0
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    dir = '../img/img_jdotes_apj/'
    if not os.path.isdir(dir):
        os.makedirs(dir)
    fig_dir = dir + run_name + '/'
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    kwargs = {"current_time": ct, "xl": srange[0], "xr": srange[1],
              "zb": srange[2], "zt": srange[3]}
    fnames = []
    fname = root_dir + 'data1/jcpara_dote00_' + species + '.gda'
    fnames.append(fname)
    fname = root_dir + 'data1/jgrad_dote00_' + species + '.gda'
    fnames.append(fname)
    fname = root_dir + 'data1/jmag_dote00_' + species + '.gda'
    fnames.append(fname)
    fname = root_dir + 'data1/jpolar_dote00_' + species + '.gda'
    fnames.append(fname)
    fname = root_dir + 'data1/jagy_dote00_' + species + '.gda'
    fnames.append(fname)
    # fname = root_dir + 'data1/jqnvpara_dote00_' + species + '.gda'
    # fnames.append(fname)
    # fname = root_dir + 'data1/jqnvperp_dote00_' + species + '.gda'
    # fnames.append(fname)
    dv = pic_info.dx_di * pic_info.dz_di * pic_info.mime
    ng = 3
    kernel = np.ones((ng, ng)) / float(ng*ng)
    fdata = []
    fdata_1d = []
    for fname in fnames:
        x, z, data = read_2d_fields(pic_info, fname, **kwargs)
        fdata_cum = np.cumsum(np.sum(data, axis=0)) * dv
        fdata_1d.append(fdata_cum)
        data_new = signal.convolve2d(data, kernel, 'same')
        fdata.append(data_new)
    fdata = np.asarray(fdata)
    fdata_1d = np.asarray(fdata_1d)
    fdata /= j0  # Normalization
    fname2 = root_dir + 'data/Ay.gda'
    x, z, Ay = read_2d_fields(pic_info, fname2, **kwargs)
    fname = 'jdotes_' + species
    bottom_panel = True
    xlim = srange[:2]
    zlim = srange[2:]
    nlevels_contour = 10
    kwargs_plots = {'current_time': ct, 'x': x, 'z': z, 'Ay': Ay,
                    'fdata': fdata, 'contour_color': contour_color,
                    'colormaps': colormaps, 'vmin': vmin, 'vmax': vmax,
                    'var_names': var_names, 'axis_pos': axis_pos, 'gaps': gaps,
                    'fig_sizes': fig_sizes, 'text_colors': text_colors,
                    'nxp': nxp, 'nzp': nzp, 'xstep': xstep, 'zstep': zstep,
                    'is_logs': is_logs, 'fname': fname, 'fig_dir': fig_dir,
                    'bottom_panel': bottom_panel, 'fdata_1d': fdata_1d,
                    'xlim': xlim, 'zlim': zlim}
    jdote_plot = PlotMultiplePanels(**kwargs_plots)
    plt.show()


def plot_jdotes_fields_s(run_name, root_dir, pic_info, species, ct, srange,
                         axis_pos, gaps, fig_sizes):
    """Plot jdote due to different drift current

    This will not use the PlotMultiplePanels class

    Args:
        run_name: the name of this run.
        root_dir: the root directory of this run.
        pic_info: PIC simulation information in a namedtuple.
        species: particle species.
        ct: current time frame
        srange: spatial range
    """
    ng = 3
    kwargs = {"current_time": ct, "xl": srange[0], "xr": srange[1],
              "zb": srange[2], "zt": srange[3]}
    dmax = []
    dmin = []
    kernel = np.ones((ng, ng)) / float(ng*ng)
    dv = pic_info.dx_di * pic_info.dz_di * pic_info.mime
    fname = root_dir + 'data1/jcpara_dote00_' + species + '.gda'
    x, z, jcpara_dote = read_2d_fields(pic_info, fname, **kwargs)
    jcpara_dote_cum = np.cumsum(np.sum(jcpara_dote, axis=0)) * dv
    dmin.append(np.min(jcpara_dote_cum))
    dmax.append(np.max(jcpara_dote_cum))
    fname = root_dir + 'data1/jgrad_dote00_' + species + '.gda'
    x, z, jgrad_dote = read_2d_fields(pic_info, fname, **kwargs)
    jgrad_dote_cum = np.cumsum(np.sum(jgrad_dote, axis=0)) * dv
    dmin.append(np.min(jgrad_dote_cum))
    dmax.append(np.max(jgrad_dote_cum))
    fname = root_dir + 'data1/jmag_dote00_' + species + '.gda'
    x, z, jmag_dote = read_2d_fields(pic_info, fname, **kwargs)
    jmag_dote_cum = np.cumsum(np.sum(jmag_dote, axis=0)) * dv
    dmin.append(np.min(jmag_dote_cum))
    dmax.append(np.max(jmag_dote_cum))
    fname = root_dir + 'data1/jpolar_dote00_' + species + '.gda'
    x, z, jpolar_dote = read_2d_fields(pic_info, fname, **kwargs)
    jpolar_dote_cum = np.cumsum(np.sum(jpolar_dote, axis=0)) * dv
    dmin.append(np.min(jpolar_dote_cum))
    dmax.append(np.max(jpolar_dote_cum))
    fname = root_dir + 'data1/jagy_dote00_' + species + '.gda'
    x, z, jagy_dote = read_2d_fields(pic_info, fname, **kwargs)
    jagy_dote_cum = np.cumsum(np.sum(jagy_dote, axis=0)) * dv
    dmin.append(np.min(jagy_dote_cum))
    dmax.append(np.max(jagy_dote_cum))
    fname = root_dir + 'data1/jqnvpara_dote00_' + species + '.gda'
    x, z, jqnvpara_dote = read_2d_fields(pic_info, fname, **kwargs)
    jqnvpara_dote_cum = np.cumsum(np.sum(jqnvpara_dote, axis=0)) * dv
    dmin.append(np.min(jqnvpara_dote_cum))
    dmax.append(np.max(jqnvpara_dote_cum))

    min_jdote = min(dmin) * 1.1
    max_jdote = max(dmax) * 1.1

    jcpara_dote = signal.convolve2d(jcpara_dote, kernel, 'same')
    jgrad_dote = signal.convolve2d(jgrad_dote, kernel, 'same')
    jmag_dote = signal.convolve2d(jmag_dote, kernel, 'same')
    jpolar_dote = signal.convolve2d(jpolar_dote, kernel, 'same')
    jagy_dote = signal.convolve2d(jagy_dote, kernel, 'same')
    jqnvpara_dote = signal.convolve2d(jqnvpara_dote, kernel, 'same')

    wpe_wce = pic_info.dtwce / pic_info.dtwpe
    va = wpe_wce / math.sqrt(pic_info.mime)  # Alfven speed of inflow region
    b0 = pic_info.b0
    j0 = 0.1 * va**2 * b0
    jcpara_dote /= j0  # Normalization
    jgrad_dote /= j0
    jmag_dote /= j0
    jpolar_dote /= j0
    jagy_dote /= j0
    jqnvpara_dote /= j0
    fname2 = root_dir + 'data/Ay.gda'
    x, z, Ay = read_2d_fields(pic_info, fname2, **kwargs)

    xmin, xmax = np.min(x), np.max(x)
    zmin, zmax = np.min(z), np.max(z)
    vmin, vmax = -1.0, 1.0
    nx, = x.shape
    nz, = z.shape
    fig = plt.figure(figsize=[9, 8])
    xs0, ys0 = 0.1, 0.7
    w1, h1 = 0.25, 0.28125
    gap = 0.03
    ax1 = fig.add_axes([xs0, ys0, w1, h1])
    p1 = ax1.imshow(jqnvpara_dote, cmap=plt.cm.seismic,
                    extent=[xmin, xmax, zmin, zmax], aspect='auto',
                    origin='lower', vmin=vmin, vmax=vmax,
                    interpolation='bicubic')
    ax1.tick_params(axis='x', labelbottom='off')
    ax1.tick_params(labelsize=16)
    ax1.contour(x, z, Ay, colors='black', linewidths=0.5)
    ax1.set_ylabel(r'$z/d_i$', fontsize=20)
    ax1.text(0.02, 0.85, r'$\boldsymbol{j}_\parallel\cdot\boldsymbol{E}$',
             color='k', fontsize=20, bbox=dict(facecolor='none', alpha=1.0,
                                               edgecolor='none', pad=10.0),
             horizontalalignment='left', verticalalignment='center',
             transform=ax1.transAxes)
    xs = xs0 + gap + w1
    ys = ys0
    ax2 = fig.add_axes([xs, ys, w1, h1])
    p2 = ax2.imshow(jcpara_dote, cmap=plt.cm.seismic,
                    extent=[xmin, xmax, zmin, zmax], aspect='auto',
                    origin='lower', vmin=vmin, vmax=vmax,
                    interpolation='bicubic')
    ax2.tick_params(axis='x', labelbottom='off')
    ax2.tick_params(axis='y', labelleft='off')
    ax2.contour(x, z, Ay, colors='black', linewidths=0.5)
    ax2.text(0.02, 0.85, r'$\boldsymbol{j}_c\cdot\boldsymbol{E}$',
             color=colors[0], fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax2.transAxes)
    xs = xs + gap + w1
    ax3 = fig.add_axes([xs, ys, w1, h1])
    p3 = ax3.imshow(jpolar_dote, cmap=plt.cm.seismic,
                    extent=[xmin, xmax, zmin, zmax], aspect='auto',
                    origin='lower', vmin=vmin, vmax=vmax,
                    interpolation='bicubic')
    ax3.tick_params(axis='x', labelbottom='off')
    ax3.tick_params(axis='y', labelleft='off')
    ax3.contour(x, z, Ay, colors='black', linewidths=0.5)
    ax3.text(0.02, 0.85, r'$\boldsymbol{j}_p\cdot\boldsymbol{E}$',
             color=colors[3], fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax3.transAxes)

    ys = ys0 - h1 - gap
    ax4 = fig.add_axes([xs0, ys, w1, h1])
    p4 = ax4.imshow(jmag_dote, cmap=plt.cm.seismic,
                    extent=[xmin, xmax, zmin, zmax], aspect='auto',
                    origin='lower', vmin=vmin, vmax=vmax,
                    interpolation='bicubic')
    ax4.tick_params(labelsize=16)
    ax4.tick_params(axis='x', labelbottom='off')
    ax4.contour(x, z, Ay, colors='black', linewidths=0.5)
    ax4.set_ylabel(r'$z/d_i$', fontsize=20)
    ax4.text(0.02, 0.85, r'$\boldsymbol{j}_m\cdot\boldsymbol{E}$',
             color=colors[2], fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax4.transAxes)
    xs = xs0 + gap + w1
    ax5 = fig.add_axes([xs, ys, w1, h1])
    p5 = ax5.imshow(jgrad_dote, cmap=plt.cm.seismic,
                    extent=[xmin, xmax, zmin, zmax], aspect='auto',
                    origin='lower', vmin=vmin, vmax=vmax,
                    interpolation='bicubic')
    ax5.tick_params(axis='x', labelbottom='off')
    ax5.tick_params(axis='y', labelleft='off')
    ax5.contour(x, z, Ay, colors='black', linewidths=0.5)
    ax5.text(0.02, 0.85, r'$\boldsymbol{j}_g\cdot\boldsymbol{E}$',
             color=colors[2], fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax5.transAxes)
    xs = xs + gap + w1
    ax6 = fig.add_axes([xs, ys, w1, h1])
    ax6.tick_params(axis='x', labelbottom='off')
    ax6.tick_params(axis='y', labelleft='off')
    p6 = ax6.imshow(jagy_dote, cmap=plt.cm.seismic,
                    extent=[xmin, xmax, zmin, zmax], aspect='auto',
                    origin='lower', vmin=vmin, vmax=vmax,
                    interpolation='bicubic')
    ax6.contour(x, z, Ay, colors='black', linewidths=0.5)
    ax6.text(0.02, 0.85, r'$\boldsymbol{j}_a\cdot\boldsymbol{E}$',
             color=colors[4], fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax6.transAxes)

    h2 = 2*h1 + gap
    cbar_ax = fig.add_axes([xs+w1+0.01, ys0-h1-gap, 0.02, h2])
    cbar1 = fig.colorbar(p1, cax=cbar_ax)
    cbar1.ax.tick_params(labelsize=16)

    ys = ys - h1 - gap
    ax7 = fig.add_axes([xs0, ys, w1, h1])
    p71, = ax7.plot(x, jqnvpara_dote_cum, linewidth=2, color='k')
    p72, = ax7.plot(x, jmag_dote_cum, linewidth=2, color=colors[2])
    p73, = ax7.plot([xmin, xmax], [0, 0], linewidth=0.5, linestyle='--')
    ax7.set_xlabel(r'$x/d_i$', fontsize=20)
    ax7.set_ylabel('Accumulation', fontsize=20)
    ax7.tick_params(labelsize=16)
    ax7.set_xlim([xmin, xmax])
    ax7.set_ylim([min_jdote, max_jdote])
    xs = xs0 + gap + w1
    ax8 = fig.add_axes([xs, ys, w1, h1])
    p81, = ax8.plot(x, jcpara_dote_cum, linewidth=2, color=colors[0])
    p82, = ax8.plot(x, jgrad_dote_cum, linewidth=2, color=colors[1])
    p83, = ax8.plot([xmin, xmax], [0, 0], linewidth=0.5, linestyle='--')
    ax8.set_xlabel(r'$x/d_i$', fontsize=20)
    ax8.tick_params(axis='y', labelleft='off')
    ax8.tick_params(labelsize=16)
    ax8.set_xlim([xmin, xmax])
    ax8.set_ylim([min_jdote, max_jdote])
    xs = xs + gap + w1
    ax9 = fig.add_axes([xs, ys, w1, h1])
    p91, = ax9.plot(x, jpolar_dote_cum, linewidth=2, color=colors[3])
    p92, = ax9.plot(x, jagy_dote_cum, linewidth=2, color=colors[4])
    p93, = ax9.plot([xmin, xmax], [0, 0], linewidth=0.5, linestyle='--')
    ax9.set_xlabel(r'$x/d_i$', fontsize=20)
    ax9.tick_params(axis='y', labelleft='off')
    ax9.tick_params(labelsize=16)
    ax9.set_xlim([xmin, xmax])
    ax9.set_ylim([min_jdote, max_jdote])

    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    dir = '../img/img_jdotes_apj/'
    path = '../img/img_jdotes_apj/'
    if not os.path.isdir(path):
        os.makedirs(path)
    fig_dir = path + run_name + '/'
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    # fname = 'jdotes_' + species + str(ct).zfill(3) + '.jpg'
    # fig.savefig(fig_dir + fname, dpi=200)
    fname = 'jdotes_' + species + str(ct).zfill(3) + '.eps'
    fig.savefig(fig_dir + fname)
    plt.show()


def plot_curvb_single(run_name, root_dir, pic_info, srange, ct):
    """Plot the curvature of the magnetic field
    """
    kwargs = {"current_time": ct, "xl": srange[0], "xr": srange[1],
              "zb": srange[2], "zt": srange[3]}
    fname = root_dir + 'data/bx.gda'
    x, z, bx = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/by.gda'
    x, z, by = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/bz.gda'
    x, z, bz = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/absB.gda'
    x, z, absB = read_2d_fields(pic_info, fname, **kwargs)
    fname2 = root_dir + 'data/Ay.gda'
    x, z, Ay = read_2d_fields(pic_info, fname2, **kwargs)
    wpe_wce = pic_info.dtwce / pic_info.dtwpe
    va = wpe_wce / math.sqrt(pic_info.mime)  # Alfven speed of inflow region
    b0 = pic_info.b0

    iabsB2 = 1.0 / absB**2

    bx *= iabsB2
    by *= iabsB2
    bz *= iabsB2
    absB *= iabsB2

    nx, = x.shape
    nz, = z.shape
    curvb_x = np.zeros((nz, nx))
    curvb_y = np.zeros((nz, nx))
    curvb_z = np.zeros((nz, nx))
    curvb_x[:nz-1, :] = -np.diff(by, axis=0)
    curvb_y[:nz-1, :] = np.diff(bx, axis=0)
    curvb_y[:, :nx-1] += np.diff(bz)
    curvb_z[:, :nx-1] = np.diff(by)

    xmin, xmax = np.min(x), np.max(x)
    zmin, zmax = np.min(z), np.max(z)
    vmin, vmax = -1.0, 1.0
    fig = plt.figure(figsize=[6, 8])
    xs0, ys0 = 0.15, 0.7
    w1, h1 = 0.375, 0.28125
    gap = 0.03
    ax1 = fig.add_axes([xs0, ys0, w1, h1])
    print np.min(curvb_z), np.max(curvb_z)
    p1 = ax1.imshow(curvb_x, cmap=plt.cm.seismic,
                    extent=[xmin, xmax, zmin, zmax], aspect='auto',
                    origin='lower', vmin=vmin, vmax=vmax,
                    interpolation='bicubic')
    ax1.tick_params(axis='x', labelbottom='off')
    ax1.tick_params(labelsize=16)
    ax1.contour(x, z, Ay, colors='black', linewidths=0.5)
    ax1.set_ylabel(r'$z/d_i$', fontsize=20)
    ax1.text(0.02, 0.85, r'$B_x$', color='k', fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax1.transAxes)
    ys = ys0 - h1 - gap
    ax2 = fig.add_axes([xs0, ys, w1, h1])
    p2 = ax2.imshow(curvb_y, cmap=plt.cm.seismic,
                    extent=[xmin, xmax, zmin, zmax], aspect='auto',
                    origin='lower', vmin=vmin, vmax=vmax,
                    interpolation='bicubic')
    ax2.tick_params(axis='x', labelbottom='off')
    ax2.tick_params(labelsize=16)
    ax2.contour(x, z, Ay, colors='black', linewidths=0.5)
    ax2.set_ylabel(r'$z/d_i$', fontsize=20)
    ax2.text(0.02, 0.85, r'$B_y$', color='k', fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax2.transAxes)
    ys = ys - h1 - gap
    ax3 = fig.add_axes([xs0, ys, w1, h1])
    p3 = ax3.imshow(curvb_z, cmap=plt.cm.seismic,
                    extent=[xmin, xmax, zmin, zmax], aspect='auto',
                    origin='lower', vmin=vmin, vmax=vmax,
                    interpolation='bicubic')
    ax3.tick_params(labelsize=16)
    ax3.contour(x, z, Ay, colors='black', linewidths=0.5)
    ax3.set_xlabel(r'$x/d_i$', fontsize=20)
    ax3.set_ylabel(r'$z/d_i$', fontsize=20)
    ax3.text(0.02, 0.85, r'$B_z$', color='k', fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax3.transAxes)
    xs = xs0 + w1 + 0.06
    # if not os.path.isdir('../img/'):
    #     os.makedirs('../img/')
    # dir = '../img/img_jdotes_apj/'
    # path = '../img/img_jdotes_apj/'
    # if not os.path.isdir(path):
    #     os.makedirs(path)
    # fig_dir = path + run_name + '/'
    # if not os.path.isdir(fig_dir):
    #     os.makedirs(fig_dir)
    # fname = 'emf_' + str(ct).zfill(3) + '.eps'
    # fig.savefig(fig_dir + fname)
    plt.show()


def plot_curvb_multi(run_name, root_dir, pic_info):
    """Plot the curvature of magnetic field for multiple time steps
    """
    # ct = 61
    # srange = np.asarray([105, 135, -15, 15])
    # plot_curvb_single(run_name, root_dir, pic_info, srange, ct)
    # ct = 92
    # srange = np.asarray([107, 154, -25, 25])
    # plot_curvb_single(run_name, root_dir, pic_info, srange, ct)
    ct = 55
    srange = np.asarray([145, 185, -20, 20])
    plot_curvb_single(run_name, root_dir, pic_info, srange, ct)


def plot_emfields_single(run_name, root_dir, pic_info, srange, ct):
    """Plot the electromagnetic fields for a single time steps
    """
    kwargs = {"current_time": ct, "xl": srange[0], "xr": srange[1],
              "zb": srange[2], "zt": srange[3]}
    fname = root_dir + 'data/bx.gda'
    x, z, bx = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/by.gda'
    x, z, by = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/bz.gda'
    x, z, bz = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/ex.gda'
    x, z, ex = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/ey.gda'
    x, z, ey = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/ez.gda'
    x, z, ez = read_2d_fields(pic_info, fname, **kwargs)
    fname2 = root_dir + 'data/Ay.gda'
    x, z, Ay = read_2d_fields(pic_info, fname2, **kwargs)
    wpe_wce = pic_info.dtwce / pic_info.dtwpe
    va = wpe_wce / math.sqrt(pic_info.mime)  # Alfven speed of inflow region
    b0 = pic_info.b0
    e0 = 0.3 * va * b0

    bx /= b0
    by /= b0
    bz /= b0
    ex /= e0
    ey /= e0
    ez /= e0

    ng = 3
    kernel = np.ones((ng, ng)) / float(ng*ng)
    ex = signal.convolve2d(ex, kernel, 'same')
    ey = signal.convolve2d(ey, kernel, 'same')
    ez = signal.convolve2d(ez, kernel, 'same')

    xmin, xmax = np.min(x), np.max(x)
    zmin, zmax = np.min(z), np.max(z)
    vmin, vmax = -1.0, 1.0
    nx, = x.shape
    nz, = z.shape
    fig = plt.figure(figsize=[6, 8])
    xs0, ys0 = 0.15, 0.7
    w1, h1 = 0.375, 0.28125
    gap = 0.03
    ax1 = fig.add_axes([xs0, ys0, w1, h1])
    p1 = ax1.imshow(bx, cmap=plt.cm.jet,
                    extent=[xmin, xmax, zmin, zmax], aspect='auto',
                    origin='lower', vmin=vmin, vmax=vmax,
                    interpolation='bicubic')
    ax1.tick_params(axis='x', labelbottom='off')
    ax1.tick_params(labelsize=16)
    ax1.contour(x, z, Ay, colors='black', linewidths=0.5)
    ax1.set_ylabel(r'$z/d_i$', fontsize=20)
    ax1.text(0.02, 0.85, r'$B_x$', color='k', fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax1.transAxes)
    ys = ys0 - h1 - gap
    ax2 = fig.add_axes([xs0, ys, w1, h1])
    p2 = ax2.imshow(by, cmap=plt.cm.jet,
                    extent=[xmin, xmax, zmin, zmax], aspect='auto',
                    origin='lower', vmin=vmin, vmax=vmax,
                    interpolation='bicubic')
    ax2.tick_params(axis='x', labelbottom='off')
    ax2.tick_params(labelsize=16)
    ax2.contour(x, z, Ay, colors='black', linewidths=0.5)
    ax2.set_ylabel(r'$z/d_i$', fontsize=20)
    ax2.text(0.02, 0.85, r'$B_y$', color='k', fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax2.transAxes)
    ys = ys - h1 - gap
    ax3 = fig.add_axes([xs0, ys, w1, h1])
    p3 = ax3.imshow(bz, cmap=plt.cm.jet,
                    extent=[xmin, xmax, zmin, zmax], aspect='auto',
                    origin='lower', vmin=vmin, vmax=vmax,
                    interpolation='bicubic')
    ax3.tick_params(labelsize=16)
    ax3.contour(x, z, Ay, colors='black', linewidths=0.5)
    ax3.set_xlabel(r'$x/d_i$', fontsize=20)
    ax3.set_ylabel(r'$z/d_i$', fontsize=20)
    ax3.text(0.02, 0.85, r'$B_z$', color='k', fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax3.transAxes)
    xs = xs0 + w1 + gap * 8.0 / 6.0
    ax4 = fig.add_axes([xs, ys0, w1, h1])
    p4 = ax4.imshow(ex, cmap=plt.cm.seismic,
                    extent=[xmin, xmax, zmin, zmax], aspect='auto',
                    origin='lower', vmin=vmin, vmax=vmax,
                    interpolation='bicubic')
    ax4.tick_params(axis='y', labelleft='off')
    ax4.tick_params(axis='x', labelbottom='off')
    ax4.tick_params(labelsize=16)
    ax4.contour(x, z, Ay, colors='black', linewidths=0.5)
    ax4.text(0.02, 0.85, r'$E_x$', color='k', fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax4.transAxes)
    ys = ys0 - h1 - gap
    ax5 = fig.add_axes([xs, ys, w1, h1])
    p5 = ax5.imshow(ey, cmap=plt.cm.seismic,
                    extent=[xmin, xmax, zmin, zmax], aspect='auto',
                    origin='lower', vmin=vmin, vmax=vmax,
                    interpolation='bicubic')
    ax5.tick_params(axis='y', labelleft='off')
    ax5.tick_params(axis='x', labelbottom='off')
    ax5.tick_params(labelsize=16)
    ax5.contour(x, z, Ay, colors='black', linewidths=0.5)
    ax5.text(0.02, 0.85, r'$E_y$', color='k', fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax5.transAxes)
    ys = ys - h1 - gap
    ax6 = fig.add_axes([xs, ys, w1, h1])
    p6 = ax6.imshow(ez, cmap=plt.cm.seismic,
                    extent=[xmin, xmax, zmin, zmax], aspect='auto',
                    origin='lower', vmin=vmin, vmax=vmax,
                    interpolation='bicubic')
    ax6.tick_params(axis='y', labelleft='off')
    ax6.tick_params(labelsize=16)
    ax6.contour(x, z, Ay, colors='black', linewidths=0.5)
    ax6.set_xlabel(r'$x/d_i$', fontsize=20)
    ax6.text(0.02, 0.85, r'$E_z$', color='k', fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax6.transAxes)
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    dir = '../img/img_jdotes_apj/'
    path = '../img/img_jdotes_apj/'
    if not os.path.isdir(path):
        os.makedirs(path)
    fig_dir = path + run_name + '/'
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    fname = 'emf_' + str(ct).zfill(3) + '.eps'
    fig.savefig(fig_dir + fname)
    plt.show()


def plot_emfields_multi(run_name, root_dir, pic_info):
    """Plot the electromagnetic fields for multiple time steps
    """
    ct = 61
    srange = np.asarray([104, 134, -15, 15])
    plot_emfields_single(run_name, root_dir, pic_info, srange, ct)
    ct = 92
    srange = np.asarray([107, 154, -25, 25])
    plot_emfields_single(run_name, root_dir, pic_info, srange, ct)
    ct = 55
    srange = np.asarray([145, 185, -20, 20])
    plot_emfields_single(run_name, root_dir, pic_info, srange, ct)


def plot_ppara_pperp(run_name, root_dir, pic_info, srange, ct):
    """Plot the parallel and perpendicular pressure for a single time steps
    """
    kwargs = {"current_time": ct, "xl": srange[0], "xr": srange[1],
              "zb": srange[2], "zt": srange[3]}
    fname = root_dir + 'data1/ppara_real00_e.gda'
    x, z, ppara_e = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data1/pperp_real00_e.gda'
    x, z, pperp_e = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data1/ppara_real00_i.gda'
    x, z, ppara_i = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data1/pperp_real00_i.gda'
    x, z, pperp_i = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/Ay.gda'
    x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
    wpe_wce = pic_info.dtwce / pic_info.dtwpe
    va = wpe_wce / math.sqrt(pic_info.mime)  # Alfven speed of inflow region
    pnorm = 0.25 * va**2

    ppara_e /= pnorm
    pperp_e /= pnorm
    ppara_i /= pnorm
    pperp_i /= pnorm

    xmin, xmax = np.min(x), np.max(x)
    zmin, zmax = np.min(z), np.max(z)
    vmin = np.min([ppara_e, pperp_e, ppara_i, pperp_i])
    vmax = np.max([ppara_e, pperp_e, ppara_i, pperp_i])
    vmin, vmax = 0.1, 100
    nx, = x.shape
    nz, = z.shape
    fig = plt.figure(figsize=[6, 6])
    xs0, ys0 = 0.15, 0.6
    w1, h1 = 0.375, 0.375
    gap = 0.05
    ax1 = fig.add_axes([xs0, ys0, w1, h1])
    p1 = ax1.imshow(ppara_e, cmap=plt.cm.hot,
                    extent=[xmin, xmax, zmin, zmax], aspect='auto',
                    origin='lower', vmin=vmin, vmax=vmax,
                    interpolation='bicubic')
    ax1.tick_params(axis='x', labelbottom='off')
    ax1.tick_params(labelsize=16)
    ax1.contour(x, z, Ay, colors='white', linewidths=0.5)
    ax1.set_ylabel(r'$z/d_i$', fontsize=20)
    ax1.text(0.02, 0.85, r'$P_{e\parallel}$', color='w', fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax1.transAxes)
    ys = ys0 - h1 - gap
    ax2 = fig.add_axes([xs0, ys, w1, h1])
    p2 = ax2.imshow(pperp_e, cmap=plt.cm.hot,
                    extent=[xmin, xmax, zmin, zmax], aspect='auto',
                    origin='lower', vmin=vmin, vmax=vmax,
                    interpolation='bicubic')
    ax2.tick_params(labelsize=16)
    ax2.contour(x, z, Ay, colors='white', linewidths=0.5)
    ax2.set_xlabel(r'$x/d_i$', fontsize=20)
    ax2.set_ylabel(r'$z/d_i$', fontsize=20)
    ax2.text(0.02, 0.85, r'$P_{e\perp}$', color='w', fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax2.transAxes)
    xs = xs0 + w1 + gap
    ax3 = fig.add_axes([xs, ys0, w1, h1])
    p3 = ax3.imshow(ppara_i, cmap=plt.cm.hot,
                    extent=[xmin, xmax, zmin, zmax], aspect='auto',
                    origin='lower', vmin=vmin, vmax=vmax,
                    interpolation='bicubic')
    ax3.tick_params(axis='x', labelbottom='off')
    ax3.tick_params(axis='y', labelleft='off')
    ax3.tick_params(labelsize=16)
    ax3.contour(x, z, Ay, colors='white', linewidths=0.5)
    ax3.text(0.02, 0.85, r'$P_{i\parallel}$', color='w', fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax3.transAxes)
    ys = ys0 - h1 - gap
    ax4 = fig.add_axes([xs, ys, w1, h1])
    p4 = ax4.imshow(pperp_i, cmap=plt.cm.hot,
                    extent=[xmin, xmax, zmin, zmax], aspect='auto',
                    origin='lower', vmin=vmin, vmax=vmax,
                    interpolation='bicubic')
    ax4.tick_params(labelsize=16)
    ax4.tick_params(axis='y', labelleft='off')
    ax4.contour(x, z, Ay, colors='white', linewidths=0.5)
    ax4.set_xlabel(r'$x/d_i$', fontsize=20)
    ax4.text(0.02, 0.85, r'$P_{i\perp}$', color='w', fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax4.transAxes)
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    dir = '../img/img_jdotes_apj/'
    path = '../img/img_jdotes_apj/'
    if not os.path.isdir(path):
        os.makedirs(path)
    fig_dir = path + run_name + '/'
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    fname = 'pre_' + str(ct).zfill(3) + '.eps'
    fig.savefig(fig_dir + fname)
    plt.show()


def plot_ppara_pperp_multi(run_name, root_dir, pic_info):
    """Plot the parallel and perpendicular pressure for multiple time steps
    """
    ct = 61
    srange = np.asarray([105, 135, -15, 15])
    plot_ppara_pperp(run_name, root_dir, pic_info, srange, ct)
    ct = 92
    srange = np.asarray([107, 154, -25, 25])
    plot_ppara_pperp(run_name, root_dir, pic_info, srange, ct)
    ct = 55
    srange = np.asarray([145, 185, -20, 20])
    plot_ppara_pperp(run_name, root_dir, pic_info, srange, ct)


def plot_jdrifts_dote_fields():
    ct = 61
    srange = np.asarray([104, 134, -15, 15])
    xs, ys = 0.22, 0.85
    w1, h1 = 0.65, 0.14
    axis_pos = [xs, ys, w1, h1]
    gaps = [0.1, 0.02]
    fig_sizes = (4, 12)
    plot_jdotes_fields_s(run_name, root_dir, pic_info, 'e', ct, srange,
                         axis_pos, gaps, fig_sizes)

    # ct = 101
    # srange = np.asarray([110, 170, -25, 25])
    # xs, ys = 0.22, 0.85
    # w1, h1 = 0.65, 0.14
    # axis_pos = [xs, ys, w1, h1]
    # gaps = [0.1, 0.02]
    # fig_sizes = (4, 12)
    # plot_jdotes_fields(run_name, root_dir, pic_info, 'i', ct, srange,
    #         axis_pos, gaps, fig_sizes)

    # ct = 170
    # srange = np.asarray([150, 200, -25, 25])
    # plot_jdotes_fields(run_name, root_dir, pic_info, 'e', ct, srange)

    # ct = 48
    # xs, ys = 0.22, 0.85
    # w1, h1 = 0.65, 0.14
    # axis_pos = [xs, ys, w1, h1]
    # gaps = [0.1, 0.02]
    # fig_sizes = (4, 12)
    # srange = np.asarray([80, 100, -25, 25])
    # plot_jdotes_fields(run_name, root_dir, pic_info, 'i', ct, srange,
    #         axis_pos, gaps, fig_sizes)

    # ct = 92
    # xs, ys = 0.22, 0.85
    # w1, h1 = 0.65, 0.14
    # axis_pos = [xs, ys, w1, h1]
    # gaps = [0.1, 0.02]
    # fig_sizes = (4, 6)
    # srange = np.asarray([107, 154, -25, 25])
    # # plot_jdotes_fields(run_name, root_dir, pic_info, 'i', ct, srange,
    # #         axis_pos, gaps, fig_sizes)
    # plot_jdotes_fields_s(run_name, root_dir, pic_info, 'e', ct, srange,
    #         axis_pos, gaps, fig_sizes)

    # ct = 206
    # srange = np.asarray([150, 200, -25, 25])
    # plot_jdotes_fields(run_name, root_dir, pic_info, 'e', ct, srange)

    # ct = 55
    # xs, ys = 0.22, 0.85
    # w1, h1 = 0.65, 0.14
    # axis_pos = [xs, ys, w1, h1]
    # gaps = [0.1, 0.02]
    # fig_sizes = (4, 12)
    # srange = np.asarray([110, 130, -25, 25])
    # plot_jdotes_fields(run_name, root_dir, pic_info, 'i', ct, srange,
    #         axis_pos, gaps, fig_sizes)

    # ct = 110
    # srange = np.asarray([110, 180, -25, 25])
    # plot_jdotes_fields(run_name, root_dir, pic_info, 'e', ct, srange)
    # ct = 200
    # srange = np.asarray([130, 200, -25, 25])
    # plot_jdotes_fields(run_name, root_dir, pic_info, 'i', ct, srange)
    # ct = 40
    # srange = np.asarray([80, 100, -25, 25])
    # plot_jdotes_fields(run_name, root_dir, pic_info, 'i', ct, srange)
    # ct = 80
    # srange = np.asarray([100, 150, -25, 25])
    # plot_jdotes_fields(run_name, root_dir, pic_info, 'i', ct, srange)
    # ct = 200
    # srange = np.asarray([150, 200, -25, 25])
    # plot_jdotes_fields(run_name, root_dir, pic_info, 'i', ct, srange)

    # ct = 109
    # xs, ys = 0.18, 0.85
    # w1, h1 = 0.68, 0.14
    # axis_pos = [xs, ys, w1, h1]
    # gaps = [0.1, 0.02]
    # fig_sizes = (5, 12)
    # srange = np.asarray([120, 200, -25, 25])
    # plot_jdotes_fields(run_name, root_dir, pic_info, 'e', ct, srange,
    #         axis_pos, gaps, fig_sizes)

    # ct = 55
    # xs, ys = 0.18, 0.85
    # w1, h1 = 0.68, 0.14
    # axis_pos = [xs, ys, w1, h1]
    # gaps = [0.1, 0.02]
    # fig_sizes = (5, 12)
    # srange = np.asarray([145, 185, -20, 20])
    # plot_jdotes_fields_s(run_name, root_dir, pic_info, 'e', ct, srange,
    #         axis_pos, gaps, fig_sizes)


def plot_fields_wcuts(run_name, root_dir, pic_info, ct, srange, xcp,
                      label_name, dnorm, fdata):
    """Plot 2D fields with cuts

    Args:
        run_name: the name of this run.
        root_dir: the root directory of this run.
        pic_info: PIC simulation information in a namedtuple.
        ct: current time frame
        srange: spatial range
        xcp: the cut x positions
        label_name: the label name of the dataset
        dnorm: the normalization for the data
        fdata: field data
    """
    kwargs = {"current_time": ct, "xl": srange[0], "xr": srange[1],
              "zb": srange[2], "zt": srange[3]}
    ng = 3
    kernel = np.ones((ng, ng)) / float(ng*ng)
    dv = pic_info.dx_di * pic_info.dz_di * pic_info.mime
    fdata = signal.convolve2d(fdata, kernel, 'same')

    fdata /= dnorm
    fname2 = root_dir + 'data/Ay.gda'
    x, z, Ay = read_2d_fields(pic_info, fname2, **kwargs)

    xmin, xmax = np.min(x), np.max(x)
    zmin, zmax = np.min(z), np.max(z)
    vmin, vmax = -1.0, 1.0
    dx = x[1] - x[0]
    dz = z[1] - z[0]
    nx, = x.shape
    nz, = z.shape
    fig = plt.figure(figsize=[10, 5])
    xs0, ys0 = 0.12, 0.14
    w1, h1 = 0.4, 0.8
    gap = 0.03
    ax1 = fig.add_axes([xs0, ys0, w1, h1])
    p1 = ax1.imshow(fdata, cmap=plt.cm.seismic,
                    extent=[xmin, xmax, zmin, zmax], aspect='auto',
                    origin='lower', vmin=vmin, vmax=vmax,
                    interpolation='bicubic')
    ax1.tick_params(labelsize=16)
    ax1.contour(x, z, Ay, colors='black', linewidths=0.5)
    ax1.set_xlabel(r'$x/d_i$', fontsize=20)
    ax1.set_ylabel(r'$z/d_i$', fontsize=20)
    ax1.text(0.02, 0.85, label_name, color='k', fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax1.transAxes)

    # Plot a cut along the vertical direction
    xcp_index = (xcp - srange[0]) / dx
    ax1.set_color_cycle(colors)
    for ix in xcp_index:
        ax1.plot([x[ix], x[ix]], [zmin, zmax], linewidth=0.5, linestyle='--')
    xs = xs0 + w1 + gap
    w2 = w1
    ax2 = fig.add_axes([xs, ys0, w2, h1])
    ax2.set_color_cycle(colors)
    for ix in xcp_index:
        cdata = fdata[:, ix]
        ax2.plot(cdata, z, linewidth=2)
    ax2.tick_params(axis='y', labelleft='off')
    ax2.tick_params(labelsize=16)
    ax2.set_ylim([zmin, zmax])
    plt.show()


def plot_jdotes_xyz(run_name, root_dir, pic_info, ct, srange, xcp,
                    jdote_x, jdote_y, jdote_z, dnorm):
    """Plot energy conversion due to 3 components of the currents

    Args:
        run_name: the name of this run.
        root_dir: the root directory of this run.
        pic_info: PIC simulation information in a namedtuple.
        ct: current time frame
        srange: spatial range
        xcp: the cut x positions
        jdote_x, jdote_y, jdote_z: energy conversion
        dnorm: the normalization for the data
    """
    kwargs = {"current_time": ct, "xl": srange[0], "xr": srange[1],
              "zb": srange[2], "zt": srange[3]}
    ng = 3
    kernel = np.ones((ng, ng)) / float(ng*ng)
    dv = pic_info.dx_di * pic_info.dz_di * pic_info.mime
    jdote_x_cum = np.cumsum(np.sum(jdote_x, axis=0)) * dv
    jdote_y_cum = np.cumsum(np.sum(jdote_y, axis=0)) * dv
    jdote_z_cum = np.cumsum(np.sum(jdote_z, axis=0)) * dv
    jdote_x = signal.convolve2d(jdote_x, kernel, 'same')
    jdote_y = signal.convolve2d(jdote_y, kernel, 'same')
    jdote_z = signal.convolve2d(jdote_z, kernel, 'same')

    jdote_x /= dnorm
    jdote_y /= dnorm
    jdote_z /= dnorm
    fname2 = root_dir + 'data/Ay.gda'
    x, z, Ay = read_2d_fields(pic_info, fname2, **kwargs)

    xmin, xmax = np.min(x), np.max(x)
    zmin, zmax = np.min(z), np.max(z)
    vmin, vmax = -1.0, 1.0
    dx = x[1] - x[0]
    dz = z[1] - z[0]
    nx, = x.shape
    nz, = z.shape
    fig = plt.figure(figsize=[5, 10])
    xs0, ys0 = 0.17, 0.78
    w1, h1 = 0.4, 0.2
    gap = 0.03
    ax1 = fig.add_axes([xs0, ys0, w1, h1])
    p1 = ax1.imshow(jdote_x, cmap=plt.cm.seismic,
                    extent=[xmin, xmax, zmin, zmax], aspect='auto',
                    origin='lower', vmin=vmin, vmax=vmax,
                    interpolation='bicubic')
    ax1.tick_params(labelsize=16)
    ax1.contour(x, z, Ay, colors='black', linewidths=0.5)
    ax1.tick_params(axis='x', labelbottom='off')
    ax1.set_ylabel(r'$z/d_i$', fontsize=20)
    ax1.text(0.02, 0.85, r'$j_xE_x$', color=colors[0], fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax1.transAxes)
    # Plot a cut along the vertical direction
    xcp_index = (xcp - srange[0]) / dx
    ax1.set_color_cycle(colors)
    for ix in xcp_index:
        ax1.plot([x[ix], x[ix]], [zmin, zmax], linewidth=0.5, linestyle='--')
    xs = xs0 + w1
    w2 = w1
    ax12 = fig.add_axes([xs, ys0, w2, h1])
    ax12.set_color_cycle(colors)
    for ix in xcp_index:
        cdata = jdote_x[:, ix]
        ax12.plot(cdata, z, linewidth=2)
    ax12.set_xlim([-1, 1])
    ax12.set_ylim([zmin, zmax])
    ax12.tick_params(axis='x', labelbottom='off')
    ax12.tick_params(axis='y', labelleft='off')
    ax12.tick_params(labelsize=16)

    ys = ys0 - h1 - gap
    ax2 = fig.add_axes([xs0, ys, w1, h1])
    p2 = ax2.imshow(jdote_y, cmap=plt.cm.seismic,
                    extent=[xmin, xmax, zmin, zmax], aspect='auto',
                    origin='lower', vmin=vmin, vmax=vmax,
                    interpolation='bicubic')
    ax2.tick_params(labelsize=16)
    ax2.contour(x, z, Ay, colors='black', linewidths=0.5)
    ax2.tick_params(axis='x', labelbottom='off')
    ax2.set_ylabel(r'$z/d_i$', fontsize=20)
    ax2.text(0.02, 0.85, r'$j_yE_y$', color=colors[1], fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax2.transAxes)
    ax2.set_color_cycle(colors)
    for ix in xcp_index:
        ax2.plot([x[ix], x[ix]], [zmin, zmax], linewidth=0.5, linestyle='--')
    ax22 = fig.add_axes([xs, ys, w1, h1])
    ax22.set_color_cycle(colors)
    for ix in xcp_index:
        cdata = jdote_y[:, ix]
        ax22.plot(cdata, z, linewidth=2)
    ax22.set_xlim([-1, 1])
    ax22.set_ylim([zmin, zmax])
    ax22.tick_params(axis='x', labelbottom='off')
    ax22.tick_params(axis='y', labelleft='off')
    ax22.tick_params(labelsize=16)

    ys -= h1 + gap
    ax3 = fig.add_axes([xs0, ys, w1, h1])
    p3 = ax3.imshow(jdote_z, cmap=plt.cm.seismic,
                    extent=[xmin, xmax, zmin, zmax], aspect='auto',
                    origin='lower', vmin=vmin, vmax=vmax,
                    interpolation='bicubic')
    ax3.tick_params(labelsize=16)
    ax3.contour(x, z, Ay, colors='black', linewidths=0.5)
    ax3.tick_params(axis='x', labelbottom='off')
    ax3.set_ylabel(r'$z/d_i$', fontsize=20)
    ax3.text(0.02, 0.85, r'$j_zE_z$', color=colors[2], fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax3.transAxes)
    ax3.set_color_cycle(colors)
    for ix in xcp_index:
        ax3.plot([x[ix], x[ix]], [zmin, zmax], linewidth=0.5, linestyle='--')
    ax32 = fig.add_axes([xs, ys, w1, h1])
    ax32.set_color_cycle(colors)
    for ix in xcp_index:
        cdata = jdote_z[:, ix]
        ax32.plot(cdata, z, linewidth=2)
    ax32.set_xlim([-1, 1])
    ax32.set_ylim([zmin, zmax])
    ax32.tick_params(axis='y', labelleft='off')
    ax32.tick_params(labelsize=16)

    ys -= h1 + gap
    ax4 = fig.add_axes([xs0, ys, w1, h1])
    ax4.set_color_cycle(colors)
    ax4.plot(x, jdote_x_cum, linewidth=2)
    ax4.plot(x, jdote_y_cum, linewidth=2)
    ax4.plot(x, jdote_z_cum, linewidth=2)
    ax4.plot([xmin, xmax], [0, 0], linewidth=0.5, color='k', linestyle='--')
    ax4.set_xlim([xmin, xmax])
    ax4.tick_params(labelsize=16)
    ax4.set_xlabel(r'$x/d_i$', fontsize=20)

    # plt.show()


def plot_jdotes_single(run_name, root_dir, pic_info, srange, ct, xcp, species):
    """Plot jdote due to a single current
    """
    kwargs = {"current_time": ct, "xl": srange[0], "xr": srange[1],
              "zb": srange[2], "zt": srange[3]}
    wpe_wce = pic_info.dtwce / pic_info.dtwpe
    va = wpe_wce / math.sqrt(pic_info.mime)  # Alfven speed of inflow region
    b0 = pic_info.b0
    jdote_norm = 0.1 * va**2 * b0
    fname = root_dir + 'data/ex.gda'
    x, z, ex = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/ey.gda'
    x, z, ey = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/ez.gda'
    x, z, ez = read_2d_fields(pic_info, fname, **kwargs)
    data_dir = 'data1'
    jtypes = ['jqnvpara', 'jcpara', 'jgrad', 'jmag', 'jpolar', 'jagy']
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    dir = '../img/img_jdotes_apj/'
    path = '../img/img_jdotes_apj/'
    if not os.path.isdir(path):
        os.makedirs(path)
    fig_dir = path + run_name + '/'
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    for jtype in jtypes:
        dname = jtype + 'x00_' + species
        fname = root_dir + data_dir + '/' + dname + '.gda'
        x, z, jqnvpara_x = read_2d_fields(pic_info, fname, **kwargs)
        dname = jtype + 'y00_' + species
        fname = root_dir + data_dir + '/' + dname + '.gda'
        x, z, jqnvpara_y = read_2d_fields(pic_info, fname, **kwargs)
        dname = jtype + 'z00_' + species
        fname = root_dir + data_dir + '/' + dname + '.gda'
        x, z, jqnvpara_z = read_2d_fields(pic_info, fname, **kwargs)
        jdote_x = jqnvpara_x * ex
        jdote_y = jqnvpara_y * ey
        jdote_z = jqnvpara_z * ez
        plot_jdotes_xyz(run_name, root_dir, pic_info, ct, srange, xcp,
                        jdote_x, jdote_y, jdote_z, jdote_norm)
        fname = jtype + '_' + str(ct).zfill(3) + '_' + species + '.eps'
        plt.savefig(fig_dir + fname)
        plt.close()
        # plt.show()


def plot_jdotes_multi(run_name, root_dir, pic_info):
    """Plot jdote due to currents at multiple time step
    """
    ct = 55
    srange = np.asarray([145, 185, -20, 20])
    xcp = [155, 165, 175]
    plot_jdotes_single(run_name, root_dir, pic_info, srange, ct, xcp, 'e')
    plot_jdotes_single(run_name, root_dir, pic_info, srange, ct, xcp, 'i')
    ct = 61
    srange = np.asarray([105, 135, -15, 15])
    xcp = [115, 125]
    plot_jdotes_single(run_name, root_dir, pic_info, srange, ct, xcp, 'e')
    plot_jdotes_single(run_name, root_dir, pic_info, srange, ct, xcp, 'i')
    ct = 92
    srange = np.asarray([107, 154, -25, 25])
    xcp = [115, 125, 135, 145]
    plot_jdotes_single(run_name, root_dir, pic_info, srange, ct, xcp, 'e')
    plot_jdotes_single(run_name, root_dir, pic_info, srange, ct, xcp, 'i')


def plot_uxyz_single(run_name, root_dir, pic_info, srange, ct, xcp, species):
    """Plot ux, uy, uz for a single time step
    """
    kwargs = {"current_time": ct, "xl": srange[0], "xr": srange[1],
              "zb": srange[2], "zt": srange[3]}
    wpe_wce = pic_info.dtwce / pic_info.dtwpe
    va = wpe_wce / math.sqrt(pic_info.mime)  # Alfven speed of inflow region
    b0 = pic_info.b0
    fname = root_dir + 'data/u' + species + 'x.gda'
    x, z, ux = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/u' + species + 'y.gda'
    x, z, uy = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/u' + species + 'z.gda'
    x, z, uz = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/bx.gda'
    x, z, bx = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/by.gda'
    x, z, by = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/bz.gda'
    x, z, bz = read_2d_fields(pic_info, fname, **kwargs)
    absB2 = bx**2 + by**2 + bz**2
    udotb = (ux*bx + uy*by + uz*bz) / absB2
    fname2 = root_dir + 'data/Ay.gda'
    x, z, Ay = read_2d_fields(pic_info, fname2, **kwargs)
    ux = udotb * bx / va
    uy = udotb * by / va
    uz = udotb * bz / va
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    dir = '../img/img_jdotes_apj/'
    path = '../img/img_jdotes_apj/'
    if not os.path.isdir(path):
        os.makedirs(path)
    fig_dir = path + run_name + '/'
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    xmin, xmax = np.min(x), np.max(x)
    zmin, zmax = np.min(z), np.max(z)
    vmin, vmax = -1.0, 1.0
    dx = x[1] - x[0]
    dz = z[1] - z[0]
    nx, = x.shape
    nz, = z.shape
    fig = plt.figure(figsize=[3, 8])
    xs0, ys0 = 0.17, 0.7
    w1, h1 = 0.75, 0.28125
    gap = 0.03
    ax1 = fig.add_axes([xs0, ys0, w1, h1])
    p1 = ax1.imshow(ux, cmap=plt.cm.seismic,
                    extent=[xmin, xmax, zmin, zmax], aspect='auto',
                    origin='lower', vmin=vmin, vmax=vmax,
                    interpolation='bicubic')
    ax1.tick_params(labelsize=16)
    ax1.contour(x, z, Ay, colors='black', linewidths=0.5)
    ax1.tick_params(axis='x', labelbottom='off')
    ax1.set_ylabel(r'$z/d_i$', fontsize=20)
    ax1.text(0.02, 0.85, r'$v_x$', color=colors[0], fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax1.transAxes)

    ys = ys0 - h1 - gap
    ax2 = fig.add_axes([xs0, ys, w1, h1])
    p2 = ax2.imshow(uy, cmap=plt.cm.seismic,
                    extent=[xmin, xmax, zmin, zmax], aspect='auto',
                    origin='lower', vmin=vmin, vmax=vmax,
                    interpolation='bicubic')
    ax2.tick_params(labelsize=16)
    ax2.contour(x, z, Ay, colors='black', linewidths=0.5)
    ax2.tick_params(axis='x', labelbottom='off')
    ax2.set_ylabel(r'$z/d_i$', fontsize=20)
    ax2.text(0.02, 0.85, r'$v_y$', color=colors[1], fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax2.transAxes)

    ys -= h1 + gap
    ax3 = fig.add_axes([xs0, ys, w1, h1])
    p3 = ax3.imshow(uz, cmap=plt.cm.seismic,
                    extent=[xmin, xmax, zmin, zmax], aspect='auto',
                    origin='lower', vmin=vmin, vmax=vmax,
                    interpolation='bicubic')
    ax3.tick_params(labelsize=16)
    ax3.contour(x, z, Ay, colors='black', linewidths=0.5)
    ax3.set_xlabel(r'$x/d_i$', fontsize=20)
    ax3.set_ylabel(r'$z/d_i$', fontsize=20)
    ax3.text(0.02, 0.85, r'$v_z$', color=colors[2], fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax3.transAxes)
    # plt.close()
    plt.show()


def plot_bulku_single(run_name, root_dir, pic_info, srange, ct, xcp):
    """Plot ux, uy, uz for single fluid a single time step
    """
    kwargs = {"current_time": ct, "xl": srange[0], "xr": srange[1],
              "zb": srange[2], "zt": srange[3]}
    wpe_wce = pic_info.dtwce / pic_info.dtwpe
    va = wpe_wce / math.sqrt(pic_info.mime)  # Alfven speed of inflow region
    b0 = pic_info.b0
    fname = root_dir + 'data/uex.gda'
    x, z, uex = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/uey.gda'
    x, z, uey = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/uez.gda'
    x, z, uez = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/ne.gda'
    x, z, ne = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/uix.gda'
    x, z, uix = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/uiy.gda'
    x, z, uiy = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/uiz.gda'
    x, z, uiz = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/ni.gda'
    x, z, ni = read_2d_fields(pic_info, fname, **kwargs)
    fname2 = root_dir + 'data/Ay.gda'
    x, z, Ay = read_2d_fields(pic_info, fname2, **kwargs)
    mime = pic_info.mime
    ntot = ne + ni * mime
    ux = (uex*ne + uix*ni*mime) / ntot
    uy = (uey*ne + uiy*ni*mime) / ntot
    uz = (uez*ne + uiz*ni*mime) / ntot
    u0 = 0.5*va
    ux /= u0
    uy /= u0
    uz /= u0
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    dir = '../img/img_jdotes_apj/'
    path = '../img/img_jdotes_apj/'
    if not os.path.isdir(path):
        os.makedirs(path)
    fig_dir = path + run_name + '/'
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    xmin, xmax = np.min(x), np.max(x)
    zmin, zmax = np.min(z), np.max(z)
    vmin, vmax = -1.0, 1.0
    dx = x[1] - x[0]
    dz = z[1] - z[0]
    nx, = x.shape
    nz, = z.shape
    fig = plt.figure(figsize=[3, 8])
    xs0, ys0 = 0.03 * 8 / 3, 0.7
    w1, h1 = 0.75, 0.28125
    gap = 0.03
    ax1 = fig.add_axes([xs0, ys0, w1, h1])
    p1 = ax1.imshow(ux, cmap=plt.cm.seismic,
                    extent=[xmin, xmax, zmin, zmax], aspect='auto',
                    origin='lower', vmin=vmin, vmax=vmax,
                    interpolation='bicubic')
    ax1.tick_params(labelsize=16)
    ax1.contour(x, z, Ay, colors='black', linewidths=0.5)
    ax1.tick_params(axis='y', labelleft='off')
    ax1.tick_params(axis='x', labelbottom='off')
    # ax1.set_ylabel(r'$z/d_i$', fontsize=20)
    ax1.text(0.02, 0.85, r'$v_x$', color='k', fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax1.transAxes)

    ys = ys0 - h1 - gap
    ax2 = fig.add_axes([xs0, ys, w1, h1])
    p2 = ax2.imshow(uy, cmap=plt.cm.seismic,
                    extent=[xmin, xmax, zmin, zmax], aspect='auto',
                    origin='lower', vmin=vmin, vmax=vmax,
                    interpolation='bicubic')
    ax2.tick_params(labelsize=16)
    ax2.contour(x, z, Ay, colors='black', linewidths=0.5)
    ax2.tick_params(axis='y', labelleft='off')
    ax2.tick_params(axis='x', labelbottom='off')
    # ax2.set_ylabel(r'$z/d_i$', fontsize=20)
    ax2.text(0.02, 0.85, r'$v_y$', color='k', fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax2.transAxes)

    ys -= h1 + gap
    ax3 = fig.add_axes([xs0, ys, w1, h1])
    p3 = ax3.imshow(uz, cmap=plt.cm.seismic,
                    extent=[xmin, xmax, zmin, zmax], aspect='auto',
                    origin='lower', vmin=vmin, vmax=vmax,
                    interpolation='bicubic')
    ax3.tick_params(labelsize=16)
    ax3.contour(x, z, Ay, colors='black', linewidths=0.5)
    ax3.set_xlabel(r'$x/d_i$', fontsize=20)
    # ax3.set_ylabel(r'$z/d_i$', fontsize=20)
    ax3.tick_params(axis='y', labelleft='off')
    ax3.text(0.02, 0.85, r'$v_z$', color='k', fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax3.transAxes)
    fname = 'bulku_' + str(ct).zfill(3) + '.eps'
    fig.savefig(fig_dir + fname)
    # plt.close()
    plt.show()


def plot_uxyz_multi(run_name, root_dir, pic_info):
    """Plot ux, uy, uz for multiple time steps
    """
    # ct = 55
    # srange = np.asarray([145, 185, -20, 20])
    # xcp = [155, 165, 175]
    # plot_uxyz_single(run_name, root_dir, pic_info, srange, ct, xcp, 'i')
    # plot_uxyz_single(run_name, root_dir, pic_info, srange, ct, xcp, 'i')
    ct = 61
    srange = np.asarray([104, 134, -15, 15])
    xcp = [115, 125]
    plot_bulku_single(run_name, root_dir, pic_info, srange, ct, xcp)
    # plot_uxyz_single(run_name, root_dir, pic_info, srange, ct, xcp, 'i')
    # plot_uxyz_single(run_name, root_dir, pic_info, srange, ct, xcp, 'i')
    # ct = 92
    # srange = np.asarray([107, 154, -25, 25])
    # xcp = [115, 125, 135, 145]
    # plot_uxyz_single(run_name, root_dir, pic_info, srange, ct, xcp, 'e')
    # plot_uxyz_single(run_name, root_dir, pic_info, srange, ct, xcp, 'i')


def plot_epara_xyz_single(run_name, root_dir, pic_info, srange, ct, xcp):
    """Plot parallel electric field for a single time step
    """
    kwargs = {"current_time": ct, "xl": srange[0], "xr": srange[1],
              "zb": srange[2], "zt": srange[3]}
    wpe_wce = pic_info.dtwce / pic_info.dtwpe
    va = wpe_wce / math.sqrt(pic_info.mime)  # Alfven speed of inflow region
    b0 = pic_info.b0
    e0 = 0.1 * b0 * va
    fname = root_dir + 'data/bx.gda'
    x, z, bx = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/by.gda'
    x, z, by = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/bz.gda'
    x, z, bz = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/ex.gda'
    x, z, ex = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/ey.gda'
    x, z, ey = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/ez.gda'
    x, z, ez = read_2d_fields(pic_info, fname, **kwargs)
    absB2 = bx**2 + by**2 + bz**2
    fname2 = root_dir + 'data/Ay.gda'
    x, z, Ay = read_2d_fields(pic_info, fname2, **kwargs)
    edotb = (ex*bx + ey*by + ez*bz) / absB2
    eparax = edotb * bx / e0
    eparay = edotb * by / e0
    eparaz = edotb * bz / e0
    ng = 3
    kernel = np.ones((ng, ng)) / float(ng*ng)
    eparax = signal.convolve2d(eparax, kernel, 'same')
    eparay = signal.convolve2d(eparay, kernel, 'same')
    eparaz = signal.convolve2d(eparaz, kernel, 'same')
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    dir = '../img/img_jdotes_apj/'
    path = '../img/img_jdotes_apj/'
    if not os.path.isdir(path):
        os.makedirs(path)
    fig_dir = path + run_name + '/'
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    xmin, xmax = np.min(x), np.max(x)
    zmin, zmax = np.min(z), np.max(z)
    vmin, vmax = -1.0, 1.0
    dx = x[1] - x[0]
    dz = z[1] - z[0]
    nx, = x.shape
    nz, = z.shape
    fig = plt.figure(figsize=[3, 8])
    xs0, ys0 = 0.17, 0.7
    w1, h1 = 0.75, 0.28125
    gap = 0.03
    ax1 = fig.add_axes([xs0, ys0, w1, h1])
    p1 = ax1.imshow(eparax, cmap=plt.cm.seismic,
                    extent=[xmin, xmax, zmin, zmax], aspect='auto',
                    origin='lower', vmin=vmin, vmax=vmax,
                    interpolation='bicubic')
    ax1.tick_params(labelsize=16)
    ax1.contour(x, z, Ay, colors='black', linewidths=0.5)
    ax1.tick_params(axis='x', labelbottom='off')
    ax1.set_ylabel(r'$z/d_i$', fontsize=20)
    ax1.text(0.02, 0.85, r'$E_{\parallel x}$', color=colors[0], fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax1.transAxes)

    ys = ys0 - h1 - gap
    ax2 = fig.add_axes([xs0, ys, w1, h1])
    p2 = ax2.imshow(eparay, cmap=plt.cm.seismic,
                    extent=[xmin, xmax, zmin, zmax], aspect='auto',
                    origin='lower', vmin=vmin, vmax=vmax,
                    interpolation='bicubic')
    ax2.tick_params(labelsize=16)
    ax2.contour(x, z, Ay, colors='black', linewidths=0.5)
    ax2.tick_params(axis='x', labelbottom='off')
    ax2.set_ylabel(r'$z/d_i$', fontsize=20)
    ax2.text(0.02, 0.85, r'$E_{\parallel y}$', color=colors[1], fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax2.transAxes)

    ys -= h1 + gap
    ax3 = fig.add_axes([xs0, ys, w1, h1])
    p3 = ax3.imshow(eparaz, cmap=plt.cm.seismic,
                    extent=[xmin, xmax, zmin, zmax], aspect='auto',
                    origin='lower', vmin=vmin, vmax=vmax,
                    interpolation='bicubic')
    ax3.tick_params(labelsize=16)
    ax3.contour(x, z, Ay, colors='black', linewidths=0.5)
    ax3.set_xlabel(r'$x/d_i$', fontsize=20)
    ax3.set_ylabel(r'$z/d_i$', fontsize=20)
    ax3.text(0.02, 0.85, r'$E_{\parallel z}$', color=colors[2], fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax3.transAxes)
    # plt.close()
    plt.show()


def plot_epara_xyz_multi(run_name, root_dir, pic_info):
    """Plot parallel electric field for multiple time steps
    """
    ct = 55
    srange = np.asarray([145, 185, -20, 20])
    xcp = [155, 165, 175]
    plot_epara_xyz_single(run_name, root_dir, pic_info, srange, ct, xcp)
    # plot_epara_xyz_multi(run_name, root_dir, pic_info, srange, ct, xcp)
    # ct = 61
    # srange = np.asarray([105, 135, -15, 15])
    # xcp = [115, 125]
    # plot_epara_xyz_multi(run_name, root_dir, pic_info, srange, ct, xcp)
    # plot_epara_xyz_multi(run_name, root_dir, pic_info, srange, ct, xcp)
    # ct = 92
    # srange = np.asarray([107, 154, -25, 25])
    # xcp = [115, 125, 135, 145]
    # plot_epara_xyz_multi(run_name, root_dir, pic_info, srange, ct, xcp)
    # plot_epara_xyz_multi(run_name, root_dir, pic_info, srange, ct, xcp)


def plot_eperp_xyz_single(run_name, root_dir, pic_info, srange, ct, xcp):
    """Plot perpendicular electric field for a single time step
    """
    kwargs = {"current_time": ct, "xl": srange[0], "xr": srange[1],
              "zb": srange[2], "zt": srange[3]}
    wpe_wce = pic_info.dtwce / pic_info.dtwpe
    va = wpe_wce / math.sqrt(pic_info.mime)  # Alfven speed of inflow region
    b0 = pic_info.b0
    e0 = 0.5 * b0 * va
    fname = root_dir + 'data/bx.gda'
    x, z, bx = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/by.gda'
    x, z, by = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/bz.gda'
    x, z, bz = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/ex.gda'
    x, z, ex = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/ey.gda'
    x, z, ey = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/ez.gda'
    x, z, ez = read_2d_fields(pic_info, fname, **kwargs)
    absB2 = bx**2 + by**2 + bz**2
    fname2 = root_dir + 'data/Ay.gda'
    x, z, Ay = read_2d_fields(pic_info, fname2, **kwargs)
    edotb = (ex*bx + ey*by + ez*bz) / absB2
    eperpx = (ex - edotb * bx) / e0
    eperpy = (ey - edotb * by) / e0
    eperpz = (ez - edotb * bz) / e0
    ng = 3
    kernel = np.ones((ng, ng)) / float(ng*ng)
    eperpx = signal.convolve2d(eperpx, kernel, 'same')
    eperpy = signal.convolve2d(eperpy, kernel, 'same')
    eperpz = signal.convolve2d(eperpz, kernel, 'same')
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    dir = '../img/img_jdotes_apj/'
    path = '../img/img_jdotes_apj/'
    if not os.path.isdir(path):
        os.makedirs(path)
    fig_dir = path + run_name + '/'
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    xmin, xmax = np.min(x), np.max(x)
    zmin, zmax = np.min(z), np.max(z)
    vmin, vmax = -1.0, 1.0
    dx = x[1] - x[0]
    dz = z[1] - z[0]
    nx, = x.shape
    nz, = z.shape
    fig = plt.figure(figsize=[3, 8])
    xs0, ys0 = 0.17, 0.7
    w1, h1 = 0.75, 0.28125
    gap = 0.03
    ax1 = fig.add_axes([xs0, ys0, w1, h1])
    p1 = ax1.imshow(eperpx, cmap=plt.cm.seismic,
                    extent=[xmin, xmax, zmin, zmax], aspect='auto',
                    origin='lower', vmin=vmin, vmax=vmax,
                    interpolation='bicubic')
    ax1.tick_params(labelsize=16)
    ax1.contour(x, z, Ay, colors='black', linewidths=0.5)
    ax1.tick_params(axis='x', labelbottom='off')
    ax1.set_ylabel(r'$z/d_i$', fontsize=20)
    ax1.text(0.02, 0.85, r'$E_{\perp x}$', color=colors[0], fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax1.transAxes)

    ys = ys0 - h1 - gap
    ax2 = fig.add_axes([xs0, ys, w1, h1])
    p2 = ax2.imshow(eperpy, cmap=plt.cm.seismic,
                    extent=[xmin, xmax, zmin, zmax], aspect='auto',
                    origin='lower', vmin=vmin, vmax=vmax,
                    interpolation='bicubic')
    ax2.tick_params(labelsize=16)
    ax2.contour(x, z, Ay, colors='black', linewidths=0.5)
    ax2.tick_params(axis='x', labelbottom='off')
    ax2.set_ylabel(r'$z/d_i$', fontsize=20)
    ax2.text(0.02, 0.85, r'$E_{\perp y}$', color=colors[1], fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax2.transAxes)

    ys -= h1 + gap
    ax3 = fig.add_axes([xs0, ys, w1, h1])
    p3 = ax3.imshow(eperpz, cmap=plt.cm.seismic,
                    extent=[xmin, xmax, zmin, zmax], aspect='auto',
                    origin='lower', vmin=vmin, vmax=vmax,
                    interpolation='bicubic')
    ax3.tick_params(labelsize=16)
    ax3.contour(x, z, Ay, colors='black', linewidths=0.5)
    ax3.set_xlabel(r'$x/d_i$', fontsize=20)
    ax3.set_ylabel(r'$z/d_i$', fontsize=20)
    ax3.text(0.02, 0.85, r'$E_{\perp z}$', color=colors[2], fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                       pad=10.0), horizontalalignment='left',
             verticalalignment='center', transform=ax3.transAxes)
    # plt.close()
    plt.show()


def plot_eperp_xyz_multi(run_name, root_dir, pic_info):
    """Plot perpendicular electric field for multiple time steps
    """
    ct = 55
    srange = np.asarray([145, 185, -20, 20])
    xcp = [155, 165, 175]
    plot_eperp_xyz_single(run_name, root_dir, pic_info, srange, ct, xcp)
    # plot_eperp_xyz_multi(run_name, root_dir, pic_info, srange, ct, xcp)
    # ct = 61
    # srange = np.asarray([105, 135, -15, 15])
    # xcp = [115, 125]
    # plot_eperp_xyz_multi(run_name, root_dir, pic_info, srange, ct, xcp)
    # plot_eperp_xyz_multi(run_name, root_dir, pic_info, srange, ct, xcp)
    # ct = 92
    # srange = np.asarray([107, 154, -25, 25])
    # xcp = [115, 125, 135, 145]
    # plot_eperp_xyz_multi(run_name, root_dir, pic_info, srange, ct, xcp)
    # plot_eperp_xyz_multi(run_name, root_dir, pic_info, srange, ct, xcp)


if __name__ == "__main__":
    # scratch_dir = '/net/scratch2/xiaocanli/'
    # run_name = "mime25_beta002_noperturb"
    # root_dir = scratch_dir + 'mime25-sigma1-beta002-200-100-noperturb/'
    # run_name = "mime25_beta002"
    # root_dir = "/net/scratch2/xiaocanli/sigma1-mime25-beta001/"
    # run_name = "mime25_beta0007"
    # root_dir = '/net/scratch2/xiaocanli/mime25-guide0-beta0007-200-100/'
    run_name = "mime25_beta002_track"
    root_dir = '/net/scratch2/guofan/sigma1-mime25-beta001-track-3/'
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    # plot_by_time(run_name, root_dir, pic_info)
    # plot_vx_time(run_name, root_dir, pic_info)
    plot_epara_eperp(pic_info, 26, root_dir)
    # plot_jpara_dote(run_name, root_dir, pic_info, 'i')
    # plot_jdrifts_dote_fields()
    # plot_jdotes_multi(run_name, root_dir, pic_info)
    # plot_uxyz_multi(run_name, root_dir, pic_info)
    # plot_epara_xyz_multi(run_name, root_dir, pic_info)
    # plot_eperp_xyz_multi(run_name, root_dir, pic_info)
    # plot_emfields_multi(run_name, root_dir, pic_info)
    # plot_ppara_pperp_multi(run_name, root_dir, pic_info)
    # plot_curvb_multi(run_name, root_dir, pic_info)
