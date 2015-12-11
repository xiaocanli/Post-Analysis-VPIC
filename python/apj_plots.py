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

font = {'family' : 'serif',
        #'color'  : 'darkred',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 24,
        }

colors = palettable.colorbrewer.qualitative.Set1_9.mpl_colors

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
    cts = [50, 200, 800]
    cts = np.asarray(cts)
    var_names = []
    for i in range(nt):
        var_name = r'$t=' + str(cts[i]) + r'/\Omega_{ci}$'
        var_names.append(var_name)
    cts /= pic_info.dt_fields
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
    kwargs = {"current_time":ct, "xl":0, "xr":200, "zb":-50, "zt":50}
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
    kwargs_plots = {'current_time':ct, 'x':x, 'z':z, 'Ay':Ay_data,
            'fdata':fdata, 'contour_color':contour_color, 'colormaps':colormaps,
            'vmin':vmin, 'vmax':vmax, 'var_names':var_names, 'axis_pos':axis_pos,
            'gaps':gaps, 'fig_sizes':fig_sizes, 'text_colors':text_colors,
            'nxp':nxp, 'nzp':nzp, 'xstep':xstep, 'zstep':zstep, 'is_logs':is_logs,
            'fname':fname, 'fig_dir':fig_dir, 'is_multi_Ay':True, 'save_eps':True}
    by_plot = PlotMultiplePanels(**kwargs_plots)
    for cbar in by_plot.cbar:
        cbar.set_ticks(np.arange(-0.8, 0.9, 0.4))
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
    cts = [50, 200]
    cts = np.asarray(cts)
    var_names = []
    for i in range(nt):
        var_name = r'$t=' + str(cts[i]) + r'/\Omega_{ci}$'
        var_names.append(var_name)
    cts /= pic_info.dt_fields
    colormaps = ['seismic'] * nt
    text_colors = ['r', 'b']
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
    kwargs = {"current_time":ct, "xl":0, "xr":200, "zb":-50, "zt":50}
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
        fdata_1d.append(ux[nz/2,:])
        Ay_data.append(Ay)
    fname = 'vx_time'
    fdata = np.asarray(fdata)
    fdata_1d = np.asarray(fdata_1d)
    fname = 'vx_time'
    bottom_panel = True
    xlim = [0, 200]
    zlim = [-50, 50]
    save_eps = True
    kwargs_plots = {'current_time':ct, 'x':x, 'z':z, 'Ay':Ay_data,
            'fdata':fdata, 'contour_color':contour_color, 'colormaps':colormaps,
            'vmin':vmin, 'vmax':vmax, 'var_names':var_names, 'axis_pos':axis_pos,
            'gaps':gaps, 'fig_sizes':fig_sizes, 'text_colors':text_colors,
            'nxp':nxp, 'nzp':nzp, 'xstep':xstep, 'zstep':zstep, 'is_logs':is_logs,
            'fname':fname, 'fig_dir':fig_dir, 'bottom_panel':bottom_panel,
            'fdata_1d':fdata_1d, 'xlim':xlim, 'zlim':zlim, 'is_multi_Ay':True,
            'save_eps':save_eps}
    vx_plot = PlotMultiplePanels(**kwargs_plots)
    for cbar in vx_plot.cbar:
        cbar.set_ticks(np.arange(-0.8, 0.9, 0.4))
    vx_plot.ax1d.set_ylabel(r'$v_x/v_A$', fontdict=font, fontsize=20)
    xmin = np.min(x)
    xmax = np.max(x)
    z0 = z[nz/2]
    vx_plot.ax[0].plot([xmin, xmax], [z0, z0], linestyle='--', color='k')
    vx_plot.ax[1].plot([xmin, xmax], [z0, z0], linestyle='--', color='k')
    vx_plot.save_figures()
    plt.show()


def plot_epara_eperp(pic_info, ct, root_dir='../../'):
    kwargs = {"current_time":ct, "xl":50, "xr":150, "zb":-10, "zt":10}
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
    kernel = np.ones((ng,ng)) / float(ng*ng)
    epara = signal.convolve2d(epara, kernel)
    eperp = signal.convolve2d(eperp, kernel)
    ey = signal.convolve2d(ey, kernel, 'same')

    nx, = x.shape
    nz, = z.shape
    width = 0.73
    height = 0.36
    xs = 0.15
    ys = 0.92 - height
    gap = 0.05

    fig = plt.figure(figsize=[7, 5])
    ax1 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep":2, "zstep":2, "vmin":-0.1, "vmax":0.1}
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
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    # ax1.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
    ax1.tick_params(axis='x', labelbottom='off')
    ax1.tick_params(labelsize=20)
    # cbar1.ax.set_ylabel(r'$E_\perp$', fontdict=font, fontsize=20)
    cbar1.set_ticks(np.arange(-0.08, 0.1, 0.04))
    cbar1.ax.tick_params(labelsize=20)
    ax1.text(0.02, 0.8, r'$E_y$', color='k', fontsize=24, 
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform = ax1.transAxes)

    ys -= height + gap
    ax2 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep":1, "zstep":1, "vmin":-0.05, "vmax":0.05}
    p2, cbar2 = plot_2d_contour(x, z, epara, ax2, fig, **kwargs_plot)
    p2.set_cmap(plt.cm.seismic)
    # p2.set_cmap(cmaps.plasma)
    ax2.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep], 
            colors='black', linewidths=0.5)
    ax2.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax2.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
    ax2.tick_params(labelsize=20)
    # cbar2.ax.set_ylabel(r'$E_\parallel$', fontdict=font, fontsize=24)
    cbar2.set_ticks(np.arange(-0.04, 0.05, 0.02))
    cbar2.ax.tick_params(labelsize=20)
    ax2.text(0.02, 0.8, r'$E_\parallel$', color='k', fontsize=24, 
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform = ax2.transAxes)

    t_wci = ct * pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(int(t_wci+0.5)) + '/\Omega_{ci}$'
    ax1.set_title(title, fontdict=font, fontsize=24)
    
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    dir = '../img/img_apj/'
    if not os.path.isdir(dir):
        os.makedirs(dir)
    fname = dir + 'epara_perp' + '_' + str(ct).zfill(3) + '.eps'
    fig.savefig(fname)

    plt.show()
    # plt.close()


if __name__ == "__main__":
    # run_name = "mime25_beta002_noperturb"
    # root_dir = '/net/scratch2/xiaocanli/mime25-sigma1-beta002-200-100-noperturb/'
    run_name = "mime25_beta002"
    root_dir = "/scratch3/xiaocanli/sigma1-mime25-beta001/"
    # run_name = "mime25_beta0007"
    # root_dir = '/net/scratch2/xiaocanli/mime25-guide0-beta0007-200-100/'
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    # plot_by_time(run_name, root_dir, pic_info)
    # plot_vx_time(run_name, root_dir, pic_info)
    plot_epara_eperp(pic_info, 20, root_dir)
