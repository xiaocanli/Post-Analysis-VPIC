"""
Functions and classes for 2D contour plots of fields.
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
import palettable
import pic_information
from contour_plots import plot_2d_contour, read_2d_fields
from energy_conversion import calc_jdotes_fraction_multi, read_data_from_json
from fields_plot import *
from pic_information import list_pic_info_dir
from runs_name_path import ApJ_long_paper_runs
from shell_functions import *
from spectrum_fitting import calc_nonthermal_fraction

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

def plot_vx_frame(run_name, root_dir, pic_info, species):
    """Plot vx of one frame

    Args:
        run_name: the name of this run.
        root_dir: the root directory of this run.
        pic_info: PIC simulation information in a namedtuple.
        species: particle species.
    """
    ct = 80
    contour_color = ['k']
    vmin = [-1.0]
    vmax = [1.0]
    xs, ys = 0.15, 0.20
    w1, h1 = 0.75, 0.75
    axis_pos = [xs, ys, w1, h1]
    gaps = [0.1, 0.04]
    fig_sizes = (6, 3)
    nxp, nzp = 1, 1
    var_names = []
    var_name = r'$V_{' + species + 'x}/V_A$'
    var_names.append(var_name)
    colormaps = ['seismic', 'seismic', 'seismic']
    text_colors = ['k']
    xstep, zstep = 2, 2
    is_logs = [False]
    wpe_wce = pic_info.dtwce / pic_info.dtwpe
    va = wpe_wce / math.sqrt(pic_info.mime)  # Alfven speed of inflow region
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    dir = '../img/img_dpp/'
    if not os.path.isdir(dir):
        os.makedirs(dir)
    fig_dir = dir + run_name + '/'
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    kwargs = {"current_time":ct, "xl":0, "xr":200, "zb":-50, "zt":50}
    fname2 = root_dir + 'data/v' + species + 'x.gda'
    if not os.path.isfile(fname2):
        fname2 = root_dir + 'data/u' + species + 'x.gda'
    x, z, vx = read_2d_fields(pic_info, fname2, **kwargs) 
    fname5 = root_dir + 'data/Ay.gda'
    x, z, Ay = read_2d_fields(pic_info, fname5, **kwargs) 
    fdata = [vx/va]
    fname = 'v' + species
    kwargs_plots = {'current_time':ct, 'x':x, 'z':z, 'Ay':Ay,
            'fdata':fdata, 'contour_color':contour_color, 'colormaps':colormaps,
            'vmin':vmin, 'vmax':vmax, 'var_names':var_names, 'axis_pos':axis_pos,
            'gaps':gaps, 'fig_sizes':fig_sizes, 'text_colors':text_colors,
            'nxp':nxp, 'nzp':nzp, 'xstep':xstep, 'zstep':zstep, 'is_logs':is_logs,
            'fname':fname, 'fig_dir':fig_dir}
    vfields_plot = PlotMultiplePanels(**kwargs_plots)
    plt.show()


def plot_anisotropy(run_name, root_dir, pic_info, species, ct):
    """Plot pressure anisotropy

    Args:
        run_name: the name of this run.
        root_dir: the root directory of this run.
        pic_info: PIC simulation information in a namedtuple.
        species: particle species.
    """
    contour_color = ['k']
    vmin = [0.1]
    vmax = [10.0]
    xs, ys = 0.15, 0.20
    w1, h1 = 0.75, 0.75
    axis_pos = [xs, ys, w1, h1]
    gaps = [0.1, 0.04]
    fig_sizes = (6, 3)
    nxp, nzp = 1, 1
    var_names = []
    var_name = r'$p_{' + species + r'\parallel}/p_{' + species + r'\perp}$'
    var_names.append(var_name)
    colormaps = ['seismic', 'seismic', 'seismic']
    text_colors = ['k']
    xstep, zstep = 2, 2
    is_logs = [True]
    wpe_wce = pic_info.dtwce / pic_info.dtwpe
    va = wpe_wce / math.sqrt(pic_info.mime)  # Alfven speed of inflow region
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    dir = '../img/img_dpp/'
    if not os.path.isdir(dir):
        os.makedirs(dir)
    fig_dir = dir + run_name + '/'
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    kwargs = {"current_time":ct, "xl":0, "xr":200, "zb":-50, "zt":50}
    fnames = []
    fnames.append(root_dir + 'data/p' + species + '-xx.gda')
    fnames.append(root_dir + 'data/p' + species + '-yy.gda')
    fnames.append(root_dir + 'data/p' + species + '-zz.gda')
    fnames.append(root_dir + 'data/p' + species + '-xy.gda')
    fnames.append(root_dir + 'data/p' + species + '-xz.gda')
    fnames.append(root_dir + 'data/p' + species + '-yz.gda')
    pre = []
    for fname in fnames:
        x, z, data = read_2d_fields(pic_info, fname, **kwargs) 
        pre.append(data)
    x, z, bx = read_2d_fields(pic_info, root_dir + "data/bx.gda", **kwargs) 
    x, z, by = read_2d_fields(pic_info, root_dir + "data/by.gda", **kwargs) 
    x, z, bz = read_2d_fields(pic_info, root_dir + "data/bz.gda", **kwargs) 
    x, z, absB = read_2d_fields(pic_info, root_dir + "data/absB.gda", **kwargs) 
    ppara = pre[0]*bx*bx + pre[1]*by*by + pre[2]*bz*bz + \
            pre[3]*bx*by*2.0 + pre[4]*bx*bz*2.0 + pre[5]*by*bz*2.0
    ppara /= absB * absB
    pperp = 0.5 * (pre[0]+pre[1]+pre[2]-ppara)
    fdata = [ppara/pperp]
    fname5 = root_dir + 'data/Ay.gda'
    x, z, Ay = read_2d_fields(pic_info, fname5, **kwargs) 
    fname = 'aniso_' + species
    kwargs_plots = {'current_time':ct, 'x':x, 'z':z, 'Ay':Ay,
            'fdata':fdata, 'contour_color':contour_color, 'colormaps':colormaps,
            'vmin':vmin, 'vmax':vmax, 'var_names':var_names, 'axis_pos':axis_pos,
            'gaps':gaps, 'fig_sizes':fig_sizes, 'text_colors':text_colors,
            'nxp':nxp, 'nzp':nzp, 'xstep':xstep, 'zstep':zstep, 'is_logs':is_logs,
            'fname':fname, 'fig_dir':fig_dir}
    pre_plot = PlotMultiplePanels(**kwargs_plots)
    plt.show()


def plot_compression_fields_single(run_name, root_dir, pic_info, species, ct):
    """Plot fields to fluid compression or shear

    Args:
        run_name: the name of this run.
        root_dir: the root directory of this run.
        pic_info: PIC simulation information in a namedtuple.
        species: particle species.
    """
    nj = 4
    contour_color = ['k'] * nj
    vmin = [-1.0] * nj
    vmax = [1.0] * nj
    vmin[0] = -0.5
    vmax[0] = 0.5
    vmin[1] = -0.5
    vmax[1] = 0.5
    xs, ys = 0.11, 0.82
    w1, h1 = 0.8, 0.16
    axis_pos = [xs, ys, w1, h1]
    gaps = [0.1, 0.02]
    fig_sizes = (8, 10)
    nxp, nzp = 1, nj
    var_names = []
    fname1 = r'$-p\nabla\cdot\boldsymbol{u}$'
    var_names.append(fname1)
    fname2 = r'$-(p_\parallel - p_\perp)b_ib_j\sigma_{ij}$'
    var_names.append(fname2)
    var_names.append(r'$\boldsymbol{u}\cdot(\nabla\cdot\mathcal{P})$')
    fname = r'$' + r'\boldsymbol{j}_' + species + r'\cdot\boldsymbol{E}' + '$'
    var_names.append(fname)
    colormaps = ['seismic'] * nj
    # text_colors = colors[0:nj]
    text_colors = ['r', 'g', 'b', 'k']
    xstep, zstep = 1, 1
    is_logs = [False] * nj
    wpe_wce = pic_info.dtwce / pic_info.dtwpe
    va = wpe_wce / math.sqrt(pic_info.mime)  # Alfven speed of inflow region
    b0 = pic_info.b0
    j0 = 0.1 * va**2 * b0
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    dir = '../img/img_dpp/'
    if not os.path.isdir(dir):
        os.makedirs(dir)
    fig_dir = dir + run_name + '/'
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    kwargs = {"current_time":ct, "xl":0, "xr":200, "zb":-50, "zt":50}
    fnames = []
    fname = root_dir + 'data1/pdiv_v00_' + species + '.gda'
    fnames.append(fname)
    fname = root_dir + 'data1/pshear00_' + species + '.gda'
    fnames.append(fname)
    fname = root_dir + 'data1/vdot_div_ptensor00_' + species + '.gda'
    fnames.append(fname)
    fname = root_dir + 'data1/jqnvpara_dote00_' + species + '.gda'
    fnames.append(fname)
    fname = root_dir + 'data1/jqnvperp_dote00_' + species + '.gda'
    fnames.append(fname)
    dv = pic_info.dx_di * pic_info.dz_di * pic_info.mime
    ng = 7
    kernel = np.ones((ng,ng)) / float(ng*ng)
    fdata = []
    fdata_1d = []
    for fname in fnames[0:3]:
        x, z, data = read_2d_fields(pic_info, fname, **kwargs) 
        fdata_cum = np.cumsum(np.sum(data, axis=0)) * dv
        fdata_1d.append(fdata_cum)
        data_new = signal.convolve2d(data, kernel, 'same')
        fdata.append(data_new)
    jdote = 0
    jdote_cum = 0
    for fname in fnames[3:5]:
        x, z, data = read_2d_fields(pic_info, fname, **kwargs) 
        fdata_cum = np.cumsum(np.sum(data, axis=0)) * dv
        jdote_cum += fdata_cum
        data_new = signal.convolve2d(data, kernel, 'same')
        jdote += data_new
    fdata.append(jdote)
    fdata_1d.append(jdote_cum)
    fdata = np.asarray(fdata)
    fdata_1d = np.asarray(fdata_1d)
    fdata /= j0  # Normalization
    fname2 = root_dir + 'data/Ay.gda'
    x, z, Ay = read_2d_fields(pic_info, fname2, **kwargs) 
    fname = 'comp_' + species
    bottom_panel = True
    xlim = [0, 200]
    zlim = [-25, 25]
    kwargs_plots = {'current_time':ct, 'x':x, 'z':z, 'Ay':Ay,
            'fdata':fdata, 'contour_color':contour_color, 'colormaps':colormaps,
            'vmin':vmin, 'vmax':vmax, 'var_names':var_names, 'axis_pos':axis_pos,
            'gaps':gaps, 'fig_sizes':fig_sizes, 'text_colors':text_colors,
            'nxp':nxp, 'nzp':nzp, 'xstep':xstep, 'zstep':zstep, 'is_logs':is_logs,
            'fname':fname, 'fig_dir':fig_dir, 'bottom_panel':bottom_panel,
            'fdata_1d':fdata_1d, 'xlim':xlim, 'zlim':zlim}
    jdote_plot = PlotMultiplePanels(**kwargs_plots)

    # plt.show()


def get_temperature_cut(run_name, root_dir, pic_info, ct):
    """get temperature cut along a line
    """
    kwargs = {"current_time":ct, "xl":0, "xr":200, "zb":-1, "zt":1}
    fnames_e = ['pe-xx', 'pe-yy', 'pe-zz', 'ne']
    fnames_i = ['pi-xx', 'pi-yy', 'pi-zz', 'ni']
    pe = 0.0
    pi = 0.0
    for name in fnames_e[0:3]:
        fname = root_dir + 'data/' + name + '.gda'
        x, z, data = read_2d_fields(pic_info, fname, **kwargs) 
        pe += data
    fname = root_dir + 'data/' + fnames_e[3] + '.gda'
    x, z, nrho = read_2d_fields(pic_info, fname, **kwargs) 
    te = pe / nrho / 3.0
    for name in fnames_i[0:3]:
        fname = root_dir + 'data/' + name + '.gda'
        x, z, data = read_2d_fields(pic_info, fname, **kwargs) 
        pi += data
    fname = root_dir + 'data/' + fnames_i[3] + '.gda'
    x, z, nrho = read_2d_fields(pic_info, fname, **kwargs) 
    ti = pi / nrho / 3.0
    nx, = x.shape
    nz, = z.shape
    return (x, z, te[nz/2, :], ti[nz/2, :])


def plot_temperature_cut(run_name, root_dir, pic_info, cts):
    """plot temperature cut along a line
    """
    tes = []
    tis = []
    for ct in range(cts[0], cts[1]):
        x, z, te, ti = get_temperature_cut(run_name, root_dir, pic_info, ct)
        tes.append(te)
        tis.append(ti)

    nt = len(tes)
    fig = plt.figure(figsize=[7, 5])
    w1, h1 = 0.78, 0.39
    xs, ys = 0.17, 0.97-h1
    gap = 0.05
    ax1 = fig.add_axes([xs, ys, w1, h1])
    ax1.plot(x, tes[0], linewidth=2, color='b', label=r'$t\Omega_{ci}=550$')
    ax1.plot(x, tes[5], linewidth=2, color='r', label=r'$t\Omega_{ci}=575$')
    ax1.set_xlim([15, 60])
    ax1.tick_params(labelsize=20)
    ax1.tick_params(axis='x', labelbottom='off')
    ax1.set_ylabel(r'$T_e$', fontdict=font, fontsize=24)
    x1 = 36.5
    x2 = 38.7
    ax1.plot([x1, x1], ax1.get_ylim(), color='b', linestyle='--')
    ax1.plot([x2, x2], ax1.get_ylim(), color='r', linestyle='--')
    ax1.set_ylim([0, 0.16])
    # leg = ax1.legend(loc=4, prop={'size':20}, ncol=1,
    #         shadow=False, fancybox=False, frameon=False)
    ax1.text(0.7, 0.3, r'$t\Omega_{ci}=550$',
            color='blue', fontsize=24,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform = ax1.transAxes)
    ax1.text(0.7, 0.1, r'$t\Omega_{ci}=575$',
            color='red', fontsize=24,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform = ax1.transAxes)
    ys -= h1 + gap
    ax2 = fig.add_axes([xs, ys, w1, h1])
    ax2.plot(x, tis[0], linewidth=2, color='b')
    ax2.plot(x, tis[5], linewidth=2, color='r')
    ax2.set_xlim(ax1.get_xlim())
    ax2.tick_params(labelsize=20)
    ax2.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
    ax2.set_ylabel(r'$T_i$', fontdict=font, fontsize=24)
    ax2.plot([x1, x1], ax2.get_ylim(), color='b', linestyle='--')
    ax2.plot([x2, x2], ax2.get_ylim(), color='r', linestyle='--')
    ax2.set_ylim([0, 0.3])
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    fig.savefig('../img/temp_cut.eps')
    plt.show()


def plot_thermal_temperature_single(run_name, root_dir, pic_info, ct):
    """Plot thermal temperature one both species for a single run

    Args:
        run_name: the name of this run.
        root_dir: the root directory of this run.
        pic_info: PIC simulation information in a namedtuple.
    """
    contour_color = ['w', 'w']
    vmin = [0, 0]
    vmax = [0.15, 0.3]
    # Change with different runs
    b0 = pic_info.b0
    wpe_wce = pic_info.dtwce / pic_info.dtwpe
    va = wpe_wce / math.sqrt(pic_info.mime)  # Alfven speed of inflow region
    mime = pic_info.mime
    va = wpe_wce / math.sqrt(mime)  # Alfven speed of inflow region
    # The standard va is 0.2, and the standard mi/me=25
    vmin = np.asarray(vmin) * va**2 * mime / (0.2**2 * 25)
    vmax = np.asarray(vmax) * va**2 * mime/ (0.2**2 * 25)
    xs, ys = 0.15, 0.60
    w1, h1 = 0.72, 0.36
    axis_pos = [xs, ys, w1, h1]
    gaps = [0.1, 0.06]
    fig_sizes = (6, 5)
    nxp, nzp = 1, 2
    var_names = [r'$T_e$', r'$T_i$']
    colormaps = ['gist_heat', 'gist_heat']
    text_colors = ['w', 'w']
    xstep, zstep = 2, 2
    is_logs = [False, False]
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    dir = '../img/img_dpp/'
    if not os.path.isdir(dir):
        os.makedirs(dir)
    fig_dir = dir + run_name + '/'
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    kwargs = {"current_time":ct, "xl":0, "xr":200, "zb":-50, "zt":50}
    fnames_e = ['pe-xx', 'pe-yy', 'pe-zz', 'ne']
    fnames_i = ['pi-xx', 'pi-yy', 'pi-zz', 'ni']
    pe = 0.0
    pi = 0.0
    for name in fnames_e[0:3]:
        fname = root_dir + 'data/' + name + '.gda'
        x, z, data = read_2d_fields(pic_info, fname, **kwargs) 
        pe += data
    fname = root_dir + 'data/' + fnames_e[3] + '.gda'
    x, z, nrho = read_2d_fields(pic_info, fname, **kwargs) 
    te = pe / nrho / 3.0
    for name in fnames_i[0:3]:
        fname = root_dir + 'data/' + name + '.gda'
        x, z, data = read_2d_fields(pic_info, fname, **kwargs) 
        pi += data
    fname = root_dir + 'data/' + fnames_i[3] + '.gda'
    x, z, nrho = read_2d_fields(pic_info, fname, **kwargs) 
    ti = pi / nrho / 3.0
    fdata = [te, ti]
    fname2 = root_dir + 'data/Ay.gda'
    x, z, Ay = read_2d_fields(pic_info, fname2, **kwargs) 
    fname = 'temp'
    kwargs_plots = {'current_time':ct, 'x':x, 'z':z, 'Ay':Ay,
            'fdata':fdata, 'contour_color':contour_color, 'colormaps':colormaps,
            'vmin':vmin, 'vmax':vmax, 'var_names':var_names, 'axis_pos':axis_pos,
            'gaps':gaps, 'fig_sizes':fig_sizes, 'text_colors':text_colors,
            'nxp':nxp, 'nzp':nzp, 'xstep':xstep, 'zstep':zstep, 'is_logs':is_logs,
            'fname':fname, 'fig_dir':fig_dir}
    pfields_plot = PlotMultiplePanels(**kwargs_plots)

    plt.show()


def compression_time(pic_info, species, jdote, ylim1, ax, root_dir='../data/'):
    """The time evolution of compression related terms.

    Args:
        pic_info: namedtuple for the PIC simulation information.
        species: 'e' for electrons, 'i' for ions.
    """
    ntf = pic_info.ntf
    tfields = pic_info.tfields
    fname = root_dir + "compression00_" + species + ".gda"
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

    fname = root_dir + "shear00_" + species + ".gda"
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

    fname = root_dir + "div_vdot_ptensor00_" + species + ".gda"
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

    fname = root_dir + "vdot_div_ptensor00_" + species + ".gda"
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

    ene_magnetic = pic_info.ene_magnetic
    b0 = pic_info.b0
    enorm = ene_magnetic[0] / b0**2
    dtwpe = pic_info.dtwpe
    dtwci = pic_info.dtwci
    dt_fields = pic_info.dt_fields * dtwpe / dtwci
    pdiv_u_cum = np.cumsum(pdiv_u) * dt_fields
    pshear_cum = np.cumsum(pshear) * dt_fields
    div_vdot_ptensor_cum = np.cumsum(div_vdot_ptensor) * dt_fields
    vdot_div_ptensor_cum = np.cumsum(vdot_div_ptensor) * dt_fields
    pdiv_u_cum /= enorm
    pshear_cum /= enorm
    div_vdot_ptensor_cum /= enorm
    vdot_div_ptensor_cum /= enorm

    # jdote = read_jdote_data(species)
    jqnudote = jdote.jqnupara_dote + jdote.jqnuperp_dote
    jqnudote_cum = jdote.jqnupara_dote_int + jdote.jqnuperp_dote_int
    jqnudote_cum /= enorm

    p1 = ax.plot(tfields, pdiv_u_cum, linewidth=2, color='r')
    p2 = ax.plot(tfields, pshear_cum, linewidth=2, color='g')
    ax.set_ylabel(r'$\varepsilon_c$', fontdict=font, fontsize=20)
    ax.tick_params(axis='x', labelbottom='off')
    ax.tick_params(labelsize=16)
    ax.legend(loc=2, prop={'size':20}, ncol=1,
            shadow=False, fancybox=False, frameon=False)
    ax.set_xlim(ax.get_xlim())


def plot_compression_time_beta(species):
    """
    
    Plot time evolution of compression and shear heating for runs with
    different plasma beta

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
    fig = plt.figure(figsize=[7, 5])
    w1, h1 = 0.8, 0.2
    xs, ys = 0.96-w1, 0.97-h1
    gap = 0.02
    nrun = len(run_names)
    ylim1 = np.zeros((nrun, 2))
    if species == 'e':
        ylim1[0, :] = -0.05, 0.15
        ylim1[1, :] = -0.3, 1.1
        ylim1[2, :] = -1.0, 5
        ylim1[3, :] = -10.0, 30.0
    else:
        ylim1[0, :] = -0.1, 0.25
        ylim1[1, :] = -0.6, 2.2
        ylim1[2, :] = -2.0, 10
        ylim1[3, :] = -20.0, 60.0
    axs = []
    for i in range(4):
        run_name = run_names[i]
        base_dir = base_dirs[i]
        picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
        jdote_fname = dir_jdote + 'jdote_' + run_name + '_' + species + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        jdote_data = read_data_from_json(jdote_fname)
        fpath_comp = '../data/compression/' + run_name + '/'
        b0 = pic_info.b0
        ylim2 = ylim1 * b0**2
        ax = fig.add_axes([xs, ys, w1, h1])
        axs.append(ax)
        compression_time(pic_info, species, jdote_data, ylim2[i,:], ax, fpath_comp)
        ys -= h1 + gap
    for ax in axs:
        ax.set_xlim([0, 1200])
    if species == 'e':
        axs[0].set_yticks(np.arange(0, 0.007, 0.002))
        axs[1].set_yticks(np.arange(0, 0.025, 0.01))
        axs[2].set_yticks(np.arange(0, 0.09, 0.02))
        axs[3].set_yticks(np.arange(0, 0.26, 0.05))
    else:
        axs[0].set_yticks(np.arange(0, 0.013, 0.004))
        axs[1].set_yticks(np.arange(0, 0.07, 0.02))
        axs[2].set_yticks(np.arange(0, 0.16, 0.05))
        axs[3].set_yticks(np.arange(0, 0.7, 0.2))
    axs[-1].tick_params(axis='x', labelbottom='on')
    axs[-1].set_xlabel(r'$t\Omega_{ci}$', fontdict=font, fontsize=20)

    label1 = r'$-p\nabla\cdot\mathbf{u}$'
    label2 = r'$-(p_\parallel - p_\perp)b_ib_j\sigma_{ij}$'
    axs[0].text(0.4, 0.2, label1, color='red', fontsize=20, 
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform = axs[0].transAxes)
    axs[0].text(0.6, 0.2, label2, color='green', fontsize=20, 
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform = axs[0].transAxes)
    axs[0].text(0.05, 0.8, r'$\beta_e=0.2$', color='black', fontsize=20, 
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform = axs[0].transAxes)
    axs[1].text(0.05, 0.8, r'$\beta_e=0.07$', color='black', fontsize=20, 
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform = axs[1].transAxes)
    axs[2].text(0.05, 0.8, r'$\beta_e=0.02$', color='black', fontsize=20, 
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform = axs[2].transAxes)
    axs[3].text(0.05, 0.8, r'$\beta_e=0.007$', color='black', fontsize=20, 
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform = axs[3].transAxes)

    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    dir = '../img/img_dpp/'
    if not os.path.isdir(dir):
        os.makedirs(dir)
    fname = dir + 'comp_beta_' + species + '.eps'
    fig.savefig(fname)
    plt.show()


def plot_compression_time_temp(species):
    """
    
    Plot time evolution of compression and shear heating for runs with
    different plasma temperature

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
    fig = plt.figure(figsize=[7, 5])
    w1, h1 = 0.8, 0.27
    xs, ys = 0.96-w1, 0.97-h1
    gap = 0.02
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
        ylim1[6, :] = -1.0, 2.2
        ylim1[7, :] = -5.0, 15.0
        ylim1[8, :] = -3.0, 7.0
    runs = [5, 6, 2]
    axs = []
    for i in runs:
        run_name = run_names[i]
        base_dir = base_dirs[i]
        picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
        jdote_fname = dir_jdote + 'jdote_' + run_name + '_' + species + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        jdote_data = read_data_from_json(jdote_fname)
        fpath_comp = '../data/compression/' + run_name + '/'
        b0 = pic_info.b0
        ylim2 = ylim1 * b0**2
        ax = fig.add_axes([xs, ys, w1, h1])
        axs.append(ax)
        compression_time(pic_info, species, jdote_data, ylim2[i,:], ax, fpath_comp)
        ys -= h1 + gap
    for ax in axs:
        ax.set_xlim([0, 1200])
    if species == 'e':
        axs[0].set_yticks(np.arange(0, 0.007, 0.002))
        axs[1].set_yticks(np.arange(0, 0.025, 0.01))
        axs[2].set_yticks(np.arange(0, 0.09, 0.02))
    else:
        axs[0].set_yticks(np.arange(0, 0.021, 0.005))
        axs[1].set_yticks(np.arange(0, 0.07, 0.02))
        axs[2].set_yticks(np.arange(0, 0.17, 0.04))
    axs[-1].tick_params(axis='x', labelbottom='on')
    axs[-1].set_xlabel(r'$t\Omega_{ci}$', fontdict=font, fontsize=20)

    label1 = r'$-p\nabla\cdot\mathbf{u}$'
    label2 = r'$-(p_\parallel - p_\perp)b_ib_j\sigma_{ij}$'
    axs[0].text(0.4, 0.2, label1, color='red', fontsize=20, 
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform = axs[0].transAxes)
    axs[0].text(0.6, 0.2, label2, color='green', fontsize=20, 
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform = axs[0].transAxes)
    axs[0].text(0.05, 0.8, r'$v_\text{the}/c=0.045$', color='black', fontsize=20, 
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform = axs[0].transAxes)
    axs[1].text(0.05, 0.8, r'$v_\text{the}/c=0.08$', color='black', fontsize=20, 
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform = axs[1].transAxes)
    axs[2].text(0.05, 0.8, r'$v_\text{the}/c=0.14$', color='black', fontsize=20, 
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform = axs[2].transAxes)

    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    dir = '../img/img_dpp/'
    if not os.path.isdir(dir):
        os.makedirs(dir)
    fname = dir + 'comp_temp_' + species + '.eps'
    fig.savefig(fname)
    plt.show()


def plot_by_time(run_name, root_dir, pic_info):
    """Plot by contour at multiple time frames

    Args:
        run_name: the name of this run.
        root_dir: the root directory of this run.
        pic_info: PIC simulation information in a namedtuple.
    """
    ct = 80
    nt = 5
    contour_color = ['k'] * nt
    vmin = [-1.0] * nt
    vmax = [1.0] * nt
    xs, ys = 0.17, 0.81
    w1, h1 = 0.68, 0.165
    fig_sizes = (5, 10)
    axis_pos = [xs, ys, w1, h1]
    gaps = [0.1, 0.02]
    nxp, nzp = 1, nt
    cts = 2**np.arange(nt) * 25
    var_names = []
    for i in range(nt):
        var_name = r'$t=' + str(cts[i]) + r'/\Omega_{ci}$'
        var_names.append(var_name)
    colormaps = ['seismic'] * 5
    text_colors = ['k'] * 5
    xstep, zstep = 2, 2
    is_logs = [False] * 5
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    dir = '../img/img_dpp/'
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
            'fname':fname, 'fig_dir':fig_dir, 'is_multi_Ay':True}
    by_plot = PlotMultiplePanels(**kwargs_plots)
    plt.show()


def plot_energy_conversion_fraction():
    """Plot energy evolution for multiple runs.
    """
    dir = '../data/pic_info/'
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    odir = '../img/ene_evolution/'
    if not os.path.isdir(odir):
        os.makedirs(odir)
    fnames = list_pic_info_dir(dir)
    print fnames
    # run_id = [2, 5, 1, 6, 7, 8, 0, 4]
    run_id = [2, 8, 1, 9, 10, 12, 0, 7]
    nrun = len(run_id)
    ene_fraction = np.zeros(nrun)
    dke_dki = np.zeros(nrun)
    irun = 0
    i = 0
    for fname in fnames:
        print i, fname
        i += 1
    for i in run_id:
        fname = fnames[i]
        print fname
        rname = fname.replace(".json", ".eps")
        oname = rname.replace("pic_info", "enes")
        oname = odir + oname
        fname = dir + fname
        pic_info = read_data_from_json(fname)
        tenergy = pic_info.tenergy
        ene_magnetic = pic_info.ene_magnetic
        kene_e = pic_info.kene_e
        kene_i = pic_info.kene_i
        dke = kene_e[-1] - kene_e[0]
        dki = kene_i[-1] - kene_i[0]
        enorm = ene_magnetic[0]
        ene_fraction[irun] = 1.0 - ene_magnetic[-1] / enorm
        dke_dki[irun] = dke / dki
        irun += 1
    x = np.arange(8)
    fig = plt.figure(figsize=[7, 5])
    xs, ys = 0.16, 0.13
    w1, h1 = 0.7, 0.8
    ax = fig.add_axes([xs, ys, w1, h1])
    ax.plot(x, ene_fraction, color='r', marker='o', markersize=10,
            linestyle='', markeredgecolor = 'r')
    labels = [r'R6', r'R1', r'R4', r'R2', r'R5', r'R3', r'R7', r'R8']
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim([-0.5, 7.5])
    ax.set_ylabel(r'$|\Delta\varepsilon_b|/\varepsilon_{b0}$', color='r',
            fontdict=font, fontsize=24)
    ax.tick_params(labelsize=20)
    ax.text(0.4, 0.15, r'$m_i/m_e=100$', color='k', fontsize=20, 
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform = ax.transAxes)
    for tl in ax.get_yticklabels():
        tl.set_color('r')
    ax1 = ax.twinx()
    ax1.plot(x, dke_dki, color='b', marker='v', markersize=10,
            linestyle='', markeredgecolor = 'b')
    ax1.set_xlim([-0.5, 7.5])
    ax1.set_ylabel(r'$\Delta K_e/\Delta K_i$', color='b',
            fontdict=font, fontsize=24)
    ax1.tick_params(labelsize=20)
    for tl in ax1.get_yticklabels():
        tl.set_color('b')
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    dir = '../img/img_dpp/'
    if not os.path.isdir(dir):
        os.makedirs(dir)
    fname = dir + 'ene_fraction_.eps'
    fig.savefig(fname)
    plt.show()


def plot_nonthermal_fraction():
    # nnth_e, enth_e = calc_nonthermal_fraction('e')
    # nnth_i, enth_i = calc_nonthermal_fraction('h')
    nth_e = [0.66, 0.55, 0.52, 0.53, 0.42, 0.49, 0.38, 0.17]
    nth_i = [0.60, 0.52, 0.49, 0.49, 0.44, 0.50, 0.40, 0.19]
    x = np.arange(8)
    fig = plt.figure(figsize=[7, 5])
    xs, ys = 0.15, 0.13
    w1, h1 = 0.8, 0.8
    ax = fig.add_axes([xs, ys, w1, h1])
    ax.plot(x, nth_e[:8], color='r', marker='o', markersize=10,
            linestyle='')
    ax.plot(x, nth_i[:8], color='b', marker='o', markersize=10,
            linestyle='')
    labels = [r'R6', r'R1', r'R2', r'R4', r'R5', r'R3', r'R7', r'R8']
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim([-0.5, 7.5])
    ax.set_ylabel(r'Nonthermal Fraction', fontdict=font, fontsize=24)
    ax.tick_params(labelsize=20)
    ax.text(0.05, 0.15, r'Electron', color='r', fontsize=24, 
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform = ax.transAxes)
    ax.text(0.05, 0.25, r'Ion', color='b', fontsize=24, 
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform = ax.transAxes)
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    dir = '../img/img_dpp/'
    if not os.path.isdir(dir):
        os.makedirs(dir)
    fname = dir + 'nth_fraction_.eps'
    fig.savefig(fname)
    plt.show()


def plot_jpara_dote_fraction():
    """Plot energy evolution from the parallel and perpendicular directions.
    """
    dir = '../data/jdote_data/'
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    odir = '../img/jdote/'
    if not os.path.isdir(odir):
        os.makedirs(odir)
    base_dirs, run_names = ApJ_long_paper_runs()
    run_id = [3, 2, 6, 7, 4, 5, 1, 0]
    nrun = len(run_id)
    fraction_jpara_e = np.zeros(nrun)
    fraction_jpara_i = np.zeros(nrun)
    fraction_jperp_e = np.zeros(nrun)
    fraction_jperp_i = np.zeros(nrun)
    irun = 0
    for i in run_id:
        run_name = run_names[i]
        picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
        jdote_e_fname = '../data/jdote_data/jdote_' + run_name + '_e.json'
        jdote_i_fname = '../data/jdote_data/jdote_' + run_name + '_i.json'
        pic_info = read_data_from_json(picinfo_fname)
        kene_e = pic_info.kene_e
        kene_i = pic_info.kene_i
        dkene_e = kene_e[-1] - kene_e[0]
        dkene_i = kene_i[-1] - kene_i[0]
        jdote_e = read_data_from_json(jdote_e_fname)
        jdote_i = read_data_from_json(jdote_i_fname)
        jpara_dote_e = jdote_e.jqnupara_dote_int
        jperp_dote_e = jdote_e.jqnuperp_dote_int
        jpara_dote_i = jdote_i.jqnupara_dote_int
        jperp_dote_i = jdote_i.jqnuperp_dote_int
        fraction_jpara_e[irun] = jpara_dote_e[-1] / dkene_e
        fraction_jperp_e[irun] = jperp_dote_e[-1] / dkene_e
        fraction_jpara_i[irun] = jpara_dote_i[-1] / dkene_i
        fraction_jperp_i[irun] = jperp_dote_i[-1] / dkene_i
        irun += 1
    x = np.arange(8)
    fig = plt.figure(figsize=[7, 5])
    xs, ys = 0.18, 0.13
    w1, h1 = 0.78, 0.8
    ax = fig.add_axes([xs, ys, w1, h1])
    ax.plot(x, fraction_jpara_e[:8], color='r', marker='o', markersize=10,
            linestyle='')
    ax.plot(x, fraction_jpara_i[:8], color='b', marker='o', markersize=10,
            linestyle='')
    labels = [r'R6', r'R1', r'R2', r'R4', r'R5', r'R3', r'R7', r'R8']
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim([-0.5, 7.5])
    ax.set_ylabel(r'Fraction of Parallel Acceleration', fontdict=font, fontsize=24)
    ax.tick_params(labelsize=20)
    ax.text(0.05, 0.15, r'Electron', color='r', fontsize=24, 
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform = ax.transAxes)
    ax.text(0.05, 0.25, r'Ion', color='b', fontsize=24, 
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform = ax.transAxes)
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    dir = '../img/img_dpp/'
    if not os.path.isdir(dir):
        os.makedirs(dir)
    fname = dir + 'para_acc_fraction.eps'
    fig.savefig(fname)
    plt.show()


def plot_jdrifts_fraction(species):
    jdote_drifts_np = calc_jdotes_fraction_multi(species)
    run_id = [3, 2, 6, 7, 4, 5, 1, 0]
    nrun = len(run_id)
    irun = 0
    jdrifts_fraction = np.zeros((nrun, 3))
    for i in run_id:
        fraction = jdote_drifts_np[i]
        jdrifts_fraction[irun, 0] = fraction[0]
        jdrifts_fraction[irun, 1] = fraction[3]
        jdrifts_fraction[irun, 2] = fraction[2]
        irun += 1
    x = np.arange(8)
    fig = plt.figure(figsize=[7, 5])
    xs, ys = 0.18, 0.13
    w1, h1 = 0.78, 0.8
    ax = fig.add_axes([xs, ys, w1, h1])
    ax.plot(x, jdrifts_fraction[:8, 0], color='b', marker='o', markersize=10,
            linestyle='')
    ax.plot(x, jdrifts_fraction[:8, 1], color='g', marker='o', markersize=10,
            linestyle='')
    ax.plot(x, jdrifts_fraction[:8, 2], color='r', marker='o', markersize=10,
            linestyle='')
    labels = [r'R6', r'R1', r'R2', r'R4', r'R5', r'R3', r'R7', r'R8']
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim([-0.5, 7.5])
    ax.set_ylabel(r'Ratio to the energy conversion', fontdict=font, fontsize=24)
    ax.tick_params(labelsize=20)
    ax.text(0.05, 0.05, r'$\boldsymbol{j}_c\cdot\boldsymbol{E}$', color='b', fontsize=24, 
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform = ax.transAxes)
    ax.text(0.25, 0.05, r'$\boldsymbol{j}_g\cdot\boldsymbol{E}$', color='g', fontsize=24, 
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform = ax.transAxes)
    ax.text(0.45, 0.05, r'$\boldsymbol{j}_m\cdot\boldsymbol{E}$', color='r', fontsize=24, 
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform = ax.transAxes)
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    dir = '../img/img_dpp/'
    if not os.path.isdir(dir):
        os.makedirs(dir)
    fname = dir + 'jdrifts_fraction_' + species + '.eps'
    fig.savefig(fname)
    plt.show()


def plot_energy_conversion_fraction_beta():
    """Plot energy evolution for multiple runs with different beta
    """
    picinfo_dir = '../data/pic_info/'
    mkdir_p(picinfo_dir)
    odir = '../img/ene_evolution/'
    mkdir_p(odir)
    run_names = ['sigma1-mime25-beta0002', 'mime25_beta0007',
                 'mime25_beta002', 'mime25_beta007', 'mime25_beta02']
    labels = ['R7\n 0.0002', 'R6\n 0.007', 'R1\n 0.02', 'R7\n 0.07', 'R8\n 0.2']
    run_label = r'$\beta_e = $'
    fname = 'ene_fraction_beta.eps'
    # run_names = ['mime25_beta002', 'mime25_beta002_sigma033',
    #         'mime25_beta002_sigma01']
    # labels = ['R1\n $1.0$', 'R2\n $\sqrt{3}$', 'R3\n $\sqrt{10}$']
    # run_label = r'$\omega_{pe} / \Omega_{ce} = $'
    # fname = 'ene_fraction_wpe_wce.eps'
    nrun = len(run_names)
    ene_fraction = np.zeros(nrun)
    dke_dki = np.zeros(nrun)
    irun = 0
    for run_name in run_names:
        print run_name
        rname = run_name.replace(".json", ".eps")
        oname = rname.replace("pic_info", "enes")
        oname = odir + oname
        picinfo_fname = picinfo_dir + 'pic_info_' + run_name + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        tenergy = pic_info.tenergy
        ene_magnetic = pic_info.ene_magnetic
        kene_e = pic_info.kene_e
        kene_i = pic_info.kene_i
        dke = kene_e[-1] - kene_e[0]
        dki = kene_i[-1] - kene_i[0]
        enorm = ene_magnetic[0]
        ene_fraction[irun] = 1.0 - ene_magnetic[-1] / enorm
        dke_dki[irun] = dke / dki
        irun += 1
    x = np.arange(nrun)
    fig = plt.figure(figsize=[7, 5])
    xs, ys = 0.16, 0.13
    w1, h1 = 0.7, 0.8
    ax = fig.add_axes([xs, ys, w1, h1])
    ax.plot(x, ene_fraction, color='r', marker='o', markersize=10,
            linestyle='', markeredgecolor = 'r')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim([-0.5, nrun - 0.5])
    ax.set_ylabel(r'$|\Delta\varepsilon_b|/\varepsilon_{b0}$', color='r',
            fontdict=font, fontsize=24)
    ax.tick_params(labelsize=20)
    ax.text(-0.12, -0.12, run_label, color='k', fontsize=20, 
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform = ax.transAxes)
    for tl in ax.get_yticklabels():
        tl.set_color('r')
    ax1 = ax.twinx()
    ax1.plot(x, dke_dki, color='b', marker='D', markersize=10,
            linestyle='', markeredgecolor = 'b')
    ax1.set_xlim([-0.5, nrun - 0.5])
    ax1.set_ylabel(r'$\Delta K_e/\Delta K_i$', color='b',
            fontdict=font, fontsize=24)
    ax1.tick_params(labelsize=20)
    for tl in ax1.get_yticklabels():
        tl.set_color('b')
    img_dir = '../img/img_dpp/'
    mkdir_p(img_dir)
    fig.savefig(img_dir + fname)
    plt.show()


if __name__ == "__main__":
    run_name = "mime25_beta002_noperturb"
    root_dir = '/net/scratch2/xiaocanli/mime25-sigma1-beta002-200-100-noperturb/'
    # run_name = "mime25_beta002"
    # root_dir = "/scratch3/xiaocanli/sigma1-mime25-beta001/"
    # run_name = "mime25_beta0007"
    # root_dir = '/net/scratch2/xiaocanli/mime25-guide0-beta0007-200-100/'
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    # plot_vx_frame(run_name, root_dir, pic_info, 'i')
    # plot_anisotropy(run_name, root_dir, pic_info, 'i', 62)
    # plot_compression_fields_single(run_name, root_dir, pic_info, 'i', 24)
    # for ct in range(110, 121):
    #     plot_compression_fields_single(run_name, root_dir, pic_info, 'e', ct)
    #     plt.close()
    # for ct in range(110, 121):
    #     plot_compression_fields_single(run_name, root_dir, pic_info, 'i', ct)
    #     plt.close()
    # cts = [110, 116]
    # plot_temperature_cut(run_name, root_dir, pic_info, cts)
    # plot_thermal_temperature_single(run_name, root_dir, pic_info, 115)
    # plot_compression_time_beta('e')
    # plot_compression_time_temp('e')
    # plot_by_time(run_name, root_dir, pic_info)
    # plot_energy_conversion_fraction()
    plot_energy_conversion_fraction_beta()
    # plot_nonthermal_fraction()
    # plot_jpara_dote_fraction()
    # plot_jdrifts_fraction('i')
