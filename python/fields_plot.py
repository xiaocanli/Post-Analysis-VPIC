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
import palettable
import sys

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

font = {'family' : 'serif',
        #'color'  : 'darkred',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 24,
        }

colors = palettable.colorbrewer.qualitative.Dark2_8.mpl_colors

class PlotMultiplePanels(object):
    def __init__(self, **kwargs):
        """Plot a figure with multiple panels
        """
        self.x = kwargs['x']
        self.z = kwargs['z']
        self.fdata = kwargs["fdata"]
        self.Ay = kwargs["Ay"]
        self.nx, = self.x.shape
        self.nz, = self.z.shape
        self.contour_color = kwargs["contour_color"]
        self.colormaps = kwargs["colormaps"]
        self.vmin = kwargs["vmin"]
        self.vmax = kwargs["vmax"]
        self.var_names = kwargs["var_names"]
        self.axis_pos = kwargs["axis_pos"]
        self.gaps = kwargs["gaps"]
        self.fig_sizes = kwargs["fig_sizes"]
        self.text_colors = kwargs["text_colors"]
        self.nxp = kwargs["nxp"]
        self.nzp = kwargs["nzp"]
        self.xstep = kwargs["xstep"]
        self.zstep = kwargs["zstep"]
        self.is_logs = kwargs["is_logs"]
        self.fname = kwargs["fname"]
        self.ct = kwargs["current_time"]
        self.fig_dir = kwargs["fig_dir"]
        # Whether to use a bottom panel for line plots
        if "bottom_panel" in kwargs:
            self.bottom_panel = kwargs["bottom_panel"]
            self.fdata_1d = kwargs["fdata_1d"]

        self.fig = plt.figure(figsize=self.fig_sizes)
        self.ax = []
        self.im = []
        self.co = []

        w1, h1 = self.axis_pos[2], self.axis_pos[3]
        for j in range(self.nzp):
            ys = self.axis_pos[1] - (h1 + self.gaps[1])*j
            for i in range(self.nxp):
                np = self.nxp*j + i
                xs = self.axis_pos[0] + (w1 + self.gaps[0])*i
                self.ax1 = self.fig.add_axes([xs, ys, w1, h1])
                self.ax.append(self.ax1)
                self.kwargs_plot = {"xstep":self.xstep, "zstep":self.zstep,
                        "is_log":self.is_logs[np], "vmin":self.vmin[np],
                        "vmax":self.vmax[np]}
                im1, cbar1 = plot_2d_contour(self.x, self.z, self.fdata[np],
                        self.ax1, self.fig, **self.kwargs_plot)
                self.im.append(im1)
                im1.set_cmap(plt.cm.get_cmap(self.colormaps[np]))
                co1 = self.ax1.contour(self.x[0:self.nx:self.xstep],
                        self.z[0:self.nz:self.zstep],
                        self.Ay[0:self.nz:self.zstep, 0:self.nx:self.xstep], 
                        colors=self.contour_color[np], linewidths=0.5)
                self.co.append(co1)
                self.ax1.tick_params(labelsize=16)
                self.ax1.tick_params(axis='x', labelbottom='off')
                self.ax1.autoscale(1,'both',1)
                self.ax1.text(0.1, 0.8, self.var_names[np],
                        color=self.text_colors[np], fontsize=24, 
                        bbox=dict(facecolor='none', alpha=1.0, edgecolor='none',
                            pad=10.0), horizontalalignment='center',
                        verticalalignment='center',
                        transform=self.ax1.transAxes)

        if "xlim" in kwargs:
            self.xlim = kwargs["xlim"]
            for ax1 in self.ax:
                ax1.set_xlim(self.xlim)
        if "zlim" in kwargs:
            self.zlim = kwargs["zlim"]
            for ax1 in self.ax:
                ax1.set_ylim(self.zlim)

        for j in range(self.nzp):
            np = self.nxp * j
            self.ax[np].set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
            for i in range(1, self.nxp):
                self.ax[np+i].set_ylabel('')
                self.ax[np+i].tick_params(axis='y', labelleft='off')

        if not self.bottom_panel:
            np = self.nxp * (self.nzp - 1)
            for i in range(self.nxp):
                self.ax[np+i].tick_params(axis='x', labelbottom='on')
                self.ax[np+i].set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)

        # Use bottom panel for line plots
        if self.bottom_panel:
            ys -= h1 + self.gaps[1]
            w2 = w1 * 0.98 - 0.05 / self.fig_sizes[0]
            xs = self.axis_pos[0]
            self.ax1d = self.fig.add_axes([xs, ys, w2, h1])
            self.ax1d.set_color_cycle(colors)
            self.p1d = []
            for j in range(self.nzp):
                for i in range(self.nxp):
                    np = self.nxp*j + i
                    p1, = self.ax1d.plot(self.x, self.fdata_1d[np], linewidth=2)
                    self.p1d.append(p1)
            self.ax1d.tick_params(labelsize=16)
            self.ax1d.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
            self.ax1d.set_ylabel(r'Accumulation', fontdict=font, fontsize=20)

        self.save_figures()

    def update_fields(self, ct, fdata, Ay):
        """Update the fields data.
        """
        self.ct = ct
        self.fdata = fdata
        self.Ay = Ay
        for j in range(self.nzp):
            for i in range(self.nxp):
                np = self.nxp*j + i
                self.im[np].set_data(self.fdata[np])
                for coll in self.co[np].collections:
                        coll.remove() 
                self.co[np] = self.ax[np].contour(self.x[0:self.nx:self.xstep],
                        self.z[0:self.nz:self.zstep],
                        self.Ay[0:self.nz:self.zstep, 0:self.nx:self.xstep], 
                        colors=self.contour_color[np], linewidths=0.5)
        self.fig.canvas.draw_idle()
        self.save_figures()

    def update_plot_1d(self, fdata_1d):
        """Update 1D plots
        """
        self.fdata_1d = fdata_1d
        for j in range(self.nzp):
            for i in range(self.nxp):
                np = self.nxp*j + i
                self.p1d[np].set_ydata(self.fdata_1d[np])
        self.ax1d.relim()
        self.ax1d.autoscale()

    def save_figures(self):
        fname = self.fig_dir + self.fname + '_' + str(self.ct).zfill(3) + '.jpg'
        self.fig.savefig(fname, dpi=200)


def plot_magnetic_fields(run_name, root_dir, pic_info):
    """Plot magnetic fields

    Args:
        run_name: the name of this run.
        root_dir: the root directory of this run.
        pic_info: PIC simulation information in a namedtuple.
    """
    ct = 0
    contour_color = ['k', 'k', 'k', 'k']
    vmin = [0, -1, -1, -1]
    vmax = [2, 1, 1, 1]
    xs, ys = 0.09, 0.60
    w1, h1 = 0.36, 0.36
    axis_pos = [xs, ys, w1, h1]
    gaps = [0.1, 0.06]
    fig_sizes = (10, 5)
    nxp, nzp = 2, 2
    var_names = [r'$B$', r'$B_x$', r'$B_y$', r'$B_z$']
    colormaps = ['gist_heat', 'seismic', 'seismic', 'seismic']
    text_colors = ['w', 'w', 'k', 'k']
    xstep, zstep = 2, 2
    is_logs = [False, False, False, False]
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    dir = '../img/img_bfields/'
    if not os.path.isdir(dir):
        os.makedirs(dir)
    fig_dir = dir + run_name + '/'
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    kwargs = {"current_time":ct, "xl":0, "xr":200, "zb":-50, "zt":50}
    fname1 = root_dir + 'data/absB.gda'
    x, z, absB = read_2d_fields(pic_info, fname1, **kwargs) 
    fname2 = root_dir + 'data/bx.gda'
    x, z, bx = read_2d_fields(pic_info, fname2, **kwargs) 
    fname3 = root_dir + 'data/by.gda'
    x, z, by = read_2d_fields(pic_info, fname3, **kwargs) 
    fname4 = root_dir + 'data/bz.gda'
    x, z, bz = read_2d_fields(pic_info, fname4, **kwargs) 
    fname5 = root_dir + 'data/Ay.gda'
    x, z, Ay = read_2d_fields(pic_info, fname5, **kwargs) 
    fdata = [absB, bx, by, bz]
    # Change with different runs
    b0 = pic_info.b0
    vmin = b0 * np.asarray(vmin)
    vmax = b0 * np.asarray(vmax)
    fname = 'bfields'
    kwargs_plots = {'current_time':ct, 'x':x, 'z':z, 'Ay':Ay,
            'fdata':fdata, 'contour_color':contour_color, 'colormaps':colormaps,
            'vmin':vmin, 'vmax':vmax, 'var_names':var_names, 'axis_pos':axis_pos,
            'gaps':gaps, 'fig_sizes':fig_sizes, 'text_colors':text_colors,
            'nxp':nxp, 'nzp':nzp, 'xstep':xstep, 'zstep':zstep, 'is_logs':is_logs,
            'fname':fname, 'fig_dir':fig_dir}
    bfields_plot = PlotMultiplePanels(**kwargs_plots)
    for ct in range(1, pic_info.ntf):
        kwargs["current_time"] = ct
        x, z, absB = read_2d_fields(pic_info, fname1, **kwargs) 
        x, z, bx = read_2d_fields(pic_info, fname2, **kwargs) 
        x, z, by = read_2d_fields(pic_info, fname3, **kwargs) 
        x, z, bz = read_2d_fields(pic_info, fname4, **kwargs) 
        x, z, Ay = read_2d_fields(pic_info, fname5, **kwargs) 
        fdata = [absB, bx, by, bz]
        bfields_plot.update_fields(ct, fdata, Ay)

    # plt.show()


def plot_electric_fields(run_name, root_dir, pic_info):
    """Plot magnetic fields

    Args:
        run_name: the name of this run.
        root_dir: the root directory of this run.
        pic_info: PIC simulation information in a namedtuple.
    """
    ct = 0
    contour_color = ['k', 'k', 'k']
    vmin = [-0.1, -0.1, -0.1]
    vmax = [0.1, 0.1, 0.1]
    xs, ys = 0.15, 0.70
    w1, h1 = 0.7, 0.2625
    axis_pos = [xs, ys, w1, h1]
    gaps = [0.1, 0.04]
    fig_sizes = (6, 8)
    nxp, nzp = 1, 3
    var_names = [r'$E_x$', r'$E_y$', r'$E_z$']
    colormaps = ['seismic', 'seismic', 'seismic']
    text_colors = ['k', 'k', 'k']
    xstep, zstep = 2, 2
    is_logs = [False, False, False]
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    dir = '../img/img_efields/'
    if not os.path.isdir(dir):
        os.makedirs(dir)
    fig_dir = dir + run_name + '/'
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    kwargs = {"current_time":ct, "xl":0, "xr":200, "zb":-50, "zt":50}
    fname2 = root_dir + 'data/ex.gda'
    x, z, ex = read_2d_fields(pic_info, fname2, **kwargs) 
    fname3 = root_dir + 'data/ey.gda'
    x, z, ey = read_2d_fields(pic_info, fname3, **kwargs) 
    fname4 = root_dir + 'data/ez.gda'
    x, z, ez = read_2d_fields(pic_info, fname4, **kwargs) 
    fname5 = root_dir + 'data/Ay.gda'
    x, z, Ay = read_2d_fields(pic_info, fname5, **kwargs) 
    fdata = [ex, ey, ez]
    fname = 'efields'
    # Change with different runs
    b0 = pic_info.b0
    wpe_wce = pic_info.dtwce / pic_info.dtwpe
    va = wpe_wce / math.sqrt(pic_info.mime)  # Alfven speed of inflow region
    vmin = np.asarray(vmin) * b0 * va / 0.2
    vmax = np.asarray(vmax) * b0 * va / 0.2
    kwargs_plots = {'current_time':ct, 'x':x, 'z':z, 'Ay':Ay,
            'fdata':fdata, 'contour_color':contour_color, 'colormaps':colormaps,
            'vmin':vmin, 'vmax':vmax, 'var_names':var_names, 'axis_pos':axis_pos,
            'gaps':gaps, 'fig_sizes':fig_sizes, 'text_colors':text_colors,
            'nxp':nxp, 'nzp':nzp, 'xstep':xstep, 'zstep':zstep, 'is_logs':is_logs,
            'fname':fname, 'fig_dir':fig_dir}
    efields_plot = PlotMultiplePanels(**kwargs_plots)
    for ct in range(1, pic_info.ntf):
        kwargs["current_time"] = ct
        x, z, ex = read_2d_fields(pic_info, fname2, **kwargs) 
        x, z, ey = read_2d_fields(pic_info, fname3, **kwargs) 
        x, z, ez = read_2d_fields(pic_info, fname4, **kwargs) 
        x, z, Ay = read_2d_fields(pic_info, fname5, **kwargs) 
        fdata = [ex, ey, ez]
        efields_plot.update_fields(ct, fdata, Ay)

    # plt.show()


def plot_current_densities(run_name, root_dir, pic_info):
    """Plot electric current densities

    Args:
        run_name: the name of this run.
        root_dir: the root directory of this run.
        pic_info: PIC simulation information in a namedtuple.
    """
    ct = 0
    contour_color = ['k', 'k', 'k', 'k']
    vmin = [0.01, -1, -1, -1]
    vmax = [2, 1, 1, 1]
    xs, ys = 0.09, 0.60
    w1, h1 = 0.36, 0.36
    axis_pos = [xs, ys, w1, h1]
    gaps = [0.1, 0.06]
    fig_sizes = (10, 5)
    nxp, nzp = 2, 2
    var_names = [r'$j$', r'$j_x$', r'$j_y$', r'$j_z$']
    colormaps = ['gist_heat', 'seismic', 'seismic', 'seismic']
    text_colors = ['w', 'k', 'k', 'k']
    xstep, zstep = 2, 2
    is_logs = [True, False, False, False]
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    dir = '../img/img_current_densities/'
    if not os.path.isdir(dir):
        os.makedirs(dir)
    fig_dir = dir + run_name + '/'
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    kwargs = {"current_time":ct, "xl":0, "xr":200, "zb":-50, "zt":50}
    fname1 = root_dir + 'data/absJ.gda'
    x, z, absJ = read_2d_fields(pic_info, fname1, **kwargs) 
    fname2 = root_dir + 'data/jx.gda'
    x, z, jx = read_2d_fields(pic_info, fname2, **kwargs) 
    fname3 = root_dir + 'data/jy.gda'
    x, z, jy = read_2d_fields(pic_info, fname3, **kwargs) 
    fname4 = root_dir + 'data/jz.gda'
    x, z, jz = read_2d_fields(pic_info, fname4, **kwargs) 
    fname5 = root_dir + 'data/Ay.gda'
    x, z, Ay = read_2d_fields(pic_info, fname5, **kwargs) 
    fdata = [absJ, jx, jy, jz]
    # Change with different runs
    b0 = pic_info.b0
    vmin = np.asarray(vmin) * b0
    vmax = np.asarray(vmax) * b0
    fname = 'jfields'
    kwargs_plots = {'current_time':ct, 'x':x, 'z':z, 'Ay':Ay,
            'fdata':fdata, 'contour_color':contour_color, 'colormaps':colormaps,
            'vmin':vmin, 'vmax':vmax, 'var_names':var_names, 'axis_pos':axis_pos,
            'gaps':gaps, 'fig_sizes':fig_sizes, 'text_colors':text_colors,
            'nxp':nxp, 'nzp':nzp, 'xstep':xstep, 'zstep':zstep, 'is_logs':is_logs,
            'fname':fname, 'fig_dir':fig_dir}
    jfields_plot = PlotMultiplePanels(**kwargs_plots)
    for ct in range(1, pic_info.ntf):
        kwargs["current_time"] = ct
        x, z, absJ = read_2d_fields(pic_info, fname1, **kwargs) 
        x, z, jx = read_2d_fields(pic_info, fname2, **kwargs) 
        x, z, jy = read_2d_fields(pic_info, fname3, **kwargs) 
        x, z, jz = read_2d_fields(pic_info, fname4, **kwargs) 
        x, z, Ay = read_2d_fields(pic_info, fname5, **kwargs) 
        fdata = [absJ, jx, jy, jz]
        jfields_plot.update_fields(ct, fdata, Ay)

    # plt.show()


def plot_number_densities(run_name, root_dir, pic_info):
    """Plot particle number densities

    Args:
        run_name: the name of this run.
        root_dir: the root directory of this run.
        pic_info: PIC simulation information in a namedtuple.
    """
    ct = 0
    contour_color = ['k', 'k']
    vmin = [0.1, 0.1]
    vmax = [10, 10]
    xs, ys = 0.18, 0.60
    w1, h1 = 0.72, 0.36
    axis_pos = [xs, ys, w1, h1]
    gaps = [0.1, 0.06]
    fig_sizes = (5, 5)
    nxp, nzp = 1, 2
    var_names = [r'$n_e$', r'$n_i$']
    colormaps = ['seismic', 'seismic']
    text_colors = ['k', 'k']
    xstep, zstep = 2, 2
    is_logs = [True, True]
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    dir = '../img/img_number_densities/'
    if not os.path.isdir(dir):
        os.makedirs(dir)
    fig_dir = dir + run_name + '/'
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    kwargs = {"current_time":ct, "xl":0, "xr":200, "zb":-50, "zt":50}
    fname1 = root_dir + 'data/ne.gda'
    x, z, ne = read_2d_fields(pic_info, fname1, **kwargs) 
    fname2 = root_dir + 'data/ni.gda'
    x, z, ni = read_2d_fields(pic_info, fname2, **kwargs) 
    fname3 = root_dir + 'data/Ay.gda'
    x, z, Ay = read_2d_fields(pic_info, fname3, **kwargs) 
    fdata = [ne, ni]
    fname = 'nrho'
    kwargs_plots = {'current_time':ct, 'x':x, 'z':z, 'Ay':Ay,
            'fdata':fdata, 'contour_color':contour_color, 'colormaps':colormaps,
            'vmin':vmin, 'vmax':vmax, 'var_names':var_names, 'axis_pos':axis_pos,
            'gaps':gaps, 'fig_sizes':fig_sizes, 'text_colors':text_colors,
            'nxp':nxp, 'nzp':nzp, 'xstep':xstep, 'zstep':zstep, 'is_logs':is_logs,
            'fname':fname, 'fig_dir':fig_dir}
    nfields_plot = PlotMultiplePanels(**kwargs_plots)
    for ct in range(1, pic_info.ntf):
        kwargs["current_time"] = ct
        x, z, ne = read_2d_fields(pic_info, fname1, **kwargs) 
        x, z, ni = read_2d_fields(pic_info, fname2, **kwargs) 
        x, z, Ay = read_2d_fields(pic_info, fname3, **kwargs) 
        fdata = [ne, ni]
        nfields_plot.update_fields(ct, fdata, Ay)

    # plt.show()


def plot_energy_band(run_name, root_dir, pic_info, species):
    """Plot particle fraction in different energy band

    Args:
        run_name: the name of this run.
        root_dir: the root directory of this run.
        pic_info: PIC simulation information in a namedtuple.
        species: particle species.
    """
    fname1 = root_dir + 'data/' + species + 'EB06.gda'
    if os.path.isfile(fname1):
        nbands = 10
    else:
        nbands = 5
    ct = 0
    contour_color = ['k'] * nbands
    vmin = [0.01] * nbands
    vmax = [1] * nbands
    if nbands == 5:
        xs, ys = 0.17, 0.81
        w1, h1 = 0.68, 0.165
        fig_sizes = (5, 10)
        nxp, nzp = 1, nbands
    else:
        xs, ys = 0.10, 0.81
        w1, h1 = 0.34, 0.165
        fig_sizes = (10, 10)
        nxp, nzp = 2, nbands/2
    axis_pos = [xs, ys, w1, h1]
    gaps = [0.1, 0.02]
    var_names = []
    for i in range(nbands):
        var_names.append(str(i+1).zfill(2))
    colormaps = ['seismic'] * nbands
    text_colors = ['k'] * nbands
    xstep, zstep = 2, 2
    is_logs = [True] * nbands
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    dir = '../img/img_' + species + 'EB/'
    if not os.path.isdir(dir):
        os.makedirs(dir)
    fig_dir = dir + run_name + '/'
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    kwargs = {"current_time":ct, "xl":0, "xr":200, "zb":-50, "zt":50}
    fdata = []
    fnames = []
    for i in range(1, nbands+1):
        fname1 = root_dir + 'data/' + species + 'EB' + str(i).zfill(2) + '.gda'
        fnames.append(fname1)
        x, z, data = read_2d_fields(pic_info, fname1, **kwargs) 
        fdata.append(data)
    fname2 = root_dir + 'data/Ay.gda'
    x, z, Ay = read_2d_fields(pic_info, fname2, **kwargs) 
    fname = species + 'EB'
    kwargs_plots = {'current_time':ct, 'x':x, 'z':z, 'Ay':Ay,
            'fdata':fdata, 'contour_color':contour_color, 'colormaps':colormaps,
            'vmin':vmin, 'vmax':vmax, 'var_names':var_names, 'axis_pos':axis_pos,
            'gaps':gaps, 'fig_sizes':fig_sizes, 'text_colors':text_colors,
            'nxp':nxp, 'nzp':nzp, 'xstep':xstep, 'zstep':zstep, 'is_logs':is_logs,
            'fname':fname, 'fig_dir':fig_dir}
    ebfields_plot = PlotMultiplePanels(**kwargs_plots)
    for ct in range(1, pic_info.ntf):
        kwargs["current_time"] = ct
        fdata = []
        for i in range(0, nbands):
            x, z, data = read_2d_fields(pic_info, fnames[i], **kwargs) 
            fdata.append(data)
        x, z, Ay = read_2d_fields(pic_info, fname2, **kwargs) 
        ebfields_plot.update_fields(ct, fdata, Ay)

    # plt.show()


def plot_pressure_tensor(run_name, root_dir, pic_info, species):
    """Plot pressure tensor

    Args:
        run_name: the name of this run.
        root_dir: the root directory of this run.
        pic_info: PIC simulation information in a namedtuple.
        species: particle species.
    """
    ct = 0
    contour_color = ['w', 'k', 'k', 'w', 'k', 'w']
    vmin = [0, -0.1, -0.1, 0, -0.1, 0]
    vmax = [0.5, 0.1, 0.1, 0.5, 0.1, 0.5]
    xs, ys = 0.10, 0.7
    w1, h1 = 0.364, 0.26
    fig_sizes = (10, 7)
    nxp, nzp = 2, 3
    axis_pos = [xs, ys, w1, h1]
    gaps = [0.1, 0.02]
    var_names = [r'$P_{xx}$', r'$P_{xy}$', r'$P_{xz}$', r'$P_{yy}$',
            r'$P_{yz}$', r'$P_{zz}$']
    colormaps = ['gist_heat', 'seismic', 'seismic', 'gist_heat',
            'seismic', 'gist_heat']
    text_colors = ['w', 'k', 'k', 'w', 'k', 'w']
    xstep, zstep = 2, 2
    is_logs = [False] * 6
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    dir = '../img/img_pressure_tensor/'
    if not os.path.isdir(dir):
        os.makedirs(dir)
    fig_dir = dir + run_name + '/'
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    kwargs = {"current_time":ct, "xl":0, "xr":200, "zb":-50, "zt":50}
    fnames = []
    fdata = []
    fname = root_dir + 'data/p' + species + '-xx.gda'
    fnames.append(fname)
    fname = root_dir + 'data/p' + species + '-xy.gda'
    fnames.append(fname)
    fname = root_dir + 'data/p' + species + '-xz.gda'
    fnames.append(fname)
    fname = root_dir + 'data/p' + species + '-yy.gda'
    fnames.append(fname)
    fname = root_dir + 'data/p' + species + '-yz.gda'
    fnames.append(fname)
    fname = root_dir + 'data/p' + species + '-zz.gda'
    fnames.append(fname)
    for fname in fnames:
        x, z, data = read_2d_fields(pic_info, fname, **kwargs) 
        fdata.append(data)
    fname2 = root_dir + 'data/Ay.gda'
    x, z, Ay = read_2d_fields(pic_info, fname2, **kwargs) 
    fname = 'p' + species
    kwargs_plots = {'current_time':ct, 'x':x, 'z':z, 'Ay':Ay,
            'fdata':fdata, 'contour_color':contour_color, 'colormaps':colormaps,
            'vmin':vmin, 'vmax':vmax, 'var_names':var_names, 'axis_pos':axis_pos,
            'gaps':gaps, 'fig_sizes':fig_sizes, 'text_colors':text_colors,
            'nxp':nxp, 'nzp':nzp, 'xstep':xstep, 'zstep':zstep, 'is_logs':is_logs,
            'fname':fname, 'fig_dir':fig_dir}
    pfields_plot = PlotMultiplePanels(**kwargs_plots)
    for ct in range(1, pic_info.ntf):
        kwargs["current_time"] = ct
        fdata = []
        for fname in fnames:
            x, z, data = read_2d_fields(pic_info, fname, **kwargs) 
            fdata.append(data)
        x, z, Ay = read_2d_fields(pic_info, fname2, **kwargs) 
        pfields_plot.update_fields(ct, fdata, Ay)

    # plt.show()


def plot_velocity_fields(run_name, root_dir, pic_info, species):
    """Plot magnetic fields

    Args:
        run_name: the name of this run.
        root_dir: the root directory of this run.
        pic_info: PIC simulation information in a namedtuple.
        species: particle species.
    """
    ct = 0
    contour_color = ['k', 'k', 'k']
    vmin = [-1.0, -1.0, -1.0]
    vmax = [1.0, 1.0, 1.0]
    xs, ys = 0.15, 0.70
    w1, h1 = 0.7, 0.2625
    axis_pos = [xs, ys, w1, h1]
    gaps = [0.1, 0.04]
    fig_sizes = (6, 8)
    nxp, nzp = 1, 3
    var_names = []
    var_name = r'$V_{' + species + 'x}/V_A$'
    var_names.append(var_name)
    var_name = r'$V_{' + species + 'y}/V_A$'
    var_names.append(var_name)
    var_name = r'$V_{' + species + 'z}/V_A$'
    var_names.append(var_name)
    colormaps = ['seismic', 'seismic', 'seismic']
    text_colors = ['k', 'k', 'k']
    xstep, zstep = 2, 2
    is_logs = [False, False, False]
    wpe_wce = pic_info.dtwce / pic_info.dtwpe
    va = wpe_wce / math.sqrt(pic_info.mime)  # Alfven speed of inflow region
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    dir = '../img/img_velocity/'
    if not os.path.isdir(dir):
        os.makedirs(dir)
    fig_dir = dir + run_name + '/'
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    kwargs = {"current_time":ct, "xl":0, "xr":200, "zb":-50, "zt":50}
    fname2 = root_dir + 'data/v' + species + 'x.gda'
    if os.path.isfile(fname2):
        fname3 = root_dir + 'data/v' + species + 'y.gda'
        fname4 = root_dir + 'data/v' + species + 'z.gda'
    else:
        fname2 = root_dir + 'data/u' + species + 'x.gda'
        fname3 = root_dir + 'data/u' + species + 'y.gda'
        fname4 = root_dir + 'data/u' + species + 'z.gda'
    x, z, vx = read_2d_fields(pic_info, fname2, **kwargs) 
    x, z, vy = read_2d_fields(pic_info, fname3, **kwargs) 
    x, z, vz = read_2d_fields(pic_info, fname4, **kwargs) 
    fname5 = root_dir + 'data/Ay.gda'
    x, z, Ay = read_2d_fields(pic_info, fname5, **kwargs) 
    fdata = [vx/va, vy/va, vz/va]
    fname = 'v' + species
    kwargs_plots = {'current_time':ct, 'x':x, 'z':z, 'Ay':Ay,
            'fdata':fdata, 'contour_color':contour_color, 'colormaps':colormaps,
            'vmin':vmin, 'vmax':vmax, 'var_names':var_names, 'axis_pos':axis_pos,
            'gaps':gaps, 'fig_sizes':fig_sizes, 'text_colors':text_colors,
            'nxp':nxp, 'nzp':nzp, 'xstep':xstep, 'zstep':zstep, 'is_logs':is_logs,
            'fname':fname, 'fig_dir':fig_dir}
    vfields_plot = PlotMultiplePanels(**kwargs_plots)
    for ct in range(1, pic_info.ntf):
        kwargs["current_time"] = ct
        x, z, vx = read_2d_fields(pic_info, fname2, **kwargs) 
        x, z, vy = read_2d_fields(pic_info, fname3, **kwargs) 
        x, z, vz = read_2d_fields(pic_info, fname4, **kwargs) 
        x, z, Ay = read_2d_fields(pic_info, fname5, **kwargs) 
        fdata = [vx/va, vy/va, vz/va]
        vfields_plot.update_fields(ct, fdata, Ay)

    # plt.show()


def plot_fields_single(run_name, root_dir, pic_info):
    """Plot fields for a single run

    Args:
        run_name: the name of this run.
        root_dir: the root directory of this run.
        pic_info: PIC simulation information in a namedtuple.
    """
    plot_magnetic_fields(run_name, root_dir, pic_info)
    plt.close()
    plot_electric_fields(run_name, root_dir, pic_info)
    plt.close()
    plot_current_densities(run_name, root_dir, pic_info)
    plt.close()
    plot_number_densities(run_name, root_dir, pic_info)
    plt.close()
    plot_energy_band(run_name, root_dir, pic_info, 'e')
    plt.close()
    plot_energy_band(run_name, root_dir, pic_info, 'i')
    plt.close()
    plot_pressure_tensor(run_name, root_dir, pic_info, 'e')
    plt.close()
    plot_pressure_tensor(run_name, root_dir, pic_info, 'i')
    plt.close()
    plot_velocity_fields(run_name, root_dir, pic_info, 'e')
    plt.close()
    plot_velocity_fields(run_name, root_dir, pic_info, 'i')
    plt.close()


def plot_fields_cmdline():
    """Plot fields for one with command line arguments
    """
    args = sys.argv
    run_name = args[1]
    root_dir = args[2]
    type_plot = int(args[3])
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    if type_plot == 1:
        plot_magnetic_fields(run_name, root_dir, pic_info)
    elif type_plot == 2:
        plot_electric_fields(run_name, root_dir, pic_info)
    elif type_plot == 3:
        plot_current_densities(run_name, root_dir, pic_info)
    elif type_plot == 4:
        plot_number_densities(run_name, root_dir, pic_info)
    elif type_plot == 5:
        plot_energy_band(run_name, root_dir, pic_info, 'e')
    elif type_plot == 6:
        plot_energy_band(run_name, root_dir, pic_info, 'i')
    elif type_plot == 7:
        plot_pressure_tensor(run_name, root_dir, pic_info, 'e')
    elif type_plot == 8:
        plot_pressure_tensor(run_name, root_dir, pic_info, 'i')
    elif type_plot == 9:
        plot_velocity_fields(run_name, root_dir, pic_info, 'e')
    elif type_plot == 10:
        plot_velocity_fields(run_name, root_dir, pic_info, 'i')

def plot_fields_multi():
    """Plot fields for multiple runs
    """
    base_dirs, run_names = ApJ_long_paper_runs()
    for root_dir, run_name in zip(base_dirs, run_names):
        picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        plot_fields_single(run_name, root_dir, pic_info)


def plot_jdote_fields(run_name, root_dir, pic_info, species):
    """Plot jdote due to different drift current

    Args:
        run_name: the name of this run.
        root_dir: the root directory of this run.
        pic_info: PIC simulation information in a namedtuple.
        species: particle species.
    """
    ct = 0
    nj = 7
    contour_color = ['k'] * nj
    vmin = [-1.0] * nj
    vmax = [1.0] * nj
    xs, ys = 0.11, 0.88
    w1, h1 = 0.8, 0.1
    axis_pos = [xs, ys, w1, h1]
    gaps = [0.1, 0.02]
    fig_sizes = (8, 16)
    nxp, nzp = 1, nj
    var_sym = ['c', 'g', 'm', 'p', 'a', '\parallel', '\perp']
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
    dir = '../img/img_jdotes/'
    if not os.path.isdir(dir):
        os.makedirs(dir)
    fig_dir = dir + run_name + '/'
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    kwargs = {"current_time":ct, "xl":0, "xr":200, "zb":-50, "zt":50}
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
    fname = root_dir + 'data1/jqnvpara_dote00_' + species + '.gda'
    fnames.append(fname)
    fname = root_dir + 'data1/jqnvperp_dote00_' + species + '.gda'
    fnames.append(fname)
    dv = pic_info.dx_di * pic_info.dz_di * pic_info.mime
    ng = 3
    kernel = np.ones((ng,ng)) / float(ng*ng)
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

    # plt.show()


def plot_jdotes_cmdline():
    """Plot jdote fields for one run with command line arguments
    """
    args = sys.argv
    run_name = args[1]
    root_dir = args[2]
    type_plot = int(args[3])
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    if type_plot == 1:
        plot_jdote_fields(run_name, root_dir, pic_info, 'e')
    elif type_plot == 2:
        plot_jdote_fields(run_name, root_dir, pic_info, 'i')


if __name__ == "__main__":
    # run_name = "mime25_beta002"
    # root_dir = "/scratch3/xiaocanli/sigma1-mime25-beta001/"
    # picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    # pic_info = read_data_from_json(picinfo_fname)
    # plot_jdote_fields(run_name, root_dir, pic_info, 'e')
    # plot_fields_multi()
    # plot_fields_cmdline()
    plot_jdotes_cmdline()
