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


def get_cmd_args():
    """Get command line arguments
    """
    default_run_name = 'sigma4E4_bg00_rad_vthe100_cool50'
    default_run_dir = ('/net/scratch2/xiaocanli/vpic_radiation/reconnection/' +
                       'grizzly/cooling_scaling_16000_8000/' +
                       'sigma4E4_bg00_rad_vthe100_cool50/')
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
    def processInput(job_id):
        print job_id
        tframe = job_id
        plot_magentic_field_one_frame(run_dir, run_name, tframe)
    cts = range(pic_info.ntf)
    ncores = multiprocessing.cpu_count()
    Parallel(n_jobs=ncores)(delayed(processInput)(ct) for ct in cts)
