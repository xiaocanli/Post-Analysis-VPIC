"""
Functions and classes for 2D contour plots of fields.
"""
import collections
import errno    
import math
import os
import os.path
import pprint
import re
import stat
import struct
import sys
import itertools
from itertools import groupby
from os import listdir
from os.path import isfile, join

import functools
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from joblib import Parallel, delayed
from matplotlib import rc
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
from scipy import optimize
from scipy.fftpack import fft2, fftshift, ifft2
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.ndimage.filters import generic_filter as gf

import color_maps as cm
import colormap.colormaps as cmaps
import palettable
import pic_information
from contour_plots import plot_2d_contour, read_2d_fields
from distinguishable_colors import *
from energy_conversion import calc_jdotes_fraction_multi, read_data_from_json
from energy_conversion import plot_energy_evolution
from fields_plot import *
from particle_distribution import *
from pic_information import list_pic_info_dir
from runs_name_path import ApJ_long_paper_runs
from shell_functions import mkdir_p
from spectrum_fitting import *

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

font = {
    'family': 'serif',
    # 'color':'darkred',
    'color': 'black',
    'weight': 'normal',
    'size': 24,
}

colors = palettable.colorbrewer.qualitative.Set1_9.mpl_colors

# colors = palettable.colorbrewer.qualitative.Dark2_8.mpl_colors

def plot_number_density(run_name, root_dir, pic_info, species, ct):
    """Plot particle number density
    """
    kwargs = {"current_time": ct, "xl": 0, "xr": 1000, "zb": -250, "zt": 250}
    fname = root_dir + 'data/n' + species + '.gda'
    # fname = root_dir + 'data/jy.gda'
    x, z, nrho = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/Ay.gda'
    # x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
    nx, = x.shape
    nz, = z.shape
    xs, ys = 0.1, 0.15
    w1, h1 = 0.85, 0.8
    fig = plt.figure(figsize=[10, 5])

    ax1 = fig.add_axes([xs, ys, w1, h1])
    kwargs_plot = {"xstep": 2, "zstep": 2, "vmin": 0.0, "vmax": 10}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1 = plot_2d_contour(x, z, nrho, ax1, fig, is_cbar=0, **kwargs_plot)
    p1.set_cmap(plt.cm.get_cmap('jet'))
    # ax1.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep],
    #         colors='black', linewidths=0.5)
    ax1.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax1.tick_params(labelsize=16)
    t_wci = ct * pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(t_wci) + '/\Omega_{ci}$'
    ax1.text(0.02, 0.9, title, color='white', fontsize=20,
        bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='left', verticalalignment='center',
        transform=ax1.transAxes)

    nrho = 'n' + species
    fdir = '../img/' + nrho + '/'
    mkdir_p(fdir)
    fname = fdir + nrho + '_' + str(ct) + '.jpg'
    fig.savefig(fname, dpi=200)

    plt.show()
    # plt.close()



if __name__ == "__main__":
    cmdargs = sys.argv
    if (len(cmdargs) > 2):
        base_directory = cmdargs[1]
        run_name = cmdargs[2]
    else:
        base_directory = '/net/scratch2/guofan/sigma1-mime25-beta001-average/'
        run_name = 'sigma1-mime25-beta001-average'
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    ct = 50
    plot_number_density(run_name, base_directory, pic_info, 'e', ct)
    nt = 200
    cts = range(nt)
    def processInput(job_id):
        print job_id
        ct = job_id
        plot_number_density(run_name, base_directory, pic_info, 'e', ct)
    ncores = multiprocessing.cpu_count()
    # Parallel(n_jobs=ncores)(delayed(processInput)(ct) for ct in cts)
