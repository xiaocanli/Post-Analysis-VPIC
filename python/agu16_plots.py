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
from energy_conversion import *
from energy_conversion import read_data_from_json
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


def plot_jdotes_fraction():
    """
    """
    picinfo_dir = '../data/pic_info/'
    mkdir_p(picinfo_dir)
    odir = '../img/ene_evolution/'
    mkdir_p(odir)
    nrun = 4
    ene_fraction = [[2.43, 1.63, 1.51, 1.34], [-1.70, -1.12, -0.58, -0.52]]
    labels = ['0.2', '0.07', '0.02', '0.007']
    fname = 'ene_fraction_beta.eps'

#     ene_fraction = [[1.51, 0.68, 0.41, 0.13], [-0.58, -0.06, 0.05, 0.02]]
#     labels = ['0.0', '0.2', '0.5', '1.0']
#     fname = 'ene_fraction_bg.eps'

    ene_fraction = np.asarray(ene_fraction)
    ene_fraction = ene_fraction.T
    x = np.arange(nrun)
    fig = plt.figure(figsize=[7, 4])
    xs, ys = 0.1, 0.18
    w1, h1 = 0.85, 0.8
    ax = fig.add_axes([xs, ys, w1, h1])
    ax.plot(x, ene_fraction[:, 0], color='k', marker='o', markersize=12,
            linestyle='', markeredgecolor = 'k', fillstyle='none',
            linewidth=2, markerfacecolor='None', markeredgewidth=2,
            label=r'$\boldsymbol{j}_c\cdot\boldsymbol{E}$')
    ax.plot(x, ene_fraction[:, 1], color='k', marker='D', markersize=12,
            linestyle='', markeredgecolor = 'k', fillstyle='none',
            linewidth=2, markerfacecolor='None', markeredgewidth=2,
            label=r'$\boldsymbol{j}_g\cdot\boldsymbol{E}$')
    ax.plot(x, ene_fraction[:, 0] + ene_fraction[:, 1], color='b', marker='d',
            markersize=12, linestyle='', markeredgecolor = 'b', fillstyle='none',
            linewidth=2, markerfacecolor='None', markeredgewidth=2,
            label='Sum')
    leg = ax.legend(loc=4, prop={'size':16}, ncol=2,
            shadow=False, fancybox=True, frameon=True)
    ax.plot()
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim([-0.5, nrun - 0.5])
    ax.plot([-0.5, nrun - 0.5], [1, 1], color='k', linestyle='--')
    ax.plot([-0.5, nrun - 0.5], [0, 0], color='k', linestyle='--')

    ax.set_ylim([-2.0, 2.6])
    ax.set_xlabel(r'$\beta_e$', color='k', fontdict=font, fontsize=24)

    # ax.set_ylim([-0.7, 1.6])
    # ax.set_xlabel(r'$B_g/B_0$', color='k', fontdict=font, fontsize=24)

    ax.tick_params(labelsize=20)
    img_dir = '../img/img_agu16/'
    mkdir_p(img_dir)
    fig.savefig(img_dir + fname)
    plt.show()


if __name__ == "__main__":
    # run_name = "mime25_beta002_noperturb"
    # root_dir = '/net/scratch2/xiaocanli/mime25-sigma1-beta002-200-100-noperturb/'
    # picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    # pic_info = read_data_from_json(picinfo_fname)
    plot_jdotes_fraction()
