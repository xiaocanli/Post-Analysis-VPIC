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
import subprocess
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

def fields_movie_cmdline():
    """Make movies of fields using command line arguments
    """
    args = sys.argv
    run_name = args[1]
    root_dir = args[2]
    type_plot = int(args[3])
    mdir = '../img/movies/'
    if not os.path.isdir(mdir):
        os.makedirs(mdir)
    if type_plot == 1:
        fig_dir = root_dir + 'img_bfields/' + run_name + '/'
        fname = fig_dir + 'bfields_%3d' + '.jpg'
        movie_name = 'bfields_' + run_name + '.mp4'
    elif type_plot == 2:
        fig_dir = root_dir + 'img_efields/' + run_name + '/'
        fname = fig_dir + 'efields_%3d' + '.jpg'
        movie_name = 'efields_' + run_name + '.mp4'
    elif type_plot == 3:
        fig_dir = root_dir + 'img_current_densities/' + run_name + '/'
        fname = fig_dir + 'jfields_%3d' + '.jpg'
        movie_name = 'jfields_' + run_name + '.mp4'
    elif type_plot == 4:
        fig_dir = root_dir + 'img_eEB/' + run_name + '/'
        fname = fig_dir + 'eEB_%3d' + '.jpg'
        movie_name = 'eEB_' + run_name + '.mp4'
    elif type_plot == 5:
        fig_dir = root_dir + 'img_iEB/' + run_name + '/'
        fname = fig_dir + 'iEB_%3d' + '.jpg'
        movie_name = 'iEB_' + run_name + '.mp4'
    elif type_plot == 6:
        fig_dir = root_dir + 'img_number_densities/' + run_name + '/'
        fname = fig_dir + 'nrho_%3d' + '.jpg'
        movie_name = 'nrho_' + run_name + '.mp4'
    elif type_plot == 7:
        fig_dir = root_dir + 'img_pressure_tensor/' + run_name + '/'
        fname = fig_dir + 'pe_%3d' + '.jpg'
        movie_name = 'pe_' + run_name + '.mp4'
    elif type_plot == 8:
        fig_dir = root_dir + 'img_pressure_tensor/' + run_name + '/'
        fname = fig_dir + 'pi_%3d' + '.jpg'
        movie_name = 'pi_' + run_name + '.mp4'
    elif type_plot == 9:
        fig_dir = root_dir + 'img_velocity/' + run_name + '/'
        fname = fig_dir + 've_%3d' + '.jpg'
        movie_name = 've_' + run_name + '.mp4'
    elif type_plot == 10:
        fig_dir = root_dir + 'img_velocity/' + run_name + '/'
        fname = fig_dir + 'vi_%3d' + '.jpg'
        movie_name = 'vi_' + run_name + '.mp4'
    elif type_plot == 11:
        fig_dir = root_dir + 'img_jdotes/' + run_name + '/'
        fname = fig_dir + 'jdotes_e_%3d' + '.jpg'
        movie_name = 'jdotes_e_' + run_name + '.mp4'
    elif type_plot == 12:
        fig_dir = root_dir + 'img_jdotes/' + run_name + '/'
        fname = fig_dir + 'jdotes_i_%3d' + '.jpg'
        movie_name = 'jdotes_i_' + run_name + '.mp4'
    elif type_plot == 13:
        fig_dir = root_dir + 'img_temperature/' + run_name + '/'
        fname = fig_dir + 'temp_%3d' + '.jpg'
        movie_name = 'temp_' + run_name + '.mp4'
    elif type_plot == 14:
        fig_dir = root_dir + 'img_bulk_internal/' + run_name + '/'
        fname = fig_dir + 'bulk_internal_e_%3d' + '.jpg'
        movie_name = 'bulk_e_' + run_name + '.mp4'
    elif type_plot == 15:
        fig_dir = root_dir + 'img_bulk_internal/' + run_name + '/'
        fname = fig_dir + 'bulk_internal_i_%3d' + '.jpg'
        movie_name = 'bulk_i_' + run_name + '.mp4'
    elif type_plot == 16:
        fig_dir = root_dir + 'img_comp/' + run_name + '/'
        fname = fig_dir + 'comp_e_%3d' + '.jpg'
        movie_name = 'comp_e_' + run_name + '.mp4'
    elif type_plot == 17:
        fig_dir = root_dir + 'img_comp/' + run_name + '/'
        fname = fig_dir + 'comp_i_%3d' + '.jpg'
        movie_name = 'comp_i_' + run_name + '.mp4'

    cmd = 'ffmpeg -r 20 -f image2 -i ' + fname + \
            ' -f mp4 -q:v 0 -vcodec mpeg4 -r 20 ' + \
            mdir + movie_name
    print cmd
    p1 = subprocess.Popen([cmd], cwd='./', stdout=open('outfile.out', 'w'),
            stderr=subprocess.STDOUT, shell=True)
    p1.wait()


if __name__ == "__main__":
    fields_movie_cmdline()
