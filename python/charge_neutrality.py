"""
Analysis procedures for compression related terms.
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
from scipy.interpolate import spline
import math
import os.path
import struct
import collections
import pic_information
from contour_plots import read_2d_fields, plot_2d_contour
import colormap.colormaps as cmaps

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

font = {'family' : 'serif',
        #'color'  : 'darkred',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 24,
        }

def plot_charge_neutrality(pic_info, current_time):
    """Plot compression related terms.

    Args:
        pic_info: namedtuple for the PIC simulation information.
        species: 'e' for electrons, 'i' for ions.
        current_time: current time frame.
    """
    kwargs = {"current_time":current_time, "xl":0, "xr":400, "zb":-100, "zt":100}
    x, z, ne = read_2d_fields(pic_info, "../../data/ne.gda", **kwargs) 
    x, z, ni = read_2d_fields(pic_info, "../../data/ni.gda", **kwargs) 
    x, z, Ay = read_2d_fields(pic_info, "../../data/Ay.gda", **kwargs) 
    nx, = x.shape
    nz, = z.shape
    width = 0.75
    height = 0.76
    xs = 0.12
    ys = 0.95 - height
    gap = 0.05
    fig = plt.figure(figsize=[10,4])
    ax1 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep":1, "zstep":1, "vmin":-1, "vmax":1}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, (ni-ne) / (ne+ni), ax1, fig, **kwargs_plot)
    # p1.set_cmap(cmaps.inferno)
    p1.set_cmap(plt.cm.seismic)
    ax1.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep], 
            colors='black', linewidths=0.5)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax1.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
    ax1.tick_params(labelsize=20)
    cbar1.ax.set_ylabel(r'$(n_i-n_e)/(n_i+n_e)$', fontdict=font, fontsize=24)
    # cbar1.set_ticks(np.arange(-0.1, 0.15, 0.1))
    cbar1.ax.tick_params(labelsize=20)
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    fname = '../img/q_' + str(current_time).zfill(3) + '.jpg'
    fig.savefig(fname, dpi=200)
    plt.show()

if __name__ == "__main__":
    pic_info = pic_information.get_pic_info('../../')
    ntp = pic_info.ntp
    plot_charge_neutrality(pic_info, 40)
