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
from energy_conversion import read_jdote_data

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

font = {'family' : 'serif',
        #'color'  : 'darkred',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 24,
        }

def plot_particle_drift(pic_info, species, current_time):
    """Plot compression related terms.

    Args:
        pic_info: namedtuple for the PIC simulation information.
        species: 'e' for electrons, 'i' for ions.
        current_time: current time frame.
    """
    print(current_time)

    width = 0.75
    height = 0.11
    xs = 0.12
    ys = 0.98 - height
    gap = 0.025

    vmin = -0.1
    vmax = 0.1
    fig = plt.figure(figsize=[10,14])
    kwargs_plot = {"xstep":1, "zstep":1, "vmin":vmin, "vmax":vmax}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    
    kwargs = {"current_time":current_time, "xl":0, "xr":200, "zb":-50, "zt":50}
    x, z, Ay = read_2d_fields(pic_info, "../../data/Ay.gda", **kwargs) 
    nx, = x.shape
    nz, = z.shape
    zl = nz / 4
    zt = nz - zl

    nbands = 10
    data_sum = np.zeros((nbands, nx))
    data_acc = np.zeros((nbands, nx))
    
    nb = 0
    for iband in range(1, nbands+1, 2):
        fname = "../../data1/jperp_dote_" + species + "_" + str(iband).zfill(2) + ".gda"
        x, z, data = read_2d_fields(pic_info, fname, **kwargs) 

        nk = 5
        kernel = np.ones((nk,nk)) / float(nk*nk)
        data = signal.convolve2d(data, kernel, mode='same')
        ax1 = fig.add_axes([xs, ys, width, height])
        p1, cbar1 = plot_2d_contour(x, z[zl:zt], data[zl:zt:zstep, 0:nx:xstep],
                ax1, fig, **kwargs_plot)
        p1.set_cmap(plt.cm.seismic)
        ax1.contour(x[0:nx:xstep], z[zl:zt:zstep], Ay[zl:zt:zstep, 0:nx:xstep], 
                colors='black', linewidths=0.5)
        ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
        ax1.tick_params(labelsize=20)
        ax1.tick_params(axis='x', labelbottom='off')
        # cbar1.set_ticks(np.arange(-0.004, 0.005, 0.002))
        cbar1.ax.tick_params(labelsize=20)
        ys -= height + gap
        data_sum[nb, :] = np.sum(data, axis=0)
        data_acc[nb, :] = np.cumsum(data_sum[nb, :])
        nb += 1

    ys0 = 0.1
    height0 = ys + height - ys0
    w1, h1 = fig.get_size_inches()
    width0 = width * 0.98 - 0.05 / w1
    ax1 = fig.add_axes([xs, ys0, width0, height0])
    for i in range(nb):
        fname = 'Band' + str(i+1).zfill(2)
        ax1.plot(x, data_acc[i, :], linewidth=2, label=fname)

    ax1.tick_params(labelsize=20)
    ax1.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
    ax1.legend(loc=1, prop={'size':20}, ncol=1,
            shadow=False, fancybox=False, frameon=False)
    
    plt.show()


if __name__ == "__main__":
    pic_info = pic_information.get_pic_info('../../')
    ntp = pic_info.ntp
    plot_particle_drift(pic_info, 'e', 7)
