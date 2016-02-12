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

    if species == 'i':
        vmin = -0.5
        vmax = 0.5
    else:
        vmin = -0.8
        vmax = 0.8
    fig = plt.figure(figsize=[10,14])
    kwargs_plot = {"xstep":1, "zstep":1, "vmin":vmin, "vmax":vmax}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]

    ratio = pic_info.particle_interval / pic_info.fields_interval
    
    ct = (current_time+1) * ratio
    kwargs = {"current_time":ct, "xl":0, "xr":200, "zb":-50, "zt":50}
    x, z, Ay = read_2d_fields(pic_info, "../../data/Ay.gda", **kwargs) 
    nx, = x.shape
    nz, = z.shape
    zl = nz / 4
    zt = nz - zl

    nbands = 5
    data_sum = np.zeros((nbands, nx))
    data_acc = np.zeros((nbands, nx))
    
    nb = 0
    kwargs = {"current_time":current_time, "xl":0, "xr":200, "zb":-50, "zt":50}
    for iband in range(1, nbands+1):
        fname = "../../data1/jpara_dote_" + species + "_" + str(iband).zfill(2) + ".gda"
        x, z, jpara_dote = read_2d_fields(pic_info, fname, **kwargs) 
        fname = "../../data1/jperp_dote_" + species + "_" + str(iband).zfill(2) + ".gda"
        x, z, jperp_dote = read_2d_fields(pic_info, fname, **kwargs) 
        data = jpara_dote + jperp_dote

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
        cbar1.ax.tick_params(labelsize=20)
        if species == 'i':
            cbar1.set_ticks(np.arange(-0.4, 0.5, 0.2))
        else:
            cbar1.set_ticks(np.arange(-0.8, 0.9, 0.4))
        ys -= height + gap
        data_sum[nb, :] = np.sum(data, axis=0)
        data_acc[nb, :] = np.cumsum(data_sum[nb, :])
        nb += 1

    dv = pic_info.dx_di * pic_info.dz_di * pic_info.mime
    data_sum *= dv
    data_acc *= dv

    ys0 = 0.1
    height0 = ys + height - ys0
    w1, h1 = fig.get_size_inches()
    width0 = width * 0.98 - 0.05 / w1
    ax1 = fig.add_axes([xs, ys0, width0, height0])
    for i in range(nb):
        fname = 'Band' + str(i+1).zfill(2)
        # ax1.plot(x, data_sum[i, :], linewidth=2, label=fname)
        ax1.plot(x, data_acc[i, :], linewidth=2, label=fname)

    ax1.tick_params(labelsize=20)
    ax1.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
    ax1.set_ylabel(r'Accumulation', fontdict=font, fontsize=24)
    ax1.legend(loc=2, prop={'size':20}, ncol=1,
            shadow=False, fancybox=False, frameon=False)
    
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    if not os.path.isdir('../img/img_particle_drift/'):
        os.makedirs('../img/img_particle_drift/')
    fname = 'ene_gain_' + str(current_time).zfill(3) + '_' + species + '.jpg'
    fname = '../img/img_particle_drift/' + fname
    fig.savefig(fname, dpi=200)
    # plt.close()
    plt.show()


if __name__ == "__main__":
    pic_info = pic_information.get_pic_info('../../')
    ntp = pic_info.ntp
    # for ct in range(ntp):
    #     plot_particle_drift(pic_info, 'e', ct)
    plot_particle_drift(pic_info, 'e', 12)
