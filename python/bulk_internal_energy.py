"""
Analysis procedures for bulk and internal energies.
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LogNorm
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
from contour_plots import read_2d_fields, plot_2d_contour
from energy_conversion import read_jdote_data

mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

font = {'family' : 'serif',
        #'color'  : 'darkred',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 24,
        }

def bulk_energy(pic_info, species, current_time):
    """Bulk energy and internal energy.

    Args:
        pic_info: namedtuple for the PIC simulation information.
        species: 'e' for electrons, 'i' for ions.
        current_time: current time frame.
    """
    kwargs = {"current_time":current_time, "xl":0, "xr":200, "zb":-15, "zt":15}
    fname = "../../data/u" + species + "x.gda"
    x, z, ux = read_2d_fields(pic_info, fname, **kwargs) 
    fname = "../../data/u" + species + "y.gda"
    x, z, uy = read_2d_fields(pic_info, fname, **kwargs) 
    fname = "../../data/u" + species + "z.gda"
    x, z, uz = read_2d_fields(pic_info, fname, **kwargs) 
    fname = "../../data/n" + species + ".gda"
    x, z, nrho = read_2d_fields(pic_info, fname, **kwargs) 
    fname = "../../data/p" + species + "-xx.gda"
    x, z, pxx = read_2d_fields(pic_info, fname, **kwargs) 
    fname = "../../data/p" + species + "-yy.gda"
    x, z, pyy = read_2d_fields(pic_info, fname, **kwargs) 
    fname = "../../data/p" + species + "-zz.gda"
    x, z, pzz = read_2d_fields(pic_info, fname, **kwargs) 
    x, z, Ay = read_2d_fields(pic_info, "../../data/Ay.gda", **kwargs) 

    if species == 'e':
        ptl_mass = 1.0
    else:
        ptl_mass = pic_info.mime

    internal_ene = (pxx + pyy + pzz) * 0.5
    bulk_ene = 0.5 * ptl_mass * nrho * (ux**2 + uy**2 + uz**2)

    nx, = x.shape
    nz, = z.shape
    width = 0.75
    height = 0.7
    xs = 0.12
    ys = 0.9 - height
    fig = plt.figure(figsize=[10,4])
    ax1 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep":1, "zstep":1, "is_log":True, "vmin":0.1, "vmax":10.0}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, bulk_ene/internal_ene,
            ax1, fig, **kwargs_plot)
    p1.set_cmap(plt.cm.seismic)
    ax1.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep], 
            colors='black', linewidths=0.5)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax1.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
    ax1.tick_params(labelsize=24)
    cbar1.ax.set_ylabel(r'$K/u$',
            fontdict=font, fontsize=24)
    cbar1.ax.tick_params(labelsize=24)
    
    t_wci = current_time*pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(t_wci) + '/\Omega_{ci}$'
    ax1.set_title(title, fontdict=font, fontsize=24)

    plt.show()
    # if not os.path.isdir('../img/'):
    #     os.makedirs('../img/')
    # if not os.path.isdir('../img/img_bulk_internal/'):
    #     os.makedirs('../img/img_bulk_internal/')
    # dir = '../img/img_bulk_internal/'
    # fname = 'bulk_internal' + str(current_time).zfill(3) + '_' + species + '.jpg'
    # fname = dir + fname
    # fig.savefig(fname)
    # plt.close()


def bulk_energy_change_rate(pic_info, species, current_time):
    """Bulk energy change rate.

    Args:
        pic_info: namedtuple for the PIC simulation information.
        species: 'e' for electrons, 'i' for ions.
        current_time: current time frame.
    """
    kwargs = {"current_time":current_time-1, "xl":0, "xr":200,
            "zb":-15, "zt":15}
    fname = "../../data/u" + species + "x.gda"
    x, z, ux = read_2d_fields(pic_info, fname, **kwargs) 
    fname = "../../data/u" + species + "y.gda"
    x, z, uy = read_2d_fields(pic_info, fname, **kwargs) 
    fname = "../../data/u" + species + "z.gda"
    x, z, uz = read_2d_fields(pic_info, fname, **kwargs) 
    fname = "../../data/n" + species + ".gda"
    x, z, nrho = read_2d_fields(pic_info, fname, **kwargs) 

    if species == 'e':
        ptl_mass = 1.0
    else:
        ptl_mass = pic_info.mime

    bulk_ene1 = 0.5 * ptl_mass * nrho * (ux**2 + uy**2 + uz**2)

    kwargs = {"current_time":current_time+1, "xl":0, "xr":200,
            "zb":-15, "zt":15}
    fname = "../../data/u" + species + "x.gda"
    x, z, ux = read_2d_fields(pic_info, fname, **kwargs) 
    fname = "../../data/u" + species + "y.gda"
    x, z, uy = read_2d_fields(pic_info, fname, **kwargs) 
    fname = "../../data/u" + species + "z.gda"
    x, z, uz = read_2d_fields(pic_info, fname, **kwargs) 
    fname = "../../data/n" + species + ".gda"
    x, z, nrho = read_2d_fields(pic_info, fname, **kwargs) 

    bulk_ene2 = 0.5 * ptl_mass * nrho * (ux**2 + uy**2 + uz**2)

    kwargs = {"current_time":current_time, "xl":0, "xr":200, "zb":-15, "zt":15}
    x, z, Ay = read_2d_fields(pic_info, "../../data/Ay.gda", **kwargs) 

    bulk_ene_rate = bulk_ene2 - bulk_ene1

    nx, = x.shape
    nz, = z.shape
    width = 0.75
    height = 0.7
    xs = 0.12
    ys = 0.9 - height
    fig = plt.figure(figsize=[10,4])
    ax1 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep":1, "zstep":1, "vmin":0.1, "vmax":-0.1}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, bulk_ene_rate,
            ax1, fig, **kwargs_plot)
    p1.set_cmap(plt.cm.seismic)
    ax1.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep], 
            colors='black', linewidths=0.5)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax1.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
    ax1.tick_params(labelsize=24)
    cbar1.ax.set_ylabel(r'$K/u$',
            fontdict=font, fontsize=24)
    cbar1.ax.tick_params(labelsize=24)
    
    t_wci = current_time*pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(t_wci) + '/\Omega_{ci}$'
    ax1.set_title(title, fontdict=font, fontsize=24)

    plt.show()
    # if not os.path.isdir('../img/'):
    #     os.makedirs('../img/')
    # if not os.path.isdir('../img/img_bulk_internal/'):
    #     os.makedirs('../img/img_bulk_internal/')
    # dir = '../img/img_bulk_internal/'
    # fname = 'bulk_internal' + str(current_time).zfill(3) + '_' + species + '.jpg'
    # fname = dir + fname
    # fig.savefig(fname)
    # plt.close()


if __name__ == "__main__":
    pic_info = pic_information.get_pic_info('../../')
    ntp = pic_info.ntp
    # bulk_energy(pic_info, 'e', 12)
    bulk_energy_change_rate(pic_info, 'e', 17)
    # for ct in range(pic_info.ntf):
    #     bulk_energy(pic_info, 'i', ct)
