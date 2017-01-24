"""
Particle number density of different energy band along a cut.
"""
import collections
import math
import os.path
import struct

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D

import pic_information
from contour_plots import plot_2d_contour, read_2d_fields

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

font = {'family' : 'serif',
        #'color'  : 'darkred',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 24,
        }


def plot_average_energy(pic_info, species, current_time):
    """Plot plasma beta and number density.

    Args:
        pic_info: namedtuple for the PIC simulation information.
        species: 'e' for electrons, 'i' for ions.
        current_time: current time frame.
    """
    kwargs = {"current_time":current_time, "xl":0, "xr":80, "zb":-20, "zt":20}
    fname = "../../data/n" + species + ".gda"
    x, z, num_rho = read_2d_fields(pic_info, fname, **kwargs) 
    fname = "../../data/p" + species + "-xx.gda"
    x, z, pxx = read_2d_fields(pic_info, fname, **kwargs) 
    fname = "../../data/p" + species + "-yy.gda"
    x, z, pyy = read_2d_fields(pic_info, fname, **kwargs) 
    fname = "../../data/p" + species + "-zz.gda"
    x, z, pzz = read_2d_fields(pic_info, fname, **kwargs) 
    fname = "../../data/u" + species + "x.gda"
    x, z, ux = read_2d_fields(pic_info, fname, **kwargs) 
    fname = "../../data/u" + species + "y.gda"
    x, z, uy = read_2d_fields(pic_info, fname, **kwargs) 
    fname = "../../data/u" + species + "z.gda"
    x, z, uz = read_2d_fields(pic_info, fname, **kwargs) 
    x, z, Ay = read_2d_fields(pic_info, "../../data/Ay.gda", **kwargs) 
    if species == 'e':
        ptl_mass = 1
    else:
        ptl_mass = pic_info.mime
    ene_avg = (pxx + pyy + pzz) / (2.0*num_rho) + \
              0.5*(ux*ux + uy*uy + uz*uz)*ptl_mass
    nx, = x.shape
    nz, = z.shape
    width = 0.78
    height = 0.78
    xs = 0.12
    ys = 0.92 - height
    fig = plt.figure(figsize=[10, 5])
    ax1 = fig.add_axes([xs, ys, width, height])
    if species == 'e':
        vmax = 0.2
    else:
        vmax = 1.4
    kwargs_plot = {"xstep":2, "zstep":2, "is_log":False,
            "vmin":0.0, "vmax":vmax}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, ene_avg, ax1, fig, **kwargs_plot)
    # p1.set_cmap(plt.cm.seismic)
    ax1.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep], 
            colors='black', linewidths=0.5)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax1.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
    ax1.tick_params(labelsize=20)
    fname = r'$E_{avg}$'
    cbar1.ax.set_ylabel(fname, fontdict=font, fontsize=24)
    # cbar1.set_ticks(np.arange(0.0, 0.22, 0.05))
    # cbar1.ax.set_yticklabels(['$0.2$', '$1.0$', '$5.0$'])
    cbar1.ax.tick_params(labelsize=20)
    
    t_wci = current_time*pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(t_wci) + '/\Omega_{ci}$'
    ax1.set_title(title, fontdict=font, fontsize=24)

    if not os.path.isdir('../img_num_rho/'):
        os.makedirs('../img_num_rho/')
    fname = '../img_num_rho/ene_avg_' + species + '_' + \
            str(current_time).zfill(3) + '.jpg'
    fig.savefig(fname, dpi=300)

    plt.show()
    # plt.close()


def cut_points(startp, endp, npoints, pic_info):
    """Get the points of the cut line.

    Args:
        startp: starting point.
        endp: ending point.
        npoints: number of the points.
        pic_info: namedtuple for the PIC simulation information.
    Returns:
        lcorner: the indices of the lower left of the cells in which
            the line points are.
        weights: the weight for 2D linear interpolation.
        coords: the coordinates of the points along the cut.
    """
    dx = (endp[0] - startp[0]) / (npoints-1)
    dz = (endp[1] - startp[1]) / (npoints-1)
    lcorner = np.zeros((2, npoints))
    weights = np.zeros((4, npoints))
    coords = np.zeros((2, npoints))

    for i in range(npoints):
        x = startp[0] + i * dx
        z = startp[1] + i * dz + pic_info.lz_di * 0.5
        ix = x / pic_info.dx_di
        iz = z / pic_info.dz_di
        lcorner[0, i] = math.floor(ix)
        lcorner[1, i] = math.floor(iz)
        deltax = ix - lcorner[0, i]
        deltaz = iz - lcorner[1, i]
        weights[0, i] = (1.0 - deltax) * (1.0 - deltaz)
        weights[1, i] = deltax * (1.0 - deltaz)
        weights[2, i] = deltax * deltaz
        weights[3, i] = (1.0 - deltax) * deltaz
        coords[0, i] = x
        coords[1, i] = z -  + pic_info.lz_di * 0.5

    return (coords, lcorner, weights)


def values_along_cut(fname, current_time, lcorner, weights):
    """Values along a straight cut.

    Args:
        fname: the filename of the data.
        current_time: current time frame.
        lcorner: the indices of the lower left of the cells in which
            the line points are.
        weights: the weight for 2D linear interpolation.
    Returns:
        dvalue: the values of the data along a cut.
    """
    kwargs = {"current_time":current_time, "xl":0, "xr":200, "zb":-50, "zt":50}
    x, z, data = read_2d_fields(pic_info, fname, **kwargs) 
    tmp, npoints = lcorner.shape
    dvalue = np.zeros(npoints)
    for i in range(npoints):
        dvalue[i] += data[lcorner[1, i], lcorner[0, i]] * weights[0, i]
        dvalue[i] += data[lcorner[1, i], lcorner[0, i]+1] * weights[1, i]
        dvalue[i] += data[lcorner[1, i]+1, lcorner[0, i]+1] * weights[2, i]
        dvalue[i] += data[lcorner[1, i]+1, lcorner[0, i]] * weights[3, i]

    return dvalue

def number_density_along_cut(current_time, coords, lcorner, weights):
    """Particle number density along a cut.

    Args:
        current_time: current time frame.
        coords: the coordinates of the points along the cut.
        lcorner: the indices of the lower left of the cells in which
            the line points are.
        weights: the weight for 2D linear interpolation.
    """
    xl, xr = 45, 80
    zb, zt = -10, 10
    kwargs = {"current_time":current_time, "xl":xl, "xr":xr, "zb":zb, "zt":zt}
    fname = '../../data/ne.gda'
    x, z, ne = read_2d_fields(pic_info, fname, **kwargs) 
    fname = '../../data/ni.gda'
    x, z, ni = read_2d_fields(pic_info, fname, **kwargs) 

    tmp, npoints = lcorner.shape
    dists = np.zeros(npoints)
    for i in range(npoints):
        dists[i] = math.sqrt((coords[1, i]-coords[1, 0])**2 + 
                (coords[0, i]-coords[0, 0])**2)
    ne_total = np.zeros(npoints)
    ni_total = np.zeros(npoints)

    xshift = math.floor(xl / pic_info.dx_di)
    zshift = math.floor((zb + pic_info.lz_di * 0.5) / pic_info.dz_di)
    lcorner[0, :] -= xshift
    lcorner[1, :] -= zshift

    nbands = 10
    # for iband in range(1, nbands+1):
    for iband in range(1, 2):
        fname = '../../data/eEB' + str(iband).zfill(2) + '.gda'
        x, z, eEB = read_2d_fields(pic_info, fname, **kwargs) 
        fname = '../../data/iEB' + str(iband).zfill(2) + '.gda'
        x, z, iEB = read_2d_fields(pic_info, fname, **kwargs) 
        x, z, Ay = read_2d_fields(pic_info, "../../data/Ay.gda", **kwargs) 
        nrho_band_e = eEB * ne
        nrho_band_i = iEB * ni

        ne_line = np.zeros(npoints)
        ni_line = np.zeros(npoints)
        for i in range(npoints):
            ne_line[i] += nrho_band_e[lcorner[1, i], lcorner[0, i]] * weights[0, i]
            ne_line[i] += nrho_band_e[lcorner[1, i], lcorner[0, i]+1] * weights[1, i]
            ne_line[i] += nrho_band_e[lcorner[1, i]+1, lcorner[0, i]+1] * weights[2, i]
            ne_line[i] += nrho_band_e[lcorner[1, i]+1, lcorner[0, i]] * weights[3, i]
            ni_line[i] += nrho_band_i[lcorner[1, i], lcorner[0, i]] * weights[0, i]
            ni_line[i] += nrho_band_i[lcorner[1, i], lcorner[0, i]+1] * weights[1, i]
            ni_line[i] += nrho_band_i[lcorner[1, i]+1, lcorner[0, i]+1] * weights[2, i]
            ni_line[i] += nrho_band_i[lcorner[1, i]+1, lcorner[0, i]] * weights[3, i]

        ne_total += ne_line
        ni_total += ni_line

        nx, = x.shape
        nz, = z.shape
        width = 0.78
        height = 0.78
        xs = 0.12
        ys = 0.92 - height
        fig = plt.figure(figsize=[10,5])
        ax1 = fig.add_axes([xs, ys, width, height])
        # vmax = math.floor(np.max(nrho_band_i) / 0.1 - 1.0) * 0.1
        kwargs_plot = {"xstep":1, "zstep":1, "is_log":False}
        xstep = kwargs_plot["xstep"]
        zstep = kwargs_plot["zstep"]
        ixs = 0
        p1, cbar1 = plot_2d_contour(x[ixs:nx], z, nrho_band_e[:, ixs:nx],
                ax1, fig, **kwargs_plot)
        # p1.set_cmap(plt.cm.seismic)
        ax1.contour(x[ixs:nx:xstep], z[0:nz:zstep],
                Ay[0:nz:zstep, ixs:nx:xstep], 
                colors='black', linewidths=0.5)
        ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
        ax1.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
        ax1.tick_params(labelsize=20)
        cbar1.ax.set_ylabel(r'$n_e$', fontdict=font, fontsize=24)
        # cbar1.set_ticks(np.arange(0.0, 0.2, 0.04))
        # cbar1.ax.set_yticklabels(['$0.2$', '$1.0$', '$5.0$'])
        cbar1.ax.tick_params(labelsize=20)
        p2 = ax1.plot(coords[0,:], coords[1,:], color='white', linewidth=2)
        ax1.text(coords[0, -1]+1.0, coords[1, -1]-1.0, 'B',
                color='white', fontsize=24, 
                bbox=dict(facecolor='none', alpha=1.0,
                    edgecolor='none', pad=10.0))
        ax1.text(coords[0, 0]-2.5, coords[1, 0]-1.0, 'A',
                color='white', fontsize=24, 
                bbox=dict(facecolor='none', alpha=1.0,
                    edgecolor='none', pad=10.0))
        fname = '../img_nrho_band/nrho_e_' + str(iband).zfill(2) + '.jpg'
        fig.savefig(fname, dpi=300)

        fig = plt.figure(figsize=[10,5])
        ax1 = fig.add_axes([xs, ys, width, height])
        kwargs_plot = {"xstep":1, "zstep":1, "is_log":False}
        xstep = kwargs_plot["xstep"]
        zstep = kwargs_plot["zstep"]
        p1, cbar1 = plot_2d_contour(x[ixs:nx], z, nrho_band_i[:, ixs:nx], ax1,
                fig, **kwargs_plot)
        # p1.set_cmap(plt.cm.seismic)
        ax1.contour(x[ixs:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, ixs:nx:xstep], 
                colors='black', linewidths=0.5)
        ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
        ax1.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
        ax1.tick_params(labelsize=20)
        cbar1.ax.set_ylabel(r'$n_i$', fontdict=font, fontsize=24)
        # cbar1.set_ticks(np.arange(0.0, 0.4, 0.1))
        # cbar1.ax.set_yticklabels(['$0.2$', '$1.0$', '$5.0$'])
        cbar1.ax.tick_params(labelsize=20)
        p2 = ax1.plot(coords[0,:], coords[1,:], color='white', linewidth=2)
        ax1.text(coords[0, -1]+1.0, coords[1, -1]-1.0,
                'B', color='white', fontsize=24, 
                bbox=dict(facecolor='none', alpha=1.0,
                    edgecolor='none', pad=10.0))
        ax1.text(coords[0, 0]-2.5, coords[1, 0]-1.0, 'A',
                color='white', fontsize=24, 
                bbox=dict(facecolor='none', alpha=1.0,
                    edgecolor='none', pad=10.0))
        fname = '../img_nrho_band/nrho_i_' + str(iband).zfill(2) + '.jpg'
        fig.savefig(fname, dpi=300)

        t_wci = current_time*pic_info.dt_fields
        title = r'$t = ' + "{:10.1f}".format(t_wci) + '/\Omega_{ci}$'
        ax1.set_title(title, fontdict=font, fontsize=24)

        fig = plt.figure(figsize=[7, 5])
        width = 0.69
        height = 0.8
        xs = 0.16
        ys = 0.95 - height
        ax = fig.add_axes([xs, ys, width, height])
        ax.plot(dists, ne_line, color='r', linewidth=2, label='Electron')
        ax.set_xlim([np.min(dists), np.max(dists)])
        ax.set_xlabel(r'Length/$d_i$', fontdict=font, fontsize=24)
        ax.set_ylabel(r'$n_e$', fontdict=font, fontsize=24, color='r')
        ax.text(0.0, -0.13, 'A', color='black', fontsize=24, 
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='center', verticalalignment='center',
                transform = ax.transAxes)
        ax.text(1.0, -0.13, 'B', color='black', fontsize=24, 
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='center', verticalalignment='center',
                transform = ax.transAxes)
        ax.set_xlim([np.min(dists), np.max(dists)])
        ax.tick_params(labelsize=20)
        for tl in ax.get_yticklabels():
            tl.set_color('r')
        ax1 = ax.twinx()
        ax1.plot(dists, ni_line, color='b', linewidth=2, label='Ion')
        ax1.set_ylabel(r'$n_i$', fontdict=font, fontsize=24, color='b')
        ax1.tick_params(labelsize=20)
        ax1.set_xlim([np.min(dists), np.max(dists)])
        for tl in ax1.get_yticklabels():
            tl.set_color('b')
        # ax.legend(loc=2, prop={'size':24}, ncol=1,
        #         shadow=False, fancybox=False, frameon=False)
        fname = '../img_nrho_band/nei_' + str(current_time).zfill(3) + \
                '_' + str(iband).zfill(2) + '.eps'
        fig.savefig(fname)

        # plt.show()
        plt.close('all')

    # p1 = plt.plot(ne_total/ni_total, linewidth=2)
    plt.show()

if __name__ == "__main__":
    pic_info = pic_information.get_pic_info('../../')
    ntp = pic_info.ntp
    vthe = pic_info.vthe
    startp = [52.0, 5.0]
    endp = [70.0, 0.0]
    coords, lcorner, weights = cut_points(startp, endp, 200, pic_info)
    number_density_along_cut(12, coords, lcorner, weights)
    # fname = '../../data/eEB05.gda'
    # dvalue = values_along_cut(fname, 80, lcorner, weights)
    # plot_average_energy(pic_info, 'e', 12)
    # for i in range(pic_info.ntf):
    #     plot_average_energy(pic_info, 'e', i)
    # for i in range(pic_info.ntf):
    #     plot_average_energy(pic_info, 'i', i)
