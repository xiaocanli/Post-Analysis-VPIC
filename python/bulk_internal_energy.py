"""
Analysis procedures for bulk and internal energies.
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

def bulk_energy(pic_info, species, current_time):
    """Bulk energy and internal energy.

    Args:
        pic_info: namedtuple for the PIC simulation information.
        species: 'e' for electrons, 'i' for ions.
        current_time: current time frame.
    """
    kwargs = {"current_time":current_time, "xl":0, "xr":200, "zb":-20, "zt":20}
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
    # gama = 1.0 / np.sqrt(1.0 - (ux**2 + uy**2 + uz**2))
    # gama = np.sqrt(1.0 + ux**2 + uy**2 + uz**2)
    # bulk_ene2 = (gama - 1) * ptl_mass * nrho

    # print np.sum(bulk_ene), np.sum(bulk_ene2)

    nx, = x.shape
    nz, = z.shape
    width = 0.75
    height = 0.23
    xs = 0.12
    ys = 0.93 - height
    fig = plt.figure(figsize=[10,6])
    ax1 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep":1, "zstep":1, "is_log":True, "vmin":0.1, "vmax":10.0}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, bulk_ene/internal_ene,
            ax1, fig, **kwargs_plot)
    p1.set_cmap(plt.cm.seismic)
    ax1.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep], 
            colors='white', linewidths=0.5)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    # ax1.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
    ax1.tick_params(axis='x', labelbottom='off')
    ax1.tick_params(labelsize=20)
    cbar1.ax.set_ylabel(r'$K/u$', fontdict=font, fontsize=24)
    cbar1.ax.tick_params(labelsize=20)
    
    t_wci = current_time*pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(t_wci) + '/\Omega_{ci}$'
    ax1.set_title(title, fontdict=font, fontsize=24)

    ys -= height + 0.05
    ax2 = fig.add_axes([xs, ys, width, height])
    if species == 'e':
        vmax = 0.2
    else:
        vmax = 0.8
    kwargs_plot = {"xstep":1, "zstep":1, "vmin":0, "vmax": vmax}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p2, cbar2 = plot_2d_contour(x, z, bulk_ene,
            ax2, fig, **kwargs_plot)
    p2.set_cmap(plt.cm.nipy_spectral)
    ax2.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep], 
            colors='white', linewidths=0.5)
    ax2.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax2.tick_params(axis='x', labelbottom='off')
    ax2.tick_params(labelsize=20)
    cbar2.ax.set_ylabel(r'$K$', fontdict=font, fontsize=24)
    if species == 'e':
        cbar2.set_ticks(np.arange(0, 0.2, 0.04))
    else:
        cbar2.set_ticks(np.arange(0, 0.9, 0.2))
    cbar2.ax.tick_params(labelsize=20)
    
    ys -= height + 0.05
    ax3 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep":1, "zstep":1, "vmin":0, "vmax": 0.8}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p3, cbar3 = plot_2d_contour(x, z, internal_ene,
            ax3, fig, **kwargs_plot)
    p3.set_cmap(plt.cm.nipy_spectral)
    ax3.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep], 
            colors='white', linewidths=0.5)
    ax3.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax3.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
    ax3.tick_params(labelsize=20)
    cbar3.ax.set_ylabel(r'$u$', fontdict=font, fontsize=24)
    cbar3.set_ticks(np.arange(0, 0.9, 0.2))
    cbar3.ax.tick_params(labelsize=20)

    # plt.show()
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    if not os.path.isdir('../img/img_bulk_internal/'):
        os.makedirs('../img/img_bulk_internal/')
    dir = '../img/img_bulk_internal/'
    fname = 'bulk_internal' + str(current_time).zfill(3) + '_' + species + '.jpg'
    fname = dir + fname
    fig.savefig(fname, dpi=400)
    plt.close()


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


def plot_bulk_energy(pic_info, species):
    """Plot energy time evolution.

    Plot time evolution of bulk and internal energies. 
    """
    tfields = pic_info.tfields

    fname = '../data/bulk_internal_energy_' + species + '.dat'
    f = open(fname, 'r')
    content = np.genfromtxt(f)
    ux2 = content[:, 0]
    uy2 = content[:, 1]
    uz2 = content[:, 2]
    pxx = content[:, 3]
    pyy = content[:, 4]
    pzz = content[:, 5]
    f.close()

    ratio_ene = (ux2 + uy2 + uz2) / (pxx + pyy + pzz)
    fig = plt.figure(figsize=[7, 7])
    w1, h1 = 0.8, 0.25
    xs, ys = 0.15, 0.96 - h1
    ax1 = fig.add_axes([xs, ys, w1, h1])
    p1, = ax1.plot(tfields, ux2, linewidth=2, color='r', label=r'$x$')
    p2, = ax1.plot(tfields, uy2, linewidth=2, color='g', label=r'$y$')
    p3, = ax1.plot(tfields, uz2, linewidth=2, color='b', label=r'$z$')
    ax1.set_ylabel(r'Bulk', fontdict=font, fontsize=20)
    ax1.tick_params(axis='x', labelbottom='off')
    ax1.tick_params(labelsize=16)
    ax1.legend(loc=1, prop={'size':20}, ncol=1,
            shadow=False, fancybox=False, frameon=False)

    ys -= h1 + 0.05
    ax2 = fig.add_axes([xs, ys, w1, h1])
    p4, = ax2.plot(tfields, pxx, linewidth=2, color='r', label=r'$x$')
    p5, = ax2.plot(tfields, pyy, linewidth=2, color='g', label=r'$y$')
    p6, = ax2.plot(tfields, pzz, linewidth=2, color='b', label=r'$z$')
    ax2.legend(loc=4, prop={'size':20}, ncol=1,
            shadow=False, fancybox=False, frameon=False)
    ax2.set_ylabel(r'Internal', fontdict=font, fontsize=20)
    ax2.tick_params(axis='x', labelbottom='off')
    ax2.tick_params(labelsize=16)

    ys -= h1 + 0.05
    ax3 = fig.add_axes([xs, ys, w1, h1])
    p7, = ax3.plot(tfields, ratio_ene, linewidth=2, color='k')
    ax3.set_xlabel(r'$t\Omega_{ci}$', fontdict=font, fontsize=20)
    ax3.set_ylabel(r'Bulk/Internal', fontdict=font, fontsize=20)
    ax3.tick_params(labelsize=16)

    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    fname = '../img/bulk_internal_ene_' + species + '.eps'
    fig.savefig(fname)
    plt.show()


def check_energy(pic_info, species):
    """
    Check if the sum of bulk and internal energies is the same as the
    particle energy.

    """
    tenergy = pic_info.tenergy
    if species == 'e':
        kene = pic_info.kene_e
        label1 = r'$K_e$'
    else:
        kene = pic_info.kene_i
        label1 = r'$K_i$'
    tfields = pic_info.tfields

    # enorm = ene_bx[0]

    fname = '../data/bulk_internal_energy_' + species + '.dat'
    f = open(fname, 'r')
    content = np.genfromtxt(f)
    ux2 = content[:, 0]
    uy2 = content[:, 1]
    uz2 = content[:, 2]
    pxx = content[:, 3]
    pyy = content[:, 4]
    pzz = content[:, 5]
    f.close()

    ene_tot = ux2 + uy2 + uz2 + pxx + pyy + pzz
    fig = plt.figure(figsize=[7, 5])
    w1, h1 = 0.8, 0.8
    xs, ys = 0.15, 0.96 - h1
    ax1 = fig.add_axes([xs, ys, w1, h1])
    p1, = ax1.plot(tenergy, kene, linewidth=2, color='r',
            label=label1)
    p2, = ax1.plot(tfields, ene_tot, linewidth=2, color='b',
            label='Internal + Bulk')
    ax1.set_ylabel(r'Energy', fontdict=font, fontsize=20)
    ax1.tick_params(axis='x', labelbottom='off')
    ax1.tick_params(labelsize=16)
    ax1.legend(loc=4, prop={'size':20}, ncol=1,
            shadow=False, fancybox=False, frameon=False)

    plt.show()


def set_energy_density_bins(nbins):
    """Set logarithmic energy and number density bins.

    Args:
        nbins: number of bins.
    """
    emin = 1.0E-10
    emax = 1.0
    nmin = 0.01
    nmax = 10.0
    emin_log = math.log10(emin)
    emax_log = math.log10(emax)
    nmin_log = math.log10(nmin)
    nmax_log = math.log10(nmax)
    de_log = (emax_log - emin_log) / (nbins - 1)
    dn_log = (nmax_log - nmin_log) / (nbins - 1)
    ebins = np.zeros(nbins)
    nrho_bins = np.zeros(nbins)
    for i in range(nbins):
        ebins[i] = emin * 10**(de_log*i)
        nrho_bins[i] = nmin * 10**(dn_log*i)
    return (ebins, nrho_bins)


def bulk_energy_distribution(pic_info, species):
    """Get the distribution of bulk flow energy.

    Args:
        pic_info: namedtuple for the PIC simulation information.
        species: 'e' for electrons, 'i' for ions.
    """
    if species == 'e':
        ptl_mass = 1.0
    else:
        ptl_mass = pic_info.mime

    nbins = 100
    ebins, nrho_bins = set_energy_density_bins(nbins)

    ntf = pic_info.ntf
    ehist = np.zeros((ntf, nbins-1))
    erho_hist = np.zeros((ntf, nbins-1))
    nhist = np.zeros((ntf, nbins-1))
    for ct in range(ntf):
        kwargs = {"current_time":ct, "xl":0, "xr":200, "zb":-50, "zt":50}
        fname = "../../data/u" + species + "x.gda"
        x, z, ux = read_2d_fields(pic_info, fname, **kwargs) 
        fname = "../../data/u" + species + "y.gda"
        x, z, uy = read_2d_fields(pic_info, fname, **kwargs) 
        fname = "../../data/u" + species + "z.gda"
        x, z, uz = read_2d_fields(pic_info, fname, **kwargs) 
        fname = "../../data/n" + species + ".gda"
        x, z, nrho = read_2d_fields(pic_info, fname, **kwargs) 
        bene = 0.5 * ptl_mass * (ux*ux + uy*uy + uz*uz)
        bene_density = 0.5 * ptl_mass * nrho * (ux*ux + uy*uy + uz*uz)
        ehist[ct, :], bin_edges = np.histogram(bene, bins=ebins, density=True)
        erho_hist[ct, :], bin_edges = np.histogram(bene_density,
                bins=ebins, density=True)
        nhist[ct, :], bin_edges = np.histogram(nrho, bins=nrho_bins, density=True)
    f = open('../data/bulk_energy.dat', 'w')
    np.savetxt(f, ehist)
    f.close()
    f = open('../data/bulk_energy_density.dat', 'w')
    np.savetxt(f, erho_hist)
    f.close()
    f = open('../data/number_density.dat', 'w')
    np.savetxt(f, nhist)
    f.close()

def plot_bulk_energy_distribution(pic_info, species):
    """Get the distribution of bulk flow energy.

    Args:
        pic_info: namedtuple for the PIC simulation information.
        species: 'e' for electrons, 'i' for ions.
    """
    nbins = 100
    ebins, nrho_bins = set_energy_density_bins(nbins)

    ntf = pic_info.ntf
    ehist = np.zeros((ntf, nbins-1))
    erho_hist = np.zeros((ntf, nbins-1))
    nhist = np.zeros((ntf, nbins-1))
    f = open('../data/bulk_energy.dat', 'r')
    ehist = np.genfromtxt(f)
    f.close()
    f = open('../data/bulk_energy_density.dat', 'r')
    erho_hist = np.genfromtxt(f)
    f.close()
    f = open('../data/number_density.dat', 'r')
    nhist = np.genfromtxt(f)
    f.close()
    tfields = pic_info.tfields
    fig = plt.figure(figsize=[7, 5])
    ax = fig.add_axes([0.18, 0.15, 0.78, 0.8])
    tmin = np.min(tfields)
    tmax = np.max(tfields)
    emin = np.min(ebins)
    emax = np.max(ebins)
    # for ct in range(5, 10):
    #     p1 = ax.loglog(ebins[1:], erho_hist[ct, :], linewidth=2)
    # # p1 = ax.imshow(np.log10(erho_hist.transpose()), cmap=plt.cm.jet,
    # #         extent=[tmin, tmax, emin, emax],
    # #         aspect='auto', origin='lower',
    # #         interpolation='spline16')
    # ax.tick_params(labelsize=20)
    # plt.show()

    fig = plt.figure(figsize=[7, 5])
    ax = fig.add_axes([0.18, 0.15, 0.78, 0.8])
    ax.loglog(ebins[1:], ehist[12, :], color='black', linewidth=2)
    ax.set_xlabel(r'$E_b$', fontdict=font, fontsize=24)
    ax.set_ylabel(r'$f(E_b)$', fontdict=font, fontsize=24)
    ax.tick_params(labelsize=20)

    fig = plt.figure(figsize=[7, 5])
    ax = fig.add_axes([0.18, 0.15, 0.78, 0.8])
    ax.loglog(ebins[1:], erho_hist[12, :], color='black', linewidth=2)
    ax.set_xlabel(r'$E_b$', fontdict=font, fontsize=24)
    ax.set_ylabel(r'$f(E_b)$', fontdict=font, fontsize=24)
    ax.tick_params(labelsize=20)

    fig = plt.figure(figsize=[7, 5])
    ax1 = fig.add_axes([0.18, 0.15, 0.78, 0.8])
    ax1.loglog(nrho_bins[1:], nhist[12, :], color='black', linewidth=2)
    fname = r'$n_' + species + '$'
    ax1.set_xlabel(fname, fontdict=font, fontsize=24)
    fname = r'$f(n_' + species + ')$'
    ax1.set_ylabel(fname, fontdict=font, fontsize=24)
    ax1.tick_params(labelsize=20)

    plt.show()


if __name__ == "__main__":
    pic_info = pic_information.get_pic_info('../../')
    ntp = pic_info.ntp
    # bulk_energy(pic_info, 'i', 12)
    # bulk_energy_change_rate(pic_info, 'e', 17)
    for ct in range(pic_info.ntf):
        bulk_energy(pic_info, 'e', ct)
    for ct in range(pic_info.ntf):
        bulk_energy(pic_info, 'i', ct)
    # plot_bulk_energy(pic_info, 'i')
    # check_energy(pic_info, 'e')
    # plot_bulk_energy_distribution(pic_info, 'i')
