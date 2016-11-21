"""
Analysis procedures for HERTS project
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LogNorm
from matplotlib import rc
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
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
import palettable
import sys
from shell_functions import mkdir_p
from plasma_params import calc_plasma_parameters
from scipy import signal
import multiprocessing
from joblib import Parallel, delayed
from particle_distribution import *
import itertools

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

font = {'family' : 'serif',
        #'color'  : 'darkred',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 24,
        }


def plot_nrho(run_name, root_dir, pic_info, ct,
              plasma_type='solar_wind', drange=[0.0, 1.0, 0.0, 1.0]):
    """Plot particle number densities

    Args:
        run_name: the name of this run.
        root_dir: the root directory of this run.
        pic_info: PIC simulation information in a namedtuple.
        ct: current time frame
        plasma_type: default is solar wind plasma
        drange: the relative range of the data to plot
    """
    params = calc_plasma_parameters(plasma_type)
    if plasma_type is 'lab':
        n0 = 1.0
    if plasma_type is 'lab_updated':
        n0 = 1.0
    else:
        n0 = params['ne']
    kwargs = {"current_time":ct, "xl":0, "xr":200, "zb":-50, "zt":50}
    fname1 = root_dir + 'data/ne.gda'
    x, z, ne_all = read_2d_fields(pic_info, fname1, **kwargs)
    fname2 = root_dir + 'data/ni.gda'
    x, z, ni_all = read_2d_fields(pic_info, fname2, **kwargs)
    nx, = x.shape
    nz, = z.shape
    xs = int(drange[0] * nx)
    xe = int(drange[1] * nx)
    zs = int(drange[2] * nz)
    ze = int(drange[3] * nz)
    ne = ne_all[zs:ze, xs:xe] * n0
    ni = ni_all[zs:ze, xs:xe] * n0
    if plasma_type is 'lab':
        vmin, vmax = 0, 6
    if plasma_type is 'lab_updated':
        vmin, vmax = 0, 6
    else:
        vmin, vmax = 0, 15
    xmax = np.max(x)
    di = math.sqrt(pic_info.mime)  # de is 1.0 in VPIC simulation
    if plasma_type is 'lab':
        # solar_wind plasma: m
        norm = params['di'] / 100 # di in params is in cm
        label_unit = 'm'
    if plasma_type is 'lab_updated':
        # solar_wind plasma: m
        norm = params['di'] / 100 # di in params is in cm
        label_unit = 'm'
    else:
        # solar_wind plasma: km
        norm = params['di'] / 1E5 # di in params is in cm
        label_unit = 'km'
    x = x[xs:xe] * norm
    z = z[zs:ze] * norm
    xmin = np.min(x)
    xmax = np.max(x)
    zmin = np.min(z)
    zmax = np.max(z)
    print xmax, zmax * 2
    print pic_info.dtwpe / params['wpe']
    fig = plt.figure(figsize=[10, 10])
    xs, ys = 0.12, 0.56
    w1, h1 = 0.8, 0.415
    ax = fig.add_axes([xs, ys, w1, h1])
    print 'Maximum and minimum electron density', np.max(ne), np.min(ne)
    p1 = ax.imshow(ne, cmap=plt.cm.rainbow,
            extent=[xmin, xmax, zmin, zmax],
            aspect='auto', origin='lower',
            # norm=LogNorm(vmin=0.1, vmax=30),
            vmin=vmin, vmax=vmax,
            interpolation='bicubic')
    ax.tick_params(labelsize=16)
    ax.tick_params(axis='x', labelleft='off')
    fname = r'$y$ / ' + label_unit
    ax.set_ylabel(fname, fontdict=font, fontsize=20)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cbar = fig.colorbar(p1, cax=cax)
    cbar.ax.tick_params(labelsize=16)
    if plasma_type is 'lab' or 'lab_updated':
        cbar.ax.set_ylabel(r'$n/n_0$', fontdict=font, fontsize=20)
    else:
        cbar.ax.set_ylabel('cm$^{-3}$', fontdict=font, fontsize=20)
    
    ys -= h1 + 0.05
    ax1 = fig.add_axes([xs, ys, w1, h1])
    print 'Maximum and minimum ion density', np.max(ni), np.min(ni)
    p2 = ax1.imshow(ni, cmap=plt.cm.rainbow,
            extent=[xmin, xmax, zmin, zmax],
            aspect='auto', origin='lower',
            vmin=vmin, vmax=vmax,
            interpolation='bicubic')
    ax1.tick_params(labelsize=16)
    fname = r'$x$ / ' + label_unit
    ax1.set_xlabel(fname, fontdict=font, fontsize=20)
    fname = r'$y$ / ' + label_unit
    ax1.set_ylabel(fname, fontdict=font, fontsize=20)

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cbar = fig.colorbar(p2, cax=cax)
    cbar.ax.tick_params(labelsize=16)
    if plasma_type is 'lab' or 'lab_updated':
        cbar.ax.set_ylabel(r'$n/n_0$', fontdict=font, fontsize=20)
    else:
        cbar.ax.set_ylabel('cm$^{-3}$', fontdict=font, fontsize=20)

    ax.text(0.1, 0.9, r'$n_e$',
            color='k', fontsize=24,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='center',
            transform = ax.transAxes)
    ax1.text(0.1, 0.9, r'$n_i$',
            color='k', fontsize=24,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='center',
            transform = ax1.transAxes)

    fdir = '../img/density/'
    mkdir_p(fdir)
    # fig.savefig('../img/ne_ni.jpg', dpi=300)
    fname = fdir + 'nei_' + str(ct) + '.jpg'
    fig.savefig(fname, dpi=200)

    plt.close()
    # plt.show()


def plot_emf(run_name, root_dir, pic_info):
    """Plot electromagnetic fields

    Args:
        run_name: the name of this run.
        root_dir: the root directory of this run.
        pic_info: PIC simulation information in a namedtuple.
    """
    ct = pic_info.ntf - 1
    # ct = 50
    n0 = 5.2
    kwargs = {"current_time":ct, "xl":0.005, "xr":0.008,
              "zb":-0.00075, "zt":0.00075}
    fname1 = root_dir + 'data/ex.gda'
    x, z, ex = read_2d_fields(pic_info, fname1, **kwargs)
    fname2 = root_dir + 'data/ey.gda'
    x, z, ey = read_2d_fields(pic_info, fname2, **kwargs)
    fname2 = root_dir + 'data/ez.gda'
    x, z, ez = read_2d_fields(pic_info, fname2, **kwargs)
    vmin, vmax = -0.1, 0.1
    xmax = np.max(x)
    norm = xmax * 1.16
    x /= norm
    z /= norm
    xmin = np.min(x)
    xmax = np.max(x)
    zmin = np.min(z)
    zmax = np.max(z)
    nx, = x.shape
    nz, = z.shape
    fig = plt.figure(figsize=[10, 14])
    xs, ys = 0.12, 0.70
    w1, h1 = 0.78, 0.27
    ax = fig.add_axes([xs, ys, w1, h1])
    p1 = ax.imshow(ex, cmap=plt.cm.seismic,
            extent=[xmin, xmax, zmin, zmax],
            aspect='auto', origin='lower',
            # norm=LogNorm(vmin=0.1, vmax=30),
            vmin=vmin, vmax=vmax,
            interpolation='bicubic')
    ax.tick_params(labelsize=16)
    ax.tick_params(axis='x', labelleft='off')
    ax.set_ylabel(r'$y$ / km', fontdict=font, fontsize=20)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cbar = fig.colorbar(p1, cax=cax)
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.set_ylabel('cm$^{-3}$', fontdict=font, fontsize=20)
    
    ys -= h1 + 0.05
    ax1 = fig.add_axes([xs, ys, w1, h1])
    p2 = ax1.imshow(ez, cmap=plt.cm.seismic,
            extent=[xmin, xmax, zmin, zmax],
            aspect='auto', origin='lower',
            vmin=vmin, vmax=vmax,
            interpolation='bicubic')
    ax1.tick_params(labelsize=16)
    ax1.set_xlabel(r'$x$ / km', fontdict=font, fontsize=20)
    ax1.set_ylabel(r'$y$ / km', fontdict=font, fontsize=20)

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cbar = fig.colorbar(p2, cax=cax)
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.set_ylabel('cm$^{-3}$', fontdict=font, fontsize=20)

    ys -= h1 + 0.05
    ax2 = fig.add_axes([xs, ys, w1, h1])
    p2 = ax2.imshow(ey, cmap=plt.cm.seismic,
            extent=[xmin, xmax, zmin, zmax],
            aspect='auto', origin='lower',
            vmin=vmin, vmax=vmax,
            interpolation='bicubic')
    ax2.tick_params(labelsize=16)
    ax2.set_xlabel(r'$x$ / km', fontdict=font, fontsize=20)
    ax2.set_ylabel(r'$y$ / km', fontdict=font, fontsize=20)

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cbar = fig.colorbar(p2, cax=cax)
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.set_ylabel('cm$^{-3}$', fontdict=font, fontsize=20)

    ax.text(0.1, 0.9, r'$E_x$',
            color='k', fontsize=24,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='center',
            transform = ax.transAxes)
    ax1.text(0.1, 0.9, r'$E_y$',
            color='k', fontsize=24,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='center',
            transform = ax1.transAxes)
    ax2.text(0.1, 0.9, r'$E_z$',
            color='k', fontsize=24,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='center',
            transform = ax2.transAxes)

    fig.savefig('../img/emf.jpg', dpi=300)

    plt.show()


def plot_force_2d(run_name, root_dir, pic_info):
    """Plot 2d force distributions

    Args:
        run_name: the name of this run.
        root_dir: the root directory of this run.
        pic_info: PIC simulation information in a namedtuple.
    """
    ct = pic_info.ntf - 1
    # ct = 50
    n0 = 5.2
    kwargs = {"current_time":ct, "xl":0, "xr":1, "zb":-0.5, "zt":0.5}
    fname1 = root_dir + 'data/ex.gda'
    x, z, ex = read_2d_fields(pic_info, fname1, **kwargs)
    fname2 = root_dir + 'data/ey.gda'
    x, z, ey = read_2d_fields(pic_info, fname2, **kwargs)
    fname2 = root_dir + 'data/ez.gda'
    x, z, ez = read_2d_fields(pic_info, fname2, **kwargs)
    fname2 = root_dir + 'data/ne.gda'
    x, z, ne = read_2d_fields(pic_info, fname2, **kwargs)
    fname2 = root_dir + 'data/ni.gda'
    x, z, ni = read_2d_fields(pic_info, fname2, **kwargs)

    # force_ex = -ne * ex
    # force_ey = -ne * ey
    # force_ez = -ne * ez
    # force_ex = ni * ex
    # force_ey = ni * ey
    # force_ez = ni * ez
    force_ex = (ni - ne) * ex
    force_ey = (ni - ne) * ey
    force_ez = (ni - ne) * ez

    force_cumx = np.cumsum(np.sum(force_ex, axis=0))
    force_cumy = np.cumsum(np.sum(force_ey, axis=0))
    force_cumz = np.cumsum(np.sum(force_ez, axis=0))

    vmin, vmax = -0.1, 0.1
    xmax = np.max(x)
    norm = xmax * 1.16
    x /= norm
    z /= norm
    xmin = np.min(x)
    xmax = np.max(x)
    zmin = np.min(z)
    zmax = np.max(z)
    nx, = x.shape
    nz, = z.shape
    fig = plt.figure(figsize=[10, 14])
    xs, ys = 0.12, 0.70
    w1, h1 = 0.78, 0.27
    ax = fig.add_axes([xs, ys, w1, h1])
    p1 = ax.imshow(force_ex, cmap=plt.cm.seismic,
            extent=[xmin, xmax, zmin, zmax],
            aspect='auto', origin='lower',
            # norm=LogNorm(vmin=0.1, vmax=30),
            vmin=vmin, vmax=vmax,
            interpolation='bicubic')
    ax.tick_params(labelsize=16)
    ax.tick_params(axis='x', labelleft='off')
    ax.set_ylabel(r'$y$ / km', fontdict=font, fontsize=20)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cbar = fig.colorbar(p1, cax=cax)
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.set_ylabel('cm$^{-3}$', fontdict=font, fontsize=20)
    
    ys -= h1 + 0.05
    ax1 = fig.add_axes([xs, ys, w1, h1])
    p2 = ax1.imshow(force_ez, cmap=plt.cm.seismic,
            extent=[xmin, xmax, zmin, zmax],
            aspect='auto', origin='lower',
            vmin=vmin, vmax=vmax,
            interpolation='bicubic')
    ax1.tick_params(labelsize=16)
    ax1.tick_params(axis='x', labelleft='off')
    ax1.set_ylabel(r'$y$ / km', fontdict=font, fontsize=20)

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cbar = fig.colorbar(p2, cax=cax)
    cbar.ax.tick_params(labelsize=16)
    cbar.ax.set_ylabel('cm$^{-3}$', fontdict=font, fontsize=20)

    ys -= h1 + 0.05
    width1, height1 = fig.get_size_inches()
    w1 = w1 * 0.97 - 0.05 / width1
    ax2 = fig.add_axes([xs, ys, w1, h1])
    # p2 = ax2.imshow(force_ey, cmap=plt.cm.seismic,
    #         extent=[xmin, xmax, zmin, zmax],
    #         aspect='auto', origin='lower',
    #         vmin=vmin, vmax=vmax,
    #         interpolation='bicubic')
    # ax2.tick_params(labelsize=16)
    # ax2.set_xlabel(r'$x$ / km', fontdict=font, fontsize=20)
    # ax2.set_ylabel(r'$y$ / km', fontdict=font, fontsize=20)

    # divider = make_axes_locatable(ax2)
    # cax = divider.append_axes("right", size="3%", pad=0.05)
    # cbar = fig.colorbar(p2, cax=cax)
    # cbar.ax.tick_params(labelsize=16)
    # cbar.ax.set_ylabel('cm$^{-3}$', fontdict=font, fontsize=20)
    ax2.plot(x, force_cumx, linewidth=2, color='r')
    ax2.plot(x, force_cumy, linewidth=2, color='g')
    ax2.plot(x, force_cumz, linewidth=2, color='b')
    ax2.set_xlim([xmin, xmax])
    ax2.tick_params(labelsize=16)
    ax2.set_xlabel(r'$x$ / km', fontdict=font, fontsize=20)

    ax.text(0.1, 0.9, r'$qE_x$',
            color='k', fontsize=24,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='center',
            transform = ax.transAxes)
    ax1.text(0.1, 0.9, r'$qE_y$',
            color='k', fontsize=24,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='center',
            transform = ax1.transAxes)
    ax2.text(0.1, 0.9, r'$qE_z$',
            color='k', fontsize=24,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='center',
            transform = ax2.transAxes)

    fig.savefig('../img/emf.jpg', dpi=300)

    plt.show()


def calc_force_charge_efield_single(job_id, drange):
    print job_id
    ct = job_id
    data_dir = '../data/force/'
    mkdir_p(data_dir)
    force_single = np.zeros(3)
    kwargs = {"current_time":ct, "xl":0, "xr":200, "zb":-50, "zt":50}
    fname1 = root_dir + 'data/ne.gda'
    x, z, ne_all = read_2d_fields(pic_info, fname1, **kwargs)
    fname2 = root_dir + 'data/ni.gda'
    x, z, ni_all = read_2d_fields(pic_info, fname2, **kwargs)
    fname = root_dir + 'data/ex.gda'
    x, z, ex_all = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/ey.gda'
    x, z, ey_all = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + 'data/ez.gda'
    x, z, ez_all = read_2d_fields(pic_info, fname, **kwargs)
    nx, = x.shape
    nz, = z.shape
    xs = int(drange[0] * nx)
    xe = int(drange[1] * nx)
    zs = int(drange[2] * nz)
    ze = int(drange[3] * nz)
    ne = ne_all[zs:ze, xs:xe]
    ni = ni_all[zs:ze, xs:xe]
    ex = ex_all[zs:ze, xs:xe]
    ey = ey_all[zs:ze, xs:xe]
    ez = ez_all[zs:ze, xs:xe]
    ntot = ni - ne
    force_single[0] = np.sum(ntot * ex)
    force_single[1] = np.sum(ntot * ey)
    force_single[2] = np.sum(ntot * ez)
    fname = data_dir + 'force_' + str(ct) + '.dat'
    force_single.tofile(fname)


def calc_force_charge_efield(root_dir, pic_info, drange=[0.0, 1.0, 0.0, 1.0]):
    """Calculate force using charge density and electric field
    """
    ntf = pic_info.ntf
    dx = pic_info.dx_di
    dz = pic_info.dz_di
    cts = range(ntf)
    ncores = multiprocessing.cpu_count()
    Parallel(n_jobs=ncores)(delayed(calc_force_charge_efield_single)(ct, drange) for ct in cts)
    force = np.zeros((3, ntf))
    data_dir = '../data/force/'
    for ct in cts:
        fname = data_dir + 'force_' + str(ct) + '.dat'
        force[:, ct] = np.fromfile(fname)

    mkdir_p('../data/')
    force.tofile('../data/force_partial.dat')


def plot_force(run_name, root_dir, pic_info, plasma_type='solar_wind',
               force_norm=1E9):
    """Plot force on tether

    Args:
        run_name: the name of this run.
        root_dir: the root directory of this run.
        pic_info: PIC simulation information in a namedtuple.
        plasma_type: default is solar wind plasma
        force_norm: the normalization of force.
                    Default: 1E9, so Newton is convected to nano-Newton
    """
    ntf = pic_info.ntf
    dx = pic_info.dx_di
    dz = pic_info.dz_di
    params = calc_plasma_parameters(plasma_type)
    c0 = 3.0E8   # m/s
    qe = 1.6E-19
    me = 1.0   # Mass normalization in VPIC
    ec = 1.0   # Charge normalization in VPIC
    c = 1.0    # Light speed in PIC
    wpe = 1.0  # Electron plasma frequency in VPIC 
    e0_real = c0 * params['B'] * 1E-4  # Real value
    wpe_wce = params['wpe'] / params['wce']
    wce = wpe / wpe_wce  # in simulation
    wci = wce / params['mi_me']
    wci_real = params['wci'] # real value
    wpe_real = params['wpe']
    b0 = me*c*wce/ec  # Asymptotic magnetic field strength in VPIC
    e0 = c * b0

    efield_norm = e0_real / e0
    force = np.fromfile('../data/force_partial.dat')
    force = force.reshape((3, ntf))
    di = params['di'] # in cm
    ni = params['ni'] # in #/cm^3
    norm = dx * dz * (di/100)**2 * (ni*1E6) * qe * efield_norm
    norm *= force_norm  # N -> nN
    force *= norm
    force_tot = np.sqrt(force[0,:]**2 + force[1,:]**2 + force[2,:]**2)

    dt = pic_info.dt_fields / wci_real
    dt *= 1E3 # second to millisecond
    t = np.arange(ntf) * dt

    fig = plt.figure(figsize=[7, 5])
    xs, ys = 0.13, 0.13
    w1, h1 = 0.8, 0.8
    ax = fig.add_axes([xs, ys, w1, h1])
    fc = -force[0]
    pf = np.polyfit(t, fc, 5)
    fc_fit = np.poly1d(pf)
    ax.plot(t, fc, color='k', linewidth=2)
    ax.plot(t, fc_fit(t), color='r', linewidth=3)
    ylims = ax.get_ylim()
    ax.set_xlim([t[0], t[-1]])
    ax.set_ylim([0.0, ylims[-1]-0.1])
    ax.tick_params(labelsize=16)
    ax.set_xlabel('t/ms', fontdict=font, fontsize=20)
    if force_norm == 1E9:
        fname = 'nN/m'
    elif force_norm == 1E6:
        fname = r'$\mu$N/m'
    elif force_norm == 1E3:
        fname = 'mN/m'
    else:
        fname = 'N/m'

    ax.set_ylabel(fname, fontdict=font, fontsize=20)
    fig.savefig('../img/force.jpg', dpi=300)
    plt.show()


def plot_vel(run_name, root_dir, pic_info, species, plasma_type='solar_wind'):
    """Plot particle velocities

    Args:
        run_name: the name of this run.
        root_dir: the root directory of this run.
        pic_info: PIC simulation information in a namedtuple.
        species: particle species
        plasma_type: default is solar wind plasma
    """
    ct = pic_info.ntf - 1
    params = calc_plasma_parameters(plasma_type)
    if plasma_type is 'lab':
        n0 = 1.0
    else:
        n0 = params['ne']
    kwargs = {"current_time":ct, "xl":0, "xr":200, "zb":-50, "zt":50}
    fname1 = root_dir + 'data/v' + species + 'x.gda'
    x, z, vx = read_2d_fields(pic_info, fname1, **kwargs)
    fname1 = root_dir + 'data/v' + species + 'z.gda'
    x, z, vz = read_2d_fields(pic_info, fname1, **kwargs)

    vmin, vmax = -0.01, 0.01
    if plasma_type is 'lab':
        # solar_wind plasma: m
        norm = params['di'] / 100 # di in params is in cm
        label_unit = 'm'
    if plasma_type is 'lab_updated':
        # solar_wind plasma: m
        norm = params['di'] / 100 # di in params is in cm
        label_unit = 'm'
        vmin, vmax = -0.001, 0.001
    else:
        # solar_wind plasma: km
        norm = params['di'] / 1E5 # di in params is in cm
        label_unit = 'km'
    x *= norm
    z *= norm
    ng = 3
    kernel = np.ones((ng, ng)) / float(ng*ng)
    vx = signal.convolve2d(vx, kernel, 'same')
    vz = signal.convolve2d(vz, kernel, 'same')
    if species is 'i':
        vmin /= 10
        vmax /= 10
    xmin = np.min(x)
    xmax = np.max(x)
    zmin = np.min(z)
    zmax = np.max(z)
    nx, = x.shape
    nz, = z.shape
    fig = plt.figure(figsize=[10, 10])
    xs, ys = 0.09, 0.56
    w1, h1 = 0.78, 0.39
    ax = fig.add_axes([xs, ys, w1, h1])
    print 'Maximum and minimum velocity', np.max(vx), np.min(vx)
    color_map = plt.cm.jet
    p1 = ax.imshow(vx, cmap=color_map,
            extent=[xmin, xmax, zmin, zmax],
            aspect='auto', origin='lower',
            # norm=LogNorm(vmin=0.1, vmax=30),
            vmin=vmin, vmax=vmax,
            interpolation='bicubic')
    ax.tick_params(labelsize=16)
    ax.tick_params(axis='x', labelleft='off')
    fname = r'$y$ / ' + label_unit
    ax.set_ylabel(fname, fontdict=font, fontsize=20)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cbar = fig.colorbar(p1, cax=cax)
    cbar.ax.tick_params(labelsize=16)
    fname = r'$v_{' + species + 'x}/c$'
    cbar.ax.set_ylabel(fname, fontdict=font, fontsize=20)

    ys -= h1 + 0.05
    ax1 = fig.add_axes([xs, ys, w1, h1])
    print 'Maximum and minimum velocity', np.max(vz), np.min(vz)
    color_map = plt.cm.seismic
    p2 = ax1.imshow(vz, cmap=color_map,
            extent=[xmin, xmax, zmin, zmax],
            aspect='auto', origin='lower',
            vmin=vmin, vmax=vmax,
            interpolation='bicubic')
    ax1.tick_params(labelsize=16)
    fname = r'$x$ / ' + label_unit
    ax1.set_xlabel(fname, fontdict=font, fontsize=20)
    fname = r'$y$ / ' + label_unit
    ax1.set_ylabel(fname, fontdict=font, fontsize=20)

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cbar = fig.colorbar(p2, cax=cax)
    cbar.ax.tick_params(labelsize=16)
    fname = r'$v_{' + species + 'z}/c$'
    cbar.ax.set_ylabel(fname, fontdict=font, fontsize=20)

    if species is 'i':
        den = 1.0
    else:
        den = 2.0

    # v = np.sqrt(vx*vx + vz*vz)
    # lw = 5 * v / v.max()
    # xx, zz = np.meshgrid(x, z)
    # strm = ax.streamplot(xx, zz, vx, vz, linewidth=lw, color='k',
    #         density=[den, den], arrowsize=4)
    # ax.set_xlim([xmin, xmax])
    # ax.set_ylim([zmin, zmax])
    # strm = ax1.streamplot(xx, zz, vx, vz, linewidth=lw, color='k',
    #         density=[den, den], arrowsize=4)
    # ax1.set_xlim([xmin, xmax])
    # ax1.set_ylim([zmin, zmax])

    fname = r'$v_{' + species + 'x}$'
    ax.text(0.1, 0.9, fname,
            color='k', fontsize=24,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='center',
            transform = ax.transAxes)
    fname = r'$v_{' + species + 'z}$'
    ax1.text(0.1, 0.9, fname,
            color='k', fontsize=24,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='center',
            transform = ax1.transAxes)

    fname = '../img/vx_vz_' + species + '.jpg'
    fig.savefig(fname, dpi=300)

    plt.show()


def plot_magnetic_field(run_name, root_dir, pic_info, plasma_type='solar_wind'):
    """Plot magnetic field

    Args:
        run_name: the name of this run.
        root_dir: the root directory of this run.
        pic_info: PIC simulation information in a namedtuple.
    """
    ct = pic_info.ntf - 1
    kwargs = {"current_time":ct, "xl":0, "xr":2, "zb":-1.0, "zt":1.0}
    fname2 = root_dir + 'data/bz.gda'
    x, z, by = read_2d_fields(pic_info, fname2, **kwargs)
    ng = 3
    kernel = np.ones((ng, ng)) / float(ng*ng)
    by = signal.convolve2d(by, kernel, 'same')
    xmax = np.max(x)
    if plasma_type is 'lab':
        # solar_wind plasma: m
        norm = params['di'] / 100 # di in params is in cm
        label_unit = 'm'
    else:
        # solar_wind plasma: km
        norm = params['di'] / 1E5 # di in params is in cm
        label_unit = 'km'
    x *= norm
    z *= norm
    xmin = np.min(x)
    xmax = np.max(x)
    zmin = np.min(z)
    zmax = np.max(z)
    nx, = x.shape
    nz, = z.shape
    fig = plt.figure(figsize=[10, 4])
    xs, ys = 0.12, 0.15
    w1, h1 = 0.8, 0.8
    vmin, vmax = 0.03, 0.13
    ax = fig.add_axes([xs, ys, w1, h1])
    print 'Max and min of by:', by.max(), by.min()
    p1 = ax.imshow(by, cmap=plt.cm.jet,
            extent=[xmin, xmax, zmin, zmax],
            aspect='auto', origin='lower',
            # norm=LogNorm(vmin=0.1, vmax=30),
            vmin=vmin, vmax=vmax,
            interpolation='bicubic')
    ax.tick_params(labelsize=16)
    fname = r'$x$ / ' + label_unit
    ax.set_xlabel(fname, fontdict=font, fontsize=20)
    fname = r'$y$ / ' + label_unit
    ax.set_ylabel(fname, fontdict=font, fontsize=20)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cbar = fig.colorbar(p1, cax=cax)
    cbar.ax.tick_params(labelsize=16)
    # cbar.ax.set_ylabel('cm$^{-3}$', fontdict=font, fontsize=20)
    
    ax.text(0.05, 0.9, r'$B_z$',
            color='k', fontsize=24,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='center',
            transform = ax.transAxes)

    fig.savefig('../img/by.jpg', dpi=300)

    plt.show()


def plot_vel_xyz(run_name, root_dir, pic_info, species, ct,
                 plasma_type='solar_wind', drange=[0.0, 1.0, 0.0, 1.0]):
    """Plot particle 3-component velocities

    Args:
        run_name: the name of this run.
        root_dir: the root directory of this run.
        pic_info: PIC simulation information in a namedtuple.
        species: particle species
        ct: current time frame
        plasma_type: default is solar wind plasma
        drange: relative data range
    """
    params = calc_plasma_parameters(plasma_type)
    if plasma_type is 'lab':
        n0 = 1.0
    else:
        n0 = params['ne']
    kwargs = {"current_time":ct, "xl":0, "xr":200, "zb":-50, "zt":50}
    fname1 = root_dir + 'data/v' + species + 'x.gda'
    x, z, vx_all = read_2d_fields(pic_info, fname1, **kwargs)
    fname1 = root_dir + 'data/v' + species + 'y.gda'
    x, z, vy_all = read_2d_fields(pic_info, fname1, **kwargs)
    fname1 = root_dir + 'data/v' + species + 'z.gda'
    x, z, vz_all = read_2d_fields(pic_info, fname1, **kwargs)
    nx, = x.shape
    nz, = z.shape

    xs = int(drange[0] * nx)
    xe = int(drange[1] * nx)
    zs = int(drange[2] * nz)
    ze = int(drange[3] * nz)
    vx = vx_all[zs:ze, xs:xe]
    vy = vy_all[zs:ze, xs:xe]
    vz = vz_all[zs:ze, xs:xe]

    vmin, vmax = -0.01, 0.01
    if plasma_type is 'lab':
        # solar_wind plasma: m
        norm = params['di'] / 100 # di in params is in cm
        label_unit = 'm'
    if plasma_type is 'lab_updated':
        # solar_wind plasma: m
        norm = params['di'] / 100 # di in params is in cm
        label_unit = 'm'
        vmin, vmax = -0.001, 0.001
    else:
        # solar_wind plasma: km
        norm = params['di'] / 1E5 # di in params is in cm
        label_unit = 'km'
    x = x[xs:xe] * norm
    z = z[zs:ze] * norm
    ng = 3
    kernel = np.ones((ng, ng)) / float(ng*ng)
    vx = signal.convolve2d(vx, kernel, 'same')
    vy = signal.convolve2d(vy, kernel, 'same')
    vz = signal.convolve2d(vz, kernel, 'same')
    if species is 'i':
        vmin /= 10
        vmax /= 10
    c0 = 3E5     # light speed in km/s
    vmin *= c0
    vmax *= c0
    vx *= c0
    vy *= c0
    vz *= c0
    xmin = np.min(x)
    xmax = np.max(x)
    zmin = np.min(z)
    zmax = np.max(z)
    fig = plt.figure(figsize=[10, 14])
    xs, ys = 0.12, 0.7
    w1, h1 = 0.78, 0.28
    gap = 0.03
    ax = fig.add_axes([xs, ys, w1, h1])
    print 'Maximum and minimum velocity', np.max(vx), np.min(vx)
    color_map = plt.cm.jet
    p1 = ax.imshow(vx, cmap=color_map,
            extent=[xmin, xmax, zmin, zmax],
            aspect='auto', origin='lower',
            # norm=LogNorm(vmin=0.1, vmax=30),
            vmin=vmin, vmax=vmax,
            interpolation='bicubic')
    ax.tick_params(labelsize=16)
    ax.tick_params(axis='x', labelleft='off')
    fname = r'$y$ / ' + label_unit
    ax.set_ylabel(fname, fontdict=font, fontsize=20)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cbar = fig.colorbar(p1, cax=cax)
    cbar.ax.tick_params(labelsize=16)
    fname = r'$v_{' + species + 'x}$ (km/s)'
    cbar.ax.set_ylabel(fname, fontdict=font, fontsize=20)

    ys -= h1 + gap
    ax1 = fig.add_axes([xs, ys, w1, h1])
    print 'Maximum and minimum velocity', np.max(vy), np.min(vy)
    color_map = plt.cm.seismic
    p2 = ax1.imshow(vz, cmap=color_map,
            extent=[xmin, xmax, zmin, zmax],
            aspect='auto', origin='lower',
            vmin=vmin, vmax=vmax,
            interpolation='bicubic')
    ax1.tick_params(labelsize=16)
    fname = r'$y$ / ' + label_unit
    ax1.set_ylabel(fname, fontdict=font, fontsize=20)

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cbar = fig.colorbar(p2, cax=cax)
    cbar.ax.tick_params(labelsize=16)
    fname = r'$v_{' + species + 'y}$ (km/s)'
    cbar.ax.set_ylabel(fname, fontdict=font, fontsize=20)

    ys -= h1 + gap
    ax2 = fig.add_axes([xs, ys, w1, h1])
    print 'Maximum and minimum velocity', np.max(vz), np.min(vz)
    color_map = plt.cm.seismic
    p3 = ax2.imshow(-vy, cmap=color_map,
            extent=[xmin, xmax, zmin, zmax],
            aspect='auto', origin='lower',
            vmin=vmin, vmax=vmax,
            interpolation='bicubic')
    ax2.tick_params(labelsize=16)
    fname = r'$x$ / ' + label_unit
    ax2.set_xlabel(fname, fontdict=font, fontsize=20)
    fname = r'$y$ / ' + label_unit
    ax2.set_ylabel(fname, fontdict=font, fontsize=20)

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="3%", pad=0.05)
    cbar = fig.colorbar(p3, cax=cax)
    cbar.ax.tick_params(labelsize=16)
    fname = r'$v_{' + species + 'z}$ (km/s)'
    cbar.ax.set_ylabel(fname, fontdict=font, fontsize=20)

    if species is 'i':
        den = 1.0
    else:
        den = 2.0

    v = np.sqrt(vx*vx + vz*vz)
    lw = 5 * v / v.max()
    xx, zz = np.meshgrid(x, z)
    strm = ax.streamplot(xx, zz, vx, vz, linewidth=1, color='k',
            density=[den, den], arrowsize=4)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([zmin, zmax])
    strm = ax1.streamplot(xx, zz, vx, vz, linewidth=1, color='k',
            density=[den, den], arrowsize=4)
    ax1.set_xlim([xmin, xmax])
    ax1.set_ylim([zmin, zmax])
    strm = ax2.streamplot(xx, zz, vx, vz, linewidth=1, color='k',
            density=[den, den], arrowsize=4)
    ax2.set_xlim([xmin, xmax])
    ax2.set_ylim([zmin, zmax])

    fname = r'$v_{' + species + 'x}$'
    ax.text(0.1, 0.9, fname,
            color='k', fontsize=24,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='center',
            transform = ax.transAxes)
    fname = r'$v_{' + species + 'y}$'
    ax1.text(0.1, 0.9, fname,
            color='k', fontsize=24,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='center',
            transform = ax1.transAxes)
    fname = r'$v_{' + species + 'z}$'
    ax2.text(0.1, 0.9, fname,
            color='k', fontsize=24,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='center',
            transform = ax2.transAxes)

    fname = '../img/vxyz_' + str(ct) + '_' + species + '.jpg'
    fig.savefig(fname, dpi=200)

    plt.close()
    # plt.show()


def get_particle_number(base_dir, pic_info, species, tindex):
    """Get the total particle number at a time frame
    """
    dir_name = base_dir + 'particle/T.' + str(tindex) + '/'
    fbase = dir_name + species + '.' + str(tindex) + '.'
    tx = pic_info.topology_x
    ty = pic_info.topology_y
    tz = pic_info.topology_z
    ntot = 0
    for (ix, iy, iz) in itertools.product(range(tx), range(ty), range(tz)):
        mpi_rank = ix + iy*tx + iz*tx*ty
        fname = fbase + str(mpi_rank)
        with open(fname, 'r') as fh:
            read_boilerplate(fh)
            v0, pheader, offset = read_particle_header(fh)
            ntot += pheader.dim
    print ntot


def read_particle_number(base_dir, pic_info):
    """Get the total particle number for all time frames
    """
    fbase = base_dir + 'rundata/particle_number.'
    tx = pic_info.topology_x
    ty = pic_info.topology_y
    tz = pic_info.topology_z
    ntot = [0, 0]
    fname = fbase + str(0)
    with open(fname, 'r') as fh:
        data = np.genfromtxt(fh)
        nt, sz = data.shape
    ntot = np.zeros((nt, 2))
    t = np.zeros(nt)
    for (ix, iy, iz) in itertools.product(range(tx), range(ty), range(tz)):
        mpi_rank = ix + iy*tx + iz*tx*ty
        fname = fbase + str(mpi_rank)
        with open(fname, 'r') as fh:
            data = np.genfromtxt(fh)
            ntot += data[:, 1:3]
    t = data[:, 0]
    dt = pic_info.dtwpe
    t *= dt

    plt.plot(t, ntot[:, 0] / ntot[:, 1], linewidth=2)
    plt.show()


if __name__ == "__main__":
    run_name = 'test'
    root_dir = '../../'
    pic_info = pic_information.get_pic_info(root_dir)
    # force_norm = 1E3
    # plasma_type = 'lab'
    force_norm = 1E9
    ct = pic_info.ntf - 1
    plasma_type = 'lab_updated'
    params = calc_plasma_parameters(plasma_type)
    cts = range(pic_info.ntf)
    drange = [0.2, 0.4, 0.4, 0.6]
    # drange = [0.0, 1.0, 0.0, 1.0]
    def processInput(job_id):
        print job_id
        # plot_vel_xyz(run_name, root_dir, pic_info, 'e', job_id, plasma_type, drange)
        plot_nrho(run_name, root_dir, pic_info, job_id, plasma_type, drange)
    ncores = multiprocessing.cpu_count()
    # Parallel(n_jobs=ncores)(delayed(processInput)(ct) for ct in cts)
    # plot_nrho(run_name, root_dir, pic_info, ct, plasma_type, drange)
    # plot_vel(run_name, root_dir, pic_info, 'e', plasma_type)
    # plot_vel_xyz(run_name, root_dir, pic_info, 'e', ct, plasma_type, drange)
    # plot_emf(run_name, root_dir, pic_info)
    # plot_magnetic_field(run_name, root_dir, pic_info, plasma_type)
    # plot_force_2d(run_name, root_dir, pic_info)
    # calc_force_charge_efield(root_dir, pic_info, drange)
    # plot_force(run_name, root_dir, pic_info, plasma_type, force_norm)
    # tindex = 1600700
    # get_particle_number(root_dir, pic_info, 'eparticle', tindex)
    # get_particle_number(root_dir, pic_info, 'hparticle', tindex)
    read_particle_number(root_dir, pic_info)
