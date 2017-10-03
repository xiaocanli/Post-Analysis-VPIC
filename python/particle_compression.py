"""
Analysis procedures for particle energy spectrum.
"""
import argparse
import collections
import gc
import math
import os
import os.path
import struct
import subprocess
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style as style
import multiprocessing
import numpy as np
from joblib import Parallel, delayed
from matplotlib import rc
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage.filters import median_filter, gaussian_filter

import palettable
from contour_plots import read_2d_fields
from dolointerpolation import MultilinearInterpolator
from energy_conversion import read_data_from_json
from particle_distribution import read_particle_data
from shell_functions import mkdir_p

style.use(['seaborn-white', 'seaborn-paper'])
# rc('font', **{'family': 'serif', 'serif': ["Times", "Palatino", "serif"]})
# rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc("font", family="Times New Roman")
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

colors_Set1_9 = palettable.colorbrewer.qualitative.Set1_9.mpl_colors
colors_Dark2_8 = palettable.colorbrewer.qualitative.Dark2_8.mpl_colors
colors_Paired_12 = palettable.colorbrewer.qualitative.Paired_12.mpl_colors
colors_Tableau_10 = palettable.tableau.Tableau_10.mpl_colors
colors_GreenOrange_6 = palettable.tableau.GreenOrange_6.mpl_colors
colors_Bold_10 = palettable.cartocolors.qualitative.Bold_10.mpl_colors

font = {
    'family': 'serif',
    #'color'  : 'darkred',
    'color': 'black',
    'weight': 'normal',
    'size': 24,
    }

def read_hydro_header(fh):
    """Read hydro file header

    Args:
        fh: file handler.
    """
    offset = 23  # the size of the boilerplate is 23
    tmp1 = np.memmap(
        fh, dtype='int32', mode='r', offset=offset, shape=(6), order='F')
    offset += 6 * 4
    tmp2 = np.memmap(
        fh, dtype='float32', mode='r', offset=offset, shape=(10), order='F')
    offset += 10 * 4
    tmp3 = np.memmap(
        fh, dtype='int32', mode='r', offset=offset, shape=(4), order='F')
    offset += 4 * 4
    v0header = collections.namedtuple("v0header", [
        "version", "type", "nt", "nx", "ny", "nz", "dt", "dx", "dy", "dz",
        "x0", "y0", "z0", "cvac", "eps0", "damp", "rank", "ndom", "spid",
        "spqm"
    ])
    v0 = v0header(
        version=tmp1[0],
        type=tmp1[1],
        nt=tmp1[2],
        nx=tmp1[3],
        ny=tmp1[4],
        nz=tmp1[5],
        dt=tmp2[0],
        dx=tmp2[1],
        dy=tmp2[2],
        dz=tmp2[3],
        x0=tmp2[4],
        y0=tmp2[5],
        z0=tmp2[6],
        cvac=tmp2[7],
        eps0=tmp2[8],
        damp=tmp2[9],
        rank=tmp3[0],
        ndom=tmp3[1],
        spid=tmp3[2],
        spqm=tmp3[3])
    header_hydro = collections.namedtuple("header_hydro",
                                          ["size", "ndim", "nc"])
    tmp4 = np.memmap(
        fh, dtype='int32', mode='r', offset=offset, shape=(5), order='F')
    hheader = header_hydro(size=tmp4[0], ndim=tmp4[1], nc=tmp4[2:])
    offset += 5 * 4
    return (v0, hheader, offset)


def read_hydro(fname, nx2, ny2, nz2, dsize):
    """
    """
    with open(fname, 'r') as fh:
        v0, hheader, offset = read_hydro_header(fh)
        fh.seek(offset, os.SEEK_SET)
        fdata = np.fromfile(fh, dtype=np.float32)

    sz, = fdata.shape
    nvar = sz / dsize
    fdata = fdata.reshape((nvar, nz2, ny2, nx2))
    return (v0, hheader, fdata)


def read_fields(fname, nx2, ny2, nz2, dsize):
    """
    """
    with open(fname, 'r') as fh:
        v0, hheader, offset = read_hydro_header(fh) # The same header as hydro
        fh.seek(offset, os.SEEK_SET)
        fdata = np.fromfile(fh, dtype=np.float32)

    sz, = fdata.shape
    nvar = sz / dsize
    fdata = fdata.reshape((nvar, nz2, ny2, nx2))
    return (v0, hheader, fdata)


def read_hydro_velocity_density(fname, nx2, ny2, nz2, dsize):
    """
    """
    offset = 123 # boilerplate and header size
    nvar = 4
    with open(fname, 'r') as fh:
        fh.seek(offset, os.SEEK_SET)
        fdata = np.fromfile(fh, dtype=np.float32, count=dsize*nvar)

    fdata = fdata.reshape((nvar, nz2, ny2, nx2))
    vx = fdata[0, :, 1, :]
    vy = fdata[1, :, 1, :]
    vz = fdata[2, :, 1, :]
    nrho = fdata[3, :, 1, :]
    inrho = div0(1, nrho)
    vx *= inrho
    vy *= inrho
    vz *= inrho
    nrho = np.abs(nrho)
    return (vx, vy, vz, nrho)


def read_hydro_four_velocity_density(fname, nx2, ny2, nz2, dsize):
    """
    """
    offset = 123 # boilerplate and header size
    nvar = 7
    with open(fname, 'r') as fh:
        fh.seek(offset, os.SEEK_SET)
        fdata = np.fromfile(fh, dtype=np.float32, count=dsize*nvar)

    fdata = fdata.reshape((nvar, nz2, ny2, nx2))
    nrho = np.abs(fdata[3, :, 1, :])
    ux = fdata[4, :, 1, :]
    uy = fdata[5, :, 1, :]
    uz = fdata[6, :, 1, :]
    inrho = div0(1, nrho)
    ux *= inrho
    uy *= inrho
    uz *= inrho
    return (ux, uy, uz, nrho)


def div0(a, b):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0]

    From: http://stackoverflow.com/a/35696047/2561161

    """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[~np.isfinite(c)] = 0  # -inf inf NaN
    return c


def interp_hydro_particle(pic_info, run_dir, tindex, rank):
    """
    """
    nx = pic_info.nx / pic_info.topology_x
    ny = pic_info.ny / pic_info.topology_y
    nz = pic_info.nz / pic_info.topology_z
    nx2 = nx + 2
    ny2 = ny + 2
    nz2 = nz + 2
    dsize = nx2 * ny2 * nz2
    mime = pic_info.mime
    smime = math.sqrt(mime)
    dx = pic_info.dx_di * smime
    dy = pic_info.dy_di * smime
    dz = pic_info.dz_di * smime

    hydro_dir = run_dir + 'hydro/T.' + str(tindex) + '/'
    fields_dir = run_dir + 'fields/T.' + str(tindex) + '/'
    particle_dir = run_dir + 'particle/T.' + str(tindex) + '/'
    ehydro_name = hydro_dir + 'ehydro.' + str(tindex)
    Hhydro_name = hydro_dir + 'Hhydro.' + str(tindex)
    eparticle_name = particle_dir + 'eparticle.' + str(tindex)
    hparticle_name = particle_dir + 'hparticle.' + str(tindex)
    field_name = fields_dir + 'fields.' + str(tindex)

    # fname = ehydro_name + '.' + str(rank)
    # (vex, vey, vez, ne) = read_hydro_velocity_density(fname, nx2, ny2, nz2, dsize)
    # vex = -vex
    # vey = -vey
    # vez = -vez
    # fname = Hhydro_name + '.' + str(rank)
    # (vix, viy, viz, ni) = read_hydro_velocity_density(fname, nx2, ny2, nz2, dsize)

    fname = eparticle_name + '.' + str(rank)
    (v0, pheader, ptl) = read_particle_data(fname)

    fname = field_name + '.' + str(rank)
    (v0, pheader, fields) = read_fields(fname, nx2, ny2, nz2, dsize)

    # inrho = div0(1.0 , ne + ni*mime)
    # vx = (vex*ne + vix*ni*mime) * inrho
    # vy = (vey*ne + viy*ni*mime) * inrho
    # vz = (vez*ne + viz*ni*mime) * inrho
    # divv = np.gradient(vx, dx, axis=1) + np.gradient(vz, dz, axis=0)

    dxp = ptl['dxyz'][:, 0]
    dzp = ptl['dxyz'][:, 2]
    icell = ptl['icell']
    nx = v0.nx + 2
    ny = v0.ny + 2
    nz = v0.nz + 2
    iz = icell // (nx * ny)
    # iy = (icell % (nx * ny)) // nx
    ix = icell % nx
    x_ptl = ((ix - 1.0) + (dxp + 1.0) * 0.5) * v0.dx + v0.x0
    z_ptl = ((iz - 1.0) + (dzp + 1.0) * 0.5) * v0.dz + v0.z0
    del icell, dxp, dzp, ix, iz
    nptl, = x_ptl.shape
    # x = np.linspace(v0.x0 - v0.dx, v0.x0 + v0.nx * v0.dx, nx)
    # z = np.linspace(v0.z0 - v0.dz, v0.z0 + v0.nz * v0.dz, nz)
    x = np.linspace(v0.x0, v0.x0 + v0.nx * v0.dx, nx - 1)
    z = np.linspace(v0.z0, v0.z0 + v0.nz * v0.dz, nz - 1)

    ux = ptl['u'][:, 0]
    uy = ptl['u'][:, 1]
    uz = ptl['u'][:, 2]
    gamma = np.sqrt(1 + np.sum(ptl['u']**2, axis=1))
    del ptl
    ene_ptl = gamma - 1.0
    igamma = 1.0 / gamma
    vxp = ux * igamma
    vyp = uy * igamma
    vzp = uz * igamma
    ex = fields[0, :, 1, :]
    ey = fields[1, :, 1, :]
    ez = fields[2, :, 1, :]
    bx = fields[3, :, 1, :]
    by = fields[4, :, 1, :]
    bz = fields[5, :, 1, :]

    del fields, ux, uy, uz, igamma

    f_ex = RectBivariateSpline(x, z, ex[1:, 1:].T)
    f_ey = RectBivariateSpline(x, z, ey[1:, 1:].T)
    f_ez = RectBivariateSpline(x, z, ez[1:, 1:].T)
    f_bx = RectBivariateSpline(x, z, bx[1:, 1:].T)
    f_by = RectBivariateSpline(x, z, by[1:, 1:].T)
    f_bz = RectBivariateSpline(x, z, bz[1:, 1:].T)
    ex_ptl = f_ex(x_ptl, z_ptl, grid=False)
    ey_ptl = f_ey(x_ptl, z_ptl, grid=False)
    ez_ptl = f_ez(x_ptl, z_ptl, grid=False)
    bx_ptl = f_bx(x_ptl, z_ptl, grid=False)
    by_ptl = f_by(x_ptl, z_ptl, grid=False)
    bz_ptl = f_bz(x_ptl, z_ptl, grid=False)

    del ex, ey, ez, bx, by, bz, x_ptl, z_ptl, x, z
    del f_ex, f_ey, f_ez, f_bx, f_by, f_bz

    ib2_ptl = 1.0 / (bx_ptl**2 + by_ptl**2 + bz_ptl**2)
    exb_ptl = ex_ptl * bx_ptl + ey_ptl * by_ptl + ez_ptl * bz_ptl
    ex_para_ptl = exb_ptl * bx_ptl * ib2_ptl
    ey_para_ptl = exb_ptl * by_ptl * ib2_ptl
    ez_para_ptl = exb_ptl * bz_ptl * ib2_ptl
    ex_perp_ptl = ex_ptl - ex_para_ptl
    ey_perp_ptl = ey_ptl - ey_para_ptl
    ez_perp_ptl = ez_ptl - ez_para_ptl

    del ex_ptl, ey_ptl, ez_ptl, bx_ptl, by_ptl, bz_ptl, ib2_ptl, exb_ptl

    de_para = -(vxp * ex_para_ptl + vyp * ey_para_ptl + vzp * ez_para_ptl)
    de_perp = -(vxp * ex_perp_ptl + vyp * ey_perp_ptl + vzp * ez_perp_ptl)
    de_tot = de_para + de_perp
    de_para_fraction = de_para / de_tot
    de_perp_fraction = 1.0 - de_para_fraction

    del vxp, vyp, vzp
    del ex_para_ptl, ey_para_ptl, ez_para_ptl
    del ex_perp_ptl, ey_perp_ptl, ez_perp_ptl

    # de_para = np.sort(de_para)
    # plt.plot(de_para)
    # # plt.ylim([-2, 2])

    # pdivv = -2.0/3.0 * (gamma - 1) * divv_ptl
    # # pdivv_fraction = -pdivv / de_tot
    print("Ratio of parallel heating: %d, %f" %
          (rank, np.sum(de_para)/np.sum(de_tot)))
    print("Parallel and perpendicular heating: %d, %f, %f" %
          (rank, np.sum(de_para), np.sum(de_perp)))
    # print("Energy change due to compression: %d, %f" % (rank, np.sum(pdivv)))
    print("Maximum and minimum energy gain: %12.5e, %12.5e, %12.5e, %12.5e" %
          (np.max(de_para), np.min(de_para), np.max(de_perp), np.min(de_perp)))

    nbins = 500
    drange = [[1, 1.1], [0, 1]]
    ebins = np.logspace(-3, 1, nbins)
    fbins = np.linspace(-2, 2, nbins*2)
    # fbins = np.linspace(-6E-2, 6E-2, nbins*2)
    df = fbins[1] - fbins[0]

    fdir = run_dir + 'data_ene/'
    mkdir_p(fdir)
    print("Maximum and minimum gamma: %12.5e, %12.5e" %
          (np.max(gamma-1), np.min(gamma-1)))
    # hist_perp, fbin_edges, fbin_edges = np.histogram2d(
    #         gamma-1, de_perp_fraction, bins=[ebins, fbins])
    # hist_para, fbin_edges, fbin_edges = np.histogram2d(
    #         gamma-1, de_para_fraction, bins=[ebins, fbins])

    hist_de_para, bin_edges = np.histogram(gamma-1, bins=ebins, weights=de_para)
    hist_de_perp, bin_edges = np.histogram(gamma-1, bins=ebins, weights=de_perp)
    # fname = fdir + 'hist_de_para.' + str(tindex) + '.' + str(rank)
    # hist_de_para.tofile(fname)
    # fname = fdir + 'hist_de_perp.' + str(tindex) + '.' + str(rank)
    # hist_de_perp.tofile(fname)

    hist_nptl, bin_edges = np.histogram(gamma-1, bins=ebins)
    hist_nptl = hist_nptl.astype(np.float)
    # fname = fdir + 'hist_nptl.' + str(tindex) + '.' + str(rank)
    # hist_nptl.tofile(fname)

    # plt.semilogx(ebins[:-1], hist_de_para, linewidth=2)
    # plt.semilogx(ebins[:-1], hist_de_perp, linewidth=2)

    # # nptl_gamma = np.sum(hist_xy, axis=1)
    # # hist_xy /= nptl_gamma[:, None]
    # xmin, xmax = np.min(ebins), np.max(ebins)
    # ymin, ymax = np.min(fbins), np.max(fbins)
    # plt.imshow(hist_perp.T, aspect='auto', cmap=plt.cm.Blues,
    #         origin='lower', extent=[xmin,xmax,ymin,ymax],
    #         norm=LogNorm(vmin=1, vmax=1E3),
    #         interpolation='bicubic')
    # plt.imshow(hist_para.T, aspect='auto', cmap=plt.cm.Reds,
    #         origin='lower', extent=[xmin,xmax,ymin,ymax],
    #         norm=LogNorm(vmin=1, vmax=1E3),
    #         interpolation='bicubic')
    # plt.show()

    # hist_para, bin_edges = np.histogram(de_para_fraction, bins=fbins,
    #                                     density=True)
    # hist_perp, bin_edges = np.histogram(de_perp_fraction, bins=fbins,
    #                                     density=True)
    # plt.plot(bin_edges[:-1], np.cumsum(hist_para)*df, linewidth=2, color='r')
    # plt.plot(bin_edges[:-1], np.cumsum(hist_perp)*df, linewidth=2, color='b')

    # hist_pdivv, bin_edges = np.histogram(pdivv_fraction, bins=fbins,
    #                                      density=True)
    # hist_xy, gbins, ebins = np.histogram2d(
    #         gamma, pdivv_fraction, bins=[nbins, 2*nbins], range=drange)
    # plt.semilogy(bin_edges[:-1], hist_pdivv, linewidth=2, color='b')
    # hist_gamma = np.sum(hist_xy*ebins[None, :-1], axis=1) / nptl_gamma
    # plt.plot(hist_gamma)

    del gamma, ene_ptl
    del de_para, de_perp, de_tot, de_para_fraction, de_perp_fraction
    del hist_de_para, hist_de_perp, bin_edges, hist_nptl


def calc_pdivv_from_fluid(pic_info, run_dir, tindex):
    """
    """
    nx = pic_info.nx / pic_info.topology_x
    ny = pic_info.ny / pic_info.topology_y
    nz = pic_info.nz / pic_info.topology_z
    nx2 = nx + 2
    ny2 = ny + 2
    nz2 = nz + 2
    dsize = nx2 * ny2 * nz2
    mime = pic_info.mime
    smime = math.sqrt(mime)
    dx = pic_info.dx_di * smime
    dy = pic_info.dy_di * smime
    dz = pic_info.dz_di * smime

    hydro_dir = run_dir + 'hydro/T.' + str(tindex) + '/'
    ehydro_name = hydro_dir + 'ehydro.' + str(tindex)
    Hhydro_name = hydro_dir + 'Hhydro.' + str(tindex)

    tx = pic_info.topology_x
    ty = pic_info.topology_y
    tz = pic_info.topology_z
    ncx = pic_info.nx / tx
    ncz = pic_info.nz / tz
    nprocs = tx * ty * tz
    pdivv = 0.0
    vxt = np.zeros((pic_info.nz, pic_info.nx))
    vyt = np.zeros((pic_info.nz, pic_info.nx))
    vzt = np.zeros((pic_info.nz, pic_info.nx))
    p2t = np.zeros((pic_info.nz, pic_info.nx))
    for rank in range(nprocs):
        print rank
        fname = ehydro_name + '.' + str(rank)
        # (vex, vey, vez, ne) = read_hydro_velocity_density(fname, nx2, ny2, nz2, dsize)
        # vex = -vex
        # vey = -vey
        # vez = -vez
        v0, pheader, ehydro = read_hydro(fname, nx2, ny2, nz2, dsize)
        vex = -ehydro[0, :, 1, :]
        vey = -ehydro[1, :, 1, :]
        vez = -ehydro[2, :, 1, :]
        ne = np.abs(ehydro[3, :, 1, :])
        uex = ehydro[4, :, 1, :]
        uey = ehydro[5, :, 1, :]
        uez = ehydro[6, :, 1, :]
        texx = ehydro[8, :, 1, :]
        teyy = ehydro[9, :, 1, :]
        tezz = ehydro[10, :, 1, :]
        ine = div0(1.0, ne)
        vex *= ine
        vey *= ine
        vez *= ine
        p2 = (texx + teyy + tezz - vex*uex - vey*uey - vez*uez) / 3.0
        del ehydro, ine, texx, teyy, tezz

        fname = Hhydro_name + '.' + str(rank)
        # (vix, viy, viz, ni) = read_hydro_velocity_density(fname, nx2, ny2, nz2, dsize)
        v0, pheader, hhydro = read_hydro(fname, nx2, ny2, nz2, dsize)
        vix = hhydro[0, :, 1, :]
        viy = hhydro[1, :, 1, :]
        viz = hhydro[2, :, 1, :]
        ni = np.abs(hhydro[3, :, 1, :])
        uix = hhydro[4, :, 1, :]
        uiy = hhydro[5, :, 1, :]
        uiz = hhydro[6, :, 1, :]
        ini = div0(1.0, ni)
        vix *= ini
        viy *= ini
        viz *= ini
        del hhydro, ini

        inrho = div0(1.0, ne + ni*mime)
        vx = (vex*ne + vix*ni*mime) * inrho
        vy = (vey*ne + viy*ni*mime) * inrho
        vz = (vez*ne + viz*ni*mime) * inrho
        iz = rank // tx
        ix = rank % tx
        vxt[ncz*iz:ncz*(iz+1), ncx*ix:ncx*(ix+1)] = vx[1:-1, 1:-1]
        vyt[ncz*iz:ncz*(iz+1), ncx*ix:ncx*(ix+1)] = vy[1:-1, 1:-1]
        vzt[ncz*iz:ncz*(iz+1), ncx*ix:ncx*(ix+1)] = vz[1:-1, 1:-1]
        p2t[ncz*iz:ncz*(iz+1), ncx*ix:ncx*(ix+1)] = p2[1:-1, 1:-1]
        divv = np.gradient(vx[1:, 1:], v0.dx, axis=1) + \
               np.gradient(vz[1:, 1:], v0.dz, axis=0)
        pdivv += np.sum(-p2[1:-1, 1:-1] * divv[:-1, :-1] * v0.dx * v0.dz)

    # plt.imshow(vxt)
    # plt.show()
    divv = np.gradient(vxt, v0.dx, axis=1) + np.gradient(vzt, v0.dz, axis=0)
    pdivv2 = np.sum(-p2t * divv * v0.dx * v0.dz)

    print pdivv, pdivv2


def interp_particle_compression(pic_info, run_dir, tindex, tindex_pre, tindex_post,
                                rank, species='e', exb_drift=False, verbose=True,
                                fitting_method=RegularGridInterpolator):
    """
    """
    if species == 'e':
        pmass = 1.0
        charge = -1.0
    else:
        pmass = pic_info.mime
        charge = 1.0
    nx = pic_info.nx / pic_info.topology_x
    ny = pic_info.ny / pic_info.topology_y
    nz = pic_info.nz / pic_info.topology_z
    nx2 = nx + 2
    ny2 = ny + 2
    nz2 = nz + 2
    dsize = nx2 * ny2 * nz2
    mime = pic_info.mime
    smime = math.sqrt(mime)
    dx = pic_info.dx_di * smime
    dy = pic_info.dy_di * smime
    dz = pic_info.dz_di * smime

    # file names
    hydro_dir = run_dir + 'hydro/T.' + str(tindex) + '/'
    fields_dir = run_dir + 'fields/T.' + str(tindex) + '/'
    particle_dir = run_dir + 'particle/T.' + str(tindex) + '/'
    ehydro_name = hydro_dir + 'ehydro.' + str(tindex)
    Hhydro_name = hydro_dir + 'Hhydro.' + str(tindex)
    eparticle_name = particle_dir + 'eparticle.' + str(tindex)
    hparticle_name = particle_dir + 'hparticle.' + str(tindex)
    field_name = fields_dir + 'fields.' + str(tindex)

    hydro_dir_pre = run_dir + 'hydro/T.' + str(tindex_pre) + '/'
    hydro_dir_post = run_dir + 'hydro/T.' + str(tindex_post) + '/'
    ehydro_name_pre = hydro_dir_pre + 'ehydro.' + str(tindex_pre)
    Hhydro_name_pre = hydro_dir_pre + 'Hhydro.' + str(tindex_pre)
    ehydro_name_post = hydro_dir_post + 'ehydro.' + str(tindex_post)
    Hhydro_name_post = hydro_dir_post + 'Hhydro.' + str(tindex_post)

    # read field data
    fname = field_name + '.' + str(rank)
    (v0, pheader, fields) = read_fields(fname, nx2, ny2, nz2, dsize)
    ex = fields[0, :, 1, :]
    ey = fields[1, :, 1, :]
    ez = fields[2, :, 1, :]
    bx = fields[3, :, 1, :]
    by = fields[4, :, 1, :]
    bz = fields[5, :, 1, :]
    ib2 = 1.0 / (bx**2 + by**2 + bz**2)
    # print ey[0, :], ey[-1, :]
    # print ey[:, 0], ey[:, -1]
    del fields

    # read electron hydro data
    fname = ehydro_name + '.' + str(rank)
    v0, pheader, ehydro = read_hydro(fname, nx2, ny2, nz2, dsize)
    vex = -ehydro[0, :, 1, :]
    vey = -ehydro[1, :, 1, :]
    vez = -ehydro[2, :, 1, :]
    ne = np.abs(ehydro[3, :, 1, :])
    uex = ehydro[4, :, 1, :]
    uey = ehydro[5, :, 1, :]
    uez = ehydro[6, :, 1, :]
    kee = ehydro[7, :, 1, :]
    texx = ehydro[8, :, 1, :]
    teyy = ehydro[9, :, 1, :]
    tezz = ehydro[10, :, 1, :]
    texy = ehydro[11, :, 1, :]
    texz = ehydro[12, :, 1, :]
    teyz = ehydro[13, :, 1, :]
    ine = div0(1.0, ne)
    vex *= ine
    vey *= ine
    vez *= ine
    texx -= vex*uex
    teyy -= vey*uey
    tezz -= vez*uez
    teyx = texy - uex*vey
    tezx = texz - uex*vez
    tezy = teyz - uey*vez
    texy -= vex*uey
    texz -= vex*uez
    teyz -= vey*uez
    pe_para = (texx*bx**2 + teyy*by**2 + tezz*bz**2 +
               (texy + teyx)*bx*by + (texz + tezx)*bx*bz + (teyz + tezy)*by*bz)
    pe_para *= ib2
    pe_perp = 0.5 * (texx + teyy + tezz - pe_para)
    pe = (pe_para + 2.0 * pe_perp) / 3.0
    del ehydro, ine, texx, teyy, tezz
    del texy, texz, teyz, teyx, tezx, tezy

    # read ion hydro data
    fname = Hhydro_name + '.' + str(rank)
    v0, pheader, hhydro = read_hydro(fname, nx2, ny2, nz2, dsize)
    vix = hhydro[0, :, 1, :]
    viy = hhydro[1, :, 1, :]
    viz = hhydro[2, :, 1, :]
    ni = hhydro[3, :, 1, :]
    uix = hhydro[4, :, 1, :]
    uiy = hhydro[5, :, 1, :]
    uiz = hhydro[6, :, 1, :]
    kei = hhydro[7, :, 1, :]
    tixx = hhydro[8, :, 1, :]
    tiyy = hhydro[9, :, 1, :]
    tizz = hhydro[10, :, 1, :]
    tixy = hhydro[11, :, 1, :]
    tixz = hhydro[12, :, 1, :]
    tiyz = hhydro[13, :, 1, :]
    ini = div0(1.0, ni)
    vix *= ini
    viy *= ini
    viz *= ini
    tixx -= vix*uix
    tiyy -= viy*uiy
    tizz -= viz*uiz
    tiyx = tixy - uix*viy
    tizx = tixz - uix*viz
    tizy = tiyz - uiy*viz
    tixy -= vix*uiy
    tixz -= vix*uiz
    tiyz -= viy*uiz
    pi_para = (tixx*bx**2 + tiyy*by**2 + tizz*bz**2 +
               (tixy + tiyx)*bx*by + (tixz + tizx)*bx*bz + (tiyz + tizy)*by*bz)
    pi_para *= ib2
    pi_perp = 0.5 * (tixx + tiyy + tizz - pi_para)
    pi = (pi_para + 2.0 * pi_perp) / 3.0
    del hhydro, ini, tixx, tiyy, tizz
    del tixy, tixz, tiyz, tiyx, tizx, tizy

    # read fluid velocity from previous and latter time steps
    if species == 'e':
        fname = ehydro_name_pre + '.' + str(rank)
    else:
        fname = Hhydro_name_pre + '.' + str(rank)
    # vx_pre, vy_pre, vz_pre, nrho_pre = \
    #         read_hydro_velocity_density(fname, nx2, ny2, nz2, dsize);
    ux_pre, uy_pre, uz_pre, nrho_pre = \
            read_hydro_four_velocity_density(fname, nx2, ny2, nz2, dsize)
    if species == 'e':
        fname = ehydro_name_post + '.' + str(rank)
    else:
        fname = Hhydro_name_post + '.' + str(rank)
    # vx_post, vy_post, vz_post, nrho_post = \
    #         read_hydro_velocity_density(fname, nx2, ny2, nz2, dsize);
    ux_post, uy_post, uz_post, nrho_post = \
            read_hydro_four_velocity_density(fname, nx2, ny2, nz2, dsize)
    dtf = pic_info.dtwpe * (tindex_post - tindex_pre)
    # dvx_dt = (vx_post - vx_pre) / dtf
    # dvy_dt = (vy_post - vy_pre) / dtf
    # dvz_dt = (vz_post - vz_pre) / dtf
    dvx_dt = (ux_post - ux_pre) / dtf
    dvy_dt = (uy_post - uy_pre) / dtf
    dvz_dt = (uz_post - uz_pre) / dtf
    if species == 'e':
        inrho = div0(1.0, ne)
        dvx_dt[1:, 1:] += vex[1:, 1:] * np.gradient(uex[1:, 1:]*inrho[1:, 1:], dx, axis=1) + \
                          vez[1:, 1:] * np.gradient(uex[1:, 1:]*inrho[1:, 1:], dz, axis=0)
        dvy_dt[1:, 1:] += vex[1:, 1:] * np.gradient(uey[1:, 1:]*inrho[1:, 1:], dx, axis=1) + \
                          vez[1:, 1:] * np.gradient(uey[1:, 1:]*inrho[1:, 1:], dz, axis=0)
        dvz_dt[1:, 1:] += vex[1:, 1:] * np.gradient(uez[1:, 1:]*inrho[1:, 1:], dx, axis=1) + \
                          vez[1:, 1:] * np.gradient(uez[1:, 1:]*inrho[1:, 1:], dz, axis=0)
    else:
        inrho = div0(1.0, ni)
        dvx_dt[1:, 1:] += vix[1:, 1:] * np.gradient(uix[1:, 1:]*inrho[1:, 1:], dx, axis=1) + \
                          viz[1:, 1:] * np.gradient(uix[1:, 1:]*inrho[1:, 1:], dz, axis=0)
        dvy_dt[1:, 1:] += vix[1:, 1:] * np.gradient(uiy[1:, 1:]*inrho[1:, 1:], dx, axis=1) + \
                          viz[1:, 1:] * np.gradient(uiy[1:, 1:]*inrho[1:, 1:], dz, axis=0)
        dvz_dt[1:, 1:] += vix[1:, 1:] * np.gradient(uiz[1:, 1:]*inrho[1:, 1:], dx, axis=1) + \
                          viz[1:, 1:] * np.gradient(uiz[1:, 1:]*inrho[1:, 1:], dz, axis=0)
    dvx_dt /= pmass
    dvy_dt /= pmass
    dvz_dt /= pmass
    # del vx_pre, vy_pre, vz_pre
    # del vx_post, vy_post, vz_postt
    del ux_pre, uy_pre, uz_pre
    del ux_post, uy_post, uz_post
    del inrho, nrho_pre, nrho_post

    if species == 'e':
        fname = eparticle_name + '.' + str(rank)
        pres = pe
        ppara = pe_para
        pperp = pe_perp
        ke_density = kee
    else:
        fname = hparticle_name + '.' + str(rank)
        pres = pi
        ppara = pi_para
        pperp = pi_perp
        ke_density = kei

    del pe, pi, pe_para, pe_perp, pi_para, pi_perp, kee, kei
    if verbose:
        print('Total pressure from fluid: %f' %
              (np.sum(pres[1:-1, 1:-1])*v0.dx*v0.dz))
        print('Para, perp pressure and anisotropy from fluid: %f %f %f' %
              (np.sum(ppara[1:-1, 1:-1])*v0.dx*v0.dz,
               np.sum(pperp[1:-1, 1:-1])*v0.dx*v0.dz,
               (np.sum(ppara[1:-1, 1:-1]) - np.sum(pperp[1:-1, 1:-1]))*v0.dx*v0.dz))
        print('Total kinetic energy from fluid: %f' %
              (np.sum(ke_density[1:-1, 1:-1])*v0.dx*v0.dz))

    del ke_density

    # read particle data
    (v0, pheader, ptl) = read_particle_data(fname)
    dxp = ptl['dxyz'][:, 0]
    dzp = ptl['dxyz'][:, 2]
    icell = ptl['icell']
    uxp = ptl['u'][:, 0]
    uyp = ptl['u'][:, 1]
    uzp = ptl['u'][:, 2]
    q = ptl['q']
    nx = v0.nx + 2
    ny = v0.ny + 2
    nz = v0.nz + 2
    iz = icell // (nx * ny)
    ix = icell % nx
    x_ptl = ((ix - 1.0) + (dxp + 1.0) * 0.5) * v0.dx + v0.x0
    z_ptl = ((iz - 1.0) + (dzp + 1.0) * 0.5) * v0.dz + v0.z0
    gamma = np.sqrt(1 + np.sum(ptl['u']**2, axis=1))
    igamma = 1.0 / gamma
    vxp = uxp * igamma
    vyp = uyp * igamma
    vzp = uzp * igamma
    del ptl, icell, dxp, dzp, ix, iz, igamma

    dx = v0.dx
    dz = v0.dz
    dxh = 0.5 * dx
    dzh = 0.5 * dz
    x = np.linspace(v0.x0, v0.x0 + v0.nx * dx, nx - 1)
    z = np.linspace(v0.z0, v0.z0 + v0.nz * dz, nz - 1)
    x1 = np.linspace(v0.x0 - dxh, v0.x0 + v0.nx * dx + dxh, nx)
    z1 = np.linspace(v0.z0 - dzh, v0.z0 + v0.nz * dz + dzh, nz)
    x2 = np.linspace(v0.x0 - dx, v0.x0 + v0.nx * dx, nx)
    z2 = np.linspace(v0.z0 - dz, v0.z0 + v0.nz * dz, nz)

    # interpolate electric and magnetic fields
    if fitting_method == RectBivariateSpline:
        f_ex = RectBivariateSpline(x1, z, ex[1:, :].T)
        f_ey = RectBivariateSpline(x, z, ey[1:, 1:].T)
        f_ez = RectBivariateSpline(x, z1, ez[:, 1:].T)
        f_bx = RectBivariateSpline(x2, z1, bx[:, :].T)
        f_by = RectBivariateSpline(x1, z1, by[:, :].T)
        f_bz = RectBivariateSpline(x1, z2, bz[:, :].T)
        ex_ptl = f_ex(x_ptl, z_ptl, grid=False)
        ey_ptl = f_ey(x_ptl, z_ptl, grid=False)
        ez_ptl = f_ez(x_ptl, z_ptl, grid=False)
        bx_ptl = f_bx(x_ptl, z_ptl, grid=False)
        by_ptl = f_by(x_ptl, z_ptl, grid=False)
        bz_ptl = f_bz(x_ptl, z_ptl, grid=False)
    elif fitting_method == RegularGridInterpolator:
        f_ex = RegularGridInterpolator((x1, z), ex[1:, :].T)
        f_ey = RegularGridInterpolator((x, z), ey[1:, 1:].T)
        f_ez = RegularGridInterpolator((x, z1), ez[:, 1:].T)
        f_bx = RegularGridInterpolator((x2, z1), bx[:, :].T)
        f_by = RegularGridInterpolator((x1, z1), by[:, :].T)
        f_bz = RegularGridInterpolator((x1, z2), bz[:, :].T)
        # f_ex = RegularGridInterpolator((x, z), ex[1:, 1:].T)
        # f_ey = RegularGridInterpolator((x, z), ey[1:, 1:].T)
        # f_ez = RegularGridInterpolator((x, z), ez[1:, 1:].T)
        # f_bx = RegularGridInterpolator((x, z), bx[1:, 1:].T)
        # f_by = RegularGridInterpolator((x, z), by[1:, 1:].T)
        # f_bz = RegularGridInterpolator((x, z), bz[1:, 1:].T)
        ex_ptl = f_ex((x_ptl, z_ptl))
        ey_ptl = f_ey((x_ptl, z_ptl))
        ez_ptl = f_ez((x_ptl, z_ptl))
        bx_ptl = f_bx((x_ptl, z_ptl))
        by_ptl = f_by((x_ptl, z_ptl))
        bz_ptl = f_bz((x_ptl, z_ptl))
    del f_ex, f_ey, f_ez, f_bx, f_by, f_bz
    del x1, z1, x2, z2

    # interpolate compressional terms
    inrho = div0(1.0, ne + ni*mime)
    if exb_drift:
        vx = (ey * bz - ez * by) * ib2
        vy = (ez * bx - ex * bz) * ib2
        vz = (ex * by - ey * bx) * ib2
    else:
        vx = (vex*ne + vix*ni*mime) * inrho
        vy = (vey*ne + viy*ni*mime) * inrho
        vz = (vez*ne + viz*ni*mime) * inrho
    vxb = vx * bx + vy * by + vz * bz
    vx_perp = vx - vxb * bx * ib2
    vy_perp = vy - vxb * by * ib2
    vz_perp = vz - vxb * bz * ib2
    divv = np.zeros(vx.shape)
    div_vperp = np.zeros(vx.shape)
    dvperpx_dx = np.zeros(vx.shape)
    dvperpy_dx = np.zeros(vx.shape)
    dvperpz_dx = np.zeros(vx.shape)
    dvperpx_dz = np.zeros(vx.shape)
    dvperpy_dz = np.zeros(vx.shape)
    dvperpz_dz = np.zeros(vx.shape)
    bbsigma_perp = np.zeros(vx.shape)
    dvperpx_dx[1:, 1:] = np.gradient(vx_perp[1:, 1:], dx, axis=1)
    dvperpy_dx[1:, 1:] = np.gradient(vy_perp[1:, 1:], dx, axis=1)
    dvperpz_dx[1:, 1:] = np.gradient(vz_perp[1:, 1:], dx, axis=1)
    dvperpx_dz[1:, 1:] = np.gradient(vx_perp[1:, 1:], dz, axis=0)
    dvperpy_dz[1:, 1:] = np.gradient(vy_perp[1:, 1:], dz, axis=0)
    dvperpz_dz[1:, 1:] = np.gradient(vz_perp[1:, 1:], dz, axis=0)
    divv[1:, 1:] = np.gradient(vx[1:, 1:], dx, axis=1) + \
                   np.gradient(vz[1:, 1:], dz, axis=0)
    div_vperp[1:, 1:] = dvperpx_dx[1:, 1:] + dvperpz_dz[1:, 1:]
    # div_vperp[1:, 1:] = (bx[1:, 1:] * np.gradient(-ey[1:, 1:], dz, axis=0) +
    #                      by[1:, 1:] * (np.gradient(ex[1:, 1:], dz, axis=0) -
    #                                    np.gradient(ez[1:, 1:], dx, axis=1)) +
    #                      bz[1:, 1:] * np.gradient(ey[1:, 1:], dx, axis=1) +
    #                      ex[1:, 1:] * np.gradient(by[1:, 1:], dz, axis=0) -
    #                      ey[1:, 1:] * (np.gradient(bx[1:, 1:], dz, axis=0) -
    #                                    np.gradient(bz[1:, 1:], dx, axis=1)) -
    #                      ez[1:, 1:] * np.gradient(by[1:, 1:], dx, axis=1)) * ib2[1:, 1:] - \
    #                     div0(vx[1:, 1:] * np.gradient(ib2[1:, 1:], dx, axis=1) +
    #                      vz[1:, 1:] * np.gradient(ib2[1:, 1:], dz, axis=0), ib2[1:, 1:])
    bbsigma_perp[1:, 1:] = (dvperpx_dx[1:, 1:] - (1./3.) * div_vperp[1:, 1:]) * bx[1:, 1:]**2 + \
                           (-(1./3.) * div_vperp[1:, 1:]) * by[1:, 1:]**2 + \
                           (dvperpz_dz[1:, 1:] - (1./3.) * div_vperp[1:, 1:]) * bz[1:, 1:]**2 + \
                           dvperpy_dx[1:, 1:] * bx[1:, 1:] * by[1:, 1:] + \
                           (dvperpx_dz[1:, 1:] + dvperpz_dx[1:, 1:]) * bx[1:, 1:] * bz[1:, 1:] + \
                           dvperpy_dz[1:, 1:] * by[1:, 1:] * bz[1:, 1:]
    bbsigma_perp *= ib2

    if fitting_method == RectBivariateSpline:
        f_divv = RectBivariateSpline(x, z, divv[1:, 1:].T)
        f_div_vperp = RectBivariateSpline(x, z, div_vperp[1:, 1:].T)
        f_bbsigma_perp = RectBivariateSpline(x, z, bbsigma_perp[1:, 1:].T)
        f_dvperpx_dx = RectBivariateSpline(x, z, dvperpx_dx[1:, 1:].T)
        f_dvperpy_dx = RectBivariateSpline(x, z, dvperpy_dx[1:, 1:].T)
        f_dvperpz_dx = RectBivariateSpline(x, z, dvperpz_dx[1:, 1:].T)
        f_dvperpx_dz = RectBivariateSpline(x, z, dvperpx_dz[1:, 1:].T)
        f_dvperpy_dz = RectBivariateSpline(x, z, dvperpy_dz[1:, 1:].T)
        f_dvperpz_dz = RectBivariateSpline(x, z, dvperpz_dz[1:, 1:].T)
        divv_ptl = f_divv(x_ptl, z_ptl, grid=False)
        div_vperp_ptl = f_div_vperp(x_ptl, z_ptl, grid=False)
        bbsigma_perp_ptl = f_bbsigma_perp(x_ptl, z_ptl, grid=False)
        dvperpx_dx_ptl = f_dvperpx_dx(x_ptl, z_ptl, grid=False)
        dvperpy_dx_ptl = f_dvperpy_dx(x_ptl, z_ptl, grid=False)
        dvperpz_dx_ptl = f_dvperpz_dx(x_ptl, z_ptl, grid=False)
        dvperpx_dz_ptl = f_dvperpx_dz(x_ptl, z_ptl, grid=False)
        dvperpy_dz_ptl = f_dvperpy_dz(x_ptl, z_ptl, grid=False)
        dvperpz_dz_ptl = f_dvperpz_dz(x_ptl, z_ptl, grid=False)
    elif fitting_method == RegularGridInterpolator:
        f_divv = RegularGridInterpolator((x, z), divv[1:, 1:].T)
        f_div_vperp = RegularGridInterpolator((x, z), div_vperp[1:, 1:].T)
        f_bbsigma_perp = RegularGridInterpolator((x, z), bbsigma_perp[1:, 1:].T)
        f_dvperpx_dx = RegularGridInterpolator((x, z), dvperpx_dx[1:, 1:].T)
        f_dvperpy_dx = RegularGridInterpolator((x, z), dvperpy_dx[1:, 1:].T)
        f_dvperpz_dx = RegularGridInterpolator((x, z), dvperpz_dx[1:, 1:].T)
        f_dvperpx_dz = RegularGridInterpolator((x, z), dvperpx_dz[1:, 1:].T)
        f_dvperpy_dz = RegularGridInterpolator((x, z), dvperpy_dz[1:, 1:].T)
        f_dvperpz_dz = RegularGridInterpolator((x, z), dvperpz_dz[1:, 1:].T)
        divv_ptl = f_divv((x_ptl, z_ptl))
        div_vperp_ptl = f_div_vperp((x_ptl, z_ptl))
        bbsigma_perp_ptl = f_bbsigma_perp((x_ptl, z_ptl))
        dvperpx_dx_ptl = f_dvperpx_dx((x_ptl, z_ptl))
        dvperpy_dx_ptl = f_dvperpy_dx((x_ptl, z_ptl))
        dvperpz_dx_ptl = f_dvperpz_dx((x_ptl, z_ptl))
        dvperpx_dz_ptl = f_dvperpx_dz((x_ptl, z_ptl))
        dvperpy_dz_ptl = f_dvperpy_dz((x_ptl, z_ptl))
        dvperpz_dz_ptl = f_dvperpz_dz((x_ptl, z_ptl))

    ds = v0.dx * v0.dz
    if verbose:
        print("pdivv and pdiv_vperp and pshear from fluid: %f %f %f" %
              (np.sum(-pres[1:, 1:]*divv[1:, 1:])*ds,
               np.sum(-pres[1:, 1:]*div_vperp[1:, 1:])*ds,
               np.sum(-(ppara[1:, 1:] - pperp[1:, 1:])*bbsigma_perp[1:, 1:])*ds))
    del f_divv, f_div_vperp, f_bbsigma_perp
    del f_dvperpx_dx, f_dvperpy_dx, f_dvperpz_dx
    del f_dvperpx_dz, f_dvperpy_dz, f_dvperpz_dz
    del inrho, pres, divv, div_vperp, vxb, ib2, vx_perp, vy_perp, vz_perp
    del dvperpx_dx, dvperpy_dx, dvperpz_dx, dvperpx_dz, dvperpy_dz, dvperpz_dz
    del ppara, pperp, bbsigma_perp

    # interpolate motional electric field
    einx = vz*by - vy*bz
    einy = vx*bz - vz*bx
    einz = vy*bx - vx*by
    if fitting_method == RectBivariateSpline:
        f_einx = RectBivariateSpline(x, z, einx[1:, 1:].T)
        f_einy = RectBivariateSpline(x, z, einy[1:, 1:].T)
        f_einz = RectBivariateSpline(x, z, einz[1:, 1:].T)
        einx_ptl = f_einx(x_ptl, z_ptl, grid=False)
        einy_ptl = f_einy(x_ptl, z_ptl, grid=False)
        einz_ptl = f_einz(x_ptl, z_ptl, grid=False)
    elif fitting_method == RegularGridInterpolator:
        f_einx = RegularGridInterpolator((x, z), einx[1:, 1:].T)
        f_einy = RegularGridInterpolator((x, z), einy[1:, 1:].T)
        f_einz = RegularGridInterpolator((x, z), einz[1:, 1:].T)
        einx_ptl = f_einx((x_ptl, z_ptl))
        einy_ptl = f_einy((x_ptl, z_ptl))
        einz_ptl = f_einz((x_ptl, z_ptl))
    del f_einx, f_einy, f_einz
    del ex, ey, ez, bx, by, bz
    del vx, vy, vz, einx, einy, einz

    # interpolate fluid velocities
    if species == 'e':
        if fitting_method == RectBivariateSpline:
            f_vx = RectBivariateSpline(x, z, vex[1:, 1:].T)
            f_vy = RectBivariateSpline(x, z, vey[1:, 1:].T)
            f_vz = RectBivariateSpline(x, z, vez[1:, 1:].T)
            f_ux = RectBivariateSpline(x, z, uex[1:, 1:].T/ne[1:, 1:].T)
            f_uy = RectBivariateSpline(x, z, uey[1:, 1:].T/ne[1:, 1:].T)
            f_uz = RectBivariateSpline(x, z, uez[1:, 1:].T/ne[1:, 1:].T)
            f_vx_ux = RectBivariateSpline(x, z, vex[1:, 1:].T*uex[1:, 1:].T/ne[1:, 1:].T)
            f_vy_uy = RectBivariateSpline(x, z, vey[1:, 1:].T*uey[1:, 1:].T/ne[1:, 1:].T)
            f_vz_uz = RectBivariateSpline(x, z, vez[1:, 1:].T*uez[1:, 1:].T/ne[1:, 1:].T)
            f_vx_uy = RectBivariateSpline(x, z, vex[1:, 1:].T*uey[1:, 1:].T/ne[1:, 1:].T)
            f_vx_uz = RectBivariateSpline(x, z, vex[1:, 1:].T*uez[1:, 1:].T/ne[1:, 1:].T)
            f_vy_uz = RectBivariateSpline(x, z, vey[1:, 1:].T*uez[1:, 1:].T/ne[1:, 1:].T)
            f_vy_ux = RectBivariateSpline(x, z, vey[1:, 1:].T*uex[1:, 1:].T/ne[1:, 1:].T)
            f_vz_ux = RectBivariateSpline(x, z, vez[1:, 1:].T*uex[1:, 1:].T/ne[1:, 1:].T)
            f_vz_uy = RectBivariateSpline(x, z, vez[1:, 1:].T*uey[1:, 1:].T/ne[1:, 1:].T)
        elif fitting_method == RegularGridInterpolator:
            f_vx = RegularGridInterpolator((x, z), vex[1:, 1:].T)
            f_vy = RegularGridInterpolator((x, z), vey[1:, 1:].T)
            f_vz = RegularGridInterpolator((x, z), vez[1:, 1:].T)
            f_ux = RegularGridInterpolator((x, z), uex[1:, 1:].T/ne[1:, 1:].T)
            f_uy = RegularGridInterpolator((x, z), uey[1:, 1:].T/ne[1:, 1:].T)
            f_uz = RegularGridInterpolator((x, z), uez[1:, 1:].T/ne[1:, 1:].T)
            f_vx_ux = RegularGridInterpolator((x, z), vex[1:, 1:].T*uex[1:, 1:].T/ne[1:, 1:].T)
            f_vy_uy = RegularGridInterpolator((x, z), vey[1:, 1:].T*uey[1:, 1:].T/ne[1:, 1:].T)
            f_vz_uz = RegularGridInterpolator((x, z), vez[1:, 1:].T*uez[1:, 1:].T/ne[1:, 1:].T)
            f_vx_uy = RegularGridInterpolator((x, z), vex[1:, 1:].T*uey[1:, 1:].T/ne[1:, 1:].T)
            f_vx_uz = RegularGridInterpolator((x, z), vex[1:, 1:].T*uez[1:, 1:].T/ne[1:, 1:].T)
            f_vy_uz = RegularGridInterpolator((x, z), vey[1:, 1:].T*uez[1:, 1:].T/ne[1:, 1:].T)
            f_vy_ux = RegularGridInterpolator((x, z), vey[1:, 1:].T*uex[1:, 1:].T/ne[1:, 1:].T)
            f_vz_ux = RegularGridInterpolator((x, z), vez[1:, 1:].T*uex[1:, 1:].T/ne[1:, 1:].T)
            f_vz_uy = RegularGridInterpolator((x, z), vez[1:, 1:].T*uey[1:, 1:].T/ne[1:, 1:].T)
    else:
        if fitting_method == RectBivariateSpline:
            f_vx = RectBivariateSpline(x, z, vix[1:, 1:].T)
            f_vy = RectBivariateSpline(x, z, viy[1:, 1:].T)
            f_vz = RectBivariateSpline(x, z, viz[1:, 1:].T)
            f_ux = RectBivariateSpline(x, z, uix[1:, 1:].T/ni[1:, 1:].T)
            f_uy = RectBivariateSpline(x, z, uiy[1:, 1:].T/ni[1:, 1:].T)
            f_uz = RectBivariateSpline(x, z, uiz[1:, 1:].T/ni[1:, 1:].T)
            f_vx_ux = RectBivariateSpline(x, z, vix[1:, 1:].T*uix[1:, 1:].T/ni[1:, 1:].T)
            f_vy_uy = RectBivariateSpline(x, z, viy[1:, 1:].T*uiy[1:, 1:].T/ni[1:, 1:].T)
            f_vz_uz = RectBivariateSpline(x, z, viz[1:, 1:].T*uiz[1:, 1:].T/ni[1:, 1:].T)
            f_vx_uy = RectBivariateSpline(x, z, vix[1:, 1:].T*uiy[1:, 1:].T/ni[1:, 1:].T)
            f_vx_uz = RectBivariateSpline(x, z, vix[1:, 1:].T*uiz[1:, 1:].T/ni[1:, 1:].T)
            f_vy_uz = RectBivariateSpline(x, z, viy[1:, 1:].T*uiz[1:, 1:].T/ni[1:, 1:].T)
            f_vy_ux = RectBivariateSpline(x, z, viy[1:, 1:].T*uix[1:, 1:].T/ni[1:, 1:].T)
            f_vz_ux = RectBivariateSpline(x, z, viz[1:, 1:].T*uix[1:, 1:].T/ni[1:, 1:].T)
            f_vz_uy = RectBivariateSpline(x, z, viz[1:, 1:].T*uiy[1:, 1:].T/ni[1:, 1:].T)
        elif fitting_method == RegularGridInterpolator:
            f_vx = RegularGridInterpolator((x, z), vix[1:, 1:].T)
            f_vy = RegularGridInterpolator((x, z), viy[1:, 1:].T)
            f_vz = RegularGridInterpolator((x, z), viz[1:, 1:].T)
            f_ux = RegularGridInterpolator((x, z), uix[1:, 1:].T/ni[1:, 1:].T)
            f_uy = RegularGridInterpolator((x, z), uiy[1:, 1:].T/ni[1:, 1:].T)
            f_uz = RegularGridInterpolator((x, z), uiz[1:, 1:].T/ni[1:, 1:].T)
            f_vx_ux = RegularGridInterpolator((x, z), vix[1:, 1:].T*uix[1:, 1:].T/ni[1:, 1:].T)
            f_vy_uy = RegularGridInterpolator((x, z), viy[1:, 1:].T*uiy[1:, 1:].T/ni[1:, 1:].T)
            f_vz_uz = RegularGridInterpolator((x, z), viz[1:, 1:].T*uiz[1:, 1:].T/ni[1:, 1:].T)
            f_vx_uy = RegularGridInterpolator((x, z), vix[1:, 1:].T*uiy[1:, 1:].T/ni[1:, 1:].T)
            f_vx_uz = RegularGridInterpolator((x, z), vix[1:, 1:].T*uiz[1:, 1:].T/ni[1:, 1:].T)
            f_vy_uz = RegularGridInterpolator((x, z), viy[1:, 1:].T*uiz[1:, 1:].T/ni[1:, 1:].T)
            f_vy_ux = RegularGridInterpolator((x, z), viy[1:, 1:].T*uix[1:, 1:].T/ni[1:, 1:].T)
            f_vz_ux = RegularGridInterpolator((x, z), viz[1:, 1:].T*uix[1:, 1:].T/ni[1:, 1:].T)
            f_vz_uy = RegularGridInterpolator((x, z), viz[1:, 1:].T*uiy[1:, 1:].T/ni[1:, 1:].T)

    if fitting_method == RectBivariateSpline:
        vx_ux_ptl = f_vx_ux(x_ptl, z_ptl, grid=False)
        vy_uy_ptl = f_vy_uy(x_ptl, z_ptl, grid=False)
        vz_uz_ptl = f_vz_uz(x_ptl, z_ptl, grid=False)
        vx_uy_ptl = f_vx_uy(x_ptl, z_ptl, grid=False)
        vx_uz_ptl = f_vx_uz(x_ptl, z_ptl, grid=False)
        vy_uz_ptl = f_vy_uz(x_ptl, z_ptl, grid=False)
        vy_ux_ptl = f_vy_ux(x_ptl, z_ptl, grid=False)
        vz_ux_ptl = f_vz_ux(x_ptl, z_ptl, grid=False)
        vz_uy_ptl = f_vz_uy(x_ptl, z_ptl, grid=False)
        vx_ptl = f_vx(x_ptl, z_ptl, grid=False)
        vy_ptl = f_vy(x_ptl, z_ptl, grid=False)
        vz_ptl = f_vz(x_ptl, z_ptl, grid=False)
        ux_ptl = f_ux(x_ptl, z_ptl, grid=False)
        uy_ptl = f_uy(x_ptl, z_ptl, grid=False)
        uz_ptl = f_uz(x_ptl, z_ptl, grid=False)
    elif fitting_method == RegularGridInterpolator:
        vx_ux_ptl = f_vx_ux((x_ptl, z_ptl))
        vy_uy_ptl = f_vy_uy((x_ptl, z_ptl))
        vz_uz_ptl = f_vz_uz((x_ptl, z_ptl))
        vx_uy_ptl = f_vx_uy((x_ptl, z_ptl))
        vx_uz_ptl = f_vx_uz((x_ptl, z_ptl))
        vy_uz_ptl = f_vy_uz((x_ptl, z_ptl))
        vy_ux_ptl = f_vy_ux((x_ptl, z_ptl))
        vz_ux_ptl = f_vz_ux((x_ptl, z_ptl))
        vz_uy_ptl = f_vz_uy((x_ptl, z_ptl))
        vx_ptl = f_vx((x_ptl, z_ptl))
        vy_ptl = f_vy((x_ptl, z_ptl))
        vz_ptl = f_vz((x_ptl, z_ptl))
        ux_ptl = f_ux((x_ptl, z_ptl))
        uy_ptl = f_uy((x_ptl, z_ptl))
        uz_ptl = f_uz((x_ptl, z_ptl))
    del f_vx_ux, f_vy_uy, f_vz_uz, f_vx, f_vy, f_vz, f_ux, f_uy, f_uz
    del f_vx_uy, f_vx_uz, f_vy_uz, f_vy_ux, f_vz_ux, f_vz_uy
    del vex, vey, vez, uex, uey, uez, ne
    del vix, viy, viz, uix, uiy, uiz, ni

    # interpolate fluid acceleration
    if fitting_method == RectBivariateSpline:
        f_dvx_dt = RectBivariateSpline(x, z, dvx_dt[1:, 1:].T)
        f_dvy_dt = RectBivariateSpline(x, z, dvy_dt[1:, 1:].T)
        f_dvz_dt = RectBivariateSpline(x, z, dvz_dt[1:, 1:].T)
        dvx_dt_ptl = f_dvx_dt(x_ptl, z_ptl, grid=False)
        dvy_dt_ptl = f_dvy_dt(x_ptl, z_ptl, grid=False)
        dvz_dt_ptl = f_dvz_dt(x_ptl, z_ptl, grid=False)
    else:
        f_dvx_dt = RegularGridInterpolator((x, z), dvx_dt[1:, 1:].T)
        f_dvy_dt = RegularGridInterpolator((x, z), dvy_dt[1:, 1:].T)
        f_dvz_dt = RegularGridInterpolator((x, z), dvz_dt[1:, 1:].T)
        dvx_dt_ptl = f_dvx_dt((x_ptl, z_ptl))
        dvy_dt_ptl = f_dvy_dt((x_ptl, z_ptl))
        dvz_dt_ptl = f_dvz_dt((x_ptl, z_ptl))
    del f_dvx_dt, f_dvy_dt, f_dvz_dt
    del dvx_dt, dvy_dt, dvz_dt
    del x_ptl, z_ptl, x, z

    # compressional and shear heating
    weight = abs(q[0])
    pxx = (vxp * uxp) * pmass + vx_ux_ptl - vx_ptl * uxp * pmass - vxp * ux_ptl
    pyy = (vyp * uyp) * pmass + vy_uy_ptl - vy_ptl * uyp * pmass - vyp * uy_ptl
    pzz = (vzp * uzp) * pmass + vz_uz_ptl - vz_ptl * uzp * pmass - vzp * uz_ptl
    pxy = (vxp * uyp) * pmass + vx_uy_ptl - vx_ptl * uyp * pmass - vxp * uy_ptl
    pxz = (vxp * uzp) * pmass + vx_uz_ptl - vx_ptl * uzp * pmass - vxp * uz_ptl
    pyz = (vyp * uzp) * pmass + vy_uz_ptl - vy_ptl * uzp * pmass - vyp * uz_ptl
    pyx = (vyp * uxp) * pmass + vy_ux_ptl - vy_ptl * uxp * pmass - vyp * ux_ptl
    pzx = (vzp * uxp) * pmass + vz_ux_ptl - vz_ptl * uxp * pmass - vzp * ux_ptl
    pzy = (vzp * uyp) * pmass + vz_uy_ptl - vz_ptl * uyp * pmass - vzp * uy_ptl
    pxx *= weight
    pyy *= weight
    pzz *= weight
    pxy *= weight
    pxz *= weight
    pyz *= weight
    pyx *= weight
    pzx *= weight
    pzy *= weight
    pscalar = (pxx + pyy + pzz) / 3.0
    pdivv = -pscalar * divv_ptl
    pdiv_vperp = -pscalar * div_vperp_ptl
    ptensor_dv = -(pxx * dvperpx_dx_ptl + pxy * dvperpy_dx_ptl + pxz * dvperpz_dx_ptl)
    ptensor_dv -= pzx * dvperpx_dz_ptl + pzy * dvperpy_dz_ptl + pzz * dvperpz_dz_ptl

    bx2 = bx_ptl**2
    by2 = by_ptl**2
    bz2 = bz_ptl**2
    bxy = bx_ptl * by_ptl
    bxz = bx_ptl * bz_ptl
    byz = by_ptl * bz_ptl
    ib2_ptl = 1.0 / (bx2 + by2 + bz2)
    ppara_ptl = pxx * bx2 + pyy * by2 + pzz * bz2 + \
                (pxy + pyx) * bxy + (pxz + pzx) * bxz + (pyz + pzy) * byz

    ppara_ptl *= ib2_ptl
    pperp_ptl = 0.5 * (pscalar * 3 - ppara_ptl)
    pshear = (pperp_ptl - ppara_ptl) * bbsigma_perp_ptl

    if verbose:
        print('Total pressure from particles: %f' % np.sum(pscalar))
        print('Para, perp pressure and anisotropy from particles: %f %f %f' %
              (np.sum(ppara_ptl), np.sum(pperp_ptl), np.sum(ppara_ptl) - np.sum(pperp_ptl)))
        print("pdivv, pdiv_vperp, pshear and ptensor_dv from particles: %f %f %f %f" %
              (np.sum(pdivv), np.sum(pdiv_vperp), np.sum(pshear), np.sum(ptensor_dv)))
        print('Total kinetic energy from particles: %f' % (np.sum(gamma - 1) * weight * pmass))
    del bx2, by2, bz2, bxy, bxz, byz
    del bbsigma_perp_ptl, ppara_ptl, pperp_ptl
    del pscalar, vx_ptl, vy_ptl, vz_ptl, ux_ptl, uy_ptl, uz_ptl
    del vx_ux_ptl, vy_uy_ptl, vz_uz_ptl, divv_ptl, div_vperp_ptl
    del vx_uy_ptl, vx_uz_ptl, vy_uz_ptl, vy_ux_ptl, vz_ux_ptl, vz_uy_ptl
    del pxx, pyy, pzz, pxy, pxz, pyz, pyx, pzx, pzy
    del dvperpx_dx_ptl, dvperpy_dx_ptl, dvperpz_dx_ptl
    del dvperpx_dz_ptl, dvperpy_dz_ptl, dvperpz_dz_ptl
    del uxp, uyp, uzp, q

    # parallel and perpendicular heating
    exb_ptl = ex_ptl * bx_ptl + ey_ptl * by_ptl + ez_ptl * bz_ptl
    ex_para_ptl = exb_ptl * bx_ptl * ib2_ptl
    ey_para_ptl = exb_ptl * by_ptl * ib2_ptl
    ez_para_ptl = exb_ptl * bz_ptl * ib2_ptl
    ex_perp_ptl = ex_ptl - ex_para_ptl
    ey_perp_ptl = ey_ptl - ey_para_ptl
    ez_perp_ptl = ez_ptl - ez_para_ptl

    de_para = charge * (vxp * ex_para_ptl + vyp * ey_para_ptl + vzp * ez_para_ptl) * weight
    de_perp = charge * (vxp * ex_perp_ptl + vyp * ey_perp_ptl + vzp * ez_perp_ptl) * weight
    de_tot = de_para + de_perp

    del exb_ptl
    del ex_para_ptl, ey_para_ptl, ez_para_ptl
    del ex_perp_ptl, ey_perp_ptl, ez_perp_ptl

    # heating due to inertial term (fluid acceleration term)
    de_dvdt = (dvz_dt_ptl * by_ptl - dvy_dt_ptl * bz_ptl) * ex_ptl + \
              (dvx_dt_ptl * bz_ptl - dvz_dt_ptl * bx_ptl) * ey_ptl + \
              (dvy_dt_ptl * bx_ptl - dvx_dt_ptl * by_ptl) * ez_ptl
    de_dvdt *= pmass * weight * ib2_ptl

    del ex_ptl, ey_ptl, ez_ptl, bx_ptl, by_ptl, bz_ptl, ib2_ptl
    del dvx_dt_ptl, dvy_dt_ptl, dvz_dt_ptl

    # heating due -vxb electric field
    de_vxb = charge * (vxp*einx_ptl + vyp*einy_ptl + vzp*einz_ptl) * weight
    del einx_ptl, einy_ptl, einz_ptl
    del vxp, vyp, vzp

    if verbose:
        print("Parallel and perpendicular heating: %d, %f, %f" %
              (rank, np.sum(de_para), np.sum(de_perp)))
        print("Heating due to ideal electric field: %d, %f" % (rank, np.sum(de_vxb)))
        print("Maximum and minimum energy gain: %12.5e, %12.5e, %12.5e, %12.5e" %
              (np.max(de_para), np.min(de_para), np.max(de_perp), np.min(de_perp)))
        print("Heating due inertial term: %f" % np.sum(de_dvdt))

    # get the distribution and save the data
    nbins = 60
    drange = [[1, 1.1], [0, 1]]
    ebins = np.logspace(-4, 2, nbins) / math.sqrt(pmass)

    fdir = run_dir + 'data_ene/'
    mkdir_p(fdir)
    if verbose:
        print("Maximum and minimum energy: %12.5e, %12.5e" %
              (np.max(gamma-1), np.min(gamma-1)))

    hist_de_para, bin_edges = np.histogram(gamma-1, bins=ebins, weights=de_para)
    hist_de_perp, bin_edges = np.histogram(gamma-1, bins=ebins, weights=de_perp)
    hist_de_vxb, bin_edges = np.histogram(gamma-1, bins=ebins, weights=de_vxb)
    fname = fdir + 'hist_de_para.' + str(tindex) + '.' + str(rank)
    hist_de_para.tofile(fname)
    fname = fdir + 'hist_de_perp.' + str(tindex) + '.' + str(rank)
    hist_de_perp.tofile(fname)
    fname = fdir + 'hist_de_vxb.' + str(tindex) + '.' + str(rank)
    hist_de_vxb.tofile(fname)
    del hist_de_para, hist_de_perp, hist_de_vxb, bin_edges
    del de_para, de_perp, de_tot, de_vxb

    hist_pdivv, bin_edges = np.histogram(gamma-1, bins=ebins, weights=pdivv)
    hist_pdiv_vperp, bin_edges = np.histogram(gamma-1, bins=ebins, weights=pdiv_vperp)
    hist_pshear, bin_edges = np.histogram(gamma-1, bins=ebins, weights=pshear)
    hist_ptensor_dv, bin_edges = np.histogram(gamma-1, bins=ebins, weights=ptensor_dv)
    fname = fdir + 'hist_pdivv.' + str(tindex) + '.' + str(rank)
    hist_pdivv.tofile(fname)
    fname = fdir + 'hist_pdiv_vperp.' + str(tindex) + '.' + str(rank)
    hist_pdiv_vperp.tofile(fname)
    fname = fdir + 'hist_pshear.' + str(tindex) + '.' + str(rank)
    hist_pshear.tofile(fname)
    fname = fdir + 'hist_ptensor_dv.' + str(tindex) + '.' + str(rank)
    hist_ptensor_dv.tofile(fname)
    del hist_pdivv, hist_pdiv_vperp, hist_pshear, hist_ptensor_dv, bin_edges
    del pdivv, pdiv_vperp, pshear, ptensor_dv

    hist_de_dvdt, bin_edges = np.histogram(gamma-1, bins=ebins, weights=de_dvdt)
    fname = fdir + 'hist_de_dvdt.' + str(tindex) + '.' + str(rank)
    hist_de_dvdt.tofile(fname)
    del hist_de_dvdt, bin_edges, de_dvdt

    hist_nptl, bin_edges = np.histogram(gamma-1, bins=ebins)
    hist_nptl = hist_nptl.astype(np.float)
    fname = fdir + 'hist_nptl.' + str(tindex) + '.' + str(rank)
    hist_nptl.tofile(fname)
    del hist_nptl, bin_edges
    del gamma


def fill_boundary_values(data_pre):
    """Fill boundary values
    """
    nz, nx = data_pre.shape
    data_after = np.zeros((nz + 2, nx + 2))
    data_after[1:-1, 1:-1] = data_pre
    # data_after[0, :] = data_after[nz, :]
    # data_after[-1, :] = data_after[1, :]
    # data_after[:, 0] = data_after[:, nx]
    # data_after[:, -1] = data_after[:, 1]

    return data_after


def interpolation_single_rank(run_dir, rank, pmass, species, tindex,
                              fitting_functions):
    """
    """
    # if rank % 50 == 0:
    #     print("Rank: %d" % rank)
    print("Rank: %d" % rank)
    particle_dir = run_dir + 'particle/T.' + str(tindex) + '/'
    eparticle_name = particle_dir + 'eparticle.' + str(tindex)
    hparticle_name = particle_dir + 'hparticle.' + str(tindex)
    if species == 'e':
        charge = -1.0
        fname = eparticle_name + '.' + str(rank)
    else:
        charge = 1.0
        fname = hparticle_name + '.' + str(rank)
    # get the distribution and save the data
    nbins = 60
    ebins = np.logspace(-4, 2, nbins + 1) / math.sqrt(pmass)

    # read particle data
    (v0, pheader, ptl) = read_particle_data(fname)
    dxp = ptl['dxyz'][:, 0]
    dzp = ptl['dxyz'][:, 2]
    icell = ptl['icell']
    uxp = ptl['u'][:, 0]
    uyp = ptl['u'][:, 1]
    uzp = ptl['u'][:, 2]
    q = ptl['q']
    nx = v0.nx + 2
    ny = v0.ny + 2
    nz = v0.nz + 2
    iz = icell // (nx * ny)
    ix = icell % nx
    x_ptl = ((ix - 1.0) + (dxp + 1.0) * 0.5) * v0.dx + v0.x0
    z_ptl = ((iz - 1.0) + (dzp + 1.0) * 0.5) * v0.dz + v0.z0
    gamma = np.sqrt(1 + np.sum(ptl['u']**2, axis=1))
    igamma = 1.0 / gamma
    vxp = uxp * igamma
    vyp = uyp * igamma
    vzp = uzp * igamma
    weight = abs(q[0])
    coord = np.vstack((x_ptl, z_ptl))
    del ptl, icell, dxp, dzp, ix, iz, igamma, q

    # ex_ptl = fitting_functions['f_ex'](x_ptl, z_ptl, grid=False)
    # ey_ptl = fitting_functions['f_ey'](x_ptl, z_ptl, grid=False)
    # ez_ptl = fitting_functions['f_ez'](x_ptl, z_ptl, grid=False)
    # bx_ptl = fitting_functions['f_bx'](x_ptl, z_ptl, grid=False)
    # by_ptl = fitting_functions['f_by'](x_ptl, z_ptl, grid=False)
    # bz_ptl = fitting_functions['f_bz'](x_ptl, z_ptl, grid=False)
    ex_ptl = fitting_functions['f_ex'](coord)
    ey_ptl = fitting_functions['f_ey'](coord)
    ez_ptl = fitting_functions['f_ez'](coord)
    bx_ptl = fitting_functions['f_bx'](coord)
    by_ptl = fitting_functions['f_by'](coord)
    bz_ptl = fitting_functions['f_bz'](coord)
    bx2 = bx_ptl**2
    by2 = by_ptl**2
    bz2 = bz_ptl**2
    bxy = bx_ptl * by_ptl
    bxz = bx_ptl * bz_ptl
    byz = by_ptl * bz_ptl
    ib2_ptl = div0(1.0, bx2 + by2 + bz2)

    # parallel and perpendicular heating
    # exb_ptl = ex_ptl * bx_ptl + ey_ptl * by_ptl + ez_ptl * bz_ptl
    # ex_para_ptl = exb_ptl * bx_ptl * ib2_ptl
    # ey_para_ptl = exb_ptl * by_ptl * ib2_ptl
    # ez_para_ptl = exb_ptl * bz_ptl * ib2_ptl
    # ex_perp_ptl = ex_ptl - ex_para_ptl
    # ey_perp_ptl = ey_ptl - ey_para_ptl
    # ez_perp_ptl = ez_ptl - ez_para_ptl
    # de_para = charge * (vxp * ex_para_ptl + vyp * ey_para_ptl + vzp * ez_para_ptl) * weight
    # de_perp = charge * (vxp * ex_perp_ptl + vyp * ey_perp_ptl + vzp * ez_perp_ptl) * weight
    # del exb_ptl
    # del ex_para_ptl, ey_para_ptl, ez_para_ptl
    # del ex_perp_ptl, ey_perp_ptl, ez_perp_ptl
    vdotb_ptl = vxp * bx_ptl + vyp * by_ptl + vzp * bz_ptl
    de_para = vdotb_ptl * (bx_ptl * ex_ptl + by_ptl * ey_ptl + bz_ptl * ez_ptl)
    de_para *= ib2_ptl * charge * weight
    de_perp = (vxp * ex_ptl + vyp * ey_ptl + vzp * ez_ptl) * charge * weight
    de_perp -= de_para

    # heating due to inertial term (fluid acceleration term)
    # dux_dt_ptl = fitting_functions['f_dux_dt'](x_ptl, z_ptl, grid=False)
    # duy_dt_ptl = fitting_functions['f_duy_dt'](x_ptl, z_ptl, grid=False)
    # duz_dt_ptl = fitting_functions['f_duz_dt'](x_ptl, z_ptl, grid=False)
    dux_dt_ptl = fitting_functions['f_dux_dt'](coord)
    duy_dt_ptl = fitting_functions['f_duy_dt'](coord)
    duz_dt_ptl = fitting_functions['f_duz_dt'](coord)

    # dux_dx_ptl = fitting_functions['f_dux_dx']((x_ptl, z_ptl))
    # duy_dx_ptl = fitting_functions['f_duy_dx']((x_ptl, z_ptl))
    # duz_dx_ptl = fitting_functions['f_duz_dx']((x_ptl, z_ptl))
    # dux_dz_ptl = fitting_functions['f_dux_dz']((x_ptl, z_ptl))
    # duy_dz_ptl = fitting_functions['f_duy_dz']((x_ptl, z_ptl))
    # duz_dz_ptl = fitting_functions['f_duz_dz']((x_ptl, z_ptl))
    divv_species_ptl = fitting_functions['f_divv_species']((x_ptl, z_ptl))
    # vx_ptl = fitting_functions['f_vx'](x_ptl, z_ptl, grid=False)
    # vy_ptl = fitting_functions['f_vy'](x_ptl, z_ptl, grid=False)
    # vz_ptl = fitting_functions['f_vz'](x_ptl, z_ptl, grid=False)
    # ux_ptl = fitting_functions['f_ux'](x_ptl, z_ptl, grid=False)
    # uy_ptl = fitting_functions['f_uy'](x_ptl, z_ptl, grid=False)
    # uz_ptl = fitting_functions['f_uz'](x_ptl, z_ptl, grid=False)
    vx_ptl = fitting_functions['f_vx'](coord)
    vy_ptl = fitting_functions['f_vy'](coord)
    vz_ptl = fitting_functions['f_vz'](coord)
    ux_ptl = fitting_functions['f_ux'](coord)
    uy_ptl = fitting_functions['f_uy'](coord)
    uz_ptl = fitting_functions['f_uz'](coord)

    # dux_dt_ptl += vxp * dux_dx_ptl + vzp * dux_dz_ptl
    # duy_dt_ptl += vxp * duy_dx_ptl + vzp * duy_dz_ptl
    # duz_dt_ptl += vxp * duz_dx_ptl + vzp * duz_dz_ptl

    # dux_dt_ptl -= vxp * dux_dx_ptl + vzp * dux_dz_ptl
    # duy_dt_ptl -= vxp * duy_dx_ptl + vzp * duy_dz_ptl
    # duz_dt_ptl -= vxp * duz_dx_ptl + vzp * duz_dz_ptl

    # dux_dt_ptl -= divv_species_ptl * uxp
    # duy_dt_ptl -= divv_species_ptl * uyp
    # duz_dt_ptl -= divv_species_ptl * uzp

    # dux_dt_ptl += divv_species_ptl * ux_ptl
    # duy_dt_ptl += divv_species_ptl * uy_ptl
    # duz_dt_ptl += divv_species_ptl * uz_ptl

    # dux_dt_ptl += vx_ptl * dux_dx_ptl + vz_ptl * dux_dz_ptl
    # duy_dt_ptl += vx_ptl * duy_dx_ptl + vz_ptl * duy_dz_ptl
    # duz_dt_ptl += vx_ptl * duz_dx_ptl + vz_ptl * duz_dz_ptl

    de_dudt = (duz_dt_ptl * by_ptl - duy_dt_ptl * bz_ptl) * ex_ptl + \
              (dux_dt_ptl * bz_ptl - duz_dt_ptl * bx_ptl) * ey_ptl + \
              (duy_dt_ptl * bx_ptl - dux_dt_ptl * by_ptl) * ez_ptl
    de_dudt *= pmass * weight * ib2_ptl
    # de_dudt = dux_dt_ptl * vxp + duy_dt_ptl * vyp + duz_dt_ptl * vzp
    # de_dudt = (dux_dt_ptl * (vxp - vdotb_ptl * bx_ptl * ib2_ptl) +
    #            duy_dt_ptl * (vyp - vdotb_ptl * by_ptl * ib2_ptl) +
    #            duz_dt_ptl * (vzp - vdotb_ptl * bz_ptl * ib2_ptl))
    # de_dudt *= pmass * weight
    del vdotb_ptl

    # de_dudt *= pmass * weight * ib2_ptl * (gamma - 1.0)
    # de_dudt = pmass * weight * dux_dt_ptl * (gamma - 1.0)
    # dux_dt_ptl = uxp * divv_species_ptl
    # duy_dt_ptl = uyp * divv_species_ptl
    # duz_dt_ptl = uzp * divv_species_ptl
    # dux_dt_ptl = vxp * dux_dx_ptl + vzp * dux_dz_ptl
    # duy_dt_ptl = vxp * duy_dx_ptl + vzp * duy_dz_ptl
    # duz_dt_ptl = vxp * duz_dx_ptl + vzp * duz_dz_ptl
    # de_dudt += ((duz_dt_ptl * by_ptl - duy_dt_ptl * bz_ptl) * ex_ptl +
    #             (dux_dt_ptl * bz_ptl - duz_dt_ptl * bx_ptl) * ey_ptl +
    #             (duy_dt_ptl * bx_ptl - dux_dt_ptl * by_ptl) * ez_ptl) * pmass * weight * ib2_ptl
    # de_dudt = divv_species_ptl - (uxp * dux_dt_ptl +
    #                               uyp * duy_dt_ptl +
    #                               uzp * duz_dt_ptl)
    # de_dudt *= pmass * weight

    # vex = ey_ptl * bz_ptl - ez_ptl * by_ptl
    # vey = ez_ptl * bx_ptl - ex_ptl * bz_ptl
    # vez = ex_ptl * by_ptl - ey_ptl * bx_ptl
    # ve2 = (vex**2 + vey**2 + vez**2) * ib2_ptl * ib2_ptl
    # ve2[ve2 >= 1] = 0.0
    # gamma1 = np.squeeze(gamma * np.sqrt(1.0/(1.0 - ve2)) * (1.0 - vex*vxp - vey*vyp - vez*vzp))

    # del vex, vey, vez, ve2

    del ex_ptl, ey_ptl, ez_ptl
    del dux_dt_ptl, duy_dt_ptl, duz_dt_ptl
    # del dux_dx_ptl, duy_dx_ptl, duz_dx_ptl
    # del dux_dz_ptl, duy_dz_ptl, duz_dz_ptl
    del divv_species_ptl

    # heating due to conservation of mu
    # db_dt_ptl = fitting_functions['f_db_dt'](x_ptl, z_ptl, grid=False)
    db_dt_ptl = fitting_functions['f_db_dt'](coord)
    upara = uxp * bx_ptl + uyp * vy_ptl + uzp * bz_ptl
    uperp2 = uxp**2 + uyp**2 + uzp**2 - upara**2 * ib2_ptl
    de_cons_mu = 0.5 * (pmass * uperp2 * np.sqrt(ib2_ptl) / gamma) * db_dt_ptl * weight

    del bx_ptl, by_ptl, bz_ptl
    del db_dt_ptl, upara, uperp2

    # compressional and shear heating
    # vx_ux_ptl = fitting_functions['f_vx_ux']((x_ptl, z_ptl))
    # vy_uy_ptl = fitting_functions['f_vy_uy']((x_ptl, z_ptl))
    # vz_uz_ptl = fitting_functions['f_vz_uz']((x_ptl, z_ptl))
    # vx_uy_ptl = fitting_functions['f_vx_uy']((x_ptl, z_ptl))
    # vx_uz_ptl = fitting_functions['f_vx_uz']((x_ptl, z_ptl))
    # vy_uz_ptl = fitting_functions['f_vy_uz']((x_ptl, z_ptl))
    # vy_ux_ptl = fitting_functions['f_vy_ux']((x_ptl, z_ptl))
    # vz_ux_ptl = fitting_functions['f_vz_ux']((x_ptl, z_ptl))
    # vz_uy_ptl = fitting_functions['f_vz_uy']((x_ptl, z_ptl))

    # pxx = (vxp * uxp) * pmass + vx_ux_ptl * pmass - vx_ptl * uxp * pmass - vxp * ux_ptl * pmass
    # pyy = (vyp * uyp) * pmass + vy_uy_ptl * pmass - vy_ptl * uyp * pmass - vyp * uy_ptl * pmass
    # pzz = (vzp * uzp) * pmass + vz_uz_ptl * pmass - vz_ptl * uzp * pmass - vzp * uz_ptl * pmass
    # pxy = (vxp * uyp) * pmass + vx_uy_ptl * pmass - vx_ptl * uyp * pmass - vxp * uy_ptl * pmass
    # pxz = (vxp * uzp) * pmass + vx_uz_ptl * pmass - vx_ptl * uzp * pmass - vxp * uz_ptl * pmass
    # pyz = (vyp * uzp) * pmass + vy_uz_ptl * pmass - vy_ptl * uzp * pmass - vyp * uz_ptl * pmass
    # pyx = (vyp * uxp) * pmass + vy_ux_ptl * pmass - vy_ptl * uxp * pmass - vyp * ux_ptl * pmass
    # pzx = (vzp * uxp) * pmass + vz_ux_ptl * pmass - vz_ptl * uxp * pmass - vzp * ux_ptl * pmass
    # pzy = (vzp * uyp) * pmass + vz_uy_ptl * pmass - vz_ptl * uyp * pmass - vzp * uy_ptl * pmass
    pxx = (vxp - vx_ptl) * (uxp - ux_ptl) * pmass
    pyy = (vyp - vy_ptl) * (uyp - uy_ptl) * pmass
    pzz = (vzp - vz_ptl) * (uzp - uz_ptl) * pmass
    pxy = (vxp - vx_ptl) * (uyp - uy_ptl) * pmass
    pxz = (vxp - vx_ptl) * (uzp - uz_ptl) * pmass
    pyz = (vyp - vy_ptl) * (uzp - uz_ptl) * pmass
    pyx = (vyp - vy_ptl) * (uxp - ux_ptl) * pmass
    pzx = (vzp - vz_ptl) * (uxp - ux_ptl) * pmass
    pzy = (vzp - vz_ptl) * (uyp - uy_ptl) * pmass
    pxx *= weight
    pyy *= weight
    pzz *= weight
    pxy *= weight
    pxz *= weight
    pyz *= weight
    pyx *= weight
    pzx *= weight
    pzy *= weight

    del vxp, vyp, vzp, uxp, uyp, uzp
    del vx_ptl, vy_ptl, vz_ptl
    del ux_ptl, uy_ptl, uz_ptl
    # del vx_ux_ptl, vy_uy_ptl, vz_uz_ptl, vx_uy_ptl, vx_uz_ptl, vy_uz_ptl
    # del vy_ux_ptl, vz_ux_ptl, vz_uy_ptl

    pscalar = (pxx + pyy + pzz) / 3.0
    ppara_ptl = pxx * bx2 + pyy * by2 + pzz * bz2 + \
                (pxy + pyx) * bxy + (pxz + pzx) * bxz + (pyz + pzy) * byz
    ppara_ptl *= ib2_ptl
    pperp_ptl = 0.5 * (pscalar * 3 - ppara_ptl)

    # divv_ptl = fitting_functions['f_divv'](x_ptl, z_ptl, grid=False)
    # div_vperp_ptl = fitting_functions['f_div_vperp'](x_ptl, z_ptl, grid=False)
    # bbsigma_perp_ptl = fitting_functions['f_bbsigma_perp'](x_ptl, z_ptl, grid=False)
    # dvperpx_dx_ptl = fitting_functions['f_dvperpx_dx'](x_ptl, z_ptl, grid=False)
    # dvperpy_dx_ptl = fitting_functions['f_dvperpy_dx'](x_ptl, z_ptl, grid=False)
    # dvperpz_dx_ptl = fitting_functions['f_dvperpz_dx'](x_ptl, z_ptl, grid=False)
    # dvperpx_dz_ptl = fitting_functions['f_dvperpx_dz'](x_ptl, z_ptl, grid=False)
    # dvperpy_dz_ptl = fitting_functions['f_dvperpy_dz'](x_ptl, z_ptl, grid=False)
    # dvperpz_dz_ptl = fitting_functions['f_dvperpz_dz'](x_ptl, z_ptl, grid=False)
    divv_ptl = fitting_functions['f_divv'](coord)
    div_vperp_ptl = fitting_functions['f_div_vperp'](coord)
    bbsigma_perp_ptl = fitting_functions['f_bbsigma_perp'](coord)
    dvperpx_dx_ptl = fitting_functions['f_dvperpx_dx'](coord)
    dvperpy_dx_ptl = fitting_functions['f_dvperpy_dx'](coord)
    dvperpz_dx_ptl = fitting_functions['f_dvperpz_dx'](coord)
    dvperpx_dz_ptl = fitting_functions['f_dvperpx_dz'](coord)
    dvperpy_dz_ptl = fitting_functions['f_dvperpy_dz'](coord)
    dvperpz_dz_ptl = fitting_functions['f_dvperpz_dz'](coord)

    bbsigma_perp_ptl = (dvperpx_dx_ptl - (1./3.) * div_vperp_ptl) * bx2 + \
            (-(1./3.) * div_vperp_ptl) * by2 + \
            (dvperpz_dz_ptl - (1./3.) * div_vperp_ptl) * bz2 + \
            dvperpy_dx_ptl * bxy + \
            (dvperpx_dz_ptl + dvperpz_dx_ptl) * bxz + dvperpy_dz_ptl * byz
    bbsigma_perp_ptl *= ib2_ptl

    del bx2, by2, bz2, bxy, bxz, byz, ib2_ptl

    pdivv = -pscalar * divv_ptl
    pdiv_vperp = -pscalar * div_vperp_ptl
    ptensor_dv = -(pxx * dvperpx_dx_ptl + pyx * dvperpy_dx_ptl + pzx * dvperpz_dx_ptl)
    ptensor_dv -= pxz * dvperpx_dz_ptl + pyz * dvperpy_dz_ptl + pzz * dvperpz_dz_ptl
    pshear = (pperp_ptl - ppara_ptl) * bbsigma_perp_ptl

    del bbsigma_perp_ptl, divv_ptl, div_vperp_ptl
    del pscalar, ppara_ptl, pperp_ptl
    del pxx, pyy, pzz, pxy, pxz, pyz, pyx, pzx, pzy
    del dvperpx_dx_ptl, dvperpy_dx_ptl, dvperpz_dx_ptl
    del dvperpx_dz_ptl, dvperpy_dz_ptl, dvperpz_dz_ptl

    # flux term
    # div_ptensor_vperp_ptl = fitting_functions['f_div_ptensor_vperp'](x_ptl, z_ptl, grid=False)
    # div_pperp_vperp_ptl = fitting_functions['f_div_pperp_vperp'](x_ptl, z_ptl, grid=False)
    div_ptensor_vperp_ptl = fitting_functions['f_div_ptensor_vperp'](coord)
    div_pperp_vperp_ptl = fitting_functions['f_div_pperp_vperp'](coord)

    div_ptensor_vperp_ptl *= weight
    div_pperp_vperp_ptl *= weight

    hists = np.zeros((11, nbins))

    # hists[0, :], bin_edges = np.histogram(gamma-1, bins=ebins, weights=np.squeeze(de_para))
    hists[0, :], bin_edges = np.histogram(gamma1-1, bins=ebins, weights=np.squeeze(de_para))
    hists[1, :], bin_edges = np.histogram(gamma-1, bins=ebins, weights=np.squeeze(de_perp))
    hists[2, :], bin_edges = np.histogram(gamma-1, bins=ebins, weights=np.squeeze(pdivv))
    hists[3, :], bin_edges = np.histogram(gamma-1, bins=ebins, weights=np.squeeze(pdiv_vperp))
    hists[4, :], bin_edges = np.histogram(gamma-1, bins=ebins, weights=np.squeeze(pshear))
    hists[5, :], bin_edges = np.histogram(gamma-1, bins=ebins, weights=np.squeeze(ptensor_dv))
    hists[6, :], bin_edges = np.histogram(gamma-1, bins=ebins, weights=np.squeeze(de_dudt))
    hists[7, :], bin_edges = np.histogram(gamma-1, bins=ebins, weights=np.squeeze(de_cons_mu))
    hists[8, :], bin_edges = np.histogram(gamma-1, bins=ebins,
                                          weights=np.squeeze(div_ptensor_vperp_ptl))
    hists[9, :], bin_edges = np.histogram(gamma-1, bins=ebins,
                                          weights=np.squeeze(div_pperp_vperp_ptl))
    hists[10, :], bin_edges = np.histogram(gamma-1, bins=ebins)

    del x_ptl, z_ptl, gamma, bin_edges
    del de_para, de_perp, pdivv, pdiv_vperp, pshear, ptensor_dv, de_dudt, de_cons_mu
    del div_ptensor_vperp_ptl, div_pperp_vperp_ptl
    del coord
    del fitting_functions

    return hists


def interp_particle_compression_single(pic_info, run_dir, run_name, tindex, tindex_pre,
                                       tindex_post, species='e'):
    """Use single field files to interpolate compression effects
    """
    if species == 'e':
        pmass = 1.0
        charge = -1.0
    else:
        pmass = pic_info.mime
        charge = 1.0
    mime = pic_info.mime
    smime = math.sqrt(mime)
    current_time = tindex / pic_info.fields_interval
    dv = pic_info.dx_di * pic_info.dz_di * pic_info.mime

    particle_dir = run_dir + 'particle/T.' + str(tindex) + '/'
    eparticle_name = particle_dir + 'eparticle.' + str(tindex)
    hparticle_name = particle_dir + 'hparticle.' + str(tindex)
    if species == 'e':
        fname = eparticle_name + '.0'
    else:
        fname = hparticle_name + '.0'
    # read particle data
    (v0, pheader, ptl) = read_particle_data(fname)
    dx = v0.dx
    dz = v0.dz
    dxh = dx * 0.5
    dzh = dz * 0.5
    nx_pic = pic_info.nx
    nz_pic = pic_info.nz
    lx_pic = pic_info.lx_di * smime
    lz_pic = pic_info.lz_di * smime
    x1 = np.linspace(-dxh, lx_pic + dxh, nx_pic + 2)
    x2 = np.linspace(-dx, lx_pic, nx_pic + 2)
    z1 = np.linspace(-dzh - 0.5 * lz_pic, 0.5 * lz_pic + dzh, nz_pic + 2)
    z2 = np.linspace(-dz - 0.5 * lz_pic, 0.5 * lz_pic, nz_pic + 2)
    points_x, points_z = np.broadcast_arrays(x2.reshape(-1, 1), z2)
    points = np.vstack((points_x.flatten(),
                        points_z.flatten())).T
    smin = [x2[0], z2[0]]
    smax = [x2[-1], z2[-1]]
    orders = [len(x2), len(z2)]
    del points_x, points_z

    ng = 3
    kernel = np.ones((ng, ng)) / float(ng * ng)

    fitting_functions = {}

    kwargs = {"current_time": current_time,
              "xl": 0, "xr": 200, "zb": -50, "zt": 50}
    # read velocity fields
    fname = run_dir + "data/v" + species + "x.gda"
    x, z, fdata = read_2d_fields(pic_info, fname, **kwargs)
    vx_pic = fill_boundary_values(fdata)
    fname = run_dir + "data/v" + species + "y.gda"
    x, z, fdata = read_2d_fields(pic_info, fname, **kwargs)
    vy_pic = fill_boundary_values(fdata)
    fname = run_dir + "data/v" + species + "z.gda"
    x, z, fdata = read_2d_fields(pic_info, fname, **kwargs)
    vz_pic = fill_boundary_values(fdata)
    fname = run_dir + "data/u" + species + "x.gda"
    x, z, fdata = read_2d_fields(pic_info, fname, **kwargs)
    ux_pic = fill_boundary_values(fdata)
    fname = run_dir + "data/u" + species + "y.gda"
    x, z, fdata = read_2d_fields(pic_info, fname, **kwargs)
    uy_pic = fill_boundary_values(fdata)
    fname = run_dir + "data/u" + species + "z.gda"
    x, z, fdata = read_2d_fields(pic_info, fname, **kwargs)
    uz_pic = fill_boundary_values(fdata)

    # This will be updated latter
    dux_dt = vx_pic * np.gradient(ux_pic, dx, axis=1) + \
             vz_pic * np.gradient(ux_pic, dz, axis=0)
    duy_dt = vx_pic * np.gradient(uy_pic, dx, axis=1) + \
             vz_pic * np.gradient(uy_pic, dz, axis=0)
    duz_dt = vx_pic * np.gradient(uz_pic, dx, axis=1) + \
             vz_pic * np.gradient(uz_pic, dz, axis=0)

    fname = run_dir + "data/n" + species + ".gda"
    x, z, fdata = read_2d_fields(pic_info, fname, **kwargs)
    nrho_pic = fill_boundary_values(fdata)

    # dux_dt = vx_pic * np.gradient(nrho_pic * ux_pic, dx, axis=1) + \
    #          vz_pic * np.gradient(nrho_pic * ux_pic, dz, axis=0)
    # duy_dt = vx_pic * np.gradient(nrho_pic * uy_pic, dx, axis=1) + \
    #          vz_pic * np.gradient(nrho_pic * uy_pic, dz, axis=0)
    # duz_dt = vx_pic * np.gradient(nrho_pic * uz_pic, dx, axis=1) + \
    #          vz_pic * np.gradient(nrho_pic * uz_pic, dz, axis=0)

    order = 1

    # fitting_functions['f_vx'] = RectBivariateSpline(x2, z2, vx_pic.T, kx=order, ky=order)
    # fitting_functions['f_vy'] = RectBivariateSpline(x2, z2, vy_pic.T, kx=order, ky=order)
    # fitting_functions['f_vz'] = RectBivariateSpline(x2, z2, vz_pic.T, kx=order, ky=order)
    # fitting_functions['f_ux'] = RectBivariateSpline(x2, z2, ux_pic.T, kx=order, ky=order)
    # fitting_functions['f_uy'] = RectBivariateSpline(x2, z2, uy_pic.T, kx=order, ky=order)
    # fitting_functions['f_uz'] = RectBivariateSpline(x2, z2, uz_pic.T, kx=order, ky=order)
    # fitting_functions['f_vx'] = RegularGridInterpolator((x2, z2), vx_pic.T)
    # fitting_functions['f_vy'] = RegularGridInterpolator((x2, z2), vy_pic.T)
    # fitting_functions['f_vz'] = RegularGridInterpolator((x2, z2), vz_pic.T)
    # fitting_functions['f_ux'] = RegularGridInterpolator((x2, z2), ux_pic.T)
    # fitting_functions['f_uy'] = RegularGridInterpolator((x2, z2), uy_pic.T)
    # fitting_functions['f_uz'] = RegularGridInterpolator((x2, z2), uz_pic.T)
    # fitting_functions['f_vx'] = LinearNDInterpolator(points, np.transpose(vx_pic).flatten())
    # fitting_functions['f_vy'] = LinearNDInterpolator(points, np.transpose(vy_pic).flatten())
    # fitting_functions['f_vz'] = LinearNDInterpolator(points, np.transpose(vz_pic).flatten())
    # fitting_functions['f_ux'] = LinearNDInterpolator(points, np.transpose(ux_pic).flatten())
    # fitting_functions['f_uy'] = LinearNDInterpolator(points, np.transpose(uy_pic).flatten())
    # fitting_functions['f_uz'] = LinearNDInterpolator(points, np.transpose(uz_pic).flatten())
    fitting_functions['f_vx'] = MultilinearInterpolator(smin, smax, orders)
    fitting_functions['f_vy'] = MultilinearInterpolator(smin, smax, orders)
    fitting_functions['f_vz'] = MultilinearInterpolator(smin, smax, orders)
    fitting_functions['f_ux'] = MultilinearInterpolator(smin, smax, orders)
    fitting_functions['f_uy'] = MultilinearInterpolator(smin, smax, orders)
    fitting_functions['f_uz'] = MultilinearInterpolator(smin, smax, orders)
    fitting_functions['f_vx'].set_values(np.atleast_2d(np.transpose(vx_pic).flatten()))
    fitting_functions['f_vy'].set_values(np.atleast_2d(np.transpose(vy_pic).flatten()))
    fitting_functions['f_vz'].set_values(np.atleast_2d(np.transpose(vz_pic).flatten()))
    fitting_functions['f_ux'].set_values(np.atleast_2d(np.transpose(ux_pic).flatten()))
    fitting_functions['f_uy'].set_values(np.atleast_2d(np.transpose(uy_pic).flatten()))
    fitting_functions['f_uz'].set_values(np.atleast_2d(np.transpose(uz_pic).flatten()))

    # fitting_functions['f_vx_ux'] = RegularGridInterpolator((x2, z2), vx_pic.T*ux_pic.T)
    # fitting_functions['f_vy_uy'] = RegularGridInterpolator((x2, z2), vy_pic.T*uy_pic.T)
    # fitting_functions['f_vz_uz'] = RegularGridInterpolator((x2, z2), vz_pic.T*uz_pic.T)
    # fitting_functions['f_vx_uy'] = RegularGridInterpolator((x2, z2), vx_pic.T*uy_pic.T)
    # fitting_functions['f_vx_uz'] = RegularGridInterpolator((x2, z2), vx_pic.T*uz_pic.T)
    # fitting_functions['f_vy_uz'] = RegularGridInterpolator((x2, z2), vy_pic.T*uz_pic.T)
    # fitting_functions['f_vy_ux'] = RegularGridInterpolator((x2, z2), vy_pic.T*ux_pic.T)
    # fitting_functions['f_vz_ux'] = RegularGridInterpolator((x2, z2), vz_pic.T*ux_pic.T)
    # fitting_functions['f_vz_uy'] = RegularGridInterpolator((x2, z2), vz_pic.T*uy_pic.T)

    # read electric and magnetic fields
    nx = pic_info.nx
    nz = pic_info.nz
    fname = run_dir + "data/ex.gda"
    x, z, fdata = read_2d_fields(pic_info, fname, **kwargs)
    ex = fill_boundary_values(fdata)
    # ex[:, 1:nx+1] = (ex[:, 0:nx] + ex[:, 1:nx+1]) * 0.5

    fname = run_dir + "data/ey.gda"
    x, z, fdata = read_2d_fields(pic_info, fname, **kwargs)
    ey = fill_boundary_values(fdata)

    fname = run_dir + "data/ez.gda"
    x, z, fdata = read_2d_fields(pic_info, fname, **kwargs)
    ez = fill_boundary_values(fdata)
    # ez[1:nz+1, :] = (ez[0:nz, :] + ez[1:nz+1, :]) * 0.5

    fname = run_dir + "data/bx.gda"
    x, z, fdata = read_2d_fields(pic_info, fname, **kwargs)
    bx = fill_boundary_values(fdata)
    # bx[1:nz+1, :] = (bx[0:nz, :] + bx[1:nz+1, :]) * 0.5

    fname = run_dir + "data/by.gda"
    x, z, fdata = read_2d_fields(pic_info, fname, **kwargs)
    by = fill_boundary_values(fdata)
    # by[1:nz+1, 1:nx+1] = (by[0:nz, 0:nx] + by[1:nz+1, 0:nx] +
    #                       by[0:nz, 1:nx+1] + by[1:nz+1, 1:nx+1]) * 0.25

    fname = run_dir + "data/bz.gda"
    x, z, fdata = read_2d_fields(pic_info, fname, **kwargs)
    bz = fill_boundary_values(fdata)
    # bz[:, 1:nx+1] = (bz[:, 0:nx] + bz[:, 1:nx+1]) * 0.5

    sigma = 3
    ex = median_filter(ex, sigma)
    ey = median_filter(ey, sigma)
    ez = median_filter(ez, sigma)

    # fitting_functions['f_ex'] = RectBivariateSpline(x2, z2, ex.T, kx=order, ky=order)
    # fitting_functions['f_ey'] = RectBivariateSpline(x2, z2, ey.T, kx=order, ky=order)
    # fitting_functions['f_ez'] = RectBivariateSpline(x2, z2, ez.T, kx=order, ky=order)
    # fitting_functions['f_bx'] = RectBivariateSpline(x2, z2, bx.T, kx=order, ky=order)
    # fitting_functions['f_by'] = RectBivariateSpline(x2, z2, by.T, kx=order, ky=order)
    # fitting_functions['f_bz'] = RectBivariateSpline(x2, z2, bz.T, kx=order, ky=order)
    # fitting_functions['f_ex'] = RegularGridInterpolator((x2, z2), ex.T)
    # fitting_functions['f_ey'] = RegularGridInterpolator((x2, z2), ey.T)
    # fitting_functions['f_ez'] = RegularGridInterpolator((x2, z2), ez.T)
    # fitting_functions['f_bx'] = RegularGridInterpolator((x2, z2), bx.T)
    # fitting_functions['f_by'] = RegularGridInterpolator((x2, z2), by.T)
    # fitting_functions['f_bz'] = RegularGridInterpolator((x2, z2), bz.T)
    # fitting_functions['f_ex'] = LinearNDInterpolator(points, np.transpose(ex).flatten())
    # fitting_functions['f_ey'] = LinearNDInterpolator(points, np.transpose(ey).flatten())
    # fitting_functions['f_ez'] = LinearNDInterpolator(points, np.transpose(ez).flatten())
    # fitting_functions['f_bx'] = LinearNDInterpolator(points, np.transpose(bx).flatten())
    # fitting_functions['f_by'] = LinearNDInterpolator(points, np.transpose(by).flatten())
    # fitting_functions['f_bz'] = LinearNDInterpolator(points, np.transpose(bz).flatten())
    fitting_functions['f_ex'] = MultilinearInterpolator(smin, smax, orders)
    fitting_functions['f_ey'] = MultilinearInterpolator(smin, smax, orders)
    fitting_functions['f_ez'] = MultilinearInterpolator(smin, smax, orders)
    fitting_functions['f_bx'] = MultilinearInterpolator(smin, smax, orders)
    fitting_functions['f_by'] = MultilinearInterpolator(smin, smax, orders)
    fitting_functions['f_bz'] = MultilinearInterpolator(smin, smax, orders)
    fitting_functions['f_ex'].set_values(np.atleast_2d(np.transpose(ex).flatten()))
    fitting_functions['f_ey'].set_values(np.atleast_2d(np.transpose(ey).flatten()))
    fitting_functions['f_ez'].set_values(np.atleast_2d(np.transpose(ez).flatten()))
    fitting_functions['f_bx'].set_values(np.atleast_2d(np.transpose(bx).flatten()))
    fitting_functions['f_by'].set_values(np.atleast_2d(np.transpose(by).flatten()))
    fitting_functions['f_bz'].set_values(np.atleast_2d(np.transpose(bz).flatten()))

    # exb drift velocity
    absB = np.sqrt(bx**2 + by**2 + bz**2)
    ib2 = div0(1.0, absB**2)
    vx = (ey * bz - ez * by) * ib2
    vy = (ez * bx - ex * bz) * ib2
    vz = (ex * by - ey * bx) * ib2
    vxb = vx * bx + vy * by + vz * bz
    vx_perp = vx - vxb * bx * ib2
    vy_perp = vy - vxb * by * ib2
    vz_perp = vz - vxb * bz * ib2

    divv = np.gradient(vx, dx, axis=1) + np.gradient(vz, dz, axis=0)
    div_vperp = np.gradient(vx_perp, dx, axis=1) + np.gradient(vz_perp, dz, axis=0)

    del ex, ey, ez, vxb
    del vx, vy, vz

    # read pressure fields
    fname = run_dir + "data/p" + species + "-xx.gda"
    x, z, fdata = read_2d_fields(pic_info, fname, **kwargs)
    pxx_pic = fill_boundary_values(fdata)
    fname = run_dir + "data/p" + species + "-yy.gda"
    x, z, fdata = read_2d_fields(pic_info, fname, **kwargs)
    pyy_pic = fill_boundary_values(fdata)
    fname = run_dir + "data/p" + species + "-zz.gda"
    x, z, fdata = read_2d_fields(pic_info, fname, **kwargs)
    pzz_pic = fill_boundary_values(fdata)
    fname = run_dir + "data/p" + species + "-xy.gda"
    x, z, fdata = read_2d_fields(pic_info, fname, **kwargs)
    pxy_pic = fill_boundary_values(fdata)
    fname = run_dir + "data/p" + species + "-xz.gda"
    x, z, fdata = read_2d_fields(pic_info, fname, **kwargs)
    pxz_pic = fill_boundary_values(fdata)
    fname = run_dir + "data/p" + species + "-yz.gda"
    x, z, fdata = read_2d_fields(pic_info, fname, **kwargs)
    pyz_pic = fill_boundary_values(fdata)
    fname = run_dir + "data/p" + species + "-yx.gda"
    x, z, fdata = read_2d_fields(pic_info, fname, **kwargs)
    pyx_pic = fill_boundary_values(fdata)
    fname = run_dir + "data/p" + species + "-zx.gda"
    x, z, fdata = read_2d_fields(pic_info, fname, **kwargs)
    pzx_pic = fill_boundary_values(fdata)
    fname = run_dir + "data/p" + species + "-zy.gda"
    x, z, fdata = read_2d_fields(pic_info, fname, **kwargs)
    pzy_pic = fill_boundary_values(fdata)

    ppara = pxx_pic*bx**2 + pyy_pic*by**2 + pzz_pic*bz**2 + \
            (pxy_pic + pyx_pic)*bx*by + (pxz_pic + pzx_pic)*bx*bz + \
            (pyz_pic + pzy_pic)*by*bz
    ppara *= ib2
    pperp = 0.5 * (pxx_pic + pyy_pic + pzz_pic - ppara)

    div_ptensor_vperp = np.gradient(pxx_pic*vx_perp + pyx_pic*vy_perp +
                                    pzx_pic*vz_perp, dx, axis=1) + \
                        np.gradient(pxz_pic*vx_perp + pyz_pic*vy_perp +
                                    pzz_pic*vz_perp, dz, axis=0)
    div_pperp_vperp = np.gradient(pperp * vx_perp, dx, axis=1) + \
                      np.gradient(pperp * vz_perp, dz, axis=0)
    # div_ptensor_vperp = div0(div_ptensor_vperp, nrho_pic)
    # div_pperp_vperp = div0(div_pperp_vperp, nrho_pic)

    # fitting_functions['f_div_ptensor_vperp'] = RectBivariateSpline(
    #         x2, z2, div_ptensor_vperp.T, kx=order, ky=order)
    # fitting_functions['f_div_pperp_vperp'] = RectBivariateSpline(
    #         x2, z2, div_pperp_vperp.T, kx=order, ky=order)
    # fitting_functions['f_div_ptensor_vperp'] = RegularGridInterpolator(
    #         (x2, z2), div_ptensor_vperp.T)
    # fitting_functions['f_div_pperp_vperp'] = RegularGridInterpolator(
    #         (x2, z2), div_pperp_vperp.T)
    # fitting_functions['f_div_ptensor_vperp'] = LinearNDInterpolator(
    #         points, np.transpose(div_ptensor_vperp).flatten())
    # fitting_functions['f_div_pperp_vperp'] = LinearNDInterpolator(
    #         points, np.transpose(div_pperp_vperp).flatten())
    fitting_functions['f_div_ptensor_vperp'] = MultilinearInterpolator(smin, smax, orders)
    fitting_functions['f_div_pperp_vperp'] = MultilinearInterpolator(smin, smax, orders)
    fitting_functions['f_div_ptensor_vperp'].set_values(
            np.atleast_2d(np.transpose(div_ptensor_vperp).flatten()))
    fitting_functions['f_div_pperp_vperp'].set_values(
            np.atleast_2d(np.transpose(div_pperp_vperp).flatten()))
    del div_ptensor_vperp, div_pperp_vperp

    del ppara, pperp
    del pxx_pic, pyy_pic, pzz_pic
    del pxy_pic, pxz_pic, pyz_pic
    del pyx_pic, pzx_pic, pzy_pic

    dvperpx_dx = np.gradient(vx_perp, dx, axis=1)
    dvperpy_dx = np.gradient(vy_perp, dx, axis=1)
    dvperpz_dx = np.gradient(vz_perp, dx, axis=1)
    dvperpx_dz = np.gradient(vx_perp, dz, axis=0)
    dvperpy_dz = np.gradient(vy_perp, dz, axis=0)
    dvperpz_dz = np.gradient(vz_perp, dz, axis=0)

    bbsigma_perp = (dvperpx_dx - (1./3.) * div_vperp) * bx**2 + \
                   (-(1./3.) * div_vperp) * by**2 + \
                   (dvperpz_dz - (1./3.) * div_vperp) * bz**2 + \
                   dvperpy_dx * bx * by + \
                   (dvperpx_dz + dvperpz_dx) * bx * bz + dvperpy_dz * by * bz
    bbsigma_perp *= ib2

    # fitting_functions['f_divv'] = RectBivariateSpline(x2, z2, divv.T, kx=order, ky=order)
    # fitting_functions['f_div_vperp'] = RectBivariateSpline(x2, z2, div_vperp.T,
    #                                                        kx=order, ky=order)
    # fitting_functions['f_bbsigma_perp'] = RectBivariateSpline(x2, z2, bbsigma_perp.T,
    #                                                           kx=order, ky=order)
    # fitting_functions['f_dvperpx_dx'] = RectBivariateSpline(x2, z2, dvperpx_dx.T,
    #                                                         kx=order, ky=order)
    # fitting_functions['f_dvperpy_dx'] = RectBivariateSpline(x2, z2, dvperpy_dx.T,
    #                                                         kx=order, ky=order)
    # fitting_functions['f_dvperpz_dx'] = RectBivariateSpline(x2, z2, dvperpz_dx.T,
    #                                                         kx=order, ky=order)
    # fitting_functions['f_dvperpx_dz'] = RectBivariateSpline(x2, z2, dvperpx_dz.T,
    #                                                         kx=order, ky=order)
    # fitting_functions['f_dvperpy_dz'] = RectBivariateSpline(x2, z2, dvperpy_dz.T,
    #                                                         kx=order, ky=order)
    # fitting_functions['f_dvperpz_dz'] = RectBivariateSpline(x2, z2, dvperpz_dz.T,
    #                                                         kx=order, ky=order)
    # fitting_functions['f_divv'] = RegularGridInterpolator((x2, z2), divv.T)
    # fitting_functions['f_div_vperp'] = RegularGridInterpolator((x2, z2), div_vperp.T)
    # fitting_functions['f_bbsigma_perp'] = RegularGridInterpolator((x2, z2), bbsigma_perp.T)
    # fitting_functions['f_dvperpx_dx'] = RegularGridInterpolator((x2, z2), dvperpx_dx.T)
    # fitting_functions['f_dvperpy_dx'] = RegularGridInterpolator((x2, z2), dvperpy_dx.T)
    # fitting_functions['f_dvperpz_dx'] = RegularGridInterpolator((x2, z2), dvperpz_dx.T)
    # fitting_functions['f_dvperpx_dz'] = RegularGridInterpolator((x2, z2), dvperpx_dz.T)
    # fitting_functions['f_dvperpy_dz'] = RegularGridInterpolator((x2, z2), dvperpy_dz.T)
    # fitting_functions['f_dvperpz_dz'] = RegularGridInterpolator((x2, z2), dvperpz_dz.T)
    # fitting_functions['f_divv'] = LinearNDInterpolator(points, np.transpose(divv).flatten())
    # fitting_functions['f_div_vperp'] = LinearNDInterpolator(
    #         points, np.transpose(div_vperp).flatten())
    # fitting_functions['f_bbsigma_perp'] = LinearNDInterpolator(
    #         points, np.transpose(bbsigma_perp).flatten())
    # fitting_functions['f_dvperpx_dx'] = LinearNDInterpolator(
    #         points, np.transpose(dvperpx_dx).flatten())
    # fitting_functions['f_dvperpy_dx'] = LinearNDInterpolator(
    #         points, np.transpose(dvperpy_dx).flatten())
    # fitting_functions['f_dvperpz_dx'] = LinearNDInterpolator(
    #         points, np.transpose(dvperpz_dx).flatten())
    # fitting_functions['f_dvperpx_dz'] = LinearNDInterpolator(
    #         points, np.transpose(dvperpx_dz).flatten())
    # fitting_functions['f_dvperpy_dz'] = LinearNDInterpolator(
    #         points, np.transpose(dvperpy_dz).flatten())
    # fitting_functions['f_dvperpz_dz'] = LinearNDInterpolator(
    #         points, np.transpose(dvperpz_dz).flatten())

    fitting_functions['f_divv'] = MultilinearInterpolator(smin, smax, orders)
    fitting_functions['f_div_vperp'] = MultilinearInterpolator(smin, smax, orders)
    fitting_functions['f_bbsigma_perp'] = MultilinearInterpolator(smin, smax, orders)
    fitting_functions['f_dvperpx_dx'] = MultilinearInterpolator(smin, smax, orders)
    fitting_functions['f_dvperpy_dx'] = MultilinearInterpolator(smin, smax, orders)
    fitting_functions['f_dvperpz_dx'] = MultilinearInterpolator(smin, smax, orders)
    fitting_functions['f_dvperpx_dz'] = MultilinearInterpolator(smin, smax, orders)
    fitting_functions['f_dvperpy_dz'] = MultilinearInterpolator(smin, smax, orders)
    fitting_functions['f_dvperpz_dz'] = MultilinearInterpolator(smin, smax, orders)
    fitting_functions['f_divv'].set_values(np.atleast_2d(np.transpose(divv).flatten()))
    fitting_functions['f_div_vperp'].set_values(np.atleast_2d(np.transpose(div_vperp).flatten()))
    fitting_functions['f_bbsigma_perp'].set_values(np.atleast_2d(
            np.transpose(bbsigma_perp).flatten()))
    fitting_functions['f_dvperpx_dx'].set_values(np.atleast_2d(np.transpose(dvperpx_dx).flatten()))
    fitting_functions['f_dvperpy_dx'].set_values(np.atleast_2d(np.transpose(dvperpy_dx).flatten()))
    fitting_functions['f_dvperpz_dx'].set_values(np.atleast_2d(np.transpose(dvperpz_dx).flatten()))
    fitting_functions['f_dvperpx_dz'].set_values(np.atleast_2d(np.transpose(dvperpx_dz).flatten()))
    fitting_functions['f_dvperpy_dz'].set_values(np.atleast_2d(np.transpose(dvperpy_dz).flatten()))
    fitting_functions['f_dvperpz_dz'].set_values(np.atleast_2d(np.transpose(dvperpz_dz).flatten()))

    del divv, div_vperp, bbsigma_perp
    del dvperpx_dx, dvperpy_dx, dvperpz_dx
    del dvperpx_dz, dvperpy_dz, dvperpz_dz
    del bx, by, bz, absB, ib2

    # read data from previous and next time step
    fname = run_dir + "data/u" + species + "x_pre.gda"
    x, z, fdata = read_2d_fields(pic_info, fname, **kwargs)
    ux_pre = fill_boundary_values(fdata)
    fname = run_dir + "data/u" + species + "y_pre.gda"
    x, z, fdata = read_2d_fields(pic_info, fname, **kwargs)
    uy_pre = fill_boundary_values(fdata)
    fname = run_dir + "data/u" + species + "z_pre.gda"
    x, z, fdata = read_2d_fields(pic_info, fname, **kwargs)
    uz_pre = fill_boundary_values(fdata)
    fname = run_dir + "data/n" + species + "_pre.gda"
    x, z, fdata = read_2d_fields(pic_info, fname, **kwargs)
    nrho_pre = fill_boundary_values(fdata)
    fname = run_dir + "data/absB_pre.gda"
    x, z, fdata = read_2d_fields(pic_info, fname, **kwargs)
    absB_pre = fill_boundary_values(fdata)
    fname = run_dir + "data/u" + species + "x_post.gda"
    x, z, fdata = read_2d_fields(pic_info, fname, **kwargs)
    ux_post = fill_boundary_values(fdata)
    fname = run_dir + "data/u" + species + "y_post.gda"
    x, z, fdata = read_2d_fields(pic_info, fname, **kwargs)
    uy_post = fill_boundary_values(fdata)
    fname = run_dir + "data/u" + species + "z_post.gda"
    x, z, fdata = read_2d_fields(pic_info, fname, **kwargs)
    uz_post = fill_boundary_values(fdata)
    fname = run_dir + "data/n" + species + "_post.gda"
    x, z, fdata = read_2d_fields(pic_info, fname, **kwargs)
    nrho_post = fill_boundary_values(fdata)
    fname = run_dir + "data/absB_post.gda"
    x, z, fdata = read_2d_fields(pic_info, fname, **kwargs)
    absB_post = fill_boundary_values(fdata)
    fname = run_dir + "data/ke-" + species + ".gda"
    x, z, fdata = read_2d_fields(pic_info, fname, **kwargs)
    ke_pic = fill_boundary_values(fdata)

    fname = run_dir + "data/ex_pre.gda"
    x, z, fdata = read_2d_fields(pic_info, fname, **kwargs)
    ex_pre = fill_boundary_values(fdata)
    # ex_pre[:, 1:nx+1] = (ex_pre[:, 0:nx] + ex_pre[:, 1:nx+1]) * 0.5

    fname = run_dir + "data/ey_pre.gda"
    x, z, fdata = read_2d_fields(pic_info, fname, **kwargs)
    ey_pre = fill_boundary_values(fdata)

    fname = run_dir + "data/ez_pre.gda"
    x, z, fdata = read_2d_fields(pic_info, fname, **kwargs)
    ez_pre = fill_boundary_values(fdata)
    # ez_pre[1:nz+1, :] = (ez_pre[0:nz, :] + ez_pre[1:nz+1, :]) * 0.5

    fname = run_dir + "data/bx_pre.gda"
    x, z, fdata = read_2d_fields(pic_info, fname, **kwargs)
    bx_pre = fill_boundary_values(fdata)
    # bx_pre[1:nz+1, :] = (bx_pre[0:nz, :] + bx_pre[1:nz+1, :]) * 0.5

    fname = run_dir + "data/by_pre.gda"
    x, z, fdata = read_2d_fields(pic_info, fname, **kwargs)
    by_pre = fill_boundary_values(fdata)
    # by_pre[1:nz+1, 1:nx+1] = (by_pre[0:nz, 0:nx] + by_pre[1:nz+1, 0:nx] +
    #                           by_pre[0:nz, 1:nx+1] + by_pre[1:nz+1, 1:nx+1]) * 0.25

    fname = run_dir + "data/bz_pre.gda"
    x, z, fdata = read_2d_fields(pic_info, fname, **kwargs)
    bz_pre = fill_boundary_values(fdata)
    # bz_pre[:, 1:nx+1] = (bz_pre[:, 0:nx] + bz_pre[:, 1:nx+1]) * 0.5

    ib2_pre = div0(1.0, absB_pre**2)
    ex_pre = median_filter(ex_pre, sigma)
    ey_pre = median_filter(ey_pre, sigma)
    ez_pre = median_filter(ez_pre, sigma)
    vexb_pre_x = (ey_pre * bz_pre - ez_pre * by_pre) * ib2_pre
    vexb_pre_y = (ez_pre * bx_pre - ex_pre * bz_pre) * ib2_pre
    vexb_pre_z = (ex_pre * by_pre - ey_pre * bx_pre) * ib2_pre

    del ex_pre, ey_pre, ez_pre
    del bx_pre, by_pre, bz_pre
    del ib2_pre

    fname = run_dir + "data/ex_post.gda"
    x, z, fdata = read_2d_fields(pic_info, fname, **kwargs)
    ex_post = fill_boundary_values(fdata)
    # ex_post[:, 1:nx+1] = (ex_post[:, 0:nx] + ex_post[:, 1:nx+1]) * 0.5

    fname = run_dir + "data/ey_post.gda"
    x, z, fdata = read_2d_fields(pic_info, fname, **kwargs)
    ey_post = fill_boundary_values(fdata)

    fname = run_dir + "data/ez_post.gda"
    x, z, fdata = read_2d_fields(pic_info, fname, **kwargs)
    ez_post = fill_boundary_values(fdata)
    # ez_post[1:nz+1, :] = (ez_post[0:nz, :] + ez_post[1:nz+1, :]) * 0.5

    fname = run_dir + "data/bx_post.gda"
    x, z, fdata = read_2d_fields(pic_info, fname, **kwargs)
    bx_post = fill_boundary_values(fdata)
    # bx_post[1:nz+1, :] = (bx_post[0:nz, :] + bx_post[1:nz+1, :]) * 0.5

    fname = run_dir + "data/by_post.gda"
    x, z, fdata = read_2d_fields(pic_info, fname, **kwargs)
    by_post = fill_boundary_values(fdata)
    # by_post[1:nz+1, 1:nx+1] = (by_post[0:nz, 0:nx] + by_post[1:nz+1, 0:nx] +
    #                            by_post[0:nz, 1:nx+1] + by_post[1:nz+1, 1:nx+1]) * 0.25

    fname = run_dir + "data/bz_post.gda"
    x, z, fdata = read_2d_fields(pic_info, fname, **kwargs)
    bz_post = fill_boundary_values(fdata)
    # bz_post[:, 1:nx+1] = (bz_post[:, 0:nx] + bz_post[:, 1:nx+1]) * 0.5

    ib2_post = div0(1.0, absB_post**2)
    ex_post = median_filter(ex_post, sigma)
    ey_post = median_filter(ey_post, sigma)
    ez_post = median_filter(ez_post, sigma)
    vexb_post_x = (ey_post * bz_post - ez_post * by_post) * ib2_post
    vexb_post_y = (ez_post * bx_post - ex_post * bz_post) * ib2_post
    vexb_post_z = (ex_post * by_post - ey_post * bx_post) * ib2_post

    del ex_post, ey_post, ez_post
    del bx_post, by_post, bz_post
    del ib2_post
    del fdata

    dtf = pic_info.dtwpe * (tindex_post - tindex_pre)
    dux_dt += (ux_post - ux_pre) / dtf
    duy_dt += (uy_post - uy_pre) / dtf
    duz_dt += (uz_post - uz_pre) / dtf
    # de_dudt = dux_dt * vx_perp + duy_dt * vy_perp + duz_dt * vz_perp
    # de_dudt = dux_dt * vx_pic + duy_dt * vy_pic + duz_dt * vz_pic
    # de_dudt *= nrho_pic * pmass
    # print("Min and max: %f %f" % (np.min(de_dudt), np.max(de_dudt)))
    # print np.sum(de_dudt) * dv
    # vmin = -0.1
    # vmax = -vmin
    # # plt.imshow(jpolar_dote, cmap=plt.cm.seismic, vmin=vmin, vmax=vmax)
    # de_dudt_cum = np.cumsum(np.sum(de_dudt, axis=0))
    # plt.plot(de_dudt_cum * dv, linewidth=2)

    # # dux_dt = vx_perp * dux_dt + vy_perp * duy_dt + vz_perp * duz_dt
    # # dux_dt += (ux_post * nrho_post - ux_pre * nrho_pre) / dtf
    # # duy_dt += (uy_post * nrho_post - uy_pre * nrho_pre) / dtf
    # # duz_dt += (uz_post * nrho_post - uz_pre * nrho_pre) / dtf
    # dux_dt = (ux_post * nrho_post - ux_pre * nrho_pre) / dtf
    # duy_dt = (uy_post * nrho_post - uy_pre * nrho_pre) / dtf
    # duz_dt = (uz_post * nrho_post - uz_pre * nrho_pre) / dtf
    # dux_dt = div0(dux_dt, ke_pic / pmass)
    # duy_dt = div0(duy_dt, ke_pic / pmass)
    # duz_dt = div0(duz_dt, ke_pic / pmass)
    divv_species = np.gradient(vx_pic, dx, axis=1) + \
                   np.gradient(vz_pic, dz, axis=0)
    # dux_dt += ux_pic * divv_species
    # duy_dt += uy_pic * divv_species
    # duz_dt += uz_pic * divv_species
    # dux_dt = vx_perp * dux_dt + vy_perp * duy_dt + vz_perp * duz_dt

    # idt = 1.0 / dtf
    # dux_dt = ((vexb_post_x - vexb_pre_x) * idt +
    #           vx_pic * np.gradient(vx_perp, dx, axis=1) +
    #           vz_pic * np.gradient(vx_perp, dz, axis=0))
    # duy_dt = ((vexb_post_y - vexb_pre_y) * idt +
    #           vx_pic * np.gradient(vy_perp, dx, axis=1) +
    #           vz_pic * np.gradient(vy_perp, dz, axis=0))
    # duz_dt = ((vexb_post_z - vexb_pre_z) * idt +
    #           vx_pic * np.gradient(vz_perp, dx, axis=1) +
    #           vz_pic * np.gradient(vz_perp, dz, axis=0))
    # udot_vexb = ux_pic * vx_perp + uy_pic * vy_perp + uz_pic * vz_perp
    # udot_vexb_pre = (ux_pre * vexb_pre_x + uy_pre * vexb_pre_y +
    #                  uz_pre * vexb_pre_z)
    # udot_vexb_post = (ux_post * vexb_post_x + uy_post * vexb_post_y +
    #                   uz_post * vexb_post_z)

    # divv_species = (udot_vexb_post - udot_vexb_pre) * idt
    # divv_species += (vx_pic * np.gradient(udot_vexb, dx, axis=1) +
    #                  vz_pic * np.gradient(udot_vexb, dz, axis=0))
    # # jpolar_dote = divv_species * nrho_pic * pmass
    # jpolar_dote = (divv_species - ux_pic * dux_dt - uy_pic * duy_dt -
    #                uz_pic * duz_dt) * nrho_pic * pmass
    # print("Min and max: %f %f" % (np.min(jpolar_dote), np.max(jpolar_dote)))
    # print np.sum(jpolar_dote) * dv
    # vmin = -0.1
    # vmax = -vmin
    # # plt.imshow(jpolar_dote, cmap=plt.cm.seismic, vmin=vmin, vmax=vmax)
    # jpolar_dote_cum = np.cumsum(np.sum(jpolar_dote, axis=0))
    # plt.plot(jpolar_dote_cum * dv, linewidth=2)

    del vexb_pre_x, vexb_pre_y, vexb_pre_z
    del vexb_post_x, vexb_post_y, vexb_post_z
    # del udot_vexb, udot_vexb_pre, udot_vexb_post

    del vx_pic, vy_pic, vz_pic
    del vx_perp, vy_perp, vz_perp

    # dux_dt = div0(dux_dt * nrho_pic, ke_pic) * pmass
    # duy_dt = div0(duy_dt * nrho_pic, ke_pic) * pmass
    # duz_dt = div0(duz_dt * nrho_pic, ke_pic) * pmass
    # dux_dx = np.gradient(ux_pic, dx, axis=1)
    # dux_dz = np.gradient(ux_pic, dz, axis=0)
    # duy_dx = np.gradient(uy_pic, dx, axis=1)
    # duy_dz = np.gradient(uy_pic, dz, axis=0)
    # duz_dx = np.gradient(uz_pic, dx, axis=1)
    # duz_dz = np.gradient(uz_pic, dz, axis=0)
    # ng = 3
    # kernel = np.ones((ng, ng)) / float(ng * ng)
    # absB_pre = signal.convolve2d(absB_pre, kernel, 'same')
    # absB_post = signal.convolve2d(absB_post, kernel, 'same')
    db_dt = (absB_post - absB_pre) / dtf
    # db_dt = np.gradient(ey, dz, axis=0) * bx - \
    #         (np.gradient(ex, dz, axis=0) - np.gradient(ez, dx, axis=1)) * by - \
    #         np.gradient(ey, dx, axis=1) * bz
    # db_dt /= absB

    del ux_pic, uy_pic, uz_pic
    del ux_pre, uy_pre, uz_pre
    del ux_post, uy_post, uz_post
    del absB_pre, absB_post
    del nrho_pre, nrho_post
    del ke_pic, nrho_pic

    # fitting_functions['f_dux_dt'] = RectBivariateSpline(x2, z2, dux_dt.T, kx=order, ky=order)
    # fitting_functions['f_duy_dt'] = RectBivariateSpline(x2, z2, duy_dt.T, kx=order, ky=order)
    # fitting_functions['f_duz_dt'] = RectBivariateSpline(x2, z2, duz_dt.T, kx=order, ky=order)
    # fitting_functions['f_db_dt']  = RectBivariateSpline(x2, z2, db_dt.T, kx=order, ky=order)
    # fitting_functions['f_dux_dt'] = RegularGridInterpolator((x2, z2), dux_dt.T)
    # fitting_functions['f_duy_dt'] = RegularGridInterpolator((x2, z2), duy_dt.T)
    # fitting_functions['f_duz_dt'] = RegularGridInterpolator((x2, z2), duz_dt.T)
    # fitting_functions['f_db_dt']  = RegularGridInterpolator((x2, z2), db_dt.T)
    # fitting_functions['f_dux_dt'] = LinearNDInterpolator(points, np.transpose(dux_dt).flatten())
    # fitting_functions['f_duy_dt'] = LinearNDInterpolator(points, np.transpose(duy_dt).flatten())
    # fitting_functions['f_duz_dt'] = LinearNDInterpolator(points, np.transpose(duz_dt).flatten())
    # fitting_functions['f_db_dt']  = LinearNDInterpolator(points, np.transpose(db_dt).flatten())
    fitting_functions['f_dux_dt'] = MultilinearInterpolator(smin, smax, orders)
    fitting_functions['f_duy_dt'] = MultilinearInterpolator(smin, smax, orders)
    fitting_functions['f_duz_dt'] = MultilinearInterpolator(smin, smax, orders)
    fitting_functions['f_db_dt'] = MultilinearInterpolator(smin, smax, orders)
    fitting_functions['f_dux_dt'].set_values(np.atleast_2d(np.transpose(dux_dt).flatten()))
    fitting_functions['f_duy_dt'].set_values(np.atleast_2d(np.transpose(duy_dt).flatten()))
    fitting_functions['f_duz_dt'].set_values(np.atleast_2d(np.transpose(duz_dt).flatten()))
    fitting_functions['f_db_dt'].set_values(np.atleast_2d(np.transpose(db_dt).flatten()))

    # fitting_functions['f_dux_dx'] = RegularGridInterpolator((x2, z2), dux_dx.T)
    # fitting_functions['f_duy_dx'] = RegularGridInterpolator((x2, z2), duy_dx.T)
    # fitting_functions['f_duz_dx'] = RegularGridInterpolator((x2, z2), duz_dx.T)
    # fitting_functions['f_dux_dz'] = RegularGridInterpolator((x2, z2), dux_dz.T)
    # fitting_functions['f_duy_dz'] = RegularGridInterpolator((x2, z2), duy_dz.T)
    # fitting_functions['f_duz_dz'] = RegularGridInterpolator((x2, z2), duz_dz.T)
    # fitting_functions['f_dux_dx'] = MultilinearInterpolator(smin, smax, orders)
    # fitting_functions['f_duy_dx'] = MultilinearInterpolator(smin, smax, orders)
    # fitting_functions['f_duz_dx'] = MultilinearInterpolator(smin, smax, orders)
    # fitting_functions['f_dux_dz'] = MultilinearInterpolator(smin, smax, orders)
    # fitting_functions['f_duy_dz'] = MultilinearInterpolator(smin, smax, orders)
    # fitting_functions['f_duz_dz'] = MultilinearInterpolator(smin, smax, orders)
    # fitting_functions['f_dux_dx'].set_values(np.atleast_2d(np.transpose(dux_dx).flatten()))
    # fitting_functions['f_duy_dx'].set_values(np.atleast_2d(np.transpose(duy_dx).flatten()))
    # fitting_functions['f_duz_dx'].set_values(np.atleast_2d(np.transpose(duz_dx).flatten()))
    # fitting_functions['f_dux_dz'].set_values(np.atleast_2d(np.transpose(dux_dz).flatten()))
    # fitting_functions['f_duy_dz'].set_values(np.atleast_2d(np.transpose(duy_dz).flatten()))
    # fitting_functions['f_duz_dz'].set_values(np.atleast_2d(np.transpose(duz_dz).flatten()))
    # fitting_functions['f_divv_species'] = RegularGridInterpolator((x2, z2), divv_species.T)
    fitting_functions['f_divv_species'] = MultilinearInterpolator(smin, smax, orders)
    fitting_functions['f_divv_species'].set_values(np.atleast_2d(np.transpose(divv_species).flatten()))

    del divv_species
    del dux_dt, duy_dt, duz_dt
    # del dux_dx, duy_dx, duz_dx
    # del dux_dz, duy_dz, duz_dz
    del db_dt

    # get the distribution and save the data
    nbins = 60
    drange = [[1, 1.1], [0, 1]]
    ebins = np.logspace(-4, 2, nbins) / math.sqrt(pmass)

    fdir = '../data/particle_compression/' + run_name + '/'
    mkdir_p(fdir)

    nprocs = pic_info.topology_x * pic_info.topology_y * pic_info.topology_z
    hist_pdivv = np.zeros(nbins - 1) 

    ranks = range(nprocs)
    hists = np.zeros((11, nbins))
    for rank in ranks:
        hists += interpolation_single_rank(run_dir, rank, pmass, species, tindex,
                                           fitting_functions)
    fname = fdir + 'hists_' + species + '.' + str(tindex) + '.all'
    hists.tofile(fname)

    # ncores = multiprocessing.cpu_count()
    # fdata = Parallel(n_jobs=ncores)(delayed(interpolation_single_rank)(rank,
    #     pmass, species, tindex, fitting_functions) for rank in ranks)
    # fdata = np.asarray(fdata)
    # hists = np.sum(fdata, axis=0)
    # fname = fdir + 'hists_' + species + '.' + str(tindex) + '.all'
    # hists.tofile(fname)

    # return fitting_functions


def combine_files(nprocs, run_dir, tindex, data_dir, var_name, species='e'):
    """
    """
    fdir = run_dir + data_dir + '/'
    fname = fdir + var_name + '.' + str(tindex) + '.' + str(0)
    fdata = np.fromfile(fname)
    for rank in range(1, nprocs):
        print("rank %d" % rank)
        fname = fdir + var_name + '.' + str(tindex) + '.' + str(rank)
        fdata += np.fromfile(fname)
    fdir = fdir + 'combined/'
    mkdir_p(fdir)
    fname = fdir + var_name + '_' + species + '.' + str(tindex)
    fdata.tofile(fname)


def plot_hist_para_perp(nprocs, run_dir, tindex):
    """
    """
    nbins = 50
    drange = [[1, 1.1], [0, 1]]
    ebins = np.logspace(-3, 2, nbins)
    fbins = np.linspace(-2, 2, nbins*2)
    # fbins = np.linspace(-6E-2, 6E-2, nbins*2)
    df = fbins[1] - fbins[0]
    fdir = run_dir + 'data_ene/'

    # combine_files(nprocs, run_dir, tindex, 'data_ene', 'hist_para')
    # combine_files(nprocs, run_dir, tindex, 'data_ene', 'hist_perp')
    fname = fdir + 'hist_para.' + str(tindex)
    hist_para = np.fromfile(fname)
    fname = fdir + 'hist_perp.' + str(tindex)
    hist_perp = np.fromfile(fname)

    hist_para = hist_para.reshape((nbins-1, 2*nbins-1))
    hist_perp = hist_perp.reshape((nbins-1, 2*nbins-1))

    nptl_gamma = np.sum(hist_para, axis=1)
    hist_para = div0(hist_para, nptl_gamma[:, None])
    nptl_gamma = np.sum(hist_perp, axis=1)
    hist_perp = div0(hist_perp, nptl_gamma[:, None])
    xmin, xmax = np.min(ebins), np.max(ebins)
    ymin, ymax = np.min(fbins), np.max(fbins)
    # print("Max and min of hist_perp: %d, %d" %
    #         (np.max(hist_perp), np.min(hist_perp)))
    # print("Max and min of hist_para: %d, %d" %
    #         (np.max(hist_para), np.min(hist_para)))
    # xs, ys = 0.15, 0.15
    # w1, h1 = 0.8, 0.8
    # fig = plt.figure(figsize=[7, 5])
    # ax1 = fig.add_axes([xs, ys, w1, h1])
    # ax1.plot(fbins[:-1], hist_para[400, :], linewidth=2)
    # ax1.plot(fbins[:-1], hist_perp[400, :], linewidth=2)
    # ax1.set_xlim([-0.2, 1.2])
    # ax1.set_xlabel('Fraction', fontdict=font, fontsize=20)
    # ax1.set_ylabel('f', fontdict=font, fontsize=20)
    # ax1.tick_params(labelsize=16)
    plt.imshow(hist_perp.T, aspect='auto', cmap=plt.cm.jet,
               origin='lower', extent=[xmin, xmax, ymin, ymax],
               norm=LogNorm(vmin=1E-3, vmax=1),
               interpolation='bicubic')
    # plt.imshow(hist_perp.T, aspect='auto', cmap=plt.cm.jet,
    #         origin='lower', extent=[xmin,xmax,ymin,ymax],
    #         norm=LogNorm(vmin=1, vmax=1E6),
    #         interpolation='bicubic')
    # plt.imshow(hist_para.T, aspect='auto', cmap=plt.cm.Reds,
    #         origin='lower', extent=[xmin,xmax,ymin,ymax],
    #         norm=LogNorm(vmin=1, vmax=1E6),
            # interpolation='bicubic')
    plt.show()



def plot_hist_de_para_perp(nprocs, run_dir, run_name, pic_info, tindex,
                           species='e', if_combine_files=False, if_normalize=False):
    """
    """
    nbins = 60
    drange = [[1, 1.1], [0, 1]]
    ebins = np.logspace(-4, 2, nbins)
    fbins = np.linspace(-2, 2, nbins*2)
    # fbins = np.linspace(-6E-2, 6E-2, nbins*2)
    df = fbins[1] - fbins[0]
    fdir = run_dir + 'data_ene/'

    if if_combine_files:
        combine_files(nprocs, run_dir, tindex, 'data_ene', 'hist_de_para', species)
        combine_files(nprocs, run_dir, tindex, 'data_ene', 'hist_de_perp', species)
        combine_files(nprocs, run_dir, tindex, 'data_ene', 'hist_de_vxb', species)
        combine_files(nprocs, run_dir, tindex, 'data_ene', 'hist_nptl', species)
        combine_files(nprocs, run_dir, tindex, 'data_ene', 'hist_pdivv', species)
        combine_files(nprocs, run_dir, tindex, 'data_ene', 'hist_pdiv_vperp', species)
        combine_files(nprocs, run_dir, tindex, 'data_ene', 'hist_pshear', species)
        combine_files(nprocs, run_dir, tindex, 'data_ene', 'hist_ptensor_dv', species)
        combine_files(nprocs, run_dir, tindex, 'data_ene', 'hist_de_dvdt', species)
    fdir += 'combined/'
    fname_post = '_' + species + '.' + str(tindex)
    fname = fdir + 'hist_de_para' + fname_post
    hist_de_para = np.fromfile(fname)
    fname = fdir + 'hist_de_perp' + fname_post
    hist_de_perp = np.fromfile(fname)
    fname = fdir + 'hist_de_vxb' + fname_post
    hist_de_vxb = np.fromfile(fname)
    fname = fdir + 'hist_nptl' + fname_post
    hist_nptl = np.fromfile(fname)
    fname = fdir + 'hist_pdivv' + fname_post
    # fname = run_dir + '/data_ene/hist_pdivv.' + str(tindex) + '.test'
    hist_pdivv = np.fromfile(fname)
    fname = fdir + 'hist_pdiv_vperp' + fname_post
    hist_pdiv_vperp = np.fromfile(fname)
    fname = fdir + 'hist_pshear' + fname_post
    hist_pshear = np.fromfile(fname)
    fname = fdir + 'hist_ptensor_dv' + fname_post
    hist_ptensor_dv = np.fromfile(fname)
    fname = fdir + 'hist_de_dvdt' + fname_post
    hist_de_dvdt = np.fromfile(fname)

    hist_de_para = np.resize(hist_de_para, (nbins))
    hist_de_perp = np.resize(hist_de_perp, (nbins))
    hist_de_vxb = np.resize(hist_de_vxb, (nbins))
    hist_nptl = np.resize(hist_nptl, (nbins))
    hist_pdivv = np.resize(hist_pdivv, (nbins))
    hist_pdiv_vperp = np.resize(hist_pdiv_vperp, (nbins))
    hist_pshear = np.resize(hist_pshear, (nbins))
    hist_ptensor_dv = np.resize(hist_ptensor_dv, (nbins))
    hist_de_dvdt = np.resize(hist_de_dvdt, (nbins))

    nsum = 1
    hist_de_para = np.sum(hist_de_para.reshape([nbins/nsum, nsum]), axis=1)
    hist_de_perp = np.sum(hist_de_perp.reshape([nbins/nsum, nsum]), axis=1)
    hist_de_vxb = np.sum(hist_de_vxb.reshape([nbins/nsum, nsum]), axis=1)
    hist_nptl = np.sum(hist_nptl.reshape([nbins/nsum, nsum]), axis=1)
    hist_pdivv = np.sum(hist_pdivv.reshape([nbins/nsum, nsum]), axis=1)
    hist_pdiv_vperp = np.sum(hist_pdiv_vperp.reshape([nbins/nsum, nsum]), axis=1)
    hist_pshear = np.sum(hist_pshear.reshape([nbins/nsum, nsum]), axis=1)
    hist_ptensor_dv = np.sum(hist_ptensor_dv.reshape([nbins/nsum, nsum]), axis=1)
    hist_de_dvdt = np.sum(hist_de_dvdt.reshape([nbins/nsum, nsum]), axis=1)
    hist_de_para[-1] = hist_de_para[-2]
    hist_de_perp[-1] = hist_de_perp[-2]
    hist_de_vxb[-1] = hist_de_vxb[-2]
    hist_nptl[-1] = hist_nptl[-2]
    hist_pdivv[-1] = hist_pdivv[-2]
    hist_pdiv_vperp[-1] = hist_pdiv_vperp[-2]
    hist_pshear[-1] = hist_pshear[-2]
    hist_ptensor_dv[-1] = hist_ptensor_dv[-2]
    hist_de_dvdt[-1] = hist_de_dvdt[-2]
    emin_log = math.log10(np.min(ebins))
    emax_log = math.log10(np.max(ebins))
    ebins = np.logspace(emin_log, emax_log, nbins/nsum)

    hist_de_tot = hist_de_para + hist_de_perp

    print("Parallel and perpendicular heating: %f %f" %
          (np.sum(hist_de_para), np.sum(hist_de_perp)))
    print("Heating due to ideal electric field: %f" % (np.sum(hist_de_vxb)))
    print("Number of particles: %d" % (np.sum(hist_nptl)))
    print("pdivv, pdiv_vperp, pshear, ptensor_dv, inertial term: %f %f %f %f %f" %
          (np.sum(hist_pdivv), np.sum(hist_pdiv_vperp),
           np.sum(hist_pshear), np.sum(hist_ptensor_dv),
           np.sum(hist_de_dvdt)))

    if if_normalize:
        hist_de_para = div0(hist_de_para, hist_nptl+0.0)
        hist_de_perp = div0(hist_de_perp, hist_nptl+0.0)
        hist_de_vxb = div0(hist_de_vxb, hist_nptl+0.0)
        hist_pdivv = div0(hist_pdivv, hist_nptl+0.0)
        hist_pdiv_vperp = div0(hist_pdiv_vperp, hist_nptl+0.0)
        hist_pshear = div0(hist_pshear, hist_nptl+0.0)
        hist_ptensor_dv = div0(hist_ptensor_dv, hist_nptl+0.0)
        hist_de_dvdt = div0(hist_de_dvdt, hist_nptl+0.0)

    de_para_fraction = div0(hist_de_para, hist_de_tot)
    de_perp_fraction = div0(hist_de_perp, hist_de_tot)

    xs, ys = 0.15, 0.15
    w1, h1 = 0.8, 0.8
    fig = plt.figure(figsize=[7, 5])
    ax1 = fig.add_axes([xs, ys, w1, h1])
    ax1.semilogx(ebins, hist_de_para, linewidth=2, label=r'$\parallel$')
    ax1.semilogx(ebins, hist_de_perp, linewidth=2, label=r'$\perp$')
    # ax1.semilogx(ebins, hist_de_vxb, linewidth=2,
    #              linestyle='--', label=r'$-\boldsymbol{u}\times\boldsymbol{B}$')
    # ax1.semilogx(ebins, hist_de_para + hist_de_perp, linewidth=2,
    #              color='k', label='Total')
    ax1.semilogx(ebins, hist_pdivv, linewidth=2,
                 label=r'$-p\nabla\cdot\boldsymbol{u}$')
    # ax1.semilogx(ebins, hist_pdiv_vperp, linewidth=2,
    #              label=r'$-p\nabla\cdot\boldsymbol{u}_\perp$')
    ax1.semilogx(ebins, hist_pshear, linewidth=2,
                 label=r'$-(p_\parallel-p_\perp)b_ib_j\sigma_{ij}$')
    ax1.semilogx(ebins, hist_ptensor_dv, linewidth=2,
                 label=r'$-\mathcal{P}:\nabla\boldsymbol{u}$')
    # data_sum = hist_ptensor_dv + hist_de_dvdt
    data_sum = hist_pdiv_vperp + hist_pshear + hist_de_dvdt
    ax1.semilogx(ebins, data_sum, linewidth=2, label='Sum')
    ax1.semilogx(ebins, hist_de_dvdt, linewidth=2,
                 label='Inertial term')
    # ax1.loglog(ebins, hist_nptl, linewidth=2, color='k', label='Total')
    if species == 'e':
        ax1.set_xlim([1E-3, 20])
    else:
        ax1.set_xlim([1E-4, 2])
    # ax1.set_ylim([-1E-7, 1E-7])
    ax1.set_xlabel(r'$\gamma-1$', fontdict=font, fontsize=20)
    ax1.set_ylabel(r'$f$', fontdict=font, fontsize=20)
    ax1.tick_params(labelsize=16)
    leg = ax1.legend(loc=2, prop={'size': 20}, ncol=1,
                     shadow=False, fancybox=False, frameon=False)
    fdir = '../img/de_para_perp/' + run_name + '/'
    mkdir_p(fdir)
    fname = fdir + 'de_para_perp_' + species + '_' + str(tindex) + '.eps'
    fig.savefig(fname)
    # plt.close()


def get_fields_tindex(tindex, pic_info):
    """Get fields time indices before and after current time step

    Args:
        tindex: current time index
        pic_info: pic simulation information
    """
    # finterval = pic_info.fields_interval
    # ntf = pic_info.ntf
    # if tindex > finterval:
    #     tindex_pre = tindex - finterval
    # else:
    #     tindex_pre = tindex
    # if tindex < ntf * finterval:
    #     tindex_post = tindex + finterval
    # else:
    #     tindex_post = tindex
    if tindex > 0:
        tindex_pre = tindex - 1
    else:
        tindex_pre = 0
    tindex_post = tindex + 1
    return (tindex_pre, tindex_post)


def plot_compression_heating(run_name, tindex, species):
    """
    """
    nbins = 60
    drange = [[1, 1.1], [0, 1]]
    ebins = np.logspace(-4, 2, nbins + 1)
    fbins = np.linspace(-2, 2, nbins*2)
    # fbins = np.linspace(-6E-2, 6E-2, nbins*2)
    df = fbins[1] - fbins[0]
    fdir = '../data/particle_compression/' + run_name + '/'
    fname = fdir + 'hists_' + species + '.' + str(tindex) + '.all'
    fdata = np.fromfile(fname)
    sz, = fdata.shape
    nvar = sz / nbins
    fdata = fdata.reshape((nvar, nbins))
    hist_de_para = fdata[0, :]
    hist_de_perp = fdata[1, :]
    hist_pdivv = fdata[2, :]
    hist_pdiv_vperp = fdata[3, :]
    hist_pshear = fdata[4, :]
    hist_ptensor_dv = fdata[5, :]
    hist_de_dudt = fdata[6, :]
    hist_de_cons_mu = fdata[7, :]
    hist_div_ptensor_vperp = fdata[8, :]
    hist_div_pperp_vperp = fdata[9, :]
    hist_nptl = fdata[-1, :]
    if species == 'e':
        charge = r'$-e$'
    else:
        charge = r'$e$'
    colors = colors_Set1_9[:5] + colors_Set1_9[6:] # remove yellow color
    # colors = palettable.cartocolors.qualitative.Vivid_7.mpl_colors

    print("-----------------------------------------------------------------------")
    print("%10s " * 10 % ("de_para", "de_perp", "pdivv", "pdiv_vperp", "pshear",
                          "ptensor", "de_dudt", "de_cons_mu", "flux_total",
                          "flux_perp"))
    print("-----------------------------------------------------------------------")
    print("%10.4f " * 10 % (np.sum(hist_de_para), np.sum(hist_de_perp),
                            np.sum(hist_pdivv), np.sum(hist_pdiv_vperp),
                            np.sum(hist_pshear), np.sum(hist_ptensor_dv),
                            np.sum(hist_de_dudt), np.sum(hist_de_cons_mu),
                            np.sum(hist_div_ptensor_vperp),
                            np.sum(hist_div_pperp_vperp)))

    xs, ys = 0.15, 0.15
    w1, h1 = 0.8, 0.8
    fig = plt.figure(figsize=[7, 5])
    ax1 = fig.add_axes([xs, ys, w1, h1])
    ax1.set_prop_cycle('color', colors)
    label1 = charge + r'$\boldsymbol{v}_\parallel\cdot\boldsymbol{E}_\parallel$'
    label2 = charge + r'$\boldsymbol{v}_\perp\cdot\boldsymbol{E}_\perp$'
    label3 = r'$-p\nabla\cdot\boldsymbol{v}_E$'
    label4 = r'$-(p_\parallel-p_\perp)b_ib_j\sigma_{ij}$'
    label5 = r'$-\mathbf{P}:\nabla\boldsymbol{v}_E$'
    if species == 'e':
        label6 = r'$m_e(d\boldsymbol{u}_e/dt)\cdot\boldsymbol{v}_E$'
    else:
        label6 = r'$m_i(d\boldsymbol{u}_i/dt)\cdot\boldsymbol{v}_E$'
    label7 = label5 + r'$+$' + label6
    data_sum = hist_ptensor_dv + hist_de_dudt
    # data_sum = hist_pdiv_vperp + hist_pshear + hist_de_dudt
    ax1.semilogx(ebins[:-1], data_sum, color='k', linewidth=2, label=label7)
    ax1.semilogx(ebins[:-1], hist_de_perp, color=colors[0], linewidth=2, label=label2)
    ax1.semilogx(ebins[:-1], hist_pdiv_vperp, color=colors[1], linewidth=2, label=label3)
    ax1.semilogx(ebins[:-1], hist_pshear, color=colors[2], linewidth=2, label=label4)
    ax1.semilogx(ebins[:-1], hist_de_dudt, color=colors[0], linewidth=2,
                 linestyle='--', label=label6)
    ax1.semilogx(ebins[:-1], hist_ptensor_dv, color=colors[1], linewidth=2,
                 linestyle='--', label=label5)
    ax1.semilogx(ebins[:-1], hist_de_para, color=colors[2], linewidth=2,
                 linestyle='--', label=label1)
    ax1.semilogx(ebins[:-1], hist_de_perp - data_sum, linewidth=2,
                 color='k', linestyle='--', label='Difference')
    # ax1.semilogx(ebins[:-1], hist_div_ptensor_vperp, linewidth=2,
    #              label='Flux term')
    # ax1.semilogx(ebins[:-1], hist_div_pperp_vperp, linewidth=2,
    #              label='Flux term perp')
    # ax1.semilogx(ebins[:-1], hist_de_cons_mu, linewidth=2,
    #              label=r'$\mu$ conservation')
    # data_sum = hist_ptensor_dv + hist_de_dudt + hist_div_ptensor_vperp
    # ax1.loglog(ebins, hist_nptl, linewidth=2, color='k', label='Total')
    if species == 'e':
        ax1.set_xlim([1E-3, 20])
    else:
        ax1.set_xlim([1E-4, 2])
    # ax1.set_ylim([-1E-7, 1E-7])
    ax1.set_xlabel(r'$\gamma-1$', fontdict=font, fontsize=20)
    ax1.set_ylabel(r'$f$', fontdict=font, fontsize=20)
    ax1.tick_params(labelsize=16)
    leg = ax1.legend(loc=2, prop={'size': 16}, ncol=1,
                     shadow=False, fancybox=False, frameon=False)
    # for color, text in zip(colors, leg.get_texts()):
    #     text.set_color(color)
    # ax1.text(0.02, 0.90, label7, color=colors[6], fontsize=16,
    #          horizontalalignment='left', verticalalignment='bottom',
    #          transform = ax1.transAxes)
    # ax1.text(0.02, 0.80, label1, color=colors[0], fontsize=16,
    #          horizontalalignment='left', verticalalignment='bottom',
    #          transform = ax1.transAxes)
    # ax1.text(0.02, 0.70, label2, color=colors[1], fontsize=16,
    #          horizontalalignment='left', verticalalignment='bottom',
    #          transform = ax1.transAxes)
    # ax1.text(0.02, 0.60, label3, color=colors[2], fontsize=16,
    #          horizontalalignment='left', verticalalignment='bottom',
    #          transform = ax1.transAxes)
    # ax1.text(0.02, 0.50, label4, color=colors[3], fontsize=16,
    #          horizontalalignment='left', verticalalignment='bottom',
    #          transform = ax1.transAxes)
    # ax1.text(0.02, 0.40, label5, color=colors[4], fontsize=16,
    #          horizontalalignment='left', verticalalignment='bottom',
    #          transform = ax1.transAxes)
    # ax1.text(0.02, 0.30, label6, color=colors[5], fontsize=16,
    #          horizontalalignment='left', verticalalignment='bottom',
    #          transform = ax1.transAxes)
    fdir = '../img/de_para_perp/' + run_name + '/'
    mkdir_p(fdir)
    fname = fdir + 'de_para_perp_' + species + '_' + str(tindex) + '.eps'
    fig.savefig(fname)
    # plt.close()


def save_econv_data(fdata, fdir, species, tindex):
    """
    """
    fdata = np.asarray(fdata)
    hists = np.sum(fdata, axis=0)
    fname = fdir + 'hists_' + species + '.' + str(tindex) + '.all'
    hists.tofile(fname)


def combine_files_single_core(nprocs, run_dir, run_name, species):
    """
    """
    combine_files(nprocs, run_dir, tindex, 'data_ene', 'hist_de_para', species)
    combine_files(nprocs, run_dir, tindex, 'data_ene', 'hist_de_perp', species)
    combine_files(nprocs, run_dir, tindex, 'data_ene', 'hist_de_vxb', species)
    combine_files(nprocs, run_dir, tindex, 'data_ene', 'hist_nptl', species)
    combine_files(nprocs, run_dir, tindex, 'data_ene', 'hist_pdivv', species)
    combine_files(nprocs, run_dir, tindex, 'data_ene', 'hist_pdiv_vperp', species)
    combine_files(nprocs, run_dir, tindex, 'data_ene', 'hist_pshear', species)
    combine_files(nprocs, run_dir, tindex, 'data_ene', 'hist_ptensor_dv', species)
    combine_files(nprocs, run_dir, tindex, 'data_ene', 'hist_de_dvdt', species)


def get_cmd_args():
    """Get command line arguments
    """
    # default_run_name = 'mime25_beta002_guide00'
    # default_run_dir = \
    #         '/net/scratch3/xiaocanli/reconnection/mime25-sigma1-beta002-guide00-200-100/'
    # default_run_name = 'dump_test'
    # default_run_dir = '/net/scratch3/xiaocanli/reconnection/dump_test/'
    default_run_name = 'mime25_beta008_guide00_frequent_dump'
    default_run_dir = '/net/scratch3/xiaocanli/reconnection/frequent_dump/' + \
            'mime25_beta008_guide00_frequent_dump/'
    parser = argparse.ArgumentParser(description='Compression analysis based on particles')
    parser.add_argument('--combine_files', action="store_true", default=False,
                        help='whether to combine files')
    parser.add_argument('--normalize', action="store_true", default=False,
                        help='whether to normalized the distributions')
    parser.add_argument('--species', action="store", default='e',
                        help='particle species')
    parser.add_argument('--run_dir', action="store", default=default_run_dir,
                        help='run directory')
    parser.add_argument('--run_name', action="store", default=default_run_name,
                        help='run name')
    parser.add_argument('--ct_particle', action="store", default='4', type=int,
                        help='time frame for particle dump')
    parser.add_argument('--mpi_rank', action="store", default='50', type=int,
                        help='MPI rank')
    parser.add_argument('--single_core', action="store_true", default=False,
                        help='only analyze particles in one core at a time')
    parser.add_argument('--only_plotting', action="store_true", default=False,
                        help='whether only plotting data without calculation')
    parser.add_argument('--verbose', action="store_true", default=False,
                        help='whether to show diagnostic information')
    parser.add_argument('--multi_frames', action="store_true", default=False,
                        help='whether analyzing multiple frames')
    return parser.parse_args()


def main():
    """
    """
    args = get_cmd_args()
    run_name = args.run_name
    run_dir = args.run_dir
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pint = pic_info.particle_interval
    ntp = pic_info.ntp
    ct_particle = args.ct_particle
    tindex = pint * ct_particle
    tindex_pre, tindex_post = get_fields_tindex(tindex, pic_info)
    rank = args.mpi_rank
    exb_drift = True
    species = args.species
    if_combine_files = args.combine_files
    if_normalize = args.normalize
    nprocs = pic_info.topology_x * pic_info.topology_y * pic_info.topology_z
    ranks = range(nprocs)
    verbose = args.verbose
    single_core = args.single_core
    multi_frames = args.multi_frames
    ncores = multiprocessing.cpu_count()
    if species == 'e':
        charge = -1.0
        pmass = 1.0
    else:
        charge = 1.0
        pmass = pic_info.mime
    cmd = 'rm data_ene/hist_*'
    fdir = run_dir + 'data_ene/'
    mkdir_p(fdir)
    nbins = 60
    cts = range(1, ntp)
    def processInput(job_id):
        print job_id
        rank = job_id
        interp_particle_compression(pic_info, run_dir, tindex, tindex_pre,
                                    tindex_post, rank, species, exb_drift,
                                    verbose)
    def processFrames(job_id):
        print job_id
        ct = job_id
        print("Time frame: %d of %d" % (ct, ntp))
        tindex = pint * ct
        tindex_pre, tindex_post = get_fields_tindex(tindex, pic_info)
        interp_particle_compression_single(pic_info, run_dir, run_name, tindex, tindex_pre,
                                           tindex_post, species)

    if multi_frames:
        for ct in range(1, ntp):
            print("Time frame: %d of %d" % (ct, ntp))
            tindex = pint * ct
            tindex_pre, tindex_post = get_fields_tindex(tindex, pic_info)
            if single_core:
                if not args.only_plotting:
                    Parallel(n_jobs=ncores)(delayed(processInput)(rank) for rank in ranks)
                    combine_files_single_core(nprocs, run_dir, run_name, species)
                    p1 = subprocess.Popen([cmd], cwd=run_dir, stdout=open('outfile.out', 'w'),
                                          stderr=subprocess.STDOUT, shell=True)
                    p1.wait()

                plot_hist_de_para_perp(nprocs, run_dir, run_name, pic_info,
                                       tindex, species, if_combine_files,
                                       if_normalize)
            else:
                if not args.only_plotting:
                    interp_particle_compression_single(pic_info, run_dir, run_name, tindex,
                                                       tindex_pre, tindex_post, species)
                    # fdata = Parallel(n_jobs=ncores, max_nbytes=1e6)\
                    #         (delayed(interpolation_single_rank)
                    #          (run_dir, rank, pmass, species, tindex, fitting_functions)
                    #          for rank in ranks)
                    # save_econv_data(fdata, fdir, species, tindex)
                    # del fitting_functions
                plot_compression_heating(run_name, tindex, species)
                plt.close()
            gc.collect()
    else:
        if single_core:
            # if not args.only_plotting:
            #     Parallel(n_jobs=ncores)(delayed(processInput)(rank) for rank in ranks)
            # plot_hist_de_para_perp(nprocs, run_dir, run_name, pic_info, tindex,
            #                        species, if_combine_files, if_normalize)
            interp_particle_compression(pic_info, run_dir, tindex, tindex_pre,
                                        tindex_post, rank, species, exb_drift,
                                        verbose)
        else:
            if not args.only_plotting:
                interp_particle_compression_single(pic_info, run_dir, run_name, tindex,
                                                   tindex_pre, tindex_post, species)
                # hists = np.zeros((11, nbins))
                # for rank in ranks:
                #     print(rank)
                #     hists += interpolation_single_rank(run_dir, rank, pmass, species, tindex,
                #                                        fitting_functions)
                # fname = fdir + 'hists_' + species + '.' + str(tindex) + '.all'
                # hists.tofile(fname)

                # ncores = ntp - 1
                # Parallel(n_jobs=ncores)(delayed(processFrames)(ct) for ct in cts)

                # ranks = range(36)
                # fdata = Parallel(n_jobs=ncores)(delayed(interpolation_single_rank)(run_dir, rank,
                #     pmass, species, tindex, fitting_functions) for rank in ranks)
                # save_econv_data(fdata, fdir, species, tindex)
                # del fitting_functions
            plot_compression_heating(run_name, tindex, species)
            plt.show()


if __name__ == "__main__":
    main()
