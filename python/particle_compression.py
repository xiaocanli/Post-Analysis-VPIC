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
from scipy.interpolate import RectBivariateSpline

import pic_information
from energy_conversion import read_data_from_json
from particle_distribution import read_particle_data
from shell_functions import mkdir_p

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

font = {
    'family': 'serif',
    #'color'  : 'darkred',
    'color': 'black',
    'weight': 'normal',
    'size': 24,
}

# style.use(['dark_background'])
# style.use('fivethirtyeight')
style.use('ggplot')

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
    vx, vy, vz are actually current densities
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
    nrho = np.abs(fdata[3, :, 1, :])
    return (vx, vy, vz, nrho)


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
        ni  = np.abs(hhydro[3, :, 1, :])
        uix = hhydro[4, :, 1, :]
        uiy = hhydro[5, :, 1, :]
        uiz = hhydro[6, :, 1, :]
        ini = div0(1.0, ni)
        vix *= ini
        viy *= ini
        viz *= ini
        del hhydro, ini

        inrho = div0(1.0 , ne + ni*mime)
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


def interp_particle_compression(pic_info, run_dir, tindex, rank, species='e',
                                exb_drift=False):
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

    # file names
    hydro_dir = run_dir + 'hydro/T.' + str(tindex) + '/'
    fields_dir = run_dir + 'fields/T.' + str(tindex) + '/'
    particle_dir = run_dir + 'particle/T.' + str(tindex) + '/'
    ehydro_name = hydro_dir + 'ehydro.' + str(tindex)
    Hhydro_name = hydro_dir + 'Hhydro.' + str(tindex)
    eparticle_name = particle_dir + 'eparticle.' + str(tindex)
    hparticle_name = particle_dir + 'hparticle.' + str(tindex)
    field_name = fields_dir + 'fields.' + str(tindex)

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
    texy -= vex*uey
    texz -= vex*uez
    teyz -= vey*uez
    teyx = texy - uex*vey
    tezx = texz - uex*vez
    tezy = teyz - uey*vez
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
    ni  = hhydro[3, :, 1, :]
    uix = hhydro[4, :, 1, :]
    uiy = hhydro[5, :, 1, :]
    uiz = hhydro[6, :, 1, :]
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
    tixy -= vix*uiy
    tixz -= vix*uiz
    tiyz -= viy*uiz
    tiyx = tixy - uix*viy
    tizx = tixz - uix*viz
    tizy = tiyz - uiy*viz
    pi_para = (tixx*bx**2 + tiyy*by**2 + tizz*bz**2 +
               (tixy + tiyx)*bx*by + (tixz + tizx)*bx*bz + (tiyz + tizy)*by*bz)
    pi_para *= ib2
    pi_perp = 0.5 * (tixx + tiyy + tizz - pi_para)
    pi = (pi_para + 2.0 * pi_perp) / 3.0
    del hhydro, ini, tixx, tiyy, tizz
    del tixy, tixz, tiyz, tiyx, tizx, tizy

    if species == 'e':
        fname = eparticle_name + '.' + str(rank)
        pmass = 1.0
        charge = -1.0
        pres = pe
        ppara = pe_para
        pperp = pe_perp
    else:
        fname = hparticle_name + '.' + str(rank)
        pmass = pic_info.mime
        charge = 1.0
        pres = pi
        ppara = pi_para
        pperp = pi_perp

    del pe, pi, pe_para, pe_perp, pi_para, pi_perp
    print('Total pressure from fluid: %f' % (np.sum(pres[1:-1,1:-1])*v0.dx*v0.dz))
    print('Total anisotropy from fluid: %f' % 
            (np.sum(ppara[1:-1,1:-1] - pperp[1:-1,1:-1])*v0.dx*v0.dz))

    # read particle data
    (v0, pheader, ptl) = read_particle_data(fname)
    dxp = ptl['dxyz'][:, 0]
    dzp = ptl['dxyz'][:, 2]
    icell = ptl['icell']
    uxp = ptl['u'][:, 0]
    uyp = ptl['u'][:, 1]
    uzp = ptl['u'][:, 2]
    q =  ptl['q']
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

    x = np.linspace(v0.x0, v0.x0 + v0.nx * v0.dx, nx - 1)
    z = np.linspace(v0.z0, v0.z0 + v0.nz * v0.dz, nz - 1)

    # interpolate electric and magnetic fields
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
    del f_ex, f_ey, f_ez, f_bx, f_by, f_bz

    # interpolate compressional terms
    inrho = div0(1.0 , ne + ni*mime)
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
    bbsigma_perp = np.zeros(vx.shape)
    divv[1:, 1:] = np.gradient(vx[1:, 1:], dx, axis=1) + \
                   np.gradient(vz[1:, 1:], dz, axis=0)
    div_vperp[1:, 1:] = np.gradient(vx_perp[1:, 1:], dx, axis=1) + \
                        np.gradient(vz_perp[1:, 1:], dz, axis=0)
    bbsigma_perp[1:, 1:] = (np.gradient(vx_perp[1:, 1:], dx, axis=1) - \
                           (1./3.) * div_vperp[1:, 1:]) * bx[1:, 1:]**2 + \
                           (np.gradient(vz_perp[1:, 1:], dz, axis=0) - \
                           (1./3.) * div_vperp[1:, 1:]) * bz[1:, 1:]**2 + \
                           0.5 * (np.gradient(vy_perp[1:, 1:], dx, axis=1)) * \
                                 bx[1:, 1:] * by[1:, 1:] + \
                           0.5 * (np.gradient(vx_perp[1:, 1:], dz, axis=0) + 
                                  np.gradient(vz_perp[1:, 1:], dx, axis=1)) * \
                                 bx[1:, 1:] * bz[1:, 1:] + \
                           0.5 * (np.gradient(vy_perp[1:, 1:], dz, axis=0)) * \
                                 by[1:, 1:] * bz[1:, 1:]
    bbsigma_perp *= ib2
                           
    f_divv = RectBivariateSpline(x, z, divv[1:,1:].T)
    f_div_vperp = RectBivariateSpline(x, z, div_vperp[1:,1:].T)
    f_bbsigma_perp = RectBivariateSpline(x, z, bbsigma_perp[1:,1:].T)

    divv_ptl = f_divv(x_ptl, z_ptl, grid=False)
    div_vperp_ptl = f_div_vperp(x_ptl, z_ptl, grid=False)
    bbsigma_perp_ptl = f_bbsigma_perp(x_ptl, z_ptl, grid=False)
    ds = v0.dx * v0.dz
    print("pdivv and pdiv_vperp and pshear from fluid: %f %f %f" %
            (np.sum(-pres[1:, 1:]*divv[1:, 1:])*ds,
             np.sum(-pres[1:, 1:]*div_vperp[1:, 1:])*ds,
             np.sum(-(ppara[1:, 1:] - pperp[1:, 1:])*bbsigma_perp[1:, 1:])*ds))
    del f_divv, f_div_vperp, f_bbsigma_perp
    del inrho, pres, divv, div_vperp, vxb, ib2, vx_perp, vy_perp, vz_perp
    del ppara, pperp, bbsigma_perp

    # interpolate motional electric field
    einx = vz*by - vy*bz
    einy = vx*bz - vz*bx
    einz = vy*bx - vx*by
    f_einx = RectBivariateSpline(x, z, einx[1:,1:].T)
    f_einy = RectBivariateSpline(x, z, einy[1:,1:].T)
    f_einz = RectBivariateSpline(x, z, einz[1:,1:].T)
    einx_ptl = f_einx(x_ptl, z_ptl, grid=False)
    einy_ptl = f_einy(x_ptl, z_ptl, grid=False)
    einz_ptl = f_einz(x_ptl, z_ptl, grid=False)
    del f_einx, f_einy, f_einz
    del ex, ey, ez, bx, by, bz
    del vx, vy, vz, einx, einy, einz

    # interpolate fluid velocities
    if species == 'e':
        f_vx = RectBivariateSpline(x, z, vex[1:,1:].T)
        f_vy = RectBivariateSpline(x, z, vey[1:,1:].T)
        f_vz = RectBivariateSpline(x, z, vez[1:,1:].T)
        f_ux = RectBivariateSpline(x, z, uex[1:,1:].T/ne[1:,1:].T) # including particle mass
        f_uy = RectBivariateSpline(x, z, uey[1:,1:].T/ne[1:,1:].T)
        f_uz = RectBivariateSpline(x, z, uez[1:,1:].T/ne[1:,1:].T)
        f_ux_vx = RectBivariateSpline(x, z, uex[1:,1:].T*vex[1:,1:].T/ne[1:,1:].T)
        f_uy_vy = RectBivariateSpline(x, z, uey[1:,1:].T*vey[1:,1:].T/ne[1:,1:].T)
        f_uz_vz = RectBivariateSpline(x, z, uez[1:,1:].T*vez[1:,1:].T/ne[1:,1:].T)
        f_ux_vy = RectBivariateSpline(x, z, uex[1:,1:].T*vey[1:,1:].T/ne[1:,1:].T)
        f_ux_vz = RectBivariateSpline(x, z, uex[1:,1:].T*vez[1:,1:].T/ne[1:,1:].T)
        f_uy_vz = RectBivariateSpline(x, z, uey[1:,1:].T*vez[1:,1:].T/ne[1:,1:].T)
        f_vx_uy = RectBivariateSpline(x, z, vex[1:,1:].T*uey[1:,1:].T/ne[1:,1:].T)
        f_vx_uz = RectBivariateSpline(x, z, vex[1:,1:].T*uez[1:,1:].T/ne[1:,1:].T)
        f_vy_uz = RectBivariateSpline(x, z, vey[1:,1:].T*uez[1:,1:].T/ne[1:,1:].T)
    else:
        f_vx = RectBivariateSpline(x, z, vix[1:,1:].T)
        f_vy = RectBivariateSpline(x, z, viy[1:,1:].T)
        f_vz = RectBivariateSpline(x, z, viz[1:,1:].T)
        f_ux = RectBivariateSpline(x, z, uix[1:,1:].T/ni[1:,1:].T)
        f_uy = RectBivariateSpline(x, z, uiy[1:,1:].T/ni[1:,1:].T)
        f_uz = RectBivariateSpline(x, z, uiz[1:,1:].T/ni[1:,1:].T)
        f_ux_vx = RectBivariateSpline(x, z, uix[1:,1:].T*vix[1:,1:].T/ni[1:,1:].T)
        f_uy_vy = RectBivariateSpline(x, z, uiy[1:,1:].T*viy[1:,1:].T/ni[1:,1:].T)
        f_uz_vz = RectBivariateSpline(x, z, uiz[1:,1:].T*viz[1:,1:].T/ni[1:,1:].T)
        f_ux_vy = RectBivariateSpline(x, z, uix[1:,1:].T*viy[1:,1:].T/ni[1:,1:].T)
        f_ux_vz = RectBivariateSpline(x, z, uix[1:,1:].T*viz[1:,1:].T/ni[1:,1:].T)
        f_uy_vz = RectBivariateSpline(x, z, uiy[1:,1:].T*viz[1:,1:].T/ni[1:,1:].T)
        f_vx_uy = RectBivariateSpline(x, z, vix[1:,1:].T*uiy[1:,1:].T/ni[1:,1:].T)
        f_vx_uz = RectBivariateSpline(x, z, vix[1:,1:].T*uiz[1:,1:].T/ni[1:,1:].T)
        f_vy_uz = RectBivariateSpline(x, z, viy[1:,1:].T*uiz[1:,1:].T/ni[1:,1:].T)

    ux_vx_ptl = f_ux_vx(x_ptl, z_ptl, grid=False)
    uy_vy_ptl = f_uy_vy(x_ptl, z_ptl, grid=False)
    uz_vz_ptl = f_uz_vz(x_ptl, z_ptl, grid=False)
    ux_vy_ptl = f_ux_vy(x_ptl, z_ptl, grid=False)
    ux_vz_ptl = f_ux_vz(x_ptl, z_ptl, grid=False)
    uy_vz_ptl = f_uy_vz(x_ptl, z_ptl, grid=False)
    vx_uy_ptl = f_vx_uy(x_ptl, z_ptl, grid=False)
    vx_uz_ptl = f_vx_uz(x_ptl, z_ptl, grid=False)
    vy_uz_ptl = f_vy_uz(x_ptl, z_ptl, grid=False)
    vx_ptl = f_vx(x_ptl, z_ptl, grid=False)
    vy_ptl = f_vy(x_ptl, z_ptl, grid=False)
    vz_ptl = f_vz(x_ptl, z_ptl, grid=False)
    ux_ptl = f_ux(x_ptl, z_ptl, grid=False)
    uy_ptl = f_uy(x_ptl, z_ptl, grid=False)
    uz_ptl = f_uz(x_ptl, z_ptl, grid=False)
    del f_ux_vx, f_uy_vy, f_uz_vz, f_vx, f_vy, f_vz, f_ux, f_uy, f_uz
    del f_ux_vy, f_ux_vz, f_uy_vz, f_vx_uy, f_vx_uz, f_vy_uz
    del vex, vey, vez, uex, uey, uez, ne
    del vix, viy, viz, uix, uiy, uiz, ni
    del x_ptl, z_ptl, x, z

    # compressional and shear heating
    weight = abs(q[0])
    pscalar = (uxp*vxp + uyp*vyp + uzp*vzp) * pmass
    pscalar += ux_vx_ptl + uy_vy_ptl + uz_vz_ptl
    pscalar -= (uxp*vx_ptl + uyp*vy_ptl + uzp*vz_ptl) * pmass
    pscalar -= vxp*ux_ptl + vyp*uy_ptl + vzp*uz_ptl # ux_ptl includes pmass
    pscalar *= weight / 3.0
    pdivv = -pscalar * divv_ptl
    pdiv_vperp = -pscalar * div_vperp_ptl

    bx2 = bx_ptl**2
    by2 = by_ptl**2
    bz2 = bz_ptl**2
    bxy = bx_ptl * by_ptl
    bxz = bx_ptl * bz_ptl
    byz = by_ptl * bz_ptl
    ib2_ptl = 1.0 / (bx2 + by2 + bz2)
    ppara_ptl = (uxp*vxp*bx2 + uyp*vyp*by2 + uzp*vzp*bz2) * pmass
    ppara_ptl += ux_vx_ptl*bx2 + uy_vy_ptl*by2 + uz_vz_ptl*bz2
    ppara_ptl -= (uxp*vx_ptl*bx2 + uyp*vy_ptl*by2 + uzp*vz_ptl*bz2) * pmass
    ppara_ptl -= vxp*ux_ptl*bx2 + vyp*uy_ptl*by2 + vzp*uz_ptl*bz2

    ppara_ptl += (uxp*vyp*bxy + uxp*vzp*bxz + uyp*vzp*byz) * pmass
    ppara_ptl += ux_vy_ptl*bxy + ux_vz_ptl*bxz + uy_vz_ptl*byz
    ppara_ptl -= (uxp*vy_ptl*bxy + uxp*vz_ptl*bxz + uyp*vz_ptl*byz) * pmass
    ppara_ptl -= vyp*ux_ptl*bxy + vzp*ux_ptl*bxz + vzp*uy_ptl*byz

    ppara_ptl += (vxp*uyp*bxy + vxp*uzp*bxz + vyp*uzp*byz) * pmass
    ppara_ptl += vx_uy_ptl*bxy + vx_uz_ptl*bxz + vy_uz_ptl*byz
    ppara_ptl -= (uyp*vx_ptl*bxy + uzp*vx_ptl*bxz + uzp*vy_ptl*byz) * pmass
    ppara_ptl -= vxp*uy_ptl*bxy + vxp*uz_ptl*bxz + vyp*uz_ptl*byz

    ppara_ptl *= ib2_ptl * weight
    pperp_ptl = 0.5 * (pscalar * 3 - ppara_ptl)
    pshear = (pperp_ptl - ppara_ptl) * bbsigma_perp_ptl

    print('Total pressure from particles: %f' % (np.sum(pscalar)))
    print('Total anisotropy from particles: %f' %
            (np.sum(ppara_ptl - pperp_ptl)))
    print("pdivv and pdiv_vperp and pshear from particles: %f %f %f" %
            (np.sum(pdivv), np.sum(pdiv_vperp), np.sum(pshear)))
    del bx2, by2, bz2, bxy, bxz, byz
    del bbsigma_perp_ptl, ppara_ptl, pperp_ptl
    del pscalar, vx_ptl, vy_ptl, vz_ptl, ux_ptl, uy_ptl, uz_ptl
    del ux_vx_ptl, uy_vy_ptl, uz_vz_ptl, divv_ptl, div_vperp_ptl
    del ux_vy_ptl, ux_vz_ptl, uy_vz_ptl, vx_uy_ptl, vx_uz_ptl, vy_uz_ptl
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

    del ex_ptl, ey_ptl, ez_ptl, bx_ptl, by_ptl, bz_ptl, ib2_ptl, exb_ptl
    del ex_para_ptl, ey_para_ptl, ez_para_ptl
    del ex_perp_ptl, ey_perp_ptl, ez_perp_ptl
    
    # heating due -vxb electric field
    de_vxb = charge * (vxp*einx_ptl + vyp*einy_ptl + vzp*einz_ptl) * weight
    del einx_ptl, einy_ptl, einz_ptl
    del vxp, vyp, vzp

    print("Parallel and perpendicular heating: %d, %f, %f" % 
            (rank, np.sum(de_para), np.sum(de_perp)))
    print("Heating due to ideal electric field: %d, %f" % 
            (rank, np.sum(de_vxb)))
    print("Ratio of parallel heating: %d, %f" % 
            (rank, np.sum(de_para)/np.sum(de_tot)))
    print("Maximum and minimum energy gain: %12.5e, %12.5e, %12.5e, %12.5e" %
            (np.max(de_para), np.min(de_para), np.max(de_perp), np.min(de_perp)))

    # get the distribution and save the data
    nbins = 50
    drange = [[1, 1.1], [0, 1]]
    ebins = np.logspace(-3, 2, nbins) / math.sqrt(pmass)

    fdir = run_dir + 'data_ene/'
    mkdir_p(fdir)
    print("Maximum and minimum gamma: %12.5e, %12.5e" %
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
    fname = fdir + 'hist_pdivv.' + str(tindex) + '.' + str(rank)
    hist_pdivv.tofile(fname)
    fname = fdir + 'hist_pdiv_vperp.' + str(tindex) + '.' + str(rank)
    hist_pdiv_vperp.tofile(fname)
    fname = fdir + 'hist_pshear.' + str(tindex) + '.' + str(rank)
    hist_pshear.tofile(fname)
    del hist_pdivv, hist_pdiv_vperp, hist_pshear, bin_edges
    del pdivv, pdiv_vperp, pshear

    hist_nptl, bin_edges = np.histogram(gamma-1, bins=ebins)
    hist_nptl = hist_nptl.astype(np.float)
    fname = fdir + 'hist_nptl.' + str(tindex) + '.' + str(rank)
    hist_nptl.tofile(fname)
    del hist_nptl, bin_edges
    del gamma


def combine_files(nprocs, run_dir, tindex, data_dir, var_name, species='e'):
    """
    """
    fdir = run_dir + data_dir + '/'
    fname = fdir + var_name + '.' + str(tindex) + '.' + str(0)
    fdata = np.fromfile(fname)
    for rank in range(1, nprocs):
        print("rank %d" % (rank))
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
            origin='lower', extent=[xmin,xmax,ymin,ymax],
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
    nbins = 50
    drange = [[1, 1.1], [0, 1]]
    ebins = np.logspace(-3, 2, nbins)
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
    hist_pdivv = np.fromfile(fname)
    fname = fdir + 'hist_pdiv_vperp' + fname_post
    hist_pdiv_vperp = np.fromfile(fname)
    fname = fdir + 'hist_pshear' + fname_post
    hist_pshear = np.fromfile(fname)

    hist_de_para = np.resize(hist_de_para, (nbins))
    hist_de_perp = np.resize(hist_de_perp, (nbins))
    hist_de_vxb = np.resize(hist_de_vxb, (nbins))
    hist_nptl = np.resize(hist_nptl, (nbins))
    hist_pdivv = np.resize(hist_pdivv, (nbins))
    hist_pdiv_vperp = np.resize(hist_pdiv_vperp, (nbins))
    hist_pshear = np.resize(hist_pshear, (nbins))

    nsum = 1
    hist_de_para = np.sum(hist_de_para.reshape([nbins/nsum, nsum]), axis=1)
    hist_de_perp = np.sum(hist_de_perp.reshape([nbins/nsum, nsum]), axis=1)
    hist_de_vxb = np.sum(hist_de_vxb.reshape([nbins/nsum, nsum]), axis=1)
    hist_nptl = np.sum(hist_nptl.reshape([nbins/nsum, nsum]), axis=1)
    hist_pdivv = np.sum(hist_pdivv.reshape([nbins/nsum, nsum]), axis=1)
    hist_pdiv_vperp = np.sum(hist_pdiv_vperp.reshape([nbins/nsum, nsum]), axis=1)
    hist_pshear = np.sum(hist_pshear.reshape([nbins/nsum, nsum]), axis=1)
    hist_de_para[-1] = hist_de_para[-2]
    hist_de_perp[-1] = hist_de_perp[-2]
    hist_de_vxb[-1] = hist_de_vxb[-2]
    hist_nptl[-1] = hist_nptl[-2]
    hist_pdivv[-1] = hist_pdivv[-2]
    hist_pdiv_vperp[-1] = hist_pdiv_vperp[-2]
    hist_pshear[-1] = hist_pshear[-2]
    emin_log = math.log10(np.min(ebins))
    emax_log = math.log10(np.max(ebins))
    ebins = np.logspace(emin_log, emax_log, nbins/nsum)

    hist_de_tot = hist_de_para + hist_de_perp

    print("Parallel and perpendicular heating: %f %f" %
            (np.sum(hist_de_para), np.sum(hist_de_perp)))
    print("Heating due to ideal electric field: %f" % (np.sum(hist_de_vxb)))
    print("Number of particles: %d" % (np.sum(hist_nptl)))
    print("Compressional and shear heating: %f %f %f" %
            (np.sum(hist_pdivv), np.sum(hist_pdiv_vperp), np.sum(hist_pshear)))

    if if_normalize:
        hist_de_para = div0(hist_de_para, hist_nptl+0.0)
        hist_de_perp = div0(hist_de_perp, hist_nptl+0.0)
        hist_de_vxb = div0(hist_de_vxb, hist_nptl+0.0)
        hist_pdivv = div0(hist_pdivv, hist_nptl+0.0)
        hist_pdiv_vperp = div0(hist_pdiv_vperp, hist_nptl+0.0)
        hist_pshear = div0(hist_pshear, hist_nptl+0.0)

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
    # ax1.semilogx(ebins, hist_pdivv, linewidth=2, color='g',
    #              label=r'$-p\nabla\cdot\boldsymbol{u}$')
    ax1.semilogx(ebins, hist_pdiv_vperp + hist_pshear, linewidth=2,
                 label=r'$-p\nabla\cdot\boldsymbol{u}_\perp$')
    ax1.semilogx(ebins, hist_pshear, linewidth=2,
                 label=r'$-(p_\parallel-p_\perp)b_ib_j\sigma_{ij}$')
    # ax1.loglog(ebins, hist_nptl, linewidth=2, color='k', label='Total')
    ax1.set_xlim([1E-3, 20])
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
    plt.show()


def hist_multi_timeframes(pic_info, ranks):
    """
    """
    pint = pic_info.particle_interval
    ntp = pic_info.ntp
    cmd = 'rm data_ene/hist_*'
    for ct in range(1, ntp+1):
        print("Time frame: %d of %d" % (ct, ntp))
        tindex = pint * ct
        # plot_hist_de_para_perp(nprocs, run_dir, run_name, tindex)
        Parallel(n_jobs=ncores)(delayed(processInput)(rank) for rank in ranks)
        combine_files(nprocs, run_dir, tindex, 'data_ene', 'hist_de_para')
        combine_files(nprocs, run_dir, tindex, 'data_ene', 'hist_de_perp')
        combine_files(nprocs, run_dir, tindex, 'data_ene', 'hist_nptl')
        combine_files(nprocs, run_dir, tindex, 'data_ene', 'hist_pdivv')
        combine_files(nprocs, run_dir, tindex, 'data_ene', 'hist_pdiv_vperp')
        p1 = subprocess.Popen([cmd], cwd=run_dir, stdout=open('outfile.out', 'w'),
                stderr=subprocess.STDOUT, shell=True)
        p1.wait()
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compression analysis based on particles')
    parser.add_argument('--is_combine_files', action="store_true", default=False)
    parser.add_argument('--is_normalize', action="store_true", default=False)
    parser.add_argument('--species', action="store", default='e')
    args = parser.parse_args()
    cmdargs = sys.argv
    if (len(cmdargs) > 2):
        run_dir = cmdargs[1]
        run_name = cmdargs[2]
    else:
        run_dir = '/net/scratch3/xiaocanli/reconnection/mime25-sigma1-beta008-guide00-200-100/'
        run_name = 'mime25_beta008_guide00'
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pint = pic_info.particle_interval
    ntp = pic_info.ntp
    tindex = pint * 4
    rank = 50
    exb_drift = True
    species = args.species
    if_combine_files = args.is_combine_files
    if_normalize = False
    nprocs = pic_info.topology_x * pic_info.topology_y * pic_info.topology_z
    # interp_hydro_particle(pic_info, run_dir, tindex, rank)
    # interp_particle_compression(pic_info, run_dir, tindex, rank, species,
    #                             exb_drift)
    # plot_hist_para_perp(nprocs, run_dir, tindex)
    # calc_pdivv_from_fluid(pic_info, run_dir, tindex)
    ranks = range(nprocs)
    def processInput(job_id):
        print job_id
        rank = job_id
        # interp_hydro_particle(pic_info, run_dir, tindex, rank)
        interp_particle_compression(pic_info, run_dir, tindex, rank, species,
                                    exb_drift)
    ncores = multiprocessing.cpu_count()
    # Parallel(n_jobs=ncores)(delayed(processInput)(rank) for rank in ranks)
    plot_hist_de_para_perp(nprocs, run_dir, run_name, pic_info, tindex, species,
                           if_combine_files, if_normalize)
    # hist_multi_timeframes(pic_info, ranks)
