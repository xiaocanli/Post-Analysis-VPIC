"""
Analysis procedures for particle energy spectrum.
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

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

font = {'family' : 'serif',
        #'color'  : 'darkred',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 24,
        }

def read_boilerplate(fh):
    """Read boilerplate of a file

    Args:
        fh: file handler
    """
    offset = 0
    sizearr = np.memmap(fh, dtype='int8', mode='r', offset=offset,
            shape=(5), order='F')
    offset += 5
    cafevar = np.memmap(fh, dtype='int16', mode='r', offset=offset,
            shape=(1), order='F')
    offset += 2
    deadbeefvar = np.memmap(fh, dtype='int32', mode='r', offset=offset,
            shape=(1), order='F')
    offset += 4
    realone = np.memmap(fh, dtype='float32', mode='r', offset=offset,
            shape=(1), order='F')
    offset += 4
    doubleone = np.memmap(fh, dtype='float64', mode='r', offset=offset,
            shape=(1), order='F')


def read_particle_header(fh):
    """Read particle file header

    Args:
        fh: file handler.
    """
    offset = 23     # the size of the boilerplate is 23
    tmp1 = np.memmap(fh, dtype='int32', mode='r', offset=offset,
            shape=(6), order='F')
    offset += 6 * 4
    tmp2 = np.memmap(fh, dtype='float32', mode='r', offset=offset,
            shape=(10), order='F')
    offset += 10 * 4
    tmp3 = np.memmap(fh, dtype='int32', mode='r', offset=offset,
            shape=(4), order='F')
    v0header = collections.namedtuple("v0header", ["version", "type", "nt",
        "nx", "ny", "nz", "dt", "dx", "dy", "dz", "x0", "y0", "z0", "cvac",
        "eps0", "damp", "rank", "ndom", "spid", "spqm"])
    v0 = v0header(version=tmp1[0], type=tmp1[1], nt=tmp1[2], nx=tmp1[3],
            ny=tmp1[4], nz=tmp1[5], dt=tmp2[0], dx=tmp2[1], dy=tmp2[2],
            dz=tmp2[3], x0=tmp2[4], y0=tmp2[5], z0=tmp2[6], cvac=tmp2[7],
            eps0=tmp2[8], damp=tmp2[9], rank=tmp3[0], ndom=tmp3[1],
            spid=tmp3[2], spqm=tmp3[3])
    header_particle = collections.namedtuple("header_particle", ["size",
        "ndim", "dim"])
    offset += 4 * 4
    tmp4 = np.memmap(fh, dtype='int32', mode='r', offset=offset,
            shape=(3), order='F')
    pheader = header_particle(size=tmp4[0], ndim=tmp4[1], dim=tmp4[2])
    offset += 3 * 4
    return (v0, pheader, offset)


def read_particle_data(fname):
    """Read particle information from a file.

    Args:
        fname: file name.
    """
    fh = open(fname, 'r')
    read_boilerplate(fh)
    v0, pheader, offset = read_particle_header(fh)
    nptl = pheader.dim
    particle_type = np.dtype([('dxyz', np.float32, 3), ('icell', np.int32),
            ('u', np.float32, 3), ('q', np.float32)])
    fh.seek(offset, os.SEEK_SET)
    data = np.fromfile(fh, dtype=particle_type, count=nptl)
    fh.close()
    return (v0, pheader, data)


def calc_velocity_distribution(v0, pheader, ptl, pic_info, corners, nbins):
    """Calculate particle velocity distribution

    Args:
        v0: the header info for the grid.
        pheader: the header info for the particles.
        pic_info: namedtuple for the PIC simulation information.
        corners: the corners of the box in di.
        nbins: number of bins in each dimension.
    """
    dx = ptl['dxyz'][:, 0]
    dy = ptl['dxyz'][:, 1]
    dz = ptl['dxyz'][:, 2]
    icell = ptl['icell']
    ux = ptl['u'][:, 0]
    uy = ptl['u'][:, 1]
    uz = ptl['u'][:, 2]

    nx = v0.nx + 2
    ny = v0.ny + 2
    nz = v0.nz + 2
    iz = icell // (nx*ny)
    iy = (icell - iz*nx*ny) // nx
    ix = icell - iz*nx*ny - iy*nx

    z = v0.z0 + ((iz - 1.0) + (dz + 1.0) * 0.5) * v0.dz
    y = v0.y0 + ((iy - 1.0) + (dy + 1.0) * 0.5) * v0.dy
    x = v0.x0 + ((ix - 1.0) + (dx + 1.0) * 0.5) * v0.dx

    # di -> de
    smime = math.sqrt(pic_info.mime)
    x /= smime
    y /= smime
    z /= smime

    mask = ((x >= corners[0][0]) & (x <= corners[0][1]) &
            (y >= corners[1][0]) & (y <= corners[1][1]) &
            (z >= corners[2][0]) & (z <= corners[2][1]))
    ux_d = ux[mask]
    uy_d = uy[mask]
    uz_d = uz[mask]

    range = [[-1.0, 1.0], [-1.0, 1.0]]
    hist_xy, xedges, yedges = np.histogram2d(uy_d, ux_d, 
            bins=nbins, range=range)
    hist_xz, xedges, yedges = np.histogram2d(uz_d, ux_d, 
            bins=nbins, range=range)
    hist_yz, xedges, yedges = np.histogram2d(uz_d, uy_d, 
            bins=nbins, range=range)

    return (hist_xy, hist_xz, hist_yz, xedges, yedges)


def get_particle_distribution(base_directory, tindex, corners, mpi_ranks):
    """Read particle information.

    Args:
        base_directory: the base directory for the simulation data.
        tindex: the time index.
        corners: the corners of the box in di.
        mpi_ranks: PIC simulation MPI ranks for a selected region.
    """
    pic_info = pic_information.get_pic_info(base_directory)
    dir_name = base_directory + 'particle/T.' + str(tindex) + '/'
    fbase = dir_name + 'eparticle' + '.' + str(tindex) + '.'
    tx = pic_info.topology_x
    ty = pic_info.topology_y
    tz = pic_info.topology_z
    nbins = 64
    hist_xy = np.zeros((nbins, nbins))
    hist_xz = np.zeros((nbins, nbins))
    hist_yz = np.zeros((nbins, nbins))
    mpi_ranks = np.asarray(mpi_ranks)
    for ix in range(mpi_ranks[0, 0], mpi_ranks[0, 1]+1):
        for iy in range(mpi_ranks[1, 0], mpi_ranks[1, 1]+1):
            for iz in range(mpi_ranks[2, 0], mpi_ranks[2, 1]+1):
                mpi_rank = ix + iy*tx + iz*tx*ty
                fname = fbase + str(mpi_rank)
                (v0, pheader, data) = read_particle_data(fname)
                (vhist_xy, vhist_xz, vhist_yz, x, y) = \
                        calc_velocity_distribution(v0, pheader,
                        data, pic_info, corners, nbins)
                hist_xy += vhist_xy
                hist_xz += vhist_xz
                hist_yz += vhist_yz
    vmax = np.max([np.max(hist_xy), np.max(hist_xz), np.max(hist_yz)])
    vmax = math.log10(vmax)
    xs, ys = 0.04, 0.1
    w1, h1 = 0.28, 0.84
    gap = 0.05
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_axes([xs, ys, w1, h1])
    p1 = ax1.imshow(np.log10(hist_xy), cmap=plt.cm.jet,
            extent=[np.min(x), np.max(x), np.min(y), np.max(y)],
            aspect='auto', origin='lower',
            vmin = 0.0, vmax = vmax)
            # interpolation='bicubic')
    xs += w1 + 0.05
    ax2 = fig.add_axes([xs, ys, w1, h1])
    p2 = ax2.imshow(np.log10(hist_xz), cmap=plt.cm.jet,
            extent=[np.min(x), np.max(x), np.min(y), np.max(y)],
            aspect='auto', origin='lower',
            vmin = 0.0, vmax = vmax)
            # interpolation='bicubic')
    xs += w1 + 0.05
    ax3 = fig.add_axes([xs, ys, w1, h1])
    p3 = ax3.imshow(np.log10(hist_yz), cmap=plt.cm.jet,
            extent=[np.min(x), np.max(x), np.min(y), np.max(y)],
            aspect='auto', origin='lower',
            vmin = 0.0, vmax = vmax)
            # interpolation='bicubic')
    p1.set_cmap(plt.cm.get_cmap('idl40'))
    p2.set_cmap(plt.cm.get_cmap('idl40'))
    p3.set_cmap(plt.cm.get_cmap('idl40'))
    plt.show()



def set_mpi_ranks(pic_info, center=np.zeros(3), sizes=np.ones(3)*4):
    """Set MPI ranks for getting particle data

    Args:
        pic_info: namedtuple for the PIC simulation information.
        center: the center of a box in di.
        sizes: the sizes of the box in grids.
    Returns:
        corners: the corners of the box in di.
        mpi_ranks: MPI ranks in which the box is.
    """
    # The domain sizes for each MPI process (in di)
    dx_domain = pic_info.lx_di / pic_info.topology_x
    dy_domain = pic_info.ly_di / pic_info.topology_y
    dz_domain = pic_info.lz_di / pic_info.topology_z
    lx_di = pic_info.lx_di
    ly_di = pic_info.ly_di
    lz_di = pic_info.lz_di

    # The sizes of each cell
    dx_di = pic_info.dx_di
    dy_di = pic_info.dy_di
    dz_di = pic_info.dz_di
    hsize = sizes / 2.0
    xs = center[0] - hsize[0] * dx_di
    xe = center[0] + hsize[0] * dx_di
    ys = center[1] - hsize[1] * dy_di
    ye = center[1] + hsize[1] * dy_di
    zs = center[2] - hsize[2] * dz_di
    ze = center[2] + hsize[2] * dz_di

    # x in [0, lx_di], y in [-ly_di/2, ly_di/2], z in [-lz_di/2, lz_di/2]
    if (xs < 0): xs = 0.0
    if (xs > lx_di): xs = lx_di
    if (xe < 0): xe = 0.0
    if (xe > lx_di): xe = lx_di
    if (ys < -ly_di*0.5): ys = -ly_di*0.5
    if (ys > ly_di*0.5): ys = ly_di*0.5
    if (ye < -ly_di*0.5): ye = -ly_di*0.5
    if (ye > ly_di*0.5): ye = ly_di*0.5
    if (zs < -lz_di*0.5): zs = -lz_di*0.5
    if (zs > lz_di*0.5): zs = lz_di*0.5
    if (ze < -lz_di*0.5): ze = -lz_di*0.5
    if (ze > lz_di*0.5): ze = lz_di*0.5

    ixs = int(math.floor(xs / dx_domain))
    ixe = int(math.floor(xe / dx_domain))
    iys = int(math.floor((ys + ly_di*0.5) / dy_domain))
    iye = int(math.floor((ye + ly_di*0.5) / dy_domain))
    izs = int(math.floor((zs + lz_di*0.5) / dz_domain))
    ize = int(math.floor((ze + lz_di*0.5) / dz_domain))
    if (ixe >= pic_info.topology_x):
        ixe = pic_info.topology_x - 1
    if (iye >= pic_info.topology_y):
        iye = pic_info.topology_y - 1
    if (ize >= pic_info.topology_z):
        ize = pic_info.topology_z - 1

    corners = np.zeros((3, 2))
    mpi_ranks = np.zeros((3, 2))
    corners = [[xs, xe], [ys, ye], [zs, ze]]
    mpi_ranks = [[ixs, ixe], [iys, iye], [izs, ize]]
    return (corners, mpi_ranks)


if __name__ == "__main__":
    base_directory = '../../'
    pic_info = pic_information.get_pic_info(base_directory)
    ntp = pic_info.ntp
    vthe = pic_info.vthe
    particle_interval = pic_info.particle_interval
    pos = [pic_info.lx_di/2, 0.0, 40.0]
    corners, mpi_ranks = set_mpi_ranks(pic_info, pos)
    ct = 1 * particle_interval
    get_particle_distribution(base_directory, ct, corners, mpi_ranks)
