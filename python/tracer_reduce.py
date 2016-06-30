"""
Analysis procedures to deal with reduced particle tracer
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
import h5py
from shell_functions import *
from particle_emf import *

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

font = {'family' : 'serif',
        #'color'  : 'darkred',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 24,
        }

def read_var_single(group, dset_name):
    """Read only a single data point from a HDF5 group
    """
    dset = group[dset_name]
    return dset[0]


def get_meta_data(pic_info):
    """Get tracer meta data
    """
    fname = '../../tracer/T.0/grid_metadata_ion_tracer.h5p'
    meta_data = {}
    with h5py.File(fname, 'r') as fh:
        key = 'Step#0'
        group = fh[key]
        dset = group['np_local']
        sz, = dset.shape
        np_local = read_var(group, 'np_local', sz)
        x0 = read_var(group, 'x0', sz)
        y0 = read_var(group, 'y0', sz)
        z0 = read_var(group, 'z0', sz)
        dx = read_var_single(group, 'dx')
        dy = read_var_single(group, 'dy')
        dz = read_var_single(group, 'dz')
        nx = read_var_single(group, 'nx')
        ny = read_var_single(group, 'ny')
        nz = read_var_single(group, 'nz')

    dx_mpi =  nx * dx
    dy_mpi =  ny * dy
    dz_mpi =  nz * dz

    meta_data = {'np_local': np_local, 'x0': x0, 'y0': y0, 'z0': z0,
                 'grid_size_mpi': [dx_mpi, dy_mpi, dz_mpi],
                 'grid_size': [dx, dy, dz],
                 'grid_dims': [nx, ny, nz]}

    return meta_data


def sort_tracer_data(pic_info, pmin, meta_data, ct, species, root_path='../../'):
    """Sort tracer data
    """
    fpath = root_path + 'tracer/T.' + str(ct) + '/' 
    fname_reduced = fpath + species + '_tracer_reduced.h5p'
    with h5py.File(fname_reduced, 'r') as fh:
        key = 'Step#' + str(ct)
        group = fh[key]
        dset = group['q']
        sz, = dset.shape
        dX = read_var(group, 'dX', sz)
        dY = read_var(group, 'dY', sz)
        dZ = read_var(group, 'dZ', sz)
        Ux = read_var(group, 'Ux', sz)
        Uy = read_var(group, 'Uy', sz)
        Uz = read_var(group, 'Uz', sz)
        icell = read_var(group, 'i', sz)
        q = read_var(group, 'q', sz)

    grid_size_mpi = meta_data['grid_size_mpi']
    grid_dims = meta_data['grid_dims']
    grid_size = meta_data['grid_size']

    tpx = pic_info.topology_x
    tpy = pic_info.topology_y
    tpz = pic_info.topology_z
    xmin, ymin, zmin = pmin
    dx_mpi, dy_mpi, dz_mpi = grid_size_mpi
    dx, dy, dz = grid_size
    nx, ny, nz = grid_dims
    ix = (dX - xmin) // dx_mpi
    iy = (dY - ymin) // dy_mpi
    iz = (dZ - zmin) // dz_mpi
    ix[ix > tpx - 1] = tpx - 1
    iy[iy > tpy - 1] = tpy - 1
    iz[iz > tpz - 1] = tpz - 1
    mpi_rank = ix + iy * tpx + iz * tpx * tpy
    mpi_rank = mpi_rank.astype(np.int32)
    nx1 = nx + 2
    ny1 = ny + 2
    nz1 = nz + 2
    izp = icell / (nx1 * ny1)
    iyp = (icell % (nx1 * ny1)) / nx1
    ixp = icell - izp * nx1 * ny1 - iyp * nx1
    idx = 1 / dx
    idy = 1 / dy
    idz = 1 / dz
    dX = ((dX - ix * dx_mpi - xmin) * idx - ixp + 1) * 2 - 1
    dY = ((dY - iy * dy_mpi - ymin) * idy - iyp + 1) * 2 - 1
    dZ = ((dZ - iz * dz_mpi - zmin) * idz - izp + 1) * 2 - 1
    dX[(dX < -1) & (ixp == nx)] = 1.0
    dX[(dX > 1) & (ixp == 1)] = -1.0
    dY[(dY < -1) & (iyp == ny)] = 1.0
    dY[(dY > 1) & (iyp == 1)] = -1.0
    dZ[(dZ < -1) & (izp == nz)] = 1.0
    dZ[(dZ > 1) & (izp == 1)] = -1.0

    sort_index = np.argsort(mpi_rank)
    dX = dX[sort_index]
    dY = dY[sort_index]
    dZ = dZ[sort_index]
    Ux = Ux[sort_index]
    Uy = Uy[sort_index]
    Uz = Uz[sort_index]
    icell = icell[sort_index]
    q = q[sort_index]
    
    ncpu = tpx * tpy * tpz
    elements, repeats = np.unique(mpi_rank, return_counts=True)

    fname = fpath + 'grid_metadata' + species + '_tracer_reduced.h5p'
    with h5py.File(fname, 'w') as fh:
        gname = 'Step#' + str(ct)
        grp = fh.create_group(gname)
        grp.create_dataset('dx', (1, ), data=grid_size[0])
        grp.create_dataset('dy', (1, ), data=grid_size[1])
        grp.create_dataset('dz', (1, ), data=grid_size[2])
        grp.create_dataset('nx', (1, ), data=grid_dims[0])
        grp.create_dataset('ny', (1, ), data=grid_dims[1])
        grp.create_dataset('nz', (1, ), data=grid_dims[2])
        grp.create_dataset('x0', (ncpu, ), data=meta_data['x0'])
        grp.create_dataset('y0', (ncpu, ), data=meta_data['y0'])
        grp.create_dataset('z0', (ncpu, ), data=meta_data['z0'])
        grp.create_dataset('np_local', (ncpu, ), data=repeats)

    nptl, = dX.shape
    fname_reduced_sorted = fpath + species + '_tracer_reduced_sorted.h5p'
    with h5py.File(fname_reduced, 'w') as fh:
        gname = 'Step#' + str(ct)
        grp = fh.create_group(gname)
        grp.create_dataset('dX', (nptl, ), data=dX)
        grp.create_dataset('dY', (nptl, ), data=dY)
        grp.create_dataset('dZ', (nptl, ), data=dZ)
        grp.create_dataset('Ux', (nptl, ), data=Ux)
        grp.create_dataset('Uy', (nptl, ), data=Uy)
        grp.create_dataset('Uz', (nptl, ), data=Uz)
        grp.create_dataset('i', (nptl, ), data=icell)
        grp.create_dataset('q', (nptl, ), data=q)


if __name__ == "__main__":
    pic_info = pic_information.get_pic_info('../../')
    meta_data = get_meta_data(pic_info)
    xmin = np.min(meta_data['x0'])
    ymin = np.min(meta_data['y0'])
    zmin = np.min(meta_data['z0'])
    pmin = [xmin, ymin, zmin]
    for ct in xrange(0, 770, 7):
        print ct
        sort_tracer_data(pic_info, pmin, meta_data, ct, 'ion')
