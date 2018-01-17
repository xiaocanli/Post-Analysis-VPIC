"""
Module contain functions to transfer PIC fields to MHD fields for particle
transport code, which required the single-fluid velocity and magnetic field.
The transport code requires the fields are reorganized to include two ghost
cells at the each side of the simulation domain.
"""
import argparse
import collections
import math
import multiprocessing
import os.path
import struct

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from matplotlib import rc
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D

import pic_information
from contour_plots import read_2d_fields
from json_functions import read_data_from_json
from shell_functions import mkdir_p


def transfer_pic_to_mhd(run_dir, run_name, tframe):
    """Transfer the required fields

    Args:
        run_dir: simulation directory
        run_name: name of the simulation run
        tframe: time frame
        boundary_x: boundary condition along x-direction
                    0 for periodic; 1 for others
        boundary_z: boundary condition along z-direction
    """
    print("Time frame: %d" % tframe)
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    lx_di = pic_info.lx_di
    lz_di = pic_info.lz_di
    kwargs = {"current_time": tframe, "xl": 0, "xr": lx_di,
              "zb": -0.5 * lz_di, "zt": 0.5 * lz_di}
    fname = run_dir + "data/bx.gda"
    x, z, bx = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/by.gda"
    x, z, by = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/bz.gda"
    x, z, bz = read_2d_fields(pic_info, fname, **kwargs)
    nx, = x.shape
    nz, = z.shape
    mhd_data = np.zeros((nz+4, nx+4, 8), dtype=np.float32)

    # We need to switch y and z directions
    absB = np.sqrt(bx**2 + by**2 + bz**2)
    mhd_data[2:nz+2, 2:nx+2, 4] = bx
    mhd_data[2:nz+2, 2:nx+2, 5] = bz
    mhd_data[2:nz+2, 2:nx+2, 6] = -by
    mhd_data[2:nz+2, 2:nx+2, 7] = absB

    del bx, by, bz, absB

    mime = pic_info.mime

    # Electron
    fname = run_dir + "data/ne.gda"
    x, z, ne = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/vex.gda"
    x, z, vex = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/vey.gda"
    x, z, vey = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/vez.gda"
    x, z, vez = read_2d_fields(pic_info, fname, **kwargs)

    vx = ne * vex
    vy = ne * vey
    vz = ne * vez

    del vex, vey, vez

    # Ion
    fname = run_dir + "data/ni.gda"
    x, z, ni = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/vix.gda"
    x, z, vix = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/viy.gda"
    x, z, viy = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/viz.gda"
    x, z, viz = read_2d_fields(pic_info, fname, **kwargs)

    imass = 1.0 / (ne + ni * mime)
    vx = (vx + ni * mime * vix) * imass
    vy = (vy + ni * mime * viy) * imass
    vz = (vz + ni * mime * viz) * imass

    del vix, viy, viz, ne, ni, imass

    absV = np.sqrt(vx**2 + vy**2 + vz**2)
    mhd_data[2:nz+2, 2:nx+2, 0] = vx
    mhd_data[2:nz+2, 2:nx+2, 1] = vz
    mhd_data[2:nz+2, 2:nx+2, 2] = -vy
    mhd_data[2:nz+2, 2:nx+2, 3] = absV

    del vx, vy, vz, absV

    # Assuming periodic boundary along x for fields and particles
    # Assuming conducting boundary along z for fields and reflective for particles
    mhd_data[:, 0:2, :] = mhd_data[:, nx-1:nx+1, :]
    mhd_data[:, nx+2:, :] = mhd_data[:, 3:5, :]
    mhd_data[0:2, :, :] = mhd_data[3:1:-1, :, :]
    mhd_data[nz+2:, :, :] = mhd_data[nz+1:nz-1:-1, :, :]

    fpath = run_dir + 'bin_data/'
    mkdir_p(fpath)
    fname = fpath + 'mhd_data_' + str(tframe).zfill(4)
    print(mhd_data.shape)
    print(np.isfortran(mhd_data))
    mhd_data.tofile(fname)


def save_mhd_config(run_name):
    """Save MHD configuration

    Need to switch y and z directions

    Args:
        run_name: simulation run name
    """
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)

    double_data = np.zeros(13)
    int_data = np.zeros(14, dtype=np.int32)
    smime = math.sqrt(pic_info.mime)
    lx = pic_info.lx_di * smime
    ly = pic_info.lz_di * smime
    lz = 0.0
    nx = pic_info.nx
    ny = pic_info.nz
    nz = pic_info.ny
    if nx > 1:
        double_data[0] = lx / nx
    else:
        double_data[0] = 0.0
    if ny > 1:
        double_data[1] = ly / ny
    else:
        double_data[1] = 0.0
    if nz > 1:
        double_data[2] = lz / nz
    else:
        double_data[2] = 0.0
    double_data[3] = 0.0
    double_data[4] = -0.5 * ly
    double_data[5] = 0.0 
    double_data[6] = lx
    double_data[7] = 0.5 * ly
    double_data[8] = 0.0
    double_data[9]  = lx
    double_data[10] = ly
    double_data[11] = lz
    double_data[12] = pic_info.dtwpe * pic_info.fields_interval
    int_data[0] = pic_info.nx
    int_data[1] = pic_info.nz
    int_data[2] = pic_info.ny
    int_data[3] = pic_info.nx / pic_info.topology_x
    int_data[4] = pic_info.nz / pic_info.topology_z
    int_data[5] = pic_info.ny / pic_info.topology_y
    int_data[6] = pic_info.topology_x
    int_data[7] = pic_info.topology_z
    int_data[8] = pic_info.topology_y
    int_data[9] = 9
    int_data[10] = 0 # Periodic boundary condition as default
    int_data[11] = 0
    int_data[12] = 0

    int_data[13] = 0

    fpath = run_dir + 'bin_data/'
    mkdir_p(fpath)
    fname = fpath + 'mhd_config.dat'
    double_data.tofile(fname)
    f = open(fname, 'a')
    int_data.tofile(f)
    f.close()


def get_cmd_args():
    """Get command line arguments
    """
    default_run_name = 'mime25_beta002_guide00_frequent_dump'
    default_run_dir = '/net/scratch3/xiaocanli/reconnection/frequent_dump/' + \
            'mime25_beta002_guide00_frequent_dump/'
    parser = argparse.ArgumentParser(description='Compression analysis based on fluids')
    parser.add_argument('--species', action="store", default='e',
                        help='particle species')
    parser.add_argument('--run_dir', action="store", default=default_run_dir,
                        help='run directory')
    parser.add_argument('--run_name', action="store", default=default_run_name,
                        help='run name')
    parser.add_argument('--tframe_fields', action="store", default='30', type=int,
                        help='Time frame for fields')
    parser.add_argument('--multi_frames', action="store_true", default=False,
                        help='whether analyzing multiple frames')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_cmd_args()
    run_name = args.run_name
    run_dir = args.run_dir
    species = args.species
    tframe_fields = args.tframe_fields
    multi_frames = args.multi_frames
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    save_mhd_config(run_name)
    # transfer_pic_to_mhd(run_dir, run_name, 0)
    ncores = multiprocessing.cpu_count()
    ncores = 10
    cts = range(pic_info.ntf)
    def processInput(job_id):
        time_frame = job_id
        transfer_pic_to_mhd(run_dir, run_name, time_frame)
    Parallel(n_jobs=ncores)(delayed(processInput)(ct) for ct in cts)
