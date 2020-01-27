#!/usr/bin/env python3
"""
Analysis procedures for 2D contour plots.
"""
from __future__ import print_function

import argparse
import itertools
import json
import math
import multiprocessing

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import pic_information
from shell_functions import mkdir_p

plt.style.use("seaborn-deep")
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = \
[r"\usepackage{amsmath, bm}",
 r"\DeclareMathAlphabet{\mathsfit}{\encodingdefault}{\sfdefault}{m}{sl}",
 r"\SetMathAlphabet{\mathsfit}{bold}{\encodingdefault}{\sfdefault}{bx}{sl}",
 r"\newcommand{\tensorsym}[1]{\bm{\mathsfit{#1}}}"]
COLORS = palettable.colorbrewer.qualitative.Set1_9.mpl_colors


def read_2D_fields(config):
    """Read 2D fields data from binary or HDF5 files

    Args:
        config(dict): configuration for reading the data
    """
    var = config["var"]
    run_dir = config["run_dir"]
    tframe = config["tframe"]
    if config["hdf5"]:
        if var in ["bx", "by", "bz", "ex", "ey", "ez"]:
            if "b" in var:
                var = "c" + var
            data_dir = run_dir + "field_hdf5/"
        else:
            data_dir = run_dir + "hydro_hdf5/"
    else:
        data_dir = run_dir + "data/"

    fname = config["run_dir"] +
    print("Reading data from %s" % fname)
    print("xrange: (%f di, %f di)" % (xl, xr))
    print("zrange: (%f di, %f di)" % (zb, zt))
    nx = pic_info.nx
    nz = pic_info.nz
    x_di = np.copy(pic_info.x_di)
    z_di = np.copy(pic_info.z_di)
    dx_di = pic_info.dx_di
    dz_di = pic_info.dz_di
    xmin = np.min(x_di)
    xmax = np.max(x_di)
    zmin = np.min(z_di)
    zmax = np.max(z_di)
    if (xl <= xmin):
        xl_index = 0
    else:
        xl_index = int(math.floor((xl - xmin) / dx_di))
    if (xr >= xmax):
        xr_index = nx - 1
    else:
        xr_index = int(math.ceil((xr - xmin) / dx_di))
    if (zb <= zmin):
        zb_index = 0
    else:
        zb_index = int(math.floor((zb - zmin) / dz_di))
    if (zt >= zmax):
        zt_index = nz - 1
    else:
        zt_index = int(math.ceil((zt - zmin) / dz_di))
    nx1 = xr_index - xl_index + 1
    nz1 = zt_index - zb_index + 1
    fp = np.zeros((nz1, nx1), dtype=np.float32)
    offset = nx * nz * tframe * 4
    fdata = np.memmap(fname, dtype='float32',
                      mode='r', offset=offset,
                      shape=(nz, nx), order='C')
    xc = x_di[xl_index:xr_index + 1]
    zc = z_di[zb_index:zt_index + 1]
    fp = fdata[zb_index:zt_index + 1, xl_index:xr_index + 1]
    return (xc, zc, fp)


if __name__ == "__main__":
    pass
