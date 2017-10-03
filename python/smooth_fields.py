"""
Functions to smooth fields
"""
import argparse
import collections
import gc
import itertools
import math
import os
import os.path
import struct
import subprocess
import sys

import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
from joblib import Parallel, delayed
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
from scipy.interpolate import LinearNDInterpolator
from scipy.ndimage.filters import median_filter, gaussian_filter

from contour_plots import read_2d_fields
from energy_conversion import read_data_from_json


def smooth_electric_field(run_dir, pic_info, efield_name, tframe):
    """
    """
    kwargs = {"current_time": tframe, "xl": 0, "xr": pic_info.lx_di,
              "zb": -0.5 * pic_info.lz_di, "zt": 0.5 * pic_info.lz_di}
    size_one_frame = pic_info.nx * pic_info.nz * 4
    fname = run_dir + "data/" + efield_name + "_original.gda"
    statinfo = os.stat(fname)
    file_size = statinfo.st_size
    if file_size < size_one_frame * (tframe + 1):
        return
    else:
        x, z, fdata = read_2d_fields(pic_info, fname, **kwargs)
    sigma = 3
    fdata = gaussian_filter(fdata, sigma)
    fname = run_dir + "data/" + efield_name + ".gda"
    with open(fname, 'a') as f:
        offset = size_one_frame * tframe
        f.seek(offset, os.SEEK_SET)
        fdata.tofile(f)


def get_cmd_args():
    """Get command line arguments
    """
    default_run_name = 'mime25_beta032_guide00_frequent_dump'
    default_run_dir = ('/net/scratch3/xiaocanli/reconnection/frequent_dump/' +
                       default_run_name + '/')
    parser = argparse.ArgumentParser(description='Smooth fields')
    parser.add_argument('--species', action="store", default='e',
                        help='particle species')
    parser.add_argument('--run_dir', action="store", default=default_run_dir,
                        help='run directory')
    parser.add_argument('--run_name', action="store", default=default_run_name,
                        help='run name')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_cmd_args()
    run_name = args.run_name
    run_dir = args.run_dir
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    tframes = range(pic_info.ntf)
    runs_root_dir = "/net/scratch3/xiaocanli/reconnection/frequent_dump/"
    run_names = ["mime25_beta002_guide00_frequent_dump",
                 "mime25_beta002_guide02_frequent_dump",
                 "mime25_beta002_guide05_frequent_dump",
                 "mime25_beta002_guide10_frequent_dump",
                 "mime25_beta008_guide00_frequent_dump",
                 "mime25_beta032_guide00_frequent_dump"]
    enames = ["ex", "ey", "ez"]
    suffixs = ["", "_pre", "_post"]
    efield_names = [ename + suffixs for ename, suffixs in
                    itertools.product(enames, suffixs)]
    run_efields = [{"run_name": run_name, "efield_name": efield_name}
                   for run_name, efield_name
                   in itertools.product(run_names, efield_names)]
    # for tframe in range(10):
    #     print("Time frame: %d" % tframe)
    #     smooth_electric_field(run_dir, pic_info, "ex", tframe)
    def processInput(run_efield):
        run_name = run_efield["run_name"]
        picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
        efield_name = run_efield["efield_name"]
        print("Run name and electric field name: %s %s" % (run_name, efield_name))
        run_dir = runs_root_dir + run_name + '/'
        for tframe in tframes:
            smooth_electric_field(run_dir, pic_info, efield_name, tframe)
    ncores = multiprocessing.cpu_count()
    Parallel(n_jobs=ncores)(delayed(processInput)(run_efield)
                            for run_efield in run_efields)
