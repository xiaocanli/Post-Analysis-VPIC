"""
Procedures to calculate derived fields
"""
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
import multiprocessing
import numpy as np
from joblib import Parallel, delayed
from matplotlib import rc

import pic_information
from energy_conversion import read_data_from_json
from shell_functions import mkdir_p


def calc_vsingle(run_dir, mime):
    """Calculate single fluid velocity
    """
    fname = run_dir + 'data/ne.gda'
    ne = np.fromfile(fname, dtype=np.float32)
    fname = run_dir + 'data/ni.gda'
    ni = np.fromfile(fname, dtype=np.float32)
    inrho = 1.0 / (ne + ni * mime)

    fdir = run_dir + 'data1/'
    mkdir_p(fdir)

    fname = run_dir + 'data/vex.gda'
    ve = np.fromfile(fname, dtype=np.float32)
    fname = run_dir + 'data/vix.gda'
    vi = np.fromfile(fname, dtype=np.float32)
    vs = (ve * ne + vi * ni * mime) * inrho
    vs.tofile(fdir + 'vx.gda')

    fname = run_dir + 'data/vey.gda'
    ve = np.fromfile(fname, dtype=np.float32)
    fname = run_dir + 'data/viy.gda'
    vi = np.fromfile(fname, dtype=np.float32)
    vs = (ve * ne + vi * ni * mime) * inrho
    vs.tofile(fdir + 'vy.gda')

    fname = run_dir + 'data/vez.gda'
    ve = np.fromfile(fname, dtype=np.float32)
    fname = run_dir + 'data/viz.gda'
    vi = np.fromfile(fname, dtype=np.float32)
    vs = (ve * ne + vi * ni * mime) * inrho
    vs.tofile(fdir + 'vz.gda')


if __name__ == "__main__":
    cmdargs = sys.argv
    if (len(cmdargs) > 2):
        run_dir = cmdargs[1]
        run_name = cmdargs[2]
    else:
        run_dir = '/net/scratch3/xiaocanli/reconnection/mime25-sigma1-beta002-guide00-200-100/'
        run_name = 'mime25_beta002_guide00'
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    mime = pic_info.mime
    calc_vsingle(run_dir, mime)
    def processInput(job_id):
        print job_id
    ncores = multiprocessing.cpu_count()
    # Parallel(n_jobs=ncores)(delayed(processInput)(rank) for rank in ranks)
