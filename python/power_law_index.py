#!/usr/bin/env python3
"""
Particle energy spectrum for runs to determine power-law indices
"""
from __future__ import print_function

import argparse
import itertools
import json
import math
import multiprocessing
import os

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager
import numpy as np
import palettable
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
from matplotlib.patches import PathPatch, Path
from scipy.optimize import curve_fit
from scipy import signal, stats
from scipy.ndimage.filters import gaussian_filter
from scipy.special import kv

import fitting_funcs
import pic_information
from contour_plots import read_2d_fields
from joblib import Parallel, delayed
from json_functions import read_data_from_json
from shell_functions import mkdir_p

plt.style.use("seaborn-deep")
mpl.rc('text', usetex=True)
mpl.rcParams["text.latex.preamble"] = \
        (r"\usepackage{amsmath, bm}" +
         r"\DeclareMathAlphabet{\mathsfit}{\encodingdefault}{\sfdefault}{m}{sl}" +
         r"\SetMathAlphabet{\mathsfit}{bold}{\encodingdefault}{\sfdefault}{bx}{sl}" +
         r"\newcommand{\tensorsym}[1]{\bm{\mathsfit{#1}}}")
COLORS = palettable.colorbrewer.qualitative.Set1_9.mpl_colors


def find_nearest(array, value):
    """Find nearest value in an array
    """
    idx = (np.abs(array-value)).argmin()
    return (idx, array[idx])


def div0(a, b):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0]

    From: http://stackoverflow.com/a/35696047/2561161

    """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[~np.isfinite(c)] = 0  # -inf inf NaN
    return c


def CurlyBrace(x, y, width=1/8, height=1., curliness=1/np.e, pointing='left', **patch_kw):
    '''Source: https://github.com/bensondaled/curly_brace
    Create a matplotlib patch corresponding to a curly brace (i.e. this thing: "{")
    Parameters
    ----------
    x : float
        x position of left edge of patch
    y : float
        y position of bottom edge of patch
    width : float
        horizontal span of patch
    height : float
        vertical span of patch
    curliness : float
        positive value indicating extent of curliness; default (1/e) tends to look nice
    pointing : str
        direction in which the curly brace points (currently supports 'left' and 'right')
    **patch_kw : any keyword args accepted by matplotlib's Patch
    Returns
    -------
    matplotlib PathPatch corresponding to curly brace

    Notes
    -----
    It is useful to supply the `transform` parameter to specify the coordinate system for the Patch.
    To add to Axes `ax`:
    cb = CurlyBrace(x, y)
    ax.add_artist(cb)
    This has been written as a function that returns a Patch
    because I saw no use in making it a class, though one could extend matplotlib's
    Patch as an alternate implementation.

    Thanks to:
    https://graphicdesign.stackexchange.com/questions/86334/inkscape-easy-way-to-create-curly-brace-bracket
    http://www.inkscapeforum.com/viewtopic.php?t=11228
    https://css-tricks.com/svg-path-syntax-illustrated-guide/
    https://matplotlib.org/users/path_tutorial.html
    Ben Deverett, 2018.
    Examples
    --------
    >>>from curly_brace_patch import CurlyBrace
    >>>import matplotlib.pyplot as pl
    >>>fig,ax = pl.subplots()
    >>>brace = CurlyBrace(x=.4, y=.2, width=.2, height=.6, pointing='right', transform=ax.transAxes, color='magenta')
    >>>ax.add_artist(brace)
    '''

    verts = np.array([[width,0],
                      [0,0],
                      [width, curliness],
                      [0,.5],
                      [width, 1-curliness],
                      [0,1],
                      [width,1]])

    if pointing == 'left':
        pass
    elif pointing == 'right':
        verts[:,0] = width - verts[:,0]

    verts[:,1] *= height

    verts[:,0] += x
    verts[:,1] += y

    codes = [Path.MOVETO,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             Path.CURVE4,
             ]

    path = Path(verts, codes)

    # convert `color` parameter to `edgecolor`, since that's the assumed intention
    patch_kw['edgecolor'] = patch_kw.pop('color', 'black')

    pp = PathPatch(path, facecolor='none', **patch_kw)
    return pp


def get_vpic_info(pic_run_dir):
    """Get information of the VPIC simulation
    """
    with open(pic_run_dir + '/info') as f:
        content = f.readlines()
    f.close()
    vpic_info = {}
    for line in content[1:]:
        if "=" in line:
            line_splits = line.split("=")
        elif ":" in line:
            line_splits = line.split(":")

        tail = line_splits[1].split("\n")
        vpic_info[line_splits[0].strip()] = float(tail[0])
    return vpic_info


def get_spect_params(pic_run="sigma01_bg005_4000de_triggered"):
    """power law parameters for different runs
    """
    spect_params = {}
    if pic_run == "sigma01_bg005_4000de_triggered":
        spect_params["power_index"] = -2.5
        spect_params["energy_range"] = [20, 80]
        spect_params["emax"] = 1E3
        spect_params["norm"] = 3.0
    elif pic_run == "sigma04_bg005_4000de_triggered":
        spect_params["power_index"] = -2.8
        spect_params["energy_range"] = [40, 300]
        spect_params["emax"] = 2E3
        spect_params["norm"] = 3.0
    elif pic_run == "sigma16_bg005_4000de_triggered":
        spect_params["power_index"] = -2.5
        spect_params["energy_range"] = [40, 300]
        spect_params["emax"] = 2E3
        spect_params["norm"] = 3.0
    elif pic_run == "sigma64_bg005_4000de_triggered":
        spect_params["power_index"] = -1.7
        spect_params["energy_range"] = [10, 200]
        spect_params["emax"] = 2E3
        spect_params["norm"] = 3.0
    elif pic_run == "sigma256_bg005_4000de_triggered":
        spect_params["power_index"] = -1.8
        spect_params["energy_range"] = [20, 200]
        spect_params["emax"] = 1E3
        spect_params["norm"] = 3.0
    elif pic_run == "sigma01_bg01_4000de_triggered":
        spect_params["power_index"] = -4.5
        spect_params["energy_range"] = [70, 400]
        spect_params["emax"] = 2E3
        spect_params["norm"] = 3.0
    elif pic_run == "sigma04_bg01_4000de_triggered":
        spect_params["power_index"] = -3.0
        spect_params["energy_range"] = [40, 300]
        spect_params["emax"] = 2E3
        spect_params["norm"] = 3.0
    elif pic_run == "sigma16_bg01_4000de_triggered":
        spect_params["power_index"] = -2.5
        spect_params["energy_range"] = [40, 300]
        spect_params["emax"] = 2E3
        spect_params["norm"] = 3.0
    elif pic_run == "sigma64_bg01_4000de_triggered":
        spect_params["power_index"] = -1.7
        spect_params["energy_range"] = [10, 200]
        spect_params["emax"] = 2E3
        spect_params["norm"] = 3.0
    elif pic_run == "sigma256_bg01_4000de_triggered":
        spect_params["power_index"] = -1.7
        spect_params["energy_range"] = [20, 200]
        spect_params["emax"] = 1E3
        spect_params["norm"] = 3.0
    elif pic_run == "sigma01_bg02_4000de_triggered":
        spect_params["power_index"] = -3.5
        spect_params["energy_range"] = [30, 200]
        spect_params["emax"] = 1E3
        spect_params["norm"] = 3.0
    elif pic_run == "sigma04_bg02_4000de_triggered":
        spect_params["power_index"] = -3.0
        spect_params["energy_range"] = [40, 400]
        spect_params["emax"] = 2E3
        spect_params["norm"] = 3.0
    elif pic_run == "sigma16_bg02_4000de_triggered":
        spect_params["power_index"] = -2.1
        spect_params["energy_range"] = [20, 200]
        spect_params["emax"] = 2E3
        spect_params["norm"] = 3.0
    elif pic_run == "sigma64_bg02_4000de_triggered":
        spect_params["power_index"] = -1.8
        spect_params["energy_range"] = [10, 200]
        spect_params["emax"] = 2E3
        spect_params["norm"] = 3.0
    elif pic_run == "sigma256_bg02_4000de_triggered":
        spect_params["power_index"] = -1.7
        spect_params["energy_range"] = [20, 200]
        spect_params["emax"] = 1E3
        spect_params["norm"] = 3.0
    elif pic_run == "sigma01_bg04_4000de_triggered":
        spect_params["power_index"] = -3.65
        spect_params["energy_range"] = [30, 100]
        spect_params["emax"] = 1E3
        spect_params["norm"] = 3.0
    elif pic_run == "sigma04_bg04_4000de_triggered":
        spect_params["power_index"] = -2.8
        spect_params["energy_range"] = [30, 200]
        spect_params["emax"] = 2E3
        spect_params["norm"] = 3.0
    elif pic_run == "sigma16_bg04_4000de_triggered":
        spect_params["power_index"] = -2.3
        spect_params["energy_range"] = [20, 300]
        spect_params["emax"] = 2E3
        spect_params["norm"] = 3.0
    elif pic_run == "sigma64_bg04_4000de_triggered":
        spect_params["power_index"] = -1.9
        spect_params["energy_range"] = [10, 200]
        spect_params["emax"] = 2E3
        spect_params["norm"] = 3.0
    elif pic_run == "sigma256_bg04_4000de_triggered":
        spect_params["power_index"] = -1.7
        spect_params["energy_range"] = [20, 200]
        spect_params["emax"] = 1E3
        spect_params["norm"] = 3.0
    elif pic_run == "sigma01_bg10_4000de_triggered":
        spect_params["power_index"] = -3.6
        spect_params["energy_range"] = [50, 300]
        spect_params["emax"] = 1E3
        spect_params["norm"] = 3.0
    elif pic_run == "sigma04_bg10_4000de_triggered":
        spect_params["power_index"] = -3.2
        spect_params["energy_range"] = [40, 300]
        spect_params["emax"] = 2E3
        spect_params["norm"] = 3.0
    elif pic_run == "sigma16_bg10_4000de_triggered":
        spect_params["power_index"] = -2.7
        spect_params["energy_range"] = [30, 300]
        spect_params["emax"] = 2E3
        spect_params["norm"] = 3.0
    elif pic_run == "sigma64_bg10_4000de_triggered":
        spect_params["power_index"] = -2.0
        spect_params["energy_range"] = [10, 200]
        spect_params["emax"] = 2E3
        spect_params["norm"] = 3.0
    elif pic_run == "sigma256_bg10_4000de_triggered":
        spect_params["power_index"] = -1.7
        spect_params["energy_range"] = [20, 100]
        spect_params["emax"] = 1E3
        spect_params["norm"] = 3.0
    elif pic_run == "sigmae25-4_bg00_800de_triggered":
        spect_params["power_index"] = -4.3
        spect_params["energy_range"] = [20, 100]
        spect_params["emax"] = 5E2
        spect_params["norm"] = 3.0
    elif pic_run == "sigmae25_bg00_800de_triggered":
        spect_params["power_index"] = -2.8
        spect_params["energy_range"] = [30, 200]
        spect_params["emax"] = 1E3
        spect_params["norm"] = 3.0
    elif pic_run == "sigmae100_bg00_800de_triggered":
        spect_params["power_index"] = -1.3
        spect_params["energy_range"] = [20, 300]
        spect_params["emax"] = 5E3
        spect_params["norm"] = 3.0
    elif pic_run == "sigmae400_bg00_800de_triggered":
        spect_params["power_index"] = -1.3
        spect_params["energy_range"] = [10, 1000]
        spect_params["emax"] = 2E4
        spect_params["norm"] = 3.0
    elif pic_run == "more_dump_test":
        spect_params["power_index"] = -1.7
        spect_params["energy_range"] = [20, 200]
        spect_params["emax"] = 1E3
        spect_params["norm"] = 3.0
    elif pic_run == "mime1_sigmae100_vthe04_db025_1024de":
        spect_params["power_index"] = -3.5
        spect_params["energy_range"] = [100, 600]
        spect_params["emax"] = 2E4
        spect_params["norm"] = 3.0
    elif pic_run == "mime1_sigmae100_vthe04_db05_1024de":
        spect_params["power_index"] = -2.8
        spect_params["energy_range"] = [100, 800]
        spect_params["emax"] = 2E4
        spect_params["norm"] = 3.0
    elif pic_run == "mime1_sigmae100_vthe04_db1_1024de":
        spect_params["power_index"] = -2.6
        spect_params["energy_range"] = [250, 2500]
        spect_params["emax"] = 2E4
        spect_params["norm"] = 3.0
    elif pic_run == "mime1_sigmae100_vthe04_db4_1024de":
        spect_params["power_index"] = -2.0
        spect_params["energy_range"] = [500, 2500]
        spect_params["emax"] = 2E4
        spect_params["norm"] = 3.0
    elif pic_run == "mime1_sigmae100_vthe04_db16_1024de":
        spect_params["power_index"] = -1.0
        spect_params["energy_range"] = [1000, 3000]
        spect_params["emax"] = 5E4
        spect_params["norm"] = 2.0
    elif pic_run == "mime1_sigmae40_vthe04_db025_1024de":
        spect_params["power_index"] = -2.6
        spect_params["energy_range"] = [250, 2500]
        spect_params["emax"] = 2E4
        spect_params["norm"] = 3.0
    elif pic_run == "mime1_sigmae40_vthe04_db05_1024de":
        spect_params["power_index"] = -3.1
        spect_params["energy_range"] = [50, 500]
        spect_params["emax"] = 2E4
        spect_params["norm"] = 3.0
    elif pic_run == "mime1_sigmae40_vthe04_db1_1024de":
        spect_params["power_index"] = -2.7
        spect_params["energy_range"] = [100, 2000]
        spect_params["emax"] = 2E4
        spect_params["norm"] = 3.0
    elif pic_run == "mime1_sigmae40_vthe04_db4_1024de":
        spect_params["power_index"] = -2.0
        spect_params["energy_range"] = [250, 2000]
        spect_params["emax"] = 2E4
        spect_params["norm"] = 3.0
    elif pic_run == "mime1_sigmae40_vthe04_db16_1024de":
        spect_params["power_index"] = -1.2
        spect_params["energy_range"] = [400, 2000]
        spect_params["emax"] = 5E4
        spect_params["norm"] = 2.0
    elif pic_run == "mime1_sigmae10_vthe04_db025_1024de":
        spect_params["power_index"] = -2.6
        spect_params["energy_range"] = [250, 2500]
        spect_params["emax"] = 2E4
        spect_params["norm"] = 3.0
    elif pic_run == "mime1_sigmae10_vthe04_db05_1024de":
        spect_params["power_index"] = -3.2
        spect_params["energy_range"] = [40, 400]
        spect_params["emax"] = 2E4
        spect_params["norm"] = 3.0
    elif pic_run == "mime1_sigmae10_vthe04_db1_1024de":
        spect_params["power_index"] = -3.0
        spect_params["energy_range"] = [30, 500]
        spect_params["emax"] = 2E4
        spect_params["norm"] = 3.0
    elif pic_run == "mime1_sigmae10_vthe04_db4_1024de":
        spect_params["power_index"] = -2.4
        spect_params["energy_range"] = [100, 1000]
        spect_params["emax"] = 2E4
        spect_params["norm"] = 3.0
    elif pic_run == "mime1_sigmae10_vthe04_db16_1024de":
        spect_params["power_index"] = -1.7
        spect_params["energy_range"] = [200, 1000]
        spect_params["emax"] = 5E4
        spect_params["norm"] = 3.0
    elif pic_run == "turbulent-sheet3D-mixing-sigma100":
        spect_params["power_index"] = -1.8
        spect_params["energy_range"] = [20, 200]
        spect_params["emax"] = 5E4
        spect_params["norm"] = 3.0
    elif pic_run == "3D_mime1_sigmae100_vthe04_db1_1024de":
        spect_params["power_index"] = -2.6
        spect_params["energy_range"] = [250, 2500]
        spect_params["emax"] = 2E4
        spect_params["norm"] = 3.0
    elif pic_run == "mime1_sigmae100_vthe04_db1_256de":
        spect_params["power_index"] = -2.6
        spect_params["energy_range"] = [250, 2500]
        spect_params["emax"] = 2E4
        spect_params["norm"] = 3.0
    elif pic_run == "3D_mime1_sigmae100_vthe04_db1_256de":
        spect_params["power_index"] = -2.5
        spect_params["energy_range"] = [50, 500]
        spect_params["emax"] = 2E4
        spect_params["norm"] = 3.0
    elif pic_run == "3D-Lx150-bg0.2-150ppc-2048KNL-tracking":
        spect_params["power_index"] = -4.0
        spect_params["energy_range"] = [25, 250]
        spect_params["emax"] = 1E3
        spect_params["norm"] = 3.0
    elif pic_run == "mime1836_sigmaic001_bg00":
        spect_params["power_index"] = -3.1
        spect_params["energy_range"] = [40, 200]
        spect_params["emax"] = 1E3
        spect_params["norm"] = 3.0
    elif pic_run == "mime1836_sigmaic004_bg00":
        spect_params["power_index"] = -3.4
        spect_params["energy_range"] = [40, 300]
        spect_params["emax"] = 2E3
        spect_params["norm"] = 3.0
    elif pic_run == "mime1836_sigmaic016_bg00":
        spect_params["power_index"] = -3.2
        spect_params["energy_range"] = [40, 500]
        spect_params["emax"] = 2E3
        spect_params["norm"] = 3.0
    elif pic_run == "mime1836_sigmaic016_bg00_new":
        spect_params["power_index"] = -3.2
        spect_params["energy_range"] = [40, 500]
        spect_params["emax"] = 2E3
        spect_params["norm"] = 3.0
    elif pic_run == "mime1836_sigmaic064_bg00":
        spect_params["power_index"] = -2.7
        spect_params["energy_range"] = [40, 600]
        spect_params["emax"] = 5E3
        spect_params["norm"] = 3.0
    elif pic_run == "mime1836_sigmaic256_bg00":
        spect_params["power_index"] = -2.5
        spect_params["energy_range"] = [40, 600]
        spect_params["emax"] = 5E3
        spect_params["norm"] = 3.0
    elif pic_run == "mime1836_sigmaic001_bg05":
        spect_params["power_index"] = -3.4
        spect_params["energy_range"] = [30, 300]
        spect_params["emax"] = 1E3
        spect_params["norm"] = 3.0
    elif pic_run == "mime1836_sigmaic004_bg05":
        spect_params["power_index"] = -3.4
        spect_params["energy_range"] = [40, 300]
        spect_params["emax"] = 2E3
        spect_params["norm"] = 3.0
    elif pic_run == "mime1836_sigmaic016_bg05":
        spect_params["power_index"] = -3.2
        spect_params["energy_range"] = [40, 500]
        spect_params["emax"] = 2E3
        spect_params["norm"] = 3.0
    elif pic_run == "mime1836_sigmaic064_bg05":
        spect_params["power_index"] = -2.7
        spect_params["energy_range"] = [40, 600]
        spect_params["emax"] = 5E3
        spect_params["norm"] = 3.0
    elif pic_run == "mime1836_sigmaic256_bg05":
        spect_params["power_index"] = -2.5
        spect_params["energy_range"] = [40, 600]
        spect_params["emax"] = 5E3
        spect_params["norm"] = 3.0
    elif pic_run == "mime1836_sigmaic001_bg20":
        spect_params["power_index"] = -3.1
        spect_params["energy_range"] = [40, 200]
        spect_params["emax"] = 1E3
        spect_params["norm"] = 3.0
    else:
        spect_params["power_index"] = -1.2
        spect_params["energy_range"] = [200, 1000]
        spect_params["emax"] = 5E4
        spect_params["norm"] = 3.0

    return spect_params


def get_time_interval(pic_run="sigma01_bg005_4000de_triggered"):
    """Time interval for calculating rates for different runs
    """
    tinterval = 100
    if "4000de_triggered" in pic_run:
        tinterval = 100
    elif "sigmae" in pic_run:
        tinterval = 1000
    elif "3D-Lx150" in pic_run:
        tinterval = 20
    elif "mime1_" in pic_run:
        tinterval = 1000

    return tinterval


def plot_spectrum_multi(plot_config, show_plot=True):
    """Plot particle energy spectrum for multiple time frames

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_run_dir = pic_info.run_dir
    vpic_info = get_vpic_info(pic_run_dir)
    emin = vpic_info["emin_spect"]
    emax = vpic_info["emax_spect"]
    nbins = int(vpic_info["nbins"])
    dloge = (math.log10(emax) - math.log10(emin)) / (nbins - 1)
    emin0 = 10**(math.log10(emin) - dloge)
    ebins = np.logspace(math.log10(emin0), math.log10(emax), nbins+1)
    ebins_mid = (ebins[:-1] + ebins[1:]) * 0.5
    debins = np.diff(ebins)
    nptot = pic_info.nx * pic_info.ny * pic_info.nz * pic_info.nppc
    dtwpe_fields = math.ceil(pic_info.fields_interval * pic_info.dtwpe / 0.1) * 0.1
    lx_de = pic_info.lx_di * math.sqrt(pic_info.mime)

    ntf = len(os.listdir(pic_run_dir + "spectrum/"))
    # # Set it to 3 light crossing time
    # nframes = int(lx_de * 3 / dtwpe_fields)
    # if ntf < nframes:
    #     nframes = ntf
    nframes = ntf
    tframes = range(nframes)
    flogs = np.zeros((nframes, nbins))
    if species in ["e", "electron"]:
        sname = "electron"
    else:
        sname = "ion"
    temp = pic_info.Te if species in ['e', 'electron'] else pic_info.Ti
    print("Particle initial temperature (Lorentz factor): %f" % temp)

    fig = plt.figure(figsize=[7, 5])
    rect = [0.12, 0.16, 0.72, 0.8]
    ax = fig.add_axes(rect)
    for iframe, tframe in enumerate(tframes):
        print("Time frame: %d" % tframe)
        tindex = tframe * int(vpic_info["spectrum_interval"])
        fdir = pic_run_dir + '/spectrum_combined/'
        fname = fdir + 'spectrum_' + sname + '_' + str(tindex) + '.dat'
        ebins_mid_run = ebins_mid / temp
        debins_run = debins / temp
        if os.path.isfile(fname):
            flog = np.fromfile(fname, dtype=np.float32)
            espect = flog[3:] / debins_run / nptot  # the first 3 are magnetic field
            color = plt.cm.jet(tframe/float(nframes), 1)
            flogs[iframe, :] = espect
            ax.loglog(ebins_mid_run, espect, linewidth=1, color=color)

    spect_params = get_spect_params(pic_run)
    pindex = spect_params["power_index"]
    emin, emax = spect_params["energy_range"]
    fpower = ebins_mid_run**pindex
    es, _ = find_nearest(ebins_mid_run, emin)
    ee, _ = find_nearest(ebins_mid_run, emax)
    fnorm = (espect[es] / fpower[es]) * spect_params["norm"]
    fpower *= fnorm
    power_index = "{%0.1f}" % pindex
    pname = r'$\propto (\gamma - 1)^{' + power_index + '}$'
    ax.loglog(ebins_mid_run[es:ee], fpower[es:ee], linewidth=1,
              color='k', label=pname)
    ax.legend(loc=1, prop={'size': 12}, ncol=1,
             shadow=False, fancybox=False, frameon=False)
    temp_text = r"$T_" + species + "=" + str(temp) + r"$"
    ax.text(0.05, 0.05, temp_text, color='k', fontsize=16,
            bbox=dict(facecolor='none', alpha=1.0,
                      edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes)
    if species == "e":
        ax.set_xlim([1E0, spect_params["emax"]])
        ax.set_ylim([1E-9, 1E0])
    else:
        ax.set_xlim([1E-4, 4*spect_params["emax"]/pic_info.mime])
        ax.set_ylim([1E-7, 1E3])
    ax.tick_params(bottom=True, top=True, left=True, right=False)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlabel(r'$(\gamma - 1)/T_' + species + '$', fontsize=16)
    ax.set_ylabel(r'$f(\gamma - 1)$', fontsize=16)
    ax.tick_params(labelsize=12)

    rect_cbar = np.copy(rect)
    rect_cbar[0] += rect[2] + 0.02
    rect_cbar[2] = 0.02
    cax = fig.add_axes(rect_cbar)
    ts = tframes[0] * dtwpe_fields
    te = tframes[-1] * dtwpe_fields
    sm = plt.cm.ScalarMappable(cmap=plt.cm.jet,
                               norm=plt.Normalize(vmin=ts, vmax=te))
    cax.tick_params(axis='x', which='major', direction='in')
    # fake up the array of the scalar mappable. Urgh...
    sm._A = []
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(r'$t\omega_{pe}$', fontsize=16)
    cbar.ax.tick_params(labelsize=12)

    fdir = '../img/power_law_index/spectrum_all_frames/'
    mkdir_p(fdir)
    fname = fdir + 'spectrum_' + pic_run + species + '.pdf'
    fig.savefig(fname)

    # # tratio = pic_info.particle_interval // pic_info.fields_interval
    # tratio = 10
    # dflogs = np.gradient(flogs, axis=0)
    # dflogs[dflogs<=0] = np.nan
    # fig = plt.figure(figsize=[7, 5])
    # rect = [0.14, 0.16, 0.82, 0.8]
    # ax = fig.add_axes(rect)
    # p1, = ax.loglog(ebins_mid_run, flogs[tratio] - flogs[0],
    #                 color='k', nonposy='mask')
    # ax.loglog(ebins_mid_run[es:ee], fpower[es:ee]*5E-3, linewidth=1, color='r')
    # if species == 'e':
    #     ax.set_xlim([1E0, 2E4])
    #     ax.set_ylim([1E-9, 1E0])
    # else:
    #     ax.set_xlim([1E-1, 2E3])
    #     ax.set_ylim([1E-9, 1E0])
    # ax.tick_params(bottom=True, top=True, left=True, right=False)
    # ax.tick_params(axis='x', which='minor', direction='in')
    # ax.tick_params(axis='x', which='major', direction='in')
    # ax.tick_params(axis='y', which='minor', direction='in')
    # ax.tick_params(axis='y', which='major', direction='in')
    # ax.set_xlabel(r'$\gamma - 1$', fontsize=16)
    # ax.set_ylabel(r'$f(\gamma - 1)$', fontsize=16)
    # ax.tick_params(labelsize=12)
    # fdir = '../img/power_law_index/spectrum/' + pic_run + '/'
    # mkdir_p(fdir)
    # for iframe, tframe in enumerate(tframes[tratio::tratio]):
    #     print("Time frame: %d" % tframe)
    #     p1.set_ydata(flogs[tframe] - flogs[tframe-tratio])
    #     fig.canvas.draw()
    #     fname = fdir + 'diff_spectrum_' + species + '_' + str(tframe) + '.pdf'
    #     fig.savefig(fname)
    #     fig.canvas.flush_events()
    # plt.close()

    if show_plot:
        plt.show()
    else:
        plt.close()


def func_double_maxwell_juttner(gamma, theta1, theta2, f1, f2):
    """
    Function for fitting with double Maxwell-Juttner distribution

    Args:
        gamma: Lorentz factor the particle
        theta1: lower thermal temperature / m_ec^2
        theta2: higher thermal temperature / m_ec^2
        f1: normalization for the lower thermal core
        f2: normalization for the higher thermal core
    """
    beta = np.sqrt(1 - 1.0 / gamma**2)
    return (f1 * (gamma**2 * beta/(theta1*kv(2.0, 1/theta1))) * np.exp(-gamma/theta1) +
            f2 * (gamma**2 * beta/(theta2*kv(2.0, 1/theta2))) * np.exp(-gamma/theta2))


def fit_thermal_core(gamma, f):
    """
    """
    estart = 0
    ng = 3
    kernel = np.ones(ng) / float(ng)
    # fnew = gaussian_filter(f, 5)
    # fnew = np.convolve(f, kernel, 'same')
    es = np.argmax(f) - 150
    ee = np.argmax(f) + 150
    popt, pcov = curve_fit(func_double_maxwell_juttner, gamma[:ee], f[:ee])
    fth = func_double_maxwell_juttner(gamma, popt[0], popt[1], popt[2], popt[3])
    print('Energy with maximum flux: ', gamma[np.argmax(f)])
    print('Temperature for the lower thermal core: ', popt[0])
    print('Temperature for the higher thermal core: ', popt[1])
    plt.loglog(gamma-1, f)
    plt.loglog(gamma-1, fth)
    plt.loglog(gamma-1, f - fth)
    # plt.loglog(gamma[es:ee], f[es:ee])
    # plt.loglog(gamma, f - fth)
    plt.ylim([1E-10, 1E-1])
    plt.show()
    return (fth, popt)


def spectrum_sigma_bg(plot_config, show_plot=True):
    """Plot particle energy spectrum for runs with different sigma and Bg

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    if species in ["e", "electron"]:
        sname = "electron"
    else:
        sname = "ion"

    root_dir = "/net/scratch4/xiaocanli/reconnection/power_law_index/"
    # bgs = ["00", "05", "10", "20", "10"]
    # labels = [r"$b_g=0.0$", r"$b_g=0.5$", r"$b_g=1.0$", r"$b_g=2.0$"]
    # bgs = ["00", "05", "10", "20", "10"]
    # labels = [r"$b_g=0.0$", r"$b_g=0.5$", r"$b_g=1.0$", r"$b_g=2.0$"]
    # sigmas = ["001", "004", "016", "064", "256"]
    sigmas = ["256"]
    labels = [r"$\sigma_{ic}=0.1$", r"$\sigma_{ic}=0.4$", r"$\sigma_{ic}=1.6$",
              r"$\sigma_{ic}=6.4$", r"$\sigma_{ic}=25.6$"]
    pic_runs = ["mime1836_sigmaic" + sigma + "_bg05" for sigma in sigmas]

    fig = plt.figure(figsize=[7, 5])
    rect = [0.14, 0.16, 0.82, 0.8]
    ax = fig.add_axes(rect)
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    ax.set_prop_cycle('color', COLORS)
    for irun, pic_run in enumerate(pic_runs):
        print("PIC run name: %s" % pic_run)
        if "sigmaic256" in pic_run and "bg00" not in pic_run:
            emin, emax = 1E-3, 1E7
        else:
            emin, emax = 1E-4, 1E6
        nbins = 1000
        dloge = (math.log10(emax) - math.log10(emin)) / (nbins - 1)
        emin0 = 10**(math.log10(emin) - dloge)
        ebins = np.logspace(math.log10(emin0), math.log10(emax), nbins+1)
        ebins_mid = (ebins[:-1] + ebins[1:]) * 0.5
        debins = np.diff(ebins)

        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        pic_run_dir = pic_info.run_dir
        ntf = len(os.listdir(pic_run_dir + "spectrum/"))
        nptot = pic_info.nx * pic_info.ny * pic_info.nz * pic_info.nppc
        tframe = ntf - 1
        tindex = tframe * int(pic_info.fields_interval)
        fdir = pic_run_dir + 'spectrum_combined/'
        fname = fdir + 'spectrum_' + sname + '_' + str(tindex) + '.dat'
        if species in ['e', 'electron']:
            ebins_mid_run = ebins_mid / pic_info.Te
            debins_run = debins / pic_info.Te
        else:
            ebins_mid_run = ebins_mid / pic_info.Ti
            debins_run = debins / pic_info.Ti
        pmass = 1.0 if species in ["e", "electron"] else pic_info.mime
        if os.path.isfile(fname):
            flog = np.fromfile(fname, dtype=np.float32)
            espect = flog[3:] / debins_run / nptot  # the first 3 are magnetic field
        gamma = ebins_mid + 1.0
        fth_l, popt = fit_thermal_core(gamma, espect)
        # fnth = espect - fth_l
        # imax = np.argmax(fnth)
        # ns = imax - 50
        # fth_h, popt = fit_thermal_core(gamma[ns:], fnth[ns:])
        # ax.loglog(ebins_mid_run*pmass, espect, linewidth=2,
        #           label=labels[irun])
        # ax.loglog(ebins_mid_run*pmass, fth_l, linewidth=2)

    # ax.legend(loc=3, prop={'size': 16}, ncol=1,
        #      shadow=False, fancybox=False, frameon=False)

    # if species == 'e':
        # ax.set_xlim([1E-1, 1E4])
        # ax.set_ylim([1E-10, 1E-1])
    # else:
        # ax.set_xlim([1E-1, 1E4])
        # ax.set_ylim([1E-8, 1E0])
    # ax.tick_params(bottom=True, top=True, left=True, right=False)
    # ax.tick_params(axis='x', which='minor', direction='in')
    # ax.tick_params(axis='x', which='major', direction='in')
    # ax.tick_params(axis='y', which='minor', direction='in')
    # ax.tick_params(axis='y', which='major', direction='in')
    # ax.set_xlabel(r'$(\gamma - 1)m_' + species + 'c^2/T_' + species + '$', fontsize=16)
    # ax.set_ylabel(r'$(\gamma - 1)f(\gamma - 1)$', fontsize=16)
    # ax.tick_params(labelsize=12)
    # ename = 'electron' if species == 'e' else 'ion'
    # # fdir = '../img/power_law_index/spectrum_bg/'
    # # mkdir_p(fdir)
    # # fname = fdir + 'spectrum_bg_' + sigma_type + '_' + species + '.pdf'
    # # fig.savefig(fname)
    # plt.show()


def get_all_runs(root_dir):
    """
    """
    pic_runs = ["sigma01_bg005_4000de_triggered", "sigma04_bg005_4000de_triggered",
                "sigma16_bg005_4000de_triggered", "sigma64_bg005_4000de_triggered",
                "sigma256_bg005_4000de_triggered", "sigmae6_bg005_800de_triggered",
                "sigmae25_bg005_800de_triggered", "sigmae100_bg005_800de_triggered",
                "sigmae400_bg005_800de_triggered"]
    labels = [r"$b_g=0.05, m_i/m_e=1836, \sigma=0.1$",
              r"$b_g=0.05, m_i/m_e=1836, \sigma=0.4$",
              r"$b_g=0.05, m_i/m_e=1836, \sigma=1.6$",
              r"$b_g=0.05, m_i/m_e=1836, \sigma=6.4$",
              r"$b_g=0.05, m_i/m_e=1836, \sigma=25.6$",
              r"$m_i/m_e=1, b_g=0.05, \sigma_e=6.25$",
              r"$m_i/m_e=1, b_g=0.05, \sigma_e=25$",
              r"$m_i/m_e=1, b_g=0.05, \sigma_e=100$",
              r"$m_i/m_e=1, b_g=0.05, \sigma_e=400$"]
    return (pic_runs, labels)


def plot_spectrum_all_runs(plot_config, show_plot=True):
    """Plot final particle energy spectrum for all runs

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    if species in ["e", "electron"]:
        sname = "electron"
    else:
        sname = "Ion"

    emin, emax = 1E-4, 1E6
    nbins = 1000
    dloge = (math.log10(emax) - math.log10(emin)) / (nbins - 1)
    emin0 = 10**(math.log10(emin) - dloge)
    ebins = np.logspace(math.log10(emin0), math.log10(emax), nbins+1)
    ebins_mid = (ebins[:-1] + ebins[1:]) * 0.5
    debins = np.diff(ebins)

    root_dir = "/net/scratch4/xiaocanli/reconnection/power_law_index/"
    pic_runs, labels = get_all_runs(root_dir)

    fig = plt.figure(figsize=[7, 5])
    rect = [0.14, 0.16, 0.82, 0.8]
    ax = fig.add_axes(rect)
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    ax.set_prop_cycle('color', COLORS)
    runs, rune = 3, 5
    for irun, pic_run in enumerate(pic_runs[runs:rune]):
        print("PIC run name: %s" % pic_run)
        pic_run_dir = root_dir + pic_run
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        ntf = pic_info.ntf
        nptot = pic_info.nx * pic_info.ny * pic_info.nz * pic_info.nppc
        tframe = ntf - 1
        print("Time frame: %d" % tframe)
        tindex = tframe * pic_info.fields_interval
        fdir = pic_run_dir + '/spectrum_combined/'
        fname = fdir + 'spectrum_' + sname + '_' + str(tindex) + '.dat'
        if species in ['e', 'electron']:
            ebins_mid_run = ebins_mid / pic_info.Te
            debins_run = debins / pic_info.Te
        else:
            ebins_mid_run = ebins_mid / pic_info.Ti
            debins_run = debins / pic_info.Ti
        if os.path.isfile(fname):
            flog = np.fromfile(fname, dtype=np.float32)
            espect = flog[3:] / debins_run / nptot  # the first 3 are magnetic field
            ax.loglog(ebins_mid_run, espect, linewidth=2, label=labels[irun+runs])

    ax.legend(loc=3, prop={'size': 12}, ncol=1,
             shadow=False, fancybox=False, frameon=False)
    # pindex = -2.7
    # fpower = ebins_mid**pindex * 5E2
    # es, _ = find_nearest(ebins_mid, 200)
    # ee, _ = find_nearest(ebins_mid, 1000)
    # ax.loglog(ebins_mid[es:ee], fpower[es:ee], linewidth=1, color='k')
    # power_index = "{%0.1f}" % pindex
    # pname = r'$\propto \varepsilon^{' + power_index + '}$'

    if species == 'e':
        ax.set_xlim([1E-1, 1E4])
        ax.set_ylim([1E-9, 1E0])
    else:
        ax.set_xlim([1E-1, 2E3])
        ax.set_ylim([1E-9, 1E0])
    ax.tick_params(bottom=True, top=True, left=True, right=False)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlabel(r'$\gamma - 1$', fontsize=16)
    ax.set_ylabel(r'$f(\gamma - 1)$', fontsize=16)
    ax.tick_params(labelsize=12)
    ename = 'electron' if species == 'e' else 'ion'
    # fdir = '../img/power_law_index/spectrum/' + pic_run + '/'
    # mkdir_p(fdir)
    # fname = fdir + 'spectrum_' + species + '.pdf'
    # fig.savefig(fname)
    plt.show()


def plot_spectrum_bg(plot_config, show_plot=True):
    """Plot final particle energy spectrum for runs with the same guide field

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    if species in ["e", "electron"]:
        sname = "electron"
    else:
        sname = "Ion"

    emin, emax = 1E-4, 1E6
    nbins = 1000
    dloge = (math.log10(emax) - math.log10(emin)) / (nbins - 1)
    emin0 = 10**(math.log10(emin) - dloge)
    ebins = np.logspace(math.log10(emin0), math.log10(emax), nbins+1)
    ebins_mid = (ebins[:-1] + ebins[1:]) * 0.5
    debins = np.diff(ebins)

    root_dir = "/net/scratch4/xiaocanli/reconnection/power_law_index/"
    sigma_type = plot_config["sigma_type"]
    bgs = ["005", "01", "02", "04", "10"]
    pic_runs = [sigma_type + "_bg" + bg + "_4000de_triggered" for bg in bgs]
    labels = [r"$b_g=0.05$", r"$b_g=0.1$", r"$b_g=0.2$", r"$b_g=0.4$", r"$b_g=1.0$"]

    fig = plt.figure(figsize=[7, 5])
    rect = [0.14, 0.16, 0.82, 0.8]
    ax = fig.add_axes(rect)
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    ax.set_prop_cycle('color', COLORS)
    for irun, pic_run in enumerate(pic_runs):
        print("PIC run name: %s" % pic_run)
        pic_run_dir = root_dir + pic_run
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        ntf = pic_info.ntf
        nptot = pic_info.nx * pic_info.ny * pic_info.nz * pic_info.nppc
        tframe = ntf - 1
        print("Time frame: %d" % tframe)
        tindex = tframe * pic_info.fields_interval
        fdir = pic_run_dir + '/spectrum_combined/'
        fname = fdir + 'spectrum_' + sname + '_' + str(tindex) + '.dat'
        if species in ['e', 'electron']:
            ebins_mid_run = ebins_mid / pic_info.Te
            debins_run = debins / pic_info.Te
        else:
            ebins_mid_run = ebins_mid / pic_info.Ti
            debins_run = debins / pic_info.Ti
        if os.path.isfile(fname):
            flog = np.fromfile(fname, dtype=np.float32)
            espect = flog[3:] / debins_run / nptot  # the first 3 are magnetic field
            ax.loglog(ebins_mid_run, espect, linewidth=2, label=labels[irun])

    ax.legend(loc=3, prop={'size': 16}, ncol=1,
             shadow=False, fancybox=False, frameon=False)

    if species == 'e':
        ax.set_xlim([1E-1, 2E3])
        ax.set_ylim([1E-9, 1E0])
    else:
        ax.set_xlim([1E-1, 2E3])
        ax.set_ylim([1E-9, 1E0])
    ax.tick_params(bottom=True, top=True, left=True, right=False)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlabel(r'$(\gamma - 1)/T_' + species + '$', fontsize=16)
    ax.set_ylabel(r'$f(\gamma - 1)$', fontsize=16)
    ax.tick_params(labelsize=12)
    ename = 'electron' if species == 'e' else 'ion'
    fdir = '../img/power_law_index/spectrum_bg/'
    mkdir_p(fdir)
    fname = fdir + 'spectrum_bg_' + sigma_type + '_' + species + '.pdf'
    fig.savefig(fname)
    plt.show()


def number_rate_high_energy(plot_config, show_plot=True):
    """Plot the change rate of the number of high-energy particles

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
    root_dir = "/net/scratch4/xiaocanli/reconnection/power_law_index/"
    pic_run_dir = root_dir + pic_run + '/'
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    nptot = pic_info.nx * pic_info.ny * pic_info.nz * pic_info.nppc
    ntf = pic_info.ntf
    emin, emax = 1E-4, 1E6
    nbins = 1000
    dloge = (math.log10(emax) - math.log10(emin)) / (nbins - 1)
    emin0 = 10**(math.log10(emin) - dloge)
    ebins = np.logspace(math.log10(emin0), math.log10(emax), nbins+1)
    ebins_mid = (ebins[:-1] + ebins[1:]) * 0.5
    debins = np.diff(ebins)
    temp = pic_info.Te if species in ['e', 'electron'] else pic_info.Ti
    ebins_mid_run = ebins_mid / temp
    spect_params = get_spect_params(pic_run)
    emin, emax = spect_params["energy_range"]
    es, _ = find_nearest(ebins_mid_run, emin)
    ee, _ = find_nearest(ebins_mid_run, emax)

    tframes = range(ntf)
    nframes = len(tframes)
    nhigh = np.zeros((nframes))
    if species in ["e", "electron"]:
        sname = "electron"
    else:
        sname = "Ion"

    for iframe, tframe in enumerate(tframes):
        print("Time frame: %d" % tframe)
        tindex = tframe * pic_info.fields_interval
        fdir = pic_run_dir + '/spectrum_combined/'
        fname = fdir + 'spectrum_' + sname + '_' + str(tindex) + '.dat'
        debins_run = debins / temp
        if os.path.isfile(fname):
            flog = np.fromfile(fname, dtype=np.float32)[3:]
            nhigh[iframe] = np.sum(flog[es:]) / nptot

    vpic_info = get_vpic_info(pic_run_dir)
    dt_spectrum = vpic_info["spectrum_interval"] * vpic_info["dt*wpe"]
    twpe = np.asarray(tframes) * dt_spectrum
    dnhigh = np.gradient(nhigh) / dt_spectrum
    fig = plt.figure(figsize=[7, 5])
    rect = [0.2, 0.14, 0.72, 0.8]
    ax = fig.add_axes(rect)
    ax.plot(twpe, dnhigh, linewidth=2)
    ax.plot([0, twpe[-1]], [0, 0], linewidth=1, linestyle='--', color='k')
    ax.set_xlim([0, twpe[-1]])
    ax.tick_params(bottom=True, top=True, left=True, right=False)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlabel(r'$t\omega_{pe}$', fontsize=16)
    ax.set_ylabel(r'$dN_\text{high}/dt$', fontsize=16)
    ax.tick_params(labelsize=12)
    fdir = '../img/power_law_index/nhigh_rate/'
    mkdir_p(fdir)
    fname = fdir + 'nhigh_rate_' + pic_run + '.pdf'
    fig.savefig(fname)

    if show_plot:
        plt.show()
    else:
        plt.close()


def energy_conversion_all_runs(plot_config, show_plot=True):
    """Plot energy conversion for all runs

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    if species in ["e", "electron"]:
        sname = "electron"
    else:
        sname = "Ion"

    emin, emax = 1E-4, 1E6
    nbins = 1000
    dloge = (math.log10(emax) - math.log10(emin)) / (nbins - 1)
    emin0 = 10**(math.log10(emin) - dloge)
    ebins = np.logspace(math.log10(emin0), math.log10(emax), nbins+1)
    ebins_mid = (ebins[:-1] + ebins[1:]) * 0.5
    debins = np.diff(ebins)

    root_dir = "/net/scratch4/xiaocanli/reconnection/power_law_index/"
    pic_runs, labels = get_all_runs(root_dir)

    runs, rune = 0, 9
    for irun, pic_run in enumerate(pic_runs[runs:rune]):
    # for irun, pic_run in enumerate(pic_runs[0:1]):
        print("PIC run name: %s" % pic_run)
        pic_run_dir = root_dir + pic_run
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        tenergy = pic_info.tenergy
        enorm = pic_info.ene_magnetic[0]
        dt_ene = pic_info.dtwpe * pic_info.energy_interval
        tenergy *= pic_info.dtwpe / pic_info.dtwci

        ene_bx = pic_info.ene_bx
        ene_by = pic_info.ene_by
        ene_bz = pic_info.ene_bz
        ene_magnetic = pic_info.ene_magnetic
        ene_electric = pic_info.ene_electric
        kene_e = pic_info.kene_e
        kene_i = pic_info.kene_i

        etot = ene_magnetic + ene_electric + kene_e + kene_i
        print("Energy conservation: %e" % ((etot[-1] - etot[0]) / etot[0]))
        print("Energy conversion: %e" %
              ((ene_magnetic[-1] - ene_magnetic[0]) / ene_magnetic[0]))

        ene_bx /= enorm
        ene_by /= enorm
        ene_bz /= enorm
        ene_magnetic /= enorm
        ene_electric /= enorm
        kene_e /= enorm
        kene_i /= enorm

        dene_magnetic = np.gradient(ene_magnetic) / dt_ene
        dene_electric = np.gradient(ene_electric) / dt_ene
        dkene_e = np.gradient(kene_e) / dt_ene
        dkene_i = np.gradient(kene_i) / dt_ene

        fig = plt.figure(figsize=[7, 5])
        rect = [0.15, 0.15, 0.8, 0.8]
        ax = fig.add_axes(rect)
        ax.set_prop_cycle('color', COLORS)
        ax.plot(tenergy, -dene_magnetic, label="-magnetic")
        ax.plot(tenergy, dene_electric, label="electric")
        ax.plot(tenergy, dkene_e, label="electron")
        ax.plot(tenergy, dkene_i, label="ion")

        ax.legend(loc=2, prop={'size': 16}, ncol=1,
                 shadow=False, fancybox=False, frameon=False)
        ax.set_xlim([0, tenergy[-1]])
        ax.tick_params(bottom=True, top=True, left=True, right=False)
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')
        ax.set_xlabel(r'$t\omega_{pe}$', fontsize=16)
        ax.set_ylabel('Energy conversion rate', fontsize=16)
        ax.tick_params(labelsize=12)
        fdir = '../img/power_law_index/econv/'
        mkdir_p(fdir)
        fname = fdir + 'econv_' + pic_run + '.pdf'
        fig.savefig(fname)
        plt.close()
        # plt.show()


def check_density(plot_config, show_plot=True):
    """Whether to check maximum density

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    tframe = plot_config["tframe"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    kwargs = {"current_time": tframe,
              "xl": 0, "xr": pic_info.lx_di,
              "zb": -pic_info.lz_di*0.5, "zt": pic_info.lz_di*0.5}
    fname = pic_run_dir + "data/ne.gda"
    tframes = range(0, pic_info.ntf, 10)
    nframe = len(tframes)
    nmaxs = np.zeros(nframe)
    for iframe, tframe in enumerate(tframes):
        kwargs["current_time"] = tframe
        x, z, ne = read_2d_fields(pic_info, fname, **kwargs)
        nz, nx = ne.shape
        ne_z1 = np.mean(ne[:nz//2, :], axis=0)
        ne_z2 = np.mean(ne[nz//2:, :], axis=0)
        nmaxs[iframe] = max([np.max(ne_z1), np.max(ne_z2)])
    fig = plt.figure(figsize=[7, 5])
    rect = [0.12, 0.16, 0.70, 0.8]
    ax = fig.add_axes(rect)
    ax.plot(nmaxs, linewidth=2)
    plt.show()


def fluid_energization(plot_config, show_plot=True):
    """Plot fluid energization

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    if species == 'e':
        ylim = [-700, 700]
    else:
        ylim = [-700, 700]
    fig1 = plt.figure(figsize=[9, 3.0])
    box1 = [0.1, 0.18, 0.85, 0.68]
    axs1 = []
    fig2 = plt.figure(figsize=[9, 3.0])
    axs2 = []
    fig3 = plt.figure(figsize=[9, 3.0])
    axs3 = []
    fig4 = plt.figure(figsize=[9, 3.0])
    axs4 = []
    fig5 = plt.figure(figsize=[9, 3.0])
    axs5 = []
    fig6 = plt.figure(figsize=[9, 3.0])
    axs6 = []

    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    if "3D" in pic_run:
        enorm = pic_info.ny
    else:
        enorm = 1.0
    tfields = pic_info.tfields
    tenergy = pic_info.tenergy
    fname = "../data/fluid_energization/" + pic_run + "/"
    fname += "emf_ptensor_" + species + '.gda'
    fluid_ene = np.fromfile(fname, dtype=np.float32)
    nvar = int(fluid_ene[0])
    nframes = int(fluid_ene[1])
    fluid_ene[2:] /= enorm
    curv_drift_dote = fluid_ene[2:nframes+2]
    bulk_curv_dote = fluid_ene[nframes+2:2*nframes+2]
    grad_drift_dote = fluid_ene[2*nframes+2:3*nframes+2]
    magnetization_dote = fluid_ene[3*nframes+2:4*nframes+2]
    comp_ene = fluid_ene[4*nframes+2:5*nframes+2]
    shear_ene = fluid_ene[5*nframes+2:6*nframes+2]
    ptensor_ene = fluid_ene[6*nframes+2:7*nframes+2]
    pgyro_ene = fluid_ene[7*nframes+2:8*nframes+2]

    fname = "../data/fluid_energization/" + pic_run + "/"
    fname += "para_perp_acc_" + species + '.gda'
    fluid_ene = np.fromfile(fname, dtype=np.float32)
    nvar = int(fluid_ene[0])
    nframes = int(fluid_ene[1])
    fluid_ene[2:] /= enorm
    acc_drift_dote_t_perp = fluid_ene[2:nframes+2]
    acc_drift_dote_t_para = fluid_ene[nframes+2:2*nframes+2]
    acc_drift_dote_s_perp = fluid_ene[2*nframes+2:3*nframes+2]
    acc_drift_dote_s_para = fluid_ene[3*nframes+2:4*nframes+2]
    acc_drift_dote_t = acc_drift_dote_t_para + acc_drift_dote_t_perp
    acc_drift_dote_s = acc_drift_dote_s_para + acc_drift_dote_s_perp
    acc_drift_dote = acc_drift_dote_t + acc_drift_dote_s
    epara_ene = fluid_ene[4*nframes+2:5*nframes+2]
    eperp_ene = fluid_ene[5*nframes+2:6*nframes+2]
    acc_drift_dote[-1] = acc_drift_dote[-2]

    jperp_dote = curv_drift_dote + grad_drift_dote + magnetization_dote
    jagy_dote = ptensor_ene - pgyro_ene
    etot_ene = epara_ene + eperp_ene
    etot = np.sum(etot_ene)

    print("Eparallel contributes %0.2f" % (np.sum(epara_ene)/etot))
    print("Eperp contributes %0.2f" % (np.sum(eperp_ene)/etot))
    print("Gyrotropic pressure contributes %0.2f" % (np.sum(pgyro_ene)/etot))
    print("Non-gyrotropic pressure contributes %0.2f" % (np.sum(jagy_dote)/etot))
    print("Curvature drift contributes %0.2f" % (np.sum(curv_drift_dote)/etot))
    print("Gard-B drift contributes %0.2f" % (np.sum(grad_drift_dote)/etot))
    print("Compression contributes %0.2f" % (np.sum(comp_ene)/etot))
    print("Shear contributes %0.2f" % (np.sum(shear_ene)/etot))

    if species == 'e':
        dkene = pic_info.dkene_e
    else:
        dkene = pic_info.dkene_i
    dkene /= enorm

    if nframes < pic_info.ntf:
        tfields_adjust = tfields[:(nframes-pic_info.ntf)]
    else:
        tfields_adjust = tfields

    ax = fig1.add_axes(box1)
    axs1.append(ax)
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    ax.set_prop_cycle('color', COLORS)
    label2 = (r'$\boldsymbol{j}_{' + species + '\perp}' +
              r'\cdot\boldsymbol{E}_\perp$')
    ax.plot(tfields_adjust, eperp_ene, linewidth=1, label=label2)
    label1 = (r'$\boldsymbol{j}_{' + species + '\parallel}' +
              r'\cdot\boldsymbol{E}_\parallel$')
    ax.plot(tfields_adjust, epara_ene, linewidth=1, label=label1)
    label6 = r'$dK_' + species + '/dt$'
    ax.plot(tenergy, dkene, linewidth=1, label=label6)
    label3 = r'$(\nabla\cdot\tensorsym{P}_' + species + r')\cdot\boldsymbol{v}_E$'
    ax.plot(tfields_adjust, ptensor_ene, linewidth=1, label=label3)
    ax.plot([0, tenergy.max()], [0, 0], color='k', linestyle='--')
    ax.set_xlim([0, np.max(tfields_adjust)])
    # ax.set_ylim(ylim)
    ax.tick_params(labelsize=10)
    ax.set_ylabel('Energization', fontsize=12)
    ax.set_xlabel(r'$t\omega_{pe}$', fontsize=12)

    ax = fig2.add_axes(box1)
    axs2.append(ax)
    # COLORS = palettable.colorbrewer.qualitative.Set1_9.mpl_colors
    ax.set_prop_cycle('color', COLORS)
    label2 = (r'$\boldsymbol{j}_{' + species + '\perp}' +
              r'\cdot\boldsymbol{E}_\perp$')
    ax.plot(tfields_adjust, eperp_ene, linewidth=1, label=label2)
    ax.plot(tfields_adjust, curv_drift_dote, linewidth=1, label='Curvature')
    ax.plot(tfields_adjust, grad_drift_dote, linewidth=1, label='Gradient')
    ax.plot(tfields_adjust, magnetization_dote, linewidth=1, label='Magnetization')
    ax.plot([0, tenergy.max()], [0, 0], color='k', linestyle='--')
    ax.set_xlim([0, np.max(tfields_adjust)])
    # ax.set_ylim(ylim)
    ax.tick_params(labelsize=10)
    ax.set_ylabel('Energization', fontsize=12)
    ax.set_xlabel(r'$t\omega_{pe}$', fontsize=12)

    ax = fig3.add_axes(box1)
    axs3.append(ax)
    # COLORS = palettable.colorbrewer.qualitative.Set1_9.mpl_colors
    ax.set_prop_cycle('color', COLORS)
    label2 = (r'$\boldsymbol{j}_{' + species + '\perp}' +
              r'\cdot\boldsymbol{E}_\perp' + '$')
    ax.plot(tfields_adjust, eperp_ene, linewidth=1, label=label2)
    ax.plot(tfields_adjust, comp_ene, linewidth=1, label='Compression')
    ax.plot(tfields_adjust, shear_ene, linewidth=1, label='Shear')
    # label2 = (r'$\boldsymbol{j}_{' + species + '\perp}' +
    #           r'\cdot\boldsymbol{E}_\perp -' + 'n_' + species +
    #           'm_' + species + r'(d\boldsymbol{u}_' + species +
    #           r'/dt)\cdot\boldsymbol{v}_E$')
    # ax.plot(tfields_adjust, eperp_ene - acc_drift_dote, linewidth=1, label=label2)
    label4 = (r'$\boldsymbol{j}_{' + species + r'-\text{agy}}' +
              r'\cdot\boldsymbol{E}_\perp$')
    ax.plot(tfields_adjust, jagy_dote, linewidth=1, label=label4)
    # jdote_sum = comp_ene + shear_ene + jagy_dote
    # ax.plot(tfields_adjust, jdote_sum, linewidth=1)
    ax.plot([0, tenergy.max()], [0, 0], color='k', linestyle='--')
    ax.set_xlim([0, np.max(tfields_adjust)])
    # ax.set_ylim(ylim)
    ax.tick_params(labelsize=10)
    ax.set_ylabel('Energization', fontsize=12)
    ax.set_xlabel(r'$t\omega_{pe}$', fontsize=12)

    ax = fig4.add_axes(box1)
    axs4.append(ax)
    ax.set_prop_cycle('color', COLORS)
    label2 = (r'$\boldsymbol{j}_{' + species + '\perp}' +
              r'\cdot\boldsymbol{E}_\perp$')
    ax.plot(tfields_adjust, eperp_ene, linewidth=1, label=label2)
    jdote_sum = (curv_drift_dote + grad_drift_dote + magnetization_dote +
                 jagy_dote + acc_drift_dote)
    ax.plot(tfields_adjust, jdote_sum, linewidth=1,
            label='Drifts+Magnetization+Agyrotropic')
    ax.plot(tfields_adjust, acc_drift_dote, linewidth=1, label='Flow inertial')
    ax.plot([0, tenergy.max()], [0, 0], color='k', linestyle='--')
    ax.set_xlim([0, np.max(tfields_adjust)])
    # ax.set_ylim(ylim)
    ax.tick_params(labelsize=10)
    ax.set_ylabel('Energization', fontsize=12)
    ax.set_xlabel(r'$t\omega_{pe}$', fontsize=12)

    ax = fig5.add_axes(box1)
    axs5.append(ax)
    ax.set_prop_cycle('color', COLORS)
    ax.plot(tfields_adjust, acc_drift_dote_t_para, linewidth=1,
            label=r'Inertial (time $\parallel$)')
    ax.plot(tfields_adjust, acc_drift_dote_t_perp, linewidth=1,
            label=r'Inertial (time $\perp$)')
    ax.plot(tfields_adjust, acc_drift_dote_s_para, linewidth=1,
            label=r'Inertial (spatial $\parallel$)')
    ax.plot(tfields_adjust, acc_drift_dote_s_perp, linewidth=1,
            label=r'Inertial (spatial $\perp$)')
    ax.plot([0, tenergy.max()], [0, 0], color='k', linestyle='--')
    ax.set_xlim([0, np.max(tfields_adjust)])
    ax.tick_params(labelsize=10)
    ax.set_ylabel('Energization', fontsize=12)
    ax.set_xlabel(r'$t\omega_{pe}$', fontsize=12)

    ax = fig6.add_axes(box1)
    axs6.append(ax)
    ax.set_prop_cycle('color', COLORS)
    ax.plot(tfields_adjust, div0(acc_drift_dote_t, curv_drift_dote),
            linewidth=1, label='Inertial terms / Curvature Drift')
    ax.plot([0, tenergy.max()], [0, 0], color='k', linestyle='--')
    ax.set_xlim([0, np.max(tfields_adjust)])
    ax.set_ylim([-10, 10])
    ax.tick_params(labelsize=10)
    ax.set_ylabel('Energization', fontsize=12)
    ax.set_xlabel(r'$t\omega_{pe}$', fontsize=12)

    axs1[0].legend(loc='upper center', prop={'size': 12}, ncol=4,
                   bbox_to_anchor=(0.5, 1.22),
                   shadow=False, fancybox=False, frameon=False)
    axs2[0].legend(loc='upper center', prop={'size': 12}, ncol=4,
                   bbox_to_anchor=(0.5, 1.22),
                   shadow=False, fancybox=False, frameon=False)
    axs3[0].legend(loc='upper center', prop={'size': 12}, ncol=4,
                   bbox_to_anchor=(0.5, 1.22),
                   shadow=False, fancybox=False, frameon=False)
    axs4[0].legend(loc='upper center', prop={'size': 12}, ncol=4,
                   bbox_to_anchor=(0.5, 1.22),
                   shadow=False, fancybox=False, frameon=False)
    axs5[0].legend(loc='upper center', prop={'size': 12}, ncol=4,
                   bbox_to_anchor=(0.5, 1.22),
                   shadow=False, fancybox=False, frameon=False)
    axs6[0].legend(loc='upper center', prop={'size': 12}, ncol=4,
                   bbox_to_anchor=(0.5, 1.22),
                   shadow=False, fancybox=False, frameon=False)
    fdir = '../img/power_law_index/fluid_energization/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + 'fluid_ene_' + species + '.pdf'
    fig1.savefig(fname)
    fname = fdir + 'fluid_drift_' + species + '.pdf'
    fig2.savefig(fname)
    fname = fdir + 'fluid_comp_shear_' + species + '.pdf'
    fig3.savefig(fname)
    fname = fdir + 'polar_total_perp' + species + '.pdf'
    fig4.savefig(fname)
    fname = fdir + 'inertial_terms' + species + '.pdf'
    fig5.savefig(fname)
    fname = fdir + 'ratio_inertial_curvature' + species + '.pdf'
    fig6.savefig(fname)

    if show_plot:
        plt.show()
    else:
        plt.close('all')


def fluid_ene_fraction(plot_config, show_plot=True):
    """Calculate fluid energization fraction

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    if species == 'e':
        ylim = [-700, 700]
    else:
        ylim = [-700, 700]

    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    if "3D" in pic_run:
        enorm = pic_info.ny
    else:
        enorm = 1.0
    fname = "../data/fluid_energization/" + pic_run + "/"
    fname += "emf_ptensor_" + species + '.gda'
    fluid_ene = np.fromfile(fname, dtype=np.float32)
    nvar = int(fluid_ene[0])
    nframes = int(fluid_ene[1])
    fluid_ene[2:] /= enorm
    curv_drift_dote = fluid_ene[2:nframes+2]
    bulk_curv_dote = fluid_ene[nframes+2:2*nframes+2]
    grad_drift_dote = fluid_ene[2*nframes+2:3*nframes+2]
    magnetization_dote = fluid_ene[3*nframes+2:4*nframes+2]
    comp_ene = fluid_ene[4*nframes+2:5*nframes+2]
    shear_ene = fluid_ene[5*nframes+2:6*nframes+2]
    ptensor_ene = fluid_ene[6*nframes+2:7*nframes+2]
    pgyro_ene = fluid_ene[7*nframes+2:8*nframes+2]

    fname = "../data/fluid_energization/" + pic_run + "/"
    fname += "para_perp_acc_" + species + '.gda'
    fluid_ene = np.fromfile(fname, dtype=np.float32)
    nvar = int(fluid_ene[0])
    nframes = int(fluid_ene[1])
    fluid_ene[2:] /= enorm
    acc_drift_dote_t_perp = fluid_ene[2:nframes+2]
    acc_drift_dote_t_para = fluid_ene[nframes+2:2*nframes+2]
    acc_drift_dote_s_perp = fluid_ene[2*nframes+2:3*nframes+2]
    acc_drift_dote_s_para = fluid_ene[3*nframes+2:4*nframes+2]
    acc_drift_dote_t = acc_drift_dote_t_para + acc_drift_dote_t_perp
    acc_drift_dote_s = acc_drift_dote_s_para + acc_drift_dote_s_perp
    acc_drift_dote = acc_drift_dote_t + acc_drift_dote_s
    epara_ene = fluid_ene[4*nframes+2:5*nframes+2]
    eperp_ene = fluid_ene[5*nframes+2:6*nframes+2]
    acc_drift_dote[-1] = acc_drift_dote[-2]

    dt_fields = pic_info.fields_interval * pic_info.dtwpe
    curv_cumsum = np.cumsum(curv_drift_dote) * dt_fields
    tot_cumsum = np.cumsum(epara_ene + eperp_ene) * dt_fields

    jperp_dote = curv_drift_dote + grad_drift_dote + magnetization_dote
    jagy_dote = ptensor_ene - jperp_dote
    if species == 'e':
        kene = pic_info.kene_e
    else:
        kene = pic_info.kene_i
    dkene = kene - kene[0]

    tfields = pic_info.tfields * pic_info.dtwpe / pic_info.dtwci
    tenergy = pic_info.tenergy * pic_info.dtwpe / pic_info.dtwci
    if nframes < pic_info.ntf:
        tfields_adjust = tfields[:(nframes-pic_info.ntf)]
    else:
        tfields_adjust = tfields

    fig = plt.figure(figsize=[7, 5])
    rect = [0.12, 0.12, 0.82, 0.8]
    ax = fig.add_axes(rect)
    ax.plot(tfields_adjust, div0(curv_cumsum, tot_cumsum))
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.tick_params(labelsize=12)
    ax.set_ylim([0, 2])
    ax.grid(True)
    ax.set_xlabel(r'$t\omega_{pe}$', fontsize=16)
    ax.set_ylabel('Energization due to curvature drift', fontsize=16)
    fdir = '../img/power_law_index/curv_ene_frac/'
    mkdir_p(fdir)
    fname = fdir + 'frac_' + pic_run + '_' + species + '.pdf'
    fig.savefig(fname)
    plt.show()


def get_plot_setup(plot_type, species):
    """Get plotting setup for different type
    """
    if plot_type == 'total':
        if species == 'e':
            ylims = [[-0.005, 0.01], [-0.001, 0.005],
                     [-0.0005, 0.0025], [-0.0005, 0.0025]]
        else:
            ylims = [[-0.001, 0.003], [-0.0005, 0.002],
                     [-0.0002, 0.0005], [-0.0002, 0.0004]]
        data_indices = [1, 2]
    if plot_type == 'perpendicular':
        if species == 'e':
            ylims = [[-0.01, 0.02], [-0.005, 0.01],
                     [-0.002, 0.005], [-0.002, 0.005]]
        else:
            ylims = [[-0.001, 0.003], [-0.0005, 0.002],
                     [-0.0002, 0.0005], [-0.0002, 0.0004]]
        data_indices = [2]
    if plot_type == 'parallel':
        if species == 'e':
            ylims = [[-0.01, 0.02], [-0.005, 0.01],
                     [-0.002, 0.005], [-0.002, 0.005]]
        else:
            ylims = [[-0.001, 0.003], [-0.0005, 0.002],
                     [-0.0002, 0.0005], [-0.0002, 0.0004]]
        data_indices = [1]
    elif plot_type == 'curvature':
        if species == 'e':
            ylims = [[-0.005, 0.01], [-0.001, 0.005],
                     [-0.0005, 0.0025], [-0.0005, 0.0025]]
        else:
            ylims = [[-0.001, 0.004], [-0.0005, 0.0015],
                     [-0.0002, 0.0003], [-0.0002, 0.0004]]
        data_indices = [5]
    elif plot_type == 'gradient':
        if species == 'e':
            ylims = [[-0.01, 0.04], [-0.005, 0.02],
                     [-0.002, 0.01], [-0.002, 0.01]]
        else:
            ylims = [[-0.0002, 0.00025], [-0.0002, 0.00025],
                     [-0.0001, 0.0001], [-0.0001, 0.0001]]
        data_indices = [6]
    elif plot_type == 'inertial':
        if species == 'e':
            ylims = [[-0.00075, 0.00025], [-0.0002, 0.0001],
                     [-0.0001, 0.0001], [-0.0002, 0.0001]]
        else:
            ylims = [[-0.0012, 0.0004], [-0.0008, 0.0001],
                     [-0.0001, 0.0001], [-0.0002, 0.0001]]
        data_indices = [11, 12]
    elif plot_type == 'polarization':
        if species == 'e':
            ylims = [[-0.00025, 0.00025], [-0.00025, 0.00025],
                     [-0.0001, 0.0001], [-0.0002, 0.0001]]
        else:
            ylims = [[-0.001, 0.002], [-0.0005, 0.0015],
                     [-0.0002, 0.0005], [-0.0002, 0.00025]]
        data_indices = [15, 16]
    elif plot_type == 'parallel_drift':
        if species == 'e':
            ylims = [[-0.00025, 0.0004], [-0.00005, 0.0001],
                     [-0.00005, 0.0001], [-0.0002, 0.0001]]
        else:
            ylims = [[-0.00002, 0.0002], [-0.00005, 0.0001],
                     [-0.00005, 0.0001], [-0.00005, 0.00005]]
        data_indices = [7]
    elif plot_type == 'mu':
        if species == 'e':
            ylims = [[-0.0003, 0.0002], [-0.0002, 0.0005],
                     [-0.0001, 0.0003], [-0.0001, 0.0003]]
        else:
            ylims = [[-0.0003, 0.0002], [-0.0002, 0.0005],
                     [-0.0001, 0.0003], [-0.0001, 0.0003]]
        data_indices = [8]
    elif plot_type == 'compression':
        if species == 'e':
            ylims = [[-0.001, 0.003], [-0.0005, 0.001],
                     [-0.0002, 0.0004], [-0.0002, 0.0004]]
        else:
            ylims = [[-0.001, 0.003], [-0.0005, 0.002],
                     [-0.0002, 0.0005], [-0.0002, 0.0004]]
        data_indices = [3]
    elif plot_type == 'shear':
        if species == 'e':
            ylims = [[-0.0005, 0.0015], [-0.00025, 0.0005],
                     [-0.0001, 0.0002], [-0.0001, 0.0002]]
        else:
            ylims = [[-0.0005, 0.0015], [-0.00025, 0.0005],
                     [-0.0001, 0.0002], [-0.0001, 0.0002]]
        data_indices = [4]

    return (ylims, data_indices)


def particle_energization_multi(plot_config):
    """Plot particle-based energization for multiple types

    Args:
        plot_config: plotting configuration
    """
    plot_config1 = plot_config.copy()
    plot_types = ["total", "perpendicular", "parallel",
                  "curvature", "gradient", "inertial",
                  "polarization", "parallel_drift", "mu",
                  "compression", "shear"]
    for ptype in plot_types:
        plot_config1["plot_type"] = ptype
        particle_energization(plot_config1, show_plot=False)


def particle_energization(plot_config, show_plot=True):
    """Particle-based energization

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    tframe = plot_config["tframe"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]

    ylims, data_indices = get_plot_setup(plot_config["plot_type"], species)

    fpath = "../data/particle_interp/" + pic_run + "/"
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    dtp = math.ceil(pic_info.dt_particles / 0.1) * 0.1
    if species in ["e", "electron"]:
        sname = "electron"
    else:
        sname = "Ion"
    temp = pic_info.Te if species in ['e', 'electron'] else pic_info.Ti
    print("Particle initial temperature (Lorentz factor): %d" % temp)

    tstarts = [10, 21, 31, 41]
    tends = [20, 30, 40, int(pic_info.ntp)-1]
    nplots = len(tstarts)

    fnorm = 1E-2
    for iplot in range(nplots):
        tstart = tstarts[iplot]
        tend = tends[iplot]
        ylim = np.asarray(ylims[iplot]) / fnorm
        fig1 = plt.figure(figsize=[9.6, 4.0])
        box1 = [0.1, 0.2, 0.8, 0.75]
        axs1 = []

        nframes = tend - tstart

        ax = fig1.add_axes(box1)
        for tframe in range(tstart, tend + 1):
            tstep = tframe * pic_info.particle_interval
            tframe_fluid = tstep // pic_info.fields_interval
            fname = fpath + "particle_energization_" + species + "_" + str(tstep) + ".gda"
            fdata = np.fromfile(fname, dtype=np.float32)
            nbins = int(fdata[0])
            nbinx = int(fdata[1])
            nvar = int(fdata[2])
            ebins = fdata[3:nbins+3] / temp
            fbins = np.sum(fdata[nbins+3:].reshape((nvar, nbinx, nbins)), axis=1)

            if species == 'i':
                ebins *= pic_info.mime  # ebins are actually gamma
            fbins[1:, :] = div0(fbins[1:, :], fbins[0, :])

            color = plt.cm.jet((tframe - tstart)/float(nframes), 1)
            fdata = np.zeros(nbins)
            for idata in data_indices:
                fdata += fbins[idata, :]
            ax.semilogx(ebins, fdata/fnorm, linewidth=1, color=color)
        if species == 'e':
            ax.set_xlim([1E0, 2000])
        else:
            ax.set_xlim([1E0, 2000])
        ax.set_ylim(ylim)
        box1[0] += box1[2] + 0.02
        ax.set_ylabel(r'$\left<\nu^\text{I}_\text{COM}\right>/10^{-3}$', fontsize=10)

        ax.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$', fontsize=10)
        ax.tick_params(labelsize=8)

        ax.plot(ax.get_xlim(), [0, 0], linestyle='--', color='k')
        ax.tick_params(bottom=True, top=False, left=True, right=False)
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')

        box1[0] -= box1[2] + 0.02
        rect_cbar = np.copy(box1)
        rect_cbar[0] += box1[2] + 0.01
        rect_cbar[2] = 0.015
        cax = fig1.add_axes(rect_cbar)
        cax.tick_params(axis='y', which='major', direction='in')
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Spectral_r,
                                   norm=plt.Normalize(vmin=tstart * dtp,
                                                      vmax=tend * dtp))
        # fake up the array of the scalar mappable. Urgh...
        sm._A = []
        cbar = fig1.colorbar(sm, cax=cax)
        cbar.set_label(r'$t\Omega_{ci}$', fontsize=12)
        cbar.set_ticks((np.linspace(tstart, tend, tend - tstart + 1) * dtp))
        cbar.ax.tick_params(labelsize=10)

        # bg_str = str(int(bg * 10)).zfill(2)
        # fdir = '../img/cori_3d/particle_energization/bg' + bg_str + '/'
        # mkdir_p(fdir)
        # fname = (fdir + 'particle_' + plot_config["plot_type"] + '_' +
        #          species + '_' + str(iplot) + '.pdf')
        # fig1.savefig(fname)

    if show_plot:
        plt.show()
    else:
        plt.close('all')


def calc_vexb_kappa(plot_config):
    """Get the vexb dot magnetic curvature for the 2D simulations
    """
    tframe = plot_config["tframe"]
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_topox = pic_info.topology_x
    pic_topoy = pic_info.topology_y
    pic_topoz = pic_info.topology_z
    nx, ny, nz = pic_info.nx, pic_info.ny, pic_info.nz
    xmin, xmax = 0, pic_info.lx_di
    ymin, ymax = -pic_info.ly_di * 0.5, pic_info.ly_di * 0.5
    zmin, zmax = -pic_info.lz_di * 0.5, pic_info.lz_di * 0.5
    smime = math.sqrt(pic_info.mime)
    dx_de = pic_info.dx_di * smime
    dy_de = pic_info.dy_di * smime
    dz_de = pic_info.dz_di * smime
    kwargs = {"current_time": tframe,
              "xl": 0, "xr": pic_info.lx_di,
              "zb": -pic_info.lz_di, "zt": pic_info.lz_di}
    fname = pic_run_dir + "data/bx.gda"
    x, z, bx = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/by.gda"
    x, z, by = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/bz.gda"
    x, z, bz = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/ex.gda"
    x, z, ex = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/ey.gda"
    x, z, ey = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/ez.gda"
    x, z, ez = read_2d_fields(pic_info, fname, **kwargs)
    ib = 1.0/np.sqrt(bx**2 + by**2 + bz**2)
    bx = bx * ib
    by = by * ib
    bz = bz * ib
    kappax = (bx * np.gradient(bx, axis=1) / dx_de +
              bz * np.gradient(bx, axis=0) / dz_de)
    kappay = (bx * np.gradient(by, axis=1) / dx_de +
              bz * np.gradient(by, axis=0) / dz_de)
    kappaz = (bx * np.gradient(bz, axis=1) / dx_de +
              bz * np.gradient(bz, axis=0) / dz_de)
    kappa = np.sqrt(kappax**2 + kappay**2 + kappaz**2)
    kmin, kmax = 1E-6, 1E2
    nbins = 80
    kbins = np.logspace(math.log10(kmin), math.log10(kmax), nbins+1)
    kdist, _ = np.histogram(kappa, bins=kbins)
    fdata = np.zeros(nbins + 3)
    fdata[0] = kmin
    fdata[1] = kmax
    fdata[2] = nbins
    fdata[3:] = kdist
    fdir = '../data/power_law_index/kappa_dist/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + 'kappa_dist_' + str(tframe) + '.dat'
    fdata.tofile(fname)

    vexb_x = (ey * bz - ez * by) * ib
    vexb_y = (ez * bx - ex * bz) * ib
    vexb_z = (ex * by - ey * bx) * ib
    vexb_kappa = vexb_x * kappax + vexb_y * kappay + vexb_z * kappaz
    vkmin, vkmax = 1E-5, 1E3
    nbins = 80
    vkbins = np.zeros(2*nbins+3)
    fbins = np.logspace(math.log10(vkmin), math.log10(vkmax), nbins+1)
    vkbins[:nbins+1] = -fbins[::-1]
    vkbins[nbins+2:]= fbins
    vkdist, _ = np.histogram(vexb_kappa, bins=vkbins)
    fdata = np.zeros(2*nbins + 5)
    fdata[0] = vkmin
    fdata[1] = vkmax
    fdata[2] = nbins
    fdata[3:] = vkdist
    fdir = '../data/power_law_index/vexb_kappa_dist/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + 'vexb_kappa_dist_' + str(tframe) + '.dat'
    fdata.tofile(fname)
    # fbins_mid = 0.5 * (fbins[1:] + fbins[:-1])
    # vkdist = vkdist / np.diff(vkbins)
    # plt.loglog(fbins_mid, vkdist[nbins+2:])
    # plt.loglog(fbins_mid[::-1], vkdist[:nbins])
    # plt.show()

    fdir = pic_run_dir + "vexb_kappa/"
    mkdir_p(fdir)
    tindex = tframe * pic_info.fields_interval
    fname = fdir + 'vexb_kappa_' + str(tindex) + '.h5'
    with h5py.File(fname, 'a') as fh:
        dname = "Timestep_" + str(tindex)
        if dname in fh:
            fh[dname].write_direct(vexb_kappa)
        else:
            grp = fh.create_dataset(dname, (nz, nx), data=vexb_kappa)


def vexb_kappa_dist(plot_config, show_plot=True):
    """Get 2D histogram vexb_kappa and kappa
    """
    pic_run = plot_config["pic_run"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_run_dir = pic_info.run_dir
    ntf = pic_info.ntf
    kmin, kmax = 1E-6, 1E2
    nbins = 80
    kbins = np.logspace(math.log10(kmin), math.log10(kmax), nbins+1)
    kbins_mid = 0.5 * (kbins[1:] + kbins[:-1])

    vkmin, vkmax = 1E-5, 1E-1
    nbins = 40
    vkbins = np.zeros(2*nbins+3)
    fbins = np.logspace(math.log10(vkmin), math.log10(vkmax), nbins+1)
    vkbins[:nbins+1] = -fbins[::-1]
    vkbins[nbins+2:]= fbins
    vkbins_mid = 0.5 * (vkbins[1:] + vkbins[:-1])
    fdir = '../data/power_law_index/kappa_dist/'
    mkdir_p(fdir)
    oname = fdir + "kappa_dist_" + pic_run + ".h5"
    with h5py.File(oname, "w") as fh:
        fh.create_dataset("kappa_bins", kbins.shape, data=kbins)
        fh.create_dataset("vkappa_bins", vkbins.shape, data=vkbins)
        grp = fh.create_group("dist")
    for tframe in range(ntf):
        print("%d of %d" % (tframe, ntf))
        pic_topox = pic_info.topology_x
        pic_topoy = pic_info.topology_y
        pic_topoz = pic_info.topology_z
        nx, ny, nz = pic_info.nx, pic_info.ny, pic_info.nz
        xmin, xmax = 0, pic_info.lx_di
        ymin, ymax = -pic_info.ly_di * 0.5, pic_info.ly_di * 0.5
        zmin, zmax = -pic_info.lz_di * 0.5, pic_info.lz_di * 0.5
        smime = math.sqrt(pic_info.mime)
        dx_de = pic_info.dx_di * smime
        dy_de = pic_info.dy_di * smime
        dz_de = pic_info.dz_di * smime
        if "bg005" in pic_run:
            kwargs = {"current_time": tframe,
                      "xl": 0, "xr": pic_info.lx_di,
                      "zb": -pic_info.lz_di, "zt": pic_info.lz_di}
            fname = pic_run_dir + "data/bx.gda"
            x, z, bx = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_run_dir + "data/by.gda"
            x, z, by = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_run_dir + "data/bz.gda"
            x, z, bz = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_run_dir + "data/ex.gda"
            x, z, ex = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_run_dir + "data/ey.gda"
            x, z, ey = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_run_dir + "data/ez.gda"
            x, z, ez = read_2d_fields(pic_info, fname, **kwargs)
        else:
            fields_interval = int(pic_info.fields_interval)
            tindex = fields_interval * tframe
            fname = (pic_run_dir + "field_hdf5/T." + str(tindex) +
                     "/fields_" + str(tindex) + ".h5")
            emf = {}
            with h5py.File(fname, 'r') as fh:
                group = fh["Timestep_" + str(tindex)]
                for var in group:
                    dset = group[var]
                    emf[var]= dset[:, 0, :]
                bx = emf["cbx"]
                by = emf["cby"]
                bz = emf["cbz"]
                ex = emf["ex"]
                ey = emf["ey"]
                ez = emf["ez"]

        ib = 1.0/np.sqrt(bx**2 + by**2 + bz**2)
        bx = bx * ib
        by = by * ib
        bz = bz * ib
        kappax = (bx * np.gradient(bx, axis=1) / dx_de +
                  bz * np.gradient(bx, axis=0) / dz_de)
        kappay = (bx * np.gradient(by, axis=1) / dx_de +
                  bz * np.gradient(by, axis=0) / dz_de)
        kappaz = (bx * np.gradient(bz, axis=1) / dx_de +
                  bz * np.gradient(bz, axis=0) / dz_de)
        kappa = np.sqrt(kappax**2 + kappay**2 + kappaz**2)
        vexb_x = (ey * bz - ez * by) * ib
        vexb_y = (ez * bx - ex * bz) * ib
        vexb_z = (ex * by - ey * bx) * ib
        vexb_kappa = vexb_x * kappax + vexb_y * kappay + vexb_z * kappaz

        hist_xy, _, _ = np.histogram2d(kappa.flatten(), vexb_kappa.flatten(),
                                       bins=[kbins, vkbins])
        with h5py.File(oname, "a") as fh:
            grp = fh["dist"]
            dname = "Frame#" + str(tframe)
            grp.create_dataset(dname, hist_xy.shape, data=hist_xy)

    # fig = plt.figure(figsize=[7, 5])
    # rect = [0.15, 0.15, 0.8, 0.8]
    # ax = fig.add_axes(rect)
    # vmin, vmax = 1, 5E5
    # kappa_dist = np.sum(hist_xy, axis=1) / np.diff(kbins)
    # ik1 = np.argmax(kappa_dist)
    # ik2 = np.max(np.nonzero(kappa_dist))
    # fpowerh = kbins_mid[ik1:ik2]**-3.5
    # fpowerl = kbins_mid[:ik1]**0.5
    # fnormh = kappa_dist[ik1] / fpowerh[0] * 5
    # fnorml = kappa_dist[ik1] / fpowerl[-1] * 2
    # ax.loglog(kbins_mid, kappa_dist)
    # ax.loglog(kbins_mid[ik1:ik2], fpowerh*fnormh)
    # ax.loglog(kbins_mid[:ik1], fpowerl*fnorml)
    # # nbins = vkbins_mid.shape[0]
    # # vkappa_dist = np.sum(hist_xy, axis=0)
    # # ax.loglog(vkbins_mid[nbins//2:], vkappa_dist[nbins//2:0:-1])
    # # ax.loglog(vkbins_mid[nbins//2:], vkappa_dist[nbins//2:])
    # # print(hist_xy.min(), hist_xy.max())
    # # p2 = ax.imshow(hist_xy, extent=[xmin, xmax, zmin, zmax],
    # #                norm = LogNorm(vmin=vmin, vmax=vmax),
    # #                cmap=plt.cm.plasma, aspect='auto',
    # #                origin='lower', interpolation='none')
    # plt.show()


def velocity_dist(plot_config, show_plot=True):
    """Get the distributions of velocity field
    """
    pic_run = plot_config["pic_run"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_run_dir = pic_info.run_dir
    ntf = pic_info.ntf

    umin, umax = 1E-3, 1E1
    nbins = 40
    ubins = np.zeros(2*nbins+3)
    fbins = np.logspace(math.log10(umin), math.log10(umax), nbins+1)
    ubins[:nbins+1] = -fbins[::-1]
    ubins[nbins+2:]= fbins
    ubins_mid = 0.5 * (ubins[1:] + ubins[:-1])
    fdir = '../data/power_law_index/vel_dist/'
    mkdir_p(fdir)
    oname = fdir + "kappa_dist_" + pic_run + ".h5"
    with h5py.File(oname, "w") as fh:
        fh.create_dataset("ubins", ubins.shape, data=ubins)
        grp = fh.create_group("ux_dist")
        grp = fh.create_group("uy_dist")
        grp = fh.create_group("uz_dist")
    for tframe in range(ntf):
        print("%d of %d" % (tframe, ntf))
        pic_topox = pic_info.topology_x
        pic_topoy = pic_info.topology_y
        pic_topoz = pic_info.topology_z
        nx, ny, nz = pic_info.nx, pic_info.ny, pic_info.nz
        xmin, xmax = 0, pic_info.lx_di
        ymin, ymax = -pic_info.ly_di * 0.5, pic_info.ly_di * 0.5
        zmin, zmax = -pic_info.lz_di * 0.5, pic_info.lz_di * 0.5
        smime = math.sqrt(pic_info.mime)
        dx_de = pic_info.dx_di * smime
        dy_de = pic_info.dy_di * smime
        dz_de = pic_info.dz_di * smime
        if "bg005" in pic_run:
            kwargs = {"current_time": tframe,
                      "xl": xmin, "xr": xmax,
                      "zb": zmin, "zt": zmax}
            fname = pic_run_dir + "data/uex.gda"
            x, z, uex = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_run_dir + "data/uey.gda"
            x, z, uey = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_run_dir + "data/uez.gda"
            x, z, uez = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_run_dir + "data/uix.gda"
            x, z, uix = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_run_dir + "data/uiy.gda"
            x, z, uiy = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_run_dir + "data/uiz.gda"
            x, z, uiz = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_run_dir + "data/ne.gda"
            x, z, ne = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_run_dir + "data/ni.gda"
            x, z, ni = read_2d_fields(pic_info, fname, **kwargs)
            irho = 1.0 / (ne + ni * mime)
            ux = (uex * ne + uix * ni * mime) * irho
            uy = (uey * ne + uiy * ni * mime) * irho
            uz = (uez * ne + uiz * ni * mime) * irho
        else:
            fields_interval = pic_info.fields_interval
            tindex = fields_interval * tframe
            xs, xe = 0, pic_info.nx
            zs, ze = 0, pic_info.nz
            vels = {}
            for species in ["e", "i"]:
                sname = "electron" if species == 'e' else "ion"
                fname = (pic_run_dir + "hydro_hdf5/T." + str(tindex) +
                         "/hydro_" + sname + "_" + str(tindex) + ".h5")
                hydro = {}
                with h5py.File(fname, 'r') as fh:
                    group = fh["Timestep_" + str(tindex)]
                    for var in ["rho", "px", "py", "pz"]:
                        dset = group[var]
                        hydro[var]= dset[xs:xe, 0, zs:ze].T

                vels["n" + species] = hydro["rho"]
                vels["p" + species + "x"] = hydro["px"]
                vels["p" + species + "y"] = hydro["py"]
                vels["p" + species + "z"] = hydro["pz"]
            irho =  1.0 / (vels["ne"] + vels["ni"] * mime)
            ux = (vels["pex"] + vels["pix"]) * irho
            uy = (vels["pey"] + vels["piy"]) * irho
            uz = (vels["pez"] + vels["piz"]) * irho

        histx, _ = np.histogram(ux.flatten(), bins=ubins)
        histy, _ = np.histogram(uy.flatten(), bins=ubins)
        histz, _ = np.histogram(uz.flatten(), bins=ubins)
        with h5py.File(oname, "a") as fh:
            grp = fh["ux_dist"]
            dname = "Frame#" + str(tframe)
            grp.create_dataset(dname, histx.shape, data=histx)
            grp = fh["uy_dist"]
            dname = "Frame#" + str(tframe)
            grp.create_dataset(dname, histy.shape, data=histy)
            grp = fh["uz_dist"]
            dname = "Frame#" + str(tframe)
            grp.create_dataset(dname, histz.shape, data=histz)


def calc_curvature_radius(plot_config):
    """Get the radius of magnetic curvature
    """
    tframe = plot_config["tframe"]
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_topox = pic_info.topology_x
    pic_topoy = pic_info.topology_y
    pic_topoz = pic_info.topology_z
    nx, ny, nz = pic_info.nx, pic_info.ny, pic_info.nz
    xmin, xmax = 0, pic_info.lx_di
    zmin, zmax = -pic_info.lz_di * 0.125, pic_info.lz_di * 0.125
    smime = math.sqrt(pic_info.mime)
    dx_de = pic_info.dx_di * smime
    dy_de = pic_info.dy_di * smime
    dz_de = pic_info.dz_di * smime
    kwargs = {"current_time": tframe,
              "xl": xmin, "xr": xmax,
              "zb": zmin, "zt": zmax}
    fname = pic_run_dir + "data/bx.gda"
    x, z, bx = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/by.gda"
    x, z, by = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/bz.gda"
    x, z, bz = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/Ay.gda"
    x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
    ib = 1.0/np.sqrt(bx**2 + by**2 + bz**2)
    bx = bx * ib
    by = by * ib
    bz = bz * ib
    kappax = (bx * np.gradient(bx, axis=1) / dx_de +
              bz * np.gradient(bx, axis=0) / dz_de)
    kappay = (bx * np.gradient(by, axis=1) / dx_de +
              bz * np.gradient(by, axis=0) / dz_de)
    kappaz = (bx * np.gradient(bz, axis=1) / dx_de +
              bz * np.gradient(bz, axis=0) / dz_de)
    curv_radius = 1.0 / np.sqrt(kappax**2 + kappay**2 + kappaz**2)
    fig = plt.figure(figsize=[12, 5])
    rect0 = [0.08, 0.55, 0.62, 0.4]
    hgap, vgap = 0.03, 0.05
    rect = np.copy(rect0)
    ax = fig.add_axes(rect)
    vmin, vmax = 10, 1E4
    ng = 5
    kernel = np.ones((ng,ng)) / float(ng*ng)
    curv_radius = signal.convolve2d(curv_radius, kernel, mode='same')
    # xcuts = [18, 20, 22, 24]
    xcuts = np.linspace(23, 24, 6)
    zcuts = [0]
    xindices = [find_nearest(x, xcut)[0] for xcut in xcuts]
    zindices = [find_nearest(z, zcut)[0] for zcut in zcuts]
    ng = 3
    kernel = np.ones((ng,ng)) / float(ng*ng)
    field = curv_radius
    p2 = ax.imshow(field, extent=[xmin, xmax, zmin, zmax],
                   norm = LogNorm(vmin=vmin, vmax=vmax),
                   cmap=plt.cm.plasma, aspect='auto',
                   origin='lower', interpolation='none')
    levels = np.logspace(math.log10(np.min(field)),
                         math.log10(np.max(field)), 10)
    levels = [50]
    cs = ax.contour(x, z, field, colors='k', linewidths=0.5, levels=levels)
    levels = np.linspace(np.min(Ay), np.max(Ay), 20)
    cs = ax.contour(x, z, Ay, colors='k', linewidths=0.5, levels=levels)
    for xcut in xcuts:
        ax.plot([xcut, xcut], [zmin, zmax], color='k',
                linewidth=0.5, linestyle=':')
    rect[1] -= rect[3] + vgap
    ax = fig.add_axes(rect)
    for iz in zindices:
        ax.plot(x, field[iz, :])
    ax.set_xlim([xmin, xmax])
    # ax.set_ylim([0, 100])

    rect = np.copy(rect0)
    rect[0] += rect[2] + hgap
    rect[2] = 0.25
    ax = fig.add_axes(rect)
    for ix in xindices:
        fdata = field[:, ix]
        ng = 5
        kernel = np.ones((ng)) / float(ng)
        fdata = signal.convolve(fdata, kernel, mode='same')
        ax.plot(fdata, z)
    # ax.set_xlim([0, 100])
    ax.set_ylim([zmin, zmax])

    plt.show()


def calc_vdotE(plot_config):
    """Get the velocity dot electric field
    """
    tframe = plot_config["tframe"]
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_topox = pic_info.topology_x
    pic_topoy = pic_info.topology_y
    pic_topoz = pic_info.topology_z
    nx, ny, nz = pic_info.nx, pic_info.ny, pic_info.nz
    xmin, xmax = 0, pic_info.lx_di
    zmin, zmax = -pic_info.lz_di * 0.125, pic_info.lz_di * 0.125
    smime = math.sqrt(pic_info.mime)
    dx_de = pic_info.dx_di * smime
    dy_de = pic_info.dy_di * smime
    dz_de = pic_info.dz_di * smime
    kwargs = {"current_time": tframe,
              "xl": xmin, "xr": xmax,
              "zb": zmin, "zt": zmax}
    fname = pic_run_dir + "data/ex.gda"
    x, z, ex = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/ey.gda"
    x, z, ey = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/ez.gda"
    x, z, ez = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/v" + species + "x.gda"
    x, z, vx = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/v" + species + "y.gda"
    x, z, vy = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/v" + species + "z.gda"
    x, z, vz = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/Ay.gda"
    x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)

    pcharge = -1 if species in ['e', 'electron'] else 1
    vdotE = pcharge * (vx * ex + vy * ey + vz * ez)

    fig = plt.figure(figsize=[12, 5])
    rect0 = [0.08, 0.15, 0.82, 0.8]
    hgap, vgap = 0.03, 0.05
    rect = np.copy(rect0)
    ax = fig.add_axes(rect)
    vmin, vmax = 10, 1E4
    ng = 5
    kernel = np.ones((ng,ng)) / float(ng*ng)
    vdotE = signal.convolve2d(vdotE, kernel, mode='same')
    p2 = ax.imshow(vdotE, extent=[xmin, xmax, zmin, zmax],
                   vmin=-5E-1, vmax=5E-1,
                   cmap=plt.cm.seismic, aspect='auto',
                   origin='lower', interpolation='none')
    print(np.min(vdotE), np.max(vdotE))
    levels = np.linspace(np.min(vdotE), np.max(vdotE), 10)
    levels = [0.1]
    cs = ax.contour(x, z, np.abs(vdotE), colors='k', linewidths=0.5,
                    levels=levels)
    levels = np.linspace(np.min(Ay), np.max(Ay), 20)
    cs = ax.contour(x, z, Ay, colors='k', linewidths=0.5, levels=levels)

    plt.show()


def plot_vexb_kappa(plot_config, show_plot=True):
    """
    Plot the distribution of vexb dot magnetic curvature for the 2D simulations
    """
    mpl.rc('text', usetex=True)
    tframe = plot_config["tframe"]
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    tindex = tframe * pic_info.fields_interval
    fdir = '../data/power_law_index/vexb_kappa_dist/' + pic_run + '/'
    fname = fdir + 'vexb_kappa_dist_' + str(tframe) + '.dat'
    fdata = np.fromfile(fname)
    vkmin = fdata[0]
    vkmax = fdata[1]
    nbins = fdata[2]
    vkdist = fdata[3:]
    nbins = 80
    vkbins = np.zeros(2*nbins+3)
    fbins = np.logspace(math.log10(vkmin), math.log10(vkmax), nbins+1)
    vkbins[:nbins+1] = -fbins[::-1]
    vkbins[nbins+2:]= fbins
    vkbins_mid = 0.5 * (vkbins[1:] + vkbins[:-1])
    vkdist = div0(vkdist, np.diff(vkbins))
    fig = plt.figure(figsize=[7, 8])
    rect = [0.12, 0.35, 0.82, 0.6]
    ax = fig.add_axes(rect)
    ax.set_prop_cycle('color', COLORS)
    text1 = r'\boldsymbol{v}_{\boldsymbol{E}}\cdot\boldsymbol{\kappa}'
    label1 = r'$' + text1 + ' > 0$'
    ax.loglog(vkbins_mid[nbins+1:], vkdist[nbins+1:], label=label1)
    label2 = r'$' + text1 + ' < 0$'
    ax.loglog(vkbins_mid[nbins+1:], np.flip(vkdist[:nbins+1]), label=label2)
    ax.legend(loc=3, prop={'size': 16}, ncol=1,
             shadow=False, fancybox=False, frameon=False)
    ax.tick_params(bottom=True, top=True, left=True, right=False)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.tick_params(labelsize=12)
    xlabel = r'$|' + text1 + r'|$'
    ax.set_xlabel(xlabel, fontsize=16)
    ylabel = r'$f(|' + text1 + r'|)$'
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_xlim([1E-5, 1E2])
    ax.set_ylim([1E-1, 1E11])
    gamma_e = 4.0/3 if pic_info.Te > 0.1 else 1.5  # Adiabatic index
    gamma_i = 5.0/3
    # In typical VPIC simulation, me=1, c=1
    n0 = pic_info.n0
    mime = pic_info.mime
    enthalpy_e =  n0 * (1 + pic_info.Te * gamma_e / (gamma_e - 1))
    enthalpy_i =  n0 * (mime + pic_info.Ti * gamma_i / (gamma_i - 1))
    enthalpy = enthalpy_e + enthalpy_i
    sigma = pic_info.b0**2 / enthalpy
    va = math.sqrt(sigma / (sigma + 1))
    if int(mime) == 1:
        ndi_max = 60.0
    elif int(mime) == 25:
        ndi_max = 40.0
    elif mime > 1000:
        ndi_max = 10.0
    else:
        ndi_max = 10.0
    vsh = va/math.sqrt(mime)/ndi_max  # vA/(ndi_max * di)
    ax.plot([vsh, vsh], ax.get_ylim(), color='k',
            linewidth=1, linestyle='-')
    ax.plot([vsh*0.5, vsh*0.5], ax.get_ylim(), color='k',
            linewidth=1, linestyle='--')
    ax.plot([vsh*2, vsh*2], ax.get_ylim(), color='k',
            linewidth=1, linestyle='--')
    dtwpe_fields = math.ceil(pic_info.fields_interval * pic_info.dtwpe / 0.1) * 0.1
    twpe = tframe * dtwpe_fields
    text = r'$t\omega_{pe}=' + str(twpe) + '$'
    ax.set_title(text, fontsize=16)

    rect[3] = 0.2
    rect[1] -= rect[3] + 0.07
    xmin, xmax = 0, pic_info.lx_di
    zmin, zmax = -pic_info.lz_di * 0.5, pic_info.lz_di * 0.5
    fname = pic_run_dir + "vexb_kappa/vexb_kappa_" + str(tindex) + ".h5"
    with h5py.File(fname, 'r') as fh:
        dname = "Timestep_" + str(tindex)
        dset = fh[dname]
        vexb_kappa = np.zeros(dset.shape, dset.dtype)
        dset.read_direct(vexb_kappa)
    nz, nx = vexb_kappa.shape
    x = np.linspace(xmin, xmax, nx, endpoint=False)
    z = np.linspace(zmin, zmax, nz, endpoint=False)
    vmin, vmax = -vsh*25, vsh*25
    ax = fig.add_axes(rect)
    ng = 5
    kernel = np.ones((ng,ng)) / float(ng*ng)
    vexb_kappa = signal.convolve2d(vexb_kappa, kernel, mode='same')
    p2 = ax.imshow(vexb_kappa, extent=[xmin, xmax, zmin, zmax],
                   vmin=vmin, vmax=vmax,
                   cmap=plt.cm.seismic, aspect='auto',
                   origin='lower', interpolation='none')
    levels = [vsh]
    cs = ax.contour(x, z, np.abs(vexb_kappa), colors='k',
                    linewidths=0.5, levels=levels)
    ax.set_ylim([0.5 * zmin, 0.5 * zmax])
    ax.tick_params(bottom=True, top=True, left=True, right=False)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.tick_params(labelsize=12)
    ax.set_xlabel(r'$x/d_i$', fontsize=16)
    ax.set_ylabel(r'$z/d_i$', fontsize=16)
    rect_cbar = [0.72, 0.24, 0.2, 0.01]
    cbar_ax = fig.add_axes(rect_cbar)
    cbar = fig.colorbar(p2, cax=cbar_ax, orientation='horizontal',
                        extend='both')
    cbar.ax.tick_params(labelsize=16)
    lablel = r'$' + text1 + r'$'
    cbar_ax.set_xlabel(lablel, fontsize=16)
    fdir = '../img/power_law_index/vexb_kappa_dist/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + 'vexb_kappa_dist_' + species + '_' + str(tframe) + '.jpg'
    fig.savefig(fname, dpi=400)

    if show_plot:
        plt.show()
    else:
        plt.close()


def comp_vexb_kappa(plot_config, show_plot=True):
    """
    Compare the distribution of vexb dot magnetic curvature
    """
    mpl.rc('text', usetex=True)
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    fdir = '../data/power_law_index/vexb_kappa_dist/' + pic_run + '/'

    vkappas = np.zeros(pic_info.ntf)
    for tframe in range(pic_info.ntf):
        print("Time frame: %d" % tframe)
        fname = fdir + 'vexb_kappa_dist_' + str(tframe) + '.dat'
        if not os.path.isfile(fname):
            break
        fdata = np.fromfile(fname)
        vkmin = fdata[0]
        vkmax = fdata[1]
        nbins = fdata[2]
        vkdist = fdata[3:]
        nbins = 80
        vkbins = np.zeros(2*nbins+3)
        fbins = np.logspace(math.log10(vkmin), math.log10(vkmax), nbins+1)
        vkbins[:nbins+1] = -fbins[::-1]
        vkbins[nbins+2:]= fbins
        vkbins_mid = 0.5 * (vkbins[1:] + vkbins[:-1])
        vkdist = div0(vkdist, np.diff(vkbins))
        color = plt.cm.jet(tframe/float(pic_info.ntf), 1)
        dratio = vkdist[nbins+1:] / np.flip(vkdist[:nbins+1])
        vkappas[tframe] = vkbins_mid[nbins + 1 + np.argmax(dratio > 1.1)]

    fig = plt.figure(figsize=[7, 8])
    rect = [0.12, 0.35, 0.82, 0.6]
    ax = fig.add_axes(rect)
    ax.set_prop_cycle('color', COLORS)
    ax.plot(vkappas)

    if show_plot:
        plt.show()
    else:
        plt.close()


def velocity_profile(plot_config):
    """Get the radius of magnetic curvature
    """
    tframe = plot_config["tframe"]
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_topox = pic_info.topology_x
    pic_topoy = pic_info.topology_y
    pic_topoz = pic_info.topology_z
    nx, ny, nz = pic_info.nx, pic_info.ny, pic_info.nz
    xmin, xmax = 0, pic_info.lx_di * 0.3
    zmin, zmax = -pic_info.lz_di * 0.125, pic_info.lz_di * 0.125
    smime = math.sqrt(pic_info.mime)
    dx_de = pic_info.dx_di * smime
    dy_de = pic_info.dy_di * smime
    dz_de = pic_info.dz_di * smime
    gamma_e = 4.0/3 if pic_info.Te > 0.1 else 1.5  # Adiabatic index
    gamma_i = 5.0/3
    # In typical VPIC simulation, me=1, c=1
    n0 = pic_info.n0
    mime = pic_info.mime
    enthalpy_e =  n0 * (1 + pic_info.Te * gamma_e / (gamma_e - 1))
    enthalpy_i =  n0 * (mime + pic_info.Ti * gamma_i / (gamma_i - 1))
    enthalpy = enthalpy_e + enthalpy_i
    sigma = pic_info.b0**2 / enthalpy
    va = math.sqrt(sigma / (sigma + 1))
    if int(mime) == 1:
        ndi_max = 60.0
    elif int(mime) == 25:
        ndi_max = 40.0
    elif mime > 1000:
        ndi_max = 10.0
    else:
        ndi_max = 10.0
    vsh = va/math.sqrt(mime)/ndi_max  # vA/(ndi_max * di)
    # vsh = 0.001
    kwargs = {"current_time": tframe,
              "xl": xmin, "xr": xmax,
              "zb": zmin, "zt": zmax}
    fname = pic_run_dir + "data/bx.gda"
    x, z, bx = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/by.gda"
    x, z, by = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/bz.gda"
    x, z, bz = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/ex.gda"
    x, z, ex = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/ey.gda"
    x, z, ey = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/ez.gda"
    x, z, ez = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/Ay.gda"
    x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
    ib = 1.0/np.sqrt(bx**2 + by**2 + bz**2)
    bx = bx * ib
    by = by * ib
    bz = bz * ib
    vexb_x = (ey * bz - ez * by) * ib
    vexb_y = (ez * bx - ex * bz) * ib
    vexb_z = (ex * by - ey * bx) * ib
    kappax = (bx * np.gradient(bx, axis=1) / dx_de +
              bz * np.gradient(bx, axis=0) / dz_de)
    kappay = (bx * np.gradient(by, axis=1) / dx_de +
              bz * np.gradient(by, axis=0) / dz_de)
    kappaz = (bx * np.gradient(bz, axis=1) / dx_de +
              bz * np.gradient(bz, axis=0) / dz_de)
    vexb_kappa = vexb_x * kappax + vexb_y * kappay + vexb_z * kappaz
    fig = plt.figure(figsize=[12, 5])
    rect0 = [0.08, 0.55, 0.62, 0.4]
    hgap, vgap = 0.03, 0.05
    rect = np.copy(rect0)
    ax = fig.add_axes(rect)
    xcuts = np.linspace(15, 24, 6)
    # xcuts = np.linspace(30, 40, 6)
    zcuts = [0.2]
    # zcuts = [0.0]
    xindices = [find_nearest(x, xcut)[0] for xcut in xcuts]
    zindices = [find_nearest(z, zcut)[0] for zcut in zcuts]
    ng = 5
    kernel = np.ones((ng,ng)) / float(ng*ng)
    field = vexb_kappa
    vmin, vmax = -vsh*25, vsh*25
    # field = vexb_x
    # vmin, vmax = -1.0, 1.0
    field = signal.convolve(field, kernel, mode='same')
    p2 = ax.imshow(field, extent=[xmin, xmax, zmin, zmax],
                   vmin=vmin, vmax=vmax,
                   cmap=plt.cm.seismic, aspect='auto',
                   origin='lower', interpolation='none')
    levels = np.linspace(np.min(Ay), np.max(Ay), 40)
    cs = ax.contour(x, z, Ay, colors='k', linewidths=0.5, levels=levels)
    levels = [vsh]
    cs = ax.contour(x, z, np.abs(field), colors='k',
                    linewidths=0.5, levels=levels)
    for xcut in xcuts:
        ax.plot([xcut, xcut], [zmin, zmax], color='k',
                linewidth=0.5, linestyle=':')
    for zcut in zcuts:
        ax.plot([xmin, xmax], [zcut, zcut], color='k',
                linewidth=0.5, linestyle=':')
    rect[1] -= rect[3] + vgap
    ax = fig.add_axes(rect)
    for iz in zindices:
        ax.plot(x, field[iz, :])
    ax.set_xlim([xmin, xmax])
    # ax.set_ylim([0, 100])

    rect = np.copy(rect0)
    rect[0] += rect[2] + hgap
    rect[2] = 0.25
    ax = fig.add_axes(rect)
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    ax.set_prop_cycle('color', COLORS)
    for ix, xcut in zip(xindices, xcuts):
        fdata = field[:, ix]
        ng = 5
        kernel = np.ones((ng)) / float(ng)
        fdata = signal.convolve(fdata, kernel, mode='same')
        ax.plot(fdata, z, label=str(xcut))
    ax.legend(loc=2, prop={'size': 12}, ncol=1,
             shadow=False, fancybox=False, frameon=False)
    # ax.set_xlim([0, 100])
    ax.set_ylim([zmin, zmax])

    plt.show()


def calc_flow_acc(plot_config, show_plot=True):
    """Calculate flow acceleration
    """
    tframe = plot_config["tframe"]
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_topox = pic_info.topology_x
    pic_topoy = pic_info.topology_y
    pic_topoz = pic_info.topology_z
    nx, ny, nz = pic_info.nx, pic_info.ny, pic_info.nz
    xmin, xmax = 0, pic_info.lx_di
    ymin, ymax = -pic_info.ly_di * 0.5, pic_info.ly_di * 0.5
    zmin, zmax = -pic_info.lz_di * 0.5, pic_info.lz_di * 0.5
    smime = math.sqrt(pic_info.mime)
    dx_de = pic_info.dx_di * smime
    dy_de = pic_info.dy_di * smime
    dz_de = pic_info.dz_di * smime
    kwargs = {"current_time": tframe,
              "xl": 0, "xr": pic_info.lx_di,
              "zb": -pic_info.lz_di, "zt": pic_info.lz_di}
    fname = pic_run_dir + "data/bx.gda"
    x, z, bx = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/by.gda"
    x, z, by = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/bz.gda"
    x, z, bz = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/ex.gda"
    x, z, ex = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/ey.gda"
    x, z, ey = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_run_dir + "data/ez.gda"
    x, z, ez = read_2d_fields(pic_info, fname, **kwargs)
    ib = 1.0/np.sqrt(bx**2 + by**2 + bz**2)
    bx = bx * ib
    by = by * ib
    bz = bz * ib
    vexb_x = (ey * bz - ez * by) * ib
    vexb_y = (ez * bx - ex * bz) * ib
    vexb_z = (ex * by - ey * bx) * ib

    dvmin, dvmax = 1E0, 1E8
    nbins = 80
    dvbins = np.logspace(math.log10(dvmin), math.log10(dvmax), nbins+1)
    dvbins_mid = 0.5 * (dvbins[1:] + dvbins[:-1])
    fdata = div0(1.0, np.abs(np.gradient(vexb_x, dx_de, axis=1)))
    dvdist, _ = np.histogram(fdata, bins=dvbins)
    dvdist = dvdist / np.diff(dvbins)

    fig = plt.figure(figsize=[7, 5])
    rect = [0.12, 0.12, 0.82, 0.8]
    ax = fig.add_axes(rect)
    ax.loglog(dvbins_mid, dvdist)
    ax.tick_params(bottom=True, top=True, left=True, right=False)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.tick_params(labelsize=12)

    plt.show()


def plot_spect_species(plot_config, show_plot=True):
    """plot energy spectrum for different test-particle species

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    tstart = plot_config["tstart"]
    tend = plot_config["tend"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    if species in ["e", "electron"]:
        sname = "electron"
    else:
        sname = "Ion"
    sp_list = [sname]
    sp_name_list = ["Regular " + sname]
    test_species = ["wo_epara", "wo_eparay", "wo_egtb", "egtb", "egtb_egtb"]
    species_name = [r"$E_\parallel=0$", r"$E_{\parallel y}=0$",
                    r"$E=0 \text{ when } E>B$", "tracers",
                    r"tracers passing $E>B$"]
    for tsp, tsp_name in zip(test_species, species_name):
        sp_list.append(sname + '_' + tsp)
        sp_name_list.append("Test " + sname + ": " + tsp_name)
    norms = [1, 10, 10, 10, 10, 10, 10]
    emin, emax = 1E-4, 1E6
    nbins = 1000
    dloge = (math.log10(emax) - math.log10(emin)) / (nbins - 1)
    emin0 = 10**(math.log10(emin) - dloge)
    ebins = np.logspace(math.log10(emin0), math.log10(emax), nbins+1)
    ebins_mid = (ebins[:-1] + ebins[1:]) * 0.5
    debins = np.diff(ebins)
    dtwpe_fields = math.ceil(pic_info.fields_interval * pic_info.dtwpe / 0.1) * 0.1
    espect_elb = np.zeros(nbins)
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    for tframe in range(tstart, tend+1):
        print("Time frame: %d" % tframe)
        tindex = tframe * pic_info.fields_interval
        fig = plt.figure(figsize=[7, 5])
        rect = [0.12, 0.12, 0.82, 0.8]
        ax = fig.add_axes(rect)
        ax.set_prop_cycle('color', COLORS)
        isp = 0
        for sp, sp_name in zip(sp_list, sp_name_list):
            print(sp)
            fname = (pic_run_dir + "spectrum_combined/spectrum_" + sp +
                     "_" + str(tindex) + ".dat")
            flog = np.fromfile(fname, dtype=np.float32)
            espect = flog[3:]  # the first 3 are magnetic field
            if sp != sname + "_egtb":
                ax.loglog(ebins_mid, espect*norms[isp], linewidth=2, label=sp_name)
            else:
                espect_elb = np.copy(espect)
            if sp == sname + "_egtb_egtb":
                espect_elb -= espect
            isp += 1
        sp_name = r"tracers passing $E\leq B$"
        ax.loglog(ebins_mid, espect_elb*norms[isp], linewidth=2, label=sp_name)
        ax.legend(loc=3, prop={'size': 12}, ncol=1,
                 shadow=False, fancybox=False, frameon=False)
        ax.set_xlim([1E-1, 2E3])
        ax.set_ylim([1E3, 5E7])
        ax.set_xlabel(r'$\gamma - 1$', fontsize=16)
        ax.set_ylabel(r'$(\gamma - 1)f(\gamma - 1)$', fontsize=16)
        ax.tick_params(labelsize=12)
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')
        twpe = tframe * dtwpe_fields
        text1 = r'$t\omega_{pe}=' + str(twpe) + '$'
        ax.set_title(text1, fontsize=16)
        fdir = '../img/power_law_index/spect_species/' + pic_run + '/'
        mkdir_p(fdir)
        fname = fdir + 'spects_' + species + '_' + str(tframe) + '.pdf'
        fig.savefig(fname)
        if show_plot:
            plt.show()
        else:
            plt.close()


def acc_esc_rate(plot_config, show_plot=True):
    """calculate particle acceleration and escape rates

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    tframe = plot_config["tframe"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    alpha_min = plot_config["vkappa_threshold"]
    mime = pic_info.mime
    ntp = len(os.listdir(pic_run_dir + "particle/"))
    spect_params = get_spect_params(pic_run)
    pindex = spect_params["power_index"]
    emin, emax = spect_params["energy_range"]
    n0 = pic_info.n0
    gamma_e = 4.0/3 if pic_info.Te > 0.1 else 1.5  # Adiabatic index
    gamma_i = 5.0/3
    # In typical VPIC simulation, me=1, c=1
    enthalpy_e =  n0 * (1 + pic_info.Te * gamma_e / (gamma_e - 1))
    enthalpy_i =  n0 * (mime + pic_info.Ti * gamma_i / (gamma_i - 1))
    enthalpy = enthalpy_e + enthalpy_i
    sigma = pic_info.b0**2 / enthalpy
    sigma_c = pic_info.b0**2 / (n0 * (1 + mime))
    sigmae_c = pic_info.b0**2 / n0
    sigmai_c = pic_info.b0**2 / (n0 * mime)
    sigmae_h = pic_info.b0**2 / enthalpy_e
    sigmai_h = pic_info.b0**2 / enthalpy_i
    beta_e = 2 * n0 * pic_info.Te / pic_info.b0**2

    # # Get guide field strength from PIC run name
    # res = list(filter(lambda x: 'bg' in x, pic_run.split('_')))[0]
    # bg_str = res[2] + '.' + res[3:]
    # bg = float(bg_str)

    if species in ["e", "electron"]:
        sname = "electron"
    else:
        sname = "Ion"
    temp = pic_info.Te if species in ['e', 'electron'] else pic_info.Ti

    va = math.sqrt(sigma / (sigma + 1))
    if int(mime) == 1:
        ndi_max = 60.0
    elif int(mime) == 25:
        ndi_max = 40.0
    elif mime > 1000:
        ndi_max = 10.0
    else:
        ndi_max = 10.0
    # ndi_max = 10 * math.sqrt(1+pic_info.Ti/pic_info.mime)
    alpha_min = va/math.sqrt(mime)/ndi_max  # vA/(ndi_max * di)
    # alpha_min /= bg**2 + 1.0  # consider guide field

    fpath = "../data/particle_interp/" + pic_run + "/"
    tindex = pic_info.particle_interval
    fname = fpath + "acc_rate_dist_" + species + "_" + str(tindex) + ".gda"
    fdata = np.fromfile(fname, dtype=np.float64)
    nvar = int(fdata[2])   # number of variables

    nhigh_acc_t = np.zeros(ntp)
    nhigh_esc_t = np.zeros(ntp)
    arate_acc_t = np.zeros([nvar, ntp])
    arate_esc_t = np.zeros([nvar, ntp])

    for tframe in range(1, ntp):
        print("Time frame: %d" % tframe)
        tindex = tframe * pic_info.particle_interval
        fname = fpath + "acc_rate_dist_" + species + "_" + str(tindex) + ".gda"
        fdata = np.fromfile(fname, dtype=np.float64)
        nalpha = int(fdata[0]) # number of bins of the rates
        nbins = int(fdata[1])  # number of bins in energy
        nvar = int(fdata[2])   # number of variables
        ebins = fdata[3:nbins+3]  # energy bins
        ebins /= temp  # normalize by initial temperature
        alpha_bins = fdata[nbins+3:nbins+nalpha+3]  # acceleration rates bins
        alpha_bins_mid = 0.5 * (alpha_bins[1:] + alpha_bins[:-1])
        # variables: 0-Epara, 1-Eperp, 2-compression, 3-shear, 4-curvature,
        #   5-gradient, 6-para_drift, 7-mu, 8-polar_time, 9-polar_spatial,
        #   10-inertial_time, 11-inertial_spatial, 12-polar_fluid_time,
        #   13-polar_fluid_spatial, 14-polar_time_v, 15-polar_spatial_v,
        #   16-ptensor
        # fdist = fdata[nbins+nalpha+3:].reshape((nvar, nbins, (nalpha+1)*4))
        fdist = fdata[nbins+nalpha+3:].reshape((nvar, nbins, (nalpha+1)*6))

        es, _ = find_nearest(ebins, emin)
        ee, _ = find_nearest(ebins, emax)
        alpha_s, _ = find_nearest(alpha_bins, alpha_min)
        fdist_high = np.sum(fdist[:, es:ee, :], axis=1)
        nalpha0 = nalpha + 1
        nptl_bins = fdist_high[:, :nalpha0*2]
        dene_bins = fdist_high[:, nalpha0*2:nalpha0*4]
        alpha_bins = div0(dene_bins, nptl_bins)
        fnptl = nptl_bins[4]
        fdene = dene_bins[4]
        pos_range = np.arange(nalpha0+1, nalpha0*2-1)
        neg_range = np.arange(nalpha0-2, 0, -1)
        fnptl_pos = fnptl[pos_range]
        fnptl_neg = fnptl[neg_range]
        fdene_pos = fdene[pos_range]
        fdene_neg = fdene[neg_range]
        if tframe == 40:
            plt.loglog(alpha_bins_mid, fnptl_pos)
            plt.loglog(alpha_bins_mid, fnptl_neg)
        nhigh_acc_t[tframe] = np.sum(fnptl_pos[alpha_s:] + fnptl_neg[alpha_s:])
        nhigh_esc_t[tframe] = np.sum(fnptl_pos + fnptl_neg) - nhigh_acc_t[tframe]

        fnptl_var_pos = nptl_bins[:, pos_range]
        fnptl_var_neg = nptl_bins[:, neg_range]
        fdene_var_pos = dene_bins[:, pos_range]
        fdene_var_neg = dene_bins[:, neg_range]

        arate_acc_t[:, tframe] = np.sum(fdene_var_pos[:, alpha_s:] +
                                        fdene_var_neg[:, alpha_s:], axis=1)
        arate_esc_t[:, tframe] = (np.sum(fdene_var_pos + fdene_var_neg, axis=1) -
                                  arate_acc_t[:, tframe])

    print("Particle initial temperature (Lorentz factor): %f" % temp)
    print("Alfven speed: %f" % va)
    print("alpha_min va/(%0.1f * di): %f" % (ndi_max, alpha_min))
    print("sigma: %f" % sigma)
    print("sigma_c: %f" % sigma_c)
    print("sigmae_c: %f" % sigmae_c)
    print("sigmae_h: %f" % sigmae_h)
    print("sigmai_c: %f" % sigmai_c)
    print("sigmai_h: %f" % sigmai_h)

    dtwpe_particle = pic_info.particle_interval * pic_info.dtwpe
    tparticles = np.arange(0, ntp) * dtwpe_particle
    tmin, tmax = tparticles[0], tparticles[-1]
    nhigh_tot = nhigh_acc_t + nhigh_esc_t
    dndt_inj = np.gradient(nhigh_tot, dtwpe_particle)
    dndt_esc = np.gradient(nhigh_esc_t, 0.5)
    dndt_tot = np.gradient(nhigh_tot, 0.5)
    npre = np.zeros(nhigh_acc_t.shape)
    npre[0] = nhigh_acc_t[0]
    npre[1:] = nhigh_acc_t[:-1]
    npre_esc = np.zeros(nhigh_acc_t.shape)
    npre_esc[0] = nhigh_esc_t[0]
    npre_esc[1:] = nhigh_esc_t[:-1]
    # btmp = npre + dndt_inj * dtwpe_particle
    # esc_rate = div0(btmp - np.sqrt(btmp**2 - 2 * npre * dndt_esc),
    #                 2 * npre * dtwpe_particle)
    # esc_rate = div0(np.gradient(nhigh_esc_t, dtwpe_particle),
    #                 npre + dndt_tot * 0.2)
    # esc_rate = (npre + dndt_inj * dtwpe_particle - nhigh_acc_t) / (npre * dtwpe_particle)
    # atmp = npre * 0.5
    # btmp = -(npre + 0.5 * dndt_inj * dtwpe_particle)
    # ctmp = npre + dndt_inj * dtwpe_particle - nhigh_acc_t
    # esc_rate = div0(-btmp - np.sqrt(btmp**2 - 4 * atmp * ctmp), npre) / (dtwpe_particle)
    # acc_rate = div0(arate_acc_t[0, :] + arate_acc_t[1, :], nhigh_acc_t)
    acc_rate = div0(arate_acc_t[1, :], nhigh_acc_t)
    acc_rate1 = div0(arate_acc_t[4, :], nhigh_acc_t)
    acc_rate = acc_rate1
    acc_rate_esc = div0(arate_esc_t[0, :] + arate_esc_t[1, :], nhigh_esc_t)
    acc_rate_esc_mid = np.copy(acc_rate_esc)
    acc_rate_esc_mid[1:] = 0.5 * (acc_rate_esc[1:] + acc_rate_esc[:-1])
    acc_rate_esc_mid2 = np.copy(acc_rate_esc)
    acc_rate_esc_mid2[1:-1] = 0.5 * (acc_rate_esc[1:-1] + acc_rate_esc[2:])
    # dnhigh_esc = (npre_esc * (acc_rate_esc_mid*dtwpe_particle) +
    #               nhigh_esc_t * (acc_rate_esc_mid2*dtwpe_particle)) * (-pindex)
    # esc_rate = div0(np.gradient(nhigh_esc_t, dtwpe_particle) -
    #                 dnhigh_esc/(2*dtwpe_particle), nhigh_acc_t)
    esc_rate = div0(np.gradient(nhigh_esc_t, dtwpe_particle), nhigh_acc_t)
    power_index = 1 + div0(esc_rate, acc_rate)

    # acceleration and escape rate
    fig = plt.figure(figsize=[7, 5])
    rect = [0.14, 0.12, 0.82, 0.84]
    ax = fig.add_axes(rect)
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    pstr = "%0.2f" % (-pindex-1)
    label = r"Acceleration rate$\times$" + r"$" + pstr + r"$"
    # ax.plot(tparticles, acc_rate*(-pindex-1), linewidth=2, label=label)
    # ax.plot(tparticles, esc_rate, linewidth=2, label="Escape rate")
    # ax.set_xlim([tmin, tmax])
    ax.scatter(acc_rate, esc_rate)
    ax.set_ylim(np.asarray([-alpha_min, 4*alpha_min]))
    ax.tick_params(bottom=True, top=True, left=True, right=False)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.legend(loc='best', prop={'size': 12}, ncol=1,
             shadow=False, fancybox=False, frameon=False)
    ax.set_xlabel(r'$t\omega_{pe}$', fontsize=16)
    ax.set_ylabel(r'Rates$/\omega_{pe}$', fontsize=16)
    ax.tick_params(labelsize=12)
    fdir = '../img/power_law_index/power_index/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + 'acc_esc_rates_' + species + '.pdf'
    fig.savefig(fname)

    # estimated power-law index
    fig = plt.figure(figsize=[7, 5])
    rect = [0.10, 0.12, 0.86, 0.84]
    ax = fig.add_axes(rect)
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    ax.set_prop_cycle('color', COLORS)
    ax.plot(tparticles, power_index, marker='o', linewidth=2)
    ax.plot([tmin, tmax], [1, 1], color='k', linewidth=1, linestyle='--')
    ax.plot([tmin, tmax], [-pindex, -pindex], color='k',
            linewidth=1, linestyle='--')
    ypos = -pindex / 6
    ax.text(0.05, ypos, str(-pindex), color='k', fontsize=16,
            bbox=dict(facecolor='none', alpha=1.0,
                      edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes)
    ax.set_xlim([tmin, tmax])
    ax.set_ylim([0, 6])
    ax.tick_params(bottom=True, top=True, left=True, right=False)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlabel(r'$t\omega_{pe}$', fontsize=16)
    text1 = r'$1+(\alpha\tau_\text{esc})^{-1}$'
    ax.set_ylabel(text1, fontsize=16)
    ax.tick_params(labelsize=12)
    fdir = '../img/power_law_index/power_index/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + 'power_index_' + species + '.pdf'
    fig.savefig(fname)

    # particle number changing rate
    fig = plt.figure(figsize=[7, 5])
    rect = [0.10, 0.12, 0.86, 0.84]
    ax = fig.add_axes(rect)
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    ax.set_prop_cycle('color', COLORS)
    dn_acc = np.gradient(nhigh_acc_t, dtwpe_particle)
    dn_esc = np.gradient(nhigh_esc_t, dtwpe_particle)
    ax.plot(tparticles, dn_acc, linewidth=2, label=r'$dN_\text{acc}/dt$')
    ax.plot(tparticles, dn_esc, linewidth=2, label=r'$dN_\text{esc}/dt$')
    ax.plot([tparticles[0], tparticles[-1]], [0, 0], color='k',
            linewidth=1, linestyle='--')
    ax.set_xlim([tmin, tmax])
    ax.tick_params(bottom=True, top=True, left=True, right=False)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.legend(loc=1, prop={'size': 12}, ncol=1,
             shadow=False, fancybox=False, frameon=False)
    ax.set_xlabel(r'$t\omega_{pe}$', fontsize=16)
    ax.set_ylabel(r'$dN/dt$', fontsize=16)
    ax.tick_params(labelsize=12)
    fdir = '../img/power_law_index/power_index/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + 'dn_dt_' + species + '.pdf'
    fig.savefig(fname)

    # Compare total rate with rate due to curvature drift
    fig = plt.figure(figsize=[7, 5])
    rect = [0.10, 0.12, 0.86, 0.84]
    ax = fig.add_axes(rect)
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    ax.set_prop_cycle('color', COLORS)
    ax.plot(tparticles, div0(acc_rate, acc_rate1), linewidth=2,
            label='Total/Curvature')
    ax.set_xlim([tmin, tmax])
    ax.set_ylim([-1, 5])
    ax.grid(True)
    ax.tick_params(bottom=True, top=True, left=True, right=False)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.legend(loc=1, prop={'size': 12}, ncol=1,
             shadow=False, fancybox=False, frameon=False)
    ax.set_xlabel(r'$t\omega_{pe}$', fontsize=16)
    ax.set_ylabel('Total/Curvature', fontsize=16)
    ax.tick_params(labelsize=12)
    fdir = '../img/power_law_index/power_index/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + 'acc_rates_compare_' + species + '.pdf'
    fig.savefig(fname)

    plt.show()
    # plt.close('all')


def vkappa_escape_boundary(plot_config, show_plot=True):
    """Check the distribution of vkappa to find escape boundary

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    tframe = plot_config["tframe"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    alpha_min = plot_config["vkappa_threshold"]
    mime = pic_info.mime
    ntp = int(pic_info.ntp)
    spect_params = get_spect_params(pic_run)
    pindex = spect_params["power_index"]
    emin, emax = spect_params["energy_range"]
    n0 = pic_info.n0
    gamma_e = 4.0/3 if pic_info.Te > 0.1 else 1.5  # Adiabatic index
    gamma_i = 5.0/3
    # In typical VPIC simulation, me=1, c=1
    enthalpy_e =  n0 * (1 + pic_info.Te * gamma_e / (gamma_e - 1))
    enthalpy_i =  n0 * (mime + pic_info.Ti * gamma_i / (gamma_i - 1))
    enthalpy = enthalpy_e + enthalpy_i
    sigma = pic_info.b0**2 / enthalpy
    sigma_c = pic_info.b0**2 / (n0 * (1 + mime))
    sigmae_c = pic_info.b0**2 / n0
    sigmai_c = pic_info.b0**2 / (n0 * mime)
    sigmae_h = pic_info.b0**2 / enthalpy_e
    sigmai_h = pic_info.b0**2 / enthalpy_i
    beta_e = 2 * n0 * pic_info.Te / pic_info.b0**2

    if species in ["e", "electron"]:
        sname = "electron"
    else:
        sname = "Ion"
    temp = pic_info.Te if species in ['e', 'electron'] else pic_info.Ti

    va = math.sqrt(sigma / (sigma + 1))

    fpath = "../data/particle_interp/" + pic_run + "/"
    tindex = pic_info.particle_interval
    fname = fpath + "acc_rate_dist_" + species + "_" + str(tindex) + ".gda"
    fdata = np.fromfile(fname, dtype=np.float32)
    nvar = int(fdata[2])   # number of variables

    ts, te = 1, 50
    ntests = te - ts + 1
    ndi_maxs = np.linspace(ts, te, ntests)
    nhigh_acc_t = np.zeros([ntp, ntests])
    nhigh_esc_t = np.zeros([ntp, ntests])
    arate_acc_t = np.zeros([nvar, ntp, ntests])
    arate_esc_t = np.zeros([nvar, ntp, ntests])

    for tframe in range(1, ntp):
        print("Time frame: %d" % tframe)
        tindex = tframe * pic_info.particle_interval
        fname = fpath + "acc_rate_dist_" + species + "_" + str(tindex) + ".gda"
        fdata = np.fromfile(fname, dtype=np.float32)
        nalpha = int(fdata[0]) # number of bins of the rates
        nbins = int(fdata[1])  # number of bins in energy
        nvar = int(fdata[2])   # number of variables
        ebins = fdata[3:nbins+3]  # energy bins
        ebins /= temp  # normalize by initial temperature
        alpha_bins = fdata[nbins+3:nbins+nalpha+3]  # acceleration rates bins
        alpha_bins_mid = 0.5 * (alpha_bins[1:] + alpha_bins[:-1])
        # variables: 0-Epara, 1-Eperp, 2-compression, 3-shear, 4-curvature,
        #   5-gradient, 6-para_drift, 7-mu, 8-polar_time, 9-polar_spatial,
        #   10-inertial_time, 11-inertial_spatial, 12-polar_fluid_time,
        #   13-polar_fluid_spatial, 14-polar_time_v, 15-polar_spatial_v,
        #   16-ptensor
        fdist = fdata[nbins+nalpha+3:].reshape((nvar, nbins, (nalpha+1)*4))

        es, _ = find_nearest(ebins, emin)
        ee, _ = find_nearest(ebins, emax)
        for itest, ndi_max in enumerate(ndi_maxs):
            alpha_min = va/math.sqrt(mime)/ndi_max  # vA/(ndi_max * di)
            alpha_s, _ = find_nearest(alpha_bins, alpha_min)
            fdist_high = np.sum(fdist[:, es:ee, :], axis=1)
            nalpha0 = nalpha + 1
            nptl_bins = fdist_high[:, :nalpha0*2]
            dene_bins = fdist_high[:, nalpha0*2:]
            fnptl = nptl_bins[4]
            fdene = dene_bins[4]
            pos_range = np.arange(nalpha0+1, nalpha0*2-1)
            neg_range = np.arange(nalpha0-2, 0, -1)
            fnptl_pos = fnptl[pos_range]
            fnptl_neg = fnptl[neg_range]
            fdene_pos = fdene[pos_range]
            fdene_neg = fdene[neg_range]
            nhigh_acc_t[tframe, itest] = (np.sum(fnptl_pos[alpha_s:] +
                                                 fnptl_neg[alpha_s:]))
            nhigh_esc_t[tframe, itest] = (np.sum(fnptl_pos + fnptl_neg) -
                                          nhigh_acc_t[tframe, itest])

            fnptl_var_pos = nptl_bins[:, pos_range]
            fnptl_var_neg = nptl_bins[:, neg_range]
            fdene_var_pos = dene_bins[:, pos_range]
            fdene_var_neg = dene_bins[:, neg_range]

            arate_acc_t[:, tframe, itest] = np.sum(fdene_var_pos[:, alpha_s:] +
                                                   fdene_var_neg[:, alpha_s:], axis=1)
            arate_esc_t[:, tframe, itest] = (np.sum(fdene_var_pos + fdene_var_neg, axis=1) -
                                             arate_acc_t[:, tframe, itest])

    print("Particle initial temperature (Lorentz factor): %f" % temp)
    print("Alfven speed: %f" % va)
    print("alpha_min va/(%0.1f * di): %f" % (ndi_max, alpha_min))
    print("sigma: %f" % sigma)
    print("sigma_c: %f" % sigma_c)
    print("sigmae_c: %f" % sigmae_c)
    print("sigmae_h: %f" % sigmae_h)
    print("sigmai_c: %f" % sigmai_c)
    print("sigmai_h: %f" % sigmai_h)

    dtwpe_particle = pic_info.particle_interval * pic_info.dtwpe
    tparticles = np.arange(0, ntp) * dtwpe_particle
    tmin, tmax = tparticles[0], tparticles[-1]
    esc_rate = div0(np.gradient(nhigh_esc_t, dtwpe_particle, axis=0),
                    nhigh_acc_t)
    acc_rate = div0(arate_acc_t[4], nhigh_acc_t)
    acc_rate1 = div0(arate_acc_t[0] + arate_acc_t[1], nhigh_acc_t)
    acc_rate2 = div0(arate_esc_t[0] + arate_esc_t[1], nhigh_esc_t)
    power_index = 1 + div0(esc_rate, acc_rate)

    # acceleration and escape rate
    fdir = '../img/power_law_index/acc_rates/' + pic_run + '/'
    mkdir_p(fdir)
    fig = plt.figure(figsize=[7, 5])
    rect = [0.14, 0.12, 0.82, 0.84]
    ax = fig.add_axes(rect)
    pindex_pred = 1 + div0(esc_rate, acc_rate1).T
    pdiff = pindex_pred + pindex
    # ax.plot(div0(esc_rate[:, -1], acc_rate1[:, -1]))
    p1 = ax.imshow(pdiff, vmin=-2, vmax=2,
                   cmap=plt.cm.coolwarm, aspect='auto',
                   origin='lower', interpolation='bicubic')
    levels = [0]
    tframes = np.linspace(0, ntp-1, ntp)
    dis = np.linspace(0, ntests-1, ntests)
    cs = ax.contour(tframes, dis, pdiff, colors='k',
                    linewidths=1, levels=levels)

    plt.show()


def rates_based_vkappa(plot_config, show_plot=True):
    """calculate particle acceleration and escape rates

    This is based on rates bins with vkappa, either particle-based or grid-based

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    tframe = plot_config["tframe"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    alpha_min = plot_config["vkappa_threshold"]
    mime = pic_info.mime
    ntp = int(pic_info.ntp)
    spect_params = get_spect_params(pic_run)
    pindex = spect_params["power_index"]
    emin, emax = spect_params["energy_range"]
    n0 = pic_info.n0
    gamma_e = 4.0/3 if pic_info.Te > 0.1 else 1.5  # Adiabatic index
    gamma_i = 5.0/3
    # In typical VPIC simulation, me=1, c=1
    enthalpy_e =  n0 * (1 + pic_info.Te * gamma_e / (gamma_e - 1))
    enthalpy_i =  n0 * (mime + pic_info.Ti * gamma_i / (gamma_i - 1))
    enthalpy = enthalpy_e + enthalpy_i
    sigma = pic_info.b0**2 / enthalpy
    sigma_c = pic_info.b0**2 / (n0 * (1 + mime))
    sigmae_c = pic_info.b0**2 / n0
    sigmai_c = pic_info.b0**2 / (n0 * mime)
    sigmae_h = pic_info.b0**2 / enthalpy_e
    sigmai_h = pic_info.b0**2 / enthalpy_i
    beta_e = 2 * n0 * pic_info.Te / pic_info.b0**2

    # Get guide field strength from PIC run name
    res = list(filter(lambda x: 'bg' in x, pic_run.split('_')))[0]
    bg_str = res[2] + '.' + res[3:]
    bg = float(bg_str)

    if species in ["e", "electron"]:
        sname = "electron"
    else:
        sname = "Ion"
    temp = pic_info.Te if species in ['e', 'electron'] else pic_info.Ti

    va = math.sqrt(sigma / (sigma + 1))
    if int(mime) == 1:
        ndi_max = 60.0
    elif int(mime) == 25:
        ndi_max = 40.0
    elif mime > 1000:
        ndi_max = 10.0
    else:
        ndi_max = 10.0
    # ndi_max = 10 * math.sqrt(1+pic_info.Ti/pic_info.mime)
    alpha_min = va/math.sqrt(mime)/ndi_max  # vA/(ndi_max * di)
    # alpha_min /= bg**2 + 1.0  # consider guide field
    # alpha_min = 0.15

    fpath = "../data/particle_interp/" + pic_run + "/"
    tindex = pic_info.particle_interval
    fname = fpath + "acc_rate_dist_" + species + "_" + str(tindex) + ".gda"
    fdata = np.fromfile(fname, dtype=np.float64)
    nvar = int(fdata[2])   # number of variables
    # nvar = int(fdata[3])   # number of variables

    nhigh_acc_t = np.zeros(ntp)
    nhigh_esc_t = np.zeros(ntp)
    ntot_acc_t = np.zeros(ntp)
    ntot_esc_t = np.zeros(ntp)
    arate_acc_t = np.zeros([nvar, ntp])
    arate_esc_t = np.zeros([nvar, ntp])
    arate2_acc_t = np.zeros([nvar, ntp])
    arate2_esc_t = np.zeros([nvar, ntp])

    # vkappas = np.zeros(pic_info.ntf)
    # fdir = '../data/power_law_index/vexb_kappa_dist/' + pic_run + '/'
    # for tframe in range(pic_info.ntf):
    #     print("Time frame: %d" % tframe)
    #     fname = fdir + 'vexb_kappa_dist_' + str(tframe) + '.dat'
    #     if not os.path.isfile(fname):
    #         break
    #     fdata = np.fromfile(fname)
    #     vkmin = fdata[0]
    #     vkmax = fdata[1]
    #     nbins = fdata[2]
    #     vkdist = fdata[3:]
    #     nbins = 80
    #     vkbins = np.zeros(2*nbins+3)
    #     fbins = np.logspace(math.log10(vkmin), math.log10(vkmax), nbins+1)
    #     vkbins[:nbins+1] = -fbins[::-1]
    #     vkbins[nbins+2:]= fbins
    #     vkbins_mid = 0.5 * (vkbins[1:] + vkbins[:-1])
    #     vkdist = div0(vkdist, np.diff(vkbins))
    #     color = plt.cm.jet(tframe/float(pic_info.ntf), 1)
    #     dratio = vkdist[nbins+1:] / np.flip(vkdist[:nbins+1])
    #     vkappas[tframe] = vkbins_mid[nbins + 1 + np.argmax(dratio > 1.2)]

    for tframe in range(1, ntp):
        print("Time frame: %d" % tframe)
        tindex = tframe * pic_info.particle_interval
        # fname = fpath + "acc_rate_dist_vfluid_dote_" + species + "_" + str(tindex) + ".gda"
        # fname = fpath + "acc_rate_dist_vkappa_" + species + "_" + str(tindex) + ".gda"
        fname = fpath + "acc_rate_dist_vkappa_grid_" + species + "_" + str(tindex) + ".gda"
        fdata = np.fromfile(fname, dtype=np.float64)
        nalpha = int(fdata[0]) # number of bins of the rates
        nbins = int(fdata[1])  # number of bins along x
        nvar = int(fdata[2])   # number of variables
        ebins = fdata[3:nbins+3]  # energy bins
        ebins /= temp  # normalize by initial temperature
        alpha_bins = fdata[nbins+3:nbins+nalpha+3]  # acceleration rates bins
        alpha_bins_mid = 0.5 * (alpha_bins[1:] + alpha_bins[:-1])
        # variables: 0-Epara, 1-Eperp, 2-compression, 3-shear, 4-curvature,
        #   5-gradient, 6-para_drift, 7-mu, 8-polar_time, 9-polar_spatial,
        #   10-inertial_time, 11-inertial_spatial, 12-polar_fluid_time,
        #   13-polar_fluid_spatial, 14-polar_time_v, 15-polar_spatial_v,
        #   16-ptensor
        fdist = fdata[nbins+nalpha+3:].reshape((nvar+1, nbins, (nalpha+1)*4))

        es, _ = find_nearest(ebins, emin)
        ee, _ = find_nearest(ebins, emax)
        ethe, _ = find_nearest(ebins, 1)
        # alpha_min = vkappas[tframe * pic_info.particle_interval//pic_info.fields_interval]
        alpha_s, _ = find_nearest(alpha_bins, alpha_min)
        fdist_high = np.sum(fdist[:, es:ee, :], axis=1)
        fdist_tot = np.sum(fdist[:, ethe+10:, :], axis=1)
        nalpha0 = nalpha + 1
        nptl_bins = fdist_high[0, :nalpha0*2]
        dene_bins = fdist_high[1:, :nalpha0*2]
        dene2_bins = fdist_high[1:, nalpha0*2:]
        fnptl = np.copy(nptl_bins)
        fntot = np.copy(fdist_tot[0, :nalpha0*2])
        fdene = dene_bins[4]
        pos_range = np.arange(nalpha0+1, nalpha0*2-1)
        neg_range = np.arange(nalpha0-2, 0, -1)
        fnptl_pos = fnptl[pos_range]
        fnptl_neg = fnptl[neg_range]
        fntot_pos = fntot[pos_range]
        fntot_neg = fntot[neg_range]
        fdene_pos = fdene[pos_range]
        fdene_neg = fdene[neg_range]
        nhigh_acc_t[tframe] = np.sum(fnptl_pos[alpha_s:] + fnptl_neg[alpha_s:])
        nhigh_esc_t[tframe] = np.sum(fnptl_pos + fnptl_neg) - nhigh_acc_t[tframe]
        ntot_acc_t[tframe] = np.sum(fntot_pos[alpha_s:] + fntot_neg[alpha_s:])
        ntot_esc_t[tframe] = np.sum(fntot_pos + fntot_neg) - ntot_acc_t[tframe]

        fdene_var_pos = dene_bins[:, pos_range]
        fdene_var_neg = dene_bins[:, neg_range]
        fdene2_var_pos = dene2_bins[:, pos_range]
        fdene2_var_neg = dene2_bins[:, neg_range]

        arate_acc_t[:, tframe] = np.sum(fdene_var_pos[:, alpha_s:] +
                                        fdene_var_neg[:, alpha_s:], axis=1)
        arate_esc_t[:, tframe] = (np.sum(fdene_var_pos + fdene_var_neg, axis=1) -
                                  arate_acc_t[:, tframe])
        arate_acc_t[:, tframe] = div0(arate_acc_t[:, tframe], nhigh_acc_t[tframe])
        arate_esc_t[:, tframe] = div0(arate_esc_t[:, tframe], nhigh_esc_t[tframe])

        arate2_acc_t[:, tframe] = np.sum(fdene2_var_pos[:, alpha_s:] +
                                         fdene2_var_neg[:, alpha_s:], axis=1)
        arate2_esc_t[:, tframe] = (np.sum(fdene2_var_pos + fdene2_var_neg, axis=1) -
                                   arate2_acc_t[:, tframe])
        arate2_acc_t[:, tframe] = (div0(arate2_acc_t[:, tframe], nhigh_acc_t[tframe]) -
                                   arate_acc_t[:, tframe]**2)
        arate2_esc_t[:, tframe] = (div0(arate2_esc_t[:, tframe], nhigh_esc_t[tframe]) -
                                   arate_esc_t[:, tframe]**2)


    print("Particle initial temperature (Lorentz factor): %f" % temp)
    print("Alfven speed: %f" % va)
    print("alpha_min va/(%0.1f * di): %f" % (ndi_max, alpha_min))
    print("sigma: %f" % sigma)
    print("sigma_c: %f" % sigma_c)
    print("sigmae_c: %f" % sigmae_c)
    print("sigmae_h: %f" % sigmae_h)
    print("sigmai_c: %f" % sigmai_c)
    print("sigmai_h: %f" % sigmai_h)

    dtwpe_particle = pic_info.particle_interval * pic_info.dtwpe
    tparticles = np.arange(0, ntp) * dtwpe_particle
    tmin, tmax = tparticles[0], tparticles[-1]
    nhigh_tot = nhigh_acc_t + nhigh_esc_t
    dndt_inj = np.gradient(nhigh_tot, dtwpe_particle)
    dndt_esc = np.gradient(nhigh_esc_t, 0.5)
    dndt_tot = np.gradient(nhigh_tot, 0.5)
    npre = np.zeros(nhigh_acc_t.shape)
    npre[0] = nhigh_acc_t[0]
    npre[1:] = nhigh_acc_t[:-1]
    npre_esc = np.zeros(nhigh_acc_t.shape)
    npre_esc[0] = nhigh_esc_t[0]
    npre_esc[1:] = nhigh_esc_t[:-1]
    # btmp = npre + dndt_inj * dtwpe_particle
    # esc_rate = div0(btmp - np.sqrt(btmp**2 - 2 * npre * dndt_esc),
    #                 2 * npre * dtwpe_particle)
    # esc_rate = div0(np.gradient(nhigh_esc_t, dtwpe_particle),
    #                 npre + dndt_tot * 0.2)
    # esc_rate = (npre + dndt_inj * dtwpe_particle - nhigh_acc_t) / (npre * dtwpe_particle)
    # atmp = npre * 0.5
    # btmp = -(npre + 0.5 * dndt_inj * dtwpe_particle)
    # ctmp = npre + dndt_inj * dtwpe_particle - nhigh_acc_t
    # esc_rate = div0(-btmp - np.sqrt(btmp**2 - 4 * atmp * ctmp), npre) / (dtwpe_particle)
    acc_rate = arate_acc_t[0, :] + arate_acc_t[1, :]
    acc_rate1 = arate_acc_t[4, :]
    acc_rate_esc = arate_esc_t[0, :] + arate_esc_t[1, :]
    acc_rate_esc_mid = np.copy(acc_rate_esc)
    acc_rate_esc_mid[1:] = 0.5 * (acc_rate_esc[1:] + acc_rate_esc[:-1])
    acc_rate_esc_mid2 = np.copy(acc_rate_esc)
    acc_rate_esc_mid2[1:-1] = 0.5 * (acc_rate_esc[1:-1] + acc_rate_esc[2:])
    dnhigh_esc = (npre_esc * (acc_rate_esc_mid*dtwpe_particle) +
                  nhigh_esc_t * (acc_rate_esc_mid2*dtwpe_particle)) * (-pindex)
    # esc_rate = div0(np.gradient(nhigh_esc_t, dtwpe_particle) -
    #                 dnhigh_esc/(2*dtwpe_particle), nhigh_acc_t)
    esc_rate = div0(np.gradient(nhigh_esc_t, dtwpe_particle), nhigh_acc_t)
    # esc_rate = div0(np.gradient(ntot_esc_t, dtwpe_particle), ntot_acc_t)
    power_index = 1 + div0(esc_rate, acc_rate)
    # power_index = 1.5 * (1 + 16 * div0(esc_rate, acc_rate) / 9)**0.5 - 0.5
    # d0 = (acc_rate + esc_rate - acc_rate * abs(pindex) / (pindex**2 - abs(pindex))
    d0 = (acc_rate + esc_rate - acc_rate * abs(pindex)) / (pindex**2 - 3*abs(pindex) + 2)

    # acceleration and escape rate
    arate_test = np.fromfile("../data/test.dat")
    fig = plt.figure(figsize=[7, 5])
    rect = [0.14, 0.12, 0.82, 0.84]
    ax = fig.add_axes(rect)
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    pstr = "%0.2f" % (-pindex-1)
    label = r"Acceleration rate$\times$" + r"$" + pstr + r"$"
    ax.plot(tparticles, acc_rate*(-pindex-1), linewidth=2, label=label)
    # ax.plot(tparticles, arate_test*(-pindex-1), linewidth=2)
    ax.plot(tparticles, esc_rate, linewidth=2, label="Escape rate")
    ax.set_xlim([tmin, tmax])
    # ax.scatter(acc_rate, esc_rate)
    ax.set_ylim(np.asarray([-alpha_min, 4*alpha_min]))
    ax.tick_params(bottom=True, top=True, left=True, right=False)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.legend(loc='best', prop={'size': 12}, ncol=1,
             shadow=False, fancybox=False, frameon=False)
    ax.set_xlabel(r'$t\omega_{pe}$', fontsize=16)
    ax.set_ylabel(r'Rates$/\omega_{pe}$', fontsize=16)
    ax.tick_params(labelsize=12)
    fdir = '../img/power_law_index/power_index/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + 'acc_esc_rates_' + species + '.pdf'
    fig.savefig(fname)

    # estimated power-law index
    fig = plt.figure(figsize=[7, 5])
    rect = [0.10, 0.12, 0.86, 0.84]
    ax = fig.add_axes(rect)
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    ax.set_prop_cycle('color', COLORS)
    ax.plot(tparticles, power_index, marker='o', linewidth=2)
    ax.plot([tmin, tmax], [1, 1], color='k', linewidth=1, linestyle='--')
    ax.plot([tmin, tmax], [-pindex, -pindex], color='k',
            linewidth=1, linestyle='--')
    ypos = -pindex / 6
    ax.text(0.05, ypos, str(-pindex), color='k', fontsize=16,
            bbox=dict(facecolor='none', alpha=1.0,
                      edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes)
    ax.set_xlim([tmin, tmax])
    ax.set_ylim([0, 6])
    ax.tick_params(bottom=True, top=True, left=True, right=False)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlabel(r'$t\omega_{pe}$', fontsize=16)
    text1 = r'$1+(\alpha\tau_\text{esc})^{-1}$'
    ax.set_ylabel(text1, fontsize=16)
    ax.tick_params(labelsize=12)
    fdir = '../img/power_law_index/power_index/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + 'power_index_' + species + '.pdf'
    fig.savefig(fname)

    # # particle number changing rate
    # fig = plt.figure(figsize=[7, 5])
    # rect = [0.10, 0.12, 0.86, 0.84]
    # ax = fig.add_axes(rect)
    # COLORS = palettable.tableau.Tableau_10.mpl_colors
    # ax.set_prop_cycle('color', COLORS)
    # dn_acc = np.gradient(nhigh_acc_t, dtwpe_particle)
    # dn_esc = np.gradient(nhigh_esc_t, dtwpe_particle)
    # ax.plot(tparticles, dn_acc, linewidth=2, label=r'$dN_\text{acc}/dt$')
    # ax.plot(tparticles, dn_esc, linewidth=2, label=r'$dN_\text{esc}/dt$')
    # ax.plot([tparticles[0], tparticles[-1]], [0, 0], color='k',
    #         linewidth=1, linestyle='--')
    # ax.set_xlim([tmin, tmax])
    # ax.tick_params(bottom=True, top=True, left=True, right=False)
    # ax.tick_params(axis='x', which='minor', direction='in')
    # ax.tick_params(axis='x', which='major', direction='in')
    # ax.tick_params(axis='y', which='minor', direction='in')
    # ax.tick_params(axis='y', which='major', direction='in')
    # ax.legend(loc=1, prop={'size': 12}, ncol=1,
    #          shadow=False, fancybox=False, frameon=False)
    # ax.set_xlabel(r'$t\omega_{pe}$', fontsize=16)
    # ax.set_ylabel(r'$dN/dt$', fontsize=16)
    # ax.tick_params(labelsize=12)
    # fdir = '../img/power_law_index/power_index/' + pic_run + '/'
    # mkdir_p(fdir)
    # fname = fdir + 'dn_dt_' + species + '.pdf'
    # fig.savefig(fname)

    # # Compare total rate with rate due to curvature drift
    # fig = plt.figure(figsize=[7, 5])
    # rect = [0.10, 0.12, 0.86, 0.84]
    # ax = fig.add_axes(rect)
    # COLORS = palettable.tableau.Tableau_10.mpl_colors
    # ax.set_prop_cycle('color', COLORS)
    # ax.plot(tparticles, div0(acc_rate, acc_rate1), linewidth=2,
    #         label='Total/Curvature')
    # ax.set_xlim([tmin, tmax])
    # ax.set_ylim([-1, 5])
    # ax.grid(True)
    # ax.tick_params(bottom=True, top=True, left=True, right=False)
    # ax.tick_params(axis='x', which='minor', direction='in')
    # ax.tick_params(axis='x', which='major', direction='in')
    # ax.tick_params(axis='y', which='minor', direction='in')
    # ax.tick_params(axis='y', which='major', direction='in')
    # ax.legend(loc=1, prop={'size': 12}, ncol=1,
    #          shadow=False, fancybox=False, frameon=False)
    # ax.set_xlabel(r'$t\omega_{pe}$', fontsize=16)
    # ax.set_ylabel('Total/Curvature', fontsize=16)
    # ax.tick_params(labelsize=12)
    # fdir = '../img/power_law_index/power_index/' + pic_run + '/'
    # mkdir_p(fdir)
    # fname = fdir + 'acc_rates_compare_' + species + '.pdf'
    # fig.savefig(fname)

    plt.show()
    # plt.close('all')


def calc_acc_rate(plot_config, show_plot=True):
    """calculate particle acceleration rates

    This is based on rates bins with vkappa, either particle-based or grid-based

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    tframe = plot_config["tframe"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    alpha_min = plot_config["vkappa_threshold"]
    mime = pic_info.mime
    ntp = int(pic_info.ntp)
    spect_params = get_spect_params(pic_run)
    pindex = spect_params["power_index"]
    emin, emax = spect_params["energy_range"]
    n0 = pic_info.n0
    gamma_e = 4.0/3 if pic_info.Te > 0.1 else 1.5  # Adiabatic index
    gamma_i = 5.0/3
    # In typical VPIC simulation, me=1, c=1
    enthalpy_e =  n0 * (1 + pic_info.Te * gamma_e / (gamma_e - 1))
    enthalpy_i =  n0 * (mime + pic_info.Ti * gamma_i / (gamma_i - 1))
    enthalpy = enthalpy_e + enthalpy_i
    sigma = pic_info.b0**2 / enthalpy
    sigma_c = pic_info.b0**2 / (n0 * (1 + mime))
    sigmae_c = pic_info.b0**2 / n0
    sigmai_c = pic_info.b0**2 / (n0 * mime)
    sigmae_h = pic_info.b0**2 / enthalpy_e
    sigmai_h = pic_info.b0**2 / enthalpy_i
    beta_e = 2 * n0 * pic_info.Te / pic_info.b0**2

    # Get guide field strength from PIC run name
    res = list(filter(lambda x: 'bg' in x, pic_run.split('_')))[0]
    bg_str = res[2] + '.' + res[3:]
    bg = float(bg_str)

    if species in ["e", "electron"]:
        sname = "electron"
    else:
        sname = "Ion"
    temp = pic_info.Te if species in ['e', 'electron'] else pic_info.Ti

    va = math.sqrt(sigma / (sigma + 1))
    if int(mime) == 1:
        ndi_max = 60.0
    elif int(mime) == 25:
        ndi_max = 40.0
    elif mime > 1000:
        ndi_max = 10.0
    else:
        ndi_max = 10.0
    alpha_min = va/math.sqrt(mime)/ndi_max  # vA/(ndi_max * di)

    fpath = "../data/particle_interp/" + pic_run + "/"
    tindex = pic_info.particle_interval
    fname = fpath + "acc_rate_dist_vkappa_grid_" + species + "_" + str(tindex) + ".gda"
    fdata = np.fromfile(fname, dtype=np.float64)
    nalpha = int(fdata[0]) # number of bins of the rates
    nbins = int(fdata[1])  # number of bins along x
    nvar = int(fdata[2])   # number of variables
    ebins = fdata[3:nbins+3]  # energy bins
    ebins /= temp  # normalize by initial temperature
    alpha_bins = fdata[nbins+3:nbins+nalpha+3]  # acceleration rates bins
    alpha_bins_mid = 0.5 * (alpha_bins[1:] + alpha_bins[:-1])
    es, emin1 = find_nearest(ebins, emin)
    ee, emax1 = find_nearest(ebins, emax)
    alpha_s, alpha = find_nearest(alpha_bins, alpha_min)
    nalpha0 = nalpha + 1
    pos_range = np.arange(nalpha0+1, nalpha0*2-1)
    neg_range = np.arange(nalpha0-2, 0, -1)

    arate_acc_t = np.zeros(ntp)
    acc_rate_time = np.zeros([ntp, nbins-es])
    nhigh_acc_time = np.zeros([ntp, nbins-es])
    nhigh_esc_time = np.zeros([ntp, nbins-es])
    esc_rate_time = np.zeros([ntp, nbins-es])

    # for tframe in range(1, ntp):
    for tframe in range(25, 30):
        print("Time frame: %d" % tframe)
        tindex = tframe * pic_info.particle_interval
        fname = fpath + "acc_rate_dist_vkappa_" + species + "_" + str(tindex) + ".gda"
        # fname = fpath + "acc_rate_dist_vkappa_grid_" + species + "_" + str(tindex) + ".gda"
        fdata = np.fromfile(fname, dtype=np.float64)
        # variables: 0-Epara, 1-Eperp, 2-compression, 3-shear, 4-curvature,
        #   5-gradient, 6-para_drift, 7-mu, 8-polar_time, 9-polar_spatial,
        #   10-inertial_time, 11-inertial_spatial, 12-polar_fluid_time,
        #   13-polar_fluid_spatial, 14-polar_time_v, 15-polar_spatial_v,
        #   16-ptensor
        fdist = fdata[nbins+nalpha+3:].reshape((nvar+1, nbins, nalpha0*4))

        nptl_bins = fdist[0, :, :nalpha0*2]
        dene_bins = fdist[1:, :, :nalpha0*2]
        fdist_acc = fdist[:, :, alpha_s:]
        fnptl = np.copy(nptl_bins)
        fnptl_pos = fnptl[:, pos_range]
        fnptl_neg = fnptl[:, neg_range]
        fdene_pos = dene_bins[:, :, pos_range]
        fdene_neg = dene_bins[:, :, neg_range]
        nptl = np.sum(fnptl_pos[:, alpha_s:] + fnptl_neg[:, alpha_s:], axis=1)
        dene = np.sum(fdene_pos[:, :, alpha_s:] + fdene_neg[:, :, alpha_s:], axis=2)
        dene_tot = dene[0] + dene[1]
        # dene_tot = dene[4]
        acc_rate = div0(dene_tot, nptl)
        arate_single = np.sum(dene_tot[es:ee]) / np.sum(nptl[es:ee])
        arate_acc_t[tframe] = arate_single
        acc_rate_time[tframe, :] = acc_rate[es:]
        nhigh_acc_time[tframe, :] = nptl[es:]
        nhigh_esc_time[tframe, :] = np.sum(fnptl_pos[es:, :alpha_s] +
                                           fnptl_neg[es:, :alpha_s], axis=1)
        fig = plt.figure(figsize=[7, 5])
        rect = [0.14, 0.12, 0.8, 0.84]
        ax = fig.add_axes(rect)
        ax.semilogx(ebins[es:], acc_rate[es:])
        ax.semilogx([ebins[es], ebins[ee]], [arate_single, arate_single])
        ax1 = ax.twinx()
        ax1.loglog(ebins[es:], nptl[es:], linestyle='--')
        plt.show()

    dtwpe_particle = pic_info.particle_interval * pic_info.dtwpe
    esc_rate_time = div0(np.gradient(nhigh_esc_time, dtwpe_particle, axis=0), nhigh_acc_time)

    # tframe = 25
    # plt.semilogx(ebins[es:], acc_rate_time[tframe])
    # plt.semilogx(ebins[es:], esc_rate_time[tframe])
    # plt.scatter(acc_rate_time, esc_rate_time)
    # plt.xlim(np.asarray([-alpha_min, 4*alpha_min]))
    # plt.ylim(np.asarray([-alpha_min, 4*alpha_min]))
    # plt.show()

    arate_acc_t.tofile('../data/test.dat')

    # plt.show()


def plot_sigma_power(plot_config, show_plot=True):
    """Plot sigma parameters versus power-law index all runs

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    if species in ["e", "electron"]:
        sname = "electron"
    else:
        sname = "Ion"

    # root_dir = "/net/scratch4/xiaocanli/reconnection/power_law_index/"
    # pic_runs, labels = get_all_runs(root_dir)
    pic_runs = []
    bg_runs = []
    binplane = []
    sigmas = ["01", "04", "16", "64"]
    bgs = ["005", "01", "02", "04", "10"]
    bgs_float = [0.05, 0.1, 0.2, 0.4, 1.0]
    nsigma = len(sigmas)
    nbg = len(bgs)
    for sigma in sigmas:
        for ibg, bg in enumerate(bgs):
            pic_run = "sigma" + sigma + "_bg" + bg + "_4000de_triggered"
            picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
            pic_info = read_data_from_json(picinfo_fname)
            pic_runs.append(pic_run)
            bg_runs.append(bgs_float[ibg])
            binplane.append(pic_info.b0)
    nruns_1836 = len(pic_runs)
    sigmaes = ["100", "40", "10"]
    db2s = ["05", "1", "4"]
    bgs_float = [1/math.sqrt(0.5), 1.0, 0.5]
    nsigmae = len(sigmaes)
    ndb = len(db2s)
    for sigmae in sigmaes:
        for ibg, db2 in enumerate(db2s):
            pic_run = "mime1_sigmae" + sigmae + "_vthe04_db" + db2 + "_1024de"
            picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
            pic_info = read_data_from_json(picinfo_fname)
            pic_runs.append(pic_run)
            bg_runs.append(bgs_float[ibg])
            binplane.append(pic_info.b0 / bgs_float[ibg] / (2 * math.sqrt(2.0)))
    pic_run = "sigmae400_bg00_800de_triggered"
    pic_runs.append(pic_run)
    bg_runs.append(0.0)
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    binplane.append(pic_info.b0)
    nruns = len(pic_runs)

    runs, rune = 0, nruns
    sigma = np.zeros(rune - runs)
    sigma_c = np.zeros(rune - runs)
    sigmae_c = np.zeros(rune - runs)
    sigmai_c = np.zeros(rune - runs)
    sigmae_h = np.zeros(rune - runs)
    sigmai_h = np.zeros(rune - runs)
    beta_e = np.zeros(rune - runs)
    pindex = np.zeros(rune - runs)
    va4 = np.zeros(rune - runs)  # four Alfven speed
    pindex_pred = np.zeros(rune - runs)
    for irun, pic_run in enumerate(pic_runs[runs:rune]):
        print("PIC run name: %s" % pic_run)
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        pic_run_dir = pic_info.run_dir
        mime = pic_info.mime
        n0 = pic_info.n0
        gamma_e = 4.0/3 if pic_info.Te > 0.1 else 1.5  # Adiabatic index
        gamma_i = 5.0/3
        temp = pic_info.Te if species in ['e', 'electron'] else pic_info.Ti
        print("Particle initial temperature (Lorentz factor): %f" % temp)
        # In typical VPIC simulation, me=1, c=1
        enthalpy_e =  n0 * (1 + pic_info.Te * gamma_e / (gamma_e - 1))
        enthalpy_i =  n0 * (mime + pic_info.Ti * gamma_i / (gamma_i - 1))
        enthalpy = enthalpy_e + enthalpy_i
        b0 = binplane[irun]
        sigma[irun] = b0**2 / enthalpy
        sigma_c[irun] = b0**2 / (n0 * (1 + mime))
        sigmae_c[irun] = b0**2 / n0
        sigmai_c[irun] = b0**2 / (n0 * mime)
        sigmae_h[irun] = b0**2 / enthalpy_e
        sigmai_h[irun] = b0**2 / enthalpy_i
        beta_e[irun] = 2 * n0 * pic_info.Te / b0**2
        spect_params = get_spect_params(pic_run)
        pindex[irun] = -spect_params["power_index"]
        va = math.sqrt(sigma[irun] / (sigma[irun] + 1))
        va4[irun] = va / math.sqrt(1 - va**2)
        pindex_pred[irun] = 1 + 2*va / (va*0.5 + 2*va**2)
        gamma_the = pic_info.Te + 1
        vthe = math.sqrt(gamma_the**2 - 1) / gamma_the
        gamma_va = 1.0 / math.sqrt(1 - va**2)
        factor = gamma_va**2
        # factor = gamma_va**2 * va**2 / 3 + 1
        btot = math.sqrt(1 + bg_runs[irun]**2)
        ctheta = 1.0 / btot
        pindex_va = 1 + 8.0 * va / (factor * (1 + 2*va*ctheta + va**2) - 1)
        pindex_pred[irun] = pindex_va
        print("va, gamma_va, pindex_va: %0.2f, %0.2f, %0.2f" %
              (va, gamma_va, pindex_va))

    fdata = [sigma, sigma_c, sigmae_c, sigmae_h,
             sigmai_c, sigmai_h, beta_e, pindex_pred]
    labels = [r"$\sigma$", r"$\sigma_c$",
              r"$\sigma_{e, c}$", r"$\sigma_{e, h}$",
              r"$\sigma_{i, c}$", r"$\sigma_{i, h}$",
              r"$\beta_e$", r"$\gamma v_A$",
              r"$1 + 2v_A/(4v_A/3 + 2v_A^2)$"]
    npanels = len(fdata)
    fig = plt.figure(figsize=[12, 12])
    nrows, ncols = 4, 2
    rect0 = [0.07, 0.78, 0.42, 0.19]
    hgap, vgap = 0.05, 0.05
    axs = []
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    for ipanel in range(npanels):
        col = ipanel // nrows
        row = ipanel % nrows
        rect = np.copy(rect0)
        rect[0] = rect0[0] + col * (rect0[2] + hgap)
        rect[1] = rect0[1] - row * (rect0[3] + vgap)
        ax = fig.add_axes(rect)
        ax.set_prop_cycle('color', COLORS)
        axs.append(ax)
        COLORS = palettable.tableau.Tableau_10.mpl_colors
        ax.set_prop_cycle('color', COLORS)
        if ipanel == npanels - 1:
            ax.plot(fdata[ipanel][:nruns_1836], pindex[:nruns_1836],
                    linestyle='none', marker='o', markersize=10)
            ax.plot(fdata[ipanel][nruns_1836:], pindex[nruns_1836:],
                    linestyle='none', marker='o', markersize=10)
        else:
            ax.semilogx(fdata[ipanel][:nruns_1836], pindex[:nruns_1836],
                        linestyle='none', marker='o', markersize=10)
            ax.semilogx(fdata[ipanel][nruns_1836:], pindex[nruns_1836:],
                        linestyle='none', marker='o', markersize=10)
        ax.set_ylim([0, 5])
        ax.tick_params(bottom=True, top=True, left=True, right=False)
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')
        ax.set_xlabel(labels[ipanel], fontsize=16)
        ax.set_ylabel(r'$p$', fontsize=16)
        ax.tick_params(labelsize=12)
        rect[1] -= rect[3] + vgap

    axs[0].text(0.7, 0.95, r"$m_i/m_e=1836$",
                color=COLORS[0], fontsize=16,
                bbox=dict(facecolor='none', alpha=1.0,
                          edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='top',
                transform=axs[0].transAxes)
    axs[0].text(0.7, 0.85, r"$m_i/m_e=1$",
                color=COLORS[1], fontsize=16,
                bbox=dict(facecolor='none', alpha=1.0,
                          edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='top',
                transform=axs[0].transAxes)
    ename = 'electron' if species == 'e' else 'ion'
    # fdir = '../img/power_law_index/spectrum/' + pic_run + '/'
    # mkdir_p(fdir)
    # fname = fdir + 'spectrum_' + species + '.pdf'
    # fig.savefig(fname)
    plt.show()


def pindex_sde(plot_config, show_plot=True):
    """Calculate power-law index using SDE

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    if species in ["e", "electron"]:
        sname = "electron"
    else:
        sname = "Ion"

    pic_runs = []
    bg_runs = []
    binplane = []
    sigmas = ["01", "04", "16", "64"]
    bgs = ["005", "01", "02", "04", "10"]
    bgs_float = [0.05, 0.1, 0.2, 0.4, 1.0]
    nsigma = len(sigmas)
    nbg = len(bgs)
    # for sigma in sigmas:
    #     for ibg, bg in enumerate(bgs):
    #         pic_run = "sigma" + sigma + "_bg" + bg + "_4000de_triggered"
    #         picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    #         pic_info = read_data_from_json(picinfo_fname)
    #         pic_runs.append(pic_run)
    #         bg_runs.append(bgs_float[ibg])
    #         binplane.append(pic_info.b0)
    # nruns_1836 = len(pic_runs)
    sigmaes = ["100", "40", "10"]
    db2s = ["05", "1", "4"]
    bgs_float = [1/math.sqrt(0.5), 1.0, 0.5]
    nsigmae = len(sigmaes)
    ndb = len(db2s)
    for sigmae in sigmaes:
        for ibg, db2 in enumerate(db2s):
            pic_run = "mime1_sigmae" + sigmae + "_vthe04_db" + db2 + "_1024de"
            picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
            pic_info = read_data_from_json(picinfo_fname)
            print(pic_info.ene_bx[0], pic_info.ene_by[0], pic_info.ene_bz[0])
            pic_runs.append(pic_run)
            bg_runs.append(bgs_float[ibg])
            # binplane.append(pic_info.b0 / bgs_float[ibg] / (2 * math.sqrt(2.0)))
            binplane.append(pic_info.b0 / bgs_float[ibg])
    pic_run = "sigmae400_bg00_800de_triggered"
    pic_runs.append(pic_run)
    bg_runs.append(0.0)
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    binplane.append(pic_info.b0)
    nruns = len(pic_runs)

    runs, rune = 0, nruns
    sigma = np.zeros(rune - runs)
    sigma_c = np.zeros(rune - runs)
    sigmae_c = np.zeros(rune - runs)
    sigmai_c = np.zeros(rune - runs)
    sigmae_h = np.zeros(rune - runs)
    sigmai_h = np.zeros(rune - runs)
    beta_e = np.zeros(rune - runs)
    pindex = np.zeros(rune - runs)
    va4 = np.zeros(rune - runs)  # four Alfven speed
    ps = np.zeros((2, rune - runs))
    pindices = []
    for irun, pic_run in enumerate(pic_runs[runs:rune]):
        print("PIC run name: %s" % pic_run)
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        pic_run_dir = pic_info.run_dir
        mime = pic_info.mime
        n0 = pic_info.n0
        gamma_e = 4.0/3 if pic_info.Te > 0.1 else 1.5  # Adiabatic index
        gamma_i = 5.0/3
        temp = pic_info.Te if species in ['e', 'electron'] else pic_info.Ti
        print("Particle initial temperature (Lorentz factor): %f" % temp)
        # In typical VPIC simulation, me=1, c=1
        enthalpy_e =  n0 * (1 + pic_info.Te * gamma_e / (gamma_e - 1))
        enthalpy_i =  n0 * (mime + pic_info.Ti * gamma_i / (gamma_i - 1))
        enthalpy = enthalpy_e + enthalpy_i
        b0 = binplane[irun]
        br = b0 * 0.5
        sigma[irun] = b0**2 / enthalpy
        sigma_r = br**2 / enthalpy
        sigma_c[irun] = b0**2 / (n0 * (1 + mime))
        sigmae_c[irun] = b0**2 / n0
        sigmai_c[irun] = b0**2 / (n0 * mime)
        sigmae_h[irun] = b0**2 / enthalpy_e
        sigmai_h[irun] = b0**2 / enthalpy_i
        beta_e[irun] = 2 * n0 * pic_info.Te / b0**2
        spect_params = get_spect_params(pic_run)
        pindex[irun] = -spect_params["power_index"]
        va = math.sqrt(sigma_r / (sigma_r+sigma[irun]*bg_runs[irun]**2 + 1))
        va4[irun] = va / math.sqrt(1 - va**2)
        gamma_the = pic_info.Te + 1
        vthe = math.sqrt(gamma_the**2 - 1) / gamma_the
        gamma_va = 1.0 / math.sqrt(1 - va**2)
        factor = gamma_va**2
        btot = math.sqrt(1 + bg_runs[irun]**2)
        ctheta = 1.0 / btot
        lx_de = pic_info.lx_di * math.sqrt(pic_info.mime)
        # l = lx_de / 4
        l = math.sqrt(pic_info.mime) * 25
        acc_rate = (factor * (1 + va + va**2) - 1) / (4 * l)
        esc_rate = va / l
        dee_rate = acc_rate / 2
        nsteps = 10000
        if "mime1_sigmae1_" in pic_run:
            dee_rate = acc_rate / 2
            params= {"acc_rate": 0.001,
                     "esc_rate": 0.001 * esc_rate / acc_rate,
                     "dee_rate": 0.001 * dee_rate / acc_rate,
                     "nsteps": int(lx_de)*4,
                     "nptl": 1E5}
            p0,_,_ = solve_sde_pindex(params, pic_run, reconnection=False)
        else:
            dee_rate = acc_rate / 100
            params= {"acc_rate": 0.001,
                     "esc_rate": 0.001 * esc_rate / acc_rate,
                     "dee_rate": 0.001 * dee_rate / acc_rate,
                     "nsteps": int(lx_de)*4,
                     "nptl": 1E5}
            p0,_,_ = solve_sde_pindex(params, pic_run, reconnection=True)
        # pindices.append(p0)

    # plt.scatter(pindex, pindices)
    # plt.xlim([0, 5])
    # plt.ylim([0, 5])
    # plt.show()


def pindex_bg_sigma(plot_config, show_plot=True):
    """Compare power-law indices in model and simulations for all runs

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    if species in ["e", "electron"]:
        sname = "electron"
    else:
        sname = "Ion"
    reconnection_run = True
    turbulence_run = False

    pic_runs = []
    bg_runs = []
    binplane = []
    if reconnection_run:
        # sigmas = ["01", "04", "16", "64"]
        # bgs = ["005", "01", "02", "04", "10"]
        # bgs_float = [0.05, 0.1, 0.2, 0.4, 1.0]
        # bgs = ["02"]
        # bgs_float = [0.2]
        # bgs = ["10"]
        # bgs_float = [1.0]
        sigmas = ["001", "004", "016", "064", "256"]
        bgs = ["00", "05"]
        bgs_float = [0.0, 0.5]
        nsigma = len(sigmas)
        nbg = len(bgs)
        for sigma in sigmas:
            for ibg, bg in enumerate(bgs):
                # pic_run = "sigma" + sigma + "_bg" + bg + "_4000de_triggered"
                pic_run = "mime1836_sigmaic" + sigma + "_bg" + bg
                picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
                pic_info = read_data_from_json(picinfo_fname)
                pic_runs.append(pic_run)
                bg_runs.append(bgs_float[ibg])
                binplane.append(pic_info.b0)
        nruns_1836 = len(pic_runs)
    else:
        nsigma = 0
        nbg = 0

    if turbulence_run:
        sigmaes = ["100", "40", "10"]
        db2s = ["05", "1", "4"]
        bgs_float = [1/math.sqrt(0.5), 1.0, 0.5]
        nsigmae = len(sigmaes)
        ndb = len(db2s)
        for sigmae in sigmaes:
            for ibg, db2 in enumerate(db2s):
                pic_run = "mime1_sigmae" + sigmae + "_vthe04_db" + db2 + "_1024de"
                picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
                pic_info = read_data_from_json(picinfo_fname)
                print(pic_info.ene_bx[0], pic_info.ene_by[0], pic_info.ene_bz[0])
                pic_runs.append(pic_run)
                bg_runs.append(bgs_float[ibg])
                # binplane.append(pic_info.b0 / bgs_float[ibg] / (2 * math.sqrt(2.0)))
                binplane.append(pic_info.b0 / bgs_float[ibg])
    else:
        nsigmae = 0
        ndb = 0

    if reconnection_run:
        pic_run = "sigmae400_bg00_800de_triggered"
        pic_runs.append(pic_run)
        bg_runs.append(0.0)
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        binplane.append(pic_info.b0)
    nruns = len(pic_runs)

    runs, rune = 0, nruns
    sigma = np.zeros(rune - runs)
    sigma_c = np.zeros(rune - runs)
    sigmae_c = np.zeros(rune - runs)
    sigmai_c = np.zeros(rune - runs)
    sigmae_h = np.zeros(rune - runs)
    sigmai_h = np.zeros(rune - runs)
    beta_e = np.zeros(rune - runs)
    pindex = np.zeros(rune - runs)
    va4 = np.zeros(rune - runs)  # four Alfven speed
    ps = np.zeros((2, rune - runs))
    for irun, pic_run in enumerate(pic_runs[runs:rune]):
        print("-"*20)
        print("PIC run name: %s" % pic_run)
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        pic_run_dir = pic_info.run_dir
        mime = pic_info.mime
        n0 = pic_info.n0
        gamma_e = 4.0/3 if pic_info.Te > 0.1 else 1.5  # Adiabatic index
        gamma_i = 5.0/3
        temp = pic_info.Te if species in ['e', 'electron'] else pic_info.Ti
        # print("Particle initial temperature (Lorentz factor): %f" % temp)
        # In typical VPIC simulation, me=1, c=1
        enthalpy_e =  n0 * (1 + pic_info.Te * gamma_e / (gamma_e - 1))
        enthalpy_i =  n0 * (mime + pic_info.Ti * gamma_i / (gamma_i - 1))
        enthalpy = enthalpy_e + enthalpy_i
        b0 = binplane[irun]
        br = b0
        bg = bg_runs[irun]
        sigma[irun] = b0**2 / enthalpy
        sigma_r = br**2 / enthalpy
        sigma_c[irun] = b0**2 / (n0 * (1 + mime))
        sigmae_c[irun] = b0**2 / n0
        sigmai_c[irun] = b0**2 / (n0 * mime)
        sigmae_h[irun] = b0**2 / enthalpy_e
        sigmai_h[irun] = b0**2 / enthalpy_i
        beta_e[irun] = 2 * n0 * pic_info.Te / b0**2
        spect_params = get_spect_params(pic_run)
        pindex[irun] = -spect_params["power_index"]
        va = math.sqrt(sigma_r / (sigma_r+sigma[irun]*bg**2 + 1))
        sigma_tot = sigma_r + sigma[irun]*bg**2
        va0 = math.sqrt(sigma_tot / (sigma_tot + 1))
        va4[irun] = va / math.sqrt(1 - va**2)
        gamma_the = pic_info.Te + 1
        vthe = math.sqrt(gamma_the**2 - 1) / gamma_the
        gamma_thi = pic_info.Ti / pic_info.mime + 1
        vthi = math.sqrt(gamma_thi**2 - 1) / gamma_thi
        btot = math.sqrt(1 + bg**2)
        de = np.sqrt(gamma_the)
        di = np.sqrt(pic_info.mime * gamma_thi)
        rhoe = gamma_the * vthe / (btot * b0)
        rhoi = gamma_thi * pic_info.mime * vthi / (btot * b0)
        gamma_va = 1.0 / math.sqrt(1 - va**2)
        gamma_va0 = 1.0 / math.sqrt(1 - va0**2)
        factor = gamma_va**2
        # factor = gamma_va**2 * va**2 / 3 + 1
        # factor = (gamma_va**2 + gamma_va + 1) / 3
        factor1 = (gamma_va**2 + gamma_va + 1) / 3
        factor2 = (gamma_va+1)**(3/2) * (gamma_va-1)**(1/2) / 3
        factor3 = factor1 - 1
        ctheta = 1.0 / btot
        lx_de = pic_info.lx_di * math.sqrt(pic_info.mime)
        if "mime1_sigmae" in pic_run:
            l = lx_de / 2
        else:
            # l = lx_de * va / 3
            l = lx_de / 9
            # l = lx_de / 10
        # acc_rate = (factor * (1 + va + va**2) - 1) / (4 * l)
        l0 = max(di, rhoi)
        l1 = l
        acc_rate = va * math.log(l1/l0) / (l1-l0) / 3 / 3
        # acc_rate = va * math.log(l1/l0) / (l1-l0) / 3 / gamma_va
        # acc_rate = (gamma_va**2 * (1 + va + va**2) - 1) * math.log(l1/l0) / (l1-l0) / 6
        if "mime1_sigmae" in pic_run:
            lc = l
            # sigma0 = 1.0 / (1.0 + bg**2)**(1/3)
            sigma0 = 1.0 / bg**2
            # if sigma0 > 1:
            #     sigma0 = 1.0
            csc_3pi_5 = 1.05146
            # omega0 = btot * b0 / gamma
            gamma = pic_info.Te / beta_e[irun] + 1
            acc_rate = (gamma + 1) * va * math.log(l1/l0) / (l1-l0) / (3 * gamma)
            omega0 = bg * b0 / gamma
            vtheh = math.sqrt(1 - 1/gamma**2)
            kpara = 3 * vtheh**3 * csc_3pi_5 / (20 * lc * omega0**2 * sigma0)
            kpara *= 1.0 + (72.0 / 7) * (omega0 * lc / vtheh)**(5/3)
            # print(vtheh, sigma0, omega0**(1/3)*sigma0)
            # esc_rate = va**2 / kpara
            esc_rate = kpara / l**2
        else:
            # esc_rate = 1.0 / l
            # esc_rate = 1.0 / l / gamma_va
            esc_rate = va / l
        print("Acceleration and escape rates: %e, %e" % (acc_rate, esc_rate))
        pindex_va = 1.0 + esc_rate / acc_rate
        # pindex_va = 1 + 8.0 * va / (factor * (1 + 2*va*ctheta + va**2) - 1)
        # pindex_va = 1 + 8.0 * va / (factor1 + 2*factor2*ctheta + factor3 - 1)
        ps[1, irun] = pindex[irun]
        ps[0, irun] = pindex_va
        # ps[0, irun] = 1 + 4.5 * math.sqrt(1 + bg**2) / math.log(l1/l0)
        # print("va, gamma_va, pindex_va: %0.2f, %0.2f, %0.2f" %
        #       (va, gamma_va, pindex_va))

    fig = plt.figure(figsize=[7, 7])
    rect = [0.11, 0.12, 0.85, 0.85]
    hgap, vgap = 0.07, 0.04
    ax = fig.add_axes(rect)
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    ax.set_prop_cycle('color', COLORS)
    n1 = nsigma * nbg + nsigmae * ndb
    ltext = r"$\sigma_e=400, m_i/m_e=1$"
    if reconnection_run:
        plots_rec = []
        p3 = ax.scatter(ps[1, n1], ps[0, n1], marker='X', label=ltext, s=100)
        plots_rec = [p3]
        for i in range(nsigma):
            sigma = str(int(sigmas[i]) / 10)
            ltext = r"$\sigma_i=" + sigma + "$"
            alphas = np.linspace(0.2, 1, nbg)
            rgba_colors = np.zeros((nbg,4))
            rgba_colors[:,0] = COLORS[i+1][0]
            rgba_colors[:,1] = COLORS[i+1][1]
            rgba_colors[:,2] = COLORS[i+1][2]
            # the fourth column needs to be your alphas
            rgba_colors[:, 3] = alphas
            # if sigmas[i] in ["01", "16"]:
            #     rs, re = i*nbg+2, (i+1)*nbg
            #     ax.scatter(ps[1, rs:re], ps[0, rs:re], color=rgba_colors[2:],
            #                marker='o', s=100)
            # else:
            #     rs, re = i*nbg, (i+1)*nbg
            #     ax.scatter(ps[1, rs:re], ps[0, rs:re], color=rgba_colors,
            #                marker='o', s=100)
            rs, re = i*nbg, (i+1)*nbg
            ax.scatter(ps[1, rs:re], ps[0, rs:re], color=rgba_colors,
                       marker='o', s=100)
            p1 = ax.scatter(ps[1, rs:re]+10, ps[0, rs:re]+10,
                            label=ltext, color=COLORS[i+1],
                            marker='o', s=100)
            plots_rec.append(p1)
    if turbulence_run:
        if reconnection_run:
            shift = nsigma * nbg
        else:
            shift = 0
        plots_turb = []
        for i in range(nsigmae):
            ltext = r"$\sigma_e=" + sigmaes[i] + "$"
            ndb = len(db2s)
            alphas = np.linspace(0.2, 1, ndb)
            rgba_colors = np.zeros((ndb,4))
            rgba_colors[:,0] = COLORS[i+nsigma+1][0]
            rgba_colors[:,1] = COLORS[i+nsigma+1][1]
            rgba_colors[:,2] = COLORS[i+nsigma+1][2]
            # the fourth column needs to be your alphas
            rgba_colors[:, 3] = alphas
            rs, re = i*ndb+shift, (i+1)*ndb+shift
            ax.scatter(ps[1, rs:re], ps[0, rs:re], color=rgba_colors,
                       marker='d', s=100)
            p1 = ax.scatter(ps[1, rs:re]+10, ps[0, rs:re]+10,
                            label=ltext, color=COLORS[i+nsigma+1],
                            marker='d', s=100)
            plots_turb.append(p1)

        # leg1 = ax.legend(loc=2, handles=plots_rec, prop={'size': 16}, ncol=1,
        #                  shadow=False, fancybox=False, frameon=False,
        #                  title="Reconnection", title_fontsize=16)

        # leg2 = ax.legend(loc=4, handles=plots_turb, prop={'size': 16}, ncol=1,
        #                  shadow=False, fancybox=False, frameon=False,
        #                  title="Turbulence", title_fontsize=16)
        # # ax.add_artist(leg1)
        # brace = CurlyBrace(x=0.28, y=0.665, width=0.05, height=.2, pointing='right',
        #                    transform=ax.transAxes, color='k')
        # ax.add_artist(brace)
    # text1 = r"$m_i/m_e=1836$"
    # ax.text(0.34, 0.765, text1, color='k', fontsize=16, rotation=0,
    #         bbox=dict(facecolor='none', alpha=1.0,
    #                   edgecolor='none', pad=10.0),
    #         horizontalalignment='left', verticalalignment='center',
    #         transform=ax.transAxes)

    # slope, intercept, r_value, p_value, std_err = stats.linregress(ps[1], ps[0])
    # # print(slope, intercept, r_value, p_value, std_err)

    # pmodel = np.linspace(1, 5, 10)
    # ax.plot(pmodel, intercept + slope*pmodel, color='k',
    #         alpha=0.5, label='fitted line', zorder=0)

    # text1 = (r"$p_m=" + (r"%0.3f" % slope) + r"p_s" + (r"%0.3f" % intercept) + r"$")
    # ax.text(0.77, 0.87, text1, color='k', fontsize=12, rotation=47.11,
    #         bbox=dict(facecolor='none', alpha=1.0,
    #                   edgecolor='none', pad=10.0),
    #         horizontalalignment='center', verticalalignment='center',
    #         transform=ax.transAxes)
    # text2 = (r"$R^2=" + (r"%0.3f" % r_value) + r"$")
    # ax.text(0.75, 0.9, text2, color='k', fontsize=12, rotation=47.11,
    #         bbox=dict(facecolor='none', alpha=1.0,
    #                   edgecolor='none', pad=10.0),
    #         horizontalalignment='center', verticalalignment='center',
    #         transform=ax.transAxes)
    ax.tick_params(bottom=True, top=True, left=True, right=False)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlabel(r'Power-law index $p_s$ (simulation)', fontsize=20)
    ax.set_ylabel(r'Power-law index $p_m$ (model)', fontsize=20)
    ax.tick_params(labelsize=12)

    # ax.plot([1, 5.0], [1, 5.0], linewidth=1, linestyle='--', color='k')
    # ax.plot([1, 5.0], [1.5, 5.5], linewidth=1, linestyle='--', color='k')
    # ax.plot([1, 5.0], [0.5, 4.5], linewidth=1, linestyle='--', color='k')
    # ax.set_xlim([1, 5.0])
    # ax.set_ylim([1, 5.0])
    # ax.set_xticks(range(1, 6))
    # ax.set_yticks(range(1, 6))
    fdir = '../img/power_law_index/spectrum/'
    mkdir_p(fdir)
    fname = fdir + 'spectra_model.pdf'
    fig.savefig(fname)
    plt.show()


def analytical_power_index(plot_config, show_plot=True):
    """Plot analytical power-law index

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    if species in ["e", "electron"]:
        sname = "electron"
    else:
        sname = "Ion"

    root_dir = "/net/scratch4/xiaocanli/reconnection/power_law_index/"
    pic_runs, labels = get_all_runs(root_dir)
    nruns = len(pic_runs)

    runs, rune = 0, 5
    sigma = np.zeros(rune - runs)
    sigma_c = np.zeros(rune - runs)
    sigmae_c = np.zeros(rune - runs)
    sigmai_c = np.zeros(rune - runs)
    sigmae_h = np.zeros(rune - runs)
    sigmai_h = np.zeros(rune - runs)
    beta_e = np.zeros(rune - runs)
    pindex = np.zeros(rune - runs)
    va4 = np.zeros(rune - runs)  # four Alfven speed
    pindex_pred = np.zeros(rune - runs)
    for irun, pic_run in enumerate(pic_runs[runs:rune]):
        print("PIC run name: %s" % pic_run)
        pic_run_dir = root_dir + pic_run
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        mime = pic_info.mime
        n0 = pic_info.n0
        gamma_e = 4.0/3 if pic_info.Te > 0.1 else 1.5  # Adiabatic index
        gamma_i = 5.0/3
        # In typical VPIC simulation, me=1, c=1
        enthalpy_e =  n0 * (1 + pic_info.Te * gamma_e / (gamma_e - 1))
        enthalpy_i =  n0 * (mime + pic_info.Ti * gamma_i / (gamma_i - 1))
        enthalpy = enthalpy_e + enthalpy_i
        sigma[irun] = pic_info.b0**2 / enthalpy
        sigma_c[irun] = pic_info.b0**2 / (n0 * (1 + mime))
        sigmae_c[irun] = pic_info.b0**2 / n0
        sigmai_c[irun] = pic_info.b0**2 / (n0 * mime)
        sigmae_h[irun] = pic_info.b0**2 / enthalpy_e
        sigmai_h[irun] = pic_info.b0**2 / enthalpy_i
        beta_e[irun] = 2 * n0 * pic_info.Te / pic_info.b0**2
        spect_params = get_spect_params(pic_run)
        pindex[irun] = -spect_params["power_index"]
        va = math.sqrt(sigma[irun] / (sigma[irun] + 1))
        va4[irun] = va / math.sqrt(1 - va**2)
        # pindex_pred[irun] = 1 + 2*va / (va*0.5 + 2*va**2)
        gamma_the = pic_info.Te + 1
        vthe = math.sqrt(gamma_the**2 - 1) / gamma_the
        gamma_va = 1.0 / math.sqrt(1 - va**2)
        factor = gamma_va**2
        # factor = gamma_va**2 * va**2 / 3 + 1
        pindex_va = 1 + 8.0 * va / (factor * (1 + va + va**2) - 1)
        pindex_pred[irun] = pindex_va
        print("va, gamma_va, pindex_va: %0.2f, %0.2f, %0.2f" %
              (va, gamma_va, pindex_va))

    sigma_array = np.logspace(-1, 2, 300)
    va = np.sqrt(sigma_array / (sigma_array + 1))
    gamma_va = 1.0 / np.sqrt(1 - va**2)
    factor = gamma_va**2
    pindex_array = 1 + 8.0 * va / (factor * (1 + va + va**2) - 1)
    fig = plt.figure(figsize=[7, 5])
    rect = [0.12, 0.16, 0.82, 0.8]
    ax = fig.add_axes(rect)
    ax.semilogx(sigma_array, pindex_array, linestyle='-', linewidth=2,
                label='prediction')
    ax.semilogx(sigma, pindex, linestyle='none',
                marker='o', markersize=10, label='simulation')
    ax.legend(loc=1, prop={'size': 16}, ncol=1,
             shadow=False, fancybox=False, frameon=False)
    ax.tick_params(bottom=True, top=False, left=True, right=True)
    ax.set_xlabel(r"$\sigma$", fontsize=20)
    ax.set_ylabel(r'$p$', fontsize=20)
    ax.tick_params(labelsize=16)
    fdir = '../img/power_law_index/'
    mkdir_p(fdir)
    fname = fdir + 'spect_pred_simul_' + species + '.pdf'
    fig.savefig(fname)
    plt.show()


def acceleration_rate(plot_config, show_plot=True):
    """Particle-based energization acceleration rate

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    tframe = plot_config["tframe"]
    tstart = plot_config["tstart"]
    tend = plot_config["tend"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]

    fnorm = 1E-3
    fig1 = plt.figure(figsize=[16, 10])
    rect0 = [0.05, 0.75, 0.15, 0.2]
    hgap, vgap = 0.04, 0.02
    row, col = 4, 5
    axs = []
    for i in range(row):
        rect = np.copy(rect0)
        rect[1] -= (rect[3] + vgap) * i
        for j in range(col):
            ax = fig1.add_axes(rect)
            ax.set_prop_cycle('color', COLORS)
            rect[0] += rect[2] + hgap
            if i < row - 1:
                ax.tick_params(axis='x', labelbottom=False)
            else:
                ax.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$', fontsize=12)
            ax.tick_params(bottom=True, top=False, left=True, right=True)
            ax.tick_params(axis='x', which='minor', direction='in', top=True)
            ax.tick_params(axis='x', which='major', direction='in', top=True)
            ax.tick_params(axis='y', which='minor', direction='in')
            ax.tick_params(axis='y', which='major', direction='in')
            ax.set_xlim([1E1, 1E4])
            axs.append(ax)
    labels = [r'$\boldsymbol{E}_\parallel$', r'$\boldsymbol{E}_\perp$',
              'Compression', 'Shear', 'Curvature', 'Gradient',
              'Parallel drift', r'$\mu$ conservation', 'Polar-time',
              'Polar-spatial', 'Inertial-time', 'Inertial-spatial',
              'Polar-fluid-time', 'Polar-fluid-spatial',
              'Polar-time-v', 'Polar-spatial-v', 'Ptensor',
              r'$\boldsymbol{E}_\parallel + \boldsymbol{E}_\perp$',
              'Compression + Shear', 'Curvature + Gradient']

    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    dtp = math.ceil(pic_info.dt_particles / 0.1) * 0.1
    tstep = tframe * pic_info.particle_interval
    tframe_fluid = tstep // pic_info.fields_interval
    fpath = "../data/particle_interp/" + pic_run + "/"
    fname = fpath + "particle_energization_" + species + "_" + str(tstep) + ".gda"
    fdata = np.fromfile(fname, dtype=np.float32)
    nbins = int(fdata[0])
    nbinx = int(fdata[1])
    nvar = int(fdata[2])
    ebins = fdata[3:nbins+3]
    fbins = np.sum(fdata[nbins+3:].reshape((nvar, nbinx, nbins)), axis=1)
    temp = pic_info.Te if species in ['e', 'electron'] else pic_info.Ti
    print("Particle initial temperature (Lorentz factor): %f" % temp)
    ebins /= temp

    if species == 'i':
        ebins *= pic_info.mime  # ebins are actually gamma
    fbins[1:, :] = div0(fbins[1:, :], fbins[0, :])
    dee = fbins[18:, :] - fbins[1:18, :]**2

    emin, emax = 1E1, 1E4
    es, _ = find_nearest(ebins, emin)
    ee, _ = find_nearest(ebins, emax)

    ymax = np.max(fbins[3:6, es:ee+1])
    for iplot in range(17):
        ax = axs[iplot]
        ax.semilogx(ebins[es:ee+1], fbins[iplot+1, es:ee+1],
                    marker='o', markersize=4, linestyle='-', linewidth=1)
        ax.grid(True)

        ax.set_ylim([-5E-4, ymax])

        ax.text(0.05, 0.85, labels[iplot], color='k', fontsize=12,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='bottom',
                transform=ax.transAxes)
        if iplot == 2:
            text1 = r'$t\Omega_{ci}=' + str(tframe*10) + '$'
            ax.set_title(text1, fontsize=20)
        ax.plot(ax.get_xlim(), [0, 0], color='k', linestyle='--')
        ax.tick_params(labelsize=10)
    ax = axs[17]
    ax.semilogx(ebins[es:ee+1], fbins[1, es:ee+1] + fbins[2, es:ee+1],
                marker='o', markersize=4, linestyle='-', linewidth=1)
    ax.grid(True)
    ax.set_ylim([-5E-4, ymax])
    ax.text(0.05, 0.05, labels[17], color='k', fontsize=12,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes)

    ax = axs[18]
    ax.semilogx(ebins[es:ee+1], fbins[3, es:ee+1] + fbins[4, es:ee+1],
                marker='o', markersize=4, linestyle='-', linewidth=1)
    ax.grid(True)
    ax.set_ylim([-5E-4, ymax])
    ax.text(0.05, 0.05, labels[18], color='k', fontsize=12,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes)

    ax = axs[19]
    ax.semilogx(ebins[es:ee+1], fbins[5, es:ee+1] + fbins[6, es:ee+1],
                marker='o', markersize=4, linestyle='-', linewidth=1)
    ax.grid(True)
    ax.set_ylim([-5E-4, ymax])
    ax.text(0.05, 0.05, labels[19], color='k', fontsize=12,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes)

    # fdir = '../img/cori_3d/acceleration_rates/bg' + bg_str + '/'
    # mkdir_p(fdir)
    # fname = fdir + 'acc_rates_' + str(tframe) + '.pdf'
    # fig1.savefig(fname)

    if show_plot:
        plt.show()
    else:
        plt.close('all')


def acceleration_rate_std(plot_config, show_plot=True):
    """The standard deviation of acceleration rates

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    tframe = plot_config["tframe"]
    tstart = plot_config["tstart"]
    tend = plot_config["tend"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]

    fnorm = 1E-3
    fig1 = plt.figure(figsize=[16, 10])
    rect0 = [0.05, 0.75, 0.15, 0.2]
    hgap, vgap = 0.04, 0.02
    row, col = 4, 5
    axs = []
    for i in range(row):
        rect = np.copy(rect0)
        rect[1] -= (rect[3] + vgap) * i
        for j in range(col):
            ax = fig1.add_axes(rect)
            ax.set_prop_cycle('color', COLORS)
            rect[0] += rect[2] + hgap
            if i < row - 1:
                ax.tick_params(axis='x', labelbottom=False)
            else:
                ax.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$', fontsize=12)
            ax.tick_params(bottom=True, top=False, left=True, right=True)
            ax.tick_params(axis='x', which='minor', direction='in', top=True)
            ax.tick_params(axis='x', which='major', direction='in', top=True)
            ax.tick_params(axis='y', which='minor', direction='in')
            ax.tick_params(axis='y', which='major', direction='in')
            # ax.set_xlim([1E0, 5E2])
            axs.append(ax)
    labels = [r'$\boldsymbol{E}_\parallel$', r'$\boldsymbol{E}_\perp$',
              'Compression', 'Shear', 'Curvature', 'Gradient',
              'Parallel drift', r'$\mu$ conservation', 'Polar-time',
              'Polar-spatial', 'Inertial-time', 'Inertial-spatial',
              'Polar-fluid-time', 'Polar-fluid-spatial',
              'Polar-time-v', 'Polar-spatial-v', 'Ptensor',
              r'$\boldsymbol{E}_\parallel + \boldsymbol{E}_\perp$',
              'Compression + Shear', 'Curvature + Gradient']

    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    dtp = math.ceil(pic_info.dt_particles / 0.1) * 0.1
    tstep = tframe * pic_info.particle_interval
    tframe_fluid = tstep // pic_info.fields_interval
    fpath = "../data/particle_interp/" + pic_run + "/"
    fname = fpath + "particle_energization_" + species + "_" + str(tstep) + ".gda"
    fdata = np.fromfile(fname, dtype=np.float32)
    nbins = int(fdata[0])
    nbinx = int(fdata[1])
    nvar = int(fdata[2])
    ebins = fdata[3:nbins+3]
    gamma = ebins + 1
    fbins = np.sum(fdata[nbins+3:].reshape((nvar, nbinx, nbins)), axis=1)
    pbins = np.sqrt((ebins + 1)**2 - 1)
    pbins_mid = (pbins[:-1] + pbins[1:]) * 0.5
    dpbins = np.diff(pbins)

    if species == 'i':
        ebins *= pic_info.mime  # ebins are actually gamma
    fbins[1:, :] = div0(fbins[1:, :], fbins[0, :])
    dee = fbins[18:, :] - fbins[1:18, :]**2
    dgamma = fbins[1:18, :] * ebins
    dgamma2 = fbins[18:, :] * ebins**2
    dp = gamma * dgamma / np.sqrt(gamma**2 - 1)
    dp2 = dgamma2 + 2 * dgamma
    dpp = (dp2 - dp**2) / pbins**2

    temp = pic_info.Te if species in ['e', 'electron'] else pic_info.Ti
    print("Particle initial temperature (Lorentz factor): %f" % temp)
    ebins /= temp

    emin, emax = 1E0, 1E3
    es, _ = find_nearest(ebins, emin)
    ee, _ = find_nearest(ebins, emax)

    ymax = np.max(dee[:5, es:ee+1])
    for iplot in range(17):
        ax = axs[iplot]
        ax.loglog(ebins[es:ee+1], dee[iplot, es:ee+1]*ebins[es:ee+1],
                  marker='o', markersize=4, linestyle='-', linewidth=1)
        # ax.loglog(pbins, dpp[iplot],
        #           marker='o', markersize=4, linestyle='-', linewidth=1)
        ax.grid(True)

        # ax.set_ylim([-5E-4, ymax])

        ax.text(0.05, 0.05, labels[iplot], color='k', fontsize=12,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='bottom',
                transform=ax.transAxes)
        if iplot == 2:
            ax.text(0.4, 0.95, '3D', color=COLORS[0], fontsize=16,
                    bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                    horizontalalignment='center', verticalalignment='top',
                    transform=ax.transAxes)
            ax.text(0.6, 0.95, '2D', color=COLORS[1], fontsize=16,
                    bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                    horizontalalignment='center', verticalalignment='top',
                    transform=ax.transAxes)
            text1 = r'$t\Omega_{ci}=' + str(tframe*10) + '$'
            ax.set_title(text1, fontsize=20)
        # ax.plot(ax.get_xlim(), [0, 0], color='k', linestyle='--')
        ax.tick_params(labelsize=10)
        # ax = axs[17]
        # ax.semilogx(ebins[es:ee+1], fbins[1, es:ee+1] + fbins[2, es:ee+1],
        #             marker='o', markersize=4, linestyle='-', linewidth=1)
        # ax.grid(True)
        # ax.set_ylim([-5E-4, ymax])
        # ax.text(0.05, 0.05, labels[17], color='k', fontsize=12,
        #         bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        #         horizontalalignment='left', verticalalignment='bottom',
        #         transform=ax.transAxes)

        # ax = axs[18]
        # ax.semilogx(ebins[es:ee+1], fbins[3, es:ee+1] + fbins[4, es:ee+1],
        #             marker='o', markersize=4, linestyle='-', linewidth=1)
        # ax.grid(True)
        # ax.set_ylim([-5E-4, ymax])
        # ax.text(0.05, 0.05, labels[18], color='k', fontsize=12,
        #         bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        #         horizontalalignment='left', verticalalignment='bottom',
        #         transform=ax.transAxes)

        # ax = axs[19]
        # ax.semilogx(ebins[es:ee+1], fbins[5, es:ee+1] + fbins[6, es:ee+1],
        #             marker='o', markersize=4, linestyle='-', linewidth=1)
        # ax.grid(True)
        # ax.set_ylim([-5E-4, ymax])
        # ax.text(0.05, 0.05, labels[19], color='k', fontsize=12,
        #         bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        #         horizontalalignment='left', verticalalignment='bottom',
        #         transform=ax.transAxes)

    # fdir = '../img/cori_3d/acceleration_rates/bg' + bg_str + '/'
    # mkdir_p(fdir)
    # fname = fdir + 'acc_rates_' + str(tframe) + '.pdf'
    # fig1.savefig(fname)

    if show_plot:
        plt.show()
    else:
        plt.close('all')


def diffusion_coefficient(plot_config, show_plot=True):
    """Calculate energy diffusion coefficient

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    tframe = plot_config["tframe"]
    tstart = plot_config["tstart"]
    tend = plot_config["tend"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]

    fig = plt.figure(figsize=[7, 5])
    rect = [0.12, 0.16, 0.72, 0.8]
    ax = fig.add_axes(rect)

    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    alpha_min = plot_config["vkappa_threshold"]
    mime = pic_info.mime
    ntp = len(os.listdir(pic_run_dir + "particle/"))
    spect_params = get_spect_params(pic_run)
    pindex = spect_params["power_index"]
    emin, emax = spect_params["energy_range"]
    n0 = pic_info.n0
    gamma_e = 4.0/3 if pic_info.Te > 0.1 else 1.5  # Adiabatic index
    gamma_i = 5.0/3
    # In typical VPIC simulation, me=1, c=1
    enthalpy_e =  n0 * (1 + pic_info.Te * gamma_e / (gamma_e - 1))
    enthalpy_i =  n0 * (mime + pic_info.Ti * gamma_i / (gamma_i - 1))
    enthalpy = enthalpy_e + enthalpy_i
    sigma = pic_info.b0**2 / enthalpy
    sigma_c = pic_info.b0**2 / (n0 * (1 + mime))
    sigmae_c = pic_info.b0**2 / n0
    sigmai_c = pic_info.b0**2 / (n0 * mime)
    sigmae_h = pic_info.b0**2 / enthalpy_e
    sigmai_h = pic_info.b0**2 / enthalpy_i
    beta_e = 2 * n0 * pic_info.Te / pic_info.b0**2

    if species in ["e", "electron"]:
        sname = "electron"
    else:
        sname = "Ion"
    temp = pic_info.Te if species in ['e', 'electron'] else pic_info.Ti

    va = math.sqrt(sigma / (sigma + 1))
    if int(mime) == 1:
        ndi_max = 60.0
    elif int(mime) == 25:
        ndi_max = 40.0
    elif mime > 1000:
        ndi_max = 10.0
    else:
        ndi_max = 10.0
    # ndi_max = 10 * math.sqrt(1+pic_info.Ti/pic_info.mime)
    alpha_min = va/math.sqrt(mime)/ndi_max  # vA/(ndi_max * di)
    # alpha_min /= bg**2 + 1.0  # consider guide field

    fpath = "../data/particle_interp/" + pic_run + "/"
    tindex = pic_info.particle_interval
    fname = fpath + "acc_rate_dist_vkappa_grid_" + species + "_" + str(tindex) + ".gda"
    fdata = np.fromfile(fname, dtype=np.float32)
    nvar = int(fdata[2])   # number of variables

    nhigh_acc_t = np.zeros(ntp)
    nhigh_esc_t = np.zeros(ntp)
    arate_acc_t = np.zeros([nvar, ntp])
    arate_esc_t = np.zeros([nvar, ntp])
    arate2_acc_t = np.zeros([nvar, ntp])
    arate2_esc_t = np.zeros([nvar, ntp])

    for tframe in range(1, int(pic_info.ntp)):
        print("Time frame: %d" % tframe)
        tindex = tframe * pic_info.particle_interval
        fname = fpath + "acc_rate_dist_vkappa_grid_" + species + "_" + str(tindex) + ".gda"
        fdata = np.fromfile(fname, dtype=np.float64)
        nalpha = int(fdata[0]) # number of bins of the rates
        nbins = int(fdata[1])  # number of bins along x
        nvar = int(fdata[2])   # number of variables
        ebins = fdata[3:nbins+3]  # energy bins
        ebins /= temp  # normalize by initial temperature
        alpha_bins = fdata[nbins+3:nbins+nalpha+3]  # acceleration rates bins
        alpha_bins_mid = 0.5 * (alpha_bins[1:] + alpha_bins[:-1])
        # variables: 0-Epara, 1-Eperp, 2-compression, 3-shear, 4-curvature,
        #   5-gradient, 6-para_drift, 7-mu, 8-polar_time, 9-polar_spatial,
        #   10-inertial_time, 11-inertial_spatial, 12-polar_fluid_time,
        #   13-polar_fluid_spatial, 14-polar_time_v, 15-polar_spatial_v,
        #   16-ptensor
        fdist = fdata[nbins+nalpha+3:].reshape((nvar+1, nbins, (nalpha+1)*4))

        es, _ = find_nearest(ebins, emin)
        ee, _ = find_nearest(ebins, emax)
        ethe, _ = find_nearest(ebins, 1)
        nalpha0 = nalpha + 1
        alpha_s, _ = find_nearest(alpha_bins, alpha_min)
        pos_range = np.arange(nalpha0+1, nalpha0*2-1)
        neg_range = np.arange(nalpha0-2, 0, -1)
        fnptl = fdist[0, :, :nalpha0*2]
        falpha = fdist[1:, :, :nalpha0*2]
        falpha_sq = fdist[1:, :, nalpha0*2:]
        fnptl_pos = fnptl[:, pos_range]
        fnptl_neg = fnptl[:, neg_range]
        falpha_pos = falpha[:, :, pos_range]
        falpha_neg = falpha[:, :, neg_range]
        falpha_sq_pos = falpha_sq[:, :, pos_range]
        falpha_sq_neg = falpha_sq[:, :, neg_range]
        alpha = div0(np.sum(falpha, axis=2), np.sum(fnptl, axis=1))
        alpha_sq = div0(np.sum(falpha_sq, axis=2), np.sum(fnptl, axis=1))
        alpha_acc = div0(np.sum(falpha_pos[:, :, alpha_s:] +
                                falpha_neg[:, :, alpha_s:], axis=2),
                         np.sum(fnptl_pos[:, alpha_s:] +
                                fnptl_neg[:, alpha_s:], axis=1))
        alpha_sq_acc = div0(np.sum(falpha_sq_pos[:, :, alpha_s:] +
                                   falpha_sq_neg[:, :, alpha_s:], axis=2),
                            np.sum(fnptl_pos[:, alpha_s:] +
                                   fnptl_neg[:, alpha_s:], axis=1))
        w = 1.164153e-03
        dee = alpha_sq*w - (alpha*w**2)**2
        dee_acc = alpha_sq_acc*w - (alpha_acc*w**2)**2

        plt.loglog(ebins[es:ee+1], dee_acc[1, es:ee+1]*ebins[es:ee+1]**2)

        # fdist_high = np.sum(fdist[:, es:ee, :], axis=1)
        # nptl_bins = fdist_high[:, :nalpha0*2]
        # dene_bins = fdist_high[:, nalpha0*2:nalpha0*4]
        # dene2_bins = fdist_high[:, nalpha0*4:]
        # alpha_bins = div0(dene_bins, nptl_bins)
        # fnptl = nptl_bins[1]
        # fdene = dene_bins[1]
        # fnptl_pos = fnptl[pos_range]
        # fnptl_neg = fnptl[neg_range]
        # fdene_pos = fdene[pos_range]
        # fdene_neg = fdene[neg_range]
        # # if tframe == 30:
        # #     plt.semilogx(alpha_bins_mid, fdene_pos)
        # #     plt.semilogx(alpha_bins_mid, -fdene_neg)
        # nhigh_acc_t[tframe] = np.sum(fnptl_pos[alpha_s:] + fnptl_neg[alpha_s:])
        # nhigh_esc_t[tframe] = np.sum(fnptl_pos + fnptl_neg) - nhigh_acc_t[tframe]

        # fdene_var_pos = dene_bins[:, pos_range]
        # fdene_var_neg = dene_bins[:, neg_range]
        # fdene2_var_pos = dene2_bins[:, pos_range]
        # fdene2_var_neg = dene2_bins[:, neg_range]

        # arate_acc_t[:, tframe] = np.sum(fdene_var_pos[:, alpha_s:] +
        #                                 fdene_var_neg[:, alpha_s:], axis=1)
        # arate_esc_t[:, tframe] = (np.sum(fdene_var_pos + fdene_var_neg, axis=1) -
        #                           arate_acc_t[:, tframe])
        # arate2_acc_t[:, tframe] = np.sum(fdene2_var_pos[:, alpha_s:] +
        #                                  fdene2_var_neg[:, alpha_s:], axis=1)
        # arate2_esc_t[:, tframe] = (np.sum(fdene2_var_pos + fdene2_var_neg, axis=1) -
        #                            arate2_acc_t[:, tframe])

    print("Particle initial temperature (Lorentz factor): %f" % temp)
    print("Alfven speed: %f" % va)
    print("alpha_min va/(%0.1f * di): %f" % (ndi_max, alpha_min))
    print("sigma: %f" % sigma)
    print("sigma_c: %f" % sigma_c)
    print("sigmae_c: %f" % sigmae_c)
    print("sigmae_h: %f" % sigmae_h)
    print("sigmai_c: %f" % sigmai_c)
    print("sigmai_h: %f" % sigmai_h)

    # dtwpe_particle = pic_info.particle_interval * pic_info.dtwpe
    # tparticles = np.arange(0, ntp) * dtwpe_particle
    # tmin, tmax = tparticles[0], tparticles[-1]
    # acc_rate = div0(arate_acc_t[1, :], nhigh_acc_t)
    # acc_rate_esc = div0(arate_esc_t[1, :], nhigh_esc_t)
    # dee_acc = div0(arate2_acc_t[1, :], nhigh_acc_t) - acc_rate**2
    # dee_esc = div0(arate2_esc_t[1, :], nhigh_esc_t) - acc_rate_esc**2
    # esc_rate = div0(np.gradient(nhigh_esc_t, dtwpe_particle), nhigh_acc_t)
    # power_index = 1 + div0(esc_rate, acc_rate)

    # plt.plot(acc_rate, color='r')
    # plt.plot(esc_rate, color='b')
    # plt.ylim([-0.001, 0.005])

    # ax.loglog(ebins[es:ee+1], dee[1, es:ee+1] * ebins[es:ee+1]**1.5,
    #           marker='o', markersize=4, linestyle='-', linewidth=1,
    #           color = plt.cm.jet(tframe/float(pic_info.ntp), 1),
    #           nonposy='mask')

    if show_plot:
        plt.show()
    else:
        plt.close('all')


def plot_trajectory(plot_config, show_plot=True):
    """Plot particle trajectory
    """
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    tracer_dir = "/net/scratch3/xiaocan/vpic-sorter/data/power_law_index/"
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    dtwpe_tracer = pic_info.dtwpe * pic_info.tracer_interval
    wce_wpe = pic_info.dtwce / pic_info.dtwpe
    fname = tracer_dir + pic_run + "/electrons_ntraj500_1emax.h5p"
    fh = h5py.File(fname, 'r')
    particle_tags = list(fh.keys())
    nptl = len(particle_tags)
    print("Total number of particles: %d" % nptl)
    group = fh[particle_tags[0]]
    dset = group['dX']
    nframes, = dset.shape
    ptl = {}
    for dset_name in group:
        dset = group[dset_name]
        ptl[str(dset_name)] = np.zeros(dset.shape, dset.dtype)
        dset.read_direct(ptl[str(dset_name)])
    gamma0 = 1.0 / np.sqrt(1 + ptl["Ux"]**2 + ptl["Uy"]**2 + ptl["Uz"]**2)
    ttracer = np.arange(0, nframes) * dtwpe_tracer
    tmin, tmax = ttracer[0], ttracer[-1]
    smime = math.sqrt(pic_info.mime)
    lx_de = smime * pic_info.lx_di
    lz_de = smime * pic_info.lz_di

    img_dir = '../img/power_law_index/tracer_traj/' + pic_run + '/'
    mkdir_p(img_dir)
    if species in ["e", "electron"]:
        sname = "electron"
    else:
        sname = "Ion"

    ng = 201
    kernel = np.ones(ng) / float(ng)

    # for iptl in range(nptl):
    for iptl in range(9, 10):
        print("Particle: %d of %d" % (iptl, nptl))
        group = fh[particle_tags[iptl]]
        for dset_name in group:
            dset = group[dset_name]
            dset.read_direct(ptl[str(dset_name)])
        gamma = np.sqrt(1 + ptl["Ux"]**2 + ptl["Uy"]**2 + ptl["Uz"]**2)
        tesc = np.argmax(gamma > 0.9*gamma[-1])
        dgamma = gamma - gamma[0]
        igamma = 1.0 / gamma
        vx = ptl["Ux"] * igamma
        vy = ptl["Uy"] * igamma
        vz = ptl["Uz"] * igamma
        x = ptl["dX"]
        y = ptl["dY"]
        z = ptl["dZ"]
        ex = ptl["Ex"]
        ey = ptl["Ey"]
        ez = ptl["Ez"]
        bx = ptl["Bx"]
        by = ptl["By"]
        bz = ptl["Bz"]
        edotb = ex*bx + ey*by + ez*bz
        ib2 = 1.0 / (bx**2 + by**2 + bz**2)
        # vdx = (ey*bz - ez*by) * ib2
        # vdy = (ez*bx - ex*bz) * ib2
        # vdz = (ex*by - ey*bx) * ib2
        # vd = np.sqrt(vdx**2 + vdy**2 + vdz**2)
        # gvd = 1.0 / np.sqrt(1 - vd**2)
        # gamma_p = gvd * gamma * (1.0 - vdx*vx - vdy*vy - vdz*vz)
        # gamma_smooth = signal.convolve(gamma, kernel, mode='same')
        eparax = edotb * bx * ib2
        eparay = edotb * by * ib2
        eparaz = edotb * bz * ib2
        wtot = np.cumsum(-(ex*vx + ey*vy + ez*vz)) * dtwpe_tracer
        wpara = np.cumsum(-(eparax * vx + eparay * vy + eparaz * vz)) * dtwpe_tracer
        wperp = wtot - wpara
        fig = plt.figure(figsize=[13, 7])
        rect0 = [0.07, 0.56, 0.3, 0.4]
        rect = np.copy(rect0)
        hgap, vgap = 0.07, 0.08
        ax = fig.add_axes(rect)
        COLORS = palettable.tableau.Tableau_10.mpl_colors
        ax.set_prop_cycle('color', COLORS)
        ax.plot(ttracer, wpara, linewidth=2, label=r'$W_\parallel$')
        ax.plot(ttracer, wperp, linewidth=2, label=r'$W_\perp$')
        # ax.plot(ttracer, wpara + wperp, linewidth=2,
        #         label=r'$W_\parallel + $' + r'$W_\perp$')
        ax.plot(ttracer, gamma, linewidth=2, label=r'$\Delta\gamma$')
        # ax.plot(ttracer, gamma_smooth-gamma0, linewidth=2, label=r'$\Delta\gamma^\prime$')
        ylim = ax.get_ylim()
        ax.plot([ttracer[tesc], ttracer[tesc]], ylim, color='k',
                linewidth=1, linestyle='--')
        ax.set_xlim([tmin, tmax])
        ax.set_ylim(ylim)
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')
        ax.set_xlabel(r'$t\omega_{pe}$', fontsize=16)
        ax.set_ylabel('Energy change', fontsize=16)
        ax.tick_params(labelsize=12)
        ax.legend(loc=6, prop={'size': 12}, ncol=1,
                 shadow=False, fancybox=False, frameon=False)

        rect[0] += rect[2] + hgap
        rect[2] = 0.5
        gamma_norm = gamma / gamma[0]
        ax = fig.add_axes(rect)
        ax.set_prop_cycle('color', COLORS)
        points = np.array([x, z]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        # Create a continuous norm to map from data points to colors
        norm = plt.Normalize(gamma_norm.min(), gamma_norm.max())
        lc = LineCollection(segments, cmap='jet', norm=norm)
        # Set the values used for colormapping
        lc.set_array(gamma_norm)
        lc.set_linewidth(2)
        ax.add_collection(lc)
        ax.plot([x[tesc]], [z[tesc]], marker='o', markersize=7, color="k")
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')
        ax.set_xlabel(r'$x$', fontsize=16)
        ax.set_ylabel(r'$z$', fontsize=16)
        ax.tick_params(labelsize=12)
        ax.set_xlim([0, lx_de])
        ax.set_ylim([-0.5*lz_de, 0.5*lz_de])
        # Colorbar
        rect_cbar = np.copy(rect)
        rect_cbar[0] = rect[0] + rect[2] + 0.01
        rect_cbar[2] = 0.01
        cax = fig.add_axes(rect_cbar)
        print("min and max of energy/thermal energy: %f, %f" %
              (gamma_norm.min(), gamma_norm.max()))
        sm = plt.cm.ScalarMappable(cmap=plt.cm.jet,
                                   norm=plt.Normalize(vmin=gamma_norm.min(),
                                                      vmax=gamma_norm.max()))
        cax.tick_params(axis='y', which='major', direction='in')
        sm._A = []
        cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
        cax.tick_params(labelrotation=90)
        cbar.set_label(r'$(\gamma-1)/\varepsilon_\text{th}$',
                       fontsize=16, labelpad=1)
        # ticks = np.linspace(0, te_spect, 6) * dtf
        # ticks = np.concatenate(([10], ticks))
        # cbar.set_ticks(ticks)
        # cax.tick_params(labelrotation=90)
        # # cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(),
        # #                         rotation='vertical')
        cbar.ax.tick_params(labelsize=12)
        cax.yaxis.set_ticks_position('right')
        cax.yaxis.set_label_position('right')

        rect = np.copy(rect0)
        rect[1] -= rect[3] + vgap
        rect[2] = 0.25
        ax = fig.add_axes(rect)
        ax.set_prop_cycle('color', COLORS)
        ax.plot(x, dgamma, linewidth=2)
        ax.plot([x[tesc]], [gamma[tesc]], marker='o', markersize=7, color="k")
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')
        ax.set_xlabel(r'$x$', fontsize=16)
        ax.set_ylabel(r'$\Delta\gamma$', fontsize=16)
        ax.tick_params(labelsize=12)

        rect[0] += rect[2] + hgap
        ax = fig.add_axes(rect)
        ax.set_prop_cycle('color', COLORS)
        ax.plot(ttracer, x, linewidth=2)
        ax.plot([ttracer[tesc]], [x[tesc]], marker='o', markersize=7, color="k")
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')
        ax.set_xlabel(r'$t\omega_{pe}$', fontsize=16)
        ax.set_ylabel(r'$x$', fontsize=16)
        ax.tick_params(labelsize=12)

        fname = img_dir + sname + "_tracer_" + str(iptl) + ".pdf"
        fig.savefig(fname)

        # plt.close()
        plt.show()

    fh.close()


def adjust_pos(pos, length):
    """Adjust position for periodic boundary conditions

    Args:
        pos: the position along one axis
        length: the box size along that axis
    """
    crossings = []
    offsets = []
    offset = 0
    nt, = pos.shape
    pos_b = np.zeros(nt)
    pos_b = np.copy(pos)
    for i in range(nt - 1):
        if (pos[i] - pos[i + 1] > 0.1 * length):
            crossings.append(i)
            offset += length
            offsets.append(offset)
        if (pos[i] - pos[i + 1] < -0.1 * length):
            crossings.append(i)
            offset -= length
            offsets.append(offset)
    nc = len(crossings)
    if nc > 0:
        crossings = np.asarray(crossings)
        offsets = np.asarray(offsets)
        for i in range(nc - 1):
            pos_b[crossings[i] + 1:crossings[i + 1] + 1] += offsets[i]
        pos_b[crossings[nc - 1] + 1:] += offsets[nc - 1]
    return pos_b


def plot_trajectory_j(plot_config, show_plot=True):
    """Plot particle trajectory with current density
    """
    species = plot_config["species"]
    pic_run = "sigma64_bg005_4000de_triggered"
    tracer_dir = "/net/scratch3/xiaocan/vpic-sorter/data/power_law_index/"
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_run_dir = pic_info.run_dir
    dtwpe_tracer = pic_info.dtwpe * pic_info.tracer_interval
    wce_wpe = pic_info.dtwce / pic_info.dtwpe
    smime = math.sqrt(pic_info.mime)
    lx_de = smime * pic_info.lx_di
    lz_de = smime * pic_info.lz_di
    vpic_info = get_vpic_info(pic_run_dir)
    fields_interval = int(vpic_info["fields_interval"])
    tracer_interval = int(vpic_info["tracer_interval"])
    fname = tracer_dir + pic_run + "/electrons_ntraj500_1emax.h5p"
    ticksize = 16
    labelsize = 20
    temp = pic_info.Te if species in ['e', 'electron'] else pic_info.Ti
    with h5py.File(fname, 'r') as fh:
        particle_tags = list(fh.keys())
        nptl = len(particle_tags)
        print("Total number of particles: %d" % nptl)
        group = fh[particle_tags[0]]
        dset = group['dX']
        nframes, = dset.shape
        ptl = {}
        for dset_name in group:
            dset = group[dset_name]
            ptl[str(dset_name)] = np.zeros(dset.shape, dset.dtype)
            dset.read_direct(ptl[str(dset_name)])
        gamma0 = 1.0 / np.sqrt(1 + ptl["Ux"]**2 + ptl["Uy"]**2 + ptl["Uz"]**2)
        ttracer = np.arange(0, nframes) * dtwpe_tracer
        tmin, tmax = ttracer[0], ttracer[-1]

        iptl = 268
        print("Particle: %d of %d" % (iptl, nptl))
        group = fh[particle_tags[iptl]]
        for dset_name in group:
            dset = group[dset_name]
            dset.read_direct(ptl[str(dset_name)])
    gamma = np.sqrt(1 + ptl["Ux"]**2 + ptl["Uy"]**2 + ptl["Uz"]**2)
    tesc = np.argmax(gamma > 0.9*gamma[-1])
    tesc_field = int(tesc * tracer_interval // fields_interval) + 1
    dgamma = gamma - gamma[0]
    igamma = 1.0 / gamma
    vx = ptl["Ux"] * igamma
    vy = ptl["Uy"] * igamma
    vz = ptl["Uz"] * igamma
    smime = math.sqrt(pic_info.mime)
    x = adjust_pos(ptl['dX'], pic_info.lx_di * smime)
    y = adjust_pos(ptl['dY'], pic_info.ly_di * smime)
    z = adjust_pos(ptl['dZ'], pic_info.lz_di * smime)
    ex = ptl["Ex"]
    ey = ptl["Ey"]
    ez = ptl["Ez"]
    bx = ptl["Bx"]
    by = ptl["By"]
    bz = ptl["Bz"]
    edotb = ex*bx + ey*by + ez*bz
    ib2 = 1.0 / (bx**2 + by**2 + bz**2)
    eparax = edotb * bx * ib2
    eparay = edotb * by * ib2
    eparaz = edotb * bz * ib2
    wtot = np.cumsum(-(ex*vx + ey*vy + ez*vz)) * dtwpe_tracer
    wpara = np.cumsum(-(eparax * vx + eparay * vy + eparaz * vz)) * dtwpe_tracer
    wperp = wtot - wpara

    fig = plt.figure(figsize=[6, 8])
    rect0 = [0.16, 0.7, 0.70, 0.2625]
    hgap, vgap = 0.04, 0.06
    rect = np.copy(rect0)
    tindex = tesc_field * fields_interval
    kwargs = {"current_time": tesc_field,
              "xl": 0, "xr": pic_info.lx_di,
              "zb": -pic_info.lz_di*0.5, "zt": pic_info.lz_di*0.5}
    fname = pic_run_dir + "data/absJ.gda"
    xgrid, zgrid, absj = read_2d_fields(pic_info, fname, **kwargs)
    nz, nx = absj.shape
    gamma_norm = gamma / temp
    ax = fig.add_axes(rect)
    nxl = nx*10//32
    fdata = np.roll(absj, nxl, axis=1)
    im1 = ax.imshow(fdata[:, :],
                    extent=[-lx_de*nxl/nx, lx_de-lx_de*nxl/nx,
                            -0.5*lz_de, 0.5*lz_de],
                    vmin=0, vmax=2.5,
                    cmap=plt.cm.Greys, aspect='auto',
                    origin='lower', interpolation='bicubic')
    points = np.array([x, z]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(gamma.min(), gamma.max())
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    # Set the values used for colormapping
    lc.set_array(gamma)
    lc.set_linewidth(2)
    ax.add_collection(lc)
    ax.plot([x[tesc]], [z[tesc]], marker='o', markersize=7, color="k")
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    # ax.set_xlabel(r'$x/d_e$', fontsize=labelsize)
    ax.set_ylabel(r'$z/d_e$', fontsize=labelsize)
    ax.tick_params(labelsize=ticksize)
    ax.set_xlim([-lx_de*nxl/nx, lx_de-lx_de*nxl/nx])
    ax.set_ylim([-0.5*lz_de, 0.5*lz_de])
    # Colorbar
    rect_cbar = np.copy(rect)
    rect_cbar[0] = rect[0] + rect[2] + 0.02
    rect_cbar[2] = 0.02
    rect_cbar[1] = rect[1] + rect[3] / 8
    rect_cbar[3] = rect[3] * 0.75
    cax = fig.add_axes(rect_cbar)
    print("min and max of energy/thermal energy: %f, %f" %
          (gamma_norm.min(), gamma_norm.max()))
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                               norm=plt.Normalize(vmin=gamma_norm.min(),
                                                  vmax=gamma_norm.max()))
    cax.tick_params(axis='y', which='major', direction='in')
    sm._A = []
    cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
    cax.tick_params(labelrotation=90)
    cbar.set_label(r'$\gamma/T_e$',
                   fontsize=labelsize, labelpad=1)
    cbar.ax.tick_params(labelsize=ticksize)
    cax.yaxis.set_ticks_position('right')
    cax.yaxis.set_label_position('right')

    ax.text(-0.2, 0.9, "(a)",
            color='k', fontsize=20,
            bbox=dict(facecolor='none', alpha=1.0,
                      edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes)

    rect[0] += rect[2] * 0.67
    rect[1] += rect[3] * 0.45
    rect[2] *= 0.3
    rect[3] *= 0.5
    ax = fig.add_axes(rect)
    ax.patch.set_alpha(0.8)
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    ax.set_prop_cycle('color', COLORS)
    ax.plot(ttracer, gamma_norm, linewidth=1,
            label=r'$\gamma/T_e$')
    ylim = ax.get_ylim()
    ax.plot([ttracer[tesc], ttracer[tesc]], ylim, color='k',
            linewidth=1, linestyle='--')
    ax.set_xlim([tmin, tmax])
    ax.set_ylim(ylim)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlabel(r'$t\omega_{pe}$', fontsize=16)
    ax.set_ylabel(r'$\gamma/T_e$', fontsize=16)
    ax.tick_params(labelsize=12)

    rect = np.copy(rect0)
    rect = np.copy(rect0)
    rect[3] = 0.525
    rect[1] -= rect[3] + vgap
    pic_run = "mime1_sigmae100_vthe04_db1_1024de"
    tracer_dir = '../data/power_law_index/trajectory/' + pic_run + '/'
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_run_dir = pic_info.run_dir
    dtwpe_tracer = pic_info.dtwpe * pic_info.tracer_interval
    wce_wpe = pic_info.dtwce / pic_info.dtwpe
    smime = math.sqrt(pic_info.mime)
    lx_de = smime * pic_info.lx_di
    lz_de = smime * pic_info.lz_di
    vpic_info = get_vpic_info(pic_run_dir)
    fields_interval = int(vpic_info["fields_interval"])
    tracer_interval = int(vpic_info["tracer_interval"])
    fname = tracer_dir + 'electron_traj_band0.h5'
    temp = pic_info.Te if species in ['e', 'electron'] else pic_info.Ti
    with h5py.File(fname, 'r') as fh:
        particle_tags = list(fh.keys())
        nptl = len(particle_tags)
        print("Total number of particles: %d" % nptl)
        group = fh[particle_tags[0]]
        dset = group['dX']
        nframes, = dset.shape
        ptl = {}
        for dset_name in group:
            dset = group[dset_name]
            ptl[str(dset_name)] = np.zeros(dset.shape, dset.dtype)
            dset.read_direct(ptl[str(dset_name)])
        gamma0 = 1.0 / np.sqrt(1 + ptl["Ux"]**2 + ptl["Uy"]**2 + ptl["Uz"]**2)
        ttracer = np.arange(0, nframes) * dtwpe_tracer
        tmin, tmax = ttracer[0], ttracer[-1]

        iptl = 91
        print("Particle: %d of %d" % (iptl, nptl))
        group = fh[particle_tags[iptl]]
        for dset_name in group:
            dset = group[dset_name]
            dset.read_direct(ptl[str(dset_name)])
    gamma = np.sqrt(1 + ptl["Ux"]**2 + ptl["Uy"]**2 + ptl["Uz"]**2)
    tesc = np.argmax(gamma > 0.9*gamma[-1])
    tesc_field = int(tesc * tracer_interval // fields_interval) + 1
    dgamma = gamma - gamma[0]
    igamma = 1.0 / gamma
    vx = ptl["Ux"] * igamma
    vy = ptl["Uy"] * igamma
    vz = ptl["Uz"] * igamma
    x = ptl["dX"]
    y = ptl["dY"]
    z = ptl["dZ"]
    ex = ptl["Ex"]
    ey = ptl["Ey"]
    ez = ptl["Ez"]
    bx = ptl["Bx"]
    by = ptl["By"]
    bz = ptl["Bz"]
    edotb = ex*bx + ey*by + ez*bz
    ib2 = 1.0 / (bx**2 + by**2 + bz**2)
    eparax = edotb * bx * ib2
    eparay = edotb * by * ib2
    eparaz = edotb * bz * ib2
    wtot = np.cumsum(-(ex*vx + ey*vy + ez*vz)) * dtwpe_tracer
    wpara = np.cumsum(-(eparax * vx + eparay * vy + eparaz * vz)) * dtwpe_tracer
    wperp = wtot - wpara

    tindex = tesc_field * fields_interval
    fname = (pic_run_dir + "hydro_hdf5/T." + str(tindex) +
             "/hydro_electron_" + str(tindex) + ".h5")
    je = {}
    with h5py.File(fname, 'r') as fh:
        group = fh["Timestep_" + str(tindex)]
        for var in ["jx", "jy", "jz"]:
            dset = group[var]
            je[var] = np.zeros(dset.shape, dtype=dset.dtype)
            dset.read_direct(je[var])
    fname = (pic_run_dir + "hydro_hdf5/T." + str(tindex) +
             "/hydro_ion_" + str(tindex) + ".h5")
    ji = {}
    with h5py.File(fname, 'r') as fh:
        group = fh["Timestep_" + str(tindex)]
        for var in ["jx", "jy", "jz"]:
            dset = group[var]
            ji[var] = np.zeros(dset.shape, dtype=dset.dtype)
            dset.read_direct(ji[var])

    absj = np.squeeze(np.sqrt((je["jx"] + ji["jx"])**2 +
                              (je["jy"] + ji["jy"])**2 +
                              (je["jz"] + ji["jz"])**2))
    jy = np.squeeze(je["jy"] + ji["jy"])
    nz, nx = absj.shape
    gamma_norm = gamma / temp
    ax = fig.add_axes(rect)
    im1 = ax.imshow(absj[:nx//2, nz//2:].T,
                    extent=[0, 0.5*lx_de, 0, 0.5*lz_de],
                    vmin=0.5, vmax=2.5,
                    cmap=plt.cm.Greys, aspect='auto',
                    origin='lower', interpolation='bicubic')
    points = np.array([x, z]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(gamma.min(), gamma.max())
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    # Set the values used for colormapping
    lc.set_array(gamma)
    lc.set_linewidth(2)
    ax.add_collection(lc)
    ax.plot([x[tesc]], [z[tesc]], marker='o', markersize=7, color="k")
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlabel(r'$x/d_e$', fontsize=labelsize)
    ax.set_ylabel(r'$z/d_e$', fontsize=labelsize)
    ax.tick_params(labelsize=ticksize)
    ax.set_xlim([0, 0.5*lx_de])
    ax.set_ylim([0, 0.5*lz_de])
    # Colorbar
    rect_cbar = np.copy(rect)
    rect_cbar[0] = rect[0] + rect[2] + 0.02
    rect_cbar[2] = 0.02
    rect_cbar[1] = rect[1] + rect[3] / 8
    rect_cbar[3] = rect[3] * 0.75
    cax = fig.add_axes(rect_cbar)
    print("min and max of energy/thermal energy: %f, %f" %
          (gamma_norm.min(), gamma_norm.max()))
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                               norm=plt.Normalize(vmin=gamma_norm.min(),
                                                  vmax=gamma_norm.max()))
    cax.tick_params(axis='y', which='major', direction='in')
    sm._A = []
    cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
    cax.tick_params(labelrotation=90)
    cbar.set_label(r'$\gamma/T_e$',
                   fontsize=labelsize, labelpad=1)
    cbar.ax.tick_params(labelsize=ticksize)
    cax.yaxis.set_ticks_position('right')
    cax.yaxis.set_label_position('right')

    ax.text(-0.2, 0.9, "(b)",
            color='k', fontsize=20,
            bbox=dict(facecolor='none', alpha=1.0,
                      edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes)

    rect[0] += rect[2] * 0.67
    rect[1] += rect[3] * 0.13
    rect[2] *= 0.3
    rect[3] *= 0.25
    ax = fig.add_axes(rect)
    ax.patch.set_alpha(0.8)
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    ax.set_prop_cycle('color', COLORS)
    ax.plot(ttracer, gamma_norm, linewidth=1, label=r'$\Delta\gamma$')
    ylim = ax.get_ylim()
    ax.plot([ttracer[tesc], ttracer[tesc]], ylim, color='k',
            linewidth=1, linestyle='--')
    ax.set_xlim([tmin, tmax])
    ax.set_ylim(ylim)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlabel(r'$t\omega_{pe}$', fontsize=16)
    ax.set_ylabel(r'$\gamma/T_e$', fontsize=16)
    ax.tick_params(labelsize=12)

    img_dir = '../img/power_law_index/tracer_traj/'
    mkdir_p(img_dir)
    if species in ["e", "electron"]:
        sname = "electron"
    else:
        sname = "Ion"
    fname = img_dir + sname + "_traj_runs_2" + ".pdf"
    fig.savefig(fname)

    plt.show()


def read_var(group, dset_name, sz):
    """Read data from a HDF5 group

    Args:
        group: one HDF5 group
        var: the dataset name
        sz: the size of the data
    """
    dset = group[dset_name]
    fdata = np.zeros(sz, dtype=dset.dtype)
    dset.read_direct(fdata)
    return fdata


def calc_rates_tracer(plot_config, show_plot=True):
    """
    Calculate particle acceleration and escape rates using tracer particles

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
    if pic_run == "turbulent-sheet3D-mixing-sigma100":
        dtwpe = 9.866826e-02
        dtwpe_tracer = dtwpe * 80
        pic_run_dir = "/net/scratch3/guofan/trinity/turbulent-sheet3D-mixing-sigma100/"
        tracer_dir = pic_run_dir + 'tracer/'
    else:
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        if "3D-Lx150" in pic_run:
            pic_run_dir = "/net/scratch3/xiaocan/reconnection/Cori_runs/" + pic_run + "/"
        else:
            pic_run_dir = pic_info.run_dir
        dtwpe = pic_info.dtwpe
        dtwpe_tracer = dtwpe * pic_info.tracer_interval
        tracer_dir = pic_run_dir + 'tracer/tracer1/'

    file_list = os.listdir(tracer_dir)
    tframes = []
    if pic_run == "turbulent-sheet3D-mixing-sigma100":
        fname = tracer_dir + 'T.0/electron_tracer_sorted.h5p'
        with h5py.File(fname, 'r') as fh:
            group = fh["Step#0"]
            dataset = group["dX"]
            nptl0, = dataset.shape
        for file_name in file_list:
            if "T." in file_name:
                fsplit = file_name.split(".")
                tindex = int(fsplit[-1])
                fname = tracer_dir + 'T.' + str(tindex) + '/electron_tracer_sorted.h5p'
                if os.path.isfile(fname):
                    with h5py.File(fname, 'r') as fh:
                        group = fh["Step#" + str(tindex)]
                        nkeys = len(group.keys())
                        dataset = group["dX"]
                        nptl, = dataset.shape
                        if nptl == nptl0 and nkeys == 17:
                            tframes.append(tindex)
    else:
        for file_name in file_list:
            fsplit = file_name.split(".")
            tindex = int(fsplit[-1])
            tframes.append(tindex)
    tframes = np.sort(np.asarray(tframes))
    nfiles = len(tframes)

    if species in ["e", "electron"]:
        sname = "electron"
        pmass = 1.0
        pcharge = -1.0
    else:
        sname = "H"
        pcharge = 1.0
        if pic_run == "turbulent-sheet3D-mixing-sigma100":
            pmass = 1.0
        else:
            pmass = pic_info.mime

    # Check how many steps in each file
    if pic_run == "turbulent-sheet3D-mixing-sigma100":
        fname = tracer_dir + 'T.0/electron_tracer_sorted.h5p'
    else:
        fname = tracer_dir + 'T.0/electron_tracer_qtag_sorted.h5p'
    with h5py.File(fname, 'r') as fh:
        groups = list(fh.keys())
        nsteps_file = len(groups)
        tsteps = [int(group[5:]) for group in groups]
        tsteps.sort()
    if len(tsteps) > 1:
        tinterval = tsteps[1] - tsteps[0]
    else:
        nsteps_file = 1
        tinterval = tframes[1] - tframes[0]
        tinterval_file = tinterval

    # Last time step
    tmax_dir = tframes[-1]
    if pic_run == "turbulent-sheet3D-mixing-sigma100":
        fname = tracer_dir + 'T.' + str(tmax_dir) + '/electron_tracer_sorted.h5p'
    else:
        fname = tracer_dir + 'T.' + str(tmax_dir) + '/electron_tracer_qtag_sorted.h5p'
    with h5py.File(fname, 'r') as fh:
        groups = list(fh.keys())
        tsteps = [int(group[5:]) for group in groups]
        tsteps.sort()
        tmax = tsteps[-1]
        group = fh['Step#' + str(tmax)]
        dset = group['dX']
        nptl, = dset.shape
        ptl = {}
        for dset in group:
            dset = str(dset)
            ptl[str(dset)] = read_var(group, dset, nptl)
    ene_final = np.sqrt(1 + ptl["Ux"]**2 + ptl["Uy"]**2 + ptl["Uz"]**2) - 1
    spect_params = get_spect_params(pic_run)
    emin, emax = spect_params["energy_range"]
    if "3D-Lx150" in pic_run:
        vthe = pic_info.vthe
        gama = 1.0 / math.sqrt(1.0 - 3 * vthe**2)
        temp = gama - 1.0
    elif pic_run == "turbulent-sheet3D-mixing-sigma100":
        temp = 1.0
    else:
        temp = pic_info.Te if species in ['e', 'electron'] else pic_info.Ti
    # cond_energetic_final = ene_final > (emin * temp)

    if pic_run == "turbulent-sheet3D-mixing-sigma100":
        nframes = len(tframes)
    else:
        nframes = tmax // tinterval + 1

    escaped_all = np.zeros(nptl, dtype=bool)
    acc_high = np.zeros(nptl, dtype=bool)
    threshold = 0.9  # we treat particles to escape when reaching this*final_gamma
    dset_names = ['Ux', 'Uy', 'Uz', 'Ex', 'Ey', 'Ez']
    if "3D-Lx150" in pic_run:
        nbins = 16
    else:
        nbins = 20
    acc_rate_sum = np.zeros([nframes, nbins])
    acc_rate_esc = np.zeros([nframes, nbins])
    nptl_acc = np.zeros([nframes, nbins])
    nptl_esc = np.zeros([nframes, nbins])
    dnptl_esc = np.zeros([nframes, nbins])
    ebins = np.logspace(math.log10(emin*temp/2), math.log10(emax*temp*2), nbins+1)

    if pic_run == "turbulent-sheet3D-mixing-sigma100":
        tracer_interval = 10
    else:
        tracer_interval = pic_info.tracer_interval
    dee_interval = get_time_interval(pic_run)
    for ifile, tindex_file in enumerate(tframes):
        print("File %d of %d" % (ifile, nfiles))
        if pic_run == "turbulent-sheet3D-mixing-sigma100":
            fname = tracer_dir + 'T.' + str(tindex_file) + '/' + sname + '_tracer_sorted.h5p'
        else:
            fname = tracer_dir + 'T.' + str(tindex_file) + '/' + sname + '_tracer_qtag_sorted.h5p'
        with h5py.File(fname, 'r') as fh:
            for istep in range(nsteps_file):
                tindex = tindex_file + istep * tracer_interval
                tframe = ifile * nsteps_file + istep
                if tindex > tmax:
                    break
                gname = 'Step#' + str(tindex)
                group = fh[gname]
                for dset_name in dset_names:
                    dset = group[dset_name]
                    dset.read_direct(ptl[dset_name])
                gamma = np.sqrt(1.0 + ptl["Ux"]**2 + ptl["Uy"]**2 + ptl["Uz"]**2)
                ene = gamma - 1
                cond_escape = ene > (ene_final * threshold)  # Close to final energy
                # Particle should be energetic in the end
                # cond_escape = np.logical_and(cond_escape, cond_energetic_final)
                escaped_new = np.logical_and(cond_escape, np.logical_not(escaped_all))
                escaped_all = np.logical_or(cond_escape, escaped_all)
                # Particle in acceleration regions should be energetic.
                # They cannot not be escaped particles at the same time.
                cond_energetic = ene > (emin * temp/2)  # Energetic enough
                # cond_energetic = np.logical_and(cond_energetic, cond_energetic_final)
                acc_high = np.logical_and(cond_energetic, np.logical_not(escaped_all))
                dene = ptl["Ux"] * ptl["Ex"] + ptl["Uy"] * ptl["Ey"] + ptl["Uz"] * ptl["Ez"]
                dene /= gamma * pcharge
                for ibin in range(nbins):
                    cond_bin = np.logical_and(ene > ebins[ibin], ene <= ebins[ibin+1])
                    cond = np.logical_and(escaped_new, cond_bin)
                    dnptl_esc[tframe, ibin] = np.sum(cond)
                    cond = np.logical_and(escaped_all, cond_bin)
                    nptl_esc[tframe, ibin] = np.sum(cond)
                    acc_rate_esc[tframe, ibin] = np.sum(dene[cond] / ene[cond])
                    cond = np.logical_and(acc_high, cond_bin)
                    nptl_acc[tframe, ibin] = np.sum(cond)
                    acc_rate_sum[tframe, ibin] = np.sum(dene[cond] / ene[cond])

                # High-energy particles for calculating energy diffusion
                if tframe % dee_interval == 0:
                    ibin_max = -1
                    ptl_indices = np.zeros((nbins, nptl), dtype=bool)
                    for ibin in range(nbins):
                        ptl_indices[ibin] = np.logical_and(ene > ebins[ibin],
                                                           ene < ebins[ibin+1])
                        if np.sum(ptl_indices[ibin]) > 0:
                            ibin_max = ibin
                    tstart = tframe
                    gamma_avg = np.zeros([nbins, dee_interval])
                    dgamma = np.zeros([nbins, dee_interval])
                for ibin in range(0, ibin_max+1):
                    # cond = np.logical_and(np.logical_not(escaped_all), ptl_indices[ibin])
                    # gamma_selected = gamma[cond]
                    # if np.sum(cond) > 0:
                    gamma_selected = gamma[ptl_indices[ibin]]
                    if len(gamma_selected) > 0:
                        gamma_avg[ibin, tframe-tstart] = np.mean(gamma_selected)
                        if len(gamma_selected) == 1:
                            dgamma[ibin, tframe-tstart] = 0.0
                        else:
                            dtmp = np.mean(gamma_selected**2) - gamma_avg[ibin, tframe-tstart]**2
                            if dtmp > 0:
                                dgamma[ibin, tframe-tstart] = math.sqrt(dtmp)
                            else:
                                dgamma[ibin, tframe-tstart] = 0.0

                if (tframe + 1) % dee_interval == 0 or tindex == tmax:
                    fdir = '../data/power_law_index/rates_tracer/' + pic_run + '/'
                    mkdir_p(fdir)
                    fname = fdir + 'gamma_avg_' + str(tframe//dee_interval) + '.dat'
                    gamma_avg.tofile(fname)
                    fname = fdir + 'dgamma_' + str(tframe//dee_interval) + '.dat'
                    dgamma.tofile(fname)

    fdir = '../data/power_law_index/rates_tracer/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + 'nptl_esc.dat'
    nptl_esc.tofile(fname)
    fname = fdir + 'dnptl_esc.dat'
    dnptl_esc.tofile(fname)
    fname = fdir + 'nptl_acc.dat'
    nptl_acc.tofile(fname)
    fname = fdir + 'acc_rate_sum.dat'
    acc_rate_sum.tofile(fname)
    fname = fdir + 'acc_rate_esc.dat'
    acc_rate_esc.tofile(fname)
    fname = fdir + 'ebins.dat'
    ebins.tofile(fname)
    twpe = tframes * dtwpe
    fname = fdir + 'twpe.dat'
    twpe.tofile(fname)


def test_calc_rates(plot_config, show_plot=True):
    """
    Test the method to calculate acceleration and escape rates
    """
    acc_rate = 1E-3
    esc_rate = 1.5E-3
    dee_rate = 1E-3
    nsteps = 4000
    nptl0 = 0
    ninj = int(1E5 / nsteps)
    dt = 1.0
    nptl = nptl0 + nsteps*ninj
    ptl = np.zeros(nptl)
    escaped_all = np.zeros(nptl, dtype=bool)
    esc_cond = np.zeros(nptl, dtype=bool)
    p0 = 1.0
    ptl[:nptl0] = p0
    dene = np.ones(nptl) * acc_rate
    nbins = 40
    ebins = np.logspace(-1, 3, nbins+1)
    ebins_mid = 0.5 * (ebins[:-1] + ebins[1:])
    nptl_acc = np.zeros([nsteps, nbins])
    nptl_esc = np.zeros([nsteps, nbins])
    dnptl_esc = np.zeros([nsteps, nbins])
    acc_rate_sum = np.zeros([nsteps, nbins])
    for tframe in range(nsteps):
        print("Step: %d" % tframe)
        nptl_old = nptl0 + tframe * ninj
        esc_cond[:nptl_old] = np.random.choice(a=[False, True], size=nptl_old,
                                               p=[1-esc_rate, esc_rate])
        escaped_new = np.logical_and(esc_cond, np.logical_not(escaped_all))
        escaped_all = np.logical_or(esc_cond, escaped_all)
        acc_cond = np.logical_not(escaped_all)
        nptl_new = nptl_old + ninj
        ptl[nptl_old:nptl_new] = p0
        dwt = np.random.normal(0, math.sqrt(dt), nptl)
        ptl[acc_cond] += (acc_rate - 0.5*acc_rate*np.log(ptl[acc_cond])) * ptl[acc_cond] * dt
        for ibin in range(nbins):
            cond_bin = np.logical_and(ptl > ebins[ibin], ptl <= ebins[ibin+1])
            cond = np.logical_and(escaped_new, cond_bin)
            dnptl_esc[tframe, ibin] = np.sum(cond)
            cond = np.logical_and(escaped_all, cond_bin)
            nptl_esc[tframe, ibin] = np.sum(cond)
            cond = np.logical_and(acc_cond, cond_bin)
            nptl_acc[tframe, ibin] = np.sum(cond)
            acc_rate_sum[tframe, ibin] = np.sum(dene[cond] - 0.5*acc_rate*np.log(ptl[cond]) )

    # nptl_esc = gaussian_filter(nptl_esc, sigma=[5, 1])
    # nptl_acc = gaussian_filter(nptl_acc, sigma=[5, 1])
    # acc_rate_sum = gaussian_filter(acc_rate_sum, sigma=[5, 1])
    # dnptl_esc = gaussian_filter(dnptl_esc+1E-10, sigma=[5, 1])

    acc_rate_sum_r = np.mean(acc_rate_sum, axis=0)
    nptl_acc_r = np.mean(nptl_acc, axis=0)
    nptl_esc_r = np.mean(nptl_esc, axis=0)
    dnptl_esc_r = np.mean(dnptl_esc, axis=0)
    acc_rate_r = div0(acc_rate_sum_r, nptl_acc_r)
    acc_esc_r = div0(acc_rate_sum_r, nptl_esc_r)
    esc_rate2_r = div0(dnptl_esc_r, nptl_acc_r) / dt

    nbins = 100
    bins = np.logspace(-1, 3, nbins+1)
    bins_mid = 0.5 * (bins[:-1] + bins[1:])
    hist, _ = np.histogram(ptl, bins=bins)
    espect = hist/np.diff(bins)
    es, _ = find_nearest(bins_mid, 2.0)
    ee, _ = find_nearest(bins_mid, 10.0)
    popt, _ = curve_fit(func_power, bins_mid[es:ee], espect[es:ee])
    pindex = popt[0]

    fpower = bins_mid[es:ee]**pindex
    fpower *= espect[es] * 3 / fpower[0]
    power_index = "{%0.2f}" % pindex
    pname = r'$\propto \varepsilon^{' + power_index + '}$'

    fig = plt.figure(figsize=[7, 8])
    rect = [0.15, 0.55, 0.8, 0.43]
    vgap = 0.02
    ax = fig.add_axes(rect)
    ax.loglog(bins_mid, espect)
    ax.loglog(bins_mid[es:ee], fpower, label=pname, linestyle='--',
              color='k')
    ax.legend(loc=1, prop={'size': 16}, ncol=1,
              shadow=False, fancybox=False, frameon=False)
    ax.tick_params(bottom=True, top=False, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top=True)
    ax.tick_params(axis='x', which='major', direction='in', top=True)
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlabel(r'$\varepsilon$', fontsize=16)
    ax.set_ylabel(r'$f(\varepsilon)$', fontsize=16)
    ax.tick_params(labelsize=16)

    rect[1] -= rect[3] + vgap
    ax = fig.add_axes(rect)
    ax.set_prop_cycle('color', COLORS)
    ax.semilogx(ebins_mid, acc_rate_r, label=r"$\alpha_\text{acc}$")
    ax.semilogx(ebins_mid, esc_rate2_r, linewidth=2, label=r"$\alpha_\text{esc}$")
    ax.legend(loc=1, prop={'size': 16}, ncol=1,
              shadow=False, fancybox=False, frameon=False)
    ax.set_xlim(ebins[0], ebins[-1])
    ax.tick_params(bottom=True, top=True, left=True, right=False)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_ylabel('Rates', fontsize=20)
    ax.tick_params(axis='x', labelbottom=False)
    ax.tick_params(labelsize=16)
    plt.show()


def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]


def plot_rates_tracer(plot_config, show_plot=True):
    """
    Plot particle acceleration and escape rates using tracer particles

    """
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
    if pic_run == "turbulent-sheet3D-mixing-sigma100":
        dtwpe_tracer = 9.866826e-02 * 80
        pic_run_dir = "/net/scratch3/guofan/trinity/turbulent-sheet3D-mixing-sigma100/"
        tracer_dir = pic_run_dir + 'tracer/'
    else:
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        if "3D-Lx150" in pic_run:
            pic_run_dir = "/net/scratch3/xiaocan/reconnection/Cori_runs/" + pic_run + "/"
        else:
            pic_run_dir = pic_info.run_dir
        dtwpe_tracer = pic_info.dtwpe * pic_info.tracer_interval
        tracer_dir = pic_run_dir + 'tracer/tracer1/'

    if "3D-Lx150" in pic_run:
        vthe = pic_info.vthe
        gama = 1.0 / math.sqrt(1.0 - 3 * vthe**2)
        temp = gama - 1.0
    elif pic_run == "turbulent-sheet3D-mixing-sigma100":
        temp = 1.0
    else:
        temp = pic_info.Te if species in ['e', 'electron'] else pic_info.Ti

    fdir = '../data/power_law_index/rates_tracer/' + pic_run + '/'
    fname = fdir + 'ebins.dat'
    ebins = np.fromfile(fname)
    ebins /= temp
    ebins_mid = 0.5 * (ebins[1:] + ebins[:-1])
    nbins, = ebins.shape
    fname = fdir + 'nptl_esc.dat'
    nptl_esc = np.fromfile(fname).reshape([-1, nbins-1])
    fname = fdir + 'dnptl_esc.dat'
    dnptl_esc = np.fromfile(fname).reshape([-1, nbins-1])
    fname = fdir + 'nptl_acc.dat'
    nptl_acc = np.fromfile(fname).reshape([-1, nbins-1])
    fname = fdir + 'acc_rate_sum.dat'
    acc_rate_sum = np.fromfile(fname).reshape([-1, nbins-1])
    fname = fdir + 'acc_rate_esc.dat'
    acc_rate_esc = np.fromfile(fname).reshape([-1, nbins-1])
    spect_params = get_spect_params(pic_run)
    power_index = spect_params["power_index"]

    nptl_esc = gaussian_filter(nptl_esc, sigma=[5, 1])
    nptl_acc = gaussian_filter(nptl_acc, sigma=[5, 1])
    acc_rate_sum = gaussian_filter(acc_rate_sum, sigma=[5, 1])
    dnptl_esc = gaussian_filter(dnptl_esc+1E-10, sigma=[5, 1])

    rfactor = get_time_interval(pic_run)
    nframes, _ = acc_rate_sum.shape
    n1 = (nframes // rfactor) * rfactor
    nfr = n1 // rfactor

#     acc_rate_sum_r = np.mean(acc_rate_sum[:n1].reshape([nfr, rfactor, -1]), axis=1)
#     acc_rate_esc_r = np.mean(acc_rate_esc[:n1].reshape([nfr, rfactor, -1]), axis=1)
#     nptl_acc_r = np.mean(nptl_acc[:n1].reshape([nfr, rfactor, -1]), axis=1)
#     nptl_esc_r = np.mean(nptl_esc[:n1].reshape([nfr, rfactor, -1]), axis=1)
#     dnptl_esc_r = np.mean(dnptl_esc[:n1].reshape([nfr, rfactor, -1]), axis=1)
#     acc_rate_r = div0(acc_rate_sum_r, nptl_acc_r)
#     acc_esc_r = div0(acc_rate_sum_r, nptl_esc_r)
#     esc_rate_r = div0(np.gradient(nptl_esc_r, axis=0), nptl_acc_r) / dtwpe_tracer / rfactor
#     esc_rate2_r = div0(dnptl_esc_r, nptl_acc_r) / dtwpe_tracer

    eend = nframes
    acc_rate_sum_r = np.mean(acc_rate_sum[:eend], axis=0)
    acc_rate_esc_r = np.mean(acc_rate_esc[:eend], axis=0)
    nptl_acc_r = np.mean(nptl_acc[:eend], axis=0)
    nptl_esc_r = np.mean(nptl_esc[:eend], axis=0)
    dnptl_esc_r = np.mean(dnptl_esc[:eend], axis=0)
    acc_rate_r = div0(acc_rate_sum_r, nptl_acc_r)
    acc_esc_r = div0(acc_rate_sum_r, nptl_esc_r)
    if pic_run == "turbulent-sheet3D-mixing-sigma100":
        esc_rate = div0(dnptl_esc, nptl_acc)
        fname = fdir + 'twpe.dat'
        twpe = np.fromfile(fname)
        for ibin in range(nbins-1):
            esc_rate[1:, ibin] /= np.diff(twpe)
        esc_rate2_r = np.mean(esc_rate[:eend], axis=0)
    else:
        esc_rate2_r = div0(dnptl_esc_r, nptl_acc_r) / dtwpe_tracer
    # plt.plot(np.sum(nptl_acc, axis=1))
    # # fpower = ebins_mid**power_index
    # # fpower *= nptl_acc_r[0] / fpower[0]
    # # plt.loglog(ebins_mid, fpower)
    # # plt.loglog(ebins_mid, dnptl_esc_r)
    # plt.show()

    vpic_info = get_vpic_info(pic_run_dir)
    emin = vpic_info["emin_spect"]
    emax = vpic_info["emax_spect"]
    nbins = int(vpic_info["nbins"])
    dloge = (math.log10(emax) - math.log10(emin)) / (nbins - 1)
    emin0 = 10**(math.log10(emin) - dloge)
    ebins_spect = np.logspace(math.log10(emin0), math.log10(emax), nbins+1)
    ebins_spect_mid = (ebins_spect[:-1] + ebins_spect[1:]) * 0.5
    debins_spect = np.diff(ebins_spect)
    tframe = pic_info.ntf - 1
    print("Time frame: %d" % tframe)
    tindex = tframe * int(vpic_info["spectrum_interval"])
    fdir = pic_run_dir + '/spectrum_combined/'
    sname = "electron" if species == 'e' else "ion"
    fname = fdir + 'spectrum_' + sname + '_' + str(tindex) + '.dat'
    ebins_spect_mid_run = ebins_spect_mid / temp
    debins_spect_run = debins_spect / temp
    if os.path.isfile(fname):
        flog = np.fromfile(fname, dtype=np.float32)
    nptot = pic_info.nx * pic_info.ny * pic_info.nz * pic_info.nppc
    espect = flog[3:] / debins_spect_run / nptot  # the first 3 are magnetic field
    pindex_e = -np.gradient(np.log10(espect+1E-15)) / np.gradient(np.log10(ebins_spect_mid))
    # pindex_e = gaussian_filter(pindex_e, sigma=5)

    fig = plt.figure(figsize=[7, 8])
    rect = [0.15, 0.55, 0.8, 0.43]
    vgap = 0.02
    ax = fig.add_axes(rect)
    ax.set_prop_cycle('color', COLORS)
    ax.semilogx(ebins_mid, acc_rate_r, label=r"$\alpha_\text{acc}$")
    fdata = (-power_index - 1) * acc_rate_r
    ax.semilogx(ebins_mid, fdata, linewidth=2, label=r"$(p-1)\alpha_\text{acc}$")
    ax.semilogx(ebins_mid, esc_rate2_r, linewidth=2, label=r"$\alpha_\text{esc}$")
    ax.legend(loc=1, prop={'size': 16}, ncol=1,
              shadow=False, fancybox=False, frameon=False)
    ax.set_xlim(ebins[0], ebins[-1])
    ax.tick_params(bottom=True, top=True, left=True, right=False)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_ylabel('Rates', fontsize=20)
    ax.tick_params(axis='x', labelbottom=False)
    ax.tick_params(labelsize=16)

    rect[1] -= rect[3] + vgap
    ax = fig.add_axes(rect)
    colors1 = palettable.tableau.Tableau_10.mpl_colors
    ax.set_prop_cycle('color', colors1)
    pindex = 1 + div0(esc_rate2_r, acc_rate_r)
    label = r"$1+(\alpha_\text{esc}/\alpha)$"
    ax.semilogx(ebins_mid, pindex, linewidth=2, label=label)
    pindex += (np.gradient(np.log(acc_rate_r)) /
               (math.log(ebins[1]) - math.log(ebins[0])))
    pindex -= (np.gradient(np.log(esc_rate2_r)) /
               (math.log(ebins[1]) - math.log(ebins[0])))
    label = (r"$1+(\alpha_\text{esc}/\alpha_\text{acc})+" +
             r"\partial\ln\alpha_\text{acc}/\partial\ln E-" +
             r"\partial\ln\alpha_\text{esc}/\partial\ln E$")
    ax.semilogx(ebins_mid, pindex, linewidth=2, label=label)
    emin, emax = spect_params["energy_range"]
    es, _ = find_nearest(ebins_mid, emin)
    ee, _ = find_nearest(ebins_mid, emax)
    pindex_avg = np.mean(pindex[es:ee])
    ax.semilogx([ebins[0], ebins[-1]], [pindex_avg, pindex_avg],
                linewidth=1, linestyle='--', color='k')
    ax.set_ylim([pindex_avg-1, pindex_avg+1])
    ylim = ax.get_ylim()
    ax.semilogx(ebins_spect_mid_run, pindex_e, linewidth=2)
    ax.plot([emin, emin], ylim, color='k', linewidth=0.5,
            linestyle='--')
    ax.plot([emax, emax], ylim, color='k', linewidth=0.5,
            linestyle='--')
    pname = "{%0.2f}" % pindex_avg
    ylim = ax.get_ylim()
    ypos = (pindex_avg - ylim[0]) / (ylim[1] - ylim[0])
    ax.text(0.95, ypos, pname, color='k', fontsize=12,
            bbox=dict(facecolor='none', alpha=1.0,
                      edgecolor='none', pad=10.0),
            horizontalalignment='right', verticalalignment='bottom',
            transform=ax.transAxes)
    ax.set_xlim(ebins[0], ebins[-1])
    ax.legend(loc=2, prop={'size': 16}, ncol=1,
              shadow=False, fancybox=False, frameon=False)
    ax.tick_params(bottom=True, top=True, left=True, right=False)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.tick_params(labelsize=16)
    ax.set_xlabel(r'$\gamma/T_e$', fontsize=20)
    ax.set_ylabel('Power-law index', fontsize=20)
    img_dir = '../img/power_law_index/power_index_tracer_overall/'
    mkdir_p(img_dir)
    fname = img_dir + "pindex_tracer_" + pic_run + ".pdf"
    fig.savefig(fname)

    fdir = '../data/power_law_index/power_index_tracer_overall/'
    mkdir_p(fdir)
    fdata = np.asarray([pindex_avg])
    fname = fdir + "pindex_tracer_" + pic_run + ".dat"
    fdata.tofile(fname)
    if show_plot:
        plt.show()
    else:
        plt.close()

    # acc_rate_bins = div0(acc_rate_sum, nptl_acc)
    # acc_esc_bins = div0(acc_rate_esc, nptl_esc)
    # acc_rate_r = np.mean(acc_rate_bins[:n1].reshape([nfr, rfactor, -1]), axis=1)
    # acc_rate_std = np.std(acc_rate_bins[:n1].reshape([nfr, rfactor, -1]), axis=1)
    # acc_esc_r = np.mean(acc_esc_bins[:n1].reshape([nfr, rfactor, -1]), axis=1)
    # fdata = np.gradient(nptl_esc, axis=0)
    # fdata -= (1-np.exp(acc_esc_bins*dtwpe_tracer)**-0.7) * nptl_esc
    # esc_rate_bins = div0(fdata, nptl_acc) / dtwpe_tracer
    # esc_rate_r = np.mean(esc_rate_bins[:n1].reshape([nfr, rfactor, -1]), axis=1)
    # esc_rate2_bins = div0(dnptl_esc, nptl_acc) / dtwpe_tracer
    # fdata = esc_rate2_bins[:n1].reshape([nfr, rfactor, -1])
    # esc_rate2_r = np.zeros([nfr, nbins-1])
    # esc_rate_std = np.zeros([nfr, nbins-1])
    # tdata = dnptl_esc[:n1].reshape([nfr, rfactor, -1])
    # esc_rate2_r = np.mean(esc_rate2_bins[:n1].reshape([nfr, rfactor, -1]), axis=1)
    # esc_rate_std = np.std(esc_rate2_bins[:n1].reshape([nfr, rfactor, -1]), axis=1)

    # dtfr = rfactor * dtwpe_tracer
    # for tframe in range(7, nfr):
    #     print("Frame %d of %d" % (tframe, nfr))
    #     fig = plt.figure(figsize=[7, 8])
    #     rect = [0.15, 0.55, 0.8, 0.43]
    #     vgap = 0.02
    #     ax = fig.add_axes(rect)
    #     ax.set_prop_cycle('color', COLORS)
    #     ax.semilogx(ebins_mid, acc_rate_r[tframe], label=r"$\alpha_\text{acc}$")
    #     fdata = (-power_index - 1) * acc_rate_r[tframe]
    #     ax.semilogx(ebins_mid, fdata, linewidth=2,
    #                 label=r"$(p-1)\alpha_\text{acc}$")
    #     ax.semilogx(ebins_mid, esc_rate2_r[tframe], linewidth=2,
    #                 label=r"$\alpha_\text{esc}$")
    #     ax.legend(loc=1, prop={'size': 16}, ncol=1,
    #               shadow=False, fancybox=False, frameon=False)
    #     ax.set_xlim(ebins[0], ebins[-1])
    #     ax.tick_params(bottom=True, top=True, left=True, right=False)
    #     ax.tick_params(axis='x', which='minor', direction='in')
    #     ax.tick_params(axis='x', which='major', direction='in')
    #     ax.tick_params(axis='y', which='minor', direction='in')
    #     ax.tick_params(axis='y', which='major', direction='in')
    #     ax.set_ylabel('Rates', fontsize=20)
    #     ax.tick_params(axis='x', labelbottom=False)
    #     ax.tick_params(labelsize=16)

    #     rect[1] -= rect[3] + vgap
    #     ax = fig.add_axes(rect)
    #     colors1 = palettable.tableau.Tableau_10.mpl_colors
    #     ax.set_prop_cycle('color', colors1)
    #     pindex = 1 + div0(esc_rate2_r[tframe], acc_rate_r[tframe])
    #     label = r"$1+(\alpha_\text{esc}/\alpha)$"
    #     ax.semilogx(ebins_mid, pindex, linewidth=2, label=label)
    #     pindex += (np.gradient(np.log(acc_rate_r[tframe])) /
    #                (math.log(ebins[1]) - math.log(ebins[0])))
    #     pindex -= (np.gradient(np.log(esc_rate2_r[tframe])) /
    #                (math.log(ebins[1]) - math.log(ebins[0])))
    #     label = (r"$1+(\alpha_\text{esc}/\alpha)+" +
    #              r"\partial\ln\alpha_\text{acc}/\partial\ln E-" +
    #              r"\partial\ln\alpha_\text{esc}/\partial\ln E$")
    #     ax.semilogx(ebins_mid, pindex, linewidth=2, label=label)
    #     ax.semilogx([ebins[0], ebins[-1]], [-power_index, -power_index],
    #                 linewidth=1, linestyle='--', color='k')
    #     ax.set_xlim(ebins[0], ebins[-1])
    #     ax.legend(loc=2, prop={'size': 16}, ncol=1,
    #               shadow=False, fancybox=False, frameon=False)
    #     ax.tick_params(bottom=True, top=True, left=True, right=False)
    #     ax.tick_params(axis='x', which='minor', direction='in')
    #     ax.tick_params(axis='x', which='major', direction='in')
    #     ax.tick_params(axis='y', which='minor', direction='in')
    #     ax.tick_params(axis='y', which='major', direction='in')
    #     ax.tick_params(labelsize=16)
    #     ax.set_xlabel(r'$\gamma/T_e$', fontsize=20)
    #     ax.set_ylabel('Power-law index', fontsize=20)
    #     img_dir = '../img/power_law_index/power_index_tracer/' + pic_run + '/'
    #     mkdir_p(img_dir)
    #     fname = img_dir + "pindex_tracer_" + str(tframe) + ".pdf"
    #     fig.savefig(fname)
    #     if show_plot:
    #         plt.show()
    #     else:
    #         plt.close()


def rates_sigma_bg(plot_config, show_plot=True):
    """
    Plot acceleration and escape rates scaling with sigma and bg

    """
    species = plot_config["species"]
    if species in ["e", "electron"]:
        sname = "electron"
    else:
        sname = "Ion"
    reconnection_run = True
    turbulence_run = False

    pic_runs = []
    bg_runs = []
    binplane = []
    if reconnection_run:
        sigmas = ["01", "04", "16", "64"]
        bgs = ["005", "01", "02", "04", "10"]
        bgs_float = [0.05, 0.1, 0.2, 0.4, 1.0]
        nsigma = len(sigmas)
        nbg = len(bgs)
        for sigma in sigmas:
            for ibg, bg in enumerate(bgs):
                pic_run = "sigma" + sigma + "_bg" + bg + "_4000de_triggered"
                picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
                pic_info = read_data_from_json(picinfo_fname)
                pic_runs.append(pic_run)
                bg_runs.append(bgs_float[ibg])
                binplane.append(pic_info.b0)
        nruns_1836 = len(pic_runs)
    else:
        nsigma = 0
        nbg = 0

    # if turbulence_run:
    #     sigmaes = ["100", "40", "10"]
    #     db2s = ["05", "1", "4"]
    #     bgs_float = [1/math.sqrt(0.5), 1.0, 0.5]
    #     nsigmae = len(sigmaes)
    #     ndb = len(db2s)
    #     for sigmae in sigmaes:
    #         for ibg, db2 in enumerate(db2s):
    #             pic_run = "mime1_sigmae" + sigmae + "_vthe04_db" + db2 + "_1024de"
    #             picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    #             pic_info = read_data_from_json(picinfo_fname)
    #             print(pic_info.ene_bx[0], pic_info.ene_by[0], pic_info.ene_bz[0])
    #             pic_runs.append(pic_run)
    #             bg_runs.append(bgs_float[ibg])
    #             # binplane.append(pic_info.b0 / bgs_float[ibg] / (2 * math.sqrt(2.0)))
    #             binplane.append(pic_info.b0 / bgs_float[ibg])
    # else:
    #     nsigmae = 0
    #     ndb = 0

    # if reconnection_run:
    #     pic_run = "sigmae400_bg00_800de_triggered"
    #     pic_runs.append(pic_run)
    #     bg_runs.append(0.0)
    #     picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    #     pic_info = read_data_from_json(picinfo_fname)
    #     binplane.append(pic_info.b0)
    nruns = len(pic_runs)

    sigma = np.zeros([nsigma, nbg])
    sigma_c = np.zeros([nsigma, nbg])
    sigmae_c = np.zeros([nsigma, nbg])
    sigmai_c = np.zeros([nsigma, nbg])
    sigmae_h = np.zeros([nsigma, nbg])
    sigmai_h = np.zeros([nsigma, nbg])
    beta_e = np.zeros([nsigma, nbg])
    pindex = np.zeros([nsigma, nbg])
    arate = np.zeros([nsigma, nbg])
    erate = np.zeros([nsigma, nbg])
    var = np.zeros([nsigma, nbg])

    gamma_va = 1 / np.sqrt(1 - var**2)
    fig1 = plt.figure(figsize=[12, 12])
    rect0 = [0.10, 0.7, 0.39, 0.25]
    hgap, vgap = 0.06, 0.08
    axs_acc = []
    for ibg in range(nbg):
        row = ibg // 2
        col = ibg % 2
        rect = np.copy(rect0)
        rect[0] += col * (rect[2] + vgap)
        rect[1] -= row * (rect[3] + hgap)
        ax = fig1.add_axes(rect)
        ax.set_prop_cycle('color', COLORS)
        axs_acc.append(ax)
        ax.grid()
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.set_xlabel(r'$(\gamma-1)/T_e$', fontsize=16)
        if col == 0:
            ax.set_ylabel('Acceleration Rate', fontsize=16)
        ax.tick_params(labelsize=16)

    fig2 = plt.figure(figsize=[12, 12])
    rect0 = [0.10, 0.7, 0.39, 0.25]
    hgap, vgap = 0.06, 0.08
    axs_esc = []
    for ibg in range(nbg):
        row = ibg // 2
        col = ibg % 2
        rect = np.copy(rect0)
        rect[0] += col * (rect[2] + vgap)
        rect[1] -= row * (rect[3] + hgap)
        ax = fig2.add_axes(rect)
        ax.set_prop_cycle('color', COLORS)
        axs_esc.append(ax)
        ax.grid()
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.set_xlabel(r'$(\gamma-1)/T_e$', fontsize=16)
        if col == 0:
            ax.set_ylabel('Escape Rate', fontsize=16)
        ax.tick_params(labelsize=16)

    fig3 = plt.figure(figsize=[12, 8])
    rect0 = [0.10, 0.58, 0.37, 0.39]
    hgap, vgap = 0.10, 0.08
    axs_acc2 = []
    for isigma in range(nsigma):
        row = isigma // 2
        col = isigma % 2
        rect = np.copy(rect0)
        rect[0] += col * (rect[2] + vgap)
        rect[1] -= row * (rect[3] + hgap)
        ax = fig3.add_axes(rect)
        ax.set_prop_cycle('color', COLORS)
        axs_acc2.append(ax)
        ax.grid()
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.set_xlabel(r'$(\gamma-1)/T_e$', fontsize=16)
        if col == 0:
            ax.set_ylabel('Acceleration Rate', fontsize=16)
        ax.tick_params(labelsize=16)

    fig4 = plt.figure(figsize=[12, 8])
    rect0 = [0.10, 0.58, 0.37, 0.39]
    hgap, vgap = 0.10, 0.08
    axs_esc2 = []
    for isigma in range(nsigma):
        row = isigma // 2
        col = isigma % 2
        rect = np.copy(rect0)
        rect[0] += col * (rect[2] + vgap)
        rect[1] -= row * (rect[3] + hgap)
        ax = fig4.add_axes(rect)
        ax.set_prop_cycle('color', COLORS)
        axs_esc2.append(ax)
        ax.grid()
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.set_xlabel(r'$(\gamma-1)/T_e$', fontsize=16)
        if col == 0:
            ax.set_ylabel('Escape Rate', fontsize=16)
        ax.tick_params(labelsize=16)

    for irun, pic_run in enumerate(pic_runs):
        print("-"*20)
        print("PIC run name: %s" % pic_run)
        isigma = irun // nbg
        ibg = irun % nbg
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        pic_run_dir = pic_info.run_dir
        mime = pic_info.mime
        n0 = pic_info.n0
        gamma_e = 4.0/3 if pic_info.Te > 0.1 else 1.5  # Adiabatic index
        gamma_i = 5.0/3
        temp = pic_info.Te if species in ['e', 'electron'] else pic_info.Ti
        # In typical VPIC simulation, me=1, c=1
        enthalpy_e =  n0 * (1 + pic_info.Te * gamma_e / (gamma_e - 1))
        enthalpy_i =  n0 * (mime + pic_info.Ti * gamma_i / (gamma_i - 1))
        enthalpy = enthalpy_e + enthalpy_i
        b0 = binplane[irun]
        br = b0
        bg = bg_runs[irun]
        sigma[isigma, ibg] = b0**2 / enthalpy
        sigma_r = br**2 / enthalpy
        sigma_c[isigma, ibg] = b0**2 / (n0 * (1 + mime))
        sigmae_c[isigma, ibg] = b0**2 / n0
        sigmai_c[isigma, ibg] = b0**2 / (n0 * mime)
        sigmae_h[isigma, ibg] = b0**2 / enthalpy_e
        sigmai_h[isigma, ibg] = b0**2 / enthalpy_i
        beta_e[isigma, ibg] = 2 * n0 * pic_info.Te / b0**2
        spect_params = get_spect_params(pic_run)
        pindex[isigma, ibg] = -spect_params["power_index"]
        va = math.sqrt(sigma_r / (sigma_r+sigma[isigma, ibg]*bg**2 + 1))
        var[isigma, ibg] = va
        sigma_tot = sigma_r + sigma[isigma, ibg]*bg**2
        va0 = math.sqrt(sigma_tot / (sigma_tot + 1))

        fdir = '../data/power_law_index/rates_tracer/' + pic_run + '/'
        fname = fdir + 'ebins.dat'
        ebins = np.fromfile(fname)
        ebins /= temp
        ebins_mid = 0.5 * (ebins[1:] + ebins[:-1])
        nbins, = ebins.shape
        fname = fdir + 'nptl_esc.dat'
        nptl_esc = np.fromfile(fname).reshape([-1, nbins-1])
        fname = fdir + 'dnptl_esc.dat'
        dnptl_esc = np.fromfile(fname).reshape([-1, nbins-1])
        fname = fdir + 'nptl_acc.dat'
        nptl_acc = np.fromfile(fname).reshape([-1, nbins-1])
        fname = fdir + 'acc_rate_sum.dat'
        acc_rate_sum = np.fromfile(fname).reshape([-1, nbins-1])
        fname = fdir + 'acc_rate_esc.dat'
        acc_rate_esc = np.fromfile(fname).reshape([-1, nbins-1])
        spect_params = get_spect_params(pic_run)
        power_index = spect_params["power_index"]

        nptl_esc = gaussian_filter(nptl_esc+1, sigma=[5, 1])
        nptl_acc = gaussian_filter(nptl_acc+1, sigma=[5, 1])
        acc_rate_sum = gaussian_filter(acc_rate_sum, sigma=[5, 1])
        dnptl_esc = gaussian_filter(dnptl_esc+1E-5, sigma=[5, 1])

        rfactor = get_time_interval(pic_run)
        nframes, _ = acc_rate_sum.shape
        n1 = (nframes // rfactor) * rfactor
        nfr = n1 // rfactor

        eend = nframes
        acc_rate_sum_r = np.mean(acc_rate_sum[:eend], axis=0)
        acc_rate_esc_r = np.mean(acc_rate_esc[:eend], axis=0)
        nptl_acc_r = np.mean(nptl_acc[:eend], axis=0)
        nptl_esc_r = np.mean(nptl_esc[:eend], axis=0)
        dnptl_esc_r = np.mean(dnptl_esc[:eend], axis=0)
        acc_rate_r = div0(acc_rate_sum_r, nptl_acc_r)
        acc_esc_r = div0(acc_rate_sum_r, nptl_esc_r)
        dtwpe_tracer = pic_info.dtwpe * pic_info.tracer_interval
        if pic_run == "turbulent-sheet3D-mixing-sigma100":
            esc_rate = div0(dnptl_esc, nptl_acc)
            fname = fdir + 'twpe.dat'
            twpe = np.fromfile(fname)
            for ibin in range(nbins-1):
                esc_rate[1:, ibin] /= np.diff(twpe)
            esc_rate2_r = np.mean(esc_rate[:eend], axis=0)
        else:
            esc_rate2_r = div0(dnptl_esc_r, nptl_acc_r) / dtwpe_tracer

        ax = axs_acc[ibg]
        text = r"$\sigma_r=" + ("%0.1f" % sigma[isigma, ibg]) + "$"
        ax.semilogx(ebins_mid, acc_rate_r, linewidth=2, label=text)

        ax = axs_esc[ibg]
        text = r"$\sigma_r=" + ("%0.1f" % sigma[isigma, ibg]) + "$"
        ax.semilogx(ebins_mid, esc_rate2_r, linewidth=2, label=text)

        ax = axs_acc2[isigma]
        text = r"$B_g=" + str(bgs_float[ibg]) + "$"
        ax.semilogx(ebins_mid, acc_rate_r, linewidth=2, label=text)

        ax = axs_esc2[isigma]
        text = r"$B_g=" + str(bgs_float[ibg]) + "$"
        ax.semilogx(ebins_mid, esc_rate2_r, linewidth=2, label=text)

        arate[isigma, ibg] = np.max(acc_rate_r[:nbins//2])
        erate[isigma, ibg] = np.mean(esc_rate2_r[:nbins//2])

    for ibg, ax in enumerate(axs_acc):
        ax.set_ylim([0, 0.0012])
        ax.legend(loc=3, prop={'size': 16}, ncol=1,
                  shadow=False, fancybox=False, frameon=True)
        text = r"$B_g=" + str(bgs_float[ibg]) + "$"
        ax.text(0.97, 0.97, text, color='k', fontsize=16,
                bbox=dict(facecolor='none', alpha=1.0,
                          edgecolor='none', pad=10.0),
                horizontalalignment='right', verticalalignment='top',
                transform=ax.transAxes)
    for ibg, ax in enumerate(axs_esc):
        ax.set_ylim([0, 0.003])
        ax.legend(loc=2, prop={'size': 16}, ncol=1,
                  shadow=False, fancybox=False, frameon=True)
        text = r"$B_g=" + str(bgs_float[ibg]) + "$"
        ax.text(0.97, 0.97, text, color='k', fontsize=16,
                bbox=dict(facecolor='none', alpha=1.0,
                          edgecolor='none', pad=10.0),
                horizontalalignment='right', verticalalignment='top',
                transform=ax.transAxes)
    for isigma, ax in enumerate(axs_acc2):
        ax.set_ylim([0, 0.0012])
        ax.legend(loc=3, prop={'size': 16}, ncol=1,
                  shadow=False, fancybox=False, frameon=True)
        text = r"$\sigma_r=" + ("%0.1f" % sigma[isigma, 0]) + "$"
        ax.text(0.97, 0.97, text, color='k', fontsize=16,
                bbox=dict(facecolor='none', alpha=1.0,
                          edgecolor='none', pad=10.0),
                horizontalalignment='right', verticalalignment='top',
                transform=ax.transAxes)
    for isigma, ax in enumerate(axs_esc2):
        ax.set_ylim([0, 0.003])
        ax.legend(loc=2, prop={'size': 16}, ncol=1,
                  shadow=False, fancybox=False, frameon=True)
        text = r"$\sigma_r=" + ("%0.1f" % sigma[isigma, 0]) + "$"
        ax.text(0.97, 0.97, text, color='k', fontsize=16,
                bbox=dict(facecolor='none', alpha=1.0,
                          edgecolor='none', pad=10.0),
                horizontalalignment='right', verticalalignment='top',
                transform=ax.transAxes)
    fdir = "../img/power_law_index/rates_sigma_bg/"
    mkdir_p(fdir)
    fig1.savefig(fdir + "arates_sigma.pdf")
    fig2.savefig(fdir + "erates_sigma.pdf")
    fig3.savefig(fdir + "arates_bg.pdf")
    fig4.savefig(fdir + "erates_bg.pdf")
    plt.show()

#     gamma_va = 1 / np.sqrt(1 - var**2)
#     fig = plt.figure(figsize=[12, 12])
#     rect0 = [0.10, 0.7, 0.39, 0.25]
#     hgap, vgap = 0.06, 0.08
#     for ibg in range(nbg):
#         row = ibg // 2
#         col = ibg % 2
#         rect = np.copy(rect0)
#         rect[0] += col * (rect[2] + vgap)
#         rect[1] -= row * (rect[3] + hgap)
#         ax = fig.add_axes(rect)
#         ax.set_prop_cycle('color', COLORS)
#         ax.plot(sigma[:, ibg], arate[:, ibg],
#                 linewidth=2, marker='o', label="Acc")
#         ax.plot(sigma[:, ibg], erate[:, ibg],
#                 linewidth=2, marker='o', label="Esc")
#         ax.grid()
#         ax.tick_params(bottom=True, top=True, left=True, right=False)
#         ax.tick_params(axis='x', which='minor', direction='in')
#         ax.tick_params(axis='x', which='major', direction='in')
#         ax.tick_params(axis='y', which='minor', direction='in')
#         ax.tick_params(axis='y', which='major', direction='in')
#         ax.set_xlabel(r'$\sigma_r$', fontsize=20)
#         if col == 0:
#             ax.set_ylabel('Rates', fontsize=20)
#         ax.tick_params(labelsize=16)
#         ax.legend(loc=1, prop={'size': 16}, ncol=1,
#                   shadow=False, fancybox=False, frameon=True)
#     plt.show()


def plot_dee_tracer(plot_config, show_plot=True):
    """plot energy diffusion coefficients calculated from tracers
    """
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_run_dir = pic_info.run_dir
    dtwpe_tracer = pic_info.dtwpe * pic_info.tracer_interval

    dee_interval = get_time_interval(pic_run)
    fdir = '../data/power_law_index/rates_tracer/' + pic_run + '/'
    file_list = os.listdir(fdir)
    nfiles = 0
    for fname in file_list:
        if "gamma_avg" in fname:
            nfiles += 1

    for tframe in range(nfiles):
        fname = fdir + 'gamma_avg_' + str(tframe) + '.dat'
        gamma_avg = np.fromfile(fname).reshape([-1, dee_interval])
        fname = fdir + 'dgamma_' + str(tframe) + '.dat'
        dgamma = np.fromfile(fname).reshape([-1, dee_interval])
        nbins, _ = dgamma.shape
        fdata = dgamma**2
        for ibin in range(nbins):
            color = plt.cm.jet(ibin/float(nbins), 1)
            plt.plot(fdata[ibin], color=color)
        plt.show()


def get_trajectory(plot_config, show_plot=True):
    """Get particle trajectories from sorted tracer files

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_run_dir = pic_info.run_dir
    dtwpe_tracer = pic_info.dtwpe * pic_info.tracer_interval

    tracer_dir = pic_run_dir + 'tracer/tracer1/'
    nfiles = len(os.listdir(tracer_dir))
    file_list = os.listdir(tracer_dir)
    tframes = []
    for file_name in file_list:
        fsplit = file_name.split(".")
        tindex = int(fsplit[-1])
        tframes.append(tindex)
    tframes = np.sort(np.asarray(tframes))

    if species in ["e", "electron"]:
        sname = "electron"
        pmass = 1.0
        pcharge = -1.0
    else:
        sname = "H"
        pmass = pic_info.mime
        pcharge = 1.0

    # Check how many steps in each file
    fname = tracer_dir + 'T.0/electron_tracer_qtag_sorted.h5p'
    with h5py.File(fname, 'r') as fh:
        groups = list(fh.keys())
        nsteps_file = len(groups)
        tsteps = [int(group[5:]) for group in groups]
        tsteps.sort()
    if len(tsteps) > 1:
        tinterval = tsteps[1] - tsteps[0]
    else:
        nsteps_file = 1
        tinterval = tframes[1] - tframes[0]
        tinterval_file = tinterval

    # Select tracers from the last time step
    tmax_dir = tframes[-1]
    fname = tracer_dir + 'T.' + str(tmax_dir) + '/electron_tracer_qtag_sorted.h5p'
    with h5py.File(fname, 'r') as fh:
        groups = list(fh.keys())
        tsteps = [int(group[5:]) for group in groups]
        tsteps.sort()
        tmax = tsteps[-1]
        group = fh['Step#' + str(tmax)]
        dset = group['dX']
        nptl, = dset.shape
        ptl = {}
        for dset in group:
            dset = str(dset)
            ptl[str(dset)] = read_var(group, dset, nptl)
    nkeys = len(ptl.keys())
    gamma_final = np.sqrt(1 + ptl["Ux"]**2 + ptl["Uy"]**2 + ptl["Uz"]**2)
    gamma_max = np.max(gamma_final)
    nptl = 200
    band_interval = 5
    nbands = 3
    nptl_t = np.sum(gamma_final > gamma_max/band_interval)
    if nptl > nptl_t:
        nptl = nptl_t
    ptl_selected = np.zeros([nbands, nptl], dtype=int)
    for iband in range(nbands):
        cond = np.logical_and(gamma_final > gamma_max / band_interval**(iband+1),
                              gamma_final < gamma_max / band_interval**iband)
        index, = np.where(cond)
        ptl_selected[iband, :] = np.random.choice(index, nptl, replace=False)
    ptl_selected = np.sort(ptl_selected, axis=1)
    tags = ptl["q"][ptl_selected]

    nframes = tmax // tinterval + 1

    ptls = np.zeros([nbands, nptl, nframes, nkeys], dtype=np.float32)

    tracer_interval = pic_info.tracer_interval
    for ifile in range(nfiles):
        print("File %d of %d" % (ifile, nfiles))
        tindex_file = ifile * nsteps_file * tracer_interval
        fname = tracer_dir + 'T.' + str(tindex_file) + '/' + sname + '_tracer_qtag_sorted.h5p'
        with h5py.File(fname, 'r') as fh:
            for istep in range(nsteps_file):
                tindex = tindex_file + istep * tracer_interval
                tframe = ifile * nsteps_file + istep
                if tindex > tmax:
                    break
                gname = 'Step#' + str(tindex)
                group = fh[gname]
                for dset_name in group:
                    dset = group[dset_name]
                    dset.read_direct(ptl[dset_name])
                for ikey, key in enumerate(ptl.keys()):
                    for iband in range(nbands):
                        ptls[iband, :, tframe, ikey] = ptl[key][ptl_selected[iband, :]]
    for iband in range(nbands):
        fdir = '../data/power_law_index/trajectory/' + pic_run + '/'
        mkdir_p(fdir)
        fname = fdir + sname + '_traj_band' + str(iband) + '.h5'
        with h5py.File(fname, 'w') as fh:
            for iptl in range(nptl):
                print("Particle " + str(iptl))
                tag = 'Particle#' + str(tags[iband, iptl])
                grp = fh.create_group(tag)
                for ikey, key in enumerate(ptl.keys()):
                    grp.create_dataset(key, (nframes, ),
                                       data=ptls[iband, iptl, :, ikey])


def plot_trajectory_band(plot_config, show_plot=True):
    """Plot particle trajectory in different energy band
    """
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    dtwpe_tracer = pic_info.dtwpe * pic_info.tracer_interval
    wce_wpe = pic_info.dtwce / pic_info.dtwpe
    smime = math.sqrt(pic_info.mime)
    xmin, xmax = 0, pic_info.lx_di * smime
    zmin, zmax = -0.5*pic_info.lz_di*smime, 0.5*pic_info.lz_di*smime

    if species in ["e", "electron"]:
        sname = "electron"
    else:
        sname = "Ion"
    fdir = '../data/power_law_index/trajectory/' + pic_run + '/'
    fname = fdir + sname + '_traj_band1.h5'
    ptl = {}
    with h5py.File(fname, 'r') as fh:
        particle_tags = list(fh.keys())
        nptl = len(particle_tags)
        print("Total number of particles: %d" % nptl)
        group = fh[particle_tags[0]]
        dset = group['dX']
        nframes, = dset.shape
        nkeys = len(group.keys())
        for dset_name in group:
            dset = group[dset_name]
            ptl[str(dset_name)] = np.zeros(dset.shape, dset.dtype)
    ttracer = np.arange(0, nframes) * dtwpe_tracer
    tmin, tmax = ttracer[0], ttracer[-1]

    nbands = 3
    for iband in range(nbands):
        fname = fdir + sname + '_traj_band' + str(iband) + '.h5'
        img_dir = '../img/power_law_index/tracer_traj/' + pic_run + '/'
        img_dir += 'band' + str(iband) + '/'
        mkdir_p(img_dir)
        with h5py.File(fname, 'r') as fh:
            for iptl, gname in enumerate(fh):
                print("Particle name: ", gname)
                group = fh[gname]
                for dset_name in group:
                    dset = group[dset_name]
                    ptl[str(dset_name)] = np.zeros(dset.shape, dset.dtype)
                    dset.read_direct(ptl[str(dset_name)])

                gamma = np.sqrt(1 + ptl["Ux"]**2 + ptl["Uy"]**2 + ptl["Uz"]**2)
                tesc = np.argmax(gamma > 0.9*gamma[-1])
                dgamma = gamma - gamma[0]
                gamma_norm = gamma / gamma[0]
                igamma = 1.0 / gamma
                vx = ptl["Ux"] * igamma
                vy = ptl["Uy"] * igamma
                vz = ptl["Uz"] * igamma
                x = ptl["dX"]
                y = ptl["dY"]
                z = ptl["dZ"]
                ex = ptl["Ex"]
                ey = ptl["Ey"]
                ez = ptl["Ez"]
                bx = ptl["Bx"]
                by = ptl["By"]
                bz = ptl["Bz"]
                edotb = ex*bx + ey*by + ez*bz
                ib2 = 1.0 / (bx**2 + by**2 + bz**2)
                eparax = edotb * bx * ib2
                eparay = edotb * by * ib2
                eparaz = edotb * bz * ib2
                wtot = np.cumsum(-(ex*vx + ey*vy + ez*vz)) * dtwpe_tracer
                wpara = np.cumsum(-(eparax * vx + eparay * vy + eparaz * vz)) * dtwpe_tracer
                wperp = wtot - wpara
                wpara_norm = wpara / gamma[0]
                wperp_norm = wperp / gamma[0]
                fig = plt.figure(figsize=[5, 7])
                rect = [0.15, 0.56, 0.72, 0.4]
                hgap, vgap = 0.03, 0.08
                ax = fig.add_axes(rect)
                COLORS = palettable.tableau.Tableau_10.mpl_colors
                ax.set_prop_cycle('color', COLORS)
                ax.plot(ttracer, wpara, linewidth=2, label=r'$W_\parallel$')
                ax.plot(ttracer, wperp, linewidth=2, label=r'$W_\perp$')
                ax.plot(ttracer, gamma, linewidth=2, label=r'$\gamma$')
                ylim = ax.get_ylim()
                ax.plot([ttracer[tesc], ttracer[tesc]], ylim, color='k',
                        linewidth=1, linestyle='--')
                ax.set_xlim([tmin, tmax])
                ax.set_ylim(ylim)
                ax.tick_params(bottom=True, top=True, left=True, right=True)
                ax.tick_params(axis='x', which='minor', direction='in')
                ax.tick_params(axis='x', which='major', direction='in')
                ax.tick_params(axis='y', which='minor', direction='in')
                ax.tick_params(axis='y', which='major', direction='in')
                ax.set_xlabel(r'$t\omega_{pe}$', fontsize=16)
                ax.set_ylabel('Energy change', fontsize=16)
                ax.tick_params(labelsize=12)
                ax.legend(loc=6, prop={'size': 12}, ncol=1,
                         shadow=False, fancybox=False, frameon=False)

                rect[1] -= rect[3] + vgap
                gamma_norm = gamma / gamma[0]
                ax = fig.add_axes(rect)
                ax.set_prop_cycle('color', COLORS)
                points = np.array([x, z]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                # Create a continuous norm to map from data points to colors
                norm = plt.Normalize(gamma.min(), gamma.max())
                lc = LineCollection(segments, cmap='jet', norm=norm)
                # Set the values used for colormapping
                lc.set_array(gamma)
                lc.set_linewidth(2)
                ax.add_collection(lc)
                ax.plot([x[tesc]], [z[tesc]], marker='o', markersize=7, color="k")
                ax.tick_params(bottom=True, top=True, left=True, right=True)
                ax.tick_params(axis='x', which='minor', direction='in')
                ax.tick_params(axis='x', which='major', direction='in')
                ax.tick_params(axis='y', which='minor', direction='in')
                ax.tick_params(axis='y', which='major', direction='in')
                ax.set_xlabel(r'$x$', fontsize=16)
                ax.set_ylabel(r'$z$', fontsize=16)
                ax.tick_params(labelsize=12)
                ax.set_xlim([xmin, xmax])
                ax.set_ylim([zmin, zmax])
                # Colorbar
                rect_cbar = np.copy(rect)
                rect_cbar[0] = rect[0] + rect[2] + 0.01
                rect_cbar[2] = 0.02

                cax = fig.add_axes(rect_cbar)
                print("min and max of energy: %f, %f" %
                      (gamma.min(), gamma.max()))
                sm = plt.cm.ScalarMappable(cmap=plt.cm.jet,
                                           norm=plt.Normalize(vmin=gamma.min(),
                                                              vmax=gamma.max()))
                cax.tick_params(axis='y', which='major', direction='in')
                sm._A = []
                cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
                cax.tick_params(labelrotation=90)
                cbar.set_label(r'$(\gamma-1)/\varepsilon_\text{th}$',
                               fontsize=16, labelpad=1)
                cbar.ax.tick_params(labelsize=12)
                cax.yaxis.set_ticks_position('right')
                cax.yaxis.set_label_position('right')

                fname = img_dir + sname + "_tracer_" + str(iptl) + ".pdf"
                fig.savefig(fname)

                plt.close()
                # plt.show()


def calc_dee_tracer(plot_config, show_plot=True):
    """Get energy diffusing using sorted tracer files

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    dtwpe_tracer = pic_info.dtwpe * pic_info.tracer_interval

    tracer_dir = pic_run_dir + 'tracer/tracer1/'
    nframes = len(os.listdir(tracer_dir))
    file_list = os.listdir(tracer_dir)
    tframes = []
    for file_name in file_list:
        fsplit = file_name.split(".")
        tindex = int(fsplit[-1])
        tframes.append(tindex)
    tframes = np.sort(np.asarray(tframes))
    tinterval = tframes[1] - tframes[0]

    if species in ["e", "electron"]:
        sname = "electron"
        pmass = 1.0
        pcharge = -1.0
    else:
        sname = "H"
        pmass = pic_info.mime
        pcharge = 1.0

    # Select tracers from the last time step
    tstart = 1000
    tshift = 200
    tindex = tinterval * tstart
    fname = tracer_dir + 'T.' + str(tindex) + '/electron_tracer_qtag_sorted.h5p'
    with h5py.File(fname, 'r') as fh:
        group = fh['Step#' + str(tindex)]
        dset = group['dX']
        nptl, = dset.shape
        ptl = {}
        for dset in group:
            dset = str(dset)
            ptl[str(dset)] = read_var(group, dset, nptl)
    nkeys = len(ptl.keys())
    gamma = np.sqrt(1 + ptl["Ux"]**2 + ptl["Uy"]**2 + ptl["Uz"]**2)
    temp = pic_info.Te if species in ['e', 'electron'] else pic_info.Ti
    # emin, emax = 1, 1E3
    spect_params = get_spect_params(pic_run)
    emin, emax = spect_params["energy_range"]
    nbins = 20
    emin *= temp
    emax *= temp
    dloge = (math.log10(emax) - math.log10(emin)) / (nbins - 1)
    emin0 = 10**(math.log10(emin) - dloge)
    ebins = np.logspace(math.log10(emin0), math.log10(emax), nbins+1)
    ebins_mid = (ebins[:-1] + ebins[1:]) * 0.5
    debins = np.diff(ebins)
    fbins, _ = np.histogram(gamma, bins=ebins)
    # fbins = fbins / debins
    # plt.loglog(ebins_mid, fbins)
    # plt.show()
    ptl_indices = {}
    for ibin in range(nbins):
        cond = np.logical_and(gamma > ebins[ibin], gamma < ebins[ibin+1])
        if np.sum(cond):
            ptl_indices[ibin] = np.where(cond)
    ibin_max = np.max(list(ptl_indices.keys()))

    spectra = np.zeros([nbins, tshift, nbins])
    gamma_avg = np.zeros([nbins, tshift])
    dgamma = np.zeros([nbins, tshift])
    ttracer = np.linspace(1, tshift, tshift) * dtwpe_tracer
    for tframe in range(tstart, tstart+tshift):
        print("Time frame %d of %d" % (tframe, nframes))
        tindex = tframe * pic_info.tracer_interval
        fname = tracer_dir + 'T.' + str(tindex) + '/' + sname + '_tracer_qtag_sorted.h5p'
        with h5py.File(fname, 'r') as fh:
            gname = 'Step#' + str(tindex)
            group = fh[gname]
            for dset_name in ["Ux", "Uy", "Uz"]:
                dset = group[dset_name]
                dset.read_direct(ptl[dset_name])
        gamma = np.sqrt(1 + ptl["Ux"]**2 + ptl["Uy"]**2 + ptl["Uz"]**2)
        for ibin in range(ibin_max+1):
            gamma_selected = gamma[ptl_indices[ibin]]
            spectra[ibin, tframe-tstart, :], _ = np.histogram(gamma_selected, bins=ebins)
            gamma_avg[ibin, tframe-tstart] = np.mean(gamma_selected)
            if len(gamma_selected) == 1:
                dgamma[ibin, tframe-tstart] = 0.0
            else:
                dgamma[ibin, tframe-tstart] = math.sqrt(np.mean(gamma_selected**2) -
                                                        gamma_avg[ibin, tframe-tstart]**2)

    # img_dir = '../img/power_law_index/dee_tracer/' + pic_run + '/tstart_' + str(tstart) + '/'
    # mkdir_p(img_dir)
    # for ibin in range(nbins):
    #     fig = plt.figure(figsize=[7, 5])
    #     rect = [0.12, 0.16, 0.72, 0.8]
    #     ax = fig.add_axes(rect)
    #     # for iframe in range(tshift):
    #     #     color = plt.cm.jet(iframe/float(tshift), 1)
    #     #     ax.loglog(ebins_mid, spectra[ibin, iframe, :], color=color)
    #     # fname = img_dir + 'ibin_' + str(ibin) + '.pdf'
    #     # fig.savefig(fname)
    #     # plt.close()
    #     ax.plot(ttracer, gamma_avg[ibin] - gamma_avg[ibin, 0])
    #     fname = img_dir + 'gamma_avg_' + str(ibin) + '.pdf'
    #     fig.savefig(fname)
    #     plt.close()

    #     fig = plt.figure(figsize=[7, 5])
    #     rect = [0.12, 0.16, 0.72, 0.8]
    #     ax = fig.add_axes(rect)
    #     ax.plot(ttracer, dgamma[ibin])
    #     fname = img_dir + 'dgamma_' + str(ibin) + '.pdf'
    #     fig.savefig(fname)
    #     plt.close()

    # fig = plt.figure(figsize=[7, 5])
    # rect = [0.12, 0.16, 0.72, 0.8]
    # ax = fig.add_axes(rect)
    # for ibin in range(nbins):
    #     color = plt.cm.jet(ibin/float(nbins), 1)
    #     ax.loglog(ttracer, dgamma[ibin, :], color=color)
    # plt.show()

    de = np.zeros(nbins)
    dee = np.zeros(nbins)
    for ibin in range(nbins):
        de[ibin] = gamma_avg[ibin, -10] - gamma_avg[ibin, 20]
        de[ibin] /= ttracer[-10] - ttracer[20]
        dee[ibin] = np.mean(dgamma[ibin, 20:] / np.sqrt(ttracer[20:]))**2

    fig = plt.figure(figsize=[7, 5])
    rect = [0.12, 0.16, 0.72, 0.8]
    ax = fig.add_axes(rect)
    ax.loglog(ebins_mid, dee*ebins_mid**0.75)
    ax.loglog(ebins_mid, ebins_mid**2/5)
    fig = plt.figure(figsize=[7, 5])
    rect = [0.12, 0.16, 0.72, 0.8]
    ax = fig.add_axes(rect)
    ax.loglog(ebins_mid, de*ebins_mid**0.5)
    ax.loglog(ebins_mid, ebins_mid*10)
    plt.show()


def get_uout_max(plot_config):
    """Get the maximum of the outflow 4-velocity
    """
    tframe = plot_config["tframe"]
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    pic_topox = pic_info.topology_x
    pic_topoy = pic_info.topology_y
    pic_topoz = pic_info.topology_z
    nx, ny, nz = pic_info.nx, pic_info.ny, pic_info.nz
    xmin, xmax = 0, pic_info.lx_di
    zmin, zmax = -pic_info.lz_di * 0.125, pic_info.lz_di * 0.125
    mime = pic_info.mime
    smime = math.sqrt(mime)
    dx_de = pic_info.dx_di * smime
    dy_de = pic_info.dy_di * smime
    dz_de = pic_info.dz_di * smime
    if "bg005" in pic_run:
        kwargs = {"current_time": tframe,
                  "xl": xmin, "xr": xmax,
                  "zb": zmin, "zt": zmax}
        fname = pic_run_dir + "data/uex.gda"
        x, z, uex = read_2d_fields(pic_info, fname, **kwargs)
        fname = pic_run_dir + "data/uix.gda"
        x, z, uix = read_2d_fields(pic_info, fname, **kwargs)
        fname = pic_run_dir + "data/ne.gda"
        x, z, ne = read_2d_fields(pic_info, fname, **kwargs)
        fname = pic_run_dir + "data/ni.gda"
        x, z, ni = read_2d_fields(pic_info, fname, **kwargs)
        ux = (uex * ne + uix * ni * mime) / (ne + ni * mime)
    else:
        fields_interval = pic_info.fields_interval
        tindex = fields_interval * tframe
        xs, xe = 0, pic_info.nx
        zs, ze = 0, pic_info.nz
        vels = {}
        for species in ["e", "i"]:
            sname = "electron" if species == 'e' else "ion"
            fname = (pic_run_dir + "hydro_hdf5/T." + str(tindex) +
                     "/hydro_" + sname + "_" + str(tindex) + ".h5")
            hydro = {}
            with h5py.File(fname, 'r') as fh:
                group = fh["Timestep_" + str(tindex)]
                for var in ["rho", "jx", "jy", "jz"]:
                    dset = group[var]
                    hydro[var]= dset[xs:xe, 0, zs:ze]

            vels["n" + species] = hydro["rho"]
            irho = 1.0 / hydro["rho"]
            vels["v" + species + "x"] = hydro["jx"] * irho
            vels["v" + species + "y"] = hydro["jy"] * irho
            vels["v" + species + "z"] = hydro["jz"] * irho
        ux = (vels["vex"] * vels["ne"] +
              vels["vix"] * vels["ni"] * mime) / (vels["ne"] + vels["ni"] * mime)
        ux = ux.T
    nzr, nxr = ux.shape
    gamma = np.sqrt(ux[nzr//2, :]**2 + 1)
    plt.plot(ux[nzr//2, :] / gamma)
    plt.show()


def plot_spectrum_four_runs(plot_config, show_plot=True):
    """Plot particle energy spectrum for 4 different runs

    Args:
        plot_config: plotting configuration
    """
    species = plot_config["species"]
    pic_runs = ["sigmae400_bg00_800de_triggered",
                "sigma64_bg005_4000de_triggered",
                "mime1_sigmae100_vthe04_db1_1024de",
                "sigma01_bg01_4000de_triggered"]
    labels = ["R1", "R2", "T1", "R3"]
    xpos = [0.5, 0.55, 0.67, 0.63]
    ypos = [0.75, 0.8, 0.73, 0.63]
    fig = plt.figure(figsize=[7, 5])
    rect0 = [0.14, 0.58, 0.32, 0.38]
    hgap, vgap = 0.06, 0.06
    for irun, pic_run in enumerate(pic_runs):
        row = irun // 2
        col = irun % 2
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        if pic_run == "3D-Lx150-bg0.2-150ppc-2048KNL-tracking":
            pic_run_dir = "/net/scratch3/xiaocan/reconnection/Cori_runs/" + pic_run + "/"
        else:
            pic_run_dir = pic_info.run_dir
        vpic_info = get_vpic_info(pic_run_dir)
        emin = vpic_info["emin_spect"]
        emax = vpic_info["emax_spect"]
        nbins = int(vpic_info["nbins"])
        dloge = (math.log10(emax) - math.log10(emin)) / (nbins - 1)
        emin0 = 10**(math.log10(emin) - dloge)
        ebins = np.logspace(math.log10(emin0), math.log10(emax), nbins+1)
        ebins_mid = (ebins[:-1] + ebins[1:]) * 0.5
        debins = np.diff(ebins)
        nptot = pic_info.nx * pic_info.ny * pic_info.nz * pic_info.nppc

        ntf = pic_info.ntf
        spect_dir = pic_run_dir + "spectrum/"
        if os.path.isdir(spect_dir):
            nts = len(os.listdir(spect_dir))
        else:
            nts = ntf
        if pic_run == "3D-Lx150-bg0.2-150ppc-2048KNL-tracking":
            tframes = range(nts)
        else:
            tframes = range(0, nts, 10)
        tframes = list(tframes)
        if tframes[-1] < nts - 1:
            tframes.append(nts - 1)
        nframes = len(tframes)
        flogs = np.zeros((nframes, nbins))
        if species in ["e", "electron"]:
            if pic_run == "3D-Lx150-bg0.2-150ppc-2048KNL-tracking":
                sname = "e"
            else:
                sname = "electron"
        else:
            sname = "ion"
        if pic_run == "3D-Lx150-bg0.2-150ppc-2048KNL-tracking":
            if species == 'e':
                vth = pic_info.vthe
            else:
                vth = pic_info.vthi
            gama = 1.0 / math.sqrt(1.0 - 3 * vth**2)
            temp = gama - 1.0
            spectrum_interval = pic_info.fields_interval
        else:
            temp = pic_info.Te if species in ['e', 'electron'] else pic_info.Ti
            spectrum_interval = vpic_info["spectrum_interval"]
        print("Particle initial temperature (Lorentz factor): %f" % temp)

        rect = np.copy(rect0)
        rect[0] = rect0[0] + (rect0[2] + hgap) * col
        rect[1] = rect0[1] - (rect0[3] + vgap) * row
        ax = fig.add_axes(rect)
        for iframe, tframe in enumerate(tframes):
            # print("Time frame: %d" % tframe)
            tindex = tframe * int(spectrum_interval)
            fdir = pic_run_dir + '/spectrum_combined/'
            fname = fdir + 'spectrum_' + sname + '_' + str(tindex) + '.dat'
            ebins_mid_run = ebins_mid / temp
            debins_run = debins / temp
            if os.path.isfile(fname):
                flog = np.fromfile(fname, dtype=np.float32)
                espect = flog[3:] / debins_run / nptot  # the first 3 are magnetic field
                color = plt.cm.plasma_r(tframe/float(nts), 1)
                flogs[iframe, :] = espect
                ax.loglog(ebins_mid_run, espect, linewidth=0.5, color=color,
                          nonpositive='clip')

        spect_params = get_spect_params(pic_run)
        pindex = spect_params["power_index"]
        emin, emax = spect_params["energy_range"]
        fpower = ebins_mid_run**pindex
        es, _ = find_nearest(ebins_mid_run, emin)
        ee, _ = find_nearest(ebins_mid_run, emax)
        fnorm = (espect[es] / fpower[es]) * spect_params["norm"]
        fpower *= fnorm
        power_index = "{%0.1f}" % -pindex
        pname = r'$\propto (\gamma - 1)^{' + power_index + '}$'
        ax.loglog(ebins_mid_run[es:ee], fpower[es:ee], linewidth=1,
                  linestyle='--', color='k', label=pname)
        # ax.legend(loc=1, prop={'size': 12}, ncol=1,
        #          shadow=False, fancybox=False, frameon=False)
        pname = r'$p=' + power_index + '$'
        ax.text(xpos[irun], ypos[irun], pname, color='k', fontsize=16,
                bbox=dict(facecolor='none', alpha=1.0,
                          edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='bottom',
                transform=ax.transAxes)
        ax.set_xlim([1E0, spect_params["emax"]])
        ax.set_ylim([1E-10, 1E0])
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='out', bottom=False)
        ax.tick_params(axis='x', which='major', direction='out')
        ax.tick_params(axis='y', which='minor', direction='out')
        ax.tick_params(axis='y', which='major', direction='out')
        if row == 1:
            ax.set_xlabel(r'$(\gamma - 1)/T_' + species + '$', fontsize=20)
        else:
            ax.tick_params(axis='x', labelbottom=False)
        if col == 0:
            ax.set_ylabel(r'$f(\gamma - 1)$', fontsize=20)
        else:
            ax.tick_params(axis='y', labelleft=False)
        ax.tick_params(labelsize=16)
        dtwpe_spect = math.ceil(spectrum_interval * pic_info.dtwpe / 0.1) * 0.1
        if pic_run == "3D-Lx150-bg0.2-150ppc-2048KNL-tracking":
            n0 = 1.0
        else:
            n0 = pic_info.n0
        mime = pic_info.mime
        gamma_e = 4.0/3 if temp > 0.1 else 1.5  # Adiabatic index
        gamma_i = 5.0/3
        enthalpy_e =  n0 * (1 + temp * gamma_e / (gamma_e - 1))
        enthalpy_i =  n0 * (mime + temp * gamma_i / (gamma_i - 1))
        enthalpy = enthalpy_e + enthalpy_i
        sigma = pic_info.b0**2 / enthalpy
        va = math.sqrt(sigma / (sigma + 1))
        taua = pic_info.lx_di * math.sqrt(mime) / va
        ts = tframes[0] * dtwpe_spect / taua
        te = tframes[-1] * dtwpe_spect / taua
        # print(te)
        ax.text(0.03, 0.03, labels[irun],
                color='k', fontsize=20,
                bbox=dict(facecolor='none', alpha=1.0,
                          edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='bottom',
                transform=ax.transAxes)

        # tenergy = pic_info.tenergy
        # enorm = pic_info.ene_magnetic[0]
        # ene_magnetic = pic_info.ene_magnetic
        # ene_electric = pic_info.ene_electric
        # kene_e = pic_info.kene_e
        # kene_i = pic_info.kene_i
        # ene_magnetic /= enorm
        # kene_e /= enorm
        # kene_i /= enorm

        # rect1 = np.copy(rect)
        # rect1[0] += 0.08
        # rect1[1] += 0.05
        # rect1[2] = 0.4 * rect[2]
        # rect1[3] = 0.3 * rect[3]
        # ax = fig.add_axes(rect1)
        # ax.plot(tenergy, ene_magnetic, linewidth=1, linestyle='--', color=color)
        # ax.plot(tenergy, kene_e, linewidth=1, linestyle='-', color=color)
        # ax.tick_params(bottom=True, top=True, left=True, right=True)
        # ax.tick_params(axis='x', which='minor', direction='in', bottom=False)
        # ax.tick_params(axis='x', which='major', direction='in')
        # ax.tick_params(axis='y', which='minor', direction='in')
        # ax.tick_params(axis='y', which='major', direction='in')
        # ax.tick_params(labelsize=12)

    rect_cbar = np.copy(rect)
    rect_cbar[0] += rect[2] + 0.02
    rect_cbar[2] = 0.015
    rect_cbar[1] += (rect[3] + vgap) * 0.5
    cax = fig.add_axes(rect_cbar)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma_r,
                               norm=plt.Normalize(vmin=0, vmax=1))
    cax.tick_params(axis='x', which='major', direction='in')
    # fake up the array of the scalar mappable. Urgh...
    sm._A = []
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(r'$t/t_\text{max}$', fontsize=20)
    cbar.ax.tick_params(labelsize=16)

    fdir = '../img/power_law_index/'
    mkdir_p(fdir)
    fname = fdir + 'spectra_runs_' + species + '.pdf'
    fig.savefig(fname)

    plt.show()


def func_power(xvar, pindex, const):
    """Function for fitting with power-law expression.
    """
    return const * np.power(xvar, pindex)


def find_nearest(array, value):
    """Find nearest value in an array
    """
    idx = (np.abs(array-value)).argmin()
    return (idx, array[idx])


def solve_sde(plot_config, show_plot=True):
    """Solve an stochastic differential equation
    """
    # acc_rate = 0.001
    # esc_rate = 0.0007
    # dee_rate = 0.00025
    acc_rate = 0.001
    esc_rate = 0.0078
    dee_rate = 0.002
    nsteps = 4000
    nptl0 = 0
    ninj = int(1E5 / nsteps)
    dt = 1.0
    nptl = nptl0 + nsteps*ninj
    ptl = np.zeros(nptl)
    p0 = 1.0
    ptl[:nptl0] = p0
    for i in range(nsteps):
        print("Step: %d" % i)
        nptl_old = nptl0 + i * ninj
        # esc_cond = np.random.choice(a=[False, True], size=nptl_old,
        #                             p=[1-esc_rate, esc_rate])
        rd = np.random.random(nptl_old)
        esc_cond = rd < (esc_rate * (1.0 - np.log(ptl[:nptl_old])/4))
        ptl[:nptl_old][esc_cond] = -1
        nptl_new = nptl_old + ninj
        ptl[nptl_old:nptl_new] = p0
        dwt = np.random.normal(0, math.sqrt(dt), nptl)
        cond = ptl > 0
        # ptl[cond] += (acc_rate) * ptl[cond] * dt
        # ptl[cond] += (2*dee_rate + acc_rate / ptl[cond]) * ptl[cond] * dt
        ptl[cond] += 2 * dee_rate * (1.0 - np.log(ptl[cond])/4) * ptl[cond] * dt
        ptl[cond] += np.sqrt(2*dee_rate*ptl[cond]**2) * dwt[cond]

    nbins = 100
    bins = np.logspace(-1, 3, nbins+1)
    bins_mid = 0.5 * (bins[:-1] + bins[1:])
    hist, _ = np.histogram(ptl, bins=bins)
    espect = hist/np.diff(bins)
    es, _ = find_nearest(bins_mid, 2.0)
    ee, _ = find_nearest(bins_mid, 10.0)
    popt, _ = curve_fit(func_power, bins_mid[es:ee], espect[es:ee])
    pindex = popt[0]

    fpower = bins_mid[es:ee]**pindex
    fpower *= espect[es] * 3 / fpower[0]
    power_index = "{%0.2f}" % pindex
    pname = r'$\propto \varepsilon^{' + power_index + '}$'

    fig = plt.figure(figsize=[7, 5])
    rect = [0.15, 0.15, 0.8, 0.8]
    ax = fig.add_axes(rect)
    ax.loglog(bins_mid, espect)
    ax.loglog(bins_mid[es:ee], fpower, label=pname, linestyle='--',
              color='k')
    ax.legend(loc=1, prop={'size': 16}, ncol=1,
              shadow=False, fancybox=False, frameon=False)
    ax.tick_params(bottom=True, top=False, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top=True)
    ax.tick_params(axis='x', which='major', direction='in', top=True)
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlabel(r'$\varepsilon$', fontsize=16)
    ax.set_ylabel(r'$f(\varepsilon)$', fontsize=16)
    ax.tick_params(labelsize=16)

    a = dee_rate
    b = acc_rate - 3*dee_rate
    c = 2*dee_rate - acc_rate - esc_rate
    pmodel = (-b + math.sqrt(b**2 - 4*a*c)) / (2 * a)
    print("Simulated power-law index: %f" % -pindex)
    print("Modeled power-law index: %f" % pmodel)

    dee_rate_array = np.logspace(-4, 4, 800) * acc_rate
    a = dee_rate_array
    b = acc_rate - 3*dee_rate_array
    c = 2*dee_rate_array - acc_rate - esc_rate
    pmodel = (-b + np.sqrt(b**2 - 4*a*c)) / (2 * a)
    fig = plt.figure(figsize=[7, 5])
    rect = [0.15, 0.15, 0.8, 0.8]
    ax = fig.add_axes(rect)
    x = dee_rate_array/acc_rate
    ax.semilogx(x, pmodel)
    ax.plot(ax.get_xlim(), [-pindex, -pindex], linestyle='--', color='k')
    ax.set_xlim([x.min(), x.max()])
    ax.grid()
    ax.tick_params(bottom=True, top=False, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top=True)
    ax.tick_params(axis='x', which='major', direction='in', top=True)
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlabel(r'$D_{pp0}/r_\text{acc}$', fontsize=16)
    ax.set_ylabel('Power-law index', fontsize=16)
    ax.tick_params(labelsize=16)

    plt.show()


def solve_sde_pindex(params, run_name, reconnection=True):
    """Solve an stochastic differential equation to get power-law index
    """
    acc_rate = params["acc_rate"]
    esc_rate = params["esc_rate"]
    dee_rate = params["dee_rate"]
    nsteps = params["nsteps"]
    nptl = params["nptl"]
    nptl0 = 0
    ninj = int(nptl / nsteps)
    dt = 1.0
    nptl = nptl0 + nsteps*ninj
    ptl = np.zeros(nptl)
    p0 = 1.0
    ptl[:nptl0] = p0
    for i in range(nsteps):
        if i % 100 == 0:
            print("Step: %d" % i)
        nptl_old = nptl0 + i * ninj
        if reconnection:
            esc_cond = np.random.choice(a=[False, True], size=nptl_old,
                                        p=[1-esc_rate, esc_rate])
        else:
            rd = np.random.random(nptl_old)
            esc_cond = rd < esc_rate * (1 - np.log(ptl[:nptl_old])/4)
        ptl[:nptl_old][esc_cond] = -1
        nptl_new = nptl_old + ninj
        ptl[nptl_old:nptl_new] = p0
        dwt = np.random.normal(0, math.sqrt(dt), nptl)
        cond = ptl > 0
        # ptl[cond] += acc_rate * ptl[cond] * dt
        # ptl[cond] += (2*dee_rate + acc_rate / ptl[cond]) * ptl[cond] * dt
        ptl[cond] += acc_rate * (1 - np.log(ptl[cond])/4) * ptl[cond] * dt
        if not reconnection:
            ptl[cond] += np.sqrt(2*dee_rate*ptl[cond]**2) * dwt[cond]

    nbins = 100
    bins = np.logspace(-1, 3, nbins+1)
    bins_mid = 0.5 * (bins[:-1] + bins[1:])
    hist, _ = np.histogram(ptl, bins=bins)
    espect = hist/np.diff(bins)
    es, _ = find_nearest(bins_mid, 2.0)
    ee, _ = find_nearest(bins_mid, 20.0)
    popt, _ = curve_fit(func_power, bins_mid[es:ee], espect[es:ee])
    pindex = popt[0]

    fpower = bins_mid[es:ee]**pindex
    fpower *= espect[es] * 3 / fpower[0]
    power_index = "{%0.2f}" % pindex
    pname = r'$\propto \varepsilon^{' + power_index + '}$'

    fig = plt.figure(figsize=[7, 5])
    rect = [0.15, 0.15, 0.8, 0.8]
    ax = fig.add_axes(rect)
    ax.loglog(bins_mid, espect)
    ax.loglog(bins_mid[es:ee], fpower, label=pname, linestyle='--',
              color='k')
    ax.legend(loc=1, prop={'size': 16}, ncol=1,
              shadow=False, fancybox=False, frameon=False)
    ax.tick_params(bottom=True, top=False, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top=True)
    ax.tick_params(axis='x', which='major', direction='in', top=True)
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlabel(r'$\varepsilon$', fontsize=16)
    ax.set_ylabel(r'$f(\varepsilon)$', fontsize=16)
    ax.tick_params(labelsize=16)

    print("Simulated power-law index: %f" % -pindex)

    fdir = "../img/power_law_index/pindex_sde/"
    mkdir_p(fdir)
    fname = fdir + "pindex_" + run_name + ".pdf"
    fig.savefig(fname)
    # plt.close()
    plt.show()

    return (-pindex, bins_mid, espect)


def power_index_model(plot_config, show_plot=True):
    """
    Plot power-law index calculated from model

    """
    species = plot_config["species"]
    fig = plt.figure(figsize=[7, 5])
    rect0 = [0.14, 0.57, 0.38, 0.4]
    hgap, vgap = 0.07, 0.04

    pic_runs = ["mime1_sigmae100_vthe04_db1_1024de",
                "sigma64_bg005_4000de_triggered"]

    for irun, pic_run in enumerate(pic_runs):
        pic_run_dir = plot_config["pic_run_dir"]
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        vpic_info = get_vpic_info(pic_run_dir)
        dtwpe_tracer = pic_info.dtwpe * pic_info.tracer_interval
        temp = pic_info.Te if species in ['e', 'electron'] else pic_info.Ti
        fdir = '../data/power_law_index/rates_tracer/' + pic_run + '/'
        fname = fdir + 'ebins.dat'
        ebins = np.fromfile(fname)
        ebins /= temp
        ebins_mid = 0.5 * (ebins[1:] + ebins[:-1])
        nbins, = ebins.shape
        ebins_new = np.logspace(math.log10(ebins.min()),
                                math.log10(ebins.max()),
                                nbins*10)
        fname = fdir + 'nptl_esc.dat'
        nptl_esc = np.fromfile(fname).reshape([-1, nbins-1])
        fname = fdir + 'dnptl_esc.dat'
        dnptl_esc = np.fromfile(fname).reshape([-1, nbins-1])
        fname = fdir + 'nptl_acc.dat'
        nptl_acc = np.fromfile(fname).reshape([-1, nbins-1])
        fname = fdir + 'acc_rate_sum.dat'
        acc_rate_sum = np.fromfile(fname).reshape([-1, nbins-1])
        fname = fdir + 'acc_rate_esc.dat'
        acc_rate_esc = np.fromfile(fname).reshape([-1, nbins-1])
        spect_params = get_spect_params(pic_run)
        power_index = spect_params["power_index"]
        emin, emax = spect_params["energy_range"]
        if irun == 0:
            emax = 2E3
        else:
            emax = 1E2
        nframes, _ = acc_rate_esc.shape

        # eend = int(nframes * 0.9)
        eend = nframes
        acc_rate_sum_r = np.mean(acc_rate_sum[:eend], axis=0)
        acc_rate_esc_r = np.mean(acc_rate_esc[:eend], axis=0)
        nptl_acc_r = np.mean(nptl_acc[:eend], axis=0)
        nptl_esc_r = np.mean(nptl_esc[:eend], axis=0)
        dnptl_esc_r = np.mean(dnptl_esc[:eend], axis=0)
        acc_rate_r = div0(acc_rate_sum_r, nptl_acc_r)
        acc_esc_r = div0(acc_rate_sum_r, nptl_esc_r)
        esc_rate2_r = div0(dnptl_esc_r, nptl_acc_r) / dtwpe_tracer
        rect = np.copy(rect0)
        rect[0] += (rect[2] + hgap) * irun
        ax = fig.add_axes(rect)
        ax.set_prop_cycle('color', COLORS)
        # print(np.asarray(COLORS[0])*256)
        # print(np.asarray(COLORS[1])*256)
        # print(np.asarray(COLORS[2])*256)
        ax.semilogx(ebins_mid, acc_rate_r*1E3, linewidth=2,
                    # marker='o', markersize=5,
                    label=r"$10^3\alpha_\text{acc}$")
        ax.semilogx(ebins_mid, esc_rate2_r*1E3, linewidth=2,
                    # marker='o', markersize=5,
                    label=r"$10^3\alpha_\text{esc}$")
        ylim = ax.get_ylim()
        # ax.plot([emin, emin], ylim, color='k', linewidth=1,
        #         linestyle='--')
        # ax.plot([emax, emax], ylim, color='k', linewidth=1,
        #         linestyle='--')
        ax.fill_betweenx(ylim, emin, emax, alpha=0.2, color='grey')
        ax.set_ylim(ylim)

        ax.text(0.35, 0.05, r"$\alpha_\text{acc}$",
                color=COLORS[0], fontsize=20,
                bbox=dict(facecolor='none', alpha=1.0,
                          edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='bottom',
                transform=ax.transAxes)
        ax.text(0.35, 0.15, r"$\alpha_\text{esc}$",
                color=COLORS[1], fontsize=20,
                bbox=dict(facecolor='none', alpha=1.0,
                          edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='bottom',
                transform=ax.transAxes)
        ax.set_xlim(ebins[0], ebins[-1])
        ax.tick_params(bottom=True, top=True, left=True, right=False)
        ax.tick_params(axis='x', which='minor', direction='out')
        ax.tick_params(axis='x', which='major', direction='out')
        ax.tick_params(axis='y', which='minor', direction='out')
        ax.tick_params(axis='y', which='major', direction='out')
        if irun == 0:
            ax.set_ylabel(r'$10^3$Rates$/\omega_{pe}$', fontsize=20)
        ax.tick_params(axis='x', labelbottom=False)
        ax.tick_params(labelsize=16)

        text1 = "(a)" if irun == 0 else "(c)"
        ax.text(0.03, 0.85, text1,
                color='k', fontsize=20,
                bbox=dict(facecolor='none', alpha=1.0,
                          edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='bottom',
                transform=ax.transAxes)

        rect[1] -= rect[3] + vgap
        ax = fig.add_axes(rect)
        pindex = 1 + div0(esc_rate2_r, acc_rate_r)
        label = r"$1+(\alpha_\text{esc}/\alpha)$"
        # ax.semilogx(ebins_mid, pindex, linewidth=1, label=label,
        #             color='k', linestyle=':')
        pindex += (np.gradient(np.log(acc_rate_r)) /
                   (math.log(ebins[1]) - math.log(ebins[0])))
        pindex -= (np.gradient(np.log(esc_rate2_r)) /
                   (math.log(ebins[1]) - math.log(ebins[0])))
        pindex1 = (np.gradient(np.log(acc_rate_r)) /
                  (math.log(ebins[1]) - math.log(ebins[0])))
        pindex2 = (np.gradient(np.log(esc_rate2_r)) /
                  (math.log(ebins[1]) - math.log(ebins[0])))
        label = (r"$1+(\alpha_\text{esc}/\alpha_\text{acc})+" +
                 r"\partial\ln\alpha_\text{acc}/\partial\ln E-" +
                 r"\partial\ln\alpha_\text{esc}/\partial\ln E$")
        ax.semilogx(ebins_mid, pindex, linewidth=2, color='k',
                    # marker='o', markersize=5,
                    label=label)
        # # ax.semilogx(ebins_mid, pindex1-pindex2, linewidth=2, color='k')
        # ax.semilogx(ebins_mid, pindex1, linewidth=2, color='k')
        # ax.semilogx(ebins_mid, pindex2, linewidth=2, color='k')
        es, _ = find_nearest(ebins_mid, emin)
        ee, _ = find_nearest(ebins_mid, emax)
        pindex_avg = np.mean(pindex[es:ee+1])
        ax.semilogx([ebins[0], ebins[-1]], [pindex_avg, pindex_avg],
                    linewidth=1, linestyle='--', color='k')
        pname = "{%0.2f}" % pindex_avg
        ylim = ax.get_ylim()
        ypos = (pindex_avg - ylim[0]) / (ylim[1] - ylim[0])
        ax.text(0.95, ypos, pname, color='k', fontsize=16,
                bbox=dict(facecolor='none', alpha=1.0,
                          edgecolor='none', pad=10.0),
                horizontalalignment='right', verticalalignment='bottom',
                transform=ax.transAxes)

        ylim = ax.get_ylim()
        # ax.plot([emin, emin], ylim, color='k', linewidth=1,
        #         linestyle='--')
        # ax.plot([emax, emax], ylim, color='k', linewidth=1,
        #         linestyle='--')
        ax.fill_betweenx(ylim, emin, emax, alpha=0.2, color='grey')
        ax.set_ylim(ylim)

        ax.set_xlim(ebins[0], ebins[-1])
        # if irun == 0:
        #     ax.set_ylim([2, 4])
        # else:
        #     ax.set_ylim([1, 3])
        # ax.legend(loc=2, prop={'size': 16}, ncol=1,
        #           shadow=False, fancybox=False, frameon=False)
        ax.tick_params(bottom=True, top=True, left=True, right=False)
        ax.tick_params(axis='x', which='minor', direction='out')
        ax.tick_params(axis='x', which='major', direction='out')
        ax.tick_params(axis='y', which='minor', direction='out')
        ax.tick_params(axis='y', which='major', direction='out')
        ax.tick_params(labelsize=16)
        ax.set_xlabel(r'$(\gamma-1)/T_e$', fontsize=20)
        if irun == 0:
            ax.set_ylabel('Power-law index', fontsize=20)

        text1 = "(b)" if irun == 0 else "(d)"
        ax.text(0.03, 0.85, text1,
                color='k', fontsize=20,
                bbox=dict(facecolor='none', alpha=1.0,
                          edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='bottom',
                transform=ax.transAxes)
    img_dir = '../img/power_law_index/power_index_tracer/'
    mkdir_p(img_dir)
    fname = img_dir + "pindex_model.pdf"
    fig.savefig(fname)
    plt.show()


def power_index_rates(plot_config, show_plot=True):
    """
    Plot power-law index calculated from rates

    """
    species = plot_config["species"]
    fig = plt.figure(figsize=[7, 5])
    rect0 = [0.14, 0.57, 0.38, 0.4]
    hgap, vgap = 0.07, 0.04

    pic_runs = ["mime1_sigmae100_vthe04_db1_1024de",
                "sigma64_bg005_4000de_triggered"]

    for irun, pic_run in enumerate(pic_runs):
        pic_run_dir = plot_config["pic_run_dir"]
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        vpic_info = get_vpic_info(pic_run_dir)
        dtwpe_tracer = pic_info.dtwpe * pic_info.tracer_interval
        temp = pic_info.Te if species in ['e', 'electron'] else pic_info.Ti
        fdir = '../data/power_law_index/rates_tracer/' + pic_run + '/'
        fname = fdir + 'ebins.dat'
        ebins = np.fromfile(fname)
        ebins /= temp
        ebins_mid = 0.5 * (ebins[1:] + ebins[:-1])
        nbins, = ebins.shape
        ebins_new = np.logspace(math.log10(ebins.min()),
                                math.log10(ebins.max()),
                                nbins*10)
        fname = fdir + 'nptl_esc.dat'
        nptl_esc = np.fromfile(fname).reshape([-1, nbins-1])
        fname = fdir + 'dnptl_esc.dat'
        dnptl_esc = np.fromfile(fname).reshape([-1, nbins-1])
        fname = fdir + 'nptl_acc.dat'
        nptl_acc = np.fromfile(fname).reshape([-1, nbins-1])
        fname = fdir + 'acc_rate_sum.dat'
        acc_rate_sum = np.fromfile(fname).reshape([-1, nbins-1])
        fname = fdir + 'acc_rate_esc.dat'
        acc_rate_esc = np.fromfile(fname).reshape([-1, nbins-1])
        spect_params = get_spect_params(pic_run)
        power_index = spect_params["power_index"]
        emin, emax = spect_params["energy_range"]
        if irun == 0:
            emax = 2E3
        else:
            emax = 1E2
        nframes, _ = acc_rate_esc.shape

        mime = pic_info.mime
        n0 = pic_info.n0
        gamma_e = 4.0/3 if pic_info.Te > 0.1 else 1.5  # Adiabatic index
        gamma_i = 5.0/3
        print("Particle initial temperature (Lorentz factor): %f" % temp)
        # In typical VPIC simulation, me=1, c=1
        enthalpy_e =  n0 * (1 + pic_info.Te * gamma_e / (gamma_e - 1))
        enthalpy_i =  n0 * (mime + pic_info.Ti * gamma_i / (gamma_i - 1))
        enthalpy = enthalpy_e + enthalpy_i
        if irun == 0:
            b0 = pic_info.b0 / (2 * math.sqrt(2.0))
            bg = 1.0
        else:
            b0 = pic_info.b0
            bg = 0.05
        # br = b0 * 0.5
        br = b0 * 1.0
        sigma = b0**2 / enthalpy
        sigma_r = br**2 / enthalpy
        sigma_c = b0**2 / (n0 * (1 + mime))
        sigmae_c = b0**2 / n0
        sigmai_c = b0**2 / (n0 * mime)
        sigmae_h = b0**2 / enthalpy_e
        sigmai_h = b0**2 / enthalpy_i
        beta_e = 2 * n0 * pic_info.Te / b0**2
        spect_params = get_spect_params(pic_run)
        pindex = -spect_params["power_index"]
        if irun == 0:
            # va = math.sqrt(sigma / (sigma + 1))
            # va = math.sqrt(sigma / (sigma+sigma*bg**2 + 1))
            va = math.sqrt(sigma_r / (sigma_r+sigma*bg**2 + 1))
        else:
            va = math.sqrt(sigma_r / (sigma_r+sigma*bg**2 + 1))
        gamma_the = pic_info.Te + 1
        vthe = math.sqrt(gamma_the**2 - 1) / gamma_the
        gamma_thi = pic_info.Ti / pic_info.mime + 1
        vthi = math.sqrt(gamma_thi**2 - 1) / gamma_thi
        gamma_va = 1.0 / math.sqrt(1 - va**2)
        factor = gamma_va**2
        btot = math.sqrt(1 + bg**2)
        ctheta = 1.0 / btot
        lx_de = pic_info.lx_di * math.sqrt(pic_info.mime)
        if irun == 0:
            l = lx_de / 16
        else:
            # l = lx_de * 0.45
            l = lx_de / 7.5
        nbins, = ebins_mid.shape
        gamma = ebins_mid * pic_info.Te + 1
        v = np.sqrt(1 - gamma**-2)
        omega0 = btot * b0 / gamma
        rg = v / omega0
        di = np.sqrt(pic_info.mime * gamma_thi)
        rhoi = gamma_thi * pic_info.mime / (btot * b0)
        rmin = min(di, rhoi) * (1 + bg**2)
        l1 = l * (1 + bg**2) * np.ones(nbins)
        l0 = np.zeros(l1.shape)
        l0[rg>rmin] = rg[rg>rmin]
        l0[rg<rmin] = rmin
        l0 *= (1 + bg**2)

        lc = l
        sigma0 = 1.0 / bg**2
        if sigma0 > 1:
            sigma0 = 1.0
        # sigma0 = 1.0 / (1.0 + bg**2)**(1/3)
        csc_3pi_5 = 1.05146
        kpara = 3 * v**3 * csc_3pi_5 / (20 * lc * omega0**2 * sigma0)
        kpara *= 1.0 + (72.0 / 7) * (omega0 * lc / v)**(5/3)

        acc_rate = (factor * (1 + va + va**2) - 1) / (4 * l)
        if irun == 0:
            esc_rate = 1 / l
        else:
            # esc_rate = va / l
            # esc_rate = 1.0 / (l * (1 + bg**2))
            esc_rate = 1.0 / l / gamma_va
        lmfp = bg**2 * lc
        dee_rate = gamma_va**2 * va**2 / lmfp / 3
        # if irun == 0:
        #     acc_rate_model = 2 * dee_rate * (1.0 - np.log(ebins_mid/emin)/4)
        #     esc_rate_model = esc_rate * (1.0 - np.log(ebins_mid/emin)/4)
        # else:
        #     acc_rate_model = acc_rate * (1.0 - np.log(ebins_mid/emin)/4)
        #     esc_rate_model = esc_rate * np.ones(ebins_mid.shape)
        acc_rate_model = ((gamma+1)/(3*gamma)) * va * np.log(l1/l0) / (l1-l0)
        if irun == 0:
            esc_rate_model = va**2 / kpara
        else:
            acc_rate_model /= gamma_va
            esc_rate_model = esc_rate * np.ones(ebins_mid.shape)

        # eend = int(nframes * 0.9)
        eend = nframes
        acc_rate_sum_r = np.mean(acc_rate_sum[:eend], axis=0)
        acc_rate_esc_r = np.mean(acc_rate_esc[:eend], axis=0)
        nptl_acc_r = np.mean(nptl_acc[:eend], axis=0)
        nptl_esc_r = np.mean(nptl_esc[:eend], axis=0)
        dnptl_esc_r = np.mean(dnptl_esc[:eend], axis=0)
        acc_rate_r = div0(acc_rate_sum_r, nptl_acc_r)
        acc_esc_r = div0(acc_rate_sum_r, nptl_esc_r)
        esc_rate2_r = div0(dnptl_esc_r, nptl_acc_r) / dtwpe_tracer
        rect = np.copy(rect0)
        rect[0] += (rect[2] + hgap) * irun
        ax = fig.add_axes(rect)
        ax.set_prop_cycle('color', COLORS)
        # print(np.asarray(COLORS[0])*256)
        # print(np.asarray(COLORS[1])*256)
        # print(np.asarray(COLORS[2])*256)
        ax.semilogx(ebins_mid, acc_rate_r*1E3, linewidth=2,
                    linestyle='None', marker='o',
                    label=r"$10^3\alpha_\text{acc}$")
        ax.semilogx(ebins_mid, esc_rate2_r*1E3, linewidth=2,
                    linestyle='None', marker='o',
                    label=r"$10^3\alpha_\text{esc}$")
        ax.semilogx(ebins_mid, acc_rate_model*1E3, linewidth=1,
                    color='k', label=r"$10^3\alpha_\text{acc}$")
        ax.semilogx(ebins_mid, esc_rate_model*1E3, linewidth=1,
                    color='k', label=r"$10^3\alpha_\text{esc}$")
        if irun == 0:
            ax.set_ylim([1.0, 8.0])
        else:
            ax.set_ylim([0.2, 1.0])
        ylim = ax.get_ylim()
        # ax.plot([emin, emin], ylim, color='k', linewidth=1,
        #         linestyle='--')
        # ax.plot([emax, emax], ylim, color='k', linewidth=1,
        #         linestyle='--')
        ax.fill_betweenx(ylim, emin, emax, alpha=0.2, color='grey')
        ax.set_ylim(ylim)

        ax.text(0.35, 0.05, r"$\alpha_\text{acc}$",
                color=COLORS[0], fontsize=20,
                bbox=dict(facecolor='none', alpha=1.0,
                          edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='bottom',
                transform=ax.transAxes)
        ax.text(0.35, 0.15, r"$\alpha_\text{esc}$",
                color=COLORS[1], fontsize=20,
                bbox=dict(facecolor='none', alpha=1.0,
                          edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='bottom',
                transform=ax.transAxes)
        ax.set_xlim(ebins[0], ebins[-1])
        ax.tick_params(bottom=True, top=True, left=True, right=False)
        ax.tick_params(axis='x', which='minor', direction='out')
        ax.tick_params(axis='x', which='major', direction='out')
        ax.tick_params(axis='y', which='minor', direction='out')
        ax.tick_params(axis='y', which='major', direction='out')
        if irun == 0:
            ax.set_ylabel(r'$10^3$Rates$/\omega_{pe}$', fontsize=20)
        ax.tick_params(axis='x', labelbottom=False)
        ax.tick_params(labelsize=16)

        text1 = "(a)" if irun == 0 else "(c)"
        ax.text(0.03, 0.85, text1,
                color='k', fontsize=20,
                bbox=dict(facecolor='none', alpha=1.0,
                          edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='bottom',
                transform=ax.transAxes)

        rect[1] -= rect[3] + vgap
        ax = fig.add_axes(rect)
        lx_de = pic_info.lx_di * math.sqrt(pic_info.mime)
        params= {"acc_rate": acc_rate,
                 "esc_rate": esc_rate,
                 "dee_rate": dee_rate,
                 "nsteps": int(lx_de)*2,
                 "nptl": 1E6}
        # p0,_,_ = solve_sde_pindex(params, pic_run, reconnection=False)
        pindex = 1 + div0(esc_rate2_r, acc_rate_r)
        label = r"$1+(\alpha_\text{esc}/\alpha)$"
        # ax.semilogx(ebins_mid, pindex, linewidth=1, label=label,
        #             color='k', linestyle=':')
        pindex += (np.gradient(np.log(acc_rate_r)) /
                   (math.log(ebins[1]) - math.log(ebins[0])))
        pindex -= (np.gradient(np.log(esc_rate2_r)) /
                   (math.log(ebins[1]) - math.log(ebins[0])))
        pindex1 = (np.gradient(np.log(acc_rate_r)) /
                  (math.log(ebins[1]) - math.log(ebins[0])))
        pindex2 = (np.gradient(np.log(esc_rate2_r)) /
                  (math.log(ebins[1]) - math.log(ebins[0])))
        label = (r"$1+(\alpha_\text{esc}/\alpha_\text{acc})+" +
                 r"\partial\ln\alpha_\text{acc}/\partial\ln E-" +
                 r"\partial\ln\alpha_\text{esc}/\partial\ln E$")
        ax.semilogx(ebins_mid, pindex, linewidth=2, color='k',
                    # marker='o', markersize=5,
                    label=label)
        # # ax.semilogx(ebins_mid, pindex1-pindex2, linewidth=2, color='k')
        # ax.semilogx(ebins_mid, pindex1, linewidth=2, color='k')
        # ax.semilogx(ebins_mid, pindex2, linewidth=2, color='k')
        es, _ = find_nearest(ebins_mid, emin)
        ee, _ = find_nearest(ebins_mid, emax)
        pindex_avg = np.mean(pindex[es:ee+1])
        ax.semilogx([ebins[0], ebins[-1]], [pindex_avg, pindex_avg],
                    linewidth=1, linestyle='--', color='k')
        pname = "{%0.2f}" % pindex_avg
        ylim = ax.get_ylim()
        ypos = (pindex_avg - ylim[0]) / (ylim[1] - ylim[0])
        ax.text(0.95, ypos, pname, color='k', fontsize=16,
                bbox=dict(facecolor='none', alpha=1.0,
                          edgecolor='none', pad=10.0),
                horizontalalignment='right', verticalalignment='bottom',
                transform=ax.transAxes)

        ylim = ax.get_ylim()
        # ax.plot([emin, emin], ylim, color='k', linewidth=1,
        #         linestyle='--')
        # ax.plot([emax, emax], ylim, color='k', linewidth=1,
        #         linestyle='--')
        ax.fill_betweenx(ylim, emin, emax, alpha=0.2, color='grey')
        ax.set_ylim(ylim)

        ax.set_xlim(ebins[0], ebins[-1])
        # if irun == 0:
        #     ax.set_ylim([2, 4])
        # else:
        #     ax.set_ylim([1, 3])
        # ax.legend(loc=2, prop={'size': 16}, ncol=1,
        #           shadow=False, fancybox=False, frameon=False)
        ax.tick_params(bottom=True, top=True, left=True, right=False)
        ax.tick_params(axis='x', which='minor', direction='out')
        ax.tick_params(axis='x', which='major', direction='out')
        ax.tick_params(axis='y', which='minor', direction='out')
        ax.tick_params(axis='y', which='major', direction='out')
        ax.tick_params(labelsize=16)
        ax.set_xlabel(r'$(\gamma-1)/T_e$', fontsize=20)
        if irun == 0:
            ax.set_ylabel('Power-law index', fontsize=20)

        text1 = "(b)" if irun == 0 else "(d)"
        ax.text(0.03, 0.85, text1,
                color='k', fontsize=20,
                bbox=dict(facecolor='none', alpha=1.0,
                          edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='bottom',
                transform=ax.transAxes)
    img_dir = '../img/power_law_index/power_index_tracer/'
    mkdir_p(img_dir)
    fname = img_dir + "pindex_model.pdf"
    fig.savefig(fname)
    plt.show()


def power_law_scaling(plot_config, args):
    """Analysis for multiple runs
    """
    pic_runs = []
    sigmas = ["01", "04", "16", "64"]
    bgs = ["005", "01", "02", "04", "10"]
    nsigma = len(sigmas)
    nbg = len(bgs)
    for sigma in sigmas:
        for bg in bgs:
            pic_run = "sigma" + sigma + "_bg" + bg + "_4000de_triggered"
            pic_runs.append(pic_run)
    sigmaes = ["100", "40", "10"]
    db2s = ["05", "1", "4"]
    nsigmae = len(sigmaes)
    ndb = len(db2s)
    for sigmae in sigmaes:
        for db2 in db2s:
            pic_run = "mime1_sigmae" + sigmae + "_vthe04_db" + db2 + "_1024de"
            pic_runs.append(pic_run)
    pic_runs.append("sigmae400_bg00_800de_triggered")
    # pic_runs.append("3D-Lx150-bg0.2-150ppc-2048KNL-tracking")
    nruns = len(pic_runs)
    ps = np.zeros((2, nruns))
    for irun, pic_run in enumerate(pic_runs):
        picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        print(pic_info.nx, pic_info.nz)
        if "3D-Lx150" in pic_run:
            n0 = 1.0
            vthe = pic_info.vthe
            gama = 1.0 / math.sqrt(1.0 - 3 * vthe**2)
            te = gama - 1.0
        else:
            n0 = pic_info.n0
            te = pic_info.Te
        ti = pic_info.ti_te * te
        mime = pic_info.mime
        gamma_e = 4.0/3 if te > 0.1 else 1.5  # Adiabatic index
        gamma_i = 5.0/3
        enthalpy_e =  n0 * (1 + te * gamma_e / (gamma_e - 1))
        enthalpy_i =  n0 * (mime + ti * gamma_i / (gamma_i - 1))
        enthalpy = enthalpy_e + enthalpy_i
        sigma = pic_info.b0**2 / enthalpy
        va = math.sqrt(sigma / (sigma + 1))
        gva = va / math.sqrt(1 - va**2)
        print("Alfven 4-velocity: %f" % gva)
        fdir = '../data/power_law_index/power_index_tracer_overall/'
        fname = fdir + "pindex_tracer_" + pic_run + ".dat"
        fdata = np.fromfile(fname)
        ps[0, irun] = fdata[0]
        spect_params = get_spect_params(pic_run)
        pindex = spect_params["power_index"]
        ps[1, irun] = -pindex
    fig = plt.figure(figsize=[7, 7])
    rect = [0.11, 0.12, 0.85, 0.85]
    hgap, vgap = 0.07, 0.04
    ax = fig.add_axes(rect)
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    ax.set_prop_cycle('color', COLORS)
    n1 = nsigma * nbg + nsigmae * ndb
    ltext = r"$\sigma_e=400, m_i/m_e=1$"
    p3 = ax.scatter(ps[1, n1], ps[0, n1], marker='X', label=ltext, s=100)
    # ltext = r"3D Reconnection, $m_i/m_e=25, \beta_e=0.02$"
    # p1 = ax.scatter(ps[0, n1+1], ps[1, n1+1], marker='*', label=ltext, s=100)
    plots_rec = [p3]
    for i in range(nsigma):
        sigma = str(int(sigmas[i]) / 10)
        ltext = r"$\sigma_i=" + sigma + "$"
        alphas = np.linspace(0.2, 1, nbg)
        rgba_colors = np.zeros((nbg,4))
        rgba_colors[:,0] = COLORS[i+1][0]
        rgba_colors[:,1] = COLORS[i+1][1]
        rgba_colors[:,2] = COLORS[i+1][2]
        # the fourth column needs to be your alphas
        rgba_colors[:, 3] = alphas
        rs, re = i*nbg, (i+1)*nbg
        ax.scatter(ps[1, rs:re], ps[0, rs:re], color=rgba_colors,
                   marker='o', s=100)
        p1 = ax.scatter(ps[1, rs:re]+10, ps[0, rs:re]+10,
                        label=ltext, color=COLORS[i+1],
                        marker='o', s=100)
        plots_rec.append(p1)
    # Turbulence run
    shift = nsigma * nbg
    plots_turb = []
    for i in range(nsigmae):
        ltext = r"$\sigma_e=" + sigmaes[i] + "$"
        ndb = len(db2s)
        alphas = np.linspace(0.2, 1, ndb)
        rgba_colors = np.zeros((ndb,4))
        rgba_colors[:,0] = COLORS[i+nsigma+1][0]
        rgba_colors[:,1] = COLORS[i+nsigma+1][1]
        rgba_colors[:,2] = COLORS[i+nsigma+1][2]
        # the fourth column needs to be your alphas
        rgba_colors[:, 3] = alphas
        rs, re = i*ndb+shift, (i+1)*ndb+shift
        ax.scatter(ps[1, rs:re], ps[0, rs:re], color=rgba_colors,
                   marker='d', s=100)
        p1 = ax.scatter(ps[1, rs:re]+10, ps[0, rs:re]+10,
                        label=ltext, color=COLORS[i+nsigma+1],
                        marker='d', s=100)
        plots_turb.append(p1)

    leg1 = ax.legend(loc=2, handles=plots_rec, prop={'size': 16}, ncol=1,
                     shadow=False, fancybox=False, frameon=False,
                     title="Reconnection", title_fontsize=16)

    leg2 = ax.legend(loc=4, handles=plots_turb, prop={'size': 16}, ncol=1,
                     shadow=False, fancybox=False, frameon=False,
                     title="Turbulence", title_fontsize=16)
    ax.add_artist(leg1)
    brace = CurlyBrace(x=0.28, y=0.665, width=0.05, height=.2, pointing='right',
                       transform=ax.transAxes, color='k')
    ax.add_artist(brace)
    text1 = r"$m_i/m_e=1836$"
    ax.text(0.34, 0.765, text1, color='k', fontsize=16, rotation=0,
            bbox=dict(facecolor='none', alpha=1.0,
                      edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)

    slope, intercept, r_value, p_value, std_err = stats.linregress(ps[1], ps[0])
    # print(slope, intercept, r_value, p_value, std_err)

    pmodel = np.linspace(1, 5, 10)
    ax.plot(pmodel, intercept + slope*pmodel, color='k',
            alpha=0.5, label='fitted line', zorder=0)

    text1 = (r"$p_m=" + (r"%0.3f" % slope) + r"p_s" + (r"%0.3f" % intercept) + r"$")
    ax.text(0.77, 0.87, text1, color='k', fontsize=12, rotation=47.11,
            bbox=dict(facecolor='none', alpha=1.0,
                      edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes)
    text2 = (r"$R^2=" + (r"%0.3f" % r_value) + r"$")
    ax.text(0.75, 0.9, text2, color='k', fontsize=12, rotation=47.11,
            bbox=dict(facecolor='none', alpha=1.0,
                      edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes)

    ax.plot([1, 5.0], [1, 5.0], linewidth=1, linestyle='--', color='k')
    ax.set_xlim([1, 5.0])
    ax.set_ylim([1, 5.0])
    ax.set_xticks(range(1, 6))
    ax.set_yticks(range(1, 6))
    ax.tick_params(bottom=True, top=True, left=True, right=False)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlabel(r'Power-law index $p_s$ (simulation)', fontsize=20)
    ax.set_ylabel(r'Power-law index $p_m$ (model)', fontsize=20)
    ax.tick_params(labelsize=16)
    img_dir = '../img/power_law_index/power_index_tracer/'
    mkdir_p(img_dir)
    fname = img_dir + "pindex_scaling.pdf"
    fig.savefig(fname)
    plt.show()


def plot_nacc_nesc(plot_config, args):
    """Plot the number of accelerating particles and escaped particles
    """
    pic_runs = []
    # sigmas = ["01", "04", "16", "64"]
    sigmas = ["01"]
    bgs = ["005", "01", "02", "04", "10"]
    nsigma = len(sigmas)
    nbg = len(bgs)
    for sigma in sigmas:
        for bg in bgs:
            pic_run = "sigma" + sigma + "_bg" + bg + "_4000de_triggered"
            pic_runs.append(pic_run)
    # sigmaes = ["100", "40", "10"]
    # db2s = ["05", "1", "4"]
    # nsigmae = len(sigmaes)
    # ndb = len(db2s)
    # for sigmae in sigmaes:
    #     for db2 in db2s:
    #         pic_run = "mime1_sigmae" + sigmae + "_vthe04_db" + db2 + "_1024de"
    #         pic_runs.append(pic_run)
    # pic_runs.append("sigmae400_bg00_800de_triggered")
    # # pic_runs.append("3D-Lx150-bg0.2-150ppc-2048KNL-tracking")
    nruns = len(pic_runs)
    ps = np.zeros((2, nruns))
    for irun, pic_run in enumerate(pic_runs):
        print(pic_run)
        fdir = '../data/power_law_index/rates_tracer/' + pic_run + '/'
        fname = fdir + 'ebins.dat'
        ebins = np.fromfile(fname)
        ebins_mid = 0.5 * (ebins[1:] + ebins[:-1])
        nbins, = ebins.shape
        fname = fdir + 'nptl_esc.dat'
        nesc_band = np.fromfile(fname).reshape([-1, nbins-1])
        nptl_esc = np.sum(nesc_band, axis=1)
        fname = fdir + 'nptl_acc.dat'
        nacc_band = np.fromfile(fname).reshape([-1, nbins-1])
        nptl_acc = np.sum(nacc_band, axis=1)
        nptl_tot = nptl_acc + nptl_esc
        fig = plt.figure(figsize=[7, 5])
        rect = [0.11, 0.12, 0.85, 0.85]
        hgap, vgap = 0.07, 0.04
        ax = fig.add_axes(rect)
        COLORS = palettable.tableau.Tableau_10.mpl_colors
        ax.set_prop_cycle('color', COLORS)
        ax.plot(nptl_acc / nptl_tot)
        ax.plot(nptl_esc / nptl_tot)
        # nesc_band = gaussian_filter(nesc_band+1, sigma=[5, 1])
        # nacc_band = gaussian_filter(nacc_band+1, sigma=[5, 1])
        # ax.plot(np.diff(nesc_band, axis=0) / nacc_band[:-1, :])
        ax.grid()

        ax.tick_params(bottom=True, top=True, left=True, right=False)
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')
        ax.tick_params(labelsize=16)
        img_dir = '../img/power_law_index/nacc_nesc/'
        mkdir_p(img_dir)
        # fname = img_dir + "nacc_nesc_" + pic_run + ".pdf"
        # fig.savefig(fname)
        # plt.close()
        plt.show()


def plot_absj(plot_config, show_plot=True):
    """Plot current density
    """
    sigmas = ["001", "004", "016", "064", "256"]
    bgs = ["00", "05"]
    bgs_float = [0.0, 0.5]
    nsigma = len(sigmas)
    nbg = len(bgs)
    axs = []
    fig = plt.figure(figsize=[20, 8])
    rect0 = [0.06, 0.76, 0.15, 0.2]
    hgap, vgap = 0.04, 0.03
    for isigma in range(nsigma):
        rect = np.copy(rect0)
        rect[0] += isigma * (rect[2] + hgap)
        for ibg in range(nbg):
            ax = fig.add_axes(rect)
            axs.append(ax)
            rect[1] -= rect[3] + vgap
            ax.set_prop_cycle('color', COLORS)
            ax.tick_params(bottom=True, top=True, left=True, right=True)
            ax.tick_params(axis='x', which='minor', direction='in')
            ax.tick_params(axis='x', which='major', direction='in')
            ax.tick_params(axis='y', which='minor', direction='in')
            ax.tick_params(axis='y', which='major', direction='in')
            if ibg == nbg - 1:
                ax.set_xlabel(r'$x/d_e$', fontsize=16)
            else:
                ax.tick_params(axis='x', labelbottom=False)
            if isigma == 0:
                ax.set_ylabel(r'$z/d_e$', fontsize=16)
            ax.tick_params(labelsize=12)
    tframe = plot_config["tframe"]
    for isigma, sigma in enumerate(sigmas):
        for ibg, bg in enumerate(bgs):
            pic_run = "mime1836_sigmaic" + sigma + "_bg" + bg
            picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
            pic_info = read_data_from_json(picinfo_fname)
            pic_run_dir = pic_info.run_dir
            fields_interval = pic_info.fields_interval
            tindex = fields_interval * tframe
            smime = math.sqrt(pic_info.mime)
            lx_de = pic_info.lx_di * smime
            lz_de = pic_info.lz_di * smime
            xmin, xmax = 0, lx_de
            zmin, zmax = -0.5 * lz_de, 0.5 * lz_de

            fname = (pic_run_dir + "hydro_hdf5/T." + str(tindex) +
                     "/hydro_electron_" + str(tindex) + ".h5")
            je = {}
            with h5py.File(fname, 'r') as fh:
                group = fh["Timestep_" + str(tindex)]
                for var in ["jx", "jy", "jz"]:
                    dset = group[var]
                    je[var] = np.zeros(dset.shape, dtype=dset.dtype)
                    dset.read_direct(je[var])

            fname = (pic_run_dir + "hydro_hdf5/T." + str(tindex) +
                     "/hydro_ion_" + str(tindex) + ".h5")
            ji = {}
            with h5py.File(fname, 'r') as fh:
                group = fh["Timestep_" + str(tindex)]
                for var in ["jx", "jy", "jz"]:
                    dset = group[var]
                    ji[var] = np.zeros(dset.shape, dtype=dset.dtype)
                    dset.read_direct(ji[var])

            xgrid = np.linspace(xmin, xmax, pic_info.nx)
            zgrid = np.linspace(zmin, zmax, pic_info.nz)

            absj = np.squeeze(np.sqrt((je["jx"] + ji["jx"])**2 +
                                      (je["jy"] + ji["jy"])**2 +
                                      (je["jz"] + ji["jz"])**2))
            ax = axs[isigma*nbg + ibg]
            im1 = ax.imshow(absj.T,
                            extent=[xmin, xmax, zmin, zmax],
                            norm = LogNorm(vmin=1E-2, vmax=1E1),
                            cmap=plt.cm.viridis, aspect='auto',
                            origin='lower', interpolation='bicubic')

            # # Magnetic field lines
            # kwargs = {"current_time": tframe,
            #           "xl": 0, "xr": pic_info.lx_di,
            #           "zb": -0.5*pic_info.lz_di, "zt": 0.5*pic_info.lz_di}
            # fname = pic_run_dir + "data/Ay.gda"
            # x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
            # ax.contour(xgrid, zgrid, Ay, colors='k', linewidths=0.5,
            #            levels=np.linspace(np.min(Ay), np.max(Ay), 20))

            ax.set_xlim([xmin, xmax])
            ax.set_ylim([zmin, zmax])
            # rect_cbar = np.copy(rect)
            # rect_cbar[0] += rect[2] + 0.01
            # rect_cbar[2] = 0.02
            # rect_cbar[1] += rect[3] * 0.25
            # rect_cbar[3] = rect[3] * 0.5
            # cbar_ax = fig.add_axes(rect_cbar)
            # cbar = fig.colorbar(im1, cax=cbar_ax, extend='max')
            # cbar_ax.tick_params(bottom=False, top=False, left=False, right=True)
            # cbar_ax.tick_params(axis='y', which='major', direction='out')
            # cbar_ax.tick_params(axis='y', which='minor', direction='in', right=False)
            # cbar_ax.set_title(r'$|\boldsymbol{J}|$', fontsize=20)
            # cbar.ax.tick_params(labelsize=12)
            # twpe = math.ceil(tindex * pic_info.dtwpe / 0.1) * 0.1
    # text1 = r'$t\omega_{pe}=' + ("{%0.0f}" % twpe) + '$'
    # fig.suptitle(text1, fontsize=20)
    # img_dir = '../img/rate_problem/absj/' + pic_run + '/'
    # mkdir_p(img_dir)
    # fname = img_dir + "absj_" + str(tframe) + ".jpg"
    # fig.savefig(fname, dpi=200)

    if show_plot:
        plt.show()
    else:
        plt.close()


def process_run(plot_config, args, pic_run):
    """process one run"""
    plot_config["pic_run"] = pic_run
    if args.rates_tracer:
        calc_rates_tracer(plot_config, args.show_plot)
    elif args.ve_kappa_dist:
        vexb_kappa_dist(plot_config, args.show_plot)
    elif args.vel_dist:
        velocity_dist(plot_config, args.show_plot)


def analysis_multi_runs(plot_config, args):
    """Analysis for multiple runs
    """
    pic_runs = []
    # sigmas = ["01", "04", "16", "64"]
    # bgs = ["005", "01", "02", "04", "10"]
    # for sigma in sigmas:
    #     for bg in bgs:
    #         pic_run = "sigma" + sigma + "_bg" + bg + "_4000de_triggered"
    #         pic_runs.append(pic_run)
    # pic_runs.append("mime1_sigmae100_vthe04_db1_1024de")
    # pic_runs.append("sigmae400_bg00_800de_triggered")
    # pic_runs.append("3D-Lx150-bg0.2-150ppc-2048KNL-tracking")
    # pic_runs.append("mime1836_sigmaic001_bg05")
    # pic_runs.append("mime1836_sigmaic004_bg05")
    # pic_runs.append("mime1836_sigmaic016_bg05")
    # pic_runs.append("mime1836_sigmaic064_bg05")
    # pic_runs.append("mime1836_sigmaic256_bg05")
    pic_runs.append("mime1836_sigmaic001_bg00")
    pic_runs.append("mime1836_sigmaic004_bg00")
    pic_runs.append("mime1836_sigmaic001_bg10")
    pic_runs.append("mime1836_sigmaic004_bg10")
    pic_runs.append("mime1836_sigmaic016_bg10")
    pic_runs.append("mime1836_sigmaic064_bg10")
    pic_runs.append("mime1836_sigmaic256_bg10")
    pic_runs.append("mime1836_sigmaic001_bg20")
    pic_runs.append("mime1836_sigmaic004_bg20")
    pic_runs.append("mime1836_sigmaic016_bg20")
    pic_runs.append("mime1836_sigmaic064_bg20")
    pic_runs.append("mime1836_sigmaic256_bg20")
    ncores = multiprocessing.cpu_count()
    # ncores = 18
    Parallel(n_jobs=ncores)(delayed(process_run)(plot_config, args, pic_run)
                            for pic_run in pic_runs)


def get_cmd_args():
    """Get command line arguments
    """
    # default_pic_run = 'sigmae100_bg005_800de_triggered'
    default_pic_run = 'sigma64_bg005_4000de_triggered'
    default_pic_run_dir = ('/net/scratch4/xiaocanli/reconnection/power_law_index/' +
                           default_pic_run + '/')
    parser = argparse.ArgumentParser(description='Analysis for runs to determine power-law indices')
    parser.add_argument('--pic_run', action="store",
                        default=default_pic_run, help='PIC run name')
    parser.add_argument('--pic_run_dir', action="store",
                        default=default_pic_run_dir, help='PIC run directory')
    parser.add_argument('--species', action="store",
                        default="e", help='Particle species')
    parser.add_argument('--tframe', action="store", default='20', type=int,
                        help='Time frame')
    parser.add_argument('--multi_frames', action="store_true", default=False,
                        help='whether to analyze multiple frames')
    parser.add_argument('--time_loop', action="store_true", default=False,
                        help='whether to use a time loop to analyze multiple frames')
    parser.add_argument('--tstart', action="store", default='0', type=int,
                        help='starting time frame')
    parser.add_argument('--tend', action="store", default='40', type=int,
                        help='ending time frame')
    parser.add_argument('--multi_runs', action="store_true", default=False,
                        help='whether to analyze multiple runs')
    parser.add_argument('--vkappa_threshold', action="store", default='4E-2',
                        type=float, help='the threshold for vexb_dot_kappa')
    parser.add_argument('--all_frames', action="store_true", default=False,
                        help='whether to analyze all frames')
    parser.add_argument('--show_plot', action="store_true", default=False,
                        help='whether to show plot')
    parser.add_argument('--plot_spect', action="store_true", default=False,
                        help='whether to plot particle energy spectrum')
    parser.add_argument('--check_density', action="store_true", default=False,
                        help='whether to check maximum density')
    parser.add_argument('--fluid_ene', action="store_true", default=False,
                        help='whether to plot fluid energization')
    parser.add_argument('--fluid_ene_frac', action="store_true", default=False,
                        help='whether to calculate fluid energization fraction')
    parser.add_argument('--particle_ene', action="store_true", default=False,
                        help='whether to plot particle energization')
    parser.add_argument('--plot_type', action="store", default='total', type=str,
                        help='Particle plot type')
    parser.add_argument('--multi_types', action="store_true", default=False,
                        help='Multiple particle plot types')
    parser.add_argument('--calc_vexb_kappa', action="store_true", default=False,
                        help='whether to calculate vexb dot magnetic curvature')
    parser.add_argument('--calc_curv_radius', action="store_true", default=False,
                        help='whether to calculate the radius magnetic curvature')
    parser.add_argument('--calc_vdotE', action="store_true", default=False,
                        help='whether to calculate velocity dot electric field')
    parser.add_argument('--vel_profile', action="store_true", default=False,
                        help='whether to get the velocity profile')
    parser.add_argument('--plot_vexb_kappa', action="store_true", default=False,
                        help='whether to plot vexb dot magnetic curvature')
    parser.add_argument('--comp_vexb_kappa', action="store_true", default=False,
                        help='whether to compare vexb_dot_kappa')
    parser.add_argument('--spect_species', action="store_true", default=False,
                        help='energy spectrum for different species')
    parser.add_argument('--acc_esc_rate', action="store_true", default=False,
                        help='calculate particle acceleration and escape rates')
    parser.add_argument('--escape_boundary', action="store_true", default=False,
                        help='get the escape boundary by checking vkappa')
    parser.add_argument('--rates_vkappa', action="store_true", default=False,
                        help='calculate acceleration and escape rates based ' +
                        'distributions of accelerations rate binned with vkappa')
    parser.add_argument('--all_runs', action="store_true", default=False,
                        help='whether to do analysis for all runs')
    parser.add_argument('--econv', action="store_true", default=False,
                        help='whether to plot energy conversion')
    parser.add_argument('--sigma_power', action="store_true", default=False,
                        help='whether to plot sigma parameters versus power-law index')
    parser.add_argument('--ana_power', action="store_true", default=False,
                        help='whether to plot analytical power-law index')
    parser.add_argument('--acc_rate', action="store_true", default=False,
                        help='whether to plot acceleration rate')
    parser.add_argument('--calc_flow_acc', action="store_true", default=False,
                        help='whether to calculate flow acceleration')
    parser.add_argument('--spect_bg', action="store_true", default=False,
                        help='whether to plot spectrum changing with Bg')
    parser.add_argument('--spect_sigma_bg', action="store_true", default=False,
                        help='whether to plot spectrum changing with sigma and Bg')
    parser.add_argument('--sigma_type', action="store", default='sigma01', type=str,
                        help='Run with specific sigma parameters')
    parser.add_argument('--bg', action="store", default='005', type=int,
                        help='Guide field code')
    parser.add_argument('--acc_rate_std', action="store_true", default=False,
                        help='whether to plot the standard deviation of the acceleration rates')
    parser.add_argument('--calc_dee', action="store_true", default=False,
                        help='whether to calculate energy diffusion coefficient')
    parser.add_argument('--calc_acc_rate', action="store_true", default=False,
                        help='whether to calculate particle acceleration rate')
    parser.add_argument('--plot_traj', action="store_true", default=False,
                        help='whether to plot tracer particle trajectory')
    parser.add_argument('--plot_traj_j', action="store_true", default=False,
                        help='whether to plot tracer particle trajectory with current density')
    parser.add_argument('--rates_tracer', action="store_true", default=False,
                        help='whether to calculate rates using tracer')
    parser.add_argument('--plot_rates_tracer', action="store_true", default=False,
                        help='whether to plot rates using tracer')
    parser.add_argument('--rates_sigma_bg', action="store_true", default=False,
                        help='whether to plot rates scaling with sigma and bg')
    parser.add_argument('--get_traj', action="store_true", default=False,
                        help='whether to get trajectory from tracer files')
    parser.add_argument('--traj_band', action="store_true", default=False,
                        help='whether to plot tracer particle trajectory in different band')
    parser.add_argument('--dee_tracer', action="store_true", default=False,
                        help='whether to calculate energy diffusion using tracer files')
    parser.add_argument('--plot_dee_tracer', action="store_true", default=False,
                        help='whether to plot energy diffusion using tracer files')
    parser.add_argument('--uout_max', action="store_true", default=False,
                        help='whether to get the maximum of the reconnection outflow')
    parser.add_argument('--high_rate', action="store_true", default=False,
                        help='whether to plot the change rate of the number of high-energy particles')
    parser.add_argument('--spect_four', action="store_true", default=False,
                        help='whether to plot spectra for four runs')
    parser.add_argument('--solve_sde', action="store_true", default=False,
                        help='whether to solve the SDE')
    parser.add_argument('--pindex_model', action="store_true", default=False,
                        help='whether to plot power-law index from the model')
    parser.add_argument('--pindex_scaling', action="store_true", default=False,
                        help='whether to plot the scaling of the power-law indices')
    parser.add_argument('--nacc_nesc', action="store_true", default=False,
                        help='plot the number of accelerating and escaped particles')
    parser.add_argument('--ve_kappa_dist', action="store_true", default=False,
                        help='get the 2D distribution of vexb_kappa and kappa')
    parser.add_argument('--test_calc_rates', action="store_true", default=False,
                        help='test the method to calculate rates')
    parser.add_argument('--vel_dist', action="store_true", default=False,
                        help='whether to get the distribution of velocity')
    parser.add_argument('--plot_absj', action="store_true", default=False,
                        help='whether to plot current density')
    return parser.parse_args()


def analysis_single_frames(plot_config, args):
    """Analysis for multiple time frames
    """
    tframe = args.tframe
    if args.plot_spect:
        if args.all_frames:
            plot_spectrum_multi(plot_config)
        if args.all_runs:
            plot_spectrum_all_runs(plot_config)
    elif args.econv:
        if args.all_runs:
            energy_conversion_all_runs(plot_config)
    elif args.check_density:
        check_density(plot_config)
    elif args.fluid_ene:
        fluid_energization(plot_config, show_plot=args.show_plot)
    elif args.fluid_ene_frac:
        fluid_ene_fraction(plot_config, show_plot=False)
    elif args.particle_ene:
        if args.multi_types:
            particle_energization_multi(plot_config)
        else:
            particle_energization(plot_config)
    elif args.calc_vexb_kappa:
        calc_vexb_kappa(plot_config)
    elif args.calc_curv_radius:
        calc_curvature_radius(plot_config)
    elif args.calc_vdotE:
        calc_vdotE(plot_config)
    elif args.plot_vexb_kappa:
        plot_vexb_kappa(plot_config)
    elif args.comp_vexb_kappa:
        comp_vexb_kappa(plot_config)
    elif args.vel_profile:
        velocity_profile(plot_config)
    elif args.spect_species:
        plot_spect_species(plot_config, args.show_plot)
    elif args.acc_esc_rate:
        acc_esc_rate(plot_config, args.show_plot)
    elif args.escape_boundary:
        vkappa_escape_boundary(plot_config, args.show_plot)
    elif args.rates_vkappa:
        rates_based_vkappa(plot_config, args.show_plot)
    elif args.sigma_power:
        plot_sigma_power(plot_config, args.show_plot)
    elif args.ana_power:
        analytical_power_index(plot_config, args.show_plot)
    elif args.acc_rate:
        acceleration_rate(plot_config, show_plot=True)
    elif args.calc_flow_acc:
        calc_flow_acc(plot_config, show_plot=True)
    elif args.spect_bg:
        plot_spectrum_bg(plot_config, show_plot=True)
    elif args.spect_sigma_bg:
        spectrum_sigma_bg(plot_config, show_plot=True)
    elif args.acc_rate_std:
        acceleration_rate_std(plot_config, show_plot=True)
    elif args.calc_dee:
        diffusion_coefficient(plot_config, show_plot=True)
    elif args.calc_acc_rate:
        calc_acc_rate(plot_config, show_plot=True)
    elif args.plot_traj:
        plot_trajectory(plot_config, args.show_plot)
    elif args.plot_traj_j:
        plot_trajectory_j(plot_config, args.show_plot)
    elif args.rates_tracer:
        calc_rates_tracer(plot_config, args.show_plot)
    elif args.plot_rates_tracer:
        plot_rates_tracer(plot_config, args.show_plot)
    elif args.rates_sigma_bg:
        rates_sigma_bg(plot_config, args.show_plot)
    elif args.plot_dee_tracer:
        plot_dee_tracer(plot_config, args.show_plot)
    elif args.get_traj:
        get_trajectory(plot_config, args.show_plot)
    elif args.traj_band:
        plot_trajectory_band(plot_config, args.show_plot)
    elif args.dee_tracer:
        calc_dee_tracer(plot_config, args.show_plot)
    elif args.uout_max:
        get_uout_max(plot_config)
    elif args.high_rate:
        number_rate_high_energy(plot_config, args.show_plot)
    elif args.spect_four:
        plot_spectrum_four_runs(plot_config, args.show_plot)
    elif args.solve_sde:
        solve_sde(plot_config, args.show_plot)
    elif args.pindex_model:
        # power_index_model(plot_config, args.show_plot)
        # power_index_rates(plot_config, args.show_plot)
        pindex_bg_sigma(plot_config, args.show_plot)
        # pindex_sde(plot_config, args.show_plot)
    elif args.pindex_scaling:
        power_law_scaling(plot_config, args.show_plot)
    elif args.nacc_nesc:
        plot_nacc_nesc(plot_config, args.show_plot)
    elif args.ve_kappa_dist:
        vexb_kappa_dist(plot_config, args.show_plot)
    elif args.test_calc_rates:
        test_calc_rates(plot_config, args.show_plot)
    elif args.plot_absj:
        plot_absj(plot_config, args.show_plot)


def process_input(plot_config, args, tframe):
    """process one time frame"""
    plot_config["tframe"] = tframe
    if args.calc_vexb_kappa:
        calc_vexb_kappa(plot_config)
    elif args.plot_vexb_kappa:
        plot_vexb_kappa(plot_config, show_plot=False)


def analysis_multi_frames(plot_config, args):
    """Analysis for multiple time frames
    """
    tframes = range(plot_config["tstart"], plot_config["tend"] + 1)
    if args.time_loop:
        for tframe in tframes:
            print("Time frame: %d" % tframe)
            plot_config["tframe"] = tframe
            if args.calc_vexb_kappa:
                calc_vexb_kappa(plot_config)
            elif args.plot_vexb_kappa:
                plot_vexb_kappa(plot_config, show_plot=False)
            elif args.plot_absj:
                plot_absj(plot_config, show_plot=False)
    else:
        ncores = multiprocessing.cpu_count()
        ncores = 18
        Parallel(n_jobs=ncores)(delayed(process_input)(plot_config, args, tframe)
                                for tframe in tframes)


def main():
    """business logic for when running this module as the primary one!"""
    args = get_cmd_args()
    plot_config = {}
    plot_config["pic_run"] = args.pic_run
    plot_config["pic_run_dir"] = args.pic_run_dir
    plot_config["tframe"] = args.tframe
    plot_config["tstart"] = args.tstart
    plot_config["tend"] = args.tend
    plot_config["species"] = args.species
    plot_config["plot_type"] = args.plot_type
    plot_config["vkappa_threshold"] = args.vkappa_threshold
    plot_config["sigma_type"] = args.sigma_type
    plot_config["bg"] = args.bg
    if args.multi_runs:
        analysis_multi_runs(plot_config, args)
    else:
        if args.multi_frames:
            analysis_multi_frames(plot_config, args)
        else:
            analysis_single_frames(plot_config, args)


if __name__ == "__main__":
    main()
