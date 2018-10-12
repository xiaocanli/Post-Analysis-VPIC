"""
#!/usr/bin/env python3
"""
from __future__ import print_function

import argparse
import errno
import itertools
import json
import math
import multiprocessing
import os

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import palettable
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy.interpolate import interp1d

import fitting_funcs
import pic_information
from contour_plots import read_2d_fields
from joblib import Parallel, delayed
from json_functions import read_data_from_json
from shell_functions import mkdir_p

plt.style.use("seaborn-deep")
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = \
[r"\usepackage{amsmath, bm}",
 r"\DeclareMathAlphabet{\mathsfit}{\encodingdefault}{\sfdefault}{m}{sl}",
 r"\SetMathAlphabet{\mathsfit}{bold}{\encodingdefault}{\sfdefault}{bx}{sl}",
 r"\newcommand{\tensorsym}[1]{\bm{\mathsfit{#1}}}"]
COLORS = palettable.colorbrewer.qualitative.Set1_9.mpl_colors


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def combine_spectrum(plot_config):
    """Combine the spectrum in the whole box

    Here we assume that PIC only splits z into different zones
    """
    pic_run_dir = plot_config["pic_run_dir"]
    spect_dir = plot_config["spect_dir"]
    species = plot_config["species"]
    mpi_size = plot_config["mpi_size"]
    mpi_sizex = plot_config["mpi_sizex"]
    mpi_sizey = plot_config["mpi_sizey"]
    mpi_sizez = plot_config["mpi_sizez"]
    num_fold = plot_config["num_fold"]
    nbins = plot_config["nbins"]
    tindex = plot_config["tframe"] * plot_config["tinterval"]
    fdir = pic_run_dir + spect_dir + '/0/T.' + str(tindex) + '/'
    fname = fdir + 'spectrum-' + species + 'hydro.' + str(tindex) + '.0'
    fdata = np.fromfile(fname, dtype=np.float32)
    nzones = fdata.shape[0] // nbins
    fspect = np.zeros((nzones * mpi_sizez, mpi_sizey, mpi_sizex, nbins))
    nxy = mpi_sizex * mpi_sizey
    for mpi_rank in range(mpi_size):
        print("MPI rank: %d" % mpi_rank)
        iz = mpi_rank // nxy
        iy = (mpi_rank % nxy) // mpi_sizex
        ix = mpi_rank % mpi_sizex
        findex = mpi_rank // num_fold
        fdir = (pic_run_dir + spect_dir + '/' + str(findex) +
                '/T.' + str(tindex) + '/')
        fname = (fdir + 'spectrum-' + species + 'hydro.' +
                 str(tindex) + '.' + str(mpi_rank))
        fdata = np.fromfile(fname, dtype=np.float32)
        # fdata = fdata.reshape((nzones, nbins))
        # fspect[iz*nzones:(iz+1)*nzones, iy, ix, :] = fdata

    fdir = ('/net/scratch3/xiaocanli/reconnection/NERSC_ADAM/' +
            plot_config['pic_run'] + '/spectrum/')
    mkdir_p(fdir)
    fname = fdir + species + 'spectrum.gda'
    fspect.tofile(fname)


def fit_thermal_core(ene, f):
    """Fit to get the thermal core of the particle distribution.

    Fit the thermal core of the particle distribution.
    The thermal core is fitted as a Maxwellian distribution.

    Args:
        ene: the energy bins array.
        f: the particle flux distribution.

    Returns:
        fthermal: thermal part of the particle distribution.
    """
    print('Fitting to get the thermal core of the particle distribution')
    estart = 0
    ng = 3
    kernel = np.ones(ng) / float(ng)
    fnew = np.convolve(f, kernel, 'same')
    nshift = 10  # grids shift for fitting thermal core.
    eend = np.argmax(fnew) + nshift
    popt, pcov = curve_fit(fitting_funcs.func_maxwellian,
                           ene[estart:eend], f[estart:eend])
    fthermal = fitting_funcs.func_maxwellian(ene, popt[0], popt[1])
    print('Energy with maximum flux: %f' % ene[eend - 10])
    print('Energy with maximum flux in fitted thermal core: %f' % (0.5 / popt[1]))
    return fthermal


def accumulated_particle_info(ene, f):
    """
    Get the accumulated particle number and total energy from
    the distribution function.

    Args:
        ene: the energy bins array.
        f: the energy distribution array.
    Returns:
        nacc_ene: the accumulated particle number with energy.
        eacc_ene: the accumulated particle total energy with energy.
    """
    nbins, = f.shape
    dlogE = (math.log10(max(ene)) - math.log10(min(ene))) / nbins
    nacc_ene = np.zeros(nbins)
    eacc_ene = np.zeros(nbins)
    nacc_ene[0] = f[0] * ene[0]
    eacc_ene[0] = 0.5 * f[0] * ene[0]**2
    for i in range(1, nbins):
        nacc_ene[i] = f[i] * (ene[i] + ene[i - 1]) * 0.5 + nacc_ene[i - 1]
        eacc_ene[i] = 0.5 * f[i] * (ene[i] - ene[i - 1]) * (
            ene[i] + ene[i - 1])
        eacc_ene[i] += eacc_ene[i - 1]
    nacc_ene *= dlogE
    eacc_ene *= dlogE
    return (nacc_ene, eacc_ene)


def plot_spectrum(plot_config):
    """Plot local spectrum
    """
    pic_run_dir = plot_config["pic_run_dir"]
    spect_dir = plot_config["spect_dir"]
    species = plot_config["species"]
    mpi_size = plot_config["mpi_size"]
    mpi_sizex = plot_config["mpi_sizex"]
    mpi_sizey = plot_config["mpi_sizey"]
    mpi_sizez = plot_config["mpi_sizez"]
    num_fold = plot_config["num_fold"]
    emin = plot_config["emin"]
    emax = plot_config["emax"]
    nbins = plot_config["nbins"]
    tframe = plot_config["tframe"]
    delog = (math.log10(emax) - math.log10(emin)) / (nbins - 1)
    emin = 10**(math.log10(emin) - delog) # adjust
    ebins = np.logspace(math.log10(emin), math.log10(emax), nbins + 1)
    ebins_mid = (ebins[1:] + ebins[:-1]) * 0.5
    debins = np.diff(ebins)
    tindex = tframe * plot_config["tinterval"]
    fdir = ('/net/scratch3/xiaocanli/reconnection/NERSC_ADAM/' +
            'LOCAL-SPECTRA-NEW/')
    fname = fdir + 'spectrum_' + species + '_' + str(tindex) + '.gda'
    fspect = np.fromfile(fname, dtype=np.float32)
    fspect = fspect.reshape((-1, mpi_sizey, mpi_sizex, nbins))
    fspect /= debins
    img_dir = fdir + 'img/'
    mkdir_p(img_dir)
    if tframe < 13:
        fig = plt.figure(figsize=[7, 5])
        rect = [0.12, 0.12, 0.85, 0.85]
        ax1 = fig.add_axes(rect)
        ax1.tick_params(bottom=True, top=True, left=True, right=True)
        ax1.tick_params(axis='x', which='minor', direction='in', top='on')
        ax1.tick_params(axis='x', which='major', direction='in')
        ax1.tick_params(axis='y', which='minor', direction='in', left='on')
        ax1.tick_params(axis='y', which='major', direction='in')
        ftot = fspect[8, mpi_sizey//2, mpi_sizex//2, :]
        fth = fit_thermal_core(ebins_mid, ftot)
        fnth = ftot - fth
        ntot, etot = accumulated_particle_info(ebins, ftot)
        nth, eth = accumulated_particle_info(ebins, fth)
        nnth, enth = accumulated_particle_info(ebins, fnth)
        frac_th = "{%0.2f}" % ((nth[-1] / ntot[-1]) * 100)
        frac_nth = "{%0.2f}" % ((nnth[-1] / ntot[-1]) * 100)
        ax1.loglog(ebins_mid, ftot, linewidth=2, label='total')
        label_th = r'thermal-core $(' + frac_th + '\%)$'
        ax1.loglog(ebins_mid, fth, linewidth=1, label=label_th)
        label_nth = r'rest $(' + frac_nth + '\%)$'
        ax1.loglog(ebins_mid[nbins//4:], fnth[nbins//4:],
                   linewidth=1, label=label_nth)
        ax1.set_xlim([1E-2, 5E0])
        ax1.set_ylim([1E1, 1E7])
        ax1.tick_params(labelsize=12)
        ax1.set_xlabel(r'$\gamma - 1$', fontsize=16)
        ax1.set_ylabel(r'$f(\gamma - 1)$', fontsize=16)
        ax1.legend(loc=3, prop={'size': 16}, ncol=1,
                   shadow=False, fancybox=False, frameon=False)
        fname = img_dir + 'local_spect_' + species + '_' + str(tindex) + '.pdf'
        fig.savefig(fname)
    else:
        fdir = img_dir + 'T.' + str(tindex) + '/'
        mkdir_p(fdir)
        for iz in [7, 8]:
            for iy in range(0, mpi_sizey, 32):
                for ix in [mpi_sizex//2-1, mpi_sizex//2]:
                    fig = plt.figure(figsize=[7, 5])
                    rect = [0.12, 0.12, 0.85, 0.85]
                    ax1 = fig.add_axes(rect)
                    ax1.tick_params(bottom=True, top=True, left=True, right=True)
                    ax1.tick_params(axis='x', which='minor', direction='in', top='on')
                    ax1.tick_params(axis='x', which='major', direction='in')
                    ax1.tick_params(axis='y', which='minor', direction='in', left='on')
                    ax1.tick_params(axis='y', which='major', direction='in')
                    ftot = fspect[iz, iy, ix, :]
                    fth = fit_thermal_core(ebins_mid, ftot)
                    fnth = ftot - fth
                    ntot, etot = accumulated_particle_info(ebins, ftot)
                    nth, eth = accumulated_particle_info(ebins, fth)
                    nnth, enth = accumulated_particle_info(ebins, fnth)
                    frac_th = "{%0.2f}" % ((nth[-1] / ntot[-1]) * 100)
                    frac_nth = "{%0.2f}" % ((nnth[-1] / ntot[-1]) * 100)
                    ax1.loglog(ebins_mid, ftot, linewidth=2, label='total')
                    label_th = r'thermal-core $(' + frac_th + '\%)$'
                    ax1.loglog(ebins_mid, fth, linewidth=1, label=label_th)
                    label_nth = r'rest $(' + frac_nth + '\%)$'
                    ax1.loglog(ebins_mid[nbins//4:], fnth[nbins//4:],
                               linewidth=1, label=label_nth)
                    ax1.set_xlim([1E-2, 5E0])
                    ax1.set_ylim([1E1, 1E7])
                    ax1.tick_params(labelsize=12)
                    ax1.set_xlabel(r'$\gamma - 1$', fontsize=16)
                    ax1.set_ylabel(r'$f(\gamma - 1)$', fontsize=16)
                    ax1.legend(loc=3, prop={'size': 16}, ncol=1,
                               shadow=False, fancybox=False, frameon=False)
                    fname = (fdir + 'local_spect_' + species + '_' +
                             str(tindex) + '_ix' + str(ix) + '_iy' + str(iy) +
                             '_iz' + str(iz) + '.pdf')
                    fig.savefig(fname)
                    plt.close()
    plt.show()


def get_cmd_args():
    """Get command line arguments
    """
    default_pic_run = 'CORI-RUN1'
    default_pic_run_dir = ('/net/scratch3/stanier/' + default_pic_run + '/')
    parser = argparse.ArgumentParser(description='Analysis for local spectrum')
    parser.add_argument('--pic_run', action="store",
                        default=default_pic_run, help='PIC run name')
    parser.add_argument('--pic_run_dir', action="store",
                        default=default_pic_run_dir, help='PIC run directory')
    parser.add_argument('--spect_dir', action="store",
                        default='LOCAL-SPECTRA-NEW/hydro',
                        help='Directory of the local spectra')
    parser.add_argument('--species', action="store",
                        default="e", help='Particle species')
    parser.add_argument('--mpi_size', action="store", default='131072',
                        type=int, help='MPI size for the PIC simulation')
    parser.add_argument('--mpi_sizex', action="store", default='256',
                        type=int, help='MPI size for PIC along x')
    parser.add_argument('--mpi_sizey', action="store", default='256',
                        type=int, help='MPI size for PIC along y')
    parser.add_argument('--mpi_sizez', action="store", default='2',
                        type=int, help='MPI size for PIC along z')
    parser.add_argument('--num_fold', action="store", default='32',
                        type=int, help='# of files in each sub-directory')
    parser.add_argument('--tinterval', action="store", default='2732', type=int,
                        help='Time interval for the spectrum dump')
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
    parser.add_argument('--nbins', action="store", default='600', type=int,
                        help='Number of energy bins')
    parser.add_argument('--emin', action="store", default='1E-3', type=float,
                        help='Minimum energy')
    parser.add_argument('--emax', action="store", default='1E3', type=float,
                        help='Maximum energy')
    parser.add_argument('--combine_spectrum', action="store_true", default=False,
                        help="whether to combine the spectrum")
    parser.add_argument('--plot_spectrum', action="store_true", default=False,
                        help="whether to plot local spectrum")
    return parser.parse_args()


def analysis_single_frames(plot_config, args):
    """Analysis for multiple time frames
    """
    if args.combine_spectrum:
        combine_spectrum(plot_config)
    if args.plot_spectrum:
        plot_spectrum(plot_config)


def process_input(plot_config, args, tframe):
    """process one time frame"""
    plot_config["tframe"] = tframe


def analysis_multi_frames(plot_config, args):
    """Analysis for multiple time frames
    """
    pass


def main():
    """business logic for when running this module as the primary one!"""
    args = get_cmd_args()
    plot_config = {}
    plot_config["pic_run"] = args.pic_run
    plot_config["pic_run_dir"] = args.pic_run_dir
    plot_config["spect_dir"] = args.spect_dir
    plot_config["species"] = args.species
    plot_config["mpi_size"] = args.mpi_size
    plot_config["mpi_sizex"] = args.mpi_sizex
    plot_config["mpi_sizey"] = args.mpi_sizey
    plot_config["mpi_sizez"] = args.mpi_sizez
    plot_config["num_fold"] = args.num_fold
    plot_config["tinterval"] = args.tinterval
    plot_config["tframe"] = args.tframe
    plot_config["tstart"] = args.tstart
    plot_config["tend"] = args.tend
    plot_config["nbins"] = args.nbins
    plot_config["emin"] = args.emin
    plot_config["emax"] = args.emax
    if args.multi_frames:
        analysis_multi_frames(plot_config, args)
    else:
        analysis_single_frames(plot_config, args)


if __name__ == "__main__":
    main()
