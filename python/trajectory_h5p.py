"""
#!/usr/bin/env python3
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


def read_particle_data(iptl, particle_tags, pic_info, fh):
    """Read particle data for a HDF5 file

    Args:
        iptl: particle index
        particles_tags: all the particle tags
        pic_info: PIC simulation information
        fh: HDF5 file handler
    """
    group = fh[particle_tags[iptl]]
    dset = group['dX']
    sz, = dset.shape
    ptl = {}
    for dset in group:
        dset = str(dset)
        ptl[str(dset)] = read_var(group, dset, sz)

    gama = np.sqrt(ptl['Ux']**2 + ptl['Uy']**2 + ptl['Uz']**2 + 1)
    dtwci = pic_info.dtwci
    dtwpe = pic_info.dtwpe
    dt = pic_info.dt_fields * dtwpe / dtwci
    tfields = np.arange(sz) * dt
    smime = math.sqrt(pic_info.mime)
    ptl['t'] = tfields
    ptl['gamma'] = gama

    # Some data points may be zeros
    nt = np.count_nonzero(ptl['q'])
    index = np.nonzero(ptl['q'])
    for dset in ptl:
        ptl[dset] = ptl[dset][index]
    ptl['dX'] /= smime
    ptl['dY'] /= smime
    ptl['dZ'] /= smime

    return (ptl, sz)


def transfer_to_h5part(plot_config):
    """Transfer current HDF5 file to H5Part format

    All particles at the same time step are stored in the same time step

    """
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    traj_dir = plot_config["traj_dir"]
    if plot_config["species"] == 'e':
        species = 'electron'
    else:
        species = 'H'
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    fpath_traj = pic_run_dir + traj_dir + '/'
    fname = fpath_traj + species + 's.h5p'
    fh_in = h5py.File(fname, 'r')
    particle_tags = list(fh_in.keys())
    nptl = len(particle_tags)
    ptl, ntf = read_particle_data(0, particle_tags, pic_info, fh_in)
    Ux = np.zeros(ntf * nptl, dtype=ptl['Ux'].dtype)
    Uy = np.zeros(ntf * nptl, dtype=ptl['Uy'].dtype)
    Uz = np.zeros(ntf * nptl, dtype=ptl['Uz'].dtype)
    dX = np.zeros(ntf * nptl, dtype=ptl['dX'].dtype)
    dY = np.zeros(ntf * nptl, dtype=ptl['dY'].dtype)
    dZ = np.zeros(ntf * nptl, dtype=ptl['dZ'].dtype)
    i = np.zeros(ntf * nptl, dtype=ptl['i'].dtype)
    q = np.zeros(ntf * nptl, dtype=ptl['q'].dtype)
    gamma = np.zeros(ntf * nptl, dtype=ptl['gamma'].dtype)
    t = np.zeros(ntf * nptl, dtype=ptl['t'].dtype)
    if 'Bx' in ptl:
        Bx = np.zeros(ntf * nptl, dtype=ptl['Bx'].dtype)
        By = np.zeros(ntf * nptl, dtype=ptl['By'].dtype)
        Bz = np.zeros(ntf * nptl, dtype=ptl['Bz'].dtype)
        Ex = np.zeros(ntf * nptl, dtype=ptl['Ex'].dtype)
        Ey = np.zeros(ntf * nptl, dtype=ptl['Ey'].dtype)
        Ez = np.zeros(ntf * nptl, dtype=ptl['Ez'].dtype)
    if 'Vx' in ptl:
        Vx = np.zeros(ntf * nptl, dtype=ptl['Vx'].dtype)
        Vy = np.zeros(ntf * nptl, dtype=ptl['Vy'].dtype)
        Vz = np.zeros(ntf * nptl, dtype=ptl['Vz'].dtype)
        ne = np.zeros(ntf * nptl, dtype=ptl['ne'].dtype)
        ni = np.zeros(ntf * nptl, dtype=ptl['ni'].dtype)

    # Read all particle data
    for iptl in range(nptl):
        ptl, ntf = read_particle_data(iptl, particle_tags, pic_info, fh_in)
        Ux[iptl::nptl] = ptl['Ux']
        Uy[iptl::nptl] = ptl['Uy']
        Uz[iptl::nptl] = ptl['Uz']
        dX[iptl::nptl] = ptl['dX']
        dY[iptl::nptl] = ptl['dY']
        dZ[iptl::nptl] = ptl['dZ']
        i[iptl::nptl] = ptl['i']
        q[iptl::nptl] = ptl['q']
        gamma[iptl::nptl] = ptl['gamma']
        t[iptl::nptl] = ptl['t']
        if 'Bx' in ptl:
            Bx[iptl::nptl] = ptl['Bx']
            By[iptl::nptl] = ptl['By']
            Bz[iptl::nptl] = ptl['Bz']
            Ex[iptl::nptl] = ptl['Ex']
            Ey[iptl::nptl] = ptl['Ey']
            Ez[iptl::nptl] = ptl['Ez']
        if 'Vx' in ptl:
            Vx[iptl::nptl] = ptl['Vx']
            Vy[iptl::nptl] = ptl['Vy']
            Vz[iptl::nptl] = ptl['Vz']
            ne[iptl::nptl] = ptl['ne']
            ni[iptl::nptl] = ptl['ni']

    fh_in.close()

    fname_out = fpath_traj + species + 's.h5part'
    fh_out = h5py.File(fname_out, 'w')

    for tindex in range(0, ntf):
        grp = fh_out.create_group('Step#' + str(tindex))
        es, ee = tindex * nptl, (tindex + 1) * nptl
        grp.create_dataset('Ux', (nptl, ), data=Ux[es:ee])
        grp.create_dataset('Uy', (nptl, ), data=Uy[es:ee])
        grp.create_dataset('Uz', (nptl, ), data=Uz[es:ee])
        grp.create_dataset('dX', (nptl, ), data=dX[es:ee])
        grp.create_dataset('dY', (nptl, ), data=dY[es:ee])
        grp.create_dataset('dZ', (nptl, ), data=dZ[es:ee])
        grp.create_dataset('i', (nptl, ), data=i[es:ee])
        grp.create_dataset('q', (nptl, ), data=q[es:ee])
        grp.create_dataset('gamma', (nptl, ), data=gamma[es:ee])
        grp.create_dataset('t', (nptl, ), data=t[es:ee])
        if 'Bx' in ptl:
            grp.create_dataset('Bx', (nptl, ), data=Bx[es:ee])
            grp.create_dataset('By', (nptl, ), data=By[es:ee])
            grp.create_dataset('Bz', (nptl, ), data=Bz[es:ee])
            grp.create_dataset('Ex', (nptl, ), data=Ex[es:ee])
            grp.create_dataset('Ey', (nptl, ), data=Ey[es:ee])
            grp.create_dataset('Ez', (nptl, ), data=Ez[es:ee])
        if 'Vx' in ptl:
            grp.create_dataset('Vx', (nptl, ), data=Vx[es:ee])
            grp.create_dataset('Vy', (nptl, ), data=Vy[es:ee])
            grp.create_dataset('Vz', (nptl, ), data=Vz[es:ee])
            grp.create_dataset('ne', (nptl, ), data=ne[es:ee])
            grp.create_dataset('ni', (nptl, ), data=ni[es:ee])

    fh_out.close()


def particle_trajectory(plot_config):
    """Plotting particle trajectory
    """
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    traj_dir = plot_config["traj_dir"]
    if plot_config["species"] == 'e':
        species = 'electron'
    else:
        species = 'H'
    fname = pic_run_dir + traj_dir + '/' + species + 's.h5p'
    pindex = 107
    with h5py.File(fname, 'r') as fh:
        particle_tags = list(fh.keys())
        nptl = len(particle_tags)
        group = fh[particle_tags[pindex]]
        dset = group['dX']
        ntf, = dset.shape
        ptl = {}
        for dset in group:
            dset = str(dset)
            ptl[str(dset)] = read_var(group, dset, ntf)
    told = np.arange(ntf)
    tnew = np.arange(0, ntf - 0.5, 0.5)

    # for key in ptl:
    #     f = interp1d(told, ptl[key], kind='quadratic')
    #     ptl[key] = f(tnew)

    vx = ptl['Vx']
    vy = ptl['Vy']
    vz = ptl['Vz']
    ux = ptl['Ux']
    uy = ptl['Uy']
    uz = ptl['Uz']
    px = ptl['dX']
    py = ptl['dY']
    pz = ptl['dZ']
    ex = ptl['Ex']
    ey = ptl['Ey']
    ez = ptl['Ez']
    bx = ptl['Bx']
    by = ptl['By']
    bz = ptl['Bz']
    ne = ptl['ne']
    ni = ptl['ni']

    gamma = np.sqrt(1 + ux**2 + uy**2 + uz**2)

    cmap = plt.cm.seismic
    fig = plt.figure(figsize=[7, 6])
    xs, ys = 0.12, 0.12
    w1, h1 = 0.8, 0.75
    color1 = 'k'
    color2 = 'b'
    ax1 = fig.add_axes([xs, ys, w1, h1])
    # ax1.plot(tnew, gamma - 1, linewidth=1, color=color1)
    # ax1.plot(px, gamma - 1, linewidth=1, color=color1)
    # ax1.plot(px, pz, linewidth=1, color=color1)
    ax1.plot(ni, gamma - 1, linewidth=1, color=color1)
    ax1.tick_params(labelsize=16)
    ax1.set_xlabel(r'$x/d_e$', fontsize=24)
    ax1.set_ylabel(r'$\gamma - 1$', fontsize=24)
    plt.show()


def get_cmd_args():
    """Get command line arguments
    """
    default_pic_run = '2D-Lx150-bg0.2-150ppc-16KNL'
    default_pic_run_dir = ('/net/scratch3/xiaocanli/reconnection/Cori_runs/' +
                           default_pic_run + '/')
    parser = argparse.ArgumentParser(description='Analysis for Cori 3D runs')
    parser.add_argument('--pic_run', action="store",
                        default=default_pic_run, help='PIC run name')
    parser.add_argument('--pic_run_dir', action="store",
                        default=default_pic_run_dir, help='PIC run directory')
    parser.add_argument('--traj_dir', action="store",
                        default='trajectory', help='Directory of trajectory')
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
    parser.add_argument('--plot_traj', action="store_true", default=False,
                        help="whether to plot particle trajectory")
    parser.add_argument('--trans_h5part', action="store_true", default=False,
                        help="whether to trajectory data into H5Part")
    return parser.parse_args()


def analysis_single_frames(plot_config, args):
    """Analysis for multiple time frames
    """
    if args.plot_traj:
        particle_trajectory(plot_config)
    if args.trans_h5part:
        transfer_to_h5part(plot_config)


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
    plot_config["traj_dir"] = args.traj_dir
    plot_config["species"] = args.species
    if args.multi_frames:
        analysis_multi_frames(plot_config, args)
    else:
        analysis_single_frames(plot_config, args)


if __name__ == "__main__":
    main()
