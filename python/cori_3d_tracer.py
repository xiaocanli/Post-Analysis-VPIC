#!/usr/bin/env python3
"""
Particle tracer for the Cori runs
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
import pandas as pd
from evtk.hl import pointsToVTK
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, SymLogNorm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy import signal
from scipy.optimize import curve_fit

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
    dt_tracer = pic_info.tracer_interval * dtwpe
    tfields = np.arange(sz) * dt_tracer
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

    All particles at the same time step are stored in the same time step,
    so it can be loaded into ParaView directly.

    Args:
        plot_config: plot configuration
    """
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    fname = "../data/trajectory/" + pic_run + "/" + plot_config["traj_file"]
    fh = h5py.File(fname, 'r')
    particle_tags = list(fh.keys())
    nptl = len(particle_tags)
    print("Total number of particles: %d" % nptl)
    ptl, ntf = read_particle_data(0, particle_tags, pic_info, fh)
    print("Number of time steps: %d" % ntf)
    ptl_dist = {}
    for key in ptl:
        ptl_dist[key] = np.zeros(ntf * nptl, dtype=ptl[key].dtype)
    for iptl in range(nptl):
        print("Particle ID: %d" % iptl)
        ptl, ntf = read_particle_data(iptl, particle_tags, pic_info, fh)
        for key in ptl:
            ptl_dist[key][iptl::nptl] = ptl[key]
    fh.close()

    fname_out = fname.replace('.h5p', '.h5part')
    with h5py.File(fname_out, 'w') as fh_out:
        for tindex in range(0, ntf):
            print("Time frame: %d" % tindex)
            grp = fh_out.create_group('Step#' + str(tindex))
            index = range(tindex * nptl, (tindex + 1) * nptl)
            for key in ptl:
                grp.create_dataset(key, (nptl, ), data=ptl_dist[key][index])


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


def plot_trajectory(plot_config, show_plot=True):
    """Plot particle trajectory
    """
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    pindex = plot_config['iptl']
    species = plot_config['species']
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    qm = -1 if species == 'e' else 1.0/pic_info.mime
    fname = "../data/trajectory/" + pic_run + "/" + plot_config["traj_file"]
    fh = h5py.File(fname, 'r')
    particle_tags = list(fh.keys())
    nptl = len(particle_tags)
    ptl, sz = read_particle_data(pindex, particle_tags, pic_info, fh)
    smime = math.sqrt(pic_info.mime)
    lx_de = pic_info.lx_di * smime
    ly_de = pic_info.ly_di * smime
    lz_de = pic_info.lz_di * smime
    xpos = adjust_pos(ptl['dX'], pic_info.lx_di)
    ypos = adjust_pos(ptl['dY'], pic_info.ly_di)
    zpos = adjust_pos(ptl['dZ'], pic_info.lz_di)
    kene = ptl["gamma"] - 1
    absB = np.sqrt(ptl['Bx']**2 + ptl['By']**2 + ptl['Bz']**2)
    absU = np.sqrt(ptl['Ux']**2 + ptl['Uy']**2 + ptl['Uz']**2)
    mu = (ptl['Ux'] * ptl['Bx'] + ptl['Uy'] * ptl['By'] +
          ptl['Uz'] * ptl['Bz']) / (absB * absU)
    pitch_angle = np.arccos(mu) * 180 / math.pi
    dt_tracer = pic_info.tracer_interval * pic_info.dtwpe
    ex_ideal = ptl['By'] * ptl['Vz'] - ptl['Bz'] * ptl['Vy']
    ey_ideal = ptl['Bz'] * ptl['Vx'] - ptl['Bx'] * ptl['Vz']
    ez_ideal = ptl['Bx'] * ptl['Vy'] - ptl['By'] * ptl['Vx']
    vdotE = qm * (ptl['Ux'] * ptl['Ex'] + ptl['Uy'] * ptl['Ey'] +
                  ptl['Uz'] * ptl['Ez']) / ptl['gamma']
    vdotE_ideal = qm * (ptl['Ux'] * ex_ideal + ptl['Uy'] * ey_ideal +
                        ptl['Uz'] * ez_ideal) / ptl['gamma']
    vdotE_acc = np.cumsum(vdotE) * dt_tracer
    vdotE_ideal_acc = np.cumsum(vdotE_ideal) * dt_tracer

    fig = plt.figure(figsize=[12, 12])
    rect0 = [0.07, 0.48, 0.55, 0.5]
    hgap, vgap = 0.07, 0.01
    ax = fig.add_axes(rect0, projection='3d')
    points = np.array([xpos, ypos, zpos]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(kene.min(), kene.max())
    lc = Line3DCollection(segments, cmap='jet', norm=norm)
    # Set the values used for colormapping
    lc.set_array(kene)
    lc.set_linewidth(2)
    ax.add_collection3d(lc)
    ax.set_xlim(xpos.min(), xpos.max())
    ax.set_ylim(ypos.min(), ypos.max())
    ax.set_zlim(zpos.min(), zpos.max())
    ax.set_xlabel('$x/d_i$', fontsize=16)
    ax.set_ylabel('$y/d_i$', fontsize=16)
    ax.set_zlabel('$z/d_i$', fontsize=16)
    ax.tick_params(labelsize=12)

    # t-kene
    rect = np.copy(rect0)
    rect[0] += hgap + rect[2]
    rect[2] = 0.27
    rect[3] = 0.12
    rect[1] = 0.85
    ax = fig.add_axes(rect)
    points = np.array([ptl["t"], kene]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(kene.min(), kene.max())
    lc = LineCollection(segments, cmap='jet', norm=norm)
    # Set the values used for colormapping
    lc.set_array(kene)
    lc.set_linewidth(2)
    ax.add_collection(lc)
    # ax.plot(ptl["t"], vdotE_acc)
    # ax.plot(ptl["t"], vdotE_ideal_acc)
    ax.set_xlim([ptl["t"].min(), ptl["t"].max()])
    ax.set_ylim([kene.min(), kene.max()])
    ax.set_ylabel('$\gamma - 1$', fontsize=16)
    ax.tick_params(axis='x', labelbottom='off')
    ax.tick_params(labelsize=12)

    # t-Ux, Uy, Uz
    rect[1] -= rect[3] + vgap
    ax = fig.add_axes(rect)
    p1, = ax.plot(ptl["t"], ptl["Ux"], linewidth=2)
    p2, = ax.plot(ptl["t"], ptl["Uy"], linewidth=2)
    p3, = ax.plot(ptl["t"], ptl["Uz"], linewidth=2)
    ax.set_xlim([ptl["t"].min(), ptl["t"].max()])
    ax.set_ylabel('$U_x, U_y, U_z$', fontsize=16)
    ax.tick_params(axis='x', labelbottom='off')
    ax.tick_params(labelsize=12)
    ax.text(1.02, 0.6, '$x$', color=p1.get_color(), fontsize=16,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)
    ax.text(1.02, 0.5, '$y$', color=p2.get_color(), fontsize=16,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)
    ax.text(1.02, 0.4, '$z$', color=p3.get_color(), fontsize=16,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)

    # t-ni
    rect[1] -= rect[3] + vgap
    ax = fig.add_axes(rect)
    ax.plot(ptl["t"], ptl["ni"], linewidth=2)
    ax.set_xlim([ptl["t"].min(), ptl["t"].max()])
    ax.set_ylabel('$n_i$', fontsize=16)
    ax.tick_params(axis='x', labelbottom='off')
    ax.tick_params(labelsize=12)

    # t-pitch_angle
    rect[1] -= rect[3] + vgap
    ax = fig.add_axes(rect)
    ax.plot(ptl["t"], pitch_angle, linewidth=2)
    ax.set_xlim([ptl["t"].min(), ptl["t"].max()])
    ax.plot([ptl["t"].min(), ptl["t"].max()], [90, 90],
            linestyle='--', linewidth=1, color='k')
    ax.set_ylabel('pitch angle', fontsize=16)
    ax.tick_params(axis='x', labelbottom='off')
    ax.tick_params(labelsize=12)

    # t-Bx, By, Bz
    rect[1] -= rect[3] + vgap
    ax = fig.add_axes(rect)
    p1, = ax.plot(ptl["t"], ptl["Bx"], linewidth=2)
    p2, = ax.plot(ptl["t"], ptl["By"], linewidth=2)
    p3, = ax.plot(ptl["t"], ptl["Bz"], linewidth=2)
    ax.set_xlim([ptl["t"].min(), ptl["t"].max()])
    ax.set_ylabel('$B_x, B_y, B_z$', fontsize=16)
    ax.tick_params(axis='x', labelbottom='off')
    ax.tick_params(labelsize=12)
    ax.text(1.02, 0.6, '$x$', color=p1.get_color(), fontsize=16,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)
    ax.text(1.02, 0.5, '$y$', color=p2.get_color(), fontsize=16,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)
    ax.text(1.02, 0.4, '$z$', color=p3.get_color(), fontsize=16,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)

    # t-Ex, Ey, Ez
    rect[1] -= rect[3] + vgap
    ax = fig.add_axes(rect)
    p1, = ax.plot(ptl["t"], ptl["Ex"], linewidth=2)
    p2, = ax.plot(ptl["t"], ptl["Ey"], linewidth=2)
    p3, = ax.plot(ptl["t"], ptl["Ez"], linewidth=2)
    ax.set_xlim([ptl["t"].min(), ptl["t"].max()])
    ax.set_ylabel('$E_x, E_y, E_z$', fontsize=16)
    ax.tick_params(axis='x', labelbottom='off')
    ax.tick_params(labelsize=12)
    ax.text(1.02, 0.6, '$x$', color=p1.get_color(), fontsize=16,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)
    ax.text(1.02, 0.5, '$y$', color=p2.get_color(), fontsize=16,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)
    ax.text(1.02, 0.4, '$z$', color=p3.get_color(), fontsize=16,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)

    # t-Vx, Vy, Vz
    rect[1] -= rect[3] + vgap
    ax = fig.add_axes(rect)
    p1, = ax.plot(ptl["t"], ptl["Vx"], linewidth=2)
    p2, = ax.plot(ptl["t"], ptl["Vy"], linewidth=2)
    p3, = ax.plot(ptl["t"], ptl["Vz"], linewidth=2)
    ax.set_xlim([ptl["t"].min(), ptl["t"].max()])
    ax.set_xlabel('$t\omega_{pe}$', fontsize=16)
    ax.set_ylabel('$V_x, V_y, V_z$', fontsize=16)
    ax.tick_params(labelsize=12)
    ax.text(1.02, 0.6, '$x$', color=p1.get_color(), fontsize=16,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)
    ax.text(1.02, 0.5, '$y$', color=p2.get_color(), fontsize=16,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)
    ax.text(1.02, 0.4, '$z$', color=p3.get_color(), fontsize=16,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)

    # xy
    rect = np.copy(rect0)
    rect[0] = 0.09
    rect[2] = 0.22
    rect[3] = 0.18
    rect[1] -= vgap + rect[3]
    rect1 = np.copy(rect)
    hgap, vgap = 0.06, 0.01
    ax = fig.add_axes(rect)
    points = np.array([xpos, ypos]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(kene.min(), kene.max())
    lc = LineCollection(segments, cmap='jet', norm=norm)
    lc.set_array(kene)
    lc.set_linewidth(2)
    ax.add_collection(lc)
    ax.set_xlim([xpos.min(), xpos.max()])
    ax.set_ylim([ypos.min(), ypos.max()])
    ax.set_xlabel('$x/d_i$', fontsize=16)
    ax.set_ylabel('$y/d_i$', fontsize=16)
    ax.tick_params(labelsize=12)

    # xz
    rect[1] -= rect[2] + vgap
    ax = fig.add_axes(rect)
    points = np.array([xpos, zpos]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(kene.min(), kene.max())
    lc = LineCollection(segments, cmap='jet', norm=norm)
    lc.set_array(kene)
    lc.set_linewidth(2)
    ax.add_collection(lc)
    ax.set_xlim([xpos.min(), xpos.max()])
    ax.set_ylim([zpos.min(), zpos.max()])
    ax.set_xlabel('$x/d_i$', fontsize=16)
    ax.set_ylabel('$z/d_i$', fontsize=16)
    ax.tick_params(labelsize=12)

    # zy
    rect[1] += rect[2] + vgap
    rect[0] += rect[2] + hgap
    ax = fig.add_axes(rect)
    points = np.array([zpos, ypos]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(kene.min(), kene.max())
    lc = LineCollection(segments, cmap='jet', norm=norm)
    lc.set_array(kene)
    lc.set_linewidth(2)
    ax.add_collection(lc)
    ax.set_xlim([zpos.min(), zpos.max()])
    ax.set_ylim([ypos.min(), ypos.max()])
    ax.set_xlabel('$z/d_i$', fontsize=16)
    ax.tick_params(axis='y', labelleft='off')
    ax.tick_params(labelsize=12)

    # x-gamma
    rect[1] -= rect[2] + vgap
    ax = fig.add_axes(rect)
    points = np.array([xpos, kene]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(-0.1, 0.1)
    lc = LineCollection(segments, cmap='seismic', norm=norm)
    lc.set_array(ptl["Vx"])
    lc.set_linewidth(2)
    ax.add_collection(lc)
    ax.set_xlim([xpos.min(), xpos.max()])
    ax.set_ylim([kene.min(), kene.max()])
    ax.set_xlabel('$x/d_i$', fontsize=16)
    ax.set_ylabel('$\gamma-1$', fontsize=16)
    ax.tick_params(labelsize=12)
    ax.text(0.85, 0.05, '$V_x$', color='k', fontsize=16,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)

    fdir = '../img/cori_3d/tracer/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + species + 'tracer_' + str(pindex) + '.jpg'
    fig.savefig(fname, dpi=300)

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_trajectory_simple(plot_config, show_plot=True):
    """Plot particle trajectory in a simple way
    """
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    pindex = plot_config['iptl']
    species = plot_config['species']
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    fname = "../data/trajectory/" + pic_run + "/" + plot_config["traj_file"]
    fh = h5py.File(fname, 'r')
    particle_tags = list(fh.keys())
    nptl = len(particle_tags)
    ptl, sz = read_particle_data(pindex, particle_tags, pic_info, fh)
    smime = math.sqrt(pic_info.mime)
    lx_de = pic_info.lx_di * smime
    ly_de = pic_info.ly_di * smime
    lz_de = pic_info.lz_di * smime
    xpos = adjust_pos(ptl['dX'], pic_info.lx_di)
    ypos = adjust_pos(ptl['dY'], pic_info.ly_di)
    zpos = adjust_pos(ptl['dZ'], pic_info.lz_di)
    kene = ptl["gamma"] - 1

    fig = plt.figure(figsize=[12, 12])
    rect0 = [0.07, 0.48, 0.55, 0.5]
    hgap, vgap = 0.07, 0.01
    ax = fig.add_axes(rect0, projection='3d')
    points = np.array([xpos, ypos, zpos]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(kene.min(), kene.max())
    lc = Line3DCollection(segments, cmap='jet', norm=norm)
    # Set the values used for colormapping
    lc.set_array(kene)
    lc.set_linewidth(2)
    ax.add_collection3d(lc)
    ax.set_xlim(xpos.min(), xpos.max())
    ax.set_ylim(ypos.min(), ypos.max())
    ax.set_zlim(zpos.min(), zpos.max())
    ax.set_xlabel('$x/d_i$', fontsize=16)
    ax.set_ylabel('$y/d_i$', fontsize=16)
    ax.set_zlabel('$z/d_i$', fontsize=16)
    ax.tick_params(labelsize=12)

    # t-kene
    rect = np.copy(rect0)
    rect[0] += hgap + rect[2]
    rect[2] = 0.28
    rect[1] += 0.05
    rect[3] -= 0.05
    ax = fig.add_axes(rect)
    points = np.array([ptl["t"], kene]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(kene.min(), kene.max())
    lc = LineCollection(segments, cmap='jet', norm=norm)
    # Set the values used for colormapping
    lc.set_array(kene)
    lc.set_linewidth(2)
    ax.add_collection(lc)
    ax.set_xlim([ptl["t"].min(), ptl["t"].max()])
    ax.set_ylim([kene.min(), kene.max()])
    ax.set_xlabel('$t\omega_{pe}$', fontsize=16)
    ax.set_ylabel('$\gamma - 1$', fontsize=16)
    ax.tick_params(labelsize=12)

    # xz
    rect = np.copy(rect0)
    rect[2] = 0.42
    rect[3] = 0.41
    rect[1] -= rect[3] + vgap
    ax = fig.add_axes(rect)
    points = np.array([xpos, zpos]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(kene.min(), kene.max())
    lc = LineCollection(segments, cmap='jet', norm=norm)
    lc.set_array(kene)
    lc.set_linewidth(2)
    ax.add_collection(lc)
    ax.set_xlim([xpos.min(), xpos.max()])
    ax.set_ylim([zpos.min(), zpos.max()])
    ax.set_xlabel('$x/d_i$', fontsize=16)
    ax.set_ylabel('$z/d_i$', fontsize=16)
    ax.tick_params(labelsize=12)

    # x-gamma
    rect[0] += rect[2] + hgap
    ax = fig.add_axes(rect)
    points = np.array([xpos, kene]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(-0.1, 0.1)
    lc = LineCollection(segments, cmap='seismic', norm=norm)
    lc.set_array(ptl["Vx"])
    lc.set_linewidth(2)
    ax.add_collection(lc)
    ax.set_xlim([xpos.min(), xpos.max()])
    ax.set_ylim([kene.min(), kene.max()])
    ax.set_xlabel('$x/d_i$', fontsize=16)
    ax.set_ylabel('$\gamma-1$', fontsize=16)
    ax.tick_params(labelsize=12)
    text1 = 'Color-coded by ' + '$V_x$'
    ax.text(0.98, 0.05, text1, color='k', fontsize=24,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='right', verticalalignment='center',
            transform=ax.transAxes)

    fdir = '../img/cori_3d/tracer_200/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + species + 'tracer_' + str(pindex) + '.jpg'
    fig.savefig(fname, dpi=300)

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_trajectory_movie(plot_config, show_plot=True):
    """Plot particle trajectory for trajectory movie
    """
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    pindex = plot_config['iptl']
    species = plot_config['species']
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    fname = "../data/trajectory/" + pic_run + "/" + plot_config["traj_file"]
    fh = h5py.File(fname, 'r')
    particle_tags = list(fh.keys())
    nptl = len(particle_tags)
    ptl, sz = read_particle_data(pindex, particle_tags, pic_info, fh)
    smime = math.sqrt(pic_info.mime)
    lx_de = pic_info.lx_di * smime
    ly_de = pic_info.ly_di * smime
    lz_de = pic_info.lz_di * smime
    xpos = adjust_pos(ptl['dX'], pic_info.lx_di)
    ypos = adjust_pos(ptl['dY'], pic_info.ly_di)
    zpos = adjust_pos(ptl['dZ'], pic_info.lz_di)
    kene = ptl["gamma"] - 1

    fdir = '../img/cori_3d/tracer_200/' + pic_run + '/'
    fdir += 'tracer_' + str(pindex) + '/'
    mkdir_p(fdir)

    for tframe in range(sz):
    # for tframe in range(0, 1):
        print("Time frame: %d" % tframe)
        fig = plt.figure(figsize=[12, 12])
        rect0 = [0.07, 0.48, 0.55, 0.5]
        hgap, vgap = 0.07, 0.01
        ax = fig.add_axes(rect0, projection='3d')
        points = np.array([xpos, ypos, zpos]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        # Create a continuous norm to map from data points to colors
        norm = plt.Normalize(kene.min(), kene.max())
        lc = Line3DCollection(segments, cmap='jet', norm=norm)
        # Set the values used for colormapping
        lc.set_array(kene)
        lc.set_linewidth(2)
        ax.add_collection3d(lc)
        ax.plot([xpos[tframe]], [ypos[tframe]], [zpos[tframe]],
                marker='o', markersize=10, color="k")
        ax.set_xlim(xpos.min(), xpos.max())
        ax.set_ylim(ypos.min(), ypos.max())
        ax.set_zlim(zpos.min(), zpos.max())
        ax.set_xlabel('$x/d_i$', fontsize=16)
        ax.set_ylabel('$y/d_i$', fontsize=16)
        ax.set_zlabel('$z/d_i$', fontsize=16)
        ax.tick_params(labelsize=12)

        # t-kene
        rect = np.copy(rect0)
        rect[0] += hgap + rect[2]
        rect[2] = 0.28
        rect[1] += 0.05
        rect[3] -= 0.05
        ax = fig.add_axes(rect)
        points = np.array([ptl["t"], kene]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        # Create a continuous norm to map from data points to colors
        norm = plt.Normalize(kene.min(), kene.max())
        lc = LineCollection(segments, cmap='jet', norm=norm)
        # Set the values used for colormapping
        lc.set_array(kene)
        lc.set_linewidth(2)
        ax.add_collection(lc)
        ax.plot([ptl["t"][tframe]], [kene[tframe]],
                marker='o', markersize=10, color="k")
        ax.set_xlim([ptl["t"].min(), ptl["t"].max()])
        ax.set_ylim([kene.min(), kene.max()])
        ax.set_xlabel('$t\omega_{pe}$', fontsize=16)
        ax.set_ylabel('$\gamma - 1$', fontsize=16)
        ax.tick_params(labelsize=12)

        # xz
        rect = np.copy(rect0)
        rect[2] = 0.42
        rect[3] = 0.41
        rect[1] -= rect[3] + vgap
        ax = fig.add_axes(rect)
        points = np.array([xpos, zpos]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(kene.min(), kene.max())
        lc = LineCollection(segments, cmap='jet', norm=norm)
        lc.set_array(kene)
        lc.set_linewidth(2)
        ax.add_collection(lc)
        ax.plot([xpos[tframe]], [zpos[tframe]],
                marker='o', markersize=10, color="k")
        ax.set_xlim([xpos.min(), xpos.max()])
        ax.set_ylim([zpos.min(), zpos.max()])
        ax.set_xlabel('$x/d_i$', fontsize=16)
        ax.set_ylabel('$z/d_i$', fontsize=16)
        ax.tick_params(labelsize=12)

        # x-gamma
        rect[0] += rect[2] + hgap
        ax = fig.add_axes(rect)
        points = np.array([xpos, kene]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        norm = plt.Normalize(-0.1, 0.1)
        lc = LineCollection(segments, cmap='seismic', norm=norm)
        lc.set_array(ptl["Vx"])
        lc.set_linewidth(2)
        ax.add_collection(lc)
        ax.plot([xpos[tframe]], [kene[tframe]],
                marker='o', markersize=10, color="k")
        ax.set_xlim([xpos.min(), xpos.max()])
        ax.set_ylim([kene.min(), kene.max()])
        ax.set_xlabel('$x/d_i$', fontsize=16)
        ax.set_ylabel('$\gamma-1$', fontsize=16)
        ax.tick_params(labelsize=12)
        text1 = 'Color-coded by ' + '$V_x$'
        ax.text(0.98, 0.05, text1, color='k', fontsize=24,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='right', verticalalignment='center',
                transform=ax.transAxes)

        fname = (fdir + species + 'tracer_' + str(pindex) +
                 '_' + str(tframe) + '.jpg')
        fig.savefig(fname, dpi=300)

        # plt.show()
        plt.close()


def trans_trajectory_vtu(plot_config, show_plot=True):
    """Transfer particle trajectory into *.vtu format
    """
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    pindex = plot_config['iptl']
    species = plot_config['species']
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    fname = "../data/trajectory/" + pic_run + "/" + plot_config["traj_file"]
    fh = h5py.File(fname, 'r')
    particle_tags = list(fh.keys())
    nptl = len(particle_tags)
    ptl, sz = read_particle_data(pindex, particle_tags, pic_info, fh)
    smime = math.sqrt(pic_info.mime)
    lx_de = pic_info.lx_di * smime
    ly_de = pic_info.ly_di * smime
    lz_de = pic_info.lz_di * smime
    xpos = adjust_pos(ptl['dX'], pic_info.lx_di)
    ypos = adjust_pos(ptl['dY'], pic_info.ly_di)
    zpos = adjust_pos(ptl['dZ'], pic_info.lz_di)
    fdata = {}
    for key in ptl:
        if key not in ['dX', 'dY', 'dZ']:
            fdata[key] = ptl[key]
    fdir = '../data/cori_3d/tracer_vtu/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + "tracer" + str(pindex)
    pointsToVTK(fname, xpos, ypos, zpos, data=fdata)


def trans_trajectory_h5part(plot_config, show_plot=True):
    """Transfer particle trajectory into h5part format
    """
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    pindex = plot_config['iptl']
    species = plot_config['species']
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    fname = "../data/trajectory/" + pic_run + "/" + plot_config["traj_file"]
    fh = h5py.File(fname, 'r')
    particle_tags = list(fh.keys())
    nptl = len(particle_tags)
    ptl, sz = read_particle_data(pindex, particle_tags, pic_info, fh)
    smime = math.sqrt(pic_info.mime)
    nframes, = ptl['dX'].shape
    pdata = np.zeros([6, nframes])
    # ptl['dX'] *= smime
    # ptl['dY'] *= smime
    # ptl['dZ'] *= smime
    fdir = '../data/cori_3d/tracer_h5part/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + "tracer" + str(pindex) + '-di.h5part'
    with h5py.File(fname, 'w') as fh_out:
        for tindex in range(0, nframes):
            print("Time frame: %d" % tindex)
            grp = fh_out.create_group('Step#' + str(tindex))
            index = range(tindex * nptl, (tindex + 1) * nptl)
            for key in ptl:
                grp.create_dataset(key, (1, ), data=ptl[key][tindex:tindex+1])


def trans_trajectory_csv(plot_config):
    """Transfer current HDF5 file to CSV

    Each CSV file contains one trajectory

    """
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    iptl = plot_config['iptl']
    species = plot_config['species']
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    fname = "../data/trajectory/" + pic_run + "/" + plot_config["traj_file"]
    fh = h5py.File(fname, 'r')
    particle_tags = list(fh.keys())
    nptl = len(particle_tags)
    ptl, sz = read_particle_data(iptl, particle_tags, pic_info, fh)
    smime = math.sqrt(pic_info.mime)
    lx_de = pic_info.lx_di * smime
    ly_de = pic_info.ly_di * smime
    lz_de = pic_info.lz_di * smime
    pdata = np.zeros([21, sz])
    pdata[0] = adjust_pos(ptl['dX'], pic_info.lx_di)
    pdata[1] = adjust_pos(ptl['dY'], pic_info.ly_di)
    pdata[2] = adjust_pos(ptl['dZ'], pic_info.lz_di)
    keys = ['dX', 'dY', 'dZ']
    pindex = 3
    for key in ptl:
        if key not in ['dX', 'dY', 'dZ']:
            pdata[pindex] = ptl[key]
            keys.append(key)
            pindex += 1
    fdir = '../data/cori_3d/tracer_csv/' + pic_run + '/'
    mkdir_p(fdir)
    fname = fdir + 'tracer_' + str(iptl) + '.csv'
    # np.savetxt(fname, pdata.T, delimiter=",",
    #            header="x,y,z,ux,uy,uz,gamma,t,Ex,Ey,Ez,Bx,By,Bz")
    df = pd.DataFrame(pdata.T)
    df.to_csv(fname, mode='w', index=True, header=keys)


def get_crossings(pos, length):
    """Get the crossing points along one axis

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
    for i in range(nt-1):
        if (pos[i]-pos[i+1] > 0.1*length):
            crossings.append(i)
            offset += length
            offsets.append(offset)
        if (pos[i]-pos[i+1] < -0.1*length):
            crossings.append(i)
            offset -= length
            offsets.append(offset)
    nc = len(crossings)
    if nc > 0:
        crossings = np.asarray(crossings)
        offsets = np.asarray(offsets)
    return (nc, crossings, offsets)


def plot_traj_pub(plot_config, show_plot=True):
    """Plot particle trajectory for publication
    """
    tframe = plot_config["tframe"]
    bg = plot_config["bg"]
    bg_str = str(int(bg * 10)).zfill(2)
    pic_run = "3D-Lx150-bg" + str(bg) + "-150ppc-2048KNL"
    if bg_str == '02':
        pic_run += "-tracking"
    root_dir = "/net/scratch3/xiaocanli/reconnection/Cori_runs/"
    pic_run_dir = root_dir + pic_run + '/'
    pindex = plot_config['iptl']
    species = plot_config['species']
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    fname = "../data/trajectory/" + pic_run + "/" + plot_config["traj_file"]
    fh = h5py.File(fname, 'r')
    particle_tags = list(fh.keys())
    nptl = len(particle_tags)
    ptl, sz = read_particle_data(pindex, particle_tags, pic_info, fh)
    smime = math.sqrt(pic_info.mime)
    lx_de = pic_info.lx_di * smime
    ly_de = pic_info.ly_di * smime
    lz_de = pic_info.lz_di * smime
    x_di = pic_info.x_di
    y_di = pic_info.y_di
    z_di = pic_info.z_di
    xr2_di = x_di[1::2]
    yr2_di = y_di[1::2]
    zr2_di = z_di[1::2]
    xpos = adjust_pos(ptl['dX'], pic_info.lx_di)
    ypos = adjust_pos(ptl['dY'], pic_info.ly_di)
    zpos = adjust_pos(ptl['dZ'], pic_info.lz_di)
    kene = ptl['gamma'] - 1

    pic_run = "3D-Lx150-bg" + str(bg) + "-150ppc-2048KNL"
    pic_run_dir = root_dir + pic_run + '/'
    stride_particle_dump = pic_info.stride_particle_dump
    if species == 'e':
        vth = pic_info.vthe
    else:
        vth = pic_info.vthi
    nbins = 1000
    ndata = nbins + 3  # including magnetic field
    tindex = pic_info.particle_interval * tframe
    gama = 1.0 / math.sqrt(1.0 - 3 * vth**2)
    eth = gama - 1.0
    ebins = np.logspace(-6, 4, nbins)
    ebins /= eth
    kene /= eth
    xmin, xmax = 0, pic_info.lx_di
    zmin, zmax = -pic_info.lz_di * 0.5, pic_info.lz_di * 0.5
    nx, ny, nz = pic_info.nx, pic_info.ny, pic_info.nz
    nxr2, nyr2, nzr2 = nx // 2, ny // 2, nz // 2
    nxr4, nyr4, nzr4 = nx // 4, ny // 4, nz // 4
    smime = math.sqrt(pic_info.mime)
    fname = pic_run_dir + "data-smooth/vexb_kappa_" + str(tindex) + ".gda"
    vdot_kappa = np.fromfile(fname, dtype=np.float32)
    vdot_kappa = vdot_kappa.reshape((nzr2, nyr2, nxr2))

    nrhos = []
    nbands = 7
    for iband in range(nbands):
        fname = (pic_run_dir + "data-smooth2/n" + species + "_" +
                 str(iband) + "_" + str(tindex) + ".gda")
        nrho = np.fromfile(fname, dtype=np.float32)
        nrho = nrho.reshape((nzr4, nyr4, nxr4)) * stride_particle_dump
        nrhos.append(nrho)

    yslice = 440
    print("y-position of slice: %f" % (yslice*pic_info.dy_di*2 - 0.5*pic_info.ly_di))

    # fig = plt.figure(figsize=[3.5, 3.0])
    # rect = [0.13, 0.73, 0.75, 0.235]
    fig = plt.figure(figsize=[3.5, 4.5])
    rect = [0.13, 0.83, 0.75, 0.155]
    vgap = 0.03
    ax = fig.add_axes(rect)
    points = np.array([xpos, zpos]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(kene.min(), kene.max())
    lc = LineCollection(segments, cmap='jet', norm=norm)
    lc.set_array(kene)
    lc.set_linewidth(1)
    ax.add_collection(lc)
    # ax.plot(xpos, zpos, marker='o', markersize=10, color="k")
    # ax.set_xlim([xpos.min(), xpos.max()])
    # ax.set_ylim([zpos.min(), zpos.max()])
    ylim = [-12, 12]
    ax.set_xlim([25, 180])
    ax.set_ylim(ylim)
    ax.plot([xmax, xmax], ylim, color='k', linewidth=0.5,
            linestyle='--')
    ticks = np.linspace(25, 175, 7, dtype=int)
    tick_labels = [str(i) if i <= xmax else str(int(i-xmax)) for i in ticks]
    tick_labels = [r'$' + tl + '$' for tl in tick_labels]
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel('$z/d_i$', fontsize=10, labelpad=0)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.tick_params(labelsize=8)
    # Colorbar
    rect_cbar = np.copy(rect)
    rect_cbar[0] = rect[0] + rect[2] + 0.01
    rect_cbar[2] = 0.02
    cax = fig.add_axes(rect_cbar)
    print("min and max of energy/thermal energy: %f, %f" %
          (kene.min(), kene.max()))
    sm = plt.cm.ScalarMappable(cmap=plt.cm.jet,
                               norm=plt.Normalize(vmin=kene.min(),
                                                  vmax=kene.max()))
    cax.tick_params(axis='y', which='major', direction='in')
    sm._A = []
    cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
    cax.tick_params(labelrotation=90)
    cbar.set_label(r'$(\gamma-1)/\varepsilon_\text{th}$',
                   fontsize=10, labelpad=1)
    # ticks = np.linspace(0, te_spect, 6) * dtf
    # ticks = np.concatenate(([10], ticks))
    # cbar.set_ticks(ticks)
    # cax.tick_params(labelrotation=90)
    # # cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(),
    # #                         rotation='vertical')
    cbar.ax.tick_params(labelsize=8)
    cax.yaxis.set_ticks_position('right')
    cax.yaxis.set_label_position('right')

    # rect[1] -= 0.08 + rect[3]
    rect[1] -= 0.05 + rect[3]
    ax1 = fig.add_axes(rect)
    nmin, nmax = 1E-4, 1E-2
    iband = 4
    label1 = (r'3D: $n(' + str(2**(iband-1)*10) + r'\varepsilon_\text{th} < ' +
              r'\varepsilon < ' + str(2**iband*10) + r'\varepsilon_\text{th})$')
    p1 = ax1.imshow(nrhos[iband][:, yslice//2, :] + 1E-10,
                    extent=[xmin, xmax, zmin, zmax],
                    norm = LogNorm(vmin=nmin, vmax=nmax),
                    cmap=plt.cm.plasma, aspect='auto',
                    origin='lower', interpolation='bicubic')
    ax1.tick_params(bottom=True, top=True, left=True, right=True)
    ax1.tick_params(axis='x', which='minor', direction='in')
    ax1.tick_params(axis='x', which='major', direction='in')
    ax1.tick_params(axis='y', which='minor', direction='in')
    ax1.tick_params(axis='y', which='major', direction='in')
    ax1.set_ylabel('$z/d_i$', fontsize=10, labelpad=0)
    ax1.tick_params(labelsize=8)
    ax1.tick_params(axis='x', labelbottom=False)
    ax1.set_ylim([-20, 20])
    ax1.text(0.97, 0.8, label1, color='w', fontsize=8,
             bbox=dict(facecolor='none', alpha=0.75,
                       edgecolor='none', boxstyle="round,pad=0.1"),
             horizontalalignment='right',
             verticalalignment='center',
             transform=ax1.transAxes)
    rect_cbar = np.copy(rect)
    rect_cbar[0] += rect[2] + 0.01
    rect_cbar[2] = 0.02
    cbar_ax = fig.add_axes(rect_cbar)
    cbar = fig.colorbar(p1, cax=cbar_ax, extend='both')
    cbar_ax.tick_params(bottom=False, top=False, left=False, right=True)
    cbar_ax.tick_params(axis='y', which='major', direction='in')
    cbar_ax.tick_params(axis='y', which='minor', direction='in', right=False)
    cbar.ax.tick_params(labelsize=8)

    rect[1] -= vgap + rect[3]
    ax2 = fig.add_axes(rect)
    vmin, vmax = -1.0, 1.0
    knorm = 100 if bg_str == '02' else 400
    fdata = vdot_kappa[:, yslice, :]*knorm
    ng = 3
    kernel = np.ones((ng,ng)) / float(ng*ng)
    fdata = signal.convolve2d(fdata, kernel, mode='same')
    p1 = ax2.imshow(fdata, extent=[xmin, xmax, zmin, zmax],
                    vmin=vmin, vmax=vmax,
                    cmap=plt.cm.seismic, aspect='auto',
                    origin='lower', interpolation='bicubic')
    cs = ax2.contour(xr2_di, zr2_di, np.abs(fdata), colors='k',
                     linewidths=0.25, levels=[0.1])
    ax2.tick_params(bottom=True, top=True, left=True, right=True)
    ax2.tick_params(axis='x', which='minor', direction='in')
    ax2.tick_params(axis='x', which='major', direction='in')
    ax2.tick_params(axis='y', which='minor', direction='in')
    ax2.tick_params(axis='y', which='major', direction='in')
    ax2.set_ylabel('$z/d_i$', fontsize=10, labelpad=0)
    ax2.tick_params(axis='x', labelbottom=False)
    ax2.tick_params(labelsize=8)
    ax2.set_ylim([-20, 20])
    label1 = r'3D: $' + str(knorm) + r'\boldsymbol{v}_{\boldsymbol{E}}\cdot\boldsymbol{\kappa}$'
    ax2.text(0.97, 0.8, label1, color='k', fontsize=8,
             bbox=dict(facecolor='none', alpha=0.5,
                       edgecolor='none', boxstyle="round,pad=0.2"),
             horizontalalignment='right',
             verticalalignment='center',
             transform=ax2.transAxes)
    rect_cbar = np.copy(rect)
    rect_cbar[0] += rect[2] + 0.01
    rect_cbar[2] = 0.02
    cbar_ax = fig.add_axes(rect_cbar)
    cbar = fig.colorbar(p1, cax=cbar_ax, extend='both')
    cbar_ax.tick_params(axis='y', which='major', direction='in')
    cbar_ax.tick_params(axis='y', which='minor', direction='in')
    cbar.ax.tick_params(labelsize=8)

    pic_run = "2D-Lx150-bg" + str(bg) + "-150ppc-16KNL"
    pic_run_dir = root_dir + pic_run + '/'
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    nx, nz = pic_info.nx, pic_info.nz
    iband = 4
    tindex = tframe * pic_info.fields_interval
    fname = (pic_run_dir + "data-smooth2/n" + species + "_" +
             str(iband) + "_" + str(tindex) + ".gda")
    nhigh = np.fromfile(fname, dtype=np.float32)
    nhigh = nhigh.reshape((nz, nx))
    rect[1] -= vgap + rect[3]
    ax3 = fig.add_axes(rect)
    p3 = ax3.imshow(nhigh + 1E-10,
                    extent=[xmin, xmax, zmin, zmax],
                    norm = LogNorm(vmin=nmin, vmax=nmax),
                    cmap=plt.cm.plasma, aspect='auto',
                    origin='lower', interpolation='bicubic')
    ax3.tick_params(bottom=True, top=True, left=True, right=True)
    ax3.tick_params(axis='x', which='minor', direction='in')
    ax3.tick_params(axis='x', which='major', direction='in')
    ax3.tick_params(axis='y', which='minor', direction='in')
    ax3.tick_params(axis='y', which='major', direction='in')
    ax3.set_ylabel('$z/d_i$', fontsize=10, labelpad=0)
    ax3.tick_params(labelsize=8)
    ax3.tick_params(axis='x', labelbottom=False)
    ax3.set_ylim([-20, 20])
    label3 = (r'2D: $n(' + str(2**(iband-1)*10) + r'\varepsilon_\text{th} < ' +
              r'\varepsilon < ' + str(2**iband*10) + r'\varepsilon_\text{th})$')
    ax3.text(0.97, 0.8, label3, color='w', fontsize=8,
             bbox=dict(facecolor='none', alpha=0.75,
                       edgecolor='none', boxstyle="round,pad=0.1"),
             horizontalalignment='right',
             verticalalignment='center',
             transform=ax3.transAxes)
    rect_cbar = np.copy(rect)
    rect_cbar[0] += rect[2] + 0.01
    rect_cbar[2] = 0.02
    cbar_ax = fig.add_axes(rect_cbar)
    cbar = fig.colorbar(p3, cax=cbar_ax, extend='both')
    cbar_ax.tick_params(bottom=False, top=False, left=False, right=True)
    cbar_ax.tick_params(axis='y', which='major', direction='in')
    cbar_ax.tick_params(axis='y', which='minor', direction='in', right=False)
    cbar.ax.tick_params(labelsize=8)

    kwargs = {"current_time": tframe,
              "xl": 0, "xr": pic_info.lx_di,
              "zb": -pic_info.lz_di, "zt": pic_info.lz_di}
    fname = pic_run_dir + "data/vexb_kappa.gda"
    x, z, vexb_kappa = read_2d_fields(pic_info, fname, **kwargs)
    rect[1] -= vgap + rect[3]
    ax4 = fig.add_axes(rect)
    vexb_kappa = vexb_kappa * 100
    vmin, vmax = -1.0, 1.0
    p4 = ax4.imshow(vexb_kappa, extent=[xmin, xmax, zmin, zmax],
                    vmin=vmin, vmax=vmax,
                    cmap=plt.cm.seismic, aspect='auto',
                    origin='lower', interpolation='bicubic')
    ax4.tick_params(bottom=True, top=True, left=True, right=True)
    ax4.tick_params(axis='x', which='minor', direction='in')
    ax4.tick_params(axis='x', which='major', direction='in')
    ax4.tick_params(axis='y', which='minor', direction='in')
    ax4.tick_params(axis='y', which='major', direction='in')
    ax4.set_xlabel('$x/d_i$', fontsize=10, labelpad=0)
    ax4.set_ylabel('$z/d_i$', fontsize=10, labelpad=0)
    ax4.tick_params(labelsize=8)
    ax4.set_ylim([-20, 20])
    label4 = r'2D: $' + str(knorm) + r'\boldsymbol{v}_{\boldsymbol{E}}\cdot\boldsymbol{\kappa}$'
    ax4.text(0.97, 0.8, label4, color='k', fontsize=8,
             bbox=dict(facecolor='none', alpha=0.5,
                       edgecolor='none', boxstyle="round,pad=0.2"),
             horizontalalignment='right',
             verticalalignment='center',
             transform=ax4.transAxes)
    rect_cbar = np.copy(rect)
    rect_cbar[0] += rect[2] + 0.01
    rect_cbar[2] = 0.02
    cbar_ax = fig.add_axes(rect_cbar)
    cbar = fig.colorbar(p4, cax=cbar_ax, extend='both')
    cbar_ax.tick_params(axis='y', which='major', direction='in')
    cbar_ax.tick_params(axis='y', which='minor', direction='in')
    cbar.ax.tick_params(labelsize=8)

    fdir = '../img/cori_3d/tracer_nhigh_vkappa/' + pic_run + '/tframe_' + str(tframe) + '/'
    mkdir_p(fdir)
    fname = (fdir + 'tracer' + str(pindex) + '_' + species +
             '_yslice_' + str(yslice) + ".pdf")
    fig.savefig(fname, dpi=400)

    if show_plot:
        plt.show()
    else:
        plt.close()


def traj_xz_xgamma(plot_config, show_plot=True):
    """Plot particle trajectory in x-z and x-gamma format
    """
    tframe = plot_config["tframe"]
    bg = plot_config["bg"]
    bg_str = str(int(bg * 10)).zfill(2)
    pic_run = "3D-Lx150-bg" + str(bg) + "-150ppc-2048KNL"
    if bg_str == '02':
        pic_run += "-tracking"
    root_dir = "/net/scratch3/xiaocanli/reconnection/Cori_runs/"
    pic_run_dir = root_dir + pic_run + '/'
    pindex = plot_config['iptl']
    species = plot_config['species']
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    fname = "../data/trajectory/" + pic_run + "/" + plot_config["traj_file"]
    fh = h5py.File(fname, 'r')
    particle_tags = list(fh.keys())
    nptl = len(particle_tags)
    ptl, sz = read_particle_data(pindex, particle_tags, pic_info, fh)
    smime = math.sqrt(pic_info.mime)
    lx_de = pic_info.lx_di * smime
    ly_de = pic_info.ly_di * smime
    lz_de = pic_info.lz_di * smime
    x_di = pic_info.x_di
    y_di = pic_info.y_di
    z_di = pic_info.z_di
    xr2_di = x_di[1::2]
    yr2_di = y_di[1::2]
    zr2_di = z_di[1::2]
    xpos = adjust_pos(ptl['dX'], pic_info.lx_di)
    ypos = adjust_pos(ptl['dY'], pic_info.ly_di)
    zpos = adjust_pos(ptl['dZ'], pic_info.lz_di)
    kene = ptl['gamma'] - 1

    pic_run = "3D-Lx150-bg" + str(bg) + "-150ppc-2048KNL"
    pic_run_dir = root_dir + pic_run + '/'
    stride_particle_dump = pic_info.stride_particle_dump
    if species == 'e':
        vth = pic_info.vthe
    else:
        vth = pic_info.vthi
    nbins = 1000
    ndata = nbins + 3  # including magnetic field
    tindex = pic_info.particle_interval * tframe
    gama = 1.0 / math.sqrt(1.0 - 3 * vth**2)
    eth = gama - 1.0
    ebins = np.logspace(-6, 4, nbins)
    ebins /= eth
    kene /= eth
    xmin, xmax = 0, pic_info.lx_di
    zmin, zmax = -pic_info.lz_di * 0.5, pic_info.lz_di * 0.5
    nx, ny, nz = pic_info.nx, pic_info.ny, pic_info.nz
    nxr2, nyr2, nzr2 = nx // 2, ny // 2, nz // 2
    nxr4, nyr4, nzr4 = nx // 4, ny // 4, nz // 4
    smime = math.sqrt(pic_info.mime)

    fig = plt.figure(figsize=[3.5, 3.0])
    rect = [0.14, 0.76, 0.74, 0.2]
    vgap = 0.03
    ax = fig.add_axes(rect)
    points = np.array([xpos, zpos]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(kene.min(), kene.max())
    lc = LineCollection(segments, cmap='jet', norm=norm)
    lc.set_array(kene)
    lc.set_linewidth(1)
    ax.add_collection(lc)
    ylim = [-12, 12]
    ax.set_xlim([25, 180])
    ax.set_ylim(ylim)
    ax.plot([xmax, xmax], ylim, color='k', linewidth=0.5,
            linestyle='--')
    # ticks = np.linspace(25, 175, 7, dtype=int)
    # tick_labels = [str(i) if i <= xmax else str(int(i-xmax)) for i in ticks]
    # tick_labels = [r'$' + tl + '$' for tl in tick_labels]
    # ax.set_xticks(ticks)
    # ax.set_xticklabels(tick_labels)
    ax.tick_params(axis='x', labelbottom=False)
    ax.set_ylabel('$z/d_i$', fontsize=10, labelpad=1.5)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.tick_params(labelsize=8)
    # Colorbar
    rect_cbar = np.copy(rect)
    rect_cbar[0] = rect[0] + rect[2] + 0.01
    rect_cbar[2] = 0.02
    cax = fig.add_axes(rect_cbar)
    print("min and max of energy/thermal energy: %f, %f" %
          (kene.min(), kene.max()))
    sm = plt.cm.ScalarMappable(cmap=plt.cm.jet,
                               norm=plt.Normalize(vmin=kene.min(),
                                                  vmax=kene.max()))
    cax.tick_params(axis='y', which='major', direction='in')
    sm._A = []
    cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
    cax.tick_params(labelrotation=90)
    cbar.set_label(r'$\varepsilon/\varepsilon_\text{th}$',
                   fontsize=10, labelpad=1)
    cbar.ax.tick_params(labelsize=8)
    cax.yaxis.set_ticks_position('right')
    cax.yaxis.set_label_position('right')

    # x-gamma
    rect[1] -= vgap + 0.6
    rect[3] = 0.6
    ax1 = fig.add_axes(rect)
    ax1.plot(xpos, kene, color='k', linewidth=1, linestyle='-')
    ylim = [0, 250]
    ax1.plot([xmax, xmax], ylim, color='k',
             linewidth=0.5, linestyle='--')
    ax1.set_ylabel(r'$\varepsilon/\varepsilon_\text{th}$', fontsize=10)
    ax1.tick_params(bottom=True, top=True, left=True, right=True)
    ax1.tick_params(axis='x', which='minor', direction='in')
    ax1.tick_params(axis='x', which='major', direction='in')
    ax1.tick_params(axis='y', which='minor', direction='in')
    ax1.tick_params(axis='y', which='major', direction='in')
    ax1.tick_params(labelsize=8)
    ax1.set_xlim(ax.get_xlim())
    ax1.set_ylim(ylim)
    ticks = np.linspace(25, 175, 7, dtype=int)
    tick_labels = [str(i) if i <= xmax else str(int(i-xmax)) for i in ticks]
    tick_labels = [r'$' + tl + '$' for tl in tick_labels]
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(tick_labels)
    ax1.set_xlabel('$x/d_i$', fontsize=10)

    fdir = '../img/cori_3d/tracer_pub/' + pic_run + '/'
    mkdir_p(fdir)
    fname = (fdir + 'tracer' + str(pindex) + '_' + species + ".pdf")
    fig.savefig(fname)

    if show_plot:
        plt.show()
    else:
        plt.close()


def piecewise_trajectory(plot_config):
    """Save piecewise trajectory
    """
    iptl = plot_config['iptl']
    tint = plot_config['tint']
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    species = plot_config['species']
    iptl = plot_config['iptl']
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    fname = "../data/trajectory/" + pic_run + "/" + plot_config["traj_file"]
    fh = h5py.File(fname, 'r')
    file = h5py.File(fname,'r')
    particle_tags = list(fh.keys())
    nptl = len(particle_tags)
    ptl, sz = read_particle_data(iptl, particle_tags, pic_info, fh)
    smime = math.sqrt(pic_info.mime)
    lx_de = pic_info.lx_di * smime
    ly_de = pic_info.ly_di * smime
    lz_de = pic_info.lz_di * smime
    nframes, = ptl['dX'].shape
    pdata = np.zeros([6, nframes])
    pdata[0] = np.array(ptl['dX']) * smime
    pdata[1] = np.array(ptl['dY']) * smime
    pdata[2] = np.array(ptl['dZ']) * smime
    pdata[3] = np.array(ptl['Ux'])
    pdata[4] = np.array(ptl['Uy'])
    pdata[5] = np.array(ptl['Uz'])
    fdir = '../data/cori_3d/piecewise_trajectory/' + pic_run + '/'
    fdir += 'ptl_' + str(iptl) + '/'
    mkdir_p(fdir)
    nc_x, cross_x, offsets_x = get_crossings(pdata[0], lx_de)
    nc_y, cross_y, offsets_y = get_crossings(pdata[1], ly_de)
    nc_z, cross_z, offsets_z = get_crossings(pdata[2], lz_de)
    crossings = np.unique(np.sort(np.hstack((cross_x, cross_y, cross_z))))
    ncross, = crossings.shape
    icross1 = 0
    icross2 = 0
    for tframe in range(nframes):
        print("Time frame: %d" % tframe)
        if tframe > crossings[icross2]:
            icross1 = icross2
            if icross2 < ncross - 1:
                icross2 += 1
            else:
                icross2 = ncross - 1
        neighbors = [0, crossings[icross1],
                     tframe - tint, tframe, tframe + tint,
                     crossings[icross2], nframes]
        neighbors = sorted(list(set(neighbors)))
        tindex = neighbors.index(tframe)
        if tframe == 0:
            ts = 0
            te = int(neighbors[tindex + 1])
        elif tframe == crossings[icross2]:
            ts = int(neighbors[tindex - 1])
            te = tframe + 1
        else:
            ts = int(neighbors[tindex - 1])
            te = int(neighbors[tindex + 1])
        fname = fdir + "tframe_" + str(tframe)
        pointsToVTK(fname, pdata[0, ts+1:te], pdata[1, ts+1:te], pdata[2, ts+1:te],
                    data = {"ux": pdata[3, ts+1:te],
                            "uy": pdata[4, ts+1:te],
                            "uz": pdata[5, ts+1:te]})


def piecewise_trajectory_cross(plot_config):
    """Save piecewise trajectory for boundary-cross case
    """
    iptl = plot_config['iptl']
    tint = plot_config['tint']
    pic_run = plot_config["pic_run"]
    pic_run_dir = plot_config["pic_run_dir"]
    species = plot_config['species']
    iptl = plot_config['iptl']
    picinfo_fname = '../data/pic_info/pic_info_' + pic_run + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    fname = "../data/trajectory/" + pic_run + "/" + plot_config["traj_file"]
    fh = h5py.File(fname, 'r')
    file = h5py.File(fname,'r')
    particle_tags = list(fh.keys())
    nptl = len(particle_tags)
    ptl, sz = read_particle_data(iptl, particle_tags, pic_info, fh)
    smime = math.sqrt(pic_info.mime)
    lx_de = pic_info.lx_di * smime
    ly_de = pic_info.ly_di * smime
    lz_de = pic_info.lz_di * smime
    nframes, = ptl['dX'].shape
    pdata = np.zeros([8, nframes])
    pdata[0] = np.array(ptl['dX'])
    pdata[1] = np.array(ptl['dY'])
    pdata[2] = np.array(ptl['dZ'])
    pdata[3] = np.array(ptl['Ux'])
    pdata[4] = np.array(ptl['Uy'])
    pdata[5] = np.array(ptl['Uz'])
    pdata[6] = np.array(ptl['t'])
    pdata[7] = np.array(ptl['gamma'])
    fdir = '../data/cori_3d/piecewise_trajectory_cross/' + pic_run + '/'
    fdir += 'ptl_' + str(iptl) + '/'
    mkdir_p(fdir)
    pdata[0] = adjust_pos(pdata[0], pic_info.lx_di)
    pdata[1] = adjust_pos(pdata[1], pic_info.ly_di)
    pdata[2] = adjust_pos(pdata[2], pic_info.lz_di)
    for tframe in range(nframes):
        print("Time frame: %d" % tframe)
        if tframe == 0:
            ts, te = 0, 2
        elif tframe == nframes - 1:
            ts, te = nframes - 2, nframes
        else:
            ts, te = tframe - 1, tframe + 1
        fname = fdir + "tframe_" + str(tframe)
        pointsToVTK(fname, pdata[0, ts:te], pdata[1, ts:te], pdata[2, ts:te],
                    data = {"ux": pdata[3, ts:te],
                            "uy": pdata[4, ts:te],
                            "uz": pdata[5, ts:te],
                            "t": pdata[6, ts:te],
                            "gamma": pdata[7, ts:te]})


def get_cmd_args():
    """Get command line arguments
    """
    default_pic_run = '3D-Lx150-bg0.2-150ppc-2048KNL-tracking'
    default_pic_run_dir = ('/net/scratch3/xiaocanli/reconnection/Cori_runs/' +
                           default_pic_run + '/')
    parser = argparse.ArgumentParser(description='Analysis for Cori 3D runs')
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
    parser.add_argument('--bg', action="store", default='0.2', type=float,
                        help='Guide field strength')
    parser.add_argument('--iptl', action="store", default='0', type=int,
                        help='particle tracer ID')
    parser.add_argument('--nptl', action="store", default='1000', type=int,
                        help='Total number of particle tracers')
    parser.add_argument('--traj_file', action="store", default='electrons_200.h5p',
                        help='Trajectory file name')
    parser.add_argument('--tint', action="store", default='20', type=int,
                        help='Number of steps before and after current step ' +
                        'for piecewise trajectory')
    parser.add_argument('--plot_traj', action="store_true", default=False,
                        help='whether to plot particle trajectory')
    parser.add_argument('--plot_traj_simple', action="store_true", default=False,
                        help='whether to plot particle trajectory in a simple way')
    parser.add_argument('--plot_traj_pub', action="store_true", default=False,
                        help='whether to plot particle trajectory for publication')
    parser.add_argument('--traj_xz_xgamma', action="store_true", default=False,
                        help='whether to plot particle trajectory in xz and xgamma format')
    parser.add_argument('--to_h5part', action="store_true", default=False,
                        help='whether to transfer trajectory data into H5Part')
    parser.add_argument('--multi_tracer', action="store_true", default=False,
                        help='whether to analyze multiple tracers')
    parser.add_argument('--traj_movie', action="store_true", default=False,
                        help='whether to plot for a trajectory movie')
    parser.add_argument('--trans_traj_vtu', action="store_true", default=False,
                        help='whether to transfer trajectory to vtu format')
    parser.add_argument('--trans_traj_csv', action="store_true", default=False,
                        help='whether to transfer trajectory to csv format')
    parser.add_argument('--trans_traj_h5part', action="store_true", default=False,
                        help='whether to transfer trajectory to h5part format')
    parser.add_argument('--piecewise_traj', action="store_true", default=False,
                        help="whether to get piecewise trajectory")
    parser.add_argument('--piecewise_traj_cross', action="store_true", default=False,
                        help="whether to get piecewise trajectory for boundary-cross case")
    return parser.parse_args()


def analysis_single_frames(plot_config, args):
    """Analysis for multiple time frames
    """
    tframe = args.tframe
    if args.plot_traj:
        if args.multi_tracer:
            for iptl in range(plot_config['nptl']):
                print("Particle ID: %d" % iptl)
                plot_config['iptl'] = iptl
                plot_trajectory(plot_config, show_plot=False)
        else:
            plot_trajectory(plot_config)
    elif args.plot_traj_simple:
        if args.multi_tracer:
            for iptl in range(plot_config['nptl']):
                print("Particle ID: %d" % iptl)
                plot_config['iptl'] = iptl
                plot_trajectory_simple(plot_config, show_plot=False)
        else:
            plot_trajectory_simple(plot_config)
    elif args.traj_movie:
            plot_trajectory_movie(plot_config)
    elif args.to_h5part:
        transfer_to_h5part(plot_config)
    elif args.trans_traj_vtu:
        trans_trajectory_vtu(plot_config)
    elif args.trans_traj_h5part:
        trans_trajectory_h5part(plot_config)
    elif args.trans_traj_csv:
        trans_trajectory_csv(plot_config)
    elif args.plot_traj_pub:
        plot_traj_pub(plot_config)
    elif args.traj_xz_xgamma:
        traj_xz_xgamma(plot_config)
    elif args.piecewise_traj:
        piecewise_trajectory(plot_config)
    elif args.piecewise_traj_cross:
        piecewise_trajectory_cross(plot_config)


def process_input(plot_config, args, tframe):
    """process one time frame"""
    plot_config["tframe"] = tframe


def analysis_multi_frames(plot_config, args):
    """Analysis for multiple time frames
    """
    tframes = range(plot_config["tstart"], plot_config["tend"] + 1)
    if args.time_loop:
        for tframe in tframes:
            print("Time frame: %d" % tframe)
            plot_config["tframe"] = tframe
            pass
    else:
        ncores = multiprocessing.cpu_count()
        ncores = 8
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
    plot_config["traj_file"] = args.traj_file
    plot_config["iptl"] = args.iptl
    plot_config["nptl"] = args.nptl
    plot_config["bg"] = args.bg
    plot_config["tint"] = args.tint
    if args.multi_frames:
        analysis_multi_frames(plot_config, args)
    else:
        analysis_single_frames(plot_config, args)


if __name__ == "__main__":
    main()
