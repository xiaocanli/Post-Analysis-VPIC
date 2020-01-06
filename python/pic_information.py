#!/usr/bin/env python3
"""
Get particle-in-cell (VPIC) simulation information and save it in a JSON file.
You can run the code by "python pic_information.py $pic_run_dir $pic_run_name",
where pic_run_dir is the directory for your PIC run and pic_run_name is the
unique name for your PIC run. The JSON file will be saved in "../data/pic_info/".
"""
import collections
import errno
import math
import os.path
import struct
import sys
from os import listdir
from os.path import isfile, join

import numpy as np
import simplejson as json

from runs_name_path import *
from serialize_json import data_to_json, json_to_data


def get_pic_info(base_directory, run_name):
    """Get particle-in-cell simulation information.

    Args:
        base_directory: the base directory for different runs.
        run_name: name of the simulation
    """
    pic_initial_info = read_pic_info(base_directory)
    dtwpe = pic_initial_info.dtwpe
    dtwce = pic_initial_info.dtwce
    dtwci = pic_initial_info.dtwci
    dtwpi = dtwpe / math.sqrt(pic_initial_info.mime)
    energy_interval = pic_initial_info.energy_interval
    deck_file = get_main_source_filename(base_directory)
    fields_interval, particle_interval, trace_interval = \
            get_output_intervals(dtwpe, dtwce, dtwpi, dtwci, base_directory, deck_file)
    ntf = get_fields_frames(base_directory, fields_interval)
    dt_fields = fields_interval * dtwci
    dt_particles = particle_interval * dtwci
    ntp = ntf / (particle_interval / fields_interval)
    tparticles = np.arange(ntp) * dt_particles
    tfields = np.arange(ntf) * dt_fields
    dt_energy = energy_interval * dtwci
    dte_wpe = dt_energy * dtwpe / dtwci
    pic_ene = read_pic_energies(dt_energy, dte_wpe, base_directory)

    pic_times = collections.namedtuple("pic_times",
                                       ['ntf', 'dt_fields', 'tfields', 'ntp',
                                        'dt_particles', 'tparticles',
                                        'dt_energy', 'fields_interval',
                                        'particle_interval', 'trace_interval'])
    pic_times_info = pic_times(ntf=ntf,
                               dt_fields=dt_fields,
                               dt_particles=dt_particles,
                               tfields=tfields,
                               dt_energy=dt_energy,
                               ntp=ntp,
                               tparticles=tparticles,
                               fields_interval=fields_interval,
                               particle_interval=particle_interval,
                               trace_interval=trace_interval)
    pic_topology = get_pic_topology(base_directory, deck_file)
    pic_run = collections.namedtuple("pic_run", ['run_dir', 'run_name'])
    pic_run_info = pic_run(run_dir=base_directory,
                           run_name=run_name)
    pic_information = collections.namedtuple(
        "pic_information", pic_initial_info._fields + pic_times_info._fields +
        pic_ene._fields + pic_topology._fields + pic_run_info._fields)
    pic_info = pic_information(*(pic_initial_info + pic_times_info +
                                 pic_ene + pic_topology + pic_run_info))
    return pic_info


def read_pic_energies(dte_wci, dte_wpe, base_directory):
    """Read particle-in-cell simulation energies.

    Args:
        dte_wci: the time interval for energies diagnostics (in 1/wci).
        dte_wpe: the time interval for energies diagnostics (in 1/wpe).
        base_directory: the base directory for different runs.
    """
    fname = base_directory + 'rundata/energies'
    try:
        f = open(fname, 'r')
    except IOError:
        print('cannot open %s' % fname)
        fname = base_directory + 'energies'
        print('switch file to %s' % fname)
        f = open(fname, 'r')
    f.close()
    content = np.genfromtxt(fname, skip_header=3)
    f.close()
    nte, nvar = content.shape
    tenergy = np.arange(nte) * dte_wci
    ene_ex = content[:, 1]
    ene_ey = content[:, 2]
    ene_ez = content[:, 3]
    ene_bx = content[:, 4]
    ene_by = content[:, 5]
    ene_bz = content[:, 6]
    kene_i = content[:, 7]  # kinetic energy for ions
    kene_e = content[:, 8]
    ene_electric = ene_ex + ene_ey + ene_ez
    ene_magnetic = ene_bx + ene_by + ene_bz
    dene_ex = np.gradient(ene_ex) / dte_wpe
    dene_ey = np.gradient(ene_ey) / dte_wpe
    dene_ez = np.gradient(ene_ez) / dte_wpe
    dene_bx = np.gradient(ene_bx) / dte_wpe
    dene_by = np.gradient(ene_by) / dte_wpe
    dene_bz = np.gradient(ene_bz) / dte_wpe
    dene_electric = np.gradient(ene_electric) / dte_wpe
    dene_magnetic = np.gradient(ene_magnetic) / dte_wpe
    dkene_i = np.gradient(kene_i) / dte_wpe
    dkene_e = np.gradient(kene_e) / dte_wpe
    pic_energies = collections.namedtuple('pic_energies', [
        'nte', 'tenergy', 'ene_ex', 'ene_ey', 'ene_ez', 'ene_bx', 'ene_by',
        'ene_bz', 'kene_i', 'kene_e', 'ene_electric', 'ene_magnetic',
        'dene_ex', 'dene_ey', 'dene_ez', 'dene_bx', 'dene_by', 'dene_bz',
        'dkene_i', 'dkene_e', 'dene_electric', 'dene_magnetic'
    ])
    pic_ene = pic_energies(nte=nte,
                           tenergy=tenergy,
                           ene_ex=ene_ex,
                           ene_ey=ene_ey,
                           ene_ez=ene_ez,
                           ene_bx=ene_bx,
                           ene_by=ene_by,
                           ene_bz=ene_bz,
                           kene_i=kene_i,
                           kene_e=kene_e,
                           ene_electric=ene_electric,
                           ene_magnetic=ene_magnetic,
                           dene_ex=dene_ex,
                           dene_ey=dene_ey,
                           dene_ez=dene_ez,
                           dene_bx=dene_bx,
                           dene_by=dene_by,
                           dene_bz=dene_bz,
                           dkene_i=dkene_i,
                           dkene_e=dkene_e,
                           dene_electric=dene_electric,
                           dene_magnetic=dene_magnetic)
    return pic_ene


def get_fields_frames(base_directory, fields_interval):
    """Get the total number of time frames for fields.

    Args:
        base_directory: the base directory for different runs.
        fields_interval: time interval to dump fields
    Returns:
        ntf: the total number of output time frames for fields.
    """
    pic_initial_info = read_pic_info(base_directory)
    nx = pic_initial_info.nx
    ny = pic_initial_info.ny
    nz = pic_initial_info.nz
    fname_fields = base_directory + '/fields/T.1'
    fname_fields_sub_dir = base_directory + '/fields/0/T.1'
    fname_fields_h5 = base_directory + '/field_hdf5/T.0'
    fname_ex = base_directory + '/data/ex.gda'
    fname_Ex = base_directory + '/data/Ex.gda'
    fname_bx = base_directory + '/data/bx_0.gda'
    fname_Bx = base_directory + '/data/Bx_0.gda'
    if os.path.isfile(fname_bx) or os.path.isfile(fname_Bx):
        current_time = 0
        ntf = 0
        is_exist = True
        while (is_exist):
            ntf += 1
            current_time += fields_interval
            fname1 = base_directory + '/data/bx_' + str(current_time) + '.gda'
            fname2 = base_directory + '/data/Bx_' + str(current_time) + '.gda'
            is_exist = os.path.isfile(fname1) or os.path.isfile(fname2)
    elif os.path.isfile(fname_ex):
        file_size = os.path.getsize(fname_ex)
        ntf = int(file_size / (nx * ny * nz * 4))
    elif os.path.isfile(fname_Ex):
        file_size = os.path.getsize(fname_Ex)
        ntf = int(file_size / (nx * ny * nz * 4))
    elif os.path.isdir(fname_fields):
        current_time = 0
        ntf = 0
        is_exist = True
        while (is_exist):
            ntf += 1
            current_time += fields_interval
            fname = base_directory + '/fields/T.' + str(current_time)
            is_exist = os.path.isdir(fname)
    elif os.path.isdir(fname_fields_sub_dir):
        current_time = 0
        ntf = 0
        is_exist = True
        while (is_exist):
            ntf += 1
            current_time += fields_interval
            fname = base_directory + '/fields/0/T.' + str(current_time)
            is_exist = os.path.isdir(fname)
    elif os.path.isdir(fname_fields_h5):
        current_time = 0
        ntf = 0
        is_exist = True
        while (is_exist):
            ntf += 1
            current_time += fields_interval
            fname = base_directory + '/field_hdf5/T.' + str(current_time)
            is_exist = os.path.isdir(fname)
    else:
        print('Cannot find the files to calculate the total frames of fields.')
        return
    return ntf


def get_main_source_filename(base_directory):
    """Get the source file name.

    Get the configuration source file name for the PIC simulation.

    Args:
        base_directory: the base directory for different runs.
    """
    fname = base_directory + '/Makefile'
    try:
        f = open(fname, 'r')
    except IOError as e:
        if e.errno == errno.ENOENT:
            fname = input("The deck filename? ")
            fname = base_directory + '/' + fname
        else:
            raise e
    else:
        content = f.readlines()
        f.close()
        nlines = len(content)
        current_line = 0
        cond1 = not '.cxx' in content[current_line]
        cond2 = not '.cc' in content[current_line]
        cond3 = '#' in content[current_line]
        while (cond1 and cond2) or cond3:
            current_line += 1
            cond1 = not '.cxx' in content[current_line]
            cond2 = not '.cc' in content[current_line]
            cond3 = '#' in content[current_line]
        single_line = content[current_line]
        line_splits = single_line.split(" ")
        filename = line_splits[1]
        fname = base_directory + '/' + filename[:-1]
    return fname


def get_output_intervals(dtwpe, dtwce, dtwpi, dtwci, base_directory, deck_file):
    """
    Get output intervals from the main configuration file for current PIC
    simulation.

    Args:
        dtwpe: the time step in 1/wpe.
        dtwce: the time step in 1/wce.
        dtwpi: the time step in 1/wpi.
        dtwci: the time step in 1/wci.
        base_directory: the base directory for different runs.
        deck_file: simulation deck source file
    """
    fname = deck_file
    try:
        f = open(fname, 'r')
    except IOError:
        print('cannot open %s' % fname)
    else:
        content = f.readlines()
        f.close()
        nlines = len(content)
        current_line = 0
        cond1 = not 'int interval = ' in content[current_line]
        cond2 = '//' in content[current_line]  # commented out
        while cond1 or cond2:
            current_line += 1
            cond1 = not 'int interval = ' in content[current_line]
            cond2 = '//' in content[current_line]  # commented out
        if not '(' in content[current_line]:
            single_line = content[current_line]
            if '*' in content[current_line]:
                line_splits = single_line.split('*')
                word_splits = line_splits[0].split('=')
                time_ratio = float(word_splits[1])
            else:
                line_splits = single_line.split('=')
                time_ratio = 1.0
            word_splits = line_splits[1].split(";")
            word = 'int ' + word_splits[0] + ' = '
            cline = current_line
            # go back to the number for word_splits[0]
            cond1 = not word in content[current_line]
            cond2 = '//' in content[current_line]  # commented out
            while cond1 or cond2:
                current_line -= 1
                cond1 = not word in content[current_line]
                cond2 = '//' in content[current_line]  # commented out
            interval = get_time_interval(content[current_line], dtwpe, dtwce,
                                         dtwpi, dtwci)
            # We assume the interval if trace_interval
            trace_interval = interval
            interval = int(interval * time_ratio)
        else:
            interval = get_time_interval(content[current_line], dtwpe, dtwce,
                                         dtwpi, dtwci)
            trace_interval = 0

        fields_interval = interval

        while not 'int eparticle_interval' in content[current_line]:
            current_line += 1
        single_line = content[current_line]
        line_splits = single_line.split("=")
        if '*' in line_splits[1]:
            word_splits = line_splits[1].split("*")
            particle_interval = int(word_splits[0]) * interval
        else:
            particle_interval = interval

    return (fields_interval, particle_interval, trace_interval)


def get_time_interval(line, dtwpe, dtwce, dtwpi, dtwci):
    """Get time interval from a line

    The line is in the form: int *** = int(5.0/***);

    Args:
        line: one single line
        dtwpe: the time step in 1/wpe.
        dtwce: the time step in 1/wce.
        dtwpi: the time step in 1/wpi.
        dtwci: the time step in 1/wci.
    """
    line_splits = line.split("(")
    word_splits = line_splits[1].split("/")
    interval = float(word_splits[0])
    word2_splits = line_splits[2].split("*")
    dt = 0.0
    if word2_splits[0] == "wpe":
        dt = dtwpe
    elif word2_splits[0] == "wce":
        dt = dtwce
    elif word2_splits[0] == "wpi":
        dt = dtwpi
    elif word2_splits[0] == "wci":
        dt = dtwci

    interval = int(interval / dt)
    return interval


def read_pic_info(base_directory):
    """Read particle-in-cell simulation information.

    Args:
        pic_info: a namedtuple for PIC initial information.
    """
    fname = base_directory + '/info'
    with open(fname) as f:
        content = f.readlines()
    f.close()
    nlines = len(content)
    current_line = 0
    sigmae_c, current_line = get_variable_value('sigma', current_line, content)
    ti_te, current_line = get_variable_value('Ti/Te', current_line, content)
    te, current_line = get_variable_value('Te', current_line, content, match_name=True)
    ti, current_line = get_variable_value('Ti', current_line, content, match_name=True)
    wpe_wce, current_line = get_variable_value('wpe/wce', current_line, content)
    mime, current_line = get_variable_value('mi/me', current_line, content)
    lx, current_line = get_variable_value('Lx/di', current_line, content)
    ly, current_line = get_variable_value('Ly/di', current_line, content)
    lz, current_line = get_variable_value('Lz/di', current_line, content)
    nx, current_line = get_variable_value('nx', current_line, content, int)
    ny, current_line = get_variable_value('ny', current_line, content, int)
    nz, current_line = get_variable_value('nz', current_line, content, int)
    courant, current_line = get_variable_value('courant', current_line, content)
    nproc, current_line = get_variable_value('nproc', current_line, content, int)
    nppc, current_line = get_variable_value('nppc', current_line, content, int)
    b0, current_line = get_variable_value('b0', current_line, content)
    ne, current_line = get_variable_value('Ne', current_line, content)
    dtwpe, current_line = get_variable_value('dt*wpe', current_line, content)
    try:
        dtwce, current_line = get_variable_value_h('dt*wce', content)
    except:
        dtwce = dtwpe * b0
    try:
        dtwci, current_line = get_variable_value_h('dt*wci', content)
    except:
        dtwci = dtwce / mime
    energy_interval, current_line = get_variable_value('energies_interval',
                                                       current_line, content)
    dxde, current_line = get_variable_value('dx/de', current_line, content)
    dyde, current_line = get_variable_value('dy/de', current_line, content)
    dzde, current_line = get_variable_value('dz/de', current_line, content)
    dxdi = dxde / math.sqrt(mime)
    dydi = dyde / math.sqrt(mime)
    dzdi = dzde / math.sqrt(mime)
    x = np.arange(nx) * dxdi
    y = (np.arange(ny) - ny / 2.0 + 0.5) * dydi
    z = (np.arange(nz) - nz / 2.0 + 0.5) * dzdi
    dx_rhoi, current_line = get_variable_value('dx/rhoi', current_line, content)
    dx_rhoe, current_line = get_variable_value('dx/rhoe', current_line, content)
    dx_debye, current_line = get_variable_value('dx/debye', current_line, content)
    n0, current_line = get_variable_value('n0', current_line, content)
    if any('vthi/c' in s for s in content):
        vthi, current_line = get_variable_value('vthi/c', current_line,
                                                content)
        vthe, current_line = get_variable_value('vthe/c', current_line,
                                                content)
    else:  # highly relativistic cases
        vthe = 1.0
        vthi = 1.0
    restart_interval, current_line = get_variable_value('restart_interval',
                                                        current_line, content, int)
    fields_interval_info, current_line = get_variable_value('fields_interval',
                                                            current_line, content, int)
    ehydro_interval, current_line = get_variable_value('ehydro_interval',
                                                       current_line, content, int)
    Hhydro_interval, current_line = get_variable_value('Hhydro_interval',
                                                       current_line, content, int)
    eparticle_interval, current_line = get_variable_value('eparticle_interval',
                                                          current_line, content, int)
    Hparticle_interval, current_line = get_variable_value('Hparticle_interval',
                                                          current_line, content, int)
    quota_check_interval, current_line = get_variable_value('quota_check_interval',
                                                            current_line, content, int)
    particle_tracing, current_line = get_variable_value('particle_tracing',
                                                        current_line, content, int)
    tracer_interval, current_line = get_variable_value('tracer_interval',
                                                       current_line, content, int)
    tracer_pass1_interval, current_line = get_variable_value('tracer_pass1_interval',
                                                             current_line, content, int)
    tracer_pass2_interval, current_line = get_variable_value('tracer_pass2_interval',
                                                             current_line, content, int)
    ntracer, current_line = get_variable_value('Ntracer', current_line, content, int)
    emf_at_tracer, current_line = get_variable_value('emf_at_tracer',
                                                     current_line, content, int)
    hydro_at_tracer, current_line = get_variable_value('hydro_at_tracer',
                                                       current_line, content, int)
    dump_traj_directly, current_line = get_variable_value('dump_traj_directly',
                                                          current_line, content, int)
    num_tracer_fields_add, current_line = get_variable_value('num_tracer_fields_add',
                                                             current_line, content, int)
    emax_band, current_line = get_variable_value('emax_band', current_line, content)
    emin_band, current_line = get_variable_value('emin_band', current_line, content)
    nbands, current_line = get_variable_value('nbands', current_line, content, int)
    emax_spect, current_line = get_variable_value('emax_spect', current_line, content)
    emin_spect, current_line = get_variable_value('emin_spect', current_line, content)
    nbins_spect, current_line = get_variable_value('nbins', current_line, content, int)
    nx_zone, current_line = get_variable_value('nx_zone', current_line, content, int)
    ny_zone, current_line = get_variable_value('ny_zone', current_line, content, int)
    nz_zone, current_line = get_variable_value('nz_zone', current_line, content, int)
    stride_particle_dump, current_line = get_variable_value('stride_particle_dump',
                                                            current_line, content, int)

    pic_init_info = collections.namedtuple('pic_init_info',
                                           ['sigmae_c', 'ti_te', 'Ti', 'Te',
                                            'wpe_wce', 'mime',
                                            'lx_di', 'ly_di', 'lz_di',
                                            'nx', 'ny', 'nz',
                                            'courant', 'nproc',
                                            'nppc', 'b0', 'ne',
                                            'dtwpe', 'dtwce', 'dtwci',
                                            'dx_di', 'dy_di', 'dz_di',
                                            'x_di', 'y_di', 'z_di',
                                            'dx_rhoi', 'dx_rhoe', 'dx_debye',
                                            'n0',
                                            'restart_interval',
                                            'fields_interval_info',
                                            'ehydro_interval',
                                            'Hhydro_interval',
                                            'eparticle_interval',
                                            'Hparticle_interval',
                                            'energy_interval',
                                            'quota_check_interval',
                                            'particle_tracing',
                                            'tracer_interval',
                                            'tracer_pass1_interval',
                                            'tracer_pass2_interval',
                                            'ntracer', 'emf_at_tracer',
                                            'hydro_at_tracer',
                                            'dump_traj_directly',
                                            'num_tracer_fields_add',
                                            'emax_band', 'emin_band', 'nbands',
                                            'emax_spect', 'emin_spect', 'nbins_spect',
                                            'nx_zone', 'ny_zone', 'nz_zone',
                                            'stride_particle_dump',
                                            'vthi', 'vthe'])
    pic_info = pic_init_info(sigmae_c=sigmae_c, ti_te=ti_te, Te=te, Ti=ti,
                             wpe_wce=wpe_wce, mime=mime,
                             lx_di=lx, ly_di=ly, lz_di=lz,
                             nx=nx, ny=ny, nz=nz,
                             courant=courant, nproc=nproc,
                             nppc=nppc, b0=b0, ne=ne,
                             dtwpe=dtwpe, dtwce=dtwce, dtwci=dtwci,
                             dx_di=dxdi, dy_di=dydi, dz_di=dzdi, n0=n0,
                             x_di=x, y_di=y, z_di=z,
                             dx_rhoi=dx_rhoi, dx_rhoe=dx_rhoe, dx_debye=dx_debye,
                             restart_interval=restart_interval,
                             fields_interval_info=fields_interval_info,
                             ehydro_interval=ehydro_interval,
                             Hhydro_interval=Hhydro_interval,
                             eparticle_interval=eparticle_interval,
                             Hparticle_interval=Hparticle_interval,
                             energy_interval=energy_interval,
                             quota_check_interval=quota_check_interval,
                             particle_tracing=particle_tracing,
                             tracer_interval=tracer_interval,
                             tracer_pass1_interval=tracer_pass1_interval,
                             tracer_pass2_interval=tracer_pass2_interval,
                             ntracer=ntracer,
                             emf_at_tracer=emf_at_tracer,
                             hydro_at_tracer=hydro_at_tracer,
                             dump_traj_directly=dump_traj_directly,
                             num_tracer_fields_add=num_tracer_fields_add,
                             emax_band=emax_band, emin_band=emin_band, nbands=nbands,
                             emax_spect=emax_spect, emin_spect=emin_spect,
                             nbins_spect=nbins_spect,
                             nx_zone=nx_zone, ny_zone=ny_zone, nz_zone=nz_zone,
                             stride_particle_dump=stride_particle_dump,
                             vthi=vthi, vthe=vthe)
    return pic_info


def get_variable_value(variable_name, current_line, content,
                       data_type=float, match_name=False):
    """
    Get the value of one variable from the content of the information file.

    Args:
        variable_name: the variable name.
        current_line: current line number.
        content: the content of the information file.
        data_type: data type (float, int, ...)
        match_name: whether to match the variable name
    Returns:
        variable_value: the value of the variable.
        line_number: current line number after the operations.
    """
    line_number = current_line
    nlines = len(content)
    single_line = content[line_number]
    cond1 = not variable_name in single_line
    cond2 = line_number < nlines
    if match_name and (not cond1):
        if "=" in single_line:
            line_splits = single_line.split("=")
        elif ":" in single_line:
            line_splits = single_line.split(":")
        if line_splits[0].strip() != variable_name:
            cond1 = True
    while cond1 and cond2:
        line_number += 1
        cond2 = line_number < nlines
        if cond2:
            single_line = content[line_number]
            cond1 = not variable_name in single_line
        else:
            cond1 = True
        if match_name and (not cond1):
            if "=" in single_line:
                line_splits = single_line.split("=")
            elif ":" in single_line:
                line_splits = single_line.split(":")
            if line_splits[0].strip() != variable_name:
                cond1 = True
    if cond1 and (not cond2):  # no such variable_name
        variable_value = 0.0
        line_number = current_line
    else:
        single_line = content[line_number]
        if "=" in single_line:
            line_splits = single_line.split("=")
        elif ":" in single_line:
            line_splits = single_line.split(":")
        variable_value = float(line_splits[1])
    variable_value = data_type(variable_value)
    return (variable_value, line_number)


def get_variable_value_h(variable_name, content):
    """
    """
    line_number = 0
    for i, s in enumerate(content):
        if variable_name in s:
            single_line = content[i]
            line_splits = single_line.split("=")
            variable_value = float(line_splits[1])
            return (variable_value, line_number)

    raise StandardError(variable_name + ' is not found')


def get_pic_topology(base_directory, deck_file):
    """Get the PIC simulation topology

    Args:
        base_directory: the base directory for different runs.
        deck_file: simulation deck source file
    """
    fname = deck_file
    try:
        f = open(fname, 'r')
    except IOError:
        print('cannot open %s' % fname)
    else:
        content = f.readlines()
        f.close()
        nlines = len(content)
        current_line = 0
        while not 'double topology_x =' in content[current_line]:
            current_line += 1
        single_line = content[current_line]
        line_splits = single_line.split("=")
        word_splits = line_splits[1].split(";")
        topology_x = int(word_splits[0])
        current_line += 1
        single_line = content[current_line]
        line_splits = single_line.split("=")
        word_splits = line_splits[1].split(";")
        topology_y = int(word_splits[0])
        current_line += 1
        single_line = content[current_line]
        line_splits = single_line.split("=")
        word_splits = line_splits[1].split(";")
        topology_z = int(word_splits[0])
    pic_topology = collections.namedtuple(
        'pic_topology', ['topology_x', 'topology_y', 'topology_z'])
    pic_topo = pic_topology(
        topology_x=topology_x, topology_y=topology_y, topology_z=topology_z)
    return pic_topo


def save_pic_info_json():
    """Save pic_info for different runs as json format
    """
    if not os.path.isdir('../data/'):
        os.makedirs('../data/')
    dir = '../data/pic_info/'
    if not os.path.isdir(dir):
        os.makedirs(dir)

    # base_dirs, run_names = ApJ_long_paper_runs()
    # base_dirs, run_names = guide_field_runs()
    # base_dirs, run_names = high_sigma_runs()
    base_dirs, run_names = shock_sheet_runs()
    for base_dir, run_name in zip(base_dirs, run_names):
        pic_info = get_pic_info(base_dir)
        pic_info_json = data_to_json(pic_info)
        fname = dir + 'pic_info_' + run_name + '.json'
        with open(fname, 'w') as f:
            json.dump(pic_info_json, f)


def list_pic_info_dir(filepath):
    """List all of the json files of the PIC information

    Args:
        filepath: the filepath saving the json files.

    Returns:
        pic_infos: the list of filenames.
    """
    pic_infos = [f for f in listdir(filepath) if isfile(join(filepath, f))]
    return pic_infos


if __name__ == "__main__":
    cmdargs = sys.argv
    if (len(cmdargs) > 2):
        base_directory = cmdargs[1]
        run_name = cmdargs[2]
    else:
        base_directory = '/net/scratch2/guofan/sigma1-mime25-beta001-average/'
        run_name = 'sigma1-mime25-beta001-average'
    pic_info = get_pic_info(base_directory, run_name)
    pic_info_json = data_to_json(pic_info)
    fname = '../data/pic_info/pic_info_' + run_name + '.json'
    with open(fname, 'w') as f:
        json.dump(pic_info_json, f)
    # save_pic_info_json()
    # list_pic_info_dir('../data/pic_info/')
