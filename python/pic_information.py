"""
Read particle-in-cell (VPIC) simulation information.
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
    ntf = get_fields_frames(base_directory)
    energy_interval = pic_initial_info.energy_interval
    deck_file = get_main_source_filename(base_directory)
    fields_interval, particle_interval, trace_interval = \
            get_output_intervals(dtwpe, dtwce, dtwpi, dtwci, base_directory, deck_file)
    dt_fields = fields_interval * dtwci
    dt_particles = particle_interval * dtwci
    ntp = ntf / (particle_interval / fields_interval)
    tparticles = np.arange(ntp) * dt_particles
    tfields = np.arange(ntf) * dt_fields
    dt_energy = energy_interval * dtwci
    dte_wpe = dt_energy * dtwpe / dtwci
    pic_ene = read_pic_energies(dt_energy, dte_wpe, base_directory)
    pic_times = collections.namedtuple("pic_times", [
        'ntf', 'dt_fields', 'tfields', 'ntp', 'dt_particles', 'tparticles',
        'dt_energy', 'fields_interval', 'particle_interval', 'trace_interval'
    ])
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
    pic_ene = pic_energies(
        nte=nte,
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


def get_fields_frames(base_directory):
    """Get the total number of time frames for fields.

    Args:
        base_directory: the base directory for different runs.
    Returns:
        ntf: the total number of output time frames for fields.
    """
    pic_initial_info = read_pic_info(base_directory)
    nx = pic_initial_info.nx
    ny = pic_initial_info.ny
    nz = pic_initial_info.nz
    fname_fields = base_directory + '/fields/T.1'
    fname_ex = base_directory + '/data/ex.gda'
    fname_Ex = base_directory + '/data/Ex.gda'
    fname_bx = base_directory + '/data/bx_0.gda'
    fname_Bx = base_directory + '/data/Bx_0.gda'
    if os.path.isfile(fname_bx) or os.path.isfile(fname_Bx):
        current_time = 1
        is_exist = False
        while (not is_exist):
            current_time += 1
            fname1 = base_directory + '/data/bx_' + str(current_time) + '.gda'
            fname2 = base_directory + '/data/Bx_' + str(current_time) + '.gda'
            is_exist = os.path.isfile(fname1) or os.path.isfile(fname2)
        fields_interval = current_time
        ntf = 1
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
        current_time = 1
        is_exist = False
        while (not is_exist):
            current_time += 1
            fname = base_directory + '/fields/T.' + str(current_time)
            is_exist = os.path.isdir(fname)
        fields_interval = current_time
        ntf = 1
        is_exist = True
        while (is_exist):
            ntf += 1
            current_time += fields_interval
            fname = base_directory + '/fields/T.' + str(current_time)
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
    mime, current_line = get_variable_value('mi/me', current_line, content)
    lx, current_line = get_variable_value('Lx/di', current_line, content)
    ly, current_line = get_variable_value('Ly/di', current_line, content)
    lz, current_line = get_variable_value('Lz/di', current_line, content)
    nx, current_line = get_variable_value('nx', current_line, content)
    ny, current_line = get_variable_value('ny', current_line, content)
    nz, current_line = get_variable_value('nz', current_line, content)
    nx = int(nx)
    ny = int(ny)
    nz = int(nz)
    nppc, current_line = get_variable_value('nppc', current_line, content)
    b0, current_line = get_variable_value('b0', current_line, content)
    dtwpe, current_line = get_variable_value('dt*wpe', current_line, content)
    try:
        dtwce, current_line = get_variable_value_h('dt*wce', content)
    except:
        dtwce = dtwpe * b0
    try:
        dtwci, current_line = get_variable_value_h('dt*wci', content)
    except:
        dtwci = dtwce / mime
    while not 'energies_interval' in content[current_line]:
        current_line += 1
    single_line = content[current_line]
    line_splits = single_line.split(":")
    energy_interval = float(line_splits[1])
    dxde, current_line = get_variable_value('dx/de', current_line, content)
    dyde, current_line = get_variable_value('dy/de', current_line, content)
    dzde, current_line = get_variable_value('dz/de', current_line, content)
    dxdi = dxde / math.sqrt(mime)
    dydi = dyde / math.sqrt(mime)
    dzdi = dzde / math.sqrt(mime)
    x = np.arange(nx) * dxdi
    y = (np.arange(ny) - ny / 2.0 + 0.5) * dydi
    z = (np.arange(nz) - nz / 2.0 + 0.5) * dzdi
    if any('vthi/c' in s for s in content):
        vthi, current_line = get_variable_value('vthi/c', current_line,
                                                content)
        vthe, current_line = get_variable_value('vthe/c', current_line,
                                                content)
    else:
        vthe = 1.0
        vthi = 1.0

    pic_init_info = collections.namedtuple('pic_init_info', [
        'mime', 'lx_di', 'ly_di', 'lz_di', 'nx', 'ny', 'nz', 'dx_di', 'dy_di',
        'dz_di', 'x_di', 'y_di', 'z_di', 'nppc', 'b0', 'dtwpe', 'dtwce',
        'dtwci', 'energy_interval', 'vthi', 'vthe'
    ])
    pic_info = pic_init_info(
        mime=mime,
        lx_di=lx,
        ly_di=ly,
        lz_di=lz,
        nx=nx,
        ny=ny,
        nz=nz,
        dx_di=dxdi,
        dy_di=dydi,
        dz_di=dzdi,
        x_di=x,
        y_di=y,
        z_di=z,
        nppc=nppc,
        b0=b0,
        dtwpe=dtwpe,
        dtwce=dtwce,
        dtwci=dtwci,
        energy_interval=energy_interval,
        vthi=vthi,
        vthe=vthe)
    return pic_info


def get_variable_value(variable_name, current_line, content):
    """
    Get the value of one variable from the content of the information file.

    Args:
        variable_name: the variable name.
        current_line: current line number.
        content: the content of the information file.
    Returns:
        variable_value: the value of the variable.
        line_number: current line number after the operations.
    """
    line_number = current_line
    while not variable_name in content[line_number]:
        line_number += 1
    single_line = content[line_number]
    line_splits = single_line.split("=")
    variable_value = float(line_splits[1])
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
    # base_directory = '../../'
    # pic_info = get_pic_info(base_directory)
    # pic_info_json = data_to_json(pic_info)
    # run_name = 'nersc_large'
    # base_directory = '/net/scratch2/guofan/sigma1-mime25-beta0001/'
    # run_name = 'mime25_beta0001'
    # base_directory = '/net/scratch2/guofan/sigma1-mime25-beta0002-1127'
    # run_name = 'sigma1-mime25-beta0002'
    # base_directory = '/net/scratch2/guofan/for_Senbei/2D-90-Mach4-sheet6-2'
    # run_name = '2D-90-Mach4-sheet6-2'
    # base_directory = '/net/scratch2/guofan/for_Senbei/2D-90-Mach4-sheet6-3'
    # run_name = '2D-90-Mach4-sheet6-3'
    # base_directory = '/net/scratch3/xiaocanli/herts/tether_potential_tests/v200_b0_wce'
    # run_name = 'v200_b0_wce'
    # base_directory = '/net/scratch1/guofan/Project2017/low-beta/sigma1-mime25-beta0002/'
    # run_name = 'sigma1-mime25-beta0002-fan'
    # base_directory = '/net/scratch2/guofan/for_Xiaocan/sigma100-lx300/'
    # run_name = 'sigma100-lx300'
    pic_info = get_pic_info(base_directory, run_name)
    pic_info_json = data_to_json(pic_info)
    fname = '../data/pic_info/pic_info_' + run_name + '.json'
    with open(fname, 'w') as f:
        json.dump(pic_info_json, f)
    # save_pic_info_json()
    # list_pic_info_dir('../data/pic_info/')
