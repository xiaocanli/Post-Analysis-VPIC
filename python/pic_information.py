"""
Read particle-in-cell (VPIC) simulation information.
"""
import numpy as np
import math
import os.path
import struct
import collections

def get_pic_info(base_directory):
    """Get particle-in-cell simulation information.

    Args:
        base_directory: the base directory for different runs.
    """
    pic_initial_info = read_pic_info(base_directory)
    ntf = get_fields_frames(base_directory)
    energy_interval = pic_initial_info.energy_interval
    dtwci = pic_initial_info.dtwci
    fields_interval, particle_interval = \
            get_output_intervals(dtwci, base_directory)
    dt_fields = fields_interval * dtwci
    dt_particles = particle_interval * dtwci
    ntp = ntf / (particle_interval/fields_interval)
    tparticles = np.arange(ntp) * dt_particles
    tfields = np.arange(ntf) * dt_fields
    dt_energy = energy_interval * dtwci
    dtwpe = pic_initial_info.dtwpe
    dte_wpe = dt_energy * dtwpe / dtwci
    pic_ene = read_pic_energies(dt_energy, dte_wpe, base_directory)
    pic_times = collections.namedtuple("pic_times", 
            ['ntf', 'dt_fields', 'tfields', 'ntp', 'dt_particles', 
                'tparticles', 'dt_energy', 'fields_interval', 'particle_interval'])
    pic_times_info = pic_times(ntf=ntf, dt_fields=dt_fields,
            dt_particles=dt_particles, tfields=tfields, dt_energy=dt_energy,
            ntp=ntp, tparticles=tparticles, fields_interval=fields_interval,
            particle_interval=particle_interval)
    pic_topology = get_pic_topology(base_directory)
    pic_information = collections.namedtuple("pic_info", 
            pic_initial_info._fields + pic_times_info._fields +
            pic_ene._fields + pic_topology._fields)
    pic_info = pic_information(*(pic_initial_info + pic_times_info +
        pic_ene + pic_topology))
    return pic_info


def read_pic_energies(dte_wci, dte_wpe, base_directory):
    """Read particle-in-cell simulation energies.

    Args:
        dte_wci: the time interval for energies diagnostics (in 1/wci).
        dte_wpe: the time interval for energies diagnostics (in 1/wpe).
        base_directory: the base directory for different runs.
    """
    fname = base_directory + '/energies'
    try:
        f = open(fname, 'r')
    except IOError:
        print 'cannot open ', fname
        return
    else:
        content = np.genfromtxt(f, skip_header=3)
        f.close()
        nte, nvar = content.shape
        tenergy = np.arange(nte) * dte_wci
        ene_ex = content[:,1]
        ene_ey = content[:,2]
        ene_ez = content[:,3]
        ene_bx = content[:,4]
        ene_by = content[:,5]
        ene_bz = content[:,6]
        kene_i = content[:,7] # kinetic energy for ions
        kene_e = content[:,8]
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
        pic_energies = collections.namedtuple('pic_energies',
                ['nte', 'tenergy', 'ene_ex', 'ene_ey', 'ene_ez', 'ene_bx',
                    'ene_by', 'ene_bz','kene_i', 'kene_e', 'ene_electric',
                    'ene_magnetic', 'dene_ex', 'dene_ey', 'dene_ez', 'dene_bx',
                    'dene_by', 'dene_bz','dkene_i', 'dkene_e', 'dene_electric',
                    'dene_magnetic'])
        pic_ene = pic_energies(nte=nte, tenergy=tenergy, ene_ex=ene_ex,
                ene_ey=ene_ey, ene_ez=ene_ez, ene_bx=ene_bx, ene_by=ene_by,
                ene_bz=ene_bz, kene_i=kene_i, kene_e=kene_e,
                ene_electric=ene_electric, ene_magnetic=ene_magnetic, 
                dene_ex=dene_ex, dene_ey=dene_ey, dene_ez=dene_ez,
                dene_bx=dene_bx, dene_by=dene_by, dene_bz=dene_bz,
                dkene_i=dkene_i, dkene_e=dkene_e,
                dene_electric=dene_electric, dene_magnetic=dene_magnetic)
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
    fname = base_directory + '/data/ex.gda'
    fname_fields = base_directory + '/fields/T.1'
    fname_bx = base_directory + '/data/bx_0.gda'
    if (os.path.isfile(fname_bx)):
        current_time = 1
        is_exist = False
        while (not is_exist):
            current_time += 1
            fname = base_directory + '/data/bx_' + str(current_time) + '.gda'
            is_exist = os.path.isfile(fname)
        fields_interval = current_time
        ntf = 1
        is_exist = True
        while (is_exist):
            ntf += 1
            current_time += fields_interval
            fname = base_directory + '/data/bx_' + str(current_time) + '.gda'
            is_exist = os.path.isfile(fname)
    elif (os.path.isfile(fname)):
        file_size = os.path.getsize(fname)
        ntf = int(file_size/(nx*ny*nz*4))
    elif (os.path.isdir(fname_fields)):
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
        print 'Cannot find the files to calculate the total frames of fields.'
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
    except IOError:
        print 'cannot open ', fname
    else:
        content = f.readlines()
        f.close()
        nlines = len(content)
        current_line = 0
        while not 'vpic' in content[current_line]: current_line += 1
        single_line = content[current_line]
        line_splits = single_line.split(".op")
        word_splits = line_splits[1].split(" ")

    filename = word_splits[1]
    fname = base_directory + '/' + filename[:-1]
    return fname


def get_output_intervals(dtwci, base_directory):
    """
    Get output intervals from the main configuration file for current PIC
    simulation.
    
    Args:
        dtwci: the time step in 1/wci.
        base_directory: the base directory for different runs.
    """
    fname = get_main_source_filename(base_directory)
    try:
        f = open(fname, 'r')
    except IOError:
        print 'cannot open ', fname
    else:
        content = f.readlines()
        f.close()
        nlines = len(content)
        current_line = 0
        while not 'int interval' in content[current_line]: current_line += 1
        single_line = content[current_line]
        line_splits = single_line.split("(")
        word_splits = line_splits[1].split("/")
        interval = float(word_splits[0])
        interval = int(interval/dtwci)
        while not 'int eparticle_interval' in content[current_line]: 
            current_line += 1
        single_line = content[current_line]
        line_splits = single_line.split("=")
        word_splits = line_splits[1].split("*")
        particle_interval = float(word_splits[0]) * interval
    fields_interval = interval
    particle_interval = int(particle_interval)
    return (fields_interval, particle_interval)

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
    dtwce, current_line = get_variable_value('dt*wce', current_line, content)
    dtwci, current_line = get_variable_value('dt*wci', current_line, content)
    while not 'energies_interval' in content[current_line]: current_line += 1
    single_line = content[current_line]
    line_splits = single_line.split(":")
    energy_interval = float(line_splits[1])
    dxde, current_line = get_variable_value('dx/de', current_line, content)
    dyde, current_line = get_variable_value('dy/de', current_line, content)
    dzde, current_line = get_variable_value('dz/de', current_line, content)
    dxdi = dxde / math.sqrt(mime)
    dydi = dyde / math.sqrt(mime)
    dzdi = dzde / math.sqrt(mime)
    x = np.arange(nx)*dxdi
    y = (np.arange(ny)-ny/2.0+0.5)*dydi
    z = (np.arange(nz)-nz/2.0+0.5)*dzdi
    vthi, current_line = get_variable_value('vthi/c', current_line, content)
    vthe, current_line = get_variable_value('vthe/c', current_line, content)

    pic_information = collections.namedtuple('pic_info',
            ['mime', 'lx_di', 'ly_di', 'lz_di', 'nx', 'ny', 'nz',
                'dx_di', 'dy_di', 'dz_di', 'x_di', 'y_di', 'z_di', 'nppc', 'b0',
                'dtwpe', 'dtwce', 'dtwci', 'energy_interval', 'vthi', 'vthe'])
    pic_info = pic_information(mime=mime, lx_di=lx, ly_di=ly, lz_di=lz,
            nx=nx, ny=ny, nz=nz, dx_di=dxdi, dy_di=dydi, dz_di=dzdi, 
            x_di=x, y_di=y, z_di=z, nppc=nppc, b0=b0, dtwpe=dtwpe, dtwce=dtwce,
            dtwci=dtwci, energy_interval=energy_interval, vthi=vthi, vthe=vthe)
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
    while not variable_name in content[line_number]: line_number += 1
    single_line = content[line_number]
    line_splits = single_line.split("=")
    variable_value = float(line_splits[1])
    return (variable_value, line_number)


def get_pic_topology(base_directory):
    """Get the PIC simulation topology

    Args:
        base_directory: the base directory for different runs.
    """
    fname = get_main_source_filename(base_directory)
    try:
        f = open(fname, 'r')
    except IOError:
        print 'cannot open ', fname
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
    pic_topology = collections.namedtuple('pic_topology',
            ['topology_x', 'topology_y', 'topology_z'])
    pic_topo = pic_topology(topology_x = topology_x,
            topology_y = topology_y, topology_z = topology_z)
    return pic_topo


if __name__ == "__main__":
    base_directory = '../../'
    pic_info = get_pic_info(base_directory)
