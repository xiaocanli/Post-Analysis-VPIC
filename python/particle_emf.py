"""
Analysis procedures for particle tracking
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LogNorm
from matplotlib import rc
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import math
import os.path
import struct
import collections
import pic_information
from scipy.interpolate import interp1d
import h5py
from shell_functions import *
from energy_conversion import *

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

font = {'family' : 'serif',
        #'color'  : 'darkred',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 24,
        }

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


def interp_data(sz_new, sz_old, ptl, pic_info, interp_kind='linear'):
    """Particle data interpolation

    Args:
        sz_new: new data size
        sz_old: old data size
        ptl: particle data in a dictionary
        pic_info: PIC simulation information
        interp_kind: interpolate kind
    """
    dtwci = pic_info.dtwci
    dtwpe = pic_info.dtwpe
    dt = pic_info.dt_fields * dtwpe / dtwci
    t = ptl['t']
    tnew = np.linspace(t[0], t[-1], sz_new)
    dt_new = sz_old * dt / sz_new

    for key in ptl:
        if key != 't':
            f = interp1d(t, ptl[key], kind=interp_kind)
            ptl[key] = f(tnew)
    ptl['t'] = tnew

    return ptl


def particle_energy(iptl, particle_tags, pic_info, stride, odir, fh):
    """Plotting particle energy change

    Args:
        iptl: particle index
        particles_tags: all the particle tags
        pic_info: PIC simulation information
        stride: the stride interval along the time
        odir: output directory for saving figures
        fh: HDF5 file handler
    """
    dtwci = pic_info.dtwci
    dtwpe = pic_info.dtwpe
    dt = pic_info.dt_fields * dtwpe / dtwci
    ptl, sz = read_particle_data(iptl, particle_tags, pic_info, fh)
    t = ptl['t']
    sz_new = sz
    tnew = np.linspace(t[0], t[-1], sz_new)
    dt_new = sz * dt / sz_new
    ptl = interp_data(sz_new, sz, ptl, pic_info, interp_kind='linear')
    ibtot2 = 1.0/(ptl['Bx']**2 + ptl['By']**2 + ptl['Bz']**2)
    edotb = ptl['Ex']*ptl['Bx'] + ptl['Ey']*ptl['By'] + ptl['Ez']*ptl['Bz']
    Eparax = edotb * ibtot2 * ptl['Bx']
    Eparay = edotb * ibtot2 * ptl['By']
    Eparaz = edotb * ibtot2 * ptl['Bz']
    Eperpx = ptl['Ex'] - Eparax
    Eperpy = ptl['Ey'] - Eparay
    Eperpz = ptl['Ez'] - Eparaz
    gama = ptl['gamma']
    jdote_para = -(ptl['Ux']*Eparax + ptl['Uy']*Eparay + \
                   ptl['Uz']*Eparaz) / gama
    jdote_perp = -(ptl['Ux']*Eperpx + ptl['Uy']*Eperpy + \
                   ptl['Uz']*Eperpz) / gama
    jdote_tot = -(ptl['Ux']*ptl['Ex'] + ptl['Uy']*ptl['Ey'] + \
                  ptl['Uz']*ptl['Ez']) / gama

    if 'Vx' in ptl:
        Einx = ptl['Vz']*ptl['By'] - ptl['Vy']*ptl['Bz']
        Einy = ptl['Vx']*ptl['Bz'] - ptl['Vz']*ptl['Bx']
        Einz = ptl['Vy']*ptl['Bx'] - ptl['Vx']*ptl['By']
        jdote_in = -(ptl['Ux']*Einx + ptl['Uy']*Einy + \
                     ptl['Uz']*Einz) / gama
        jdote_in_cum = np.cumsum(jdote_in[::stride]) * dt_new * stride

    jdote_para_cum = np.cumsum(jdote_para[::stride]) * dt_new * stride
    jdote_perp_cum = np.cumsum(jdote_perp[::stride]) * dt_new * stride
    jdote_tot_cum = np.cumsum(jdote_tot[::stride]) * dt_new * stride

    fig = plt.figure(figsize=[7, 5])
    xs, ys = 0.13, 0.13
    w1, h1 = 0.8, 0.8
    ax = fig.add_axes([xs, ys, w1, h1])
    ax.plot(tnew[::stride], jdote_tot_cum, linewidth=2, color='r', 
            label=r'$\int q\boldsymbol{v}\cdot\boldsymbol{E}$')
    ax.plot(tnew, gama-gama[0], linewidth=2, color='k',
            label=r'$\gamma-\gamma_0$')
    ax.plot(tnew[::stride], jdote_para_cum, linewidth=2, color='b',
            label=r'$\int q\boldsymbol{v}_\parallel\cdot\boldsymbol{E}$')
    ax.plot(tnew[::stride], jdote_perp_cum, linewidth=2, color='r',
            label=r'$\int q\boldsymbol{v}_\perp\cdot\boldsymbol{E}$')
    if 'Vx' in ptl:
        ax.plot(tnew[::stride], jdote_in_cum, linewidth=2, color='m',
                label=r'$\int q\boldsymbol{v}_\perp\cdot\boldsymbol{E}_{vB}$')
    leg = ax.legend(loc=2, prop={'size':20}, ncol=1,
            shadow=False, fancybox=False, frameon=False)
    ax.tick_params(labelsize=16)
    ax.set_xlabel(r'$t\omega_{pe}$', fontdict=font, fontsize=24)
    # fname = odir + 'ptl_ene_' + str(iptl) + '_' + str(stride) + '.eps'
    # plt.savefig(fname)
    plt.show()
    # plt.close()


def adjust_pos(pos, length):
    """Adjust position for periodic boundary conditions.

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
        for i in range(nc-1):
            pos_b[crossings[i]+1 : crossings[i+1]+1] += offsets[i]
        pos_b[crossings[nc-1]+1:] += offsets[nc-1]
    return pos_b


def save_new_data(iptl, particle_tags, pic_info, fh_in, fh_out):
    """Save adjusted particle data into file

    Args:
        iptl: particle index
        particles_tags: all the particle tags
        pic_info: PIC simulation information
        fh_in: HDF5 file handler for the input file
        fh_out: HDF5 file handler for the output file
    """
    ptl, sz = read_particle_data(iptl, particle_tags, pic_info, fh)
    ptl['dX'] = adjust_pos(ptl['dX'], pic_info.lx_di)
    ptl['dY'] = adjust_pos(ptl['dY'], pic_info.ly_di)
    ptl['dY'] = adjust_pos(ptl['dY'], pic_info.lz_di)
    sz_new = sz * 10
    ptl = interp_data(sz_new, sz, ptl, pic_info, interp_kind='linear')
    grp = fh_out.create_group(particle_tags[iptl])
    for key in ptl:
        grp.create_dataset(key, (sz_new, ), data=ptl[key])


def plot_particle_energy_conversion(fh, particle_tags, pic_info):
    """Plot particle energy conversion w.r.t to local magnetic field
    """
    odir = '../img/ptl_ene/'
    mkdir_p(odir)

    for i in range(1):
        stride = 2**i
        for iptl in range(1):
            print(iptl)
            particle_energy(iptl, particle_tags, pic_info, stride, odir, fh)
    

def save_shifted_trajectory(fh, filepath, particle_tags, pic_info):
    """Save shifted particle trajectory at boundaries
    """
    fname_out = filepath + 'electrons_interp.h5p'
    with h5py.File(fname_out, 'w') as fh_out:
        for iptl in range(1):
            print(iptl)
            save_new_data(iptl, particle_tags, pic_info, fh, fh_out)


def transfer_to_h5part(particle_tags, pic_info, fh, filepath):
    """Transfer current HDF5 file to H5Part format, in which all particles
    at the same time step are stored in the same time step
    """
    nptl = len(particle_tags)
    ptl, sz = read_particle_data(0, particle_tags, pic_info, fh)
    Ux = np.zeros(sz * nptl, dtype = ptl['Ux'].dtype)
    Uy = np.zeros(sz * nptl, dtype = ptl['Uy'].dtype)
    Uz = np.zeros(sz * nptl, dtype = ptl['Uz'].dtype)
    dX = np.zeros(sz * nptl, dtype = ptl['dX'].dtype)
    dY = np.zeros(sz * nptl, dtype = ptl['dY'].dtype)
    dZ = np.zeros(sz * nptl, dtype = ptl['dZ'].dtype)
    i = np.zeros(sz * nptl, dtype = ptl['i'].dtype)
    q = np.zeros(sz * nptl, dtype = ptl['q'].dtype)
    gamma = np.zeros(sz * nptl, dtype = ptl['gamma'].dtype)
    for iptl in range(nptl):
        print iptl
        ptl, sz = read_particle_data(iptl, particle_tags, pic_info, fh)
        Ux[iptl::nptl] = ptl['Ux']
        Uy[iptl::nptl] = ptl['Uy']
        Uz[iptl::nptl] = ptl['Uz']
        dX[iptl::nptl] = ptl['dX']
        dY[iptl::nptl] = ptl['dY']
        dZ[iptl::nptl] = ptl['dZ']
        i[iptl::nptl] = ptl['i']
        q[iptl::nptl] = ptl['q']
        gamma[iptl::nptl] = ptl['gamma']

    nx, ny, nz = pic_info.nx, pic_info.ny, pic_info.nz
    lx, ly, lz = pic_info.lx_di, pic_info.ly_di, pic_info.lz_di
    dX *= nx / lx * 0.5
    dY *= ny / ly * 0.5
    dZ *= nz / lz * 0.5
    dY += ny * 0.25
    dZ += nz * 0.25

    tinterval = 13
    tratio = 8
    fname = filepath + 'electrons.h5part'
    with h5py.File(fname, 'w') as fh_out:
        for t in range(0, sz*tinterval):
            print t
            ct = t / tinterval
            # grp = fh_out.create_group('Step#'+str(ct))
            grp = fh_out.create_group('Step#'+str(t))
            index = range(ct*nptl, (ct+1)*nptl)
            grp.create_dataset('Ux', (nptl, ), data=Ux[index])
            grp.create_dataset('Uy', (nptl, ), data=Uy[index])
            grp.create_dataset('Uz', (nptl, ), data=Uz[index])
            grp.create_dataset('dX', (nptl, ), data=dX[index])
            grp.create_dataset('dY', (nptl, ), data=dY[index])
            grp.create_dataset('dZ', (nptl, ), data=dZ[index])
            grp.create_dataset('i', (nptl, ), data=i[index])
            grp.create_dataset('q', (nptl, ), data=q[index])
            grp.create_dataset('gamma', (nptl, ), data=gamma[index])


if __name__ == "__main__":
    filepath = '/net/scratch3/xiaocanli/open3d-full/'
    # pic_info = pic_information.get_pic_info(filepath)
    run_name = 'nersc_large'
    picinfo_fname = '../data/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    filepath += 'pic_analysis/vpic-sorter/data/'
    species = 'e'
    if species == 'i':
        fname = filepath + 'ions.h5p'
    else:
        fname = filepath + 'electrons_2.h5p'
    with h5py.File(fname, 'r') as fh:
        particle_tags = fh.keys()
        # save_shifted_trajectory(fh, filepath, particle_tags, pic_info)
        transfer_to_h5part(particle_tags, pic_info, fh, filepath)
