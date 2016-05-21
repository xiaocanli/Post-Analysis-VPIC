"""
Analysis procedures for particle energy spectrum.
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
import h5py

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

font = {'family' : 'serif',
        #'color'  : 'darkred',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 24,
        }

def read_single_dataset(sz, dname, group):
    fdata = np.zeros(sz)
    dset = group[dname]
    dset.read_direct(fdata)
    return fdata


def read_particle_data(iptl, particle_tags, pic_info):
    """Read particle data
    """
    group = file[particle_tags[iptl]]
    dset_ux = group['Ux']
    sz, = dset_ux.shape
    ux = read_single_dataset(sz, 'Ux', group)
    uy = read_single_dataset(sz, 'Uy', group)
    uz = read_single_dataset(sz, 'Uz', group)
    Bx = read_single_dataset(sz, 'Bx', group)
    By = read_single_dataset(sz, 'By', group)
    Bz = read_single_dataset(sz, 'Bz', group)
    Ex = read_single_dataset(sz, 'Ex', group)
    Ey = read_single_dataset(sz, 'Ey', group)
    Ez = read_single_dataset(sz, 'Ez', group)
    q = read_single_dataset(sz, 'q', group)
    # vx = read_single_dataset(sz, 'Vx', group)
    # vy = read_single_dataset(sz, 'Vy', group)
    # vz = read_single_dataset(sz, 'Vz', group)
    return (sz, ux, uy, uz, Bx, By, Bz, Ex, Ey, Ez, q)


def particle_energy(iptl, particle_tags, pic_info, stride, odir):
    """Plot particle energy change
    """
    sz, ux, uy, uz, Bx, By, Bz, Ex, Ey, Ez, q = \
            read_particle_data(iptl, particle_tags, pic_info)

    gama = np.sqrt(ux*ux + uy*uy + uz*uz + 1)

    ibtot2 = 1.0/(Bx*Bx + By*By + Bz*Bz)
    edotb = Ex*Bx + Ey*By + Ez*Bz
    Eparax = edotb * ibtot2 * Bx
    Eparay = edotb * ibtot2 * By
    Eparaz = edotb * ibtot2 * Bz
    Eperpx = Ex - Eparax
    Eperpy = Ey - Eparay
    Eperpz = Ez - Eparaz
    jdote_para = -(ux*Eparax + uy*Eparay + uz*Eparaz) / gama
    jdote_perp = -(ux*Eperpx + uy*Eperpy + uz*Eperpz) / gama
    jdote_tot = -(ux*Ex + uy*Ey + uz*Ez) / gama

    # Einx = vz*By - vy*Bz
    # Einy = vx*Bz - vz*Bx
    # Einz = vy*Bx - vx*By
    # jdote_in = -(ux*Einx + uy*Einy + uz*Einz) / gama

    dtwci = pic_info.dtwci
    dtwpe = pic_info.dtwpe
    # dt = pic_info.dt_fields * dtwpe / dtwci
    dt = dtwpe
    jdote_para_cum = np.cumsum(jdote_para[::stride]) * dt * stride
    jdote_perp_cum = np.cumsum(jdote_perp[::stride]) * dt * stride
    # jdote_in_cum = np.cumsum(jdote_in[::stride]) * dt * stride
    # jdote_tot_cum = jdote_para_cum + jdote_perp_cum
    jdote_tot_cum = np.cumsum(jdote_tot[::stride]) * dt * stride
    tfields = np.arange(sz) * dt

    fig = plt.figure(figsize=[7, 5])
    xs, ys = 0.13, 0.13
    w1, h1 = 0.8, 0.8
    ax = fig.add_axes([xs, ys, w1, h1])
    ax.plot(tfields[::stride], jdote_tot_cum, linewidth=2, color='r', 
            label=r'$\int q\boldsymbol{v}\cdot\boldsymbol{E}$')
    ax.plot(tfields, gama - gama[0], linewidth=2, color='k', label=r'$\gamma-1$')
    ax.plot(tfields[::stride], jdote_para_cum, linewidth=2, color='g',
            label=r'$\int q\boldsymbol{v}_\parallel\cdot\boldsymbol{E}$')
    ax.plot(tfields[::stride], jdote_perp_cum, linewidth=2, color='b',
            label=r'$\int q\boldsymbol{v}_\perp\cdot\boldsymbol{E}$')
    # ax.plot(tfields[::stride], jdote_in_cum, linewidth=2, color='m',
    #         label=r'$\int q\boldsymbol{v}_\perp\cdot\boldsymbol{E}_{vB}$')
    leg = ax.legend(loc=2, prop={'size':20}, ncol=1,
            shadow=False, fancybox=False, frameon=False)
    ax.tick_params(labelsize=16)
    ax.set_xlabel(r'$t\omega_{pe}$', fontdict=font, fontsize=24)
    # fname = odir + 'ptl_ene_' + str(iptl) + '_' + str(stride) + '.eps'
    # plt.savefig(fname)
    plt.show()
    # plt.close()

def average_over_time(navg, fdata, sz):
    """Average fdata over navg time points
    """
    return np.mean(fdata.reshape(-1, navg), axis=1)


def energy_gain(iptl, particle_tags, pic_info, navg, odir):
    """Plot particle energy gain due to different current

    Args:
        iptl: tags of particles to be plotted
        particles_tags: all the particle tags
        pic_info: simulation information
        navg: average over navg time points
        odir: output directory
    """
    sz, ux, uy, uz, Bx, By, Bz, Ex, Ey, Ez, q = \
            read_particle_data(iptl, particle_tags, pic_info)
    sz1 = (sz / navg) * navg
    ux = average_over_time(navg, ux[:sz1], sz1)
    uy = average_over_time(navg, uy[:sz1], sz1)
    uz = average_over_time(navg, uz[:sz1], sz1)
    Bx = average_over_time(navg, Bx[:sz1], sz1)
    By = average_over_time(navg, By[:sz1], sz1)
    Bz = average_over_time(navg, Bz[:sz1], sz1)
    Ex = average_over_time(navg, Ex[:sz1], sz1)
    Ey = average_over_time(navg, Ey[:sz1], sz1)
    Ez = average_over_time(navg, Ez[:sz1], sz1)

    gama = np.sqrt(ux*ux + uy*uy + uz*uz + 1)

    ibtot2 = 1.0/(Bx*Bx + By*By + Bz*Bz)
    edotb = Ex*Bx + Ey*By + Ez*Bz
    Eparax = edotb * ibtot2 * Bx
    Eparay = edotb * ibtot2 * By
    Eparaz = edotb * ibtot2 * Bz
    Eperpx = Ex - Eparax
    Eperpy = Ey - Eparay
    Eperpz = Ez - Eparaz
    jdote_para = -(ux*Eparax + uy*Eparay + uz*Eparaz) / gama
    jdote_perp = -(ux*Eperpx + uy*Eperpy + uz*Eperpz) / gama
    jdote_tot = -(ux*Ex + uy*Ey + uz*Ez) / gama

    dtwci = pic_info.dtwci
    dtwpe = pic_info.dtwpe
    dt = dtwpe * navg
    jdote_para_cum = np.cumsum(jdote_para) * dt
    jdote_perp_cum = np.cumsum(jdote_perp) * dt
    jdote_tot_cum = np.cumsum(jdote_tot) * dt
    tfields = np.arange(sz1/navg) * dt

    fig = plt.figure(figsize=[7, 5])
    xs, ys = 0.13, 0.13
    w1, h1 = 0.8, 0.8
    ax = fig.add_axes([xs, ys, w1, h1])
    ax.plot(tfields, jdote_tot_cum, linewidth=2, color='r', 
            label=r'$\int q\boldsymbol{v}\cdot\boldsymbol{E}$')
    ax.plot(tfields, gama - gama[0], linewidth=2, color='k', label=r'$\gamma-1$')
    ax.plot(tfields, jdote_para_cum, linewidth=2, color='g',
            label=r'$\int q\boldsymbol{v}_\parallel\cdot\boldsymbol{E}$')
    ax.plot(tfields, jdote_perp_cum, linewidth=2, color='b',
            label=r'$\int q\boldsymbol{v}_\perp\cdot\boldsymbol{E}$')
    leg = ax.legend(loc=2, prop={'size':20}, ncol=1,
            shadow=False, fancybox=False, frameon=False)
    ax.tick_params(labelsize=16)
    ax.set_xlabel(r'$t\omega_{pe}$', fontdict=font, fontsize=24)
    plt.show()


if __name__ == "__main__":
    pic_info = pic_information.get_pic_info('../../')
    filepath = '../../'
    filepath += 'pic_analysis/vpic-sorter/data/'
    species = 'e'
    if species == 'i':
        fname = filepath + 'ions.h5p'
    else:
        fname = filepath + 'electrons_2.h5p'
    file = h5py.File(fname, 'r')
    particle_tags = []
    for item in file:
        particle_tags.append(item)
    nptl = len(particle_tags)
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    odir = '../img/ptl_ene/'
    if not os.path.isdir(odir):
        os.makedirs(odir)
    for i in range(1):
        stride = 2**i
        for iptl in range(30, 80):
            print(iptl)
            particle_energy(iptl, particle_tags, pic_info, stride, odir)
    # navg = 10
    # for iptl in range(130, 150):
    #     print(iptl)
    #     energy_gain(iptl, particle_tags, pic_info, navg, odir)
    file.close()
