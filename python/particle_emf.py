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

def particle_energy(iptl, particle_tags, pic_info, odir):
    group = file[particle_tags[iptl]]
    dset_ux = group['Ux']
    sz, = dset_ux.shape
    ux = np.zeros(sz)
    uy = np.zeros(sz)
    uz = np.zeros(sz)
    Bx = np.zeros(sz)
    By = np.zeros(sz)
    Bz = np.zeros(sz)
    Ex = np.zeros(sz)
    Ey = np.zeros(sz)
    Ez = np.zeros(sz)
    dset_ux = group['Ux']
    dset_uy = group['Uy']
    dset_uz = group['Uz']
    dset_bx = group['Bx']
    dset_by = group['By']
    dset_bz = group['Bz']
    dset_ex = group['Ex']
    dset_ey = group['Ey']
    dset_ez = group['Ez']
    dset_ux.read_direct(ux)
    dset_uy.read_direct(uy)
    dset_uz.read_direct(uz)
    dset_bx.read_direct(Bx)
    dset_by.read_direct(By)
    dset_bz.read_direct(Bz)
    dset_ex.read_direct(Ex)
    dset_ey.read_direct(Ey)
    dset_ez.read_direct(Ez)

    gama = np.sqrt(ux*ux + uy*uy + uz*uz + 1) - 1.0

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

    dtwci = pic_info.dtwci
    dtwpe = pic_info.dtwpe
    dt = pic_info.dt_fields * dtwpe / dtwci
    jdote_para_cum = np.cumsum(jdote_para) * dt
    jdote_perp_cum = np.cumsum(jdote_perp) * dt
    jdote_tot_cum = jdote_para_cum + jdote_perp_cum
    tfields = np.arange(sz) * dt

    fig = plt.figure(figsize=[7, 5])
    xs, ys = 0.13, 0.13
    w1, h1 = 0.8, 0.8
    ax = fig.add_axes([xs, ys, w1, h1])
    ax.plot(tfields, jdote_tot_cum, linewidth=2, color='r', 
            label=r'$\int q\mathbf{v}\cdot\mathbf{E}$')
    ax.plot(tfields, gama, linewidth=2, color='k', label=r'$\gamma$')
    ax.plot(tfields, jdote_para_cum, linewidth=2, color='g',
            label=r'$\int q\mathbf{v}_\parallel\cdot\mathbf{E}$')
    ax.plot(tfields, jdote_perp_cum, linewidth=2, color='b',
            label=r'$\int q\mathbf{v}_\perp\cdot\mathbf{E}$')
    leg = ax.legend(loc=7, prop={'size':20}, ncol=1,
            shadow=False, fancybox=False, frameon=False)
    ax.tick_params(labelsize=16)
    ax.set_xlabel(r'$t\Omega_{pe}$', fontdict=font, fontsize=24)
    fname = odir + 'ptl_ene_' + str(iptl) + '.eps'
    plt.savefig(fname)
    # plt.show()
    plt.close()


if __name__ == "__main__":
    pic_info = pic_information.get_pic_info('../../')
    filepath = '/net/scratch3/guofan/turbulent-sheet3D-mixing-trinity-Jan29-test/'
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
    for iptl in range(nptl):
        print(iptl)
        particle_energy(iptl, particle_tags, pic_info, odir)
    file.close()
