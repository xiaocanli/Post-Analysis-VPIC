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
from scipy.interpolate import interp1d
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

def read_var(group, var, sz):
    fdata = np.zeros(sz)
    dset= group[var]
    dset.read_direct(fdata)
    return fdata

def read_data(iptl, particle_tags, pic_info):
    group = file[particle_tags[iptl]]
    dset_ux = group['q']
    sz, = dset_ux.shape
    x = read_var(group, 'dX', sz)
    y = read_var(group, 'dY', sz)
    z = read_var(group, 'dZ', sz)
    ux = read_var(group, 'Ux', sz)
    uy = read_var(group, 'Uy', sz)
    uz = read_var(group, 'Uz', sz)
    vx = read_var(group, 'Vx', sz)
    vy = read_var(group, 'Vy', sz)
    vz = read_var(group, 'Vz', sz)
    Bx = read_var(group, 'Bx', sz)
    By = read_var(group, 'By', sz)
    Bz = read_var(group, 'Bz', sz)
    Ex = read_var(group, 'Ex', sz)
    Ey = read_var(group, 'Ey', sz)
    Ez = read_var(group, 'Ez', sz)
    q = read_var(group, 'q', sz)

    gama = np.sqrt(ux*ux + uy*uy + uz*uz + 1)
    dtwci = pic_info.dtwci
    dtwpe = pic_info.dtwpe
    dt = pic_info.dt_fields * dtwpe / dtwci
    tfields = np.arange(sz) * dt
    smime = math.sqrt(pic_info.mime)

    nt = np.count_nonzero(q)
    index = np.nonzero(q)
    x = x[index]
    y = y[index]
    z = z[index]
    ux = ux[index]
    uy = uy[index]
    uz = uz[index]
    vx = vx[index]
    vy = vy[index]
    vz = vz[index]
    Bx = Bx[index]
    By = By[index]
    Bz = Bz[index]
    Ex = Ex[index]
    Ey = Ey[index]
    Ez = Ez[index]

    x /= smime
    y /= smime
    z /= smime

    q = q[index]
    tfields = tfields[index]
    gama = gama[index]
    return (x, y, z, ux, uy, uz, vx, vy, vz, Bx, By, Bz, Ex, Ey, Ez, q, tfields,
            gama, sz)


def interp_data(sz_new, x, y, z, ux, uy, uz, vx, vy, vz, Bx, By, Bz, Ex, Ey, Ez,
        q, tfields, gama, pic_info, sz, interp_kind='linear'):
    dtwci = pic_info.dtwci
    dtwpe = pic_info.dtwpe
    dt = pic_info.dt_fields * dtwpe / dtwci
    tmax = tfields[-1]
    tnew = np.linspace(0, tmax, sz_new)
    dt_new = sz * dt / sz_new

    f = interp1d(tfields, x, kind=interp_kind)
    x = f(tnew)
    f = interp1d(tfields, y, kind=interp_kind)
    y = f(tnew)
    f = interp1d(tfields, z, kind=interp_kind)
    z = f(tnew)
    f = interp1d(tfields, ux, kind=interp_kind)
    ux = f(tnew)
    f = interp1d(tfields, uy, kind=interp_kind)
    uy = f(tnew)
    f = interp1d(tfields, uz, kind=interp_kind)
    uz = f(tnew)
    f = interp1d(tfields, vx, kind=interp_kind)
    vx = f(tnew)
    f = interp1d(tfields, vy, kind=interp_kind)
    vy = f(tnew)
    f = interp1d(tfields, vz, kind=interp_kind)
    vz = f(tnew)
    f = interp1d(tfields, Bx, kind=interp_kind)
    Bx = f(tnew)
    f = interp1d(tfields, By, kind=interp_kind)
    By = f(tnew)
    f = interp1d(tfields, Bz, kind=interp_kind)
    Bz = f(tnew)
    f = interp1d(tfields, Ex, kind=interp_kind)
    Ex = f(tnew)
    f = interp1d(tfields, Ey, kind=interp_kind)
    Ey = f(tnew)
    f = interp1d(tfields, Ez, kind=interp_kind)
    Ez = f(tnew)
    f = interp1d(tfields, gama, kind=interp_kind)
    gama = f(tnew)
    return (x, y, z, ux, uy, uz, vx, vy, vz, Bx, By, Bz, Ex, Ey, Ez, gama,
            dt_new, tnew)


def particle_energy(iptl, particle_tags, pic_info, stride, odir):
    x, y, z, ux, uy, uz, vx, vy, vz, Bx, By, Bz, Ex, Ey, Ez, q, tfields, \
            gama, sz = read_data(iptl, particle_tags, pic_info)
    x, y, z, ux, uy, uz, vx, vy, vz, Bx, By, Bz, Ex, Ey, Ez, gama, \
            dt_new, tnew  = interp_data(sz, x, y, z, ux, uy, uz, \
            vx, vy, vz, Bx, By, Bz, Ex, Ey, Ez, q, tfields, gama, pic_info, sz)
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

    Einx = vz*By - vy*Bz
    Einy = vx*Bz - vz*Bx
    Einz = vy*Bx - vx*By
    jdote_in = -(ux*Einx + uy*Einy + uz*Einz) / gama

    jdote_para_cum = np.cumsum(jdote_para[::stride]) * dt_new * stride
    jdote_perp_cum = np.cumsum(jdote_perp[::stride]) * dt_new * stride
    jdote_in_cum = np.cumsum(jdote_in[::stride]) * dt_new * stride
    jdote_tot_cum = jdote_para_cum + jdote_perp_cum

    fig = plt.figure(figsize=[7, 5])
    xs, ys = 0.13, 0.13
    w1, h1 = 0.8, 0.8
    ax = fig.add_axes([xs, ys, w1, h1])
    # ax.plot(tnew[::stride], jdote_tot_cum, linewidth=2, color='r', 
    #         label=r'$\int q\boldsymbol{v}\cdot\boldsymbol{E}$')
    ax.plot(tnew, gama-1, linewidth=2, color='k', label=r'$\gamma-1$')
    ax.plot(tnew[::stride], jdote_para_cum, linewidth=2, color='b',
            label=r'$\int q\boldsymbol{v}_\parallel\cdot\boldsymbol{E}$')
    ax.plot(tnew[::stride], jdote_perp_cum, linewidth=2, color='r',
            label=r'$\int q\boldsymbol{v}_\perp\cdot\boldsymbol{E}$')
    # ax.plot(tnew[::stride], jdote_in_cum, linewidth=2, color='m',
    #         label=r'$\int q\boldsymbol{v}_\perp\cdot\boldsymbol{E}_{vB}$')
    leg = ax.legend(loc=2, prop={'size':20}, ncol=1,
            shadow=False, fancybox=False, frameon=False)
    ax.set_xlim([0, tnew[-1]])
    ax.tick_params(labelsize=16)
    ax.set_xlabel(r'$t\omega_{pe}$', fontdict=font, fontsize=24)
    fname = odir + 'ptl_ene_' + str(iptl) + '_' + str(stride) + '.eps'
    plt.savefig(fname)
    plt.show()
    # plt.close()


def adjust_pos(pos, length):
    """Adjust position for periodic boundary conditions.
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


def save_new_data(iptl, particle_tags, pic_info, fh):
    x, y, z, ux, uy, uz, vx, vy, vz, Bx, By, Bz, Ex, Ey, Ez, q, tfields, \
            gama, sz = read_data(iptl, particle_tags, pic_info)
    x = adjust_pos(x, pic_info.lx_di)
    y = adjust_pos(y, pic_info.ly_di)
    z = adjust_pos(z, pic_info.lz_di)
    x, y, z, ux, uy, uz, vx, vy, vz, Bx, By, Bz, Ex, Ey, Ez, gama, \
            dt_new, tfields = interp_data(sz*10, x, y, z, ux, uy, uz, \
            vx, vy, vz, Bx, By, Bz, Ex, Ey, Ez, q, tfields, gama, pic_info,
            sz, 'cubic')
    sz, = tfields.shape
    grp = fh.create_group(particle_tags[iptl])
    grp.create_dataset("X", (sz, ), data=x)
    grp.create_dataset("Y", (sz, ), data=y)
    grp.create_dataset("Z", (sz, ), data=z)
    grp.create_dataset("Ux", (sz, ), data=ux)
    grp.create_dataset("Uy", (sz, ), data=uy)
    grp.create_dataset("Uz", (sz, ), data=uz)
    grp.create_dataset("Vx", (sz, ), data=vx)
    grp.create_dataset("Vy", (sz, ), data=vy)
    grp.create_dataset("Vz", (sz, ), data=vz)
    grp.create_dataset("Ex", (sz, ), data=Ex)
    grp.create_dataset("Ey", (sz, ), data=Ey)
    grp.create_dataset("Ez", (sz, ), data=Ez)
    grp.create_dataset("Bx", (sz, ), data=Bx)
    grp.create_dataset("By", (sz, ), data=By)
    grp.create_dataset("Bz", (sz, ), data=Bz)
    grp.create_dataset("T", (sz, ), data=tfields)


if __name__ == "__main__":
    # filepath = '/net/scratch3/guofan/trinity/turbulent-sheet3D-mixing-sigma100/'
    filepath = '/net/scratch3/xiaocanli/turbulence-large-dB-tracking/'
    pic_info = pic_information.get_pic_info(filepath)
    filepath += 'pic_analysis/vpic-sorter/data/'
    species = 'e'
    if species == 'i':
        fname = filepath + 'ions.h5p'
    else:
        fname = filepath + 'electrons_2.h5p'
    fname2 = filepath + 'electrons_interp.h5p'
    file = h5py.File(fname, 'r')
    particle_tags = []
    for item in file:
        particle_tags.append(item)
    nptl = len(particle_tags)
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    odir = '../img/ptl_ene/'

    file2 = h5py.File(fname2, 'w')

    if not os.path.isdir(odir):
        os.makedirs(odir)
    for i in range(1):
        stride = 2**i
        for iptl in range(nptl):
            print(iptl)
            # particle_energy(iptl, particle_tags, pic_info, stride, odir)
            save_new_data(iptl, particle_tags, pic_info, file2)
    file.close()
    file2.close()
