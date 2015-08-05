"""
Analysis procedures for particle energy spectrum.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy import signal
import math
import os.path
from os import listdir
from os.path import isfile, join
import struct
import collections
import pic_information
from contour_plots import read_2d_fields, plot_2d_contour

mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

font = {'family' : 'serif',
        #'color'  : 'darkred',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 24,
        }

def get_file_names():
    """Get the file names in the traj folder.
    """
    traj_path = '../../traj/'
    fnames = [ f for f in listdir(traj_path) if isfile(join(traj_path,f)) ]
    fnames.sort()
    ntraj_e = 0
    for filename in fnames:
        if filename.startswith('e'):
            ntraj_e += 1
        else:
            break
    fnames_e = fnames[:ntraj_e]
    fnames_i = fnames[ntraj_e:]
    return (fnames_e, fnames_i)

def read_traj_data(filename):
    """Read the information of one particle trajectory.

    Args:
        filename: the filename of the trajectory file.
    """
    nvar = 13  # variables at each point
    ptl_traj_info = collections.namedtuple("ptl_traj_info",
            ['t', 'x', 'y', 'z', 'ux', 'uy', 'uz',
                'ex', 'ey', 'ez', 'bx', 'by', 'bz'])
    print 'Reading trajectory data from ', filename
    fname = '../../traj/' + filename
    statinfo = os.stat(fname)
    nstep = statinfo.st_size / nvar
    nstep /= np.dtype(np.float32).itemsize
    traj_data = np.zeros((nvar, nstep), dtype=np.float32)
    traj_data = np.memmap(fname, dtype='float32', mode='r', 
            offset=0, shape=((nvar, nstep)), order='F')
    ptl_traj = ptl_traj_info(t=traj_data[0,:],
            x=traj_data[1,:], y=traj_data[2,:], z=traj_data[3,:],
            ux=traj_data[4,:], uy=traj_data[5,:], uz=traj_data[6,:],
            ex=traj_data[7,:], ey=traj_data[8,:], ez=traj_data[9,:],
            bx=traj_data[10,:], by=traj_data[11,:], bz=traj_data[12,:])
    return ptl_traj

def plot_traj(filename, pic_info, species, iptl):
    """Plot particle trajectory information.

    Args:
        filename: the filename to read the data.
        pic_info: namedtuple for the PIC simulation information.
        species: particle species. 'e' for electron. 'i' for ion.
        iptl: particle ID.
    """
    ptl_traj = read_traj_data(filename)
    gama = np.sqrt(ptl_traj.ux**2 + ptl_traj.uy**2 + ptl_traj.uz**2 + 1.0)
    mime = pic_info.mime
    # de scale to di scale
    ptl_x = ptl_traj.x / math.sqrt(mime)
    ptl_y = ptl_traj.y / math.sqrt(mime)
    ptl_z = ptl_traj.z / math.sqrt(mime)
    # 1/wpe to 1/wci
    t = ptl_traj.t * pic_info.dtwci / pic_info.dtwpe

    xl, xr = 0, 200
    zb, zt = -20, 20
    kwargs = {"current_time":50, "xl":xl, "xr":xr, "zb":zb, "zt":zt}
    fname = "../../data/by.gda"
    x, z, data_2d = read_2d_fields(pic_info, fname, **kwargs) 
    x, z, Ay = read_2d_fields(pic_info, "../../data/Ay.gda", **kwargs) 
    nx, = x.shape
    nz, = z.shape

    # p1 = ax1.plot(ptl_x, gama-1.0, color='k')
    # ax1.set_xlabel(r'$t\Omega_{ci}$', fontdict=font)
    # ax1.set_ylabel(r'$E/m_ic^2$', fontdict=font)
    # ax1.tick_params(labelsize=20)

    width = 14.0
    height = 8.0
    fig = plt.figure(figsize=[width, height])
    w1, h1 = 0.82, 0.27
    xs, ys = 0.10, 0.98-h1
    ax1 = fig.add_axes([xs, ys, w1, h1])
    kwargs_plot = {"xstep":2, "zstep":2, "is_log":False, "vmin":-1.0, "vmax":1.0}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    im1, cbar1 = plot_2d_contour(x, z, data_2d, ax1, fig, **kwargs_plot)
    im1.set_cmap(plt.cm.seismic)
    ax1.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep], 
            colors='black', linewidths=0.5)
    ax1.tick_params(labelsize=20)
    ax1.tick_params(axis='x', labelbottom='off')
    ax1.set_xlim([xl, xr])
    ax1.set_ylim([zb, zt])
    ax1.set_ylabel(r'$z/d_i$', fontdict=font)
    cbar1.ax.tick_params(labelsize=20)
    cbar1.ax.set_ylabel(r'$B_y$', fontdict=font, fontsize=24)
    p1 = ax1.scatter(ptl_x, ptl_z, s=0.5)

    # ax2 = fig.add_axes([xs, ys, w1, h1])
    # p2 = ax2.plot(t, gama-1.0, color='k')
    # ax2.set_xlabel(r'$t\Omega_{ci}$', fontdict=font)
    # ax2.set_ylabel(r'$E/m_ic^2$', fontdict=font)
    # ax2.tick_params(labelsize=20)

    gap = 0.04
    ys -= h1 + gap
    ax2 = fig.add_axes([xs, ys, w1*0.98-0.05/width, h1])
    p2 = ax2.plot(ptl_x, gama-1.0, color='k')
    ax2.set_ylabel(r'$E/m_ic^2$', fontdict=font)
    ax2.tick_params(labelsize=20)
    ax2.tick_params(axis='x', labelbottom='off')
    xmin, xmax = ax2.get_xlim()

    ys -= h1 + gap
    ax3 = fig.add_axes([xs, ys, w1*0.98-0.05/width, h1])
    p3 = ax3.plot(ptl_x, ptl_y, color='k')
    ax3.set_xlabel(r'$x/d_i$', fontdict=font)
    ax3.set_ylabel(r'$y/d_i$', fontdict=font)
    ax3.tick_params(labelsize=20)

    ax1.set_xlim([xmin, xmax])
    ax3.set_xlim([xmin, xmax])

    dir = '../img_traj_' + species + '/'
    if not os.path.isdir(dir):
        os.makedirs(dir)
    fname = dir + 'traj_' + species + '_' + str(iptl).zfill(4) + '_1.eps'
    fig.savefig(fname)

    height = 15.0
    fig = plt.figure(figsize=[width, height])
    w1, h1 = 0.88, 0.135
    xs, ys = 0.10, 0.98-h1
    gap = 0.025

    ax1 = fig.add_axes([xs, ys, w1, h1])
    p1 = ax1.plot(t, gama-1.0, color='k')
    ax1.set_ylabel(r'$E/m_ic^2$', fontdict=font)
    ax1.tick_params(labelsize=20)
    ax1.tick_params(axis='x', labelbottom='off')
    ax1.text(0.4, -0.07, r'$x$', color='red', fontsize=32, 
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='center',
            transform = ax1.transAxes)
    ax1.text(0.5, -0.07, r'$y$', color='green', fontsize=32, 
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='center',
            transform = ax1.transAxes)
    ax1.text(0.6, -0.07, r'$z$', color='blue', fontsize=32, 
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='center',
            transform = ax1.transAxes)

    ys -= h1 + gap
    ax2 = fig.add_axes([xs, ys, w1, h1])
    p21 = ax2.plot(t, ptl_traj.ux, color='r', label=r'u_x')
    p22 = ax2.plot(t, ptl_traj.uy, color='g', label=r'u_y')
    p23 = ax2.plot(t, ptl_traj.uz, color='b', label=r'u_z')
    ax2.set_ylabel(r'$u_x, u_y, u_z$', fontdict=font)
    ax2.tick_params(labelsize=20)
    ax2.tick_params(axis='x', labelbottom='off')

    kernel = 9
    tmax = np.max(t)
    ys -= h1 + gap
    ax31 = fig.add_axes([xs, ys, w1, h1])
    ex = signal.medfilt(ptl_traj.ex, kernel_size=(kernel))
    p31 = ax31.plot(t, ex, color='r', label=r'E_x')
    ax31.set_ylabel(r'$E_x$', fontdict=font)
    ax31.tick_params(labelsize=20)
    ax31.tick_params(axis='x', labelbottom='off')
    ax31.plot([0, tmax], [0, 0], '--', color='k')

    ys -= h1 + gap
    ax32 = fig.add_axes([xs, ys, w1, h1])
    ey = signal.medfilt(ptl_traj.ey, kernel_size=(kernel))
    p32 = ax32.plot(t, ey, color='g', label=r'E_y')
    ax32.set_ylabel(r'$E_y$', fontdict=font)
    ax32.tick_params(labelsize=20)
    ax32.tick_params(axis='x', labelbottom='off')
    ax32.plot([0, tmax], [0, 0], '--', color='k')

    ys -= h1 + gap
    ax33 = fig.add_axes([xs, ys, w1, h1])
    ez = signal.medfilt(ptl_traj.ez, kernel_size=(kernel))
    p33 = ax33.plot(t, ez, color='b', label=r'E_z')
    ax33.set_ylabel(r'$E_z$', fontdict=font)
    ax33.tick_params(labelsize=20)
    ax33.tick_params(axis='x', labelbottom='off')
    ax33.plot([0, tmax], [0, 0], '--', color='k')

    ys -= h1 + gap
    ax4 = fig.add_axes([xs, ys, w1, h1])
    p41 = ax4.plot(t, ptl_traj.bx, color='r', label=r'B_x')
    p42 = ax4.plot(t, ptl_traj.by, color='g', label=r'B_y')
    p43 = ax4.plot(t, ptl_traj.bz, color='b', label=r'B_z')
    ax4.set_xlabel(r'$t\Omega_{ci}$', fontdict=font)
    ax4.set_ylabel(r'$B_x, B_y, B_z$', fontdict=font)
    ax4.tick_params(labelsize=20)
    xmax = 800
    ax1.set_xlim([0, xmax])
    ax2.set_xlim([0, xmax])
    ax31.set_xlim([0, xmax])
    ax32.set_xlim([0, xmax])
    ax33.set_xlim([0, xmax])
    ax4.set_xlim([0, xmax])

    fname = dir + 'traj_' + species + '_' + str(iptl).zfill(4) + '_2.eps'
    fig.savefig(fname)

    plt.show()


if __name__ == "__main__":
    pic_info = pic_information.get_pic_info('../../')
    ntp = pic_info.ntp
    vthe = pic_info.vthe
    fnames_e, fnames_i = get_file_names()
    iptl = 316
    plot_traj(fnames_i[iptl], pic_info, 'i', iptl)
