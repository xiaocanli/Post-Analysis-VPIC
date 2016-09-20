"""
Analysis procedures to calculate power spectrum
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
from energy_conversion import read_data_from_json
from shell_functions import mkdir_p
from joblib import Parallel, delayed
from contour_plots import read_2d_fields, plot_2d_contour
import multiprocessing

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

font = {'family' : 'serif',
        #'color'  : 'darkred',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 24,
        }


def calc_power_spectrum(pic_info, ct, run_name, base_dir='../../'):
    """Plot plasma beta and number density.

    Args:
        pic_info: namedtuple for the PIC simulation information.
        ct: current time frame.
    """
    xmin, xmax = 0, pic_info.lx_di
    xmin, xmax = 0, 105
    zmin, zmax = -0.5*pic_info.lz_di, 0.5*pic_info.lz_di
    kwargs = {"current_time":ct, "xl":xmin, "xr":xmax, "zb":zmin, "zt":zmax}
    fname = base_dir + 'data/vex.gda'
    x, z, vel = read_2d_fields(pic_info, fname, **kwargs) 
    nx, = x.shape
    nz, = z.shape
    data_cum = np.sum(vel, axis=0) / nz
    data_grad = np.abs(np.gradient(data_cum))
    xs = 5
    max_index = np.argmax(data_grad[xs:])
    xm = x[max_index]

    xmin, xmax = 0, xm
    fname = base_dir + 'data/bx.gda'
    x, z, bx = read_2d_fields(pic_info, fname, **kwargs) 
    fname = base_dir + 'data/by.gda'
    x, z, by = read_2d_fields(pic_info, fname, **kwargs) 
    fname = base_dir + 'data/bz.gda'
    x, z, bz = read_2d_fields(pic_info, fname, **kwargs) 
    smime = math.sqrt(pic_info.mime)
    lx = np.max(x) - np.min(x)
    lz = np.max(z) - np.min(z)

    bx_k = np.fft.rfft2(bx)
    by_k = np.fft.rfft2(by)
    bz_k = np.fft.rfft2(bz)
    b2_k = np.absolute(bx_k)**2 + np.absolute(by_k)**2 + np.absolute(bz_k)**2
    xstep = lx / nx
    kx = np.fft.fftfreq(nx, xstep)
    idx = np.argsort(kx)
    zstep = lz / nz
    kz = np.fft.fftfreq(nz, zstep)
    idz = np.argsort(kz)
    print np.min(kx), np.max(kx), np.min(kz), np.max(kz)

    kxs, kzs = np.meshgrid(kx[:nx//2+1], kz)
    ks = np.sqrt(kxs*kxs + kzs*kzs)
    kmin, kmax = np.min(ks), np.max(ks)
    kbins = np.linspace(kmin, kmax, nx//2+1, endpoint=True)
    ps, kbins_edges = np.histogram(ks, bins=kbins, weights=b2_k, normed=True)
    w1, h1 = 0.8, 0.8
    xs, ys = 0.15, 0.95 - h1
    fig = plt.figure(figsize=[7, 5])
    ax1 = fig.add_axes([xs, ys, w1, h1])
    ax1.loglog(kbins_edges[:-1], ps, linewidth=2)
    psm = np.argmax(ps)
    pindex = -3.0
    power_k = kbins[psm:]**-3.0
    shift = 400
    ax1.loglog(kbins[psm:psm+shift], power_k[:shift]*2.0E2/power_k[psm],
            linestyle='--', linewidth=2, color='k')
    power_index = "{%0.1f}" % pindex
    tname = r'$\sim k^{' + power_index + '}$'
    ax1.text(0.4, 0.8, tname, color='black', fontsize=24,
            horizontalalignment='left', verticalalignment='center',
            transform = ax1.transAxes)
    ax1.tick_params(labelsize=16)
    ax1.set_xlabel(r'$kd_i$', fontdict=font, fontsize=20)
    ax1.set_ylabel(r'$E_B(k)$', fontdict=font, fontsize=20)
    ax1.set_ylim([1E-2, 3E1])
    ax1.set_ylim([1E-7, 1E3])

    fig_dir = '../img/img_power_spectrum/' + run_name + '/'
    mkdir_p(fig_dir)
    fname = fig_dir + '/ps_' + str(ct).zfill(3) + '.jpg'
    fig.savefig(fname, dpi=300)

    # plt.show()
    plt.close()


if __name__ == "__main__":
    base_dir = '/net/scratch3/xiaocanli/2D-90-Mach4-sheet4-multi/'
    run_name = '2D-90-Mach4-sheet4-multi'
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    ct = pic_info.ntf - 2
    # calc_power_spectrum(pic_info, ct, run_name, base_dir)
    cts = range(10, pic_info.ntf - 1)
    def processInput(ct):
        print ct
        calc_power_spectrum(pic_info, ct, run_name, base_dir)
    num_cores = multiprocessing.cpu_count()
    Parallel(n_jobs=num_cores)(delayed(processInput)(ct) for ct in cts)
