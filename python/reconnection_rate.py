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
from scipy import signal
import numpy as np
import math
import os.path
import struct
import collections
import pic_information
from contour_plots import read_2d_fields, plot_2d_contour

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

font = {'family' : 'serif',
        #'color'  : 'darkred',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 24,
        }


def calc_reconnection_rate(base_dir):
    """Calculate reconnection rate.

    Args:
        base_dir: the directory base.
    """
    pic_info = pic_information.get_pic_info(base_dir)
    ntf = pic_info.ntf
    phi = np.zeros(ntf)
    fname = base_dir + 'data/Ay.gda'
    for ct in range(ntf):
        kwargs = {"current_time":ct, "xl":0, "xr":200, "zb":-1, "zt":1}
        x, z, Ay = read_2d_fields(pic_info, fname, **kwargs) 
        nz, = z.shape
        # max_ay = np.max(np.sum(Ay[nz/2-1:nz/2+1, :], axis=0)/2)
        # min_ay = np.min(np.sum(Ay[nz/2-1:nz/2+1, :], axis=0)/2)
        max_ay = np.max(Ay[nz/2-1:nz/2+1, :])
        min_ay = np.min(Ay[nz/2-1:nz/2+1, :])
        phi[ct] = max_ay - min_ay
    nk = 3
    phi = signal.medfilt(phi, kernel_size=nk)
    dtwpe = pic_info.dtwpe
    dtwce = pic_info.dtwce
    dtwci = pic_info.dtwci
    mime = pic_info.mime
    dtf_wpe = pic_info.dt_fields * dtwpe / dtwci
    reconnection_rate = np.gradient(phi) / dtf_wpe
    b0 = pic_info.b0
    va = dtwce * math.sqrt(1.0/mime) / dtwpe
    reconnection_rate /= b0 * va
    tfields = pic_info.tfields

    return (tfields, reconnection_rate)


def save_reconnection_rate(tfields, reconnection_rate, fname):
    """Save the calculated reconnection rate.
    """
    if not os.path.isdir('../data/'):
        os.makedirs('../data/')
    filename = '../data/' + fname
    f = open(filename, 'w')
    np.savetxt(f, (tfields, reconnection_rate))
    f.close()


def plot_reconnection_rate(base_dir):
    """Calculate and plot the reconnection rate

    Args:
        base_dir: the directory base.
    """
    tfields, reconnection_rate = calc_reconnection_rate(base_dir)
    fig = plt.figure(figsize=[7, 5])
    ax = fig.add_axes([0.18, 0.15, 0.78, 0.8])
    ax.plot(tfields, reconnection_rate, color='black', linewidth=2)
    ax.set_xlabel(r'$t\Omega_{ci}$', fontdict=font, fontsize=24)
    ax.set_ylabel(r'$E_R$', fontdict=font, fontsize=24)
    ax.tick_params(labelsize=20)
    ax.set_ylim([0, 0.12])
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    fname = '../img/er.eps'
    fig.savefig(fname)
    plt.show()


def calc_multi_reconnection_rate():
    """Calculate reconnection rate for multiple runs
    """
    # base_dir = '/net/scratch2/xiaocanli/mime25-sigma01-beta02-200-100/'
    # t, rate = calc_reconnection_rate(base_dir)
    # save_reconnection_rate(t, rate, 'rate_mime25_beta02.dat')

    # base_dir = '/net/scratch2/xiaocanli/mime25-sigma033-beta006-200-100/'
    # t, rate = calc_reconnection_rate(base_dir)
    # save_reconnection_rate(t, rate, 'rate_mime25_beta007.dat')

    # base_dir = '/scratch3/xiaocanli/sigma1-mime25-beta001/'
    # t, rate = calc_reconnection_rate(base_dir)
    # save_reconnection_rate(t, rate, 'rate_mime25_beta002.dat')

    # base_dir = '/scratch3/xiaocanli/sigma1-mime25-beta0003-npc200/'
    # t, rate = calc_reconnection_rate(base_dir)
    # save_reconnection_rate(t, rate, 'rate_mime25_beta0007.dat')

    # base_dir = '/scratch3/xiaocanli/sigma1-mime100-beta001-mustang/'
    # t, rate = calc_reconnection_rate(base_dir)
    # save_reconnection_rate(t, rate, 'rate_mime100_beta002.dat')

    # base_dir = '/scratch3/xiaocanli/mime25-guide0-beta001-200-100/'
    # t, rate = calc_reconnection_rate(base_dir)
    # save_reconnection_rate(t, rate, 'rate_mime25_beta002_sigma01.dat')

    # base_dir = '/scratch3/xiaocanli/mime25-guide0-beta001-200-100-sigma033/'
    # t, rate = calc_reconnection_rate(base_dir)
    # save_reconnection_rate(t, rate, 'rate_mime25_beta002_sigma033.dat')


def plot_multi_reconnection_rate():
    """Calculate reconnection rate for multiple runs
    """
    path = '../data/'
    fname = path + 'rate_mime25_beta02.dat'
    tf1, rate1 = np.genfromtxt(fname)
    fname = path + 'rate_mime25_beta007.dat'
    tf2, rate2 = np.genfromtxt(fname)
    fname = path + 'rate_mime25_beta002.dat'
    tf3, rate3 = np.genfromtxt(fname)
    fname = path + 'rate_mime25_beta0007.dat'
    tf4, rate4 = np.genfromtxt(fname)
    fname = path + 'rate_mime100_beta002.dat'
    tf5, rate5 = np.genfromtxt(fname)
    fname = path + 'rate_mime25_beta002_sigma01.dat'
    tf6, rate6 = np.genfromtxt(fname)
    fname = path + 'rate_mime25_beta002_sigma033.dat'
    tf7, rate7 = np.genfromtxt(fname)

    fig = plt.figure(figsize=[7, 5])
    ax = fig.add_axes([0.18, 0.15, 0.78, 0.8])
    # ax.plot(tf1, rate1, linewidth=2)
    # ax.plot(tf2, rate2, linewidth=2)
    ax.plot(tf3, rate3, linewidth=2)
    ax.plot(tf4, rate4, linewidth=2)
    # ax.plot(tf5, rate5, linewidth=2)
    ax.plot(tf6, rate6, linewidth=2)
    ax.plot(tf7, rate7, linewidth=2)
    ax.set_xlabel(r'$t\Omega_{ci}$', fontdict=font, fontsize=24)
    ax.set_ylabel(r'$E_R$', fontdict=font, fontsize=24)
    ax.tick_params(labelsize=20)
    ax.set_ylim([0, 0.12])
    plt.show()


if __name__ == "__main__":
    # plot_reconnection_rate('../../')
    # calc_multi_reconnection_rate()
    plot_multi_reconnection_rate()
