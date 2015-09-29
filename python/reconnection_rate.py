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


def calc_reconnection_rate(pic_info):
    """Calculate reconnection rate.

    Args:
        pic_info: namedtuple for the PIC simulation information.
    """
    ntf = pic_info.ntf
    phi = np.zeros(ntf)
    for ct in range(ntf):
        kwargs = {"current_time":ct, "xl":0, "xr":200, "zb":-1, "zt":1}
        x, z, Ay = read_2d_fields(pic_info, "../../data/Ay.gda", **kwargs) 
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

if __name__ == "__main__":
    pic_info = pic_information.get_pic_info('../../')
    calc_reconnection_rate(pic_info)
