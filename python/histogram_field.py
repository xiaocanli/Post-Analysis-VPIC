"""
Analysis procedures for particle energy spectrum.
"""
import collections
import math
import os.path
import struct

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D

import pic_information
from contour_plots import read_2d_fields

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

font = {'family' : 'serif',
        #'color'  : 'darkred',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 24,
        }

def histogram_field(pic_info, var_name, fname, **kwargs):
    """Plot a histogram of a field.

    Args:
        pic_info: namedtuple for the PIC simulation information.
        var_name: variable name.
        fname: file name.
    """
    x, z, field_data = read_2d_fields(pic_info, fname, **kwargs) 
    dmax = np.max(field_data)
    dmin = np.min(field_data)
    dmin = -100.0
    dmax = 100.0
    nb = 1000
    print np.sum(field_data)
    bins = np.linspace(dmin, dmax, nb)
    hist, bin_edges = np.histogram(field_data, bins)
    print np.sum(hist)
    print 'Maximum and Minimum of the field: ', dmax, dmin
    fig = plt.figure(figsize=[7, 5])
    width = 0.8
    height = 0.8
    ax = fig.add_axes([0.12, 0.15, width, height])
    hist_f = np.array(hist, dtype=np.float64)
    hist_f /= np.max(hist_f)
    p1 = ax.semilogy(bins[:nb-1], hist_f, linewidth=2)
    ax.tick_params(labelsize=16)
    ax.set_xlabel(var_name, fontdict=font, fontsize=20)
    qname = r'$f($' + var_name + '$)$'
    ax.set_ylabel(qname, fontdict=font, fontsize=20)
    plt.show()

if __name__ == "__main__":
    pic_info = pic_information.get_pic_info("../../")
    kwargs = {"current_time":40, "xl":0, "xr":400, "zb":-100, "zt":100}
    #histogram_field(pic_info, r'$n_e$', "../data/ne.gda", **kwargs)
    #histogram_field(pic_info, r'$|\mathbf{B}|$', "../data/absB.gda", **kwargs)
    # histogram_field(pic_info, r'$A_e$', "../data1/agyrotropy00_e.gda", **kwargs)
    histogram_field(pic_info, r'$A_e$', "../../data1/jpolar_dote00_e.gda", **kwargs)
