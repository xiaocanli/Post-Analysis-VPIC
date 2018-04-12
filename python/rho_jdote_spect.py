"""
Analysis procedures for compression and jdote. It was used to generate a movie
that combines electron density, energization terms, and electron energy spectrum
"""
import argparse
import collections
import math
import multiprocessing
import os
import os.path
import struct
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import seaborn as sns
import simplejson as json
from joblib import Parallel, delayed
from matplotlib import rc
from scipy import signal

import fitting_funcs
import palettable
import pic_information
import spectrum_fitting as spect_fit
from contour_plots import read_2d_fields
from json_functions import read_data_from_json
from shell_functions import mkdir_p

style.use(['seaborn-white', 'seaborn-paper', 'seaborn-ticks'])
rc("font", family="Times New Roman")
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

colors_Set1_9 = palettable.colorbrewer.qualitative.Set1_9.mpl_colors
colors_Dark2_8 = palettable.colorbrewer.qualitative.Dark2_8.mpl_colors
colors_Paired_12 = palettable.colorbrewer.qualitative.Paired_12.mpl_colors
colors_Tableau_10 = palettable.tableau.Tableau_10.mpl_colors
colors_GreenOrange_6 = palettable.tableau.GreenOrange_6.mpl_colors

font = {
    'family': 'serif',
    'color': 'black',
    'weight': 'normal',
    'size': 24,
}

def plot_ne_jdote(pic_info, root_dir, run_name, current_time):
    """
    Plot electron number density and jdote

    Args:
        pic_info: namedtuple for the PIC simulation information.
        root_dir: simulation root directory
        current_time: current time frame.
    """
    print("Time frame: %d" % current_time)
    smime = math.sqrt(pic_info.mime)
    dx = pic_info.dx_di * smime
    dz = pic_info.dz_di * smime
    lx_de = pic_info.lx_di * smime
    lz_de = pic_info.lz_di * smime
    kwargs = {"current_time": current_time, "xl": 0, "xr": lx_de,
              "zb": -0.5 * lx_de, "zt": 0.5 * lx_de}
    fname = root_dir + "data/ne.gda"
    x, z, ne = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/Ay.gda"
    x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)

    fname = root_dir + "data/ex.gda"
    x, z, ex = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/ey.gda"
    x, z, ey = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/ez.gda"
    x, z, ez = read_2d_fields(pic_info, fname, **kwargs)

    fname = root_dir + "data/vex.gda"
    x, z, vex = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/vey.gda"
    x, z, vey = read_2d_fields(pic_info, fname, **kwargs)
    fname = root_dir + "data/vez.gda"
    x, z, vez = read_2d_fields(pic_info, fname, **kwargs)

    wpe_wce = pic_info.dtwce / pic_info.dtwpe
    va = wpe_wce / math.sqrt(pic_info.mime)
    vex /= va
    vey /= va
    vez /= va

    jdote = -ne * (ex * vex + ey * vey + ez * vez)
    ng = 3
    kernel = np.ones((ng, ng)) / float(ng * ng)
    jdote = signal.convolve2d(jdote, kernel)

    jdote /= pic_info.b0 * va
    
    xmin, xmax = np.min(x), np.max(x)
    zmin, zmax = np.min(z), np.max(z)

    # w0, h0 = 0.41, 0.11
    w0, h0 = 0.4, 0.42
    xs0, ys0 = 0.06, 0.98 - h0
    vgap, hgap = 0.03, 0.1

    def plot_one_field(fdata, ax, text, text_color, label_bottom='on',
                       label_left='on', ylabel=False, vmin=0, vmax=10,
                       colormap=plt.cm.seismic, xs=xs0, ys=ys0, ay_color='k',
                       color_bar=False):
        plt.tick_params(labelsize=16)
        p1 = ax.imshow(fdata, vmin=vmin, vmax=vmax, cmap=colormap,
                       extent=[xmin, xmax, zmin, zmax], aspect='auto',
                       origin='lower', interpolation='bicubic')
        ax.tick_params(axis='x', labelbottom=label_bottom)
        ax.tick_params(axis='y', labelleft=label_left)
        if ylabel:
            ax.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
        ax.contour(x, z, Ay, colors=ay_color, linewidths=0.5)
        ax.text(0.02, 0.85, text, color=text_color, fontsize=20,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)
        if color_bar:
            xs1 = xs + w0 * 1.02
            w1 = w0 * 0.04
            cax = fig.add_axes([xs1, ys, w1, h0])
            cbar = fig.colorbar(p1, cax=cax)
            cbar.ax.tick_params(labelsize=16)
            return (p1, cbar)
        else:
            return p1

    fig = plt.figure(figsize=[16, 8])
    xs, ys = xs0, ys0
    ax1 = fig.add_axes([xs, ys, w0, h0])
    text1 = r'$n_e$'
    print("min and max of electron density: %f %f" % (np.min(ne), np.max(ne)))
    nmin, nmax = 1.0, 4.0
    ax1.tick_params(axis='x', which='minor', direction='in')
    ax1.tick_params(axis='x', which='major', direction='in')
    ax1.tick_params(axis='y', which='minor', direction='in')
    ax1.tick_params(axis='y', which='major', direction='in')
    p1, cbar1 = plot_one_field(ne, ax1, text1, 'w', label_bottom='off',
                               label_left='on', ylabel=True, vmin=nmin,
                               vmax=nmax, colormap=plt.cm.viridis, xs=xs, ys=ys,
                               ay_color='w', color_bar=True)
    cbar1.set_ticks(np.arange(nmin, nmax + 0.5, 0.5))

    ys -= h0 + vgap
    ax2 = fig.add_axes([xs, ys, w0, h0])
    vmin, vmax = -0.1, 0.1
    text2 = r'$\boldsymbol{j}_e\cdot\boldsymbol{E}$'
    ax2.tick_params(axis='x', which='minor', direction='in')
    ax2.tick_params(axis='x', which='major', direction='in')
    ax2.tick_params(axis='y', which='minor', direction='in')
    ax2.tick_params(axis='y', which='major', direction='in')
    print("min and max of jdote: %f %f" % (np.min(jdote), np.max(jdote)))
    p2 = plot_one_field(jdote, ax2, text2, 'k', label_bottom='on',
                        label_left='on', ylabel=True, vmin=vmin, vmax=vmax,
                        colormap=plt.cm.seismic, xs=xs, ys=ys)
    ax2.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    xs1 = xs + w0 * 1.02
    w1 = w0 * 0.04
    cax2 = fig.add_axes([xs1, ys, w1, h0])
    cbar2 = fig.colorbar(p2, cax=cax2)
    cbar2.ax.tick_params(labelsize=16)

    xs = xs1 + w1 + hgap
    h1 = h0 * 2 + vgap
    w1 = 0.3
    ax3 = fig.add_axes([xs, ys, w0, h1])
    ax3.tick_params(axis='x', which='minor', direction='in')
    ax3.tick_params(axis='x', which='major', direction='in')
    ax3.tick_params(axis='y', which='minor', direction='in')
    ax3.tick_params(axis='y', which='major', direction='in')
    vth = pic_info.vthe
    gama = 1.0 / math.sqrt(1.0 - 3 * vth**2)
    eth = gama - 1.0

    fdir = '../data/spectra/' + run_name + '/'
    n0 = pic_info.nx * pic_info.ny * pic_info.nz * pic_info.nppc
    tratio = pic_info.particle_interval // pic_info.fields_interval
    nt = current_time // tratio
    
    fname = fdir + 'spectrum-e.1'
    elin, flin, elog, flog_e = spect_fit.get_energy_distribution(fname, n0)
    elog_norm_e = spect_fit.get_normalized_energy('e', elog, pic_info)
    f_intial = fitting_funcs.func_maxwellian(elog, n0, 1.5 / eth)
    nacc, eacc = spect_fit.accumulated_particle_info(elog, f_intial)
    f_intial /= nacc[-1]
    ax3.loglog(elog_norm_e, f_intial, linewidth=1, color='k',
               linestyle='--', label='initial')

    for ct in range(1, nt + 1):
        fname = fdir + 'spectrum-e.' + str(ct)
        elin, flin, elog, flog_e = spect_fit.get_energy_distribution(fname, n0)
        nacc, eacc = spect_fit.accumulated_particle_info(elog, flog_e)
        flog_e /= nacc[-1]

        if (ct != nt):
            ax3.loglog(elog_norm_e, flog_e, linewidth=1, color='k')
        else:
            ax3.loglog(elog_norm_e, flog_e, linewidth=3, color='r')

    ax3.set_xlim([5E-2, 5E2])
    ax3.set_ylim([1E-8, 2E2])
    ax3.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$',
                   fontdict=font, fontsize=20)
    ax3.set_ylabel(r'$f(\varepsilon)$', fontdict=font, fontsize=20)
    ax3.tick_params(labelsize=16)
    t_wci = current_time * pic_info.dt_fields
    title = r'$t\Omega_{ci} = ' + "{:10.1f}".format(t_wci) + '$'
    ax3.text(0.02, 0.05, title, color='k', fontsize=20,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='left', verticalalignment='center',
             transform=ax3.transAxes)

    fdir = '../img/img_apjl/ne_jdote/' + run_name + '/'
    mkdir_p(fdir)
    fname = fdir + 'ne_jdote_' + str(current_time) + '.jpg'
    fig.savefig(fname, dpi=200)
    plt.close()
    # plt.show()


def get_cmd_args():
    """Get command line arguments """
    default_run_name = 'mime25_beta002_guide00_frequent_dump'
    default_run_dir = '/net/scratch3/xiaocanli/reconnection/frequent_dump/' + \
            'mime25_beta002_guide00_frequent_dump/'
    parser = argparse.ArgumentParser(description='Compression analysis based on fluids')
    parser.add_argument('--species', action="store", default='e',
                        help='particle species')
    parser.add_argument('--run_dir', action="store", default=default_run_dir,
                        help='run directory')
    parser.add_argument('--run_name', action="store", default=default_run_name,
                        help='run name')
    parser.add_argument('--tframe_fields', action="store", default='30', type=int,
                        help='Time frame for fields')
    parser.add_argument('--multi_frames', action="store_true", default=False,
                        help='whether analyzing multiple frames')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_cmd_args()
    run_name = args.run_name
    run_dir = args.run_dir
    species = args.species
    tframe_fields = args.tframe_fields
    multi_frames = args.multi_frames
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    tratio = pic_info.particle_interval / pic_info.fields_interval
    # plot_ne_jdote(pic_info, run_dir, run_name, 30)
    cts = range(pic_info.ntf)
    def processInput(job_id):
        print job_id
        ct = job_id
        plot_ne_jdote(pic_info, run_dir, run_name, ct)
    # ncores = multiprocessing.cpu_count()
    ncores = 8
    Parallel(n_jobs=ncores)(delayed(processInput)(ct) for ct in cts)
