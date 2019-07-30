"""
Analysis procedures for the paper on high mass-ratio
"""
import argparse
import itertools
import math
import multiprocessing
import operator
import os.path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import palettable
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
from scipy import signal
from scipy.interpolate import interp1d, interp2d
from scipy.ndimage.filters import median_filter, gaussian_filter

import fitting_funcs
import pic_information
from contour_plots import read_2d_fields
from joblib import Parallel, delayed
from json_functions import read_data_from_json
from shell_functions import mkdir_p

plt.style.use("seaborn-deep")
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = \
[r"\usepackage{amsmath, bm}",
 r"\DeclareMathAlphabet{\mathsfit}{\encodingdefault}{\sfdefault}{m}{sl}",
 r"\SetMathAlphabet{\mathsfit}{bold}{\encodingdefault}{\sfdefault}{bx}{sl}",
 r"\newcommand{\tensorsym}[1]{\bm{\mathsfit{#1}}}"]
COLORS = palettable.colorbrewer.qualitative.Set1_9.mpl_colors

FONT = {'family': 'serif',
        'color': 'black',
        'weight': 'normal',
        'size': 24}


def div0(a, b):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0]

    From: http://stackoverflow.com/a/35696047/2561161

    """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[~np.isfinite(c)] = 0  # -inf inf NaN
    return c


def calc_reconnection_rate(run_dir, run_name):
    """Calculate reconnection rate.

    Args:
        run_dir: the run root directory
        run_name: PIC run name
    """
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    ntf = pic_info.ntf
    phi = np.zeros(ntf)
    fname = run_dir + 'data/Ay.gda'
    for tframe in range(ntf):
        kwargs = {"current_time": tframe,
                  "xl": 0, "xr": pic_info.lx_di,
                  "zb": -pic_info.lz_di*0.1, "zt": pic_info.lz_di*0.1}
        x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
        nz, = z.shape
        max_ay = np.max(Ay[nz // 2 - 1:nz // 2 + 1, :])
        min_ay = np.min(Ay[nz // 2 - 1:nz // 2 + 1, :])
        phi[tframe] = max_ay - min_ay
    nk = 3
    # phi = signal.medfilt(phi, kernel_size=nk)
    dtwpe = pic_info.dtwpe
    dtwce = pic_info.dtwce
    dtwci = pic_info.dtwci
    mime = pic_info.mime
    dtf_wpe = pic_info.dt_fields * dtwpe / dtwci
    reconnection_rate = np.gradient(phi) / dtf_wpe
    b0 = pic_info.b0
    va = dtwce * math.sqrt(1.0 / mime) / dtwpe
    reconnection_rate /= b0 * va
    # reconnection_rate[-1] = reconnection_rate[-2]
    tfields = pic_info.tfields

    return (tfields, reconnection_rate)


def calc_rrate_multi(mime, const_va=False):
    """Calculate reconnection rate for multiple runs

    Args:
        mime: ion to electron mass ratio
        const_va: whether the Alfven speed is constant for different mass ratio
    """
    if const_va and mime != 400:
        base_dir = "/net/scratch4/xiaocanli/reconnection/mime" + str(mime) + "_high/"
    elif mime == 400:
        base_dir = "/net/scratch4/xiaocanli/reconnection/mime" + str(mime) + "/"
    else:
        base_dir = "/net/scratch3/xiaocanli/reconnection/mime" + str(mime) + "/"
    for bguide in ["00", "02", "04", "08", '16', '32', '64']:
        run_name = 'mime' + str(mime) + '_beta002_' + 'bg' + str(bguide)
        if const_va and mime != 400:
            run_name += '_high'
        run_dir = base_dir + run_name + "/"
        tfields, rrate = calc_reconnection_rate(run_dir, run_name)
        odir = "../data/rate/"
        mkdir_p(odir)
        fname = odir + "rrate_" + run_name + ".dat"
        np.savetxt(fname, (tfields, rrate))


def onset_tframes(const_va=False):
    """Reconnection onset time frames
    Args:
        const_va: whether the Alfven speed is the same for different mass ratio
    """
    if const_va:
        tframes = {"25": 32,
                   "100": 35,
                   "400": 38}
    else:
        tframes = {"25": 29,
                   "100": 29,
                   "400": 38}
    return tframes


def shift_tframes(const_va=False):
    """Shifted time frames to match different mass ratio
    Args:
        const_va: whether the Alfven speed is the same for different mass ratio
    """
    begs = onset_tframes(const_va)
    beg_min = min(begs.values())
    tframes = {key: begs[key] - beg_min for key in begs.keys()}
    return tframes


def plot_rrate_multi(mime, const_va=False):
    """Plot reconnection rate for multiple runs
    """
    fig = plt.figure(figsize=[3.5, 2.5])
    ax = fig.add_axes([0.16, 0.16, 0.78, 0.78])
    ax.set_prop_cycle('color', COLORS)
    bgs = np.asarray([0, 0.2, 0.4, 0.8])
    ng = 3
    kernel = np.ones(ng) / float(ng)
    tmin = tmax = 0.0
    for bg in bgs:
        bg_str = str(int(bg * 10)).zfill(2)
        run_name = 'mime' + str(mime) + '_beta002_' + 'bg' + bg_str
        if const_va and mime != 400:
            run_name += '_high'
        fdir = "../data/rate/"
        fname = fdir + "rrate_" + run_name + ".dat"
        tfields, rrate = np.genfromtxt(fname)
        rrate = np.convolve(rrate, kernel, 'same')
        ltext = r"$B_g=" + str(bg) + "$"
        ax.plot(tfields, rrate, linewidth=1, label=ltext)
        tmin = min(tmin, tfields.min())
        tmax = max(tmin, tfields.max())
    ax.set_ylim([0, 0.13])
    if mime == 400:
        ax.set_xlim([0, 120])
    else:
        if const_va:
            ax.set_xlim([0, 120])
        else:
            ax.set_xlim([0, 200])
    begs = onset_tframes(const_va)
    beg = begs[str(mime)]
    ax.plot([beg, beg], ax.get_ylim(), color='k', linestyle='--',
            linewidth=0.5)
    text1 = r'$m_i/m_e = ' + str(mime) + '$'
    ypos = 0.02 if mime == 25 else 0.9
    ax.text(0.97, ypos, text1, color='k', fontsize=10,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='right', verticalalignment='bottom',
            transform=ax.transAxes)
    text2 = r'$t\Omega_{ci} = ' + str(beg) + '$'
    if const_va:
        if mime == 25:
            xpos = 0.27
        elif mime == 100:
            xpos = 0.29
        else:
            xpos = 0.31
    else:
        xpos = 0.15
    ax.text(xpos, 0.6, text2, color='k', fontsize=10, rotation=90,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='right', verticalalignment='bottom',
            transform=ax.transAxes)
    if mime == 25:
        ax.legend(loc=1, prop={'size': 8}, ncol=1,
                  shadow=False, fancybox=False, frameon=False)
    ax.tick_params(bottom=True, top=False, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top=True)
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.tick_params(labelsize=8)
    ax.set_xlabel(r'$t\Omega_{ci}$', fontdict=FONT, fontsize=10)
    ax.set_ylabel(r'$E_R$', fontdict=FONT, fontsize=10)
    fdir = '../img/rate/'
    mkdir_p(fdir)
    if const_va:
        fname = fdir + 'rrate_' + str(mime) + '_high.pdf'
    else:
        fname = fdir + 'rrate_' + str(mime) + '.pdf'
    fig.savefig(fname)
    plt.show()


def plot_rrate_mime(bg, const_va=False):
    """Plot reconnection rate for runs with the same guide field
    """
    mimes = np.asarray([25, 100, 400])
    ng = 3
    kernel = np.ones(ng) / float(ng)
    tmin = tmax = 0.0
    bg_str = str(int(bg * 10)).zfill(2)
    tshifts = shift_tframes(const_va)
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    fig = plt.figure(figsize=[3.5, 2.5])
    ax = fig.add_axes([0.16, 0.16, 0.78, 0.78])
    ax.set_prop_cycle('color', COLORS)
    for mime in mimes:
        run_name = 'mime' + str(mime) + '_beta002_' + 'bg' + bg_str
        if const_va and mime != 400:
            run_name += '_high'
        fdir = "../data/rate/"
        fname = fdir + "rrate_" + run_name + ".dat"
        tfields, rrate = np.genfromtxt(fname)
        # tfields -= tshifts[str(mime)]
        # rrate = np.convolve(rrate, kernel, 'same')
        ltext = r'$m_i/m_e = ' + str(mime) + '$'
        ax.plot(tfields, rrate, linewidth=1, label=ltext)
        tmin = min(tmin, tfields.min())
        tmax = max(tmin, tfields.max())
    ax.set_ylim([0, 0.13])
    if mime == 400:
        ax.set_xlim([0, 120])
    else:
        if const_va:
            ax.set_xlim([0, 120])
        else:
            ax.set_xlim([0, 200])
    text1 = r"$B_g=" + str(bg) + "$"
    ax.text(0.97, 0.9, text1, color='k', fontsize=10,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='right', verticalalignment='bottom',
            transform=ax.transAxes)
    if bg == 0.0:
        # ax.legend(loc=8, prop={'size': 8}, ncol=1,
        #           shadow=False, fancybox=False, frameon=False)
        ax.text(0.7, 0.21, r'$m_i/m_e=25$', color=COLORS[0], fontsize=8,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='bottom',
                transform=ax.transAxes)
        ax.text(0.7, 0.13, r'$m_i/m_e=100$', color=COLORS[1], fontsize=8,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='bottom',
                transform=ax.transAxes)
        ax.text(0.7, 0.05, r'$m_i/m_e=400$', color=COLORS[2], fontsize=8,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='bottom',
                transform=ax.transAxes)
    ax.tick_params(bottom=True, top=False, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top=True)
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.tick_params(labelsize=8)
    ax.set_xlabel(r'$t\Omega_{ci}$', fontdict=FONT, fontsize=10)
    ax.set_ylabel(r'$E_R$', fontdict=FONT, fontsize=10)
    fdir = '../img/rate/'
    mkdir_p(fdir)
    if const_va:
        fname = fdir + 'rrate_bg' + bg_str + '_high.pdf'
    else:
        fname = fdir + 'rrate_bg' + bg_str + '.pdf'
    fig.savefig(fname)
    plt.show()


def energy_evolution(bg, const_va=False):
    """Plot energy evolution for runs with the same guide field

    Args:
        bg: guide field strength
        const_va: whether the Alfven speed is constant for different mass ratio
    """
    bg_str = str(int(bg * 10)).zfill(2)
    mimes = np.asarray([25, 100, 400])
    fig = plt.figure(figsize=[3.5, 2.5])
    ax = fig.add_axes([0.13, 0.15, 0.8, 0.8])
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    ax.set_prop_cycle('color', COLORS)
    tshifts = shift_tframes(const_va)
    for mime in mimes:
        run_name = 'mime' + str(mime) + '_beta002_' + 'bg' + bg_str
        if const_va and mime != 400:
            run_name += '_high'
        picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        tenergy = pic_info.tenergy
        tenergy -= tshifts[str(mime)]
        ene_electric = pic_info.ene_electric
        ene_magnetic = pic_info.ene_magnetic
        kene_e = pic_info.kene_e
        kene_i = pic_info.kene_i
        ene_bx = pic_info.ene_bx
        ene_by = pic_info.ene_by
        ene_bz = pic_info.ene_bz

        enorm = ene_bx[0]

        tindex, t0 = find_nearest(tenergy, 100)
        ltext = r"$m_i/m_e=" + str(mime) + "$"
        p1, = ax.plot(tenergy[:tindex+1],
                      (ene_magnetic[:tindex+1] - ene_magnetic[0]) / enorm,
                      linewidth=1, label=ltext)
        p2, = ax.plot(tenergy[:tindex+1],
                      (kene_i[:tindex+1] - kene_i[0]) / enorm,
                      color=p1.get_color(), linestyle='--', linewidth=1)
        p3, = ax.plot(tenergy[:tindex+1],
                      (kene_e[:tindex+1] - kene_e[0]) / enorm,
                      color=p1.get_color(), linestyle='-.', linewidth=1)
        dkm = ene_magnetic[tindex] - ene_magnetic[0]
        dke = kene_e[tindex] - kene_e[0]
        dki = kene_i[tindex] - kene_i[0]
        print(dkm / enorm)
        print(dke / enorm, dke / dkm)
        print(dki / enorm, dki / dkm)
    ax.set_xlim([0, 100])
    if const_va:
        if bg == 0.0:
            ypos = [0.72, 0.55, 0.35]
            angle = [30, 10, -30]
        elif bg == 0.2:
            ypos = [0.70, 0.52, 0.35]
            angle = [25, 7, -30]
        elif bg == 0.4:
            ypos = [0.72, 0.55, 0.33]
            angle = [25, 10, -30]
        elif bg == 0.8:
            ypos = [0.70, 0.55, 0.37]
            angle = [20, 10, -25]
        elif bg == 1.6:
            ypos = [0.68, 0.53, 0.35]
            angle = [20, 10, -25]
        elif bg == 3.2:
            ypos = [0.68, 0.53, 0.36]
            angle = [17, 10, -15]
        elif bg == 6.4:
            ypos = [0.73, 0.56, 0.40]
            angle = [0, 0, 0]
    else:
        if bg == 0.0:
            ypos = [0.75, 0.53, 0.28]
            angle = [30, 10, -30]
        elif bg == 0.2:
            ypos = [0.73, 0.54, 0.32]
            angle = [25, 7, -30]
        elif bg == 0.4:
            ypos = [0.75, 0.55, 0.30]
            angle = [25, 10, -30]
        elif bg == 0.8:
            ypos = [0.75, 0.55, 0.28]
            angle = [25, 10, -30]
    ax.text(0.5, ypos[0], r'$\Delta K_i/\varepsilon_{Bx0}$', color='k',
            rotation=angle[0], fontsize=8,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)
    ax.text(0.5, ypos[1], r'$\Delta K_e/\varepsilon_{Bx0}$', color='k',
            rotation=angle[1], fontsize=8,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)
    ax.text(0.5, ypos[2], r'$\Delta \varepsilon_{B}/\varepsilon_{Bx0}$',
            color='k', rotation=angle[2], fontsize=8,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)
    ax.text(0.03, 0.21, r'$m_i/m_e=25$', color=COLORS[0], fontsize=8,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes)
    ax.text(0.03, 0.13, r'$m_i/m_e=100$', color=COLORS[1], fontsize=8,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes)
    ax.text(0.03, 0.05, r'$m_i/m_e=400$', color=COLORS[2], fontsize=8,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes)
    ax.text(0.03, 0.85, r'$B_g=' + str(bg) + '$', color='k', fontsize=10,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top=True)
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.tick_params(labelsize=8)
    ax.set_xlabel(r'$t\Omega_{ci}$', fontdict=FONT, fontsize=10)
    # ax.legend(loc=3, prop={'size': 10}, ncol=1,
    #           shadow=False, fancybox=False, frameon=False)
    fdir = '../img/img_high_mime/'
    mkdir_p(fdir)
    if const_va:
        fname = fdir + 'econv_' + bg_str + '_high.pdf'
    else:
        fname = fdir + 'econv_' + bg_str + '.pdf'
    fig.savefig(fname)
    plt.show()


def energy_evolution_fraction():
    """Plot energy evolution fraction for all runs

    Args:
        bg: guide field strength
    """
    bg_str = str(int(bg * 10)).zfill(2)
    mimes = np.asarray([25, 100, 400])
    fig = plt.figure(figsize=[3.5, 2.5])
    ax = fig.add_axes([0.13, 0.15, 0.8, 0.8])
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    ax.set_prop_cycle('color', COLORS)
    for mime in mimes:
        run_name = "mime" + str(mime) + "_beta002_bg" + bg_str
        picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        tenergy = pic_info.tenergy
        if mime == 400:
            tenergy += 12
        ene_electric = pic_info.ene_electric
        ene_magnetic = pic_info.ene_magnetic
        kene_e = pic_info.kene_e
        kene_i = pic_info.kene_i
        ene_bx = pic_info.ene_bx
        ene_by = pic_info.ene_by
        ene_bz = pic_info.ene_bz

        enorm = ene_bx[0]

        ltext = r"$m_i/m_e=" + str(mime) + "$"
        p1, = ax.plot(tenergy, (ene_magnetic - ene_magnetic[0]) / enorm,
                      linewidth=1, label=ltext)
        p2, = ax.plot(tenergy, (kene_i - kene_i[0]) / enorm,
                      color=p1.get_color(), linestyle='--', linewidth=1)
        p3, = ax.plot(tenergy, (kene_e - kene_e[0]) / enorm,
                      color=p1.get_color(), linestyle='-.', linewidth=1)
    ax.set_xlim([0, 200])
    ylim = ax.get_ylim()
    ylen = ylim[1] - ylim[0]
    height1 = ((kene_i[-1] - kene_i[0]) / enorm - ylim[0]) / ylen + 0.08
    height2 = ((kene_e[-1] - kene_e[0]) / enorm - ylim[0]) / ylen - 0.08
    height3 = ((ene_magnetic[-1] - ene_magnetic[0]) / enorm - ylim[0]) / ylen - 0.1
    ax.text(0.5, height1, r'$\Delta K_i/\varepsilon_{Bx0}$', color='k',
            rotation=15, fontsize=10,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)
    ax.text(0.5, height2, r'$\Delta K_e/\varepsilon_{Bx0}$', color='k', fontsize=10,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)
    ax.text(0.5, height3, r'$\Delta \varepsilon_{B}/\varepsilon_{Bx0}$',
            color='k', rotation=-20, fontsize=10,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)
    ax.text(0.03, 0.25, r'$m_i/m_e=25$', color=COLORS[0], fontsize=8,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes)
    ax.text(0.03, 0.15, r'$m_i/m_e=100$', color=COLORS[1], fontsize=8,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes)
    ax.text(0.03, 0.05, r'$m_i/m_e=400$', color=COLORS[2], fontsize=8,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes)
    ax.text(0.03, 0.85, r'$B_g=' + str(bg) + '$', color='k', fontsize=10,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top=True)
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.tick_params(labelsize=8)
    ax.set_xlabel(r'$t\Omega_{ci}$', fontdict=FONT, fontsize=10)
    # ax.legend(loc=3, prop={'size': 10}, ncol=1,
    #           shadow=False, fancybox=False, frameon=False)
    fdir = '../img/img_high_mime/'
    mkdir_p(fdir)
    fname = fdir + 'econv_' + bg_str + '.pdf'
    fig.savefig(fname)
    plt.show()


def get_tmin_bg(bg, const_va):
    """Get the minimum time frame for a single guide field

    Args:
        bg: guide field strength
        const_va: whether the Alfven speed is constant for different mass ratio
    """
    mimes = np.asarray([25, 100, 400])
    bgs = np.asarray([0.0, 0.2, 0.4, 0.8])
    tmin = 1000.0
    tshifts = shift_tframes(const_va)
    print("Guide field: %0.1f" % bg)
    for imime, mime in enumerate(mimes):
        bg_str = str(int(bg * 10)).zfill(2)
        run_name = 'mime' + str(mime) + '_beta002_' + 'bg' + bg_str
        if const_va and mime != 400:
            run_name += '_high'
        picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        tenergy = pic_info.tenergy
        tenergy -= tshifts[str(mime)]
        tmin = min(tmin, tenergy.max())
    return tmin


def get_tmin(const_va):
    """Get the minimum time frame

    Args:
        const_va: whether the Alfven speed is constant for different mass ratio
    """
    mimes = np.asarray([25, 100, 400])
    bgs = np.asarray([0.0, 0.2, 0.4, 0.8])
    tmin = 1000.0
    tshifts = shift_tframes(const_va)
    for ibg, bg in enumerate(bgs):
        tmin = min(tmin, get_tmin_bg(bg, const_va))
    return tmin


def energy_conversion(const_va=False, high_bg=False):
    """Plot energy conversion rate for all runs
    Args:
        const_va: whether the Alfven speed is constant for different mass ratio
        high_bg: whether to include runs with higher guide field
    """
    mimes = np.asarray([25, 100, 400])
    if high_bg:
        bgs = np.asarray([0.0, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4])
    else:
        bgs = np.asarray([0.0, 0.2, 0.4, 0.8])
    nmime, = mimes.shape
    nbg, = bgs.shape

    tshifts = shift_tframes(const_va)
    tmin = get_tmin(const_va)

    # Second pass to get the energy conversion at tmin
    econv_rates = np.zeros((nmime, nbg, 7))
    for imime, mime in enumerate(mimes):
        for ibg, bg in enumerate(bgs):
            print("Guide field: %0.1f" % bg)
            bg_str = str(int(bg * 10)).zfill(2)
            run_name = 'mime' + str(mime) + '_beta002_' + 'bg' + bg_str
            if const_va and mime != 400:
                run_name += '_high'
            picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
            pic_info = read_data_from_json(picinfo_fname)
            tenergy = pic_info.tenergy
            tfields = pic_info.tfields
            tenergy -= tshifts[str(mime)]
            tfields -= tshifts[str(mime)]
            tindex_e, t0 = find_nearest(tenergy, tmin)
            tindex_f, t0 = find_nearest(tfields, tmin)
            fname = "../data/bulk_internal_energy/" + run_name + "/"
            fname += "bulk_internal_energy_e.dat"
            fdata = np.fromfile(fname, dtype=np.float32)
            sz, = fdata.shape
            nframes = (sz//2)//4
            bene_e4 = fdata[:sz//2].reshape(nframes, -1)
            iene_e4 = fdata[sz//2:].reshape(nframes, -1)
            fname = "../data/bulk_internal_energy/" + run_name + "/"
            fname += "bulk_internal_energy_i.dat"
            fdata = np.fromfile(fname, dtype=np.float32)
            bene_i4 = fdata[:sz//2].reshape(nframes, -1)
            iene_i4 = fdata[sz//2:].reshape(nframes, -1)
            bene_e = bene_e4[:, -1]
            iene_e = iene_e4[:, -1]
            bene_i = bene_i4[:, -1]
            iene_i = iene_i4[:, -1]
            ene_magnetic = pic_info.ene_magnetic
            kene_e = pic_info.kene_e
            kene_i = pic_info.kene_i
            ene_magnetic = pic_info.ene_magnetic
            econv_rates[imime, ibg, 0] = (ene_magnetic[tindex_e] -
                                          ene_magnetic[0]) / ene_magnetic[0]
            econv_rates[imime, ibg, 1] = (kene_e[tindex_e] - kene_e[0]) / ene_magnetic[0]
            econv_rates[imime, ibg, 2] = (kene_i[tindex_e] - kene_i[0]) / ene_magnetic[0]
            econv_rates[imime, ibg, 3] = (iene_e[tindex_f] - iene_e[0]) / ene_magnetic[0]
            econv_rates[imime, ibg, 4] = (iene_i[tindex_f] - iene_i[0]) / ene_magnetic[0]
            econv_rates[imime, ibg, 5] = (bene_e[tindex_f] - bene_e[0]) / ene_magnetic[0]
            econv_rates[imime, ibg, 6] = (bene_i[tindex_f] - bene_i[0]) / ene_magnetic[0]
            print("Energy conversion:", econv_rates[imime, ibg, :])

    if high_bg:
        fig = plt.figure(figsize=[7.0, 2.5])
        rect = [0.06, 0.15, 0.4, 0.8]
        ax = fig.add_axes(rect)
    else:
        fig = plt.figure(figsize=[3.5, 2.5])
        ax = fig.add_axes([0.13, 0.15, 0.8, 0.8])
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    for imime, mime in enumerate(mimes):
        ax.plot(bgs[:4], econv_rates[imime, :4, 0], marker='v', markersize=4,
                linestyle='-', linewidth=1, color=COLORS[imime])
        ax.plot(bgs[:4], econv_rates[imime, :4, 1], marker='x', markersize=4,
                linestyle='-', linewidth=1, color=COLORS[imime])
        ax.plot(bgs[:4], econv_rates[imime, :4, 2], marker='o', markersize=4,
                linestyle='-', linewidth=1, color=COLORS[imime])
    print("Ion energy gain / Electron energy gain:")
    print(econv_rates[:, :, 2]/econv_rates[:, :, 1])
    print("Ion internal energy gain / Electron internal energy gain:")
    print(econv_rates[:, :, 4]/econv_rates[:, :, 3])
    print("Ion bulk energy gain:")
    print(econv_rates[:, :, 6])

    ax.text(0.6, 0.80, r'$\Delta K_i/\varepsilon_{Bx0}$', color='k',
            rotation=-10, fontsize=8,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)
    ax.text(0.3, 0.6, r'$\Delta K_e/\varepsilon_{Bx0}$', color='k', fontsize=8,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)
    ax.text(0.6, 0.24, r'$\Delta \varepsilon_{B}/\varepsilon_{Bx0}$',
            color='k', rotation=15, fontsize=8,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)
    ax.set_xlim([-0.05, 0.85])
    ax.plot(ax.get_xlim(), [0, 0], color='k', linewidth=0.5, linestyle='--')
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top=True)
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlabel(r'$B_g$', fontsize=10)
    ax.tick_params(labelsize=8)

    # # embedded plot for internal energy
    # ax1 = fig.add_axes([0.23, 0.38, 0.27, 0.27])
    # for imime, mime in enumerate(mimes):
    #     ax1.plot(bgs, econv_rates[imime, :, 4], marker='x', markersize=4,
    #              linestyle='--', linewidth=1, color=COLORS[imime])
    #     ax1.plot(bgs, econv_rates[imime, :, 6], marker='x', markersize=4,
    #              linestyle='-.', linewidth=1, color=COLORS[imime])
    # ax1.tick_params(bottom=True, top=False, left=True, right=True)
    # ax1.tick_params(axis='x', which='minor', direction='in', top=True)
    # ax1.tick_params(axis='x', which='major', direction='in')
    # ax1.tick_params(axis='y', which='minor', direction='in')
    # ax1.tick_params(axis='y', which='major', direction='in')
    # ax1.tick_params(axis='x', labelbottom=False)
    # ax1.tick_params(labelsize=6)
    # ax1.set_xlim(ax.get_xlim())
    # ax1.set_ylim([0, 0.06])
    # ax1.text(0.5, 0.75, 'internal', color='k', fontsize=6,
    #          bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
    #          horizontalalignment='left', verticalalignment='bottom',
    #          transform=ax1.transAxes)

    # additional plot for higher guide field
    if high_bg:
        rect[0] += rect[2] + 0.1
        ax1 = fig.add_axes(rect)
        for imime, mime in enumerate(mimes):
            ax1.plot(bgs[4:], econv_rates[imime, 4:, 0], marker='v', markersize=4,
                     linestyle='-', linewidth=1, color=COLORS[imime])
            ax1.plot(bgs[4:], econv_rates[imime, 4:, 1], marker='x', markersize=4,
                     linestyle='-', linewidth=1, color=COLORS[imime])
            ax1.plot(bgs[4:], econv_rates[imime, 4:, 2], marker='o', markersize=4,
                     linestyle='-', linewidth=1, color=COLORS[imime])
        ax1.tick_params(bottom=True, top=False, left=True, right=True)
        ax1.tick_params(axis='x', which='minor', direction='in', top=True)
        ax1.tick_params(axis='x', which='major', direction='in')
        ax1.tick_params(axis='y', which='minor', direction='in')
        ax1.tick_params(axis='y', which='major', direction='in')
        ax1.tick_params(labelsize=6)
        ax1.set_xlim([1.5, 6.5])
        ax1.plot(ax1.get_xlim(), [0, 0], color='k', linewidth=0.5, linestyle='--')
        ax1.tick_params(bottom=True, top=True, left=True, right=True)
        ax1.tick_params(axis='x', which='minor', direction='in', top=True)
        ax1.tick_params(axis='x', which='major', direction='in')
        ax1.tick_params(axis='y', which='minor', direction='in')
        ax1.tick_params(axis='y', which='major', direction='in')
        ax1.set_xlabel(r'$B_g$', fontsize=10)
        ax1.tick_params(labelsize=8)
        ax1.text(0.6, 0.63, r'$\Delta K_i/\varepsilon_{Bx0}$', color='k',
                 rotation=-5, fontsize=8,
                 bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                 horizontalalignment='left', verticalalignment='center',
                 transform=ax1.transAxes)
        ax1.text(0.1, 0.55, r'$\Delta K_e/\varepsilon_{Bx0}$', color='k',
                 rotation=-10, fontsize=8,
                 bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                 horizontalalignment='left', verticalalignment='center',
                 transform=ax1.transAxes)
        ax1.text (0.6, 0.43, r'$\Delta \varepsilon_{B}/\varepsilon_{Bx0}$',
                 color='k', rotation=10, fontsize=8,
                 bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                 horizontalalignment='left', verticalalignment='center',
                 transform=ax1.transAxes)

    fdir = '../img/img_high_mime/'
    mkdir_p(fdir)
    if const_va:
        if high_bg:
            fname = fdir + 'econvs_high_bg_high.pdf'
        else:
            fname = fdir + 'econvs_high.pdf'
    else:
        if high_bg:
            fname = fdir + 'econvs_high_bg.pdf'
        else:
            fname = fdir + 'econvs.pdf'
    fig.savefig(fname)

    # bulk and internal energies for ions
    if high_bg:
        fig = plt.figure(figsize=[7.0, 2.5])
        rect = [0.06, 0.15, 0.4, 0.8]
        ax = fig.add_axes(rect)
    else:
        fig = plt.figure(figsize=[3.5, 2.5])
        ax = fig.add_axes([0.13, 0.15, 0.8, 0.8])
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    fdata = np.zeros(len(bgs)) - 10
    for imime, mime in enumerate(mimes):
        ax.plot(bgs[:4], econv_rates[imime, :4, 4], marker='o', markersize=4,
                linestyle='--', linewidth=1, color=COLORS[imime])
        ax.plot(bgs[:4], econv_rates[imime, :4, 6], marker='o', markersize=4,
                linestyle='-', linewidth=1, color=COLORS[imime])
        if imime == 0:
            ax.plot(bgs, fdata, linestyle='--', marker='o', markersize=4,
                    linewidth=1, color='k', label='ion internal')
            ax.plot(bgs, fdata, linestyle='-', marker='o', markersize=4,
                    linewidth=1, color='k', label='ion bulk')
    ax.legend(loc=3, prop={'size': 10}, ncol=1,
              shadow=False, fancybox=False, frameon=False)
    ax.tick_params(bottom=True, top=False, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top=True)
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlabel(r'$B_g$', fontsize=10)
    ax.tick_params(labelsize=8)
    ax.set_xlim([-0.05, 0.85])
    ax.set_ylim([0, 0.06])

    # higher guide field
    if high_bg:
        rect[0] += rect[2] + 0.1
        ax1 = fig.add_axes(rect)
        for imime, mime in enumerate(mimes):
            ax1.plot(bgs[4:], econv_rates[imime, 4:, 4], marker='o', markersize=4,
                     linestyle='--', linewidth=1, color=COLORS[imime])
            ax1.plot(bgs[4:], econv_rates[imime, 4:, 6], marker='o', markersize=4,
                     linestyle='-', linewidth=1, color=COLORS[imime])
        ax1.tick_params(bottom=True, top=False, left=True, right=True)
        ax1.tick_params(axis='x', which='minor', direction='in', top=True)
        ax1.tick_params(axis='x', which='major', direction='in')
        ax1.tick_params(axis='y', which='minor', direction='in')
        ax1.tick_params(axis='y', which='major', direction='in')
        ax1.set_xlabel(r'$B_g$', fontsize=10)
        ax1.tick_params(labelsize=8)
        ax1.set_xlim([1.5, 6.5])

    if const_va:
        if high_bg:
            fname = fdir + 'bulk_internal_i_high_bg_high.pdf'
        else:
            fname = fdir + 'bulk_internal_i_high.pdf'
    else:
        if high_bg:
            fname = fdir + 'bulk_internal_i_high_bg.pdf'
        else:
            fname = fdir + 'bulk_internal_i.pdf'
    fig.savefig(fname)
    plt.show()


def internal_energy_conversion(const_va=False):
    """Plot internal energy conversion rate for all runs
    """
    mimes = np.asarray([25, 100, 400])
    bgs = np.asarray([0.0, 0.2, 0.4, 0.8])
    nmime, = mimes.shape
    nbg, = bgs.shape
    tmin = get_tmin(const_va)
    # Second pass to get the energy conversion at tmin
    econv_rates = np.zeros((nmime, nbg, 7))
    tshifts = shift_tframes(const_va)
    for imime, mime in enumerate(mimes):
        for ibg, bg in enumerate(bgs):
            print("Guide field: %0.1f" % bg)
            bg_str = str(int(bg * 10)).zfill(2)
            run_name = 'mime' + str(mime) + '_beta002_' + 'bg' + bg_str
            if const_va and mime != 400:
                run_name += '_high'
            picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
            pic_info = read_data_from_json(picinfo_fname)
            tenergy = pic_info.tenergy
            tfields = pic_info.tfields
            tenergy -= tshifts[str(mime)]
            tfields -= tshifts[str(mime)]
            tindex_e, t0 = find_nearest(tenergy, tmin)
            tindex_f, t0 = find_nearest(tfields, tmin)
            fname = "../data/bulk_internal_energy/" + run_name + "/"
            fname += "bulk_internal_energy_e.dat"
            fdata = np.fromfile(fname, dtype=np.float32)
            sz, = fdata.shape
            nframes = (sz//2)//4
            bene_e4 = fdata[:sz//2].reshape(nframes, -1)
            iene_e4 = fdata[sz//2:].reshape(nframes, -1)
            fname = "../data/bulk_internal_energy/" + run_name + "/"
            fname += "bulk_internal_energy_i.dat"
            fdata = np.fromfile(fname, dtype=np.float32)
            bene_i4 = fdata[:sz//2].reshape(nframes, -1)
            iene_i4 = fdata[sz//2:].reshape(nframes, -1)
            bene_e = bene_e4[:, -1]
            iene_e = iene_e4[:, -1]
            bene_i = bene_i4[:, -1]
            iene_i = iene_i4[:, -1]
            ene_magnetic = pic_info.ene_magnetic
            kene_e = pic_info.kene_e
            kene_i = pic_info.kene_i
            ene_magnetic = pic_info.ene_magnetic
            econv_rates[imime, ibg, 0] = (ene_magnetic[tindex_e] -
                                          ene_magnetic[0]) / ene_magnetic[0]
            econv_rates[imime, ibg, 1] = (kene_e[tindex_e] - kene_e[0]) / ene_magnetic[0]
            econv_rates[imime, ibg, 2] = (kene_i[tindex_e] - kene_i[0]) / ene_magnetic[0]
            econv_rates[imime, ibg, 3] = (iene_e[tindex_f] - iene_e[0]) / ene_magnetic[0]
            econv_rates[imime, ibg, 4] = (iene_i[tindex_f] - iene_i[0]) / ene_magnetic[0]
            econv_rates[imime, ibg, 5] = (bene_e[tindex_f] - bene_e[0]) / ene_magnetic[0]
            econv_rates[imime, ibg, 6] = (bene_i[tindex_f] - bene_i[0]) / ene_magnetic[0]
            print("Energy conversion:", econv_rates[imime, ibg, :])

    fig = plt.figure(figsize=[3.5, 2.5])
    ax = fig.add_axes([0.13, 0.15, 0.8, 0.8])
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    for imime, mime in enumerate(mimes):
        ax.plot(bgs, econv_rates[imime, :, 0], marker='v', markersize=4,
                linestyle='-', linewidth=1, color=COLORS[imime])
        ax.plot(bgs, econv_rates[imime, :, 3], marker='o', markersize=4,
                linestyle='-', linewidth=1, color=COLORS[imime])
        ax.plot(bgs, econv_rates[imime, :, 4], marker='x', markersize=4,
                linestyle='-', linewidth=1, color=COLORS[imime])
    ax.text(0.6, 0.80, r'$\Delta U_i/\varepsilon_{Bx0}$', color='k',
            rotation=-5, fontsize=8,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)
    ypos = 0.7 if const_va else 0.68
    ax.text(0.2, ypos, r'$\Delta U_e/\varepsilon_{Bx0}$', color='k', fontsize=8,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)
    ypos = 0.32 if const_va else 0.3
    ax.text(0.6, ypos, r'$\Delta \varepsilon_{B}/\varepsilon_{Bx0}$',
            color='k', rotation=20, fontsize=8,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)
    ax.text(0.03, 0.45, r'$m_i/m_e=25$', color=COLORS[0], fontsize=8,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes)
    ax.text(0.03, 0.37, r'$m_i/m_e=100$', color=COLORS[1], fontsize=8,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes)
    ax.text(0.03, 0.29, r'$m_i/m_e=400$', color=COLORS[2], fontsize=8,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes)
    ax.set_xlim([-0.05, 0.85])
    ax.plot(ax.get_xlim(), [0, 0], color='k', linewidth=0.5, linestyle='--')
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top=True)
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlabel(r'$B_g$', fontsize=10)
    ax.tick_params(labelsize=8)
    fdir = '../img/img_high_mime/'
    mkdir_p(fdir)
    if const_va:
        fname = fdir + 'econvs_internal_high.pdf'
    else:
        fname = fdir + 'econvs_internal.pdf'
    fig.savefig(fname)
    plt.show()


def get_bulk_inernal(run_name):
    """Get bulk and internal energizes

    Args:
        run_name: PIC run name
    """
    fname = "../data/bulk_internal_energy/" + run_name + "/"
    fname += "bulk_internal_energy_e.dat"
    fdata = np.fromfile(fname, dtype=np.float32)
    sz, = fdata.shape
    nframes = (sz//2)//4
    bene_e4 = fdata[:sz//2].reshape(nframes, -1)
    iene_e4 = fdata[sz//2:].reshape(nframes, -1)
    fname = "../data/bulk_internal_energy/" + run_name + "/"
    fname += "bulk_internal_energy_i.dat"
    fdata = np.fromfile(fname, dtype=np.float32)
    bene_i4 = fdata[:sz//2].reshape(nframes, -1)
    iene_i4 = fdata[sz//2:].reshape(nframes, -1)
    bene_e = bene_e4[:, -1]
    iene_e = iene_e4[:, -1]
    bene_i = bene_i4[:, -1]
    iene_i = iene_i4[:, -1]
    return (bene_e, iene_e, bene_i, iene_i)


def internal_energy_evolution(bg, const_va):
    """Plot internal energy evolution for runs with the same guide field

    Args:
        bg: guide field strength
        const_va: whether the Alfven speed is constant for different mass ratio
    """
    bg_str = str(int(bg * 10)).zfill(2)
    mimes = np.asarray([25, 100, 400])
    fig = plt.figure(figsize=[3.5, 2.5])
    ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    ax.set_prop_cycle('color', COLORS)
    tshifts = shift_tframes(const_va)
    tmin = get_tmin_bg(bg, const_va)
    for mime in mimes:
        run_name = 'mime' + str(mime) + '_beta002_' + 'bg' + bg_str
        if const_va and mime != 400:
            run_name += '_high'
        picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        tenergy = pic_info.tenergy
        tfields = pic_info.tfields
        tenergy -= tshifts[str(mime)]
        tfields -= tshifts[str(mime)]
        tindex_e, t0 = find_nearest(tenergy, tmin)
        tindex_f, t0 = find_nearest(tfields, tmin)
        ene_electric = pic_info.ene_electric
        ene_magnetic = pic_info.ene_magnetic
        kene_e = pic_info.kene_e
        kene_i = pic_info.kene_i
        ene_bx = pic_info.ene_bx
        ene_by = pic_info.ene_by
        ene_bz = pic_info.ene_bz
        bene_e, iene_e, bene_i, iene_i = get_bulk_inernal(run_name)

        enorm = ene_bx[0]

        ltext = r"$m_i/m_e=" + str(mime) + "$"
        # p1, = ax.plot(tenergy[:tindex_e],
        #               (ene_magnetic[:tindex_e] - ene_magnetic[0]) / enorm,
        #               linewidth=1, label=ltext)
        p2, = ax.plot(tfields[:tindex_f],
                      (iene_i[:tindex_f] - iene_i[0]) / enorm,
                      linestyle='--', linewidth=1)
        p3, = ax.plot(tfields[:tindex_f],
                      (iene_e[:tindex_f] - iene_e[0]) / enorm,
                      color=p2.get_color(), linestyle='-.', linewidth=1)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top=True)
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlim([0, 100])
    ylim = ax.get_ylim()
    ylen = ylim[1] - ylim[0]
    # if const_va:
    #     if bg == 0.0:
    #         ypos = [0.72, 0.72, 0.38]
    #         angle = [25, 10, -35]
    #     elif bg == 0.2:
    #         ypos = [0.72, 0.72, 0.38]
    #         angle = [25, 10, -35]
    #     elif bg == 0.4:
    #         ypos = [0.75, 0.75, 0.4]
    #         angle = [15, 10, -35]
    #     elif bg == 0.8:
    #         ypos = [0.82, 0.82, 0.55]
    #         angle = [15, 10, -35]
    # else:
    #     if bg == 0.0:
    #         ypos = [0.75, 0.7, 0.32]
    #         angle = [20, 10, -35]
    #     elif bg == 0.2:
    #         ypos = [0.75, 0.7, 0.32]
    #         angle = [20, 10, -35]
    #     elif bg == 0.4:
    #         ypos = [0.78, 0.78, 0.4]
    #         angle = [20, 10, -35]
    #     elif bg == 0.8:
    #         ypos = [0.8, 0.8, 0.38]
    #         angle = [20, 10, -35]
    if const_va:
        if bg == 0.0:
            ypos = [0.3, 0.25, 0.32]
            angle = [55, 30, -35]
        elif bg == 0.2:
            ypos = [0.25, 0.25, 0.32]
            angle = [50, 30, -35]
        elif bg == 0.4:
            ypos = [0.30, 0.30, 0.32]
            angle = [50, 30, -35]
        elif bg == 0.8:
            ypos = [0.33, 0.33, 0.32]
            angle = [45, 45, -35]
    else:
        if bg == 0.0:
            ypos = [0.4, 0.25, 0.32]
            angle = [55, 30, -35]
        elif bg == 0.2:
            ypos = [0.32, 0.25, 0.32]
            angle = [55, 30, -35]
        elif bg == 0.4:
            ypos = [0.35, 0.33, 0.32]
            angle = [55, 30, -35]
        elif bg == 0.8:
            ypos = [0.35, 0.40, 0.32]
            angle = [50, 50, -35]
    ax.text(0.7, ypos[0], r'$\Delta U_i/\varepsilon_{Bx0}$', color='k', fontsize=8,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            rotation=angle[0], horizontalalignment='center', verticalalignment='bottom',
            transform=ax.transAxes)
    ax.text(0.7, ypos[1], r'$\Delta U_e/\varepsilon_{Bx0}$', color='k', fontsize=8,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            rotation=angle[1], horizontalalignment='center', verticalalignment='top',
            transform=ax.transAxes)
    # ax.text(0.7, ypos[2], r'$\Delta \varepsilon_{B}/\varepsilon_{Bx0}$',
    #         color='k', fontsize=8,
    #         bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
    #         rotation=angle[2], horizontalalignment='center', verticalalignment='top',
    #         transform=ax.transAxes)
    ax.text(0.03, 0.41, r'$m_i/m_e=25$', color=COLORS[0], fontsize=8,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes)
    ax.text(0.03, 0.33, r'$m_i/m_e=100$', color=COLORS[1], fontsize=8,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes)
    ax.text(0.03, 0.25, r'$m_i/m_e=400$', color=COLORS[2], fontsize=8,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes)
    ax.text(0.03, 0.85, r'$B_g=' + str(bg) + '$', color='k', fontsize=10,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes)
    ax.tick_params(labelsize=8)
    ax.set_xlabel(r'$t\Omega_{ci}$', fontdict=FONT, fontsize=10)
    # ax.legend(loc=3, prop={'size': 16}, ncol=1,
    #           shadow=False, fancybox=False, frameon=False)
    fdir = '../img/img_high_mime/'
    mkdir_p(fdir)
    if const_va:
        fname = fdir + 'internal_econv_' + bg_str + '_high.pdf'
    else:
        fname = fdir + 'internal_econv_' + bg_str + '.pdf'
    fig.savefig(fname)
    plt.show()


def energy_partition(bg, const_va):
    """Plot energy energy partition between ion and electrons

    Args:
        bg: guide field strength
        const_va: whether the Alfven speed is constant for different mass ratio
    """
    bg_str = str(int(bg * 10)).zfill(2)
    mimes = np.asarray([25, 100, 400])
    fig = plt.figure(figsize=[7, 5])
    ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    ax.set_prop_cycle('color', COLORS)
    tshifts = shift_tframes(const_va)
    for mime in mimes:
        run_name = 'mime' + str(mime) + '_beta002_' + 'bg' + bg_str
        if const_va and mime != 400:
            run_name += '_high'
        picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        tenergy = pic_info.tenergy
        tenergy -= tshifts[str(mime)]
        kene_e = pic_info.kene_e
        kene_i = pic_info.kene_i

        ene_ratio = div0((kene_i - kene_i[0]), (kene_e - kene_e[0]))

        ltext = r"$m_i/m_e=" + str(mime) + "$"
        p1, = ax.plot(tenergy, ene_ratio, linewidth=3, label=ltext)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlim([0, 120.0])
    ax.set_ylim([1.0, 4.0])
    ax.tick_params(labelsize=16)
    ax.set_xlabel(r'$t\Omega_{ci}$', fontdict=FONT, fontsize=20)
    ax.set_ylabel(r'$\Delta K_i/\Delta K_e$', fontdict=FONT, fontsize=20)
    ax.legend(loc=4, prop={'size': 16}, ncol=1,
              shadow=False, fancybox=False, frameon=False)
    fdir = '../img/img_high_mime/'
    mkdir_p(fdir)
    if const_va:
        fname = fdir + 'epartition_' + bg_str + '_high.pdf'
    else:
        fname = fdir + 'epartition_' + bg_str + '.pdf'
    fig.savefig(fname)
    plt.show()


def energy_partition_mime(const_va=False, high_bg=False):
    """Plot energy partition for all runs
    Args:
        const_va: whether the Alfven speed is constant for different mass ratio
        high_bg: whether to include runs with higher guide field
    """
    mimes = np.asarray([25, 100, 400])
    if high_bg:
        bgs = np.asarray([0.0, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4])
    else:
        bgs = np.asarray([0.0, 0.2, 0.4, 0.8])
    nmime, = mimes.shape
    nbg, = bgs.shape

    tshifts = shift_tframes(const_va)
    tmin = get_tmin(const_va)

    # Second pass to get the energy conversion at tmin
    econv_rates = np.zeros((nmime, nbg, 7))
    for imime, mime in enumerate(mimes):
        for ibg, bg in enumerate(bgs):
            print("Guide field: %0.1f" % bg)
            bg_str = str(int(bg * 10)).zfill(2)
            run_name = 'mime' + str(mime) + '_beta002_' + 'bg' + bg_str
            if const_va and mime != 400:
                run_name += '_high'
            picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
            pic_info = read_data_from_json(picinfo_fname)
            tenergy = pic_info.tenergy
            tfields = pic_info.tfields
            tenergy -= tshifts[str(mime)]
            tfields -= tshifts[str(mime)]
            tindex_e, t0 = find_nearest(tenergy, tmin)
            tindex_f, t0 = find_nearest(tfields, tmin)
            fname = "../data/bulk_internal_energy/" + run_name + "/"
            fname += "bulk_internal_energy_e.dat"
            fdata = np.fromfile(fname, dtype=np.float32)
            sz, = fdata.shape
            nframes = (sz//2)//4
            bene_e4 = fdata[:sz//2].reshape(nframes, -1)
            iene_e4 = fdata[sz//2:].reshape(nframes, -1)
            fname = "../data/bulk_internal_energy/" + run_name + "/"
            fname += "bulk_internal_energy_i.dat"
            fdata = np.fromfile(fname, dtype=np.float32)
            bene_i4 = fdata[:sz//2].reshape(nframes, -1)
            iene_i4 = fdata[sz//2:].reshape(nframes, -1)
            bene_e = bene_e4[:, -1]
            iene_e = iene_e4[:, -1]
            bene_i = bene_i4[:, -1]
            iene_i = iene_i4[:, -1]
            ene_magnetic = pic_info.ene_magnetic
            kene_e = pic_info.kene_e
            kene_i = pic_info.kene_i
            ene_magnetic = pic_info.ene_magnetic
            econv_rates[imime, ibg, 0] = (ene_magnetic[tindex_e] -
                                          ene_magnetic[0]) / ene_magnetic[0]
            econv_rates[imime, ibg, 1] = (kene_e[tindex_e] - kene_e[0]) / ene_magnetic[0]
            econv_rates[imime, ibg, 2] = (kene_i[tindex_e] - kene_i[0]) / ene_magnetic[0]
            econv_rates[imime, ibg, 3] = (iene_e[tindex_f] - iene_e[0]) / ene_magnetic[0]
            econv_rates[imime, ibg, 4] = (iene_i[tindex_f] - iene_i[0]) / ene_magnetic[0]
            econv_rates[imime, ibg, 5] = (bene_e[tindex_f] - bene_e[0]) / ene_magnetic[0]
            econv_rates[imime, ibg, 6] = (bene_i[tindex_f] - bene_i[0]) / ene_magnetic[0]
            print("Energy conversion:", econv_rates[imime, ibg, :])

    if high_bg:
        fig = plt.figure(figsize=[7.0, 2.5])
        rect = [0.06, 0.15, 0.4, 0.8]
        ax = fig.add_axes(rect)
    else:
        fig = plt.figure(figsize=[3.5, 2.5])
        ax = fig.add_axes([0.13, 0.15, 0.8, 0.8])
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    for imime, mime in enumerate(mimes):
        ax.plot(bgs[:4], econv_rates[imime, :4, 2]/econv_rates[imime, :4, 1],
                marker='o', markersize=4, linestyle='--', linewidth=1,
                color=COLORS[imime])
        ax.plot(bgs[:4], econv_rates[imime, :4, 4]/econv_rates[imime, :4, 3],
                marker='o', markersize=4, linestyle='-', linewidth=1,
                color=COLORS[imime])
    print("Ion energy gain / Electron energy gain:")
    print(econv_rates[:, :, 2]/econv_rates[:, :, 1])
    print("Ion internal energy gain / Electron internal energy gain:")
    print(econv_rates[:, :, 4]/econv_rates[:, :, 3])
    print("Ion bulk energy gain:")
    print(econv_rates[:, :, 6])

    ax.text(0.6, 0.71, r'$\Delta K_i/\Delta K_e$', color='k', fontsize=8,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)
    ax.text(0.6, 0.30, r'$\Delta U_i/\Delta U_e$', color='k',
            rotation=-10, fontsize=8,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)
    ax.set_xlim([-0.05, 0.85])
    ax.set_ylim([0.5, 4.0])
    ax.plot(ax.get_xlim(), [0, 0], color='k', linewidth=0.5, linestyle='--')
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top=False)
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlabel(r'$B_g$', fontsize=10)
    ax.tick_params(labelsize=8)

    # higher guide field
    if high_bg:
        rect[0] += rect[2] + 0.1
        ax1 = fig.add_axes(rect)
        for imime, mime in enumerate(mimes):
            ax1.plot(bgs[4:], econv_rates[imime, 4:, 2]/econv_rates[imime, 4:, 1],
                     marker='o', markersize=4, linestyle='--', linewidth=1,
                     color=COLORS[imime])
            ax1.plot(bgs[4:], econv_rates[imime, 4:, 4]/econv_rates[imime, 4:, 3],
                     marker='o', markersize=4, linestyle='-', linewidth=1,
                     color=COLORS[imime])

        ax1.set_xlim([1.5, 6.5])
        ax1.tick_params(bottom=True, top=True, left=True, right=True)
        ax1.tick_params(axis='x', which='minor', direction='in', top=False)
        ax1.tick_params(axis='x', which='major', direction='in')
        ax1.tick_params(axis='y', which='minor', direction='in')
        ax1.tick_params(axis='y', which='major', direction='in')
        ax1.set_xlabel(r'$B_g$', fontsize=10)
        ax1.tick_params(labelsize=8)

    fdir = '../img/img_high_mime/'
    mkdir_p(fdir)
    if const_va:
        if high_bg:
            fname = fdir + 'ene_part_high_bg_high.pdf'
        else:
            fname = fdir + 'ene_part_high.pdf'
    else:
        if high_bg:
            fname = fdir + 'ene_part_high_bg.pdf'
        else:
            fname = fdir + 'ene_part.pdf'
    fig.savefig(fname)
    plt.show()


def fit_thermal_core(ene, f):
    """Fit to get the thermal core of the particle distribution.

    Fit the thermal core of the particle distribution.
    The thermal core is fitted as a Maxwellian distribution.

    Args:
        ene: the energy bins array.
        f: the particle flux distribution.

    Returns:
        fthermal: thermal part of the particle distribution.
    """
    print('Fitting to get the thermal core of the particle distribution')
    estart = 0
    ng = 3
    kernel = np.ones(ng) / float(ng)
    fnew = np.convolve(f, kernel, 'same')
    nshift = 10  # grids shift for fitting thermal core.
    eend = np.argmax(f) + nshift
    emax = ene[np.argmax(f)]
    bguess = 1.0 / (3 * emax)
    aguess = f.max() / (np.sqrt(emax) * np.exp(-bguess * emax))
    popt, pcov = curve_fit(fitting_funcs.func_maxwellian,
                           ene[estart:eend], f[estart:eend],
                           p0=[aguess, bguess])
    fthermal = fitting_funcs.func_maxwellian(ene, popt[0], popt[1])
    print('Energy with maximum flux: %f' % ene[eend - 10])
    print('Energy with maximum flux in fitted thermal core: %f' % (0.5 / popt[1]))
    return fthermal


def fit_thermal_tail(ene, f):
    """Fit to get the tail of the particle distribution.

    The tail is fitted as a Maxwellian distribution.

    Args:
        ene: the energy bins array.
        f: the particle flux distribution.
    """
    print('Fitting to get the thermal core of the particle distribution')
    estart = 0
    nshift = 10  # grids shift for fitting thermal core.
    eend = np.argmax(f) + nshift
    emax = ene[np.argmax(f)]
    bguess = 1.0 / (3 * emax)
    aguess = f.max() / (np.sqrt(emax) * np.exp(-bguess * emax))
    popt, pcov = curve_fit(fitting_funcs.func_maxwellian,
                           ene[estart:eend], f[estart:eend],
                           p0=[aguess, bguess])
    print('Energy with maximum flux in fitted thermal tail: %f' % (0.5 / popt[1]))
    return popt


def accumulated_particle_info(ene, f):
    """
    Get the accumulated particle number and total energy from
    the distribution function.

    Args:
        ene: the energy bins array.
        f: the energy distribution array.
    Returns:
        nacc_ene: the accumulated particle number with energy.
        eacc_ene: the accumulated particle total energy with energy.
    """
    nbins, = f.shape
    dlogE = (math.log10(max(ene)) - math.log10(min(ene))) / nbins
    nacc_ene = np.zeros(nbins)
    eacc_ene = np.zeros(nbins)
    nacc_ene[0] = f[0] * ene[0]
    eacc_ene[0] = 0.5 * f[0] * ene[0]**2
    for i in range(1, nbins):
        nacc_ene[i] = f[i] * (ene[i] + ene[i - 1]) * 0.5 + nacc_ene[i - 1]
        eacc_ene[i] = 0.5 * f[i] * (ene[i] - ene[i - 1]) * (
            ene[i] + ene[i - 1])
        eacc_ene[i] += eacc_ene[i - 1]
    nacc_ene *= dlogE
    eacc_ene *= dlogE
    return (nacc_ene, eacc_ene)


def energy_spectrum_early(bg, species, tframe, show_plot=True):
    """Plot energy spectrum early in the simulations

    Args:
        bg: guide field strength
        species: particle species
        tframe: time frame
    """
    if species == 'h':
        species = 'H'
    spect_info = {"nbins": 800,
                  "emin": 1E-5,
                  "emax": 1E3}
    bg_str = str(int(bg * 10)).zfill(2)
    mimes = np.asarray([25, 100, 400])
    fig = plt.figure(figsize=[7, 5])
    ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    ax.set_prop_cycle('color', COLORS)
    for mime in mimes:
        run_name = "mime" + str(mime) + "_beta002_bg" + bg_str
        picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        tenergy = pic_info.tenergy
        if mime == 400:
            tframe1 = tframe
        else:
            tframe1 = tframe + 12
        if species == 'e':
            vth = pic_info.vthe
        else:
            vth = pic_info.vthi
        gama = 1.0 / math.sqrt(1.0 - 3 * vth**2)
        eth = gama - 1.0
        emin_log = math.log10(spect_info["emin"])
        emax_log = math.log10(spect_info["emax"])
        nbins = spect_info["nbins"]
        delog = (emax_log - emin_log) / nbins
        emin_log = emin_log - delog
        emax_log = emax_log - delog
        elog = np.logspace(emin_log, emax_log, nbins + 1)
        elog_mid = 0.5 * (elog[:-1] + elog[1:])
        elog /= eth
        elog_mid /= eth
        nptot = pic_info.nx * pic_info.ny * pic_info.nz * pic_info.nppc
        fdir = '../data/spectra/' + run_name + '/'
        fname = fdir + 'spectrum-' + species.lower() + '.' + str(tframe1)
        flog = np.fromfile(fname)
        flog /= nptot
        if species != 'e':
            flog /= pic_info.mime
        flog[0] = flog[1]  # remove spike at 0
        fth = fit_thermal_core(elog_mid, flog)
        fnth = flog - fth
        eindex, ene = find_nearest(elog_mid, 1E-1)
        fth[:eindex] += fnth[:eindex]
        fnth[:eindex] = 0
        eindex, ene = find_nearest(elog_mid, 1E2)
        popt = fit_thermal_tail(elog_mid[eindex:], flog[eindex:])
        fth2 = fitting_funcs.func_maxwellian(elog_mid, popt[0], popt[1])

        ltext = r"$m_i/m_e=" + str(mime) + "$"
        p1, = ax.loglog(elog_mid, flog, linewidth=2, label=ltext)
        # p1, = ax.loglog(elog_mid, fth, linewidth=1)
        # p2, = ax.loglog(elog_mid, fnth, linewidth=2)
        # p3, = ax.loglog(elog_mid, fth2, linewidth=1)
    fnorm = 1E3 if species == 'e' else 1E5
    pindex = -3.0
    fpower = fnorm * elog_mid**pindex
    power_index = "{%0.1f}" % pindex
    pname = r'$\sim \varepsilon^{' + power_index + '}$'
    es_index, es = find_nearest(elog_mid, 10)
    ee_index, ee = find_nearest(elog_mid, 100)
    ax.loglog(elog_mid[es_index:ee_index], fpower[es_index:ee_index],
              linewidth=2, color='k', label=pname)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top=True)
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    if species == 'e':
        ax.set_xlim([1E-1, 1E3])
    else:
        ax.set_xlim([1E-1, 1E3])
    ax.set_ylim([1E-9, 1E2])
    ax.tick_params(labelsize=16)
    ax.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$',
                  fontdict=FONT, fontsize=20)
    ax.set_ylabel(r'$f(\varepsilon)$', fontdict=FONT, fontsize=20)
    ax.legend(loc=1, prop={'size': 16}, ncol=1,
              shadow=False, fancybox=False, frameon=False)
    sname = 'electron' if species == 'e' else 'ion'
    fdir = '../img/img_high_mime/spect_compare/' + sname + '/'
    mkdir_p(fdir)
    fname = fdir + 'spect_' + bg_str + '_' + str(tframe) + '.pdf'
    fig.savefig(fname)
    if show_plot:
        plt.show()
    else:
        plt.close()


def stacked_spectrum(species='e', const_va=False, show_plot=True):
    """Plot stacked particle energy spectrum

    Args:
        species: 'e' for electrons, 'H' for ions
        const_va: whether the Alfven speed is the same for different mass ratio
    """
    mimes = [25, 100, 400]
    # mimes = [25]
    bgs = [0.0, 0.2, 0.4, 0.8]
    # bgs = [0.0]
    # ntf = 94 if const_va else 101
    ntf = 101
    fmin = 1E-9

    tshifts = shift_tframes(const_va)
    # base thermal energy
    run_name = 'mime400_beta002_bg00'
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    if species == 'e':
        vth = pic_info.vthe
    else:
        vth = pic_info.vthi
    gama = 1.0 / math.sqrt(1.0 - 3 * vth**2)
    eth0 = gama - 1.0

    fig = plt.figure(figsize=[7, 4])
    rect0 = [0.11, 0.68, 0.18, 0.26]
    hgap, vgap = 0.015, 0.02

    for imime, mime in enumerate(mimes):
        if const_va or mime == 400:
            spect_info = {"nbins": 1000,
                          "emin": 1E-6,
                          "emax": 1E4}
        else:
            spect_info = {"nbins": 800,
                          "emin": 1E-5,
                          "emax": 1E3}
        emin_log = math.log10(spect_info["emin"])
        emax_log = math.log10(spect_info["emax"])
        nbins = spect_info["nbins"]
        delog = (emax_log - emin_log) / nbins
        emin_log -= delog
        elog = np.logspace(emin_log, emax_log, nbins + 1)
        elog_mid = 0.5 * (elog[:-1] + elog[1:])
        delog = np.diff(elog)

        rect = np.copy(rect0)
        for ibg, bg in enumerate(bgs):
            bg_str = str(int(bg * 10)).zfill(2)
            run_name = 'mime' + str(mime) + '_beta002_' + 'bg' + bg_str
            if const_va and mime != 400:
                run_name += '_high'
            picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
            pic_info = read_data_from_json(picinfo_fname)
            pic_run_dir = pic_info.run_dir
            if species == 'e':
                vth = pic_info.vthe
                species_name = 'e'
            else:
                if const_va or mime == 400:
                    species_name = 'i'
                else:
                    species_name = 'H'
                vth = pic_info.vthi
            gama = 1.0 / math.sqrt(1.0 - 3 * vth**2)
            eth = gama - 1.0
            elog = elog_mid / eth
            eindex1, ene = find_nearest(elog, 2)
            eindex2, ene = find_nearest(elog, 1000)
            eindex10, ene = find_nearest(elog, 10)
            nptot = pic_info.nx * pic_info.ny * pic_info.nz * pic_info.nppc

            tframes = range(ntf)
            nframes = len(tframes)
            flogs = np.zeros((nframes, nbins))
            for iframe, tframe in enumerate(tframes):
                # tframe1 = tframe + tshifts[str(mime)]
                tframe1 = tframe
                print("Time frame: %d" % tframe1)
                if const_va or mime == 400:
                    tindex1 = tframe1 * pic_info.ehydro_interval
                    fdir = pic_run_dir + 'spectrum_combined/'
                    fname = fdir + 'spectrum_' + species_name + '_' + str(tindex1) + '.dat'
                    fdata = np.fromfile(fname, dtype=np.float32)
                    flog = fdata[3:] # the first 3 are magnetic field components
                    flog /= delog
                    # re-normalize using the thermal of energy of mime=400
                    flog /= eth0 * 400 / (eth * pic_info.mime)
                else:
                    fdir = '../data/spectra/' + run_name + '/'
                    fname = fdir + 'spectrum-' + species_name.lower() + '.' + str(tframe1)
                    flog = np.fromfile(fname)
                flog /= nptot
                if species != 'e':
                    flog /= pic_info.mime

                color = plt.cm.jet(tframe/float(ntf), 1)
                flogs[iframe, :] = flog
                if iframe == 25 or iframe == nframes - 1:
                    ntot, etot = accumulated_particle_info(elog, flog)
                    print(">10 thermal energy (number fraction): %f" %
                          ((ntot[-1] - ntot[eindex10])/ntot[-1]))
                    print(">10 thermal energy (energy fraction): %f" %
                          ((etot[-1] - etot[eindex10])/etot[-1]))
            flogs += fmin
            # fdata = np.diff(np.log10(flogs[:, eindex1:eindex2]), axis=0).T
            fdata = np.log10(flogs[1:, eindex1:eindex2] / flogs[:-1, eindex1:eindex2]).T
            ng = 3
            kernel = np.ones((ng,ng)) / float(ng*ng)
            fdata = signal.convolve2d(fdata, kernel, mode='same')
            vmin, vmax = -0.1, 0.1

            ax = fig.add_axes(rect)

            cmap = palettable.colorbrewer.diverging.RdGy_11_r.mpl_colormap
            img = ax.imshow(fdata, extent=[1, ntf-1, math.log10(elog[eindex1]),
                                           math.log10(elog[eindex2])],
                            vmin=vmin, vmax=vmax, cmap=plt.cm.seismic,
                            aspect='auto', origin='lower', interpolation='bicubic')
            ax.tick_params(bottom=True, top=True, left=True, right=True)
            ax.tick_params(axis='x', which='minor', direction='in')
            ax.tick_params(axis='x', which='major', direction='in')
            ax.tick_params(axis='y', which='minor', direction='in')
            ax.tick_params(axis='y', which='major', direction='in')
            ax.set_xticks([25, 50, 75, 100])
            if imime == len(mimes) - 1:
                ax.set_xlabel(r'$t\Omega_{ci}$', fontsize=10)
            else:
                ax.tick_params(axis='x', labelbottom=False)
            if ibg == 0:
                ax.set_yticks([1, 2, 3])
                ax.set_yticklabels([r'$10$', r'$10^2$', r'$10^3$'])
                ax.set_ylabel(r'$\varepsilon/\varepsilon_\text{th}$', fontsize=10)
            else:
                ax.tick_params(axis='y', labelleft=False)
            ax.tick_params(labelsize=8)
            rect[0] += rect[2] + hgap
            if imime == 0:
                title = r"$B_g=" + str(bg) + "$"
                ax.set_title(title, fontsize=10)
            if ibg == 0:
                text1 = r'$m_i/m_e=' + str(mime) + '$'
                ax.text(-0.48, 0.5, text1, color='k', fontsize=10,
                        rotation='vertical',
                        bbox=dict(facecolor='none', alpha=1.0,
                                  edgecolor='none', pad=10.0),
                        horizontalalignment='left', verticalalignment='center',
                        transform=ax.transAxes)
            # if imime == 2:
            #     ax.plot([1, ntf-1], [math.log10(50), math.log10(50)],
            #             linewidth=0.5, color='k')
        rect0[1] -= rect0[3] + vgap
    rect0[1] += rect0[3] + vgap
    rect_cbar = np.copy(rect0)
    rect_cbar[0] += rect[2]*4 + hgap*4
    rect_cbar[1] = rect0[1] + 0.5*rect0[3]
    rect_cbar[2] = 0.01
    rect_cbar[3] = rect0[3] * 2 + vgap * 2
    cbar_ax = fig.add_axes(rect_cbar)
    cbar_ax.tick_params(right=True)
    cbar_ax.tick_params(axis='y', which='minor', direction='in')
    cbar_ax.tick_params(axis='y', which='major', direction='in')
    cbar = fig.colorbar(img, cax=cbar_ax, orientation='vertical',
                        extend='both')
    cbar.set_ticks(np.linspace(vmin, vmax, 5))
    label1 = r'$\log_{10}((f(\varepsilon, t) + \epsilon)/(f(\varepsilon, t-\Delta t) + \epsilon))$'
    cbar_ax.set_ylabel(label1, fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    fdir = '../img/img_high_mime/'
    mkdir_p(fdir)
    if const_va:
        fname = fdir + 'stacked_espect_' + species + '_high.pdf'
    else:
        fname = fdir + 'stacked_espect_' + species + '.pdf'
    fig.savefig(fname, dpi=200)
    if show_plot:
        plt.show()
    else:
        plt.close()


def evolving_spectrum(species='e', const_va=False, high_bg=False, show_plot=True):
    """Plot evolving energy spectrum

    Args:
        species: 'e' for electrons, 'H' for ions
        const_va: whether the Alfven speed is the same for different mass ratio
        high_bg: whether to include runs with higher guide field
    """
    mimes = [25, 100, 400]
    # mimes = [25]
    if high_bg:
        bgs = [0.0, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
    else:
        bgs = [0.0, 0.2, 0.4, 0.8]
    # bgs = [0.0]
    ntf = 101
    fmin = 1E-9

    tshifts = shift_tframes(const_va)
    # base thermal energy
    run_name = 'mime400_beta002_bg00'
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    if species == 'e':
        vth = pic_info.vthe
    else:
        vth = pic_info.vthi
    gama = 1.0 / math.sqrt(1.0 - 3 * pic_info.vthe**2)
    eth0 = {}
    eth0["e"] = gama - 1.0
    gama = 1.0 / math.sqrt(1.0 - 3 * pic_info.vthi**2)
    eth0["i"] = (gama - 1.0) * pic_info.mime

    if high_bg:
        fig = plt.figure(figsize=[11, 4])
        rect0 = [0.07, 0.68, 0.12, 0.26]
        hgap, vgap = 0.01, 0.02
    else:
        fig = plt.figure(figsize=[7, 4])
        rect0 = [0.11, 0.68, 0.2, 0.26]
        hgap, vgap = 0.015, 0.02

    if const_va:
        tframes = [40, 60, 94]
    else:
        tframes = [40, 60, 90]
    nframes = len(tframes)
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    for iframe, tframe in enumerate(tframes):
        rect = np.copy(rect0)
        for ibg, bg in enumerate(bgs):
            bg_str = str(int(bg * 10)).zfill(2)
            ax = fig.add_axes(rect)
            ax.set_prop_cycle('color', COLORS)
            for imime, mime in enumerate(mimes):
                if const_va or mime == 400:
                    spect_info = {"nbins": 1000,
                                  "emin": 1E-6,
                                  "emax": 1E4}
                else:
                    spect_info = {"nbins": 800,
                                  "emin": 1E-5,
                                  "emax": 1E3}
                emin_log = math.log10(spect_info["emin"])
                emax_log = math.log10(spect_info["emax"])
                nbins = spect_info["nbins"]
                delog = (emax_log - emin_log) / nbins
                emin_log -= delog
                elog = np.logspace(emin_log, emax_log, nbins + 1)
                elog_mid = 0.5 * (elog[:-1] + elog[1:])
                delog = np.diff(elog)

                run_name = 'mime' + str(mime) + '_beta002_' + 'bg' + bg_str
                if const_va and mime != 400:
                    run_name += '_high'
                picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
                pic_info = read_data_from_json(picinfo_fname)
                pic_run_dir = pic_info.run_dir
                if species == 'e':
                    vth = pic_info.vthe
                    species_name = 'e'
                    pmass = 1.0
                else:
                    if const_va or mime == 400:
                        species_name = 'i'
                    else:
                        species_name = 'H'
                    vth = pic_info.vthi
                    pmass = pic_info.mime
                gama = 1.0 / math.sqrt(1.0 - 3 * vth**2)
                eth = gama - 1.0
                elog_mid /= eth
                eindex1, ene = find_nearest(elog_mid, 2)
                eindex2, ene = find_nearest(elog_mid, 1000)
                nptot = pic_info.nx * pic_info.ny * pic_info.nz * pic_info.nppc

                if bg < 1.0:
                    tframe1 = tframe + tshifts[str(mime)]
                else:
                    tframe1 = tframe + 3 if mime > 25 else tframe
                print("Time frame: %d" % tframe1)
                if const_va or mime == 400:
                    tindex1 = tframe1 * pic_info.ehydro_interval
                    fdir = pic_run_dir + 'spectrum_combined/'
                    fname = fdir + 'spectrum_' + species_name + '_' + str(tindex1) + '.dat'
                    fdata = np.fromfile(fname, dtype=np.float32)
                    flog = fdata[3:] # the first 3 are magnetic field components
                    flog /= delog
                else:
                    fdir = '../data/spectra/' + run_name + '/'
                    fname = fdir + 'spectrum-' + species_name.lower() + '.' + str(tframe1)
                    flog = np.fromfile(fname)
                flog /= nptot
                # re-normalize using the thermal of energy of mime=400
                flog /= eth0[species] / eth

                ax.loglog(elog_mid, flog, linewidth=1)
                ax.tick_params(bottom=True, top=True, left=True, right=True)
                ax.tick_params(axis='x', which='minor', direction='in', top=True)
                ax.tick_params(axis='x', which='major', direction='in')
                ax.tick_params(axis='y', which='minor', direction='in')
                ax.tick_params(axis='y', which='major', direction='in')
                if imime == 0 and ibg == 1 and (species in ['H', 'i']):
                    wpe_wce = pic_info.dtwce / pic_info.dtwpe
                    va = wpe_wce / math.sqrt(pic_info.mime)
                    bene = 0.5 * va**2 / eth
                    ax.plot([bene, bene], [1E-8, 1E2], color='k', linewidth=0.5)
                    if iframe == 1:
                        text1 = r'$m_iv_A^2/2$'
                        ax.text(0.55, 0.1, text1, color='k',
                                fontsize=10, rotation=90,
                                bbox=dict(facecolor='none', alpha=1.0,
                                          edgecolor='none', pad=10.0),
                                horizontalalignment='right',
                                verticalalignment='bottom',
                                transform=ax.transAxes)
            if iframe > 0:
                if species == 'e':
                    pindex = -3.5 if const_va else -3.0
                    fnorm = 10 if const_va else 2
                    fpower = elog**pindex * 500
                    power_index = "{%0.1f}" % pindex
                    pname = r'$\propto \varepsilon^{' + power_index + '}$'
                    es_index, es = find_nearest(elog, 1)
                    ee_index, ee = find_nearest(elog, 500)
                    ax.loglog(elog[es_index:ee_index], fpower[es_index:ee_index]*fnorm,
                              linewidth=0.5, color='k', linestyle='--', label=pname)
                    # ax.loglog(elog[es_index:ee_index], fpower[es_index:ee_index]/10,
                    #           linewidth=0.5, color='k', linestyle='--')
                else:
                    pindex = -1.0
                    fpower = elog**pindex * 5
                    power_index = "{%0.1f}" % pindex
                    pname = r'$\propto \varepsilon^{' + power_index + '}$'
                    es_index, es = find_nearest(elog, 1)
                    ee_index, ee = find_nearest(elog, 500)
                    ax.loglog(elog[es_index:ee_index], fpower[es_index:ee_index],
                              linewidth=0.5, color='k', linestyle='--', label=pname)
                    if iframe == 1:
                        pindex = -7.5
                        fpower = elog**pindex * 5E12
                        power_index = "{%0.1f}" % pindex
                        pname = r'$\propto \varepsilon^{' + power_index + '}$'
                        ax.loglog(elog[es_index:ee_index], fpower[es_index:ee_index],
                                  linewidth=0.5, color='k', linestyle='-.', label=pname)
                    else:
                        pindex = -6.0
                        fpower = elog**pindex * 2E10
                        power_index = "{%0.1f}" % pindex
                        pname = r'$\propto \varepsilon^{' + power_index + '}$'
                        p1, = ax.loglog(elog[es_index:ee_index], fpower[es_index:ee_index],
                                        linewidth=0.5, color='k', linestyle=':', label=pname)
            if iframe == 1 and ibg == 0:
                ax.legend(loc=3, prop={'size': 10}, ncol=1,
                          shadow=False, fancybox=False, frameon=False)
            elif iframe == 2 and ibg == 0 and species != 'e':
                ax.legend(handles=[p1], loc=3, prop={'size': 10}, ncol=1,
                          shadow=False, fancybox=False, frameon=False)
            ax.set_xlim([1, 500])
            ax.set_ylim([1E-8, 1E2])
            if iframe == 0 and ibg == 0:
                ax.text(0.05, 0.36, r'$m_i/m_e=25$', color=COLORS[0], fontsize=8,
                        bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                        horizontalalignment='left', verticalalignment='bottom',
                        transform=ax.transAxes)
                ax.text(0.05, 0.23, r'$m_i/m_e=100$', color=COLORS[1], fontsize=8,
                        bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                        horizontalalignment='left', verticalalignment='bottom',
                        transform=ax.transAxes)
                ax.text(0.05, 0.1, r'$m_i/m_e=400$', color=COLORS[2], fontsize=8,
                        bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                        horizontalalignment='left', verticalalignment='bottom',
                        transform=ax.transAxes)
            if iframe == nframes - 1:
                ax.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$', fontsize=10)
            else:
                ax.tick_params(axis='x', labelbottom=False)
            if ibg == 0:
                ax.set_ylabel(r'$f(\varepsilon)$', fontsize=10)
            else:
                ax.tick_params(axis='y', labelleft=False)
            ax.tick_params(labelsize=8)
            rect[0] += rect[2] + hgap
            if iframe == 0:
                title = r"$B_g=" + str(bg) + "$"
                ax.set_title(title, fontsize=10)
            if ibg == 0:
                text1 = r'$t\Omega_{ci}=' + str(tframe) + '$'
                ax.text(-0.48, 0.5, text1, color='k', fontsize=10,
                        rotation='vertical',
                        bbox=dict(facecolor='none', alpha=1.0,
                                  edgecolor='none', pad=10.0),
                        horizontalalignment='left', verticalalignment='center',
                        transform=ax.transAxes)
        rect0[1] -= rect0[3] + vgap
    rect0[1] += rect0[3] + vgap

    fdir = '../img/img_high_mime/'
    mkdir_p(fdir)
    if const_va:
        if high_bg:
            fname = fdir + 'evolve_espect_high_bg_' + species + '_high.pdf'
        else:
            fname = fdir + 'evolve_espect_' + species + '_high.pdf'
    else:
        if high_bg:
            fname = fdir + 'evolve_espect_high_bg_' + species + '.pdf'
        else:
            fname = fdir + 'evolve_espect_' + species + '.pdf'
    fig.savefig(fname)
    if show_plot:
        plt.show()
    else:
        plt.close()


def evolving_momentum_spectrum(species='e', const_va=False, high_bg=False, show_plot=True):
    """Plot evolving momentum spectrum

    Args:
        species: 'e' for electrons, 'H' for ions
        const_va: whether the Alfven speed is the same for different mass ratio
        high_bg: whether to include runs with higher guide field
    """
    mimes = [25, 100, 400]
    # mimes = [25]
    if high_bg:
        bgs = [0.0, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
    else:
        bgs = [0.0, 0.2, 0.4, 0.8]
    # bgs = [0.0]
    ntf = 101
    fmin = 1E-9

    tshifts = shift_tframes(const_va)
    # base thermal energy
    run_name = 'mime400_beta002_bg00'
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    if species == 'e':
        vth = pic_info.vthe
    else:
        vth = pic_info.vthi
    gama = 1.0 / math.sqrt(1.0 - 3 * pic_info.vthe**2)
    eth0 = {}
    pth0 = {}
    eth0["e"] = gama - 1.0
    pth0["e"] = math.sqrt(gama**2 - 1)
    gama = 1.0 / math.sqrt(1.0 - 3 * pic_info.vthi**2)
    eth0["i"] = (gama - 1.0) * pic_info.mime
    pth0["i"] = math.sqrt(gama**2 - 1.0)

    if high_bg:
        fig = plt.figure(figsize=[11, 4])
        rect0 = [0.07, 0.68, 0.12, 0.26]
        hgap, vgap = 0.01, 0.02
    else:
        fig = plt.figure(figsize=[7, 4])
        rect0 = [0.11, 0.68, 0.2, 0.26]
        hgap, vgap = 0.015, 0.02

    if const_va:
        tframes = [40, 60, 94]
    else:
        tframes = [40, 60, 90]
    nframes = len(tframes)
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    for iframe, tframe in enumerate(tframes):
        rect = np.copy(rect0)
        for ibg, bg in enumerate(bgs):
            bg_str = str(int(bg * 10)).zfill(2)
            ax = fig.add_axes(rect)
            ax.set_prop_cycle('color', COLORS)
            for imime, mime in enumerate(mimes):
                if const_va or mime == 400:
                    spect_info = {"nbins": 1000,
                                  "emin": 1E-6,
                                  "emax": 1E4}
                else:
                    spect_info = {"nbins": 800,
                                  "emin": 1E-5,
                                  "emax": 1E3}
                emin_log = math.log10(spect_info["emin"])
                emax_log = math.log10(spect_info["emax"])
                nbins = spect_info["nbins"]
                delog = (emax_log - emin_log) / nbins
                emin_log -= delog
                elog = np.logspace(emin_log, emax_log, nbins + 1)
                elog_mid = 0.5 * (elog[:-1] + elog[1:])
                delog = np.diff(elog)
                plog = np.sqrt((elog + 1)**2 - 1)
                plog_mid = (plog[:-1] + plog[1:]) * 0.5
                dplog = np.diff(plog)

                run_name = 'mime' + str(mime) + '_beta002_' + 'bg' + bg_str
                if const_va and mime != 400:
                    run_name += '_high'
                picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
                pic_info = read_data_from_json(picinfo_fname)
                pic_run_dir = pic_info.run_dir
                if species == 'e':
                    vth = pic_info.vthe
                    species_name = 'e'
                    pmass = 1.0
                else:
                    if const_va or mime == 400:
                        species_name = 'i'
                    else:
                        species_name = 'H'
                    vth = pic_info.vthi
                    pmass = pic_info.mime
                gama = 1.0 / math.sqrt(1.0 - 3 * vth**2)
                eth = gama - 1.0
                pth = math.sqrt(gama**2 - 1)
                elog_mid /= eth
                plog_mid /= pth
                eindex1, ene = find_nearest(elog_mid, 2)
                eindex2, ene = find_nearest(elog_mid, 1000)
                nptot = pic_info.nx * pic_info.ny * pic_info.nz * pic_info.nppc

                if bg < 1.0:
                    tframe1 = tframe + tshifts[str(mime)]
                else:
                    tframe1 = tframe + 3 if mime > 25 else tframe
                print("Time frame: %d" % tframe1)
                if const_va or mime == 400:
                    tindex1 = tframe1 * pic_info.ehydro_interval
                    fdir = pic_run_dir + 'spectrum_combined/'
                    fname = fdir + 'spectrum_' + species_name + '_' + str(tindex1) + '.dat'
                    fdata = np.fromfile(fname, dtype=np.float32)
                    flog = fdata[3:] # the first 3 are magnetic field components
                    pspect = flog / dplog
                    flog /= delog
                else:
                    fdir = '../data/spectra/' + run_name + '/'
                    fname = fdir + 'spectrum-' + species_name.lower() + '.' + str(tframe1)
                    flog = np.fromfile(fname)
                    pspect = flog * delog / dplog
                flog /= nptot
                pspect /= nptot
                # re-normalize using the thermal of energy of mime=400
                flog /= eth0[species] / eth
                pspect /= pth0[species] / pth

                ax.loglog(plog_mid, pspect, linewidth=1)
                ax.tick_params(bottom=True, top=True, left=True, right=True)
                ax.tick_params(axis='x', which='minor', direction='in', top=True)
                ax.tick_params(axis='x', which='major', direction='in')
                ax.tick_params(axis='y', which='minor', direction='in')
                ax.tick_params(axis='y', which='major', direction='in')
                if imime == 0 and ibg == 1 and (species in ['H', 'i']):
                    wpe_wce = pic_info.dtwce / pic_info.dtwpe
                    va = wpe_wce / math.sqrt(pic_info.mime)
                    bmom = va / pth
                    ax.plot([bmom, bmom], [1E-7, 1E3], color='k', linewidth=0.5)
                    if iframe == 1:
                        text1 = r'$m_iv_A$'
                        ax.text(0.50, 0.1, text1, color='k',
                                fontsize=10, rotation=90,
                                bbox=dict(facecolor='none', alpha=1.0,
                                          edgecolor='none', pad=10.0),
                                horizontalalignment='right',
                                verticalalignment='bottom',
                                transform=ax.transAxes)
            if iframe > 0:
                if species == 'e':
                    pindex = -6.0 if const_va else -5.0
                    fnorm = 10 if const_va else 2
                    fpower = elog**pindex * 200
                    power_index = "{%0.1f}" % pindex
                    pname = r'$\propto p^{' + power_index + '}$'
                    es_index, es = find_nearest(elog, 1)
                    ee_index, ee = find_nearest(elog, 500)
                    ax.loglog(elog[es_index:ee_index], fpower[es_index:ee_index]*fnorm,
                              linewidth=0.5, color='k', linestyle='--', label=pname)
                    # ax.loglog(elog[es_index:ee_index], fpower[es_index:ee_index]/10,
                    #           linewidth=0.5, color='k', linestyle='--')
                else:
                    pindex = -1.0
                    fpower = elog**pindex * 5
                    power_index = "{%0.1f}" % pindex
                    pname = r'$\propto p^{' + power_index + '}$'
                    es_index, es = find_nearest(elog, 1)
                    ee_index, ee = find_nearest(elog, 500)
                    ax.loglog(elog[es_index:ee_index], fpower[es_index:ee_index],
                              linewidth=0.5, color='k', linestyle='--', label=pname)
                    if iframe == 1:
                        pindex = -9.0
                        fpower = elog**pindex * 5E7
                        power_index = "{%0.1f}" % pindex
                        pname = r'$\propto p^{' + power_index + '}$'
                        ax.loglog(elog[es_index:ee_index], fpower[es_index:ee_index],
                                  linewidth=0.5, color='k', linestyle='-.', label=pname)
                    else:
                        pindex = -10.0
                        fpower = elog**pindex * 2E9
                        power_index = "{%0.1f}" % pindex
                        pname = r'$\propto p^{' + power_index + '}$'
                        p1, = ax.loglog(elog[es_index:ee_index], fpower[es_index:ee_index],
                                        linewidth=0.5, color='k', linestyle=':', label=pname)
            if iframe == 1 and ibg == 0:
                ax.legend(loc=3, prop={'size': 10}, ncol=1,
                          shadow=False, fancybox=False, frameon=False)
            elif iframe == 2 and ibg == 0 and species != 'e':
                ax.legend(handles=[p1], loc=3, prop={'size': 10}, ncol=1,
                          shadow=False, fancybox=False, frameon=False)
            ax.set_xlim([5E-1, 50])
            if species == 'e':
                ax.set_ylim([1E-8, 1E2])
            else:
                ax.set_ylim([1E-7, 1E3])
            if iframe == 0 and ibg == 0:
                ax.text(0.05, 0.36, r'$m_i/m_e=25$', color=COLORS[0], fontsize=8,
                        bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                        horizontalalignment='left', verticalalignment='bottom',
                        transform=ax.transAxes)
                ax.text(0.05, 0.23, r'$m_i/m_e=100$', color=COLORS[1], fontsize=8,
                        bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                        horizontalalignment='left', verticalalignment='bottom',
                        transform=ax.transAxes)
                ax.text(0.05, 0.1, r'$m_i/m_e=400$', color=COLORS[2], fontsize=8,
                        bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                        horizontalalignment='left', verticalalignment='bottom',
                        transform=ax.transAxes)
            if iframe == nframes - 1:
                ax.set_xlabel(r'$p/p_\text{th}$', fontsize=10)
            else:
                ax.tick_params(axis='x', labelbottom=False)
            if ibg == 0:
                ax.set_ylabel(r'$f(p)$', fontsize=10)
            else:
                ax.tick_params(axis='y', labelleft=False)
            ax.tick_params(labelsize=8)
            rect[0] += rect[2] + hgap
            if iframe == 0:
                title = r"$B_g=" + str(bg) + "$"
                ax.set_title(title, fontsize=10)
            if ibg == 0:
                text1 = r'$t\Omega_{ci}=' + str(tframe) + '$'
                ax.text(-0.48, 0.5, text1, color='k', fontsize=10,
                        rotation='vertical',
                        bbox=dict(facecolor='none', alpha=1.0,
                                  edgecolor='none', pad=10.0),
                        horizontalalignment='left', verticalalignment='center',
                        transform=ax.transAxes)
        rect0[1] -= rect0[3] + vgap
    rect0[1] += rect0[3] + vgap

    fdir = '../img/img_high_mime/'
    mkdir_p(fdir)
    if const_va:
        if high_bg:
            fname = fdir + 'evolve_pspect_high_bg_' + species + '_high.pdf'
        else:
            fname = fdir + 'evolve_pspect_' + species + '_high.pdf'
    else:
        if high_bg:
            fname = fdir + 'evolve_pspect_high_bg_' + species + '.pdf'
        else:
            fname = fdir + 'evolve_pspect_' + species + '.pdf'
    fig.savefig(fname)
    if show_plot:
        plt.show()
    else:
        plt.close()


def internal_energy_partition(bg, const_va):
    """Plot internal energy energy partition between ion and electrons

    Args:
        bg: guide field strength
        const_va: whether the Alfven speed is constant for different mass ratio
    """
    bg_str = str(int(bg * 10)).zfill(2)
    mimes = np.asarray([25, 100, 400])
    tshifts = shift_tframes(const_va)
    fig = plt.figure(figsize=[7, 5])
    ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    ax.set_prop_cycle('color', COLORS)
    for mime in mimes:
        run_name = 'mime' + str(mime) + '_beta002_' + 'bg' + bg_str
        if const_va and mime != 400:
            run_name += '_high'
        picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        tenergy = pic_info.tenergy
        tfields = pic_info.tfields
        tenergy -= tshifts[str(mime)]
        tfields -= tshifts[str(mime)]
        fname = "../data/bulk_internal_energy/" + run_name + "/"
        fname += "bulk_internal_energy_e.dat"
        fdata = np.fromfile(fname, dtype=np.float32)
        sz, = fdata.shape
        nframes = (sz//2)//4
        bene_e = fdata[:sz//2].reshape(nframes, -1)
        iene_e = fdata[sz//2:].reshape(nframes, -1)
        fname = "../data/bulk_internal_energy/" + run_name + "/"
        fname += "bulk_internal_energy_i.dat"
        fdata = np.fromfile(fname, dtype=np.float32)
        bene_i = fdata[:sz//2].reshape(nframes, -1)
        iene_i = fdata[sz//2:].reshape(nframes, -1)
        kene_e = pic_info.kene_e
        kene_i = pic_info.kene_i

        ene_ratio = div0((iene_i[:, -1] - iene_i[0, -1]),
                         (iene_e[:, -1] - iene_e[0, -1]))

        ltext = r"$m_i/m_e=" + str(mime) + "$"
        p1, = ax.plot(tfields, ene_ratio, linewidth=3, label=ltext)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_xlim([0.0, 120.0])
    ax.set_ylim([0.0, 3.0])
    ax.tick_params(labelsize=16)
    ax.set_xlabel(r'$t\Omega_{ci}$', fontdict=FONT, fontsize=20)
    ax.set_ylabel(r'$\Delta K_i/\Delta K_e$', fontdict=FONT, fontsize=20)
    ax.legend(loc=1, prop={'size': 16}, ncol=1,
              shadow=False, fancybox=False, frameon=False)
    fdir = '../img/img_high_mime/'
    mkdir_p(fdir)
    if const_va:
        fname = fdir + 'internal_epartition_' + bg_str + '_high.pdf'
    else:
        fname = fdir + 'internal_epartition_' + bg_str + '.pdf'
    fig.savefig(fname)
    plt.show()


def plot_jy(tframe, const_va=False, high_bg=False, show_plot=True):
    """Plot out-of-plane current density for different runs
    Args:
        tframe: time frame
        const_va: whether the Alfven speed is the same for different mass ratio
        high_bg: whether to include runs with higher guide field
    """
    if high_bg:
        bgs = np.asarray([0.0, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4])
    else:
        bgs = np.asarray([0.0, 0.2, 0.4, 0.8])
    mimes = np.asarray([25, 100, 400])
    # mimes = np.asarray([25])
    if const_va:
        dmins = np.asarray([-0.025, -0.025, -0.025])
        dmaxs = np.asarray([0.075, 0.075, 0.075])
        lmins = np.asarray([-0.02, -0.02, -0.02])
        lmaxs = np.asarray([0.06, 0.06, 0.06])
    else:
        dmins = np.asarray([-0.1, -0.05, -0.025])
        dmaxs = np.asarray([0.3, 0.15, 0.075])
        lmins = np.asarray([-0.1, -0.04, -0.02])
        lmaxs = np.asarray([0.3, 0.12, 0.06])
    if high_bg:
        fig = plt.figure(figsize=[7, 5])
        rect0 = [0.12, 0.84, 0.27, 0.1]
        hgap, vgap = 0.022, 0.012
    else:
        fig = plt.figure(figsize=[7, 3])
        rect0 = [0.12, 0.76, 0.27, 0.16]
        hgap, vgap = 0.022, 0.02
    rect = np.copy(rect0)
    nbg, = bgs.shape
    nmime, = mimes.shape
    tshifts = shift_tframes(const_va)
    for ibg, bg in enumerate(bgs):
        rect[1] = rect0[1] - ibg * (vgap + rect0[3])
        for imime, mime in enumerate(mimes):
            bg_str = str(int(bg * 10)).zfill(2)
            run_name = 'mime' + str(mime) + '_beta002_' + 'bg' + bg_str
            if const_va and mime != 400:
                run_name += '_high'
            picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
            pic_info = read_data_from_json(picinfo_fname)
            smime = math.sqrt(pic_info.mime)
            tframe_shift = tframe + tshifts[str(mime)]
            kwargs = {"current_time": tframe_shift,
                      "xl": 0, "xr": pic_info.lx_di,
                      "zb": -pic_info.lz_di*0.25, "zt": pic_info.lz_di*0.25}
            fname = pic_info.run_dir + "data/jy.gda"
            x, z, jy = read_2d_fields(pic_info, fname, **kwargs)
            sizes = [x[0], x[-1], z[0], z[-1]]
            print("Min and Max of Jy: %f %f" % (np.min(jy), np.max(jy)))
            fname = pic_info.run_dir + "data/Ay.gda"
            x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
            rect[0] = rect0[0] + imime * (hgap + rect0[2])
            ax = fig.add_axes(rect)
            ax.tick_params(bottom=True, top=False, left=True, right=True)
            ax.tick_params(axis='x', which='minor', direction='in', top=True)
            ax.tick_params(axis='x', which='major', direction='in')
            ax.tick_params(axis='y', which='minor', direction='in')
            ax.tick_params(axis='y', which='major', direction='in')
            p1 = ax.imshow(jy, vmin=dmins[imime], vmax=dmaxs[imime],
                           extent=sizes, cmap=plt.cm.inferno, aspect='auto',
                           origin='lower', interpolation='bicubic')
            ax.contour(x, z, Ay, colors='k', linewidths=0.5)
            ax.tick_params(labelsize=8)
            if ibg < nbg - 1:
                ax.tick_params(axis='x', labelbottom=False)
            else:
                ax.set_xlabel(r'$x/d_i$', fontsize=10)
            if imime > 0:
                ax.tick_params(axis='y', labelleft=False)
            else:
                ax.set_ylabel(r'$z/d_i$', fontsize=10)

            if ibg == 0:
                title = r"$m_i/m_e=" + str(mime) + "$"
                plt.title(title, fontsize=10)
            if ibg == nbg - 1:
                if (const_va and mime == 100) or not const_va:
                    rect_cbar = np.copy(rect)
                    if const_va:
                        rect_cbar[0] = rect[0] - rect[2] * 0.5
                        rect_cbar[2] = rect[2] + rect[2]
                    if high_bg:
                        rect_cbar[1] = rect[1] - vgap * 9
                    else:
                        rect_cbar[1] = rect[1] - vgap * 7
                    rect_cbar[3] = 0.02
                    cbar_ax = fig.add_axes(rect_cbar)
                    cbar_ax.tick_params(bottom=True )
                    # cbar_ax.tick_params(axis='x', which='minor', direction='in')
                    # cbar_ax.tick_params(axis='x', which='major', direction='in')
                    cbar = fig.colorbar(p1, cax=cbar_ax, orientation='horizontal',
                                        extend='both')
                    cbar.set_ticks(np.linspace(lmins[imime], lmaxs[imime], num=5))
                    cbar.ax.tick_params(labelsize=8)
            if imime == 0:
                if ibg == 0:
                    ax.text(-0.35, 0.9, r'$B_g$', color='k', fontsize=10,
                            bbox=dict(facecolor='none', alpha=1.0,
                                      edgecolor='none', pad=10.0),
                            horizontalalignment='center',
                            verticalalignment='center',
                            transform=ax.transAxes)
                text = r"$" + str(bg) + "$"
                ax.text(-0.35, 0.5, text, color='k', fontsize=10,
                        bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes)

    fdir = '../img/img_high_mime/jy/'
    if high_bg:
        fdir += 'high_bg/'
    mkdir_p(fdir)
    if const_va:
        fname = fdir + 'jys_' + str(tframe) + '_high.pdf'
    else:
        fname = fdir + 'jys_' + str(tframe) + '.pdf'
    fig.savefig(fname, dpi=200)
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_va(tframe, show_plot=True):
    """Plot the Alfven speed
    """
    bgs = np.asarray([0.0, 0.2, 0.4, 0.8])
    # mimes = np.asarray([25, 100, 400])
    mimes = np.asarray([400])
    dmins = np.asarray([0.5, 0.5, 0.5])
    dmaxs = np.asarray([1.5, 1.5, 1.5])
    lmins = dmins
    lmaxs = dmaxs
    fig = plt.figure(figsize=[7, 3])
    rect0 = [0.12, 0.76, 0.27, 0.16]
    rect = np.copy(rect0)
    hgap, vgap = 0.022, 0.02
    nbg, = bgs.shape
    nmime, = mimes.shape
    for ibg, bg in enumerate(bgs):
        rect[1] = rect0[1] - ibg * (vgap + rect0[3])
        for imime, mime in enumerate(mimes):
            bg_str = str(int(bg * 10)).zfill(2)
            run_name = "mime" + str(mime) + "_beta002_bg" + bg_str
            picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
            pic_info = read_data_from_json(picinfo_fname)
            wpe_wce = pic_info.dtwce / pic_info.dtwpe
            va0 = wpe_wce / math.sqrt(pic_info.mime)
            smime = math.sqrt(pic_info.mime)
            tframe_shift = (tframe + 12) if mime != 400 else tframe
            kwargs = {"current_time": tframe_shift,
                      "xl": 0, "xr": pic_info.lx_di,
                      "zb": -pic_info.lz_di*0.25, "zt": pic_info.lz_di*0.25}
            fname = pic_info.run_dir + "data/bx.gda"
            x, z, bx = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_info.run_dir + "data/ni.gda"
            x, z, ni = read_2d_fields(pic_info, fname, **kwargs)
            va = np.abs(bx) / np.sqrt(ni * pic_info.mime)
            va /= va0
            sizes = [x[0], x[-1], z[0], z[-1]]
            print("Min and Max of va %f %f" % (np.min(va), np.max(va)))
            fname = pic_info.run_dir + "data/Ay.gda"
            x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
            rect[0] = rect0[0] + imime * (hgap + rect0[2])
            ax = fig.add_axes(rect)
            ax.tick_params(bottom=True, top=False, left=True, right=True)
            ax.tick_params(axis='x', which='minor', direction='in', top=True)
            ax.tick_params(axis='x', which='major', direction='in')
            ax.tick_params(axis='y', which='minor', direction='in')
            ax.tick_params(axis='y', which='major', direction='in')
            p1 = ax.imshow(va, vmin=dmins[imime], vmax=dmaxs[imime],
                           extent=sizes, cmap=plt.cm.seismic, aspect='auto',
                           origin='lower', interpolation='bicubic')
            ax.contour(x, z, Ay, colors='k', linewidths=0.5)
            ax.tick_params(labelsize=8)
            if ibg < nbg - 1:
                ax.tick_params(axis='x', labelbottom=False)
            else:
                ax.set_xlabel(r'$x/d_i$', fontsize=10)
            if imime > 0:
                ax.tick_params(axis='y', labelleft=False)
            else:
                ax.set_ylabel(r'$z/d_i$', fontsize=10)

            if ibg == 0:
                title = r"$m_i/m_e=" + str(mime) + "$"
                plt.title(title, fontsize=10)
            if ibg == nbg - 1:
                rect_cbar = np.copy(rect)
                rect_cbar[1] = rect[1] - vgap * 7
                rect_cbar[3] = 0.02
                cbar_ax = fig.add_axes(rect_cbar)
                cbar_ax.tick_params(bottom=True )
                # cbar_ax.tick_params(axis='x', which='minor', direction='in')
                # cbar_ax.tick_params(axis='x', which='major', direction='in')
                cbar = fig.colorbar(p1, cax=cbar_ax, orientation='horizontal',
                                    extend='both')
                cbar.set_ticks(np.linspace(lmins[imime], lmaxs[imime], num=5))
                cbar.ax.tick_params(labelsize=8)
            if imime == 0:
                if ibg == 0:
                    ax.text(-0.35, 0.9, r'$B_g$', color='k', fontsize=10,
                            bbox=dict(facecolor='none', alpha=1.0,
                                      edgecolor='none', pad=10.0),
                            horizontalalignment='center',
                            verticalalignment='center',
                            transform=ax.transAxes)
                text = r"$" + str(bg) + "$"
                ax.text(-0.35, 0.5, text, color='k', fontsize=10,
                        bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes)

    # fdir = '../img/img_high_mime/jy/'
    # mkdir_p(fdir)
    # fname = fdir + 'jys_' + str(tframe) + '.pdf'
    # fig.savefig(fname, dpi=200)
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_bulkv(tframe, const_va, show_plot=True):
    """Plot the bulk flow velocity

    Args:
        tframe: time frame
        const_va: whether the Alfven speed is constant for different mass ratio
    """
    # bgs = np.asarray([0.0, 0.2, 0.4, 0.8])
    bgs = np.asarray([0.0])
    mimes = np.asarray([25, 100, 400])
    # mimes = np.asarray([400])
    dmins = np.asarray([-1.0, -1.0, -1.0])
    dmaxs = np.asarray([1.0, 1.0, 1.0])
    lmins = np.asarray([-0.02, -0.02, -0.02])
    lmaxs = np.asarray([0.06, 0.06, 0.06])
    lmins = dmins
    lmaxs = dmaxs
    fig = plt.figure(figsize=[7, 3])
    rect0 = [0.12, 0.76, 0.27, 0.16]
    rect = np.copy(rect0)
    hgap, vgap = 0.022, 0.02
    nbg, = bgs.shape
    nmime, = mimes.shape
    tshifts = shift_tframes(const_va)
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    for ibg, bg in enumerate(bgs):
        rect[1] = rect0[1] - ibg * (vgap + rect0[3])
        ax = fig.add_axes(rect)
        ax.set_prop_cycle('color', COLORS)
        for imime, mime in enumerate(mimes):
            bg_str = str(int(bg * 10)).zfill(2)
            run_name = 'mime' + str(mime) + '_beta002_' + 'bg' + bg_str
            if const_va and mime != 400:
                run_name += '_high'
            picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
            pic_info = read_data_from_json(picinfo_fname)
            wpe_wce = pic_info.dtwce / pic_info.dtwpe
            va0 = wpe_wce / math.sqrt(pic_info.mime)
            smime = math.sqrt(pic_info.mime)
            tframe_shift = tframe + tshifts[str(mime)]
            kwargs = {"current_time": tframe_shift,
                      "xl": 0, "xr": pic_info.lx_di,
                      "zb": -pic_info.lz_di*0.25, "zt": pic_info.lz_di*0.25}
            fname = pic_info.run_dir + "data/vex.gda"
            x, z, vex = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_info.run_dir + "data/vey.gda"
            x, z, vey = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_info.run_dir + "data/vez.gda"
            x, z, vez = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_info.run_dir + "data/vix.gda"
            x, z, vix = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_info.run_dir + "data/viy.gda"
            x, z, viy = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_info.run_dir + "data/viz.gda"
            x, z, viz = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_info.run_dir + "data/ne.gda"
            x, z, ne = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_info.run_dir + "data/ni.gda"
            x, z, ni = read_2d_fields(pic_info, fname, **kwargs)
            irho = 1.0 / (ne + ni * mime)
            vtx = (vex * ne + vix * ni * mime) * irho
            vty = (vey * ne + viy * ni * mime) * irho
            vtz = (vez * ne + viz * ni * mime) * irho
            sizes = [x[0], x[-1], z[0], z[-1]]
            fname = pic_info.run_dir + "data/Ay.gda"
            x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
            nxr, = x.shape
            nzr, = z.shape
            ax.tick_params(bottom=True, top=False, left=True, right=True)
            ax.tick_params(axis='x', which='minor', direction='in', top=True)
            ax.tick_params(axis='x', which='major', direction='in')
            ax.tick_params(axis='y', which='minor', direction='in')
            ax.tick_params(axis='y', which='major', direction='in')
            ax.plot(np.abs(vtx[nzr//2, :]/va0))
            # p1 = ax.imshow(vtx/va0, vmin=dmins[imime], vmax=dmaxs[imime],
            #                extent=sizes, cmap=plt.cm.seismic, aspect='auto',
            #                origin='lower', interpolation='bicubic')
            # ax.contour(x, z, Ay, colors='k', linewidths=0.5)
            # ax.tick_params(labelsize=8)
            # if ibg < nbg - 1:
            #     ax.tick_params(axis='x', labelbottom=False)
            # else:
            #     ax.set_xlabel(r'$x/d_i$', fontsize=10)
            # if imime > 0:
            #     ax.tick_params(axis='y', labelleft=False)
            # else:
            #     ax.set_ylabel(r'$z/d_i$', fontsize=10)

            # if ibg == 0:
            #     title = r"$m_i/m_e=" + str(mime) + "$"
            #     plt.title(title, fontsize=10)
            # if ibg == nbg - 1:
            #     if (const_va and mime == 100) or not const_va:
            #         rect_cbar = np.copy(rect)
            #         if const_va:
            #             rect_cbar[0] = rect[0] - rect[2] * 0.5
            #             rect_cbar[2] = rect[2] + rect[2]
            #         rect_cbar[1] = rect[1] - vgap * 7
            #         rect_cbar[3] = 0.02
            #         cbar_ax = fig.add_axes(rect_cbar)
            #         cbar_ax.tick_params(bottom=True )
            #         # cbar_ax.tick_params(axis='x', which='minor', direction='in')
            #         # cbar_ax.tick_params(axis='x', which='major', direction='in')
            #         cbar = fig.colorbar(p1, cax=cbar_ax, orientation='horizontal',
            #                             extend='both')
            #         cbar.set_ticks(np.linspace(lmins[imime], lmaxs[imime], num=5))
            #         cbar.ax.tick_params(labelsize=8)
            # if imime == 0:
            #     if ibg == 0:
            #         ax.text(-0.35, 0.9, r'$B_g$', color='k', fontsize=10,
            #                 bbox=dict(facecolor='none', alpha=1.0,
            #                           edgecolor='none', pad=10.0),
            #                 horizontalalignment='center',
            #                 verticalalignment='center',
            #                 transform=ax.transAxes)
            #     text = r"$" + str(bg) + "$"
            #     ax.text(-0.35, 0.5, text, color='k', fontsize=10,
            #             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            #             horizontalalignment='center', verticalalignment='center',
            #             transform=ax.transAxes)

    # fdir = '../img/img_high_mime/jy/'
    # mkdir_p(fdir)
    # if const_va:
    #     fname = fdir + 'jys_' + str(tframe) + '_high.pdf'
    # else:
    #     fname = fdir + 'jys_' + str(tframe) + '.pdf'
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_pressure_anisotropy(plot_config, show_plot=True):
    """Plot pressure anisotropy for different runs
    """
    tframe = plot_config["tframe"]
    species = plot_config["species"]
    bgs = np.asarray([0.0, 0.2, 0.4, 0.8])
    # bgs = np.asarray([0.8])
    mimes = np.asarray([25, 100, 400])
    # mimes = np.asarray([400])
    dmins = np.asarray([0.5, 0.5, 0.5])
    dmaxs = np.asarray([8, 6, 4])
    lmins = dmins
    lmaxs = dmaxs
    fig = plt.figure(figsize=[7, 3])
    rect0 = [0.12, 0.76, 0.27, 0.16]
    rect = np.copy(rect0)
    hgap, vgap = 0.022, 0.02
    nbg, = bgs.shape
    nmime, = mimes.shape
    for ibg, bg in enumerate(bgs):
        rect[1] = rect0[1] - ibg * (vgap + rect0[3])
        for imime, mime in enumerate(mimes):
            bg_str = str(int(bg * 10)).zfill(2)
            run_name = "mime" + str(mime) + "_beta002_bg" + bg_str
            picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
            pic_info = read_data_from_json(picinfo_fname)
            smime = math.sqrt(pic_info.mime)
            tframe_shift = (tframe + 12) if mime != 400 else tframe
            kwargs = {"current_time": tframe_shift,
                      "xl": 0, "xr": pic_info.lx_di,
                      "zb": -pic_info.lz_di*0.25, "zt": pic_info.lz_di*0.25}
            fname = pic_info.run_dir + "data/p" + species + "-xx.gda"
            x, z, pxx = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_info.run_dir + "data/p" + species + "-xy.gda"
            x, z, pxy = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_info.run_dir + "data/p" + species + "-xz.gda"
            x, z, pxz = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_info.run_dir + "data/p" + species + "-yx.gda"
            x, z, pyx = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_info.run_dir + "data/p" + species + "-yy.gda"
            x, z, pyy = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_info.run_dir + "data/p" + species + "-yz.gda"
            x, z, pyz = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_info.run_dir + "data/p" + species + "-zx.gda"
            x, z, pzx = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_info.run_dir + "data/p" + species + "-zy.gda"
            x, z, pzy = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_info.run_dir + "data/p" + species + "-zz.gda"
            x, z, pzz = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_info.run_dir + "data/bx.gda"
            x, z, bx = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_info.run_dir + "data/by.gda"
            x, z, by = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_info.run_dir + "data/bz.gda"
            x, z, bz = read_2d_fields(pic_info, fname, **kwargs)
            ib2 = 1.0 / (bx*bx + by*by + bz*bz)
            ppara = (pxx * bx**2 + pyy * by**2 + pzz * bz**2 +
                     (pxy + pyx) * bx * by +
                     (pxz + pzx) * bx * bz +
                     (pyz + pzy) * by * bz)
            ppara = ppara * ib2
            pperp = 0.5 * (pxx + pyy + pzz - ppara)
            sizes = [x[0], x[-1], z[0], z[-1]]
            fname = pic_info.run_dir + "data/Ay.gda"
            x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
            rect[0] = rect0[0] + imime * (hgap + rect0[2])
            ax = fig.add_axes(rect)
            ax.tick_params(bottom=True, top=False, left=True, right=True)
            ax.tick_params(axis='x', which='minor', direction='in', top=True)
            ax.tick_params(axis='x', which='major', direction='in')
            ax.tick_params(axis='y', which='minor', direction='in')
            ax.tick_params(axis='y', which='major', direction='in')
            p1 = ax.imshow(ppara/pperp,
                           # norm = LogNorm(vmin=dmins[imime], vmax=dmaxs[imime]),
                           vmin=dmins[imime], vmax=dmaxs[imime],
                           extent=sizes, cmap=plt.cm.viridis, aspect='auto',
                           origin='lower', interpolation='bicubic')
            ax.contour(x, z, Ay, colors='k', linewidths=0.5)
            ax.tick_params(labelsize=8)
            if ibg < nbg - 1:
                ax.tick_params(axis='x', labelbottom=False)
            else:
                ax.set_xlabel(r'$x/d_i$', fontsize=10)
            if imime > 0:
                ax.tick_params(axis='y', labelleft=False)
            else:
                ax.set_ylabel(r'$z/d_i$', fontsize=10)

            if ibg == 0:
                title = r"$m_i/m_e=" + str(mime) + "$"
                plt.title(title, fontsize=10)
            if ibg == nbg - 1:
                rect_cbar = np.copy(rect)
                rect_cbar[1] = rect[1] - vgap * 7
                rect_cbar[3] = 0.02
                cbar_ax = fig.add_axes(rect_cbar)
                cbar_ax.tick_params(bottom=True )
                cbar_ax.tick_params(axis='x', which='minor', direction='in')
                cbar_ax.tick_params(axis='x', which='major', direction='out')
                cbar = fig.colorbar(p1, cax=cbar_ax, orientation='horizontal',
                                    extend='both')
                # cbar.set_ticks(np.linspace(lmins[imime], lmaxs[imime], num=5))
                cbar.ax.tick_params(labelsize=8)
            if imime == 0:
                if ibg == 0:
                    ax.text(-0.35, 0.9, r'$B_g$', color='k', fontsize=10,
                            bbox=dict(facecolor='none', alpha=1.0,
                                      edgecolor='none', pad=10.0),
                            horizontalalignment='center',
                            verticalalignment='center',
                            transform=ax.transAxes)
                text = r"$" + str(bg) + "$"
                ax.text(-0.35, 0.5, text, color='k', fontsize=10,
                        bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes)

    fdir = '../img/img_high_mime/anisotropy/'
    mkdir_p(fdir)
    fname = fdir + 'anisotropy_' + species + '_' + str(tframe) + '.pdf'
    fig.savefig(fname, dpi=300)
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_ene2d(plot_config, show_plot=True):
    """Plot 2D energization terms
    """
    tframe = plot_config["tframe"]
    species = plot_config["species"]
    # bgs = np.asarray([0.0, 0.2, 0.4, 0.8])
    bgs = np.asarray([0.4])
    # mimes = np.asarray([25, 100, 400])
    mimes = np.asarray([25])
    dmins = np.asarray([-2E-4, -0.05, -0.025])
    dmaxs = -dmins
    lmins = np.asarray([-0.1, -0.04, -0.02])
    lmaxs = np.asarray([0.3, 0.12, 0.06])
    fig = plt.figure(figsize=[7, 3])
    rect0 = [0.12, 0.16, 0.77, 0.76]
    rect = np.copy(rect0)
    hgap, vgap = 0.022, 0.02
    nbg, = bgs.shape
    nmime, = mimes.shape
    ng = 5
    kernel = np.ones((ng,ng)) / float(ng*ng)
    for ibg, bg in enumerate(bgs):
        rect[1] = rect0[1] - ibg * (vgap + rect0[3])
        for imime, mime in enumerate(mimes):
            bg_str = str(int(bg * 10)).zfill(2)
            run_name = "mime" + str(mime) + "_beta002_bg" + bg_str
            picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
            pic_info = read_data_from_json(picinfo_fname)
            smime = math.sqrt(pic_info.mime)
            dx = pic_info.dx_di * smime
            dz = pic_info.dz_di * smime
            tframe_shift = (tframe + 12) if mime != 400 else tframe
            kwargs = {"current_time": tframe_shift,
                      "xl": 0, "xr": pic_info.lx_di,
                      "zb": -pic_info.lz_di*0.25, "zt": pic_info.lz_di*0.25}
            fname = pic_info.run_dir + "data/p" + species + "-xx.gda"
            x, z, pxx = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_info.run_dir + "data/p" + species + "-xy.gda"
            x, z, pxy = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_info.run_dir + "data/p" + species + "-xz.gda"
            x, z, pxz = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_info.run_dir + "data/p" + species + "-yx.gda"
            x, z, pyx = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_info.run_dir + "data/p" + species + "-yy.gda"
            x, z, pyy = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_info.run_dir + "data/p" + species + "-yz.gda"
            x, z, pyz = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_info.run_dir + "data/p" + species + "-zx.gda"
            x, z, pzx = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_info.run_dir + "data/p" + species + "-zy.gda"
            x, z, pzy = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_info.run_dir + "data/p" + species + "-zz.gda"
            x, z, pzz = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_info.run_dir + "data/ex.gda"
            x, z, ex = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_info.run_dir + "data/ey.gda"
            x, z, ey = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_info.run_dir + "data/ez.gda"
            x, z, ez = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_info.run_dir + "data/bx.gda"
            x, z, bx = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_info.run_dir + "data/by.gda"
            x, z, by = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_info.run_dir + "data/bz.gda"
            x, z, bz = read_2d_fields(pic_info, fname, **kwargs)
            ib2 = 1.0 / (bx*bx + by*by + bz*bz)
            ppara = (pxx * bx**2 + pyy * by**2 + pzz * bz**2 +
                     (pxy + pyx) * bx * by +
                     (pxz + pzx) * bx * bz +
                     (pyz + pzy) * by * bz)
            ppara = ppara * ib2
            pperp = 0.5 * (pxx + pyy + pzz - ppara)
            vexb_x = (ey * bz - ez * by) * ib2
            vexb_y = (ez * bx - ex * bz) * ib2
            vexb_z = (ex * by - ey * bx) * ib2
            sizes = [x[0], x[-1], z[0], z[-1]]
            divpt_vexb = ((np.gradient(pxx, dx, axis=1) +
                           np.gradient(pxz, dz, axis=0)) * vexb_x +
                          (np.gradient(pyx, dx, axis=1) +
                           np.gradient(pyz, dz, axis=0)) * vexb_y +
                          (np.gradient(pzx, dx, axis=1) +
                           np.gradient(pzz, dz, axis=0)) * vexb_z)
            # gyrotropic pressure tensor
            pdiff = ppara - pperp
            pxx = pperp + pdiff * bx * bx * ib2
            pxz = pdiff * bx * bz * ib2
            pyx = pdiff * by * bx * ib2
            pyz = pdiff * by * bz * ib2
            pzx = pdiff * bz * bx * ib2
            pzz = pperp + pdiff * bz * bz * ib2
            divpg_vexb = ((np.gradient(pxx, dx, axis=1) +
                           np.gradient(pxz, dz, axis=0)) * vexb_x +
                          (np.gradient(pyx, dx, axis=1) +
                           np.gradient(pyz, dz, axis=0)) * vexb_y +
                          (np.gradient(pzx, dx, axis=1) +
                           np.gradient(pzz, dz, axis=0)) * vexb_z)
            l1 = pxx + pyy + pzz
            l2 = (pxx * pyy + pxx * pzz + pyy * pzz -
                  pxy * pyx - pxz * pzx - pyz * pzy)
            q = 1 - 4 * l2 / ((l1 - ppara) * (l1 + 3 * ppara))
            # fname = pic_info.run_dir + "data/Ay.gda"
            # x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
            rect[0] = rect0[0] + imime * (hgap + rect0[2])
            ax = fig.add_axes(rect)
            ax.tick_params(bottom=True, top=False, left=True, right=True)
            ax.tick_params(axis='x', which='minor', direction='in', top=True)
            ax.tick_params(axis='x', which='major', direction='in')
            ax.tick_params(axis='y', which='minor', direction='in')
            ax.tick_params(axis='y', which='major', direction='in')
            fdata = divpt_vexb - divpg_vexb
            fdata = signal.convolve2d(fdata, kernel, mode='same')
            # p1 = ax.imshow(fdata, vmin=dmins[imime], vmax=dmaxs[imime],
                           # extent=sizes, cmap=plt.cm.seismic, aspect='auto',
            p1 = ax.imshow(q, vmin=0, vmax=0.2,
                           extent=sizes, cmap=plt.cm.viridis, aspect='auto',
                           origin='lower', interpolation='bicubic')
            # ax.contour(x, z, Ay, colors='k', linewidths=0.5)
            ax.tick_params(labelsize=8)
            if ibg < nbg - 1:
                ax.tick_params(axis='x', labelbottom=False)
            else:
                ax.set_xlabel(r'$x/d_i$', fontsize=10)
            if imime > 0:
                ax.tick_params(axis='y', labelleft=False)
            else:
                ax.set_ylabel(r'$z/d_i$', fontsize=10)

            if ibg == 0:
                title = r"$m_i/m_e=" + str(mime) + "$"
                plt.title(title, fontsize=10)
            if ibg == nbg - 1:
                rect_cbar = np.copy(rect)
                rect_cbar[1] = rect[1] - vgap * 7
                rect_cbar[3] = 0.02
                cbar_ax = fig.add_axes(rect_cbar)
                cbar_ax.tick_params(bottom=True )
                cbar_ax.tick_params(axis='x', which='minor', direction='in')
                cbar_ax.tick_params(axis='x', which='major', direction='in')
                cbar = fig.colorbar(p1, cax=cbar_ax, orientation='horizontal')
                cbar.set_ticks(np.linspace(lmins[imime], lmaxs[imime], num=5))
                cbar.ax.tick_params(labelsize=8)
            if imime == 0:
                if ibg == 0:
                    ax.text(-0.35, 0.9, r'$B_g$', color='k', fontsize=10,
                            bbox=dict(facecolor='none', alpha=1.0,
                                      edgecolor='none', pad=10.0),
                            horizontalalignment='center',
                            verticalalignment='center',
                            transform=ax.transAxes)
                text = r"$" + str(bg) + "$"
                ax.text(-0.35, 0.5, text, color='k', fontsize=10,
                        bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes)

    # fdir = '../img/img_high_mime/jy/'
    # mkdir_p(fdir)
    # fname = fdir + 'jys_' + str(tframe) + '.pdf'
    # fig.savefig(fname, dpi=200)
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_agyq(plot_config, show_plot=True):
    """Plot Q parameters defined by M. Swisdak to quantify the agyrotropy
    """
    tframe = plot_config["tframe"]
    species = plot_config["species"]
    bgs = np.asarray([0.0, 0.2, 0.4, 0.8])
    # bgs = np.asarray([0.4])
    mimes = np.asarray([25, 100, 400])
    # mimes = np.asarray([25])
    dmins = np.asarray([0, 0, 0])
    if species == 'e':
        dmaxs = np.asarray([0.1, 0.1, 0.1])
    else:
        dmaxs = np.asarray([0.3, 0.3, 0.3])
    lmins = np.copy(dmins)
    lmaxs = np.copy(dmaxs)
    fig = plt.figure(figsize=[7, 3])
    rect0 = [0.12, 0.76, 0.27, 0.16]
    rect = np.copy(rect0)
    hgap, vgap = 0.022, 0.02
    nbg, = bgs.shape
    nmime, = mimes.shape
    ng = 5
    kernel = np.ones((ng,ng)) / float(ng*ng)
    for ibg, bg in enumerate(bgs):
        rect[1] = rect0[1] - ibg * (vgap + rect0[3])
        for imime, mime in enumerate(mimes):
            bg_str = str(int(bg * 10)).zfill(2)
            run_name = "mime" + str(mime) + "_beta002_bg" + bg_str
            picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
            pic_info = read_data_from_json(picinfo_fname)
            smime = math.sqrt(pic_info.mime)
            dx = pic_info.dx_di * smime
            dz = pic_info.dz_di * smime
            tframe_shift = (tframe + 12) if mime != 400 else tframe
            kwargs = {"current_time": tframe_shift,
                      "xl": 0, "xr": pic_info.lx_di,
                      "zb": -pic_info.lz_di*0.25, "zt": pic_info.lz_di*0.25}
            fname = pic_info.run_dir + "data/p" + species + "-xx.gda"
            x, z, pxx = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_info.run_dir + "data/p" + species + "-xy.gda"
            x, z, pxy = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_info.run_dir + "data/p" + species + "-xz.gda"
            x, z, pxz = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_info.run_dir + "data/p" + species + "-yx.gda"
            x, z, pyx = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_info.run_dir + "data/p" + species + "-yy.gda"
            x, z, pyy = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_info.run_dir + "data/p" + species + "-yz.gda"
            x, z, pyz = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_info.run_dir + "data/p" + species + "-zx.gda"
            x, z, pzx = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_info.run_dir + "data/p" + species + "-zy.gda"
            x, z, pzy = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_info.run_dir + "data/p" + species + "-zz.gda"
            x, z, pzz = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_info.run_dir + "data/bx.gda"
            x, z, bx = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_info.run_dir + "data/by.gda"
            x, z, by = read_2d_fields(pic_info, fname, **kwargs)
            fname = pic_info.run_dir + "data/bz.gda"
            x, z, bz = read_2d_fields(pic_info, fname, **kwargs)
            ib2 = 1.0 / (bx*bx + by*by + bz*bz)
            ppara = (pxx * bx**2 + pyy * by**2 + pzz * bz**2 +
                     (pxy + pyx) * bx * by +
                     (pxz + pzx) * bx * bz +
                     (pyz + pzy) * by * bz)
            ppara = ppara * ib2
            l1 = pxx + pyy + pzz
            l2 = (pxx * pyy + pxx * pzz + pyy * pzz -
                  pxy * pyx - pxz * pzx - pyz * pzy)
            q = 1 - 4 * l2 / ((l1 - ppara) * (l1 + 3 * ppara))
            # fname = pic_info.run_dir + "data/Ay.gda"
            # x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
            rect[0] = rect0[0] + imime * (hgap + rect0[2])
            ax = fig.add_axes(rect)
            ax.tick_params(bottom=True, top=False, left=True, right=True)
            ax.tick_params(axis='x', which='minor', direction='in', top=True)
            ax.tick_params(axis='x', which='major', direction='in')
            ax.tick_params(axis='y', which='minor', direction='in')
            ax.tick_params(axis='y', which='major', direction='in')
            sizes = [x[0], x[-1], z[0], z[-1]]
            p1 = ax.imshow(q, vmin=dmins[imime], vmax=dmaxs[imime],
                           extent=sizes, cmap=plt.cm.Greys, aspect='auto',
                           origin='lower', interpolation='bicubic')
            # ax.contour(x, z, Ay, colors='k', linewidths=0.5)
            ax.tick_params(labelsize=8)
            if ibg < nbg - 1:
                ax.tick_params(axis='x', labelbottom=False)
            else:
                ax.set_xlabel(r'$x/d_i$', fontsize=10)
            if imime > 0:
                ax.tick_params(axis='y', labelleft=False)
            else:
                ax.set_ylabel(r'$z/d_i$', fontsize=10)

            if ibg == 0:
                title = r"$m_i/m_e=" + str(mime) + "$"
                plt.title(title, fontsize=10)
            if ibg == nbg - 1:
                rect_cbar = np.copy(rect)
                rect_cbar[1] = rect[1] - vgap * 7
                rect_cbar[3] = 0.02
                cbar_ax = fig.add_axes(rect_cbar)
                cbar_ax.tick_params(bottom=True )
                cbar_ax.tick_params(axis='x', which='minor', direction='in')
                cbar_ax.tick_params(axis='x', which='major', direction='in')
                cbar = fig.colorbar(p1, cax=cbar_ax, orientation='horizontal')
                cbar.set_ticks(np.linspace(lmins[imime], lmaxs[imime], num=5))
                cbar.ax.tick_params(labelsize=8)
            if imime == 0:
                if ibg == 0:
                    ax.text(-0.35, 0.9, r'$B_g$', color='k', fontsize=10,
                            bbox=dict(facecolor='none', alpha=1.0,
                                      edgecolor='none', pad=10.0),
                            horizontalalignment='center',
                            verticalalignment='center',
                            transform=ax.transAxes)
                text = r"$" + str(bg) + "$"
                ax.text(-0.35, 0.5, text, color='k', fontsize=10,
                        bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes)

    # fdir = '../img/img_high_mime/jy/'
    # mkdir_p(fdir)
    # fname = fdir + 'jys_' + str(tframe) + '.pdf'
    # fig.savefig(fname, dpi=200)
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_temp(plot_config, show_plot=True):
    """Plot plasma temperature
    """
    mime = plot_config["mime"]
    bg = plot_config["bg"]
    tframe = plot_config["tframe"]
    species = plot_config["species"]

    bg_str = str(int(bg * 10)).zfill(2)
    run_name = "mime" + str(mime) + "_beta002_bg" + bg_str
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    if species == 'e':
        vth = pic_info.vthe
    else:
        vth = pic_info.vthi
    gama = 1.0 / math.sqrt(1.0 - 3 * vth**2)
    eth = gama - 1.0
    temp0 = eth / 1.5
    smime = math.sqrt(pic_info.mime)
    dx = pic_info.dx_di * smime
    dz = pic_info.dz_di * smime
    kene_e = pic_info.kene_e
    kene_i = pic_info.kene_i
    ene_magnetic = pic_info.ene_magnetic
    tenergy = pic_info.tenergy

    kwargs = {"current_time": tframe,
              "xl": 0, "xr": pic_info.lx_di,
              "zb": -pic_info.lz_di*0.25, "zt": pic_info.lz_di*0.25}
    fname = pic_info.run_dir + "data/p" + species + "-xx.gda"
    x, z, pxx = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_info.run_dir + "data/p" + species + "-yy.gda"
    x, z, pyy = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_info.run_dir + "data/p" + species + "-zz.gda"
    x, z, pzz = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_info.run_dir + "data/n" + species + ".gda"
    x, z, nrho = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_info.run_dir + "data/Ay.gda"
    x, z, ay = read_2d_fields(pic_info, fname, **kwargs)
    temp = (pxx + pyy + pzz) / (3 * nrho)
    fig = plt.figure(figsize=[7, 4])
    rect = [0.09, 0.6, 0.82, 0.36]
    hgap, vgap = 0.022, 0.02
    ax = fig.add_axes(rect)
    ax.tick_params(bottom=True, top=False, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top=True)
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    sizes = [x[0], x[-1], z[0], z[-1]]
    p1 = ax.imshow(temp/temp0, vmin=1, vmax=10,
                   extent=sizes, cmap=plt.cm.inferno, aspect='auto',
                   origin='lower', interpolation='bicubic')
    nx, nz = len(x), len(z)
    ax.contour(x, z[:nz//2], ay[:nz//2, :], colors='w',
               linewidths=1, linestyles='--')
    ax.contour(x, z[nz//2:], ay[nz//2:, :], colors='w', linewidths=1)
    ax.tick_params(labelsize=8)
    ax.set_xlabel(r'$x/d_i$', fontsize=10)
    ax.set_ylabel(r'$z/d_i$', fontsize=10)
    rect_cbar = np.copy(rect)
    rect_cbar[0] += rect[2] + hgap
    rect_cbar[2] = 0.01
    cbar_ax = fig.add_axes(rect_cbar)
    cbar_ax.tick_params(bottom=True )
    cbar_ax.tick_params(axis='y', which='minor', direction='in')
    cbar_ax.tick_params(axis='y', which='major', direction='in')
    cbar = fig.colorbar(p1, cax=cbar_ax, extend='both')
    cbar.set_ticks(np.linspace(1, 9, num=5))
    label1 = r'$T_' + species + '/T_0$'
    cbar.ax.set_ylabel(label1, fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    hgap1 = 0.1
    rect1 = np.copy(rect)
    rect1[1] = 0.1
    rect1[2] = 0.5 * (rect[2] - hgap1)
    rect1[3] = 0.4
    ax = fig.add_axes(rect1)
    ax.tick_params(bottom=True, top=False, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top=True)
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.tick_params(labelsize=8)

    enorm = ene_magnetic[0]
    ax.set_prop_cycle('color', COLORS)
    p1, = ax.plot(tenergy, (ene_magnetic - ene_magnetic[0]) / enorm,
                  linewidth=2)
    p2, = ax.plot(tenergy, (kene_i - kene_i[0]) / enorm, linewidth=2)
    p3, = ax.plot(tenergy, (kene_e - kene_e[0]) / enorm, linewidth=2)
    ax.set_xlim([0, 200])
    ax.set_ylim([-0.2, 0.15])
    ax.plot([0, 200], [0, 0], linewidth=0.5, linestyle='--', color='k')
    dkm = ene_magnetic[-1] - ene_magnetic[0]
    dke = kene_e[-1] - kene_e[0]
    dki = kene_i[-1] - kene_i[0]
    ylim = ax.get_ylim()
    ylen = ylim[1] - ylim[0]
    twci = tframe * pic_info.dt_fields
    ax.plot([twci, twci], ylim, linewidth=0.5, linestyle='-', color='k')
    height1 = ((kene_i[-1] - kene_i[0]) / enorm - ylim[0]) / ylen - 0.06
    height2 = ((kene_e[-1] - kene_e[0]) / enorm - ylim[0]) / ylen - 0.15
    height3 = ((ene_magnetic[-1] - ene_magnetic[0]) / enorm - ylim[0]) / ylen + 0.27
    ax.text(0.5, height1, 'ion', color=p2.get_color(), rotation=15, fontsize=10,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)
    ax.text(0.5, height2, 'electron', color=p3.get_color(),
            rotation=8, fontsize=10,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)
    ax.text(0.5, height3, 'magnetic', color=p1.get_color(),
            rotation=-20, fontsize=10,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)
    ax.set_xlabel(r'$t\Omega_{ci}$', fontsize=10)

    rect1[0] += rect1[2] + hgap1
    ax = fig.add_axes(rect1)
    ax.tick_params(bottom=True, top=False, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top=True)
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.tick_params(labelsize=8)
    if species == 'h':
        species = 'H'
    spect_info = {"nbins": 800,
                  "emin": 1E-5,
                  "emax": 1E3}
    emin_log = math.log10(spect_info["emin"])
    emax_log = math.log10(spect_info["emax"])
    nbins = spect_info["nbins"]
    delog = (emax_log - emin_log) / nbins
    emin_log = emin_log - delog
    emax_log = emax_log - delog
    elog = np.logspace(emin_log, emax_log, nbins + 1)
    elog_mid = 0.5 * (elog[:-1] + elog[1:])
    elog /= eth
    elog_mid /= eth
    nptot = pic_info.nx * pic_info.ny * pic_info.nz * pic_info.nppc
    fdir = '../data/spectra/' + run_name + '/'
    for tframe1 in range(tframe+1):
        fname = fdir + 'spectrum-' + species.lower() + '.' + str(tframe1)
        flog = np.fromfile(fname)
        flog /= nptot
        if species != 'e':
            flog /= pic_info.mime
        flog[0] = flog[1]  # remove spike at 0
        color = plt.cm.Spectral_r((tframe1 - 0)/float(pic_info.ntf), 1)
        p1, = ax.loglog(elog_mid, flog, linewidth=2, color=color)
        if species == 'e':
            ax.set_xlim([1E-1, 5E2])
        else:
            ax.set_xlim([1E-1, 1E3])
        ax.set_ylim([1E-8, 1E2])
        ax.tick_params(labelsize=8)
        ax.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$', fontsize=10)
        ax.set_ylabel(r'$f(\varepsilon)$', fontsize=10)

    fdir = '../img/img_high_mime/temp/' + run_name + '/'
    mkdir_p(fdir)
    fname = fdir + 'temp_' + str(tframe) + '.pdf'
    fig.savefig(fname, dpi=300)
    fname = fdir + 'temp_' + str(tframe) + '.jpg'
    fig.savefig(fname, dpi=300)
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_nrho(plot_config, show_plot=True):
    """Plot plasma density
    """
    mime = plot_config["mime"]
    bg = plot_config["bg"]
    tframe = plot_config["tframe"]
    species = plot_config["species"]

    bg_str = str(int(bg * 10)).zfill(2)
    run_name = "mime" + str(mime) + "_beta002_bg" + bg_str
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    if species == 'e':
        vth = pic_info.vthe
    else:
        vth = pic_info.vthi
    gama = 1.0 / math.sqrt(1.0 - 3 * vth**2)
    eth = gama - 1.0
    temp0 = eth / 1.5
    smime = math.sqrt(pic_info.mime)
    dx = pic_info.dx_di * smime
    dz = pic_info.dz_di * smime
    kene_e = pic_info.kene_e
    kene_i = pic_info.kene_i
    ene_magnetic = pic_info.ene_magnetic
    tenergy = pic_info.tenergy

    kwargs = {"current_time": tframe,
              "xl": 0, "xr": pic_info.lx_di,
              "zb": -pic_info.lz_di*0.25, "zt": pic_info.lz_di*0.25}
    fname = pic_info.run_dir + "data/n" + species + ".gda"
    x, z, nrho = read_2d_fields(pic_info, fname, **kwargs)
    fname = pic_info.run_dir + "data/Ay.gda"
    x, z, ay = read_2d_fields(pic_info, fname, **kwargs)
    fig = plt.figure(figsize=[7, 4])
    rect = [0.09, 0.6, 0.82, 0.36]
    hgap, vgap = 0.022, 0.02
    ax = fig.add_axes(rect)
    ax.tick_params(bottom=True, top=False, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top=True)
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    sizes = [x[0], x[-1], z[0], z[-1]]
    p1 = ax.imshow(nrho, vmin=0.5, vmax=3,
                   extent=sizes, cmap=plt.cm.inferno, aspect='auto',
                   origin='lower', interpolation='bicubic')
    nx, nz = len(x), len(z)
    # ax.contour(x, z[:nz//2], ay[:nz//2, :], colors='w',
    #            linewidths=1, linestyles='--')
    # ax.contour(x, z[nz//2:], ay[nz//2:, :], colors='w', linewidths=1)
    ax.contour(x, z, ay, colors='w', linewidths=1)
    ax.tick_params(labelsize=8)
    ax.set_xlabel(r'$x/d_i$', fontsize=10)
    ax.set_ylabel(r'$z/d_i$', fontsize=10)
    rect_cbar = np.copy(rect)
    rect_cbar[0] += rect[2] + hgap
    rect_cbar[2] = 0.01
    cbar_ax = fig.add_axes(rect_cbar)
    cbar_ax.tick_params(bottom=True )
    cbar_ax.tick_params(axis='y', which='minor', direction='in')
    cbar_ax.tick_params(axis='y', which='major', direction='in')
    cbar = fig.colorbar(p1, cax=cbar_ax, extend='both')
    cbar.set_ticks(np.linspace(1, 3, num=3))
    label1 = r'$n_' + species + '/n_0$'
    cbar.ax.set_ylabel(label1, fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    hgap1 = 0.1
    rect1 = np.copy(rect)
    rect1[1] = 0.1
    rect1[2] = 0.5 * (rect[2] - hgap1)
    rect1[3] = 0.4
    ax = fig.add_axes(rect1)
    ax.tick_params(bottom=True, top=False, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top=True)
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.tick_params(labelsize=8)

    enorm = ene_magnetic[0]
    ax.set_prop_cycle('color', COLORS)
    p1, = ax.plot(tenergy, (ene_magnetic - ene_magnetic[0]) / enorm,
                  linewidth=2)
    p2, = ax.plot(tenergy, (kene_i - kene_i[0]) / enorm, linewidth=2)
    p3, = ax.plot(tenergy, (kene_e - kene_e[0]) / enorm, linewidth=2)
    ax.set_xlim([0, 200])
    ax.set_ylim([-0.2, 0.15])
    ax.plot([0, 200], [0, 0], linewidth=0.5, linestyle='--', color='k')
    dkm = ene_magnetic[-1] - ene_magnetic[0]
    dke = kene_e[-1] - kene_e[0]
    dki = kene_i[-1] - kene_i[0]
    ylim = ax.get_ylim()
    ylen = ylim[1] - ylim[0]
    twci = tframe * pic_info.dt_fields
    ax.plot([twci, twci], ylim, linewidth=0.5, linestyle='-', color='k')
    height1 = ((kene_i[-1] - kene_i[0]) / enorm - ylim[0]) / ylen - 0.06
    height2 = ((kene_e[-1] - kene_e[0]) / enorm - ylim[0]) / ylen - 0.15
    height3 = ((ene_magnetic[-1] - ene_magnetic[0]) / enorm - ylim[0]) / ylen + 0.27
    ax.text(0.5, height1, 'ion', color=p2.get_color(), rotation=15, fontsize=12,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)
    ax.text(0.5, height2, 'electron', color=p3.get_color(),
            rotation=8, fontsize=12,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)
    ax.text(0.5, height3, 'magnetic', color=p1.get_color(),
            rotation=-20, fontsize=12,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)
    ax.set_xlabel(r'$t\Omega_{ci}$', fontsize=10)

    rect1[0] += rect1[2] + hgap1
    ax = fig.add_axes(rect1)
    ax.tick_params(bottom=True, top=False, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top=True)
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.tick_params(labelsize=8)
    if species == 'h':
        species = 'H'
    spect_info = {"nbins": 800,
                  "emin": 1E-5,
                  "emax": 1E3}
    emin_log = math.log10(spect_info["emin"])
    emax_log = math.log10(spect_info["emax"])
    nbins = spect_info["nbins"]
    delog = (emax_log - emin_log) / nbins
    emin_log = emin_log - delog
    emax_log = emax_log - delog
    elog = np.logspace(emin_log, emax_log, nbins + 1)
    elog_mid = 0.5 * (elog[:-1] + elog[1:])
    elog /= eth
    elog_mid /= eth
    nptot = pic_info.nx * pic_info.ny * pic_info.nz * pic_info.nppc
    fdir = '../data/spectra/' + run_name + '/'
    for tframe1 in range(tframe+1):
        fname = fdir + 'spectrum-' + species.lower() + '.' + str(tframe1)
        flog = np.fromfile(fname)
        flog /= nptot
        if species != 'e':
            flog /= pic_info.mime
        flog[0] = flog[1]  # remove spike at 0
        color = plt.cm.Spectral_r((tframe1 - 0)/float(pic_info.ntf), 1)
        p1, = ax.loglog(elog_mid, flog, linewidth=2, color=color)
        if species == 'e':
            ax.set_xlim([1E-1, 5E2])
        else:
            ax.set_xlim([1E-1, 1E3])
        ax.set_ylim([1E-8, 1E2])
        ax.tick_params(labelsize=8)
        ax.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$', fontsize=10)
        ax.set_ylabel(r'$f(\varepsilon)$', fontsize=10)

    fdir = '../img/img_high_mime/nrho/' + run_name + '/'
    mkdir_p(fdir)
    fname = fdir + 'n' + species + '_' + str(tframe) + '.pdf'
    fig.savefig(fname, dpi=300)
    fname = fdir + 'n' + species + '_' + str(tframe) + '.jpg'
    fig.savefig(fname, dpi=300)
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_agyq_bg(plot_config, show_plot=True):
    """Plot Q parameters defined by M. Swisdak to quantify the agyrotropy
    """
    tframe = plot_config["tframe"]
    species = plot_config["species"]
    bg = plot_config["bg"]
    mimes = np.asarray([25, 100, 400])
    # mimes = np.asarray([25])
    dmins = np.asarray([0, 0, 0])
    if species == 'e':
        dmaxs = np.asarray([0.1, 0.1, 0.1])
    else:
        dmaxs = np.asarray([0.5, 0.5, 0.5])
    lmins = np.copy(dmins)
    lmaxs = np.copy(dmaxs)
    fig = plt.figure(figsize=[3.5, 2.4])
    rect0 = [0.15, 0.71, 0.72, 0.26]
    rect = np.copy(rect0)
    hgap, vgap = 0.022, 0.02
    nmime, = mimes.shape
    ng = 5
    kernel = np.ones((ng,ng)) / float(ng*ng)
    bg_str = str(int(bg * 10)).zfill(2)
    qbins = np.logspace(-2, 0, 201)
    qbins_mid = (qbins[:-1] + qbins[1:]) * 0.5
    qbins_diff = np.diff(qbins)
    qdist = []
    qnorm = [1, 4, 16]
    for imime, mime in enumerate(mimes):
        run_name = "mime" + str(mime) + "_beta002_bg" + bg_str
        picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        smime = math.sqrt(pic_info.mime)
        dx = pic_info.dx_di * smime
        dz = pic_info.dz_di * smime
        tframe_shift = (tframe + 12) if mime != 400 else tframe
        kwargs = {"current_time": tframe_shift,
                  "xl": 0, "xr": pic_info.lx_di,
                  "zb": -pic_info.lz_di*0.25, "zt": pic_info.lz_di*0.25}
        fname = pic_info.run_dir + "data/p" + species + "-xx.gda"
        x, z, pxx = read_2d_fields(pic_info, fname, **kwargs)
        fname = pic_info.run_dir + "data/p" + species + "-xy.gda"
        x, z, pxy = read_2d_fields(pic_info, fname, **kwargs)
        fname = pic_info.run_dir + "data/p" + species + "-xz.gda"
        x, z, pxz = read_2d_fields(pic_info, fname, **kwargs)
        fname = pic_info.run_dir + "data/p" + species + "-yx.gda"
        x, z, pyx = read_2d_fields(pic_info, fname, **kwargs)
        fname = pic_info.run_dir + "data/p" + species + "-yy.gda"
        x, z, pyy = read_2d_fields(pic_info, fname, **kwargs)
        fname = pic_info.run_dir + "data/p" + species + "-yz.gda"
        x, z, pyz = read_2d_fields(pic_info, fname, **kwargs)
        fname = pic_info.run_dir + "data/p" + species + "-zx.gda"
        x, z, pzx = read_2d_fields(pic_info, fname, **kwargs)
        fname = pic_info.run_dir + "data/p" + species + "-zy.gda"
        x, z, pzy = read_2d_fields(pic_info, fname, **kwargs)
        fname = pic_info.run_dir + "data/p" + species + "-zz.gda"
        x, z, pzz = read_2d_fields(pic_info, fname, **kwargs)
        fname = pic_info.run_dir + "data/bx.gda"
        x, z, bx = read_2d_fields(pic_info, fname, **kwargs)
        fname = pic_info.run_dir + "data/by.gda"
        x, z, by = read_2d_fields(pic_info, fname, **kwargs)
        fname = pic_info.run_dir + "data/bz.gda"
        x, z, bz = read_2d_fields(pic_info, fname, **kwargs)
        ib2 = 1.0 / (bx*bx + by*by + bz*bz)
        ppara = (pxx * bx**2 + pyy * by**2 + pzz * bz**2 +
                 (pxy + pyx) * bx * by +
                 (pxz + pzx) * bx * bz +
                 (pyz + pzy) * by * bz)
        ppara = ppara * ib2
        l1 = pxx + pyy + pzz
        l2 = (pxx * pyy + pxx * pzz + pyy * pzz -
              pxy * pyx - pxz * pzx - pyz * pzy)
        q = 1 - 4 * l2 / ((l1 - ppara) * (l1 + 3 * ppara))
        hist, bin_edges = np.histogram(q, bins=qbins)
        qdist.append(hist / qnorm[imime])
        # fname = pic_info.run_dir + "data/Ay.gda"
        # x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
        ax = fig.add_axes(rect)
        ax.tick_params(bottom=True, top=False, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in', top=True)
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')
        sizes = [x[0], x[-1], z[0], z[-1]]
        p1 = ax.imshow(q, vmin=dmins[imime], vmax=dmaxs[imime],
                       extent=sizes, cmap=plt.cm.viridis, aspect='auto',
                       origin='lower', interpolation='bicubic')
        # ax.contour(x, z, Ay, colors='k', linewidths=0.5)
        ax.tick_params(labelsize=8)
        if imime == len(mimes) - 1:
            ax.set_xlabel(r'$x/d_i$', fontsize=10)
        else:
            ax.tick_params(axis='x', labelbottom=False)
        ax.set_ylabel(r'$z/d_i$', fontsize=10)

        if imime == 1:
            rect_cbar = np.copy(rect)
            rect_cbar[0] += rect[2] + hgap
            rect_cbar[1] = rect[1] - vgap - rect[3] * 0.5
            rect_cbar[2] = 0.02
            rect_cbar[3] = rect[3] * 2 + vgap * 2
            cbar_ax = fig.add_axes(rect_cbar)
            cbar_ax.tick_params(bottom=True )
            cbar_ax.tick_params(axis='y', which='minor', direction='in')
            cbar_ax.tick_params(axis='y', which='major', direction='in')
            cbar = fig.colorbar(p1, cax=cbar_ax)
            cbar.set_ticks(np.linspace(lmins[imime], lmaxs[imime], num=6))
            cbar.ax.xaxis.set_label_position('top')
            cbar.ax.set_xlabel(r'$Q$', fontsize=10)
            cbar.ax.tick_params(labelsize=8)
        text1 = r'$m_i/m_e = ' + str(mime) + '$'
        ax.text(0.03, 0.85, text1, color='w', fontsize=10,
                bbox=dict(facecolor='none', alpha=1.0,
                          edgecolor='none', pad=10.0),
                horizontalalignment='left',
                verticalalignment='center',
                transform=ax.transAxes)
        rect[1] -= rect0[3] + vgap

    fdir = '../img/img_high_mime/Q/'
    mkdir_p(fdir)
    fname = fdir + 'q_' + species + '_' + bg_str + '_' + str(tframe) + '.pdf'
    fig.savefig(fname, dpi=300)

    fdir = '../data/data_high_mime/Q/'
    mkdir_p(fdir)
    fname = fdir + 'q_' + species + '_' + bg_str + '_' + str(tframe) + '.dat'
    qdist = np.asarray(qdist)
    qdist.tofile(fname)
    # fig = plt.figure(figsize=[3.5, 2.4])
    # rect = [0.15, 0.15, 0.8, 0.8]
    # ax = fig.add_axes(rect)
    # ax.loglog(qbins_mid, qdist[0]/qbins_diff, color=COLORS[0])
    # ax.loglog(qbins_mid, qdist[1]/qbins_diff, color=COLORS[1])
    # ax.loglog(qbins_mid, qdist[2]/qbins_diff, color=COLORS[2])

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_stacked_agyq(plot_config, show_plot=True):
    """Plot stacked Q parameters defined by M. Swisdak
    """
    tframe = plot_config["tframe"]
    tstart = plot_config["tstart"]
    tend = plot_config["tend"]
    species = plot_config["species"]
    bg = plot_config["bg"]
    mimes = np.asarray([25, 100, 400])
    bg_str = str(int(bg * 10)).zfill(2)
    nbins = 200
    qbins = np.logspace(-2, 0, nbins + 1)
    qbins_mid = (qbins[:-1] + qbins[1:]) * 0.5
    qbins_diff = np.diff(qbins)
    fdir = '../data/data_high_mime/Q/'
    qdist = []
    for tframe in range(tstart, tend + 1):
        fname = (fdir + 'q_' + species + '_' + bg_str +
                 '_' + str(tframe) + '.dat')
        fdata = np.fromfile(fname)
        fdata = fdata.reshape((-1, nbins)) / qbins_diff
        qdist.append(fdata)

    qdist = np.asarray(qdist)
    tframes = np.linspace(tstart, tend + 1, tend - tstart + 1)
    tframes_new = np.linspace(tstart, tend + 1, (tend - tstart)*10 + 1)
    qbins_new = np.logspace(-2, 0, nbins*10 + 1)
    qbins_mid_new = (qbins_new[:-1] + qbins_new[1:]) * 0.5

    fig = plt.figure(figsize=[3.5, 2.4])
    rect = [0.13, 0.15, 0.74, 0.8]
    hgap, vgap = 0.02, 0.02
    vmin, vmax = 1E-1, 1E1
    fdata = div0(qdist[:, 2, :], qdist[:, 0, :]).T
    fdata[fdata == 0.0] = 1.0
    f = interp2d(tframes, qbins_mid, fdata, kind='cubic')
    fdata_new = f(tframes_new, qbins_mid_new)
    Xn, Yn = np.meshgrid(tframes_new, qbins_mid_new)
    ax = fig.add_axes(rect)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top=True)
    ax.tick_params(axis='x', which='major', direction='in', top=True)
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_yscale('log')
    img = ax.pcolormesh(Xn, Yn, fdata_new,
                        cmap=plt.cm.seismic,
                        norm = LogNorm(vmin=vmin, vmax=vmax))
    ax.tick_params(labelsize=8)
    ax.set_xlabel(r'$t\Omega_{ci}$', fontsize=10)
    # rect_cbar = np.copy(rect)
    # rect_cbar[0] += rect[2] + hgap
    # rect_cbar[2] = 0.02
    # cbar_ax = fig.add_axes(rect_cbar)
    # cbar_ax.tick_params(right=True)
    # cbar = fig.colorbar(img, cax=cbar_ax, orientation='vertical',
    #                     extend='both')
    # cbar.set_ticks(np.linspace(vmin, vmax, 5))
    # cbar.ax.tick_params(labelsize=8)
    plt.show()


def fluid_energization(mime, bg, species, show_plot=True):
    """Plot fluid energization

    Args:
        mime: ion-to-electron mass ratio
        bg: guide-field strength
        species: particle species
    """
    bg_str = str(int(bg * 10)).zfill(2)
    run_name = "mime" + str(mime) + "_beta002_bg" + bg_str
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    tfields = pic_info.tfields
    tenergy = pic_info.tenergy
    fname = "../data/fluid_energization/" + run_name + "/"
    fname += "emf_ptensor_" + species + '.gda'
    fluid_ene = np.fromfile(fname, dtype=np.float32)
    nvar = int(fluid_ene[0])
    nframes = int(fluid_ene[1])
    curv_drift_dote = fluid_ene[2:nframes+2]
    bulk_curv_dote = fluid_ene[nframes+2:2*nframes+2]
    grad_drift_dote = fluid_ene[2*nframes+2:3*nframes+2]
    magnetization_dote = fluid_ene[3*nframes+2:4*nframes+2]
    comp_ene = fluid_ene[4*nframes+2:5*nframes+2]
    shear_ene = fluid_ene[5*nframes+2:6*nframes+2]
    ptensor_ene = fluid_ene[6*nframes+2:7*nframes+2]
    pgyro_ene = fluid_ene[7*nframes+2:8*nframes+2]

    fname = "../data/fluid_energization/" + run_name + "/"
    fname += "para_perp_acc_" + species + '.gda'
    fluid_ene = np.fromfile(fname, dtype=np.float32)
    nvar = int(fluid_ene[0])
    nframes = int(fluid_ene[1])
    acc_drift_dote_t = fluid_ene[2:nframes+2]
    acc_drift_dote_s = fluid_ene[nframes+2:2*nframes+2]
    acc_drift_dote = acc_drift_dote_t + acc_drift_dote_s
    epara_ene = fluid_ene[2*nframes+2:3*nframes+2]
    eperp_ene = fluid_ene[3*nframes+2:4*nframes+2]
    acc_drift_dote[-1] = acc_drift_dote[-2]

    jperp_dote = curv_drift_dote + grad_drift_dote + magnetization_dote
    jagy_dote = ptensor_ene - jperp_dote
    if species == 'e':
        dkene = pic_info.dkene_e
    else:
        dkene = pic_info.dkene_i

    if species == 'e':
        if bg_str == "00":
            ylim = [-1.3, 3.3]
        elif bg_str == "02":
            ylim = [-1, 3]
        elif bg_str == "04":
            ylim = [-0.5, 1.7]
        else:
            ylim = [-0.5, 1.7]
    else:
        if bg_str == "00":
            ylim = [-5.0, 10.0]
        elif bg_str == "02":
            ylim = [-2, 6.0]
        elif bg_str == "04":
            ylim = [-2.5, 5.0]
        else:
            ylim = [-2, 4.0]

    fig = plt.figure(figsize=[7, 5])
    ax = fig.add_axes([0.15, 0.15, 0.8, 0.65])
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    ax.set_prop_cycle('color', COLORS)
    label1 = (r'$\boldsymbol{j}_{' + species + '\parallel}' +
              r'\cdot\boldsymbol{E}_\parallel$')
    ax.plot(tfields, epara_ene, linewidth=2, label=label1)
    label2 = (r'$\boldsymbol{j}_{' + species + '\perp}' +
              r'\cdot\boldsymbol{E}_\perp$')
    ax.plot(tfields, eperp_ene, linewidth=2, label=label2)
    label3 = r'$(\nabla\cdot\tensorsym{P}_' + species + r')\cdot\boldsymbol{v}_E$'
    ax.plot(tfields, ptensor_ene, linewidth=2, label=label3)
    label4 = (r'$\boldsymbol{j}_{' + species + r'-\text{agy}}' +
              r'\cdot\boldsymbol{E}_\perp$')
    # ax.plot(tfields, jagy_dote, linewidth=2, label=label4)
    # label5 = (r'$\boldsymbol{j}_{' + species + '\parallel}' +
    #           r'\cdot\boldsymbol{E}_\parallel + ' +
    #           r'\boldsymbol{j}_{' + species + '\perp}' +
    #           r'\cdot\boldsymbol{E}_\perp$')
    # ax.plot(tfields, epara_ene + eperp_ene, linewidth=2, label=label5)
    label6 = r'$dK_' + species + '/dt$'
    ax.plot(tenergy, dkene, linewidth=2, label=label6)
    ax.legend(loc='upper center', prop={'size': 16}, ncol=2,
              bbox_to_anchor=(0.5, 1.28),
              shadow=False, fancybox=False, frameon=False)
    ax.set_xlim([0, np.max(tfields)])
    ax.set_ylim(ylim)
    ax.tick_params(labelsize=16)
    ax.set_xlabel(r'$t\Omega_{ci}$', fontdict=FONT, fontsize=20)
    ax.set_ylabel('Energization', fontdict=FONT, fontsize=20)

    fdir = '../img/img_high_mime/fluid_energization/'
    mkdir_p(fdir)
    fname = fdir + 'fluid_ene_' + run_name + '_' + species + '.pdf'
    fig.savefig(fname)

    fig = plt.figure(figsize=[7, 5])
    ax = fig.add_axes([0.15, 0.15, 0.8, 0.65])
    COLORS = palettable.colorbrewer.qualitative.Set1_9.mpl_colors
    ax.set_prop_cycle('color', COLORS)
    ax.plot(tfields, curv_drift_dote, linewidth=2, label='Curvature')
    # ax.plot(tfields, bulk_curv_dote, linewidth=2, label='Bulk Curvature')
    ax.plot(tfields, grad_drift_dote, linewidth=2, label='Gradient')
    ax.plot(tfields, magnetization_dote, linewidth=2, label='Magnetization')
    ax.plot(tfields, acc_drift_dote, linewidth=2, label='Polarization')
    label2 = (r'$\boldsymbol{j}_{' + species + '\perp}' +
              r'\cdot\boldsymbol{E}_\perp$')
    ax.plot(tfields, eperp_ene, linewidth=2, label=label2)
    jdote_sum = (curv_drift_dote + grad_drift_dote +
                 magnetization_dote + jagy_dote + acc_drift_dote)
    # ax.plot(tfields, jdote_sum, linewidth=2)
    ax.legend(loc='upper center', prop={'size': 16}, ncol=3,
              bbox_to_anchor=(0.5, 1.28),
              shadow=False, fancybox=False, frameon=False)
    ax.set_xlim([0, np.max(tfields)])
    ax.set_ylim(ylim)
    ax.tick_params(labelsize=16)
    ax.set_xlabel(r'$t\Omega_{ci}$', fontdict=FONT, fontsize=20)
    ax.set_ylabel('Energization', fontdict=FONT, fontsize=20)

    fdir = '../img/img_high_mime/fluid_energization/'
    mkdir_p(fdir)
    fname = fdir + 'fluid_drift_' + run_name + '_' + species + '.pdf'
    fig.savefig(fname)

    fig = plt.figure(figsize=[7, 5])
    ax = fig.add_axes([0.15, 0.15, 0.8, 0.65])
    COLORS = palettable.colorbrewer.qualitative.Set1_9.mpl_colors
    ax.set_prop_cycle('color', COLORS)
    ax.plot(tfields, comp_ene, linewidth=2, label='Compression')
    ax.plot(tfields, shear_ene, linewidth=2, label='Shear')
    label2 = (r'$\boldsymbol{j}_{' + species + '\perp}' +
              r'\cdot\boldsymbol{E}_\perp -' + 'n_' + species +
              'm_' + species + r'(d\boldsymbol{u}_' + species +
              r'/dt)\cdot\boldsymbol{v}_E$')
    ax.plot(tfields, eperp_ene - acc_drift_dote, linewidth=2, label=label2)
    label4 = (r'$\boldsymbol{j}_{' + species + r'-\text{agy}}' +
              r'\cdot\boldsymbol{E}_\perp$')
    ax.plot(tfields, jagy_dote, linewidth=2, label=label4)
    # jdote_sum = comp_ene + shear_ene + jagy_dote
    # ax.plot(tfields, jdote_sum, linewidth=2)
    ax.legend(loc='upper center', prop={'size': 16}, ncol=2,
              bbox_to_anchor=(0.5, 1.28),
              shadow=False, fancybox=False, frameon=False)
    ax.set_xlim([0, np.max(tfields)])
    ax.set_ylim(ylim)
    ax.tick_params(labelsize=16)
    ax.set_xlabel(r'$t\Omega_{ci}$', fontdict=FONT, fontsize=20)
    ax.set_ylabel('Energization', fontdict=FONT, fontsize=20)

    fdir = '../img/img_high_mime/fluid_energization/'
    mkdir_p(fdir)
    fname = fdir + 'fluid_comp_shear_' + run_name + '_' + species + '.pdf'
    fig.savefig(fname)

    # fig = plt.figure(figsize=[7, 5])
    # ax = fig.add_axes([0.15, 0.15, 0.8, 0.65])
    # COLORS = palettable.colorbrewer.qualitative.Set1_9.mpl_colors
    # ax.set_prop_cycle('color', COLORS)
    # # ax.plot(tfields, comp_ene, linewidth=2, label='Compression')
    # # ax.plot(tfields, shear_ene, linewidth=2, label='Shear')
    # ax.plot(tfields, comp_ene + shear_ene,
    #         linewidth=2, label='Compression + shear')
    # jdote_drifts = curv_drift_dote + grad_drift_dote + magnetization_dote
    # ax.plot(tfields, jdote_drifts, linewidth=2, label='Drifts')
    # # ax.plot(tfields, ptensor_ene, linewidth=2, label='Ptensor')
    # # ax.plot(tfields, pgyro_ene, linewidth=2, label='Pgyro')
    # ax.legend(loc='upper center', prop={'size': 16}, ncol=2,
    #           bbox_to_anchor=(0.5, 1.28),
    #           shadow=False, fancybox=False, frameon=False)
    # ax.set_xlim([0, np.max(tfields)])
    # ax.tick_params(labelsize=16)
    # ax.set_xlabel(r'$t\Omega_{ci}$', fontdict=FONT, fontsize=20)
    # ax.set_ylabel('Energization', fontdict=FONT, fontsize=20)

    if show_plot:
        plt.show()
    else:
        plt.close('all')


def find_nearest(array, value):
    """Find nearest value in an array
    """
    idx = (np.abs(array-value)).argmin()
    return (idx, array[idx])


def fluid_energization_mime(bg, species, show_plot=True):
    """Plot fluid energization for different mime in the same figure

    Args:
        bg: guide-field strength
        species: particle species
    """
    COLORS = palettable.colorbrewer.qualitative.Set1_9.mpl_colors
    bg_str = str(int(bg * 10)).zfill(2)
    mimes = [25, 100, 400]
    fig1 = plt.figure(figsize=[3.5, 6])
    box1 = [0.15, 0.66, 0.8, 0.27]
    axs1 = []
    fig2 = plt.figure(figsize=[3.5, 6])
    box2 = [0.15, 0.66, 0.8, 0.27]
    axs2 = []
    fig3 = plt.figure(figsize=[3.5, 6])
    box3 = [0.15, 0.66, 0.8, 0.27]
    axs3 = []
    tshift = 12
    if species == 'e':
        if bg_str == "00":
            ylim1 = [-1.0, 3.0]
            ylim2 = [-1.0, 3.0]
            ylim3 = [-1.0, 3.0]
        elif bg_str == "02":
            ylim1 = [-1.0, 2.5]
            ylim2 = [-1.0, 2.5]
            ylim3 = [-1.0, 2.5]
        elif bg_str == "04":
            ylim1 = [-0.5, 1.7]
            ylim2 = [-0.5, 1.7]
            ylim3 = [-0.5, 1.7]
        else:
            ylim1 = [-0.2, 1.2]
            ylim2 = [-0.2, 0.7]
            ylim3 = [-0.2, 0.6]
    else:
        if bg_str == "00":
            ylim1 = [-2.0, 6.0]
            ylim2 = [-5.0, 10.0]
            ylim3 = [-2.0, 5.0]
        elif bg_str == "02":
            ylim1 = [-1, 6.0]
            ylim2 = [-2.5, 5.5]
            ylim3 = [-1, 6.0]
        elif bg_str == "04":
            ylim1 = [-1.5, 5.5]
            ylim2 = [-3.0, 4.5]
            ylim3 = [-0.5, 4.0]
        else:
            ylim1 = [-1, 3.0]
            ylim2 = [-1, 3.0]
            ylim3 = [-0.5, 1.2]

    for mime in mimes:
        run_name = "mime" + str(mime) + "_beta002_bg" + bg_str
        picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        tfields = pic_info.tfields
        tenergy = pic_info.tenergy
        if mime != 400:
            tfields -= tshift
            tenergy -= tshift
        fname = "../data/fluid_energization/" + run_name + "/"
        fname += "emf_ptensor_" + species + '.gda'
        fluid_ene = np.fromfile(fname, dtype=np.float32)
        nvar = int(fluid_ene[0])
        nframes = int(fluid_ene[1])
        curv_drift_dote = fluid_ene[2:nframes+2]
        bulk_curv_dote = fluid_ene[nframes+2:2*nframes+2]
        grad_drift_dote = fluid_ene[2*nframes+2:3*nframes+2]
        magnetization_dote = fluid_ene[3*nframes+2:4*nframes+2]
        comp_ene = fluid_ene[4*nframes+2:5*nframes+2]
        shear_ene = fluid_ene[5*nframes+2:6*nframes+2]
        ptensor_ene = fluid_ene[6*nframes+2:7*nframes+2]
        pgyro_ene = fluid_ene[7*nframes+2:8*nframes+2]

        fname = "../data/fluid_energization/" + run_name + "/"
        fname += "para_perp_acc_" + species + '.gda'
        fluid_ene = np.fromfile(fname, dtype=np.float32)
        nvar = int(fluid_ene[0])
        nframes = int(fluid_ene[1])
        acc_drift_dote_t = fluid_ene[2:nframes+2]
        acc_drift_dote_s = fluid_ene[nframes+2:2*nframes+2]
        acc_drift_dote = acc_drift_dote_t + acc_drift_dote_s
        epara_ene = fluid_ene[2*nframes+2:3*nframes+2]
        eperp_ene = fluid_ene[3*nframes+2:4*nframes+2]
        acc_drift_dote[-1] = acc_drift_dote[-2]

        jperp_dote = curv_drift_dote + grad_drift_dote + magnetization_dote
        jagy_dote = ptensor_ene - pgyro_ene
        if species == 'e':
            dkene = pic_info.dkene_e
            kene = pic_info.kene_e
        else:
            dkene = pic_info.dkene_i
            kene = pic_info.kene_i
        ene_mag = pic_info.ene_magnetic
        t100_index, t100 = find_nearest(tenergy, 100)
        print("Initial magnetic energy: %f" % ene_mag[0])
        print("Particle energy at 100\Omega_{ci} / initial magnetic energy: %f" %
              (kene[t100_index] / ene_mag[0]))

        ax = fig1.add_axes(box1)
        axs1.append(ax)
        COLORS = palettable.tableau.Tableau_10.mpl_colors
        ax.set_prop_cycle('color', COLORS)
        label1 = (r'$\boldsymbol{j}_{' + species + '\parallel}' +
                  r'\cdot\boldsymbol{E}_\parallel$')
        ax.plot(tfields, epara_ene, linewidth=1, label=label1)
        label2 = (r'$\boldsymbol{j}_{' + species + '\perp}' +
                  r'\cdot\boldsymbol{E}_\perp$')
        ax.plot(tfields, eperp_ene, linewidth=1, label=label2)
        label3 = r'$(\nabla\cdot\tensorsym{P}_' + species + r')\cdot\boldsymbol{v}_E$'
        ax.plot(tfields, ptensor_ene, linewidth=1, label=label3)
        label4 = (r'$\boldsymbol{j}_{' + species + r'-\text{agy}}' +
                  r'\cdot\boldsymbol{E}_\perp$')
        # ax.plot(tfields, jagy_dote, linewidth=1, label=label4)
        # label5 = (r'$\boldsymbol{j}_{' + species + '\parallel}' +
        #           r'\cdot\boldsymbol{E}_\parallel + ' +
        #           r'\boldsymbol{j}_{' + species + '\perp}' +
        #           r'\cdot\boldsymbol{E}_\perp$')
        # ax.plot(tfields, epara_ene + eperp_ene, linewidth=1, label=label5)
        label6 = r'$dK_' + species + '/dt$'
        ax.plot(tenergy, dkene, linewidth=1, label=label6)
        ax.plot([0, 100], [0, 0], linestyle='--', color='k')
        ax.set_xlim([0, 100])
        ax.set_ylim(ylim1)
        ax.tick_params(labelsize=8)
        if mime == 400:
            ax.set_xlabel(r'$t\Omega_{ci}$', fontdict=FONT, fontsize=10)
        else:
            xticks = ax.get_xticks()
            xticks_labels = [str(int(x + tshift)) for x in xticks]
            ax.set_xticklabels(xticks_labels)
        ax.set_ylabel('Energization', fontdict=FONT, fontsize=10)
        text1 = r'$m_i/m_e=' + str(mime) + '$'
        ax.text(0.02, 0.9, text1, color='k', fontsize=10,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in', top=True)
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')

        ax = fig2.add_axes(box1)
        axs2.append(ax)
        ax.set_prop_cycle('color', COLORS)
        ax.plot(tfields, curv_drift_dote, linewidth=1, label='Curvature')
        # ax.plot(tfields, bulk_curv_dote, linewidth=1, label='Bulk Curvature')
        ax.plot(tfields, grad_drift_dote, linewidth=1, label='Gradient')
        ax.plot(tfields, magnetization_dote, linewidth=1, label='Magnetization')
        ax.plot(tfields, acc_drift_dote, linewidth=1, label='Inertial')
        label2 = (r'$\boldsymbol{j}_{' + species + '\perp}' +
                  r'\cdot\boldsymbol{E}_\perp$')
        ax.plot(tfields, eperp_ene, linewidth=1, label=label2)
        jdote_sum = (curv_drift_dote + grad_drift_dote +
                     magnetization_dote + jagy_dote + acc_drift_dote)
        # ax.plot(tfields, jdote_sum, linewidth=1)
        ax.plot([0, 100], [0, 0], linestyle='--', color='k')
        ax.set_xlim([0, 100])
        ax.set_ylim(ylim2)
        ax.tick_params(labelsize=8)
        if mime == 400:
            ax.set_xlabel(r'$t\Omega_{ci}$', fontdict=FONT, fontsize=10)
        else:
            xticks = ax.get_xticks()
            xticks_labels = [str(int(x + tshift)) for x in xticks]
            ax.set_xticklabels(xticks_labels)
        ax.set_ylabel('Energization', fontdict=FONT, fontsize=10)
        text1 = r'$m_i/m_e=' + str(mime) + '$'
        ax.text(0.02, 0.9, text1, color='k', fontsize=10,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in', top=True)
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')

        ax = fig3.add_axes(box1)
        axs3.append(ax)
        ax.set_prop_cycle('color', COLORS)
        ax.plot(tfields, comp_ene, linewidth=1, label='Compression')
        ax.plot(tfields, shear_ene, linewidth=1, label='Shear')
        # label2 = (r'$\boldsymbol{j}_{' + species + '\perp}' +
        #           r'\cdot\boldsymbol{E}_\perp -' + 'n_' + species +
        #           'm_' + species + r'(d\boldsymbol{u}_' + species +
        #           r'/dt)\cdot\boldsymbol{v}_E$')
        label2 = (r'$\boldsymbol{j}_{' + species + '\perp}' +
                  r'\cdot\boldsymbol{E}_\perp -$' + 'inertial')
        ax.plot(tfields, eperp_ene - acc_drift_dote, linewidth=1, label=label2)
        label4 = (r'$\boldsymbol{j}_{' + species + r'-\text{agy}}' +
                  r'\cdot\boldsymbol{E}_\perp$')
        ax.plot(tfields, jagy_dote, linewidth=1, label=label4)
        # jdote_sum = comp_ene + shear_ene + jagy_dote
        # ax.plot(tfields, jdote_sum, linewidth=1)
        ax.plot([0, 100], [0, 0], linestyle='--', color='k')
        ax.set_xlim([0, 100])
        ax.set_ylim(ylim3)
        ax.tick_params(labelsize=8)
        if mime == 400:
            ax.set_xlabel(r'$t\Omega_{ci}$', fontdict=FONT, fontsize=10)
        else:
            xticks = ax.get_xticks()
            xticks_labels = [str(int(x + tshift)) for x in xticks]
            ax.set_xticklabels(xticks_labels)
        ax.set_ylabel('Energization', fontdict=FONT, fontsize=10)
        text1 = r'$m_i/m_e=' + str(mime) + '$'
        ax.text(0.02, 0.9, text1, color='k', fontsize=10,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in', top=True)
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')

        box1[1] -= box1[3] + 0.03

    axs1[0].legend(loc='upper center', prop={'size': 10}, ncol=2,
                   bbox_to_anchor=(0.5, 1.3),
                   shadow=False, fancybox=False, frameon=False,
                   columnspacing=0.1)
    axs2[0].legend(loc='upper center', prop={'size': 10}, ncol=3,
                   bbox_to_anchor=(0.5, 1.3),
                   shadow=False, fancybox=False, frameon=False,
                   columnspacing=0.2, handletextpad=0.1)
    axs3[0].legend(loc='upper center', prop={'size': 10}, ncol=2,
                   bbox_to_anchor=(0.5, 1.3),
                   shadow=False, fancybox=False, frameon=False,
                   columnspacing=0.1, handletextpad=0.1)
    fdir = '../img/img_high_mime/fluid_energization/'
    mkdir_p(fdir)
    fname = fdir + 'fluid_ene_bg' + bg_str + '_' + species + '.pdf'
    fig1.savefig(fname)

    fdir = '../img/img_high_mime/fluid_energization/'
    mkdir_p(fdir)
    fname = fdir + 'fluid_drift_bg' + bg_str + '_' + species + '.pdf'
    fig2.savefig(fname)

    fdir = '../img/img_high_mime/fluid_energization/'
    mkdir_p(fdir)
    fname = fdir + 'fluid_comp_shear_bg' + bg_str + '_' + species + '.pdf'
    fig3.savefig(fname)

    if show_plot:
        plt.show()
    else:
        plt.close('all')


def get_cumsum_jdote(jdote, dt):
    """
    Return the cumulative sum of the jdote along time
    """
    jdote[0] = 0  # Make sure that it starts from 0
    jdote_cum = np.cumsum(jdote) * dt
    return jdote_cum


def fluid_energization_fraction(species, const_va=False, high_bg=False, show_plot=True):
    """Plot fluid energization fraction

    Args:
        species: particle species
        const_va: whether the Alfven speed is constant for different mass ratio
        high_bg: whether to include runs with higher guide field
    """
    mimes = [25, 100, 400]
    if high_bg:
        bgs = [0.0, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
    else:
        bgs = [0.0, 0.2, 0.4, 0.8]
    jdotes_cum = np.zeros((22, len(mimes), len(bgs)))
    tshifts = shift_tframes(const_va)
    enorms = []
    for imime, mime in enumerate(mimes):
        run_name = 'mime' + str(mime) + '_beta002_bg00'
        if const_va and mime != 400:
            run_name += '_high'
        picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        mag_ene = pic_info.ene_magnetic[0]
        enorms.append(mag_ene)
    enorms = np.asarray(enorms)
    enorms = enorms.max() / enorms

    for imime, mime in enumerate(mimes):
        for ibg, bg in enumerate(bgs):
            bg_str = str(int(bg * 10)).zfill(2)
            run_name = 'mime' + str(mime) + '_beta002_' + 'bg' + bg_str
            if const_va and mime != 400:
                run_name += '_high'
            picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
            pic_info = read_data_from_json(picinfo_fname)
            tfields = pic_info.tfields
            tenergy = pic_info.tenergy
            dtf = pic_info.dt_fields * pic_info.dtwpe / pic_info.dtwci
            tenergy -= tshifts[str(mime)]
            tfields -= tshifts[str(mime)]
            fname = "../data/fluid_energization/" + run_name + "/"
            fname += "emf_ptensor_" + species + '.gda'
            fluid_ene = np.fromfile(fname, dtype=np.float32)
            nvar = int(fluid_ene[0])
            nframes = int(fluid_ene[1])
            curv_drift_dote = fluid_ene[2:nframes+2]
            bulk_curv_dote = fluid_ene[nframes+2:2*nframes+2]
            grad_drift_dote = fluid_ene[2*nframes+2:3*nframes+2]
            magnetization_dote = fluid_ene[3*nframes+2:4*nframes+2]
            comp_ene = fluid_ene[4*nframes+2:5*nframes+2]
            shear_ene = fluid_ene[5*nframes+2:6*nframes+2]
            ptensor_ene = fluid_ene[6*nframes+2:7*nframes+2]
            pgyro_ene = fluid_ene[7*nframes+2:8*nframes+2]
            para_drift_ene = fluid_ene[8*nframes+2:9*nframes+2]
            mu_ene = fluid_ene[9*nframes+2:10*nframes+2]

            curv_drift_dote_cum = get_cumsum_jdote(curv_drift_dote, dtf)
            bulk_curv_dote_cum = get_cumsum_jdote(bulk_curv_dote, dtf)
            grad_drift_dote_cum = get_cumsum_jdote(grad_drift_dote, dtf)
            magnetization_dote_cum = get_cumsum_jdote(magnetization_dote, dtf)
            comp_ene_cum = get_cumsum_jdote(comp_ene, dtf)
            shear_ene_cum = get_cumsum_jdote(shear_ene, dtf)
            ptensor_ene_cum = get_cumsum_jdote(ptensor_ene, dtf)
            pgyro_ene_cum = get_cumsum_jdote(pgyro_ene, dtf)
            para_drift_ene_cum = get_cumsum_jdote(para_drift_ene, dtf)
            mu_ene_cum = get_cumsum_jdote(mu_ene, dtf)

            curv_drift_dote_cum -= curv_drift_dote_cum[0]
            bulk_curv_dote_cum -= bulk_curv_dote_cum[0]
            grad_drift_dote_cum -= grad_drift_dote_cum[0]
            magnetization_dote_cum -= magnetization_dote_cum[0]
            comp_ene_cum -= comp_ene_cum[0]
            shear_ene_cum -= shear_ene_cum[0]
            ptensor_ene_cum -= ptensor_ene_cum[0]
            pgyro_ene_cum -= pgyro_ene_cum[0]
            para_drift_ene_cum -= para_drift_ene_cum[0]
            mu_ene_cum -= mu_ene_cum[0]

            fname = "../data/fluid_energization/" + run_name + "/"
            fname += "para_perp_acc_" + species + '.gda'
            fluid_ene = np.fromfile(fname, dtype=np.float32)
            nvar = int(fluid_ene[0])
            nframes = int(fluid_ene[1])
            acc_drift_perp_dote_t = fluid_ene[2:nframes+2]
            acc_drift_para_dote_t = fluid_ene[nframes+2:2*nframes+2]
            acc_drift_perp_dote_s = fluid_ene[2*nframes+2:3*nframes+2]
            acc_drift_para_dote_s = fluid_ene[3*nframes+2:4*nframes+2]
            acc_drift_dote_t = acc_drift_para_dote_t + acc_drift_perp_dote_t
            acc_drift_dote_s = acc_drift_para_dote_s + acc_drift_perp_dote_s
            acc_drift_dote = acc_drift_dote_t + acc_drift_dote_s
            epara_ene = fluid_ene[4*nframes+2:5*nframes+2]
            eperp_ene = fluid_ene[5*nframes+2:6*nframes+2]
            acc_drift_dote[-1] = acc_drift_dote[-2]

            acc_drift_para_dote_t_cum = get_cumsum_jdote(acc_drift_para_dote_t, dtf)
            acc_drift_perp_dote_t_cum = get_cumsum_jdote(acc_drift_perp_dote_t, dtf)
            acc_drift_para_dote_s_cum = get_cumsum_jdote(acc_drift_para_dote_s, dtf)
            acc_drift_perp_dote_s_cum = get_cumsum_jdote(acc_drift_perp_dote_s, dtf)
            acc_drift_dote_t_cum = get_cumsum_jdote(acc_drift_dote_t, dtf)
            acc_drift_dote_s_cum = get_cumsum_jdote(acc_drift_dote_s, dtf)
            acc_drift_dote_cum = get_cumsum_jdote(acc_drift_dote, dtf)
            epara_ene_cum = get_cumsum_jdote(epara_ene, dtf)
            eperp_ene_cum = get_cumsum_jdote(eperp_ene, dtf)

            acc_drift_para_dote_t_cum -= acc_drift_para_dote_t_cum[0]
            acc_drift_perp_dote_t_cum -= acc_drift_perp_dote_t_cum[0]
            acc_drift_para_dote_s_cum -= acc_drift_para_dote_s_cum[0]
            acc_drift_perp_dote_s_cum -= acc_drift_perp_dote_s_cum[0]
            acc_drift_dote_t_cum -= acc_drift_dote_t_cum[0]
            acc_drift_dote_s_cum -= acc_drift_dote_s_cum[0]
            acc_drift_dote_cum -= acc_drift_dote_cum[0]
            epara_ene_cum -= epara_ene_cum[0]
            eperp_ene_cum -= eperp_ene_cum[0]

            jperp_dote = curv_drift_dote + grad_drift_dote + magnetization_dote
            jagy_dote = ptensor_ene - pgyro_ene

            jperp_dote_cum = get_cumsum_jdote(jperp_dote, dtf)
            jagy_dote_cum = get_cumsum_jdote(jagy_dote, dtf)

            jperp_dote_cum -= jperp_dote_cum[0]
            jagy_dote_cum -= jagy_dote_cum[0]

            if species == 'e':
                dkene = pic_info.dkene_e
                kene = pic_info.kene_e
            else:
                dkene = pic_info.dkene_i
                kene = pic_info.kene_i
            ene_mag = pic_info.ene_magnetic
            tindex_e, te = find_nearest(tenergy, 100)
            tindex_f, tf = find_nearest(tfields, 100)
            jdotes_cum[0, imime, ibg] = curv_drift_dote_cum[tindex_f]
            jdotes_cum[1, imime, ibg] = bulk_curv_dote_cum[tindex_f]
            jdotes_cum[2, imime, ibg] = grad_drift_dote_cum[tindex_f]
            jdotes_cum[3, imime, ibg] = magnetization_dote_cum[tindex_f]
            jdotes_cum[4, imime, ibg] = comp_ene_cum[tindex_f]
            jdotes_cum[5, imime, ibg] = shear_ene_cum[tindex_f]
            jdotes_cum[6, imime, ibg] = ptensor_ene_cum[tindex_f]
            jdotes_cum[7, imime, ibg] = pgyro_ene_cum[tindex_f]
            jdotes_cum[8, imime, ibg] = para_drift_ene_cum[tindex_f]
            jdotes_cum[9, imime, ibg] = mu_ene_cum[tindex_f]
            jdotes_cum[10, imime, ibg] = acc_drift_para_dote_t_cum[tindex_f]
            jdotes_cum[11, imime, ibg] = acc_drift_perp_dote_t_cum[tindex_f]
            jdotes_cum[12, imime, ibg] = acc_drift_para_dote_s_cum[tindex_f]
            jdotes_cum[13, imime, ibg] = acc_drift_perp_dote_s_cum[tindex_f]
            jdotes_cum[14, imime, ibg] = acc_drift_dote_t_cum[tindex_f]
            jdotes_cum[15, imime, ibg] = acc_drift_dote_s_cum[tindex_f]
            jdotes_cum[16, imime, ibg] = acc_drift_dote_cum[tindex_f]
            jdotes_cum[17, imime, ibg] = epara_ene_cum[tindex_f]
            jdotes_cum[18, imime, ibg] = eperp_ene_cum[tindex_f]
            jdotes_cum[19, imime, ibg] = jperp_dote_cum[tindex_f]
            jdotes_cum[20, imime, ibg] = jagy_dote_cum[tindex_f]
            jdotes_cum[-1, imime, ibg] = kene[tindex_e] - kene[0]
            # jdotes_cum[:, imime, ibg] *= enorms[imime]

    jdotes_cum[:-1, :] /= jdotes_cum[-1, :]

    COLORS = palettable.tableau.Tableau_10.mpl_colors
    fig1 = plt.figure(figsize=[3.5, 5])
    box1 = [0.12, 0.77, 0.83, 0.16]
    vgap = 0.01
    ax = fig1.add_axes(box1)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top=True)
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    for imime in range(len(mimes)):
        ax.plot(bgs, jdotes_cum[17, imime, :],
                linewidth=1, marker='o', markersize=4, color=COLORS[imime])
        ax.plot(bgs, jdotes_cum[18, imime, :], linewidth=1, marker='x',
                markersize=4, color=COLORS[imime])
    if not high_bg:
        ax.text(0.03, 0.6, r'$m_i/m_e=25$', color=COLORS[0], fontsize=8,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='bottom',
                transform=ax.transAxes)
        ax.text(0.03, 0.45, r'$m_i/m_e=100$', color=COLORS[1], fontsize=8,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='bottom',
                transform=ax.transAxes)
        ax.text(0.03, 0.3, r'$m_i/m_e=400$', color=COLORS[2], fontsize=8,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='bottom',
                transform=ax.transAxes)
    label1 = (r'$\boldsymbol{j}_{' + species + '\parallel}' +
              r'\cdot\boldsymbol{E}_\parallel$')
    label2 = (r'$\boldsymbol{j}_{' + species + '\perp}' +
              r'\cdot\boldsymbol{E}_\perp$')
    if const_va:
        if high_bg:
            angle = 0 if species == 'e' else 0
            ypos = 0.6 if species == 'e' else 0.77
        else:
            angle = -10 if species == 'e' else 0
            ypos = 0.68 if species == 'e' else 0.67
    else:
        angle = -15 if species == 'e' else 0
        ypos = 0.6 if species == 'e' else 0.67
    ax.text(0.7, ypos, label2, color='k', fontsize=8, rotation=angle,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes)
    if const_va:
        if high_bg:
            angle = 0 if species == 'e' else 0
            ypos = 0.3 if species == 'e' else 0.10
        else:
            angle = 10 if species == 'e' else 0
            ypos = 0.05 if species == 'e' else 0.25
    else:
        angle = 15 if species == 'e' else 0
        ypos = 0.08 if species == 'e' else 0.25
    ax.text(0.7, ypos, label1, color='k', fontsize=8, rotation=angle,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes)
    if high_bg:
        ax.set_xlim([-0.1, 6.5])
    else:
        ax.set_xlim([-0.05, 0.9])
    ax.plot(ax.get_xlim(), [0, 0], linewidth=0.5, linestyle='--', color='k')
    ax.tick_params(axis='x', labelbottom=False)
    ax.tick_params(labelsize=8)
    sp = 'Electron' if species == 'e' else 'Ion'
    label1 = sp + ' energization terms'
    label1 += r'$/\Delta K_' + species + r'\text{ at }t\Omega_{ci}=100$'
    ax.set_title(label1, fontsize=10)

    box1[1] -= box1[3] + vgap
    ax = fig1.add_axes(box1)
    ax.set_prop_cycle('color', COLORS)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top=True)
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    for imime in range(len(mimes)):
        ax.plot(bgs, jdotes_cum[4, imime, :],
                linewidth=1, marker='x', markersize=4, color=COLORS[imime])
        ax.plot(bgs, jdotes_cum[5, imime, :], linewidth=1, marker='o',
                markersize=4, color=COLORS[imime])
    if high_bg:
        ax.set_xlim([-0.1, 6.5])
    else:
        ax.set_xlim([-0.05, 0.9])
    ax.plot(ax.get_xlim(), [0, 0], linewidth=0.5, linestyle='--', color='k')
    ax.tick_params(axis='x', labelbottom=False)
    ax.tick_params(labelsize=8)
    if const_va:
        xpos = 0.1 if high_bg else 0.6
        if high_bg:
            angle = -45 if species == 'e' else -10
            ypos = 0.2 if species == 'e' else 0.37
        else:
            angle = -11 if species == 'e' else -10
            ypos = 0.45 if species == 'e' else 0.37
    else:
        angle = -11 if species == 'e' else -8
        ypos = 0.45 if species == 'e' else 0.37
        xpos = 0.6
    ax.text(xpos, ypos, 'compression', color='k', fontsize=8, rotation=angle,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes)
    if const_va:
        xpos = 0.1 if high_bg else 0.65
        if high_bg:
            angle = 0 if species == 'e' else -7
            ypos = 0.05 if species == 'e' else 0.15
        else:
            angle = -7 if species == 'e' else -7
            ypos = 0.13 if species == 'e' else 0.15
    else:
        angle = -6 if species == 'e' else -7
        ypos = 0.15 if species == 'e' else 0.12
        xpos = 0.65
    ax.text(xpos, ypos, 'shear', color='k', fontsize=8, rotation=angle,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes)

    box1[1] -= box1[3] + vgap
    ax = fig1.add_axes(box1)
    ax.set_prop_cycle('color', COLORS)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top=True)
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    for imime in range(len(mimes)):
        ax.plot(bgs, jdotes_cum[0, imime, :],
                linewidth=1, marker='x', markersize=4, color=COLORS[imime])
        ax.plot(bgs, jdotes_cum[2, imime, :], linewidth=1, marker='o',
                markersize=4, color=COLORS[imime])
    if high_bg:
        ax.set_xlim([-0.1, 6.5])
    else:
        ax.set_xlim([-0.05, 0.9])
    ax.plot(ax.get_xlim(), [0, 0], linewidth=0.5, linestyle='--', color='k')
    ax.tick_params(axis='x', labelbottom=False)
    ax.tick_params(labelsize=8)
    if const_va:
        xpos = 0.1 if high_bg else 0.6
        if high_bg:
            angle = -30 if species == 'e' else -5
            ypos = 0.32 if species == 'e' else 0.35
        else:
            angle = -5 if species == 'e' else -5
            ypos = 0.52 if species == 'e' else 0.35
    else:
        angle = -10 if species == 'e' else -5
        ypos = 0.48 if species == 'e' else 0.36
        xpos = 0.6
    ax.text(xpos, ypos, 'curvature', color='k', fontsize=8, rotation=angle,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes)
    if const_va:
        xpos = 0.1 if high_bg else 0.6
        if high_bg:
            angle = 0 if species == 'e' else 0
            ypos = 0.07 if species == 'e' else 0.10
        else:
            angle = -3 if species == 'e' else 0
            ypos = 0.27 if species == 'e' else 0.10
    else:
        angle = -3 if species == 'e' else 0
        ypos = 0.26 if species == 'e' else 0.12
        xpos = 0.6
    ax.text(xpos, ypos, 'gradient', color='k', fontsize=8, rotation=angle,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes)
    # label1 = r'Energization$/\Delta K_' + species + r'\text{ at }t\Omega_{ci}=100$'
    # ax.text(-0.18, 0.5, label1, color='k', fontsize=10, rotation='vertical',
    #         bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
    #         horizontalalignment='left', verticalalignment='center',
    #         transform=ax.transAxes)

    box1[1] -= box1[3] + vgap
    ax = fig1.add_axes(box1)
    ax.set_prop_cycle('color', COLORS)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top=True)
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    for imime in range(len(mimes)):
        ax.plot(bgs, jdotes_cum[3, imime, :],
                linewidth=1, marker='o', markersize=4, color=COLORS[imime])
        ax.plot(bgs, jdotes_cum[16, imime, :], linewidth=1, marker='x',
                markersize=4, color=COLORS[imime])
        # ax.plot(bgs, jdotes_cum[3, imime, :],
        #         linewidth=1, marker='o', markersize=4, color=COLORS[imime])
        # ax.plot(bgs, jdotes_cum[9, imime, :],
        #         linewidth=1, marker='x', markersize=4, color=COLORS[imime])
        # ax.plot(bgs, jdotes_cum[10, imime, :] + jdotes_cum[12, imime, :],
        #         linewidth=1, marker='o', markersize=4, color=COLORS[imime])
        # ax.plot(bgs, jdotes_cum[11, imime, :] + jdotes_cum[13, imime, :],
        #         linewidth=1, marker='x', markersize=4, color=COLORS[imime])
    if high_bg:
        ax.set_xlim([-0.1, 6.5])
    else:
        ax.set_xlim([-0.05, 0.9])
    ax.plot(ax.get_xlim(), [0, 0], linewidth=0.5, linestyle='--', color='k')
    ax.tick_params(axis='x', labelbottom=False)
    ax.tick_params(labelsize=8)
    if const_va:
        xpos = 0.1 if high_bg else 0.6
        if high_bg:
            angle = 0 if species == 'e' else 14
            ypos = 0.73 if species == 'e' else 0.70
        else:
            angle = 0 if species == 'e' else 14
            ypos = 0.73 if species == 'e' else 0.70
    else:
        angle = 0 if species == 'e' else 12
        ypos = 0.74 if species == 'e' else 0.71
        xpos = 0.6
    ax.text(xpos, ypos, 'inertial', color='k', fontsize=8, rotation=angle,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes)
    if const_va:
        xpos = 0.1 if high_bg else 0.6
        if high_bg:
            angle = 10 if species == 'e' else 0
            ypos = 0.18 if species == 'e' else 0.15
        else:
            angle = 3 if species == 'e' else 0
            ypos = 0.18 if species == 'e' else 0.15
    else:
        angle = 6 if species == 'e' else 0
        ypos = 0.32 if species == 'e' else 0.16
        xpos = 0.6
    ax.text(xpos, ypos, 'magnetization', color='k', fontsize=8, rotation=angle,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes)

    box1[1] -= box1[3] + vgap
    ax = fig1.add_axes(box1)
    ax.tick_params(bottom=True, top=True, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top=True)
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.set_prop_cycle('color', COLORS)
    for imime in range(len(mimes)):
        ax.plot(bgs, jdotes_cum[20, imime, :],
                linewidth=1, marker='o', markersize=4, color=COLORS[imime])
        ax.plot(bgs, jdotes_cum[7, imime, :],
                linewidth=1, marker='x', markersize=4, color=COLORS[imime])
    if high_bg:
        ax.set_xlim([-0.1, 6.5])
    else:
        ax.set_xlim([-0.05, 0.9])
    ax.plot(ax.get_xlim(), [0, 0], linewidth=0.5, linestyle='--', color='k')
    ax.tick_params(labelsize=8)
    ax.set_xlabel(r'$B_g/B_0$', fontdict=FONT, fontsize=10)
    if const_va:
        if high_bg:
            angle = -45 if species == 'e' else -17
            xpos = 0.10 if species == 'e' else 0.1
            ypos = 0.28 if species == 'e' else 0.22
        else:
            angle = -10 if species == 'e' else -17
            xpos = 0.65 if species == 'e' else 0.2
            ypos = 0.58 if species == 'e' else 0.62
    else:
        angle = -16 if species == 'e' else -18
        xpos = 0.65 if species == 'e' else 0.2
        ypos = 0.53 if species == 'e' else 0.6
    ax.text(xpos, ypos, 'gyrotropic', color='k', fontsize=8, rotation=angle,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes)
    if const_va:
        if high_bg:
            angle = -10 if species == 'e' else 0
            xpos = 0.02 if species == 'e' else 0.05
            ypos = 0.11 if species == 'e' else 0.05
        else:
            angle = -3 if species == 'e' else 12
            xpos = 0.65 if species == 'e' else 0.05
            ypos = 0.15 if species == 'e' else 0.20
    else:
        angle = -3 if species == 'e' else 12
        xpos = 0.65 if species == 'e' else 0.05
        ypos = 0.16 if species == 'e' else 0.20
    ax.text(xpos, ypos, 'agyrotropic', color='k', fontsize=8, rotation=angle,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes)
    fdir = '../img/img_high_mime/'
    mkdir_p(fdir)
    if const_va:
        if high_bg:
            fname = fdir + 'fluid_ene_high_bg_' + species + '_high.pdf'
        else:
            fname = fdir + 'fluid_ene_' + species + '_high.pdf'
    else:
        if high_bg:
            fname = fdir + 'fluid_ene_high_bg_' + species + '.pdf'
        else:
            fname = fdir + 'fluid_ene_' + species + '.pdf'
    fig1.savefig(fname)
    plt.show()


def compare_fluid_energization(species, bg, const_va=False, show_plot=True):
    """compare fluid energization for runs with different mass ratio

    Args:
        species: particle species
        bg: guide field strength
        const_va: whether the Alfven speed is constant for different mass ratio
    """
    mimes = [25, 100, 400]
    bg_str = str(int(bg * 10)).zfill(2)
    tshifts = shift_tframes(const_va)
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    fig1 = plt.figure(figsize=[3.5, 5])
    box1 = [0.12, 0.77, 0.83, 0.16]
    vgap = 0.01
    axs = []
    for i in range(5):
        ax = fig1.add_axes(box1)
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in', top=True)
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')
        ax.set_xlim([30, 100])
        if species == 'e':
            if bg_str == '00':
                ax.set_ylim([-0.2, 1.0])
            elif bg_str == '02':
                ax.set_ylim([-0.1, 0.6])
            elif bg_str == '04':
                ax.set_ylim([-0.1, 0.4])
            elif bg_str == '08':
                ax.set_ylim([-0.1, 0.22])
        else:
            if bg_str == '00':
                ax.set_ylim([-0.8, 2.2])
            elif bg_str == '02':
                ax.set_ylim([-0.1, 0.6])
            elif bg_str == '04':
                ax.set_ylim([-0.1, 0.4])
            elif bg_str == '08':
                ax.set_ylim([-0.1, 0.22])
        axs.append(ax)
        box1[1] -= box1[3] + vgap

    enorms = []
    for imime, mime in enumerate(mimes):
        run_name = 'mime' + str(mime) + '_beta002_' + 'bg' + bg_str
        if const_va and mime != 400:
            run_name += '_high'
        picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        mag_ene = pic_info.ene_magnetic[0]
        enorms.append(mag_ene * pic_info.wpe_wce)
    enorms = np.asarray(enorms)
    enorms = enorms.max() / enorms

    for imime, mime in enumerate(mimes):
        run_name = 'mime' + str(mime) + '_beta002_' + 'bg' + bg_str
        if const_va and mime != 400:
            run_name += '_high'
        picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        tfields = pic_info.tfields
        tenergy = pic_info.tenergy
        dtf = pic_info.dt_fields * pic_info.dtwpe / pic_info.dtwci
        tenergy -= tshifts[str(mime)]
        tfields -= tshifts[str(mime)]
        fname = "../data/fluid_energization/" + run_name + "/"
        fname += "emf_ptensor_" + species + '.gda'
        fluid_ene = np.fromfile(fname, dtype=np.float32)
        nvar = int(fluid_ene[0])
        nframes = int(fluid_ene[1])
        curv_drift_dote = fluid_ene[2:nframes+2]
        bulk_curv_dote = fluid_ene[nframes+2:2*nframes+2]
        grad_drift_dote = fluid_ene[2*nframes+2:3*nframes+2]
        magnetization_dote = fluid_ene[3*nframes+2:4*nframes+2]
        comp_ene = fluid_ene[4*nframes+2:5*nframes+2]
        shear_ene = fluid_ene[5*nframes+2:6*nframes+2]
        ptensor_ene = fluid_ene[6*nframes+2:7*nframes+2]
        pgyro_ene = fluid_ene[7*nframes+2:8*nframes+2]
        para_drift_ene = fluid_ene[8*nframes+2:9*nframes+2]
        mu_ene = fluid_ene[9*nframes+2:10*nframes+2]

        fname = "../data/fluid_energization/" + run_name + "/"
        fname += "para_perp_acc_" + species + '.gda'
        fluid_ene = np.fromfile(fname, dtype=np.float32)
        nvar = int(fluid_ene[0])
        nframes = int(fluid_ene[1])
        acc_drift_perp_dote_t = fluid_ene[2:nframes+2]
        acc_drift_para_dote_t = fluid_ene[nframes+2:2*nframes+2]
        acc_drift_perp_dote_s = fluid_ene[2*nframes+2:3*nframes+2]
        acc_drift_para_dote_s = fluid_ene[3*nframes+2:4*nframes+2]
        acc_drift_dote_t = acc_drift_para_dote_t + acc_drift_perp_dote_t
        acc_drift_dote_s = acc_drift_para_dote_s + acc_drift_perp_dote_s
        acc_drift_dote = acc_drift_dote_t + acc_drift_dote_s
        epara_ene = fluid_ene[4*nframes+2:5*nframes+2]
        eperp_ene = fluid_ene[5*nframes+2:6*nframes+2]
        acc_drift_dote[-1] = acc_drift_dote[-2]

        jperp_dote = curv_drift_dote + grad_drift_dote + magnetization_dote
        jagy_dote = ptensor_ene - pgyro_ene

        if species == 'e':
            dkene = pic_info.dkene_e
            kene = pic_info.kene_e
        else:
            dkene = pic_info.dkene_i
            kene = pic_info.kene_i
        ene_mag = pic_info.ene_magnetic
        tindex_e, te = find_nearest(tenergy, 100)
        tindex_f, tf = find_nearest(tfields, 100)

        enorm = enorms[imime]

        ax = axs[0]
        epara_ene[0] = 0
        eperp_ene[0] = 0
        ax.plot(tfields, epara_ene*enorm, linewidth=1,
                linestyle='--', color=COLORS[imime])
        ax.plot(tfields, eperp_ene*enorm, linewidth=1, color=COLORS[imime])
        ax.plot(ax.get_xlim(), [0, 0], linewidth=0.5, linestyle='--', color='k')

        ax.tick_params(axis='x', labelbottom=False)
        ax.tick_params(labelsize=8)
        sp = 'Electron' if species == 'e' else 'Ion'
        label1 = sp + ' energization terms'
        ax.set_title(label1, fontsize=10)

        ax = axs[1]
        ax.plot(tfields, comp_ene*enorm, linewidth=1, color=COLORS[imime])
        ax.plot(tfields, shear_ene*enorm, linewidth=1,
                linestyle='--', color=COLORS[imime])
        ax.plot(ax.get_xlim(), [0, 0], linewidth=0.5, linestyle='--', color='k')
        ax.tick_params(axis='x', labelbottom=False)
        ax.tick_params(labelsize=8)

        ax = axs[2]
        ax.plot(tfields, curv_drift_dote*enorm, linewidth=1, color=COLORS[imime])
        ax.plot(tfields, grad_drift_dote*enorm, linewidth=1,
                linestyle='--', color=COLORS[imime])
        ax.plot(ax.get_xlim(), [0, 0], linewidth=0.5, linestyle='--', color='k')
        ax.tick_params(axis='x', labelbottom=False)
        ax.tick_params(labelsize=8)

        ax = axs[3]
        ax.plot(tfields, magnetization_dote*enorm, linewidth=1, color=COLORS[imime])
        ax.plot(tfields, acc_drift_dote*enorm, linewidth=1,
                linestyle='--', color=COLORS[imime])
        ax.plot(ax.get_xlim(), [0, 0], linewidth=0.5, linestyle='--', color='k')
        ax.tick_params(axis='x', labelbottom=False)
        ax.tick_params(labelsize=8)

        ax = axs[4]
        ax.plot(tfields, jagy_dote*enorm, linewidth=1,
                linestyle='--', color=COLORS[imime])
        ax.plot(tfields, pgyro_ene*enorm, linewidth=1, color=COLORS[imime])
        ax.plot(ax.get_xlim(), [0, 0], linewidth=0.5, linestyle='--', color='k')
        ax.set_xlabel(r'$t\Omega_{ci}$', fontdict=FONT, fontsize=10)
        ax.tick_params(labelsize=8)
    fdir = '../img/img_high_mime/'
    mkdir_p(fdir)
    if const_va:
        fname = fdir + 'fluid_ene_' + species + '_high.pdf'
    else:
        fname = fdir + 'fluid_ene_' + species + '.pdf'
    # fig1.savefig(fname)
    plt.show()


def bulk_internal_energy(bg, species, show_plot=True):
    """Plot bulk energy and internal energy

    Args:
        bg: guide-field strength
        species: particle species
    """
    COLORS = palettable.colorbrewer.qualitative.Set1_9.mpl_colors
    bg_str = str(int(bg * 10)).zfill(2)
    mimes = [25, 100, 400]
    fig1 = plt.figure(figsize=[7, 12])
    box1 = [0.15, 0.66, 0.8, 0.27]
    axs1 = []
    fig2 = plt.figure(figsize=[7, 12])
    box2 = [0.15, 0.66, 0.8, 0.27]
    axs2 = []
    tshift = 12
    if species == 'e':
        if bg_str == "00":
            ylim1 = [-0.5, 2.5]
            ylim2 = [0, 0.08]
        elif bg_str == "02":
            ylim1 = [-0.5, 2.5]
            ylim2 = [0, 0.08]
        elif bg_str == "04":
            ylim1 = [-0.5, 2.0]
            ylim2 = [0, 0.07]
        else:
            ylim1 = [-0.5, 1.2]
            ylim2 = [0, 0.06]
    else:
        if bg_str == "00":
            ylim1 = [-2.5, 5.0]
            ylim2 = [0, 0.13]
        elif bg_str == "02":
            ylim1 = [-3.0, 5.5]
            ylim2 = [0, 0.13]
        elif bg_str == "04":
            ylim1 = [-3.5, 4.5]
            ylim2 = [0, 0.11]
        else:
            ylim1 = [-1.0, 2.5]
            ylim2 = [0, 0.07]

    for mime in mimes:
        run_name = "mime" + str(mime) + "_beta002_bg" + bg_str
        picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        tfields = pic_info.tfields
        tenergy = pic_info.tenergy
        fname = "../data/fluid_energization/" + run_name + "/"
        fname += "para_perp_acc_" + species + '.gda'
        fluid_ene = np.fromfile(fname, dtype=np.float32)
        nframes = int(fluid_ene[1])
        acc_drift_dote_t = fluid_ene[2:nframes+2]
        acc_drift_dote_s = fluid_ene[nframes+2:2*nframes+2]
        acc_drift_dote = acc_drift_dote_t + acc_drift_dote_s
        acc_drift_dote[-1] = acc_drift_dote[-2]
        if species == 'e':
            dkene = pic_info.dkene_e
            kene = pic_info.kene_e
        else:
            dkene = pic_info.dkene_i
            kene = pic_info.kene_i
        if mime != 400:
            tfields -= tshift
            tenergy -= tshift
        enorm = pic_info.ene_bx[0]
        fname = "../data/bulk_internal_energy/" + run_name + "/"
        fname += "bulk_internal_energy_" + species + '.dat'
        fdata = np.fromfile(fname, dtype=np.float32)
        sz, = fdata.shape
        nframes = (sz//2)//4
        bene = fdata[:sz//2].reshape(nframes, -1)
        iene = fdata[sz//2:].reshape(nframes, -1)
        dt_fields = (tfields[1] - tfields[0]) * pic_info.dtwpe / pic_info.dtwci

        ax = fig1.add_axes(box1)
        axs1.append(ax)
        COLORS = palettable.tableau.Tableau_10.mpl_colors
        ax.set_prop_cycle('color', COLORS)
        label1 = r'$dK_{B' + species + '}/dt$'
        ax.plot(tfields, np.gradient(bene[:, -1]) / dt_fields, linewidth=2,
                label=label1)
        label2 = r'$dU_' + species + '/dt$'
        ax.plot(tfields, np.gradient(iene[:, -1]) / dt_fields, linewidth=2,
                label=label2)
        ax.plot(tfields, acc_drift_dote, linewidth=2, label='Inertial')
        ax.tick_params(labelsize=16)
        ax.plot([0, 100], [0, 0], linestyle='--', color='k')
        ax.set_xlim([0, 100])
        ax.set_ylim(ylim1)
        ax.tick_params(labelsize=16)
        if mime == 400:
            ax.set_xlabel(r'$t\Omega_{ci}$', fontdict=FONT, fontsize=20)
        else:
            xticks = ax.get_xticks()
            xticks_labels = [str(int(x + tshift)) for x in xticks]
            ax.set_xticklabels(xticks_labels)
        ax.set_ylabel('Energy change rate', fontdict=FONT, fontsize=20)
        text1 = r'$m_i/m_e=' + str(mime) + '$'
        ax.text(0.02, 0.9, text1, color='k', fontsize=20,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)

        ax = fig2.add_axes(box1)
        axs2.append(ax)
        COLORS = palettable.tableau.Tableau_10.mpl_colors
        ax.set_prop_cycle('color', COLORS)
        label1 = r'$K_{B' + species + '}$'
        ax.plot(tfields, bene[:, -1]/enorm, linewidth=2, label=label1)
        label2 = r'$U_' + species + '$'
        ax.plot(tfields, iene[:, -1]/enorm, linewidth=2, label=label2)
        ax.tick_params(labelsize=16)
        ax.plot([0, 100], [0, 0], linestyle='--', color='k')
        ax.set_xlim([0, 100])
        ax.set_ylim(ylim2)
        ax.tick_params(labelsize=16)
        if mime == 400:
            ax.set_xlabel(r'$t\Omega_{ci}$', fontdict=FONT, fontsize=20)
        else:
            xticks = ax.get_xticks()
            xticks_labels = [str(int(x + tshift)) for x in xticks]
            ax.set_xticklabels(xticks_labels)
        ax.set_ylabel(r'Energy$/\varepsilon_{Bx0}$', fontdict=FONT, fontsize=20)
        text1 = r'$m_i/m_e=' + str(mime) + '$'
        ax.text(0.02, 0.9, text1, color='k', fontsize=20,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)
        box1[1] -= box1[3] + 0.03

    axs1[0].legend(loc='upper center', prop={'size': 20}, ncol=3,
                   bbox_to_anchor=(0.5, 1.25),
                   shadow=False, fancybox=False, frameon=False)
    axs2[0].legend(loc='upper center', prop={'size': 20}, ncol=3,
                   bbox_to_anchor=(0.5, 1.25),
                   shadow=False, fancybox=False, frameon=False)

    fdir = '../img/img_high_mime/bulk_internal/'
    mkdir_p(fdir)
    fname = fdir + 'bulk_internal_rate_bg' + bg_str + '_' + species + '.pdf'
    fig1.savefig(fname)

    fdir = '../img/img_high_mime/bulk_internal/'
    mkdir_p(fdir)
    fname = fdir + 'bulk_internal_bg' + bg_str + '_' + species + '.pdf'
    fig2.savefig(fname)

    plt.show()


def para_perp_energization(run_name, species, tframe, show_plot=True):
    """Particle energization due to parallel and perpendicular electric field

    Args:
        run_name: PIC simulation run name
        species: particle species
        tframe: time frame
    """
    fpath = "../data/particle_interp/" + run_name + "/"
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    tstep = tframe * pic_info.particle_interval
    tframe_fluid = tstep // pic_info.fields_interval
    print(tstep, tframe, tframe_fluid)
    fname = fpath + "particle_energization_" + species + "_" + str(tstep) + ".gda"
    fdata = np.fromfile(fname, dtype=np.float32)
    nbins = int(fdata[0])
    nvar = int(fdata[1])
    ebins = fdata[2:nbins+2]
    fbins = fdata[nbins+2:].reshape((nvar, nbins))

    if species == 'i':
        ebins *= pic_info.mime  # ebins are actually gamma
    if species == 'e':
        fbins[1:, :] = div0(fbins[1:, :], fbins[0, :])
    else:
        fbins[1:, :] = div0(fbins[1:, :], fbins[0, :] * pic_info.mime)

    # normalized with thermal energy
    if species == 'e':
        vth = pic_info.vthe
    else:
        vth = pic_info.vthi
    gama = 1.0 / math.sqrt(1.0 - 3 * vth**2)
    eth = gama - 1.0
    if species == 'i':
        eth *= pic_info.mime

    ebins /= eth

    colors_Set1_9 = palettable.colorbrewer.qualitative.Set1_9.mpl_colors
    colors = colors_Set1_9[:5] + colors_Set1_9[6:] # remove yellow color
    xs, ys = 0.2, 0.15
    w1, h1 = 0.7, 0.8
    fig = plt.figure(figsize=[7, 5])
    ax1 = fig.add_axes([xs, ys, w1, h1])
    ax1.set_prop_cycle('color', colors)
    ax1.semilogx(ebins, fbins[1, :], linewidth=2, label=r"$E_\parallel$")
    ax1.semilogx(ebins, fbins[2, :], linewidth=2, label=r"$E_\perp$")
    ax1.semilogx(ebins, fbins[1, :] + fbins[2, :], linewidth=2,
                 label="Total")
    leg = ax1.legend(loc=1, prop={'size': 20}, ncol=1,
                     shadow=False, fancybox=False, frameon=False)
    smime_norm = pic_info.mime / 25.0
    if species == 'e':
        ax1.set_xlim([1E0, 500])
        ax1.set_ylim([-0.002/smime_norm, 0.004/smime_norm])
    else:
        ax1.set_xlim([1E0, 700])
        ax1.set_ylim([-0.002/smime_norm, 0.004/smime_norm])
    xlim = ax1.get_xlim()
    ax1.plot(xlim, [0, 0], color='k', linestyle='--')
    ax1.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$', fontdict=FONT, fontsize=20)
    ax1.set_ylabel('Acceleration Rate', fontdict=FONT, fontsize=20)
    ax1.tick_params(labelsize=16)

    xlim = ax1.get_xlim()
    ax2 = ax1.twinx()
    ax2.loglog(ebins, fbins[0, :]/np.gradient(ebins), linewidth=3,
               color='k')
    ax2.set_ylabel(r'$f(\varepsilon)$', fontdict=FONT, fontsize=20)
    ax2.set_xlim(xlim)
    ax2.set_ylim([1E-6, 1E6])
    ax2.tick_params(labelsize=16)

    fdir = '../img/img_high_mime/particle_energization/' + run_name + '/'
    mkdir_p(fdir)
    fname = fdir + 'para_perp_' + species + '_' + str(tframe) + '.pdf'
    fig.savefig(fname)

    if show_plot:
        plt.show()
    else:
        plt.close()


def comp_shear_energization(run_name, species, tframe, show_plot=True):
    """Particle energization due to compression and shear

    Args:
        run_name: PIC simulation run name
        species: particle species
        tframe: time frame
    """
    fpath = "../data/particle_interp/" + run_name + "/"
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    tstep = tframe * pic_info.particle_interval
    tframe_fluid = tstep // pic_info.fields_interval
    print(tstep, tframe, tframe_fluid)
    fname = fpath + "particle_energization_" + species + "_" + str(tstep) + ".gda"
    fdata = np.fromfile(fname, dtype=np.float32)
    nbins = int(fdata[0])
    nvar = int(fdata[1])
    ebins = fdata[2:nbins+2]
    fbins = fdata[nbins+2:].reshape((nvar, nbins))

    if species == 'i':
        ebins *= pic_info.mime  # ebins are actually gamma
    if species == 'e':
        fbins[1:, :] = div0(fbins[1:, :], fbins[0, :])
    else:
        fbins[1:, :] = div0(fbins[1:, :], fbins[0, :] * pic_info.mime)

    # normalized with thermal energy
    if species == 'e':
        vth = pic_info.vthe
    else:
        vth = pic_info.vthi
    gama = 1.0 / math.sqrt(1.0 - 3 * vth**2)
    eth = gama - 1.0
    if species == 'i':
        eth *= pic_info.mime

    ebins /= eth

    colors_Set1_9 = palettable.colorbrewer.qualitative.Set1_9.mpl_colors
    colors = colors_Set1_9[:5] + colors_Set1_9[6:] # remove yellow color
    xs, ys = 0.2, 0.15
    w1, h1 = 0.7, 0.8
    fig = plt.figure(figsize=[7, 5])
    ax1 = fig.add_axes([xs, ys, w1, h1])
    ax1.set_prop_cycle('color', colors)
    ax1.semilogx(ebins, fbins[2, :], linewidth=2, label=r"$E_\perp$")
    ax1.semilogx(ebins, fbins[3, :], linewidth=2, label="Compression")
    ax1.semilogx(ebins, fbins[4, :], linewidth=2, label="Shear")
    ax1.semilogx(ebins, fbins[13, :] + fbins[14, :], linewidth=2, label="Polar")
    ene = fbins[3, :] + fbins[4, :] + fbins[13, :] + fbins[14, :]
    ax1.semilogx(ebins, ene, linewidth=2, label="Sum")
    leg = ax1.legend(loc=3, prop={'size': 16}, ncol=2,
                     shadow=False, fancybox=False, frameon=False)
    smime_norm = pic_info.mime / 25.0
    if species == 'e':
        ax1.set_xlim([1E0, 500])
        ax1.set_ylim([-0.002/smime_norm, 0.004/smime_norm])
    else:
        ax1.set_xlim([1E0, 700])
        ax1.set_ylim([-0.002/smime_norm, 0.004/smime_norm])
    xlim = ax1.get_xlim()
    ax1.plot(xlim, [0, 0], color='k', linestyle='--')
    ax1.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$', fontdict=FONT, fontsize=20)
    ax1.set_ylabel('Acceleration Rate', fontdict=FONT, fontsize=20)
    ax1.tick_params(labelsize=16)

    xlim = ax1.get_xlim()
    ax2 = ax1.twinx()
    ax2.loglog(ebins, fbins[0, :]/np.gradient(ebins), linewidth=3,
               color='k')
    ax2.set_ylabel(r'$f(\varepsilon)$', fontdict=FONT, fontsize=20)
    ax2.set_xlim(xlim)
    ax2.set_ylim([1E-6, 1E6])
    ax2.tick_params(labelsize=16)

    fdir = '../img/img_high_mime/particle_energization/' + run_name + '/'
    mkdir_p(fdir)
    fname = fdir + 'comp_shear_' + species + '_' + str(tframe) + '.pdf'
    fig.savefig(fname)

    if show_plot:
        plt.show()
    else:
        plt.close()


def drift_energization(run_name, species, tframe, show_plot=True):
    """Particle energization due to particle drifts

    Args:
        run_name: PIC simulation run name
        species: particle species
        tframe: time frame
    """
    fpath = "../data/particle_interp/" + run_name + "/"
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    tstep = tframe * pic_info.particle_interval
    tframe_fluid = tstep // pic_info.fields_interval
    print(tstep, tframe, tframe_fluid)
    fname = fpath + "particle_energization_" + species + "_" + str(tstep) + ".gda"
    fdata = np.fromfile(fname, dtype=np.float32)
    nbins = int(fdata[0])
    nvar = int(fdata[1])
    ebins = fdata[2:nbins+2]
    fbins = fdata[nbins+2:].reshape((nvar, nbins))

    if species == 'i':
        ebins *= pic_info.mime  # ebins are actually gamma
    if species == 'e':
        fbins[1:, :] = div0(fbins[1:, :], fbins[0, :])
    else:
        fbins[1:, :] = div0(fbins[1:, :], fbins[0, :] * pic_info.mime)

    # normalized with thermal energy
    if species == 'e':
        vth = pic_info.vthe
    else:
        vth = pic_info.vthi
    gama = 1.0 / math.sqrt(1.0 - 3 * vth**2)
    eth = gama - 1.0
    if species == 'i':
        eth *= pic_info.mime

    ebins /= eth

    colors_Set1_9 = palettable.colorbrewer.qualitative.Set1_9.mpl_colors
    colors = colors_Set1_9[:5] + colors_Set1_9[6:] # remove yellow color
    xs, ys = 0.2, 0.15
    w1, h1 = 0.7, 0.8
    fig = plt.figure(figsize=[7, 5])
    ax1 = fig.add_axes([xs, ys, w1, h1])
    ax1.set_prop_cycle('color', colors)
    ax1.semilogx(ebins, fbins[2, :], linewidth=2, label=r"$E_\perp$")
    ax1.semilogx(ebins, fbins[5, :], linewidth=2, label="Curvature")
    ax1.semilogx(ebins, fbins[6, :], linewidth=2, label="Gradient")
    # ax1.semilogx(ebins, fbins[7, :], linewidth=2, label="Parallel")
    # ax1.semilogx(ebins, fbins[8, :], linewidth=2, label="Magnetic Moment")
    ax1.semilogx(ebins, fbins[11, :] + fbins[12, :], linewidth=2, label="Inertial")
    ax1.semilogx(ebins, fbins[15, :] + fbins[16, :], linewidth=2, label="Polar")
    ene = (fbins[5, :] + fbins[6, :] +
           fbins[7, :] + fbins[8, :] +
           fbins[11, :] + fbins[12, :] +
           fbins[15, :] + fbins[16, :])
    # ax1.semilogx(ebins, ene, linewidth=2, label="Sum")
    leg = ax1.legend(loc=3, prop={'size': 16}, ncol=2,
                     shadow=False, fancybox=False, frameon=False)
    smime_norm = pic_info.mime / 25.0
    if species == 'e':
        ax1.set_xlim([1E0, 500])
        ax1.set_ylim([-0.002/smime_norm, 0.004/smime_norm])
    else:
        ax1.set_xlim([1E0, 700])
        ax1.set_ylim([-0.002/smime_norm, 0.004/smime_norm])
    xlim = ax1.get_xlim()
    ax1.plot(xlim, [0, 0], color='k', linestyle='--')
    ax1.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$', fontdict=FONT, fontsize=20)
    ax1.set_ylabel('Acceleration Rate', fontdict=FONT, fontsize=20)
    ax1.tick_params(labelsize=16)

    xlim = ax1.get_xlim()
    ax2 = ax1.twinx()
    ax2.loglog(ebins, fbins[0, :]/np.gradient(ebins), linewidth=3,
               color='k')
    ax2.set_ylabel(r'$f(\varepsilon)$', fontdict=FONT, fontsize=20)
    ax2.set_xlim(xlim)
    ax2.set_ylim([1E-6, 1E6])
    ax2.tick_params(labelsize=16)

    fdir = '../img/img_high_mime/particle_energization/' + run_name + '/'
    mkdir_p(fdir)
    fname = fdir + 'drifts_' + species + '_' + str(tframe) + '.pdf'
    fig.savefig(fname)

    if show_plot:
        plt.show()
    else:
        plt.close()


def model_energization(run_name, species, tframe, show_plot=True):
    """Particle energization using different models

    Args:
        run_name: PIC simulation run name
        species: particle species
        tframe: time frame
    """
    fpath = "../data/particle_interp/" + run_name + "/"
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    tstep = tframe * pic_info.particle_interval
    tframe_fluid = tstep // pic_info.fields_interval
    print(tstep, tframe, tframe_fluid)
    fname = fpath + "particle_energization_" + species + "_" + str(tstep) + ".gda"
    fdata = np.fromfile(fname, dtype=np.float32)
    nbins = int(fdata[0])
    nvar = int(fdata[1])
    ebins = fdata[2:nbins+2]
    fbins = fdata[nbins+2:].reshape((nvar, nbins))

    if species == 'i':
        ebins *= pic_info.mime  # ebins are actually gamma
    if species == 'e':
        fbins[1:, :] = div0(fbins[1:, :], fbins[0, :])
    else:
        fbins[1:, :] = div0(fbins[1:, :], fbins[0, :] * pic_info.mime)

    # normalized with thermal energy
    if species == 'e':
        vth = pic_info.vthe
    else:
        vth = pic_info.vthi
    gama = 1.0 / math.sqrt(1.0 - 3 * vth**2)
    eth = gama - 1.0
    if species == 'i':
        eth *= pic_info.mime

    ebins /= eth

    colors_Set1_9 = palettable.colorbrewer.qualitative.Set1_9.mpl_colors
    colors = colors_Set1_9[:5] + colors_Set1_9[6:] # remove yellow color
    xs, ys = 0.2, 0.15
    w1, h1 = 0.7, 0.8
    fig = plt.figure(figsize=[7, 5])
    ax1 = fig.add_axes([xs, ys, w1, h1])
    ax1.set_prop_cycle('color', colors)
    ax1.semilogx(ebins, fbins[1, :], linewidth=2, label=r"$E_\parallel$")
    ax1.semilogx(ebins, fbins[2, :], linewidth=2, label=r"$E_\perp$")
    comp_shear_polar = fbins[3, :] + fbins[4, :] + fbins[13, :] + fbins[14, :]
    ax1.semilogx(ebins, comp_shear_polar, linewidth=2, label=r"Comp+Shear+Polar")
    ene = (fbins[5, :] + fbins[6, :] +
           fbins[7, :] + fbins[8, :] +
           fbins[11, :] + fbins[12, :] +
           fbins[15, :] + fbins[16, :])
    ax1.semilogx(ebins, ene, linewidth=2, label="Particle Drifts")
    leg = ax1.legend(loc=3, prop={'size': 16}, ncol=2,
                     shadow=False, fancybox=False, frameon=False)
    smime_norm = pic_info.mime / 25.0
    if species == 'e':
        ax1.set_xlim([1E0, 500])
        ax1.set_ylim([-0.002/smime_norm, 0.004/smime_norm])
    else:
        ax1.set_xlim([1E0, 700])
        ax1.set_ylim([-0.002/smime_norm, 0.004/smime_norm])
    xlim = ax1.get_xlim()
    ax1.plot(xlim, [0, 0], color='k', linestyle='--')
    ax1.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$', fontdict=FONT, fontsize=20)
    ax1.set_ylabel('Acceleration Rate', fontdict=FONT, fontsize=20)
    ax1.tick_params(labelsize=16)

    xlim = ax1.get_xlim()
    ax2 = ax1.twinx()
    ax2.loglog(ebins, fbins[0, :]/np.gradient(ebins), linewidth=3,
               color='k')
    ax2.set_ylabel(r'$f(\varepsilon)$', fontdict=FONT, fontsize=20)
    ax2.set_xlim(xlim)
    ax2.set_ylim([1E-6, 1E6])
    ax2.tick_params(labelsize=16)

    fdir = '../img/img_high_mime/particle_energization/' + run_name + '/'
    mkdir_p(fdir)
    fname = fdir + 'model_ene_' + species + '_' + str(tframe) + '.pdf'
    fig.savefig(fname)

    if show_plot:
        plt.show()
    else:
        plt.close()


def particle_energization2(plot_config):
    """Particle-based energization
    """
    mime = plot_config["mime"]
    bg = plot_config["bg"]
    tframe = plot_config["tframe"]
    species = plot_config["species"]
    bg_str = str(int(bg * 10)).zfill(2)
    run_name = "mime" + str(mime) + "_beta002_" + "bg" + bg_str
    fpath = "../data/particle_interp/" + run_name + "/"
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    tstep = tframe * pic_info.particle_interval
    tframe_fluid = tstep // pic_info.fields_interval
    print(tstep, tframe, tframe_fluid)
    fname = fpath + "particle_energization_" + species + "_" + str(tstep) + ".gda"
    fdata = np.fromfile(fname, dtype=np.float32)
    nbins = int(fdata[0])
    nvar = int(fdata[1])
    ebins = fdata[2:nbins+2]
    fbins = fdata[nbins+2:].reshape((nvar, nbins))

    fname = "../data/fluid_energization/" + run_name + "/"
    fname += "emf_ptensor_" + species + '.gda'
    fluid_ene = np.fromfile(fname, dtype=np.float32)
    nvar = int(fluid_ene[0])
    nframes = int(fluid_ene[1])
    jcurv_dote = fluid_ene[2:nframes+2]
    bulk_curv_dote = fluid_ene[nframes+2:2*nframes+2]
    jgrad_dote = fluid_ene[2*nframes+2:3*nframes+2]
    jmag_dote = fluid_ene[3*nframes+2:4*nframes+2]
    comp_ene = fluid_ene[4*nframes+2:5*nframes+2]
    shear_ene = fluid_ene[5*nframes+2:6*nframes+2]
    ptensor_ene = fluid_ene[6*nframes+2:7*nframes+2]
    pgyro_ene = fluid_ene[7*nframes+2:8*nframes+2]

    fname = "../data/fluid_energization/" + run_name + "/"
    fname += "para_perp_acc_" + species + '.gda'
    fluid_ene = np.fromfile(fname, dtype=np.float32)
    nvar = int(fluid_ene[0])
    nframes = int(fluid_ene[1])
    acc_drift_dote_t = fluid_ene[2:nframes+2]
    acc_drift_dote_s = fluid_ene[nframes+2:2*nframes+2]
    acc_drift_dote = acc_drift_dote_t + acc_drift_dote_s
    epara_ene = fluid_ene[2*nframes+2:3*nframes+2]
    eperp_ene = fluid_ene[3*nframes+2:4*nframes+2]
    acc_drift_dote[-1] = acc_drift_dote[-2]

    fbins *= ebins
    print('{:>38} {:>10.7f}'.format("Parallel electric field:",
                                    np.sum(fbins[1, :])))
    print('{:>38} {:>10.7f}'.format("Perpendicular electric field:",
                                    np.sum(fbins[2, :])))
    print('{:>38} {:>10.7f}'.format("Compression:", np.sum(fbins[3, :])))
    print('{:>38} {:>10.7f}'.format("Shear:", np.sum(fbins[4, :])))
    print('{:>38} {:>10.7f}'.format("Parallel drift:", np.sum(fbins[7, :])))
    print('{:>38} {:>10.7f}'.format("Conservation of mu:", np.sum(fbins[8, :])))
    print('{:>38} {:>10.7f}'.format("Polarization drift (time):",
                                    np.sum(fbins[9, :])))
    print('{:>38} {:>10.7f}'.format("Polarization drift (spatial):",
                                    np.sum(fbins[10, :])))
    print('{:>38} {:>10.7f}'.format("Initial drift (time):",
                                    np.sum(fbins[11, :])))
    print('{:>38} {:>10.7f}'.format("Initial drift (spatial):",
                                    np.sum(fbins[12, :])))
    print('{:>38} {:>10.7f}'.format("Curvature drift:", np.sum(fbins[5, :])))
    print('{:>38} {:>10.7f}'.format("Gradient drift:", np.sum(fbins[6, :])))
    print('{:>38} {:>10.7f}'.format("Fluid polar (time):",
                                    np.sum(fbins[13, :])))
    print('{:>38} {:>10.7f}'.format("Fluid polar (spatial):",
                                    np.sum(fbins[14, :])))
    print('{:>38} {:>10.7f}'.format("Vperp polar:",
                                    np.sum(fbins[15, :] + fbins[16, :])))
    print('{:>38} {:>10.7f}'.format("Total drifts:",
                                    np.sum(fbins[5, :] + fbins[6, :] +
                                           fbins[7, :] + fbins[8, :] +
                                           fbins[11, :] + fbins[12, :] +
                                           fbins[15, :] + fbins[16, :])))
    print('{:>38} {:>10.7f}'.format("Comp+Shear+Polar:",
                                    np.sum(fbins[3, :] + fbins[4, :] +
                                           fbins[13, :] + fbins[14, :])))
    print('{:>38} {:>10.7f}'.format("Polar + Inertial (time):",
                                    np.sum(fbins[11, :] + fbins[15, :])))
    print('{:>38} {:>10.7f}'.format("Polar + Inertial (spatial):",
                                    np.sum(fbins[12, :] + fbins[16, :])))
    print('')
    print('{:>38} {:>10.7f}'.format("Curvature drift (fluid):",
                                    jcurv_dote[tframe_fluid]))
    print('{:>38} {:>10.7f}'.format("Bulk Curvature (fluid):",
                                    bulk_curv_dote[tframe_fluid]))
    print('{:>38} {:>10.7f}'.format("Gradient drift (fluid):",
                                    jgrad_dote[tframe_fluid]))
    print('{:>38} {:>10.7f}'.format("Parallel electric field (fluid):",
                                    epara_ene[tframe_fluid]))
    print('{:>38} {:>10.7f}'.format("Perpendicular electric field (fluid):",
                                    eperp_ene[tframe_fluid]))
    print('{:>38} {:>10.7f}'.format("Fluid acceleration:",
                                    acc_drift_dote[tframe_fluid]))
    print('{:>38} {:>10.7f}'.format("Fluid acceleration (time):",
                                    acc_drift_dote_t[tframe_fluid]))
    print('{:>38} {:>10.7f}'.format("Fluid acceleration (spatial):",
                                    (acc_drift_dote_s[tframe_fluid])))
    print('{:>38} {:>10.7f}'.format("Magnetization (fluid):",
                                    jmag_dote[tframe_fluid]))
    print('{:>38} {:>10.7f}'.format("Compression (fluid):",
                                    comp_ene[tframe_fluid]))
    print('{:>38} {:>10.7f}'.format("Shear (fluid):", shear_ene[tframe_fluid]))
    print('{:>38} {:>10.7f}'.format("ptensor (fluid):",
                                    ptensor_ene[tframe_fluid]))
    print('{:>38} {:>10.7f}'.format("Comp+shear+polar (fluid):",
                                    (comp_ene[tframe_fluid] +
                                     shear_ene[tframe_fluid] +
                                     acc_drift_dote[tframe_fluid])))
    fbins /= ebins

    if species == 'i':
        ebins *= pic_info.mime  # ebins are actually gamma
    if species == 'e':
        fbins[1:, :] = div0(fbins[1:, :], fbins[0, :])
    else:
        fbins[1:, :] = div0(fbins[1:, :], fbins[0, :] * pic_info.mime)

    # normalized with thermal energy
    if species == 'e':
        vth = pic_info.vthe
    else:
        vth = pic_info.vthi
    gama = 1.0 / math.sqrt(1.0 - 3 * vth**2)
    eth = gama - 1.0
    if species == 'i':
        eth *= pic_info.mime

    ebins /= eth

    colors_Set1_9 = palettable.colorbrewer.qualitative.Set1_9.mpl_colors
    colors = colors_Set1_9[:5] + colors_Set1_9[6:] # remove yellow color
    xs, ys = 0.17, 0.15
    w1, h1 = 0.75, 0.8
    fig = plt.figure(figsize=[7, 5])
    ax1 = fig.add_axes([xs, ys, w1, h1])
    ax1.set_prop_cycle('color', colors)
    ax1.semilogx(ebins, fbins[1, :], linewidth=2, label="para")
    ax1.semilogx(ebins, fbins[2, :], linewidth=2, label="perp")
    # ax1.semilogx(ebins, fbins[3, :], linewidth=2, label="comp")
    # ax1.semilogx(ebins, fbins[4, :], linewidth=2, label="shear")
    # ax1.semilogx(ebins, fbins[17, :] + fbins[13, :] + fbins[14, :],
    #              linewidth=2, label="ptensor + polar")
    comp_shear_polar = (fbins[3, :] + fbins[4, :] +
                        fbins[13, :] + fbins[14, :])
    # comp_shear_polar = (fbins[3, :] + fbins[4, :])
    ax1.semilogx(ebins, comp_shear_polar, color='k',
                 linewidth=2, label="comp + shear + polar")
    drifts_ene = (fbins[5, :] + fbins[6, :] +
                  fbins[7, :] + fbins[8, :] +
                  # fbins[9, :] +
                  # fbins[10, :] +
                  # fbins[13, :] +
                  # fbins[14, :] +
                  fbins[15, :] +
                  fbins[16, :] +
                  fbins[11, :] + fbins[12, :])
    ax1.semilogx(ebins, drifts_ene, linewidth=2, label="All drifts + mu")
    # ax1.semilogx(ebins, fbins[2, :] - drifts_ene,
    #              linewidth=2, label="Polar target")
    # ax1.semilogx(ebins, fbins[5, :], linewidth=2, label="Curvature")
    # ax1.semilogx(ebins, fbins[6, :], linewidth=2, label="Gradient")
    # ax1.semilogx(ebins, fbins[7, :], linewidth=2, label="Parallel Drift")
    # ax1.semilogx(ebins, fbins[8, :], linewidth=2, label=r"$\mu$")
    initial_drift = (fbins[5, :] + fbins[11, :] + fbins[12, :])
    # ax1.semilogx(ebins, initial_drift, linewidth=2, label="Inertial")
    polar_drift = fbins[9, :] + fbins[10, :]
    # ax1.semilogx(ebins, polar_drift, linewidth=2, label="Polar")
    # ax1.semilogx(ebins, fbins[9, :], linewidth=2, label="Polar T")
    # ax1.semilogx(ebins, fbins[10, :], linewidth=2, label="Polar S")
    # ax1.semilogx(ebins, fbins[13, :], linewidth=2, label="Fluid polar T")
    # ax1.semilogx(ebins, fbins[14, :], linewidth=2, label="Fluid polar S")
    fluid_polar_target = fbins[2, :] - fbins[3, :] - fbins[4, :] - fbins[13, :]
    # ax1.semilogx(ebins, fluid_polar_target, linewidth=2,
    #              label="Fluid polar S (target)")
    # ax1.semilogx(ebins, fbins[11, :], linewidth=2, label="Initial T")
    # ax1.semilogx(ebins, fbins[12, :], linewidth=2, label="Initial S")
    # ax1.semilogx(ebins, fbins[15, :], linewidth=2, label="Polar-V T")
    # ax1.semilogx(ebins, fbins[16, :], linewidth=2, label="Polar-V S")
    ax1.plot(ax1.get_xlim(), [0, 0], linestyle='--', color='k')
    leg = ax1.legend(loc=3, prop={'size': 20}, ncol=1,
                     shadow=False, fancybox=False, frameon=False)
    if species == 'e':
        ax1.set_xlim([1E0, 500])
        # ax1.set_ylim([-0.002, 0.002])
    else:
        ax1.set_xlim([1E0, 500])
        ax1.set_ylim([-0.004, 0.004])
    ax1.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$', fontdict=FONT, fontsize=20)
    ax1.set_ylabel('Acceleration Rate', fontdict=FONT, fontsize=20)
    ax1.tick_params(labelsize=16)

    xlim = ax1.get_xlim()
    ax2 = ax1.twinx()
    ax2.loglog(ebins, fbins[0, :]/np.gradient(ebins), linewidth=3,
               color='k')
    ax2.set_xlim(xlim)
    ax2.tick_params(labelsize=16)

    plt.show()


def read_particle_energization(run_name, species, tindex):
    """
    The data format in the particle energization file:
         0. Particle energy bins
         1. Particle number
         2. Energization due to parallel electric field
         3. Energization due to perpendicular electric field
         4. Compression energization
         5. Shear energization
         6. Curvature drift acceleration
         7. Gradient drift acceleration
         8. Parallel drift acceleration
         9. Energization due to the conservation of magnetic moment
        10. Polarization drift acceleration (time)
        11. Polarization drift acceleration (spatial)
        12. Inertial drift acceleration (time)
        13. Inertial drift acceleration (spatial)
        14. Polarization drift in fluid form (time)
        15. Polarization drift in fluid form (spatial)
        16. Polarization drift using v instead of u (time)
        17. Polarization drift using v instead of u (spatial)
        18. Energization due to whole pressure tensor
    """
    fpath = "../data/particle_interp/" + run_name + "/"
    fname = (fpath + "particle_energization_" + species +
             "_" + str(tindex) + ".gda")
    fdata = np.fromfile(fname, dtype=np.float32)
    # The first two numbers are number of bins and variables
    nbins = int(fdata[0])
    nbinx = int(fdata[1])
    nvar = int(fdata[2])
    ebins = fdata[3:nbins+3]
    fdata = fdata[nbins+3:].reshape((nvar, nbinx, nbins))
    fbins = np.sum(fdata, axis=1)
    return (ebins, fbins)


def particle_energization_bg(plot_config):
    """Particle-based energization for simulations with the same guide field
    """
    mime = plot_config["mime"]
    bg = plot_config["bg"]
    tframe = plot_config["tframe"]
    species = plot_config["species"]
    bg_str = str(int(bg * 10)).zfill(2)
    mimes = [25, 100, 400]
    fig = plt.figure(figsize=[8, 3])
    rect = [0.08, 0.17, 0.28, 0.78]
    hgap, vgap = 0.02, 0.02
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    colors = np.copy(COLORS)
    colors[-1] = colors[5]
    colors[5] = COLORS[-1]
    pene = []
    tstart, tend = 5, 10
    nframes = tend - tstart + 1
    for imime, mime in enumerate(mimes):
        ax = fig.add_axes(rect)
        run_name = "mime" + str(mime) + "_beta002_" + "bg" + bg_str
        picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        ntp = int(pic_info.ntp)
        fnorm = 1E-3 * 25 / mime
        pene_frame = []
        for tframe in range(tstart, tend + 1):
            tstep = tframe * pic_info.particle_interval
            ebins, fbins = read_particle_energization(run_name, species, tstep)
            nvar, nbins = fbins.shape
            # ebins = ebins[1::4]
            # fbins = fbins[:, 1:].reshape((-1, nbins//4, 4)).sum(axis=2)

            if species == 'i':
                ebins *= pic_info.mime  # ebins are actually gamma
            if species == 'e':
                fbins[1:, :] = div0(fbins[1:, :], fbins[0, :])
            else:
                fbins[1:, :] = div0(fbins[1:, :], fbins[0, :] * pic_info.mime)
            ebins_new = np.logspace(math.log10(ebins.min()),
                                    math.log10(ebins.max()),
                                    4*(nbins - 1) + 1)
            ebins_new[0] = ebins[0]
            ebins_new[-1] = ebins[-1]
            nbins, = ebins_new.shape
            fbins_new = np.zeros((nvar, nbins))
            for ivar in range(nvar-1):
                f = interp1d(ebins, fbins[ivar+1, :], kind='quadratic')
                fbins_new[ivar+1, :] = f(ebins_new)
            ebins = ebins_new
            fbins = fbins_new

            # normalized with thermal energy
            if species == 'e':
                vth = pic_info.vthe
            else:
                vth = pic_info.vthi
            gama = 1.0 / math.sqrt(1.0 - 3 * vth**2)
            eth = gama - 1.0
            if species == 'i':
                eth *= pic_info.mime
            ebins /= eth

            eindex, ene0 = find_nearest(ebins, 100)

            # color = plt.cm.jet((tframe - tstart)/float(nframes), 1)
            color = colors[tframe - tstart]
            fdata = np.copy(fbins[2, :])
            fdata /= fnorm
            pene_frame.append(fbins/fnorm)
            ax.semilogx(ebins, fdata, color=color, marker='o',
                        markersize=3, linestyle='-')
            # ax.semilogx(ebins, fdata, color=color, linewidth=3)

        pene.append(pene_frame)
        ax.plot([1, 100], [0, 0], color='k', linestyle='--', linewidth=0.5)
        ax.set_xlim([1, 100])
        ax.set_ylim([-1, 3])
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in', top=True)
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')
        ax.tick_params(labelsize=12)
        ax.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$',
                      fontdict=FONT, fontsize=16)
        if imime == 0:
            ax.set_ylabel('Acceleration rate', fontsize=16)
        else:
            ax.tick_params(axis='y', labelleft=False)

        rect[0] += rect[2] + hgap
    plt.close()
    pene = np.asarray(pene)
    fig = plt.figure(figsize=[14, 10])
    rect0 = [0.07, 0.77, 0.22, 0.21]
    hgap, vgap = 0.01, 0.02
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    ax.set_prop_cycle('color', COLORS)
    tframes = range(tstart, tend + 1)
    eindex1, ene1 = find_nearest(ebins, 10)
    eindex2, ene2 = find_nearest(ebins, 200)
    elog = ebins[eindex1:eindex2+1]
    ivar = 2
    mime_index = [1, 0, 2]
    texts = ["parallel E", "perpendicular E", "compression", "shear",
             "curvature", "gradient", "parallel drift", "magnetic moment",
             "polarization-t", "polarization-s", "inertial-t", "inertial-s",
             "polarization-ft", "polarization-fs",
             "polarization-vt", "polarization-vs"]
    for ivar in range(1, nvar-1):
        if ivar % 4 == 1:
            rect = np.copy(rect0)
        ax = fig.add_axes(rect)
        for imime in mime_index:
            ts = 1 if imime == 2 else 2
            fmin_var = np.min(pene[imime, ts:, ivar, eindex1:eindex2+1], axis=0)
            fmax_var = np.max(pene[imime, ts:, ivar, eindex1:eindex2+1], axis=0)
            ax.fill_between(elog, fmin_var, fmax_var,
                            where=fmax_var >= fmin_var,
                            facecolor=COLORS[imime], interpolate=True,
                            alpha=0.8-imime*0.25)
        ax.set_xscale("log")
        ax.set_xlim([elog.min(), elog.max()])
        ax.set_ylim([-3, 5])
        ax.plot(ax.get_xlim(), [0, 0],
                linewidth=1.0, linestyle='--', color='k')
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in', top=True)
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')
        ax.text(0.05, 0.05, texts[ivar-1], color='k', fontsize=12,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='bottom',
                transform=ax.transAxes)
        if (ivar - 1) // 4 == 3:
            ax.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$', fontsize=16)
        else:
            ax.tick_params(axis='x', labelbottom=False)
        if ivar % 4 == 1:
            ax.set_ylabel('Acceleration rate', fontsize=16)
        else:
            ax.tick_params(axis='y', labelleft=False)
        rect[0] += rect[2] + hgap
        if ivar % 4 == 3:
            rect0[1] -= rect0[3] + vgap
    fdir = '../img/img_high_mime/pene_terms/'
    mkdir_p(fdir)
    fname = fdir + 'pene_' + species + '_' + bg_str + '.pdf'
    fig.savefig(fname)
    plt.show()


def particle_energization_sample(plot_config, const_va=False):
    """Particle-based energization samples
    Args:
        const_va: whether the Alfven speed is constant for different mass ratio
    """
    mime = plot_config["mime"]
    bg = plot_config["bg"]
    tframe = plot_config["tframe"]
    species = plot_config["species"]
    bg_str = str(int(bg * 10)).zfill(2)
    mimes = [25, 100, 400]
    fig = plt.figure(figsize=[7, 3.5])
    rect0 = [0.12, 0.68, 0.27, 0.24]
    hgap, vgap = 0.02, 0.03
    tframes = np.asarray([3, 5, 7, 9])
    nframes = len(tframes)
    for imime, mime in enumerate(mimes):
        if mime != 400:
            colors = np.copy(COLORS)
            colors[-1] = colors[5]
            colors[5] = COLORS[-1]
        else:
            colors = palettable.tableau.ColorBlind_10.mpl_colors
        run_name = "mime" + str(mime) + "_beta002_" + "bg" + bg_str
        if const_va and mime != 400:
            run_name += '_high'
        picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        ntp = int(pic_info.ntp)
        # Normalize to wci
        fnorm = 1.0 / (mime * pic_info.dtwpe / pic_info.dtwce)
        pene_frame = []
        # tframes = np.asarray(range(tstart, tend + 1, 1))
        if mime == 400:
            tframes += 1
        for iframe, tframe in enumerate(tframes):
            tstep = tframe * pic_info.particle_interval
            ebins, fbins = read_particle_energization(run_name, species, tstep)
            nvar, nbins = fbins.shape
            # ebins = ebins[1::4]
            # fbins = fbins[:, 1:].reshape((-1, nbins//4, 4)).sum(axis=2)

            if species == 'i':
                ebins *= pic_info.mime  # ebins are actually gamma
            if species == 'e':
                fbins[1:, :] = div0(fbins[1:, :], fbins[0, :])
            else:
                fbins[1:, :] = div0(fbins[1:, :], fbins[0, :])
            ebins_new = np.logspace(math.log10(ebins.min()),
                                    math.log10(ebins.max()),
                                    4*(nbins - 1) + 1)
            ebins_new[0] = ebins[0]
            ebins_new[-1] = ebins[-1]
            nbins_new, = ebins_new.shape
            fbins_new = np.zeros((nvar, nbins_new))
            for ivar in range(nvar-1):
                f = interp1d(ebins, fbins[ivar+1, :], kind='quadratic')
                fbins_new[ivar+1, :] = f(ebins_new)
            # ebins = ebins_new
            # fbins = fbins_new
            # nbins = nbins_new

            # normalized with thermal energy
            if species == 'e':
                vth = pic_info.vthe
            else:
                vth = pic_info.vthi
            gama = 1.0 / math.sqrt(1.0 - 3 * vth**2)
            eth = gama - 1.0
            if species == 'i':
                eth *= pic_info.mime
            ebins /= eth

            eindex, ene0 = find_nearest(ebins, 100)

            pene_frame.append(fbins/fnorm)

        pene_frame = np.asarray(pene_frame)
        rect = np.copy(rect0)
        if species == 'e':
            var = [[1], [2], [5]]
            texts = [r'$E_\parallel$', r'$E_\perp$', 'Curvature']
        else:
            var = [[11, 12], [13, 14], [5]]
            texts = ["Inertial'", 'Polarization', 'Curvature']

        for ivar, var_indices in enumerate(var):
            ax = fig.add_axes(rect)
            for iframe, tframe in enumerate(tframes):
                color = colors[iframe]
                flog = np.zeros(nbins)
                for var_index in var_indices:
                    flog += pene_frame[iframe, var_index, :]
                ax.semilogx(ebins, flog, color=color,
                            linestyle='-', linewidth=1.0)

            ax.plot([1, 200], [0, 0], color='k', linestyle='--', linewidth=0.5)
            ax.set_xlim([1, 200])
            if species == 'i':
                if bg_str == '00':
                    ax.set_ylim([-0.05, 0.08])
                elif bg_str == '02':
                    ax.set_ylim([-0.05, 0.06])
                elif bg_str == '04':
                    ax.set_ylim([-0.05, 0.05])
                else:
                    ax.set_ylim([-0.05, 0.05])
            else:
                if bg_str == '00':
                    ax.set_ylim([-0.03, 0.08])
                elif bg_str == '02':
                    ax.set_ylim([-0.03, 0.08])
                elif bg_str == '04':
                    ax.set_ylim([-0.03, 0.08])
                else:
                    ax.set_ylim([-0.03, 0.08])
            ax.tick_params(bottom=True, top=True, left=True, right=True)
            ax.tick_params(axis='x', which='minor', direction='in', top=True)
            ax.tick_params(axis='x', which='major', direction='in')
            ax.tick_params(axis='y', which='minor', direction='in')
            ax.tick_params(axis='y', which='major', direction='in')
            ax.tick_params(labelsize=8)
            if ivar == len(var) - 1:
                ax.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$',
                              fontdict=FONT, fontsize=10)
            else:
                ax.tick_params(axis='x', labelbottom=False)

            if imime == 0:
                ax.set_ylabel(r'$\alpha/\Omega_{ci}$', fontsize=10)
            else:
                ax.tick_params(axis='y', labelleft=False)
            if ivar == 0:
                title = r"$m_i/m_e=" + str(mime) + "$"
                ax.set_title(title, fontsize=10)
            if imime == 0:
                ax.text(-0.39, 0.5, texts[ivar], color='k', fontsize=10,
                        rotation='vertical',
                        bbox=dict(facecolor='none', alpha=1.0,
                                  edgecolor='none', pad=10.0),
                        horizontalalignment='left', verticalalignment='center',
                        transform=ax.transAxes)

            if imime == 0 and ivar == 0:
                rect_cbar = np.copy(rect)
                rect_cbar[0] += hgap * 0.5
                rect_cbar[1] = rect[1] + rect[3] * 0.8
                rect_cbar[2] = rect[2] * 0.4
                rect_cbar[3] = 0.02
                cax = fig.add_axes(rect_cbar)
                Set1_4 = palettable.colorbrewer.qualitative.Set1_4.mpl_colors
                cmap = mpl.colors.ListedColormap(Set1_4)
                tmin = 30
                tmax = 110
                sm = plt.cm.ScalarMappable(cmap=cmap,
                                           norm=plt.Normalize(vmin=tmin,
                                                              vmax=tmax))
                cax.tick_params(axis='x', which='major', direction='in')
                sm._A = []
                cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
                cbar.ax.yaxis.set_label_position('right')
                cbar.ax.set_ylabel(r'$t\Omega_{ci}$', fontsize=8,
                                   rotation=0, labelpad=10)
                cbar.set_ticks([40, 60, 80, 100])
                cbar.ax.tick_params(labelsize=8)

            if imime == 2 and ivar == 0:
                rect_cbar = np.copy(rect)
                rect_cbar[0] += hgap * 0.5
                rect_cbar[1] = rect[1] + rect[3] * 0.8
                rect_cbar[2] = rect[2] * 0.4
                rect_cbar[3] = 0.02
                cax = fig.add_axes(rect_cbar)
                ColorBlind_10 = palettable.tableau.ColorBlind_10.mpl_colors
                cmap = mpl.colors.ListedColormap(ColorBlind_10[:4])
                tmin = 20
                tmax = 100
                sm = plt.cm.ScalarMappable(cmap=cmap,
                                           norm=plt.Normalize(vmin=tmin,
                                                              vmax=tmax))
                cax.tick_params(axis='x', which='major', direction='in')
                sm._A = []
                cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
                cbar.ax.yaxis.set_label_position('right')
                cbar.ax.set_ylabel(r'$t\Omega_{ci}$', fontsize=8,
                                   rotation=0, labelpad=10)
                cbar.set_ticks([30, 50, 70, 90])
                cbar.set_ticklabels([20, 50, 70, 90])
                cbar.ax.tick_params(labelsize=8)
            rect[1] -= rect[3] + vgap

        rect0[0] += rect0[2] + hgap

    fdir = '../img/img_high_mime/pene_evolve/'
    mkdir_p(fdir)
    fname = fdir + 'pene_' + species + '_' + bg_str + '.pdf'
    fig.savefig(fname)
    plt.show()


def compare_particle_energization(plot_config, const_va=False):
    """Compare particle-based energization samples
    Args:
        const_va: whether the Alfven speed is constant for different mass ratio
    """
    mime = plot_config["mime"]
    bg = plot_config["bg"]
    tframe = plot_config["tframe"]
    species = plot_config["species"]
    bg_str = str(int(bg * 10)).zfill(2)
    mimes = [25, 100, 400]
    if bg > 1.0:
        tframes = np.linspace(5, 10, 6, dtype=np.int)
    else:
        tframes = np.linspace(4, 9, 6, dtype=np.int)
    nframes = len(tframes)
    colors = [palettable.colorbrewer.sequential.Blues_7.mpl_colors,
              palettable.colorbrewer.sequential.Oranges_7.mpl_colors,
              palettable.colorbrewer.sequential.Greens_7.mpl_colors]
    pene_frame = []
    ebins_mime = []
    for imime, mime in enumerate(mimes):
        run_name = "mime" + str(mime) + "_beta002_" + "bg" + bg_str
        if const_va and mime != 400:
            run_name += '_high'
        picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        ntp = int(pic_info.ntp)
        # Normalize to wci
        fnorm = 1.0 / (mime * pic_info.dtwpe / pic_info.dtwce)
        if mime == 400:
            tframes += 1
        for iframe, tframe in enumerate(tframes):
            tstep = tframe * pic_info.particle_interval
            ebins, fbins = read_particle_energization(run_name, species, tstep)
            nvar, nbins = fbins.shape

            if species == 'i':
                ebins *= pic_info.mime  # ebins are actually gamma
            if species == 'e':
                fbins[1:, :] = div0(fbins[1:, :], fbins[0, :])
            else:
                fbins[1:, :] = div0(fbins[1:, :], fbins[0, :])
            ebins_new = np.logspace(math.log10(ebins.min()),
                                    math.log10(ebins.max()),
                                    4*(nbins - 1) + 1)
            ebins_new[0] = ebins[0]
            ebins_new[-1] = ebins[-1]
            nbins_new, = ebins_new.shape
            fbins_new = np.zeros((nvar, nbins_new))
            for ivar in range(nvar-1):
                f = interp1d(ebins, fbins[ivar+1, :], kind='quadratic')
                fbins_new[ivar+1, :] = f(ebins_new)

            # normalized with thermal energy
            if species == 'e':
                vth = pic_info.vthe
            else:
                vth = pic_info.vthi
            gama = 1.0 / math.sqrt(1.0 - 3 * vth**2)
            eth = gama - 1.0
            if species == 'i':
                eth *= pic_info.mime
            ebins /= eth

            eindex, ene0 = find_nearest(ebins, 100)

            pene_frame.append(fbins/fnorm)
        ebins_mime.append(ebins)

    if species == 'e':
        var = [[1], [5]]
        texts = [r'$E_\parallel$', 'Curvature']
    else:
        var = [[11, 12], [13, 14], [5]]
        texts = ["Inertial'", 'Polarization', 'Curvature']
    pene_frame = np.asarray(pene_frame)
    nframes_tot, nvar, nbins = pene_frame.shape
    pene_frame = pene_frame.reshape(len(mimes), nframes_tot//len(mimes), nvar, nbins)
    if species == 'e':
        fig = plt.figure(figsize=[3.5, 5])
        rect0 = [0.19, 0.82, 0.37, 0.13]
        hgap, vgap = 0.02, 0.015
    else:
        fig = plt.figure(figsize=[5, 5])
        rect0 = [0.12, 0.82, 0.255, 0.13]
        hgap, vgap = 0.04, 0.015

    COLORS = palettable.tableau.Tableau_10.mpl_colors
    nframes = len(tframes)
    for ivar, var_indices in enumerate(var):
        rect = np.copy(rect0)
        for iframe, tframe in enumerate(tframes):
            ax = fig.add_axes(rect)
            rect[1] -= rect[3] + vgap
            for imime, mime in enumerate(mimes):
                flog = np.zeros(nbins)
                for var_index in var_indices:
                    flog += pene_frame[imime, iframe, var_index, :]
                ax.semilogx(ebins_mime[imime], flog, color=COLORS[imime],
                            linestyle='-', marker='o', markersize=3, linewidth=1.0)
                # ax.semilogx(ebins_mime[imime], flog, color=COLORS[imime],
                #             linestyle='-', linewidth=1.0)
            ax.plot([1, 200], [0, 0], color='k', linestyle='--', linewidth=0.5)
            if species == 'e':
                if bg > 1.0:
                    ax.set_xlim([5, 500])
                else:
                    ax.set_xlim([5, 100])
            else:
                if bg > 1.0:
                    ax.set_xlim([1, 500])
                else:
                    ax.set_xlim([1, 100])
            if species == 'i':
                if bg_str == '00':
                    ax.set_ylim([-0.05, 0.10])
                elif bg_str == '02':
                    ax.set_ylim([-0.04, 0.07])
                elif bg_str == '04':
                    ax.set_ylim([-0.02, 0.05])
                else:
                    ax.set_ylim([-0.02, 0.05])
            else:
                if bg_str == '00':
                    ax.set_ylim([-0.05, 0.08])
                elif bg_str == '02':
                    ax.set_ylim([-0.03, 0.08])
                elif bg_str == '04':
                    ax.set_ylim([-0.03, 0.05])
                else:
                    ax.set_ylim([-0.03, 0.05])
            if iframe != nframes - 1:
                ax.tick_params(axis='x', labelbottom=False)
            else:
                ax.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$', fontsize=10)
            if ivar != 0:
                ax.tick_params(axis='y', labelleft=False)
            ax.tick_params(labelsize=8)
            ax.tick_params(bottom=True, top=False, left=True, right=True)
            ax.tick_params(axis='x', which='minor', direction='in', top=True)
            ax.tick_params(axis='x', which='major', direction='in')
            ax.tick_params(axis='y', which='minor', direction='in')
            ax.tick_params(axis='y', which='major', direction='in')
            if iframe == 0:
                ax.set_title(texts[ivar], fontsize=10)
            if ivar == 0:
                text1 = r'$t\Omega_{ci}=' + str(tframe*10-10) + '$'
                ax.text(-0.39, 0.5, text1, color='k', fontsize=10,
                        rotation='vertical',
                        bbox=dict(facecolor='none', alpha=1.0,
                                  edgecolor='none', pad=10.0),
                        horizontalalignment='left', verticalalignment='center',
                        transform=ax.transAxes)
            if ivar == 0 and iframe == 2:
                ax.text(0.07, 0.70, r'$m_i/m_e=25$', color=COLORS[0], fontsize=8,
                        bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                        horizontalalignment='left', verticalalignment='bottom',
                        transform=ax.transAxes)
                ax.text(0.07, 0.55, r'$m_i/m_e=100$', color=COLORS[1], fontsize=8,
                        bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                        horizontalalignment='left', verticalalignment='bottom',
                        transform=ax.transAxes)
                ax.text(0.07, 0.40, r'$m_i/m_e=400$', color=COLORS[2], fontsize=8,
                        bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                        horizontalalignment='left', verticalalignment='bottom',
                        transform=ax.transAxes)
        rect0[0] += rect0[2] + hgap


    fdir = '../img/img_high_mime/pene_evolve2/'
    mkdir_p(fdir)
    if const_va:
        fname = fdir + 'pene_' + species + '_' + bg_str + '_high.pdf'
    else:
        fname = fdir + 'pene_' + species + '_' + bg_str + '.pdf'
    fig.savefig(fname)
    plt.show()


def fluid_energization_multi(species):
    """Plot fluid energization for multiple runs
    """
    bgs = np.asarray([0.0, 0.2, 0.4, 0.8])
    mimes = np.asarray([25, 100, 400])
    for ibg, bg in enumerate(bgs):
        for imime, mime in enumerate(mimes):
            fluid_energization(mime, bg, species, show_plot=False)


def get_length_scales(const_va):
    """Get different length scales in a PIC simulation
    """
    mimes = np.asarray([25, 100, 400])
    nmime = len(mimes)
    scales = {}
    deltas_r = []
    debye_lens_r = []
    for imime, mime in enumerate(mimes):
        run_name = 'mime' + str(mime) + '_beta002_' + 'bg00'
        if const_va and mime != 400:
            run_name += '_high'
        picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        de = 1 / math.sqrt(pic_info.mime)
        di = de * math.sqrt(pic_info.mime)
        rhoe = pic_info.vthe * pic_info.dtwpe / pic_info.dtwce * de
        rhoi = pic_info.vthi * pic_info.dtwpe / pic_info.dtwci * de
        scales[r'$1/d_e(' + str(mime) + ')$'] = de
        scales[r'$1/d_i$'] = di
        scales[r'$1/\rho_e(' + str(mime) + ')$'] = rhoe
        scales[r'$1/\rho_i(' + str(mime) + ')$'] = rhoi
        scales[r'$1/L_x$'] = pic_info.lx_di
        deltas_r.append(1.0/pic_info.dx_di)
        debye_lens_r.append(math.sqrt(pic_info.mime)/pic_info.vthe)
    scales_sorted = sorted(scales.items(), key=lambda kv: kv[1])
    keys = []
    values = []
    for (key, value) in scales_sorted:
        keys.append(key)
        values.append(value)
    return (keys, values, deltas_r, debye_lens_r)


def plot_length_scales(ax1, const_va):
    """Plot different length scales in a PIC simulation
    Args:
        ax1: axis for the plot
        const_va: whether the Alfven speed is constant for different mass ratio
    """
    keys, values, deltas_r, debye_lens_r = get_length_scales(const_va)
    ys = np.zeros(len(values))
    ax1.spines["left"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.yaxis.set_ticks_position('none')
    ax1.xaxis.set_ticks_position('bottom')
    ax1.semilogx(values, ys, linestyle='none')
    # ticks = np.unique(1/np.asarray(values[::-1]))
    # ticks = np.asarray([0.01, 1, 5, 10, 20, 50, 100, 200])
    ticks = np.asarray([0.1, 1, 5, 10, 20, 50, 100, 200])
    deltas_r = np.asarray(deltas_r)
    debye_lens_r = np.asarray(debye_lens_r)

    xlim = ax1.get_xlim()
    ax1.set_xticks(ticks)
    # ax1.set_xticklabels(keys[::-1])
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    lenx = np.log10(xlim[1]) - math.log10(xlim[0])
    xpos = (np.log10(ticks) - math.log10(xlim[0])) / lenx
    # ax1.text(xpos[0], -2, r'$L_x^{-1}$', color='k', fontsize=10,
    #          bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
    #          horizontalalignment='center', verticalalignment='top',
    #          transform=ax1.transAxes)
    ax1.text(-0.1, 1, r'$10^{-2}$', color='k', fontsize=8,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='center', verticalalignment='top',
             transform=ax1.transAxes)
    ax1.text(-0.1, -2, r'$L_x^{-1}$', color='k', fontsize=10,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='center', verticalalignment='top',
             transform=ax1.transAxes)
    ax1.text(xpos[1], -2, r'$d_i^{-1}$', color='k', fontsize=10,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='center', verticalalignment='top',
             transform=ax1.transAxes)
    ax1.text(xpos[2], -2, r'$d_e^{-1}$', color=COLORS[0], fontsize=10,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='center', verticalalignment='top',
             transform=ax1.transAxes)
    ax1.text(xpos[3], -2, r'$d_e^{-1}$', color=COLORS[1], fontsize=10,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='center', verticalalignment='top',
             transform=ax1.transAxes)
    ax1.text(xpos[3], -9, r'$\rho_i^{-1}$', color='k', fontsize=10,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='center', verticalalignment='bottom',
             transform=ax1.transAxes)
    ax1.text(xpos[4], -2, r'$d_e^{-1}$', color=COLORS[2], fontsize=10,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='center', verticalalignment='top',
             transform=ax1.transAxes)
    ax1.text(xpos[5], -2, r'$\rho_e^{-1}$', color=COLORS[0], fontsize=10,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='center', verticalalignment='top',
             transform=ax1.transAxes)
    ax1.text(xpos[6], -2, r'$\rho_e^{-1}$', color=COLORS[1], fontsize=10,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='center', verticalalignment='top',
             transform=ax1.transAxes)
    ax1.text(xpos[7], -2, r'$\rho_e^{-1}$', color=COLORS[2], fontsize=10,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='center', verticalalignment='top',
             transform=ax1.transAxes)

    xpos1 = (np.log10(deltas_r) - math.log10(xlim[0])) / lenx
    if const_va:
        ax1.plot([deltas_r[0], deltas_r[0]], [0, 1], linewidth=1.0, color='k')
        ax1.text(xpos1[2], 1, r'$\Delta^{-1}$', color='k', fontsize=10,
                 bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                 horizontalalignment='center', verticalalignment='bottom',
                 transform=ax1.transAxes)
    else:
        ax1.plot([deltas_r[0], deltas_r[0]], [0, 1], linewidth=1.0, color='k')
        ax1.plot([deltas_r[1], deltas_r[1]], [0, 1], linewidth=1.0, color='k')
        ax1.plot([deltas_r[2], deltas_r[2]], [0, 1], linewidth=1.0, color='k')
        ax1.text(xpos1[0], 1, r'$\Delta^{-1}$', color=COLORS[0], fontsize=10,
                 bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                 horizontalalignment='center', verticalalignment='bottom',
                 transform=ax1.transAxes)
        ax1.text(xpos1[1], 1, r'$\Delta^{-1}$', color=COLORS[1], fontsize=10,
                 bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                 horizontalalignment='center', verticalalignment='bottom',
                 transform=ax1.transAxes)
        ax1.text(xpos1[2], 1, r'$\Delta^{-1}$', color=COLORS[2], fontsize=10,
                 bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                 horizontalalignment='center', verticalalignment='bottom',
                 transform=ax1.transAxes)

    xpos2 = (np.log10(debye_lens_r) - math.log10(xlim[0])) / lenx
    if const_va:
        ax1.text(xpos2[2], -9, r'$\lambda_D^{-1}$', color='k', fontsize=10,
                 bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                 horizontalalignment='center', verticalalignment='bottom',
                 transform=ax1.transAxes)
    else:
        ax1.text(xpos2[0], -9, r'$\lambda_D^{-1}$', color=COLORS[0], fontsize=10,
                 bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                 horizontalalignment='center', verticalalignment='bottom',
                 transform=ax1.transAxes)
        ax1.text(xpos2[1], -9, r'$\lambda_D^{-1}$', color=COLORS[1], fontsize=10,
                 bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                 horizontalalignment='center', verticalalignment='bottom',
                 transform=ax1.transAxes)
        ax1.text(xpos2[2], -9, r'$\lambda_D^{-1}$', color=COLORS[2], fontsize=10,
                 bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                 horizontalalignment='center', verticalalignment='bottom',
                 transform=ax1.transAxes)

    ax1.tick_params(bottom=True, top=False, left=False, right=False)
    ax1.tick_params(axis='x', which='minor', direction='in', bottom=False)
    ax1.tick_params(axis='y', labelleft=False)
    ax1.tick_params(axis='x', labelbottom=False)
    ax1.tick_params(labelsize=8)


def plot_spatial_scales(const_va):
    """Plot spatial scales only
    Args:
        const_va: whether the Alfven speed is constant for different mass ratio
    """
    mimes = np.asarray([25, 100, 400])
    tshifts = shift_tframes(const_va)
    max_shift = max(tshifts.values())
    toffset = {key: max_shift - tshifts[key] for key in tshifts.keys()}
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    fig = plt.figure(figsize=[3.5, 1.0])
    rect = [0.14, 0.94, 0.82, 0.02]
    ax = fig.add_axes(rect)

    for mime in mimes:
        run_name = 'mime' + str(mime) + '_beta002_' + 'bg00'
        if const_va and mime != 400:
            run_name += '_high'
        fdir = "../data/kappa_dist/" + run_name + '/'
        ax.loglog([1E-1, 2E2], [0.1, 0.1], linestyle='none', linewidth=1)
    ax.tick_params(axis='y', labelleft=False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.tick_params(labelsize=8)
    ax.set_xlim([1E-1, 2E2])
    ax.set_xlabel(r'$kd_i$', fontdict=FONT, fontsize=10, labelpad=-5)

    rect[1] = 0.5
    rect[3] = 0.05
    ax1 = fig.add_axes(rect)

    xlim = ax.get_xlim()
    ax1.set_xlim(xlim)
    plot_length_scales(ax1, const_va)
    ax1.text(-0.15, -9, r'$m_i/m_e=$', color='k', fontsize=10,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='left', verticalalignment='bottom',
             transform=ax1.transAxes)
    ax1.text(0.06, -9, r'$25$', color=COLORS[0], fontsize=10,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='left', verticalalignment='bottom',
             transform=ax1.transAxes)
    ax1.text(0.12, -9, r'$100$', color=COLORS[1], fontsize=10,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='left', verticalalignment='bottom',
             transform=ax1.transAxes)
    ax1.text(0.20, -9, r'$400$', color=COLORS[2], fontsize=10,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='left', verticalalignment='bottom',
             transform=ax1.transAxes)
    fdir = '../img/img_high_mime/'
    mkdir_p(fdir)
    if const_va:
        fname = fdir + 'spatial_scales_high.pdf'
    else:
        fname = fdir + 'spatial_scales.pdf'
    fig.savefig(fname)

    plt.show()


def plot_spatial_scales_arrow(const_va):
    """Plot spatial scales with arrow
    Args:
        const_va: whether the Alfven speed is constant for different mass ratio
    """
    mimes = np.asarray([25, 100, 400])
    tshifts = shift_tframes(const_va)
    max_shift = max(tshifts.values())
    toffset = {key: max_shift - tshifts[key] for key in tshifts.keys()}
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    fig = plt.figure(figsize=[3.5, 1.0])
    rect = [0.04, 0.30, 0.86, 0.3]
    ax1 = fig.add_axes(rect)
    for direction in ["left", "right", "bottom", "top"]:
        ax1.spines[direction].set_visible(False)
    ax1.tick_params(axis='y', labelleft=False)
    xmin_log, xmax_log = math.log10(0.5), math.log10(300)
    ax1.set_xlim([xmin_log, xmax_log])
    ax1.set_ylim([0, 1])
    xmin, xmax = ax1.get_xlim()
    ax1.arrow(0, 0, xmax, 0, color="k", clip_on=False,
              head_width=0.17, head_length=0.1, linewidth=1)
    ax1.plot([xmin, 0], [0, 0], color='k', linewidth=1,
             linestyle=':')
    ax1.text(1.05, 0, r'$kd_i$', color='k', fontsize=8,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='left', verticalalignment='center',
             transform=ax1.transAxes)

    keys, values, deltas_r, debye_lens_r = get_length_scales(const_va)
    deltas_r = np.asarray(deltas_r)
    debye_lens_r = np.asarray(debye_lens_r)

    xlim = ax1.get_xlim()
    lenx = xlim[1] - xlim[0]
    major_ticks = np.asarray([xmin, 0, 1, 2])
    minor_ticks = np.log10(np.linspace(2, 9, 8))
    minor_ticks = np.concatenate((minor_ticks, np.log10(np.linspace(20, 90, 8))))
    minor_ticks = np.concatenate((minor_ticks, np.log10(np.linspace(200, 300, 2))))
    ax1.set_xticks(major_ticks)
    ax1.set_xticks(minor_ticks, minor=True)
    ax1.tick_params(bottom=True, top=False, left=False, right=False)
    ax1.tick_params(axis='x', which='minor', direction='out')
    ax1.tick_params(axis='x', which='major', direction='out')
    ax1.tick_params(axis='x', labelbottom=False)
    xpos = -xmin / lenx
    ax1.text(xpos, -0.3, r'$10^0$', color='k', fontsize=8,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='center', verticalalignment='top',
             transform=ax1.transAxes)
    xpos = (1-xmin) / lenx
    ax1.text(xpos, -0.3, r'$10^1$', color='k', fontsize=8,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='center', verticalalignment='top',
             transform=ax1.transAxes)
    xpos = (2-xmin) / lenx
    ax1.text(xpos, -0.3, r'$10^2$', color='k', fontsize=8,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='center', verticalalignment='top',
             transform=ax1.transAxes)

    COLORS = palettable.tableau.Tableau_10.mpl_colors
    ax1.text(0, -0.3, r'$10^{-2}$', color='k', fontsize=8,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='center', verticalalignment='top',
             transform=ax1.transAxes)
    ax1.text(0, 0.5, r'$L_x^{-1}$', color='k', fontsize=8,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='center', verticalalignment='top',
             transform=ax1.transAxes)
    ticks = np.asarray([1, 5, 10, 20, 50, 100, 200])
    ticks_log = np.log10(ticks)
    xpos = (ticks_log - xlim[0]) / lenx
    # rect2 = np.copy(rect)
    # rect2[3] = 0.05
    # rect2[1] = rect[1] - rect2[3] - 0.008
    # rect2[2] = (2.4 - xmin) * rect[2] / lenx
    # ax2 = fig.add_axes(rect2)
    # for direction in ["left", "right", "bottom", "top"]:
    #     ax2.spines[direction].set_visible(False)
    # ax2.tick_params(axis='x', labelbottom=False)
    # ax2.tick_params(axis='y', labelleft=False)
    # ax2.set_xlim([xmin, 2.4])
    # ax2.set_xticks(ticks_log)
    # ax2.set_yticks([])
    # ax2.tick_params(bottom=False, top=True, left=False, right=False)
    # ax2.tick_params(axis='x', which='minor', direction='in', top=True)
    # ax2.tick_params(axis='x', which='major', direction='in')
    ax1.text(xpos[0], 0.5, r'$d_i^{-1}$', color='k', fontsize=8,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='center', verticalalignment='top',
             transform=ax1.transAxes)
    ax1.text(xpos[1], 0.5, r'$d_e^{-1}$', color=COLORS[0], fontsize=8,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='center', verticalalignment='top',
             transform=ax1.transAxes)
    ax1.text(xpos[2], 0.5, r'$d_e^{-1}$', color=COLORS[1], fontsize=8,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='center', verticalalignment='top',
             transform=ax1.transAxes)
    ax1.text(xpos[2], 1.1, r'$\rho_i^{-1}$', color='k', fontsize=8,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='center', verticalalignment='top',
             transform=ax1.transAxes)
    ax1.text(xpos[3], 0.5, r'$d_e^{-1}$', color=COLORS[2], fontsize=8,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='center', verticalalignment='top',
             transform=ax1.transAxes)
    ax1.text(xpos[4], 0.5, r'$\rho_e^{-1}$', color=COLORS[0], fontsize=8,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='center', verticalalignment='top',
             transform=ax1.transAxes)
    ax1.text(xpos[5], 0.5, r'$\rho_e^{-1}$', color=COLORS[1], fontsize=8,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='center', verticalalignment='top',
             transform=ax1.transAxes)
    ax1.text(xpos[6], 0.5, r'$\rho_e^{-1}$', color=COLORS[2], fontsize=8,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='center', verticalalignment='top',
             transform=ax1.transAxes)

    deltas_r_log = np.log10(deltas_r)
    xpos1 = (deltas_r_log - xlim[0]) / lenx
    if const_va:
        # ax1.plot([deltas_r_log[0], deltas_r_log[0]], [0.15, 0.7], linewidth=0.5, color='k')
        ax1.text(xpos1[2], 1.1, r'$\Delta^{-1}$', color='k', fontsize=8,
                 bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                 horizontalalignment='center', verticalalignment='top',
                 transform=ax1.transAxes)
    else:
        # ax1.plot([deltas_r_log[0], deltas_r_log[0]], [-0.15, -0.7], linewidth=0.5, color='k')
        # ax1.plot([deltas_r_log[1], deltas_r_log[1]], [-0.15, -0.7], linewidth=0.5, color='k')
        # ax1.plot([deltas_r_log[2], deltas_r_log[2]], [-0.15, -0.7], linewidth=0.5, color='k')
        ax1.text(xpos1[0], 1.7, r'$\Delta^{-1}$', color=COLORS[0], fontsize=8,
                 bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                 horizontalalignment='center', verticalalignment='top',
                 transform=ax1.transAxes)
        ax1.text(xpos1[1], 1.7, r'$\Delta^{-1}$', color=COLORS[1], fontsize=8,
                 bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                 horizontalalignment='center', verticalalignment='top',
                 transform=ax1.transAxes)
        ax1.text(xpos1[2], 1.7, r'$\Delta^{-1}$', color=COLORS[2], fontsize=8,
                 bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                 horizontalalignment='center', verticalalignment='top',
                 transform=ax1.transAxes)

    xpos2 = (np.log10(debye_lens_r) - xlim[0]) / lenx
    if const_va:
        ax1.text(xpos2[2], 1.1, r'$\lambda_D^{-1}$', color='k', fontsize=8,
                 bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                 horizontalalignment='center', verticalalignment='top',
                 transform=ax1.transAxes)
    else:
        ax1.text(xpos2[0], 1.1, r'$\lambda_D^{-1}$', color=COLORS[0], fontsize=8,
                 bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                 horizontalalignment='center', verticalalignment='top',
                 transform=ax1.transAxes)
        ax1.text(xpos2[1], 1.1, r'$\lambda_D^{-1}$', color=COLORS[1], fontsize=8,
                 bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                 horizontalalignment='center', verticalalignment='top',
                 transform=ax1.transAxes)
        ax1.text(xpos2[2], 1.1, r'$\lambda_D^{-1}$', color=COLORS[2], fontsize=8,
                 bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                 horizontalalignment='center', verticalalignment='top',
                 transform=ax1.transAxes)

    rect3 = [0.01, 0.75, 0.4, 0.2]
    ax3 = fig.add_axes(rect3)
    for direction in ["left", "right", "bottom", "top"]:
        ax3.spines[direction].set_visible(False)
    ax3.tick_params(axis='x', labelbottom=False)
    ax3.tick_params(axis='y', labelleft=False)
    ax3.tick_params(bottom=False, top=False, left=False, right=False)
    fancybox = mpatches.FancyBboxPatch([0.05, 0.05], 0.9, 0.9,
                                       boxstyle=mpatches.BoxStyle('square', pad=0.),
                                       linewidth=0.5, alpha=0.1,
                                       facecolor=[0, 0, 0], edgecolor=(1, 1, 1))
    ax3.add_patch(fancybox)
    ax3.text(0.07, 0.45, r'$m_i/m_e=$', color='k', fontsize=10,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='left', verticalalignment='center',
             transform=ax3.transAxes)
    ax3.text(0.48, 0.45, r'$25$', color=COLORS[0], fontsize=10,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='left', verticalalignment='center',
             transform=ax3.transAxes)
    ax3.text(0.60, 0.45, r'$100$', color=COLORS[1], fontsize=10,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='left', verticalalignment='center',
             transform=ax3.transAxes)
    ax3.text(0.76, 0.45, r'$400$', color=COLORS[2], fontsize=10,
             bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
             horizontalalignment='left', verticalalignment='center',
             transform=ax3.transAxes)
    fdir = '../img/img_high_mime/'
    mkdir_p(fdir)
    if const_va:
        fname = fdir + 'spatial_scales_arrow2_high.pdf'
    else:
        fname = fdir + 'spatial_scales_arrow2.pdf'
    fig.savefig(fname)

    plt.show()


def calc_kappa_dist(bg, tframe, const_va, show_plot=True):
    """Calculate the magnetic curvature

    Args:
        bg: guide field strength
        tframe: time frame
        const_va: whether the Alfven speed is constant for different mass ratio
    """
    # mimes = np.asarray([25, 100, 400])
    mimes = np.asarray([25])
    fig = plt.figure(figsize=[7, 5])
    rect = [0.12, 0.12, 0.8, 0.8]
    hgap, vgap = 0.022, 0.02
    nmime, = mimes.shape
    tshifts = shift_tframes(const_va)
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    bg_str = str(int(bg * 10)).zfill(2)
    ax = fig.add_axes(rect)
    ax.set_prop_cycle('color', COLORS)
    kappa_min, kappa_max = 1E-2, 1E3
    nbins_kappa = 500
    kappa_bins_edge = np.logspace(math.log10(kappa_min),
                                  math.log10(kappa_max), nbins_kappa+1)
    kappa_bins = 0.5 * (kappa_bins_edge[:-1] + kappa_bins_edge[1:])
    dkappa = np.diff(kappa_bins_edge)
    for imime, mime in enumerate(mimes):
        run_name = 'mime' + str(mime) + '_beta002_' + 'bg' + bg_str
        if const_va and mime != 400:
            run_name += '_high'
        picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        idx = 1.0 / pic_info.dx_di
        idz = 1.0 / pic_info.dz_di
        smime = math.sqrt(pic_info.mime)
        tframe_shift = tframe + tshifts[str(mime)]
        kwargs = {"current_time": tframe_shift,
                  "xl": 0, "xr": pic_info.lx_di,
                  "zb": -pic_info.lz_di*0.5, "zt": pic_info.lz_di*0.5}
        fname = pic_info.run_dir + "data/bx.gda"
        x, z, bx = read_2d_fields(pic_info, fname, **kwargs)
        fname = pic_info.run_dir + "data/by.gda"
        x, z, by = read_2d_fields(pic_info, fname, **kwargs)
        fname = pic_info.run_dir + "data/bz.gda"
        x, z, bz = read_2d_fields(pic_info, fname, **kwargs)
        iabsb = 1.0 / np.sqrt(bx**2 + by**2 + bz**2)
        bx = bx * iabsb
        by = by * iabsb
        bz = bz * iabsb
        kappax = bx * np.gradient(bx, axis=1) * idx + bz * np.gradient(bx, axis=0) * idz
        kappay = bx * np.gradient(by, axis=1) * idx + bz * np.gradient(by, axis=0) * idz
        kappaz = bx * np.gradient(bz, axis=1) * idx + bz * np.gradient(bz, axis=0) * idz
        kappa = np.sqrt(kappax**2 + kappay**2 + kappaz**2)
        p1 = ax.imshow(np.abs(kappa), vmin=1E-2, vmax=1E0,
                       cmap=plt.cm.inferno, aspect='auto',
                       origin='lower', interpolation='bicubic')
        fkappa, _ = np.histogram(kappa, bins=kappa_bins_edge)
        fkappa = fkappa / dkappa
        # ax.loglog(kappa_bins, fkappa)
        # ax.plot(kappax[:, 4096])
        # ax.plot(kappay[:, 4096])
        # ax.plot(kappaz[:, 4096])
        # ax.plot(kappa[:, 4096])

    # ax.tick_params(bottom=True, top=False, left=True, right=True)
    # ax.tick_params(axis='x', which='minor', direction='in', top=True)
    # ax.tick_params(axis='x', which='major', direction='in')
    # ax.tick_params(axis='y', which='minor', direction='in')
    # ax.tick_params(axis='y', which='major', direction='in')
    # ax.set_xlim([1E-2, 2E2])
    # ax.set_ylim([1E0, 1E9])
    # ax.set_xlabel(r'$\kappa d_i$', fontdict=FONT, fontsize=10)
    # ax.set_ylabel(r'$f(\kappa)$', fontdict=FONT, fontsize=10)
    # ax.text(0.03, 0.21, r'$m_i/m_e=25$', color=COLORS[0], fontsize=10,
    #         bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
    #         horizontalalignment='left', verticalalignment='bottom',
    #         transform=ax.transAxes)
    # ax.text(0.03, 0.13, r'$m_i/m_e=100$', color=COLORS[1], fontsize=10,
    #         bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
    #         horizontalalignment='left', verticalalignment='bottom',
    #         transform=ax.transAxes)
    # ax.text(0.03, 0.05, r'$m_i/m_e=400$', color=COLORS[2], fontsize=10,
    #         bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
    #         horizontalalignment='left', verticalalignment='bottom',
    #         transform=ax.transAxes)
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_kappa_dist(bg, tframe, const_va, show_plot=True):
    """Plot the distribution of magnetic curvature
    Args:
        bg: guide field strength
        tframe: time frame
        const_va: whether the Alfven speed is constant for different mass ratio
    """
    mimes = np.asarray([25, 100, 400])
    bg_str = str(int(bg * 10)).zfill(2)
    tshifts = shift_tframes(const_va)
    max_shift = max(tshifts.values())
    toffset = {key: max_shift - tshifts[key] for key in tshifts.keys()}
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    fig = plt.figure(figsize=[3.5, 2.5])
    rect = [0.14, 0.34, 0.82, 0.64]
    ax = fig.add_axes(rect)
    ax.set_prop_cycle('color', COLORS)

    for mime in mimes:
        run_name = 'mime' + str(mime) + '_beta002_' + 'bg' + bg_str
        if const_va and mime != 400:
            run_name += '_high'
        fdir = "../data/kappa_dist/" + run_name + '/'
        tframe_shifted = tframe + tshifts[str(mime)]
        # tframe_shifted = tframe + 10 if mime == 400 else tframe
        fname = fdir + "fkappa_" + str(tframe_shifted) + ".gda"
        fdata = np.fromfile(fname, dtype=np.float32)
        nbins_kappa = int(fdata[0])
        kappa_bins_edge = fdata[1:nbins_kappa+2]
        kappa_bins = 0.5 * (kappa_bins_edge[:-1] + kappa_bins_edge[1:])
        fkappa = fdata[nbins_kappa+2:]
        fkappa /= np.diff(kappa_bins_edge)
        if not const_va:
            fkappa *= 400.0 / mime
        ax.loglog(kappa_bins, fkappa, linewidth=1)

    ax.tick_params(bottom=True, top=False, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top=True)
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.tick_params(labelsize=8)
    ax.set_xlim([1E-2, 2E2])
    ax.set_ylim([1E0, 1E9])
    ax.set_xlabel(r'$\kappa d_i$', fontdict=FONT, fontsize=10)
    ax.set_ylabel(r'$f(\kappa)$', fontdict=FONT, fontsize=10)
    ax.text(0.03, 0.21, r'$m_i/m_e=25$', color=COLORS[0], fontsize=10,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes)
    ax.text(0.03, 0.13, r'$m_i/m_e=100$', color=COLORS[1], fontsize=10,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes)
    ax.text(0.03, 0.05, r'$m_i/m_e=400$', color=COLORS[2], fontsize=10,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes)

    rect[1] = 0.19
    rect[3] = 0.02
    ax1 = fig.add_axes(rect)

    xlim = ax.get_xlim()
    ax1.set_xlim(xlim)
    plot_length_scales(ax1, const_va)
    fdir = '../img/img_high_mime/kappa_dist/bg' + bg_str + '/'
    mkdir_p(fdir)
    if const_va:
        fname = fdir + 'kappa_dist_high_' + str(tframe) + '.pdf'
    else:
        fname = fdir + 'kappa_dist_' + str(tframe) + '.pdf'
    fig.savefig(fname)

    if show_plot:
        plt.show()
    else:
        plt.close()


def energetic_rho(plot_config, const_va, show_plot=True):
    """Plot densities for energetic particles
    Args:
        plot_config: plot configuration
        const_va: whether the Alfven speed is constant for different mass ratio
    """
    mimes = np.asarray([25, 100, 400])
    nmins = [0.001, 0.0003, 0.0001]
    nmaxs = [0.1, 0.03, 0.01]
    fig = plt.figure(figsize=[7, 5])
    rect = [0.12, 0.71, 0.78, 0.27]
    hgap, vgap = 0.02, 0.03
    nmime, = mimes.shape
    tshifts = shift_tframes(const_va)
    bg_str = str(int(plot_config["bg"] * 10)).zfill(2)
    tframe = plot_config["tframe"]
    species = plot_config["species"]
    for imime, mime in enumerate(mimes):
        run_name = 'mime' + str(mime) + '_beta002_' + 'bg' + bg_str
        if const_va and mime != 400:
            run_name += '_high'
        picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        pic_run_dir = pic_info.run_dir
        if species == 'e':
            vth = pic_info.vthe
        else:
            vth = pic_info.vthi
        tratio = int(pic_info.particle_interval / pic_info.fields_interval)
        tframe_fields = tframe * tratio
        if mime == 400:
            tframe_fields += 10
            tframe_shifted += 1
        else:
            tframe_shifted = tframe
        xmin, xmax = 0, pic_info.lx_di
        zmin, zmax = -pic_info.lz_di * 0.5, pic_info.lz_di * 0.5
        nmin, nmax = nmins[imime], nmaxs[imime]
        nx, nz = pic_info.nx//4, pic_info.nz//4
        kwargs = {"current_time": tframe_fields,
                  "xl": 0, "xr": pic_info.lx_di,
                  "zb": -pic_info.lz_di, "zt": pic_info.lz_di}
        fname = pic_run_dir + "data/Ay.gda"
        x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)

        nbands = 7
        ntot = np.zeros((nz, nx))
        nhigh = np.zeros((nz, nx))
        tindex = tframe_shifted * pic_info.particle_interval
        for iband in range(nbands):
            print("Energy band: %d" % iband)
            fname = (pic_run_dir + "data-smooth2/n" + species + "_" +
                     str(iband) + "_" + str(tindex) + ".gda")
            fdata = np.fromfile(fname, dtype=np.float32)
            nrho = fdata.reshape((nz, nx))
            if iband >= 3:
                nhigh += nrho
            ntot += nrho
        nhigh += 1E-6

        # fraction_h = nhigh / ntot
        ax = fig.add_axes(rect)
        p1 = ax.imshow(nhigh[:, :],
                       extent=[xmin, xmax, zmin, zmax],
                       # vmin=nmin, vmax=nmax,
                       norm = LogNorm(vmin=nmin, vmax=nmax),
                       cmap=plt.cm.inferno, aspect='auto',
                       origin='lower', interpolation='bicubic')
        ax.contour(x, z, Ay, colors='w', linewidths=0.5)
        ax.set_ylim([-10, 10])
        ax.tick_params(bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='x', which='minor', direction='in')
        ax.tick_params(axis='x', which='major', direction='in')
        ax.tick_params(axis='y', which='minor', direction='in')
        ax.tick_params(axis='y', which='major', direction='in')
        if imime < nmime - 1:
            ax.tick_params(axis='x', labelbottom=False)
        else:
            ax.set_xlabel(r'$x/d_i$', fontsize=16)
        ax.set_ylabel(r'$z/d_i$', fontsize=16)
        ax.tick_params(labelsize=12)

        rect_cbar = np.copy(rect)
        rect_cbar[0] += rect[2] + 0.02
        rect_cbar[2] = 0.015
        cbar_ax = fig.add_axes(rect_cbar)
        cbar = fig.colorbar(p1, cax=cbar_ax, extend='both')
        # cbar.set_ticks(np.linspace(0.05, 0.2, num=4))
        cbar.ax.tick_params(labelsize=10)
        label1 = r'$n(\varepsilon > 40\varepsilon_\text{th})$'
        # cbar_ax.set_ylabel(label1, fontsize=24)
        # ax.set_title(label1, fontsize=24)
        # ax.text(0.98, 0.88, label1, color='k', fontsize=16,
        #         bbox=dict(facecolor='w', alpha=1.0,
        #                   edgecolor='none', boxstyle="round,pad=0.1"),
        #         horizontalalignment='right',
        #         verticalalignment='center',
        #         transform=ax.transAxes)
        rect[1] -= rect[3] + vgap

    fdir = '../img/img_high_mime/energetic_rho/bg' + bg_str + '/'
    mkdir_p(fdir)
    if const_va:
        fname = fdir + 'energetic_high_' + species + '_' + str(tframe) + ".jpg"
    else:
        fname = fdir + 'energetic_' + species + '_' + str(tframe) + ".jpg"
    fig.savefig(fname, dpi=400)

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_magnetic_power_spectrum(bg, tframe, const_va, show_plot=True):
    """plot the magnetic power spectrum

    Args:
        bg: guide field strength
        tframe: time frame
        const_va: whether the Alfven speed is constant for different mass ratio
    """
    mimes = np.asarray([25, 100, 400])
    nmime, = mimes.shape
    tshifts = shift_tframes(const_va)
    bg_str = str(int(bg * 10)).zfill(2)
    fig = plt.figure(figsize=[3.5, 2.5])
    rect = [0.14, 0.34, 0.82, 0.62]
    ax = fig.add_axes(rect)
    COLORS = palettable.tableau.Tableau_10.mpl_colors
    ax.set_prop_cycle('color', COLORS)
    b0s = []
    for imime, mime in enumerate(mimes):
        run_name = 'mime' + str(mime) + '_beta002_' + 'bg' + bg_str
        if const_va and mime != 400:
            run_name += '_high'
        picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        b0s.append(pic_info.b0)
    b0s = np.asarray(b0s)
    b0_max = max(b0s)
    fnorm = (b0_max**2 / b0s**2) * (max(mimes) / np.asarray(mimes))

    for imime, mime in enumerate(mimes):
        run_name = 'mime' + str(mime) + '_beta002_' + 'bg' + bg_str
        if const_va and mime != 400:
            run_name += '_high'
        picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        fdir = '../data/power_spectrum/' + run_name + '/bz/'
        tframe_shifted = tframe + tshifts[str(mime)]
        fname = fdir + 'bz_' + str(tframe_shifted) + '.gda'
        fdata = np.fromfile(fname)
        nbins_k = int(fdata[0])
        kbins_edge = fdata[1:nbins_k+2]
        kbins = 0.5 * (kbins_edge[1:] + kbins_edge[:-1])
        power_spect = fdata[nbins_k+2:]
        ax.loglog(kbins, power_spect * fnorm[imime], linewidth=1)

    kstart, krange = 25, 22
    pindex = -2.0
    power_k = kbins[kstart:]**pindex * 5E2
    ax.loglog(kbins[kstart:kstart+krange], power_k[:krange] * 2,
              linestyle='--', linewidth=1, color='k')
    power_index = "{%0.1f}" % pindex
    tname = r'$\sim k^{' + power_index + '}$'
    ax.text(0.5, 0.4, tname, color='black', fontsize=10,
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)
    ax.text(0.03, 0.21, r'$m_i/m_e=25$', color=COLORS[0], fontsize=10,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes)
    ax.text(0.03, 0.13, r'$m_i/m_e=100$', color=COLORS[1], fontsize=10,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes)
    ax.text(0.03, 0.05, r'$m_i/m_e=400$', color=COLORS[2], fontsize=10,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='bottom',
            transform=ax.transAxes)

    ax.tick_params(labelsize=16)
    ax.tick_params(bottom=True, top=False, left=True, right=True)
    ax.tick_params(axis='x', which='minor', direction='in', top=True)
    ax.tick_params(axis='x', which='major', direction='in', top=True)
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')
    ax.tick_params(labelsize=8)
    ax.set_xlim([5E-2, 2E2])
    ax.set_ylim([1E0, 1E6])
    ax.set_xlabel(r'$kd_i$', fontsize=10)
    ax.set_ylabel(r'$E_{B_z}(k)$', fontsize=10)

    rect[1] = 0.19
    rect[3] = 0.02
    ax1 = fig.add_axes(rect)

    xlim = ax.get_xlim()
    ax1.set_xlim(xlim)
    plot_length_scales(ax1, const_va)

    fdir = '../img/img_high_mime/power_spectrum/bg' + bg_str + '/'
    mkdir_p(fdir)
    if const_va:
        fname = fdir + '/mag_power_high_' + str(tframe) + '.pdf'
    else:
        fname = fdir + '/mag_power_' + str(tframe) + '.pdf'
    fig.savefig(fname)

    if show_plot:
        plt.show()
    else:
        plt.close()


def calc_magnetic_power_spectrum(bg, mime, tframe, const_va):
    """calculate the magnetic power spectrum

    Args:
        bg: guide field strength
        mime: proton to electron mass ratio
        tframe: time frame
        const_va: whether the Alfven speed is constant for different mass ratio
    """
    bg_str = str(int(bg * 10)).zfill(2)
    run_name = 'mime' + str(mime) + '_beta002_' + 'bg' + bg_str
    if const_va and mime != 400:
        run_name += '_high'
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    kwargs = {"current_time": tframe,
              "xl": 0, "xr": pic_info.lx_di,
              "zb": -pic_info.lz_di*0.5, "zt": pic_info.lz_di*0.5}
    components = ['x', 'y', 'z']
    nbins_k = 64
    fdata = np.zeros(2 * nbins_k + 2)
    fdata[0] = nbins_k
    for comp in components:
        fname = pic_info.run_dir + "data/b" + comp + ".gda"
        x, z, bfield = read_2d_fields(pic_info, fname, **kwargs)
        lx = np.max(x) - np.min(x)
        lz = np.max(z) - np.min(z)
        nx, = x.shape
        nz, = z.shape
        bfield_k = np.fft.rfft2(bfield)
        b2_k = np.absolute(bfield_k)**2
        dvol = pic_info.dx_di * pic_info.dz_di * pic_info.mime
        ene_b = np.sum(0.5*bfield**2) * dvol

        xstep = x[1] - x[0]
        kx = np.fft.fftfreq(nx, xstep)
        idx = np.argsort(kx)
        zstep = z[1] - z[0]
        kz = np.fft.fftfreq(nz, zstep)
        idz = np.argsort(kz)

        kxs, kzs = np.meshgrid(kx[:nx//2 + 1], kz)
        ks = np.sqrt(kxs * kxs + kzs * kzs)
        kmin = 1E-2
        kmax = np.max(ks)
        kmin_log, kmax_log = math.log10(kmin), math.log10(kmax)
        kbins = np.logspace(kmin_log, kmax_log, nbins_k + 1, endpoint=True)
        power_spect, kbins_edges = np.histogram(ks, bins=kbins,
                                                weights=b2_k * ks, density=True)
        power_spect *= ene_b
        fdata[1:nbins_k+2] = kbins
        fdata[nbins_k+2:] = power_spect
        fdir = '../data/power_spectrum/' + run_name + '/b' + comp + '/'
        mkdir_p(fdir)
        fname = fdir + 'b' + comp + '_' + str(tframe) + '.gda'
        fdata.tofile(fname)


def get_cmd_args():
    """Get command line arguments
    """
    default_run_name = 'mime25_beta002_bg00_lx100'
    default_run_dir = ('/net/scratch3/xiaocanli/reconnection/mime25/' +
                       'mime25_beta002_bg00/')
    parser = argparse.ArgumentParser(description='High-mass-ratio runs')
    parser.add_argument('--run_dir', action="store", default=default_run_dir,
                        help='run directory')
    parser.add_argument('--run_name', action="store", default=default_run_name,
                        help='run name')
    parser.add_argument('--multi_frames', action="store_true", default=False,
                        help='whether analyzing multiple frames')
    parser.add_argument('--time_loop', action="store_true", default=False,
                        help='whether analyzing multiple frames using a time loop')
    parser.add_argument('--multi_runs', action="store_true", default=False,
                        help='whether analyzing multiple runs')
    parser.add_argument('--species', action="store", default='e',
                        help='particle species')
    parser.add_argument('--mime', action="store", default='25', type=int,
                        help='ion-to-electron mass ratio')
    parser.add_argument('--bg', action="store", default='0.0', type=float,
                        help='ion-to-electron mass ratio')
    parser.add_argument('--tframe', action="store", default='30', type=int,
                        help='Time frame for fields')
    parser.add_argument('--tstart', action="store", default='0', type=int,
                        help='Starting time frame')
    parser.add_argument('--tend', action="store", default='30', type=int,
                        help='Ending time frame')
    parser.add_argument('--calc_rrate', action="store_true", default=False,
                        help='whether calculating reconnection rate')
    parser.add_argument('--plot_rrate', action="store_true", default=False,
                        help='whether plotting reconnection rate')
    parser.add_argument('--plot_rrate_mime', action="store_true", default=False,
                        help='whether plotting reconnection rate for different mi/me')
    parser.add_argument('--ene_evol', action="store_true", default=False,
                        help='whether plotting energy evolution')
    parser.add_argument('--ene_conv', action="store_true", default=False,
                        help='whether calculating energy conversion rate')
    parser.add_argument('--iene_conv', action="store_true", default=False,
                        help='whether calculating internal energy conversion rate')
    parser.add_argument('--ene_part', action="store_true", default=False,
                        help='whether plotting energy partition')
    parser.add_argument('--ene_part_mime', action="store_true", default=False,
                        help='whether plotting energy partition for different mime')
    parser.add_argument('--plot_jy', action="store_true", default=False,
                        help='whether plotting jy')
    parser.add_argument('--plot_va', action="store_true", default=False,
                        help='whether plotting the Alfven speed')
    parser.add_argument('--plot_bulkv', action="store_true", default=False,
                        help='whether plotting the bulk flow velocity')
    parser.add_argument('--plot_anisotropy', action="store_true", default=False,
                        help='whether plotting pressure anisotropy')
    parser.add_argument('--fluid_energization', action="store_true", default=False,
                        help='whether plotting fluid energization terms')
    parser.add_argument('--particle_energization', action="store_true", default=False,
                        help='whether plotting particle energization terms')
    parser.add_argument('--pene_bg', action="store_true", default=False,
                        help='whether plotting particle energization for the same bg')
    parser.add_argument('--pene_sample', action="store_true", default=False,
                        help='whether plotting particle energization samples')
    parser.add_argument('--comp_pene', action="store_true", default=False,
                        help='whether compare particle energization for different mass ratio')
    parser.add_argument('--para_perp', action="store_true", default=False,
                        help='whether plotting particle energization due to' +
                        ' parallel and perpendicular electric field')
    parser.add_argument('--comp_shear', action="store_true", default=False,
                        help='whether plotting particle energization due to' +
                        ' compression and shear')
    parser.add_argument('--drifts', action="store_true", default=False,
                        help='whether plotting particle energization due to' +
                        ' particle drifts')
    parser.add_argument('--model_ene', action="store_true", default=False,
                        help='whether plotting particle energization using' +
                        ' different models')
    parser.add_argument('--fluid_ene_mime', action="store_true", default=False,
                        help='whether plotting fluid energization for different mime')
    parser.add_argument('--fluid_ene_frac', action="store_true", default=False,
                        help='whether plotting fluid energization fraction')
    parser.add_argument('--compare_fluid_ene', action="store_true", default=False,
                        help='whether comparing fluid energization')
    parser.add_argument('--bulk_internal', action="store_true", default=False,
                        help='whether plotting bulk and internal energy')
    parser.add_argument('--iene_evol', action="store_true", default=False,
                        help='whether plotting internal energy evolution')
    parser.add_argument('--iene_part', action="store_true", default=False,
                        help='whether plotting internal energy partition')
    parser.add_argument('--espect_early', action="store_true", default=False,
                        help='whether plotting energy spectrum early in the simulation')
    parser.add_argument('--stacked_spect', action="store_true", default=False,
                        help='whether plotting stacked energy spectrum')
    parser.add_argument('--evolve_spect', action="store_true", default=False,
                        help='whether plotting evolving energy spectrum')
    parser.add_argument('--evolve_pspect', action="store_true", default=False,
                        help='whether plotting evolving momentum spectrum')
    parser.add_argument('--plot_ene2d', action="store_true", default=False,
                        help='whether plotting 2D energization terms')
    parser.add_argument('--plot_agyq', action="store_true", default=False,
                        help='whether plotting Q agyrotropy parameter')
    parser.add_argument('--plot_agyq_bg', action="store_true", default=False,
                        help='whether plotting Q agyrotropy parameter for a single Bg')
    parser.add_argument('--plot_temp', action="store_true", default=False,
                        help='whether plotting plasma temperature')
    parser.add_argument('--plot_nrho', action="store_true", default=False,
                        help='whether plotting electron density')
    parser.add_argument('--stacked_agyq', action="store_true", default=False,
                        help='whether plotting stacked Q agyrotropy parameter')
    parser.add_argument('--const_va', action="store_true", default=False,
                        help='whether Alfven speed is constant for different mass ratio')
    parser.add_argument('--calc_kappa_dist', action="store_true", default=False,
                        help='whether to calculate the distribution of kappa')
    parser.add_argument('--kappa_dist', action="store_true", default=False,
                        help='whether to plot the distribution of kappa')
    parser.add_argument('--energetic_rho', action="store_true", default=False,
                        help='whether to plot energetic particle density')
    parser.add_argument('--calc_mag_power', action="store_true", default=False,
                        help='whether to calculate magnetic power-spectrum')
    parser.add_argument('--plot_mag_power', action="store_true", default=False,
                        help='whether to calculate magnetic power-spectrum')
    parser.add_argument('--spatial_scales', action="store_true", default=False,
                        help='whether to plot different spatial scales')
    parser.add_argument('--spatial_scales_arrow', action="store_true", default=False,
                        help='whether to plot different spatial scales with arrow')
    parser.add_argument('--high_bg', action="store_true", default=False,
                        help='whether to include runs with a higher guide field')
    return parser.parse_args()


def analysis_single_frame(plot_config, args):
    """Analysis for single time frame
    """
    if args.multi_runs:
        if args.calc_rrate:
            calc_rrate_multi(args.mime, args.const_va)
        if args.plot_rrate:
            plot_rrate_multi(args.mime, args.const_va)
        if args.fluid_energization:
            fluid_energization_multi(args.species)
    else:
        if args.ene_evol:
            energy_evolution(args.bg, args.const_va)
        if args.iene_evol:
            internal_energy_evolution(args.bg, args.const_va)
        if args.ene_conv:
            energy_conversion(args.const_va, args.high_bg)
        if args.iene_conv:
            internal_energy_conversion(args.const_va)
        if args.ene_part:
            energy_partition(args.bg, args.const_va)
        if args.iene_part:
            internal_energy_partition(args.bg, args.const_va)
        if args.ene_part_mime:
            energy_partition_mime(args.const_va, args.high_bg)
        if args.plot_rrate_mime:
            plot_rrate_mime(args.bg, args.const_va)
        if args.plot_jy:
            plot_jy(args.tframe, args.const_va, args.high_bg, show_plot=True)
        if args.plot_va:
            plot_va(args.tframe, show_plot=True)
        if args.plot_bulkv:
            plot_bulkv(args.tframe, args.const_va, show_plot=True)
        if args.plot_anisotropy:
            plot_pressure_anisotropy(plot_config, show_plot=True)
        if args.fluid_energization:
            fluid_energization(args.mime, args.bg,
                               args.species, show_plot=True)
        if args.particle_energization:
            particle_energization2(plot_config)
        if args.pene_bg:
            particle_energization_bg(plot_config)
        if args.pene_sample:
            particle_energization_sample(plot_config, args.const_va)
        if args.comp_pene:
            compare_particle_energization(plot_config, args.const_va)
        if args.para_perp:
            para_perp_energization(args.run_name, args.species, args.tframe)
        if args.comp_shear:
            comp_shear_energization(args.run_name, args.species, args.tframe)
        if args.drifts:
            drift_energization(args.run_name, args.species, args.tframe)
        if args.model_ene:
            model_energization(args.run_name, args.species, args.tframe)
        if args.fluid_ene_mime:
            fluid_energization_mime(args.bg, args.species, show_plot=True)
        if args.fluid_ene_frac:
            fluid_energization_fraction(args.species, args.const_va, args.high_bg)
        if args.compare_fluid_ene:
            compare_fluid_energization(args.species, args.bg, args.const_va)
        if args.bulk_internal:
            bulk_internal_energy(args.bg, args.species, show_plot=True)
        if args.espect_early:
            energy_spectrum_early(args.bg, args.species, args.tframe)
        if args.stacked_spect:
            stacked_spectrum(args.species, args.const_va)
        if args.evolve_spect:
            evolving_spectrum(args.species, args.const_va, args.high_bg)
        if args.evolve_pspect:
            evolving_momentum_spectrum(args.species, args.const_va, args.high_bg)
        if args.plot_ene2d:
            plot_ene2d(plot_config)
        if args.plot_agyq:
            plot_agyq(plot_config)
        if args.plot_temp:
            plot_temp(plot_config)
        if args.plot_nrho:
            plot_nrho(plot_config)
        if args.plot_agyq_bg:
            plot_agyq_bg(plot_config)
        if args.stacked_agyq:
            plot_stacked_agyq(plot_config)
        if args.calc_kappa_dist:
            calc_kappa_dist(args.bg, args.tframe, args.const_va)
        if args.kappa_dist:
            plot_kappa_dist(args.bg, args.tframe, args.const_va)
        if args.energetic_rho:
            energetic_rho(plot_config, args.const_va)
        if args.calc_mag_power:
            calc_magnetic_power_spectrum(args.bg, args.mime, args.tframe, args.const_va)
        if args.plot_mag_power:
            plot_magnetic_power_spectrum(args.bg, args.tframe, args.const_va)
        if args.spatial_scales:
            plot_spatial_scales(args.const_va)
        if args.spatial_scales_arrow:
            plot_spatial_scales_arrow(args.const_va)


def process_input(args, plot_config, tframe):
    """process one time frame"""
    plot_config["tframe"] = tframe
    print("Time frame %d" % tframe)
    if args.plot_jy:
        plot_jy(tframe, args.const_va, args.high_bg, show_plot=False)
    elif args.espect_early:
        energy_spectrum_early(args.bg, args.species, tframe, show_plot=False)
    elif args.plot_agyq_bg:
        plot_agyq_bg(plot_config, show_plot=False)
    elif args.plot_anisotropy:
        plot_pressure_anisotropy(plot_config, show_plot=False)
    elif args.plot_temp:
        plot_temp(plot_config, show_plot=False)
    elif args.plot_nrho:
        plot_nrho(plot_config, show_plot=False)
    elif args.kappa_dist:
        plot_kappa_dist(args.bg, tframe, args.const_va, show_plot=False)
    elif args.calc_mag_power:
        calc_magnetic_power_spectrum(args.bg, args.mime, tframe, args.const_va)
    elif args.plot_mag_power:
        plot_magnetic_power_spectrum(args.bg, tframe, args.const_va, show_plot=False)


def analysis_multi_frames(plot_config, args):
    """Analysis for multiple time frames
    """
    tframes = range(args.tstart, args.tend + 1)
    if args.time_loop:
        for tframe in tframes:
            plot_config["tframe"] = tframe
            if args.para_perp:
                para_perp_energization(args.run_name, args.species,
                                       tframe, show_plot=False)
            if args.comp_shear:
                comp_shear_energization(args.run_name, args.species,
                                        tframe, show_plot=False)
            if args.drifts:
                drift_energization(args.run_name, args.species,
                                   tframe, show_plot=False)
            if args.model_ene:
                model_energization(args.run_name, args.species,
                                   tframe, show_plot=False)
            if args.energetic_rho:
                energetic_rho(plot_config, args.const_va, show_plot=False)
    else:
        # ncores = multiprocessing.cpu_count()
        ncores = 16
        Parallel(n_jobs=ncores)(delayed(process_input)(args, plot_config, tframe)
                                for tframe in tframes)


def main():
    """business logic for when running this module as the primary one!"""
    args = get_cmd_args()
    plot_config = {}
    plot_config["pic_run"] = args.run_name
    plot_config["pic_run_dir"] = args.run_dir
    plot_config["tframe"] = args.tframe
    plot_config["tstart"] = args.tstart
    plot_config["tend"] = args.tend
    plot_config["species"] = args.species
    plot_config["bg"] = args.bg
    plot_config["mime"] = args.mime
    plot_config["high_bg"] = args.high_bg
    if args.multi_frames:
        analysis_multi_frames(plot_config, args)
    else:
        analysis_single_frame(plot_config, args)


if __name__ == "__main__":
    main()
