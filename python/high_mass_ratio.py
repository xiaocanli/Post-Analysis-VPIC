"""
Analysis procedures for the paper on high mass-ratio
"""
import argparse
import itertools
import math
import multiprocessing
import os.path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import palettable
from scipy import signal

import pic_information
from contour_plots import read_2d_fields
from joblib import Parallel, delayed
from json_functions import read_data_from_json
from shell_functions import mkdir_p

mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
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
    phi = signal.medfilt(phi, kernel_size=nk)
    dtwpe = pic_info.dtwpe
    dtwce = pic_info.dtwce
    dtwci = pic_info.dtwci
    mime = pic_info.mime
    dtf_wpe = pic_info.dt_fields * dtwpe / dtwci
    reconnection_rate = np.gradient(phi) / dtf_wpe
    b0 = pic_info.b0
    va = dtwce * math.sqrt(1.0 / mime) / dtwpe
    reconnection_rate /= b0 * va
    reconnection_rate[-1] = reconnection_rate[-2]
    tfields = pic_info.tfields

    return (tfields, reconnection_rate)


def calc_rrate_multi(mime):
    """Calculate reconnection rate for multiple runs

    Args:
        mime: ion to electron mass ratio
    """
    base_dir = "/net/scratch3/xiaocanli/reconnection/mime" + str(mime) + "/"
    for bguide in ["00", "02", "04", "08"]:
        run_name = "mime" + str(mime) + "_beta002_bg" + str(bguide)
        run_dir = base_dir + run_name + "/"
        tfields, rrate = calc_reconnection_rate(run_dir, run_name)
        odir = "../data/rate/"
        mkdir_p(odir)
        fname = odir + "rrate_" + run_name + ".dat"
        np.savetxt(fname, (tfields, rrate))


def plot_rrate_multi(mime):
    """Plot reconnection rate for multiple runs
    """
    fig = plt.figure(figsize=[7, 5])
    ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
    ax.set_prop_cycle('color', COLORS)
    bgs = np.asarray([0, 0.2, 0.4, 0.8])
    for bg in bgs:
        bg_str = str(int(bg * 10)).zfill(2)
        run_name = "mime" + str(mime) + "_beta002_bg" + bg_str
        fdir = "../data/rate/"
        fname = fdir + "rrate_" + run_name + ".dat"
        tfields, rrate = np.genfromtxt(fname)
        ltext = r"$B_g=" + str(bg) + "$"
        ax.plot(tfields, rrate, linewidth=2, label=ltext)
    ax.set_ylim([0, 0.13])
    if mime == 400:
        beg = 18
    else:
        beg = 30
    ax.plot([beg, beg], ax.get_ylim(), color='k', linestyle='--')
    ax.legend(loc=1, prop={'size': 16}, ncol=2,
              shadow=False, fancybox=False, frameon=False)
    ax.tick_params(labelsize=16)
    ax.set_xlabel(r'$t\Omega_{ci}$', fontdict=FONT, fontsize=24)
    ax.set_ylabel(r'$E_R$', fontdict=FONT, fontsize=24)
    fdir = '../img/rate/'
    mkdir_p(fdir)
    fname = fdir + 'rrate_' + str(mime) + '.pdf'
    fig.savefig(fname)
    plt.show()


def energy_evolution(bg):
    """Plot energy evolution for runs with the same guide field

    Args:
        bg: guide field strength
    """
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
                      linewidth=3, label=ltext)
        p2, = ax.plot(tenergy, (kene_i - kene_i[0]) / enorm,
                      color=p1.get_color(), linestyle='--', linewidth=3)
        p3, = ax.plot(tenergy, (kene_e - kene_e[0]) / enorm,
                      color=p1.get_color(), linestyle='-.', linewidth=3)
    ylim = ax.get_ylim()
    ylen = ylim[1] - ylim[0]
    height1 = ((kene_i[-1] - kene_i[0]) / enorm - ylim[0]) / ylen + 0.05
    height2 = ((kene_e[-1] - kene_e[0]) / enorm - ylim[0]) / ylen - 0.1
    height3 = ((ene_magnetic[-1] - ene_magnetic[0]) / enorm - ylim[0]) / ylen - 0.1
    ax.text(0.5, height1, r'$\Delta K_i/\varepsilon_{Bx0}$', color='k', fontsize=20,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)
    ax.text(0.5, height2, r'$\Delta K_e/\varepsilon_{Bx0}$', color='k', fontsize=20,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)
    ax.text(0.5, height3, r'$\Delta \varepsilon_{B}/\varepsilon_{Bx0}$',
            color='k', fontsize=20,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)
    ax.tick_params(labelsize=16)
    ax.set_xlabel(r'$t\Omega_{ci}$', fontdict=FONT, fontsize=20)
    ax.legend(loc=3, prop={'size': 16}, ncol=1,
              shadow=False, fancybox=False, frameon=False)
    fdir = '../img/img_high_mime/'
    mkdir_p(fdir)
    fname = fdir + 'econv_' + bg_str + '.pdf'
    fig.savefig(fname)
    plt.show()


def internal_energy_evolution(bg):
    """Plot internal energy evolution for runs with the same guide field

    Args:
        bg: guide field strength
    """
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
        tfields = pic_info.tfields
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
        if mime == 400:
            tenergy += 12
            tfields += 12
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
                      linewidth=3, label=ltext)
        p2, = ax.plot(tfields, (iene_i - iene_i[0]) / enorm,
                      color=p1.get_color(), linestyle='--', linewidth=3)
        p3, = ax.plot(tfields, (iene_e - iene_e[0]) / enorm,
                      color=p1.get_color(), linestyle='-.', linewidth=3)
    ylim = ax.get_ylim()
    ylen = ylim[1] - ylim[0]
    height1 = ((iene_i[-1] - iene_i[0]) / enorm - ylim[0]) / ylen + 0.05
    height2 = ((iene_e[-1] - iene_e[0]) / enorm - ylim[0]) / ylen - 0.1
    height3 = ((ene_magnetic[-1] - ene_magnetic[0]) / enorm - ylim[0]) / ylen - 0.1
    ax.text(0.5, height1, r'$\Delta U_i/\varepsilon_{Bx0}$', color='k', fontsize=20,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)
    ax.text(0.5, height2, r'$\Delta U_e/\varepsilon_{Bx0}$', color='k', fontsize=20,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)
    ax.text(0.5, height3, r'$\Delta \varepsilon_{B}/\varepsilon_{Bx0}$',
            color='k', fontsize=20,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)
    ax.tick_params(labelsize=16)
    ax.set_xlabel(r'$t\Omega_{ci}$', fontdict=FONT, fontsize=20)
    ax.legend(loc=3, prop={'size': 16}, ncol=1,
              shadow=False, fancybox=False, frameon=False)
    fdir = '../img/img_high_mime/'
    mkdir_p(fdir)
    fname = fdir + 'internal_econv_' + bg_str + '.pdf'
    fig.savefig(fname)
    plt.show()


def energy_partition(bg):
    """Plot energy energy partition between ion and electrons

    Args:
        bg: guide field strength
    """
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
            tenergy += 12
        kene_e = pic_info.kene_e
        kene_i = pic_info.kene_i

        ene_ratio = div0((kene_i - kene_i[0]), (kene_e - kene_e[0]))

        ltext = r"$m_i/m_e=" + str(mime) + "$"
        p1, = ax.plot(tenergy, ene_ratio, linewidth=3, label=ltext)
    ax.set_ylim([1.0, 4.0])
    ax.tick_params(labelsize=16)
    ax.set_xlabel(r'$t\Omega_{ci}$', fontdict=FONT, fontsize=20)
    ax.set_ylabel(r'$\Delta K_i/\Delta K_e$', fontdict=FONT, fontsize=20)
    ax.legend(loc=1, prop={'size': 16}, ncol=1,
              shadow=False, fancybox=False, frameon=False)
    fdir = '../img/img_high_mime/'
    mkdir_p(fdir)
    fname = fdir + 'epartition_' + bg_str + '.pdf'
    fig.savefig(fname)
    plt.show()


def internal_energy_partition(bg):
    """Plot internal energy energy partition between ion and electrons

    Args:
        bg: guide field strength
    """
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
        tfields = pic_info.tfields
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
        if mime == 400:
            tenergy += 12
            tfields += 12
        kene_e = pic_info.kene_e
        kene_i = pic_info.kene_i

        ene_ratio = div0((iene_i[:, -1] - iene_i[0, -1]),
                         (iene_e[:, -1] - iene_e[0, -1]))

        ltext = r"$m_i/m_e=" + str(mime) + "$"
        p1, = ax.plot(tfields, ene_ratio, linewidth=3, label=ltext)
    ax.set_ylim([0.5, 3.0])
    ax.tick_params(labelsize=16)
    ax.set_xlabel(r'$t\Omega_{ci}$', fontdict=FONT, fontsize=20)
    ax.set_ylabel(r'$\Delta K_i/\Delta K_e$', fontdict=FONT, fontsize=20)
    ax.legend(loc=1, prop={'size': 16}, ncol=1,
              shadow=False, fancybox=False, frameon=False)
    fdir = '../img/img_high_mime/'
    mkdir_p(fdir)
    fname = fdir + 'internal_epartition_' + bg_str + '.pdf'
    fig.savefig(fname)
    plt.show()


def plot_jy(tframe, show_plot=True):
    """Plot out-of-plane current density for different runs

    Args:
        run_dir: PIC run directory
    """
    bgs = np.asarray([0.0, 0.2, 0.4, 0.8])
    mimes = np.asarray([25, 100, 400])
    dmins = np.asarray([-0.1, -0.067, -0.033])
    dmaxs = np.asarray([0.3, 0.2, 0.1])
    lmins = np.asarray([-0.1, -0.06, -0.03])
    lmaxs = np.asarray([0.3, 0.18, 0.09])
    fig = plt.figure(figsize=[15, 8])
    rect0 = [0.1, 0.77, 0.28, 0.18]
    rect = np.copy(rect0)
    hgap, vgap = 0.022, 0.022
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
            fname = pic_info.run_dir + "data/jy.gda"
            x, z, jy = read_2d_fields(pic_info, fname, **kwargs)
            sizes = [x[0], x[-1], z[0], z[-1]]
            print("Min and Max of Jy: %f %f" % (np.min(jy), np.max(jy)))
            fname = pic_info.run_dir + "data/Ay.gda"
            x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
            rect[0] = rect0[0] + imime * (hgap + rect0[2])
            ax = fig.add_axes(rect)
            p1 = ax.imshow(jy, vmin=dmins[imime], vmax=dmaxs[imime],
                           extent=sizes, cmap=plt.cm.jet, aspect='auto',
                           origin='lower', interpolation='bicubic')
            ax.contour(x, z, Ay, colors='k', linewidths=0.5)
            ax.tick_params(labelsize=16)
            if ibg < nbg - 1:
                ax.tick_params(axis='x', labelbottom='off')
            else:
                ax.set_xlabel(r'$x/d_i$', fontsize=20)
            if imime > 0:
                ax.tick_params(axis='y', labelleft='off')
            else:
                ax.set_ylabel(r'$z/d_i$', fontsize=20)

            if ibg == 0:
                title = r"$m_i/m_e=" + str(mime) + "$"
                plt.title(title, fontsize=24)
            if ibg == nbg - 1:
                rect_cbar = np.copy(rect)
                rect_cbar[1] = rect[1] - vgap * 5
                rect_cbar[3] = 0.02
                cbar_ax = fig.add_axes(rect_cbar)
                cbar = fig.colorbar(p1, cax=cbar_ax, orientation='horizontal')
                cbar.set_ticks(np.linspace(lmins[imime], lmaxs[imime], num=5))
                cbar.ax.tick_params(labelsize=16)
            if imime == 0:
                text = r"$B_g=" + str(bg) + "$"
                ax.text(-0.25, 0.5, text, color='k',
                        fontsize=24, rotation='vertical',
                        bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes)

    fdir = '../img/img_high_mime/jy/'
    mkdir_p(fdir)
    fname = fdir + 'jys_' + str(tframe) + '.jpg'
    fig.savefig(fname, dpi=200)
    if show_plot:
        plt.show()
    else:
        plt.close()


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
    fig1 = plt.figure(figsize=[7, 12])
    box1 = [0.15, 0.66, 0.8, 0.27]
    axs1 = []
    fig2 = plt.figure(figsize=[7, 12])
    box2 = [0.15, 0.66, 0.8, 0.27]
    axs2 = []
    fig3 = plt.figure(figsize=[7, 12])
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
        jagy_dote = ptensor_ene - jperp_dote
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
        ax.set_ylabel('Energization', fontdict=FONT, fontsize=20)
        text1 = r'$m_i/m_e=' + str(mime) + '$'
        ax.text(0.02, 0.9, text1, color='k', fontsize=20,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)

        ax = fig2.add_axes(box1)
        axs2.append(ax)
        ax.set_prop_cycle('color', COLORS)
        ax.plot(tfields, curv_drift_dote, linewidth=2, label='Curvature')
        # ax.plot(tfields, bulk_curv_dote, linewidth=2, label='Bulk Curvature')
        ax.plot(tfields, grad_drift_dote, linewidth=2, label='Gradient')
        ax.plot(tfields, magnetization_dote, linewidth=2, label='Magnetization')
        ax.plot(tfields, acc_drift_dote, linewidth=2, label='Inertial')
        label2 = (r'$\boldsymbol{j}_{' + species + '\perp}' +
                  r'\cdot\boldsymbol{E}_\perp$')
        ax.plot(tfields, eperp_ene, linewidth=2, label=label2)
        jdote_sum = (curv_drift_dote + grad_drift_dote +
                     magnetization_dote + jagy_dote + acc_drift_dote)
        # ax.plot(tfields, jdote_sum, linewidth=2)
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
        ax.set_ylabel('Energization', fontdict=FONT, fontsize=20)
        text1 = r'$m_i/m_e=' + str(mime) + '$'
        ax.text(0.02, 0.9, text1, color='k', fontsize=20,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)

        ax = fig3.add_axes(box1)
        axs3.append(ax)
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
        ax.tick_params(labelsize=16)
        ax.plot([0, 100], [0, 0], linestyle='--', color='k')
        ax.set_xlim([0, 100])
        ax.set_ylim(ylim3)
        ax.tick_params(labelsize=16)
        if mime == 400:
            ax.set_xlabel(r'$t\Omega_{ci}$', fontdict=FONT, fontsize=20)
        else:
            xticks = ax.get_xticks()
            xticks_labels = [str(int(x + tshift)) for x in xticks]
            ax.set_xticklabels(xticks_labels)
        ax.set_ylabel('Energization', fontdict=FONT, fontsize=20)
        text1 = r'$m_i/m_e=' + str(mime) + '$'
        ax.text(0.02, 0.9, text1, color='k', fontsize=20,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)

        box1[1] -= box1[3] + 0.03

    axs1[0].legend(loc='upper center', prop={'size': 20}, ncol=2,
                   bbox_to_anchor=(0.5, 1.3),
                   shadow=False, fancybox=False, frameon=False,
                   columnspacing=0.1)
    axs2[0].legend(loc='upper center', prop={'size': 20}, ncol=3,
                   bbox_to_anchor=(0.5, 1.3),
                   shadow=False, fancybox=False, frameon=False,
                   columnspacing=0.2, handletextpad=0.1)
    axs3[0].legend(loc='upper center', prop={'size': 20}, ncol=2,
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


def particle_energization2(run_name, species, tframe):
    """Particle-based energization

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


def fluid_energization_multi(species):
    """Plot fluid energization for multiple runs
    """
    bgs = np.asarray([0.0, 0.2, 0.4, 0.8])
    mimes = np.asarray([25, 100, 400])
    for ibg, bg in enumerate(bgs):
        for imime, mime in enumerate(mimes):
            fluid_energization(mime, bg, species, show_plot=False)


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
    parser.add_argument('--ene_evol', action="store_true", default=False,
                        help='whether plotting energy evolution')
    parser.add_argument('--ene_part', action="store_true", default=False,
                        help='whether plotting energy partition')
    parser.add_argument('--plot_jy', action="store_true", default=False,
                        help='whether plotting jy')
    parser.add_argument('--fluid_energization', action="store_true", default=False,
                        help='whether plotting fluid energization terms')
    parser.add_argument('--particle_energization', action="store_true", default=False,
                        help='whether plotting particle energization terms')
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
    parser.add_argument('--bulk_internal', action="store_true", default=False,
                        help='whether plotting bulk and internal energy')
    parser.add_argument('--iene_evol', action="store_true", default=False,
                        help='whether plotting internal energy evolution')
    parser.add_argument('--iene_part', action="store_true", default=False,
                        help='whether plotting internal energy partition')
    return parser.parse_args()


def analysis_single_frame(args):
    """Analysis for single time frame
    """
    if args.multi_runs:
        if args.calc_rrate:
            calc_rrate_multi(args.mime)
        if args.plot_rrate:
            plot_rrate_multi(args.mime)
        if args.fluid_energization:
            fluid_energization_multi(args.species)
    else:
        if args.ene_evol:
            energy_evolution(args.bg)
        if args.ene_part:
            energy_partition(args.bg)
        if args.plot_jy:
            plot_jy(args.tframe, show_plot=True)
        if args.fluid_energization:
            fluid_energization(args.mime, args.bg,
                               args.species, show_plot=True)
        if args.particle_energization:
            particle_energization2(args.run_name, args.species, args.tframe)
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
        if args.bulk_internal:
            bulk_internal_energy(args.bg, args.species, show_plot=True)
        if args.iene_evol:
            internal_energy_evolution(args.bg)
        if args.iene_part:
            internal_energy_partition(args.bg)


def process_input(args, tframe):
    """process one time frame"""
    if args.plot_jy:
        plot_jy(tframe, show_plot=False)


def analysis_multi_frames(args):
    """Analysis for multiple time frames
    """
    tframes = range(args.tstart, args.tend + 1)
    if args.time_loop:
        for tframe in tframes:
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
    else:
        ncores = multiprocessing.cpu_count()
        ncores = 8
        Parallel(n_jobs=ncores)(delayed(process_input)(args, tframe)
                                for tframe in tframes)


def main():
    """business logic for when running this module as the primary one!"""
    args = get_cmd_args()
    if args.multi_frames:
        analysis_multi_frames(args)
    else:
        analysis_single_frame(args)


if __name__ == "__main__":
    main()
