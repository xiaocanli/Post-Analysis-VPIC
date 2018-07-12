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

import pic_information
from contour_plots import read_2d_fields
from joblib import Parallel, delayed
from json_functions import read_data_from_json
from reconnection_rate import calc_reconnection_rate
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


def calc_rrate_multi(mime):
    """Calculate reconnection rate for multiple runs

    Args:
        mime: ion to electron mass ratio
    """
    base_dir = "/net/scratch3/xiaocanli/reconnection/mime" + str(mime) + "/"
    for bguide in ["00", "02", "04", "08"]:
        run_name = "mime" + str(mime) + "_beta002_bg" + str(bguide)
        if mime == 25:
            run_name_u = run_name + "_lx100"
        else:
            run_name_u = run_name
        run_dir = base_dir + run_name + "/"
        tfields, rrate = calc_reconnection_rate(run_dir)
        odir = "../data/rate/"
        mkdir_p(odir)
        fname = odir + "rrate_" + run_name_u + ".dat"
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
        if mime == 25:
            run_name_u = run_name + "_lx100"
        else:
            run_name_u = run_name
        fdir = "../data/rate/"
        fname = fdir + "rrate_" + run_name_u + ".dat"
        tfields, rrate = np.genfromtxt(fname)
        ltext = r"$B_g=" + str(bg) + "$"
        ax.plot(tfields, rrate, linewidth=2, label=ltext)
    ax.set_ylim([0, 0.13])
    beg = 18
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
        if mime == 25:
            run_name += "_lx100"
        picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        tenergy = pic_info.tenergy
        if mime == 400:
            tenergy += 13
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
        if mime == 25:
            run_name += "_lx100"
        picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
        pic_info = read_data_from_json(picinfo_fname)
        tenergy = pic_info.tenergy
        if mime == 400:
            tenergy += 13
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
            if mime == 25:
                run_name += "_lx100"
            picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
            pic_info = read_data_from_json(picinfo_fname)
            smime = math.sqrt(pic_info.mime)
            tframe_shift = (tframe + 13) if mime != 400 else tframe
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
    grad_drift_dote = fluid_ene[nframes+2:2*nframes+2]
    magnetization_dote = fluid_ene[2*nframes+2:3*nframes+2]
    comp_ene = fluid_ene[3*nframes+2:4*nframes+2]
    shear_ene = fluid_ene[4*nframes+2:5*nframes+2]
    ptensor_ene = fluid_ene[5*nframes+2:6*nframes+2]
    pgyro_ene = fluid_ene[6*nframes+2:7*nframes+2]

    fname = "../data/fluid_energization/" + run_name + "/"
    fname += "para_perp_acc_" + species + '.gda'
    fluid_ene = np.fromfile(fname, dtype=np.float32)
    nvar = int(fluid_ene[0])
    nframes = int(fluid_ene[1])
    acc_drift_dote = fluid_ene[2:nframes+2]
    epara_ene = fluid_ene[nframes+2:2*nframes+2]
    eperp_ene = fluid_ene[2*nframes+2:3*nframes+2]
    acc_drift_dote[-1] = acc_drift_dote[-2]

    jperp_dote = curv_drift_dote + grad_drift_dote + magnetization_dote
    jagy_dote = ptensor_ene - jperp_dote
    if species == 'e':
        dkene = pic_info.dkene_e
    else:
        dkene = pic_info.dkene_i

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
    ax.plot(tfields, jagy_dote, linewidth=2, label=label4)
    # label5 = (r'$\boldsymbol{j}_{' + species + '\parallel}' +
    #           r'\cdot\boldsymbol{E}_\parallel + ' +
    #           r'\boldsymbol{j}_{' + species + '\perp}' +
    #           r'\cdot\boldsymbol{E}_\perp$')
    # ax.plot(tfields, epara_ene + eperp_ene, linewidth=2, label=label5)
    # label6 = r'$dK_' + species + '/dt$'
    # ax.plot(tenergy, dkene, linewidth=2, label=label6)
    ax.legend(loc='upper center', prop={'size': 16}, ncol=2,
              bbox_to_anchor=(0.5, 1.28),
              shadow=False, fancybox=False, frameon=False)
    ax.set_xlim([0, np.max(tfields)])
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
    ax.plot(tfields, grad_drift_dote, linewidth=2, label='Gradient')
    ax.plot(tfields, magnetization_dote, linewidth=2, label='Magnetization')
    ax.plot(tfields, acc_drift_dote, linewidth=2, label='Polarization')
    label2 = (r'$\boldsymbol{j}_{' + species + '\perp}' +
              r'\cdot\boldsymbol{E}_\perp$')
    ax.plot(tfields, eperp_ene, linewidth=2, label=label2)
    # jdote_sum = (curv_drift_dote + grad_drift_dote +
    #              magnetization_dote + jagy_dote + acc_drift_dote)
    # ax.plot(tfields, jdote_sum, linewidth=2)
    ax.legend(loc='upper center', prop={'size': 16}, ncol=3,
              bbox_to_anchor=(0.5, 1.28),
              shadow=False, fancybox=False, frameon=False)
    ax.set_xlim([0, np.max(tfields)])
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
    ax.tick_params(labelsize=16)
    ax.set_xlabel(r'$t\Omega_{ci}$', fontdict=FONT, fontsize=20)
    ax.set_ylabel('Energization', fontdict=FONT, fontsize=20)

    fdir = '../img/img_high_mime/fluid_energization/'
    mkdir_p(fdir)
    fname = fdir + 'fluid_comp_shear_' + run_name + '_' + species + '.pdf'
    fig.savefig(fname)

    fig = plt.figure(figsize=[7, 5])
    ax = fig.add_axes([0.15, 0.15, 0.8, 0.65])
    COLORS = palettable.colorbrewer.qualitative.Set1_9.mpl_colors
    ax.set_prop_cycle('color', COLORS)
    # ax.plot(tfields, comp_ene, linewidth=2, label='Compression')
    # ax.plot(tfields, shear_ene, linewidth=2, label='Shear')
    ax.plot(tfields, comp_ene + shear_ene,
            linewidth=2, label='Compression + shear')
    jdote_drifts = curv_drift_dote + grad_drift_dote + magnetization_dote
    ax.plot(tfields, jdote_drifts, linewidth=2, label='Drifts')
    # ax.plot(tfields, ptensor_ene, linewidth=2, label='Ptensor')
    # ax.plot(tfields, pgyro_ene, linewidth=2, label='Pgyro')
    ax.legend(loc='upper center', prop={'size': 16}, ncol=2,
              bbox_to_anchor=(0.5, 1.28),
              shadow=False, fancybox=False, frameon=False)
    ax.set_xlim([0, np.max(tfields)])
    ax.tick_params(labelsize=16)
    ax.set_xlabel(r'$t\Omega_{ci}$', fontdict=FONT, fontsize=20)
    ax.set_ylabel('Energization', fontdict=FONT, fontsize=20)

    if show_plot:
        plt.show()
    else:
        plt.close('all')


def particle_energization(run_name, species, tframe):
    """Particle-based enerigzation

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
    fname = fpath + "para_perp_" + species + "_" + str(tstep) + ".gda"
    fdata = np.fromfile(fname, dtype=np.float32)
    nbins = int(fdata[0])
    nvar = int(fdata[1])
    ebins = fdata[2:nbins+2]
    fbins = fdata[nbins+2:].reshape((nvar, nbins))
    ndist = fbins[0, :]
    para_ptl = fbins[1, :]
    perp_ptl = fbins[2, :]

    fname = fpath + "comp_shear_" + species + "_" + str(tstep) + ".gda"
    fdata = np.fromfile(fname, dtype=np.float32)
    nbins = int(fdata[0])
    nvar = int(fdata[1])
    ebins = fdata[2:nbins+2]
    fbins = fdata[nbins+2:].reshape((nvar, nbins))
    comp_ptl = fbins[1, :]
    shear_ptl = fbins[2, :]

    fname = fpath + "curv_grad_para_" + species + "_" + str(tstep) + ".gda"
    fdata = np.fromfile(fname, dtype=np.float32)
    nbins = int(fdata[0])
    nvar = int(fdata[1])
    ebins = fdata[2:nbins+2]
    curv_grad_para = fdata[nbins+2:].reshape((nvar, nbins))

    fname = fpath + "magnetic_moment_" + species + "_" + str(tstep) + ".gda"
    fdata = np.fromfile(fname, dtype=np.float32)
    nbins = int(fdata[0])
    nvar = int(fdata[1])
    ebins = fdata[2:nbins+2]
    magnetic_moment = fdata[nbins+2:].reshape((nvar, nbins))

    fname = fpath + "polarization_initial_time_" + species + "_" + str(tstep) + ".gda"
    fdata = np.fromfile(fname, dtype=np.float32)
    nbins = int(fdata[0])
    nvar = int(fdata[1])
    ebins = fdata[2:nbins+2]
    polar_initial_time = fdata[nbins+2:].reshape((nvar, nbins))

    fname = fpath + "polarization_initial_spatial_" + species + "_" + str(tstep) + ".gda"
    fdata = np.fromfile(fname, dtype=np.float32)
    nbins = int(fdata[0])
    nvar = int(fdata[1])
    ebins = fdata[2:nbins+2]
    polar_initial_spatial = fdata[nbins+2:].reshape((nvar, nbins))

    fname = "../data/fluid_energization/" + run_name + "/"
    fname += "emf_ptensor_" + species + '.gda'
    fluid_ene = np.fromfile(fname, dtype=np.float32)
    nvar = int(fluid_ene[0])
    nframes = int(fluid_ene[1])
    jcurv_dote = fluid_ene[2:nframes+2]
    jgrad_dote = fluid_ene[nframes+2:2*nframes+2]
    jmag_dote = fluid_ene[2*nframes+2:3*nframes+2]
    comp_ene = fluid_ene[3*nframes+2:4*nframes+2]
    shear_ene = fluid_ene[4*nframes+2:5*nframes+2]
    ptensor_ene = fluid_ene[5*nframes+2:6*nframes+2]

    fname = "../data/fluid_energization/" + run_name + "/"
    fname += "para_perp_acc_" + species + '.gda'
    fluid_ene = np.fromfile(fname, dtype=np.float32)
    nvar = int(fluid_ene[0])
    nframes = int(fluid_ene[1])
    acc_drift_dote = fluid_ene[2:nframes+2]
    epara_ene = fluid_ene[nframes+2:2*nframes+2]
    eperp_ene = fluid_ene[2*nframes+2:3*nframes+2]
    acc_drift_dote[-1] = acc_drift_dote[-2]

    print("Parallel electric field: %f" % np.sum(para_ptl))
    print("Perpendicular electric field: %f" % np.sum(perp_ptl))
    print("Compression: %f" % np.sum(comp_ptl))
    print("Shear: %f" % np.sum(shear_ptl))
    print("Parallel drift: %f" % np.sum(curv_grad_para[3, :]))
    print("Conservation of mu: %f" % np.sum(magnetic_moment[1, :]))
    print("Polarization drift (time): %f" % np.sum(polar_initial_time[1, :]))
    print("Polarization drift (spatial): %f" % np.sum(polar_initial_spatial[1, :]))
    print("Initial drift (time): %f" % np.sum(polar_initial_time[2, :]))
    print("Initial drift (spatial): %f" % np.sum(polar_initial_spatial[2, :]))
    print("Curvature drift: %f" % np.sum(curv_grad_para[1, :]))
    print("Gradient drift: %f" % np.sum(curv_grad_para[2, :]))
    print("Curvature drift (fluid): %f" % jcurv_dote[tframe_fluid])
    print("Gradient drift (fluid): %f" % jgrad_dote[tframe_fluid])
    print("Parallel electric field (fluid): %f" % epara_ene[tframe_fluid])
    print("Perpendicular electric field (fluid): %f" % eperp_ene[tframe_fluid])
    print("Fluid acceleration (fluid): %f" % acc_drift_dote[tframe_fluid])
    print("Magnetization (fluid): %f" % jmag_dote[tframe_fluid])
    print("Compression (fluid): %f" % comp_ene[tframe_fluid])
    print("Shear (fluid): %f" % shear_ene[tframe_fluid])
    print("ptensor (fluid): %f" % ptensor_ene[tframe_fluid])

    para_ptl = div0(para_ptl, ndist * ebins)
    perp_ptl = div0(perp_ptl, ndist * ebins)
    comp_ptl = div0(comp_ptl, ndist * ebins)
    shear_ptl = div0(shear_ptl, ndist * ebins)
    curv_grad_para[1:, :] = div0(curv_grad_para[1:, :],
                                 curv_grad_para[0, :] * ebins)
    magnetic_moment[1, :] = div0(magnetic_moment[1, :],
                                 magnetic_moment[0, :] * ebins)
    polar_initial_time[1:, :] = div0(polar_initial_time[1:, :],
                                     polar_initial_time[0, :] * ebins)
    polar_initial_spatial[1:, :] = div0(polar_initial_spatial[1:, :],
                                        polar_initial_spatial[0, :] * ebins)

    if species == 'i':
        para_ptl /= pic_info.mime
        perp_ptl /= pic_info.mime
        comp_ptl /= pic_info.mime
        shear_ptl /= pic_info.mime
        curv_grad_para[1:, :] /= pic_info.mime
        magnetic_moment[1, :] /= pic_info.mime
        polar_initial_time[1:, :] /= pic_info.mime
        polar_initial_spatial[1:, :] /= pic_info.mime

    # normalized with thermal energy
    if species == 'e':
        vth = pic_info.vthe
    else:
        vth = pic_info.vthi
    gama = 1.0 / math.sqrt(1.0 - 3 * vth**2)
    eth = gama - 1.0

    ebins /= eth
    # A factor of mime when accumulating the distributions
    if species == 'i':
        ebins *= pic_info.mime

    colors_Set1_9 = palettable.colorbrewer.qualitative.Set1_9.mpl_colors
    colors = colors_Set1_9[:5] + colors_Set1_9[6:] # remove yellow color
    xs, ys = 0.17, 0.15
    w1, h1 = 0.8, 0.8
    fig = plt.figure(figsize=[7, 5])
    ax1 = fig.add_axes([xs, ys, w1, h1])
    ax1.set_prop_cycle('color', colors)
    ax1.semilogx(ebins, para_ptl, linewidth=2, label="para")
    ax1.semilogx(ebins, perp_ptl, linewidth=2, label="perp")
    # ax1.semilogx(ebins, comp_ptl, linewidth=2, label="comp")
    # ax1.semilogx(ebins, shear_ptl, linewidth=2, label="shear")
    ax1.semilogx(ebins, (comp_ptl + shear_ptl), color='k',
                 linewidth=2, label="comp + shear")
    drifts_ene = (curv_grad_para[1, :] + curv_grad_para[2, :] +
                  curv_grad_para[3, :] + magnetic_moment[1, :] +
                  polar_initial_time[1, :] + polar_initial_spatial[1, :] +
                  polar_initial_time[2, :] + polar_initial_spatial[2, :])
    # ax1.semilogx(ebins, drifts_ene, linewidth=2, label="Curvature + Gradient + mu")
    # ax1.semilogx(ebins, curv_grad_para[1, :], linewidth=2,
    #              label="Curvature")
    # ax1.semilogx(ebins, curv_grad_para[2, :], linewidth=2,
    #              label="Gradient")
    # ax1.semilogx(ebins, curv_grad_para[3, :], linewidth=2,
    #              label="Parallel Drift")
    # ax1.semilogx(ebins, magnetic_moment[1, :], linewidth=2,
    #              label=r"$\mu$")
    initial_drift = (polar_initial_time[2, :] + polar_initial_spatial[2, :] +
                     curv_grad_para[1, :])
    ax1.semilogx(ebins, initial_drift, linewidth=2, label="Initial")
    polar_drift = polar_initial_time[1, :] + polar_initial_spatial[1, :]
    ax1.semilogx(ebins, polar_drift, linewidth=2, label="Polar")
    # ax1.semilogx(ebins, polar_initial_time[1, :], linewidth=2, label="Polar T")
    # ax1.semilogx(ebins, polar_initial_spatial[1, :], linewidth=2, label="Polar S")
    # ax1.semilogx(ebins, polar_initial_time[2, :], linewidth=2, label="Initial T")
    # ax1.semilogx(ebins, polar_initial_spatial[2, :], linewidth=2, label="Initial S")
    ax1.plot(ax1.get_xlim(), [0, 0], linestyle='--', color='k')
    leg = ax1.legend(loc=2, prop={'size': 20}, ncol=1,
                     shadow=False, fancybox=False, frameon=False)
    if species == 'e':
        ax1.set_xlim([1E-1, 200])
        ax1.set_ylim([-0.003, 0.003])
    else:
        ax1.set_xlim([1E0, 200])
        ax1.set_ylim([-0.05, 0.05])
    ax1.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$', fontdict=FONT, fontsize=20)
    ax1.set_ylabel('Acceleration Rate', fontdict=FONT, fontsize=20)
    ax1.tick_params(labelsize=16)
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
            particle_energization(args.run_name, args.species, args.tframe)


def process_input(args, tframe):
    """process one time frame"""
    if args.plot_jy:
        plot_jy(tframe, show_plot=False)


def analysis_multi_frames(args):
    """Analysis for multiple time frames
    """
    tframes = range(args.tstart, args.tend + 1)
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
