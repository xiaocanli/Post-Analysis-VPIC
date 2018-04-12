"""
Functions to analyze particle-based energization
"""
import argparse
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import palettable
from json_functions import read_data_from_json
from shell_functions import mkdir_p

mpl.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

FONT = {'family': 'serif',
        'color': 'black',
        'weight': 'normal',
        'size': 24}

def test_particle_energization():
    """Test particle-based enerigzation
    """
    fpath = "/net/scratch3/xiaocanli/pic_analysis/data/particle_interp/"
    species = 'e'
    tstep = 41360
    tinterval = 1034 
    tframe = tstep // tinterval
    fname = fpath + "para_perp_comp_shear_" + species + "_" + str(tstep) + ".gda"
    # fname = fpath + "particle_energization_e_132400.gda"
    fdata = np.fromfile(fname, dtype=np.float32)
    nbins = int(fdata[0])
    nvar = int(fdata[1])
    ebins = fdata[2:nbins+2]
    fbins = fdata[nbins+2:].reshape((nvar, nbins))

    fname = fpath + "curv_grad_para_" + species + "_" + str(tstep) + ".gda"
    # fname = fpath + "particle_energization_e_132400.gda"
    fdata = np.fromfile(fname, dtype=np.float32)
    nbins = int(fdata[0])
    nvar = int(fdata[1])
    ebins = fdata[2:nbins+2]
    curv_grad_para = fdata[nbins+2:].reshape((nvar, nbins))

    vth = 0.1
    gama = 1.0 / math.sqrt(1.0 - 3 * vth**2)
    eth = gama - 1.0
    ebins /= eth

    fname = "../data/fluid_energization/"
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

    print("Parallel electric field: %f" % np.sum(fbins[0, :]))
    print("Perpendicular electric field: %f" % np.sum(fbins[1, :]))
    print("Compression: %f" % np.sum(fbins[2, :]))
    print("Shear: %f" % np.sum(fbins[3, :]))
    print("Curvature drift: %f" % np.sum(curv_grad_para[0, :]))
    print("Gradient drift: %f" % np.sum(curv_grad_para[1, :]))
    print("Curvature drift (fluid): %f" % jcurv_dote[tframe])
    print("Gradient drift (fluid): %f" % jgrad_dote[tframe])
    print("Parallel electric field (fluid): %f" % jpara_dote[tframe])
    print("Perpendicular electric field (fluid): %f" % jperp_dote[tframe])

    colors_Set1_9 = palettable.colorbrewer.qualitative.Set1_9.mpl_colors
    colors = colors_Set1_9[:5] + colors_Set1_9[6:] # remove yellow color
    xs, ys = 0.15, 0.15
    w1, h1 = 0.8, 0.8
    fig = plt.figure(figsize=[7, 5])
    ax1 = fig.add_axes([xs, ys, w1, h1])
    ax1.set_prop_cycle('color', colors)
    ax1.semilogx(ebins, fbins[0, :], linewidth=2, label="para")
    ax1.semilogx(ebins, fbins[1, :], linewidth=2, label="perp")
    ax1.semilogx(ebins, fbins[2, :], linewidth=2, label="comp")
    ax1.semilogx(ebins, fbins[3, :], linewidth=2, label="shear")
    ax1.semilogx(ebins, fbins[2, :] + fbins[3, :], color='k', linewidth=2)
    ax1.semilogx(ebins, (curv_grad_para[0, :] + curv_grad_para[1, :]), linewidth=2)
    # ax1.semilogx(ebins, curv_grad_para[0, :], linewidth=2)
    # ax1.semilogx(ebins, curv_grad_para[1, :], linewidth=2)
    leg = ax1.legend(loc=1, prop={'size': 20}, ncol=1,
                     shadow=False, fancybox=False, frameon=False)
    if species == 'e':
        ax1.set_xlim([1E-1, 500])
    else:
        ax1.set_xlim([1E-3, 5])
    ax1.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$', fontdict=FONT, fontsize=20)
    ax1.set_ylabel('Energization', fontdict=FONT, fontsize=20)
    ax1.tick_params(labelsize=16)
    plt.show()


def test_fluid_energization():
    """Test fluid-based energization
    """
    run_name = "mime25_beta002_guide00_frequent_dump"
    species = "e"
    jdote_name = '../data/jdote_data/jdote_' + run_name + '_' + species + '.json'
    jdote = read_data_from_json(jdote_name)
    jcurv_dote = jdote.jcpara_dote
    jgrad_dote = jdote.jgrad_dote
    jmag_dote = jdote.jmag_dote
    jpara_dote = jdote.jqnupara_dote
    jperp_dote = jdote.jqnuperp_dote

    jdote_name = '../data/jpolar_dote/jpolar_dote_' + run_name + '_' + species + '.dat'
    jpolar_dote = np.fromfile(jdote_name, np.float32)

    fname = "../data/fluid_energization/"
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

    fname = "../data/fluid_energization/"
    fname += "para_perp_acc_" + species + '.gda'
    fluid_ene = np.fromfile(fname, dtype=np.float32)
    nvar = int(fluid_ene[0])
    nframes = int(fluid_ene[1])
    acc_drift_dote = fluid_ene[2:nframes+2]
    para_ene = fluid_ene[2*nframes+2:3*nframes+2]
    perp_ene = fluid_ene[3*nframes+2:4*nframes+2]
    acc_drift_dote = fluid_ene[2:nframes+2]
    acc_drift_dote[-1] = acc_drift_dote[-2]

    fname = "../data/fluid_energization/"
    fname += "para_perp_acc_" + species + '.gda'
    fluid_ene = np.fromfile(fname, dtype=np.float32)
    nvar = int(fluid_ene[0])
    nframes = int(fluid_ene[1])
    acc_drift_dote = fluid_ene[2:nframes+2]
    para_ene = fluid_ene[nframes+2:2*nframes+2]
    perp_ene = fluid_ene[2*nframes+2:3*nframes+2]
    acc_drift_dote[-1] = acc_drift_dote[-2]
    # plt.plot(jcurv_dote, linewidth=2)
    # plt.plot(jgrad_dote, linewidth=2)
    # plt.plot(jmag_dote, linewidth=2)
    # plt.plot(curv_drift_dote)
    # plt.plot(grad_drift_dote)
    # plt.plot(-magnetization_dote)
    jperp_dote_n = curv_drift_dote + grad_drift_dote + magnetization_dote
    jperp_dote_n += acc_drift_dote
    plt.plot(jpara_dote + jperp_dote, linewidth=2)
    # plt.plot(jperp_dote, linewidth=2)
    # plt.plot(jpara_dote, linewidth=2)
    # plt.plot(jdote.jpolar_dote, linewidth=2)
    # plt.plot(jperp_dote_n)
    plt.plot(para_ene + perp_ene)
    # plt.plot(para_ene)
    # plt.plot(perp_ene)
    # plt.plot(ptensor_ene + acc_drift_dote)
    # plt.plot(comp_ene + shear_ene)
    # plt.plot(acc_drift_dote)
    plt.show()


if __name__ == "__main__":
    # test_particle_energization()
    test_fluid_energization()
