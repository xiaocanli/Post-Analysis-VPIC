"""
Analysis procedures for energy conversion.
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

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

font = {'family' : 'serif',
        #'color'  : 'darkred',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 24,
        }

def plot_energy_evolution(pic_info):
    """Plot energy time evolution.

    Plot time evolution of magnetic, electric, electron and ion kinetic
    energies. 
    """
    tenergy = pic_info.tenergy
    ene_electric = pic_info.ene_electric
    ene_magnetic = pic_info.ene_magnetic
    kene_e = pic_info.kene_e
    kene_i = pic_info.kene_i
    ene_bx = pic_info.ene_bx
    ene_by = pic_info.ene_by
    ene_bz = pic_info.ene_bz

    enorm = ene_bx[0]

    fig = plt.figure(figsize=[3.5,2.5])
    ax = fig.add_axes([0.22, 0.22, 0.75, 0.73])
    p1, = ax.plot(tenergy, ene_bx/enorm, linewidth=2, 
            label=r'$B_x^2(t)$', color='b')
    p2, = ax.plot(tenergy, kene_i/enorm, linewidth=2, 
            color='g', label=r'$\Delta K_i$')
    p3, = ax.plot(tenergy, kene_e/enorm, linewidth=2, 
            color='r', label=r'$\Delta K_e$')
    p4, = ax.plot(tenergy, 100*ene_electric/enorm, linewidth=2, 
            color='m', label='$100E^2$')
    ax.set_xlim([0, 1190])
    ax.set_ylim([0, 1.05])

    ax.set_xlabel(r'$t\Omega_{ci}$', fontdict=font, fontsize=20)
    ax.set_ylabel(r'Energy/$\varepsilon_{bx}(0)$', fontdict=font, fontsize=20)

    ax.text(500, 0.85, r'$\varepsilon_{bx}(t)$', color='blue', fontsize=24)
    ax.text(500, 0.65, r'$100\varepsilon_e$', color='m', fontsize=24)
    ax.text(900, 0.85, r'$K_e$', color='red', fontsize=24)
    ax.text(900, 0.65, r'$K_i$', color='green', fontsize=24)
   
    plt.tick_params(labelsize=16)
    #plt.savefig('pic_ene.eps')

    print 'The final fraction of ebx: ', ene_bx[-1]/enorm
    print 'The ratio of energy gain to the initial ebx: ', \
            (kene_e[-1]-kene_e[0])/enorm, (kene_i[-1]-kene_i[0])/enorm 
    print 'The ratio of the initial kene_e and kene_i to the initial ebx: ',\
            kene_e[0]/enorm, kene_i[0]/enorm
    print 'The ratio of the final kene_e and kene_i to the initial ebx: ',\
            kene_e[-1]/enorm, kene_i[-1]/enorm
    init_ene = pic_info.ene_electric[0] + pic_info.ene_magnetic[0] + \
               kene_e[0] + kene_i[0]
    final_ene = pic_info.ene_electric[-1] + pic_info.ene_magnetic[-1] + \
               kene_e[-1] + kene_i[-1]
    print 'Energy conservation: ', final_ene / init_ene
    plt.show()

def plot_particle_energy_gain():
    """Plot particle energy gain for cases with different beta.

    """
    # beta_e = 0.005
    pic_info = pic_information.get_pic_info(
            '../../mime25-beta00025-guide0-200-100-nppc200')
    tenergy1 = pic_info.tenergy
    kene_e1 = pic_info.kene_e
    kene_i1 = pic_info.kene_i

    # beta_e = 0.02
    pic_info = pic_information.get_pic_info(
            '../../mime25-beta001-guide0-200-100-nppc400')
    tenergy2 = pic_info.tenergy
    kene_e2 = pic_info.kene_e
    kene_i2 = pic_info.kene_i

    # beta_e = 0.06
    pic_info = pic_information.get_pic_info(
            '../../mime25-beta003-guide0-200-100-nppc200')
    tenergy3 = pic_info.tenergy
    kene_e3 = pic_info.kene_e
    kene_i3 = pic_info.kene_i

    # beta_e = 0.2
    pic_info = pic_information.get_pic_info(
            '../../mime25-beta01-guide0-200-100-nppc200')
    tenergy4 = pic_info.tenergy
    kene_e4 = pic_info.kene_e
    kene_i4 = pic_info.kene_i

    # Estimate the energy gain for beta_e = 0.0072 using beta_e = 0.005
    kene_e12 = kene_e1[0] + (kene_e1-kene_e1[0])*0.005/0.0072
    kene_i12 = kene_i1[0] + (kene_i1-kene_i1[0])*0.005/0.0072

    print 'The ratio of electron energy gain to its initial energy: '
    print '    beta_e = 0.0072, 0.02, 0.06, 0.2: ', \
            (kene_e12[-1]-kene_e12[0])/kene_e12[0], \
            (kene_e2[-1]-kene_e2[0])/kene_e2[0], \
            (kene_e3[-1]-kene_e3[0])/kene_e3[0], \
            (kene_e4[-1]-kene_e4[0])/kene_e4[0]
    # Electrons
    fig = plt.figure(figsize=[3.5,2.5])
    ax = fig.add_axes([0.22, 0.22, 0.75, 0.73])
    ax.plot(tenergy1, (kene_e12-kene_e12[0])/kene_e12[0], 'b', linewidth=2)
    ax.plot(tenergy2, (kene_e2-kene_e2[0])/kene_e2[0], 'r', linewidth=2)
    ax.plot(tenergy3, (kene_e3-kene_e3[0])/kene_e3[0], 'orange', linewidth=2)
    ax.plot(tenergy4, (kene_e4-kene_e4[0])/kene_e4[0], 'g', linewidth=2)
    ax.set_xlim([0, 1190])
    #ax.set_ylim([0, 1.05])

    #plt.title('Energy spectrum', fontdict=font)
    ax.set_xlabel(r'$t\Omega_{ci}$', fontdict=font, fontsize=20)
    ax.set_ylabel(r'$\Delta K_e/K_e(0)$', fontdict=font, fontsize=20)
    plt.tick_params(labelsize=16)
   
    ax.text(680, 8.8, r'$\beta_e=0.007$', color='blue', 
            rotation=5, fontsize=16)
    ax.text(680, 5, r'$\beta_e=0.02$', color='red', 
            rotation=4, fontsize=16)
    ax.text(680, 2.1, r'$\beta_e=0.06$', color='orange', 
            rotation=0, fontsize=16)
    ax.text(680, -1.5, r'$\beta_e=0.2$', color='green', 
            rotation=0, fontsize=16)
    # Ions
    fig = plt.figure(figsize=[3.5,2.5])
    ax = fig.add_axes([0.22, 0.22, 0.75, 0.73])
    ax.plot(tenergy1, (kene_i12-kene_i12[0])/kene_i12[0], 'b', linewidth=2)
    ax.plot(tenergy2, (kene_i2-kene_i2[0])/kene_i2[0], 'r', linewidth=2)
    ax.plot(tenergy3, (kene_i3-kene_i3[0])/kene_i3[0], 'orange', linewidth=2)
    ax.plot(tenergy4, (kene_i4-kene_i4[0])/kene_i4[0], 'g', linewidth=2)
    ax.set_xlim([0, 1190])
    ax.set_ylim([-5, 30])

    #plt.title('Energy spectrum', fontdict=font)
    ax.set_xlabel(r'$t\Omega_{ci}$', fontdict=font, fontsize=20)
    ax.set_ylabel(r'$\Delta K_i/K_i(0)$', fontdict=font, fontsize=20)
    plt.tick_params(labelsize=16)
   
    ax.text(680, 22, r'$\beta_e=0.007$', color='blue', 
            rotation=0, fontsize=16)
    ax.text(680, 9, r'$\beta_e=0.02$', color='red', 
            rotation=0, fontsize=16)
    ax.text(680, 3, r'$\beta_e=0.06$', color='orange', 
            rotation=0, fontsize=16)
    ax.text(680, -4, r'$\beta_e=0.2$', color='green', 
            rotation=0, fontsize=16)
    plt.show()

def read_jdote_data(species):
    """Read j.E data.

    Args:
        species: particle species. 'e' for electron, 'h' for ion.
    """
    pic_info = pic_information.get_pic_info('../../')
    ntf = pic_info.ntf
    dt_fields = pic_info.dt_fields
    dtf_wpe = dt_fields * pic_info.dtwpe / pic_info.dtwci
    fname = "../data/jdote00_" + species + ".gda"
    fh = open(fname, 'r')
    data = fh.read()
    fh.close()
    njote = 16  # different kind of data.
    jdote_data = np.zeros((ntf, njote))
    index_start = 0
    index_end = 4
    for current_time in range(ntf):
        for i in range(njote):
            jdote_data[current_time, i], = \
                    struct.unpack('f', data[index_start:index_end])
            index_start = index_end
            index_end += 4
    jcpara_dote = jdote_data[:, 0]
    jcperp_dote = jdote_data[:, 1]
    jmag_dote   = jdote_data[:, 2]
    jgrad_dote  = jdote_data[:, 3]
    jdiagm_dote = jdote_data[:, 4]
    jpolar_dote = jdote_data[:, 5]
    jexb_dote   = jdote_data[:, 6]
    jpara_dote  = jdote_data[:, 7]
    jperp_dote  = jdote_data[:, 8]
    jperp1_dote = jdote_data[:, 9]
    jperp2_dote = jdote_data[:, 10]
    jqnupara_dote = jdote_data[:, 11]
    jqnuperp_dote = jdote_data[:, 12]
    jagy_dote     = jdote_data[:, 14]
    jtot_dote     = jdote_data[:, 13]
    jdivu_dote     = jdote_data[:, 15]
    jcpara_dote_int = cumulate_with_time(jcpara_dote, dtf_wpe, ntf)
    jcperp_dote_int = cumulate_with_time(jcperp_dote, dtf_wpe, ntf)
    jmag_dote_int   = cumulate_with_time(jmag_dote, dtf_wpe, ntf)
    jgrad_dote_int  = cumulate_with_time(jgrad_dote, dtf_wpe, ntf)
    jdiagm_dote_int = cumulate_with_time(jdiagm_dote, dtf_wpe, ntf)
    jpolar_dote_int = cumulate_with_time(jpolar_dote, dtf_wpe, ntf)
    jexb_dote_int   = cumulate_with_time(jexb_dote, dtf_wpe, ntf)
    jpara_dote_int  = cumulate_with_time(jpara_dote, dtf_wpe, ntf)
    jperp_dote_int  = cumulate_with_time(jperp_dote, dtf_wpe, ntf)
    jperp1_dote_int = cumulate_with_time(jperp1_dote, dtf_wpe, ntf)
    jperp2_dote_int = cumulate_with_time(jperp2_dote, dtf_wpe, ntf)
    jqnupara_dote_int = cumulate_with_time(jqnupara_dote, dtf_wpe, ntf)
    jqnuperp_dote_int = cumulate_with_time(jqnuperp_dote, dtf_wpe, ntf)
    jagy_dote_int     = cumulate_with_time(jagy_dote, dtf_wpe, ntf)
    jtot_dote_int     = cumulate_with_time(jtot_dote, dtf_wpe, ntf)
    jdivu_dote_int     = cumulate_with_time(jdivu_dote, dtf_wpe, ntf)
    jdote_collection = collections.namedtuple('jdote_collection',
            ['jcpara_dote', 'jcperp_dote', 'jmag_dote', 'jgrad_dote',
                'jdiagm_dote', 'jpolar_dote', 'jexb_dote', 'jpara_dote',
                'jperp_dote', 'jperp1_dote', 'jperp2_dote', 'jqnupara_dote',
                'jqnuperp_dote', 'jagy_dote', 'jtot_dote', 'jdivu_dote',
                'jcpara_dote_int', 'jcperp_dote_int', 'jmag_dote_int',
                'jgrad_dote_int', 'jdiagm_dote_int', 'jpolar_dote_int',
                'jexb_dote_int', 'jpara_dote_int', 'jperp_dote_int',
                'jperp1_dote_int', 'jperp2_dote_int', 'jqnupara_dote_int',
                'jqnuperp_dote_int', 'jagy_dote_int', 'jtot_dote_int',
                'jdivu_dote_int'])
    jdote = jdote_collection(jcpara_dote, jcperp_dote, jmag_dote, jgrad_dote,
            jdiagm_dote, jpolar_dote, jexb_dote, jpara_dote, jperp_dote,
            jperp1_dote, jperp2_dote, jqnupara_dote, jqnuperp_dote,
            jagy_dote, jtot_dote, jdivu_dote,
            jcpara_dote_int, jcperp_dote_int, jmag_dote_int, jgrad_dote_int,
            jdiagm_dote_int, jpolar_dote_int, jexb_dote_int, jpara_dote_int, 
            jperp_dote_int, jperp1_dote_int, jperp2_dote_int, 
            jqnupara_dote_int, jqnuperp_dote_int, jagy_dote_int,
            jtot_dote_int, jdivu_dote_int)
    return jdote

def cumulate_with_time(f, dt, ntf):
    """
    Args:
        f: the time evolution of one field.
        dt: the time step.
        ntf: number of time frames for fields.
    """
    f_cumulative = np.zeros(ntf) # originally 0
    for i in range(1,ntf):
        f_cumulative[i] = f_cumulative[i-1] + 0.5*(f[i]+f[i-1])*dt
    return f_cumulative

def plot_jdotes_evolution(species):
    """
    Plot the time evolution of the energy conversion due to different currents.

    Args:
        species: particle species. 'e' for electron, 'h' for ion.
    """
    jdote = read_jdote_data(species)
    pic_info = pic_information.get_pic_info('../../')
    jdote_tot_drifts = jdote.jcpara_dote + jdote.jgrad_dote + \
            jdote.jmag_dote + jdote.jpolar_dote \
            # + jdote.jqnupara_dote + jdote.jagy_dote
    jdote_tot_drifts_int = jdote.jcpara_dote_int + jdote.jgrad_dote_int + \
            jdote.jmag_dote_int + jdote.jpolar_dote_int \
            # + jdote.jqnupara_dote_int + jdote.jagy_dote_int
    if species == 'e':
        dkene = pic_info.dkene_e
        kene = pic_info.kene_e
        kename = '$\Delta K_e$'
    else:
        dkene = pic_info.dkene_i
        kene = pic_info.kene_i
        kename = '$\Delta K_i$'

    tfields = pic_info.tfields
    tenergy = pic_info.tenergy
    #fig, axes = plt.subplots(2, sharex=True, sharey=False)
    fig = plt.figure(figsize=[7, 4])
   
    width = 0.82
    height = 0.4
    xs = 0.14
    ys = 0.15
    #mpl.rc('text', usetex=False)
    ax1 = fig.add_axes([xs, ys+height, width, height])
    ax1.plot(tfields, jdote.jcpara_dote, lw=2, color='b', 
            label=r'$\mathbf{j}_c\cdot\mathbf{E}$')
    ax1.plot(tfields, jdote.jgrad_dote, lw=2, color='g',
            label=r'$\mathbf{j}_g\cdot\mathbf{E}$')
    ax1.plot(tfields, jdote.jmag_dote, lw=2, color='r',
            label=r'$\mathbf{j}_m\cdot\mathbf{E}$')
    #axes[0].plot(tf, jpolar_dote, lw=2, label=r'$\mathbf{j}_p\cdot\mathbf{E}$')
    #ax1.plot(tf, jqnupara_dote, lw=2, 
    #        label=r'$\mathbf{j}_\parallel\cdot\mathbf{E}$')
    ax1.plot(tfields, jdote_tot_drifts, lw=2, color='m', 
            label=r'$\mathbf{j}\cdot\mathbf{E}$')
    ax1.plot(tenergy, dkene, lw=2, color='k', label=kename)
    ax1.plot([np.min(tenergy), np.max(tenergy)], [0,0], 'k--')
    ax1.set_ylabel(r'$d\varepsilon_c/dt$', fontdict=font, fontsize=20)
    ax1.tick_params(reset=True, labelsize=16)
    ax1.tick_params(axis='x', labelbottom='off')
    #start, end = ax1.get_ylim()
    #ax1.yaxis.set_ticks(np.arange(start+0.1, end, 0.2))
    ax1.set_xlim([0, 800])

    enorm = pic_info.ene_bx[0]
    ax2 = fig.add_axes([xs, ys, width, height])
    ax2.plot(tfields, jdote.jcpara_dote_int/enorm, lw=2, color='b')
    ax2.plot(tfields, jdote.jgrad_dote_int/enorm, lw=2, color='g')
    ax2.plot(tfields, jdote.jmag_dote_int/enorm, lw=2, color='r')
    #ax2.plot(tf, jqnupara_dote_int, lw=2)
    ax2.plot(tfields, jdote_tot_drifts_int/enorm, lw=2, color='m')
    ax2.plot(tenergy, (kene-kene[0])/enorm, color='k', lw=2)
    ax2.plot([np.min(tenergy), np.max(tenergy)], [0,0], 'k--')

    ax2.set_xlabel(r'$t\Omega_{ci}$', fontdict=font, fontsize=20)
    ax2.set_ylabel(r'$\varepsilon_c$', fontdict=font, fontsize=20)
    ax2.tick_params(labelsize=16)
    ax2.set_xlim([0, 800])
    start, end = ax2.get_ylim()
    ax2.yaxis.set_ticks(np.arange(start+0.05, end, 0.1))

    #ax1.legend(loc=1, prop={'size':16}, ncol=2,
    #        shadow=True, fancybox=True)

    ax1.text(0.65, 0.85, r'$\mathbf{j}_g\cdot\mathbf{E}$', color='g', fontsize=20,
            horizontalalignment='left', verticalalignment='center',
            transform = ax1.transAxes)
    ax1.text(0.85, 0.85, r'$\mathbf{j}_m\cdot\mathbf{E}$', color='r', fontsize=20,
            horizontalalignment='left', verticalalignment='center',
            transform = ax1.transAxes)
    ax1.text(0.85, 0.65, r'$\mathbf{j}_c\cdot\mathbf{E}$', color='b', fontsize=20,
            horizontalalignment='left', verticalalignment='center',
            transform = ax1.transAxes)
    ax1.text(0.65, 0.15, r'$dK_e/dt$', color='k', fontsize=20,
            horizontalalignment='left', verticalalignment='center',
            transform = ax1.transAxes)
    ax1.text(0.85, 0.15, r"$\mathbf{j}_\perp\cdot\mathbf{E}$", color='m',
            fontsize=20, horizontalalignment='left', verticalalignment='center',
            transform = ax1.transAxes)

    td = 100
    print 'The fraction of perpendicular heating (model): ', \
            jdote_tot_drifts_int[td]/(kene[td]-kene[0])
    print 'The fraction of perpendicular heating (simulation): ', \
            jdote.jqnuperp_dote_int[-1]/(kene[-1]-kene[0])

    #plt.tight_layout()
    #fname = 'jdrifts_dote_' + species + '.eps'
    #fig.savefig(fname)
    plt.show()

def plot_jpara_perp_dote():
    """Plot the parallel and perpendicular heating using the simulation data.

    """
    jdote = read_jdote_data('e')
    pic_info = pic_information.get_pic_info('../../')

    tfields = pic_info.tfields
    tenergy = pic_info.tenergy
    jtot_dote = jdote.jqnupara_dote + jdote.jqnuperp_dote
    jtot_dote_int = jdote.jqnupara_dote_int + jdote.jqnuperp_dote_int
    dkene = pic_info.dkene_e
    kene = pic_info.kene_e
    kename = '$\Delta K_e$'

    #fig, axes = plt.subplots(2, sharex=True, sharey=False)
    fig = plt.figure(figsize=[7, 5])
   
    width = 0.84
    height = 0.4
    xs = 0.13
    ys = 0.97-height
    #mpl.rc('text', usetex=False)
    ax1 = fig.add_axes([xs, ys, width, height])
    ax1.plot(tfields, jdote.jqnupara_dote, lw=2, color='b', 
            label=r'$\mathbf{j}_{e\parallel}\cdot\mathbf{E}$')
    ax1.plot(tfields, jdote.jqnuperp_dote, lw=2, color='g',
            label=r'$\mathbf{j}_{e\perp}\cdot\mathbf{E}$')
    ax1.plot(tfields, jtot_dote, lw=2, color='r',
            label=r'$\mathbf{j}\cdot\mathbf{E}$')
    ax1.plot(tenergy, dkene, lw=2, color='k', label=kename)
    ax1.plot([np.min(tenergy), np.max(tenergy)], [0,0], 'k--')
    ax1.set_ylabel(r'$d\varepsilon_c/dt$', fontdict=font, fontsize=24)
    ax1.tick_params(reset=True, labelsize=20)
    ax1.tick_params(axis='x', labelbottom='off')
    ax1.set_ylim([-2, 6])
    ax1.yaxis.set_ticks(np.arange(-1, 6, 1))
    ax1.set_xlim([0, 800])
    ax1.text(10, 5, r'$(e)$', color='black', fontsize=24,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0))
    ax1.text(200, -1.5, r'$\mathbf{j}_{\parallel}\cdot\mathbf{E}$',
            color='blue', fontsize=24)
    ax1.text(350, -1.5, r'$\mathbf{j}_{\perp}\cdot\mathbf{E}$',
            color='green', fontsize=24)
    ax1.text(500, -1.5, r'$(\mathbf{j}_{\parallel}+\mathbf{j}_\perp)\cdot\mathbf{E}$',
            color='red', fontsize=24)
    ax1.text(650, 4, r'$dK_e/dt$', color='black', fontsize=24)

    jdote = read_jdote_data('i')
    jtot_dote = jdote.jqnupara_dote + jdote.jqnuperp_dote
    jtot_dote_int = jdote.jqnupara_dote_int + jdote.jqnuperp_dote_int
    dkene = pic_info.dkene_i
    kene = pic_info.kene_i
    kename = '$\Delta K_i$'

    ys -= height + 0.03
    ax2 = fig.add_axes([xs, ys, width, height])
    ax2.plot(tfields, jdote.jqnupara_dote, lw=2, color='b', 
            label=r'$\mathbf{j}_{e\parallel}\cdot\mathbf{E}$')
    ax2.plot(tfields, jdote.jqnuperp_dote, lw=2, color='g',
            label=r'$\mathbf{j}_{e\perp}\cdot\mathbf{E}$')
    ax2.plot(tfields, jtot_dote, lw=2, color='r',
            label=r'$\mathbf{j}\cdot\mathbf{E}$')
    ax2.plot(tenergy, dkene, lw=2, color='k', label=kename)
    ax2.plot([np.min(tenergy), np.max(tenergy)], [0,0], 'k--')
    ax2.set_xlabel(r'$t\Omega_{ci}$', fontdict=font, fontsize=24)
    ax2.set_ylabel(r'$d\varepsilon_c/dt$', fontdict=font, fontsize=24)
    ax2.tick_params(reset=True, labelsize=20)
    ax2.set_ylim([-2, 10])
    ax2.yaxis.set_ticks(np.arange(-2, 11, 2))
    ax2.set_xlim([0, 800])
    ax2.text(650, 7, r'$dK_i/dt$', color='black', fontsize=24)
    ax2.text(10, 8, r'$(i)$', color='black', fontsize=24,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0))

    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    fig.savefig('../img/jpp_dote.eps')

    plt.show()


if __name__ == "__main__":
    pic_info = pic_information.get_pic_info('../../')
    jdote = read_jdote_data('e')
    # plot_energy_evolution(pic_info)
    # plot_particle_energy_gain()
    plot_jdotes_evolution('i')
    # plot_jpara_perp_dote()
