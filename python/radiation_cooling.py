"""
Analysis procedures for runs with radiation cooling
"""
import argparse
import math
import multiprocessing

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import palettable
from joblib import Parallel, delayed
from matplotlib.colors import LogNorm
from scipy.ndimage.filters import gaussian_filter, median_filter

from contour_plots import read_2d_fields
from energy_conversion import read_data_from_json
from shell_functions import mkdir_p

plt.style.use("seaborn-deep")
mpl.rc("font", family="Times New Roman")
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

FONT = {'family': 'serif',
        'color': 'black',
        'weight': 'normal',
        'size': 24}

def plot_bfield_single(run_dir, run_name, tframe, show_plot=True):
    """Plot magnetic field for a single time frame

    Args:
        run_dir: PIC simulation root directory
        run_name: PIC simulation run name
        tframe: time frame
    """
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    kwargs = {"current_time": tframe, "xl": 0, "xr": pic_info.lx_di,
              "zb": -0.5 * pic_info.lz_di, "zt": 0.5 * pic_info.lz_di}
    size_one_frame = pic_info.nx * pic_info.nz * 4
    fname = run_dir + "data/bx.gda"
    x, z, bx = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/by.gda"
    x, z, by = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/bz.gda"
    x, z, bz = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/absB.gda"
    x, z, absB = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/Ay.gda"
    x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
    xmin, xmax = np.min(x), np.max(x)
    zmin, zmax = np.min(z), np.max(z)

    w0, h0 = 0.78, 0.2
    xs0, ys0 = 0.09, 0.95 - h0
    vgap, hgap = 0.02, 0.02

    vmax1 = 300
    vmin1 = -vmax1
    def plot_one_field(fdata, ax, text, text_color, label_bottom='on',
                       label_left='on', ylabel=False, ay_color='k',
                       vmin=vmin1, vmax=vmax1, cmap1=plt.cm.seismic,
                       log_scale=False):
        plt.tick_params(labelsize=16)
        if log_scale:
            im1 = ax.imshow(fdata, cmap=cmap1,
                            extent=[xmin, xmax, zmin, zmax], aspect='auto',
                            origin='lower', interpolation='bicubic')
            im1.norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            im1 = ax.imshow(fdata, cmap=cmap1, vmin=vmin, vmax=vmax,
                            extent=[xmin, xmax, zmin, zmax], aspect='auto',
                            origin='lower', interpolation='bicubic')
        ax.tick_params(axis='x', labelbottom=label_bottom)
        ax.tick_params(axis='y', labelleft=label_left)
        if ylabel:
            ax.set_ylabel(r'$z/d_i$', fontdict=FONT, fontsize=20)
        ax.contour(x, z, Ay, colors=ay_color, linewidths=0.5)
        ax.text(0.02, 0.85, text, color=text_color, fontsize=20,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)
        return im1
    fig = plt.figure(figsize=[12, 12])
    xs, ys = xs0, ys0
    ax1 = fig.add_axes([xs, ys, w0, h0])
    im1 = plot_one_field(bx, ax1, r'$B_x$', 'w', label_bottom='off',
                         label_left='on', ylabel=True, ay_color='k')
    ys -= h0 + vgap
    ax2 = fig.add_axes([xs, ys, w0, h0])
    im2 = plot_one_field(by, ax2, r'$B_y$', 'k', label_bottom='off',
                         label_left='on', ylabel=True, ay_color='k')
    ys -= h0 + vgap
    ax3 = fig.add_axes([xs, ys, w0, h0])
    im3 = plot_one_field(bz, ax3, r'$B_z$', 'k', label_bottom='off',
                         label_left='on', ylabel=True, ay_color='k')
    xs1 = xs + w0 + hgap
    w1 = 0.03
    h1 = 3 * h0 + 2 * vgap
    cax1 = fig.add_axes([xs1, ys, w1, h1])
    cbar1 = fig.colorbar(im1, cax=cax1)
    cbar1.ax.tick_params(labelsize=16)
    ys -= h0 + vgap
    ax4 = fig.add_axes([xs, ys, w0, h0])
    im4 = plot_one_field(absB, ax4, r'$B$', 'k', label_bottom='on',
                         label_left='on', ylabel=True, ay_color='k',
                         vmin=10, vmax=vmax1, cmap1=plt.cm.viridis)
    ax4.set_xlabel(r'$x/d_i$', fontdict=FONT, fontsize=20)
    cax2 = fig.add_axes([xs1, ys, w1, h0])
    cbar2 = fig.colorbar(im4, cax=cax2)
    cbar2.ax.tick_params(labelsize=16)

    t_wci = tframe * pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(t_wci) + '\Omega_{ci}^{-1}$'
    title += ' (frame: %d)' % tframe
    ax1.set_title(title, fontdict=FONT, fontsize=24)

    fdir = '../img/radiation_cooling/magnetic_field/' + run_name + '/'
    mkdir_p(fdir)
    fname = fdir + 'bfields_' + str(tframe) + '.jpg'
    fig.savefig(fname, dpi=200)

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_density_eband(run_dir, run_name, tframe, species):
    """Plot density for a single energy band

    Args:
        run_dir: PIC simulation root directory
        run_name: PIC simulation run name
        tframe: time frame
        species: particle species, 'e' or 'i'
    """
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    kwargs = {"current_time": tframe, "xl": 0, "xr": pic_info.lx_di,
              "zb": -0.5 * pic_info.lz_di, "zt": 0.5 * pic_info.lz_di}
    size_one_frame = pic_info.nx * pic_info.nz * 4
    fname = run_dir + "data/" + species + "EB02.gda"
    x, z, eb02 = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/" + species + "EB03.gda"
    x, z, eb03 = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/" + species + "EB04.gda"
    x, z, eb04 = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/" + species + "EB05.gda"
    x, z, eb05 = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/Ay.gda"
    x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
    xmin, xmax = np.min(x), np.max(x)
    zmin, zmax = np.min(z), np.max(z)

    w0, h0 = 0.78, 0.2
    xs0, ys0 = 0.09, 0.95 - h0
    vgap, hgap = 0.02, 0.02

    vmax1 = 1.0E2
    vmin1 = 1
    def plot_one_field(fdata, ax, text, text_color, label_bottom='on',
                       label_left='on', ylabel=False, ay_color='k',
                       vmin=vmin1, vmax=vmax1, cmap1=plt.cm.seismic,
                       log_scale=False):
        plt.tick_params(labelsize=16)
        if log_scale:
            im1 = ax.imshow(fdata, cmap=cmap1,
                            extent=[xmin, xmax, zmin, zmax], aspect='auto',
                            origin='lower', interpolation='bicubic')
            im1.norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            im1 = ax.imshow(fdata, cmap=cmap1, vmin=vmin, vmax=vmax,
                            extent=[xmin, xmax, zmin, zmax], aspect='auto',
                            origin='lower', interpolation='bicubic')
        ax.tick_params(axis='x', labelbottom=label_bottom)
        ax.tick_params(axis='y', labelleft=label_left)
        if ylabel:
            ax.set_ylabel(r'$z/d_i$', fontdict=FONT, fontsize=20)
        ax.contour(x, z, Ay, colors=ay_color, linewidths=0.5)
        ax.text(0.02, 0.85, text, color=text_color, fontsize=20,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)
        return im1
    fig = plt.figure(figsize=[12, 12])
    xs, ys = xs0, ys0
    ax1 = fig.add_axes([xs, ys, w0, h0])
    label1 = r'$N(2\times 10^3<E<4\times 10^3)$'
    im1 = plot_one_field(eb02, ax1, label1, 'k', label_bottom='off',
                         label_left='on', ylabel=True, ay_color='k',
                         cmap1=plt.cm.viridis, log_scale=True)
    ys -= h0 + vgap
    ax2 = fig.add_axes([xs, ys, w0, h0])
    label2 = r'$N(4\times 10^3<E<8\times 10^3)$'
    im2 = plot_one_field(eb03, ax2, label2, 'k', label_bottom='off',
                         label_left='on', ylabel=True, ay_color='k',
                         cmap1=plt.cm.viridis, log_scale=True)
    ys -= h0 + vgap
    ax3 = fig.add_axes([xs, ys, w0, h0])
    label3 = r'$N(8\times 10^3<E<1.6\times 10^4)$'
    im3 = plot_one_field(eb04, ax3, label3, 'k', label_bottom='off',
                         label_left='on', ylabel=True, ay_color='k',
                         cmap1=plt.cm.viridis, log_scale=True)
    ys -= h0 + vgap
    ax4 = fig.add_axes([xs, ys, w0, h0])
    label4 = r'$N(1.6\times 10^4<E<3.2\times 10^4)$'
    im4 = plot_one_field(eb05, ax4, label4, 'k', label_bottom='on',
                         label_left='on', ylabel=True, ay_color='k',
                         cmap1=plt.cm.viridis, log_scale=True)
    ax4.set_xlabel(r'$x/d_i$', fontdict=FONT, fontsize=20)
    xs1 = xs + w0 + hgap
    w1 = 0.03
    h1 = 4 * h0 + 3 * vgap
    cax1 = fig.add_axes([xs1, ys, w1, h1])
    cbar1 = fig.colorbar(im1, cax=cax1)
    cbar1.ax.tick_params(labelsize=16)

    t_wci = tframe * pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(t_wci) + '\Omega_{ci}^{-1}$'
    title += ' (frame: %d)' % tframe
    ax1.set_title(title, fontdict=FONT, fontsize=24)

    fdir = '../img/radiation_cooling/density_eband/' + run_name + '/'
    mkdir_p(fdir)
    fname = fdir + 'density_eband' + str(tframe) + '.jpg'
    fig.savefig(fname, dpi=200)

    plt.close()
    # plt.show()


def radiation_map(plot_config, show_plot=True):
    """Plot magnetic field, high-energy particle distribution, and radiation map

    Args:
        plot_config: dictionary holding PIC simulation directory, run name,
                   time frame, and particle species to plot
    """
    picinfo_fname = ('../data/pic_info/pic_info_' +
                     plot_config["run_name"] + '.json')
    pic_info = read_data_from_json(picinfo_fname)
    kwargs = {"current_time": plot_config["tframe"],
              "xl": 0, "xr": pic_info.lx_di,
              "zb": -0.5 * pic_info.lz_di, "zt": 0.5 * pic_info.lz_di}
    size_one_frame = pic_info.nx * pic_info.nz * 4
    fname = plot_config["run_dir"] + "data/absB.gda"
    xgrid, zgrid, absB = read_2d_fields(pic_info, fname, **kwargs)

    fname = plot_config["run_dir"] + "data/" + plot_config["species"] + "EB02.gda"
    xgrid, zgrid, eb02 = read_2d_fields(pic_info, fname, **kwargs)
    fname = plot_config["run_dir"] + "data/" + plot_config["species"] + "EB03.gda"
    xgrid, zgrid, eb03 = read_2d_fields(pic_info, fname, **kwargs)
    fname = plot_config["run_dir"] + "data/" + plot_config["species"] + "EB04.gda"
    xgrid, zgrid, eb04 = read_2d_fields(pic_info, fname, **kwargs)
    prho = eb02 + eb03 + eb04
    text2 = r"$5000<\gamma<20000$"

    # fname = plot_config["run_dir"] + "data/" + plot_config["species"] + "EB06.gda"
    # xgrid, zgrid, eb06 = read_2d_fields(pic_info, fname, **kwargs)
    # prho = eb06
    # text2 = r"$2500<\gamma<3000$"

    # fname = plot_config["run_dir"] + "data/" + plot_config["species"] + "EB04.gda"
    # xgrid, zgrid, eb04 = read_2d_fields(pic_info, fname, **kwargs)
    # prho = eb04
    # text2 = r"$8000<\gamma<16000$"

    fname = plot_config["run_dir"] + "data/Ay.gda"
    xgrid, zgrid, Ay = read_2d_fields(pic_info, fname, **kwargs)
    smime = math.sqrt(pic_info.mime)
    xgrid *= smime
    zgrid *= smime
    sizes = [xgrid[0], xgrid[-1], zgrid[0], zgrid[-1]]

    fig = plt.figure(figsize=[12, 14])
    rect = [0.10, 0.66, 0.8, 0.28]
    ax1 = fig.add_axes(rect)
    img = ax1.imshow(absB/2000, extent=sizes, aspect='auto',
                     # vmin=0, vmax=500,
                     norm = LogNorm(vmin=0.005, vmax=0.25),
                     cmap=plt.cm.inferno, origin='lower')
    Ay_min = np.min(Ay)
    Ay_max = np.max(Ay)
    levels = np.linspace(Ay_min, Ay_max, 20)
    ax1.contour(xgrid, zgrid, Ay, colors='k',
                levels=levels, linewidths=0.5)
    ax1.tick_params(labelsize=16)
    ax1.set_ylabel(r'$z/d_e$', fontsize=20)
    ax1.tick_params(axis='x', labelbottom='off')
    rect1 = np.copy(rect)
    rect1[0] += rect1[2] + 0.02
    rect1[2] = 0.03
    cax1 = fig.add_axes(rect1)
    cbar1 = fig.colorbar(img, cax=cax1)
    cbar1.ax.tick_params(labelsize=16)
    cbar1.ax.set_title(r"$G$", fontsize=32)
    ax1.text(0.02, 0.9, r"$|B|$", color='k', fontsize=32,
             bbox=dict(facecolor='none', alpha=1.0,
                       edgecolor='none', pad=10.0),
             horizontalalignment='left', verticalalignment='center',
             transform=ax1.transAxes)

    rect[1] -= rect[3] + 0.02
    ax2 = fig.add_axes(rect)
    img = ax2.imshow(prho, extent=sizes, aspect='auto',
                     cmap=plt.cm.viridis, origin='lower',
                     norm = LogNorm(vmin=1, vmax=100))
    ax2.contour(xgrid, zgrid, Ay, colors='k',
                levels=levels, linewidths=0.5)
    ax2.tick_params(labelsize=16)
    ax2.set_ylabel(r'$z/d_e$', fontsize=20)
    ax2.tick_params(axis='x', labelbottom='off')
    rect2 = np.copy(rect)
    rect2[0] += rect2[2] + 0.02
    rect2[2] = 0.03
    cax2 = fig.add_axes(rect2)
    cbar2 = fig.colorbar(img, cax=cax2)
    cbar2.ax.tick_params(labelsize=16)
    ax2.text(0.02, 0.9, text2, color='k', fontsize=32,
             bbox=dict(facecolor='none', alpha=1.0,
                       edgecolor='none', pad=10.0),
             horizontalalignment='left', verticalalignment='center',
             transform=ax2.transAxes)

    rect[1] -= rect[3] + 0.02
    ax3 = fig.add_axes(rect)
    im_pol = contour_radiation(plot_config, pic_info, ax3)
    ax3.contour(xgrid, zgrid, Ay, colors='k',
                levels=levels, linewidths=0.5)
    ax3.set_xlabel(r'$x/d_e$', fontsize=20)
    ax3.set_ylabel(r'$z/d_e$', fontsize=20)
    rect[0] += rect[2] + 0.02
    rect[2] = 0.03
    cax3 = fig.add_axes(rect)
    cbar3 = fig.colorbar(im_pol, cax=cax3)
    cbar3.ax.tick_params(labelsize=16)
    ax3.text(0.02, 0.9, 'Optical', color='k', fontsize=32,
             bbox=dict(facecolor='none', alpha=1.0,
                       edgecolor='none', pad=10.0),
             horizontalalignment='left', verticalalignment='center',
             transform=ax3.transAxes)

    title = "Observation angle: " + str(plot_config["obs_ang"]) + "$^\circ$"
    title += ' (frame: %d)' % plot_config["tframe"]
    fig.suptitle(title, fontsize=24)

    fdir = '../img/radiation_cooling/absb_rad_map/' + plot_config["run_name"] + '/'
    mkdir_p(fdir)
    fname = (fdir + 'rad_map' + str(plot_config["tframe"]) + "_" +
             str(plot_config["obs_ang"]) + ".jpg")
    fig.savefig(fname, dpi=200)

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_momentum_flux(run_dir, run_name, tframe):
    """Plot momentum flux

    Args:
        run_dir: PIC simulation root directory
        run_name: PIC simulation run name
        tframe: time frame
    """
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    kwargs = {"current_time": tframe, "xl": 0, "xr": pic_info.lx_di,
              "zb": -0.5 * pic_info.lz_di, "zt": 0.5 * pic_info.lz_di}
    size_one_frame = pic_info.nx * pic_info.nz * 4
    # fname = run_dir + "data/uex.gda"
    # x, z, uex = read_2d_fields(pic_info, fname, **kwargs)
    # fname = run_dir + "data/uey.gda"
    # x, z, uey = read_2d_fields(pic_info, fname, **kwargs)
    # fname = run_dir + "data/uez.gda"
    # x, z, uez = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/vex.gda"
    x, z, vex = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/vey.gda"
    x, z, vey = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/vez.gda"
    x, z, vez = read_2d_fields(pic_info, fname, **kwargs)
    # fname = run_dir + "data/uix.gda"
    # x, z, uix = read_2d_fields(pic_info, fname, **kwargs)
    # fname = run_dir + "data/uiy.gda"
    # x, z, uiy = read_2d_fields(pic_info, fname, **kwargs)
    # fname = run_dir + "data/uiz.gda"
    # x, z, uiz = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/ne.gda"
    x, z, ne = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/ni.gda"
    x, z, ni = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + "data/Ay.gda"
    x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
    xmin, xmax = np.min(x), np.max(x)
    zmin, zmax = np.min(z), np.max(z)

    smime = math.sqrt(pic_info.mime)
    dx = pic_info.dx_di * smime
    dz = pic_info.dz_di * smime

    curv_vex = -np.gradient(vey, dz, axis=0)
    curv_vey = np.gradient(vex, dz, axis=0) - np.gradient(vez, dx, axis=1)
    curv_vez = np.gradient(vey, dx, axis=1)
    curv_vex = gaussian_filter(curv_vex, 3)
    curv_vey = gaussian_filter(curv_vey, 3)
    curv_vez = gaussian_filter(curv_vez, 3)

    xv, zv = np.meshgrid(x, z)

    w0, h0 = 0.78, 0.2
    xs0, ys0 = 0.09, 0.95 - h0
    vgap, hgap = 0.02, 0.02

    vmax1 = 1.0E-2
    vmin1 = -vmax1
    def plot_one_field(fdata, ax, text, text_color, label_bottom='on',
                       label_left='on', ylabel=False, ay_color='k',
                       vmin=vmin1, vmax=vmax1, cmap1=plt.cm.seismic,
                       log_scale=False):
        plt.tick_params(labelsize=16)
        if log_scale:
            im1 = ax.imshow(fdata, cmap=cmap1,
                            extent=[xmin, xmax, zmin, zmax], aspect='auto',
                            origin='lower', interpolation='bicubic')
            im1.norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            im1 = ax.imshow(fdata, cmap=cmap1, vmin=vmin, vmax=vmax,
                            extent=[xmin, xmax, zmin, zmax], aspect='auto',
                            origin='lower', interpolation='bicubic')
        ax.tick_params(axis='x', labelbottom=label_bottom)
        ax.tick_params(axis='y', labelleft=label_left)
        if ylabel:
            ax.set_ylabel(r'$z/d_i$', fontdict=FONT, fontsize=20)
        ax.contour(x, z, Ay, colors=ay_color, linewidths=0.5)
        ax.text(0.02, 0.85, text, color=text_color, fontsize=20,
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)
        return im1
    fig = plt.figure(figsize=[12, 12])
    xs, ys = xs0, ys0
    ax1 = fig.add_axes([xs, ys, w0, h0])
    im1 = plot_one_field(curv_vex, ax1, r'$n_em_eu_{ex}$', 'k',
                         label_bottom='off', label_left='on', ylabel=True,
                         ay_color='k')
    # strm = ax1.streamplot(xv, zv, vex, vez, linewidth=1, color=vex,
    #                       density=[2,2], cmap=plt.cm.binary)
    ys -= h0 + vgap
    ax2 = fig.add_axes([xs, ys, w0, h0])
    im2 = plot_one_field(curv_vey, ax2, r'$n_em_eu_{ey}$', 'k',
                         label_bottom='off', label_left='on', ylabel=True,
                         ay_color='k')
    ys -= h0 + vgap
    ax3 = fig.add_axes([xs, ys, w0, h0])
    im3 = plot_one_field(curv_vez, ax3, r'$n_em_eu_{ez}$', 'k',
                         label_bottom='off', label_left='on', ylabel=True,
                         ay_color='k')
    # xs1 = xs + w0 + hgap
    # w1 = 0.03
    # h1 = 3 * h0 + 2 * vgap
    # cax1 = fig.add_axes([xs1, ys, w1, h1])
    # cbar1 = fig.colorbar(im1, cax=cax1)
    # cbar1.ax.tick_params(labelsize=16)
    # ys -= h0 + vgap
    # ax4 = fig.add_axes([xs, ys, w0, h0])
    # im4 = plot_one_field(absB, ax4, r'$B$', 'k', label_bottom='on',
    #                      label_left='on', ylabel=True, ay_color='k',
    #                      vmin=10, vmax=300, cmap1=plt.cm.viridis)
    # ax4.set_xlabel(r'$x/d_i$', fontdict=FONT, fontsize=20)
    # cax2 = fig.add_axes([xs1, ys, w1, h0])
    # cbar2 = fig.colorbar(im4, cax=cax2)
    # cbar2.ax.tick_params(labelsize=16)

    t_wci = tframe * pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(t_wci) + '\Omega_{ci}^{-1}$'
    title += ' (frame: %d)' % tframe
    ax1.set_title(title, fontdict=FONT, fontsize=24)

    # fdir = '../img/radiation_cooling/magnetic_field/' + run_name + '/'
    # mkdir_p(fdir)
    # fname = fdir + 'bfields_' + str(tframe) + '.jpg'
    # fig.savefig(fname, dpi=200)

    # plt.close()
    plt.show()


def plot_dist_2d(run_dir, run_name, tframe):
    """Plot density distributions for particles at different energy band

    Args:
        run_dir: PIC simulation root directory
        run_name: PIC simulation run name
        tframe: time frame
    """
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    mpi_sizex = pic_info.topology_x
    mpi_sizez = pic_info.topology_z
    nbins = 1000
    ndata = nbins + 3  # including bx, by, bz

    rank = 0
    tindex = tframe * pic_info.fields_interval
    fname_pre = (run_dir + 'hydro/T.' + str(tindex) +
                 '/spectrum-ehydro.' + str(tindex))
    fname = fname_pre + '.' + str(rank)
    fdata = np.fromfile(fname, dtype=np.float32)
    sz, = fdata.shape
    nzone = sz / ndata

    fname = run_dir + 'distributions_2d/dist_3e3.' + str(tframe)
    nrho_3e3 = np.fromfile(fname)
    nrho_3e3 = nrho_3e3.reshape((mpi_sizex, nzone*mpi_sizez))
    fname = run_dir + 'distributions_2d/dist_1e4.' + str(tframe)
    nrho_1e4 = np.fromfile(fname)
    nrho_1e4 = nrho_1e4.reshape((mpi_sizex, nzone*mpi_sizez))
    fname = run_dir + 'distributions_2d/dist_3e4.' + str(tframe)
    nrho_3e4 = np.fromfile(fname)
    nrho_3e4 = nrho_3e4.reshape((mpi_sizex, nzone*mpi_sizez))

    kwargs = {"current_time": tframe, "xl": 0, "xr": 1000, "zb": -250, "zt": 250}
    fname = run_dir + 'data/absB.gda'
    x, z, absB = read_2d_fields(pic_info, fname, **kwargs)
    fname = run_dir + 'data/Ay.gda'
    x, z, Ay = read_2d_fields(pic_info, fname, **kwargs)
    nx, = x.shape
    nz, = z.shape

    smime = math.sqrt(pic_info.mime)
    x *= smime # di -> de
    z *= smime

    xmin, xmax = 0, pic_info.lx_di * smime
    lz_de = pic_info.lz_di * smime
    zmin, zmax = -0.5 * lz_de, 0.5 * lz_de
    vmin, vmax = 1E2, 1E5

    fig = plt.figure(figsize=[8, 12])
    xs0, ys0 = 0.14, 0.74
    w1, h1 = 0.76, 0.205
    gap = 0.02
    ax1 = fig.add_axes([xs0, ys0, w1, h1])
    print(np.min(nrho_3e3), np.max(nrho_3e3))
    p1 = ax1.imshow(nrho_3e3.T, cmap=plt.cm.jet,
            extent=[xmin, xmax, zmin, zmax],
            aspect='auto', origin='lower',
            norm = LogNorm(vmin=vmin, vmax=vmax),
            interpolation='bicubic')
    ax1.tick_params(labelsize=16)
    ax1.tick_params(axis='x', labelbottom='off')
    nlevels = 20
    levels = np.linspace(np.min(Ay), np.max(Ay), nlevels)
    ax1.contour(x, z, Ay, colors='k', levels=levels, linewidths=0.5)
    ax1.set_ylabel(r'$z/d_e$', fontsize=20)
    fname1 = r'$N(2\times 10^3<E<4.5\times 10^3)$'
    ax1.text(0.02, 0.9, fname1, color='k', fontsize=20,
        bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='left', verticalalignment='center',
        transform=ax1.transAxes)

    ys = ys0 - gap - h1
    ax2 = fig.add_axes([xs0, ys, w1, h1])
    print(np.min(nrho_1e4), np.max(nrho_1e4))
    p2 = ax2.imshow(nrho_1e4.T, cmap=plt.cm.jet,
            extent=[xmin, xmax, zmin, zmax],
            aspect='auto', origin='lower',
            norm = LogNorm(vmin=vmin, vmax=vmax),
            interpolation='bicubic')
    ax2.tick_params(labelsize=16)
    ax2.tick_params(axis='x', labelbottom='off')
    ax2.contour(x, z, Ay, colors='black', levels=levels, linewidths=0.5)
    ax2.set_ylabel(r'$z/d_e$', fontsize=20)
    fname2 = r'$N(10^4/1.5<E<1.5\times 10^4)$'
    ax2.text(0.02, 0.9, fname2, color='black', fontsize=20,
        bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='left', verticalalignment='center',
        transform=ax2.transAxes)

    ys -= gap + h1
    ax3 = fig.add_axes([xs0, ys, w1, h1])
    print(np.min(nrho_3e4), np.max(nrho_3e4))
    p3 = ax3.imshow(nrho_3e4.T, cmap=plt.cm.jet,
            extent=[xmin, xmax, zmin, zmax],
            aspect='auto', origin='lower',
            norm = LogNorm(vmin=vmin, vmax=vmax),
            interpolation='bicubic')
    ax3.tick_params(labelsize=16)
    ax3.tick_params(axis='x', labelbottom='off')
    ax3.contour(x, z, Ay, colors='black', levels=levels, linewidths=0.5)
    ax3.set_ylabel(r'$z/d_e$', fontsize=20)
    fname3 = r'$N(2\times 10^4<E<4.5\times 10^4)$'
    ax3.text(0.02, 0.9, fname3, color='black', fontsize=20,
        bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='left', verticalalignment='center',
        transform=ax3.transAxes)

    xs = xs0 + w1 + 0.01
    cax = fig.add_axes([xs, ys, 0.02, 3*h1+2*gap])
    cbar = fig.colorbar(p3, cax=cax)
    cbar.ax.tick_params(labelsize=16)

    bmin, bmax = 10, 320

    ys -= gap + h1
    ax4 = fig.add_axes([xs0, ys, w1, h1])
    print(np.min(absB), np.max(absB))
    p4 = ax4.imshow(absB, cmap=plt.cm.jet,
            extent=[xmin, xmax, zmin, zmax],
            aspect='auto', origin='lower',
            vmin=bmin, vmax=bmax,
            interpolation='bicubic')
    ax4.tick_params(labelsize=16)
    ax4.contour(x, z, Ay, colors='black', levels=levels, linewidths=0.5)
    ax4.set_xlabel(r'$x/d_e$', fontsize=20)
    ax4.set_ylabel(r'$z/d_e$', fontsize=20)
    fname4 = r'$B$'
    ax4.text(0.02, 0.9, fname4, color='black', fontsize=20,
        bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
        horizontalalignment='left', verticalalignment='center',
        transform=ax4.transAxes)

    xs = xs0 + w1 + 0.01
    cax1 = fig.add_axes([xs, ys, 0.02, h1])
    cbar1 = fig.colorbar(p4, cax=cax1)
    cbar1.ax.tick_params(labelsize=16)

    t_wci = tframe * pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(t_wci) + '\Omega_{ci}^{-1}$'
    title += ' (frame: %d)' % tframe
    ax1.set_title(title, fontdict=FONT, fontsize=24)

    fdir = '../img/radiation_cooling/nene_b/' + run_name + '/'
    mkdir_p(fdir)
    fname = fdir + 'nene_b_' + str(tframe) + '.jpg'
    fig.savefig(fname, dpi=200)

    plt.close()
    # plt.show()


def contour_radiation(plot_config, pic_info, ax):
    """Plot contour of radiation map

    Args:
        plot_config: plot configuration
        ax: plot axis
        pic_info: PIC simulation information
    """
    # read radiation map data
    ebin_str = str(plot_config["energy_band"]).zfill(4)
    tframe_str = str(plot_config["tframe"] + 9).zfill(4)  # shift 9 frames
    map_dir = plot_config["map_dir"]
    if plot_config["old_rad"]:
        fname = map_dir + 'totflux' + tframe_str + '.dat'
        tot_flux = np.genfromtxt(fname)
        fname = map_dir + 'polangl' + tframe_str + '.dat'
        pol_angl = np.genfromtxt(fname)
        fname = map_dir + 'polflux' + tframe_str + '.dat'
        pol_flux = np.genfromtxt(fname)
    else:
        fname = map_dir + 'map' + ebin_str + tframe_str + '.dat'
        fdata = np.genfromtxt(fname)
        sz = fdata.shape
        dsz = sz[0] // 4
        tot_flux = fdata[0:dsz, :]
        pol_flux = fdata[dsz:2*dsz, :]
        pol_angl = fdata[2*dsz:3*dsz, :]

    tot_flux[np.isnan(tot_flux)] = 0.0
    pol_flux[np.isnan(pol_flux)] = 0.0
    tot_flux[tot_flux < 1E-21] = 0.0
    pol_flux[pol_flux < 1E-21] = 0.0
    tot_flux = np.fliplr(tot_flux)
    pol_angl = np.fliplr(pol_angl)
    pol_flux = np.fliplr(pol_flux)
    tot_flux = np.flipud(tot_flux)
    pol_angl = np.flipud(pol_angl)
    pol_flux = np.flipud(pol_flux)

    # plot radiation total flux
    smime = math.sqrt(pic_info.mime)
    lx_de = pic_info.lx_di * smime
    lz_de = pic_info.lz_di * smime
    xmin = lx_de * plot_config["xrange"][0]
    xmax = lx_de * plot_config["xrange"][1]
    zmin = lz_de * (plot_config["zrange"][0] - 0.5)
    zmax = lz_de * (plot_config["zrange"][1] - 0.5)
    nx_rad, nz_rad = tot_flux.shape
    ixl = int(nx_rad * plot_config["xrange"][0])
    ixr = int(nx_rad * plot_config["xrange"][1])
    izb = int(nz_rad * plot_config["zrange"][0])
    izt = int(nz_rad * plot_config["zrange"][1])
    vmin = np.min(tot_flux)
    vmax = np.max(tot_flux)
    print("Min and max of total flux: %e %e" % (vmin, vmax))
    print("Min and max of polarization angle: %e %e" %
          (np.min(pol_angl), np.max(pol_angl)))
    print("Min and max of polarization flux %e %e" %
          (np.min(pol_flux), np.max(pol_flux)))
    vmin, vmax = plot_config["flux_range"][0]
    p1 = ax.imshow(tot_flux[ixl:ixr, izb:izt].T,
                   cmap=plt.cm.Oranges,
                   extent=[xmin, xmax, zmin, zmax],
                   norm=LogNorm(vmin=vmin, vmax=vmax),
                   aspect='auto', origin='lower',
                   interpolation='bicubic')
    ax.tick_params(labelsize=16)
    ax.set_xlabel(r'$x/d_e$', fontsize=20)
    ax.set_ylabel(r'$z/d_e$', fontsize=20)
    ax.tick_params(axis='x', which='minor', direction='in')
    ax.tick_params(axis='x', which='major', direction='in')
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in')

    # plot polarization angle and flux
    quiveropts = dict(color='black', headlength=0, pivot='middle',
                      scale=plot_config["pflux_scale"],
                      linewidth=.5, units='width', headwidth=1,
                      headaxislength=0)
    x_rad = np.linspace(0, lx_de, nx_rad)
    z_rad = np.linspace(-0.5 * lz_de, 0.5 * lz_de, nz_rad)
    X, Z = np.meshgrid(x_rad, z_rad)
    U = np.transpose(pol_flux*np.sin(pol_angl*math.pi/180))
    V = np.transpose(pol_flux*np.cos(pol_angl*math.pi/180))
    s = 1
    print(U.shape)
    Q = ax.quiver(X[izb:izt:s, ixl:ixr:s],
                  Z[izb:izt:s, ixl:ixr:s],
                  U[izb:izt:s, ixl:ixr:s],
                  V[izb:izt:s, ixl:ixr:s],
                  **quiveropts)

    return p1


def radiation_map_tri(plot_config, show_plot=True):
    """
    Plot magnetic field, high-energy particle distribution, and radiation map
    for three time steps

    Args:
        plot_config: dictionary holding PIC simulation directory, run name,
                   time frame, and particle species to plot
    """
    picinfo_fname = ('../data/pic_info/pic_info_' +
                     plot_config["run_name"] + '.json')
    pic_info = read_data_from_json(picinfo_fname)

    fig = plt.figure(figsize=[15, 8])
    rect0 = [0.09, 0.68, 0.27, 0.26]
    hgap, vgap = 0.01, 0.03

    for iframe, tframe in enumerate(plot_config["tframes"]):
        rect = np.copy(rect0)
        rect[0] += (rect0[2] + hgap) * iframe
        kwargs = {"current_time": tframe,
                  "xl": pic_info.lx_di * plot_config["xrange"][0],
                  "xr": pic_info.lx_di * plot_config["xrange"][1],
                  "zb": pic_info.lz_di * (plot_config["zrange"][0] - 0.5),
                  "zt": pic_info.lz_di * (plot_config["zrange"][1] - 0.5)}
        size_one_frame = pic_info.nx * pic_info.nz * 4
        fname = plot_config["run_dir"] + "data/absB.gda"
        xgrid, zgrid, absB = read_2d_fields(pic_info, fname, **kwargs)

        fname = plot_config["run_dir"] + "data/" + plot_config["species"] + "EB02.gda"
        xgrid, zgrid, eb02 = read_2d_fields(pic_info, fname, **kwargs)
        fname = plot_config["run_dir"] + "data/" + plot_config["species"] + "EB03.gda"
        xgrid, zgrid, eb03 = read_2d_fields(pic_info, fname, **kwargs)
        fname = plot_config["run_dir"] + "data/" + plot_config["species"] + "EB04.gda"
        xgrid, zgrid, eb04 = read_2d_fields(pic_info, fname, **kwargs)
        prho = eb02 + eb03 + eb04
        text2 = r"$5000<\gamma<20000$"

        fname = plot_config["run_dir"] + "data/Ay.gda"
        xgrid, zgrid, Ay = read_2d_fields(pic_info, fname, **kwargs)
        smime = math.sqrt(pic_info.mime)
        xgrid *= smime
        zgrid *= smime
        xlim = np.asarray([kwargs["xl"], kwargs["xr"]]) * smime
        ylim = np.asarray([kwargs["zb"], kwargs["zt"]]) * smime
        sizes = [xgrid[0], xgrid[-1], zgrid[0], zgrid[-1]]

        ax1 = fig.add_axes(rect)
        img = ax1.imshow(absB/2000, extent=sizes, aspect='auto',
                         # vmin=0, vmax=500,
                         norm = LogNorm(vmin=0.005, vmax=0.25),
                         cmap=plt.cm.inferno, origin='lower')
        Ay_min = np.min(Ay)
        Ay_max = np.max(Ay)
        levels = np.linspace(Ay_min, Ay_max, 20)
        ax1.contour(xgrid, zgrid, Ay, colors='k',
                    levels=levels, linewidths=0.5)
        ax1.set_xlim(xlim)
        ax1.set_ylim(ylim)
        # ax1.set_yticks(np.linspace(-3000, 3000, num=5))
        ax1.set_yticks(np.linspace(-1400, 1400, num=5))
        # ax1.set_yticks(np.linspace(-2000, 2000, num=5))
        ax1.tick_params(labelsize=16)
        if iframe == 0:
            ax1.set_ylabel(r'$z/d_e$', fontsize=20)
        else:
            ax1.tick_params(axis='y', labelleft='off')
        ax1.tick_params(axis='x', labelbottom='off')

        twpi = pic_info.dtwpe * pic_info.fields_interval * tframe / smime
        tlc = "{%0.2f}" % (twpi / pic_info.lx_di)
        title = r"$" + tlc + r"\tau_\text{lc}$"
        plt.title(title, fontsize=24)

        rect1 = np.copy(rect)
        rect1[0] += rect1[2] + 0.01
        rect1[2] = 0.02
        if iframe == 2:
            cax1 = fig.add_axes(rect1)
            cbar1 = fig.colorbar(img, cax=cax1)
            cbar1.ax.tick_params(labelsize=16)
            cbar1.ax.set_title(r"$G$", fontsize=24)
        if iframe == 0:
            ax1.text(0.02, 0.88, r"$|B|$", color='k', fontsize=24,
                     bbox=dict(facecolor='none', alpha=1.0,
                               edgecolor='none', pad=10.0),
                     horizontalalignment='left', verticalalignment='center',
                     transform=ax1.transAxes)


        rect[1] -= rect[3] + vgap
        ax2 = fig.add_axes(rect)
        img = ax2.imshow(prho, extent=sizes, aspect='auto',
                         cmap=plt.cm.viridis, origin='lower',
                         norm = LogNorm(vmin=1, vmax=100))
        ax2.contour(xgrid, zgrid, Ay, colors='k',
                    levels=levels, linewidths=0.5)
        ax2.set_xlim(xlim)
        ax2.set_ylim(ylim)
        # ax2.set_yticks(np.linspace(-3000, 3000, num=5))
        ax2.set_yticks(np.linspace(-1400, 1400, num=5))
        # ax2.set_yticks(np.linspace(-2000, 2000, num=5))
        ax2.tick_params(labelsize=16)
        if iframe == 0:
            ax2.set_ylabel(r'$z/d_e$', fontsize=20)
        else:
            ax2.tick_params(axis='y', labelleft='off')
        ax2.tick_params(axis='x', labelbottom='off')
        rect2 = np.copy(rect)
        rect2[0] += rect2[2] + 0.01
        rect2[2] = 0.02
        if iframe == 2:
            cax2 = fig.add_axes(rect2)
            cbar2 = fig.colorbar(img, cax=cax2)
            cbar2.ax.tick_params(labelsize=16)
        if iframe == 0:
            ax2.text(0.02, 0.88, text2, color='k', fontsize=24,
                     bbox=dict(facecolor='none', alpha=1.0,
                               edgecolor='none', pad=10.0),
                     horizontalalignment='left', verticalalignment='center',
                     transform=ax2.transAxes)

        rect[1] -= rect[3] + vgap
        ax3 = fig.add_axes(rect)
        plot_config["tframe"] = tframe
        im_pol = contour_radiation(plot_config, pic_info, ax3)
        ax3.contour(xgrid, zgrid, Ay, colors='k',
                    levels=levels, linewidths=0.5)
        ax3.set_xlim(xlim)
        ax3.set_ylim(ylim)
        # ax3.set_xticks(np.linspace(0, 15000, num=4))
        # ax3.set_yticks(np.linspace(-3000, 3000, num=5))
        ax3.set_xticks(np.linspace(12500, 15500, num=4))
        ax3.set_yticks(np.linspace(-1400, 1400, num=5))
        # ax3.set_xticks(np.linspace(0, 8000, num=4))
        # ax3.set_yticks(np.linspace(-2000, 2000, num=5))
        ax3.set_xlabel(r'$x/d_e$', fontsize=20)
        if iframe == 0:
            ax3.set_ylabel(r'$z/d_e$', fontsize=20)
        else:
            ax3.set_ylabel('', fontsize=20)
            ax3.tick_params(axis='y', labelleft='off')
        rect[0] += rect[2] + 0.01
        rect[2] = 0.02
        if iframe == 2:
            cax3 = fig.add_axes(rect)
            cbar3 = fig.colorbar(im_pol, cax=cax3)
            cbar3.ax.tick_params(labelsize=16)
            cbar3.ax.set_yticklabels([r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$',
                                      r'$10^0$'], fontsize=16)
        if iframe == 0:
            ax3.text(0.02, 0.88, 'Optical', color='k', fontsize=24,
                     bbox=dict(facecolor='none', alpha=1.0,
                               edgecolor='none', pad=10.0),
                     horizontalalignment='left', verticalalignment='center',
                     transform=ax3.transAxes)

    fdir = ('../img/radiation_cooling/absb_rad_map/' +
            plot_config["run_name"] + '/tri_times' + '/')
    mkdir_p(fdir)
    tfs = ""
    for tframe in plot_config["tframes"]:
        tfs += str(tframe) + "_"
    fname = (fdir + 'rad_map' + tfs + str(plot_config["obs_ang"]) + ".jpg")
    fig.savefig(fname, dpi=200)

    if show_plot:
        plt.show()
    else:
        plt.close()


def get_energy_bins(run_dir, source_file):
    """Get energy bins information

    Args:
        run_dir: the run directory of the PIC simulation
        source_file: file including these information
    """
    fname = run_dir + source_file
    nbins = 1000
    emin = 1E-3
    emax = 1E7
    with open(fname, 'r') as f:
        content = f.readlines()
        f.close()
        nlines = len(content)
        current_line = 0
        while not 'int nbin =' in content[current_line]:
            current_line += 1
        single_line = content[current_line]
        line_splits = single_line.split("=")
        word_splits = line_splits[1].split(";")
        nbins = int(word_splits[0])
        current_line += 1
        single_line = content[current_line]
        line_splits = single_line.split("=")
        word_splits = line_splits[1].split(";")
        emin = float(word_splits[0])
        current_line += 1
        single_line = content[current_line]
        line_splits = single_line.split("=")
        word_splits = line_splits[1].split(";")
        emax = float(word_splits[0])
    return (nbins, emin, emax)


def reduce_spectrum(plot_config):
    """Reduce energy spectrum from the binary files for each MPI rank

    Args:
        plot_config: plot configuration
    """
    run_name = plot_config["run_name"]
    run_dir = plot_config["run_dir"]
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    rank = 0
    interval = pic_info.fields_interval
    mpi_size = pic_info.topology_x * pic_info.topology_y * pic_info.topology_z

    nbins = plot_config["nbins"]
    ndata = nbins + 3  # including bx, by, bz
    tindex = plot_config["tframe"] * interval
    species = plot_config["species"]
    fname_pre = run_dir + 'hydro/T.' + str(tindex)
    if species in ['e', 'electron']:
        fname_pre = fname_pre + '/spectrum-ehydro.' + str(tindex)
    else:
        fname_pre = fname_pre + '/spectrum-Hhydro.' + str(tindex)
    fname = fname_pre + '.' + str(rank)
    fdata = np.fromfile(fname, dtype=np.float32)
    sz, = fdata.shape
    nzone = sz // ndata
    for rank in range(1, mpi_size):
        fname = fname_pre + '.' + str(rank)
        fdata += np.fromfile(fname, dtype=np.float32)
    print("number of zones: %d" % nzone)
    flog_tot = np.zeros(nbins)
    for i in range(nzone):
        flog = fdata[i*ndata+3:(i+1)*ndata]
        flog_tot += flog
    emin_log = math.log10(plot_config["emin"])
    emax_log = math.log10(plot_config["emax"])
    dloge = (emax_log - emin_log) / (nbins - 1)
    emin_log_adjust = emin_log - dloge
    elog = np.logspace(emin_log_adjust, emax_log, nbins + 1)
    elog_mid = 0.5 * (elog[1:] + elog[:-1])
    delog = np.diff(elog)
    flog_tot /= delog
    fdir = '../data/spectra/' + run_name + '/'
    mkdir_p(fdir)
    if species in ['e', 'electron']:
        fname = fdir + 'spectrum-e.' + str(plot_config["tframe"])
    else:
        fname = fdir + 'spectrum-H.' + str(plot_config["tframe"])
    flog_tot.tofile(fname)


def plot_spectrum(plot_config):
    """Plot energy spectrum
    """
    emin_log = math.log10(plot_config["emin"])
    emax_log = math.log10(plot_config["emax"])
    nbins = plot_config["nbins"]
    dloge = (emax_log - emin_log) / (nbins - 1)
    emin_log_adjust = emin_log - dloge
    elog = np.logspace(emin_log_adjust, emax_log, nbins + 1)
    elog_mid = 0.5 * (elog[1:] + elog[:-1])
    fig = plt.figure(figsize=[7, 5])
    xs, ys = 0.15, 0.15
    w1, h1 = 0.8, 0.8
    ax = fig.add_axes([xs, ys, w1, h1])
    nframes = plot_config["nframes"]
    fdir = '../data/spectra/' + plot_config["run_name"] + '/'
    species = plot_config["species"]
    for tframe in range(0, nframes, 3):
        if species in ['e', 'electron']:
            fname = fdir + 'spectrum-e.' + str(tframe)
        else:
            fname = fdir + 'spectrum-H.' + str(tframe)
        flog = np.fromfile(fname)
        color = plt.cm.jet(tframe / float(nframes), 1)
        ax.loglog(elog_mid, flog, linewidth=2, color=color)

    tframe = nframes - 1
    if species in ['e', 'electron']:
        fname = fdir + 'spectrum-e.' + str(tframe)
    else:
        fname = fdir + 'spectrum-H.' + str(tframe)
    flog = np.fromfile(fname)
    color = plt.cm.jet(tframe / float(nframes), 1)
    # ax.loglog(elog_mid*5, flog*10, linewidth=2, color='k')

    ax.tick_params(labelsize=16)
    ax.tick_params(axis='x', which='minor', direction='in', top=True)
    ax.tick_params(axis='x', which='major', direction='in', top=True)
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', which='major', direction='in', right=True)
    ax.set_xlabel(r'$\gamma-1$', fontsize=20)
    ax.set_ylabel(r'$f(\gamma-1)$', fontsize=20)
    ax.set_xlim([1E1, 5E5])
    ax.set_ylim([1E-3, 1E7])
    fdir = '../img/radiation_cooling/spectra/'
    mkdir_p(fdir)
    fname = fdir + 'spect_time_' + plot_config["run_name"] + '_' + species + '.pdf'
    fig.savefig(fname)
    plt.show()


def get_nz_local(run_dir, source_file):
    """Get the number of cells along z for each local zone

    Args:
        run_dir: the run directory of the PIC simulation
        source_file: file including these information
    """
    fname = run_dir + source_file
    nz_local = 16
    with open(fname, 'r') as f:
        content = f.readlines()
        f.close()
        nlines = len(content)
        current_line = 0
        while not 'int nz_local =' in content[current_line]:
            current_line += 1
        single_line = content[current_line]
        line_splits = single_line.split("=")
        word_splits = line_splits[1].split(";")
        nz_local = int(word_splits[0])
    return nz_local


def spect_bfield_3dpol_new(plot_config):
    """Local spectrum and magnetic field for the new version of 3DPol
    """
    run_name = plot_config["run_name"]
    run_dir = plot_config["run_dir"]
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    interval = pic_info.fields_interval
    mpi_sizex = pic_info.topology_x
    mpi_sizey = pic_info.topology_y
    mpi_sizez = pic_info.topology_z
    mpi_size = mpi_sizex * mpi_sizey * mpi_sizez
    nbins = plot_config["nbins"]
    tframe = plot_config["tframe"]
    tindex = tframe * interval
    species = plot_config["species"]
    nz_local = plot_config["nz_local"]
    reduce_factor_z = plot_config["reduce_factor_z"]
    # We assume nz is dividable by (nz_local * reduce_factor_z)
    nradz = pic_info.nz // (nz_local * reduce_factor_z)
    nradx = pic_info.nx * nradz // pic_info.nz
    reduce_factor_x  = mpi_sizex // nradx
    nradz_local = nradz // mpi_sizez  # number of zones along z in each MPI rank

    emin_log = math.log10(plot_config["emin"])
    emax_log = math.log10(plot_config["emax"])
    dloge = (emax_log - emin_log) / (nbins - 1)
    emin_log_adjust = emin_log - dloge
    elog = np.logspace(emin_log_adjust, emax_log, nbins + 1)
    elog_mid = 0.5 * (elog[1:] + elog[:-1])
    delog = np.diff(elog)

    ndata = nbins + 3  # including bx, by, bz
    tindex = tframe * interval
    fname_pre = run_dir + 'hydro/T.' + str(tindex)
    fname_pre = fname_pre + '/spectrum-ehydro.' + str(tindex)
    fname = fname_pre + '.0'
    fdata = np.fromfile(fname, dtype=np.float32)
    sz, = fdata.shape
    nzone = sz // ndata
    output_dir = '../data/data-radiation-3dpol-new/' + run_name + '/'
    mkdir_p(output_dir)
    fname0 = output_dir + "radiation-"
    cit = str(tframe).zfill(3)
    ciy = str(1).zfill(3)

    nbins_skip = 300  # Skip some bins that have no particles
    nbins_target = 100  # Assume we do not need all the bins

    nreduce = (nbins - nbins_skip) // nbins_target
    frad = np.zeros([nradz, nradx, nbins_target + 6])
    bx   = np.zeros([nradz, nradx])
    by   = np.zeros([nradz, nradx])
    bz   = np.zeros([nradz, nradx])

    for mpi_rankz in range(mpi_sizez):
        for mpi_rankx in range(mpi_sizex):
            ix = mpi_rankx // reduce_factor_x
            mpi_rank = mpi_rankx + mpi_rankz * mpi_sizex
            fname = fname_pre + '.' + str(mpi_rank)
            fdata = np.fromfile(fname, dtype=np.float32)
            for zone_z in range(nzone):
                iz = zone_z // reduce_factor_z + mpi_rankz * nradz_local
                bx[iz, ix] += fdata[zone_z * ndata]
                by[iz, ix] += fdata[zone_z * ndata + 1]
                bz[iz, ix] += fdata[zone_z * ndata + 2]
                flog  = fdata[zone_z*ndata+3:(zone_z+1)*ndata]
                flog /= delog
                frad[iz, ix, 6:] += flog[nbins_skip::nreduce]

    bx /= reduce_factor_x * reduce_factor_z
    by /= reduce_factor_x * reduce_factor_z
    bz /= reduce_factor_x * reduce_factor_z
    bmag = np.sqrt(bx**2 + by**2 + bz**2)
    btheta = np.arccos(bz/bmag)
    bphi = np.arctan(by/bx)
    frad[:, :, 0] = bmag
    frad[:, :, 1] = btheta
    frad[:, :, 2] = bphi
    frad[:, :, 3] = 1.0
    frad[:, :, 4] = 0.0
    frad[:, :, 5] = 0.0
    fname = fname0 + cit
    frad.tofile(fname)


def get_cmd_args():
    """Get command line arguments
    """
    default_run_name = 'sigma4E4_bg00_rad_vthe100_cool100_b'
    default_run_dir = ('/net/scratch2/xiaocanli/vpic_radiation/reconnection/' +
                       'grizzly/cooling_scaling_16000_8000/' +
                       default_run_name + '/')
    default_map_dir = default_run_dir + 'map/'
    parser = argparse.ArgumentParser(description='Radiation cooling analysis')
    parser.add_argument('--species', action="store", default='e',
                        help='particle species')
    parser.add_argument('--run_dir', action="store", default=default_run_dir,
                        help='run directory')
    parser.add_argument('--run_name', action="store", default=default_run_name,
                        help='run name')
    parser.add_argument('--tframe', action="store", default='30', type=int,
                        help='Time frame')
    parser.add_argument('--multi_frames', action="store_true", default=False,
                        help='whether to analyze multiple frames')
    parser.add_argument('--tstart', action="store", default='0', type=int,
                        help='Starting time frame')
    parser.add_argument('--tend', action="store", default='10', type=int,
                        help='Ending time frame')
    parser.add_argument('--bfield_single', action="store_true", default=False,
                        help='plot magnetic field for a single time step')
    parser.add_argument('--density_eband', action="store_true", default=False,
                        help='plot particle density at different energy band')
    parser.add_argument('--dist_2d', action="store_true", default=False,
                        help='plot 2D contour particle distributions')
    parser.add_argument('--momentum_flux', action="store_true", default=False,
                        help='plot 2D contour particle distributions')
    parser.add_argument('--radiation_map', action="store_true", default=False,
                        help='plot 2D contour of magnetic field and radiation')
    parser.add_argument('--radiation_map_tri', action="store_true", default=False,
                        help=('plot 2D contour of magnetic field and ' +
                              'radiation for three time frames'))
    parser.add_argument('--map_dir', action="store", default=default_map_dir,
                        help='radiation map directory')
    parser.add_argument("--energy_band", action="store", default='1', type=int,
                        help="Energy band")
    parser.add_argument("--old_rad", action="store_true", default=False,
                        help='whether it is old radiation file')
    parser.add_argument("--flux_min", action="store", default='1E-8',
                        type=float, help="minimum total flux to plot")
    parser.add_argument("--flux_max", action="store", default='1E-5',
                        type=float, help="maximum total flux to plot")
    parser.add_argument("--pflux_scale", action="store", default='0.1',
                        type=float, help="polarization flux scale factor")
    parser.add_argument('--obs_ang', action="store", default='0', type=int,
                        help='observation angle')
    parser.add_argument('--reduce_spect', action="store_true", default=False,
                        help='whether to reduce particle energy spectrum')
    parser.add_argument('--plot_spect', action="store_true", default=False,
                        help='whether to plot particle energy spectrum')
    parser.add_argument('--data_3dpol', action="store_true", default=False,
                        help='whether to get data for 3DPol')
    parser.add_argument("--reduce_factor_z", action="store", default='1',
                        type=int, help="Reduce factor along z for local spectrum")
    return parser.parse_args()


def analysis_single_frame(args, plot_config):
    """Analysis for single time frame
    """
    run_dir = plot_config["run_dir"]
    run_name = plot_config["run_name"]
    species = plot_config["species"]
    plot_config["tframe"] = args.tframe
    plot_config["energy_band"] = args.energy_band
    plot_config["old_rad"] = args.old_rad
    plot_config["flux_range"] = [args.flux_min, args.flux_max],
    plot_config["pflux_scale"] = args.pflux_scale
    plot_config["obs_ang"] = args.obs_ang
    plot_config["map_dir"] = args.map_dir
    plot_config["nframes"] = args.tend - args.tstart + 1
    if args.bfield_single:
        plot_bfield_single(run_dir, run_name, args.tframe, show_plot=True)
    if args.density_eband:
        plot_density_eband(run_dir, run_name, args.tframe, species)
    if args.dist_2d:
        plot_dist_2d(run_dir, run_name, args.tframe)
    if args.momentum_flux:
        plot_momentum_flux(run_dir, run_name, args.tframe)
    if args.radiation_map:
        radiation_map(plot_config, show_plot=True)
    if args.radiation_map_tri:
        # plot_config["xrange"] = [0.0, 1.0]
        # plot_config["zrange"] = [0.0, 1.0]
        # plot_config["tframes"] = [2, 30, 130]
        plot_config["xrange"] = [0.75, 1.0]
        plot_config["zrange"] = [0.325, 0.675]
        plot_config["tframes"] = [40, 47, 51]
        # plot_config["xrange"] = [0, 0.5]
        # plot_config["zrange"] = [0.25, 0.75]
        # plot_config["tframes"] = [55, 69, 75]
        radiation_map_tri(plot_config, show_plot=True)
    if args.reduce_spect:
        reduce_spectrum(plot_config)
    if args.plot_spect:
        plot_spectrum(plot_config)
    if args.data_3dpol:
        spect_bfield_3dpol_new(plot_config)


def process_input(args, plot_config, tframe):
    """process one time frame"""
    run_dir = plot_config["run_dir"]
    run_name = plot_config["run_name"]
    species = plot_config["species"]
    plot_config["tframe"] = tframe
    if args.bfield_single:
        plot_bfield_single(run_dir, run_name, tframe, show_plot=False)
    if args.density_eband:
        plot_density_eband(run_dir, run_name, tframe, species)
    if args.dist_2d:
        plot_dist_2d(run_dir, run_name, tframe)
    if args.radiation_map:
        plot_config["energy_band"] = args.energy_band
        plot_config["old_rad"] = args.old_rad
        plot_config["flux_range"] = [args.flux_min, args.flux_max],
        plot_config["pflux_scale"] = args.pflux_scale
        plot_config["obs_ang"] = args.obs_ang
        plot_config["map_dir"] = args.map_dir
        radiation_map(plot_config, show_plot=False)
    if args.reduce_spect:
        reduce_spectrum(plot_config)
    if args.data_3dpol:
        spect_bfield_3dpol_new(plot_config)


def analysis_multi_frames(args, plot_config):
    """Analysis for multiple time frames
    """
    tframes = range(args.tstart, args.tend + 1)
    ncores = multiprocessing.cpu_count()
    ncores = 36
    Parallel(n_jobs=ncores)(delayed(process_input)(args, plot_config, tframe)
                            for tframe in tframes)


def main():
    """business logic for when running this module as the primary one!"""
    args = get_cmd_args()
    plot_config = {}
    plot_config["run_name"] = args.run_name
    plot_config["run_dir"] = args.run_dir
    plot_config["species"] = args.species
    nbins, emin, emax = get_energy_bins(args.run_dir, "energy_local.cxx")
    plot_config["nbins"] = nbins
    plot_config["emin"] = emin
    plot_config["emax"] = emax
    plot_config["nz_local"] = get_nz_local(args.run_dir, "energy_local.cxx")
    plot_config["reduce_factor_z"] = args.reduce_factor_z
    picinfo_fname = '../data/pic_info/pic_info_' + args.run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    if args.multi_frames:
        analysis_multi_frames(args, plot_config)
    else:
        analysis_single_frame(args, plot_config)


if __name__ == "__main__":
    main()
