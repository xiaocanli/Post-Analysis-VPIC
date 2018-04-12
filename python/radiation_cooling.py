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

style.use(['seaborn-white', 'seaborn-paper'])
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

    vmax1 = 3.0E2
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
                         vmin=10, vmax=300, cmap1=plt.cm.viridis)
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

    # fname = plot_config["run_dir"] + "data/" + plot_config["species"] + "EB02.gda"
    # xgrid, zgrid, eb02 = read_2d_fields(pic_info, fname, **kwargs)
    # fname = plot_config["run_dir"] + "data/" + plot_config["species"] + "EB03.gda"
    # xgrid, zgrid, eb03 = read_2d_fields(pic_info, fname, **kwargs)
    # fname = plot_config["run_dir"] + "data/" + plot_config["species"] + "EB04.gda"
    # xgrid, zgrid, eb04 = read_2d_fields(pic_info, fname, **kwargs)
    # prho = eb02 + eb03 + eb04
    # text2 = r"$5000<\gamma<20000$"

    # fname = plot_config["run_dir"] + "data/" + plot_config["species"] + "EB06.gda"
    # xgrid, zgrid, eb06 = read_2d_fields(pic_info, fname, **kwargs)
    # prho = eb06
    # text2 = r"$2500<\gamma<3000$"

    fname = plot_config["run_dir"] + "data/" + plot_config["species"] + "EB04.gda"
    xgrid, zgrid, eb04 = read_2d_fields(pic_info, fname, **kwargs)
    prho = eb04
    text2 = r"$8000<\gamma<16000$"

    fname = plot_config["run_dir"] + "data/Ay.gda"
    xgrid, zgrid, Ay = read_2d_fields(pic_info, fname, **kwargs)
    smime = math.sqrt(pic_info.mime)
    xgrid *= smime
    zgrid *= smime
    sizes = [xgrid[0], xgrid[-1], zgrid[0], zgrid[-1]]

    fig = plt.figure(figsize=[12, 14])
    rect = [0.10, 0.66, 0.8, 0.28]
    ax1 = fig.add_axes(rect)
    img = ax1.imshow(absB, extent=sizes, aspect='auto',
                     cmap=plt.cm.inferno,
                     origin='lower', vmin=10, vmax=300)
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
    xmin, xmax = 0, lx_de
    zmin, zmax = -0.5 * lz_de, 0.5 * lz_de
    vmin = np.min(tot_flux)
    vmax = np.max(tot_flux)
    print("Min and max of total flux: %e %e" % (vmin, vmax))
    print("Min and max of polarization angle: %e %e" %
          (np.min(pol_angl), np.max(pol_angl)))
    print("Min and max of polarization flux %e %e" %
          (np.min(pol_flux), np.max(pol_flux)))
    vmin, vmax = plot_config["flux_range"][0]
    p1 = ax.imshow(tot_flux.T, cmap=plt.cm.Oranges,
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
    nx_rad, nz_rad = tot_flux.shape
    x_rad = np.linspace(0, lx_de, nx_rad)
    z_rad = np.linspace(-0.5 * lz_de, 0.5 * lz_de, nz_rad)
    X, Z = np.meshgrid(x_rad, z_rad)
    U = np.transpose(pol_flux*np.sin(pol_angl*math.pi/180))
    V = np.transpose(pol_flux*np.cos(pol_angl*math.pi/180))
    s = 1
    Q = ax.quiver(X[::s, ::s], Z[::s, ::s], U[::s, ::s], V[::s, ::s],
                  **quiveropts)

    return p1


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
    return parser.parse_args()


def analysis_single_frame(args, plot_config):
    """Analysis for single time frame
    """
    run_dir = plot_config["run_dir"]
    run_name = plot_config["run_name"]
    species = plot_config["species"]
    if args.bfield_single:
        plot_bfield_single(run_dir, run_name, args.tframe, show_plot=True)
    if args.density_eband:
        plot_density_eband(run_dir, run_name, args.tframe, species)
    if args.dist_2d:
        plot_dist_2d(run_dir, run_name, args.tframe)
    if args.momentum_flux:
        plot_momentum_flux(run_dir, run_name, args.tframe)
    if args.radiation_map:
        plot_config["tframe"] = args.tframe
        plot_config["energy_band"] = args.energy_band
        plot_config["old_rad"] = args.old_rad
        plot_config["flux_range"] = [args.flux_min, args.flux_max],
        plot_config["pflux_scale"] = args.pflux_scale
        plot_config["obs_ang"] = args.obs_ang
        plot_config["map_dir"] = args.map_dir
        radiation_map(plot_config, show_plot=True)


def process_input(args, plot_config, tframe):
    """process one time frame"""
    run_dir = plot_config["run_dir"]
    run_name = plot_config["run_name"]
    species = plot_config["species"]
    if args.bfield_single:
        plot_bfield_single(run_dir, run_name, tframe)
    if args.density_eband:
        plot_density_eband(run_dir, run_name, tframe, species)
    if args.dist_2d:
        plot_dist_2d(run_dir, run_name, tframe)
    if args.radiation_map:
        plot_config["tframe"] = tframe
        plot_config["energy_band"] = args.energy_band
        plot_config["old_rad"] = args.old_rad
        plot_config["flux_range"] = [args.flux_min, args.flux_max],
        plot_config["pflux_scale"] = args.pflux_scale
        plot_config["obs_ang"] = args.obs_ang
        plot_config["map_dir"] = args.map_dir
        radiation_map(plot_config, show_plot=False)


def analysis_multi_frames(args, plot_config):
    """Analysis for multiple time frames
    """
    tframes = range(args.tstart, args.tend + 1)
    ncores = multiprocessing.cpu_count()
    ncores = 16
    Parallel(n_jobs=ncores)(delayed(process_input)(args, plot_config, tframe)
                            for tframe in tframes)


def main():
    """business logic for when running this module as the primary one!"""
    args = get_cmd_args()
    plot_config = {}
    plot_config["run_name"] = args.run_name
    plot_config["run_dir"] = args.run_dir
    plot_config["species"] = args.species
    picinfo_fname = '../data/pic_info/pic_info_' + args.run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    if args.multi_frames:
        analysis_multi_frames(args, plot_config)
    else:
        analysis_single_frame(args, plot_config)


if __name__ == "__main__":
    main()
