"""
Analysis procedures for particle energy spectrum.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy import signal
import math
import os.path
from os import listdir
from os.path import isfile, join
import struct
import collections
import pic_information
from contour_plots import read_2d_fields, plot_2d_contour
from particle_distribution import *
from spectrum_fitting import get_normalized_energy, fit_thermal_core

mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

font = {'family' : 'serif',
        #'color'  : 'darkred',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 24,
        }

class EspectrumVdist(object):
    def __init__(self, **kwargs):
        """
        Args:
            kwargs_dist: arguments for getting particle distributions.
        """
        self.var_field = kwargs['var_field']
        self.var_name = kwargs['var_name']
        self.ct_ptl = kwargs['ct_ptl']
        self.field_range = kwargs['field_range']
        self.species = kwargs['species']
        self.center = kwargs['center']
        self.sizes = kwargs['sizes']
        self.nbins = kwargs['nbins']
        self.vmin = kwargs['vmin']
        self.vmax = kwargs['vmax']
        self.fpath_vdist = kwargs['fpath_vdist']
        self.fpath_spect = kwargs['fpath_spect']
        self.get_kwargs_dist()
        self.pic_info = pic_information.get_pic_info('../../')
        self.smime = math.sqrt(pic_info.mime)
        self.get_box_coords()
        self.get_dists_info()
        self.ratio_ptl_fields = \
                pic_info.particle_interval / pic_info.fields_interval
        self.ct_field = self.ratio_ptl_fields * self.ct_ptl
        self.elog_norm = get_normalized_energy(self.species, self.elog,
                self.pic_info)
        self.fit_nonthermal()
        self.read_field_data()
        self.distribution_plot()
        self.fig_dist.canvas.mpl_connect('button_press_event',self.click)

    def get_dists_info(self):
        """Get the distribution information
        """
        get_spectrum_vdist(self.pic_info, **self.kwargs_dist)
        self.read_distributions()
        self.vbins_short = self.fvel.vbins_short
        self.vbins_long = self.fvel.vbins_long
        self.elin = self.fene.elin
        self.elog = self.fene.elog
        self.dmin = self.fvel.vmin
        self.dmax = self.fvel.vmax
        self.vmax_2d = self.fvel.vmax_2d
        self.vmin_2d = self.fvel.vmin_2d
        self.vmax_1d = self.fvel.vmax_1d
        self.vmin_1d = self.fvel.vmin_1d

    def get_kwargs_dist(self):
        """Get arguments for getting particle distributions
        """
        self.kwargs_dist = {'center':center, 'sizes':sizes, 'nbins':self.nbins,
                'vmin':self.vmin, 'vmax':self.vmax, 'tframe':self.ct_ptl,
                'species':self.species}

    def read_distributions(self):
        """Read velocity and energy distributions
        """
        self.fvel = read_velocity_distribution(self.species, self.ct_ptl,
                self.pic_info, self.fpath_vdist)
        self.fene = read_energy_distribution(self.species, self.ct_ptl,
                self.pic_info, self.fpath_spect)

    def get_box_coords(self):
        """Get the coordinates to plot the box
        """
        self.center_di = self.kwargs_dist['center'] / smime
        self.sizes_di = self.kwargs_dist['sizes'] * self.pic_info.dx_di
        self.bxl = self.center_di[0] - self.sizes_di[0] * 0.5
        self.bxh = self.center_di[0] + self.sizes_di[0] * 0.5
        self.bzl = self.center_di[2] - self.sizes_di[2] * 0.5
        self.bzh = self.center_di[2] + self.sizes_di[2] * 0.5
        self.xbox = [self.bxl, self.bxh, self.bxh, self.bxl, self.bxl]
        self.zbox = [self.bzl, self.bzl, self.bzh, self.bzh, self.bzl]

    def read_field_data(self):
        """Read 2D field
        """
        self.xl, self.xr = self.field_range[0:2]
        self.zb, self.zt = self.field_range[2:4]
        kwargs = {"current_time":self.ct_field, "xl":self.xl, "xr":self.xr,
                "zb":self.zb, "zt":self.zt}
        fname_field = '../../data/' + self.var_field + '.gda'
        fname_Ay = '../../data/Ay.gda'
        self.x1, self.z1, data = read_2d_fields(self.pic_info, fname_field,
                **kwargs)
        ng = 5
        kernel = np.ones((ng,ng)) / float(ng*ng)
        self.fdata1 = signal.convolve2d(data, kernel)
        self.x1, self.z1, self.Ay1 = read_2d_fields(self.pic_info,
                fname_Ay, **kwargs)
        self.nx1, = self.x1.shape
        self.nz1, = self.z1.shape
        self.xmax1 = np.max(self.x1)
        self.zmax1 = np.max(self.z1)
        self.xmin1 = np.min(self.x1)
        self.zmin1 = np.min(self.z1)

    def distribution_plot(self):
        self.fig_width, self.fig_height = 10, 10
        self.fig_dist = plt.figure(figsize=(self.fig_width, self.fig_height))
        self.cmap = plt.cm.get_cmap('coolwarm')
        self.cmap_dist = plt.cm.get_cmap('jet')
        self.color_Ay = 'black'

        xs, ys = 0.12, 0.7
        w1, h1 = 0.78, 0.25
        w2, h2 = 0.22, 0.22
        h3 = h2 * 0.5
        gaph = 0.10  # Horizontal
        gapv = 0.05  # Vertical
        gapv2 = 0.06  # Vertical
        gaph2 = 0.02
        gapv3 = 0.02
        ye = ys + h1
        xs1 = 0.08
        self.xz_axis = self.fig_dist.add_axes([xs, ys, w1, h1])
        self.vmax = max(abs(np.min(self.fdata1)), abs(np.max(self.fdata1)))
        self.vmax *= 0.5
        self.vmax = 0.05
        xst = zst = 4
        tst = 1
        self.xst = xst
        self.zst = zst
        nx1, nz1 = self.nx1, self.nz1
        self.im1 = self.xz_axis.imshow(self.fdata1[0:nz1:zst, 0:nx1:xst],
                cmap=self.cmap,
                extent=[self.xmin1, self.xmax1, self.zmin1, self.zmax1],
                aspect='auto', origin='lower',
                vmin = -self.vmax, vmax = self.vmax,
                interpolation='bicubic')
        divider = make_axes_locatable(self.xz_axis)
        self.cax = divider.append_axes("right", size="2%", pad=0.05)
        self.cbar = self.fig_dist.colorbar(self.im1, cax=self.cax)
        self.cbar.ax.tick_params(labelsize=16)
        self.cbar.set_ticks(np.arange(-0.04, 0.05, 0.02))
        self.Ay_max = np.max(self.Ay1)
        self.Ay_min = np.min(self.Ay1)
        self.levels = np.linspace(self.Ay_min, self.Ay_max, 10)
        self.xz_axis.contour(self.x1[0:nx1:xst], self.z1[0:nz1:zst],
                self.Ay1[0:nz1:zst, 0:nx1:xst], colors=self.color_Ay,
                linewidths=0.5, levels=self.levels)
        self.xz_axis.tick_params(labelsize=16)
        self.xz_axis.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
        self.xz_axis.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
        self.xz_axis.autoscale(1,'both',1)

        # Plot a box indicating where to get the distributions
        self.pbox, = self.xz_axis.plot(self.xbox, self.zbox, color='k')

        # fvel_xy
        xs2 = xs1
        ys -= h1 + gapv
        self.vxy_axis = self.fig_dist.add_axes([xs2, ys, w2, h2])
        self.pvxy = self.vxy_axis.imshow(self.fvel.fvel_xy,
                cmap=self.cmap_dist,
                extent=[-self.dmax, self.dmax, -self.dmax, self.dmax],
                aspect='auto', origin='lower',
                norm=LogNorm(vmin=self.vmin_2d, vmax=self.vmax_2d),
                interpolation='bicubic')
        self.vxy_axis.plot([0, 0], [-self.dmax, self.dmax], color='w',
                linestyle='--')
        self.vxy_axis.plot([-self.dmax, self.dmax], [0, 0], color='w',
                linestyle='--')
        self.vxy_axis.set_xlabel(r'$u_x$', fontdict=font, fontsize=20)
        self.vxy_axis.set_ylabel(r'$u_y$', fontdict=font, fontsize=20)
        self.vxy_axis.tick_params(labelsize=16)

        # fvel_xz
        xs2 += w2 + gaph
        self.vxz_axis = self.fig_dist.add_axes([xs2, ys, w2, h2])
        self.pvxz = self.vxz_axis.imshow(self.fvel.fvel_xz,
                cmap=self.cmap_dist,
                extent=[-self.dmax, self.dmax, -self.dmax, self.dmax],
                aspect='auto', origin='lower',
                norm=LogNorm(vmin=self.vmin_2d, vmax=self.vmax_2d),
                interpolation='bicubic')
        self.vxz_axis.plot([0, 0], [-self.dmax, self.dmax], color='w',
                linestyle='--')
        self.vxz_axis.plot([-self.dmax, self.dmax], [0, 0], color='w',
                linestyle='--')
        self.vxz_axis.set_xlabel(r'$u_x$', fontdict=font, fontsize=20)
        self.vxz_axis.set_ylabel(r'$u_z$', fontdict=font, fontsize=20)
        self.vxz_axis.tick_params(labelsize=16)

        # fvel_yz
        xs2 += w2 + gaph
        self.vyz_axis = self.fig_dist.add_axes([xs2, ys, w2, h2])
        self.pvyz = self.vyz_axis.imshow(self.fvel.fvel_yz,
                cmap=self.cmap_dist,
                extent=[-self.dmax, self.dmax, -self.dmax, self.dmax],
                aspect='auto', origin='lower',
                norm=LogNorm(vmin=self.vmin_2d, vmax=self.vmax_2d),
                interpolation='bicubic')
        self.vyz_axis.plot([0, 0], [-self.dmax, self.dmax], color='w',
                linestyle='--')
        self.vyz_axis.plot([-self.dmax, self.dmax], [0, 0], color='w',
                linestyle='--')
        self.vyz_axis.set_xlabel(r'$u_y$', fontdict=font, fontsize=20)
        self.vyz_axis.set_ylabel(r'$u_z$', fontdict=font, fontsize=20)
        self.vyz_axis.tick_params(labelsize=16)

        # fvel_para_perp
        xs2 = xs1
        ys -= h3 + gapv2
        self.v2d_axis = self.fig_dist.add_axes([xs2, ys, w2, h3])
        self.pv2d = self.v2d_axis.imshow(self.fvel.fvel_para_perp,
                cmap=self.cmap_dist,
                extent=[-self.dmax, self.dmax, 0, self.dmax],
                aspect='auto', origin='lower',
                norm=LogNorm(vmin=self.vmin_2d, vmax=self.vmax_2d),
                interpolation='bicubic')
        self.v2d_axis.plot([0, 0], [0, self.dmax], color='w', linestyle='--')
        self.v2d_axis.set_ylabel(r'$u_\perp$', fontdict=font, fontsize=20)
        self.v2d_axis.tick_params(labelsize=16)
        self.v2d_axis.tick_params(axis='x', labelbottom='off')

        # fvel_perp
        xs2 += w2 + gaph2
        self.vperp_axis = self.fig_dist.add_axes([xs2, ys, w2, h3])
        self.pvperp, = self.vperp_axis.semilogx(self.fvel.fvel_perp,
                self.vbins_short, color='k', linewidth=2)
        self.vperp_axis.set_yticks(np.arange(0, self.dmax+0.1, 1.0))
        self.vperp_axis.set_xlabel(r'$f(u_\perp)$', fontdict=font, fontsize=20)
        self.vperp_axis.tick_params(axis='y', labelleft='off')
        self.vperp_axis.tick_params(labelsize=16)

        # fvel_para
        ys -= h3 + gapv2
        xs2 = xs1
        self.vpara_axis = self.fig_dist.add_axes([xs2, ys, w2, h3])
        self.pvpara, = self.vpara_axis.semilogy(self.vbins_long,
                self.fvel.fvel_para, color='k', linewidth=2)
        self.vpara_axis.set_xticks(np.arange(-self.dmax, self.dmax+0.1, 1.0))
        self.vpara_axis.set_xlabel(r'$u_\parallel$', fontdict=font, fontsize=20)
        self.vpara_axis.set_ylabel(r'$f(u_\parallel)$', fontdict=font, fontsize=20)
        self.vpara_axis.tick_params(labelsize=16)

        # energy spectrum
        xs2 += (w2 + gaph2)*2 + gaph
        xe = xs1 + 2 * gaph + 3 * w2
        w3 = xe - xs2
        h4 = h3 * 2 + gapv2
        self.espect_axis = self.fig_dist.add_axes([xs2, ys, w3, h4])
        self.pespect, = self.espect_axis.loglog(self.elog_norm,
                self.fene.flog, color='k', linewidth=2)
        self.espect_axis.set_xlabel(r'$\varepsilon/\varepsilon_\text{th}$',
                fontdict=font, fontsize=20)
        self.espect_axis.set_ylabel(r'$f(\varepsilon)$',
                fontdict=font, fontsize=20)
        self.espect_axis.tick_params(labelsize=16)
        color = self.pespect.get_color()
        self.pnth, = self.espect_axis.loglog(self.elog_norm, self.fnonthermal,
                color=color)

        # set limes
        self.set_axes_lims()

    def fit_nonthermal(self):
        self.fthermal = fit_thermal_core(self.elog, self.fene.flog)
        self.fnonthermal = self.fene.flog - self.fthermal

    def set_axes_lims(self):
        self.vxy_axis.set_xticks(np.arange(-self.dmax, self.dmax+0.1, 1.0))
        self.vxy_axis.set_yticks(np.arange(-self.dmax, self.dmax+0.1, 1.0))
        self.vxz_axis.set_xticks(np.arange(-self.dmax, self.dmax+0.1, 1.0))
        self.vxz_axis.set_yticks(np.arange(-self.dmax, self.dmax+0.1, 1.0))
        self.vyz_axis.set_xticks(np.arange(-self.dmax, self.dmax+0.1, 1.0))
        self.vyz_axis.set_yticks(np.arange(-self.dmax, self.dmax+0.1, 1.0))
        self.v2d_axis.set_xticks(np.arange(-self.dmax, self.dmax+0.1, 1.0))
        self.v2d_axis.set_yticks(np.arange(0, self.dmax+0.1, 1.0))
        self.vpara_axis.set_xlim([-self.dmax, self.dmax])
        self.vperp_axis.set_ylim([0, self.dmax])

    def click(self, event):
        """
        What to do, if a click on the figure happens:
            1. Check which axis
            2. Get data coord's.
            3. Plot resulting data.
            4. Update Figure
        """
        if event.inaxes==self.xz_axis:
            xpos = event.xdata
            ypos = event.ydata
            print xpos, ypos
            pos = np.asarray([xpos, 0.0, ypos]) * self.smime
            if event.button==1:
                self.kwargs_dist['center'] = pos 
                print self.kwargs_dist
                self.get_box_coords()
                self.update_box_plot()
                self.get_dists_info()
                self.fit_nonthermal()
                self.update_distribution_plot()

        plt.draw()

    def update_box_plot(self):
        self.pbox.set_xdata(self.xbox)
        self.pbox.set_ydata(self.zbox)

    def update_distribution_plot(self):
        self.pvxy.set_data(self.fvel.fvel_xy)
        self.pvxz.set_data(self.fvel.fvel_xz)
        self.pvyz.set_data(self.fvel.fvel_yz)
        self.pv2d.set_data(self.fvel.fvel_para_perp)
        self.pvpara.set_ydata(self.fvel.fvel_para)
        self.pvperp.set_xdata(self.fvel.fvel_perp)
        self.pespect.set_ydata(self.fene.flog)
        self.pnth.set_ydata(self.fnonthermal)
        self.vpara_axis.relim()
        self.vperp_axis.relim()
        self.espect_axis.relim()
        self.vpara_axis.autoscale()
        self.vperp_axis.autoscale()
        self.espect_axis.autoscale()
        self.set_axes_lims()
        self.fig_dist.canvas.draw_idle()


if __name__ == "__main__":
    pic_info = pic_information.get_pic_info('../../')
    ntp = pic_info.ntp
    vthe = pic_info.vthe
    var_field = 'ey'
    var_name = '$E_y$'
    field_range = [0, 200, -20, 20]
    ct_ptl = 16

    smime = math.sqrt(pic_info.mime)
    lx_de = pic_info.lx_di * smime

    species = 'e'
    center = [0.5*lx_de, 0.0, 0.0]
    sizes = [128, 1, 256]
    center = np.asarray(center)
    sizes = np.asarray(sizes)
    nbins = 64
    vmin, vmax = 0, 2.0
    fpath_vdist = '../vdistributions/'
    fpath_spect = '../spectrum/'
    kwargs = {'ct_ptl':ct_ptl, 'var_field':var_field, 'var_name':var_name,
            'species':species, 'field_range':field_range, 'center':center,
            'sizes':sizes, 'nbins':nbins, 'vmax':vmax, 'vmin':vmin,
            'fpath_vdist':fpath_vdist, 'fpath_spect':fpath_spect}
    fig_v = EspectrumVdist(**kwargs)
    plt.show()
