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
import color_maps as cm
from contour_plots import read_2d_fields, plot_2d_contour
from particle_trajectory import *
import palettable

mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

colors = palettable.colorbrewer.qualitative.Set1_9.mpl_colors

font = {'family' : 'serif',
        #'color'  : 'darkred',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 24,
        }

class ParticleTrajectory153(object):
    def __init__(self, **kwargs):
        self.var_field = kwargs['var_field']
        self.var_name = kwargs['var_name']
        self.ct = kwargs['ct']
        self.field_range = kwargs['field_range']
        self.pic_info = pic_information.get_pic_info('../../')
        self.read_field_data()
        self.nptl = kwargs['nptl']
        self.iptl = kwargs['iptl']
        self.species = kwargs['species']
        self.traj_names = kwargs['traj_names']
        self.emin = kwargs['emin']
        self.emax = kwargs['emax']
        self.ymin = kwargs['ymin']
        self.ymax = kwargs['ymax']
        self.indicator_color = kwargs['indicator_color']
        self.ptl = read_traj_data(self.traj_names[self.iptl])
        self.smime = math.sqrt(self.pic_info.mime)
        self.ct_ptl = np.zeros(3, dtype='int')
        for i in range(3):
            self.ct_ptl[i] = self.ct[i] * \
                    self.pic_info.fields_interval / self.pic_info.trace_interval
        self.lx_di = self.pic_info.lx_di
        self.ly_di = self.pic_info.ly_di
        self.xmax1 = np.max(self.x1)
        self.zmax1 = np.max(self.z1)
        self.xmin1 = np.min(self.x1)
        self.zmin1 = np.min(self.z1)
        self.xmax2 = np.max(self.x2)
        self.zmax2 = np.max(self.z2)
        self.xmin2 = np.min(self.x2)
        self.zmin2 = np.min(self.z2)
        self.xmax3 = np.max(self.x3)
        self.zmax3 = np.max(self.z3)
        self.xmin3 = np.min(self.x3)
        self.zmin3 = np.min(self.z3)
        self.calc_derived_particle_info()
        self.get_particle_current_time()

        if self.species == 'e':
            self.threshold_ene = 0.5
        else:
            self.threshold_ene = 0.05

        # For saving figures
        if not os.path.isdir('../img/'):
            os.makedirs('../img/')
        self.fig_dir = '../img/img_traj_apj/'
        if not os.path.isdir(self.fig_dir):
            os.makedirs(self.fig_dir)

        self.fig_width = 7
        self.energy_plot()
        self.energy_plot_indicator()
        self.save_figures()
        
    def get_particle_current_time(self):
        self.t0 = self.t[self.ct_ptl]
        self.x0 = self.px[self.ct_ptl]
        self.xb0 = self.pxb[self.ct_ptl]
        self.y0 = self.py[self.ct_ptl]
        self.z0 = self.pz[self.ct_ptl]
        self.gama0 = self.gama[self.ct_ptl]

    def calc_derived_particle_info(self):
        if self.species == 'e':
            self.charge = -1.0
        else:
            self.charge = 1.0
        self.t = self.ptl.t * self.pic_info.dtwci / self.pic_info.dtwpe
        self.nt, = self.t.shape
        self.px = self.ptl.x / self.smime  # Change de to di
        self.py = self.ptl.y / self.smime
        self.pz = self.ptl.z / self.smime
        self.adjust_px()
        self.adjust_py()
        self.gama = np.sqrt(self.ptl.ux**2 + self.ptl.uy**2 + self.ptl.uz**2 + 1.0)
        self.mint = 0
        self.maxt = np.max(self.t)
        self.jdote_x = self.ptl.ux * self.ptl.ex * self.charge / self.gama
        self.jdote_y = self.ptl.uy * self.ptl.ey * self.charge / self.gama
        self.jdote_z = self.ptl.uz * self.ptl.ez * self.charge / self.gama
        self.dt = np.zeros(self.nt)
        self.dt[0:self.nt-1] = np.diff(self.t)
        self.jdote_x_cum = np.cumsum(self.jdote_x) * self.dt
        self.jdote_y_cum = np.cumsum(self.jdote_y) * self.dt
        self.jdote_z_cum = np.cumsum(self.jdote_z) * self.dt
        self.jdote_tot_cum = self.jdote_x_cum + self.jdote_y_cum + self.jdote_z_cum
        kernel = 9
        self.ex = signal.medfilt(self.ptl.ex, kernel_size=(kernel))
        self.ey = signal.medfilt(self.ptl.ey, kernel_size=(kernel))
        self.ez = signal.medfilt(self.ptl.ez, kernel_size=(kernel))
        self.xmin_b = np.min(self.pxb)
        self.xmax_b = np.max(self.pxb)

    def read_field_data(self):
        self.xl, self.xr = self.field_range[0, 0:2]
        self.zb, self.zt = self.field_range[0, 2:4]
        kwargs = {"current_time":self.ct[0], "xl":self.xl, "xr":self.xr,
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

        self.xl, self.xr = self.field_range[1, 0:2]
        self.zb, self.zt = self.field_range[1, 2:4]
        kwargs = {"current_time":self.ct[1], "xl":self.xl, "xr":self.xr,
                "zb":self.zb, "zt":self.zt}
        self.x2, self.z2, data = read_2d_fields(self.pic_info, fname_field,
                **kwargs)
        self.fdata2 = signal.convolve2d(data, kernel)
        self.x2, self.z2, self.Ay2 = read_2d_fields(self.pic_info,
                fname_Ay, **kwargs)
        self.nx2, = self.x2.shape
        self.nz2, = self.z2.shape

        self.xl, self.xr = self.field_range[2, 0:2]
        self.zb, self.zt = self.field_range[2, 2:4]
        kwargs = {"current_time":self.ct[2], "xl":self.xl, "xr":self.xr,
                "zb":self.zb, "zt":self.zt}
        self.x3, self.z3, data = read_2d_fields(self.pic_info, fname_field,
                **kwargs)
        self.fdata3 = signal.convolve2d(data, kernel)
        self.x3, self.z3, self.Ay3 = read_2d_fields(self.pic_info,
                fname_Ay, **kwargs)
        self.nx3, = self.x3.shape
        self.nz3, = self.z3.shape

    def adjust_py(self):
        """Adjust py for periodic boundary conditions.
        """
        crossings = []
        offsets = []
        offset = 0
        for i in range(self.nt-1):
            if (self.py[i]-self.py[i+1] > 0.4*self.ly_di):
                crossings.append(i)
                offset += self.ly_di
                offsets.append(offset)
            if (self.py[i]-self.py[i+1] < -0.4*self.ly_di):
                crossings.append(i)
                offset -= self.ly_di
                offsets.append(offset)
        nc = len(crossings)
        if nc > 0:
            crossings = np.asarray(crossings)
            offsets = np.asarray(offsets)
            for i in range(nc-1):
                self.py[crossings[i]+1 : crossings[i+1]+1] += offsets[i]
            self.py[crossings[nc-1]+1:] += offsets[nc-1]

    def adjust_px(self):
        """Adjust px for periodic boundary conditions.
        """
        crossings = []
        offsets = []
        offset = 0
        self.pxb = np.zeros(self.nt)
        self.pxb = np.copy(self.px)
        for i in range(self.nt-1):
            if (self.px[i]-self.px[i+1] > 0.4*self.lx_di):
                crossings.append(i)
                offset += self.lx_di
                offsets.append(offset)
            if (self.px[i]-self.px[i+1] < -0.4*self.lx_di):
                crossings.append(i)
                offset -= self.lx_di
                offsets.append(offset)
        nc = len(crossings)
        if nc > 0:
            crossings = np.asarray(crossings)
            offsets = np.asarray(offsets)
            for i in range(nc-1):
                self.pxb[crossings[i]+1 : crossings[i+1]+1] += offsets[i]
            self.pxb[crossings[nc-1]+1:] += offsets[nc-1]

    def energy_plot(self):
        self.fig_ene = plt.figure(figsize=(self.fig_width, 5))
        self.cmap = plt.cm.get_cmap('coolwarm')
        self.color_Ay = 'black'
        self.color_pxz = 'black'

        xs, ys = 0.12, 0.7
        w1, h1 = 0.78, 0.25
        h2 = 0.52
        gap = 0.05
        ye = ys + h1
        self.xz_axis = self.fig_ene.add_axes([xs, ys, w1, h1])
        self.vmax = max(abs(np.min(self.fdata1)), abs(np.max(self.fdata1)))
        self.vmax *= 0.5
        self.vmax = 0.05
        xst = zst = 4
        tst = 1
        self.xst = xst
        self.zst = zst
        nx1, nz1 = self.nx1, self.nz1
        nx2, nz2 = self.nx2, self.nz2
        nx3, nz3 = self.nx3, self.nz3
        self.im1 = self.xz_axis.imshow(self.fdata1[0:nz1:zst, 0:nx1:xst],
                cmap=self.cmap,
                extent=[self.xmin1, self.xmax1, self.zmin1, self.zmax1],
                aspect='auto', origin='lower',
                vmin = -self.vmax, vmax = self.vmax,
                interpolation='bicubic')
        self.im2 = self.xz_axis.imshow(self.fdata2[0:nz1:zst, 0:nx1:xst],
                cmap=self.cmap,
                extent=[self.xmin2, self.xmax2, self.zmin2, self.zmax2],
                aspect='auto', origin='lower',
                vmin = -self.vmax, vmax = self.vmax,
                interpolation='bicubic')
        self.im3 = self.xz_axis.imshow(self.fdata3[0:nz1:zst, 0:nx1:xst],
                cmap=self.cmap,
                extent=[self.xmin3, self.xmax3, self.zmin3, self.zmax3],
                aspect='auto', origin='lower',
                vmin = -self.vmax, vmax = self.vmax,
                interpolation='bicubic')
        divider = make_axes_locatable(self.xz_axis)
        self.cax = divider.append_axes("right", size="2%", pad=0.05)
        self.cbar = self.fig_ene.colorbar(self.im1, cax=self.cax)
        self.cbar.ax.tick_params(labelsize=16)
        self.cbar.set_ticks(np.arange(-0.04, 0.05, 0.02))
        self.Ay_max = np.max(self.Ay1)
        self.Ay_min = np.min(self.Ay1)
        self.levels = np.linspace(self.Ay_min, self.Ay_max, 10)
        self.xz_axis.contour(self.x1[0:nx1:xst], self.z1[0:nz1:zst],
                self.Ay1[0:nz1:zst, 0:nx1:xst], colors=self.color_Ay,
                linewidths=0.5, levels=self.levels)
        self.xz_axis.contour(self.x2[0:nx2:xst], self.z2[0:nz2:zst],
                self.Ay2[0:nz2:zst, 0:nx2:xst], colors=self.color_Ay,
                linewidths=0.5, levels=self.levels)
        self.xz_axis.contour(self.x3[0:nx3:xst], self.z3[0:nz3:zst],
                self.Ay3[0:nz3:zst, 0:nx3:xst], colors=self.color_Ay,
                linewidths=0.5, levels=self.levels)
        self.xz_axis.tick_params(labelsize=16)
        self.xz_axis.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
        self.xz_axis.tick_params(axis='x', labelbottom='off')
        self.xz_axis.autoscale(1,'both',1)

        # xz plot
        # self.pxz, = self.xz_axis.plot(self.px, self.pz, linewidth=2,
        #         color='k', marker='.', markersize=1, linestyle='None')
        self.pxz, = self.xz_axis.plot(self.pxb[::tst], self.pz[::tst],
                color=self.color_pxz)

        # x-energy after periodic x correction
        ys -= h2 + gap
        width, height = self.fig_ene.get_size_inches()
        w2 = w1 * 0.98 - 0.05 / width
        self.xeb_axis = self.fig_ene.add_axes([xs, ys, w2, h2])
        self.xeb_axis.tick_params(labelsize=16)
        self.pxe_b, = self.xeb_axis.plot(self.pxb[::tst],
                self.gama[::tst] - 1.0, color=colors[0])
        self.xeb_axis.tick_params(labelsize=16)
        self.xeb_axis.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
        self.xeb_axis.set_ylabel(r'$\gamma - 1$', fontdict=font,
                fontsize=20, color=colors[0])
        self.xeb_axis.set_ylim([self.emin, self.emax])
        for tl in self.xeb_axis.get_yticklabels():
            tl.set_color(colors[0])
        self.xmin_b, self.xmax_b = self.xeb_axis.get_xlim()
        self.xz_axis.set_xlim(self.xmin_b, self.xmax_b)

        # x-y plot after periodic x correction
        ys -= h2 + gap
        self.xyb_axis = self.xeb_axis.twinx()
        self.xyb_axis.tick_params(labelsize=16)
        self.pxy_b, = self.xyb_axis.plot(self.pxb[::tst], self.py[::tst],
                color=colors[1])
        for tl in self.xyb_axis.get_yticklabels():
            tl.set_color(colors[1])
        self.xmin_b, self.xmax_b = self.xeb_axis.get_xlim()
        self.xyb_axis.set_xlim([self.xmin_b, self.xmax_b])
        # self.pxy_help_b, = self.xyb_axis.plot([self.xmin_b, self.xmax_b], [0, 0],
        #         color='r', linestyle='--')
        self.xyb_axis.tick_params(labelsize=16)
        self.xyb_axis.set_ylabel(r'$y/d_i$', fontdict=font,
                fontsize=20, color=colors[1])
        self.xyb_axis.set_ylim([self.ymin, self.ymax])
        self.xyb_axis.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)

        # Extra fields contour
        self.plot_extra_contour()

    def plot_extra_contour(self):
        if (self.xmax_b > self.pic_info.lx_di):
            ex = self.xmax_b - self.pic_info.lx_di
            kwargs = {"current_time":self.ct[2], "xl":0, "xr":ex,
                    "zb":self.field_range[2,2], "zt":self.field_range[2,3]}
            fname_field = '../../data/' + self.var_field + '.gda'
            fname_Ay = '../../data/Ay.gda'
            x, z, fdata = read_2d_fields(self.pic_info, fname_field, **kwargs)
            ng = 5
            kernel = np.ones((ng,ng)) / float(ng*ng)
            fdata = signal.convolve2d(fdata, kernel)
            x, z, Ay = read_2d_fields(self.pic_info, fname_Ay, **kwargs)
            nx, = x.shape
            nz, = z.shape
            ex_grid = ex / self.pic_info.dx_di + 1
            xmin = self.pic_info.lx_di
            xmax = xmin + ex
            self.im1 = self.xz_axis.imshow(fdata[0:nz:self.zst, 0:nx:self.xst],
                    cmap=self.cmap, extent=[xmin, xmax, self.zmin3, self.zmax3],
                    aspect='auto', origin='lower',
                    vmin=-self.vmax, vmax=self.vmax,
                    interpolation='bicubic')
            x += xmin
            self.xz_axis.contour(x[0:nx:self.xst], z[0:nz:self.zst],
                    Ay[0:nz:self.zst, 0:nx:self.xst], colors='black',
                    linewidths=0.5, levels=self.levels)

    def energy_plot_indicator(self):
        self.pxz_dot, = self.xz_axis.plot(self.xb0, self.z0, marker='x',
                markersize=10, mew=2, linestyle='None',
                color=self.indicator_color)
        self.pxby_dot, = self.xyb_axis.plot(self.xb0, self.y0, marker='x',
                markersize=10, mew=2, linestyle='None',
                color=self.indicator_color)
        self.pxbe_dot, = self.xeb_axis.plot(self.xb0, self.gama0-1, marker='x',
                markersize=10, mew=2, linestyle='None',
                color=self.indicator_color)

    def save_figures(self):
        fname = self.fig_dir + 'traj_' + self.species + '_' + \
                str(self.iptl).zfill(4) + '_1.pdf'
        self.fig_ene.savefig(fname)
        fname = self.fig_dir + 'traj_' + self.species + '_' + \
                str(self.iptl).zfill(4) + '_1.jpg'
        self.fig_ene.savefig(fname, dpi=200)


class ParticleTrajectory154(object):
    def __init__(self, **kwargs):
        self.var_field = kwargs['var_field']
        self.var_name = kwargs['var_name']
        self.ct = kwargs['ct']
        self.field_range = kwargs['field_range']
        self.pic_info = pic_information.get_pic_info('../../')
        self.read_field_data()
        self.nptl = kwargs['nptl']
        self.iptl = kwargs['iptl']
        self.species = kwargs['species']
        self.traj_names = kwargs['traj_names']
        self.emin = kwargs['emin']
        self.emax = kwargs['emax']
        self.ymin = kwargs['ymin']
        self.ymax = kwargs['ymax']
        self.indicator_color = kwargs['indicator_color']
        self.ptl = read_traj_data(self.traj_names[self.iptl])
        self.smime = math.sqrt(self.pic_info.mime)
        self.ct_ptl = np.zeros(3, dtype='int')
        self.ct_ptl = self.ct * \
                self.pic_info.fields_interval / self.pic_info.trace_interval
        self.lx_di = self.pic_info.lx_di
        self.ly_di = self.pic_info.ly_di
        self.xmax1 = np.max(self.x1)
        self.zmax1 = np.max(self.z1)
        self.xmin1 = np.min(self.x1)
        self.zmin1 = np.min(self.z1)
        self.calc_derived_particle_info()
        self.get_particle_current_time()

        if self.species == 'e':
            self.threshold_ene = 0.5
        else:
            self.threshold_ene = 0.05

        # For saving figures
        if not os.path.isdir('../img/'):
            os.makedirs('../img/')
        self.fig_dir = '../img/img_traj_apj/'
        if not os.path.isdir(self.fig_dir):
            os.makedirs(self.fig_dir)

        self.fig_width = 7
        self.energy_plot()
        self.energy_plot_indicator()
        self.save_figures()
        
    def get_particle_current_time(self):
        self.t0 = self.t[self.ct_ptl]
        self.x0 = self.px[self.ct_ptl]
        self.xb0 = self.pxb[self.ct_ptl]
        self.y0 = self.py[self.ct_ptl]
        self.z0 = self.pz[self.ct_ptl]
        self.gama0 = self.gama[self.ct_ptl]

    def calc_derived_particle_info(self):
        if self.species == 'e':
            self.charge = -1.0
        else:
            self.charge = 1.0
        self.t = self.ptl.t * self.pic_info.dtwci / self.pic_info.dtwpe
        self.nt, = self.t.shape
        self.px = self.ptl.x / self.smime  # Change de to di
        self.py = self.ptl.y / self.smime
        self.pz = self.ptl.z / self.smime
        self.adjust_px()
        self.adjust_py()
        self.gama = np.sqrt(self.ptl.ux**2 + self.ptl.uy**2 + self.ptl.uz**2 + 1.0)
        self.mint = 0
        self.maxt = np.max(self.t)
        self.jdote_x = self.ptl.ux * self.ptl.ex * self.charge / self.gama
        self.jdote_y = self.ptl.uy * self.ptl.ey * self.charge / self.gama
        self.jdote_z = self.ptl.uz * self.ptl.ez * self.charge / self.gama
        self.dt = np.zeros(self.nt)
        self.dt[0:self.nt-1] = np.diff(self.t)
        self.jdote_x_cum = np.cumsum(self.jdote_x) * self.dt
        self.jdote_y_cum = np.cumsum(self.jdote_y) * self.dt
        self.jdote_z_cum = np.cumsum(self.jdote_z) * self.dt
        self.jdote_tot_cum = self.jdote_x_cum + self.jdote_y_cum + self.jdote_z_cum
        kernel = 9
        self.ex = signal.medfilt(self.ptl.ex, kernel_size=(kernel))
        self.ey = signal.medfilt(self.ptl.ey, kernel_size=(kernel))
        self.ez = signal.medfilt(self.ptl.ez, kernel_size=(kernel))
        self.xmin_b = np.min(self.pxb)
        self.xmax_b = np.max(self.pxb)

    def read_field_data(self):
        self.xl, self.xr = self.field_range[0:2]
        self.zb, self.zt = self.field_range[2:4]
        kwargs = {"current_time":self.ct, "xl":self.xl, "xr":self.xr,
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

    def adjust_py(self):
        """Adjust py for periodic boundary conditions.
        """
        crossings = []
        offsets = []
        offset = 0
        for i in range(self.nt-1):
            if (self.py[i]-self.py[i+1] > 0.4*self.ly_di):
                crossings.append(i)
                offset += self.ly_di
                offsets.append(offset)
            if (self.py[i]-self.py[i+1] < -0.4*self.ly_di):
                crossings.append(i)
                offset -= self.ly_di
                offsets.append(offset)
        nc = len(crossings)
        if nc > 0:
            crossings = np.asarray(crossings)
            offsets = np.asarray(offsets)
            for i in range(nc-1):
                self.py[crossings[i]+1 : crossings[i+1]+1] += offsets[i]
            self.py[crossings[nc-1]+1:] += offsets[nc-1]

    def adjust_px(self):
        """Adjust px for periodic boundary conditions.
        """
        crossings = []
        offsets = []
        offset = 0
        self.pxb = np.zeros(self.nt)
        self.pxb = np.copy(self.px)
        for i in range(self.nt-1):
            if (self.px[i]-self.px[i+1] > 0.4*self.lx_di):
                crossings.append(i)
                offset += self.lx_di
                offsets.append(offset)
            if (self.px[i]-self.px[i+1] < -0.4*self.lx_di):
                crossings.append(i)
                offset -= self.lx_di
                offsets.append(offset)
        nc = len(crossings)
        if nc > 0:
            crossings = np.asarray(crossings)
            offsets = np.asarray(offsets)
            for i in range(nc-1):
                self.pxb[crossings[i]+1 : crossings[i+1]+1] += offsets[i]
            self.pxb[crossings[nc-1]+1:] += offsets[nc-1]

    def energy_plot(self):
        self.fig_ene = plt.figure(figsize=(self.fig_width, 5))
        self.cmap = plt.cm.get_cmap('coolwarm')
        self.color_Ay = 'black'
        self.color_pxz = 'black'

        xs, ys = 0.14, 0.7
        w1, h1 = 0.76, 0.25
        h2 = 0.52
        gap = 0.05
        ye = ys + h1
        self.xz_axis = self.fig_ene.add_axes([xs, ys, w1, h1])
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
        self.cbar = self.fig_ene.colorbar(self.im1, cax=self.cax)
        self.cbar.ax.tick_params(labelsize=16)
        self.cbar.set_ticks(np.arange(-0.04, 0.05, 0.02))
        self.Ay_max = np.max(self.Ay1)
        self.Ay_min = np.min(self.Ay1)
        self.levels = np.linspace(self.Ay_min, self.Ay_max, 10)
        self.xz_axis.contour(self.x1[0:nx1:xst], self.z1[0:nz1:zst],
                self.Ay1[0:nz1:zst, 0:nx1:xst], colors=self.color_Ay,
                linewidths=0.5, levels=self.levels)
        self.xz_axis.tick_params(labelsize=16)
        self.xz_axis.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
        self.xz_axis.tick_params(axis='x', labelbottom='off')
        self.xz_axis.autoscale(1,'both',1)

        # xz plot
        # self.pxz, = self.xz_axis.plot(self.px, self.pz, linewidth=2,
        #         color='k', marker='.', markersize=1, linestyle='None')
        self.pxz, = self.xz_axis.plot(self.pxb[::tst], self.pz[::tst],
                color=self.color_pxz)

        # x-energy after periodic x correction
        ys -= h2 + gap
        width, height = self.fig_ene.get_size_inches()
        w2 = w1 * 0.98 - 0.05 / width
        self.xeb_axis = self.fig_ene.add_axes([xs, ys, w2, h2])
        self.xeb_axis.tick_params(labelsize=16)
        self.pxe_b, = self.xeb_axis.plot(self.pxb[::tst],
                self.gama[::tst] - 1.0, color=colors[0])
        self.xeb_axis.tick_params(labelsize=16)
        self.xeb_axis.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
        self.xeb_axis.set_ylabel(r'$\gamma - 1$', fontdict=font,
                fontsize=20, color=colors[0])
        self.xeb_axis.set_ylim([self.emin, self.emax])
        for tl in self.xeb_axis.get_yticklabels():
            tl.set_color(colors[0])
        self.xmin_b, self.xmax_b = self.xz_axis.get_xlim()
        self.xeb_axis.set_xlim(self.xmin_b, self.xmax_b)

        # x-y plot after periodic x correction
        ys -= h2 + gap
        self.xyb_axis = self.xeb_axis.twinx()
        self.xyb_axis.tick_params(labelsize=16)
        self.pxy_b, = self.xyb_axis.plot(self.pxb[::tst], self.py[::tst],
                color=colors[1])
        for tl in self.xyb_axis.get_yticklabels():
            tl.set_color(colors[1])
        self.xmin_b, self.xmax_b = self.xz_axis.get_xlim()
        self.xyb_axis.set_xlim([self.xmin_b, self.xmax_b])
        # self.pxy_help_b, = self.xyb_axis.plot([self.xmin_b, self.xmax_b], [0, 0],
        #         color='r', linestyle='--')
        self.xyb_axis.tick_params(labelsize=16)
        self.xyb_axis.set_ylabel(r'$y/d_i$', fontdict=font,
                fontsize=20, color=colors[1])
        self.xyb_axis.set_ylim([self.ymin, self.ymax])
        self.xyb_axis.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)

        # Extra fields contour
        self.plot_extra_contour()

    def plot_extra_contour(self):
        if (self.xmax_b > self.pic_info.lx_di):
            ex = self.xmax_b - self.pic_info.lx_di
            kwargs = {"current_time":self.ct, "xl":0, "xr":ex,
                    "zb":self.field_range[2], "zt":self.field_range[3]}
            fname_field = '../../data/' + self.var_field + '.gda'
            fname_Ay = '../../data/Ay.gda'
            x, z, fdata = read_2d_fields(self.pic_info, fname_field, **kwargs)
            ng = 5
            kernel = np.ones((ng,ng)) / float(ng*ng)
            fdata = signal.convolve2d(fdata, kernel)
            x, z, Ay = read_2d_fields(self.pic_info, fname_Ay, **kwargs)
            nx, = x.shape
            nz, = z.shape
            ex_grid = ex / self.pic_info.dx_di + 1
            xmin = self.pic_info.lx_di
            xmax = xmin + ex
            self.im1 = self.xz_axis.imshow(fdata[0:nz:self.zst, 0:nx:self.xst],
                    cmap=self.cmap, extent=[xmin, xmax, self.zmin1, self.zmax1],
                    aspect='auto', origin='lower',
                    vmin=-self.vmax, vmax=self.vmax,
                    interpolation='bicubic')
            x += xmin
            self.xz_axis.contour(x[0:nx:self.xst], z[0:nz:self.zst],
                    Ay[0:nz:self.zst, 0:nx:self.xst], colors='black',
                    linewidths=0.5, levels=self.levels)

    def energy_plot_indicator(self):
        self.pxz_dot, = self.xz_axis.plot(self.xb0, self.z0, marker='x',
                markersize=10, mew=2, linestyle='None',
                color=self.indicator_color)
        self.pxby_dot, = self.xyb_axis.plot(self.xb0, self.y0, marker='x',
                markersize=10, mew=2, linestyle='None',
                color=self.indicator_color)
        self.pxbe_dot, = self.xeb_axis.plot(self.xb0, self.gama0-1, marker='x',
                markersize=10, mew=2, linestyle='None',
                color=self.indicator_color)

    def save_figures(self):
        fname = self.fig_dir + 'traj_' + self.species + '_' + \
                str(self.iptl).zfill(4) + '_1.pdf'
        self.fig_ene.savefig(fname)
        fname = self.fig_dir + 'traj_' + self.species + '_' + \
                str(self.iptl).zfill(4) + '_1.jpg'
        self.fig_ene.savefig(fname, dpi=200)


def plot_electron_trajectory(fnames, species, pic_info):
    """Plot electron trajectory.

    Args:
        fnames: file names for the trajectory files.
        species: particle species.
        pic_info: particle information namedtuple.
    """
    nptl = len(fnames)

    iptl = 89
    var_field = 'ey'
    var_name = '$E_y$'
    emin, emax = -2.5, 2.5
    ymin, ymax = -50, 40
    field_range = np.zeros((3, 4))
    field_range[:, 2] = -10
    field_range[:, 3] = 10
    field_range[0, 0:2] = [80, 130]
    field_range[1, 0:2] = [130, 165]
    field_range[2, 0:2] = [165, 200]
    ct = [61, 100, 170]
    indicator_color = colors[2]

    # iptl = 153
    # var_field = 'ey'
    # var_name = '$E_y$'
    # emin, emax = 0, 2.1
    # ymin, ymax = -70, 10
    # field_range = np.zeros((3, 4))
    # field_range[:, 2] = -20
    # field_range[:, 3] = 20
    # field_range[0, 0:2] = [80, 120]
    # field_range[1, 0:2] = [120, 160]
    # field_range[2, 0:2] = [160, 200]
    # ct = [48, 92, 206]
    # indicator_color = colors[2]

    # iptl = 382
    # var_field = 'ey'
    # var_name = '$E_y$'
    # emin, emax = -1.0, 1.5
    # ymin, ymax = -40, 200
    # field_range = np.zeros((3, 4))
    # field_range[:, 2] = -12
    # field_range[:, 3] = 12
    # field_range[0, 0:2] = [80, 130]
    # field_range[1, 0:2] = [130, 165]
    # field_range[2, 0:2] = [165, 200]
    # ct = [55, 110, 200]
    # indicator_color = colors[2]

    species = 'e'
    kwargs = {'nptl':nptl, 'iptl':iptl, 'ct':ct, 'var_field':var_field,
            'var_name':var_name, 'species':species, 'traj_names':fnames,
            'field_range':field_range, 'emax':emax, 'emin':emin,
            'ymin':ymin, 'ymax':ymax, 'indicator_color':indicator_color}
    fig_v = ParticleTrajectory153(**kwargs)
    plt.show()


def plot_particle_energies(fnames, species, pic_info):
    """Plot kinetic energy for multiple particles.
    """
    iptls = [89, 153, 382]
    species = 'e'
    ct = np.zeros((3, 3), dtype='int')
    ct[0, :] = [61, 100, 170]
    ct[1, :] = [48, 92, 206]
    ct[2, :] = [55, 110, 200]
    t_ratio = pic_info.fields_interval / pic_info.trace_interval
    smime = math.sqrt(pic_info.mime)

    fig = plt.figure(figsize=(7, 5))
    xs, ys = 0.12, 0.72
    w1, h1 = 0.78, 0.25
    gap = 0.05
    emax = [2.5, 2.0, 1.4]
    tags = ['a', 'b', 'c']
    for i in range(3):
        ax = fig.add_axes([xs, ys, w1, h1])
        ptl = read_traj_data(fnames[iptls[i]])
        gama = np.sqrt(ptl.ux**2 + ptl.uy**2 + ptl.uz**2 + 1)
        px = ptl.x / smime  # Change de to di
        py = ptl.y / smime
        pz = ptl.z / smime
        t = ptl.t * pic_info.dtwci / pic_info.dtwpe
        p1, = ax.plot(t, gama - 1, color='k')
        color = p1.get_color()
        for j in range(3):
            ct_ptl = ct[i, j] * t_ratio
            te_dot, = ax.plot(t[ct_ptl], gama[ct_ptl]-1, marker='x',
                    markersize=10, mew=2, linestyle='None', color=colors[2])
        ax.set_ylim([0, emax[i]])
        ax.tick_params(labelsize=16)
        ax.set_ylabel(r'$\gamma - 1$', fontdict=font, fontsize=20)
        if i < 2:
            ax.tick_params(axis='x', labelbottom='off')
        ax.text(0.05, 0.8, tags[i], color='k', fontsize=20,
                horizontalalignment='left', verticalalignment='center',
                transform = ax.transAxes)
        ys -= h1 + gap
    ax.set_xlabel(r'$t\Omega_{ci}$', fontdict=font, fontsize=20)
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    fig_dir = '../img/img_traj_apj/'
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    fname = fig_dir + 'tene.eps'
    fig.savefig(fname)
    plt.show()


def plot_ion_trajectory(fnames, species, pic_info):
    """Plot ion trajectory.

    Args:
        fnames: file names for the trajectory files.
        species: particle species.
        pic_info: particle information namedtuple.
    """
    nptl = len(fnames)

    # iptl = 58
    # var_field = 'ey'
    # var_name = '$E_y$'
    # emin, emax = 0, 0.13
    # ymin, ymax = 0, 170
    # field_range = np.zeros((3, 4))
    # field_range[:, 2] = -30
    # field_range[:, 3] = 30
    # field_range[0, 0:2] = [80, 100]
    # field_range[1, 0:2] = [100, 150]
    # field_range[2, 0:2] = [150, 200]
    # ct = [40, 80, 220]
    # indicator_color = colors[2]

    # iptl = 400
    # var_field = 'ey'
    # var_name = '$E_y$'
    # emin, emax = -0.25, 0.25
    # ymin, ymax = -60, 100
    # field_range = [120, 200, -30, 30]
    # ct = 109
    # indicator_color = colors[2]

    iptl = 1239
    var_field = 'ey'
    var_name = '$E_y$'
    emin, emax = -0.10, 0.14
    ymin, ymax = 0, 100
    field_range = [140, 200, -20, 20]
    ct = 55
    indicator_color = colors[2]

    species = 'i'
    kwargs = {'nptl':nptl, 'iptl':iptl, 'ct':ct, 'var_field':var_field,
            'var_name':var_name, 'species':species, 'traj_names':fnames,
            'field_range':field_range, 'emax':emax, 'emin':emin,
            'ymin':ymin, 'ymax':ymax, 'indicator_color':indicator_color}
    fig_v = ParticleTrajectory154(**kwargs)
    plt.show()


def plot_ions_energies(fnames, species, pic_info):
    """Plot kinetic energy for multiple ions.
    """
    iptls = [58, 400, 1239]
    species = 'i'
    ct = np.zeros((3, 3), dtype='int')
    ct[0, :] = [40, 80, 229]
    ct[1, :] = [109, 109, 109]
    ct[2, :] = [55, 55, 55]
    t_ratio = pic_info.fields_interval / pic_info.trace_interval
    smime = math.sqrt(pic_info.mime)

    fig = plt.figure(figsize=(7, 5))
    xs, ys = 0.12, 0.72
    w1, h1 = 0.78, 0.25
    gap = 0.05
    emax = [0.13, 0.25, 0.14]
    tags = ['a', 'b', 'c']
    for i in range(3):
        ax = fig.add_axes([xs, ys, w1, h1])
        ptl = read_traj_data(fnames[iptls[i]])
        gama = np.sqrt(ptl.ux**2 + ptl.uy**2 + ptl.uz**2 + 1)
        px = ptl.x / smime  # Change de to di
        py = ptl.y / smime
        pz = ptl.z / smime
        t = ptl.t * pic_info.dtwci / pic_info.dtwpe
        p1, = ax.plot(t, gama - 1, color='k')
        color = p1.get_color()
        if i == 0:
            for j in range(3):
                ct_ptl = ct[i, j] * t_ratio
                te_dot, = ax.plot(t[ct_ptl], gama[ct_ptl]-1, marker='x',
                        markersize=10, mew=2, linestyle='None', color=colors[2])
        else:
            ct_ptl = ct[i, 0] * t_ratio
            te_dot, = ax.plot(t[ct_ptl], gama[ct_ptl]-1, marker='x',
                    markersize=10, mew=2, linestyle='None', color=colors[2])
        ax.set_ylim([0, emax[i]])
        ax.tick_params(labelsize=16)
        ax.set_ylabel(r'$\gamma - 1$', fontdict=font, fontsize=20)
        if i < 2:
            ax.tick_params(axis='x', labelbottom='off')
        ax.text(0.05, 0.8, tags[i], color='k', fontsize=20,
                horizontalalignment='left', verticalalignment='center',
                transform = ax.transAxes)
        ys -= h1 + gap
    ax.set_xlabel(r'$t\Omega_{ci}$', fontdict=font, fontsize=20)
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    fig_dir = '../img/img_traj_apj/'
    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir)
    fname = fig_dir + 'tene_i.eps'
    fig.savefig(fname)
    plt.show()


if __name__ == "__main__":
    pic_info = pic_information.get_pic_info('../../')
    ntp = pic_info.ntp
    vthe = pic_info.vthe
    print("Number of field frames: %d" % pic_info.ntf)
    fnames_e, fnames_i = get_file_names()
    # plot_electron_trajectory(fnames_e, 'e', pic_info)
    # plot_ion_trajectory(fnames_i, 'i', pic_info)
    # plot_particle_energies(fnames_e, 'e', pic_info)
    plot_ions_energies(fnames_i, 'i', pic_info)
