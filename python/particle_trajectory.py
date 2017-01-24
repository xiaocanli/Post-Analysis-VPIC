"""
Analysis procedures for particle energy spectrum.
"""
import collections
import math
import os.path
import struct
from os import listdir
from os.path import isfile, join

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from scipy import signal

import pic_information
from contour_plots import plot_2d_contour, read_2d_fields

mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

font = {'family' : 'serif',
        #'color'  : 'darkred',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 24,
        }

def get_file_names(root_dir='../../'):
    """Get the file names in the traj folder.
    """
    traj_path = root_dir + 'traj/'
    fnames = [ f for f in listdir(traj_path) if isfile(join(traj_path,f)) ]
    fnames.sort()
    ntraj_e = 0
    for filename in fnames:
        if filename.startswith('e'):
            ntraj_e += 1
        else:
            break
    fnames_e = fnames[:ntraj_e]
    fnames_i = fnames[ntraj_e:]
    return (fnames_e, fnames_i)

def read_traj_data(filename):
    """Read the information of one particle trajectory.

    Args:
        filename: the filename of the trajectory file.
    """
    nvar = 13  # variables at each point
    ptl_traj_info = collections.namedtuple("ptl_traj_info",
            ['t', 'x', 'y', 'z', 'ux', 'uy', 'uz',
                'ex', 'ey', 'ez', 'bx', 'by', 'bz'])
    print 'Reading trajectory data from ', filename
    fname = '../../traj/' + filename
    statinfo = os.stat(fname)
    nstep = statinfo.st_size / nvar
    nstep /= np.dtype(np.float32).itemsize
    traj_data = np.zeros((nvar, nstep), dtype=np.float32)
    traj_data = np.memmap(fname, dtype='float32', mode='r', 
            offset=0, shape=((nvar, nstep)), order='F')
    ptl_traj = ptl_traj_info(t=traj_data[0,:],
            x=traj_data[1,:], y=traj_data[2,:], z=traj_data[3,:],
            ux=traj_data[4,:], uy=traj_data[5,:], uz=traj_data[6,:],
            ex=traj_data[7,:], ey=traj_data[8,:], ez=traj_data[9,:],
            bx=traj_data[10,:], by=traj_data[11,:], bz=traj_data[12,:])
    return ptl_traj


class ParticleTrajectory(object):
    def __init__(self, nptl, iptl, x, y, fdata, Ay, init_ft, var_field, var_name,
            species, traj_names):
        """
        Args:
            nptl: number of particles.
            x, y: the x, y coordinates for the 2D fields.
            fdata: the field data array.
            Ay: y component of the vector potential.
            init_ft: the initial field time frame.
            var_field: the name of the field data for reading.
            var_name: the name of the field for labeling.
            species: particle species.
            traj_names: the names of the trajectory files.
            iptl: particle index for current plot.
        """
        self.x = x
        self.y = y
        self.fdata = fdata
        self.Ay = Ay
        self.nptl = nptl
        self.ct = init_ft
        self.iptl = iptl
        self.var_field = var_field
        self.var_name = var_name
        self.species = species
        self.traj_names = traj_names
        self.ptl = read_traj_data(self.traj_names[self.iptl])
        self.pic_info = pic_information.get_pic_info('../../')
        self.smime = math.sqrt(self.pic_info.mime)
        self.ct_ptl = self.ct * \
                self.pic_info.fields_interval / self.pic_info.trace_interval
        self.lx_di = self.pic_info.lx_di
        self.ly_di = self.pic_info.ly_di
        self.xmax = np.max(self.x)
        self.ymax = np.max(self.y)
        self.xmin = np.min(self.x)
        self.ymin = np.min(self.y)
        self.calc_derived_particle_info()
        self.get_particle_current_time()

        if self.species == 'e':
            self.threshold_ene = 0.5
        else:
            self.threshold_ene = 0.05

        # For saving figures
        if not os.path.isdir('../img/'):
            os.makedirs('../img/')
        self.fig_dir = '../img/img_traj_' + species + '/'
        if not os.path.isdir(self.fig_dir):
            os.makedirs(self.fig_dir)
        self.fig_dir_energetic = '../img/img_traj_energetic_' + species + '/'
        if not os.path.isdir(self.fig_dir_energetic):
            os.makedirs(self.fig_dir_energetic)

        self.fig_width = 10
        self.energy_plot()
        self.energy_plot_indicator()
        # self.fields_plot()
        # self.jdote_plot()
        self.save_figures()
        
    def get_particle_current_time(self):
        self.t0 = self.t[self.ct_ptl]
        self.x0 = self.px[self.ct_ptl]
        self.xb0 = self.pxb[self.ct_ptl]
        self.y0 = self.py[self.ct_ptl]
        self.z0 = self.pz[self.ct_ptl]
        self.gama0 = self.gama[self.ct_ptl]
        self.ux0 = self.ptl.ux[self.ct_ptl]
        self.uy0 = self.ptl.uy[self.ct_ptl]
        self.uz0 = self.ptl.uz[self.ct_ptl]
        self.ex0 = self.ex[self.ct_ptl]
        self.ey0 = self.ey[self.ct_ptl]
        self.ez0 = self.ez[self.ct_ptl]
        self.bx0 = self.ptl.bx[self.ct_ptl]
        self.by0 = self.ptl.by[self.ct_ptl]
        self.bz0 = self.ptl.bz[self.ct_ptl]
        self.jdote_x0 = self.jdote_x[self.ct_ptl]
        self.jdote_y0 = self.jdote_y[self.ct_ptl]
        self.jdote_z0 = self.jdote_z[self.ct_ptl]
        self.jdote_x_cum0 = self.jdote_x_cum[self.ct_ptl]
        self.jdote_y_cum0 = self.jdote_y_cum[self.ct_ptl]
        self.jdote_z_cum0 = self.jdote_z_cum[self.ct_ptl]

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
        self.fig_ene = plt.figure(figsize=(self.fig_width*2, 15))

        xs, ys = 0.05, 0.78
        w1, h1 = 0.4, 0.2
        h2 = 0.14
        gap = 0.02
        ye = ys + h1
        self.xz_axis = self.fig_ene.add_axes([xs, ys, w1, h1])
        vmax = min(abs(np.min(self.fdata)), abs(np.max(self.fdata)))
        vmax *= 0.2
        self.im1 = self.xz_axis.imshow(self.fdata, cmap=plt.cm.seismic,
                extent=[self.xmin, self.xmax, self.ymin, self.ymax],
                aspect='auto', origin='lower',
                vmin = -vmax, vmax = vmax,
                interpolation='bicubic')
        divider = make_axes_locatable(self.xz_axis)
        self.cax = divider.append_axes("right", size="2%", pad=0.05)
        self.cbar = self.fig_ene.colorbar(self.im1, cax=self.cax)
        self.cbar.ax.tick_params(labelsize=16)
        self.cbar.ax.set_ylabel(self.var_name, fontdict=font, fontsize=24)
        self.xz_axis.contour(self.x, self.y, self.Ay, colors='black', linewidths=0.5)
        self.xz_axis.tick_params(labelsize=16)
        self.xz_axis.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
        self.xz_axis.tick_params(axis='x', labelbottom='off')
        self.xz_axis.autoscale(1,'both',1)

        # xz plot
        # self.pxz, = self.xz_axis.plot(self.px, self.pz, linewidth=2,
        #         color='k', marker='.', markersize=1, linestyle='-')
        self.pxz, = self.xz_axis.plot(self.px, self.pz, color='k')

        # x-energy
        ys -= h2 + gap
        width, height = self.fig_ene.get_size_inches()
        w2 = w1 * 0.98 - 0.05 / width
        self.xe_axis = self.fig_ene.add_axes([xs, ys, w2, h2])
        self.xe_axis.tick_params(labelsize=16)
        self.pxe, = self.xe_axis.plot(self.px, self.gama - 1.0,
                linewidth=2, color='r')
        self.xe_axis.set_xlim([self.xmin, self.xmax])
        self.xe_axis.set_ylabel(r'$\gamma - 1$', fontdict=font, fontsize=20)
        self.xe_axis.tick_params(axis='x', labelbottom='off')

        # x-y plot
        ys -= h2 + gap
        self.xy_axis = self.fig_ene.add_axes([xs, ys, w2, h2])
        self.xy_axis.tick_params(labelsize=16)
        self.pxy, = self.xy_axis.plot(self.px, self.py, linewidth=2, color='k')
        self.pxy_help, = self.xy_axis.plot([self.xmin, self.xmax], [0, 0],
                color='r', linestyle='--')
        self.xy_axis.set_xlim([self.xmin, self.xmax])
        self.xy_axis.set_ylabel(r'$\gamma - 1$', fontdict=font, fontsize=20)
        self.xy_axis.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
        self.xy_axis.set_ylabel(r'$y/d_i$', fontdict=font, fontsize=20)

        # Energy plot
        gap = 0.06
        ys -= h2 + gap
        self.te_axis = self.fig_ene.add_axes([xs, ys, w2, h2])
        self.pte, = self.te_axis.plot(self.t, self.gama - 1.0, linewidth=2, color='k')
        self.te_axis.set_ylabel(r'$\gamma - 1$', fontdict=font, fontsize=20)
        self.te_axis.set_xlabel(r'$t\Omega_{ci}$', fontdict=font, fontsize=20)
        self.te_axis.tick_params(labelsize=16)

        # z-energy
        ys -= h2 + gap
        self.ze_axis = self.fig_ene.add_axes([xs, ys, w2, h2])
        self.ze_axis.tick_params(labelsize=16)
        self.pze, = self.ze_axis.plot(self.pz, self.gama - 1.0,
                linewidth=2, color='b')
        self.ze_axis.set_ylabel(r'$\gamma - 1$', fontdict=font, fontsize=20)
        self.ze_axis.set_xlabel(r'$z/d_i$', fontdict=font, fontsize=20)

        # y-energy
        xs += w1 + 0.1
        w3 = 0.43
        ys += h2 * 2 + gap * 2
        h3 = ye - ys
        self.ye_axis = self.fig_ene.add_axes([xs, ys, w3, h3])
        self.ye_axis.tick_params(labelsize=16)
        self.pye, = self.ye_axis.plot(self.py, self.gama - 1.0,
                linewidth=2, color='g')
        self.ye_axis.set_ylabel(r'$\gamma - 1$', fontdict=font, fontsize=20)
        self.ye_axis.set_xlabel(r'$y/d_i$', fontdict=font, fontsize=20)

        # x-energy after periodic x correction
        ys -= h2 + gap
        width, height = self.fig_ene.get_size_inches()
        w2 = w1 * 0.98 - 0.05 / width
        self.xeb_axis = self.fig_ene.add_axes([xs, ys, w3, h2])
        self.xeb_axis.tick_params(labelsize=16)
        self.pxe_b, = self.xeb_axis.plot(self.pxb, self.gama - 1.0,
                linewidth=2, color='r')
        self.xeb_axis.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
        self.xeb_axis.set_ylabel(r'$\gamma - 1$', fontdict=font, fontsize=20)

        # x-y plot after periodic x correction
        ys -= h2 + gap
        self.xyb_axis = self.fig_ene.add_axes([xs, ys, w3, h2])
        self.xyb_axis.tick_params(labelsize=16)
        self.pxy_b, = self.xyb_axis.plot(self.pxb, self.py, linewidth=2, color='k')
        self.xmin_b, self.xmax_b = self.xeb_axis.get_xlim()
        self.xyb_axis.set_xlim([self.xmin_b, self.xmax_b])
        self.pxy_help_b, = self.xyb_axis.plot([self.xmin_b, self.xmax_b], [0, 0],
                color='r', linestyle='--')
        self.xyb_axis.set_ylabel(r'$\gamma - 1$', fontdict=font, fontsize=20)
        self.xyb_axis.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
        self.xyb_axis.set_ylabel(r'$y/d_i$', fontdict=font, fontsize=20)

    def energy_plot_indicator(self):
        self.pxz_dot, = self.xz_axis.plot(self.x0, self.z0, marker='*',
                markersize=15, linestyle='None', color='g')
        self.pxy_dot, = self.xy_axis.plot(self.x0, self.y0, marker='*',
                markersize=15, linestyle='None', color='g')
        self.pxe_dot, = self.xe_axis.plot(self.x0, self.gama0-1, marker='*',
                markersize=15, linestyle='None', color='g')
        self.pte_dot, = self.te_axis.plot(self.t0, self.gama0-1, marker='*',
                markersize=15, linestyle='None', color='g')
        self.pze_dot, = self.ze_axis.plot(self.z0, self.gama0-1, marker='*',
                markersize=15, linestyle='None', color='g')
        self.pye_dot, = self.ye_axis.plot(self.y0, self.gama0-1, marker='*',
                markersize=15, linestyle='None', color='r')
        self.pxby_dot, = self.xyb_axis.plot(self.xb0, self.y0, marker='*',
                markersize=15, linestyle='None', color='g')
        self.pxbe_dot, = self.xeb_axis.plot(self.xb0, self.gama0-1, marker='*',
                markersize=15, linestyle='None', color='g')

    def fields_plot(self):
        height = 15.0
        self.fig_emf = plt.figure(figsize=[self.fig_width*2, height])
        w1, h1 = 0.88, 0.135
        xs, ys = 0.10, 0.98-h1
        gap = 0.025

        self.t_gama_axis = self.fig_emf.add_axes([xs, ys, w1, h1])
        self.pt_gama, = self.t_gama_axis.plot(self.t, self.gama-1.0, color='k')
        self.t_gama_axis.set_ylabel(r'$E/m_ic^2$', fontdict=font)
        self.t_gama_axis.tick_params(labelsize=20)
        self.t_gama_axis.tick_params(axis='x', labelbottom='off')
        self.t_gama_axis.plot([self.mint, self.maxt], [0, 0], '--', color='k')
        self.t_gama_axis.text(0.4, -0.07, r'$x$', color='red', fontsize=32, 
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='center', verticalalignment='center',
                transform = self.t_gama_axis.transAxes)
        self.t_gama_axis.text(0.5, -0.07, r'$y$', color='green', fontsize=32, 
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='center', verticalalignment='center',
                transform = self.t_gama_axis.transAxes)
        self.t_gama_axis.text(0.6, -0.07, r'$z$', color='blue', fontsize=32, 
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='center', verticalalignment='center',
                transform = self.t_gama_axis.transAxes)

        ys -= h1 + gap
        self.uxyz_axis = self.fig_emf.add_axes([xs, ys, w1, h1])
        self.pux, = self.uxyz_axis.plot(self.t, self.ptl.ux, color='r', label=r'u_x')
        self.puy, = self.uxyz_axis.plot(self.t, self.ptl.uy, color='g', label=r'u_y')
        self.puz, = self.uxyz_axis.plot(self.t, self.ptl.uz, color='b', label=r'u_z')
        self.uxyz_axis.plot([self.mint, self.maxt], [0, 0], '--', color='k')
        self.uxyz_axis.set_ylabel(r'$u_x, u_y, u_z$', fontdict=font)
        self.uxyz_axis.tick_params(labelsize=20)
        self.uxyz_axis.tick_params(axis='x', labelbottom='off')

        ys -= h1 + gap
        self.ex_axis = self.fig_emf.add_axes([xs, ys, w1, h1])
        self.pex, = self.ex_axis.plot(self.t, self.ex, color='r', label=r'E_x')
        self.ex_axis.set_ylabel(r'$E_x$', fontdict=font)
        self.ex_axis.tick_params(labelsize=20)
        self.ex_axis.tick_params(axis='x', labelbottom='off')
        self.ex_axis.plot([self.mint, self.maxt], [0, 0], '--', color='k')

        ys -= h1 + gap
        self.ey_axis = self.fig_emf.add_axes([xs, ys, w1, h1])
        self.pey, = self.ey_axis.plot(self.t, self.ey, color='g', label=r'E_x')
        self.ey_axis.set_ylabel(r'$E_y$', fontdict=font)
        self.ey_axis.tick_params(labelsize=20)
        self.ey_axis.tick_params(axis='x', labelbottom='off')
        self.ey_axis.plot([self.mint, self.maxt], [0, 0], '--', color='k')

        ys -= h1 + gap
        self.ez_axis = self.fig_emf.add_axes([xs, ys, w1, h1])
        self.pez, = self.ez_axis.plot(self.t, self.ez, color='b', label=r'E_x')
        self.ez_axis.set_ylabel(r'$E_z$', fontdict=font)
        self.ez_axis.tick_params(labelsize=20)
        self.ez_axis.tick_params(axis='x', labelbottom='off')
        self.ez_axis.plot([self.mint, self.maxt], [0, 0], '--', color='k')

        ys -= h1 + gap
        self.bxyz_axis = self.fig_emf.add_axes([xs, ys, w1, h1])
        self.pbx, = self.bxyz_axis.plot(self.t, self.ptl.bx, color='r', label=r'u_x')
        self.pby, = self.bxyz_axis.plot(self.t, self.ptl.by, color='g', label=r'u_y')
        self.pbz, = self.bxyz_axis.plot(self.t, self.ptl.bz, color='b', label=r'u_z')
        self.bxyz_axis.plot([self.mint, self.maxt], [0, 0], '--', color='k')
        self.bxyz_axis.set_xlabel(r'$t\Omega_{ci}$', fontdict=font)
        self.bxyz_axis.set_ylabel(r'$B_x, B_y, B_z$', fontdict=font)
        self.bxyz_axis.tick_params(labelsize=20)
        self.t_gama_axis.set_xlim([self.mint, self.maxt])
        self.uxyz_axis.set_xlim([self.mint, self.maxt])
        self.ex_axis.set_xlim([self.mint, self.maxt])
        self.ey_axis.set_xlim([self.mint, self.maxt])
        self.ez_axis.set_xlim([self.mint, self.maxt])
        self.bxyz_axis.set_xlim([self.mint, self.maxt])

    def jdote_plot(self):
        height = 6.0
        self.fig_jdote = plt.figure(figsize=[self.fig_width*2, height])
        w1, h1 = 0.88, 0.4
        xs, ys = 0.10, 0.97-h1
        gap = 0.05

        self.jdote_xyz_axis = self.fig_jdote.add_axes([xs, ys, w1, h1])
        self.pjdote_x, = self.jdote_xyz_axis.plot(self.t, self.jdote_x, color='r')
        self.pjdote_y, = self.jdote_xyz_axis.plot(self.t, self.jdote_y, color='g')
        self.pjdote_z, = self.jdote_xyz_axis.plot(self.t, self.jdote_z, color='b')
        self.jdote_xyz_axis.tick_params(labelsize=20)
        self.jdote_xyz_axis.tick_params(axis='x', labelbottom='off')
        self.jdote_xyz_axis.plot([self.mint, self.maxt], [0, 0], '--', color='k')
        if self.species == 'e':
            self.charge_name = '-e'
        else:
            self.charge_name = 'e'
        text1 = r'$' + self.charge_name + 'u_x' + 'E_x' + '$'
        self.jdote_xyz_axis.text(0.1, 0.1, text1, color='red', fontsize=32, 
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='center', verticalalignment='center',
                transform = self.jdote_xyz_axis.transAxes)
        text2 = r'$' + self.charge_name + 'u_y' + 'E_y' + '$'
        self.jdote_xyz_axis.text(0.2, 0.1, text2, color='green', fontsize=32, 
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='center', verticalalignment='center',
                transform = self.jdote_xyz_axis.transAxes)
        text3 = r'$' + self.charge_name + 'u_z' + 'E_z' + '$'
        self.jdote_xyz_axis.text(0.3, 0.1, text3, color='blue', fontsize=32, 
                bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
                horizontalalignment='center', verticalalignment='center',
                transform = self.jdote_xyz_axis.transAxes)

        ys -= h1 + gap
        self.jdote_cum_axis = self.fig_jdote.add_axes([xs, ys, w1, h1])
        self.pjdote_x_cum, = self.jdote_cum_axis.plot(self.t, self.jdote_x_cum, color='r')
        self.pjdote_y_cum, = self.jdote_cum_axis.plot(self.t, self.jdote_y_cum, color='g')
        self.pjdote_z_cum, = self.jdote_cum_axis.plot(self.t, self.jdote_z_cum, color='b')
        self.pjdote_t_cum, = self.jdote_cum_axis.plot(self.t, self.jdote_tot_cum, color='k')
        self.jdote_cum_axis.tick_params(labelsize=20)
        self.jdote_cum_axis.set_xlabel(r'$t\Omega_{ci}$', fontdict=font)
        self.jdote_cum_axis.plot([self.mint, self.maxt], [0, 0], '--', color='k')

    def save_figures(self):
        fname = self.fig_dir + 'traj_' + self.species + '_' + \
                str(self.iptl).zfill(4) + '_1.jpg'
        self.fig_ene.savefig(fname)
        # fname = self.fig_dir + 'traj_' + self.species + '_' + \
        #         str(self.iptl).zfill(4) + '_2.jpg'
        # self.fig_emf.savefig(fname)
        # fname = self.fig_dir + 'traj_' + self.species + '_' + \
        #         str(self.iptl).zfill(4) + '_3.jpg'
        # self.fig_jdote.savefig(fname)

    def save_figures_energetic(self):
        fname = self.fig_dir_energetic + 'traj_' + self.species + '_' + \
                str(self.iptl).zfill(4) + '_1.jpg'
        self.fig_ene.savefig(fname)

    def update_particle(self):
        self.ptl = read_traj_data(self.traj_names[self.iptl])
        self.calc_derived_particle_info()
        self.get_particle_current_time()
        self.update_fig_ene_indicator()
        self.update_fig_ene()
        # self.update_fig_emf()
        # self.update_fig_jdote()
        if ((self.gama[-1] - 1) > self.threshold_ene):
            self.save_figures_energetic()
        else:
            self.save_figures()

    def update_fig_ene(self):
        self.pxz.set_xdata(self.px)
        self.pxz.set_ydata(self.pz)
        self.pxy.set_xdata(self.px)
        self.pxy.set_ydata(self.py)
        self.pxy_b.set_xdata(self.pxb)
        self.pxy_b.set_ydata(self.py)
        self.pxy_help_b.set_xdata([self.xmin_b, self.xmax_b])
        self.pte.set_xdata(self.t)
        self.pte.set_ydata(self.gama - 1.0)
        self.pxe.set_xdata(self.px)
        self.pxe_b.set_xdata(self.pxb)
        self.pye.set_xdata(self.py)
        self.pze.set_xdata(self.pz)
        self.pxe.set_ydata(self.gama - 1.0)
        self.pxe_b.set_ydata(self.gama - 1.0)
        self.pye.set_ydata(self.gama - 1.0)
        self.pze.set_ydata(self.gama - 1.0)
        # self.xz_axis.relim()
        self.te_axis.relim()
        self.xy_axis.relim()
        self.xe_axis.relim()
        self.xyb_axis.relim()
        self.xeb_axis.relim()
        self.ye_axis.relim()
        self.ze_axis.relim()
        self.xz_axis.autoscale()
        self.te_axis.autoscale()
        self.xy_axis.autoscale()
        self.xe_axis.autoscale()
        self.xyb_axis.autoscale()
        self.xeb_axis.autoscale()
        self.ye_axis.autoscale()
        self.ze_axis.autoscale()
        self.xy_axis.set_xlim([self.xmin, self.xmax])
        self.xe_axis.set_xlim([self.xmin, self.xmax])
        self.xyb_axis.set_xlim([self.xmin_b, self.xmax_b])
        self.xeb_axis.set_xlim([self.xmin_b, self.xmax_b])
        self.fig_ene.canvas.draw_idle()

    def update_fig_ene_indicator(self):
        self.pxz_dot.set_xdata(self.x0)
        self.pxz_dot.set_ydata(self.z0)
        self.pxy_dot.set_xdata(self.x0)
        self.pxy_dot.set_ydata(self.y0)
        self.pxby_dot.set_xdata(self.xb0)
        self.pxby_dot.set_ydata(self.y0)
        self.pte_dot.set_xdata(self.t0)
        self.pte_dot.set_ydata(self.gama0 - 1.0)
        self.pxe_dot.set_xdata(self.x0)
        self.pxbe_dot.set_xdata(self.xb0)
        self.pye_dot.set_xdata(self.y0)
        self.pze_dot.set_xdata(self.z0)
        self.pxe_dot.set_ydata(self.gama0 - 1.0)
        self.pxbe_dot.set_ydata(self.gama0 - 1.0)
        self.pye_dot.set_ydata(self.gama0 - 1.0)
        self.pze_dot.set_ydata(self.gama0 - 1.0)

    def update_fig_emf(self):
        self.pt_gama.set_xdata(self.t)
        self.pt_gama.set_ydata(self.gama - 1.0)
        self.pux.set_xdata(self.t)
        self.pux.set_ydata(self.ptl.ux)
        self.puy.set_xdata(self.t)
        self.puy.set_ydata(self.ptl.uy)
        self.puz.set_xdata(self.t)
        self.puz.set_ydata(self.ptl.uz)
        self.pex.set_xdata(self.t)
        self.pex.set_ydata(self.ptl.ex)
        self.pey.set_xdata(self.t)
        self.pey.set_ydata(self.ey)
        self.pez.set_xdata(self.t)
        self.pez.set_ydata(self.ez)
        self.pbx.set_xdata(self.t)
        self.pbx.set_ydata(self.ptl.bx)
        self.pby.set_xdata(self.t)
        self.pby.set_ydata(self.ptl.by)
        self.pbz.set_xdata(self.t)
        self.pbz.set_ydata(self.ptl.bz)
        self.t_gama_axis.relim()
        self.uxyz_axis.relim()
        self.ex_axis.relim()
        self.ey_axis.relim()
        self.ez_axis.relim()
        self.uxyz_axis.relim()
        self.t_gama_axis.autoscale_view()
        self.uxyz_axis.autoscale_view()
        self.ex_axis.autoscale_view()
        self.ey_axis.autoscale_view()
        self.ez_axis.autoscale_view()
        self.uxyz_axis.autoscale_view()
        self.fig_emf.canvas.draw_idle()

    def update_fig_jdote(self):
        self.pjdote_x.set_xdata(self.t)
        self.pjdote_y.set_xdata(self.t)
        self.pjdote_z.set_xdata(self.t)
        self.pjdote_x_cum.set_xdata(self.t)
        self.pjdote_y_cum.set_xdata(self.t)
        self.pjdote_z_cum.set_xdata(self.t)
        self.pjdote_x.set_ydata(self.jdote_x)
        self.pjdote_y.set_ydata(self.jdote_y)
        self.pjdote_z.set_ydata(self.jdote_z)
        self.pjdote_x_cum.set_ydata(self.jdote_x_cum)
        self.pjdote_y_cum.set_ydata(self.jdote_y_cum)
        self.pjdote_z_cum.set_ydata(self.jdote_z_cum)
        self.jdote_xyz_axis.relim()
        self.jdote_cum_axis.relim()
        self.jdote_xyz_axis.autoscale_view()
        self.jdote_cum_axis.autoscale_view()
        self.fig_jdote.canvas.draw_idle()


def plot_traj(filename, pic_info, species, iptl, mint, maxt):
    """Plot particle trajectory information.

    Args:
        filename: the filename to read the data.
        pic_info: namedtuple for the PIC simulation information.
        species: particle species. 'e' for electron. 'i' for ion.
        iptl: particle ID.
        mint, maxt: minimum and maximum time for plotting.
    """
    ptl_traj = read_traj_data(filename)
    gama = np.sqrt(ptl_traj.ux**2 + ptl_traj.uy**2 + ptl_traj.uz**2 + 1.0)
    mime = pic_info.mime
    # de scale to di scale
    ptl_x = ptl_traj.x / math.sqrt(mime)
    ptl_y = ptl_traj.y / math.sqrt(mime)
    ptl_z = ptl_traj.z / math.sqrt(mime)
    # 1/wpe to 1/wci
    t = ptl_traj.t * pic_info.dtwci / pic_info.dtwpe

    xl, xr = 0, 200
    zb, zt = -20, 20
    kwargs = {"current_time":50, "xl":xl, "xr":xr, "zb":zb, "zt":zt}
    fname = "../../data/ey.gda"
    x, z, data_2d = read_2d_fields(pic_info, fname, **kwargs) 
    x, z, Ay = read_2d_fields(pic_info, "../../data/Ay.gda", **kwargs) 
    nx, = x.shape
    nz, = z.shape

    # p1 = ax1.plot(ptl_x, gama-1.0, color='k')
    # ax1.set_xlabel(r'$t\Omega_{ci}$', fontdict=font)
    # ax1.set_ylabel(r'$E/m_ic^2$', fontdict=font)
    # ax1.tick_params(labelsize=20)

    width = 14.0
    height = 8.0
    fig = plt.figure(figsize=[width, height])
    w1, h1 = 0.82, 0.27
    xs, ys = 0.10, 0.98-h1
    ax1 = fig.add_axes([xs, ys, w1, h1])
    kwargs_plot = {"xstep":2, "zstep":2, "is_log":False, "vmin":-1.0, "vmax":1.0}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    im1, cbar1 = plot_2d_contour(x, z, data_2d, ax1, fig, **kwargs_plot)
    im1.set_cmap(plt.cm.seismic)
    ax1.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep], 
            colors='black', linewidths=0.5)
    ax1.tick_params(labelsize=20)
    ax1.tick_params(axis='x', labelbottom='off')
    ax1.set_xlim([xl, xr])
    ax1.set_ylim([zb, zt])
    ax1.set_ylabel(r'$z/d_i$', fontdict=font)
    cbar1.ax.tick_params(labelsize=20)
    cbar1.ax.set_ylabel(r'$B_y$', fontdict=font, fontsize=24)
    p1 = ax1.scatter(ptl_x, ptl_z, s=0.5)

    # ax2 = fig.add_axes([xs, ys, w1, h1])
    # p2 = ax2.plot(t, gama-1.0, color='k')
    # ax2.set_xlabel(r'$t\Omega_{ci}$', fontdict=font)
    # ax2.set_ylabel(r'$E/m_ic^2$', fontdict=font)
    # ax2.tick_params(labelsize=20)

    gap = 0.04
    ys -= h1 + gap
    ax2 = fig.add_axes([xs, ys, w1*0.98-0.05/width, h1])
    p2 = ax2.plot(ptl_x, gama-1.0, color='k')
    ax2.set_ylabel(r'$E/m_ic^2$', fontdict=font)
    ax2.tick_params(labelsize=20)
    ax2.tick_params(axis='x', labelbottom='off')
    xmin, xmax = ax2.get_xlim()

    ys -= h1 + gap
    ax3 = fig.add_axes([xs, ys, w1*0.98-0.05/width, h1])
    p3 = ax3.plot(ptl_x, ptl_y, color='k')
    ax3.set_xlabel(r'$x/d_i$', fontdict=font)
    ax3.set_ylabel(r'$y/d_i$', fontdict=font)
    ax3.tick_params(labelsize=20)

    ax1.set_xlim([xmin, xmax])
    ax3.set_xlim([xmin, xmax])

    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    dir = '../img/img_traj_' + species + '/'
    if not os.path.isdir(dir):
        os.makedirs(dir)
    fname = dir + 'traj_' + species + '_' + str(iptl).zfill(4) + '_1.jpg'
    fig.savefig(fname, dpi=300)

    height = 15.0
    fig = plt.figure(figsize=[width, height])
    w1, h1 = 0.88, 0.135
    xs, ys = 0.10, 0.98-h1
    gap = 0.025

    dt = t[1] - t[0]
    ct1 = int(mint / dt)
    ct2 = int(maxt / dt)

    ax1 = fig.add_axes([xs, ys, w1, h1])
    p1 = ax1.plot(t[ct1:ct2], gama[ct1:ct2]-1.0, color='k')
    ax1.set_ylabel(r'$E/m_ic^2$', fontdict=font)
    ax1.tick_params(labelsize=20)
    ax1.tick_params(axis='x', labelbottom='off')
    ax1.plot([mint, maxt], [0, 0], '--', color='k')
    ax1.text(0.4, -0.07, r'$x$', color='red', fontsize=32, 
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='center',
            transform = ax1.transAxes)
    ax1.text(0.5, -0.07, r'$y$', color='green', fontsize=32, 
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='center',
            transform = ax1.transAxes)
    ax1.text(0.6, -0.07, r'$z$', color='blue', fontsize=32, 
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='center',
            transform = ax1.transAxes)

    ys -= h1 + gap
    ax2 = fig.add_axes([xs, ys, w1, h1])
    p21 = ax2.plot(t[ct1:ct2], ptl_traj.ux[ct1:ct2], color='r', label=r'u_x')
    p22 = ax2.plot(t[ct1:ct2], ptl_traj.uy[ct1:ct2], color='g', label=r'u_y')
    p23 = ax2.plot(t[ct1:ct2], ptl_traj.uz[ct1:ct2], color='b', label=r'u_z')
    ax2.plot([mint, maxt], [0, 0], '--', color='k')
    ax2.set_ylabel(r'$u_x, u_y, u_z$', fontdict=font)
    ax2.tick_params(labelsize=20)
    ax2.tick_params(axis='x', labelbottom='off')

    kernel = 9
    tmax = np.max(t)
    ys -= h1 + gap
    ax31 = fig.add_axes([xs, ys, w1, h1])
    ex = signal.medfilt(ptl_traj.ex, kernel_size=(kernel))
    p31 = ax31.plot(t[ct1:ct2], ex[ct1:ct2], color='r', label=r'E_x')
    ax31.set_ylabel(r'$E_x$', fontdict=font)
    ax31.tick_params(labelsize=20)
    ax31.tick_params(axis='x', labelbottom='off')
    ax31.plot([mint, maxt], [0, 0], '--', color='k')

    ys -= h1 + gap
    ax32 = fig.add_axes([xs, ys, w1, h1])
    ey = signal.medfilt(ptl_traj.ey, kernel_size=(kernel))
    p32 = ax32.plot(t[ct1:ct2], ey[ct1:ct2], color='g', label=r'E_y')
    ax32.set_ylabel(r'$E_y$', fontdict=font)
    ax32.tick_params(labelsize=20)
    ax32.tick_params(axis='x', labelbottom='off')
    ax32.plot([mint, maxt], [0, 0], '--', color='k')

    ys -= h1 + gap
    ax33 = fig.add_axes([xs, ys, w1, h1])
    ez = signal.medfilt(ptl_traj.ez, kernel_size=(kernel))
    p33 = ax33.plot(t[ct1:ct2], ez[ct1:ct2], color='b', label=r'E_z')
    ax33.set_ylabel(r'$E_z$', fontdict=font)
    ax33.tick_params(labelsize=20)
    ax33.tick_params(axis='x', labelbottom='off')
    ax33.plot([mint, maxt], [0, 0], '--', color='k')

    ys -= h1 + gap
    ax4 = fig.add_axes([xs, ys, w1, h1])
    p41 = ax4.plot(t[ct1:ct2], ptl_traj.bx[ct1:ct2], color='r', label=r'B_x')
    p42 = ax4.plot(t[ct1:ct2], ptl_traj.by[ct1:ct2], color='g', label=r'B_y')
    p43 = ax4.plot(t[ct1:ct2], ptl_traj.bz[ct1:ct2], color='b', label=r'B_z')
    ax4.set_xlabel(r'$t\Omega_{ci}$', fontdict=font)
    ax4.set_ylabel(r'$B_x, B_y, B_z$', fontdict=font)
    ax4.tick_params(labelsize=20)
    ax4.plot([mint, maxt], [0, 0], '--', color='k')
    ax1.set_xlim([mint, maxt])
    ax2.set_xlim([mint, maxt])
    ax31.set_xlim([mint, maxt])
    ax32.set_xlim([mint, maxt])
    ax33.set_xlim([mint, maxt])
    ax4.set_xlim([mint, maxt])

    fname = dir + 'traj_' + species + '_' + str(iptl).zfill(4) + '_2.jpg'
    fig.savefig(fname, dpi=300)

    height = 6.0
    fig = plt.figure(figsize=[width, height])
    w1, h1 = 0.88, 0.4
    xs, ys = 0.10, 0.97-h1
    gap = 0.05

    dt = t[1] - t[0]
    ct1 = int(mint / dt)
    ct2 = int(maxt / dt)

    if species == 'e':
        charge = -1.0
    else:
        charge = 1.0
    nt, = t.shape
    jdote_x = ptl_traj.ux * ptl_traj.ex * charge / gama
    jdote_y = ptl_traj.uy * ptl_traj.ey * charge / gama
    jdote_z = ptl_traj.uz * ptl_traj.ez * charge / gama
    # jdote_x = (ptl_traj.uy * ptl_traj.bz / gama - 
    #            ptl_traj.uz * ptl_traj.by / gama + ptl_traj.ex) * charge
    # jdote_y = (ptl_traj.uz * ptl_traj.bx / gama - 
    #            ptl_traj.ux * ptl_traj.bz / gama + ptl_traj.ey) * charge
    # jdote_z = (ptl_traj.ux * ptl_traj.by / gama - 
    #            ptl_traj.uy * ptl_traj.bx / gama + ptl_traj.ez) * charge
    dt = np.zeros(nt)
    dt[0:nt-1] = np.diff(t)
    jdote_x_cum = np.cumsum(jdote_x) * dt
    jdote_y_cum = np.cumsum(jdote_y) * dt
    jdote_z_cum = np.cumsum(jdote_z) * dt
    jdote_tot_cum = jdote_x_cum + jdote_y_cum + jdote_z_cum
    
    ax1 = fig.add_axes([xs, ys, w1, h1])
    p1 = ax1.plot(t[ct1:ct2], jdote_x[ct1:ct2], color='r')
    p2 = ax1.plot(t[ct1:ct2], jdote_y[ct1:ct2], color='g')
    p3 = ax1.plot(t[ct1:ct2], jdote_z[ct1:ct2], color='b')
    ax1.tick_params(labelsize=20)
    ax1.tick_params(axis='x', labelbottom='off')
    ax1.plot([mint, maxt], [0, 0], '--', color='k')
    if species == 'e':
        charge = '-e'
    else:
        charge = 'e'
    text1 = r'$' + charge + 'u_x' + 'E_x' + '$'
    ax1.text(0.1, 0.1, text1, color='red', fontsize=32, 
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='center',
            transform = ax1.transAxes)
    text2 = r'$' + charge + 'u_y' + 'E_y' + '$'
    ax1.text(0.2, 0.1, text2, color='green', fontsize=32, 
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='center',
            transform = ax1.transAxes)
    text3 = r'$' + charge + 'u_z' + 'E_z' + '$'
    ax1.text(0.3, 0.1, text3, color='blue', fontsize=32, 
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='center',
            transform = ax1.transAxes)

    ys -= h1 + gap
    ax2 = fig.add_axes([xs, ys, w1, h1])
    p1 = ax2.plot(t[ct1:ct2], jdote_x_cum[ct1:ct2], color='r')
    p2 = ax2.plot(t[ct1:ct2], jdote_y_cum[ct1:ct2], color='g')
    p3 = ax2.plot(t[ct1:ct2], jdote_z_cum[ct1:ct2], color='b')
    p4 = ax2.plot(t[ct1:ct2], jdote_tot_cum[ct1:ct2], color='k')
    ax2.tick_params(labelsize=20)
    ax2.set_xlabel(r'$t\Omega_{ci}$', fontdict=font)
    ax2.plot([mint, maxt], [0, 0], '--', color='k')

    plt.show()


def plot_ptl_traj(filename, pic_info, species, iptl, mint, maxt):
    """Plot particle trajectory information.

    Args:
        filename: the filename to read the data.
        pic_info: namedtuple for the PIC simulation information.
        species: particle species. 'e' for electron. 'i' for ion.
        iptl: particle ID.
        mint, maxt: minimum and maximum time for plotting.
    """
    ptl_traj = read_traj_data(filename)
    gama = np.sqrt(ptl_traj.ux**2 + ptl_traj.uy**2 + ptl_traj.uz**2 + 1.0)
    mime = pic_info.mime
    # de scale to di scale
    ptl_x = ptl_traj.x / math.sqrt(mime)
    ptl_y = ptl_traj.y / math.sqrt(mime)
    ptl_z = ptl_traj.z / math.sqrt(mime)
    # 1/wpe to 1/wci
    t = ptl_traj.t * pic_info.dtwci / pic_info.dtwpe

    xl, xr = 0, 50
    zb, zt = -20, 20
    kwargs = {"current_time":65, "xl":xl, "xr":xr, "zb":zb, "zt":zt}
    fname = "../../data/uex.gda"
    x, z, uex = read_2d_fields(pic_info, fname, **kwargs) 
    fname = "../../data/ne.gda"
    x, z, ne = read_2d_fields(pic_info, fname, **kwargs) 
    fname = "../../data/uix.gda"
    x, z, uix = read_2d_fields(pic_info, fname, **kwargs) 
    fname = "../../data/ni.gda"
    x, z, ni = read_2d_fields(pic_info, fname, **kwargs) 
    ux = (uex*ne + uix*ni*pic_info.mime) / (ne + ni*pic_info.mime)
    x, z, Ay = read_2d_fields(pic_info, "../../data/Ay.gda", **kwargs) 
    nx, = x.shape
    nz, = z.shape

    width = 8.0
    height = 8.0
    fig = plt.figure(figsize=[width, height])
    w1, h1 = 0.74, 0.42
    xs, ys = 0.13, 0.98-h1
    ax1 = fig.add_axes([xs, ys, w1, h1])
    kwargs_plot = {"xstep":2, "zstep":2, "is_log":False, "vmin":-1.0, "vmax":1.0}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    va = 0.2  # Alfven speed
    im1, cbar1 = plot_2d_contour(x, z, ux/va, ax1, fig, **kwargs_plot)
    im1.set_cmap(plt.cm.seismic)
    ax1.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep], 
            colors='black', linewidths=0.5)
    ax1.tick_params(labelsize=20)
    ax1.tick_params(axis='x', labelbottom='off')
    ax1.set_xlim([xl, xr])
    ax1.set_ylim([zb, zt])
    ax1.set_ylabel(r'$z/d_i$', fontdict=font)
    cbar1.ax.tick_params(labelsize=20)
    cbar1.ax.set_ylabel(r'$u_x/V_A$', fontdict=font, fontsize=24)
    tstop = 1990
    p1 = ax1.plot(ptl_x[0:tstop], ptl_z[0:tstop], linewidth=2, color='k')

    gap = 0.04
    ys -= h1 + gap
    ax2 = fig.add_axes([xs, ys, w1*0.98-0.05/width, h1])
    eth = pic_info.vthi**2 * 3
    p2 = ax2.plot(ptl_x[0:tstop], (gama[0:tstop]-1.0)/eth, color='k', linewidth=2)
    ax2.set_xlabel(r'$x/d_i$', fontdict=font)
    ax2.set_ylabel(r'$E/E_{\text{thi}}$', fontdict=font)
    ax2.tick_params(labelsize=20)
    # ax2.tick_params(axis='x', labelbottom='off')
    xmin, xmax = ax1.get_xlim()
    ax2.set_xlim([xmin, xmax])

    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    fname = '../img/' + 'traj_' + species + '_' + str(iptl).zfill(4) + '_1.jpg'
    fig.savefig(fname, dpi=300)

    plt.show()


def plot_particle_trajectory(fnames, species, pic_info):
    """Plot particle trajectory.

    Args:
        fnames: file names for the trajectory files.
        species: particle species.
        pic_info: particle information namedtuple.
    """
    init_ft = 190
    nptl = len(fnames)
    var_field = 'ey'
    var_name = '$E_y$'
    kwargs = {"current_time":init_ft, "xl":0, "xr":200, "zb":-50, "zt":50}
    fname = '../../data/' + var_field + '.gda'
    x, z, data = read_2d_fields(pic_info, fname, **kwargs) 
    ng = 3
    kernel = np.ones((ng,ng)) / float(ng*ng)
    data = signal.convolve2d(data, kernel)
    fname = '../../data/Ay.gda'
    x, z, Ay = read_2d_fields(pic_info, fname, **kwargs) 
    iptl = 393
    fig_v = ParticleTrajectory(nptl, iptl, x, z, data, Ay, init_ft, var_field,
            var_name, species, fnames)
    plt.show()
    # for iptl in range(nptl):
    # # for iptl in range(70, 80):
    #     print(iptl)
    #     fig_v.iptl = iptl
    #     fig_v.update_particle()
    # plt.close()


if __name__ == "__main__":
    pic_info = pic_information.get_pic_info('../../')
    ntp = pic_info.ntp
    vthe = pic_info.vthe
    fnames_e, fnames_i = get_file_names()
    iptl = 479
    # plot_traj(fnames_e[iptl], pic_info, 'i', iptl, 0, 800)
    # plot_ptl_traj(fnames_i[iptl], pic_info, 'i', iptl, 0, 400)
    plot_particle_trajectory(fnames_e, 'e', pic_info)
    # plot_particle_trajectory(fnames_i, 'i')
