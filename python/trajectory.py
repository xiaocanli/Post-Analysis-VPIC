"""
Test if the 'q' key is unique.
"""
import os
import h5py
import math
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import MaxNLocator
from matplotlib.widgets import Cursor, Button, Slider
from mpl_toolkits.axes_grid1 import make_axes_locatable
import collections
import timeit
import struct
import collections
import pic_information
from contour_plots import read_2d_fields, plot_2d_contour
import color_maps as cm
import colormap.colormaps as cmaps
import palettable

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

font = {'family' : 'serif',
        #'color'  : 'darkred',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 24,
        }

particles = collections.namedtuple("particles", 
        ["x", "y", "z", "ux", "uy", "uz"])

class Viewer2d(object):
    def __init__(self, file, nptl, x, y, fdata, init_ft, var_field, var_name,
            species):
        """
        Shows a given array in a 2d-viewer.
        Input: z, an 2d array.
        x,y coordinters are optional.
        """
        self.x = x
        self.y = y
        self.fdata = fdata
        self.file = file
        self.nptl = nptl
        self.ct = init_ft
        self.iptl = 0
        self.particle_tags = []
        self.species = species
        for item in self.file:
            self.particle_tags.append(item)
        group = file[self.particle_tags[self.iptl]]
        dset_ux = group['Ux']
        self.sz, = dset_ux.shape
        self.particle_info = ParticleInfo(self.sz)
        self.ptl = self.particle_info.get_particle_info(group)
        self.pic_info = pic_information.get_pic_info('../../')
        self.smime = math.sqrt(self.pic_info.mime)
        self.px = self.ptl.x / self.smime
        self.py = self.ptl.y / self.smime
        self.pz = self.ptl.z / self.smime
        self.gama = np.sqrt(self.ptl.ux**2 + self.ptl.uy**2 + self.ptl.uz**2 + 1.0)

        self.fig = plt.figure(figsize=(10, 10))

        self.pxz_axis = self.fig.add_axes([0.1, 0.73, 0.8, 0.25])
        vmax = min(abs(np.min(self.fdata)), abs(np.max(self.fdata)))
        vmax *= 0.2
        self.im1 = self.pxz_axis.imshow(self.fdata, cmap=plt.cm.seismic,
                extent=[np.min(self.x), np.max(self.x), np.min(self.y), np.max(self.y)],
                aspect='auto', origin='lower',
                vmin = -vmax, vmax = vmax,
                interpolation='bicubic')
        divider = make_axes_locatable(self.pxz_axis)
        self.cax = divider.append_axes("right", size="2%", pad=0.05)
        self.cbar = self.fig.colorbar(self.im1, cax=self.cax)
        self.cbar.ax.tick_params(labelsize=16)
        self.cbar.ax.set_ylabel(var_name, fontdict=font, fontsize=24)
        self.pxz_axis.tick_params(labelsize=16)
        self.pxz_axis.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
        self.pxz_axis.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
        self.pxz_axis.autoscale(1,'both',1)

        # xz plot
        self.pxz, = self.pxz_axis.plot(self.px, self.pz, linewidth=2,
                color='k', marker='.', markersize=1, linestyle='')

        # Energy plot
        self.ene_axis = self.fig.add_axes([0.1, 0.46, 0.35, 0.2])
        self.pene, = self.ene_axis.plot(self.gama - 1.0, linewidth=2, color='k')
        self.ene_axis.set_ylabel(r'$\gamma - 1$', fontdict=font, fontsize=20)
        self.ene_axis.set_xlabel(r'Time', fontdict=font, fontsize=20)
        self.ene_axis.tick_params(labelsize=16)

        # x-energy
        self.xe_axis = self.fig.add_axes([0.6, 0.46, 0.35, 0.2])
        self.xe_axis.tick_params(labelsize=16)
        self.pxe, = self.xe_axis.plot(self.px, self.gama - 1.0,
                linewidth=2, color='r')
        self.xe_axis.set_ylabel(r'$\gamma - 1$', fontdict=font, fontsize=20)
        self.xe_axis.set_xlabel(r'$x$', fontdict=font, fontsize=20)

        # y-energy
        self.ye_axis = self.fig.add_axes([0.1, 0.2, 0.35, 0.2])
        self.ye_axis.tick_params(labelsize=16)
        self.pye, = self.ye_axis.plot(self.py, self.gama - 1.0,
                linewidth=2, color='g')
        self.ye_axis.set_ylabel(r'$\gamma - 1$', fontdict=font, fontsize=20)
        self.ye_axis.set_xlabel(r'$y$', fontdict=font, fontsize=20)

        # z-energy
        self.ze_axis = self.fig.add_axes([0.6, 0.2, 0.35, 0.2])
        self.ze_axis.tick_params(labelsize=16)
        self.pze, = self.ze_axis.plot(self.pz, self.gama - 1.0,
                linewidth=2, color='b')
        self.ze_axis.set_ylabel(r'$\gamma - 1$', fontdict=font, fontsize=20)
        self.ze_axis.set_xlabel(r'$z$', fontdict=font, fontsize=20)

        # Slider to choose particle
        self.sliderax = plt.axes([0.1, 0.1, 0.8, 0.03])
        particle_list = np.arange(1, self.nptl)
        self.slider = DiscreteSlider(self.sliderax, 'Particle', 1, self.nptl,\
                allowed_vals=particle_list, valinit=particle_list[self.iptl])
        self.slider.on_changed(self.update_particle_slider)

        # Slider to choose time frames for fields
        self.field_sliderax = plt.axes([0.1, 0.05, 0.8, 0.03])
        tframes = np.arange(1, self.pic_info.ntf)
        self.field_slider = DiscreteSlider(self.field_sliderax, 'Field', 1,
                self.pic_info.ntf, allowed_vals=tframes, valinit=tframes[self.ct-1])
        self.field_slider.on_changed(self.update_field)

        self._widgets=[self.slider, self.field_slider]
        self.save_figure()

    def update_particle_slider(self, val):
        self.iptl = self.slider.val - 1
        self.update_particle()

    def update_particle(self):
        group = file[self.particle_tags[self.iptl]]
        self.ptl = self.particle_info.get_particle_info(group)
        self.px = self.ptl.x / self.smime
        self.py = self.ptl.y / self.smime
        self.pz = self.ptl.z / self.smime
        self.gama = np.sqrt(self.ptl.ux**2 + self.ptl.uy**2 + self.ptl.uz**2 + 1.0)
        emax = np.max(self.gama - 1.0)
        xmin = np.min(self.px)
        xmax = np.max(self.px)
        ymin = np.min(self.py)
        ymax = np.max(self.py)
        zmin = np.min(self.pz)
        zmax = np.max(self.pz)
        self.ene_axis.set_ylim([0, emax*1.1])
        self.xe_axis.set_ylim([0, emax*1.1])
        self.ye_axis.set_ylim([0, emax*1.1])
        self.ze_axis.set_ylim([0, emax*1.1])
        self.xe_axis.set_xlim([xmin*0.9, xmax*1.1])
        if ymin < 0:
            yl = ymin * 1.1
        else:
            yl = ymin * 0.9

        if ymax < 0:
            yr = ymax * 0.9
        else:
            yr = ymax * 1.1

        if zmin < 0:
            zl = zmin * 1.1
        else:
            zl = zmin * 0.9

        if zmax < 0:
            zr = zmax * 0.9
        else:
            zr = zmax * 1.1
        self.ye_axis.set_xlim([yl, yr])
        self.ze_axis.set_xlim([zl, zr])
        self.pxz.set_xdata(self.px)
        self.pxz.set_ydata(self.pz)
        self.pene.set_ydata(self.gama - 1.0)
        self.pxe.set_xdata(self.px)
        self.pye.set_xdata(self.py)
        self.pze.set_xdata(self.pz)
        self.pxe.set_ydata(self.gama - 1.0)
        self.pye.set_ydata(self.gama - 1.0)
        self.pze.set_ydata(self.gama - 1.0)
        self.fig.canvas.draw_idle()
        self.save_figure()

    def update_field(self, val):
        self.ct = self.field_slider.val
        kwargs = {"current_time":self.ct, "xl":0, "xr":400, "zb":-100, "zt":100}
        fname = '../../data/' + var_field + '.gda'
        x, z, self.fdata = read_2d_fields(self.pic_info, fname, **kwargs) 
        self.im1.set_data(self.fdata)
        self.fig.canvas.draw_idle()

    def save_figure(self):
        if not os.path.isdir('../img/'):
            os.makedirs('../img/')
        if not os.path.isdir('../img/img_traj3/'):
            os.makedirs('../img/img_traj3/')
        fname = '../img/img_traj3/traj_' + str(self.iptl) + '_' + \
                str(self.ct).zfill(3) + '_' + self.species + '.jpg'
        self.fig.savefig(fname, dpi=200)


class DiscreteSlider(Slider):
    """A matplotlib slider widget with discrete steps."""
    def __init__(self, *args, **kwargs):
        """
        Identical to Slider.__init__, except for the new keyword 'allowed_vals'.
        This keyword specifies the allowed positions of the slider
        """
        self.allowed_vals = kwargs.pop('allowed_vals',None)
        self.previous_val = kwargs['valinit']
        Slider.__init__(self, *args, **kwargs)
        if self.allowed_vals is None:
            self.allowed_vals = [self.valmin,self.valmax]

    def set_val(self, val):
        discrete_val = self.allowed_vals[abs(val-self.allowed_vals).argmin()]
        xy = self.poly.xy
        xy[2] = discrete_val, 1
        xy[3] = discrete_val, 0
        self.poly.xy = xy
        self.valtext.set_text(self.valfmt % discrete_val)
        if self.drawon: 
            self.ax.figure.canvas.draw()
        self.val = discrete_val
        if self.previous_val!=discrete_val:
            self.previous_val = discrete_val
            if not self.eventson: 
                return
            for cid, func in self.observers.iteritems():
                func(discrete_val)


class ParticleInfo(object):
    def __init__(self, sz):
        """
        Initialize information for one particle.
        """
        self.ux = np.zeros(sz)
        self.uy = np.zeros(sz)
        self.uz = np.zeros(sz)
        self.x = np.zeros(sz)
        self.y = np.zeros(sz)
        self.z = np.zeros(sz)

    def get_particle_info(self, group):
        """
        Read the information.
        """
        dset_ux = group['Ux']
        dset_uy = group['Uy']
        dset_uz = group['Uz']
        dset_x = group['dX']
        dset_y = group['dY']
        dset_z = group['dZ']
        dset_ux.read_direct(self.ux)
        dset_uy.read_direct(self.uy)
        dset_uz.read_direct(self.uz)
        dset_x.read_direct(self.x)
        dset_y.read_direct(self.y)
        dset_z.read_direct(self.z)
        ptl = particles(x=self.x, y=self.y, z=self.z,
                ux=self.ux, uy=self.uy, uz=self.uz)
        return ptl

def plot_ptl_traj(file, species):
    """Plot particle trajectorie.

    Args:
        file: the file ID for a HDF5 file.
        species: particle species.
    """
    init_ft = 40
    ngroups = len(file)
    var_field = 'jy'
    var_name = '$j_y$'
    kwargs = {"current_time":init_ft, "xl":0, "xr":400, "zb":-100, "zt":100}
    fname = '../../data/' + var_field + '.gda'
    x, z, data = read_2d_fields(pic_info, fname, **kwargs) 
    fig_v = Viewer2d(file, ngroups, x, z, data, init_ft, var_field, var_name,
            species)
    plt.show()
    # for iptl in range(fig_v.nptl):
    #     print(iptl)
    #     fig_v.iptl = iptl
    #     fig_v.update_particle()
    plt.close()


def get_particle_info(fname, iptl):
    """Get particle information from a file.

    Args:
        fname: file name for the HDF5 file.
        iptl: particle index.
    """
    file = h5py.File(fname, 'r')
    pic_info = pic_information.get_pic_info('../../')
    particle_tags = []
    for item in file:
        particle_tags.append(item)
    group = file[particle_tags[iptl]]
    dset_ux = group['Ux']
    sz, = dset_ux.shape
    particle_info = ParticleInfo(sz)
    ptl = particle_info.get_particle_info(group)
    pic_info = pic_information.get_pic_info('../../')
    smime = math.sqrt(pic_info.mime)
    px = ptl.x / smime
    py = ptl.y / smime
    pz = ptl.z / smime
    gama = np.sqrt(ptl.ux**2 + ptl.uy**2 + ptl.uz**2 + 1.0)
    file.close()
    return (px, py, pz, gama)


def plot_ptl_traj_direct():
    """Plot multiple particle trajectories in the same file.

    This is for direct acceleration.
    """
    filepath = '/net/scratch1/guofan/share/ultra-sigma/'
    filepath += 'sigma1e4-mime100-4000-track/pic_analysis/vpic-sorter/data/'
    fname = filepath + 'electrons.h5p'
    iptl = 739
    pxe, pye, pze, gamae = get_particle_info(fname, iptl)

    filepath = '/net/scratch1/guofan/share/ultra-sigma/'
    filepath += 'sigma1e4-mime100-4000-track/pic_analysis/vpic-sorter/data/'
    fname = filepath + 'ions.h5p'
    iptl = 202
    pxi, pyi, pzi, gamai = get_particle_info(fname, iptl)

    var_field = 'ey'
    var_name = '$E_y$'
    kwargs = {"current_time":40, "xl":200, "xr":400, "zb":-50, "zt":50}
    fname = '../../data/' + var_field + '.gda'
    x, z, fdata = read_2d_fields(pic_info, fname, **kwargs) 
    fname = '../../data/Ay.gda'
    x, z, Ay = read_2d_fields(pic_info, fname, **kwargs) 

    # Change from di to de
    smime = math.sqrt(pic_info.mime)
    x *= smime
    z *= smime
    pxi *= smime
    pyi *= smime
    pzi *= smime
    pxe *= smime
    pye *= smime
    pze *= smime

    colors = palettable.colorbrewer.qualitative.Set1_9.mpl_colors
    fig = plt.figure(figsize=(7, 6))

    xs, ys = 0.15, 0.73
    width, height = 0.75, 0.24
    gap = 0.04

    pxz_axis = fig.add_axes([xs, ys, width, height])
    vmax = min(abs(np.min(fdata)), abs(np.max(fdata)))
    vmax *= 0.2
    im1 = pxz_axis.imshow(fdata, cmap=plt.cm.jet,
            extent=[np.min(x), np.max(x), np.min(z), np.max(z)],
            aspect='auto', origin='lower',
            vmin = -vmax, vmax = vmax,
            interpolation='bicubic')
    im1.set_cmap('coolwarm')
    divider = make_axes_locatable(pxz_axis)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    cbar = fig.colorbar(im1, cax=cax)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_ticks(np.arange(-60, 70, 30))
    pxz_axis.contour(x, z, Ay, colors='black', linewidths=0.5)
    pxz_axis.tick_params(labelsize=16)
    pxz_axis.set_ylabel(r'$z/d_e$', fontdict=font, fontsize=20)
    pxz_axis.tick_params(axis='x', labelbottom='off')
    pxz_axis.autoscale(1,'both',1)
    pxz_axis.text(0.05, 0.8, var_name, color='black', fontsize=24, 
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='center',
            transform = pxz_axis.transAxes)

    # xz plot
    tmax_e = 1100
    tmax_i = 1000
    pxze, = pxz_axis.plot(pxe[:tmax_e], pze[:tmax_e], linewidth=2,
            color='r')
    pxzi, = pxz_axis.plot(pxi[:tmax_i], pzi[:tmax_i], linewidth=2,
            color='b')

    # x-energy
    ys -= height + gap
    w1, h1 = fig.get_size_inches()
    width1 = width * 0.98 - 0.05 / w1
    xe_axis = fig.add_axes([xs, ys, width1, height])
    xe_axis.tick_params(labelsize=16)
    pxe, = xe_axis.plot(pxe[:tmax_e], (gamae[:tmax_e] - 1.0)/1E4,
            linewidth=2, color='r', label='electron')
    xe_axis.set_ylabel(r'$E_k/(10^4m_ec^2)$', fontdict=font,
            fontsize=20)
    xe_axis.set_xlabel(r'$x/d_e$', fontdict=font, fontsize=20)
    xe_axis.set_xlim([np.min(x), np.max(x)])
    # for tl in xe_axis.get_yticklabels():
    #     tl.set_color('r')
    # ax1 = xe_axis.twinx()
    pene, = xe_axis.plot(pxi[:tmax_i], (gamai[:tmax_i] - 1.0)*pic_info.mime/1E4,
            linewidth=2, color='b', label='ion')
    # xe_axis.set_ylabel(r'$(\gamma_i - 1)/10^2$', fontdict=font, fontsize=20)
    xe_axis.tick_params(labelsize=16)
    xe_axis.set_xlim([np.min(x), np.max(x)])
    # for tl in ax1.get_yticklabels():
    #     tl.set_color('b')
    # leg = xe_axis.legend(loc=2, prop={'size':20}, ncol=1,
    #         shadow=False, fancybox=False, frameon=False)
    xe_axis.text(0.05, 0.8, 'Electron',
            color='red', fontsize=20,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform = xe_axis.transAxes)
    xe_axis.text(0.05, 0.6, 'Ion',
            color='blue', fontsize=20,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform = xe_axis.transAxes)

    # Energy plot
    nt, = gamae.shape
    # This is not general
    tptl = np.arange(nt) * pic_info.fields_interval / 20 * pic_info.dtwpe
    gap = 0.1
    ys -= height + gap
    ene_axis = fig.add_axes([xs, ys, width1, height])
    pene, = ene_axis.plot(tptl[:tmax_e], (gamae[:tmax_e] - 1.0)/1E4,
            linewidth=2, color='r')
    ene_axis.set_ylabel(r'$E_k/(10^4m_ec^2)$', fontdict=font,
            fontsize=20)
    ene_axis.set_xlabel(r'$t\omega_{pe}$', fontdict=font, fontsize=20)
    ene_axis.tick_params(labelsize=16)
    # for tl in ene_axis.get_yticklabels():
    #     tl.set_color('r')

    # ax1 = ene_axis.twinx()
    pene, = ene_axis.plot(tptl[:tmax_i], (gamai[:tmax_i] - 1.0)*pic_info.mime/1E4,
            linewidth=2)
    # ax1.set_ylabel(r'$E_k/(10^4m_ec^2)$', fontdict=font, fontsize=20)
    ene_axis.tick_params(labelsize=16)
    # for tl in ax1.get_yticklabels():
    #     tl.set_color('b')

    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    fname = '../img/traj_direct.eps'
    fig.savefig(fname)

    plt.show()


def plot_ptl_traj_fermi():
    """Plot multiple particle trajectories in the same file.

    This is for Fermi acceleration.
    """
    # Ratio of fields interval to particle tracking interval
    ratio_interval_fields_track = 20
    filepath = '/net/scratch1/guofan/share/ultra-sigma/'
    filepath += 'sigma1e4-mime100-4000-track/pic_analysis/vpic-sorter/data/'
    fname = filepath + 'electrons.h5p'
    # iptl = 330
    iptl = 68
    pxe, pye, pze, gamae = get_particle_info(fname, iptl)

    filepath = '/net/scratch1/guofan/share/ultra-sigma/'
    filepath += 'sigma1e4-mime100-4000-track/pic_analysis/vpic-sorter/data/'
    fname = filepath + 'ions_2.h5p'
    # iptl = 478
    iptl = 659
    pxi, pyi, pzi, gamai = get_particle_info(fname, iptl)

    # var_field = 'ey'
    # var_name = '$E_y$'
    var_field = 'vx'
    var_name = '$v_x$'
    ct = 49
    ct_wpe = pic_info.fields_interval * ct * pic_info.dtwpe
    ct_track = ct * ratio_interval_fields_track
    kwargs = {"current_time":ct, "xl":100, "xr":200, "zb":-50, "zt":50}
    fname = '../../data/' + var_field + '.gda'
    x, z, fdata = read_2d_fields(pic_info, fname, **kwargs) 
    fname = '../../data/Ay.gda'
    x, z, Ay = read_2d_fields(pic_info, fname, **kwargs) 

    # Change from di to de
    smime = math.sqrt(pic_info.mime)
    x *= smime
    z *= smime
    pxi *= smime
    pyi *= smime
    pzi *= smime
    pxe *= smime
    pye *= smime
    pze *= smime

    colors = palettable.colorbrewer.qualitative.Set1_9.mpl_colors
    fig = plt.figure(figsize=(7, 6))

    xs, ys = 0.15, 0.73
    width, height = 0.75, 0.24
    gap = 0.04

    pxz_axis = fig.add_axes([xs, ys, width, height])
    vmax = min(abs(np.min(fdata)), abs(np.max(fdata)))
    # vmax = 1.0
    im1 = pxz_axis.imshow(fdata, cmap=plt.cm.jet,
            extent=[np.min(x), np.max(x), np.min(z), np.max(z)],
            aspect='auto', origin='lower',
            vmin = -vmax, vmax = vmax,
            interpolation='bicubic')
    # im1.set_cmap(cmaps.viridis)
    im1.set_cmap('coolwarm')
    divider = make_axes_locatable(pxz_axis)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    cbar = fig.colorbar(im1, cax=cax)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_ticks(np.arange(-0.8, 1.0, 0.4))
    pxz_axis.contour(x, z, Ay, colors='black', linewidths=0.5)
    pxz_axis.tick_params(labelsize=16)
    pxz_axis.set_ylabel(r'$z/d_e$', fontdict=font, fontsize=20)
    pxz_axis.tick_params(axis='x', labelbottom='off')
    pxz_axis.autoscale(1,'both',1)
    pxz_axis.text(0.05, 0.8, var_name, color='black', fontsize=24, 
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='center', verticalalignment='center',
            transform = pxz_axis.transAxes)
    pxz_axis.set_xlim([np.min(x), np.max(x)])
    pxz_axis.set_ylim([np.min(z), np.max(z)])

    # xz plot
    nt, = pxe.shape
    # tmax_e = 1600
    tmax_e = 1200
    tmax_i = 1500
    pxze, = pxz_axis.plot(pxe[:tmax_e], pze[:tmax_e], linewidth=2,
            color='r')
    pxzi, = pxz_axis.plot(pxi[:tmax_i], pzi[:tmax_i], linewidth=2,
            color='b')
    pdot_e, = pxz_axis.plot(pxe[ct_track], pze[ct_track], marker='x',
            markersize=10, linestyle='None', color='r')
    pdot_i, = pxz_axis.plot(pxi[ct_track], pzi[ct_track], marker='x',
            markersize=10, linestyle='None', color='b')

    # x-energy
    ys -= height + gap
    w1, h1 = fig.get_size_inches()
    width1 = width * 0.98 - 0.05 / w1
    xe_axis = fig.add_axes([xs, ys, width1, height])
    xe_axis.tick_params(labelsize=16)
    kee = (gamae - 1.0)/1E4
    px_ene_e, = xe_axis.plot(pxe[:tmax_e], kee[:tmax_e], linewidth=2, color='r')
    xe_axis.set_ylabel(r'$E_k/(10^4m_ec^2)$', fontdict=font,
            fontsize=20)
    xe_axis.set_xlabel(r'$x/d_e$', fontdict=font, fontsize=20)
    xe_axis.set_xlim([np.min(x), np.max(x)])
    # for tl in xe_axis.get_yticklabels():
    #     tl.set_color('r')
    # ax1 = xe_axis.twinx()
    kei = (gamai - 1.0) * pic_info.mime / 1E4
    px_ene_i, = xe_axis.plot(pxi[:tmax_i], kei[:tmax_i], linewidth=2, color='b')
    # xe_axis.set_ylabel(r'$(\gamma_i - 1)/10^2$', fontdict=font, fontsize=20)
    xe_axis.tick_params(labelsize=16)
    xe_axis.set_xlim([np.min(x), np.max(x)])
    # for tl in ax1.get_yticklabels():
    #     tl.set_color('b')
    xe_axis.text(0.05, 0.8, 'Electron',
            color='red', fontsize=20,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform = xe_axis.transAxes)
    xe_axis.text(0.05, 0.6, 'Ion',
            color='blue', fontsize=20,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform = xe_axis.transAxes)
    pxe_e, = xe_axis.plot(pxe[ct_track], kee[ct_track], marker='x',
            markersize=10, linestyle='None', color='r')
    pxe_i, = xe_axis.plot(pxi[ct_track], kei[ct_track], marker='x',
            markersize=10, linestyle='None', color='b')

    # Energy plot
    nt, = gamae.shape
    # This is not general
    tptl = np.arange(nt) * pic_info.fields_interval * pic_info.dtwpe 
    tptl /= ratio_interval_fields_track
    gap = 0.1
    ys -= height + gap
    ene_axis = fig.add_axes([xs, ys, width1, height])
    pene_e, = ene_axis.plot(tptl[:tmax_e], kee[:tmax_e],
            linewidth=2, color='r')
    ene_axis.set_ylabel(r'$E_k/(10^4m_ec^2)$', fontdict=font,
            fontsize=20)
    ene_axis.set_xlabel(r'$t\omega_{pe}$', fontdict=font, fontsize=20)
    ene_axis.tick_params(labelsize=16)
    # for tl in ene_axis.get_yticklabels():
    #     tl.set_color('r')

    # ax1 = ene_axis.twinx()
    pene_i, = ene_axis.plot(tptl[:tmax_i], kei[:tmax_i],
            linewidth=2, color='b')
    # ene_axis.set_ylabel(r'$(\gamma_i - 1)/10^2$', fontdict=font, fontsize=20, color='b')
    ene_axis.tick_params(labelsize=16)
    # for tl in ene_axis.get_yticklabels():
    #     tl.set_color('b')
    pene_e_dot, = ene_axis.plot(tptl[ct_track], kee[ct_track], marker='x',
            markersize=10, linestyle='None', color='r')
    pene_i_dot, = ene_axis.plot(tptl[ct_track], kei[ct_track], marker='x',
            markersize=10, linestyle='None', color='b')

    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    fname = '../img/traj_fermi_vx.eps'
    fig.savefig(fname)

    plt.show()
  
if __name__ == "__main__":
    pic_info = pic_information.get_pic_info('../../')
    # filepath = '/net/scratch1/guofan/share/ultra-sigma/'
    # filepath += 'sigma1e4-mime100-4000-track/pic_analysis/vpic-sorter/data/'
    # species = 'i'
    # if species == 'i':
    #     fname = filepath + 'ions.h5p'
    # else:
    #     fname = filepath + 'electrons.h5p'
    # file = h5py.File(fname, 'r')
    # plot_ptl_traj(file, species)
    # file.close()
    plot_ptl_traj_direct()
    # plot_ptl_traj_fermi()
