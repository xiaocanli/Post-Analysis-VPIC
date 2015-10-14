"""
Analysis procedures for 2D contour plots.
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LogNorm
from matplotlib import rc
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy.ndimage.filters import generic_filter as gf
from scipy import signal
from scipy.fftpack import fft2, ifft2, fftshift
import math
import os.path
import struct
import collections
import pic_information
import color_maps as cm
import colormap.colormaps as cmaps

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

font = {'family' : 'serif',
        #'color'  : 'darkred',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 24,
        }

def read_2d_fields(pic_info, fname, current_time, xl, xr, zb, zt):
    """Read 2D fields data from file.
    
    Args:
        pic_info: namedtuple for the PIC simulation information.
        fname: the filename.
        current_time: current time frame.
        xl, xr: left and right x position in di (ion skin length).
        zb, zt: top and bottom z position in di.
    """
    print 'Reading data from ', fname
    print 'The spatial range (di): ', \
            'x_left = ', xl, 'x_right = ', xr, \
            'z_bottom = ', zb, 'z_top = ', zt
    nx = pic_info.nx
    nz = pic_info.nz
    x_di = pic_info.x_di
    z_di = pic_info.z_di
    dx_di = pic_info.dx_di
    dz_di = pic_info.dz_di
    xmin = np.min(x_di)
    xmax = np.max(x_di)
    zmin = np.min(z_di)
    zmax = np.max(z_di)
    if (xl <= xmin):
        xl_index = 0
    else:
        xl_index = int(math.floor((xl-xmin) / dx_di))
    if (xr >= xmax):
        xr_index = nx - 1
    else:
        xr_index = int(math.ceil((xr-xmin) / dx_di))
    if (zb <= zmin):
        zb_index = 0
    else:
        zb_index = int(math.floor((zb-zmin) / dz_di))
    if (zt >= zmax):
        zt_index = nz - 1
    else:
        zt_index = int(math.ceil((zt-zmin) / dz_di))
    nx1 = xr_index-xl_index+1
    nz1 = zt_index-zb_index+1
    fp = np.zeros((nz1, nx1))
    offset = nx*nz*current_time*4 + zb_index*nx*4 + xl_index*4
    for k in range(nz1):
        fp[k, :] = np.memmap(fname, dtype='float32', mode='r', 
                offset=offset, shape=(nx1), order='F')
        offset += nx * 4
    return (x_di[xl_index:xr_index+1], z_di[zb_index:zt_index+1], fp)

def plot_2d_contour(x, z, field_data, ax, fig, **kwargs):
    """Plot contour of 2D fields.

    Args:
        x, z: the x and z coordinates for the field data.
        field_data: the 2D field data set.
        ax: axes object.
        fig: figure object.
    Returns:
        p1: plot object.
        cbar: color bar object.
    """
    xmin = np.min(x)
    xmax = np.max(x)
    zmin = np.min(z)
    zmax = np.max(z)
    nx, = x.shape
    nz, = z.shape
    if (kwargs and "xstep" in kwargs and "zstep" in kwargs):
        data = field_data[0:nz:kwargs["zstep"], 0:nx:kwargs["xstep"]]
    else:
        data = field_data
    print "Maximum and minimum of the data: ", np.max(data), np.min(data)
    if (kwargs and "vmin" in kwargs and "vmax" in kwargs):
        p1 = ax.imshow(data, cmap=plt.cm.jet,
                extent=[xmin, xmax, zmin, zmax],
                aspect='auto', origin='lower',
                vmin=kwargs["vmin"], vmax=kwargs["vmax"],
                interpolation='bicubic')
    else:
        p1 = ax.imshow(data, cmap=plt.cm.jet,
                extent=[xmin, xmax, zmin, zmax],
                aspect='auto',
                origin='lower',
                interpolation='spline16')
    # Log scale plot
    if (kwargs and "is_log" in kwargs and kwargs["is_log"] == True):
        p1.norm=LogNorm(vmin=kwargs["vmin"], vmax=kwargs["vmax"])
    ax.tick_params(labelsize=16)
    ax.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    #ax.set_title(r'$\beta_e$', fontsize=24)

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    cbar = fig.colorbar(p1, cax=cax)
    cbar.ax.tick_params(labelsize=16)
    return (p1, cbar)

def plot_jy(pic_info, species, current_time):
    """Plot out-of-plane current density.

    Args:
        pic_info: namedtuple for the PIC simulation information.
        species: 'e' for electrons, 'i' for ions.
        current_time: current time frame.
    """
    print(current_time)
    kwargs = {"current_time":current_time, "xl":0, "xr":200, "zb":-20, "zt":20}
    x, z, jy = read_2d_fields(pic_info, "../../data/jy.gda", **kwargs) 
    x, z, Ay = read_2d_fields(pic_info, "../../data/Ay.gda", **kwargs) 
    nx, = x.shape
    nz, = z.shape
    width = 0.75
    height = 0.7
    xs = 0.12
    ys = 0.9 - height
    fig = plt.figure(figsize=[10,4])
    ax1 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep":1, "zstep":1, "vmin":-0.5, "vmax":0.5}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, jy, ax1, fig, **kwargs_plot)
    p1.set_cmap(plt.cm.get_cmap('seismic'))
    # p1.set_cmap(cmaps.inferno)
    ax1.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep], 
            colors='black', linewidths=0.5)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax1.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
    ax1.tick_params(labelsize=24)
    cbar1.ax.set_ylabel(r'$j_y$', fontdict=font, fontsize=24)
    cbar1.set_ticks(np.arange(-0.4, 0.5, 0.2))
    cbar1.ax.tick_params(labelsize=24)
    
    t_wci = current_time*pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(t_wci) + '/\Omega_{ci}$'
    ax1.set_title(title, fontdict=font, fontsize=24)

    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    if not os.path.isdir('../img/img_jy2/'):
        os.makedirs('../img/img_jy2/')
    fname = '../img/img_jy2/jy_' + species + '_' + \
            str(current_time).zfill(3) + '.jpg'
    fig.savefig(fname, dpi=200)

    plt.show()
    # plt.close()

def plot_number_density(pic_info, species, current_time):
    """Plot plasma beta and number density.

    Args:
        pic_info: namedtuple for the PIC simulation information.
        species: 'e' for electrons, 'i' for ions.
        current_time: current time frame.
    """
    kwargs = {"current_time":current_time, "xl":0, "xr":200, "zb":-50, "zt":50}
    x, z, num_rho = read_2d_fields(pic_info, "../data/ne.gda", **kwargs) 
    x, z, Ay = read_2d_fields(pic_info, "../data/Ay.gda", **kwargs) 
    nx, = x.shape
    nz, = z.shape
    width = 0.8
    height = 0.78
    xs = 0.10
    ys = 0.9 - height
    fig = plt.figure(figsize=[16,8])
    ax1 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep":2, "zstep":2, "is_log":True, "vmin":0.1, "vmax":10}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, num_rho, ax1, fig, **kwargs_plot)
    p1.set_cmap(plt.cm.seismic)
    ax1.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep], 
            colors='black', linewidths=0.5)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=32)
    ax1.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=32)
    ax1.tick_params(labelsize=32)
    cbar1.ax.set_ylabel(r'$n_e$', fontdict=font, fontsize=32)
    cbar1.ax.tick_params(labelsize=32)
    
    t_wci = current_time*pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(t_wci) + '/\Omega_{ci}$'
    ax1.set_title(title, fontdict=font, fontsize=32)

    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    if not os.path.isdir('../img/img_num_rho/'):
        os.makedirs('../img/img_num_rho/')
    fname = '../img/img_num_rho/num_rho_' + species + '_' + \
            str(current_time).zfill(3) + '.jpg'
    fig.savefig(fname)

    plt.show()
    #plt.close()

def plot_anistropy(pic_info, species):
    """Plot plasma beta and number density.

    Args:
        pic_info: namedtuple for the PIC simulation information.
        species: 'e' for electrons, 'i' for ions.
    """
    kwargs = {"current_time":37, "xl":0, "xr":200, "zb":-20, "zt":20}
    fname = '../../data/p' + species + '-xx.gda'
    x, z, pxx = read_2d_fields(pic_info, fname, **kwargs) 
    fname = '../../data/p' + species + '-yy.gda'
    x, z, pyy = read_2d_fields(pic_info, fname, **kwargs) 
    fname = '../../data/p' + species + '-zz.gda'
    x, z, pzz = read_2d_fields(pic_info, fname, **kwargs) 
    fname = '../../data/p' + species + '-xy.gda'
    x, z, pxy = read_2d_fields(pic_info, fname, **kwargs) 
    fname = '../../data/p' + species + '-xz.gda'
    x, z, pxz = read_2d_fields(pic_info, fname, **kwargs) 
    fname = '../../data/p' + species + '-yz.gda'
    x, z, pyz = read_2d_fields(pic_info, fname, **kwargs) 
    x, z, bx = read_2d_fields(pic_info, "../../data/bx.gda", **kwargs) 
    x, z, by = read_2d_fields(pic_info, "../../data/by.gda", **kwargs) 
    x, z, bz = read_2d_fields(pic_info, "../../data/bz.gda", **kwargs) 
    x, z, absB = read_2d_fields(pic_info, "../../data/absB.gda", **kwargs) 
    x, z, Ay = read_2d_fields(pic_info, "../../data/Ay.gda", **kwargs) 
    ppara = pxx*bx*bx + pyy*by*by + pzz*bz*bz + \
            pxy*bx*by*2.0 + pxz*bx*bz*2.0 + pyz*by*bz*2.0
    ppara /= absB * absB
    pperp = 0.5 * (pxx+pyy+pzz-ppara)
    nx, = x.shape
    nz, = z.shape
    beta_e = (pxx+pyy+pzz)*2/(3*absB**2)
    width = 0.8
    height = 0.8
    xs = 0.12
    ys = 0.9 - height
    fig = plt.figure(figsize=[7,2])
    ax1 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep":2, "zstep":2, "is_log":True, "vmin":0.1, "vmax":10}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, ppara/pperp, ax1, fig, **kwargs_plot)
    p1.set_cmap(plt.cm.seismic)
    ax1.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep], 
            colors='black', linewidths=0.5)
    ax1.tick_params(axis='x', labelbottom='off')

    plt.show()

def plot_beta_rho(pic_info):
    """Plot plasma beta and number density.

    Args:
        pic_info: namedtuple for the PIC simulation information.
    """
    kwargs = {"current_time":20, "xl":0, "xr":200, "zb":-10, "zt":10}
    x, z, pexx = read_2d_fields(pic_info, "../../data/pe-xx.gda", **kwargs) 
    x, z, peyy = read_2d_fields(pic_info, "../../data/pe-yy.gda", **kwargs) 
    x, z, pezz = read_2d_fields(pic_info, "../../data/pe-zz.gda", **kwargs) 
    x, z, absB = read_2d_fields(pic_info, "../../data/absB.gda", **kwargs) 
    x, z, eEB05 = read_2d_fields(pic_info, "../../data/eEB05.gda", **kwargs) 
    x, z, Ay = read_2d_fields(pic_info, "../../data/Ay.gda", **kwargs) 
    nx, = x.shape
    nz, = z.shape
    beta_e = (pexx+peyy+pezz)*2/(3*absB**2)
    width = 0.8
    height = 0.3
    xs = 0.12
    ys = 0.9 - height
    fig = plt.figure(figsize=[7,4])
    ax1 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep":2, "zstep":2, "is_log":True, "vmin":0.01, "vmax":10}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, beta_e, ax1, fig, **kwargs_plot)
    p1.set_cmap(cmaps.plasma)
    ax1.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep], 
            colors='white', linewidths=0.5)
    ax1.tick_params(axis='x', labelbottom='off')
    ax1.set_title(r'$\beta_e$', fontsize=24)

    ys -= height + 0.15
    ax2 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep":2, "zstep":2}
    p2, cbar2 = plot_2d_contour(x, z, eEB05, ax2, fig, **kwargs_plot)
    # p2.set_cmap(cmaps.magma)
    # p2.set_cmap(cmaps.inferno)
    p2.set_cmap(cmaps.plasma)
    # p2.set_cmap(cmaps.viridis)
    ax2.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep], 
            colors='white', linewidths=0.5)
    ax2.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    ax2.set_title(r'$n_{acc}/n_e$', fontsize=24)
    cbar2.set_ticks(np.arange(0.2, 1.0, 0.2))
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    fig.savefig('../img/beta_e_ne_2.eps')
    plt.show()

def plot_jdote_2d(pic_info):
    """Plot jdotE due to drift current.
    Args:
        pic_info: namedtuple for the PIC simulation information.
    """
    kwargs = {"current_time":40, "xl":0, "xr":200, "zb":-10, "zt":10}
    x, z, jcpara_dote = read_2d_fields(pic_info, 
            "../data1/jcpara_dote00_e.gda", **kwargs) 
    x, z, jgrad_dote = read_2d_fields(pic_info, 
            "../data1/jgrad_dote00_e.gda", **kwargs) 
    x, z, agyp = read_2d_fields(pic_info, 
            "../data1/agyrotropy00_e.gda", **kwargs) 
    x, z, Ay = read_2d_fields(pic_info, "../data/Ay.gda", **kwargs) 
    jdote_norm = 1E3
    jcpara_dote *= jdote_norm
    jgrad_dote *= jdote_norm
    nx, = x.shape
    nz, = z.shape
    width = 0.78
    height = 0.25
    xs = 0.14
    xe = 0.94 - xs
    ys = 0.96 - height
    fig = plt.figure(figsize=(7,5))
    ax1 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep":2, "zstep":2, "vmin":-1, "vmax":1}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, jcpara_dote, ax1, fig, **kwargs_plot)
    p1.set_cmap(cmaps.viridis)
    ax1.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep], 
            colors='black', linewidths=0.5)
    cbar1.set_ticks(np.arange(-0.8, 1.0, 0.4))
    ax1.tick_params(axis='x', labelbottom='off')
    ax1.text(5, 5.2, r'$\mathbf{j}_c\cdot\mathbf{E}$', color='blue', fontsize=24, 
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0))
    
    ys -= height + 0.035
    ax2 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep":2, "zstep":2, "vmin":-1, "vmax":1}
    p2, cbar2 = plot_2d_contour(x, z, jcpara_dote, ax2, fig, **kwargs_plot)
    p2.set_cmap(cmaps.viridis)
    ax2.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep], 
            colors='black', linewidths=0.5)
    cbar2.set_ticks(np.arange(-0.8, 1.0, 0.4))
    ax2.tick_params(axis='x', labelbottom='off')
    ax2.text(5, 5, r'$\mathbf{j}_g\cdot\mathbf{E}$', color='green', fontsize=24, 
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0))

    ys -= height + 0.035
    ax3 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep":2, "zstep":2, "vmin":0, "vmax":1.5}
    p3, cbar3 = plot_2d_contour(x, z, agyp, ax3, fig, **kwargs_plot)
    p3.set_cmap(cmaps.viridis)
    ax3.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep], 
            colors='black', linewidths=0.5)
    cbar3.set_ticks(np.arange(0, 1.6, 0.4))
    ax3.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    ax3.text(5, 5, r'$A_e$', color='black', fontsize=24, 
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0))

    plt.show()

def plot_phi_parallel(pic_info):
    """Plot parallel potential.
    Args:
        pic_info: namedtuple for the PIC simulation information.
    """
    kwargs = {"current_time":37, "xl":0, "xr":200, "zb":-50, "zt":50}
    x, z, phi_para = read_2d_fields(pic_info, 
            "../../data1/phi_para.gda", **kwargs) 
    #x, z, Ay = read_2d_fields(pic_info, "../../data/Ay.gda", **kwargs) 

    #phi_para_new = phi_para
    phi_para_new = signal.medfilt2d(phi_para, kernel_size=(9,9))
    #phi_para_new = signal.wiener(phi_para, mysize=5)
    #ng = 9
    #kernel = np.ones((ng,ng)) / float(ng*ng)
    #phi_para_new = signal.convolve2d(phi_para, kernel)

    # fft_transform = np.fft.fft2(phi_para)
    # power_spectrum = np.abs(fft_transform)**2
    # scaled_power_spectrum = np.log10(power_spectrum)
    # scaled_ps0 = scaled_power_spectrum - np.max(scaled_power_spectrum)
    # # print np.min(scaled_ps0), np.max(scaled_ps0)

    # freqx = np.fft.fftfreq(x.size)
    # freqz = np.fft.fftfreq(z.size)
    # xv, zv = np.meshgrid(freqx, freqz)
    # #print np.max(freqx), np.min(freqx)
    # #m = np.ma.masked_where(scaled_ps0 > -8.8, scaled_ps0)
    # #new_fft_spectrum = np.ma.masked_array(fft_transform, m.mask)
    # new_fft_spectrum = fft_transform.copy()
    # #new_fft_spectrum[scaled_ps0 < -1.3] = 0.0
    # print np.max(xv), np.min(xv)
    # new_fft_spectrum[xv > 0.02] = 0.0
    # new_fft_spectrum[xv < -0.02] = 0.0
    # new_fft_spectrum[zv > 0.02] = 0.0
    # new_fft_spectrum[zv < -0.02] = 0.0

    # data = np.fft.ifft2(new_fft_spectrum)

    nx, = x.shape
    nz, = z.shape
    width = 0.85
    height = 0.85
    xs = 0.1
    xe = 0.95 - xs
    ys = 0.96 - height
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep":1, "zstep":1, "vmin":-0.05, "vmax":0.05}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    im1, cbar1 = plot_2d_contour(x, z, phi_para_new, ax1, fig, **kwargs_plot)
    #im1 = plt.imshow(data.real, vmin=-0.1, vmax=0.1)
    im1.set_cmap(plt.cm.seismic)
    
    #ax1.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep], 
    #        colors='black', linewidths=0.5, 
    #        levels=np.arange(np.min(Ay), np.max(Ay), 15))
    cbar1.set_ticks(np.arange(-0.2, 0.2, 0.05))
    #ax1.tick_params(axis='x', labelbottom='off')
    ax1.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    ax1.set_ylabel(r'$y/d_i$', fontdict=font, fontsize=20)
    
    #plt.close()
    plt.show()

def plot_Ey(pic_info, species, current_time):
    """Plot out-of-plane current density.

    Args:
        pic_info: namedtuple for the PIC simulation information.
        species: 'e' for electrons, 'i' for ions.
        current_time: current time frame.
    """
    print(current_time)
    kwargs = {"current_time":current_time, "xl":0, "xr":200, "zb":-50, "zt":50}
    x, z, ey = read_2d_fields(pic_info, "../../data/ey.gda", **kwargs) 
    x, z, Ay = read_2d_fields(pic_info, "../../data/Ay.gda", **kwargs) 
    nx, = x.shape
    nz, = z.shape
    width = 0.75
    height = 0.7
    xs = 0.12
    ys = 0.9 - height
    fig = plt.figure(figsize=[10,4])
    ax1 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep":1, "zstep":1, "vmin":-0.1, "vmax":0.1}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, ey, ax1, fig, **kwargs_plot)
    p1.set_cmap(cmaps.plasma)
    ax1.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep], 
            colors='black', linewidths=0.5)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax1.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
    ax1.tick_params(labelsize=24)
    cbar1.ax.set_ylabel(r'$E_y$', fontdict=font, fontsize=24)
    cbar1.set_ticks(np.arange(-0.08, 0.1, 0.04))
    cbar1.ax.tick_params(labelsize=24)
    
    t_wci = current_time*pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(t_wci) + '/\Omega_{ci}$'
    ax1.set_title(title, fontdict=font, fontsize=24)

    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    if not os.path.isdir('../img/img_ey/'):
        os.makedirs('../img/img_ey/')
    fname = '../img/img_ey/ey_' + str(current_time).zfill(3) + '.jpg'
    fig.savefig(fname, dpi=200)

    plt.show()
    # plt.close()

def plot_jy_Ey(pic_info, species, current_time):
    """Plot out-of-plane current density.

    Args:
        pic_info: namedtuple for the PIC simulation information.
        species: 'e' for electrons, 'i' for ions.
        current_time: current time frame.
    """
    print(current_time)
    kwargs = {"current_time":current_time, "xl":0, "xr":200, "zb":-50, "zt":50}
    x, z, ey = read_2d_fields(pic_info, "../data/ey.gda", **kwargs) 
    x, z, jy = read_2d_fields(pic_info, "../data/jy.gda", **kwargs) 
    x, z, Ay = read_2d_fields(pic_info, "../data/Ay.gda", **kwargs) 
    nx, = x.shape
    nz, = z.shape
    width = 0.75
    height = 0.7
    xs = 0.12
    ys = 0.9 - height
    fig = plt.figure(figsize=[10,4])
    ax1 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep":1, "zstep":1, "vmin":-0.01, "vmax":0.01}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, jy*ey, ax1, fig, **kwargs_plot)
    p1.set_cmap(plt.cm.seismic)
    ax1.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep], 
            colors='black', linewidths=0.5)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax1.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
    ax1.tick_params(labelsize=24)
    cbar1.ax.set_ylabel(r'$j_yE_y$', fontdict=font, fontsize=24)
    cbar1.set_ticks(np.arange(-0.01, 0.015, 0.01))
    cbar1.ax.tick_params(labelsize=24)
    
    t_wci = current_time*pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(t_wci) + '/\Omega_{ci}$'
    ax1.set_title(title, fontdict=font, fontsize=24)

    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    if not os.path.isdir('../img/img_jy_ey/'):
        os.makedirs('../img/img_jy_ey/')
    fname = '../img/img_jy_ey/jy_ey' + '_' + str(current_time).zfill(3) + '.jpg'
    fig.savefig(fname, dpi=200)

    #plt.show()
    plt.close()

def plot_jpolar_dote(pic_info, species, current_time):
    """Plot out-of-plane current density.

    Args:
        pic_info: namedtuple for the PIC simulation information.
        species: 'e' for electrons, 'i' for ions.
        current_time: current time frame.
    """
    print(current_time)
    kwargs = {"current_time":current_time, "xl":0, "xr":400, "zb":-100, "zt":100}
    x, z, jpolar_dote = read_2d_fields(pic_info, "../../data1/jpolar_dote00_e.gda", **kwargs) 
    # x, z, Ay = read_2d_fields(pic_info, "../../data/Ay.gda", **kwargs) 
    nx, = x.shape
    nz, = z.shape
    width = 0.75
    height = 0.7
    xs = 0.12
    ys = 0.9 - height
    fig = plt.figure(figsize=[10,4])
    ax1 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep":1, "zstep":1, "vmin":-1, "vmax":1}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, jpolar_dote, ax1, fig, **kwargs_plot)
    p1.set_cmap(plt.cm.seismic)
    # ax1.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep], 
    #         colors='black', linewidths=0.5)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax1.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
    ax1.tick_params(labelsize=24)
    # cbar1.ax.set_ylabel(r'$j_yE_y$', fontdict=font, fontsize=24)
    # cbar1.set_ticks(np.arange(-0.01, 0.015, 0.01))
    cbar1.ax.tick_params(labelsize=24)
    plt.show()

def plot_epara(pic_info, species, current_time):
    kwargs = {"current_time":current_time, "xl":0, "xr":200, "zb":-20, "zt":20}
    x, z, bx = read_2d_fields(pic_info, "../../data/bx.gda", **kwargs) 
    x, z, by = read_2d_fields(pic_info, "../../data/by.gda", **kwargs) 
    x, z, bz = read_2d_fields(pic_info, "../../data/bz.gda", **kwargs) 
    x, z, absB = read_2d_fields(pic_info, "../../data/absB.gda", **kwargs) 
    x, z, ex = read_2d_fields(pic_info, "../../data/ex.gda", **kwargs) 
    x, z, ey = read_2d_fields(pic_info, "../../data/ey.gda", **kwargs) 
    x, z, ez = read_2d_fields(pic_info, "../../data/ez.gda", **kwargs) 
    x, z, Ay = read_2d_fields(pic_info, "../../data/Ay.gda", **kwargs) 

    absE = np.sqrt(ex*ex + ey*ey + ez*ez)
    epara = (ex*bx + ey*by + ez*bz) / absB
    eperp = np.sqrt(absE*absE - epara*epara)
    ng = 3
    kernel = np.ones((ng,ng)) / float(ng*ng)
    epara = signal.convolve2d(epara, kernel)
    eperp = signal.convolve2d(eperp, kernel)

    nx, = x.shape
    nz, = z.shape
    width = 0.75
    height = 0.37
    xs = 0.12
    ys = 0.92 - height
    gap = 0.05

    fig = plt.figure(figsize=[10,6])
    ax1 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep":1, "zstep":1, "vmin":0, "vmax":0.1}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, eperp, ax1, fig, **kwargs_plot)
    p1.set_cmap(cmaps.inferno)
    ax1.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep], 
            colors='black', linewidths=0.5)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    # ax1.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
    ax1.tick_params(axis='x', labelbottom='off')
    ax1.tick_params(labelsize=20)
    cbar1.ax.set_ylabel(r'$E_\perp$', fontdict=font, fontsize=24)
    # cbar1.set_ticks(np.arange(-0.1, 0.15, 0.1))
    cbar1.ax.tick_params(labelsize=20)

    ys -= height + 0.05
    ax2 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep":1, "zstep":1, "vmin":-0.05, "vmax":0.05}
    p2, cbar2 = plot_2d_contour(x, z, epara, ax2, fig, **kwargs_plot)
    p2.set_cmap(plt.cm.seismic)
    # p2.set_cmap(cmaps.plasma)
    ax2.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep], 
            colors='black', linewidths=0.5)
    ax2.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax2.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
    ax2.tick_params(labelsize=20)
    cbar2.ax.set_ylabel(r'$E_\parallel$', fontdict=font, fontsize=24)
    # cbar1.set_ticks(np.arange(-0.1, 0.15, 0.1))
    cbar2.ax.tick_params(labelsize=20)
    
    t_wci = current_time*pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(t_wci) + '/\Omega_{ci}$'
    ax1.set_title(title, fontdict=font, fontsize=24)

    # if not os.path.isdir('../img/'):
    #     os.makedirs('../img/')
    # if not os.path.isdir('../img/img_jy_ey/'):
    #     os.makedirs('../img/img_jy_ey/')
    # fname = '../img/img_jy_ey/jy_ey' + '_' + str(current_time).zfill(3) + '.jpg'
    # fig.savefig(fname, dpi=200)

    plt.show()
    # plt.close()


def plot_diff_fields(pic_info, species, current_time):
    """Plot the differential of the fields.

    Args:
        pic_info: namedtuple for the PIC simulation information.
        species: 'e' for electrons, 'i' for ions.
        current_time: current time frame.
    """
    print(current_time)
    kwargs = {"current_time":current_time, "xl":0, "xr":200, "zb":-20, "zt":20}
    x, z, data = read_2d_fields(pic_info, "../../data/absB.gda", **kwargs) 
    x, z, Ay = read_2d_fields(pic_info, "../../data/Ay.gda", **kwargs) 
    nx, = x.shape
    nz, = z.shape
    width = 0.75
    height = 0.7
    xs = 0.12
    ys = 0.9 - height
    fig = plt.figure(figsize=[10,4])
    ax1 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep":1, "zstep":1, "vmin":-0.01, "vmax":0.01}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    data_new = np.zeros((nz, nx))
    # data_new[:, 0:nx-1] = data[:, 1:nx] - data[:, 0:nx-1]
    data_new[0:nz-1, :] = data[1:nz, :] - data[0:nz-1, :]
    ng = 5
    kernel = np.ones((ng,ng)) / float(ng*ng)
    data_new = signal.convolve2d(data_new, kernel)
    p1, cbar1 = plot_2d_contour(x, z, data_new, ax1, fig, **kwargs_plot)
    p1.set_cmap(plt.cm.get_cmap('seismic'))
    # p1.set_cmap(cmaps.inferno)
    ax1.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep], 
            colors='black', linewidths=0.5)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax1.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
    ax1.tick_params(labelsize=24)
    cbar1.ax.set_ylabel(r'$j_y$', fontdict=font, fontsize=24)
    cbar1.set_ticks(np.arange(-0.4, 0.5, 0.2))
    cbar1.ax.tick_params(labelsize=24)
    
    t_wci = current_time*pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(t_wci) + '/\Omega_{ci}$'
    ax1.set_title(title, fontdict=font, fontsize=24)

    plt.show()


def plot_jpara_perp(pic_info, species, current_time):
    """Plot the energy conversion from parallel and perpendicular direction.

    Args:
        pic_info: namedtuple for the PIC simulation information.
        species: 'e' for electrons, 'i' for ions.
        current_time: current time frame.
    """
    print(current_time)
    kwargs = {"current_time":current_time, "xl":0, "xr":200, "zb":-20, "zt":20}
    x, z, data = read_2d_fields(pic_info, "../../data1/jqnvperp_dote00_e.gda", **kwargs) 
    x, z, Ay = read_2d_fields(pic_info, "../../data/Ay.gda", **kwargs) 
    nx, = x.shape
    nz, = z.shape
    width = 0.75
    height = 0.7
    xs = 0.12
    ys = 0.9 - height
    fig = plt.figure(figsize=[10,4])
    ax1 = fig.add_axes([xs, ys, width, height])
    dmax = -0.0005
    kwargs_plot = {"xstep":1, "zstep":1, "vmin":-dmax, "vmax":dmax}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    data_new = np.zeros((nz, nx))
    ng = 5
    kernel = np.ones((ng,ng)) / float(ng*ng)
    data_new = signal.convolve2d(data, kernel)
    p1, cbar1 = plot_2d_contour(x, z, data_new, ax1, fig, **kwargs_plot)
    p1.set_cmap(plt.cm.get_cmap('seismic'))
    # p1.set_cmap(cmaps.inferno)
    ax1.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep], 
            colors='black', linewidths=0.5)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax1.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
    ax1.tick_params(labelsize=24)
    # cbar1.ax.set_ylabel(r'$j_y$', fontdict=font, fontsize=24)
    # cbar1.set_ticks(np.arange(-0.4, 0.5, 0.2))
    cbar1.ax.tick_params(labelsize=24)
    
    t_wci = current_time*pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(t_wci) + '/\Omega_{ci}$'
    ax1.set_title(title, fontdict=font, fontsize=24)

    plt.show()


def plot_ux(pic_info, species, current_time):
    """Plot the in-plane bulk velocity field.

    Args:
        pic_info: namedtuple for the PIC simulation information.
        species: 'e' for electrons, 'i' for ions.
        current_time: current time frame.
    """
    print(current_time)
    kwargs = {"current_time":current_time, "xl":0, "xr":200, "zb":-20, "zt":20}
    fname = "../../data/vex.gda"
    if not os.path.isfile(fname):
        fname = "../../data/uex.gda"
    x, z, uex = read_2d_fields(pic_info, fname, **kwargs) 
    x, z, ne = read_2d_fields(pic_info, "../../data/ne.gda", **kwargs) 
    fname = "../../data/vix.gda"
    if not os.path.isfile(fname):
        fname = "../../data/uix.gda"
    x, z, uix = read_2d_fields(pic_info, fname, **kwargs) 
    x, z, ni = read_2d_fields(pic_info, "../../data/ni.gda", **kwargs) 
    x, z, Ay = read_2d_fields(pic_info, "../../data/Ay.gda", **kwargs) 
    wpe_wce = pic_info.dtwce / pic_info.dtwpe
    va = wpe_wce / math.sqrt(pic_info.mime)  # Alfven speed of inflow region
    ux = (uex*ne + uix*ni*pic_info.mime) / (ne + ni*pic_info.mime)
    ux /= va
    nx, = x.shape
    nz, = z.shape
    width = 0.75
    height = 0.4
    xs = 0.15
    ys = 0.92 - height
    fig = plt.figure(figsize=[7,5])
    ax1 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep":1, "zstep":1, "vmin":-1.0, "vmax":1.0}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = plot_2d_contour(x, z, ux, ax1, fig, **kwargs_plot)
    p1.set_cmap(plt.cm.get_cmap('seismic'))
    ax1.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep], 
            colors='black', linewidths=0.5)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=24)
    ax1.tick_params(axis='x', labelbottom='off')
    ax1.tick_params(labelsize=20)
    cbar1.set_ticks(np.arange(-0.8, 1.0, 0.4))
    cbar1.ax.tick_params(labelsize=20)
    ax1.text(0.02, 0.8, r'$u_x/V_A$', color='k', fontsize=24, 
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0),
            horizontalalignment='left', verticalalignment='center',
            transform = ax1.transAxes)
    
    t_wci = current_time*pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(t_wci) + '/\Omega_{ci}$'
    ax1.set_title(title, fontdict=font, fontsize=24)

    gap = 0.06
    ys0 = 0.15
    height0 = ys - gap - ys0
    w1, h1 = fig.get_size_inches()
    width0 = width * 0.98 - 0.05 / w1
    ax2 = fig.add_axes([xs, ys0, width0, height0])
    ax2.plot(x, ux[nz/2, :], color='k', linewidth=1)
    ax2.plot([np.min(x), np.max(x)], [0, 0], linestyle='--', color='k')
    ax2.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=24)
    ax2.set_ylabel(r'$u_x/V_A$', fontdict=font, fontsize=24)
    ax2.tick_params(labelsize=20)

    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    # fig.savefig(fname, dpi=200)

    plt.show()
    # plt.close()

if __name__ == "__main__":
    pic_info = pic_information.get_pic_info('../../')
    ntp = pic_info.ntp
    # plot_beta_rho(pic_info)
    # plot_jdote_2d(pic_info)
    # plot_anistropy(pic_info, 'e')
    # plot_phi_parallel(pic_info)
    # maps = sorted(m for m in plt.cm.datad if not m.endswith("_r"))
    # nmaps = len(maps) + 1
    # print nmaps
    # for i in range(pic_info.ntf):
    #     # plot_number_density(pic_info, 'e', i)
    #     # plot_jy(pic_info, 'e', i)
    #     plot_Ey(pic_info, 'e', i)
    # plot_number_density(pic_info, 'e', 40)
    # plot_jy(pic_info, 'e', 120)
    plot_ux(pic_info, 'e', 14)
    # plot_diff_fields(pic_info, 'e', 120)
    # plot_jpara_perp(pic_info, 'e', 120)
    # plot_Ey(pic_info, 'e', 40)
    # plot_jy_Ey(pic_info, 'e', 40)
    # for i in range(pic_info.ntf):
    #     plot_Ey(pic_info, 'e', i)

    # for i in range(pic_info.ntf):
    #     plot_jy_Ey(pic_info, 'e', i)
    # plot_jpolar_dote(pic_info, 'e', 30)
    # plot_epara(pic_info, 'e', 40)
