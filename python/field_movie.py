"""
Procedures to make movies of the fields.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.animation as animation
import numpy as np
import math
import os.path
import struct
import collections
import pic_information
import contour_plots
from contour_plots import read_2d_fields, plot_2d_contour

mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

font = {'family' : 'serif',
        #'color'  : 'darkred',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 24,
        }

pic_info = pic_information.get_pic_info('..')

width = 0.8
height = 0.78
xs = 0.10
ys = 0.9 - height
fig = plt.figure(figsize=[8,4])
ax1 = fig.add_axes([xs, ys, width, height])
filename = '../data/jy.gda'

def read_and_plot(ct, filename):
    """
    Read 2D field and do contour plot.

    Args:
        ct: current time frame.
        filename: the file name including its path.
    """
    kwargs = {"current_time":ct, "xl":0, "xr":200, "zb":-50, "zt":50}
    x, z, num_rho = read_2d_fields(pic_info, filename, **kwargs) 
    x, z, Ay = read_2d_fields(pic_info, "../data/Ay.gda", **kwargs) 

    nx, = x.shape
    nz, = z.shape
    xmin = np.min(x)
    xmax = np.max(x)
    zmin = np.min(z)
    zmax = np.max(z)

    kwargs_plot = {"xstep":2, "zstep":2, "is_log":True, "vmin":-1, "vmax":1}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    data = num_rho[0:nz:zstep, 0:nx:xstep]
    print "Maximum and minimum of the data: ", np.max(data), np.min(data)

    im1 = ax1.imshow(data, cmap=plt.cm.seismic,
                   extent=[xmin, xmax, zmin, zmax],
                   aspect='auto', origin='lower',
                   vmin=kwargs_plot["vmin"], vmax=kwargs_plot["vmax"],
                   interpolation='bicubic')
                   # norm=LogNorm(vmin=kwargs_plot["vmin"], 
                   #     vmax=kwargs_plot["vmax"]))
    ax1.contour(x[0:nx:xstep], z[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep], 
                colors='black', linewidths=0.5)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=32)
    ax1.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=32)
    ax1.tick_params(labelsize=32)
    
    t_wci = current_time * pic_info.dt_fields
    title = r'$t = ' + "{:10.1f}".format(t_wci) + '/\Omega_{ci}$'
    ax1.set_title(title, fontdict=font, fontsize=32)

    return im1

current_time = -1

def field_movie(*args):
    global current_time, cbar1
    ax1.cla()
    current_time += 1
    #im1 = read_and_plot(current_time, filename)
    # Initialize the plot
    im1 = read_and_plot(current_time, filename)
    
    if (current_time == 0):
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="2%", pad=0.05)
        cbar = fig.colorbar(im1, cax=cax)
        cbar.ax.set_ylabel(r'$j_y$', fontdict=font, fontsize=32)
        cbar.ax.tick_params(labelsize=32)
    return im1,

ani = animation.FuncAnimation(fig, field_movie, frames=40, blit=True)
mywriter = animation.FFMpegWriter()
ani.save('jy.mp4', fps=20, writer=mywriter, bitrate=-1, dpi=200,
         codec="libx264", extra_args=['-pix_fmt', 'yuv420p'])
#plt.show()

plt.close()
