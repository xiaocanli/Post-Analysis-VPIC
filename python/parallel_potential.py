"""
Analysis procedures for particle energy spectrum.
"""
import collections
import math
import os.path
import struct

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib import _cntr as cntr
from matplotlib import rc
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from scipy import array as vector
from scipy.linalg import norm

import contour_plots
import pic_information

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

font = {
    'family': 'serif',
    #'color'  : 'darkred',
    'color': 'black',
    'weight': 'normal',
    'size': 24,
}


def parallel_potential(pic_info):
    """Calculate parallel potential defined by Jan Egedal.

    Args:
        pic_info: namedtuple for the PIC simulation information.
    """
    kwargs = {"current_time": 40, "xl": 0, "xr": 200, "zb": -50, "zt": 50}
    x, z, Ay = contour_plots.read_2d_fields(pic_info, "../data/Ay.gda",
                                            **kwargs)
    x, z, Ex = contour_plots.read_2d_fields(pic_info, "../data/ex.gda",
                                            **kwargs)
    x, z, Ey = contour_plots.read_2d_fields(pic_info, "../data/ey.gda",
                                            **kwargs)
    x, z, Ez = contour_plots.read_2d_fields(pic_info, "../data/ez.gda",
                                            **kwargs)
    x, z, Bx = contour_plots.read_2d_fields(pic_info, "../data/bx.gda",
                                            **kwargs)
    x, z, By = contour_plots.read_2d_fields(pic_info, "../data/by.gda",
                                            **kwargs)
    xarr, zarr, Bz = contour_plots.read_2d_fields(pic_info, "../data/bz.gda",
                                                  **kwargs)
    absB = np.sqrt(Bx**2 + By**2 + Bz**2)
    Epara = (Ex * Bx + Ey * By + Ez * Bz) / absB
    nx, = x.shape
    nz, = z.shape

    phi_parallel = np.zeros((nz, nx))
    dx_di = pic_info.dx_di
    dz_di = pic_info.dz_di
    #deltas = math.sqrt(dx_di**2 + dz_di**2)
    #hds = deltas * 0.5
    hmax = dx_di * 100
    h = hmax / 4.0
    # Cash-Karp parameters
    a = [0.0, 0.2, 0.3, 0.6, 1.0, 0.875]
    b = [[], [0.2], [3.0 / 40.0, 9.0 / 40.0], [0.3, -0.9, 1.2],
         [-11.0 / 54.0, 2.5, -70.0 / 27.0, 35.0 / 27.0], [
             1631.0 / 55296.0, 175.0 / 512.0, 575.0 / 13824.0, 44275.0 /
             110592.0, 253.0 / 4096.0
         ]]
    c = [37.0 / 378.0, 0.0, 250.0 / 621.0, 125.0 / 594.0, 0.0, 512.0 / 1771.0]
    dc = [
        c[0] - 2825.0 / 27648.0, c[1] - 0.0, c[2] - 18575.0 / 48384.0,
        c[3] - 13525.0 / 55296.0, c[4] - 277.00 / 14336.0, c[5] - 0.25
    ]

    def F(t, x, z):
        indices_bl, indices_tr, delta = grid_indices(x, 0, z, nx, 1, nz, dx_di,
                                                     1, dz_di)
        ix1 = indices_bl[0]
        iz1 = indices_bl[2]
        ix2 = indices_tr[0]
        iz2 = indices_tr[2]
        offsetx = delta[0]
        offsetz = delta[2]
        v1 = (1.0 - offsetx) * (1.0 - offsetz)
        v2 = offsetx * (1.0 - offsetz)
        v3 = offsetx * offsetz
        v4 = (1.0 - offsetx) * offsetz
        if (ix1 < nx and ix2 < nx and iz1 < nz and iz2 < nz):
            bx = Bx[iz1, ix1] * v1 + Bx[iz1, ix2] * v2 + Bx[
                iz2, ix2] * v3 + Bx[iz2, ix1] * v4
            by = By[iz1, ix1] * v1 + By[iz1, ix2] * v2 + By[
                iz2, ix2] * v3 + By[iz2, ix1] * v4
            bz = Bz[iz1, ix1] * v1 + Bz[iz1, ix2] * v2 + Bz[
                iz2, ix2] * v3 + Bz[iz2, ix1] * v4
            ex = Ex[iz1, ix1] * v1 + Ex[iz1, ix2] * v2 + Ex[
                iz2, ix2] * v3 + Ex[iz2, ix1] * v4
            ey = Ey[iz1, ix1] * v1 + Ey[iz1, ix2] * v2 + Ey[
                iz2, ix2] * v3 + Ey[iz2, ix1] * v4
            ez = Ez[iz1, ix1] * v1 + Ez[iz1, ix2] * v2 + Ez[
                iz2, ix2] * v3 + Ez[iz2, ix1] * v4
            absB = math.sqrt(bx**2 + bz**2)
            deltax1 = bx / absB
            deltay1 = by * deltax1 / bx
            deltaz1 = bz / absB
        else:
            ex = 0
            ey = 0
            ez = 0
            deltax1 = 0
            deltay1 = 0
            deltaz1 = 0
        return (deltax1, deltay1, deltaz1, ex, ey, ez)

    tol = 1e-5

    for i in range(2900, 3300):
        print i
        x0 = xarr[i] - xarr[0]
        #for k in range(0,nz,8):
        for k in range(950, 1098):
            z0 = zarr[k] - zarr[0]
            y = [x0, z0]
            nstep = 0
            xlist = [x0]
            zlist = [z0]
            t = 0
            x = x0
            z = z0
            while (x >= 0 and x <= (xarr[-1] - xarr[0]) and z >= 0 and
                   z <= (zarr[-1] - zarr[0]) and t < 4E2):
                # Compute k[i] function values.
                kx = [None] * 6
                ky = [None] * 6
                kz = [None] * 6
                kx[0], ky[0], kz[0], ex0, ey0, ez0 = F(t, x, z)
                kx[1], ky[1], kz[1], ex1, ey1, ez1 = F(
                    t + a[1] * h, x + h * (kx[0] * b[1][0]),
                    z + h * (kz[0] * b[1][0]))
                kx[2], ky[2], kz[2], ex2, ey2, ez2 = F(
                    t + a[2] * h, x + h * (kx[0] * b[2][0] + kx[1] * b[2][1]),
                    z + h * (kz[0] * b[2][0] + kz[1] * b[2][1]))
                kx[3], ky[3], kz[3], ex3, ey3, ez3 = F(
                    t + a[3] * h, x + h *
                    (kx[0] * b[3][0] + kx[1] * b[3][1] + kx[2] * b[3][2]),
                    z + h *
                    (kz[0] * b[3][0] + kz[1] * b[3][1] + kz[2] * b[3][2]))
                kx[4], ky[4], kz[4], ex4, ey4, ez4 = F(
                    t + a[4] * h, x + h * (kx[0] * b[4][0] + kx[1] * b[4][1] +
                                           kx[2] * b[4][2] + kx[3] * b[4][3]),
                    z + h * (kz[0] * b[4][0] + kz[1] * b[4][1] + kz[2] *
                             b[4][2] + kz[3] * b[4][3]))
                kx[5], ky[5], kz[5], ex5, ey5, ez5 = F(
                    t + a[5] * h,
                    x + h * (kx[0] * b[5][0] + kx[1] * b[5][1] + kx[2] *
                             b[5][2] + kx[3] * b[5][3] + kx[4] * b[5][4]),
                    z + h * (kz[0] * b[5][0] + kz[1] * b[5][1] + kz[2] *
                             b[5][2] + kz[3] * b[5][3] + kz[4] * b[5][4]))

                # Estimate current error and current maximum error.
                E = norm([
                    h * (kx[0] * dc[0] + kx[1] * dc[1] + kx[2] * dc[2] + kx[3]
                         * dc[3] + kx[4] * dc[4] + kx[5] * dc[5]),
                    h * (kz[0] * dc[0] + kz[1] * dc[1] + kz[2] * dc[2] + kz[3]
                         * dc[3] + kz[4] * dc[4] + kz[5] * dc[5])
                ])
                Emax = tol * max(norm(y), 1.0)

                # Update solution if error is OK.
                if E < Emax:
                    t += h
                    dx = h * (kx[0] * c[0] + kx[1] * c[1] + kx[2] * c[2] +
                              kx[3] * c[3] + kx[4] * c[4] + kx[5] * c[5])
                    dy = h * (ky[0] * c[0] + ky[1] * c[1] + ky[2] * c[2] +
                              ky[3] * c[3] + ky[4] * c[4] + ky[5] * c[5])
                    dz = h * (kz[0] * c[0] + kz[1] * c[1] + kz[2] * c[2] +
                              kz[3] * c[3] + kz[4] * c[4] + kz[5] * c[5])
                    y[0] += dx
                    y[1] += dz
                    x = y[0]
                    z = y[1]
                    xlist.append(x)
                    zlist.append(z)
                    phi_parallel[k, i] += \
                            h*(kx[0]*c[0]*ex0+kx[1]*c[1]*ex1+kx[2]*c[2]*ex2+ \
                            kx[3]*c[3]*ex3+kx[4]*c[4]*ex4+kx[5]*c[5]*ex5) + \
                            h*(ky[0]*c[0]*ey0+ky[1]*c[1]*ey1+ky[2]*c[2]*ey2+ \
                            ky[3]*c[3]*ey3+ky[4]*c[4]*ey4+ky[5]*c[5]*ey5) + \
                            h*(kz[0]*c[0]*ez0+kz[1]*c[1]*ez1+kz[2]*c[2]*ez2+ \
                            kz[3]*c[3]*ez3+kz[4]*c[4]*ez4+kz[5]*c[5]*ez5)
                    #out += [(t, list(y))]

                # Update step size
                if E > 0.0:
                    h = min(hmax, 0.85 * h * (Emax / E)**0.2)
            if (t > 395):
                phi_parallel[k, i] = 0

#
#                deltax1, deltaz1, x1, z1, ex1, ez1 = middle_step_rk4(x, z, nx, nz, 
#                        dx_di, dz_di, Bx, Bz, Ex, Ez, hds)
#                deltax2, deltaz2, x2, z2, ex2, ez2 = middle_step_rk4(x1, z1, nx, nz,
#                        dx_di, dz_di, Bx, Bz, Ex, Ez, hds)
#                deltax3, deltaz3, x3, z3, ex3, ez3 = middle_step_rk4(x, z, nx, nz, 
#                        dx_di, dz_di, Bx, Bz, Ex, Ez, deltas)
#                deltax4, deltaz4, x4, z4, ex4, ez4 = middle_step_rk4(x, z, nx, nz, 
#                        dx_di, dz_di, Bx, Bz, Ex, Ez, hds)
#                    
#                deltax = deltas/6 * (deltax1 + 2*deltax2 + 2*deltax3 + deltax4)
#                deltaz = deltas/6 * (deltaz1 + 2*deltaz2 + 2*deltaz3 + deltaz4)
#                x += deltax
#                z += deltaz
#                total_lengh += deltas
#                xlist.append(x)
#                zlist.append(z)
#                nstep += 1
#                phi_parallel[k, i] += deltas/6 * (ex1*deltax1 + ez1*deltaz1 +
#                        2.0*ex2*deltax2 + 2.0*ez2*deltaz2 +
#                        2.0*ex3*deltax3 + 2.0*ez3*deltaz3 +
#                        ex4*deltax4 + ez4*deltaz4)
#                length = math.sqrt((x-x0)**2 + (z-z0)**2)
#                if (length < dx_di and nstep > 20):
#                    phi_parallel[k, i] = 0
#                    break
#            #print k, nstep, total_lengh
#    phi_parallel = np.fromfile('phi_parallel.dat')
#    phi_parallel.tofile('phi_parallel.dat')
#    print np.max(phi_parallel), np.min(phi_parallel)
    width = 0.78
    height = 0.75
    xs = 0.14
    xe = 0.94 - xs
    ys = 0.9 - height
    #fig = plt.figure(figsize=(7,5))
    fig = plt.figure(figsize=(7, 2))
    ax1 = fig.add_axes([xs, ys, width, height])
    #kwargs_plot = {"xstep":2, "zstep":2, "vmin":-0.05, "vmax":0.05}
    kwargs_plot = {"xstep": 1, "zstep": 1, "vmin": -0.2, "vmax": 0.2}
    #kwargs_plot = {"xstep":1, "zstep":1, "vmin":-0.05, "vmax":0.05}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    phi_parallel = np.reshape(phi_parallel, (nz, nx))
    p1, cbar1 = contour_plots.plot_2d_contour(xarr, zarr, phi_parallel, ax1,
                                              fig, **kwargs_plot)
    #    xmin = np.min(xarr)
    #    zmin = np.min(zarr)
    #    xmax = np.max(xarr)
    #    zmax = np.max(zarr)
    #    p1 = ax1.imshow(phi_parallel[0:nz:8, 0:nx:8], cmap=plt.cm.jet,
    #            extent=[xmin, xmax, zmin, zmax],
    #            aspect='auto', origin='lower',
    #            vmin=kwargs_plot["vmin"], vmax=kwargs_plot["vmax"],
    #            interpolation='quadric')
    p1.set_cmap(plt.cm.seismic)
    cs = ax1.contour(
        xarr[0:nx:xstep],
        zarr[0:nz:zstep],
        Ay[0:nz:zstep, 0:nx:xstep],
        colors='black',
        linewidths=0.5,
        levels=np.arange(1, 250, 10))
    #cbar1.set_ticks(np.arange(-0.8, 1.0, 0.4))
    #ax1.tick_params(axis='x', labelbottom='off')
    plt.show()


def trace_field_line(pic_info):
    """Calculate parallel potential defined by Jan Egedal.

    Args:
        pic_info: namedtuple for the PIC simulation information.
    """
    kwargs = {"current_time": 40, "xl": 0, "xr": 200, "zb": -50, "zt": 50}
    x, z, Ay = contour_plots.read_2d_fields(pic_info, "../data/Ay.gda",
                                            **kwargs)
    x, z, Bx = contour_plots.read_2d_fields(pic_info, "../data/bx.gda",
                                            **kwargs)
    x, z, Bz = contour_plots.read_2d_fields(pic_info, "../data/bz.gda",
                                            **kwargs)
    xarr, zarr, Bz = contour_plots.read_2d_fields(pic_info, "../data/bz.gda",
                                                  **kwargs)
    x, z, Ex = contour_plots.read_2d_fields(pic_info, "../data/ex.gda",
                                            **kwargs)
    x, z, Ez = contour_plots.read_2d_fields(pic_info, "../data/ez.gda",
                                            **kwargs)
    nx, = x.shape
    nz, = z.shape
    print nx, nz

    x0 = 199.0
    z0 = 45.0

    i = int(x0 / pic_info.dx_di)
    k = int(z0 / pic_info.dz_di)
    #    x0 = xarr[i]
    #    z0 = zarr[k] - zarr[0]
    #    nstep = 0
    #    xlist = [x0]
    #    zlist = [z0]
    #    dx_di = pic_info.dx_di
    #    dz_di = pic_info.dz_di
    #    deltas = math.sqrt(dx_di**2 + dz_di**2)*0.1
    #    hds = deltas * 0.5
    #    total_lengh = 0
    #    x = x0
    #    z = z0
    #    while (x > xarr[0] and x < xarr[-1] and z > 0
    #            and z < (zarr[-1]-zarr[0]) and total_lengh < 1E2):
    #        deltax1, deltaz1, x1, z1, ex, ez = middle_step_rk4(x, z, nx, nz, 
    #                dx_di, dz_di, Bx, Bz, Ex, Ez, hds)
    #        deltax2, deltaz2, x2, z2, ex, ez = middle_step_rk4(x1, z1, nx, nz,
    #                dx_di, dz_di, Bx, Bz, Ex, Ez, hds)
    #        deltax3, deltaz3, x3, z3, ex, ez = middle_step_rk4(x, z, nx, nz, 
    #                dx_di, dz_di, Bx, Bz, Ex, Ez, deltas)
    #        deltax4, deltaz4, x4, z4, ex, ez = middle_step_rk4(x, z, nx, nz, 
    #                dx_di, dz_di, Bx, Bz, Ex, Ez, hds)
    #            
    #        x += deltas/6 * (deltax1 + 2*deltax2 + 2*deltax3 + deltax4)
    #        z += deltas/6 * (deltaz1 + 2*deltaz2 + 2*deltaz3 + deltaz4)
    #        total_lengh += deltas
    #        xlist.append(x)
    #        zlist.append(z)
    #        nstep += 1
    #        length = math.sqrt((x-x0)**2 + (z-z0)**2)
    #        if (length < dx_di and nstep > 20):
    #            print length
    #            break

    dx_di = pic_info.dx_di
    dz_di = pic_info.dz_di
    #deltas = math.sqrt(dx_di**2 + dz_di**2)
    #hds = deltas * 0.5
    hmax = dx_di * 100
    h = hmax / 4.0
    # Cash-Karp parameters
    a = [0.0, 0.2, 0.3, 0.6, 1.0, 0.875]
    b = [[], [0.2], [3.0 / 40.0, 9.0 / 40.0], [0.3, -0.9, 1.2],
         [-11.0 / 54.0, 2.5, -70.0 / 27.0, 35.0 / 27.0], [
             1631.0 / 55296.0, 175.0 / 512.0, 575.0 / 13824.0, 44275.0 /
             110592.0, 253.0 / 4096.0
         ]]
    c = [37.0 / 378.0, 0.0, 250.0 / 621.0, 125.0 / 594.0, 0.0, 512.0 / 1771.0]
    dc = [
        c[0] - 2825.0 / 27648.0, c[1] - 0.0, c[2] - 18575.0 / 48384.0,
        c[3] - 13525.0 / 55296.0, c[4] - 277.00 / 14336.0, c[5] - 0.25
    ]

    def F(t, x, z):
        indices_bl, indices_tr, delta = grid_indices(x, 0, z, nx, 1, nz, dx_di,
                                                     1, dz_di)
        ix1 = indices_bl[0]
        iz1 = indices_bl[2]
        ix2 = indices_tr[0]
        iz2 = indices_tr[2]
        offsetx = delta[0]
        offsetz = delta[2]
        v1 = (1.0 - offsetx) * (1.0 - offsetz)
        v2 = offsetx * (1.0 - offsetz)
        v3 = offsetx * offsetz
        v4 = (1.0 - offsetx) * offsetz
        if (ix1 < nx and ix2 < nx and iz1 < nz and iz2 < nz):
            bx = Bx[iz1, ix1] * v1 + Bx[iz1, ix2] * v2 + Bx[
                iz2, ix2] * v3 + Bx[iz2, ix1] * v4
            bz = Bz[iz1, ix1] * v1 + Bz[iz1, ix2] * v2 + Bz[
                iz2, ix2] * v3 + Bz[iz2, ix1] * v4
            ex = Ex[iz1, ix1] * v1 + Ex[iz1, ix2] * v2 + Ex[
                iz2, ix2] * v3 + Ex[iz2, ix1] * v4
            ez = Ez[iz1, ix1] * v1 + Ez[iz1, ix2] * v2 + Ez[
                iz2, ix2] * v3 + Ez[iz2, ix1] * v4
            absB = math.sqrt(bx**2 + bz**2)
            deltax1 = bx / absB
            deltaz1 = bz / absB
        else:
            ex = 0
            ez = 0
            deltax1 = 0
            deltaz1 = 0
        return (deltax1, deltaz1, ex, ez)

    tol = 1e-5

    x0 = xarr[i] - xarr[0]
    z0 = zarr[k] - zarr[0]
    y = [x0, z0]
    nstep = 0
    xlist = [x0]
    zlist = [z0]
    t = 0
    x = x0
    z = z0
    while (x > 0 and x < (xarr[-1] - xarr[0]) and z > 0 and
           z < (zarr[-1] - zarr[0]) and t < 4E2):
        # Compute k[i] function values.
        kx = [None] * 6
        kz = [None] * 6
        kx[0], kz[0], ex0, ez0 = F(t, x, z)
        kx[1], kz[1], ex1, ez1 = F(t + a[1] * h, x + h * (kx[0] * b[1][0]),
                                   z + h * (kz[0] * b[1][0]))
        kx[2], kz[2], ex2, ez2 = F(t + a[2] * h,
                                   x + h * (kx[0] * b[2][0] + kx[1] * b[2][1]),
                                   z + h * (kz[0] * b[2][0] + kz[1] * b[2][1]))
        kx[3], kz[3], ex3, ez3 = F(
            t + a[3] * h,
            x + h * (kx[0] * b[3][0] + kx[1] * b[3][1] + kx[2] * b[3][2]),
            z + h * (kz[0] * b[3][0] + kz[1] * b[3][1] + kz[2] * b[3][2]))
        kx[4], kz[4], ex4, ez4 = F(t + a[4] * h,
                                   x + h * (kx[0] * b[4][0] + kx[1] * b[4][1] +
                                            kx[2] * b[4][2] + kx[3] * b[4][3]),
                                   z + h * (kz[0] * b[4][0] + kz[1] * b[4][1] +
                                            kz[2] * b[4][2] + kz[3] * b[4][3]))
        kx[5], kz[5], ex5, ez5 = F(
            t + a[5] * h,
            x + h * (kx[0] * b[5][0] + kx[1] * b[5][1] + kx[2] * b[5][2] +
                     kx[3] * b[5][3] + kx[4] * b[5][4]),
            z + h * (kz[0] * b[5][0] + kz[1] * b[5][1] + kz[2] * b[5][2] +
                     kz[3] * b[5][3] + kz[4] * b[5][4]))

        # Estimate current error and current maximum error.
        E = norm([
            h * (kx[0] * dc[0] + kx[1] * dc[1] + kx[2] * dc[2] + kx[3] * dc[3]
                 + kx[4] * dc[4] + kx[5] * dc[5]),
            h * (kz[0] * dc[0] + kz[1] * dc[1] + kz[2] * dc[2] + kz[3] * dc[3]
                 + kz[4] * dc[4] + kz[5] * dc[5])
        ])
        Emax = tol * max(norm(y), 1.0)

        # Update solution if error is OK.
        if E < Emax:
            t += h
            y[0] += h * (kx[0] * c[0] + kx[1] * c[1] + kx[2] * c[2] + kx[3] *
                         c[3] + kx[4] * c[4] + kx[5] * c[5])
            y[1] += h * (kz[0] * c[0] + kz[1] * c[1] + kz[2] * c[2] + kz[3] *
                         c[3] + kz[4] * c[4] + kz[5] * c[5])
            x = y[0]
            z = y[1]
            xlist.append(x)
            zlist.append(z)
            #out += [(t, list(y))]

        # Update step size
        if E > 0.0:
            h = min(hmax, 0.85 * h * (Emax / E)**0.2)

    width = 0.78
    height = 0.75
    xs = 0.14
    xe = 0.94 - xs
    ys = 0.9 - height
    #fig = plt.figure(figsize=(7,5))
    fig = plt.figure(figsize=(7, 2))
    ax1 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep": 2, "zstep": 2}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = contour_plots.plot_2d_contour(xarr, zarr, Ay, ax1, fig,
                                              **kwargs_plot)
    p1.set_cmap(plt.cm.seismic)
    #    cs = ax1.contour(xarr[0:nx:xstep], zarr[0:nz:zstep], Ay[0:nz:zstep, 0:nx:xstep], 
    #            colors='white', linewidths=0.5, levels=np.arange(0, 252, 1))
    p2 = ax1.plot(xlist, zlist + zarr[0], color='black')
    #cbar1.set_ticks(np.arange(-0.8, 1.0, 0.4))
    #ax1.tick_params(axis='x', labelbottom='off')
    plt.show()


def middle_step_rk4(x, z, nx, nz, dx, dz, Bx, Bz, Ex, Ez, ds):
    """Middle step of Runge-Kutta method to trace the magnetic field line.

    Args:
        x, z: the coordinates of current point.
        nx, nz: the dimensions of the data.
        pic_info: namedtuple for the PIC simulation information.
        Ex, Ez: the electric field arrays.
        Bx, Bz: the magnetic field arrays.
        ds: the step size.
    """
    indices_bl, indices_tr, delta = grid_indices(x, 0, z, nx, 1, nz, dx, 1, dz)
    ix1 = indices_bl[0]
    iz1 = indices_bl[2]
    ix2 = indices_tr[0]
    iz2 = indices_tr[2]
    offsetx = delta[0]
    offsetz = delta[2]
    v1 = (1.0 - offsetx) * (1.0 - offsetz)
    v2 = offsetx * (1.0 - offsetz)
    v3 = offsetx * offsetz
    v4 = (1.0 - offsetx) * offsetz
    if (ix1 < nx and ix2 < nx and iz1 < nz and iz2 < nz):
        bx = Bx[iz1, ix1] * v1 + Bx[iz1, ix2] * v2 + Bx[iz2, ix2] * v3 + Bx[
            iz2, ix1] * v4
        bz = Bz[iz1, ix1] * v1 + Bz[iz1, ix2] * v2 + Bz[iz2, ix2] * v3 + Bz[
            iz2, ix1] * v4
        ex = Ex[iz1, ix1] * v1 + Ex[iz1, ix2] * v2 + Ex[iz2, ix2] * v3 + Ex[
            iz2, ix1] * v4
        ez = Ez[iz1, ix1] * v1 + Ez[iz1, ix2] * v2 + Ez[iz2, ix2] * v3 + Ez[
            iz2, ix1] * v4
        absB = math.sqrt(bx**2 + bz**2)
        deltax1 = bx / absB
        deltaz1 = bz / absB
    else:
        ex = 0
        ez = 0
        deltax1 = 0
        deltaz1 = 0
    x1 = x + deltax1 * ds
    z1 = z + deltaz1 * ds
    return (deltax1, deltaz1, x1, z1, ex, ez)


def grid_indices(x, y, z, nx, ny, nz, dx, dy, dz):
    """Get the grid indices for point (x, y, z).

    Args:
        x, y, z: the coordinates of the point.
        nx, ny, nz: the domain sizes.
        dx, dy, dz: the grid sizes.

    Returns:
        indices_bl: the bottom left corner indices.
        indices_tr: the top right corner indices.
        delta: the offsets from the bottom left corner [0, 1].
    """
    indices_bl = np.zeros(3, dtype=np.int)
    indices_tr = np.zeros(3, dtype=np.int)
    delta = np.zeros(3)
    indices_bl[0] = int(math.floor(x / dx))
    indices_bl[1] = int(math.floor(y / dy))
    indices_bl[2] = int(math.floor(z / dz))
    indices_tr[0] = indices_bl[0] + 1
    indices_tr[1] = indices_bl[1] + 1
    indices_tr[2] = indices_bl[2] + 1
    delta[0] = x / dx - indices_bl[0]
    delta[1] = y / dy - indices_bl[1]
    delta[2] = z / dz - indices_bl[2]
    return (indices_bl, indices_tr, delta)


def cal_phi_parallel(pic_info):
    """Calculate parallel potential defined by Jan Egedal.

    Args:
        pic_info: namedtuple for the PIC simulation information.
    """
    kwargs = {"current_time": 160, "xl": 0, "xr": 200, "zb": -50, "zt": 50}
    x, z, Ay = contour_plots.read_2d_fields(pic_info, "../data/Ay.gda",
                                            **kwargs)
    x, z, Bx = contour_plots.read_2d_fields(pic_info, "../data/bx.gda",
                                            **kwargs)
    x, z, By = contour_plots.read_2d_fields(pic_info, "../data/by.gda",
                                            **kwargs)
    xarr, zarr, Bz = contour_plots.read_2d_fields(pic_info, "../data/bz.gda",
                                                  **kwargs)
    x, z, Ex = contour_plots.read_2d_fields(pic_info, "../data/ex.gda",
                                            **kwargs)
    x, z, Ey = contour_plots.read_2d_fields(pic_info, "../data/ey.gda",
                                            **kwargs)
    x, z, Ez = contour_plots.read_2d_fields(pic_info, "../data/ez.gda",
                                            **kwargs)
    absB = np.sqrt(Bx * Bx + By * By + Bz * Bz)
    Epara = (Ex * Bx + Ey * By + Ez * Bz) / absB
    etot = np.sqrt(Ex * Ex + Ey * Ey + Ez * Ez)
    nx, = x.shape
    nz, = z.shape
    print nx, nz

    width = 0.78
    height = 0.75
    xs = 0.14
    xe = 0.94 - xs
    ys = 0.9 - height
    #fig = plt.figure(figsize=(7,5))
    fig = plt.figure(figsize=(7, 2))
    ax1 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep": 2, "zstep": 2}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    #p1, cbar1 = contour_plots.plot_2d_contour(xarr, zarr, etot, 
    #        ax1, fig, **kwargs_plot)
    #p1.set_cmap(plt.cm.seismic)
    cs = ax1.contour(
        xarr[0:nx:xstep],
        zarr[0:nz:zstep],
        Ay[0:nz:zstep, 0:nx:xstep],
        colors='white',
        linewidths=0.5,
        levels=np.arange(150, 252, 1))
    xlist = []
    zlist = []
    dx_di = pic_info.dx_di
    dz_di = pic_info.dz_di
    fig, ax = plt.subplots()
    #fig, ax2 = plt.subplots()
    phi_parallel = np.zeros((nz, nx))
    #for csp in cs.collections[65:70]:
    for csp in cs.collections[0:10]:
        for p in csp.get_paths():
            v = p.vertices
            x = v[:, 0]
            z = v[:, 1]
            if (math.fabs(x[0] - x[-1]) > dx_di or
                    math.fabs(z[0] - z[-1]) > dz_di):
                xlist.extend(x)
                zlist.extend(z)
                lenx = len(x)
                phip = np.zeros(lenx)
                epara = np.zeros(lenx)
                if (x[-1] < x[0]):
                    x = x[::-1]
                    z = z[::-1]
                for i in range(1, lenx):
                    dx = x[i] - x[i - 1]
                    dz = z[i] - z[i - 1]
                    indices_bl, indices_tr, delta = grid_indices(
                        x[i] - xarr[0], 0, z[i] - zarr[0], nx, 1, nz, dx_di, 1,
                        dz_di)
                    ix1 = indices_bl[0]
                    iz1 = indices_bl[2]
                    ix2 = indices_tr[0]
                    iz2 = indices_tr[2]
                    offsetx = delta[0]
                    offsetz = delta[2]
                    v1 = (1.0 - offsetx) * (1.0 - offsetz)
                    v2 = offsetx * (1.0 - offsetz)
                    v3 = offsetx * offsetz
                    v4 = (1.0 - offsetx) * offsetz
                    bx = Bx[iz1, ix1] * v1 + Bx[iz1, ix2] * v2 + Bx[
                        iz2, ix2] * v3 + Bx[iz2, ix1] * v4
                    by = By[iz1, ix1] * v1 + By[iz1, ix2] * v2 + By[
                        iz2, ix2] * v3 + By[iz2, ix1] * v4
                    bz = Bz[iz1, ix1] * v1 + Bz[iz1, ix2] * v2 + Bz[
                        iz2, ix2] * v3 + Bz[iz2, ix1] * v4
                    ex = Ex[iz1, ix1] * v1 + Ex[iz1, ix2] * v2 + Ex[
                        iz2, ix2] * v3 + Ex[iz2, ix1] * v4
                    ey = Ey[iz1, ix1] * v1 + Ey[iz1, ix2] * v2 + Ey[
                        iz2, ix2] * v3 + Ey[iz2, ix1] * v4
                    ez = Ez[iz1, ix1] * v1 + Ez[iz1, ix2] * v2 + Ez[
                        iz2, ix2] * v3 + Ez[iz2, ix1] * v4
                    epara[i] = Epara[iz1,ix1]*v1 + Epara[iz1,ix2]*v2 + \
                            Epara[iz2,ix2]*v3 + Epara[iz2,ix1]*v4
                    dy = by * dx / bx
                    #btot = math.sqrt(bx*bx + by*by + bz*bz)
                    #ds = math.sqrt(dx*dx + dy*dy + dz*dz)
                    #print math.acos((bx*dx + by*dy + bz*dz) / (btot*ds))
                    phip[i] = phip[i - 1] + ex * dx + ey * dy + ez * dz
                    ix = int(math.floor(x[i] / dx_di))
                    iz = int(math.floor((z[i] - zarr[0]) / dz_di))
                    phi_parallel[iz, ix] = phip[i]
                ax.plot(x, phip)
                #ax.plot(x, epara)
    print np.max(phi_parallel), np.min(phi_parallel)
    fig = plt.figure(figsize=(7, 2))
    ax1 = fig.add_axes([xs, ys, width, height])
    kwargs_plot = {"xstep": 2, "zstep": 2}
    xstep = kwargs_plot["xstep"]
    zstep = kwargs_plot["zstep"]
    p1, cbar1 = contour_plots.plot_2d_contour(xarr, zarr, phi_parallel, ax1,
                                              fig, **kwargs_plot)
    p1.set_cmap(plt.cm.seismic)
    plt.show()


if __name__ == "__main__":
    pic_info = pic_information.get_pic_info('../..')
    #parallel_potential(pic_info)
    trace_field_line(pic_info)
    #cal_phi_parallel(pic_info)
