"""
Calculate ExB drift velocity
"""
import argparse
import math
import multiprocessing
import os

import numpy as np
from joblib import Parallel, delayed
from scipy.ndimage.filters import gaussian_filter

from contour_plots import read_2d_fields
from dolointerpolation import MultilinearInterpolator
from energy_conversion import read_data_from_json


def calc_exb(run_dir, run_name, tframe, coords):
    """Calculate ExB drift velocity

    Args:
        run_dir: PIC run directory
        run_name: PIC run name
        trame: time frame
        coords: coordinates for different fields
    """
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    nx = pic_info.nx
    nz = pic_info.nz
    kwargs = {"current_time": tframe, "xl": 0, "xr": pic_info.lx_di,
              "zb": -0.5 * pic_info.lz_di, "zt": 0.5 * pic_info.lz_di}
    size_one_frame = pic_info.nx * pic_info.nz * 4
    sigma = 3
    fname = run_dir + "data/bx.gda"
    _, _, bx = read_2d_fields(pic_info, fname, **kwargs)
    f = MultilinearInterpolator(coords["smin_ez_bx"],
                                coords["smax_ez_bx"],
                                coords["orders"],
                                dtype=np.float32)
    f.set_values(np.atleast_2d(np.transpose(bx).flatten()))
    bx = np.transpose(f(coords["coord"]).reshape((nx, nz)))

    fname = run_dir + "data/by.gda"
    _, _, by = read_2d_fields(pic_info, fname, **kwargs)
    f = MultilinearInterpolator(coords["smin_by"],
                                coords["smax_by"],
                                coords["orders"],
                                dtype=np.float32)
    f.set_values(np.atleast_2d(np.transpose(by).flatten()))
    by = np.transpose(f(coords["coord"]).reshape((nx, nz)))

    fname = run_dir + "data/bz.gda"
    _, _, bz = read_2d_fields(pic_info, fname, **kwargs)
    f = MultilinearInterpolator(coords["smin_ex_bz"],
                                coords["smax_ex_bz"],
                                coords["orders"],
                                dtype=np.float32)
    f.set_values(np.atleast_2d(np.transpose(bz).flatten()))
    bz = np.transpose(f(coords["coord"]).reshape((nx, nz)))

    fname = run_dir + "data/ex.gda"
    _, _, ex = read_2d_fields(pic_info, fname, **kwargs)
    ex = gaussian_filter(ex, sigma)
    f = MultilinearInterpolator(coords["smin_ex_bz"],
                                coords["smax_ex_bz"],
                                coords["orders"],
                                dtype=np.float32)
    f.set_values(np.atleast_2d(np.transpose(ex).flatten()))
    ex = np.transpose(f(coords["coord"]).reshape((nx, nz)))

    fname = run_dir + "data/ey.gda"
    _, _, ey = read_2d_fields(pic_info, fname, **kwargs)
    ey = gaussian_filter(ey, sigma)
    f = MultilinearInterpolator(coords["smin_h"],
                                coords["smax_h"],
                                coords["orders"],
                                dtype=np.float32)
    f.set_values(np.atleast_2d(np.transpose(ey).flatten()))
    ey = np.transpose(f(coords["coord"]).reshape((nx, nz)))

    fname = run_dir + "data/ez.gda"
    _, _, ez = read_2d_fields(pic_info, fname, **kwargs)
    ez = gaussian_filter(ez, sigma)
    f = MultilinearInterpolator(coords["smin_ez_bx"],
                                coords["smax_ez_bx"],
                                coords["orders"],
                                dtype=np.float32)
    f.set_values(np.atleast_2d(np.transpose(ez).flatten()))
    ez = np.transpose(f(coords["coord"]).reshape((nx, nz)))

    ib2 = 1.0 / (bx**2 + by**2 + bz**2)
    exb_x = (ey * bz - ez * by) * ib2
    exb_y = (ez * bx - ex * bz) * ib2
    exb_z = (ex * by - ey * bx) * ib2

    fname = run_dir + "data/exb_x.gda"
    with open(fname, 'a+') as f:
        offset = size_one_frame * tframe
        f.seek(offset, os.SEEK_SET)
        exb_x.tofile(f)

    fname = run_dir + "data/exb_y.gda"
    with open(fname, 'a+') as f:
        offset = size_one_frame * tframe
        f.seek(offset, os.SEEK_SET)
        exb_y.tofile(f)

    fname = run_dir + "data/exb_z.gda"
    with open(fname, 'a+') as f:
        offset = size_one_frame * tframe
        f.seek(offset, os.SEEK_SET)
        exb_z.tofile(f)


def get_coordinates(pic_info):
    """Get the coordinates where the fields are
    """
    smime = math.sqrt(pic_info.mime)
    dx = pic_info.dx_di * smime
    dz = pic_info.dz_di * smime
    dxh = dx * 0.5
    dzh = dz * 0.5
    nx_pic = pic_info.nx
    nz_pic = pic_info.nz
    lx_pic = pic_info.lx_di * smime
    lz_pic = pic_info.lz_di * smime
    x1 = np.linspace(-dxh, lx_pic + dxh, nx_pic + 2)
    x2 = np.linspace(-dx, lx_pic, nx_pic + 2)
    z1 = np.linspace(-dzh - 0.5 * lz_pic, 0.5 * lz_pic + dzh, nz_pic + 2)
    z2 = np.linspace(-dz - 0.5 * lz_pic, 0.5 * lz_pic, nz_pic + 2)
    points_x, points_z = np.broadcast_arrays(x2[1:-1].reshape(-1, 1), z2[1:-1])
    coord = np.vstack((points_x.flatten(), points_z.flatten()))
    orders = [nx_pic, nz_pic]
    smin_h = [x2[1], z2[1]]         # for hydro, Ey
    smax_h = [x2[-2], z2[-2]]
    smin_ex_bz = [x1[1], z2[1]]     # for Ex, Bz
    smax_ex_bz = [x1[-2], z2[-2]]
    smin_ez_bx = [x2[1], z1[1]]     # for Ez, Bx
    smax_ez_bx = [x2[-2], z1[-2]]
    smin_by = [x1[1], z1[1]]        # for By
    smax_by = [x1[-2], z1[-2]]

    coords = {"coord": coord, "orders": orders, "smin_h": smin_h,
              "smax_h": smax_h, "smin_ex_bz": smin_ex_bz,
              "smax_ex_bz": smax_ex_bz, "smin_ez_bx": smin_ez_bx,
              "smax_ez_bx": smax_ez_bx, "smin_by": smin_by,
              "smax_by": smax_by}

    return coords


def get_cmd_args():
    """Get command line arguments
    """
    default_run_name = 'mime400_beta002_bg00'
    default_run_dir = ('/net/scratch3/xiaocanli/reconnection/mime400/' +
                       default_run_name + '/')
    parser = argparse.ArgumentParser(description='Calculate ExB drift')
    parser.add_argument('--species', action="store", default='e',
                        help='particle species')
    parser.add_argument('--run_dir', action="store", default=default_run_dir,
                        help='run directory')
    parser.add_argument('--run_name', action="store", default=default_run_name,
                        help='run name')
    parser.add_argument('--multi_runs', action="store_true", default=False,
                        help='whether analyzing multiple PIC runs')
    return parser.parse_args()


def process_input(runs_root_dir, run_name, coords):
    """process one PIC run"""
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    tframes = range(pic_info.ntf)
    run_dir = runs_root_dir + run_name + '/'
    for tframe in tframes:
        calc_exb(run_dir, run_name, tframe, coords)


def main():
    """business logic for when running this module as the primary one!"""
    args = get_cmd_args()
    run_name = args.run_name
    run_dir = args.run_dir
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    coords = get_coordinates(pic_info)
    # runs_root_dir = "/net/scratch3/xiaocanli/reconnection/mime400/"
    # run_names = ["mime400_beta002_bg00",
    #              "mime400_beta002_bg02",
    #              "mime400_beta002_bg04",
    #              "mime400_beta002_bg08"]
    runs_root_dir = "/net/scratch3/xiaocanli/reconnection/frequent_dump/"
    run_names = ["mime25_beta002_guide00_frequent_dump",
                 "mime25_beta002_guide02_frequent_dump",
                 "mime25_beta002_guide05_frequent_dump",
                 "mime25_beta002_guide10_frequent_dump"]
    if args.multi_runs:
        ncores = multiprocessing.cpu_count()
        ncores = 4
        Parallel(n_jobs=ncores)(delayed(process_input)(runs_root_dir,
                                                       run_name, coords)
                                for run_name in run_names)
    else:
        calc_exb(run_dir, run_name, 10, coords)


if __name__ == "__main__":
    main()
