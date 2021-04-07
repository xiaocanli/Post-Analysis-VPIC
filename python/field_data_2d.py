"""
Module of procedures dealing with 2D field or hydro data.
For 3D simulations, it will take a 2D slice of the data.
"""
import math

import h5py
import numpy as np


def read_2d_gda_fields(pic_info, fname, current_time, xl, xr, zb, zt):
    """Read 2D fields data from a gda file

    Args:
        pic_info (namedtuple): for the PIC simulation information
        fname (string): the filename
        current_time (integer): current time frame
        xl, xr (float): left and right x position in di
        zb, zt (float): top and bottom z position in di
    """
    print("Reading data from %s" % fname)
    print("xrange: (%f, %f)" % (xl, xr))
    print("zrange: (%f, %f)" % (zb, zt))
    nx = pic_info.nx
    nz = pic_info.nz
    x_di = np.copy(pic_info.x_di)
    z_di = np.copy(pic_info.z_di)
    dx_di = pic_info.dx_di
    dz_di = pic_info.dz_di
    xmin = np.min(x_di)
    xmax = np.max(x_di)
    zmin = np.min(z_di)
    zmax = np.max(z_di)
    if xl <= xmin:
        xl_index = 0
    else:
        xl_index = int(math.floor((xl - xmin) / dx_di))
    if xr >= xmax:
        xr_index = nx - 1
    else:
        xr_index = int(math.ceil((xr - xmin) / dx_di))
    if zb <= zmin:
        zb_index = 0
    else:
        zb_index = int(math.floor((zb - zmin) / dz_di))
    if zt >= zmax:
        zt_index = nz - 1
    else:
        zt_index = int(math.ceil((zt - zmin) / dz_di))
    nx1 = xr_index - xl_index + 1
    nz1 = zt_index - zb_index + 1
    fp = np.zeros((nz1, nx1), dtype=np.float32)
    offset = nx * nz * current_time * 4
    fdata = np.memmap(fname,
                      dtype='float32',
                      mode='r',
                      offset=offset,
                      shape=(nz, nx),
                      order='C')
    xc = x_di[xl_index:xr_index + 1]
    zc = z_di[zb_index:zt_index + 1]
    fp = fdata[zb_index:zt_index + 1, xl_index:xr_index + 1]
    return (xc, zc, fp)


def read_fields_h5(vname, tindex, **kwargs):
    """read electric and magnetic fields in HDF5 format

    Args:
        vname (string): variable name
        tindex (int): time index
    """
    pic_run_dir = kwargs.get("run_dir")
    if kwargs.get("smoothed_data"):
        fname = (pic_run_dir + kwargs.get("dir_smooth_data") + "/fields_" +
                 str(tindex) + ".h5")
    else:
        if kwargs.get("time_averaged_field"):
            fdir = pic_run_dir + "fields-avg-hdf5/T." + str(tindex) + "/"
        else:
            fdir = pic_run_dir + "field_hdf5/T." + str(tindex) + "/"
        fname = fdir + "fields_" + str(tindex) + ".h5"
    normal = kwargs.get("normal")
    plane_index = kwargs.get("plane_index")
    with h5py.File(fname, 'r') as fh:
        group = fh["Timestep_" + str(tindex)]
        if vname == "absb":
            bvec = {}
            for var in ["cbx", "cby", "cbz"]:
                dset = group[var]
                if normal == 'x':
                    bvec[var] = dset[plane_index, :, :]
                elif normal == 'y':
                    bvec[var] = dset[:, plane_index, :]
                else:
                    bvec[var] = dset[:, :, plane_index]
            field_2d = np.sqrt(bvec["cbx"]**2 + bvec["cby"]**2 +
                               bvec["cbz"]**2)
        else:
            dset = group[vname]
            if normal == 'x':
                field_2d = dset[plane_index, :, :]
            elif normal == 'y':
                field_2d = dset[:, plane_index, :]
            else:
                field_2d = dset[:, :, plane_index]
    return field_2d


def read_current_density_h5(vname, tindex, **kwargs):
    """read current density in HDF5 format

    Args:
        vname (string): variable name
        tindex (int): time index
    """
    pic_run_dir = kwargs.get("run_dir")
    # Electron
    if kwargs.get("smoothed_data"):
        fname = (pic_run_dir + kwargs.get("dir_smooth_data") + "/hydro_ion_" +
                 str(tindex) + ".h5")
    else:
        if kwargs.get("time_averaged_field"):
            data_dir = pic_run_dir + "hydro-avg-hdf5/T." + str(tindex) + "/"
        else:
            data_dir = pic_run_dir + "hydro_hdf5/T." + str(tindex) + "/"
        fname = data_dir + "hydro_electron_" + str(tindex) + ".h5"
    normal = kwargs.get("normal")
    plane_index = kwargs.get("plane_index")
    with h5py.File(fname, 'r') as fh:
        group = fh["Timestep_" + str(tindex)]
        if vname == "absj":
            j = {}
            for var in ["jx", "jy", "jz"]:
                dset = group[var]
                if normal == 'x':
                    j[var] = dset[plane_index, :, :]
                elif normal == 'y':
                    j[var] = dset[:, plane_index, :]
                else:
                    j[var] = dset[:, :, plane_index]
            field_2d = np.sqrt(j["jx"]**2 + j["jy"]**2 + j["jz"]**2)
        else:
            dset = group[vname]
            if normal == 'x':
                field_2d = dset[plane_index, :, :]
            elif normal == 'y':
                field_2d = dset[:, plane_index, :]
            else:
                field_2d = dset[:, :, plane_index]

    # Ion
    if kwargs.get("smoothed_data"):
        fname = (pic_run_dir + kwargs.get("dir_smooth_data") + "/hydro_ion_" +
                 str(tindex) + ".h5")
    else:
        fname = data_dir + "hydro_ion_" + str(tindex) + ".h5"
    with h5py.File(fname, 'r') as fh:
        group = fh["Timestep_" + str(tindex)]
        if vname == "absj":
            for var in ["jx", "jy", "jz"]:
                dset = group[var]
                if normal == 'x':
                    j[var] += dset[plane_index, :, :]
                elif normal == 'y':
                    j[var] += dset[:, plane_index, :]
                else:
                    j[var] += dset[:, :, plane_index]
            field_2d = np.sqrt(j["jx"]**2 + j["jy"]**2 + j["jz"]**2)
        else:
            dset = group[vname]
            if normal == 'x':
                field_2d += dset[plane_index, :, :]
            elif normal == 'y':
                field_2d += dset[:, plane_index, :]
            else:
                field_2d += dset[:, :, plane_index]
    return field_2d


def read_hydro_h5(vname, tindex, **kwargs):
    """Read hydro data from file in HDF5 format

    Args:
        vname (string): variable name
        tindex (int): time index
    """
    pic_run_dir = kwargs.get("run_dir")
    ehydro_list = [
        "ne", "vex", "vey", "vez", "pexx", "pexy", "pexz", "peyx", "peyy",
        "peyz", "pezx", "pezy", "pezz"
    ]
    ehydro_list += ["uex", "uey", "uez"]
    if vname in ehydro_list:
        if kwargs.get("smoothed_data"):
            fname = (pic_run_dir + kwargs.get("dir_smooth_data") +
                     "/hydro_electron_" + str(tindex) + ".h5")
        else:
            if kwargs.get("time_averaged_field"):
                data_dir = (pic_run_dir + "hydro-avg-hdf5/T." + str(tindex) +
                            "/")
            else:
                data_dir = pic_run_dir + "hydro_hdf5/T." + str(tindex) + "/"
            fname = data_dir + "hydro_electron_" + str(tindex) + ".h5"
        pmass = 1.0
    else:
        if kwargs.get("smoothed_data"):
            fname = (pic_run_dir + kwargs.get("dir_smooth_data") +
                     "/hydro_ion_" + str(tindex) + ".h5")
        else:
            fname = data_dir + "hydro_ion_" + str(tindex) + ".h5"
        pmass = kwargs.get("mi_me")
    normal = kwargs.get("normal")
    plane_index = kwargs.get("plane_index")
    with h5py.File(fname, 'r') as fh:
        group = fh["Timestep_" + str(tindex)]
        if vname[0] == 'n':
            var = "rho"
        elif vname[0] == 'v':
            var = "j" + vname[-1]
        elif vname[0] == 'u':
            var = "p" + vname[-1]
        else:
            vtmp = "t" + vname[2:]
            if vtmp in group:
                var = vtmp
            else:
                var = "t" + vname[-1] + vname[-2]
        dset = group[var]
        if normal == 'x':
            field_2d = dset[plane_index, :, :]
        elif normal == 'y':
            field_2d = dset[:, plane_index, :]
        else:
            field_2d = dset[:, :, plane_index]
        if vname[0] == 'n':
            field_2d = np.abs(field_2d)
        elif vname[0] == 'v':
            dset = group["rho"]
            if normal == 'x':
                field_2d /= dset[plane_index, :, :]
            elif normal == 'y':
                field_2d /= dset[:, plane_index, :]
            else:
                field_2d /= dset[:, :, plane_index]
        elif vname[0] == 'u':
            dset = group["rho"]
            if normal == 'x':
                field_2d /= np.abs(dset[plane_index, :, :])
            elif normal == 'y':
                field_2d /= np.abs(dset[:, plane_index, :])
            else:
                field_2d /= np.abs(dset[:, :, plane_index])
            field_2d /= pmass
        else:
            dset = group["rho"]
            if normal == 'x':
                rho = dset[plane_index, :, :]
            elif normal == 'y':
                rho = dset[:, plane_index, :]
            else:
                rho = dset[:, :, plane_index]
            dset = group["j" + vname[-2]]
            if normal == 'x':
                v = dset[plane_index, :, :] / rho
            elif normal == 'y':
                v = dset[:, plane_index, :] / rho
            else:
                v = dset[:, :, plane_index] / rho
            dset = group["p" + vname[-1]]
            if normal == 'x':
                field_2d -= v * dset[plane_index, :, :]
            elif normal == 'y':
                field_2d -= v * dset[:, plane_index, :]
            else:
                field_2d -= v * dset[:, :, plane_index]
    return field_2d


def read_2d_hdf5_fields(vname, tindex, **kwargs):
    """Read data from an HDF5 file

    Args:
        vname (string): variable name
        tindex (int): time index
    """
    # the simulation directory
    if "run_dir" in kwargs:
        run_dir = kwargs.get("run_dir")
    else:
        run_dir = "./"

    # whether the data is smoothed spatially
    if "smoothed_data" in kwargs:
        smoothed_data = kwargs.get("smoothed_data")
        if "dir_smooth_data" in kwargs:
            dir_smooth_data = kwargs.get("dir_smooth_data")
        else:
            dir_smooth_data = "../data_smooth/"
    else:
        smoothed_data = False
        dir_smooth_data = ""

    # whether the data is time averaged
    if "time_averaged_field" in kwargs:
        time_averaged_field = kwargs.get("time_averaged_field")
    else:
        time_averaged_field = False

    # the normal direction
    if "normal" in kwargs:
        normal = kwargs.get("normal")
    else:
        normal = "y"

    # the plane index along the normal direction
    if "plane_index" in kwargs:
        plane_index = kwargs.get("plane_index")
    else:
        plane_index = "y"

    # ion to electron mass ratio
    if "plane_index" in kwargs:
        mi_me = kwargs.get("mi_me")
    else:
        mi_me = 1.0

    data_config = {
        "run_dir": run_dir,
        "smoothed_data": smoothed_data,
        "dir_smooth_data": dir_smooth_data,
        "time_averaged_field": time_averaged_field,
        "normal": normal,
        "plane_index": plane_index,
        "mi_me": mi_me
    }

    fields_list = ["cbx", "cby", "cbz", "absb", "ex", "ey", "ez"]
    jlist = ["jx", "jy", "jz", "absj"]
    if vname in fields_list:  # electric and magnetic fields
        field_2d = read_fields_h5(vname, tindex, **data_config)
    elif vname in jlist:  # current density
        field_2d = read_current_density_h5(vname, tindex, **data_config)
    else:  # density, velocity, momentum, pressure tensor
        field_2d = read_hydro_h5(vname, tindex, **data_config)
    return field_2d
