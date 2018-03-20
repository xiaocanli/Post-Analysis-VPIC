"""
Analysis procedures for combining particle energy spectrum
"""
import argparse
import math
import multiprocessing

import numpy as np
from joblib import Parallel, delayed

from json_functions import read_data_from_json
from shell_functions import mkdir_p

# define some spectrum parameters here
NBINS = 800
EMIN = 1E-5
EMAX = 1E3
INCLUDE_BFIELDS = False

def combine_energy_spectrum(run_dir, run_name, tframe, species='e'):
    """Combine particle energy spectrum from different mpi_rank

    Args:
        run_dir: PIC simulation directory
        run_name: PIC simulation run name
        tframe: time frame
        species: 'e' for electrons, 'H' for ions
    """
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    interval = pic_info.fields_interval
    mpi_size = pic_info.topology_x * pic_info.topology_y * pic_info.topology_z
    rank = 0
    ndata = (NBINS + 3) if INCLUDE_BFIELDS else NBINS
    tindex = tframe * interval
    fname_pre = run_dir + 'hydro/T.' + str(tindex)
    if species == 'h':
        species = 'H'
    fname_pre += '/spectrum-' + species + 'hydro.' + str(tindex)
    fname = fname_pre + '.' + str(rank)
    fdata = np.fromfile(fname, dtype=np.float32)
    dsz, = fdata.shape
    nzone = dsz / ndata
    for rank in range(1, mpi_size):
        fname = fname_pre + '.' + str(rank)
        fdata += np.fromfile(fname, dtype=np.float32)
    print("number of zones: %d" % nzone)
    flog_tot = np.zeros(NBINS)
    for i in range(nzone):
        if INCLUDE_BFIELDS:
            flog = fdata[i*ndata+3:(i+1)*ndata]
        else:
            flog = fdata[i*ndata:(i+1)*ndata]
        flog_tot += flog
    emin_log = math.log10(EMIN)
    emax_log = math.log10(EMAX)
    elog = 10**(np.linspace(emin_log, emax_log, NBINS))
    delog = np.gradient(elog)
    flog_tot /= delog
    fdir = '../data/spectra/' + run_name + '/'
    mkdir_p(fdir)
    fname = fdir + 'spectrum-' + species.lower() + '.' + str(tframe)
    flog_tot.tofile(fname)


def get_cmd_args():
    """Get command line arguments """
    default_run_name = 'mime400_beta002_bg00'
    default_run_dir = ('/net/scratch3/xiaocanli/reconnection/mime400/' +
                       'mime400_beta002_bg00/')
    parser = argparse.ArgumentParser(description='Combining particle spectra')
    parser.add_argument('--species', action="store", default='e',
                        help='particle species')
    parser.add_argument('--run_dir', action="store", default=default_run_dir,
                        help='run directory')
    parser.add_argument('--run_name', action="store", default=default_run_name,
                        help='run name')
    parser.add_argument('--tframe', action="store", default='30', type=int,
                        help='Time frame for fields')
    parser.add_argument('--multi_frames', action="store_true", default=False,
                        help='whether analyzing multiple frames')
    return parser.parse_args()


def process_input(run_dir, run_name, tframe):
    """process one time frame"""
    print("Time frame: %d" % tframe)
    combine_energy_spectrum(run_dir, run_name, tframe, species='e')
    combine_energy_spectrum(run_dir, run_name, tframe, species='h')


def main():
    """business logic for when running this module as the primary one!"""
    args = get_cmd_args()
    run_name = args.run_name
    run_dir = args.run_dir
    picinfo_fname = '../data/pic_info/pic_info_' + run_name + '.json'
    pic_info = read_data_from_json(picinfo_fname)
    if args.multi_frames:
        ncores = multiprocessing.cpu_count()
        tframes = range(pic_info.ntf)
        Parallel(n_jobs=ncores)(delayed(process_input)(run_dir,
                                                       run_name,
                                                       tframe)
                                for tframe in tframes)
    else:
        combine_energy_spectrum(run_dir, run_name,
                                args.tframe, species=args.species)


if __name__ == "__main__":
    main()
