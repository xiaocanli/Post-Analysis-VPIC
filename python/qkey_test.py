"""
Test if the 'q' key is unique.
"""
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D


def test_unique_q(step):
    """Test if the 'q' key is unique
    """
    # step = 10
    # fname = '/scratch3/xiaocanli/open3d-h5tracer/particle/T.'
    # fname = fname + str(step) + '/electron_tracer.h5p'
    run_path = '/net/scratch3/xiaocanli/tracer_test_avg/'
    fname = run_path + '/tracer_hdf5/T.1000/electron_tracer.h5p'
    file = h5py.File(fname, 'r')
    group_name = '/Step#' + str(step)
    group = file[group_name]
    dset_q = group['q']
    sz, = dset_q.shape
    qkey = np.zeros(sz)
    dset_q.read_direct(qkey)
    print qkey[0:10]
    uq, szq = np.unique(qkey, return_counts=True)
    # print sz, len(szq)
    print sz, len(uq)
    file.close


if __name__ == "__main__":
    for i in range(1000):
        test_unique_q(i)
