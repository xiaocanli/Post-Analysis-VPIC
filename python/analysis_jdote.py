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
from matplotlib import rc
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D

import pic_information

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

font = {'family' : 'serif',
        #'color'  : 'darkred',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 24,
        }

if __name__ == "__main__":
    pic_info = pic_information.get_pic_info()
    ntp = pic_info.ntp
    vthe = pic_info.ntp
