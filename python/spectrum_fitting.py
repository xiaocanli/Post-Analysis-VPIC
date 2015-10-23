"""
Analysis procedures for particle energy spectrum fitting.
"""
import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.optimize import curve_fit
import os.path
import pic_information
import fitting_funcs
import palettable
from runs_name_path import ApJ_long_paper_runs

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

font = {'family': 'serif',
        #'color'  : 'darkred',
        'color': 'black',
        'weight': 'normal',
        'size': 24,
        }


def accumulated_particle_info(ene, f):
    """
    Get the accumulated particle number and total energy from
    the distribution function.

    Args:
        ene: the energy bins array.
        f: the energy distribution array.
    Returns:
        nacc_ene: the accumulated particle number with energy.
        eacc_ene: the accumulated particle total energy with energy.
    """
    nbins, = f.shape
    dlogE = (math.log10(max(ene))-math.log10(min(ene))) / nbins
    nacc_ene = np.zeros(nbins)
    eacc_ene = np.zeros(nbins)
    nacc_ene[0] = f[0] * ene[0]
    eacc_ene[0] = 0.5 * f[0] * ene[0]**2
    for i in range(1, nbins):
        nacc_ene[i] = f[i] * (ene[i]+ene[i-1]) * 0.5 + nacc_ene[i-1]
        eacc_ene[i] = 0.5 * f[i] * (ene[i]-ene[i-1]) * (ene[i]+ene[i-1])
        eacc_ene[i] += eacc_ene[i-1]
    nacc_ene *= dlogE
    eacc_ene *= dlogE
    return (nacc_ene, eacc_ene)


def get_thermal_total(ene, f, fthermal, fnorm):
    """Get total and thermal particle number and energy.

    Args:
        ene: the energy bins array.
        f: the particle energy distribution array.
        fthermal: thermal part of the particle distribution.
        fnorm: normalization value for f.

    Returns:
        nthermal: particle number of thermal part.
        ntot: total particle number.
        ethermal: particle kinetic energy of thermal part.
        etot: total particle kinetic energy.
    """
    nacc, eacc = accumulated_particle_info(ene, f)
    ntot = nacc[-1]
    etot = eacc[-1]
    nacc_thermal, eacc_thermal = accumulated_particle_info(ene, fthermal)
    nthermal = nacc_thermal[-1]
    ethermal = eacc_thermal[-1]
    nthermal *= fnorm
    ethermal *= fnorm
    ntot *= fnorm
    etot *= fnorm
    print 'Thermal and total particles: ', nthermal, ntot
    print 'Thermal and total energies: ', ethermal, etot
    print '---------------------------------------------------------------'
    return (nthermal, ntot, ethermal, etot)


def fit_thermal_core(ene, f):
    """Fit to get the thermal core of the particle distribution.

    Fit the thermal core of the particle distribution.
    The thermal core is fitted as a Maxwellian distribution.

    Args:
        ene: the energy bins array.
        f: the particle flux distribution.

    Returns:
        fthermal: thermal part of the particle distribution.
    """
    print 'Fitting to get the thermal core of the particle distribution.'
    estart = 0
    eend = np.argmax(f) + 10  # 10 grids shift for fitting thermal core.
    popt, pcov = curve_fit(fitting_funcs.func_maxwellian,
                           ene[estart:eend], f[estart:eend])
    fthermal = fitting_funcs.func_maxwellian(ene, popt[0], popt[1])
    print 'Energy with maximum flux: ', ene[eend - 10]
    print 'Energy with maximum flux in fitted thermal core: ', 0.5/popt[1]
    print 'Thermal core fitting coefficients: '
    print popt
    print '---------------------------------------------------------------'
    return fthermal


def background_thermal_core(ene, f, vth, mime):
    """Fit background thermal core.

    Fit the background thermal core of the particle distribution. The
    background will be far away from the current sheet, so we don't have to
    consider the drift velocities. The thermal energy is calculated from
    the initial thermal velocity.

    Args:
        ene: the energy bins array.
        f: the particle flux distribution.
        vth: thermal speed.
        mime: mass ratio

    Returns:
        fthermal: thermal part of the particle distribution.
    """
    print('Fitting background thermal core')
    gama = 1.0 / math.sqrt(1.0 - 3.0*vth**2)
    thermalEnergy = (gama - 1) * mime
    fthermal = fitting_funcs.func_maxwellian(ene, 1.0, 1.5/thermalEnergy)
    nanMinIndex = np.nanargmin(f/fthermal)
    tindex = np.argmin(f[:nanMinIndex]/fthermal[:nanMinIndex])
    fthermal *= f[tindex]/fthermal[tindex]
    #fthermal *= f[0]/fthermal[0]
    print('---------------------------------------------------------------')
    return fthermal


def lower_thermal_core(ene, f):
    """Fit the thermal core with lower particle energy.

    Fit the thermal core with lower energy, which is not supposed to be
    in the non-thermal particles.

    Args:
        ene: the energy bins array.
        f: the particle flux distribution, which is the original particle
            distribution subtracted by the background plasma.

    Returns:
        fthermal: thermal part of the particle distribution f.
    """
    print('Fitting lower energy thermal core...')
    estart = 0
    eend = np.argmax(f)
    emin = np.argmin(f[:eend])
    popt, pcov = curve_fit(fitting_funcs.func_maxwellian,
                           ene[estart:emin], f[estart:emin])
    fthermal = fitting_funcs.func_maxwellian(ene, popt[0], popt[1])
    fthermal[:emin] += f[:emin] - fthermal[:emin]
    fthermal[emin:] = 0.0
    print 'Lower thermal core fitting coefficients: '
    print popt
    print('---------------------------------------------------------------')
    return fthermal


def fit_nonthermal_power_law(ene, f, fthermal, species, eshift, erange):
    """Power-law fitting for nonthermal particles.

    Using a linear function to fit for reducing fitting error.
    If f = b * x^a, log(f) = log(b) + a*log(x)

    Args:
        ene: the energy bins array.
        f: the particle flux array.
        fthermal: thermal part of the particle distribution.
        species: particle species. 'e' for electron, 'h' for ion.
        eshift: the shift from the maximum of the nonthermal distribution.
        erange: the energy bins of the part for fitting.

    Returns:
        fpowerlaw: the power-law fitting of the non-thermal part of the
            particle distribution.
        e_start, e_end: the starting and ending energy bin index for fitting.
        popt: the fitting parameters.
    """
    fnonthermal = f - fthermal
    estart = np.argmax(fnonthermal) + eshift
    eend = estart + erange
    popt, pcov = curve_fit(fitting_funcs.func_line,
                           np.log10(ene[estart:eend]),
                           np.log10(fnonthermal[estart:eend]))
    print 'Starting and ending energies for fitting: ', ene[estart], ene[eend]
    print '---------------------------------------------------------------'
    fpowerlaw = fitting_funcs.func_line(np.log10(ene), popt[0], popt[1])
    fpowerlaw = np.power(10, fpowerlaw)
    return (fpowerlaw, estart, eend, popt)


def fit_powerlaw_whole(ene, f, species):
    """Power-law fitting for the high energy part of the whole spectrum.

    Args:
        ene: the energy bins array.
        f: the particle flux array.
        species: particle species. 'e' for electron, 'h' for ion.

    Returns:
        fpower: the power-law fitting of the non-thermal part of the
            particle distribution.
    """
    estart = np.argmax(f) + 50
    print "Energy bin index with maximum flux: ", np.argmax(f)
    if (species == 'e'):
        power_range = 90   # for electrons
    else:
        power_range = 130  # for ions
    eend = estart + power_range
    popt, pcov = curve_fit(fitting_funcs.func_line,
                           np.log10(ene[estart:eend]),
                           np.log10(f[estart:eend]))
    print 'Starting and ending energies for fitting: ', ene[estart], ene[eend]
    print 'Power-law fitting coefficients for all particles: '
    print popt
    print '---------------------------------------------------------------'
    fpower = fitting_funcs.func_line(np.log10(ene), popt[0], popt[1])
    fpower = np.power(10, fpower)
    npower, epower = accumulated_particle_info(ene[estart:eend],
                                               fpower[estart:eend])
    ntot, etot = accumulated_particle_info(ene, f)
    nportion = npower[-1] / ntot[-1]
    eportion = epower[-1] / etot[-1]
    return (fpower, estart, eend, popt, nportion, eportion)


def plot_spectrum(it, species, pic_info, ax, is_power, is_thermal):
    """Plotting the energy spectrum.
    Args:
        it: the time point index.
        species: particle species. 'e' for electron, 'h' for ion.
        pic_info: namedtuple for the PIC simulation information.
        ax: axes object.
        is_power: whether to plot power-law spectrum fitting.
        is_thermal: whether to plot the thermal core.
    """
    # Get particle spectra energy bins and flux
    fname = "../spectrum/spectrum-" + species + "." + str(it).zfill(len(str(it)))
    nx = pic_info.nx
    ny = pic_info.ny
    nz = pic_info.nz
    nppc = pic_info.nppc
    if species == 'e':
        ptl_mass = 1.0
    else:
        ptl_mass = pic_info.mime
    fnorm = nx * ny * nz * nppc * ptl_mass
    if (os.path.isfile(fname)):
        ene_lin, flin, ene_log, flog = get_energy_distribution(fname, fnorm)
    else:
        print "ERROR: the spectrum data file doesn't exist."
        return
    ene_log_norm = get_normalized_energy(species, ene_log, pic_info)
    # The whole the energy spectrum.
    p1, = ax.loglog(ene_log_norm, flog, linewidth=2)

    color = p1.get_color()
    # Fit the thermal core and plot thermal distribution.
    fthermal = fit_thermal_core(ene_log, flog)
    fnonthermal = flog - fthermal
    if is_thermal:
        p2, = ax.loglog(ene_log_norm, fthermal, linewidth=2,
                        color='k', linestyle='--', label='Thermal')
    if is_power:
        # Fit the high energy part as a power-law distribution.
        fpower_whole, estart, eend, popt, nportion, \
            eportion = fit_powerlaw_whole(ene_log, flog, species)
        plot_powerlaw_whole(ene_log_norm, fpower_whole, estart,
                            eend, popt, color)
    if (species == 'e'):
        ax.set_xlim([5E-2, 2E2])
        ax.set_ylim([1E-5, 2E2])
    else:
        ax.set_xlim([np.min(ene_log_norm), 1E3])
        ax.set_ylim([1E-5, 2E2])


def plot_powerlaw_whole(ene, fpower_whole, es, ee, popt, color):
    """Plot power-law fitted spectrum for the overall spectrum.

    Args:
        ene: the energy bins array.
        f: the particle flux array.
        fpower_whole: the fitted power-law spectrum.
        es, ee: the starting and ending energy bin index for fitting.
        popt: the fitting parameters.
        color: color to plot the line.

    """
    powerIndex = "{%0.2f}" % popt[0]
    # powerIndex = str(-1)
    pname = '$\sim E^{' + powerIndex + '}$'
    shift = 40
    p1, = plt.loglog(ene[es-shift:ee+shift+1],
                     fpower_whole[es-shift:ee+shift+1]*4, linewidth=2,
                     linestyle='--', color=color, label=pname)
    plt.text(30, 8, pname, color=color, rotation=-0, fontsize=20)


def get_normalized_energy(species, ene_bins, pic_info):
    """Normalize the energies to the initial thermal energy

    Args:
        species: particle species. 'e' for electron, 'h' for ion.
        ene_bins: the energy bins.
    """
    if (species == 'e'):
        vth = pic_info.vthe
    else:
        vth = pic_info.vthi
    gama = 1.0 / math.sqrt(1.0 - 3*vth**2)
    eth = gama - 1.0
    ene_bins_norm = ene_bins / eth
    return ene_bins_norm


def get_energy_distribution(fname, fnorm):
    """ Get energy bins and corresponding particle flux.

    Get linear and logarithm energy bins and particle flux.

    Args:
        fname: file name.
        fnorm: normalization for the distribution.

    Returns:
        ene_lin: linear scale of energy bins.
        ene_log: logarithm scale of energy bins.
        flin: particle flux corresponding to ene_lin.
        flog: particle flux corresponding to ene_log.
    """
    data = read_spectrum_data(fname)
    ene_lin = data[:, 0]  # Linear scale energy bins
    flin = data[:, 1]     # Flux using linear energy bins
    print 'Total number of particles: ', sum(flin)  # Total number of electrons
    print 'Normalization of the energy distribution: ', fnorm

    ene_log = data[:, 2]  # Logarithm scale energy bins
    flog = data[:, 3]     # Flux using Logarithm scale bins
    flog /= fnorm         # Normalized by the maximum value.
    return (ene_lin, flin, ene_log, flog)


def plot_spectrum_series(ntp, species, pic_info):
    """Plot a series of energy spectra.

    Args:
        ntp: total number of time frames.
        species: particle species. 'e' for electron, 'h' for ion.
        pic_info: namedtuple for the PIC simulation information.
    """
    fig, ax = plt.subplots(figsize=[7, 5])
    for current_time in range(1, ntp-1, 2):
        plot_spectrum(current_time, species, pic_info, ax, False, False)
    plot_spectrum(ntp, species, pic_info, ax, True, False)

    if (species == 'e'):
        vth = pic_info.vthe
    else:
        vth = pic_info.vthi
    gama = 1.0 / math.sqrt(1.0 - 3*vth**2)
    eth = gama - 1.0
    fname = "../spectrum/whole/spectrum-" + species + \
            "." + str(1).zfill(len(str(1)))
    nx = pic_info.nx
    ny = pic_info.ny
    nz = pic_info.nz
    nppc = pic_info.nppc
    fnorm = nx * ny * nz * nppc
    ene_lin, flin, ene_log, flog = get_energy_distribution(fname, fnorm)
    ene_log_norm = get_normalized_energy(species, ene_log, pic_info)

    f_intial = fitting_funcs.func_maxwellian(ene_log, fnorm, 1.5/eth)
    nacc_ene, eacc_ene = accumulated_particle_info(ene_log, f_intial)
    p41, = ax.loglog(ene_log_norm, f_intial/nacc_ene[-1], linewidth=2, 
            color='k', linestyle='--', label=r'Initial')
    ax.set_xlabel('$E/E_{th}$', fontdict=font)
    ax.set_ylabel('$f(E)/N_0$', fontdict=font)
    ax.tick_params(labelsize=20)
    plt.tight_layout()
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    fig.savefig('../img/spect_time.eps')
    plt.show()


def plot_spectrum_bulk(ntp, species, pic_info):
    """Plot a series of energy spectra at bulk energy decay time.

    Args:
        ntp: total number of time frames.
        species: particle species. 'e' for electron, 'h' for ion.
        pic_info: namedtuple for the PIC simulation information.
    """
    fig, ax = plt.subplots(figsize=[7, 5])
    for current_time in range(1, ntp-1, 2):
        plot_spectrum(current_time, species, pic_info, ax, False, False)
    plot_spectrum(ntp, species, pic_info, ax, True, False)

    if (species == 'e'):
        vth = pic_info.vthe
    else:
        vth = pic_info.vthi
    gama = 1.0 / math.sqrt(1.0 - 3*vth**2)
    eth = gama - 1.0
    fname = "../spectrum/whole/spectrum-" + species + \
            "." + str(1).zfill(len(str(1)))
    nx = pic_info.nx
    ny = pic_info.ny
    nz = pic_info.nz
    nppc = pic_info.nppc
    fnorm = nx * ny * nz * nppc
    ene_lin, flin, ene_log, flog = get_energy_distribution(fname, fnorm)
    ene_log_norm = get_normalized_energy(species, ene_log, pic_info)

    f_intial = fitting_funcs.func_maxwellian(ene_log, fnorm, 1.5/eth)
    nacc_ene, eacc_ene = accumulated_particle_info(ene_log, f_intial)
    p41, = ax.loglog(ene_log_norm, f_intial/nacc_ene[-1], linewidth=2, 
            color='k', linestyle='--', label=r'Initial')
    ax.set_xlabel('$E/E_{th}$', fontdict=font)
    ax.set_ylabel('$f(E)/N_0$', fontdict=font)
    ax.tick_params(labelsize=20)
    plt.tight_layout()
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    fig.savefig('../img/spect_time.eps')
    plt.show()


def plot_spectrum_bulk(ntp, species, pic_info):
    """Plot a series of energy spectra at bulk energy decay time.

    Args:
        ntp: total number of time frames.
        species: particle species. 'e' for electron, 'h' for ion.
        pic_info: namedtuple for the PIC simulation information.
    """
    fig, ax = plt.subplots(figsize=[7, 5])
    # ax.set_color_cycle(palettable.colorbrewer.qualitative.Accent_6.mpl_colors)
    # colors = palettable.colorbrewer.qualitative.Accent_6.mpl_colors
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    ax.set_color_cycle(colors)
    dtp = int(pic_info.dt_particles) + 1
    i = 0
    for ct in range(2, 8):
        plot_spectrum(ct, species, pic_info, ax, False, False)
        tname = str(dtp * ct) + '$\Omega_{ci}^{-1}$'
        ys = 0.6 - 0.1*i
        ax.text(0.05, ys, tname, color=colors[i], fontsize=20,
                horizontalalignment='left', verticalalignment='center',
                transform = ax.transAxes)
        i += 1


    ax.set_xlabel('$E/E_{th}$', fontdict=font)
    ax.set_ylabel('$f(E)$', fontdict=font)
    ax.tick_params(labelsize=20)
    plt.tight_layout()
    if not os.path.isdir('../img/'):
        os.makedirs('../img/')
    fname = 'spect_time_bulk_' + species + '.eps'
    fig.savefig('../img/' + fname)
    plt.show()


def read_spectrum_data(fname):
    """Read particle energy spectrum data.

    Read particle energy spectrum data at time point it from file.

    Args:
        fname: the file name of the energy spectrum.

    Returns:
        data: the energy bins data and corresponding flux.
            Linear bin + Linear flux + Logarithm bins + Logarithm flux
    """
    try:
        f = open(fname, 'r')
    except IOError:
        print "cannot open ", fname
    else:
        data = np.genfromtxt(f, delimiter='')
        f.close()
        return data

def maximum_energy(ntp, species, pic_info):
    """Plot a series of energy spectra.

    Args:
        ntp: total number of time frames.
        species: particle species. 'e' for electron, 'h' for ion.
        pic_info: namedtuple for the PIC simulation information.
    Return:
        max_ene: the maximum energy at each time step.
    """
    max_ene = np.zeros(ntp)
    for ct in range(1, ntp, 1):
        # Get particle spectra energy bins and flux
        fname = "../spectrum/spectrum-" + species + "." + str(ct).zfill(len(str(ct)))
        nx = pic_info.nx
        ny = pic_info.ny
        nz = pic_info.nz
        nppc = pic_info.nppc
        if species == 'e':
            ptl_mass = 1.0
        else:
            ptl_mass = pic_info.mime
        fnorm = nx * ny * nz * nppc * ptl_mass
        if (os.path.isfile(fname)):
            ene_lin, flin, ene_log, flog = get_energy_distribution(fname, fnorm)
        else:
            print "ERROR: the spectrum data file doesn't exist."
            return
        ene_log_norm = get_normalized_energy(species, ene_log, pic_info)
        max_ene[ct] = ene_log[np.max(np.nonzero(flog))]

    if (species == 'e'):
        vth = pic_info.vthe
    else:
        vth = pic_info.vthi
    gama = 1.0 / math.sqrt(1.0 - 3*vth**2)
    eth = gama - 1.0

    return max_ene/eth


def plot_maximum_energy(ntp, pic_info):
    """Plot a series of energy spectra.

    Args:
        ntp: total number of time frames.
        pic_info: namedtuple for the PIC simulation information.
    """
    max_ene_e = maximum_energy(ntp, 'e', pic_info)
    max_ene_i = maximum_energy(ntp, 'h', pic_info)

    fig = plt.figure(figsize=[7, 5])
    width = 0.69
    height = 0.8
    xs = 0.16
    ys = 0.95 - height
    ax = fig.add_axes([xs, ys, width, height])
    tparticles = pic_info.tparticles
    p1 = ax.plot(tparticles, max_ene_e, color='r', linewidth=2)
    ax.set_xlabel('$t\Omega_{ci}$', fontdict=font)
    ax.set_ylabel('$E_{maxe}/E_{the}$', fontdict=font, color='r')
    for tl in ax.get_yticklabels():
        tl.set_color('r')
    ax.tick_params(labelsize=20)
    ax1 = ax.twinx()
    p2 = ax1.plot(tparticles, max_ene_i, color='b', linewidth=2)
    ax1.tick_params(labelsize=20)
    ax1.set_ylabel('$E_{maxi}/E_{thi}$', fontdict=font, color='b')
    for tl in ax1.get_yticklabels():
        tl.set_color('b')
    plt.show()


def move_energy_spectra():
    if not os.path.isdir('../data/'):
        os.makedirs('../data/')
    dir = '../data/spectra/'
    if not os.path.isdir(dir):
        os.makedirs(dir)
    base_dirs, run_names = ApJ_long_paper_runs()
    for base_dir, run_name in zip(base_dirs, run_names):
        fpath = dir + run_name
        if not os.path.isdir(fpath):
            os.makedirs(fpath)
        command = "cp " + base_dir + "/pic_analysis/spectrum/* " + fpath
        os.system(command)


if __name__ == "__main__":
    # pic_info = pic_information.get_pic_info('../../')
    # ntp = pic_info.ntp
    # vthe = pic_info.vthe
    # plot_spectrum_series(ntp, 'e', pic_info)
    # plot_spectrum_bulk(ntp, 'e', pic_info)
    # plot_maximum_energy(ntp, pic_info)
    move_energy_spectra()
