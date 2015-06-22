"""
Analysis procedures for particle energy spectrum.
"""
import h5py
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit
import os.path
import struct
# Optionally set font to Computer Modern to avoid common missing font errors
#mpl.rc('font', family='serif', serif='cm10')
#mpl.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})

mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

font = {'family' : 'serif',
        #'color'  : 'darkred',
        'color'  : 'black',
        'weight' : 'normal',
        'size'   : 24,
        }

def func(x, a, b):
    """
    Function for fitting with exponential expression.
    """
    return a*np.sqrt(x)*np.exp(-b*x)

def funcLine(x, a, b):
    """
    Function for fitting with power-law expression.
    Both x and y are given by log values. That's why a linear expression
    is given.
    """
    return a * x + b

def funcPower(x, a, b):
    """
    Function for fitting with power-law expression.
    """
    return b * np.power(x, -a)

def funcFull(x, c1, c2, c3, c4, c5, c6, c7):
    """
    Function for fitting with a thermal core + a power law with
    exponential cutoff.
    f = c_1\sqrt{x}\exp{-c_2x} + c_3 x^{-c_4}min[1, \exp{-(x-c_6)/c_7}].
    c_3 is going to be zero if x < c_5.
    """
    thermalCore = c1 * np.sqrt(x) * np.exp(-c2*x)
    a = map(lambda y: 0 if y < c5 else 1, x)
    b = map(lambda y: 0 if y < c6 else 1, x)
    #b1 = map(lambda y: 1 - y, b)
    a = np.array(a)
    b = np.array(b)
    b1 = 1.0 - b
    #powerLaw = c3 * a * np.power(x, -c4) * (b1 + b * np.exp(-c7*(x-c6)))
    powerLaw = 0.001*a * np.power(x, -c4) * b1
    return thermalCore + powerLaw

def funcFullExp(x, c1, c2, c3, c4, c5, c6, c7):
    """
    Function for fitting with a thermal core + a power law with
    exponential cutoff. x and f are log scale.
    f = c_1\sqrt{x}\exp{-c_2x} + c_3 x^{-c_4}min[1, \exp{-(x-c_6)/c_7}].
    c_3 is going to be zero if x < c_5.
    """
    x = np.power(10, x)
    thermalCore = c1 * np.sqrt(x) * np.exp(-c2*x)
    a = map(lambda y: 0 if y < c5 else 1, x)
    b = map(lambda y: 0 if y < c6 else 1, x)
    #b1 = map(lambda y: 1 - y, b)
    a = np.array(a)
    b = np.array(b)
    b1 = 1.0 - b
    powerLaw = c3 * a * np.power(x, -c4) * (b1 + b * np.exp(-c7*(x-c6)))
    #print thermalCore + powerLaw
    return np.log10(thermalCore + powerLaw)

def funcExp(x, c1, c2):
    """Define an exponential function.
    """
    return c1 * np.exp(x*c2)

def PlotSpectrum(it, species, vthe, mime, isPlotNonThermal):
    """Plotting the energy spectrum.
    Args:
        it: the time point index.
        species: particle species. 'e' for electron, 'h' for ion.
        vthe: electron thermal speed.
        mime: ion and electron mass ratio.
        isPlotNonThermal: boolean for whether to plot non-thermal part.
    """
    # Get particle spectra energy bins and flux
    fname = "../spectrum-" + species +  "." + str(it).zfill(len(str(it)))
    if (os.path.isfile(fname)):
        eneLin, fLin, eneLog, fLog, fnorm = \
                GetEnergyBinsAndSpectra(it, species)
    else:
        return

    # Normalized by thermal energy
    if (species == 'e'):
        vth = vthe
    else:
        vth = vthe / math.sqrt(mime)
    gama = 1.0 / math.sqrt(1.0 - 3*vth**2)
    eth = gama - 1.0
    #gama = 1.0 / math.sqrt(1.0 - vth**2)
    #eth = (gama - 1.0)*1.5
    eneLogNorm = eneLog / eth

    # The original spectrum
    if (species == 'e'):
        fname = 'All electrons'
    else:
        fname = 'All ions'

    fig, ax = plt.subplots(figsize=[7,5])
    if isPlotNonThermal:
        p1, = ax.loglog(eneLogNorm, fLog, linewidth=2, 
                color='b', label=fname)
    else:
        p1, = ax.loglog(eneLogNorm, fLog, linewidth=2, 
                color='k', label=fname)

    es = 320
    ee = 430
    popt, pcov = curve_fit(funcExp, eneLog[es:ee], fLog[es:ee])
    fExp = funcExp(eneLog[es:ee], popt[0], popt[1])
    ax.loglog(eneLogNorm[es:ee], fExp)
    ax.loglog(eneLogNorm[es], fLog[es], 'bo')
    ax.loglog(eneLogNorm[ee], fLog[ee], 'bo')

    # Fit the thermal core and plot thermal and non-thermal part.
    fthermal = FitThermalCore(eneLog, fLog)
    #fthermal = BackgroundThermalCore(eneLog, fLog, vth, mime)

    fNonthermal = fLog - fthermal
    #fLowerThermal = LowerThermalCore(eneLog, fNonthermal)
    #fNonthermal -= fLowerThermal
    if isPlotNonThermal:
        p21, = ax.loglog(eneLogNorm, fthermal, linewidth=1, 
                color='k', linestyle='--', label='Thermal')
    else:
        p21, = ax.loglog(eneLogNorm, fthermal, linewidth=2, 
                color='k', linestyle='--', label='Thermal')

    if isPlotNonThermal:
        p22, = ax.loglog(eneLogNorm, fNonthermal, linewidth=2, 
                color='r', label='Non-thermal')
    #p23, = ax.loglog(eneLog, fLowerThermal, linewidth=1,
    #        color='k', linestyle='--', label = 'Lower thermal')

    # Power-law fitting of the non-thermal part of particle distribution
    if isPlotNonThermal:
        fNonthermalPower, es, ee, popt = \
                FitNonthermalPowerLaw(eneLog, fLog, fthermal, species)
        PlotNonthermalPowerLaw(eneLogNorm, fNonthermalPower, 
                fNonthermal, es, ee, popt)

    # Power-law fitting of the total particle distribution at the
    # non-thermal energy range.
    fOverallPower, es, ee, popt, nPortion, ePortion = \
            FitOverallPowerLaw(eneLog, fLog, species)
#    PlotOverallPowerLaw(eneLogNorm, fLog, fOverallPower, es, ee, popt, 
#            isPlotNonThermal)

#    # Fit the whole particle spectrum
#    fFit = FitWholeSpectrum(eneLog, fLog)
#
#    # Fit the whole particle spectrum using logarithm values
#    fLogFit = FitWholeLogSpectrum(eneLog, fLog)
#
    # Get total and thermal particle number and energy.
    nthermal, ntot, ethermal, etot = \
            GetThermalAndTotal(eneLog, fLog, fthermal, fnorm)

    ax.set_ylim([1E-5, 5E1])
    if (species == 'e'):
        ax.set_xlim([np.min(eneLogNorm), 2E2])
    else:
        ax.set_xlim([np.min(eneLogNorm), 1E3])
    #fig.set_size_inches(6.2, 4.7)

    #ax.set_title('Energy spectrum', fontdict=font)
    #ax.set_xlabel('$\gamma-1$', fontdict=font)
    #ax.set_ylabel('$f(\gamma)/N_0$', fontdict=font)
    ax.set_xlabel('$E/E_{th}$', fontdict=font)
    ax.set_ylabel('$f(E)/N_0$', fontdict=font)

    if species == 'e':
        particleName = 'electrons'
    else:
        particleName = 'ions'
  
#    if not isPlotNonThermal:
#        text1 = 'The power-law ' + particleName + ' \n$\sim$' + \
#                str(int(((nPortion)*100))) + \
#                '% of all ' + particleName + ' \n$\sim$' + \
#                str(int(((ePortion)*100))) + \
#                '% of the kinetic energy'
#    else:
#    text1 = 'Non-thermal ' + particleName + ' \n$\sim$' + \
#            str(int(((1-nthermal/ntot)*100))) + \
#            '% of all ' + particleName + ' \n$\sim$' + \
#            str(int(((1-ethermal/etot)*100))) + \
#            '% of the kinetic energy'
#            #str(50) + \
#            #str(90) + \
#
#    #ax.text(np.min(eneLogNorm)*100, 1E-3, text1, fontsize=16)
#    ax.text(np.min(eneLogNorm)*1.5, 2E-4, text1, fontsize=16)

    ax.legend(loc=3, prop={'size':16}, ncol=1,
            shadow=True, fancybox=True)
    ax.tick_params(labelsize=20)
    plt.tight_layout()
    #plt.grid(True)
    fname = 'espectrum' + str(it) + '-' + species + '.eps'
    #fname = 'espectrum' + str(it) + '-' + species + '-whole.eps'
    fig.savefig(fname)

    plt.show()
    #plt.close()

def GetEnergyBinsAndSpectra(it, species):
    """ Get energy bins and corresponding particle flux.

    Get linear and logarithm energy bins and particle flux.

    Args:
        it: time point index.
        species: particle species. 'e' for electron, 'h' for ion.

    Returns:
        eneLin: linear scale of energy bins.
        eneLog: logarithm scale of energy bins.
        fLin: particle flux corresponding to eneLin.
        fLog: particle flux corresponding to eneLog.
        fnorm: normalization of the particle flux.
    """
    data = ReadSpectrumData(it, species)
    #dimx,dimy = data.shape
    #print dimx, dimy

    eneLin = data[:, 0] # Linear scale energy bins
    fLin = data[:, 1]   # Flux using linear energy bins
    print sum(fLin)  # Total number of electrons

    eneLog = data[:,2]  # Logarithm scale energy bins
    fLog = data[:,3]    # Flux using Logarithm scale bins
    #fnorm = max(fLog)
    fnorm = 4096*2048*400.0
    fLog /= fnorm       # Normalized by the maximum value.
    return (eneLin, fLin, eneLog, fLog, fnorm)

def GetThermalAndTotal(ene, f, fthermal, fnorm):
    """Get total and thermal particle number and energy.

    Args:
        ene: the energy bins array.
        f: the particle flux array.
        fthermal: thermal part of the particle distribution.
        fnorm: normalization value for f.

    Returns:
        nthermal: particle number of thermal part.
        ntot: total particle number.
        ethermal: particle kinetic energy of thermal part.
        etot: total particle kinetic energy.
    """
    nacc, eacc = ParticleNumberEnergy(ene, f)
    ntot = nacc[-1]
    etot = eacc[-1]
    naccThermal, eaccThermal = ParticleNumberEnergy(ene, fthermal)
    nthermal = naccThermal[-1]
    ethermal = eaccThermal[-1]
    nthermal *= fnorm
    ethermal *= fnorm
    ntot *= fnorm
    etot *= fnorm
    print 'Thermal and total particles: ', nthermal, ntot
    print 'Thermal and total energies: ', ethermal, etot
    print '---------------------------------------------------------------'
    return (nthermal, ntot, ethermal, etot)

def FitThermalCore(ene, f):
    """Fit to get the thermal core of the particle distribution.

    Fit the thermal core of the particle distribution.
    The thermal core is fitted as a Maxwellian distribution.

    Args:
        ene: the energy bins array.
        f: the particle flux distribution.

    Returns:
        fthermal: thermal part of the particle distribution.
    """
    estart = 0
    eend = np.argmax(f) + 10 # 10 grids shift for fitting thermal core.
    popt, pcov = curve_fit(func, ene[estart:eend], f[estart:eend])
    fthermal = func(ene, popt[0], popt[1])
    print 'Energy with maximum flux: ', ene[eend - 10]
    print 'Energy with maximum flux in fitted thermal core: ', 0.5/popt[1]
    print 'Thermal core fitting coefficients: '
    print popt
    print '---------------------------------------------------------------'
    return fthermal

def BackgroundThermalCore(ene, f, vth, mime):
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
    print('Fitting background thermal core...')
    gama = 1.0 / math.sqrt(1.0 - 3.0*vth**2)
    thermalEnergy = (gama - 1) * mime
    print 1.5/thermalEnergy
    fthermal = func(ene, 1.0, 1.5/thermalEnergy)
    nanMinIndex = np.nanargmin(f/fthermal)
    tindex = np.argmin(f[:nanMinIndex]/fthermal[:nanMinIndex])
    fthermal *= f[tindex]/fthermal[tindex]
    #fthermal *= f[0]/fthermal[0]
    print('---------------------------------------------------------------')
    return fthermal

def LowerThermalCore(ene, f):
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
    popt, pcov = curve_fit(func, ene[estart:emin], f[estart:emin])
    fthermal = func(ene, popt[0], popt[1])
    fthermal[:emin] += f[:emin] - fthermal[:emin]
    fthermal[emin:] = 0.0
    print 'Lower thermal core fitting coefficients: '
    print popt
    print('---------------------------------------------------------------')
    return fthermal

def FitNonthermalPowerLaw(ene, f, fthermal, species):
    """Power-law fitting for non-thermal particles.

    Args:
        ene: the energy bins array.
        f: the particle flux array.
        fthermal: thermal part of the particle distribution.
        species: particle species. 'e' for electron, 'h' for ion.

    Returns:
        fPower: the power-law fitting of the non-thermal part of the
            particle distribution.
        es, ee: the starting and ending energy bin index for fitting.
        popt: the fitting parameters.
    """
    fNonthermal = f - fthermal
    if (species == 'e'):
        es = np.argmax(fNonthermal) + 20
        rangePower = 50 # for electrons
    else:
        es = np.argmax(fNonthermal) + 10
        rangePower = 80  # for ions
    ee = es + rangePower
    # Using a linear function to fit for reducing fitting error.
    # See if f = b * x^a, log(f) = log(b) + a*log(x)
    popt, pcov = curve_fit(funcLine, np.log10(ene[es:ee]), 
            np.log10(fNonthermal[es:ee]))
    print 'Starting and ending energies for fitting: ', ene[es], ene[ee]
    print 'Power-law fitting coefficients for non-thermal particles: '
    print popt
    print '---------------------------------------------------------------'
    fPower = funcLine(np.log10(ene), popt[0], popt[1])
    fPower = np.power(10, fPower)
    return (fPower, es, ee, popt)

def PlotNonthermalPowerLaw(ene, fPower, fNonthermal, es, ee, popt):
    """Plot power-law fitting of the non-thermal particles.

    Args:
        ene: the energy bins array.
        fPower: the power-law fitting of the non-thermal part of the
            particle distribution.
        fNonthermal: non-thermal part of the particle distribution.
        es, ee: the starting and ending energy bin index for fitting.
        popt: the fitting parameters.
    """
    popt[0] = -1.0
    powerIndex = "{%0.2f}" % popt[0]
    #pname = '$\sim (\gamma-1)^' + powerIndex + '$'
    pname = '$\sim E^' + powerIndex + '$'
    shift = 20
    p1, = plt.loglog(ene[es-shift:ee+1+shift], 
            fPower[es-shift:ee+1+shift]*2, linewidth=2, 
            color='r', linestyle='--', label=pname)
    #p21, = plt.loglog(ene[es], fNonthermal[es], 'ro')
    #p22, = plt.loglog(ene[ee], fNonthermal[ee], 'ro')
    return

def FitOverallPowerLaw(ene, f, species):
    """Power-law fitting for the overall spectrum.

    Args:
        ene: the energy bins array.
        f: the particle flux array.
        species: particle species. 'e' for electron, 'h' for ion.

    Returns:
        fPower: the power-law fitting of the non-thermal part of the
            particle distribution.
    """
    es = np.argmax(f) + 50
    print "Energy bin index with maximum flux: ", np.argmax(f)
    if (species == 'e'):
        rangePower = 90 # for electrons
    else:
        rangePower = 130  # for ions
    ee = es + rangePower
    popt, pcov = curve_fit(funcLine, np.log10(ene[es:ee]), np.log10(f[es:ee]))
    print 'Starting and ending energies for fitting: ', ene[es], ene[ee]
    print 'Power-law fitting coefficients for all particles: '
    print popt
    print '---------------------------------------------------------------'
    fPower = funcLine(np.log10(ene), popt[0], popt[1])
    fPower = np.power(10, fPower)
    nPower, ePower = ParticleNumberEnergy(ene[es:ee], fPower[es:ee])
    ntot, etot = ParticleNumberEnergy(ene, f)
    nPortion = nPower[-1] / ntot[-1]
    ePortion = ePower[-1] / etot[-1]
    return (fPower, es, ee, popt, nPortion, ePortion)

def PlotOverallPowerLaw(ene, f, fPower, es, ee, popt, isPlotNonThermal):
    """Plot power-law fitted spectrum for the overall spectrum.

    Args:
        ene: the energy bins array.
        f: the particle flux array.
        fPower: the fitted power-law spectrum for the overall spectrum.
        es, ee: the starting and ending energy bin index for fitting.
        popt: the fitting parameters.
        isPlotNonThermal: boolean for whether to plot non-thermal part.

    """
    powerIndex = "{%0.2f}" % popt[0]
    #pname = '$\sim (\gamma-1)^' + powerIndex + '$'
    pname = '$\sim E^' + powerIndex + '$'
    shift = 40
    if isPlotNonThermal:
        p1, = plt.loglog(ene[es-shift:ee+shift+1], 
                fPower[es-shift:ee+shift+1]*4, linewidth=2, 
                linestyle='--', color='b', label=pname)
        #p21, = plt.loglog(ene[es], f[es], 'bo')
        #p22, = plt.loglog(ene[ee], f[ee], 'bo')
    else:
        p1, = plt.loglog(ene[es-shift:ee+shift+1], 
                fPower[es-shift:ee+shift+1]*2, linewidth=2, 
                linestyle='--', color='r', label=pname)

def FitWholeSpectrum(ene, f):
    """Fit the whole the spectrum.

    Fit the whole the particle spectrum using a function with a thermal core +
    power-law + exponential decay.

    Args:
        ene: the energy bins array.
        f: the particle flux array.

    Returns:
        fFit: the power-law fitting of the non-thermal part of the
            particle distribution.
    """
    nonZeros = f[np.nonzero(f)]
    maxNonZero = np.argwhere(f == nonZeros[-1])
    maxNonZero -= 10
    popt, pcov = curve_fit(funcFull, ene[:maxNonZero], f[:maxNonZero], 
            p0=[115, 2500, 0.005, 0.74, 0.0015, 0.01, 150])
    print popt
    print '---------------------------------------------------------------'
    fFit = funcFull(ene, popt[0], popt[1], popt[2], 
            popt[3], popt[4], popt[5], popt[6])
    p1, = plt.loglog(ene, fFit, linewidth=2, color='k', 
            label='Fitted spectrum')
    dene = math.log10(ene[1]) - math.log10(ene[0])
    ebin1 = math.floor((math.log10(popt[4]) - math.log10(ene[0]))/dene)
    ebin2 = math.floor((math.log10(popt[5]) - math.log10(ene[0]))/dene)
    p21, = plt.loglog(ene[ebin1], f[ebin1], 'ro')
    p22, = plt.loglog(ene[ebin2], f[ebin2], 'ro')
    return fFit

def FitWholeLogSpectrum(ene, f):
    """Fit the whole the spectrum at logarithm scale.

    Fit the whole the particle spectrum using a function with a thermal core +
    power-law + exponential decay. Both the energy bins and particle
    distribution are transformed to logarithm values.

    Args:
        ene: the energy bins array.
        f: the particle flux array.

    Returns:
        fLogFit: the power-law fitting of the non-thermal part of the
            particle distribution.
    """
    nonZeros = f[np.nonzero(f)]
    maxNonZero = np.argwhere(f == nonZeros[-1])
    maxNonZero -= 10
    popt, pcov = curve_fit(funcFullExp, np.log10(ene[:maxNonZero]), 
            np.log10(f[:maxNonZero]), 
            p0=[115, 2500, 0.0005, 1, 0.002, 0.06, 1500])
    print popt
    print '---------------------------------------------------------------'
    fLogFit = funcFullExp(np.log10(ene), popt[0], popt[1], popt[2], 
            popt[3], popt[4], popt[5], popt[6])
    return fLogFit

def AccumulatedParticleNumberAlongEnergy(species):
    """ Accumulate particle number along energy and plot it.

    Args:
        species: particle species.
    """
    for it in range(48, 49):
        data = ReadSpectrumData(it, species)
        ene_lin, fe_lin, ene_log, fe_log, fnorm = EnergyBinsAndFlux(data)
        nacc_ene, eacc_ene = ParticleNumberEnergy(ene_log, fe_log)
        p1, = plt.semilogx(ene_log, nacc_ene / nacc_ene[-1], 
                linewidth = 2, color = 'b', label = 'Particle number $n$')
        p2, = plt.semilogx(ene_log, eacc_ene / eacc_ene[-1],
                linewidth = 2, color = 'g', label = 'Particle energy $E$')
        plt.ylim([-0.05, 1.05])
        plt.xlim([1E-4, 100])

        #plt.title('Energy spectrum', fontdict=font)
        plt.xlabel('$\gamma-1$', fontdict=font)
        plt.ylabel('Accumulated $n$ and $E$', fontdict=font)
       
        plt.legend(loc=4, prop={'size':16})
        plt.tick_params(labelsize=20)
        plt.tight_layout()
        #plt.grid(True)
        plt.savefig('n_ene_acc.eps')
    
        plt.show()

def EvolutionOfDifferentialNumberAndEnergy(nt, species, vthe, pmass):
    """Plot the evolution of the particle number and energy in each energy bins

    Args:
        nt: total number of time frames.
        species: particle species.
        vthe: electron thermal velocity.
        pmass: particle mass
    """
    eneLogNorm, ndiff_ene, ediff_ene = \
            DifferentialParticleNumberAlongEnergy(1, species, vthe, pmass)
    nbins, = eneLogNorm.shape
    ndiff_time = np.zeros((nbins, nt))
    ediff_time = np.zeros((nbins, nt))
    for it in range(1, nt+1):
        fname = "../spectrum-" + species +  "." + str(it).zfill(len(str(it)))
        if (os.path.isfile(fname)):
            eneLogNorm, ndiff_ene, ediff_ene = \
                    DifferentialParticleNumberAlongEnergy(it, species, vthe, pmass)
        else:
            continue
        ndiff_time[:, it-1] = ndiff_ene
        ediff_time[:, it-1] = ediff_ene

    t = np.arange(nt)
    fig, ax = plt.subplots()
    p1 = ax.imshow(np.log10(ediff_time), cmap=plt.cm.hsv,
            extent=[0, 1, 0, 1], origin='lower')
    emax = np.max(eneLogNorm)
    emin = np.min(eneLogNorm)
    emax_log = math.log10(emax)
    emin_log = math.log10(emin)
    emax_log_int = math.floor(emax_log)
    emin_log_int = math.ceil(emin_log)
    lenx = emax_log - emin_log
    xs = (emin_log_int - emin_log) / lenx
    xe = (emax_log_int - emin_log) / lenx
    xticks = np.linspace(xs, xe, 6)
    xtick_labels = np.power(10, xticks * lenx + emin_log)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    plt.show()

def DifferentialParticleNumberAlongEnergy(it, species, vthe, pmass):
    """Particle number and energy in each energy bins.

    Args:
        it: the time frame index.
        species: particle species.
        vthe: electron thermal velocity.
        pmass: particle mass
    """
    data = ReadSpectrumData(it, species)
    ene_lin, fe_lin, ene_log, fe_log, fnorm = EnergyBinsAndFlux(data)
    # Normalized by thermal energy
    vth = vthe / math.sqrt(pmass)
    gama = 1.0 / math.sqrt(1.0 - 3*vth**2)
    eth = gama - 1.0
    eneLogNorm = ene_log / eth

    nacc_ene, eacc_ene = ParticleNumberEnergy(ene_log, fe_log)
    dimx, = ene_log.shape
    ndiff_ene = np.zeros(dimx)
    ediff_ene = np.zeros(dimx)
    ndiff_ene[1:-1] = nacc_ene[1:-1] - nacc_ene[0:-2]
    ediff_ene[1:-1] = eacc_ene[1:-1] - eacc_ene[0:-2]
    return (eneLogNorm, ndiff_ene, ediff_ene)

def PlotDifferentialNumberAndEnergy(it, species, vthe, pmass):
    """Plot the particle number and energy in each energy bin.

    Args:
        species: particle species.
        ene: the energy bins with logarithm scale.
        ndiff, ediff: particle number and energy in each energy bin.
    """
    fname = "../spectrum-" + species +  "." + str(it).zfill(len(str(it)))
    if (os.path.isfile(fname)):
        ene, ndiff, ediff = \
                DifferentialParticleNumberAlongEnergy(it, species, vthe, pmass)
    else:
        return
    ndiff_max = np.max(ndiff)
    ediff_max = np.max(ediff)
    fig, ax = plt.subplots()
    p1, = ax.loglog(ene, ndiff / ndiff_max, 
            linewidth = 2, color = 'b', label = 'Number $n$')
    p2, = ax.loglog(ene, ediff / ediff_max,
            linewidth = 2, color = 'g', label = 'Energy $E$')
    ax.set_ylim([1.0E-7, 2])
    if (species == 'e'):
        ax.set_xlim([np.min(ene), 4E2])
    else:
        ax.set_xlim([np.min(ene), 2E3])

    #plt.title('Energy spectrum', fontdict=font)
    ax.set_xlabel('$E/E_{th}$', fontdict=font)
    ax.set_ylabel('$n$ and $E$ in each bins', fontdict=font)
   
    plt.legend(loc=0, prop={'size':16})
    plt.tick_params(labelsize=20)
    plt.tight_layout()
    #plt.grid(True)
    fname = 'n_ene_diff' + str(it) + '_' + species + '.eps'
    plt.savefig(fname)

    plt.show()

def ParticleNumberEnergy(ene, f):
    """
    Get the particle number and total energy from distribution
    function.
    """
    nbins, = f.shape
    dlogE = (math.log10(max(ene))-math.log10(min(ene))) / nbins
    nacc_ene = np.zeros(nbins) # Accumulated particle number along energy
    eacc_ene = np.zeros(nbins) # Accumulated particle energy along energy
    nacc_ene[0] = f[0] * ene[0]
    eacc_ene[0] = 0.5 * f[0] * ene[0]**2
    for i in range(1, nbins):
        nacc_ene[i] = f[i] * (ene[i]+ene[i-1]) * 0.5 + nacc_ene[i-1]
        eacc_ene[i] = 0.5 * f[i] * (ene[i]-ene[i-1]) * (ene[i]+ene[i-1])
        eacc_ene[i] += eacc_ene[i-1]
    nacc_ene *= dlogE
    eacc_ene *= dlogE
    return (nacc_ene, eacc_ene)

def EnergyBinsAndFlux(data):
    """
    Get the linear and logarithm energy bins and corresponding flux.
    """
    eneLin = data[:,0]
    fLin = data[:,1]
    eneLog = data[:,2]
    fLog = data[:,3]
    fnorm = max(fLog)
    fLog /= fnorm # Normalized by the maximum flux
    return (eneLin, fLin, eneLog, fLog, fnorm)

def ParticleNumberEnergyEvolution(species, vthe, mime):
    """Check the evolution of particle numbers and energies.

    Check the evolution of particle numbers and energies in thermal
    part and non-thermal part.

    Args:
        species: particle species. 'e' for electron, 'h' for ion.
        vthe: electron thermal speed.
        mime: ion and electron mass ratio.
    """
    nt = 48
    dtwci = 4.833738e-03 
    interval = int(2.5/dtwci)
    dt = 10 * interval * dtwci
    t = np.arange(nt) * dt
    tmax = max(t)
    tmin = min(t)
    ntot = np.zeros(nt)
    etot = np.zeros(nt)
    nNonthermal = np.zeros(nt)
    eNonthermal = np.zeros(nt)
    nthermal = np.zeros(nt)
    ethermal = np.zeros(nt)
    nNonthermalDiff = np.zeros(nt)
    eNonthermalDiff = np.zeros(nt)
    if (species == 'e'):
        vth = vthe
    else:
        vth = vthe / math.sqrt(mime)
    for it in range(1, nt + 1):
        fname = "../spectrum-" + species +  "." + str(it).zfill(len(str(it)))
        if (os.path.isfile(fname)):
            data = ReadSpectrumData(it, species)
        else:
            data1 = ReadSpectrumData(it+1, species)
            data2 = ReadSpectrumData(it-1, species)
            data = (data1 + data2) / 2

        eneLin, fLin, eneLog, fLog, fnorm = EnergyBinsAndFlux(data)
        fthermal = FitThermalCore(eneLog, fLog)
        #fthermal = BackgroundThermalCore(eneLog, fLog, vth)
        fNonthermal = fLog - fthermal
        nacc_ene, eacc_ene = ParticleNumberEnergy(eneLog, fLog)
        ntot[it-1] = nacc_ene[-1] * fnorm
        etot[it-1] = eacc_ene[-1] * fnorm
        nacc_ene, eacc_ene = ParticleNumberEnergy(eneLog, fNonthermal)
        nNonthermal[it-1] = nacc_ene[-1] * fnorm
        eNonthermal[it-1] = eacc_ene[-1] * fnorm
        nacc_ene, eacc_ene = ParticleNumberEnergy(eneLog, fthermal)
        nthermal[it-1] = nacc_ene[-1] * fnorm
        ethermal[it-1] = eacc_ene[-1] * fnorm
        #print ntot*fnorm, sum(fLin)
    for it in range(1, nt):
        nNonthermalDiff[it] = nNonthermal[it] - nNonthermal[it-1]
        eNonthermalDiff[it] = eNonthermal[it] - eNonthermal[it-1]
#    #ax1 = plt.subplot(211)
#    f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=False)
#    p1, = ax1.plot(t, nNonthermal / nNonthermal[-1], 
#            linewidth = 2, color = 'b', 
#            label = 'Non-thermal particle number $n$')
#    p2, = ax1.plot(t, eNonthermal / eNonthermal[-1],
#            linewidth = 2, color = 'g', 
#            label = 'Non-thermal particle energy $E$')
#    ax1.set_xlim([tmin, tmax])
#    ax1.set_ylim([-0.05, 1.05])
#    ax1.tick_params(labelsize=20)
#    ax1.set_ylabel('Accumulated', fontdict=font)
#    p1, = ax2.plot(t, nNonthermalDiff / nNonthermalDiff[-1], 
#            linewidth = 2, color = 'b', 
#            label = 'Non-thermal particle number $n$')
#    p2, = ax2.plot(t, eNonthermalDiff / eNonthermalDiff[-1],
#            linewidth = 2, color = 'g', 
#            label = 'Non-thermal particle energy $E$')
#    ax2.set_xlim([tmin, tmax])
#    ax2.tick_params(labelsize=20)
#    ax2.set_xlabel('$t\omega_{ci}$', fontdict=font)
#    ax2.set_ylabel('Differential', fontdict=font)
#
#    p3, = ax2.plot([tmin, tmax], [0, 0], 'k--')
#
#    f.subplots_adjust(hspace=0)
#    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
#    ax1.legend(loc=4, prop={'size':16})
#    f.tight_layout()
#    #plt.grid(True)
#    fname = 'n_ene_t' + species + '.eps'
#    plt.savefig(fname)
#    
#    plt.show()

    nNonThermalPortion = 1.0 - nthermal / ntot
    eNonThermalPortion = 1.0 - ethermal / etot
    nNonThermalPortionDiff = np.zeros(nt)
    eNonThermalPortionDiff = np.zeros(nt)
    for i in range(1, nt):
        nNonThermalPortionDiff[i] = \
                nNonThermalPortion[i] - nNonThermalPortion[i-1]
        eNonThermalPortionDiff[i] = \
                eNonThermalPortion[i] - eNonThermalPortion[i-1]

    #f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=False)
    f, ax1 = plt.subplots()
    nNonThermalPortion[0] = 0
    eNonThermalPortion[0] = 0
    p1, = ax1.plot(t, nNonThermalPortion, 
            linewidth = 2, color = 'b', 
            label = r'$n_{nth} / n_{tot}$')
    p2, = ax1.plot(t, eNonThermalPortion, 
            linewidth = 2, color = 'g', 
            label = r'$E_{nth} / E_{tot}$')
    ax1.set_xlim([tmin, tmax])
    ax1.set_ylim([-0.05, 1.05])
    ax1.tick_params(labelsize=20)
    ax1.set_xlabel(r'$t\Omega_{ci}$', fontdict=font)
    ax1.set_ylabel(r'Non-thermal portion', fontdict=font)

    f = open('nPortion_beta001_1.dat', 'w')
    for i in range(nt):
        f.write(str(t[i]) + ' ')
        f.write(str(nNonThermalPortion[i]) + ' ')
        f.write(str(eNonThermalPortion[i]))
        f.write('\n')
    f.close()
#    p1, = ax2.plot(t, nNonThermalPortionDiff, 
#            linewidth = 2, color = 'b', 
#            label = 'Non-thermal particle number $n$')
#    p2, = ax2.plot(t, eNonThermalPortionDiff,
#            linewidth = 2, color = 'g', 
#            label = 'Non-thermal particle energy $E$')
#    ax2.set_xlim([tmin, tmax])
#    ax2.tick_params(labelsize=20)
#    ax2.set_xlabel(r'$t\omega_{ci}$', fontdict=font)
#    ax2.set_ylabel(r'$dP_{non-thermal}/dt$', fontdict=font)
#
#    p3, = ax2.plot([tmin, tmax], [0, 0], 'k--')
#
#    f.subplots_adjust(hspace=0)
#    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
    ax1.legend(loc=4, prop={'size':24}, ncol=2,
            shadow=True, fancybox=True)
    plt.tight_layout()
    #plt.grid(True)
    fname = 'n_ene_portion_' + species + '.eps'
    plt.savefig(fname)
    
    plt.show()

def ReadSpectrumData(it, species):
    """Read particle energy spectrum data.

    Read particle energy spectrum data at time point it from
    file.

    Args:
        it: the time point index.
        species: particle species. 'e' for electron, 'h' for ion.

    Returns:
        data: the energy bins data and corresponding flux.
            Linear bin + Linear flux + Logarithm bins + Logarithm flux
    """
    fname = "../spectrum-" + species +  "." + str(it).zfill(len(str(it)))
    #fname = "../spectrum-zone1-" + species +  "." + str(it).zfill(len(str(it)))
    try:
        f = open(fname, 'r')
    except IOError:
        print "cannot open ", fname
    else:
        data = np.genfromtxt(f, delimiter='')
        f.close()
        return data

def PlotEnergyEvolution(ax):
    """Plot energy evolution.

    Plot time evolution of magnetic, electric, electron and ion kinetic
    energies. The layout of the read data is
    time, electric energy, magnetic energy, ion kinetic energy,
    electron kinetic energy, 3 components of magnetic energy.

    Args:
        ax: one axes object.
    """
    f = open('ene_pic.dat', 'r')
    data = np.genfromtxt(f, delimiter='')
    f.close()

    t = data[:,0]
    eneE = data[:,1]
    eneB = data[:,2]
    Ki = data[:,3]
    Ke = data[:,4]
    eneBx = data[:,5]
    eneBy = data[:,6]
    eneBz = data[:,7]

    normE = eneBx[0]

    #fig, ax = plt.subplots()
    p21, = ax.plot(t, eneBx/normE, linewidth=2, 
            label=r'$B_x^2(t)$', color='b')
#    p2, = plt.plot(t, eneB/normE, \
#            linewidth=2, label='$B^2$')
    p3, = ax.plot(t, Ki/normE, linewidth=2, 
            color='g', label=r'$\Delta K_i$')
    p4, = ax.plot(t, Ke/normE, linewidth=2, 
            color='r', label=r'$\Delta K_e$')
    p1, = ax.plot(t, 100*eneE/normE, linewidth=2, 
            color='m', label='$100E^2$')
#    p22, = ax.plot(t, (eneBy-eneBy[0])/normE, linewidth=2, 
#            linestyle='--', label=r'$\Delta B_y^2$')
#    p23, = ax.plot(t, (eneBz-eneBz[0])/normE, linewidth=2, 
#            linestyle='-.', label=r'$\Delta B_z^2$')
    ax.set_xlim([0, 1190])
    ax.set_ylim([0, 1.05])

    #fig.set_size_inches(6.2, 4.7)

    #plt.title('Energy spectrum', fontdict=font)
    ax.set_xlabel(r'$t\Omega_{ci}$', fontdict=font, fontsize=20)
    ax.set_ylabel(r'Energy/$\varepsilon_{bx}(0)$', fontdict=font, fontsize=20)

    ax.text(500, 0.85, r'$\varepsilon_{bx}(t)$', color='blue', fontsize=24)
    ax.text(500, 0.65, r'$100\varepsilon_e$', color='m', fontsize=24)
    ax.text(900, 0.85, r'$K_e$', color='red', fontsize=24)
    ax.text(900, 0.65, r'$K_i$', color='green', fontsize=24)
   
    #plt.legend(loc=1, prop={'size':12}, ncol=2, framealpha=0.0)
    #        #shadow=True, fancybox=True)
    plt.tick_params(labelsize=16)
    #plt.tight_layout()
    #plt.grid(True)
    #plt.savefig('pic_ene.eps')

    print eneBx[-1]/normE, (Ki[-1]-Ki[0])/normE, (Ke[-1]-Ke[0])/normE 
    print Ki[-1] / Ki[0], Ke[-1] / Ke[0], Ki[0]/normE, Ke[0]/normE
    
    #plt.show()

def ReadJDriftsDoteData(species):
    """Read j.E data.

    Args:
        species: particle species. 'e' for electron, 'h' for ion.
    """
    fname = "jdote_drifts_" + species + ".dat"
    f = open(fname, 'r')
    data = np.genfromtxt(f, delimiter='')
    f.close()
    tf = data[:, 0]
    jcpara_dote = data[:, 1]
    jcperp_dote = data[:, 2]
    jmag_dote   = data[:, 3]
    jgrad_dote  = data[:, 4]
    jdiagm_dote = data[:, 5]
    jpolar_dote = data[:, 6]
    jexb_dote   = data[:, 7]
    jpara_dote  = data[:, 8]
    jperp_dote  = data[:, 9]
    jperp1_dote = data[:, 10]
    jperp2_dote = data[:, 11]
    jqnupara_dote = data[:, 12]
    jqnuperp_dote = data[:, 13]
    jagy_dote     = data[:, 14]
    jtot_dote     = data[:, 15]
    jcpara_dote_int = data[:, 16]
    jcperp_dote_int = data[:, 17]
    jmag_dote_int   = data[:, 18]
    jgrad_dote_int  = data[:, 19]
    jdiagm_dote_int = data[:, 20]
    jpolar_dote_int = data[:, 21]
    jexb_dote_int   = data[:, 22]
    jpara_dote_int  = data[:, 23]
    jperp_dote_int  = data[:, 24]
    jperp1_dote_int = data[:, 25]
    jperp2_dote_int = data[:, 26]
    jqnupara_dote_int = data[:, 27]
    jqnuperp_dote_int = data[:, 28]
    jagy_dote_int     = data[:, 29]
    jtot_dote_int     = data[:, 30]
    return (tf, jcpara_dote, jcperp_dote, jmag_dote, jgrad_dote,
            jdiagm_dote, jpolar_dote, jexb_dote, jpara_dote, jperp_dote,
            jperp1_dote, jperp2_dote, jqnupara_dote, jqnuperp_dote,
            jagy_dote, jtot_dote, 
            jcpara_dote_int, jcperp_dote_int, jmag_dote_int, jgrad_dote_int,
            jdiagm_dote_int, jpolar_dote_int, jexb_dote_int, jpara_dote_int, 
            jperp_dote_int, jperp1_dote_int, jperp2_dote_int, 
            jqnupara_dote_int, jqnuperp_dote_int, jagy_dote_int, jtot_dote_int)

def ReadEnergiesPIC():
    f = open("energies_pic.dat", 'r')
    data = np.genfromtxt(f, delimiter='')
    f.close()
    te = data[:, 0]
    dkei = data[:, 1]
    dkee = data[:, 2]
    de_electric = data[:, 3]
    de_magnetic = data[:, 4]
    de_bx = data[:, 5]
    de_by = data[:, 6]
    de_bz = data[:, 7]
    kei = data[:, 8]
    kee = data[:, 9]
    e_electric = data[:, 10]
    e_magnetic = data[:, 11]
    e_bx = data[:, 12]
    e_by = data[:, 13]
    e_bz = data[:, 14]
    return (te, dkei, dkee, de_electric, de_magnetic, de_bx, de_by, de_bz,
            kei, kee, e_electric, e_magnetic, e_bx, e_by, e_bz)

def jDriftsDote(species):
    """Plot the energy conversion due to drifts currents.

    Args:
        species: particle species. 'e' for electron, 'h' for ion.
    """
    tf, jcpara_dote, jcperp_dote, jmag_dote, jgrad_dote, \
    jdiagm_dote, jpolar_dote, jexb_dote, jpara_dote, jperp_dote, \
    jperp1_dote, jperp2_dote, jqnupara_dote, jqnuperp_dote, \
    jagy_dote, jtot_dote, \
    jcpara_dote_int, jcperp_dote_int, jmag_dote_int, jgrad_dote_int, \
    jdiagm_dote_int, jpolar_dote_int, jexb_dote_int, jpara_dote_int, \
    jperp_dote_int, jperp1_dote_int, jperp2_dote_int, \
    jqnupara_dote_int, jqnuperp_dote_int, jagy_dote_int, jtot_dote_int \
    = ReadJDriftsDoteData(species)

    te, dkei, dkee, de_electric, de_magnetic, de_bx, de_by, de_bz, \
    kei, kee, e_electric, e_magnetic, e_bx, e_by, e_bz = \
    ReadEnergiesPIC()

#    jdote_tot_drifts = jcpara_dote + jgrad_dote + jmag_dote + jpolar_dote + \
#            jagy_dote + jqnupara_dote
#    jdote_tot_drifts_int = jcpara_dote_int + jgrad_dote_int + jmag_dote_int + \
#            jpolar_dote_int + jagy_dote_int + jqnupara_dote_int
#    jdote_tot_drifts = jcpara_dote + jgrad_dote + jmag_dote + jpolar_dote + \
#            jqnupara_dote
#    jdote_tot_drifts_int = jcpara_dote_int + jgrad_dote_int + jmag_dote_int + \
#            jpolar_dote_int + jqnupara_dote_int
    jdote_tot_drifts = jcpara_dote + jgrad_dote + jmag_dote + jpolar_dote
    jdote_tot_drifts_int = jcpara_dote_int + jgrad_dote_int + jmag_dote_int + \
            jpolar_dote_int
    if species == 'e':
        dke = dkee
        ke = kee
        kename = '$\Delta K_e$'
    else:
        dke = dkei
        ke = kei
        kename = '$\Delta K_i$'

    #fig, axes = plt.subplots(2, sharex=True, sharey=False)
    fig = plt.figure(figsize=[7, 4])
   
    width = 0.82
    height = 0.4
    xs = 0.14
    ys = 0.15
    #mpl.rc('text', usetex=False)
    ax1 = fig.add_axes([xs, ys+height, width, height])
    ax1.plot(tf, jcpara_dote, lw=2, color='b', 
            label=r'$\mathbf{j}_c\cdot\mathbf{E}$')
    ax1.plot(tf, jgrad_dote, lw=2, color='g',
            label=r'$\mathbf{j}_g\cdot\mathbf{E}$')
    ax1.plot(tf, jmag_dote, lw=2, color='r',
            label=r'$\mathbf{j}_m\cdot\mathbf{E}$')
    #axes[0].plot(tf, jpolar_dote, lw=2, label=r'$\mathbf{j}_p\cdot\mathbf{E}$')
    #ax1.plot(tf, jqnupara_dote, lw=2, 
    #        label=r'$\mathbf{j}_\parallel\cdot\mathbf{E}$')
    ax1.plot(tf, jdote_tot_drifts, lw=2, color='m', 
            label=r'$\mathbf{j}\cdot\mathbf{E}$')
    ax1.plot(te, dke, lw=2, color='k', label=kename)
    ax1.plot([np.min(te), np.max(te)], [0,0], 'k--')
    ax1.set_ylabel(r'$d\varepsilon_c/dt$', fontdict=font, fontsize=20)
    ax1.tick_params(reset=True, labelsize=16)
    ax1.tick_params(axis='x', labelbottom='off')
    start, end = ax1.get_ylim()
    ax1.yaxis.set_ticks(np.arange(start+0.1, end, 0.2))
    ax1.set_xlim([0, 800])

    ax2 = fig.add_axes([xs, ys, width, height])
    ax2.plot(tf, jcpara_dote_int, lw=2, color='b')
    ax2.plot(tf, jgrad_dote_int, lw=2, color='g')
    ax2.plot(tf, jmag_dote_int, lw=2, color='r')
    #ax2.plot(tf, jqnupara_dote_int, lw=2)
    ax2.plot(tf, jdote_tot_drifts_int, lw=2, color='m')
    ax2.plot(te, ke-ke[0], color='k', lw=2)
    ax2.plot([np.min(te), np.max(te)], [0,0], 'k--')

    ax2.set_xlabel(r'$t\Omega_{ci}$', fontdict=font, fontsize=20)
    ax2.set_ylabel(r'$\varepsilon_c$', fontdict=font, fontsize=20)
    ax2.tick_params(labelsize=16)
    ax2.set_xlim([0, 800])
    start, end = ax2.get_ylim()
    ax2.yaxis.set_ticks(np.arange(start+0.05, end, 0.1))

    #ax1.legend(loc=1, prop={'size':16}, ncol=2,
    #        shadow=True, fancybox=True)

    ax1.text(690, 0.15, r'$\mathbf{j}_c\cdot\mathbf{E}$', color='b', fontsize=20)
    ax1.text(690, 0.35, r'$\mathbf{j}_g\cdot\mathbf{E}$', color='g', fontsize=20)
    ax1.text(550, 0.35, r'$\mathbf{j}_m\cdot\mathbf{E}$', color='r', fontsize=20)
    ax1.text(550, -0.3, r'$dK_e/dt$', color='k', fontsize=20)
    ax1.text(690, -0.3, r"$\mathbf{j}_\perp\cdot\mathbf{E}$", color='m', fontsize=20)
#    ax1.text(690, 0.1, r'$\boldsymbol{j}_c\cdot\boldsymbol{E}$', color='b', fontsize=20)
#    ax1.text(570, 0.3, r'$\boldsymbol{j}_g\cdot\boldsymbol{E}$', color='g', fontsize=20)
#    ax1.text(690, 0.3, r'$\boldsymbol{j}_m\cdot\boldsymbol{E}$', color='r', fontsize=20)
#    #ax1.text(500, -0.35, r'$\boldsymbol{j}_\parallel\cdot\boldsymbol{E}$', color='c', fontsize=20)
#    ax1.text(570, 0.1, r'$dK_e/dt$', color='k', fontsize=20)
#    ax1.text(690, -0.3, r"$\boldsymbol{j}'_\perp\cdot\boldsymbol{E}$", color='m', fontsize=20)
#    #mpl.rc('text', usetex=False)


    td = 320
    print 'The fraction of perpendicular heating (model): ', \
            jdote_tot_drifts_int[td]/(ke[td]-ke[0])
    print 'The fraction of perpendicular heating (simulation): ', \
            jqnuperp_dote_int[-1]/(ke[-1]-ke[0])

    #plt.tight_layout()
    fname = 'jdrifts_dote_' + species + '.eps'
    fig.savefig(fname)
    plt.show()

def PlotParticleEnergy(ax):
    """Plot particle energy evolution for different beta.
    Args:
        ax: one axes object.
    """
    f = open('ene_pic.dat', 'r')
    data = np.genfromtxt(f, delimiter='')
    f.close()
    t1 = data[:,0]
    Ke1 = data[:,4]

    fname = '../../mime25-beta00025-guide0-200-100-nppc200/ana/ene_pic.dat'
    f = open(fname, 'r')
    data = np.genfromtxt(f, delimiter='')
    f.close()
    t2 = data[:,0]
    Ke2 = data[:,4]

    fname = '../../mime25-beta01-guide0-200-100-nppc200/ana/ene_pic.dat'
    f = open(fname, 'r')
    data = np.genfromtxt(f, delimiter='')
    f.close()
    t3 = data[:,0]
    Ke3 = data[:,4]

    fname = '../../mime25-beta003-guide0-200-100-nppc200/ana/ene_pic.dat'
    f = open(fname, 'r')
    data = np.genfromtxt(f, delimiter='')
    f.close()
    t4 = data[:,0]
    Ke4 = data[:,4]
    
    print (Ke2[-1]-Ke2[0])/Ke2[0], \
            (Ke1[-1]-Ke1[0])/Ke1[0], (Ke3[-1]-Ke3[0])/Ke3[0]

    # This is actually for the case with beta_e = 0.0072
    ax.plot(t2, (Ke2-Ke2[0])*0.005/(0.0072*Ke2[0]), 'b', linewidth=2)

    ax.plot(t1, (Ke1-Ke1[0])/Ke1[0], 'r', linewidth=2)
    ax.plot(t3, (Ke3-Ke3[0])/Ke3[0], 'g', linewidth=2)
    ax.plot(t4, (Ke4-Ke4[0])/Ke4[0], 'orange', linewidth=2)
    ax.set_xlim([0, 1190])
    #ax.set_ylim([0, 1.05])

    #plt.title('Energy spectrum', fontdict=font)
    ax.set_xlabel(r'$t\Omega_{ci}$', fontdict=font, fontsize=20)
    ax.set_ylabel(r'$\Delta K_e/K_e(0)$', fontdict=font, fontsize=20)
    plt.tick_params(labelsize=16)
   
    ax.text(680, 8.8, r'$\beta_e=0.007$', color='blue', 
            rotation=5, fontsize=16)
    ax.text(680, 5, r'$\beta_e=0.02$', color='red', 
            rotation=4, fontsize=16)
    ax.text(680, 2.1, r'$\beta_e=0.06$', color='orange', 
            rotation=0, fontsize=16)
    ax.text(680, -1.5, r'$\beta_e=0.2$', color='green', 
            rotation=0, fontsize=16)

def Read2DfieldData(fname):
    """Read 2D field data from file with name 'fname'.
    Args:
        fname: file name.
    """
    f = open(fname, 'rb')
    data = f.read()

    nx, nz, = struct.unpack('2f', data[0:8])
    nx = int(nx)
    nz = int(nz)
    x = np.zeros(nx)
    z = np.zeros(nz)
    index_start = 8
    index_end = 12
    for ix in range(nx):
        x[ix], = struct.unpack('f', data[index_start:index_end])
        index_start = index_end
        index_end += 4
    for iz in range(nz):
        z[iz], = struct.unpack('f', data[index_start:index_end])
        index_start = index_end
        index_end += 4
    fdata = np.zeros((nx, nz))
    for iz in range(nz):
        for ix in range(nx):
            fdata[ix][iz], = struct.unpack('f', data[index_start:index_end])
            index_start = index_end
            index_end += 4
    f.close()
    return (nx, nz, x, z, fdata)

def PlotJdriftsDote():
    """Plot j dot E due to curvature drift and gradient B drift.
    """
    nx, nz, x, z, jcpara = Read2DfieldData('jcpara_0040.gda')
    nx, nz, x, z, jgrad = Read2DfieldData('jgrad_0040.gda')
    nx, nz, x, z, agy = Read2DfieldData('agyrotropy1_0040.gda')
    nx, nz, x, z, Ay = Read2DfieldData('Ay_0040.gda')
    jcpara *= 1000
    jgrad *= 1000
    width = 0.78
    #height = 0.38
    height = 0.25
    xs = 0.14
    xe = 0.94 - xs
    ys = 0.96 - height
    fig = plt.figure(figsize=(7,5))
    ax1 = fig.add_axes([xs, ys, width, height])
    xmin = np.min(x)
    xmax = np.max(x)
    zmin = np.min(z)
    zmax = np.max(z)
    dmax = 1.0
    p11 = ax1.imshow(jcpara.transpose(1,0), cmap=plt.cm.seismic,
            extent=[xmin, xmax, zmin, zmax],
            aspect='auto',
            origin='lower',
            vmin=-dmax, vmax=dmax,
            interpolation='spline16')
    p12 = ax1.contour(x, z, Ay.transpose(1,0), colors='black', linewidths=0.5)
    ax1.tick_params(axis='x', labelbottom='off')
    ax1.tick_params(labelsize=16)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax1.text(5, 55.2, r'$\mathbf{j}_c\cdot\mathbf{E}$', color='blue',
            fontsize=24, 
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0))

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    cbar1 = fig.colorbar(p11, cax=cax)
    #cbar1.ax.set_ylabel(r'$\mathbf{j}_c\cdot\mathbf{E}$', 
    #        fontsize=16, color='blue')
    cbar1.set_ticks(np.arange(-0.8, 1.0, 0.4))
    cbar1.ax.tick_params(labelsize=16)

    ys -= height + 0.035
    ax2 = fig.add_axes([xs, ys, width, height])
    p21 = ax2.imshow(jgrad.transpose(1,0), cmap=plt.cm.seismic,
            extent=[xmin, xmax, zmin, zmax],
            aspect='auto',
            origin='lower',
            vmin=-dmax, vmax=dmax,
            interpolation='spline16')
    p22 = ax2.contour(x, z, Ay.transpose(1,0), colors='black', linewidths=0.5)
    ax2.tick_params(axis='x', labelbottom='off')
    ax2.tick_params(labelsize=16)
    #ax2.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=16)
    ax2.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax2.text(5, 55, r'$\mathbf{j}_g\cdot\mathbf{E}$', color='green',
            fontsize=24, 
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0))

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    cbar2 = fig.colorbar(p21, cax=cax)
    #cbar2.ax.set_ylabel(r'$\mathbf{j}_g\cdot\mathbf{E}$', 
    #        fontsize=16, color='green')
    cbar2.set_ticks(np.arange(-0.8, 1.0, 0.4))
    cbar2.ax.tick_params(labelsize=16)

    ys -= height + 0.035
    ax3 = fig.add_axes([xs, ys, width, height])
    p31 = ax3.imshow(agy.transpose(1,0), cmap=plt.cm.binary,
            extent=[xmin, xmax, zmin, zmax],
            aspect='auto',
            origin='lower',
            #vmin=-dmax, vmax=dmax,
            interpolation='spline16')
    p32 = ax3.contour(x, z, Ay.transpose(1,0), colors='black', linewidths=0.5)
    ax3.tick_params(labelsize=16)
    ax3.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    ax3.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    ax3.text(5, 55, r'$A_e$', color='black',
            fontsize=24, 
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0))

    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    cbar3 = fig.colorbar(p31, cax=cax)
    #cbar2.ax.set_ylabel(r'$\mathbf{j}_g\cdot\mathbf{E}$', 
    #        fontsize=16, color='green')
    cbar3.set_ticks(np.arange(0, 1.8, 0.4))
    cbar3.ax.tick_params(labelsize=16)

    plt.savefig('jcm_dote.eps')

    print np.sum(jcpara), np.sum(jgrad)
    
    plt.show()

def PlotAgyrotropy():
    """Plot pressure agyrotropy.
    """
    nx, nz, x, z, agy = Read2DfieldData('agyrotropy00_e_sbox_0040.gda')
    nx, nz, x, z, Ay = Read2DfieldData('Ay_sbox_0040.gda')
    width = 0.82
    height = 0.64
    fig = plt.figure(figsize=(7,1.8))
    ax1 = fig.add_axes([0.12, 0.3, width, height])
    xmin = np.min(x)
    xmax = np.max(x)
    zmin = np.min(z)
    zmax = np.max(z)
    p11 = ax1.imshow(agy.transpose(1,0), cmap=plt.cm.jet,
            extent=[xmin, xmax, zmin, zmax],
            aspect='auto',
            origin='lower',
            interpolation='spline16')
    p12 = ax1.contour(x, z, Ay.transpose(1,0), colors='white')
    #ax1.tick_params(axis='x', labelbottom='off')
    ax1.tick_params(labelsize=16)
    ax1.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=16)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=16)
    ax1.text(2.5, 54.5, r'$A_e$', color='orange', fontsize=24,
            bbox=dict(facecolor='blue', alpha=1.0, edgecolor='blue', pad=10.0))

    # plot two lines
    xmax = np.max(x)
    xmin = np.min(x)
    zmax = np.max(z)
    zmin = np.min(z)
    zmid = (zmin+zmax)*0.5
    l1 = ax1.plot([xmin, xmax], [zmid-1, zmid-1], 
            color='red', linewidth=1, linestyle='--')
    l1 = ax1.plot([xmin, xmax], [zmid+1, zmid+1], 
            color='red', linewidth=1, linestyle='--')

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    cbar1 = fig.colorbar(p11, cax=cax)
    #cbar1.ax.set_ylabel(r'$A_e$', fontsize=16)
    yticks = np.linspace(0.2, 1.4, 7)
    cbar1.set_ticks(yticks)

    plt.savefig('agyrotropy.eps')
    plt.show()

def HistogramB():
    """Get the histogram of |B|.
    """
    nx, nz, x, z, absB = Read2DfieldData('absB_0040.gda')
    maxB = np.max(absB)
    minB = np.min(absB)
    nb = 100
    bins = np.linspace(minB, maxB, nb)
    hist, bin_edges = np.histogram(absB, bins)
    print 'Maximum and Minimum of B: ', np.max(absB), np.min(absB)
    p1 = plt.plot(bins[:nb-1], hist)
    plt.show()

def HistogramAgy():
    """Get the histogram of agyrotropy.
    """
    nx, nz, x, z, agy = Read2DfieldData('agyrotropy00_e_sbox_0040.gda')
    maxAgy = np.max(agy)
    minAgy = np.min(agy)
    nb = 100
    bins = np.linspace(minAgy, maxAgy, nb)
    hist, bin_edges = np.histogram(agy, bins)
    print 'Maximum and Minimum of Agyrotropy: ', maxAgy, minAgy
    p1 = plt.plot(bins[:nb-1], hist)
    plt.show()

def HistogramAgy_Time():
    """Get the histogram of agyrotropy at different time.
    """
    fig, ax = plt.subplots()
    for it in range(20, 400, 100):
        fname = 'agyrotropy/agyrotropy00_e_sbox_' + str(it).zfill(4) + '.gda'
        nx, nz, x, z, agy = Read2DfieldData(fname)
        maxAgy = np.max(agy)
        minAgy = np.min(agy)
        nb = 100
        bins = np.linspace(minAgy, maxAgy, nb)
        hist, bin_edges = np.histogram(agy, bins)
        hist /= (bins[1:nb] - bins[0:nb-1])
        print 'Maximum and Minimum of Agyrotropy: ', maxAgy, minAgy
        p1 = ax.plot(bins[:nb-1], hist, linewidth=2)
    
    ax.set_xlim([0, 0.5])
    plt.show()


def HistogramB_Agy():
    """Get the histogram of |B| and agyrotropy.
    """
    nx, nz, x, z, absB = Read2DfieldData('absB_0040.gda')
    maxB = np.max(absB)
    minB = np.min(absB)
    nb = 100
    db = (maxB - minB) / nb
    binsB = np.linspace(minB, maxB, nb)
    histB, bin_edges = np.histogram(absB, binsB)
    histB = histB / float(np.sum(histB))
    histB = histB / db

    nx, nz, x, z, agy = Read2DfieldData('agyrotropy1_0040.gda')
    maxAgy = np.max(agy)
    minAgy = np.min(agy)
    dAgy = (maxAgy - minAgy) / nb
    binsAgy = np.linspace(minAgy, maxAgy, nb)
    histAgy, bin_edges = np.histogram(agy, binsAgy)
    histAgy = histAgy / float(np.sum(histAgy))
    histAgy = histAgy / dAgy

    fig = plt.figure(figsize=(7,2.5))
    width = 0.37
    height = 0.73
    ax1 = fig.add_axes([0.12, 0.22, width, height])
    p1 = ax1.plot(binsB[:nb-1], histB, color='k', linewidth=2)
    ax1.set_xlabel(r'$B/B_0$', fontdict=font, fontsize=16)
    ax1.set_ylabel(r'$f(B)$', fontdict=font, fontsize=16)
    plt.tick_params(labelsize=16)

    ax2 = fig.add_axes([0.62, 0.22, width, height])
    p2 = ax2.plot(binsAgy[:nb-1], histAgy, color='k', linewidth=2)
    ax2.set_xlim([0, 1.5])
    ax2.set_xlabel(r'$A_e$', fontdict=font, fontsize=16)
    ax2.set_ylabel(r'$f(A_e)$', fontdict=font, fontsize=16)
    plt.tick_params(labelsize=16)
    plt.savefig('hist_B_agy.eps')
    plt.show()

def PlotBetaRho():
    """Plot plasma beta and number density.
    """
    nx, nz, x, z, beta_e = Read2DfieldData('beta_e_0040.gda')
    nx, nz, x, z, ne = Read2DfieldData('ne_0040.gda')
    nx, nz, x, z, Ay = Read2DfieldData('Ay_0040.gda')
    width = 0.8
    height = 0.3
    xs = 0.12
    ys = 0.9 - height
    fig = plt.figure(figsize=(7,4))
    ax1 = fig.add_axes([xs, ys, width, height])
    xmin = np.min(x)
    xmax = np.max(x)
    zmin = np.min(z)
    zmax = np.max(z)
    p11 = ax1.imshow(beta_e.transpose(1,0), cmap=plt.cm.jet,
            extent=[xmin, xmax, zmin, zmax],
            aspect='auto',
            origin='lower',
            interpolation='spline16',
            norm=LogNorm(vmin=0.01, vmax=10))
    p12 = ax1.contour(x, z, Ay.transpose(1,0), colors='white', linewidths=0.5)
    ax1.tick_params(axis='x', labelbottom='off')
    ax1.tick_params(labelsize=16)
    ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    #ax1.text(10, 55.5, r'$\beta_e$', color='yellow', fontsize=24,
    #        bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0))
    ax1.set_title(r'$\beta_e$', fontsize=24)

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    cbar1 = fig.colorbar(p11, cax=cax)
    cbar1.ax.tick_params(labelsize=16)
    #cbar1.ax.set_xlabel(r'$\beta_e$', fontsize=16)
    #cbar1.ax.xaxis.set_label_position('top') 

    ys -= height + 0.15
    ax2 = fig.add_axes([xs, ys, width, height])
    p21 = ax2.imshow(ne.transpose(1,0), cmap=plt.cm.jet,
            extent=[xmin, xmax, zmin, zmax],
            aspect='auto',
            origin='lower',
            interpolation='spline16')
    p22 = ax2.contour(x, z, Ay.transpose(1,0), colors='white', linewidths=0.5)
    ax2.tick_params(labelsize=16)
    ax2.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=20)
    ax2.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=20)
    #ax2.text(10, 55.1, r'$n_{acc}/n_e$', color='yellow', fontsize=24,
    #        bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0))
    ax2.set_title(r'$n_{acc}/n_e$', fontsize=24)

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    cbar2 = fig.colorbar(p21, cax=cax)
    cbar2.set_ticks(np.arange(0.2, 1.0, 0.2))
    #cbar2.ax.set_ylabel(r'$n_{>2.7E_{th}}/n_e$', fontsize=16)
    cbar2.ax.tick_params(labelsize=16)
    plt.savefig('beta_e_ne.eps')
    plt.show()

def Histogram_Jcdote_Jgdote():
    """Get the histogram of j_c\cdot E and j_g\cdot E.
    """
    nx, nz, x, z, jcdote = Read2DfieldData('jcpara_0040.gda')
    maxJc = np.max(jcdote)
    minJc = np.min(jcdote)
    maxJ = 0.00001
    minJc = -maxJ
    maxJc = maxJ
    nb = 100
    dJc = (maxJc - minJc) / nb
    binsJc = np.linspace(minJc, maxJc, nb)
    histJc, bin_edges = np.histogram(jcdote, binsJc)
    histJc = histJc / float(np.sum(histJc))
    histJc = histJc / dJc

    nx, nz, x, z, jgdote = Read2DfieldData('jgrad_0040.gda')
    maxJg = np.max(jgdote)
    minJg = np.min(jgdote)
    minJg = -maxJ
    maxJg = maxJ
    dJg = (maxJg - minJg) / nb
    binsJg = np.linspace(minJg, maxJg, nb)
    histJg, bin_edges = np.histogram(jgdote, binsJg)
    histJg = histJg / float(np.sum(histJg))
    histJg = histJg / dJg

    fig = plt.figure(figsize=(7,2.5))
    width = 0.37
    height = 0.73
    ax1 = fig.add_axes([0.12, 0.22, width, height])
    p1 = ax1.plot(binsJc[:nb-1], histJc, color='k', linewidth=2)
    ax1.set_xlabel(r'$B/B_0$', fontdict=font, fontsize=16)
    ax1.set_ylabel(r'$f(B)$', fontdict=font, fontsize=16)
    ax1.set_xlim([-maxJ, maxJ])
    plt.tick_params(labelsize=16)

    ax2 = fig.add_axes([0.62, 0.22, width, height])
    p2 = ax2.plot(binsJg[:nb-1], histJg, color='k', linewidth=2)
    ax2.set_xlim([0, 1.5])
    ax2.set_xlabel(r'$A_e$', fontdict=font, fontsize=16)
    ax2.set_ylabel(r'$f(A_e)$', fontdict=font, fontsize=16)
    ax2.set_xlim([-maxJ, maxJ])
    plt.tick_params(labelsize=16)
    plt.savefig('jc_jg_dote.eps')

    ec1 = histJc*np.diff(binsJc)
    ec2 = histJg*np.diff(binsJg)
    #print np.sum(ec1), np.sum(ec2)
    print np.sum(jcdote), np.sum(jgdote)
    plt.show()

def Histogram_curvB_gradB():
    """Get the histogram of the magnetic length scales..
    """
    nx, nz, x, z, data1 = Read2DfieldData('curvRadius_0040.gda')
    max1 = np.max(data1)
    min1 = np.min(data1)
    max0 = 20
    min1 = 0
    max1 = max0
    nb = 100
    db = (max1 - min1) / nb
    bins1 = np.linspace(min1, max1, nb)
    hist1, bin_edges = np.histogram(data1, bins1)
    #hist1 = hist1 / float(np.sum(hist1))
    #hist1 = hist1 / db

    nx, nz, x, z, data2 = Read2DfieldData('lengthGradB_0040.gda')
    max2 = np.max(data2)
    min2 = np.min(data2)
    min2 = 0
    max2 = max0
    db = (max2 - min2) / nb
    bins2 = np.linspace(min2, max2, nb)
    hist2, bin_edges = np.histogram(data2, bins2)
    #hist2 = hist2 / float(np.sum(hist2))
    #hist2 = hist2 / db

    fig = plt.figure(figsize=(7,2.5))
    width = 0.37
    height = 0.73
    ax1 = fig.add_axes([0.12, 0.22, width, height])
    p1 = ax1.plot(bins1[:nb-1], hist1, color='k', linewidth=2)
    #ax1.set_xlabel(r'$B/B_0$', fontdict=font, fontsize=16)
    #ax1.set_ylabel(r'$f(B)$', fontdict=font, fontsize=16)
    #ax1.set_xlim([-maxJ, maxJ])
    plt.tick_params(labelsize=16)

    ax2 = fig.add_axes([0.62, 0.22, width, height])
    p2 = ax2.plot(bins2[:nb-1], hist2, color='k', linewidth=2)
    #ax2.set_xlim([0, 1.5])
    #ax2.set_xlabel(r'$A_e$', fontdict=font, fontsize=16)
    #ax2.set_ylabel(r'$f(A_e)$', fontdict=font, fontsize=16)
    plt.tick_params(labelsize=16)
    plt.show()

def PlotNumberRho():
    """Plot number density.
    """
    nx, nz, x, z, ne = Read2DfieldData('ne_0027.gda')
    nx, nz, x, z, Ay = Read2DfieldData('Ay_0027.gda')
    width = 1.0
    height = 0.96
    fig = plt.figure(figsize=(7,0.5))
    ax1 = fig.add_axes([0.0, 0.02, width, height])
    xmin = np.min(x)
    xmax = np.max(x)
    zmin = np.min(z)
    zmax = np.max(z)
    p11 = ax1.imshow(ne.transpose(1,0), cmap=plt.cm.jet,
            extent=[xmin, xmax, zmin, zmax],
            aspect='auto',
            origin='lower',
            interpolation='spline16')
    p12 = ax1.contour(x, z, Ay.transpose(1,0), colors='white')
    ax1.tick_params(axis='x', labelbottom='off')
    ax1.tick_params(axis='y', labelleft='off')
    #ax1.tick_params(labelsize=16)
    #ax1.set_xlabel(r'$x/d_i$', fontdict=font, fontsize=16)
    #ax1.set_ylabel(r'$z/d_i$', fontdict=font, fontsize=16)
    #ax1.text(2.5, 54.5, r'$A_e$', color='orange', fontsize=24,
    #        bbox=dict(facecolor='blue', alpha=1.0, edgecolor='blue', pad=10.0))

    ## create an axes on the right side of ax. The width of cax will be 5%
    ## of ax and the padding between cax and ax will be fixed at 0.05 inch.
    #divider = make_axes_locatable(ax1)
    #cax = divider.append_axes("right", size="2%", pad=0.05)
    #cbar1 = fig.colorbar(p11, cax=cax)
    ##cbar1.ax.set_ylabel(r'$A_e$', fontsize=16)
    ##yticks = np.linspace(0.2, 1.4, 7)
    ##cbar1.set_ticks(yticks)

    plt.savefig('numRho.eps')
    plt.show()

def jParaPerpDote():
    """Plot the parallel and perpendicular heating.
    """
    tf, jcpara_dote, jcperp_dote, jmag_dote, jgrad_dote, \
    jdiagm_dote, jpolar_dote, jexb_dote, jpara_dote, jperp_dote, \
    jperp1_dote, jperp2_dote, jqnupara_dote, jqnuperp_dote, \
    jagy_dote, jtot_dote, \
    jcpara_dote_int, jcperp_dote_int, jmag_dote_int, jgrad_dote_int, \
    jdiagm_dote_int, jpolar_dote_int, jexb_dote_int, jpara_dote_int, \
    jperp_dote_int, jperp1_dote_int, jperp2_dote_int, \
    jqnupara_dote_int, jqnuperp_dote_int, jagy_dote_int, jtot_dote_int \
    = ReadJDriftsDoteData('e')

    te, dkei, dkee, de_electric, de_magnetic, de_bx, de_by, de_bz, \
    kei, kee, e_electric, e_magnetic, e_bx, e_by, e_bz = \
    ReadEnergiesPIC()

    jtot_dote = jqnupara_dote + jqnuperp_dote
    jtot_dote_int = jqnupara_dote_int + jqnuperp_dote_int
    dke = dkee
    ke = kee
    kename = '$\Delta K_e$'

    #fig, axes = plt.subplots(2, sharex=True, sharey=False)
    fig = plt.figure(figsize=[7, 5])
   
    width = 0.86
    height = 0.4
    xs = 0.10
    ys = 0.97-height
    #mpl.rc('text', usetex=False)
    ax1 = fig.add_axes([xs, ys, width, height])
    ax1.plot(tf, jqnupara_dote, lw=2, color='b', 
            label=r'$\mathbf{j}_{e\parallel}\cdot\mathbf{E}$')
    ax1.plot(tf, jqnuperp_dote, lw=2, color='g',
            label=r'$\mathbf{j}_{e\perp}\cdot\mathbf{E}$')
    ax1.plot(tf, jtot_dote, lw=2, color='r',
            label=r'$\mathbf{j}\cdot\mathbf{E}$')
    ax1.plot(te, dke, lw=2, color='k', label=kename)
    ax1.plot([np.min(te), np.max(te)], [0,0], 'k--')
    ax1.set_ylabel(r'$d\varepsilon_c/dt$', fontdict=font, fontsize=24)
    ax1.tick_params(reset=True, labelsize=20)
    ax1.tick_params(axis='x', labelbottom='off')
    ax1.set_ylim([-0.15, 0.35])
    ax1.yaxis.set_ticks(np.arange(-0.1, 0.4, 0.1))
    ax1.set_xlim([0, 800])
    ax1.text(10, 0.25, r'$(e)$', color='black', fontsize=24,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0))

    tf, jcpara_dote, jcperp_dote, jmag_dote, jgrad_dote, \
    jdiagm_dote, jpolar_dote, jexb_dote, jpara_dote, jperp_dote, \
    jperp1_dote, jperp2_dote, jqnupara_dote, jqnuperp_dote, \
    jagy_dote, jtot_dote, \
    jcpara_dote_int, jcperp_dote_int, jmag_dote_int, jgrad_dote_int, \
    jdiagm_dote_int, jpolar_dote_int, jexb_dote_int, jpara_dote_int, \
    jperp_dote_int, jperp1_dote_int, jperp2_dote_int, \
    jqnupara_dote_int, jqnuperp_dote_int, jagy_dote_int, jtot_dote_int \
    = ReadJDriftsDoteData('i')

    jtot_dote = jqnupara_dote + jqnuperp_dote
    jtot_dote_int = jqnupara_dote_int + jqnuperp_dote_int
    dke = dkei
    ke = kei
    kename = '$\Delta K_i$'

    #mpl.rc('text', usetex=False)
    ys -= height + 0.03
    ax2 = fig.add_axes([xs, ys, width, height])
    ax2.plot(tf, jqnupara_dote, lw=2, color='b', 
            label=r'$\mathbf{j}_{e\parallel}\cdot\mathbf{E}$')
    ax2.plot(tf, jqnuperp_dote, lw=2, color='g',
            label=r'$\mathbf{j}_{e\perp}\cdot\mathbf{E}$')
    ax2.plot(tf, jtot_dote, lw=2, color='r',
            label=r'$\mathbf{j}\cdot\mathbf{E}$')
    ax2.plot(te, dke, lw=2, color='k', label=kename)
    ax2.plot([np.min(te), np.max(te)], [0,0], 'k--')
    ax2.set_xlabel(r'$t\Omega_{ci}$', fontdict=font, fontsize=24)
    ax2.set_ylabel(r'$d\varepsilon_c/dt$', fontdict=font, fontsize=24)
    ax2.tick_params(reset=True, labelsize=20)
    ax2.set_ylim([-0.2, 0.8])
    ax2.yaxis.set_ticks(np.arange(-0.2, 0.9, 0.2))
    ax2.set_xlim([0, 800])
    ax2.text(400, 0.6, r'$\mathbf{j}_{\parallel}\cdot\mathbf{E}$',
            color='blue', fontsize=24)
    ax2.text(650, 0.6, r'$\mathbf{j}_{\perp}\cdot\mathbf{E}$',
            color='green', fontsize=24)
    ax2.text(400, 0.4, r'$(\mathbf{j}_{\parallel}+\mathbf{j}_\perp)\cdot\mathbf{E}$',
            color='red', fontsize=24)
    ax2.text(650, 0.4, r'$dK_e/dt$',
            color='black', fontsize=24)
    ax2.text(10, 0.6, r'$(i)$', color='black', fontsize=24,
            bbox=dict(facecolor='none', alpha=1.0, edgecolor='none', pad=10.0))

    fig.savefig('jpp_dote.eps')

    plt.show()

def plotPscalarDist():
    """Plot the distribution of scalar pressure.
    """
    f = open('data/pScalar_dist00_e.gda', 'r')
    data = f.read()
    nt, = struct.unpack('i', data[0:4])
    nbins, = struct.unpack('i', data[4:8])
    pScalar_dist = np.zeros([nt,nbins])
    pScalar = np.zeros(nbins)

    index_start = 8
    index_end = 12
    for ibin in range(nbins):
        pScalar[ibin], = struct.unpack('f', data[index_start:index_end])
        index_start = index_end
        index_end += 4

    for it in range(nt):
        for ibin in range(nbins):
            pScalar_dist[it,ibin], = struct.unpack('f', data[index_start:index_end])
            index_start = index_end
            index_end += 4
    f.close()
    fig, ax = plt.subplots()
    p1 = ax.loglog(pScalar, pScalar_dist[nt-1])
    plt.show()

if __name__ == "__main__":
    vthe = 0.1   # electron thermal velocity
    mime = 25.0  # ion and electron mass ratio
    nt = 48
    #for it in range(nt, nt+1):
    #    PlotSpectrum(it, 'e', vthe, mime, False)
    #    #PlotSpectrum(it, 'h', vthe, mime, True)
    #ParticleNumberEnergyEvolution('e', vthe, mime)
    #ParticleNumberEnergyEvolution('h', vthe, mime)
    #AccumulatedParticleNumberAlongEnergy('h')
    #PlotDifferentialNumberAndEnergy(nt, 'e', vthe, 1)
    #PlotDifferentialNumberAndEnergy(nt, 'h', vthe, mime)
    #EvolutionOfDifferentialNumberAndEnergy(nt, 'e', vthe, 1)
    #EvolutionOfDifferentialNumberAndEnergy(nt, 'h', vthe, mime)
    #jDriftsDote('e')
    #jDriftsDote('i')

    #fig = plt.figure(figsize=(7,2.5))
    #width = 0.35
    #height = 0.73
    #ax1 = fig.add_axes([0.12, 0.22, width, height])
    #PlotEnergyEvolution(ax1)
    #ax2 = fig.add_axes([0.64, 0.22, width, height])
    #PlotParticleEnergy(ax2)
    #ax2.set_ylim([-2, 12])
    #plt.savefig('ene_pic_beta.eps')
    #plt.show()

    #PlotBetaRho()
    #PlotJdriftsDote()
    #PlotAgyrotropy()
    #HistogramB()
    #HistogramAgy()
    #HistogramAgy_Time()
    #HistogramB_Agy()
    #Histogram_Jcdote_Jgdote()
    #Histogram_curvB_gradB()
    #PlotNumberRho()

    #jParaPerpDote()
    #plotPscalarDist()
