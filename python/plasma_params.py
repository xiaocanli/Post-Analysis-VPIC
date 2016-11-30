from math import *

c0 = 3.0E5    # km/s
T0 = 1.0E6    # Kelvin
n0 = 1.0E9    # cm^-3
kb = 1.38E-23 # Boltzmann constant 
qe = 1.6E-19
me = 9.1E-31

def print_params(params):
    """
    """
    print "mi/me        =", params['mi_me']
    print "sqrt(mi/me)  =", sqrt(params['mi_me'])
    print "Ti/Te        =", params['Ti_Te']
    print "Te           =", params['Te'], "K"
    print "Ti           =", params['Ti'], "K"
    print "ne           =", params['ne'], "cm^-3"
    print "ni           =", params['ni'], "cm^-3"
    print "B            =", params['B'], "Gauss"
    print "vthe/c       =", params['vthe'] / c0
    print "vthi/c       =", params['vthi'] / c0
    print "wpe/wce      =", params['wpe'] / params['wce']
    print "wpe/wci      =", params['wpe'] / params['wci']
    print "de/km        =", params['de']/1E5
    print "di/km        =", params['di']/1E5
    print "Debye/cm     =", params['debye']
    print "di/de        =", params['di'] / params['de']
    print "de/Debye     =", params['de'] / params['debye']
    print "V_A          =", params['va'], "km/s"
    print "nu_e         =", params['nu_e'], "/ s"
    print "nu_e/wce     =", params['nu_e'] / params['wce']
    print "nu_e/wci     =", params['nu_e'] / params['wci']
    print "nu_e/wpe     =", params['nu_e'] / params['wpe']
    print "lambda_mfp   =", params['lambda_mfp']/1E2, 'm'
    print ' '

def params_Gosling():
    mi_me = 1836.0
    Ti_Te = 7.0 / 6.0
    Te = 1.4E5
    Ti = Ti_Te * Te
    ne = 8.7    # cm^-3
    ni_ne = 1.0
    ni = ni_ne * ne
    B = 6.2E-5  # Gauss
    vthe = 3.88E3 * sqrt(Te/T0)  # Te is normalized by T0
    vthi = vthe * sqrt(Ti_Te) / sqrt(mi_me)
    wpe = 1.78E9 * sqrt(ne/n0)   # ne is normalized by n0
    wce = 1.76E8 * B / 10.0       # B is normalized by 10 Gauss
    wpi = wpe * sqrt(ni_ne) / sqrt(mi_me)
    wci = wce / mi_me
    de = c0 * 1E5 / wpe # cm
    di = c0 * 1E5 / wpi
    va = 6.89E2 * (B/10.0) / sqrt(ni/n0) # km/s
    debye = 0.22 * sqrt((Te/T0)/(ne/n0))
    params = {'mi_me': mi_me, 'Ti_Te': Ti_Te, 'Te': Te, 'Ti': Ti,
              'vthe': vthe, 'vthi': vthi, 'wpe': wpe, 'wce': wce,
              'de': de, 'di': di, 'va': va, 'debye': debye}
    print_params(params)


def solar_wind_params():
    """Solar wind plasma parameters
    """
    mi_me = 1836.0
    Ti_Te = 1.0
    Te = 1.5E5
    Ti = Ti_Te * Te
    ne = 9.0    # cm^-3
    ni_ne = 1.0
    ni = ni_ne * ne
    B = 6.2E-5  # Gauss
    return (mi_me, Ti, Te, ni, ne, B)


def lab_plasma_params():
    """Laboratory plasma parameters
    """
    mi_me = 40*1836.0
    Ti_Te = 90.9
    Te = 1.28E4
    Ti = Ti_Te * Te
    ne = 3E6    # cm^-3
    ni_ne = 1.0
    ni = ni_ne * ne
    B = 0.5  # Gauss
    return (mi_me, Ti, Te, ni, ne, B)


def lab_plasma_params_updated():
    """Updated laboratory plasma parameters
    """
    mi_me = 40*1836.0
    Ti_Te = 1.0 / 26.0
    ethe  =  1.1  # in eV
    Te = ethe * qe / kb # in Kelvin
    Ti = Ti_Te * Te
    ne = 1.8E6    # cm^-3
    ni_ne = 1.0
    ni = ni_ne * ne
    B = 0.5  # Gauss
    return (mi_me, Ti, Te, ni, ne, B)


def calc_plasma_parameters(plasma_type):
    """
    """
    if plasma_type is 'solar_wind':
        mi_me, Ti, Te, ni, ne, B = solar_wind_params()
    elif plasma_type is 'lab':
        mi_me, Ti, Te, ni, ne, B = lab_plasma_params()
    elif plasma_type is 'lab_updated':
        mi_me, Ti, Te, ni, ne, B = lab_plasma_params_updated()
    else:
        raise ValueError("Wrong plasma type")
    Ti_Te = Ti / Te
    ni_ne = ni / ne
    vthe = 3.88E3 * sqrt(Te/T0)  # Te is normalized by T0
    vthi = vthe * sqrt(Ti_Te) / sqrt(mi_me)
    wpe = 1.78E9 * sqrt(ne/n0)   # ne is normalized by n0
    wce = 1.76E8 * B / 10.0       # B is normalized by 10 Gauss
    wpi = wpe * sqrt(ni_ne) / sqrt(mi_me)
    wci = wce / mi_me
    de = c0 * 1E5 / wpe # cm
    di = c0 * 1E5 / wpi
    va = 6.89E2 * (B/10.0) / sqrt(ni/n0) # km/s
    debye = 0.22 * sqrt((Te/T0)/(ne/n0))
    ln_gama = 16.16 + log((Te/T0)**1.5 / sqrt(ne/n0))
    nu_e = 3.63 * (ne/n0) * ln_gama * (Te/T0)**(-1.5) # electron collision freq
    lambda_mfp = 1.07E8 * (Te/T0)**2 / (ne/n0) / ln_gama # electron mean free path
    params = {'mi_me': mi_me, 'Ti_Te': Ti_Te, 'Te': Te, 'Ti': Ti,
            'ni': ni, 'ne': ne, 'B': B, 'vthe': vthe, 'vthi': vthi,
            'wpe': wpe, 'wce': wce, 'wce': wce, 'wci': wci,
            'de': de, 'di': di, 'va': va, 'debye': debye, 'nu_e': nu_e,
            'lambda_mfp': lambda_mfp}
    return params


def electric_potential_field_vacuum(r, rw, r0, V0):
    """Electric potential and electric field of a charged tether in vacuum

    Args:
        r: the distance to the center of the tether
        rw: the radius of the tether
        r0: the distance where the potential is required to vanish
        V0: the tether potential
    """
    V = V0 * log(r0/r) / log(r0/rw)
    E = (V0 / log(r0/rw)) * (1/r)
    return (V, E)


def electric_potential_field_plasma(r, rw, r0, V0):
    """Electric potential and electric field of a charged tether

    In a solar wind plasma

    Args:
        r: the distance to the center of the tether
        rw: the radius of the tether
        r0: the distance where the potential is required to vanish
        V0: the tether potential
    """
    V = V0 * log(1 + (r0/r)**2) / log(1 + (r0/rw)**2)
    E = (V0 / log(1 + (r0/rw)**2)) * (1/(1+(r0/r)**2)) * (2*r0**2/r**3)
    # V = V0 * log(1 + (r0/r)**2) / (2 * log(r0/rw))
    # E = (V0 / (2*log(r0/rw))) * (1/(1+(r0/r)**2)) * (2*r0**2/r**3)
    return (V, E)


def electric_field_norm(V0_tether, radius_tether, params):
    """Calculate normalized electric field for PIC simulation

    Args:
        radius_tether: tether radius in mm
        V0_tether: electric potential at tether surface with radius radius_tether
        params: plasma parameters
    """
    me = 1.0   # Mass normalization in VPIC
    ec = 1.0   # Charge normalization in VPIC
    c = 1.0    # Light speed in PIC
    wpe = 1.0  # Electron plasma frequency in VPIC 
    e0_real = c0 * 1E3 * params['B'] * 1E-4  # Real value
    wpe_wce = params['wpe'] / params['wce']
    wce = wpe / wpe_wce
    b0 = me*c*wce/ec  # Asymptotic magnetic field strength in VPIC
    e0 = c * b0

    efield_norm = e0 / e0_real

    r0 = 2 * params['debye'] * 1E-2  # 2 times of the Debye length. cm->m
    rw1 = radius_tether * 1E-3       # meter -> mm
    V1, E1 = electric_potential_field_plasma(rw1, rw1, r0, V0_tether)
    rw2 = 1.0E-5      # 0.01 mm tether
    V2, E2 = electric_potential_field_plasma(rw1, rw2, r0, V0_tether)
    norm = E1 / E2    # Match the electric field
    V0_001mm = norm * V0_tether
    V3, E3 = electric_potential_field_plasma(rw2, rw2, r0, V0_001mm)

    # print V0_001mm / (log(r0/rw2))

    de = params['de'] / 1E2  # in m
    rw_pic = 1E-5 / de       # radius of the 0.01 mm tether in simulation
    e0_pic = efield_norm * E3 * rw_pic

    print 'Electric field normalization in VPIC: ', e0_pic


if __name__ == "__main__":
    # V0 = 5600.0
    # radius_tether = 1.0
    # params = calc_plasma_parameters('solar_wind')
    # print_params(params)
    # electric_field_norm(V0, radius_tether, params)
    # params = calc_plasma_parameters('lab')
    # print_params(params)
    # electric_field_norm(V0, radius_tether, params)
    V0 = 100.0
    radius_tether = 1.8
    params = calc_plasma_parameters('lab_updated')
    print_params(params)
    electric_field_norm(V0, radius_tether, params)
