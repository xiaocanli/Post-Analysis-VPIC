"""
Fitting functions that are commonly used in data analysis.
"""
import math

import numpy as np


def func_maxwellian(x, a, b):
    """
    Function for fitting with exponential expression.
    """
    return a*np.sqrt(x)*np.exp(-b*x)

def func_line(x, a, b):
    """
    Function for fitting with power-law expression.
    Both x and y are given by log values. That's why a linear expression
    is given.
    """
    return a * x + b

def func_power(x, a, b):
    """
    Function for fitting with power-law expression.
    """
    return b * np.power(x, -a)

def func_full(x, c1, c2, c3, c4, c5, c6, c7):
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

def func_full_exp(x, c1, c2, c3, c4, c5, c6, c7):
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

def func_exp(x, c1, c2):
    """Define an exponential function.
    """
    return c1 * np.exp(x*c2)
