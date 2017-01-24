
"""
ODE Solver: RK45 version 1.1.

Public domain, Connelly Barnes 2005.
"""

import math

import scipy
from scipy import array as vector
from scipy.linalg import norm


def solve(f, t0, tfinal, y0, tol = 1e-7):
  """
  Solve an ODE numerically using RK45.

  Solves dy/dt = f(t, y).  Returns a list of (t, y) tuples.
  Reference: http://www.library.cornell.edu/nr/bookcpdf/c16-2.pdf
  """

  def F(*args):
    return vector(f(*args))

  t = t0
  hmax = (tfinal - t0) / 128.0
  h = hmax / 4.0
  y = vector(y0)              # Column vector (nx1).
  out = [(t, list(y))]

  # Cash-Karp parameters
  a = [ 0.0, 0.2, 0.3, 0.6, 1.0, 0.875 ]
  b = [[],
       [0.2],
       [3.0/40.0, 9.0/40.0],
       [0.3, -0.9, 1.2],
       [-11.0/54.0, 2.5, -70.0/27.0, 35.0/27.0],
       [1631.0/55296.0, 175.0/512.0, 575.0/13824.0, 44275.0/110592.0, 253.0/4096.0]]
  c  = [37.0/378.0, 0.0, 250.0/621.0, 125.0/594.0, 0.0, 512.0/1771.0]
  dc = [c[0]-2825.0/27648.0, c[1]-0.0, c[2]-18575.0/48384.0,
        c[3]-13525.0/55296.0, c[4]-277.00/14336.0, c[5]-0.25]

  while t < tfinal:
    if t + h > tfinal:
      h = tfinal - t
    if t + h <= t:
      raise ValueError('Singularity in ODE')

    # Compute k[i] function values.
    k = [None] * 6
    k[0] = F(t, y)
    k[1] = F(t+a[1]*h, y+h*(k[0]*b[1][0]))
    k[2] = F(t+a[2]*h, y+h*(k[0]*b[2][0]+k[1]*b[2][1]))
    k[3] = F(t+a[3]*h, y+h*(k[0]*b[3][0]+k[1]*b[3][1]+k[2]*b[3][2]))
    k[4] = F(t+a[4]*h, y+h*(k[0]*b[4][0]+k[1]*b[4][1]+k[2]*b[4][2]+k[3]*b[4][3]))
    k[5] = F(t+a[5]*h, y+h*(k[0]*b[5][0]+k[1]*b[5][1]+k[2]*b[5][2]+k[3]*b[5][3]+k[4]*b[5][4]))

    # Estimate current error and current maximum error.
    E = norm(h*(k[0]*dc[0]+k[1]*dc[1]+k[2]*dc[2]+k[3]*dc[3]+k[4]*dc[4]+k[5]*dc[5]))
    Emax = tol*max(norm(y), 1.0)

    # Update solution if error is OK.
    if E < Emax:
      t += h
      y += h*(k[0]*c[0]+k[1]*c[1]+k[2]*c[2]+k[3]*c[3]+k[4]*c[4]+k[5]*c[5])
      out += [(t, list(y))]

    # Update step size
    if E > 0.0:
      h = min(hmax, 0.85*h*(Emax/E)**0.2)

  return out

def solve_func(f, t0, tfinal, y0, tol = 1e-5):
  """Returns y(t) function, uses linear interpolation."""
  L = solve(f, t0, tfinal, y0, tol)
  T = [t for (t, y) in L]
  Y = [y for (t, y) in L]
  interp_func = scipy.interpolate.interp1d(T, Y, axis=0)
  def func(t):
    return interp_func(t)[0]
  return func

def test_rk45(tol, max_error, max_evals):
  """
  Orbiting space shuttle test case.
  """

  y0 = vector([0.994, 0.0, 0.0, -2.00158510637908252240537862224])
  Period = 17.0652166
  fevals = [0]

  def yp(t, y):
    fevals[0] += 1
    mu = 0.012277471
    muhat = 1 - mu

    u1 = y[0]
    u1p = y[1]
    u2 = y[2]
    u2p = y[3]

    D1 = ((u1 + mu)**2 + u2**2)
    D1 = D1 * math.sqrt(D1)
    D2 = ((u1 - muhat)**2 + u2**2)
    D2 = D2 * math.sqrt(D2)

    return (u1p,
            u1 + 2 * u2p - muhat*(u1+mu) / D1 - mu*(u1-muhat) / D2,
            u2p,
            u2 - 2 * u1p - muhat * u2 / D1 - mu * u2 / D2)
  
  sol = solve(yp, 0.0, Period, y0, tol)
  ylast = sol[len(sol)-1][1]

#  print 'Fevals:', fevals[0]
#  print 'Error:', norm(ylast - y0)
  assert fevals[0] < max_evals
  assert norm(ylast - y0) < max_error

def test():
  """
  Unit tests.
  """

  print 'Testing:'
  test_rk45(tol=1e-6, max_error=0.01, max_evals=2000)
  test_rk45(tol=1e-9, max_error=3e-5, max_evals=5000)
  print '  rk45:        OK'

if __name__ == '__main__':
  test()
