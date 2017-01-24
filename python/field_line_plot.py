import matplotlib.pylab as plt
import numpy as np

font = {
    'family': 'serif',
    #'color'  : 'darkred',
    'color': 'black',
    'weight': 'normal',
    'size': 24,
}

f = open('../xz_field.dat', 'r')
data = np.genfromtxt(f, delimiter='')
f.close()

np, ndim = data.shape
print 'Total number of points: ', np

fig, ax = plt.subplots()
p1 = ax.plot(data[:, 0], data[:, 1], linewidth=2)
ax.set_xlim([0, 200])
ax.set_xlabel(r'$x$', fontdict=font)
ax.set_ylabel(r'$z$', fontdict=font)
ax.tick_params(labelsize=16)
plt.show()
