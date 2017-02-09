from math import *

import matplotlib.pylab as plt
import numpy as np
from matplotlib import animation
from scipy.integrate import odeint

# solve the ode problem of the double compound pendulum again

g = 9.82
L = 0.5
m = 0.1


def dx(x, t):
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]

    dx1 = 6.0 / (m * L**2) * (2 * x3 - 3 * cos(x1 - x2) * x4) / (
        16 - 9 * cos(x1 - x2)**2)
    dx2 = 6.0 / (m * L**2) * (8 * x4 - 3 * cos(x1 - x2) * x3) / (
        16 - 9 * cos(x1 - x2)**2)
    dx3 = -0.5 * m * L**2 * (dx1 * dx2 * sin(x1 - x2) + 3 * (g / L) * sin(x1))
    dx4 = -0.5 * m * L**2 * (-dx1 * dx2 * sin(x1 - x2) + (g / L) * sin(x2))
    return [dx1, dx2, dx3, dx4]


x0 = [pi / 2, pi / 2, 0, 0]  # initial state
t = np.linspace(0, 10, 250)  # time coordinates
x = odeint(dx, x0, t)  # solve the ODE problem

fig, ax = plt.subplots(figsize=(5, 5))

ax.set_ylim([-1.5, 0.5])
ax.set_xlim([1, -1])

pendulum1, = ax.plot([], [], color="red", lw=2)
pendulum2, = ax.plot([], [], color="blue", lw=2)


def init():
    pendulum1.set_data([], [])
    pendulum2.set_data([], [])


def update(n):
    # n = frame counter
    # calculate the positions of the pendulums
    x1 = +L * sin(x[n, 0])
    y1 = -L * cos(x[n, 0])
    x2 = x1 + L * sin(x[n, 1])
    y2 = y1 - L * cos(x[n, 1])

    # update the line data
    pendulum1.set_data([0, x1], [0, y1])
    pendulum2.set_data([x1, x2], [y1, y2])


anim = animation.FuncAnimation(
    fig, update, init_func=init, frames=len(t), blit=True)

mywriter = animation.FFMpegWriter()
# anim.save can be called in a few different ways, some which might or might not work
# on different platforms and with different versions of matplotlib and video encoders
#anim.save('animation.mp4', fps=20, extra_args=['-vcodec', 'libx264'], writer=animation.FFMpegWriter())
#anim.save('animation.mp4', fps=20, extra_args=['-vcodec', 'libx264'])
#anim.save('animation.mp4', fps=20, writer="ffmpeg", codec="libx264")
anim.save('animation.mp4', fps=20, writer=mywriter, codec="libx264")

plt.close(fig)
