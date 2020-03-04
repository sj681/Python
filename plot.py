import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import lognorm

x,y,r = np.loadtxt('dist2',  unpack = True)

# Do some stats
mean = np.average(r)
std = np.nanstd(r)
r = r[np.nonzero(r)]

print(f'mean: {mean}, std/mean: {std/mean}')

# Create circles
circles = (plt.Circle((xi,yi),ri,fill=False) for xi, yi, ri in zip(x, y, r))

# Create figure and axis, then add circles
fig, ax = plt.subplots()
for circle in circles:
    ax.add_patch(circle)

# Neaten up
plt.axis('scaled')
plt.xlabel('x (nm)')
plt.ylabel('y (nm)')
plt.show()
