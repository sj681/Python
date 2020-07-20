import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import lognorm

x,y,r = np.loadtxt('dist2',  unpack = True)
mean = np.average(r)
std = np.nanstd(r)
r = r[np.nonzero(r)]

print(f'mean: {mean}, std/mean: {std/mean}')

shape,loc,scale = lognorm.fit(r)
print(f'shape: {shape}, loc: {loc}, scale: {scale}')


x2 = np.linspace(mean - 3*std, mean + 3*std, 100)

plt.hist(r, bins=25, density=1,histtype='bar',linewidth=0.1, color='#cdd0d0')
plt.plot(x2, lognorm.pdf(x2, shape, loc, scale),linewidth=2.0,color='#20547C')

plt.show()
