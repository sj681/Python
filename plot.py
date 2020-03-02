import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import lognorm

x,y,r = np.loadtxt('dist2',  unpack = True)

mean = np.average(r)
std = np.nanstd(r)
r[r == 0] = np.nan
r = r[~np.isnan(r)]

print(mean,std/mean)


for i in range(0, len(x)):
    circle = plt.Circle((x[i],y[i]),r[i],fill=False)
    ax=plt.gca()
    ax.add_patch(circle)
    plt.axis('scaled')
plt.xlabel('x (nm)')
plt.ylabel('y (nm)')
plt.show()
