import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.stats as stats
import matplotlib
from scipy.stats import lognorm

plt.xlabel('grain diameter (nm)')
plt.ylabel('normalised intensity')

g,r = np.loadtxt('grains4.txt',  unpack = True)
r[r ==0] = np.nan
r[r > 100] = np.nan
r = r[~np.isnan(r)]

mean = np.nanmean(r)
std = np.nanstd(r)

print(mean,std/mean)

#matplotlib.rcParams.update({'font.size': 20})
shape,loc,scale = lognorm.fit(r)

print(shape,loc,scale)
mean2 = input("what is the mean radius? ")
std2 = input("what is the std? ")
x = np.linspace(0, 100, 100)
print(std2, 0.0001,mean2)



plt.hist(r, bins=100, normed=1,histtype='bar',linewidth=0.1, color='#cdd0d0', label='modelled grains')
plt.plot(x, lognorm.pdf(x, std/mean, 0.00001, mean),linewidth=2.0,color='#20547C', label='modelled grain distribution: mean = %.2f, std = %.2f' %(mean, std/mean))
#plt.plot(x, pdf2, linewidth=2, color='r')
plt.plot(x, lognorm.pdf(x,std2, 0.0001,mean2),linewidth=2.0,color='#C07F28', label='input grain distribution: mean = %.2f, std = %.2f' %(mean2, std2))
leg = plt.legend()
plt.xlabel("grain radius (Angstrom)")
plt.ylabel("Probability")
plt.legend(loc='upper right', frameon=False)

plt.show()
