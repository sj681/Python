import matplotlib.pyplot as plt
import numpy as np
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

print(f'mean: {mean}, std/mean: {std/mean}')

#matplotlib.rcParams.update({'font.size': 20})
shape,loc,scale = lognorm.fit(r)

print(shape,loc,scale)
mean2 = float(input("what is the mean radius? "))
std2 = float(input("what is the std? "))
x = np.linspace(0, 100, 100)
print(std2, 0.0001,mean2)

modelled_label = ('modelled grain distribution: ' 
    f'mean = {mean:.2f}, std = {std/mean:.2f}')
input_label = ('input grain distribution: '
    f'mean = {mean2:.2f}, std = {std2:.2f}')


plt.hist(r, bins=100, density=1,histtype='bar',linewidth=0.1, color='#cdd0d0', 
    label='modelled grains')
plt.plot(x, lognorm.pdf(x, std/mean, 0.00001, mean), linewidth=2.0, 
    color='#20547C', label=modelled_label)
plt.plot(x, lognorm.pdf(x,std2, 0.0001,mean2),linewidth=2.0,color='#C07F28', 
    label=input_label)
plt.legend(loc='upper right', frameon=False)
plt.xlabel("grain radius (Angstrom)")
plt.ylabel("Probability")

plt.show()
