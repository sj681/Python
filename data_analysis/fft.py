import numpy as np
from numpy import fft
import os
import sys
import math
from matplotlib import pyplot as plt

#a program to loop through all the files in data and calculate an fft of each file & print the fft to an image

path="./data/"

for ofile in os.walk(path):
    for o in ofile[2]:

        dir = path+o
        #ignore the first 100000 lines as it has to equilibriate first.
        t, mx= np.loadtxt(dir, usecols=(0, 3), unpack=True, skiprows= 100007)
        amp_t = mx
        # Calculate frequencies
        f = fft.fftfreq(n=len(t), d=t[1]-t[0])

        # Do the fft and normalize
        amp_f = fft.fft(amp_t)
        L = len(amp_f)
        amp_f /= L

        L = int(L/2)
        f = f[1:L]
        amp_f = amp_f[1:L] + amp_f[:L+1:-1]

        # output
        fig, (t_ax, f_ax) = plt.subplots(ncols=2)
        t_ax.set_title('Oscillations in time')
        t_ax.plot(t, amp_t)
        f_ax.set_title('Fourier Transform')
        f_ax.plot(f, abs(amp_f))
        fig.tight_layout()
        plt.show()
