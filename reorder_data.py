import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.fftpack
from scipy.optimize import curve_fit
import scipy.interpolate


def fit_func(x,A,d,c_inf):
    return A*exp(-x/d) - c_inf

def FFTcorr(x,y,T,output_data,output_dataw): #,i_S,i_F):
    fout=open(output_data,'w+')
    foutw=open(output_dataw,'w+')
    n1=len(y)
    # Number of samplepoints
    N=len(x)
    #print "N =", N
    # sample spacing
    T = 1.0               #time is 0.1*10.0**(-12)

    # plot after the first np lines
    npfft=2
    temp = x[0]
    #### need to put the column of x and Correlation
    # x = data[i_S:i_F,0]  # position
    # N=i_F-i_S
    # y = data[i_S:i_F,4]  # correlation
    window = np.hamming(N)   # windowing function
    xmin,xmax = [min(x),max(x)]
    #print "max,min",xmax, xmin
    x_grid=np.linspace(xmin,xmax,N) # start>min value in the data;max value<max value in data,nr of points
    T=x_grid[1]-x_grid[0]
    #print "Delta x is:",T
    y_interp = scipy.interpolate.interp1d(x, y)
    y_int =y_interp(x_grid)
    N=np.size(y_int)
    #print "N is :",N
    x_new = x #np.linspace(x_grid[0], x_grid[-1], 5*N)
    y_new = y #fit_func(x_new,a,b,c)
    x_old=x
    y_old=y
    y=y_int
    x=x_grid
    #print "size y, y_interpolare", np.size(y),np.size(y_int)
    y1 = y #- fit_func(x_grid,a,b,c)
    y2 = y1* window
    yf = scipy.fftpack.fft(y)
    y1f = scipy.fftpack.fft(y1)
    y2f = scipy.fftpack.fft(y2)
    peakfft=max(abs(yf))
    index_peak=np.argmax(abs(yf))
    peakfftright=max(abs(yf[index_peak+2:N/2]))
    index_peakright=np.argmax(abs(yf[index_peak+2:N/2]))
    peakffleft=max(abs(yf[:index_peak-2]))
    index_peakleft=np.argmax(abs(yf[:index_peak-2]))
    xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
    print "index", index_peak,peakfft,np.shape(yf)
    print "peak at ",xf[index_peak], " peak value: ", abs(yf[index_peak]),abs(peakfft)
    plt.ioff()  # turn of diplay plots
    xf_lim=min(index_peak+1000,N/2)
    print >> fout, '\n'.join(
        '{index0} \t {x} \t {z} \t {y.real} \t {y.imag}'.format(index0=1,x=a,y=b, z=2.0/N*abs(b))
        for a, b in zip(xf[npfft:], yf[npfft:])
    )
    print >> fout, ''
    print >> foutw, '\n'.join(
        '{index0} \t {x} \t {z} \t {y.real} \t {y.imag}'.format(index0=1,x=a,y=b, z=2.0/N*abs(b))
        for a, b in zip(xf[npfft:], y2f[npfft:])
    )
    print >> foutw, ''
    # print >> fout, xf,'\t',yf,'\t',i,'\n'
    plt.figure(figsize=(15,15))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
    plt.subplot(4, 1, 1)
    plt.title("Original data + interpolation")
    #plt.plot(x_old,y_old,'o',x,y,'o', x_new, y_new,'r')
    plt.plot(x_old,y_old,'o')
    # the original data
    plt.subplot(4, 2, 3)
    plt.title("Original data")
    plt.plot(x, y)
    plt.subplot(4, 2, 5)
    plt.title("FFT")
    plt.plot(xf, 2.0/N * np.abs(yf[0:N/2]))
    plt.subplot(4, 2, 7)
    plt.title("FFT around peak")
    plt.plot(xf[npfft:xf_lim], 2.0/(N) * np.abs(yf[0:N/2])[npfft:xf_lim])


    #corrected data and windowing
    plt.subplot(4, 2, 4)
    plt.title("Corrected data+ window")
    plt.plot(x, y2)
    plt.subplot(4, 2, 6)
    plt.title("FFT")
    plt.plot(xf, 2.0/N * np.abs(y2f[0:N/2]))
    plt.subplot(4, 2, 8)
    plt.title("FFT around peak")
    plt.plot(xf[npfft:xf_lim], 2.0/(N) * np.abs(y2f[0:N/2])[npfft:xf_lim])

    plt.savefig(output_data+".png")
    #print index_peak,index_peakright,N/2
    return xf[index_peak], 2.0/N*abs(yf[index_peak]),xf[index_peakright+index_peak+2], 2.0/N*abs(yf[index_peakright+index_peak+2]),xf[index_peakleft], 2.0/N*abs(yf[index_peakleft])




input_files=[15,30,35,40,45,50]

for file in input_files:
    file_in='output_'+ str(file)
    my_list = [2,3,4,6,7,8,10,11,12,14,15,16]
    my_list2 = [0,2,3,4,6,7,8,10,11,12,14,15,16]
    #where the data is
    path="./"
    #whre to save the results; you need to make the folder first
    pathout="./"

    skip = 500006

    print file_in
    for i in range(1,20,1):
        print i
        for list in my_list:
            my_list2.append(int(list) + 36*i)

    print my_list2
    try:
        data= np.loadtxt(file_in,unpack=True, skiprows=skip, usecols=(my_list2))
        x =  data[0][:]
        print x[0]
        diff = 1e-13
        minus = x[0] - diff
        #print "minus" ,minus,diff, data[0][0], data[0][1]
        for i in range(0,len(x)):
            x[i] = x[i] -  minus
        S = 12
        for i in range(0,20):
            for j in range(0,13):
                y = data[12*i + 1 + j][:]
                print i , j, my_list2[12*i + 1 + j]
                file_out=pathout+file_in+'S' + str(i) + 'xyz'+str(j)+'.FFTresult'
                file_outw=pathout+file_in+'S' + str(i) + 'xyz'+str(j)+'windowing.dat'
                px,py,pxr,pyr,pxl,pyl=FFTcorr(x,y,1.0,file_out,file_outw)

        print "peak:",file_in,px,py,pxr,pyr,pxl,pyl
        #print >> foutr, file_in,px,py,pxr,pyr,pxl,pyl
    except:
        print "nah"
