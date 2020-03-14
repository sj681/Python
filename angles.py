import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import lognorm
import math
import os


def length(v):
  return math.sqrt(np.dot(v, v))

def angle(v1, v2):
  return math.acos(np.dot(v1, v2) / (length(v1) * length(v2)))



for filename in os.listdir(r'/Users/sj681/Documents/PhD/IrMn_results/disordered_composition/data'):
    filepath = '/Users/sj681/Documents/PhD/IrMn_results/disordered_composition/data/'+filename


    line = np.loadtxt(filepath,  unpack = True)

    S1= [line[2], line[3], line[4]]
    S2= [line[6], line[7], line[8]]
    S3= [line[10], line[11], line[12]]
    S4= [line[14], line[15], line[16]]

    ms = (line[5] + line[9] + line[13] + line[17])/4
    angles = []
    angles.append(angle(S1,S2)*57.2958)
    angles.append(angle(S1,S3)*57.2958)
    angles.append(angle(S2,S3)*57.2958)
    angles.append(angle(S1,S4)*57.2958)
    angles.append(angle(S2,S4)*57.2958)
    angles.append(angle(S3,S4)*57.2958)
    av = np.mean(angles)
    error = np.std(angles)

    print filename, angles[0], angles[1], angles[2], angles[3], angles[4], angles[5],ms,av,error
