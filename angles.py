import numpy as np
import itertools
from pathlib import Path


def length(v):
    return np.sqrt(np.dot(v, v))


def angle(v1, v2):
    return np.arccos(np.dot(v1, v2) / (length(v1) * length(v2)))


degrees = 180 / np.pi

p = Path('./angles_dir/')

for filepath in p.iterdir():

    line = np.loadtxt(filepath)

    S1 = line[2:5]
    S2 = line[6:9]
    S3 = line[10:13]
    S4 = line[14:17]

    ms = (line[5] + line[9] + line[13] + line[17]) / 4
    vector_combinations = itertools.combinations((S1, S2, S3, S4), r=2)
    angles = [angle(*vc) * degrees for vc in vector_combinations]

    av = np.mean(angles)
    error = np.std(angles)

    print(filepath.name, *angles, ms, av, error)
