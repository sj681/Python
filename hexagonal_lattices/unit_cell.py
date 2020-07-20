import numpy as np
from numpy import fft
import os
import sys
import math
from matplotlib import pyplot as plt
import itertools as itr

#A code to generate the structure + define the properties of BaFe0

#opens files to save the unit cell + properties.
fucf = open('BaFeO.ucf', 'w')
fmat = open('BaFeO.mat', 'w')

#loads in the orthogonalised unit cell
atom_type= np.loadtxt("orthogonal2", usecols=(1), unpack=True,dtype=np.str)
xyz= np.loadtxt("orthogonal2", usecols=(2,3,4))


#defines the unit cell sizes in x,y,z
a = 6.01681046
b = 2*a*np.cos(0.523599)
c = 23.49536771
aac = np.array([a, b, c])

#converts from meV to J/atom
conversion=1e-3*1.60218e-19*2.5*2.5

type2 = np.zeros(len(atom_type))
individual_types = set(atom_type)
Ntypes=len(set(atom_type))
materials=["O", "Ba", "f1", "g2", "k", "a", "b"]
sublattice = [0,1,2,3,4,5,6]
nn_distance = [0,0,5,5,5,5,5]

#Saves the strenght of the exchange interaction (J[materiali][materialj]) to an array
Jnn=[[0,0,0,0,0,0,0],
    [0,0,0,0,0,0, 0],
    [0,0,0.3,1,-4,-5, 0.1],
    [0,0,1,0.6,-5,0, -5],
    [0,0,-4,-5,1.5,0, 2],
    [0,0,-5,0,0,0, 0],
    [0,0,0.1,-5,2,0,0]]


N_atoms=[0,0,0,0,0,0,0]

#Saves the neareset neighbour distance between material i and material j
nn_distance=[[0,0,0,0,0,0,0],
    [0,0,0,0,0,0, 0],
    [0,0,4,4,4,4, 7],
    [0,0,4,3,4,6, 4],
    [0,0,4,4,3.5,4, 4],
    [0,0,4,6,3.5,6.5, 6],
    [0,0,6.5,4,4,6, 6.5]]

#loops over all materials and atoms and if the material = the atom type save the material tpye to an array type2
for i in range(0, len(materials)):
    for j in range(0, len(atom_type)):
        if (materials[i] == atom_type[j]):
            type2[j] = sublattice[i]


#outputs a list of the unit cell atoms to the unit cell file
fucf.write("#unit cell size\n")
fucf.write(str(a) + "\t"+str(b) + "\t"+str(c)+"\n")
fucf.write("#unit cell vectors\n")
fucf.write("1,0,0\n")
fucf.write("0,1,0\n")
fucf.write("0,0,1\n")
fucf.write("#Atoms\n")
fucf.write(str(len(atom_type)) + "\t"+ str(Ntypes)+"\n")
for i in range(0,len(atom_type)):
    fucf.write(str(i) + "\t"+str(xyz[i,0]/a) + "\t"+str(xyz[i,1]/b)+ "\t"+str(xyz[i,2]/c) + "\t" + str(type2[i]) +"\t0\t0\n")
    N_atoms[int(type2[i])] = N_atoms[int(type2[i])] + 1

fucf.write("#Interactions\n")
fucf.write("736\tnormalised-isotropic\n")
fmat.write("material:num-materials=7\n")




#ignores the nonmagnetis Ba,O atoms
magnetictype = type2[type2>1]
magnetic_coords = xyz[type2>1,:]

#predefine the iteration over x,y,z
n=0
xyz_pos = list(itr.product(range(-1, 2), repeat=3))
#iterates over the atoms in two loops i,j to calcaulte the distance between all the atoms to see if they are nearest neighbours
atomj_loop = enumerate(zip(magnetic_coords, magnetictype))
atomj_loop = [[j, mxyz, mtype] for j, (mxyz, mtype) in atomj_loop]

atomi_loop = enumerate(zip(magnetic_coords, magnetictype))
atomi_loop = [[i, mxyz, mtype] for i, (mxyz, mtype) in atomi_loop]

for atomj, mxyz_j, mtype_j in atomj_loop:
    for xyz_p in xyz_pos:
        atomj_xyz = np.add(mxyz_j, xyz_p*aac)
        for atomi, mxyz_i, mtype_i in atomi_loop:
            dxyz = atomj_xyz - mxyz_i
            distance = np.sqrt(sum(dxyz*dxyz))
            #check to see if they are nearest neighbours.
            if(distance < nn_distance[int(mtype_i)][int(mtype_j)] and distance > 0.1):
                nn[int(mtype_i)][int(mtype_j)] = nn[int(mtype_i)][int(mtype_j)] + 1
                fucf.write(str(n) + "\t" +str(atomi) + "\t" + str(atomj) + "\t" + str(xyz_p[0]) + "\t" + str(xyz_p[1]) + "\t" +str(xyz_p[2]) + "\t1\n" )
                n = n + 1

#print statement for verification.
#for i in range(2,7):
#    for j in range(2,7):
#        print (i,j,nn[i][j], N_atoms[i], nn[i][j]/N_atoms[i])

#saves the material parameters + unit cell files
for n in range(0,2):
    i = n+1
    fmat.write("material["+str(i)+"]:material-name="+str(materials[n])+"\n")
    fmat.write("material["+str(i)+"]:material-element="+str(materials[n])+"\n")
    fmat.write("material["+str(i)+"]:damping-constant="+str(1.0)+"\n")
    fmat.write("material["+str(i)+"]:atomic-spin-moment="+str(2.5)+"!muB\n")
    fmat.write("material["+str(i)+"]:unit-cell-category="+str(i)+"\n")
    fmat.write("material["+str(i)+"]:non-magnetic=keep\n")

for n in range(2,7):
    i = n+1
    fmat.write("material["+str(i)+"]:material-name="+str(materials[n])+"\n")
    fmat.write("material["+str(i)+"]:material-element="+str(materials[n])+"\n")
    fmat.write("material["+str(i)+"]:damping-constant="+str(1.0)+"\n")
    fmat.write("material["+str(i)+"]:atomic-spin-moment="+str(2.5)+"!muB\n")
    fmat.write("material["+str(i)+"]:unit-cell-category="+str(i)+"\n")
    for nn in range(2,7):
        j = nn + 1
        exchange = Jnn[n][nn]*conversion
        fmat.write("material["+str(i)+"]:exchange-matrix["+str(j)+"]="+str(exchange)+"\n")
#    fmat.write("material["+str(i)+"]:non-magnetic=keep\n")
