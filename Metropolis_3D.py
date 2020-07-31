                                                                      # -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 21:41:35 2019

@author: dawid
"""

import numpy as np
import math as math


n = 10000   #Number of trials
s = 16   #Number of particles
L = [100, 100, 100]  #Dimensions of box



"""   
Pair-wise potential
"""
def V(r):
    R = 20.
    if r == 0:
        return 0.
    else:
        return math.exp(-r**2/R)



"""
Hamiltonian of the ensemble
Computes the total energy of the system
Needed to find the probability density from the Boltzmann distribution
"""
def H(X):
    global E; E = 0
    for i in range(s):
        j = i+1
        while j < s:
            x_dist = min(np.absolute(X[i][0]-X[j][0]), np.absolute(X[i][0]-X[j][0]+L[0]), np.absolute(X[i][0]-X[j][0]-L[0]))
            y_dist = min(np.absolute(X[i][1]-X[j][1]), np.absolute(X[i][1]-X[j][1]+L[1]), np.absolute(X[i][1]-X[j][1]-L[1]))
            z_dist = min(np.absolute(X[i][2]-X[j][2]), np.absolute(X[i][2]-X[j][2]+L[2]), np.absolute(X[i][2]-X[j][2]-L[2]))
            dist = np.sqrt(x_dist**2 + y_dist**2 + z_dist**2)
            E += V(dist)
            j += 1
    return E



"""
Sampling function: given a current location x, 
propose a new candidate location x2
"""
def new(X):
    X2 = np.zeros((s,3))
    for i in range(s):
        for j in range(3):
            X2[i][j] = X[i][j] + 5*(np.random.rand()-0.5)
    return X2



"""
Metropolis algorithm for the canonical ensemble
Returns the total energy of the system with error and list of positions 
with each trial
"""
def Metropolis_Boltzmann(n):   
    #Initial positions:
    X = np.zeros((s,3))
    seed = L[0]*np.random.uniform()
    span = 10*(np.random.uniform()-0.5)
    for i in range(s):
        for j in range(3):
            X[i][j] += seed + span*np.random.uniform()
            if 0. < X[i][j] < L[j]:
                continue
            elif X[i][j] < 0.:
                X[i][j] += L[j]
            else:
                X[i][j] -= L[j]    
    positions = [X]
    
    count = 0
    #Perform Metropolis sampling:
    for i in range(n):
        if count == 511:
            break
        X2 = new(X)
        for i in range(s):
            for j in range(3):
                if 0. < X2[i][j] < L[j]:
                    continue
                elif X2[i][j] < 0.:
                    X2[i][j] += L[j]
                else:
                    X2[i][j] -= L[j]
        if np.random.uniform() < np.minimum(1.0, np.exp(H(X)-H(X2))):
            X = X2
            positions.append(X)
            count += 1
          
    return positions


data = []
batch = 0

data_points = 250   #excludes time-reversed data; added in automatically
for batch in range(data_points):
    positions = Metropolis_Boltzmann(n)

    P = np.array(positions)
    
    positions_reverse = np.flip(positions, 0)
    PR = np.array(positions_reverse)
    
    data.append(P); data.append(PR)
    print(2*batch+2, "/", 2*data_points)

data = np.array(data)
np.save("data", data)
print(data.shape)


labels = np.zeros(2*data_points)
labels[::2] = 1
np.save("labels", labels)