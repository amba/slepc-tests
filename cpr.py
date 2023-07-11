#!/usr/bin/env python3 

import numpy as np
import sys
import glob
import argparse
import io
import os.path
import sys
import matplotlib.pyplot as plt
import scipy.signal


if np.__version__ < '1.14.1':
    sys.exit("numpy version " + np.__version__ + " is too old")
    
def open_3d_file(file):
    fh = open(file, 'r')
    header = fh.readline().rstrip()
    contents = fh.read().rstrip()
    
    list_of_blocks = contents.split("\n\n")
    num_blocks = len(list_of_blocks)
    arrays = []
    for i, block in enumerate(list_of_blocks):
#        print("reading block %d / %d" % (i, num_blocks))
        arrays.append(np.genfromtxt(io.StringIO(block)))
    first_shape = arrays[0].shape
    for i in range(len(arrays)-1, -1, -1):
        shape = arrays[i].shape
        if shape != first_shape:
            print("block ", i, " with first line", arrays[i][0], " does not match :", shape, " != ", first_shape)
            del arrays[i]
    return np.stack(arrays), header


def save_3d_file(output_file, data, header):
    fh = open(output_file, 'w')
    fh.write(header + "\n")
    for block in data:
        np.savetxt(fh, block, fmt="%.17g", delimiter="\t")
        fh.write("\n")
    fh.close()



data, header = open_3d_file("output-spin.dat")

num_evs = data.shape[2] - 2
print("num_evs = ", num_evs)
# cpr for each ky
ky_vals = data[:,0,0]
phi_vals = data[0,:,1]
N_phi = phi_vals.size

eta_vals = []
phi0_vals = []
for block in data:
    F_vals = []
    k_y = block[0,0]
    for line in block:
        all_evs = np.sort(line[2:])
        evs = -all_evs[int(num_evs/2):]
       # evs = evs[0:32]
        F_vals.append(np.sum(evs))
    F_vals = np.array(F_vals)
    phi0_vals.append(phi_vals[np.argmin(F_vals)])
    #plt.plot(phi_vals/np.pi, F_vals, label="k_y = %g" % k_y)
    I_vals = np.gradient(F_vals)
    I_max = np.amax(I_vals)
    I_min = np.amin(I_vals)
    eta = (I_max + I_min) / (I_max + np.abs(I_min))
    eta_vals.append(eta)
    print("k_y = %.2g, η = %.2g\n" % (k_y, eta))
    plt.plot(phi_vals/np.pi, np.gradient(F_vals), label="k_y = %g" % k_y)
plt.xlabel('phi / π')
plt.ylabel('F/Δ')
plt.legend()
plt.grid()
plt.show()



phi0_vals = np.array(phi0_vals)
plt.xlabel('k_y')
plt.ylabel('phi0 / π')
plt.plot(ky_vals, phi0_vals / np.pi)
plt.show()
