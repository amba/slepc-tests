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



data, header = open_3d_file("output.dat")

num_evs = data.shape[2] - 2
print("num_evs = ", num_evs)
# cpr for each ky
ky_vals = data[:,0,0]
phi_vals = data[0,:,1]
N_phi = phi_vals.size

eta_vals = []
phi0_vals = []
all_I_vals = []
for block in data:
    F_vals = []
    k_y = block[0,0]
    ev_vals = []
    for line in block:
        all_evs = np.sort(line[2:])
        i_start = np.argmax(all_evs > 0)
        print("i_start = ", i_start, " first ev = ", all_evs[i_start])
        
        evs = all_evs[i_start:i_start + int(num_evs/2) - 2]
        ev_vals.append(evs)
        # evs = evs[0:32]
        F_vals.append(-np.sum(evs))
    F_vals = np.array(F_vals)
    phi0_vals.append(phi_vals[np.argmin(F_vals)])
    #plt.plot(phi_vals, F_vals, label="k_y = %g" % k_y)
    I_vals = np.gradient(F_vals)
    all_I_vals.append(I_vals)
    I_max = np.amax(I_vals)
    I_min = np.amin(I_vals)
    eta = (I_max + I_min) / (I_max + np.abs(I_min))
    eta_vals.append(eta)
  #  plt.plot(phi_vals / np.pi, ev_vals)
 #   plt.grid()
#    plt.show()
    #plt.plot(phi_vals / np.pi, F_vals, '.', label="disorder = %g" % k_y)
    plt.plot(phi_vals, np.gradient(F_vals), label="disorder = %g" % k_y)

plt.xlabel('phi / π')
plt.ylabel('I (a.u.)')
plt.legend()
plt.grid()
plt.show()

