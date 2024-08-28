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
    # first_shape = arrays[0].shape
    # for i in range(len(arrays)-1, -1, -1):
    #     shape = arrays[i].shape
    #     if shape != first_shape:
    #         print("block ", i, " with first line", arrays[i][0], " does not match :", shape, " != ", first_shape)
    #         del arrays[i]
    return arrays, header


def save_3d_file(output_file, data, header):
    fh = open(output_file, 'w')
    fh.write(header + "\n")
    for block in data:
        np.savetxt(fh, block, fmt="%.17g", delimiter="\t")
        fh.write("\n")
    fh.close()

filename = sys.argv[1]

data, header = open_3d_file(filename)

#num_evs = data.shape[2] - 2
#print("num_evs = ", num_evs)
# cpr for each ky
# ky_vals = data[:,0,0]
# phi_vals = data[0,:,1]
# N_phi = phi_vals.size

block = data[0]
F_vals = []
phi_vals = block[:,0]
disorder = block[0,0]
num_evs = block.shape[1] - 1
ev_vals = []
ev_vals_hist = []
max_ev = np.amax(block[:,1:])
print("max_ev = ", max_ev)

for line in block:
    all_evs = np.sort(line[1:])
    i_start = np.argmax(all_evs > 0)
    #print("i_start = ", i_start, " first ev = ", all_evs[i_start])

    evs = all_evs[i_start:i_start + int(num_evs/2)]
    evs_hist,edges = np.histogram(evs, range=(0,max_ev+0.1), bins=401)
    ev_vals.append(evs)
    ev_vals_hist.append(evs_hist)
plt.plot(phi_vals, ev_vals,color='black')
plt.title(filename)
plt.grid()
plt.ylim((0,max_ev+0.1))
plt.show()
ev_density = np.array(ev_vals_hist).swapaxes(0,1)
plt.imshow(ev_density,origin='lower', cmap='Greys')
plt.show()

