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
from scipy.optimize import curve_fit
import glob
from matplotlib import cm

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


# data_file = sys.argv[1]
# if not data_file:
#     data_file = 'output.dat'
datafile = sys.argv[1]

data, header = open_3d_file(datafile)

num_evs = data.shape[2] - 1
print("num_evs = ", num_evs)
# cpr for each ky
phi_vals = data[0,:,0]
N_phi = phi_vals.size

def plot_cpr(block, use_num_evs):
    F_vals = []
    ev_vals = []
    for line in block:
        all_evs = np.sort(line[1:])
        i_start = np.argmax(all_evs > 0)
  #      print("i_start = ", i_start, " first ev = ", all_evs[i_start])

        evs = all_evs[i_start:i_start + int(use_num_evs/2) - 2]
        ev_vals.append(evs)
        # evs = evs[0:32]
        F_vals.append(-np.sum(evs))
    F_vals = np.array(F_vals)
    imin = np.argmin(F_vals)
    I_vals = np.gradient(F_vals)
    Icp = np.amax(I_vals)
    Icm = np.amin(I_vals)
    eta = (Icp + Icm) / (Icp - Icm)
    print("num_evs = %d, eta = %g" % (use_num_evs, eta))
    #print("data block: ", data_block)
    #p0 = [0.8, np.amax(I_vals)]
    #p0 = [np.amax(I_vals),]
    #istart = np.argmin(phi_vals < 0)
    #iend = np.argmin(phi_vals[istart+1:] < 1)
    # fit = curve_fit(cpr_KO1, phi_vals+0.0001, I_vals, p0 = p0)
    # fit_tau = curve_fit(cpr, phi_vals, I_vals, p0=[0.9, np.amax(I_vals)])
    # print(fit)

    #    plt.plot(phi_vals / np.pi, ev_vals)
    #    plt.grid()
    #    plt.show()
    #    plt.plot(phi_vals, F_vals, '.', label="disorder = %g" % k_y)
    #I_vals = np.gradient(F_vals)
    # I_vals /= np.amax(I_vals)
    #plt.plot(phi_vals, F_vals, '.', label=file)
    plt.plot(phi_vals, I_vals, '.', c=cm.plasma(use_num_evs / num_evs))
    #plt.plot(phi_vals, cpr_KO1(phi_vals+0.0001, *fit[0]), label="KO1-fit")
#plt.plot(phi_vals, cpr(phi_vals, *fit_tau[0]), label="tau = %g" % fit_tau[0][0])
# print("data: ", output_data)


for use_evs in range(10,num_evs,10):
    plot_cpr(data[0], use_evs)
plt.legend()
plt.grid()
plt.show()
