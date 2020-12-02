#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from calibration import C_to_E

channel, data = np.loadtxt("../data/Sulfate.txt",unpack=True)
energy, energy_err = C_to_E(channel)
data1, data2 = np.split(data,2)
E1, E2 = np.split(energy,2)
E1_err, E2_err = np.split(energy_err,2)

def create_bins(x,x_err,resolution=64):
    # It's implicitly assumed that this function receives a 512-long input
    # array. Other sizes might break some behaviour at the bin borders
    bins, bins_err = np.zeros(resolution), np.zeros(resolution)
    bin_contents = np.split(x,resolution)
    bin_contents_err = np.split(x_err,resolution)

    for i in range(resolution):
        bins[i]+=np.sum(bin_contents[i])
        bins_err[i]+=np.sqrt( np.sum(bin_contents_err[i]**2) )

    return bins, bins_err

def inv_breit_wigner(x,O,A,w,g):
    return O - A/( (x**2-w**2)**2 + g**2*w**2)

E1_binned, E1_binned_err = create_bins(E1,E1_err,resolution=32)
data1_binned, data1_binned_err = create_bins(data1,np.sqrt(data1)/len(data1),resolution=32)

plt.errorbar(E1_binned,data1_binned,yerr=data1_binned_err*40,ls="None",capsize=2)
# plt.errorbar(E1,data1,xerr=E1_err,yerr=np.sqrt(data1),ls="None")
plt.show()
