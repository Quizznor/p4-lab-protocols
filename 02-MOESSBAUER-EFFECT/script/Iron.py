#!/usr/bin/env python3

import matplotlib.pyplot as plt
from calibration import C_to_E
import numpy as np

channels, data = np.loadtxt("../data/Iron.txt",unpack=True)
ch1, ch2 = np.split(channels,2)
d1, d2 = np.split(data,2)

def create_bins(x,x_err=None,res=128):

    bin_content = np.split(x,res)
    bins = np.zeros(res)

    if x_err is not None:
        bin_content_err = np.split(x_err,res)
        bins_err = np.zeros(res)

    for i in range(res):
        if x_err is not None:
            bins[i] = np.sum(np.array(bin_content[i]))
            bins_err = np.sqrt( np.sum(np.array(bin_content_err)**2) )
        else:
            bins[i] = np.mean(np.array(bin_content[i]))

    if x_err is not None:
        return bins, bins_err
    else:
        return bins

ch1_binned, ch2_binned = create_bins(ch1), create_bins(ch2)
d1_binned,d1_err_binned = create_bins(d1,np.sqrt(d1))
d2_binned,d2_err_binned = create_bins(d2,np.sqrt(d2))

plt.errorbar(ch1_binned,d1_binned,yerr=d1_err_binned,ls="None",capsize=1.5)
plt.errorbar(ch2_binned,d2_binned,yerr=d2_err_binned,ls="None",capsize=1.5)

plt.show()
