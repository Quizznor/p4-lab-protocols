#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True

from mössbauer_aux import print_results
from mössbauer_aux import perform_fits
from mössbauer_aux import create_bins
from mössbauer_aux import draw_cuts
from mössbauer_aux import draw_fits
from mössbauer_aux import inv_bw
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 24})
plt.rc('axes', labelsize=28)

channels, data = np.loadtxt("../data/Vacromium.txt",unpack=True)
(ch1, ch2), (d1, d2) = np.split(channels,2), np.split(data,2)

# ch1_binned, ch2_binned = create_bins(ch1), create_bins(ch2)-512
# d1_binned, d1_err_binned = create_bins(d1,np.sqrt(d1))
# d2_binned, d2_err_binned = create_bins(d2,np.sqrt(d2))
# d2_binned, d2_err_binned = d2_binned[::-1], d2_err_binned[::-1]

ch1_binned, ch2_binned = ch1, ch2
d1_binned, d1_err_binned = d1[::-1], np.sqrt(d1[::-1])
d2_binned, d2_err_binned = d2[::-1], np.sqrt(d2[::-1])

plt.scatter(ch1_binned,d1_binned,marker="s",s=5,label="Ch1 data",zorder=50)
plt.scatter(ch2_binned,d2_binned,marker="o",s=10,label="Ch2 data",zorder=50)
plt.errorbar(ch1_binned,d1_binned,yerr=d1_err_binned,ls="None",capsize=1.5,elinewidth=0.4,zorder=49)
plt.errorbar(ch2_binned,d2_binned,yerr=d2_err_binned,ls="None",capsize=1.5,elinewidth=0.4,zorder=49)

# eyeballing this works good enough, the below cuts are given for res = 128
d1_cuts = [0,-1]
d2_cuts = [0,-1]
# draw_cuts(ch1_binned,d1_cuts)
# draw_cuts(ch2_binned,d2_cuts)

fits_d1 = perform_fits(ch1_binned,d1_binned,d1_err_binned,d1_cuts,ch1=True)
fits_d2 = perform_fits(ch2_binned,d2_binned,d2_err_binned,d2_cuts,ch1=False)

draw_fits(ch1_binned,fits_d1,d1_cuts,ch1=True)
draw_fits(ch2_binned,fits_d2,d2_cuts,ch1=False)

print("\nChannel 1:")
print_results(fits_d1)

print("\nChannel 2:")
print_results(fits_d2)

plt.xlabel("MCS channel number",labelpad=20)
plt.ylabel("Binned count",labelpad=20)
# plt.ylim(8500,10800)
# plt.xlim(0,512)
plt.legend()
plt.show()
