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

channels, data = np.loadtxt("../data/Vacromium.txt",unpack=True)
(ch1, ch2), (d1, d2) = np.split(channels,2), np.split(data,2)

ch1_binned, ch2_binned = create_bins(ch1), create_bins(ch2)
d1_binned, d1_err_binned = create_bins(d1,np.sqrt(d1))
d2_binned, d2_err_binned = create_bins(d2,np.sqrt(d2))

plt.scatter(ch1_binned,d1_binned,marker="s",s=5), plt.scatter(ch2_binned,d2_binned,s=5)
plt.errorbar(ch1_binned,d1_binned,yerr=d1_err_binned,ls="None",capsize=1.5,elinewidth=0.4)
plt.errorbar(ch2_binned,d2_binned,yerr=d2_err_binned,ls="None",capsize=1.5,elinewidth=0.4)

# eyeballing this works good enough, the below cuts are given for res = 128
d1_cuts = [0,-1]
d2_cuts = [0,-1]
# draw_cuts(ch1_binned,d1_cuts)
# draw_cuts(ch2_binned,d2_cuts)

fits_d1 = perform_fits(ch1_binned,d1_binned,d1_err_binned,d1_cuts)
fits_d2 = perform_fits(ch2_binned,d2_binned,d2_err_binned,d2_cuts)
draw_fits(ch1_binned,fits_d1,d1_cuts)
draw_fits(ch2_binned,fits_d2,d2_cuts)

print("\nChannel 1:")
print_results(fits_d1)

print("\nChannel 2:")
print_results(fits_d2)

plt.xlabel("MCS channel number")
plt.ylabel("Binned count")
plt.show()
