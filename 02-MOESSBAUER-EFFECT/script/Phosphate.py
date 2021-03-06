#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True

from mössbauer_aux import print_results
from mössbauer_aux import perform_fits
from mössbauer_aux import create_bins
from mössbauer_aux import calc_Egrad
from mössbauer_aux import draw_cuts
from mössbauer_aux import draw_fits
from mössbauer_aux import inv_bw
from mössbauer_aux import C_to_v
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 24})
plt.rc('axes', labelsize=28)

channels, data = np.loadtxt("../data/Phosphate.txt",unpack=True)
(ch1, ch2), (d1, d2) = np.split(channels,2), np.split(data,2)

ch1_binned, ch2_binned = create_bins(ch1), create_bins(ch2)-512
d1_binned, d1_err_binned = create_bins(d1,np.sqrt(d1))
d2_binned, d2_err_binned = create_bins(d2,np.sqrt(d2))
d2_binned, d2_err_binned = d2_binned[::-1], d2_err_binned[::-1]
v1_binned, v2_binned = C_to_v(ch1_binned), C_to_v(ch2_binned)

plt.scatter(v1_binned,d1_binned,marker="s",s=5,label="Ch1 data",zorder=50)
plt.scatter(v2_binned,d2_binned,s=10,label="Ch2 data",zorder=50)
plt.errorbar(v1_binned,d1_binned,yerr=d1_err_binned,ls="None",capsize=1.5,elinewidth=0.4,zorder=49)
plt.errorbar(v2_binned,d2_binned,yerr=d2_err_binned,ls="None",capsize=1.5,elinewidth=0.4,zorder=49)

# eyeballing this works good enough, the below cuts are given for res = 128
d1_cuts = [0,62,-1]
d2_cuts = [0,62,-1]
# draw_cuts(ch1_binned,d1_cuts)
# draw_cuts(ch2_binned,d2_cuts)

fits_d1 = perform_fits(v1_binned,d1_binned,d1_err_binned,d1_cuts,ch1=True)
fits_d2 = perform_fits(v2_binned,d2_binned,d2_err_binned,d2_cuts,ch1=False)

draw_fits(v1_binned,fits_d1,d1_cuts,ch1=True,override=[(-1.5,4600),(0.5,4800)])
draw_fits(v2_binned,fits_d2,d2_cuts,ch1=False)

print_results(fits_d1,fits_d2,table=False)

v_12_Ch1, v_32_Ch1 = fits_d1[0][0][-1], fits_d1[1][0][-1]
v_12_Ch2, v_32_Ch2 = fits_d2[0][0][-1], fits_d2[1][0][-1]
v_32, v_12 = np.mean([v_32_Ch1,v_32_Ch2]), np.mean([v_12_Ch1,v_12_Ch2])
v_32_err, v_12_err = np.std([v_32_Ch1,v_32_Ch2]), np.std([v_12_Ch1,v_12_Ch2])
dVdz, dVdz_err = calc_Egrad(v_32,v_12,v_32_err,v_12_err)

print(dVdz,dVdz_err)

plt.xlabel(r"$\gamma$-source velocity ($\frac{\mathrm{mm}}{\mathrm{s}}$)",labelpad=20)
plt.ylabel("Binned count",labelpad=20)
plt.xlim(-10.1,10.1)
plt.legend()
# plt.show()
