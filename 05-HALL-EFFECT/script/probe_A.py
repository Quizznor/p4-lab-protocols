#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 20})
plt.rc('axes', labelsize=26)

# Probe A
T, U_cc_plus, U_hall_plus, U_cc_zero, U_hall_zero, U_cc_minus, U_hall_minus = np.loadtxt("../data/hall_A.txt",unpack=True)
I = np.array([1e-3 if T[i] < 50 else 1.8e-3 for i in range(len(T))])    # in Ampere
U_hall = (0.5 * (U_hall_plus - U_hall_minus) - U_hall_zero) * 1e-3      # in Volt
length, width, height = 19e-3, 10e-3, 1e-3                              # in Meter
sigma = ( I/(width*height) ) / ( U_cc_zero/length )                     # in 1/( Ohm meter )
R_hall = ( U_hall/width ) / ( I/(width*height) * -0.5 )                 # in m³/Coulomb
delta_U, delta_I = 0.0005, 0.05e-3                                      # in Volt, Ampere
sigma_err = sigma * np.sqrt( (delta_I/I)**2 + (delta_U/U_cc_zero)**2 )
U_hall_err = np.sqrt( (0.5*delta_U)**2 + (0.5*delta_U)**2 + (delta_U)**2 )
R_hall_err = R_hall * np.sqrt( (U_hall_err/U_hall)**2 + (delta_I/I)**2 )

# # plot everything from first bullet point
# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
#
# ax1.set_xlabel("Temperature ($°$C)")
# ax1.set_ylabel("Conductivity ($\Omega^{-1}\,\mathrm{m}^{-1}$)")
# ax2.set_ylabel(r"Hall coefficient ( $\mathrm{m}^3\,\mathrm{C}^{-1}$ )",labelpad=10)
# ax1.errorbar(T,sigma,xerr=1.5,yerr=sigma_err,ls="none",c="C0",capsize=2)
# ax1.scatter(T,sigma,marker="s",s=10,label="Conductivity",c="C0")
# ax2.errorbar(T,R_hall,xerr=1.5,yerr=R_hall_err,ls="none",c="C1",capsize=2)
# ax2.scatter(T,R_hall,marker="o",s=10,label="Hall coefficient",c="C1")
# fig.legend()

# TODO

plt.show()
