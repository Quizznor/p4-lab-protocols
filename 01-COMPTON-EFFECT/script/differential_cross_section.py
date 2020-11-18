#!/usr/bin/env py

# Don't create a pycache directory from importing auxiliary files
import sys
sys.dont_write_bytecode = True

# main program
import numpy as np
import matplotlib.pyplot as plt
from detector_efficiency import efficiency, efficiency_err
from main_measurement import channels, data_bkg, data_sig

raw_to_rate = lambda x: np.sum(x)/300.
angles = np.arange(0,101,10)
rates = []

for angle in angles:

    sig = data_sig[int(angle/10)]
    bkg = data_bkg[int(angle/10)]

    # TODO: Normalise each channel count to detector accuracy

    # rates.append(raw_to_rate() - raw_to_rate() )

# solid angle
r_crystal = 0.5*2.55                                    # cm
d_target_crystal = 21.5                                 # cm
dOmega = np.pi*(r_crystal)**2/(d_target_crystal)**2     # sr

# radiative flux
years_since_1971 = 49.49                                # years
phi0 = 10000 * 1.54e6                                   # 1/(m²s)
phi_now = phi0*np.exp(-years_since_1971*np.log(2)/30)   # 1/(m²s)

# collision partners
d_target, l_target = 0.01, 0.01                         # m
rho_al, N_a = 2700, 6.022e13                            # kg/m³
A_al, Z_al = 26.982, 13
n = N_a/A_al * Z_al * rho_al * np.pi * (d_target/2)**2 * l_target

dsigma_dOmega = np.array(rates)/dOmega * 1/(phi_now * n)

plt.plot(angles[2:],dsigma_dOmega[2:])
plt.show()
