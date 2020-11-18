#!/usr/bin/env py

# Don't create a pycache directory from importing auxiliary files
import sys
sys.dont_write_bytecode = True

# main program
import numpy as np
import matplotlib.pyplot as plt
from detector_efficiency import efficiency, efficiency_err
from main_measurement import channels, data_bkg, data_sig
from ecal_gauge import E, E_err

raw_to_rate = lambda x: np.sum(x)/300.
rad = lambda x: 2*np.pi * x/360
angles = np.arange(0,101,10)
E_gamma, m_e = 661.659, 510.999
rates, rates_err = [], []

for angle in angles:

    sig = data_sig[int(angle/10)]
    bkg = data_bkg[int(angle/10)]

    # normalise w.r.t detector efficiency
    for channel in channels:
        energy_theo = E_gamma/(1 + E_gamma/m_e * (1-np.cos(rad(angle))))
        energy, energy_err = E(channel), E_err(channel)
        sig_corr, bkg_corr = sig/energy, bkg/energy
        sig_corr_err = sig/energy**2 * E_err(channel)
        bkg_corr_err = bkg/energy**2 * E_err(channel)

    rates.append(raw_to_rate(sig_corr) - raw_to_rate(bkg_corr) )

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

dsigma_dOmega = np.array(rates)/dOmega * 1/(phi_now * n) # 1/s 1/sr m²s = m²/sr

# theoretical values from Klein-Nishina
angle_theo = np.arange(0,121,10)
kn_theo = np.array([0.7952,0.7536,0.6465,0.5132,0.3887,0.2904,0.2208,0.1753,0.1472,0.1309,0.1222,0.1180,0.1166])*1e-25 # cm²

plt.plot(angles[2:],dsigma_dOmega[2:])
plt.plot(angle_theo,kn_theo)
plt.show()
