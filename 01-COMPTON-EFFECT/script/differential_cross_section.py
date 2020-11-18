#!/usr/bin/env py

# Don't create a pycache directory from importing auxiliary files
import sys
sys.dont_write_bytecode = True

# main program
import numpy as np
import matplotlib.pyplot as plt
from detector_efficiency import efficiency, efficiency_err
from main_measurement import channels, data_bkg, data_sig
from ecal_gauge import E, E_err, gaussian, Cs137_std
from scipy.signal import find_peaks

rad = lambda x: 2*np.pi * x/360

energy, energy_err = E(channels), E_err(channels)
E_gamma, m_e = 661.659, 510.999
angles = np.arange(20,101,10)
rates, rates_err = [], []
Cs137_std = E(Cs137_std)

for angle in angles:

    sig = np.array(data_sig[int(angle/10)])
    bkg = np.array(data_bkg[int(angle/10)])
    channel = np.argmax(sig - bkg)
    energy_theo, energy_theo_err = E(channel), E_err(channel)#
    N, N_err = np.sum(sig - bkg), np.sum(sig + bkg)

    rates.append( N / efficiency(energy_theo) / 300 )
    rates_err.append( 1/300 * np.sqrt( (np.sum(sig) + np.sum(bkg))/efficiency(energy_theo)
    + (N/efficiency(energy_theo) * efficiency_err(energy_theo))**2 ) )

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

# theoretical values from Klein-Nishina
angle_theo, _, KN_theo = np.loadtxt("../data/klein_nishina_theo.txt",unpack=True)
KN_theo*=1e-25                                    # m²

KN_exp = np.array(rates)/dOmega * 1/(phi_now * n)       # 1/s 1/sr m²s = m²/sr

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

# ax1.scatter(angles[2:],KN_exp[2:])
ax1.errorbar(angles,KN_exp)
ax2.plot(angle_theo,KN_theo)
plt.show()
