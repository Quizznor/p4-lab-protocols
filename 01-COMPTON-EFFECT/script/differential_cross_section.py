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

rad = lambda x: 2*np.pi * x/360

energy, energy_err = E(channels), E_err(channels)
E_gamma, m_e = 661.659, 510.999
angles = np.arange(20,101,10)
rates, rates_err = [], []

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
A_crystal = np.pi* (2.55/2)**2           # cm²
d_target_crystal = 21.5                  # cm
dOmega = A_crystal/(d_target_crystal)**2 # sr
dOmega_err = 0


# radiative flux (assum phi_0 was measured on 15th july 1971)
days_since_phi_0 = 18023
days_since_phi_0_err = 15
half_lives = days_since_phi_0 / (30.17 * 365.25)
ratio_remaining = np.power(0.5, half_lives )
phi_0 =  1.54e6 * 1e4 * ratio_remaining     # per m²s
phi_0_err = 0.09e6 * 1e4 * ratio_remaining  # per m²s

# volume target
d_target, l_target = 1.0, 1.0      # cm
d_target_err, l_target_err = 0.05,0.05
A_target = np.pi * (d_target/2)**2 # cm²
A_target_err = np.pi*(d_target/2)*d_target_err
V_target = A_target * l_target     # cm³
V_err = np.sqrt( (A_target*l_target_err)**2
               + (l_target*A_target_err)**2 )

# collision partners
rho_al, N_a = 2.7, 6.022e23     # g/cm³, 1/mol
A_al, Z_al = 26.982, 13         # g/mol, 1
N_mole = N_a/A_al * Z_al        # 1/g
n =  N_mole * rho_al * V_target # 1
n_err = n * V_err/V_target      # 1

# Putting it all together for cross section
rates, rates_err = np.array(rates), np.array(rates_err)
denumerator = dOmega * phi_0 * n
KN_exp = rates/denumerator *1e4 # cm²
KN_exp_err = 1e4 * np.sqrt((rates_err/denumerator )**2 +
    ( rates/denumerator * (n_err/n + phi_0_err/phi_0) )**2)

# theoretical values from Klein-Nishina
angle_theo, _, KN_theo = np.loadtxt("../data/klein_nishina_theo.txt",unpack=True)

if __name__ == '__main__':
    print(f"\ndOmega = {dOmega:.5f} +- 0 sr")
    print(f"phi_0  = {phi_0:0.3e} +- {phi_0_err:0.3e} 1/m²s")
    print(f"V_target = {V_target:0.3f} +- {V_err:0.3f} cm³")
    print(f"n_elec = {n:0.3e} +- {n_err:0.3e} electrons")

    plt.plot(angle_theo,KN_theo,label="Klein-Nishina")
    plt.errorbar(angles,KN_exp*1e25,xerr=1,yerr=KN_exp_err*1e25,ls="none",label="measured")

    plt.xlabel(r"scattering angle ($^\circ$)")
    plt.ylabel(r"$\frac{\mathrm{d}\sigma}{\mathrm{d}\Omega} (10^{-25}$ cm)")
    plt.legend()
    plt.show()
