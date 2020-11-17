#!/usr/bin/env py

import glob
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 16})
plt.rc('axes', labelsize=26)

background = glob.glob("../data/background_*.txt")
compton = glob.glob("../data/compton_*.txt")

# handle data (ecal data stems from ./ecal-gauge.py)
raw_to_rate = lambda x: np.sum(x)/300.
# E = np.poly1d([5.8435111939455116, 30.19841333330094])
# COV = [[ 7.35545507e-02,-5.20139779e+00],[-5.20139779e+00,4.83552687e+02]]
# E_err = lambda x: np.sqrt( np.array([x,1]).T @ COV @ np.array([x,1]) )

rates = []
for i in range(len(background)):
    angle = int(i*10)
    channel, bkg = np.loadtxt(background[i],unpack=True)
    channel, sig = np.loadtxt(compton[i],unpack=True)
    rates.append(raw_to_rate(sig) - raw_to_rate(bkg))

# solid angle
r_crystal = 0.5*2.55                                    # cm
d_target_crystal = 21.5                                 # cm
dOmega = np.pi*(r_crystal)**2/(d_target_crystal)**2     # sr

# radiative flux
years_since_1971 = 49.49                                # years
phi0 = 10000 * 1.54e6                                   # 1/(m²s)
phi_now = phi0*np.exp(-years_since_1971*np.log(2)/30)   # 1/(m²s)

# collision partners
d_target = 0.01                                         # m
l_target = 0.01                                         # m
rho_al = 2700                                           # kg/m³
N_a = 6.022e13
A_al, Z_al = 26.982, 13
n = N_a/A_al * Z_al * rho_al * np.pi * (d_target/2)**2 * l_target

# detector efficiency
energies = np.arange(250,601,50)
epsilon = 1/np.array([1.29,1.40,1.54,1.68,1.80,1.90,2.00,2.07])

params, cov = np.polyfit(energies,epsilon, 2, cov=True)
efficiency_err = lambda x: np.sqrt( np.array([x**2,x,1]).T @ cov @ np.array([x**2,x,1]) )
efficiency = np.poly1d(params)

X = np.linspace(250,600,1000)
upper = efficiency(X) + efficiency_err(X)
lower = efficiency(X) - efficiency_err(X)
plt.plot(X,efficiency(X))
plt.fill_between(X,upper,lower,alpha=0.4)
plt.scatter(energies,epsilon)

plt.show()

# plt.plot(np.arange(10,110,10),rates[1:])
# plt.show()
