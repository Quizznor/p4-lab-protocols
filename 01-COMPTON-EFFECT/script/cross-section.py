#!/usr/bin/env py

import matplotlib.pyplot as plt
import numpy as np

# increase all fonts for readability
plt.rcParams.update({'font.size': 28})

r0 = 2.81794033e-15 # in m
E0 = 510.998946e-3  # in MeV

E = np.logspace(-4,4,1000)
x = E/E0

sigma_sqrm = 2*np.pi*r0**2 * (0.75*( (1+x)/(x**3) * ( (2*x*(1+x))/(1+2*x) - np.log(1+2*x)) ) + 1/(2*x)*np.log(1+2*x) - (1+3*x)/(1+2*x)**2)
sigma_barn = sigma_sqrm*1e28
plt.plot(x,sigma_barn)
plt.xlabel("Incident photon energy (MeV)",fontsize=35,labelpad=20)
plt.ylabel("Total cross section (b)",fontsize=35,labelpad=20)
plt.xscale("log")
plt.ylim(0,0.55)
plt.xlim(0.001,10000)
plt.tight_layout()
plt.show()
