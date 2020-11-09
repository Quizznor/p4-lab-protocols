#!/usr/bin/env py

import matplotlib.pyplot as plt
import numpy as np

# increase all fonts for readability
plt.rcParams.update({'font.size': 28})

compton_continuum = lambda x: 1+1e-4*(x-50/2)**2                      # Using a simple quadratic to simulate the Compton continuum
compton_edge = lambda x,L,x0,k: L/(1+np.exp(k*(x-x0)))-0.007                # Using the logistic function to simulate the Compton edge
photo_peak = lambda x,A,x0,mu: A*np.exp(-(x-x0)**2/mu**2)                   # Using a simple Gaussian-like to simulate the photo peak

E_gamma = 140.757
theta_to_E = lambda t: E_gamma**2/511 * (1-np.cos(2*np.pi*t/360)) / (1+E_gamma/511 * (1-np.cos(2*np.pi*t/360)))

def compton_spectrum(E):
    compton_spectrum = np.zeros(len(E))

    for i,xval in enumerate(E):
        if xval <= 45:
            compton_spectrum[i] = compton_continuum(xval)-0.006
        elif xval > 45 and xval < 55:
            N_max = compton_continuum(47)
            compton_spectrum[i] = compton_edge(xval,N_max,50,1)
        elif xval > 55:
            compton_spectrum[i] = photo_peak(xval,1.9,80,4)

    return compton_spectrum

E = np.linspace(0,130,1000)
compton_values = compton_spectrum(E)

# for theta in np.arange(0,181,30):
#     plt.axvline(theta_to_E(theta),c="grey",label=theta,ls="--")

print(theta_to_E(180))

plt.plot(E,compton_values)
plt.ylim(0,2)
plt.xlim(-0.3,100)

plt.xticks([]), plt.yticks([])
plt.xlabel("electron energy (a.u)",fontsize=35,labelpad=20)
plt.ylabel("electron count (a.u.)",fontsize=35,labelpad=20)

plt.show()
