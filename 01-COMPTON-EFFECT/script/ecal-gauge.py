#!/usr/bin/env py

import matplotlib.pyplot as plt
from scipy import optimize
import numpy as np

plt.rcParams.update({'font.size': 20})
plt.rc('axes', labelsize=26)
datadir = "../data"

channels_Co57, counts_Co57 = np.genfromtxt(f"{datadir}/gauge_Co-57.txt",unpack=True,max_rows=300)
channels_Co60, counts_Co60 = np.genfromtxt(f"{datadir}/gauge_Co-60.txt",unpack=True,max_rows=300)
channels_Na22, counts_Na22 = np.genfromtxt(f"{datadir}/gauge_Na-22.txt",unpack=True,max_rows=300)
channels_Cs137, counts_Cs137 = np.genfromtxt(f"{datadir}/gauge_Cs-137.txt",unpack=True,max_rows=300)

# split the data into the signal photopeak region
split = lambda x,low,high: x[low:high]

Co57_cut, peak_Co57 = split(channels_Co57,0,30), split(counts_Co57,0,30)
# Co60_cut, peak_Co60 = split(channels_Co60,5,56), split(counts_Co60,5,56)
Na22_cut, peak_Na22 = split(channels_Na22,70,110), split(counts_Na22,70,110)
Cs137_cut, peak_Cs137 = split(channels_Cs137,87,122), split(counts_Cs137,87,122)

# fit a gaussian to the photopeak
def gaussian(x,N,mu,sigma,offset):
    return N/np.sqrt(2*np.pi*sigma**2) * np.exp(-0.5*(x-mu)**2/sigma**2) + offset
def gaussian_no_offset(x,N,mu,sigma):
    return N/np.sqrt(2*np.pi*sigma**2) * np.exp(-0.5*(x-mu)**2/sigma**2)

params_Co57, cmat = optimize.curve_fit(gaussian_no_offset,Co57_cut,peak_Co57,p0=[1e3,14,4])
# params_Co60, cmat = optimize.curve_fit(gaussian,Co60_cut,peak_Co60,p0=[300,22,10,54])
params_Na22, cmat = optimize.curve_fit(gaussian,Na22_cut,peak_Na22,p0=[100,98,4,0])
params_Cs137, cmat = optimize.curve_fit(gaussian,Cs137_cut,peak_Cs137,p0=[150,104,4,100])

print(cmat)

# plot the different spectra and each gauge measurement
ax1 = plt.subplot2grid((5, 1), (0, 0),rowspan=3)
ax2 = plt.subplot2grid((5, 1), (3, 0),rowspan=2, sharex=ax1)

ax1.plot(channels_Co57,counts_Co57,c="red",alpha=0.4)
ax1.plot(channels_Co60,counts_Co60,c="orange",alpha=0.4,label="Cobalt-60")
ax1.plot(channels_Na22,counts_Na22,c="green",alpha=0.4)
ax1.plot(channels_Cs137,counts_Cs137,c="blue",alpha=0.4)

X_Co57 = np.linspace(min(Co57_cut),max(Co57_cut),1000)

ax1.plot(X_Co57,gaussian_no_offset(X_Co57,*params_Co57),c="red",label="Cobalt-57")
# ax1.plot(Co60_cut,gaussian(Co60_cut,*params_Co60),c="orange",label="Cobalt-60")
ax1.plot(Na22_cut,gaussian(Na22_cut,*params_Na22),c="green",label="Sodium-22")
ax1.plot(Cs137_cut,gaussian(Cs137_cut,*params_Cs137),c="blue",label="Caesium-137")
ax1.legend()


# Perform the MAC calibration
channels = [params_Co57[1],params_Na22[1],params_Cs137[1]]
energies = [122.061,546.544,661.659]  # keV

X = np.linspace(0,300,300)
params, cov = np.polyfit(channels, energies, 1, cov=True)
linear_error = lambda x: np.sqrt( np.array([x,1]).T @ cov @ np.array([x,1]) )
MAC_calibration = np.poly1d(params)
upper = MAC_calibration(X)+linear_error(X)
lower = MAC_calibration(X)-linear_error(X)

for i,channel in enumerate(channels):
    ax1.axvline(channel,ls="--",c="black")
    ax2.axvline(channel,ls="--",c="black")


ax2.plot(X,MAC_calibration(X),zorder=1)
ax2.fill_between(X,lower,upper,alpha=0.4)
ax2.scatter(channels,energies,s=70,c="black",zorder=2)

ax1.axvline(195.560,c="k",ls="--",alpha=0.4)
ax1.axvline(222.777,c="k",ls="--",alpha=0.4)
ax2.axvline(195.560,c="k",ls="--",alpha=0.4)
ax2.axvline(222.777,c="k",ls="--",alpha=0.4)

ax1.set_xlim(0,300)
ax1.set_ylim(0,1250)
ax2.set_ylim(0,2000)
ax1.set_ylabel("Electron count")
ax2.set_ylabel("Energy (keV)")
ax2.set_xlabel("MAC Channel number")
plt.subplots_adjust(wspace=0, hspace=0.7)
plt.tight_layout()
plt.show()
