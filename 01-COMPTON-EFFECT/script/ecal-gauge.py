#!/usr/bin/env py

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

plt.rcParams.update({'font.size': 28})
datadir = "../data"

channels_Co57, counts_Co57 = np.genfromtxt(f"{datadir}/gauge_Co-57.txt",unpack=True,max_rows=300)
channels_Co60, counts_Co60 = np.genfromtxt(f"{datadir}/gauge_Co-60.txt",unpack=True,max_rows=300)
channels_Na22, counts_Na22 = np.genfromtxt(f"{datadir}/gauge_Na-22.txt",unpack=True,max_rows=300)
channels_Cs137, counts_Cs137 = np.genfromtxt(f"{datadir}/gauge_Cs-137.txt",unpack=True,max_rows=300)

# plot the different spectra for each gauge measurement
#plt.figure()
#plt.title("Calibration of the MAC")
#plt.plot(channels_Co57,counts_Co57,label="Cobalt-57")
#plt.plot(channels_Co60,counts_Co60,label="Cobalt-60")
#plt.plot(channels_Na22,counts_Na22,label="Sodium-22")
#plt.plot(channels_Cs137,counts_Cs137,label="Caesium-137")

#plt.legend()

split = lambda x,low,high: x[low:high]

Co57_cut, peak_Co57 = split(channels_Co57,0,30), split(counts_Co57,0,30)
Co60_cut, peak_Co60 = split(channels_Co60,5,56), split(counts_Co60,5,56)
Na22_cut, peak_Na22 = split(channels_Na22,70,110), split(counts_Na22,70,110)
Cs137_cut, peak_Cs137 = split(channels_Cs137,87,122), split(counts_Cs137,87,122)

def gaussian(x,N,mu,sigma,offset):
    return N/np.sqrt(2*np.pi*sigma**2) * np.exp(-0.5*(x-mu)**2/sigma**2) + offset

params_Co57, cmat = optimize.curve_fit(gaussian,Co57_cut,peak_Co57,p0=[1e3,14,4,0])
params_Co60, cmat = optimize.curve_fit(gaussian,Co60_cut,peak_Co60,p0=[300,22,10,54])
params_Na22, cmat = optimize.curve_fit(gaussian,Na22_cut,peak_Na22,p0=[100,98,4,0])
params_Cs137, cmat = optimize.curve_fit(gaussian,Cs137_cut,peak_Cs137,p0=[150,104,4,100])


#plt.figure()
#plt.plot(Co57_cut,gaussian(Co57_cut,*params_Co57),c="red")
#plt.plot(Co60_cut,gaussian(Co60_cut,*params_Co60),c="red")
#plt.plot(Na22_cut,gaussian(Na22_cut,*params_Na22),c="orange")
#plt.plot(Cs137_cut,gaussian(Cs137_cut,*params_Cs137),c="blue")
#plt.plot(Co57_cut,peak_Co57,label="Co-57")
#plt.plot(Co60_cut,peak_Co60,label="Co-60")
#plt.plot(Na22_cut,peak_Na22,label="Na22")
#plt.plot(Cs137_cut,peak_Cs137,label="Cs-137")
#plt.legend()

channels = [params_Co57[1],params_Co60[1],params_Na22[1],params_Cs137[1]]
energies = [122.061,0,546.544,661.659]  # keV

print(channels)
print(energies)

plt.scatter(channels, energies)
plt.show()
