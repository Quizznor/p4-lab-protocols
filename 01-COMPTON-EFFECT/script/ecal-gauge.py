#!/usr/bin/env py

import matplotlib.pyplot as plt
import numpy as np
import sys

plt.rcParams.update({'font.size': 28})
datadir = "../data"

channels_Co57, counts_Co57 = np.genfromtxt(f"{datadir}/gauge_Co-57.txt",unpack=True,max_rows=300)
channels_Co60, counts_Co60 = np.genfromtxt(f"{datadir}/gauge_Co-60.txt",unpack=True,max_rows=300)
channels_Na22, counts_Na22 = np.genfromtxt(f"{datadir}/gauge_Na-22.txt",unpack=True,max_rows=300)
channels_Cs137, counts_Cs137 = np.genfromtxt(f"{datadir}/gauge_Cs-137.txt",unpack=True,max_rows=300)

# plot the different spectra for each gauge measurement
# plt.figure()
# plt.title("Calibration of the MAC")
# plt.plot(channels_Co57,counts_Co57,label="Cobalt-57")
# plt.plot(channels_Co60,counts_Co60,label="Cobalt-60")
# plt.plot(channels_Na22,counts_Na22,label="Sodium-22")
# plt.plot(channels_Cs137,counts_Cs137,label="Caesium-137")
#
# plt.legend()

split = lambda x,low,high: x[low:high]

Co57_cut, peak_Co57 = split(channels_Co57,0,30), split(counts_Co57,0,30)
# Co60_cut, peak_Co60 = split(channels_Co60,3,60), split(counts_Co60,3,60)
Na22_cut, peak_Na22 = split(channels_Na22,70,110), split(counts_Na22,70,110)
Cs137_cut, peak_Cs137 = split(channels_Cs137,87,122), split(counts_Cs137,87,122)


plt.figure()
plt.plot(Co57_cut,peak_Co57,label="Co-57")
# plt.plot(Co60_cut,peak_Co60,label="Co-60")
plt.plot(Na22_cut,peak_Na22,label="Na22")
plt.plot(Cs137_cut,peak_Cs137,label="Cs-137")
plt.legend()
plt.show()
