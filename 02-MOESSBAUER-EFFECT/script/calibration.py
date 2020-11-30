#!/usr/bin/env py

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as opt

plt.rcParams.update({'font.size': 16})
plt.rc('axes', labelsize=26)

channels, data = np.genfromtxt("../data/calib.txt",unpack=True,skip_header=40,skip_footer=900)
channel_signal, data_signal = channels[20:34], data[20:34]

def breit_wigner(x,A,omega,gamma):
    return A/( (x**2-omega**2)**2 + gamma**2*omega**2 )

popt, pcov = opt.curve_fit(breit_wigner, channel_signal, data_signal, p0=[5e5*max(data),68,15],maxfev=int(1e7),sigma=np.sqrt(data_signal))

# def bw_grad(x,*parameters):
#     A,w,g = parameters
#     denum = 1/( (x**2 - w**2)**2 + g**2*w**2)
#     dw = -(4*w*(x**2-w**2 + 2*g**2*w))/denum**2
#     dg = -2*g*w/denum**2
#
#     return np.array([denum,dw,dg])
#
# plt.fill_between(channel_signal,upper,lower,color="C1",alpha=0.4)

print(popt)
print(pcov)

plt.errorbar(channels,data,yerr=np.sqrt(data),ls=":",capsize=2,zorder=1,label="Measurement data")
plt.plot(channel_signal,breit_wigner(channel_signal, *popt),zorder=2,label="Breit-Wigner fit")
plt.axvline(popt[1],ls="--",c="gray")
plt.xlabel("MCS channel number")
plt.ylabel("Event count")
plt.ylim(min(data))
plt.legend()
plt.show()
