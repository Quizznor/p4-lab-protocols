#!/usr/bin/env py

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as opt

channels, data = np.genfromtxt("../data/Calibration.txt",unpack=True,skip_header=40,skip_footer=900)
channel_signal, data_signal = channels[20:34], data[20:34]

def breit_wigner(x,A,omega,gamma):
    return A/( (x**2-omega**2)**2 + gamma**2*omega**2 )

popt, pcov = opt.curve_fit(breit_wigner, channel_signal, data_signal, p0=[5e5*max(data),68,15],maxfev=int(1e7),sigma=np.sqrt(data_signal))

def bw_err(xvals,pcov,parameters,reduced=True):

    # reduced = True neglects error on the Amplitude of the Breit-Wigner function
    # and keeps errors in a reasonable bound. We don't care about amplitude anyhow

    COV = np.delete(np.delete(pcov,0,1),0,0) if reduced else pcov
    [A,w,g] = parameters
    upper, lower = [], []

    for x in xvals:
        denum = ( (x**2 - w**2)**2 + g**2*w**2)
        dw = -(4*w*(x**2-w**2) + 2*g**2*w)*A/denum**2
        dg = -2*g*w*A/denum**2

        GRAD = np.array([dw,dg]) if reduced else np.array([denum,dw,dg])
        err = np.sqrt( GRAD.T @ COV @ GRAD )
        upper.append(breit_wigner(x,A,w,g) + err)
        lower.append(breit_wigner(x,A,w,g) - err)

    return np.array(upper),np.array(lower)

upper,lower = bw_err(channel_signal,pcov,popt,reduced=True)
C, C_err = popt[1],np.sqrt(pcov[1][1])

def C_to_E(channel):
    # 14.4 keV corresponds to the transition we are looking at
    # returns energy as well as error in energy to a corresponding channel
    return 14.4/C * channel, 14.4/(C-Cerr) * channel - 14.4/(C+Cerr) * channel

if __name__ == "__main__":
    print(f"\n\nBest fit to 14.4 keV line - CHANNEL #{C:.3f}+-{C_err:.3f}")
    print(f"A, omega_0, gamma = {popt}")
    print(f"Errs = +-{np.sqrt(np.diag(pcov))}")
    print("COV matrix given as:")
    print(pcov)

    plt.rcParams.update({'font.size': 16})
    plt.rc('axes', labelsize=26)
    plt.fill_between(channel_signal,upper,lower,color="C1",alpha=0.4)
    plt.errorbar(channels,data,yerr=np.sqrt(data),ls=":",capsize=2,zorder=1,label="Measurement data")
    plt.plot(channel_signal,breit_wigner(channel_signal, *popt),lw=0.5,zorder=2,label="Breit-Wigner fit")
    plt.axvline(C,c="k",lw=0.7,ls="--",label=r"$E_0 = 14.4\,$keV")
    plt.axvspan(C-C_err,C+C_err,color="k",alpha=0.3)
    plt.xlabel("MCS channel number")
    plt.ylabel("Event count")
    plt.ylim(min(data))
    plt.legend()
    plt.show()
