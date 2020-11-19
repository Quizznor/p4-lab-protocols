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
from scipy import optimize as opt

angles = np.arange(20,101,10)
rad = lambda x: 2*np.pi * x/360
E_gamma, m_e_theo = 661.659, 510.999
E_dash, E_dash_err = [], []
print()

for angle in angles:

    sig = np.array(data_sig[int(angle/10)])
    bkg = np.array(data_bkg[int(angle/10)])

    # cut data to signal region
    channel = np.argmax(sig - bkg)
    low = channel - int(Cs137_std)
    high = channel + int(Cs137_std)
    signal_corr = sig[low:high] - bkg[low:high]
    energy = E(channels[low:high])

    params, pcov = opt.curve_fit(gaussian,energy,signal_corr,p0=[max(signal_corr),E(channel),Cs137_std,0])
    peak, peak_err = params[1],np.sqrt(np.diag(pcov)[1])
    E_dash.append(peak)
    E_dash_err.append(peak_err)

    print(f"{angle} & {1-np.cos(rad(angle)):.3f} & {peak:.2f}\pm{peak_err:.2f} & {1/peak:.4f}\pm{peak_err/peak**2*10:.4f}\\\\")

xerr = np.sin(rad(angles)) * rad(1)
yerr = np.array(E_dash_err)/np.array(E_dash)**2

def linear(x,a=m_e_theo):
    return 1/a * x + 1/E_gamma

m_e, COV = opt.curve_fit(linear,1-np.cos(rad(angles)),1/np.array(E_dash),p0=[m_e_theo])
m_e, m_e_err = m_e[0], np.sqrt(COV[0][0])

print(f"\nm_e = {m_e:.2f} +- {m_e_err:.2f} keV")
print(f"Deviation from theoretical value: m_e/m_e_theo = {(1-m_e/m_e_theo)*1e2:.2f}%")

X = np.linspace(0,2,1000)
upper = linear(X,m_e-m_e_err)
lower = linear(X,m_e+m_e_err)
plt.errorbar(1-np.cos(rad(angles)),1/np.array(E_dash),xerr=xerr,yerr=yerr*10,ls="none",label="measured")
plt.fill_between(X,upper,lower,alpha=0.4)
plt.plot(X,linear(X,m_e),label=r"$m_{e,\mathrm{exp}} = %.3f$ keV"%(m_e))
plt.plot(X,linear(X),c="k",label=r"$m_{e,\mathrm{theo}} = %.3f$ keV"%(m_e_theo))
plt.xlabel(r"1 - $\cos\,\theta$")
plt.ylabel(r"$\frac{1}{E_{\gamma,f}}$ ($\frac{1}{\mathrm{keV}}$)")
plt.legend()
plt.show()
