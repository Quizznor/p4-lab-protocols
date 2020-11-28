#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
plt.rc('axes', labelsize=26)

def breit_wigner(x,omega,gamma):

    breit_wigner = 1/( (x**2-omega**2)**2 + gamma**2*omega**2 )
    return breit_wigner/max(breit_wigner), 0.5 * ( np.sqrt(omega**2 + gamma*omega) - np.sqrt(omega**2 - gamma*omega) )

E, shift = 4000, 25
X = np.linspace(E-5*shift,E+5*shift,1000)
BW_tight,FWHM_tight = breit_wigner(X,E,4)
BW_loose,FWHM_loose = breit_wigner(X,E,12)

# figure this out better
overlap = [min(BW_tight[i],BW_loose[i]) for i in range(len(X))]

# plt.axvline(E,c="k",ls="--",alpha=0.4)
# plt.axvline(E+shift,c="k",ls="--",alpha=0.4)
# plt.axvline(E-shift,ymax=0.8,c="k",ls="--",alpha=0.4)
# plt.plot(X-shift,BW_tight,label="Emission spectrum")
# plt.plot(X+shift,BW_tight,label="Absorption spectrum")
# plt.tick_params(axis='both', which='major',pad=20)
# plt.text(E-shift-2, 1.14,r"$\Gamma_0$",fontdict=dict(fontsize=25))
# plt.annotate('',(E-shift-FWHM_tight,1.1),(E-shift+FWHM_tight,1.1), arrowprops=dict(arrowstyle="<|-|>",facecolor="k"))
# plt.xticks([E-shift,E,E+shift],[r"$E_0-\frac{p_\gamma^2}{2m}$",r"$E_0$",r"$E_0+\frac{p_\gamma^2}{2m}$"])
# plt.xlim(E-2.5*shift,E+2.5*shift)
# plt.yticks([],[])
# plt.ylim(0,1.3)
# plt.legend(loc="upper right")



plt.figure()
plt.axvline(E,c="k",ls="--",alpha=0.4)
plt.axvline(E+shift,c="k",ls="--",alpha=0.4)
plt.axvline(E-shift,ymax=0.8,c="k",ls="--",alpha=0.4)
plt.plot(X-shift,BW_loose,label="Emission spectrum")
plt.plot(X+shift,BW_loose,label="Absorption spectrum")
plt.tick_params(axis='both', which='major',pad=20)
plt.text(E-shift-7, 1.14,r"$\Gamma_0+\Gamma_\mathrm{D}$",fontdict=dict(fontsize=25))
plt.annotate('',(E-shift-FWHM_loose,1.1),(E-shift+FWHM_loose,1.1), arrowprops=dict(arrowstyle="<|-|>",facecolor="k"))
plt.xticks([E-shift,E,E+shift],[r"$E_0-\frac{p_\gamma^2}{2m}$",r"$E_0$",r"$E_0+\frac{p_\gamma^2}{2m}$"])
plt.fill_between(X,0,overlap,color="k",alpha=0.4)
plt.xlim(E-2.5*shift,E+2.5*shift)
plt.yticks([],[])
plt.ylim(0,1.3)
plt.legend(loc="upper right")
plt.show()