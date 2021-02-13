#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 20})
plt.rc('axes', labelsize=26)

# Probe A
T, U_cc_plus, U_hall_plus, U_cc_zero, U_hall_zero, U_cc_minus, U_hall_minus = np.loadtxt("../data/hall_A.txt",unpack=True)
T += 273                                                                # in Kelvin
I = np.array([1e-3 if T[i] < 50 else 1.8e-3 for i in range(len(T))])    # in Ampere
U_hall = ( 0.5 * (U_hall_plus - U_hall_minus)) * 1e-3                   # in Volt
length, width, height = 19e-3, 10e-3, 1e-3                              # in Meter
sigma = ( I/(width*height) ) / ( U_cc_zero/length )                     # in 1/( Ohm meter )
R_hall = ( U_hall/width ) / ( I/(width*height) * 0.5 )                  # in m³/Coulomb
delta_U, delta_I = 0.0005, 0.05e-3                                      # in Volt, Ampere
sigma_err = sigma * np.sqrt( (delta_I/I)**2 + (delta_U/U_cc_zero)**2 )
U_hall_err = np.sqrt( (0.5*delta_U)**2 + (0.5*delta_U)**2 + (delta_U)**2 )
R_hall_err = R_hall * np.sqrt( (U_hall_err/U_hall)**2 + (delta_I/I)**2 )


# # plot everything from first bullet point
# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
# ax1.set_xlabel("Temperatur (K)")
# ax1.set_ylabel("Leitfähigkeit ($\Omega^{-1}\,\mathrm{m}^{-1}$)")
# ax2.set_ylabel(r"Hallkoeffizient ( $\mathrm{m}^3\,\mathrm{C}^{-1}$ )",labelpad=10)
# ax1.errorbar(T,sigma,xerr=1.5,yerr=sigma_err,ls="none",c="C0",capsize=2)
# ax1.scatter(T,sigma,marker="s",s=10,label="Leitfähigkeit",c="C0")
# ax2.errorbar(T,R_hall,xerr=1.5,yerr=R_hall_err,ls="none",c="C1",capsize=2)
# ax2.scatter(T,R_hall,marker="o",s=10,label="Hallkoeffizient",c="C1")
# fig.legend(loc="upper left", bbox_to_anchor=(0,1), bbox_transform=ax1.transAxes)

# # plot both lines in one graph from second bullet point
# plt.figure()
# plt.yscale("log")
# plt.axvline(45+273,ls="--",c="gray")
# plt.annotate(r"I$_\mathrm{A}$ = 1.0 mA",(310,430),rotation=90)
# plt.arrow(315,360,-4,0,length_includes_head=False,width=4,head_width=25,head_length=1,fc="k")
# plt.annotate(r"I$_\mathrm{A}$ = 1.8 mA",(320,430),rotation=90)
# plt.arrow(320.5,360,4,0,length_includes_head=False,width=4,head_width=25,head_length=1,fc="k")
# plt.errorbar(T[:-3],1/abs(R_hall[:-3]),label=r"$|R_\mathrm{Hall}|^{-1}$ / $\mathrm{m}^3\,\mathrm{C}^{-1}$",xerr=1.5,yerr=R_hall_err[:-3]/R_hall[:-3]**2,ls="none",capsize=2)
# plt.errorbar(T[-3:],1/abs(R_hall[-3:]),xerr=1.5,yerr=[R_hall_err[-3:]/R_hall[-3:]**2,[0,0,0]],capsize=2,c="C0",ls="none")
# plt.errorbar(T[-3:],1/abs(R_hall[-3:]),xerr=1.5,yerr=400,capsize=2,c="C0",ls="none",uplims=True)
# plt.errorbar(T,sigma,label=r"$\sigma$ / $\Omega^{-1}\,\mathrm{m}^{-1}$",xerr=1.5,yerr=sigma_err,capsize=2,ls="none")
# plt.xlabel("Temperatur (K)")
# plt.legend()

# extrinsic region
T_ext = T[8:17]
mu_ext = sigma[8:17] * abs(R_hall[8:17])
(m_ext,c_ext), pcov_ext = np.polyfit(np.log(T_ext), np.log(mu_ext), 1, cov=True)
model_ext = np.exp(m_ext*np.log(T_ext) + c_ext)
grad_ext = [ np.array([T_ext[i] * model_ext[i], model_ext[i] ]) for i in range(len(T_ext))]
errors_ext = [ np.sqrt(grad_ext[i].T @ pcov_ext @ grad_ext[i]) for i in range(len(T_ext))]

# intrinsic region
T_int = T[25:]
mu_int = sigma[25:] * abs(R_hall[25:])
(m_int,c_int), pcov_int = np.polyfit(np.log(T_int), np.log(mu_int), 1, cov=True)
model_int = np.exp(m_int*np.log(T_int) + c_int)
grad_int = [ np.array([T_int[i] * model_int[i], model_int[i] ]) for i in range(len(T_int))]
errors_int = [ np.sqrt(grad_int[i].T @ pcov_int @ grad_int[i]) for i in range(len(T_int))]

# # plot double logarithmic plot from second bullet poext
# plt.figure()
# plt.xscale("log")
# plt.yscale("log")
# plt.scatter(T,sigma * abs(R_hall), s=10,label=r"Messdaten $\pm\,1\sigma$")
# plt.errorbar(T[:-7],sigma[:-7] * abs(R_hall[:-7]),xerr=1.5,yerr=np.sqrt( (R_hall[:-7] * sigma_err[:-7])**2 + (sigma[:-7] * R_hall_err[:-7])**2 ),ls="none",capsize=2)
# plt.errorbar(T[-7:],sigma[-7:] * abs(R_hall[-7:]),xerr=1.5,yerr=[[0,0,0,0,0,0,0],np.sqrt( (R_hall[-7:] * sigma_err[-7:])**2 + (sigma[-7:] * R_hall_err[-7:])**2 )],ls="none",capsize=2,c="C0")
# plt.errorbar(T[-7:],sigma[-7:] * abs(R_hall[-7:]),xerr=1.5,yerr=[[1.5e-2,1.6e-2,1.7e-2,1.8e-2,2.2e-2,2e-2,2e-2],[0,0,0,0,0,0,0]],ls="none",capsize=2,c="C0",uplims=True)
# plt.plot(T_int,model_int,label="intrinsischer Bereich")
# plt.plot(T_ext,np.exp(m_ext*np.log(T_ext) + c_ext),label="extrinsischer Bereich")
# plt.yticks(np.array(range(1,11))*1e-1,["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"])
# plt.xticks([100,200,300,400],["100","200","300","400"])
# plt.xlabel("Temperatur (K)")
# plt.ylabel("Mobilität ($\mathrm{m}^2\,\mathrm{V}^{-1}\,\mathrm{s}^{-1}$)")
# plt.ylim(0.05,1)
# plt.legend()

# charge carrier concentration and band gap
b = lambda T: 1.24553 + 107e-5 * T
e = 1.602e-19                                                           # in Coulomb
b_int = b(T_int)
b_err = b(T_int[0] + 1.5) - b(T_int[0])
ccc = 1/(e * R_hall[8:17]) * (1-b_int)/(1+b_int)                        # in 1/Meter³
ccc_err = 1e-20 * ccc * np.sqrt( (R_hall_err[25:]/R_hall[25:])**2 + (b_err * (1/(1-b_int) + 1/(1+b_int)))**2 )
y_data = np.log(ccc/T_int**(2/3))
y_data_err = np.sqrt( (ccc_err/ccc)**2 + (2*1.5/(3*T_int))**2 )
popt, pcov = np.polyfit(1/T_int,y_data,1,w=1/y_data_err,cov=True)
model = np.poly1d(popt)(1/T_int)
grad = [np.array([1/T_int[i],1]) for i in range(len(model))]
model_err = [ np.sqrt(grad[i].T @ pcov @ grad[i]) for i in range(len(model))]
E_gap, E_gap_err = - 2 * 0.08617e-3 * popt[0], 2 * 0.08617e-3 * np.sqrt(pcov[0][0])
E_gap_300, E_gap_300_err =

print(f"Germanium Bandgap: E = {E_gap*10:.3f} +- {E_gap_err*10:.3f} eV")
print(f"at 300 Kelvin: E_300 = {E_gap_300:.3f} +- {E_gap_300_err:.3f} eV")
print(f"Y axis intercept : y = {popt[1]:.1f} +- {np.sqrt(pcov[0][0])*1e-1:.1f} eV")
print(f"Charge carrier concentration at 300K: {ccc_300} +- {ccc_300_err} 1/m3")


# # plot charge carrier concentration from fourth bullet point
# plt.figure()
# plt.errorbar(T_int,ccc*1e-19,xerr=1.5,yerr=ccc_err,ls="none",capsize=2,marker="s")
# plt.ylabel(r"Ladungsträgerkonzentration ($10^{19}\;\mathrm{m}^{-3}$)",labelpad=20)
# plt.xlabel("Temperatur (K)")

# plot Arrhenius presentation from fifth bullet point
plt.figure()
plt.xlabel("inv. Temperatur (K$^{-1}$)")
plt.ylabel(r"log( $\frac{n_i}{T^{2/3}}$ )")
plt.plot(1/T_int,model,c="C0",ls="--",label=r"Modell $\pm\,1\sigma$")
plt.fill_between(1/T_int,model-model_err,model+model_err,color="C0",alpha=0.2)
plt.errorbar(1/T_int,y_data,xerr=1.5/T_int**2,yerr=y_data_err,ls="none",capsize=2,marker="s",label="Messdaten",markersize=3)
plt.legend()

plt.show()
