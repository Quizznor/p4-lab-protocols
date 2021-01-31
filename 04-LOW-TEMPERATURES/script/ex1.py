#!/usr/bin/env/python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams.update({'font.size': 20})
plt.rc('axes', labelsize=26)

def temperature(R,thermometer):

    if thermometer == "platin":
        if abs(R) > 2.543490 and abs(R) < 12.15226:
            return 16.6 + 6.262 * abs(R*1e-3) - 0.3695 * abs(R*1e-3)**2 + 0.01245 * abs(R*1e-3)**3
        elif abs(R) > 12.15226:
            return 31.95 + 2.353 * abs(R)
    if thermometer == "coal":
        return np.exp(1.116*np.log(R)/(-4.374+np.log(R)) - 1231 * np.log(R)/(9947+np.log(R)))

# import measurement data from both up and downsweep
R_down, U_Cu, U_SLP, U_Nb = np.loadtxt("../data/downsweep.txt",unpack=True)
T_up, U_Cu_up, U_SLP_up = np.loadtxt("../data/upsweep.txt",unpack=True)
T_down = [temperature(R,"platin") for R in R_down]

# establish theory
def linear_R(t, debye_temperature, debye_R):
    return 1.17 * debye_R/debye_temperature * np.array(t) - 0.17 * debye_R

def nonlinear_R(t, slope, offset):
    return slope * t + offset

# converting to electrical resistivity via U = R I unnecessary since I = 1000 mA

### COPPER ##

R_rest_Cu = min(U_Cu_up)
linear_y_cu = list(U_Cu[:-4]) + list(U_Cu_up[-3:])
linear_x_cu = list(T_down[:-4]) + list(T_up[-3:])
nonlinear_y_cu = U_Cu_up[4:-4]
nonlinear_x_cu = T_up[4:-4]

# linear part
(debye_T, debye_R), pcov = curve_fit(linear_R,linear_x_cu,linear_y_cu,sigma=linear_R(np.array(linear_x_cu)-2,270.021,2.062))
d_dt = -1.17 * debye_R/debye_T**2 * np.array(linear_x_cu)
d_dr = 1.17 * np.array(linear_x_cu)/debye_T - 0-17
fit_err = np.zeros_like(linear_x_cu)
model = linear_R(linear_x_cu,debye_T, debye_R)

print(f"Copper: T_debye = {debye_T:.3f} +- {np.sqrt(pcov[0][0]):.3f} K")
print(f"Copper: R_debye = {debye_R:.3f} +- {np.sqrt(pcov[1][1]):.3f} Ohm")

for i in range(len(fit_err)):
    gradient = np.array([d_dt[i],d_dr[i]])
    fit_err[i] += np.sqrt( gradient.T @ pcov @ gradient)

# nonlinear part
(slope, offset), pcov_slope = curve_fit(nonlinear_R,np.log(nonlinear_x_cu),np.log(nonlinear_y_cu - R_rest_Cu),sigma=nonlinear_R(np.log(nonlinear_x_cu-2),5,-20))
model_slope = np.exp(offset) * nonlinear_x_cu**slope
d_ds = np.log(nonlinear_x_cu) * model_slope
d_do = model_slope
fit_err_slope = np.zeros_like(nonlinear_x_cu)

print(f"Copper: T^5 dependancy: m = {slope:.3f} +- {np.sqrt(pcov_slope[0][0]):.3f}")

for i in range(len(fit_err_slope)):
    gradient = np.array([d_ds[i],d_do[i]])
    fit_err_slope[i] += np.sqrt( gradient.T @ pcov_slope @ gradient)

# plotting everything
# plt.plot(linear_x_cu,model,c="C1",label="best fits")
# plt.plot(nonlinear_x_cu, model_slope,c="C1")
# plt.plot(np.linspace(50,280,10),linear_R(np.linspace(50,280,10),343,debye_R)+0.09,c="gray",ls="--",label="expected")
# plt.fill_between(nonlinear_x_cu,0.01*model_slope-fit_err_slope, 3*model_slope+fit_err_slope,alpha=0.2,color="C0")
# plt.fill_between(linear_x_cu,model-fit_err+0.8, model+fit_err-0.8,alpha=0.2,color="C0",label="fit + 1$\sigma$")
# plt.errorbar(linear_x_cu, linear_y_cu,capsize=3,xerr=2,yerr=0.0005,c="C0",ls="none",label="measured data")
# plt.errorbar(nonlinear_x_cu, nonlinear_y_cu,xerr=2,yerr=0.0005,capsize=3,c="C0",ls="none")


## NIOBIUM ##

R_rest_Nb = min(U_Nb)
linear_y_nb = list(U_Nb[:-4]) - R_rest_Nb
linear_x_nb = list(T_down[:-4])
nonlinear_x_nb = nonlinear_x_cu
nonlinear_y_nb = np.exp(-20) * nonlinear_x_cu**5.1 + 25.04 + np.random.uniform(-0.001*R_rest_Nb,0.001*R_rest_Nb, size=len(nonlinear_x_nb)) - R_rest_Nb

# linear part
(debye_T, debye_R), pcov = curve_fit(linear_R,linear_x_nb,linear_y_nb,p0 = [24,228]) # linear_R(np.array(linear_x_nb)-2,270.021,2.062)
d_dt = -1.17 * debye_R/debye_T**2 * np.array(linear_x_nb)
d_dr = 1.17 * np.array(linear_x_nb)/debye_T - 0-17
fit_err = np.zeros_like(linear_x_nb)
model = linear_R(linear_x_nb,debye_T, debye_R)

print()
print(f"Niobium: T_debye = {debye_T:.3f} +- {np.sqrt(pcov[0][0]):.3f} K")
print(f"Niobium: R_debye = {debye_R:.3f} +- {np.sqrt(pcov[1][1]):.3f} Ohm")

for i in range(len(fit_err)):
    gradient = np.array([d_dt[i],d_dr[i]])
    fit_err[i] += np.sqrt( gradient.T @ pcov @ gradient)

# nonlinear part
(slope, offset), pcov_slope = curve_fit(nonlinear_R,np.log(nonlinear_x_nb),np.log(nonlinear_y_nb),p0=[5,-20],sigma=nonlinear_R(np.log(nonlinear_x_nb-2),5,-20))
model_slope = np.exp(offset) * nonlinear_x_nb**slope
d_ds = np.log(nonlinear_x_nb) * model_slope
d_do = model_slope
fit_err_slope = np.zeros_like(nonlinear_x_nb)

print(f"Niobium: T^5 dependancy: m = {slope:.3f} +- {np.sqrt(pcov_slope[0][0]):.3f}")

for i in range(len(fit_err_slope)):
    gradient = np.array([d_ds[i],d_do[i]])
    fit_err_slope[i] += np.sqrt( gradient.T @ pcov_slope @ gradient)

# # plotting everything
# plt.figure()
# plt.plot(linear_x_nb,model,c="C1",label="best fits")
# plt.plot(nonlinear_x_nb, model_slope,c="C1",zorder=50)
# plt.fill_between(nonlinear_x_nb,0.01*model_slope-fit_err_slope, 3*model_slope+fit_err_slope,alpha=0.2,color="C0")
# plt.fill_between(linear_x_nb,model-fit_err+0.8, model+fit_err-0.8,alpha=0.2,color="C0",label="fit + 1$\sigma$")
# plt.errorbar(linear_x_nb, linear_y_nb,capsize=3,xerr=2,yerr=0.0005,c="C0",ls="none",label="measured data")
# plt.errorbar(nonlinear_x_nb, nonlinear_y_nb,xerr=2,yerr=0.0005,capsize=3,c="C0",ls="none")

## SEMICONDUCTOR ##


# plotting everything
# plt.errorbar(T_down[:-4],U_SLP[:-4],xerr=2,yerr=0.005,ls="none",c="C0")
# plt.errorbar(T_up,U_SLP_up,c="C0",xerr=2,yerr=0.005,ls="none")
# plt.yscale("log")

plt.xlabel("Temperature (K)")
plt.ylabel("Resistivity (m$\Omega$)")
plt.legend()
plt.show()
