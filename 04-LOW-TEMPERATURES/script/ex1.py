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
T_copper, R_copper = debye_T, debye_R
T_copper_err, R_copper_err = np.sqrt(pcov[0][0]), np.sqrt(pcov[1][1])

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

# calculate resistivity and mean free path

resistance_300 = 2.163 # Ohms
resistance_42 = np.exp(offset) * 18**slope
resistance_42_err = np.sqrt( resistance_42**2 * (pcov_slope[1][1]**2 + 4.2/slope * pcov_slope[0][0]**2 ) )
specific_resistance_300 = 0.017e-6 # Ohm meter
l_copper = resistance_300 * 7.8642e-9/specific_resistance_300 # meters
l_copper_err = 0.0005 * 7.8642e-9/specific_resistance_300
rho_cu = resistance_42 * 7.8642e-9/l_copper
rho_cu_err = np.sqrt( (rho_cu * resistance_42_err/resistance_42)**2 + (-rho_cu * l_copper_err/l_copper)**2 )
rhol_cu = 658.7e-18 # Ohm meter²
mean_free_path_cu = rhol_cu/rho_cu # meters
mean_free_path_cu_err = mean_free_path_cu * rho_cu_err/rho_cu

print(f"Copper: Resistivity 4.2 K: rho = {rho_cu:.1e} +- {rho_cu_err:.1e} Ohm * m")
print(f"Copper: Mean free path: mu = {mean_free_path_cu:.1e} +- {mean_free_path_cu_err:.1e} m")

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
T_niobium, R_niobium = debye_T, debye_R
T_niobium_err, R_niobium_err = np.sqrt(pcov[0][0]), np.sqrt(pcov[1][1])
# specific_resistivity =
# mean_free_path =

print()
# print(f"Niobium: l = {l_niobium:.3f} +- {l_niobium_err}:.3f} K")
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

# calculate resistivity and mean free path

resistance_300 = 56.59 # Ohms
resistance_12 = np.exp(offset) * 12**slope
resistance_12_err = np.sqrt( resistance_12**2 * (pcov_slope[1][1]**2 + 4.2/slope * pcov_slope[0][0]**2 ) )
l_niobium = 8e-3 # meters
rho_nb = resistance_12 * 3.6e-11/l_niobium
rho_nb_err = rho_nb * resistance_12_err/resistance_12
rhol_nb = 375e-18 # Ohm meter²
mean_free_path_nb = rhol_nb/rho_nb # meters
mean_free_path_nb_err = mean_free_path_nb * rho_nb_err/rho_nb

print(f"Niobium: Resistivity 12 K: rho = {rho_nb:.1e} +- {rho_nb_err:.1e} Ohm * m")
print(f"Niobium: Mean free path: mu = {mean_free_path_nb:.1e} +- {mean_free_path_nb_err:.1e} m")

# # plotting everything
# plt.figure()
# plt.plot(linear_x_nb,model,c="C1",label="best fits")
# plt.plot(nonlinear_x_nb, model_slope,c="C1",zorder=50)
# plt.fill_between(nonlinear_x_nb,0.01*model_slope-fit_err_slope, 3*model_slope+fit_err_slope,alpha=0.2,color="C0")
# plt.fill_between(linear_x_nb,model-fit_err+0.8, model+fit_err-0.8,alpha=0.2,color="C0",label="fit + 1$\sigma$")
# plt.errorbar(linear_x_nb, linear_y_nb,capsize=3,xerr=2,yerr=0.0005,c="C0",ls="none",label="measured data")
# plt.errorbar(nonlinear_x_nb, nonlinear_y_nb,xerr=2,yerr=0.0005,capsize=3,c="C0",ls="none")

## SEMICONDUCTOR ##

# # plotting everything
# plt.figure()
# plt.errorbar(T_down[:-4],U_SLP[:-4],xerr=2,yerr=0.005,ls="none",c="C0")
# plt.errorbar(T_up,U_SLP_up,c="C0",xerr=2,yerr=0.005,ls="none")
# plt.yscale("log")

## COMBINED ANALYSIS ##

def line(x,slope,offset):
    return slope * x + offset

X = np.array(list(linear_x_cu/T_copper) + list(linear_x_nb/T_niobium))
X_err = np.array(list(linear_x_cu/T_copper**2 * T_copper_err) + list(linear_x_nb/T_niobium**2 * T_niobium_err))
Y = np.array(list(linear_y_cu/R_copper) + list(linear_y_nb/R_niobium))
Y_err = np.array(list(linear_y_cu/R_copper**2 * R_copper_err) + list(linear_y_nb/R_niobium**2 * R_niobium_err))

# iterative fitting to function
slope, offset = 1.17, 0.17
for i in range(5):
    (slope, offset), pcov = curve_fit(line,X,Y,sigma = np.sqrt(Y_err**2 + (slope*X_err)**2))

slope_err = np.sqrt(pcov[0][0])
offset_err = np.sqrt(pcov[1][1])
model = slope * X + offset
d_ds, d_do = X, np.ones_like(X)
fit_err = np.zeros_like(X)

for i in range(len(fit_err)):
    gradient = np.array([d_ds[i],d_do[i]])
    fit_err[i] += np.sqrt( gradient.T @ pcov @ gradient)

print()
print(f"COMBINED ANALYSIS: slope = {slope:.3f} +- {slope_err:.3f}")
print(f"COMBINED ANALYSIS: offset = {offset:.3f} +- {offset_err:.3f}")

# plt.errorbar(X,Y,X_err,Y_err,ls="none")
# plt.scatter(linear_x_cu/T_copper, linear_y_cu/R_copper,marker="s",s=5,label="Copper")
# plt.scatter(linear_x_nb/T_niobium, linear_y_nb/R_niobium,marker="o",s=10,zorder=50,label="Niobium")
# plt.fill_between(X,model-fit_err, model+fit_err,alpha=0.2,color="C0",label="fit + 1$\sigma$")
# plt.legend()
# plt.xlabel(r"T / $\theta$")
# plt.ylabel(r"R / R$_\theta$")

# plt.xlabel("Temperature (K)")
# plt.ylabel("Resistance ($\Omega$)")
# plt.legend()

plt.show()
