#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def hysteresis(x,A,B,C,D):
    return A*np.tanh(B*(x-C))+D*x

def prepare_data(data,bin_tails=False):
    x,y = data

    # Eliminate y offset
    y -= np.mean(y)

    return x,y

T = [0,10,72,75]
H_c = np.zeros_like(T)
H_c_err = np.zeros_like(T)

plt.rcParams.update({'font.size': 20})
plt.rc('axes', labelsize=26)
fig, axes = plt.subplots(2,2,sharey="row",sharex="row")
fig.subplots_adjust(hspace=0.65,wspace=0.05)
axes = axes.ravel()

for i,temperature in enumerate(["00","10","72","75"]):

    data_x,data_y = prepare_data(np.loadtxt(f"../data/sample_C6_{temperature}.DAT",unpack=True))
    X = np.linspace(0.95*min(data_x),1.05*max(data_x),1000)

    # Split in ascending / descending branch
    xmin, xmax = np.argmin(data_x), np.argmax(data_x)
    complement_mask = np.ones(len(data_x),dtype=bool)
    complement_mask[xmax:xmin] = False

    x_split = [data_x[complement_mask], data_x[xmax:xmin]]
    y_split = [data_y[complement_mask], data_y[xmax:xmin]]

    fit_params, fit_errors = [], []

    for j,branch in enumerate(["descending", "ascending"]):

        treshold = [30,30,30,30,30,30][i]
        x,y = x_split[j],y_split[j]

        # initial guesses
        x_tail,y_tail = x[1:treshold], y[1:treshold]
        slope, intercept = np.polyfit(x_tail,y_tail,1)

        p0 = [max(y),0.1,x[np.argmin(np.abs(x))],slope]
        lower = [0,0.01,-100,0.995*p0[3]]
        upper = [1.1*p0[0],10,100,1.005*p0[3]]

        popt, pcov = curve_fit(hysteresis,x,y,p0=p0,maxfev=10000,bounds=[lower,upper])
        fit_params.append(popt),fit_errors.append(np.diag(pcov))

        # error calculation
        # dA = lambda x: np.tanh(popt[1]*(x-popt[2]))
        # dB = lambda x: popt[0]*(x-popt[2])/np.cosh(popt[1]*(x-popt[2]))**2
        # dC = lambda x: -popt[0]*popt[1]/np.cosh(popt[1]*(x-popt[2]))**2
        # dD = lambda x: x
        #
        # grad, dH = np.array([[dA(x),dB(x),dC(x),dD(x)] for x in X]), np.zeros_like(X)
        # for k,x_val in enumerate(X):
        #     dH[k] += np.sqrt(grad[k].T @ pcov @ grad[k])

    A, A_err = 0.5*(fit_params[0][0]+fit_params[1][0]), 0.5*np.sqrt(fit_errors[0][0]**2+fit_errors[1][0]**2)
    B, B_err = 0.5*(fit_params[0][1]+fit_params[1][1]), 0.5*np.sqrt(fit_errors[0][1]**2+fit_errors[1][1]**2)
    C, C_err = 0.5*(fit_params[0][2]-fit_params[1][2]), 0.5*np.sqrt(fit_errors[0][2]**2+fit_errors[1][2]**2)
    D, D_err = 0.5*(fit_params[0][3]+fit_params[1][3]), 0.5*np.sqrt(fit_errors[0][3]**2+fit_errors[1][3]**2)

    # print(f"SI {thickness} nanometer & SI {A:.4f}pm{A_err:.4f} millidegree & SI {B*1e3:.4f}pm{B_err*1e3:.4f} permillioersted \
    # & SI {C:.2f}pm{C_err:.2f} oersted & SI {D*1e3:.2f}pm{D_err*1e3:.2f} microdegreeperoersted")

    axes[i].set_title(f"T = {int(temperature)} °C")
    axes[i].scatter(data_x,data_y,s=1,alpha=0.9,label="data")
    axes[i].plot(X,hysteresis(X,A,B,C,D),lw=2,c="C1",label="model")
    axes[i].plot(X,hysteresis(X,A,B,-C,D),lw=2,c="C1")
    axes[i].legend(fontsize=16,loc="lower right")

    H_c[i] += C
    H_c_err[i] += C_err

    # axes[i].fill_between(X,hysteresis(X,A,B,C,D)-dH,hysteresis(X,A,B,C,D)+dH,alpha=0.2)
    # axes[i].fill_between(X,hysteresis(X,A,B,-C,D)-dH,hysteresis(X,A,B,-C,D)+dH,alpha=0.2)

fig.text(0.52, 0.02, 'Magnetic field strength (Oe)', ha='center',fontsize=20)
fig.text(0.04, 0.5, 'Kerr-Rotation + const. (mdeg)', va='center', rotation='vertical',fontsize=24)

plt.figure()
plt.errorbar(T,H_c,xerr=2,yerr=H_c_err,ls="none")

plt.show()