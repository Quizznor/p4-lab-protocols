#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def hysteresis(x,A,B,C,D):
    return A*np.tanh(B*(x-C))+D*x

def prepare_data(data,bin_tails=False):
    x,y = data

    # Eliminate y and x offset
    y -= np.mean(y)
    # x -= np.mean(x)

    return x,y

plt.rcParams.update({'font.size': 20})
plt.rc('axes', labelsize=26)

fig, axes = plt.subplots(2,2, sharey=True,sharex=True)
fig.subplots_adjust(hspace=0.22,wspace=0.05)
axes = axes.ravel()

for i,distance in enumerate([10,13,15,17,20,25][:-2]):

    data_x,data_y = prepare_data(np.loadtxt(f"../data/sample_A_{distance}.DAT",unpack=True))
    X = np.linspace(0.95*min(data_x),1.05*max(data_x),1000)

    # Split in ascending / descending branch
    xmin, xmax = np.argmin(data_x), np.argmax(data_x)
    complement_mask = np.ones(len(data_x),dtype=bool)
    complement_mask[xmax:xmin] = False

    x_split = [data_x[complement_mask], data_x[xmax:xmin]]
    y_split = [data_y[complement_mask], data_y[xmax:xmin]]

    fit_params, fit_errors = [], []

    for j,branch in enumerate(["descending", "ascending"]):
        x,y = x_split[j],y_split[j]

        popt, pcov = curve_fit(hysteresis,x,y,p0=[max(y),0,x[np.argmin(np.abs(x))],0],sigma=0.01*y,maxfev=10000)
        fit_params.append(popt),fit_errors.append(np.diag(pcov))

        # error calculation
        dA = lambda x: np.tanh(popt[1]*(x-popt[2]))
        dB = lambda x: popt[0]*(x-popt[2])/np.cosh(popt[1]*(x-popt[2]))**2
        dC = lambda x: -popt[0]*popt[1]/np.cosh(popt[1]*(x-popt[2]))**2
        dD = lambda x: x

        grad, dH = np.array([[dA(x),dB(x),dC(x),dD(x)] for x in X]), np.zeros_like(X)
        for k,x_val in enumerate(X):
            dH[k] += np.sqrt(grad[k].T @ pcov @ grad[k])

    A, A_err = 0.5*(fit_params[0][0]+fit_params[1][0]), 0.5*np.sqrt(fit_errors[0][0]**2+fit_errors[1][0]**2)
    B, B_err = 0.5*(fit_params[0][1]+fit_params[1][1]), 0.5*np.sqrt(fit_errors[0][1]**2+fit_errors[1][1]**2)
    C, C_err = 0.5*(fit_params[0][2]-fit_params[1][2]), 0.5*np.sqrt(fit_errors[0][2]**2+fit_errors[1][2]**2)
    D, D_err = 0.5*(fit_params[0][3]+fit_params[1][3]), 0.5*np.sqrt(fit_errors[0][3]**2+fit_errors[1][3]**2)

    axes[i].set_title(f"d = {distance} mm")
    axes[i].scatter(data_x,data_y,s=1,alpha=0.9,label="data")
    axes[i].plot(X,hysteresis(X,A,B,C,D),lw=2,c="C1",label="model")
    axes[i].fill_between(X,hysteresis(X,A,B,C,D)-dH,hysteresis(X,A,B,C,D)+dH,alpha=0.2)
    axes[i].plot(X,hysteresis(X,A,B,-C,D),lw=2,c="C1")
    axes[i].fill_between(X,hysteresis(X,A,B,-C,D)-dH,hysteresis(X,A,B,-C,D)+dH,alpha=0.2)
    axes[i].legend(fontsize=18)

fig.text(0.52, 0.02, 'Magnetic field strength (Oe)', ha='center',fontsize=20)
fig.text(0.06, 0.5, 'Kerr-Rotation + const. (mdeg)', va='center', rotation='vertical',fontsize=24)

plt.show()
