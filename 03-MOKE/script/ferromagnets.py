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

plt.rcParams.update({'font.size': 20})
plt.rc('axes', labelsize=26)

fig, axes = plt.subplots(3,2,sharey="row",sharex="row")
fig.subplots_adjust(hspace=0.65,wspace=0.05)
axes = axes.ravel()

for i,distance in enumerate([10,13,15,17,20,25]):

    thickness = [0.4,0.5,0.7,0.9,1.4,2.0][i]

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

        treshold = [170,180,200,140,110,40][i]
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

    # Calculate the effective anisotropy constant
    H_s = X[np.argmin(np.abs(A-hysteresis(X,A,B,C,D)))]

    K_eff = 0.5 * A * (C - H_s)

    print(f"SI{thickness}nanometer & SI{A:.2f}millidegree & {H_s:.2f}oersted & {C:.2f}oersted & {K_eff:.2f}millidegreeperoersted \\\\")

    # print(f"SI {thickness} nanometer & SI {A:.4f}pm{A_err:.4f} millidegree & SI {B*1e3:.4f}pm{B_err*1e3:.4f} permillioersted \
    # & SI {C:.2f}pm{C_err:.2f} oersted & SI {D*1e3:.2f}pm{D_err*1e3:.2f} microdegreeperoersted")

    axes[i].set_title(f"d = {thickness} nm")
    axes[i].scatter(data_x,data_y,s=1,alpha=0.9,label="data")
    axes[i].plot(X,hysteresis(X,A,B,C,D),lw=2,c="C1",label="model")
    axes[i].plot(X,hysteresis(X,A,B,-C,D),lw=2,c="C1")
    axes[i].legend(fontsize=16,loc="lower right")


    # axes[i].fill_between(X,hysteresis(X,A,B,C,D)-dH,hysteresis(X,A,B,C,D)+dH,alpha=0.2)
    # axes[i].fill_between(X,hysteresis(X,A,B,-C,D)-dH,hysteresis(X,A,B,-C,D)+dH,alpha=0.2)

fig.text(0.52, 0.02, 'Magnetic field strength (Oe)', ha='center',fontsize=20)
fig.text(0.04, 0.5, 'Kerr-Rotation + const. (mdeg)', va='center', rotation='vertical',fontsize=24)

plt.show()
