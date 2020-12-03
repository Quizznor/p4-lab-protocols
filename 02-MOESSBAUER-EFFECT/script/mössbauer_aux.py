#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True

from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from calibration import C_to_E
import numpy as np

# Create <res> bins from a given dataset. Bins are filled with either the mean value
# of data values or return the sum of data values if an array of errors is given.
def create_bins(x,x_err=None,res=128):

    bin_content = np.split(x,res)
    bins = np.zeros(res)

    if x_err is not None:
        bin_content_err = np.split(x_err,res)
        bins_err = np.zeros(res)

    for i in range(res):
        if x_err is not None:
            bins[i] = np.sum(np.array(bin_content[i]))
            bins_err[i] = np.sqrt( bins[i] )
        else:
            bins[i] = np.mean(np.array(bin_content[i]))

    if x_err is not None:
        return bins, bins_err
    else:
        return bins

# The shape fitted to the absorption peaks
def inv_bw(x,O,A,G,w0):
    return O - A * 0.5*G/( (x**2-w0**2)**2 + (0.5*G)**2 )

# Fit several inv_breit_wigner to a data set. <cuts> specifies
# the indices at which the datasets will be split for analysis.
def perform_fits(x_data,y_data,y_err,cuts,ch1):
    fit_results = []

    for i in range(len(cuts[:-1])):
        low, high = cuts[i], cuts[i+1]
        x, y, err = x_data[low:high], y_data[low:high], y_err[low:high]

        # estimating the initial guess and fit boundaries
        O_i = np.mean([y[0],y[-1]])
        w_i = x[np.argmin(y)]
        A_i = 3e5
        G_i = 1000
        initial_guess = np.array([O_i,A_i,w_i,G_i])

        lower, higher = [O_i-1e3,0,w_i-10,0], [O_i+1e3,np.inf,w_i+10,np.inf]
        popt, pcov = curve_fit(inv_bw,x,y,initial_guess,err,bounds=[lower,higher],maxfev=int(1e5),method="dogbox")
        pcov = np.delete(np.delete(pcov,1,1),1,0) # get rid of errors in A
        fit_results.append([popt, pcov])

    return fit_results

# Visualize the fits obtained in perform_fits
def draw_fits(x_data,fit_results,cuts,ch1):
    c = "C0" if ch1 else "C1"
    ls = "--" if ch1 else ":"
    lw = 0.5 if ch1 else 1

    for i in range(len(cuts[:-1])):
        low, high = cuts[i], cuts[i+1]
        X = np.linspace(x_data[low],x_data[high],10000)
        (O,A,w,g), pcov = fit_results[i][0], fit_results[i][1]
        model = inv_bw(X,O,A,w,g)

        # calculate 1-sigma errorband around fit
        denum = ( (X**2 - w**2)**2 + g**2*w**2)
        dw = -(4*w*(X**2-w**2) + 2*g**2*w)/denum**2
        dg = -2*g*w/denum**2

        err = np.zeros_like(X)
        for i in range(len(X)):
            grad = np.array([1,dw[i],dg[i]])
            err[i] = np.sqrt( grad.T @ pcov @ grad )

        plt.plot(X,model,lw=lw,ls=ls,c=c)
        plt.fill_between(X,model+err,model-err,alpha=0.15,color=c)

# Useful to see where the datasets are split
def draw_cuts(x_data,cuts):
    for cut in cuts:
        plt.axvline(x_data[cut],c="gray",ls="--",zorder=1)

# Print out results of optimization
def print_results(fit_results):

    for i in range(len(fit_results)):
        O, A = fit_results[i][0][0], fit_results[i][0][1]
        w, G = fit_results[i][0][2], fit_results[i][0][3]
        w_err = np.sqrt( np.diag(fit_results[i][1])[1] )
        G_err = np.sqrt( np.diag(fit_results[i][1])[2] )
        # wg_corr_sqr = fit_results[i][1][1,2]

        # FWHM = np.sqrt(w**2 + g*w) - np.sqrt(w**2 - g*w)
        # dw = 0.5 * ( (2*w+g)/np.sqrt(w**2 + g*w) - (2*2-g)/np.sqrt(w**2 - g*w) )
        # dg = 0.5 * ( g/np.sqrt(w**2 + g*w) + g/np.sqrt(w**2 - g*w) )
        # FWHM_err = np.sqrt( dw**2 * w_err**2 + dg**2 * g_err**2 + dw*dg*wg_corr_sqr )

        print(f"{i+1}: A = {A:.1f}, O = {O:.1f}")
        print(f"{i+1}: w_0 = {w:.1f} +- {w_err:.2f}, g = {G:.6f} +- {G_err:.6f}\n")
        # print(FWHM)
        # print(f" `-> FWHM = {FWHM:.6f} +- {FWHM_err:.6f}\n")
