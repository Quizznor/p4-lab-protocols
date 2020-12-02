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
def inv_bw(x,O,A,w,g):
    return O - A/( (x**2-w**2)**2 + g**2*w**2)

# Fit several inv_breit_wigner to a data set. <cuts> specifies
# the indices at which the datasets will be split for analysis.
def perform_fits(x_data,y_data,y_err,cuts):
    fit_results = []

    for i in range(len(cuts[:-1])):
        low, high = cuts[i], cuts[i+1]
        x, y, err = x_data[low:high], y_data[low:high], y_err[low:high]
        O, A, w, g = np.mean([y[0],y[-1]]), 5e5*(np.mean([y[0],y[-1]])-min(y)), x[np.argmin(y)], 1
        initial_guess = np.array([O,A,w,g])
        bounds = [ [O-1e3,0,w-2,0],[O+1e3,5*A,w+2,np.inf] ]
        popt, pcov = curve_fit(inv_bw,x,y,initial_guess,err,bounds=bounds,maxfev=int(1e5))
        pcov = np.delete(np.delete(pcov,1,1),1,0) # get rid of errors in A
        fit_results.append([popt, pcov])

    return fit_results

# Visualize the fits obtained in perform_fits
def draw_fits(x_data,fit_results,cuts):

    for i in range(len(cuts[:-1])):
        low, high = cuts[i], cuts[i+1]
        X = np.linspace(x_data[low],x_data[high],1000)
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

        plt.plot(X,model,lw=1)
        plt.fill_between(X,model+err,model-err,alpha=0.2)

# Useful to see where the datasets are split
def draw_cuts(x_data,cuts):
    for cut in cuts:
        plt.axvline(x_data[cut],c="gray",ls="--",zorder=1)

# Print out results of optimization
def print_results(fit_results):

    for i in range(len(fit_results)):
        print(f"{i+1}: omega_0 = {fit_results[i][0][2]:.2f} +- {np.sqrt(np.diag(fit_results[i][1])[1]):.3f}")
