#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True

from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from calibration import C_to_E
import numpy as np

# returns the velocity of the gamma source corresponding to a specific channel in mm/s
def C_to_v(C):
    return (C - 512/2)/max(C-512/2) * 10 # mm/s

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
    return O - A/( (x-w0)**2 + (0.5*G)**2 )

# Fit several inv_breit_wigner to a data set. <cuts> specifies
# the indices at which the datasets will be split for analysis.
def perform_fits(x_data,y_data,y_err,cuts,ch1):
    fit_results = []

    for i in range(len(cuts[:-1])):
        low, high = cuts[i], cuts[i+1]
        x, y, err = x_data[low:high], y_data[low:high], y_err[low:high]

        # estimating the initial guess and fit boundaries
        O_i = np.mean([y[0],y[-1]])
        FWHM = O_i - 0.5*(O_i-min(y))
        left, right = x[np.argmin(y):], x[:np.argmin(y)]
        G_i = left[np.argmin(np.abs(FWHM-left))] - right[np.argmin(np.abs(FWHM-right))]
        w_i = x[np.argmin(y)]
        A_i = inv_bw(w_i,O_i,1,G_i,w_i)/min(y)

        initial_guess = np.array([O_i,A_i,G_i,w_i])
        lower, higher = [0,0,0,w_i-0.5], [10*O_i,120,10,w_i+0.5]
        popt, pcov = curve_fit(inv_bw,x,y,initial_guess,err,bounds=[lower,higher],maxfev=int(1e5))
        fit_results.append([popt, pcov])

    return fit_results

# Visualize the fits obtained in perform_fits
def draw_fits(x_data,fit_results,cuts,ch1,override=None):
    c = "C0" if ch1 else "C1"
    ls = "--" if ch1 else ":"
    lw = 0.5 if ch1 else 1

    for j,i in enumerate(range(len(cuts[:-1]))):
        low, high = cuts[i], cuts[i+1]
        X = np.linspace(x_data[low],x_data[high],10000)
        (O,A,G,w), pcov = fit_results[i][0], fit_results[i][1]
        model = inv_bw(X,O,A,G,w)

        # calculate 1-sigma errorband around fit
        G_err = np.sqrt(np.diag(pcov)[2])
        denum = ( (X - w)**2 + 0.25*G**2 )
        dA = np.zeros_like(denum) # -0.5*G/denum # in order to get rid of errors in A tat blow everything up
        dG = 0.25*G/denum**2 - 0.5/denum
        dw = -(X-w)*G/denum**2
        err = np.zeros_like(X)

        for i in range(len(X)):
            grad = np.array([1,dA[i],dG[i],dw[i]])
            err[i] = np.sqrt( grad.T @ pcov @ grad )

        plt.plot(X,model,lw=lw,ls=ls,c=c)
        xy = (w-1,0.995*min(model)) if override is None else override[j]
        ch1 and plt.annotate(f"#{j+1}",xy)

        if G_err < 120:
            plt.fill_between(X,model+err,model-err,alpha=0.15,color=c)

# Useful to see where the datasets are split
def draw_cuts(x_data,cuts):
    for cut in cuts:
        plt.axvline(x_data[cut],c="gray",ls="--",zorder=1)

# Print out results of optimization in TeX table format (if desired)
def print_results(fit_results1,fit_results2, table=False):
    print()
    for i in range(len(fit_results1)):
        O1, O1_err = fit_results1[i][0][0], np.sqrt( np.diag(fit_results1[i][1])[0] )
        A1, A1_err = fit_results1[i][0][1], np.sqrt( np.diag(fit_results1[i][1])[1] )
        G1, G1_err = fit_results1[i][0][2], np.sqrt( np.diag(fit_results1[i][1])[2] )
        w1, w1_err = fit_results1[i][0][3], np.sqrt( np.diag(fit_results1[i][1])[3] )

        O2, O2_err = fit_results2[i][0][0], np.sqrt( np.diag(fit_results2[i][1])[0] )
        A2, A2_err = fit_results2[i][0][1], np.sqrt( np.diag(fit_results2[i][1])[1] )
        G2, G2_err = fit_results2[i][0][2], np.sqrt( np.diag(fit_results2[i][1])[2] )
        w2, w2_err = fit_results2[i][0][3], np.sqrt( np.diag(fit_results2[i][1])[3] )

        if not table:
            print("Results:    Ch1         / Ch2")
            print(f"Peak {i+1}: O = {O1:.0f} +- {O1_err:.0f} / {O2:.0f} +- {O2_err:.0f}")
            print(f"Peak {i+1}: A = {A1:.1f} +- {A1_err:.1f} / {A2:.2f} +- {A2_err:.1f}")
            print(f"Peak {i+1}: G = {G1:.4f} +- {G1_err:.4f} / {G2:.4f} +- {G2_err:.4f}")
            print(f"Peak {i+1}: v = {w1:.4f} +- {w1_err:.4f} / {w2:.4f} +- {w2_err:.4f}\n")

        G2_err = "$\inf$" if G2_err >1e3 else f"{G2_err:.3f}"

        table and print(f"\multirow{{2}}{{*}}{{\#%i}} & ${O1:.0f}\pm{O1_err:.0f}$ & ${A1:.0f}\pm{A1_err:.1f}$ & ${w1:.2f}\pm{w1_err:.2f}$ \
        %s & ${G1:.3f}\pm{G1_err:.3f}$ & Ch1 \\\\\n%s & ${O2:.0f}\pm{O2_err:.0f}$ & ${A2:.0f}\pm{A2_err:.1f}$ & ${w2:.2f}\pm{w2_err:.2f}$ \
        %s & ${G2:.3f}\pm{G2_err:s}$ & Ch2 \\\\\n%s\hline"%(i+1,"\b"*9,"\t"*3+"\b"*1,"\b"*9,"\t"*3+"\b"))
