#!/usr/bin/env py

# Don't create a pycache directory from importing auxiliary files
import sys
sys.dont_write_bytecode = True

import numpy as np
from ecal_gauge import E, E_err

data_bkg, data_sig = [], []
angles = np.arange(0,101,10)

for angle in np.arange(0,101,10):
    channel, bkg = np.loadtxt(f"../data/background_{angle}.txt",unpack=True)
    channel, sig = np.loadtxt(f"../data/compton_{angle}.txt",unpack=True)
    data_bkg.append(bkg), data_sig.append(sig)
    channels = channel


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import warnings
    warnings.simplefilter('ignore', UserWarning)

    plt.rcParams.update({'font.size': 16})
    plt.rc('axes', labelsize=26)

    # set up big subplot for global axes and stuff
    fig = plt.figure(frameon=False)
    fig.subplots_adjust(hspace = .3, wspace=.001)
    ax_global = fig.add_subplot(111,xticks=[],yticks=[],frameon=False)
    ax_global.set_xlabel("Energy (keV)",labelpad=30)
    ax_global.set_ylabel("Total event count",labelpad=50)

    # for angle in np.arange(0,101,10): # plotting the whole thing
    for i,angle in enumerate([100,40,0],1): # plotting just parts for better visualisation
        channel, bkg = np.loadtxt(f"../data/background_{angle}.txt",unpack=True)
        channel, sig = np.loadtxt(f"../data/compton_{angle}.txt",unpack=True)

        # set up subplot for angle measurement
        # ax = fig.add_subplot(3,4,angle/10+1) # plotting the whole thing
        x, y = 0 if i<3 else 1, 0 if i!=2 else 1
        xlim = 800 if i<3 else 1500
        span = 1 if i<3 else 2

        ax = plt.subplot2grid((2,2), (x,y), colspan=span)

        ax.plot(E(channel),bkg,c="C0",label="no target")
        ax.plot(E(channel),sig,c="C1",label="Al target")
        ax.set_title(r"$\theta$ = {}$\!^\circ$".format(angle))

        # shorten yticks for more beautiful plot
        locs = ax.get_yticks()
        labels = [str(loc)[:-2] for loc in locs]                  # Turning floats to ints (getting rid of the trailing '.0')
        labels = [label[::-1] for label in labels]                # Reversing the string for easier replacement patterns
        labels = [label.replace("000","k") for label in labels]   # Shortening strings to thousands, if applicable
        labels = [label.replace("kk","M") for label in labels]    # Shortening strings to millions, if applicable
        labels = [label[::-1] for label in labels]                # rereversing the strings and setting them as new labels
        ax.set_yticklabels(labels)
        ax.set_yticks(locs)
        ax.set_xticks([0,500,1000,1500])

        # indicate where the expected compton energy is
        E_gamma, m_e = 661.659, 510.999
        energy_theo = E_gamma/(1 + E_gamma/m_e * (1-np.cos(2*np.pi*angle/360)))
        channel_theo = (energy_theo - 30.2)/5.8
        ax.axvline(energy_theo,c="k",ls="--")
        ax.axvspan(energy_theo - E_err(channel_theo),energy_theo + E_err(channel_theo),alpha=0.3,color="k")

        ax.set_ylim(0,1.35*max(sig))
        ax.set_xlim(35,xlim)

        if i==3:
            plt.legend(fontsize=20)

    ax_global.plot(1,1,c="C0",label="no target")
    ax_global.plot(1,1,c="C1",label="Al-target")

    # ax_global.legend(loc="lower right",labelspacing=2,fontsize=20,borderpad=1.4,borderaxespad=0,framealpha=1)
    plt.show()
