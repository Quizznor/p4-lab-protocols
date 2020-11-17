#!/usr/bin/env py

import glob
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 16})
plt.rc('axes', labelsize=26)

background = glob.glob("../data/background_*.txt")
compton = glob.glob("../data/compton_*.txt")

# set up big subplot for global axes and stuff
fig = plt.figure(frameon=False)
fig.subplots_adjust(hspace = .5, wspace=.001)
ax_global = fig.add_subplot(111,xticks=[],yticks=[],frameon=False)
ax_global.set_xlabel("MAC Channel number",labelpad=30)
ax_global.set_ylabel("Total event count",labelpad=50)

for i in range(len(background)):
    angle = int(i*10)
    channel, bkg = np.loadtxt(background[i],unpack=True)
    channel, sig = np.loadtxt(compton[i],unpack=True)

    # set up subplot for angle measurement
    ax = fig.add_subplot(3,4,i+1)
    ax.plot(channel,bkg,c="C0")
    ax.plot(channel,sig,c="C1")
    ax.set_xticks([0,100,200])
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

    ax.set_ylim(0,1.35*max(sig))
    ax.set_xlim(0,300)

ax_global.plot(1,1,c="C0",label="no target")
ax_global.plot(1,1,c="C1",label="Al-target")

ax_global.legend(loc="lower right",labelspacing=2,fontsize=20,borderpad=1.4,borderaxespad=0,framealpha=1)
plt.show()
