#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

def prepare_data(data,bin_tails=False):
    x,y = data

    # Eliminate y and x offset
    y -= np.mean(y)
    x -= np.mean(x)


    # Discontinued this because it is probably useless.
    # # Bin the hysteresis tails
    # if bin_tails:
    #
    #     xmax, xmin = np.argmax(x), np.argmin(x)
    #
    #     # bin the upper end of the hysteresis
    #     x_up_low,y_up_low = [x[xmax-i] for i in range(int(len(x)/8))], [y[xmax-i] for i in range(int(len(x)/8))]
    #     x_up_high,y_up_high = [x[xmax+i] for i in range(int(len(x)/5))], [y[xmax+i] for i in range(int(len(x)/5))]

    return x,y

x1,y1 = prepare_data(np.loadtxt("../data/sample_A_10.DAT",unpack=True),bin_tails=True)
x2,y2 = prepare_data(np.loadtxt("../data/sample_A_17.DAT",unpack=True),bin_tails=True)

#plt.axhline(y=0.9*max(y1))
# plt.scatter(x1,y1)
# plt.scatter(x2,y2)
plt.scatter(x3,y3)
plt.show()
