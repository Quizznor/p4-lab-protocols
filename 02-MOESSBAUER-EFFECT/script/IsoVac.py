#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True

from mössbauer_aux import perform_fits
from mössbauer_aux import create_bins
from mössbauer_aux import C_to_v
import numpy as np


channels, data = np.loadtxt("../data/Vacromium.txt",unpack=True)
(ch1, ch2), (d1, d2) = np.split(channels,2), np.split(data,2)

ch1_binned, ch2_binned = create_bins(ch1), create_bins(ch2)-512
d1_binned, d1_err_binned = create_bins(d1,np.sqrt(d1))
d2_binned, d2_err_binned = create_bins(d2,np.sqrt(d2))
d2_binned, d2_err_binned = d2_binned[::-1], d2_err_binned[::-1]
v1_binned, v2_binned = C_to_v(ch1_binned), C_to_v(ch2_binned)

d1_cuts = [0,-1]
d2_cuts = [0,-1]


fits_d1 = perform_fits(v1_binned,d1_binned,d1_err_binned,d1_cuts,ch1=True)
fits_d2 = perform_fits(v2_binned,d2_binned,d2_err_binned,d2_cuts,ch1=False)

w1er = []
w1 = []
w2er = []
w2 = []

ma = len(fits_d1) #Anzahl der Peaks
for i in range(ma):
    w1.append(fits_d1[i][0][3])
    w1er.append(np.sqrt( np.diag(fits_d1[i][1])[3]))
    w2.append(fits_d2[i][0][3])
    w2er.append(np.sqrt( np.diag(fits_d2[i][1])[3]))
#print(ma)
print(w1,w2)
g=[]
ger=[]

for i in range(1):
    g.append((w1[i]+w1[ma-1-i])/2)
    g.append((w2[i]+w2[ma-1-i])/2)
    ger.append(np.sqrt(w1er[i]**2+w1er[ma-1-i]**2)/2)
    ger.append(np.sqrt(w2er[i]**2+w2er[ma-1-i]**2)/2)


delt=np.mean(g)
delter=np.sqrt(np.sum(np.square(ger)))/len(g)  

stat = np.std(g)/np.sqrt(len(g))

#print(delt,"+-",delter+stat)

#umrechnung
#256 channel = 10 mm/s
#0.2mm/s = 10**-8 eV
#1 channel = 1.23046875 *10**-9 eV
si=1/0.2*10**(-8)
iso=delt*si
isoer=(stat+delter)*si
print(iso,"+-",isoer)
