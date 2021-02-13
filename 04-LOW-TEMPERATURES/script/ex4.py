#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
#from scipy.constants import k

data1 = np.loadtxt(f"../data/downsweep2.txt",unpack=True)
data2 = np.loadtxt(f"../data/upsweep.txt",unpack=True)
RSil = np.concatenate((data1[2],data2[2])) # Widerstand in Ohm
U= data1[0]
I=1

def temperature(R):
    if abs(R) < 12.15226:
        return 16.61 + 6.262 * abs(R) - 0.3695 * abs(R)**2 + 0.01245 * abs(R)**3
    elif abs(R) > 12.15226:
        return 31.95 + 2.353 * abs(R)

T1=[]
for i in range(len(U)):
    T1.append(temperature(U[i]))
T=np.concatenate((T1,data2[0])) #Temperatur in K

def linfit(x,A,B):
    return A*x+B
popt, pcov = curve_fit(linfit,1/data2[0][3:20],np.log(1/data2[2][3:20]))

perr= np.sqrt(np.diag(pcov))
print(popt[0],"+-",perr[0])

k=8.617343 *10**-2 # in meV/K

E= -2*k*popt[0] #meV
Eer=-2*k*perr[0] #meV
print(E,"+-",Eer)



h = np.arange(0,0.15,0.01)
plt.plot(h,linfit(h,popt[0],popt[1]),label="Fit")
plt.plot(1/T,np.log(1/RSil),'o',label="Data")

plt.xlabel("Inverse temperature in 1/K")
plt.ylabel("ln(1/R)")
plt.grid()
plt.legend()

plt.show()
