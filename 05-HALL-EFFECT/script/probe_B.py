#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

d = np.loadtxt(f"../data/hall_B.txt",unpack=True) * 10**(-3)# da in mV
ger = np.loadtxt(f"../data/mobility_A.txt",unpack=True)

Uer= 0.5 * 10**(-6)

T = d[0] *1000 + 273.15 #in K
Ter = 1.5

Uleit = []
Uleiter=[]
for i in range(len(d[1])):
    Uleit.append((d[1][i]+d[3][i]+d[5][i])/3)
    Uleiter.append(3**(-1/2)*Uer)

B = 0.5 # Tesla
dU = (d[2]-d[6])/2 - d[4]
dUer= np.sqrt(3/2 * Uer)

mu = np.log(2) /np.pi * np.abs(dU) /Uleit /B
muer= np.log(2) /np.pi*np.sqrt((dUer /B /Uleit)**2 + (Uleiter *np.abs(dU) / B * np.power(Uleit,-2) )**2)

#print(mu,"+-",muer) # in SI in m**2 / V /s

def polfit(x,A,B,C,D):
    return A*x**3+B*x**2+C*x+D
popt, pcov = curve_fit(polfit,T,mu)

h = np.arange(50,200,1)
np.savetxt("gaas.txt", [mu,muer])
#plt.plot(h,polfit(h,popt[0],popt[1],popt[2],popt[3]),label="Fit")
#print(polfit(300,popt[0],popt[1],popt[2],popt[3]))

plt.plot(T,mu,"x",label="GaAs")
plt.plot(ger[0],ger[1],"x",label="Ge")
plt.xlabel("Temperatur in K")
plt.ylabel("Beweglichkeit in m²/ Vs")
plt.grid()
plt.legend()
plt.show()
