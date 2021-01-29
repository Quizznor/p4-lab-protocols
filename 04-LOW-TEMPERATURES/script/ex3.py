import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def temperature(R):
    return np.exp(1.116*np.log(R)/(-4.374+np.log(R)) - 1231 * np.log(R)/(9947+np.log(R)))

R0=966
T0 = temperature(float(R0))


deltax=np.array([0,12.5,9.5,6.5,13,7.5,8,8])*0.1 # in cm
deltaR= deltax * 20 # in Ohm
R=np.zeros(len(deltaR))
for i in range(len(deltaR)):
    R[i] = R0 + np.sum(deltaR[0:i+1])
#print(R)
T = temperature(R)
#print(T)

I = np.array([0,1.5,3,4.5,6,7.5,9,10.5]) # in A
B = np.array([0,0.071895,0.14379,0.215685,0.28758,0.359475,0.43137,0.503265]) # in Tesla

def linfit(x,A,B):
    return A*x+B

#popt, pcov = curve_fit(linfit,T,B)
popt, pcov = curve_fit(linfit,T[0:3],B[0:3])
#print(popt)
#print(pcov)
perr = np.sqrt(np.diag(pcov))
phi0= 2.07*np.float_power(10, -15)
E = np.sqrt(-phi0/(2*np.pi*T0*popt[0]))
Eer = 0.5*E*perr[0]/popt[0]

print(E,"+-",Eer)

l = np.power(E,2)/39*np.power(10,9)
ler= 2*l*Eer/E
print(l,"+-",ler)





plt.plot(T,B,'o',label="Data")
h= np.arange(8.2,9.4,0.1)
plt.plot(h,linfit(h,popt[0],popt[1]),label="Fit")

plt.xlabel("Temperature in K")
plt.ylabel("Magnetic field in T")
plt.grid()
plt.legend()
plt.show()
