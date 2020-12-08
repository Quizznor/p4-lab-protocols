#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True

from mössbauer_aux import perform_fits
from mössbauer_aux import create_bins
from mössbauer_aux import C_to_v

import numpy as np
#from scipy.constants import c

channels, data = np.loadtxt("../data/Iron.txt",unpack=True)
(ch1, ch2), (d1, d2) = np.split(channels,2), np.split(data,2)

ch1_binned, ch2_binned = create_bins(ch1), create_bins(ch2)-512
d1_binned, d1_err_binned = create_bins(d1,np.sqrt(d1))
d2_binned, d2_err_binned = create_bins(d2,np.sqrt(d2))
d2_binned, d2_err_binned = d2_binned[::-1], d2_err_binned[::-1]
v1_binned, v2_binned = C_to_v(ch1_binned), C_to_v(ch2_binned)

d1_cuts = [0,42,54,64,74,86,-1]
d2_cuts = [0,42,53,64,73,84,-1]


fits_d1 = perform_fits(v1_binned,d1_binned,d1_err_binned,d1_cuts,ch1=True)
fits_d2 = perform_fits(v2_binned,d2_binned,d2_err_binned,d2_cuts,ch1=False)

w1er = []
w1 = []
w2er = []
w2 = []

ma = len(fits_d1) #Anzahl der Peaks
for i in range(ma):#Auslesen
    w1.append(fits_d1[i][0][3])
    w1er.append(np.sqrt( np.diag(fits_d1[i][1])[3]))
    w2.append(fits_d2[i][0][3])
    w2er.append(np.sqrt( np.diag(fits_d2[i][1])[3]))

#print(w1)
#print(w2)
g=[]
ger=[]
for i in range(int(ma*0.5)):#diff der peak paare
    g.append((w1[i]+w1[ma-1-i])/2)
    g.append((w2[i]+w2[ma-1-i])/2)
    ger.append(np.sqrt(w1er[i]**2+w1er[ma-1-i]**2)/2)
    ger.append(np.sqrt(w2er[i]**2+w2er[ma-1-i]**2)/2)
    
#print(g)
delt=np.mean(g)#delta berechnen in mm/s
delter=np.sqrt(np.sum(np.square(ger))) / len(g)  

stat = np.std(g)/np.sqrt(len(g))
print("Isometrieverschiebung in mm/s")
print(delt,"+-",delter+stat)
'''
umrechnung
256 channel = 10 mm/s
0.2mm/s = 10**-8 eV
1 channel = 1.953125 *10**-9 eV

umrechnung
256 channel = 6.3 mm/s
0.2mm/s = 10**-8 eV
1 channel = 1.23046875 *10**-9 eV
'''
si=1/0.2*10**(-8)
iso=delt*si      #in eV
isoer=(stat+delter)*si
print("Isometrieverschiebung in eV")
print(iso,"+-",isoer)



#magnetfeld

vl = (w1 + delt ) /1000 # peaks zu geschwindigkeit
vr = (w2 + delt ) /1000# m/s
v=np.array([vl]+[vr])
vler=np.sqrt(np.square(delter+stat)+np.square(w1er)) /1000
vrer=np.sqrt(np.square(delter+stat)+np.square(w2er)) /1000

#print(vl)
#print("+-",v1er)
#print(np.sum(v)) # sollte gegen null
'''
#4Faelle mit je links/rechts
F1 = []
F2 = []
F3 = []
F4 = []


for j in range(2):
    F1.append(v[j][5]-2*v[j][4]-v[j][3])
    F2.append(v[j][5]-2*v[j][4]+v[j][3])
    F3.append(v[j][5]-v[j][4]-2*v[j][3])
    F4.append(-v[j][5]+v[j][4]+2*v[j][3])
    
F=np.array([F1]+[F2]+[F3]+[F4])

#print(F)# Fall 2 ist am naechsten
'''
c = 299792458 #m/s
E0=14400 #eV
E0er = 50#eV
muk = 3.15 * 10**(-8)# in eV/T Kernmagneton
mug = 0.0903 * muk #eV/T
muger = 0.0007 * muk# in eV/T
Ig = 0.5 # grundzustand spin
Ia = 1.5 # angeregter zst


B1 = (Ig*E0* (-vl[4]-vl[3]) ) / (mug*c)
B2 = (Ig*E0* (-vr[4]-vr[3]) ) / (mug*c)


B1stat = E0*Ig /(mug*c)*np.sqrt(vler[4]**2+vler[3]**2) 
B2stat = E0*Ig /(mug*c)*np.sqrt(vrer[4]**2+vrer[3]**2)

B1sys = np.sqrt(E0er**2 * ((vler[4]+vler[3])*Ig/(mug*c))**2  +  muger**2 * ((vler[4]+vler[3])*E0*Ig/(mug**2*c))**2)
B2sys = np.sqrt(E0er**2 * ((vrer[4]+vrer[3])*Ig/(mug*c))**2  +  muger**2 * ((vrer[4]+vrer[3])*E0*Ig/(mug**2*c))**2)
print("Magnetfeld in T")
print(B1,"+-",B1sys+B1stat)
print(B2,"+-",B2sys+B2stat)
B = (B1+B2)/2
B1er = B1sys+B1stat
B2er = B2sys+B2stat
Ber = 0.5 *np.sqrt(B1er**2+B2er**2) + np.std(np.array([B1]+[B2]))/np.sqrt(2)
print(B,"+-",Ber)


mua1 = ((vl[5]-vl[4])*Ia*E0)  /  (B1*c)
mua2 = ((vr[5]-vr[4])*Ia*E0)  /  (B2*c)


mua1stat = np.sqrt( vler[5]**2 * (Ia*E0/B1/c)**2 + vler[4]**2 *(Ia*E0/B1/c)**2 + B1er**2 * ((vl[5]-vl[4])*Ia*E0/B1**2/c)**2)
mua2stat = np.sqrt( vrer[5]**2 * (Ia*E0/B2/c)**2 + vrer[4]**2 *(Ia*E0/B2/c)**2 + B2er**2 * ((vr[5]-vr[4])*Ia*E0/B2**2/c)**2)

mua1sys = np.sqrt( E0er**2 * ((vl[5]-vl[4])*Ia/B1/c)**2 )
mua2sys = np.sqrt( E0er**2 * ((vr[5]-vr[4])*Ia/B2/c)**2 )

mua1er = mua1sys+mua1stat
mua2er = mua2sys+mua2stat
mua = (mua1+mua2)/2
muaer = 0.5 *np.sqrt(mua1er**2+mua2er**2) + np.std(np.array([mua1]+[mua2]))/np.sqrt(2)

print("Magnetisches Moment in eV/T")
print(mua1,"+-",mua1sys+mua1stat)
print(mua2,"+-",mua2sys+mua2stat)
print(mua,"+-",muaer)





