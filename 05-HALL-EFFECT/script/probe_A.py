#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

# Probe A
T_A, Ulp_A, Uhp_A, Uln_A, Uhn_A, Ulm_A, Uhm_A = np.loadtxt("../data/hall_A.txt",unpack=True)
# Conduction_A, Hall_A = [Ulp_A, Uln_A, Ulm_A], [Uhp_A*1e-3, Uhn_A*1e-3, Uhm_A*1e-3]

Current_A = np.array([1e-3 if T_A[i] < 50 else 1.8e-3 for i in range(len(T_A))]) # in Ampere

plt.plot(T_A,sigma_A)
plt.show()
