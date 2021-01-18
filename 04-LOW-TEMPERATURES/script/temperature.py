import numpy as np
import sys

def temperature(R):

    if sys.argv[2] == "platin":
        if abs(R) > 2.543490 and abs(R) < 12.15226:
            return 16.6 + 6.262 * abs(R*1e-3) - 0.3695 * abs(R*1e-3)**2 + 0.01245 * abs(R*1e-3)**3
        elif abs(R) > 12.15226:
            return 31.95 + 2.353 * abs(R)

    if sys.argv[2] == "coal":
        return np.exp(1.116*np.log(R)/(-4.374+np.log(R)) - 1231 * np.log(R)/(9947+np.log(R)))


T = temperature(float(sys.argv[1]))

if __name__ == "__main__":
    print("\n"+f"T = {T:.3f} K ~ {T-273:.3f} °C"+"\n")
