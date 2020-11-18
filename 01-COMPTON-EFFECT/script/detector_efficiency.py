import glob
import numpy as np


# detector efficiency, data points read off from data sheet
energies = np.arange(250,601,50)
epsilon = 1/np.array([1.29,1.40,1.54,1.68,1.80,1.90,2.00,2.07])

params, cov = np.polyfit(energies,epsilon, 2, cov=True)
efficiency_err = lambda x: np.sqrt( np.array([x**2,x,1]).T @ cov @ np.array([x**2,x,1]) )
efficiency = np.poly1d(params)

if __name__ == '__main__':

    import warnings
    warnings.simplefilter('ignore', np.VisibleDeprecationWarning)

    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 16})
    plt.rc('axes', labelsize=26)

    X = np.linspace(250,600,1000)
    upper = efficiency(X) + efficiency_err(X)
    lower = efficiency(X) - efficiency_err(X)
    plt.plot(X,efficiency(X))
    plt.fill_between(X,upper,lower,alpha=0.4)
    plt.scatter(energies,epsilon)

    plt.show()
