#!/usr/bin/env py

# Don't create a pycache directory from importing auxiliary files
import sys
sys.dont_write_bytecode = True

# main program
import numpy as np
import matplotlib.pyplot as plt
from detector_efficiency import efficiency, efficiency_err
from differential_cross_section import dOmega,dOmega_err
from differential_cross_section import phi_0, phi_0_err
from dataclasses import dataclass
from ecal_gauge import E, E_err

@dataclass
class Element:
    symbol: str
    name: str
    atomic_number: int
    atomic_mass: float
    density: float
    count_total: float = None
    cross_section_relative: float = None
    factor: int = 1

elements = [
	Element('Al', 'aluminium', 13, 26.981539, 2.7),
	Element('Fe', 'iron', 26, 55.845, 7.874),
	Element('Cu', 'copper', 29, 63.546, 8.96),
	Element('Pb', 'lead', 82, 207.2, 11.34,2)
]

channels, bkg = np.loadtxt("../data/background_20.txt",unpack=True)
print()

for element in elements:

    channels, sig = np.loadtxt(f"../data/material_{element.symbol}_20.txt",unpack=True)
    R_corr = (element.factor * np.sum(sig - bkg))/300
    R_corr_err = (element.factor * np.sqrt( np.sum(sig + bkg) ))/300
    rel_diff = (R_corr * element.atomic_mass)/(element.atomic_number * element.density)
    rel_diff_err = rel_diff * R_corr_err/R_corr

    print(f"{element.symbol} & {element.atomic_number}/{element.atomic_mass:.3f} & {R_corr:.2f}$\pm${R_corr_err:.2f} & {rel_diff:.2f}$\pm${rel_diff_err:.2f} \\\\")
