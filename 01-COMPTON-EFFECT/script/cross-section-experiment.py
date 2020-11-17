#!/usr/bin/env py

import glob

background = glob.glob("../data/background_*.txt")
compton = glob.glob("../data/compton_*.txt")

print(len(background))
print(compton)
