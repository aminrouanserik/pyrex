import numpy as np
import matplotlib.pyplot as plt
from pyrex.main import *
from pyrex.core import *

sim = "SXS:BBH:1358"


eccentricity = 0.099
x = np.float64(0.14139682985076596)
distance, inclination, coa_phase = 10, 0, 0
spin1x, spin1z, spin1y = 0, 0, 0
spin2x, spin2z, spin2y = 0, 0, 0

mass1 = mass2 = 25

std_phenom = {
    "mass1": mass1,
    "mass2": mass2,
    "eccentricity": eccentricity,
    "approximant": "IMRPhenomD",
    "spin1x": spin1x,
    "spin1y": spin1y,
    "spin1z": spin1z,
    "spin2x": spin2x,
    "spin2y": spin2y,
    "spin2z": spin2z,
    "inclination": inclination,
    "distance": distance,
    "coa_phase": coa_phase,
    "f_lower": 25,
    "x": x,
}
std_seob = {
    "mass1": mass1,
    "mass2": mass2,
    "eccentricity": eccentricity,
    "approximant": "SEOBNRv4",
    "spin1x": spin1x,
    "spin1y": spin1y,
    "spin1z": spin1z,
    "spin2x": spin2x,
    "spin2y": spin2y,
    "spin2z": spin2z,
    "inclination": inclination,
    "distance": distance,
    "coa_phase": coa_phase,
    "f_lower": 25,
    "x": x,
}

phen_ecc = Cookware(**std_phenom)
seob_ecc = Cookware(**std_seob)

# Extensive cleaning needed, more cutting happens than should happen
plt.figure()
plt.plot(phen_ecc.time, phen_ecc.h22, label="SEOBNR")
plt.plot(seob_ecc.time, seob_ecc.h22, label="Phenom")
# plt.plot(dsim.time, dsim[2, 2], label="NR simulation")
# plt.ylim(-1e-18, 1e-18)
plt.legend()
plt.tight_layout()
plt.show()
