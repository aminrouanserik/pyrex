import numpy as np
import matplotlib.pyplot as plt
from qcextender.dimensionlesswaveform import DimensionlessWaveform
from pyrex.main import *
from pyrex.core import *

sim = "SXS:BBH:1358"
dimensionless = DimensionlessWaveform.from_sim(sim)

eccentricity = 0.16475441548479375
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

dimension = dimensionless.to_Waveform(25, mass1 + mass2, distance)

phen_ecc = Cookware(**std_phenom)
seob_ecc = Cookware(**std_seob)

plt.plot(phen_ecc.time, phen_ecc.h22, label="SEOBNR")
plt.plot(seob_ecc.time, seob_ecc.h22, label="Phenom")
plt.plot(dimension.time, dimension[2, 2], label="NR simulation")
# plt.ylim(-1e-18, 1e-18)
plt.legend()
plt.tight_layout()
plt.show()
