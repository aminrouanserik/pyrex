import numpy as np
import matplotlib.pyplot as plt
from qcextender.dimensionlesswaveform import DimensionlessWaveform
from qcextender.waveform import Waveform
from pyrex.main import *
from pyrex.core import *

sim = "SXS:BBH:1155"
dimensionless = DimensionlessWaveform.from_sim(sim)

eccentricity = 0
x = np.float64(0.14139682985076596)
distance, inclination, coa_phase = 10, 0, 0
spin1x, spin1z, spin1y = 0, 0, 0
spin2x, spin2z, spin2y = 0, 0, 0

mass1 = mass2 = 12.5

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

phen_ecc = Cookware(**std_phenom).get_wave()
seob_ecc = Cookware(**std_seob).get_wave()

kwargs = {
    "mass1": 12.5,
    "mass2": 12.5,
    "inclination": 0,
    "coa_phase": 0,
    "delta_t": 1.0 / 4196,
    "f_lower": 25,
    "f_ref": 25,
    "distance": 10,
}

phenom = Waveform.from_model("IMRPhenomD", [(2, 2)], **kwargs)
seob = Waveform.from_model("SEOBNRv4", [(2, 2)], **kwargs)

plt.plot(phen_ecc.time, phen_ecc[2, 2], label="IMRPhenomD_Pyrex")
plt.plot(seob_ecc.time, seob_ecc[2, 2], label="SEOBNRv4_Pyrex")
plt.plot(phenom.time, phenom[2, 2], label="IMRPhenomD")
plt.plot(seob.time, seob[2, 2], label="SEOBNRv4")
# plt.plot(dimension.time, dimension[2, 2], label="NR simulation")
# plt.ylim(-1e-18, 1e-18)
plt.legend()
plt.tight_layout()
plt.show()

plt.plot(phen_ecc.time, phen_ecc.phase(), label="IMRPhenomD_Pyrex")
plt.plot(seob_ecc.time, seob_ecc.phase(), label="SEOBNRv4_Pyrex")
plt.plot(phenom.time, phenom.phase(), label="IMRPhenomD")
plt.plot(seob.time, seob.phase(), label="SEOBNRv4")
plt.xlabel("Time (s)")
plt.ylabel("Phase")
plt.legend()
plt.show()

plt.plot(phen_ecc.time, phen_ecc.amp(), label="IMRPhenomD_Pyrex")
plt.plot(seob_ecc.time, seob_ecc.amp(), label="SEOBNRv4_Pyrex")
plt.plot(phenom.time, phenom.amp(), label="IMRPhenomD")
plt.plot(seob.time, seob.amp(), label="SEOBNRv4")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (m)")
plt.legend()
plt.show()
