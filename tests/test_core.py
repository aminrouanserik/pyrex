import numpy as np
import matplotlib.pyplot as plt
from qcextender.dimensionlesswaveform import DimensionlessWaveform
from qcextender.waveform import Waveform
from pyrex.main import *
from pyrex.core import *

sim = "SXS:BBH:1155"
dimensionless = DimensionlessWaveform.from_sim(sim)

eccentricity = 0
distance, inclination, coa_phase = 10, 0, 0
spin1x, spin1z, spin1y = 0, 0, 0
spin2x, spin2z, spin2y = 0, 0, 0

mass1 = mass2 = 25

dimension = dimensionless.to_Waveform(25, mass1 + mass2, distance)

kwargs = {
    "mass1": 10,
    "mass2": 10,
    "inclination": 0,
    "coa_phase": 0,
    "delta_t": 1.0 / 4196,
    "f_lower": 20,
    "f_ref": 25,
    "distance": 10,
}

phenom = Waveform.from_model("IMRPhenomD", [(2, 2)], **kwargs)
seob = Waveform.from_model("SEOBNRv4", [(2, 2)], **kwargs)

kwargs = {
    "mass1": 10,
    "mass2": 10,
    "inclination": 0,
    "coa_phase": 0,
    "delta_t": 1.0 / 4196,
    "f_lower": 20,
    "f_ref": 25,
    "distance": 10,
    "eccentricity": 0,
}

phen_ecc = main("IMRPhenomD", [(2, 2)], **kwargs)
seob_ecc = main("SEOBNRv4", [(2, 2)], **kwargs)

print(seob.match(seob_ecc))
print(phenom.match(phen_ecc))
print(dimension.match(seob_ecc))
print(dimension.match(phen_ecc))

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
