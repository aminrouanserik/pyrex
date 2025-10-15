from pyrex.core import main
import matplotlib.pyplot as plt

kwargs = {
    "mass1": 25,
    "mass2": 25,
    "inclination": 0,
    "coa_phase": 0,
    "delta_t": 1.0 / 16300,
    "f_lower": 20,
    "f_ref": 25,
    "distance": 10,
    "eccentricity": 0,
    "spin1": (0, 0, 0),
    "spin2": (0, 0, 0),
}

phen_ecc = main("IMRPhenomD", [(2, 2)], **kwargs)
seob_ecc = main("SEOBNRv4", [(2, 2)], **kwargs)

plt.plot(phen_ecc.time, phen_ecc[2, 2], label="IMRPhenomD_Pyrex")
plt.plot(seob_ecc.time, seob_ecc[2, 2], label="SEOBNRv4_Pyrex")
# plt.plot(phenom.time, phenom[2, 2], label="IMRPhenomD")
# plt.plot(seob.time, seob[2, 2], label="SEOBNRv4")
# plt.plot(dimension.time, dimension[2, 2], label="NR simulation")
# plt.ylim(-1e-18, 1e-18)
plt.legend()
plt.tight_layout()
plt.show()
