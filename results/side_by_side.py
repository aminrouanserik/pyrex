import matplotlib.pyplot as plt
import numpy as np
from qcextender import units
from qcextender.waveform import Waveform

from pyrex.eccentric_extension import generate_eccentric_waveform


def cut(wave):
    strain = []
    strain.append(
        wave[2, 2][
            np.where(wave.time > units.tM_to_tSI(-1500, wave.metadata.total_mass))
        ]
    )
    time = wave.time[
        np.where(wave.time > units.tM_to_tSI(-1500, wave.metadata.total_mass))
    ]
    return Waveform(strain, time, wave.metadata)


kwargs = {
    "mass1": 20,
    "mass2": 20,
    "inclination": 0,
    "coa_phase": 0,
    "delta_t": 1.0 / 16000,
    "f_lower": 20,
    "f_ref": 20,
    "distance": 10,
    "eccentricity": 0.093,
}

phen = cut(Waveform.from_model("IMRPhenomTE", [(2, 2)], **kwargs))
phen_ecc = cut(
    generate_eccentric_waveform(
        "IMRPhenomTE",
        [(2, 2)],
        "examples/sample_data/pyrexdata.pkl",
        cut=False,
        one_dimensional=True,
        **kwargs,
    )
)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].plot(phen_ecc.time, phen_ecc[2, 2], label="IMRPhenomT_Pyrex")
axes[0].plot(phen.time, phen[2, 2], label="IMRPhenomTE")
axes[0].set_ylabel("Strain (m)")
axes[0].set_xlabel("Time (s)")
axes[0].legend()

axes[1].plot(phen_ecc.time, phen_ecc.phase(), label="IMRPhenomT_Pyrex")
axes[1].plot(phen.time, phen.phase(), label="IMRPhenomTE")
axes[1].set_ylabel("Phase (rad)")
axes[1].set_xlabel("Time (s)")
axes[1].legend()

axes[2].plot(phen_ecc.time, phen_ecc.amp(), label="IMRPhenomT_Pyrex")
axes[2].plot(phen.time, phen.amp(), label="IMRPhenomTE")
axes[2].set_ylabel("Amplitude (m)")
axes[2].set_xlabel("Time (s)")
axes[2].legend()

plt.tight_layout()
plt.show()
