import matplotlib.colors as colors
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


q = 1
spin1 = (0, 0, 0)
spin2 = (0, 0, 0)
distance, inclination, coa_phase = 10, 0, 0
f_lower = 10

eccentricities = np.linspace(0, 0.2, 50)
total_masses = np.linspace(20, 150, 50)

mismatches = np.zeros((len(eccentricities), len(total_masses)))

for i, e in enumerate(eccentricities):
    for j, M in enumerate(total_masses):
        mass1 = q * M / (q + 1)
        mass2 = M / (q + 1)
        kwargs = {
            "mass1": mass1,
            "mass2": mass2,
            "eccentricity": e,
            "spin1": spin1,
            "spin2": spin2,
            "inclination": inclination,
            "distance": distance,
            "coa_phase": coa_phase,
            "delta_t": 1.0 / 4096,
            "f_ref": 20,
            "f_lower": f_lower,
        }

        phen_ecc = generate_eccentric_waveform(
            "IMRPhenomTE",
            [(2, 2)],
            "examples/sample_data/pyrexdata_egw.pkl",
            **kwargs,
        )
        phen_circ = cut(Waveform.from_model("IMRPhenomTE", [(2, 2)], **kwargs))
        try:
            mismatches[i, j] = 1 - phen_ecc.match(phen_circ, f_lower)
        except:
            continue

plt.imshow(
    mismatches,
    origin="lower",
    aspect="auto",
    extent=(
        total_masses.min(),
        total_masses.max(),
        eccentricities.min(),
        eccentricities.max(),
    ),
    norm=colors.LogNorm(
        vmin=np.nanmin(mismatches[mismatches > 0]),
        vmax=np.nanmax(mismatches),
    ),
)
plt.colorbar(label=r"$\mathcal{M}$")
plt.xlabel(r"$M$")
plt.ylabel(r"$e_{\mathrm{gw}}$")
plt.yticks([0, 0.05, 0.1, 0.15, 0.2])

plt.title("Mismatch Between IMRPhenomTE and IMRPhenomTPyrex, After Fixes")
plt.tight_layout()
plt.show()
