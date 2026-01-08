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


def isco_freq(wave: Waveform):
    G = 6.67e-11
    m = 1.9e30
    c = 299792458
    f_max = c**3 / (6 ** (3 / 2) * np.pi * G * wave.metadata.total_mass * m)
    omegas = wave.omega()
    indices = np.where(omegas < 2 * np.pi * f_max)[0]
    if len(indices) == 0:
        mask = np.array([], dtype=int)
    else:
        breaks = np.where(np.diff(indices) != 1)[0] + 1
        segments = np.split(indices, breaks)

        mask = max(segments, key=len)

    strain = []
    strain.append(wave.amp()[mask] * np.exp(1j * wave.phase()[mask]))

    return Waveform(strain, wave.time[mask], wave.metadata)


q = 1
spin1 = (0, 0, 0)
spin2 = (0, 0, 0)
distance, inclination, coa_phase = 10, 0, 0
f_lower = 20

eccentricities = np.linspace(1e-6, 0.2, 50)
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
            "delta_t": 1.0 / 5000,
            "f_ref": (0.075 / 50) ** (3 / 2) * 2 * np.pi,
            "f_lower": f_lower,
        }

        phen_ecc = generate_eccentric_waveform(
            "IMRPhenomTE",
            [(2, 2)],
            "examples/sample_data/pyrexdata_egw.pkl",
            **kwargs,
        )

        # kwargs.pop("eccentricity")
        phen_circ = cut(Waveform.from_model("IMRPhenomTE", [(2, 2)], **kwargs))
        try:
            mismatches[i, j] = 1 - phen_ecc.match(phen_circ, f_lower)
        except:
            continue

plt.figure(figsize=(8, 6))
plt.imshow(
    mismatches,
    origin="lower",
    aspect="auto",
    extent=[
        total_masses.min(),
        total_masses.max(),
        eccentricities.min(),
        eccentricities.max(),
    ],
    cmap="viridis",
    norm=colors.LogNorm(
        vmin=np.nanmin(mismatches[mismatches > 0]),  # avoid log(0)
        vmax=np.nanmax(mismatches),
    ),
)
plt.colorbar(label=r"$_{10}\log(\mathcal{M}$)")
plt.xlabel(r"Total Mass [M$_{\odot}$]")
plt.ylabel(r"$e_{gw}$")
plt.title("Mismatch between IMRPhenomTE and IMRPhenomTPyrex")
plt.tight_layout()
plt.show()
