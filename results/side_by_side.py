import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
    "mass1": 30,
    "mass2": 30,
    "inclination": 0,
    "coa_phase": 0,
    "delta_t": 1.0 / 4096,
    "f_lower": 10,
    "f_ref": 20,
    "distance": 100,
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

fig, axes = plt.subplots(
    2, 2, figsize=(12, 7), sharex="col", height_ratios=[5 / 7, 2 / 7]
)

axes[0, 0].plot(
    units.tSI_to_tM(phen_ecc.time, 60),
    phen_ecc.phase(),
    label=r"$\mathrm{IMRPhenomTPyrex}$",
)
axes[0, 0].plot(
    units.tSI_to_tM(phen.time, 60), phen.phase(), label=r"$\mathrm{IMRPhenomTE}$"
)
axes[0, 0].set_ylabel(r"$\phi_{22}$")
axes[0, 0].set_xlim(units.tSI_to_tM(phen.time, 60)[0], 100)
axes[0, 0].set_ylim(-10, 160)
axes[0, 0].set_yticks([0, 40, 80, 120, 160])
axes[0, 0].xaxis.set_major_locator(mticker.MaxNLocator(6))

axes[0, 1].plot(
    units.tSI_to_tM(phen_ecc.time, 60),
    units.hSI_to_hM(phen_ecc.amp(), 60, 100),
    label="IMRPhenomTPyrex",
)
axes[0, 1].plot(
    units.tSI_to_tM(phen.time, 60),
    units.hSI_to_hM(phen.amp(), 60, 100),
    label="IMRPhenomTE",
)
axes[0, 1].set_ylabel(r"$\mathcal{A}_{22}$")
axes[0, 1].set_yticks([0, 0.1, 0.2, 0.3, 0.4])
axes[0, 1].set_xlim(units.tSI_to_tM(phen.time, 60)[0], 100)
axes[0, 1].xaxis.set_major_locator(mticker.MaxNLocator(6))

axes[1, 0].plot(
    units.tSI_to_tM(phen.time, 60), phen.phase() - phen_ecc.phase(), label="IMRPhenomTE"
)
axes[1, 0].set_ylabel(r"$\Delta\phi_{22}$")
axes[1, 0].set_xlabel(r"$\mathrm{t}/\mathrm{M}$")
axes[1, 0].set_xlim(units.tSI_to_tM(phen.time, 60)[0], 100)

axes[1, 1].plot(
    units.tSI_to_tM(phen_ecc.time, 60),
    units.hSI_to_hM(phen_ecc.amp(), 60, 100) - units.hSI_to_hM(phen.amp(), 60, 100),
)
axes[1, 1].set_ylabel(r"$\Delta \mathcal{A}_{22}$")
axes[1, 1].set_xlabel(r"$\mathrm{t}/\mathrm{M}$")
axes[1, 1].set_xlim(units.tSI_to_tM(phen.time, 60)[0], 100)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.055),
    ncols=2,
)
fig.align_ylabels()


plt.tight_layout()
plt.show()
