import matplotlib.pyplot as plt
import qcextender.units as units
from qcextender.waveform import Waveform

from pyrex.eccentric_extension import generate_eccentric_waveform

kwargs = {
    "mass1": 18,
    "mass2": 18,
    "inclination": 0,
    "coa_phase": 0,
    "delta_t": 1.0 / 4096,
    "f_lower": 10,
    "f_ref": 20,
    "distance": 100,
    "eccentricity": 0.1,
}

phen = Waveform.from_model("IMRPhenomTE", **kwargs)
phen_ecc = generate_eccentric_waveform(
    "IMRPhenomTE",
    [(2, 2)],
    "examples/sample_data/pyrexdata_egw.pkl",
    cut=False,
    **kwargs,
)

fig, axes = plt.subplots(1, 1, figsize=(10, 5))

ax_full = axes
ax_full.plot(
    units.tSI_to_tM(phen_ecc.time, 36),
    units.hSI_to_hM(phen_ecc[2, 2], 36, 100),
    label=r"$\mathrm{IMRPhenomTPyrex}$",
)
ax_full.plot(
    units.tSI_to_tM(phen.time, 36),
    units.hSI_to_hM(phen[2, 2], 36, 100),
    label=r"$\mathrm{IMRPhenomTE}$",
)

ax_full.set_ylabel(r"$\Re h_{22}$")
ax_full.set_xlabel(r"$\mathrm{t}/\mathrm{M}$")
ax_full.set_yticks([-1, -0.5, 0, 0.5, 1])
ax_full.set_xlim(units.tSI_to_tM(phen.time, 36)[0], units.tSI_to_tM(phen.time, 36)[-1])
ax_full.set_title("Long Modes Fail at Early Times")

handles, labels = axes.get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.055),
    ncols=2,
)

plt.tight_layout()
plt.show()
