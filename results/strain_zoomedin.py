import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from qcextender import units
from qcextender.waveform import Waveform

from pyrex.eccentric_extension import generate_eccentric_waveform

# --- your data prep ---
kwargs = {
    "mass1": 15,
    "mass2": 15,
    "inclination": 0,
    "coa_phase": 0,
    "delta_t": 1.0 / 5000,
    "f_lower": 20,
    "f_ref": (0.075 / 50) ** (3 / 2) * 2 * np.pi,
    "distance": 10,
}

phen = Waveform.from_model("IMRPhenomTE", [(2, 2)], **kwargs)
kwargs.update({"eccentricity": 0.1})
phen_ecc = generate_eccentric_waveform(
    "IMRPhenomTE",
    [(2, 2)],
    "examples/sample_data/pyrexdata_egw.pkl",
    cut=False,
    **kwargs,
)

zoom_t_min = phen.time[12443]
zoom_t_max = phen.time[-1]
zoom_y_min = min(phen_ecc[2, 2][12443:]) * 1.2
zoom_y_max = max(phen_ecc[2, 2][12443:]) * 1.2

fig, axes = plt.subplots(1, 2, figsize=(8, 5), width_ratios=[0.7, 0.3])

ax_full = axes[0]
ax_full.plot(phen_ecc.time, phen_ecc[2, 2], label="IMRPhenomTPyrex")
ax_full.plot(phen.time, phen[2, 2], label="IMRPhenomD")

ax_full.set_title("Full Waveform", fontsize=14)
ax_full.set_ylabel("Strain (m)")
ax_full.set_xlabel("Time (s)")
ax_full.legend(frameon=False)

rect = patches.Rectangle(
    (zoom_t_min, zoom_y_min),
    zoom_t_max - zoom_t_min,
    zoom_y_max - zoom_y_min,
    linewidth=1.3,
    edgecolor="black",
    linestyle="--",
    facecolor="none",
)
ax_full.add_patch(rect)


ax_zoom = axes[1]
ax_zoom.plot(phen_ecc.time, phen_ecc[2, 2], label="IMRPhenomDPyrex")
ax_zoom.plot(phen.time, phen[2, 2], label="IMRPhenomD")

ax_zoom.set_xlim(zoom_t_min, zoom_t_max)
ax_zoom.set_ylim(zoom_y_min, zoom_y_max)
ax_zoom.set_title("Zoomed Region", fontsize=14)
# ax_zoom.set_ylabel("Strain (m)")
ax_zoom.set_xlabel("Time (s)")
ax_zoom.legend(frameon=False)

# conns = [
#     ((zoom_t_min, zoom_y_max), (zoom_t_min, zoom_y_max)),
#     ((zoom_t_max, zoom_y_max), (zoom_t_max, zoom_y_max)),
# ]

plt.tight_layout()
plt.show()
