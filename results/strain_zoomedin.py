import matplotlib.patches as patches
import matplotlib.pyplot as plt
from qcextender.waveform import Waveform

from pyrex.eccentric_extension import generate_eccentric_waveform

kwargs = {
    "mass1": 15,
    "mass2": 15,
    "inclination": 0,
    "coa_phase": 0,
    "delta_t": 1.0 / 4096,
    "f_lower": 10,
    "f_ref": 20,
    "distance": 100,
    "eccentricity": 0.1,
}

phen = Waveform.from_model("IMRPhenomTE", [(2, 2)], **kwargs)
phen_ecc = generate_eccentric_waveform(
    "IMRPhenomTE",
    [(2, 2)],
    "examples/sample_data/pyrexdata_egw.pkl",
    cut=False,
    **kwargs,
)

zoom_t_min = phen.time[10000]
zoom_t_max = phen.time[-1]
zoom_y_min = min(phen_ecc[2, 2][10000:]) * 1.2
zoom_y_max = max(phen_ecc[2, 2][10000:]) * 1.2

fig, axes = plt.subplots(1, 2, figsize=(10, 5), width_ratios=[0.65, 0.35])

ax_full = axes[0]
ax_full.plot(phen_ecc.time, phen_ecc[2, 2], label="IMRPhenomTPyrex")
ax_full.plot(phen.time, phen[2, 2], label="IMRPhenomTE")

ax_full.set_title("Full Waveform", fontsize=14)
ax_full.set_ylabel("Strain (m)")
ax_full.set_xlabel("Time (s)")

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
ax_zoom.plot(phen_ecc.time, phen_ecc[2, 2], label="IMRPhenomTPyrex")
ax_zoom.plot(phen.time, phen[2, 2], label="IMRPhenomTE")

ax_zoom.set_xlim(zoom_t_min, zoom_t_max)
ax_zoom.set_ylim(zoom_y_min, zoom_y_max)
ax_zoom.set_title("Around merger", fontsize=14)
ax_zoom.set_xlabel("Time (s)")
ax_zoom.legend(loc="upper right")

plt.tight_layout()
plt.show()
