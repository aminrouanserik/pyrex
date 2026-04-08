import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from qcextender.dimensionlesswaveform import DimensionlessWaveform
from scipy.interpolate import make_interp_spline

from pyrex.eccentric_fit import fitting_eccentric_function
from pyrex.functions import f_sin

e = DimensionlessWaveform.from_sim("SXS:BBH:0322")
qc = DimensionlessWaveform.from_sim("SXS:BBH:0318")

qcomega, eomega = qc.omega(), e.omega()
qcamp, eamp = qc.amp(), e.amp()

times = np.linspace(qc.time[0], -31, 50000)

# Correcting for length differences
index = np.abs(qc.time - e.time[0]).argmin()
qcomega_interp = make_interp_spline(qc.time[index:], qcomega[index:])(times)
qcamp_interp = make_interp_spline(qc.time[index:], qcamp[index:])(times)
eomega_interp = make_interp_spline(e.time, eomega)(times)
eamp_interp = make_interp_spline(e.time, eamp)(times)

e_omega = [
    (eomega_interp[i] - qcomega_interp[i]) / (2 * qcomega_interp[i])
    for i in range(len(times))
]
e_amp = [
    (eamp_interp[i] - qcamp_interp[i]) / (2 * qcamp_interp[i])
    for i in range(len(times))
]

omega_pwr = -59 / 24
amp_pwr = -83 / 24

e_omega_a, e_omega_b, e_omega_f, e_omega_phi = fitting_eccentric_function(
    omega_pwr, np.array(e_omega), qcomega_interp
)
e_amp_a, e_amp_b, e_amp_f, e_amp_phi = fitting_eccentric_function(
    amp_pwr, np.array(e_amp), qcamp_interp
)

# Shift is crucial
shift_omega = qcomega_interp[(np.abs(times - times[0])).argmin()]
shift_amp = qcamp_interp[(np.abs(times - times[0])).argmin()]

x_omega = qcomega_interp**omega_pwr - shift_omega**omega_pwr
x_amp = qcamp_interp**amp_pwr - shift_amp**amp_pwr

fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

axes[0].plot(
    qcomega_interp, e_omega, color=prop_cycle[0], label=r"$\mathrm{NR-derived}$"
)
axes[0].plot(
    qcomega_interp,
    f_sin(x_omega, e_omega_a, e_omega_b, e_omega_f, e_omega_phi),
    linestyle="dashed",
    color=prop_cycle[1],
    label=r"$\mathrm{Replicated}$",
)
axes[0].set_ylabel(r"$e_{\omega_{22}}$")
axes[0].set_xlabel(r"$\omega_c$")
axes[0].set_yticks([-0.06, -0.03, 0, 0.03, 0.06])
axes[0].xaxis.set_major_locator(ticker.MaxNLocator(6))

axes[1].plot(qcamp_interp, e_amp, color=prop_cycle[0], label=r"Fit")
axes[1].plot(
    qcamp_interp,
    f_sin(x_amp, e_amp_a, e_amp_b, e_amp_f, e_amp_phi),
    linestyle="dashed",
    color=prop_cycle[1],
    label=r"Fit",
)
axes[1].set_ylabel(r"$e_{\mathcal{A}_{22}}$")
axes[1].set_xlabel(r"$\mathcal{A}_c$")
axes[1].set_yticks([-0.06, -0.03, 0, 0.03, 0.06])
axes[1].xaxis.set_major_locator(ticker.MaxNLocator(6))

fig.suptitle("Parameterised Fit is Suboptimal")
handles, labels = axes[0].get_legend_handles_labels()
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
