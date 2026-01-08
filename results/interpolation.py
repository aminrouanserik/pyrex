import matplotlib.pyplot as plt
import numpy as np

from pyrex.eccentric_extension import get_fit_params, read_pkl

training_dict = read_pkl("examples/sample_data/pyrexdata.pkl")

eccentricities = np.linspace(0, 0.2, 50)
mass_ratios = np.linspace(1, 3, 50)

x = (20 / (2 * np.pi)) ** (2 / 3)

grid = np.zeros((len(mass_ratios), len(eccentricities)))
grid2 = np.zeros((len(mass_ratios), len(eccentricities)))

for i, e in enumerate(eccentricities):
    for j, q in enumerate(mass_ratios):
        par_omega_1d, par_amp_1d = get_fit_params(
            training_dict, q, e, x, one_dimensional=True
        )
        par_omega_2d, par_amp_2d = get_fit_params(
            training_dict, q, e, x, one_dimensional=False
        )
        grid[i, j] = par_omega_1d[1]
        grid2[i, j] = par_omega_2d[1]

fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)

im1 = axes[0].imshow(
    grid,
    origin="lower",
    aspect="auto",
    extent=[
        mass_ratios.min(),
        mass_ratios.max(),
        eccentricities.min(),
        eccentricities.max(),
    ],
    cmap="cividis",
)
axes[0].set_title(r"B$_\omega$ 1-Dimensional Interpolation")
axes[0].set_xlabel(r"$q$")
axes[0].set_ylabel(r"$e_{gw}$")
fig.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(
    grid2,
    origin="lower",
    aspect="auto",
    extent=[
        mass_ratios.min(),
        mass_ratios.max(),
        eccentricities.min(),
        eccentricities.max(),
    ],
    cmap="cividis",
)
axes[1].set_title(r"B$_\omega$ 2-Dimensional Interpolation")
axes[1].set_xlabel(r"$q$")
fig.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.show()
