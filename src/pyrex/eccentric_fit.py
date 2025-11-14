import pickle
from collections.abc import Callable

import numpy as np
from gw_eccentricity import measure_eccentricity
from numpy.typing import NDArray
from qcextender.dimensionlesswaveform import DimensionlessWaveform
from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter

from pyrex.tools import calculate_x, fit_sin


def fit_waveform_eccentricity(
    names: list[str],
    outfname: str,
    q: list[float] | None = None,
    e_ref: list[float] | None = None,
) -> None:
    """Fits eccentric contributions to the amplitude and instantaneous frequency. Please make sure that every
    mass ratio has a complimentary zero eccentricity simulation.

    Args:
        names (list[str]): List of SXS simulation names of the binary simulations.
        outfname (str): The filename in which to save the fit parameters.
        q (list[float], optional): List of mass ratios of the binary simulations. Defaults to None, in which case the simulation
        metadata is used.
        e_ref (list[float], optional): List of eccentricities at the reference frequency of the binary simulations.
        Defaults to None, in which case it is calculted by `gw_eccentricity`, using the `Amplitude` method.

    Output:
        File called outfname.pkl.
    """
    waves = []

    if not q:
        q = []

    if not e_ref:
        e_ref = []
        for name in names:
            sim, e, mass_ratio = components(name)
            e_ref.append(e)
            q.append(mass_ratio)
            waves.append(sim)
    else:
        for name in names:
            sim, _, _ = components(name)
            waves.append(sim)

    threshold = 1e-4
    circ_waves = []
    ecc_waves, ecc_q, ecc_e = [], [], []

    for w, e, qv in zip(waves, e_ref, q):
        if e < threshold:
            circ_waves.append(w)
        else:
            ecc_waves.append(w)
            ecc_q.append(qv)
            ecc_e.append(e)

    # Need a common time grid for all waves for this code to work. Boundaries and length can be tinkered with
    new_time = np.linspace(-1500.0, -29, 15221)

    circ_lookup = construct_lookup(circ_waves, new_time)
    circ_lookup_omega = {q: pair[0] for q, pair in circ_lookup.items()}
    circ_lookup_amp = {q: pair[1] for q, pair in circ_lookup.items()}

    omega_params, amp_params, x = [], [], []
    for ew, e, mass_ratio in zip(ecc_waves, ecc_e, ecc_q):
        if mass_ratio not in circ_lookup:
            raise ValueError(f"No circular reference for q={mass_ratio}")

        circ_omega = circ_lookup_omega[mass_ratio]
        circ_amp = circ_lookup_amp[mass_ratio]

        e_omega = get_e_X(ew, e, circ_omega, new_time, lambda w: w.omega())
        e_amp = get_e_X(ew, e, circ_amp, new_time, lambda w: w.amp(), filter_order=3)

        omega_params.append(fitting_eccentric_function(-59 / 24, e_omega, circ_omega))
        amp_params.append(fitting_eccentric_function(-83 / 24, e_amp, circ_amp))

        x.append(calculate_x(ew.time, ew.omega(), new_time))

    omega_array, amp_array = np.array(omega_params), np.array(amp_params)

    results: dict[str, NDArray[np.floating] | list[float]] = {
        "q": ecc_q,
        "e_ref": ecc_e,
        "x": x,
        "A_omega": omega_array[:, 0],
        "B_omega": omega_array[:, 1],
        "freq_omega": omega_array[:, 2],
        "phi_omega": omega_array[:, 3],
        "A_amp": amp_array[:, 0],
        "B_amp": amp_array[:, 1],
        "freq_amp": amp_array[:, 2],
        "phi_amp": amp_array[:, 3],
    }

    # write and store the data
    if outfname:
        write_pkl(outfname, results)


def components(name: str) -> tuple[DimensionlessWaveform, float, float]:
    fref_in = 0.0075
    sim = DimensionlessWaveform.from_sim(name)
    try:
        result = measure_eccentricity(
            fref_in=fref_in,
            method="Amplitude",
            dataDict={"t": sim.time, "hlm": {(2, 2): sim[2, 2]}},
        )
        e = float(result["eccentricity"])
    except Exception:
        e = 0.0
    return sim, e, round(sim.metadata.q, 2)


def construct_lookup(
    circ_waves: list[DimensionlessWaveform], new_time: NDArray[np.floating]
) -> dict[float, tuple[NDArray[np.floating], NDArray[np.floating]]]:
    circ_lookup = {}
    for c in circ_waves:
        time = c.time
        mask = (time > (time[0] + 250)) & (time <= -29)
        t = time[mask]
        omega = c.omega()[mask]
        amp = c.amp()[mask]
        q_key = round(c.metadata.q, 2)
        circ_lookup[q_key] = (
            make_interp_spline(t, omega)(new_time),
            make_interp_spline(t, amp)(new_time),
        )
    return circ_lookup


def get_e_X(
    wave: DimensionlessWaveform,
    eccentricity: float,
    circ_vals: NDArray[np.floating],
    new_time: NDArray[np.floating],
    get_component: Callable[[DimensionlessWaveform], NDArray[np.floating]],
    filter_order: int = 2,
) -> NDArray[np.floating]:
    mask = (wave.time > (wave.time[0] + 250)) & (wave.time <= -29)
    ecc_interp = make_interp_spline(wave.time[mask], get_component(wave)[mask])
    ecc_vals = ecc_interp(new_time)

    e_X = (ecc_vals - circ_vals) / (2.0 * circ_vals)
    if eccentricity > 0:
        e_X = savgol_filter(e_X, 501, filter_order)
    return e_X


def fitting_eccentric_function(
    power: float,
    ecc_X: NDArray[np.floating],
    circ_X: NDArray[np.floating],
) -> NDArray[np.floating]:
    x = (circ_X) ** power - (circ_X[0]) ** power
    y = ecc_X
    par, _ = fit_sin(x, y)
    return par


def write_pkl(
    outfname: str, data_dict: dict[str, NDArray[np.floating] | list[float]]
) -> None:
    """Writes a dictionary to a pickle file in a specified directory.

    Args:
        outfname (str): Path to the file to write to.
        data_dict (dict): Dictionary to write to the file.
    """
    f = open(outfname, "wb")
    pickle.dump(data_dict, f)
    f.close()
