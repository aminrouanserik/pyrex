"""
Utilities for fitting eccentricity-induced corrections to gravitational-wave
amplitude and instantaneous frequency using SXS numerical relativity simulations.

This module implements the workflow described in
`fit_waveform_eccentricity`, namely:

1. Loading SXS waveforms and extracting (or computing) their reference
   eccentricities and mass ratios.
2. Separating simulations into circular and eccentric sets.
3. Interpolating all waveforms onto a common time grid.
4. Constructing lookup tables for circular amplitude and frequency.
5. Computing eccentric contributions to amplitude and omega.
6. Fitting these contributions to a damped sinusoidal model.

The primary output is a pickle file containing mass ratios, reference
eccentricities, geometric frequency, and best-fit coefficients
for both amplitude and frequency modulations. These fits are intended for
use in surrogate modeling, waveform augmentation, or studies of
eccentricity evolution in NR simulations.

The module consists of helper functions for:
- downloading and preprocessing waveforms (`components`)
- constructing circular lookup tables (`construct_lookup`)
- computing eccentric contributions (`get_e_X`)
- fitting sinusoidal eccentricity corrections (`fit_sin`, `fitting_eccentric_function`)
- PN-related quantities (`calculate_x`)
- serializing results (`write_pkl`)
"""

import pickle
from collections.abc import Callable

import numpy as np
from gw_eccentricity import measure_eccentricity
from numpy.typing import NDArray
from qcextender.dimensionlesswaveform import DimensionlessWaveform
from scipy.interpolate import make_interp_spline
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

from pyrex.functions import f_sin


def fit_waveform_eccentricity(
    names: list[str],
    outfname: str,
    q: list[float] | None = None,
    e_ref: list[float] | None = None,
) -> None:
    """
    Fit the eccentricity-induced corrections to both the amplitude and the
    instantaneous gravitational wave frequency for a set of SXS simulations.

    This routine assumes that **each mass ratio included in the fit has a
    corresponding zero eccentricity (circular) simulation**, which is required
    to extract eccentric contributions consistently.

    The procedure:
        1. Load simulations and extract (or compute) reference eccentricities
           and mass ratios, if not given by the user.
        2. Separate simulations into circular and eccentric subsets based on a
           threshold eccentricity.
        3. Interpolate all waveforms onto a common time grid.
        4. Construct circular lookups for amplitude and omega.
        5. For each eccentric simulation, compute the eccentric contribution to
           omega and amplitude relative to the circular reference.
        6. Fit the eccentricity modulation model and store all fit parameters.

    Args:
        names (list[str]):
            List of SXS simulation identifiers.
        outfname (str):
            Filename (without extension) to which the resulting fit parameters
            will be written as a pickle file (``outfname.pkl``).
        q (list[float] | None, optional):
            Mass ratios corresponding to the simulations.
            If ``None`` (default), mass ratios are extracted automatically from
            the simulation metadata.
        e_ref (list[float] | None, optional):
            Eccentricities evaluated at the reference frequency for each
            simulation.
            If ``None`` (default), these values are computed using
            ``gw_eccentricity`` with the ``Amplitude`` method.

    Output:
        A pickle file ``outfname.pkl`` containing a dictionary with:
            - ``"q"``: list[float]
            - ``"e_ref"``: list[float]
            - ``"x"``: list[NDArray]
            - ``A_omega, B_omega, freq_omega, phi_omega``: NDArray[np.floating]
            - ``A_amp, B_amp, freq_amp, phi_amp``: NDArray[np.floating]
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
    write_pkl(outfname, results)


def components(name: str) -> tuple[DimensionlessWaveform, float, float]:
    """
    Downloads (or loads from cache) the SXS waveform with the given `name` ID.
    Returns the simulation as a `DimensionlessWaveform`, the eccentricity
    computed via `gw_eccentricity`, and the mass ratio extracted from the
    simulation metadata.

    Args:
        name (str): The simulation ID as in the SXS catalog.

    Returns:
        tuple[DimensionlessWaveform, float, float]:
            A tuple containing:
            - DimensionlessWaveform: The waveform loaded in from SXS and processed by ``qcextender``.
            - float: The eccentricity as calculated by ``gw_eccentricity`` using the ``Amplitude`` method.
            - float: The mass ratio taken from the ``reference_mass_ratio`` metadata entry.
    """
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
    """
    Creates and returns a circular `DimensionlessWaveform` component lookup dictionary, containing amplitude and instantaneous frequency,
    with the mass ratio as key. Recasts the circular `DimensionlessWaveform` to a new time grid using a spline.

    Args:
        circ_waves (list[DimensionlessWaveform]): A list of circular `DimensionlessWaveform`
        objects with which to construct the lookup object.
        new_time (NDArray[np.floating]): The time grid the components are cast to.

    Returns:
        dict[float, tuple[NDArray[np.floating], NDArray[np.floating]]]:
            Dictionary indexed by mass ratio, where each value is a tuple:
            (omega_resampled, amplitude_resampled), both cast to `new_time`.
    """
    circ_lookup = {}
    for c in circ_waves:
        time = c.time
        mask = (time >= -1500) & (time <= -29)
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
    """
    Compute the eccentric contribution to a waveform component X
    (either amplitude or instantaneous frequency).

    The quantity is defined as:
        e_X = (X_ecc - X_circ) / (2 * X_circ)

    where:
        - X_ecc is the component extracted from the eccentric waveform,
        - X_circ is the corresponding circular reference component.

    Args:
        wave (DimensionlessWaveform): Eccentric SXS simulation.
        eccentricity (float): The eccentricity calculated using `gw_eccentricity`.
        circ_vals (NDArray[np.floating]): The circular value of the component X from the lookup.
        new_time (NDArray[np.floating]): The time grid to which the circular components were cast.
        get_component (Callable[[DimensionlessWaveform], NDArray[np.floating]]): Either w.amp()
            or w.omega() with w the eccentric waveform.
        filter_order (int, optional): Polynomial order for the Savitzky-Golay filter. Defaults to 2.

    Returns:
        NDArray[np.floating]: The eccentric contribution to X, the amplitude or instantaneous frequency.
    """
    mask = (wave.time >= -1500) & (wave.time <= -29)
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
) -> tuple[float, float, float, float]:
    """
    Fits the eccentric component X to a sinusoidal model of the form

        f(x) = A * exp(B * x) * sin(freq * x / (2π) + phase),

    where x is defined as (circ_X**power - circ_X[0]**power).

    Args:
        power (float): Power used in transforming the circular component.
        ecc_X (NDArray): Eccentric waveform component.
        circ_X (NDArray): Circular waveform component.

    Returns:
        tuple[float, float, float, float]: (A, B, freq, phase) from the fit.
    """
    x = (circ_X) ** power - (circ_X[0]) ** power
    y = ecc_X
    par = fit_sin(x, y)
    return par


def fit_sin(
    xdata: NDArray[np.floating], ydata: NDArray[np.floating]
) -> tuple[float, float, float, float]:
    """
    Fits the data (xdata, ydata) to the sinusoidal model `f_sin` using
    bounded nonlinear least squares.

    Args:
        xdata (NDArray[np.floating]): Input x-values.
        ydata (NDArray[np.floating]): Input y-values to fit.

    Returns:
        tuple[float, float, float, float]:
            The best-fit parameters (A, B, freq, phase).
    """
    lower_bounds = [-np.inf, 0.0, -np.inf, -np.inf]
    upper_bounds = [np.inf, 1e-2, np.inf, np.inf]

    popt, _ = curve_fit(
        f_sin,
        xdata,
        ydata,
        p0=[0.1, 5e-4, 0.1, -2000],
        bounds=(lower_bounds, upper_bounds),
    )

    return popt


def calculate_x(
    old_time: NDArray[np.floating],
    omega: NDArray[np.floating],
    new_time: NDArray[np.floating],
) -> float:
    """
    Computes the geometric frequency x = omega(t)^(2/3) at the first point of `new_time`,
    where omega(t) is obtained by spline-interpolating the provided frequency array.

    Args:
        old_time (NDArray[np.floating]): Time grid corresponding to the input `omega`.
        omega (NDArray[np.floating]): Orbital frequency values.
        new_time (NDArray[np.floating]): Time grid on which to evaluate x.

    Returns:
        float: The value of omega(new_time[0])^(2/3).
    """
    interp_omega = make_interp_spline(old_time, omega)
    x = interp_omega(new_time[0]) ** (2 / 3)
    return x


def write_pkl(
    outfname: str, data_dict: dict[str, NDArray[np.floating] | list[float]]
) -> None:
    """
    Writes the provided dictionary to a pickle file.

    Args:
        outfname (str): Path to the output pickle file.
        data_dict (dict[str, NDArray[np.floating] | list[float]]):
            Dictionary containing arrays or lists to serialize.
    """
    f = open(outfname, "wb")
    pickle.dump(data_dict, f)
    f.close()
