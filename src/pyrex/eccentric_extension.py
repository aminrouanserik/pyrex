"""
Eccentric gravitational-waveform construction.

This module contains all routines required to augment a quasi-circular
gravitational waveform with an eccentricity correction based on precomputed
training data. The workflow includes:

1. Loading a dictionary of fitted parameters encoding eccentric modulation of
   orbital frequency and amplitude.
2. Interpolating these parameters over mass ratio, eccentricity, and the
   geometric frequency x.
3. Reconstructing the early-time eccentric amplitude and phase by applying
   parametrized exponential-sine fits and integrating the reconstructed
   frequency.
4. Selecting and stitching the reconstructed segment to the late-time portion
   of the original waveform with proper phase alignment.
5. Smoothing the transition region using linear stitching and Savitzky-Golay
   filtering.

The high-level function `generate_eccentric_waveform` wraps this entire
pipeline and returns an eccentric `Waveform` instance compatible.
"""

import pickle

import numpy as np
from numpy.typing import NDArray
from qcextender import units
from qcextender.waveform import Waveform
from scipy import integrate
from scipy.interpolate import RBFInterpolator, interp1d
from scipy.signal import savgol_filter

from pyrex.functions import f_sin


def generate_eccentric_waveform(
    approximant: str,
    mode: list[tuple[int, int]],
    dirfile: str = "examples/sample_data/pyrexdata.pkl",
    cut: bool = True,
    **kwargs,
) -> Waveform:
    """
    Generate a waveform with an eccentricity correction applied.

    This function constructs a quasi-circular waveform using the specified
    `approximant` and `mode`, then augments it with eccentricity information
    using precomputed training data. The eccentricity must be passed in
    `kwargs` under the key `"eccentricity"`.

    Note:
        - The underlying eccentricity correction uses the callable `construct`
            and the training dictionary loaded from `dirfile`.

    Args:
        approximant (str):
            Name of the underlying quasi-circular waveform model
            (e.g., "IMRPhenomD", "TaylorF2", etc.).
        mode (list[tuple[int, int]]):
            List of `(l, m)` harmonic modes to generate for the base waveform.
        dirfile (str, optional):
            Path to the pickle file containing the eccentricity-interpolation
            training dictionary. Defaults to `"../examples/sample_data/pyrexdata.pkl"`.
        cut (bool, optional):
            Whether to truncate or condition the waveform after the
            eccentricity correction step at -1500M. Defaults to `True`.
        **kwargs:
            Additional keyword arguments passed to `Waveform.from_model`.
            Must include:
                eccentricity (float): The target eccentricity. If zero,
                a small nonzero value (`1e-30`) is substituted for numerical stability.
            Other keys are forwarded directly to the underlying waveform model.

    Raises:
        KeyError:
            If `"eccentricity"` is not provided in `kwargs`.
        FileNotFoundError:
            If the specified `dirfile` does not exist.

    Returns:
        (Waveform): A new `Waveform` instance with eccentricity applied via
            `wave.add_eccentricity(...)`.
    """
    eccentricity = kwargs.pop("eccentricity")
    if eccentricity == 0:
        eccentricity = 1e-30
    elif eccentricity > 1:
        raise ValueError(
            "An eccentricity above 1 is not physical, please change this parameter."
        )
    elif eccentricity > 0.2:
        raise Warning("This eccentricity is beyond the calibration range.")
    wave = Waveform.from_model(approximant, mode, **kwargs)

    training_dict = read_pkl(dirfile)

    kwargs = {
        "training_dict": training_dict,
        "q": wave.metadata.q,
        "eccentricity": eccentricity,
        "cut": cut,
    }
    newwave = wave.add_eccentricity(construct, kwargs, eccentricity)
    return newwave


def construct(
    wave: Waveform,
    mode: tuple[int, int],
    training_dict: dict[str, NDArray[np.floating] | list[float]],
    q: float,
    eccentricity: float,
    cut: bool,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """
    Construct an eccentric waveform segment by stitching together the
    eccentric reconstruction and the remaining portion of the base
    quasi-circular waveform.

    This function:
        1. Computes the early-time amplitude and phase corrected for the
           target eccentricity using `eccentric_from_circular`.
        2. Extracts the late-time part of the original waveform starting
           at the mask boundary.
        3. Aligns phases at the stitching point.
        4. Concatenates early and late segments into continuous
           time/phase/amplitude arrays.
        5. Smooths the joints to avoid numerical discontinuities.

    Args:
        wave (Waveform):
            The original quasi-circular waveform object to extend with
            eccentricity corrections.
        mode (tuple[int, int]):
            The `(l, m)` mode being constructed. Used for waveform
            components within `wave`.
        training_dict (dict):
            Dictionary of precomputed eccentric-interpolation data
            (e.g., amplitudes, phases, eccentricity grids). Typically
            loaded from a pickle file and used by `eccentric_from_circular`.
        q (float):
            Mass ratio of the binary system. Required by the interpolation model.
        eccentricity (float):
            Target eccentricity at the reference frequency for the
            reconstruction. A zero value should already have been replaced
            upstream with a small non-zero proxy for numerical stability.
        cut (bool):
            Whether to truncate/condition the early-time portion during
            the eccentric reconstruction step.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            A tuple containing:
                - `time_construct`: 1D array of the stitched time samples.
                - `phase_construct`: 1D array of the eccentric-corrected
                  orbital (or GW) phase.
                - `amp_construct`: 1D array of the reconstructed amplitude.
    """
    # masked_waveform returns one extra index to allow for phase alignment
    early_time, amp_rec, phase_rec, mask = eccentric_from_circular(
        wave, training_dict, q, eccentricity, cut
    )
    late_time, late_amp, late_phase = sliced_waveform(wave, mask[0][-1])

    # Makes sure phases align properly
    phase_rec += late_phase[0] - phase_rec[-1]

    # Cut off from one to correct for the extra entry allowing for the phase line-up
    amp_construct = np.concatenate((amp_rec, late_amp[1:]))
    phase_construct = np.concatenate((phase_rec, late_phase[1:]))
    time_construct = np.concatenate((early_time, late_time[1:]))

    assert wave.metadata.total_mass is not None

    # Necessary now before solving ivp
    amp_construct = smooth_joint(
        time_construct,
        amp_construct,
        wave.metadata.total_mass,
    )
    phase_construct = smooth_joint(
        time_construct,
        phase_construct,
        wave.metadata.total_mass,
    )

    return time_construct, phase_construct, amp_construct


def sliced_waveform(
    wave: Waveform, index: int
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """
    Slice a waveform at a given index and return the late-time (post-index)
    segment of its time, amplitude, and phase arrays.

    Args:
        wave (Waveform):
            The full waveform object containing time, amplitude, and phase data.
        index (int):
            The starting index for the slice. All returned arrays begin at
            `wave.time[index]` and extend to the end of the waveform.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            A tuple consisting of:
                - `near_merger_time`: Sliced time array from `index` onward.
                - `new_amp`: Amplitude array sliced from `index` onward.
                - `new_phase`: Phase array sliced from `index` onward.
    """
    time = wave.time

    near_merger_time = time[index:]
    new_amp = wave.amp()[index:]
    new_phase = wave.phase()[index:]
    return near_merger_time, new_amp, new_phase


def eccentric_from_circular(
    wave: Waveform,
    training_dict: dict[str, NDArray[np.floating] | list[float]],
    q: float,
    eccentricity: float,
    cut: bool,
    phase_pwr: float = -59.0 / 24,
    amp_pwr: float = -83.0 / 24,
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    tuple[NDArray[np.intp], ...],
]:
    """
    Reconstruct the early-time eccentric amplitude and phase of a waveform
    based on a circular baseline and pre-fitted eccentricity modulation
    parameters.

    This function:
    - Converts the circular waveform to mass-rescaled units.
    - Selects a time window appropriate for applying the eccentric correction.
    - Retrieves the fitted sinusoidal modulation parameters for amplitude and
      frequency from the training dictionary.
    - Applies the eccentric corrections using parametrized exponential sine
      fits.
    - Reconstructs the eccentric frequency, integrates it to obtain phase,
      and rescales amplitude back into SI units.

    Args:
        wave (Waveform):
            The circular waveform from which eccentric corrections will
            be constructed.
        training_dict (dict[str, NDArray | list[float]]):
            The dictionary containing the precomputed fit parameters for
            eccentric modulation, typically generated from training data.
        q (float):
            The symmetric mass ratio or mass ratio parameter used to
            select the appropriate fitting coefficients.
        eccentricity (float):
            The target eccentricity at reference frequency for which the
            reconstruction is performed.
        cut (bool):
            Whether to restrict the reconstruction to an early-time window
            (e.g., `-1500 < t < -29` in mass-rescaled units). If `False`,
            only the upper bound is applied.
        phase_pwr (float, optional):
            Power-law exponent used when fitting the frequency-domain
            modulation. Defaults to -59/24.
        amp_pwr (float, optional):
            Power-law exponent used when fitting the amplitude modulation.
            Defaults to -83/24.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
            A tuple containing:
            - `new_time`: The rescaled time array (in SI units) corresponding
              to the early eccentric segment.
            - `amp_rec`: The reconstructed eccentric amplitude segment.
            - `phase_rec`: The reconstructed eccentric phase segment obtained
              by integrating the reconstructed frequency.
            - `mask`: A tuple of index arrays used to select the portion of the
              circular waveform on which eccentric corrections were applied.
    """
    assert wave.metadata.total_mass is not None
    assert wave.metadata.distance is not None

    time = units.tSI_to_tM(wave.time, wave.metadata.total_mass)
    omega = units.fSI_to_fM(wave.omega(), wave.metadata.total_mass)
    amp = units.mSI_to_mM(wave.amp(), wave.metadata.total_mass, wave.metadata.distance)

    # assert isinstance(time, np.ndarray)
    assert isinstance(omega, np.ndarray)

    # Lower bound dependent on the fit, in this case -1500
    if cut:
        mask = np.where((time > -1500) & (time < -29))
    else:
        mask = np.where((time < -29))
    new_time = time[mask]

    x = omega[mask][0] ** (2 / 3)

    # check requirements
    par_omega, par_amp = get_fit_params(training_dict, q, eccentricity, x)

    if not np.any(omega):
        amp_rec = np.zeros(len(new_time))
        phase_rec = np.zeros(len(new_time))
    else:
        omega_circ = omega[mask]
        amp_circ = amp[mask]

        # Arbitrary shift is crucial
        shift_omega = omega[(np.abs(new_time + 1500)).argmin()]
        shift_amp = amp[(np.abs(new_time + 1500)).argmin()]

        x_omega = omega_circ**phase_pwr - shift_omega**phase_pwr
        x_amp = amp_circ**amp_pwr - shift_amp**amp_pwr

        fit_ex_omega = f_sin(
            x_omega, par_omega[0], par_omega[1], par_omega[2], par_omega[3]
        )
        fit_ex_amp = f_sin(x_amp, par_amp[0], par_amp[1], par_amp[2], par_amp[3])

        omega_rec = fit_ex_omega * 2 * omega_circ + omega_circ
        amp_rec = fit_ex_amp * 2 * amp_circ + amp_circ

        new_time = units.tM_to_tSI(new_time, wave.metadata.total_mass)

        # Minus sign crucial for circ
        phase_rec = integrate.cumulative_trapezoid(
            units.fM_to_fSI(-omega_rec, wave.metadata.total_mass), new_time, initial=0
        )
        amp_rec = units.mM_to_mSI(
            amp_rec, wave.metadata.total_mass, wave.metadata.distance
        )

    return new_time, amp_rec, phase_rec, mask  # type: ignore


def get_fit_params(
    training_dict: dict[str, NDArray[np.floating] | list[float]],
    q: float,
    eccentricity: float,
    x: float,
) -> tuple[list[float], list[float]]:
    """
    Interpolate the precomputed sinusoidal fit parameters for the eccentric
    modulation of frequency and amplitude, given the target mass ratio,
    eccentricity, and power-law-transformed variable ``x``.

    This function extracts the training data (mass ratios, eccentricities,
    and x-values) along with their associated fitted parameters. It then
    interpolates these parameters to the desired (q, e, x) point.

    Args:
        training_dict (dict[str, NDArray | list[float]]):
            Dictionary containing training grids for mass ratio, eccentricity,
            power-law variable ``x``, and the fitted parameters for the
            frequency and amplitude eccentric corrections.
        q (float):
            The mass-ratio value for which to obtain interpolated parameters.
        eccentricity (float):
            The eccentricity at reference frequency for which the parameters
            should be interpolated.
        x (float):
            The power-law-transformed variable (derived from circular ω or
            amplitude) at the target evaluation point.

    Returns:
        tuple[list[float], list[float]]:
            Two lists of parameters:

            - ``omega_params``: ``[A_ω, B_ω, freq_ω, phi_ω]``
              Parameters governing the eccentric modulation of the frequency.

            - ``amp_params``: ``[A_amp, B_amp, freq_amp, phi_amp]``
              Parameters governing the eccentric modulation of the amplitude.

            These parameter sets can be passed to the sinusoidal reconstruction
            function ``f_sin`` to evaluate eccentric corrections.
    """
    train_q, train_ecc, train_x, omega, amp = get_noncirc_params(training_dict)

    training_quant = [train_q, train_ecc, train_x]
    test_quant = [q, eccentricity, x]

    A_omega, B_omega, freq_omega, phi_omega = interpol_key_quant(
        training_quant, omega, test_quant
    )
    A_amp, B_amp, freq_amp, phi_amp = interpol_key_quant(
        training_quant, amp, test_quant
    )

    omega_params = [A_omega, B_omega, freq_omega, phi_omega]
    amp_params = [A_amp, B_amp, freq_amp, phi_amp]

    return omega_params, amp_params


def interpol_key_quant(
    training_quant: list[list[float]],
    training_keys: list[NDArray[np.floating]],
    test_quant: list[float],
) -> tuple[float, float, float, float]:
    """
    Interpolates the fitted sinusoidal parameters (A, B, freq, phi) for the
    eccentric correction, given training grids and the target (q, e, x).

    Args:
        training_quant (list[list[float]]):
            Lists of training values for mass ratio, eccentricity, and x.
        training_keys (list[NDArray[np.floating]]):
            Arrays containing the fitted parameters A, B-related quantity,
            frequency-related quantity, and phase from training data.
        test_quant (list[float]):
            The target values [q, eccentricity, x] at which to interpolate.

    Returns:
        tuple[float, float, float, float]:
            The interpolated parameters A, B, freq, and phi.
    """
    A = interpolate_quantities(
        training_quant[1],
        training_quant[0],
        np.abs(training_keys[0]),
        test_quant[1],
        test_quant[0],
    )

    B = np.log(
        interpolate_quantities(
            training_quant[1],
            training_quant[0],
            np.exp(training_keys[1]) * np.abs(training_keys[0]),
            test_quant[1],
            test_quant[0],
        )
        / A
    )
    freq = np.sqrt(
        1.0
        / (
            interpolate_quantities(
                training_quant[1],
                training_quant[0],
                1.0 / np.asarray(training_keys[2]) ** 2,
                test_quant[1],
                test_quant[0],
            )
        )
    )
    phi = interp1D(
        np.asarray(training_quant[2]),
        np.asarray(training_keys[3]),
        test_quant[2],
    )

    e = test_quant[1]
    A = A * (e / (e + 1e-6))

    return float(A), float(B), float(freq), float(phi)


def smooth_joint(
    time: NDArray[np.floating], y: NDArray[np.floating], total_mass: float
) -> NDArray[np.floating]:
    """
    Smooths the transition region of a waveform quantity by linearly stitching
    a small window near merger and applying a Savitzky-Golay filter.

    Args:
        time (NDArray[np.floating]):
            The time array of the waveform (in SI units).
        y (NDArray[np.floating]):
            The waveform quantity to be smoothed (amplitude or phase).
        total_mass (float):
            Total mass of the binary, used for converting M-units to SI time.

    Returns:
        NDArray[np.floating]: The smoothed waveform array.
    """
    tarray = np.where(
        (time < units.tM_to_tSI(-25, total_mass))
        & (time >= units.tM_to_tSI(-46, total_mass))
    )

    first = tarray[0][0]
    last = tarray[0][-1]
    y[first:last] = np.interp(
        time[first:last], [time[first], time[last]], [y[first], y[last]]
    )

    y_inter = savgol_filter(y, 31, 3)
    return y_inter


def interp1D(
    trainkey: NDArray[np.floating], trainval: NDArray[np.floating], testkey: float
) -> float:
    """
    Performs 1D interpolation (with extrapolation if needed) on training data
    to estimate the value at a given test key.

    Args:
        trainkey (NDArray[np.floating]):
            Array of training keys used for interpolation.
        trainval (NDArray[np.floating]):
            Array of training values corresponding to `trainkey`.
        testkey (float):
            The key at which the interpolated value is evaluated.

    Returns:
        float: The interpolated (or extrapolated) value.
    """
    newkey, newval = check_duplicate_training(list(trainkey), list(trainval))

    if testkey < min(trainkey) or testkey > max(trainkey):
        interp = interp1d(newkey, newval, fill_value="extrapolate")  # type: ignore
    else:
        interp = interp1d(newkey, newval)
    return interp(testkey)


def interpolate_quantities(
    eccentricities: list[float],
    mass_ratios: list[float],
    interpolant_values: NDArray[np.floating],
    e: float,
    q: float,
) -> NDArray[np.floating]:
    """
    Interpolates a set of values defined over (eccentricity, mass ratio) using an RBF interpolator.

    Args:
        eccentricities (list[float]): Training eccentricity samples.
        mass_ratios (list[float]): Training mass-ratio samples.
        interpolant_values (NDArray[np.floating]): Values to be interpolated.
        e (float): Test eccentricity point.
        q (float): Test mass-ratio point.

    Returns:
        NDArray[np.floating]: Interpolated value(s), squeezed to 1D if possible.
    """
    coordinates = np.array([eccentricities, mass_ratios]).T
    interpolator = RBFInterpolator(coordinates, interpolant_values, kernel="linear")
    return interpolator(np.column_stack([np.atleast_1d(e), np.atleast_1d(q)])).squeeze()


def check_duplicate_training(
    trainkey: list[float], trainval: list[float]
) -> tuple[list[float], list[float]]:
    """
    Collapses duplicate training keys by replacing their multiple values with the median.

    Args:
        trainkey (list[float]): Training keys, possibly containing duplicates.
        trainval (list[float]): Training values corresponding to `trainkey`.

    Returns:
        tuple[list[float], list[float]]:
            A pair `(newkey, newval)` where duplicates in `trainkey` are removed
            and their associated values in `trainval` are replaced by the median.
    """
    d = {}
    newkey = []
    newval = []

    for a, b in zip(trainkey, trainval):
        d.setdefault(a, []).append(b)

    for key in d:
        newkey.append(key)
        newval.append(np.median(d[key]))
    return newkey, newval


def get_noncirc_params(
    somedict: dict,
) -> tuple[
    list[float],
    list[float],
    list[float],
    list[NDArray[np.floating]],
    list[NDArray[np.floating]],
]:
    """
    Extract non-circular (eccentricity-related) parameters from the input dictionary.

    Args:
        somedict (dict):
            Dictionary containing all non-circular parameters. It must include the keys:
                - "q": mass ratio values
                - "e_ref": reference eccentricities
                - "x": PN expansion parameter
                - "A_omega", "B_omega", "freq_omega", "phi_omega": phase-related coefficients
                - "A_amp", "B_amp", "freq_amp", "phi_amp": amplitude-related coefficients

    Returns:
        tuple[
            list[float],
            list[float],
            list[float],
            list[NDArray[np.floating]],
            list[NDArray[np.floating]],
        ]:
            A tuple containing:
                - ecc_q: mass ratios
                - ecc_e: eccentricities
                - ecc_x: PN parameters
                - par_omega: list of omega-related parameter arrays
                - par_amp: list of amplitude-related parameter arrays
    """
    ecc_q = somedict["q"]
    ecc_e = somedict["e_ref"]
    ecc_x = somedict["x"]
    ecc_A_omega = somedict["A_omega"]
    ecc_B_omega = somedict["B_omega"]
    ecc_freq_omega = somedict["freq_omega"]
    ecc_phi_omega = somedict["phi_omega"]
    ecc_A_amp = somedict["A_amp"]
    ecc_B_amp = somedict["B_amp"]
    ecc_freq_amp = somedict["freq_amp"]
    ecc_phi_amp = somedict["phi_amp"]

    par_omega = [ecc_A_omega, ecc_B_omega, ecc_freq_omega, ecc_phi_omega]
    par_amp = [ecc_A_amp, ecc_B_amp, ecc_freq_amp, ecc_phi_amp]
    return ecc_q, ecc_e, ecc_x, par_omega, par_amp


def read_pkl(file_dir: str) -> dict[str, NDArray[np.floating] | list[float]]:
    """
    Read and return data from a pickle file.

    Args:
        file_dir (str):
            Path to the pickle file to load.

    Returns:
        dict[str, NDArray[np.floating] | list[float]]:
            The unpickled data stored in the file, typically containing arrays
            or lists used for training or interpolation.
    """
    with open(file_dir, "rb") as f:
        data = pickle.load(f)
    return data
