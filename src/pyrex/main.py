import numpy as np
from pyrex.tools import fit_sin, calculate_x
from pyrex.basics import write_pkl
from scipy.signal import savgol_filter
from scipy.interpolate import make_interp_spline
from gw_eccentricity import measure_eccentricity
from qcextender.dimensionlesswaveform import DimensionlessWaveform


def glassware(
    q: list[float],
    names: list[str],
    outfname: str,
    e_ref: list[float] = None,
) -> None:
    """Fits eccentric contributions to the amplitude and instantaneous frequency. Please make sure that every
    mass ratio has a complimentary zero eccentricity simulation.

    Args:
        q (list[float]): List of mass ratios of the binary simulations.
        names (list[str]): List of SXS simulation names of the binary simulations.
        outfname (str): The filename in which to save the fit parameters.
        e_ref (list[float]): List of eccentricities at the reference frequency of the binary simulations.
        Defaults to None, in which case it is calculted by `gw_eccentricity`.

    Output:
        File called outfname.pkl.
    """
    if not e_ref:
        e_ref = []
        fref_in = 0.0075
        for name in names:
            sim = DimensionlessWaveform.from_sim(name)
            try:
                return_dict = measure_eccentricity(
                    fref_in=fref_in,
                    method="ResidualAmplitude",
                    dataDict={
                        "t": sim.time,
                        "hlm": {(2, 2): sim[2, 2]},
                    },
                )
                eccentricity = return_dict["eccentricity"]
            except:
                eccentricity = 0
            # mean_anomaly = return_dict["mean_anomaly"]
            e_ref.append(eccentricity)

    waves = components(names)
    circ_waves = get_circ_waves(waves, e_ref)

    circ_lookup = {}

    for c in circ_waves:
        time = c.time
        mask = (time > (time[0] + 250)) & (time <= -29)

        t = time[mask]
        omega = c.omega()[mask]
        amp = c.amp()[mask]

        q_key = round(c.metadata.q, 0)
        circ_lookup[q_key] = (
            make_interp_spline(t, omega),
            make_interp_spline(t, amp),
        )

    # Need a common time grid for all waves for this code to work. Boundaries and length can be tinkered with
    begin_tm = -1500.0
    end_tm = -29
    len_tm = 15221
    new_time = np.linspace(begin_tm, end_tm, len_tm)

    e_amp, e_omega = compute_e_estimator(waves, e_ref, circ_lookup, new_time)

    results = fit_model(waves, circ_lookup, new_time, e_omega, e_amp)
    x = compute_xquant(waves, new_time)

    # write and store the data
    results.update({"q": q, "e_ref": e_ref, "x": x})
    if outfname:
        write_pkl(outfname, results)


def components(names: list[str]) -> list[DimensionlessWaveform]:
    """Returns a list of simulations from the SXS catalog.

    Args:
        names (list[str]): The names of the waveforms in the SXS catalog.

    Returns:
        list[DimensionlessWaveform]: List of waveforms from the SXS catalog wrapped in the DimensionlessWaveform object.
    """
    sims = []
    for name in names:
        sims.append(DimensionlessWaveform.from_sim(name))
    return sims


def compute_e_estimator(
    waves: list[DimensionlessWaveform],
    eccentricities: list[float],
    circ_lookup: dict[str, tuple[callable, callable]],
    new_time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimates the contribution of eccentricity on the amplitude and instantaneous frequency.

    Args:
        waves (list[DimensionlessWaveform]): List of waveforms from the SXS catalog wrapped in the DimensionlessWaveform object.
        eccentricities (list[float]): List of eccentricities at the reference frequency, corresponding to the SXS simulations in waves.
        circ_lookup (dict[str, tuple[callable, callable]]): Stores splines of the circular simulations' amplitude and instantaneous frequency, stored by mass ratio.
        new_time (np.ndarray): New time grid to cast the waveform to.

    Returns:
        tuple[np.ndarray, np.ndarray]: The eccentricity-caused contributions to the amplitude and instantaneous frequency respectively.
    """
    circ_lookup_omega = {q: pair[0] for q, pair in circ_lookup.items()}
    circ_lookup_amp = {q: pair[1] for q, pair in circ_lookup.items()}

    e_amp, e_omega = [], []
    for w, e in zip(waves, eccentricities):
        e_amp.append(get_e_amp(w, e, circ_lookup_amp, new_time))
        e_omega.append(get_e_omega(w, e, circ_lookup_omega, new_time))

    return e_amp, e_omega


def fit_model(
    waves: list[DimensionlessWaveform],
    circ_lookup: dict[str, tuple[callable, callable]],
    new_time: np.ndarray,
    e_omega: list,
    e_amp: list,
    omega_power: float = -59 / 24,
    amp_power: float = -83 / 24,
) -> dict[str, np.ndarray]:
    """Performs a fit of the eccentricity caused contributions to the instantaneous frequency and amplitude to a sinusoid.

    Args:
        waves (list[DimensionlessWaveform]): List of waveforms from the SXS catalog wrapped in the DimensionlessWaveform object.
        circ_lookup (dict[str, tuple[callable, callable]]): Stores splines of the circular simulations' amplitude and instantaneous frequency, stored by mass ratio.
        new_time (np.ndarray): New time grid to cast the waveform to.
        e_omega (list): The eccentricity-caused contribution to the instantaneous frequency.
        e_amp (list): The eccentricity-caused contribution to the amplitude.
        omega_power (float, optional): Power applied to the instantaneous frequency to approximate eccentric contributions. Defaults to -59/24.
        amp_power (float, optional): Power applied to the amplitude to approximate eccentric contributions. Defaults to -83/24.

    Raises:
        ValueError: Raised if an eccentric waveform was called without a non-eccentric counterpart (only the mass ratio is checked).

    Returns:
        dict[str, np.ndarray]: A dictionary containing the fit results.
    """
    omega_params, amp_params = [], []
    fit_omega, fit_amp = [], []

    for wave, omega, amp in zip(waves, e_omega, e_amp):

        # This needs to be fixed throughout, just take closest q instead or take from model instead.
        q = round(wave.metadata.q, 0)

        if q not in circ_lookup:
            raise ValueError(f"No circular reference for q={q}")

        interp_omega_c, interp_amp_c = circ_lookup[q]

        circ_omega_vals = interp_omega_c(new_time)
        circ_amp_vals = interp_amp_c(new_time)

        omega_paramsr, fit_omegar = fitting_eccentric_function(
            omega_power, omega, circ_omega_vals
        )
        amp_paramsr, fit_ampr = fitting_eccentric_function(
            amp_power, amp, circ_amp_vals
        )

        omega_params.append(omega_paramsr)
        amp_params.append(amp_paramsr)
        fit_omega.append(fit_omegar)
        fit_amp.append(fit_ampr)

    omega_params, amp_params = np.array(omega_params), np.array(amp_params)

    results = {
        "A_omega": omega_params[:, 0],
        "B_omega": omega_params[:, 1],
        "freq_omega": omega_params[:, 2],
        "phi_omega": omega_params[:, 3],
        "fit_omega": fit_omega,
        "A_amp": amp_params[:, 0],
        "B_amp": amp_params[:, 1],
        "freq_amp": amp_params[:, 2],
        "phi_amp": amp_params[:, 3],
        "fit_amp": fit_amp,
    }

    return results


def get_circ_waves(
    waves: list[DimensionlessWaveform], eccentricities: list[float]
) -> list[DimensionlessWaveform]:
    """Returns the circular waveforms from a list of waveforms.

    Args:
        waves (list[DimensionlessWaveform]): List of waveforms from the SXS catalog wrapped in the DimensionlessWaveform object.
        eccentricities (list[float]): List of eccentricities at the reference frequency, corresponding to the SXS simulations in waves.

    Returns:
        list[DimensionlessWaveform]: List of circular SXS waveforms wrapped in DimensionlessWaveform objects.
    """
    circ_waves = []

    # Every value of q should have an example where eccentricity is 0
    for wave, eccentricity in zip(waves, eccentricities):
        if eccentricity == 0:
            circ_waves.append(wave)
    return circ_waves


def get_e_X(
    wave: DimensionlessWaveform,
    eccentricity: float,
    circ_lookup: dict[str, tuple[callable, callable]],
    new_time: np.ndarray,
    get_component: callable,
    filter_comp: int = 2,
) -> np.ndarray:
    """Computes the eccentricity caused contribution to the amplitude or instantaneous frequency for `wave`.

    Args:
        wave (DimensionlessWaveform): Single SXS simulation wrapped in a DimensionlessWaveform object.
        eccentricity (float): Corresponding eccentricity at the reference frequency.
        circ_lookup (dict[str, tuple[callable, callable]]): Lookup dictionary containing splines for the amplitude and instantaneous frequency, sorted by mass ratio.
        new_time (np.ndarray): New time grid to cast the waveform to.
        get_component (callable): Function for amplitude or instantaneous frequency.
        filter_comp (int, optional): The order of the derivative the Savitzky-Golay filter computes. Defaults to 2.

    Raises:
        ValueError: Raised if an eccentric waveform was called without a non-eccentric counterpart (only the mass ratio is checked).

    Returns:
        np.ndarray: The eccentricity caused contribution to the amplitude or instantaneous frequency for `wave`.
    """
    q = np.round(wave.metadata.q, 2)
    if q not in circ_lookup:
        raise ValueError(f"No circular reference for q={q}")

    mask = (wave.time > (wave.time[0] + 250)) & (wave.time <= -29)
    circ_interp = circ_lookup[q]
    ecc_interp = make_interp_spline(wave.time[mask], get_component(wave)[mask])

    circ_vals = circ_interp(new_time)
    ecc_vals = ecc_interp(new_time)

    e_X = (ecc_vals - circ_vals) / (2.0 * circ_vals)

    if eccentricity > 0:
        e_X = savgol_filter(e_X, 501, filter_comp)

    return e_X


def get_e_amp(
    wave: DimensionlessWaveform,
    eccentricity: float,
    circ_lookup: dict[str, tuple[callable, callable]],
    new_time: np.ndarray,
) -> np.ndarray:
    """Computes the eccentricity caused contribution to the amplitude for `wave`.

    Args:
        wave (DimensionlessWaveform): Single SXS simulation wrapped in a DimensionlessWaveform object.
        eccentricity (float): Corresponding eccentricity at the reference frequency.
        circ_lookup (dict[str, tuple[callable, callable]]): Lookup dictionary containing splines for the amplitude and instantaneous frequency, sorted by mass ratio.
        new_time (np.ndarray): New time grid to cast the waveform to.

    Returns:
        np.ndarray: The eccentricity caused contribution to the amplitude for `wave`.
    """
    return get_e_X(
        wave, eccentricity, circ_lookup, new_time, lambda w: w.amp(), filter_comp=3
    )


def get_e_omega(
    wave: DimensionlessWaveform,
    eccentricity: float,
    circ_lookup: dict[str, tuple[callable, callable]],
    new_time: np.ndarray,
) -> np.ndarray:
    """Computes the eccentricity caused contribution to the instantaneous frequency for `wave`.

    Args:
        wave (DimensionlessWaveform): Single SXS simulation wrapped in a DimensionlessWaveform object.
        eccentricity (float): Corresponding eccentricity at the reference frequency.
        circ_lookup (dict[str, tuple[callable, callable]]): Lookup dictionary containing splines for the amplitude and instantaneous frequency, sorted by mass ratio.
        new_time (np.ndarray): New time grid to cast the waveform to.

    Returns:
        np.ndarray: The eccentricity caused contribution to the instantaneous frequency for `wave`.
    """
    return get_e_X(wave, eccentricity, circ_lookup, new_time, lambda w: w.omega())


def fitting_eccentric_function(
    pwr: float, e_amp_phase: np.ndarray, interpol_circ: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Calculates the fit.

    Args:
        pwr (float): Power to be applied to the amplitude or instantaneous frequency.
        e_amp_phase (np.ndarray): Eccentricity caused amplitude or instantaneous frequency.
        interpol_circ (np.ndarray): Circular waveform amplitude or instantaneous frequency.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of the fitted parameters and values.
    """
    x = (interpol_circ) ** pwr - (interpol_circ[0]) ** pwr
    y = e_amp_phase
    par, fsn = fit_sin(x, y)
    return par, fsn


def compute_xquant(
    waves: list[DimensionlessWaveform], new_time: np.ndarray
) -> list[np.ndarray]:
    """Computes the dimensionless frequency.

    Args:
        waves (list[DimensionlessWaveform]): List of waveforms from the SXS catalog wrapped in the DimensionlessWaveform object.
        new_time (np.ndarray): New time grid to cast the waveform to.

    Returns:
        list[np.ndarray]: Dimensionless frequency for every waveform in `waves`
    """
    x = [calculate_x(wave.time, wave.omega(), new_time) for wave in waves]
    return x
