import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import make_interp_spline


def interp_omega(
    time_circular: np.ndarray, time_eccentric: np.ndarray, omega_circular: np.ndarray
) -> np.ndarray:
    """Interpolates omega and casts to a new timegrid using a spline.

    Args:
        time_circular (np.ndarray): The old time grid.
        time_eccentric (np.ndarray): The new time grid.
        omega_circular (np.ndarray): The values of omega with which to create the spline.

    Returns:
        np.ndarray: The omega cast to a new time grid.
    """
    interpol = make_interp_spline(time_circular, omega_circular)
    omega_interp = interpol(time_eccentric)
    return omega_interp


def f_sin(
    xdata: np.ndarray, amplitude: float, B: float, freq: np.ndarray, phase: float
) -> np.ndarray:
    """The fitting function of the eccentricity caused modulations to the amplitude or omega.

    Args:
        xdata (np.ndarray): The circularized amplitude or omega to which the power law has been applied.
        amplitude (float): The amplitude parameter A of the fitting function.
        B (float): The factor in the exponent of the fitting function.
        freq (np.ndarray): The frequency of the waveform.
        phase (float): The free fitting parameter phi.

    Returns:
        np.ndarray: Eccentricity caused modulations to the amplitude or omega.
    """
    sin_func = (
        amplitude * np.exp(B * xdata) * np.sin(xdata * freq / (2 * np.pi) + phase)
    )
    return sin_func


def fit_sin(xdata: np.ndarray, ydata: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Performs the fit of eccentricity caused modulations to the amplitude or omega.

    Args:
        xdata (np.ndarray): The circularized amplitude or omega to which the power law has been applied.
        ydata (np.ndarray): The eccentricity caused amplitude or omega.

    Returns:
        tuple[np.ndarray, np.ndarray]: Optimal values for the parameter and its evaluation.
    """
    popt, _ = curve_fit(f_sin, xdata, ydata)
    fit_result = f_sin(xdata, *popt)
    return popt, fit_result


def calculate_x(
    old_time: np.ndarray, omega: np.ndarray, new_time: np.ndarray
) -> np.ndarray:
    """Calculates the geometric frequency x. Will create a spline on which new values for x are computed.

    Args:
        old_time (np.ndarray): The original time grid.
        omega (np.ndarray): The values for omega on the original time grid.
        new_time (np.ndarray): The new time grid.

    Returns:
        np.ndarray: The geometric frequency on the `new_time` timegrid.
    """
    interp_omega = make_interp_spline(old_time, omega)
    x = interp_omega(new_time[0]) ** (2 / 3)
    return x


def get_noncirc_params(
    somedict: dict,
) -> tuple[list[float], list[float], list[float], list[list[any]], list[list[any]]]:
    """Extracts and organizes non-circular waveform fitting parameters from a results dictionary.

    Args:
        somedict (dict): Dictionary containing fitted parameter results. Must include the following keys:
            - "q" (float): Mass ratio.
            - "e_ref" (float): Reference eccentricity.
            - "x" (float): Dimensionless post-Newtonian parameter or time reference.
            - "A_omega", "B_omega", "freq_omega", "phi_omega" (float or list[float]):
              Frequency fit parameters.
            - "A_amp", "B_amp", "freq_amp", "phi_amp" (float or list[float]):
              Amplitude fit parameters.

    Returns:
        tuple[list[float], list[float], list[float], list[list[any]], list[list[any]]]: A tuple containing:
            - ecc_q (list[float]): Mass ratio values.
            - ecc_e (list[float]): Reference eccentricity values.
            - ecc_x (list[float]): PN or time parameter values.
            - par_omega (list[list[any]]): Lists of frequency-related fit parameters
              [A_omega, B_omega, freq_omega, phi_omega].
            - par_amp (list[list[any]]): Lists of amplitude-related fit parameters
              [A_amp, B_amp, freq_amp, phi_amp].
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
