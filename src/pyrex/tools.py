import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.optimize import curve_fit


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
    lower_bounds = [-np.inf, 0.0, -np.inf, -np.inf]
    upper_bounds = [np.inf, 1e-2, np.inf, np.inf]

    popt, _ = curve_fit(
        f_sin,
        xdata,
        ydata,
        p0=[0.1, 5e-4, 0.1, -2000],
        bounds=(lower_bounds, upper_bounds),
    )
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
