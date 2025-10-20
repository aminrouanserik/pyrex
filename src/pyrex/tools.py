import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import make_interp_spline


def interp_omega(time_circular, time_eccentric, omega_circular):
    """
    Interpolate omega circular to the time points of the eccentric waveforms.
    Parameters
    ----------
    time_circular   : []
                    1 dimensional array of time sample in the circular data.
    time_eccentric  : []
                    1 dimensional array of time sample in the eccentric data.
    omega_circular  : []
                    1 dimensional array of the original omega circular data.

    Returns
    ------
    omega_interp    : []
                    1 dimensional array of the interpolated omega circular following time sample of the eccentric data.
    """

    interpol = make_interp_spline(time_circular, omega_circular)
    omega_interp = interpol(time_eccentric)
    return omega_interp


def f_sin(xdata, amplitude, B, freq, phase):
    """
    Computes sinusoidal function given the input parameters.

    Parameters
    ----------
    time_sample   : []
                1 dimensional array of time sample.
    freq          : {float}
                Frequency parameter.
    amplitude     : {float}
                Amplitude parameter.
    phase         : {float}
                Phase parameter.
    offset        : {float}
                Offset parameter (for the fitting).

    Returns
    ------
    sin_func      : []
                1 dimensional array of a sinusoidal function.

    """
    sin_func = (
        amplitude * np.exp(B * xdata) * np.sin(xdata * freq / (2 * np.pi) + phase)
    )
    return sin_func


def fit_sin(xdata, ydata):
    """
    Computes the optimize curve fitting for a sinusoidal function with sqrt(time_sample).

    Parameters
    ----------
    time_sample   : []
                1 dimensional array of time sample.
    data          : []
                1 dimensional array of data to be fitted to a sinusoidal function.

    Returns
    ------
    popt        : []
                1 dimensional array of the fitting parameters (frequency, amplitude, phase, and offset).
    fit_result  : []
                1 dimensional array of the fitted data.
    """
    popt, _ = curve_fit(f_sin, xdata, ydata)
    fit_result = f_sin(xdata, *popt)
    return popt, fit_result


def calculate_x(old_time, omega, new_time):
    """
    Compute x at the beginning of new time array.
    """
    interp_omega = make_interp_spline(old_time, omega)
    x = interp_omega(new_time[0]) ** (2.0 / 3)
    return x


def get_noncirc_params(somedict):

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


__all__ = [
    "interp_omega",
    "f_sin",
    "fit_sin",
    "calculate_x",
    "get_noncirc_params",
]
