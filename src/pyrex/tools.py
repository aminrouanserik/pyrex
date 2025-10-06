import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.interpolate import make_interp_spline
from qcextender import units


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
    # TODO: fix the function description.
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
    popt, pcov = curve_fit(f_sin, xdata, ydata)
    fit_result = f_sin(xdata, *popt)
    return popt, fit_result


def fitting_eccentric_function(pwr, e_amp_phase, interpol_circ):
    x = (interpol_circ) ** pwr - (interpol_circ[0]) ** pwr
    y = e_amp_phase
    par, fsn = fit_sin(x, y)
    return par, fsn


def find_x(old_time, omega, new_time):
    """
    Compute x at the beginning of new time array.
    """
    interp_omega = make_interp_spline(old_time, omega)
    x = interp_omega(new_time[0]) ** (2.0 / 3)
    return x


def get_noncirc_params(somedict):
    ecc_q = []
    ecc_e = []
    ecc_x = []
    ecc_A_omega = []
    ecc_B_omega = []
    ecc_freq_omega = []
    ecc_phi_omega = []
    ecc_A_amp = []
    ecc_B_amp = []
    ecc_freq_amp = []
    ecc_phi_amp = []

    for i in range(len(somedict["names"])):
        if somedict["e_ref"][i] > 3e-3:
            ecc_q.append(somedict["q"][i])
            ecc_e.append(somedict["e_ref"][i])
            ecc_x.append(somedict["x"][i])
            ecc_A_omega.append(somedict["A_omega"][i])
            ecc_B_omega.append(somedict["B_omega"][i])
            ecc_freq_omega.append(somedict["freq_omega"][i])
            ecc_phi_omega.append(somedict["phi_omega"][i])
            ecc_A_amp.append(somedict["A_amp"][i])
            ecc_B_amp.append(somedict["B_amp"][i])
            ecc_freq_amp.append(somedict["freq_amp"][i])
            ecc_phi_amp.append(somedict["phi_amp"][i])

    par_omega = [ecc_A_omega, ecc_B_omega, ecc_freq_omega, ecc_phi_omega]
    par_amp = [ecc_A_amp, ecc_B_amp, ecc_freq_amp, ecc_phi_amp]
    return ecc_q, ecc_e, ecc_x, par_omega, par_amp


def near_merger(wave):

    time = wave.time
    interp_amp = make_interp_spline(time, wave.amp())
    interp_phase = make_interp_spline(time, wave.phase())
    end_time = time[-1]
    delta_t = wave.metadata.delta_t
    near_merger_time = np.arange(
        units.tM_to_tSI(-29.0, wave.metadata.total_mass) + delta_t,
        end_time + delta_t,
        delta_t,
    )
    new_amp = interp_amp(near_merger_time)
    new_phase = interp_phase(near_merger_time)
    return near_merger_time, new_amp, new_phase


def smooth_joint(time, y, total_mass):
    """
    Smooth the joint curve of the twist and the late merger.
    Parameters
    ----------
    x        : []
               Full time array of the curve.
    y        : []
               Full amplitude array of the curve.

    Returns
    ------
    y_inter   : []
                Smoothed amplitude array.
    """

    tarray = np.where(
        np.logical_and(
            time < units.tM_to_tSI(-25, total_mass),
            time >= units.tM_to_tSI(-46, total_mass),
        )
    )
    # tarray=where(np.logical_and(x<-31*total_mass*lal.MTSUN_SI,x>=-80*total_mass*lal.MTSUN_SI))
    first = tarray[0][0]
    last = tarray[0][-1]
    y[first:last] = np.interp(
        time[first:last], [time[first], time[last]], [y[first], y[last]]
    )
    # y[first:last] = savgol_filter(y[first:last], 9, 3)
    y_inter = savgol_filter(y, 31, 3)
    return y_inter


__all__ = [
    "interp_omega",
    "f_sin",
    "fit_sin",
    "fitting_eccentric_function",
    "find_x",
    "get_noncirc_params",
    "near_merger",
    "smooth_joint",
]
