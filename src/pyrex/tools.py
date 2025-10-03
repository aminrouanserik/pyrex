# Copyright (C) 2020 Yoshinta Setyawati <yoshintaes@gmail.com>
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#

"""
Computes waveform's amplitude, phase, omega, time sample, and strain components and align them.
"""

__author__ = "Yoshinta Setyawati"

import numpy as np
from numpy import *
import lalsimulation as ls
import lal
import h5py
import sxs
from pyrex.decor import *
from pyrex.basics import *
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema, find_peaks, savgol_filter
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import warnings
from qcextender import units
from qcextender.utils import spherical_harmonics

warnings.filterwarnings("ignore")


def get_components(name):
    sim = sxs.load(name, ignore_deprecation=True, extrapolation="Outer").h

    times = sim.t
    h22 = sim[:, sim.index(2, 2)]
    amp22 = sim[:, sim.index(2, 2)].abs
    phase22 = sim[:, sim.index(2, 2)].arg_unwrapped

    return times, amp22, phase22, h22


def t_align(names, dt=0.4, t_junk=250.0, t_circ=-29.0):
    """
    Align waveform such that the peak amplitude is at t=0 and chopped -29M before merger (max t).
    Modify the delta t of every waveform with the same number.

    Parameters
    ----------
    names       : {str}
                       A set of waveform simulation name such as 'SXS_BBH_1081'.
    t_peak     : {float}
               Merger time before alignment (in positive t/M).
    dt          : {float}
               delta t of the new time samples. Default 0.4.
    t_chopped   : {float}
               t final before binary circularizes. Default -29M.

    Returns
    ------
    time_window : []
                      Array of the new sample time.
    amp_window	: []
                      Array of the new amplitude.
        phase_window: []
                      Array of the aligned phase.
        h22_window  : []
                      Array of the aligned strain.
    """

    amp_window = []
    phase_window = []
    h22_window = []
    new_time = []
    time_window = []

    for name in names:
        temp_time, temp_amp, temp_phase, temp_h22 = get_components(name)
        timeshift = temp_time - temp_time[argmax(temp_amp)]
        shifted_time = arange(timeshift[0], timeshift[::-1][0], dt)

        amp_inter = spline(timeshift, temp_amp)
        phase_inter = spline(timeshift, temp_phase)
        h22r_inter = spline(timeshift, temp_h22.real)
        h22i_inter = spline(timeshift, temp_h22.imag)

        amp = amp_inter(shifted_time)
        phase = phase_inter(shifted_time)
        h22r = h22r_inter(shifted_time)
        h22i = h22i_inter(shifted_time)

        array_early_inspiral = int(t_junk / dt)  # remove the junk radiation
        array_late_inspiral = int(
            argmax(amp) + t_circ / dt
        )  # due to circularization for low q & low e binaries, remove some t before merger.

        time_window.append(shifted_time[array_early_inspiral:array_late_inspiral])
        amp_window.append(amp[array_early_inspiral:array_late_inspiral])
        h22_window.append(
            h22r[array_early_inspiral:array_late_inspiral]
            + 1j * h22i[array_early_inspiral:array_late_inspiral]
        )
        phase_window.append(-unwrap(angle(h22_window[-1])))

    return (
        asarray(time_window, dtype=object),
        asarray(amp_window, dtype=object),
        asarray(phase_window, dtype=object),
        asarray(h22_window, dtype=object),
    )


def compute_omega(time_sample, hlm):
    """
    Computes omega from time sample and hlm.
    Omega=d/dt (arg hlm) [Husa 2008]

    Parameters
    ----------
    time_sample : []
                      1 dimensional array of time sample of the strain data.
    hlm         : []
              1 dimensional array of strain with (l=2, m=2) or (l=2, m=-2) mode.

    Returns
    ------
    omega     : []
                      1 dimensional array of omega.

    """
    omega = gradient(-unwrap(angle(hlm)), time_sample)
    return omega


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

    interpol = spline(time_circular, omega_circular)
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
    sin_func = amplitude * exp(B * xdata) * sin(xdata * freq / (2 * pi) + phase)
    # sin_func=amplitude*sin(time_sample * freq + phase)+ offset
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


def find_locals(data, local_min=True, sfilter=True):
    """
    Find local minima/maxima of a given data.

    Parameters
    ----------
    data        : []
                1 dimensional array of data.
    local_min   : bool
                If True, find local minima, otherwise local maxima. Default=True.
    sfilter      : bool
                If True, filter to remove noise will be applied (smooth curve) with savgol filter. Default=True.
    Returns
    ------
    local_array  : []
                1 dimensional array of local minima/maxima from a given function.

    """
    if sfilter:
        new_data = savgol_filter(data, 501, 2)
    else:
        new_data = data

    if local_min:
        local = argrelextrema(new_data, less)
    else:
        local = argrelextrema(new_data, greater)
    local_array = asarray(local).reshape(len(local[0]))
    return local_array


def find_roots(x, y):
    """
    Find the values of the x data from a given y data.

    Parameters
    ----------
    x        : []
            1 dimensional array of the x data.
    y        : []
            1 dimensional array of the y data.

    Returns
    ------
    roots_data: []
            1 dimensional array of x values from a given y data.
    """

    s = abs(diff(sign(y))).astype(bool)
    roots_data = x[:-1][s] + diff(x)[s] / (abs(y[1:][s] / y[:-1][s]) + 1)
    return roots_data


def find_intercept(x, y, y_to_find):
    """
    Find the values of the x data from a given positive/negative values of the y data.

    Parameters
    ----------
    x        : []
            1 dimensional array of the x data.
    y        : []
            1 dimensional array of the y data.
    y_to_find: {float}
            The y value that intercepts y (always positive value).

    Returns
    ------
    roots_pos: []
            1 dimensional array of the x values from a given +y_to_find data.
    roots_neg: []
            1 dimensional array of the x values from a given -y_to_find data.

    """
    if y_to_find < 0:
        error("y_to_find is always positive.")
    else:
        roots_pos = find_roots(x, y - y_to_find)
        roots_neg = find_roots(x, y + y_to_find)
    return roots_pos, roots_neg


def compute_residual(time_sample, component, deg=4):
    """
    Computes the residual of sqrt polynomial fits of a given data.

    Parameters
    ----------
    time_sample : []
                1 dimensional positive array of time sample.
    component   : []
                1 dimensional array of the component to be fitted.
    deg         : {int}
                degree of the polynomial function to fit the data. Default=4.

    Returns
    ------
    res         : []
                1 dimensional array of the residual of the fitted function.
    B_sec       : []
                1 dimensional array of a polynomial function fitted to the data.
    """
    time_sample = abs(time_sample)
    B_t = component
    p = poly1d(polyfit(sqrt(time_sample), B_t, deg=deg))
    B_sec = p(sqrt(time_sample))
    res = B_t - B_sec

    return res, B_sec


def time_window_greater(time, time_point, data):
    """
    Windows data in time series greater than a point in time.
    This function cuts early signal in time.

    Parameters
    ----------
    time            : []
                    1 dimensional array of time samples.
    time_point      : {float}
                    Minimum time in the data after the window.
    data            : []
                    Data to be put in the window.


    Returns
    ------
    new_data       : []
                    Data in the window.

    """
    window = where(time > time_point)
    new_data = data[window]
    return new_data


def noisy_peaks(data, prominence=0.1):
    """
    Finds local maxima in a noisy data.

    Parameters
    ----------
    data        : []
                1 dimensional array to find its peak.
    prominence  : {float}
                The minimum height necessary to descend to get from the summit to any higher terrain.
                Default=0.1.

    Returns
    ------
    peaks       : []
                1 dimensional array that contains array numbers of the local maxima (peaks) in the noisy data.

    """

    peaks, _ = find_peaks(data, prominence=0.1)
    return peaks


def find_x(old_time, omega, new_time):
    """
    Compute x at the beginning of new time array.
    """
    interp_omega = spline(old_time, omega)
    x = interp_omega(new_time[0]) ** (2.0 / 3)
    return x


def get_nr_hlm(Hlm, Ylm, amp_scale):
    if Ylm != 0:
        hlm = Hlm / (amp_scale * Ylm)
    else:
        hlm = zeros(len(Hlm))
    return hlm


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
    interp_amp = spline(time, wave.amp())
    interp_phase = spline(time, wave.phase())
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


def smooth_joint(x, y, total_mass):
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

    tarray = where(
        logical_and(
            x < -25 * total_mass * lal.MTSUN_SI, x >= -46 * total_mass * lal.MTSUN_SI
        )
    )
    # tarray=where(logical_and(x<-31*total_mass*lal.MTSUN_SI,x>=-80*total_mass*lal.MTSUN_SI))
    first = tarray[0][0]
    last = tarray[0][-1]
    y[first:last] = interp(x[first:last], [x[first], x[last]], [y[first], y[last]])
    # y[first:last] = savgol_filter(y[first:last], 9, 3)
    y_inter = savgol_filter(y, 31, 3)
    return y_inter


__all__ = [
    "get_components",
    "t_align",
    "compute_omega",
    "interp_omega",
    "f_sin",
    "fit_sin",
    "fitting_eccentric_function",
    "find_locals",
    "find_roots",
    "find_intercept",
    "compute_residual",
    "time_window_greater",
    "noisy_peaks",
    "find_x",
    "get_nr_hlm",
    "get_noncirc_params",
    "near_merger",
    "smooth_joint",
]
