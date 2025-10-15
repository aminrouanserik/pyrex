import numpy as np
from pyrex.tools import get_noncirc_params, f_sin
from pyrex.basics import interp1D, checkIfFilesExist, read_pkl
from scipy import integrate
from scipy.signal import savgol_filter
from qcextender.waveform import Waveform
from qcextender import units


def main(approximant, mode, **kwargs):
    eccentricity = kwargs.pop("eccentricity")

    wave = Waveform.from_model(approximant, mode, **kwargs)

    x = wave.omega()[0] ** (2 / 3)

    # check requirements
    training = checkIfFilesExist()

    training_dict = read_pkl(training)
    omega_keys, amp_keys = get_key_quant(
        training_dict, wave.metadata.q, eccentricity, x
    )

    # if eccentricity > 3e-2:
    kwargs = {"omega_keys": omega_keys, "amp_keys": amp_keys}
    newwave = wave.add_eccentricity(
        construct,
        eccentricity,
        [(2, 2)],
        omega_keys=omega_keys,
        amp_keys=amp_keys,
    )

    return newwave


def construct(wave, mode, omega_keys, amp_keys):
    timenew, amp_rec, phase_rec, mask = eccentric_from_circular(
        omega_keys, amp_keys, wave
    )

    late_time, late_amp, late_phase = near_merger(wave, mask)

    phase_rec += late_phase[0] - phase_rec[-1]

    amp_construct = np.concatenate((amp_rec, late_amp[1:]))
    phase_construct = np.concatenate((phase_rec, late_phase[1:]))
    time_construct = np.concatenate((timenew, late_time[1:]))

    return time_construct, phase_construct, amp_construct


def near_merger(wave, mask):
    time = wave.time
    near_merger_time = time[mask[0][-1] :]
    new_amp = wave.amp()[mask[0][-1] :]
    new_phase = wave.phase()[mask[0][-1] :]
    return near_merger_time, new_amp, new_phase


def eccentric_from_circular(
    par_omega,
    par_amp,
    wave,
    phase_pwr=-59.0 / 24,
    amp_pwr=-83.0 / 24,
):
    time = units.tSI_to_tM(wave.time, wave.metadata.total_mass)
    omega = units.fSI_to_fM(wave.omega(), wave.metadata.total_mass)
    amp = units.mSI_to_mM(wave.amp(), wave.metadata.total_mass, wave.metadata.distance)
    phases = wave.phase()
    # max_phase = phases[np.argmax(amp)]

    # Lower bound shouldn't be hardcoded
    mask = np.where((time < -29))  # (time > -5000) &
    new_time = time[mask]

    if max(abs(omega)) == 0:
        amp_rec = np.zeros(len(new_time))
        phase_rec = np.zeros(len(new_time))
    else:
        omega_circ = omega[mask]
        amp_circ = amp[mask]

        # Shift appears crucial, but why
        shift_omega = omega[
            (np.abs(new_time + 1500)).argmin()
        ]  # omega[np.argmax(np.abs(amp))]
        shift_amp = amp[
            (np.abs(new_time + 1500)).argmin()
        ]  # amp[np.argmax(np.abs(amp))]

        # x_omega = omega_circ**phase_pwr - shift_omega**phase_pwr
        # x_amp = amp_circ**amp_pwr - shift_amp**amp_pwr

        # fit_ex_omega = f_sin(
        #     x_omega, par_omega[0], par_omega[1], par_omega[2], par_omega[3]
        # )
        # fit_ex_amp = f_sin(x_amp, par_amp[0], par_amp[1], par_amp[2], par_amp[3])

        # Look about the same as in the paper, signs are negative for omega (crucial)
        # omega_rec = fit_ex_omega * 2 * (-omega_circ) - omega_circ
        # amp_rec = fit_ex_amp * 2 * amp_circ + amp_circ

        new_time = units.tM_to_tSI(new_time, wave.metadata.total_mass)

        # Minus sign crucial for circ, checking if also when fitting again
        phase_rec = integrate.cumulative_trapezoid(
            units.fM_to_fSI(-omega_circ, wave.metadata.total_mass), new_time, initial=0
        )

        amp_rec = units.mM_to_mSI(
            amp_circ, wave.metadata.total_mass, wave.metadata.distance
        )

    return new_time, amp_rec, phase_rec, mask


def get_key_quant(training_dict, q, eccentricity, x):

    eq, ee, ex, eomg, eamp = get_noncirc_params(training_dict)
    training_quant = [eq, ee, ex]

    test_quant = [q, eccentricity, x]

    A_omega, B_omega, freq_omega, phi_omega = interpol_key_quant(
        training_quant, eomg, test_quant
    )
    A_amp, B_amp, freq_amp, phi_amp = interpol_key_quant(
        training_quant, eamp, test_quant
    )

    omega_keys = [A_omega, B_omega, freq_omega, phi_omega]
    amp_keys = [A_amp, B_amp, freq_amp, phi_amp]

    return omega_keys, amp_keys


def interpol_key_quant(training_quant, training_keys, test_quant):
    forA = float(interp1D(training_quant[1], training_keys[0], test_quant[1]))
    A = float(
        interp1D(training_quant[1], np.abs(np.asarray(training_keys[0])), test_quant[1])
    )
    B = np.log(
        (
            interp1D(
                training_quant[1],
                training_keys[0] * np.exp(training_keys[1]),
                test_quant[1],
            )
        )
        / np.asarray(forA)
    )
    freq = np.sqrt(
        1.0
        / (
            interp1D(
                np.asarray(training_quant[0]),
                1.0 / np.asarray(training_keys[2]) ** 2,
                test_quant[0],
            )
        )
    )
    phi = float(
        interp1D(
            np.asarray(training_quant[2]),
            np.asarray(training_keys[3]),
            np.asarray(test_quant[2]),
        )
    )

    return A, B, freq, phi


# def smooth_joint(time, y, total_mass):
#     """
#     Smooth the joint curve of the twist and the late merger.
#     Parameters
#     ----------
#     x        : []
#                Full time array of the curve.
#     y        : []
#                Full amplitude array of the curve.

#     Returns
#     ------
#     y_inter   : []
#                 Smoothed amplitude array.
#     """

#     tarray = np.where(
#         (time < units.tM_to_tSI(-25, total_mass))
#         & (time >= units.tM_to_tSI(-46, total_mass))
#     )

#     first = tarray[0][0]
#     last = tarray[0][-1]
#     y[first:last] = np.interp(
#         time[first:last], [time[first], time[last]], [y[first], y[last]]
#     )
#     # # y[first:last] = savgol_filter(y[first:last], 9, 3)
#     y_inter = savgol_filter(y, 31, 3)
#     return y_inter
