import pickle

import numpy as np
from numpy.typing import NDArray
from qcextender import units
from qcextender.waveform import Waveform
from scipy import integrate
from scipy.interpolate import RBFInterpolator, interp1d
from scipy.signal import savgol_filter

from pyrex.functions import f_sin


def main(
    approximant: str,
    mode: list[tuple[int, int]],
    dirfile: str = "/home/amin/Projects/School/Masters/25_26-Thesis/pyrex/data/pyrexdata.pkl",
    cut: bool = True,
    **kwargs,
) -> Waveform:
    eccentricity = kwargs.pop("eccentricity")
    if eccentricity == 0:
        eccentricity = 1e-30
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
    time = units.tSI_to_tM(wave.time, wave.metadata.total_mass)
    omega = units.fSI_to_fM(wave.omega(), wave.metadata.total_mass)
    amp = units.mSI_to_mM(wave.amp(), wave.metadata.total_mass, wave.metadata.distance)

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

    return new_time, amp_rec, phase_rec, mask


def get_fit_params(
    training_dict: dict[str, NDArray[np.floating] | list[float]],
    q: float,
    eccentricity: float,
    x: float,
) -> tuple[list[float], list[float]]:
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
                np.asarray(training_quant[0]),
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


def smooth_joint(time: np.ndarray, y: np.ndarray, total_mass: float) -> np.ndarray:
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


def read_pkl(file_dir: str) -> dict[str, NDArray[np.floating] | list[float]]:
    with open(file_dir, "rb") as f:
        data = pickle.load(f)
    return data


def interp1D(
    trainkey: NDArray[np.floating], trainval: NDArray[np.floating], testkey: float
) -> float:
    newkey, newval = check_duplicate_training(list(trainkey), list(trainval))

    if testkey < min(trainkey) or testkey > max(trainkey):
        interp = interp1d(newkey, newval, fill_value="extrapolate")
    else:
        interp = interp1d(newkey, newval)
    return interp(testkey)


def interpolate_quantities(eccentricities, mass_ratios, interpolant_values, e, q):
    coordinates = np.array([eccentricities, mass_ratios]).T
    interpolator = RBFInterpolator(coordinates, interpolant_values, kernel="linear")
    return interpolator(np.column_stack([np.atleast_1d(e), np.atleast_1d(q)])).squeeze()


def check_duplicate_training(
    trainkey: list[float], trainval: list[float]
) -> tuple[list[float], list[float]]:
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
