import numpy as np
from pyrex.tools import get_noncirc_params, f_sin
from pyrex.basics import interp1D, get_filename, read_pkl
from scipy import integrate
from scipy.signal import savgol_filter
from qcextender.waveform import Waveform
from qcextender import units


def main(approximant: str, mode: list[tuple[int, int]], **kwargs) -> Waveform:
    """Generates a qcextender Waveform object and returns with added eccentricity modulations.

    Args:
        approximant (str): Quasi-circular approximant to generate the initial Waveform with.
        mode (list[tuple[int, int]]): Which mode to generate and add eccentricity modulations to.

    Returns:
        Waveform: Waveform object with added eccentricity modulations.
    """
    eccentricity = kwargs.pop("eccentricity")
    wave = Waveform.from_model(approximant, mode, **kwargs)

    kwargs = {
        "q": wave.metadata.q,
        "eccentricity": eccentricity,
    }
    newwave = wave.add_eccentricity(construct, kwargs, eccentricity)
    return newwave


def construct(
    wave: Waveform, mode: tuple[int, int], q: float, eccentricity: float
) -> tuple[np.ndarray]:
    """Constrcuts a new Waveform strain by adding eccentricity to the inspiral and connects it to the original circular merger.

    Args:
        wave (Waveform): Quasi-circular Waveform to add eccentricity to.
        mode (tuple[int, int]): Mode, unused but required for Waveform.add_eccentricity().
        q (float): Mass ratio of the binary, always above 1.
        eccentricity (float): Eccentricity to be added to the quasi-circular waveform.

    Returns:
        tuple[np.ndarray]: The time, phase and amplitude of the new, eccentric Waveform.
    """
    # masked_waveform returns one extra index to allow for phase alignment
    early_time, amp_rec, phase_rec, mask = eccentric_from_circular(
        wave, q, eccentricity
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

    return time_construct, phase_construct, amp_construct


def sliced_waveform(wave: Waveform, index: int) -> tuple[np.ndarray]:
    """Returns an array starting at the specified index with a corresponding phase and amplitude.

    Args:
        wave (Waveform): The Waveform to be sliced.
        index (np.ndarray): The index to slice the waveform at.

    Returns:
        tuple[np.ndarray]: A tuple of the sliced time and corresponding amplitude and phase.
    """
    time = wave.time

    near_merger_time = time[index:]
    new_amp = wave.amp()[index:]
    new_phase = wave.phase()[index:]
    return near_merger_time, new_amp, new_phase


def eccentric_from_circular(
    wave: Waveform,
    q: float,
    eccentricity: float,
    phase_pwr: float = -59.0 / 24,
    amp_pwr: float = -83.0 / 24,
) -> tuple[np.ndarray]:
    """

    Args:
        wave (Waveform): Quasi-circular Waveform to add eccentricity to.
        q (float): Mass ratio of the binary, always above 1.
        eccentricity (float): Eccentricity to be added to the quasi-circular waveform.
        phase_pwr (float, optional): Power in the power law for the phase modulations. Defaults to -59.0/24.
        amp_pwr (float, optional): Power in the power law for the amplitude modulations. Defaults to -83.0/24.

    Returns:
        tuple[np.ndarray]: A tuple of the masked time and corresponding amplitude and phase including eccentricity modulations.
    """
    time = units.tSI_to_tM(wave.time, wave.metadata.total_mass)
    omega = units.fSI_to_fM(wave.omega(), wave.metadata.total_mass)
    amp = units.mSI_to_mM(wave.amp(), wave.metadata.total_mass, wave.metadata.distance)

    # Lower bound shouldn't be hardcoded
    mask = np.where((time > -1500) & (time < -29))
    new_time = time[mask]

    x = omega[mask][0] ** (2 / 3)

    # check requirements
    training = get_filename()
    training_dict = read_pkl(training)
    par_omega, par_amp = get_fit_params(training_dict, q, eccentricity, x)

    if not np.any(omega):
        amp_rec = np.zeros(len(new_time))
        phase_rec = np.zeros(len(new_time))
    else:
        omega_circ = omega[mask]
        amp_circ = amp[mask]

        # Arbitrary shift is crucial
        shift_omega = omega[0]
        shift_amp = amp[0]

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
    training_dict: dict, q: float, eccentricity: float, x: float
) -> tuple[list]:
    """Gets fit parameters for the sinusoidal eccentricity modulation fits.

    Args:
        training_dict (dict): Dictionary with all fit parameters.
        q (float): Mass ratio of the binary, always above 1.
        eccentricity (float): Eccentricity to be added to the quasi-circular waveform.
        x (float): Dimensionless variable x at the start of the waveform.

    Returns:
        tuple[list]: The parameters for the omega and amplitude keys.
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


# Investigation necessary
def interpol_key_quant(
    training_quant: list[list], training_keys: list[list], test_quant: list[list]
) -> tuple[float, float, float, float]:
    """Returns the interpolated fit parameters.

    Args:
        training_quant (list[list]): Lists of mass ratios, eccentricities and starting x during training.
        training_keys (list[list]): List of omega or amplitude of simulations during training.
        test_quant (list[list]): Quantities requested.

    Returns:
        tuple[float, float, float, float]: Interpolated fit parameters, amplitude, power, frequency, and phase.
    """
    forA = float(interp1D(training_quant[1], training_keys[0], test_quant[1]))
    A = float(interp1D(training_quant[1], np.abs(training_keys[0]), test_quant[1]))
    B = np.log(
        (
            interp1D(
                training_quant[1],
                training_keys[0] * np.exp(training_keys[1]),
                test_quant[1],
            )
        )
        / forA
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


def smooth_joint(time: np.ndarray, y: np.ndarray, total_mass: float) -> np.ndarray:
    """Uses the Savitzky-Golay filter to smoothen the transition area from eccentric to quasi-circular.

    Args:
        time (np.ndarray): The time array.
        y (np.ndarray): Quantity to smoothen, either phase or amplitude.
        total_mass (float): The total mass of the inspiral.

    Returns:
        np.ndarray: The filtered phase or amplitude.
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
    # # y[first:last] = savgol_filter(y[first:last], 9, 3)
    y_inter = savgol_filter(y, 31, 3)
    return y_inter
