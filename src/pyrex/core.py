import numpy as np
from pyrex.tools import near_merger, smooth_joint, get_noncirc_params, f_sin
from pyrex.basics import interp1D, checkIfFilesExist, read_pkl
from scipy.interpolate import make_interp_spline
from scipy import integrate
from qcextender.waveform import Waveform
from qcextender import units


class Cookware:
    """
    A class to twist any analytic circular waveforms into eccentric model.
    """

    def __init__(
        self,
        approximant,
        mass1,
        mass2,
        spin1x,
        spin1y,
        spin1z,
        spin2x,
        spin2y,
        spin2z,
        eccentricity,
        x,
        inclination,
        distance,
        coa_phase,
        sample_rate=4096.0,
        varphi=None,
    ):
        """
        Initiates Cookware class for non-spinning, low eccentricity, and mass ratio<=3 binaries.

        Parameters
        ----------
        mass1         : {float}
                      Mass of the hevaiest object (MSun).
        mass2         : {float}
                  Dimensionless spin parameters.
        approximant   : {str}
                  Waveform approximant of analytic waves.

        chi           : {float}
                  Spin of the system.
        distance      : {float}
                  Distance of the two bodies (Mpc).
        inclination   : {float}
                  Inclination angle (rad).
        coa_phase     : {float}
                  Coalescence phase (rad).

        Returns
        ------
        times         : []
                     Time sample array.
        h22	          : []
                     Complex numbers of the eccentric l=2, m=2 mode.
        """
        self.approximant = approximant
        self.mass1 = mass1
        self.mass2 = mass2
        self.spin1x = spin1x
        self.spin1y = spin1y
        self.spin1z = spin1z
        self.spin2x = spin2x
        self.spin2y = spin2y
        self.spin2z = spin2z
        self.eccentricity = eccentricity
        self.inclination = inclination
        self.distance = distance
        self.coa_phase = coa_phase
        self.x = x
        self.varphi = varphi

        # generate analytic waveform
        kwargs = {
            "mass1": mass1,
            "mass2": mass2,
            "inclination": inclination,
            "coa_phase": coa_phase,
            "delta_t": 1.0 / sample_rate,
            "f_lower": 20,
            "f_ref": 25,  # Change to be specific to waveform model used, want to do that in the generation.
            "distance": distance,
        }
        self.wave = Waveform.from_model(approximant, [(2, 2)], **kwargs)

        # check requirements
        self.checkParBoundaris()
        training = checkIfFilesExist()

        training_dict = read_pkl(training)
        self.get_key_quant(training_dict)

        if eccentricity > 3e-2:
            # phase, amp = self.construct()
            kwargs = {"omega_keys": self.omega_keys, "amp_keys": self.amp_keys}
            self.wave.add_eccentricity(
                self.construct,
                [(2, 2)],
                omega_keys=self.omega_keys,
                amp_keys=self.amp_keys,
            )

            self.time = self.wave.time
            self.h22 = self.wave[2, 2]

    @staticmethod
    def checkEccentricInp(eccentricity):
        if eccentricity < 0.0 or (eccentricity > 0.2 and eccentricity < 1.0):
            print("This version has only been calibrated up to eccentricity<0.2.")
        elif eccentricity >= 1.0:
            raise ValueError("Change eccentricity value (e<1)!")
        else:
            pass

    def checkParBoundaris(self):
        chi1 = np.sqrt(self.spin1x**2 + self.spin1y**2 + self.spin1z**2)
        chi2 = np.sqrt(self.spin2x**2 + self.spin2y**2 + self.spin2z**2)
        if abs(chi1) == 0.0 and abs(chi2) == 0.0:
            self.q = self.mass1 / self.mass2
            if self.q >= 1 and self.q <= 3:
                Cookware.checkEccentricInp(self.eccentricity)
            elif self.q > 3:
                print("This version has only been calibrated up to q<=3.")
                Cookware.checkEccentricInp(self.eccentricity)
            else:
                raise ValueError("Please correct your mass ratio, only for q>=1.")
        else:
            raise ValueError(
                "This version has only been calibrated to non-spinning binaries."
            )

    @staticmethod
    def interpol_key_quant(training_quant, training_keys, test_quant):
        """
        Interpolate key quantities.
        """
        forA = float(interp1D(training_quant[1], training_keys[0], test_quant[1]))
        A = float(
            interp1D(
                training_quant[1], abs(np.asarray(training_keys[0])), test_quant[1]
            )
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

    def get_key_quant(self, training_dict):

        q = self.mass1 / self.mass2
        eq, ee, ex, eomg, eamp = get_noncirc_params(training_dict)
        training_quant = [eq, ee, ex]

        test_quant = [q, self.eccentricity, self.x]

        A_omega, B_omega, freq_omega, phi_omega = Cookware.interpol_key_quant(
            training_quant, eomg, test_quant
        )
        A_amp, B_amp, freq_amp, phi_amp = Cookware.interpol_key_quant(
            training_quant, eamp, test_quant
        )
        if self.varphi:
            phi_omega = self.varphi[1]
            phi_amp = self.varphi[0]
        self.omega_keys = [A_omega, B_omega, freq_omega, phi_omega]
        self.amp_keys = [A_amp, B_amp, freq_amp, phi_amp]

    @staticmethod
    def construct(wave, mode, omega_keys, amp_keys):
        timenew, amp_rec, phase_rec = eccentric_from_circular(
            omega_keys, amp_keys, wave
        )

        late_time, late_amp, late_phase = near_merger(wave)
        amp_construct = np.concatenate((amp_rec, late_amp))
        phase_construct = np.concatenate((phase_rec, late_phase))

        h_model = amp_construct * np.exp(phase_construct * 1j)

        amp_model = np.abs(h_model)
        phase_model = np.unwrap(np.angle(h_model))
        # time_construct = np.concatenate((timenew, late_time))  # * timescale

        amp_model = smooth_joint(wave.time, amp_model, wave.metadata.total_mass)
        phase_model = smooth_joint(wave.time, phase_model, wave.metadata.total_mass)

        h_model = amp_model * np.exp(phase_model * 1j)
        print(phase_model, amp_model)
        return phase_model, amp_model


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

    dt = units.tSI_to_tM(wave.metadata.delta_t, wave.metadata.total_mass)
    ntime = np.arange(time[0], -29.0, dt)
    if max(abs(omega)) == 0:
        new_time = ntime
        amp_rec = np.zeros(len(new_time))
        phase_rec = np.zeros(len(new_time))
    else:
        interp_omega = make_interp_spline(time, omega)
        interp_amp = make_interp_spline(time, amp)

        new_time = ntime

        omega_circ = -interp_omega(new_time)
        amp_circ = interp_amp(new_time)
        shift_omega = -interp_omega(-1500)
        shift_amp = interp_amp(-1500)

        x_omega = np.nan_to_num(
            (-omega_circ) ** phase_pwr - (-shift_omega) ** phase_pwr
        )
        x_amp = np.nan_to_num((amp_circ) ** amp_pwr - shift_amp**amp_pwr)

        fit_ex_omega = f_sin(
            x_omega, par_omega[0], par_omega[1], par_omega[2], par_omega[3]
        )
        fit_ex_amp = f_sin(x_amp, par_amp[0], par_amp[1], par_amp[2], par_amp[3])
        omega_rec = fit_ex_omega * 2 * omega_circ + omega_circ

        mask = np.isfinite(omega_rec)
        omega_rec = np.interp(new_time, new_time[mask], omega_rec[mask])

        phase_rec = integrate.cumulative_trapezoid(
            units.fM_to_fSI(omega_rec, wave.metadata.total_mass), new_time, initial=0
        )
        amp_rec = fit_ex_amp * 2 * amp_circ + amp_circ

        amp_mask = np.isfinite(amp_rec)
        amp_rec = make_interp_spline(new_time[amp_mask], amp_rec[amp_mask])(new_time)
        amp_rec = units.mM_to_mSI(
            amp_rec, wave.metadata.total_mass, wave.metadata.distance
        )

        phase_mask = np.isfinite(phase_rec)
        phase_rec = make_interp_spline(new_time[phase_mask], phase_rec[phase_mask])(
            new_time
        )
        new_time = units.tM_to_tSI(new_time, wave.metadata.total_mass)
    return new_time, amp_rec, phase_rec
