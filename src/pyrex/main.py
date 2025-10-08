import numpy as np
from pyrex.tools import fitting_eccentric_function, find_x
from pyrex.basics import write_pkl, checkIfDuplicates
from scipy.signal import savgol_filter
from scipy.interpolate import make_interp_spline
from qcextender.dimensionlesswaveform import DimensionlessWaveform


class Glassware(object):
    """
    A class to measure the eccentricity of a given NR waveform.
    """

    def __init__(self, q, chi, names, e_ref, outfname=None):
        """
        Initiates Glassware class for non-spinning, low eccentricity, and mass ratio<=3 binaries.

        Parameters
        ----------
        q           : []
                      Mass ratio.
        chi         : {float}
                  Dimensionless spin parameters.
        names       : []
                  Simulation names.
        e_ref       : []
                  e at the reference frequency ('e_comm').

        Returns
        ------
        times     : []
                              Array of the sample time.
        amp22	  : []
                              Array of the amplitude of the l=2, m=2 mode.
        phase22   : []
                              Array of the phase of the l=e, m=e mode.
        h22       : []
                          Array of the l=2, m=2 strain.
        """

        if abs(chi) == 0.0:
            if all(i >= 1.0 for i in q) and all(i <= 3.0 for i in q):
                self.q = q
                self.chi = chi
                self.names = names
                self.e_ref = e_ref
            else:
                raise ValueError("Please correct your mass ratio, only for q<=3.")
        else:
            raise ValueError(
                "Please correct your spin, only for the non-spinning binaries, s1x=s1y=s1z=s2x=s2y=s2z=0."
            )
        self.components()
        self.compute_e_estimator()
        self.fit_model()
        self.compute_xquant()

        # write and store the data
        data_dict = self.__dict__
        if outfname:
            write_pkl(outfname, data_dict)

    def components(self):
        """
        Computes and align the amplitude, phase, strain of the l=2, m=2 mode of NR waveforms.

        Parameters
        ----------
        time_peak   : {float}
                    The maximum amplitude before alignment.
        """
        times, amps, phases, h22s, omegas = [], [], [], [], []
        for name in self.names:
            sim = DimensionlessWaveform.from_sim(name)
            times.append(sim.time)
            amps.append(sim.amp())
            phases.append(sim.phase())
            h22s.append(sim[2, 2])
            omegas.append(sim.omega())

        self.time = np.asarray(times)
        self.amp = np.asarray(amps)
        self.phase = np.asarray(phases)
        self.h22 = np.asarray(h22s)
        self.omega = np.asarray(omegas)

    def check_double_circ(self):
        circ_q = []
        circ_names = []
        circ_amp = []
        circ_phase = []
        circ_omega = []
        circ_time = []
        for i in range(len(self.names)):
            if self.e_ref[i] == 0:
                circ_q.append(self.q[i])
                circ_names.append(self.names[i])
                circ_amp.append(self.amp[i])
                circ_phase.append(self.phase[i])
                circ_omega.append(self.omega[i])
                circ_time.append(self.time[i])
        if checkIfDuplicates(circ_q):
            raise ValueError(
                "Please check duplicates of mass ratio and eccentricity in the provided circular waveforms."
            )
        else:
            for j in range(len(self.q)):
                if self.q[j] not in circ_q:
                    raise ValueError(
                        '"Simulation name {} has no circular waveform with the same mass ratio"'.format(
                            self.names[j]
                        )
                    )
                else:
                    pass
        return circ_names, circ_q, circ_time, circ_amp, circ_phase, circ_omega

    @staticmethod
    def get_eX(self, circ_q, circ_time, circ, component, new_time, filter_comp=2):

        eX = []
        for i in range(len(circ_q)):
            circs = make_interp_spline(circ_time[i], circ[i])
            for j in range(len(self.q)):
                ecc = make_interp_spline(self.time[j], component[j])
                if self.q[j] == circ_q[i]:
                    eX_filter = (ecc(new_time) - circs(new_time)) / (
                        2.0 * circs(new_time)
                    )
                    if self.e_ref[j] != 0:
                        eX_filter = savgol_filter(eX_filter, 501, filter_comp)
                    eX.append(eX_filter)
        return eX

    def compute_e_estimator(self):
        """
        Computes eccentricity from omega as a function in time (see Husa).

        Parameters
        ----------
        time_circular    : []
                         1 dimensional array to of time samples in circular eccentricity.
        omega_circular   : []
                         1 dimensional array to of omega in circular eccentricity.
        h22              : []
                         1 dimensional array to of h22 in circular eccentricity.

        """
        begin_tm = -1500.0  # Hardcoded, based on what exactly?
        end_tm = -29  # -31
        len_tm = 15221  # Why is this even specified?
        # dt=0.09664644309623327
        new_time = np.linspace(begin_tm, end_tm, len_tm)  # arange(begin_tm,end_tm,dt)

        circ_names, circ_q, circ_time, circ_amp, circ_phase, circ_omega = (
            self.check_double_circ()
        )

        eX_omega = Glassware.get_eX(
            self, circ_q, circ_time, circ_omega, self.omega, new_time
        )
        eX_amp = Glassware.get_eX(
            self, circ_q, circ_time, circ_amp, self.amp, new_time, filter_comp=3
        )
        self.eX_omega = eX_omega
        self.eX_amp = eX_amp
        self.new_time = new_time

    def fit_model(self):
        phase_params = np.zeros((len(self.names), 4))
        amp_params = np.zeros((len(self.names), 4))
        fit_phase = []
        fit_amp = []

        circ_names, circ_q, circ_time, circ_amp, circ_phase, circ_omega = (
            Glassware.check_double_circ(self)
        )

        for i in range(len(circ_omega)):
            interp_omega_c = make_interp_spline(circ_time[i], circ_omega[i])
            interp_amp_c = make_interp_spline(circ_time[i], circ_amp[i])
            for j in range(len(self.names)):
                if self.q[j] == circ_q[i]:
                    phase_params[j], fit_phaser = fitting_eccentric_function(
                        -59.0 / 24, self.eX_omega[j], interp_omega_c(self.new_time)
                    )
                    amp_params[j], fit_ampr = fitting_eccentric_function(
                        -83.0 / 24, self.eX_amp[j], interp_amp_c(self.new_time)
                    )

                    fit_phase.append(fit_phaser)
                    fit_amp.append(fit_ampr)

        self.A_omega = phase_params.T[0]
        self.B_omega = phase_params.T[1]
        self.freq_omega = phase_params.T[2]
        self.phi_omega = phase_params.T[3]
        self.fit_omega = fit_phase

        self.A_amp = amp_params.T[0]
        self.B_amp = amp_params.T[1]
        self.freq_amp = amp_params.T[2]
        self.phi_amp = amp_params.T[3]
        self.fit_amp = fit_amp

    def compute_xquant(self):
        xquant = []
        for i in range(len(self.names)):
            xquant.append(find_x(self.time[i], self.omega[i], self.new_time))
        self.x = xquant
