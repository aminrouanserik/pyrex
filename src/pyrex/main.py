import numpy as np
from pyrex.tools import fit_sin, calculate_x
from pyrex.basics import write_pkl
from scipy.signal import savgol_filter
from scipy.interpolate import make_interp_spline
from qcextender.dimensionlesswaveform import DimensionlessWaveform


def glassware(q, chi, names, e_ref, outfname=None):

    if abs(chi) != 0.0:
        raise ValueError(
            "Please correct your spin, only for the non-spinning binaries, s1x=s1y=s1z=s2x=s2y=s2z=0."
        )
    if not all(1.0 <= i <= 3.0 for i in q):
        raise ValueError("Please correct your mass ratio, only for q<=3.")

    waves = components(names)
    circ_waves = get_circ_waves(waves, e_ref)

    circ_lookup = {}

    for c in circ_waves:
        time = c.time
        mask = (time > (time[0] + 250)) & (time <= -29)

        t = time[mask]
        omega = c.omega()[mask]
        amp = c.amp()[mask]

        q_key = round(c.metadata.q, 0)
        circ_lookup[q_key] = (
            make_interp_spline(t, omega),
            make_interp_spline(t, amp),
        )

    # Need a common time grid for all waves for this code to work. Boundaries and length can be tinkered with
    begin_tm = -1500.0
    end_tm = -29
    len_tm = 15221
    new_time = np.linspace(begin_tm, end_tm, len_tm)

    e_amp, e_omega, new_time = compute_e_estimator(waves, e_ref, circ_lookup, new_time)

    results = fit_model(waves, circ_lookup, new_time, e_omega, e_amp)
    x = compute_xquant(waves, new_time)

    # write and store the data
    results.update({"q": q, "e_ref": e_ref, "x": x})
    if outfname:
        write_pkl(outfname, results)


def components(names):
    sims = []
    for name in names:
        sims.append(DimensionlessWaveform.from_sim(name))
    return sims


def compute_e_estimator(waves, eccentricities, circ_lookup, new_time):
    circ_lookup_omega = {q: pair[0] for q, pair in circ_lookup.items()}
    circ_lookup_amp = {q: pair[1] for q, pair in circ_lookup.items()}

    e_amp, e_omega = [], []
    for w, e in zip(waves, eccentricities):
        e_amp.append(get_e_amp(w, e, circ_lookup_amp, new_time))
        e_omega.append(get_e_omega(w, e, circ_lookup_omega, new_time))

    return e_amp, e_omega, new_time


def fit_model(
    waves,
    circ_lookup,
    new_time,
    e_omega,
    e_amp,
    omega_power=-59 / 24,
    amp_power=-83 / 24,
):
    omega_params, amp_params = [], []
    fit_omega, fit_amp = [], []

    for wave, omega, amp in zip(waves, e_omega, e_amp):

        # This needs to be fixed throughout, just take closest q instead or take from model instead.
        q = round(wave.metadata.q, 0)

        if q not in circ_lookup:
            raise ValueError(f"No circular reference for q={q}")

        interp_omega_c, interp_amp_c = circ_lookup[q]

        circ_omega_vals = interp_omega_c(new_time)
        circ_amp_vals = interp_amp_c(new_time)

        omega_paramsr, fit_omegar = fitting_eccentric_function(
            omega_power, omega, circ_omega_vals
        )
        amp_paramsr, fit_ampr = fitting_eccentric_function(
            amp_power, amp, circ_amp_vals
        )

        omega_params.append(omega_paramsr)
        amp_params.append(amp_paramsr)
        fit_omega.append(fit_omegar)
        fit_amp.append(fit_ampr)

    omega_params, amp_params = np.array(omega_params), np.array(amp_params)

    results = {
        "A_omega": omega_params[:, 0],
        "B_omega": omega_params[:, 1],
        "freq_omega": omega_params[:, 2],
        "phi_omega": omega_params[:, 3],
        "fit_omega": fit_omega,
        "A_amp": amp_params[:, 0],
        "B_amp": amp_params[:, 1],
        "freq_amp": amp_params[:, 2],
        "phi_amp": amp_params[:, 3],
        "fit_amp": fit_amp,
    }

    return results


def get_circ_waves(waves, eccentricities):
    circ_waves = []

    # Every value of q should have an example where eccentricity is 0
    for wave, eccentricity in zip(waves, eccentricities):
        if eccentricity == 0:
            circ_waves.append(wave)
    return circ_waves


def get_e_X(wave, eccentricity, circ_lookup, new_time, get_component, filter_comp=2):

    q = np.round(wave.metadata.q, 2)
    if q not in circ_lookup:
        raise ValueError(f"No circular reference for q={q}")

    mask = (wave.time > (wave.time[0] + 250)) & (wave.time <= -29)
    circ_interp = circ_lookup[q]
    ecc_interp = make_interp_spline(wave.time[mask], get_component(wave)[mask])

    circ_vals = circ_interp(new_time)
    ecc_vals = ecc_interp(new_time)

    e_X = (ecc_vals - circ_vals) / (2.0 * circ_vals)

    if eccentricity > 0:
        e_X = savgol_filter(e_X, 501, filter_comp)

    return e_X


def get_e_amp(wave, eccentricity, circ_lookup, new_time):
    return get_e_X(
        wave, eccentricity, circ_lookup, new_time, lambda w: w.amp(), filter_comp=3
    )


def get_e_omega(wave, eccentricity, circ_lookup, new_time):
    return get_e_X(
        wave,
        eccentricity,
        circ_lookup,
        new_time,
        lambda w: w.omega(),
        filter_comp=2,
    )


def fitting_eccentric_function(pwr, e_amp_phase, interpol_circ):
    x = (interpol_circ) ** pwr - (interpol_circ[0]) ** pwr
    y = e_amp_phase
    par, fsn = fit_sin(x, y)
    return par, fsn


def compute_xquant(waves, new_time):
    x = [calculate_x(wave.time, wave.omega(), new_time) for wave in waves]
    return x
