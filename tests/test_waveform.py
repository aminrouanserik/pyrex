import numpy as np
import pytest

from pyrex.eccentric_extension import generate_eccentric_waveform


def test_eccentric_phase_differs_from_circular():
    wf0 = generate_eccentric_waveform(
        "IMRPhenomD",
        [(2, 2)],
        eccentricity=0.0,
        mass1=20,
        mass2=20,
        distance=10,
        delta_t=1 / 4096,
        f_lower=20,
    )
    wf05 = generate_eccentric_waveform(
        "IMRPhenomD",
        [(2, 2)],
        eccentricity=0.2,
        mass1=20,
        mass2=20,
        distance=10,
        delta_t=1 / 4096,
        f_lower=20,
    )
    assert not np.allclose(wf0.phase(), wf05.phase(), rtol=1e-2)


def test_circular_limit():
    wf1 = generate_eccentric_waveform(
        "IMRPhenomD",
        [(2, 2)],
        eccentricity=0.0,
        mass1=20.0,
        mass2=20.0,
        distance=10,
        delta_t=1 / 4096,
        f_lower=20,
    )
    wf2 = generate_eccentric_waveform(
        "IMRPhenomD",
        [(2, 2)],
        eccentricity=1e-6,
        mass1=20.0,
        mass2=20.0,
        distance=10,
        delta_t=1 / 4096,
        f_lower=20,
    )
    assert np.allclose(wf1[2, 2], wf2[2, 2], rtol=1e-3)


def test_waveform_structure():
    wf = generate_eccentric_waveform(
        "IMRPhenomD",
        [(2, 2)],
        mass1=20.0,
        mass2=20.0,
        eccentricity=0,
        distance=10,
        delta_t=1 / 4096,
        f_lower=20,
    )
    assert hasattr(wf, "time")
    assert hasattr(wf, "strain")
    assert len(wf.time) == len(wf[2, 2])


def test_invalid_approximant():
    with pytest.raises(ValueError):
        generate_eccentric_waveform(
            approximant="InvalidModel",
            mode=[(2, 2)],
            mass1=20.0,
            mass2=20.0,
            eccentricity=0.0,
            distance=10,
            delta_t=1 / 4096,
            f_lower=20,
        )


def test_invalid_eccentricity():
    import pytest

    with pytest.raises(ValueError):
        generate_eccentric_waveform(
            "IMRPhenomD",
            [(2, 2)],
            mass1=20,
            mass2=20,
            distance=10,
            eccentricity=1.5,  # invalid
            delta_t=1 / 4096,
            f_lower=20,
        )
