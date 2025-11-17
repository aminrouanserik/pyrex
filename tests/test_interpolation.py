import numpy as np

from pyrex.eccentric_extension import interpol_key_quant, interpolate_quantities


def test_interpolation_basic_run():
    e = np.array([0, 0.1, 0.2])
    q = np.array([1, 2, 3])
    vals = np.array([0, 1, 2])
    out = interpolate_quantities(list(e), list(q), vals, 0.05, 1.5)
    assert np.isfinite(out)


def test_interpol_key_quant_zero_e_limit():
    training_quant = [
        [1.0, 2.0],  # q
        [0.0, 0.1],  # e
        [0.0, 1.0],  # x
    ]

    training_keys = [
        np.array([0.0, 10.0]),
        np.array([0.0, 1.0]),
        np.array([10.0, 20.0]),
        np.array([0.0, np.pi / 2]),
    ]

    test_quant = [1.5, 0.0, 0.5]

    A, B, freq, phi = interpol_key_quant(training_quant, training_keys, test_quant)

    assert abs(A) < 1e-6
