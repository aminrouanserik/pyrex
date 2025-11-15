"""
Provides fitting functions used to model eccentricity-induced modulations
in transformed gravitational-wave quantities such as circularized amplitude
or orbital frequency.

Currently implemented:
    • f_sin — A sinusoidal modulation with an exponential envelope:
          f(x) = A * exp(B * x) * sin( (freq * x) / (2π) + phase )

All functions return NumPy arrays and operate element-wise on input data.
Type hints follow `numpy.typing.NDArray` conventions.
"""

import numpy as np
from numpy.typing import NDArray


def f_sin(
    xdata: NDArray[np.floating], amplitude: float, B: float, freq: float, phase: float
) -> NDArray[np.floating]:
    """
    Compute the sinusoidal modulation used to model eccentricity-induced
    oscillations in the amplitude or orbital frequency.

    The functional form is:
        f(x) = A * exp(B * x) * sin( (freq * x) / (2π) + phase )

    Args:
        xdata (NDArray[np.floating]): The transformed circular quantity (e.g. omega^(2/3)^p or
        amplitude^p) over which the modulation is evaluated.
        amplitude (float): The amplitude parameter A of the modulation.
        B (float): Exponential growth/decay factor applied to the envelope.
        freq (float): Oscillation frequency parameter of the sinusoid.
        phase (float): Phase offset phi applied to the sinusoid.

    Returns:
        NDArray[np.floating]: An array containing the eccentricity-driven modulation evaluated
        at each point in `xdata`.
    """
    sin_func = (
        amplitude * np.exp(B * xdata) * np.sin(xdata * freq / (2 * np.pi) + phase)
    )
    return sin_func
