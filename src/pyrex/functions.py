import numpy as np
from numpy.typing import NDArray


def f_sin(
    xdata: NDArray[np.floating], amplitude: float, B: float, freq: float, phase: float
) -> NDArray[np.floating]:
    """The fitting function of the eccentricity caused modulations to the amplitude or omega.

    Args:
        xdata (NDArray[np.floating]): The circularized amplitude or omega to which the power law has been applied.
        amplitude (float): The amplitude parameter A of the fitting function.
        B (float): The factor in the exponent of the fitting function.
        freq (float): The frequency fitting parameter.
        phase (float): The free fitting parameter phi.

    Returns:
        NDArray[np.floating]: Eccentricity caused modulations to the amplitude or omega.
    """
    sin_func = (
        amplitude * np.exp(B * xdata) * np.sin(xdata * freq / (2 * np.pi) + phase)
    )
    return sin_func
