import matplotlib.pyplot as plt
import numpy as np
from gw_eccentricity import measure_eccentricity
from qcextender.dimensionlesswaveform import DimensionlessWaveform

sims = [
    # "SXS:BBH:0180v2.0",
    "SXS:BBH:1355",
    "SXS:BBH:1357",
    "SXS:BBH:1362",
    "SXS:BBH:1363v2.0",
    # "SXS:BBH:0184v2.0",
    "SXS:BBH:1364",
    "SXS:BBH:1368",
    "SXS:BBH:1369",
    # "SXS:BBH:0183v2.0",
    "SXS:BBH:1373",
    "SXS:BBH:1374",
]

zero = DimensionlessWaveform.from_sim("SXS:BBH:0180v2.0")

fref_in = 0.0075
for simname in sims:
    sim = DimensionlessWaveform.from_sim(simname)
    return_dict = measure_eccentricity(
        fref_in=fref_in,
        method="ResidualAmplitude",
        dataDict={
            "t": sim.time,
            "hlm": {(2, 2): sim[2, 2]},
            "t_zeroecc": zero.time,
            "hlm_zeroecc": {(2, 2): zero[2, 2]},
        },
    )
    fref_out = return_dict["fref_out"]
    eccentricity = return_dict["eccentricity"]
    mean_anomaly = return_dict["mean_anomaly"]
    gwecc_object = return_dict["gwecc_object"]
    print(
        f"simulation = {simname:20s} eccentricity = {eccentricity:.6f}, mean anomaly = {mean_anomaly:.6f}"
    )
