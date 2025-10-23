from pyrex.main import glassware
import numpy as np

sims = [
    "SXS:BBH:0180",
    "SXS:BBH:1355",
    "SXS:BBH:1357",
    "SXS:BBH:1362",
    "SXS:BBH:1363",
    "SXS:BBH:0184",
    "SXS:BBH:1364",
    "SXS:BBH:1368",
    "SXS:BBH:1369",
    "SXS:BBH:0183",
    "SXS:BBH:1373",
    "SXS:BBH:1374",
]
q = [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0]
e_ref = [0.0, 0.053, 0.097, 0.189, 0.192, 0.0, 0.044, 0.097, 0.185, 0.0, 0.093, 0.18]

training = glassware(
    q=q,
    chi=0,
    names=sims,
    e_ref=e_ref,
    outfname="/home/amin/Projects/School/Masters/25_26-Thesis/pyrex/data/pyrexdata.pkl",
)
